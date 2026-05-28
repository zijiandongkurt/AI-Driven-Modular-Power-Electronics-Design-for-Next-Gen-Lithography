"""
new_rl_updater.py  —  HPC-tuned drop-in replacement (v2: KL + entropy).

RLConfig defaults below are tuned for **H100 80GB (single GPU, bf16,
no quantization)** — i.e. the sweet-spot config we recommend on the
fattest realistic node.  See the per-field "← H100" annotations.

What's new in v2 (post 20-step diagnostic):
  • Policy loss switched from sum-of-log-probs (v1) → per-token mean.
    Standard PPO/GRPO/TRL formulation. Side effects:
      - Length-invariant: a 1024-tok completion no longer drowns a 100-tok one.
      - Compatible scale with KL (also per-token mean), so kl_beta is
        interpretable as a fraction of policy-loss magnitude (v1 with
        sum-policy + per-token-KL would have made KL effectively zero).
      - LR scale matches TRL/HF norms; the new default (1e-5) is the
        right magnitude for "cautious LoRA fine-tuning on per-token loss".
  • KL penalty actually implemented (was hardcoded to 0 before).
    Uses PEFT's `disable_adapter()` to get reference log-probs from the
    base model with ZERO extra memory cost. Standard trick from TRL's
    PPOTrainer when no separate ref_model is given.
  • Optional entropy bonus to fight mode collapse (we saw 3/4 candidates
    produce identical netlists in the last 10 steps of the 20-step run).
  • n<2 graceful skip (kept from v1).
  • Metrics now expose kl_loss, entropy, grad_norm, all_same_reward
    so the user can diagnose without re-running.

Two-step downgrade for tighter cards (without editing this file):

  40GB A100  →  scale-down via SLURM env vars, see
                hpc_configs/slurm/train_single_40gb.slurm
                (sets GRPO_LORA_R, GRPO_LORA_ALPHA, GRPO_LR, GRPO_MAX_LENGTH)

  80GB A100  →  same H100 defaults work; just don't quantize

Memory budget @ H100 defaults (bf16, n=4 samples):
      base model weights ...... 28.0 GB
      LoRA (r=16, 7 targets) ..  0.4 GB
      AdamW state (fp32) ......  0.5 GB
      4 × seq=2048 activations . ~40   GB
      KV / misc ...............  ~6   GB
      ─────────────────────────────────
      total .................... ~75 GB out of 93 GB  →  ~18 GB margin

   (KL ref pass uses `disable_adapter()` + torch.no_grad → no extra grad
   memory, just one extra forward of the SAME tensors.)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class RLConfig:
    """
    Configuration for GRPO-style LoRA policy update.

    Defaults tuned for **H100 80GB single-GPU bf16** training run.
    See header for the memory budget breakdown.
    """

    # LoRA settings ── H100 ───────────────────────────────────────────────
    lora_r: int = 16             # ← H100 (was 8): twice the capacity
    lora_alpha: int = 32         # ← H100 (was 16): keep alpha = 2 × r
    lora_dropout: float = 0.05
    lora_targets: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # Training settings ── H100 (v2 anti-divergence tuning) ──────────────
    # NOTE: v2 switched policy loss from sum-of-log-probs (v1) to
    # per-token mean (TRL/PPO standard).  That changes the natural LR
    # scale by ~n_tok=200×.  The v2 default below targets the same
    # "effective per-parameter step size" the SFT path uses (2e-4 for
    # 5 epochs), shrunk 20× because we want RL to be CAUTIOUS:
    learning_rate: float = 1e-5   # ← v2 default. If mean_reward is flat
                                  #   after 30 steps, bump to 3e-5 or 5e-5.
                                  #   If you see KL exploding, drop to 5e-6.
    max_grad_norm: float = 1.0
    kl_beta: float = 0.05         # ← v2 (was 0): KL(current || base) penalty
                                  #   anchors the policy and prevents the
                                  #   reward-hacking mode-collapse we saw
                                  #   in steps 11-20 of the diag run.
                                  #   Scale: per-token mean KL is O(0.01-0.1),
                                  #   so kl_beta=0.05 → KL contributes
                                  #   ~5% of policy-loss magnitude.
    entropy_beta: float = 0.0     # ← v2 NEW: optional entropy bonus.
                                  #   Bump to 0.01 if you still see all-4
                                  #   candidates emitting identical text.

    # Sequence control ── H100 ───────────────────────────────────────────
    max_length: int = 2048       # ← H100 (was 1024): 80GB has the headroom
    max_prompt_length: int = 1536  # ← H100 (was 768):  fit full SYSTEM_PROMPT
    max_completion_length: int = 1024  # ← H100 (was 512): supports multi-phase

    # Runtime ─────────────────────────────────────────────────────────────
    bf16: bool = True
    save_every: int = 5          # ← H100 (was 10): denser checkpoints
    output_dir: str = "./checkpoints/grpo-lora"


class RLUpdater:
    """
    GRPO-style policy updater.

    It updates only LoRA parameters using externally provided rewards.
    Uses PEFT's `disable_adapter()` for KL reference (no extra memory).
    """

    def __init__(self, engine, config: Optional[RLConfig] = None):
        from peft import LoraConfig, TaskType, get_peft_model

        self.cfg = config or RLConfig()
        self.engine = engine
        self.step_id = 0

        # Apply LoRA if the model is not already a PEFT model.
        if not getattr(self.engine, "_is_peft", False):
            lora_config = LoraConfig(
                r=self.cfg.lora_r,
                lora_alpha=self.cfg.lora_alpha,
                lora_dropout=self.cfg.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=self.cfg.lora_targets,
            )

            self.engine._model = get_peft_model(self.engine._model, lora_config)
            self.engine._is_peft = True
            self.engine._model.print_trainable_parameters()

        trainable_params = [
            p for p in self.engine.model.parameters()
            if p.requires_grad
        ]

        if not trainable_params:
            raise RuntimeError("No trainable parameters found for RL update.")

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.cfg.learning_rate,
        )

        # Detect whether the underlying PEFT model supports adapter
        # disabling (it does for any PeftModel wrap, which we always do
        # above).  We use `disable_adapter()` to compute reference
        # log-probs from the base model — zero extra GPU memory.
        self._has_disable_adapter = hasattr(self.engine.model, "disable_adapter")
        if self.cfg.kl_beta > 0 and not self._has_disable_adapter:
            logger.warning(
                "RLConfig.kl_beta=%.4f but model does not expose "
                "disable_adapter(); KL penalty will be skipped.",
                self.cfg.kl_beta,
            )

    def _device(self):
        return self.engine.model.device

    def _tokenizer(self):
        return self.engine.tokenizer

    def _normalize_rewards(self, rewards: List[float]) -> torch.Tensor:
        """
        Convert raw rewards into normalized advantages.

        GRPO uses relative reward ranking within the batch.
        """
        r = torch.tensor(rewards, dtype=torch.float32)

        # Reward normalization: focus on relative performance within the batch.
        r_mean = r.mean()

        # Use population std for stable normalization.
        # clamp prevents division by zero when rewards have no variance.
        r_std = r.std(unbiased=False).clamp(min=1e-8)

        return (r - r_mean) / r_std

    def _prepare_text_pair(self, prompt: str, completion: str) -> tuple[str, str]:
        """
        Shorten prompt and completion at text level before tokenization.

        This avoids the common failure mode where a long prompt consumes
        the whole context window and leaves no completion tokens.
        """
        prompt = prompt.strip()
        completion = completion.strip()

        if not prompt:
            prompt = "Generate a valid LTspice netlist.\n\nNetlist:\n"

        if not completion:
            completion = ".end"

        return prompt, completion

    def _encode_prompt_completion(
        self,
        prompt: str,
        completion: str,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Tokenize prompt and completion separately, then concatenate.

        This guarantees that completion tokens are preserved.
        """
        tok = self._tokenizer()
        device = self._device()

        prompt, completion = self._prepare_text_pair(prompt, completion)

        prompt_enc = tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.max_prompt_length,
        )

        completion_enc = tok(
            completion,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.max_completion_length,
        )

        prompt_ids = prompt_enc["input_ids"]
        completion_ids = completion_enc["input_ids"]

        if completion_ids.shape[1] == 0:
            return None

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        if input_ids.shape[1] > self.cfg.max_length:
            # Keep full completion as much as possible.
            completion_len = min(completion_ids.shape[1], self.cfg.max_completion_length)
            available_prompt_len = max(self.cfg.max_length - completion_len, 1)

            prompt_ids = prompt_ids[:, -available_prompt_len:]
            completion_ids = completion_ids[:, :completion_len]
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "prompt_len": torch.tensor(prompt_ids.shape[1], device=device),
            "completion_len": torch.tensor(completion_ids.shape[1], device=device),
        }

    # ── Forward helpers ─────────────────────────────────────────────────

    def _forward_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        no_grad: bool = False,
    ) -> torch.Tensor:
        """Run one forward pass and return logits.

        If no_grad=True, wraps in torch.no_grad() (used for the ref pass).
        """
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx, torch.amp.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self.cfg.bf16 and torch.cuda.is_available(),
        ):
            outputs = self.engine.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        return outputs.logits

    def _per_token_stats(
        self,
        prompt: str,
        completion: str,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Compute everything we need for one (prompt, completion) pair:

        Returns:
            {
                "sum_log_prob":  scalar tensor with grad,  Σ log P_θ(yt | y<t, x)
                "ref_sum_log_prob": scalar tensor WITHOUT grad,
                                    Σ log P_ref(yt | y<t, x),  or None
                "entropy":       scalar tensor with grad,
                                 mean per-token H(P_θ) over completion tokens
                "n_tokens":      int — length of the completion span
            }

        ref_sum_log_prob and entropy are only computed when their betas
        are non-zero (saves compute).
        """
        encoded = self._encode_prompt_completion(prompt, completion)
        if encoded is None:
            return None

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        prompt_len = int(encoded["prompt_len"].item())

        if prompt_len >= input_ids.shape[1]:
            return None

        # ── Policy forward (with grad) ──────────────────────────────
        logits = self._forward_logits(input_ids, attention_mask, no_grad=False)

        # Predict token t using logits at position t-1.
        shift_logits = logits[:, prompt_len - 1:-1, :]
        shift_labels = input_ids[:, prompt_len:]

        if shift_logits.shape[1] == 0 or shift_labels.shape[1] == 0:
            return None

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=2,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)              # (1, T)

        out: Dict[str, torch.Tensor] = {
            "sum_log_prob": token_log_probs.sum(),
            "n_tokens": int(token_log_probs.shape[1]),
        }

        # ── Entropy (optional) ──────────────────────────────────────
        if self.cfg.entropy_beta > 0:
            probs = log_probs.exp()
            # H = -Σ p log p, summed over vocab, averaged over completion tokens
            entropy_per_tok = -(probs * log_probs).sum(dim=-1)   # (1, T)
            out["entropy"] = entropy_per_tok.mean()
        else:
            out["entropy"] = None  # type: ignore[assignment]

        # ── Reference forward (no grad, adapter disabled) ───────────
        if self.cfg.kl_beta > 0 and self._has_disable_adapter:
            with self.engine.model.disable_adapter():
                ref_logits = self._forward_logits(
                    input_ids, attention_mask, no_grad=True,
                )
            ref_shift_logits = ref_logits[:, prompt_len - 1:-1, :]
            ref_log_probs = F.log_softmax(ref_shift_logits, dim=-1)
            ref_token_log_probs = ref_log_probs.gather(
                dim=2,
                index=shift_labels.unsqueeze(-1),
            ).squeeze(-1)
            out["ref_sum_log_prob"] = ref_token_log_probs.sum().detach()
        else:
            out["ref_sum_log_prob"] = None  # type: ignore[assignment]

        return out

    def update(
        self,
        prompts: List[str],
        completions: List[str],
        rewards: List[float],
    ) -> Dict:
        """
        Run one GRPO-style update step.

        Total loss = mean( -A_i · log π(y_i|x_i) )
                     + kl_beta  · mean( log π - log π_ref )
                     - entropy_beta · mean( H[π] )
        """
        if not (len(prompts) == len(completions) == len(rewards)):
            raise ValueError("prompts, completions, and rewards must have same length.")

        if len(rewards) < 2:
            # Skip the update gracefully instead of crashing the whole
            # training loop.  GRPO's group-relative advantage = (r-mean)/std
            # is meaningless with n=1 (std=0), but losing a single batch
            # to bad luck (e.g. simulator failed 3/4 candidates) shouldn't
            # take down a 5-step run.  Caller sees skipped=True and can
            # decide whether to retry, double n, etc.
            self.step_id += 1
            print(f"WARN: GRPO step {self.step_id} skipped — only "
                  f"{len(rewards)} valid sample(s), need >= 2.")
            return {
                "step": self.step_id,
                "num_input_samples": len(rewards),
                "num_valid_samples": 0,
                "skipped": True,
                "skip_reason": f"only {len(rewards)} valid sample(s)",
                "raw_rewards": list(map(float, rewards)),
                "mean_reward": float(sum(rewards) / max(len(rewards), 1)),
                "policy_loss": None,
                "kl_loss": None,
                "entropy_loss": None,
                "advantages": [],
                "total_loss": None,
                "kl_div": 0.0,
                "max_length": self.cfg.max_length,
                "max_prompt_length": self.cfg.max_prompt_length,
                "max_completion_length": self.cfg.max_completion_length,
            }

        # Mode-collapse detector: if EVERY reward in the batch is
        # identical, advantages are all zero and the gradient is zero.
        # We still take an "update" so the optimizer keeps state, but we
        # log it so the user can see why nothing's happening.
        all_same_reward = max(rewards) - min(rewards) < 1e-9

        self.engine.model.train()

        advantages = self._normalize_rewards(rewards).to(self._device())

        policy_losses: list[torch.Tensor] = []
        kl_terms: list[torch.Tensor] = []
        entropy_terms: list[torch.Tensor] = []
        valid_rewards: list[float] = []
        valid_advantages: list[float] = []

        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            stats = self._per_token_stats(prompt, completion)

            if stats is None:
                print(f"Skipping sample {i}: no valid completion tokens.")
                continue

            seq_log_prob: torch.Tensor = stats["sum_log_prob"]
            n_tok: int = stats["n_tokens"]

            advantage = advantages[i]

            # Policy gradient loss:
            # high advantage -> increase probability
            # low advantage  -> decrease probability
            #
            # We use per-token mean (standard PPO/GRPO/TRL formulation)
            # for THREE reasons:
            #   1. Length invariance: a 1024-tok completion no longer
            #      drowns out a 100-tok one in batch mean.
            #   2. Compatible scale with KL (also per-token mean), so
            #      kl_beta is interpretable as a fraction of policy
            #      loss magnitude.  v1's sum-of-log-probs vs per-token
            #      KL would have made KL effectively zero.
            #   3. The LR scale matches TRL/HF standards, so future
            #      maintainers can compare apples to apples.
            policy_loss = -advantage * (seq_log_prob / max(n_tok, 1))
            policy_losses.append(policy_loss)

            # KL: per-token mean of (log π - log π_ref).
            # Per-sample KL is then approximated by the diff of sums /
            # n_tok, which gives the same gradient direction as full
            # KL(π||π_ref) and is the standard PPO/GRPO approximation.
            ref_log_prob = stats.get("ref_sum_log_prob")
            if ref_log_prob is not None:
                kl_per_tok = (seq_log_prob - ref_log_prob) / max(n_tok, 1)
                kl_terms.append(kl_per_tok)

            # Entropy bonus (we MAXIMIZE entropy → SUBTRACT from loss).
            ent = stats.get("entropy")
            if ent is not None:
                entropy_terms.append(ent)

            valid_rewards.append(float(rewards[i]))
            valid_advantages.append(float(advantage.detach().cpu()))

        if not policy_losses:
            raise RuntimeError(
                "No valid samples for RL update. "
                "Likely prompt/completion truncation removed all completion tokens."
            )

        policy_loss_mean = torch.stack(policy_losses).mean()

        if kl_terms:
            kl_mean = torch.stack(kl_terms).mean()
        else:
            kl_mean = torch.tensor(0.0, device=self._device())

        if entropy_terms:
            entropy_mean = torch.stack(entropy_terms).mean()
        else:
            entropy_mean = torch.tensor(0.0, device=self._device())

        loss = (
            policy_loss_mean
            + self.cfg.kl_beta * kl_mean
            - self.cfg.entropy_beta * entropy_mean
        )

        self.optimizer.zero_grad()
        loss.backward()

        trainable_params = [
            p for p in self.engine.model.parameters()
            if p.requires_grad
        ]

        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_params,
            self.cfg.max_grad_norm,
        )

        self.optimizer.step()

        self.engine.model.eval()
        self.step_id += 1

        mean_reward = sum(valid_rewards) / len(valid_rewards)

        metrics = {
            "step": self.step_id,
            "num_input_samples": len(rewards),
            "num_valid_samples": len(valid_rewards),
            "mean_reward": round(mean_reward, 6),
            "raw_rewards": valid_rewards,
            "advantages": valid_advantages,
            "policy_loss": round(float(policy_loss_mean.detach().cpu()), 6),
            "kl_loss":     round(float(kl_mean.detach().cpu()),         6),
            "entropy":     round(float(entropy_mean.detach().cpu()),    6),
            "kl_div":      round(float(kl_mean.detach().cpu()),         6),
            "grad_norm":   round(float(grad_norm.detach().cpu()),       6),
            "total_loss":  round(float(loss.detach().cpu()),            6),
            "all_same_reward": all_same_reward,
            "kl_beta":      self.cfg.kl_beta,
            "entropy_beta": self.cfg.entropy_beta,
            "learning_rate": self.cfg.learning_rate,
            "max_length": self.cfg.max_length,
            "max_prompt_length": self.cfg.max_prompt_length,
            "max_completion_length": self.cfg.max_completion_length,
        }

        if self.cfg.save_every > 0 and self.step_id % self.cfg.save_every == 0:
            self.save(f"{self.cfg.output_dir}/step-{self.step_id}")

        return metrics

    def save(self, path: Optional[str] = None) -> Path:
        """
        Save current LoRA adapter.
        """
        save_path = Path(path or f"{self.cfg.output_dir}/final")
        save_path.mkdir(parents=True, exist_ok=True)

        self.engine.model.save_pretrained(str(save_path))
        self.engine.tokenizer.save_pretrained(str(save_path))

        print(f"Saved RL LoRA adapter to: {save_path}")

        return save_path

    @property
    def step(self) -> int:
        return self.step_id
