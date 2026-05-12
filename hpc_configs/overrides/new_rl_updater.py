"""
new_rl_updater.py  —  HPC-tuned drop-in replacement.

RLConfig defaults below are tuned for **H100 80GB (single GPU, bf16,
no quantization)** — i.e. the sweet-spot config we recommend on the
fattest realistic node.  See the per-field "← H100" annotations.

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

Algorithm itself is byte-for-byte identical to the project's version —
only the defaults moved.
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

    # Training settings ── H100 ───────────────────────────────────────────
    learning_rate: float = 2e-5  # ← H100 (was 1e-5): bigger LoRA tolerates +lr
    max_grad_norm: float = 1.0
    kl_beta: float = 0.0         # leave 0 until updater implements ref-model KL

    # Sequence control ── H100 ────────────────────────────────────────────
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

    def _completion_log_prob(
        self,
        prompt: str,
        completion: str,
    ) -> Optional[torch.Tensor]:
        """
        Compute log P(completion | prompt).

        Only completion tokens contribute to the loss.
        """
        encoded = self._encode_prompt_completion(prompt, completion)

        if encoded is None:
            return None

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        prompt_len = int(encoded["prompt_len"].item())

        if prompt_len >= input_ids.shape[1]:
            return None

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self.cfg.bf16 and torch.cuda.is_available(),
        ):
            outputs = self.engine.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits

        # Predict token t using logits at position t-1.
        shift_logits = logits[:, prompt_len - 1:-1, :]
        shift_labels = input_ids[:, prompt_len:]

        if shift_logits.shape[1] == 0 or shift_labels.shape[1] == 0:
            return None

        log_probs = F.log_softmax(shift_logits, dim=-1)

        token_log_probs = log_probs.gather(
            dim=2,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)

        return token_log_probs.sum()

    def update(
        self,
        prompts: List[str],
        completions: List[str],
        rewards: List[float],
    ) -> Dict:
        """
        Run one GRPO-style update step.
        """
        if not (len(prompts) == len(completions) == len(rewards)):
            raise ValueError("prompts, completions, and rewards must have same length.")

        if len(rewards) < 2:
            raise ValueError("GRPO update requires at least 2 samples for reward normalization.")

        self.engine.model.train()

        advantages = self._normalize_rewards(rewards).to(self._device())

        policy_losses = []
        valid_rewards = []
        valid_advantages = []

        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            seq_log_prob = self._completion_log_prob(prompt, completion)

            if seq_log_prob is None:
                print(f"Skipping sample {i}: no valid completion tokens.")
                continue

            advantage = advantages[i]

            # Policy gradient loss:
            # high advantage -> increase probability
            # low advantage  -> decrease probability
            policy_loss = -advantage * seq_log_prob

            policy_losses.append(policy_loss)
            valid_rewards.append(float(rewards[i]))
            valid_advantages.append(float(advantage.detach().cpu()))

        if not policy_losses:
            raise RuntimeError(
                "No valid samples for RL update. "
                "Likely prompt/completion truncation removed all completion tokens."
            )

        loss = torch.stack(policy_losses).mean()

        self.optimizer.zero_grad()
        loss.backward()

        trainable_params = [
            p for p in self.engine.model.parameters()
            if p.requires_grad
        ]

        torch.nn.utils.clip_grad_norm_(
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
            "policy_loss": round(float(loss.detach().cpu()), 6),
            "kl_div": 0.0,
            "total_loss": round(float(loss.detach().cpu()), 6),
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
