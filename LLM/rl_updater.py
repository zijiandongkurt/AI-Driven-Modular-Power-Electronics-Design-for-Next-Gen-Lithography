"""
rl_updater.py — RL policy update module (GRPO).

Only one job: receive fitness scores from external modules,
compute the policy gradient, update LoRA weights.

This module does NOT run validation, simulation, or reward computation.
It only consumes their results.

Depends on:
    - llm_engine_minimal.py (LLMEngine, for model/tokenizer access)

Architecture:
    ┌─────────────┐     generate()      ┌──────────────┐
    │ LLMEngine   │ ──────────────────► │  Orchestrator │
    └─────────────┘                     │  (external)   │
          ▲                             │              │
          │  update_policy()            │  validate()  │
          │  (this module)              │  simulate()  │
          │                             │  reward()    │
    ┌─────────────┐     fitness_scores  └──────┬───────┘
    │ RLUpdater   │ ◄──────────────────────────┘
    └─────────────┘

Usage:
    from rl_updater import RLUpdater, RLConfig

    engine = LLMEngine("./checkpoints/sft-merged")
    updater = RLUpdater(engine, RLConfig())

    for step in range(100):
        # External orchestrator handles generation + eval
        prompts = [constraint.to_prompt() for constraint in batch]
        completions = [engine.generate(c).netlist for c in batch]
        scores = [your_pipeline(c) for c in completions]  # float list

        metrics = updater.update(prompts, completions, scores)
        if metrics["mean_reward"] >= 0.92:
            break

    updater.save("./checkpoints/grpo-lora/final")
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ── Config ───────────────────────────────────────────────────────────────

@dataclass
class RLConfig:
    """All knobs for GRPO policy updates."""

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_targets: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    # GRPO
    kl_beta: float = 0.1               # KL penalty weight
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0          # Gradient clipping

    # Output
    output_dir: str = "./checkpoints/grpo-lora"
    save_every: int = 10                # Save checkpoint every N updates

    # Hardware
    bf16: bool = True


# ── Updater ──────────────────────────────────────────────────────────────

class RLUpdater:
    """
    Receives external reward signals and updates LLM policy weights via GRPO.

    Public interface (what the orchestrator calls):
        updater.update(prompts, completions, rewards) -> dict   (metrics)
        updater.save(path)                                       (checkpoint)

    This class does NOT generate text or compute rewards.
    It only performs the weight update given external scores.
    """

    def __init__(self, engine, config: Optional[RLConfig] = None):
        """
        Args:
            engine: An LLMEngine instance (from llm_engine_minimal.py).
                    Its .model and .tokenizer are used directly.
            config: GRPO hyperparameters.
        """
        from peft import LoraConfig, TaskType, get_peft_model

        self.cfg = config or RLConfig()
        self._engine = engine
        self._step = 0

        # Apply RL-stage LoRA if model is not already a PEFT model
        if not engine._is_peft:
            lora_cfg = LoraConfig(
                r=self.cfg.lora_r,
                lora_alpha=self.cfg.lora_alpha,
                lora_dropout=self.cfg.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=self.cfg.lora_targets,
            )
            engine._model = get_peft_model(engine._model, lora_cfg)
            engine._is_peft = True
            engine._model.print_trainable_parameters()

        # Store reference policy log-probs (frozen snapshot for KL penalty)
        self._ref_model = self._make_ref_copy()

        # Optimizer — only trainable (LoRA) parameters
        trainable = [p for p in engine.model.parameters() if p.requires_grad]
        self._optimizer = torch.optim.AdamW(trainable, lr=self.cfg.learning_rate)

        logger.info(
            f"RLUpdater initialized: {sum(p.numel() for p in trainable):,} "
            f"trainable params, lr={self.cfg.learning_rate}"
        )

    # ── Core update ──────────────────────────────────────────────────

    def update(
        self,
        prompts: list[str],
        completions: list[str],
        rewards: list[float],
    ) -> dict:
        """
        Perform one GRPO policy gradient step.

        Args:
            prompts:     List of prompt strings (constraint text).
            completions: List of generated netlist strings (from engine.generate).
            rewards:     List of fitness scores (from external reward module).
                         Length must match completions.

        Returns:
            Metrics dict with keys:
                step, mean_reward, std_reward, policy_loss, kl_div

        GRPO algorithm:
            1. Normalize rewards within the batch (relative ranking).
            2. Compute log P(completion | prompt) under current policy.
            3. Compute log P(completion | prompt) under reference policy.
            4. Policy loss = -mean(normalized_reward * log_prob).
            5. KL penalty = beta * mean(log_prob - ref_log_prob).
            6. Total loss = policy_loss + KL penalty.
            7. Backprop and update LoRA weights.
        """
        assert len(prompts) == len(completions) == len(rewards)

        device = self._engine.model.device
        tok = self._engine.tokenizer

        # Step 1: Normalize rewards (GRPO's key insight — no critic needed)
        r = torch.tensor(rewards, dtype=torch.float32)
        r_mean, r_std = r.mean(), r.std().clamp(min=1e-8)
        normalized = (r - r_mean) / r_std

        # Step 2-3: Compute log-probs under current and reference policies
        total_policy_loss = torch.tensor(0.0, device=device)
        total_kl = torch.tensor(0.0, device=device)
        n = len(prompts)

        self._engine.model.train()

        for i in range(n):
            # Tokenize prompt + completion together
            full_text = prompts[i] + completions[i]
            enc = tok(full_text, return_tensors="pt", truncation=True, max_length=2048)
            input_ids = enc["input_ids"].to(device)
            attn_mask = enc["attention_mask"].to(device)

            # Find where completion starts
            prompt_enc = tok(prompts[i], return_tensors="pt")
            prompt_len = prompt_enc["input_ids"].shape[1]
            if prompt_len >= input_ids.shape[1]:
                continue  # Degenerate case: no completion tokens

            # Current policy log-probs
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.cfg.bf16):
                logits = self._engine.model(input_ids, attention_mask=attn_mask).logits

            # Shift: predict token t from position t-1
            shift_logits = logits[:, prompt_len - 1:-1, :]
            shift_labels = input_ids[:, prompt_len:]
            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
            seq_log_prob = token_log_probs.sum()

            # Reference policy log-probs (no grad)
            with torch.no_grad():
                ref_logits = self._ref_model(input_ids, attention_mask=attn_mask).logits
            ref_shift = ref_logits[:, prompt_len - 1:-1, :]
            ref_log_probs = F.log_softmax(ref_shift, dim=-1)
            ref_token_lp = ref_log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
            ref_seq_lp = ref_token_lp.sum()

            # Accumulate losses
            advantage = normalized[i].to(device)
            total_policy_loss += -advantage * seq_log_prob
            total_kl += (seq_log_prob - ref_seq_lp)

        # Step 6: Combine losses
        total_policy_loss /= n
        total_kl /= n
        loss = total_policy_loss + self.cfg.kl_beta * total_kl

        # Step 7: Backprop and update
        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self._engine.model.parameters() if p.requires_grad],
            self.cfg.max_grad_norm,
        )
        self._optimizer.step()

        self._engine.model.eval()
        self._step += 1

        # Auto-save checkpoint
        if self.cfg.save_every > 0 and self._step % self.cfg.save_every == 0:
            self.save(f"{self.cfg.output_dir}/step-{self._step}")

        metrics = {
            "step": self._step,
            "mean_reward": round(r_mean.item(), 4),
            "std_reward": round(r_std.item(), 4),
            "policy_loss": round(total_policy_loss.item(), 4),
            "kl_div": round(total_kl.item(), 4),
            "total_loss": round(loss.item(), 4),
        }
        logger.info(
            f"Step {self._step}: reward={metrics['mean_reward']:.3f}±{metrics['std_reward']:.3f} "
            f"loss={metrics['total_loss']:.4f} kl={metrics['kl_div']:.4f}"
        )
        return metrics

    # ── Save / Load ──────────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> Path:
        """Save current LoRA adapter weights."""
        out = Path(path or f"{self.cfg.output_dir}/final")
        out.mkdir(parents=True, exist_ok=True)
        self._engine.model.save_pretrained(str(out))
        self._engine.tokenizer.save_pretrained(str(out))
        logger.info(f"RL adapter saved: {out}")
        return out

    @property
    def step(self) -> int:
        return self._step

    # ── Internals ────────────────────────────────────────────────────

    def _make_ref_copy(self):
        """
        Create a frozen copy of the current model for KL reference.

        The reference policy is the model state at the START of RL training
        (i.e., the SFT checkpoint). KL divergence between the current policy
        and this reference prevents the model from drifting too far.
        """
        import copy
        ref = copy.deepcopy(self._engine.model)
        for p in ref.parameters():
            p.requires_grad = False
        ref.eval()
        return ref
