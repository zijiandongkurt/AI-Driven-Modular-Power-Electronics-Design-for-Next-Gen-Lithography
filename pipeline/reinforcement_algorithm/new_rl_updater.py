"""
new_rl_updater.py

A lightweight GRPO-style LoRA updater for the reinforcement_algorithm module.

Responsibilities:
- Receive prompts, completions, rewards, and optional grouped advantages
- Compute log probabilities of completions
- Update LoRA trainable parameters
- Save LoRA adapter checkpoints

This module does NOT:
- Generate netlists
- Validate netlists
- Run LTspice
- Compute reward scores
- Compute grouped GRPO advantages

Grouped GRPO note:
- GRPOTrainer should compute advantages within each prompt group.
- This updater only applies the policy update using those advantages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict

import torch
import torch.nn.functional as F


@dataclass
class RLConfig:
    """
    Configuration for GRPO-style LoRA policy update.
    """

    # LoRA settings
    lora_r: int = 4
    lora_alpha: int = 8
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

    # Training settings
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0

    # Placeholder for future KL/reference-model support.
    # Currently this is not used because no reference model is loaded.
    kl_beta: float = 0.0

    # Sequence control
    max_length: int = 512
    max_prompt_length: int = 128
    max_completion_length: int = 256

    # Runtime
    bf16: bool = True
    save_every: int = 10
    output_dir: str = "./checkpoints/grpo-lora"


class RLUpdater:
    """
    GRPO-style policy updater.

    This class updates only LoRA parameters.
    Grouped advantage calculation is handled by GRPOTrainer.
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
        Fallback normalization.

        Used only when GRPOTrainer does not provide external advantages.
        For true grouped GRPO, advantages should be passed in explicitly.
        """
        r = torch.tensor(rewards, dtype=torch.float32)

        r_mean = r.mean()
        r_std = r.std(unbiased=False).clamp(min=1e-8)

        return (r - r_mean) / r_std

    def _prepare_text_pair(self, prompt: str, completion: str) -> tuple[str, str]:
        """
        Clean empty prompt/completion edge cases.
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

        This helps preserve completion tokens when the prompt is long.
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
            # Keep completion tokens as much as possible.
            completion_len = min(
                completion_ids.shape[1],
                self.cfg.max_completion_length,
            )
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
        Compute mean log P(completion | prompt).

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

        # Mean is more stable than sum for variable completion length.
        return token_log_probs.mean()

    def update(
        self,
        prompts: List[str],
        completions: List[str],
        rewards: List[float],
        advantages: Optional[List[float]] = None,
    ) -> Dict:
        """
        Run one LoRA policy update.

        If advantages are provided, they are assumed to be already normalized
        within each GRPO prompt group.
        """
        if not (len(prompts) == len(completions) == len(rewards)):
            raise ValueError("prompts, completions, and rewards must have same length.")

        if advantages is not None and len(advantages) != len(rewards):
            raise ValueError("advantages must have same length as rewards.")

        if not rewards:
            raise ValueError("No samples provided to RLUpdater.update().")

        self.engine.model.train()

        # NEW: use grouped advantages from GRPOTrainer when provided.
        if advantages is None:
            if len(rewards) < 2:
                raise ValueError(
                    "At least 2 samples are required when advantages are not provided."
                )

            advantages_tensor = self._normalize_rewards(rewards).to(self._device())
            advantage_source = "normalized_rewards"
        else:
            advantages_tensor = torch.tensor(
                advantages,
                dtype=torch.float32,
                device=self._device(),
            )
            advantage_source = "external_grouped_advantages"

        policy_losses = []
        valid_rewards = []
        valid_advantages = []
        valid_log_probs = []
        valid_indices = []

        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            seq_log_prob = self._completion_log_prob(prompt, completion)

            if seq_log_prob is None:
                print(f"Skipping sample {i}: no valid completion tokens.")
                continue

            advantage = advantages_tensor[i]

            # NEW: skip unstable numerical values.
            if not torch.isfinite(advantage) or not torch.isfinite(seq_log_prob):
                print(f"Skipping sample {i}: non-finite advantage or log_prob.")
                continue

            # Policy gradient loss:
            # positive advantage -> increase probability
            # negative advantage -> decrease probability
            policy_loss = -advantage * seq_log_prob

            policy_losses.append(policy_loss)
            valid_rewards.append(float(rewards[i]))
            valid_advantages.append(float(advantage.detach().cpu()))
            valid_log_probs.append(float(seq_log_prob.detach().cpu()))
            valid_indices.append(i)

        if not policy_losses:
            raise RuntimeError(
                "No valid samples for RL update. "
                "Likely prompt/completion truncation removed all completion tokens."
            )

        loss = torch.stack(policy_losses).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        trainable_params = [
            p for p in self.engine.model.parameters()
            if p.requires_grad
        ]

        # NEW: keep grad norm for debugging.
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_params,
            self.cfg.max_grad_norm,
        )

        self.optimizer.step()

        # Clear gradients after step to reduce VRAM pressure in long RL loops.
        self.optimizer.zero_grad(set_to_none=True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.engine.model.eval()
        self.step_id += 1

        mean_reward = sum(valid_rewards) / len(valid_rewards)

        metrics = {
            "step": self.step_id,
            "num_input_samples": len(rewards),
            "num_valid_samples": len(valid_rewards),
            "valid_indices": valid_indices,
            "mean_reward": round(mean_reward, 6),
            "raw_rewards": valid_rewards,
            "advantages": valid_advantages,
            "advantage_source": advantage_source,
            "policy_loss": round(float(loss.detach().cpu()), 6),
            "kl_div": 0.0,
            "kl_beta": self.cfg.kl_beta,
            "completion_log_probs": valid_log_probs,
            "grad_norm": round(float(grad_norm.detach().cpu()), 6),
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