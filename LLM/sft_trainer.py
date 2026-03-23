"""
sft_trainer.py — SFT warm-start training module.

Only one job: fine-tune a base model on (constraint, netlist) pairs
using LoRA, then save the adapter.

Depends on:
    - llm_engine_minimal.py (LLMEngine, for model/tokenizer access)

Usage:
    from sft_trainer import SFTWarmStart, SFTConfig

    config = SFTConfig(data_path="./data/sft_pairs.jsonl")
    trainer = SFTWarmStart("Qwen/Qwen2.5-Coder-7B", config)
    adapter_path = trainer.train()
    trainer.merge_and_save("./checkpoints/sft-merged")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from peft import LoraConfig, TaskType, get_peft_model

logger = logging.getLogger(__name__)


# ── Config ───────────────────────────────────────────────────────────────

@dataclass
class SFTConfig:
    """All knobs for SFT training. Sensible defaults for Qwen2.5-Coder-7B."""

    # Data
    data_path: str = ""                        # JSONL: {"prompt": ..., "completion": ...}
    max_seq_length: int = 2048

    # LoRA
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_targets: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Training
    num_epochs: int = 3
    batch_size: int = 4
    grad_accum_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    lr_scheduler: str = "cosine"

    # Output
    output_dir: str = "./checkpoints/sft-lora"
    save_total_limit: int = 3

    # Hardware
    bf16: bool = True
    gradient_checkpointing: bool = True
    quantization: Optional[str] = None         # None | "4bit" | "8bit"

    @property
    def lora_config(self) -> LoraConfig:
        return LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=self.lora_targets,
        )


# ── Trainer ──────────────────────────────────────────────────────────────

class SFTWarmStart:
    """
    Runs LoRA SFT on (constraint JSON -> SPICE netlist) pairs.

    Public interface:
        trainer.train()             -> Path  (adapter location)
        trainer.merge_and_save(dir) -> Path  (merged full model)
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-Coder-7B",
        config: Optional[SFTConfig] = None,
    ):
        from llm_engine_minimal import LLMEngine

        self.cfg = config or SFTConfig()
        self._engine = LLMEngine(model_id, quantization=self.cfg.quantization)

        # Apply LoRA to the engine's model
        self._engine._model = get_peft_model(
            self._engine._model, self.cfg.lora_config
        )
        self._engine._is_peft = True
        self._engine._model.print_trainable_parameters()

    def train(self) -> Path:
        """
        Run SFT training. Returns path to saved LoRA adapter.

        Data format (JSONL, one per line):
            {"prompt": "{\"vin\":12,\"vout_target\":5}", "completion": "Vin VIN 0 DC 12\\nL1 ..."}
        """
        from datasets import load_dataset
        from trl import SFTTrainer, SFTConfig as TRLSFTConfig

        dataset = load_dataset(
            "json", data_files=self.cfg.data_path, split="train"
        )

        def fmt(example):
            return (
                f"### Constraint:\n{example['prompt']}\n\n"
                f"### SPICE Netlist:\n{example['completion']}"
            )

        args = TRLSFTConfig(
            output_dir=self.cfg.output_dir,
            num_train_epochs=self.cfg.num_epochs,
            per_device_train_batch_size=self.cfg.batch_size,
            gradient_accumulation_steps=self.cfg.grad_accum_steps,
            learning_rate=self.cfg.learning_rate,
            lr_scheduler_type=self.cfg.lr_scheduler,
            warmup_ratio=self.cfg.warmup_ratio,
            bf16=self.cfg.bf16,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=self.cfg.save_total_limit,
            max_seq_length=self.cfg.max_seq_length,
            gradient_checkpointing=self.cfg.gradient_checkpointing,
        )

        trainer = SFTTrainer(
            model=self._engine.model,
            args=args,
            train_dataset=dataset,
            formatting_func=fmt,
            processing_class=self._engine.tokenizer,
        )

        trainer.train()

        out = Path(self.cfg.output_dir) / "final"
        trainer.save_model(str(out))
        self._engine.tokenizer.save_pretrained(str(out))
        logger.info(f"SFT adapter saved: {out}")
        return out

    def merge_and_save(self, output_path: str) -> Path:
        """Merge LoRA weights into base model and save as a standalone checkpoint."""
        self._engine._model = self._engine._model.merge_and_unload()
        self._engine._is_peft = False
        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)
        self._engine._model.save_pretrained(str(out))
        self._engine._tokenizer.save_pretrained(str(out))
        logger.info(f"Merged model saved: {out}")
        return out
