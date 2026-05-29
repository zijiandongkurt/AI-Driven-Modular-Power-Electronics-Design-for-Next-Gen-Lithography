"""
sft_loop.py — HPC-friendly driver for the SFT trainer.

Loads configuration from sft_config.json at repo root.

Usage:
    python sft_loop.py
    python sft_loop.py --config sft_config.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.llm_topology_generation.llm_api import TopologyLLM
from pipeline.SFT.sft_trainer import SFTTrainer, SFTConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="sft_config.json")
    args = parser.parse_args()

    config_path = Path(args.config)
    assert config_path.exists(), f"sft_config.json not found at {config_path.resolve()}"
    with config_path.open("r") as f:
        config = json.load(f)

    model_id   = config.get("model_id",      "Qwen/Qwen3-14B")
    quant      = config.get("quantization",  None)
    train_path = config.get("train_path",    "pipeline/data/sft/sft_train.jsonl")
    val_path   = config.get("val_path",      "pipeline/data/sft/sft_val.jsonl")
    output_dir = config.get("output_dir",    "checkpoints/sft/sft_001")

    cfg = SFTConfig(
        learning_rate=config.get("learning_rate", 2e-4),
        batch_size=config.get("batch_size",        4),
        epochs=config.get("epochs",                5),
        lora_r=config.get("lora_r",                16),
        lora_alpha=config.get("lora_alpha",        32),
        max_length=config.get("max_length",        2048),
        max_prompt_length=int(config.get("max_length", 2048) * 0.75),
        max_completion_length=int(config.get("max_length", 2048) * 0.50),
        output_dir=output_dir,
    )

    print("════════════════════════════════════════════════════════════════")
    print(" SFT runner")
    print("════════════════════════════════════════════════════════════════")
    print(f"  model_id      : {model_id}")
    print(f"  quantization  : {quant or 'bf16 (full)'}")
    print(f"  train_path    : {train_path}")
    print(f"  val_path      : {val_path}")
    print(f"  output_dir    : {output_dir}")
    print(f"  epochs        : {cfg.epochs}")
    print(f"  batch_size    : {cfg.batch_size}")
    print(f"  learning_rate : {cfg.learning_rate}")
    print(f"  lora_r/alpha  : {cfg.lora_r}/{cfg.lora_alpha}")
    print(f"  max_length    : {cfg.max_length}")
    print()

    print("[sft_loop] Loading TopologyLLM...")
    llm = TopologyLLM(model_id=model_id, quantization=quant)

    trainer = SFTTrainer(llm.engine, cfg)
    history = trainer.train(
        train_path=train_path,
        val_path=val_path if Path(val_path).exists() else None,
    )

    final = trainer.save(f"{output_dir}/final")

    hist_path = Path(output_dir) / "history.json"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    with hist_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print()
    print("════════════════════════════════════════════════════════════════")
    print(f"[sft_loop] final adapter : {final}")
    print(f"[sft_loop] history       : {hist_path}")
    print("════════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()