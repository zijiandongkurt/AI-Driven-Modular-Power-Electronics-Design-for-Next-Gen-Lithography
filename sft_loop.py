"""
run_sft.py — HPC-friendly driver for the SFT trainer.

Mirrors hpc_configs/overrides/rl_demo.py's structure: reads SLURM-side
environment variables, sets up cache paths, instantiates TopologyLLM +
SFTTrainer, runs `.train()` on the canonical training JSONL.

Usage:
    # Local Windows / WSL:
    python scripts/run_sft.py

    # On HPC (via SLURM):
    sbatch hpc_configs/slurm/train_sft.slurm

Knobs (env vars, all optional; CLI flags override env vars):

    Cache / runtime
        HF_HUB_CACHE / HF_HOME              → standard HF cache location
        GRPO_MODEL_ID                       → "Qwen/Qwen3-14B" (default)
        GRPO_QUANTIZATION                   → "" or "4bit"

    SFT hyperparams
        SFT_TRAIN_PATH                      → data/sft/sft_train.jsonl
        SFT_VAL_PATH                        → data/sft/sft_val.jsonl
        SFT_EPOCHS                          → 5
        SFT_BATCH_SIZE                      → 4
        SFT_LR                              → 2e-4
        SFT_LORA_R                          → 16
        SFT_LORA_ALPHA                      → 32
        SFT_MAX_LENGTH                      → 2048
        SFT_OUTPUT_DIR                      → ./checkpoints/sft-lora

Output:
    <SFT_OUTPUT_DIR>/epoch-1/  ... /epoch-N/   per-epoch checkpoints
    <SFT_OUTPUT_DIR>/final/                    final LoRA adapter
    <SFT_OUTPUT_DIR>/history.json              per-epoch train/val loss
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ── 1. Wire HF cache env vars (auto-detect flat vs nested layout) ──────────
# IMPORTANT: We deliberately do NOT set HF_HOME to a shared cache dir.
# HF_HOME controls where huggingface_hub reads its auth `token` file.
# Snellius's /projects/2/managed_datasets/hf_cache_dir/ contains a `token`
# owned by another user that we cannot read; HF Hub only catches
# FileNotFoundError there, so it crashes with PermissionError.
# Leave HF_HOME at its default (~/.cache/huggingface) and only point the
# model cache vars at the shared dir.
_cache = (os.environ.get("HF_HUB_CACHE")
          or os.environ.get("HF_CACHE_DIR")
          or os.environ.get("HF_HOME"))
if _cache:
    _flat   = glob.glob(os.path.join(_cache, "models--*"))
    _nested = glob.glob(os.path.join(_cache, "hub", "models--*"))
    _hub    = _cache if (_flat and not _nested) else os.path.join(_cache, "hub")
    os.environ["HF_HUB_CACHE"]       = _hub
    os.environ["TRANSFORMERS_CACHE"] = _hub
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

# ── 2. Imports (after env vars are in place) ───────────────────────────────
from pipeline.llm_topology_generation.llm_api import TopologyLLM
from pipeline.SFT.sft_trainer import SFTTrainer, SFTConfig


def _env_str(name, default):
    return os.environ.get(name, default)


def _env_int(name, default):
    v = os.environ.get(name)
    return int(v) if v is not None else default


def _env_float(name, default):
    v = os.environ.get(name)
    return float(v) if v is not None else default


def main():
    # ── Resolve every knob ─────────────────────────────────────────────
    model_id     = _env_str("GRPO_MODEL_ID",     "Qwen/Qwen3-14B")
    quant        = _env_str("GRPO_QUANTIZATION", "") or None

    train_path   = _env_str("SFT_TRAIN_PATH",    str(REPO_ROOT / "data" / "sft" / "sft_train.jsonl"))
    val_path     = _env_str("SFT_VAL_PATH",      str(REPO_ROOT / "data" / "sft" / "sft_val.jsonl"))

    epochs       = _env_int  ("SFT_EPOCHS",      5)
    batch_size   = _env_int  ("SFT_BATCH_SIZE",  4)
    lr           = _env_float("SFT_LR",          2e-4)
    lora_r       = _env_int  ("SFT_LORA_R",      16)
    lora_alpha   = _env_int  ("SFT_LORA_ALPHA",  32)
    max_length   = _env_int  ("SFT_MAX_LENGTH",  2048)
    output_dir   = _env_str  ("SFT_OUTPUT_DIR",  str(REPO_ROOT / "checkpoints" / "sft-lora"))

    print("════════════════════════════════════════════════════════════════")
    print(" SFT runner")
    print("════════════════════════════════════════════════════════════════")
    print(f"  model_id      : {model_id}")
    print(f"  quantization  : {quant or 'bf16 (full)'}")
    print(f"  train_path    : {train_path}")
    print(f"  val_path      : {val_path}")
    print(f"  epochs        : {epochs}")
    print(f"  batch_size    : {batch_size}")
    print(f"  learning_rate : {lr}")
    print(f"  lora_r/alpha  : {lora_r}/{lora_alpha}")
    print(f"  max_length    : {max_length}")
    print(f"  output_dir    : {output_dir}")
    print()

    # ── Build the SFT config ───────────────────────────────────────────
    cfg = SFTConfig(
        learning_rate=lr,
        batch_size=batch_size,
        epochs=epochs,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        max_length=max_length,
        # max_prompt / max_completion follow max_length proportionally
        max_prompt_length=int(max_length * 0.75),
        max_completion_length=int(max_length * 0.50),
        output_dir=output_dir,
    )

    # ── Load model + run ───────────────────────────────────────────────
    print("[run_sft] loading TopologyLLM ...")
    llm = TopologyLLM(model_id=model_id, quantization=quant)

    trainer = SFTTrainer(llm.engine, cfg)
    history = trainer.train(
        train_path=train_path,
        val_path=val_path if Path(val_path).exists() else None,
    )

    final = trainer.save(f"{output_dir}/final")

    # ── History dump ───────────────────────────────────────────────────
    hist_path = Path(output_dir) / "history.json"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    with hist_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print()
    print("════════════════════════════════════════════════════════════════")
    print(f"[run_sft] final adapter: {final}")
    print(f"[run_sft] history:       {hist_path}")
    print("════════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
