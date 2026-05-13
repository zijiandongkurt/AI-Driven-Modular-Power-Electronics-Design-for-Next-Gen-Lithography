#!/bin/bash
# =====================================================================
# run_sft_smoke.sh — interactive SFT smoke test on HPC (NO sbatch)
# =====================================================================
# Usage (from inside a GPU node you already have):
#
#     # Default full run (~5-30 min depending on GPU + dataset size)
#     bash scripts/run_sft_smoke.sh
#
#     # Quick smoke (~2-3 min) — 2 epochs, batch=2, seq=1024
#     SFT_EPOCHS=2 SFT_BATCH_SIZE=2 SFT_MAX_LENGTH=1024 \
#         bash scripts/run_sft_smoke.sh
#
#     # Force 4-bit quantization (use this if you only got a 40GB card)
#     GRPO_QUANTIZATION=4bit bash scripts/run_sft_smoke.sh
#
# Output:
#     checkpoints/sft-lora/epoch-{N}/    per-epoch LoRA checkpoints
#     checkpoints/sft-lora/final/        final adapter
#     checkpoints/sft-lora/history.json  train/val loss curve
# =====================================================================

set -e
cd "$(dirname "$0")/.."          # cd to repo root

# ── Activate venv (only complain if it's actually missing) ───────────
if [ ! -f .venv/bin/activate ]; then
  echo "ERROR: .venv not found at $(pwd)/.venv — did you run env.sh?" >&2
  exit 1
fi
source .venv/bin/activate

# ── Python / HF env ──────────────────────────────────────────────────
export PYTHONPATH="$(pwd)"

# IMPORTANT: do NOT set HF_HOME to the shared cache dir.  HF Hub reads
# its auth `token` from $HF_HOME/token, and Snellius's shared cache has
# a token file owned by another user we can't read (PermissionError).
# Leave HF_HOME at its default (~/.cache/huggingface) and only point the
# model cache vars at the shared dir.
export HF_HUB_CACHE=${HF_HUB_CACHE:-/projects/2/managed_datasets/hf_cache_dir}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_HUB_DISABLE_IMPLICIT_TOKEN=${HF_HUB_DISABLE_IMPLICIT_TOKEN:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
# Defensive: if a previous shell had HF_HOME pointing at the shared dir,
# unset it.  Default location is ~/.cache/huggingface.
if [ "${HF_HOME:-}" = "/projects/2/managed_datasets/hf_cache_dir" ]; then
    echo "  → unsetting HF_HOME (it was pointing at the shared cache)"
    unset HF_HOME
fi

# ── Auto-detect GPU & decide quantization ────────────────────────────
GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
if [ -z "$GRPO_QUANTIZATION" ]; then
  if [ "$GPU_MEM_MB" -lt 50000 ] && [ "$GPU_MEM_MB" -gt 0 ]; then
    export GRPO_QUANTIZATION="4bit"
    echo "  → 40GB-class GPU detected (${GPU_MEM_MB}MiB), enabling 4-bit"
  else
    export GRPO_QUANTIZATION=""
    echo "  → 80GB-class GPU detected (${GPU_MEM_MB}MiB), bf16 full precision"
  fi
fi

# ── Sanity diagnostics ───────────────────────────────────────────────
echo
echo "─── nvidia-smi ─────────────────────────────────────────────"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

echo
echo "─── dependency check ──────────────────────────────────────"
python - <<'PY'
import importlib, sys
need = ["torch", "transformers", "peft", "accelerate"]
opt  = ["bitsandbytes", "flash_attn"]
ok = True
for pkg in need:
    try:
        m = importlib.import_module(pkg)
        print(f"  ✅ {pkg:<15} {getattr(m, '__version__', '?')}")
    except ImportError as e:
        print(f"  ❌ {pkg:<15} MISSING — {e}")
        ok = False
for pkg in opt:
    try:
        m = importlib.import_module(pkg)
        print(f"  ✅ {pkg:<15} {getattr(m, '__version__', '?')}  (optional)")
    except ImportError:
        print(f"  ⚠️  {pkg:<15} not installed  (optional)")
if not ok:
    sys.exit(2)
PY

echo
echo "─── SFT knobs ──────────────────────────────────────────────"
echo "  GRPO_MODEL_ID     = ${GRPO_MODEL_ID:-Qwen/Qwen3-14B}"
echo "  GRPO_QUANTIZATION = ${GRPO_QUANTIZATION:-<bf16>}"
echo "  SFT_EPOCHS        = ${SFT_EPOCHS:-5 (default)}"
echo "  SFT_BATCH_SIZE    = ${SFT_BATCH_SIZE:-4 (default)}"
echo "  SFT_MAX_LENGTH    = ${SFT_MAX_LENGTH:-2048 (default)}"
echo "  SFT_TRAIN_PATH    = ${SFT_TRAIN_PATH:-data/sft/sft_train.jsonl (default)}"
echo

echo "─── start SFT training ────────────────────────────────────"
python scripts/run_sft.py

echo
echo "─── history.json ──────────────────────────────────────────"
if [ -f checkpoints/sft-lora/history.json ]; then
  cat checkpoints/sft-lora/history.json
fi

echo
echo "DONE — exit code 0"
