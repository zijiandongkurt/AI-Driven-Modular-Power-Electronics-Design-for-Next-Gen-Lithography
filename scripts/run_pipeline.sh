#!/bin/bash
# =====================================================================
# run_pipeline.sh — orchestrate the full SFT → GRPO chain on HPC
# =====================================================================
# Usage (from inside an interactive GPU session, NO sbatch):
#
#     bash scripts/run_pipeline.sh sft           # re-train SFT only
#     bash scripts/run_pipeline.sh smoke         # GRPO single-step smoke
#     bash scripts/run_pipeline.sh full          # GRPO full loop (needs LTspice)
#     bash scripts/run_pipeline.sh all           # sft → smoke → full
#
# Defaults assume an H100 80 GB GPU with bf16.  Override via env vars
# (see run_sft_smoke.sh / grpo_smoke.py / grpo_full.py headers).
# =====================================================================

set -e
cd "$(dirname "$0")/.."          # repo root

STAGE="${1:-smoke}"

# ── Activate venv ────────────────────────────────────────────────────
if [ ! -f .venv/bin/activate ]; then
  echo "ERROR: .venv missing. Run env.sh first." >&2
  exit 1
fi
source .venv/bin/activate

# ── Cache + auth env (NEVER set HF_HOME to shared dir) ───────────────
unset HF_HOME
export PYTHONPATH="$(pwd)"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/projects/2/managed_datasets/hf_cache_dir}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_IMPLICIT_TOKEN="${HF_HUB_DISABLE_IMPLICIT_TOKEN:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# ── GPU detect ───────────────────────────────────────────────────────
GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
if [ "$GPU_MEM_MB" -lt 50000 ] && [ "$GPU_MEM_MB" -gt 0 ]; then
  export GRPO_QUANTIZATION="${GRPO_QUANTIZATION:-4bit}"
  echo "[pipeline] 40GB-class GPU → 4bit quantization"
else
  export GRPO_QUANTIZATION="${GRPO_QUANTIZATION:-}"
  echo "[pipeline] 80GB-class GPU (${GPU_MEM_MB} MiB) → bf16"
fi

# ── Dispatch ─────────────────────────────────────────────────────────
run_sft() {
  echo ""; echo "================================================================="
  echo " STAGE: SFT (retrain on 647 samples, ~10 min on H100)"
  echo "================================================================="
  # Use the OOM-safe defaults already validated (bs=2, seq=1024, 5 epoch)
  SFT_EPOCHS="${SFT_EPOCHS:-5}" \
  SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-2}" \
  SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-1024}" \
    python scripts/run_sft.py
  echo "[pipeline] SFT done. checkpoint at checkpoints/sft-lora/epoch-3/"
}

run_smoke() {
  echo ""; echo "================================================================="
  echo " STAGE: GRPO smoke (~30 s, no simulator)"
  echo "================================================================="
  python scripts/grpo_smoke.py
}

run_full() {
  echo ""; echo "================================================================="
  echo " STAGE: GRPO full loop (needs LTspice container)"
  echo "================================================================="
  # Verify Atakan's container hooks exist
  if [ ! -f "$HOME/run_ltspice_snellius.sh" ]; then
    echo "WARN: ~/run_ltspice_snellius.sh not found."
    echo "      Full-loop training requires Atakan's containerized LTspice."
    echo "      Skipping. Re-run with the container script in place."
    return 1
  fi
  python scripts/grpo_full.py
}

case "$STAGE" in
  sft)    run_sft   ;;
  smoke)  run_smoke ;;
  full)   run_full  ;;
  all)    run_sft && run_smoke && run_full ;;
  *)      echo "Unknown stage: $STAGE" >&2
          echo "Usage: bash scripts/run_pipeline.sh [sft|smoke|full|all]" >&2
          exit 2 ;;
esac

echo ""
echo "[pipeline] STAGE '$STAGE' complete."
