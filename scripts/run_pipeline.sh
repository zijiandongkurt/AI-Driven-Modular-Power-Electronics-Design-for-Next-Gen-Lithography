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

  # Resolve the LTspice launcher:  user env var first,
  # then in-repo defaults (docker → snellius → workstation),
  # then Atakan's legacy ~/run_ltspice_snellius.sh as last resort.
  CANDIDATES=(
      "${LTSPICE_LAUNCHER:-}"
      "$(pwd)/containers/ltspice/run_ltspice_docker.sh"
      "$(pwd)/containers/ltspice/run_ltspice_snellius.sh"
      "$(pwd)/containers/ltspice/run_ltspice.sh"
      "$HOME/run_ltspice_snellius.sh"
  )

  LAUNCHER=""
  for c in "${CANDIDATES[@]}"; do
      if [ -n "$c" ] && [ -f "$c" ]; then
          LAUNCHER="$c"
          break
      fi
  done

  if [ -z "$LAUNCHER" ]; then
      echo "WARN: no LTspice launcher script found." >&2
      echo "      Checked:" >&2
      for c in "${CANDIDATES[@]}"; do [ -n "$c" ] && echo "        $c" >&2; done
      echo "      Build the container first:  bash containers/ltspice/build.sh" >&2
      echo "      Then export LTSPICE_LAUNCHER to one of containers/ltspice/run_ltspice*.sh" >&2
      return 1
  fi

  # Sanity-check the .sif exists too (whichever launcher we picked will need one)
  SIF="${LTSPICE_SIF:-$HOME/container_custom.sif}"
  if [ ! -f "$SIF" ]; then
      echo "WARN: container image not found at $SIF" >&2
      echo "      Pull/build it:  bash containers/ltspice/build.sh" >&2
      return 1
  fi

  # And the bind-mount source dir
  FILES_DIR="${LTSPICE_FILES_DIR:-$HOME/ltspice-files}"
  if [ ! -d "$FILES_DIR" ]; then
      echo "  → mkdir -p $FILES_DIR"
      mkdir -p "$FILES_DIR"
  fi

  echo "  [container] launcher = $LAUNCHER"
  echo "  [container] sif      = $SIF"
  echo "  [container] files    = $FILES_DIR"

  # Re-export so Python runner sees the resolved paths
  export LTSPICE_LAUNCHER="$LAUNCHER"
  export LTSPICE_SIF="$SIF"
  export LTSPICE_FILES_DIR="$FILES_DIR"

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
