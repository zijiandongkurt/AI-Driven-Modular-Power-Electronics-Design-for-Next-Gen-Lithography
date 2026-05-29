#!/bin/bash
# =====================================================================
# benchmark_sft.sh — run the base-vs-SFT head-to-head on Snellius
# =====================================================================
# Usage (inside an interactive H100 session, after `bash env.sh` once):
#
#     bash scripts/benchmark_sft.sh                 # default: 20 prompts × 4 cands
#     BENCH_N_PROMPTS=10 bash scripts/benchmark_sft.sh    # quick smoke
#     BENCH_N_PROMPTS=50 BENCH_K_CANDS=4 bash scripts/benchmark_sft.sh
#
# Wall-time estimate on H100 (with the LTspice container working):
#     ~ (n_prompts × k_cands × 15s gen + 10s sim) × 2 models
#       20 × 4 × 25s × 2  ≈  ~70 min
#       10 × 4 × 25s × 2  ≈  ~35 min   ← recommended for the meeting demo
#
# Output:
#     meeting/benchmark_results.csv     one row per candidate
#     meeting/benchmark_summary.md      table for the slide doc
# =====================================================================

set -e
cd "$(dirname "$0")/.."

# venv + env-vars (same single source of truth as run_pipeline.sh)
if [ ! -f .venv/bin/activate ]; then
  echo "ERROR: .venv missing. Run env.sh first." >&2
  exit 1
fi
source .venv/bin/activate

unset HF_HOME
export PYTHONPATH="$(pwd)"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/projects/2/managed_datasets/hf_cache_dir}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_IMPLICIT_TOKEN="${HF_HUB_DISABLE_IMPLICIT_TOKEN:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# LTspice container resolution (same priority list as run_pipeline.sh full)
CANDIDATES=(
    "${LTSPICE_LAUNCHER:-}"
    "$(pwd)/containers/ltspice/run_ltspice_docker.sh"
    "$(pwd)/containers/ltspice/run_ltspice_snellius.sh"
    "$(pwd)/containers/ltspice/run_ltspice.sh"
    "$HOME/run_ltspice_snellius.sh"
)
for c in "${CANDIDATES[@]}"; do
    if [ -n "$c" ] && [ -f "$c" ]; then
        export LTSPICE_LAUNCHER="$c"
        break
    fi
done

export LTSPICE_SIF="${LTSPICE_SIF:-$HOME/container_custom.sif}"
export LTSPICE_FILES_DIR="${LTSPICE_FILES_DIR:-$HOME/ltspice-files}"
mkdir -p "$LTSPICE_FILES_DIR"

if [ ! -f "$LTSPICE_LAUNCHER" ] || [ ! -f "$LTSPICE_SIF" ]; then
    echo "WARN: LTspice container not ready — simulator will fail" >&2
    echo "      LAUNCHER=$LTSPICE_LAUNCHER" >&2
    echo "      SIF=$LTSPICE_SIF" >&2
    echo "      Fix:  bash containers/ltspice/build.sh" >&2
fi

# Defaults
export SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH:-./checkpoints/sft-lora/epoch-3}"
export BENCH_N_PROMPTS="${BENCH_N_PROMPTS:-20}"
export BENCH_K_CANDS="${BENCH_K_CANDS:-4}"
export BENCH_OUT_DIR="${BENCH_OUT_DIR:-meeting}"
export BENCH_TEMP="${BENCH_TEMP:-0.7}"

echo "[benchmark_sft] launcher = ${LTSPICE_LAUNCHER:-<none>}"
echo "[benchmark_sft] sif      = $LTSPICE_SIF"
echo "[benchmark_sft] adapter  = $SFT_ADAPTER_PATH"
echo "[benchmark_sft] n×k      = $BENCH_N_PROMPTS × $BENCH_K_CANDS"
echo ""

python scripts/benchmark_sft.py

echo ""
echo "[benchmark_sft] DONE"
echo "[benchmark_sft] CSV     : $BENCH_OUT_DIR/benchmark_results.csv"
echo "[benchmark_sft] Summary : $BENCH_OUT_DIR/benchmark_summary.md"
