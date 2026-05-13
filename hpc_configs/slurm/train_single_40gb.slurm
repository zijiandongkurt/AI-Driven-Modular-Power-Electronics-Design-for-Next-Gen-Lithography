#!/bin/bash
# ============================================================================
# Scenario C — Single 40 GB GPU (most common case on Snellius gpu_a100)
# ============================================================================
# Use this when:
#   - You only got 1 A100 40GB allocated
#   - You still want to run GRPO on Qwen3-14B
#
# How it fits (with 4-bit NF4 quantization):
#   model weights (NF4) ........  7 GB
#   LoRA + optimizer state .....  0.3 GB
#   4 × seq=1024 activations ... 24 GB
#   misc / KV / workspace ......  3 GB
#   ─────────────────────────────────
#   total ...................... ~34 GB out of 40 GB  →  ~6 GB margin
#
# Quantization (4-bit NF4) is REQUIRED here.  Without it, Qwen3-14B bf16
# alone is 28 GB — that leaves only 12 GB for activations, which can't
# fit a 4-sample × 1024-token batch.
#
# Trade-off vs Scenario A (80GB) / B (4×40GB):
#   + much easier to schedule
#   + 4× cheaper in GPU-hours
#   - 4-bit base model: marginally lower generation quality
#     (LoRA adapters are still fp32 — what we're training is unchanged)
#
# ============================================================================
# SLURM resource request
# ============================================================================
#SBATCH --job-name=grpo_single40
#SBATCH --partition=gpu_a100                 # Snellius default = 40GB
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=01:00:00                      # 1h — ~20 GRPO iterations w/ quant

#SBATCH --account=scur2545                   # ← replace with your account
#SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=your.email@example.com  # ← uncomment + fill in
#SBATCH --output=logs/grpo_single40_%j.out
#SBATCH --error=logs/grpo_single40_%j.err

# ============================================================================
# Setup
# ============================================================================
mkdir -p logs

# ⚠️  Module versions MUST match what env.sh used to build the venv.
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1

# ============================================================================
# Project paths
# ============================================================================
LOCAL_FOLDER=$HOME/AI-Driven-Modular-Power-Electronics-Design-for-Next-Gen-Lithography

cd "$LOCAL_FOLDER" || { echo "Repo not found at $LOCAL_FOLDER"; exit 1; }
source .venv/bin/activate || { echo ".venv not found — did you run env.sh?"; exit 2; }

# Sanity-check the GPU we got
echo "── nvidia-smi inside the job ────────────────────────────────────"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$GPU_MEM_MB" -lt 50000 ]; then
  echo "✅ Confirmed 40GB-class GPU ($GPU_MEM_MB MiB) — 4-bit quantization REQUIRED."
else
  echo "⚠️  SLURM gave you an ${GPU_MEM_MB}MiB GPU — you can disable quantization."
fi

python --version

# ============================================================================
# Environment knobs (read by overrides/rl_demo.py — see README §7)
# ============================================================================
# IMPORTANT: do NOT set HF_HOME to the shared cache dir.  HF Hub reads
# its auth `token` from $HF_HOME/token, and the shared cache contains a
# token file owned by another user we can't read (PermissionError).
export HF_HUB_CACHE=/projects/2/managed_datasets/hf_cache_dir
export TRANSFORMERS_CACHE=/projects/2/managed_datasets/hf_cache_dir
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1

# ── 4-bit quantization is MANDATORY on a 40GB card with Qwen3-14B ──────────
export GRPO_QUANTIZATION="4bit"

# Optional: keep tokenizer parallelism off so 18 CPU cores don't oversubscribe
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=18

# ── Scale H100 defaults DOWN for 40GB ──────────────────────────────────────
# The overrides default to H100 80GB-tuned values:
#   lora_r=16, lora_alpha=32, lr=2e-5, max_length=2048
# 4-bit-quantized 14B on 40GB can't afford those.  Override via env vars:
export GRPO_LORA_R=8
export GRPO_LORA_ALPHA=16
export GRPO_LR=1e-5
export GRPO_MAX_LENGTH=1024     # max_prompt and max_completion auto-shrink

# OOM ladder (uncomment progressively if still OOM):
#   step 1: tighter seq
##export GRPO_MAX_LENGTH=768
#   step 2: tighter seq again
##export GRPO_MAX_LENGTH=512
#   step 3: also cap samples
##export GRPO_N_SAMPLES=2

export GRPO_MODEL_ID="Qwen/Qwen3-14B"

# ============================================================================
# Run
# ============================================================================
PYTHONPATH="$LOCAL_FOLDER" python -u "$LOCAL_FOLDER/pipeline/reinforcement_algorithm/rl_demo.py"

# ============================================================================
# Post-run: capture peak GPU mem so we can decide whether to grow seq next time
# ============================================================================
echo "── nvidia-smi after run ─────────────────────────────────────────"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

echo "DONE — exit code $?"
