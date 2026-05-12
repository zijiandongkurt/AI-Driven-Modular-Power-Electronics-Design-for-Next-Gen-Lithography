#!/bin/bash
# ============================================================================
# Scenario A — Single 80 GB GPU (A100 80GB on Snellius, or H100 80GB HBM3)
# ============================================================================
# Use this when:
#   - You can get a single 80GB card
#   - You want the SIMPLEST setup (no model parallelism, no quantization)
#   - Qwen3-14B bf16 fits with room for activations
#
# Expected VRAM usage with our RLConfig override (max_length=1024, n=4):
#   model weights (bf16) .... 28 GB
#   LoRA + optimizer state .. ~0.3 GB
#   4 × seq=1024 activations . ~24 GB
#   misc ................... ~3 GB
#   ─────────────────────────────
#   total ................... ~55 GB out of 80 GB  →  ~25 GB margin
#
# ============================================================================
# SLURM resource request
# ============================================================================
#SBATCH --job-name=grpo_single80
#SBATCH --partition=gpu_a100                 # Snellius A100 partition
##SBATCH --partition=gpu_h100                # uncomment to use H100 80GB instead
#SBATCH --constraint=a100_80gb               # ← force the 80GB SKU (drop on gpu_h100)
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18                   # Snellius recommends 18 cores per A100
#SBATCH --mem=120G                           # 120 GB host RAM (for HF model staging)
#SBATCH --time=02:00:00                      # 2h — enough for ~30 GRPO iterations

#SBATCH --account=scur2545                   # ← replace with your account
#SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=your.email@example.com  # ← uncomment + fill in to get email
#SBATCH --output=logs/grpo_single80_%j.out
#SBATCH --error=logs/grpo_single80_%j.err

# ============================================================================
# Setup
# ============================================================================
mkdir -p logs

# ⚠️  These module loads MUST match the ones env.sh used to build the venv.
# If you built the venv with `module load 2024`, change `2023` below to `2024`.
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1

# ============================================================================
# Project paths
# ============================================================================
LOCAL_FOLDER=$HOME/AI-Driven-Modular-Power-Electronics-Design-for-Next-Gen-Lithography
# Point this at wherever you cloned the repo; the rest of the script is relative.

cd "$LOCAL_FOLDER" || { echo "Repo not found at $LOCAL_FOLDER"; exit 1; }
source .venv/bin/activate || { echo ".venv not found — did you run env.sh?"; exit 2; }

# Verify GPU + Python
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
python --version

# ============================================================================
# Environment knobs read by our overridden rl_demo.py
# ============================================================================
# HuggingFace cache. /projects/2/managed_datasets/hf_cache_dir is the Snellius
# shared cache that already contains Qwen3-14B. If it's not readable to you,
# point this at a /scratch path you `huggingface-cli download`ed into yourself.
export HF_HOME=/projects/2/managed_datasets/hf_cache_dir
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Quantization: leave EMPTY for full bf16 on 80GB (best quality, what we want here).
export GRPO_QUANTIZATION=""

# Model: leave default unless you want to test with the smaller 7B
export GRPO_MODEL_ID="Qwen/Qwen3-14B"

# ============================================================================
# Run
# ============================================================================
PYTHONPATH="$LOCAL_FOLDER" python -u "$LOCAL_FOLDER/pipeline/reinforcement_algorithm/rl_demo.py"

echo "DONE — exit code $?"
