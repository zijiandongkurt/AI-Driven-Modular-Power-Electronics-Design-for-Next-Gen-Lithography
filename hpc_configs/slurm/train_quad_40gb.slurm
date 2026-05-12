#!/bin/bash
# ============================================================================
# Scenario B — 4× 40 GB GPUs (4× A100 40GB on Snellius default gpu_a100)
# ============================================================================
# Use this when:
#   - You can't get an 80GB card (Snellius gpu_a100 default is 40GB)
#   - You're OK with model-parallel forward (slower than single-GPU but works)
#
# How it fits:
#   device_map="auto" shards Qwen3-14B (28GB bf16) across 4 GPUs → ~7 GB/GPU
#   leaves ~30 GB/GPU for activations + workspace
#   This is plenty for our RLConfig (max_length=1024, n=4).
#
# Trade-off vs single 80GB:
#   + uses cheaper / more common 40GB cards
#   - forward pass crosses GPUs via NCCL hooks → ~30-50% slower
#   - uses 4× the GPU-hour budget per step
#
# ============================================================================
# SLURM resource request
# ============================================================================
#SBATCH --job-name=grpo_quad40
#SBATCH --partition=gpu_a100                 # Snellius A100 default = 40GB
#SBATCH --nodes=1
#SBATCH --gpus=4                             # 4 GPUs on one node
#SBATCH --cpus-per-task=72                   # 18 cores × 4 GPUs
#SBATCH --mem=480G                           # 120 GB × 4 GPUs of host RAM
#SBATCH --time=02:00:00

#SBATCH --account=scur2545                   # ← replace with your account
#SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=your.email@example.com  # ← uncomment + fill in
#SBATCH --output=logs/grpo_quad40_%j.out
#SBATCH --error=logs/grpo_quad40_%j.err

# ============================================================================
# Setup
# ============================================================================
mkdir -p logs

# ⚠️  Module versions MUST match what env.sh used. See README §3.
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

# Show all 4 GPUs SLURM gave us
echo "── nvidia-smi inside the job ────────────────────────────────────"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
python --version

# ============================================================================
# Environment knobs
# ============================================================================
export HF_HOME=/projects/2/managed_datasets/hf_cache_dir
export HF_HUB_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ── Multi-GPU control ───────────────────────────────────────────────────────
# device_map="auto" inside llm_engine_minimal.py will auto-shard across visible
# GPUs.  CUDA_VISIBLE_DEVICES is set automatically by SLURM to the 4 assigned
# GPUs, so we don't need to touch it.

# Optional: disable NCCL P2P if you hit "duplicate GPU detected" or hang
# (some Snellius node pairs have flaky NVLink topology).
# export NCCL_P2P_DISABLE=1

# Optional: set HF parallelism limits so 72 CPU cores don't oversubscribe
export OMP_NUM_THREADS=18
export TOKENIZERS_PARALLELISM=false

# ── Quantization belt-and-suspenders ────────────────────────────────────────
# Pure model-parallel without quantization is enough on 4×40GB, but enabling
# 4-bit NF4 makes each shard ~2GB (instead of 7GB) and lets you grow seq
# length or batch size later without redoing the resource math.
# Set to "4bit" if you want it on, leave empty for full bf16 sharding.
export GRPO_QUANTIZATION=""    # try "4bit" if you OOM

export GRPO_MODEL_ID="Qwen/Qwen3-14B"

# ── RLConfig: shard's per-GPU budget = ~30GB, keep H100-tuned defaults ─────
# Don't override anything here — let rl_demo.py use new_rl_updater.py's
# H100-tuned defaults (lora_r=16, max_length=2048, lr=2e-5, ...).
# If you OOM, scale down the same way 40GB single-GPU does:
##export GRPO_LORA_R=8
##export GRPO_LR=1e-5
##export GRPO_MAX_LENGTH=1024

# ============================================================================
# Run
# ============================================================================
PYTHONPATH="$LOCAL_FOLDER" python -u "$LOCAL_FOLDER/pipeline/reinforcement_algorithm/rl_demo.py"

echo "DONE — exit code $?"
