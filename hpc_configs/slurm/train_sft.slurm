#!/bin/bash
# ============================================================================
# SFT (supervised fine-tuning) — Qwen3-14B + LoRA
# ============================================================================
# Trains on data/sft/sft_train.jsonl (81 samples), validates on
# data/sft/sft_val.jsonl (9 held-out samples).
#
# Expected wall time on H100 80GB bf16:
#   81 samples × 5 epochs ÷ batch_size=4 ≈ 100 optimizer steps
#   ~10-15s per step (forward + backward + AdamW)
#   total ≈ 20-25 minutes (including model load)
#
# Expected VRAM on H100 80GB bf16 (no quantization):
#   model weights ........... 28 GB
#   LoRA + AdamW ...........  0.4 GB
#   4 × seq=2048 activations . 40 GB
#   misc / KV / workspace ...  6 GB
#   ────────────────────────────────
#   total ................... ~74 GB out of 93 GB
#
# After it finishes:
#   checkpoints/sft-lora/epoch-{1..5}/    per-epoch checkpoints
#   checkpoints/sft-lora/final/            best LoRA adapter
#   checkpoints/sft-lora/history.json      per-epoch train/val loss curve
#
# To continue with GRPO from this SFT checkpoint:
#   In rl_demo.py, after constructing TopologyLLM:
#       llm.engine.load_adapter("sft", "./checkpoints/sft-lora/final")
#       llm.engine.set_active("sft")
#   Then GRPOTrainer(...) trains on top of the SFT-warmed weights.
# ============================================================================

#SBATCH --job-name=sft
#SBATCH --partition=gpu_h100         # H100 80GB
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16           # H100 nodes = 16 cores per GPU
#SBATCH --mem=120G
#SBATCH --time=01:00:00              # 1h is plenty for 81 samples × 5 epochs

#SBATCH --account=scur2545           # ← replace with your account name
#SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=your.email@example.com
#SBATCH --output=logs/sft_%j.out
#SBATCH --error=logs/sft_%j.err

mkdir -p logs

# ── Modules (must match env.sh's build environment) ────────────────────────
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1

# ── Project paths ──────────────────────────────────────────────────────────
LOCAL_FOLDER=$HOME/AI-Driven-Modular-Power-Electronics-Design-for-Next-Gen-Lithography
cd "$LOCAL_FOLDER" || { echo "Repo not found at $LOCAL_FOLDER"; exit 1; }
source .venv/bin/activate || { echo ".venv not found — did you run env.sh?"; exit 2; }

echo "── nvidia-smi ─────────────────────────────────────────────────"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# ── HF cache (Snellius shared, flat layout) ────────────────────────────────
export HF_HOME=/projects/2/managed_datasets/hf_cache_dir
export HF_HUB_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# ── Model / SFT knobs ──────────────────────────────────────────────────────
export GRPO_MODEL_ID="Qwen/Qwen3-14B"
export GRPO_QUANTIZATION=""           # H100 80GB → bf16 full precision

# SFT hyperparameters (defaults inside SFTConfig match these)
export SFT_TRAIN_PATH="$LOCAL_FOLDER/data/sft/sft_train.jsonl"
export SFT_VAL_PATH="$LOCAL_FOLDER/data/sft/sft_val.jsonl"
export SFT_EPOCHS=5
export SFT_BATCH_SIZE=4
export SFT_LR=2e-4
export SFT_LORA_R=16
export SFT_LORA_ALPHA=32
export SFT_MAX_LENGTH=2048
export SFT_OUTPUT_DIR="$LOCAL_FOLDER/checkpoints/sft-lora"

# ── Run ────────────────────────────────────────────────────────────────────
PYTHONPATH="$LOCAL_FOLDER" python -u "$LOCAL_FOLDER/scripts/run_sft.py"

# ── Post-run snapshot ──────────────────────────────────────────────────────
echo "── nvidia-smi after run ───────────────────────────────────────"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

echo "── history.json ───────────────────────────────────────────────"
cat "$SFT_OUTPUT_DIR/history.json" 2>/dev/null || echo "(no history)"

echo "DONE — exit code $?"
