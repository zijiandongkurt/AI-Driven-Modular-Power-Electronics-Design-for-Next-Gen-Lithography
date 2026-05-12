#!/bin/bash
# ============================================================================
# Qwen I/O smoke test  (no GRPO, no validator, no simulator)
# ============================================================================
# 用途:
#   - 验证你抢到的 GPU 节点能正常加载 Qwen 并生成内容
#   - 跑 GRPO 之前先确认环境通了,省得复杂任务卡在 import / cache / quant
#   - 大约 3-5 分钟跑完 (加载 ~15s + 生成 2 候选 ~30s)
#
# 默认配置假设单 40GB 卡:启用 4-bit 量化.如果你抢到了 80GB 卡,
# 把 GRPO_QUANTIZATION 那一行注释掉即可.
# ============================================================================

#SBATCH --job-name=qwen_smoke
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:15:00                      # 15 分钟够了

#SBATCH --account=scur2545                   # ← replace
#SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=your.email@example.com
#SBATCH --output=logs/qwen_smoke_%j.out
#SBATCH --error=logs/qwen_smoke_%j.err

# ============================================================================
# Setup
# ============================================================================
mkdir -p logs

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1

LOCAL_FOLDER=$HOME/AI-Driven-Modular-Power-Electronics-Design-for-Next-Gen-Lithography

cd "$LOCAL_FOLDER" || { echo "Repo not found at $LOCAL_FOLDER"; exit 1; }
source .venv/bin/activate || { echo ".venv not found — did you run env.sh?"; exit 2; }

# Show the GPU we got
echo "── nvidia-smi ────────────────────────────────────────────────"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

# ============================================================================
# Knobs
# ============================================================================
export HF_CACHE_DIR=/projects/2/managed_datasets/hf_cache_dir
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

export GRPO_MODEL_ID="Qwen/Qwen3-14B"

# 自动决定要不要量化:< 50GB 就开 4-bit
if [ "$GPU_MEM_MB" -lt 50000 ]; then
  echo "→ 40GB-class GPU detected — enabling 4-bit quantization"
  export GRPO_QUANTIZATION="4bit"
else
  echo "→ 80GB-class GPU detected — running full bf16"
  export GRPO_QUANTIZATION=""
fi

# Demo-specific
export QWEN_DEMO_N=2                # 生成 2 个候选,够看效果
export QWEN_DEMO_MAX_NEW=512

# ============================================================================
# Run
# ============================================================================
PYTHONPATH="$LOCAL_FOLDER" python -u "$LOCAL_FOLDER/hpc_configs/qwen_io_demo.py"

echo ""
echo "── nvidia-smi after run ──────────────────────────────────────"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

echo "DONE — exit $?"
