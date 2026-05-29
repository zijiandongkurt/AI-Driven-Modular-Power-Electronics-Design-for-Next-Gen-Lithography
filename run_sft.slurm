#!/bin/bash

#SBATCH -t 4:00:00
#SBATCH -p gpu_h100
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=j.d.escarate.munoz@tue.nl
#SBATCH -o ./logs/slurm-sft-%j.out

module load 2024
module load CUDA/12.6.0
module load cuDNN/9.5.0.50-CUDA-12.6.0

cd $HOME/AI-Driven-Modular-Power-Electronics-Design-for-Next-Gen-Lithography

mkdir -p logs

source $HOME/AI-Driven-Modular-Power-Electronics-Design-for-Next-Gen-Lithography/.venv/bin/activate

python -m py_compile \
sft_loop.py \
pipeline/llm_topology_generation/llm_api.py \
pipeline/SFT/sft_trainer.py

python sft_loop.py --config sft_config.json
