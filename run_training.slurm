#!/bin/bash

#SBATCH -t 48:00:00
#SBATCH -p gpu_h100
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=a.kumas@student.tue.nl
#SBATCH -o ./logs/slurm-train-%j.out

module load 2024
module load CUDA/12.6.0
module load cuDNN/9.5.0.50-CUDA-12.6.0

cd $HOME/AI-Driven-Modular-Power-Electronics-Design-for-Next-Gen-Lithography

mkdir -p logs

source $HOME/AI-Driven-Modular-Power-Electronics-Design-for-Next-Gen-Lithography/.venv/bin/activate

python -m py_compile \
training_loop.py \
training_loop_random.py \
pipeline/llm_topology_generation/llm_api.py \
pipeline/llm_topology_generation/net_writer.py \
pipeline/reinforcement_algorithm/grpo_trainer.py \
pipeline/reinforcement_algorithm/new_rl_updater.py \
pipeline/graphs_and_visualizations/plot_probabilities.py \
pipeline/graphs_and_visualizations/plot_cumulative_probabilities.py \
pipeline/utility/summary_logger.py

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python training_loop_random.py
