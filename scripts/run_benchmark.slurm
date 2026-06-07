#!/bin/bash

#SBATCH -t 72:00:00
#SBATCH -p gpu_h100
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=a.kumas@student.tue.nl
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=a.kumas@student.tue.nl
#SBATCH -o /home/akumas/AI-Driven-Modular-Power-Electronics-Design-for-Next-Gen-Lithography/logs/slurm-benchmark-%j.out

module load 2024
module load CUDA/12.6.0
module load cuDNN/9.5.0.50-CUDA-12.6.0

cd $HOME/AI-Driven-Modular-Power-Electronics-Design-for-Next-Gen-Lithography

mkdir -p logs

source $HOME/AI-Driven-Modular-Power-Electronics-Design-for-Next-Gen-Lithography/.venv/bin/activate

python -m py_compile \
experiments/run_benchmark.py \
run_inference.py \
pipeline/llm_topology_generation/llm_api.py \
pipeline/llm_topology_generation/net_writer.py \
pipeline/utility/netlist_database.py \
pipeline/utility/summary_logger.py

python experiments/run_benchmark.py --config configs/benchmark_config.json
