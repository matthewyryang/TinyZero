#!/bin/bash
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=60G
#SBATCH --time 48:00:00
#SBATCH --partition=preempt


start=$1
end=$2

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

cd /home/myang4/TinyZero
python examples/data_preprocess/math_difficulty.py --dataset_start $start --dataset_end $end > /home/myang4/TinyZero/logs/pass_at_k_${start}_${end}.log 2>&1
