#!/bin/bash
#SBATCH --gres=gpu:L40S:2
#SBATCH --cpus-per-task=4
#SBATCH --mem 40GB
#SBATCH --time 48:00:00
#SBATCH --partition=preempt


source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm3

cd /home/myang4/TinyZero

difficulty=$1

export NCCL_P2P_DISABLE=1

# For all machines
export OUTLINES_CACHE_DIR=/tmp/.outlines # hugging face cache
export VLLM_LOGGING_LEVEL=DEBUG


python data_preprocess/multiply/pass_at_k.py --input_dataset_file /data/group_data/rl/datasets/multiply/multiply-train-$difficulty.json --difficulty $difficulty --dataset_start 0 --dataset_end 10  > /home/myang4/TinyZero/logs/multiply-$difficulty.log 2>&1

