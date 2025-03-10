#!/bin/bash
#SBATCH --gres=gpu:A100_80GB:2
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time 48:00:00
#SBATCH --partition=rl

source ~/miniconda3/etc/profile.d/conda.sh
conda activate zero

CONTEXT_LENGTH=4096
DATA_DISTRIBUTION="equal" 

export N_GPUS=2
export CONTEXT_LENGTH=$CONTEXT_LENGTH
export BASE_MODEL=/data/group_data/rl/myang4/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b
export DATA_DIR="/home/myang4/TinyZero/data/$DATA_DISTRIBUTION"
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME="countdown-$DATA_DISTRIBUTION-$CONTEXT_LENGTH"
export VLLM_ATTENTION_BACKEND=XFORMERS

cd /home/myang4/TinyZero

bash ./scripts/train_tiny_zero.sh > /home/myang4/TinyZero/logs/countdown-$DATA_DISTRIBUTION-$CONTEXT_LENGTH.log 2>&1