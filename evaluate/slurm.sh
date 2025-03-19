#!/bin/bash
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time 48:00:00
#SBATCH --partition=preempt

CONTEXT_LENGTH=$1
CHECKPOINT=$2

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

cd /home/myang4/TinyZero
python evaluate/evaluate_checkpoint.py --checkpoint=$CHECKPOINT --context-length=$CONTEXT_LENGTH 2>&1 | tee /home/myang4/TinyZero/logs/evaluate-$CHECKPOINT-$CONTEXT_LENGTH.log
