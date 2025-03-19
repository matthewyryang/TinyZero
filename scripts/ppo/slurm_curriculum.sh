#!/bin/bash
#SBATCH --gres=gpu:A100_80GB:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time 48:00:00
#SBATCH --partition=rl

source ~/miniconda3/etc/profile.d/conda.sh
conda activate zero

CONTEXT_LENGTH=1024
DATA_DISTRIBUTION="countdown"

export N_GPUS=2
export CONTEXT_LENGTH=$CONTEXT_LENGTH
export BASE_MODEL=/data/user_data/myang4/countdown/countdown-equal-512/actor/global_step_150
export DATA_DIR="/home/myang4/TinyZero/data/$DATA_DISTRIBUTION"
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME="$DATA_DISTRIBUTION-$CONTEXT_LENGTH-length-curriculum"
export VLLM_ATTENTION_BACKEND=XFORMERS

# ray stop --force && ray start --head

cd /home/myang4/TinyZero

bash /home/myang4/TinyZero/scripts/ppo/train_tiny_zero_curriculum.sh > /home/myang4/TinyZero/logs/$DATA_DISTRIBUTION-$CONTEXT_LENGTH-length-curriculum.log 2>&1
