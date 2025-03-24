#!/bin/bash
#SBATCH --gres=gpu:A100_80GB:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time 48:00:00
#SBATCH --partition=rl

source ~/miniconda3/etc/profile.d/conda.sh
conda activate zero-rl

CONTEXT_LENGTH=1024
DATA_DISTRIBUTION="countdown-3-4-5-6"

export N_GPUS=2
export BATCH_SIZE=4
export CONTEXT_LENGTH=$CONTEXT_LENGTH

export BASE_MODEL=/data/group_data/rl/cognitive_behaviors/global_step_75
# export BASE_MODEL=/data/user_data/myang4/countdown/countdown-3-4-5-6-256-llama-grpo/actor/global_step_600
# export BASE_MODEL=/data/user_data/myang4/countdown/count-3-4-5-6-to-countdown-6-7-8-9-256-llama-grpo/actor/global_step_600

export DATA_DIR="/home/myang4/TinyZero/data/$DATA_DISTRIBUTION"
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME="$DATA_DISTRIBUTION-$CONTEXT_LENGTH-llama-grpo-token-level-gradient-no-std"
export OUTPUT_DIR="/data/user_data/myang4/countdown/$EXPERIMENT_NAME"
export VLLM_ATTENTION_BACKEND=XFORMERS
export SAVE_FREQ=-1

# ray stop --force && ray start --head

cd /home/myang4/TinyZero

bash /home/myang4/TinyZero/scripts/grpo/grpo.sh > /home/myang4/TinyZero/logs/$EXPERIMENT_NAME.log 2>&1
