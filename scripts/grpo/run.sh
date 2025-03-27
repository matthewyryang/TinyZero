
CONTEXT_LENGTH=1024
DATA_DISTRIBUTION="countdown-3-4-5-6"

export N_GPUS=2
export BATCH_SIZE=4
export CONTEXT_LENGTH=$CONTEXT_LENGTH

export BASE_MODEL=d1shs0ap/cognitive-behaviors-Llama-3.2-3B

export DATA_DIR="/home/cmu/TinyZero/data/$DATA_DISTRIBUTION"
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME="$DATA_DISTRIBUTION-$CONTEXT_LENGTH-grpo-aha"
export OUTPUT_DIR="/data/user_data/cmu/countdown/$EXPERIMENT_NAME"
export VLLM_ATTENTION_BACKEND=XFORMERS
export SAVE_FREQ=-1

# ray stop --force && ray start --head

bash /home/cmu/TinyZero/scripts/grpo/grpo.sh > /home/cmu/TinyZero/logs/$EXPERIMENT_NAME.log 2>&1
