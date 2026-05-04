# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -x

export HF_DATASETS_CACHE=/tmp/hf_cache
export HF_HOME=/tmp/hf_home

fuser -k 8000/tcp 2>/dev/null || true

tp=1
dp=8
gpus=8
sample_size=5
rollout_memory_utilization=0.8

hop_ratio=${1:-4321}
if [ $# -ge 1 ]; then
    shift
fi

algorithm=grpo_batch
grpo_group_size=1
reward_group_size=5
model=/input/model/Qwen/Qwen2.5-3B-Instruct
model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')

challenger_step=30
data_partition=3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR=/input/output/zero_no2
EXP_DIR="$OUTPUT_DIR/checkpoints/dr-zero"
MODEL_PATH="challenger_iter3_ratio${hop_ratio}_${algorithm}_group${grpo_group_size}-${reward_group_size}_${model_name}"
CKPT_PATH="${EXP_DIR}/${MODEL_PATH}/global_step_${challenger_step}"

CONFIG_PATH="$SCRIPT_DIR/config"
TOOL_CONFIG="$CONFIG_PATH/search_tool_config.yaml"

TRAIN_DATA="/input/data/Drzero/zero_ratio${hop_ratio}.parquet"
TRAIN_DATA_OUT="$OUTPUT_DIR/data/zero_${MODEL_PATH}.parquet"


LOG_DIR=$OUTPUT_DIR/logs
mkdir -p $LOG_DIR
export TENSORBOARD_DIR=$OUTPUT_DIR/tensorboard

NFS_INDEX='/input/data/Drzero/corpus/e5_Flat.index'
NFS_CORPUS='/input/data/Drzero/corpus/wiki-18.jsonl'

PYTHONUNBUFFERED=1 python search/retrieval_server.py \
    --index_path="$NFS_INDEX" \
    --corpus_path="$NFS_CORPUS" \
    --retriever_model='/input/model/e5-base-v2' \
    --retriever_name='e5' \
    --topk 3 > $LOG_DIR/retrieval_server.log 2>&1 &
RETRIEVAL_PID=$!

echo "[$(date '+%H:%M:%S')] [INFO] retrieval server 대기 중 (port 8000)..."
until bash -c "echo > /dev/tcp/127.0.0.1/8000" 2>/dev/null; do
    if ! kill -0 $RETRIEVAL_PID 2>/dev/null; then
        echo "[$(date '+%H:%M:%S')] [ERROR] retrieval server 종료됨. 로그:"; cat $LOG_DIR/retrieval_server.log; exit 1
    fi
    echo "[$(date '+%H:%M:%S')] [INFO] retrieval server 아직 준비 안 됨, 5초 후 재시도..."
    sleep 5
done
echo "[$(date '+%H:%M:%S')] [INFO] retrieval server 준비 완료"

python -m verl.trainer.main_generation \
    --config-path="$CONFIG_PATH" \
    --config-name='search_multiturn_grpo' \
    +ckpt_path=$CKPT_PATH \
    data.prompt_key=prompt \
    +data.path=$TRAIN_DATA \
    +data.partition=$data_partition \
    +data.batch_size=512 \
    +data.output_path=$TRAIN_DATA_OUT \
    actor_rollout_ref.model.path=$model \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.n=${sample_size} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.prompt_length=1536 \
    actor_rollout_ref.rollout.response_length=2560 \
    actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_memory_utilization} \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=${gpus} $@
