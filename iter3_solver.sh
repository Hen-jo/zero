# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -x

export HF_DATASETS_CACHE=/tmp/hf_cache
export HF_HOME=/tmp/hf_home

kill -9 $(lsof -t -i :8000);

cur_iter=3
prev_iter=2

tp=2
dp=4
gpus=8
batch_per_gpu=2
rollout_memory_utilization=0.5

challenger_algorithm=grpo_batch
challenger_grpo_group_size=1
challenger_reward_group_size=5

hop_ratio=${1:-4321}
if [ $# -ge 1 ]; then
    shift
fi

algorithm=grpo
grpo_group_size=16
model=/input/model/Qwen/Qwen2.5-3B-Instruct
model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR=/input/output/zero
CONFIG_PATH="$SCRIPT_DIR/config"
TOOL_CONFIG="$CONFIG_PATH/search_tool_config.yaml"

CHALLENGER_NAME="challenger_iter${cur_iter}_ratio${hop_ratio}_${challenger_algorithm}_group${challenger_grpo_group_size}-${challenger_reward_group_size}_${model_name}"
SOLVER_NAME="solver_iter${cur_iter}_ratio${hop_ratio}_${algorithm}_group${grpo_group_size}_${model_name}"

TRAIN_DATA="$OUTPUT_DIR/data/zero_${CHALLENGER_NAME}.parquet"
VAL_DATA="/input/data/Drzero/test.parquet"
VAL_DATA_DIR="$OUTPUT_DIR/data/${SOLVER_NAME}"

RESUME_PATH="$OUTPUT_DIR/checkpoints/dr-zero/solver_iter${prev_iter}_ratio${hop_ratio}_${algorithm}_group${grpo_group_size}_${model_name}/solver_iter${prev_iter}_hf"


LOG_DIR=$OUTPUT_DIR/logs
mkdir -p $LOG_DIR
export TENSORBOARD_DIR=$OUTPUT_DIR/tensorboard

NFS_INDEX='/input/data/Drzero/corpus/e5_Flat.index'
NFS_CORPUS='/input/data/Drzero/corpus/wiki-18.jsonl'
LOCAL_CACHE='/tmp/drzero_corpus'
mkdir -p $LOCAL_CACHE

for FILE in "$NFS_INDEX" "$NFS_CORPUS"; do
    BASENAME=$(basename "$FILE")
    if [ ! -f "$LOCAL_CACHE/$BASENAME" ]; then
        echo "[$(date '+%H:%M:%S')] NFS → 로컬 복사 중: $BASENAME ..."
        cp "$FILE" "$LOCAL_CACHE/$BASENAME"
        echo "[$(date '+%H:%M:%S')] 복사 완료: $BASENAME"
    else
        echo "[$(date '+%H:%M:%S')] 로컬 캐시 사용: $BASENAME"
    fi
done

PYTHONUNBUFFERED=1 python search/retrieval_server.py \
    --index_path="$LOCAL_CACHE/e5_Flat.index" \
    --corpus_path="$LOCAL_CACHE/wiki-18.jsonl" \
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

python -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='search_multiturn_grpo' \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA"  \
    data.train_batch_size=128 \
    data.max_prompt_length=512 \
    algorithm.use_kl_in_reward=False \
    algorithm.adv_estimator=${algorithm} \
    actor_rollout_ref.model.path=$RESUME_PATH \
    actor_rollout_ref.actor.grad_clip=0.1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${batch_per_gpu} \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.n=${grpo_group_size} \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_memory_utilization} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tp} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${batch_per_gpu} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${batch_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    trainer.logger='["console", "tensorboard"]' \
    trainer.project_name="dr-zero" \
    trainer.experiment_name=$SOLVER_NAME \
    trainer.default_local_dir=$OUTPUT_DIR/checkpoints/dr-zero/$SOLVER_NAME \
    trainer.n_gpus_per_node=${gpus} \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=25 \
    trainer.val_before_train=False \
    trainer.validation_data_dir=${VAL_DATA_DIR} \
    trainer.total_epochs=1 $@