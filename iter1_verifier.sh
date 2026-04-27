# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -x

export HF_DATASETS_CACHE=/tmp/hf_cache
export HF_HOME=/tmp/hf_home

fuser -k 8000/tcp 2>/dev/null || true
fuser -k 8001/tcp 2>/dev/null || true

tp=2
dp=4
gpus=8
batch_per_gpu=2
rollout_memory_utilization=0.5

hop_ratio=${1:-4321}
if [ $# -ge 1 ]; then
    shift
fi

algorithm=grpo
grpo_group_size=5
model=/input/model/Qwen2-5-3b
model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')

# Start from Challenger iter1 checkpoint (same model doing backward reasoning)
challenger_algorithm=grpo_batch
challenger_grpo_group_size=1
challenger_reward_group_size=5
challenger_step=50

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR=/input/output/zero_no2
EXP_DIR="$OUTPUT_DIR/checkpoints/dr-zero"
CONFIG_PATH="$SCRIPT_DIR/config"
TOOL_CONFIG="$CONFIG_PATH/search_tool_config.yaml"

CHALLENGER_NAME="challenger_iter1_ratio${hop_ratio}_${challenger_algorithm}_group${challenger_grpo_group_size}-${challenger_reward_group_size}_${model_name}"
CKPT_PATH="${EXP_DIR}/${CHALLENGER_NAME}/global_step_${challenger_step}"

VERIFIER_NAME="verifier_iter1_ratio${hop_ratio}_${algorithm}_group${grpo_group_size}_${model_name}"
TRAIN_DATA="$OUTPUT_DIR/data/verifier_${CHALLENGER_NAME}/train.parquet"
VAL_DATA="/input/data/Drzero/test.parquet"


LOG_DIR=$OUTPUT_DIR/logs
mkdir -p $LOG_DIR
export TENSORBOARD_DIR=$OUTPUT_DIR/tensorboard

SOLVER_NAME="solver_iter1_ratio${hop_ratio}_${algorithm}_group${grpo_group_size}_${model_name}"
GEN_DATA_PATH="$OUTPUT_DIR/data/zero_${CHALLENGER_NAME}.parquet"
SOLVER_PRED_PATH="$OUTPUT_DIR/data/solver_pred_${SOLVER_NAME}.parquet"
VERIFIER_DATA_DIR="$OUTPUT_DIR/data/verifier_${CHALLENGER_NAME}"

echo "[$(date '+%H:%M:%S')] [0/3] Verifier 학습 데이터 생성 중..."
python "$SCRIPT_DIR/scripts/data_process/verifier_data.py" \
    --gen_data_path="$GEN_DATA_PATH" \
    --solver_pred_path="$SOLVER_PRED_PATH" \
    --local_dir="$VERIFIER_DATA_DIR"
echo "[$(date '+%H:%M:%S')] [0/3] Verifier 데이터 생성 완료"

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

echo "[$(date '+%H:%M:%S')] [1/3] retrieval server 시작..."
PYTHONUNBUFFERED=1 python $SCRIPT_DIR/search/retrieval_server.py \
    --index_path="$LOCAL_CACHE/e5_Flat.index" \
    --corpus_path="$LOCAL_CACHE/wiki-18.jsonl" \
    --retriever_model='/input/model/e5-base-v2' \
    --retriever_name='e5' \
    --topk 3 > $LOG_DIR/retrieval_server.log 2>&1 &
RETRIEVAL_PID=$!

wait_for_port() {
    local port=$1 pid=$2 log=$3 elapsed=0
    until bash -c "echo > /dev/tcp/127.0.0.1/${port}" 2>/dev/null; do
        if ! kill -0 $pid 2>/dev/null; then
            echo "[$(date '+%H:%M:%S')] [ERROR] port ${port} 서버 종료됨. 로그:"; cat $log; exit 1
        fi
        if [ $((elapsed % 3600)) -eq 0 ] && [ $elapsed -gt 0 ]; then
            echo "[$(date '+%H:%M:%S')] [INFO] port ${port} 대기 중 (${elapsed}초 경과), 최근 로그:"
            tail -3 $log
        fi
        sleep 5; elapsed=$((elapsed + 5))
    done
    echo "[$(date '+%H:%M:%S')] [INFO] port ${port} 준비 완료"
}

wait_for_port 8000 $RETRIEVAL_PID $LOG_DIR/retrieval_server.log

echo "[$(date '+%H:%M:%S')] [2/3] sglang 시작..."
python -m sglang.launch_server \
    --model=${model} \
    --port=8001 \
    --tool-call-parser=qwen25 \
    --mem-fraction-static=${rollout_memory_utilization} \
    --dp-size=${dp} \
    --tp-size=${tp} \
    --log-level=error > $LOG_DIR/sglang_server.log 2>&1 &
SGLANG_PID=$!

wait_for_port 8001 $SGLANG_PID $LOG_DIR/sglang_server.log

echo "[$(date '+%H:%M:%S')] [3/3] Verifier 학습 시작: ${VERIFIER_NAME}"
python -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='search_multiturn_grpo' \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    algorithm.use_kl_in_reward=False \
    algorithm.adv_estimator=${algorithm} \
    actor_rollout_ref.model.path=${CKPT_PATH} \
    actor_rollout_ref.actor.grad_clip=0.1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
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
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$TOOL_CONFIG \
    custom_reward_function.name=compute_verifier_score \
    custom_reward_function.path=verl/custom_reward/verifier_reward_function.py \
    trainer.logger='["console", "tensorboard"]' \
    trainer.project_name="dr-zero" \
    trainer.experiment_name=$VERIFIER_NAME \
    trainer.default_local_dir=$OUTPUT_DIR/checkpoints/dr-zero/$VERIFIER_NAME \
    trainer.n_gpus_per_node=${gpus} \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=25 \
    trainer.val_before_train=False \
    trainer.total_epochs=1 \
    trainer.total_training_steps=50 $@
