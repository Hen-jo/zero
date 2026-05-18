# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -x

fuser -k 8000/tcp 2>/dev/null || true

tp=2
dp=2
gpus=4
batch_per_gpu=1
rollout_memory_utilization=0.5

challenger_algorithm=grpo_batch
challenger_grpo_group_size=1
challenger_reward_group_size=5

hop_ratio=${1:-4321}
if [ $# -ge 1 ]; then
    shift
fi

algorithm=grpo
grpo_group_size=5
solver_total_steps=50
model=/input/model/Qwen/Qwen2.5-3B-Instruct
model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR=/input/output/zero_no2
CONFIG_PATH="$SCRIPT_DIR/config"
TOOL_CONFIG="$CONFIG_PATH/search_tool_config_solver.yaml"

CHALLENGER_NAME="challenger_iter1_ratio${hop_ratio}_${challenger_algorithm}_group${challenger_grpo_group_size}-${challenger_reward_group_size}_${model_name}"
SOLVER_NAME="solver_iter1_ratio${hop_ratio}_${algorithm}_group${grpo_group_size}_${model_name}"

TRAIN_DATA="$OUTPUT_DIR/data/zero_${CHALLENGER_NAME}.parquet"
VAL_DATA_DIR="$OUTPUT_DIR/data/${SOLVER_NAME}"
VAL_DATA="/input/data/Drzero/test.parquet"

# 마지막 step rollout 저장 경로 (verifier_gen_data 추론 대체)
SOLVER_PRED_OUT="$OUTPUT_DIR/data/solver_pred_${SOLVER_NAME}.parquet"


# source activate llm;

export HF_DATASETS_CACHE=/tmp/hf_cache
export HF_HOME=/tmp/hf_home

LOG_DIR=$OUTPUT_DIR/logs
mkdir -p $LOG_DIR || { echo "[ERROR] LOG_DIR 생성 실패: $LOG_DIR (권한 또는 경로 확인)"; exit 1; }
export TENSORBOARD_DIR=$OUTPUT_DIR/tensorboard

NFS_INDEX='/input/data/Drzero/corpus/e5_Flat.index'
NFS_CORPUS='/input/data/Drzero/corpus/wiki-18.jsonl'

PYTHONUNBUFFERED=1 python search/retrieval_server.py \
    --index_path="$NFS_INDEX" \
    --corpus_path="$NFS_CORPUS" \
    --retriever_model='/input/model/e5-base-v2' \
    --retriever_name='e5' \
    --faiss_gpu \
    --topk 3 > $LOG_DIR/retrieval_server.log 2>&1 &
RETRIEVAL_PID=$!

echo "[INFO] retrieval server 대기 중 (port 8000)..."
until bash -c "echo > /dev/tcp/127.0.0.1/8000" 2>/dev/null; do
    if ! kill -0 $RETRIEVAL_PID 2>/dev/null; then
        echo "[ERROR] retrieval server 종료됨. 로그:"; cat $LOG_DIR/retrieval_server.log; exit 1
    fi
    echo "[INFO] retrieval server 아직 준비 안 됨, 5초 후 재시도..."
    sleep 5
done
echo "[INFO] retrieval server 준비 완료"

python -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='search_multiturn_grpo' \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA"  \
    data.train_batch_size=128 \
    data.max_prompt_length=512 \
    algorithm.use_kl_in_reward=False \
    algorithm.adv_estimator=${algorithm} \
    actor_rollout_ref.model.path=${model} \
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
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    trainer.logger='["console", "tensorboard"]' \
    trainer.project_name="dr-zero" \
    trainer.experiment_name=$SOLVER_NAME \
    trainer.default_local_dir=$OUTPUT_DIR/checkpoints/dr-zero/$SOLVER_NAME \
    trainer.n_gpus_per_node=${gpus} \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.max_actor_ckpt_to_keep=1 \
    data.val_batch_size=64 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.validation_data_dir=${VAL_DATA_DIR} \
    trainer.rollout_data_dir=${VAL_DATA_DIR}/train_rollouts \
    trainer.rollout_log_freq=25 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=${solver_total_steps} \
    +trainer.save_last_rollout_path="$SOLVER_PRED_OUT" $@

# ── Solver HF 포맷 변환 (iter2_challenger용) ─────────────────────────────
SOLVER_CKPT="${OUTPUT_DIR}/checkpoints/dr-zero/${SOLVER_NAME}/global_step_${solver_total_steps}"
SOLVER_HF_CKPT="${OUTPUT_DIR}/checkpoints/dr-zero/${SOLVER_NAME}/solver_hf"
if [ ! -f "$SOLVER_HF_CKPT/config.json" ]; then
    echo "[$(date '+%H:%M:%S')] Solver 체크포인트 → HF 변환 중..."
    rm -rf "$SOLVER_HF_CKPT"
    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "${SOLVER_CKPT}/actor" \
        --target_dir "$SOLVER_HF_CKPT"
    echo "[$(date '+%H:%M:%S')] Solver HF 변환 완료: $SOLVER_HF_CKPT"
else
    echo "[$(date '+%H:%M:%S')] Solver HF 이미 존재, 건너뜀: $SOLVER_HF_CKPT"
fi
