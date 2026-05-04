# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# Runs solver inference on the full challenger gen data (33K+ samples)
# to produce real (question, solver_pred, is_correct) pairs for verifier training.

set -x

export HF_DATASETS_CACHE=/tmp/hf_cache
export HF_HOME=/tmp/hf_home

fuser -k 8000/tcp 2>/dev/null || true

tp=1
dp=8
gpus=8
sample_size=1
rollout_memory_utilization=0.8

hop_ratio=${1:-4321}
if [ $# -ge 1 ]; then
    shift
fi

challenger_algorithm=grpo_batch
challenger_grpo_group_size=1
challenger_reward_group_size=5
solver_algorithm=grpo
solver_grpo_group_size=5
model=/input/model/Qwen/Qwen2.5-3B-Instruct
model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')

solver_step=50

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR=/input/output/original
EXP_DIR="$OUTPUT_DIR/checkpoints/dr-zero"
CONFIG_PATH="$SCRIPT_DIR/config"
TOOL_CONFIG="$CONFIG_PATH/search_tool_config.yaml"

CHALLENGER_NAME="challenger_iter1_ratio${hop_ratio}_${challenger_algorithm}_group${challenger_grpo_group_size}-${challenger_reward_group_size}_${model_name}"
SOLVER_NAME="solver_iter1_ratio${hop_ratio}_${solver_algorithm}_group${solver_grpo_group_size}_${model_name}"

# Input: challenger gen data (questions + ground truth)
TRAIN_DATA="$OUTPUT_DIR/data/zero_${CHALLENGER_NAME}.parquet"

# Solver checkpoint (latest step)
CKPT_PATH="${EXP_DIR}/${SOLVER_NAME}/global_step_${solver_step}"

# Output: solver predictions on all challenger questions
SOLVER_PRED_OUT="$OUTPUT_DIR/data/solver_pred_full_${SOLVER_NAME}.parquet"

LOG_DIR=$OUTPUT_DIR/logs
mkdir -p $LOG_DIR

echo "[$(date '+%H:%M:%S')] Solver gen: ${SOLVER_NAME} @ step ${solver_step}"
echo "[$(date '+%H:%M:%S')] Input : ${TRAIN_DATA}"
echo "[$(date '+%H:%M:%S')] Output: ${SOLVER_PRED_OUT}"

PYTHONUNBUFFERED=1 python search/retrieval_server.py \
    --index_path='/input/data/Drzero/corpus/e5_Flat.index' \
    --corpus_path='/input/data/Drzero/corpus/wiki-18.jsonl' \
    --retriever_model='/input/model/e5-base-v2' \
    --retriever_name='e5' \
    --topk 3 > $LOG_DIR/retrieval_server.log 2>&1 &
RETRIEVAL_PID=$!

echo "[$(date '+%H:%M:%S')] retrieval server 대기 중 (port 8000)..."
until bash -c "echo > /dev/tcp/127.0.0.1/8000" 2>/dev/null; do
    if ! kill -0 $RETRIEVAL_PID 2>/dev/null; then
        echo "[$(date '+%H:%M:%S')] [ERROR] retrieval server 종료됨. 로그:"; cat $LOG_DIR/retrieval_server.log; exit 1
    fi
    sleep 5
done
echo "[$(date '+%H:%M:%S')] retrieval server 준비 완료"

python -m verl.trainer.main_solver_generation \
    --config-path="$CONFIG_PATH" \
    --config-name='search_multiturn_grpo' \
    +ckpt_path=$CKPT_PATH \
    data.prompt_key=prompt \
    +data.path=$TRAIN_DATA \
    +data.partition=1 \
    +data.batch_size=512 \
    +data.output_path=$SOLVER_PRED_OUT \
    actor_rollout_ref.model.path=$model \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.n=${sample_size} \
    actor_rollout_ref.rollout.temperature=0.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.prompt_length=1536 \
    actor_rollout_ref.rollout.response_length=2560 \
    actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_memory_utilization} \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=${gpus} $@

echo "[$(date '+%H:%M:%S')] 완료: ${SOLVER_PRED_OUT}"
