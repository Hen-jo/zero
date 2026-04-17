# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -x

kill -9 $(lsof -t -i :8000);

cur_iter=2

tp=1
dp=8
gpus=8
sample_size=5
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
solver_step=100

model=/input/model/Qwen2-5-3b
model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR=/input/output/zero
EXP_DIR="$OUTPUT_DIR/checkpoints/dr-zero"
CONFIG_PATH="$SCRIPT_DIR/config"
TOOL_CONFIG="$CONFIG_PATH/search_tool_config.yaml"

CHALLENGER_NAME="challenger_iter${cur_iter}_ratio${hop_ratio}_${challenger_algorithm}_group${challenger_grpo_group_size}-${challenger_reward_group_size}_${model_name}"
SOLVER_NAME="solver_iter${cur_iter}_ratio${hop_ratio}_${solver_algorithm}_group${solver_grpo_group_size}_${model_name}"

TRAIN_DATA="$OUTPUT_DIR/data/zero_${CHALLENGER_NAME}.parquet"
SOLVER_PRED_OUT="$OUTPUT_DIR/data/solver_pred_${SOLVER_NAME}.parquet"

CKPT_PATH="${EXP_DIR}/${SOLVER_NAME}/global_step_${solver_step}"

python search/retrieval_server.py \
    --index_path='/input/data/Drzero/corpus/e5_Flat.index' \
    --corpus_path='/input/data/Drzero/corpus/wiki-18.jsonl' \
    --retriever_model='/input/model/e5-base-v2' \
    --retriever_name='e5' \
    --faiss_gpu \
    --topk 3 &

echo "[INFO] retrieval server 대기 중 (port 8000)..."
until bash -c "echo > /dev/tcp/127.0.0.1/8000" 2>/dev/null; do
    echo "[INFO] retrieval server 아직 준비 안 됨, 5초 후 재시도..."
    sleep 5
done
echo "[INFO] retrieval server 준비 완료"

python -m verl.trainer.main_generation \
    --config-path="$CONFIG_PATH" \
    --config-name='search_multiturn_grpo' \
    +ckpt_path=$CKPT_PATH \
    data.prompt_key=prompt \
    +data.path=$TRAIN_DATA \
    +data.batch_size=512 \
    +data.output_path=$SOLVER_PRED_OUT \
    actor_rollout_ref.model.path=$model \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.n=${sample_size} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.prompt_length=512 \
    actor_rollout_ref.rollout.response_length=2048 \
    actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_memory_utilization} \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=${gpus} $@
