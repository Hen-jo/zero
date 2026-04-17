# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -x

export HF_DATASETS_CACHE=/input/cache/huggingface

kill -9 $(lsof -t -i :8000);
kill -9 $(lsof -t -i :8001);

cur_iter=3
prev_iter=2

tp=2
dp=4
gpus=8
batch_per_gpu=2
rollout_memory_utilization=0.25

solver_algorithm=grpo
solver_grpo_group_size=5

hop_ratio=${1:-4321}
if [ $# -ge 1 ]; then
    shift
fi

algorithm=grpo_batch
grpo_group_size=1
reward_group_size=5
model=/input/model/Qwen2-5-3b
model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR=/input/output/zero
LOG_DIR=$OUTPUT_DIR/logs
mkdir -p $LOG_DIR
CONFIG_PATH="$SCRIPT_DIR/config"
TOOL_CONFIG="$CONFIG_PATH/search_tool_config.yaml"

TRAIN_DATA="/input/data/Drzero/zero_ratio${hop_ratio}.parquet"
VAL_DATA="/input/data/Drzero/test.parquet"

SOLVER_NAME="solver_iter${prev_iter}_hf"
CHALLENGER_NAME="challenger_iter${cur_iter}_ratio${hop_ratio}_${algorithm}_group${grpo_group_size}-${reward_group_size}_${model_name}"
SOLVER_PATH="$OUTPUT_DIR/checkpoints/dr-zero/solver_iter${prev_iter}_ratio${hop_ratio}_${solver_algorithm}_group${solver_grpo_group_size}_${model_name}/${SOLVER_NAME}"
RESUME_PATH="$OUTPUT_DIR/checkpoints/dr-zero/challenger_iter${prev_iter}_ratio${hop_ratio}_${algorithm}_group${grpo_group_size}-${reward_group_size}_${model_name}/global_step_100"


echo "[$(date '+%H:%M:%S')] [1/3] retrieval server 시작..."
PYTHONUNBUFFERED=1 python search/retrieval_server.py \
    --index_path='/input/data/Drzero/corpus/e5_Flat.index' \
    --corpus_path='/input/data/Drzero/corpus/wiki-18.jsonl' \
    --retriever_model='/input/model/e5-base-v2' \
    --retriever_name='e5' \
    --faiss_gpu \
    --topk 3 > $LOG_DIR/retrieval_server.log 2>&1 &
RETRIEVAL_PID=$!

wait_for_port() {
    local port=$1 pid=$2 log=$3 elapsed=0 timeout=7200
    until bash -c "echo > /dev/tcp/127.0.0.1/${port}" 2>/dev/null; do
        if ! kill -0 $pid 2>/dev/null; then
            echo "[$(date '+%H:%M:%S')] [ERROR] port ${port} 서버 종료됨. 로그:"; cat $log; exit 1
        fi
        if [ $elapsed -ge $timeout ]; then
            echo "[$(date '+%H:%M:%S')] [ERROR] port ${port} 타임아웃 (${timeout}초). 로그:"; cat $log; exit 1
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

echo "[$(date '+%H:%M:%S')] [2/3] sglang 시작 ($SOLVER_NAME)..."
python -m sglang.launch_server \
    --model=$SOLVER_PATH \
    --served-model-name=$SOLVER_NAME \
    --port=8001 \
    --tool-call-parser=qwen25 \
    --mem-fraction-static=${rollout_memory_utilization} \
    --dp-size=${dp} \
    --tp-size=${tp} \
    --log-level=error > $LOG_DIR/sglang_server.log 2>&1 &
SGLANG_PID=$!

wait_for_port 8001 $SGLANG_PID $LOG_DIR/sglang_server.log

echo "[$(date '+%H:%M:%S')] [3/3] 학습 시작: $CHALLENGER_NAME"
python -m verl.trainer.main_ppo \
    --config-path=$CONFIG_PATH \
    --config-name="search_multiturn_grpo" \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA  \
    data.train_batch_size=256 \
    algorithm.use_kl_in_reward=False \
    algorithm.adv_estimator=${algorithm} \
    actor_rollout_ref.model.path=${model} \
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
    reward_model.reward_manager=batch \
    custom_reward_function.name=compute_challenger_score_batch \
    custom_reward_function.path=verl/custom_reward/reward_function.py \
    custom_reward_function.reward_kwargs.model_name=$SOLVER_NAME \
    custom_reward_function.reward_kwargs.base_url="http://127.0.0.1:8001" \
    custom_reward_function.reward_kwargs.reward_rollout_n=${reward_group_size} \
    trainer.logger='["console"]' \
    trainer.project_name="dr-zero" \
    trainer.experiment_name=$CHALLENGER_NAME \
    trainer.default_local_dir=$OUTPUT_DIR/checkpoints \
    trainer.resume_mode="resume_path" \
    trainer.resume_from_path=$RESUME_PATH \
    trainer.n_gpus_per_node=${gpus} \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.total_epochs=1 $@