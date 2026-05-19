# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

set -x

export HF_DATASETS_CACHE=/tmp/hf_cache
export HF_HOME=/tmp/hf_home

fuser -k 8000/tcp 2>/dev/null || true
fuser -k 8001/tcp 2>/dev/null || true

tp=2
dp=2
gpus=4
batch_per_gpu=2
rollout_memory_utilization=0.5

hop_ratio=${1:-4321}
if [ $# -ge 1 ]; then
    shift
fi

algorithm=grpo_batch
grpo_group_size=1
model=/input/model/Qwen
model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')

# Challenger iter1 checkpoint (verifier starts from here)
challenger_algorithm=grpo_batch
challenger_grpo_group_size=1
challenger_reward_group_size=5
challenger_step=50

# Solver iter1 checkpoint (for generating verifier training data)
solver_algorithm=grpo
solver_grpo_group_size=5
solver_step=50

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR=/input/output/zero_no2
EXP_DIR="$OUTPUT_DIR/checkpoints/dr-zero"
CONFIG_PATH="$SCRIPT_DIR/config"
TOOL_CONFIG="$CONFIG_PATH/search_tool_config.yaml"

CHALLENGER_NAME="challenger_iter1_ratio${hop_ratio}_${challenger_algorithm}_group${challenger_grpo_group_size}-${challenger_reward_group_size}_${model_name}"
SOLVER_NAME="solver_iter1_ratio${hop_ratio}_${solver_algorithm}_group${solver_grpo_group_size}_${model_name}"
VERIFIER_NAME="verifier_iter1_ratio${hop_ratio}_${algorithm}_group${grpo_group_size}_${model_name}"

CHALLENGER_CKPT="${EXP_DIR}/${CHALLENGER_NAME}/challenger_hf"
# main_solver_generation.py uses wg.load_checkpoint which expects FSDP shards
# (actor/model_world_size_8_rank_X.pt) — use global_step dir directly, not solver_hf.
SOLVER_CKPT="${EXP_DIR}/${SOLVER_NAME}/global_step_${solver_step}"
SOLVER_HF_CKPT="${EXP_DIR}/${SOLVER_NAME}/solver_hf"  # kept for reference, not used here

GEN_DATA_PATH="$OUTPUT_DIR/data/zero_${CHALLENGER_NAME}.parquet"
SOLVER_PRED_FULL="$OUTPUT_DIR/data/solver_pred_full_${SOLVER_NAME}.parquet"
VERIFIER_DATA_DIR="$OUTPUT_DIR/data/verifier_${CHALLENGER_NAME}"
TRAIN_DATA="$VERIFIER_DATA_DIR/train.parquet"
VAL_DATA="/input/data/Drzero/test.parquet"
    
LOG_DIR=$OUTPUT_DIR/logs
mkdir -p $LOG_DIR || { echo "[ERROR] LOG_DIR 생성 실패: $LOG_DIR (권한 또는 경로 확인)"; exit 1; }
export TENSORBOARD_DIR=$OUTPUT_DIR/tensorboard

wait_for_port() {
    local port=$1 pid=$2 log=$3 elapsed=0
    until bash -c "echo > /dev/tcp/127.0.0.1/${port}" 2>/dev/null; do
        if ! kill -0 $pid 2>/dev/null; then
            echo "[$(date '+%H:%M:%S')] [ERROR] port ${port} 서버 종료됨. 로그:"; cat $log; exit 1
        fi
        if [ $((elapsed % 3600)) -eq 0 ] && [ $elapsed -gt 0 ]; then
            echo "[$(date '+%H:%M:%S')] [INFO] port ${port} 대기 중 (${elapsed}초 경과)"
            tail -3 $log
        fi
        sleep 5; elapsed=$((elapsed + 5))
    done
    echo "[$(date '+%H:%M:%S')] [INFO] port ${port} 준비 완료"
}

# ── [0/4] Challenger / Solver 체크포인트 변환 ─────────────────────────────
# Check config.json as completion sentinel (empty dir = incomplete conversion)
if [ ! -f "$CHALLENGER_CKPT/config.json" ]; then
    echo "[$(date '+%H:%M:%S')] [0/4] Challenger 체크포인트 변환 중..."
    rm -rf "$CHALLENGER_CKPT"
    [ ! -e "${EXP_DIR}/${CHALLENGER_NAME}/global_step_${challenger_step}/actor/huggingface" ] && \
        ln -sfn "$model" "${EXP_DIR}/${CHALLENGER_NAME}/global_step_${challenger_step}/actor/huggingface"
    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "${EXP_DIR}/${CHALLENGER_NAME}/global_step_${challenger_step}/actor" \
        --target_dir "$CHALLENGER_CKPT"
    echo "[$(date '+%H:%M:%S')] [0/4] Challenger 변환 완료: $CHALLENGER_CKPT"
else
    echo "[$(date '+%H:%M:%S')] [0/4] Challenger 이미 변환됨, 건너뜀: $CHALLENGER_CKPT"
fi

# Solver: no HF conversion needed — main_solver_generation.py loads FSDP shards directly.
if [ ! -d "$SOLVER_CKPT/actor" ]; then
    echo "[$(date '+%H:%M:%S')] [0/4] [ERROR] Solver FSDP checkpoint not found: $SOLVER_CKPT/actor"
    exit 1
else
    echo "[$(date '+%H:%M:%S')] [0/4] Solver FSDP checkpoint 확인됨: $SOLVER_CKPT"
fi

# ── [1/4] Retrieval server ────────────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] [1/4] retrieval server 시작..."
PYTHONUNBUFFERED=1 python $SCRIPT_DIR/search/retrieval_server.py \
    --index_path='/input/data/Drzero/corpus/e5_Flat.index' \
    --corpus_path='/input/data/Drzero/corpus/wiki-18.jsonl' \
    --retriever_model='/input/model/e5-base-v2' \
    --retriever_name='e5' \
    --faiss_gpu \
    --topk 3 > $LOG_DIR/retrieval_server.log 2>&1 &
RETRIEVAL_PID=$!
wait_for_port 8000 $RETRIEVAL_PID $LOG_DIR/retrieval_server.log

# ── [2/4] Solver inference (샘플링된 입력만 사용, 1920 훈련 샘플 기준 2x 버퍼) ──
SAMPLED_GEN_DATA="$OUTPUT_DIR/data/sampled_gen_${CHALLENGER_NAME}.parquet"
if [ ! -f "$SAMPLED_GEN_DATA" ]; then
    python -c "
import pandas as pd
df = pd.read_parquet('$GEN_DATA_PATH')
# 2500 rows: 1920 target + ~30% buffer for empty-answer skips, stratified by data_source
if 'data_source' in df.columns and len(df) > 2500:
    sampled = df.groupby('data_source', group_keys=False).apply(
        lambda g: g.sample(n=max(1, round(2500 * len(g) / len(df))), random_state=42))
    sampled = sampled.sample(n=min(2500, len(sampled)), random_state=42)
else:
    sampled = df.sample(n=min(2500, len(df)), random_state=42)
sampled.reset_index(drop=True).to_parquet('$SAMPLED_GEN_DATA', index=False)
print(f'Sampled {len(sampled)} / {len(df)} rows for solver inference')
"
fi

if [ ! -f "$SOLVER_PRED_FULL" ]; then
    echo "[$(date '+%H:%M:%S')] [2/4] Solver inference 시작 (${SOLVER_NAME} @ step ${solver_step})..."
    python -m verl.trainer.main_solver_generation \
        --config-path="$CONFIG_PATH" \
        --config-name='search_multiturn_grpo' \
        +ckpt_path=$SOLVER_CKPT \
        data.prompt_key=prompt \
        +data.path=$SAMPLED_GEN_DATA \
        +data.partition=null \
        +data.batch_size=512 \
        +data.output_path=$SOLVER_PRED_FULL \
        actor_rollout_ref.model.path=$model \
        actor_rollout_ref.rollout.name=sglang \
        actor_rollout_ref.rollout.n=1 \
        actor_rollout_ref.rollout.temperature=0.0 \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.prompt_length=1536 \
        actor_rollout_ref.rollout.response_length=2560 \
        actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=${gpus}
    echo "[$(date '+%H:%M:%S')] [2/4] Solver inference 완료 → ${SOLVER_PRED_FULL}"
else
    echo "[$(date '+%H:%M:%S')] [2/4] Solver pred 이미 존재, 건너뜀: ${SOLVER_PRED_FULL}"
fi

# ── [3/4] Verifier 학습 데이터 생성 ──────────────────────────────────────
echo "[$(date '+%H:%M:%S')] [3/4] Verifier 학습 데이터 생성 중..."
python "$SCRIPT_DIR/scripts/data_process/verifier_data.py" \
    --gen_data_path="$GEN_DATA_PATH" \
    --solver_pred_path="$SOLVER_PRED_FULL" \
    --local_dir="$VERIFIER_DATA_DIR" \
    --max_samples=1920
echo "[$(date '+%H:%M:%S')] [3/4] Verifier 데이터 생성 완료 → $VERIFIER_DATA_DIR"

# ── [4/4] Verifier 학습 ───────────────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] [4/4] Verifier 학습 시작: ${VERIFIER_NAME}"
python -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='search_multiturn_grpo' \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    algorithm.use_kl_in_reward=False \
    algorithm.adv_estimator=${algorithm} \
    actor_rollout_ref.model.path=${CHALLENGER_CKPT} \
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
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$TOOL_CONFIG \
    reward_model.reward_manager=batch \
    custom_reward_function.name=compute_verifier_score \
    custom_reward_function.path=verl/custom_reward/verifier_reward_function.py \
    trainer.logger='["console", "tensorboard"]' \
    trainer.project_name="dr-zero" \
    trainer.experiment_name=$VERIFIER_NAME \
    trainer.default_local_dir=$OUTPUT_DIR/checkpoints/dr-zero/$VERIFIER_NAME \
    trainer.n_gpus_per_node=${gpus} \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.total_epochs=1 \
    trainer.total_training_steps=50 $@

# ── [5/4] Verifier HF 포맷 변환 (iter2용) ────────────────────────────────
echo "[$(date '+%H:%M:%S')] [5/4] Verifier 체크포인트 → HF 변환 중..."
bash "$SCRIPT_DIR/convert.sh" 1 ${hop_ratio} ${VERIFIER_NAME}
echo "[$(date '+%H:%M:%S')] [5/4] 변환 완료"
