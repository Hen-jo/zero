#!/bin/bash
# Convert iter1 solver FSDP checkpoint → HF format (needed by iter2_challenger.sh)
set -x

export HF_DATASETS_CACHE=/tmp/hf_cache
export HF_HOME=/tmp/hf_home

hop_ratio=${1:-4321}

algorithm=grpo
grpo_group_size=5
solver_total_steps=30
model=/input/model/Qwen/Qwen2.5-3B-Instruct
model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')

OUTPUT_DIR=/input/output/zero_no2
EXP_DIR="$OUTPUT_DIR/checkpoints/dr-zero"

SOLVER_NAME="solver_iter1_ratio${hop_ratio}_${algorithm}_group${grpo_group_size}_${model_name}"
SOLVER_CKPT="${EXP_DIR}/${SOLVER_NAME}/global_step_${solver_total_steps}"
SOLVER_HF_CKPT="${EXP_DIR}/${SOLVER_NAME}/solver_hf"

if [ ! -d "$SOLVER_CKPT/actor" ]; then
    echo "[ERROR] FSDP checkpoint not found: $SOLVER_CKPT/actor"
    exit 1
fi

if [ ! -f "$SOLVER_HF_CKPT/config.json" ]; then
    echo "[$(date '+%H:%M:%S')] Solver 체크포인트 → HF 변환 중..."
    rm -rf "$SOLVER_HF_CKPT"
    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "${SOLVER_CKPT}/actor" \
        --target_dir "$SOLVER_HF_CKPT"
    echo "[$(date '+%H:%M:%S')] 변환 완료: $SOLVER_HF_CKPT"
else
    echo "[$(date '+%H:%M:%S')] 이미 존재, 건너뜀: $SOLVER_HF_CKPT"
fi
