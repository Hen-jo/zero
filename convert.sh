set -e

iter=${1:-1}
hop_ratio=${2:-4321}
# Optional: pass full experiment name directly as 3rd arg
exp_name=${3:-""}

algorithm=grpo_batch
grpo_group_size=1
model=/input/model/Qwen/Qwen2.5-3B-Instruct
model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')

OUTPUT_DIR=/input/output/zero_no2
if [ -n "$exp_name" ]; then
    CKPT_BASE="$OUTPUT_DIR/checkpoints/dr-zero/${exp_name}"
else
    CKPT_BASE="$OUTPUT_DIR/checkpoints/dr-zero/verifier_iter${iter}_ratio${hop_ratio}_${algorithm}_group${grpo_group_size}_${model_name}"
fi
TARGET_DIR="$CKPT_BASE/verifier_iter${iter}_hf"

# 최신 global_step 자동 탐지
LATEST_STEP=$(ls "$CKPT_BASE" | grep '^global_step_' | sort -t_ -k3 -n | tail -1)

if [ -z "$LATEST_STEP" ]; then
    echo "[ERROR] $CKPT_BASE 에서 체크포인트를 찾을 수 없습니다."
    exit 1
fi

ACTOR_DIR="$CKPT_BASE/$LATEST_STEP/actor"
echo "[INFO] iter=${iter}, step=${LATEST_STEP}, verifier actor=${ACTOR_DIR}"
echo "[INFO] target=${TARGET_DIR}"

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$ACTOR_DIR" \
    --target_dir "$TARGET_DIR"

echo "[INFO] 변환 완료: $TARGET_DIR"
