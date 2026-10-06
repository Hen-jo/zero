#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
set -euo pipefail

# Required: a complete local HF export, not an FSDP global_step directory.
# Iteration 1 normally uses the converted challenger; later iterations use the
# preceding verifier's converted export. No checkpoint location is guessed.
: "${VERIFIER_INIT_MODEL:?Set VERIFIER_INIT_MODEL to an explicitly selected, converted HF checkpoint directory}"
cur_iter=1
hop_ratio=${1:-4321}
if (( $# )); then shift; fi
[[ "$hop_ratio" =~ ^[0-9]+$ ]] || { echo "Invalid hop ratio: $hop_ratio" >&2; exit 1; }

# The base-model name identifies the existing challenger/solver artifacts.
# It is deliberately independent of the selected warm-start export directory.
model=${BASE_MODEL:-/input/model/Qwen/Qwen2.5-3B-Instruct}
model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
algorithm=grpo
grpo_group_size=5
tp=2
gpus=8
batch_per_gpu=2
rollout_memory_utilization=0.5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/tmp/hf_cache}
export HF_HOME=${HF_HOME:-/tmp/hf_home}
OUTPUT_DIR=${OUTPUT_DIR:-/input/output/revzero}
EXP_DIR="$OUTPUT_DIR/checkpoints/dr-zero"
CONFIG_PATH="$SCRIPT_DIR/config"
TOOL_CONFIG="$CONFIG_PATH/search_tool_config.yaml"

CHALLENGER_NAME="challenger_iter${cur_iter}_ratio${hop_ratio}_grpo_batch_group1-5_${model_name}"
SOLVER_NAME="solver_iter${cur_iter}_ratio${hop_ratio}_grpo_group5_${model_name}"
VERIFIER_NAME="verifier_iter${cur_iter}_ratio${hop_ratio}_${algorithm}_group${grpo_group_size}_${model_name}"
GEN_DATA_PATH=${GEN_DATA_PATH:-"$OUTPUT_DIR/data/zero_${CHALLENGER_NAME}.parquet"}
SOLVER_PRED_PATH=${SOLVER_PRED_PATH:-"$OUTPUT_DIR/data/solver_pred_${SOLVER_NAME}.parquet"}
VERIFIER_DATA_DIR=${VERIFIER_DATA_DIR:-"$OUTPUT_DIR/data/verifier_${CHALLENGER_NAME}"}
TRAIN_DATA="$VERIFIER_DATA_DIR/train.parquet"
# Only source-group-held-out verifier examples have compatible binary labels.
VAL_DATA="$VERIFIER_DATA_DIR/val.parquet"
NFS_INDEX=${NFS_INDEX:-/input/data/Drzero/corpus/e5_Flat.index}
NFS_CORPUS=${NFS_CORPUS:-/input/data/Drzero/corpus/wiki-18.jsonl}
RETRIEVER_MODEL=${RETRIEVER_MODEL:-/input/model/e5-base-v2}
LOCAL_CACHE=${LOCAL_CACHE:-/tmp/drzero_corpus}

for flag in VERIFIER_ALLOW_PARTIAL_COVERAGE VERIFIER_OVERWRITE_DATA VERIFIER_PREFLIGHT_ONLY; do
    value=${!flag:-0}
    [[ "$value" == 0 || "$value" == 1 ]] || { echo "$flag must be 0 or 1" >&2; exit 1; }
done
preflight_args=()
for override in "$@"; do preflight_args+=("--trainer-override=$override"); done
# All artifact/config checks and the strict ID join happen before any server or
# GPU operation. This helper imports neither torch nor the verl package.
python -m scripts.verifier_preflight inputs \
    --repo "$SCRIPT_DIR" --init-model "$VERIFIER_INIT_MODEL" \
    --gen-data "$GEN_DATA_PATH" --solver-pred "$SOLVER_PRED_PATH" \
    --config-dir "$CONFIG_PATH" --tool-config "$TOOL_CONFIG" \
    --index "$NFS_INDEX" --corpus "$NFS_CORPUS" \
    --retriever-model "$RETRIEVER_MODEL" "${preflight_args[@]}"

builder_args=()
# Existing solver exports can contain only the final training suffix. Accepting
# less than full generated-item coverage is an explicit opt-in, never positional.
if [[ ${VERIFIER_ALLOW_PARTIAL_COVERAGE:-0} == 1 ]]; then
    builder_args+=(--allow_subset)
fi
if [[ ${VERIFIER_OVERWRITE_DATA:-0} == 1 ]]; then
    builder_args+=(--overwrite)
fi
python -m scripts.data_process.verifier_data \
    --gen_data_path "$GEN_DATA_PATH" --solver_pred_path "$SOLVER_PRED_PATH" \
    --local_dir "$VERIFIER_DATA_DIR" \
    --eval_fraction "${VERIFIER_EVAL_FRACTION:-0.1}" \
    --seed "${VERIFIER_SPLIT_SEED:-0}" "${builder_args[@]}"
python -m scripts.verifier_preflight data --train "$TRAIN_DATA" --val "$VAL_DATA"
if [[ ${VERIFIER_PREFLIGHT_ONLY:-0} == 1 ]]; then
    echo "Verifier preflight complete. No servers or training started."
    echo "Training checkpoints would be saved to: $EXP_DIR/$VERIFIER_NAME"
    exit 0
fi
python -m scripts.verifier_preflight runtime

LOG_DIR="$OUTPUT_DIR/logs/$VERIFIER_NAME"
mkdir -p "$LOG_DIR" "$LOCAL_CACHE"
export TENSORBOARD_DIR="$OUTPUT_DIR/tensorboard"
# Never kill unrelated port owners. A busy retrieval port is a preflight error.
# Only the retrieval process started by this launcher is cleaned up on exit.
RETRIEVAL_PID=
cleanup() {
    if [[ -n "$RETRIEVAL_PID" ]]; then
        kill "$RETRIEVAL_PID" 2>/dev/null || true
        wait "$RETRIEVAL_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for file in "$NFS_INDEX" "$NFS_CORPUS"; do
    name=$(basename "$file")
    if [[ ! -s "$LOCAL_CACHE/$name" ]]; then
        # An interrupted copy cannot leave a seemingly usable partial cache.
        cp -- "$file" "$LOCAL_CACHE/$name.tmp.$$"
        mv -- "$LOCAL_CACHE/$name.tmp.$$" "$LOCAL_CACHE/$name"
    fi
done
PYTHONUNBUFFERED=1 python "$SCRIPT_DIR/search/retrieval_server.py" \
    --index_path="$LOCAL_CACHE/$(basename "$NFS_INDEX")" \
    --corpus_path="$LOCAL_CACHE/$(basename "$NFS_CORPUS")" \
    --retriever_model="$RETRIEVER_MODEL" --retriever_name=e5 --topk 3 \
    > "$LOG_DIR/retrieval_server.log" 2>&1 &
RETRIEVAL_PID=$!

elapsed=0
until (echo > /dev/tcp/127.0.0.1/8000) 2>/dev/null; do
    if ! kill -0 "$RETRIEVAL_PID" 2>/dev/null; then
        cat "$LOG_DIR/retrieval_server.log" >&2
        echo "Retrieval server exited before becoming ready" >&2
        exit 1
    fi
    if (( elapsed >= 1800 )); then
        tail -30 "$LOG_DIR/retrieval_server.log" >&2
        echo "Retrieval server was not ready after 1800 seconds" >&2
        exit 1
    fi
    sleep 5
    elapsed=$((elapsed + 5))
done
kill -0 "$RETRIEVAL_PID" 2>/dev/null || { echo "Retrieval process exited" >&2; exit 1; }

# The verifier reward is local symbolic scoring. The trainer owns its rollout
# engine; no separate reward-model SGLang server is needed on port 8001.
python -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" --config-name=search_multiturn_grpo \
    "data.train_files=$TRAIN_DATA" "data.val_files=$VAL_DATA" \
    data.train_batch_size=256 data.max_prompt_length=512 \
    algorithm.use_kl_in_reward=False "algorithm.adv_estimator=$algorithm" \
    "actor_rollout_ref.model.path=$VERIFIER_INIT_MODEL" \
    actor_rollout_ref.actor.grad_clip=0.1 actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$batch_per_gpu" \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    "actor_rollout_ref.rollout.n=$grpo_group_size" actor_rollout_ref.rollout.name=sglang \
    "actor_rollout_ref.rollout.gpu_memory_utilization=$rollout_memory_utilization" \
    "actor_rollout_ref.rollout.tensor_model_parallel_size=$tp" \
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$batch_per_gpu" \
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$batch_per_gpu" \
    actor_rollout_ref.actor.use_kl_loss=False \
    "actor_rollout_ref.rollout.multi_turn.tool_config_path=$TOOL_CONFIG" \
    reward_model.reward_manager=batch \
    +reward_model.validation_with_default_compute_score=False \
    custom_reward_function.name=compute_verifier_score \
    "custom_reward_function.path=$SCRIPT_DIR/verl/custom_reward/verifier_reward_function.py" \
    'trainer.logger=["console", "tensorboard"]' trainer.project_name=dr-zero \
    "trainer.experiment_name=$VERIFIER_NAME" \
    "trainer.default_local_dir=$EXP_DIR/$VERIFIER_NAME" \
    trainer.resume_mode=disable \
    "trainer.n_gpus_per_node=$gpus" trainer.nnodes=1 \
    trainer.save_freq=25 trainer.test_freq=25 trainer.val_before_train=False \
    trainer.total_epochs=1 "$@"
