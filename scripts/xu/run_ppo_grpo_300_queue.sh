#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/xu/xu_exp/Search-R1}
DATA_ROOT=${DATA_ROOT:-/mnt/xu/xu_data/Search-R1}
LOG_DIR=${LOG_DIR:-${PROJECT_ROOT}/logs}
RUN_GROUP=${RUN_GROUP:-$(date +%Y%m%d_%H%M%S)}
QUEUE_FSDP_OFFLOAD=${QUEUE_FSDP_OFFLOAD:-true}

mkdir -p "${LOG_DIR}"

sample_gpu_until_exit() {
  local exp="$1"
  local out="${LOG_DIR}/gpu_peak_${exp}.log"
  local seen=0

  {
    echo "# exp=${exp} run_group=${RUN_GROUP} started=$(date -Is)"
    while true; do
      if docker ps -a --format '{{.Names}} {{.Status}}' | grep -q "${exp}"; then
        seen=1
      fi
      if [[ "${seen}" == "1" ]]; then
        date -Is
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
      fi
      if [[ "${seen}" == "1" ]] && docker ps -a --format '{{.Names}} {{.Status}}' | grep "${exp}" | grep -Eq 'Exited|Dead'; then
        break
      fi
      sleep 5
    done
    echo "# ended=$(date -Is)"
  } >"${out}" 2>&1
}

run_one() {
  local method="$1"
  local exp="$2"
  shift 2

  echo "queue_start method=${method} exp=${exp} time=$(date -Is)"
  sample_gpu_until_exit "${exp}" &
  local sampler_pid=$!

  set +e
  env \
    RUN_ID="${RUN_GROUP}_${method}" \
    EXPERIMENT_NAME="${exp}" \
    MODEL_PATH="${DATA_ROOT}/models/Qwen2.5-3B-Instruct" \
    TOTAL_STEPS=300 \
    MAX_TURNS=3 \
    TRAIN_BATCH_SIZE=512 \
    VAL_BATCH_SIZE=256 \
    MAX_RESPONSE_LENGTH=500 \
    MAX_OBS_LENGTH=500 \
    MAX_PROMPT_LENGTH=4096 \
    MAX_START_LENGTH=2048 \
    FSDP_OFFLOAD="${QUEUE_FSDP_OFFLOAD}" \
    PPO_MINI_BATCH_SIZE=256 \
    PPO_MICRO_BATCH_SIZE=32 \
    LOG_PROB_MICRO_BATCH_SIZE=64 \
    REF_LOG_PROB_MICRO_BATCH_SIZE=64 \
    CRITIC_PPO_MINI_BATCH_SIZE=256 \
    CRITIC_PPO_MICRO_BATCH_SIZE=8 \
    ACTOR_LR_WARMUP_RATIO=0.285 \
    CRITIC_LR_WARMUP_RATIO=0.015 \
    ROLLOUT_GPU_MEMORY_UTILIZATION=0.35 \
    VAL_BEFORE_TRAIN=true \
    VAL_SOURCE_DATA=/data/Search-R1/datasets/nq_hotpotqa_train/test.parquet \
    VAL_DATA_SOURCE_FILTER=nq \
    VAL_DATA_NUM=256 \
    SAVE_FREQ=100 \
    TEST_FREQ=100 \
    RAY_TMPFS_SIZE=32g \
    MAX_IOWAIT_PERCENT=30 \
    MAX_BLOCKED_PROCS=80 \
    "$@" \
    "${PROJECT_ROOT}/scripts/xu/run_phase1_flat_ppo_safe_docker.sh"
  local rc=$?
  set -e

  sleep 10
  if kill -0 "${sampler_pid}" >/dev/null 2>&1; then
    kill "${sampler_pid}" >/dev/null 2>&1 || true
  fi
  wait "${sampler_pid}" >/dev/null 2>&1 || true

  echo "queue_done method=${method} exp=${exp} rc=${rc} time=$(date -Is)"
  return "${rc}"
}

main() {
  cd "${PROJECT_ROOT}"

  local offload_tag
  if [[ "${QUEUE_FSDP_OFFLOAD}" == "true" ]]; then
    offload_tag="offtrue"
  else
    offload_tag="offfalse"
  fi

  local ppo_exp="phase6-ppo-qwen2.5-3b-instruct-300step-nq256-${offload_tag}-safe-${RUN_GROUP}"
  local grpo_exp="phase6-grpo-qwen2.5-3b-instruct-300step-n3-nq256-${offload_tag}-safe-${RUN_GROUP}"

  echo "queue_group=${RUN_GROUP}"
  echo "queue_fsdp_offload=${QUEUE_FSDP_OFFLOAD}"
  echo "ppo_exp=${ppo_exp}"
  echo "grpo_exp=${grpo_exp}"

  run_one ppo "${ppo_exp}" \
    ALGORITHM_ADV_ESTIMATOR=gae \
    ROLLOUT_N_AGENT=1 \
    ACTOR_USE_KL_LOSS=false

  run_one grpo "${grpo_exp}" \
    ALGORITHM_ADV_ESTIMATOR=grpo \
    ROLLOUT_N_AGENT=3 \
    ACTOR_USE_KL_LOSS=true \
    ACTOR_KL_LOSS_COEF=0.001 \
    ACTOR_KL_LOSS_TYPE=low_var_kl
}

main "$@"
