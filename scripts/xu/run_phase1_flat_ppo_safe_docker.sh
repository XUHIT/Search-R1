#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/xu/xu_exp/Search-R1}
DATA_ROOT=${DATA_ROOT:-/mnt/xu/xu_data/Search-R1}
TRAIN_IMAGE=${TRAIN_IMAGE:-searchr1:cu121-vllm063-flashattn}
RETRIEVER_IMAGE=${RETRIEVER_IMAGE:-searchr1-retriever:cu121-faiss18}
WANDB_ENV_FILE=${WANDB_ENV_FILE:-${PROJECT_ROOT}/.secrets/wandb.env}
MODEL_PATH=${MODEL_PATH:-${DATA_ROOT}/models/Qwen2.5-3B}

RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
TOTAL_STEPS=${TOTAL_STEPS:-3}
MAX_TURNS=${MAX_TURNS:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}
PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}
LOG_PROB_MICRO_BATCH_SIZE=${LOG_PROB_MICRO_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}
REF_LOG_PROB_MICRO_BATCH_SIZE=${REF_LOG_PROB_MICRO_BATCH_SIZE:-${LOG_PROB_MICRO_BATCH_SIZE}}
CRITIC_PPO_MINI_BATCH_SIZE=${CRITIC_PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}
CRITIC_PPO_MICRO_BATCH_SIZE=${CRITIC_PPO_MICRO_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}
ACTOR_LR_WARMUP_RATIO=${ACTOR_LR_WARMUP_RATIO:-0.0}
CRITIC_LR_WARMUP_RATIO=${CRITIC_LR_WARMUP_RATIO:-0.0}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}
ALGORITHM_ADV_ESTIMATOR=${ALGORITHM_ADV_ESTIMATOR:-gae}
ROLLOUT_N_AGENT=${ROLLOUT_N_AGENT:-1}
ACTOR_USE_KL_LOSS=${ACTOR_USE_KL_LOSS:-false}
ACTOR_KL_LOSS_COEF=${ACTOR_KL_LOSS_COEF:-0.001}
ACTOR_KL_LOSS_TYPE=${ACTOR_KL_LOSS_TYPE:-low_var_kl}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-64}
MAX_OBS_LENGTH=${MAX_OBS_LENGTH:-512}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1536}
MAX_START_LENGTH=${MAX_START_LENGTH:-768}
VAL_DATA_NUM=${VAL_DATA_NUM:-${VAL_BATCH_SIZE}}
VAL_SOURCE_DATA=${VAL_SOURCE_DATA:-}
VAL_DATA_SOURCE_FILTER=${VAL_DATA_SOURCE_FILTER:-}
PHASE_DATA_DIR=${PHASE_DATA_DIR:-}
FSDP_OFFLOAD=${FSDP_OFFLOAD:-false}
TRAINER_LOGGER=${TRAINER_LOGGER:-"['console','wandb']"}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_PROJECT=${WANDB_PROJECT:-search-r1}
VAL_ONLY=${VAL_ONLY:-false}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-false}
SAVE_FREQ=${SAVE_FREQ:--1}
TEST_FREQ=${TEST_FREQ:-10}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-phase1-flat-ppo-qwen2.5-3b-${TOTAL_STEPS}step-${MAX_TURNS}turn-safe-${RUN_ID}}
CONTAINER_MODEL_PATH=${CONTAINER_MODEL_PATH:-${MODEL_PATH}}
if [[ "${MODEL_PATH}" == "${DATA_ROOT}/"* ]]; then
  CONTAINER_MODEL_PATH="/data/Search-R1/${MODEL_PATH#"${DATA_ROOT}/"}"
elif [[ "${MODEL_PATH}" == "${PROJECT_ROOT}/"* ]]; then
  CONTAINER_MODEL_PATH="/workspace/Search-R1/${MODEL_PATH#"${PROJECT_ROOT}/"}"
fi

LOG_DIR=${LOG_DIR:-${PROJECT_ROOT}/logs}
TRAIN_CONTAINER=${TRAIN_CONTAINER:-${EXPERIMENT_NAME}}
RETRIEVER_CONTAINER=${RETRIEVER_CONTAINER:-searchr1-retriever-flat-safe}
RETRIEVER_LOG=${RETRIEVER_LOG:-${LOG_DIR}/retriever_flat_safe_${RUN_ID}.log}
TRAIN_OUTER_LOG=${TRAIN_OUTER_LOG:-${LOG_DIR}/${EXPERIMENT_NAME}.outer.log}
RAY_LOG_COPY_DIR=${RAY_LOG_COPY_DIR:-${LOG_DIR}/ray_${EXPERIMENT_NAME}}
RAY_TMPFS_SIZE=${RAY_TMPFS_SIZE:-32g}
WANDB_SYNC_AFTER=${WANDB_SYNC_AFTER:-1}
CLEANUP_RETRIEVER=${CLEANUP_RETRIEVER:-1}
ALLOW_EXISTING_RETRIEVER=${ALLOW_EXISTING_RETRIEVER:-0}
RETRIEVER_READY_TIMEOUT_SECONDS=${RETRIEVER_READY_TIMEOUT_SECONDS:-600}
RETRIEVER_READY_INTERVAL_SECONDS=${RETRIEVER_READY_INTERVAL_SECONDS:-5}
CHECK_ONLY=${CHECK_ONLY:-0}
STARTED_RETRIEVER=0

MAX_IOWAIT_PERCENT=${MAX_IOWAIT_PERCENT:-20}
MAX_BLOCKED_PROCS=${MAX_BLOCKED_PROCS:-40}
MAX_SWAP_OUT_KB=${MAX_SWAP_OUT_KB:-0}
MAX_GPU_USED_MIB=${MAX_GPU_USED_MIB:-128}
MIN_MNT_FREE_GIB=${MIN_MNT_FREE_GIB:-2048}
NICE_LEVEL=${NICE_LEVEL:-10}
IONICE_LEVEL=${IONICE_LEVEL:-7}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "missing required command: $1" >&2
    exit 2
  }
}

ceil_to_int() {
  awk -v value="$1" 'BEGIN { printf("%d", value == int(value) ? value : int(value) + 1) }'
}

check_resources() {
  require_cmd docker
  require_cmd awk
  require_cmd curl
  require_cmd vmstat
  require_cmd df
  require_cmd nvidia-smi

  mkdir -p "${LOG_DIR}"
  test -d "${PROJECT_ROOT}"
  test -d "${DATA_ROOT}"

  local mnt_free_gib
  mnt_free_gib=$(df -BG /mnt | awk 'NR==2 { gsub("G", "", $4); print $4 }')
  if (( mnt_free_gib < MIN_MNT_FREE_GIB )); then
    echo "refuse_to_start: /mnt free ${mnt_free_gib}GiB < ${MIN_MNT_FREE_GIB}GiB" >&2
    exit 3
  fi

  local vm_line blocked iowait swap_out
  vm_line=$(vmstat 1 4 | tail -n 1)
  blocked=$(awk '{ print $2 }' <<<"${vm_line}")
  swap_out=$(awk '{ print $8 }' <<<"${vm_line}")
  iowait=$(awk '{ print $(NF-1) }' <<<"${vm_line}")
  if (( blocked > MAX_BLOCKED_PROCS )); then
    echo "refuse_to_start: blocked processes ${blocked} > ${MAX_BLOCKED_PROCS}" >&2
    exit 3
  fi
  if (( swap_out > MAX_SWAP_OUT_KB )); then
    echo "refuse_to_start: swap out ${swap_out} KiB/s > ${MAX_SWAP_OUT_KB}" >&2
    exit 3
  fi
  if (( $(ceil_to_int "${iowait}") > MAX_IOWAIT_PERCENT )); then
    echo "refuse_to_start: iowait ${iowait}% > ${MAX_IOWAIT_PERCENT}%" >&2
    exit 3
  fi

  local gpu_used_max
  gpu_used_max=$(
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
      awk 'BEGIN { max = 0 } { if ($1 > max) max = $1 } END { print max }'
  )
  if (( gpu_used_max > MAX_GPU_USED_MIB )); then
    echo "refuse_to_start: some GPU already uses ${gpu_used_max}MiB > ${MAX_GPU_USED_MIB}MiB" >&2
    exit 3
  fi

  if docker ps --format '{{.Names}}' | grep -Eq '^(searchr1-|phase[0-9]+-|eval-)'; then
    echo "refuse_to_start: existing Search-R1 container is running" >&2
    docker ps --format '  {{.Names}} {{.Status}} {{.Image}}' | grep -E 'searchr1|phase[0-9]+|eval-' >&2 || true
    exit 3
  fi

  echo "resource_check_ok: /mnt_free=${mnt_free_gib}GiB blocked=${blocked} iowait=${iowait}% swap_out=${swap_out}KiB/s gpu_used_max=${gpu_used_max}MiB"
}

start_retriever() {
  if curl -fsS -X POST http://127.0.0.1:8000/retrieve \
    -H 'Content-Type: application/json' \
    -d '{"queries":["where is the Eiffel Tower located"],"topk":1,"return_scores":true}' \
    >/dev/null 2>&1; then
    if [[ "${ALLOW_EXISTING_RETRIEVER}" != "1" ]]; then
      echo "refuse_to_start: port 8000 already has a retriever-like service; set ALLOW_EXISTING_RETRIEVER=1 to reuse it" >&2
      exit 3
    fi
    echo "using existing retriever on 127.0.0.1:8000"
    return 0
  fi

  docker rm -f "${RETRIEVER_CONTAINER}" >/dev/null 2>&1 || true
  docker run -d --name "${RETRIEVER_CONTAINER}" \
    --gpus all \
    --network host \
    --ipc=host \
    --shm-size=64g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -v "${PROJECT_ROOT}:/workspace/Search-R1" \
    -v "${DATA_ROOT}:/data/Search-R1" \
    -w /workspace/Search-R1 \
    "${RETRIEVER_IMAGE}" \
    bash /workspace/Search-R1/scripts/xu/start_retriever_flat_gpu.sh \
    >"${RETRIEVER_LOG}" 2>&1
  STARTED_RETRIEVER=1

  local attempts
  attempts=$((RETRIEVER_READY_TIMEOUT_SECONDS / RETRIEVER_READY_INTERVAL_SECONDS))
  if (( attempts < 1 )); then
    attempts=1
  fi
  for _ in $(seq 1 "${attempts}"); do
    if curl -fsS -X POST http://127.0.0.1:8000/retrieve \
      -H 'Content-Type: application/json' \
      -d '{"queries":["where is the Eiffel Tower located"],"topk":1,"return_scores":true}' \
      >/dev/null 2>&1; then
      echo "retriever ready: ${RETRIEVER_CONTAINER}"
      return 0
    fi
    sleep "${RETRIEVER_READY_INTERVAL_SECONDS}"
  done

  echo "retriever did not become ready within ${RETRIEVER_READY_TIMEOUT_SECONDS}s; see ${RETRIEVER_LOG}" >&2
  docker logs "${RETRIEVER_CONTAINER}" >&2 || true
  exit 4
}

cleanup() {
  if [[ "${STARTED_RETRIEVER}" == "1" && "${CLEANUP_RETRIEVER}" == "1" ]]; then
    docker rm -f "${RETRIEVER_CONTAINER}" >/dev/null 2>&1 || true
  fi
}

run_training() {
  local env_file_args=()
  if [[ "${WANDB_MODE}" == "online" && "${TRAINER_LOGGER}" == *wandb* ]]; then
    if [[ ! -r "${WANDB_ENV_FILE}" ]]; then
      echo "refuse_to_start: WANDB env file is not readable: ${WANDB_ENV_FILE}" >&2
      exit 2
    fi
    env_file_args=(--env-file "${WANDB_ENV_FILE}")
  fi

  docker rm -f "${TRAIN_CONTAINER}" >/dev/null 2>&1 || true

  set +e
  docker run --name "${TRAIN_CONTAINER}" \
    --gpus all \
    --network host \
    --ipc=host \
    --shm-size=64g \
    --tmpfs "/tmp/ray:rw,nosuid,nodev,size=${RAY_TMPFS_SIZE},mode=1777" \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    "${env_file_args[@]}" \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -e VLLM_ATTENTION_BACKEND=XFORMERS \
    -e WANDB_MODE="${WANDB_MODE}" \
    -e WANDB_PROJECT="${WANDB_PROJECT}" \
    -e TRAINER_LOGGER="${TRAINER_LOGGER}" \
    -e HF_HOME=/data/Search-R1/hf_cache \
    -e TRANSFORMERS_CACHE=/data/Search-R1/hf_cache/transformers \
    -e HF_DATASETS_CACHE=/data/Search-R1/hf_cache/datasets \
    -e TOKENIZERS_PARALLELISM=true \
    -e RETRIEVER_URL=http://127.0.0.1:8000/retrieve \
    -e MODEL_PATH="${CONTAINER_MODEL_PATH}" \
    -e PHASE_DATA_DIR="${PHASE_DATA_DIR}" \
    -e VAL_SOURCE_DATA="${VAL_SOURCE_DATA}" \
    -e VAL_DATA_SOURCE_FILTER="${VAL_DATA_SOURCE_FILTER}" \
    -e VAL_DATA_NUM="${VAL_DATA_NUM}" \
    -e VAL_ONLY="${VAL_ONLY}" \
    -e VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN}" \
    -e SAVE_FREQ="${SAVE_FREQ}" \
    -e TEST_FREQ="${TEST_FREQ}" \
    -e EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
    -e TOTAL_STEPS="${TOTAL_STEPS}" \
    -e MAX_TURNS="${MAX_TURNS}" \
    -e TRAIN_GPUS=8 \
    -e TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
    -e VAL_BATCH_SIZE="${VAL_BATCH_SIZE}" \
    -e PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE}" \
    -e PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE}" \
    -e LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE}" \
    -e REF_LOG_PROB_MICRO_BATCH_SIZE="${REF_LOG_PROB_MICRO_BATCH_SIZE}" \
    -e CRITIC_PPO_MINI_BATCH_SIZE="${CRITIC_PPO_MINI_BATCH_SIZE}" \
    -e CRITIC_PPO_MICRO_BATCH_SIZE="${CRITIC_PPO_MICRO_BATCH_SIZE}" \
    -e ACTOR_LR_WARMUP_RATIO="${ACTOR_LR_WARMUP_RATIO}" \
    -e CRITIC_LR_WARMUP_RATIO="${CRITIC_LR_WARMUP_RATIO}" \
    -e ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    -e ALGORITHM_ADV_ESTIMATOR="${ALGORITHM_ADV_ESTIMATOR}" \
    -e ROLLOUT_N_AGENT="${ROLLOUT_N_AGENT}" \
    -e ACTOR_USE_KL_LOSS="${ACTOR_USE_KL_LOSS}" \
    -e ACTOR_KL_LOSS_COEF="${ACTOR_KL_LOSS_COEF}" \
    -e ACTOR_KL_LOSS_TYPE="${ACTOR_KL_LOSS_TYPE}" \
    -e MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH}" \
    -e MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH}" \
    -e MAX_START_LENGTH="${MAX_START_LENGTH}" \
    -e MAX_OBS_LENGTH="${MAX_OBS_LENGTH}" \
    -e ACTOR_PARAM_OFFLOAD="${FSDP_OFFLOAD}" \
    -e ACTOR_GRAD_OFFLOAD="${FSDP_OFFLOAD}" \
    -e ACTOR_OPTIMIZER_OFFLOAD="${FSDP_OFFLOAD}" \
    -e REF_PARAM_OFFLOAD="${FSDP_OFFLOAD}" \
    -e CRITIC_PARAM_OFFLOAD="${FSDP_OFFLOAD}" \
    -e CRITIC_GRAD_OFFLOAD="${FSDP_OFFLOAD}" \
    -e CRITIC_OPTIMIZER_OFFLOAD="${FSDP_OFFLOAD}" \
    -v "${PROJECT_ROOT}:/workspace/Search-R1" \
    -v "${DATA_ROOT}:/data/Search-R1" \
    -w /workspace/Search-R1 \
    "${TRAIN_IMAGE}" \
    bash -lc "exec nice -n ${NICE_LEVEL} ionice -c2 -n${IONICE_LEVEL} bash /workspace/Search-R1/scripts/xu/train_phase1_flat_ppo_wandb.sh" \
    2>&1 | tee "${TRAIN_OUTER_LOG}"
  local rc=${PIPESTATUS[0]}
  set -e

  if [[ "${WANDB_SYNC_AFTER}" == "1" && "${WANDB_MODE}" == "online" && "${TRAINER_LOGGER}" == *wandb* ]]; then
    docker run --rm \
      --network host \
      "${env_file_args[@]}" \
      -v "${PROJECT_ROOT}:/workspace/Search-R1" \
      -w /workspace/Search-R1 \
      "${TRAIN_IMAGE}" \
      bash -lc 'latest=$(ls -td wandb/run-* 2>/dev/null | head -1 || true); if [[ -n "${latest}" ]]; then echo "wandb_sync=${latest}"; wandb sync "${latest}" || true; fi'
  fi

  rm -rf "${RAY_LOG_COPY_DIR}"
  mkdir -p "${RAY_LOG_COPY_DIR}"
  docker cp "${TRAIN_CONTAINER}:/tmp/ray/." "${RAY_LOG_COPY_DIR}/" >/dev/null 2>&1 || true
  echo "ray_logs=${RAY_LOG_COPY_DIR}"
  echo "train_log=${TRAIN_OUTER_LOG}"
  return "${rc}"
}

main() {
  trap cleanup EXIT
  check_resources
  if [[ "${CHECK_ONLY}" == "1" ]]; then
    exit 0
  fi
  start_retriever
  run_training
}

main "$@"
