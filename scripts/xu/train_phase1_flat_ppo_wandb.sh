#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export WANDB_MODE=${WANDB_MODE:-online}
export HF_HOME=${HF_HOME:-/data/Search-R1/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/data/Search-R1/hf_cache/transformers}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/data/Search-R1/hf_cache/datasets}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}

WORK_DIR=${WORK_DIR:-/workspace/Search-R1}
DATA_ROOT=${DATA_ROOT:-/data/Search-R1}
SOURCE_DATA=${SOURCE_DATA:-${DATA_ROOT}/datasets/nq_hotpotqa_train/train.parquet}
VAL_SOURCE_DATA=${VAL_SOURCE_DATA:-}
VAL_DATA_SOURCE_FILTER=${VAL_DATA_SOURCE_FILTER:-}
PHASE_DATA_DIR=${PHASE_DATA_DIR:-${WORK_DIR}/data/phase1_flat_ppo_wandb}
LOG_DIR=${LOG_DIR:-${WORK_DIR}/logs}
MODEL_PATH=${MODEL_PATH:-${DATA_ROOT}/models/Qwen2.5-3B}
RETRIEVER_URL=${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}
TRAIN_GPUS=${TRAIN_GPUS:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-${TRAIN_GPUS}}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}
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
VAL_DATA_NUM=${VAL_DATA_NUM:-${VAL_BATCH_SIZE}}
TOTAL_STEPS=${TOTAL_STEPS:-20}
MAX_TURNS=${MAX_TURNS:-2}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1536}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-128}
MAX_START_LENGTH=${MAX_START_LENGTH:-768}
MAX_OBS_LENGTH=${MAX_OBS_LENGTH:-256}
TRAINER_LOGGER=${TRAINER_LOGGER:-"['console','wandb']"}
VAL_ONLY=${VAL_ONLY:-false}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-false}
SAVE_FREQ=${SAVE_FREQ:--1}
TEST_FREQ=${TEST_FREQ:-10}
WAND_PROJECT=${WANDB_PROJECT:-search-r1}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-phase1-flat-ppo-qwen2.5-3b-${TOTAL_STEPS}step-${MAX_TURNS}turn}
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-true}
ACTOR_GRAD_OFFLOAD=${ACTOR_GRAD_OFFLOAD:-true}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-true}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-true}
CRITIC_PARAM_OFFLOAD=${CRITIC_PARAM_OFFLOAD:-true}
CRITIC_GRAD_OFFLOAD=${CRITIC_GRAD_OFFLOAD:-true}
CRITIC_OPTIMIZER_OFFLOAD=${CRITIC_OPTIMIZER_OFFLOAD:-true}
export WORK_DIR DATA_ROOT SOURCE_DATA VAL_SOURCE_DATA VAL_DATA_SOURCE_FILTER PHASE_DATA_DIR LOG_DIR MODEL_PATH RETRIEVER_URL
export TRAIN_GPUS TRAIN_BATCH_SIZE VAL_BATCH_SIZE VAL_DATA_NUM TOTAL_STEPS MAX_TURNS
export PPO_MINI_BATCH_SIZE PPO_MICRO_BATCH_SIZE LOG_PROB_MICRO_BATCH_SIZE
export REF_LOG_PROB_MICRO_BATCH_SIZE CRITIC_PPO_MINI_BATCH_SIZE CRITIC_PPO_MICRO_BATCH_SIZE
export ACTOR_LR_WARMUP_RATIO CRITIC_LR_WARMUP_RATIO ROLLOUT_GPU_MEMORY_UTILIZATION
export ALGORITHM_ADV_ESTIMATOR ROLLOUT_N_AGENT ACTOR_USE_KL_LOSS ACTOR_KL_LOSS_COEF ACTOR_KL_LOSS_TYPE
export MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH MAX_START_LENGTH MAX_OBS_LENGTH
export VAL_ONLY VAL_BEFORE_TRAIN SAVE_FREQ TEST_FREQ
export WAND_PROJECT EXPERIMENT_NAME
export ACTOR_PARAM_OFFLOAD ACTOR_GRAD_OFFLOAD ACTOR_OPTIMIZER_OFFLOAD
export REF_PARAM_OFFLOAD CRITIC_PARAM_OFFLOAD CRITIC_GRAD_OFFLOAD CRITIC_OPTIMIZER_OFFLOAD

mkdir -p "${PHASE_DATA_DIR}" "${LOG_DIR}" "${WORK_DIR}/verl_checkpoints/${EXPERIMENT_NAME}"
test -d "${MODEL_PATH}"

if [[ "${WANDB_MODE}" == "online" && "${TRAINER_LOGGER}" == *"wandb"* && -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is required for online wandb logging." >&2
  exit 2
fi

python - <<'PY'
import os
import pandas as pd

src = os.environ["SOURCE_DATA"]
out = os.environ["PHASE_DATA_DIR"]
train_batch = int(os.environ["TRAIN_BATCH_SIZE"])
val_batch = int(os.environ["VAL_BATCH_SIZE"])
total_steps = int(os.environ["TOTAL_STEPS"])
val_data_num = os.environ["VAL_DATA_NUM"]
train_rows = max(train_batch * total_steps, train_batch, 8)
val_rows = None if val_data_num == "null" else max(int(val_data_num), val_batch, 8)
os.makedirs(out, exist_ok=True)

df = pd.read_parquet(src)
train_parts = []
val_parts = []
for name in ["nq", "hotpotqa"]:
    sub = df[df["data_source"] == name]
    if len(sub):
        split = max(train_rows // 2, 1)
        train_parts.append(sub.head(split))
        if val_rows is not None:
            val_parts.append(sub.iloc[split:split + max(val_rows // 2, 1)])

train = pd.concat(train_parts, ignore_index=True).head(train_rows) if train_parts else df.head(train_rows)
if len(train) < train_rows:
    train = df.head(train_rows)

val_source = os.environ.get("VAL_SOURCE_DATA") or ""
val_filter = os.environ.get("VAL_DATA_SOURCE_FILTER") or ""
if val_source:
    val_df = pd.read_parquet(val_source)
    if val_filter:
        val_df = val_df[val_df["data_source"] == val_filter]
    val = val_df if val_rows is None else val_df.head(val_rows)
else:
    val = pd.concat(val_parts, ignore_index=True).head(val_rows) if val_parts else df.iloc[train_rows:train_rows + val_rows]
    if len(val) < val_rows:
        val = df.tail(val_rows)

train.to_parquet(os.path.join(out, "train.parquet"))
val.to_parquet(os.path.join(out, "test.parquet"))
print(f"phase1 train rows: {len(train)}")
print(train["data_source"].value_counts().to_string())
print(f"phase1 val rows: {len(val)}")
print(val["data_source"].value_counts().to_string())
PY

python - <<'PY'
import os
import requests

url = os.environ["RETRIEVER_URL"]
resp = requests.post(url, json={"queries": ["where is the Eiffel Tower located"], "topk": 3, "return_scores": True}, timeout=60)
resp.raise_for_status()
body = resp.json()
doc = body["result"][0][0]["document"]["contents"]
print("retriever_ok", resp.status_code, doc[:120].replace("\n", " "))
PY

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
  data.train_files="${PHASE_DATA_DIR}/train.parquet" \
  data.val_files="${PHASE_DATA_DIR}/test.parquet" \
  data.train_data_num=null \
  data.val_data_num="${VAL_DATA_NUM}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.max_start_length="${MAX_START_LENGTH}" \
  data.max_obs_length="${MAX_OBS_LENGTH}" \
  data.shuffle_train_dataloader=True \
  algorithm.adv_estimator="${ALGORITHM_ADV_ESTIMATOR}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="${ACTOR_LR_WARMUP_RATIO}" \
  actor_rollout_ref.actor.use_kl_loss="${ACTOR_USE_KL_LOSS}" \
  actor_rollout_ref.actor.kl_loss_coef="${ACTOR_KL_LOSS_COEF}" \
  actor_rollout_ref.actor.kl_loss_type="${ACTOR_KL_LOSS_TYPE}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size="${PPO_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}" \
  actor_rollout_ref.actor.fsdp_config.grad_offload="${ACTOR_GRAD_OFFLOAD}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size="${LOG_PROB_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.n_agent="${ROLLOUT_N_AGENT}" \
  actor_rollout_ref.rollout.temperature=1 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.actor.state_masking=true \
  actor_rollout_ref.ref.log_prob_micro_batch_size="${REF_LOG_PROB_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.ref.fsdp_config.param_offload="${REF_PARAM_OFFLOAD}" \
  critic.optim.lr=1e-5 \
  critic.model.path="${MODEL_PATH}" \
  critic.model.use_remove_padding=True \
  critic.model.enable_gradient_checkpointing=true \
  critic.optim.lr_warmup_steps_ratio="${CRITIC_LR_WARMUP_RATIO}" \
  critic.ppo_mini_batch_size="${CRITIC_PPO_MINI_BATCH_SIZE}" \
  critic.ppo_micro_batch_size="${CRITIC_PPO_MICRO_BATCH_SIZE}" \
  critic.model.fsdp_config.param_offload="${CRITIC_PARAM_OFFLOAD}" \
  critic.model.fsdp_config.grad_offload="${CRITIC_GRAD_OFFLOAD}" \
  critic.model.fsdp_config.optimizer_offload="${CRITIC_OPTIMIZER_OFFLOAD}" \
  algorithm.kl_ctrl.kl_coef=0.001 \
  algorithm.no_think_rl=false \
  trainer.critic_warmup=0 \
  trainer.logger="${TRAINER_LOGGER}" \
  +trainer.val_only="${VAL_ONLY}" \
  +trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node="${TRAIN_GPUS}" \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.project_name="${WAND_PROJECT}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.total_epochs=1 \
  trainer.total_training_steps="${TOTAL_STEPS}" \
  trainer.default_local_dir="${WORK_DIR}/verl_checkpoints/${EXPERIMENT_NAME}" \
  max_turns="${MAX_TURNS}" \
  do_search=true \
  retriever.url="${RETRIEVER_URL}" \
  retriever.topk=3 \
  2>&1 | tee "${LOG_DIR}/${EXPERIMENT_NAME}.log"
