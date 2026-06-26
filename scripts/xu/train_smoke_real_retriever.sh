#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export WANDB_MODE=${WANDB_MODE:-offline}
export HF_HOME=${HF_HOME:-/data/Search-R1/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/data/Search-R1/hf_cache/transformers}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/data/Search-R1/hf_cache/datasets}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}

WORK_DIR=${WORK_DIR:-/workspace/Search-R1}
DATA_ROOT=${DATA_ROOT:-/data/Search-R1}
SOURCE_DATA=${SOURCE_DATA:-${DATA_ROOT}/datasets/nq_hotpotqa_train/train.parquet}
SMOKE_DATA_DIR=${SMOKE_DATA_DIR:-${WORK_DIR}/data/smoke_train_real_retriever}
LOG_DIR=${LOG_DIR:-${WORK_DIR}/logs}
MODEL_PATH=${MODEL_PATH:-${DATA_ROOT}/models/Qwen2.5-3B}
RETRIEVER_URL=${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}
TRAIN_GPUS=${TRAIN_GPUS:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-${TRAIN_GPUS}}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}
TOTAL_STEPS=${TOTAL_STEPS:-1}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-smoke-qwen2.5-3b-real-retriever-${TOTAL_STEPS}step}
export WORK_DIR DATA_ROOT SOURCE_DATA SMOKE_DATA_DIR LOG_DIR MODEL_PATH RETRIEVER_URL
export TRAIN_GPUS TRAIN_BATCH_SIZE VAL_BATCH_SIZE TOTAL_STEPS EXPERIMENT_NAME

mkdir -p "${SMOKE_DATA_DIR}" "${LOG_DIR}" "${WORK_DIR}/verl_checkpoints/${EXPERIMENT_NAME}"
test -d "${MODEL_PATH}"

python - <<'PY'
import os
import pandas as pd

src = os.environ["SOURCE_DATA"]
out = os.environ["SMOKE_DATA_DIR"]
train_batch = int(os.environ["TRAIN_BATCH_SIZE"])
rows = max(train_batch, 8)
os.makedirs(out, exist_ok=True)

df = pd.read_parquet(src)
parts = []
for name in ["nq", "hotpotqa"]:
    sub = df[df["data_source"] == name].head(max(rows // 2, 1))
    if len(sub):
        parts.append(sub)
sample = pd.concat(parts, ignore_index=True).head(rows)
if len(sample) < rows:
    sample = df.head(rows)

sample.to_parquet(os.path.join(out, "train.parquet"))
sample.to_parquet(os.path.join(out, "test.parquet"))
print(f"real retriever smoke rows: {len(sample)}")
print(sample["data_source"].value_counts().to_string())
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
  data.train_files="${SMOKE_DATA_DIR}/train.parquet" \
  data.val_files="${SMOKE_DATA_DIR}/test.parquet" \
  data.train_data_num=null \
  data.val_data_num="${VAL_BATCH_SIZE}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.max_prompt_length=1024 \
  data.max_response_length=64 \
  data.max_start_length=512 \
  data.max_obs_length=128 \
  data.shuffle_train_dataloader=False \
  algorithm.adv_estimator=gae \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size="${TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.grad_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.rollout.log_prob_micro_batch_size="${TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.rollout.n_agent=1 \
  actor_rollout_ref.rollout.temperature=1 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.actor.state_masking=true \
  actor_rollout_ref.ref.log_prob_micro_batch_size="${TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  critic.optim.lr=1e-5 \
  critic.model.path="${MODEL_PATH}" \
  critic.model.use_remove_padding=True \
  critic.model.enable_gradient_checkpointing=true \
  critic.optim.lr_warmup_steps_ratio=0.0 \
  critic.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
  critic.ppo_micro_batch_size="${TRAIN_BATCH_SIZE}" \
  critic.model.fsdp_config.param_offload=true \
  critic.model.fsdp_config.grad_offload=true \
  critic.model.fsdp_config.optimizer_offload=true \
  algorithm.kl_ctrl.kl_coef=0.001 \
  algorithm.no_think_rl=false \
  trainer.critic_warmup=0 \
  trainer.logger=['console'] \
  +trainer.val_only=false \
  +trainer.val_before_train=false \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node="${TRAIN_GPUS}" \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.project_name=Search-R1 \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.total_epochs=1 \
  trainer.total_training_steps="${TOTAL_STEPS}" \
  trainer.default_local_dir="${WORK_DIR}/verl_checkpoints/${EXPERIMENT_NAME}" \
  max_turns=1 \
  do_search=true \
  retriever.url="${RETRIEVER_URL}" \
  retriever.topk=3
