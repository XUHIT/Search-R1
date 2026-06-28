#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export WANDB_MODE=${WANDB_MODE:-offline}
export HF_HOME=${HF_HOME:-/data/Search-R1/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/data/Search-R1/hf_cache/transformers}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/data/Search-R1/hf_cache/datasets}
export TOKENIZERS_PARALLELISM=true

WORK_DIR=${WORK_DIR:-/workspace/Search-R1}
DATA_ROOT=${DATA_ROOT:-/data/Search-R1}
LOG_DIR=${LOG_DIR:-${WORK_DIR}/logs}

RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
MODEL_PATH=${MODEL_PATH:-${DATA_ROOT}/models/Qwen2.5-3B-Instruct}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-smoke-fake-retriever-param-${RUN_ID}}
SMOKE_DATA_DIR=${SMOKE_DATA_DIR:-${WORK_DIR}/data/${EXPERIMENT_NAME}}
FAKE_RETRIEVER_PORT=${FAKE_RETRIEVER_PORT:-8001}

TOTAL_STEPS=${TOTAL_STEPS:-1}
MAX_TURNS=${MAX_TURNS:-3}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-512}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-256}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-256}
PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-64}
LOG_PROB_MICRO_BATCH_SIZE=${LOG_PROB_MICRO_BATCH_SIZE:-128}
REF_LOG_PROB_MICRO_BATCH_SIZE=${REF_LOG_PROB_MICRO_BATCH_SIZE:-${LOG_PROB_MICRO_BATCH_SIZE}}
CRITIC_PPO_MINI_BATCH_SIZE=${CRITIC_PPO_MINI_BATCH_SIZE:-256}
CRITIC_PPO_MICRO_BATCH_SIZE=${CRITIC_PPO_MICRO_BATCH_SIZE:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-500}
MAX_START_LENGTH=${MAX_START_LENGTH:-2048}
MAX_OBS_LENGTH=${MAX_OBS_LENGTH:-500}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}
ACTOR_LR_WARMUP_RATIO=${ACTOR_LR_WARMUP_RATIO:-0.285}
CRITIC_LR_WARMUP_RATIO=${CRITIC_LR_WARMUP_RATIO:-0.015}
FSDP_OFFLOAD=${FSDP_OFFLOAD:-false}

ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-${FSDP_OFFLOAD}}
ACTOR_GRAD_OFFLOAD=${ACTOR_GRAD_OFFLOAD:-${FSDP_OFFLOAD}}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-${FSDP_OFFLOAD}}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-${FSDP_OFFLOAD}}
CRITIC_PARAM_OFFLOAD=${CRITIC_PARAM_OFFLOAD:-${FSDP_OFFLOAD}}
CRITIC_GRAD_OFFLOAD=${CRITIC_GRAD_OFFLOAD:-${FSDP_OFFLOAD}}
CRITIC_OPTIMIZER_OFFLOAD=${CRITIC_OPTIMIZER_OFFLOAD:-${FSDP_OFFLOAD}}

export RUN_ID SMOKE_DATA_DIR TRAIN_BATCH_SIZE VAL_BATCH_SIZE FAKE_RETRIEVER_PORT

mkdir -p "${SMOKE_DATA_DIR}" "${LOG_DIR}" "${WORK_DIR}/verl_checkpoints/${EXPERIMENT_NAME}"

python - <<'PY'
import math
import os

import pandas as pd

src = "/data/Search-R1/datasets/nq_hotpotqa_train/train.parquet"
out = os.environ["SMOKE_DATA_DIR"]
train_batch_size = int(os.environ["TRAIN_BATCH_SIZE"])
val_batch_size = int(os.environ["VAL_BATCH_SIZE"])
rows_needed = max(train_batch_size, val_batch_size, 16)

os.makedirs(out, exist_ok=True)
df = pd.read_parquet(src)
base = pd.concat(
    [
        df[df["data_source"] == "nq"].head(max(8, rows_needed // 2)),
        df[df["data_source"] == "hotpotqa"].head(max(8, rows_needed // 2)),
    ],
    ignore_index=True,
)
if len(base) < rows_needed:
    repeats = math.ceil(rows_needed / len(base))
    base = pd.concat([base] * repeats, ignore_index=True)

sample = base.head(rows_needed).reset_index(drop=True)
sample.to_parquet(os.path.join(out, "train.parquet"))
sample.to_parquet(os.path.join(out, "test.parquet"))
print(f"smoke rows: {len(sample)}")
print(sample["data_source"].value_counts().to_string())
PY

python - <<'PY' > "${LOG_DIR}/fake_retriever_${FAKE_RETRIEVER_PORT}_${RUN_ID}.log" 2>&1 &
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

port = int(os.environ["FAKE_RETRIEVER_PORT"])

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"fake retriever ok")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        queries = payload.get("queries") or []
        topk = int(payload.get("topk") or 3)
        result = []
        for query in queries:
            docs = []
            for i in range(topk):
                docs.append(
                    {
                        "document": {
                            "id": f"fake-{i + 1}",
                            "contents": (
                                f"Fake Document {i + 1}\n"
                                f"Fake retrieved passage for query: {query}. "
                                "This passage is only used to verify the Search-R1 training stack."
                            ),
                        },
                        "score": float(topk - i),
                    }
                )
            result.append(docs)
        body = json.dumps({"result": result}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        return

HTTPServer(("127.0.0.1", port), Handler).serve_forever()
PY

FAKE_RETRIEVER_PID=$!
trap 'kill ${FAKE_RETRIEVER_PID} 2>/dev/null || true' EXIT

python - <<'PY'
import os
import requests

port = os.environ["FAKE_RETRIEVER_PORT"]
resp = requests.post(
    f"http://127.0.0.1:{port}/retrieve",
    json={"queries": ["smoke query"], "topk": 3},
    timeout=10,
)
resp.raise_for_status()
first_doc = resp.json()["result"][0][0]["document"]["contents"]
print(resp.status_code, first_doc[:80])
PY

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
  data.train_files="${SMOKE_DATA_DIR}/train.parquet" \
  data.val_files="${SMOKE_DATA_DIR}/test.parquet" \
  data.train_data_num=null \
  data.val_data_num="${VAL_BATCH_SIZE}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.max_start_length="${MAX_START_LENGTH}" \
  data.max_obs_length="${MAX_OBS_LENGTH}" \
  data.shuffle_train_dataloader=False \
  algorithm.adv_estimator=gae \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="${ACTOR_LR_WARMUP_RATIO}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size="${PPO_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}" \
  actor_rollout_ref.actor.fsdp_config.grad_offload="${ACTOR_GRAD_OFFLOAD}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size="${LOG_PROB_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.n_agent=1 \
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
  trainer.logger="['console']" \
  +trainer.val_only=false \
  +trainer.val_before_train=false \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.project_name=search-r1 \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.total_epochs=1 \
  trainer.total_training_steps="${TOTAL_STEPS}" \
  trainer.default_local_dir="${WORK_DIR}/verl_checkpoints/${EXPERIMENT_NAME}" \
  max_turns="${MAX_TURNS}" \
  do_search=true \
  retriever.url="http://127.0.0.1:${FAKE_RETRIEVER_PORT}/retrieve" \
  retriever.topk=3 \
  2>&1 | tee "${LOG_DIR}/${EXPERIMENT_NAME}.log"
