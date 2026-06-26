#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export WANDB_MODE=${WANDB_MODE:-offline}
export HF_HOME=${HF_HOME:-/data/Search-R1/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/data/Search-R1/hf_cache/transformers}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/data/Search-R1/hf_cache/datasets}
export TOKENIZERS_PARALLELISM=true

WORK_DIR=/workspace/Search-R1
DATA_ROOT=/data/Search-R1
SMOKE_DATA_DIR="${WORK_DIR}/data/smoke_train_fake_retriever"
LOG_DIR="${WORK_DIR}/logs"
MODEL_PATH="${DATA_ROOT}/models/Qwen2.5-3B"
EXPERIMENT_NAME=smoke-qwen2.5-3b-fake-retriever-1step

mkdir -p "${SMOKE_DATA_DIR}" "${LOG_DIR}" "${WORK_DIR}/verl_checkpoints/${EXPERIMENT_NAME}"

python - <<'PY'
import os
import pandas as pd

src = "/data/Search-R1/datasets/nq_hotpotqa_train/train.parquet"
out = "/workspace/Search-R1/data/smoke_train_fake_retriever"
os.makedirs(out, exist_ok=True)

df = pd.read_parquet(src)
sample = pd.concat([
    df[df["data_source"] == "nq"].head(4),
    df[df["data_source"] == "hotpotqa"].head(4),
], ignore_index=True)

sample.to_parquet(os.path.join(out, "train.parquet"))
sample.to_parquet(os.path.join(out, "test.parquet"))
print(f"smoke rows: {len(sample)}")
print(sample["data_source"].value_counts().to_string())
PY

python - <<'PY' > "${LOG_DIR}/fake_retriever_8000.log" 2>&1 &
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

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
                docs.append({
                    "document": {
                        "id": f"fake-{i + 1}",
                        "contents": (
                            f"Fake Document {i + 1}\n"
                            f"Fake retrieved passage for query: {query}. "
                            "This passage is only used to verify the Search-R1 training stack."
                        ),
                    },
                    "score": float(topk - i),
                })
            result.append(docs)
        body = json.dumps({"result": result}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        return

HTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
PY

FAKE_RETRIEVER_PID=$!
trap 'kill ${FAKE_RETRIEVER_PID} 2>/dev/null || true' EXIT

python - <<'PY'
import requests
resp = requests.post(
    "http://127.0.0.1:8000/retrieve",
    json={"queries": ["smoke query"], "topk": 3},
    timeout=10,
)
first_doc = resp.json()["result"][0][0]["document"]["contents"]
print(resp.status_code, first_doc[:80])
PY

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
  data.train_files="${SMOKE_DATA_DIR}/train.parquet" \
  data.val_files="${SMOKE_DATA_DIR}/test.parquet" \
  data.train_data_num=null \
  data.val_data_num=8 \
  data.train_batch_size=8 \
  data.val_batch_size=8 \
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
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size=8 \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.grad_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.rollout.n_agent=1 \
  actor_rollout_ref.rollout.temperature=1 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.actor.state_masking=true \
  actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  critic.optim.lr=1e-5 \
  critic.model.path="${MODEL_PATH}" \
  critic.model.use_remove_padding=True \
  critic.model.enable_gradient_checkpointing=true \
  critic.optim.lr_warmup_steps_ratio=0.0 \
  critic.ppo_mini_batch_size=8 \
  critic.ppo_micro_batch_size=8 \
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
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.project_name=Search-R1 \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  trainer.default_local_dir="${WORK_DIR}/verl_checkpoints/${EXPERIMENT_NAME}" \
  max_turns=1 \
  do_search=true \
  retriever.url="http://127.0.0.1:8000/retrieve" \
  retriever.topk=3
