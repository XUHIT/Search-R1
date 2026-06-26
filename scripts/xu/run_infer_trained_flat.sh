#!/usr/bin/env bash
set -euo pipefail

WORK_DIR=${WORK_DIR:-/workspace/Search-R1}
DATA_ROOT=${DATA_ROOT:-/data/Search-R1}
LOG_DIR=${LOG_DIR:-${WORK_DIR}/logs}
MODEL_ID=${MODEL_ID:-${DATA_ROOT}/models/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo}
RETRIEVER_URL=${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}
QUESTION=${QUESTION:-"Mike Barnett negotiated many contracts including which player that went on to become general manager of CSKA Moscow of the Kontinental Hockey League?"}
TOPK=${TOPK:-3}
MAX_TURNS=${MAX_TURNS:-5}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
TEMPERATURE=${TEMPERATURE:-0.7}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/infer_trained_flat_$(date +%Y%m%d_%H%M%S).log}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_HOME=${HF_HOME:-${DATA_ROOT}/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${DATA_ROOT}/hf_cache/transformers}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${DATA_ROOT}/hf_cache/datasets}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}
export MODEL_ID RETRIEVER_URL QUESTION TOPK MAX_TURNS MAX_NEW_TOKENS TEMPERATURE

mkdir -p "${LOG_DIR}"
test -d "${MODEL_ID}"

python - <<'PY'
import os
import requests

url = os.environ["RETRIEVER_URL"]
resp = requests.post(
    url,
    json={"queries": ["where is the Eiffel Tower located"], "topk": 3, "return_scores": True},
    timeout=60,
)
resp.raise_for_status()
doc = resp.json()["result"][0][0]["document"]["contents"]
print("retriever_ok", resp.status_code, doc[:120].replace("\n", " "))
PY

cd "${WORK_DIR}"

{
  date "+%Y-%m-%d %H:%M:%S %Z"
  echo "MODEL_ID=${MODEL_ID}"
  echo "RETRIEVER_URL=${RETRIEVER_URL}"
  echo "QUESTION=${QUESTION}"
  echo "TOPK=${TOPK}"
  echo "MAX_TURNS=${MAX_TURNS}"
  echo "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
  python scripts/xu/infer_trained_model.py
} 2>&1 | tee "${LOG_FILE}"

echo "infer_log=${LOG_FILE}"
