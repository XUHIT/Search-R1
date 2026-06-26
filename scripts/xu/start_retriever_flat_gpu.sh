#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HF_HOME=${HF_HOME:-/data/Search-R1/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/data/Search-R1/hf_cache/transformers}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/data/Search-R1/hf_cache/datasets}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}

DATA_ROOT=${DATA_ROOT:-/data/Search-R1}
WORK_DIR=${WORK_DIR:-/workspace/Search-R1}
RETRIEVAL_DIR=${RETRIEVAL_DIR:-${DATA_ROOT}/retrieval/wiki18_e5_flat}
INDEX_PATH=${INDEX_PATH:-${RETRIEVAL_DIR}/e5_Flat.index}
CORPUS_PATH=${CORPUS_PATH:-${RETRIEVAL_DIR}/wiki-18.jsonl}
RETRIEVER_MODEL=${RETRIEVER_MODEL:-${DATA_ROOT}/models/e5-base-v2}
TOPK=${TOPK:-3}

test -s "${INDEX_PATH}"
test -s "${CORPUS_PATH}"
test -d "${RETRIEVER_MODEL}"

cd "${WORK_DIR}"

python search_r1/search/retrieval_server.py \
  --index_path "${INDEX_PATH}" \
  --corpus_path "${CORPUS_PATH}" \
  --topk "${TOPK}" \
  --retriever_name e5 \
  --retriever_model "${RETRIEVER_MODEL}" \
  --faiss_gpu
