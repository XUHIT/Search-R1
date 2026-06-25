# Search-R1 Retriever Docker

This image follows the project's optional retriever environment:

- Python 3.10
- PyTorch 2.4.0 with CUDA 12.1
- faiss-gpu 1.8.0
- OpenJDK 21 for pyserini/Lucene
- transformers, datasets, pyserini 1.2.0
- FastAPI and uvicorn

Build:

```bash
docker build -t searchr1-retriever:cu121-faiss18 -f docker/retriever/Dockerfile .
```

Run the default e5 flat retriever with host networking:

```bash
docker run --rm --gpus all --network host \
  -v /path/to/retriever-data:/data/retriever \
  searchr1-retriever:cu121-faiss18 \
  python search_r1/search/retrieval_server.py \
    --index_path /data/retriever/e5_Flat.index \
    --corpus_path /data/retriever/wiki-18.jsonl \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model intfloat/e5-base-v2 \
    --faiss_gpu
```

The training container expects the retrieval API at:

```text
http://127.0.0.1:8000/retrieve
```
