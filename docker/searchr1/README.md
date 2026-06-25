# Search-R1 Docker

This image packages the Search-R1 training environment:

- CUDA 12.1 devel base image
- Python 3.9 in a conda environment named `searchr1`
- PyTorch 2.4.0 CUDA 12.1
- vLLM 0.6.3
- transformers 4.47.1
- flash-attn 2.6.3 built from source
- the vendored Search-R1/verl package from this repository

The image is intended to be built on RTX 3090 and run on A100. The Dockerfile
sets:

```bash
TORCH_CUDA_ARCH_LIST="8.0;8.6"
```

This includes both A100 (`sm80`) and RTX 3090 (`sm86`) kernels.

Build from the repository root:

```bash
docker build -t searchr1:cu121-vllm063-flashattn -f docker/searchr1/Dockerfile .
docker save -o searchr1_cu121_vllm063_flashattn.tar searchr1:cu121-vllm063-flashattn
```

Run on a GPU host with NVIDIA Container Toolkit:

```bash
docker run --rm --gpus all -it searchr1:cu121-vllm063-flashattn bash
```

This image intentionally does not include the optional retriever environment
(`faiss-gpu`, `pyserini`, `fastapi`). Keep that as a separate image or service.
