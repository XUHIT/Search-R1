FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ARG CONDA_ENV=retriever
ARG PYTHON_VERSION=3.10
ARG MINIFORGE_URL=https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-Linux-x86_64.sh

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    JAVA_HOME=/opt/conda/envs/retriever \
    PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/opt/conda/envs/retriever/lib/python3.10/site-packages/torch/lib:/opt/conda/envs/retriever/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    PIP_EXTRA_INDEX_URL=https://pypi.org/simple \
    PIP_DEFAULT_TIMEOUT=180 \
    PIP_RETRIES=10 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface

SHELL ["/bin/bash", "-lc"]

RUN sed -i \
        -e 's|http://archive.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g' \
        -e 's|http://security.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g' \
        /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        bzip2 \
        ca-certificates \
        curl \
        git \
        libgomp1 \
        openssh-client \
        procps \
        wget && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL "${MINIFORGE_URL}" -o /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm -f /tmp/miniforge.sh && \
    conda create -n "${CONDA_ENV}" "python=${PYTHON_VERSION}" -y && \
    conda install -n "${CONDA_ENV}" -y openjdk=21 -c conda-forge && \
    conda clean -afy

RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate "${CONDA_ENV}" && \
    conda install -y --override-channels --strict-channel-priority \
        pytorch==2.4.0 \
        torchvision==0.19.0 \
        torchaudio==2.4.0 \
        pytorch-cuda=12.1 \
        -c pytorch \
        -c nvidia \
        -c defaults && \
    conda clean -afy

RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate "${CONDA_ENV}" && \
    conda install -y --override-channels --strict-channel-priority \
        faiss-gpu=1.8.0 \
        -c pytorch \
        -c nvidia \
        -c defaults && \
    conda clean -afy

COPY docker/retriever/wheelhouse/pyserini-1.2.0.tar.gz /tmp/wheelhouse/pyserini-1.2.0.tar.gz

RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate "${CONDA_ENV}" && \
    python -m pip install --upgrade pip setuptools wheel packaging && \
    PIP_EXTRA_INDEX_URL= pip install --timeout 600 --retries 20 --prefer-binary \
        --index-url https://mirrors.aliyun.com/pypi/simple/ \
        transformers \
        datasets \
        /tmp/wheelhouse/pyserini-1.2.0.tar.gz \
        uvicorn \
        fastapi

ENV OPENAI_API_KEY=sk-dummy-pyserini-local-retriever
ENV PYTHONPATH=/workspace/Search-R1:$PYTHONPATH

WORKDIR /workspace/Search-R1
COPY . /workspace/Search-R1

RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate "${CONDA_ENV}" && \
    python docker/retriever/smoke_check.py

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "retriever"]
CMD ["bash"]
