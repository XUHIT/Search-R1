FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ARG CONDA_ENV=searchr1
ARG PYTHON_VERSION=3.9
ARG FLASH_ATTN_VERSION=2.6.3
ARG MINIFORGE_URL=https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-Linux-x86_64.sh

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    PIP_EXTRA_INDEX_URL=https://pypi.org/simple \
    PIP_DEFAULT_TIMEOUT=180 \
    PIP_RETRIES=10 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6" \
    FORCE_CUDA=1 \
    MAX_JOBS=4 \
    VLLM_ATTENTION_BACKEND=XFORMERS \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    WANDB_MODE=offline

SHELL ["/bin/bash", "-lc"]

RUN sed -i \
        -e 's|http://archive.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g' \
        -e 's|http://security.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g' \
        /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        bash \
        bzip2 \
        ca-certificates \
        curl \
        git \
        libaio-dev \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        openssh-client \
        build-essential \
        ninja-build \
        pkg-config \
        wget && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL "${MINIFORGE_URL}" -o /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm -f /tmp/miniforge.sh && \
    conda config --system --remove-key channels || true && \
    conda config --system --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge && \
    conda config --system --set channel_priority strict && \
    conda create -n "${CONDA_ENV}" "python=${PYTHON_VERSION}" -y && \
    conda clean -afy

RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate "${CONDA_ENV}" && \
    python -m pip install --upgrade pip setuptools wheel packaging ninja

RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate "${CONDA_ENV}" && \
    pip install --timeout 120 --retries 10 torch==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu121 \
        --extra-index-url https://mirrors.aliyun.com/pypi/simple/

RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate "${CONDA_ENV}" && \
    PIP_EXTRA_INDEX_URL= pip install --timeout 180 --retries 20 --prefer-binary \
        --index-url https://pypi.org/simple \
        vllm==0.6.3

RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate "${CONDA_ENV}" && \
    PIP_EXTRA_INDEX_URL= pip install --timeout 180 --retries 20 --prefer-binary \
        --index-url https://pypi.org/simple \
        accelerate \
        codetiming \
        datasets \
        dill \
        hydra-core \
        IPython \
        matplotlib \
        "numpy<2" \
        pandas \
        pybind11 \
        ray \
        einops \
        "transformers==4.47.1" \
        "tensordict<0.6" \
        sentencepiece

RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate "${CONDA_ENV}" && \
    FLASH_ATTN_SRC=/tmp/flash-attn-src && \
    rm -rf "${FLASH_ATTN_SRC}" && \
    mkdir -p "${FLASH_ATTN_SRC}" && \
    PIP_EXTRA_INDEX_URL= pip download --timeout 180 --retries 10 --no-deps --no-build-isolation --no-binary flash-attn \
        --index-url https://pypi.org/simple \
        "flash-attn==${FLASH_ATTN_VERSION}" \
        -d "${FLASH_ATTN_SRC}" && \
    tar -xzf "${FLASH_ATTN_SRC}/flash_attn-${FLASH_ATTN_VERSION}.tar.gz" -C "${FLASH_ATTN_SRC}" && \
    python -c 'import os; from pathlib import Path; p = Path("/tmp/flash-attn-src") / ("flash_attn-" + os.environ["FLASH_ATTN_VERSION"]) / "setup.py"; text = p.read_text(); needle = "    cc_flag.append(\"-gencode\")\n    cc_flag.append(\"arch=compute_80,code=sm_80\")\n"; repl = needle + "    cc_flag.append(\"-gencode\")\n    cc_flag.append(\"arch=compute_86,code=sm_86\")\n"; sm90 = "    if CUDA_HOME is not None:\n        if bare_metal_version >= Version(\"11.8\"):\n            cc_flag.append(\"-gencode\")\n            cc_flag.append(\"arch=compute_90,code=sm_90\")\n"; assert needle in text and sm90 in text; text = text.replace(needle, repl, 1).replace(sm90, "", 1); assert "compute_90" not in text; p.write_text(text)' && \
    grep -n "compute_86" "${FLASH_ATTN_SRC}/flash_attn-${FLASH_ATTN_VERSION}/setup.py" && \
    FLASH_ATTENTION_FORCE_BUILD=TRUE PIP_EXTRA_INDEX_URL= pip install -v --timeout 180 --retries 10 --no-deps \
        --index-url https://pypi.org/simple \
        "${FLASH_ATTN_SRC}/flash_attn-${FLASH_ATTN_VERSION}" --no-build-isolation && \
    PIP_EXTRA_INDEX_URL= pip install --timeout 180 --retries 20 \
        --index-url https://pypi.org/simple \
        wandb && \
    rm -rf "${FLASH_ATTN_SRC}"

WORKDIR /workspace/Search-R1
COPY . /workspace/Search-R1

RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate "${CONDA_ENV}" && \
    pip install -e . --no-deps && \
    pip check

RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate "${CONDA_ENV}" && \
    python docker/searchr1/smoke_check.py

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "searchr1"]
CMD ["bash"]
