# Base image: 25.04-py3 (Ubuntu 24.04, Python 3.12, CUDA 12.9.0)
# This matches nvcr.io/nvidia/pytorch:25.04-py3 used in the original verl Dockerfile.
# Use your company's internal equivalent: 25.04-py3-{build-date}
FROM nvcr.io/nvidia/pytorch:25.04-py3

ENV MAX_JOBS=8
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_ROOT_USER_ACTION=ignore
ENV HF_HUB_ENABLE_HF_TRANSFER="1"

RUN pip install --upgrade pip setuptools packaging

# ── Step 1: flashinfer (MUST come before sglang) ─────────────────────────────
# --no-build-isolation is required so flashinfer can find the pre-installed PyTorch.
# Without this flag, flashinfer builds against the wrong torch and sglang crashes.
RUN pip install --no-cache-dir --no-build-isolation flashinfer-python==0.2.6.post1

# ── Step 2: sglang (use [srt], NOT [all]) ─────────────────────────────────────
# sglang[all] pulls in vllm as an optional backend. vllm requires compiling CUDA
# kernels and fails in most managed environments. sglang[srt] is the native
# SGLang runtime and is all verl needs. --tool-call-parser=qwen25 works with [srt].
RUN pip install --no-cache-dir "sglang[srt]==0.4.8"
RUN pip install --no-cache-dir torch-memory-saver

# ── Step 3: Core ML / training packages ───────────────────────────────────────
RUN pip install --no-cache-dir \
    "tensordict==0.6.2" \
    "transformers[hf_xet]>=4.51.0" \
    accelerate \
    datasets \
    peft \
    hf-transfer \
    "numpy<2.0.0" \
    "pyarrow>=19.0.1" \
    pandas \
    "ray[default]" \
    codetiming \
    hydra-core \
    pylatexenc \
    qwen-vl-utils \
    wandb \
    dill \
    pybind11 \
    liger-kernel \
    mathruler \
    blobfile \
    xgrammar

# ── Step 4: API / serving ─────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    "fastapi[standard]>=0.115.0" \
    "optree>=0.13.0" \
    "pydantic>=2.9" \
    "grpcio>=1.62.1" \
    "uvicorn[standard]" \
    aiohttp \
    requests

# ── Step 5: Search / retrieval ────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    faiss-gpu \
    sentence-transformers \
    google-api-python-client \
    chardet \
    beautifulsoup4 \
    tqdm

# ── Step 6: NVIDIA / infra ────────────────────────────────────────────────────
RUN pip uninstall -y pynvml nvidia-ml-py && \
    pip install --no-cache-dir "nvidia-ml-py>=12.560.30"
RUN pip install --no-cache-dir nvidia-cudnn-cu12==9.8.0.87
RUN pip install --no-cache-dir mbridge

# ── Step 7: Dev / test ────────────────────────────────────────────────────────
RUN pip install --no-cache-dir pytest py-spy pyext pre-commit ruff

# ── Step 8: TransformerEngine (source build, order matters) ───────────────────
RUN export NVTE_FRAMEWORK=pytorch && \
    pip install --no-deps --no-cache-dir --no-build-isolation \
    git+https://github.com/NVIDIA/TransformerEngine.git@v2.3

# ── Step 9: Megatron-LM (source build) ────────────────────────────────────────
RUN pip install --no-deps --no-cache-dir --no-build-isolation \
    git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.12.2

# ── Step 10: verl ─────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir verl==0.5.0

COPY . /workspace
WORKDIR /workspace
