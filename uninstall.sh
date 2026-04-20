#!/usr/bin/env bash
# 직접 pip install한 패키지만 제거 (이미지 기본 패키지 유지)
# 사용법: bash uninstall.sh
set -e

echo "================================================================"
echo " pip 설치 패키지 제거"
echo "================================================================"

pip uninstall -y \
    flashinfer-python \
    flashinfer \
    "sglang" \
    torch-memory-saver \
    xgrammar \
    tensordict \
    transformers \
    accelerate \
    peft \
    datasets \
    tokenizers \
    sentencepiece \
    hf-transfer \
    pyarrow \
    pandas \
    scipy \
    ray \
    codetiming \
    fastapi \
    uvicorn \
    pydantic \
    grpcio \
    aiohttp \
    requests \
    httpx \
    openai \
    faiss-gpu \
    faiss-cpu \
    sentence-transformers \
    google-api-python-client \
    chardet \
    beautifulsoup4 \
    tqdm \
    hydra-core \
    omegaconf \
    wandb \
    dill \
    pybind11 \
    blobfile \
    pylatexenc \
    qwen-vl-utils \
    mathruler \
    pyext \
    mbridge \
    liger-kernel \
    nvidia-ml-py \
    pynvml \
    nvidia-cudnn-cu12 \
    verl \
    pytest \
    ruff \
    pre-commit \
    optree \
    packaging \
    setuptools \
    2>/dev/null || true

echo ""
echo "[완료] 제거 완료. 현재 pip 목록:"
pip list
