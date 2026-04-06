# Dr. Zero: Self-Evolving Search Agents
# Base: PyTorch 2.3 + CUDA 12.1 (verl 0.5.0 / sglang 호환)
# 사내 Nexus mirror 사용
FROM docker-registry.hmc.co.kr/pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 사내 APT 저장소 설정 (Ubuntu 22.04 기준)
RUN rm -f /etc/apt/sources.list.d/* && \
    echo 'Acquire::https::Verify-Peer "false";' > /etc/apt/apt.conf.d/80-ssl-exceptions && \
    echo 'Acquire::https::Verify-Host "false";' >> /etc/apt/apt.conf.d/80-ssl-exceptions && \
    echo 'deb [trusted=yes arch=amd64] https://nexus.auto-hmg.io/repository/apt-ubuntu-jammy/ jammy main restricted multiverse universe' > /etc/apt/sources.list && \
    echo 'deb [trusted=yes arch=amd64] https://nexus.auto-hmg.io/repository/apt-ubuntu-jammy/ jammy-updates main restricted multiverse universe' >> /etc/apt/sources.list && \
    echo 'deb [trusted=yes arch=amd64] https://nexus.auto-hmg.io/repository/apt-ubuntu-jammy/ jammy-security main restricted multiverse universe' >> /etc/apt/sources.list

# 사내 pip 저장소 설정
RUN echo "[global]" > /etc/pip.conf && \
    echo "trusted-host = nexus.auto-hmg.io" >> /etc/pip.conf && \
    echo "timeout = 6000" >> /etc/pip.conf && \
    echo "" >> /etc/pip.conf && \
    echo "[install]" >> /etc/pip.conf && \
    echo "index-url = https://nexus.auto-hmg.io/repository/pypi-proxy/simple/" >> /etc/pip.conf && \
    echo "" >> /etc/pip.conf && \
    echo "[search]" >> /etc/pip.conf && \
    echo "index-url = https://nexus.auto-hmg.io/repository/pypi-proxy/pypi" >> /etc/pip.conf

# 시스템 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl build-essential openjdk-11-jdk \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# verl 0.5.0 + sglang 의존성 설치
RUN pip install --upgrade pip && \
    pip install \
        "verl==0.5.0" \
        "sglang[all]>=0.3.0" \
        "transformers>=4.40.0" \
        "datasets>=2.18.0" \
        "accelerate>=0.28.0" \
        "peft>=0.10.0" \
        "trl>=0.8.0" \
        "hydra-core>=1.3.0" \
        "omegaconf>=2.3.0" \
        "wandb>=0.16.0" \
        "fastapi>=0.110.0" \
        "uvicorn[standard]>=0.29.0" \
        "pydantic>=2.0.0" \
        "numpy>=1.26.0" \
        "scipy>=1.12.0" \
        "tqdm>=4.66.0" \
        "einops" \
        "tensorboard" \
        "ray[default]>=2.10.0" \
        "pyzmq" \
        "codetiming" \
        "tensordict>=0.4.0" \
        "httpx>=0.27.0" \
        "openai>=1.0.0"

# faiss-gpu (CUDA 12.1 호환)
RUN pip install faiss-gpu-cu12

# flash-attention (학습 속도 향상, CUDA kernel 빌드 필요)
RUN pip install flash-attn --no-build-isolation

# pyserini (BM25, 선택적 사용)
RUN pip install pyserini

# 프로젝트 코드 복사
COPY . /workspace/drzero
WORKDIR /workspace/drzero

# 로컬 verl 패키지를 PYTHONPATH에 추가
# (프로젝트의 커스텀 verl/ 디렉토리가 설치된 verl을 오버라이드)
ENV PYTHONPATH="/workspace/drzero:${PYTHONPATH}"

# 데이터/코퍼스/체크포인트용 볼륨 마운트 포인트
RUN mkdir -p /workspace/drzero/data \
             /workspace/drzero/corpus \
             /workspace/drzero/outputs \
             /workspace/drzero/checkpoints

VOLUME ["/workspace/drzero/data", "/workspace/drzero/corpus", "/workspace/drzero/checkpoints"]

# retrieval server 기본 포트
EXPOSE 8000 8001

CMD ["/bin/bash"]
