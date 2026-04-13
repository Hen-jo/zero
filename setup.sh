#!/usr/bin/env bash
# Dr. Zero 개발 환경 설치 스크립트
# 사용법: bash setup.sh
# 서버: harbor.auto-hmg.io/hydra-prebuilt/nvidia/pytorch:23.08-py3-260316
#        GPU: T4×4  CPU: 32core  RAM: 240GiB
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================================"
echo " Dr. Zero — 환경 설치"
echo " 프로젝트 경로: $SCRIPT_DIR"
echo "================================================================"

# ── 0. 현재 환경 확인 ──────────────────────────────────────────────
python_ver=$(python --version 2>&1)
torch_ver=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "NOT_FOUND")
cuda_ver=$(python -c "import torch; print(torch.version.cuda or 'N/A')" 2>/dev/null || echo "N/A")
echo "[INFO] Python  : $python_ver"
echo "[INFO] PyTorch : $torch_ver"
echo "[INFO] CUDA    : $cuda_ver"

# ── 1. PyTorch 버전 확인 및 필요 시 업그레이드 ───────────────────
# tensordict 0.6.2, sglang 0.4.8 모두 PyTorch 2.5+ 필요
# requirements.txt 권장: torch 2.6.0 cu124
need_torch_upgrade=false
torch_major=$(python -c "import torch; v=torch.__version__.split('.'); print(int(v[0]))" 2>/dev/null || echo 0)
torch_minor=$(python -c "import torch; v=torch.__version__.split('.'); print(int(v[1]))" 2>/dev/null || echo 0)

if [ "$torch_major" -lt 2 ] || { [ "$torch_major" -eq 2 ] && [ "$torch_minor" -lt 5 ]; }; then
    echo ""
    echo "[WARN] PyTorch $torch_ver 가 너무 낮습니다 (2.5+ 필요)."
    echo "[WARN] PyTorch 2.6.0 (cu124) 를 설치합니다..."
    pip install --quiet torch==2.6.0 torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124
    torch_ver=$(python -c "import torch; print(torch.__version__)")
    echo "[OK] PyTorch 업그레이드 완료: $torch_ver"
    need_torch_upgrade=true
else
    echo "[OK] PyTorch $torch_ver — 버전 요건 충족"
fi

pip install --quiet --upgrade pip setuptools packaging

# ── 2. flashinfer (반드시 sglang 전에, --no-build-isolation 필수) ─
echo ""
echo "[STEP 2] flashinfer 설치..."
if python -c "import flashinfer" 2>/dev/null; then
    echo "[OK] flashinfer 이미 설치됨"
else
    # --no-build-isolation: 이미 설치된 PyTorch를 빌드 시 사용 (없으면 엉뚱한 torch로 빌드됨)
    pip install --no-cache-dir --no-build-isolation flashinfer-python==0.2.6.post1 \
        || echo "[WARN] flashinfer 설치 실패 — sglang은 fallback backend 사용"
fi

# ── 3. sglang ([srt] 전용: vllm 의존성 제거) ─────────────────────
echo ""
echo "[STEP 3] sglang[srt]==0.4.8 설치..."
pip install --no-cache-dir "sglang[srt]==0.4.8"
pip install --no-cache-dir torch-memory-saver

# ── 4. 핵심 ML 패키지 ─────────────────────────────────────────────
echo ""
echo "[STEP 4] 핵심 ML 패키지 설치..."
pip install --no-cache-dir \
    "tensordict==0.6.2" \
    "transformers[hf_xet]>=4.51.0" \
    accelerate \
    datasets \
    peft \
    tokenizers \
    sentencepiece \
    hf-transfer \
    "numpy<2.0.0" \
    "pyarrow>=19.0.1" \
    pandas \
    scipy

# ── 5. 분산 학습 ──────────────────────────────────────────────────
echo ""
echo "[STEP 5] Ray & 분산 학습 패키지 설치..."
pip install --no-cache-dir "ray[default]" codetiming

# ── 6. Serving / API ──────────────────────────────────────────────
echo ""
echo "[STEP 6] serving 패키지 설치..."
pip install --no-cache-dir \
    "fastapi[standard]>=0.115.0" \
    "optree>=0.13.0" \
    "pydantic>=2.9" \
    "grpcio>=1.62.1" \
    "uvicorn[standard]" \
    aiohttp \
    requests \
    httpx \
    openai

# ── 7. 검색 / 리트리벌 ───────────────────────────────────────────
echo ""
echo "[STEP 7] 검색·리트리벌 패키지 설치..."
# faiss-gpu: CUDA 버전 맞으면 GPU, 안 되면 CPU fallback
pip install --no-cache-dir faiss-gpu \
    || { echo "[WARN] faiss-gpu 실패 → faiss-cpu 설치"; pip install --no-cache-dir faiss-cpu; }
pip install --no-cache-dir \
    sentence-transformers \
    google-api-python-client \
    chardet \
    beautifulsoup4 \
    tqdm

# ── 8. 학습 / 설정 유틸 ───────────────────────────────────────────
echo ""
echo "[STEP 8] 학습 설정 유틸 설치..."
pip install --no-cache-dir \
    hydra-core \
    omegaconf \
    wandb \
    dill \
    pybind11 \
    "blobfile" \
    xgrammar \
    pylatexenc \
    qwen-vl-utils \
    mathruler \
    pyext \
    mbridge

# ── 9. NVIDIA / GPU 유틸 ─────────────────────────────────────────
echo ""
echo "[STEP 9] NVIDIA 유틸 설치..."
pip uninstall -y pynvml nvidia-ml-py 2>/dev/null || true
pip install --no-cache-dir "nvidia-ml-py>=12.560.30"
# cudnn: 이미 시스템에 있으면 충돌 가능 → 실패해도 무시
pip install --no-cache-dir "nvidia-cudnn-cu12==9.8.0.87" \
    || echo "[WARN] nvidia-cudnn-cu12 설치 실패 — 시스템 cudnn 사용"

# ── 10. liger-kernel (PyTorch 2.6 기준) ──────────────────────────
echo ""
echo "[STEP 10] liger-kernel 설치..."
pip install --no-cache-dir liger-kernel \
    || echo "[WARN] liger-kernel 설치 실패 — 없어도 기본 동작 가능"

# ── 11. verl (PyPI) ───────────────────────────────────────────────
echo ""
echo "[STEP 11] verl==0.5.0 (PyPI) 설치..."
pip install --no-cache-dir verl==0.5.0

# ── 12. Dev 도구 ──────────────────────────────────────────────────
echo ""
echo "[STEP 12] dev 도구 설치..."
pip install --no-cache-dir pytest ruff

# ── 13. PYTHONPATH 설정 (로컬 verl/ 코드가 PyPI verl보다 우선) ──
echo ""
echo "[STEP 13] PYTHONPATH 설정..."
PYTHONPATH_LINE="export PYTHONPATH=\"$SCRIPT_DIR:\$PYTHONPATH\""

# ~/.bashrc에 없으면 추가
if ! grep -qF "$SCRIPT_DIR" ~/.bashrc 2>/dev/null; then
    echo "" >> ~/.bashrc
    echo "# Dr. Zero — 로컬 verl 코드 우선 사용" >> ~/.bashrc
    echo "$PYTHONPATH_LINE" >> ~/.bashrc
    echo "[OK] ~/.bashrc에 PYTHONPATH 추가됨"
fi

# 현재 셸에 즉시 적용
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# ── 완료 및 검증 ──────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " 설치 완료 — 환경 검증 중..."
echo "================================================================"
python "$SCRIPT_DIR/check_env.py"

echo ""
echo "================================================================"
echo " 다음 단계:"
echo "   cd $SCRIPT_DIR"
echo "   bash iter1_challenger.sh          # 기본 (hop_ratio=4321)"
echo "   bash iter1_challenger.sh 4321     # hop_ratio 직접 지정"
echo "================================================================"
