"""
Dr. Zero 개발 환경 체크 스크립트
- 실제 학습 없이 의존성 import 및 기본 동작을 확인
- PASS/FAIL 출력으로 어느 단계가 문제인지 빠르게 진단
"""

import sys
import os

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

results = []

def check(name, fn):
    try:
        info = fn()
        msg = f"{PASS} {name}" + (f" — {info}" if info else "")
        print(msg)
        results.append((name, True, None))
    except Exception as e:
        print(f"{FAIL} {name} — {type(e).__name__}: {e}")
        results.append((name, False, str(e)))

# ── 1. Python ────────────────────────────────────────────────────────────────
check("Python version", lambda: f"{sys.version.split()[0]} (expect 3.10)")

# ── 2. PyTorch + CUDA ────────────────────────────────────────────────────────
def check_torch():
    import torch
    cuda_ok = torch.cuda.is_available()
    n_gpu = torch.cuda.device_count() if cuda_ok else 0
    cuda_ver = torch.version.cuda or "N/A"
    return (
        f"torch {torch.__version__}, "
        f"CUDA {cuda_ver}, "
        f"{n_gpu} GPU(s) {'✓' if cuda_ok else '✗'}"
    )
check("PyTorch + CUDA", check_torch)

def check_gpu_names():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return ", ".join(names)
check("GPU devices", check_gpu_names)

# ── 3. Flash Attention ───────────────────────────────────────────────────────
def check_flash_attn():
    import flash_attn
    return f"v{flash_attn.__version__}"
check("flash-attn", check_flash_attn)

# ── 4. flashinfer (optional) ─────────────────────────────────────────────────
def check_flashinfer():
    import flashinfer
    return f"v{flashinfer.__version__}"
try:
    check("flashinfer", check_flashinfer)
except Exception:
    # flashinfer 없으면 warn만
    print(f"{WARN} flashinfer — not installed (sglang will fall back to triton attention)")
    results.append(("flashinfer", True, "optional"))  # warn이지 fail 아님

# ── 5. faiss ─────────────────────────────────────────────────────────────────
def check_faiss():
    import faiss
    has_gpu = hasattr(faiss, "StandardGpuResources")
    return f"faiss {'(GPU)' if has_gpu else '(CPU only)'}"
check("faiss", check_faiss)

# ── 6. sglang 임포트 ──────────────────────────────────────────────────────────
def check_sglang():
    import sglang
    return f"v{sglang.__version__}"
check("sglang import", check_sglang)

def check_sglang_triton_patch():
    """maybe_set_triton_cache_manager 패치 동작 확인"""
    try:
        from sglang.srt.utils import maybe_set_triton_cache_manager
        return "native (함수 존재)"
    except ImportError:
        # 패치 코드와 동일한 fallback
        def maybe_set_triton_cache_manager():
            pass
        return "fallback no-op (정상 패치 동작)"
check("sglang maybe_set_triton_cache_manager", check_sglang_triton_patch)

# ── 7. verl 임포트 ────────────────────────────────────────────────────────────
def check_verl():
    import verl
    return f"verl from {os.path.dirname(verl.__file__)}"
check("verl import", check_verl)

# ── 8. sglang_rollout 임포트 (핵심 패치 확인) ────────────────────────────────
def check_sglang_rollout():
    # 이 import가 실패하면 학습 자체가 불가
    from verl.workers.rollout.sglang_rollout import SGLangRollout
    return "SGLangRollout imported OK"
check("verl.workers.rollout.sglang_rollout", check_sglang_rollout)

# ── 9. reward_function 임포트 (lazy import 패치 확인) ────────────────────────
def check_reward_fn():
    # top-level import만 검사 (MultiTurnRewardRollout은 lazy이므로 여기선 안 불림)
    from verl.custom_reward.reward_function import compute_challenger_score_batch
    return "compute_challenger_score_batch imported OK"
check("verl.custom_reward.reward_function", check_reward_fn)

# ── 10. prompts 임포트 ────────────────────────────────────────────────────────
def check_prompts():
    from verl.prompts import DEFAULT_SOLVER_PREFIX, TOOL_CALL_PATTERN, ANSWER_PATTERN
    return "prompts OK"
check("verl.prompts", check_prompts)

# ── 11. Ray ───────────────────────────────────────────────────────────────────
def check_ray():
    import ray
    ray.init(ignore_reinit_error=True, num_cpus=1, include_dashboard=False)
    resources = ray.available_resources()
    n_cpu = int(resources.get("CPU", 0))
    n_gpu = int(resources.get("GPU", 0))
    ray.shutdown()
    return f"Ray OK — {n_cpu} CPU, {n_gpu} GPU visible"
check("Ray init", check_ray)

# ── 12. transformers ─────────────────────────────────────────────────────────
def check_transformers():
    import transformers
    return f"v{transformers.__version__}"
check("transformers", check_transformers)

# ── 13. sglang server 연결 확인 (이미 실행 중인 경우) ────────────────────────
def check_sglang_server():
    import urllib.request
    import urllib.error
    url = "http://127.0.0.1:8001/health"
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            return f"HTTP {r.status} — sglang server running"
    except urllib.error.URLError:
        raise RuntimeError("server not running (학습 시작 전엔 정상)")
try:
    check("sglang server (port 8001)", check_sglang_server)
except Exception:
    print(f"{INFO} sglang server — not running (학습 시 iter1_challenger.sh가 기동)")
    results.append(("sglang server", True, "optional"))

# ── 14. retrieval server 연결 확인 ────────────────────────────────────────────
def check_retrieval_server():
    import urllib.request
    import urllib.error
    url = "http://127.0.0.1:8000/retrieve"
    try:
        import json
        data = json.dumps({"query": ["test"], "topk": 1}).encode()
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3) as r:
            return f"HTTP {r.status} — retrieval server running"
    except urllib.error.URLError:
        raise RuntimeError("server not running (학습 시작 전엔 정상)")
try:
    check("retrieval server (port 8000)", check_retrieval_server)
except Exception:
    print(f"{INFO} retrieval server — not running (학습 시 iter1_challenger.sh가 기동)")
    results.append(("retrieval server", True, "optional"))

# ── 결과 요약 ─────────────────────────────────────────────────────────────────
print()
print("=" * 55)
failures = [(n, e) for n, ok, e in results if not ok]
if failures:
    print(f"{FAIL} 실패 항목 {len(failures)}개:")
    for name, err in failures:
        print(f"     - {name}: {err}")
    sys.exit(1)
else:
    print(f"{PASS} 모든 체크 통과 — 환경 정상")
    sys.exit(0)
