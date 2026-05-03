#!/bin/bash
# CPU FAISS retrieval server 실제 성능 측정
# - 배치 크기별 레이턴시 (n=1,5,10,20,50)
# - 동시 요청 수별 처리량 (concurrency=1,5,10,20,30,50)
# → num_workers / rate_limit 최적값 판단용

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR=/tmp/retrieval_perf_$$
mkdir -p "$LOG_DIR"

cleanup() {
    echo ""
    echo "[cleanup] 서버 종료..."
    [ -n "$SERVER_PID" ] && kill $SERVER_PID 2>/dev/null || true
    rm -rf "$LOG_DIR"
}
trap cleanup EXIT

# ── [1] 서버 시작 ────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "[1/3] retrieval server 시작 (CPU FAISS)"
echo "========================================"
PYTHONUNBUFFERED=1 python3 "$SCRIPT_DIR/search/retrieval_server.py" \
    --index_path='/input/data/Drzero/corpus/e5_Flat.index' \
    --corpus_path='/input/data/Drzero/corpus/wiki-18.jsonl' \
    --retriever_model='/input/model/e5-base-v2' \
    --retriever_name='e5' \
    --topk 3 > "$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!
echo "[startup] PID=$SERVER_PID, port 8000 대기 중..."

elapsed=0
until bash -c "echo > /dev/tcp/127.0.0.1/8000" 2>/dev/null; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "[ERROR] 서버 종료됨. 로그:"; cat "$LOG_DIR/server.log"; exit 1
    fi
    sleep 10; elapsed=$((elapsed + 10))
    [ $((elapsed % 120)) -eq 0 ] && echo "  ... ${elapsed}초 경과 (인덱스 로딩 중)"
    [ $elapsed -ge 1800 ] && echo "[ERROR] 1800초 타임아웃" && cat "$LOG_DIR/server.log" && exit 1
done
echo "[startup] 준비 완료 (${elapsed}초 소요)"

# ── [2] 배치 크기별 레이턴시 + [3] 동시 요청 처리량 ────────────────────────
echo ""
echo "========================================"
echo "[2/3] 배치 크기별 레이턴시  &  [3/3] 동시 요청 처리량"
echo "========================================"
python3 - <<'PYEOF'
import requests, time, json, concurrent.futures, statistics

URL = "http://127.0.0.1:8000/retrieve"

def make_payload(n):
    return {"queries": [f"What is the capital of country number {i}?" for i in range(n)], "topk": 3}

def send(payload):
    t0 = time.time()
    r = requests.post(URL, json=payload, timeout=600)
    return time.time() - t0, r.status_code

# ── 배치 크기별 레이턴시 ──────────────────────────────────────────────────
print("\n[레이턴시] 배치 크기별 (5회 평균, 1회 warmup)")
print(f"  {'n':>4}  {'avg':>7}  {'min':>7}  {'max':>7}  {'p95':>7}")
print(f"  {'-'*40}")

for n in [1, 5, 10, 20, 50]:
    payload = make_payload(n)
    send(payload)  # warmup
    lats = [send(payload)[0] for _ in range(5)]
    lats_s = sorted(lats)
    print(f"  {n:>4}  {statistics.mean(lats):>6.2f}s  {min(lats):>6.2f}s  {max(lats):>6.2f}s  {lats_s[int(0.95*len(lats_s))]:>6.2f}s")

# ── 동시 요청 처리량 ──────────────────────────────────────────────────────
print("\n[처리량] 동시 요청 수별 (n=10 쿼리, 총 60회 요청)")
print(f"  {'concur':>7}  {'throughput':>11}  {'avg_lat':>8}  {'p95_lat':>8}  {'ok':>6}")
print(f"  {'-'*50}")

payload = make_payload(10)
N_TOTAL = 20

for concurrency in [1, 5, 10, 20, 30]:
    t_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(send, payload) for _ in range(N_TOTAL)]
        results = [f.result() for f in concurrent.futures.as_completed(futs)]
    elapsed = time.time() - t_start

    lats = [r[0] for r in results]
    ok = sum(1 for r in results if r[1] == 200)
    throughput = N_TOTAL / elapsed
    p95 = sorted(lats)[int(0.95 * len(lats))]
    print(f"  {concurrency:>7}  {throughput:>9.1f}/s  {statistics.mean(lats):>7.2f}s  {p95:>7.2f}s  {ok:>4}/{N_TOTAL}")

print("\n[해석 가이드]")
print("  - throughput이 plateau되는 concurrency = 서버 포화점")
print("  - avg_lat이 급격히 증가하기 직전 concurrency = 최적 num_workers 후보")
print("  - rate_limit은 보통 num_workers 이하로 설정 (큐 overflow 방지)")
PYEOF

echo ""
echo "========================================"
echo "서버 로그 마지막 5줄:"
echo "========================================"
tail -5 "$LOG_DIR/server.log"
