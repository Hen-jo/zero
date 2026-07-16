<div align="center">

<img src="assets/image1.png" width="700"/>

# 강화학습 기반 자기 대전을 통한 다중 홉 역질문 복원

**Don't Just Ask — Reconstruct: Trajectory-Conditioned Inverse Question Reconstruction for Self-Play Multi-hop QA**

조재헝 · 윤재욱 · 이상호 · 하승구 · 정의량 · 나욱진

현대자동차그룹 CVLCM AI팀

[![GitLab](https://img.shields.io/badge/Code-GitLab-orange)](https://gitlab.hmc.co.kr/cvlcmai/2026_hmc_paper/vef_zero)
[![Base](https://img.shields.io/badge/Base-DrZero-blue)](https://arxiv.org/abs/2601.07055)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

</div>

---

## 개요

Self-play 프레임워크는 어려운 질문을 생성하는 **제안자(Proposer)**와 이를 푸는 **해결자(Solver)**를 공동 학습시키지만, 제안자의 보상이 단순 solver 실패율에만 의존하면 **근거가 약하거나 shortcut으로 풀리는 질문**을 만들 수 있습니다.

본 구현은 DrZero에 **trajectory-conditioned inverse question reconstructor**를 추가합니다. Solver가 남긴 최종 검색질문과 답에서 출발해 verifier가 검색을 역방향으로 수행하고, 숨겨진 원 질문을 복원합니다. Verifier는 자연어로 좋은/나쁜 질문을 판정하는 LLM judge가 아닙니다. 원 질문의 의미·핵심 entity·relation을 실제로 복원해야 GRPO 보상을 받습니다. 학습된 verifier 체크포인트는 다음 iteration의 proposer 초기화에 사용됩니다.

---

## 핵심 아이디어

<img src="assets/image2.png" width="650"/>

기존 verifier는 `질문 + solver 답 → 정오 판정`에 가깝습니다. 이 구현은 verifier와 proposer의 출력 공간을 질문으로 맞춥니다.

```text
Proposer P_t:       문서/검색 → 원 질문 q, 주장 답 a*
Solver S_t:         q → search trajectory T → solver 답 â
Verifier V_t:       (a*, â, 마지막 검색질문 u_k) → 역검색 → 복원 질문 q̂
GRPO reward:        q̂와 숨겨진 q의 semantic/keyword 일치도
다음 proposer:      P_(t+1) ← V_t checkpoint
```

원 질문 `q`은 verifier prompt에 절대 포함되지 않으며, reward metadata에만 canonical target으로 저장됩니다. 마지막 검색질문 하나는 다대일 정보이므로 verifier는 도구 검색을 통해 이전 entity와 relation을 찾아야 합니다.

---

## Verifier 학습 데이터와 보상

`verl.trainer.main_solver_generation`은 solver prediction parquet에 다음을 저장합니다.

| 필드 | 용도 |
|---|---|
| `solver_answer` | solver의 최종 답 `â` |
| `is_correct` | `â`와 주장 답 `a*`의 EM 일치 여부 |
| `solver_response` | 전체 solver trajectory 감사(audit)용 원문 |
| `solver_search_queries` | 순방향 검색 query 목록(JSON) |
| `solver_last_query` | inverse reconstruction의 terminal anchor `u_k` |

`scripts/data_process/verifier_data.py`는 원 질문에서 stopword를 제거한 content/entity/relation keyword 집합을 만들고, 이를 reward metadata에만 저장합니다. Terminal query가 없는 trajectory는 inverse reconstruction 학습에서 제외됩니다.

Verifier는 `<answer>복원 질문</answer>` 형식으로 질문 하나만 출력합니다. 보상은 다음과 같습니다.

\[
R_V = 0.10R_{format} + 0.10R_{think} + 0.10R_{search}
      + 0.30F1_{keyword} + 0.35\,cos_{SBERT} + 0.05R_{no\text{-}answer\ leakage}
\]

- `keyword-F1`: 원 질문과 복원 질문의 정규화 content/entity/relation keyword F1
- `SBERT cosine`: 동의어·표현 변화에 견고한 의미 유사도. 고정(frozen) scorer만 사용
- `search`: tool call 수가 아니라 검색 시도 유무만 보상해, 불필요한 tool-call 나열을 막음
- `no-answer-leakage`: 복원 질문에 기준 답이 그대로 노출되면 감점

Sentence-BERT는 기본적으로 `sentence-transformers/all-MiniLM-L6-v2`를 사용합니다. 폐쇄망에서는 사전에 내려받은 local path를 지정하세요.

```bash
export VERIFIER_SBERT_MODEL=/input/model/all-MiniLM-L6-v2
export VERIFIER_SBERT_DEVICE=cpu  # 별도 GPU가 있으면 cuda로 변경 가능
```

모델을 불러올 수 없으면 코드가 중단되지 않고 keyword-F1 semantic fallback으로 동작하지만, 논문 실험에서는 frozen SBERT checkpoint를 명시적으로 고정하는 것을 권장합니다.

---

## 평가 원칙

이 방법의 핵심 가설은 **역질문 복원으로 학습한 verifier weight가 answer-verdict verifier보다 proposer와 더 잘 정렬된다**는 것입니다. 따라서 기존 answer/verdict 방식에서 얻은 수치를 이 구현의 결과로 사용하면 안 됩니다. 다음 조건을 동일 compute/seed에서 비교해야 합니다.

1. verifier 없음
2. 기존 answer + verdict verifier
3. 마지막 검색질문을 그대로 paraphrase하는 reconstruction baseline
4. 제안 방식: 역검색 기반 inverse question reconstructor

최종 평가는 학습 루프 밖에서 수행합니다. `LLMjudge/`는 reward source가 아니라 질문의 factuality, chain validity, shortcut, 실제 hop 수를 평가하는 외부 평가기로 유지합니다.

---

## 환경 설정

Python 환경에 PyTorch, transformers, faiss-gpu, verl==0.5.0 등이 필요합니다. 의존성 목록은 [여기](https://github.com/volcengine/verl/blob/v0.5.0/requirements.txt)와 [여기](https://github.com/volcengine/verl/blob/v0.5.0/requirements_sglang.txt)를 참조하세요.

---

## 데이터 및 검색 엔진 준비

Wikipedia 영문 덤프를 다운로드하고 retriever 인덱스를 구축합니다 (기본: `intfloat/e5-base-v2`).

```bash
save_path=./corpus
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

초기 학습/평가 데이터 준비:

```bash
python process_train.py --local_dir ./data
python process_test.py --local_dir ./data
```

---

## 학습 실행

각 iteration은 **Proposer → 데이터 생성 → Solver trajectory 생성 → Inverse Question Reconstructor** 순서로 진행됩니다.

### Iteration 1

```bash
# 1. Proposer 학습
bash iter1_challenger.sh

# 2. 학습 데이터 생성
bash iter1_gen_data.sh

# 3. Solver 학습
bash iter1_solver.sh

# 4. Solver 체크포인트 변환 (HF 포맷)
bash convert.sh

# 5. 역질문 복원 verifier 학습 (본 연구 추가 단계)
bash iter1_verifier.sh
```

### Iteration 2, 3...

```bash
# Iter 2
bash iter2_challenger.sh && bash iter2_gen_data.sh && bash iter2_solver.sh && bash convert.sh && bash iter2_verifier.sh

# Iter 3
bash iter3_challenger.sh && bash iter3_gen_data.sh && bash iter3_solver.sh && bash convert.sh && bash iter3_verifier.sh
```

---

## 코드 구조

```
├── iter{1,2,3}_challenger.sh       # Proposer 학습
├── iter{1,2,3}_gen_data.sh         # 학습 데이터 생성
├── iter{1,2,3}_solver.sh           # Solver 학습
├── iter{1,2,3}_verifier.sh         # Inverse-question-reconstruction verifier GRPO
├── verl/custom_reward/
│   ├── reward_function.py          # Proposer 보상 함수
│   └── verifier_reward_function.py # keyword/SBERT 기반 reconstruction 보상
├── verl/trainer/
│   └── main_solver_generation.py   # solver 답·전체 trajectory·마지막 검색질문 저장
├── scripts/data_process/
│   └── verifier_data.py            # hidden-question reconstruction 데이터 생성
├── LLMjudge/                       # 학습과 분리된 외부 질문 품질 평가
└── search/                         # 검색 서버
```

---

## 인용

본 연구의 기반인 DrZero를 인용할 경우:

```bibtex
@article{yue2026dr,
  title={Dr. Zero: Self-Evolving Search Agents without Training Data},
  author={Yue, Zhenrui and Upasani, Kartikeya and Yang, Xianjun and Ge, Suyu and Nie, Shaoliang and Mao, Yuning and Liu, Zhe and Wang, Dong},
  journal={arXiv preprint arXiv:2601.07055},
  year={2026}
}
```

---

## 라이선스

비상업적 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE.md)를 참조하세요.

## 감사의 말

본 코드는 [Search-R1](https://github.com/PeterGriffinJin/Search-R1)과 [VeRL](https://github.com/volcengine/verl)을 기반으로 구현되었습니다.

---

<div align="center">
현대자동차그룹 CVLCM AI팀 | 2026 사내 학술대회
</div>
