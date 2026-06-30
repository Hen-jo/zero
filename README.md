<div align="center">

<img src="assets/image1.png" width="700"/>

# 강화학습 기반 자기 대전을 통한 다중 홉 질의응답 역추론 성능 향상

**Don't Just Ask — Verify: Rethinking Reward Signals for Self-Play Multi-hop QA**

조재헝 · 윤재욱 · 이상호 · 하승구 · 정의량 · 나욱진

현대자동차그룹 CVLCM AI팀

[![GitLab](https://img.shields.io/badge/Code-GitLab-orange)](https://gitlab.hmc.co.kr/cvlcmai/2026_hmc_paper/vef_zero)

</div>

---

## 개요

Self-play 프레임워크는 어려운 질문을 생성하는 **제안자(Proposer)**와 이를 푸는 **해결자(Solver)**를 공동 학습시키지만, 제안자의 보상이 단순히 solver의 실패율에만 의존할 경우 **논리적으로 불완전하거나 근거가 약한 질문**을 생성하는 방향으로 학습될 수 있습니다.

본 연구에서는 **역방향 검증자(Backward Verifier)**를 도입합니다. 검증자는 hop 체인을 역방향으로 추적하는 검색 기반 추론을 통해, 답변에서 출발하여 원래 질문의 시작 엔티티까지 복원되지 않는 질문에 대해 제안자에게 패널티를 부여합니다.

---

## 핵심 아이디어

<img src="assets/image2.png" width="650"/>

기존 접근법은 solver가 정답을 맞히는지(순방향)만 검증합니다. 본 연구는 **역방향 검증**을 추가합니다:

- **유효한 질문**: 정답에서 역방향 추적 시 질문의 시작 엔티티까지 회복 가능
- **유효하지 않은 질문**: 역방향 추적 실패 → 제안자에게 낮은 보상

이를 통해 제안자는 논리적으로 완결된 hop 체인을 갖춘 질문을 생성하도록 유도됩니다.

---

## 방법론

### 학습 파이프라인

각 iteration마다 세 단계를 반복합니다:

1. **Proposer 학습**: 문서에서 multi-hop 질문과 답변 생성
2. **Solver 학습**: 검색 기반 추론으로 질문에 답변
3. **Verifier 학습**: 제안된 답변에서 역방향으로 hop 체인 추적, 검증 결과를 proposer 보상에 반영

### Verifier 보상 설계

```
Reward = (1 + think_reward + tool_reward + em_score + verdict_score) / 5
```

- **em_score**: 검색을 통해 찾은 정답과 ground truth의 일치 여부
- **verdict_score**: yes/no 판단의 정확도 (is_correct label과 비교)
- **tool_reward**: 역방향 검색 수행 여부 (단순 yes/no shortcut 방지)

---

## 실험 결과

<img src="assets/image3.png" width="700"/>

7개 QA 벤치마크에서 DrZero 대비 일관된 성능 향상:

| 지표 | DrZero | **Ours** |
|---|---|---|
| 평균 Exact Match | 0.326 | **0.343** |
| LLM-Judge 유효성 | 16.1% | **25.3%** |
| HNR | 60.7% | **69.7%** |

특히 **다중 홉 벤치마크(2Wiki, Bamboogle)**에서 가장 큰 성능 향상을 보입니다.

---

## 코드 구조

```
├── iter{1,2,3}_challenger.sh   # Proposer 학습
├── iter{1,2,3}_solver.sh       # Solver 학습
├── iter{1,2,3}_verifier.sh     # Verifier 학습
├── iter{1,2,3}_gen_data.sh     # 데이터 생성
├── verl/custom_reward/
│   ├── reward_function.py          # Proposer/Challenger 보상 함수
│   └── verifier_reward_function.py # Verifier 보상 함수
├── scripts/data_process/
│   └── verifier_data.py        # Verifier 학습 데이터 생성
├── LLMjudge/                   # LLM 기반 질문 품질 평가
└── search/                     # 검색 서버
```

---

## 환경 설정

```bash
# 저장소 클론
git clone https://gitlab.hmc.co.kr/cvlcmai/2026_hmc_paper/vef_zero.git
cd vef_zero

# 환경 설치
bash setup.sh

# 검색 인덱스 구축
bash search/build_index.sh
```

---

## 학습 실행

```bash
# Iter 1: Proposer → Solver → Verifier 순서로 실행
bash iter1_challenger.sh
bash iter1_solver.sh
bash iter1_verifier.sh

# Iter 2
bash iter2_gen_data.sh   # Proposer가 생성한 질문으로 데이터 구성
bash iter2_challenger.sh
bash iter2_solver.sh
bash iter2_verifier.sh
```

---

## 키워드

Self-Play · 다중 홉 질의응답 · 역방향 검증 · 강화학습 · 검색 증강 생성 · Proposer-Solver

---

<div align="center">
현대자동차그룹 CVLCM AI팀 | 2026 사내 학술대회
</div>
