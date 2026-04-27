"""
pandas chained assignment 버그 재현 테스트
- .attr = value (attribute 할당) vs ['key'] = value (item 할당) vs dict mutation
"""
import numpy as np
import pandas as pd

print(f"pandas version: {pd.__version__}\n")

# 실제 gendata 구조와 동일하게 세팅
df = pd.DataFrame({
    "prompt": [
        np.array([{"role": "user", "content": "When did Jameela Siddiqi die?"}]),
        np.array([{"role": "user", "content": "Where was X born?"}]),
    ],
    "reward_model": [
        {"ground_truth": {"target": ["original_answer_1"]}, "style": "rule"},
        {"ground_truth": {"target": ["original_answer_2"]}, "style": "rule"},
    ],
    "metadata": [
        {"raw_context": "original_context_1"},
        {"raw_context": "original_context_2"},
    ],
})

new_prompt_val = np.array([{"role": "user", "content": "CHALLENGER GENERATED QUESTION"}])
new_gt_val     = ["challenger_answer"]
new_meta_val   = {"raw_context": "challenger_context"}

print("=" * 60)
print("BEFORE")
print(f"  prompt[0]        : {df.iloc[0]['prompt']}")
print(f"  reward_model[0]  : {df.iloc[0]['reward_model']}")
print(f"  metadata[0]      : {df.iloc[0]['metadata']}")

# --- 원본 buggy 코드 방식 ---
df.iloc[0].prompt = new_prompt_val                                          # attribute 할당
df.iloc[0]['reward_model']['ground_truth']['target'] = new_gt_val           # dict mutation
df.iloc[0].metadata = new_meta_val                                          # attribute 할당

print("\n" + "=" * 60)
print("AFTER (original buggy code)")
print(f"  prompt[0]        : {df.iloc[0]['prompt']}")
print(f"  reward_model[0]  : {df.iloc[0]['reward_model']}")
print(f"  metadata[0]      : {df.iloc[0]['metadata']}")

changed_prompt   = not np.array_equal(df.iloc[0]['prompt'], np.array([{"role": "user", "content": "When did Jameela Siddiqi die?"}]))
changed_gt       = df.iloc[0]['reward_model']['ground_truth']['target'] == new_gt_val
changed_metadata = df.iloc[0]['metadata'] == new_meta_val

print(f"\n  prompt updated?        : {changed_prompt}")
print(f"  reward_model updated?  : {changed_gt}")
print(f"  metadata updated?      : {changed_metadata}")

print("\n" + "=" * 60)
print("DIAGNOSIS")
if not changed_prompt and changed_gt:
    print("  ❌ BUG CONFIRMED: prompt 미갱신 + reward_model 갱신 → Q/A 불일치!")
    print("     → Solver 학습 데이터: 원본 질문 + 챌린저 답변 → EM 항상 0")
elif changed_prompt:
    print("  ✅ prompt도 갱신됨 (예상과 다름, pandas 내부 동작 확인 필요)")
else:
    print("  ❓ 모두 미갱신")
