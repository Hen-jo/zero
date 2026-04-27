"""
pandas 2.x vs 3.x 동작 차이 메커니즘 분석
"""
import numpy as np
import pandas as pd

print(f"pandas version: {pd.__version__}")
print(f"CoW enabled: {pd.options.mode.copy_on_write}\n")

df = pd.DataFrame({
    "prompt": [
        np.array([{"role": "user", "content": "ORIGINAL QUESTION"}]),
    ],
    "reward_model": [
        {"ground_truth": {"target": ["original_answer"]}, "style": "rule"},
    ],
})

# 1. iloc이 view인지 copy인지
row = df.iloc[0]
print(f"[1] row is view? shares memory with df:")
print(f"    id(df['prompt'].values) = {id(df['prompt'].values)}")
print(f"    id(row.values)          = {id(row.values)}")
print(f"    np.shares_memory: {np.shares_memory(df['prompt'].values, row.values)}")

# 2. attribute 할당 시 __setattr__ → __setitem__ 라우팅 여부
print(f"\n[2] 'prompt' in row.index? {('prompt' in row.index)}")

# series.__setattr__ 소스 확인
import inspect
src = inspect.getsource(pd.Series.__setattr__)
# __setitem__ 호출 여부
routes_to_setitem = "self[name] = value" in src or "self.__setitem__" in src
print(f"    Series.__setattr__ routes unknown attrs to __setitem__? {routes_to_setitem}")

# 3. 실제 테스트
new_val = np.array([{"role": "user", "content": "NEW QUESTION"}])
row.prompt = new_val  # attribute 할당

print(f"\n[3] After row.prompt = new_val:")
print(f"    df['prompt'][0] = {df.iloc[0]['prompt']}")
updated = df.iloc[0]['prompt'][0]['content'] == "NEW QUESTION"
print(f"    DataFrame updated? {updated}")
