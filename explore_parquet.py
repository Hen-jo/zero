import pandas as pd
import json

FPATH = r"results/zero_challenger_iter1_ratio4321_grpo_batch_group1-5_qwen2.5-3b-instruct (2).parquet"
df = pd.read_parquet(FPATH)
print(pd.__version__)
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}\n")

def show(row_idx=0):
    row = df.iloc[row_idx]
    print(f"{'='*60} Row {row_idx} {'='*60}")

    # data_source
    print(f"\n[data_source]\n  {row['data_source']}")

    # ability
    print(f"\n[ability]\n  {row['ability']}")

    # prompt
    prompt = row['prompt']
    if hasattr(prompt, 'tolist'):
        prompt = prompt.tolist()
    print(f"\n[prompt]")
    for msg in prompt:
        print(f"  role: {msg['role']}")
        print(f"  content: {msg['content'][:]}")

    # reward_model
    rm = row['reward_model']
    print(f"\n[reward_model]")
    print(f"  ground_truth: {rm['ground_truth']}")
    print(f"  style: {rm['style']}")

    # extra_info
    print(f"\n[extra_info]\n  {row['extra_info']}")

    # metadata
    meta = row['metadata']
    if isinstance(meta, dict):
        raw = meta.get('raw_context', '')
        print(f"\n[metadata.raw_context] (first 300 chars)\n  {str(raw)[:300]}")
    else:
        print(f"\n[metadata]\n  {meta}")

def show_col(col, n=5):
    """특정 컬럼 n개 샘플 출력"""
    print(f"\n{'='*40} {col} (n={n}) {'='*40}")
    for i in range(min(n, len(df))):
        val = df.iloc[i][col]
        if hasattr(val, 'tolist'):
            val = val.tolist()
        print(f"  [{i}] {str(val)[:200]}")

def show_questions(n=10):
    """질문만 빠르게 확인"""
    print(f"\n{'='*40} Questions (n={n}) {'='*40}")
    for i in range(min(n, len(df))):
        prompt = df.iloc[i]['prompt']
        if hasattr(prompt, 'tolist'):
            prompt = prompt.tolist()
        content = prompt[0]['content']
        question = content.split('Question:')[-1].strip() if 'Question:' in content else content[-150:]
        target = df.iloc[i]['reward_model']['ground_truth']['target']
        src = df.iloc[i]['data_source']
        print(f"  [{i}][{src}] Q: {question[:120]}")
        print(f"         A: {target}")

def check_source_doc_contamination(n=100):
    """'source document' 표현이 질문에 포함된 비율"""
    keywords = ['source document', 'the document', 'in the text', 'mentioned in']
    counts = {k: 0 for k in keywords}
    sample = min(n, len(df))
    for i in range(sample):
        prompt = df.iloc[i]['prompt']
        if hasattr(prompt, 'tolist'):
            prompt = prompt.tolist()
        content = prompt[0]['content'].lower()
        for k in keywords:
            if k in content:
                counts[k] += 1
    print(f"\n{'='*40} Source doc contamination (n={sample}) {'='*40}")
    for k, v in counts.items():
        print(f"  '{k}': {v}/{sample} ({100*v/sample:.1f}%)")


if __name__ == "__main__":
    # 기본 실행 시 전체 개요
    show(0)
    print()
    show_questions(10)
    print()
    check_source_doc_contamination(1000)
