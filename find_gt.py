import pandas as pd

FPATH = r"results/zero_challenger_iter1_ratio4321_grpo_batch_group1-5_qwen2.5-3b-instruct (2).parquet"
df = pd.read_parquet(FPATH)

keywords = [
    "garter snake",
    "Santi Crisogono",
]

for kw in keywords:
    hits = []
    for i, row in df.iterrows():
        prompt = row['prompt']
        if hasattr(prompt, 'tolist'):
            prompt = prompt.tolist()
        content = prompt[0]['content'] if prompt else ''
        if kw.lower() in content.lower():
            gt = row['reward_model']['ground_truth']['target']
            hits.append((i, content[-200:], gt))
    print(f"\n=== '{kw}' ===")
    for idx, q, gt in hits[:3]:
        print(f"  [{idx}] Q: ...{q}")
        print(f"        GT: {gt}")
