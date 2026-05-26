"""
run_analysis.py

Runs LLM judge + HNR computation on train.parquet.

Usage:
  python run_analysis.py [--n N] [--model MODEL] [--output OUTPUT]

  --n       : samples per hop level for LLM judge (default 100)
  --model   : single model key (default: gpt-4.1-mini)
  --output  : output parquet path (default: results/train_analysis_results.parquet)
"""
import argparse
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from judge import Judge, MultiModelJudge
from config import DEFAULT_JUDGE_MODELS


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_question_from_verifier_prompt(prompt_field) -> str:
    """Extract question text from verifier training prompt."""
    try:
        content = prompt_field[0].get("content", "") if isinstance(prompt_field[0], dict) else str(prompt_field[0])
    except (IndexError, TypeError):
        content = str(prompt_field)
    # verifier prompt format: "...\nQuestion: {q}\nProposed answer: {a}"
    m = re.search(r"Question:\s*(.+?)(?:\nProposed answer:|$)", content, re.DOTALL)
    if m:
        return m.group(1).strip()
    return content.strip()


def judge_row(args) -> dict:
    idx, question, answer, judge = args
    result = judge.judge(question, answer)
    return {"idx": idx, **result.to_dict()}


# ── HNR Computation ───────────────────────────────────────────────────────────

def compute_hnr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute HNR statistics per hop_instruction from merged data.

    A question passes HNR iff:
      - follows_hop=True (proposer issued the correct number of searches)
      - redundant_hop=False (gold answer not in intermediate retrieval steps)

    HNR = |{q : follows_hop AND NOT redundant_hop}| / |Q|
    """
    rows = []
    for hop in sorted(df["hop_instruction"].dropna().unique()):
        sub = df[df["hop_instruction"] == hop]
        n_total = len(sub)
        n_follows = sub["follows_hop"].sum()
        n_redundant_given_follows = sub[sub["follows_hop"] == True]["redundant_hop"].sum()
        n_genuine = n_follows - n_redundant_given_follows  # follows AND non-redundant

        hnr = n_genuine / n_total if n_total > 0 else float("nan")
        rows.append({
            "hop_instruction": int(hop),
            "n_total": n_total,
            "n_follows_hop": int(n_follows),
            "follows_hop_rate": round(n_follows / n_total, 4) if n_total else float("nan"),
            "n_redundant (given follows)": int(n_redundant_given_follows),
            "HNR": round(hnr, 4),
        })
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100,
                        help="Samples per hop level for LLM judge (default 100)")
    parser.add_argument("--model", default="gpt-4.1-mini",
                        help="Single model for judging (default gpt-4.1-mini). "
                             "Use 'ensemble' for 3-model average.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip LLM judge, only compute HNR")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    train_path = root / "results" / "train.parquet"
    ana_path   = root / "results" / "challenger_iter1_analysis.csv"

    # ── Load & Merge ──────────────────────────────────────────────────────────
    print("Loading train.parquet ...")
    df_train = pd.read_parquet(train_path)
    df_train["idx"] = df_train["extra_info"].apply(
        lambda x: x.get("index") if isinstance(x, dict) else None
    )
    print(f"  {len(df_train):,} verifier training rows")

    print("Loading challenger_iter1_analysis.csv ...")
    df_ana = pd.read_csv(ana_path)
    print(f"  {len(df_ana):,} proposer generation rows")

    df = df_train.merge(df_ana, on="idx", how="left")
    df["extracted_question"] = df["generated_question"].fillna(
        df["prompt"].apply(extract_question_from_verifier_prompt)
    )
    df["extracted_answer"] = df["generated_answer"]
    print(f"  Merged: {len(df):,} rows")
    print()

    # ── HNR ───────────────────────────────────────────────────────────────────
    print("=" * 55)
    print("HNR COMPUTATION (unique proposer questions)")
    print("=" * 55)
    uniq = df.drop_duplicates("idx").copy()
    hnr_df = compute_hnr(uniq)
    print(hnr_df.to_string(index=False))
    print()
    # Overall
    n_total = len(uniq)
    n_genuine = (uniq["follows_hop"] & ~uniq["redundant_hop"]).sum()
    print(f"Overall HNR: {n_genuine}/{n_total} = {n_genuine/n_total:.4f}")
    print()

    if args.skip_judge:
        print("Skipping LLM judge (--skip-judge).")
        return

    # ── Sample for LLM judge ──────────────────────────────────────────────────
    print(f"Sampling {args.n} questions per hop level for LLM judge ...")
    uniq_valid = uniq.dropna(subset=["extracted_question", "extracted_answer"])
    samples = []
    for hop in sorted(uniq_valid["hop_instruction"].dropna().unique()):
        sub = uniq_valid[uniq_valid["hop_instruction"] == hop]
        k = min(args.n, len(sub))
        samples.append(sub.sample(k, random_state=args.seed))
    sample_df = pd.concat(samples, ignore_index=True)
    print(f"  Total sample: {len(sample_df):,} questions")
    print(f"  Per hop: {sample_df['hop_instruction'].value_counts().sort_index().to_dict()}")
    print()

    # ── Build judge ───────────────────────────────────────────────────────────
    if args.model == "ensemble":
        judge = MultiModelJudge()
        mode_label = f"ensemble({', '.join(DEFAULT_JUDGE_MODELS)})"
    else:
        judge = Judge(model_key=args.model)
        mode_label = args.model

    print(f"Judging {len(sample_df):,} rows | model={mode_label} | workers={args.workers} ...")

    tasks = [
        (i, sample_df.at[i, "extracted_question"],
            str(sample_df.at[i, "extracted_answer"]), judge)
        for i in range(len(sample_df))
    ]

    all_results: list[dict] = [None] * len(tasks)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(judge_row, task): task[0] for task in tasks}
        with tqdm(total=len(tasks), desc="Judging") as pbar:
            for future in as_completed(futures):
                i = futures[future]
                try:
                    all_results[i] = future.result()
                except Exception as e:
                    all_results[i] = {"idx": i, "score": None,
                                      "parse_error": True, "reasoning": str(e)}
                pbar.update(1)
    elapsed = time.time() - t0

    result_df = pd.DataFrame(all_results).drop(
        columns=["idx", "question", "answer"], errors="ignore"
    )
    for col in result_df.columns:
        sample_df[col] = result_df[col].values

    # ── Summary ───────────────────────────────────────────────────────────────
    valid = sample_df[sample_df["score"].notna()]
    print()
    print("=" * 55)
    print(f"LLM JUDGE RESULTS  valid={len(valid):,}  time={elapsed:.1f}s")
    print("=" * 55)
    print(f"  Overall score: mean={valid['score'].mean():.3f}  std={valid['score'].std():.3f}")
    print()
    print("  Per hop_instruction:")
    for hop in sorted(valid["hop_instruction"].dropna().unique()):
        sub = valid[valid["hop_instruction"] == hop]
        print(f"    hop={int(hop)}: n={len(sub)}  mean={sub['score'].mean():.3f}  std={sub['score'].std():.3f}")

    print()
    print("  Per data_source:")
    for ds in sorted(valid["data_source_x"].dropna().unique()):
        sub = valid[valid["data_source_x"] == ds]
        print(f"    {ds}: n={len(sub)}  mean={sub['score'].mean():.3f}")

    # Score tier distribution
    bins = [(0.8, 1.01, "High [0.8~1.0]"),
            (0.5, 0.80, "Mid  [0.5~0.8)"),
            (0.0, 0.50, "Low  [0.0~0.5)")]
    print()
    print("  Score distribution:")
    for lo, hi, label in bins:
        cnt = ((valid["score"] >= lo) & (valid["score"] < hi)).sum()
        pct = cnt / len(valid) * 100
        print(f"    {label}: {cnt:4,} ({pct:5.1f}%)")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = root / "LLMjudge" / "results"
    out_dir.mkdir(exist_ok=True)
    if args.output:
        out_path = Path(args.output)
    else:
        safe_model = args.model.replace("/", "_").replace(".", "-")
        out_path = out_dir / f"train_{safe_model}_judge_results.parquet"

    sample_df.to_parquet(out_path, index=False)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
