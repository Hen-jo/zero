"""
Analyze judge results parquet.

Usage:
  python analyze_results.py results/gpt-4-1-mini_judge_results.parquet
  python analyze_results.py results/gpt-4-1-mini_judge_results.parquet --examples 5
"""
import argparse
import sys
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to judge results parquet")
    parser.add_argument("--examples", type=int, default=3, help="Examples per score tier")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(path)
    valid = df[~df["parse_error"].fillna(False)].copy()

    print(f"\nTotal rows:     {len(df):,}")
    print(f"Valid (parsed): {len(valid):,}")
    print(f"Parse errors:   {df['parse_error'].fillna(False).sum():,}")

    if len(valid) == 0:
        return

    print(f"\n{'='*50}")
    print("SCORE DISTRIBUTION")
    print(f"{'='*50}")
    print(f"  mean: {valid['score'].mean():.3f}")
    print(f"  std:  {valid['score'].std():.3f}")
    print(f"  min:  {valid['score'].min():.3f}")
    print(f"  max:  {valid['score'].max():.3f}")

    bins = [(0.8, 1.01, "High  [0.8~1.0]"),
            (0.5, 0.8,  "Mid   [0.5~0.8)"),
            (0.0, 0.5,  "Low   [0.0~0.5)")]
    for lo, hi, label in bins:
        cnt = ((valid["score"] >= lo) & (valid["score"] < hi)).sum()
        pct = cnt / len(valid) * 100
        print(f"  {label}: {cnt:5,} ({pct:5.1f}%)")

    if "data_source" in valid.columns:
        print(f"\n{'='*50}")
        print("PER DATA_SOURCE")
        print(f"{'='*50}")
        for src in sorted(valid["data_source"].dropna().unique()):
            sub = valid[valid["data_source"] == src]
            print(f"  {src}  n={len(sub):5,}  mean={sub['score'].mean():.3f}  std={sub['score'].std():.3f}")

    print(f"\n{'='*50}")
    print(f"EXAMPLES (top {args.examples} per tier)")
    print(f"{'='*50}")
    for lo, hi, label in bins:
        subset = valid[(valid["score"] >= lo) & (valid["score"] < hi)]
        if subset.empty:
            continue
        print(f"\n--- {label} ---")
        for _, row in subset.head(args.examples).iterrows():
            print(f"  Q: {str(row['extracted_question'])[:150]}")
            print(f"  A: {row['extracted_answer']}")
            print(f"  score={row['score']:.2f}  {str(row['reasoning'])[:150]}")
            print()


if __name__ == "__main__":
    main()
