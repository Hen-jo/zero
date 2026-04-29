"""
verifier_data.py

Challenger gen data + solver last-rollout predictions → verifier train.parquet

Logic:
  1. Load challenger gen data (questions + ground truth).
  2. Load solver_pred (last batch of solver rollout, 128 responses).
  3. Check EM for each solver response vs ground truth.
  4. Report solver accuracy and build verifier training set from the full
     challenger gen data (same prompt/reward_model format, usable by GRPO).
"""

import argparse
import os
import re
import string

import numpy as np
import pandas as pd

ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    pred_norm = normalize_answer(prediction)
    for ga in golden_answers:
        if normalize_answer(str(ga)) == pred_norm:
            return 1
    return 0


def extract_answer(response: str) -> str:
    matches = ANSWER_PATTERN.findall(response)
    return matches[-1].strip() if matches else response.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_data_path", required=True,
                        help="challenger gen data parquet (zero_challenger_*.parquet)")
    parser.add_argument("--solver_pred_path", required=True,
                        help="solver last-rollout parquet (solver_pred_*.parquet)")
    parser.add_argument("--local_dir", required=True,
                        help="output directory for train.parquet")
    args = parser.parse_args()

    gen_data = pd.read_parquet(args.gen_data_path)
    solver_pred = pd.read_parquet(args.solver_pred_path)

    n_gen = len(gen_data)
    n_solver = len(solver_pred)
    print(f"[verifier_data] Challenger gen data : {n_gen} samples")
    print(f"[verifier_data] Solver pred          : {n_solver} samples")

    # ── Solver accuracy on last batch ─────────────────────────────────────
    # Solver processes data sequentially (shuffle=False).
    # Last batch corresponds to gen_data rows [n_gen - n_solver : n_gen]
    # if epochs > 1, otherwise rows [(last_step-1)*bs : last_step*bs].
    # We use the tail of gen_data as the best approximation.
    matched_gen = gen_data.tail(n_solver).reset_index(drop=True)
    responses = solver_pred["responses"].tolist()

    correct = 0
    for resp, row in zip(responses, matched_gen.itertuples(index=False)):
        try:
            gt = list(row.reward_model["ground_truth"]["target"])
        except Exception:
            continue
        pred = extract_answer(resp)
        correct += em_check(pred, gt)

    print(f"[verifier_data] Solver EM on last batch: {correct}/{n_solver} "
          f"= {correct / max(n_solver, 1):.2%}")

    # ── Build verifier training set ───────────────────────────────────────
    # The verifier is trained with GRPO on the same question format.
    # It generates backward-reasoning traces; compute_verifier_score rewards
    # traces that correctly verify solver outputs.
    # We use the full challenger gen data so the verifier sees diverse questions.
    train_data = gen_data.copy()

    os.makedirs(args.local_dir, exist_ok=True)
    out_path = os.path.join(args.local_dir, "train.parquet")
    train_data.to_parquet(out_path, index=False)
    print(f"[verifier_data] Saved {len(train_data)} verifier training samples → {out_path}")


if __name__ == "__main__":
    main()
