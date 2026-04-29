"""
verifier_data.py

Creates verifier training data from full solver predictions.

The verifier's task:
  Input : question + solver's proposed answer
  Output: Is this answer correct? → <answer>yes</answer> or <answer>no</answer>

Data source:
  solver_pred_path : full parquet from main_solver_generation.py
                     has 'solver_answer' and 'is_correct' columns for all 33K+ questions
"""

import argparse
import os
import re
import copy

import numpy as np
import pandas as pd

VERIFIER_SYSTEM = (
    "You are given a question and a proposed answer. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you need to look up information to verify the answer, "
    "you can call a search engine by <tool_call> query </tool_call> and it will return "
    "the top searched results between <tool_response> and </tool_response>. "
    "You can search as many times as you want. "
    "After verification, determine whether the proposed answer is correct and "
    "output your judgment as <answer>yes</answer> or <answer>no</answer>."
)


def extract_question(prompt_messages) -> str:
    """Extract the question text from a prompt message list."""
    for msg in prompt_messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            m = re.search(r"Question:\s*(.+)$", content, re.DOTALL)
            if m:
                return m.group(1).strip()
            return content.strip()
    return ""


def make_verifier_prompt(question: str, proposed_answer: str) -> list:
    content = (
        f"{VERIFIER_SYSTEM}\n\n"
        f"Question: {question}\n"
        f"Proposed answer: {proposed_answer}"
    )
    return [{"role": "user", "content": content}]



def make_row(base_row: dict, question: str, proposed_answer: str, is_correct: bool) -> dict:
    row = copy.deepcopy(base_row)
    row["prompt"] = make_verifier_prompt(question, proposed_answer)
    row["reward_model"] = {
        "ground_truth": {"target": np.array(["yes" if is_correct else "no"])},
        "style": "rule",
    }
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_data_path", required=True,
                        help="challenger gen data parquet (for question extraction fallback)")
    parser.add_argument("--solver_pred_path", required=True,
                        help="full solver prediction parquet (has solver_answer + is_correct)")
    parser.add_argument("--local_dir", required=True,
                        help="output directory for train.parquet")
    args = parser.parse_args()

    solver_pred = pd.read_parquet(args.solver_pred_path)
    print(f"[verifier_data] Solver predictions: {len(solver_pred)} samples")

    if "solver_answer" not in solver_pred.columns or "is_correct" not in solver_pred.columns:
        raise ValueError(
            "solver_pred_path must have 'solver_answer' and 'is_correct' columns. "
            "Run main_solver_generation.py first."
        )

    records = []
    n_yes = 0
    n_no = 0
    n_skip = 0

    for _, row in solver_pred.iterrows():
        solver_answer = str(row["solver_answer"]).strip()
        is_correct = bool(row["is_correct"])

        # Skip rows where solver produced no answer
        if not solver_answer:
            n_skip += 1
            continue

        question = extract_question(row["prompt"])
        if not question:
            n_skip += 1
            continue

        records.append(make_row(row.to_dict(), question, solver_answer, is_correct))
        if is_correct:
            n_yes += 1
        else:
            n_no += 1

    print(f"[verifier_data] yes={n_yes}, no={n_no}, skipped={n_skip}")

    train_df = pd.DataFrame(records)
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs(args.local_dir, exist_ok=True)
    out_path = os.path.join(args.local_dir, "train.parquet")
    train_df.to_parquet(out_path, index=False)
    print(f"[verifier_data] Saved {len(train_df)} samples → {out_path}")


if __name__ == "__main__":
    main()
