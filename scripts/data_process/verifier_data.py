"""
verifier_data.py  —  Option C: Answer + Verdict

Verifier task:
  Input : question + solver's proposed answer
  Output: <answer>CORRECT_ANSWER. yes/no</answer>

ground_truth = actual answer string (e.g. "Auburn University")
  → em_check(factual_answer, ground_truth) 로 reward 계산
"""

import argparse
import os
import re
import copy

import numpy as np
import pandas as pd

VERIFIER_SYSTEM = (
    "You are given a question and a proposed answer. "
    "To verify, reason BACKWARD: start from the proposed answer and trace the reasoning "
    "chain in reverse using search — do NOT start from the question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "For each backward hop, call a search engine by <tool_call> query </tool_call> and it will return "
    "the top searched results between <tool_response> and </tool_response>. "
    "Search as many times as needed to trace back to the question's starting entity. "
    "After backward verification, output the correct answer followed by your judgment: "
    "<answer>CORRECT_ANSWER. yes</answer> if the proposed answer is correct, "
    "or <answer>CORRECT_ANSWER. no</answer> if it is wrong. "
    "CORRECT_ANSWER must be the actual correct answer based on your backward search."
)


def extract_question(prompt_messages) -> str:
    for msg in prompt_messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            m = re.search(r"Question:\s*(.+)$", content, re.DOTALL)
            if m:
                return m.group(1).strip()
            return content.strip()
    return ""


def extract_ground_truth(row) -> str:
    """reward_model 필드에서 실제 정답 추출."""
    try:
        rm = row.get("reward_model", {})
        if isinstance(rm, dict):
            target = rm.get("ground_truth", {}).get("target", [])
            return str(list(target)[0]).strip()
    except Exception:
        pass
    return ""


def make_verifier_prompt(question: str, proposed_answer: str) -> list:
    content = (
        f"{VERIFIER_SYSTEM}\n\n"
        f"Question: {question}\n"
        f"Proposed answer: {proposed_answer}"
    )
    return [{"role": "user", "content": content}]


def make_row(base_row: dict, question: str, proposed_answer: str, ground_truth_answer: str, is_correct: bool) -> dict:
    row = copy.deepcopy(base_row)
    row["prompt"] = make_verifier_prompt(question, proposed_answer)
    row["reward_model"] = {
        "ground_truth": {
            "target": np.array([ground_truth_answer]),
            "is_correct": np.array([is_correct]),
        },
        "style": "rule",
    }
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_data_path", required=True)
    parser.add_argument("--solver_pred_path", required=True)
    parser.add_argument("--local_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    solver_pred = pd.read_parquet(args.solver_pred_path)
    print(f"[verifier_data] Solver predictions: {len(solver_pred)} samples")

    if "solver_answer" not in solver_pred.columns or "is_correct" not in solver_pred.columns:
        raise ValueError(
            "solver_pred_path must have 'solver_answer' and 'is_correct' columns."
        )

    records = []
    n_correct = 0
    n_wrong = 0
    n_skip = 0

    for _, row in solver_pred.iterrows():
        solver_answer = str(row["solver_answer"]).strip()
        is_correct = bool(row["is_correct"])

        if not solver_answer:
            n_skip += 1
            continue

        question = extract_question(row["prompt"])
        if not question:
            n_skip += 1
            continue

        gt_answer = extract_ground_truth(row)
        if not gt_answer:
            n_skip += 1
            continue

        records.append(make_row(row.to_dict(), question, solver_answer, gt_answer, is_correct))
        if is_correct:
            n_correct += 1
        else:
            n_wrong += 1

    print(f"[verifier_data] correct={n_correct}, wrong={n_wrong}, skipped={n_skip}")

    train_df = pd.DataFrame(records)

    if args.max_samples and args.max_samples < len(train_df):
        if "data_source" in train_df.columns:
            sampled = (
                train_df.groupby("data_source", group_keys=False)
                .apply(lambda g: g.sample(
                    n=max(1, round(args.max_samples * len(g) / len(train_df))),
                    random_state=42
                ))
            )
            if len(sampled) > args.max_samples:
                sampled = sampled.sample(n=args.max_samples, random_state=42)
            train_df = sampled
        else:
            train_df = train_df.sample(n=args.max_samples, random_state=42)
        print(f"[verifier_data] Sampled {len(train_df)} / max_samples={args.max_samples}")

    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs(args.local_dir, exist_ok=True)
    out_path = os.path.join(args.local_dir, "train.parquet")
    train_df.to_parquet(out_path, index=False)
    print(f"[verifier_data] Saved {len(train_df)} samples → {out_path}")


if __name__ == "__main__":
    main()
