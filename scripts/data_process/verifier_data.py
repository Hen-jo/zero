"""
verifier_data.py

Creates verifier training data.

The verifier's task:
  Input : question + solver's proposed answer
  Output: Is this answer correct? → <answer>yes</answer> or <answer>no</answer>

Data sources:
  1. Matched examples  : last-batch solver predictions paired with their questions
                         (EM → yes/no label)
  2. Positive examples : (question, ground_truth) → yes
  3. Negative examples : (question, another_question's_ground_truth) → no
     Negative answers are drawn by shuffling ground-truth answers across samples.
"""

import argparse
import os
import re
import string
import copy

import numpy as np
import pandas as pd

ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

VERIFIER_SYSTEM = (
    "You are given a question and a proposed answer. "
    "Determine whether the proposed answer is correct. "
    "Reason step by step inside <think> and </think>, "
    "then output your judgment as <answer>yes</answer> or <answer>no</answer>."
)


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
            return True
    return False


def extract_answer(response: str) -> str:
    matches = ANSWER_PATTERN.findall(response)
    return matches[-1].strip() if matches else response.strip()


def extract_question(prompt_messages) -> str:
    """Extract the question text from a prompt message list."""
    for msg in prompt_messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            # Try to grab the last sentence after 'Question:'
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
    # Reward model: yes/no label so that em_check in reward fn works directly
    row["reward_model"] = {
        "ground_truth": {"target": np.array(["yes" if is_correct else "no"])},
        "style": "rule",
    }
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_data_path", required=True,
                        help="challenger gen data parquet")
    parser.add_argument("--solver_pred_path", required=True,
                        help="solver last-rollout parquet (responses column)")
    parser.add_argument("--local_dir", required=True,
                        help="output directory for train.parquet")
    parser.add_argument("--neg_ratio", type=float, default=1.0,
                        help="ratio of negative synthetic examples to positive ones")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="solver training batch size (to locate last batch)")
    args = parser.parse_args()

    gen_data = pd.read_parquet(args.gen_data_path)
    solver_pred = pd.read_parquet(args.solver_pred_path)

    n_gen = len(gen_data)
    n_solver = len(solver_pred)
    print(f"[verifier_data] Challenger gen data : {n_gen} samples")
    print(f"[verifier_data] Solver pred          : {n_solver} samples")

    # ── 1. Match solver predictions to their questions ─────────────────────
    # shuffle=False → sequential processing.
    # Last batch = gen_data.iloc[n_gen - n_solver : n_gen]  (tail approximation)
    matched_gen = gen_data.tail(n_solver).reset_index(drop=True)
    responses = solver_pred["responses"].tolist()

    records = []
    n_correct = 0

    for i, (resp, (_, row)) in enumerate(zip(responses, matched_gen.iterrows())):
        try:
            gt = list(row["reward_model"]["ground_truth"]["target"])
        except Exception:
            continue

        question = extract_question(row["prompt"])
        pred = extract_answer(resp)
        is_correct = em_check(pred, gt)
        if is_correct:
            n_correct += 1

        records.append(make_row(row.to_dict(), question, pred, is_correct))

    print(f"[verifier_data] Solver EM on last batch: "
          f"{n_correct}/{n_solver} = {n_correct / max(n_solver, 1):.2%}")

    # ── 2. Synthetic positive examples: (question, ground_truth) → yes ────
    rng = np.random.default_rng(42)
    pos_indices = rng.choice(n_gen, size=min(n_gen, 2000), replace=False)
    for idx in pos_indices:
        row = gen_data.iloc[idx]
        try:
            gt = list(row["reward_model"]["ground_truth"]["target"])
        except Exception:
            continue
        question = extract_question(row["prompt"])
        proposed = str(gt[0]) if gt else ""
        if proposed:
            records.append(make_row(row.to_dict(), question, proposed, True))

    # ── 3. Synthetic negative examples: (question, wrong_answer) → no ─────
    neg_size = int(len(pos_indices) * args.neg_ratio)
    neg_q_indices = rng.choice(n_gen, size=neg_size, replace=True)
    # Wrong answer = ground truth of a different (shuffled) sample
    neg_a_indices = (neg_q_indices + rng.integers(1, n_gen, size=neg_size)) % n_gen
    for qi, ai in zip(neg_q_indices, neg_a_indices):
        row_q = gen_data.iloc[qi]
        row_a = gen_data.iloc[ai]
        try:
            wrong_gt = list(row_a["reward_model"]["ground_truth"]["target"])
            correct_gt = list(row_q["reward_model"]["ground_truth"]["target"])
        except Exception:
            continue
        wrong_ans = str(wrong_gt[0]) if wrong_gt else ""
        # Skip if the wrong answer happens to be correct
        if not wrong_ans or em_check(wrong_ans, correct_gt):
            continue
        question = extract_question(row_q["prompt"])
        records.append(make_row(row_q.to_dict(), question, wrong_ans, False))

    # ── 4. Save ───────────────────────────────────────────────────────────
    train_df = pd.DataFrame(records)
    # Shuffle so yes/no examples are interleaved
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs(args.local_dir, exist_ok=True)
    out_path = os.path.join(args.local_dir, "train.parquet")
    train_df.to_parquet(out_path, index=False)

    n_yes = sum(1 for r in records if "yes" in str(r["reward_model"]["ground_truth"]["target"]))
    n_no = len(records) - n_yes
    print(f"[verifier_data] Training set: {len(train_df)} samples "
          f"(yes={n_yes}, no={n_no}) → {out_path}")


if __name__ == "__main__":
    main()
