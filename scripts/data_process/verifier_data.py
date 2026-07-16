"""Build data for trajectory-conditioned inverse question reconstruction.

The original proposer question is deliberately *not* exposed to the verifier.
Instead, the verifier receives the ground-truth answer, the solver's final
answer, and the solver's final forward search query. It must search backward
through the corpus and reconstruct the source question. The hidden source
question and its deterministic keyword set are retained only in reward_model
metadata for GRPO scoring.
"""

import argparse
import copy
import json
import os
import re

import numpy as np
import pandas as pd


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]*")

# Kept deliberately small: the reward should retain relation words such as
# "founded", "directed", and "located", rather than only named entities.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "has", "have", "how", "in", "is", "it", "of", "on", "or",
    "the", "that", "their", "this", "to", "was", "were", "what", "when", "where",
    "which", "who", "whom", "why", "with", "would",
}

VERIFIER_SYSTEM = (
    "You are a trajectory-conditioned inverse question reconstructor. "
    "The source question is hidden from you. You are given a reference answer, "
    "a solver's final answer, and the solver's final forward search query. "
    "Use the search tool to reason BACKWARD from the answer and terminal query "
    "toward the entities and relations that could have formed the source question. "
    "Do not merely paraphrase the terminal query and do not reveal the answer in "
    "the reconstructed question. Think in <think>...</think>; make evidence-seeking "
    "search calls with <tool_call>...</tool_call>. Finally output exactly one natural "
    "language question and nothing else in <answer>RECONSTRUCTED QUESTION</answer>."
)


def normalize_space(text: str) -> str:
    return " ".join(str(text).split())


def extract_keywords(text: str) -> list[str]:
    """Deterministically extract content/entity/relation tokens for reward scoring."""
    seen = set()
    keywords = []
    for token in TOKEN_PATTERN.findall(str(text).lower()):
        token = token.strip("'-")
        if len(token) < 2 or token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        keywords.append(token)
    return keywords


def extract_question(prompt_messages) -> str:
    for msg in prompt_messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            match = re.search(r"Question:\s*(.+)$", content, re.DOTALL)
            if match:
                return normalize_space(match.group(1))
            return normalize_space(content)
    return ""


def extract_ground_truth(row) -> str:
    """Extract the proposer-claimed reference answer from reward_model metadata."""
    try:
        reward_model = row.get("reward_model", {})
        if isinstance(reward_model, dict):
            target = reward_model.get("ground_truth", {}).get("target", [])
            return normalize_space(list(target)[0])
    except (IndexError, KeyError, TypeError):
        pass
    return ""


def extract_last_query(row) -> str:
    """Read the terminal solver query, with a JSON-trajectory compatibility fallback."""
    direct = normalize_space(row.get("solver_last_query", ""))
    if direct:
        return direct

    raw_queries = row.get("solver_search_queries", "")
    try:
        queries = json.loads(raw_queries) if isinstance(raw_queries, str) else raw_queries
        if isinstance(queries, list):
            for query in reversed(queries):
                query = normalize_space(query)
                if query:
                    return query
    except (json.JSONDecodeError, TypeError):
        pass
    return ""


def make_verifier_prompt(
    ground_truth_answer: str,
    solver_answer: str,
    last_query: str,
) -> list:
    content = (
        f"{VERIFIER_SYSTEM}\n\n"
        f"Reference answer: {ground_truth_answer}\n"
        f"Solver final answer: {solver_answer}\n"
        f"Solver final forward search query: {last_query}"
    )
    return [{"role": "user", "content": content}]


def make_row(
    base_row: dict,
    question: str,
    solver_answer: str,
    ground_truth_answer: str,
    is_correct: bool,
    last_query: str,
) -> dict:
    row = copy.deepcopy(base_row)
    keywords = extract_keywords(question)
    row["prompt"] = make_verifier_prompt(
        ground_truth_answer=ground_truth_answer,
        solver_answer=solver_answer,
        last_query=last_query,
    )
    row["reward_model"] = {
        "ground_truth": {
            # `target` remains for backwards-compatible data inspection.
            "target": np.array([ground_truth_answer]),
            "reference_question": np.array([question]),
            "reference_keywords": np.array([json.dumps(keywords)]),
            "solver_last_query": np.array([last_query]),
            "is_correct": np.array([is_correct]),
        },
        "style": "rule",
    }
    row["reference_question"] = question
    row["reference_keywords"] = json.dumps(keywords)
    return row


def main():
    parser = argparse.ArgumentParser()
    # Retained for the existing iteration scripts. The solver prediction parquet
    # already carries the original prompt and therefore supplies all fields here.
    parser.add_argument("--gen_data_path", required=True)
    parser.add_argument("--solver_pred_path", required=True)
    parser.add_argument("--local_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    solver_pred = pd.read_parquet(args.solver_pred_path)
    print(f"[verifier_data] Solver predictions: {len(solver_pred)} samples")

    required_columns = {"solver_answer", "is_correct", "solver_last_query"}
    missing_columns = required_columns.difference(solver_pred.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            "solver_pred_path is missing trajectory fields: "
            f"{missing}. Regenerate it with verl.trainer.main_solver_generation."
        )

    records = []
    n_correct = 0
    n_wrong = 0
    n_missing_terminal_query = 0
    n_skip = 0

    for _, row in solver_pred.iterrows():
        solver_answer = normalize_space(row["solver_answer"])
        is_correct = bool(row["is_correct"])
        last_query = extract_last_query(row)

        if not solver_answer:
            n_skip += 1
            continue
        if not last_query:
            n_missing_terminal_query += 1
            continue

        question = extract_question(row["prompt"])
        ground_truth_answer = extract_ground_truth(row)
        if not question or not ground_truth_answer:
            n_skip += 1
            continue

        records.append(
            make_row(
                row.to_dict(),
                question=question,
                solver_answer=solver_answer,
                ground_truth_answer=ground_truth_answer,
                is_correct=is_correct,
                last_query=last_query,
            )
        )
        if is_correct:
            n_correct += 1
        else:
            n_wrong += 1

    print(
        "[verifier_data] "
        f"correct={n_correct}, wrong={n_wrong}, "
        f"missing_terminal_query={n_missing_terminal_query}, skipped={n_skip}"
    )

    train_df = pd.DataFrame(records)
    if train_df.empty:
        raise ValueError("No inverse-reconstruction records were created.")

    if args.max_samples and args.max_samples < len(train_df):
        if "data_source" in train_df.columns:
            sampled = train_df.groupby("data_source", group_keys=False).apply(
                lambda group: group.sample(
                    n=max(1, round(args.max_samples * len(group) / len(train_df))),
                    random_state=42,
                )
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
    print(f"[verifier_data] Saved {len(train_df)} samples -> {out_path}")


if __name__ == "__main__":
    main()
