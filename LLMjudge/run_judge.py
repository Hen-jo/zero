"""
Batch LLM judge runner.

Usage:
  python run_judge.py [OPTIONS]

Examples:
  # Ensemble judge (claude-sonnet-4-6 + gpt-5.4 + gemini-3.1-pro-preview), first 100 rows
  python run_judge.py --n 100

  # Single model (faster / cheaper)
  python run_judge.py --single-model gpt-4.1-mini --n 100

  # Custom input / output
  python run_judge.py --input path/to/data.parquet --output results/my_judge.parquet

Output columns:
  extracted_question          — cleaned question text
  extracted_answer            — ground-truth answer string
  score                       — ensemble average (0.0~1.0)
  score_<model>               — per-model score
  reasoning_<model>           — per-model reasoning
  parse_error_<model>         — per-model parse error flag
"""
import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from config import DEFAULT_JUDGE_MODELS
from judge import MultiModelJudge, Judge


# ── Helpers ───────────────────────────────────────────────────────────────────

HOP_MAP = {"search_zero_1": 1, "search_zero_2": 2, "search_zero_3": 3, "search_zero_4": 4}
PASSAGE_MAX = 400  # chars per retrieved passage shown to judge


def extract_question(prompt_field) -> str:
    try:
        item = prompt_field[0]
        content = item.get("content", "") if isinstance(item, dict) else str(item)
    except (IndexError, KeyError, TypeError):
        content = str(prompt_field)
    idx = content.rfind("Question:")
    if idx != -1:
        return content[idx + len("Question:"):].strip()
    return content.strip()


def extract_answer(reward_model_field) -> str:
    try:
        targets = reward_model_field["ground_truth"]["target"]
        if isinstance(targets, (list, tuple)):
            return ", ".join(str(t) for t in targets)
        return str(targets)
    except (KeyError, TypeError):
        return str(reward_model_field)


def _parse_query(call_raw: str) -> str:
    """Parse query from tool_call content (JSON or plain text)."""
    text = call_raw.strip()
    # Try JSON format: {"name": "search", "arguments": {"query_list": [...]}}
    try:
        obj = json.loads(text)
        queries = obj.get("arguments", {}).get("query_list", [])
        if queries:
            return "; ".join(str(q) for q in queries)
    except Exception:
        pass
    # Plain text format (baseline model): just return the text
    return text[:200]


def extract_trace(metadata_field, data_source: str) -> tuple[str | None, int]:
    """Extract formatted generation trace and hop_n from raw_context.

    Returns (trace_str, hop_n). trace_str is None if extraction fails.
    """
    hop_n = HOP_MAP.get(data_source, 2)
    try:
        raw = metadata_field.get("raw_context", "") if isinstance(metadata_field, dict) else ""
        if not raw:
            return None, hop_n

        # Extract assistant turn
        turns = re.split(r"<\|im_start\|>", raw)
        asst = next((t[9:].strip() for t in turns if t.startswith("assistant")), "")
        if not asst:
            return None, hop_n

        # tool_response (both baseline and trained model use this tag)
        responses = re.findall(r"<tool_response>(.*?)</tool_response>", asst, re.DOTALL)
        if not responses:
            return None, hop_n

        # tool_call — JSON (trained) or plain text (baseline)
        calls = re.findall(r"<tool_call>(.*?)</tool_call>", asst, re.DOTALL)
        # Pad calls with empty strings if fewer than responses
        while len(calls) < len(responses):
            calls.append("")

        lines = []
        for i, (call_raw, resp_raw) in enumerate(zip(calls, responses), 1):
            label = "intermediate" if i < len(responses) else "final"
            query_str = _parse_query(call_raw) if call_raw.strip() else "(query not captured)"
            passage = resp_raw.strip()[:PASSAGE_MAX]
            if len(resp_raw.strip()) > PASSAGE_MAX:
                passage += "..."
            lines.append(f"[Search {i} — {label}]")
            lines.append(f"  Query: {query_str}")
            lines.append(f"  Retrieved: {passage}")
            lines.append("")

        return "\n".join(lines).strip(), hop_n

    except Exception:
        return None, hop_n


def judge_row(args) -> dict:
    idx, question, answer, trace, hop_n, judge = args
    result = judge.judge(question, answer, trace=trace, hop_n=hop_n)
    return {"idx": idx, **result.to_dict()}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM Judge for multi-hop question quality")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--single-model", default=None,
        help="Use a single model instead of the 3-model ensemble (e.g. gpt-4.1-mini)"
    )
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel rows to judge simultaneously (default: 8)")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--source-filter", default=None)
    args = parser.parse_args()

    # ── Input ─────────────────────────────────────────────────────────────────
    if args.input:
        input_path = Path(args.input)
    else:
        input_path = (
            Path(__file__).parent.parent / "results" /
            "zero_challenger_iter1_ratio4321_grpo_batch_group1-5_qwen2.5-3b-instruct (2).parquet"
        )
    if not input_path.exists():
        print(f"[ERROR] Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # ── Output ────────────────────────────────────────────────────────────────
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    if args.output:
        output_path = Path(args.output)
    elif args.single_model:
        safe = args.single_model.replace("/", "_").replace(".", "-")
        output_path = results_dir / f"{safe}_judge_results.parquet"
    else:
        output_path = results_dir / "ensemble_judge_results.parquet"

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"Loading {input_path} ...")
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df):,} rows.")

    if args.source_filter:
        df = df[df["data_source"] == args.source_filter].reset_index(drop=True)
        print(f"After source filter '{args.source_filter}': {len(df):,} rows.")
    if args.offset:
        df = df.iloc[args.offset:].reset_index(drop=True)
    if args.n:
        df = df.iloc[: args.n].reset_index(drop=True)
        print(f"Using first {len(df):,} rows.")

    # ── Extract Q/A/Trace ─────────────────────────────────────────────────────
    print("Extracting questions, answers, and generation traces ...")
    df["extracted_question"] = df["prompt"].apply(extract_question)
    df["extracted_answer"]   = df["reward_model"].apply(extract_answer)

    traces, hop_ns = [], []
    for _, row in df.iterrows():
        trace, hop_n = extract_trace(row.get("metadata"), row.get("data_source", ""))
        traces.append(trace)
        hop_ns.append(hop_n)
    df["trace"] = traces
    df["hop_n"] = hop_ns

    n_with_trace = sum(1 for t in traces if t)
    print(f"  trace available: {n_with_trace:,} / {len(df):,} rows")

    # ── Build judge ───────────────────────────────────────────────────────────
    if args.single_model:
        judge = Judge(model_key=args.single_model)
        mode_label = args.single_model
    else:
        judge = MultiModelJudge()
        mode_label = f"ensemble({', '.join(DEFAULT_JUDGE_MODELS)})"

    print(f"Judging {len(df):,} rows  mode={mode_label}  workers={args.workers} ...")

    tasks = [
        (i, df.at[i, "extracted_question"], df.at[i, "extracted_answer"],
         df.at[i, "trace"], int(df.at[i, "hop_n"]), judge)
        for i in range(len(df))
    ]

    # Collect all result dicts; we'll figure out columns from the first result
    all_results: list[dict] = [None] * len(df)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(judge_row, task): task[0] for task in tasks}
        with tqdm(total=len(tasks), desc="Judging") as pbar:
            for future in as_completed(futures):
                i = futures[future]
                try:
                    all_results[i] = future.result()
                except Exception as e:
                    all_results[i] = {"idx": i, "score": None, "parse_error": True,
                                      "reasoning": f"EXCEPTION: {e}"}
                pbar.update(1)
    elapsed = time.time() - t0

    # Merge results into df
    result_df = pd.DataFrame(all_results).drop(columns=["idx", "question", "answer"],
                                                errors="ignore")
    for col in result_df.columns:
        df[col] = result_df[col].values

    # ── Summary ───────────────────────────────────────────────────────────────
    valid = df[df["score"].notna()]
    n_err = df["score"].isna().sum()
    print(f"\n{'='*55}")
    print(f"Results  valid={len(valid):,}  errors={n_err:,}  time={elapsed:.1f}s")
    print(f"{'='*55}")
    if len(valid) > 0:
        print(f"  score avg: {valid['score'].mean():.3f}  std: {valid['score'].std():.3f}")

        # Per-model breakdown (ensemble only)
        for col in df.columns:
            if col.startswith("score_") and col != "score":
                model_label = col[len("score_"):]
                sub = df[df[col].notna()]
                if len(sub):
                    print(f"  {model_label}: {sub[col].mean():.3f}")

        if "data_source" in df.columns:
            print(f"\n  Per data_source:")
            for src in sorted(df["data_source"].dropna().unique()):
                sub = df[(df["data_source"] == src) & df["score"].notna()]
                if len(sub):
                    print(f"    {src}: avg={sub['score'].mean():.3f}  (n={len(sub):,})")

    # ── Save ──────────────────────────────────────────────────────────────────
    df.to_parquet(output_path, index=False)
    print(f"\nSaved → {output_path}")


if __name__ == "__main__":
    main()
