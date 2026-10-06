"""Build source-disjoint verifier datasets from ID-bearing solver exports.

Every candidate is retained and joined by item_id, never row position. The
labels are normalized answer exact-match proxies, not proofs that backward
reasoning or a multi-hop chain is valid. Legacy positional exports must be
regenerated. A last-training-step export requires explicit --allow_subset.

Only the parquet CLI requires pandas and a parquet engine. The functions and
--help work with the Python standard library, without loading the verl runtime.
"""

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile


def _load_source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_REPO_ROOT = Path(__file__).resolve().parents[2]
_contracts = _load_source_module("_verifier_data_contracts", _REPO_ROOT / "verl/utils/verifier_contracts.py")
_prompts = _load_source_module("_verifier_data_prompts", _REPO_ROOT / "verl/prompts.py")
LABEL_SOURCE = _contracts.LABEL_SOURCE
LABEL_LIMITATION = _contracts.LABEL_LIMITATION


def _plain(value):
    """Convert nested parquet array cells to standard Python containers."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def make_verifier_row(joined):
    """Create a verl row while retaining source, tools, and rollout provenance."""
    original = _plain(joined["original"])
    export = _plain(joined["export"])
    extra_info = copy.deepcopy(original.get("extra_info") or {})
    metadata = copy.deepcopy(original.get("metadata") or {})
    if not isinstance(extra_info, dict) or not isinstance(metadata, dict):
        raise ValueError("generated extra_info and metadata must be mappings")
    tools = copy.deepcopy(extra_info.get("tools_kwargs") or original.get("tools_kwargs") or {})
    if not isinstance(tools, dict):
        raise ValueError("tools_kwargs must be a mapping")
    search = tools.setdefault("search", {})
    if not isinstance(search, dict):
        raise ValueError("search tool kwargs must be a mapping")
    create = search.setdefault("create_kwargs", {})
    if not isinstance(create, dict):
        raise ValueError("search create_kwargs must be a mapping")
    create.update({"data_source": "verifier", "ground_truth": "", "question": joined["question"]})
    interaction = copy.deepcopy(extra_info.get("interaction_kwargs") or original.get("interaction_kwargs") or {})
    if not isinstance(interaction, dict):
        raise ValueError("interaction_kwargs must be a mapping")
    extra_info.update({
        "index": joined["candidate_id"],
        "item_id": joined["item_id"],
        "candidate_id": joined["candidate_id"],
        "source_id": joined["source_id"],
        "tools_kwargs": tools,
        "need_tools_kwargs": True,
    })
    # Parquet cannot serialize an all-empty struct. RLHFDataset supplies {} for
    # omitted interaction_kwargs, so preserve only meaningful interaction data.
    if interaction:
        extra_info["interaction_kwargs"] = interaction
    else:
        extra_info.pop("interaction_kwargs", None)
    metadata.update({
        "schema_version": _contracts.SCHEMA_VERSION,
        "question": joined["question"],
        "reference_answers": joined["reference_answers"],
        "candidate_answer": joined["candidate_answer"],
        "solver_response": joined["responses"],
        "rollout_uid": export["rollout_uid"],
        "rollout_index": export["rollout_index"],
        "global_step": export["global_step"],
        "label_source": LABEL_SOURCE,
        "label_limitation": LABEL_LIMITATION,
    })
    if "full_response_trace" in export:
        metadata["solver_full_response_trace"] = export["full_response_trace"]
    if "response_text_scope" in export:
        metadata["solver_response_text_scope"] = export["response_text_scope"]
    if "training_reward" in export:
        metadata["solver_training_reward"] = export["training_reward"]
    return {
        "item_id": joined["item_id"],
        "candidate_id": joined["candidate_id"],
        "source_id": joined["source_id"],
        "data_source": "verifier",
        "prompt": [{"role": "user", "content": _prompts.DEFAULT_BACKWARD_VERIFIER_PREFIX.format(
            question=joined["question"], candidate_answer=joined["candidate_answer"],
        )}],
        "ability": "backward-verification",
        "reward_model": {"style": "rule", "ground_truth": {
            "is_correct": joined["is_correct"], "label_source": LABEL_SOURCE,
        }},
        "extra_info": extra_info,
        "metadata": metadata,
    }


def build_verifier_dataset(generated_rows, candidate_rows, *, allow_subset=False, eval_fraction=0.1, seed=0):
    """Return (train_rows, val_rows, manifest), without pandas or side effects."""
    generated_rows = [_plain(row) for row in generated_rows]
    candidate_rows = [_plain(row) for row in candidate_rows]
    joined = _contracts.join_candidates(generated_rows, candidate_rows, allow_subset=allow_subset)
    rows = [make_verifier_row(row) for row in joined]
    train, val = _contracts.split_by_source(rows, eval_fraction=eval_fraction, seed=seed)
    covered = {row["item_id"] for row in rows}
    manifest = {
        "schema_version": _contracts.SCHEMA_VERSION,
        "label_source": LABEL_SOURCE,
        "label_limitation": LABEL_LIMITATION,
        "generated_items": len(generated_rows),
        "covered_items": len(covered),
        "uncovered_items": len(generated_rows) - len(covered),
        "exported_candidates": len(candidate_rows),
        "joined_candidates": len(rows),
        "allow_subset": allow_subset,
        "eval_fraction": eval_fraction,
        "split_seed": seed,
        "split_group": "source_id",
        "splits": {},
    }
    for name, split in (("train", train), ("val", val)):
        positives = sum(row["reward_model"]["ground_truth"]["is_correct"] for row in split)
        manifest["splits"][name] = {
            "rows": len(split), "items": len({row["item_id"] for row in split}),
            "sources": len({row["source_id"] for row in split}),
            "positive": positives, "negative": len(split) - positives,
        }
    return train, val, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gen_data_path", required=True, help="ID-bearing generated-question parquet")
    parser.add_argument("--solver_pred_path", required=True, help="All-candidate solver export parquet")
    parser.add_argument("--local_dir", default="./data/verifier")
    parser.add_argument("--eval_fraction", type=float, default=0.1, help="Fraction of source groups held out; 0 means train-only")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic source split seed")
    parser.add_argument("--allow_subset", action="store_true", help="Explicitly allow items absent from training-suffix exports")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing verifier output artifacts")
    args = parser.parse_args(argv)
    try:
        import pandas as pd
    except ImportError as error:
        parser.error(f"Parquet I/O requires pandas and pyarrow (or fastparquet): {error}")
    try:
        generated = pd.read_parquet(args.gen_data_path).to_dict(orient="records")
        candidates = pd.read_parquet(args.solver_pred_path).to_dict(orient="records")
        train, val, manifest = build_verifier_dataset(
            generated, candidates, allow_subset=args.allow_subset,
            eval_fraction=args.eval_fraction, seed=args.seed,
        )
        output = Path(args.local_dir)
        names = ["train.parquet", "val.parquet", "dataset_manifest.json"]
        existing = [str(output / name) for name in names if (output / name).exists()]
        if existing and not args.overwrite:
            raise ValueError("output artifacts already exist; use --overwrite: " + ", ".join(existing))
        output.mkdir(parents=True, exist_ok=True)
        # Finish all serialization before publishing any file.
        with tempfile.TemporaryDirectory(prefix=".verifier-build-", dir=output) as temporary:
            temporary = Path(temporary)
            frame = pd.DataFrame(train)
            frame.to_parquet(temporary / "train.parquet", index=False)
            val_frame = pd.DataFrame(val) if val else frame.iloc[:0]
            val_frame.to_parquet(temporary / "val.parquet", index=False)
            (temporary / "dataset_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
            )
            for name in names:
                os.replace(temporary / name, output / name)
    except (ValueError, ImportError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Saved {len(train)} train and {len(val)} validation candidates to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
