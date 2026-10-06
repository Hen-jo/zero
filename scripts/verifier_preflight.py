"""CPU-only, offline checks for verifier launchers; never load model weights.

The checkpoint check establishes a complete local HF layout, not tensor integrity
or GPU compatibility. FSDP ``global_step_*`` directories must first be converted.
"""

import argparse
import ast
import importlib.util
import json
from pathlib import Path
import socket
import sys


class PreflightError(ValueError):
    """A required verifier launch contract is not satisfied."""


def require_file(path, description="file"):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise PreflightError(f"Missing or empty {description}: {path}")
    try:
        with path.open("rb") as stream:
            stream.read(1)
    except OSError as exc:
        raise PreflightError(f"Unreadable {description}: {path}: {exc}") from exc
    return path


def read_json(path):
    try:
        value = json.loads(require_file(path).read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"Invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PreflightError(f"Expected a JSON object: {path}")
    return value


def validate_hf_checkpoint(path, *, require_chat_template=True):
    """Check a local full-model HF export without importing torch/transformers."""
    path = Path(path).expanduser()
    if not path.is_dir():
        raise PreflightError(f"HF checkpoint must be an existing local directory: {path}")
    config = read_json(path / "config.json")
    if not isinstance(config.get("model_type"), str) or not config["model_type"]:
        raise PreflightError(f"HF config.json is missing model_type: {path}")
    tokenizer_config = read_json(path / "tokenizer_config.json")
    tokenizers = ("tokenizer.json", "tokenizer.model", "spiece.model", "vocab.txt")
    if not any((path / name).is_file() and (path / name).stat().st_size for name in tokenizers):
        require_file(path / "vocab.json", "tokenizer vocabulary")
        require_file(path / "merges.txt", "tokenizer merges")
    if require_chat_template and not tokenizer_config.get("chat_template"):
        require_file(path / "chat_template.jinja", "chat template")

    # Do not mistake FSDP shards or adapter-only weights for a full HF model.
    for name in ("model.safetensors", "pytorch_model.bin"):
        if (path / name).is_file():
            require_file(path / name, "HF model weights")
            return
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if not (path / name).is_file():
            continue
        weight_map = read_json(path / name).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise PreflightError(f"HF weight index has no weight_map: {path / name}")
        for shard in weight_map.values():
            if not isinstance(shard, str) or not shard or Path(shard).is_absolute() or ".." in Path(shard).parts:
                raise PreflightError(f"Invalid HF shard name in {path / name}: {shard!r}")
            require_file(path / shard, "HF model shard")
        return
    raise PreflightError(
        f"No full HF model weights found in {path}. Convert the FSDP actor checkpoint "
        "to HF format and set VERIFIER_INIT_MODEL to that export; a global_step directory is not sufficient."
    )


def validate_python_file(path, symbol=None):
    path = require_file(path, "Python module")
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise PreflightError(f"Invalid Python module: {path}: {exc}") from exc
    if symbol and not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol
        for node in tree.body
    ):
        raise PreflightError(f"Missing function {symbol} in {path}")


def validate_configs(config_dir, tool_config):
    try:
        import yaml
    except ImportError as exc:
        raise PreflightError("PyYAML is required for config preflight") from exc
    config_dir = Path(config_dir)
    try:
        config = yaml.safe_load(require_file(config_dir / "search_multiturn_grpo.yaml").read_text())
        tools = yaml.safe_load(require_file(tool_config).read_text())
    except yaml.YAMLError as exc:
        raise PreflightError(f"Invalid YAML configuration: {exc}") from exc
    if not isinstance(config, dict) or "ppo_trainer" not in config.get("defaults", []):
        raise PreflightError("Verifier config must inherit ppo_trainer")
    multi_turn = config.get("actor_rollout_ref", {}).get("rollout", {}).get("multi_turn", {})
    if multi_turn.get("enable") is not True:
        raise PreflightError("Verifier backward-search training requires multi_turn.enable=true")
    entries = tools.get("tools", []) if isinstance(tools, dict) else []
    search_tools = [entry for entry in entries if isinstance(entry, dict) and entry.get("class_name") == "verl.tools.search_tool.SearchTool"]
    if len(search_tools) != 1:
        raise PreflightError("Tool config must contain exactly one SearchTool")
    endpoint = search_tools[0].get("config", {}).get("retrieval_service_url")
    if endpoint not in ("http://127.0.0.1:8000/retrieve", "http://localhost:8000/retrieve"):
        raise PreflightError("Tool config must target the launcher's local retrieval endpoint on port 8000")


PROTECTED_OVERRIDES = (
    "data.train_files", "data.val_files", "actor_rollout_ref.model.path",
    "actor_rollout_ref.rollout.name", "actor_rollout_ref.rollout.multi_turn.enable",
    "actor_rollout_ref.rollout.multi_turn.tool_config_path", "custom_reward_function",
    "reward_model.reward_manager", "reward_model.validation_with_default_compute_score",
    "trainer.default_local_dir", "trainer.experiment_name", "trainer.resume_mode",
    "trainer.resume_from_path",
)


def validate_overrides(overrides):
    """Do not let trailing Hydra args silently bypass already checked inputs."""
    for override in overrides:
        if override.startswith("-") or "=" not in override:
            raise PreflightError(f"Expected a Hydra key=value override, got {override!r}")
        key = override.split("=", 1)[0].lstrip("+~")
        if not key:
            raise PreflightError(f"Expected a nonempty Hydra key, got {override!r}")
        if any(key == protected or key.startswith(protected + ".") or protected.startswith(key + ".") for protected in PROTECTED_OVERRIDES):
            raise PreflightError(f"Override {key!r} bypasses verifier preflight; use the documented launcher environment variables")


def _records_from_parquet(path):
    require_file(path, "verifier parquet")
    try:
        import pandas as pd
        # Avoid background Arrow decoder threads surviving this short-lived
        # preflight process at interpreter shutdown.
        return pd.read_parquet(path, engine="pyarrow", use_threads=False).to_dict(orient="records")
    except (ImportError, OSError, ValueError) as exc:
        raise PreflightError(f"Cannot read verifier parquet {path}: {exc}") from exc


def validate_rows(rows, split):
    if not rows:
        raise PreflightError(f"Verifier {split} split is empty; use at least two source groups")
    source_ids, item_ids, candidate_ids = set(), set(), set()
    for index, row in enumerate(rows):
        label = f"{split} row {index}"
        if not isinstance(row, dict):
            raise PreflightError(f"{label}: expected a record")
        for key, values in (("source_id", source_ids), ("item_id", item_ids), ("candidate_id", candidate_ids)):
            value = row.get(key)
            if not isinstance(value, str) or not value.strip():
                raise PreflightError(f"{label}: missing nonempty {key}")
            if key == "candidate_id" and value in values:
                raise PreflightError(f"{label}: duplicate candidate_id {value!r}")
            values.add(value)
        reward = row.get("reward_model")
        truth = reward.get("ground_truth") if isinstance(reward, dict) else None
        if not isinstance(truth, dict) or type(truth.get("is_correct")) is not bool:
            raise PreflightError(f"{label}: verifier ground_truth.is_correct must be bool; QA benchmark labels are incompatible")
        if truth.get("label_source") != "answer_em_proxy":
            raise PreflightError(f"{label}: missing answer_em_proxy label provenance")
        prompt = row.get("prompt")
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        if not isinstance(prompt, list) or not prompt:
            raise PreflightError(f"{label}: prompt must be a nonempty chat-message list")
        if any(not isinstance(message, dict) or not isinstance(message.get("content"), str) or not message["content"].strip() or message.get("role") not in ("system", "user", "assistant", "tool") for message in prompt):
            raise PreflightError(f"{label}: malformed prompt message")
        extra = row.get("extra_info")
        if not isinstance(extra, dict) or extra.get("split") != split:
            raise PreflightError(f"{label}: extra_info.split must be {split!r}")
    return source_ids, item_ids, candidate_ids


def validate_data(train_file, val_file):
    train = validate_rows(_records_from_parquet(train_file), "train")
    val = validate_rows(_records_from_parquet(val_file), "val")
    for key, train_values, val_values in zip(("source_id", "item_id", "candidate_id"), train, val):
        overlap = train_values & val_values
        if overlap:
            raise PreflightError(f"Verifier train/val leakage in {key}: {sorted(overlap)[:3]}")


def validate_runtime():
    # find_spec only: importing these packages can initialize CUDA or Ray.
    required = ("torch", "transformers", "datasets", "pandas", "pyarrow", "hydra", "omegaconf", "ray", "sglang", "faiss", "uvicorn", "fastapi")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise PreflightError("Missing runtime dependencies: " + ", ".join(missing))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        try:
            server.bind(("0.0.0.0", 8000))
        except OSError as exc:
            raise PreflightError("Retrieval port 8000 is unavailable; stop or reconfigure its owner explicitly. No process was killed.") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    inputs = subparsers.add_parser("inputs")
    for name in ("repo", "init-model", "gen-data", "solver-pred", "config-dir", "tool-config", "index", "corpus", "retriever-model"):
        inputs.add_argument("--" + name, required=True)
    inputs.add_argument("--trainer-override", action="append", default=[])
    data = subparsers.add_parser("data")
    data.add_argument("--train", required=True)
    data.add_argument("--val", required=True)
    subparsers.add_parser("runtime")
    args = parser.parse_args(argv)
    try:
        if args.stage == "inputs":
            validate_overrides(args.trainer_override)
            validate_hf_checkpoint(args.init_model)
            for name in ("gen_data", "solver_pred", "index", "corpus"):
                require_file(getattr(args, name), name)
            validate_hf_checkpoint(args.retriever_model, require_chat_template=False)
            repo = Path(args.repo)
            validate_python_file(repo / "scripts/data_process/verifier_data.py", "main")
            validate_python_file(repo / "verl/custom_reward/verifier_reward_function.py", "compute_verifier_score")
            validate_python_file(repo / "search/retrieval_server.py")
            require_file(repo / "verl/trainer/config/ppo_trainer.yaml", "base trainer config")
            validate_configs(args.config_dir, args.tool_config)
        elif args.stage == "data":
            validate_data(args.train, args.val)
        else:
            validate_runtime()
    except (PreflightError, OSError) as exc:
        print(f"[verifier preflight] ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"[verifier preflight] {args.stage} checks passed (no model weights loaded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
