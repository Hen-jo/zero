"""Offline launcher contracts. No model, training, GPU, or server is started."""

import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("verifier_preflight", ROOT / "scripts/verifier_preflight.py")
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


def make_hf(directory, *, chat=True):
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    (directory / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"} if chat else {}))
    (directory / "tokenizer.json").write_text("{}")
    # A layout fixture only: these bytes must never be loaded as model weights.
    (directory / "model.safetensors").write_bytes(b"not-real-weights")
    return directory


class CheckpointTests(unittest.TestCase):
    def test_complete_hf_layout_is_checked_without_model_imports(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = make_hf(Path(temporary) / "hf")
            preflight.validate_hf_checkpoint(checkpoint)

    def test_fsdp_root_is_not_an_hf_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = make_hf(Path(temporary) / "global_step_50")
            (checkpoint / "model.safetensors").unlink()
            (checkpoint / "model_world_size_8_rank_0.pt").write_bytes(b"fsdp")
            with self.assertRaisesRegex(preflight.PreflightError, "Convert the FSDP"):
                preflight.validate_hf_checkpoint(checkpoint)

    def test_missing_weight_shard_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = make_hf(Path(temporary) / "hf")
            (checkpoint / "model.safetensors").unlink()
            (checkpoint / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "model-00001.safetensors"}}))
            with self.assertRaisesRegex(preflight.PreflightError, "HF model shard"):
                preflight.validate_hf_checkpoint(checkpoint)
            (checkpoint / "model-00001.safetensors").write_bytes(b"shard")
            preflight.validate_hf_checkpoint(checkpoint)

    def test_verifier_requires_chat_template_but_retriever_does_not(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = make_hf(Path(temporary) / "hf", chat=False)
            with self.assertRaisesRegex(preflight.PreflightError, "chat template"):
                preflight.validate_hf_checkpoint(checkpoint)
            preflight.validate_hf_checkpoint(checkpoint, require_chat_template=False)


class SchemaTests(unittest.TestCase):
    def row(self, split="train"):
        return {
            "source_id": "source_1", "item_id": "item_1", "candidate_id": "candidate_1",
            "prompt": [{"role": "user", "content": "Is the answer correct?"}],
            "reward_model": {"ground_truth": {"is_correct": True, "label_source": "answer_em_proxy"}},
            "extra_info": {"split": split},
        }

    def test_qa_labels_cannot_be_used_as_verifier_validation(self):
        row = self.row("val")
        row["reward_model"]["ground_truth"] = "Paris"
        with self.assertRaisesRegex(preflight.PreflightError, "QA benchmark"):
            preflight.validate_rows([row], "val")

    def test_string_binary_labels_are_rejected(self):
        row = self.row()
        row["reward_model"]["ground_truth"]["is_correct"] = "False"
        with self.assertRaisesRegex(preflight.PreflightError, "must be bool"):
            preflight.validate_rows([row], "train")

    def test_duplicate_candidate_is_rejected(self):
        with self.assertRaisesRegex(preflight.PreflightError, "duplicate candidate"):
            preflight.validate_rows([self.row(), self.row()], "train")

    def test_empty_validation_is_rejected(self):
        with self.assertRaisesRegex(preflight.PreflightError, "split is empty"):
            preflight.validate_rows([], "val")

    def test_source_leakage_is_rejected(self):
        train, val = self.row(), self.row("val")
        val.update(item_id="item_2", candidate_id="candidate_2")
        with patch.object(preflight, "_records_from_parquet", side_effect=[[train], [val]]):
            with self.assertRaisesRegex(preflight.PreflightError, "leakage in source_id"):
                preflight.validate_data("train", "val")

    def test_valid_training_overrides_are_allowed(self):
        preflight.validate_overrides(["trainer.total_epochs=2", "data.train_batch_size=4"])

    def test_hydra_cannot_bypass_validated_contracts(self):
        for override in ("data.val_files=qa.parquet", "+data={val_files:qa}", "~trainer.resume_mode", "actor_rollout_ref.model.path=fsdp", "custom_reward_function.name=compute_score", "reward_model.reward_manager=naive", "+reward_model.validation_with_default_compute_score=True", "--config-name=other"):
            with self.subTest(override=override), self.assertRaises(preflight.PreflightError):
                preflight.validate_overrides([override])

    def test_every_launcher_uses_role_aware_batch_reward_manager(self):
        for iteration in (1, 2, 3):
            script = (ROOT / f"iter{iteration}_verifier.sh").read_text()
            self.assertIn("reward_model.reward_manager=batch", script)
            self.assertIn("+reward_model.validation_with_default_compute_score=False", script)

    def test_busy_retrieval_port_fails_without_killing_owner(self):
        with patch.object(preflight.importlib.util, "find_spec", return_value=object()), patch.object(preflight.socket, "socket") as socket_factory:
            socket_factory.return_value.__enter__.return_value.bind.side_effect = OSError("address in use")
            with self.assertRaisesRegex(preflight.PreflightError, "No process was killed"):
                preflight.validate_runtime()


@unittest.skipUnless(importlib.util.find_spec("pyarrow") and importlib.util.find_spec("pandas"), "parquet integration requires pandas and pyarrow")
class LauncherIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="verifier launch ")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.model = make_hf(self.base / "selected hf export")
        self.retriever = make_hf(self.base / "retriever", chat=False)
        self.index = self.base / "index"
        self.corpus = self.base / "corpus.jsonl"
        self.index.write_bytes(b"fixture-index-not-loadable")
        self.corpus.write_text('{"text":"fixture"}\n')
        # Guard even a regression in the dry-run branch: a model or trainer
        # command is refused by this shim rather than ever executed.
        binary = self.base / "bin"
        binary.mkdir()
        python = binary / "python"
        python.write_text(
            "#!/bin/sh\n"
            'if [ "$1" = "-m" ]; then\n'
            '  case "$2" in\n'
            f'    scripts.verifier_preflight|scripts.data_process.verifier_data) exec {shlex.quote(sys.executable)} "$@" ;;\n'
            "  esac\nfi\n"
            'echo "FORBIDDEN runtime command: $*" >&2\nexit 91\n'
        )
        python.chmod(0o755)
        self.env = os.environ.copy()
        for name in ("BASE_MODEL", "GEN_DATA_PATH", "SOLVER_PRED_PATH", "VERIFIER_DATA_DIR", "VERIFIER_EVAL_FRACTION", "VERIFIER_SPLIT_SEED"):
            self.env.pop(name, None)
        self.env.update({
            "PATH": str(binary) + os.pathsep + self.env["PATH"],
            "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), self.env.get("PYTHONPATH", "")])),
            "VERIFIER_INIT_MODEL": str(self.model), "RETRIEVER_MODEL": str(self.retriever),
            "NFS_INDEX": str(self.index), "NFS_CORPUS": str(self.corpus),
            "VERIFIER_PREFLIGHT_ONLY": "1", "OUTPUT_DIR": str(self.base / "output"),
            "VERIFIER_ALLOW_PARTIAL_COVERAGE": "0", "VERIFIER_OVERWRITE_DATA": "0",
        })
        contracts_spec = importlib.util.spec_from_file_location("preflight_test_contracts", ROOT / "verl/utils/verifier_contracts.py")
        contracts = importlib.util.module_from_spec(contracts_spec)
        contracts_spec.loader.exec_module(contracts)
        self.generated, self.candidates = [], []
        for i in range(4):
            question, source = f"Where is person {i}?", f"Source document number {i}."
            answers = [f"City {i}"]
            item_id = contracts.make_item_id(question, answers, source)
            source_id = contracts.make_source_id(source)
            self.generated.append({"item_id": item_id, "source_id": source_id, "question": question, "reference_answers": answers, "metadata": {"source_document": source}})
            for sample in range(2):
                response = f"<answer>{answers[0] if sample else 'Wrong city'}</answer>"
                candidate_id = contracts.make_candidate_id(item_id, response, "fixture-rollout", 1, sample)
                self.candidates.append({"item_id": item_id, "candidate_id": candidate_id, "source_id": source_id, "responses": response, "rollout_uid": "fixture-rollout", "global_step": 1, "rollout_index": sample})

    def launch(self, iteration, candidates=None, **environment):
        import pandas as pd
        challenger = f"challenger_iter{iteration}_ratio4321_grpo_batch_group1-5_qwen2.5-3b-instruct"
        solver = f"solver_iter{iteration}_ratio4321_grpo_group5_qwen2.5-3b-instruct"
        data_dir = Path(self.env["OUTPUT_DIR"]) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.generated).to_parquet(data_dir / f"zero_{challenger}.parquet", index=False)
        pd.DataFrame(self.candidates if candidates is None else candidates).to_parquet(data_dir / f"solver_pred_{solver}.parquet", index=False)
        result = subprocess.run(["bash", str(ROOT / f"iter{iteration}_verifier.sh")], cwd=self.base, env={**self.env, **environment}, text=True, capture_output=True, timeout=60)
        self.assertNotIn("FORBIDDEN runtime command", result.stderr)
        return result, data_dir / f"verifier_{challenger}"

    def test_all_launchers_build_and_validate_real_parquets_offline(self):
        for iteration in (1, 2, 3):
            with self.subTest(iteration=iteration):
                result, output = self.launch(iteration)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("No servers or training started", result.stdout)
                self.assertIn(f"/checkpoints/dr-zero/verifier_iter{iteration}_", result.stdout)
                preflight.validate_data(output / "train.parquet", output / "val.parquet")
                self.assertFalse((Path(self.env["OUTPUT_DIR"]) / "logs").exists())

    def test_partial_suffix_requires_explicit_opt_in(self):
        subset = self.candidates[:6]  # Three sources; one generated item absent.
        result, _ = self.launch(1, subset)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing candidate coverage", result.stderr)
        result, output = self.launch(1, subset, VERIFIER_ALLOW_PARTIAL_COVERAGE="1")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        manifest = json.loads((output / "dataset_manifest.json").read_text())
        self.assertTrue(manifest["allow_subset"])
        self.assertEqual(manifest["uncovered_items"], 1)

    def test_missing_init_model_fails_before_python_or_artifact_writes(self):
        for iteration in (1, 2, 3):
            environment = dict(self.env)
            environment.pop("VERIFIER_INIT_MODEL")
            result = subprocess.run(["bash", str(ROOT / f"iter{iteration}_verifier.sh")], env=environment, text=True, capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Set VERIFIER_INIT_MODEL", result.stderr)
        self.assertFalse(Path(self.env["OUTPUT_DIR"]).exists())

    def test_empty_heldout_split_stops_before_servers(self):
        result, _ = self.launch(1, VERIFIER_EVAL_FRACTION="0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("val split is empty", result.stderr)


if __name__ == "__main__":
    unittest.main()
