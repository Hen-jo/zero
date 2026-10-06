"""CPU-only contracts: python -m unittest discover -s tests -p test_verifier_contracts.py."""

import copy
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


contracts = load_module("verifier_contracts_under_test", "verl/utils/verifier_contracts.py")
builder = load_module("verifier_builder_under_test", "scripts/data_process/verifier_data.py")


def generated(question="What is France's capital?", answer="Paris", source="France document"):
    return {
        "item_id": contracts.make_item_id(question, answer, source),
        "source_id": contracts.make_source_id(source),
        "question": question,
        "reference_answers": [answer],
        "prompt": [{"role": "user", "content": question}],
        "metadata": {"source_document": source, "original_question": question},
        "extra_info": {
            "split": "original", "index": 15,
            "tools_kwargs": {"search": {"create_kwargs": {"custom_option": "kept"}}},
            "interaction_kwargs": {"name": "kept"},
        },
    }


def candidate(item, answer="Paris", index=0, step=1, uid="rollout-one"):
    response = "<think>Candidate rationale</think><answer>" + answer + "</answer>"
    return {
        "item_id": item["item_id"], "source_id": item["source_id"],
        "candidate_id": contracts.make_candidate_id(item["item_id"], response, uid, step, index),
        "rollout_uid": uid, "rollout_index": index, "global_step": step,
        "responses": response, "training_reward": 0.5,
    }


class IdentityTests(unittest.TestCase):
    def test_item_identity_independent_of_whitespace_answer_order_and_unicode_form(self):
        first = contracts.make_item_id(" A  question? ", ["Cafe\u0301", "Paris"], "Source\ntext")
        second = contracts.make_item_id("A question?", ["Paris", "Café", "Paris"], "Source text")
        self.assertEqual(first, second)
        self.assertNotEqual(first, contracts.make_item_id("Other question", ["Café", "Paris"], "Source text"))
        self.assertNotEqual(first, contracts.make_item_id("A question?", ["Paris"], "Source text"))
        self.assertNotEqual(first, contracts.make_item_id("A question?", ["Café", "Paris"], "Other source"))

    def test_candidate_slot_identity_distinguishes_repeat_step_uid_and_index(self):
        item = generated()["item_id"]
        original = contracts.make_candidate_id(item, "answer", "uid", 1, 0)
        self.assertEqual(original, contracts.make_candidate_id(item, "different text", "uid", 1, 0))
        for uid, step, index in (("uid2", 1, 0), ("uid", 2, 0), ("uid", 1, 1)):
            self.assertNotEqual(original, contracts.make_candidate_id(item, "answer", uid, step, index))

    def test_candidate_identity_matches_export_helper(self):
        export = load_module("solver_export_under_test", "verl/utils/solver_export.py")
        item = generated()
        record = export.build_export_records([item["item_id"]], [item["source_id"]], ["uid"],
                                             [0], ["<answer>Paris</answer>"], [1.0], 3, 1)[0]
        self.assertEqual(record["candidate_id"], contracts.make_candidate_id(
            item["item_id"], record["responses"], "uid", 3, 0))
        self.assertEqual(len(contracts.join_candidates([item], [record])), 1)

    def test_source_extraction_groups_different_hop_prompts(self):
        template = builder._prompts.DEFAULT_CHALLENGER_PREFIX
        first = [{"role": "user", "content": template.format(hops=2, document="same document")}]
        second = [{"role": "user", "content": template.format(hops=4, document="same document")}]
        self.assertEqual(contracts.extract_source_document(first), "same document")
        self.assertEqual(contracts.make_source_id(contracts.extract_source_document(first)),
                         contracts.make_source_id(contracts.extract_source_document(second)))

    def test_invalid_identity_fields_fail(self):
        for bad in (None, "", " ", float("nan"), 0, [], {}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                contracts.make_item_id(bad, "Paris", "source")
        for bad in (True, -1, 0.5, "1", None):
            with self.subTest(step=bad), self.assertRaises(ValueError):
                contracts.make_candidate_id("item", "text", "uid", bad, 0)


class JoinTests(unittest.TestCase):
    def setUp(self):
        self.items = [generated(), generated("What is Italy's capital?", "Rome", "Italy document")]
        self.candidates = [candidate(self.items[0]), candidate(self.items[0], "London", index=1),
                           candidate(self.items[1], "Rome", uid="second-rollout")]

    def test_reorder_keeps_all_candidates_and_correct_labels(self):
        first = contracts.join_candidates(self.items, self.candidates)
        second = contracts.join_candidates(list(reversed(self.items)), list(reversed(self.candidates)))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertEqual({row["candidate_answer"]: row["is_correct"] for row in first},
                         {"Paris": True, "London": False, "Rome": True})

    def test_duplicate_generated_and_candidate_ids_fail(self):
        with self.assertRaisesRegex(ValueError, "duplicate generated item_id"):
            contracts.join_candidates(self.items + [self.items[0]], self.candidates)
        with self.assertRaisesRegex(ValueError, "duplicate candidate_id"):
            contracts.join_candidates(self.items, self.candidates + [self.candidates[0]])

    def test_unknown_missing_and_legacy_ids_fail(self):
        unknown = candidate(generated("Other?", "Elsewhere", "Other document"))
        with self.assertRaisesRegex(ValueError, "unknown candidate item_id"):
            contracts.join_candidates(self.items, self.candidates + [unknown], allow_subset=True)
        with self.assertRaisesRegex(ValueError, "missing candidate coverage"):
            contracts.join_candidates(self.items, self.candidates[:2])
        self.assertEqual(len(contracts.join_candidates(self.items, self.candidates[:2], allow_subset=True)), 2)
        for bad in ({"responses": "<answer>Paris</answer>"}, {"item_id": self.items[0]["item_id"]}):
            with self.assertRaises(ValueError):
                contracts.join_candidates(self.items, [bad], allow_subset=True)
        legacy = copy.deepcopy(self.items[0])
        del legacy["item_id"]
        with self.assertRaisesRegex(ValueError, "generated item_id"):
            contracts.join_candidates([legacy], self.candidates[:1])

    def test_bad_content_ids_and_source_fail(self):
        altered = copy.deepcopy(self.items)
        altered[0]["question"] = "Wrong question?"
        with self.assertRaisesRegex(ValueError, "item_id does not match"):
            contracts.join_candidates(altered, self.candidates)
        altered = copy.deepcopy(self.candidates)
        altered[0]["source_id"] = self.items[1]["source_id"]
        with self.assertRaisesRegex(ValueError, "source_id does not match"):
            contracts.join_candidates(self.items, altered)
        altered = copy.deepcopy(self.candidates)
        altered[0]["global_step"] = 9
        with self.assertRaisesRegex(ValueError, "candidate_id does not match"):
            contracts.join_candidates(self.items, altered)

    def test_empty_inputs_and_malformed_answers_fail_instead_of_skipping(self):
        with self.assertRaisesRegex(ValueError, "generated dataset is empty"):
            contracts.join_candidates([], [])
        with self.assertRaisesRegex(ValueError, "candidate export is empty"):
            contracts.join_candidates(self.items, [])
        for text in ("", "missing tags", "<answer></answer>", "<answer>  </answer>", "<answer>the</answer>"):
            rows = copy.deepcopy(self.candidates)
            rows[0]["responses"] = text
            with self.subTest(text=text), self.assertRaises(ValueError):
                contracts.join_candidates(self.items, rows)

    def test_invalid_or_conflicting_precomputed_labels_fail(self):
        for label in ("False", "YES", 1, 0, None, False):
            rows = copy.deepcopy(self.candidates)
            rows[0]["is_correct"] = label
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "is_correct"):
                contracts.join_candidates(self.items, rows)
        self.candidates[0]["is_correct"] = True
        self.assertEqual(len(contracts.join_candidates(self.items, self.candidates)), 3)
        self.candidates[0]["label_source"] = "chain_proof"
        with self.assertRaisesRegex(ValueError, "label_source"):
            contracts.join_candidates(self.items, self.candidates)

    def test_reward_never_substitutes_for_answer_label(self):
        self.candidates[0]["training_reward"] = -10.0
        self.candidates[1]["training_reward"] = 10.0
        rows = contracts.join_candidates(self.items, self.candidates)
        by_answer = {row["candidate_answer"]: row["is_correct"] for row in rows}
        self.assertTrue(by_answer["Paris"])
        self.assertFalse(by_answer["London"])
        self.candidates[0]["training_reward"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite number"):
            contracts.join_candidates(self.items, self.candidates)

    def test_only_final_tag_is_used(self):
        self.assertEqual(contracts.extract_answer("<answer>Wrong</answer><answer>Paris</answer>"), "Paris")
        self.assertTrue(contracts.answer_em_proxy(" The Paris! ", ["Paris"]))


class DatasetTests(unittest.TestCase):
    def fixture(self):
        items, candidates = [], []
        for source_number in range(6):
            for question_number in range(2):
                item = generated(f"Question {source_number}-{question_number}?", "Paris", f"Source {source_number}")
                items.append(item)
                for answer, index in (("Paris", 0), ("London", 1)):
                    candidates.append(candidate(item, answer, index, uid=f"uid-{source_number}-{question_number}"))
        return items, candidates

    def test_deterministic_source_disjoint_split_and_no_lost_candidates(self):
        items, candidates = self.fixture()
        first = builder.build_verifier_dataset(items, candidates, eval_fraction=0.25, seed=11)
        random.Random(4).shuffle(items)
        random.Random(7).shuffle(candidates)
        second = builder.build_verifier_dataset(items, candidates, eval_fraction=0.25, seed=11)
        self.assertEqual(first, second)
        train, val, manifest = first
        self.assertTrue(train and val)
        self.assertEqual(len(train) + len(val), len(candidates))
        for key in ("item_id", "source_id", "candidate_id"):
            self.assertFalse({row[key] for row in train} & {row[key] for row in val})
        self.assertEqual(manifest["joined_candidates"], len(candidates))
        self.assertEqual(manifest["label_source"], "answer_em_proxy")
        self.assertIn("not a label for chain validity", manifest["label_limitation"])
        self.assertTrue(all(row["extra_info"]["split"] == "val" for row in val))

    def test_prompt_tools_labels_and_provenance_preserved(self):
        item = generated()
        train, val, manifest = builder.build_verifier_dataset([item], [candidate(item)], eval_fraction=0)
        self.assertFalse(val)
        row = train[0]
        self.assertEqual(row["reward_model"]["ground_truth"], {"is_correct": True, "label_source": "answer_em_proxy"})
        self.assertIn("Candidate Answer: Paris", row["prompt"][0]["content"])
        self.assertEqual(row["extra_info"]["tools_kwargs"]["search"]["create_kwargs"]["custom_option"], "kept")
        self.assertEqual(row["extra_info"]["interaction_kwargs"], {"name": "kept"})
        self.assertEqual(row["metadata"]["source_document"], "France document")
        self.assertEqual(row["metadata"]["solver_response"], candidate(item)["responses"])
        self.assertEqual(item["extra_info"]["split"], "original")
        self.assertEqual(manifest["uncovered_items"], 0)

    def test_insufficient_groups_and_bad_fraction_fail(self):
        item = generated()
        with self.assertRaisesRegex(ValueError, "at least two distinct sources"):
            builder.build_verifier_dataset([item], [candidate(item)])
        for fraction in (-0.1, 1, 2, True, float("nan")):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                builder.build_verifier_dataset([item], [candidate(item)], eval_fraction=fraction)

    def test_subset_coverage_is_reported(self):
        items, candidates = self.fixture()
        train, val, manifest = builder.build_verifier_dataset(items, candidates[:8], allow_subset=True)
        self.assertEqual(manifest["uncovered_items"], 8)
        self.assertEqual(manifest["covered_items"], 4)
        self.assertEqual(len(train) + len(val), 8)

    def test_cli_help_without_runtime_imports(self):
        result = subprocess.run([sys.executable, "-S", str(ROOT / "scripts/data_process/verifier_data.py"), "--help"],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--allow_subset", result.stdout)


@unittest.skipUnless(importlib.util.find_spec("pandas") and importlib.util.find_spec("pyarrow"),
                     "optional parquet integration requires pandas and pyarrow")
class ParquetIntegrationTests(unittest.TestCase):
    def test_export_to_cli_to_preflight_and_reorder_equivalence(self):
        import pandas as pd
        exporter = load_module("solver_export_integration", "verl/utils/solver_export.py")
        items, _ = DatasetTests().fixture()
        # Cover real generated data that has no optional interaction kwargs.
        for item in items:
            item["extra_info"] = {"index": 0, "split": "train"}
        ids, sources, uids, indices, responses, scores = [], [], [], [], [], []
        for offset, item in enumerate(items):
            for index, answer in enumerate(("Paris", "London")):
                ids.append(item["item_id"])
                sources.append(item["source_id"])
                uids.append(f"uid-{offset}")
                indices.append(index)
                responses.append(f"<answer>{answer}</answer>")
                scores.append(0.0 if index == 0 else 1.0)  # adversarial reward ranking
        records = exporter.build_export_records(ids, sources, uids, indices, responses, scores, 7, 2)
        self.assertEqual(len(contracts.join_candidates(items, records)), len(records))
        random.Random(19).shuffle(records)
        with tempfile.TemporaryDirectory(prefix="verifier-contract-test-") as directory:
            directory = Path(directory)
            gen_path, solver_path = directory / "gen.parquet", directory / "solver.parquet"
            pd.DataFrame(items).to_parquet(gen_path, index=False)
            pd.DataFrame(records).to_parquet(solver_path, index=False)
            command = [sys.executable, str(ROOT / "scripts/data_process/verifier_data.py"),
                       "--gen_data_path", str(gen_path), "--solver_pred_path", str(solver_path),
                       "--local_dir", str(directory / "output"), "--eval_fraction", "0.25", "--seed", "11"]
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((directory / "output/dataset_manifest.json").read_text())
            self.assertEqual(manifest["joined_candidates"], len(records))
            self.assertEqual(manifest["uncovered_items"], 0)
            preflight = subprocess.run([
                sys.executable, str(ROOT / "scripts/verifier_preflight.py"), "data",
                "--train", str(directory / "output/train.parquet"),
                "--val", str(directory / "output/val.parquet"),
            ], text=True, capture_output=True)
            self.assertEqual(preflight.returncode, 0, preflight.stderr)
            split_rows = []
            for split in ("train", "val"):
                split_rows.extend(pd.read_parquet(directory / f"output/{split}.parquet").to_dict("records"))
            self.assertEqual(len(split_rows), len(records))
            self.assertEqual(sum(row["reward_model"]["ground_truth"]["is_correct"] for row in split_rows), len(items))
            self.assertTrue(all(row["extra_info"]["tools_kwargs"]["search"]["create_kwargs"] for row in split_rows))
            # Rerunning never silently overwrites a prior dataset.
            rerun = subprocess.run(command, text=True, capture_output=True)
            self.assertNotEqual(rerun.returncode, 0)
            self.assertIn("--overwrite", rerun.stderr)
            overwritten = subprocess.run(command + ["--overwrite"], text=True, capture_output=True)
            self.assertEqual(overwritten.returncode, 0, overwritten.stderr)


if __name__ == "__main__":
    unittest.main()
