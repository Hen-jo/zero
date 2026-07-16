import json
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verifier_data = load_module("verifier_data", "scripts/data_process/verifier_data.py")
verifier_reward = load_module(
    "verifier_reward_function", "verl/custom_reward/verifier_reward_function.py"
)

extract_keywords = verifier_data.extract_keywords
extract_last_query = verifier_data.extract_last_query
make_row = verifier_data.make_row
compute_verifier_score = verifier_reward.compute_verifier_score


class InverseQuestionReconstructionTest(unittest.TestCase):
    def test_keyword_extraction_keeps_entities_and_relations(self):
        keywords = extract_keywords(
            "Which university was founded by the director of Inception?"
        )
        self.assertEqual(
            keywords,
            ["university", "founded", "director", "inception"],
        )

    def test_terminal_query_uses_direct_column_then_trajectory_fallback(self):
        self.assertEqual(
            extract_last_query({"solver_last_query": "  Inception director  "}),
            "Inception director",
        )
        self.assertEqual(
            extract_last_query(
                {
                    "solver_last_query": "",
                    "solver_search_queries": json.dumps(["first hop", "last hop"]),
                }
            ),
            "last hop",
        )

    def test_matching_reconstruction_receives_full_lexical_reward(self):
        question = "Which university was founded by the director of Inception?"
        row = make_row(
            base_row={},
            question=question,
            solver_answer="Example University",
            ground_truth_answer="Example University",
            is_correct=True,
            last_query="Inception director university",
        )
        response = (
            "<think>I will search backward from the terminal query.</think>"
            "<tool_call>Inception director university</tool_call>"
            f"<answer>{question}</answer>"
        )
        score = compute_verifier_score(
            data_sources=["hotpot_2"],
            solution_strs=[response],
            ground_truths=[row["reward_model"]["ground_truth"]],
            extra_infos=[{}],
            semantic_model_name="",
        )[0]
        self.assertAlmostEqual(score, 1.0)

    def test_answer_leakage_is_penalized(self):
        reference_question = "Which university was founded by the director of Inception?"
        ground_truth = {
            "target": ["Example University"],
            "reference_question": [reference_question],
            "reference_keywords": [json.dumps(extract_keywords(reference_question))],
        }
        response = (
            "<think>Search backward.</think><tool_call>Inception director</tool_call>"
            "<answer>Is the answer Example University?</answer>"
        )
        score = compute_verifier_score(
            data_sources=["hotpot_2"],
            solution_strs=[response],
            ground_truths=[ground_truth],
            extra_infos=[{}],
            semantic_model_name="",
        )[0]
        self.assertLess(score, 0.5)


if __name__ == "__main__":
    unittest.main()
