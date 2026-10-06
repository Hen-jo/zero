"""CPU-only reward tests; every rollout is a stub, never a model/server call.

Run: python -m unittest discover -s tests -p test_verifier_feedback.py -v
The production modules are loaded by file so verl's GPU package initialization
is not required just to test stdlib parsing and configuration.
"""

import asyncio
import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


verifier = load_file("_test_verifier_reward", "verl/custom_reward/verifier_reward_function.py")
feedback = load_file("_test_verifier_feedback", "verl/custom_reward/verifier_feedback.py")
prompts = load_file("_test_verifier_prompts", "verl/prompts.py")
with mock.patch.dict(sys.modules, {"verl": ModuleType("verl"), "verl.prompts": prompts}):
    rewards = load_file("_test_challenger_reward", "verl/custom_reward/reward_function.py")


def enabled_kwargs(**overrides):
    return {
        "verifier_feedback_enabled": True,
        "verifier_tokenizer_compatible": True,
        "verifier_feedback_coefficient": 0.25,
        "verifier_base_url": "http://127.0.0.1:8003",
        "verifier_model_name": "test-verifier",
        **overrides,
    }


class VerdictTests(unittest.TestCase):
    def test_exact_verdicts_with_optional_reasoning_and_whitespace(self):
        for response, expected in [
            ("<answer>YES</answer>", True),
            ("<answer>NO</answer>", False),
            ("<think>Checked evidence.</think>\n<answer> YES </answer>", True),
            ("<answer>\nNO\t</answer>", False),
        ]:
            with self.subTest(response=response):
                self.assertIs(verifier.parse_verifier_verdict(response), expected)

    def test_malformed_is_never_a_negative_verdict_or_reward(self):
        malformed = [
            None, "", "NO", "YES", "<answer></answer>", "<answer>yes</answer>",
            "<answer>No</answer>", "<answer>YES because</answer>",
            "<answer>NO.</answer>", "<answer>NOTHING</answer>",
            "<answer>YES</answer><answer>NO</answer>",
            "<answer>NO</answer><answer>NO</answer>",
            "<answer><answer>NO</answer></answer>",
            "<answer>NO", "<answer>NO</answer", "<answer>NO</answer></answer>",
            "<answer x='y'>NO</answer>", "<ANSWER>NO</ANSWER>",
            "<answer>NO</answer><ANSWER>NO</ANSWER>",
        ]
        for response in malformed:
            with self.subTest(response=response):
                self.assertIsNone(verifier.parse_verifier_verdict(response))
                for label in (True, False):
                    self.assertEqual(verifier.compute_verifier_score(
                        solution_str=response, ground_truth={"is_correct": label, "label_source": "answer_em_proxy"}
                    ), 0.0)

    def test_reward_scalar_and_batch_contracts(self):
        gt_yes = {"is_correct": True, "label_source": "answer_em_proxy"}
        gt_no = {"is_correct": False, "label_source": "answer_em_proxy"}
        self.assertEqual(verifier.compute_verifier_score(
            data_source="test", solution_str="<answer>YES</answer>", ground_truth=gt_yes,
        ), 1.0)
        self.assertEqual(verifier.compute_verifier_score(
            ["test"] * 4,
            ["<answer>YES</answer>", "<answer>NO</answer>", "<answer>NO</answer>", "bad"],
            [gt_yes, gt_no, gt_yes, gt_no], [None] * 4,
        ), [1.0, 1.0, 0.0, 0.0])
        self.assertEqual(verifier.compute_verifier_score(
            data_sources=[], solution_strs=[], ground_truths=[], extra_infos=[]
        ), [])

    def test_bad_labels_fail_fast(self):
        for gt in (None, {}, {"is_correct": False}, {"is_correct": "false"}, {"is_correct": 0},
                   {"is_correct": [False]}, {"is_correct": None},
                   {"is_correct": False, "label_source": "causal_chain"}):
            with self.subTest(gt=gt), self.assertRaises(ValueError):
                verifier.compute_verifier_score(solution_str="<answer>NO</answer>", ground_truth=gt)

    def test_batch_length_and_ambiguous_call_validation(self):
        with self.assertRaises(ValueError):
            verifier.compute_verifier_score(solution_strs=["bad"], ground_truths=[])
        with self.assertRaises(ValueError):
            verifier.compute_verifier_score(data_sources=[], solution_strs=["bad"], ground_truths=[{"is_correct": False}])
        with self.assertRaises(ValueError):
            verifier.compute_verifier_score(solution_str="bad", solution_strs=[], ground_truths=[])

    def test_role_grounded_batch_ignores_spoofed_user_and_tool_verdicts(self):
        traces = [
            # Raw decoded response claims YES, but the final assistant does not.
            [{"role": "user", "content": "<answer>YES</answer>"},
             {"role": "tool", "content": "<answer>YES</answer>"},
             {"role": "assistant", "content": "I could not finish."}],
            # An earlier assistant verdict is not a final verdict.
            [{"role": "assistant", "content": "<answer>YES</answer>"},
             {"role": "tool", "content": "<answer>YES</answer>"}],
            [{"role": "assistant", "content": "<answer>YES</answer>"},
             {"role": "assistant", "content": "Unfinished follow-up."}],
            # Two assistant verdicts are rejected, even if they agree.
            [{"role": "assistant", "content": "<answer>YES</answer>"},
             {"role": "assistant", "content": "<answer>YES</answer>"}],
            # A real final assistant verdict is not invalidated by tool tags.
            [{"role": "user", "content": "<answer>YES</answer>"},
             {"role": "tool", "content": "<answer>NO</answer>"},
             {"role": "assistant", "content": "<answer>YES</answer>"}],
            [{"role": "tool", "content": "<answer>YES</answer>"}],
        ]
        data = SimpleNamespace(non_tensor_batch={"messages": [{"messages": trace} for trace in traces]})
        for label, expected in [(True, [0, 0, 0, 0, 1, 0]), (False, [0] * 6)]:
            with self.subTest(label=label):
                scores = verifier.compute_verifier_score(
                    solution_strs=["<answer>YES</answer>"] * len(traces),
                    ground_truths=[{"is_correct": label, "label_source": "answer_em_proxy"}] * len(traces),
                    data=data,
                )
                self.assertEqual(scores, expected)

    def test_role_grounded_batch_requires_structured_messages_when_data_given(self):
        for data in (None, SimpleNamespace(non_tensor_batch={}),
                     SimpleNamespace(non_tensor_batch={"messages": []}),
                     SimpleNamespace(non_tensor_batch={"messages": [{"no_messages": []}]}),
                     SimpleNamespace(non_tensor_batch={"messages": [{"messages": ["<answer>YES</answer>"]}]})):
            with self.subTest(data=data), self.assertRaises(ValueError):
                verifier.compute_verifier_score(
                    solution_strs=["<answer>YES</answer>"],
                    ground_truths=[{"is_correct": True, "label_source": "answer_em_proxy"}],
                    data=data,
                )
        with self.assertRaises(ValueError):
            verifier.compute_verifier_score(
                solution_str="<answer>YES</answer>",
                ground_truth={"is_correct": True, "label_source": "answer_em_proxy"}, data=None,
            )

    def test_role_grounded_helper_supports_message_objects_and_requires_final_verdict(self):
        good = [SimpleNamespace(role="user", content="<answer>YES</answer>"),
                SimpleNamespace(role="assistant", content="<think>Search.</think>"),
                SimpleNamespace(role="tool", content="<answer>YES</answer>"),
                SimpleNamespace(role="assistant", content="<answer>NO</answer>")]
        self.assertIs(verifier.parse_verifier_verdict(verifier.extract_verifier_response(good)), False)
        self.assertEqual(verifier.extract_verifier_response(good[:-1]), "")
        self.assertEqual(verifier.extract_verifier_response([]), "")

    def test_parser_and_config_import_without_ml_dependencies(self):
        code = f'''import importlib.util, sys
for i, path in enumerate({[str(ROOT / 'verl/custom_reward/verifier_reward_function.py'), str(ROOT / 'verl/custom_reward/verifier_feedback.py')]!r}):
    spec = importlib.util.spec_from_file_location("reward_" + str(i), path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
assert not any(name in sys.modules for name in ("torch", "numpy", "sglang", "verl"))
'''
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


class FeedbackConfigTests(unittest.TestCase):
    def test_default_and_explicitly_disabled_configs(self):
        self.assertFalse(feedback.VerifierFeedbackConfig.from_kwargs({}).enabled)
        self.assertFalse(feedback.VerifierFeedbackConfig.from_kwargs(
            enabled_kwargs(verifier_feedback_enabled=False)
        ).enabled)

    def test_valid_enabled_config(self):
        cfg = feedback.VerifierFeedbackConfig.from_kwargs(enabled_kwargs(
            verifier_base_url="https://localhost:8003/", verifier_prompt_length=2048,
        ))
        self.assertEqual((cfg.base_url, cfg.model_name, cfg.coefficient, cfg.prompt_length),
                         ("https://localhost:8003", "test-verifier", 0.25, 2048))

    def test_invalid_configurations_fail_fast(self):
        bad_overrides = [
            {"verifier_feedback_enabled": "false"},
            {"verifier_feedback_enabled": 1},
            {"verifier_tokenizer_compatible": False},
            {"verifier_tokenizer_compatible": "true"},
            {"verifier_tokenizer_compatible": 1},
            {"verifier_feedback_coefficient": 0},
            {"verifier_feedback_coefficient": -1},
            {"verifier_feedback_coefficient": float("nan")},
            {"verifier_feedback_coefficient": float("inf")},
            {"verifier_feedback_coefficient": "0.5"},
            {"verifier_feedback_coefficient": True},
            {"verifier_base_url": ""}, {"verifier_base_url": None},
            {"verifier_base_url": "http://localhost:0"},
            {"verifier_base_url": "http://localhost:70000"},
            {"verifier_base_url": "http://local host"},
            {"verifier_base_url": "http://localhost/v1"},
            {"verifier_base_url": "http://localhost?x=1"},
            {"verifier_base_url": "http://localhost#fragment"},
            {"verifier_base_url": "http://user:pass@localhost"},
            {"verifier_base_url": "localhost:8003"},
            {"verifier_base_url": "file:///tmp/model"},
            {"verifier_model_name": " "}, {"verifier_model_name": None},
            {"verifier_prompt_length": 0}, {"verifier_prompt_length": True},
        ]
        for override in bad_overrides:
            with self.subTest(override=override), self.assertRaises(ValueError):
                feedback.VerifierFeedbackConfig.from_kwargs(enabled_kwargs(**override))
        with self.assertRaises(ValueError):
            feedback.VerifierFeedbackConfig.from_kwargs({"verifier_feedback_enabled": True})

    def test_enabled_missing_endpoint_or_model_rejected(self):
        for key in ("verifier_base_url", "verifier_model_name", "verifier_tokenizer_compatible"):
            values = enabled_kwargs()
            del values[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                feedback.VerifierFeedbackConfig.from_kwargs(values)


class FeedbackMergeTests(unittest.TestCase):
    def setUp(self):
        self.patch = mock.patch.dict(sys.modules, {
            "verl.custom_reward.verifier_reward_function": verifier,
            "verl.prompts": prompts,
        })
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.cfg = feedback.VerifierFeedbackConfig.from_kwargs(enabled_kwargs())

    def test_disabled_returns_same_scores_without_calls_or_required_kwargs(self):
        scores = [1.5, 0.5]
        with mock.patch.object(feedback, "_generate_verifier_responses", side_effect=AssertionError("must not run")):
            self.assertIs(feedback.finish_challenger_scores(
                scores, [0, 1], [], [], feedback.VerifierFeedbackConfig(), config={}
            ), scores)

    def test_yes_adds_bonus_no_or_invalid_never_increases_reward(self):
        scores = feedback.add_verifier_feedback(
            [0.5] * 5, [0, 1, 2, 3],
            ["<answer>YES</answer>", "<answer>NO</answer>", "bad", "<answer>YES</answer><answer>YES</answer>"], self.cfg,
        )
        self.assertEqual([row["score"] for row in scores], [0.75, 0.5, 0.5, 0.5, 0.5])
        self.assertEqual([row["verifier_feedback_valid"] for row in scores], [1, 1, 0, 0, 0])
        self.assertEqual([row["verifier_feedback_evaluated"] for row in scores], [1, 1, 1, 1, 0])
        self.assertEqual(scores[0]["verifier_feedback_bonus"], 0.25)
        self.assertEqual(scores[0]["base_score"], 0.5)

    def test_mismatched_rollout_length_and_bad_indices_rejected(self):
        for indices, responses in [([0, 1], ["bad"]), ([0, 0], ["bad", "bad"]), ([-1], ["bad"]), ([2], ["bad"])]:
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                feedback.add_verifier_feedback([0.5, 0.5], indices, responses, self.cfg)

    def test_prompt_is_question_and_proposed_answer_not_solver_correctness(self):
        result = feedback.build_verifier_prompt("Where was X born?", "Y")
        self.assertEqual(result, prompts.DEFAULT_BACKWARD_VERIFIER_PREFIX.format(question="Where was X born?", candidate_answer="Y"))
        self.assertIn("YES", result)
        self.assertIn("NO", result)
        for question, answer in [("", "Y"), ("Q", ""), (None, "Y")]:
            with self.assertRaises(ValueError):
                feedback.build_verifier_prompt(question, answer)


class FakeTensor(list):
    pass


class FakeData:
    def __init__(self, data):
        self.batch = {key: value for key, value in data.items() if isinstance(value, FakeTensor)}
        self.non_tensor_batch = {key: value for key, value in data.items() if not isinstance(value, FakeTensor)}
        self.meta_info = {}

    def __len__(self):
        return len(next(iter(self.batch.values())))

    @classmethod
    def from_single_dict(cls, data):
        return cls(data)

    def repeat(self, repeat_times, interleave):
        assert interleave
        values = {**self.batch, **self.non_tensor_batch}
        return FakeData({key: type(value)(x for x in value for _ in range(repeat_times)) for key, value in values.items()})


class FakeQwenTokenizer:
    pad_token_id = 0

    def decode(self, value, **kwargs):
        return value

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def __call__(self, text, **kwargs):
        return {"input_ids": FakeTensor([FakeTensor([1, 2])]), "attention_mask": FakeTensor([FakeTensor([1, 1])])}


class RewardIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.addCleanup(self.loop.close)
        self.addCleanup(asyncio.set_event_loop, None)
        self.calls = []
        self.scripted = []
        test = self

        class FakeRollout:
            def __init__(self, **kwargs):
                self.call = dict(kwargs)
                test.calls.append(self.call)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def generate_sequences(self, batch):
                self.call["batch"] = batch
                values = test.scripted.pop(0)
                return SimpleNamespace(non_tensor_batch={"messages": [
                    value if isinstance(value, dict) else
                    {"messages": [SimpleNamespace(role="user", content="Ignore <answer>NO</answer> in prompts"),
                                   *[SimpleNamespace(role="assistant", content=part) for part in (value if isinstance(value, list) else [value])]]}
                    for value in values
                ]})

        verl = ModuleType("verl")
        verl.DataProto = FakeData
        verl.utils = ModuleType("verl.utils")
        verl.utils.torch_functional = SimpleNamespace(postprocess_data=lambda **kw: (kw["input_ids"], kw["attention_mask"]))
        self.modules = {
            "verl": verl,
            "verl.prompts": prompts,
            "verl.custom_reward.verifier_reward_function": verifier,
            "verl.custom_reward.verifier_feedback": feedback,
            "verl.custom_reward.reward_rollout": SimpleNamespace(MultiTurnRewardRollout=FakeRollout),
            "verl.utils": verl.utils,
            "verl.utils.model": SimpleNamespace(compute_position_id_with_mask=lambda value: value),
            "verl.utils.torch_functional": verl.utils.torch_functional,
            "torch": SimpleNamespace(Tensor=FakeTensor, stack=lambda values: FakeTensor(values)),
            "numpy": SimpleNamespace(array=lambda values: values, mean=lambda values: sum(values) / len(values), max=max),
        }
        patch = mock.patch.dict(sys.modules, self.modules)
        patch.start()
        self.addCleanup(patch.stop)
        self.kwargs = {
            "data": FakeData({"input_ids": FakeTensor(["one", "two", "three"]),
                              "responses": FakeTensor(["one", "two", "three"]),
                              "data_source": ["search_zero_2"] * 3}),
            "processing_class": FakeQwenTokenizer(), "config": {"prompt_length": 1000},
            "model_name": "current-solver", "base_url": "http://127.0.0.1:8001",
            "prev_model_name": "previous-solver", "prev_base_url": "http://127.0.0.1:8002",
            "reward_rollout_n": 2, "growth_lambda": 0.5,
        }
        self.format_patch = mock.patch.object(rewards, "compute_challenger_format_scores", return_value=(
            [1.0, 1.0, 0.0], ["Question one?", "Question two?", "Bad question?"], ["Alpha", "Beta", "Gamma"],
        ))
        self.format_mock = self.format_patch.start()
        self.addCleanup(self.format_patch.stop)

    def run_reward(self, growth=False, feedback_values=None, extra_kwargs=None):
        self.scripted.append(["<answer>Alpha</answer>", "<answer>Other</answer>", "<answer>Other</answer>", "<answer>Other</answer>"])
        if growth:
            self.scripted.append(["<answer>Other</answer>"] * 4)
        if feedback_values is not None:
            self.scripted.append(feedback_values)
        fn = rewards.compute_challenger_score_batch_growth if growth else rewards.compute_challenger_score_batch
        with contextlib.redirect_stdout(io.StringIO()):
            result = fn([], [], [], [], **{**self.kwargs, **(extra_kwargs or {})})
        self.assertFalse(self.scripted, "Unconsumed fake rollouts")
        return result

    def test_difficulty_default_preserved_and_zero_verifier_calls(self):
        with mock.patch.object(feedback, "_generate_verifier_responses", side_effect=AssertionError("No verifier call")):
            self.assertEqual(self.run_reward(), [1.5, 0.5, 0.0])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0]["model_name"], "current-solver")

    def test_growth_default_preserved_and_zero_verifier_calls(self):
        with mock.patch.object(feedback, "_generate_verifier_responses", side_effect=AssertionError("No verifier call")):
            self.assertEqual(self.run_reward(growth=True), [1.625, 0.5, 0.0])
        self.assertEqual([call["model_name"] for call in self.calls], ["current-solver", "previous-solver"])

    def test_explicit_false_with_dormant_server_config_preserves_both_paths(self):
        for growth in (False, True):
            with self.subTest(growth=growth), mock.patch.object(feedback, "_generate_verifier_responses", side_effect=AssertionError("No verifier call")):
                self.assertEqual(self.run_reward(growth=growth, extra_kwargs=enabled_kwargs(verifier_feedback_enabled=False)),
                                 [1.625 if growth else 1.5, 0.5, 0.0])

    def test_enabled_feedback_preserves_difficulty_and_growth_terms(self):
        for growth in (False, True):
            with self.subTest(growth=growth):
                result = self.run_reward(growth=growth, feedback_values=["<answer>YES</answer>", "<answer>NO</answer>"], extra_kwargs=enabled_kwargs())
                self.assertEqual([row["score"] for row in result], [1.875 if growth else 1.75, 0.5, 0.0])
                call = self.calls[-1]
                self.assertEqual((call["model_name"], call["base_url"]), ("test-verifier", "http://127.0.0.1:8003"))
                self.assertEqual(len(call["batch"]), 2)  # Not repeated like solver rollouts.
                self.assertIs(call["batch"].meta_info["do_sample"], False)
                prompt = call["batch"].non_tensor_batch["raw_prompt"][0][0]["content"]
                self.assertIn("Question one?", prompt)
                self.assertIn("Alpha", prompt)
                self.assertEqual(call["config"]["prompt_length"], 1536)
                self.assertEqual(self.kwargs["config"], {"prompt_length": 1000})

    def test_multi_turn_duplicate_verdict_is_invalid(self):
        result = self.run_reward(feedback_values=[
            ["<answer>YES</answer>", "<answer>YES</answer>"], "malformed",
        ], extra_kwargs=enabled_kwargs())
        self.assertEqual([row["score"] for row in result], [1.5, 0.5, 0.0])
        self.assertEqual([row["verifier_feedback_valid"] for row in result], [0.0] * 3)

    def test_feedback_rejects_earlier_verdict_with_final_tool_or_unfinished_assistant(self):
        result = self.run_reward(feedback_values=[
            {"messages": [{"role": "assistant", "content": "<answer>YES</answer>"},
                          {"role": "tool", "content": "<answer>YES</answer>"}]},
            ["<answer>YES</answer>", "I have not finished."],
        ], extra_kwargs=enabled_kwargs())
        self.assertEqual([row["score"] for row in result], [1.5, 0.5, 0.0])
        self.assertEqual([row["verifier_feedback_valid"] for row in result], [0.0] * 3)

    def test_no_valid_questions_never_calls_any_rollout(self):
        self.format_mock.return_value = ([0.0] * 3, [""] * 3, [""] * 3)
        for growth in (False, True):
            for enabled in (False, True):
                with self.subTest(growth=growth, enabled=enabled):
                    fn = rewards.compute_challenger_score_batch_growth if growth else rewards.compute_challenger_score_batch
                    result = fn([], [], [], [], **{**self.kwargs, **enabled_kwargs(verifier_feedback_enabled=enabled)})
                    actual = [row["score"] for row in result] if enabled else result
                    self.assertEqual(actual, [0.0] * 3)
        self.assertFalse(self.calls)

    def test_invalid_feedback_config_rejected_before_any_rollout(self):
        for fn in (rewards.compute_challenger_score_batch, rewards.compute_challenger_score_batch_growth):
            with self.assertRaises(ValueError):
                fn([], [], [], [], **{**self.kwargs, "verifier_feedback_enabled": True})
        self.assertFalse(self.calls)

    def test_short_verifier_result_fails_instead_of_silently_truncating(self):
        with self.assertRaisesRegex(ValueError, "exactly one response"):
            self.run_reward(feedback_values=["<answer>YES</answer>"], extra_kwargs=enabled_kwargs())


if __name__ == "__main__":
    unittest.main()
