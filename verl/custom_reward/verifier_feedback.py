"""Experimental, opt-in answer-validity feedback to the proposer.

The verifier is trained with ``answer_em_proxy`` labels. Its YES verdict is only
an estimated answer-validity bonus, not a verified reasoning/causal-chain label.
For both difficulty and growth rewards:

    total = existing_reward + coefficient * (strict_verdict is YES)

NO and malformed verdicts both add zero. In particular, malformed verdicts are
never treated as solver failures that could increase the difficulty reward.
Disabled feedback returns the original scores and does not import the rollout
stack, tokenize prompts, or make a verifier request.

Enabled use requires verifier_tokenizer_compatible=True: the remote verifier
must share the proposer's tokenizer and vocabulary. The existing rollout helper
uses the proposer tokenizer for encoding/decoding, and cannot verify the remote
server's tokenizer identity. The flag is an explicit operator acknowledgment,
not an automatic compatibility check.
"""

import asyncio
import math
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from numbers import Real
from urllib.parse import urlsplit


@dataclass(frozen=True)
class VerifierFeedbackConfig:
    enabled: bool = False
    coefficient: float = 0.0
    base_url: str = ""
    model_name: str = ""
    prompt_length: int = 1536
    tokenizer_compatible: bool = False

    @classmethod
    def from_kwargs(cls, kwargs):
        enabled = kwargs.get("verifier_feedback_enabled", False)
        coefficient = kwargs.get("verifier_feedback_coefficient", 0.0)
        if type(enabled) is not bool:
            raise ValueError("verifier_feedback_enabled must be a boolean")
        if isinstance(coefficient, bool) or not isinstance(coefficient, Real):
            raise ValueError("verifier_feedback_coefficient must be a finite nonnegative number")
        if not math.isfinite(coefficient) or coefficient < 0:
            raise ValueError("verifier_feedback_coefficient must be a finite nonnegative number")
        tokenizer_compatible = kwargs.get("verifier_tokenizer_compatible", False)
        if type(tokenizer_compatible) is not bool:
            raise ValueError("verifier_tokenizer_compatible must be a boolean")
        if not enabled:
            # Dormant server settings never trigger a connection or model load.
            return cls(coefficient=float(coefficient))
        if coefficient <= 0:
            raise ValueError("Enabled verifier feedback requires an explicit positive coefficient")
        if not tokenizer_compatible:
            raise ValueError("Enabled verifier feedback requires verifier_tokenizer_compatible=True: the verifier must share the proposer tokenizer/vocabulary")
        base_url = kwargs.get("verifier_base_url")
        model_name = kwargs.get("verifier_model_name")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("Enabled verifier feedback requires verifier_base_url")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("Enabled verifier feedback requires verifier_model_name")
        base_url = base_url.strip().rstrip("/")
        try:
            url = urlsplit(base_url)
            port = url.port
            valid_url = (
                url.scheme in ("http", "https") and bool(url.hostname)
                and not url.path and not url.query and not url.fragment
                and url.username is None and url.password is None
                and (port is None or 0 < port <= 65535)
                and not any(ch.isspace() for ch in base_url)
            )
        except ValueError:
            valid_url = False
        if not valid_url:
            raise ValueError("verifier_base_url must be an HTTP(S) server origin, without /v1, credentials, query, or fragment")
        prompt_length = kwargs.get("verifier_prompt_length", 1536)
        if type(prompt_length) is not int or prompt_length <= 0:
            raise ValueError("verifier_prompt_length must be a positive integer")
        return cls(True, float(coefficient), base_url, model_name.strip(), prompt_length, True)


def build_verifier_prompt(question, candidate_answer):
    """Use the same question/candidate-answer task as verifier training."""
    from verl.prompts import DEFAULT_BACKWARD_VERIFIER_PREFIX

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Verifier feedback requires a nonempty question")
    if not isinstance(candidate_answer, str) or not candidate_answer.strip():
        raise ValueError("Verifier feedback requires a nonempty candidate answer")
    return DEFAULT_BACKWARD_VERIFIER_PREFIX.format(
        question=question.strip(), candidate_answer=candidate_answer.strip()
    )


def add_verifier_feedback(base_scores, indices, responses, config):
    """Merge strict verdict bonuses and numeric diagnostics into batch scores."""
    if not config.enabled:
        return base_scores
    from verl.custom_reward.verifier_reward_function import parse_verifier_verdict

    if len(indices) != len(responses):
        raise ValueError("Verifier must return exactly one response per evaluated question")
    if len(set(indices)) != len(indices) or any(
        type(idx) is not int or not 0 <= idx < len(base_scores) for idx in indices
    ):
        raise ValueError("Verifier feedback indices must be unique valid batch indices")
    result = [
        {
            "score": score,
            "base_score": score,
            "verifier_feedback_evaluated": 0.0,
            "verifier_feedback_valid": 0.0,
            "verifier_feedback_score": 0.0,
            "verifier_feedback_bonus": 0.0,
        }
        for score in base_scores
    ]
    for idx, response in zip(indices, responses):
        verdict = parse_verifier_verdict(response)
        raw_score = float(verdict is True)
        bonus = config.coefficient * raw_score
        result[idx].update(
            score=base_scores[idx] + bonus,
            verifier_feedback_evaluated=1.0,
            verifier_feedback_valid=float(verdict is not None),
            verifier_feedback_score=raw_score,
            verifier_feedback_bonus=bonus,
        )
    return result


async def _generate_verifier_responses(questions, candidate_answers, processing_class, rollout_config, config):
    """One tool-aware verifier rollout per proposed question, without repetition."""
    # Heavy imports stay behind the explicit enabled branch. Reuse the existing
    # SGLang /v1/completions + search tool implementation, not a tool-less API.
    import numpy as np
    import torch
    from verl import DataProto
    from verl.custom_reward.reward_rollout import MultiTurnRewardRollout
    from verl.custom_reward.verifier_reward_function import extract_verifier_response
    from verl.utils.model import compute_position_id_with_mask
    import verl.utils.torch_functional as verl_F

    if len(questions) != len(candidate_answers):
        raise ValueError("Verifier questions and candidate answers must have equal length")
    rollout_config = deepcopy(rollout_config)
    rollout_config["prompt_length"] = config.prompt_length
    gen_batch = defaultdict(list)
    for idx, (question, answer) in enumerate(zip(questions, candidate_answers)):
        messages = [{"role": "user", "content": build_verifier_prompt(question, answer)}]
        raw_prompt = processing_class.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        model_inputs = processing_class(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            max_length=config.prompt_length,
            pad_token_id=processing_class.pad_token_id,
            left_pad=True,
            # Never silently drop the question, candidate, or task instructions.
            truncation="error",
        )
        position_ids = compute_position_id_with_mask(attention_mask)
        row = {
            "index": idx,
            "raw_prompt": messages,
            "full_prompts": raw_prompt,
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0],
            "tools_kwargs": {"search": {"create_kwargs": {
                "data_source": "search_zero", "ground_truth": "", "question": question,
            }}},
            "interaction_kwargs": {},
        }
        for key, value in row.items():
            gen_batch[key].append(value)
    for key, values in gen_batch.items():
        gen_batch[key] = torch.stack(values) if isinstance(values[0], torch.Tensor) else np.array(values)
    batch = DataProto.from_single_dict(gen_batch)
    # Deterministic feedback reduces reward noise; one sample per proposer row.
    batch.meta_info["do_sample"] = False
    async with MultiTurnRewardRollout(
        config=rollout_config, processing_class=processing_class,
        model_name=config.model_name, base_url=config.base_url,
    ) as reward_rollout:
        outputs = await reward_rollout.generate_sequences(batch)
    responses = []
    for item in outputs.non_tensor_batch["messages"]:
        # Ignore user prompt/tool documents (which can themselves contain tags),
        # but retain ALL assistant turns so duplicate verdicts are not accepted.
        # The last message must itself be an assistant verdict: an earlier
        # verdict followed by a tool response or an unfinished turn is invalid.
        responses.append(extract_verifier_response(item["messages"]))
    return responses


def finish_challenger_scores(base_scores, indices, questions, candidate_answers, feedback_config, **kwargs):
    """Apply experimental feedback only after the existing reward is computed."""
    if not feedback_config.enabled:
        return base_scores
    selected_questions = [questions[idx] for idx in indices]
    selected_answers = [candidate_answers[idx] for idx in indices]
    if not indices:
        return add_verifier_feedback(base_scores, [], [], feedback_config)
    responses = asyncio.get_event_loop().run_until_complete(_generate_verifier_responses(
        selected_questions, selected_answers, kwargs["processing_class"], kwargs["config"], feedback_config,
    ))
    return add_verifier_feedback(base_scores, indices, responses, feedback_config)
