"""
verifier_reward_function.py

Reward function for verifier GRPO training.
The verifier generates a reasoning trace and an answer; reward = EM vs ground truth.
"""

import re
import string

import torch

ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


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
            return 1.0
    return 0.0


def extract_answer(response: str) -> str:
    matches = ANSWER_PATTERN.findall(response)
    return matches[-1].strip() if matches else ""


def compute_verifier_score(data, **kwargs):
    """
    Reward function for verifier training.
    Returns reward_tensor of shape (batch, response_len) with reward at last token.
    """
    reward_model = data.non_tensor_batch.get("reward_model", None)
    responses = data.non_tensor_batch.get("responses", None)

    # Fallback: decode from token ids
    if responses is None:
        import os
        from transformers import AutoTokenizer
        model_path = kwargs.get("model_name", "Qwen/Qwen2.5-3B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        response_ids = data.batch["responses"]
        responses = [tokenizer.decode(ids, skip_special_tokens=True) for ids in response_ids]

    batch_size = len(responses)
    response_len = data.batch["responses"].shape[1]
    reward_tensor = torch.zeros(batch_size, response_len)

    scores = []
    for i, response in enumerate(responses):
        try:
            gt = list(reward_model[i]["ground_truth"]["target"])
        except Exception:
            gt = []

        pred = extract_answer(response)
        score = em_check(pred, gt) if gt else 0.0
        scores.append(score)
        # Place reward at the last non-padding token
        reward_tensor[i, -1] = score

    return {"reward_tensor": reward_tensor, "reward_extra_info": {"em": scores}}
