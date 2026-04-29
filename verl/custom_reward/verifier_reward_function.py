"""
verifier_reward_function.py

Reward function for verifier GRPO training.

The verifier is trained to judge whether a solver's proposed answer is correct.
  - Ground truth label in training data: "yes" or "no"
  - Verifier must output <answer>yes</answer> or <answer>no</answer>
  - Reward = 1.0 if judgment matches label, 0.0 otherwise
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


def extract_answer(response: str) -> str:
    matches = ANSWER_PATTERN.findall(response)
    return matches[-1].strip() if matches else ""


def compute_verifier_score(data, **kwargs):
    """
    Reward = 1.0 if the verifier's yes/no judgment matches the ground-truth label.
    """
    reward_model = data.non_tensor_batch.get("reward_model", None)
    responses = data.non_tensor_batch.get("responses", None)

    if responses is None:
        response_ids = data.batch["responses"]
        # Decode lazily — tokenizer should be passed via kwargs if needed
        tokenizer = kwargs.get("tokenizer", None)
        if tokenizer is not None:
            responses = [tokenizer.decode(ids, skip_special_tokens=True)
                         for ids in response_ids]
        else:
            responses = [""] * response_ids.shape[0]

    batch_size = len(responses)
    response_len = data.batch["responses"].shape[1]
    reward_tensor = torch.zeros(batch_size, response_len)

    scores = []
    for i, response in enumerate(responses):
        try:
            label = str(list(reward_model[i]["ground_truth"]["target"])[0]).lower().strip()
        except Exception:
            label = ""

        pred = normalize_answer(extract_answer(response))
        score = 1.0 if pred == label else 0.0
        scores.append(score)
        reward_tensor[i, -1] = score

    return {"reward_tensor": reward_tensor, "reward_extra_info": {"accuracy": scores}}
