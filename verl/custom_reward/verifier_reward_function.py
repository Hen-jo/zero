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


def compute_verifier_score(data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
    """
    Reward = 1.0 if the verifier's yes/no judgment matches the ground-truth label.
    Returns a list of float scores (batch manager interface).
    """
    scores = []
    for response, gt in zip(solution_strs, ground_truths):
        try:
            label = str(list(gt["target"])[0]).lower().strip()
        except Exception:
            label = ""

        pred = normalize_answer(extract_answer(response))
        score = 1.0 if pred == label else 0.0
        scores.append(score)

    return scores
