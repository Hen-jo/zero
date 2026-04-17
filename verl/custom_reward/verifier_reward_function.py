import re
import string
import logging
import os

from verl.prompts import ANSWER_PATTERN

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_verifier_score(data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
    """
    Reward function for Verifier (backward reasoner) training.

    The Verifier receives (question + candidate_answer) and backward-searches
    through hops to decide YES/NO.

    Reward:
      1.0  if the YES/NO prediction matches the is_correct label
      0.0  otherwise
    """
    scores = []
    for response, gt in zip(solution_strs, ground_truths):
        ans_matches = re.findall(ANSWER_PATTERN, response, re.DOTALL)
        pred = normalize_answer(ans_matches[-1]) if ans_matches else ""

        is_correct = gt["is_correct"]           # bool: whether candidate_answer is correct
        pred_yes = pred.startswith("yes")

        scores.append(1.0 if pred_yes == is_correct else 0.0)

    return scores
