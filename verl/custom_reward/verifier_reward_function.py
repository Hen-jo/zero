"""
verifier_reward_function.py

Reward function for verifier GRPO training.

Same structure as compute_challenger_format_scores:
  final_score = (1 + think + tool + ans) / 4  if accuracy_correct else 0

  - accuracy_correct : integrity gate (pred yes/no matches ground-truth label)
  - think_reward     : used <think> at least once
  - tool_reward      : min((1 + tool_calls) / hop, 1.0) if calls match responses
                       hop==1 → 1.0 (no search needed)
  - ans_reward       : output is yes or no (always 1.0 when gate passes)
"""

import re
import string

ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
TOOL_CALL_PATTERN = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
TOOL_RESPONSE_PATTERN = re.compile(r"<tool_response>.*?</tool_response>", re.DOTALL)


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
    Reward = (1 + think + tool + ans) / 4  if accuracy_correct else 0.0
    Returns a list of float scores (batch manager interface).
    """
    scores = []
    for response, gt, data_source in zip(solution_strs, ground_truths, data_sources):
        hop = int(data_source.split("_")[-1])

        try:
            label = str(list(gt["target"])[0]).lower().strip()
        except Exception:
            label = ""

        pred = normalize_answer(extract_answer(response))
        accuracy_correct = pred == label

        # think_reward
        think_reward = 1.0 if THINK_PATTERN.search(response) else 0.0

        # tool_reward (same logic as challenger)
        tool_calls = len(TOOL_CALL_PATTERN.findall(response))
        tool_responses = len(TOOL_RESPONSE_PATTERN.findall(response))
        if hop == 1:
            tool_reward = 1.0
        elif tool_calls == tool_responses and tool_calls > 0:
            tool_reward = min((1 + tool_calls) / hop, 1.0)
        else:
            tool_reward = 0.0

        # ans_reward: yes or no format, but requires search evidence for hop > 1
        if hop > 1 and tool_calls == 0:
            ans_reward = 0.0
        else:
            ans_reward = 1.0 if pred in ["yes", "no"] else 0.0

        score = sum([1, think_reward, tool_reward, ans_reward]) / 4 if accuracy_correct else 0.0
        scores.append(score)

    return scores
