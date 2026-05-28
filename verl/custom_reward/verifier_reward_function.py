"""
verifier_reward_function.py

Reward function for verifier GRPO training.

The verifier is trained to judge whether a solver's proposed answer is correct.
  - Ground truth label in training data: "yes" or "no"
  - Verifier must output <answer>yes</answer> or <answer>no</answer>
  - Reward = 0.5 * format_reward + 0.5 * accuracy_reward

Format reward (same philosophy as challenger):
  - think_reward : used <think> at least once
  - tool_reward  : used appropriate number of searches for the hop count
                   hop==1 → no search needed (reward=1.0)
                   hop>=2 → min((1 + tool_calls) / hop, 1.0) if calls match responses
  - ans_reward   : output is yes or no
  - format = (think + tool + ans) / 3, gated on valid answer

Accuracy reward:
  - 1.0 if predicted yes/no matches ground-truth label, 0.0 otherwise
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


def compute_verifier_format_score(response: str, hop: int) -> float:
    think_reward = 1.0 if THINK_PATTERN.search(response) else 0.0

    tool_calls = len(TOOL_CALL_PATTERN.findall(response))
    tool_responses = len(TOOL_RESPONSE_PATTERN.findall(response))

    if hop == 1:
        tool_reward = 1.0
    elif tool_calls == tool_responses and tool_calls > 0:
        tool_reward = min((1 + tool_calls) / hop, 1.0)
    else:
        tool_reward = 0.0

    ans = normalize_answer(extract_answer(response))
    ans_reward = 1.0 if ans in ["yes", "no"] else 0.0

    if ans_reward == 0.0:
        return 0.0

    return (think_reward + tool_reward + ans_reward) / 3


def compute_verifier_score(data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
    """
    Reward = 0.5 * format_reward + 0.5 * accuracy_reward
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
        accuracy_reward = 1.0 if pred == label else 0.0
        format_reward = compute_verifier_format_score(response, hop)

        scores.append(0.5 * format_reward + 0.5 * accuracy_reward)

    return scores
