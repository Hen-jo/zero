"""
verifier_reward_function.py

Reward function for verifier GRPO training.

Fix 1: format reward(think/tool) decoupled from accuracy gate
Fix 2: KL via +algorithm.kl_coef in shell scripts
Bug fix: tool_responses 항상 0 버그 수정
  - <tool_response>는 framework가 user turn에 inject → response_ids에 없음
  - tool_calls만으로 tool_reward 판단
"""

import re
import string

ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
TOOL_CALL_PATTERN = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)


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
    Fix 1: format_score = (think + tool) / 2  — accuracy gate 없이 항상 적용
    accuracy_score = (1 + ans_reward) / 2  if accuracy_correct else 0.0
    final = 0.4 * format_score + 0.6 * accuracy_score

    Bug fix: tool_responses 체크 제거 (response_ids에 <tool_response> 없음)
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

        think_reward = 1.0 if THINK_PATTERN.search(response) else 0.0

        tool_calls = len(TOOL_CALL_PATTERN.findall(response))
        if hop == 1:
            tool_reward = 1.0
        elif tool_calls > 0:
            tool_reward = min((1 + tool_calls) / hop, 1.0)
        else:
            tool_reward = 0.0

        ans_reward = 1.0 if pred in ["yes", "no"] else 0.0
        if hop > 1 and tool_calls == 0:
            ans_reward = 0.0

        format_score = (think_reward + tool_reward) / 2
        accuracy_score = (1.0 + ans_reward) / 2 if accuracy_correct else 0.0
        score = 0.4 * format_score + 0.6 * accuracy_score
        scores.append(score)

    return scores
