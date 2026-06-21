"""
verifier_reward_function.py  —  Option C: Answer + Verdict

Output format: <answer>CORRECT_ANSWER. yes/no</answer>
  e.g. <answer>Auburn University. no</answer>
       <answer>Paris. yes</answer>

Reward = (1 + think + tool + em_score) / 4   (binary gate 없음)
  - em_score   : extracted factual answer vs ground truth (EM)
  - think/tool : 항상 적용, accuracy에 무관하게 tool use 강제 유지
"""

import re
import string

ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
TOOL_CALL_PATTERN = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
TOOL_RESPONSE_PATTERN = re.compile(r"<tool_response>.*?</tool_response>", re.DOTALL)

VERDICT_SUFFIXES = [". yes", ". no", ".yes", ".no", " yes", " no"]


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def em_check(prediction, golden_answer):
    if not golden_answer:
        return 0.0
    return 1.0 if normalize_answer(prediction) == normalize_answer(golden_answer) else 0.0


def extract_answer(response: str) -> str:
    matches = ANSWER_PATTERN.findall(response)
    return matches[-1].strip() if matches else ""


def extract_factual_answer(answer_text: str) -> str:
    """'CORRECT_ANSWER. yes/no' 에서 factual 부분만 추출."""
    text = answer_text.strip()
    for suffix in VERDICT_SUFFIXES:
        if text.lower().endswith(suffix):
            return text[:-len(suffix)].strip().rstrip(".")
    return text


def extract_verdict(answer_text: str) -> str:
    """'CORRECT_ANSWER. yes/no' 에서 verdict(yes/no) 추출."""
    text = answer_text.strip().lower()
    for suffix in [". yes", " yes", ".yes"]:
        if text.endswith(suffix):
            return "yes"
    for suffix in [". no", " no", ".no"]:
        if text.endswith(suffix):
            return "no"
    norm = normalize_answer(text)
    return norm if norm in ["yes", "no"] else ""


def compute_verifier_score(data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
    """
    Reward = (1 + think + tool + em_score) / 4   — binary gate 없음
    ground_truth["target"] = 실제 정답 문자열 (e.g. "Auburn University")
    """
    scores = []
    for response, gt, data_source in zip(solution_strs, ground_truths, data_sources):
        hop = int(data_source.split("_")[-1])

        try:
            ground_truth_answer = str(list(gt["target"])[0]).strip()
        except Exception:
            ground_truth_answer = ""

        answer_text = extract_answer(response)
        factual_answer = extract_factual_answer(answer_text)
        em_score = em_check(factual_answer, ground_truth_answer)

        think_reward = 1.0 if THINK_PATTERN.search(response) else 0.0

        # tool_responses는 framework가 user turn에 inject하므로 response_ids에 없음
        # → tool_calls만으로 판단
        tool_calls = len(TOOL_CALL_PATTERN.findall(response))
        if hop == 1:
            tool_reward = 1.0
        elif tool_calls > 0:
            tool_reward = min((1 + tool_calls) / hop, 1.0)
        else:
            tool_reward = 0.0

        score = (1 + think_reward + tool_reward + em_score) / 4
        scores.append(score)

    return scores
