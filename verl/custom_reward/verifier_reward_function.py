"""
verifier_reward_function.py  —  Option C: Answer + Verdict

Output format: <answer>CORRECT_ANSWER. yes/no</answer>
  e.g. <answer>Auburn University. no</answer>
       <answer>Paris. yes</answer>

Reward = (1 + think + tool + em_score) / 4   (binary gate 없음)
  - em_score   : extracted factual answer vs ground truth (EM)
  - think/tool : 항상 적용 — accuracy 무관하게 tool use 강제 유지
  - tool_reward: tool_responses 체크 제거 (response_ids에 <tool_response> 없음)

Positive transfer 근거:
  Challenger: search → 사실추론 → 긴 텍스트(질문+답)
  Verifier:  search → 사실추론 → 긴 텍스트(정답+verdict)
  → 동일한 output 분포 → weight 충돌 없음
"""

import re
import string

ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
TOOL_CALL_PATTERN = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)

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
    """'CORRECT_ANSWER. yes/no' 에서 verdict 추출 (backward verification용)."""
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
    Reward = (1 + think + tool + em_score + verdict_score) / 5
    ground_truth["target"]     = 실제 정답 문자열 (e.g. "Auburn University")
    ground_truth["is_correct"] = solver 답변 정오 여부 (bool)
    """
    scores = []
    for response, gt, data_source in zip(solution_strs, ground_truths, data_sources):
        hop = int(data_source.split("_")[-1])

        try:
            ground_truth_answer = str(list(gt["target"])[0]).strip()
        except Exception:
            ground_truth_answer = ""

        try:
            is_correct = bool(list(gt["is_correct"])[0])
        except Exception:
            is_correct = None

        answer_text = extract_answer(response)
        factual_answer = extract_factual_answer(answer_text)
        em_score = em_check(factual_answer, ground_truth_answer)

        think_reward = 1.0 if THINK_PATTERN.search(response) else 0.0

        tool_calls = len(TOOL_CALL_PATTERN.findall(response))
        if hop == 1:
            tool_reward = 1.0 if tool_calls > 0 else 0.0
        elif tool_calls > 0:
            tool_reward = min((1 + tool_calls) / hop, 1.0)
        else:
            tool_reward = 0.0

        # verdict_score: yes/no가 실제 is_correct label과 일치하는지
        # em_score가 있어야만 유효 (yes/no shortcut 방지)
        if is_correct is not None and em_score > 0:
            verdict = extract_verdict(answer_text)
            expected = "yes" if is_correct else "no"
            verdict_score = 1.0 if verdict == expected else 0.0
        else:
            verdict_score = 0.0

        score = (1 + think_reward + tool_reward + em_score + verdict_score) / 5
        scores.append(score)

    return scores
