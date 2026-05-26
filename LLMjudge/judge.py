"""
JudgeResult dataclass, single Judge, and MultiModelJudge (ensemble average).
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

from llm_client import call_llm
from judge_prompts import SYSTEM_PROMPT, build_user_prompt


@dataclass
class JudgeResult:
    question: str
    answer: str
    score: Optional[float]          # 0.0 ~ 1.0, None = parse failed
    reasoning: str
    raw_response: str
    model: str                      # which model produced this
    parse_error: bool = False

    def to_dict(self) -> dict:
        return {
            "question":      self.question,
            "answer":        self.answer,
            "score":         self.score,
            "reasoning":     self.reasoning,
            "raw_response":  self.raw_response,
            "model":         self.model,
            "parse_error":   self.parse_error,
        }


@dataclass
class EnsembleResult:
    question: str
    answer: str
    score: Optional[float]          # averaged score, None if all models failed
    per_model: dict[str, Optional[float]] = field(default_factory=dict)
    reasonings: dict[str, str]      = field(default_factory=dict)
    parse_errors: dict[str, bool]   = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "question":     self.question,
            "answer":       self.answer,
            "score":        self.score,
        }
        for model_key, s in self.per_model.items():
            safe = model_key.replace("-", "_").replace(".", "_")
            d[f"score_{safe}"]      = s
            d[f"reasoning_{safe}"]  = self.reasonings.get(model_key, "")
            d[f"parse_error_{safe}"] = self.parse_errors.get(model_key, False)
        return d


def _extract_json(text: str) -> dict:
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"No valid JSON found in: {text[:300]}")


class Judge:
    """Single-model judge."""

    def __init__(self, model_key: str, max_tokens: int = 1024):
        self.model_key = model_key
        self.max_tokens = max_tokens

    def judge(self, question: str, answer: str, trace: str | None = None, hop_n: int = 2) -> JudgeResult:
        raw = call_llm(
            model_key=self.model_key,
            system=SYSTEM_PROMPT,
            user=build_user_prompt(question, answer, trace=trace, hop_n=hop_n),
            max_tokens=self.max_tokens,
        )
        try:
            data = _extract_json(raw)
            score = float(data["score"])
            if not (0.0 <= score <= 1.0):
                raise ValueError(f"Score out of range: {score}")
            return JudgeResult(
                question=question,
                answer=answer,
                score=round(score, 4),
                reasoning=str(data.get("reasoning", "")),
                raw_response=raw,
                model=self.model_key,
                parse_error=False,
            )
        except Exception as e:
            return JudgeResult(
                question=question,
                answer=answer,
                score=None,
                reasoning=f"PARSE ERROR: {e}",
                raw_response=raw,
                model=self.model_key,
                parse_error=True,
            )


class MultiModelJudge:
    """
    Call multiple models in parallel and average their scores.
    Default ensemble: claude-sonnet-4-6, gpt-5.4, gemini-3.1-pro-preview
    """

    def __init__(
        self,
        model_keys: list[str] | None = None,
        max_tokens: int = 1024,
    ):
        from config import DEFAULT_JUDGE_MODELS
        self.model_keys = model_keys or DEFAULT_JUDGE_MODELS
        self.judges = [Judge(m, max_tokens) for m in self.model_keys]

    def judge(self, question: str, answer: str, trace: str | None = None, hop_n: int = 2) -> EnsembleResult:
        results: dict[str, JudgeResult] = {}

        # 3개 모델 병렬 호출
        with ThreadPoolExecutor(max_workers=len(self.judges)) as ex:
            futures = {ex.submit(j.judge, question, answer, trace, hop_n): j.model_key for j in self.judges}
            for future in as_completed(futures):
                model_key = futures[future]
                try:
                    results[model_key] = future.result()
                except Exception as e:
                    results[model_key] = JudgeResult(
                        question=question, answer=answer,
                        score=None, reasoning=f"EXCEPTION: {e}",
                        raw_response="", model=model_key, parse_error=True,
                    )

        valid_scores = [r.score for r in results.values() if r.score is not None]
        avg = round(sum(valid_scores) / len(valid_scores), 4) if valid_scores else None

        return EnsembleResult(
            question=question,
            answer=answer,
            score=avg,
            per_model={k: r.score for k, r in results.items()},
            reasonings={k: r.reasoning for k, r in results.items()},
            parse_errors={k: r.parse_error for k, r in results.items()},
        )
