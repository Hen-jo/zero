"""Rewards for trajectory-conditioned inverse question reconstruction.

The verifier receives a solver's terminal search query and reconstructs the
hidden source question by searching backward. GRPO never sees the source
question in its prompt: it is used only here as a fixed reference target.

The reward deliberately does not score a free-form "good/bad question" verdict.
It scores a reconstruction using deterministic keyword/entity-relation overlap,
an optional frozen Sentence-BERT similarity model, and output/search hygiene.
"""

import json
import os
import re
import string
import warnings

import numpy as np


ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
TOOL_CALL_PATTERN = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]*")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "has", "have", "how", "in", "is", "it", "of", "on", "or",
    "the", "that", "their", "this", "to", "was", "were", "what", "when", "where",
    "which", "who", "whom", "why", "with", "would",
}

_SBERT_MODELS = {}
_SBERT_WARNED = set()


def normalize_text(text: str) -> str:
    def remove_articles(value):
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def remove_punctuation(value):
        return "".join(char for char in value if char not in set(string.punctuation))

    return " ".join(remove_articles(remove_punctuation(str(text).lower())).split())


def extract_answer(response: str) -> str:
    matches = ANSWER_PATTERN.findall(response)
    return matches[-1].strip() if matches else ""


def extract_keywords(text: str) -> list[str]:
    seen = set()
    keywords = []
    for token in TOKEN_PATTERN.findall(str(text).lower()):
        token = token.strip("'-")
        if len(token) < 2 or token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        keywords.append(token)
    return keywords


def f1_score(predicted: list[str], reference: list[str]) -> float:
    predicted_set = set(predicted)
    reference_set = set(reference)
    if not predicted_set or not reference_set:
        return 0.0
    overlap = len(predicted_set.intersection(reference_set))
    precision = overlap / len(predicted_set)
    recall = overlap / len(reference_set)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _first_value(ground_truth, key: str, default=""):
    try:
        value = ground_truth[key]
        if isinstance(value, (str, bytes)):
            return value
        return list(value)[0]
    except (IndexError, KeyError, TypeError):
        return default


def _reference_keywords(ground_truth, reference_question: str) -> list[str]:
    raw_keywords = _first_value(ground_truth, "reference_keywords", "")
    if raw_keywords:
        try:
            parsed = json.loads(str(raw_keywords))
            if isinstance(parsed, list):
                return [str(keyword) for keyword in parsed]
        except json.JSONDecodeError:
            pass
    return extract_keywords(reference_question)


def _fallback_semantic_scores(predictions: list[str], references: list[str]) -> list[float]:
    """Safe lexical fallback when the optional SBERT checkpoint is unavailable."""
    return [
        f1_score(extract_keywords(prediction), extract_keywords(reference))
        for prediction, reference in zip(predictions, references)
    ]


def sentence_bert_scores(
    predictions: list[str],
    references: list[str],
    semantic_model_name: str | None = None,
    semantic_model_device: str | None = None,
) -> list[float]:
    """Return cosine similarities from a frozen local Sentence-BERT model.

    Set VERIFIER_SBERT_MODEL to a local model path or Hugging Face identifier.
    If the dependency/checkpoint is not available, training remains runnable with
    a keyword-F1 fallback and emits a one-time warning.
    """
    model_name = semantic_model_name or os.environ.get("VERIFIER_SBERT_MODEL", "")
    device = semantic_model_device or os.environ.get("VERIFIER_SBERT_DEVICE", "cpu")
    if not model_name:
        return _fallback_semantic_scores(predictions, references)

    cache_key = (model_name, device)
    try:
        if cache_key not in _SBERT_MODELS:
            from sentence_transformers import SentenceTransformer

            _SBERT_MODELS[cache_key] = SentenceTransformer(model_name, device=device)
        model = _SBERT_MODELS[cache_key]
        embeddings = model.encode(
            predictions + references,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        split = len(predictions)
        prediction_embeddings = embeddings[:split]
        reference_embeddings = embeddings[split:]
        return [
            float(np.clip(np.dot(prediction_embedding, reference_embedding), 0.0, 1.0))
            for prediction_embedding, reference_embedding in zip(
                prediction_embeddings, reference_embeddings
            )
        ]
    except Exception as exc:  # pragma: no cover - depends on runtime model cache
        if cache_key not in _SBERT_WARNED:
            _SBERT_WARNED.add(cache_key)
            warnings.warn(
                f"Could not load Sentence-BERT model {model_name!r} on {device!r} "
                f"({type(exc).__name__}: {exc}). Falling back to keyword F1.",
                RuntimeWarning,
            )
        return _fallback_semantic_scores(predictions, references)


def has_answer_leakage(reconstructed_question: str, reference_answer: str) -> bool:
    answer = normalize_text(reference_answer)
    question = normalize_text(reconstructed_question)
    # One-character or article-only answers produce many false positives.
    return len(answer) >= 4 and answer in question


def compute_verifier_score(
    data_sources,
    solution_strs,
    ground_truths,
    extra_infos,
    semantic_model_name=None,
    semantic_model_device=None,
    **kwargs,
):
    """Score reverse-search question reconstruction for GRPO.

    R = 0.10 format + 0.10 think + 0.10 search + 0.30 keyword-F1
        + 0.35 Sentence-BERT cosine + 0.05 no-answer-leakage.

    Search is binary by design: a model is rewarded for attempting evidence
    retrieval, never for inflating the number of tool calls.
    """
    reconstructed_questions = [extract_answer(response) for response in solution_strs]
    reference_questions = [
        str(_first_value(ground_truth, "reference_question", "")).strip()
        for ground_truth in ground_truths
    ]
    semantic_scores = sentence_bert_scores(
        reconstructed_questions,
        reference_questions,
        semantic_model_name=semantic_model_name,
        semantic_model_device=semantic_model_device,
    )

    scores = []
    for response, reconstructed_question, reference_question, ground_truth, semantic_score in zip(
        solution_strs,
        reconstructed_questions,
        reference_questions,
        ground_truths,
        semantic_scores,
    ):
        if not reconstructed_question or not reference_question:
            scores.append(0.0)
            continue

        reference_answer = str(_first_value(ground_truth, "target", "")).strip()
        keyword_score = f1_score(
            extract_keywords(reconstructed_question),
            _reference_keywords(ground_truth, reference_question),
        )
        format_score = float(
            8 <= len(reconstructed_question) <= 384
            and reconstructed_question.strip().endswith("?")
        )
        think_score = float(bool(THINK_PATTERN.search(response)))
        search_score = float(bool(TOOL_CALL_PATTERN.search(response)))
        no_leakage_score = float(
            not has_answer_leakage(reconstructed_question, reference_answer)
        )

        score = (
            0.10 * format_score
            + 0.10 * think_score
            + 0.10 * search_score
            + 0.30 * keyword_score
            + 0.35 * semantic_score
            + 0.05 * no_leakage_score
        )
        scores.append(float(np.clip(score, 0.0, 1.0)))

    return scores
