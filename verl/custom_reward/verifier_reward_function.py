"""Strict verdict reward for the backward verifier.

``is_correct`` is a boolean answer exact-match (EM) proxy label, not supervision
of the validity of the intermediate reasoning or causal chain. Malformed output
always scores zero, including on negative examples. This module intentionally
uses only the standard library so it can run in CPU reward workers.
"""

import re
from collections.abc import Mapping

LABEL_SOURCE = "answer_em_proxy"
_ANSWER = re.compile(r"<answer>\s*(YES|NO)\s*</answer>")
# Also count malformed/case-varied answer tags, rather than accepting the last
# well-formed tag and silently overlooking a second contradictory verdict.
_ANSWER_TAG = re.compile(r"<\s*/?\s*answer\b", re.IGNORECASE)


def parse_verifier_verdict(response):
    """Return True/False for exactly one uppercase YES/NO tag, else None.

    Reasoning outside the tag and whitespace around the verdict are allowed.
    Bare verdicts, prefixes (YES because...), mixed case, duplicate/nested tags,
    attributes, and unclosed or otherwise malformed answer tags are invalid.
    None is deliberately distinct from the valid negative verdict False.
    """
    if not isinstance(response, str) or len(_ANSWER_TAG.findall(response)) != 2:
        return None
    matches = list(_ANSWER.finditer(response))
    if len(matches) != 1:
        return None
    return matches[0].group(1) == "YES"


def extract_verifier_response(messages):
    """Return an assistant-only verdict trace, or empty text for invalid output.

    A verdict must occur in the final message, and that message must be from
    the assistant. All assistant turns are checked for duplicate answer tags.
    Tags in the user prompt or tool observations never count as a verdict.
    Structurally missing role metadata fails rather than falling back to raw
    decoded tokens, which can contain untrusted tool/user text.
    """
    if isinstance(messages, (str, bytes, Mapping)) or messages is None:
        raise ValueError("Verifier requires a sequence of role-tagged messages")
    try:
        messages = list(messages)
    except TypeError as exc:
        raise ValueError("Verifier requires a sequence of role-tagged messages") from exc
    fields = []
    for message in messages:
        if isinstance(message, Mapping):
            role, content = message.get("role"), message.get("content")
        else:
            role, content = getattr(message, "role", None), getattr(message, "content", None)
        if not isinstance(role, str):
            raise ValueError("Verifier structured messages must include a role")
        fields.append((role, content))
    if not fields or fields[-1][0] != "assistant":
        return ""
    if parse_verifier_verdict(fields[-1][1]) is None:
        return ""
    assistant_contents = [content for role, content in fields if role == "assistant"]
    if any(not isinstance(content, str) for content in assistant_contents):
        return ""
    response = "\n".join(assistant_contents)
    return response if parse_verifier_verdict(response) is not None else ""


def _role_grounded_batch_responses(data, batch_size):
    non_tensor_batch = getattr(data, "non_tensor_batch", None)
    if not isinstance(non_tensor_batch, Mapping) or "messages" not in non_tensor_batch:
        raise ValueError("Verifier batch data requires structured non_tensor_batch['messages']; raw decoded responses are unsafe")
    rows = non_tensor_batch["messages"]
    if rows is None or len(rows) != batch_size:
        raise ValueError("Verifier structured messages must have one trace per response")
    responses = []
    for row in rows:
        if isinstance(row, Mapping):
            if "messages" not in row:
                raise ValueError("Verifier structured trace is missing its messages field")
            row = row["messages"]
        responses.append(extract_verifier_response(row))
    return responses


def _score_one(response, ground_truth):
    if not isinstance(ground_truth, Mapping):
        raise ValueError("Verifier ground_truth must contain boolean is_correct")
    label = ground_truth.get("is_correct")
    if type(label) is not bool:
        raise ValueError("Verifier ground_truth.is_correct must be a boolean, not a truthy value")
    if ground_truth.get("label_source") != LABEL_SOURCE:
        raise ValueError("Verifier labels must use label_source='answer_em_proxy'")
    verdict = parse_verifier_verdict(response)
    return float(verdict is not None and verdict == label)


def compute_verifier_score(
    data_sources=None, solution_strs=None, ground_truths=None, extra_infos=None,
    *, data_source=None, solution_str=None, ground_truth=None, extra_info=None,
    **kwargs,
):
    """Reward a valid verdict matching an answer-EM proxy label with 1, else 0.

    Accept both the batch reward-manager interface (including positional batch
    arguments) and the singular keyword interface for assistant-only text.
    The training batch path requires role-grounded messages whenever ``data``
    is supplied. Without ``data``, the caller asserts that response strings are
    assistant-only; do not pass unfiltered multi-turn decoded token tensors.
    No model/retrieval request is made. Invalid labels fail, not silently cast.
    """
    if solution_strs is None:
        if "data" in kwargs:
            raise ValueError("Structured verifier data requires the batch reward interface")
        return _score_one(solution_str, ground_truth)
    if solution_str is not None or ground_truth is not None:
        raise ValueError("Do not mix scalar and batch verifier reward arguments")
    if ground_truths is None or len(solution_strs) != len(ground_truths):
        raise ValueError("Verifier responses and ground truths must have equal length")
    for name, values in (("data_sources", data_sources), ("extra_infos", extra_infos)):
        if values is not None and len(values) != len(solution_strs):
            raise ValueError(f"Verifier {name} must have one entry per response")
    responses = (
        _role_grounded_batch_responses(kwargs["data"], len(solution_strs))
        if "data" in kwargs else solution_strs
    )
    return [_score_one(response, gt) for response, gt in zip(responses, ground_truths)]
