"""Dependency-free identity and validation contracts for verifier data.

These helpers deliberately do not import torch, pandas, or the verl runtime.
Exact-match (EM) labels are an answer-correctness proxy. They do not certify
reasoning, retrieved evidence, hop counts, or a backward proof of a chain.
"""

from collections.abc import Mapping
import copy
import hashlib
import json
import math
from numbers import Integral, Real
import re
import string
import unicodedata


SCHEMA_VERSION = 1
LABEL_SOURCE = "answer_em_proxy"
LABEL_LIMITATION = (
    "Normalized answer exact match is an answer-correctness proxy, not a label "
    "for chain validity, evidence quality, or successful backward verification."
)


def require_text(value, field):
    """Reject nulls, NaN, arrays, and empty text instead of stringifying them."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value.strip()


def _canonical_text(value, field):
    return " ".join(unicodedata.normalize("NFC", require_text(value, field)).split())


def _digest(kind, payload):
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return kind + "_" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def reference_answers(value):
    """Validate gold answers, including array-valued parquet cells, without numpy."""
    if isinstance(value, str):
        value = [value]
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("reference_answers must contain at least one nonempty string")
    return sorted({_canonical_text(answer, "reference answer") for answer in value})


def extract_source_document(prompt):
    """Extract the document from this repository's challenger prompt.

    Raw source text is also accepted. Stripping the known template prevents
    different requested hop counts for one document from crossing splits.
    """
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if isinstance(prompt, (list, tuple)):
        users = [message.get("content") for message in prompt
                 if isinstance(message, Mapping) and message.get("role") == "user"]
        if not users:
            raise ValueError("source prompt must include a user message")
        prompt = users[-1]
    text = require_text(prompt, "source document")
    marker = "starting from the following source document:"
    if marker in text:
        text = text.split(marker, 1)[1]
    return require_text(text, "source document")


def make_source_id(source_content):
    return _digest("source", {"text": _canonical_text(source_content, "source content")})


def make_item_id(question, answers, source_content):
    """Stable content ID independent of batch, ordering, path, and split."""
    return _digest("item", {
        "question": _canonical_text(question, "question"),
        "reference_answers": reference_answers(answers),
        "source_id": make_source_id(source_content),
    })


def _nonnegative_integer(value, field):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return int(value)


def make_candidate_id(item_id, response, run_id, step, candidate_index):
    """Identify one sampled candidate, even when text repeats in another rollout.

    run_id is the export's rollout_uid; candidate_index is its rollout_index,
    assigned before repeat/reorder. Identity refers to the sample slot rather
    than its answer text; response must still be nonempty.
    """
    require_text(response, "responses")
    return _digest("candidate", [
        require_text(item_id, "item_id"),
        require_text(run_id, "rollout_uid"),
        _nonnegative_integer(step, "global_step"),
        _nonnegative_integer(candidate_index, "rollout_index"),
    ])


def normalize_answer(value):
    value = require_text(value, "answer").lower()
    value = "".join(char for char in value if char not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def answer_em_proxy(prediction, answers):
    """Return a strict bool; fail rather than create labels from empty answers."""
    pred = normalize_answer(prediction)
    gold = [normalize_answer(answer) for answer in reference_answers(answers)]
    if not pred or any(not answer for answer in gold):
        raise ValueError("answer EM cannot label an answer empty after normalization")
    return pred in gold


def extract_answer(response):
    """Read the final complete <answer> tag; never silently invent a negative."""
    text = require_text(response, "responses")
    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    if not matches:
        raise ValueError("responses is missing a complete <answer> tag")
    return require_text(matches[-1], "candidate answer")


def _mapping(value, field):
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def validate_generated_row(row):
    """Validate new generated schema, including IDs against their source content."""
    row = _mapping(row, "generated row")
    item_id = require_text(row.get("item_id"), "generated item_id")
    question = require_text(row.get("question"), f"{item_id}: question")
    answers = reference_answers(row.get("reference_answers"))
    metadata = _mapping(row.get("metadata"), f"{item_id}: metadata")
    source = require_text(metadata.get("source_document"), f"{item_id}: source_document")
    source_id = require_text(row.get("source_id"), f"{item_id}: source_id")
    if source_id != make_source_id(source):
        raise ValueError(f"{item_id}: source_id does not match source content")
    if item_id != make_item_id(question, answers, source):
        raise ValueError(f"{item_id}: item_id does not match question/answers/source content")
    # Do not allow labels that become meaningless under the EM normalization.
    if any(not normalize_answer(answer) for answer in answers):
        raise ValueError(f"{item_id}: reference answer is empty after normalization")
    return {"item_id": item_id, "question": question, "reference_answers": answers,
            "source_id": source_id, "original": row}


def join_candidates(generated_rows, candidate_rows, *, allow_subset=False):
    """Join every exported candidate by item_id, with no positional fallback.

    Candidate exports must carry the full provenance contract. By default every
    generated item needs a candidate. allow_subset explicitly permits unexported
    items (e.g. last-training-step exports), never unknown or duplicate IDs.
    """
    generated = {}
    for row in generated_rows:
        checked = validate_generated_row(row)
        item_id = checked["item_id"]
        if item_id in generated:
            raise ValueError(f"duplicate generated item_id: {item_id}")
        generated[item_id] = checked
    if not generated:
        raise ValueError("generated dataset is empty")

    joined, candidate_ids, provenance, covered = [], set(), set(), set()
    for row in candidate_rows:
        row = _mapping(row, "candidate row")
        item_id = require_text(row.get("item_id"), "candidate item_id")
        candidate_id = require_text(row.get("candidate_id"), "candidate_id")
        if candidate_id in candidate_ids:
            raise ValueError(f"duplicate candidate_id: {candidate_id}")
        if item_id not in generated:
            raise ValueError(f"unknown candidate item_id: {item_id}")
        response = require_text(row.get("responses"), f"{candidate_id}: responses")
        # Preserve the raw response text for provenance, including whitespace.
        response = row["responses"]
        expected = make_candidate_id(item_id, response, row.get("rollout_uid"),
                                     row.get("global_step"), row.get("rollout_index"))
        if candidate_id != expected:
            raise ValueError(f"{candidate_id}: candidate_id does not match export content/provenance")
        sample_key = (item_id, row["rollout_uid"], int(row["global_step"]), int(row["rollout_index"]))
        if sample_key in provenance:
            raise ValueError(f"duplicate candidate provenance: {sample_key}")
        source = generated[item_id]
        if require_text(row.get("source_id"), f"{candidate_id}: source_id") != source["source_id"]:
            raise ValueError(f"{candidate_id}: source_id does not match generated item")
        answer = extract_answer(response)
        if "candidate_answer" in row and row["candidate_answer"] != answer:
            raise ValueError(f"{candidate_id}: candidate_answer does not match responses")
        is_correct = answer_em_proxy(answer, source["reference_answers"])
        # Optional precomputed labels are assertions, not alternative supervision.
        if "is_correct" in row and (type(row["is_correct"]) is not bool or row["is_correct"] != is_correct):
            raise ValueError(f"{candidate_id}: is_correct must be a boolean matching answer EM")
        if "label_source" in row and row["label_source"] != LABEL_SOURCE:
            raise ValueError(f"{candidate_id}: label_source must be {LABEL_SOURCE}")
        if "training_reward" in row:
            reward = row["training_reward"]
            if isinstance(reward, bool) or not isinstance(reward, Real) or not math.isfinite(reward):
                raise ValueError(f"{candidate_id}: training_reward must be a finite number")
        joined.append({**source, "candidate_id": candidate_id, "candidate_answer": answer,
                       "responses": response, "is_correct": is_correct, "export": row})
        candidate_ids.add(candidate_id)
        provenance.add(sample_key)
        covered.add(item_id)
    if not joined:
        raise ValueError("candidate export is empty")
    missing = set(generated) - covered
    if missing and not allow_subset:
        raise ValueError(f"missing candidate coverage for {len(missing)} generated item(s); "
                         "use --allow_subset only for intentionally partial training exports")
    # Input order is never used for identity, labels, or output ordering.
    return sorted(joined, key=lambda row: (row["item_id"], row["candidate_id"]))


def split_by_source(rows, *, eval_fraction=0.1, seed=0):
    """Deterministic whole-source split, keeping every candidate of an item together.

    A positive eval_fraction requires at least two distinct sources. Set zero
    explicitly for train-only builds; evaluation must never reuse train rows.
    """
    if isinstance(eval_fraction, bool) or not isinstance(eval_fraction, Real) or not 0 <= eval_fraction < 1:
        raise ValueError("eval_fraction must be in [0, 1)")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("split seed must be an integer")
    rows = list(rows)
    if not rows:
        raise ValueError("cannot split an empty dataset")
    sources = {require_text(row.get("source_id"), "source_id") for row in rows}
    ordered = sorted(sources, key=lambda source: (_digest("split", [int(seed), source]), source))
    if eval_fraction and len(ordered) < 2:
        raise ValueError("source-disjoint train/eval split requires at least two distinct sources")
    count = min(len(ordered) - 1, max(1, round(len(ordered) * eval_fraction))) if eval_fraction else 0
    eval_sources = set(ordered[:count])
    train, evaluation = [], []
    for row in sorted(rows, key=lambda row: (row["item_id"], row["candidate_id"])):
        target, split = (evaluation, "val") if row["source_id"] in eval_sources else (train, "train")
        row = copy.deepcopy(row)
        row.setdefault("extra_info", {})["split"] = split
        target.append(row)
    return train, evaluation
