"""Pure-Python contracts for all-candidate, reorder-safe solver exports.

These records are on-policy training samples, not fixed-checkpoint evaluation.
"""
from collections.abc import Mapping
import hashlib
import json
import math


def final_assistant_response(message_record):
    """Extract only the final assistant turn; retrieved answer tags are untrusted.

    Missing/final-tool messages return empty text, which the dataset builder
    rejects as malformed. We never label a tool document as a solver prediction.
    """
    messages = message_record.get("messages") if isinstance(message_record, Mapping) else None
    if not isinstance(messages, (list, tuple)) or not messages:
        raise ValueError("Solver export requires structured rollout messages")
    message = messages[-1]
    role = message.get("role") if isinstance(message, Mapping) else getattr(message, "role", None)
    content = message.get("content") if isinstance(message, Mapping) else getattr(message, "content", None)
    if role != "assistant" or not isinstance(content, str):
        return ""
    return content


def validate_export_inputs(item_ids, source_ids):
    if not item_ids or len(item_ids) != len(source_ids):
        raise ValueError("Solver export needs nonempty, aligned item_id/source_id fields; regenerate legacy data")
    for item_id, source_id in zip(item_ids, source_ids):
        if not isinstance(item_id, str) or not item_id.strip():
            raise ValueError("Solver export requires a stable item_id on every row")
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("Solver export requires a stable source_id on every row")


def build_export_records(item_ids, source_ids, uids, rollout_indices, responses, scores, step, expected_n):
    """Validate groups by uid, never by adjacency; preserve every rollout."""
    validate_export_inputs(item_ids, source_ids)
    count = len(item_ids)
    if any(len(values) != count for values in (uids, rollout_indices, responses, scores)):
        raise ValueError("Misaligned solver export columns")
    if isinstance(expected_n, bool) or not isinstance(expected_n, int) or expected_n < 1:
        raise ValueError("expected_n must be a positive integer")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a nonnegative integer")
    groups = {}
    records = []
    for item_id, source_id, uid, index, response, score in zip(
        item_ids, source_ids, uids, rollout_indices, responses, scores
    ):
        if not isinstance(uid, str) or not uid:
            raise ValueError("Missing rollout uid")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < expected_n:
            raise ValueError("Invalid rollout_index")
        group = groups.setdefault(uid, {"item_id": item_id, "source_id": source_id, "indices": set()})
        if group["item_id"] != item_id or group["source_id"] != source_id:
            raise ValueError("Rollout uid maps to multiple items/sources")
        if index in group["indices"]:
            raise ValueError("Duplicate rollout index in uid group")
        group["indices"].add(index)
        if not isinstance(response, str) or not math.isfinite(float(score)):
            raise ValueError("Invalid response or nonfinite training reward")
        # Identity refers to the sample slot, not its answer text. This remains stable
        # when balanced batches reorder candidates and distinguishes repeated items.
        identity = json.dumps([item_id, uid, step, index], separators=(",", ":"), ensure_ascii=False)
        records.append({
            "item_id": item_id, "source_id": source_id,
            "candidate_id": "candidate_" + hashlib.sha256(identity.encode()).hexdigest(),
            "rollout_uid": uid, "rollout_index": index, "global_step": step,
            "responses": response, "training_reward": float(score),
            "export_kind": "on_policy_training_rollout",
            "response_text_scope": "final_assistant_message",
        })
    if any(len(group["indices"]) != expected_n for group in groups.values()):
        raise ValueError("Missing candidates in rollout uid group")
    return records
