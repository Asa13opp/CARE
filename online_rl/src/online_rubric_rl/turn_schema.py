from __future__ import annotations

import copy
import json
from typing import Any, Dict, Iterable, List


REQUIRED_FIELDS = (
    "turn_id",
    "dialogue_id",
    "turn_idx",
    "rubric_history",
    "query",
    "judge_history",
    "assistant_response",
)


def plain(value: Any) -> Any:
    """Convert numpy scalars/arrays and other simple containers to Python values."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def compact_json(value: Any) -> str:
    return json.dumps(plain(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_messages(messages: Iterable[Dict[str, Any]] | None) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for msg in plain(messages or []):
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip()
        content = str(msg.get("content", "")).strip()
        if role and content:
            out.append({"role": role, "content": content})
    return out


def split_history_and_query(messages: Iterable[Dict[str, Any]]) -> tuple[List[Dict[str, str]], str]:
    normalized = normalize_messages(messages)
    for idx in range(len(normalized) - 1, -1, -1):
        if normalized[idx]["role"] == "user":
            return normalized[:idx], normalized[idx]["content"]
    return normalized, ""


def build_turn_record(
    *,
    turn_id: str,
    dialogue_id: str,
    turn_idx: int,
    history_before_query: Iterable[Dict[str, Any]],
    query: str,
    assistant_response: str,
    messages_after_turn: Iterable[Dict[str, Any]] | None = None,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    rubric_history = normalize_messages(history_before_query)
    query = str(query or "").strip()
    judge_history = copy.deepcopy(rubric_history)
    if query:
        judge_history.append({"role": "user", "content": query})
    record: Dict[str, Any] = {
        "turn_id": str(turn_id),
        "dialogue_id": str(dialogue_id),
        "turn_idx": int(turn_idx),
        "rubric_history": rubric_history,
        "query": query,
        "judge_history": judge_history,
        "assistant_response": str(assistant_response or "").strip(),
        "messages_after_turn": normalize_messages(messages_after_turn),
    }
    if extra:
        record["extra"] = plain(extra)
    return record


def validate_turn_record(record: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    for field in REQUIRED_FIELDS:
        if field not in record:
            errors.append(f"{field} missing")
    if not isinstance(record.get("rubric_history"), list):
        errors.append("rubric_history must be a list")
    if not isinstance(record.get("judge_history"), list):
        errors.append("judge_history must be a list")
    if not str(record.get("query", "")).strip():
        errors.append("query empty")
    if not str(record.get("assistant_response", "")).strip():
        errors.append("assistant_response empty")
    if record.get("judge_history"):
        last = record["judge_history"][-1]
        if not isinstance(last, dict) or last.get("role") != "user":
            errors.append("judge_history must end with current user query")
    return errors

