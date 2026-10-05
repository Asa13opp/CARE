from __future__ import annotations

import json
import re
from typing import Any, Dict, Tuple

from empathy_rl.schema import normalize_rubric, parse_rubric, validate_rubric


def _validate_model_output(obj: Dict[str, Any]) -> tuple[Dict[str, Any], list[str]]:
    normalized = normalize_rubric(obj)
    errors = validate_rubric(normalized)
    return normalized, errors


def extract_json_object(text: str) -> str:
    """Extract the first JSON object from common model wrappers."""
    t = (text or "").strip()
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL | re.IGNORECASE).strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t).rsplit("```", 1)[0].strip()

    start = t.find("{")
    if start == -1:
        return t

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(t)):
        ch = t[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return t[start : idx + 1]
    return t[start:]


def _try_truncate_to_last_brace(text: str) -> str | None:
    """If text starts with '{' but is truncated (no matching '}'), find the last
    complete top-level closing brace and return the prefix up to and including it."""
    t = text.strip()
    if not t.startswith("{"):
        return None
    # Walk backwards to find the last '}' and try parsing up to it.
    pos = len(t)
    while pos > 0:
        pos = t.rfind("}", 0, pos)
        if pos == -1:
            return None
        candidate = t[: pos + 1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    return None


def parse_or_repair_rubric(text: str) -> Tuple[Dict[str, Any] | None, bool, list[str], str]:
    """Strict parse first, then wrapper cleanup, then truncation recovery."""
    stripped = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE).strip()
    obj, err = parse_rubric(stripped)
    if err is None and obj is not None:
        normalized, errors = _validate_model_output(obj)
        return (normalized if not errors else obj), stripped != (text or "").strip(), errors, json.dumps(normalized, ensure_ascii=False)

    cleaned = extract_json_object(stripped)
    obj, err = parse_rubric(cleaned)
    if err is None and obj is not None:
        normalized, errors = _validate_model_output(obj)
        return (normalized if not errors else obj), True, errors, json.dumps(normalized, ensure_ascii=False)

    # Last resort: the model output may have been truncated mid-JSON.
    # Try trimming to the last parseable closing brace.
    truncated = _try_truncate_to_last_brace(cleaned)
    if truncated is not None:
        obj, err = parse_rubric(truncated)
        if err is None and obj is not None:
            normalized, errors = _validate_model_output(obj)
            return (normalized if not errors else obj), True, errors, json.dumps(normalized, ensure_ascii=False)

    last_err = err or "unknown parse error"
    return None, False, [last_err], cleaned
