from __future__ import annotations

import os
import re
from typing import Any

from empathy_rl.judge_reward import response_length_penalty, score_response


_FORMAT_PATTERN = re.compile(r"^\s*<think>\s*.+?\s*</think>\s*\S+", re.IGNORECASE | re.DOTALL)
_THINK_TAG_PATTERN = re.compile(r"</?think\b[^>]*>", re.IGNORECASE)
_FORBIDDEN_TOKENS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")


def _plain(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


def _format_score(text: str) -> float:
    stripped = (text or "").strip()
    if not stripped:
        return -1.0
    lowered = stripped.lower()
    if any(tok in lowered for tok in _FORBIDDEN_TOKENS):
        return -1.0
    mode = os.environ.get("FORMAT_REWARD_MODE", "plain").lower()
    if mode == "think":
        return 1.0 if _FORMAT_PATTERN.search(stripped) else -1.0
    if _THINK_TAG_PATTERN.search(stripped):
        return -1.0
    return 1.0


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any],
    **kwargs: Any,
) -> dict[str, float]:
    del data_source, kwargs
    ground_truth = _plain(ground_truth)
    extra_info = _plain(extra_info)
    if not isinstance(ground_truth, dict):
        raise ValueError("ground_truth must contain query and rubric")

    history = extra_info.get("history")
    if history is None:
        history = []
    query = ground_truth.get("query") or extra_info.get("query", "")
    rubric = ground_truth.get("rubric")
    try:
        judge = score_response(history, query, rubric, solution_str)
    except ValueError:
        judge = 0.0
    fmt = _format_score(solution_str)

    judge_weight = float(os.environ.get("JUDGE_REWARD_WEIGHT", "1.0"))
    format_weight = float(os.environ.get("FORMAT_REWARD_WEIGHT", "0.5"))
    length_penalty, body_tokens, length_limit = response_length_penalty(solution_str)
    score = judge_weight * judge + format_weight * fmt + length_penalty

    return {
        "score": float(score),
        "judge_reward": float(judge),
        "format_reward": float(fmt),
        "length_penalty": float(length_penalty),
        "response_body_tokens": float(body_tokens),
        "response_length_limit": float(length_limit),
    }
