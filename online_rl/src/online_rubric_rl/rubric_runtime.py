from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Any, Dict, List

from empathy_rl.rubric_backend import RubricBackend, load_backend
from empathy_rl.rubric_parse import parse_or_repair_rubric
from empathy_rl.rubric_prompt import make_rubric_messages

from online_rubric_rl.turn_schema import compact_json


def _normalized_rubric_base_url() -> str | None:
    base_url = os.environ.get("RUBRIC_API_BASE_URL")
    if not base_url:
        return None

    if os.environ.get("RUBRIC_API_ALLOW_HTTP_FALLBACK", "1") != "1":
        return base_url

    lowered = base_url.lower()
    http_only_hosts = tuple(
        host.strip()
        for host in os.environ.get("RUBRIC_API_HTTP_ONLY_HOSTS", "10.,127.,localhost").split(",")
        if host.strip()
    )
    if lowered.startswith("https://") and lowered.removeprefix("https://").startswith(http_only_hosts):
        return "http://" + base_url[len("https://") :]
    return base_url


MOCK_RUBRIC: Dict[str, Any] = {
    "categories": [
        {
            "category": "Cognitive Empathy",
            "weight_rationale": "The current user turn needs accurate understanding of the situation.",
            "category_weight": 0.34,
            "criteria": [
                {
                    "title": "Reflects the user's situation",
                    "description": "[0] The response misses the user's situation. [1] The response barely notices the situation. [2] The response partially reflects the situation. [3] The response mostly reflects the situation. [4] The response accurately reflects the situation without unsupported details. [5] The response shows nuanced, turn-specific understanding of the situation. value: 0, 1, 2, 3, 4, 5",
                }
            ],
        },
        {
            "category": "Affective Empathy",
            "weight_rationale": "The current user turn needs emotional validation and warmth.",
            "category_weight": 0.33,
            "criteria": [
                {
                    "title": "Validates the emotion",
                    "description": "[0] The response does not acknowledge the feeling. [1] The response gives a perfunctory acknowledgment. [2] The response names a general feeling. [3] The response validates the user's feeling as understandable. [4] The response validates the feeling warmly and specifically. [5] The response deeply validates the exact emotional state with warmth and care. value: 0, 1, 2, 3, 4, 5",
                }
            ],
        },
        {
            "category": "Proactive Empathy",
            "weight_rationale": "The current user turn may benefit from a gentle next step.",
            "category_weight": 0.33,
            "criteria": [
                {
                    "title": "Offers a gentle next step",
                    "description": "[0] The response offers no useful next step. [1] The response offers a vague or mismatched next step. [2] The response offers a somewhat useful but generic next step. [3] The response offers a practical next step without pressure. [4] The response offers a tailored, gentle next step. [5] The response offers an excellent, turn-specific next step while preserving the user's autonomy. value: 0, 1, 2, 3, 4, 5",
                }
            ],
        },
    ]
}


class RubricRuntime:
    def __init__(self, backend: RubricBackend | None = None) -> None:
        self._backend = backend
        self._backend_lock = threading.Lock()
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _load_backend(self) -> RubricBackend:
        if self._backend is None:
            with self._backend_lock:
                if self._backend is None:
                    think = os.environ.get("RUBRIC_API_THINKING", "0").lower() in {"1", "true", "yes", "on"}
                    extra_body = {
                        "top_k": 20,
                        "chat_template_kwargs": {"enable_thinking": think},
                        "thinking": {"type": f"{'enabled' if think else 'disabled'}"},
                    }
                    extra_raw = os.environ.get("RUBRIC_API_EXTRA_BODY", "").strip()
                    if extra_raw:
                        extra_body.update(json.loads(extra_raw))
                    kwargs = {
                        "model": os.environ.get("RUBRIC_API_MODEL"),
                        "base_url": _normalized_rubric_base_url(),
                        "api_key": os.environ.get("RUBRIC_API_KEY", "EMPTY"),
                        "max_tokens": int(os.environ.get("RUBRIC_MAX_NEW_TOKENS", "768")),
                        "temperature": float(os.environ.get("RUBRIC_TEMPERATURE", "0")),
                        "max_workers": int(os.environ.get("RUBRIC_API_WORKERS", "8")),
                        "extra_body": extra_body,
                        "response_format": {"type": "json_object"},
                        "reasoning_effort": "high" if think else "none",
                    }
                    self._backend = load_backend("openai", **kwargs)
        return self._backend

    def generate_one(self, history: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        key = hashlib.sha256(compact_json([history, query]).encode()).hexdigest()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if os.environ.get("MOCK_RUBRIC", "0") == "1":
            result = {
                "rubric": MOCK_RUBRIC,
                "rubric_valid": True,
                "rubric_repaired": False,
                "rubric_raw": compact_json(MOCK_RUBRIC),
                "rubric_errors": [],
            }
            self._cache[key] = result
            return result
        raw = self._load_backend().generate([make_rubric_messages(history, query)])[0]
        rubric, repaired, errors, cleaned = parse_or_repair_rubric(raw)
        result = {
            "rubric": rubric,
            "rubric_valid": not errors,
            "rubric_repaired": repaired,
            "rubric_raw": raw,
            "rubric_cleaned": cleaned,
            "rubric_errors": errors,
        }
        self._cache[key] = result
        return result

    def generate_many(self, items: List[tuple[List[Dict[str, Any]], str]]) -> List[Dict[str, Any]]:
        if os.environ.get("MOCK_RUBRIC", "0") == "1" or len(items) <= 1:
            return [self.generate_one(history, query) for history, query in items]

        results: List[Dict[str, Any] | None] = [None] * len(items)
        missing: List[tuple[int, List[Dict[str, Any]], str, str]] = []
        for idx, (history, query) in enumerate(items):
            key = hashlib.sha256(compact_json([history, query]).encode()).hexdigest()
            cached = self._cache.get(key)
            if cached is None:
                missing.append((idx, history, query, key))
            else:
                results[idx] = cached

        if missing:
            messages_batch = [make_rubric_messages(history, query) for _, history, query, _ in missing]
            raw_batch = self._load_backend().generate(messages_batch)
            for (idx, _history, _query, key), raw in zip(missing, raw_batch):
                rubric, repaired, errors, cleaned = parse_or_repair_rubric(raw)
                result = {
                    "rubric": rubric,
                    "rubric_valid": not errors,
                    "rubric_repaired": repaired,
                    "rubric_raw": raw,
                    "rubric_cleaned": cleaned,
                    "rubric_errors": errors,
                }
                self._cache[key] = result
                results[idx] = result

        return [item for item in results if item is not None]
