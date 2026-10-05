from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from typing import Any, Dict, List, Tuple

from empathy_rl.schema import normalize_rubric, parse_rubric, validate_rubric

ALLOWED_SCORES = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0}
THINK_OPEN_PATTERN = re.compile(r"<think\b[^>]*>", re.IGNORECASE)
THINK_CLOSE_PATTERN = re.compile(r"</think\s*>", re.IGNORECASE)
TOKEN_ESTIMATE_PATTERN = re.compile(
    r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\sA-Za-z0-9_\u4e00-\u9fff]",
    re.UNICODE,
)

JUDGE_SYSTEM_PROMPT = (
    "You are a strict judge for an empathy evaluation rubric. You are given dialogue "
    "history, the current user query, a candidate rubric, and a candidate assistant "
    "response. Score how well the candidate response satisfies the candidate rubric.\n\n"
    "Each criterion's description contains explicit scoring anchors in the form "
    "[0] ... [1] ... [2] ... [3] ... [4] ... [5] ... value: 0, 1, 2, 3, 4, 5. "
    "Choose the score whose anchor best matches the response; do not interpolate.\n\n"
    "Judge only whether each criterion's specific behavior is present and genuine in the "
    "response. Do not reward verbosity by itself. Return only valid JSON with exactly one "
    "key: criterion_scores. criterion_scores is a list with one object per rubric "
    "criterion, each object containing category, criterion_title, and score. Use each "
    "criterion title exactly as given. Scores must be one of 0, 1, 2, 3, 4, or 5."
)

JUDGE_OUTPUT_SHAPE = {
    "criterion_scores": [
        {"category": "Cognitive Empathy", "criterion_title": "<copy rubric title>", "score": 5}
    ]
}

_client = None
_client_lock = threading.Lock()
_cache: Dict[str, Dict[str, Any]] = {}
_cache_lock = threading.Lock()
_inflight = None
_inflight_lock = threading.Lock()


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _render_history(history: List[Dict[str, Any]]) -> str:
    return "\n".join(f"{t.get('role', 'unknown')}: {t.get('content', '')}" for t in history)


def _extract_json(text: str) -> Any:
    if text is None:
        raise ValueError("empty judge response")
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t).rsplit("```", 1)[0].strip()
    try:
        return json.loads(t)
    except Exception:
        match = re.search(r"\{.*\}", t, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def extract_response_body(response_text: str, require_think_end: bool = False) -> str:
    """Return only the visible answer after the final hidden thinking block."""
    text = response_text or ""
    closes = list(THINK_CLOSE_PATTERN.finditer(text))
    if closes:
        return text[closes[-1].end() :].strip()

    # An opening tag without a close is a truncated hidden-thought block. Qwen3 may
    # put the opening tag in the prompt, so thinking-mode callers must also require
    # a close when the generated text itself has no opening tag.
    if THINK_OPEN_PATTERN.search(text) or require_think_end:
        return ""
    return text.strip()


def estimate_token_count(text: str) -> int:
    """Lightweight token estimate for reward code paths that do not have a tokenizer."""
    return len(TOKEN_ESTIMATE_PATTERN.findall((text or "").strip()))


def response_body_token_count(response_text: str, tokenizer: Any | None = None) -> int:
    body = extract_response_body(response_text)
    if not body:
        return 0
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(body, add_special_tokens=False))
        except TypeError:
            try:
                return len(tokenizer.encode(body))
            except Exception:
                pass
        except Exception:
            pass
    return estimate_token_count(body)


def response_length_penalty(response_text: str, body_token_count: int | None = None) -> tuple[float, int, int]:
    """Linear penalty for answer-body tokens over RESPONSE_LENGTH_PENALTY_MAX_TOKENS."""
    max_tokens = int(os.environ.get("RESPONSE_LENGTH_PENALTY_MAX_TOKENS", "256"))
    if max_tokens <= 0:
        count = body_token_count if body_token_count is not None else response_body_token_count(response_text)
        return 0.0, int(count), max_tokens

    count = int(body_token_count if body_token_count is not None else response_body_token_count(response_text))
    over = max(0, count - max_tokens)
    if over == 0:
        return 0.0, count, max_tokens

    weight = float(os.environ.get("RESPONSE_LENGTH_PENALTY_WEIGHT", "1.0"))
    penalty = -weight * (over / max_tokens)
    max_penalty = float(os.environ.get("RESPONSE_LENGTH_PENALTY_MAX_PENALTY", "2.0"))
    if max_penalty > 0:
        penalty = max(penalty, -max_penalty)
    return float(penalty), count, max_tokens


def _get_client():
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                import httpx
                from openai import OpenAI

                _client = OpenAI(
                    base_url=os.environ.get("JUDGE_BASE_URL", "https://tokenhub.sensetime.com/v1"),
                    api_key=os.environ["JUDGE_API_KEY"],
                    timeout=httpx.Timeout(float(os.environ.get("JUDGE_TIMEOUT", "360")), connect=10.0),
                    max_retries=0,
                )
    return _client


def _get_inflight():
    global _inflight
    if _inflight is None:
        with _inflight_lock:
            if _inflight is None:
                _inflight = threading.BoundedSemaphore(max(1, int(os.environ.get("JUDGE_MAX_INFLIGHT", "4"))))
    return _inflight


def _criterion_titles(rubric: Dict[str, Any]) -> set[Tuple[Any, Any]]:
    out = set()
    for category in rubric.get("categories", []):
        if not isinstance(category, dict):
            continue
        cat = category.get("category")
        for criterion in category.get("criteria", []):
            if isinstance(criterion, dict):
                out.add((cat, criterion.get("title")))
    return out


def coerce_criterion_scores(rubric: Dict[str, Any], raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, dict):
        raw = raw.get("criterion_scores", raw)
    if isinstance(raw, dict):
        raw = list(raw.values())
    if not isinstance(raw, list) or not raw:
        raise ValueError("criterion_scores must be a non-empty list")
    criterion_titles = _criterion_titles(rubric)
    cleaned = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        cat = item.get("category")
        title = item.get("criterion_title")
        if (cat, title) not in criterion_titles:
            continue
        try:
            score = float(item["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if score not in ALLOWED_SCORES:
            score = min(ALLOWED_SCORES, key=lambda val: abs(val - score))
        cleaned.append({"category": cat, "criterion_title": title, "score": score})
    if not cleaned:
        raise ValueError("no valid criterion scores")
    return cleaned


def gold_normalized_score(
    rubric: Dict[str, Any],
    criterion_scores: List[Dict[str, Any]],
    included_criteria: set[Tuple[Any, Any]] | None = None,
) -> float:
    criterion_titles = _criterion_titles(rubric)
    weights = {}
    for category in rubric.get("categories", []):
        if not isinstance(category, dict):
            continue
        try:
            weights[category.get("category")] = float(category.get("category_weight", 0.0))
        except (TypeError, ValueError):
            weights[category.get("category")] = 0.0
    agg = {cat: {"sum": 0.0, "n": 0} for cat in weights}
    for item in criterion_scores:
        cat = item.get("category")
        title = item.get("criterion_title")
        if cat not in agg or (cat, title) not in criterion_titles:
            continue
        if included_criteria is not None and (cat, title) not in included_criteria:
            continue
        agg[cat]["sum"] += float(item["score"])
        agg[cat]["n"] += 1
    if included_criteria is None:
        active_weights = weights
    else:
        active_weights = {cat: weight for cat, weight in weights.items() if agg[cat]["n"] > 0}
    if not active_weights:
        return 0.0
    total_weight = sum(active_weights.values()) or 1.0
    score = 0.0
    for cat, weight in active_weights.items():
        n = agg[cat]["n"]
        cat_mean = agg[cat]["sum"] / n if n else 0.0
        score += (weight / total_weight) * (cat_mean / 5.0)
    return float(score)


def build_judge_messages(
    history: List[Dict[str, Any]], query: str, rubric: Dict[str, Any], response_text: str
) -> List[Dict[str, str]]:
    # history already ends with the current user query; pass prior turns as context only
    prior_history = history[:-1] if history and history[-1].get("content", "").strip() == query.strip() else history
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _compact(
                {
                    "dialogue_history": _render_history(prior_history),
                    "current_user_query": query,
                    "candidate_rubric": rubric,
                    "candidate_response": response_text,
                    "required_output_shape": JUDGE_OUTPUT_SHAPE,
                }
            ),
        },
    ]


def call_deepseek_judge(messages: List[Dict[str, str]]) -> Any:
    retries = int(os.environ.get("JUDGE_MAX_RETRIES", "3"))
    last = None
    for attempt in range(retries):
        try:
            think = os.environ.get("JUDGE_THINKING", "0").lower() in {"1", "true", "yes", "on"}
            extra_body = {
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": think},
                "thinking": {"type": f"{'enabled' if think else 'disabled'}"},
            }
            extra_raw = os.environ.get("JUDGE_EXTRA_BODY", "").strip()
            if extra_raw:
                extra_body.update(json.loads(extra_raw))
            kwargs = {
                "model": os.environ.get("JUDGE_MODEL", "deepseek-v4-pro"),
                "messages": messages,
                "max_tokens": int(os.environ.get("JUDGE_MAX_TOKENS", "2048")),
                "temperature": float(os.environ.get("JUDGE_TEMPERATURE", "0")),
                "extra_body": extra_body,
                "response_format": {"type": "json_object"},
            }
            if think:
                kwargs["reasoning_effort"] = os.environ.get("JUDGE_REASONING_EFFORT", "high")
            with _get_inflight():
                _pace_judge_request()
                resp = _get_client().chat.completions.create(**kwargs)
            return _extract_json(resp.choices[0].message.content)
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(min(8.0, 1.5 * (attempt + 1)))
    raise RuntimeError(f"DeepSeek judge failed after {retries} retries: {last}")


def _pace_judge_request() -> None:
    """Best-effort cross-process pacing for shared judge API limits."""
    interval = float(os.environ.get("JUDGE_MIN_INTERVAL", "0"))
    if interval <= 0:
        return
    path = os.environ.get("JUDGE_PACE_FILE", "/tmp/empathy_deepseek_judge_pace.txt")
    try:
        import fcntl

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a+", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            raw = f.read().strip()
            last = float(raw) if raw else 0.0
            now = time.time()
            wait = interval - (now - last)
            if wait > 0:
                time.sleep(wait)
            f.seek(0)
            f.truncate()
            f.write(str(time.time()))
            f.flush()
            fcntl.flock(f, fcntl.LOCK_UN)
    except Exception:
        time.sleep(interval)


def score_response_details(history: List[Dict[str, Any]], query: str, rubric: Any, response_text: str) -> Dict[str, Any]:
    obj, err = parse_rubric(rubric)
    if err is not None or obj is None:
        raise ValueError(err or "invalid rubric")
    errors = validate_rubric(obj)
    if errors:
        raise ValueError("; ".join(errors[:3]))
    obj = normalize_rubric(obj)
    if os.environ.get("MOCK_JUDGE", "0") == "1":
        criterion_scores = _mock_criterion_scores(obj, response_text)
        return {
            "score": gold_normalized_score(obj, criterion_scores),
            "criterion_scores": criterion_scores,
        }
    key = hashlib.sha256(_compact([history, query, obj, response_text]).encode()).hexdigest()
    with _cache_lock:
        if key in _cache:
            return dict(_cache[key])
    raw = call_deepseek_judge(build_judge_messages(history, query, obj, response_text))
    criterion_scores = coerce_criterion_scores(obj, raw)
    result = {
        "score": gold_normalized_score(obj, criterion_scores),
        "criterion_scores": criterion_scores,
    }
    with _cache_lock:
        _cache[key] = result
    return dict(result)


def score_response(history: List[Dict[str, Any]], query: str, rubric: Any, response_text: str) -> float:
    return float(score_response_details(history, query, rubric, response_text)["score"])


def extract_completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    if isinstance(completion, dict):
        return str(completion.get("content", completion.get("text", "")))
    return str(completion)


def _score_one(i, completion, history, query, rubric, sample_id):
    try:
        response = extract_completion_text(completion)
        return score_response(history[i], query[i], rubric[i], response)
    except Exception as exc:  # noqa: BLE001
        if os.environ.get("REWARD_DEBUG_PATH"):
            _append_debug(
                os.environ["REWARD_DEBUG_PATH"],
                {
                    "sample_id": sample_id[i] if sample_id else i,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
        if os.environ.get("JUDGE_FAIL_FAST", "1") != "0":
            raise
        return 0.0


def judge_reward(prompts, completions, history, query, rubric, sample_id=None, **kwargs):
    n = len(completions)
    if n == 0:
        return []
    # Judging is network-bound; score completions concurrently. JUDGE_MAX_INFLIGHT is the
    # single concurrency knob: it sizes this thread pool and (in call_deepseek_judge) the
    # in-flight semaphore, so per-rank concurrency == JUDGE_MAX_INFLIGHT.
    max_workers = min(max(1, int(os.environ.get("JUDGE_MAX_INFLIGHT", "4"))), n)
    if max_workers == 1:
        return [_score_one(i, completions[i], history, query, rubric, sample_id) for i in range(n)]

    from concurrent.futures import ThreadPoolExecutor

    rewards = [0.0] * n
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_score_one, i, completions[i], history, query, rubric, sample_id): i
            for i in range(n)
        }
        for fut in futures:
            rewards[futures[fut]] = fut.result()
    return rewards


def format_reward(prompts, completions, **kwargs):
    """Reward completions that include a complete THINK block before the answer."""
    rewards = []
    forbidden = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")
    pattern = re.compile(r"^\s*<think>\s*.+?\s*</think>\s*\S+", re.IGNORECASE | re.DOTALL)
    for completion in completions:
        text = extract_completion_text(completion).strip()
        if not text:
            rewards.append(-1.0)
            continue
        lowered = text.lower()
        rewards.append(1.0 if pattern.search(text) and not any(tok in lowered for tok in forbidden) else -1.0)
    return rewards


def length_penalty_reward(prompts, completions, **kwargs):
    """Return 0 below the response-body token target and a capped negative reward above it."""
    target = max(1, int(os.environ.get("LENGTH_TARGET_TOKENS", "1000")))
    max_penalty = max(0.0, float(os.environ.get("LENGTH_MAX_PENALTY", "2.0")))
    completion_ids = kwargs.get("completion_ids")
    max_completion_length = int(os.environ.get("MAX_COMPLETION_LENGTH", "0") or "0")
    fail_clipped_ratio = float(os.environ.get("FAIL_ON_CLIPPED_RATIO", "0.9"))
    rewards = []
    clipped = 0
    for i, completion in enumerate(completions):
        text = extract_completion_text(completion)
        if completion_ids is not None:
            n_tokens = len(completion_ids[i])
            if max_completion_length > 0 and n_tokens >= max_completion_length:
                clipped += 1
            n_tokens = response_body_token_count(text)
        else:
            n_tokens = response_body_token_count(text)
        over_ratio = max(0.0, (n_tokens - target) / target)
        rewards.append(-min(max_penalty, over_ratio))
    if completion_ids is not None and completions:
        clipped_ratio = clipped / len(completions)
        if fail_clipped_ratio > 0 and clipped_ratio >= fail_clipped_ratio:
            raise RuntimeError(
                f"Clipped completion ratio {clipped_ratio:.3f} >= {fail_clipped_ratio:.3f}; "
                "stopping before GRPO updates on unstable max-length generations."
            )
    return rewards


def _append_debug(path: str, record: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _mock_score(rubric: Dict[str, Any], response_text: str) -> float:
    """Offline smoke-only score. Real runs leave MOCK_JUDGE unset."""
    return gold_normalized_score(rubric, _mock_criterion_scores(rubric, response_text))


def _mock_criterion_scores(rubric: Dict[str, Any], response_text: str) -> List[Dict[str, Any]]:
    text = (response_text or "").strip()
    if not text:
        length_score = 0.0
    else:
        length_score = 5.0 if len(text.split()) >= 8 else 3.0
    raw_scores = []
    for category in rubric.get("categories", []):
        for criterion in category.get("criteria", []):
            raw_scores.append(
                {
                    "category": category.get("category"),
                    "criterion_title": criterion.get("title"),
                    "score": length_score,
                }
            )
    return raw_scores
