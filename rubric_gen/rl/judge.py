"""Judge client and execution reward (R_exec).

The judge (external OpenAI-compatible API, configured via env) scores a *candidate
response* against a candidate rubric, criterion by criterion. Every criterion is scored
on the rubric's own integer 0--5 anchors in its `description`; v5 rubrics carry no
`polarity` field, so there is no positive/negative split. `gold_normalized_score` first
averages the raw criterion scores within each category, then combines category means by
the generated `category_weight`. R_exec rewards a rubric that scores the `chosen`
response above the `rejected` one.

Secrets/config come from env ONLY (never hard-code):
  JUDGE_BASE_URL, JUDGE_API_KEY, JUDGE_MODEL,
  JUDGE_EFFORT (default "max"), JUDGE_THINKING ("1"/"0", default "1"),
  JUDGE_CONCURRENCY (default 16), JUDGE_MAX_RETRIES (default 3),
  JUDGE_CACHE (path to a json cache; optional).
"""

import hashlib
import json
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

from rl.tactic_map import CATEGORY_NAMES

# v5 rubrics have no polarity; every criterion uses its generated 0--5 anchors.
ALLOWED_SCORES = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0}

JUDGE_SYSTEM_PROMPT = (
    "You are a strict judge for an empathy evaluation rubric. You are given the dialogue "
    "history, a candidate_rubric, and a candidate_response (a supporter reply to score). "
    "Score how well the candidate_response satisfies the candidate_rubric.\n\n"
    "Return ONLY valid JSON with exactly one key: criterion_scores.\n"
    "criterion_scores is a list with one item per rubric criterion, each item an object with "
    "keys category, criterion_title, and score (NO rationale). Use each criterion's own category "
    "and title exactly as given in candidate_rubric. For EVERY criterion, read that "
    "criterion's description and apply its generated [0] through [5] anchors strictly. "
    "Return one integer score from 0 to 5 for each criterion; do not invent a different "
    "scale and do not output a total."
)

JUDGE_OUTPUT_SHAPE = {
    "criterion_scores": [
        {"category": "Cognitive Empathy", "criterion_title": "<copy a title from candidate_rubric>", "score": 5}
    ]
}

_client = None
_client_lock = threading.Lock()
_cache: Dict[str, float] = {}
_cache_lock = threading.Lock()
_cache_loaded = False


# ---------- pure scoring (offline-testable) ----------

def criterion_title_map(rubric: Dict[str, Any]) -> Set[Tuple[Any, Any]]:
    """The set of (category, title) pairs that exist in the rubric. Used to drop judge
    output for criteria the judge invented or mis-named."""
    pairs = set()
    for category in rubric.get("categories", []):
        if not isinstance(category, dict):
            continue
        cat_name = category.get("category")
        for criterion in category.get("criteria", []):
            if isinstance(criterion, dict):
                pairs.add((cat_name, criterion.get("title")))
    return pairs


def gold_normalized_score(rubric: Dict[str, Any], criterion_scores: List[Dict[str, Any]]) -> float:
    """Category-weighted mean of the RAW criterion scores.

    Criteria are NOT normalized to [0,1] (that compresses the scale and shrinks the
    chosen-vs-rejected gap that drives R_exec); the raw scores are kept in {0,...,5}.
    Each category is the mean of its criteria, and the categories are combined by
    category_weight. The result is roughly in [0, 5]; it is only ever used as a difference
    in `reward_exec`, so the absolute scale is unconstrained."""
    valid = criterion_title_map(rubric)
    weights = {}
    for category in rubric.get("categories", []):
        if not isinstance(category, dict):
            continue
        weight = float(category.get("category_weight", 0.0))
        if not math.isfinite(weight):
            raise ValueError(f"non-finite category weight for {category.get('category')!r}")
        weights[category.get("category")] = weight
    agg = {cat: {"sum": 0.0, "n": 0} for cat in weights}
    seen = set()
    for item in criterion_scores:
        cat = item.get("category")
        if cat not in agg:
            continue
        pair = (cat, item.get("criterion_title"))
        if pair not in valid or pair in seen:
            continue  # judge named a criterion not in the rubric; skip it
        seen.add(pair)
        item_score = float(item["score"])
        if not math.isfinite(item_score):
            raise ValueError(f"non-finite judge score for criterion {pair!r}")
        agg[cat]["sum"] += item_score   # raw, no per-criterion normalization
        agg[cat]["n"] += 1
    total_weight = sum(weights.values())
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError(f"category weight total must be finite and positive, got {total_weight!r}")
    score = 0.0
    for cat, weight in weights.items():
        d = agg[cat]
        cat_mean = (d["sum"] / d["n"]) if d["n"] > 0 else 0.0
        score += (weight / total_weight) * cat_mean
    if not math.isfinite(score):
        raise ValueError(f"weighted judge score must be finite, got {score!r}")
    return score


def _coerce_criterion_scores(rubric: Dict[str, Any], raw: Any) -> List[Dict[str, Any]]:
    """Validate/clean judge output against the rubric; raise on irrecoverable mismatch."""
    if isinstance(raw, dict):
        raw = raw.get("criterion_scores", raw)
    if isinstance(raw, dict):
        raw = list(raw.values())
    if not isinstance(raw, list) or not raw:
        raise ValueError("judge criterion_scores must be a non-empty list")
    valid = criterion_title_map(rubric)
    out = []
    seen = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        cat, title = item.get("category"), item.get("criterion_title")
        pair = (cat, title)
        if pair not in valid:
            continue  # judge named a criterion not in the rubric; skip it
        if pair in seen:
            raise ValueError(f"duplicate judge score for criterion {pair!r}")
        try:
            score = float(item["score"])
        except (TypeError, ValueError, KeyError):
            continue
        if not math.isfinite(score):
            raise ValueError(f"non-finite judge score for criterion {pair!r}")
        # Clamp malformed numeric output to the nearest valid integer score (0--5).
        if score not in ALLOWED_SCORES:
            score = min(ALLOWED_SCORES, key=lambda a: abs(a - score))
        out.append({"category": cat, "criterion_title": title, "score": score})
        seen.add(pair)
    if seen != valid:
        missing = valid - seen
        raise ValueError(f"judge must score every rubric criterion exactly once; missing={missing!r}")
    return out


# ---------- API plumbing ----------

def _get_client():
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                from openai import OpenAI
                _client = OpenAI(
                    base_url=os.environ["JUDGE_BASE_URL"],
                    api_key=os.environ["JUDGE_API_KEY"],
                    timeout=float(os.environ.get("JUDGE_TIMEOUT", "30")),
                )
    return _client


def _extract_json(text: str) -> Any:
    if text is None:
        raise ValueError("empty judge content")
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t).rsplit("```", 1)[0]
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _judge_call_once(messages: List[Dict[str, str]]) -> Any:
    # RL default: thinking OFF for throughput (one call ~seconds vs minutes with
    # reasoning). reasoning_effort is only sent when thinking is explicitly enabled.
    thinking = os.environ.get("JUDGE_THINKING", "0") == "1"
    effort = os.environ.get("JUDGE_EFFORT", "")
    client = _get_client()
    kwargs = dict(
        model=os.environ["JUDGE_MODEL"],
        messages=messages,
        max_tokens=2048,
        temperature=float(os.environ.get("JUDGE_TEMPERATURE", "0")),
        extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
    )
    if thinking and effort:
        kwargs["reasoning_effort"] = effort
    resp = client.chat.completions.create(**kwargs)
    return _extract_json(resp.choices[0].message.content)


def _judge_call(messages: List[Dict[str, str]]) -> Any:
    """Call and parse the judge, retrying transient API/JSON failures."""
    retries = max(1, int(os.environ.get("JUDGE_MAX_RETRIES", "3")))
    last = None
    for attempt in range(retries):
        try:
            return _judge_call_once(messages)
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt + 1 < retries:
                time.sleep(min(8.0, 1.5 * (attempt + 1)))
    raise RuntimeError(
        f"judge call failed after {retries} attempts: {type(last).__name__}: {last}"
    )


def _render_history(prompt_turns: List[Dict[str, Any]]) -> str:
    return "\n".join(f"{t.get('role')}: {t.get('content', '')}" for t in prompt_turns)


def _cache_key(history: str, rubric: Dict[str, Any], response_text: str) -> str:
    # MUST include the dialogue history: the same rubric + response can be scored under
    # different histories, and the judge prompt conditions on history, so omitting it would
    # let a cached score from one sample's context leak into another.
    h = hashlib.sha256()
    h.update((history or "").encode())
    h.update(b"\x00")
    h.update(_compact(rubric).encode())
    h.update(b"\x00")
    h.update((response_text or "").encode())
    return h.hexdigest()


def score_response(prompt_turns: List[Dict[str, Any]], rubric: Dict[str, Any], response_text: str) -> float:
    """Judge `response_text` against `rubric` under this history -> raw score. Cached."""
    history = _render_history(prompt_turns)
    key = _cache_key(history, rubric, response_text)
    with _cache_lock:
        if key in _cache:
            return _cache[key]
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _compact(
                {
                    "dialogue_history": history,
                    "candidate_rubric": rubric,
                    "candidate_response": response_text,
                    "required_output_shape": JUDGE_OUTPUT_SHAPE,
                }
            ),
        },
    ]
    # Retry the complete judge pipeline, not just the HTTP/JSON layer. Models can
    # occasionally return valid JSON that omits or misnames one criterion; treating
    # that single malformed response as a fatal run-wide outage is too brittle.
    retries = max(1, int(os.environ.get("JUDGE_MAX_RETRIES", "3")))
    last = None
    for attempt in range(retries):
        try:
            raw = _judge_call_once(messages)
            scores = _coerce_criterion_scores(rubric, raw)
            val = gold_normalized_score(rubric, scores)
            break
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt + 1 < retries:
                time.sleep(min(8.0, 1.5 * (attempt + 1)))
    else:
        raise RuntimeError(
            "judge scoring failed after "
            f"{retries} attempts: {type(last).__name__}: {last}"
        ) from last
    with _cache_lock:
        _cache[key] = val
    return val


def _score_many_with_errors(
    requests: List[Tuple[List[Dict[str, Any]], Dict[str, Any], str]],
    show_progress: bool = False,
) -> Tuple[List[Optional[float]], List[Optional[str]]]:
    """Concurrent scoring with an error string retained for every failed request."""
    workers = int(os.environ.get("JUDGE_CONCURRENCY", "16"))

    def _one(req):
        try:
            return score_response(*req), None
        except Exception as exc:  # noqa: BLE001
            return None, f"{type(exc).__name__}: {exc}"

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(requests) or 1))) as ex:
        futures = {ex.submit(_one, req): i for i, req in enumerate(requests)}
        results: List[Optional[float]] = [None] * len(requests)
        errors: List[Optional[str]] = [None] * len(requests)
        completed = as_completed(futures)
        if show_progress:
            from tqdm import tqdm

            completed = tqdm(
                completed,
                total=len(futures),
                desc="Scoring with judge",
                unit="request",
                dynamic_ncols=True,
            )
        for f in completed:
            idx = futures[f]
            results[idx], errors[idx] = f.result()
        return results, errors


def score_many(
    requests: List[Tuple[List[Dict[str, Any]], Dict[str, Any], str]],
    show_progress: bool = False,
) -> List[Optional[float]]:
    """Concurrent scoring. Each request = (prompt_turns, rubric, response_text).
    Returns scores (None on failure for that request).

    ``show_progress`` is opt-in so training callers do not receive tqdm output.
    """
    results, _ = _score_many_with_errors(requests, show_progress=show_progress)
    return results


def reward_exec(
    prompt_turns: List[Dict[str, Any]],
    rubric: Dict[str, Any],
    chosen_text: str,
    rejected_text: str,
) -> float:
    """Return raw s_chosen - s_rejected (range [-5, 5]). On judge failure -> 0.0."""
    score, _ = reward_exec_with_status(prompt_turns, rubric, chosen_text, rejected_text)
    return score


def reward_exec_with_status(
    prompt_turns: List[Dict[str, Any]],
    rubric: Dict[str, Any],
    chosen_text: str,
    rejected_text: str,
) -> Tuple[float, bool]:
    """Return ``(s_chosen - s_rejected, judge_ok)``.

    ``judge_ok`` distinguishes an actual tie (0.0, True) from one or both judge
    requests failing (0.0, False), so the training reward can fail fast only for
    infrastructure/model-judge failures.
    """
    scores, errors = _score_many_with_errors(
        [(prompt_turns, rubric, chosen_text), (prompt_turns, rubric, rejected_text)]
    )
    s_chosen, s_rejected = scores
    if s_chosen is None or s_rejected is None:
        failed = []
        if s_chosen is None:
            failed.append(f"chosen={errors[0] or 'unknown error'}")
        if s_rejected is None:
            failed.append(f"rejected={errors[1] or 'unknown error'}")
        raise RuntimeError("; ".join(failed))
    difference = s_chosen - s_rejected
    if not math.isfinite(difference):
        return 0.0, False
    return difference, True
