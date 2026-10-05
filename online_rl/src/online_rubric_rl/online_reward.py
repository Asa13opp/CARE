from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from empathy_rl.judge_reward import (
    gold_normalized_score,
    response_length_penalty,
    score_response_details,
)

from online_rubric_rl.rubric_runtime import RubricRuntime
from online_rubric_rl.turn_schema import plain, validate_turn_record


_runtime = RubricRuntime()


def _fail_fast_enabled() -> bool:
    return os.environ.get("ONLINE_REWARD_FAIL_FAST", "0") != "0"


def _append_debug(record: Dict[str, Any]) -> None:
    path = os.environ.get("ONLINE_REWARD_DEBUG_PATH")
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(plain(record), ensure_ascii=False) + "\n")
    except Exception:
        pass


def _score_with_length_penalty(judge: float, rec: Dict[str, Any]) -> Dict[str, Any]:
    extra = rec.get("extra") if isinstance(rec.get("extra"), dict) else {}
    token_count = extra.get("response_body_token_count")
    try:
        token_count = int(token_count) if token_count is not None else None
    except (TypeError, ValueError):
        token_count = None
    penalty, body_tokens, length_limit = response_length_penalty(
        rec.get("assistant_response", ""),
        body_token_count=token_count,
    )
    return {
        "score": float(judge + penalty),
        "judge_reward": float(judge),
        "length_penalty": float(penalty),
        "response_body_tokens": float(body_tokens),
        "response_length_limit": float(length_limit),
    }


def _criterion_key(item: Dict[str, Any]) -> tuple[Any, Any]:
    return item.get("category"), item.get("criterion_title")


def _nonconstant_criteria(group_infos: List[Dict[str, Any]], epsilon: float = 1e-12) -> set[tuple[Any, Any]]:
    values: Dict[tuple[Any, Any], List[float]] = {}
    for info in group_infos:
        for item in info.get("criterion_scores") or []:
            if not isinstance(item, dict):
                continue
            try:
                score = float(item["score"])
            except (KeyError, TypeError, ValueError):
                continue
            values.setdefault(_criterion_key(item), []).append(score)
    return {
        key
        for key, scores in values.items()
        if len(scores) <= 1 or max(scores) - min(scores) > epsilon
    }


def _apply_group_criterion_filter(
    reward_infos: List[Dict[str, Any]],
    group_ids: List[Any] | None = None,
) -> List[Dict[str, Any]]:
    if not reward_infos:
        return reward_infos
    if group_ids is None:
        group_ids = [
            (info.get("turn_id") or info.get("dialogue_id") or idx)
            for idx, info in enumerate(reward_infos)
        ]
    groups: Dict[Any, List[int]] = {}
    for idx, group_id in enumerate(group_ids):
        groups.setdefault(group_id, []).append(idx)

    for indices in groups.values():
        if len(indices) <= 1:
            continue
        group_infos = [reward_infos[i] for i in indices]
        included = _nonconstant_criteria(group_infos)
        for i in indices:
            info = reward_infos[i]
            rubric = info.get("rubric")
            criterion_scores = info.get("criterion_scores")
            if not rubric or not criterion_scores:
                continue
            filtered_judge = gold_normalized_score(rubric, criterion_scores, included)
            old_judge = float(info.get("judge_reward", 0.0))
            info["judge_reward_unfiltered"] = old_judge
            info["judge_reward"] = float(filtered_judge)
            info["score_unfiltered"] = float(info.get("score", old_judge))
            info["score"] = float(info["score_unfiltered"] - old_judge + filtered_judge)
            info["criterion_filter_included"] = [
                {"category": key[0], "criterion_title": key[1]}
                for key in sorted(included, key=lambda x: (str(x[0]), str(x[1])))
            ]
            info["criterion_filter_excluded_count"] = float(
                max(0, len({_criterion_key(item) for item in criterion_scores}) - len(included))
            )
    return reward_infos


def score_turn(record: Dict[str, Any], runtime: RubricRuntime | None = None) -> Dict[str, Any]:
    runtime = runtime or _runtime
    record = plain(record)
    errors = validate_turn_record(record)
    if errors:
        out = {
            "score": 0.0,
            "judge_reward": 0.0,
            "rubric_valid": 0.0,
            "rubric_repaired": 0.0,
            "error": "; ".join(errors[:3]),
        }
        _append_debug({"turn": record, "reward": out})
        if _fail_fast_enabled():
            raise ValueError(out["error"])
        return out

    rubric_info = runtime.generate_one(record["rubric_history"], record["query"])
    if not rubric_info.get("rubric_valid") or not rubric_info.get("rubric"):
        out = {
            "score": 0.0,
            "judge_reward": 0.0,
            "rubric_valid": 0.0,
            "rubric_repaired": float(bool(rubric_info.get("rubric_repaired"))),
            "error": "; ".join(rubric_info.get("rubric_errors") or ["invalid rubric"])[:300],
            "rubric_raw": rubric_info.get("rubric_raw", ""),
        }
        _append_debug({"turn": record, "reward": out})
        if _fail_fast_enabled():
            raise ValueError(out["error"])
        return out

    try:
        judge_result = score_response_details(
            record["judge_history"],
            record["query"],
            rubric_info["rubric"],
            record["assistant_response"],
        )
        judge = float(judge_result["score"])
        reward_values = _score_with_length_penalty(judge, record)
        out = {
            **reward_values,
            "rubric_valid": 1.0,
            "rubric_repaired": float(bool(rubric_info.get("rubric_repaired"))),
            "error": "",
            "rubric": rubric_info["rubric"],
            "criterion_scores": judge_result.get("criterion_scores", []),
            "turn_id": record.get("turn_id"),
            "dialogue_id": record.get("dialogue_id"),
        }
        _append_debug({"turn": record, "reward": out})
        return out
    except Exception as exc:  # noqa: BLE001
        out = {
            "score": 0.0,
            "judge_reward": 0.0,
            "rubric_valid": 1.0,
            "rubric_repaired": float(bool(rubric_info.get("rubric_repaired"))),
            "error": f"{type(exc).__name__}: {exc}"[:300],
            "rubric": rubric_info["rubric"],
        }
        _append_debug({"turn": record, "reward": out})
        if _fail_fast_enabled():
            raise
        return out


def score_turns(
    records: List[Dict[str, Any]],
    runtime: RubricRuntime | None = None,
    group_ids: List[Any] | None = None,
) -> List[Dict[str, Any]]:
    """Batch-optimised variant of score_turn.

    Phase 1: validate all records (cheap, serial).
    Phase 2: generate_many – one batched rubric call for all valid records.
    Phase 3: parallel judge calls (up to JUDGE_MAX_INFLIGHT threads).
    """
    runtime = runtime or _runtime
    if not records:
        return []

    results: List[Any] = [None] * len(records)

    # Phase 1: validate
    pending: List[tuple[int, Dict[str, Any]]] = []
    for i, record in enumerate(records):
        rec = plain(record)
        errors = validate_turn_record(rec)
        if errors:
            out = {
                "score": 0.0, "judge_reward": 0.0,
                "rubric_valid": 0.0, "rubric_repaired": 0.0,
                "error": "; ".join(errors[:3]),
            }
            _append_debug({"turn": rec, "reward": out})
            if _fail_fast_enabled():
                raise ValueError(out["error"])
            results[i] = out
        else:
            pending.append((i, rec))

    if not pending:
        return results

    # Phase 2: batch rubric
    rubric_infos = runtime.generate_many(
        [(rec["rubric_history"], rec["query"]) for _, rec in pending]
    )

    rubric_ok: List[tuple[int, Dict[str, Any], Dict[str, Any]]] = []
    for (i, rec), rubric_info in zip(pending, rubric_infos):
        if not rubric_info.get("rubric_valid") or not rubric_info.get("rubric"):
            out = {
                "score": 0.0, "judge_reward": 0.0,
                "rubric_valid": 0.0,
                "rubric_repaired": float(bool(rubric_info.get("rubric_repaired"))),
                "error": "; ".join(rubric_info.get("rubric_errors") or ["invalid rubric"])[:300],
                "rubric_raw": rubric_info.get("rubric_raw", ""),
            }
            _append_debug({"turn": rec, "reward": out})
            if _fail_fast_enabled():
                raise ValueError(out["error"])
            results[i] = out
        else:
            rubric_ok.append((i, rec, rubric_info))

    if not rubric_ok:
        return results

    # Phase 3: batch judge
    def _judge_one(i: int, rec: Dict[str, Any], rubric_info: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
        try:
            judge_result = score_response_details(
                rec["judge_history"], rec["query"],
                rubric_info["rubric"], rec["assistant_response"],
            )
            judge = float(judge_result["score"])
            reward_values = _score_with_length_penalty(judge, rec)
            out = {
                **reward_values,
                "rubric_valid": 1.0,
                "rubric_repaired": float(bool(rubric_info.get("rubric_repaired"))),
                "error": "", "rubric": rubric_info["rubric"],
                "criterion_scores": judge_result.get("criterion_scores", []),
                "turn_id": rec.get("turn_id"),
                "dialogue_id": rec.get("dialogue_id"),
            }
        except Exception as exc:  # noqa: BLE001
            out = {
                "score": 0.0, "judge_reward": 0.0,
                "rubric_valid": 1.0,
                "rubric_repaired": float(bool(rubric_info.get("rubric_repaired"))),
                "error": f"{type(exc).__name__}: {exc}"[:300],
                "rubric": rubric_info["rubric"],
            }
            _append_debug({"turn": rec, "reward": out})
            if _fail_fast_enabled():
                raise
            return i, out
        _append_debug({"turn": rec, "reward": out})
        return i, out

    max_workers = min(
        max(1, int(os.environ.get("JUDGE_MAX_INFLIGHT", "32"))),
        len(rubric_ok),
    )
    if max_workers == 1:
        for args in rubric_ok:
            idx, out = _judge_one(*args)
            results[idx] = out
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_judge_one, *args): args[0] for args in rubric_ok}
            for fut in as_completed(futures):
                idx, out = fut.result()
                results[idx] = out

    return _apply_group_criterion_filter(results, group_ids)
