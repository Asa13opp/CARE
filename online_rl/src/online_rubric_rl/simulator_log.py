from __future__ import annotations

import copy
import json
import os
import threading
from collections import OrderedDict
from typing import Any, Dict, Iterable

from online_rubric_rl.turn_schema import plain


_SAVE_LOCK = threading.Lock()


def append_scored_simulator_logs(
    metadata_items: Iterable[Any],
    turn_records: Iterable[Dict[str, Any]],
    reward_infos: Iterable[Dict[str, Any]],
) -> int:
    """Append one simulator JSONL record per dialogue after online reward is known.

    ``metadata_items`` is aligned with the flattened turn batch.  Only the first
    turn needs to carry the (large) simulator record; later turns carry the same
    ``log_id`` and path so their rewards can be joined without duplicating it.
    """
    groups: OrderedDict[str, Dict[str, Any]] = OrderedDict()
    for metadata, turn, reward in zip(metadata_items, turn_records, reward_infos):
        metadata = plain(metadata)
        if not isinstance(metadata, dict):
            continue
        log_id = str(metadata.get("log_id", "")).strip()
        log_path = str(metadata.get("log_path", "")).strip()
        if not log_id or not log_path:
            continue

        group = groups.setdefault(
            log_id,
            {"log_path": log_path, "record": None, "turn_rewards": []},
        )
        if isinstance(metadata.get("record"), dict):
            group["record"] = copy.deepcopy(plain(metadata["record"]))

        turn = plain(turn) if isinstance(turn, dict) else {}
        reward = plain(reward) if isinstance(reward, dict) else {}
        # This is an internal diagnostic list used by the group criterion
        # filter.  Keep it out of the large simulator log while preserving
        # the field in the reward info used by training and debugging.
        reward.pop("criterion_filter_included", None)
        group["turn_rewards"].append(
            {
                "turn_id": turn.get("turn_id"),
                "turn_idx": turn.get("turn_idx"),
                "query": turn.get("query", ""),
                "assistant_response": turn.get("assistant_response", ""),
                **reward,
            }
        )

    saved = 0
    with _SAVE_LOCK:
        for group in groups.values():
            record = group["record"]
            if not isinstance(record, dict):
                continue
            record["online_rubric_rewards"] = group["turn_rewards"]
            log_path = group["log_path"]
            log_dir = os.path.dirname(log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            saved += 1
    return saved
