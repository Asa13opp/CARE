from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from empathy_rl.judge_reward import response_body_token_count
from online_rubric_rl.online_reward import score_turns
from online_rubric_rl.simulator_log import append_scored_simulator_logs
from online_rubric_rl.turn_schema import build_turn_record, plain, split_history_and_query


class OnlineRubricEnvironment:
    """veRL environment reward adapter for per-turn online rubric judging."""

    def __init__(self, config: Any, tokenizer: Any) -> None:
        self.config = config
        self.tokenizer = tokenizer

    def _decode_response(self, data_item: Any) -> str:
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        attention_mask = data_item.batch["attention_mask"]
        response_ids = data_item.batch["responses"]
        valid_response_length = int(attention_mask[prompt_length:].sum().item())
        valid_response_ids = response_ids[:valid_response_length]
        return self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

    def _reward_loc(self, data_item: Any) -> int:
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum().item())
        return max(0, valid_response_length - 1)

    def _record_from_item(self, data_item: Any, idx: int) -> Dict[str, Any]:
        nt = plain(data_item.non_tensor_batch)
        if isinstance(nt.get("turn_record"), dict):
            rec = dict(nt["turn_record"])
            extra = rec.get("extra") if isinstance(rec.get("extra"), dict) else {}
            if "response_body_token_count" not in extra:
                extra = dict(extra)
                extra["response_body_token_count"] = response_body_token_count(
                    rec.get("assistant_response", ""),
                    self.tokenizer,
                )
                rec["extra"] = extra
            return rec
        for key in (
            "turn_id",
            "dialogue_id",
            "turn_idx",
            "rubric_history",
            "query",
            "judge_history",
            "assistant_response",
            "messages_after_turn",
        ):
            if key in nt:
                continue
        messages = nt.get("messages_after_turn") or nt.get("messages") or []
        history_before_query, query = split_history_and_query(messages)
        assistant_response = nt.get("assistant_response") or self._decode_response(data_item)
        return build_turn_record(
            turn_id=nt.get("turn_id", f"turn_{idx}"),
            dialogue_id=nt.get("dialogue_id", f"dialogue_{idx}"),
            turn_idx=int(nt.get("turn_idx", idx)),
            history_before_query=nt.get("rubric_history", history_before_query),
            query=nt.get("query", query),
            assistant_response=assistant_response,
            messages_after_turn=messages,
            extra={
                "source_non_tensor_keys": sorted(str(k) for k in nt.keys()),
                "response_body_token_count": response_body_token_count(assistant_response, self.tokenizer),
            },
        )

    def get_reward_batched(self, data: Any, global_step: int | None = None):
        del global_step
        records: List[Dict[str, Any]] = [self._record_from_item(data[i], i) for i in range(len(data))]
        group_ids = [
            plain(data[i].non_tensor_batch).get("uid", records[i].get("turn_id", i))
            for i in range(len(data))
        ]
        reward_infos = score_turns(records, group_ids=group_ids)

        simulator_log_metadata = data.non_tensor_batch.get("simulator_log_metadata")
        if simulator_log_metadata is not None:
            append_scored_simulator_logs(simulator_log_metadata, records, reward_infos)

        original_reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_locs = [self._reward_loc(data[i]) for i in range(len(data))]
        for i, info in enumerate(reward_infos):
            score = float(info.get("score", 0.0))
            original_reward_tensor[i, reward_locs[i]] = score
            reward_tensor[i, reward_locs[i]] = score

        data.non_tensor_batch["online_reward_info"] = np.array(reward_infos, dtype=object)
        return original_reward_tensor, reward_tensor
