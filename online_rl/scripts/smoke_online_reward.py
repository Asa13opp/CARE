#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from online_rubric_rl.project_paths import SRC_ROOT  # noqa: E402

sys.path.insert(0, str(SRC_ROOT))

from online_rubric_rl.online_reward import score_turn  # noqa: E402
from online_rubric_rl.turn_schema import build_turn_record  # noqa: E402


def main() -> None:
    os.environ.setdefault("MOCK_RUBRIC", "1")
    os.environ.setdefault("MOCK_JUDGE", "1")
    record = build_turn_record(
        turn_id="smoke_0",
        dialogue_id="smoke",
        turn_idx=0,
        history_before_query=[],
        query="I do not know how to keep going with this.",
        assistant_response="That sounds exhausting. We can slow it down and pick one small thing to handle first.",
        messages_after_turn=[
            {"role": "user", "content": "I do not know how to keep going with this."},
            {
                "role": "assistant",
                "content": "That sounds exhausting. We can slow it down and pick one small thing to handle first.",
            },
        ],
    )
    print(json.dumps(score_turn(record), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
