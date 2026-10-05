"""Build the Stage-2 RL dataset (verl parquet) from the preference jsonl.

Source: supplied via --input (default: data/empathy_slim_tactic_tagged_en.jsonl)
Each record -> one verl row:
  - data_source = "rubric_rl"
  - prompt      = [ {system: SYSTEM_PROMPT}, {user: rendered dialogue history} ]
                  (the policy input; verl applies the chat template + left-truncation)
  - ability     = "rubric"
  - reward_model = {"style": "rule", "ground_truth": ""}   (unused; reward is custom)
  - extra_info  = {prompt: history turns (role-mapped) for the judge,
                   chosen_text, rejected_text, chosen_tactics, sample_id, split, index}

Role mapping: the RL data uses user/assistant; the SFT policy was trained on
seeker/supporter, so we remap user->seeker, assistant->supporter for both the policy
prompt and the judge context.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# import SYSTEM_PROMPT + renderer from the Stage-1 code
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = str(_PROJECT_ROOT / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from common import SYSTEM_PROMPT, make_messages, render_dialog_context  # noqa: E402

ROLE_MAP = {"user": "seeker", "assistant": "supporter", "seeker": "seeker", "supporter": "supporter"}


def remap_history(turns):
    out = []
    for t in turns:
        out.append({"role": ROLE_MAP.get(t.get("role"), t.get("role")), "content": t.get("content", "")})
    return out


def split_of(sample_id: str, val_frac: float) -> str:
    h = int(hashlib.sha256(str(sample_id).encode()).hexdigest(), 16) % 10_000
    return "val" if h < val_frac * 10_000 else "train"


def build_row(rec, index):
    history = remap_history(rec.get("prompt", []))
    chosen = rec.get("chosen", [{}])[0]
    rejected = rec.get("rejected", [{}])[0]
    rendered = render_dialog_context(history)
    prompt_messages = make_messages(rendered)  # [system, user]
    return {
        "data_source": "rubric_rl",
        "prompt": prompt_messages,
        "ability": "rubric",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {
            "prompt": history,
            "chosen_text": chosen.get("content", ""),
            "rejected_text": rejected.get("content", ""),
            "chosen_tactics": chosen.get("all_tactics", []),
            "sample_id": rec.get("sample_id", str(index)),
            "index": index,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    # Supply --input when the preference JSONL is stored outside the repository data directory.
    ap.add_argument("--input", default=str(_PROJECT_ROOT / "data" / "empathy_slim_tactic_tagged_en.jsonl"))
    ap.add_argument("--out-dir", default=str(_PROJECT_ROOT / "data" / "rl"))
    ap.add_argument("--val-frac", type=float, default=0.05)
    args = ap.parse_args()

    import pandas as pd

    rows = {"train": [], "val": []}
    with open(args.input, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            row = build_row(rec, i)
            split = split_of(row["extra_info"]["sample_id"], args.val_frac)
            row["extra_info"]["split"] = split
            rows[split].append(row)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for split, data in rows.items():
        df = pd.DataFrame(data)
        path = out_dir / f"{split}.parquet"
        df.to_parquet(path, index=False)
        report[split] = len(df)
    report["total"] = report["train"] + report["val"]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
