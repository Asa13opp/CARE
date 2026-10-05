"""Offline tests for the per-step reward-component tracer."""

import json

from rl import reward_trace as rt


def _write(d, step, rows):
    p = d / f"{step}.jsonl"
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


def test_summarize_means_and_fail_rate(tmp_path):
    p = _write(tmp_path, 1, [
        {"score": 1.1, "r_exec": 0.1, "r_trans": 1.0, "r_format": 1.0, "r_consistency": 0.0, "error": ""},
        {"score": "0.5", "r_exec": "-0.2", "r_trans": "0.5", "r_format": "1.0", "r_consistency": "0.0", "error": ""},
        {"score": 0.0, "r_exec": 0.0, "r_trans": 0.0, "r_format": 0.0, "r_consistency": 0.0, "error": "unparseable rubric"},
    ])
    s = rt.summarize(str(p))
    assert s["n"] == 3
    assert abs(s["score"] - (1.1 + 0.5 + 0.0) / 3) < 1e-6
    assert abs(s["r_exec"] - (0.1 - 0.2 + 0.0) / 3) < 1e-6      # stringified values coerced
    assert abs(s["r_format"] - (1.0 + 1.0 + 0.0) / 3) < 1e-6
    assert abs(s["parse_fail"] - 1 / 3) < 1e-6                   # one error row


def test_tolerates_half_written_trailing_line(tmp_path):
    p = tmp_path / "2.jsonl"
    with open(p, "w") as f:
        f.write(json.dumps({"score": 1.3, "r_exec": 0.3, "r_trans": 1.0, "r_format": 1.0, "r_consistency": 0.0, "error": ""}) + "\n")
        f.write('{"score":0.8,"r_exec":0.1,"r_tr')  # truncated mid-write
    s = rt.summarize(str(p))
    assert s["n"] == 1
    assert abs(s["score"] - 1.3) < 1e-6


def test_step_files_sorted_numeric(tmp_path):
    for step in (10, 2, 1):
        _write(tmp_path, step, [{"score": 1.0, "r_exec": 0, "r_trans": 1, "r_format": 1, "r_consistency": 0, "error": ""}])
    (tmp_path / "notes.txt").write_text("ignore me")  # non-step file ignored
    steps = [s for s, _ in rt.step_files(str(tmp_path))]
    assert steps == [1, 2, 10]


def test_empty_dir_summarize_none(tmp_path):
    assert rt.summarize(str(tmp_path / "missing.jsonl")) is None
