"""Offline tests for the preference-eval aggregation (no model/judge)."""

import math

from rl.preference_eval import summarize_preferences


def test_win_rate_and_margins():
    samples = [
        {"parsed": True, "judged": True, "margin": 1.0},    # win
        {"parsed": True, "judged": True, "margin": 0.5},    # win
        {"parsed": True, "judged": True, "margin": -0.5},   # loss
        {"parsed": True, "judged": True, "margin": 0.0},    # tie
    ]
    s = summarize_preferences(samples)
    assert s["n"] == 4 and s["n_scored"] == 4
    assert abs(s["win_rate"] - 0.5) < 1e-9
    assert abs(s["tie_rate"] - 0.25) < 1e-9
    assert abs(s["loss_rate"] - 0.25) < 1e-9
    assert abs(s["mean_margin"] - 0.25) < 1e-9
    assert abs(s["median_margin"] - 0.25) < 1e-9


def test_format_and_judge_failures_excluded_from_winrate():
    samples = [
        {"parsed": False, "judged": False, "margin": None},  # unparseable rubric
        {"parsed": True, "judged": False, "margin": None},   # judge failed
        {"parsed": True, "judged": True, "margin": 0.8},     # win
    ]
    s = summarize_preferences(samples)
    assert s["n"] == 3 and s["n_parsed"] == 2 and s["n_scored"] == 1
    assert abs(s["format_fail_rate"] - 1 / 3) < 1e-9
    assert abs(s["judge_fail_rate"] - 1 / 2) < 1e-9   # among parsed
    assert s["win_rate"] == 1.0                        # over the single scored pair


def test_all_failed_is_nan_not_crash():
    s = summarize_preferences([{"parsed": False, "judged": False, "margin": None}])
    assert s["n"] == 1 and s["n_scored"] == 0
    assert math.isnan(s["win_rate"]) and math.isnan(s["mean_margin"])
