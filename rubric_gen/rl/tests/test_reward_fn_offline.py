"""Offline tests for the verl reward entrypoint (judge monkeypatched, no network)."""

import json
import math

import pytest

import rl.reward_fn as rf
from rl.tests.test_schema import GOOD

SPEC_TACTICS = ["advice", "questioning", "emotional_expression", "paraphrasing"]


def _extra(**kw):
    base = {
        "prompt": [{"role": "user", "content": "hi"}],
        "chosen_text": "good reply",
        "rejected_text": "bad reply",
        "chosen_tactics": SPEC_TACTICS,
    }
    base.update(kw)
    return base


def test_keys_and_sum(monkeypatch):
    # Force a known r_exec; GOOD weights: Cog .5 (top tactic) > Aff .25 = Pro .25 -> trans ok
    monkeypatch.setenv("JUDGE_API_KEY", "x")
    monkeypatch.setattr(rf, "_exec_fn", lambda prompt, rubric, c, r: 0.3)
    out = rf.compute_score(solution_str=json.dumps(GOOD), extra_info=_extra(),
                           lambda_trans=0.5, gamma_format=0.5, exec_coef=1.0,
                           length_penalty_max=0.0)
    for k in ("score", "r_exec_raw", "r_exec", "r_trans", "r_format", "r_consistency", "r_chosen_bonus"):
        assert k in out
    assert out["r_format"] == 1.0
    assert out["r_trans"] == 1.0
    assert out["r_exec_raw"] == 0.3
    assert out["r_exec"] == math.tanh(0.3 / 0.7)
    assert out["r_chosen_bonus"] == 0.2
    # score = tanh(0.3 / 0.7) + 0.5*1 + 0.5*1 + 0.2
    assert abs(out["score"] - (math.tanh(0.3 / 0.7) + 1.2)) < 1e-9


def test_explicitly_disabled_judge_means_exec_zero(monkeypatch):
    monkeypatch.delenv("JUDGE_API_KEY", raising=False)
    out = rf.compute_score(
        solution_str=json.dumps(GOOD), extra_info=_extra(), length_penalty_max=0.0,
        use_judge=False,
    )
    assert out["r_exec"] == 0.0
    # default coeffs 0.5/0.5 -> 0 + 0.5 + 0.5 = 1.0
    assert abs(out["score"] - 1.0) < 1e-9


def test_missing_judge_key_stops_immediately(monkeypatch):
    monkeypatch.delenv("JUDGE_API_KEY", raising=False)
    with pytest.raises(rf.JudgeFailureRateExceeded, match="JUDGE_API_KEY"):
        rf.compute_score(solution_str=json.dumps(GOOD), extra_info=_extra())


def test_isolated_judge_failure_is_neutral(monkeypatch):
    monkeypatch.setenv("JUDGE_API_KEY", "x")
    monkeypatch.setattr(rf, "_judge_failure_guard", rf._JudgeFailureGuard())
    def boom(*a, **k):
        raise RuntimeError("judge down")
    monkeypatch.setattr(rf, "_exec_fn", boom)
    out = rf.compute_score(
        solution_str=json.dumps(GOOD), extra_info=_extra(),
        judge_failure_window=32, judge_failure_rate=1.0,
    )
    assert out["r_exec_raw"] == 0.0
    assert out["r_exec"] == 0.0
    assert out["r_chosen_bonus"] == 0.0


def test_malformed_rubric_is_penalized():
    out = rf.compute_score(solution_str="{not valid json", extra_info=_extra())
    assert out["score"] == -2.0
    assert out["r_parse_penalty"] == 2.0
    assert out["r_format"] == 0.0
    assert out["error"] == "unparseable rubric"


def test_empty_solution_is_penalized():
    out = rf.compute_score(solution_str="", extra_info=_extra())
    assert out["score"] == -2.0
    assert out["r_parse_penalty"] == 2.0


def test_schema_invalid_rubric_is_penalized():
    out = rf.compute_score(solution_str=json.dumps({}), extra_info=_extra())
    assert out["score"] == -2.0
    assert out["r_parse_penalty"] == 2.0
    assert out["error"].startswith("invalid rubric schema:")


def test_return_keys_are_uniform_across_success_and_failure(monkeypatch):
    # verl's AgentLoop/_postprocess stacks per-sample reward dicts into one array per
    # key, so a parseable and an unparseable sample in the same batch MUST expose the
    # identical key set (else KeyError: 'error'). Regression for the DAPO smoke crash.
    monkeypatch.delenv("JUDGE_API_KEY", raising=False)
    ok = rf.compute_score(solution_str=json.dumps(GOOD), extra_info=_extra(), use_judge=False)
    bad = rf.compute_score(solution_str="aget\n" + json.dumps(GOOD), extra_info=_extra())
    assert set(ok.keys()) == set(bad.keys())
    assert "error" in ok and ok["error"] == ""
    assert ok["r_parse_penalty"] == 0.0


def test_trans_penalised_when_weights_violate_tactics(monkeypatch):
    monkeypatch.delenv("JUDGE_API_KEY", raising=False)
    bad = json.loads(json.dumps(GOOD))
    # Affective, not primary Cognitive, has the highest weight. All three categories
    # are involved, so only the first penalty applies.
    for c in bad["categories"]:
        c["category_weight"] = {"Cognitive Empathy": 0.35, "Affective Empathy": 0.45, "Proactive Empathy": 0.20}[c["category"]]
    out = rf.compute_score(solution_str=json.dumps(bad), extra_info=_extra(), use_judge=False)
    assert out["r_trans"] == 0.5


def test_sustained_parse_failure_is_penalized_without_stopping():
    for _ in range(64):
        out = rf.compute_score(solution_str="{not valid json", extra_info=_extra())
        assert out["score"] == -2.0


def test_judge_false_status_is_neutral(monkeypatch):
    monkeypatch.setenv("JUDGE_API_KEY", "x")
    monkeypatch.setattr(rf, "_judge_failure_guard", rf._JudgeFailureGuard())
    monkeypatch.setattr(rf, "_exec_fn", lambda *args: (0.0, False))
    out = rf.compute_score(
        solution_str=json.dumps(GOOD), extra_info=_extra(), judge_failure_window=32
    )
    assert out["r_exec"] == 0.0


def test_complete_sustained_judge_outage_stops(monkeypatch):
    monkeypatch.setenv("JUDGE_API_KEY", "x")
    monkeypatch.setattr(rf, "_judge_failure_guard", rf._JudgeFailureGuard())
    monkeypatch.setattr(rf, "_exec_fn", lambda *args: (0.0, False))
    for _ in range(3):
        rf.compute_score(
            solution_str=json.dumps(GOOD), extra_info=_extra(),
            judge_failure_window=4, judge_failure_rate=1.0,
        )
    with pytest.raises(rf.JudgeFailureRateExceeded, match="100.0%"):
        rf.compute_score(
            solution_str=json.dumps(GOOD), extra_info=_extra(),
            judge_failure_window=4, judge_failure_rate=1.0,
        )


def test_judge_tie_does_not_count_as_failure(monkeypatch):
    monkeypatch.setenv("JUDGE_API_KEY", "x")
    monkeypatch.setattr(rf, "_exec_fn", lambda *args: (0.0, True))
    out = rf.compute_score(solution_str=json.dumps(GOOD), extra_info=_extra())
    assert out["r_chosen_bonus"] == 0.0


def test_nonfinite_judge_difference_stops(monkeypatch):
    monkeypatch.setenv("JUDGE_API_KEY", "x")
    monkeypatch.setattr(rf, "_exec_fn", lambda *args: (float("nan"), True))
    with pytest.raises(rf.NonFiniteRewardError, match="non-finite"):
        rf.compute_score(solution_str=json.dumps(GOOD), extra_info=_extra())


def test_nonfinite_reward_config_stops(monkeypatch):
    monkeypatch.delenv("JUDGE_API_KEY", raising=False)
    with pytest.raises(rf.NonFiniteRewardError, match="gamma_format"):
        rf.compute_score(
            solution_str=json.dumps(GOOD), extra_info=_extra(),
            use_judge=False, gamma_format=float("inf"),
        )
