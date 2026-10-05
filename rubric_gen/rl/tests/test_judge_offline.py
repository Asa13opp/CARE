"""Offline tests for the judge math (no API call)."""

import copy

import pytest

import rl.judge as judge
from rl.judge import _cache_key, _coerce_criterion_scores, criterion_title_map, gold_normalized_score


def test_cache_key_includes_history():
    # same rubric + response but different dialogue history must not collide (else a score
    # from one sample's context leaks into another).
    rubric = {"categories": []}
    assert _cache_key("seeker: a", rubric, "resp") != _cache_key("seeker: b", rubric, "resp")
    assert _cache_key("seeker: a", rubric, "resp") == _cache_key("seeker: a", rubric, "resp")

RUBRIC = {
    "categories": [
        {
            "category": "Cognitive Empathy",
            "category_weight": 0.5,
            "criteria": [
                {"title": "P", "description": "d"},
                {"title": "N", "description": "d"},
            ],
        },
        {
            "category": "Affective Empathy",
            "category_weight": 0.25,
            "criteria": [{"title": "A", "description": "d"}],
        },
        {
            "category": "Proactive Empathy",
            "category_weight": 0.25,
            "criteria": [{"title": "Pr", "description": "d"}],
        },
    ]
}


def test_gold_normalized_hand_computed():
    scores = [
        {"category": "Cognitive Empathy", "criterion_title": "P", "score": 2},
        {"category": "Cognitive Empathy", "criterion_title": "N", "score": 0},
        {"category": "Affective Empathy", "criterion_title": "A", "score": 1},
        {"category": "Proactive Empathy", "criterion_title": "Pr", "score": 2},
    ]
    # RAW per-category mean: Cog=(2+0)/2=1.0, Aff=1/1=1.0, Pro=2/1=2.0
    # -> 0.5*1.0 + 0.25*1.0 + 0.25*2.0 = 1.25
    assert abs(gold_normalized_score(RUBRIC, scores) - 1.25) < 1e-9


def test_category_mean_over_raw_scores():
    # Cognitive: 2 and 0 -> raw mean (2 + 0)/2 = 1.0.
    scores = [
        {"category": "Cognitive Empathy", "criterion_title": "P", "score": 2},
        {"category": "Cognitive Empathy", "criterion_title": "N", "score": 0},
        {"category": "Affective Empathy", "criterion_title": "A", "score": 2},
        {"category": "Proactive Empathy", "criterion_title": "Pr", "score": 2},
    ]
    # Cog=1.0, Aff=2.0, Pro=2.0 -> 0.5*1 + 0.25*2 + 0.25*2 = 1.5
    assert abs(gold_normalized_score(RUBRIC, scores) - 1.5) < 1e-9


def test_single_category_raw_mean():
    rubric = {"categories": [{"category": "Cognitive Empathy", "category_weight": 1.0,
                              "criteria": [{"title": "N", "description": "d"}]}]}
    assert gold_normalized_score(rubric, [{"category": "Cognitive Empathy", "criterion_title": "N", "score": 2}]) == 2.0
    assert gold_normalized_score(rubric, [{"category": "Cognitive Empathy", "criterion_title": "N", "score": 0}]) == 0.0


def test_five_point_criterion_means_then_generated_weights():
    scores = [
        {"category": "Cognitive Empathy", "criterion_title": "P", "score": 5},
        {"category": "Cognitive Empathy", "criterion_title": "N", "score": 1},
        {"category": "Affective Empathy", "criterion_title": "A", "score": 4},
        {"category": "Proactive Empathy", "criterion_title": "Pr", "score": 2},
    ]
    # Cog=(5+1)/2=3; Aff=4; Pro=2; 0.5*3 + 0.25*4 + 0.25*2 = 3.0.
    assert gold_normalized_score(RUBRIC, scores) == 3.0


def test_coerce_skips_unknown_and_snaps_scores():
    raw = {"criterion_scores": [
        {"category": "Cognitive Empathy", "criterion_title": "P", "score": 4.7},   # snaps to 5.0
        {"category": "Cognitive Empathy", "criterion_title": "N", "score": -0.4},   # snaps to 0.0
        {"category": "Affective Empathy", "criterion_title": "A", "score": 1},
        {"category": "Proactive Empathy", "criterion_title": "Pr", "score": 5},
        {"category": "Ghost", "criterion_title": "Z", "score": 9},                  # unknown -> dropped
    ]}
    out = _coerce_criterion_scores(RUBRIC, raw)
    assert len(out) == 4
    by = {(o["category"], o["criterion_title"]): o["score"] for o in out}
    assert by[("Cognitive Empathy", "P")] == 5.0
    assert by[("Cognitive Empathy", "N")] == 0.0


def test_coerce_requires_exactly_one_score_per_criterion():
    raw = {"criterion_scores": [
        {"category": "Cognitive Empathy", "criterion_title": "P", "score": 5},
        {"category": "Cognitive Empathy", "criterion_title": "N", "score": 0},
        {"category": "Affective Empathy", "criterion_title": "A", "score": 3},
        # Proactive criterion is missing.
    ]}
    try:
        _coerce_criterion_scores(RUBRIC, raw)
    except ValueError as exc:
        assert "every rubric criterion" in str(exc)
    else:
        raise AssertionError("incomplete judge output must be rejected")


def test_title_map():
    tm = criterion_title_map(RUBRIC)
    assert ("Cognitive Empathy", "P") in tm
    assert ("Cognitive Empathy", "N") in tm
    assert ("Affective Empathy", "A") in tm
    assert ("Proactive Empathy", "Pr") in tm


def test_judge_math_rejects_nonfinite_weight():
    rubric = copy.deepcopy(RUBRIC)
    rubric["categories"][0]["category_weight"] = float("nan")
    with pytest.raises(ValueError, match="non-finite category weight"):
        gold_normalized_score(rubric, [])


def test_coerce_rejects_nonfinite_score():
    raw = {"criterion_scores": [
        {"category": "Cognitive Empathy", "criterion_title": "P", "score": float("inf")},
        {"category": "Cognitive Empathy", "criterion_title": "N", "score": 0},
        {"category": "Affective Empathy", "criterion_title": "A", "score": 1},
        {"category": "Proactive Empathy", "criterion_title": "Pr", "score": 2},
    ]}
    with pytest.raises(ValueError, match="non-finite judge score"):
        _coerce_criterion_scores(RUBRIC, raw)


def test_score_response_retries_incomplete_judge_output(monkeypatch):
    incomplete = {"criterion_scores": [
        {"category": "Cognitive Empathy", "criterion_title": "P", "score": 3},
    ]}
    complete = {"criterion_scores": [
        {"category": "Cognitive Empathy", "criterion_title": "P", "score": 3},
        {"category": "Cognitive Empathy", "criterion_title": "N", "score": 1},
        {"category": "Affective Empathy", "criterion_title": "A", "score": 2},
        {"category": "Proactive Empathy", "criterion_title": "Pr", "score": 4},
    ]}
    calls = iter([incomplete, complete])
    monkeypatch.setenv("JUDGE_MAX_RETRIES", "2")
    monkeypatch.setattr(judge, "_judge_call_once", lambda _messages: next(calls))
    monkeypatch.setattr(judge.time, "sleep", lambda _seconds: None)
    judge._cache.clear()

    assert judge.score_response([], RUBRIC, "retry-me") == 2.5


def test_reward_exec_reports_failed_side_and_root_error(monkeypatch):
    monkeypatch.setattr(
        judge,
        "_score_many_with_errors",
        lambda _requests: ([None, 2.0], ["RuntimeError: upstream 503", None]),
    )
    with pytest.raises(RuntimeError, match="chosen=.*upstream 503"):
        judge.reward_exec_with_status([], RUBRIC, "chosen", "rejected")
