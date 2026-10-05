import copy
import json

import pytest

from rl.rewards import (
    RewardConfig,
    reward_format,
    reward_length_penalty,
    reward_trans,
    total_reward,
)
from rl.tests.test_schema import GOOD

SPEC_TACTICS = ["advice", "questioning", "emotional_expression", "paraphrasing"]  # Cog 2, Aff 1, Pro 1


def _weights(cog, aff, pro):
    r = copy.deepcopy(GOOD)
    for c in r["categories"]:
        c["category_weight"] = {"Cognitive Empathy": cog, "Affective Empathy": aff, "Proactive Empathy": pro}[c["category"]]
    return r


# ---- reward_format ----

def test_format_valid_is_one():
    assert reward_format(GOOD) == 1.0
    assert reward_format(json.dumps(GOOD)) == 1.0


def test_format_non_json_is_zero():
    assert reward_format("totally not json {{{") == 0.0


def test_format_two_errors_is_point_eight():
    bad = copy.deepcopy(GOOD)
    bad["categories"][0]["criteria"][0]["title"] = ""        # 1 error
    bad["categories"][1]["criteria"][0]["description"] = ""  # 1 error
    assert abs(reward_format(bad) - 0.8) < 1e-9


# ---- reward_trans ----

def test_trans_pass():
    assert reward_trans(_weights(0.5, 0.25, 0.25), SPEC_TACTICS) == 1.0


def test_trans_primary_mismatch_is_half():
    # All categories are involved, so only the highest-weight/primary mismatch applies.
    assert reward_trans(_weights(0.35, 0.45, 0.20), SPEC_TACTICS) == 0.5


def test_trans_both_penalties_zero():
    tactics = ["questioning", "paraphrasing", "validation"]  # Proactive is uninvolved.
    assert reward_trans(_weights(0.3, 0.2, 0.5), tactics) == 0.0


def test_trans_no_tactics_is_one():
    assert reward_trans(_weights(0.34, 0.33, 0.33), []) == 1.0


def test_trans_unparseable_zero():
    assert reward_trans("nope", SPEC_TACTICS) == 0.0


# ---- total_reward ----

def test_total_valid_no_exec_no_consistency():
    # R = 1.0*0 + 0.5*1 + 0.0*x + 0.5*1 = 1.0
    r = total_reward(r_exec=0.0, r_trans=1.0, r_format=1.0, r_consistency=0.0)
    assert abs(r - 1.0) < 1e-9


def test_total_malformed_is_zero():
    r = total_reward(r_exec=0.0, r_trans=0.0, r_format=0.0)
    assert r == 0.0


def test_total_uses_config_coefs():
    cfg = RewardConfig(exec_coef=1.0, lambda_trans=0.3, beta_consistency=0.2, gamma_format=0.4)
    r = total_reward(r_exec=0.5, r_trans=1.0, r_format=1.0, r_consistency=1.0, cfg=cfg)
    assert abs(r - (0.5 + 0.3 + 0.2 + 0.4)) < 1e-9


# ---- length penalty ----

def test_length_penalty_is_zero_at_target():
    assert reward_length_penalty("x" * 6500) == 0.0


def test_length_penalty_is_linear_and_capped():
    assert reward_length_penalty("x" * 5500) == 0.5
    assert reward_length_penalty("x" * 4500) == 1.0
    assert reward_length_penalty("x" * 3500) == 1.0
    assert reward_length_penalty("x" * 8000) == 0.5
    assert reward_length_penalty("x" * 9500) == 1.0
    assert reward_length_penalty("x" * 10500) == 1.0


def test_total_subtracts_length_penalty():
    r = total_reward(0.0, 0.0, 0.0, r_length_penalty=1.0)
    assert r == -0.1


def test_total_uses_length_penalty_coef():
    cfg = RewardConfig(length_penalty_coef=0.25)
    r = total_reward(0.0, 0.0, 0.0, cfg=cfg, r_length_penalty=0.4)
    assert r == -0.1


def test_total_rejects_nonfinite_component_and_config():
    with pytest.raises(ValueError, match="r_exec"):
        total_reward(float("nan"), 0.0, 0.0)
    with pytest.raises(ValueError, match="gamma_format"):
        total_reward(0.0, 0.0, 0.0, cfg=RewardConfig(gamma_format=float("inf")))
