"""Reward components for Stage-2 preference RL.

Total objective for parseable rubrics:
    R = exec_coef*R_exec + lambda*R_trans + beta*R_consistency
        + gamma*R_format + chosen_win_bonus*I(chosen > rejected)
        - length_penalty_coef*length_penalty

Unparseable rubrics take a separate fixed negative reward in ``reward_fn.py`` so they
cannot outperform valid but imperfect rubrics in a GRPO group.

This module holds the *pure* (judge-free) pieces: format and transition rewards, and
the assembly `total_reward`. R_exec (judge-based) is computed in `rl/judge.py` and
injected here. R_consistency is 0 in v1 (no dialog linkage in the data; see plan D3).
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Union

from rl.schema import parse_rubric, validate_rubric
from rl.tactic_map import category_weight_rule_errors


@dataclass
class RewardConfig:
    exec_coef: float = 1.0
    exec_tau: float = 0.7
    lambda_trans: float = 0.5
    beta_consistency: float = 0.0
    gamma_format: float = 0.5
    chosen_win_bonus: float = 0.2
    parse_failure_penalty: float = 2.0
    length_penalty_coef: float = 0.1
    length_target: float = 6500.0
    length_min: float = 4500.0
    length_max: float = 9500.0
    length_penalty_max: float = 1.0


Rubric = Union[str, Dict[str, Any]]


def reward_format(rubric: Rubric) -> float:
    """1.0 for a fully schema-valid rubric, graded down per violation, 0.0 if it does
    not even parse as JSON."""
    obj, parse_err = parse_rubric(rubric)
    if parse_err is not None:
        return 0.0
    n_errors = len(validate_rubric(obj))
    return max(0.0, 1.0 - 0.1 * min(10, n_errors))


def reward_trans(rubric: Rubric, tactics: Any) -> float:
    """Start at 1.0 and subtract 0.5 for each of the two tactic/weight violations.
    No tactic signal => 1.0 (rule vacuous). Unparseable => 0.0."""
    obj, parse_err = parse_rubric(rubric)
    if parse_err is not None:
        return 0.0
    n_violations = len(category_weight_rule_errors(obj, tactics))
    return max(0.0, 1.0 - 0.5 * min(2, n_violations))


def reward_length_penalty(
    solution: Optional[str],
    target: float = 6500.0,
    minimum: float = 4500.0,
    maximum: float = 9500.0,
    penalty_max: float = 1.0,
) -> float:
    """Return a linear length penalty, with zero at target and a capped penalty.

    Lengths at or below ``minimum`` and at or above ``maximum`` receive the full
    penalty. Between the bounds, the penalty changes linearly on either side of
    ``target``. The thresholds are deliberately expressed in characters because
    the reward entrypoint receives the generated response as a string.
    """
    values = {"target": target, "minimum": minimum, "maximum": maximum, "penalty_max": penalty_max}
    for name, value in values.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"length penalty {name} must be finite, got {value!r}")
    if minimum >= target or target >= maximum:
        raise ValueError("length_min < length_target < length_max is required")
    if penalty_max < 0.0:
        raise ValueError("length_penalty_max must be non-negative")
    length = len(solution or "")
    if length <= minimum or length >= maximum:
        return float(penalty_max)
    if length < target:
        return float(penalty_max * (target - length) / (target - minimum))
    return float(penalty_max * (length - target) / (maximum - target))


def total_reward(
    r_exec: float,
    r_trans: float,
    r_format: float,
    r_consistency: float = 0.0,
    cfg: Optional[RewardConfig] = None,
    r_chosen_bonus: float = 0.0,
    r_length_penalty: float = 0.0,
) -> float:
    cfg = cfg or RewardConfig()
    values = {
        **vars(cfg),
        "r_exec": r_exec,
        "r_trans": r_trans,
        "r_format": r_format,
        "r_consistency": r_consistency,
        "r_chosen_bonus": r_chosen_bonus,
        "r_length_penalty": r_length_penalty,
    }
    for name, value in values.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"reward value {name} must be finite, got {value!r}")
    score = (
        cfg.exec_coef * r_exec
        + cfg.lambda_trans * r_trans
        + cfg.beta_consistency * r_consistency
        + cfg.gamma_format * r_format
        + r_chosen_bonus
        - cfg.length_penalty_coef * r_length_penalty
    )
    if not math.isfinite(score):
        raise ValueError(f"final reward must be finite, got {score!r}")
    return score
