"""verl custom-reward entrypoint.

verl calls `compute_score(data_source, solution_str, ground_truth, extra_info)` per
sample (see `verl/workers/reward_manager/naive.py`). `solution_str` is the policy's
generated rubric JSON; `extra_info` is the per-sample dict packed by the dataset builder
(`prompt`, `chosen_text`, `rejected_text`, `chosen_tactics`).

Returns a dict: {score, r_exec_raw, r_exec, r_trans, r_format, r_consistency,
r_chosen_bonus, r_parse_penalty, r_length_penalty}
so verl logs each term. Ordinary scoring failures do not stop training. Only non-finite
reward values and a sustained total judge outage deliberately terminate a run.
"""

import os
import math
from collections import deque
from typing import Any, Dict, List, Optional

from rl.rewards import (
    RewardConfig,
    reward_format,
    reward_length_penalty,
    reward_trans,
    total_reward,
)
from rl.schema import parse_rubric, validate_rubric

# Lazily-bound judge exec fn; tests can monkeypatch this symbol to avoid network.
_exec_fn = None


def _get_exec_fn():
    global _exec_fn
    if _exec_fn is None:
        from rl.judge import reward_exec_with_status
        _exec_fn = reward_exec_with_status
    return _exec_fn


def _build_cfg(reward_kwargs: Dict[str, Any]) -> RewardConfig:
    def pick(key, default):
        if key in reward_kwargs:
            return float(reward_kwargs[key])
        env = os.environ.get(key.upper())
        return float(env) if env is not None else default
    cfg = RewardConfig(
        exec_coef=pick("exec_coef", 1.0),
        exec_tau=pick("exec_tau", 0.7),
        lambda_trans=pick("lambda_trans", 0.5),
        beta_consistency=pick("beta_consistency", 0.0),
        gamma_format=pick("gamma_format", 0.5),
        chosen_win_bonus=pick("chosen_win_bonus", 0.2),
        parse_failure_penalty=pick("parse_failure_penalty", 2.0),
        length_penalty_coef=pick("length_penalty_coef", 0.1),
        length_target=pick("length_target", 6500.0),
        length_min=pick("length_min", 4500.0),
        length_max=pick("length_max", 9500.0),
        length_penalty_max=pick("length_penalty_max", 1.0),
    )
    for name, value in vars(cfg).items():
        if not math.isfinite(value):
            raise NonFiniteRewardError(f"reward config {name} must be finite, got {value!r}")
    return cfg


# verl's reward manager / AgentLoop `_postprocess` stacks the per-sample reward dicts
# into one array per key (`np.array([info[key] for info in reward_extra_infos])`). That
# requires EVERY sample to return the SAME key set, so `error` is always present (""
# when there is no error) rather than added only on the failure path.
def _zero(note: str = "") -> Dict[str, Any]:
    return {
        "score": 0.0,
        "r_exec_raw": 0.0,
        "r_exec": 0.0,
        "r_trans": 0.0,
        "r_format": 0.0,
        "r_consistency": 0.0,
        "r_chosen_bonus": 0.0,
        "r_parse_penalty": 0.0,
        "r_length_penalty": 0.0,
        "error": note[:200],
    }


class JudgeFailureRateExceeded(RuntimeError):
    """Raised only when the required external judge is persistently unavailable."""


class NonFiniteRewardError(RuntimeError):
    """Raised when a reward component or final score is NaN or infinite."""


class _FailureRateGuard:
    """Worker-local rolling failure monitor.

    verl calls the custom reward function one sample at a time, potentially in several
    RewardLoopWorkers. A sliding window in each worker therefore detects a generation
    collapse without relying on cross-process mutable state. Defaults deliberately need
    a substantial window so occasional malformed samples remain trainable.
    """

    def __init__(self, name: str, exc_type: type[RuntimeError]) -> None:
        self._name = name
        self._exc_type = exc_type
        self._window: deque[bool] = deque()
        self._settings: Optional[tuple[int, float]] = None

    def record(self, failed: bool, *, window: int, rate: float) -> None:
        if window < 1:
            return  # explicit opt-out, useful for short offline probes
        if not 0.0 < rate <= 1.0:
            raise ValueError(f"{self._name} failure rate must be in (0, 1], got {rate}")
        settings = (window, rate)
        if settings != self._settings:
            self._window.clear()
            self._settings = settings
        self._window.append(failed)
        if len(self._window) > window:
            self._window.popleft()
        if len(self._window) == window:
            failures = sum(self._window)
            observed = failures / window
            if observed >= rate:
                raise self._exc_type(
                    f"{self._name} failure rate "
                    f"{observed:.1%} over the last {window} samples exceeds "
                    f"the configured threshold {rate:.1%}; stopping training"
                )


class _JudgeFailureGuard(_FailureRateGuard):
    def __init__(self) -> None:
        super().__init__("judge infrastructure", JudgeFailureRateExceeded)


_judge_failure_guard = _JudgeFailureGuard()


def _record_judge_result(ok: bool, reward_kwargs: Dict[str, Any]) -> None:
    """Stop only when a whole worker-local window has no usable judge result."""
    def pick_int(key: str, default: int) -> int:
        value = reward_kwargs.get(key, os.environ.get(key.upper(), default))
        return int(value)

    def pick_float(key: str, default: float) -> float:
        value = reward_kwargs.get(key, os.environ.get(key.upper(), default))
        return float(value)

    _judge_failure_guard.record(
        not ok,
        window=pick_int("judge_failure_window", 32),
        rate=pick_float("judge_failure_rate", 1.0),
    )


# Optional debug: set REWARD_DEBUG_PATH to append the first N (solution, extra_info,
# reward) samples as JSONL — used to diagnose zero-reward / parsing issues.
_debug_n = 0


def _debug_log(solution_str, extra_info, out):
    global _debug_n
    path = os.environ.get("REWARD_DEBUG_PATH")
    if not path:
        return
    limit = int(os.environ.get("REWARD_DEBUG_N", "20"))
    if _debug_n >= limit:
        return
    _debug_n += 1
    try:
        import json
        rec = {
            "solution_head": (solution_str or "")[:600],
            "solution_len": len(solution_str or ""),
            "has_chosen": bool((extra_info or {}).get("chosen_text")),
            "has_rejected": bool((extra_info or {}).get("rejected_text")),
            "tactics": (extra_info or {}).get("chosen_tactics"),
            "reward": out,
        }
        with open(path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def compute_score(
    data_source: Optional[str] = None,
    solution_str: Optional[str] = None,
    ground_truth: Optional[Any] = None,
    extra_info: Optional[Dict[str, Any]] = None,
    **reward_kwargs: Any,
) -> Dict[str, Any]:
    try:
        extra_info = extra_info or {}
        prompt: List[Dict[str, Any]] = extra_info.get("prompt") or []
        chosen: str = extra_info.get("chosen_text") or ""
        rejected: str = extra_info.get("rejected_text") or ""
        tactics = extra_info.get("chosen_tactics") or []
        cfg = _build_cfg(reward_kwargs)

        obj, parse_err = parse_rubric(solution_str)
        schema_errors = validate_rubric(obj) if parse_err is None else []
        rubric_valid = parse_err is None and not schema_errors
        if not rubric_valid:
            note = "unparseable rubric" if parse_err is not None else f"invalid rubric schema: {schema_errors[0]}"
            out = _zero(note)
            out["r_parse_penalty"] = cfg.parse_failure_penalty
            out["score"] = -cfg.parse_failure_penalty
            _debug_log(solution_str, extra_info, out)
            return out

        r_format = reward_format(obj)
        r_trans = reward_trans(obj, tactics)

        r_exec_raw = 0.0
        use_judge = bool(reward_kwargs.get("use_judge", True))
        if use_judge:
            if not os.environ.get("JUDGE_API_KEY"):
                raise JudgeFailureRateExceeded("JUDGE_API_KEY is required when use_judge=True")
            judge_ok = False
            if chosen and rejected:
                try:
                    result = _get_exec_fn()(prompt, obj, chosen, rejected)
                    # Keep compatibility with offline monkeypatches and older custom judge
                    # implementations that return only a scalar. The built-in judge returns
                    # (difference, success), allowing genuine 0.0 ties to remain valid.
                    if isinstance(result, tuple):
                        r_exec_raw, judge_ok = float(result[0]), bool(result[1])
                    else:
                        r_exec_raw, judge_ok = float(result), True
                except Exception:  # noqa: BLE001 - isolated judge failures are tolerated
                    r_exec_raw, judge_ok = 0.0, False
            if judge_ok and not math.isfinite(r_exec_raw):
                raise NonFiniteRewardError(
                    f"judge returned non-finite score difference {r_exec_raw!r}"
                )
            _record_judge_result(judge_ok, reward_kwargs)
            if not judge_ok:
                r_exec_raw = 0.0

        r_consistency = 0.0  # v1: dropped (no dialog linkage in data)
        if cfg.exec_tau <= 0.0:
            raise ValueError(f"exec_tau must be positive, got {cfg.exec_tau}")
        # Raw judge difference is s_chosen - s_rejected. tanh preserves its sign but
        # bounds large judge gaps to (-1, 1), preventing outliers from dominating GRPO.
        r_exec = math.tanh(r_exec_raw / cfg.exec_tau)
        # The fixed bonus is based on the untransformed strict preference; ties and
        # judge failures (both raw 0.0) receive no bonus.
        r_chosen_bonus = cfg.chosen_win_bonus if r_exec_raw > 0.0 else 0.0
        r_length_penalty = reward_length_penalty(
            solution_str,
            target=cfg.length_target,
            minimum=cfg.length_min,
            maximum=cfg.length_max,
            penalty_max=cfg.length_penalty_max,
        )
        score = total_reward(
            r_exec,
            r_trans,
            r_format,
            r_consistency,
            cfg,
            r_chosen_bonus=r_chosen_bonus,
            r_length_penalty=r_length_penalty,
        )
        components = {
            "r_exec": r_exec,
            "r_trans": r_trans,
            "r_format": r_format,
            "r_consistency": r_consistency,
            "r_chosen_bonus": r_chosen_bonus,
            "r_length_penalty": r_length_penalty,
            "score": score,
        }
        for name, value in components.items():
            if not math.isfinite(value):
                raise NonFiniteRewardError(f"{name} must be finite, got {value!r}")
        out = {
            "score": float(score),
            "r_exec_raw": float(r_exec_raw),
            "r_exec": float(r_exec),
            "r_trans": float(r_trans),
            "r_format": float(r_format),
            "r_consistency": float(r_consistency),
            "r_chosen_bonus": float(r_chosen_bonus),
            "r_parse_penalty": 0.0,
            "r_length_penalty": float(r_length_penalty),
            "error": "",  # keep key set uniform across all samples (see _zero)
        }
        _debug_log(solution_str, extra_info, out)
        return out
    except (JudgeFailureRateExceeded, NonFiniteRewardError):
        raise  # intentional fail-fast signal; do not turn a collapsed run into zero rewards
    except Exception as exc:  # noqa: BLE001 - ordinary reward failures must not crash training
        out = _zero(f"{type(exc).__name__}: {exc}")
        _debug_log(solution_str, extra_info, out)
        return out
