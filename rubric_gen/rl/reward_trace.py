"""See each reward component change over training steps.

verl logs only the *total* reward (critic/score) to console/tensorboard; the per-component
breakdown returned by `rl/reward_fn.py` (r_exec_raw / r_exec / r_trans / r_format /
r_consistency) is
not surfaced in the training metrics. But when `trainer.rollout_data_dir` is set, verl
dumps one JSONL per step (`<dir>/<step>.jsonl`), one line per rollout sample, carrying
`score` plus those component keys. This tool reads those dumps and reports per-step means
so the components are visible *during* training.

Usage (run alongside / after training, in the verl env):
  python -m rl.reward_trace --run grpov2_2026..._exec2.0  # a specific run by id
  python -m rl.reward_trace --run grpov2_2026..._exec2.0 --watch
  python -m rl.reward_trace --dir <any/rollout_data> --watch   # explicit dir still works

--run <id> resolves to output/rl/<id>/rollout_data, and with --watch --tb (no value) it
writes tensorboard curves under output/rl/<id>/reward_tb.

(real runs of run_grpo.sh / run_dapo.sh enable the dump automatically; for the 2-step
SMOKE runs it stays off.)
"""

import argparse
import glob
import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple

# repo root = two levels up from this file (rl/reward_trace.py -> rubric_gen/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COMPONENTS = ["score", "r_exec_raw", "r_exec", "r_trans", "r_format", "r_consistency", "r_chosen_bonus"]


def _num(x) -> Optional[float]:
    """Coerce a value to float; verl may stringify np.float64 via json default=str."""
    try:
        if x is None:
            return None
        f = float(x)
        return f if not math.isnan(f) else None
    except (TypeError, ValueError):
        return None


def step_files(d: str) -> List[Tuple[int, str]]:
    out = []
    for p in glob.glob(os.path.join(d, "*.jsonl")):
        base = os.path.basename(p)[:-6]
        if base.isdigit():
            out.append((int(base), p))
    return sorted(out)


def summarize(path: str) -> Optional[Dict[str, float]]:
    rows = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # tolerate a half-written trailing line
    except FileNotFoundError:
        return None
    if not rows:
        return None
    n = len(rows)
    out: Dict[str, float] = {"n": float(n)}
    for k in COMPONENTS:
        vals = [v for v in (_num(r.get(k)) for r in rows) if v is not None]
        out[k] = sum(vals) / len(vals) if vals else float("nan")
    # format failure rate: an `error` note was set (e.g. unparseable rubric)
    out["parse_fail"] = sum(1 for r in rows if str(r.get("error") or "").strip()) / n
    return out


_HDR = (
    f"{'step':>5} {'n':>4} {'score':>7} {'raw_exec':>8} {'r_exec':>7} {'r_trans':>7} "
    f"{'r_format':>8} {'r_cons':>7} {'r_bonus':>8} {'fail%':>6}"
)


def _fmt(step: int, s: Dict[str, float]) -> str:
    return (
        f"{step:>5} {int(s['n']):>4} {s['score']:>7.3f} {s['r_exec_raw']:>8.3f} {s['r_exec']:>7.3f} "
        f"{s['r_trans']:>7.3f} {s['r_format']:>8.3f} {s['r_consistency']:>7.3f} "
        f"{s['r_chosen_bonus']:>8.3f} "
        f"{100*s['parse_fail']:>5.1f}%"
    )


def _make_tb(tb_dir: str):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # noqa: BLE001
        print(f"[reward_trace] tensorboard unavailable ({exc}); skipping --tb")
        return None
    os.makedirs(tb_dir, exist_ok=True)
    return SummaryWriter(tb_dir)


def _tb_write(writer, step: int, s: Dict[str, float]):
    if writer is None:
        return
    for k in COMPONENTS:
        if not math.isnan(s[k]):
            writer.add_scalar(f"reward/{k}", s[k], step)
    writer.add_scalar("reward/parse_fail_rate", s["parse_fail"], step)
    writer.flush()


def run_once(d: str, tb_dir: Optional[str] = None):
    writer = _make_tb(tb_dir) if tb_dir else None
    print(_HDR)
    for step, path in step_files(d):
        s = summarize(path)
        if s:
            print(_fmt(step, s))
            _tb_write(writer, step, s)
    if writer:
        writer.close()


def run_watch(d: str, tb_dir: Optional[str] = None, interval: float = 5.0):
    """Print each step's component means as the step's dump becomes complete.

    A step file is treated as complete once a higher-numbered step file exists or the
    file size has been stable for one poll (verl writes each step's JSONL in one go from
    a background thread)."""
    writer = _make_tb(tb_dir) if tb_dir else None
    print(_HDR, flush=True)
    printed = set()
    sizes: Dict[str, int] = {}
    try:
        while True:
            files = step_files(d)
            max_step = files[-1][0] if files else -1
            for step, path in files:
                if step in printed:
                    continue
                try:
                    sz = os.path.getsize(path)
                except OSError:
                    continue
                stable = sizes.get(path) == sz
                sizes[path] = sz
                if step < max_step or stable:
                    s = summarize(path)
                    if s:
                        print(_fmt(step, s), flush=True)
                        _tb_write(writer, step, s)
                        printed.add(step)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[reward_trace] stopped.")
    finally:
        if writer:
            writer.close()


def main():
    ap = argparse.ArgumentParser(description="Per-step reward-component trace from verl rollout dumps.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="explicit run id under output/rl/ (e.g. grpov2_..._tag)")
    g.add_argument("--dir", help="explicit trainer.rollout_data_dir (contains <step>.jsonl)")
    ap.add_argument("--watch", action="store_true", help="live-tail and print each new step")
    ap.add_argument("--tb", nargs="?", const="__run_default__", default=None,
                    help="also write tensorboard scalars (bare flag -> output/rl/<run>/reward_tb)")
    ap.add_argument("--interval", type=float, default=5.0, help="watch poll seconds (default 5)")
    args = ap.parse_args()

    if args.run:
        run_dir = os.path.join(REPO_ROOT, "output", "rl", args.run)
        data_dir = os.path.join(run_dir, "rollout_data")
        tb = os.path.join(run_dir, "reward_tb") if args.tb == "__run_default__" else args.tb
    else:
        data_dir = args.dir
        tb = None if args.tb in (None, "__run_default__") else args.tb

    if args.watch:
        run_watch(data_dir, tb, args.interval)
    else:
        run_once(data_dir, tb)


if __name__ == "__main__":
    main()
