"""Non-pytest offline checker that writes its own result file (robust to flaky
stdout capture on a busy login node). Run: python rl/tests/run_checks.py"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

OUT = "/tmp/rl_check_result.txt"


def main():
    import rl.reward_fn as rf
    from rl.tests.test_schema import GOOD

    T = ["advice", "questioning", "emotional_expression", "paraphrasing"]

    def extra(**kw):
        b = {"prompt": [{"role": "user", "content": "hi"}], "chosen_text": "good",
             "rejected_text": "bad", "chosen_tactics": T}
        b.update(kw)
        return b

    results = []

    os.environ["JUDGE_API_KEY"] = "x"
    rf._exec_fn = lambda p, r, c, j: 0.3
    o = rf.compute_score(solution_str=json.dumps(GOOD), extra_info=extra(),
                         lambda_trans=0.5, gamma_format=0.5, exec_coef=1.0,
                         length_penalty_max=0.0)
    expected_exec = __import__("math").tanh(0.3 / 0.7)
    results.append(("keys+sum=tanh(0.3/0.7)+1.2", abs(o["score"] - (expected_exec + 1.2)) < 1e-9 and o["r_format"] == 1.0
                    and o["r_trans"] == 1.0 and o["r_exec_raw"] == 0.3 and o["r_exec"] == expected_exec
                    and o["r_chosen_bonus"] == 0.2, o["score"]))

    os.environ.pop("JUDGE_API_KEY", None)
    rf._exec_fn = None
    o = rf.compute_score(solution_str=json.dumps(GOOD), extra_info=extra(),
                         length_penalty_max=0.0, use_judge=False)
    results.append(("no_judge=1.0", o["r_exec"] == 0.0 and abs(o["score"] - 1.0) < 1e-9, o["score"]))

    os.environ["JUDGE_API_KEY"] = "x"

    def boom(*a, **k):
        raise RuntimeError("down")

    rf._exec_fn = boom
    rf._judge_failure_guard = rf._JudgeFailureGuard()
    o = rf.compute_score(
        solution_str=json.dumps(GOOD), extra_info=extra(), judge_failure_window=32
    )
    results.append(("isolated_judge_fail_is_neutral", o["r_exec"] == 0.0, o["r_exec"]))
    os.environ.pop("JUDGE_API_KEY", None)
    rf._exec_fn = None

    o = rf.compute_score(solution_str="{bad json", extra_info=extra())
    results.append(("malformed_penalty", o["score"] == -2.0 and o.get("error") == "unparseable rubric", o["score"]))

    o = rf.compute_score(solution_str="", extra_info=extra())
    results.append(("empty_penalty", o["score"] == -2.0, o["score"]))

    bad = json.loads(json.dumps(GOOD))
    for c in bad["categories"]:
        c["category_weight"] = {"Cognitive Empathy": 0.35, "Affective Empathy": 0.45,
                                "Proactive Empathy": 0.20}[c["category"]]
    o = rf.compute_score(solution_str=json.dumps(bad), extra_info=extra(), use_judge=False)
    results.append(("trans_violation_0.5", o["r_trans"] == 0.5, o["r_trans"]))

    ok = all(r[1] for r in results)
    lines = [("PASS" if p else "FAIL") + f" {n} -> {v}" for n, p, v in results]
    lines.append("ALL_PASS" if ok else "SOME_FAILED")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
