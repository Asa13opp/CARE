"""Live R_exec smoke (thinking OFF). Writes result to /tmp/rl_exec_smoke.txt."""
import json
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

OUT = "/tmp/rl_exec_smoke.txt"


def main():
    from rl.judge import reward_exec, score_many

    data_path = os.environ.get("RL_INPUT", os.path.join(PROJECT_ROOT, "data", "empathy_slim_tactic_tagged_en.jsonl"))
    rec = json.loads(open(data_path).readline())
    prompt, chosen, rejected = rec["prompt"], rec["chosen"][0]["content"], rec["rejected"][0]["content"]
    rubric = {"categories": [
        {"category": "Cognitive Empathy", "category_weight": 0.4, "criteria": [
            {"title": "Understands the situation", "description": "Accurately grasps the user's concern."},
            {"title": "Misreads the concern", "description": "Distorts or ignores what the user said."}]},
        {"category": "Affective Empathy", "category_weight": 0.3, "criteria": [
            {"title": "Validates feelings", "description": "Acknowledges the user's emotional state."}]},
        {"category": "Proactive Empathy", "category_weight": 0.3, "criteria": [
            {"title": "Offers a useful next step", "description": "Suggests a concrete, relevant action."}]},
    ]}
    t0 = time.time()
    sc, sr = score_many([(prompt, rubric, chosen), (prompt, rubric, rejected)])
    r = reward_exec(prompt, rubric, chosen, rejected)
    r_sw = reward_exec(prompt, rubric, rejected, chosen)
    dt = time.time() - t0
    lines = [
        f"s_chosen={sc}  s_rejected={sr}",
        f"R_exec={r:.4f}  R_swapped={r_sw:.4f}",
        f"antisymmetric={abs((r or 0)+(r_sw or 0))<1e-9}",
        f"in_range={sc is not None and sr is not None and 0<=sc<=5 and 0<=sr<=5 and -5<=r<=5}",
        f"elapsed_2calls={dt:.1f}s  (thinking={os.environ.get('JUDGE_THINKING','0')})",
        "DONE",
    ]
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
