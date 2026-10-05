"""Offline preference eval — the metric the RL is actually trained for.

The structural eval (scripts/eval_predictions.py) reports JSON-legal / schema-pass / weight
& criteria-count MAE vs gold, but NOT whether a generated rubric separates a known-better
response from a worse one. This does: for each val preference pair, generate a rubric from
the model, then have the judge score the chosen and rejected responses WITH that rubric, and
report win-rate (s_chosen > s_rejected), mean/median margin, plus judge- and format-failure
rates. Run it next to the structural report for a fuller picture.

Usage (verl env; source configs/rl/judge.env first):
  python -m rl.preference_eval --run grpov2_..._exec2.0         # a specific run by id
  python -m rl.preference_eval --model output/sft/<run_id>/checkpoints \
      --out output/eval/sft/preference_report.md               # explicit model still works
  python -m rl.preference_eval --base-url http://host:8000/v1  # served vLLM model

--run <id> resolves the model to output/rl/<id>/hf and defaults --out to
output/eval/<id>/preference_report.md.

The aggregation (`summarize_preferences`) is pure and unit-tested; generation + judging run
only on the box.
"""

import argparse
import json
import math
import os
import statistics as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# repo root = two levels up from this file (rl/preference_eval.py -> rubric_gen/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from tqdm import tqdm

# ---------- pure aggregation (offline-testable) ----------

def summarize_preferences(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """samples: per-pair dicts with keys
         parsed (bool): the model's rubric parsed as valid JSON,
         judged (bool): both chosen & rejected were scored by the judge,
         margin (float|None): s_chosen - s_rejected (only when judged).
    Returns win-rate / margins over the judged subset, plus failure rates."""
    n = len(samples)
    parsed = [s for s in samples if s.get("parsed")]
    judged = [s for s in parsed if s.get("judged")]
    margins = [float(s["margin"]) for s in judged if s.get("margin") is not None]
    wins = sum(1 for m in margins if m > 0)
    ties = sum(1 for m in margins if m == 0)
    losses = sum(1 for m in margins if m < 0)
    nan = float("nan")
    return {
        "n": n,
        "n_parsed": len(parsed),
        "n_scored": len(judged),
        "format_fail_rate": (n - len(parsed)) / n if n else nan,
        "judge_fail_rate": (len(parsed) - len(judged)) / len(parsed) if parsed else nan,
        "win_rate": wins / len(margins) if margins else nan,
        "tie_rate": ties / len(margins) if margins else nan,
        "loss_rate": losses / len(margins) if margins else nan,
        "mean_margin": st.mean(margins) if margins else nan,
        "median_margin": st.median(margins) if margins else nan,
    }


def render_report(title: str, summary: Dict[str, Any]) -> str:
    def pct(x):
        return "n/a" if x != x else f"{100*x:.1f}%"  # x!=x catches NaN
    def num(x):
        return "n/a" if x != x else f"{x:.3f}"
    return (
        f"# Preference eval — {title}\n\n"
        f"- pairs: {summary['n']}  (parsed {summary['n_parsed']}, scored {summary['n_scored']})\n"
        f"- **win-rate** (chosen > rejected): {pct(summary['win_rate'])}\n"
        f"- tie-rate: {pct(summary['tie_rate'])}   loss-rate: {pct(summary['loss_rate'])}\n"
        f"- **mean margin**: {num(summary['mean_margin'])}   median margin: {num(summary['median_margin'])}\n"
        f"- format-fail (rubric unparseable): {pct(summary['format_fail_rate'])}\n"
        f"- judge-fail (among parsed): {pct(summary['judge_fail_rate'])}\n"
    )


# ---------- data loading ----------

def load_val_pairs(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Read the RL val parquet -> list of {prompt(messages), history(turns), chosen, rejected}."""
    import pandas as pd  # box only

    df = pd.read_parquet(path)
    if limit:
        df = df.head(limit)
    rows = []
    for _, r in df.iterrows():
        prompt = r["prompt"]
        prompt = prompt.tolist() if hasattr(prompt, "tolist") else prompt
        ei = r["extra_info"]
        ei = dict(ei) if not isinstance(ei, dict) else ei
        hist = ei.get("prompt")
        hist = hist.tolist() if hasattr(hist, "tolist") else (hist or [])
        rows.append({
            "prompt": [dict(m) for m in prompt],
            "history": [dict(t) for t in hist],
            "chosen": ei.get("chosen_text") or "",
            "rejected": ei.get("rejected_text") or "",
        })
    return rows


# ---------- generation (box only) ----------

def make_local_generator(model_path: str, max_new_tokens: int):
    import torch  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True)
    model.eval()
    def generate(prompts: List[List[Dict[str, str]]]) -> List[str]:
        out = []
        for messages in tqdm(prompts, desc="Generating rubrics"):
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = tok(text, return_tensors="pt").to(model.device)
            import torch
            with torch.no_grad():
                ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            gen = ids[0][inputs["input_ids"].shape[1]:]
            out.append(tok.decode(gen, skip_special_tokens=True).strip())
        return out

    return generate


def make_api_generator(base_url: str, served_model: Optional[str], api_key: str, max_new_tokens: int, concurrency: int):
    """Return a greedy OpenAI-compatible generator and the resolved served model name."""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
    if served_model is None:
        models = client.models.list().data
        if not models:
            raise RuntimeError(f"No models returned by {base_url}; pass --served-model explicitly")
        served_model = models[0].id

    def call(index: int, messages: List[Dict[str, str]]):
        response = client.chat.completions.create(
            model=served_model,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError(f"Empty completion for preference pair {index}")
        return index, content.strip()

    def generate(prompts: List[List[Dict[str, str]]]) -> List[str]:
        results: List[Optional[str]] = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = [pool.submit(call, index, prompt) for index, prompt in enumerate(prompts)]
            completed = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Generating rubrics",
                unit="pair",
                dynamic_ncols=True,
            )
            for future in completed:
                index, content = future.result()
                results[index] = content
        return [result if result is not None else "" for result in results]

    return generate, served_model


# ---------- main eval loop (box only) ----------

def evaluate(generate, val_path: str, limit: Optional[int]):
    from rl.judge import score_many
    from rl.schema import parse_rubric, validate_rubric

    pairs = load_val_pairs(val_path, limit)
    rubrics = generate([p["prompt"] for p in pairs])

    # judge requests for the parsed ones: (history, rubric_obj, chosen) and (..., rejected)
    samples: List[Dict[str, Any]] = []
    judge_reqs = []
    req_index = []  # maps each pair to its (chosen_idx, rejected_idx) in judge_reqs, or None
    parsed_rubrics = []
    schema_errors = []
    for p, rub in tqdm(zip(pairs, rubrics), desc="Parsing rubrics", total=len(pairs)):
        obj, err = parse_rubric(rub)
        errors = [err] if err is not None else validate_rubric(obj)
        parsed_rubrics.append(obj)
        schema_errors.append(errors)
        if errors:
            samples.append({"parsed": False, "judged": False, "margin": None})
            req_index.append(None)
            continue
        samples.append({"parsed": True, "judged": False, "margin": None})
        ci = len(judge_reqs); judge_reqs.append((p["history"], obj, p["chosen"]))
        ri = len(judge_reqs); judge_reqs.append((p["history"], obj, p["rejected"]))
        req_index.append((ci, ri))

    scores = score_many(judge_reqs, show_progress=True) if judge_reqs else []
    records = []
    counts = {"chosen_win": 0, "tie": 0, "rejected_win": 0}
    for pair_index, (pair, rubric, sample, idx) in enumerate(zip(pairs, parsed_rubrics, samples, req_index)):
        score_chosen = score_rejected = margin = None
        winner = None
        if idx is None:
            records.append({
                "pair_index": pair_index,
                "prompt_messages": pair["prompt"],
                "history": pair["history"],
                "chosen_text": pair["chosen"],
                "rejected_text": pair["rejected"],
                "rubric_raw": rubrics[pair_index],
                "rubric_parsed": rubric,
                "rubric_parsed_ok": False,
                "rubric_schema_errors": schema_errors[pair_index],
                "score_chosen": score_chosen,
                "score_rejected": score_rejected,
                "margin": margin,
                "winner": winner,
                "judged": False,
            })
            continue
        score_chosen, score_rejected = scores[idx[0]], scores[idx[1]]
        if score_chosen is not None and score_rejected is not None:
            score_chosen, score_rejected = float(score_chosen), float(score_rejected)
            margin = score_chosen - score_rejected
            sample["judged"] = True
            sample["margin"] = margin
            if margin > 0:
                winner = "chosen"
                counts["chosen_win"] += 1
            elif margin < 0:
                winner = "rejected"
                counts["rejected_win"] += 1
            else:
                winner = "tie"
                counts["tie"] += 1
        records.append({
            "pair_index": pair_index,
            "prompt_messages": pair["prompt"],
            "history": pair["history"],
            "chosen_text": pair["chosen"],
            "rejected_text": pair["rejected"],
            "rubric_raw": rubrics[pair_index],
            "rubric_parsed": rubric,
            "rubric_parsed_ok": not schema_errors[pair_index],
            "rubric_schema_errors": schema_errors[pair_index],
            "score_chosen": score_chosen,
            "score_rejected": score_rejected,
            "margin": margin,
            "winner": winner,
            "judged": sample["judged"],
        })
    summary = summarize_preferences(samples)
    summary["counts"] = counts
    summary["val"] = val_path
    summary["judge_model"] = os.environ.get("JUDGE_MODEL")
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    return summary, records


def main():
    ap = argparse.ArgumentParser(description="Offline preference eval (win-rate / margin) for a rubric model.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="explicit run id under output/rl/ (e.g. grpov2_..._tag); uses its hf/")
    g.add_argument("--model", help="explicit HF checkpoint dir (e.g. output/sft/<run_id>/checkpoints)")
    g.add_argument("--base-url", help="OpenAI-compatible vLLM endpoint; discovers its first served model by default")
    ap.add_argument("--served-model", help="override the model name sent to the OpenAI-compatible endpoint")
    ap.add_argument("--api-key", default="EMPTY", help="API key for the served endpoint (default: EMPTY)")
    ap.add_argument("--concurrency", type=int, default=16, help="parallel generation requests for --base-url")
    ap.add_argument("--val", default="data/rl/val.parquet", help="RL val parquet with chosen/rejected")
    ap.add_argument("--out", default=None, help="write the markdown report here (also printed)")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N pairs")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    args = ap.parse_args()

    if args.run:
        model_path = os.path.join(REPO_ROOT, "output", "rl", args.run, "hf")
        out_path = args.out or os.path.join(REPO_ROOT, "output", "eval", args.run, "preference_report.md")
        title = args.run
        generate = make_local_generator(model_path, args.max_new_tokens)
    elif args.model:
        model_path = args.model
        out_path = args.out
        title = os.path.basename(args.model.rstrip("/"))
        generate = make_local_generator(model_path, args.max_new_tokens)
    else:
        generate, served_model = make_api_generator(
            args.base_url, args.served_model, args.api_key, args.max_new_tokens, args.concurrency
        )
        out_path = args.out
        title = f"api:{served_model}"

    summary, records = evaluate(generate, args.val, args.limit)
    report = render_report(title, summary)
    counts = summary["counts"]
    report += (
        f"\n- judged counts: chosen-win {counts['chosen_win']}, tie {counts['tie']}, "
        f"rejected-win {counts['rejected_win']}\n"
        f"- judge: {summary['judge_model']}\n"
        f"- per-pair records: {os.path.join(os.path.dirname(out_path) or '.', 'pair_records.jsonl')}\n"
    )
    print(report)
    print("raw:", json.dumps({k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in summary.items()}))
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            f.write(report)
        records_path = os.path.join(os.path.dirname(out_path) or ".", "pair_records.jsonl")
        with open(records_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        summary_path = os.path.join(os.path.dirname(out_path) or ".", "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {key: (None if isinstance(value, float) and math.isnan(value) else value) for key, value in summary.items()},
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
