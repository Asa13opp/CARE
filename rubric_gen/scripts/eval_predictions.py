import argparse
import json
from pathlib import Path

from common import validate_rubric


EXPECTED = {"Cognitive Empathy", "Affective Empathy", "Proactive Empathy"}


def schema_ok(rubric):
    ok, errors = validate_rubric(rubric)
    if not ok:
        return False, errors
    names = {cat.get("category") for cat in rubric["categories"]}
    if names != EXPECTED:
        errors.append(f"category names {sorted(names)}")
    weights = []
    for category in rubric["categories"]:
        rationale = category.get("weight_rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            errors.append(f"{category.get('category', '?')} weight_rationale missing or empty")
        try:
            weight = float(category.get("category_weight"))
        except (TypeError, ValueError):
            errors.append(f"{category.get('category', '?')} category_weight not numeric")
            continue
        if not 0.0 < weight <= 1.0:
            errors.append(f"{category.get('category', '?')} category_weight {weight} not in (0, 1]")
        weights.append(weight)
    weight_sum = sum(weights)
    if abs(weight_sum - 1.0) > 0.02:
        errors.append(f"category weight sum {weight_sum:.4f}")
    return not errors, errors


def main():
    parser = argparse.ArgumentParser()
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--pred", default=str(project_root / "output" / "eval" / "sft" / "predictions.jsonl"))
    parser.add_argument("--report", default=str(project_root / "output" / "eval" / "sft" / "structural_report.md"))
    args = parser.parse_args()

    total = json_ok = schema_pass = 0
    failures = []
    weight_abs_errors = []
    criteria_count_abs_errors = []
    with open(args.pred, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            row = json.loads(line)
            try:
                pred = json.loads(row["prediction"])
                json_ok += 1
            except Exception as exc:
                failures.append((row["id"], "json", str(exc), row["prediction"][:500]))
                continue
            ok, errors = schema_ok(pred)
            if ok:
                schema_pass += 1
            else:
                failures.append((row["id"], "schema", "; ".join(errors), row["prediction"][:500]))
            gold = json.loads(row["target"])
            def numeric_weights(categories):
                weights = {}
                for category in categories:
                    try:
                        weights[category.get("category")] = float(category.get("category_weight"))
                    except (AttributeError, TypeError, ValueError):
                        continue
                return weights

            pred_weights = numeric_weights(pred.get("categories", []))
            gold_weights = numeric_weights(gold.get("categories", []))
            for name in EXPECTED:
                if name in pred_weights and name in gold_weights:
                    weight_abs_errors.append(abs(pred_weights[name] - gold_weights[name]))
            pred_n_criteria = sum(len(c.get("criteria", [])) for c in pred.get("categories", []))
            gold_n_criteria = sum(len(c.get("criteria", [])) for c in gold.get("categories", []))
            criteria_count_abs_errors.append(abs(pred_n_criteria - gold_n_criteria))

    report = []
    report.append("# Rubric Generator Evaluation")
    report.append("")
    report.append(f"- total: {total}")
    report.append(f"- JSON legal rate: {json_ok / total if total else 0:.4f} ({json_ok}/{total})")
    report.append(f"- schema pass rate: {schema_pass / total if total else 0:.4f} ({schema_pass}/{total})")
    report.append(f"- category weight MAE: {sum(weight_abs_errors) / len(weight_abs_errors) if weight_abs_errors else 0:.4f}")
    report.append(f"- criteria count MAE: {sum(criteria_count_abs_errors) / len(criteria_count_abs_errors) if criteria_count_abs_errors else 0:.4f}")
    report.append("")

    report.append("## Failures")
    for item in failures[:20]:
        report.append(f"- `{item[0]}` {item[1]}: {item[2]}")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
