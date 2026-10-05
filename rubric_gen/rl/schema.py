"""Rubric format validation for R_format.

Aligned with the project's Stage-1 schema (`scripts/common.py` +
`scripts/eval_predictions.py:schema_ok`), NOT the stricter Stage-1 *labeling* rules
(2-4 criteria, >=12-word descriptions), so the SFT model's natural outputs are not
over-penalized. Returns a list of human-readable error strings (empty == fully valid).
"""

import json
import math
from typing import Any, Dict, List, Union

from rl.tactic_map import CATEGORY_NAMES

EXPECTED_CATEGORIES = set(CATEGORY_NAMES)
WEIGHT_SUM_TOL = 0.02
TOP_LEVEL_FIELDS = {"categories"}
CATEGORY_FIELDS = {"category", "weight_rationale", "category_weight", "criteria"}
CRITERION_FIELDS = {"title", "description"}


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is not allowed")


def parse_rubric(rubric: Union[str, Dict[str, Any]]):
    """Return (obj, error). STRICT: the string must be exactly one JSON object
    (surrounding whitespace allowed). Any leading/trailing junk the policy emits is a
    format violation and is NOT silently extracted/repaired — R_format must penalize it
    so the model learns to output clean JSON. If already a dict, pass through."""
    if isinstance(rubric, dict):
        return rubric, None
    if isinstance(rubric, str):
        try:
            obj = json.loads(rubric, parse_constant=_reject_nonfinite_json)
        except Exception as exc:  # noqa: BLE001 - any deviation is a format violation
            return None, f"json parse error: {exc}"
        if not isinstance(obj, dict):
            return None, f"rubric root must be an object, got {type(obj).__name__}"
        return obj, None
    return None, f"rubric must be str or dict, got {type(rubric).__name__}"


def validate_rubric(rubric: Union[str, Dict[str, Any]]) -> List[str]:
    obj, parse_err = parse_rubric(rubric)
    if parse_err is not None:
        return [parse_err]

    errors: List[str] = []
    obj_fields = set(obj)
    if obj_fields != TOP_LEVEL_FIELDS:
        missing = sorted(TOP_LEVEL_FIELDS - obj_fields)
        extra = sorted(obj_fields - TOP_LEVEL_FIELDS)
        if missing:
            errors.append(f"top-level fields missing: {missing}")
        if extra:
            errors.append(f"unknown top-level fields: {extra}")

    categories = obj.get("categories")
    if not isinstance(categories, list):
        errors.append("categories must be a list")
        return errors
    if len(categories) != len(CATEGORY_NAMES):
        errors.append(f"categories must contain exactly {len(CATEGORY_NAMES)} items")

    names = []
    weight_sum = 0.0
    all_titles = set()
    for idx, category in enumerate(categories):
        if not isinstance(category, dict):
            errors.append(f"categories[{idx}] must be an object")
            continue
        fields = set(category)
        missing = sorted(CATEGORY_FIELDS - fields)
        extra = sorted(fields - CATEGORY_FIELDS)
        if missing:
            errors.append(f"categories[{idx}] fields missing: {missing}")
        if extra:
            errors.append(f"categories[{idx}] unknown fields: {extra}")
        name = category.get("category")
        names.append(name)
        if name not in EXPECTED_CATEGORIES:
            errors.append(f"categories[{idx}] invalid category name: {name!r}")
        rationale = category.get("weight_rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            errors.append(f"{name} weight_rationale missing or empty")
        raw_weight = category.get("category_weight")
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
            errors.append(f"{name} category_weight must be a finite number")
            weight = 0.0
        else:
            weight = float(raw_weight)
        if not math.isfinite(weight):
            errors.append(f"{name} category_weight must be finite")
            weight = 0.0
        elif not 0.0 < weight <= 1.0:
            errors.append(f"{name} category_weight {weight} not in (0, 1]")
        weight_sum += weight

        criteria = category.get("criteria")
        if not isinstance(criteria, list) or not criteria:
            errors.append(f"{name} criteria missing or empty")
            continue
        for jdx, criterion in enumerate(criteria):
            if not isinstance(criterion, dict):
                errors.append(f"{name} criteria[{jdx}] must be an object")
                continue
            fields = set(criterion)
            missing = sorted(CRITERION_FIELDS - fields)
            extra = sorted(fields - CRITERION_FIELDS)
            if missing:
                errors.append(f"{name} criteria[{jdx}] fields missing: {missing}")
            if extra:
                errors.append(f"{name} criteria[{jdx}] unknown fields: {extra}")
            title = criterion.get("title")
            description = criterion.get("description")
            if not isinstance(title, str) or not title.strip():
                errors.append(f"{name} criteria[{jdx}] empty title")
            elif title.strip() in all_titles:
                errors.append(f"duplicate criterion title: {title.strip()!r}")
            else:
                all_titles.add(title.strip())
            if not isinstance(description, str) or not description.strip():
                errors.append(f"{name} criteria[{jdx}] empty description")

    if len(names) != len(set(names)):
        errors.append("category names must be unique")
    if set(names) != EXPECTED_CATEGORIES:
        errors.append(f"category set {sorted(str(n) for n in names)} != {sorted(EXPECTED_CATEGORIES)}")
    if abs(weight_sum - 1.0) > WEIGHT_SUM_TOL:
        errors.append(f"category_weight sum {weight_sum:.4f} != 1.0")
    return errors
