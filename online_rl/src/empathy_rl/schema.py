from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple, Union

LEGAL_CATEGORIES = ("Cognitive Empathy", "Affective Empathy", "Proactive Empathy")
CATEGORY_FIELDS = ("category", "weight_rationale", "category_weight", "criteria")
CRITERION_FIELDS = ("title", "description")


def parse_rubric(rubric: Union[str, Dict[str, Any]]) -> Tuple[Dict[str, Any] | None, str | None]:
    """Strictly parse one rubric JSON object."""
    if isinstance(rubric, dict):
        return rubric, None
    if not isinstance(rubric, str):
        return None, f"rubric must be str or dict, got {type(rubric).__name__}"
    try:
        obj = json.loads(rubric)
    except Exception as exc:  # noqa: BLE001
        return None, f"json parse error: {exc}"
    if not isinstance(obj, dict):
        return None, "rubric JSON must be an object"
    return obj, None


def validate_rubric(rubric: Union[str, Dict[str, Any]]) -> List[str]:
    """Validate against the rubric_gen-compatible rubric schema."""
    obj, err = parse_rubric(rubric)
    if err:
        return [err]
    assert obj is not None
    if isinstance(obj.get("rubric"), dict):
        obj = obj["rubric"]
    errors: List[str] = []
    categories = obj.get("categories")
    if not isinstance(categories, list) or not categories:
        return ["categories missing or empty"]

    names = []
    weight_sum = 0.0
    for idx, category in enumerate(categories):
        if not isinstance(category, dict):
            errors.append(f"categories[{idx}] must be an object")
            continue
        for field in CATEGORY_FIELDS:
            if field not in category:
                errors.append(f"categories[{idx}].{field} missing")
        name = category.get("category")
        names.append(name)
        if name not in LEGAL_CATEGORIES:
            errors.append(f"categories[{idx}] invalid category name: {name!r}")
        try:
            weight = float(category.get("category_weight"))
        except (TypeError, ValueError):
            errors.append(f"{name} category_weight not numeric")
            weight = 0.0
        if not 0.0 < weight <= 1.0:
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
            for field in CRITERION_FIELDS:
                if field not in criterion:
                    errors.append(f"{name} criteria[{jdx}].{field} missing")
            if not isinstance(criterion.get("title"), str) or not criterion.get("title", "").strip():
                errors.append(f"{name} criteria[{jdx}] empty title")
            if not isinstance(criterion.get("description"), str) or not criterion.get("description", "").strip():
                errors.append(f"{name} criteria[{jdx}] empty description")
            desc = str(criterion.get("description") or "")
            if "value: 0, 1, 2, 3, 4, 5" not in desc:
                errors.append(f"{name} criteria[{jdx}] description must end with allowed values 0-5")

    if set(names) != set(LEGAL_CATEGORIES):
        errors.append(f"category set {sorted(str(n) for n in names)} != {sorted(LEGAL_CATEGORIES)}")
    if abs(weight_sum - 1.0) > 0.02:
        errors.append(f"category_weight sum {weight_sum:.4f} != 1.0")
    return errors


def compact_json(value: Dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def normalize_rubric(rubric: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(rubric.get("rubric"), dict):
        rubric = rubric["rubric"]
    categories = []
    for category in rubric.get("categories", []):
        if not isinstance(category, dict):
            continue
        normalized = {key: category.get(key) for key in CATEGORY_FIELDS}
        criteria = []
        for criterion in category.get("criteria", []):
            if isinstance(criterion, dict):
                criteria.append({
                    "title": criterion.get("title"),
                    "description": criterion.get("description"),
                })
        normalized["criteria"] = criteria
        categories.append(normalized)
    return {"categories": categories}
