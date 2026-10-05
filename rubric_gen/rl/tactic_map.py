"""Tactic -> empathy-dimension -> category mappings and the category-weight rule.

Ported from the Stage-1 rubric-labeling reference. Signatures here take a plain
``tactics: list[str]`` (e.g. a response's ``all_tactics``) rather than a turn dict,
so they apply directly to the RL preference data.
"""

from typing import Any, Dict, List

CATEGORY_NAMES = ("Cognitive Empathy", "Affective Empathy", "Proactive Empathy")

EMPATHY_TO_CATEGORY = {
    "cognitive": "Cognitive Empathy",
    "affective": "Affective Empathy",
    "proactive": "Proactive Empathy",
}

# Maps a Mint supporter tactic to its empathy dimension.
MINT_TACTIC_TO_DIMENSION = {
    "questioning": "cognitive",
    "paraphrasing": "cognitive",
    "contextualizing": "cognitive",
    "reappraisal": "cognitive",
    "validation": "affective",
    "emotional_expression": "affective",
    "terms_of_endearment": "affective",
    "self_disclosure": "affective",
    "solidarity": "affective",
    "gratitude": "affective",
    "spirituality": "affective",
    "advice": "proactive",
    "information": "proactive",
    "empowerment": "proactive",
    "assistance": "proactive",
}


def _as_tactic_list(tactics: Any) -> List[str]:
    if isinstance(tactics, str):
        return [tactics]
    if isinstance(tactics, list):
        return [t for t in tactics if isinstance(t, str)]
    return []


def tactic_dimension_counts(tactics: Any) -> Dict[str, int]:
    """Count tactics per empathy category via MINT_TACTIC_TO_DIMENSION.

    Tactics not in the table are ignored.
    """
    counts = {category: 0 for category in CATEGORY_NAMES}
    for tactic in _as_tactic_list(tactics):
        dimension = MINT_TACTIC_TO_DIMENSION.get(tactic.lower().strip())
        category = EMPATHY_TO_CATEGORY.get(dimension) if dimension else None
        if category:
            counts[category] += 1
    return counts


def category_weight_rule_errors(rubric: Dict[str, Any], tactics: Any) -> List[str]:
    """Return violations for the two category-weight penalties only.

    1. The rubric's highest-weight category set must equal the chosen response's primary
       empathy category set (the categories with the most mapped tactics).
    2. A category absent from the chosen response (zero mapped tactics) may not have a
       weight greater than any category that is present.

    Tied tactic counts and tied rubric maxima are treated as sets, so this remains
    deterministic without arbitrarily choosing one "primary" category. No mapped tactic
    means no transition prior and therefore no violation.
    """
    counts = tactic_dimension_counts(tactics)
    if sum(counts.values()) == 0:
        return []
    weights = {
        category.get("category"): float(category.get("category_weight", 0.0))
        for category in rubric.get("categories", [])
        if isinstance(category, dict) and category.get("category") in CATEGORY_NAMES
    }
    snapshot = {c: weights.get(c, 0.0) for c in CATEGORY_NAMES}
    errors = []
    primary_count = max(counts.values())
    primary = {category for category, count in counts.items() if count == primary_count}
    highest_weight = max(snapshot.values())
    highest = {category for category, weight in snapshot.items() if weight == highest_weight}
    if highest != primary:
        errors.append(
            f"highest-weight categories {sorted(highest)} must match chosen primary "
            f"categories {sorted(primary)}; tactic_counts={counts}, weights={snapshot}"
        )

    involved = [category for category, count in counts.items() if count > 0]
    uninvolved = [category for category, count in counts.items() if count == 0]
    if uninvolved and max(snapshot[category] for category in uninvolved) > min(snapshot[category] for category in involved):
        errors.append(
            f"chosen-uninvolved categories {uninvolved} cannot outweigh any involved category "
            f"{involved}; tactic_counts={counts}, weights={snapshot}"
        )
    return errors
