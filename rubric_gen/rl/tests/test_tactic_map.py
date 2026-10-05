from rl.tactic_map import (
    category_weight_rule_errors,
    tactic_dimension_counts,
)


def _rubric(cog, aff, pro):
    return {
        "categories": [
            {"category": "Cognitive Empathy", "category_weight": cog, "criteria": []},
            {"category": "Affective Empathy", "category_weight": aff, "criteria": []},
            {"category": "Proactive Empathy", "category_weight": pro, "criteria": []},
        ]
    }


def test_counts_spec_example():
    # spec example: advice/questioning/emotional_expression/paraphrasing
    counts = tactic_dimension_counts(["advice", "questioning", "emotional_expression", "paraphrasing"])
    assert counts == {
        "Cognitive Empathy": 2,
        "Affective Empathy": 1,
        "Proactive Empathy": 1,
    }


def test_unknown_tactics_ignored():
    assert tactic_dimension_counts(["banana", "advice"]) == {
        "Cognitive Empathy": 0,
        "Affective Empathy": 0,
        "Proactive Empathy": 1,
    }


def test_rule_pass_when_primary_is_highest_and_all_categories_involved():
    # Cognitive is the unique primary category; no category is uninvolved.
    rubric = _rubric(0.5, 0.25, 0.25)
    assert category_weight_rule_errors(rubric, ["advice", "questioning", "emotional_expression", "paraphrasing"]) == []


def test_first_penalty_when_highest_weight_misses_primary():
    # Cognitive is primary, but Affective is the unique highest-weight category.
    rubric = _rubric(0.3, 0.4, 0.3)
    errs = category_weight_rule_errors(rubric, ["advice", "questioning", "emotional_expression", "paraphrasing"])
    assert len(errs) == 1 and "highest-weight" in errs[0]


def test_rule_no_constraint_without_tactic_signal():
    rubric = _rubric(0.34, 0.33, 0.33)
    assert category_weight_rule_errors(rubric, []) == []


def test_second_penalty_when_uninvolved_outweighs_involved():
    # Cognitive is primary and correctly highest, but uninvolved Proactive outweighs
    # involved Affective.
    errs = category_weight_rule_errors(_rubric(0.5, 0.1, 0.4), ["questioning", "paraphrasing", "validation"])
    assert len(errs) == 1 and "uninvolved" in errs[0]


def test_both_penalties_can_apply():
    # Uninvolved Proactive is highest, so it misses the Cognitive primary and outweighs
    # involved Affective.
    errs = category_weight_rule_errors(_rubric(0.3, 0.2, 0.5), ["questioning", "paraphrasing", "validation"])
    assert len(errs) == 2


def test_tied_primary_categories_require_tied_highest_weights():
    tactics = ["questioning", "validation"]  # Cognitive/Affective tie; Proactive absent.
    assert category_weight_rule_errors(_rubric(0.4, 0.4, 0.2), tactics) == []
