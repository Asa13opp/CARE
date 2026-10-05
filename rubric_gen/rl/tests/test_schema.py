import copy
import json

from rl.schema import parse_rubric, validate_rubric

GOOD = {
    "categories": [
        {
            "category": "Cognitive Empathy",
            "weight_rationale": "Understanding is most important for this concern.",
            "category_weight": 0.5,
            "criteria": [
                {"title": "Understand", "description": "grasp the problem"},
                {"title": "Misread", "description": "avoid distorting it"},
            ],
        },
        {
            "category": "Affective Empathy",
            "weight_rationale": "Emotional support is relevant but secondary here.",
            "category_weight": 0.25,
            "criteria": [
                {"title": "Validate", "description": "name the feeling"},
            ],
        },
        {
            "category": "Proactive Empathy",
            "weight_rationale": "Practical next steps are useful but less central.",
            "category_weight": 0.25,
            "criteria": [
                {"title": "Act", "description": "offer a next step"},
            ],
        },
    ]
}


def test_good_rubric_passes():
    assert validate_rubric(GOOD) == []


def test_good_rubric_as_json_string_passes():
    assert validate_rubric(json.dumps(GOOD)) == []


def test_non_json_string_is_single_error():
    errs = validate_rubric("not json at all {")
    assert len(errs) == 1 and "json parse error" in errs[0]


def test_wrong_category_name():
    bad = copy.deepcopy(GOOD)
    bad["categories"][0]["category"] = "Cognative Empathy"
    errs = validate_rubric(bad)
    assert any("invalid category name" in e for e in errs)
    assert any("category set" in e for e in errs)


def test_weight_out_of_range():
    bad = copy.deepcopy(GOOD)
    bad["categories"][0]["category_weight"] = 0.0
    errs = validate_rubric(bad)
    assert any("not in (0, 1]" in e for e in errs)


def test_weight_sum_off():
    bad = copy.deepcopy(GOOD)
    bad["categories"][0]["category_weight"] = 0.9  # sum 1.4
    errs = validate_rubric(bad)
    assert any("sum" in e for e in errs)


def test_empty_description():
    bad = copy.deepcopy(GOOD)
    bad["categories"][1]["criteria"][0]["description"] = "   "
    errs = validate_rubric(bad)
    assert any("empty description" in e for e in errs)


def test_missing_weight_rationale():
    bad = copy.deepcopy(GOOD)
    bad["categories"][0].pop("weight_rationale")
    errs = validate_rubric(bad)
    assert any("weight_rationale missing or empty" in e for e in errs)


def test_missing_categories():
    errs = validate_rubric({"foo": 1})
    assert any("top-level fields missing" in error for error in errs)
    assert any("unknown top-level fields" in error for error in errs)


def test_root_must_be_object():
    for value in ("[]", '"rubric"', "1", "null"):
        obj, error = parse_rubric(value)
        assert obj is None and "root must be an object" in error


def test_nonfinite_json_number_is_rejected():
    obj, error = parse_rubric('{"categories": NaN}')
    assert obj is None and "non-finite JSON number" in error


def test_requires_exactly_three_unique_categories():
    bad = copy.deepcopy(GOOD)
    bad["categories"].append(copy.deepcopy(bad["categories"][0]))
    errs = validate_rubric(bad)
    assert any("exactly 3" in error for error in errs)
    assert any("category names must be unique" in error for error in errs)


def test_rejects_unknown_fields_at_every_level():
    bad = copy.deepcopy(GOOD)
    bad["padding"] = "x"
    bad["categories"][0]["extra"] = 1
    bad["categories"][0]["criteria"][0]["extra"] = 1
    errs = validate_rubric(bad)
    assert sum("unknown" in error for error in errs) == 3


def test_weight_must_be_finite_numeric_not_string_or_bool():
    for value in ("0.5", True, float("nan"), float("inf")):
        bad = copy.deepcopy(GOOD)
        bad["categories"][0]["category_weight"] = value
        errs = validate_rubric(bad)
        assert any("finite" in error for error in errs)


def test_criterion_titles_must_be_globally_unique():
    bad = copy.deepcopy(GOOD)
    bad["categories"][2]["criteria"][0]["title"] = "Understand"
    assert any("duplicate criterion title" in error for error in validate_rubric(bad))


def test_rejects_leading_or_trailing_junk_strictly():
    # STRICT format: a stray token before the JSON (or trailing text) is a format
    # violation, NOT silently repaired. R_format must penalize it so the policy learns
    # to emit clean JSON.
    assert validate_rubric("aget\n" + json.dumps(GOOD)) != []
    assert validate_rubric(json.dumps(GOOD) + "\ntrailing commentary") != []
    # surrounding whitespace is fine
    assert validate_rubric("  " + json.dumps(GOOD) + "\n") == []
