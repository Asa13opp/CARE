from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Override MODEL_PATH with a local model directory when offline access is required.
MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen3-8B")
# Override DATA_PATH with the input JSONL path when it is stored outside data/.
DATA_PATH = os.environ.get("DATA_PATH", str(PROJECT_ROOT / "data" / "Mint_rubric_test.jsonl"))

LEGAL_CATEGORIES = ("Cognitive Empathy", "Affective Empathy", "Proactive Empathy")

SYSTEM_PROMPT = (
    "You are a rubric generator. Given the dialogue history up to the seeker's latest turn, "
    "output a JSON rubric evaluating the upcoming supporter response. Output JSON only, no explanation.\n"
    'Schema: {"categories":[{"category","weight_rationale","category_weight",'
    '"criteria":[{"title","description"}]}]}\n'
    '- "category" must be exactly one of: ' + ", ".join(LEGAL_CATEGORIES) + ".\n"
    '- "weight_rationale" is one concise sentence, written BEFORE category_weight, '
    "tying the seeker's latest-turn need and the supporter's empathy tactics to this "
    'category\'s weight (state whether it is the highest, lowest, or tied).\n'
    "- Escape any double quotes appearing inside string values."
)

CATEGORY_FIELDS = ("category", "weight_rationale", "category_weight", "criteria")
CRITERION_FIELDS = ("title", "description")


def has_nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() not in {"", "NaN", "nan", "{{silence}}"}
    if isinstance(value, list):
        return any(has_nonempty(item) for item in value)
    return True


def normalize_rubric(rubric: Dict[str, Any]) -> Dict[str, Any]:
    categories = []
    for category in rubric.get("categories", []):
        criteria = []
        for criterion in category.get("criteria", []):
            criteria.append({key: criterion.get(key) for key in CRITERION_FIELDS})
        normalized_category = {key: category.get(key) for key in CATEGORY_FIELDS}
        normalized_category["criteria"] = criteria
        categories.append(normalized_category)

    return {"categories": categories}


def compact_json(value: Dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def validate_rubric(rubric: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors = []
    categories = rubric.get("categories")
    if not isinstance(categories, list) or not categories:
        errors.append("categories missing or empty")
    else:
        for idx, category in enumerate(categories):
            for key in CATEGORY_FIELDS:
                if key not in category:
                    errors.append(f"categories[{idx}].{key} missing")
            criteria = category.get("criteria")
            if not isinstance(criteria, list) or not criteria:
                errors.append(f"categories[{idx}].criteria missing or empty")
            else:
                for jdx, criterion in enumerate(criteria):
                    for key in CRITERION_FIELDS:
                        if key not in criterion:
                            errors.append(f"categories[{idx}].criteria[{jdx}].{key} missing")
    return not errors, errors


def render_dialog_context(turns: Iterable[Dict[str, Any]]) -> str:
    lines = []
    for turn in turns:
        role = turn.get("role", "")
        content = str(turn.get("content", "")).strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def make_messages(context: str, target: str | None = None) -> List[Dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context},
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": target})
    return messages


def apply_chat_template(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )


def encode_completion_only(tokenizer, messages: List[Dict[str, str]], max_seq_len: int):
    prompt_messages = messages[:2]
    full_text = apply_chat_template(tokenizer, messages, add_generation_prompt=False)
    prompt_text = apply_chat_template(tokenizer, prompt_messages, add_generation_prompt=True)
    target_text = messages[2]["content"]

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        target_ids = target_ids + [eos_id]

    if len(target_ids) >= max_seq_len:
        raise ValueError(f"target length {len(target_ids)} >= max_seq_len {max_seq_len}")

    keep_prompt = max_seq_len - len(target_ids)
    truncated = len(prompt_ids) > keep_prompt
    input_ids = prompt_ids[-keep_prompt:] + target_ids
    labels = [-100] * min(len(prompt_ids), keep_prompt) + target_ids
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "truncated": truncated,
        "full_text": full_text,
    }
