from __future__ import annotations

from typing import Any, Dict, Iterable, List

from empathy_rl.schema import LEGAL_CATEGORIES

SYSTEM_PROMPT = (
    "You are a rubric generator. Given the dialogue history and the current user query, "
    "output one JSON object for evaluating the next assistant response. Output JSON only, "
    "with no markdown, no explanation, no <think> block, and no repeated JSON object.\n"
    'Schema: {"categories":[{"category","weight_rationale","category_weight",'
    '"criteria":[{"title","description"}]}]}\n'
    '- "category" must be exactly one of: ' + ", ".join(LEGAL_CATEGORIES) + ".\n"
    '- Include all three categories exactly once.\n'
    '- "category_weight" values must sum to 1.0.\n'
    '- "weight_rationale" is one concise sentence tying the current user need to the weight.\n'
    '- Each criterion description must contain six concrete scoring anchors followed by '
    '"value: 0, 1, 2, 3, 4, 5". Use this format: '
    '"[0] <absent or harmful> [1] <very weak> [2] <partial> [3] <adequate> '
    '[4] <strong> [5] <excellent and turn-specific> value: 0, 1, 2, 3, 4, 5".\n'
    '- Do not output any metadata beyond the rubric categories.\n'
    "- Escape any double quotes inside string values."
)


def render_history(history: Iterable[Dict[str, Any]]) -> str:
    lines = []
    for turn in history:
        role = str(turn.get("role", "")).strip() or "unknown"
        content = str(turn.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def make_rubric_messages(history: List[Dict[str, Any]], query: str | None = None) -> List[Dict[str, str]]:
    context = render_history(history)
    if query:
        context = f"{context}\ncurrent_user_query: {query}".strip()
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context},
    ]


def apply_chat_template(tokenizer: Any, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )
