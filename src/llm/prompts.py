from typing import Optional


_SYSTEM_PROMPT = """\
You are a mathematical reasoning assistant. Your task is to solve the Game of 24.

Rules:
- You are given exactly four numbers.
- Use each number exactly once.
- Combine them with +, -, *, / and parentheses.
- The expression must equal 24.

Think step by step. Show your reasoning inside <thought> tags.
Place your final expression inside <answer> tags.
"""


def build_cot_prompt(
    numbers: list[int],
    few_shot_examples: Optional[list[dict]] = None,
    use_seed_examples: bool = True,
) -> list[dict[str, str]]:
    """Build a chain-of-thought prompt for the Game of 24.

    Args:
        numbers: The four puzzle numbers.
        few_shot_examples: Solved examples to inject as few-shot turns.
            Each dict must have "numbers" (list[int]) and "solution" (str).
            If None and use_seed_examples is True, uses the curated SEED_EXAMPLES.
        use_seed_examples: When few_shot_examples is None, fall back to the
            hand-curated seed set. Set False for zero-shot ablations.

    Returns:
        A messages list compatible with the HuggingFace chat template format.
    """
    # Deferred import avoids a circular dependency (few_shot imports trajectory,
    # which has no dependency on prompts).
    if few_shot_examples is None and use_seed_examples:
        from .few_shot import SEED_EXAMPLES
        few_shot_examples = SEED_EXAMPLES

    messages: list[dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]

    for ex in (few_shot_examples or []):
        messages.append({
            "role": "user",
            "content": f"Numbers: {' '.join(str(n) for n in ex['numbers'])}",
        })
        messages.append({"role": "assistant", "content": str(ex["solution"])})

    messages.append({
        "role": "user",
        "content": f"Numbers: {' '.join(str(n) for n in numbers)}",
    })
    return messages
