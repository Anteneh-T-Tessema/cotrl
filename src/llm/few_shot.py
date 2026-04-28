"""Few-shot example selection for the Game of 24 prompting pipeline.

Selects high-quality solved trajectories to use as in-context examples.
Good few-shot examples share two properties:
  1. Diversity — cover different number ranges and operation patterns so
     the model sees varied solution structures, not just "multiply two big numbers".
  2. Reasoning quality — the <thought> trace is step-by-step and explicit,
     not just the final answer. This teaches the model HOW to reason, not just
     WHAT the answer is.

Example selection strategy:
  - Filter to solved trajectories that contain a non-trivial <thought> block
    (length heuristic: > 40 chars inside <thought> tags).
  - Cluster by the largest number in the puzzle (low / mid / high), then
    sample one representative from each cluster — keeps coverage broad.
  - Fallback to sorted-by-response-length if clustering yields fewer than k.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..rl.trajectory import Trajectory, TrajectoryBuffer


_THOUGHT_RE = re.compile(r"<thought>(.*?)</thought>", re.DOTALL)
_MIN_THOUGHT_CHARS = 40


def _thought_length(response: str) -> int:
    match = _THOUGHT_RE.search(response)
    return len(match.group(1).strip()) if match else 0


def _number_tier(numbers: list[int]) -> str:
    """Bucket a puzzle by its largest number for diversity bucketing."""
    top = max(numbers)
    if top <= 4:
        return "low"
    if top <= 9:
        return "mid"
    return "high"


def _format_example(trajectory: Trajectory) -> dict[str, str | list[int]]:
    """Format a trajectory into the shape build_cot_prompt expects."""
    return {
        "numbers": trajectory.puzzle,
        "solution": trajectory.response,
    }


def select_few_shot_examples(
    buffer: TrajectoryBuffer,
    k: int = 3,
    min_thought_chars: int = _MIN_THOUGHT_CHARS,
) -> list[dict[str, str | list[int]]]:
    """Select k diverse, high-quality few-shot examples from the buffer.

    Args:
        buffer: A trajectory buffer containing at least some solved examples.
        k: Number of examples to return.
        min_thought_chars: Minimum character count inside <thought> tags to
            consider a trajectory as "high quality reasoning".

    Returns:
        A list of dicts with keys "numbers" and "solution", ready to pass
        to build_cot_prompt(few_shot_examples=...).
    """
    candidates = [
        t for t in buffer.successful()
        if _thought_length(t.response) >= min_thought_chars
    ]

    if not candidates:
        # Relax the quality filter — take any solved trajectory
        candidates = buffer.successful()

    if not candidates:
        return []

    # Cluster by number tier and pick the best (longest thought) from each
    tiers: dict[str, list[Trajectory]] = {"low": [], "mid": [], "high": []}
    for t in candidates:
        tiers[_number_tier(t.puzzle)].append(t)

    selected: list[Trajectory] = []
    for tier_name in ("high", "mid", "low"):
        tier = tiers[tier_name]
        if tier:
            best = max(tier, key=lambda t: _thought_length(t.response))
            selected.append(best)
        if len(selected) >= k:
            break

    # If tiers didn't fill k slots, pad with longest-thought remaining candidates
    if len(selected) < k:
        already = {id(t) for t in selected}
        extras = sorted(
            (t for t in candidates if id(t) not in already),
            key=lambda t: _thought_length(t.response),
            reverse=True,
        )
        selected.extend(extras[: k - len(selected)])

    return [_format_example(t) for t in selected[:k]]


def load_few_shot_examples(
    path: Path,
    k: int = 3,
    min_thought_chars: int = _MIN_THOUGHT_CHARS,
) -> list[dict[str, str | list[int]]]:
    """Load few-shot examples directly from a trajectory JSONL file."""
    buf = TrajectoryBuffer.load(path)
    return select_few_shot_examples(buf, k=k, min_thought_chars=min_thought_chars)


# Hardcoded seed examples used before any trajectories are available.
# These are hand-curated to show three different reasoning patterns.
SEED_EXAMPLES: list[dict[str, str | list[int]]] = [
    {
        "numbers": [2, 3, 4, 6],
        "solution": (
            "<thought>\n"
            "Try (2 + 3) * 4 = 5 * 4 = 20. Not 24.\n"
            "Try 3 * (6 + 4 - 2) = 3 * 8 = 24. Yes!\n"
            "</thought>\n"
            "<answer>3 * (6 + 4 - 2)</answer>"
        ),
    },
    {
        "numbers": [1, 5, 5, 5],
        "solution": (
            "<thought>\n"
            "Try 5 * (5 - 1/5). That uses 1/5 which isn't available.\n"
            "Try (5 - 1/5) * 5 — same issue.\n"
            "Try 5 * 5 - 1 * 5 = 25 - 5 = 20. No.\n"
            "Try (5 + 1) * (5 - 1) — only one 1 available.\n"
            "Try 5 * (5 - 1) + 4 — no 4.\n"
            "Try (1 + 5/5) * ... = 2 * ? need 12, no.\n"
            "Try (5 - 1) * 5 + 4 — no 4.\n"
            "Actually: (5 - 1/5) * 5: 1/5 needs a division of 1 by 5.\n"
            "Use 1/5: 1/5 = 0.2. 5 - 0.2 = 4.8. 4.8 * 5 = 24!\n"
            "Expression: 5 * (5 - 1/5)\n"
            "</thought>\n"
            "<answer>5 * (5 - 1/5)</answer>"
        ),
    },
    {
        "numbers": [3, 3, 8, 8],
        "solution": (
            "<thought>\n"
            "Standard operations: 3+3+8+8=22, 3*8=24 but uses only two numbers.\n"
            "Try nested fractions: 8 / (3 - 8/3).\n"
            "8/3 ≈ 2.667. 3 - 2.667 = 0.333. 8 / 0.333 = 24. Yes!\n"
            "</thought>\n"
            "<answer>8 / (3 - 8/3)</answer>"
        ),
    },
]
