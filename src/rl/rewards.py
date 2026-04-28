"""Shaped reward functions for the Game of 24 RL loop.

Binary reward (0 or 1) works but is sparse — the model gets no signal
for responses that are structurally correct but arithmetically wrong.
Shaped rewards add partial credit that accelerates early learning:

  format_reward   +0.15  <thought> and <answer> tags both present
  numbers_reward  +0.25  correct four numbers used (wrong result)
  solve_reward    +1.00  expression evaluates to exactly 24

These are additive and capped at 1.0. The breakdown is intentional:
format < numbers < solve ensures the model cannot game a high score by
producing well-formatted wrong answers — it must actually solve the puzzle
to break past ~0.40.

For GRPO the reward is normalised within each group, so the absolute scale
matters less than the ordering. The shaped signal gives the optimiser more
gradient signal early in training, then converges to the same binary target.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..verifier.core import verify_solution, extract_expression


_THOUGHT_RE = re.compile(r"<thought>.*?</thought>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>.*?</answer>", re.DOTALL)

FORMAT_REWARD = 0.15
NUMBERS_REWARD = 0.25
SOLVE_REWARD = 1.00


@dataclass(frozen=True)
class ShapedReward:
    total: float
    format_component: float   # tag structure present
    numbers_component: float  # correct numbers used, wrong answer
    solve_component: float    # fully correct
    solved: bool
    expression: str | None
    detail: str


def compute_reward(response: str, puzzle: list[int]) -> ShapedReward:
    """Compute the shaped reward for a single model response.

    Args:
        response: Raw model output (may contain <thought> and <answer> tags).
        puzzle:   The four input numbers for this puzzle.

    Returns:
        A ShapedReward with per-component breakdown and a total capped at 1.0.
    """
    format_component = 0.0
    numbers_component = 0.0
    solve_component = 0.0

    # Format reward: both tags must be present and non-empty
    has_thought = bool(_THOUGHT_RE.search(response))
    has_answer = bool(_ANSWER_RE.search(response))
    if has_thought and has_answer:
        format_component = FORMAT_REWARD

    expression = extract_expression(response) or ""
    signal = verify_solution(expression, puzzle)

    if signal.solved:
        solve_component = SOLVE_REWARD
        detail = "solved"
    elif expression and signal.error and "mismatch" not in signal.error:
        # Expression parsed and evaluated, used right numbers, wrong result
        numbers_component = NUMBERS_REWARD
        detail = f"wrong result: {signal.error}"
    elif expression:
        detail = signal.error or "invalid expression"
    else:
        detail = "no answer extracted"

    total = min(format_component + numbers_component + solve_component, 1.0)

    return ShapedReward(
        total=total,
        format_component=format_component,
        numbers_component=numbers_component,
        solve_component=solve_component,
        solved=signal.solved,
        expression=signal.expression,
        detail=detail,
    )


def compute_batch_rewards(
    responses: list[str],
    puzzles: list[list[int]],
) -> list[ShapedReward]:
    """Compute shaped rewards for a batch of (response, puzzle) pairs."""
    assert len(responses) == len(puzzles), "responses and puzzles must be same length"
    return [compute_reward(r, p) for r, p in zip(responses, puzzles)]
