"""LLM-backed rollout policy for MCTS.

Bridges the LLMGenerator (which works on full puzzles) into the MCTS rollout
interface (which works on intermediate ExprPair states).

How it works:
  1. At rollout time, the MCTS node holds a list of ExprPairs — intermediate
     values produced by earlier tree actions (e.g. [ExprPair(2, "(8-6)"),
     ExprPair(2, "2"), ExprPair(3, "3")]).
  2. We show the LLM the current NUMERIC VALUES as a new mini-puzzle:
     "Combine 2, 2, 3 to reach 24."
  3. The LLM generates an expression over those values.
  4. We substitute each numeric token back to its originating ExprPair.expr,
     producing a full parenthesised expression over the original numbers.

This substitution step is the key: the MCTS tree has already locked in the
early actions; the LLM only needs to complete the remaining steps.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import TYPE_CHECKING

from .mcts import ExprPair, RolloutPolicy
from ..verifier.core import extract_expression

if TYPE_CHECKING:
    from ..llm.generator import LLMGenerator


_ROLLOUT_SYSTEM = """\
You are a mathematical reasoning assistant completing a Game of 24 puzzle.

You are given a set of intermediate values (already computed from the original numbers).
Combine ALL of them, each exactly once, using +, -, *, / and parentheses, to reach 24.

Show your reasoning inside <thought> tags.
Put ONLY the final expression inside <answer> tags — use the exact values shown, no others.

Example:
Values: 2, 3, 4
<thought>2 * 3 = 6. 6 * 4 = 24.</thought>
<answer>2 * 3 * 4</answer>
"""


def _format_value(v: Fraction) -> str:
    """Display a Fraction as a readable number for the LLM prompt."""
    if v.denominator == 1:
        return str(int(v))
    return f"{int(v.numerator)}/{int(v.denominator)}"


def _build_rollout_prompt(pairs: list[ExprPair]) -> list[dict[str, str]]:
    values_str = ", ".join(_format_value(p.value) for p in pairs)
    return [
        {"role": "system", "content": _ROLLOUT_SYSTEM},
        {"role": "user", "content": f"Values: {values_str}"},
    ]


def _substitute_values(expression: str, pairs: list[ExprPair]) -> str:
    """Replace numeric literals in expression with their originating ExprPair.expr.

    We replace longest values first to avoid partial-match collisions
    (e.g. replacing "12" before "1" and "2" individually).
    """
    # Sort pairs by string length of their value descending to avoid partial matches
    sorted_pairs = sorted(pairs, key=lambda p: len(_format_value(p.value)), reverse=True)
    result = expression
    used: set[int] = set()

    for idx, pair in enumerate(sorted_pairs):
        if idx in used:
            continue
        token = _format_value(pair.value)
        # Only replace if it appears as a standalone number token
        pattern = r"(?<![0-9/])(" + re.escape(token) + r")(?![0-9/])"
        substituted = re.sub(pattern, pair.expr, result, count=1)
        if substituted != result:
            result = substituted
            used.add(idx)

    return result


def make_llm_rollout_policy(generator: "LLMGenerator") -> RolloutPolicy:
    """Return a rollout policy backed by the given LLMGenerator.

    The returned callable matches the RolloutPolicy signature:
      (list[ExprPair]) -> str

    Usage:
        policy = make_llm_rollout_policy(generator)
        expr, reward = mcts_search(numbers, rollout_policy=policy)
    """
    def policy(pairs: list[ExprPair]) -> str:
        if len(pairs) == 1:
            return pairs[0].expr

        messages = _build_rollout_prompt(pairs)
        raw_output = generator.generate(messages)
        expression = extract_expression(raw_output)

        if expression is None:
            # LLM failed to produce a parseable answer — fall back to first pair
            return pairs[0].expr

        # Map the LLM's expression over intermediate values back to original exprs
        return _substitute_values(expression, pairs)

    return policy
