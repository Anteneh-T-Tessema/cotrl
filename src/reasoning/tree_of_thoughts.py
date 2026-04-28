"""Tree of Thoughts (ToT) search for the Game of 24.

ToT differs from MCTS in how candidates are generated and evaluated:

  MCTS: random or LLM rollout from a node, backpropagates binary terminal reward.
  ToT:  LLM proposes k candidate NEXT STEPS at each node (breadth-first),
        a scoring function prunes to the best b branches, and we continue
        until a terminal state is reached or the depth budget is exhausted.

For Game of 24 the state is small (max depth 3), so BFS-ToT with beam search
is efficient and interpretable. The LLM is used for PROPOSAL; the verifier
provides terminal reward. No value network is needed.

Reference: "Tree of Thoughts: Deliberate Problem Solving with LLMs" (Yao et al., 2023).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING, Optional

from .mcts import ExprPair, _apply_op
from ..verifier.core import verify_solution

if TYPE_CHECKING:
    from ..llm.generator import LLMGenerator


_PROPOSE_SYSTEM = """\
You are a Game of 24 solver. Given a set of numbers, suggest candidate next steps.

A "step" combines exactly two of the available numbers with one operator (+, -, *, /)
to produce a new intermediate value. The remaining numbers carry over unchanged.

For each step, write one line in this exact format:
  <step>A op B = C</step>

where A and B are two of the available numbers, op is one of +, -, *, /, and C is the result.

Suggest {k} diverse steps. Prioritise steps that move toward 24.
Available numbers: {numbers}
"""


def _parse_step(line: str, pairs: list[ExprPair]) -> Optional[tuple[ExprPair, list[ExprPair]]]:
    """Parse a <step>A op B = C</step> line into a new ExprPair and remaining pairs."""
    match = re.search(r"<step>\s*(.+?)\s*([+\-*/])\s*(.+?)\s*=\s*(.+?)\s*</step>", line)
    if not match:
        return None

    left_str, op, right_str = match.group(1).strip(), match.group(2), match.group(3).strip()

    def find_pair(token: str, available: list[ExprPair]) -> Optional[tuple[ExprPair, list[ExprPair]]]:
        try:
            target = Fraction(token)
        except (ValueError, ZeroDivisionError):
            return None
        for i, p in enumerate(available):
            if p.value == target:
                return p, [q for j, q in enumerate(available) if j != i]
        return None

    found_left = find_pair(left_str, pairs)
    if found_left is None:
        return None
    left_pair, after_left = found_left

    found_right = find_pair(right_str, after_left)
    if found_right is None:
        return None
    right_pair, remaining = found_right

    result = _apply_op(left_pair, right_pair, op)
    if result is None:
        return None

    return result, remaining + [result]


@dataclass
class ToTNode:
    pairs: list[ExprPair]
    depth: int = 0
    parent: Optional["ToTNode"] = field(default=None, repr=False)
    score: float = 0.0  # heuristic: proximity of best pair value to 24

    @property
    def is_terminal(self) -> bool:
        return len(self.pairs) == 1

    @property
    def heuristic(self) -> float:
        """Score a node by how close any single pair value is to 24.

        Used to prune the beam — higher is better.
        Not a learned value function; just a fast distance proxy.
        """
        if not self.pairs:
            return 0.0
        return max(
            1.0 / (1.0 + abs(float(p.value) - 24.0))
            for p in self.pairs
        )


def tot_search(
    numbers: list[int],
    generator: "LLMGenerator",
    k_proposals: int = 5,
    beam_width: int = 3,
    max_depth: int = 3,
) -> tuple[Optional[str], float]:
    """Run Tree of Thoughts BFS with beam search.

    Args:
        numbers: The four input puzzle numbers.
        generator: An LLMGenerator used to propose candidate steps.
        k_proposals: Number of candidate steps to request from the LLM per node.
        beam_width: Maximum number of nodes to keep at each depth level.
        max_depth: Maximum search depth (Game of 24 requires exactly 3 reductions).

    Returns:
        (best_expression, best_reward) — the best terminal solution found.
    """
    beam: list[ToTNode] = [
        ToTNode(pairs=[ExprPair(Fraction(n), str(n)) for n in numbers])
    ]
    best: tuple[Optional[str], float] = (None, 0.0)

    for depth in range(max_depth):
        candidates: list[ToTNode] = []

        for node in beam:
            if node.is_terminal:
                expr = node.pairs[0].expr
                signal = verify_solution(expr, numbers)
                if signal.reward > best[1]:
                    best = (expr, signal.reward)
                continue

            # Ask LLM to propose next steps from this node
            values_str = ", ".join(
                str(int(p.value)) if p.value.denominator == 1 else str(float(p.value))
                for p in node.pairs
            )
            prompt = [
                {
                    "role": "system",
                    "content": _PROPOSE_SYSTEM.format(
                        k=k_proposals, numbers=values_str
                    ),
                },
                {"role": "user", "content": f"Available: {values_str}"},
            ]
            raw = generator.generate(prompt)

            # Parse each <step> line
            for line in raw.splitlines():
                parsed = _parse_step(line, node.pairs)
                if parsed is None:
                    continue
                _, new_pairs = parsed
                child = ToTNode(
                    pairs=new_pairs,
                    depth=depth + 1,
                    parent=node,
                )
                candidates.append(child)

                # Check terminal nodes immediately
                if child.is_terminal:
                    expr = child.pairs[0].expr
                    signal = verify_solution(expr, numbers)
                    if signal.reward > best[1]:
                        best = (expr, signal.reward)

        if not candidates:
            break

        # Beam pruning: keep top-b by heuristic score
        candidates.sort(key=lambda n: n.heuristic, reverse=True)
        beam = candidates[:beam_width]

    # Final pass: verify any terminal beam nodes
    for node in beam:
        if node.is_terminal:
            expr = node.pairs[0].expr
            signal = verify_solution(expr, numbers)
            if signal.reward > best[1]:
                best = (expr, signal.reward)

    return best
