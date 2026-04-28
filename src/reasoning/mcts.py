"""Monte Carlo Tree Search over Game of 24 expression trees.

State representation:
  Each node holds a list of (numeric_value, expression_string) pairs — the
  numbers still available to combine. Initially 4 pairs, then 3, 2, 1.

  An action picks two indices i, j and an operator op, producing a new value
  and a parenthesised sub-expression. Terminal states have one pair remaining.

  The LLM acts as the rollout policy: given the current partial state, it
  generates a completion. The verifier provides the binary terminal reward.
  Random rollout is the default for development/testing without a GPU.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Optional

from ..verifier.core import verify_solution, RewardSignal


_OPS = ["+", "-", "*", "/"]
_OPS_FN: dict[str, Callable[[Fraction, Fraction], Optional[Fraction]]] = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: None if b == 0 else a / b,
}


@dataclass
class ExprPair:
    """A number still available for combination, with its expression string."""
    value: Fraction
    expr: str


def _apply_op(a: ExprPair, b: ExprPair, op: str) -> Optional[ExprPair]:
    result = _OPS_FN[op](a.value, b.value)
    if result is None:
        return None
    return ExprPair(value=result, expr=f"({a.expr} {op} {b.expr})")


def _random_rollout(pairs: list[ExprPair]) -> str:
    """Reduce a list of ExprPairs to one via random operator choices."""
    remaining = list(pairs)
    while len(remaining) > 1:
        i, j = random.sample(range(len(remaining)), 2)
        a, b = remaining[i], remaining[j]
        op = random.choice(_OPS)
        result = _apply_op(a, b, op)
        if result is None:
            op = random.choice(["+", "-", "*"])
            result = _apply_op(a, b, op)
        remaining = [p for k, p in enumerate(remaining) if k not in (i, j)]
        remaining.append(result)
    return remaining[0].expr


@dataclass
class MCTSNode:
    pairs: list[ExprPair]
    parent: Optional["MCTSNode"] = field(default=None, repr=False)
    children: list["MCTSNode"] = field(default_factory=list, repr=False)
    action_taken: Optional[str] = None  # description of the action that led here
    visits: int = 0
    total_reward: float = 0.0

    @property
    def is_terminal(self) -> bool:
        return len(self.pairs) == 1

    @property
    def ucb_score(self) -> float:
        if self.visits == 0:
            return float("inf")
        exploitation = self.total_reward / self.visits
        parent_visits = self.parent.visits if self.parent else 1
        exploration = math.sqrt(2 * math.log(parent_visits) / self.visits)
        return exploitation + exploration

    def best_child(self) -> "MCTSNode":
        return max(self.children, key=lambda c: c.ucb_score)

    def expand(self) -> list["MCTSNode"]:
        """Generate all one-step children from the current state."""
        children: list[MCTSNode] = []
        for i in range(len(self.pairs)):
            for j in range(len(self.pairs)):
                if i == j:
                    continue
                for op in _OPS:
                    result = _apply_op(self.pairs[i], self.pairs[j], op)
                    if result is None:
                        continue
                    new_pairs = [
                        p for k, p in enumerate(self.pairs) if k not in (i, j)
                    ] + [result]
                    child = MCTSNode(
                        pairs=new_pairs,
                        parent=self,
                        action_taken=f"{self.pairs[i].expr} {op} {self.pairs[j].expr}",
                    )
                    children.append(child)
        self.children = children
        return children

    def update(self, reward: float) -> None:
        self.visits += 1
        self.total_reward += reward


RolloutPolicy = Callable[[list[ExprPair]], str]


def mcts_search(
    numbers: list[int],
    rollout_policy: Optional[RolloutPolicy] = None,
    n_iterations: int = 500,
) -> tuple[Optional[str], float]:
    """Run MCTS to find a Game of 24 solution.

    Args:
        numbers: The four input puzzle numbers.
        rollout_policy: Takes a list[ExprPair] (current state) and returns an
            expression string. Defaults to random rollout. Pass an LLM-backed
            policy for the full system.
        n_iterations: Number of MCTS simulation iterations.

    Returns:
        (best_expression, best_reward) — highest-reward expression found,
        or (None, 0.0) if no solution was discovered.
    """
    policy = rollout_policy or _random_rollout
    root = MCTSNode(pairs=[ExprPair(Fraction(n), str(n)) for n in numbers])
    best: tuple[Optional[str], float] = (None, 0.0)

    for _ in range(n_iterations):
        # Selection
        node = root
        while node.children and not node.is_terminal:
            node = node.best_child()

        # Expansion
        if not node.is_terminal and node.visits > 0:
            children = node.expand()
            if children:
                node = random.choice(children)

        # Rollout
        expression = policy(node.pairs) if not node.is_terminal else node.pairs[0].expr
        signal: RewardSignal = verify_solution(expression, numbers)
        reward = signal.reward

        if reward > best[1]:
            best = (expression, reward)

        # Backpropagation
        current: Optional[MCTSNode] = node
        while current is not None:
            current.update(reward)
            current = current.parent

    return best
