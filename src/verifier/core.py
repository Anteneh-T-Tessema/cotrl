"""Deterministic reward function for the Game of 24.

This is the most critical module in the repository. If the verifier has a bug,
the RL loop will exploit it — the model will learn to produce outputs that
satisfy a broken reward signal rather than actually solving the puzzle.

All reward logic must be:
  - Purely deterministic (no LLM calls)
  - Exhaustively tested (see tests/test_verifier.py)
  - Resistant to prompt injection via malformed <thought> tags
"""

import re
import ast
import operator
from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations, product
from typing import Optional


_OPS: dict[str, object] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
}

TARGET = 24
TOLERANCE = 1e-6


@dataclass(frozen=True)
class RewardSignal:
    reward: float
    solved: bool
    expression: Optional[str]
    error: Optional[str]


def extract_expression(raw_output: str) -> Optional[str]:
    """Pull the final answer expression from a model's <thought> trace.

    Looks for patterns like:
      <answer>3 * (4 + 4)</answer>
      Answer: 3 * (4 + 4)
      = 3 * (4 + 4)

    Returns the first matched expression string, or None.
    """
    patterns = [
        r"<answer>(.*?)</answer>",
        r"[Aa]nswer:\s*(.+)",
        r"=\s*(.+?)\s*=\s*24",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw_output, re.DOTALL)
        if match:
            return match.group(1).strip()
    return None


def _safe_eval(expression: str) -> Optional[float]:
    """Evaluate a mathematical expression safely without exec/eval on arbitrary code."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return None

    def _eval_node(node: ast.expr) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp):
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            op_map = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
            }
            op_fn = op_map.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unsupported operator: {type(node.op)}")
            if isinstance(node.op, ast.Div) and abs(right) < TOLERANCE:
                raise ZeroDivisionError
            return op_fn(left, right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -_eval_node(node.operand)
        raise ValueError(f"Unsupported node type: {type(node)}")

    try:
        return _eval_node(tree.body)
    except (ValueError, ZeroDivisionError, RecursionError):
        return None


def _extract_numbers_used(expression: str) -> list[int]:
    """Return the list of integer literals found in an expression string."""
    return [int(x) for x in re.findall(r"\b\d+\b", expression)]


def verify_solution(
    expression: str,
    input_numbers: list[int],
) -> RewardSignal:
    """Verify that an expression:
      1. Uses each of the four input numbers exactly once.
      2. Evaluates to 24 using only +, -, *, /.
      3. Contains no disallowed operations (exponentiation, bit ops, etc.).

    Args:
        expression: The candidate mathematical expression (e.g. "(3 + 1) * 6").
        input_numbers: The four puzzle numbers (e.g. [3, 1, 6, 2]).

    Returns:
        RewardSignal with reward=1.0 on full success, 0.0 otherwise.
    """
    if not expression:
        return RewardSignal(reward=0.0, solved=False, expression=None, error="empty expression")

    # Guard: reject anything with disallowed tokens before AST parsing
    if re.search(r"[`$_a-zA-Z\[\]{};:!@#%^&|~]", expression):
        return RewardSignal(
            reward=0.0, solved=False, expression=expression,
            error="expression contains disallowed characters",
        )

    used = sorted(_extract_numbers_used(expression))
    expected = sorted(input_numbers)
    if used != expected:
        return RewardSignal(
            reward=0.0, solved=False, expression=expression,
            error=f"number mismatch: used {used}, expected {expected}",
        )

    result = _safe_eval(expression)
    if result is None:
        return RewardSignal(
            reward=0.0, solved=False, expression=expression,
            error="expression could not be evaluated",
        )

    solved = abs(result - TARGET) < TOLERANCE
    return RewardSignal(
        reward=1.0 if solved else 0.0,
        solved=solved,
        expression=expression,
        error=None if solved else f"evaluated to {result:.4f}, not {TARGET}",
    )


def brute_force_check(numbers: list[int]) -> Optional[str]:
    """Exhaustive brute-force solver using rational arithmetic.

    Used to label the dataset (solvable vs. unsolvable puzzles) and to
    double-check verifier correctness in tests.

    Returns the first valid expression found, or None if unsolvable.
    """
    ops = list(_OPS.keys())
    for perm in permutations(numbers):
        a, b, c, d = [Fraction(n) for n in perm]
        for op1, op2, op3 in product(ops, repeat=3):
            def apply(x: Optional[Fraction], y: Optional[Fraction], op: str) -> Optional[Fraction]:
                if x is None or y is None:
                    return None
                if op == "/" and y == 0:
                    return None
                return Fraction(_OPS[op](x, y))

            candidates = [
                (apply(apply(apply(a, b, op1), c, op2), d, op3),
                 f"(({perm[0]} {op1} {perm[1]}) {op2} {perm[2]}) {op3} {perm[3]}"),
                (apply(apply(a, apply(b, c, op2), op1), d, op3),
                 f"({perm[0]} {op1} ({perm[1]} {op2} {perm[2]})) {op3} {perm[3]}"),
                (apply(apply(a, b, op1), apply(c, d, op3), op2),
                 f"({perm[0]} {op1} {perm[1]}) {op2} ({perm[2]} {op3} {perm[3]})"),
                (apply(a, apply(apply(b, c, op2), d, op3), op1),
                 f"{perm[0]} {op1} (({perm[1]} {op2} {perm[2]}) {op3} {perm[3]})"),
                (apply(a, apply(b, apply(c, d, op3), op2), op1),
                 f"{perm[0]} {op1} ({perm[1]} {op2} ({perm[2]} {op3} {perm[3]}))"),
            ]
            for result, expr in candidates:
                if result is not None and result == TARGET:
                    return expr
    return None
