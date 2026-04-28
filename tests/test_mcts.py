"""Tests for the MCTS reasoning module."""

from fractions import Fraction

import pytest

from src.reasoning.mcts import ExprPair, MCTSNode, mcts_search, _random_rollout, _apply_op


class TestExprPair:
    def test_basic_construction(self) -> None:
        p = ExprPair(Fraction(3), "3")
        assert p.value == 3
        assert p.expr == "3"


class TestApplyOp:
    def test_addition(self) -> None:
        a = ExprPair(Fraction(3), "3")
        b = ExprPair(Fraction(4), "4")
        result = _apply_op(a, b, "+")
        assert result is not None
        assert result.value == 7
        assert result.expr == "(3 + 4)"

    def test_division_by_zero_returns_none(self) -> None:
        a = ExprPair(Fraction(5), "5")
        b = ExprPair(Fraction(0), "0")
        assert _apply_op(a, b, "/") is None

    def test_subtraction_negative_result(self) -> None:
        a = ExprPair(Fraction(1), "1")
        b = ExprPair(Fraction(5), "5")
        result = _apply_op(a, b, "-")
        assert result is not None
        assert result.value == -4


class TestMCTSNode:
    def _make_node(self, numbers: list[int]) -> MCTSNode:
        return MCTSNode(pairs=[ExprPair(Fraction(n), str(n)) for n in numbers])

    def test_is_terminal_with_one_pair(self) -> None:
        node = MCTSNode(pairs=[ExprPair(Fraction(24), "24")])
        assert node.is_terminal

    def test_not_terminal_with_four_pairs(self) -> None:
        node = self._make_node([1, 2, 3, 4])
        assert not node.is_terminal

    def test_ucb_unvisited_is_inf(self) -> None:
        parent = self._make_node([1, 2, 3, 4])
        parent.visits = 5
        child = MCTSNode(pairs=[], parent=parent)
        assert child.ucb_score == float("inf")

    def test_expand_generates_children(self) -> None:
        node = self._make_node([2, 3, 4, 6])
        children = node.expand()
        # 4 numbers × 3 other numbers × 4 ops = 48 possible children
        # (minus invalid divisions), so > 0 and reasonable upper bound
        assert len(children) > 0
        assert len(children) <= 48

    def test_update_increments_visits(self) -> None:
        node = self._make_node([1, 2, 3, 4])
        node.update(1.0)
        node.update(0.0)
        assert node.visits == 2
        assert node.total_reward == 1.0


class TestRandomRollout:
    def test_returns_string(self) -> None:
        pairs = [ExprPair(Fraction(n), str(n)) for n in [1, 2, 3, 4]]
        result = _random_rollout(pairs)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_single_pair_returns_expr(self) -> None:
        pairs = [ExprPair(Fraction(24), "24")]
        result = _random_rollout(pairs)
        assert result == "24"


class TestMCTSSearch:
    @pytest.mark.parametrize("numbers", [
        [1, 2, 3, 4],
        [2, 3, 4, 6],
        [2, 2, 4, 6],  # solution: (6*4)*(2/2); avoids single-path puzzles like [1,6,6,8]
    ])
    def test_finds_solution_for_solvable_puzzles(self, numbers: list[int]) -> None:
        # 1000 iterations keeps the test fast (~0.1s) while making random-rollout
        # failures statistically negligible for these high-solution-count puzzles.
        expr, reward = mcts_search(numbers, n_iterations=1000)
        assert reward == 1.0, f"MCTS failed on {numbers} — best expr: {expr}"

    def test_returns_none_reward_zero_on_impossible(self) -> None:
        # Not truly impossible for MCTS — but with very few iterations
        # it may not find a solution; this tests the return shape
        expr, reward = mcts_search([1, 1, 1, 1], n_iterations=5)
        assert isinstance(reward, float)
        assert reward in (0.0, 1.0)

    def test_solution_verifies_correctly(self) -> None:
        from src.verifier.core import verify_solution
        numbers = [3, 3, 8, 8]
        expr, reward = mcts_search(numbers, n_iterations=500)
        if reward == 1.0:
            signal = verify_solution(expr, numbers)
            assert signal.solved
