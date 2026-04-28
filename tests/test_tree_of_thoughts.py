"""Tests for Tree of Thoughts — fully mocked, no GPU required."""

from fractions import Fraction
from unittest.mock import MagicMock

from src.reasoning.mcts import ExprPair
from src.reasoning.tree_of_thoughts import ToTNode, _parse_step, tot_search


class TestToTNode:
    def test_terminal_with_one_pair(self) -> None:
        node = ToTNode(pairs=[ExprPair(Fraction(24), "24")])
        assert node.is_terminal

    def test_not_terminal_with_two_pairs(self) -> None:
        node = ToTNode(pairs=[ExprPair(Fraction(4), "4"), ExprPair(Fraction(6), "6")])
        assert not node.is_terminal

    def test_heuristic_closer_to_24_scores_higher(self) -> None:
        near = ToTNode(pairs=[ExprPair(Fraction(23), "23")])
        far = ToTNode(pairs=[ExprPair(Fraction(1), "1")])
        assert near.heuristic > far.heuristic

    def test_heuristic_exact_24_is_max(self) -> None:
        node = ToTNode(pairs=[ExprPair(Fraction(24), "24")])
        assert node.heuristic == 1.0


class TestParseStep:
    def _pairs(self, values: list[int]) -> list[ExprPair]:
        return [ExprPair(Fraction(v), str(v)) for v in values]

    def test_valid_step(self) -> None:
        pairs = self._pairs([3, 4, 6])
        result = _parse_step("<step>3 + 4 = 7</step>", pairs)
        assert result is not None
        new_pair, new_pairs = result
        assert new_pair.value == 7
        assert len(new_pairs) == 2  # removed 3 and 4, added 7

    def test_multiplication(self) -> None:
        pairs = self._pairs([4, 6])
        result = _parse_step("<step>4 * 6 = 24</step>", pairs)
        assert result is not None
        new_pair, _ = result
        assert new_pair.value == 24

    def test_division_by_zero_returns_none(self) -> None:
        pairs = self._pairs([5, 0, 8])
        result = _parse_step("<step>5 / 0 = inf</step>", pairs)
        assert result is None

    def test_missing_tag_returns_none(self) -> None:
        pairs = self._pairs([3, 4])
        result = _parse_step("3 + 4 = 7", pairs)
        assert result is None

    def test_value_not_in_pairs_returns_none(self) -> None:
        pairs = self._pairs([3, 4])
        result = _parse_step("<step>5 + 4 = 9</step>", pairs)
        assert result is None


class TestTotSearch:
    def _make_generator(self, response: str) -> MagicMock:
        gen = MagicMock()
        gen.generate.return_value = response
        return gen

    def test_returns_tuple(self) -> None:
        gen = self._make_generator("")
        expr, reward = tot_search([1, 2, 3, 4], generator=gen, k_proposals=2)
        assert isinstance(reward, float)
        assert reward in (0.0, 1.0)

    def test_finds_solution_when_llm_proposes_correct_steps(self) -> None:
        # Simulate an LLM that perfectly proposes the solution (1+2+3)*4 = 24
        # Step 1: 1 + 2 = 3 (from [1,2,3,4] → [3,3,4])
        # Step 2: 3 + 3 = 6 (from [3,3,4] → [6,4])
        # Step 3: 6 * 4 = 24 (from [6,4] → [24])
        responses = [
            "<step>1 + 2 = 3</step>",   # depth 0: [1,2,3,4]
            "<step>3 + 3 = 6</step>",   # depth 1: [3,3,4]
            "<step>6 * 4 = 24</step>",  # depth 2: [6,4]
        ]
        gen = self._make_generator("")
        gen.generate.side_effect = responses
        expr, reward = tot_search([1, 2, 3, 4], generator=gen, k_proposals=1, beam_width=1)
        assert reward == 1.0

    def test_no_crash_on_empty_llm_response(self) -> None:
        gen = self._make_generator("")
        expr, reward = tot_search([2, 3, 4, 6], generator=gen)
        assert reward == 0.0 or reward == 1.0  # either outcome is fine; just no crash
