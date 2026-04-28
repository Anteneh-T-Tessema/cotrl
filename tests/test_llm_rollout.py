"""Tests for the LLM rollout policy — fully mocked, no GPU required."""

from fractions import Fraction
from unittest.mock import MagicMock

from src.reasoning.mcts import ExprPair
from src.reasoning.llm_rollout import (
    make_llm_rollout_policy,
    _format_value,
    _substitute_values,
    _build_rollout_prompt,
)


class TestFormatValue:
    def test_integer(self) -> None:
        assert _format_value(Fraction(6)) == "6"

    def test_fraction(self) -> None:
        assert _format_value(Fraction(3, 4)) == "3/4"

    def test_negative(self) -> None:
        assert _format_value(Fraction(-1)) == "-1"


class TestBuildRolloutPrompt:
    def test_returns_two_messages(self) -> None:
        pairs = [ExprPair(Fraction(2), "(8-6)"), ExprPair(Fraction(3), "3")]
        messages = _build_rollout_prompt(pairs)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_values_appear_in_user_message(self) -> None:
        pairs = [ExprPair(Fraction(2), "(8-6)"), ExprPair(Fraction(3), "3")]
        messages = _build_rollout_prompt(pairs)
        assert "2" in messages[1]["content"]
        assert "3" in messages[1]["content"]


class TestSubstituteValues:
    def test_single_substitution(self) -> None:
        pairs = [ExprPair(Fraction(4), "(8-4)")]
        result = _substitute_values("4 * 6", pairs)
        assert "(8-4)" in result

    def test_no_match_returns_original(self) -> None:
        pairs = [ExprPair(Fraction(5), "(5)")]
        result = _substitute_values("3 * 8", pairs)
        assert result == "3 * 8"

    def test_multiple_substitutions(self) -> None:
        pairs = [
            ExprPair(Fraction(2), "(8-6)"),
            ExprPair(Fraction(3), "(1+2)"),
        ]
        result = _substitute_values("2 * 3 * 4", pairs)
        assert "(8-6)" in result
        assert "(1+2)" in result


class TestMakeLLMRolloutPolicy:
    def _make_generator(self, response: str) -> MagicMock:
        gen = MagicMock()
        gen.generate.return_value = response
        return gen

    def test_single_pair_returns_expr_directly(self) -> None:
        gen = self._make_generator("")
        policy = make_llm_rollout_policy(gen)
        pairs = [ExprPair(Fraction(24), "24")]
        result = policy(pairs)
        assert result == "24"
        gen.generate.assert_not_called()

    def test_calls_generator_for_multiple_pairs(self) -> None:
        gen = self._make_generator("<answer>2 * 3 * 4</answer>")
        policy = make_llm_rollout_policy(gen)
        pairs = [
            ExprPair(Fraction(2), "2"),
            ExprPair(Fraction(3), "3"),
            ExprPair(Fraction(4), "4"),
        ]
        result = policy(pairs)
        gen.generate.assert_called_once()
        assert isinstance(result, str)

    def test_fallback_when_llm_returns_no_answer(self) -> None:
        gen = self._make_generator("I cannot solve this.")
        policy = make_llm_rollout_policy(gen)
        pairs = [ExprPair(Fraction(2), "(8-6)"), ExprPair(Fraction(12), "12")]
        result = policy(pairs)
        assert result == "(8-6)"  # falls back to first pair's expr
