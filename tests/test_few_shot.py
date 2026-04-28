"""Tests for the few-shot example selector."""

import pytest
from src.llm.few_shot import (
    select_few_shot_examples,
    _thought_length,
    _number_tier,
    _format_example,
    SEED_EXAMPLES,
)
from src.rl.trajectory import Trajectory, TrajectoryBuffer


def _make_trajectory(
    puzzle: list[int],
    thought: str = "",
    solved: bool = True,
) -> Trajectory:
    response = f"<thought>{thought}</thought>\n<answer>expr</answer>" if thought else "expr"
    return Trajectory(
        puzzle=puzzle,
        prompt=[],
        response=response,
        reward=1.0 if solved else 0.0,
        solved=solved,
        expression="expr" if solved else None,
    )


class TestThoughtLength:
    def test_with_thought_tags(self) -> None:
        response = "<thought>step by step reasoning here</thought>"
        assert _thought_length(response) > 0

    def test_without_tags_returns_zero(self) -> None:
        assert _thought_length("just the answer") == 0

    def test_length_proportional_to_content(self) -> None:
        short = "<thought>a</thought>"
        long = "<thought>" + "reasoning " * 20 + "</thought>"
        assert _thought_length(long) > _thought_length(short)


class TestNumberTier:
    def test_low_tier(self) -> None:
        assert _number_tier([1, 2, 3, 4]) == "low"

    def test_mid_tier(self) -> None:
        assert _number_tier([1, 5, 6, 9]) == "mid"

    def test_high_tier(self) -> None:
        assert _number_tier([1, 5, 10, 13]) == "high"


class TestFormatExample:
    def test_keys_present(self) -> None:
        t = _make_trajectory([1, 2, 3, 4], thought="try 1+2+3*... = 24")
        ex = _format_example(t)
        assert "numbers" in ex
        assert "solution" in ex

    def test_numbers_match_puzzle(self) -> None:
        t = _make_trajectory([3, 6, 4, 2], thought="3*(6+4-2)=24")
        ex = _format_example(t)
        assert ex["numbers"] == [3, 6, 4, 2]


class TestSelectFewShotExamples:
    def _buffer_with(self, entries: list[tuple[list[int], str]]) -> TrajectoryBuffer:
        buf = TrajectoryBuffer()
        for puzzle, thought in entries:
            buf.add(_make_trajectory(puzzle, thought=thought, solved=True))
        return buf

    def test_empty_buffer_returns_empty(self) -> None:
        buf = TrajectoryBuffer()
        result = select_few_shot_examples(buf, k=3)
        assert result == []

    def test_no_solved_returns_empty(self) -> None:
        buf = TrajectoryBuffer()
        buf.add(_make_trajectory([1, 2, 3, 4], solved=False))
        assert select_few_shot_examples(buf, k=3) == []

    def test_returns_at_most_k(self) -> None:
        buf = self._buffer_with([
            ([1, 2, 3, 4], "short thought here extra padding"),
            ([2, 3, 4, 6], "another short thought here extra"),
            ([1, 5, 8, 8], "yet another thought here extra pad"),
            ([3, 6, 8, 9], "final thought entry here extra pad"),
        ])
        result = select_few_shot_examples(buf, k=2)
        assert len(result) <= 2

    def test_prefers_longer_thought(self) -> None:
        short_thought = "short"
        long_thought = "very detailed step by step reasoning that takes time"
        buf = self._buffer_with([
            ([1, 1, 1, 1], short_thought),
            ([1, 1, 1, 2], long_thought),
        ])
        result = select_few_shot_examples(buf, k=1, min_thought_chars=0)
        assert len(result) == 1
        assert long_thought in result[0]["solution"]

    def test_fallback_when_below_min_thought(self) -> None:
        buf = self._buffer_with([([1, 2, 3, 4], "tiny")])
        # min_thought_chars=100 so the trajectory fails quality filter
        # but fallback should still return it
        result = select_few_shot_examples(buf, k=1, min_thought_chars=100)
        assert len(result) == 1

    def test_format_matches_build_cot_prompt_expectation(self) -> None:
        buf = self._buffer_with([([2, 3, 4, 6], "3*(6+4-2)=24 so answer is 24")])
        result = select_few_shot_examples(buf, k=1, min_thought_chars=0)
        assert len(result) == 1
        assert isinstance(result[0]["numbers"], list)
        assert isinstance(result[0]["solution"], str)


class TestSeedExamples:
    def test_seed_examples_non_empty(self) -> None:
        assert len(SEED_EXAMPLES) > 0

    def test_seed_examples_have_required_keys(self) -> None:
        for ex in SEED_EXAMPLES:
            assert "numbers" in ex
            assert "solution" in ex

    def test_seed_examples_have_answer_tags(self) -> None:
        for ex in SEED_EXAMPLES:
            assert "<answer>" in ex["solution"]
