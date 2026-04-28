"""Tests for the shaped reward function."""

import pytest
from src.rl.rewards import (
    compute_reward,
    compute_batch_rewards,
    ShapedReward,
    FORMAT_REWARD,
    NUMBERS_REWARD,
    SOLVE_REWARD,
)


PUZZLE = [3, 6, 4, 2]
CORRECT_EXPR = "3 * (6 + 4 - 2)"
CORRECT_RESPONSE = f"<thought>3 * 8 = 24</thought>\n<answer>{CORRECT_EXPR}</answer>"
WRONG_RESULT = "<thought>trying...</thought>\n<answer>3 + 6 + 4 + 2</answer>"
NO_TAGS = "The answer is 3 * (6 + 4 - 2)"
EMPTY = ""


class TestComputeReward:
    def test_correct_solution_scores_one(self) -> None:
        r = compute_reward(CORRECT_RESPONSE, PUZZLE)
        assert r.solved is True
        assert r.total == 1.0
        assert r.solve_component == SOLVE_REWARD

    def test_correct_solution_includes_format_bonus(self) -> None:
        r = compute_reward(CORRECT_RESPONSE, PUZZLE)
        assert r.format_component == FORMAT_REWARD

    def test_wrong_result_with_tags_gets_partial_credit(self) -> None:
        r = compute_reward(WRONG_RESULT, PUZZLE)
        assert r.solved is False
        assert r.format_component == FORMAT_REWARD
        assert r.total > 0
        assert r.total < 1.0

    def test_no_tags_no_format_bonus(self) -> None:
        r = compute_reward(NO_TAGS, PUZZLE)
        assert r.format_component == 0.0

    def test_empty_response_zero_reward(self) -> None:
        r = compute_reward(EMPTY, PUZZLE)
        assert r.total == 0.0
        assert r.solved is False

    def test_total_never_exceeds_one(self) -> None:
        r = compute_reward(CORRECT_RESPONSE, PUZZLE)
        assert r.total <= 1.0

    def test_format_reward_requires_both_tags(self) -> None:
        thought_only = "<thought>trying...</thought>final answer: 24"
        r = compute_reward(thought_only, PUZZLE)
        assert r.format_component == 0.0

    def test_wrong_numbers_no_numbers_bonus(self) -> None:
        # Expression uses wrong numbers entirely
        wrong_nums = "<thought>...</thought>\n<answer>5 * 5 - 1</answer>"
        r = compute_reward(wrong_nums, PUZZLE)
        assert r.numbers_component == 0.0

    def test_detail_populated(self) -> None:
        r = compute_reward(CORRECT_RESPONSE, PUZZLE)
        assert isinstance(r.detail, str)
        assert len(r.detail) > 0

    @pytest.mark.parametrize("response,expected_solved", [
        (CORRECT_RESPONSE, True),
        (WRONG_RESULT, False),
        (EMPTY, False),
    ])
    def test_solved_flag(self, response: str, expected_solved: bool) -> None:
        r = compute_reward(response, PUZZLE)
        assert r.solved is expected_solved


class TestComputeBatchRewards:
    def test_batch_length_matches_input(self) -> None:
        responses = [CORRECT_RESPONSE, EMPTY, WRONG_RESULT]
        puzzles = [PUZZLE, PUZZLE, PUZZLE]
        results = compute_batch_rewards(responses, puzzles)
        assert len(results) == 3

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(AssertionError):
            compute_batch_rewards([CORRECT_RESPONSE], [PUZZLE, PUZZLE])

    def test_correct_response_scores_highest(self) -> None:
        results = compute_batch_rewards(
            [CORRECT_RESPONSE, WRONG_RESULT, EMPTY], [PUZZLE, PUZZLE, PUZZLE]
        )
        assert results[0].total >= results[1].total >= results[2].total
