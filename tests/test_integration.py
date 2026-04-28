"""End-to-end integration test — no GPU required.

Validates the full pipeline from puzzle generation through rollout, shaped
reward scoring, buffer accumulation, and trainer selection logic, using a
mocked LLM generator throughout.

This is the test an engineer would run to confirm the system is correctly
wired before kicking off an expensive GPU training run.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from src.data.puzzles import generate_puzzles
from src.llm.prompts import build_cot_prompt
from src.llm.few_shot import select_few_shot_examples, SEED_EXAMPLES
from src.reasoning.mcts import mcts_search
from src.reasoning.llm_rollout import make_llm_rollout_policy
from src.rl.rewards import compute_reward, ShapedReward
from src.rl.trajectory import Trajectory, TrajectoryBuffer
from src.rl.trainer import GRPOTrainer, GRPOConfig
from src.verifier.core import verify_solution, extract_expression


KNOWN_SOLVABLE = [2, 3, 4, 6]
CORRECT_RESPONSE = "<thought>3 * (6 + 4 - 2) = 3 * 8 = 24</thought>\n<answer>3 * (6 + 4 - 2)</answer>"
WRONG_RESPONSE = "<thought>I'll try 2 + 3 + 4 + 6 = 15</thought>\n<answer>2 + 3 + 4 + 6</answer>"
EMPTY_RESPONSE = ""


class TestPuzzleToReward:
    """Full path: puzzle numbers → prompt → reward signal."""

    def test_correct_response_reward_is_one(self) -> None:
        shaped = compute_reward(CORRECT_RESPONSE, KNOWN_SOLVABLE)
        assert shaped.solved is True
        assert shaped.total == 1.0

    def test_wrong_response_partial_credit(self) -> None:
        shaped = compute_reward(WRONG_RESPONSE, KNOWN_SOLVABLE)
        assert shaped.solved is False
        assert shaped.format_component > 0       # has thought + answer tags
        assert shaped.total < 1.0
        assert shaped.total > 0.0

    def test_empty_response_zero_reward(self) -> None:
        shaped = compute_reward(EMPTY_RESPONSE, KNOWN_SOLVABLE)
        assert shaped.total == 0.0


class TestRolloutToBuffer:
    """Full path: rollout → shaped reward → TrajectoryBuffer."""

    # Each entry is (puzzle_numbers, response) — numbers must match the response.
    _CASES: list[tuple[list[int], str]] = [
        (KNOWN_SOLVABLE, CORRECT_RESPONSE),   # solved
        (KNOWN_SOLVABLE, WRONG_RESPONSE),     # partial credit (tags present, wrong result)
        (KNOWN_SOLVABLE, EMPTY_RESPONSE),     # zero reward
    ]

    def _run_rollout(self, cases: list[tuple[list[int], str]]) -> TrajectoryBuffer:
        """Simulate the train_rl rollout phase with explicit puzzle–response pairs."""
        buffer = TrajectoryBuffer()
        for numbers, response in cases:
            messages = build_cot_prompt(numbers, use_seed_examples=False)
            shaped = compute_reward(response, numbers)
            buffer.add(Trajectory(
                puzzle=numbers,
                prompt=messages,
                response=response,
                reward=shaped.total,
                solved=shaped.solved,
                expression=shaped.expression,
            ))
        return buffer

    def test_buffer_stores_shaped_rewards(self) -> None:
        buffer = self._run_rollout(self._CASES)
        rewards = [t.reward for t in buffer.all()]
        assert rewards[0] == 1.0         # correct
        assert 0 < rewards[1] < 1.0      # partial credit (tags present, wrong result)
        assert rewards[2] == 0.0         # empty

    def test_solve_rate_matches_solved_flag(self) -> None:
        buffer = self._run_rollout(self._CASES[:2])  # correct + wrong
        assert buffer.solve_rate() == 0.5

    def test_buffer_serialisation_round_trip(self, tmp_path) -> None:
        buffer = self._run_rollout(self._CASES[:2])
        path = tmp_path / "traj.jsonl"
        buffer.save(path)
        loaded = TrajectoryBuffer.load(path)
        assert len(loaded.all()) == len(buffer.all())
        assert loaded.all()[0].reward == buffer.all()[0].reward


class TestTrainerSelection:
    """GRPOTrainer._select_trajectories filters correctly."""

    def _make_buffer(self, rewards: list[float]) -> TrajectoryBuffer:
        buf = TrajectoryBuffer()
        for r in rewards:
            buf.add(Trajectory(
                puzzle=[1, 2, 3, 4],
                prompt=[],
                response="",
                reward=r,
                solved=(r == 1.0),
            ))
        return buf

    def test_excludes_zero_reward(self) -> None:
        trainer = GRPOTrainer(GRPOConfig())
        buf = self._make_buffer([0.0, 0.15, 1.0])
        selected = trainer._select_trajectories(buf)
        assert all(t.reward > 0 for t in selected)
        assert len(selected) == 2

    def test_sorted_descending_by_reward(self) -> None:
        trainer = GRPOTrainer(GRPOConfig())
        buf = self._make_buffer([0.15, 1.0, 0.40])
        selected = trainer._select_trajectories(buf)
        rewards = [t.reward for t in selected]
        assert rewards == sorted(rewards, reverse=True)

    def test_empty_buffer_returns_empty(self) -> None:
        trainer = GRPOTrainer(GRPOConfig())
        buf = TrajectoryBuffer()
        assert trainer._select_trajectories(buf) == []

    def test_all_zero_returns_empty(self) -> None:
        trainer = GRPOTrainer(GRPOConfig())
        buf = self._make_buffer([0.0, 0.0])
        assert trainer._select_trajectories(buf) == []


class TestMCTSInPipeline:
    """MCTS + verifier wired correctly as a search fallback."""

    def test_mcts_finds_solution_and_verifier_accepts(self) -> None:
        numbers = [1, 2, 3, 4]
        expr, reward = mcts_search(numbers, n_iterations=300)
        assert reward == 1.0
        signal = verify_solution(expr, numbers)
        assert signal.solved

    def test_llm_rollout_policy_integration(self) -> None:
        gen = MagicMock()
        gen.generate.return_value = CORRECT_RESPONSE
        policy = make_llm_rollout_policy(gen)
        expr, reward = mcts_search(KNOWN_SOLVABLE, rollout_policy=policy, n_iterations=10)
        assert isinstance(reward, float)


class TestFewShotInPipeline:
    """Few-shot selection feeds correctly into build_cot_prompt."""

    def test_seed_examples_produce_valid_prompt(self) -> None:
        messages = build_cot_prompt([1, 2, 3, 4], use_seed_examples=True)
        roles = [m["role"] for m in messages]
        assert roles[0] == "system"
        assert roles[-1] == "user"
        # Each seed example adds a user+assistant pair
        assert len(messages) >= 2 + len(SEED_EXAMPLES) * 2

    def test_zero_shot_prompt_length(self) -> None:
        messages = build_cot_prompt([1, 2, 3, 4], use_seed_examples=False)
        assert len(messages) == 2  # system + user only

    def test_buffer_examples_selected_and_formatted(self) -> None:
        buf = TrajectoryBuffer()
        buf.add(Trajectory(
            puzzle=[2, 3, 4, 6],
            prompt=[],
            response=CORRECT_RESPONSE,
            reward=1.0,
            solved=True,
            expression="3 * (6 + 4 - 2)",
        ))
        examples = select_few_shot_examples(buf, k=1, min_thought_chars=0)
        assert len(examples) == 1
        messages = build_cot_prompt([1, 2, 3, 4], few_shot_examples=examples)
        assert len(messages) == 4  # system + 1×(user+assistant) + user
