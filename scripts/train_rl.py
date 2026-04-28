"""Sprint 4 entry point: full RL loop with GRPO fine-tuning.

Alternates between:
  1. Rollout phase  — generate trajectories, score with shaped rewards
  2. Training phase — GRPO update on all trajectories with reward > 0
  3. Evaluation     — measure solve rate and avg shaped reward per iteration

Using shaped rewards (format + numbers + solve components) instead of binary
0/1 gives GRPO more within-group variance and faster early learning.
See docs/adr/003-shaped-rewards-for-grpo.md.

Usage:
    python scripts/train_rl.py --iterations 5 --rollouts-per-iter 50
"""

import argparse
from pathlib import Path

from loguru import logger

from src.data.puzzles import generate_puzzles
from src.llm.generator import LLMGenerator, GenerationConfig
from src.llm.prompts import build_cot_prompt
from src.rl.rewards import compute_reward
from src.rl.trajectory import Trajectory, TrajectoryBuffer
from src.rl.trainer import GRPOTrainer, GRPOConfig


def rollout_phase(
    generator: LLMGenerator,
    n_puzzles: int,
    seed: int,
) -> TrajectoryBuffer:
    """Generate one iteration of trajectories with shaped rewards."""
    buffer = TrajectoryBuffer()
    dataset = generate_puzzles(n=n_puzzles, seed=seed)

    for puzzle in dataset:
        numbers = puzzle.numbers_list
        messages = build_cot_prompt(numbers)
        response = generator.generate(messages)
        shaped = compute_reward(response, numbers)

        buffer.add(Trajectory(
            puzzle=numbers,
            prompt=messages,
            response=response,
            reward=shaped.total,        # shaped reward, not binary 0/1
            solved=shaped.solved,
            expression=shaped.expression,
        ))

    return buffer


def log_iteration_stats(iteration: int, n_total: int, buffer: TrajectoryBuffer) -> None:
    all_t = buffer.all()
    solved = sum(t.solved for t in all_t)
    avg_reward = sum(t.reward for t in all_t) / max(len(all_t), 1)
    with_partial = sum(t.reward > 0 and not t.solved for t in all_t)
    logger.info(
        f"[Iter {iteration+1}/{n_total}] "
        f"Solved: {solved}/{len(all_t)} ({solved/len(all_t):.1%}) | "
        f"Partial credit: {with_partial} | "
        f"Avg shaped reward: {avg_reward:.3f}"
    )


def main(n_iterations: int, rollouts_per_iter: int, checkpoint_dir: Path) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    generator = LLMGenerator(GenerationConfig(load_in_4bit=True))
    trainer = GRPOTrainer(GRPOConfig(output_dir=str(checkpoint_dir)))

    for iteration in range(n_iterations):
        logger.info(f"=== Iteration {iteration + 1}/{n_iterations} ===")

        buffer = rollout_phase(generator, rollouts_per_iter, seed=iteration)
        log_iteration_stats(iteration, n_iterations, buffer)

        buffer.save(checkpoint_dir / f"trajectories_iter{iteration:03d}.jsonl")
        trainer.train(buffer)

    logger.info("RL training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--rollouts-per-iter", type=int, default=50)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/grpo"))
    args = parser.parse_args()
    main(args.iterations, args.rollouts_per_iter, args.checkpoint_dir)
