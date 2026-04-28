"""Sprint 1 entry point: baseline LLM + CoT prompting with MCTS fallback.

Pipeline per puzzle:
  1. LLM + CoT (fast, single-shot)
  2. If CoT fails → MCTS with random rollout (no GPU needed)
  3. If MCTS fails → MCTS with LLM rollout (GPU required, --llm-mcts flag)

Trajectories are saved to data/processed/ for downstream RL training.

Usage:
    # CoT only
    python scripts/run_baseline.py --n-puzzles 100

    # CoT + random-MCTS fallback (default)
    python scripts/run_baseline.py --n-puzzles 100 --mcts-fallback

    # CoT + LLM-MCTS fallback (requires loaded model)
    python scripts/run_baseline.py --n-puzzles 100 --mcts-fallback --llm-mcts
"""

import argparse
from pathlib import Path

from loguru import logger

from src.data.puzzles import generate_puzzles
from src.llm.generator import LLMGenerator, GenerationConfig
from src.llm.prompts import build_cot_prompt
from src.verifier.core import verify_solution, extract_expression
from src.reasoning.mcts import mcts_search
from src.reasoning.llm_rollout import make_llm_rollout_policy
from src.rl.trajectory import Trajectory, TrajectoryBuffer


def run(
    n_puzzles: int,
    output_path: Path,
    mcts_fallback: bool,
    llm_mcts: bool,
    mcts_iterations: int,
    seed: int,
) -> None:
    logger.info(f"Baseline run: {n_puzzles} puzzles, mcts_fallback={mcts_fallback}")

    generator = LLMGenerator(GenerationConfig(load_in_4bit=True))
    llm_policy = make_llm_rollout_policy(generator) if llm_mcts else None

    dataset = generate_puzzles(n=n_puzzles, seed=seed)
    buffer = TrajectoryBuffer()

    cot_solved = 0
    mcts_solved = 0

    for i, puzzle in enumerate(dataset):
        numbers = puzzle.numbers_list
        messages = build_cot_prompt(numbers)

        # Step 1: CoT
        response = generator.generate(messages)
        expression = extract_expression(response) or ""
        signal = verify_solution(expression, numbers)

        method = "cot"
        if signal.solved:
            cot_solved += 1
        elif mcts_fallback:
            # Step 2: MCTS fallback
            expr, reward = mcts_search(
                numbers,
                rollout_policy=llm_policy,
                n_iterations=mcts_iterations,
            )
            if reward == 1.0:
                expression = expr or expression
                signal = verify_solution(expression, numbers)
                mcts_solved += 1
                method = "mcts-llm" if llm_mcts else "mcts-random"

        if signal.solved:
            logger.debug(f"{numbers} → solved via {method}: {signal.expression}")

        buffer.add(Trajectory(
            puzzle=numbers,
            prompt=messages,
            response=response,
            reward=signal.reward,
            solved=signal.solved,
            expression=signal.expression,
        ))

        if (i + 1) % 10 == 0:
            total_solved = cot_solved + mcts_solved
            logger.info(
                f"[{i+1}/{n_puzzles}] "
                f"CoT: {cot_solved} | MCTS: {mcts_solved} | "
                f"Total solve rate: {total_solved/(i+1):.1%}"
            )

    buffer.save(output_path)
    total = cot_solved + mcts_solved
    logger.info(
        f"Done. CoT solved: {cot_solved} | MCTS solved: {mcts_solved} | "
        f"Total: {total}/{n_puzzles} ({total/n_puzzles:.1%})"
    )
    logger.info(f"Trajectories saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-puzzles", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("data/processed/baseline.jsonl"))
    parser.add_argument("--mcts-fallback", action="store_true", default=True)
    parser.add_argument("--llm-mcts", action="store_true", default=False)
    parser.add_argument("--mcts-iterations", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(
        n_puzzles=args.n_puzzles,
        output_path=args.output,
        mcts_fallback=args.mcts_fallback,
        llm_mcts=args.llm_mcts,
        mcts_iterations=args.mcts_iterations,
        seed=args.seed,
    )
