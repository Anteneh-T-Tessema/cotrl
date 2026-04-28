"""Compare solve rates across search strategies — no GPU required.

Runs three strategies on the same puzzle set and prints a comparison table:
  1. Random baseline    — random expression from the brute-force search space
  2. MCTS (random)      — Monte Carlo Tree Search with random rollout policy
  3. Brute force        — exhaustive search (theoretical upper bound)

This script is deliberately GPU-free so it can run in CI or on any machine
to produce reproducible benchmark numbers for the Sprint docs.

Usage:
    python scripts/compare_strategies.py --n-puzzles 200 --mcts-iterations 500
"""

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from src.data.puzzles import generate_puzzles
from src.reasoning.mcts import mcts_search
from src.verifier.core import verify_solution, brute_force_check, extract_expression


@dataclass
class StrategyResult:
    name: str
    solved: int
    total: int
    avg_time_ms: float

    @property
    def solve_rate(self) -> float:
        return self.solved / max(self.total, 1)

    def row(self) -> str:
        return (
            f"{self.name:<22} {self.solve_rate:>10.1%} "
            f"{self.solved:>8}/{self.total:<6} "
            f"{self.avg_time_ms:>12.1f}ms"
        )


def _random_expression(numbers: list[int]) -> str:
    """Generate a random expression by randomly ordering numbers and operators."""
    ops = ["+", "-", "*", "/"]
    nums = list(numbers)
    random.shuffle(nums)
    expr = str(nums[0])
    for n in nums[1:]:
        op = random.choice(ops)
        expr = f"({expr} {op} {n})"
    return expr


def run_random_baseline(puzzles, n_attempts: int = 10) -> StrategyResult:
    """Random expression generator — repeated sampling, take best of n_attempts."""
    solved = 0
    times = []
    for puzzle in puzzles:
        t0 = time.perf_counter()
        found = False
        for _ in range(n_attempts):
            expr = _random_expression(puzzle.numbers_list)
            if verify_solution(expr, puzzle.numbers_list).solved:
                found = True
                break
        times.append((time.perf_counter() - t0) * 1000)
        if found:
            solved += 1
    return StrategyResult("Random (10 attempts)", solved, len(puzzles), sum(times) / len(times))


def run_mcts(puzzles, n_iterations: int) -> StrategyResult:
    """MCTS with random rollout — no LLM, no GPU."""
    solved = 0
    times = []
    for puzzle in puzzles:
        t0 = time.perf_counter()
        _, reward = mcts_search(puzzle.numbers_list, n_iterations=n_iterations)
        times.append((time.perf_counter() - t0) * 1000)
        if reward == 1.0:
            solved += 1
    return StrategyResult(
        f"MCTS random ({n_iterations} iter)", solved, len(puzzles), sum(times) / len(times)
    )


def run_brute_force(puzzles) -> StrategyResult:
    """Exhaustive brute-force — the theoretical ceiling for this puzzle set."""
    solved = 0
    times = []
    for puzzle in puzzles:
        t0 = time.perf_counter()
        result = brute_force_check(puzzle.numbers_list)
        times.append((time.perf_counter() - t0) * 1000)
        if result is not None:
            solved += 1
    return StrategyResult("Brute force (ceiling)", solved, len(puzzles), sum(times) / len(times))


def print_table(results: list[StrategyResult], n_puzzles: int, seed: int) -> None:
    header = f"{'Strategy':<22} {'Solve Rate':>10} {'Solved':>15} {'Avg Time':>13}"
    sep = "-" * len(header)
    print(f"\nStrategy Comparison — {n_puzzles} puzzles (seed={seed})")
    print(sep)
    print(header)
    print(sep)
    for r in results:
        print(r.row())
    print(sep)


def save_results(results: list[StrategyResult], path: Path) -> None:
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            [
                {
                    "name": r.name,
                    "solved": r.solved,
                    "total": r.total,
                    "solve_rate": r.solve_rate,
                    "avg_time_ms": r.avg_time_ms,
                }
                for r in results
            ],
            f,
            indent=2,
        )
    logger.info(f"Results saved to {path}")


def main(n_puzzles: int, mcts_iterations: int, seed: int, output: Path) -> None:
    logger.info(f"Generating {n_puzzles} puzzles (seed={seed})...")
    dataset = generate_puzzles(n=n_puzzles, seed=seed)
    puzzles = list(dataset)

    results = []

    logger.info("Running random baseline...")
    results.append(run_random_baseline(puzzles))

    logger.info(f"Running MCTS (random rollout, {mcts_iterations} iterations)...")
    results.append(run_mcts(puzzles, mcts_iterations))

    logger.info("Running brute force (ceiling)...")
    results.append(run_brute_force(puzzles))

    print_table(results, n_puzzles, seed)
    save_results(results, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-puzzles", type=int, default=200)
    parser.add_argument("--mcts-iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/strategy_comparison.json"))
    args = parser.parse_args()
    main(args.n_puzzles, args.mcts_iterations, args.seed, args.output)
