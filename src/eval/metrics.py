"""Evaluation framework for comparing model solve rates across training runs.

Designed to track progress across RL iterations and make regressions visible
immediately, rather than discovering them after a full training run.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

from loguru import logger

from ..data.puzzles import PuzzleDataset
from ..verifier.core import verify_solution, extract_expression


GenerateFn = Callable[[list[int]], str]


@dataclass
class PuzzleResult:
    numbers: list[int]
    raw_output: str
    expression: Optional[str]
    solved: bool
    reward: float
    latency_ms: float


@dataclass
class EvalResult:
    run_name: str
    total: int
    solved: int
    solve_rate: float
    avg_latency_ms: float
    results: list[PuzzleResult] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"[{self.run_name}] "
            f"Solve rate: {self.solve_rate:.1%} ({self.solved}/{self.total}) | "
            f"Avg latency: {self.avg_latency_ms:.0f}ms"
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "run_name": self.run_name,
                    "total": self.total,
                    "solved": self.solved,
                    "solve_rate": self.solve_rate,
                    "avg_latency_ms": self.avg_latency_ms,
                    "results": [asdict(r) for r in self.results],
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: Path) -> "EvalResult":
        with open(path) as f:
            d = json.load(f)
        results = [PuzzleResult(**r) for r in d.pop("results", [])]
        return cls(**d, results=results)


def evaluate_model(
    generate_fn: GenerateFn,
    dataset: PuzzleDataset,
    run_name: str = "eval",
    log_interval: int = 10,
) -> EvalResult:
    """Evaluate a generation function against a labeled puzzle dataset.

    Args:
        generate_fn: A callable that takes a list of 4 ints and returns the
            model's raw text output (e.g., LLMGenerator.generate wrapped to
            return a string).
        dataset: The puzzle dataset to evaluate on.
        run_name: Label for this eval run (e.g., "baseline", "iter_003").
        log_interval: Log progress every N puzzles.

    Returns:
        An EvalResult with per-puzzle details and aggregate metrics.
    """
    puzzle_results: list[PuzzleResult] = []

    for i, puzzle in enumerate(dataset):
        t0 = time.perf_counter()
        raw_output = generate_fn(puzzle.numbers_list)
        latency_ms = (time.perf_counter() - t0) * 1000

        expression = extract_expression(raw_output) or ""
        signal = verify_solution(expression, puzzle.numbers_list)

        puzzle_results.append(PuzzleResult(
            numbers=puzzle.numbers_list,
            raw_output=raw_output,
            expression=signal.expression,
            solved=signal.solved,
            reward=signal.reward,
            latency_ms=latency_ms,
        ))

        if (i + 1) % log_interval == 0:
            solved_so_far = sum(r.solved for r in puzzle_results)
            logger.info(
                f"[{run_name}] {i+1}/{len(dataset)} | "
                f"Solve rate: {solved_so_far/(i+1):.1%}"
            )

    solved = sum(r.solved for r in puzzle_results)
    avg_latency = sum(r.latency_ms for r in puzzle_results) / max(len(puzzle_results), 1)

    result = EvalResult(
        run_name=run_name,
        total=len(dataset),
        solved=solved,
        solve_rate=solved / max(len(dataset), 1),
        avg_latency_ms=avg_latency,
        results=puzzle_results,
    )
    logger.info(result.summary())
    return result


def compare_runs(results: list[EvalResult]) -> str:
    """Format a comparison table across multiple eval runs."""
    header = f"{'Run':<20} {'Solve Rate':>12} {'Solved':>8} {'Total':>7} {'Latency ms':>12}"
    separator = "-" * len(header)
    rows = [header, separator]
    for r in results:
        rows.append(
            f"{r.run_name:<20} {r.solve_rate:>11.1%} {r.solved:>8} {r.total:>7} {r.avg_latency_ms:>11.0f}"
        )
    return "\n".join(rows)
