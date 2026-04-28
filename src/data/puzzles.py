"""Puzzle generation and dataset management for the Game of 24.

The canonical Game of 24 card deck uses numbers 1–13 (drawn from a standard
playing card deck, four suits). This module generates random puzzles from that
distribution, labels them solvable/unsolvable using the brute-force verifier,
and handles serialization for repeatability.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from ..verifier.core import brute_force_check


@dataclass(frozen=True)
class Puzzle:
    numbers: tuple[int, int, int, int]
    solvable: bool
    canonical_solution: Optional[str]  # None if unsolvable

    @property
    def numbers_list(self) -> list[int]:
        return list(self.numbers)

    def to_dict(self) -> dict:
        return {
            "numbers": list(self.numbers),
            "solvable": self.solvable,
            "canonical_solution": self.canonical_solution,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Puzzle":
        return cls(
            numbers=tuple(d["numbers"]),  # type: ignore[arg-type]
            solvable=d["solvable"],
            canonical_solution=d.get("canonical_solution"),
        )


class PuzzleDataset:
    """An ordered, serializable collection of labeled puzzles."""

    def __init__(self, puzzles: Optional[list[Puzzle]] = None) -> None:
        self._puzzles = puzzles or []

    def __len__(self) -> int:
        return len(self._puzzles)

    def __iter__(self) -> Iterator[Puzzle]:
        return iter(self._puzzles)

    def __getitem__(self, idx: int) -> Puzzle:
        return self._puzzles[idx]

    @property
    def solvable(self) -> "PuzzleDataset":
        return PuzzleDataset([p for p in self._puzzles if p.solvable])

    @property
    def unsolvable(self) -> "PuzzleDataset":
        return PuzzleDataset([p for p in self._puzzles if not p.solvable])

    @property
    def solve_rate(self) -> float:
        if not self._puzzles:
            return 0.0
        return sum(p.solvable for p in self._puzzles) / len(self._puzzles)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for puzzle in self._puzzles:
                f.write(json.dumps(puzzle.to_dict()) + "\n")

    @classmethod
    def from_file(cls, path: Path) -> "PuzzleDataset":
        puzzles = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    puzzles.append(Puzzle.from_dict(json.loads(line)))
        return cls(puzzles)

    def split(self, train_frac: float = 0.8, seed: int = 42) -> tuple["PuzzleDataset", "PuzzleDataset"]:
        rng = random.Random(seed)
        shuffled = list(self._puzzles)
        rng.shuffle(shuffled)
        cut = int(len(shuffled) * train_frac)
        return PuzzleDataset(shuffled[:cut]), PuzzleDataset(shuffled[cut:])


def _label_puzzle(numbers: tuple[int, int, int, int]) -> Puzzle:
    solution = brute_force_check(list(numbers))
    return Puzzle(
        numbers=numbers,
        solvable=solution is not None,
        canonical_solution=solution,
    )


def generate_puzzles(
    n: int,
    min_val: int = 1,
    max_val: int = 13,
    seed: int = 42,
    deduplicate: bool = True,
) -> PuzzleDataset:
    """Generate n random Game of 24 puzzles, labeled solvable/unsolvable.

    Args:
        n: Number of puzzles to generate.
        min_val: Minimum card value (default 1, matching Ace).
        max_val: Maximum card value (default 13, matching King).
        seed: RNG seed for reproducibility.
        deduplicate: Skip puzzles whose sorted number tuple has been seen before.

    Returns:
        A labeled PuzzleDataset.
    """
    rng = random.Random(seed)
    seen: set[tuple[int, ...]] = set()
    puzzles: list[Puzzle] = []

    attempts = 0
    max_attempts = n * 20

    while len(puzzles) < n and attempts < max_attempts:
        attempts += 1
        nums = tuple(sorted(rng.randint(min_val, max_val) for _ in range(4)))
        if deduplicate and nums in seen:
            continue
        seen.add(nums)
        puzzles.append(_label_puzzle(nums))  # type: ignore[arg-type]

    return PuzzleDataset(puzzles)


def load_puzzles(path: Path) -> PuzzleDataset:
    """Load a pre-labeled puzzle dataset from a JSONL file."""
    return PuzzleDataset.from_file(path)
