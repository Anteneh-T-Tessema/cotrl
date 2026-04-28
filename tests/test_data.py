"""Tests for puzzle generation and dataset management."""

import json
import tempfile
from pathlib import Path

import pytest

from src.data.puzzles import Puzzle, PuzzleDataset, generate_puzzles, load_puzzles


class TestPuzzle:
    def test_solvable_puzzle_has_solution(self) -> None:
        p = Puzzle(numbers=(1, 2, 3, 4), solvable=True, canonical_solution="(1+2+3)*4")
        assert p.solvable
        assert p.canonical_solution is not None

    def test_numbers_list(self) -> None:
        p = Puzzle(numbers=(2, 3, 4, 6), solvable=True, canonical_solution=None)
        assert p.numbers_list == [2, 3, 4, 6]

    def test_round_trip_serialization(self) -> None:
        p = Puzzle(numbers=(1, 5, 8, 8), solvable=True, canonical_solution="8/(1-1/8)")
        assert Puzzle.from_dict(p.to_dict()) == p


class TestPuzzleDataset:
    def _make_dataset(self) -> PuzzleDataset:
        return PuzzleDataset([
            Puzzle((1, 2, 3, 4), solvable=True, canonical_solution="x"),
            Puzzle((1, 1, 1, 1), solvable=False, canonical_solution=None),
            Puzzle((2, 3, 4, 6), solvable=True, canonical_solution="y"),
        ])

    def test_len(self) -> None:
        ds = self._make_dataset()
        assert len(ds) == 3

    def test_solvable_filter(self) -> None:
        ds = self._make_dataset()
        assert len(ds.solvable) == 2

    def test_unsolvable_filter(self) -> None:
        ds = self._make_dataset()
        assert len(ds.unsolvable) == 1

    def test_solve_rate(self) -> None:
        ds = self._make_dataset()
        assert abs(ds.solve_rate - 2 / 3) < 1e-9

    def test_save_and_load(self) -> None:
        ds = self._make_dataset()
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = Path(f.name)
        ds.save(path)
        loaded = load_puzzles(path)
        assert len(loaded) == len(ds)
        assert loaded[0].numbers == ds[0].numbers

    def test_split_sizes(self) -> None:
        # Generate enough puzzles to split meaningfully
        ds = generate_puzzles(n=20, seed=1)
        train, val = ds.split(train_frac=0.8, seed=0)
        assert len(train) + len(val) == len(ds)
        assert len(train) == int(len(ds) * 0.8)

    def test_empty_dataset_solve_rate(self) -> None:
        ds = PuzzleDataset([])
        assert ds.solve_rate == 0.0


class TestGeneratePuzzles:
    def test_generates_correct_count(self) -> None:
        ds = generate_puzzles(n=10, seed=7)
        assert len(ds) == 10

    def test_all_puzzles_have_four_numbers(self) -> None:
        ds = generate_puzzles(n=20, seed=7)
        for puzzle in ds:
            assert len(puzzle.numbers) == 4

    def test_numbers_in_valid_range(self) -> None:
        ds = generate_puzzles(n=50, seed=7, min_val=1, max_val=13)
        for puzzle in ds:
            assert all(1 <= n <= 13 for n in puzzle.numbers)

    def test_deduplication(self) -> None:
        ds = generate_puzzles(n=30, seed=42, deduplicate=True)
        seen = set()
        for puzzle in ds:
            key = puzzle.numbers  # already sorted (tuple)
            assert key not in seen, f"Duplicate puzzle: {key}"
            seen.add(key)

    def test_deterministic_with_seed(self) -> None:
        ds1 = generate_puzzles(n=10, seed=0)
        ds2 = generate_puzzles(n=10, seed=0)
        assert [p.numbers for p in ds1] == [p.numbers for p in ds2]

    def test_brute_force_label_correctness(self) -> None:
        ds = generate_puzzles(n=20, seed=5)
        for puzzle in ds:
            if puzzle.solvable:
                assert puzzle.canonical_solution is not None
            else:
                assert puzzle.canonical_solution is None
