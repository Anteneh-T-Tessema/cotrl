"""Generate and label a Game of 24 puzzle dataset.

Runs brute-force labeling to mark each puzzle solvable/unsolvable and records
the canonical solution. This dataset is the foundation for all downstream
evaluation and RL trajectory collection.

Usage:
    python scripts/generate_dataset.py --n 1000 --output data/raw/puzzles.jsonl
"""

import argparse
from pathlib import Path

from loguru import logger

from src.data.puzzles import generate_puzzles


def main(n: int, output: Path, seed: int) -> None:
    logger.info(f"Generating {n} labeled puzzles (seed={seed})...")
    dataset = generate_puzzles(n=n, seed=seed)

    solvable_count = len(dataset.solvable)
    logger.info(
        f"Generated {len(dataset)} puzzles | "
        f"Solvable: {solvable_count} ({solvable_count/len(dataset):.1%}) | "
        f"Unsolvable: {len(dataset.unsolvable)}"
    )

    dataset.save(output)
    logger.info(f"Saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("data/raw/puzzles.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.n, args.output, args.seed)
