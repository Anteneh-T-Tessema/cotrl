# Part 3: Puzzle Generation and Dataset Labeling

← [Part 2: The Verifier](02-verifier.md) | Next: [Part 4: LLM + CoT →](04-llm-and-cot.md)

---

## What We Need From the Data Pipeline

Training and evaluation both need:
1. **Labeled puzzles** — know which puzzles are solvable before running the model
2. **A canonical solution** — to verify the verifier itself during development
3. **Reproducible splits** — same train/test split every run for fair comparison

The data module (`src/data/puzzles.py`) handles all of this.

---

## Puzzle Distribution

Numbers are drawn uniformly from 1–13 (standard card values). But the distribution of difficulty is uneven:

```
Max number in puzzle    Solvability rate
───────────────────     ────────────────
1 – 4  (low tier)       ~91.9%
5 – 9  (mid tier)       ~91.9%
10–13  (high tier)      ~72.8%
```

High-tier puzzles are harder because combinations like [11,11,11,11] are provably unsolvable — the numbers are too large to combine to 24 with basic arithmetic. This affects how we sample few-shot examples (covered in [Part 4](04-llm-and-cot.md)).

The distribution from the EDA notebook:

```
Solvable:   384 / 500 = 76.8%
Unsolvable: 116 / 500 = 23.2%
```

---

## How Puzzles Are Generated

**File:** [`src/data/puzzles.py`](../../src/data/puzzles.py)

```python
def generate_puzzles(n, min_val=1, max_val=13, seed=42, deduplicate=True):
    rng = random.Random(seed)
    seen = set()
    puzzles = []

    while len(puzzles) < n:
        numbers = tuple(sorted(rng.randint(min_val, max_val) for _ in range(4)))

        if deduplicate and numbers in seen:
            continue
        seen.add(numbers)

        solution = brute_force_check(list(numbers))
        puzzles.append(Puzzle(
            numbers=numbers,
            solvable=solution is not None,
            canonical_solution=solution,
        ))

    return PuzzleDataset(puzzles)
```

Key decisions:
- **Sorted tuples** — `(3,8,8,3)` and `(8,3,3,8)` are deduplicated to the same puzzle
- **Seeded RNG** — same seed = same dataset every time = reproducible benchmarks
- **Brute-force labeling** — `brute_force_check()` from the verifier labels each puzzle; no ambiguity

---

## The `PuzzleDataset` Class

```python
dataset = generate_puzzles(n=1000, seed=42)

dataset.solve_rate         # 0.768 — fraction of solvable puzzles
dataset.solvable           # list of Puzzle objects where solvable=True
dataset.unsolvable         # list of Puzzle objects where solvable=False

train, test = dataset.split(test_fraction=0.2, seed=42)
dataset.save(Path("data/processed/puzzles.jsonl"))
```

A `Puzzle` looks like:

```python
Puzzle(
    numbers=(3, 8, 8, 3),
    solvable=True,
    canonical_solution="((3 - 8/3) * 8)"  # one valid solution
)
```

---

## Puzzle Difficulty Tiers

The few-shot selector (Part 4) and evaluation framework both use tiers to ensure broad coverage:

```
┌─────────────────────────────────────────────────────────┐
│              PUZZLE DIFFICULTY TIERS                    │
│                                                         │
│  LOW tier   max(numbers) ≤ 4                            │
│  e.g. [1, 2, 3, 4] → 1×2×3×4 = 24         easy        │
│                                                         │
│  MID tier   max(numbers) ≤ 9                            │
│  e.g. [1, 5, 5, 5] → 5×(5−1/5)            medium      │
│                                                         │
│  HIGH tier  max(numbers) ≤ 13                           │
│  e.g. [3, 3, 8, 8] → 8/(3−8/3)            hard        │
└─────────────────────────────────────────────────────────┘
```

---

## Running the Data Generation Script

```bash
python scripts/generate_dataset.py --n 1000

# Output:
# Generated 1000 puzzles
# Solvable: 768 (76.8%)
# Unsolvable: 232 (23.2%)
# Saved to data/processed/puzzles.jsonl
```

**What it saves** (JSONL format, one puzzle per line):

```json
{"numbers": [1, 2, 3, 4], "solvable": true, "canonical_solution": "((1 + 2 + 3) * 4)"}
{"numbers": [1, 1, 1, 1], "solvable": false, "canonical_solution": null}
{"numbers": [3, 3, 8, 8], "solvable": true, "canonical_solution": "((3 - 8 / 3) * 8)"}
```

---

## Verifying the Verifier with the Dataset

The dataset serves a secondary purpose: **smoke-testing the verifier**. Every `canonical_solution` is itself a puzzle solution, so we can verify that `verify_solution(canonical_solution, numbers) == True` for every labeled solvable puzzle.

From the Sprint 1 retrospective:

```
Verifier smoke test: PASS — 0 mismatches on all 500 puzzles
```

This cross-check catches bugs in either direction:
- If `brute_force_check` finds a solution that `verify_solution` rejects → verifier bug
- If `verify_solution` accepts an expression that evaluates to ≠ 24 → verifier bug

---

## EDA: Puzzle Distribution Charts

The notebook `notebooks/01_eda_puzzle_distribution.ipynb` generates these charts (run it to see them):

**Solve rate by tier:**

```
HIGH tier:  ████████████████████░░░░░  72.8%
MID tier:   ████████████████████████░  91.9%
LOW tier:   ████████████████████████░  91.9%
```

**Max number distribution:**

The distribution peaks at mid-range values (5-9) because those numbers have the most combinatorial flexibility — they're large enough to reach 24 via multiplication, and small enough to divide evenly.

---

## Summary

```
generate_puzzles(n=1000)
        │
        ▼
  random numbers [1-13]
  deduplicated, sorted
        │
        ▼
  brute_force_check()   ← exact Fraction arithmetic
        │
        ├── solution found  → Puzzle(solvable=True,  canonical_solution="...")
        └── no solution     → Puzzle(solvable=False, canonical_solution=None)
        │
        ▼
  PuzzleDataset
        │
        ├── .split()        → train / test
        └── .save()         → data/processed/puzzles.jsonl
```

---

Next: [Part 4 — LLM + Chain-of-Thought →](04-llm-and-cot.md)
