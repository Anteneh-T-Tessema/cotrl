# Part 0: What Is the Game of 24?

← [Tutorial Index](README.md) | Next: [Architecture →](01-architecture.md)

---

## The Puzzle

You receive four numbers. Your goal is to write a mathematical expression that:

- Uses **each of the four numbers exactly once**
- Combines them with `+`, `−`, `×`, `÷` (and parentheses)
- **Evaluates to exactly 24**

### Example 1 — Easy

```
Numbers: 2  3  4  6
```

Most people spot this quickly:

```
(2 + 3 + 4) × 6  ⟹  but wait, 2+3+4 = 9, and 9×6 = 54  ✗

Let's try: 4 × (6 − 3 + 2 − 1)  ⟹  but we don't have a 1
```

The actual solution:

```
3 × (6 + 4 − 2)  =  3 × 8  =  24  ✓
```

### Example 2 — Hard (the "infamous" [3, 3, 8, 8])

This one trips people up for minutes. Standard combinations:

```
3 + 3 + 8 + 8 = 22  ✗
3 × 8 − 3 − 8 = 13  ✗
(3 + 3) × (8 − 8) = 0  ✗
(8 + 8) × (3 − 3) = 0  ✗
```

The solution requires a nested fraction:

```
8 / (3 − 8/3)
= 8 / (9/3 − 8/3)
= 8 / (1/3)
= 24  ✓
```

This is *why* the puzzle is interesting for AI research: easy instances need simple arithmetic, hard instances need nested fraction reasoning that isn't obvious to humans, let alone language models.

---

## Why Build an AI Solver?

The Game of 24 is a **perfect RL benchmark** for three reasons:

### 1. Deterministic Reward
The verifier doesn't need a language model to judge correctness. It's pure arithmetic — either the expression evaluates to 24 or it doesn't. This avoids the "reward hacking" problem that plagues open-ended tasks.

```
Model output: "8 / (3 − 8/3)"
Verifier:      8 / (3 − 8/3) = 24.0  → reward = 1.0

Model output: "3 × 3 + 8 − 8"
Verifier:      3 × 3 + 8 − 8 = 9.0   → reward = 0.0
```

### 2. Graded Difficulty
The puzzle distribution naturally produces easy, medium, and hard instances. A good solver must generalize across all of them.

### 3. Measurable Ceiling
Brute force can enumerate all possible expressions. This gives us a theoretical ceiling (77% of randomly-generated puzzles are solvable). We can measure exactly how close our system gets.

---

## How Hard Is It? — Baseline Numbers

These are real benchmark results from this repository (200 puzzles, no GPU):

```
Strategy                    Solve Rate    Notes
─────────────────────────── ──────────── ─────────────────────────────
Random (10 attempts)              3.0%   Pure chance, lower bound
MCTS random rollout (500 iter)   58.0%   No LLM, no GPU, no training
Brute force (ceiling)            77.0%   Exhaustive — physically impossible
                                          for a model to match exactly
                                          because some puzzles require
                                          specific non-obvious paths
```

The gap from **58% → 77%** is what the RL training targets. The LLM policy replaces random rollouts with reasoned guesses — and gets fine-tuned on its own successes.

---

## What Does "Solvable" Mean?

Not every set of four numbers has a solution. For example:

```
[1, 1, 1, 1]  →  unsolvable  (max achievable: 1+1+1+1 = 4)
[11,11,11,11] →  unsolvable
[1, 2, 3, 4]  →  solvable    (1×2×3×4 = 24)
[3, 3, 8, 8]  →  solvable    (8/(3−8/3) = 24)
```

Across all possible four-number combinations (numbers 1–13), roughly **77%** are solvable. Our dataset generates puzzles and labels them using brute-force search, so we know ground truth for every puzzle.

---

## What You Will Build

By the end of this tutorial you will have a system that:

1. **Generates** labeled puzzles (solvable / unsolvable) using exact arithmetic
2. **Prompts** a language model with chain-of-thought reasoning examples
3. **Searches** using Monte Carlo Tree Search when the LLM fails
4. **Trains** the LLM on its own verified successes using GRPO
5. **Evaluates** progress with a test harness and convergence metrics

And all of it runs with **zero GPU** for the search and evaluation parts.

---

Next: [Part 1 — Architecture →](01-architecture.md)
