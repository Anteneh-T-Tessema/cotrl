# Tutorial: Game of 24 — Self-Improving Logic Solver

A complete, illustrated walk-through of the system — from a blank Python file to a self-improving AI that combines LLMs, Monte Carlo Tree Search, and Reinforcement Learning.

---

## Who This Is For

You have read a few ML papers and written some Python. You know what a neural network is and have heard of reinforcement learning. This tutorial explains exactly how the pieces fit together, with diagrams, code traces, and working examples you can run without a GPU.

---

## Tutorial Map

```
Part 0  → What Is the Game of 24?
Part 1  → How the System Is Structured
Part 2  → The Verifier: Reward Without Hallucination
Part 3  → Puzzle Generation and Brute-Force Labeling
Part 4  → LLM + Chain-of-Thought (Sprint 1)
Part 5  → Monte Carlo Tree Search (Sprint 3)
Part 6  → Shaped Rewards and GRPO Training (Sprint 4)
Part 7  → Running the Full System
```

| Chapter | File | GPU Required? |
|---------|------|--------------|
| [Part 0: The Game](00-what-is-game-of-24.md) | `00-what-is-game-of-24.md` | No |
| [Part 1: Architecture](01-architecture.md) | `01-architecture.md` | No |
| [Part 2: The Verifier](02-verifier.md) | `02-verifier.md` | No |
| [Part 3: Puzzle Data](03-puzzle-data.md) | `03-puzzle-data.md` | No |
| [Part 4: LLM + CoT](04-llm-and-cot.md) | `04-llm-and-cot.md` | No |
| [Part 5: MCTS](05-mcts.md) | `05-mcts.md` | No |
| [Part 6: RL Training](06-rl-training.md) | `06-rl-training.md` | Yes (optional) |
| [Part 7: Running It](07-running.md) | `07-running.md` | No |

---

## Five-Minute Preview

**The problem:**
```
Given [3, 8, 8, 3], write an expression using each number
exactly once that equals 24.
```

**What random guessing finds (3% solve rate):**
```
3 + 8 + 8 + 3 = 22  ✗
3 * 8 - 3 - 8 = 13  ✗
```

**What this system learns to find:**
```
8 / (3 - 8/3) = 8 / (9/3 - 8/3) = 8 / (1/3) = 24  ✓
```

**How it gets there — in three layers:**

```
Layer 1  LLM proposes expressions via chain-of-thought reasoning
Layer 2  MCTS searches when the LLM fails (no GPU needed)
Layer 3  GRPO trains the LLM on its own verified successes
```

Start reading at [Part 0](00-what-is-game-of-24.md) →
