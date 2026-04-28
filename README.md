# Game of 24 — Self-Improving Logic Solver

A production-grade AI system that combines LLM chain-of-thought reasoning, Monte Carlo Tree Search, and reinforcement learning (GRPO) to solve the Game of 24 — and continuously improves through verified feedback loops.

## Tutorial

New here? The illustrated tutorial walks through every component from scratch — no GPU required for Parts 0–5.

| Part | Topic | GPU? |
|------|-------|------|
| [0 — What is the Game of 24?](docs/tutorial/00-what-is-game-of-24.md) | Puzzle rules, difficulty, why it's a great RL benchmark | No |
| [1 — System Architecture](docs/tutorial/01-architecture.md) | Three-layer design, information flow, repository map | No |
| [2 — The Verifier](docs/tutorial/02-verifier.md) | AST-based safe evaluation, injection-resistance, brute-force labeling | No |
| [3 — Puzzle Data](docs/tutorial/03-puzzle-data.md) | Generation, difficulty tiers, solvability labeling | No |
| [4 — LLM + Chain-of-Thought](docs/tutorial/04-llm-and-cot.md) | Qwen-7B, CoT prompting, few-shot selection algorithm | No |
| [5 — Monte Carlo Tree Search](docs/tutorial/05-mcts.md) | UCB selection, ExprPair state, LLM rollout policy | No |
| [6 — GRPO Training](docs/tutorial/06-rl-training.md) | Group-relative advantage, shaped rewards, dead gradient fix | Yes |
| [7 — Running the System](docs/tutorial/07-running.md) | Install, benchmark, train, Docker, troubleshooting | Optional |

---

## Benchmark Results

Measured on 200 puzzles (seed=42), no GPU required:

| Strategy | Solve Rate | Notes |
|----------|-----------|-------|
| Random baseline (10 attempts) | 3.0% | Lower bound |
| **MCTS random rollout (500 iter)** | **58.0%** | No LLM, no GPU |
| Brute force (ceiling) | 77.0% | Exhaustive search |

MCTS alone reaches **75% of the theoretical ceiling** without any model. Replacing the random rollout policy with the fine-tuned LLM is the path to closing the gap.

## What This Is

Given four numbers, find a mathematical expression using each exactly once that evaluates to 24. The system:

1. **Prompts** a local LLM (Qwen-7B) with chain-of-thought few-shot examples
2. **Verifies** outputs through a deterministic reward function — no LLM hallucinations in the reward signal
3. **Searches** via MCTS or Tree of Thoughts when CoT fails
4. **Fine-tunes** with GRPO using shaped rewards across all non-zero trajectories

## Architecture

```
Input (4 numbers)
       │
       ▼
LLM + CoT Prompt ──► <thought> trace ──► Verifier (deterministic)
       │                                        │
       │                              ┌─────────┴──────────┐
       │                           Solved              Not solved
       │                              │                    │
       │                        reward = 1.0         MCTS / ToT
       │                              │                    │
       └──────────────────────────────┴────────────────────┘
                                      │
                              Shaped reward signal
                          (format + numbers + solve)
                                      │
                              GRPO Training Loop
                                      │
                             Improved Policy Model
```

## Repository Structure

```
src/
├── verifier/     Deterministic reward function — the critical path, 90%+ tested
├── llm/          Model loading, CoT prompts, tier-diverse few-shot selection
├── data/         Puzzle generation with brute-force labeling
├── reasoning/    MCTS (expression-tree state), LLM rollout policy, Tree of Thoughts
├── rl/           Shaped rewards, trajectory buffer, GRPO trainer
└── eval/         Evaluation framework with multi-run comparison table

scripts/
├── generate_dataset.py    Label puzzles solvable/unsolvable
├── run_baseline.py        CoT → MCTS fallback chain
├── train_rl.py            GRPO training loop with shaped rewards
├── evaluate.py            Model evaluation + run comparison
└── compare_strategies.py  GPU-free benchmark (random vs MCTS vs brute force)

docs/
├── tutorial/ Illustrated step-by-step tutorial (Parts 0–7)
├── adr/      Architecture Decision Records (model choice, RL algorithm, reward shaping)
└── sprints/  Sprint plans with retrospectives and real benchmark numbers
```

## Setup

```bash
pip install -e .
pip install -r requirements.txt

# Generate labeled puzzle dataset
python scripts/generate_dataset.py --n 1000

# Run baseline (CoT + MCTS fallback, no GPU)
python scripts/run_baseline.py --n-puzzles 100 --mcts-fallback

# Benchmark strategies without a GPU
python scripts/compare_strategies.py --n-puzzles 200 --mcts-iterations 500

# Full RL training loop (GPU required)
python scripts/train_rl.py --iterations 5 --rollouts-per-iter 50
```

## Docker

```bash
# Inference / baseline (CPU)
docker build -f docker/inference.Dockerfile -t game24-inference .
docker run game24-inference

# RL training (CUDA 12.1)
docker build -f docker/training.Dockerfile -t game24-training .
docker run --gpus all game24-training
```

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Base model | Qwen-7B-Chat | Better structured-output compliance than LLaMA-3; fits in ~5GB at 4-bit |
| RL algorithm | GRPO | No critic network; binary rewards suit group-relative advantage |
| Reward signal | Shaped (3-component) | Format + numbers + solve gives non-zero gradient early in training |
| Verifier | AST-based, no `eval()` | Injection-resistant; reward hacking is structurally impossible |
| MCTS state | Expression-tree (ExprPair) | Clean action space; exact fraction arithmetic avoids float errors |

Full rationale in [docs/adr/](docs/adr/).

## Sprint Plan

| Sprint | Focus | Status |
|--------|-------|--------|
| 1 | Baseline LLM + CoT prompting | Complete |
| 2 | Deterministic verifier + reward function | Complete |
| 3 | MCTS + Tree of Thoughts | Complete — 58% solve rate (no GPU) |
| 4 | GRPO RL loop and fine-tuning | Ready to run |

## Tests

```bash
pytest tests/ -v          # 125 tests, all passing
pytest tests/test_verifier.py --cov=src/verifier --cov-fail-under=90
```

CI enforces 90%+ coverage on the verifier on every push.
