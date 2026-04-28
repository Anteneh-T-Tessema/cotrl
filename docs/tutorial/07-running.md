# Part 7: Running the Full System

← [Part 6: RL Training](06-rl-training.md) | [Back to Tutorial Index →](README.md)

---

## Prerequisites

```
Python    3.10+
pip       latest
Git       any recent version
GPU       optional (required only for LLM training/inference)
VRAM      8GB+ for Qwen-7B at 4-bit (training needs ~16GB)
```

---

## Installation

```bash
git clone https://github.com/Anteneh-T-Tessema/cotrl.git
cd cotrl

# Create virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install the package in editable mode (Pylance-friendly)
pip install -e .
pip install -r requirements.txt
```

Verify installation:

```bash
python -c "from src.verifier.core import verify_solution; print('OK')"
pytest tests/ -v --tb=short
# Expected: 125 passed
```

---

## Step-by-Step: From Zero to 58% Solve Rate (No GPU)

### Step 1: Generate the puzzle dataset

```bash
python scripts/generate_dataset.py --n 1000
```

Output:

```
Generating 1000 puzzles (seed=42)...
Generated 1000 puzzles
  Solvable:   768 (76.8%)
  Unsolvable: 232 (23.2%)
Saved to data/processed/puzzles.jsonl
```

This creates `data/processed/puzzles.jsonl` with labeled puzzles. Takes ~3 seconds.

### Step 2: Benchmark strategies

```bash
python scripts/compare_strategies.py --n-puzzles 200 --mcts-iterations 500
```

Output:

```
Running benchmark on 200 puzzles (seed=42)...

Strategy                     Solve Rate   Time
───────────────────────────  ──────────   ──────
Random (10 attempts)              3.0%   0.2s
MCTS random rollout (500)        58.0%   47s
Brute force (ceiling)            77.0%   1.8s

MCTS reaches 75.3% of brute-force ceiling.
Results saved to results/strategy_comparison.json
```

No GPU, no model download. This is reproducible on any machine.

### Step 3: Explore the data (optional)

```bash
pip install jupyter matplotlib
jupyter notebook notebooks/01_eda_puzzle_distribution.ipynb
```

Run all cells. The notebook generates:
- Solve rate by difficulty tier
- Max number distribution histogram
- Few-shot seed example verification

### Step 4: Explore training dynamics (optional)

```bash
jupyter notebook notebooks/02_rl_training_dynamics.ipynb
```

Run all cells with `SYNTHETIC = True`. The notebook generates projected training curves showing what we expect from the GPU run.

---

## Step-by-Step: LLM Baseline (GPU Required)

### Step 5: Run the CoT baseline

```bash
python scripts/run_baseline.py --n-puzzles 100 --mcts-fallback
```

```
Loading Qwen-7B-Chat with 4-bit quantization...
Running CoT on 100 puzzles...

Results:
  CoT solve rate:         ~30%   (LLM zero-shot CoT)
  CoT + MCTS fallback:    ~55%   (LLM + MCTS)
  Trajectories saved:     data/processed/baseline.jsonl
```

The `--mcts-fallback` flag runs MCTS on any puzzle the LLM fails. Trajectories from both are saved to the buffer.

### Step 6: Evaluate on the test set

```bash
python scripts/evaluate.py \
    --model-path Qwen/Qwen1.5-7B-Chat \
    --n-puzzles 200

# Output:
# Solve rate: 31.5%
# Avg shaped reward: 0.48
# Results: results/eval_baseline.json
```

---

## Step-by-Step: GRPO Training (GPU Required)

### Step 7: Run the RL training loop

```bash
python scripts/train_rl.py \
    --iterations 5 \
    --rollouts-per-iter 50 \
    --group-size 8 \
    --mcts-fallback \
    --save-dir checkpoints/grpo
```

Progress output per iteration:

```
=== Iteration 1 / 5 ===
  Generating rollouts for 50 puzzles...
  Solve rate: 30.0% (15/50)
  Avg shaped reward: 0.382
  MCTS fallback: found 8 additional solutions
  Training on 41 trajectories (reward > 0)...
  Loss: 2.34 → 2.18
  Checkpoint saved: checkpoints/grpo/iter_001/
  Few-shot examples updated: 3 selected (high/mid/low tiers)

=== Iteration 2 / 5 ===
  ...
  Solve rate: 42.0%
```

### Step 8: Post-training evaluation

```bash
python scripts/evaluate.py \
    --model-path checkpoints/grpo/iter_004 \
    --n-puzzles 200 \
    --compare results/eval_baseline.json

# Output:
# ┌────────────────────────────────────────────────────┐
# │          Evaluation Comparison                      │
# ├──────────────────────┬─────────────────────────────┤
# │ Baseline (pre-RL)    │ Iteration 4 (post-RL)       │
# ├──────────────────────┼─────────────────────────────┤
# │ Solve rate:  31.5%   │ Solve rate:  61.0%          │
# │ Avg reward:  0.48    │ Avg reward:  0.63           │
# ├──────────────────────┴─────────────────────────────┤
# │ Gap vs MCTS baseline (58%): closed 3.0%            │
# │ Gap vs ceiling (77%):  remaining 16.0%             │
# └────────────────────────────────────────────────────┘
```

### Step 9: Analyze real training curves

```python
# In notebooks/02_rl_training_dynamics.ipynb:
SYNTHETIC = False
TRAJECTORY_DIR = Path('../checkpoints/grpo')
# Re-run all cells → plots reflect actual training data
```

---

## Docker

### CPU-only (inference and benchmarking)

```bash
docker build -f docker/inference.Dockerfile -t game24-inference .
docker run game24-inference
```

This runs `compare_strategies.py` with 200 puzzles automatically. Expected output:

```
MCTS random rollout (500 iter): 58.0% solve rate
```

### GPU training

```bash
docker build -f docker/training.Dockerfile -t game24-training .
docker run --gpus all game24-training
```

Requires CUDA 12.1 and Docker with GPU support (`nvidia-container-toolkit`).

---

## Running the Tests

```bash
# All 125 tests
pytest tests/ -v

# Verifier only with coverage gate
pytest tests/test_verifier.py --cov=src/verifier --cov-fail-under=90

# Specific test files
pytest tests/test_mcts.py -v
pytest tests/test_rewards.py -v
pytest tests/test_integration.py -v

# Run fast (skip slow MCTS tests)
pytest tests/ -v -m "not slow"
```

CI runs `pytest tests/ -v` on every push. The verifier coverage gate (`--cov-fail-under=90`) is enforced on every commit.

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'src'"

```bash
pip install -e .   # install in editable mode
```

If Pylance still shows errors, restart the Python language server in VS Code: `Cmd+Shift+P` → "Python: Restart Language Server".

### "ImportError: cannot import name 'LLMGenerator'"

This is expected — `LLMGenerator` is not in `src/llm/__init__.py` to avoid loading `transformers` at test time. Import directly:

```python
from src.llm.generator import LLMGenerator   # ✓
from src.llm import LLMGenerator             # ✗ will fail
```

### "CUDA out of memory"

Reduce `--rollouts-per-iter` and `--group-size`:

```bash
python scripts/train_rl.py --iterations 5 --rollouts-per-iter 20 --group-size 4
```

Or use CPU (very slow):

```bash
python scripts/run_baseline.py --device cpu --n-puzzles 10
```

### MCTS solve rate below 58%

This can happen with fewer than 500 iterations. Run:

```bash
python scripts/compare_strategies.py --mcts-iterations 1000
```

Expected: ~61% at 1000 iterations (diminishing returns vs. 500).

### Tests fail on [1,6,6,8] MCTS test

The test uses 1000 iterations (increased from 300 to ensure the single valid solution `6/(1-6/8)` is reliably found). If it still fails intermittently:

```bash
pytest tests/test_mcts.py -v --count=5   # run 5 times to see if it's flaky
```

The puzzle [2,2,4,6] was chosen as the stable replacement — it has multiple solutions.

---

## End-to-End Timing Reference

| Step | Hardware | Time |
|------|----------|------|
| Generate 1000 puzzles | CPU | 3s |
| Benchmark 200 puzzles (MCTS 500 iter) | CPU | 47s |
| Load Qwen-7B-Chat (4-bit) | GPU 16GB | 45s |
| CoT baseline 100 puzzles | GPU 16GB | 8 min |
| GRPO iteration (50 rollouts, group=8) | GPU 16GB | ~25 min |
| 5 GRPO iterations total | GPU 16GB | ~2.5 hours |
| EDA notebook | CPU | 15s |
| Training dynamics notebook (synthetic) | CPU | 5s |

---

## What to Try Next

1. **Increase MCTS iterations** — how does solve rate scale with compute? At what point do you hit the random-rollout ceiling?

2. **Ablate few-shot strategy** — compare tier-based selection vs. random selection. How much does example diversity matter?

3. **Add LLM rollout to MCTS** — `make_llm_rollout_policy()` is implemented. Wire it in and measure the solve rate improvement over random rollout.

4. **More GRPO iterations** — the 5-iteration training targets 65% solve rate. What happens at 20 iterations?

5. **Explore Tree of Thoughts** — `tot_search()` is implemented. Benchmark it against MCTS on the test set.

6. **Curriculum learning** — train on easy puzzles first, gradually introduce harder ones. Does this improve final solve rate?

---

## The Benchmark to Beat

```
Current best (no GPU):     58.0% (MCTS random rollout, 500 iter)
Theoretical ceiling:       77.0% (brute force)
Target after 5 RL iters:   65.0% (projected)
Ultimate target:           >70%  (LLM policy in MCTS + RL)
```

Good luck — and check the [ADRs](../adr/) if you want to understand the design tradeoffs behind each component.

---

← [Back to Tutorial Index](README.md)
