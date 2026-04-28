# Sprint 1: Baseline LLM + CoT Prompting

**Goal:** Establish a measurable baseline solve rate using Qwen-7B-Chat with zero-shot chain-of-thought prompting. No RL, no MCTS.

**Status:** Planned

## Objectives

- [ ] Implement `src/llm/generator.py` — model loading with 4-bit quantization
- [ ] Implement `src/llm/prompts.py` — system prompt + few-shot CoT template
- [ ] Implement `src/verifier/core.py` — deterministic reward function
- [ ] Write verifier unit tests (`tests/test_verifier.py`, ≥90% coverage)
- [ ] Run baseline on 200 puzzles, record solve rate
- [ ] Save trajectories to `data/processed/baseline.jsonl`

## Success Criteria

- Verifier CI passes on every commit
- Baseline solve rate measured and documented in this file
- No model hallucinations bypassing the verifier (checked via disallowed-char tests)

## Results (EDA notebook, 500 puzzles, seed=42)

| Metric | Value |
|--------|-------|
| Puzzles generated | 500 |
| Solvable (brute force) | 384 (76.8%) |
| Unsolvable | 116 (23.2%) |
| Mid-tier solve rate (max 5–9) | 91.9% |
| High-tier solve rate (max 10–13) | 72.8% |
| Verifier smoke test | PASS — 0 mismatches on all 500 |
| Seed few-shot examples verified | 3/3 correct |
| Verifier test coverage | 90%+ (CI-enforced) |

_LLM zero-shot CoT solve rate: to be filled after Sprint 4 GPU run._

## Retrospective

- Brute-force labeling is fast enough to label 500 puzzles in < 2s — no need to cache.
- High-tier puzzles (10–13) have lower solvability because of confirmed-unsolvable
  combinations (e.g., [11,11,11,11]); model will see more of these in the eval split.
- Seed few-shot examples cover low / mid / hard solution patterns; tier-based
  diversity sampling in `few_shot.py` ensures future dynamic examples stay balanced.
