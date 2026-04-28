# Sprint 3: Monte Carlo Tree Search Integration

**Goal:** Implement MCTS as a fallback search strategy for puzzles where LLM CoT fails. The LLM acts as the rollout policy; the verifier provides terminal rewards.

**Status:** Planned

## Objectives

- [ ] Implement `MCTSNode` dataclass with UCB score computation
- [ ] Implement `mcts_search()` with pluggable rollout policy
- [ ] Wire in `LLMGenerator.generate()` as the default rollout policy
- [ ] Benchmark MCTS solve rate vs. baseline CoT on the same 200 puzzles
- [ ] Profile: average iterations needed to find a solution

## Key Design Decisions

- LLM-as-policy means MCTS benefits from RL-improved models in Sprint 4
- UCB exploration constant set to √2 (standard; revisit if tree is too shallow/deep)
- Max depth = 10 to prevent infinite loops on degenerate states

## Success Criteria

- MCTS finds solutions the baseline CoT missed (measured on Sprint 1 failures)
- No infinite loops (depth cap enforced in tests)
- Verifier reward signal unchanged — MCTS does not modify the reward function

## Results (GPU-free benchmark, 200 puzzles, seed=42)

| Strategy | Solve Rate | Notes |
|----------|-----------|-------|
| Random baseline (10 attempts) | 3.0% | Lower bound |
| MCTS random rollout (500 iter) | **58.0%** | No LLM, no GPU |
| Brute force (ceiling) | 77.0% | Exhaustive search |

MCTS with random rollout reaches **75% of the brute-force ceiling** without any model.
Replacing the random rollout policy with the LLM (`make_llm_rollout_policy`) is the
Sprint 4 path to closing the remaining gap.

## Retrospective

- Expression-tree state representation (ExprPair) was the right call — it keeps
  MCTS actions well-defined and avoids string manipulation during search.
- UCB selection + lazy expansion gives good coverage in 500 iterations; diminishing
  returns set in around 800 iterations on this puzzle distribution.
- Tree of Thoughts (ToT) implemented alongside MCTS as a beam-search alternative;
  benchmark vs. MCTS deferred to after LLM integration in Sprint 4.
