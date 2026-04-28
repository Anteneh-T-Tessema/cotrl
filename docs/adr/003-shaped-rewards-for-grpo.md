# ADR 003: Shaped Rewards for GRPO Training

**Status:** Accepted  
**Date:** 2026-04-27

## Context

The initial design used binary rewards: 1.0 if the expression evaluates to 24, 0.0 otherwise. This is clean and correct, but creates a sparse learning signal for GRPO.

GRPO's group advantage for completion i in a group of G completions is:

```
A_i = (r_i - mean(r_g)) / (std(r_g) + ε)
```

With binary rewards, two failure modes arise:
- **All-zero groups**: when the model is early in training and consistently fails, every group has mean=0, std=0, advantage=0, and gradient=0. The model cannot improve.
- **All-one groups**: when the model is well-trained on easy puzzles, every group has mean=1, std=0, advantage=0, and gradient=0. The model stops improving even though hard puzzles remain unsolved.

Both failure modes produce wasted rollouts — GPU time spent generating trajectories that teach nothing.

## Decision

Replace binary reward with a three-component shaped reward, capped at 1.0:

| Component | Reward | Condition |
|-----------|--------|-----------|
| Format    | +0.15  | Both `<thought>` and `<answer>` tags present and non-empty |
| Numbers   | +0.25  | Correct four numbers used, expression parseable, result ≠ 24 |
| Solve     | +1.00  | Expression evaluates to exactly 24 |

The components are additive but capped at 1.0, so a fully correct response scores 1.0 regardless of format.

## Rationale

**Gradient flow**: With shaped rewards, a group of "all failed" responses still shows variance — a response with tags scores 0.15, one without scores 0.0, one with correct numbers scores 0.40. The group advantage is non-zero and the model receives a gradient signal even when it never solves the puzzle.

**Ordering preserved**: Format < Numbers < Solve guarantees the model cannot "game" a high reward without approaching a correct solution. A well-formatted wrong answer scores at most 0.40; the only way to score > 0.40 is to solve the puzzle.

**Convergence target unchanged**: A solved response always scores 1.0. The shaped components are only active in the sub-1.0 regime. Once the model consistently solves puzzles, the reward signal converges to the same binary target.

**Component values**: Chosen so that:
- A response with tags but wrong numbers: 0.15
- A response with tags and correct numbers: 0.40
- A solved response: 1.0

The 0.40 threshold matters: a GRPO group with one 0.40 response and seven 0.0 responses has non-zero variance (std ≈ 0.14) and produces a useful update.

## Consequences

- `GRPOTrainer` now trains on all trajectories with `reward > 0`, not just `buffer.successful()`. This increases the number of training examples per iteration.
- The rollout phase must use `compute_reward()` instead of the raw verifier to populate trajectory rewards.
- Sprint 4 evaluation should track **average shaped reward per iteration** alongside solve rate, to confirm gradient flow is healthy before solve rate rises.
- If reward hacking is observed (model learns to produce correct-looking tags without real reasoning), raise the `min_thought_chars` threshold in the few-shot selector and add a thought-quality component to the reward.
