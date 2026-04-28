# Sprint 4: GRPO RL Loop + Fine-Tuning

**Goal:** Close the self-improvement loop. Run GRPO training on successful trajectories from the rollout phase and measure solve rate improvement across iterations.

**Status:** Planned

## Objectives

- [ ] Implement `TrajectoryBuffer` with save/load for reproducibility
- [ ] Implement `GRPOTrainer` wrapping TRL's `GRPOTrainer`
- [ ] Run 5-iteration training loop (rollout → train → rollout)
- [ ] Plot solve rate per iteration
- [ ] Save final checkpoint to `checkpoints/grpo/final/`

## Iteration Targets (Aspirational)

| Iteration | Expected Solve Rate |
|-----------|-------------------|
| Baseline (Sprint 1) | ~30–40% |
| After Iter 1 | ~45% |
| After Iter 3 | ~55% |
| After Iter 5 | ~65% |

_These are estimates based on DeepSeek-Math GRPO results on similar tasks._

## Key Risks

- **Reward hacking:** Model learns to output well-formatted wrong answers that confuse the extractor. Mitigated by strict AST-based verifier.
- **Mode collapse:** Model stops exploring. Mitigated by KL penalty + group diversity sampling.
- **VRAM:** 7B model + 4-bit quant + GRPO group buffer. Monitor with `nvidia-smi` during training.

## Success Criteria

- Solve rate improves monotonically across at least 3 of 5 iterations
- No reward hacking (manual inspection of 20 random "solved" trajectories)
- Final model checkpoint reproducible from saved trajectories

## Retrospective

_To be filled in after execution._
