# ADR 002: Use GRPO Instead of PPO for RL Fine-Tuning

**Status:** Accepted  
**Date:** 2026-04-27

## Context

The RL loop requires a policy optimization algorithm to fine-tune the LLM using binary rewards from the verifier (0 = wrong, 1 = correct solution).

The two main candidates are:
- **PPO** (Proximal Policy Optimization) — standard RLHF algorithm
- **GRPO** (Group Relative Policy Optimization) — introduced in DeepSeek-Math

## Decision

Use **GRPO** via HuggingFace TRL.

## Rationale

**Memory:**  
PPO requires a separate value/critic network (same size as the policy model). On a 7B model, this doubles VRAM requirements. GRPO eliminates the critic by computing advantages from within-group reward variance.

**Signal quality:**  
Our reward is binary. PPO's Generalized Advantage Estimation (GAE) is designed to reduce variance in dense reward settings; with binary rewards, it adds complexity without meaningful variance reduction. GRPO's group-relative baseline achieves the same effect more directly: sample G completions per prompt, normalize by within-group mean reward.

**Implementation:**  
TRL ships `GRPOTrainer` (as of v0.8.6) with documented support for binary rewards. This reduces integration risk vs. implementing a custom PPO loop with a critic.

**Mathematical rationale:**  
GRPO advantage for completion i in group g of size G:

```
A_i = (r_i - mean(r_g)) / std(r_g)
```

This is equivalent to PPO advantage when the value function is perfectly estimated at the group mean — which holds for our homogeneous puzzle difficulty distribution.

## Consequences

- GRPO requires a minimum group size G ≥ 4 to compute meaningful variance. Set `group_size=8` in GRPOConfig.
- If we introduce multi-level rewards (partial credit for valid expressions that don't reach 24), revisit whether PPO's critic provides benefit.
- KL penalty against the reference model remains (`kl_coef=0.05`) to prevent reward hacking.
