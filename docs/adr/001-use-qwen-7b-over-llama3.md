# ADR 001: Use Qwen-7B-Chat over LLaMA-3-8B

**Status:** Accepted  
**Date:** 2026-04-27

## Context

We need a local LLM capable of multi-step chain-of-thought arithmetic reasoning that can be fine-tuned with GRPO on consumer or single-GPU hardware.

The two leading candidates are:
- **Qwen/Qwen1.5-7B-Chat** (Alibaba)
- **meta-llama/Meta-Llama-3-8B-Instruct** (Meta)

## Decision

Use **Qwen-7B-Chat** as the base model.

## Rationale

| Factor | Qwen-7B | LLaMA-3-8B |
|--------|---------|------------|
| Arithmetic benchmark (MATH) | Higher | Lower |
| Structured output compliance (`<thought>`, `<answer>` tags) | Reliable | Inconsistent |
| Commercial license | Qwen License (permissive for research) | LLaMA-3 Community License |
| 4-bit VRAM footprint | ~5GB | ~5.5GB |
| TRL / PEFT compatibility | Confirmed | Confirmed |

Qwen-7B's training data includes a higher proportion of mathematical and code reasoning tasks, which translates to better structured CoT output in our evaluation. The tag compliance difference is significant: unreliable `<answer>` tags break the verifier extraction pipeline and corrupt the reward signal.

## Consequences

- We are committed to the Qwen model family for Sprint 1 baseline.
- Revisit if Qwen-7B shows systematic failure modes (token budget, repetition) after Sprint 1 evaluation.
- LLaMA-3 remains a documented fallback.
