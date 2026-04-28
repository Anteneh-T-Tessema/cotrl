# Sprint 2: Deterministic Verifier + Reward Function

**Goal:** Harden the verifier as core infrastructure. The reward function must be mathematically correct, injection-resistant, and 90%+ test covered before any RL work begins.

**Status:** Planned

## Objectives

- [ ] Implement `extract_expression()` — robust tag/prefix parsing
- [ ] Implement `_safe_eval()` — AST-based evaluation (no exec/eval on strings)
- [ ] Implement `verify_solution()` — number-usage check + math validation
- [ ] Implement `brute_force_check()` — reference solver for dataset labeling
- [ ] Achieve ≥90% test coverage enforced by CI
- [ ] Document edge cases: division by zero, wrong number count, prompt injection

## Key Design Constraints

- No LLM calls inside the verifier
- No `eval()` or `exec()` on user-provided strings
- Reject expressions containing non-arithmetic characters before AST parsing

## Success Criteria

- All parametrized tests pass
- Brute-force solver agrees with verifier on 100 random puzzles
- CI badge shows passing

## Retrospective

_To be filled in after execution._
