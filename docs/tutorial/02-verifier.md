# Part 2: The Verifier — Deterministic Reward Without Hallucination

← [Part 1: Architecture](01-architecture.md) | Next: [Part 3: Puzzle Data →](03-puzzle-data.md)

---

## Why the Verifier Is the Most Critical Module

In most RL systems, the reward function is either:
- **Hand-crafted** — fragile, doesn't generalize
- **Learned** (reward model) — can be gamed by the policy

This system uses a third option: **deterministic arithmetic verification**. There is no way for the model to produce a high reward without actually solving the puzzle, because the verifier checks the math directly — no neural network, no language model, no ambiguity.

> If the verifier has a bug, the RL loop will exploit it. The model learns to satisfy a broken reward signal rather than solve the puzzle.

That's why the verifier has **90%+ test coverage enforced in CI**.

---

## What the Verifier Checks

Given an expression like `8 / (3 − 8/3)` and the puzzle `[3, 8, 8, 3]`:

```
Step 1: Extract numbers from expression → [3, 8, 8, 3]
Step 2: Compare with puzzle numbers    → match ✓
Step 3: Safe-evaluate the expression   → 24.0
Step 4: Compare to target (24)         → match ✓
Step 5: Return reward = 1.0
```

Every step fails safely:

```
Expression: "1 + __import__('os').system('rm -rf /')"
Step 1: Disallowed characters detected → reject immediately
Result: reward = 0.0, no code executed
```

---

## Code Walk-Through: `verify_solution()`

**File:** [`src/verifier/core.py`](../../src/verifier/core.py)

### Step 1: Extract the expression from model output

The model might produce various formats. The extractor handles them all:

```python
def extract_expression(raw_output: str) -> Optional[str]:
    patterns = [
        r"<answer>(.*?)</answer>",   # preferred format
        r"[Aa]nswer:\s*(.+)",        # prose format
        r"=\s*(.+?)\s*=\s*24",       # equation format
    ]
```

**Example outputs and what gets extracted:**

```
Model output: "... <answer>8 / (3 − 8/3)</answer>"
Extracted:    "8 / (3 − 8/3)"                         ✓

Model output: "The answer is: (3+3) * 4"
Extracted:    "(3+3) * 4"                              ✓

Model output: "I cannot solve this puzzle."
Extracted:    None                                     → reward = 0
```

### Step 2: Guard against injection

Before any parsing, the verifier scans for disallowed characters:

```python
if re.search(r"[`$_a-zA-Z\[\]{};:!@#%^&|~]", expression):
    return RewardSignal(reward=0.0, ..., error="disallowed characters")
```

This blocks:
- Variable names (`x`, `y`, `result`)
- Function calls (`sqrt`, `math.sin`)
- Import statements (`__import__`)
- Bitwise operators (`|`, `&`, `^`)
- Shell injection (`` ` ``, `;`)

**What IS allowed:** digits `0–9`, operators `+ - * /`, parentheses `( )`, decimal point `.`, and whitespace.

### Step 3: Check number usage

```python
used    = sorted(_extract_numbers_used(expression))   # [3, 3, 8, 8]
expected = sorted(input_numbers)                       # [3, 3, 8, 8]
if used != expected:
    return RewardSignal(reward=0.0, ..., error="number mismatch")
```

**Catches these mistakes:**

```
Expression: "8 * 3"          — only 2 numbers, not 4  ✗
Expression: "8+8+8+3"        — used 8 three times      ✗
Expression: "8/(3-8/3)"      — all four used once       ✓
```

### Step 4: Safe AST evaluation

This is the core security mechanism. Python's `ast` module parses the expression into a syntax tree. The evaluator only handles whitelisted node types:

```python
def _eval_node(node: ast.expr) -> float:
    if isinstance(node, ast.Constant):       # numeric literals only
        return float(node.value)
    if isinstance(node, ast.BinOp):          # a + b, a - b, a * b, a / b
        left  = _eval_node(node.left)
        right = _eval_node(node.right)
        ...
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_node(node.operand)     # negative numbers: -3
    raise ValueError(f"Unsupported node: {type(node)}")
```

**Why not just `eval()`?**

```python
# UNSAFE — never do this:
eval("8 / (3 - 8/3)")      # works, but also evaluates:
eval("__import__('os').system('rm -rf /')")  # disaster

# SAFE — what we do:
ast.parse("__import__('os')", mode="eval")
# → ast.Call node, not handled → raises ValueError → reward = 0
```

### Visualizing the AST

For `8 / (3 − 8/3)`:

```
         BinOp (/)
        /          \
   Constant(8)    BinOp (-)
                  /        \
           Constant(3)    BinOp (/)
                          /        \
                    Constant(8)   Constant(3)
```

The evaluator walks this tree bottom-up:
```
8/3 = 2.666...  →  3 - 2.666... = 0.333...  →  8 / 0.333... = 24.0  ✓
```

---

## Code Walk-Through: `brute_force_check()`

The brute-force solver is used to **label the dataset** — we need to know which puzzles are solvable before training. It tries every possible combination:

```
4! = 24 permutations of numbers
×  4³ = 64 operator combinations (op1, op2, op3)
×  5 parenthesization patterns
= up to 7,680 candidates per puzzle
```

The five parenthesization patterns:

```python
# Pattern 1: ((a ○ b) ○ c) ○ d  — left-deep tree
apply(apply(apply(a, b, op1), c, op2), d, op3)

# Pattern 2: (a ○ (b ○ c)) ○ d
apply(apply(a, apply(b, c, op2), op1), d, op3)

# Pattern 3: (a ○ b) ○ (c ○ d)  — balanced tree
apply(apply(a, b, op1), apply(c, d, op3), op2)

# Pattern 4: a ○ ((b ○ c) ○ d)
apply(a, apply(apply(b, c, op2), d, op3), op1)

# Pattern 5: a ○ (b ○ (c ○ d))  — right-deep tree
apply(a, apply(b, apply(c, d, op3), op2), op1)
```

Uses `Fraction` (rational arithmetic) to avoid floating-point errors:

```python
# Float arithmetic: imprecise
8 / (3 - 8/3) == 24   →  might be False due to float rounding!

# Fraction arithmetic: exact
Fraction(8) / (Fraction(3) - Fraction(8)/Fraction(3)) == Fraction(24)
→  always True
```

**Key implementation detail:** The `apply()` helper propagates `None` (division by zero) safely:

```python
def apply(x, y, op):
    if x is None or y is None:  # ← critical: propagate None
        return None
    if op == "/" and y == 0:
        return None
    return Fraction(_OPS[op](x, y))
```

Without this, `apply(None, 3, "+")` would crash with `TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'`, and the RL loop would silently corrupt training data.

---

## The Shaped Reward

`verify_solution()` returns binary rewards (0 or 1). For RL training, we need more granularity. The shaped reward in `src/rl/rewards.py` adds partial credit:

```
┌────────────────────────────────────────────────────────────┐
│                     REWARD COMPONENTS                      │
│                                                            │
│  Has <thought> + <answer> tags          +0.15 (format)    │
│                                                            │
│  Used correct 4 numbers (wrong result)  +0.25 (numbers)   │
│                                                            │
│  Expression evaluates to 24             +1.00 (solve)     │
│                                                            │
│  ─────────────────────────────────────────────────────     │
│  Maximum (format + solve, capped at 1)   1.00             │
└────────────────────────────────────────────────────────────┘
```

**Example breakdown for a partially correct response:**

```
Response: "<thought>Let me try...</thought><answer>3+3+8+8</answer>"

format_component  = 0.15  (both tags present)
numbers_component = 0.00  (3+3+8+8=22 uses right numbers but wrong result
                           → actually this gives numbers_component = 0.25)
solve_component   = 0.00  (22 ≠ 24)
total             = 0.40
```

**Why is the cap at 1.0 important?**

Without the cap, a response with `format + numbers + solve = 1.40` would outrank a clean solve. The cap ensures the signal converges to binary reward as training progresses.

---

## Testing the Verifier

The test suite (`tests/test_verifier.py`) covers:

```python
# Correct solutions
verify_solution("3 * (6 + 4 - 2)", [2, 3, 4, 6])  → reward=1.0, solved=True

# Wrong numbers
verify_solution("3 * 8", [3, 3, 8, 8])  → reward=0.0, error="number mismatch"

# Division by zero
verify_solution("8 / (3 - 3)", [8, 3, 3, 1])  → reward=0.0, error="could not be evaluated"

# Injection attempt
verify_solution("__import__('os')", [1, 2, 3, 4])  → reward=0.0, error="disallowed characters"

# Wrong result
verify_solution("3 + 3 + 8 + 8", [3, 3, 8, 8])  → reward=0.0, error="evaluated to 22.0"
```

Run the verifier tests with coverage:

```bash
pytest tests/test_verifier.py -v --cov=src/verifier --cov-fail-under=90
```

---

## Summary

| Mechanism | What It Prevents |
|-----------|-----------------|
| Character allowlist | Injection, import, variable names |
| AST evaluation (no `eval`) | Arbitrary code execution |
| Number mismatch check | Using wrong numbers, repeating numbers |
| `Fraction` arithmetic | Float rounding false positives/negatives |
| `None` propagation in `apply()` | Crash on nested division-by-zero |

The verifier is the **immune system** of the RL loop. Everything else can be retrained. A broken verifier corrupts the reward signal permanently.

---

Next: [Part 3 — Puzzle Data →](03-puzzle-data.md)
