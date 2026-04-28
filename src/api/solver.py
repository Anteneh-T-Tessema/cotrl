"""Async wrapper around the MCTS solver for use in the FastAPI app.

Runs the CPU-bound MCTS in a thread pool so it does not block the event loop.
Progress updates are pushed via an asyncio.Queue so the SSE endpoint can
stream them to the browser in real time.
"""
from __future__ import annotations

import asyncio
import random
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from typing import AsyncGenerator

from ..reasoning.mcts import ExprPair, MCTSNode, _random_rollout
from ..verifier.core import verify_solution, brute_force_check, RewardSignal

_executor = ThreadPoolExecutor(max_workers=4)


def _mcts_with_progress(
    numbers: list[int],
    n_iterations: int,
    progress_queue: "asyncio.Queue[dict]",
    loop: asyncio.AbstractEventLoop,
) -> dict:
    """Run MCTS synchronously; push progress events to an asyncio queue."""
    root = MCTSNode(pairs=[ExprPair(Fraction(n), str(n)) for n in numbers])
    best_expr: str | None = None
    best_reward: float = 0.0

    for iteration in range(n_iterations):
        # Selection
        node = root
        while node.children and not node.is_terminal:
            node = node.best_child()

        # Expansion
        if not node.is_terminal and node.visits > 0:
            children = node.expand()
            if children:
                node = random.choice(children)

        # Rollout
        expression = (
            _random_rollout(node.pairs) if not node.is_terminal else node.pairs[0].expr
        )
        signal: RewardSignal = verify_solution(expression, numbers)
        reward = signal.reward

        if reward > best_reward:
            best_reward = reward
            best_expr = expression

        # Backpropagation
        current = node
        while current is not None:
            current.update(reward)
            current = current.parent

        # Emit progress every 50 iterations
        if (iteration + 1) % 50 == 0 or (iteration + 1) == n_iterations:
            asyncio.run_coroutine_threadsafe(
                progress_queue.put({
                    "type": "progress",
                    "iteration": iteration + 1,
                    "total": n_iterations,
                    "best_expr": best_expr,
                    "best_reward": best_reward,
                    "solved": best_reward >= 1.0,
                }),
                loop,
            )

        # Short-circuit on first solve
        if best_reward >= 1.0:
            for remaining in range(iteration + 1, n_iterations, 50):
                asyncio.run_coroutine_threadsafe(
                    progress_queue.put({
                        "type": "progress",
                        "iteration": min(remaining + 50, n_iterations),
                        "total": n_iterations,
                        "best_expr": best_expr,
                        "best_reward": best_reward,
                        "solved": True,
                    }),
                    loop,
                )
            break

    return {
        "solved": best_reward >= 1.0,
        "expression": best_expr,
        "reward": best_reward,
        "method": "mcts",
    }


async def solve_stream(
    numbers: list[int],
    n_iterations: int = 500,
) -> AsyncGenerator[dict, None]:
    """Async generator that yields progress dicts as MCTS runs.

    Yields dicts with keys: type, iteration, total, best_expr, solved.
    Final dict has type='done' with the full result.
    """
    queue: asyncio.Queue[dict] = asyncio.Queue()
    loop = asyncio.get_event_loop()
    start = time.monotonic()

    future = loop.run_in_executor(
        _executor,
        _mcts_with_progress,
        numbers,
        n_iterations,
        queue,
        loop,
    )

    emitted = 0
    while not future.done() or not queue.empty():
        try:
            event = await asyncio.wait_for(queue.get(), timeout=0.1)
            yield event
            emitted += 1
        except asyncio.TimeoutError:
            continue

    result = await future
    elapsed_ms = int((time.monotonic() - start) * 1000)
    yield {
        "type": "done",
        "solved": result["solved"],
        "expression": result["expression"],
        "reward": result["reward"],
        "method": result["method"],
        "elapsed_ms": elapsed_ms,
        "iterations": n_iterations,
    }


def verify_expression(numbers: list[int], expression: str) -> dict:
    """Synchronous verifier wrapper for the /api/verify endpoint."""
    signal = verify_solution(expression, numbers)
    return {
        "valid": signal.solved,
        "reward": signal.reward,
        "expression": signal.expression,
        "error": signal.error,
    }


def random_puzzle() -> dict:
    """Return a random puzzle with known solvability."""
    rng = random.Random()
    for _ in range(100):
        nums = sorted(rng.randint(1, 13) for _ in range(4))
        solution = brute_force_check(nums)
        if solution is not None:
            return {"numbers": nums, "solvable": True, "example_solution": solution}
    return {"numbers": [1, 2, 3, 4], "solvable": True, "example_solution": "1 * 2 * 3 * 4"}
