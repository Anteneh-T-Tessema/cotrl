"""FastAPI web application for the Game of 24 solver.

Serves the static frontend from /frontend and exposes three API endpoints:
  GET  /api/random          → random solvable puzzle
  POST /api/solve           → SSE stream of MCTS progress + final result
  POST /api/verify          → verify a user-supplied expression
  GET  /api/benchmark       → pre-computed benchmark statistics
  GET  /api/health          → liveness check
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator

from .solver import solve_stream, verify_expression, random_puzzle

app = FastAPI(title="Game of 24 Solver", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"
_RESULTS_FILE = Path(__file__).parent.parent.parent / "results" / "strategy_comparison.json"


# ── API routes (must be defined before static mount) ───────────────────────────

class SolveRequest(BaseModel):
    numbers: list[int]
    iterations: int = 500

    @field_validator("numbers")
    @classmethod
    def validate_numbers(cls, v: list[int]) -> list[int]:
        if len(v) != 4:
            raise ValueError("exactly 4 numbers required")
        if not all(1 <= n <= 13 for n in v):
            raise ValueError("each number must be between 1 and 13")
        return v

    @field_validator("iterations")
    @classmethod
    def validate_iterations(cls, v: int) -> int:
        return max(100, min(v, 2000))


class VerifyRequest(BaseModel):
    numbers: list[int]
    expression: str

    @field_validator("numbers")
    @classmethod
    def validate_numbers(cls, v: list[int]) -> list[int]:
        if len(v) != 4:
            raise ValueError("exactly 4 numbers required")
        return v


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "version": "1.0.0"}


@app.get("/api/random")
async def get_random_puzzle() -> dict:
    return random_puzzle()


@app.post("/api/solve")
async def solve(req: SolveRequest) -> StreamingResponse:
    """Stream MCTS solving progress as Server-Sent Events."""
    async def event_stream():
        async for event in solve_stream(req.numbers, req.iterations):
            yield f"data: {json.dumps(event)}\n\n"
        yield "data: {\"type\": \"end\"}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/verify")
async def verify(req: VerifyRequest) -> dict:
    return verify_expression(req.numbers, req.expression)


@app.get("/api/benchmark")
async def benchmark() -> dict:
    if _RESULTS_FILE.exists():
        with open(_RESULTS_FILE) as f:
            return json.load(f)
    return {
        "strategies": [
            {"name": "Random (10 attempts)", "solve_rate": 0.03, "notes": "Lower bound"},
            {"name": "MCTS random rollout (500 iter)", "solve_rate": 0.58, "notes": "No GPU"},
            {"name": "Brute force (ceiling)", "solve_rate": 0.77, "notes": "Theoretical max"},
        ]
    }


# ── Static files (frontend) — mounted AFTER API routes ─────────────────────────

if _FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")

    @app.get("/")
    async def index():
        return FileResponse(str(_FRONTEND_DIR / "index.html"))

    @app.get("/{full_path:path}")
    async def catch_all(full_path: str):
        file = _FRONTEND_DIR / full_path
        if file.exists() and file.is_file():
            return FileResponse(str(file))
        return FileResponse(str(_FRONTEND_DIR / "index.html"))
