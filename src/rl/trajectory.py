from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import json
from pathlib import Path


@dataclass
class Trajectory:
    puzzle: list[int]
    prompt: list[dict[str, str]]
    response: str
    reward: float
    solved: bool
    expression: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "puzzle": self.puzzle,
            "prompt": self.prompt,
            "response": self.response,
            "reward": self.reward,
            "solved": self.solved,
            "expression": self.expression,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Trajectory":
        return cls(**d)


class TrajectoryBuffer:
    """Accumulates RL trajectories and supports persistence to disk."""

    def __init__(self) -> None:
        self._buffer: list[Trajectory] = []

    def add(self, trajectory: Trajectory) -> None:
        self._buffer.append(trajectory)

    def successful(self) -> list[Trajectory]:
        return [t for t in self._buffer if t.solved]

    def all(self) -> list[Trajectory]:
        return list(self._buffer)

    def solve_rate(self) -> float:
        if not self._buffer:
            return 0.0
        return len(self.successful()) / len(self._buffer)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for t in self._buffer:
                f.write(json.dumps(t.to_dict()) + "\n")

    @classmethod
    def load(cls, path: Path) -> "TrajectoryBuffer":
        buf = cls()
        with open(path) as f:
            for line in f:
                buf.add(Trajectory.from_dict(json.loads(line)))
        return buf

    def clear(self) -> None:
        self._buffer.clear()
