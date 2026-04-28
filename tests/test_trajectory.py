"""Tests for the RL trajectory buffer."""

import json
import tempfile
from pathlib import Path

from src.rl.trajectory import Trajectory, TrajectoryBuffer


def _make_trajectory(solved: bool = True) -> Trajectory:
    return Trajectory(
        puzzle=[1, 2, 3, 4],
        prompt=[{"role": "user", "content": "Numbers: 1 2 3 4"}],
        response="<answer>(1+2+3)*4</answer>",
        reward=1.0 if solved else 0.0,
        solved=solved,
        expression="(1+2+3)*4" if solved else None,
    )


class TestTrajectoryBuffer:
    def test_add_and_retrieve(self) -> None:
        buf = TrajectoryBuffer()
        buf.add(_make_trajectory(solved=True))
        buf.add(_make_trajectory(solved=False))
        assert len(buf.all()) == 2

    def test_successful_filter(self) -> None:
        buf = TrajectoryBuffer()
        buf.add(_make_trajectory(solved=True))
        buf.add(_make_trajectory(solved=False))
        buf.add(_make_trajectory(solved=True))
        assert len(buf.successful()) == 2

    def test_solve_rate_empty(self) -> None:
        assert TrajectoryBuffer().solve_rate() == 0.0

    def test_solve_rate(self) -> None:
        buf = TrajectoryBuffer()
        buf.add(_make_trajectory(solved=True))
        buf.add(_make_trajectory(solved=False))
        assert abs(buf.solve_rate() - 0.5) < 1e-9

    def test_save_and_load(self) -> None:
        buf = TrajectoryBuffer()
        buf.add(_make_trajectory(solved=True))
        buf.add(_make_trajectory(solved=False))

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = Path(f.name)

        buf.save(path)
        loaded = TrajectoryBuffer.load(path)
        assert len(loaded.all()) == 2
        assert loaded.all()[0].solved is True
        assert loaded.all()[1].solved is False

    def test_clear(self) -> None:
        buf = TrajectoryBuffer()
        buf.add(_make_trajectory())
        buf.clear()
        assert len(buf.all()) == 0

    def test_trajectory_round_trip(self) -> None:
        t = _make_trajectory(solved=True)
        assert Trajectory.from_dict(t.to_dict()) == t
