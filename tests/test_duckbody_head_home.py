"""The guard's "head at HOME" (rule 3) with the live servos' sag (phase-3 integration)."""

from __future__ import annotations

from duckmind.duckbody.bodyloop import HEAD_HOME_TOL, BodyLoop
from duckmind.duckbody.robotd_client import RobotState

SAG = (-0.13, 0.10)  # measured live at HOME (robotd --sim, standing)


def _state(t: float, neck: float, head: float) -> RobotState:
    joints = [0.0] * 15
    joints[5], joints[6] = 0.3491 + neck, 0.3491 + head
    return RobotState.from_wire(
        {
            "t_ns": 10**10 + int(round(t * 1e9)),
            "policy": "stand",
            "move": {"requested": [0, 0, 0], "applied": [0, 0, 0]},
            "safety": {"fallen": False, "limp": False, "gravity": [0.0, 0.0, -1.0]},
            "joints": joints,
            "velocities": [0.0] * 15,
            "currents_ma": [18.0] + [0.0] * 9 + [18.0] + [0.0] * 4,
            "imu": {"gyro": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]},
        }
    )


def test_a_sagging_head_commanded_home_counts_as_home() -> None:
    assert max(abs(v) for v in SAG) < HEAD_HOME_TOL < 0.175  # half down is −0.125/+0.175
    loop = BodyLoop()
    for i in range(20):
        loop.tick(_state(i * 0.02, *SAG))
    assert loop.head_home_since is not None
    assert loop.last_guard.head_home_ms >= 300


def test_a_commanded_look_never_counts_as_home() -> None:
    loop = BodyLoop()
    for i in range(20):
        loop.head_target = (-0.125, 0.175, 0.0, 0.0)  # a look the body loop sent (half down)
        loop.tick(_state(i * 0.02, -0.05, 0.05))  # measured close to HOME, still commanded down
    assert loop.head_home_since is None
    assert loop.last_guard.head_home_ms == 0
