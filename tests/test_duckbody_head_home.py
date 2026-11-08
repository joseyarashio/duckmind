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


def test_a_duck_resting_at_home_still_tells_robotd_its_head_twice_a_second() -> None:
    # robotd 0.16.1+ ([idle], on by default on a real board) takes the head and opens the beak once
    # 2 s pass with no robot.head (real duck prep 2026-11-08); before, a duck with its head at HOME
    # and nothing to do sent none at all
    loop = BodyLoop()
    sent = []
    for i in range(250):  # 5 s standing, no act
        out = loop.tick(_state(i * 0.02, 0.0, 0.0))
        sent += [i * 0.02 for intent in out.intents if intent.method == "robot.head"]
    gaps = [b - a for a, b in zip(sent, sent[1:], strict=False)]
    assert sent and sent[0] == 0.0 and max(gaps) <= 0.5 + 1e-9
    assert all(
        intent.params == {"neck_pitch": 0.0, "head_pitch": 0.0, "head_yaw": 0.0, "head_roll": 0.0}
        for intent in loop.tick(_state(5.5, 0.0, 0.0)).intents
        if intent.method == "robot.head"
    )
