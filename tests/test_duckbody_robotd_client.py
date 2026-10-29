"""duck-body's robotd client: the full robot.state, the three connections, the cadences."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import pytest

from duckmind.duckbody.protocol import Intent
from duckmind.duckbody.robotd_client import (
    Cadence,
    MoveStreamer,
    RobotdClient,
    RobotdError,
    RobotState,
)

LIVE_FRAME = {  # a phase-1 live robotd --sim frame (gait.jsonl), trimmed to the wire fields
    "t_ns": 99924856959000,
    "t": 29.7218,
    "policy": "walk",
    "move": {"applied": [0.0, 0.0, -0.0], "limited_by": ["deadman"], "requested": [0.0, 0.0, 0.0]},
    "head": [0.0, -0.0, 0.0, 0.0],
    "safety": {
        "fallen": False,
        "gain": 200,
        "gravity": [0.01124, 0.00161, -0.99994],
        "limp": False,
    },
    "loop": {"hz": 49.65396, "missed": 0},
    "joints": [
        0.00422,
        -0.08084,
        -0.45053,
        0.01069,
        0.45007,
        0.22974,
        0.44473,
        -0.04596,
        -0.01186,
        0.0,
        0.03417,
        0.086,
        0.45929,
        -0.01208,
        -0.45993,
    ],
    "targets": [0.0] * 15,
    "velocities": [0.0] * 15,
    "currents_ma": [
        0.02762,
        5.781,
        2.35757,
        8.74505,
        2.96852,
        1.69613,
        4.43298,
        0.21269,
        0.06869,
        0.0,
        0.32399,
        5.91869,
        1.18135,
        7.38418,
        1.65651,
    ],
    "odom": {"position": [-0.11511, 0.14545, 0.11672], "yaw": 2.68867},
    "imu": {"gyro": [0.0006, -0.0054, -0.00224], "quat": [0.22452, -0.00566, 0.00048, 0.97445]},
}


def test_a_live_frame_parses_whole() -> None:
    s = RobotState.from_wire(LIVE_FRAME)
    assert s.policy == "walk" and s.t_ns == 99924856959000 and s.limited_by == ("deadman",)
    assert s.head == (0.22974, 0.44473, -0.04596, -0.01186)
    assert s.leg_load == pytest.approx(36.4, abs=0.1)  # standing ≈ 36 (survey)
    assert s.odom_pos[2] == pytest.approx(0.11672)
    assert s.gain == 200 and s.loop_hz == pytest.approx(49.65396)
    assert math.degrees(s.tilt_rad or 0.0) < 1.0


def test_gravity_and_quaternion_agree() -> None:
    with_gravity = RobotState.from_wire(LIVE_FRAME)
    frame = dict(LIVE_FRAME, safety=dict(LIVE_FRAME["safety"], gravity=[0.0, 0.0, 0.0]))
    from_quat = RobotState.from_wire(frame)  # zeros = a daemon predating the field
    assert from_quat.gravity is None
    for a, b in zip(with_gravity.down or (), from_quat.down or (), strict=True):
        assert a == pytest.approx(b, abs=2e-4)


def test_empty_arrays_are_not_reported_never_zero() -> None:
    frame = dict(LIVE_FRAME, velocities=[], currents_ma=[])
    s = RobotState.from_wire(frame)
    assert s.velocities is None and s.currents_ma is None
    assert s.leg_load is None and s.head_speed is None


def test_a_broken_frame_is_refused() -> None:
    with pytest.raises(RobotdError):
        RobotState.from_wire({k: v for k, v in LIVE_FRAME.items() if k != "joints"})
    with pytest.raises(RobotdError):
        RobotState.from_wire(dict(LIVE_FRAME, safety={"fallen": "no", "limp": False}))


class FakeWire:
    """A scripted robotd connection: answers calls, queues notifications."""

    def __init__(self, answers: Mapping[str, Any] | None = None) -> None:
        self.sent: list[dict[str, Any]] = []
        self.inbox: list[dict[str, Any]] = []
        self.answers = dict(answers or {})

    def send(self, obj: Mapping[str, Any]) -> None:
        self.sent.append(dict(obj))
        if "id" in obj:
            method = obj["method"]
            result = self.answers.get(method, {"accepted": True})
            if isinstance(result, Exception):
                self.inbox.append({"jsonrpc": "2.0", "id": obj["id"], "error": str(result)})
            else:
                self.inbox.append({"jsonrpc": "2.0", "id": obj["id"], "result": result})

    def recv(self, timeout_s: float) -> dict[str, Any] | None:
        return self.inbox.pop(0) if self.inbox else None

    def close(self) -> None:
        pass


def client() -> tuple[RobotdClient, FakeWire, FakeWire, FakeWire]:
    command = FakeWire({"hello": {"api_version": 37}, "robot.model": {"asset": "alpha"}})
    state = FakeWire({"hello": {}, "robot.subscribe": {"accepted": True, "walk": "velstand.onnx"}})
    safety = FakeWire(
        {"hello": {}, "robot.health": {"healthy": True}, "robot.stop": {"accepted": True}}
    )
    c = RobotdClient(command, state, safety)
    c.start()
    return c, command, state, safety


def test_the_client_subscribes_to_every_tick() -> None:
    c, command, state, safety = client()
    subscribe = next(m for m in state.sent if m.get("method") == "robot.subscribe")
    assert subscribe["params"] == {}  # no hz: every tick
    assert c.model == {"asset": "alpha"}
    state.inbox.append({"jsonrpc": "2.0", "method": "robot.state", "params": LIVE_FRAME})
    got = c.read_state(0.1)
    assert got is not None and got.policy == "walk"
    assert c.read_state(0.0) is None


def test_moves_are_notifications_stop_and_health_use_the_safety_socket() -> None:
    c, command, state, safety = client()
    assert c.send(Intent.move(0.4, 0.0)) is None
    assert command.sent[-1] == {
        "jsonrpc": "2.0",
        "method": "robot.move",
        "params": {"vx": 0.4, "vy": 0.0, "vyaw": 0.0},
    }
    assert c.send(Intent.do("sit_toggle")) == {"accepted": True}
    assert c.send(Intent.stop()) == {"accepted": True}
    assert safety.sent[-1]["method"] == "robot.stop"
    assert c.health()["healthy"] is True


@pytest.mark.parametrize(
    "method", ["robot.enable", "robot.gains", "robot.reboot_motors", "robot.x"]
)
def test_the_client_never_sends_operator_methods(method: str) -> None:
    c, command, _, _ = client()
    before = len(command.sent)
    with pytest.raises(RobotdError):
        c.send(Intent(method, {}))
    assert len(command.sent) == before


def test_move_streamer_keeps_the_deadman_fed() -> None:
    s = MoveStreamer()
    sent = [s.step(t * 10_000_000, (0.4, 0.0)) for t in range(0, 30)]  # 10 ms ticks, 0.3 s
    times = [i * 10 for i, x in enumerate(sent) if x is not None]
    assert times[0] == 0 and all(b - a <= 20 for a, b in zip(times, times[1:], strict=False))
    assert s.step(300_000_000, None) is not None  # the first zero goes out at once
    assert all(s.step(t * 10_000_000, None) is None for t in range(31, 80))  # then silence
    assert s.step(800_000_000, (0.4, 0.2)).twist == (0.4, 0.2)  # type: ignore[union-attr]


def test_cadence() -> None:
    c = Cadence(1_000_000_000)
    assert [c.due(t * 250_000_000) for t in range(9)] == [
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
    ]
