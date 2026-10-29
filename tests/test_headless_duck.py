"""The headless duck in a real home (:mod:`duckmind.world.headless_duck`; spec §17 期3 harness).

It must answer in robotd's words — ``robot.state`` that duck-body's own parser takes, with the
deadman, the fall verdict, odometry and the frames — and move the official physics like phase 1's
HeadlessDuck did (velstand at vx 0.40 really advances ~0.16 m/s, gait.md §5). The hand's pickup
lifts it (the pickup gate's stimulus). Two runs of the same script are bit-identical (P2).

Needs numpy, MuJoCo 3.10, onnxruntime, the official MJCF (``DUCKMIND_MICRODUCK_MJCF``) and the
shipped policies (``DUCKMIND_POLICY_DIR``, else :func:`duckmind.paths.policy_dir`): a render-check test.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("numpy")
pytest.importorskip("mujoco")
pytest.importorskip("onnxruntime")

from duckmind.world import headless_duck as hd  # noqa: E402

MJCF_DIR = os.environ.get("DUCKMIND_MICRODUCK_MJCF")
if not MJCF_DIR or not (Path(MJCF_DIR) / "robot_allcollisions.xml").is_file():
    pytest.skip("needs DUCKMIND_MICRODUCK_MJCF (the official MJCF)", allow_module_level=True)
if not (hd.policy_dir() / "velstand.onnx").is_file():
    pytest.skip(f"needs the shipped policies in {hd.policy_dir()}", allow_module_level=True)


def stood(x: float = 1.2, y: float = 0.3, yaw: float = 0.0) -> hd.HeadlessDuck:
    duck = hd.HeadlessDuck("default_home", keyframe="SIT", x=x, y=y, yaw=yaw)
    ok, _ = duck.stand(timeout=12.0)
    assert ok, "the duck did not stand up from SIT"
    return duck


def walk(duck: hd.HeadlessDuck, vx: float, seconds: float, vyaw: float = 0.0) -> None:
    for _ in range(int(round(seconds * hd.CONTROL_HZ))):
        duck.move(vx, 0.0, vyaw)  # duck-body streams at 50 Hz (the deadman is 500 ms)
        duck.tick()


def test_state_is_robotd_shaped_and_duck_body_parses_it() -> None:
    duck = stood()
    client = pytest.importorskip("duckmind.duckbody.robotd_client")
    s0 = duck.state()
    duck.tick()
    s1 = duck.state()
    assert s1["t_ns"] - s0["t_ns"] == 20_000_000, (
        "one tick is 20 ms of the synthetic monotonic clock"
    )
    for key in ("move", "head", "policy", "safety", "loop", "joints", "targets", "velocities",
                "currents_ma", "odom", "imu", "frames", "t_ns"):  # fmt: skip
        assert key in s1, key
    assert len(s1["joints"]) == len(s1["velocities"]) == len(s1["currents_ma"]) == 15
    assert s1["joints"][hd.MOUTH] == 0.0, "no jaw in the official MJCF"
    parsed = client.RobotState.from_wire(s1)
    assert parsed.policy == "walk" and not parsed.fallen
    assert parsed.camera is not None and parsed.tof is not None
    assert parsed.currents_ma is not None and sum(parsed.currents_ma) > 0
    json.dumps(s1)  # plain JSON, like the wire


def test_the_deadman_zeroes_the_twist_and_keeps_the_head() -> None:
    duck = stood()
    duck.move(0.4, 0.0, 0.0)
    duck.head((0.0, 0.2, 0.0, 0.0))
    duck.run(0.3)
    assert duck.state()["move"]["requested"][0] == 0.4
    duck.run(0.4)  # 0.7 s after the one intent: stale
    state = duck.state()
    assert state["move"]["requested"] == [0.0, 0.0, 0.0]
    assert state["move"].get("limited_by") == ["deadman"]
    assert duck.intent.head == (0.0, 0.2, 0.0, 0.0), "head is harmless when stale"


def test_walking_moves_the_physics_like_phase_1() -> None:
    duck = stood()
    t0 = duck.truth()
    o0 = duck.state()["odom"]["position"]
    walk(duck, 0.40, 6.0)
    t1 = duck.truth()
    o1 = duck.state()["odom"]["position"]
    truth = math.dist(t0["trunk"][:2], t1["trunk"][:2])
    odom = math.dist(o0[:2], o1[:2])
    assert 0.6 <= truth <= 1.3, f"6 s at vx 0.40 moved {truth:.3f} m (phase 1: ~0.16 m/s)"
    assert 0.6 <= odom / truth <= 1.4, f"odometry {odom:.3f} m against truth {truth:.3f} m"
    assert not duck.safety.fallen and duck.upright()


def test_only_sit_toggle_is_modelled() -> None:
    duck = stood()
    for skill in ("ground_pick", "kick_left", "roulade"):
        answer = duck.do(skill)
        assert answer["accepted"] is False and "not modelled" in answer["reason"]
    assert duck.do("sit_toggle")["accepted"] is True
    duck.run(4.0)
    assert duck.policy == "sit"


def test_look_points_the_head_with_absolute_joints() -> None:
    duck = stood()
    answer = duck.look(0.6, 0.0, -0.05)
    assert answer["clamped"] is False
    assert duck.intent.head == tuple(answer["head"][n] for n in hd.HEAD_JOINTS)


def test_the_hand_picks_the_duck_up() -> None:
    duck = stood()
    legs = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]
    z0 = duck.truth()["trunk"][2]
    load0 = sum(duck.state()["currents_ma"][i] for i in legs)
    answer = duck.hand({"do": "pickup"})
    assert answer.get("ok", True), answer
    held_at = None
    for _ in range(int(6.0 * hd.CONTROL_HZ)):
        duck.tick()
        if held_at is None and duck.truth()["held"]:
            held_at = duck.now()
    assert held_at is not None, "the hand never took the duck"
    assert duck.truth()["trunk"][2] > z0 + 0.15, "lifted about 30 cm"
    load = sum(duck.state()["currents_ma"][i] for i in legs)
    assert load < load0, f"held legs carry less ({load:.0f} vs {load0:.0f})"


def _digest(duck: hd.HeadlessDuck, script: list[tuple[int, Any]]) -> str:
    h = hashlib.sha256()
    plan = dict(script)
    for tick in range(250):
        if tick in plan:
            vx, vyaw = plan[tick]
            duck.move(vx, 0.0, vyaw)
        elif duck.intent.twist != (0.0, 0.0, 0.0):
            duck.move(*duck.intent.twist)  # keep streaming (the deadman)
        duck.tick()
        h.update(json.dumps(duck.state(), sort_keys=True).encode())
    return h.hexdigest()


def test_two_runs_of_one_script_are_identical() -> None:
    script = [(0, (0.4, 0.0)), (100, (0.35, 0.5)), (180, (0.0, 0.0))]
    a = _digest(stood(0.5, -0.5, 1.0), script)
    b = _digest(stood(0.5, -0.5, 1.0), script)
    assert a == b
