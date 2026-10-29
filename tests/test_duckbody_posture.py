"""The posture tap: trunk height from leg FK (checked against the official MJCF), stillness,
the camera pose, the 25 Hz stream to duck-eye."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from duckmind.duckbody.posture import SEATED_EXTRA_M, LegFk, PostureTap
from duckmind.duckbody.robotd_client import RobotState

MJCF_DIR = os.environ.get("DUCKMIND_MICRODUCK_MJCF")
needs_mjcf = pytest.mark.skipif(
    not MJCF_DIR or not (Path(MJCF_DIR) / "robot_walk.xml").exists(),
    reason="needs the official checkout (DUCKMIND_MICRODUCK_MJCF)",
)
# a standing tick of the headless twin (phase-3 calibration run): true trunk height 0.1164 m
STAND_JOINTS = [
    0.0042,
    -0.0808,
    -0.4505,
    0.0107,
    0.4501,
    0.2297,
    0.4447,
    -0.046,
    -0.0119,
    0.0,
    0.0342,
    0.086,
    0.4593,
    -0.0121,
    -0.4599,
]


def state(
    t_ns: int = 0,
    *,
    joints: list[float] | None = None,
    gyro: float = 0.0,
    head_vel: float = 0.0,
    policy: str = "walk",
    frames: dict | None = None,
) -> RobotState:
    velocities = [0.0] * 15
    velocities[7] = head_vel
    params = {
        "t_ns": t_ns,
        "policy": policy,
        "move": {"requested": [0, 0, 0], "applied": [0, 0, 0]},
        "safety": {"fallen": False, "limp": False, "gravity": [0.0112, 0.0016, -0.9999]},
        "joints": joints or STAND_JOINTS,
        "velocities": velocities,
        "currents_ma": [2.0] * 15,
        "odom": {"position": [0.1, 0.2, 0.117], "yaw": 0.3},
        "imu": {"gyro": [0.0, 0.0, gyro], "quat": [1.0, 0.0, 0.0, 0.0]},
    }
    if frames is not None:
        params["frames"] = frames
    return RobotState.from_wire(params)


@needs_mjcf
def test_leg_fk_puts_the_standing_trunk_where_the_twin_has_it() -> None:
    assert MJCF_DIR is not None
    fk = LegFk.from_mjcf(Path(MJCF_DIR) / "robot_walk.xml")
    s = state()
    assert fk.height_m(s.joints, s.down or (0, 0, -1)) == pytest.approx(0.1164, abs=0.003)
    seated = fk.height_m(s.joints, s.down or (0, 0, -1), seated=True)
    assert seated - fk.height_m(s.joints, s.down or (0, 0, -1)) == pytest.approx(SEATED_EXTRA_M)


@needs_mjcf
def test_the_tap_uses_fk_for_height_and_the_camera() -> None:
    assert MJCF_DIR is not None
    tap = PostureTap.from_mjcf(Path(MJCF_DIR) / "robot_walk.xml")
    sample = tap.step(state())
    assert sample.trunk_height_mm == pytest.approx(116, abs=3)
    assert sample.camera_pos is not None and sample.camera_pos[0] > 0.0  # in front of the trunk
    frames = {
        "camera": {"pos": [0.1, 0.0, 0.2], "quat": [1, 0, 0, 0]},
        "tof": {"pos": [0.1, 0.02, 0.2], "quat": [1, 0, 0, 0]},
    }
    sample = tap.step(state(20_000_000, frames=frames))
    assert sample.camera_pos == (0.1, 0.0, 0.2)  # robotd's own frames win over FK


def test_without_a_model_the_height_is_robotds_odometry() -> None:
    sample = PostureTap().step(state())
    assert sample.trunk_height_mm == 117 and sample.camera_pos is None


def test_still_needs_150_ms_of_calm_head_and_trunk() -> None:
    tap = PostureTap()
    stills = [tap.step(state(t * 20_000_000)).still for t in range(10)]
    assert stills[:7] == [False] * 7 and stills[8] and stills[9]  # 0.16 s ≥ 150 ms
    assert not tap.step(state(220_000_000, gyro=0.5)).still
    assert not tap.step(state(240_000_000, head_vel=0.3)).still


def test_every_other_sample_goes_to_the_eye_and_the_ring_is_bounded() -> None:
    tap = PostureTap()
    samples = [tap.step(state(t * 20_000_000)) for t in range(800)]  # 16 s
    assert [PostureTap.for_eye(s) for s in samples[:4]] == [True, False, True, False]
    assert samples[-1].seq == 799
    assert tap.ring[-1].t_ns - tap.ring[0].t_ns <= 10_000_000_000
    near = tap.at(15_001_000_000)
    assert near is not None and near.t_ns == 15_000_000_000
    assert tap.at(1_000_000) is None  # fell out of the ring
