"""duck-body sends robotd's upright cv2 camera frame; duck-eye turns it by the frame's rotate.

The numbers are the physics' own at the standing HOME head (default_home, 2026-09-27 probe
``camframe_probe.py``, measurement records not in the repo: HeadFk.camera_in_trunk_cv2 against the
world camera's sensor frame, equal after the turn at every head pose tried).
"""

from __future__ import annotations

import math

from duckmind.duckbody import protocol as B
from duckmind.eye import protocol as E
from duckmind.eye.frames import PairedPose


def _mat(q: tuple[float, float, float, float]) -> list[list[float]]:
    w, x, y, z = q
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


UPRIGHT = (0.5, -0.5, 0.5, -0.5)  # robotd cv2 at HOME: x → trunk −y (right), y → −z, z → +x
SENSOR = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]  # the world camera's sensor frame (rotate 90)


def test_robotd_s_frame_turned_by_the_rotate_is_the_sensor_frame() -> None:
    assert [[round(v) for v in row] for row in _mat(UPRIGHT)] == [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    pose = PairedPose(
        0,
        100,
        (0.064, 0.0, 0.128),
        UPRIGHT,
        (1.0, 0.0, 0.0, 0.0),
        (0.349, 0.349, 0.0, 0.0),
        0,
        "hold",
        (0, 0),
        cam_frame="cv2_upright",
    )
    turned = _mat(pose.camera_pose(90).cam_quat)
    assert all(
        math.isclose(a, b, abs_tol=1e-9) for r, s in zip(turned, SENSOR) for a, b in zip(r, s)
    )
    # a sample already in the sensor frame is left alone, whatever the rotate
    same = PairedPose(
        0, 100, (0.0, 0.0, 0.1), UPRIGHT, (1.0, 0.0, 0.0, 0.0), (0.0,) * 4, 0, "hold", (0, 0)
    )
    assert same.camera_pose(90).cam_quat == UPRIGHT


def test_duck_body_marks_its_camera_frame_and_duck_eye_reads_it() -> None:
    sample = B.PoseSample(
        seq=3,
        t_ns=5,
        trunk_height_mm=120,
        camera_pos=(0.064, 0.0, 0.128),
        camera_quat=UPRIGHT,
        tof_pos=None,
        tof_quat=None,
        imu_quat=(1.0, 0.0, 0.0, 0.0),
        gyro=None,
        head=(0.349, 0.349, 0.0, 0.0),
        head_vel=None,
        odom=(0.0, 0.0, 0.0),
        epoch=0,
        still=True,
        policy="stand",
    )
    wire = sample.to_wire()
    assert wire["cam_frame"] == "cv2_upright"
    parsed = E.PoseSample.from_json(wire)
    assert parsed.cam_frame == "cv2_upright"
    assert E.PoseSample.from_json(parsed.to_json()).cam_frame == "cv2_upright"
    wire.pop("cam_frame")
    assert E.PoseSample.from_json(wire).cam_frame == "sensor"
