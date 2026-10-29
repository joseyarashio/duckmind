"""The step counter (duckmind.duckbody.footfalls) against phase 1's validated counter.

``fixtures/phase3_body/walk_frames.json`` is 12 s of live ``robotd --sim`` robot.state (phase 1
``gait.jsonl``) with the counts phase 1's reference counter (``phase1/footfalls.py``, numpy)
gives on the same frames. The port reads the official MJCF and sole anchors by path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from duckmind.duckbody.footfalls import Footfalls, load_sole_points
from duckmind.duckbody.proprio import Proprio
from duckmind.duckbody.robotd_client import RobotState
from duckmind.duckbody.tof_reproject import Tree

MJCF_DIR = os.environ.get("DUCKMIND_MICRODUCK_MJCF")
FIXTURE = Path(__file__).parent / "fixtures" / "phase3_body" / "walk_frames.json"
pytestmark = pytest.mark.skipif(
    not MJCF_DIR or not (Path(MJCF_DIR) / "robot_groundcontact.xml").exists(),
    reason="needs the official checkout (DUCKMIND_MICRODUCK_MJCF)",
)


def states() -> list[RobotState]:
    data = json.loads(FIXTURE.read_text())
    out = []
    for f in data["frames"]:
        out.append(
            RobotState.from_wire(
                {
                    "t_ns": 10**10 + int(round(f["t"] * 1e9)),
                    "policy": f["policy"],
                    "move": {"requested": f["applied"], "applied": f["applied"]},
                    "safety": {"fallen": f["fallen"], "limp": False},
                    "joints": f["joints"],
                    "odom": {"position": [f["odom"][0], f["odom"][1], 0.117], "yaw": f["odom"][2]},
                    "imu": {"gyro": [0, 0, 0], "quat": f["quat"]},
                }
            )
        )
    return out


def tree() -> Tree:
    assert MJCF_DIR is not None
    return Tree.from_mjcf(Path(MJCF_DIR) / "robot_groundcontact.xml")


def anchors() -> Path:
    assert MJCF_DIR is not None
    return Path(MJCF_DIR).resolve().parents[4] / "microduck" / "odometry" / "src" / "anchors.rs"


def test_the_port_counts_what_phase_1_counts() -> None:
    reference = json.loads(FIXTURE.read_text())["reference"]
    path = anchors()
    if not path.exists():
        pytest.skip("the official odometry anchors are not in this checkout")
    counter = Footfalls(tree(), load_sole_points(path))
    for s in states():
        counter.update(s)
    assert counter.footfalls == reference["footfalls"]
    assert abs(counter.steps - reference["steps"]) <= 1  # the last footfall's 0.25 s window


def test_the_foot_site_alone_is_close() -> None:
    reference = json.loads(FIXTURE.read_text())["reference"]
    counter = Footfalls(tree())
    for s in states():
        counter.update(s)
    assert abs(counter.footfalls - reference["footfalls"]) <= 1


def test_proprio_reports_the_steps() -> None:
    p = Proprio(feet=Footfalls(tree()))
    last = None
    for s in states():
        last = p.step(s)
    assert last is not None and last.view.steps >= 10
