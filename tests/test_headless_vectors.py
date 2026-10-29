"""Upstream test vectors for the headless robotd's pure ports (spec §17 期3 "test vectors"):

* ``duck_control::safety`` — the fall verdict (``duck-control/src/safety.rs`` tests
  a_brief_tilt_is_not_a_fall, a_sustained_tilt_is_a_fall, an_unconverged_imu_cannot_declare_a_fall,
  a_converged_imu_still_detects_a_fall);
* ``kinematics::head`` — ``HeadFk::look_at`` (``kinematics/src/head.rs`` tests
  poses_are_finite_and_unit, head_pitch_moves_the_camera, alpha_head_axis_conventions,
  look_at_actually_looks_at_the_target, look_at_keeps_the_neck_it_was_given,
  a_target_behind_the_robot_clamps_at_the_yaw_limit);
* ``odometry`` — contact odometry (``odometry/src/lib.rs`` tests alpha_anchor_sets_lie_on_the_sole,
  standing_still_stays_at_the_origin, yaw_follows_the_imu, the_mouth_moves_no_odometry,
  the_anchor_switches_only_after_the_claim_holds,
  stance_leg_motion_translates_the_trunk_continuously).

Each upstream test is re-expressed with the same numbers and the same assertions against
:mod:`duckmind.world.headless_duck`, whose FK is the official ``robot_walk.xml`` (the kinematics
crate's ``Model::alpha``) read by path — nothing official is copied. The meta-test at the bottom
checks that every upstream test name is replayed here (``dump_reference_trajectory`` and
``update_cost_per_anchor_set`` are ``#[ignore]`` upstream: a print and a timing).
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

import pytest

from duckmind import paths
from duckmind.world import headless_duck as hd

MJCF_DIR = os.environ.get("DUCKMIND_MICRODUCK_MJCF")
UPSTREAM = paths.microduck_dir()
needs_mjcf = pytest.mark.skipif(
    not MJCF_DIR or not (Path(str(MJCF_DIR)) / "robot_walk.xml").is_file(),
    reason="needs DUCKMIND_MICRODUCK_MJCF (the official robot_walk.xml)",
)
DT = 0.020  # Duration::from_millis(20)


# ── duck-control/src/safety.rs ────────────────────────────────────────────────

UPRIGHT_Z = -1.0  # Sensors::default(): gravity [0, 0, -1]
ON_ITS_SIDE_Z = 0.0  # gravity [-1, 0, 0]


def test_a_brief_tilt_is_not_a_fall() -> None:
    s = hd.SafetyVerdict()
    s.observe(ON_ITS_SIDE_Z, 0.100)
    assert not s.fallen, "100ms is under the 200ms debounce"
    s.observe(UPRIGHT_Z, 0.020)
    s.observe(ON_ITS_SIDE_Z, 0.100)
    assert not s.fallen, "an upright sample must reset the accumulator"


def test_a_sustained_tilt_is_a_fall() -> None:
    s = hd.SafetyVerdict()
    for _ in range(11):
        s.observe(ON_ITS_SIDE_Z, DT)
    assert s.fallen


def test_an_unconverged_imu_cannot_declare_a_fall() -> None:
    s = hd.SafetyVerdict()
    for _ in range(50):
        s.observe(0.0, DT, imu_ready=False)  # gravity [0, 0, 0]: what a filter mid-way reads
    assert not s.fallen


def test_a_converged_imu_still_detects_a_fall() -> None:
    s = hd.SafetyVerdict()
    for _ in range(50):
        s.observe(0.0, DT, imu_ready=True)
    assert s.fallen


def test_the_verdict_clears_on_one_upright_sample() -> None:
    """safety.rs observe: 'one upright sample clears the accumulator' — and the verdict."""
    s = hd.SafetyVerdict()
    for _ in range(11):
        s.observe(ON_ITS_SIDE_Z, DT)
    assert s.fallen
    s.observe(UPRIGHT_Z, DT)
    assert not s.fallen


def test_the_threshold_is_about_sixty_degrees() -> None:
    """fall_gravity_z −0.5: a 55° tilt (gz −0.574) never falls, 65° (gz −0.423) does."""
    ok, down = hd.SafetyVerdict(), hd.SafetyVerdict()
    for _ in range(100):
        ok.observe(-math.cos(math.radians(55)), DT)
        down.observe(-math.cos(math.radians(65)), DT)
    assert not ok.fallen and down.fallen


# ── kinematics/src/head.rs ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def fk() -> hd.HeadFk:
    return hd.HeadFk.from_mjcf(Path(str(MJCF_DIR)) / "robot_walk.xml")


@pytest.fixture(scope="module")
def ranges() -> list[tuple[float, float] | None]:
    return hd.joint_ranges(Path(str(MJCF_DIR)) / "robot_walk.xml")


def _forward(fk: hd.HeadFk, joints: list[float]) -> tuple[float, float, float]:
    return fk.camera_in_trunk_cv2(joints).quat.rotate((0.0, 0.0, 1.0))  # cv2 +z: the optical axis


@needs_mjcf
def test_poses_are_finite_and_unit(fk: hd.HeadFk) -> None:
    pose = fk.camera_in_trunk_cv2([0.0] * 4)
    assert all(math.isfinite(v) for v in pose.pos)
    n = sum(v * v for v in pose.quat.wxyz())
    assert abs(n - 1.0) < 1e-9


@needs_mjcf
def test_head_pitch_moves_the_camera(fk: hd.HeadFk) -> None:
    level = fk.camera_in_trunk_cv2([0.0] * 4).pos
    tilted = fk.camera_in_trunk_cv2([0.0, 0.5, 0.0, 0.0]).pos
    assert max(abs(level[0] - tilted[0]), abs(level[2] - tilted[2])) > 1e-3


@needs_mjcf
def test_alpha_head_axis_conventions(fk: hd.HeadFk) -> None:
    assert _forward(fk, [0.0, 0.0, 0.3, 0.0])[1] > 0.1  # +head_yaw looks left (+y)
    assert _forward(fk, [0.0, 0.3, 0.0, 0.0])[2] < -0.1  # +head_pitch looks DOWN (-z)


@needs_mjcf
@pytest.mark.parametrize(
    "target",
    [
        (1.0, 0.0, 0.0),  # dead ahead
        (0.5, 0.5, 0.1),  # up-left
        (0.4, -0.6, -0.2),  # down-right
        (0.2, 0.6, 0.05),  # hard left
        (0.3, 0.0, -0.4),  # steeply down
    ],
)
def test_look_at_actually_looks_at_the_target(
    fk: hd.HeadFk, ranges: list[tuple[float, float] | None], target: tuple[float, float, float]
) -> None:
    gaze = hd.look_at(fk, target, 0.0, ranges)
    assert not gaze.clamped, f"reachable target {target} reported clamped"
    cam = fk.camera_in_trunk_cv2(gaze.joints)
    v = [target[i] - cam.pos[i] for i in range(3)]
    forward = cam.quat.rotate((0.0, 0.0, 1.0))
    norm = math.sqrt(sum(c * c for c in v))
    along = sum(v[i] * forward[i] for i in range(3)) / norm
    assert along > 1.0 - 1e-6, f"camera misses {target}"


@needs_mjcf
def test_look_at_keeps_the_neck_it_was_given(
    fk: hd.HeadFk, ranges: list[tuple[float, float] | None]
) -> None:
    gaze = hd.look_at(fk, (0.8, 0.2, -0.1), 0.3, ranges)
    assert abs(gaze.joints[0] - 0.3) < 1e-12
    assert gaze.joints[3] == 0.0, "head_roll stays level"
    assert not gaze.clamped


@needs_mjcf
def test_a_target_behind_the_robot_clamps_at_the_yaw_limit(
    fk: hd.HeadFk, ranges: list[tuple[float, float] | None]
) -> None:
    gaze = hd.look_at(fk, (-1.0, 0.0, 0.0), 0.0, ranges)
    assert gaze.clamped, "a 180° target must not claim a lock"
    yaw_range = ranges[2]
    assert yaw_range is not None
    assert abs(abs(gaze.joints[2]) - yaw_range[1]) < 1e-9


@needs_mjcf
def test_joint_ranges_are_read_from_the_joint_elements(
    ranges: list[tuple[float, float] | None],
) -> None:
    """mjcf.rs reads a joint's own ``range`` attribute; the official head travel (±170° yaw)."""
    assert all(r is not None for r in ranges)
    yaw = ranges[2]
    assert yaw is not None and abs(yaw[1] - math.radians(170)) < 1e-6


# ── odometry/src/lib.rs ────────────────────────────────────────────────────────


def _yaw_quat(yaw: float) -> list[float]:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def _roll_quat(roll: float) -> list[float]:
    return [math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0]


@pytest.fixture()
def odo() -> hd.Odometry:
    return hd.Odometry.from_mjcf(Path(str(MJCF_DIR)) / "robot_walk.xml")


def test_alpha_anchor_sets_lie_on_the_sole() -> None:
    for name in ("alpha4", "alpha16"):
        left, right = hd.ANCHOR_SETS[name]
        assert len(left) == len(right), name
        for p in (*left, *right):
            assert -0.021 <= p[0] <= 0.035, (name, p)
            assert -0.022 <= p[1] <= 0.022, (name, p)
            assert -0.002 <= p[2] <= 0.012, (name, p)
        for p in left:  # the right sole is the left one mirrored in Y (to the sampling step)
            assert any(
                abs(p[0] - r[0]) < 3e-4 and abs(p[1] + r[1]) < 3e-4 and abs(p[2] - r[2]) < 3e-4
                for r in right
            ), (name, p)
    assert len(hd.ALPHA4[0]) == 4
    assert len(hd.ALPHA16[0]) == 16


@needs_mjcf
def test_standing_still_stays_at_the_origin(odo: hd.Odometry) -> None:
    joints = [0.0] * len(hd.JOINT_NAMES)
    for _ in range(200):
        odo.update(joints, [1.0, 0.0, 0.0, 0.0])
    x, y, z = odo.position
    assert abs(x) < 1e-9 and abs(y) < 1e-9, f"drifted to ({x}, {y})"
    assert z > 0.02, f"trunk should stand above the ground, z = {z}"
    assert odo.yaw == 0.0


@needs_mjcf
def test_yaw_follows_the_imu(odo: hd.Odometry) -> None:
    joints = [0.0] * len(hd.JOINT_NAMES)
    for _ in range(10):
        odo.update(joints, _yaw_quat(0.3))
    assert abs(odo.yaw - 0.3) < 1e-9


@needs_mjcf
def test_the_mouth_moves_no_odometry() -> None:
    walk = Path(str(MJCF_DIR)) / "robot_walk.xml"
    still, chatting = hd.Odometry.from_mjcf(walk), hd.Odometry.from_mjcf(walk)
    quiet = [0.0] * len(hd.JOINT_NAMES)
    opened = list(quiet)
    opened[hd.MOUTH] = 42.0
    for _ in range(10):
        still.update(quiet, [1.0, 0.0, 0.0, 0.0])
        chatting.update(opened, [1.0, 0.0, 0.0, 0.0])
    assert still.position == chatting.position


@needs_mjcf
def test_the_anchor_switches_only_after_the_claim_holds(odo: hd.Odometry) -> None:
    joints = [0.0] * len(hd.JOINT_NAMES)
    for _ in range(10):
        odo.update(joints, [1.0, 0.0, 0.0, 0.0])
    settled = odo.anchor_foot
    away = 0.35 if settled == 0 else -0.35  # +roll about +x pushes −y (the right foot) down
    odo.update(joints, _roll_quat(away))
    odo.update(joints, [1.0, 0.0, 0.0, 0.0])
    assert odo.anchor_foot == settled, "one glitchy tick stole the anchor"
    for _ in range(5):
        odo.update(joints, _roll_quat(away))
    assert odo.anchor_foot != settled, "a held claim must win"


@needs_mjcf
def test_stance_leg_motion_translates_the_trunk_continuously(odo: hd.Odometry) -> None:
    joints = [0.0] * len(hd.JOINT_NAMES)
    for _ in range(10):
        odo.update(joints, [1.0, 0.0, 0.0, 0.0])
    hips = (hd.JOINT_NAMES.index("left_hip_pitch"), hd.JOINT_NAMES.index("right_hip_pitch"))
    start = odo.position
    last = start
    for i in range(1, 51):
        angle = 0.2 * (i / 50.0)
        joints[hips[0]] = angle
        joints[hips[1]] = angle
        odo.update(joints, [1.0, 0.0, 0.0, 0.0])
        now = odo.position
        step = math.sqrt(sum((now[k] - last[k]) ** 2 for k in range(3)))
        assert step < 0.02, f"tick {i} jumped {step} m"
        last = now
    moved = math.hypot(last[0] - start[0], last[1] - start[1])
    assert moved > 0.005, f"a 0.2 rad hip sweep moved the trunk only {moved} m"
    assert all(math.isfinite(v) for v in last)


# ── every upstream test is replayed (or deliberately not) ─────────────────────

REPLAYED = {
    "duck-control/src/safety.rs": {
        "a_brief_tilt_is_not_a_fall",
        "a_sustained_tilt_is_a_fall",
        "an_unconverged_imu_cannot_declare_a_fall",
        "a_converged_imu_still_detects_a_fall",
    },
    "kinematics/src/head.rs": {
        "poses_are_finite_and_unit",
        "head_pitch_moves_the_camera",
        "alpha_head_axis_conventions",
        "look_at_actually_looks_at_the_target",
        "look_at_keeps_the_neck_it_was_given",
        "a_target_behind_the_robot_clamps_at_the_yaw_limit",
    },
    "odometry/src/lib.rs": {
        "alpha_anchor_sets_lie_on_the_sole",
        "standing_still_stays_at_the_origin",
        "yaw_follows_the_imu",
        "the_mouth_moves_no_odometry",
        "the_anchor_switches_only_after_the_claim_holds",
        "stance_leg_motion_translates_the_trunk_continuously",
    },
}
# Upstream tests of the same files that are not about what the headless duck ports.
NOT_PORTED = {
    "duck-control/src/safety.rs": "the gain cache, joint clamps, limit reporting, reboot and the "
    "deadman's Limit bookkeeping belong to the servo layer (the headless deadman is exercised "
    "in test_headless_duck.py)",
    "kinematics/src/head.rs": "the_tof_pose_sits_on_the_mjcf_tof_site is replayed by "
    "test_reprojector_vectors.py (the ToF pose is the Reprojector port's)",
    "odometry/src/lib.rs": "dump_reference_trajectory and update_cost_per_anchor_set are "
    "#[ignore]: a print for the Python replica and a timing",
}
SAFETY_FALL_TESTS = {
    "a_brief_tilt_is_not_a_fall",
    "a_sustained_tilt_is_a_fall",
    "an_unconverged_imu_cannot_declare_a_fall",
    "a_converged_imu_still_detects_a_fall",
}


@pytest.mark.skipif(not UPSTREAM.is_dir(), reason="needs the upstream checkout (read-only)")
def test_every_upstream_vector_is_replayed() -> None:
    here = {name.removeprefix("test_") for name in globals() if name.startswith("test_")}
    for rel, names in REPLAYED.items():
        text = (UPSTREAM / rel).read_text(encoding="utf-8")
        tests = set(re.findall(r"#\[test\][^\n]*\n(?:\s*#\[[^\n]*\n)*\s*fn (\w+)", text))
        if rel.endswith("safety.rs"):  # the verdict's tests; the rest is the servo layer's
            assert SAFETY_FALL_TESTS <= tests, f"upstream renamed: {SAFETY_FALL_TESTS - tests}"
            tests = SAFETY_FALL_TESTS
        elif rel.endswith("head.rs"):
            tests -= {"the_tof_pose_sits_on_the_mjcf_tof_site"}
        elif rel.endswith("lib.rs"):
            tests -= {"dump_reference_trajectory", "update_cost_per_anchor_set"}
        missing = tests - names
        assert not missing, f"{rel}: upstream tests not replayed here: {sorted(missing)}"
        assert names <= here, f"{rel}: listed but not defined: {sorted(names - here)}"
    assert set(NOT_PORTED) == set(REPLAYED)
