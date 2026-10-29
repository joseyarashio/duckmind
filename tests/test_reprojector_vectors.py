"""P4: duck-body's ToF reprojection against the official ``kinematics::tof::Reprojector``.

The six upstream unit tests (``microduck/kinematics/src/tof.rs``) re-expressed on the Python port
:mod:`duckmind.duckbody.tof_reproject`, with the official ``robot_walk.xml`` read by path at test
time; the port's head-chain FK is checked against the upstream MuJoCo fixture
(``kinematics/tests/fixtures/fk_alpha.json``, 64 poses × 6 sites) at the upstream's 1e-6, and the
beams, constants and frame interpretation against the upstream source text. Pure Python: runs in
the standard check whenever the official checkout is there (``DUCKMIND_MICRODUCK_MJCF``).
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
from pathlib import Path

import pytest

from duckmind.duckbody import tof_reproject as t


def _upstream() -> Path | None:
    raw = os.environ.get("DUCKMIND_UPSTREAM")
    if raw:
        return Path(raw)
    mjcf = os.environ.get("DUCKMIND_MICRODUCK_MJCF")
    if mjcf:
        parents = Path(mjcf).expanduser().resolve().parents
        if len(parents) > 4 and (parents[4] / "microduck").is_dir():
            return parents[4]
    return None


UP = _upstream()
KIN = UP / "microduck" / "kinematics" if UP else None
needs_upstream = pytest.mark.skipif(
    KIN is None or not (KIN / "src" / "tof.rs").is_file(),
    reason="needs the official checkout (DUCKMIND_MICRODUCK_MJCF)",
)

LEVEL = (0.0, 0.0, 0.0, 0.0)
CENTRE = 3 * t.COLS + 3


def one_return(zone: int, r: float) -> list[float | None]:
    ranges: list[float | None] = [None] * t.ZONES
    ranges[zone] = r
    return ranges


@pytest.fixture(scope="module")
def rp() -> t.Reprojector:
    assert KIN is not None
    return t.Reprojector.from_mjcf(KIN / "assets" / "alpha" / "robot_walk.xml")


# ── the six upstream tests ─────────────────────────────────────────────────────


@needs_upstream
def test_a_forward_return_lands_a_metre_ahead(rp: t.Reprojector) -> None:
    zones = rp.project(one_return(CENTRE, 1.0), LEVEL, t.Posture())
    zone = zones[CENTRE]
    assert zone.kind == "hit", zone
    assert zone.point is not None and zone.range is not None
    sensor = rp.sensor_in_trunk(LEVEL)
    assert 0.8 <= zone.point[0] <= 1.2
    assert abs(zone.point[1]) < 0.3
    assert abs(zone.point[2] - sensor.pos[2]) < 0.3
    assert 0.8 <= zone.range <= 1.05


@needs_upstream
def test_the_floor_is_floor_and_a_thing_before_it_is_not(rp: t.Reprojector) -> None:
    looking_down = (0.0, 0.3, 0.0, 0.0)
    zone = 6 * t.COLS + 3
    sensor = rp.sensor_in_trunk(looking_down)
    above_floor = sensor.pos[2] + rp.trunk_height_m
    direction = sensor.quat.rotate(rp.beams[zone])
    assert direction[2] < -0.3, "the test beam must look well below the horizon"
    to_floor = above_floor / -direction[2]
    at_floor = rp.project(one_return(zone, to_floor), looking_down, t.Posture())
    assert at_floor[zone].kind == "floor", at_floor[zone]
    point = at_floor[zone].point
    assert point is not None and abs(point[2] + rp.trunk_height_m) < 0.02
    interrupted = rp.project(one_return(zone, to_floor * 0.75), looking_down, t.Posture())
    assert interrupted[zone].kind == "hit", interrupted[zone]


@needs_upstream
def test_an_upward_beam_is_never_the_floor(rp: t.Reprojector) -> None:
    zones = rp.project(one_return(3, 4.0), LEVEL, t.Posture())
    assert zones[3].kind == "hit", zones[3]


@needs_upstream
def test_the_noise_band_and_the_silence_are_named(rp: t.Reprojector) -> None:
    zones = rp.project(one_return(CENTRE, 0.05), LEVEL, t.Posture())
    assert zones[CENTRE] == t.TOO_CLOSE
    assert zones[0] == t.EMPTY


@needs_upstream
def test_a_leaning_trunk_makes_a_forward_beam_floor(rp: t.Reprojector) -> None:
    lean = t.Quat.from_axis_angle((0.0, 1.0, 0.0), 0.6)
    gravity = lean.conjugate().rotate((0.0, 0.0, -1.0))
    leaned = t.Posture(gravity=gravity)
    sensor = rp.sensor_in_trunk(LEVEL)
    level = t.level_from_gravity(gravity)
    direction = level.rotate(sensor.quat.rotate(rp.beams[CENTRE]))
    assert direction[2] < -0.3, "the leaned beam must point at the floor"
    above = level.rotate(sensor.pos)[2] + rp.trunk_height_m
    to_floor = above / -direction[2]
    assert rp.project(one_return(CENTRE, to_floor), LEVEL, leaned)[CENTRE].kind == "floor"
    upright = rp.project(one_return(CENTRE, to_floor), LEVEL, t.Posture())
    assert upright[CENTRE].kind == "hit", upright[CENTRE]


@needs_upstream
def test_the_head_pose_steers_the_points(rp: t.Reprojector) -> None:
    ahead = rp.project(one_return(CENTRE, 1.0), LEVEL, t.Posture())[CENTRE]
    left = rp.project(one_return(CENTRE, 1.0), (0.0, 0.0, 0.8, 0.0), t.Posture())[CENTRE]
    assert ahead.kind == left.kind == "hit"
    assert ahead.point is not None and left.point is not None
    assert left.point[1] > ahead.point[1] + 0.3


@needs_upstream
def test_every_upstream_reprojector_test_is_re_expressed_here() -> None:
    """A new upstream test must be ported (or this list updated after reading it)."""
    assert KIN is not None
    source = (KIN / "src" / "tof.rs").read_text()
    upstream = set(re.findall(r"#\[test\]\s*fn (\w+)\(", source))
    here = {
        name.removeprefix("test_")
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    }
    assert upstream, "found no #[test] in tof.rs"
    assert upstream <= here, f"not ported: {sorted(upstream - here)}"


# ── the FK, the beams and the constants against upstream ───────────────────────


@needs_upstream
def test_fk_matches_the_upstream_mujoco_fixture_to_1e_6() -> None:
    assert KIN is not None
    tree = t.Tree.from_mjcf(KIN / "assets" / "alpha" / "robot_walk.xml")
    fixture = json.loads((KIN / "tests" / "fixtures" / "fk_alpha.json").read_text())
    assert fixture["samples"] and "tof" in fixture["site_names"]
    worst = 0.0
    for sample in fixture["samples"]:
        angles = [0.0] * len(tree.joint_names)
        for name, angle in sample["joints"].items():
            index = tree.joint_index(name)
            assert index is not None, name
            angles[index] = angle
        for site_name, expected in sample["sites"].items():
            site = tree.site(site_name)
            assert site is not None, site_name
            got = tree.site_pose(site, angles)
            worst = max(worst, *(abs(a - b) for a, b in zip(expected["pos"], got.pos)))
            q = got.quat.wxyz()
            err = min(max(abs(a - s * b) for a, b in zip(expected["quat"], q)) for s in (1.0, -1.0))
            worst = max(worst, err)
    assert worst < 1e-6


@needs_upstream
def test_the_constants_are_the_upstream_ones() -> None:
    assert KIN is not None
    tof = (KIN / "src" / "tof.rs").read_text()
    head = (KIN / "src" / "head.rs").read_text()

    def const(text: str, name: str) -> float:
        found = re.search(rf"const {name}: \w+ = ([0-9.]+);", text)
        assert found, f"{name} not found upstream"
        return float(found.group(1))

    assert const(tof, "FOV_DEG") == t.FOV_DEG
    assert const(tof, "FLOOR_SAFETY") == t.FLOOR_SAFETY
    assert const(tof, "MIN_RANGE_M") == t.MIN_RANGE_M
    assert const(tof, "ROWS") == t.ROWS and const(tof, "COLS") == t.COLS

    def quat(name: str) -> tuple[float, ...]:
        found = re.search(rf"const {name}: Quat = Quat::new\(([^)]*)\);", head)
        assert found, f"{name} not found upstream"
        return tuple(float(v) for v in found.group(1).split(","))

    assert quat("SITE_TO_CV2") == t.SITE_TO_CV2.wxyz()
    assert quat("SENSOR_IN_CV2_Q") == t.SENSOR_IN_CV2_Q.wxyz()
    joints = re.search(r"const HEAD_JOINTS: \[&str; 4\] = (\[[^\]]*\]);", head)
    assert joints and tuple(ast.literal_eval(joints.group(1))) == t.HEAD_JOINTS


@needs_upstream
def test_the_beams_are_the_official_simulator_s_beams() -> None:
    """The same 64 directions as the official ``sim/tof.py`` builds (its zone centres, row 0 the
    top, column 0 the sensor's left), recomputed here from its own recipe."""
    assert UP is not None
    sim = (UP / "microduck_rl" / "src" / "mjlab_microduck" / "sim" / "tof.py").read_text()
    assert "elevation = -centres[row]" in sim and "azimuth = -centres[col]" in sim
    half = math.radians(45.0) / 2.0
    edges = [-half + 2 * half * i / 8 for i in range(9)]
    centres = [(edges[i] + edges[i + 1]) / 2 for i in range(8)]
    for row in range(8):
        for col in range(8):
            el, az = -centres[row], -centres[col]
            want = (math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))
            got = t.BEAMS[row * 8 + col]
            assert max(abs(a - b) for a, b in zip(want, got)) < 1e-12


@needs_upstream
def test_frame_zones_follow_the_tof_crate() -> None:
    assert UP is not None
    lib = (UP / "microduck" / "tof" / "src" / "lib.rs").read_text()
    valid = re.search(r"pub const STATUS_VALID: \[u8; 2\] = \[(\d+), (\d+)\];", lib)
    empty = re.search(r"pub const STATUS_NO_TARGET: u8 = (\d+);", lib)
    assert valid and empty
    assert tuple(int(v) for v in valid.groups()) == t.STATUS_VALID
    assert int(empty.group(1)) == t.STATUS_NO_TARGET
    ranges = t.ranges_from_frame([1000, 0, 500, 700, -3, 250], [5, 5, 9, 4, 9, 255])
    assert ranges == [1.0, None, 0.5, None, None, None]


# ── the port without the checkout ──────────────────────────────────────────────


def test_level_from_gravity_edges() -> None:
    assert t.level_from_gravity((0.0, 0.0, -1.0)) == t.IDENTITY
    assert t.level_from_gravity((0.0, 0.0, -0.2)) == t.IDENTITY  # too small to trust
    flipped = t.level_from_gravity((0.0, 0.0, 1.0))
    assert flipped.rotate((0.0, 0.0, 1.0))[2] == pytest.approx(-1.0)
    tilted = t.level_from_gravity((0.3, -0.2, -0.9))
    g = (0.3, -0.2, -0.9)
    n = math.sqrt(sum(v * v for v in g))
    down = tilted.rotate(tuple(v / n for v in g))
    assert down == pytest.approx((0.0, 0.0, -1.0), abs=1e-12)


def test_project_with_a_given_sensor_pose() -> None:
    sensor = t.Pose((0.08, 0.02, 0.13), t.IDENTITY)
    zones = t.project(one_return(CENTRE, 1.0), sensor, t.Posture(), 0.12)
    assert zones[CENTRE].kind == "hit" and zones[CENTRE].range == pytest.approx(
        1.0 * math.cos(math.radians(2.8125)), rel=0.01
    )
    assert sum(z.kind == "empty" for z in zones) == 63


def test_beams_are_unit_and_ordered() -> None:
    assert len(t.BEAMS) == 64
    for beam in t.BEAMS:
        assert math.sqrt(sum(c * c for c in beam)) == pytest.approx(1.0)
    top_left, bottom_right = t.BEAMS[0], t.BEAMS[63]
    assert top_left[2] > 0 and top_left[1] > 0  # row 0 up, column 0 left (+y)
    assert bottom_right[2] < 0 and bottom_right[1] < 0


def test_duck_body_reads_sockets_never_the_simulator():
    """duck-body runs the same on the board: no world, no page, no physics engine."""
    import ast

    package = Path(t.__file__).resolve().parent
    forbidden = ("duckmind.world", "duckmind.simweb", "duckmind.duckcore", "mujoco", "mjlab")
    for path in package.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            for name in names:
                assert not name.startswith(forbidden), f"{path.name} imports {name}"


@pytest.mark.skipif(
    not os.environ.get("DUCKMIND_ROBOTD_SOCKET"),
    reason="opt-in like DUCKMIND_LIVE_SMOKE: set DUCKMIND_ROBOTD_SOCKET to a running robotd's socket",
)
def test_the_beams_and_the_sensor_pose_equal_a_running_robotd_s() -> None:
    """Bit for bit: ``robot.model`` from the official robotd (the Rust Reprojector's own beams,
    trunk height and ToF pose at head zero) against the port."""
    import socket

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(5.0)
        sock.connect(os.environ["DUCKMIND_ROBOTD_SOCKET"])
        stream = sock.makefile("rwb")
        stream.write(b'{"jsonrpc":"2.0","id":1,"method":"hello","params":{"api_version":37}}\n')
        stream.write(b'{"jsonrpc":"2.0","id":2,"method":"robot.model"}\n')
        stream.flush()
        answers = [json.loads(stream.readline()) for _ in range(2)]
    model = next(a for a in answers if a.get("id") == 2)["result"]
    assert [tuple(b) for b in model["tof_beams"]] == list(t.BEAMS)
    assert model["tof_fov_deg"] == t.FOV_DEG and model["head_joints"] == list(t.HEAD_JOINTS)
    if KIN is not None:
        fk = t.HeadFk.from_mjcf(KIN / "assets" / "alpha" / "robot_walk.xml")
        zero = fk.tof_in_trunk(LEVEL)
        assert model["trunk_height_m"] == fk.trunk_height_m
        assert tuple(model["frames_at_zero"]["tof"]["pos"]) == zero.pos
        assert tuple(model["frames_at_zero"]["tof"]["quat"]) == zero.quat.wxyz()
