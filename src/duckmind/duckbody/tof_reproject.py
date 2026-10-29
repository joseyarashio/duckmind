"""ToF reprojection: the official ``kinematics::tof::Reprojector``, ported line by line (P4).

``tofd`` publishes slant ranges along 64 fixed beams in the sensor's own frame (+x forward along
the optical axis, +y left, +z up; row 0 the top of the grid, column 0 the sensor's left). What
duck-body wants is where each return sits in the trunk frame, with the two nuisances filtered
the way the official runtime does it (``microduck/kinematics/src/tof.rs``):

* **floor returns** — in the gravity-levelled frame, a beam looking below the horizon whose slant
  range times its downward component reaches ``FLOOR_SAFETY`` (0.85) of the sensor's height
  above the floor hit the floor, not an obstacle;
* **too-close returns** — a return whose levelled horizontal reach is under ``MIN_RANGE_M``
  (0.10 m) is the sensor's own crosstalk.

On the board (phase 3) the sensor's pose in the trunk comes from ``robot.state.frames.tof``
and the beams from ``robot.model.tof_beams`` (robotd publishes both), so :func:`project` takes
the pose. :class:`HeadFk` is the official head-chain FK (``kinematics/src/head.rs`` over the
``mjcf.rs`` tree reader), for callers that have only the head joints — the simulator and the
upstream property tests, which ``tests/test_reprojector_vectors.py`` re-expresses against the
official ``robot_walk.xml`` read by path (and cross-checks with ``fk_alpha.json`` at 1e-6).

Quaternions are Hamilton, scalar first ``(w, x, y, z)``; ``a * b`` applies ``b`` first. Pure
Python: no numpy, no MuJoCo. The drop-off (cliff) rule of spec §5.1 is not part of the port and
lives in its own function when duck-body gets it (phase 3).
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

Vec3 = tuple[float, float, float]

ROWS = 8
COLS = 8
ZONES = ROWS * COLS
FOV_DEG = 45.0  # per axis, VL53L5CX / VL53L8CX
FLOOR_SAFETY = 0.85
MIN_RANGE_M = 0.10
# tof::Frame::zone: status codes ST documents as a usable range, and "nothing in range"
STATUS_VALID = (5, 9)
STATUS_NO_TARGET = 255


# ── the rigid-body algebra (kinematics/src/math.rs) ─────────────────────────────


@dataclass(frozen=True)
class Quat:
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def normalized(self) -> Quat:
        """Unit length; a degenerate input becomes the identity."""
        n = math.sqrt(self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z)
        if n < 1e-12:
            return IDENTITY
        return Quat(self.w / n, self.x / n, self.y / n, self.z / n)

    @staticmethod
    def from_axis_angle(axis: Sequence[float], angle: float) -> Quat:
        """Rotation of ``angle`` radians about a **unit** axis."""
        s, c = math.sin(0.5 * angle), math.cos(0.5 * angle)
        return Quat(c, axis[0] * s, axis[1] * s, axis[2] * s)

    def rotate(self, v: Sequence[float]) -> Vec3:
        """``q v q⁻¹`` in the cross-product form."""
        vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
        tx = 2.0 * (self.y * vz - self.z * vy)
        ty = 2.0 * (self.z * vx - self.x * vz)
        tz = 2.0 * (self.x * vy - self.y * vx)
        return (
            vx + self.w * tx + self.y * tz - self.z * ty,
            vy + self.w * ty + self.z * tx - self.x * tz,
            vz + self.w * tz + self.x * ty - self.y * tx,
        )

    def conjugate(self) -> Quat:
        return Quat(self.w, -self.x, -self.y, -self.z)

    def wxyz(self) -> tuple[float, float, float, float]:
        return (self.w, self.x, self.y, self.z)

    def __mul__(self, b: Quat) -> Quat:
        a = self
        return Quat(
            a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
            a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        )


IDENTITY = Quat()


@dataclass(frozen=True)
class Pose:
    """A rigid transform: rotate by ``quat``, then translate by ``pos``."""

    pos: Vec3 = (0.0, 0.0, 0.0)
    quat: Quat = IDENTITY

    def transform_point(self, p: Sequence[float]) -> Vec3:
        r = self.quat.rotate(p)
        return (self.pos[0] + r[0], self.pos[1] + r[1], self.pos[2] + r[2])

    def __mul__(self, b: Pose) -> Pose:
        return Pose(self.transform_point(b.pos), self.quat * b.quat)


# ── the beams ───────────────────────────────────────────────────────────────────


def beams() -> tuple[Vec3, ...]:
    """The 64 unit beam directions in the sensor frame, row-major: zone centres evenly spread
    over the field of view with a half-zone inset (row 0 the top, column 0 the sensor's left)."""
    half = math.radians(FOV_DEG / 2.0 - FOV_DEG / COLS / 2.0)
    step = 2.0 * half / (COLS - 1)
    out = []
    for i in range(ZONES):
        elevation = half - (i // COLS) * step
        azimuth = half - (i % COLS) * step
        out.append(
            (
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth),
                math.sin(elevation),
            )
        )
    return tuple(out)


BEAMS = beams()


# ── the zones ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Zone:
    """What one zone's return turned out to be: ``empty`` (no usable return), ``too_close``
    (inside the crosstalk band), ``floor`` (the beam reached the floor first; ``point`` in the
    trunk frame) or ``hit`` (something is there: ``point`` in the trunk frame, ``range`` its
    levelled horizontal distance from the trunk origin's vertical axis, metres)."""

    kind: str
    point: Vec3 | None = None
    range: float | None = None


EMPTY = Zone("empty")
TOO_CLOSE = Zone("too_close")


@dataclass(frozen=True)
class Posture:
    """The trunk's own tilt and height: projected gravity in the trunk frame (upright is about
    ``(0, 0, -1)``) and the measured trunk height above the floor (``None``: the model's
    standing rest height)."""

    gravity: Vec3 = (0.0, 0.0, -1.0)
    trunk_height_m: float | None = None


def level_from_gravity(gravity: Sequence[float]) -> Quat:
    """The rotation taking the trunk-frame gravity to straight down; the identity for a gravity
    too small to trust."""
    n = math.sqrt(sum(float(g) * float(g) for g in gravity))
    if n < 0.5:
        return IDENTITY
    g = (gravity[0] / n, gravity[1] / n, gravity[2] / n)
    down = (0.0, 0.0, -1.0)
    axis = (
        g[1] * down[2] - g[2] * down[1],
        g[2] * down[0] - g[0] * down[2],
        g[0] * down[1] - g[1] * down[0],
    )
    s = math.sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2])
    c = g[0] * down[0] + g[1] * down[1] + g[2] * down[2]
    if s < 1e-9:
        if c > 0.0:
            return IDENTITY
        return Quat.from_axis_angle((1.0, 0.0, 0.0), math.pi)  # hanging upside down
    return Quat.from_axis_angle((axis[0] / s, axis[1] / s, axis[2] / s), math.atan2(s, c))


def ranges_from_frame(distance_mm: Sequence[int], status: Sequence[int]) -> list[float | None]:
    """A ``tof.frame``'s zones as slant ranges (m), ``None`` where the zone is not a usable
    range (``tof::Frame::zone``: status 5 or 9 with a positive distance)."""
    out: list[float | None] = []
    for i in range(len(distance_mm)):
        code = int(status[i]) if i < len(status) else STATUS_NO_TARGET
        mm = int(distance_mm[i])
        out.append(mm / 1000.0 if code in STATUS_VALID and mm > 0 else None)
    return out


def project(
    ranges_m: Sequence[float | None],
    sensor: Pose,
    posture: Posture,
    trunk_height_m: float,
    beam_dirs: Sequence[Sequence[float]] = BEAMS,
) -> list[Zone]:
    """``Reprojector::project`` with the sensor's pose in the trunk frame given (the board's
    ``frames.tof``). ``trunk_height_m``: the model's standing height, used when the posture
    has no measured one."""
    level = level_from_gravity(posture.gravity)
    sensor_level = level.rotate(sensor.pos)
    height = posture.trunk_height_m if posture.trunk_height_m is not None else trunk_height_m
    above_floor = sensor_level[2] + height
    floor_threshold = above_floor * FLOOR_SAFETY
    zones = [EMPTY] * len(ranges_m)
    for i, r in enumerate(ranges_m):
        if r is None:
            continue
        direction = sensor.quat.rotate(beam_dirs[i])
        dir_level = level.rotate(direction)
        downward = -dir_level[2]
        point = (
            sensor.pos[0] + r * direction[0],
            sensor.pos[1] + r * direction[1],
            sensor.pos[2] + r * direction[2],
        )
        if above_floor > 0.0 and downward > 0.0 and r * downward >= floor_threshold:
            zones[i] = Zone("floor", point)
            continue
        horizontal = r * math.sqrt(dir_level[0] * dir_level[0] + dir_level[1] * dir_level[1])
        if horizontal < MIN_RANGE_M:
            zones[i] = TOO_CLOSE
            continue
        zones[i] = Zone("hit", point, horizontal)
    return zones


# ── the head chain (kinematics/src/mjcf.rs, lib.rs, head.rs) ────────────────────

# MJCF `head_camera` site → cv2 camera frame; sensor frame → cv2 camera frame (head.rs)
SITE_TO_CV2 = Quat(0.5, -0.5, 0.5, -0.5)
SENSOR_IN_CV2_Q = Quat(0.5, 0.5, -0.5, 0.5)
HEAD_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")


@dataclass(frozen=True)
class _Link:
    rest: Pose
    joint: tuple[int, Vec3] | None


@dataclass
class Tree:
    """The kinematic tree of an MJCF: bodies, hinge joints and sites, rooted at ``trunk_base``
    pinned to the identity (every pose is in the trunk frame)."""

    joint_names: list[str] = field(default_factory=list)
    site_names: list[str] = field(default_factory=list)
    chains: list[tuple[_Link, ...]] = field(default_factory=list)
    trunk_height_m: float = 0.0

    @classmethod
    def from_mjcf(cls, source: Path | str) -> Tree:
        """Parse a file (or MJCF text starting with ``<``) the way ``mjcf.rs`` does: only
        ``pos`` and ``quat`` (normalised) place an element; a body's first ``<joint>`` child is
        its hinge (axis normalised, default +z)."""
        text = str(source)
        root = ET.fromstring(text) if text.lstrip().startswith("<") else ET.parse(text).getroot()
        worldbody = next((c for c in root if c.tag == "worldbody"), None)
        if worldbody is None:
            raise ValueError("missing <worldbody> in MJCF")
        trunk = next(
            (c for c in worldbody if c.tag == "body" and c.get("name") == "trunk_base"), None
        )
        if trunk is None:
            raise ValueError('missing <body name="trunk_base"> in MJCF')
        tree = cls(trunk_height_m=(_floats(trunk, "pos", 3) or [0.0, 0.0, 0.0])[2])
        # bodies: (parent index, rest pose, joint) in tree order; index 0 is the root
        bodies: list[tuple[int | None, Pose, tuple[int, Vec3] | None]] = [(None, Pose(), None)]
        sites: list[tuple[str, int, Pose]] = []

        def collect_sites(node: ET.Element, body: int) -> None:
            for child in node:
                if child.tag == "site" and child.get("name"):
                    sites.append((str(child.get("name")), body, _rest(child)))

        def walk(node: ET.Element, parent: int) -> None:
            if node.get("name") is None:
                raise ValueError(f"body under body index {parent} has no name")
            joint_node = next((c for c in node if c.tag == "joint"), None)
            joint = None
            if joint_node is not None:
                axis = _normalize(_floats(joint_node, "axis", 3) or [0.0, 0.0, 1.0])
                tree.joint_names.append(joint_node.get("name") or "")
                joint = (len(tree.joint_names) - 1, axis)
            index = len(bodies)
            bodies.append((parent, _rest(node), joint))
            collect_sites(node, index)
            for child in node:
                if child.tag == "body":
                    walk(child, index)

        collect_sites(trunk, 0)
        for child in trunk:
            if child.tag == "body":
                walk(child, 0)
        for name, body, rest in sites:
            chain: list[_Link] = []
            at: int | None = body
            while at is not None:
                parent, body_rest, joint = bodies[at]
                if parent is not None:
                    chain.append(_Link(body_rest, joint))
                at = parent
            chain.reverse()
            chain.append(_Link(rest, None))
            tree.site_names.append(name)
            tree.chains.append(tuple(chain))
        return tree

    def joint_index(self, name: str) -> int | None:
        return self.joint_names.index(name) if name in self.joint_names else None

    def site(self, name: str) -> int | None:
        return self.site_names.index(name) if name in self.site_names else None

    def site_pose(self, site: int, angles: Sequence[float]) -> Pose:
        """The site's pose in the trunk frame at these joint angles (every joint)."""
        if len(angles) != len(self.joint_names):
            raise ValueError("the angles must cover every joint")
        t = Pose()
        for link in self.chains[site]:
            t = t * link.rest
            if link.joint is not None:
                index, axis = link.joint
                t = Pose(t.pos, t.quat * Quat.from_axis_angle(axis, float(angles[index])))
        return t


def _floats(node: ET.Element, attr: str, count: int) -> list[float] | None:
    raw = node.get(attr)
    if raw is None:
        return None
    values = [float(tok) for tok in raw.split()]
    if len(values) != count:
        raise ValueError(f"<{node.tag} {attr}> must have {count} components, got {len(values)}")
    return values


def _normalize(v: Sequence[float]) -> Vec3:
    n = math.sqrt(sum(c * c for c in v))
    if n < 1e-12:
        return (0.0, 0.0, 1.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def _rest(node: ET.Element) -> Pose:
    pos = _floats(node, "pos", 3) or [0.0, 0.0, 0.0]
    quat = _floats(node, "quat", 4)
    q = Quat(*quat).normalized() if quat is not None else IDENTITY
    return Pose((pos[0], pos[1], pos[2]), q)


class HeadFk:
    """The head chain (``head.rs``): the ToF sensor's and the camera's pose in the trunk frame
    from the four head joints ``(neck_pitch, head_pitch, head_yaw, head_roll)``, every other
    joint at zero."""

    def __init__(self, tree: Tree) -> None:
        self.tree = tree
        camera = tree.site("head_camera")
        if camera is None:
            raise ValueError("the model has no head_camera site")
        self.camera = camera
        self.tof = tree.site("tof")
        joints = [tree.joint_index(name) for name in HEAD_JOINTS]
        if any(j is None for j in joints):
            raise ValueError("the model lacks the head joints")
        self.joints = [int(j) for j in joints if j is not None]

    @classmethod
    def from_mjcf(cls, source: Path | str) -> HeadFk:
        return cls(Tree.from_mjcf(source))

    @property
    def trunk_height_m(self) -> float:
        return self.tree.trunk_height_m

    def _site(self, site: int, joints: Sequence[float]) -> Pose:
        angles = [0.0] * len(self.tree.joint_names)
        for index, angle in zip(self.joints, joints, strict=True):
            angles[index] = float(angle)
        return self.tree.site_pose(site, angles)

    def camera_in_trunk_cv2(self, joints: Sequence[float]) -> Pose:
        site = self._site(self.camera, joints)
        return Pose(site.pos, site.quat * SITE_TO_CV2)

    def tof_in_trunk(self, joints: Sequence[float]) -> Pose:
        """Rotating a sensor-frame vector (forward, left, up) by the quat expresses it in the
        trunk; anchored on the ``tof`` site when the model has one, else the camera."""
        if self.tof is not None:
            site = self._site(self.tof, joints)
            return Pose(site.pos, site.quat * SITE_TO_CV2 * SENSOR_IN_CV2_Q)
        cam = self.camera_in_trunk_cv2(joints)
        return Pose(cam.pos, cam.quat * SENSOR_IN_CV2_Q)


class Reprojector:
    """``kinematics::tof::Reprojector``: an FK for the head chain, the beams and the model's
    standing trunk height."""

    FLOOR_SAFETY = FLOOR_SAFETY
    MIN_RANGE_M = MIN_RANGE_M

    def __init__(self, fk: HeadFk) -> None:
        self.fk = fk
        self.beams = BEAMS
        self.trunk_height_m = fk.trunk_height_m

    @classmethod
    def from_mjcf(cls, source: Path | str) -> Reprojector:
        return cls(HeadFk.from_mjcf(source))

    def sensor_in_trunk(self, head_joints: Sequence[float]) -> Pose:
        return self.fk.tof_in_trunk(head_joints)

    def project(
        self,
        ranges_m: Sequence[float | None],
        head_joints: Sequence[float],
        posture: Posture | None = None,
    ) -> list[Zone]:
        return project(
            ranges_m,
            self.fk.tof_in_trunk(head_joints),
            posture or Posture(),
            self.trunk_height_m,
            self.beams,
        )
