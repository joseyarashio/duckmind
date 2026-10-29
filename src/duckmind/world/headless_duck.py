"""The duck without robotd: the official MJCF and the shipped policies, stepped in this process
on a composed home, answering in robotd's own words (physical-design spec §11.7, §14.7, §17 期3).

Phase 1's ``HeadlessDuck`` (a harness in the measurement records, not in the repo; HARNESS.md)
moved into the repo and put in a real home: :func:`compose` builds a home exactly as the world
server does (``homelayout.compose`` + the worldkit household + any extra layout edits, keyframes
rewritten),
and :class:`HeadlessDuck` drives the official duck in it through a port of robotd's control path:

* the command path of ``robotd/src/control.rs`` as phase 1 re-implemented it (twist and head EMA
  0.2 per 50 Hz tick; the sit flag in twist.vx; the sitstand network while seated or rising
  (1 s, flag 0), velstand otherwise; targets = HOME + scale·action, 0.9 velstand / 1.0 sitstand;
  low-pass legs 0.7 / head 0.5; ±π clamp; seated boot rises, standing boot ramps 2 s to HOME);
* robotd's deadman (``robotd-params``: a twist older than 500 ms is zeroed, the head is kept);
* the fall verdict of ``duck-control/src/safety.rs`` (:class:`SafetyVerdict`: projected gravity z
  above −0.5 held 200 ms; one upright sample clears it);
* ``robot.look`` as ``kinematics/src/head.rs`` ``HeadFk::look_at`` (:func:`look_at`), whose
  absolute joints robotd sends on as the head intent (``robotd/src/main.rs`` RobotLook);
* contact odometry, ``odometry/src/lib.rs`` (:class:`Odometry`, the ALPHA16 sole grid);
* ``robot.state`` shaped as ``duck-ipc-proto`` ``RobotState`` (:meth:`HeadlessDuck.state`): 15
  joints in wire order (the mouth at 9), velocities, the body server's current stand-in
  (|actuator force| × 100, ``sim/body_server.py``), IMU quat and gyro, projected gravity, odom,
  ``t_ns`` on a synthetic monotonic clock, and ``frames`` (camera and ToF in the trunk from the
  head FK the Reprojector port uses, ``duckbody/tof_reproject.HeadFk``).

Not modelled, and said so: the IMU filter, bus timing jitter and the daemon↔physics latency
(HARNESS.md: the live-vs-headless comparison), robotd's one-shot skills (``robot.do`` answers
``sit_toggle`` only; ``ground_pick`` and the kicks are refused with a reason), ``robot.mouth``
(no jaw in the official MJCF: joint 9 reads 0). Everything above the ``robot.state`` line — the
duck's own perception — never reads this module's truth: :meth:`HeadlessDuck.truth` is for the
evaluators only.

Deterministic: the same home, seed and command sequence give bit-identical physics (phase-3
survey probe). numpy, MuJoCo and onnxruntime are imported only when a duck is built; the pure
ports (:class:`SafetyVerdict`, :func:`look_at`, :class:`Odometry`) need neither.
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from duckmind import paths
from duckmind.duckbody.tof_reproject import HeadFk, Pose, Quat, Tree

# ── robotd's own numbers (robotd-params, duck-control, robotd control.rs defaults) ─────────────
CONTROL_HZ = 50
TICK_S = 1.0 / CONTROL_HZ
TIMESTEP_S = 0.005  # body_server.TIMESTEP: 4 physics steps per robotd tick
STEPS_PER_TICK = 4
FALL_GRAVITY_Z = -0.5  # safety.fall_gravity_z (about 60° from upright)
FALL_DEBOUNCE_S = 0.2  # safety.fall_debounce
DEADMAN_S = 0.5  # safety.deadman: an intent older than this has its twist zeroed
CMD_ALPHA = 0.2  # control.cmd_alpha
HEAD_ALPHA = 0.2  # control.head_alpha
ACTION_SCALE_WALK = 0.9  # velstand as "walk" (no standing network in seed-v5)
ACTION_SCALE_SITSTAND = 1.0
LOWPASS_LEGS = 0.7
LOWPASS_HEAD = 0.5
SITSTAND_RISE_S = 1.0  # DEFAULT_SITSTAND_RISE_S
HOME_RAMP_S = 2.0  # a standing boot ramps to HOME before the policy drives
SEATED_BOOT_RAD = 0.30  # mean |leg − home| above which a boot counts as seated
CURRENT_PER_NEWTON = 100.0  # body_server: |actuator_force| × 100, "a stand-in with the right shape"
BOOT_NS = 1_000_000_000_000  # the synthetic CLOCK_MONOTONIC at sim time 0 (any constant: relative)
WALL_EPOCH_US = 1_790_000_000_000_000  # the synthetic wall clock at sim time 0 (media.frame)

# duck-ipc-proto JOINT_NAMES: the wire order, the mouth at index 9.
JOINT_NAMES = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "mouth",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)  # fmt: skip
MOUTH = 9
# The 14 policy joints (the wire order without the mouth), and body_server's HOME_POSE on them.
JOINTS14 = tuple(name for name in JOINT_NAMES if name != "mouth")
HOME14 = (0.0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0.0, 0.0,
          0.0, 0.0873, 0.4579, 0.0049, -0.4530)  # fmt: skip
HEAD_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
HOME_HEAD = (0.3491, 0.3491, 0.0, 0.0)  # robotd.V5_HOME_HEAD: robot.head offsets are from here
LEGS14 = (0, 1, 2, 3, 4, 9, 10, 11, 12, 13)
HEAD14 = (5, 6, 7, 8)
UPRIGHT_POLICIES = frozenset({"walk", "stand"})
SKILLS = ("sit_toggle",)  # robot.do skills this emulator runs


def policy_dir() -> Path:
    """Where the shipped ONNX policies are: :func:`duckmind.paths.policy_dir`
    (``$DUCKMIND_POLICY_DIR`` else ``current`` in the policy root)."""
    return paths.policy_dir()


def official_dir(mjcf_dir: str | os.PathLike[str] | None = None) -> Path:
    from duckmind.world import homelayout

    return homelayout.official_dir(mjcf_dir)


# ═══════════════════════════════════════════════════════════════════════════════════════════════
# pure ports (no numpy, no MuJoCo)
# ═══════════════════════════════════════════════════════════════════════════════════════════════


class SafetyVerdict:
    """``duck_control::safety::Safety::observe``: the fall verdict, debounced in both directions.

    Down (projected gravity z above ``fall_gravity_z``) must hold ``fall_debounce`` to count as
    fallen; one upright sample clears the accumulator and the verdict. An IMU whose orientation
    filter has not converged gets no vote: the previous verdict is held (``safety.rs`` observe).
    """

    def __init__(
        self, fall_gravity_z: float = FALL_GRAVITY_Z, fall_debounce_s: float = FALL_DEBOUNCE_S
    ) -> None:
        self.fall_gravity_z = float(fall_gravity_z)
        self.fall_debounce_s = float(fall_debounce_s)
        self.falling_for = 0.0
        self.fallen = False

    def observe(self, gravity_z: float, dt_s: float, *, imu_ready: bool = True) -> bool:
        if not imu_ready:
            return self.fallen
        if gravity_z > self.fall_gravity_z:
            self.falling_for += float(dt_s)
            if self.falling_for >= self.fall_debounce_s - 1e-9:
                self.fallen = True
        else:
            self.falling_for = 0.0
            self.fallen = False
        return self.fallen


@dataclass(frozen=True)
class Gaze:
    """What :func:`look_at` chose: ``joints`` = (neck_pitch, head_pitch, head_yaw, head_roll),
    radians, absolute; ``clamped``: the result still misses the target (a travel limit or the
    gimbal geometry)."""

    joints: tuple[float, float, float, float]
    clamped: bool


def joint_ranges(
    mjcf: str | os.PathLike[str], names: Sequence[str] = HEAD_JOINTS
) -> list[tuple[float, float] | None]:
    """Each named joint's ``range`` attribute, as ``kinematics/src/mjcf.rs`` reads it (the joint
    element's own attribute; no class defaults), or None."""
    root = ET.parse(os.fspath(mjcf)).getroot()
    found: dict[str, tuple[float, float]] = {}
    for node in root.iter("joint"):
        name, raw = node.get("name"), node.get("range")
        if name and raw:
            lo, hi = (float(v) for v in raw.split())
            found[name] = (lo, hi)
    return [found.get(name) for name in names]


def look_at(
    fk: HeadFk,
    target_in_trunk: Sequence[float],
    neck_pitch: float,
    ranges: Sequence[tuple[float, float] | None] | None = None,
) -> Gaze:
    """``kinematics::head::HeadFk::look_at``: the head joints that point the camera's optical axis
    at a trunk-frame point. ``head_pitch`` and ``head_yaw`` aim (damped Gauss-Newton on the real
    FK, forward-difference Jacobian, λ 1e-3, step ≤ 0.7, ≤ 30 iterations, tolerance 1e-4 rad);
    ``neck_pitch`` is held as given (clamped to its range), ``head_roll`` stays 0; every step is
    clamped to the joints' travel (``ranges``, default ±π)."""
    tolerance, step_h, lam, max_step = 1e-4, 1e-5, 1e-3, 0.7
    spans = list(ranges) if ranges is not None else [None] * 4

    def span(i: int) -> tuple[float, float]:
        return spans[i] or (-math.pi, math.pi)

    def clamp(value: float, bounds: tuple[float, float]) -> float:
        return min(max(value, bounds[0]), bounds[1])

    tx, ty, tz = (float(v) for v in target_in_trunk)

    def error(joints: list[float]) -> tuple[float, float]:
        cam = fk.camera_in_trunk_cv2(joints)
        v = cam.quat.conjugate().rotate((tx - cam.pos[0], ty - cam.pos[1], tz - cam.pos[2]))
        flat = math.sqrt(v[0] * v[0] + v[2] * v[2])
        return math.atan2(v[0], v[2]), math.atan2(v[1], flat)

    joints = [clamp(float(neck_pitch), span(0)), 0.0, 0.0, 0.0]
    residual = math.inf
    for _ in range(30):
        e = error(joints)
        residual = max(abs(e[0]), abs(e[1]))
        if residual < tolerance:
            break
        jac = [[0.0, 0.0], [0.0, 0.0]]
        for col, joint in enumerate((1, 2)):
            probe = list(joints)
            probe[joint] += step_h
            ep = error(probe)
            jac[0][col] = (ep[0] - e[0]) / step_h
            jac[1][col] = (ep[1] - e[1]) / step_h
        a = jac[0][0] * jac[0][0] + jac[1][0] * jac[1][0] + lam
        b = jac[0][0] * jac[0][1] + jac[1][0] * jac[1][1]
        d = jac[0][1] * jac[0][1] + jac[1][1] * jac[1][1] + lam
        g0 = jac[0][0] * e[0] + jac[1][0] * e[1]
        g1 = jac[0][1] * e[0] + jac[1][1] * e[1]
        det = a * d - b * b
        if abs(det) < 1e-12:
            break
        s0, s1 = (-d * g0 + b * g1) / det, (b * g0 - a * g1) / det
        scale = min(max_step / math.hypot(s0, s1), 1.0) if (s0 or s1) else 1.0
        joints[1] = clamp(joints[1] + scale * s0, span(1))
        joints[2] = clamp(joints[2] + scale * s1, span(2))
    return Gaze((joints[0], joints[1], joints[2], joints[3]), residual >= tolerance)


# odometry/src/anchors.rs (GENERATED upstream by microduck_rl/scripts/odom_anchor_points.py from
# the alpha sole meshes): candidate contact points in each foot's site frame, metres.
V15 = (
    ((0.027, 0.0206, 0.0), (0.027, -0.0206, 0.0), (-0.027, -0.0206, 0.0), (-0.027, 0.0206, 0.0)),
    ((0.027, 0.0206, 0.0), (0.027, -0.0206, 0.0), (-0.027, -0.0206, 0.0), (-0.027, 0.0206, 0.0)),
)
ALPHA4 = (
    ((0.024844, 0.012452, 0.001414), (0.024029, -0.012395, -0.000680),
     (-0.011047, -0.013121, -0.000751), (-0.011862, 0.013177, 0.001515)),
    ((0.024029, 0.012395, -0.000680), (0.024844, -0.012452, 0.001415),
     (-0.011862, -0.013177, 0.001516), (-0.011047, 0.013121, -0.000751)),
)  # fmt: skip
ALPHA16 = (
    ((-0.018300, -0.019510, 0.005894), (-0.001500, -0.019659, 0.001516),
     (0.015500, -0.019659, 0.002082), (0.031699, -0.019061, 0.005531),
     (-0.018500, -0.006960, 0.002891), (-0.001500, -0.006960, -0.000607),
     (0.015500, -0.006960, -0.000573), (0.032500, -0.006960, 0.002863),
     (-0.018500, 0.005740, 0.003709), (-0.001500, 0.005740, 0.000500),
     (0.015500, 0.005740, 0.000474), (0.032500, 0.005740, 0.003683),
     (-0.018300, 0.018290, 0.007998), (-0.001500, 0.018439, 0.003859),
     (0.015500, 0.018439, 0.004318), (0.031699, 0.017841, 0.007574)),
    ((-0.018300, -0.018290, 0.007998), (-0.001500, -0.018439, 0.003859),
     (0.015500, -0.018439, 0.004320), (0.031699, -0.017841, 0.007574),
     (-0.018500, -0.005740, 0.003712), (-0.001500, -0.005740, 0.000500),
     (0.015500, -0.005740, 0.000474), (0.032500, -0.005740, 0.003681),
     (-0.018500, 0.006960, 0.002893), (-0.001500, 0.006960, -0.000607),
     (0.015500, 0.006960, -0.000573), (0.032500, 0.006960, 0.002863),
     (-0.018300, 0.019510, 0.005894), (-0.001500, 0.019659, 0.001516),
     (0.015500, 0.019659, 0.002080), (0.031699, 0.019061, 0.005531)),
)  # fmt: skip
ANCHOR_SETS = {"v15": V15, "alpha4": ALPHA4, "alpha16": ALPHA16}
SWITCH_MARGIN = -0.010  # odometry: a corner must sit below −(−0.010) = 10 mm under… see update()
SWITCH_CONFIRM_TICKS = 2


def quat_yaw(q: Quat) -> float:
    """``kinematics::Quat::yaw``: atan2 of the rotation matrix's (1,0) over (0,0)."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Odometry:
    """``odometry::Odometry``: contact odometry from the legs and the IMU.

    One sole point is the anchor, pinned to the ground at its world X/Y (Z = 0); the trunk
    follows by FK and the IMU orientation. A corner that stays the lowest point, below
    ``−SWITCH_MARGIN``, for ``SWITCH_CONFIRM_TICKS`` updates takes the anchor at its current
    world X/Y, so the estimate never jumps. Heading is the IMU's. The first update seeds the
    anchor so the trunk starts at (0, 0). FK: the ``kinematics`` crate's model, i.e. the official
    ``robot_walk.xml`` read by :class:`~duckmind.duckbody.tof_reproject.Tree`.
    """

    def __init__(self, tree: Tree, anchors: str = "alpha16") -> None:
        self.tree = tree
        feet = (tree.site("left_foot"), tree.site("right_foot"))
        if feet[0] is None or feet[1] is None:
            raise ValueError("the model has no left_foot / right_foot sites")
        self.feet = (int(feet[0]), int(feet[1]))
        self.anchor_set = anchors
        self.anchors = ANCHOR_SETS[anchors]
        self.joint_map = [tree.joint_index(name) for name in JOINT_NAMES]
        self.angles = [0.0] * len(tree.joint_names)
        self.anchor_foot = 0
        self.anchor_local: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.anchor_xy = (0.0, 0.0)
        self.position = (0.0, 0.0, 0.0)
        self.yaw = 0.0
        self.pending: tuple[int, tuple[float, float, float], tuple[float, float]] | None = None
        self.pending_ticks = 0
        self.needs_init = True

    @classmethod
    def from_mjcf(cls, source: str | os.PathLike[str], anchors: str = "alpha16") -> Odometry:
        return cls(Tree.from_mjcf(os.fspath(source)), anchors)

    def update(self, joints: Sequence[float], quat_wxyz: Sequence[float]) -> None:
        """One fresh sample: ``joints`` in JOINT_NAMES order (the mouth is ignored), the IMU's
        trunk-in-world orientation ``(w, x, y, z)``."""
        for angle, index in zip(joints, self.joint_map, strict=True):
            if index is not None:
                self.angles[index] = float(angle)
        rot = Quat(*(float(v) for v in quat_wxyz)).normalized()
        feet = (
            self.tree.site_pose(self.feet[0], self.angles),
            self.tree.site_pose(self.feet[1], self.angles),
        )
        if self.needs_init:
            world = rot.rotate(feet[self.anchor_foot].pos)
            self.anchor_xy = (world[0], world[1])
            self.needs_init = False
        self._reproject(rot, feet)
        lowest = self._lowest_corner(rot, feet)
        if lowest is None:
            self.pending = None
            self.pending_ticks = 0
        else:
            foot = lowest[0]
            if self.pending is not None and self.pending[0] == foot:
                self.pending_ticks += 1
            else:
                self.pending = lowest
                self.pending_ticks = 1
            if self.pending_ticks >= SWITCH_CONFIRM_TICKS and self.pending is not None:
                foot, local, world_xy = self.pending
                self.pending = None
                self.anchor_foot = foot
                self.anchor_local = local
                self.anchor_xy = world_xy
                self._reproject(rot, feet)
                self.pending_ticks = 0
        self.yaw = quat_yaw(rot)

    def _reproject(self, rot: Quat, feet: tuple[Pose, Pose]) -> None:
        contact = rot.rotate(feet[self.anchor_foot].transform_point(self.anchor_local))
        self.position = (
            self.anchor_xy[0] - contact[0],
            self.anchor_xy[1] - contact[1],
            -contact[2],
        )

    def _lowest_corner(
        self, rot: Quat, feet: tuple[Pose, Pose]
    ) -> tuple[int, tuple[float, float, float], tuple[float, float]] | None:
        lowest = -SWITCH_MARGIN
        best = None
        for foot in (0, 1):
            for corner in self.anchors[foot]:
                w = rot.rotate(feet[foot].transform_point(corner))
                world = (
                    self.position[0] + w[0],
                    self.position[1] + w[1],
                    self.position[2] + w[2],
                )
                if world[2] < lowest:
                    lowest = world[2]
                    best = (foot, corner, (world[0], world[1]))
        return best


def gravity_in_trunk(quat_wxyz: Sequence[float]) -> tuple[float, float, float]:
    """World down (0, 0, −1) in the trunk frame (body_server.gravity_in_trunk)."""
    q = Quat(*(float(v) for v in quat_wxyz)).normalized()
    return q.conjugate().rotate((0.0, 0.0, -1.0))


def pose_json(pose: Pose) -> dict[str, list[float]]:
    return {"pos": [float(v) for v in pose.pos], "quat": list(pose.quat.wxyz())}


# ═══════════════════════════════════════════════════════════════════════════════════════════════
# the home, composed as the world server composes it
# ═══════════════════════════════════════════════════════════════════════════════════════════════

LayoutEdit = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class Composed:
    """One compiled-ready home: the ``MjSpec`` (not compiled yet), the resolved layout and the
    household kit that made the world server's edits on it."""

    spec: Any
    layout: Any
    kit: Any


def compose(
    home: str | Mapping[str, Any] = "default_home",
    *,
    variant: str | Sequence[str] | None = None,
    collision: str = "allcollisions",
    mjcf_dir: str | os.PathLike[str] | None = None,
    household: bool = True,
) -> Composed:
    """A home (a name, a path or a layout dict) composed the world server's way: the official
    scene by path, the layout's walls, furniture, nest, things and floor, the household's edits
    (spot probe, nest, hand, people, battery, things) and the keyframes rewritten. Call it again
    for the eye's render copy: an ``MjSpec`` holding a texture from a buffer cannot be copied
    through XML, and the same call makes the same bodies in the same order."""
    from duckmind.world import homelayout, worldkit

    folder = os.fspath(mjcf_dir) if mjcf_dir is not None else None
    layout = homelayout.load(home, variant=variant, mjcf_dir=folder)
    spec = homelayout.compose(layout, collision, mjcf_dir=folder)
    kit = None
    if household:
        kit = worldkit.build()
        kit.edit_spec(spec, layout)
    homelayout.fix_keyframes(spec, layout)
    return Composed(spec=spec, layout=layout, kit=kit)


# ═══════════════════════════════════════════════════════════════════════════════════════════════
# the duck
# ═══════════════════════════════════════════════════════════════════════════════════════════════

_SESSIONS: dict[str, Any] = {}


def _session(path: Path) -> Any:
    """One ONNX session per policy file, single-threaded (a 61→14 MLP gains nothing from threads,
    and one thread keeps the result independent of the machine's load: P2)."""
    key = str(path)
    if key not in _SESSIONS:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        _SESSIONS[key] = ort.InferenceSession(
            key, sess_options=opts, providers=["CPUExecutionProvider"]
        )
    return _SESSIONS[key]


@dataclass
class Intent:
    """The last intent of each kind and when it came (sim seconds), as robotd keeps them."""

    twist: tuple[float, float, float] = (0.0, 0.0, 0.0)
    twist_at: float = -math.inf
    head: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    mouth: float = 0.0
    toggles: int = 0
    log: list[dict[str, Any]] = field(default_factory=list)


class HeadlessDuck:
    """The official duck in a composed home, driven by a port of robotd's control path.

    ``home``: a home name, path or layout dict (:func:`compose`), or an already built
    :class:`Composed`. The duck starts at ``(x, y, yaw)`` (world metres and radians; default the
    layout's spawn) in ``keyframe`` (``SIT`` as the owner's simulator boots, ``STAND``, ``HOME``),
    held in place until :meth:`enable` (as the body server holds it until torque comes on).

    Intents (take effect at the next 50 Hz tick, like robotd): :meth:`move`, :meth:`head`,
    :meth:`look`, :meth:`mouth`, :meth:`do`, :meth:`stop`. Stepping: :meth:`tick` (one robotd
    tick = 4 physics steps), :meth:`run`. What the duck's software may read: :meth:`state`
    (``robot.state``) and :meth:`health` (``robot.health``). Truth, for evaluators only:
    :meth:`truth`, :attr:`data`, :attr:`model`.
    """

    def __init__(
        self,
        home: str | Mapping[str, Any] | Composed = "default_home",
        *,
        variant: str | Sequence[str] | None = None,
        collision: str = "allcollisions",
        mjcf_dir: str | os.PathLike[str] | None = None,
        keyframe: str = "SIT",
        x: float | None = None,
        y: float | None = None,
        yaw: float | None = None,
        policies: str | os.PathLike[str] | None = None,
        household: bool = True,
        anchors: str = "alpha16",
        boot_ns: int = BOOT_NS,
        command_delay_ms: float | None = None,
    ) -> None:
        import mujoco
        import numpy as np

        from duckmind.world import worldkit

        self._mj = mujoco
        self._np = np
        built = (
            home
            if isinstance(home, Composed)
            else compose(
                home, variant=variant, collision=collision, mjcf_dir=mjcf_dir, household=household
            )
        )
        self.composed = built
        self.layout = built.layout
        self.kit = built.kit
        self.model = built.spec.compile()
        self.model.opt.timestep = TIMESTEP_S
        self.data = mujoco.MjData(self.model)
        self.world = _World(self.model, self.data)
        self.robot = worldkit.RobotIds.of(self.model)
        self.boot_ns = int(boot_ns)
        folder = official_dir(mjcf_dir)
        walk_xml = folder / "robot_walk.xml"  # the kinematics crate's model (Model::alpha)
        self.fk = HeadFk.from_mjcf(walk_xml)
        self.head_ranges = joint_ranges(walk_xml)
        self.odometry = Odometry(self.fk.tree, anchors)
        self.safety = SafetyVerdict()
        self.policies = Path(policies).expanduser() if policies is not None else policy_dir()

        def ident(kind: Any, name: str) -> int:
            found = int(mujoco.mj_name2id(self.model, kind, name))
            if found < 0:
                raise KeyError(f"the model has no {name!r}")
            return found

        self.jq = np.array(
            [self.model.jnt_qposadr[ident(mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINTS14]
        )
        self.jv = np.array(
            [self.model.jnt_dofadr[ident(mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINTS14]
        )
        self.act = np.array([ident(mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINTS14])
        self.qadr, self.vadr = self.robot.qadr, self.robot.dof
        self.trunk = self.robot.trunk
        self.home = np.array(HOME14)
        self.beak_site = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "mouth_tip"))

        self.policy = "held"
        self.released = False
        self.intent = Intent()
        self._twist_ema = np.zeros(3)
        self._head_ema = np.zeros(4)
        self._targets = np.array(HOME14)
        # The servo command delay, as the live body host has it (bodyhost.COMMAND_DELAY_ENV, 20 ms in the simulator since
        # 2026-10-05: the real servos act late): the targets reach the actuators this many physics steps after they were
        # commanded. ``None``: the same environment variable; 0: none (what this port always did).
        if command_delay_ms is None:
            command_delay_ms = float(
                (os.environ.get("DUCKMIND_BODY_COMMAND_DELAY_MS") or "0").strip() or 0
            )
        self.command_delay_steps = int(round(float(command_delay_ms) / 1000.0 / TIMESTEP_S))
        self._in_flight: Any = None
        self._prev_targets: Any = None
        self._last_action = np.zeros(14, dtype=np.float32)
        self._bringup = "limp"
        self._home_from = np.array(HOME14)
        self._home_since = 0.0
        self._sit = "up"
        self._rise_left = 0.0
        self._ticks = 0
        self._was_held = False
        self.t0 = 0.0
        self.events: list[dict[str, Any]] = []

        self.place(keyframe=keyframe, x=x, y=y, yaw=yaw)
        for part in self.kit.parts if self.kit is not None else ():
            part.ready(self.world, None)
        self._before = [self.kit.hand.before_step, self.kit.people.before_step] if self.kit else []
        self._after = (
            [self.kit.nest.after_step, self.kit.battery.after_step, self.kit.props.after_step]
            if self.kit
            else []
        )
        self.state_now = self._state()

    # ── placing ─────────────────────────────────────────────────────────────────────────────
    def place(
        self,
        *,
        keyframe: str = "SIT",
        x: float | None = None,
        y: float | None = None,
        yaw: float | None = None,
    ) -> None:
        """The duck at (x, y, yaw) in ``keyframe`` at rest (a hand setting it down), the rest of
        the world from the keyframe (the layout's things where the layout put them)."""
        mj, np = self._mj, self._np
        key = int(mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_KEY, keyframe))
        if key < 0:
            raise ValueError(f"no keyframe {keyframe!r}")
        mj.mj_resetDataKeyframe(self.model, self.data, key)
        d = self.data
        spawn = d.qpos[self.qadr : self.qadr + 3].copy()
        spawn_yaw = _yaw_of(d.qpos[self.qadr + 3 : self.qadr + 7])
        px = float(spawn[0]) if x is None else float(x)
        py = float(spawn[1]) if y is None else float(y)
        heading = spawn_yaw if yaw is None else float(yaw)
        d.qpos[self.qadr] = px
        d.qpos[self.qadr + 1] = py
        d.qpos[self.qadr + 3 : self.qadr + 7] = [
            math.cos(heading / 2),
            0.0,
            0.0,
            math.sin(heading / 2),
        ]
        d.qvel[self.vadr : self.vadr + 6] = 0.0
        d.ctrl[self.act] = d.qpos[self.jq]
        mj.mj_forward(self.model, d)
        self._settle_z()
        self._hold = (d.qpos[self.qadr : self.qadr + 7].copy(), d.qpos[self.jq].copy())
        self.released = False
        self.policy = "held"
        self._bringup = "limp"
        self._sit = "up"
        self._twist_ema = np.zeros(3)
        self._head_ema = np.zeros(4)
        self._prev_targets = None
        self._last_action = np.zeros(14, dtype=np.float32)
        self.safety = SafetyVerdict()
        self.odometry = Odometry(self.fk.tree, self.odometry.anchor_set)
        self.t0 = float(d.time)

    def teleport(self, x: float, y: float, yaw: float) -> None:
        """Set a standing duck down elsewhere as it is (a scene set-up, like the hand): the trunk
        moved to (x, y) and turned to ``yaw`` about the vertical, joints, height and tilt kept,
        velocities zeroed; odometry restarts (the IMU's yaw jumped). Evaluators and gate set-up
        only — never a thing the duck does."""
        mj, d = self._mj, self.data
        q = [float(v) for v in d.qpos[self.qadr + 3 : self.qadr + 7]]
        turn = yaw - _yaw_of(q)
        c, s = math.cos(turn / 2), math.sin(turn / 2)
        w, qx, qy, qz = q  # (c, 0, 0, s) * q: turn about the world vertical
        d.qpos[self.qadr] = float(x)
        d.qpos[self.qadr + 1] = float(y)
        d.qpos[self.qadr + 3 : self.qadr + 7] = [
            c * w - s * qz,
            c * qx - s * qy,
            c * qy + s * qx,
            c * qz + s * w,
        ]
        d.qvel[self.vadr : self.vadr + 6] = 0.0
        mj.mj_forward(self.model, d)
        self._settle_z()
        self.odometry = Odometry(self.fk.tree, self.odometry.anchor_set)
        if not self.released:
            self._hold = (d.qpos[self.qadr : self.qadr + 7].copy(), d.qpos[self.jq].copy())
        self._log("teleport", x=round(float(x), 4), y=round(float(y), 4), yaw=round(float(yaw), 4))

    def _settle_z(self) -> None:
        """Lift the duck until none of its colliders starts inside something (the keyframe's
        trunk height is for the official floor; a pad or a platform may be under it)."""
        mj, d = self._mj, self.data
        colliders = set(self.robot.colliders)
        for _ in range(200):
            mj.mj_forward(self.model, d)
            worst = 0.0
            for i in range(int(d.ncon)):
                c = d.contact[i]
                if int(c.geom1) in colliders or int(c.geom2) in colliders:
                    worst = min(worst, float(c.dist))
            if worst >= -1e-4:
                break
            d.qpos[self.qadr + 2] += -worst + 0.001

    # ── intents (robotd's surface) ───────────────────────────────────────────────────────────
    def now(self) -> float:
        return float(self.data.time)

    def t_ns(self) -> int:
        """The synthetic CLOCK_MONOTONIC of this instant (robot.state.t_ns, tof.frame.t_ns)."""
        return self.boot_ns + int(round(float(self.data.time) * 1e9))

    def wall_us(self) -> int:
        """The synthetic wall clock of this instant (media.frame captured_at_unix_us)."""
        return WALL_EPOCH_US + int(round(float(self.data.time) * 1e6))

    def enable(self) -> dict[str, Any]:
        """Torque on (body_server set_torque): a seated boot rises through sitstand, a standing
        one ramps 2 s to HOME before the policy drives."""
        np = self._np
        self.released = True
        self.data.ctrl[self.act] = self.data.qpos[self.jq]
        joints = np.array(self.data.qpos[self.jq])
        legs = list(LEGS14)
        deviation = float(np.mean(np.abs(joints[legs] - self.home[legs])))
        if deviation > SEATED_BOOT_RAD:
            self._sit, self._rise_left, self._bringup = "rising", SITSTAND_RISE_S, "ready"
            how = "seated_boot_rise"
        else:
            self._bringup, self._home_from, self._home_since = "homing", joints, self.now()
            how = "home_ramp"
        self._log("enable", bringup=how)
        return {"accepted": True, "bringup": how}

    def move(self, vx: float, vy: float = 0.0, vyaw: float = 0.0) -> dict[str, Any]:
        """``robot.move``: a twist, stale after the deadman."""
        self.intent.twist = (float(vx), float(vy), float(vyaw))
        self.intent.twist_at = self.now()
        return {"accepted": True}

    def head(self, offsets: Sequence[float]) -> dict[str, Any]:
        """``robot.head``: (neck_pitch, head_pitch, head_yaw, head_roll) as the policy's head
        command (offsets from HOME_HEAD for velstand v5)."""
        vals = tuple(float(v) for v in offsets)
        if len(vals) != 4:
            raise ValueError("robot.head takes 4 values")
        self.intent.head = (vals[0], vals[1], vals[2], vals[3])
        return {"accepted": True}

    def look(self, x: float, y: float, z: float, neck_pitch: float = 0.0) -> dict[str, Any]:
        """``robot.look``: gaze IK at a trunk-frame point; like robotd, the ABSOLUTE joints go
        out as the head intent (the known v5 mismatch, robotd.py V5_HOME_HEAD's comment)."""
        gaze = look_at(self.fk, (x, y, z), neck_pitch, self.head_ranges)
        self.intent.head = gaze.joints
        return {
            "head": dict(zip(HEAD_JOINTS, gaze.joints, strict=True)),
            "clamped": gaze.clamped,
        }

    def mouth(self, angle: float) -> dict[str, Any]:
        """``robot.mouth``: kept, not modelled (the official MJCF has no jaw)."""
        self.intent.mouth = float(angle)
        return {"accepted": True}

    def do(self, skill: str) -> dict[str, Any]:
        """``robot.do``: ``sit_toggle`` (queued for the next tick, only when ready, as robotd);
        every other skill is refused here with the reason (not modelled headless)."""
        if skill != "sit_toggle":
            return {"accepted": False, "reason": f"headless: {skill} is not modelled"}
        self.intent.toggles += 1
        self._log("do", skill=skill, policy_before=self.policy)
        return {"accepted": True}

    def stop(self) -> dict[str, Any]:
        """The safety connection's stop: the twist to zero now."""
        self.intent.twist = (0.0, 0.0, 0.0)
        self.intent.twist_at = self.now()
        return {"accepted": True}

    def hand(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """The world's hand (``x_hand``: pickup, place, upright…) — the evaluator's, never the
        duck's."""
        if self.kit is None:
            raise RuntimeError("no household in this home")
        with self.world.lock:
            return dict(self.kit.hand.command(self.world, dict(request)))

    # ── stepping ────────────────────────────────────────────────────────────────────────────
    def tick(self) -> None:
        """One robotd tick: the controller on this tick's sensors, then 4 physics steps with the
        household's hooks, then the state robotd would publish."""
        if self.released:
            self._controller()
        mj = self._mj
        delay = self.command_delay_steps if self.released else 0
        if delay > 0:
            from collections import deque

            commanded = self.data.ctrl[self.act].copy()
            if self._in_flight is None or self._in_flight.maxlen != delay + 1:
                self._in_flight = deque([commanded] * delay, maxlen=delay + 1)
        for _ in range(STEPS_PER_TICK):
            for fn in self._before:
                fn(self.world)
            if delay > 0:
                self._in_flight.append(commanded)
                self.data.ctrl[self.act] = self._in_flight[0]
            mj.mj_step(self.model, self.data)
            if not self.released:
                holding = self._hand_holding()
                if holding:
                    self._was_held = True
                elif self._was_held:  # a duck nobody enabled stays where the hand put it
                    self._was_held = False
                    d = self.data
                    self._hold = (d.qpos[self.qadr : self.qadr + 7].copy(), d.qpos[self.jq].copy())
                else:
                    self._restore()
            for fn in self._after:
                fn(self.world)
        self._ticks += 1
        self.state_now = self._state()

    def run(
        self, seconds: float, each: Callable[[HeadlessDuck], bool | None] | None = None
    ) -> bool:
        """Tick for ``seconds``; ``each`` after every tick may return True to stop early (then
        True is returned)."""
        for _ in range(max(0, int(round(seconds * CONTROL_HZ)))):
            self.tick()
            if each is not None and each(self):
                return True
        return False

    def stand(self, timeout: float = 15.0, hold: float = 1.0) -> tuple[bool, float | None]:
        """Enable if needed, ask to stand (sit_toggle every 3 s while seated, as LiveBody does),
        and wait until up (tilt < 25°, not fallen, walk/stand) for ``hold`` s."""
        start = self.now()
        if not self.released:
            self.enable()
        streak: float | None = None
        last_toggle = -math.inf
        while self.now() - start < timeout + hold:
            if self.policy == "sit" and self.now() - last_toggle >= 3.0:
                self.do("sit_toggle")
                last_toggle = self.now()
            self.tick()
            if self.upright():
                streak = self.now() if streak is None else streak
                if self.now() - streak >= hold - 1e-9:
                    return True, streak - start
            else:
                streak = None
                if self.now() - start > timeout:
                    break
        return False, None

    def upright(self) -> bool:
        return self.tilt_deg() < 25.0 and not self.safety.fallen and self.policy in UPRIGHT_POLICIES

    def tilt_deg(self) -> float:
        q = self.data.qpos[self.qadr + 3 : self.qadr + 7]
        g = gravity_in_trunk(q)
        return math.degrees(math.acos(max(-1.0, min(1.0, -g[2]))))

    def _hand_holding(self) -> bool:
        if self.kit is None:
            return False
        eq = getattr(self.kit.hand, "weld", None)
        try:
            return bool(eq is not None and self.data.eq_active[int(eq)])
        except (TypeError, ValueError, IndexError):
            return False

    def _restore(self) -> None:
        d = self.data
        trunk, joints = self._hold
        d.qpos[self.qadr : self.qadr + 7] = trunk
        d.qpos[self.jq] = joints
        d.qvel[self.vadr : self.vadr + 6] = 0.0
        d.qvel[self.jv] = 0.0

    def _obs(self, cmd13: Any) -> Any:
        np, d = self._np, self.data
        q = d.qpos[self.qadr + 3 : self.qadr + 7]
        return (
            np.concatenate(
                [
                    d.qvel[self.vadr + 3 : self.vadr + 6],  # body_server's gyro
                    np.array(gravity_in_trunk(q)),
                    d.qpos[self.jq] - self.home,
                    d.qvel[self.jv],
                    self._last_action,
                    cmd13,
                ]
            )
            .astype(np.float32)
            .reshape(1, -1)
        )

    def _controller(self) -> None:
        np, d = self._np, self.data
        t = self.now()
        stale = t - self.intent.twist_at > DEADMAN_S + 1e-9
        twist = np.zeros(3) if stale else np.array(self.intent.twist, dtype=float)
        self._twist_ema += CMD_ALPHA * (twist - self._twist_ema)
        self._head_ema += HEAD_ALPHA * (np.array(self.intent.head, dtype=float) - self._head_ema)
        q = d.qpos[self.qadr + 3 : self.qadr + 7]
        self.safety.observe(gravity_in_trunk(q)[2], TICK_S)
        if self._bringup == "homing":
            frac = (t - self._home_since) / HOME_RAMP_S
            if frac < 1.0:
                d.ctrl[self.act] = self._home_from + (self.home - self._home_from) * frac
                self._targets = np.array(d.ctrl[self.act])
                self.policy = "homing"
                return
            self._bringup = "ready"
            self._prev_targets = None
            self._last_action = np.zeros(14, dtype=np.float32)
        while self.intent.toggles > 0:
            self.intent.toggles -= 1
            if self._sit == "up":
                self._sit = "sitting"
            elif self._sit == "sitting":
                self._sit, self._rise_left = "rising", SITSTAND_RISE_S
            self._log("sit_toggle_applied", now=self._sit)
        if self._sit == "rising" and self._rise_left <= 1e-9:
            self._sit = "up"
        cmd13 = np.zeros(13, dtype=np.float32)
        cmd13[3:7] = self._head_ema
        if self._sit == "sitting":
            net, label = "alpha_sitstand", "sit"
            cmd13[0:3] = (1.0, 0.0, 0.0)
        elif self._sit == "rising":
            net, label = "alpha_sitstand", "rise"
        else:
            net, label = "velstand", "walk"
            cmd13[0:3] = self._twist_ema
        session = _session(self.policies / f"{net}.onnx")
        feed = {session.get_inputs()[0].name: self._obs(cmd13)}
        action = session.run(None, feed)[0].squeeze(0).astype(np.float32)
        self._last_action = action
        scale = ACTION_SCALE_SITSTAND if net == "alpha_sitstand" else ACTION_SCALE_WALK
        targets = self.home + scale * action.astype(float)
        if self._prev_targets is not None:
            head, legs = list(HEAD14), list(LEGS14)
            prev = self._prev_targets
            targets[head] = LOWPASS_HEAD * targets[head] + (1 - LOWPASS_HEAD) * prev[head]
            targets[legs] = LOWPASS_LEGS * targets[legs] + (1 - LOWPASS_LEGS) * prev[legs]
        self._prev_targets = targets.copy()
        targets = np.clip(targets, -math.pi, math.pi)
        d.ctrl[self.act] = targets
        self._targets = targets
        if self._sit == "rising":
            self._rise_left -= TICK_S
        self.policy = label

    # ── what robotd publishes ───────────────────────────────────────────────────────────────
    def state(self) -> dict[str, Any]:
        """``robot.state`` of the last tick (duck-ipc-proto ``RobotState``)."""
        return self.state_now

    def _state(self) -> dict[str, Any]:
        d = self.data
        quat = [float(v) for v in d.qpos[self.qadr + 3 : self.qadr + 7]]
        gyro = [float(v) for v in d.qvel[self.vadr + 3 : self.vadr + 6]]
        joints14 = [float(v) for v in d.qpos[self.jq]]
        vel14 = [float(v) for v in d.qvel[self.jv]]
        force14 = [abs(float(v)) * CURRENT_PER_NEWTON for v in d.actuator_force[self.act]]
        targets14 = [float(v) for v in self._targets]
        joints = _wire(joints14, self.intent.mouth if self.released else 0.0)
        head = joints[5:9]
        if self.released:
            self.odometry.update(joints, quat)
        stale = self.now() - self.intent.twist_at > DEADMAN_S + 1e-9
        requested = [0.0, 0.0, 0.0] if stale else list(self.intent.twist)
        return {
            "t": round(self.now() - self.t0, 6),
            "t_ns": self.t_ns(),
            "move": {
                "requested": requested,
                "applied": [float(v) for v in self._twist_ema],
                **(
                    {"limited_by": ["deadman"]}
                    if stale and self.intent.twist_at > -math.inf
                    else {}
                ),
            },
            "head": [float(v) for v in self._head_ema],
            "policy": self.policy,
            "safety": {
                "fallen": self.safety.fallen,
                "limp": False,
                "gravity": list(gravity_in_trunk(quat)),
                "gain": None,
            },
            "loop": {"hz": float(CONTROL_HZ), "missed": 0},
            "joints": joints,
            "targets": _wire(targets14, self.intent.mouth),
            "velocities": _wire(vel14, 0.0),
            "currents_ma": _wire(force14, 0.0),
            "odom": {
                "position": list(self.odometry.position),
                "yaw": float(self.odometry.yaw),
            },
            "imu": {"gyro": gyro, "quat": quat},
            "frames": {
                "camera": pose_json(self.fk.camera_in_trunk_cv2(head)),
                "tof": pose_json(self.fk.tof_in_trunk(head)),
            },
        }

    def health(self) -> dict[str, Any]:
        """``robot.health`` as the Mac sim answers it: healthy, the body server's nominal volts,
        no SoC temperature and no throttle reading (absent off Linux)."""
        return {"healthy": True, "battery": {"volts": 7.4}}

    # ── truth (evaluators only) ─────────────────────────────────────────────────────────────
    def truth(self) -> dict[str, Any]:
        """World truth for evaluators: trunk pose, tilt, the beak tip, robot↔world contacts
        (robot geom, world geom), the hand's hold. Never given to the duck's software."""
        mj, d = self._mj, self.data
        contacts: list[tuple[str, str]] = []
        robot = self.robot.bodies
        for i in range(int(d.ncon)):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            b1, b2 = int(self.model.geom_bodyid[g1]), int(self.model.geom_bodyid[g2])
            if (b1 in robot) == (b2 in robot):
                continue
            mine, other = (g1, g2) if b1 in robot else (g2, g1)
            contacts.append((_geom_name(mj, self.model, mine), _geom_name(mj, self.model, other)))
        beak = [float(v) for v in d.site_xpos[self.beak_site]] if self.beak_site >= 0 else None
        return {
            "t": self.now(),
            "trunk": [float(v) for v in d.qpos[self.qadr : self.qadr + 3]],
            "quat": [float(v) for v in d.qpos[self.qadr + 3 : self.qadr + 7]],
            "yaw": _yaw_of(d.qpos[self.qadr + 3 : self.qadr + 7]),
            "tilt_deg": self.tilt_deg(),
            "beak": beak,
            "contacts": sorted(set(contacts)),
            "held": self._hand_holding(),
            "policy": self.policy,
        }

    def _log(self, what: str, **fields: Any) -> None:
        self.events.append({"t": round(self.now(), 4), "kind": what, **fields})


class _World:
    """What the household parts expect of the world server: ``model``, ``data``, ``lock``."""

    def __init__(self, model: Any, data: Any) -> None:
        import threading

        self.model = model
        self.data = data
        self.lock = threading.RLock()
        self.bodies: list[Any] = []


def _wire(values14: Sequence[float], mouth: float) -> list[float]:
    out = [float(v) for v in values14]
    out.insert(MOUTH, float(mouth))
    return out


def _yaw_of(q: Any) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _geom_name(mj: Any, model: Any, geom: int) -> str:
    name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geom)
    if name:
        return str(name)
    body = int(model.geom_bodyid[geom])
    owner = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body) or f"body{body}"
    return f"{owner}/geom{geom}"
