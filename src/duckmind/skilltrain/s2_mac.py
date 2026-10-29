"""The Stage-2 Mac harness (tier 2 of the training plan §6): a walk network on the official duck,
both servo models, a servo command delay of 0–40 ms, robot.pose, and the R4 oracle probe.

    python -m duckmind.skilltrain.s2_mac run --servo mjcf --delay d0 --arm v5 --out DIR [--n 30]
        [--walk FILE.onnx] [--only SUBSTR ...] [--set gates|oracle|regress|all]
    python -m duckmind.skilltrain.s2_mac summarize --out DIR

**The duck** (:class:`WalkDuck`): :class:`~duckmind.skilltrain.skillduck.SkillDuck` — the official
``scene_allcollisions`` floor, one DR draw per trial (``mjcf`` the physical draw: the owner's
simulator; ``bam`` training's BAM XL330 with the recipe's DR, observation noise and IMU error) —
driven by :class:`~duckmind.skilltrain.s2_port.PoseRobotd` with the walk slot's network swapped
(``--walk``; default the official v5 ``velstand.onnx``), the neural demo's ``head_alpha`` 1.0, and the
servo command delay of the ``--delay`` label: ``dN`` = every target exactly N ms late (N a multiple
of 5; the owner's simulator today is ``d0``, the proposal ``d20``), ``j3-6`` = training's model (a lag
redrawn every 5-ms physics step in 15–30 ms). Extra truth per tick: each foot's vertical floor force
(the weight shift), the head camera's world angular velocity and optical axis (the gaze), robotd's
body block, the instantaneous actuator force (the A100's leg-RMS definition).

**Programs** (a fresh duck per trial: seated boot → rise → stand, as the owner's simulator starts):

=====================  ======================================================================
``stand/quiet``        20 s at zero command: stepping, drift, walk-off (> 40 cm), leg RMS torque
``lean/<axis><v>``     robot.pose ramped in (0.5 rad/s, 60 mm/s), held 4.5 s, ramped out (``…r``) or
                       released at once (``…snap``): achieved, t90, steps, release rate, foot load
``sway/<axis><a>@<f>`` robot.pose sine: gain and lag against the client's command
``head/pullback``      standing neck +0.3 / head −0.2 for 2.5 s (the startle's head): neck gain
``head/micro``         standing head yaw ±0.052 rad (±3°) at 1 Hz: gain
``walk/<v>/<head>``    14 s straight at v with the head at HOME / ±1.0 yaw / saccades: speed, camera
                       ⊥ rate RMS, head world-yaw SD at the stride (> 1 Hz), delivered yaw, stop
``stopgo/<v>[/pose]``  3 × (walk 3 s → stop → [0 / 0.7 / 1.5 s → pose sway 3 s → release] → …)
``turn/<v>/<wz>``      yaw gain; ``sitstand`` the sit → rise cycle; ``push/stand`` Δv 1.2–1.6 m/s
``oracle/<v>/<lead>``  (P0, v5 only) the head joints driven by the exact counter-rotation (IK on the
                       true trunk orientation, predicted ``lead`` ms ahead): what the hardware can do
=====================  ======================================================================

Seeds depend only on (condition, rep): every arm, servo model and delay sees the same ducks
(paired). Rows go to ``DIR/rows-<servo>-<delay>-<arm>.jsonl`` (resumable); ``summarize`` writes
``DIR/summary.json`` and ``DIR/report.md`` (k/n with Wilson 95 %, p50/p10–p90, candidate vs v5 on
the same ducks). The owner's :8765 realtime is read (GET /api/body only) before trials; under 0.97
the run pauses 5 minutes (WORK-RULES §5).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import zlib
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from duckmind import paths
from duckmind.skilltrain import dr as DR
from duckmind.skilltrain import metrics as M
from duckmind.skilltrain import s2_contract as C
from duckmind.skilltrain.robotd_port import PolicySet
from duckmind.skilltrain.s2_port import PoseRobotd
from duckmind.skilltrain.skillduck import SITSTAND_FILE, WALK_FILE, SkillDuck, Trace

HZ = 50
DTK = 1.0 / HZ
PHYSICS_MS = 5.0
HEAD_ALPHA = C.HEAD_ALPHA_DEMO
LEFT_FOOT, RIGHT_FOOT = 0, 1
STRIDE_HP_S = (
    0.5  # the head/trunk world yaw is high-passed by a centred 0.5-s mean (the stride band)
)
WALKOFF_M = 0.40


# ── the duck ─────────────────────────────────────────────────────────────────────────────────


def delay_of(label: str) -> tuple[str, int, int]:
    """``d20`` → ("fixed", 4, 4); ``j3-6`` → ("jitter", 3, 6) (physics steps of 5 ms)."""
    if label.startswith("d"):
        ms = float(label[1:])
        steps = ms / PHYSICS_MS
        if steps < 0 or abs(steps - round(steps)) > 1e-9:
            raise ValueError(f"delay {label}: not a whole number of 5-ms steps")
        return ("fixed", int(round(steps)), int(round(steps)))
    if label.startswith("j"):
        lo, _, hi = label[1:].partition("-")
        return ("jitter", int(lo), int(hi))
    raise ValueError(f"delay label {label!r}: dN (ms) or jLO-HI (steps)")


class S2Trace(Trace):
    """The Stage-1 trace plus the Stage-2 channels."""

    VECTORS = {
        **Trace.VECTORS,
        "foot_fz": 2,
        "cam_w": 3,
        "cam_x": 3,
        "body_cmd": 3,
        "tau": 14,
    }
    SCALARS = (*Trace.SCALARS, "pose_on")

    def __init__(self, extra: Callable[[], dict[str, Any]]) -> None:
        super().__init__()
        self._extra = extra

    def append(self, sample: dict[str, Any], label: str) -> None:
        sample = {**sample, **self._extra()}
        super().append(sample, label)


class WalkDuck(SkillDuck):
    """A :class:`SkillDuck` whose walk slot runs ``walk`` (default v5), with pose mode and a delay."""

    def __init__(
        self,
        *,
        servo: str,
        draw: DR.Draw,
        delay: str = "d0",
        walk: str | os.PathLike[str] | None = None,
        head_alpha: float = HEAD_ALPHA,
        policies: str | None = None,
        mjcf_dir: str | None = None,
        yaw: float = 0.0,
    ) -> None:
        import mujoco

        super().__init__(
            [],
            servo=servo,
            draw=draw,
            head_alpha=head_alpha,
            keyframe="SIT",
            policies=policies,
            mjcf_dir=mjcf_dir,
            yaw=yaw,
        )
        folder = Path(self.policies)
        self.walk_path = Path(walk) if walk else folder / WALK_FILE
        self.port = PoseRobotd(
            PolicySet(self.walk_path, folder / SITSTAND_FILE, []), [], head_alpha=head_alpha
        )
        self.skill_log = self.port.controller.events
        mode, lo, hi = delay_of(delay)
        self.delay_label = delay
        self.delay_model = (mode, lo, hi)
        self.lag_steps = hi
        self._lag = deque(maxlen=hi + 1)
        self._lag_rng = np.random.default_rng(int(draw.seed) + 104_729)
        self._mj = mujoco
        self._w6 = np.zeros(6)
        self._f6 = np.zeros(6)
        self.trace = S2Trace(self._extra)
        self._scratch = mujoco.MjData(self.model)
        self._jacr = np.zeros((3, self.model.nv))
        self.head_q = np.array([self.jq[i] for i in C.HEAD_IDX], dtype=int)
        self.head_v = np.array([self.jv[i] for i in C.HEAD_IDX], dtype=int)

    def _delayed(self) -> np.ndarray:
        mode, lo, hi = self.delay_model
        if mode == "fixed" or len(self._lag) <= 1:
            return self._lag[0]
        lag = int(self._lag_rng.integers(lo, hi + 1))
        return self._lag[-1 - min(lag, len(self._lag) - 1)]

    def pose(self, z: float, roll: float, pitch: float, active: bool = True) -> None:
        self.port.pose(z, roll, pitch, active)

    def _extra(self) -> dict[str, Any]:
        m, d = self.model, self.data
        mujoco = self._mj
        fz = np.zeros(2)
        robot = self.robot_geoms
        for i in range(int(d.ncon)):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            mine = (
                g1
                if g1 in robot and g2 not in robot
                else (g2 if g2 in robot and g1 not in robot else -1)
            )
            foot = self._foot_index.get(mine)
            if foot is None:
                continue
            mujoco.mj_contactForce(m, d, i, self._f6)
            frame = np.asarray(c.frame).reshape(3, 3)
            world = frame.T @ self._f6[:3]
            fz[foot] += abs(float(world[2]))
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, self.cam_site, self._w6, 0)
        cam_x = np.asarray(d.site_xmat[self.cam_site]).reshape(3, 3)[:, 0].copy()
        port = self.port
        return {
            "foot_fz": fz,
            "cam_w": self._w6[:3].copy(),
            "cam_x": cam_x,
            "body_cmd": np.array(port.body_ema, dtype=float),
            "pose_on": float(port.pose_active),
            "tau": np.abs(np.asarray(d.actuator_force[self.act], dtype=float)),
        }

    def info(self) -> dict[str, Any]:
        out = super().info()
        out["walk"] = {"file": self.walk_path.name, "sha256": _sha(self.walk_path)}
        out["delay"] = {"label": self.delay_label, "model": list(self.delay_model)}
        out.pop("skills", None)
        return out


def _sha(path: str | os.PathLike[str] | None) -> str | None:
    if not path or not Path(path).is_file():
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ── the R4 oracle (P0): the exact counter-rotation on the true trunk ─────────────────────────


class HeadingRef:
    """The slow heading frame (training's definition, :mod:`s2_contract`): a low-passed yaw rate
    plus a correction toward the trunk's yaw — no lag in a steady arc, the stride's wobble
    filtered out."""

    def __init__(self, yaw0: float) -> None:
        self.ref = float(yaw0)
        self.rate = 0.0
        self.prev = float(yaw0)

    def step(self, yaw: float, dt: float = DTK) -> float:
        dyaw = math.atan2(math.sin(yaw - self.prev), math.cos(yaw - self.prev))
        self.prev = yaw
        a = dt / C.HEADING_RATE_TAU_S
        self.rate += a * (dyaw / dt - self.rate)
        err = math.atan2(math.sin(yaw - self.ref), math.cos(yaw - self.ref))
        self.ref += (self.rate + err / C.HEADING_TAU_S) * dt
        return self.ref


def _yaw_of_quat(q: Sequence[float]) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _rotvec(R: np.ndarray) -> np.ndarray:
    """The rotation vector of a rotation matrix (small or large angles)."""
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    ang = math.acos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if ang < 1e-6:
        return 0.5 * v
    return v * (ang / (2.0 * math.sin(ang)))


class Oracle:
    """Drives the four head joints so the camera holds a SLOW reference orientation, from the TRUE
    trunk orientation predicted ``lead`` seconds ahead with its angular velocity: the upper bound of
    what a head stabiliser can do on this hardware (three Gauss-Newton steps on the camera site's
    rotational Jacobian per tick, roll about the optical axis weighted 0.3, inside the joints'
    ranges).

    The reference is the camera orientation the head would have at HOME on the current trunk,
    low-passed: its yaw by the slow heading frame (:class:`HeadingRef`), its pitch and roll by an EMA
    (τ 0.8 s) — so only the stride band is removed and the head keeps its natural mean posture (a
    level-camera reference lifts the drooping head ~12° and changes the gait). ``fixed``: the
    control — the head targets held at HOME (no counter-rotation at all)."""

    TAU_S = 0.8

    def __init__(
        self, duck: WalkDuck, lead_s: float, yaw_cmd: float = 0.0, fixed: bool = False
    ) -> None:
        self.duck = duck
        self.fixed = bool(fixed)
        self.lead_s = float(lead_s)
        self.yaw_cmd = float(yaw_cmd)
        self.heading: HeadingRef | None = None
        self.pitch_ref: float | None = None
        self.roll_ref = 0.0
        m = duck.model
        lo, hi = [], []
        for v in duck.head_v:
            j = int(m.dof_jntid[v])
            limited = bool(m.jnt_limited[j])
            lo.append(float(m.jnt_range[j, 0]) if limited else -math.pi)
            hi.append(float(m.jnt_range[j, 1]) if limited else math.pi)
        self.lo, self.hi = np.array(lo), np.array(hi)
        self.home_head = np.array(C.HOME)[list(C.HEAD_IDX)]
        self.ticks = 0

    def __call__(self, targets: np.ndarray) -> np.ndarray:
        out = np.array(targets, dtype=float)
        if self.fixed:
            out[list(C.HEAD_IDX)] = self.home_head
            return out
        duck, mujoco = self.duck, self.duck._mj
        m, d, s = duck.model, duck.data, duck._scratch
        s.qpos[:] = d.qpos
        s.qvel[:] = 0.0
        if self.lead_s > 0:
            quat = np.array(d.qpos[duck.qadr + 3 : duck.qadr + 7])
            mujoco.mju_quatIntegrate(
                quat, np.array(d.qvel[duck.vadr + 3 : duck.vadr + 6]), self.lead_s
            )
            s.qpos[duck.qadr + 3 : duck.qadr + 7] = quat
        # the camera at HOME on this trunk → the slow reference
        s.qpos[duck.head_q] = self.home_head
        mujoco.mj_kinematics(m, s)
        R_home = np.asarray(s.site_xmat[duck.cam_site]).reshape(3, 3)
        yaw_h = math.atan2(R_home[1, 0], R_home[0, 0])
        pitch_h = math.asin(max(-1.0, min(1.0, float(R_home[2, 0]))))
        roll_h = math.atan2(R_home[2, 1], R_home[2, 2])
        if self.heading is None:
            self.heading = HeadingRef(yaw_h)
            self.pitch_ref, self.roll_ref = pitch_h, roll_h
        a = DTK / self.TAU_S
        self.pitch_ref += a * (pitch_h - self.pitch_ref)
        self.roll_ref += a * (
            math.atan2(math.sin(roll_h - self.roll_ref), math.cos(roll_h - self.roll_ref))
        )
        psi = self.heading.step(yaw_h) + self.yaw_cmd
        th = -float(self.pitch_ref)  # the optical axis' z = −sin θ: pitch-up = −θ
        ph = float(self.roll_ref)
        Rz = np.array(
            [
                [math.cos(psi), -math.sin(psi), 0.0],
                [math.sin(psi), math.cos(psi), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        Ry = np.array(
            [[math.cos(th), 0.0, math.sin(th)], [0.0, 1.0, 0.0], [-math.sin(th), 0.0, math.cos(th)]]
        )
        Rx = np.array(
            [[1.0, 0.0, 0.0], [0.0, math.cos(ph), -math.sin(ph)], [0.0, math.sin(ph), math.cos(ph)]]
        )
        R_d = Rz @ Ry @ Rx
        s.qpos[duck.head_q] = d.qpos[duck.head_q]
        for _ in range(3):
            mujoco.mj_kinematics(m, s)
            mujoco.mj_comPos(m, s)
            R_c = np.asarray(s.site_xmat[duck.cam_site]).reshape(3, 3)
            e = _rotvec(R_d @ R_c.T)
            ax = R_c[:, 0]
            w = np.eye(3) - 0.7 * np.outer(ax, ax)  # roll about the optical axis matters less
            mujoco.mj_jacSite(m, s, None, duck._jacr, duck.cam_site)
            J = w @ duck._jacr[:, duck.head_v]
            dq = np.linalg.solve(J.T @ J + 1e-4 * np.eye(4), J.T @ (w @ e))
            s.qpos[duck.head_q] = np.clip(np.array(s.qpos[duck.head_q]) + dq, self.lo, self.hi)
        out[list(C.HEAD_IDX)] = np.array(s.qpos[duck.head_q])
        self.ticks += 1
        return out


# ── conditions ───────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Cond:
    key: str
    kind: str
    params: dict = field(default_factory=dict, hash=False, compare=False)
    group: str = ""
    v5_only: bool = False

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def gate_conditions() -> list[Cond]:
    """The Stage-2 visibility gates (plan §6)."""
    out = [Cond("stand/quiet", "stand", {"lively": False}, "stand")]
    for axis, v in (("roll", 0.2), ("roll", -0.2), ("pitch", 0.2), ("pitch", -0.2)):
        out.append(Cond(f"lean/{axis}{v:+g}r", "lean", {"axis": axis, "value": v, "release": "ramp"},
                        "lean"))  # fmt: skip
    out.append(
        Cond("lean/roll+0.2snap", "lean", {"axis": "roll", "value": 0.2, "release": "snap"}, "lean")
    )
    out.append(
        Cond(
            "lean/pitch-0.2snap",
            "lean",
            {"axis": "pitch", "value": -0.2, "release": "snap"},
            "lean",
        )
    )
    out.append(
        Cond(
            "lean/chest-0.15", "lean", {"axis": "pitch", "value": -0.15, "release": "ramp"}, "lean"
        )
    )
    out.append(
        Cond(
            "lean/ws_roll+0.15", "lean", {"axis": "roll", "value": 0.15, "release": "ramp"}, "lean"
        )
    )
    out.append(
        Cond(
            "lean/ws_roll-0.15", "lean", {"axis": "roll", "value": -0.15, "release": "ramp"}, "lean"
        )
    )
    for v in (-0.020, -0.010, 0.008):
        out.append(
            Cond(
                f"lean/z{v * 1000:+.0f}mm",
                "lean",
                {"axis": "z", "value": v, "release": "ramp"},
                "lean",
            )
        )
    for axis, amp, hz in (("roll", 0.2, 0.1), ("roll", 0.2, 0.25), ("roll", 0.2, 0.5), ("pitch", 0.2, 0.25),
                          ("z", 0.012, 0.25)):  # fmt: skip
        out.append(
            Cond(
                f"sway/{axis}{amp:g}@{hz:g}Hz", "sway", {"axis": axis, "amp": amp, "hz": hz}, "sway"
            )
        )
    out.append(Cond("head/pullback", "head_pullback", {"neck": 0.3, "head": -0.2}, "head"))
    out.append(Cond("head/micro", "head_micro", {"amp": 0.052, "hz": 1.0}, "head"))
    for v in (0.40, 0.55):
        out.append(Cond(f"walk/{v:.2f}/home", "walk", {"vx": v, "head": "home"}, "walk"))
    for yaw in (1.0, -1.0):
        out.append(
            Cond(
                f"walk/0.40/yaw{yaw:+g}", "walk", {"vx": 0.40, "head": "static", "yaw": yaw}, "walk"
            )
        )
    out.append(Cond("walk/0.40/sacc", "walk", {"vx": 0.40, "head": "sacc"}, "walk"))
    out.append(Cond("stopgo/0.40/pose", "stopgo", {"vx": 0.40, "pose": True}, "stopgo"))
    return out


def regression_conditions() -> list[Cond]:
    out = [
        Cond("stopgo/0.40", "stopgo", {"vx": 0.40, "pose": False}, "stopgo"),
        Cond("stopgo/0.55", "stopgo", {"vx": 0.55, "pose": False}, "stopgo"),
        Cond("turn/0.40/+0.6", "turn", {"vx": 0.40, "wz": 0.6}, "turn"),
        Cond("turn/0.40/-0.6", "turn", {"vx": 0.40, "wz": -0.6}, "turn"),
        Cond("sitstand", "sitstand", {}, "posture"),
        Cond("push/stand", "push", {"walking": False}, "push"),
        Cond("push/walk", "push", {"walking": True}, "push"),
        Cond("stand/lively", "stand", {"lively": True}, "stand"),
    ]
    return out


def oracle_conditions() -> list[Cond]:
    out = []
    for v in (0.40, 0.55):
        for lead in (0, 40, 80):
            out.append(Cond(f"oracle/{v:.2f}/lp{lead}", "walk",
                            {"vx": v, "head": "oracle", "lead_ms": lead}, "oracle", True))  # fmt: skip
        out.append(
            Cond(f"oracle/{v:.2f}/fixed", "walk", {"vx": v, "head": "fixed"}, "oracle", True)
        )
    return out


def condition_set(name: str) -> list[Cond]:
    sets = {
        "gates": gate_conditions,
        "regress": regression_conditions,
        "oracle": lambda: (
            oracle_conditions()
            + [c for c in gate_conditions() if c.kind == "walk" and "home" in c.key]
        ),
        "all": lambda: gate_conditions() + regression_conditions(),
    }
    if name not in sets:
        raise ValueError(f"condition set {name!r}: {sorted(sets)}")
    return sets[name]()


def seed_of(key: str, rep: int) -> int:
    return zlib.crc32(f"s2/{key}/{rep}".encode()) & 0x7FFFFFFF


# ── running one trial ────────────────────────────────────────────────────────────────────────


class Excluded(Exception):
    pass


def _run(duck: WalkDuck, seconds: float, each: Callable[[float], object] | None = None) -> None:
    for _ in range(max(0, int(round(seconds * HZ)))):
        if each is not None:
            each(duck.now())
        duck.tick()


class Slew:
    """duck-body's pose target ramp: the client's target moves at ≤ the envelope's rates."""

    def __init__(self) -> None:
        self.cur = np.zeros(3)

    def step(self, goal: Sequence[float]) -> np.ndarray:
        rate = np.array([C.POSE.rate_m_s, C.POSE.rate_rad_s, C.POSE.rate_rad_s]) * DTK
        self.cur = self.cur + np.clip(np.asarray(goal, float) - self.cur, -rate, rate)
        return self.cur


def _axis_vec(axis: str, value: float) -> np.ndarray:
    return np.array([value if axis == "z" else 0.0, value if axis == "roll" else 0.0,
                     value if axis == "pitch" else 0.0])  # fmt: skip


def run_trial(
    cond: Cond,
    *,
    servo: str,
    delay: str,
    rep: int,
    walk: str | None,
    policies: str | None = None,
    mjcf_dir: str | None = None,
) -> dict[str, Any]:
    began = time.perf_counter()
    seed = seed_of(cond.key, rep)
    draw = DR.draw(seed, "recipe" if servo == "bam" else "physical")
    rng = np.random.default_rng(seed + 11)
    duck = WalkDuck(servo=servo, draw=draw, delay=delay, walk=walk, policies=policies,
                    mjcf_dir=mjcf_dir, yaw=float(rng.uniform(-math.pi, math.pi)))  # fmt: skip
    row: dict[str, Any] = {"cond": cond.to_json(), "servo": servo, "delay": delay, "rep": rep,
                           "seed": seed, "dr": draw.to_json(), "duck": duck.info(), "excluded": None}  # fmt: skip
    try:
        ok, _ = duck.stand(timeout=10.0, hold=1.0)
        if not ok:
            raise Excluded("did not stand up after the seated boot")
        _run(duck, float(rng.uniform(1.0, 1.5)), lambda _t: duck.move(0.0))
        if duck.safety.fallen or duck.tilt_deg() > 30.0:
            raise Excluded(f"down before the program (tilt {duck.tilt_deg():.0f}°)")
        t0 = duck.now()
        marks = PROGRAMS[cond.kind](duck, cond, rng)
        _follow_falls(duck)
        tr = duck.trace.arrays()
        row["markers"] = {
            k: (round(v, 4) if isinstance(v, float) else v) for k, v in {"t0": t0, **marks}.items()
        }
        row["metrics"] = METRICS[cond.kind](tr, cond, {"t0": t0, **marks}, duck)
    except Excluded as why:
        row["excluded"] = str(why)
    row["wall_s"] = round(time.perf_counter() - began, 3)
    row["sim_s"] = round(duck.now(), 3)
    return row


def _follow_falls(duck: WalkDuck) -> None:
    """A duck that fell is followed until upright for 1 s, or 8 s after the fall."""
    tr = duck.trace.arrays()
    idx = np.flatnonzero(tr["fallen"] > 0.5)
    if not idx.size:
        return
    t_fall = float(tr["t"][idx[0]])
    streak = 0
    while duck.now() < t_fall + 9.0:
        duck.move(0.0)
        duck.pose(0.0, 0.0, 0.0, active=False)
        duck.tick()
        streak = streak + 1 if duck.upright() else 0
        if streak >= HZ + 1:
            break


# ── programs (each returns its markers) ──────────────────────────────────────────────────────


def p_stand(duck: WalkDuck, cond: Cond, rng: np.random.Generator) -> dict[str, float]:
    head = _lively(rng) if cond.params.get("lively") else None

    def each(t: float) -> None:
        duck.move(0.0)
        if head is not None:
            duck.head(head(t))

    _run(duck, 20.0, each)
    return {"t_end": duck.now()}


def p_lean(duck: WalkDuck, cond: Cond, rng: np.random.Generator) -> dict[str, float]:
    p = cond.params
    goal = _axis_vec(p["axis"], p["value"])
    slew = Slew()
    t_on = duck.now()

    def hold(_t: float) -> None:
        duck.move(0.0)
        duck.pose(*slew.step(goal), active=True)

    _run(duck, 4.5, hold)
    t_off = duck.now()
    if p.get("release") == "ramp":

        def out(_t: float) -> None:
            duck.move(0.0)
            duck.pose(*slew.step(np.zeros(3)), active=True)

        while np.abs(slew.cur).max() > 1e-6 and duck.now() < t_off + 1.5:
            out(duck.now())
            duck.tick()
    t_rel = duck.now()
    duck.pose(0.0, 0.0, 0.0, active=False)
    _run(duck, 2.5, lambda _t: duck.move(0.0))
    return {"t_on": t_on, "t_off": t_off, "t_release": t_rel, "t_end": duck.now()}


def p_sway(duck: WalkDuck, cond: Cond, rng: np.random.Generator) -> dict[str, float]:
    p = cond.params
    dur = max(10.0, 2.5 / p["hz"])
    t_on = duck.now()
    center = -0.006 if p["axis"] == "z" else 0.0

    def each(t: float) -> None:
        duck.move(0.0)
        v = center + p["amp"] * math.sin(2 * math.pi * p["hz"] * (t - t_on))
        duck.pose(*_axis_vec(p["axis"], v), active=True)

    _run(duck, dur, each)
    t_off = duck.now()
    duck.pose(0.0, 0.0, 0.0, active=False)
    _run(duck, 1.5, lambda _t: duck.move(0.0))
    return {"t_on": t_on, "t_off": t_off, "t_end": duck.now()}


def p_head_pullback(duck: WalkDuck, cond: Cond, rng: np.random.Generator) -> dict[str, float]:
    p = cond.params
    t_on = duck.now()
    _run(duck, 2.5, lambda _t: (duck.move(0.0), duck.head((p["neck"], p["head"], 0.0, 0.0))))
    t_off = duck.now()
    _run(duck, 1.5, lambda _t: (duck.move(0.0), duck.head((0.0, 0.0, 0.0, 0.0))))
    return {"t_on": t_on, "t_off": t_off, "t_end": duck.now()}


def p_head_micro(duck: WalkDuck, cond: Cond, rng: np.random.Generator) -> dict[str, float]:
    p = cond.params
    t_on = duck.now()

    def each(t: float) -> None:
        duck.move(0.0)
        duck.head((0.0, 0.0, p["amp"] * math.sin(2 * math.pi * p["hz"] * (t - t_on)), 0.0))

    _run(duck, 8.0, each)
    t_off = duck.now()
    _run(duck, 1.0, lambda _t: (duck.move(0.0), duck.head((0.0, 0.0, 0.0, 0.0))))
    return {"t_on": t_on, "t_off": t_off, "t_end": duck.now()}


def _lively(rng: np.random.Generator) -> Callable[[float], tuple[float, float, float, float]]:
    """OU head motion + saccades (the A100 'lively head')."""
    state = {"y": 0.0, "p": 0.0, "n": 0.0, "r": 0.0, "sac": 0.0, "next": 0.0, "t": None}

    def f(t: float) -> tuple[float, float, float, float]:
        if state["t"] is None:
            state["t"] = t
        for k, sd, lim in (("y", 0.25, 1.2), ("p", 0.2, 0.4), ("n", 0.1, 0.25), ("r", 0.12, 0.3)):
            state[k] += -state[k] * DTK / 1.5 + sd * math.sqrt(2 * DTK / 1.5) * float(
                rng.standard_normal()
            )
            state[k] = max(-lim, min(lim, state[k]))
        if t >= state["next"]:
            state["sac"] = float(rng.choice((-1, 1))) * float(rng.uniform(0.3, 0.8))
            state["next"] = t + float(rng.uniform(0.8, 3.0))
        return (state["n"], state["p"], max(-1.2, min(1.2, state["y"] + state["sac"])), state["r"])

    return f


def p_walk(duck: WalkDuck, cond: Cond, rng: np.random.Generator) -> dict[str, float]:
    p = cond.params
    vx = float(p["vx"])
    kind = p.get("head", "home")
    oracle = None
    if kind in ("oracle", "fixed"):
        oracle = Oracle(duck, p.get("lead_ms", 0) / 1000.0, 0.0, fixed=(kind == "fixed"))
        duck.port.target_hook = oracle
    sacc = {"target": 0.0, "next": 0.0, "sign": float(rng.choice((-1, 1)))}
    t_on = duck.now()

    def each(t: float) -> None:
        duck.move(vx)
        if kind == "static" and t - t_on >= 1.0:
            duck.head((0.0, 0.0, float(p["yaw"]), 0.0))
        elif kind == "sacc" and t - t_on >= 1.0:
            if t >= sacc["next"]:
                if rng.uniform() < 0.7:
                    sacc["sign"] = -sacc["sign"]
                sacc["target"] = sacc["sign"] * float(rng.uniform(0.3, 0.8))
                sacc["next"] = t + float(rng.uniform(0.8, 3.0))
            duck.head((0.0, 0.0, sacc["target"], 0.0))
        else:
            duck.head((0.0, 0.0, 0.0, 0.0))

    _run(duck, 14.0, each)
    t_stop = duck.now()
    duck.head((0.0, 0.0, 0.0, 0.0))
    _run(duck, 2.5, lambda _t: duck.move(0.0))
    duck.port.target_hook = None
    return {"t_on": t_on, "t_stop": t_stop, "t_end": duck.now()}


def p_stopgo(duck: WalkDuck, cond: Cond, rng: np.random.Generator) -> dict[str, float]:
    p = cond.params
    vx = float(p["vx"])
    marks: dict[str, Any] = {"stops": [], "pose_on": [], "pose_off": []}
    waits = (0.0, 0.7, 1.5)
    for cycle in range(3):
        _run(duck, 3.0, lambda _t: duck.move(vx))
        marks["stops"].append(duck.now())
        if p.get("pose"):
            _run(duck, waits[cycle], lambda _t: duck.move(0.0))
            slew = Slew()
            t_p = duck.now()
            marks["pose_on"].append(t_p)

            def each(t: float, t_p: float = t_p, slew: Slew = slew) -> None:
                duck.move(0.0)
                goal = (0.0, 0.15 * math.sin(2 * math.pi * 0.3 * (t - t_p)), -0.10)
                duck.pose(*slew.step(goal), active=True)

            _run(duck, 3.0, each)
            while np.abs(slew.cur).max() > 1e-6:
                duck.move(0.0)
                duck.pose(*slew.step(np.zeros(3)), active=True)
                duck.tick()
            duck.pose(0.0, 0.0, 0.0, active=False)
            marks["pose_off"].append(duck.now())
            _run(duck, 0.5, lambda _t: duck.move(0.0))
        else:
            _run(duck, 2.0, lambda _t: duck.move(0.0))
    _run(duck, 3.0, lambda _t: duck.move(0.0))
    marks["t_end"] = duck.now()
    return marks


def p_turn(duck: WalkDuck, cond: Cond, rng: np.random.Generator) -> dict[str, float]:
    p = cond.params
    t_on = duck.now()
    _run(duck, 10.0, lambda _t: duck.move(p["vx"], 0.0, p["wz"]))
    t_off = duck.now()
    _run(duck, 2.0, lambda _t: duck.move(0.0))
    return {"t_on": t_on, "t_off": t_off, "t_end": duck.now()}


def p_sitstand(duck: WalkDuck, cond: Cond, rng: np.random.Generator) -> dict[str, float]:
    t_sit = duck.now()
    duck.do("sit_toggle")
    _run(duck, 4.0)
    t_rise = duck.now()
    duck.do("sit_toggle")
    _run(duck, 4.0, lambda _t: duck.move(0.0))
    return {"t_sit": t_sit, "t_rise": t_rise, "t_end": duck.now()}


def p_push(duck: WalkDuck, cond: Cond, rng: np.random.Generator) -> dict[str, float]:
    walking = bool(cond.params.get("walking"))
    if walking:
        _run(duck, 2.0, lambda _t: duck.move(0.40))
    theta = float(rng.uniform(0.0, 2 * math.pi))
    dv = float(rng.uniform(1.2, 1.6))
    d = duck.data
    d.qvel[duck.vadr + 0] += dv * math.cos(theta)
    d.qvel[duck.vadr + 1] += dv * math.sin(theta)
    t_push = duck.now()
    down = {"v": False}

    def each(_t: float) -> None:
        down["v"] |= bool(duck.safety.fallen)
        duck.move(0.40 if walking and not down["v"] else 0.0)

    _run(duck, 10.0, each)
    return {"t_push": t_push, "t_end": duck.now(), "dv": dv}


PROGRAMS: dict[str, Callable[..., dict[str, Any]]] = {
    "stand": p_stand, "lean": p_lean, "sway": p_sway, "head_pullback": p_head_pullback,
    "head_micro": p_head_micro, "walk": p_walk, "stopgo": p_stopgo, "turn": p_turn,
    "sitstand": p_sitstand, "push": p_push,
}  # fmt: skip


# ── metrics (traces only) ────────────────────────────────────────────────────────────────────


def _i(tr: Mapping[str, Any], t: float) -> int:
    return int(min(len(tr["t"]) - 1, max(0, np.searchsorted(tr["t"], t - 1e-9))))


def _rpy(tr: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return M.rpy(np.asarray(tr["quat"], float))


def safety(tr: Mapping[str, Any], a: int, b: int) -> dict[str, Any]:
    tilt = np.asarray(tr["tilt"][a:b], float)
    return {
        "fell": bool((np.asarray(tr["fallen"][a:], float) > 0.5).any()),
        "near_fall": bool((tilt > C.NEAR_FALL_TILT_DEG).any()),
        "tilt_max": round(float(tilt.max()) if tilt.size else 0.0, 3),
    }


def leg_rms(tr: Mapping[str, Any], a: int, b: int) -> float:
    tau = np.asarray(tr["tau"][a:b], float)[:, list(C.LEG_IDX)]
    return float(np.sqrt(np.mean(tau**2))) if tau.size else float("nan")


def cam_perp_dps(tr: Mapping[str, Any], a: int, b: int) -> float:
    w = np.asarray(tr["cam_w"][a:b], float)
    x = np.asarray(tr["cam_x"][a:b], float)
    perp = w - (np.sum(w * x, axis=1, keepdims=True)) * x
    return float(np.degrees(np.sqrt(np.mean(np.sum(perp**2, axis=1))))) if len(w) else float("nan")


def highpass_sd(x: np.ndarray, win_s: float = STRIDE_HP_S) -> float:
    """SD of ``x`` minus its centred moving mean over ``win_s`` (the stride band, > ~1 Hz)."""
    k = max(3, int(round(win_s * HZ)) | 1)
    if len(x) < k + 2:
        return float("nan")
    pad = k // 2
    mean = np.convolve(np.asarray(x, float), np.ones(k) / k, mode="valid")  # centred, interior only
    return float(np.std(np.asarray(x, float)[pad : len(x) - pad] - mean))


def lsq_sine(y: np.ndarray, t: np.ndarray, hz: float) -> tuple[float, float]:
    """y ≈ a·sin(ωt) + b·cos(ωt) + c → (amplitude, lag s) with y = A sin(ω(t − lag))."""
    w = 2 * math.pi * hz
    X = np.stack([np.sin(w * t), np.cos(w * t), np.ones_like(t)], axis=1)
    sol, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = float(sol[0]), float(sol[1])
    lag = math.atan2(-b, a)
    lag = (lag + math.radians(30)) % (2 * math.pi) - math.radians(30)
    return math.hypot(a, b), lag / w


def path_speed(tr: Mapping[str, Any], a: int, b: int) -> float:
    pos = np.asarray(tr["pos"], float)
    idx = np.arange(a, b, HZ)
    if len(idx) < 2:
        return float("nan")
    d = np.linalg.norm(np.diff(pos[idx, :2], axis=0), axis=1).sum()
    return float(d / ((idx[-1] - idx[0]) * DTK))


def m_stand(tr, cond, mk, duck) -> dict[str, Any]:
    a, b = _i(tr, mk["t0"] + 1.0), _i(tr, mk["t_end"])
    pos = np.asarray(tr["pos"], float)
    roll, pitch, yaw = _rpy(tr)
    drift = float(np.linalg.norm(pos[b, :2] - pos[a, :2]))
    return {
        **safety(tr, a, b),
        "steps": int(np.asarray(tr["td"][a:b]).sum()),
        "stepped": bool(np.asarray(tr["td"][a:b]).sum() > 0),
        "drift_cm": round(100 * drift, 3),
        "walkoff": bool(drift > WALKOFF_M),
        "yaw_drift_deg": round(float(np.degrees(abs(M.wrap(yaw[b] - yaw[a])))), 3),
        "leg_rms_nm": round(leg_rms(tr, a, b), 5),
        "sway_roll_sd_deg": round(float(np.degrees(np.std(roll[a:b]))), 4),
        "sway_pitch_sd_deg": round(float(np.degrees(np.std(pitch[a:b]))), 4),
        "neck_droop_rad": round(
            float(np.mean(np.asarray(tr["q"][a:b])[:, C.NECK_PITCH]) - C.HOME[C.NECK_PITCH]), 4
        ),
    }


def _axis_series(tr: Mapping[str, Any], axis: str) -> tuple[np.ndarray, float]:
    """(achieved series in deg or mm, unit per command unit)."""
    roll, pitch, _ = _rpy(tr)
    if axis == "roll":
        return np.degrees(roll), math.degrees(1.0)
    if axis == "pitch":
        return np.degrees(pitch), math.degrees(1.0)
    return 1000.0 * np.asarray(tr["pos"], float)[:, 2], 1000.0


def m_lean(tr, cond, mk, duck) -> dict[str, Any]:
    p = cond.params
    on, off = _i(tr, mk["t_on"]), _i(tr, mk["t_off"])
    rel, end = _i(tr, mk["t_release"]), _i(tr, mk["t_end"])
    b0 = _i(tr, mk["t_on"] - 1.0)
    series, unit = _axis_series(tr, p["axis"])
    base = float(np.mean(series[b0:on]))
    h0 = _i(tr, mk["t_off"] - 1.5)
    achieved = float(np.mean(series[h0:off]) - base)
    cmd = float(p["value"]) * unit
    sgn = 1.0 if cmd >= 0 else -1.0
    reach = (
        np.flatnonzero(sgn * (series[on:off] - base) >= 0.9 * abs(achieved))
        if abs(achieved) > 1e-6
        else []
    )
    ang = np.asarray(tr["angvel"], float)
    rel_rate = (
        float(np.degrees(np.max(np.linalg.norm(ang[rel : min(end, rel + HZ), :2], axis=1))))
        if end > rel
        else 0.0
    )
    fz = np.asarray(tr["foot_fz"][h0:off], float)
    tot = fz.sum(axis=1)
    ok = tot > 1.0
    share = fz[ok].max(axis=1) / tot[ok] if ok.any() else np.array([0.5])
    load_right = float(np.mean(fz[ok, RIGHT_FOOT] / tot[ok])) if ok.any() else 0.5
    return {
        **safety(tr, on, end),
        "achieved": round(achieved, 4),
        "achieved_abs": round(abs(achieved), 4),
        "gain": round(achieved / cmd if cmd else 0.0, 4),
        "t90_s": round(float(reach[0] + 1) * DTK, 3) if len(reach) else None,
        "steps_hold": int(np.asarray(tr["td"][on:off]).sum()),
        "stepped": bool(np.asarray(tr["td"][on:rel]).sum() > 0),
        "release_rate_dps": round(rel_rate, 3),
        "load_share_max": round(float(np.mean(share)), 4),
        "load_right": round(load_right, 4),
        "leg_rms_nm": round(leg_rms(tr, h0, off), 5),
        "base": round(base, 4),
    }


def m_sway(tr, cond, mk, duck) -> dict[str, Any]:
    p = cond.params
    on, off, end = _i(tr, mk["t_on"]), _i(tr, mk["t_off"]), _i(tr, mk["t_end"])
    series, unit = _axis_series(tr, p["axis"])
    a = _i(tr, mk["t_on"] + max(2.0, 0.5 / p["hz"]))
    t = np.asarray(tr["t"][a:off], float) - mk["t_on"]
    amp, lag = lsq_sine(series[a:off] - series[a:off].mean(), t, p["hz"])
    return {
        **safety(tr, on, end),
        "amp": round(amp, 4),
        "gain": round(amp / (p["amp"] * unit), 4),
        "lag_ms": round(1000.0 * lag, 2),
        "steps": int(np.asarray(tr["td"][on:off]).sum()),
        "stepped": bool(np.asarray(tr["td"][on:off]).sum() > 0),
    }


def m_head_pullback(tr, cond, mk, duck) -> dict[str, Any]:
    p = cond.params
    on, off, end = _i(tr, mk["t_on"]), _i(tr, mk["t_off"]), _i(tr, mk["t_end"])
    q = np.asarray(tr["q"], float)
    b0 = _i(tr, mk["t_on"] - 0.5)
    h0 = _i(tr, mk["t_off"] - 1.0)
    dneck = float(np.mean(q[h0:off, C.NECK_PITCH]) - np.mean(q[b0:on, C.NECK_PITCH]))
    dhead = float(np.mean(q[h0:off, C.HEAD_PITCH]) - np.mean(q[b0:on, C.HEAD_PITCH]))
    cam = np.asarray(tr["cam_pitch"], float)
    return {
        **safety(tr, on, end),
        "neck_gain": round(dneck / p["neck"], 4),
        "head_gain": round(dhead / p["head"], 4),
        "cam_up_deg": round(float(np.mean(cam[h0:off]) - np.mean(cam[b0:on])), 3),
        "steps": int(np.asarray(tr["td"][on:end]).sum()),
    }


def m_head_micro(tr, cond, mk, duck) -> dict[str, Any]:
    p = cond.params
    on, off, end = _i(tr, mk["t_on"]), _i(tr, mk["t_off"]), _i(tr, mk["t_end"])
    a = _i(tr, mk["t_on"] + 2.0)
    q = np.asarray(tr["q"], float)[:, C.HEAD_YAW]
    t = np.asarray(tr["t"][a:off], float) - mk["t_on"]
    amp, lag = lsq_sine(q[a:off] - q[a:off].mean(), t, p["hz"])
    return {**safety(tr, on, end), "gain": round(amp / p["amp"], 4), "lag_ms": round(1000 * lag, 2),
            "steps": int(np.asarray(tr["td"][on:end]).sum())}  # fmt: skip


def gaze_numbers(tr: Mapping[str, Any], a: int, b: int) -> dict[str, float]:
    cam_x = np.asarray(tr["cam_x"], float)
    cam_yaw = np.unwrap(np.arctan2(cam_x[:, 1], cam_x[:, 0]))
    _, _, yaw = _rpy(tr)
    trunk_yaw = np.unwrap(yaw)
    cam_pitch = np.asarray(tr["cam_pitch"], float)
    q = np.asarray(tr["q"], float)
    neck = q[a:b][:, list(C.HEAD_IDX)]
    minutes = max((b - a) * DTK / 60.0, 1e-6)
    return {
        "cam_perp_dps": round(cam_perp_dps(tr, a, b), 3),
        "head_yaw_stride_sd_deg": round(math.degrees(highpass_sd(cam_yaw[a:b])), 4),
        "trunk_yaw_stride_sd_deg": round(math.degrees(highpass_sd(trunk_yaw[a:b])), 4),
        "cam_pitch_stride_sd_deg": round(highpass_sd(cam_pitch[a:b]), 4),
        "cam_pitch_mean_deg": round(float(np.mean(cam_pitch[a:b])), 3),
        "neck_travel_rad_min": round(float(np.abs(np.diff(neck, axis=0)).sum() / minutes), 3),
    }


def m_walk(tr, cond, mk, duck) -> dict[str, Any]:
    p = cond.params
    on, stop, end = _i(tr, mk["t_on"]), _i(tr, mk["t_stop"]), _i(tr, mk["t_end"])
    a, b = _i(tr, mk["t_on"] + 3.0), _i(tr, mk["t_on"] + 12.0)
    q = np.asarray(tr["q"], float)
    out = {**safety(tr, on, end), "speed": round(path_speed(tr, a, b), 5), **gaze_numbers(tr, a, b)}
    if p.get("head") == "static":
        out["yaw_delivered"] = round(
            float(np.mean(q[a:b, C.HEAD_YAW] - C.HOME[C.HEAD_YAW]) / p["yaw"]), 4
        )
    pos = np.asarray(tr["pos"], float)
    v = M.chord_speed(pos)
    still = np.flatnonzero(v[stop:end] < 0.02)
    out["stop_s"] = round(float(still[0] + 1) * DTK, 3) if still.size else None
    out["coast_cm"] = round(
        100 * float(np.linalg.norm(pos[min(end, stop + 100) - 1, :2] - pos[stop, :2])), 3
    )
    out["steps_s"] = round(float(np.asarray(tr["td"][a:b]).sum()) / ((b - a) * DTK), 3)
    return out


def m_stopgo(tr, cond, mk, duck) -> dict[str, Any]:
    a, end = _i(tr, mk["t0"]), _i(tr, mk["t_end"])
    out = {**safety(tr, a, end)}
    pos = np.asarray(tr["pos"], float)
    v = M.chord_speed(pos)
    stops = []
    for s in mk["stops"]:
        i = _i(tr, s)
        still = np.flatnonzero(v[i : min(end, i + 100)] < 0.02)
        stops.append(float(still[0] + 1) * DTK if still.size else None)
    out["stop_s"] = stops
    if mk.get("pose_on"):
        steps = 0
        for on, off in zip(mk["pose_on"], mk["pose_off"], strict=False):
            steps += int(np.asarray(tr["td"][_i(tr, on) : _i(tr, off)]).sum())
        out["steps_pose"] = steps
    return out


def m_turn(tr, cond, mk, duck) -> dict[str, Any]:
    p = cond.params
    on, off, end = _i(tr, mk["t_on"]), _i(tr, mk["t_off"]), _i(tr, mk["t_end"])
    a = _i(tr, mk["t_on"] + 3.0)
    _, _, yaw = _rpy(tr)
    yw = np.unwrap(yaw)
    rate = float((yw[off] - yw[a]) / ((off - a) * DTK))
    return {
        **safety(tr, on, end),
        "yaw_gain": round(rate / p["wz"], 4),
        "speed": round(path_speed(tr, a, off), 5),
    }


def m_sitstand(tr, cond, mk, duck) -> dict[str, Any]:
    a, end = _i(tr, mk["t_sit"]), _i(tr, mk["t_end"])
    z = np.asarray(tr["pos"], float)[:, 2]
    return {**safety(tr, a, end), "sit_ok": bool(z[_i(tr, mk["t_rise"] - 0.5)] < 0.08),
            "rise_ok": bool(z[_i(tr, mk["t_end"] - 0.5)] > 0.10)}  # fmt: skip


def m_push(tr, cond, mk, duck) -> dict[str, Any]:
    p0, end = _i(tr, mk["t_push"]), _i(tr, mk["t_end"])
    fallen = np.asarray(tr["fallen"], float) > 0.5
    idx = np.flatnonzero(fallen[p0 : min(end, p0 + 150)])
    out: dict[str, Any] = {
        "fell": bool(idx.size),
        "tilt_max": round(float(np.max(tr["tilt"][p0:end])), 2),
    }
    if idx.size:
        fi = p0 + int(idx[0])
        up = np.asarray(tr["tilt"], float) < 25.0
        run, righted = 0, None
        for i in range(fi + 1, len(up)):
            run = run + 1 if up[i] else 0
            if run >= HZ:
                righted = i
                break
        t_r = None if righted is None else (righted - fi) * DTK
        out["righted_s"] = None if t_r is None else round(t_r, 3)
        out["righted_8s"] = bool(t_r is not None and t_r <= 8.0)
    return out


METRICS: dict[str, Callable[..., dict[str, Any]]] = {
    "stand": m_stand, "lean": m_lean, "sway": m_sway, "head_pullback": m_head_pullback,
    "head_micro": m_head_micro, "walk": m_walk, "stopgo": m_stopgo, "turn": m_turn,
    "sitstand": m_sitstand, "push": m_push,
}  # fmt: skip


# ── the runner ───────────────────────────────────────────────────────────────────────────────


def _done(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            keys.add(json.loads(line)["key"])
        except (ValueError, KeyError):
            continue
    return keys


def _rows(path: Path) -> list[dict[str, Any]]:
    out = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def cmd_run(args: argparse.Namespace) -> int:
    from duckmind.skilltrain import gate as GATE

    policies = str(Path(args.policies).expanduser() if args.policies else paths.policy_dir())
    conds = condition_set(args.set)
    if args.arm != "v5":
        conds = [c for c in conds if not c.v5_only]
    if args.only:
        conds = [c for c in conds if any(s in c.key for s in args.only)]
    walk = None if args.arm == "v5" else args.walk
    if args.arm != "v5" and not walk:
        raise SystemExit("--walk FILE.onnx is required for a candidate arm")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / f"rows-{args.servo}-{args.delay}-{args.arm}.jsonl"
    done = _done(rows_path)
    guard = GATE.Guard(not args.no_guard)
    todo = [
        (c, r)
        for c in conds
        for r in range(args.n)
        if f"{c.key}/{args.servo}/{args.delay}/{args.arm}/{r}" not in done
    ]
    meta = {"servo": args.servo, "delay": args.delay, "arm": args.arm, "walk": walk, "walk_sha256": _sha(walk) if walk else _sha(Path(policies) / WALK_FILE),
            "n": args.n, "set": args.set, "conditions": [c.key for c in conds], "policies": policies}  # fmt: skip
    (out / f"meta-{args.servo}-{args.delay}-{args.arm}.json").write_text(json.dumps(meta, indent=1))
    print(f"{args.servo} {args.delay} {args.arm}: {len(todo)} trials to run", flush=True)
    began = time.monotonic()
    for i, (c, rep) in enumerate(todo):
        guard.wait()
        try:
            row = run_trial(
                c,
                servo=args.servo,
                delay=args.delay,
                rep=rep,
                walk=walk,
                policies=policies,
                mjcf_dir=args.mjcf_dir,
            )
        except Exception as error:  # one broken trial must not stop the rest
            row = {"cond": c.to_json(), "servo": args.servo, "delay": args.delay, "rep": rep,
                   "excluded": f"error: {type(error).__name__}: {error}"}  # fmt: skip
        row["key"] = f"{c.key}/{args.servo}/{args.delay}/{args.arm}/{rep}"
        row["arm"] = args.arm
        with rows_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            el = time.monotonic() - began
            print(f"  {i + 1}/{len(todo)} {row['key']} ({el:.0f} s)", flush=True)
    print(f"done in {time.monotonic() - began:.0f} s; guard pauses {guard.pauses}, realtime min "
          f"{min(guard.readings, default=None)}", flush=True)  # fmt: skip
    return 0


NUMBERS = (
    "achieved", "achieved_abs", "gain", "t90_s", "release_rate_dps", "load_share_max", "load_right", "amp",
    "lag_ms", "neck_gain", "head_gain", "cam_up_deg", "speed", "cam_perp_dps", "head_yaw_stride_sd_deg",
    "trunk_yaw_stride_sd_deg", "cam_pitch_stride_sd_deg", "cam_pitch_mean_deg", "neck_travel_rad_min",
    "yaw_delivered", "stop_s", "coast_cm", "steps_s", "yaw_gain", "leg_rms_nm", "drift_cm", "tilt_max",
    "sway_roll_sd_deg", "sway_pitch_sd_deg", "neck_droop_rad", "righted_s", "steps_hold", "steps",
)  # fmt: skip
FLAGS = ("fell", "near_fall", "stepped", "walkoff", "sit_ok", "rise_ok", "righted_8s")


def summarize_cell(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if not r.get("excluded")]
    out: dict[str, Any] = {"n": len(ok), "excluded": len(rows) - len(ok)}
    for flag in FLAGS:
        vals = [r["metrics"].get(flag) for r in ok if r["metrics"].get(flag) is not None]
        if vals:
            out[flag] = M.rate(sum(bool(v) for v in vals), len(vals))
    for num in NUMBERS:
        vals = []
        for r in ok:
            v = r["metrics"].get(num)
            if isinstance(v, list):
                vals.extend(x for x in v if x is not None)
            elif v is not None:
                vals.append(v)
        if vals:
            out[num] = M.dist(vals)
    return out


def load(out: Path) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    found: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for path in sorted(out.glob("rows-*.jsonl")):
        _, servo, delay, arm = path.stem.split("-", 3)
        found[(servo, delay, arm)] = _rows(path)
    return found


def paired(
    base: Sequence[Mapping[str, Any]], cand: Sequence[Mapping[str, Any]], num: str
) -> dict[str, Any] | None:
    """candidate − v5 on the same ducks (same condition and rep): p50 and the mean with a 95 % CI."""
    b = {(r["cond"]["key"], r["rep"]): r for r in base if not r.get("excluded")}
    diffs = []
    for r in cand:
        if r.get("excluded"):
            continue
        o = b.get((r["cond"]["key"], r["rep"]))
        if o is None:
            continue
        x, y = r["metrics"].get(num), o["metrics"].get(num)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            diffs.append(float(x) - float(y))
    if not diffs:
        return None
    arr = np.array(diffs)
    se = float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else float("nan")
    return {"n": len(arr), "mean": round(float(arr.mean()), 4), "ci95": [round(float(arr.mean() - 1.96 * se), 4),
            round(float(arr.mean() + 1.96 * se), 4)], "p50": round(float(np.median(arr)), 4)}  # fmt: skip


def summarize(out: Path) -> dict[str, Any]:
    data = load(out)
    cells: dict[str, dict[str, Any]] = {}
    for (servo, delay, arm), rows in data.items():
        by: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            by.setdefault(r["cond"]["key"], []).append(r)
        for key, rs in by.items():
            cells.setdefault(key, {})[f"{servo}/{delay}/{arm}"] = summarize_cell(rs)
    comparisons: dict[str, Any] = {}
    for (servo, delay, arm), rows in data.items():
        if arm == "v5" or (servo, delay, "v5") not in data:
            continue
        base = data[(servo, delay, "v5")]
        comp: dict[str, Any] = {}
        for num in (
            "speed",
            "cam_perp_dps",
            "head_yaw_stride_sd_deg",
            "yaw_gain",
            "leg_rms_nm",
            "stop_s",
        ):
            got = paired(base, rows, num)
            if got:
                comp[num] = got
        comparisons[f"{servo}/{delay}/{arm}"] = comp
    return {"format": "duckmind-s2-mac-1", "cells": cells, "paired_vs_v5": comparisons}


def verdicts(s: Mapping[str, Any], stage: str, arm: str, servo: str, delay: str) -> dict[str, Any]:
    """The plan's gates (:mod:`s2_contract`) from a Mac summary for one arm × servo × delay, against
    the v5 arm on the same ducks where a gate is relative. With n = 30 per cell the rate gates mostly
    stay UNRESOLVED (the A100 battery resolves them); the visibility gates are decided here."""
    cells = s.get("cells", {})
    tag, base = f"{servo}/{delay}/{arm}", f"{servo}/{delay}/v5"

    def p50(key: str, num: str, label: str = tag) -> float | None:
        d = (cells.get(key) or {}).get(label, {}).get(num)
        return None if not d or not d.get("n") else float(d["p50"])

    def pooled(prefixes: tuple[str, ...], flag: str, label: str = tag) -> tuple[int, int] | None:
        k = n = 0
        for key, per in cells.items():
            if key.startswith(prefixes) and label in per and per[label].get(flag):
                k += int(per[label][flag]["k"])
                n += int(per[label][flag]["n"])
        return (k, n) if n else None

    out: dict[str, Any] = {}

    def put(key: str, value: Any, v5: Any = None, detail: Any = None) -> None:
        gate = C.GATE_BY_KEY[key]
        out[key] = {"value": value, "verdict": C.verdict(gate, value, v5), "detail": detail}

    rolls = [p50(k, "achieved_abs") for k in ("lean/roll+0.2r", "lean/roll-0.2r")]
    rolls = [v for v in rolls if v is not None]
    put("lean_roll_deg", min(rolls) if rolls else None, detail=rolls)
    pitches = [p50(k, "achieved_abs") for k in ("lean/pitch+0.2r", "lean/pitch-0.2r")]
    pitches = [v for v in pitches if v is not None]
    put("lean_pitch_deg", min(pitches) if pitches else None, detail=pitches)
    amps = {k: round(g * math.degrees(0.2), 2) for k in ("sway/roll0.2@0.1Hz", "sway/roll0.2@0.25Hz", "sway/roll0.2@0.5Hz",
            "sway/pitch0.2@0.25Hz") if (g := p50(k, "gain")) is not None}  # fmt: skip
    put("sway_amp_deg", min(amps.values()) if amps else None, detail=amps)
    lags = {k: v for k in amps if (v := p50(k, "lag_ms")) is not None}
    put("sway_lag_ms", max(lags.values()) if lags else None, detail=lags)
    put("pose_stepped", pooled(("lean/", "sway/"), "stepped"))
    put("release_rate_dps", max([v for k in ("lean/roll+0.2snap", "lean/pitch-0.2snap") if (v := p50(k, "release_rate_dps")) is not None],
                                default=None))  # fmt: skip
    ws = [
        v
        for k in ("lean/ws_roll+0.15", "lean/ws_roll-0.15")
        if (v := p50(k, "load_share_max")) is not None
    ]
    put("weight_shift_ratio", min(ws) if ws else None, detail=ws)
    chest = p50("lean/chest-0.15", "achieved")
    put("chest_out_deg", -chest if chest is not None else None)
    crouch = p50("lean/z-20mm", "achieved")
    put("crouch_mm", -crouch if crouch is not None else None)
    put("neck_pullback_gain", p50("head/pullback", "neck_gain"))
    put("yaw_micro_gain", p50("head/micro", "gain"))
    lr, lb = p50("stand/quiet", "leg_rms_nm"), p50("stand/quiet", "leg_rms_nm", base)
    put(
        "stand_leg_rms_ratio",
        round(lr / lb, 4) if lr and lb else None,
        detail={"cand": lr, "v5": lb},
    )
    put("pose_falls", pooled(("lean/", "sway/"), "fell"))
    ratios, sd = {}, {}
    for v in ("0.40", "0.55"):
        c, b = p50(f"walk/{v}/home", "cam_perp_dps"), p50(f"walk/{v}/home", "cam_perp_dps", base)
        if c is not None and b:
            ratios[v] = round(c / b, 3)
        hs, hb = (
            p50(f"walk/{v}/home", "head_yaw_stride_sd_deg"),
            p50(f"walk/{v}/home", "head_yaw_stride_sd_deg", base),
        )
        if hs is not None and hb:
            sd[v] = round(hs / hb, 3)
    if stage == "gaze":
        put("cam_perp_ratio", max(ratios.values()) if ratios else None, detail=ratios)
        put("head_yaw_stride_sd_ratio", max(sd.values()) if sd else None, detail=sd)
        dl = [
            v
            for k in ("walk/0.40/yaw+1", "walk/0.40/yaw-1")
            if (v := p50(k, "yaw_delivered")) is not None
        ]
        put("head_yaw_delivered", min(dl) if dl else None, detail=dl)
        put("neck_travel_rad_min", p50("walk/0.40/home", "neck_travel_rad_min"))
    else:
        put("cam_perp_regress_ratio", max(ratios.values()) if ratios else None, detail=ratios)
    put("walk_falls", pooled(("walk/", "turn/"), "fell"))
    put("stopgo_near_falls", pooled(("stopgo/",), "near_fall"))
    put("push_righting_8s", pooled(("push/",), "righted_8s"))
    rt = [p50(k, "righted_s") for k in ("push/stand", "push/walk")]
    rb = [p50(k, "righted_s", base) for k in ("push/stand", "push/walk")]
    extra = [a - b for a, b in zip(rt, rb, strict=True) if a is not None and b is not None]
    put(
        "righting_p50_extra_s",
        round(max(extra), 3) if extra else None,
        detail={"cand": rt, "v5": rb},
    )
    put("sitstand_falls", pooled(("sitstand",), "fell"))
    sp = {v: round(c / b, 4) for v in ("0.40", "0.55")
          if (c := p50(f"walk/{v}/home", "speed")) is not None and (b := p50(f"walk/{v}/home", "speed", base))}  # fmt: skip
    put("walk_speed_ratio", max(sp.values(), key=lambda x: abs(x - 1)) if sp else None, detail=sp)
    tg = {k: round(c / b, 4) for k in ("turn/0.40/+0.6", "turn/0.40/-0.6")
          if (c := p50(k, "yaw_gain")) is not None and (b := p50(k, "yaw_gain", base))}  # fmt: skip
    put("turn_gain_ratio", max(tg.values(), key=lambda x: abs(x - 1)) if tg else None, detail=tg)
    put("stop_s_p50", p50("walk/0.55/home", "stop_s"))
    put(
        "stand_stepped",
        pooled(("stand/quiet",), "stepped"),
        v5=pooled(("stand/quiet",), "stepped", base),
    )
    put(
        "walkoffs",
        pooled(("stand/quiet",), "walkoff"),
        v5=pooled(("stand/quiet",), "walkoff", base),
    )
    return out


def report_md(s: Mapping[str, Any]) -> str:
    lines = [
        "# Stage-2 Mac gates (tier 2)",
        "",
        "k/n [Wilson 95 %]; numbers p50 (p10–p90) over ducks.",
        "",
    ]
    for key, per in sorted(s["cells"].items()):
        lines.append(f"## {key}")
        lines.append("")
        lines.append("| servo/delay/arm | n | falls | near-falls | stepped | key numbers |")
        lines.append("|---|---|---|---|---|---|")
        for label, c in sorted(per.items()):

            def r(name: str) -> str:
                v = c.get(name)
                if not v:
                    return "—"
                lo, hi = v["wilson95"]
                return f"{v['k']}/{v['n']} [{100 * lo:.1f}–{100 * hi:.1f}%]"

            nums = []
            for num in NUMBERS:
                d = c.get(num)
                if d and d.get("n"):
                    nums.append(f"{num} {d['p50']:.3g} ({d['p10']:.3g}–{d['p90']:.3g})")
            lines.append(
                f"| {label} | {c['n']} | {r('fell')} | {r('near_fall')} | {r('stepped')} | {'; '.join(nums[:8])} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def cmd_summarize(args: argparse.Namespace) -> int:
    out = Path(args.out)
    s = summarize(out)
    s["walk_sha256"] = {}  # which file each servo/delay/arm evaluated (s2_abilities checks it)
    for meta in sorted(out.glob("meta-*.json")):
        m = json.loads(meta.read_text())
        s["walk_sha256"][f"{m['servo']}/{m['delay']}/{m['arm']}"] = m.get("walk_sha256")
    s["verdicts"] = {}
    for servo, delay, arm in load(out):
        if arm == "v5":
            continue
        s["verdicts"][f"{servo}/{delay}/{arm}"] = verdicts(s, args.stage, arm, servo, delay)
    (out / "summary.json").write_text(json.dumps(s, indent=1, ensure_ascii=False))
    (out / "report.md").write_text(report_md(s), encoding="utf-8")
    print(f"wrote {out / 'summary.json'} and {out / 'report.md'}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m duckmind.skilltrain.s2_mac")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--servo", choices=("mjcf", "bam"), required=True)
    run.add_argument("--delay", default="d0", help="dN (ms, multiple of 5) or jLO-HI (steps)")
    run.add_argument("--arm", default="v5", help="v5, or a label for the candidate (with --walk)")
    run.add_argument("--walk", default=None, help="the candidate walk network (ONNX)")
    run.add_argument("--set", default="gates", choices=("gates", "regress", "oracle", "all"))
    run.add_argument("--n", type=int, default=30)
    run.add_argument("--out", required=True)
    run.add_argument("--only", nargs="*")
    run.add_argument("--policies")
    run.add_argument("--mjcf-dir")
    run.add_argument("--no-guard", action="store_true")
    run.set_defaults(func=cmd_run)
    su = sub.add_parser("summarize")
    su.add_argument("--out", required=True)
    su.add_argument("--stage", default="pose", choices=C.STAGES)
    su.set_defaults(func=cmd_summarize)
    args = ap.parse_args(argv)
    return int(args.func(args))


__all__ = [
    "Cond",
    "HeadingRef",
    "Oracle",
    "WalkDuck",
    "condition_set",
    "delay_of",
    "highpass_sd",
    "lsq_sine",
    "run_trial",
    "summarize",
]

if __name__ == "__main__":
    sys.exit(main())
