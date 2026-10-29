"""Hop feasibility, part (b): the ceiling of a two-foot hop and of a high step for this body.

hop:feasibility (2026-09-28). A short CPU trajectory optimisation on the official duck — the
``scene_allcollisions`` body, both servo models of the Stage-1 gates (``mjcf``: the MJCF's XML
servos, force range ±0.96 N·m, what the owner's live simulator runs; ``bam``: training's BAM XL330
m6 with the 1.75 A firmware current limit, ceiling kt·I = 0.641 N·m, back-EMF, the load-dependent
gearbox friction) — through robotd's own command path: one joint target per 20 ms tick, the legs'
low-pass 0.7, the ±π clamp, an optional servo command delay (0–40 ms, Stage 1b's finding). The head
stays at HOME. Open loop: an upper-bound *search* for what the joints can do, not a controller.

Three motion families (:data:`FAMILIES`), each a handful of keyframes whose poses and durations
CMA-ES tunes (``cma``) from the standing duck (the STAND keyframe, settled 1 s):

* ``hop`` — both legs mirrored: crouch → push (targets beyond full extension, so the servos
  saturate) → tuck → land → hold. Score: the **two-foot clearance** (the lower sole's lowest
  point at the flight's best instant: both feet at least that high at once) and the
  flight time; a fall, a tilt beyond 30°, the trunk/head touching the floor or a servo-housing
  impact is penalised.
* ``hopf`` — the same, scored on the forward distance both feet travel in the air as well (a
  hop *over* a 4 cm strip needs every sole to travel ≥ 9.4 cm in one flight: 4 cm + a 5.4 cm sole).
* ``step`` — the left leg swings, the right stands: shift (hip roll) → lift → reach → place →
  settle. Score: the swing sole's lowest point (its lift) while the stance foot stays down, and how
  far forward it is placed; the same penalties.

Every rollout records, per physics step, each servo's actuator force and joint speed (the
operating points against the servo model's torque–speed envelope), the floor forces on the feet,
the XL330 housings, the trunk and the head, the whole-body centre of mass, and robotd's fall rule.
:func:`evaluate` re-runs a found trajectory on independent DR ducks (both servo models; the
Stage-1 gates' draws; a command delay drawn 0–40 ms) — the ceiling as a distribution, with Wilson
intervals, not one lucky duck.

    python -m duckmind.skilltrain.hop_trajopt optimise --family hop --servo bam --out DIR
    python -m duckmind.skilltrain.hop_trajopt evaluate --best DIR/best-hop-bam.json --n 30 --out DIR
"""

from __future__ import annotations

import argparse
import json
import math
import time
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

HOME = np.array(
    [0.0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0.0, 0.0,
     0.0, 0.0873, 0.4579, 0.0049, -0.4530]
)  # fmt: skip
LEGS = np.array([0, 1, 2, 3, 4, 9, 10, 11, 12, 13])
L_PITCH = (2, 3, 4)  # hip_pitch, knee, ankle (left); the right leg mirrors: 11, 12, 13 negated
R_PITCH = (11, 12, 13)
L_ROLL, R_ROLL = 1, 10
TICK_S, SUBSTEPS = 0.02, 4
LEG_LP, HEAD_LP = 0.7, 0.5
FALL_Z, FALL_DEBOUNCE_S = -0.5, 0.2
HOUSING_N = 2.0
G = 9.81
SETTLE_S = 1.0
STRIP_OVER_M = 0.04 + 0.054  # a 4 cm strip + one 5.4 cm sole: the flight a hop *over* it needs


# ── the motion families: keyframes from a parameter vector ─────────────────────────────────


@dataclass(frozen=True)
class Param:
    name: str
    lo: float
    hi: float


def _hop_params() -> list[Param]:
    P = Param
    return [
        P("c_hip", 0.0, 0.9), P("c_knee", 0.0, 1.6), P("c_ankle", 0.0, 0.9),
        P("t_c", 0.10, 0.50), P("t_h", 0.0, 0.30),
        P("p_hip", -1.2, 0.3), P("p_knee", -2.0, 0.3), P("p_ankle", -1.2, 0.3), P("t_p", 0.02, 0.20),
        P("k_hip", -0.2, 1.2), P("k_knee", -0.2, 2.0), P("k_ankle", -0.8, 1.0), P("t_k", 0.02, 0.20),
        P("l_hip", -0.2, 0.8), P("l_knee", -0.3, 1.4), P("l_ankle", -0.3, 0.8), P("t_l", 0.10, 0.40),
    ]  # fmt: skip


def _step_params() -> list[Param]:
    P = Param
    return [
        P("s_roll_l", -0.35, 0.35), P("s_roll_r", -0.35, 0.35), P("s_rhip", -0.3, 0.3),
        P("s_rankle", -0.3, 0.3), P("t_s", 0.2, 0.8),
        P("u_hip", -0.2, 1.4), P("u_knee", -0.2, 2.0), P("u_ankle", -0.8, 1.2),
        P("u_rhip", -0.4, 0.4), P("u_rknee", -0.4, 0.8), P("u_rankle", -0.4, 0.4), P("t_u", 0.1, 0.6),
        P("r_hip", -0.4, 1.4), P("r_knee", -0.4, 2.0), P("r_ankle", -0.8, 1.2), P("t_r", 0.1, 0.6),
        P("d_hip", -0.6, 0.8), P("d_knee", -0.4, 1.4), P("d_ankle", -0.8, 0.8), P("t_d", 0.1, 0.6),
    ]  # fmt: skip


FAMILIES = {"hop": _hop_params, "hopf": _hop_params, "step": _step_params}


def _mirror(q: np.ndarray, left: Sequence[float]) -> np.ndarray:
    q = q.copy()
    for k, (li, ri) in enumerate(zip(L_PITCH, R_PITCH, strict=True)):
        q[li] += left[k]
        q[ri] -= left[k]
    return q


def schedule(family: str, x: Sequence[float]) -> list[tuple[np.ndarray, float]]:
    """(target pose, seconds held) keyframes, starting from HOME."""
    p = {q.name: float(v) for q, v in zip(FAMILIES[family](), x, strict=True)}
    if family in ("hop", "hopf"):
        crouch = _mirror(HOME, (p["c_hip"], p["c_knee"], p["c_ankle"]))
        push = _mirror(HOME, (p["p_hip"], p["p_knee"], p["p_ankle"]))
        tuck = _mirror(HOME, (p["k_hip"], p["k_knee"], p["k_ankle"]))
        land = _mirror(HOME, (p["l_hip"], p["l_knee"], p["l_ankle"]))
        return [
            (crouch, p["t_c"] + p["t_h"]),
            (push, p["t_p"]),
            (tuck, p["t_k"]),
            (land, p["t_l"]),
        ]
    shift = HOME.copy()
    shift[L_ROLL] += p["s_roll_l"]
    shift[R_ROLL] += p["s_roll_r"]
    shift[R_PITCH[0]] -= p["s_rhip"]
    shift[R_PITCH[2]] -= p["s_rankle"]
    up = shift.copy()
    up[L_PITCH[0]] += p["u_hip"]
    up[L_PITCH[1]] += p["u_knee"]
    up[L_PITCH[2]] += p["u_ankle"]
    up[R_PITCH[0]] -= p["u_rhip"]
    up[R_PITCH[1]] -= p["u_rknee"]
    up[R_PITCH[2]] -= p["u_rankle"]
    reach = up.copy()
    reach[L_PITCH[0]] = shift[L_PITCH[0]] + p["r_hip"]
    reach[L_PITCH[1]] = shift[L_PITCH[1]] + p["r_knee"]
    reach[L_PITCH[2]] = shift[L_PITCH[2]] + p["r_ankle"]
    down = reach.copy()
    down[L_PITCH[0]] = HOME[L_PITCH[0]] + p["d_hip"]
    down[L_PITCH[1]] = HOME[L_PITCH[1]] + p["d_knee"]
    down[L_PITCH[2]] = HOME[L_PITCH[2]] + p["d_ankle"]
    return [
        (shift, p["t_s"]),
        (up, p["t_u"]),
        (reach, p["t_r"]),
        (down, p["t_d"]),
    ]


def ticks_of(sched: list[tuple[np.ndarray, float]]) -> np.ndarray:
    """Per-tick commanded targets (N × 14): each keyframe held for its rounded tick count."""
    rows = []
    for pose, secs in sched:
        rows += [pose] * max(1, int(round(secs / TICK_S)))
    return np.clip(np.array(rows), -math.pi, math.pi)


# ── the rig: the gate duck driven directly (robotd's filter and clamp, no network) ───────────


def _support_points(vertices: np.ndarray, n_dirs: int = 240) -> np.ndarray:
    """The mesh's support points in ``n_dirs`` directions (a small hull for the lowest point)."""
    i = np.arange(n_dirs) + 0.5
    phi = np.arccos(1 - 2 * i / n_dirs)
    theta = math.pi * (1 + 5**0.5) * i
    dirs = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)
    idx = np.unique(np.argmax(vertices @ dirs.T, axis=0))
    return vertices[idx]


class Rig:
    """One duck (servo model, DR draw, command delay) standing on v5, reset for each rollout.

    The entry is the one a robotd skill gets: the duck booted seated, rose through sitstand and
    stood 1 s on velstand (the walk slot); the open-loop keyframes then drive the targets through
    robotd's filter and clamp; afterwards velstand takes over again (the hand-back) for
    ``handback_s`` with a zero twist, its filter state and last action carried over as robotd
    does."""

    def __init__(self, servo: str, draw: Any = None, lag_steps: int = 0) -> None:
        import copy

        import mujoco

        from duckmind.skilltrain import dr as DR
        from duckmind.skilltrain.skillduck import SkillDuck

        self.mj = mujoco
        self.servo_name = servo
        draw = draw if draw is not None else DR.Draw(seed=0, preset="none")
        duck = SkillDuck([], servo=servo, draw=draw, keyframe="SIT", x=0.0, y=0.0, yaw=0.0)
        self.duck = duck
        m, d = duck.model, duck.data
        self.m, self.d = m, d
        self.lag_steps = int(lag_steps)
        duck.lag_steps = self.lag_steps  # the hand-back runs with the same command delay
        from collections import deque

        duck._lag = deque(maxlen=self.lag_steps + 1)
        self.soles = []
        for g in duck.foot_geoms:
            mid = int(m.geom_dataid[int(g)])
            va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
            self.soles.append((int(g), _support_points(np.array(m.mesh_vert[va : va + vn]))))
        self.mass = float(np.sum(m.body_mass))
        self.ceiling = float(getattr(duck.servo, "ceiling_nm", 0.96))
        ok, _ = duck.stand(timeout=10.0, hold=1.0)
        self.stood = bool(ok)
        for _ in range(int(SETTLE_S / TICK_S)):
            duck.move(0.0)
            duck.tick()
        self.state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.state = np.zeros(mujoco.mj_stateSize(m, self.state_spec))
        mujoco.mj_getState(m, d, self.state, self.state_spec)
        self.saved_model = tuple(
            a.copy()
            for a in (m.dof_frictionloss, m.dof_damping, m.actuator_gainprm, m.actuator_biasprm)
        )
        sv, port, c = duck.servo, duck.port, duck.port.controller
        self.saved = {
            "servo": {k: copy.deepcopy(getattr(sv, k)) for k in ("prev_motor", "vin_now", "kp_scale", "gain")
                      if hasattr(sv, k)},
            "ctrl": (np.array(c.previous), np.array(c.last_action), c.sit, c.rise_remaining, c.last_net,
                     c.policy.active),
            "port": (port.twist_ema.copy(), port.head_ema.copy(), port.intents.twist, port.intents.twist_at),
            "duck": (list(duck._lag), duck._air.copy(), duck._feet_prev.copy(), duck.safety.falling_for,
                     duck.safety.fallen, np.array(duck._targets), duck.policy, copy.deepcopy(duck.obs_path),
                     duck._ticks),
        }  # fmt: skip
        self.z0 = float(d.subtree_com[duck.trunk][2])
        self.x0 = float(d.qpos[duck.qadr])

    def reset(self) -> None:
        import copy

        from duckmind.skilltrain.skillduck import Trace

        mj, m, d, duck = self.mj, self.m, self.d, self.duck
        mj.mj_setState(m, d, self.state, self.state_spec)
        for dst, src in zip((m.dof_frictionloss, m.dof_damping, m.actuator_gainprm, m.actuator_biasprm),
                            self.saved_model, strict=True):  # fmt: skip
            dst[:] = src
        for k, v in self.saved["servo"].items():
            setattr(duck.servo, k, copy.deepcopy(v))
        c, port = duck.port.controller, duck.port
        prev, last, sit, rise, last_net, active = self.saved["ctrl"]
        c.previous, c.last_action = prev.copy(), last.copy()
        c.sit, c.rise_remaining, c.last_net, c.active = sit, rise, last_net, None
        c.policy.active = active
        tw, he, it, at = self.saved["port"]
        port.twist_ema, port.head_ema = tw.copy(), he.copy()
        port.intents.twist, port.intents.twist_at = it, at
        lag, air, fp, ff, fa, tg, pol, obs, ticks = self.saved["duck"]
        duck._lag.clear()
        duck._lag.extend(np.array(v) for v in lag)
        duck._air, duck._feet_prev = air.copy(), fp.copy()
        duck.safety.falling_for, duck.safety.fallen = ff, fa
        duck._targets, duck.policy, duck.obs_path, duck._ticks = (
            tg.copy(),
            pol,
            copy.deepcopy(obs),
            ticks,
        )
        duck.trace = Trace()
        self.prev = np.array(prev, float)
        mj.mj_forward(m, d)

    def _filtered(self, target: np.ndarray) -> np.ndarray:
        out = np.array(target, float)
        legs = np.ones(14, bool)
        legs[[5, 6, 7, 8]] = False
        out[legs] = LEG_LP * out[legs] + (1 - LEG_LP) * self.prev[legs]
        out[~legs] = HEAD_LP * out[~legs] + (1 - HEAD_LP) * self.prev[~legs]
        self.prev = out.copy()
        return np.clip(out, -math.pi, math.pi)

    def _record(self, rec: dict[str, list], feet: np.ndarray, sums: np.ndarray, each: np.ndarray,
                target: np.ndarray, phase: int) -> None:  # fmt: skip
        d, duck = self.d, self.duck
        rec["feet"].append(feet.astype(float))
        rec["f_housing"].append(float(each.max()) if each.size else 0.0)
        rec["f_trunk"].append(float(sums[2]))
        rec["f_head"].append(float(sums[1]))
        rec["f_other"].append(float(sums[3]))
        rec["f_feet"].append(self._feet_force())
        rec["tau"].append(np.array(d.actuator_force[duck.act]))
        rec["qd"].append(np.array(d.qvel[duck.jv]))
        rec["q"].append(np.array(d.qpos[duck.jq]))
        rec["target"].append(np.array(target))
        rec["com"].append(np.array(d.subtree_com[duck.trunk]))
        rec["trunk"].append(np.array(d.qpos[duck.qadr : duck.qadr + 7]))
        rec["sole_z"].append(self._sole_low())
        rec["sole_x"].append(np.array([float(d.geom_xpos[g][0]) for g, _ in self.soles]))
        rec["vin"].append(float(getattr(duck.servo, "vin_now", 7.4)))
        rec["phase"].append(float(phase))

    def _tick(self, target: np.ndarray, rec: dict[str, list]) -> None:
        mj, m, d, duck = self.mj, self.m, self.d, self.duck
        tgt = self._filtered(target)
        for _ in range(SUBSTEPS):
            duck._lag.append(tgt)
            delayed = duck._lag[0]  # the target sent lag_steps physics steps ago
            duck.servo.before_step(m, d, delayed)
            mj.mj_step(m, d)
            feet, sums, each = duck._contacts(mj)
            self._record(rec, feet, sums, each, delayed, 0)

    def _handback_tick(self, rec: dict[str, list]) -> None:
        """One velstand tick (robotd's walk slot) through the gate duck, recorded per substep."""
        mj, duck = self.mj, self.duck
        orig = duck._contacts
        grabbed: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        def spy(mujoco: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            out = orig(mujoco)
            grabbed.append(out)
            return out

        duck._contacts = spy  # type: ignore[method-assign]
        try:
            duck.move(0.0)
            duck.tick()
        finally:
            del duck._contacts
        for feet, sums, each in grabbed:
            self._record(rec, feet, sums, each, duck._targets, 1)
        del mj

    def _sole_low(self) -> np.ndarray:
        d = self.d
        out = np.zeros(2)
        for i, (g, pts) in enumerate(self.soles):
            rot = np.asarray(d.geom_xmat[g]).reshape(3, 3)
            out[i] = float(np.min(pts @ rot[2]) + d.geom_xpos[g][2])
        return out

    def _feet_force(self) -> float:
        """The floor's vertical force on both soles (N)."""
        mj, m, d = self.mj, self.m, self.d
        feet = set(int(g) for g in self.duck.foot_geoms)
        robot = self.duck.robot_geoms
        fz = 0.0
        f6 = np.zeros(6)
        for i in range(int(d.ncon)):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in feet and g2 not in robot) or (g2 in feet and g1 not in robot):
                mj.mj_contactForce(m, d, i, f6)
                world = np.asarray(c.frame).reshape(3, 3).T @ f6[:3]
                fz += float(-world[2] if g1 in feet else world[2])  # the force on the sole
        return fz

    def rollout(self, targets: np.ndarray, handback_s: float = 2.0) -> dict[str, np.ndarray]:
        self.reset()
        keys = ("feet", "f_housing", "f_trunk", "f_head", "f_other", "f_feet", "tau", "qd", "q",
                "target", "com", "trunk", "sole_z", "sole_x", "vin", "phase")  # fmt: skip
        rec: dict[str, list] = {k: [] for k in keys}
        for row in targets:
            self._tick(row, rec)
        # hand back to velstand: robotd's filter state and the last raw action carry over
        c = self.duck.port.controller
        c.previous = self.prev.copy()
        c.last_action = np.clip((self.prev - HOME) / 0.9, -10, 10).astype(np.float32)
        for _ in range(int(handback_s / TICK_S)):
            self._handback_tick(rec)
        return {k: np.asarray(v, float) for k, v in rec.items()}


# ── scoring ──────────────────────────────────────────────────────────────────────────────────


def _tilt_deg(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz.T
    gz = -(1 - 2 * (x * x + y * y))  # projected gravity z in the trunk frame
    return np.degrees(np.arccos(np.clip(-gz, -1.0, 1.0)))


def _grav_z(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz.T
    return -(1 - 2 * (x * x + y * y))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def analyse(rec: dict[str, np.ndarray], rig: Rig, family: str) -> dict[str, Any]:
    """The rollout's numbers (per physics step, dt = 5 ms)."""
    dt = TICK_S / SUBSTEPS
    tilt = _tilt_deg(rec["trunk"][:, 3:7])
    gz = _grav_z(rec["trunk"][:, 3:7])
    down = gz > FALL_Z
    fall_steps = 0
    fallen = False
    for v in down:
        fall_steps = fall_steps + 1 if v else 0
        if fall_steps * dt >= FALL_DEBOUNCE_S - 1e-9:
            fallen = True
            break
    feet = rec["feet"] > 0.5
    body_floor = (rec["f_trunk"] > 0.5) | (rec["f_head"] > 0.5)
    both_off = (
        ~feet[:, 0] & ~feet[:, 1] & ~body_floor & (rec["f_housing"] < 0.5) & (rec["f_other"] < 0.5)
    )
    flights = [(a, b) for a, b in _runs(both_off) if (b - a) * dt >= 0.01]
    best = {"flight_ms": 0.0, "clearance_mm": 0.0, "com_rise_mm": 0.0, "fwd_mm": 0.0}
    for a, b in flights:
        sz = rec["sole_z"][a:b]
        clear = float(np.max(np.min(sz, axis=1))) * 1000
        sx = rec["sole_x"]
        fwd = float(np.min(sx[b - 1] - sx[a])) * 1000
        rise = float(np.max(rec["com"][a:b, 2]) - rig.z0) * 1000
        cand = {"flight_ms": (b - a) * dt * 1000, "clearance_mm": clear, "com_rise_mm": rise,
                "fwd_mm": fwd, "t_takeoff": a * dt, "t_land": b * dt}  # fmt: skip
        if (
            cand["clearance_mm"] + 0.01 * cand["flight_ms"]
            > best["clearance_mm"] + 0.01 * best["flight_ms"]
        ):
            best = cand
    # the step family: the swing (left) sole's lift while the stance (right) foot is down
    lift_mm = reach_mm = 0.0
    stance_lost = 0
    if family == "step":
        lz = rec["sole_z"][:, 0]
        swing = ~feet[:, 0]
        stance_lost = int(np.sum(swing & ~feet[:, 1]))
        runs = [(a, b) for a, b in _runs(swing) if (b - a) * dt >= 0.05]
        for a, b in runs:
            ok = feet[a:b, 1]
            if ok.mean() < 0.9:
                continue
            lift_mm = max(lift_mm, float(np.max(lz[a:b])) * 1000)
            reach_mm = max(
                reach_mm, float(rec["sole_x"][min(b, len(lz) - 1), 0] - rec["sole_x"][a, 0]) * 1000
            )
    # landing: the peak foot force within 0.15 s after each touchdown (× body weight), housings
    land_peak = 0.0
    for _a, b in flights:
        w = rec["f_feet"][b : b + int(0.15 / dt)]
        if w.size:
            land_peak = max(land_peak, float(w.max()))
    upright = tilt < 45.0
    housing_hits = len(_runs((rec["f_housing"] > HOUSING_N) & upright))
    tau = np.abs(rec["tau"][:, LEGS])
    qd = np.abs(rec["qd"][:, LEGS])
    tilt_end = float(tilt[-1])
    return {
        "fallen": fallen,
        "max_tilt_deg": float(tilt.max()),
        "end_tilt_deg": tilt_end,
        "flights": len(flights),
        **{k: round(float(v), 3) for k, v in best.items()},
        "lift_mm": round(lift_mm, 2),
        "reach_mm": round(reach_mm, 2),
        "stance_lost_steps": stance_lost,
        "landing_peak_N": round(land_peak, 2),
        "landing_peak_bw": round(land_peak / (rig.mass * G), 2),
        "housing_hits": housing_hits,
        "housing_peak_N": round(float(rec["f_housing"].max()), 2),
        "body_floor_steps": int(body_floor.sum()),
        "com_peak_mm": round(float(np.max(rec["com"][:, 2]) - rig.z0) * 1000, 2),
        "leg_tau_max": round(float(tau.max()), 4),
        "leg_tau_p95": round(float(np.percentile(tau.max(axis=1), 95)), 4),
        "leg_qd_max": round(float(qd.max()), 2),
        "leg_tau_over_04_ms": round(float((tau.max(axis=1) > 0.4).sum() * dt * 1000), 1),
        "at_ceiling_ms": round(float((tau.max(axis=1) >= 0.95 * rig.ceiling).sum() * dt * 1000), 1),
    }


def score(a: dict[str, Any], family: str) -> float:
    s = 0.0
    if family in ("hop", "hopf"):
        s = a["clearance_mm"] + 0.05 * a["flight_ms"] + 0.2 * max(0.0, a["com_peak_mm"])
        if family == "hopf":
            s += 0.3 * max(0.0, a["fwd_mm"])
    else:
        s = min(a["lift_mm"], 60.0) + 0.3 * min(max(a["reach_mm"], 0.0), 120.0)
        s -= 0.5 * a["stance_lost_steps"]
    if a["fallen"]:
        s -= 200.0
    s -= 3.0 * max(0.0, a["max_tilt_deg"] - 30.0)
    s -= 2.0 * max(0.0, a["end_tilt_deg"] - 15.0)
    s -= 10.0 * a["housing_hits"]
    s -= 0.2 * a["body_floor_steps"]
    return float(s)


# ── optimisation and evaluation ─────────────────────────────────────────────────────────────


def _to_x(u: np.ndarray, params: list[Param]) -> np.ndarray:
    lo = np.array([p.lo for p in params])
    hi = np.array([p.hi for p in params])
    return lo + (hi - lo) * np.clip(u, 0.0, 1.0)


def optimise(args: argparse.Namespace) -> int:
    import cma

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    params = FAMILIES[args.family]()
    rig = Rig(args.servo, lag_steps=args.lag)
    began = time.perf_counter()
    best: dict[str, Any] = {"score": -1e9}
    history = []
    rng = np.random.default_rng(args.seed)
    for restart in range(args.restarts):
        x0 = rng.uniform(0.25, 0.75, len(params))
        es = cma.CMAEvolutionStrategy(
            x0,
            args.sigma,
            {
                "bounds": [0.0, 1.0],
                "popsize": args.pop,
                "seed": args.seed + restart,
                "verbose": -9,
                "maxiter": args.gens,
            },  # fmt: skip
        )
        while not es.stop():
            us = es.ask()
            fs = []
            for u in us:
                x = _to_x(np.asarray(u), params)
                rec = rig.rollout(ticks_of(schedule(args.family, x)))
                a = analyse(rec, rig, args.family)
                s = score(a, args.family)
                fs.append(-s)
                if s > best["score"]:
                    best = {"score": s, "x": x.tolist(), "analysis": a, "restart": restart}
            es.tell(us, fs)
            history.append({"restart": restart, "gen": es.countiter, "best": round(best["score"], 3),
                            "gen_best": round(-min(fs), 3),
                            "wall_s": round(time.perf_counter() - began, 1)})  # fmt: skip
            if es.countiter % 10 == 0:
                print(f"[{args.family}/{args.servo}] r{restart} gen {es.countiter} "
                      f"gen-best {-min(fs):.2f} best {best['score']:.2f} "
                      f"{time.perf_counter() - began:.0f} s", flush=True)  # fmt: skip
    best.update(
        {
            "family": args.family,
            "servo": args.servo,
            "lag_steps": args.lag,
            "params": [p.name for p in params],
            "schedule": [
                (pose.round(4).tolist(), round(s, 3))
                for pose, s in schedule(args.family, best["x"])
            ],
            "servo_info": rig.duck.servo.info(),
            "mass_kg": rig.mass,
            "wall_s": round(time.perf_counter() - began, 1),
            "rollouts": len(history) * args.pop,
        }
    )
    name = f"best-{args.family}-{args.servo}{'-lag' + str(args.lag) if args.lag else ''}.json"
    (out / name).write_text(json.dumps(best, indent=1))
    (out / name.replace("best-", "history-")).write_text(json.dumps(history))
    print(json.dumps({k: best[k] for k in ("score", "analysis")}, indent=1))
    return 0


def envelope(rec: dict[str, np.ndarray], servo_info: dict[str, Any]) -> dict[str, Any]:
    """Operating points of the leg servos against the model's torque–speed envelope (BAM: the
    DC-motor line kt·(V − kt·ω)/R capped by kt·I_max; the XML servos: their force range)."""
    tau = np.abs(rec["tau"][:, LEGS])
    qd = np.abs(rec["qd"][:, LEGS])
    if servo_info.get("servo") == "bam":
        kt, res, imax = 0.36601349688984386, 2.8113923539223227, 1.75
        vin = rec["vin"][:, None]
        avail = np.minimum(kt * imax, np.maximum(0.0, kt * (vin - kt * qd) / res))
    else:
        avail = np.full_like(tau, float(servo_info.get("ceiling_nm", 0.96)))
    frac = np.where(avail > 1e-6, tau / avail, 1.0)
    busy = tau.max(axis=1) > 0.05
    return {
        "share_at_envelope_95": round(float((frac >= 0.95).any(axis=1)[busy].mean()), 3)
        if busy.any()
        else 0.0,
        "max_speed_rad_s": round(float(qd.max()), 2),
        "max_tau_nm": round(float(tau.max()), 3),
        "no_load_speed_rad_s_at_7v4": round(7.4 / 0.36601349688984386, 2),
    }


def evaluate(args: argparse.Namespace) -> int:
    from duckmind.skilltrain import dr as DR

    best = json.loads(Path(args.best).read_text())
    family = best["family"]
    x = np.array(best["x"])
    targets = ticks_of(schedule(family, x))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for servo in ("mjcf", "bam"):
        for rep in range(args.n):
            seed = zlib.crc32(f"hop-b/{family}/{rep}".encode()) & 0x7FFFFFFF
            draw = DR.draw(seed, "recipe" if servo == "bam" else "physical")
            lag = int(np.random.default_rng(seed + 5).integers(0, 9))  # 0–40 ms
            rig = Rig(servo, draw=draw, lag_steps=lag)
            rec = rig.rollout(targets)
            a = analyse(rec, rig, family)
            a.update(envelope(rec, rig.duck.servo.info()))
            rows.append({"servo": servo, "rep": rep, "seed": seed, "lag_ms": lag * 5, **a})
    summary: dict[str, Any] = {"best": args.best, "family": family, "n": args.n, "by_servo": {}}
    for servo in ("mjcf", "bam"):
        rs = [r for r in rows if r["servo"] == servo]
        n = len(rs)

        def rate(k: int) -> list[float]:
            lo, hi = _wilson(k, n)
            return [k, n, round(100 * lo, 1), round(100 * hi, 1)]

        def pct(key: str) -> list[float]:
            v = np.array([r[key] for r in rs], float)
            return [round(float(np.percentile(v, q)), 2) for q in (10, 50, 90)]

        summary["by_servo"][servo] = {
            "fall": rate(sum(r["fallen"] for r in rs)),
            "airborne_both_feet": rate(sum(r["flight_ms"] > 0 for r in rs)),
            "clear_5mm": rate(sum(r["clearance_mm"] >= 5 for r in rs)),
            "clear_10mm": rate(sum(r["clearance_mm"] >= 10 for r in rs)),
            "clear_20mm": rate(sum(r["clearance_mm"] >= 20 for r in rs)),
            "lift_20mm": rate(sum(r["lift_mm"] >= 20 for r in rs)),
            "lift_30mm": rate(sum(r["lift_mm"] >= 30 for r in rs)),
            "housing_hit": rate(sum(r["housing_hits"] > 0 for r in rs)),
            **{
                f"{k}_p10_50_90": pct(k)
                for k in (
                    "clearance_mm",
                    "flight_ms",
                    "com_rise_mm",
                    "fwd_mm",
                    "lift_mm",
                    "reach_mm",
                    "landing_peak_bw",
                    "leg_tau_max",
                    "leg_tau_p95",
                    "leg_qd_max",
                    "housing_peak_N",
                    "max_tilt_deg",
                )
            },  # fmt: skip
        }
    (out / f"eval-{Path(args.best).stem}.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1)
    )
    print(json.dumps(summary, indent=1))
    return 0


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="hop_trajopt")
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("optimise")
    o.add_argument("--family", choices=tuple(FAMILIES), required=True)
    o.add_argument("--servo", choices=("mjcf", "bam"), required=True)
    o.add_argument("--out", required=True)
    o.add_argument("--pop", type=int, default=16)
    o.add_argument("--gens", type=int, default=60)
    o.add_argument("--restarts", type=int, default=2)
    o.add_argument("--sigma", type=float, default=0.25)
    o.add_argument("--lag", type=int, default=0)
    o.add_argument("--seed", type=int, default=1)
    e = sub.add_parser("evaluate")
    e.add_argument("--best", required=True)
    e.add_argument("--n", type=int, default=30)
    e.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return optimise(args) if args.cmd == "optimise" else evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
