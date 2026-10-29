"""B-list skills, feasibility part (b): what this body allows — kinematics and statics (b:feasibility).

Kinematics and quasi-static torques on the official ``robot_allcollisions.xml`` (microduck_rl cb70b79,
read-only), no controller and no dynamics — the *floor* under what a trained skill must at least
produce, to put next to the XL330 limits (the BAM model's current-limited ceiling kt·I_max, the 0.4 N·m
strain proxy velstand's ``servo_stall`` uses, the plan's p95 ≤ 0.35 N·m gate):

* :func:`facts` — mass, centre of mass, leg segments, joint ranges, the standing stance.
* :func:`bam_envelope` — the BAM XL330 m6 torque–speed line at 6.5 / 7.4 / 8.2 V with the 1.75 A limit.
* :func:`swing_reach` — how high one sole can be lifted (level, in the trunk frame) with the other leg
  standing, reaching 0 / 3 / 6 cm forward: the clearance a 3–5 cm mat step needs; self-contacts checked.
* :func:`single_support` — the stance leg's static torques with the whole-body centre of mass put over
  the stance sole by the stance hip roll (the leg has no ankle roll: the trunk leans over the hip), the
  swing sole lifted 0–7 cm: the quasi-static bound for a mat step (a dynamic step needs less time on one
  leg, not less torque at the push).
* :func:`step_push` — the push onto the mat: the leg whose sole is ``h`` higher carries the duck and lifts
  it (the knee/hip torques at the start of the push, h = 3 / 4 / 5 cm).
* :func:`bow` — front-down postures: trunk pitched 0–50° nose-down and lowered, both soles flat and in
  place, the trunk slid back so the centre of mass stays over the feet; the joint angles, the trunk's and
  the beak's clearance to the floor (where the chest meets the floor = "lying low"), the symmetric
  double-support static torques.

Static equilibrium (qvel = qacc = 0): ``qfrc_bias = τ + Jᵀ w`` with ``w`` the floor wrench(es); the six
floating-base rows fix ``w`` (single support: the wrench at the stance sole; double support, symmetric
pose: per sole (Fx, Fz, My) mirrored), the joint rows give ``τ``. The centre of pressure is checked
against the sole (5.4 × 1.4 cm).

    .venv-official-cpu/bin/python -m duckmind.skilltrain.bsk_body --out DIR
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from duckmind import paths

JOINTS = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)  # fmt: skip
HOME = dict(zip(JOINTS, (0.0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0.0, 0.0,
                         0.0, 0.0873, 0.4579, 0.0049, -0.4530), strict=True))  # fmt: skip
L_CHAIN = ("left_hip_pitch", "left_knee", "left_ankle")
R_CHAIN = ("right_hip_pitch", "right_knee", "right_ankle")
SOLE_HALF = (0.027, 0.007)  # half length (x) and half width (y) of a sole (collision mesh bbox)
STRAIN_NM, GATE_P95_NM = 0.4, 0.35
G = 9.81


class Body:
    def __init__(self, folder: str | None = None) -> None:
        import mujoco

        self.mj = mujoco
        root = Path(folder) if folder else paths.microduck_mjcf_dir()
        self.m = mujoco.MjModel.from_xml_path(str(root / "robot_allcollisions.xml"))
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.qadr = {n: int(m.jnt_qposadr[m.joint(n).id]) for n in JOINTS}
        self.dadr = {n: int(m.jnt_dofadr[m.joint(n).id]) for n in JOINTS}
        self.range = {n: tuple(float(v) for v in m.jnt_range[m.joint(n).id]) for n in JOINTS}
        self.trunk = m.body("trunk_base").id
        self.site = {s: m.site(s).id for s in ("left_foot", "right_foot", "mouth_tip")}
        self.foot_geom = {s: m.geom(f"{s}_foot_collision").id for s in ("left", "right")}
        self.mass = float(m.body_subtreemass[self.trunk])
        self.robot_geoms = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]

    # ── state ──
    def pose(self, q: dict[str, float], pos=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)) -> None:
        d = self.d
        d.qpos[:] = 0.0
        d.qpos[0:3] = pos
        d.qpos[3:7] = quat_rpy(*rpy)
        for n, v in q.items():
            d.qpos[self.qadr[n]] = v
        d.qvel[:] = 0.0
        self.mj.mj_forward(self.m, d)

    def site_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        i = self.site[name]
        return self.d.site_xpos[i].copy(), self.d.site_xmat[i].reshape(3, 3).copy()

    def com(self) -> np.ndarray:
        return self.d.subtree_com[self.trunk].copy()

    def sole_bottom(self, side: str) -> np.ndarray:
        """The sole's centre: the mean of the foot collision mesh's vertices within 0.5 mm of its
        lowest point (a flat sole has many equally low vertices; one of them is not its centre)."""
        v = self._verts(self.foot_geom[side])
        low = v[:, 2].min()
        return v[v[:, 2] <= low + 5e-4].mean(axis=0)

    def sole_min_z(self, side: str) -> float:
        return float(self._verts(self.foot_geom[side])[:, 2].min())

    def _verts(self, g: int) -> np.ndarray:
        m, d = self.m, self.d
        mid = int(m.geom_dataid[g])
        a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        return (d.geom_xmat[g].reshape(3, 3) @ m.mesh_vert[a : a + n].T).T + d.geom_xpos[g]

    def _lowest(self, geoms: list[int]) -> np.ndarray:
        m, d = self.m, self.d
        best = None
        for g in geoms:
            if int(m.geom_type[g]) != int(self.mj.mjtGeom.mjGEOM_MESH):
                p = d.geom_xpos[g].copy()
            else:
                mid = int(m.geom_dataid[g])
                a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
                v = m.mesh_vert[a : a + n]
                w = (d.geom_xmat[g].reshape(3, 3) @ v.T).T + d.geom_xpos[g]
                p = w[int(np.argmin(w[:, 2]))]
            if best is None or p[2] < best[2]:
                best = p
        return np.asarray(best)

    def lowest_nonfoot(self, bodies: tuple[str, ...] | None = None) -> tuple[float, str]:
        """The lowest point of the robot's collision geoms other than the feet (world z) and its body."""
        m = self.m
        feet = set(self.foot_geom.values())
        ankles = {m.body("ankle_left").id, m.body("ankle_right").id}
        best, where = math.inf, ""
        for g in self.robot_geoms:
            if g in feet or (bodies is None and int(m.geom_bodyid[g]) in ankles):
                continue
            b = self.mj.mj_id2name(m, self.mj.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "?"
            if bodies is not None and b not in bodies:
                continue
            z = float(self._lowest([g])[2])
            if z < best:
                best, where = z, b
        return best, where

    def self_contacts(self) -> int:
        """Robot-robot contacts in the current pose (the model's own collision filtering)."""
        d = self.d
        feet_pairs = 0
        for i in range(int(d.ncon)):
            c = d.contact[i]
            if c.dist < -1e-4:
                feet_pairs += 1
        return feet_pairs

    def in_range(self, q: dict[str, float], margin: float = 0.02) -> bool:
        return all(
            self.range[n][0] + margin <= v <= self.range[n][1] - margin for n, v in q.items()
        )

    # ── statics ──
    def static_single(self, stance: str) -> dict[str, Any]:
        """τ for static equilibrium with the whole floor wrench at the stance sole's bottom centre."""
        m, d, mj = self.m, self.d, self.mj
        nv = m.nv
        p = self.sole_bottom(stance)
        jacp, jacr = np.zeros((3, nv)), np.zeros((3, nv))
        body = int(m.geom_bodyid[self.foot_geom[stance]])
        mj.mj_jac(m, d, jacp, jacr, p, body)
        J = np.vstack([jacp, jacr])  # 6 × nv
        bias = d.qfrc_bias.copy()
        w = np.linalg.solve(J[:, :6].T, bias[:6])
        tau = bias - J.T @ w
        cop = self._cop(w, p, stance)
        return {"tau": self._named(tau), "wrench": w, "cop_mm": cop, "fz_N": float(w[2])}

    def static_double_sym(self) -> dict[str, Any]:
        """Symmetric double support: per sole (Fx, Fz, My), mirrored; least squares on the 6 base rows."""
        m, d, mj = self.m, self.d, self.mj
        nv = m.nv
        Js, ps = [], []
        for side in ("left", "right"):
            p = self.sole_bottom(side)
            jacp, jacr = np.zeros((3, nv)), np.zeros((3, nv))
            mj.mj_jac(m, d, jacp, jacr, p, int(m.geom_bodyid[self.foot_geom[side]]))
            Js.append(np.vstack([jacp, jacr]))
            ps.append(p)
        # unknowns (Fx, Fz, My) applied to both soles
        sel = np.zeros((6, 3))
        sel[0, 0], sel[2, 1], sel[4, 2] = 1.0, 1.0, 1.0
        A = (Js[0].T @ sel + Js[1].T @ sel)[:6]
        bias = d.qfrc_bias.copy()
        x, *_ = np.linalg.lstsq(A, bias[:6], rcond=None)
        w = sel @ x
        tau = bias - (Js[0].T + Js[1].T) @ w
        resid = float(np.abs(A @ x - bias[:6]).max())
        return {
            "tau": self._named(tau),
            "per_sole": w,
            "resid": resid,
            "cop_mm": self._cop(w, ps[0], "left"),
        }

    def _cop(self, w: np.ndarray, p: np.ndarray, side: str) -> list[float]:
        fz = w[2]
        if abs(fz) < 1e-9:
            return [math.nan, math.nan]
        # moment about the sole point; CoP offset from p (world x/y): x = −My/Fz, y = Mx/Fz
        return [round(1000 * (-w[4] / fz), 2), round(1000 * (w[3] / fz), 2)]

    def _named(self, tau: np.ndarray) -> dict[str, float]:
        return {n: round(float(tau[self.dadr[n]]), 4) for n in JOINTS}


def quat_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])  # fmt: skip


def _rotvec(R: np.ndarray) -> np.ndarray:
    E = R
    return 0.5 * np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])


# ── part (b) computations ─────────────────────────────────────────────────────────────────────


def facts(b: Body) -> dict[str, Any]:
    b.pose(HOME)
    lf, _ = b.site_pose("left_foot")
    sole = b.sole_bottom("left")
    com = b.com()
    anchors = {}
    for n in ("left_hip_pitch", "left_knee", "left_ankle"):
        anchors[n] = b.d.xanchor[b.m.joint(n).id].copy()
    seg = {
        "hip_pitch→knee_mm": round(
            1000 * float(np.linalg.norm(anchors["left_knee"] - anchors["left_hip_pitch"])), 1
        ),
        "knee→ankle_mm": round(
            1000 * float(np.linalg.norm(anchors["left_ankle"] - anchors["left_knee"])), 1
        ),
        "ankle→sole_mm": round(1000 * float(anchors["left_ankle"][2] - sole[2]), 1),
    }
    head_mass = sum(
        float(b.m.body_mass[b.m.body(n).id])
        for n in ("neck", "neck_pitch", "yaw_roll_motion", "jaw_soft")
    )
    return {
        "mass_kg": round(b.mass, 4),
        "head_mass_kg": round(head_mass, 4),
        "head_share": round(head_mass / b.mass, 3),
        "standing_trunk_above_sole_mm": round(1000 * (0.0 - sole[2]), 1),
        "com_above_sole_mm": round(1000 * (com[2] - sole[2]), 1),
        "com_ahead_of_sole_centre_mm": round(1000 * (com[0] - sole[0]), 1),
        "stance_half_width_mm": round(1000 * abs(float(sole[1])), 1),
        "sole_mm": [54, 14],
        "segments": seg,
        "ranges_rad": {n: [round(v, 3) for v in b.range[n]] for n in JOINTS},
    }


def bam_envelope() -> dict[str, Any]:
    try:
        from bam.model import load_model
    except ImportError:
        return {"error": "bam not installed in this interpreter"}
    bm = load_model(motor_name="xl330", model="m6")
    kt, R = float(bm.kt.value), float(bm.R.value)
    imax = float(getattr(bm.actuator, "max_current", 1.75) or 1.75)
    out: dict[str, Any] = {"kt_Nm_per_A": round(kt, 4), "R_ohm": round(R, 3), "I_max_A": imax,
                           "ceiling_Nm": round(kt * imax, 3)}  # fmt: skip
    for v in (6.5, 7.4, 8.2):
        stall = kt * v / R
        w0 = v / kt
        # speed at which the voltage-limited torque drops below the current limit / below 0.4 N·m
        w_at = lambda tau: (v - tau * R / kt) / kt  # noqa: E731
        out[f"{v}V"] = {
            "stall_Nm_voltage_limited": round(stall, 3),
            "no_load_rad_s": round(w0, 2),
            "speed_at_0.4Nm_rad_s": round(w_at(0.4), 2),
            "speed_at_ceiling_rad_s": round(w_at(min(stall, kt * imax)), 2),
            "speed_at_0.2Nm_rad_s": round(w_at(0.2), 2),
        }
    return out


def _solve_leg(b: Body, base: dict[str, float], chain: tuple[str, str, str], site: str,
               target: np.ndarray, R0: np.ndarray, pos=(0, 0, 0), rpy=(0, 0, 0)) -> tuple[dict[str, float], float]:  # fmt: skip
    """(hip_pitch, knee, ankle) of one leg so its foot site reaches ``target`` (world x, z) with the
    sole's pitch of ``R0`` (least squares, mm/deg residual)."""
    from scipy.optimize import least_squares

    def res(x: np.ndarray) -> np.ndarray:
        q = dict(base)
        q.update(zip(chain, x, strict=True))
        b.pose(q, pos, rpy)
        f, R = b.site_pose(site)
        rv = _rotvec(R0.T @ R)
        return np.concatenate([(f - target)[[0, 2]] * 1000.0, [rv[1] * 57.3]])

    x0 = np.array([base[n] for n in chain])
    s = least_squares(res, x0, xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=400)
    q = dict(base)
    q.update(zip(chain, s.x, strict=True))
    return q, float(np.abs(s.fun).max())


def swing_reach(b: Body) -> list[dict[str, Any]]:
    """The highest the right sole's *lowest point* can be lifted above the floor (trunk level at the
    standing height, left leg at HOME or straightened to the leg's reach, +8 mm), with the right foot
    site ``dx`` ahead of where it stands and the sole pitched at most ±25° — the clearance a foot needs
    to pass over a mat edge of that height (joint limits with a 0.02 rad margin; kinematics only)."""
    from scipy.optimize import minimize

    from duckmind.skilltrain.rl_contract import home_dip_pose

    rows = []
    for rise in (0.0, 0.008):
        base = dict(zip(JOINTS, home_dip_pose(-rise), strict=True))
        b.pose(base)
        floor = b.sole_min_z("left")
        f0, R0 = b.site_pose("right_foot")
        lo = [b.range[n][0] + 0.02 for n in R_CHAIN]
        hi = [b.range[n][1] - 0.02 for n in R_CHAIN]
        x = np.array([base[n] for n in R_CHAIN])
        for dx in (0.0, 0.01, 0.02, 0.03, 0.04, 0.05):

            def pose_of(v: np.ndarray) -> dict[str, float]:
                q = dict(base)
                q.update(zip(R_CHAIN, v, strict=True))
                return q

            def lift(v: np.ndarray) -> float:
                b.pose(pose_of(v))
                return b.sole_min_z("right") - floor

            def reach(v: np.ndarray) -> float:
                b.pose(pose_of(v))
                return (b.site_pose("right_foot")[0][0] - f0[0]) - dx

            def pitch_ok(v: np.ndarray) -> float:
                b.pose(pose_of(v))
                return math.radians(25.0) - abs(_rotvec(R0.T @ b.site_pose("right_foot")[1])[1])

            best = None
            for start in (
                x,
                np.array([0.3, -1.2, -1.0]) * np.sign(np.array(hi)),
                np.array(hi) * 0.5,
            ):
                r = minimize(lambda v: -lift(v), np.clip(start, lo, hi), method="SLSQP",
                             bounds=list(zip(lo, hi, strict=True)),
                             constraints=[{"type": "eq", "fun": reach}, {"type": "ineq", "fun": pitch_ok}],
                             options={"maxiter": 300, "ftol": 1e-9})  # fmt: skip
                if r.success and abs(reach(r.x)) < 1e-3 and pitch_ok(r.x) > -1e-3:
                    if best is None or lift(r.x) > lift(best):
                        best = r.x
            if best is None:
                rows.append(
                    {
                        "stance_rise_mm": round(1000 * rise),
                        "dx_mm": round(1000 * dx),
                        "lift_mm": None,
                    }
                )
                continue
            b.pose(pose_of(best))
            rows.append({
                "stance_rise_mm": round(1000 * rise), "dx_mm": round(1000 * dx),
                "lift_mm": round(1000 * lift(best), 1),
                "sole_pitch_deg": round(math.degrees(_rotvec(R0.T @ b.site_pose("right_foot")[1])[1]), 1),
                "self_contacts": b.self_contacts(),
                "q": {n: round(float(v), 3) for n, v in zip(R_CHAIN, best, strict=True)},
            })  # fmt: skip
            x = best
    return rows


def single_support(b: Body) -> list[dict[str, Any]]:
    """Static balance on one sole: the stance (left) sole flat on the floor; the whole-body centre of mass
    must be over it (±7 mm laterally, the sole's half width). The leg has no ankle roll, so the only way to
    move the centre of mass sideways with the sole flat is to lean the trunk over the stance hip (hip roll
    within its ±0.384 rad range). Reports the lateral miss at the best hip roll, for the right sole lifted
    0 / 3 / 5 cm (level), and the stance torques there (the floor wrench's centre of pressure then falls
    outside the sole = no static solution)."""
    rows = []
    b.pose(HOME)
    f0r, R0r = b.site_pose("right_foot")
    for L in (0.0, 0.03, 0.05):
        swing = dict(HOME)
        if L > 0:
            swing, _ = _solve_leg(
                b, HOME, R_CHAIN, "right_foot", f0r + np.array([0.0, 0.0, L]), R0r
            )
        best = None
        for hr in np.linspace(b.range["left_hip_roll"][0], b.range["left_hip_roll"][1], 41):
            q = dict(swing)
            q["left_hip_roll"] = float(hr)
            # the trunk rotates so the left sole stays flat: find the trunk roll that levels the sole
            rolls = np.linspace(-0.8, 0.8, 161)
            errs = []
            for rr in rolls:
                b.pose(q, (0, 0, 0), (float(rr), 0, 0))
                errs.append(abs(_rotvec(b.site_pose("left_foot")[1])[0]))
            rr = float(rolls[int(np.argmin(errs))])
            b.pose(q, (0, 0, 0), (rr, 0, 0))
            miss = float(b.com()[1] - b.sole_bottom("left")[1])
            if best is None or abs(miss) < abs(best[0]):
                best = (miss, float(hr), rr)
        miss, hr, rr = best
        q = dict(swing)
        q["left_hip_roll"] = hr
        b.pose(q, (0, 0, 0), (rr, 0, 0))
        st = b.static_single("left")
        rows.append({
            "swing_lift_mm": round(1000 * L),
            "best_hip_roll_rad": round(hr, 3), "trunk_roll_deg": round(math.degrees(rr), 1),
            "com_lateral_miss_mm": round(1000 * miss, 1),
            "static_possible": abs(miss) <= 1000 * SOLE_HALF[1] / 1000,
            "cop_mm": st["cop_mm"],
        })  # fmt: skip
    return rows


def step_push(b: Body) -> list[dict[str, Any]]:
    """The push onto the mat: the right sole on the mat (``h`` up, 3 cm ahead, level), the left sole on
    the floor, trunk level at the standing height. Sagittal static torques of the mat leg (hip pitch, knee,
    ankle) when it carries the whole weight (the dynamic moment of the push; no static one-leg balance
    exists — :func:`single_support` — so the lateral moment is the step's momentum, not the sole) and in
    an even 50/50 double support."""
    rows = []
    b.pose(HOME)
    f0r, R0r = b.site_pose("right_foot")
    for h in (0.03, 0.04, 0.05):
        mat_leg, err = _solve_leg(
            b, HOME, R_CHAIN, "right_foot", f0r + np.array([0.03, 0.0, h]), R0r
        )
        ok = err < 1.0 and b.in_range({n: mat_leg[n] for n in R_CHAIN})
        if not ok:
            rows.append({"h_mm": round(1000 * h), "reachable": False, "residual": round(err, 2)})
            continue
        b.pose(mat_leg)
        one = b.static_single("right")["tau"]
        both = b.static_double_sym()  # symmetric split is only an estimate for unequal soles
        sag = ("right_hip_pitch", "right_knee", "right_ankle")
        rows.append({
            "h_mm": round(1000 * h), "reachable": True,
            "mat_leg_q": {n: round(mat_leg[n], 3) for n in sag},
            "knee_bend_deg": round(math.degrees(abs(mat_leg["right_knee"])), 1),
            "all_weight_tau_Nm": {n: round(abs(one[n]), 3) for n in sag},
            "all_weight_tau_max_Nm": round(max(abs(one[n]) for n in sag), 3),
            "half_weight_tau_max_Nm": round(max(abs(both["tau"][n]) for n in sag), 3),
        })  # fmt: skip
    return rows


def bow(b: Body) -> list[dict[str, Any]]:
    """Trunk pitched θ nose-down and its origin lowered by ``drop`` (world), both soles flat at the
    standing stance, the trunk slid along x so the centre of mass is over the soles' centre; head at
    HOME (``beak_home``) and head fully down (neck −0.9, head +1.2: ``beak_down``)."""
    from scipy.optimize import least_squares

    rows = []
    b.pose(HOME)
    sole_l = b.sole_bottom("left")
    fl0, Rl0 = b.site_pose("left_foot")
    z_stand = -float(sole_l[2])  # trunk origin above the floor, standing
    sole_centre_x = float(sole_l[0])
    for theta in (0, 10, 20, 30, 40, 50):
        for drop in (0.0, 0.010, 0.020, 0.030, 0.040, 0.050):
            th = math.radians(theta)

            def place(x: np.ndarray) -> tuple[dict[str, float], tuple[float, float, float]]:
                q = dict(HOME)
                hp, kn, an, tx = x
                q.update({"left_hip_pitch": hp, "left_knee": kn, "left_ankle": an,
                          "right_hip_pitch": -hp, "right_knee": -kn, "right_ankle": -an})  # fmt: skip
                return q, (tx, 0.0, z_stand - drop)

            def res(x: np.ndarray) -> np.ndarray:
                q, pos = place(x)
                b.pose(q, pos, (0.0, th, 0.0))
                f, R = b.site_pose("left_foot")
                sole = b.sole_bottom("left")
                com = b.com()
                rv = _rotvec(Rl0.T @ R)
                return np.array([
                    (f[0] - fl0[0]) * 1000.0,  # the sole stays where it stood (x)
                    sole[2] * 1000.0,  # on the floor
                    rv[1] * 57.3,  # flat
                    (com[0] - sole_centre_x) * 1000.0,  # centre of mass over the soles' centre
                ])  # fmt: skip

            x0 = np.array([HOME["left_hip_pitch"], HOME["left_knee"], HOME["left_ankle"], 0.0])
            s = least_squares(res, x0, max_nfev=600)
            q, pos = place(s.x)
            b.pose(q, pos, (0.0, th, 0.0))
            ok = float(np.abs(s.fun).max()) < 1.0
            legs_q = {n: q[n] for n in L_CHAIN}
            in_range = b.in_range(legs_q)
            low_trunk, _ = b.lowest_nonfoot(("trunk_base",))
            low_any, where = b.lowest_nonfoot()
            beak_home = float(b.d.site_xpos[b.site["mouth_tip"]][2])
            st = b.static_double_sym()
            tau = st["tau"]
            legs = {n: tau[n] for n in ("left_hip_pitch", "left_knee", "left_ankle")}
            qd = dict(q)
            qd["neck_pitch"] = max(b.range["neck_pitch"][0] + 0.05, HOME["neck_pitch"] - 0.9)
            qd["head_pitch"] = min(b.range["head_pitch"][1] - 0.05, HOME["head_pitch"] + 1.2)
            b.pose(qd, pos, (0.0, th, 0.0))
            beak_down = float(b.d.site_xpos[b.site["mouth_tip"]][2])
            low_down, where_down = b.lowest_nonfoot()
            st_d = b.static_double_sym()
            rows.append({
                "pitch_deg": theta, "drop_mm": round(1000 * drop),
                "solved": ok, "legs_in_range": in_range,
                "legs_q": {n: round(v, 3) for n, v in legs_q.items()},
                "trunk_slide_back_mm": round(-1000 * float(pos[0]), 1),
                "trunk_lowest_mm": round(1000 * low_trunk, 1),
                "robot_lowest_mm": round(1000 * low_any, 1), "robot_lowest_body": where,
                "beak_head_home_mm": round(1000 * beak_home, 1),
                "beak_head_down_mm": round(1000 * beak_down, 1), "lowest_head_down_mm": round(1000 * low_down, 1),
                "lowest_head_down_body": where_down,
                "leg_tau_Nm": legs, "leg_tau_max_Nm": round(max(abs(v) for v in legs.values()), 3),
                "leg_tau_max_head_down_Nm": round(max(abs(st_d["tau"][n]) for n in L_CHAIN), 3),
                "neck_tau_head_down_Nm": round(abs(st_d["tau"]["neck_pitch"]), 3),
                "cop_mm": st["cop_mm"], "resid": round(st["resid"], 6),
            })  # fmt: skip
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bsk_body")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    b = Body()
    out = {
        "facts": facts(b),
        "bam": bam_envelope(),
        "swing_reach": swing_reach(b),
        "single_support": single_support(b),
        "step_push": step_push(b),
        "bow": bow(b),
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "body.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False, default=float)
    )
    print(json.dumps({k: out[k] for k in ("facts", "bam", "swing_reach")}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
