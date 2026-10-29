"""Hop feasibility, part (b) bound: the most a two-foot hop could give this body, by physics alone.

hop:feasibility (2026-09-28). An *upper bound* with perfect balance and perfect coordination —
what no controller, trained or not, can beat on this body with these servos:

1. **Legs as one vertical actuator.** The left leg's (hip_pitch, knee, ankle) that move the trunk
   straight up/down with the sole flat and in place and the trunk level (the Stage-1 IK tables'
   method: least squares on the official ``robot_allcollisions.xml``, extended here from a 60 mm
   crouch to the straight knee at +10 mm; the right leg mirrors). ``J_i(z) = dq_i/dz`` (rad per m of
   trunk height): by virtual work each leg's vertical force is ``Σ τ_i J_i``.
2. **The servos' torque–speed curves.** ``bam``: the BAM XL330 m6 model training uses — DC motor
   ``kt (V·duty − kt ω)/R`` with duty ≤ 1, the 1.75 A firmware limit (``kt·I = 0.641`` N·m), the
   gearbox friction that grows with the motor's own torque (``load_friction_motor`` 0.267 of it),
   base and viscous friction; battery 6.5 / 7.4 / 8.2 V. ``mjcf``: the MJCF's XML servos (force range
   ±0.96 N·m, joint damping 0.053 N·m·s/rad, friction loss 0.0048). Every joint at its limit at once.
3. **The push.** ``m z̈ = 2 Σ τ_i(J_i ż) J_i − m g`` from rest in a crouch to the straight knee; the
   feet leave the floor at full extension or earlier when the legs cannot keep up (force ≤ 0).
   Then a ballistic flight; the legs may **tuck** at full speed (the feet rise relative to the
   trunk) and must be back under the body to land.

Outputs: take-off speed, centre-of-mass rise, flight time, the best two-foot clearance with a tuck,
the push's joint speeds and torques against the curves, and what a hop *over* a 4 cm strip would
need (every sole airborne over 9.4 cm of travel).

Run in the official venv (scipy + MuJoCo)::

    PYTHONPATH=src .venv-official-cpu/bin/python -m duckmind.skilltrain.hop_ceiling --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

G = 9.81
MASS = 0.7372  # kg, the official model's total
FEET_MASS = 2 * 0.030  # the ankle/foot bodies stay on the floor during the push
LEG_MASS = 2 * (0.0062 + 0.0482 + 0.0216 + 0.030)
# BAM XL330 m6 (bam git 62bd8ce, load_model("xl330", "m6"))
KT, R_OHM, I_MAX = 0.36601349688984386, 2.8113923539223227, 1.75
F_BASE, F_VISC, F_LOAD_MOTOR = 0.004771183165566, 0.005359668274599504, 0.2667860954283698
F_STRIBECK, DTHETA_STRIBECK, ALPHA_STRIBECK = (
    0.004676345799486616,
    2.890372094130307,
    8.683259907618984,
)
# the MJCF's XML servo (joints_properties.xml chosen_actuator)
XML_FORCE, XML_DAMPING, XML_FRICTION = 0.96, 0.053, 0.0048
STRIP_OVER_M = 0.04 + 0.054  # a 4 cm strip + one 5.4 cm sole


def _official_robot() -> Path:
    folder = os.environ.get("DUCKMIND_MICRODUCK_MJCF")
    if not folder:
        raise SystemExit("set DUCKMIND_MICRODUCK_MJCF to the official microduck folder")
    return Path(folder) / "robot_allcollisions.xml"


def ik_table(dips_m: np.ndarray) -> np.ndarray:
    """Left (hip_pitch, knee, ankle) at each trunk dip below HOME (+ = down), sole flat and in
    place, trunk level (least squares on the official model; kinematics only)."""
    import mujoco
    from scipy.optimize import least_squares

    m = mujoco.MjModel.from_xml_path(str(_official_robot()))
    d = mujoco.MjData(m)
    home = {"left_hip_yaw": 0.0, "left_hip_roll": -0.0873, "left_hip_pitch": -0.4579,
            "left_knee": -0.0049, "left_ankle": 0.4530}  # fmt: skip
    names = ["left_hip_pitch", "left_knee", "left_ankle"]

    def set_pose(q: dict[str, float]) -> None:
        d.qpos[:] = 0
        d.qpos[3] = 1.0
        for n, v in q.items():
            d.qpos[m.jnt_qposadr[m.joint(n).id]] = v
        mujoco.mj_kinematics(m, d)

    def site() -> tuple[np.ndarray, np.ndarray]:
        i = m.site("left_foot").id
        return d.site_xpos[i].copy(), d.site_xmat[i].reshape(3, 3).copy()

    set_pose(home)
    f0, r0 = site()
    out = []
    x = np.array([home[n] for n in names])
    for dip in dips_m:
        tgt = f0 + np.array([0.0, 0.0, dip])

        def res(v: np.ndarray, tgt: np.ndarray = tgt) -> np.ndarray:
            q = dict(home)
            q.update(dict(zip(names, v, strict=True)))
            set_pose(q)
            f, r = site()
            e = r0.T @ r
            rv = 0.5 * np.array([e[2, 1] - e[1, 2], e[0, 2] - e[2, 0], e[1, 0] - e[0, 1]])
            return np.concatenate([(f - tgt)[[0, 2]] * 1000.0, rv[[1]] * 57.3])

        sol = least_squares(res, x, xtol=1e-14, ftol=1e-14, gtol=1e-14)
        x = sol.x
        out.append([dip, *sol.x, float(np.max(np.abs(sol.fun)))])
    return np.array(out)


def servo_torque(servo: str, omega: np.ndarray, vin: float = 7.4) -> np.ndarray:
    """The most output torque a joint can give while moving at ``omega`` (≥ 0) in the driving
    direction (N·m), full duty towards a far target. Negative above the no-load speed: the motor's
    back-EMF brakes (the kinematic chain makes every joint move with the trunk)."""
    w = np.abs(omega)
    if servo == "mjcf":
        return XML_FORCE - XML_DAMPING * w - XML_FRICTION
    motor = KT * np.minimum(I_MAX, (vin - KT * w) / R_OHM)
    strib = np.exp(-np.power(w / DTHETA_STRIBECK, ALPHA_STRIBECK))
    friction = F_BASE + strib * F_STRIBECK + F_LOAD_MOTOR * np.abs(motor) + F_VISC * w
    return motor - friction


def push(
    servo: str, table: np.ndarray, crouch_m: float, vin: float = 7.4, dt: float = 1e-4
) -> dict[str, Any]:
    """Integrate the ideal vertical push from rest at ``crouch_m`` below HOME."""
    dips = table[:, 0]
    q = table[:, 1:4]
    # z = trunk height relative to HOME (up +) = -dip; J = dq/dz along the table
    order = np.argsort(-dips)  # z = −dip ascending
    z_tab = -dips[order]
    q_tab = q[order]
    jac = np.gradient(q_tab, z_tab, axis=0)  # rad per m (left leg)
    z_top = float(z_tab[-1])
    z = -crouch_m
    v = 0.0
    t = 0.0
    m_move = MASS - FEET_MASS
    peak_w = np.zeros(3)
    peak_tau = np.zeros(3)
    took_off = "full extension"
    while z < z_top:
        jz = np.array([np.interp(z, z_tab, jac[:, i]) for i in range(3)])
        w = np.abs(jz) * v
        tau = servo_torque(servo, w, vin)
        force = 2.0 * float(np.sum(tau * np.abs(jz)))  # both legs, N
        acc = force / m_move - G
        if v > 0.02 and force <= 0.05:
            took_off = "legs cannot keep up (force → 0)"
            break
        v = max(0.0, v + acc * dt)
        z += v * dt
        t += dt
        peak_w = np.maximum(peak_w, w)
        peak_tau = np.maximum(peak_tau, tau)
        if t > 2.0:
            took_off = "never"
            break
    # the whole body's CoM moves ≈ (m − feet)/m of the trunk's velocity
    v_com = v * (MASS - FEET_MASS) / MASS
    rise = v_com**2 / (2 * G)
    flight = 2 * v_com / G
    # tuck: the legs retract at the no-load-limited speed from full extension (knee first)
    jz_top = np.array([np.interp(z_top - 0.002, z_tab, jac[:, i]) for i in range(3)])
    w_nl = (vin / KT) if servo == "bam" else (XML_FORCE - XML_FRICTION) / XML_DAMPING
    rel_speed = w_nl / float(np.max(np.abs(jz_top)))  # feet up relative to the trunk (m/s)
    best = 0.0
    for tf in np.linspace(0, flight, 200):
        up = v_com * tf - 0.5 * G * tf * tf
        # tuck up to half the flight, then back down to land (symmetric): Δ = rel·min(t, T − t)
        tuck = rel_speed * min(tf, flight - tf) * (1 - LEG_MASS / MASS)
        best = max(best, up + tuck)
    return {
        "servo": servo,
        "vin": vin,
        "crouch_mm": round(crouch_m * 1000, 1),
        "stroke_mm": round((z_top + crouch_m) * 1000, 1),
        "push_ms": round(t * 1000, 1),
        "takeoff": took_off,
        "takeoff_trunk_mps": round(v, 3),
        "com_rise_mm": round(rise * 1000, 2),
        "flight_ms": round(flight * 1000, 1),
        "clear_no_tuck_mm": round(rise * 1000, 2),
        "clear_with_tuck_mm": round(best * 1000, 2),
        "peak_joint_speed_rad_s": [round(float(x), 2) for x in peak_w],
        "peak_joint_torque_nm": [round(float(x), 3) for x in peak_tau],
        "no_load_speed_rad_s": round(float(w_nl), 2),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="hop_ceiling")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    dips = np.round(np.arange(0.060, -0.0101, -0.001), 4)
    table = ik_table(dips)
    rows = []
    for servo, vins in (("mjcf", (7.4,)), ("bam", (6.5, 7.4, 8.2))):
        for vin in vins:
            for crouch in (0.010, 0.020, 0.030, 0.040, 0.050, 0.060):
                rows.append(push(servo, table, crouch, vin))
    # what a hop over a 4 cm strip needs, at the walk's own speeds
    need = []
    for u in (0.10, 0.15, 0.20, 0.25, 0.30):
        t_air = STRIP_OVER_M / u
        v0 = G * t_air / 2
        need.append({"forward_mps": u, "flight_ms": round(t_air * 1000), "takeoff_mps": round(v0, 2),
                     "com_rise_mm": round(v0 * v0 / (2 * G) * 1000)})  # fmt: skip
    best = {s: max((r for r in rows if r["servo"] == s), key=lambda r: r["clear_with_tuck_mm"])
            for s in ("mjcf", "bam")}  # fmt: skip
    out = {
        "what": "ideal two-foot vertical hop bound (perfect balance and coordination)",
        "ik_table": [[round(float(v), 5) for v in row] for row in table],
        "rows": rows,
        "best": best,
        "hop_over_4cm_strip_needs": need,
    }
    path = Path(args.out)
    path.mkdir(parents=True, exist_ok=True)
    (path / "ceiling.json").write_text(json.dumps(out, indent=1))
    for r in rows:
        print(json.dumps(r))
    print("best:", json.dumps(best, indent=1))
    print("hop over a 4 cm strip needs:", json.dumps(need))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
