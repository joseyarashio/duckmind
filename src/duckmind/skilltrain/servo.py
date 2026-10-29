"""The two servo models every Stage-1 gate runs on.

* ``mjcf`` — **the owner's live simulator**: the MJCF's own XML position servos
  (``chosen_actuator``: kp 0.55, force range ±0.96 N·m), the scene's own solver and contacts.
  robotd's gain reaches them the way the official body server applies it
  (``body_server.py`` ``set_gain``: gain and bias × kp/200).
* ``bam`` — **training's physics**: the BAM XL330 m6 model of microduck_rl's actuator config
  (``microduck_constants.py``: kp_fw 200, battery 6.5–8.2 V with a load sag of 0–0.2 V/N·m and a
  6.0 V floor, the 1.75 A firmware current limit, force range 8.2·kt/R), the
  ``mjlab_frictionloss`` branch of bam (git 62bd8ce, the commit microduck_rl's lock pins): the
  velocity-independent friction budget (Coulomb + Stribeck + directional load terms, × a per-duck
  friction scale — ``FrictionDRBamActuator``) is written into ``dof_frictionloss`` and the
  viscous part into ``dof_damping`` every physics step, with stiff friction constraints, so
  MuJoCo's solver does the static-friction clipping. A CPU port of phase 1's ``VariantDuck``
  (``rise/headless_variants.py`` in the measurement records), which the pet:body lane used for its
  4,010 trials. Training's solver (implicitfast, 30 / 50 iterations, the SitStand recipe) and
  contacts (XL330 housings ``condim`` 1, the feet ``priority`` 1) come with it
  (:func:`training_physics`).

The bam package is imported from wherever it is installed: run with
``--with "better-actuator-models @ git+https://github.com/Rhoban/bam.git@62bd8ce12154340be97e06f7f41a0ca8f116d967"``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

KP_FW = 200.0  # microduck's firmware stiffness (training cfg); robotd's gain 200 = ratio 1
VIN_RANGE = (6.5, 8.2)
VIN_DROP_RANGE = (0.0, 0.2)
VIN_MIN = 6.0
STIFF_SOLREF_FRICTION = (-5.0e4, -2.0e2)  # bam.mjlab BamActuator._STIFF_SOLREF_FRICTION
STIFF_SOLIMP_FRICTION = (0.99, 0.9999, 0.001, 0.5, 2.0)
SERVO_MESH = "xl330"  # microduck_constants.SERVO_MESH_NAME
BAM_GIT = "62bd8ce12154340be97e06f7f41a0ca8f116d967"
TRAINING_INTEGRATOR, TRAINING_ITERATIONS, TRAINING_LS_ITERATIONS = "implicitfast", 30, 50
TRAINING_SOLVER: dict[str, Any] = {
    "integrator": TRAINING_INTEGRATOR,
    "iterations": TRAINING_ITERATIONS,
    "ls_iterations": TRAINING_LS_ITERATIONS,
}
_INTEGRATORS = {"euler": 0, "rk4": 1, "implicit": 2, "implicitfast": 3}


class XmlServos:
    """The MJCF's position servos (mjcf profile). ``ceiling_nm``: the force range."""

    name = "mjcf"

    def __init__(self, model: Any, act: np.ndarray) -> None:
        self.act = np.asarray(act)
        self.base_gain = model.actuator_gainprm[self.act, 0].copy()
        self.base_bias = model.actuator_biasprm[self.act, 1].copy()
        self.ceiling_nm = float(np.min(np.abs(model.actuator_forcerange[self.act, 1])))
        self.gain = 200

    def set_gain(self, model: Any, gain: int) -> None:
        if gain == self.gain:
            return
        scale = float(gain) / 200.0
        model.actuator_gainprm[self.act, 0] = self.base_gain * scale
        model.actuator_biasprm[self.act, 1] = self.base_bias * scale
        self.gain = int(gain)

    def before_step(self, model: Any, data: Any, target: np.ndarray) -> None:
        data.ctrl[self.act] = target

    def info(self) -> dict[str, Any]:
        return {
            "servo": self.name,
            "ceiling_nm": round(self.ceiling_nm, 4),
            "kp": [round(float(v), 4) for v in np.unique(self.base_gain)],
        }


class BamServos:
    """BAM XL330 m6 as torque motors (bam profile), one duck."""

    name = "bam"

    def __init__(
        self,
        model: Any,
        data: Any,
        act: np.ndarray,
        jq: np.ndarray,
        jv: np.ndarray,
        *,
        vin: float = 7.4,
        vin_drop_gain: float = 0.0,
        vin_min: float = VIN_MIN,
        friction_scale: float = 1.0,
        armature_scale: float = 1.0,
        stiff_friction: bool = True,
    ) -> None:
        import mujoco

        try:
            from bam.actuator import NumpyBackend
            from bam.model import load_model
        except ImportError as error:  # pragma: no cover - environment
            raise ImportError(
                "the bam servo model needs bam (better-actuator-models) at git "
                f'{BAM_GIT}: uv run --with "better-actuator-models @ '
                f'git+https://github.com/Rhoban/bam.git@{BAM_GIT}"'
            ) from error
        self._mj = mujoco
        self.act, self.jq, self.jv = np.asarray(act), np.asarray(jq), np.asarray(jv)
        bm = load_model(motor_name="xl330", model="m6")
        bm.actuator.backend = NumpyBackend()
        bm.actuator.kp = KP_FW
        self.bm = bm
        kt, res = float(bm.kt.value), float(bm.R.value)
        self.force_limit = max(VIN_RANGE) * kt / res  # training: the top of vin_range
        max_current = getattr(bm.actuator, "max_current", None)
        self.ceiling_nm = float(kt * max_current) if max_current else self.force_limit
        for a in self.act:
            model.actuator_gaintype[a] = int(mujoco.mjtGain.mjGAIN_FIXED)
            model.actuator_gainprm[a, :] = 0.0
            model.actuator_gainprm[a, 0] = 1.0
            model.actuator_biastype[a] = int(mujoco.mjtBias.mjBIAS_NONE)
            model.actuator_biasprm[a, :] = 0.0
            model.actuator_forcelimited[a] = 1
            model.actuator_forcerange[a] = (-self.force_limit, self.force_limit)
        armature = float(bm.actuator.get_extra_inertia()) * float(armature_scale)
        for dof in self.jv:
            model.dof_armature[dof] = armature
            model.dof_damping[dof] = 0.0
            model.dof_frictionloss[dof] = 0.0
            if stiff_friction:
                model.dof_solref[dof] = STIFF_SOLREF_FRICTION
                model.dof_solimp[dof] = STIFF_SOLIMP_FRICTION
        mujoco.mj_setConst(model, data)
        self.vin = float(vin)
        self.vin_drop_gain = float(vin_drop_gain)
        self.vin_min = float(vin_min)
        self.friction_scale = float(friction_scale)
        self.armature = armature
        self.kp_scale = 1.0
        self.prev_motor = np.zeros(len(self.jv))
        self.vin_now = self.vin

    def set_gain(self, model: Any, gain: int) -> None:
        self.kp_scale = float(gain) / 200.0

    def before_step(self, model: Any, data: Any, target: np.ndarray) -> None:
        """One physics step of the branch's ``BamActuator.compute()`` for this duck."""
        mujoco, bm = self._mj, self.bm
        act = bm.actuator
        vin = self.vin
        if self.vin_drop_gain:
            vin = max(
                self.vin_min, vin - self.vin_drop_gain * float(np.sum(np.abs(self.prev_motor)))
            )
        self.vin_now = vin
        act.vin = vin
        act.kp = KP_FW * self.kp_scale
        q = np.array(data.qpos[self.jq])
        dq = np.array(data.qvel[self.jv])
        control = act.compute_control(np.asarray(target, float), q, dq, model.opt.timestep)
        motor = np.asarray(act.compute_torque(control, True, q, dq), float)
        # the gearbox's external load: bias + constraint forces minus our own dof-friction rows
        qfrc_fric = np.zeros(model.nv)
        n = int(data.nefc)
        if n:
            types = np.asarray(data.efc_type[:n])
            rows = np.nonzero(types == int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF))[0]
            if rows.size:
                np.add.at(
                    qfrc_fric,
                    np.asarray(data.efc_id[:n])[rows],
                    np.asarray(data.efc_force[:n])[rows],
                )
        ext = (
            -np.asarray(data.qfrc_bias)[self.jv]
            + np.asarray(data.qfrc_constraint)[self.jv]
            - qfrc_fric[self.jv]
        )
        prev_act = np.asarray(data.qfrc_actuator)[self.jv]
        strib = np.exp(-np.power(np.abs(dq) / bm.dtheta_stribeck.value, bm.alpha.value))
        fl = np.full_like(motor, bm.friction_base.value) + strib * bm.friction_stribeck.value
        fl += np.abs(
            ext * bm.load_friction_external.value - prev_act * bm.load_friction_motor.value
        )
        fl += strib * np.abs(
            ext * bm.load_friction_external_stribeck.value
            - prev_act * bm.load_friction_motor_stribeck.value
        )
        drive = (np.abs(prev_act) > np.abs(ext)).astype(float)
        fl += strib * (
            drive * bm.load_friction_external_quad.value * ext**2
            + (1.0 - drive) * bm.load_friction_motor_quad.value * prev_act**2
        )
        model.dof_frictionloss[self.jv] = fl * self.friction_scale
        model.dof_damping[self.jv] = bm.friction_viscous.value
        self.prev_motor = motor
        data.ctrl[self.act] = motor

    def info(self) -> dict[str, Any]:
        bm = self.bm
        return {
            "servo": self.name,
            "model": "xl330/m6 (bam git " + BAM_GIT[:7] + ")",
            "kp_fw": KP_FW,
            "vin": round(self.vin, 3),
            "vin_drop_gain": round(self.vin_drop_gain, 4),
            "vin_min": self.vin_min,
            "max_current_A": getattr(bm.actuator, "max_current", None),
            "ceiling_nm": round(self.ceiling_nm, 4),
            "force_limit_nm": round(self.force_limit, 4),
            "friction_scale": round(self.friction_scale, 4),
            "armature": round(self.armature, 6),
        }


def training_physics(model: Any, servo_geoms: np.ndarray, foot_geoms: np.ndarray) -> dict[str, Any]:
    """Training's solver and contact settings on a compiled model: implicitfast, 30 / 50
    iterations; XL330 housings ``condim`` 1 (``FULL_COLLISION``: named ``*_collision`` geoms);
    the feet ``priority`` 1 (their own friction rules their contacts)."""
    model.opt.integrator = _INTEGRATORS[TRAINING_INTEGRATOR]
    model.opt.iterations = TRAINING_ITERATIONS
    model.opt.ls_iterations = TRAINING_LS_ITERATIONS
    for g in servo_geoms:
        model.geom_condim[int(g)] = 1
    for g in foot_geoms:
        model.geom_condim[int(g)] = 3
        model.geom_priority[int(g)] = max(1, int(model.geom_priority[int(g)]))
    return {
        **TRAINING_SOLVER,
        "servo_housings_condim": 1,
        "feet_priority": 1,
        "servo_housings": int(len(servo_geoms)),
    }


def stall_mask(
    force: np.ndarray, vel: np.ndarray, torque_nm: float = 0.4, vel_rad_s: float = 0.5
) -> np.ndarray:
    """velstand's ``servo_stall``: > 0.4 N·m at < 0.5 rad/s."""
    return (np.abs(force) > torque_nm) & (np.abs(vel) < vel_rad_s)


__all__ = [
    "BAM_GIT",
    "BamServos",
    "KP_FW",
    "TRAINING_SOLVER",
    "VIN_DROP_RANGE",
    "VIN_RANGE",
    "XmlServos",
    "stall_mask",
    "training_physics",
]
