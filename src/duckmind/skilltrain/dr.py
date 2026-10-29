"""Per-duck domain randomisation for the Mac gates (one draw per trial, seeded, recorded).

The ranges are the velocity/velstand recipe's final ones (``microduck_velocity_env_cfg.py``, the
curriculum's end values the A100 neuro validation used), plus the training plan's R1 extra:

* **physical** (both servo models): trunk CoM ±15 mm and each head-assembly body's CoM ±10 mm
  (``HEAD_BODY_NAMES``, including training's ``bearing_roll``), trunk mass and inertia ×0.95–1.05
  (``pseudo_inertia``: CoM kept), foot friction 0.5–1.3 (training 0.7–1.3; plan §3.1 "floor μ
  down to 0.5") with the feet's ``priority`` raised so their friction rules their contacts;
* **servo** (bam only): battery 6.5–8.2 V, sag gain 0–0.2 V/N·m, BAM friction budget ×0.9–1.1,
  armature ×0.9–1.1 — and robotd's voltage adaptation sees the battery (both models);
* **path** (bam only): command delay 15–30 ms (3–6 physics steps), encoder bias ±0.015 rad per
  joint, IMU mounting error ≤ 6° (random axis) on the gyro and gravity, observation noise
  (gyro ±0.03, gravity ±0.01, joints ±0.001, velocities ±0.25, uniform), the joint velocities one
  control tick old and the IMU 0–1 tick old.

``mjcf`` runs the physical draw only: it stands for the owner's live simulator, whose robotd
reads the body server's exact state with no delay (lockstep) at a nominal 7.4 V.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

TRUNK_COM_M = 0.015
HEAD_COM_M = 0.010
MASS_RANGE = (0.95, 1.05)
FOOT_MU_RANGE = (0.5, 1.3)
VIN_RANGE = (6.5, 8.2)
VIN_DROP_RANGE = (0.0, 0.2)
BAM_FRICTION_RANGE = (0.9, 1.1)
ARMATURE_RANGE = (0.9, 1.1)
LAG_STEPS = (3, 6)
ENCODER_BIAS = 0.015
IMU_TILT_DEG = 6.0
NOISE = {"gyro": 0.03, "gravity": 0.01, "q": 0.001, "qd": 0.25}
HEAD_BODIES = ("neck", "neck_pitch", "yaw_roll_motion", "jaw_soft", "bearing_roll")
PRESETS = ("none", "physical", "recipe")


@dataclass(frozen=True)
class Draw:
    """One duck. ``preset``: none | physical | recipe."""

    seed: int
    preset: str
    trunk_com: tuple[float, float, float] = (0.0, 0.0, 0.0)
    head_com: tuple[tuple[float, float, float], ...] = ()
    mass_scale: float = 1.0
    foot_mu: float | None = None
    vin: float = 7.4
    vin_drop: float = 0.0
    bam_friction: float = 1.0
    armature: float = 1.0
    lag_steps: int = 0
    enc_bias: tuple[float, ...] = ()
    imu_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    imu_deg: float = 0.0
    noise: bool = False
    qd_lag_ticks: int = 0
    imu_lag_ticks: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        return {k: _round(v) for k, v in out.items()}


def draw(seed: int, preset: str = "recipe") -> Draw:
    """The draw for ``seed`` (numpy's PCG64 on the seed: the same seed, the same duck)."""
    if preset not in PRESETS:
        raise ValueError(f"DR preset {preset!r}; use one of {PRESETS}")
    if preset == "none":
        return Draw(seed=int(seed), preset=preset)
    rng = np.random.default_rng(int(seed))
    trunk = tuple(float(v) for v in rng.uniform(-TRUNK_COM_M, TRUNK_COM_M, 3))
    heads = tuple(
        tuple(float(v) for v in rng.uniform(-HEAD_COM_M, HEAD_COM_M, 3)) for _ in HEAD_BODIES
    )
    mass = float(rng.uniform(*MASS_RANGE))
    mu = float(rng.uniform(*FOOT_MU_RANGE))
    # drawn always (so a seed means the same physical duck under both presets)
    vin = float(rng.uniform(*VIN_RANGE))
    drop = float(rng.uniform(*VIN_DROP_RANGE))
    fric = float(rng.uniform(*BAM_FRICTION_RANGE))
    arm = float(rng.uniform(*ARMATURE_RANGE))
    lag = int(rng.integers(LAG_STEPS[0], LAG_STEPS[1] + 1))
    bias = tuple(float(v) for v in rng.uniform(-ENCODER_BIAS, ENCODER_BIAS, 14))
    axis = rng.normal(size=3)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    tilt = float(rng.uniform(0.0, IMU_TILT_DEG))
    imu_lag = int(rng.integers(0, 2))
    if preset == "physical":
        return Draw(
            seed=int(seed),
            preset=preset,
            trunk_com=_t3(trunk),
            head_com=tuple(_t3(h) for h in heads),
            mass_scale=mass,
            foot_mu=mu,
        )
    return Draw(
        seed=int(seed),
        preset=preset,
        trunk_com=_t3(trunk),
        head_com=tuple(_t3(h) for h in heads),
        mass_scale=mass,
        foot_mu=mu,
        vin=vin,
        vin_drop=drop,
        bam_friction=fric,
        armature=arm,
        lag_steps=lag,
        enc_bias=bias,
        imu_axis=(float(axis[0]), float(axis[1]), float(axis[2])),
        imu_deg=tilt,
        noise=True,
        qd_lag_ticks=1,
        imu_lag_ticks=imu_lag,
    )


def apply_physical(
    d: Draw, model: Any, data: Any, *, trunk: int, foot_geoms: Sequence[int] | np.ndarray
) -> None:
    """The physical part of a draw on a compiled model (then ``mj_setConst``)."""
    import mujoco

    if d.preset == "none":
        return
    model.body_ipos[trunk] += np.asarray(d.trunk_com)
    for name, offset in zip(HEAD_BODIES, d.head_com, strict=False):
        body = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
        if body >= 0:
            model.body_ipos[body] += np.asarray(offset)
    model.body_mass[trunk] *= d.mass_scale
    model.body_inertia[trunk] *= d.mass_scale
    if d.foot_mu is not None:
        for g in foot_geoms:
            model.geom_friction[int(g), 0] = d.foot_mu
            model.geom_priority[int(g)] = max(1, int(model.geom_priority[int(g)]))
    mujoco.mj_setConst(model, data)


class ObsPath:
    """What robotd reads, bent the way training bends it (bam preset ``recipe``): delays,
    encoder bias, IMU mounting error, uniform noise. Identity for other draws."""

    def __init__(self, d: Draw) -> None:
        self.d = d
        self.on = d.preset == "recipe"
        self.rng = np.random.default_rng(int(d.seed) + 7_919)
        self.rot = _axis_angle(d.imu_axis, math.radians(d.imu_deg))
        self.bias = np.asarray(d.enc_bias if d.enc_bias else np.zeros(14), dtype=float)
        self.prev_qd: np.ndarray | None = None
        self.prev_imu: tuple[np.ndarray, np.ndarray] | None = None

    def __call__(
        self, gyro: np.ndarray, gravity: np.ndarray, q: np.ndarray, qd: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.on:
            return gyro, gravity, q, qd
        g_now = self.rot @ np.asarray(gyro, float)
        grav_now = self.rot @ np.asarray(gravity, float)
        if self.d.imu_lag_ticks and self.prev_imu is not None:
            g_out, grav_out = self.prev_imu
        else:
            g_out, grav_out = g_now, grav_now
        self.prev_imu = (g_now, grav_now)
        qd_now = np.asarray(qd, float)
        qd_out = self.prev_qd if (self.d.qd_lag_ticks and self.prev_qd is not None) else qd_now
        self.prev_qd = qd_now
        q_out = np.asarray(q, float) + self.bias
        if self.d.noise:
            u = self.rng.uniform
            g_out = g_out + u(-NOISE["gyro"], NOISE["gyro"], 3)
            grav_out = grav_out + u(-NOISE["gravity"], NOISE["gravity"], 3)
            q_out = q_out + u(-NOISE["q"], NOISE["q"], 14)
            qd_out = qd_out + u(-NOISE["qd"], NOISE["qd"], 14)
        return g_out, grav_out, q_out, qd_out


def _axis_angle(axis: Sequence[float], angle: float) -> np.ndarray:
    k = np.asarray(axis, float)
    n = float(np.linalg.norm(k))
    if n < 1e-12 or abs(angle) < 1e-12:
        return np.eye(3)
    k = k / n
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(angle) * kx + (1.0 - math.cos(angle)) * (kx @ kx)


def _t3(v: Sequence[float]) -> tuple[float, float, float]:
    return (float(v[0]), float(v[1]), float(v[2]))


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, (list, tuple)):
        return [_round(v) for v in value]
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    return value


__all__ = ["Draw", "HEAD_BODIES", "ObsPath", "PRESETS", "apply_physical", "draw"]
