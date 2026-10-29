"""Experimental off-board native-PD balance environment, not official mjlab/BAM training.

Reuses the reviewed scene and World/Body implementation without starting its server.
Reward, reset randomization, action bound and fall thresholds are Duck Mind choices.
The 61 raw observation slots and 14 HOME-relative action slots follow:
https://github.com/pollen-robotics/microduck/blob/a9ec4b2079ef8ee7904014089c885bb07d57d63c/duck-control/src/obs.rs#L12-L31
https://github.com/pollen-robotics/microduck_rl/blob/cb70b792312d559a4da09064d92009079671815f/src/mjlab_microduck/sim/body_server.py#L61-L94

Optional dependencies: gymnasium, numpy, mujoco. No socket, viewer, servo or robotd
is constructed. Passing these local checks is not a hardware safety certificate.
"""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from duckmind.training import verify_upstream

__all__ = ["ACTION_LIMIT", "CONTROL_DT", "LocalBalanceEnv", "configure_upstream"]

ACTION_LIMIT = 0.35  # Custom bounded HOME-relative radians, also used by the exporter.
CONTROL_DT = 0.02
DECIMATION = 4
MIN_HEIGHT = 0.075
MAX_TILT_RAD = 0.65
PROFILE = "native_pd_stand_balance_experimental"


def configure_upstream(upstream: Path) -> ModuleType:
    """Import only native simulation helpers from a clean, reviewed source checkout."""
    upstream = upstream.resolve()
    verify_upstream(upstream)
    source = upstream / "src"
    for name, loaded in tuple(sys.modules.items()):
        if name == "mjlab_microduck" or name.startswith("mjlab_microduck."):
            filename = getattr(loaded, "__file__", None)
            if filename is None or not Path(filename).resolve().is_relative_to(source):
                raise ValueError("another microduck checkout is already imported")
    added_path = str(source) not in sys.path
    if added_path:
        sys.path.insert(0, str(source))
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        module = importlib.import_module("mjlab_microduck.sim.body_server")
    finally:
        sys.dont_write_bytecode = previous_bytecode
        if added_path:
            sys.path.remove(str(source))
    filename = module.__file__
    if (
        filename is None
        or Path(filename).resolve() != source / "mjlab_microduck/sim/body_server.py"
    ):
        raise ValueError("native simulator import did not resolve to reviewed checkout")
    return module


class LocalBalanceEnv(gym.Env):
    """One unheld duck in native MuJoCo, fixed zero command, 50 Hz control.

    Observations are raw (no hidden gyro/joint-velocity scaling or normalizer).
    The applied, clipped policy action is fed back at slots 34:48. Physics uses
    upstream position actuators, not the BAM M6 voltage/friction training model.
    """

    metadata = {"render_modes": [], "render_fps": 50}
    render_mode = None

    def __init__(self, upstream: Path, horizon_steps: int = 250):
        super().__init__()
        if type(horizon_steps) is not int or horizon_steps <= 0:
            raise ValueError("horizon_steps must be a positive integer")
        self.horizon_steps = horizon_steps
        self.upstream = upstream.resolve()
        self._native = configure_upstream(self.upstream)
        self.world = self._native.World(self._native.DEFAULT_SCENE, count=1)
        self.body = self._native.Body(self.world, 0)
        self.world.bodies.append(self.body)
        self._policy_indices = np.array([i for i in range(15) if i != 9])
        if (
            self._native.MOUTH_INDEX != 9
            or tuple(self.body.to_wire) != tuple(self._policy_indices)
            or self.world.model.nu != 14
            or not math.isclose(self.world.model.opt.timestep, 0.005)
        ):
            raise ValueError("upstream native body does not match reviewed 61/14/50Hz contract")
        self.home = np.asarray(self._native.HOME_POSE, dtype=np.float32)[self._policy_indices]
        self.action_space = spaces.Box(-ACTION_LIMIT, ACTION_LIMIT, shape=(14,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(61,), dtype=np.float32)
        self._previous_action = np.zeros(14, dtype=np.float32)
        self._steps = 0
        self._needs_reset = True
        self._closed = False

    def observation(self) -> np.ndarray:
        """Raw robot-observable contract; never includes privileged height or reward."""
        if self._closed:
            raise RuntimeError("environment is closed")
        state = self.body.sensors()
        positions = np.asarray(state["positions"])[self._policy_indices]
        velocities = np.asarray(state["velocities"])[self._policy_indices]
        observation = np.concatenate(
            (
                state["imu"]["gyro"],
                state["imu"]["gravity"],
                positions - self.home,
                velocities,
                self._previous_action,
                np.zeros(13),
            )
        ).astype(np.float32)
        if observation.shape != (61,) or not np.isfinite(observation).all():
            raise FloatingPointError("non-finite native simulator observation")
        return observation

    def _info(self) -> dict[str, Any]:
        state = self.body.sensors()
        height = float(state["trunk_z"])
        upright = float(np.clip(-state["imu"]["gravity"][2], -1, 1))
        tilt = math.acos(upright)
        if not math.isfinite(height) or not math.isfinite(tilt):
            raise FloatingPointError("non-finite native simulator ground truth")
        reason = "height" if height < MIN_HEIGHT else "tilt" if tilt > MAX_TILT_RAD else None
        survival_steps = max(0, self._steps - int(reason is not None))
        return {
            "profile": PROFILE,
            "actuator_model": "native_mujoco_position_pd_not_bam",
            "height": height,
            "trunk_z": height,
            "tilt_rad": tilt,
            "upright": upright,
            "steps": self._steps,
            "survival_steps": survival_steps,
            "survival_s": survival_steps * CONTROL_DT,
            "elapsed_sim_s": float(self.world.data.time),
            "fell": reason is not None,
            "termination_reason": reason,
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if self._closed:
            raise RuntimeError("environment is closed")
        super().reset(seed=seed)
        mujoco.mj_resetData(self.world.model, self.world.data)
        self.body.place(None, self._native.HOME_TRUNK_Z, 0.0)
        # Small, seeded custom perturbations; no upstream training DR claim.
        self.world.data.qpos[self.body.qpos_adr] += self.np_random.uniform(-0.005, 0.005, 14)
        roll, pitch = self.np_random.uniform(-0.035, 0.035, 2)
        cr, sr = math.cos(roll / 2), math.sin(roll / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        self.world.data.qpos[self.body.trunk + 3 : self.body.trunk + 7] = [
            cr * cp,
            sr * cp,
            cr * sp,
            -sr * sp,
        ]
        self.body.set_torque(True)  # Releases the simulator's initial placement hold.
        self.body.set_targets(list(self._native.HOME_POSE))
        mujoco.mj_forward(self.world.model, self.world.data)
        self._previous_action.fill(0)
        self._steps = 0
        self._needs_reset = False
        return self.observation(), self._info()

    def step(self, action: np.ndarray):
        if self._closed or self._needs_reset:
            raise RuntimeError("reset is required before stepping this environment")
        values = np.asarray(action, dtype=np.float32)
        if values.shape != (14,) or not np.isfinite(values).all():
            raise ValueError("action must be a finite vector of shape (14,)")
        applied = np.clip(values, -ACTION_LIMIT, ACTION_LIMIT).astype(np.float32)
        change = applied - self._previous_action
        targets = np.asarray(self._native.HOME_POSE, dtype=np.float64).copy()
        targets[self._policy_indices] = self.home + applied
        self.body.set_targets(targets.tolist())
        self.world.step(DECIMATION)
        self._previous_action = applied.copy()
        self._steps += 1
        observation, info = self.observation(), self._info()
        terminated = bool(info["fell"])
        truncated = self._steps >= self.horizon_steps and not terminated
        # Deliberately small custom balance objective, not the official velocity
        # reward/curriculum: height and tilt are privileged simulator training data.
        terms = {
            "survival": -2.0 if terminated else 1.0,
            "upright": 2.0 * max(0.0, info["upright"]),
            "height": math.exp(-(((info["height"] - 0.10) / 0.04) ** 2)),
            "action_magnitude": -0.05 * float(np.mean(applied**2)),
            "action_rate": -0.1 * float(np.mean(change**2)),
            "angular_velocity": -0.01 * float(np.sum(observation[:3] ** 2)),
        }
        info["reward_terms"] = terms
        reward = float(sum(terms.values()))
        if not math.isfinite(reward):
            raise FloatingPointError("non-finite native simulator reward")
        self._needs_reset = terminated or truncated
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        self._closed = True
        self._needs_reset = True
