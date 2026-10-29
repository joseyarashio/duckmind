"""robotd's action path as an mjlab action term: what a skill network's output becomes on the
duck, trained exactly as robotd will run it (training plan §2; ``control.rs`` ``Controller::step``
and ``main.rs``'s voltage adaptation, read-only upstream).

Per control tick (50 Hz), for the raw network output ``a`` (14, policy joint order)::

    target  = HOME + action_scale · scale_mult · a          scale_mult = 7.4 / clamp(vin, 6, 9.5)
    target  = α ⊙ target + (1 − α) ⊙ previous_target        α = 0.5 head joints, 0.7 legs
    target  = clamp(target, −π, π)                           previous_target ← target

* ``action_scale`` is the skill table's pinned ``params.action_scale`` (``rl_contract.pinned``);
  the gain ratio is the actuator's ``kp_scale`` (pinned by :func:`pin_gain_ratio`).
* ``vin`` is the BAM actuator's per-duck battery voltage (6.5–8.2 V, startup DR): robotd's
  ``voltage_adapt`` reads the battery, so the servo's voltage-dependent stiffness and robotd's
  compensation are trained together (the neuro-validation harness did the same).
* The filter's ``previous_target`` and the observation's last action start, at every episode,
  from the *entry state* (the network that drove before the skill: v5 velstand walking or
  standing, or sitstand seated — see ``rl_library``), because robotd carries both across networks.
  An entry without a recorded filter state starts unfiltered (robotd's first tick after enable).
* The encoder bias (DR) is handled by mjlab's ``JointPositionAction.apply_actions`` as before.

The evaluator hands the duck back to v5 velstand (scale 0.9) or sitstand (1.0) after the skill by
switching :meth:`RobotdJointPositionAction.set_runtime_scale`; the filter state continues, as in
robotd.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

from duckmind.skilltrain import rl_contract as C

HEAD_NAME_KEYS = ("neck", "head")


@dataclass(kw_only=True)
class RobotdJointPositionActionCfg(JointPositionActionCfg):
    """``JointPositionActionCfg`` + robotd's scale, voltage adaptation and low-pass."""

    skill_action_scale: float = 1.0
    lowpass_head: float | None = C.LOWPASS_HEAD
    lowpass_legs: float | None = C.LOWPASS_LEGS
    voltage_adapt: bool = True
    nominal_voltage: float = C.NOMINAL_V
    vin_clamp: tuple[float, float] = C.VIN_CLAMP
    actuator_clamp: float = C.ACTUATOR_CLAMP

    def build(self, env: Any) -> RobotdJointPositionAction:
        return RobotdJointPositionAction(self, env)


class RobotdJointPositionAction(JointPositionAction):
    cfg: RobotdJointPositionActionCfg

    def __init__(self, cfg: RobotdJointPositionActionCfg, env: Any):
        super().__init__(cfg=cfg, env=env)
        names = list(self._target_names)
        if len(names) != C.ACTION_LEN:
            raise ValueError(f"robotd drives {C.ACTION_LEN} servos, the action term has {names}")
        alpha = []
        for name in names:
            is_head = any(key in name for key in HEAD_NAME_KEYS)
            value = cfg.lowpass_head if is_head else cfg.lowpass_legs
            alpha.append(1.0 if value is None else float(value))
        self._alpha = torch.tensor(alpha, device=self.device).unsqueeze(0)
        n = self.num_envs
        self._prev = (
            self._offset.clone()
            if torch.is_tensor(self._offset)
            else torch.zeros(n, self.action_dim, device=self.device)
        )
        self._has_prev = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._scale_now = torch.full((n, 1), float(cfg.skill_action_scale), device=self.device)
        self.last_target = self._prev.clone()

    # ── robotd's numbers ─────────────────────────────────────────────────────────────────
    def voltage_factor(self) -> torch.Tensor:
        """``nominal / clamp(vin, 6.0, 9.5)`` per duck, (N, 1); 1 without a battery model."""
        if not self.cfg.voltage_adapt:
            return torch.ones(self.num_envs, 1, device=self.device)
        vin = None
        for act in getattr(self._entity, "actuators", []) or []:
            vin = getattr(act, "vin_tensor", None)
            if vin is not None:
                break
        if vin is None:
            return torch.ones(self.num_envs, 1, device=self.device)
        lo, hi = self.cfg.vin_clamp
        return self.cfg.nominal_voltage / vin.reshape(self.num_envs, 1).clamp(lo, hi)

    def set_runtime_scale(self, scale: torch.Tensor | float, env_ids: Any = None) -> None:
        """Override the action scale per duck (the evaluator's hand-back to velstand: 0.9)."""
        idx = slice(None) if env_ids is None else env_ids
        value = torch.as_tensor(scale, device=self.device, dtype=torch.float32)
        self._scale_now[idx] = value.reshape(-1, 1) if value.ndim else value

    @property
    def filter_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._prev, self._has_prev

    # ── mjlab hooks ──────────────────────────────────────────────────────────────────────
    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        target = self._offset + self._scale_now * self.voltage_factor() * self._raw_actions
        filtered = self._alpha * target + (1.0 - self._alpha) * self._prev
        target = torch.where(self._has_prev.unsqueeze(1), filtered, target)
        clamp = float(self.cfg.actuator_clamp)
        target = target.clamp(-clamp, clamp)
        self._prev = target.clone()
        self._has_prev[:] = True
        self._processed_actions = target
        self.last_target = target

    def reset(self, env_ids: Any = None) -> None:
        super().reset(env_ids)
        idx = slice(None) if env_ids is None else env_ids
        self._scale_now[idx] = float(self.cfg.skill_action_scale)
        entry = getattr(self._env, "skill_entry", None)
        if entry is None:
            self._has_prev[idx] = False
            self._prev[idx] = self._offset[idx] if torch.is_tensor(self._offset) else 0.0
            return
        self._prev[idx] = entry.prev_target[idx]
        self._has_prev[idx] = entry.has_prev[idx]
        # robotd feeds the previous network's raw output to the first skill tick (and the
        # action-rate regulariser should see the real jump): the action manager zeroed its
        # history just before calling this, so write the entry's last raw action into it.
        manager = self._env.action_manager
        for buf in (manager._action, manager._prev_action, manager._prev_prev_action):
            buf[idx] = entry.last_raw[idx]


def pin_gain_ratio(env: Any, env_ids: Any = None, gain_ratio: float = 1.0) -> None:
    """Startup event: robotd's gain × ``gain_ratio`` for the whole skill (the skill table's
    pinned ``params.gain_ratio``) = the BAM actuator's per-duck ``kp_scale``."""
    del env_ids
    robot = env.scene["robot"]
    for act in robot.actuators:
        scale = getattr(act, "kp_scale", None)
        if scale is not None:
            scale[:] = float(gain_ratio)
            default = getattr(act, "default_kp_scale", None)
            if default is not None:
                default[:] = float(gain_ratio)


def robotd_numbers(cfg: RobotdJointPositionActionCfg, gain_ratio: float) -> dict[str, Any]:
    """What the action term applies, for manifests and metrics."""
    return {
        "action_scale": float(cfg.skill_action_scale),
        "gain_ratio": float(gain_ratio),
        "lowpass_head": cfg.lowpass_head,
        "lowpass_legs": cfg.lowpass_legs,
        "voltage_adapt": f"{cfg.nominal_voltage}/clamp(vin,{cfg.vin_clamp[0]},{cfg.vin_clamp[1]})"
        if cfg.voltage_adapt
        else None,
        "actuator_clamp_rad": round(float(cfg.actuator_clamp), 6),
        "control_hz": C.CONTROL_HZ,
        "trained_with_lowpass": True,
    }


def lowpass_cutoff_hz(alpha: float, hz: float = C.CONTROL_HZ) -> float:
    """−3 dB frequency of y ← α·x + (1−α)·y at ``hz`` (for the docs: 0.5 → 5.5 Hz, 0.7 → 9.6)."""
    if alpha >= 1.0:
        return float("inf")
    a = 1.0 - alpha
    # |H(e^{jw})|^2 = α² / (1 − 2a cos w + a²) = 1/2
    cosw = (1.0 + a * a - 2.0 * alpha * alpha) / (2.0 * a)
    return math.acos(max(-1.0, min(1.0, cosw))) * hz / (2.0 * math.pi)


__all__ = [
    "RobotdJointPositionAction",
    "RobotdJointPositionActionCfg",
    "lowpass_cutoff_hz",
    "pin_gain_ratio",
    "robotd_numbers",
]
