"""robotd's tick with ``robot.pose`` (s2:build): the Stage-1 port plus pose mode, in Python.

:class:`PoseRobotd` is :class:`~duckmind.skilltrain.robotd_port.Robotd` with the body-pose intent of
``robotd/src/main.rs`` (read-only upstream, microduck at the pinned build):

* ``robot.pose {z, roll, pitch, active}`` stores the intent (``intents.rs`` ``PoseIntent``: body in
  the observation's order z, roll, pitch; ``active``) — no deadman, it holds until changed;
* every tick, after the twist and head EMAs: ``if pose.active { body_ema += cmd_alpha·(body −
  body_ema) } else { body_ema = 0 }`` — the body **glides** in and **snaps** back on release (the
  prototype's B-button exit), smoothed at ``cmd_alpha`` (not ``head_alpha``);
* ``Controller.step(..., body_active=pose.active)``: while pose mode is held the effective twist is
  zero (``control.rs``: ``if body_active { c.twist = [0; 3] }``) and, with no stand network loaded
  (v5 and our Stage-2 set: ``stand = "none"``), the **walk network** drives with the body block
  ``[0, 0, z, roll, pitch, 0]`` — no network handover;
* the twist EMA keeps integrating the client's twist while posing, so walking resumes from it.

A ``target_hook`` may post-process the final joint targets (the R4 oracle probe drives the head joints
itself with it); ``None`` leaves robotd's targets untouched.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from duckmind.skilltrain.robotd_port import (
    ACTUATOR_MAX,
    ACTUATOR_MIN,
    CMD_ALPHA,
    DEADMAN_S,
    DT,
    HEAD_ALPHA_DEFAULT,
    NOMINAL_V,
    VIN_CLAMP,
    Command,
    PolicySet,
    Robotd,
    Sensors,
    Step,
    Tuning,
)
from duckmind.skilltrain.table import SkillDef


class PoseRobotd(Robotd):
    """robotd's daemon tick for one simulated duck, with pose mode."""

    def __init__(
        self,
        policy: PolicySet,
        skills: Sequence[SkillDef],
        *,
        head_alpha: float = HEAD_ALPHA_DEFAULT,
        cmd_alpha: float = CMD_ALPHA,
        tuning: Tuning = Tuning(),
        voltage_adapt: bool = True,
        nominal_v: float = NOMINAL_V,
    ) -> None:
        super().__init__(
            policy,
            skills,
            head_alpha=head_alpha,
            cmd_alpha=cmd_alpha,
            tuning=tuning,
            voltage_adapt=voltage_adapt,
            nominal_v=nominal_v,
        )
        self.pose_body = np.zeros(3)
        self.pose_active = False
        self.body_ema = np.zeros(3)
        self.target_hook: Callable[[np.ndarray], np.ndarray] | None = None
        self.pose_log: list[dict[str, Any]] = []

    # ── the client's side ──
    def pose(self, z: float, roll: float, pitch: float, active: bool = True) -> None:
        """``robot.pose`` (a notification: z m, roll rad, pitch rad; ``active`` false = release)."""
        self.pose_body = np.array([float(z), float(roll), float(pitch)])
        if bool(active) != self.pose_active:
            self.pose_log.append({"active": bool(active)})
        self.pose_active = bool(active)

    # ── one tick (main.rs order) ──
    def tick(self, sensors: Sensors, now: float, battery_v: float = NOMINAL_V) -> Step:
        c = self.controller
        if self.ready:
            if self.intents.sit_toggle:
                ok, what = c.sit_toggle()
                c.events.append({"t": round(now, 4), "kind": "sit_toggle", "ok": ok, "what": what})
            for index in sorted(self.intents.skills):
                started, why = c.start_skill(index)
                if started is not False:
                    c.events.append(
                        {
                            "t": round(now, 4),
                            "kind": "started" if started else "refused",
                            "skill": c.skills[index].name,
                            "why": why,
                        }
                    )
                else:
                    c.events.append(
                        {"t": round(now, 4), "kind": "refreshed", "skill": c.skills[index].name}
                    )
        self.intents.sit_toggle = False
        self.intents.skills = set()
        stale = now - self.intents.twist_at > DEADMAN_S + 1e-9
        target = np.zeros(3) if stale else np.array(self.intents.twist, dtype=float)
        self.twist_ema += self.cmd_alpha * (target - self.twist_ema)
        self.head_ema += self.head_alpha * (np.array(self.intents.head, float) - self.head_ema)
        if self.pose_active:
            self.body_ema += self.cmd_alpha * (self.pose_body - self.body_ema)
        else:
            self.body_ema = np.zeros(3)
        command = Command(
            twist=(float(self.twist_ema[0]), float(self.twist_ema[1]), float(self.twist_ema[2])),
            head=(
                float(self.head_ema[0]),
                float(self.head_ema[1]),
                float(self.head_ema[2]),
                float(self.head_ema[3]),
            ),
            body=(float(self.body_ema[0]), float(self.body_ema[1]), float(self.body_ema[2])),
        )
        mult = 1.0
        if self.voltage_adapt and battery_v > 0.0:
            mult = self.nominal_v / min(max(battery_v, VIN_CLAMP[0]), VIN_CLAMP[1])
        step = c.step(
            sensors, command, body_active=self.pose_active, dt=DT, scale_mult=mult, now=now
        )
        clamped = np.clip(step.targets, ACTUATOR_MIN, ACTUATOR_MAX)
        if self.target_hook is not None:
            clamped = np.clip(
                np.asarray(self.target_hook(clamped.copy()), dtype=float),
                ACTUATOR_MIN,
                ACTUATOR_MAX,
            )
        self.last = Step(
            clamped, step.label, step.gain, step.busy, step.net, step.scale, step.command
        )
        return self.last


def body_block(z: float, roll: float, pitch: float) -> np.ndarray:
    """The observation's 6-wide body block for a pose (duck-control ``obs.rs``)."""
    return np.array([0.0, 0.0, float(z), float(roll), float(pitch), 0.0])


__all__ = ["PoseRobotd", "body_block"]
