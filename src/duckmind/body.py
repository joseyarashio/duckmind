"""Backend-neutral boundary between cognition and embodied skills.

Only the Embodied Skill Layer receives this port. Cognition emits semantic
intents and never receives an API capable of addressing joints or servos.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Protocol, runtime_checkable

Vector3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class SafetyObservation:
    fallen: bool
    limp: bool


@dataclass(frozen=True, slots=True)
class RobotObservation:
    # Local receipt time: unlike robotd's process-relative `t`, this can be
    # compared to local dispatch/timeout timestamps.
    monotonic_s: float
    policy: str
    movement_requested: Vector3
    movement_applied: Vector3
    limited_by: tuple[str, ...]
    safety: SafetyObservation
    backend: str = "unknown"
    metadata: Mapping[str, object] | None = None
    robot_time_s: float | None = None
    # Derived inside the body adapter from measured head joints. Cognition
    # receives a scalar verification signal, never joint indices or targets.
    head_tracking_error_rad: float | None = None


@dataclass(frozen=True, slots=True)
class BodyIntent:
    """A robotd policy intent, never a direct servo/gain/torque write."""

    method: str
    params: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.params, Mapping):
            raise ValueError("intent parameters must be an object")
        fields = {
            "robot.move": frozenset({"vx", "vy", "vyaw"}),
            "robot.look": frozenset({"x", "y", "z", "neck_pitch"}),
            "robot.head": frozenset({"neck_pitch", "head_pitch", "head_yaw", "head_roll"}),
            # 0 closed .. 1 open; the mouth servo is outside every policy (robotd clamps it).
            "robot.mouth": frozenset({"open"}),
            "robot.do": frozenset({"skill"}),
            "robot.stop": frozenset(),
        }
        if self.method not in fields:
            raise ValueError(f"method is outside the Embodied Skill Layer allowlist: {self.method}")
        extra = set(self.params).difference(fields[self.method])
        if extra:
            raise ValueError(f"unexpected parameters for {self.method}: {sorted(extra)}")
        if self.method == "robot.do":
            skill = self.params.get("skill")
            if not isinstance(skill, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", skill):
                raise ValueError("skill must be a non-empty robotd skill name")
        elif self.method == "robot.look":
            if not {"x", "y", "z"}.issubset(self.params):
                raise ValueError("robot.look requires x, y, and z in trunk-frame metres")
        for name, value in self.params.items():
            if name == "skill":
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))


@dataclass(frozen=True, slots=True)
class BodyAck:
    accepted: bool
    reason: str | None = None
    # False for a fire-and-forget continuous notification: it only reached the
    # local socket. A later robot.state sample is the execution evidence.
    confirmed: bool = True


@runtime_checkable
class BodyPort(Protocol):
    """Same API for real robotd, robotd-over-MuJoCo, and replay fakes."""

    def dispatch_intent(self, intent: BodyIntent) -> BodyAck: ...

    def observe(self) -> RobotObservation: ...

    def stop(self) -> BodyAck: ...
