"""The Act Runner's motion primitives and the rules no curiosity act may break (curiosity spec
§5.1; physical-design spec §5.3; owner decisions A1–A3, 2026-09-27).

Every primitive stays inside what velstand was trained on (M1, J3, phase-1 reversing):

======== ======================================== ============================================
name     command                                  use
======== ======================================== ============================================
leg      vx 0.40, wz = clip(2·bearing, ±1.0),     walk toward the thing, steering as it goes
         0.5–2.0 s                                (owner A3: 0.40 m/s toward a specific thing)
creep    vx 0.40, 0.5 s, then stop and look again the last 40 cm (one step ≈ :data:`calib`)
trim     vx 0.40, 0.3 s, at most 2 in a row       the last few centimetres
bump     vx 0.40, 0.6 s                           the deliberate gentle bump (§5.3)
back     vx −0.40, 0.6 s                          too close: a step back
back_arc vx −0.40, wz ±1.0, 0.6–1.0 s             turn beside a thing (reverse arc)
turn_arc vx 0.40, wz ±1.0, 0.5–1.0 s              the forward half of a three-point turn
======== ======================================== ============================================

**Never** (checked by :func:`problems` on every output the runner makes, and by the body
loop's own guard): |vx| > 0.40 m/s or |wz| > 1.0 rad/s; turning on the spot (vx = 0, wz ≠ 0);
0 < |vx| < 0.30 (velstand does not step: it only turns); reversing outside a curiosity act's
``back``/``back_arc``; a skill outside :data:`AUTONOMOUS_SKILLS`; a skill while walking or
within :data:`SKILL_STILL_MS` of the last step; ``roulade``, ``crouch``, ``roller``; walking
with the head off HOME except the final approach toward the just-seen target (owner A2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

AUTONOMOUS_SKILLS = frozenset({"ground_pick", "kick_left", "kick_right"})
FORBIDDEN_SKILLS = frozenset({"roulade", "crouch", "roller"})
VX_MAX_MM = 400  # |vx| cap for every curiosity command (curiosity §5.1, M1)
APPROACH_VX_MM = 400  # owner A3: approaching a specific thing at 0.40 m/s
MIN_STEP_VX_MM = 300  # below 0.30 m/s velstand turns but does not step (phase 2 E2)
VYAW_MAX_MRAD = 1000
SKILL_STILL_MS = 300  # no kick or pick within 0.3 s of the last step (no kick while walking)
HEAD_HOME_TOL_RAD = 0.02


@dataclass(frozen=True)
class Prim:
    """One primitive: the twist (mm/s, mrad/s) held for ``ms`` milliseconds."""

    name: str
    vx_mm: int
    vyaw_mrad: int
    ms: int
    reverse: bool = False

    def __post_init__(self) -> None:
        for problem in twist_problems(self.vx_mm, self.vyaw_mrad, reverse_ok=self.reverse):
            raise ValueError(f"{self.name}: {problem}")
        if not 100 <= self.ms <= 6000:
            raise ValueError(f"{self.name}: {self.ms} ms is outside 100–6000")


def clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def leg(bearing_mrad: int, ms: int, *, gain: float = 2.0) -> Prim:
    """Walk toward a bearing, steering as it goes (curiosity §5.1 ``leg``)."""
    vyaw = int(round(clip(gain * bearing_mrad, VYAW_MAX_MRAD)))
    return Prim("leg", APPROACH_VX_MM, vyaw, max(500, min(2000, ms)))


def creep(ms: int = 500, vyaw_mrad: int = 0) -> Prim:
    """A short step forward, steering with |wz| ≤ 1.0 (an arc: in velstand's range, M1)."""
    return Prim("creep", APPROACH_VX_MM, int(clip(vyaw_mrad, VYAW_MAX_MRAD)), ms)


def trim(vyaw_mrad: int = 0) -> Prim:
    """0.3 s: under robotd it mostly turns the duck a little (live: 6 mm, −6°)."""
    return Prim("trim", APPROACH_VX_MM, int(clip(vyaw_mrad, VYAW_MAX_MRAD)), 300)


def bump(ms: int = 600) -> Prim:
    return Prim("bump", APPROACH_VX_MM, 0, ms)


def back(ms: int = 600) -> Prim:
    return Prim("back", -VX_MAX_MM, 0, ms, reverse=True)


def back_arc(left: bool, ms: int = 600) -> Prim:
    """Reverse arc: backing up while turning. ``left`` turns the nose left (wz > 0)."""
    return Prim("back_arc", -VX_MAX_MM, VYAW_MAX_MRAD if left else -VYAW_MAX_MRAD, ms, reverse=True)


def turn_arc(left: bool, ms: int = 1000) -> Prim:
    return Prim("turn_arc", APPROACH_VX_MM, VYAW_MAX_MRAD if left else -VYAW_MAX_MRAD, ms)


def twist_problems(vx_mm: int, vyaw_mrad: int, *, reverse_ok: bool = False) -> list[str]:
    """What is wrong with this twist for a curiosity act (empty: nothing)."""
    out: list[str] = []
    if abs(vx_mm) > VX_MAX_MM:
        out.append(f"|vx| {abs(vx_mm)} mm/s > {VX_MAX_MM}")
    if abs(vyaw_mrad) > VYAW_MAX_MRAD:
        out.append(f"|wz| {abs(vyaw_mrad)} mrad/s > {VYAW_MAX_MRAD}")
    if vx_mm == 0 and vyaw_mrad != 0:
        out.append("turning on the spot (vx 0, wz not 0)")
    if 0 < abs(vx_mm) < MIN_STEP_VX_MM:
        out.append(f"|vx| {abs(vx_mm)} mm/s < {MIN_STEP_VX_MM}: velstand does not step")
    if vx_mm < 0 and not reverse_ok:
        out.append("reversing outside back/back_arc")
    return out


def head_is_home(head: tuple[float, float, float, float] | None) -> bool:
    return head is None or all(abs(v) <= HEAD_HOME_TOL_RAD for v in head)


def problems(
    *,
    twist: tuple[int, int] | None,
    do: str | None,
    head: tuple[float, float, float, float] | None,
    prim: str | None,
    ms_since_step: int,
    head_down_ok: bool,
) -> list[str]:
    """Every rule one runner output breaks (the property tests assert this stays empty).

    ``prim`` names the primitive the twist belongs to (reversing only for back/back_arc);
    ``ms_since_step`` is how long the body has been still (for skills); ``head_down_ok`` is
    owner A2's exception (the final approach toward the just-seen target)."""
    out: list[str] = []
    moving = twist is not None and twist != (0, 0)
    if twist is not None:
        out += twist_problems(*twist, reverse_ok=prim in ("back", "back_arc"))
    if do is not None:
        if do in FORBIDDEN_SKILLS or do not in AUTONOMOUS_SKILLS:
            out.append(f"skill {do!r} is not autonomous")
        if moving:
            out.append("a skill while walking")
        elif ms_since_step < SKILL_STILL_MS:
            out.append(f"a skill {ms_since_step} ms after a step (< {SKILL_STILL_MS})")
    if moving and not head_is_home(head) and not head_down_ok:
        out.append("walking with the head off HOME")
    return out


def bearing_mrad(x_mm: float, y_mm: float) -> int:
    return int(round(math.atan2(y_mm, x_mm) * 1000.0))


__all__ = [
    "APPROACH_VX_MM",
    "AUTONOMOUS_SKILLS",
    "FORBIDDEN_SKILLS",
    "MIN_STEP_VX_MM",
    "Prim",
    "SKILL_STILL_MS",
    "VX_MAX_MM",
    "VYAW_MAX_MRAD",
    "back",
    "back_arc",
    "bearing_mrad",
    "bump",
    "creep",
    "leg",
    "problems",
    "trim",
    "turn_arc",
    "twist_problems",
]
