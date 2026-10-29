"""kick-cone-v1: kick only when the whole predicted stopping region is somewhere a ball may end
up (curiosity spec §5.7, M3/M4). The body layer checks it with fresh perception; Duck Core
checks the same function with its own belief before it asks.

The kick goes along the trunk heading plus the foot's offset (:data:`FOOT_OFFSET_MRAD`, measured
live under robotd, see :mod:`duckmind.curiosity.calib`). The ball stops somewhere in the sector
``direction ± HALF_WIDTH_MRAD`` between this ball's p10 and p90 roll on this floor. Every sampled
point of that sector must be inside the leash (the excursion's activity circle minus
:data:`LEASH_MARGIN_MM`) and outside every keep-out circle (hazards, no-touch and fragile things
with :data:`KEEP_OFF_MM`, family and pets within a metre). Blocked: try the other foot; then walk
round behind the ball (at most twice); then ``dribble``; then end with ``cone_blocked``.

Integers in the odometry frame (mm, mrad). Pure.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

HALF_WIDTH_MRAD = 349  # ±20°
LEASH_MARGIN_MM = 100
KEEP_OFF_MM = 400
ANGLE_STEP_MRAD = 87  # 5°
RANGE_STEP_MM = 100
# Where each kick sends the ball relative to the trunk heading: curiosity spec §5.7 (M2, duckphys)
# said left +6°, right −13°; the phase-3 live robotd measurement (calib.CALIB.kick_dir_mrad,
# which the Act Runner passes in) says +24° / −20°.
FOOT_OFFSET_MRAD = {"left": 105, "right": -227}


@dataclass(frozen=True)
class Circle:
    x_mm: int
    y_mm: int
    r_mm: int
    why: str = "keep_out"


@dataclass(frozen=True)
class ConeCheck:
    ok: bool
    foot: str
    direction_mrad: int
    why: str | None = None
    worst_mm: int | None = None


def sector_points(
    ball: tuple[int, int],
    direction_mrad: int,
    roll_mm: tuple[int, int],
    half_mrad: int = HALF_WIDTH_MRAD,
) -> list[tuple[int, int]]:
    """The sampled stopping region: ±half around the direction, p10…p90 of the roll."""
    lo, hi = roll_mm
    lo = max(0, lo)
    hi = max(lo, hi)
    points = []
    angle = -half_mrad
    while angle <= half_mrad:
        a = (direction_mrad + angle) / 1000.0
        dist = lo
        while True:
            points.append(
                (int(round(ball[0] + dist * math.cos(a))), int(round(ball[1] + dist * math.sin(a))))
            )
            if dist >= hi:
                break
            dist = min(hi, dist + RANGE_STEP_MM)
        angle += ANGLE_STEP_MRAD
    return points


def check(
    ball: tuple[int, int],
    heading_mrad: int,
    foot: str,
    roll_mm: tuple[int, int],
    *,
    leash: Circle | None,
    keep_out: Sequence[Circle] = (),
    foot_offset_mrad: dict[str, int] | None = None,
    half_mrad: int = HALF_WIDTH_MRAD,
) -> ConeCheck:
    """Is kicking with ``foot`` now inside the rules? (odometry frame)"""
    offsets = foot_offset_mrad or FOOT_OFFSET_MRAD
    direction = heading_mrad + offsets[foot]
    worst = None
    for x, y in sector_points(ball, direction, roll_mm, half_mrad):
        if leash is not None:
            inside = leash.r_mm - LEASH_MARGIN_MM - math.hypot(x - leash.x_mm, y - leash.y_mm)
            worst = inside if worst is None else min(worst, inside)
            if inside < 0:
                return ConeCheck(False, foot, direction, "leash", int(round(inside)))
        for zone in keep_out:
            if math.hypot(x - zone.x_mm, y - zone.y_mm) < zone.r_mm:
                return ConeCheck(False, foot, direction, zone.why, None)
    return ConeCheck(True, foot, direction, None, None if worst is None else int(round(worst)))


def choose(
    ball: tuple[int, int],
    heading_mrad: int,
    preferred: str,
    roll_mm: tuple[int, int],
    *,
    leash: Circle | None,
    keep_out: Sequence[Circle] = (),
    foot_offset_mrad: dict[str, int] | None = None,
    half_mrad: int = HALF_WIDTH_MRAD,
) -> ConeCheck:
    """The preferred foot if its cone is clear, else the other foot's check (§5.7 step 1)."""
    first = check(
        ball,
        heading_mrad,
        preferred,
        roll_mm,
        leash=leash,
        keep_out=keep_out,
        foot_offset_mrad=foot_offset_mrad,
        half_mrad=half_mrad,
    )
    if first.ok:
        return first
    other = "right" if preferred == "left" else "left"
    second = check(
        ball,
        heading_mrad,
        other,
        roll_mm,
        leash=leash,
        keep_out=keep_out,
        foot_offset_mrad=foot_offset_mrad,
        half_mrad=half_mrad,
    )
    return second if second.ok else first


__all__ = ["Circle", "ConeCheck", "FOOT_OFFSET_MRAD", "check", "choose", "sector_points"]
