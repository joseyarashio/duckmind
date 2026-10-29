"""Arc paths for legs that only turn while they walk (a:play, 2026-09-28).

The duck's legs never turn on the spot: a turn is an arc at ≥ 0.35 m/s with |yaw rate| ≤ 1 rad/s
(owner rule, :func:`duckmind.neuro.contract.legal_walk`). Measured in this lane's rig (the
headless robotd port, the jaw on, 4 s at the command): ``vx 0.40, vyaw +1.0`` turned 100° over
0.53 m (radius ≈ 0.35 m), ``vyaw −1.0`` turned 179° over 0.44 m (≈ 0.2 m), ``vyaw +0.5`` only
33°. To come in on a toy along its perpendicular (its long side toward the beak, the only way the
jaw closes on it) the duck plans the shortest path of arcs and straights from where it is to a
pose on that line — Dubins' curves: two arcs and a straight between them (LSL, RSR, LSR, RSL) or
three arcs (RLR, LRL), with a turning radius no tighter than the legs make — and follows it with
a look-ahead point, re-planning when it drifts off.

Pure (math only). Poses are (x, y, heading) in the duck's odometry frame, metres and radians.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

TWO_PI = 2.0 * math.pi
WORDS = ("LSL", "RSR", "LSR", "RSL", "RLR", "LRL")
TURN_R_M = 0.35  # plan no tighter than the legs' slower (left) turn at vx 0.40, vyaw 1.0
STEP_M = 0.02  # the path's samples
LOOK_M = 0.18  # the look-ahead point: this far along the path from the nearest sample
OFF_PATH_M = 0.15  # this far off the path: plan again
REPLAN_MS = 1500  # … but not more often than this


def _mod2pi(a: float) -> float:
    return a - TWO_PI * math.floor(a / TWO_PI)


def _wrap(a: float) -> float:
    return (a + math.pi) % TWO_PI - math.pi


@dataclass(frozen=True, slots=True)
class ArcPath:
    """One Dubins path: its word (``"LSR"`` …) and three segment lengths — arcs in radians of
    turn, the straight in units of the radius — for turning radius ``r``."""

    word: str
    t: float
    p: float
    q: float
    r: float

    @property
    def length(self) -> float:
        return (self.t + self.p + self.q) * self.r


def _candidates(alpha: float, beta: float, d: float) -> dict[str, tuple[float, float, float]]:
    """The six words' (t, p, q) where they exist (Shkel & Lumelsky's normalised form: the start
    at the origin heading ``alpha`` off the start→goal line, the goal ``d`` radii along it)."""
    sa, sb, ca, cb = math.sin(alpha), math.sin(beta), math.cos(alpha), math.cos(beta)
    c_ab = math.cos(alpha - beta)
    d2 = d * d
    out: dict[str, tuple[float, float, float]] = {}
    p2 = 2.0 + d2 - 2.0 * c_ab + 2.0 * d * (sa - sb)
    if p2 >= 0.0:
        tmp = math.atan2(cb - ca, d + sa - sb)
        out["LSL"] = (_mod2pi(tmp - alpha), math.sqrt(p2), _mod2pi(beta - tmp))
    p2 = 2.0 + d2 - 2.0 * c_ab + 2.0 * d * (sb - sa)
    if p2 >= 0.0:
        tmp = math.atan2(ca - cb, d - sa + sb)
        out["RSR"] = (_mod2pi(alpha - tmp), math.sqrt(p2), _mod2pi(tmp - beta))
    p2 = -2.0 + d2 + 2.0 * c_ab + 2.0 * d * (sa + sb)
    if p2 >= 0.0:
        p = math.sqrt(p2)
        tmp = math.atan2(-ca - cb, d + sa + sb) - math.atan2(-2.0, p)
        out["LSR"] = (_mod2pi(tmp - alpha), p, _mod2pi(tmp - _mod2pi(beta)))
    p2 = -2.0 + d2 + 2.0 * c_ab - 2.0 * d * (sa + sb)
    if p2 >= 0.0:
        p = math.sqrt(p2)
        tmp = math.atan2(ca + cb, d - sa - sb) - math.atan2(2.0, p)
        out["RSL"] = (_mod2pi(alpha - tmp), p, _mod2pi(beta - tmp))
    tmp = (6.0 - d2 + 2.0 * c_ab + 2.0 * d * (sa - sb)) / 8.0
    if abs(tmp) <= 1.0:
        phi = math.atan2(ca - cb, d - sa + sb)
        p = _mod2pi(TWO_PI - math.acos(tmp))
        t = _mod2pi(alpha - phi + _mod2pi(p / 2.0))
        out["RLR"] = (t, p, _mod2pi(alpha - beta - t + _mod2pi(p)))
    tmp = (6.0 - d2 + 2.0 * c_ab + 2.0 * d * (sb - sa)) / 8.0
    if abs(tmp) <= 1.0:
        phi = math.atan2(ca - cb, d + sa - sb)
        p = _mod2pi(TWO_PI - math.acos(tmp))
        t = _mod2pi(-alpha - phi + p / 2.0)
        out["LRL"] = (t, p, _mod2pi(_mod2pi(beta) - alpha - t + _mod2pi(p)))
    return out


def _advance(
    pose: tuple[float, float, float], kind: str, ds: float, r: float
) -> tuple[float, float, float]:
    x, y, h = pose
    if kind == "S":
        return x + ds * math.cos(h), y + ds * math.sin(h), h
    sign = 1.0 if kind == "L" else -1.0
    dh = sign * ds / r
    # the chord of the arc
    cx, cy = x - sign * r * math.sin(h), y + sign * r * math.cos(h)
    h2 = h + dh
    return cx + sign * r * math.sin(h2), cy - sign * r * math.cos(h2), h2


def sample(
    start: tuple[float, float, float], path: ArcPath, step: float = STEP_M
) -> list[tuple[float, float, float]]:
    """The path's poses every ``step`` metres (start and end included)."""
    pts = [start]
    pose = start
    for kind, amount in zip(path.word, (path.t, path.p, path.q), strict=True):
        length = amount * path.r
        n = max(1, int(math.ceil(length / step))) if length > 1e-9 else 0
        for _ in range(n):
            pose = _advance(pose, kind, length / n, path.r)
            pts.append(pose)
    return pts


def paths(
    start: tuple[float, float, float], goal: tuple[float, float, float], r: float = TURN_R_M
) -> list[ArcPath]:
    """Every Dubins path from ``start`` to ``goal`` with turning radius ``r``, shortest first
    (each one checked: its sampled end lands on the goal)."""
    dx, dy = goal[0] - start[0], goal[1] - start[1]
    d = math.hypot(dx, dy) / r
    theta = _mod2pi(math.atan2(dy, dx)) if d > 1e-9 else 0.0
    alpha = _mod2pi(start[2] - theta)
    beta = _mod2pi(goal[2] - theta)
    out: list[ArcPath] = []
    for word, (t, p, q) in _candidates(alpha, beta, d).items():
        path = ArcPath(word, t, p, q, r)
        end = sample(start, path, step=0.05)[-1]
        if (
            math.hypot(end[0] - goal[0], end[1] - goal[1]) < 1e-3
            and abs(_wrap(end[2] - goal[2])) < 1e-3
        ):
            out.append(path)
    out.sort(key=lambda p: p.length)
    return out


def clearance(points: Sequence[tuple[float, float, float]], obstacle: tuple[float, float]) -> float:
    """The nearest a sampled path comes to a point (m)."""
    ox, oy = obstacle
    return min((math.hypot(x - ox, y - oy) for x, y, _ in points), default=math.inf)


def plan_clear(
    start: tuple[float, float, float],
    goal: tuple[float, float, float],
    keep_out: tuple[float, float] | None = None,
    keep_out_m: float = 0.0,
    r: float = TURN_R_M,
) -> tuple[ArcPath, list[tuple[float, float, float]]] | None:
    """The shortest path that stays ``keep_out_m`` from ``keep_out`` (sampled), or None."""
    for path in paths(start, goal, r):
        pts = sample(start, path)
        if keep_out is None or clearance(pts, keep_out) >= keep_out_m:
            return path, pts
    return None


@dataclass
class Follower:
    """Follows sampled poses with a look-ahead point: (vx, vyaw) each tick, ``done`` at the end,
    ``off`` when it has drifted too far from the path (the caller plans again)."""

    points: list[tuple[float, float, float]]
    look_m: float = LOOK_M
    vx: float = 0.40
    kp: float = 2.5
    index: int = 0
    done: bool = False
    off: float = 0.0
    trail: list[int] = field(default_factory=list)

    def step(self, pose: tuple[float, float, float]) -> tuple[float, float]:
        pts = self.points
        if not pts:
            self.done = True
            return 0.0, 0.0
        x, y, h = pose
        # the nearest sample from here on (progress never goes back), in a window ahead
        best, best_d = self.index, math.inf
        for i in range(self.index, min(len(pts), self.index + 60)):
            d = math.hypot(pts[i][0] - x, pts[i][1] - y)
            if d < best_d:
                best, best_d = i, d
        self.index = best
        self.off = best_d
        if best >= len(pts) - 2:
            self.done = True
            return 0.0, 0.0
        ahead = min(len(pts) - 1, best + max(1, int(round(self.look_m / STEP_M))))
        tx, ty, _ = pts[ahead]
        bearing = _wrap(math.atan2(ty - y, tx - x) - h)
        if len(self.trail) < 400:
            self.trail.append(best)
        return self.vx, max(-1.0, min(1.0, self.kp * bearing))


__all__ = [
    "ArcPath",
    "Follower",
    "LOOK_M",
    "OFF_PATH_M",
    "REPLAN_MS",
    "STEP_M",
    "TURN_R_M",
    "WORDS",
    "clearance",
    "paths",
    "plan_clear",
    "sample",
]
