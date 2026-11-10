"""The truth planner: what a perfect route would reach on this body. A measuring instrument of the benchmark only — it
reads the world's truth (the layout, where every thing is now, the body's true pose), so it can never be part of the duck.

:class:`Pilot` keeps a body-wide route (:mod:`.floor`) from where the body truly is to a place, re-made when a thing has
been pushed or the place has moved, and turns it into the same three numbers the errand's own steering gives (forward
speed, turn, turn-on-the-spot side) by the errand's own law: face the next point of the route on the spot when it is well
off the heading, else walk at it. The headless runner (:mod:`.runner`) puts those numbers in the place of the errand's
steering (arm ``truth``), or of everything the brain asks of the legs (arms ``truth-reflex``, ``truth-free``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

from duckmind.bench import floor as F

LOOK_M = 0.50  # the route's point it heads for is at most this far along it …
LOOK_MIN_M = 0.12  # … and at least this far (nearer points only when the route ends)
TIGHT_RADIUS_M = 0.08  # the body's bare half width: tried when there is no route with the margin
GOAL_SNAP_M = 0.30  # a place with no room for the body: the nearest spot that has, within this
OUT_SNAP_M = 0.24  # a body standing where the map says it cannot: it heads for the nearest spot it can, within this
ARC_VYAW = 0.55  # the body link lets only an arc at least 0.5 rad/s tight go on within 30 cm of something ahead: its arcs are this tight
ARC_KEEP_S = 1.0  # … for this long after the link last held a straight walk
ARC_FLIP_RAD = 0.25  # … on one side until the route's point is this far off on the other
BACK_S = (
    1.2  # the link holds the body at a touch or a stall until it backs off: a back arc this long
)
CONTACT_GATES = frozenset({"reflex_touch", "stall", "stumble"})
NEAR_GATES = frozenset({"reflex_near", "veer"})
MOVED_M = 0.03  # a thing pushed this far, or the place moved this far: the route is made again
REPLAN_MIN_S = 0.5  # … at most this often
NO_ROUTE_RETRY_S = 2.0  # no route at all: looked for again this often


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _law() -> dict[str, Any]:
    """The errand's own walking law, read from the brain under test (older trees: its values of 2026-11-07)."""
    from duckmind.neuro import brain as B
    from duckmind.neuro.contract import legal_walk

    return {
        "face": float(getattr(B, "ERRAND_FACE_RAD", 0.6)),
        "face_done": float(getattr(B, "ERRAND_FACE_DONE_RAD", 0.25)),
        "near_m": float(getattr(B, "ERRAND_NEAR_M", 0.5)),
        "near_face": float(getattr(B, "ERRAND_NEAR_FACE_RAD", 0.4)),
        "vx": float(getattr(B, "ERRAND_VX", 0.40)),
        "gain": float(getattr(B, "ERRAND_STEER_GAIN", 1.5)),
        "vyaw_max": float(getattr(B, "ERRAND_VYAW_MAX", 0.6)),
        "arc": tuple(getattr(B, "SEEK_ARC", (0.37, 0.9))),
        "reverse": float(getattr(B, "C_REVERSE_VX", -0.40)),
        "legal": legal_walk,
    }


class Pilot:
    """A route to one place over the truth floor, and the walk along it."""

    def __init__(
        self,
        fm: F.FloorMap,
        things_at: Callable[[], Mapping[str, tuple[float, float]]] | None = None,
    ) -> None:
        self.fm = fm
        self.things_at = things_at
        self.law = _law()
        self.goal: tuple[float, float] | None = None
        self.field: Any = None
        self.passable: Any = None
        self.radius = F.BODY_RADIUS_M
        self.goal_cell: tuple[int, int] | None = None
        self.facing = False
        self.replans = 0
        self.tight = 0
        self.no_route = 0
        self.ticks = 0
        self._planned_at = -1e9
        self._checked_at = -1e9
        self._arc_until = -1e9
        self._arc_side = 0.0
        self._back_until = -1e9
        self._back_side = 0.0
        self.backs = 0

    # ── the route ──

    def _plan(self, pose: tuple[float, float, float], goal: tuple[float, float], t: float) -> None:
        fm = self.fm
        if self.things_at is not None:
            fm.set_things(dict(self.things_at()))
        self.goal = goal
        self._planned_at = t
        self.replans += 1
        self.field = None
        for radius in (F.BODY_RADIUS_M, TIGHT_RADIUS_M):
            passable = fm.passable(radius)
            cell = fm.cell_of(*goal)
            if not passable[cell]:
                near = F.nearest_passable(passable, cell, int(GOAL_SNAP_M / fm.cell))
                if near is None:
                    continue
                cell = near
            field = F.cost_to_go(passable, fm.step_cost(radius), cell, fm.cell)
            here = fm.cell_of(pose[0], pose[1])
            if (
                not math.isfinite(float(field[here]))
                and F.nearest_passable(passable, here, int(OUT_SNAP_M / fm.cell), weight=field)
                is None
            ):
                continue
            self.field, self.passable, self.radius, self.goal_cell = field, passable, radius, cell
            if radius < F.BODY_RADIUS_M:
                self.tight += 1
            return

    def update(self, t: float, pose: tuple[float, float, float], goal: tuple[float, float]) -> None:
        """Keep the route current (``t``: the run's clock in s)."""
        stale = (
            self.goal is None
            or math.hypot(goal[0] - self.goal[0], goal[1] - self.goal[1]) > MOVED_M
        )
        if not stale and t - self._checked_at >= REPLAN_MIN_S:
            self._checked_at = t
            if self.things_at is not None:
                now = self.things_at()
                stale = any(
                    math.hypot(x - self.fm.at[k][0], y - self.fm.at[k][1]) > MOVED_M
                    for k, (x, y) in now.items()
                    if k in self.fm.at
                )
            if not stale and self.field is not None:
                # pushed off the route's ground: look again
                here = self.fm.cell_of(pose[0], pose[1])
                stale = (
                    not math.isfinite(float(self.field[here]))
                    and F.nearest_passable(
                        self.passable, here, int(OUT_SNAP_M / self.fm.cell), weight=self.field
                    )
                    is None
                )
            stale = stale or (self.field is None and t - self._planned_at >= NO_ROUTE_RETRY_S)
        if stale and (self.goal is None or t - self._planned_at >= REPLAN_MIN_S):
            self._plan(pose, goal, t)

    def waypoint(self, pose: tuple[float, float, float]) -> tuple[float, float] | None:
        """The world point to head for now, or None when the truth floor has no route (the caller then heads straight)."""
        fm, field = self.fm, self.field
        if field is None or self.goal is None or self.goal_cell is None:
            return None
        here = fm.cell_of(pose[0], pose[1])
        if not math.isfinite(float(field[here])):
            out = F.nearest_passable(self.passable, here, int(OUT_SNAP_M / fm.cell), weight=field)
            return None if out is None else fm.xy_of(*out)
        path = F.descend(field, here, max_cells=int(2 * LOOK_M / fm.cell))
        p = (pose[0], pose[1])
        if path and path[-1] == self.goal_cell:
            snapped = fm.cell_of(*self.goal) != self.goal_cell
            end = fm.xy_of(*self.goal_cell) if snapped else self.goal
            if snapped and math.hypot(end[0] - p[0], end[1] - p[1]) <= 0.06:
                return self.goal  # (a place with no room for the body: from the nearest spot that has, straight at it)
            if math.hypot(end[0] - p[0], end[1] - p[1]) <= LOOK_M and F.line_free(
                self.passable, fm, p, end
            ):
                return end
        best = None
        for cell in path[1:]:
            q = fm.xy_of(*cell)
            d = math.hypot(q[0] - p[0], q[1] - p[1])
            if d > LOOK_M:
                break
            if d >= LOOK_MIN_M and not F.line_free(self.passable, fm, p, q):
                break
            best = q
        if best is None and len(path) > 1:
            best = fm.xy_of(*path[1])
        return best

    # ── the walk along it ──

    def drive(
        self,
        pose: tuple[float, float, float],
        goal: tuple[float, float],
        *,
        t: float = 0.0,
        cap: float | None = None,
        bias: float = 0.0,
        away: float = 0.0,
        can_pivot: bool = True,
        gate: tuple[str, ...] = (),
        may_back: bool = False,
    ) -> tuple[float, float, float, bool]:
        """``(vx, vyaw, pivot side, back arc)`` for this tick by the errand's law toward the route's next point. ``cap`` /
        ``bias``: a reflex's speed cap and turn; ``away``: the side an escape still turns to (no facing turn back against
        it); ``gate``: why the body link changed the last request. A perfect planner knows the body's own rules: where the
        link lets only an arc through (something within 30 cm ahead) it asks for arcs toward the route, and — ``may_back``,
        when nothing else of the brain has the legs — it backs off a touch or a stall in an arc, as the link requires."""
        law = self.law
        self.ticks += 1
        wp = self.waypoint(pose)
        if wp is None:
            self.no_route += 1
            wp = goal
        bearing = _wrap(math.atan2(wp[1] - pose[1], wp[0] - pose[0]) - pose[2])
        side = 1.0 if bearing > 0.0 else -1.0
        held = set(gate)
        if may_back:
            if held & CONTACT_GATES and t >= self._back_until:
                self._back_until = t + BACK_S
                self._back_side = side
                self.backs += 1
            if t < self._back_until:
                self.facing = False
                vx, vyaw = law["legal"](law["reverse"], self._back_side, back_arc=True)
                return float(vx), float(vyaw), 0.0, True
        near = math.hypot(goal[0] - pose[0], goal[1] - pose[1]) < law["near_m"]
        if self.facing:
            self.facing = abs(bearing) > law["face_done"] and can_pivot
        elif can_pivot:
            self.facing = abs(bearing) >= (law["near_face"] if near else law["face"])
        if self.facing and side * away < 0.0:
            self.facing = False
        if self.facing:
            vx, vyaw = law["legal"](law["arc"][0], law["arc"][1] * side)
            return float(vx), float(vyaw), side, False
        speed = law["vx"] if cap is None else min(law["vx"], float(cap))
        turn = max(-law["vyaw_max"], min(law["vyaw_max"], law["gain"] * bearing)) + float(bias)
        if held & NEAR_GATES:
            self._arc_until = t + ARC_KEEP_S
        if t < self._arc_until:
            if self._arc_side == 0.0 or bearing * self._arc_side < -ARC_FLIP_RAD:
                self._arc_side = side
            if abs(turn) < ARC_VYAW:
                turn = self._arc_side * ARC_VYAW
        else:
            self._arc_side = 0.0
        vx, vyaw = law["legal"](speed, max(-1.0, min(1.0, turn)), approach=True)
        return float(vx), float(vyaw), 0.0, False

    def stats(self) -> dict[str, Any]:
        return {
            "replans": self.replans,
            "tight": self.tight,
            "backs": self.backs,
            "no_route_share": round(self.no_route / max(1, self.ticks), 3),
        }


__all__ = ["Pilot"]
