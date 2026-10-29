"""Going home round a wall: a side taken off the nest's bearing (owner item 1, 2026-10-04).

Going home, the nest is a bearing and a range from the duck's own odometry (a seen tag, or the
place it learnt). When a wall stands on that bearing, steering straight at the nest only presses the
duck into the wall: the boundary memory bends it away a little and the goal pulls it back (a local
minimum: a dock behind a wall, measured 0/16 docked). Animals meet this with the oldest trick there
is, "bug" navigation: take a side and keep to it until the way to the goal is open, then go on.

No map and no world truth. What the duck has: the boundary memory (floor line, ToF and bumps carried
by odometry; its points sit a few tens of cm off) and the ToF's centre return (15 Hz). The detour does
not steer round the wall itself: it only turns the *bearing the walk aims at* aside from the nest's,
and the boundary memory's own steering (``Boundary.steer``) bends the walk away from what is ahead,
as always. So it goes along a wall, round a corner, through a doorway the way the memory steers it.

* begin: going home has got no nearer for NO_PROGRESS_S and a seen obstacle stands on the nest's
  bearing (nearer than DETOUR_BLOCK_M and than the nest), or the ToF's centre return is that near
  with the nest about ahead, or an escape (``unstick.py``) toward the nest, for DETOUR_ON_S.
* side: the obstacle goes on the side away from the way it turns: the way the wall's line (its
  points on the bearing) runs toward the nest; with a line square to the nest, the side the ToF
  finds more room; with no line, the last side (kept DETOUR_SIDE_KEEP_S) or the freer side.
* walk: the bearing is the nest's turned aside by ``theta`` (rad, at most THETA_MAX), growing
  while the nest's own bearing is shut (the memory shows the way along it less than DETOUR_FREE_M
  clear, or the ToF's centre return is near with the nest about ahead) and shrinking when it is
  open: an opening that faces the nest is entered as it comes in line.
* end: ``theta`` back to nothing and the way open for DETOUR_OFF_S, after DETOUR_MIN_S.
* budget: DETOUR_SIDE_S on one side, then once the other side, then DETOUR_PAUSE_S of the plain walk.
* off within DETOUR_MIN_RANGE_M of the nest (the docking's own business).
* the setting ``DETOUR_ENABLED`` (``NeuroBrain(detour=...)`` overrides it per duck): off, ``step``
  hands back the nest's bearing as it is.

Pure and clock-injected (``t_ms``); deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from duckmind.neuro.boundary import Boundary
from duckmind.neuro.contract import SensorFrame, wrap_angle

DETOUR_MIN_RANGE_M = 1.2  # nearer than this to the nest: the docking's own business, no detour
DETOUR_END_RANGE_M = 0.9  # … once it began, until this near
DETOUR_BLOCK_M = (
    1.0  # a wall seen on the goal's bearing closer than this (and nearer than the goal)
)
DETOUR_TOF_M = 0.75  # the ToF's centre return this near with the goal about ahead
DETOUR_FREE_M = (
    1.5  # the goal's bearing is open when the memory shows this much clear (or the goal)
)
DETOUR_ON_S = 0.5  # blocked this long before it begins
DETOUR_OFF_S = 1.0  # open this long before it ends
DETOUR_MIN_S = 2.0  # follow at least this long before it may end
DETOUR_SIDE_S = 50.0  # one side's budget (over the ground the duck makes ~0.1 m/s: 5 m is 50 s)
DETOUR_PAUSE_S = 30.0  # both sides tried: the plain walk for this long
DETOUR_SIDE_KEEP_S = 25.0
DETOUR_KEEP_S = 15.0  # another act has the walk this long: the episode is forgotten
DETOUR_ESCAPE_S = 12.0  # an escape toward the goal counts as a block for this long
DETOUR_ESCAPE_ARC = 0.9  # … when the goal is within this of the trunk's heading
LEAVE_ARC = 1.0
NO_PROGRESS_S = 5.0  # it begins only when going home has got nowhere this long
PROGRESS_M = 0.25  # … getting this much nearer to the nest is progress
THETA_MAX = 1.7  # the nest's bearing turned aside by at most this (rad)
THETA_START = 1.2  # … when no wall line can be read
THETA_MIN0 = 0.8  # … the least it starts with
THETA_UP = 0.6  # rad/s turned further aside while the nest's bearing is shut
THETA_DOWN = 0.5  # rad/s turned back while it is open
WALL_NEAR_M = 0.55  # a wall beside it nearer than this is "there"; farther (or none) it has ended
LOST_S = 0.3  # the wall beside it gone this long: round the corner, toward it
LAT_PER_RANGE = 0.26  # the side ToF beam is 15° off: lateral distance = range * sin 15°
SIDE_HALF_M = 0.12  # points this near the trunk's line are its own body or a block mark
DETOUR_TOF_CLEAR_M = 0.9  # the ToF's centre return this far (or none) with the nest about ahead
WAYPOINT_M = 1.0
FOLLOW_CAP = 0.4  # m/s: the speed along a wall
FOLLOW_GO = 0.85  # how much of the time it walks along a wall (0..1; the rhythm's own `go`)
UPDATE_HZ = 10.0
# The setting: whether a duck made without saying goes round walls on the way home. (Measured
# 2026-10-04, tools/machine_verify/measure_unstuck.py: see docs/design/neural-duck/NOTES-brain-motor.md.)
DETOUR_ENABLED = False
BODY_CONE = (-0.17, 0.0, 0.17)
TOF_AGE_MS = 500


@dataclass(frozen=True, slots=True)
class Waypoint:
    bearing: float  # rad from the trunk, + = left
    range_m: float
    following: bool


class Detour:
    """The detour state (module doc)."""

    __slots__ = (
        "enabled",
        "following",
        "_idle_t",
        "theta",
        "wall",
        "_side_t",
        "_t0",
        "_follow_s",
        "_blocked_s",
        "_open_s",
        "_lost_s",
        "_t",
        "_last",
        "_tried",
        "_pause_until",
        "_esc_n",
        "_esc_until",
        "episodes",
        "gave_up",
        "ahead",
        "_best",
        "_best_t",
        "dbg",
    )

    def __init__(self, enabled: bool | None = None) -> None:
        self.enabled = DETOUR_ENABLED if enabled is None else bool(enabled)
        self.following = False
        self._idle_t: int | None = None
        self.theta = 0.0  # how far (rad) the walk is turned aside from the nest's bearing
        self.wall = 0  # the side the obstacle is on while it follows: +1 left, −1 right
        self._side_t = -1_000_000
        self._t0 = 0
        self._follow_s = 0.0
        self._blocked_s = 0.0
        self._open_s = 0.0
        self._lost_s = 0.0
        self._t = -1_000_000
        self._last: Waypoint | None = None
        self._tried: set[int] = set()
        self._pause_until = -1
        self._esc_n = 0
        self._esc_until = -1
        self.episodes = 0
        self.gave_up = 0
        self._best = 1e9  # the nearest the nest has been this bout (m) and when it last got nearer
        self._best_t = 0
        self.ahead = 9.9  # the way the memory shows clear along the goal's bearing (telemetry)
        self.dbg: tuple[float, float, int, float] = (
            9.9,
            9.9,
            0,
            0.0,
        )  # lat, ahead_m, flips, bearing

    def idle(self, t_ms: int) -> None:
        """Not going home this tick (another act has the walk): the episode is kept for
        DETOUR_KEEP_S (acts cut in on each other all the time), then forgotten (the side stays
        remembered a while longer)."""
        if self._idle_t is None:
            self._idle_t = t_ms
        elif t_ms - self._idle_t > DETOUR_KEEP_S * 1000:
            self.following = False
            self._blocked_s = 0.0
            self._open_s = 0.0
            self._tried = set()
            self._last = None
            self._best = 1e9
            self.theta = 0.0

    def step(
        self,
        t_ms: int,
        bound: Boundary,
        odom: tuple[float, float, float],
        frame: SensorFrame,
        goal_bearing: float,
        goal_range: float,
        *,
        escapes: int = 0,
        free_side: int = 0,
    ) -> Waypoint:
        """The bearing to walk toward now (the goal's, or the goal's turned aside)."""
        plain = Waypoint(goal_bearing, goal_range, False)
        self._idle_t = None
        if not self.enabled:
            return plain
        period = 1000.0 / UPDATE_HZ
        if self._last is not None and t_ms - self._t < period:
            if self._last.following:
                return self._last
            return plain
        dt = max(0.0, min(0.5, (t_ms - self._t) / 1000.0)) if self._t > -1_000_000 else 0.0
        self._t = t_ms
        if goal_range < self._best - PROGRESS_M or self._best > 1e8:
            self._best = goal_range
            self._best_t = t_ms
        stalled = t_ms - self._best_t >= NO_PROGRESS_S * 1000
        if escapes > self._esc_n and abs(goal_bearing) <= DETOUR_ESCAPE_ARC:
            self._esc_until = t_ms + int(DETOUR_ESCAPE_S * 1000)
        self._esc_n = escapes
        if t_ms < self._pause_until or goal_range < (
            DETOUR_END_RANGE_M if self.following else DETOUR_MIN_RANGE_M
        ):
            self.following = False
            self._last = plain
            return plain
        tof = frame.tof
        tof_ok = tof.present and tof.age_ms <= TOF_AGE_MS
        a_mm = tof.ahead_mm if tof_ok else None
        a = (a_mm / 1000.0) if a_mm is not None else 9.9
        heading = odom[2] + goal_bearing
        goal_pos = (
            odom[0] + goal_range * math.cos(heading),
            odom[1] + goal_range * math.sin(heading),
        )
        rel = bound._relative(odom, goal_pos)  # noqa: SLF001
        ahead = min(bound.clearance(odom, goal_bearing + d, rel=rel) for d in BODY_CONE)
        self.ahead = ahead
        need = min(goal_range - 0.15, DETOUR_BLOCK_M)
        goal_ahead = abs(goal_bearing) <= LEAVE_ARC
        blocked = ahead < need or (goal_ahead and a < DETOUR_TOF_M) or t_ms < self._esc_until
        if not self.following:
            self._blocked_s = self._blocked_s + dt if blocked and stalled else 0.0
            if self._blocked_s < DETOUR_ON_S:
                self._last = plain
                return plain
            self.following = True
            self._t0 = t_ms
            self._follow_s = 0.0
            self._open_s = 0.0
            self.episodes += 1
            self._tried = set()
            self._begin(rel, odom, goal_bearing, tof if tof_ok else None, free_side, t_ms)
            self._tried.add(self.wall)
        else:
            self._follow_s += dt
            open_now = (
                ahead >= min(goal_range - 0.15, DETOUR_FREE_M)
                and (not goal_ahead or a >= DETOUR_TOF_CLEAR_M)
                and t_ms >= self._esc_until
            )
            self._open_s = self._open_s + dt if open_now else 0.0
            lat = self._lateral(rel, self.wall)
            side_mm = (tof.left_mm if self.wall > 0 else tof.right_mm) if tof_ok else None
            if side_mm is not None:
                tof_lat = side_mm / 1000.0 * LAT_PER_RANGE
                lat = tof_lat if lat is None else min(lat, tof_lat)
            near = lat is not None and lat < WALL_NEAR_M
            self._lost_s = 0.0 if near else self._lost_s + dt
            # the nest's own bearing open: give the turn back; a wall beside it: hold the turn
            # aside; the wall gone (its end, an opening): give the turn back, toward the wall
            down = open_now or (not near and self._lost_s >= LOST_S)
            self.theta = max(
                0.0, min(THETA_MAX, self.theta + (-THETA_DOWN if down else THETA_UP) * dt)
            )
            if (
                self.theta <= 0.0
                and self._open_s >= DETOUR_OFF_S
                and self._follow_s >= DETOUR_MIN_S
            ):
                self.following = False
                self._side_t = t_ms
                self._blocked_s = 0.0
                self._last = plain
                return plain
            if t_ms - self._t0 > DETOUR_SIDE_S * 1000:
                if len(self._tried) < 2:
                    self.wall = -self.wall
                    self._tried.add(self.wall)
                    self._t0 = t_ms
                    self.theta = THETA_START
                else:
                    self.following = False
                    self.gave_up += 1
                    self._pause_until = t_ms + int(DETOUR_PAUSE_S * 1000)
                    self._last = plain
                    return plain
        self._side_t = t_ms
        turn = -self.wall  # the way it turns off the goal's bearing: away from the wall's side
        bearing = wrap_angle(goal_bearing + turn * self.theta)
        self._last = Waypoint(bearing, WAYPOINT_M, True)
        self.dbg = (self.theta, ahead, self.wall, bearing)
        return self._last

    def _begin(
        self,
        rel: list[tuple[float, float, float]],
        odom: tuple[float, float, float],
        goal_bearing: float,
        tof: object | None,
        free_side: int,
        t_ms: int,
    ) -> None:
        """Choose the way along the wall (``theta``) and the wall's side
        (``wall``): from the wall's line (the points on the goal's bearing) the direction that runs
        toward the goal; with no line to read, a quarter turn to the freer side."""
        gc, gs = math.cos(goal_bearing), math.sin(goal_bearing)
        pts = []
        for fx, fy, _w in rel:
            along = fx * gc + fy * gs
            across = -fx * gs + fy * gc
            if 0.0 < along <= 1.6 and abs(across) <= 0.9:
                pts.append((along, across))
        remembered = self.wall != 0 and t_ms - self._side_t <= DETOUR_SIDE_KEEP_S * 1000
        if len(pts) >= 4:
            ma = sum(p[0] for p in pts) / len(pts)
            mc = sum(p[1] for p in pts) / len(pts)
            saa = sum((p[0] - ma) ** 2 for p in pts)
            scc = sum((p[1] - mc) ** 2 for p in pts)
            sac = sum((p[0] - ma) * (p[1] - mc) for p in pts)
            tr = saa + scc
            det = saa * scc - sac * sac
            lam1 = 0.5 * (tr + math.sqrt(max(0.0, tr * tr - 4.0 * det)))
            lam2 = tr - lam1
            if lam1 > 4.0 * max(lam2, 1e-6):
                ang = 0.5 * math.atan2(2.0 * sac, saa - scc)  # the line's direction, goal frame
                cand = (ang, wrap_angle(ang + math.pi))
                pick = max(cand, key=lambda a: math.cos(a))
                if remembered:
                    # the same way as the last time (its wall side is kept), when the line allows
                    keep = [a for a in cand if self._wall_side(a, ma, mc) == self.wall]
                    if keep:
                        pick = keep[0]
                elif abs(math.cos(pick)) < 0.2:
                    # the line is square to the goal: no way is nearer, the freer side
                    side = self._freer(tof, free_side)
                    pick = min(cand, key=lambda a: abs(wrap_angle(a - side * math.pi / 2.0)))
                if abs(pick) >= 0.3:
                    self.wall = -1 if pick > 0.0 else 1
                else:
                    self.wall = self._wall_side(pick, ma, mc)
                self.theta = max(THETA_MIN0, min(THETA_MAX, abs(pick)))
                return
        # no line to read (a thing, or only an escape's word for it): a quarter turn
        turn = self.wall * -1 if remembered else self._freer(tof, free_side)
        self.wall = -turn
        self.theta = THETA_START

    @staticmethod
    def _lateral(rel: list[tuple[float, float, float]], w: int) -> float | None:
        """The nearest remembered point beside it on side ``w`` (its distance across, m), from just
        behind the trunk to 0.9 m ahead; None when the memory has none there."""
        best: float | None = None
        for fx, fy, _w in rel:
            if (
                -0.3 <= fx <= 0.9
                and fy * w > 0.0
                and SIDE_HALF_M <= abs(fy) <= 1.0
                and (best is None or abs(fy) < best)
            ):
                best = abs(fy)
        return best

    @staticmethod
    def _wall_side(direction: float, ma: float, mc: float) -> int:
        """Walking along ``direction`` (goal frame), the wall's centroid (ma, mc) is on the left (+1)
        or the right (−1)."""
        cross = math.cos(direction) * mc - math.sin(direction) * ma
        return 1 if cross > 0.0 else -1

    @staticmethod
    def _freer(tof: object | None, free_side: int) -> int:
        """+1 left / −1 right: the side the ToF finds more room, else the memory's freer side."""
        if tof is not None:
            left = getattr(tof, "left_mm", None) or 9999
            right = getattr(tof, "right_mm", None) or 9999
            if abs(left - right) > 150:
                return 1 if left > right else -1
        if free_side:
            return 1 if free_side > 0 else -1
        return -1
