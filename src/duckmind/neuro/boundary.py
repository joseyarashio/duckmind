"""Where the walls and furniture are: a boundary memory from what the duck has seen, carried by
its own odometry, for steering away *before* touching (the owner, 2026-09-28: bumping into walls
and falling in corners reads as a robot).

Animals do not find walls by walking into them. The camera's floor line (duck-eye FL-v1's five
free-floor sectors, :class:`~duckmind.neuro.contract.FreeSector`; MODEL in the prototype) says,
a few times a second while the light is good, how far the floor is clear across the picture and
what ends it. Like the boundary-vector cells of a vertebrate's navigation system, this module
keeps where the clear floor ended — as points in the duck's own contact-odometry frame, so they
stay put while the duck walks and turns (path integration), fading unless seen again (a leaky
memory: τ :data:`MEMORY_TAU_S`) and erased where a newer look shows clear floor. The ToF (when
fitted) and a bump (proprio's ``blocked``) lay points down too.

From that memory, each steering update (≤ :data:`STEER_HZ`) asks how far the way is clear along
a fan of headings in a corridor as wide as the body (:meth:`Boundary.clearance`) and gives the
walk (:class:`Steer`):

* ``veer`` — a yaw-rate bias (rad/s, + = left) toward the best clear heading, growing as the way
  ahead closes in (the closer the wall, the harder the turn: an arc, never a turn on the spot),
  nothing while the way ahead is clear beyond :data:`SLOW_M`;
* ``cap`` — a speed cap: an arc's speed when a wall is near, the slowest arc when it is close;
* ``urgency`` — 0..1, how pressing it is (the competition's ``obstacle`` / ``turn_away``);
* ``side`` — the freer side (+1 left, −1 right) for a turn-around.

A goal the behaviour walks to (a thing to investigate, a person to follow, the nest) is not a
wall: points near it do not count. Honest senses: only the frame's camera floor line, the ToF,
the body events and the odometry; never a world position. Pure and clock-injected (``t_ms``);
deterministic. Phase 5 (fixed point): points in mm, headings in mrad, weights in ppm, the decay
a per-tick constant.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from duckmind.neuro.contract import FLOOR_RANGE_M, SensorFrame, wrap_angle

MEMORY_TAU_S = 25.0  # a boundary point fades unless seen again
KEEP_WEIGHT = 0.12  # below this it is forgotten
COUNT_WEIGHT = 0.3  # a point counts for steering above this weight
MAX_POINTS = 160
POSE_KEEP_MS = 2500  # odometry history kept to place a look at its capture time
BODY_HALF_M = 0.11  # half the corridor the body needs (the duck is ~0.16 m wide, plus margin)
SLOW_M = 1.1  # the way ahead closer than this: start bending away (live: the body needs ~0.8 m
# of path to turn 90° at its slowest, tightest arc, so it must start early)
NEAR_M = 0.6  # … closer than this: an arc's speed at most
STOP_M = 0.32  # … closer than this: only the slowest, tightest arc away
GOAL_CLEAR_M = 0.22  # points this close to the goal are the goal, not a wall
# … and for a loose thing it walks up to (2026-10-05, legs): where it believes the thing is can be a
# hand's breadth or two off where the eye saw its edge laid down (the belief is a smoothed
# estimate, the points are the floor line's), and the thing is as wide as it is: the walk was bent
# away from the very thing it was walking to (measured: the way ahead "closed" 0.4 m from it)
GOAL_CLEAR_THING_M = 0.45
STEER_HZ = 10.0
# The way ahead is the nearest clearance in a ±20° cone (walls that converge — a corner — count
# before the straight corridor closes: live 2026-09-28, the explorer walked into a corner).
CONE = (-0.35, -0.175, 0.0, 0.175, 0.35)
FAN = tuple(round(-3.0 + 0.2 * i, 2) for i in range(31))  # candidate headings (rad), ±172°
# Floor it has SEEN clear (the floor lines' free rays) or walked over (its trail) is inviting;
# where it has not looked is only this inviting (a wall there has not been seen, not ruled
# out): it turns toward open floor it saw, not into the unknown beside a wall.
UNKNOWN_M = 0.9
RAY_NEAR_M = 0.6  # a seen-clear ray counts from this close to where it was seen from
TRAIL_STEP_M = 0.15  # the walked trail: a point every this far …
TRAIL_KEEP = 60  # … the last this many (~9 m)
TURN_COST = 0.22  # m of clearance a radian of turning is worth (prefer small bends)
RIGHT_BIAS_M = 0.06  # this body turns right faster (left arcs are 1.7–2.8× slower: pet:probe)
GOAL_COST = 0.35  # m per radian away from the goal
APPROACH_CAP = 0.40  # m/s: the way ahead unseen, it walks no faster than this
VEER_GAIN = 1.1  # rad/s of bias per rad toward the best heading, at full urgency
GENTLE_VEER = 0.28  # rad/s: while the wall is still beyond NEAR_M the bend stays below an arc
# Once it has begun to bend away from a wall, the other way must be clearly better (by this
# much clearance) before it swings back: no dithering between left and right while the wall
# closes in (live 2026-09-28: the veer flipped −1 → +0.5 → −1 over 2 s near the south wall, the
# arc began late, ran along the wall into the dock's guide rail and the duck fell).
COMMIT_M = 0.4
COMMIT_KEEP_MS = 2500  # … kept this long after the way ahead is clear again
# Close to a wall the body cannot pivot: an arc first sweeps the headings next to the one it walks
# on. It turns the way whose first SWEEP_RAD stays clearer, whatever lies beyond (live
# 2026-09-28: 0.44 m from the east wall, facing north-east, it chose the seen-open south and swept
# through east into the wall; the north sweep was clear).
SWEEP_RAD = 0.9
SWEEP_MARGIN_M = 0.05
BUMP_AHEAD_M = 0.12  # a bump: the thing is about here (ahead of the trunk origin)
BRUSH_SIDE_M = 0.13  # a brush along its side: something about this far to that side
TOF_HALF_RAD = 0.39  # the ToF's 45° view in thirds: ±22.5° / 3 per third … its centres at ±15°
# (fix 2026-09-28, headless trap gate: walking fast along a wall it had never looked at, it put
# its head into the slot between the sofa and the wall and stayed wedged there)
# Floor it has not SEEN clear ahead is not assumed clear: with less than this known ahead it
# slows to an arc's speed and looks where it is going (as an animal does in the unknown).
KNOWN_AHEAD_M = 0.7
# An opening narrower than the body can turn in is no way out: a heading's clearance counts at
# most this much more than its neighbours' (±NARROW_RAD) — a slot beside the sofa is a wall.
NARROW_RAD = 0.3
NARROW_EXTRA_M = 0.4
# Cornered: every heading it could sweep an arc through (±CORNER_SWEEP_RAD) blocked within
# this — a forward arc only presses into the walls: it stops (and calls for help, brain.py).
CORNERED_M = 0.42
CORNER_SWEEP_RAD = 1.6
UNDER_BODY_M = 0.1  # a point this near the trunk is under its own body (not a wall ahead)
# Where it was blocked (the duck's own stall, a second contact, pressing into something it cannot
# see — a low box under the ToF's beam, a wall in the camera's blind zone below ~0.2 m): remembered
# for a while as walls just ahead of where it stood, not erased by a floor look that cannot see
# that near (bearing it away from there is the escape's job: unstick.py)
BLOCK_TTL_MS = 30_000
# a floor look that sees clear floor past a blocked place farther than this from the lens clears it
BLOCK_SEEN_M = 0.30
BLOCK_AHEAD_M = 0.14
BLOCK_HALF_M = 0.06
BLOCK_SAME_M = 0.04
MAX_BLOCKS = 24


@dataclass(frozen=True, slots=True)
class Steer:
    """What the boundary memory asks of the walk now (see the module doc)."""

    veer: float = 0.0  # rad/s, + = left
    cap: float | None = None  # m/s, a speed cap; None = none
    urgency: float = 0.0  # 0..1
    side: int = 0  # the freer side: +1 left, −1 right, 0 no difference
    free_ahead: float = FLOOR_RANGE_M  # m, the way clear straight ahead
    best: float = 0.0  # rad, the best clear heading from the trunk now
    look_ahead: bool = False  # the way ahead is not SEEN clear: slow, and look where it goes
    cornered: bool = False  # no arc it could make has room: stop (a forward arc only presses)
    known_ahead: float = 0.0  # m, how far ahead it has seen the floor clear (or walked)


@dataclass(slots=True)
class _Point:
    x: float
    y: float
    w: float
    kind: str  # wall | thing | tof | bump | brush (a side brush's guess: the side it came from, BRUSH_SIDE_M away)


@dataclass(slots=True)
class _Ray:
    """Floor a look saw clear: from (x, y) along ``heading`` (odometry) for ``free`` m."""

    x: float
    y: float
    heading: float
    half: float
    free: float
    w: float


class Boundary:
    """The boundary memory and the steering it gives (module doc)."""

    __slots__ = (
        "points",
        "rays",
        "trail",
        "_poses",
        "_floor_seen",
        "_steer_t",
        "_steer",
        "_t_ms",
        "looks",
        "_commit",
        "_commit_t",
        "blocks",
    )

    def __init__(self) -> None:
        self.points: list[_Point] = []
        self.rays: list[_Ray] = []
        self.trail: deque[tuple[float, float]] = deque(maxlen=TRAIL_KEEP)
        self._poses: deque[tuple[int, float, float, float]] = deque()
        self._floor_seen: int | None = None
        self._steer_t = -1_000_000
        self._steer = Steer()
        self._t_ms: int | None = None
        self.looks = 0
        self._commit = 0  # the side it is bending toward (+1 left, −1 right, 0 none)
        self._commit_t = -1_000_000
        self.blocks: list[tuple[int, float, float]] = []  # (t_ms, x, y): where it was blocked

    @property
    def last(self) -> Steer:
        """The latest steering update."""
        return self._steer

    # ── memory ───────────────────────────────────────────────────────────────

    def update(self, t_ms: int, frame: SensorFrame, *, clock_offset: int = 0) -> None:
        """This tick's senses into the memory (floor line, ToF, a bump) and the fading.
        ``clock_offset``: the brain's clock minus the frame's (a look's time is the frame's)."""
        odom = frame.body.odom
        dt = 0.0 if self._t_ms is None else max(0.0, (t_ms - self._t_ms) / 1000.0)
        self._t_ms = t_ms
        if "set_down" in frame.events:
            # a hand put it down: the odometry knows nothing of the carry, so the walls it remembers are nowhere now (as
            # lastmap's ``lost``); what it sees from here on is remembered afresh
            self.points, self.rays, self.blocks = [], [], []
        if self.blocks:
            # (a place it has walked onto is not blocked)
            self.blocks = [
                b
                for b in self.blocks
                if t_ms - b[0] <= BLOCK_TTL_MS and math.hypot(b[1] - odom[0], b[2] - odom[1]) > 0.05
            ]
        self._poses.append((t_ms, odom[0], odom[1], odom[2]))
        while self._poses and t_ms - self._poses[0][0] > POSE_KEEP_MS:
            self._poses.popleft()
        if dt > 0.0 and (self.points or self.rays):
            fade = math.exp(-dt / MEMORY_TAU_S)
            for p in self.points:
                p.w *= fade
            for r in self.rays:
                r.w *= fade
            self.points = [p for p in self.points if p.w >= KEEP_WEIGHT]
            self.rays = [r for r in self.rays if r.w >= KEEP_WEIGHT]
        if frame.body.posture in ("stand", "walk") and (
            not self.trail
            or math.hypot(odom[0] - self.trail[-1][0], odom[1] - self.trail[-1][1]) >= TRAIL_STEP_M
        ):
            self.trail.append((odom[0], odom[1]))
        cam = frame.camera
        if cam.floor and cam.floor_t_ms is not None and cam.floor_t_ms != self._floor_seen:
            self._floor_seen = cam.floor_t_ms
            self._look(self._pose_at(cam.floor_t_ms + clock_offset), cam.floor)
        tof = frame.tof
        if tof.present and frame.body.posture in ("stand", "walk"):
            for bearing, mm in ((0.26, tof.left_mm), (0.0, tof.ahead_mm), (-0.26, tof.right_mm)):
                if mm is not None and mm < 1200:
                    self._add(odom, bearing, mm / 1000.0 + 0.06, "tof", 0.8)
        if "blocked" in frame.events:
            self._add(odom, 0.0, BUMP_AHEAD_M, "bump", 1.0)
        if "brushed" in frame.events and abs(frame.body.gyro[2]) > 0.3:
            # its side met something (the heading was kicked away from it): a point there
            side = 1.0 if frame.body.gyro[2] < 0.0 else -1.0  # kicked right: it is on the left
            self._add(odom, side * 1.35, BRUSH_SIDE_M, "brush", 0.8)
        if len(self.points) > MAX_POINTS:
            self.points.sort(key=lambda p: p.w)
            del self.points[: len(self.points) - MAX_POINTS]

    def note_block(self, t_ms: int, odom: tuple[float, float, float], bearing: float = 0.0) -> None:
        """It was blocked here, heading ``bearing`` from where it points: three walls just ahead
        (the width of a box), remembered :data:`BLOCK_TTL_MS`."""
        heading = odom[2] + bearing
        c, s = math.cos(heading), math.sin(heading)
        for side in (-BLOCK_HALF_M, 0.0, BLOCK_HALF_M):
            x = odom[0] + BLOCK_AHEAD_M * c - side * s
            y = odom[1] + BLOCK_AHEAD_M * s + side * c
            # the same place again: refresh it, do not pile up
            self.blocks = [b for b in self.blocks if math.hypot(b[1] - x, b[2] - y) > BLOCK_SAME_M]
            self.blocks.append((t_ms, x, y))
        del self.blocks[: max(0, len(self.blocks) - MAX_BLOCKS)]

    def commit_side(self, side: int, t_ms: int) -> None:
        """Hold the bend to ``side`` (+1 left, −1 right): an escape has chosen it."""
        if side in (-1, 1):
            self._commit = side
            self._commit_t = t_ms

    def _pose_at(self, t_ms: int) -> tuple[float, float, float]:
        """The odometry pose at a past moment (the nearest kept; the oldest if older)."""
        best = self._poses[-1] if self._poses else (t_ms, 0.0, 0.0, 0.0)
        for pose in reversed(self._poses):
            if pose[0] <= t_ms:
                best = pose
                break
            best = pose
        return best[1], best[2], best[3]

    def _add(
        self, pose: tuple[float, float, float], bearing: float, dist: float, kind: str, w: float
    ) -> None:
        heading = pose[2] + bearing
        x = pose[0] + dist * math.cos(heading)
        y = pose[1] + dist * math.sin(heading)
        for p in self.points:  # the same place again: refresh it, do not pile up
            if abs(p.x - x) < 0.05 and abs(p.y - y) < 0.05:
                p.w = max(p.w, w)
                return
        self.points.append(_Point(x, y, w, kind))

    def _look(self, pose: tuple[float, float, float], floor: tuple) -> None:
        """One floor line: clear floor carves the memory; each wall/thing end lays a point."""
        self.looks += 1
        px, py, yaw = pose
        for sector in floor:
            heading = yaw + float(sector.bearing)
            half = 0.5 * float(sector.width) + 0.04
            free = float(sector.free_m)
            # the look saw clear floor up to `free` in this wedge: what memory put there is gone
            for p in self.points:
                dx, dy = p.x - px, p.y - py
                d = math.hypot(dx, dy)
                if d < 1e-6 or d > free - 0.08:
                    continue
                if abs(wrap_angle(math.atan2(dy, dx) - heading)) <= half:
                    p.w *= 0.3
            self.rays.append(_Ray(px, py, heading, half, free, 1.0))
            if sector.end in ("wall", "thing") and free < FLOOR_RANGE_M - 0.05:
                for frac in (-0.35, 0.0, 0.35):
                    self._add(
                        (px, py, yaw),
                        float(sector.bearing) + frac * float(sector.width),
                        free,
                        str(sector.end),
                        1.0,
                    )
        self.points = [p for p in self.points if p.w >= KEEP_WEIGHT]
        if self.blocks:
            # a look sees clear floor past a place it was blocked at, far enough from the lens to
            # see (the camera is blind below ~0.2 m): the way is open there after all
            kept = []
            for b in self.blocks:
                d = math.hypot(b[1] - px, b[2] - py)
                heading_b = math.atan2(b[2] - py, b[1] - px)
                cleared = False
                if d > BLOCK_SEEN_M:
                    for sector in floor:
                        if (
                            d <= float(sector.free_m) - 0.08
                            and abs(wrap_angle(heading_b - (yaw + float(sector.bearing))))
                            <= 0.5 * float(sector.width) + 0.04
                        ):
                            cleared = True
                            break
                if not cleared:
                    kept.append(b)
            self.blocks = kept
        if len(self.rays) > 4 * MAX_POINTS:
            del self.rays[: len(self.rays) - 4 * MAX_POINTS]

    # ── steering ─────────────────────────────────────────────────────────────

    def _known(
        self, odom: tuple[float, float, float]
    ) -> tuple[list[tuple[float, float, float]], list[tuple[float, float]]]:
        """(the seen-clear rays from near here: heading from the trunk now, half width, free
        m left; the walked trail in the trunk frame now) — for :meth:`known_free`."""
        x0, y0, yaw = odom
        rays = []
        for r in self.rays:
            if r.w < COUNT_WEIGHT:
                continue
            gap = math.hypot(r.x - x0, r.y - y0)
            if gap <= RAY_NEAR_M and r.free - gap > 0.0:
                rays.append((wrap_angle(r.heading - yaw), r.half + 0.08, r.free - gap))
        c, s = math.cos(yaw), math.sin(yaw)
        trail = []
        for tx, ty in self.trail:
            dx, dy = tx - x0, ty - y0
            trail.append((dx * c + dy * s, -dx * s + dy * c))
        return rays, trail

    def known_free(
        self,
        odom: tuple[float, float, float],
        heading: float,
        known: tuple[list[tuple[float, float, float]], list[tuple[float, float]]] | None = None,
    ) -> float:
        """How far the floor along ``heading`` (rad from the trunk now) is known to be clear:
        seen clear by a look from near here, or walked over (m; 0 = never looked there)."""
        rays, trail = known if known is not None else self._known(odom)
        best = 0.0
        for rel, half, free in rays:
            if free > best and abs(wrap_angle(rel - heading)) <= half:
                best = free
        c, s = math.cos(heading), math.sin(heading)
        for fx, fy in trail:
            along = fx * c + fy * s
            if along > best and abs(-fx * s + fy * c) <= 1.5 * BODY_HALF_M:
                best = along
        return best

    def clearance(
        self,
        odom: tuple[float, float, float],
        heading: float,
        *,
        goal: tuple[float, float] | None = None,
        rel: list[tuple[float, float, float]] | None = None,
        half: float = BODY_HALF_M,
    ) -> float:
        """How far the way is clear along ``heading`` (rad from the trunk now, + = left) in a
        corridor the body fits through (``half`` m either side; m, ≤ FLOOR_RANGE_M)."""
        if rel is None:
            rel = self._relative(odom, goal)
        c, s = math.cos(heading), math.sin(heading)
        best = FLOOR_RANGE_M
        for fx, fy, _w in rel:
            along = fx * c + fy * s
            if along <= -0.02:
                continue
            across = -fx * s + fy * c
            if abs(across) <= half:
                best = min(best, max(0.0, along))
        return best

    def snapshot(
        self, centre: tuple[float, float], radius: float = 4.2
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """What the memory holds, for a route search (``clutter_route``), in the odometry frame: (the points in the way — the counted
        wall / thing / ToF / bump points and the places it was blocked —, the ground known passable — the trail it walked and the
        floor a look saw clear, a point every 0.1 m along each ray), those within ``radius`` of ``centre``."""
        cx, cy = centre

        def near(x: float, y: float) -> bool:
            return abs(x - cx) <= radius and abs(y - cy) <= radius

        obstacles = [(p.x, p.y) for p in self.points if p.w >= COUNT_WEIGHT and near(p.x, p.y)]
        obstacles += [(bx, by) for _t, bx, by in self.blocks if near(bx, by)]
        known = [t for t in self.trail if near(t[0], t[1])]
        for r in self.rays:
            if r.w < COUNT_WEIGHT:
                continue
            c, s = math.cos(r.heading), math.sin(r.heading)
            for k in range(1, int(r.free / 0.1) + 1):
                x, y = r.x + 0.1 * k * c, r.y + 0.1 * k * s
                if near(x, y):
                    known.append((x, y))
        return obstacles, known

    def clearances(
        self,
        odom: tuple[float, float, float],
        headings: list[float],
        *,
        half: float = BODY_HALF_M,
        brushes: bool = True,
    ) -> list[float]:
        """:meth:`clearance` along each of ``headings`` (rad from the trunk now), the points worked out once
        (``brushes=False``: without the side brushes' guesses)."""
        rel = self._relative(odom, None, brushes=brushes)
        return [self.clearance(odom, h, rel=rel, half=half) for h in headings]

    def ahead(
        self,
        odom: tuple[float, float, float],
        *,
        goal: tuple[float, float] | None = None,
        rel: list[tuple[float, float, float]] | None = None,
    ) -> float:
        """The way clear ahead: the nearest clearance over the ±20° cone (m)."""
        if rel is None:
            rel = self._relative(odom, goal)
        return min(self.clearance(odom, h, rel=rel) for h in CONE)

    def _relative(
        self,
        odom: tuple[float, float, float],
        goal: tuple[float, float] | None,
        *,
        walls_only: bool = False,
        clear: float = GOAL_CLEAR_M,
        brushes: bool = True,
    ) -> list[tuple[float, float, float]]:
        """The counted points in the trunk frame now (x ahead, y left), the goal's excluded
        (``walls_only``: walls, furniture, ToF, bump and brush points — not loose things — and none
        under its own body, where nothing solid can be; ``brushes=False``: not the brushes' guesses)."""
        x0, y0, yaw = odom
        c, s = math.cos(yaw), math.sin(yaw)
        out = []
        for p in self.points:
            if p.w < COUNT_WEIGHT:
                continue
            if goal is not None and math.hypot(p.x - goal[0], p.y - goal[1]) < clear:
                continue
            dx, dy = p.x - x0, p.y - y0
            if math.hypot(dx, dy) < UNDER_BODY_M:
                continue  # (under its own body: nothing solid can be ahead there)
            if (walls_only and p.kind == "thing") or (not brushes and p.kind == "brush"):
                continue
            out.append((dx * c + dy * s, -dx * s + dy * c, p.w))
        if not walls_only:
            # (where it was blocked steers it; it is not a wall it has SEEN: it does not make the
            # duck "cornered", which stops it for good — measured: a sun patch beside the wall)
            for _t, bx, by in self.blocks:
                if goal is not None and math.hypot(bx - goal[0], by - goal[1]) < clear:
                    continue
                dx, dy = bx - x0, by - y0
                out.append((dx * c + dy * s, -dx * s + dy * c, 1.0))
        return out

    def steer(
        self,
        t_ms: int,
        odom: tuple[float, float, float],
        *,
        goal_bearing: float | None = None,
        goal_range: float | None = None,
        goal_clear: float = GOAL_CLEAR_M,
    ) -> Steer:
        """The walk's bend, cap and urgency now (updated ≤ STEER_HZ; held in between)."""
        if t_ms - self._steer_t < 1000.0 / STEER_HZ:
            return self._steer
        self._steer_t = t_ms
        goal = None
        if goal_bearing is not None and goal_range is not None:
            heading = odom[2] + goal_bearing
            goal = (
                odom[0] + goal_range * math.cos(heading),
                odom[1] + goal_range * math.sin(heading),
            )
        rel = self._relative(odom, goal, clear=goal_clear)
        if self._commit and t_ms - self._commit_t > COMMIT_KEEP_MS:
            self._commit = 0
        known = self._known(odom)
        # How far ahead it has SEEN the floor clear (a look from near here, or its own trail).
        known_ahead = min(self.known_free(odom, h, known) for h in CONE[1:4])
        if goal_range is not None:
            known_ahead = max(known_ahead, min(goal_range, FLOOR_RANGE_M))  # it sees its goal
        free_ahead = self.ahead(odom, rel=rel)
        # A known wall shortens the corridor the duck needs to inspect: it already knows the
        # floor up to the obstacle, and steering below still bends around/caps that obstacle.
        # Keep looking if there is no positive floor evidence or no usable corridor; a ToF/bump
        # return by itself must never count as a floor look.
        needed_ahead = min(KNOWN_AHEAD_M, free_ahead)
        unseen = known_ahead <= 0.0 or needed_ahead <= 0.0 or known_ahead < needed_ahead
        if not rel:
            self._steer = Steer(
                cap=APPROACH_CAP if unseen else None,
                look_ahead=unseen,
                known_ahead=round(known_ahead, 2),
            )
            return self._steer
        scores = []
        seen_clear: dict[float, float] = {}  # heading → the clearance the walls it saw leave
        for h in FAN:
            seen_clear[h] = self.clearance(odom, h, rel=rel)
        for h in FAN:
            walls = seen_clear[h]
            # an opening narrower than the body can turn in is no way out (a slot is a wall)
            side = min(
                seen_clear.get(round(h - NARROW_RAD, 2), walls),
                seen_clear.get(round(h + NARROW_RAD, 2), walls),
            )
            walls = min(walls, side + NARROW_EXTRA_M)
            clear = min(walls, max(UNKNOWN_M, self.known_free(odom, h, known)))
            cost = TURN_COST * abs(h) - (RIGHT_BIAS_M if h < 0.0 else 0.0)
            if goal_bearing is not None:
                cost += GOAL_COST * abs(wrap_angle(h - goal_bearing))
            if self._commit and h * self._commit < -0.1:
                cost += COMMIT_M  # the side it is already bending toward keeps winning ties
            scores.append((min(clear, FLOOR_RANGE_M) - cost, h, clear))
        _score, best, _best_clear = max(scores)
        # cornered: walls (not a loose thing it could go round or nudge) leave no arc room
        wall_pts = self._relative(odom, goal, walls_only=True, clear=goal_clear)
        room = max(
            (self.clearance(odom, h, rel=wall_pts) for h in FAN if abs(h) <= CORNER_SWEEP_RAD),
            default=FLOOR_RANGE_M,
        )
        cornered = bool(wall_pts) and free_ahead < STOP_M and room < CORNERED_M
        if free_ahead < NEAR_M:
            # the arc's first sweep decides the way (SWEEP_RAD): the side whose next headings the
            # walls leave clearer — the commitment wins near-ties (no dithering)
            sweep = {
                way: min(c for h, c in seen_clear.items() if 0.0 < way * h <= SWEEP_RAD)
                for way in (1, -1)
            }
            way = 1 if sweep[1] >= sweep[-1] else -1
            if abs(sweep[1] - sweep[-1]) <= SWEEP_MARGIN_M or (
                self._commit and sweep[self._commit] >= sweep[-self._commit] - 2 * SWEEP_MARGIN_M
            ):
                way = self._commit or (1 if best > 0.0 else -1 if best < 0.0 else way)
            if best * way <= 0.0 or abs(best) < 0.1:
                best = max((sc, h) for sc, h, _c in scores if h * way > 0.1)[1]
        left = max(c for _s, h, c in scores if h > 0.3)
        right = max(c for _s, h, c in scores if h < -0.3)
        side = 1 if left > right + 0.1 + RIGHT_BIAS_M else (-1 if right + 0.1 > left else 0)
        if free_ahead >= SLOW_M:
            self._steer = Steer(
                free_ahead=free_ahead,
                best=best,
                side=side,
                cap=APPROACH_CAP if unseen or free_ahead < KNOWN_AHEAD_M else None,
                look_ahead=unseen,
                known_ahead=round(known_ahead, 2),
            )
            return self._steer
        urgency = min(1.0, max(0.0, (SLOW_M - free_ahead) / (SLOW_M - STOP_M)))
        if abs(best) < 0.1:
            best = 0.35 * (self._commit or side or 1) if free_ahead < NEAR_M else best
        if abs(best) >= 0.1:
            self._commit = 1 if best > 0.0 else -1
            self._commit_t = t_ms
        veer = max(-1.0, min(1.0, VEER_GAIN * best * (0.35 + 0.65 * urgency)))
        if free_ahead >= NEAR_M:
            # still well away: a gentle bend (not yet an arc — its speed stays its own)
            veer = max(-GENTLE_VEER, min(GENTLE_VEER, veer))
        cap = APPROACH_CAP if unseen or free_ahead < KNOWN_AHEAD_M else None
        if free_ahead < STOP_M:
            cap = 0.35
            veer = math.copysign(max(0.8, abs(veer)), veer if veer else float(side or 1))
        elif free_ahead < NEAR_M:
            cap = 0.40
        self._steer = Steer(
            veer=round(veer, 3),
            cap=cap,
            urgency=round(urgency, 3),
            side=side,
            free_ahead=round(free_ahead, 3),
            best=best,
            look_ahead=unseen,
            cornered=cornered,
            known_ahead=round(known_ahead, 2),
        )
        return self._steer

    def view(self, odom: tuple[float, float, float]) -> dict[str, float | int]:
        """For the telemetry: how many points it remembers, the way ahead, the looks."""
        return {
            "points": len(self.points),
            "looks": self.looks,
            "free_ahead": round(self.ahead(odom), 2),
            "urgency": self._steer.urgency,
            "veer": self._steer.veer,
        }


__all__ = ["Boundary", "Steer"]
