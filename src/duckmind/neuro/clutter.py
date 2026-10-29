"""Getting out of clutter: pick a route and keep to it (owner 2026-10-11).

Owner's words: 「東西稍微密集環境下，小鴨沒有辦法找到一條路徑到達想要去的位置，會一直鬼打牆前進後退轉彎，這必須想一個辦法，讓小鴨可以沿著某個路線出去，即便些微碰撞也合理，致跌倒也算在合理範圍」.

Between chairs, bins and a sofa's corner the duck's layers each answer one thing — a reflex stop, a back arc, a turn-around, a walk
toward what it wants, the wall memory's bend — and each answer undoes the last: forward, back, turn, forward, for minutes at one
place (the green tangle of the owner's screenshot). ``unstick.py`` gives one escape per contact (a small reverse arc, then a turn) and
the next behaviour walks it straight back in.

This module is the animal's way out of a thicket: when it notices that it is going round in one place, it stops deciding step by
step. It plans ONE route and keeps to it until it is out.

* **Dithering**, from what the duck itself knows: contacts (a bump, a ToF touch, a brush), unstick escapes and back arcs are the
  events; at least :attr:`ClutterConfig.knot_events` of them within :attr:`knot_window_ms`, all within :attr:`knot_radius_m` of
  where it is now, is a knot. A second way: it has asked to walk for much of :attr:`confine_window_ms` and has not left a
  :attr:`confine_radius_m` circle, with a couple of events in it.
* **The route** (``clutter_route.py``): the cheapest way over what it remembers — the wall memory's points are in the way, the ground it
  walked and the floor it saw clear are passable and cheap, the ground nobody has looked at is a guess and dearer — to a place
  :attr:`out_m` or more from the knot's centre, and among those the one nearest to where it was walking to (when it had somewhere to
  go). The way it came in is the cheapest known way out; with nothing known the open floor it has seen is.
* **Commit**: the route is kept while it stays passable; a new one replaces it only when it is clearly cheaper (the memory grew), the
  walk follows it a short look-ahead at a time. Arcs only (no turn on the spot): a forward arc when the way ahead has room, a reverse
  arc (the body's backing arc) when it has not, then a straight walk on the heading; a duck with the quick turn pivots on the spot. A
  stall (asked to walk, no way made) puts a blocked place there, a reverse arc and a new plan; after :attr:`max_s` it gives the place up
  for :attr:`cooldown_s` (the older layers have it back).
* **Light touches are part of it**: the escape does not abandon the route on a bump or a brush; the reflexes' stops still hold the legs
  for their half second and the walk goes on after. What always ends it: a fall, a lift, a drop, the battery, a startle, a flight,
  sleep (the brain's call; this module only reads ``active``).

Pure and clock-injected (``t_ms``); deterministic; honest senses only (the odometry, the wall memory, the ToF, the contacts).
``ClutterConfig(enabled=False)`` makes :meth:`Clutter.step` return None always.
"""

from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass
from typing import Any

from duckmind.neuro import clutter_route as CR
from duckmind.neuro.contract import wrap_angle

# the setting: OFF unless asked for (``NeuroBrain(clutter=True)``, or ``DUCKMIND_CLUTTER=1`` in the environment for a run). Measured
# 2026-10-11 (docs/design/neural-duck/CLUTTER-ESCAPE-2026-10-11.md): neither answer shows a gain that holds up, so the duck does not use it.
CLUTTER_ENABLED = os.environ.get("DUCKMIND_CLUTTER", "0") == "1"
FAN_STEP = math.radians(15.0)
FAN_N = 24


@dataclass(frozen=True, slots=True)
class ClutterConfig:
    enabled: bool = True
    # ── noticing it ──────────────────────────────────────────────────────────
    knot_window_ms: int = 40_000
    knot_radius_m: float = 0.9
    knot_events: int = 4
    event_gap_ms: int = 1500  # events closer than this are one
    confine_window_ms: int = 30_000
    confine_radius_m: float = 0.6
    confine_want: float = 0.4  # the share of the window it asked to walk
    confine_events: int = 2
    # the wide knot (the owner's tangle was a few metres across: loops round a table, in and out of the same gaps): events keep adding
    # up inside a bigger circle over a longer time
    wide_window_ms: int = 90_000
    wide_radius_m: float = 1.5
    wide_events: int = 6
    wide_want: float = 0.4
    wide_out_m: float = 1.8
    # ── the way out ──────────────────────────────────────────────────────────
    out_m: float = 1.0  # out: this far from the knot's centre …
    out_open_m: float = 0.7  # … with this much way ahead (wall memory)
    max_s: float = 30.0
    cooldown_s: float = 45.0
    replan_s: float = 1.0
    block_radii: tuple[float, ...] = (
        0.22,
        0.16,
        0.12,
    )  # the body's room, roomiest first (the soft map plans once: its margin is a cost)
    soft_follow: bool = False  # the follower goes on through a brush: no reverse arc for geometry alone, a stall is no ground made (turning counts)
    keep_ratio: float = 0.75  # a new route replaces the kept one only when it costs less than this share of what the kept one still does
    flip_ratio: float = 0.5  # … and one that sets out the other way (more than 1.2 rad off) must cost less than this share
    look_m: float = 0.5  # the walk aims at the route this far ahead
    goal_w: float = 3.0  # per metre it still is from where it was walking to (the reachable place nearest it is chosen when it cannot be reached)
    goal_range_m: float = 3.0  # (a goal seen only as a bearing is taken this far)
    goal_open_m: float = 3.0  # out, with a goal: the way straight to it clear this far (or all the way, when it is nearer)
    use_trail: bool = (
        False  # first choice: the way it came (its own trail, back to a place out of the knot)
    )
    use_plan: bool = True  # the route search (False: only the open-floor fan)
    action: str = "loosen"  # what a knot is answered with: "route" (it takes the legs) | "loosen" (the pull of what it wants is let go for a while)
    # ── the lasting map (lastmap.py): the route is planned over what the duck has learnt of the place over hours, not over the 25 s
    # wall memory; and a walk to a place the map says is behind something is a route too (no knot needed first)
    use_map: bool = False
    map_goal: bool = (
        False  # a goal walk whose straight line the map says is shut begins a route to the goal
    )
    map_goal_min_m: float = (
        0.8  # (a goal nearer than this is the docking's / the beak's own business)
    )
    map_goal_max_m: float = 5.0
    map_block_s: float = 1.5  # the straight line shut this long before it begins
    map_clear_s: float = 1.0  # … and open this long before it ends
    map_line_r: float = 0.16  # the room the straight line needs (half the corridor)
    map_goal_events: int = 0  # a map route to a goal also needs this many contacts / holds within 30 s and 1.2 m (0: the shut line alone)
    map_goal_stall_s: float = 0.0  # a map route to a goal also needs the walk to have got this long without coming nearer (0: not needed)
    map_goal_progress_m: float = 0.3  # … "nearer" meaning at least this much
    map_goal_skip_m: float = (
        0.5  # the last stretch to the goal is not looked at (a dock on a wall, a toy among toys)
    )
    map_back_known: bool = False  # with the map, a reverse arc needs ground the map KNOWS to be clear behind (not just nothing seen)
    back_known_need_m: float = (
        0.20  # … at least this much known clear ground behind to back off at all
    )
    back_known_m: float = 0.30  # … this far (the arc's own length is ~0.45 m; the tail is 0.07 m behind the trunk's origin)
    max_tries: int = 0  # a route that has failed this many times (a stall or a shut line) is given up: the older layers and the link's wall pivot take over (0: never)
    map_max_s: float = 300.0  # a detour across a home is several metres at the duck's walking pace
    waypoint_range_m: float = 0.8  # action "waypoint": the point the walk is told to head for is at least this far (a nearer one reads as 'arrived')
    loosen_s: float = 16.0  # … at most this long,
    loosen_out_m: float = 1.2  # … and (when set) only until it is this far from where the knot was (but at least loosen_min_s)
    loosen_min_s: float = 3.0
    use_goal: bool = True  # the route is to where it was going (False: only out of the knot)
    goal_there_m: float = 0.5
    goal_max_s: float = 60.0  # with somewhere to go the escape may last this long (it is a route to the place, not just out)
    fail_r: float = 0.2  # a stall puts a blocked place this big …
    fail_ahead_m: float = 0.28  # … this far ahead of the trunk
    fail_ttl_ms: int = 45_000
    # ── driving ──────────────────────────────────────────────────────────────
    go_vx: float = 0.34
    arc_vx: float = 0.37  # (the body's arc speed floor is 0.35)
    back_vx: float = -0.40
    arc_vyaw: float = 1.0
    go_gain: float = 1.4
    go_vyaw_max: float = 0.30  # a straight walk is |vyaw| <= 0.3
    aligned_rad: float = 0.30
    turn_back_ms: int = 1200
    turn_back_rad: float = (
        0.9  # a thing right ahead and the nose this far off the heading: a reverse arc
    )
    pivot_rad: float = 0.9  # (a duck with the quick turn pivots from this far off)
    arc_room_m: float = 0.45  # a forward arc needs this much way ahead
    back_room_m: float = (
        0.10  # a reverse arc needs this much behind (the tail is 7 cm behind the trunk's origin)
    )
    jam_ms: int = 1200  # held for most of this long, or the nose on it: jammed — a forward arc only presses; it backs off first
    jam_held: float = 0.4
    stall_ms: int = 2500
    stall_m: float = 0.06
    stall_held: float = 0.5  # … or the legs held for this share of stall_ms (above 1: never; the ground made is the only stall)
    stall_yaw: float = (
        0.0  # turning this much (rad) in stall_ms is progress too (0: only ground made counts)
    )
    block_m: float = 0.20  # the way along the heading closed this near for block_ms: a blocked place there, a new plan
    block_ms: int = 800
    weave_near_m: float = 0.42  # the way ahead this short in a straight walk: an arc (the ToF stop holds straight walks only) …
    weave_vyaw: float = 0.6  # … of this much
    # ── the fan, when there is no route (nothing known): the open floor ──────
    clear_cap_m: float = 1.5
    unknown_m: float = 0.35


DEFAULT = ClutterConfig()


@dataclass(frozen=True, slots=True)
class Drive:
    """What the escape asks of the legs this tick (``back``: a reverse arc)."""

    vx: float
    vyaw: float
    back: bool
    phase: str  # turn | back | go
    heading: float  # rad, the way it is going (odometry frame)
    pivot: float = 0.0  # +1 / -1: a duck with the quick turn pivots on the spot to this side (the arc above is what the others walk)


class Clutter:
    """The duck's way out of a thicket (module doc). One :meth:`step` per brain tick."""

    __slots__ = (
        "cfg",
        "_events",
        "_hist",
        "_hist_t",
        "_wide",
        "_wide_t",
        "_out_m",
        "_cool_until",
        "_discs",
        "mode",
        "heading",
        "origin",
        "t_start",
        "route",
        "_replan_ms",
        "_centre",
        "_tries",
        "_phase_until",
        "_phase",
        "_go_ref",
        "_go_yaw",
        "_go_t",
        "_block_since",
        "_weave",
        "_held",
        "_fan_set",
        "_held_prev",
        "_loosen_until",
        "_loosen_from",
        "loosened",
        "_trail0",
        "_failed_h",
        "started",
        "finished",
        "gave_up",
        "replans",
        "dbg",
        "dbg_goal",
        "_lmap",
        "_mb_since",
        "_mc_since",
        "_mb_t",
        "why",
        "map_started",
        "dbg_plan",
        "_rng_hist",
        "dbg_events",
        "waypoint",
        "quiet",
    )

    def __init__(self, cfg: ClutterConfig = DEFAULT) -> None:
        self.cfg = cfg
        self._events: deque[tuple[int, float, float, str]] = deque(maxlen=256)
        self._hist: deque[tuple[int, float, float, bool]] = deque()
        self._hist_t = -1_000_000
        self._wide: deque[tuple[int, float, float, bool]] = deque()
        self._wide_t = -1_000_000
        self._out_m = self.cfg.out_m
        self._cool_until = -1
        self._discs: list[
            tuple[int, float, float, float]
        ] = []  # (t_ms, x, y, r): a place a try failed at
        self.mode = "idle"
        self.heading = 0.0
        self.origin = (0.0, 0.0)
        self.t_start = 0
        self.route: CR.Route | None = None
        self._replan_ms = 0
        self._centre = (0.0, 0.0)
        self._tries = 0
        self._phase = "turn"
        self._phase_until = 0
        self._go_ref: tuple[float, float] | None = None
        self._go_yaw = 0.0
        self._go_t = 0
        self._block_since: int | None = None
        self._weave = 1.0
        self._fan_set = False
        self._held_prev = False
        self._loosen_until = -1
        self._loosen_from = -1
        self.loosened = 0
        self._trail0: list[
            tuple[float, float]
        ] = []  # its trail as it was when the escape began (the retrace walks a fixed path)
        self._failed_h: list[
            tuple[int, float, float, float]
        ] = []  # (t_ms, x, y, heading) of the fan's headings that failed
        self._held: deque[tuple[int, bool]] = deque()
        self.started = 0  # escapes begun (telemetry)
        self.finished = 0  # … ended out
        self.gave_up = 0  # … ended by time
        self.replans = 0  # routes adopted (telemetry)
        self.dbg_goal: tuple[float, float | None] | None = None
        self._lmap: Any = None  # the lasting map, when the brain gives one (``step``)
        self._mb_since: int | None = (
            None  # since when the map has said the straight line to the goal is shut
        )
        self._mc_since: int | None = None  # … and open (an escape on)
        self._mb_t = -1_000_000
        self.why = ""
        self.quiet = False  # the brain says: not now (playing with, eating, looking at a thing): no events are kept, no route
        self.waypoint: tuple[float, float] | None = (
            None  # (bearing from the trunk, range m) of the route's look-ahead point, while following
        )
        self._rng_hist: deque[tuple[int, float]] = (
            deque()
        )  # (t_ms, distance to the goal point) while a goal walk is asked
        self.dbg_events: list[
            tuple
        ] = []  # (t_ms, what, ...) of the last events of an escape (a development aid; bounded)
        self.dbg_plan: list[
            tuple
        ] = []  # (block_r, route points, cost, complete) of each radius tried in the last plan
        self.map_started = (
            0  # escapes begun because the map said the way to the goal is shut (telemetry)
        )
        self.dbg: list[
            tuple[float, float, float, float]
        ] = []  # the last fan (heading, score, clearance, known), when there was no route

    # ── noticing it ──────────────────────────────────────────────────────────

    @property
    def active(self) -> bool:
        """The escape has the legs (the older layers' own back arc and turn are dropped)."""
        return self.mode == "escape" and self.cfg.action != "waypoint"

    @property
    def following(self) -> bool:
        """``action="waypoint"``: a route is on, and the walk is only told where to head (:attr:`waypoint`); the legs stay the brain's."""
        return self.mode == "escape" and self.cfg.action == "waypoint"

    def note(self, t_ms: int, x: float, y: float, kind: str) -> None:
        """An event: a contact (``bump``), an unstick escape (``escape``), a back arc (``back``)."""
        if not self.cfg.enabled or self.quiet:
            return  # (off: nothing is kept; quiet: a contact with the thing it came to see is no knot)
        if self._events and t_ms - self._events[-1][0] < self.cfg.event_gap_ms:
            return
        self._events.append((t_ms, x, y, kind))

    def loosening(self, t_ms: int, odom: tuple[float, float, float] | None = None) -> bool:
        """The goal's pull is let go (``action="loosen"``): the walk goes where the open floor is, not where it wanted to go."""
        if t_ms >= self._loosen_until:
            return False
        cfg = self.cfg
        if (
            cfg.loosen_out_m > 0.0
            and odom is not None
            and t_ms - self._loosen_from >= int(cfg.loosen_min_s * 1000)
            and math.hypot(odom[0] - self.origin[0], odom[1] - self.origin[1]) >= cfg.loosen_out_m
        ):
            self._loosen_until = t_ms  # (it is out: the pull is back)
            return False
        return True

    def reset(self, t_ms: int, *, cooldown: bool = False) -> None:
        """Ends any escape (a fall, a lift, a startle...); the knot's events are forgotten."""
        self.mode = "idle"
        self._events.clear()
        self._hist.clear()
        self._wide.clear()
        if cooldown:
            self._cool_until = t_ms + int(self.cfg.cooldown_s * 1000)

    def _recent(self, t_ms: int, x: float, y: float, window_ms: int, radius: float) -> int:
        while self._events and t_ms - self._events[0][0] > max(window_ms, self.cfg.wide_window_ms):
            self._events.popleft()
        return sum(
            1
            for t, ex, ey, _k in self._events
            if t_ms - t <= window_ms and math.hypot(ex - x, ey - y) <= radius
        )

    def _knotted(self, t_ms: int, x: float, y: float, want: bool) -> str | None:
        cfg = self.cfg
        if self._recent(t_ms, x, y, cfg.knot_window_ms, cfg.knot_radius_m) >= cfg.knot_events:
            return "knot"
        if t_ms - self._wide_t >= 1000:
            self._wide_t = t_ms
            self._wide.append((t_ms, x, y, want))
        while self._wide and t_ms - self._wide[0][0] > cfg.wide_window_ms:
            self._wide.popleft()
        if (
            len(self._wide) >= 0.9 * cfg.wide_window_ms / 1000
            and t_ms - self._wide[0][0] >= 0.9 * cfg.wide_window_ms
        ):
            if sum(1 for w in self._wide if w[3]) >= cfg.wide_want * len(self._wide):
                wx = sum(w[1] for w in self._wide) / len(self._wide)
                wy = sum(w[2] for w in self._wide) / len(self._wide)
                if (
                    max(math.hypot(w[1] - wx, w[2] - wy) for w in self._wide) <= cfg.wide_radius_m
                    and self._recent(t_ms, wx, wy, cfg.wide_window_ms, cfg.wide_radius_m)
                    >= cfg.wide_events
                ):
                    return "wide"
        if t_ms - self._hist_t >= 500:
            self._hist_t = t_ms
            self._hist.append((t_ms, x, y, want))
        while self._hist and t_ms - self._hist[0][0] > cfg.confine_window_ms:
            self._hist.popleft()
        if (
            len(self._hist) >= 0.9 * cfg.confine_window_ms / 500
            and t_ms - self._hist[0][0] >= 0.9 * cfg.confine_window_ms
        ):
            wanted = sum(1 for h in self._hist if h[3]) / len(self._hist)
            if wanted >= cfg.confine_want:
                cx = sum(h[1] for h in self._hist) / len(self._hist)
                cy = sum(h[2] for h in self._hist) / len(self._hist)
                if (
                    max(math.hypot(h[1] - cx, h[2] - cy) for h in self._hist)
                    <= cfg.confine_radius_m
                    and self._recent(t_ms, x, y, cfg.confine_window_ms, cfg.confine_radius_m)
                    >= cfg.confine_events
                ):
                    return "confined"
        return None

    # ── the way out ──────────────────────────────────────────────────────────

    def _goal_world(
        self, odom: tuple[float, float, float], goal: tuple[float, float | None] | None
    ) -> tuple[float, float] | None:
        if goal is None:
            return None
        bearing, rng = goal
        r = self.cfg.goal_range_m if rng is None else min(float(rng), 6.0)
        return odom[0] + r * math.cos(odom[2] + bearing), odom[1] + r * math.sin(odom[2] + bearing)

    def _trail_route(self, x: float, y: float) -> CR.Route | None:
        """The way it came: its own trail (every 0.15 m of ground it walked) read backwards from here to the first place at least
        ``out_m`` + a hand from the knot's centre — passable by construction, drift-proof over the few metres of it that count."""
        trail = self._trail0
        if len(trail) < 4:
            return None
        pts: list[tuple[float, float]] = [(x, y)]
        far = None
        for tx, ty in reversed(trail):
            if math.hypot(tx - pts[-1][0], ty - pts[-1][1]) < 0.28:
                continue
            pts.append((tx, ty))
            if math.hypot(tx - self.origin[0], ty - self.origin[1]) >= self._out_m + 0.25:
                far = (tx, ty)
                break
        if far is None or len(pts) < 3:
            return None
        return CR.Route(tuple(pts), float(len(pts)), far, True)

    def _window(self, x: float, y: float, goal: tuple[float, float] | None) -> tuple[float, float]:
        """The centre of the route window: here, pulled toward where it is going while both stay inside (1 m in from the edge)."""
        half = (CR.N / 2) * CR.RES - 1.0
        if goal is None:
            return x, y
        gx, gy = goal[0] - x, goal[1] - y
        k = min(1.0, 2.0 * half / max(1e-9, max(abs(gx), abs(gy)))) * 0.5
        cx, cy = x + gx * k, y + gy * k
        return (
            min(x + half, max(x - half, cx)),
            min(y + half, max(y - half, cy)),
        )

    def _plan(
        self,
        t_ms: int,
        odom: tuple[float, float, float],
        bound,
        goal: tuple[float, float | None] | None,
    ) -> None:
        """A route over what it remembers (module doc); the kept one unless the new one is clearly cheaper."""
        cfg = self.cfg
        x, y, _yaw = odom
        self._replan_ms = t_ms + int(cfg.replan_s * 1000)
        if cfg.use_trail:
            if self.route is not None and getattr(self.route, "complete", False) and self._trail0:
                return  # (the retrace is one path: kept)
            route = self._trail_route(x, y)
            if route is not None:
                self.route = route
                self.replans += 1
                return
        if not cfg.use_plan:
            return
        self._discs = [d for d in self._discs if t_ms - d[0] <= cfg.fail_ttl_ms]
        gw = self._goal_world(odom, goal)
        # the window stays where the escape put it (so the maps and the routes do not slide about under the walk): round the knot, a
        # little toward where it is going; it is moved when the duck comes within a metre of its edge
        half = (CR.N / 2) * CR.RES
        if abs(x - self._centre[0]) > half - 1.0 or abs(y - self._centre[1]) > half - 1.0:
            self._centre = self._window(x, y, gw)
        lmap = self._lmap if cfg.use_map else None
        if lmap is None:
            obstacles, known = bound.snapshot(self._centre, half + 0.4)
        new = None
        maps = None
        self.dbg_plan = []
        radii = cfg.block_radii[-1:] if lmap is not None and lmap.cfg.soft else cfg.block_radii
        for block_r in radii:
            # (the roomiest way first; a tight home has gaps only the narrower radii let through; the soft map has no ring to widen)
            if lmap is not None:
                maps = lmap.maps(
                    (x, y),
                    self._centre,
                    block_r=block_r,
                    goal=gw,
                    discs=[(d[1], d[2], d[3]) for d in self._discs],
                )
            else:
                maps = CR.build(
                    (x, y),
                    obstacles,
                    [(d[1], d[2], d[3]) for d in self._discs],
                    known,
                    centre=self._centre,
                    goal=gw,
                    block_r=block_r,
                )
            new = CR.search(maps, (x, y), self.origin, self._out_m, gw, cfg.goal_w)
            self.dbg_plan.append(
                (
                    block_r,
                    None if new is None else len(new.points),
                    None if new is None else round(new.cost, 1),
                    None if new is None else new.complete,
                )
            )
            if new is not None and (new.complete or gw is None):
                break
        assert maps is not None
        old = self.route
        if old is not None and len(old.points) > 1 and old.complete:
            # (a kept route that is the way all the way is kept unless the new one is clearly cheaper; one that only got as far as the
            # memory and the window go is made again from where it has got to)
            near, _la = CR.lookahead(old.points, (x, y), 0.0)
            kept = CR.route_cost(maps, old.points, near)
            if new is None:
                if kept == float("inf"):
                    self.route = None
                return
            if kept != float("inf"):
                ratio = cfg.keep_ratio
                # (a new route that sets out the other way than the kept one is made to be much cheaper: no flipping about)
                if (
                    len(new.points) > 1
                    and abs(wrap_angle(self._first_leg(new, x, y) - self._first_leg(old, x, y)))
                    > 1.2
                ):
                    ratio = cfg.flip_ratio
                if new.cost >= ratio * kept:
                    return
        if new is not None:
            self.route = new
            self.replans += 1
        elif old is not None and CR.route_cost(maps, old.points, 0) == float("inf"):
            self.route = None

    def _first_leg(self, route: CR.Route, x: float, y: float) -> float:
        """The direction a route sets out in from here."""
        _near, la = CR.lookahead(route.points, (x, y), self.cfg.look_m)
        return math.atan2(la[1] - y, la[0] - x)

    def _heading_now(
        self, odom: tuple[float, float, float], bound, t_ms: int, *, rechoose: bool = False
    ) -> None:
        """The heading to walk: toward the look-ahead point of the route; with no route, the one the open-floor fan chose (kept: it is
        chosen again only when ``rechoose`` — at the start and when a try has failed — never from tick to tick)."""
        cfg = self.cfg
        x, y, _yaw = odom
        if self.route is not None and len(self.route.points) > 1:
            _near, la = CR.lookahead(self.route.points, (x, y), cfg.look_m)
            if math.hypot(la[0] - x, la[1] - y) > 0.08:
                self.heading = math.atan2(la[1] - y, la[0] - x)
                self._fan_set = False
                return
        if rechoose or not self._fan_set:
            self.heading = self._fan(odom, bound, t_ms)
            self._fan_set = True

    def _fan(self, odom: tuple[float, float, float], bound, _t_now: int) -> float:
        """No route: the heading (odometry frame) with the most open floor — the wall memory's clearance, as far as it is known and a
        little beyond, less a little for turning far — the one it had kept unless another is clearly better."""
        cfg = self.cfg
        _x, _y, yaw = odom
        rel = [wrap_angle(k * FAN_STEP - math.pi - yaw) for k in range(FAN_N)]
        clears = bound.clearances(odom, rel)
        known = bound._known(odom)
        best_h, best_s = self.heading, -1e9
        keep_s = None
        dbg = []
        for h_rel, c in zip(rel, clears, strict=True):
            h_abs = wrap_angle(yaw + h_rel)
            kf = bound.known_free(odom, h_rel, known)
            s = min(c, cfg.clear_cap_m, kf + cfg.unknown_m) - 0.12 * abs(h_rel)
            if c < 0.3:
                s -= 1.0
            for ft, fx, fy, fh in self._failed_h:
                if (
                    _t_now - ft <= cfg.fail_ttl_ms
                    and math.hypot(fx - _x, fy - _y) <= 1.2
                    and abs(wrap_angle(fh - h_abs)) <= 0.5
                ):
                    s -= 1.6
            dbg.append((round(h_abs, 2), round(s, 2), round(c, 2), round(kf, 2)))
            if abs(wrap_angle(h_abs - self.heading)) <= 0.5 * FAN_STEP + 1e-6:
                keep_s = s
            if s > best_s:
                best_h, best_s = h_abs, s
        self.dbg = dbg
        if keep_s is not None and keep_s >= best_s - 0.3:
            return self.heading
        return best_h

    # ── the step ─────────────────────────────────────────────────────────────

    def step(
        self,
        t_ms: int,
        odom: tuple[float, float, float],
        *,
        active: bool,
        want: bool,
        bound,
        goal: tuple[float, float | None] | None = None,
        held: bool = False,
        lmap=None,
    ) -> Drive | None:
        """One tick. ``active``: standing, may walk, nothing else has the legs (not docking, lifted, fallen, asleep, in a flight);
        ``want``: the brain asked to walk this tick; ``goal``: where it was walking to, as (bearing from the trunk in rad, range in m
        or None), or None. ``held``: the legs are held this tick (a reflex stop, a ToF touch): the odometry may still creep while the
        legs push, so a stall is also "held for most of stall_ms"."""
        cfg = self.cfg
        if not cfg.enabled:
            return None
        self.waypoint = None
        if self.quiet:
            active = False
            self._events.clear()
        if not cfg.use_goal:
            goal = None
        self.dbg_goal = goal
        x, y, yaw = odom
        if held and not self._held_prev and want:
            self.note(
                t_ms, x, y, "hold"
            )  # (the legs held on something while it wanted to walk: a contact, as the knot counts them)
        self._held_prev = held
        if not active:
            if self.mode == "escape":
                self.mode = "idle"
                self._events.clear()
            self._hist.clear()
            self._wide.clear()
            return None
        self._lmap = lmap if cfg.use_map else None
        if self.mode == "idle":
            if t_ms < self._cool_until:
                return None
            why = self._knotted(t_ms, x, y, want)
            if why is None and self._lmap is not None and cfg.map_goal:
                why = self._map_blocked(t_ms, odom, goal, want)
            if why is None:
                return None
            if cfg.action == "loosen":
                self._loosen_until = t_ms + int(cfg.loosen_s * 1000)
                self._loosen_from = t_ms
                evs = [e for e in self._events if t_ms - e[0] <= cfg.knot_window_ms]
                self.origin = (
                    (sum(e[1] for e in evs) / len(evs), sum(e[2] for e in evs) / len(evs))
                    if evs
                    else (x, y)
                )
                self.loosened += 1
                self._events.clear()
                self._hist.clear()
                self._wide.clear()
                self._cool_until = self._loosen_until + int(cfg.cooldown_s * 1000)
                return None
            self._begin(t_ms, odom, bound, goal, why)
        # ── an escape is on ──
        ahead_m = self._ahead(
            odom, bound
        )  # the way clear straight ahead / behind (the lasting map's, else the wall memory's)
        back_ang = (math.pi - 0.15, math.pi, 0.15 - math.pi)
        if self._lmap is not None and cfg.map_back_known:
            # (a reverse arc is blind: with the map it goes only over ground the map knows to be clear)
            behind_m = min(self._lmap.free_run(odom, h, cfg.back_known_m, 0.08) for h in back_ang)
            back_room = cfg.back_known_need_m
        else:
            behind_m = min(self._clear(odom, bound, h) for h in back_ang)
            back_room = cfg.back_room_m
        out = math.hypot(x - self.origin[0], y - self.origin[1])
        if self.why == "mapgoal":
            done = self._map_goal_done(t_ms, odom, goal)
            if done is not None:
                if done:
                    self.finished += 1
                    self._cool_until = t_ms + 4000
                else:
                    self._cool_until = t_ms + int(cfg.cooldown_s * 1000) // 3
                self.mode = "idle"
                self._events.clear()
                self._hist.clear()
                self._wide.clear()
                self._mb_since = None
                return None
        goal_open = True
        if self.why == "mapgoal":
            pass  # (the map's own verdict above: no knot to be out of)
        elif goal is not None:
            # (somewhere to go: it is out when the way straight to it is clear as far as the memory sees, or it is there)
            reach = cfg.goal_range_m if goal[1] is None else min(float(goal[1]), cfg.goal_range_m)
            goal_open = self._clear(odom, bound, goal[0]) >= min(reach, cfg.goal_open_m)
            if goal[1] is not None and float(goal[1]) <= cfg.goal_there_m:
                goal_open = True  # (it is there)
        if self.why != "mapgoal" and out >= self._out_m and ahead_m >= cfg.out_open_m and goal_open:
            self.finished += 1
            self.mode = "idle"
            self._events.clear()
            self._hist.clear()
            self._wide.clear()
            self._cool_until = t_ms + 4000
            return None
        limit_s = (
            cfg.map_max_s
            if self.why == "mapgoal"
            else cfg.goal_max_s
            if goal is not None
            else cfg.max_s
        )
        if t_ms - self.t_start > limit_s * 1000:
            self._give_up(t_ms)
            return None
        if t_ms >= self._replan_ms:
            self._plan(t_ms, odom, bound, goal)
        self._heading_now(odom, bound, t_ms)
        err = wrap_angle(self.heading - yaw)
        if cfg.action == "waypoint":
            self._waypoint_step(t_ms, odom, bound, goal, want, held)
            return None
        # a reverse leg is on: it turns the nose toward the heading while it backs off what was in the way
        if self._phase == "back":
            if t_ms < self._phase_until:
                side = 1.0 if err >= 0.0 else -1.0
                return Drive(cfg.back_vx, side * cfg.arc_vyaw, True, "back", self.heading)
            self._phase, self._go_ref = "go", None
        # the way along the heading is shut right here (what it sees as it goes: the ToF, the floor line): a blocked place there and a
        # new plan — the walk goes round by the route, not by bending and forgetting
        if abs(err) <= 2.0 * cfg.aligned_rad:
            if self._clear(odom, bound, err) < cfg.block_m:
                if self._block_since is None:
                    self._block_since = t_ms
                elif t_ms - self._block_since >= cfg.block_ms:
                    self._ev(t_ms, "block")
                    self._fail_here(t_ms, odom, bound, goal)
                    if cfg.max_tries and self._tries >= cfg.max_tries:
                        self._give_up(t_ms)
                        return None
                    err = wrap_angle(self.heading - yaw)
            else:
                self._block_since = None
        else:
            self._block_since = None
        # the stall: asked to go, no way made (a step's worth of ground in stall_ms), or held (touch / stop) for most of it
        self._held.append((t_ms, held))
        while self._held and t_ms - self._held[0][0] > cfg.stall_ms:
            self._held.popleft()
        held_share = sum(1 for h in self._held if h[1]) / max(1, len(self._held))
        held_long = t_ms - self._held[0][0] >= 0.9 * cfg.stall_ms and held_share >= cfg.stall_held
        if self._go_ref is None:
            self._go_ref, self._go_t, self._go_yaw = (x, y), t_ms, yaw
        if (
            math.hypot(x - self._go_ref[0], y - self._go_ref[1]) >= cfg.stall_m
            or (cfg.stall_yaw > 0.0 and abs(wrap_angle(yaw - self._go_yaw)) >= cfg.stall_yaw)
        ) and not held_long:
            self._go_ref, self._go_t, self._go_yaw = (x, y), t_ms, yaw
        elif held_long or t_ms - self._go_t >= cfg.stall_ms:
            self._ev(
                t_ms,
                "stall",
                held_long,
                round(math.hypot(x - self._go_ref[0], y - self._go_ref[1]), 2),
            )
            self._held.clear()
            self._fail_here(t_ms, odom, bound, goal)
            if cfg.max_tries and self._tries >= cfg.max_tries:
                self._give_up(t_ms)
                return None
            self._phase, self._phase_until = "back", t_ms + cfg.turn_back_ms
            err = wrap_angle(self.heading - yaw)
            side = 1.0 if err >= 0.0 else -1.0
            if behind_m >= back_room:
                return Drive(cfg.back_vx, side * cfg.arc_vyaw, True, "back", self.heading)
            self._phase = "go"
        # a thing right ahead and the nose far off the heading, or the nose jammed on it: a reverse arc turns it round / off (a forward
        # one only presses into it)
        recent = [h for h in self._held if t_ms - h[0] <= cfg.jam_ms]
        jammed = (
            bool(recent)
            and sum(1 for h in recent if h[1]) >= cfg.jam_held * len(recent)
            and (t_ms - recent[0][0] >= 0.5 * cfg.jam_ms)
        )
        geometric = (ahead_m < cfg.arc_room_m and abs(err) > cfg.turn_back_rad) or ahead_m < 0.1
        if (
            (geometric and not cfg.soft_follow) or jammed or (cfg.soft_follow and ahead_m < 0.05)
        ) and behind_m >= back_room:
            self._ev(t_ms, "turnback", round(ahead_m, 2), round(err, 2), jammed)
            self._phase, self._phase_until = "back", t_ms + cfg.turn_back_ms
            side = 1.0 if err >= 0.0 else -1.0
            return Drive(cfg.back_vx, side * cfg.arc_vyaw, True, "back", self.heading)
        # forward: an arc toward the heading, as tight as the error asks (the body cannot turn on the spot; a duck with the quick turn
        # pivots when it is far off), a straight walk when it is nearly on it
        side = 1.0 if err >= 0.0 else -1.0
        if abs(err) > cfg.aligned_rad:
            self._phase = "turn"
            vyaw = max(-cfg.arc_vyaw, min(cfg.arc_vyaw, cfg.go_gain * err))
            vyaw = side * max(abs(vyaw), 0.5)
            return Drive(
                cfg.arc_vx,
                vyaw,
                False,
                "turn",
                self.heading,
                pivot=side if abs(err) > cfg.pivot_rad else 0.0,
            )
        self._phase = "go"
        vyaw = max(-cfg.go_vyaw_max, min(cfg.go_vyaw_max, cfg.go_gain * err))
        if ahead_m < cfg.weave_near_m:
            # a thing close ahead stops a straight walk (the ToF stop): an arc goes on, toward the side the route bends to
            # (the sign of the heading error, else the side the last arc took)
            w = side if abs(err) > 0.05 else self._weave
            self._weave = w
            return Drive(cfg.arc_vx, w * cfg.weave_vyaw, False, "go", self.heading)
        return Drive(cfg.go_vx, vyaw, False, "go", self.heading)

    def _map_goal_world(
        self, odom: tuple[float, float, float], goal: tuple[float, float | None] | None
    ) -> tuple[float, float] | None:
        """Where the walk is going, as a point of the odometry frame, when it is a goal the map can speak about (not too near, not too far)."""
        cfg = self.cfg
        if goal is None:
            return None
        if goal[1] is not None and not (cfg.map_goal_min_m <= float(goal[1]) <= cfg.map_goal_max_m):
            return None
        return self._goal_world(odom, goal)

    def _map_blocked(
        self,
        t_ms: int,
        odom: tuple[float, float, float],
        goal: tuple[float, float | None] | None,
        want: bool,
    ) -> str | None:
        """The walk goes to a place and the map says a straight walk there is shut (for ``map_block_s``): "mapgoal"."""
        cfg = self.cfg
        if t_ms - self._mb_t < 500:
            return (
                None
                if self._mb_since is None or t_ms - self._mb_since < int(cfg.map_block_s * 1000)
                else "mapgoal"
            )
        self._mb_t = t_ms
        gw = self._map_goal_world(odom, goal) if want else None
        if gw is None:
            self._mb_since = None
            return None
        shut = not self._lmap.line_clear(
            (odom[0], odom[1]), gw, cfg.map_line_r, cfg.map_goal_skip_m
        )
        if shut and cfg.map_goal_events > 0:
            # (the map says shut: it is a way round only when the walk is also going wrong here: the older layers get most of the way)
            shut = self._recent(t_ms, odom[0], odom[1], 30_000, 1.2) >= cfg.map_goal_events
        if shut and cfg.map_goal_stall_s > 0.0:
            dist = math.hypot(gw[0] - odom[0], gw[1] - odom[1])
            self._rng_hist.append((t_ms, dist))
            horizon = int(cfg.map_goal_stall_s * 1000)
            while self._rng_hist and t_ms - self._rng_hist[0][0] > horizon + 2000:
                self._rng_hist.popleft()
            old = next((d for t, d in self._rng_hist if t >= t_ms - horizon), None)
            if (
                not self._rng_hist
                or t_ms - self._rng_hist[0][0] < 0.9 * horizon
                or old is None
                or old - min(d for _t, d in self._rng_hist) >= cfg.map_goal_progress_m
            ):
                shut = False  # (not yet a stall: it is coming nearer, or has not been asked long enough)
        elif not shut:
            self._rng_hist.clear()
        if not shut:
            self._mb_since = None
            return None
        if self._mb_since is None:
            self._mb_since = t_ms
        return "mapgoal" if t_ms - self._mb_since >= int(cfg.map_block_s * 1000) else None

    def _map_goal_done(
        self, t_ms: int, odom: tuple[float, float, float], goal: tuple[float, float | None] | None
    ) -> bool | None:
        """A map-goal escape: True = the straight way is open (or it is there), False = the walk no longer goes to a place, None = on."""
        cfg = self.cfg
        if goal is None:
            return False
        if goal[1] is not None and float(goal[1]) <= cfg.goal_there_m:
            return True
        gw = self._goal_world(odom, goal)
        if gw is None or self._lmap is None:
            return False
        if self._lmap.line_clear((odom[0], odom[1]), gw, cfg.map_line_r, cfg.map_goal_skip_m):
            if self._mc_since is None:
                self._mc_since = t_ms
            return True if t_ms - self._mc_since >= int(cfg.map_clear_s * 1000) else None
        self._mc_since = None
        return None

    def _waypoint_step(
        self,
        t_ms: int,
        odom: tuple[float, float, float],
        bound,
        goal: tuple[float, float | None] | None,
        want: bool,
        held: bool,
    ) -> None:
        """``action="waypoint"``: no legs, only where to head. The route is kept and re-planned as for the legs; a stall (asked to walk,
        no ground made in ``stall_ms``, or held most of it) puts a blocked place ahead and plans again."""
        cfg = self.cfg
        x, y, yaw = odom
        la_range = cfg.look_m
        if self.route is not None and len(self.route.points) > 1:
            _near, la = CR.lookahead(self.route.points, (x, y), cfg.look_m)
            la_range = max(0.3, math.hypot(la[0] - x, la[1] - y))
        self.waypoint = (wrap_angle(self.heading - yaw), max(la_range, cfg.waypoint_range_m))
        self._held.append((t_ms, held and want))
        while self._held and t_ms - self._held[0][0] > cfg.stall_ms:
            self._held.popleft()
        if not want:
            self._go_ref = None
            return None
        if self._go_ref is None:
            self._go_ref, self._go_t = (x, y), t_ms
        held_share = sum(1 for h in self._held if h[1]) / max(1, len(self._held))
        held_long = t_ms - self._held[0][0] >= 0.9 * cfg.stall_ms and held_share >= cfg.stall_held
        if math.hypot(x - self._go_ref[0], y - self._go_ref[1]) >= cfg.stall_m and not held_long:
            self._go_ref, self._go_t = (x, y), t_ms
        elif held_long or t_ms - self._go_t >= cfg.stall_ms * 2:
            self._ev(t_ms, "stall", held_long)
            self._held.clear()
            self._fail_here(t_ms, odom, bound, goal)
        return None

    def _ev(self, t_ms: int, *what) -> None:
        self.dbg_events.append((t_ms, *what))
        del self.dbg_events[:-12]

    def _clear(self, odom: tuple[float, float, float], bound, rel_heading: float) -> float:
        """The way clear along a heading from the trunk: the lasting map's when the escape plans over it, else the wall memory's."""
        if self._lmap is not None:
            return self._lmap.clearance(odom, rel_heading)
        return bound.clearance(odom, rel_heading)

    def _ahead(self, odom: tuple[float, float, float], bound) -> float:
        if self._lmap is not None:
            return min(self._lmap.clearance(odom, h) for h in (-0.2, 0.0, 0.2))
        return bound.ahead(odom)

    def _begin(
        self,
        t_ms: int,
        odom: tuple[float, float, float],
        bound,
        goal: tuple[float, float | None] | None,
        why: str = "knot",
    ) -> None:
        x, y, _yaw = odom
        wide = why == "wide"
        self.why = why
        self._mc_since = None
        if why == "mapgoal":
            self.map_started += 1
        self._out_m = self.cfg.wide_out_m if wide else self.cfg.out_m
        evs = [
            e
            for e in self._events
            if t_ms - e[0] <= (self.cfg.wide_window_ms if wide else self.cfg.knot_window_ms)
        ]
        if evs:
            cx = sum(e[1] for e in evs) / len(evs)
            cy = sum(e[2] for e in evs) / len(evs)
            # (the centre of the knot: the events' mean, pulled to where it is when they are few)
            self.origin = (0.5 * (cx + x), 0.5 * (cy + y))
        else:
            self.origin = (x, y)
        self.mode = "escape"
        self.t_start = t_ms
        self._tries = 0
        self.route = None
        self._phase = "turn"
        self._phase_until = 0
        self._go_ref = None
        self._block_since = None
        self.started += 1
        self._trail0 = list(bound.trail)
        self._centre = self._window(x, y, self._goal_world(odom, goal))
        self._plan(t_ms, odom, bound, goal)
        self.heading = odom[2]
        self._fan_set = False
        self._failed_h.clear()
        self._heading_now(odom, bound, t_ms, rechoose=True)

    def _fail_here(
        self,
        t_ms: int,
        odom: tuple[float, float, float],
        bound,
        goal: tuple[float, float | None] | None,
    ) -> None:
        """The way ahead did not give: a blocked place there (for a while) and a new plan."""
        cfg = self.cfg
        x, y, _yaw = odom
        self._discs.append(
            (
                t_ms,
                x + cfg.fail_ahead_m * math.cos(self.heading),
                y + cfg.fail_ahead_m * math.sin(self.heading),
                cfg.fail_r,
            )
        )
        del self._discs[:-8]
        self._failed_h.append((t_ms, x, y, self.heading))
        del self._failed_h[:-8]
        self._tries += 1
        self._block_since = None
        self._go_ref = None
        self.route = None  # (the failed one is not "kept")
        self._plan(t_ms, odom, bound, goal)
        self._heading_now(odom, bound, t_ms, rechoose=True)

    def _give_up(self, t_ms: int) -> None:
        self.gave_up += 1
        self.mode = "idle"
        self._events.clear()
        self._hist.clear()
        self._wide.clear()
        self._cool_until = t_ms + int(self.cfg.cooldown_s * 1000)


__all__ = ["CLUTTER_ENABLED", "DEFAULT", "Clutter", "ClutterConfig", "Drive"]
