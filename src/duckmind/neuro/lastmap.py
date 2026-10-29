"""A lasting map: where things are, built only from the duck's own senses and kept for hours (owner 2026-10-11).

Owner's words (original complaint): 「東西稍微密集環境下，小鴨沒有辦法找到一條路徑到達想要去的位置，會一直鬼打牆前進後退轉彎」.
The wall memory of ``boundary.py`` forgets in 25 s (a few sparse points); a route over it flips every second
(``CLUTTER-ESCAPE-2026-10-11.md``). This is the other half: a coarse occupancy grid that lives as long as the duck's life.

* **Cells.** 0.10 m square cells in a sparse dict (``(i, j) -> [evidence, last seen, last hit]``), evidence in log-odds style: more than
  :data:`OCC_AT` is "in the way", less than :data:`FREE_AT` is "seen clear", in between is "not known". At most :data:`MAX_CELLS` cells
  (6 000 = 60 m2): a few tens of kB, far inside the RK3566's 1 GB.
* **What lays evidence down** (nothing but the duck's own senses): the camera's floor line (the clear floor of each sector is *free*
  up to a hand short of where it ends, the end of a ``wall`` / ``thing`` sector is *occupied*), the ToF's ahead return (free up to it,
  occupied at it; a touch is a strong hit), a bump or a brush, a place the escape marked as blocked (``note_block``), and the ground the
  trunk walked over (free: it stood there). The ToF's left / right returns are not used (:attr:`LastMapConfig.use_tof_side`): where in
  the view they are is not known and they smear the map (measured, ``tools/lastmap/fp_sources.py``). A hit on a cell that was hit a
  second ago counts a fifth (the same stray return seen again from where the duck still stands is not a second witness).
* **It corrects itself.** A cell that a later look sees clear loses its occupied evidence in a few looks (a chair that was moved, a toy,
  a person); what is never looked at again fades with a 4 hour time constant (:data:`TAU_S`: hours, not the wall memory's 25 s).
  Nothing is written while the duck is fallen or for 5 s after (the legs slip), nor after a pick-up until the dock says where it is.
* **Frame.** The brain's odometry (``frame.body.odom``) is the map's frame, plus a small correction (:attr:`LastMap.corr`) that the dock
  pulls (``anchor_dock``: the dock program's tracker gives the duck's pose in the dock's frame from a tag read or from sitting on the
  contacts). The grid itself is never moved; the correction is where the odometry's drift is taken out of the pose that evidence is laid
  down at. Not built: scan matching against the map (three ToF ranges and five floor-line sectors constrain nothing across a wall).
* **For the plan.** :meth:`LastMap.maps` hands ``clutter_route`` the grid it already searches (blocked within the body's room of a counted
  cell, a ring that costs more, unknown ground a little dearer than known); a lone weak cell is not counted (:meth:`strong_cells`);
  :meth:`line_clear` says whether a straight walk is free, :meth:`clearance` how far a heading is.
* **Saved with the life** (``to_state`` / ``load_state``): cells (i, j, evidence, age in seconds), the correction and where the dock is.
  Additive: a snapshot without it is as valid as ever (``life/snapshot.py`` version 1).

Honest senses only: the SensorFrame (camera floor line, ToF, body events, odometry). The map never reads a world position; the
test-only comparison with the true map is in ``tools/lastmap``. Pure and clock-injected (``t_ms``); deterministic.
"""

from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass
from typing import Any

from duckmind.neuro import clutter_route as CR
from duckmind.neuro.contract import SensorFrame

# the setting: "0" = no map, "obs" = the map is built and saved but nothing reads it, "act" / "1" = the plan reads it (a way out of a knot),
# "actgoal" = also for walks to a place behind something, "wp" = the route only gives the walk a heading (``NeuroBrain(lastmap=...)`` overrides per duck). Measured 2026-10-11: docs/design/neural-duck/LASTING-MAP-2026-10-11.md.
LASTMAP_MODE = os.environ.get("DUCKMIND_LASTMAP", "0")


def mode_of(value: str | bool | None) -> str:
    """ "off" | "obs" | "act" (a route over the map has the legs when the duck goes round in one place) | "actgoal" (and also for a walk to a place
    the map says is behind something) | "wp" (the route only tells the walk where to head) | "soft" ("act" with a follower that goes on through a brush
    and keeps its commitment; the map's margin stays the ring ladder, a continuous soft margin is ``LastMapConfig(soft=True)``) from a setting (None: the
    environment's)."""
    if value is None:
        value = LASTMAP_MODE
    if value is True:
        return "act"
    if value is False:
        return "off"
    v = str(value).lower()
    if v in ("wp", "waypoint"):
        return "wp"
    if v in ("soft", "act+soft"):
        return "soft"
    if v in ("actgoal", "act+goal"):
        return "actgoal"
    if v in ("1", "on", "act", "true"):
        return "act"
    if v in ("obs", "observe"):
        return "obs"
    return "off"


def overrides_from_env() -> tuple[dict[str, Any], dict[str, Any]]:
    """Development aids: ``DUCKMIND_LASTMAP_LCFG`` / ``DUCKMIND_LASTMAP_CCFG`` = ``key=value,key=value`` (numbers, true / false) override the
    map's / the clutter escape's settings for a run. Empty (the normal case): nothing."""
    out = []
    for name in ("DUCKMIND_LASTMAP_LCFG", "DUCKMIND_LASTMAP_CCFG"):
        d: dict[str, Any] = {}
        for part in os.environ.get(name, "").split(","):
            if "=" not in part:
                continue
            k, v = (t.strip() for t in part.split("=", 1))
            if v.lower() in ("true", "false"):
                d[k] = v.lower() == "true"
            else:
                try:
                    d[k] = float(v)
                except ValueError:
                    d[k] = v
        out.append(d)
    return out[0], out[1]


CELL_M = 0.10
MAX_CELLS = 6000
POSE_KEEP_MS = 2500  # odometry history kept to place a look at its capture time
STEP_M = 0.05  # a ray is sampled this finely
E_MAX = 4.0  # evidence is kept within [-E_FREE_MAX, E_MAX]
E_FREE_MAX = 2.5
OCC_AT = 1.2  # at least this: in the way
FREE_AT = -0.5  # at most this: seen clear
TAU_S = (
    4.0 * 3600.0
)  # evidence forgotten with this time constant when nothing looks at the place again
SWEEP_S = 60.0
GONE_BELOW = 0.08

# evidence per observation
FLOOR_FREE = -0.30  # a floor line's clear floor, per look and cell
FLOOR_HIT = 0.70  # a floor line's wall / thing end, per look and cell
FLOOR_EDGE_M = (
    0.10  # the free stretch stops this short of the end (the line's scatter is 6 % of the distance)
)
TOF_FREE = -0.20
TOF_HIT = 0.90
TOF_MAX_MM = 1200
TOF_TOUCH = 2.0
BUMP_HIT = 1.6
BRUSH_HIT = 1.0
BLOCK_HIT = 1.4
TRAIL_FREE = -0.6
BODY_R_M = 0.07  # the trunk's footprint cells (radius) that a step proves free
TRAIL_STEP_M = 0.04
NEAR_M = 0.10  # nearer than this to the lens is under the body: no ray evidence there
FALL_PAUSE_MS = 5000  # after a fall: the legs have slipped and the odometry with them; nothing is written for this long
HIT_GAP_S = 1.0  # a second hit on a cell within this counts for only ...
HIT_REPEAT = 0.2  # ... this share
SURE_AT = 2.4  # evidence this strong is a thing even as a single cell
RING_R = 0.46  # a route is dearer within this of something …
RING_COST = 3.0  # … by up to this much a cell (at the blocked edge)

TOF_BEARINGS = ((0.26, "left_mm"), (0.0, "ahead_mm"), (-0.26, "right_mm"))
BUMP_AHEAD_M = 0.14
BRUSH_SIDE_M = 0.13

# what the duck knows of its home (MAP-EXPLORE-2026-10-12: telemetry, the growth view, the page's map): from its own map only, never
# the true layout
EDGE_WALL_CELLS = 3  # a clear cell this near (cells) a counted thing is no open edge: the floor line's clear stretch stops a hand short of a wall
PLACE_CELLS = 10  # a "place" is a 1 m square of floor …
PLACE_FREE = 30  # … of which it has seen at least this many cells clear
HOME_MARGIN_M = (
    0.6  # the unseen floor behind an open edge is guessed this far beyond what it has seen
)
HOME_FLOOD_MAX = 8000  # (cells: a bound on that guess's work)
COARSE_CELLS = 2  # the page's map: 2 x 2 cells (0.2 m) a square


@dataclass(frozen=True, slots=True)
class LastMapConfig:
    enabled: bool = True
    anchor: bool = True  # the dock's fixes pull the odometry's correction (False: a development check of what the drift does without them)
    learn: bool = (
        True  # False: the map is read, never written (a development check with the true map)
    )
    use_floor: bool = True
    use_tof: bool = True
    use_trail: bool = True
    use_tof_side: bool = False  # the ToF's left / right returns: where in the view they are is not known (the third's middle is taken); measured: they smear the map (tools/lastmap/fp_sources.py), a map without them is what the plan can use
    tof_ahead_off_m: float = 0.0  # added to the ToF's ahead return (the trunk origin to the hit: the depth summary's own reference)
    trace_src: bool = False  # a development aid: remember which sense laid each hit (``sources``)
    floor_hit_max_m: float = 1.45  # a floor line's wall / thing end farther than this lays no hit (the line's scatter grows with the distance)
    soft: bool = False  # the plan's margin from things is a cost that falls off, not a forbidden ring (``hard_r`` / ``margin_r`` / ``margin_cost``)
    hard_r: float = 0.06  # soft: a cell this near the centre of a counted one is the thing itself (blocked); nearer cells than ``margin_r`` only cost more
    margin_r: float = 0.22  # soft: the extra cost falls to nothing at this distance from a counted cell's centre …
    margin_cost: float = 2.0  # … and is this much a cell (on top of 1) right beside the thing
    margin_exp: float = 0.0  # soft: 0 = the cost falls off in a straight line; else it decays as exp(-margin_exp * distance) per metre (a Nav2-style inflation), cut at margin_r
    ring_r: float = RING_R  # a route is dearer within this of a counted cell …
    ring_cost: float = RING_COST  # … by up to this much a cell (at the blocked edge)
    unknown_cost: float = (
        CR.UNKNOWN_COST
    )  # what a route pays to cross a cell nobody has looked at (a guess, not a wall)
    tau_s: float = TAU_S
    occ_at: float = OCC_AT
    free_at: float = FREE_AT


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _compose(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Pose ``b`` given in the frame of pose ``a``, as a pose of a's parent frame."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], _wrap(a[2] + b[2])


def _inverse(a: tuple[float, float, float]) -> tuple[float, float, float]:
    c, s = math.cos(a[2]), math.sin(a[2])
    return -(c * a[0] + s * a[1]), -(-s * a[0] + c * a[1]), _wrap(-a[2])


class LastMap:
    """The lasting map (module doc). One :meth:`update` per brain tick."""

    __slots__ = (
        "cfg",
        "cells",
        "corr",
        "_poses",
        "_floor_seen",
        "_t_ms",
        "_sweep_ms",
        "_trail_xy",
        "looks",
        "updates",
        "lost",
        "_strong",
        "_dirty",
        "_t0",
        "dock_pose",
        "anchors",
        "anchor_shift_m",
        "sources",
        "_pause_ms",
        "_pending",
    )

    def __init__(self, cfg: LastMapConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else LastMapConfig()
        self.cells: dict[
            tuple[int, int], list[float]
        ] = {}  # (i, j) -> [evidence, last seen (s), last hit (s)]
        self.corr = (0.0, 0.0, 0.0)  # odometry -> map: (dx, dy, dtheta)
        self._poses: deque[tuple[int, float, float, float]] = deque()
        self._floor_seen: int | None = None
        self._t_ms: int | None = None
        self._sweep_ms = 0
        self._trail_xy: tuple[float, float] | None = None
        self.looks = 0
        self.updates = 0
        self._strong: set[tuple[int, int]] = set()
        self._dirty = True
        self.dock_pose: tuple[float, float, float] | None = (
            None  # the dock frame's origin, as a pose in the map frame (x, y, theta)
        )
        self.sources: dict[
            tuple[int, int], dict[str, int]
        ] = {}  # (cfg.trace_src) cell -> {sense: hits}
        self.anchors = 0  # dock fixes taken in
        self.anchor_shift_m = 0.0  # how far the last fix moved the pose estimate (telemetry)
        self._t0: int | None = (
            None  # the map's own clock starts at its first update (seconds since: cell times, ages)
        )
        self._pending: list[
            tuple[float, float, float, float]
        ] = []  # saved cells in the DOCK's frame (x, y, evidence, age), waiting for the first dock fix
        self._pause_ms = -1  # after a fall: nothing is written until then
        self.lost = False  # the odometry frame broke (picked up, set down): no evidence until it is anchored again

    # ── frames ───────────────────────────────────────────────────────────────

    def to_map(self, x: float, y: float) -> tuple[float, float]:
        dx, dy, th = self.corr
        c, s = math.cos(th), math.sin(th)
        return c * x - s * y + dx, s * x + c * y + dy

    def from_map(self, x: float, y: float) -> tuple[float, float]:
        dx, dy, th = self.corr
        c, s = math.cos(th), math.sin(th)
        ux, uy = x - dx, y - dy
        return c * ux + s * uy, -s * ux + c * uy

    def pose(self, odom: tuple[float, float, float]) -> tuple[float, float, float]:
        """The duck's pose in the map frame: the odometry corrected."""
        x, y = self.to_map(odom[0], odom[1])
        return x, y, odom[2] + self.corr[2]

    def anchor_dock(
        self,
        odom: tuple[float, float, float],
        in_dock: tuple[float, float, float],
        *,
        gain: float = 0.5,
        max_jump_m: float = 1.2,
    ) -> str:
        """A landmark: the duck's pose in the dock's frame (``in_dock``: x, y in m, heading in rad: a tag read, or sitting on the
        contacts) while the odometry says ``odom``. The first time sets where the dock is in the map; later ones pull the odometry's
        correction toward what the dock says (``gain``), and take the duck's lost place back when it was picked up. Returns "set",
        "fix", "relocated" or "skip" (a fix that would move the pose more than ``max_jump_m`` is a bad read, not a drift)."""
        if not self.cfg.enabled or not self.cfg.anchor:
            return "skip"
        est = self.pose(odom)
        if self.dock_pose is None:
            self.dock_pose = _compose(est, _inverse(in_dock))
            self.anchors += 1
            if self._pending:
                self._place_pending()
            return "set"
        target = _compose(self.dock_pose, in_dock)
        dxy = (target[0] - est[0], target[1] - est[1])
        dth = _wrap(target[2] - est[2])
        jump = math.hypot(*dxy)
        if self.lost:
            gain = 1.0
        elif jump > max_jump_m or abs(dth) > 0.9:
            return "skip"
        th = self.corr[2] + gain * dth
        c, s = math.cos(th), math.sin(th)
        ex, ey = est[0] + gain * dxy[0], est[1] + gain * dxy[1]
        self.corr = (ex - (c * odom[0] - s * odom[1]), ey - (s * odom[0] + c * odom[1]), th)
        self.anchor_shift_m = gain * jump
        self.anchors += 1
        self._dirty = True
        if self.lost:
            self.lost = False
            return "relocated"
        return "fix"

    def _place_pending(self) -> None:
        """The cells of a rebased map, from the dock's frame into this boot's map frame (the evidence adds to what was seen since)."""
        assert self.dock_pose is not None
        t_s = self._now_s()
        for xd, yd, e, age in self._pending:
            x, y, _ = _compose(self.dock_pose, (xd, yd, 0.0))
            key = self.cell_of(x, y)
            cell = self.cells.get(key)
            if cell is None:
                if len(self.cells) < MAX_CELLS:
                    self.cells[key] = [max(-E_FREE_MAX, min(E_MAX, e)), t_s - age, -1e9]
            else:
                cell[0] = max(-E_FREE_MAX, min(E_MAX, cell[0] + e))
        self._pending = []
        self._dirty = True

    @staticmethod
    def cell_of(x: float, y: float) -> tuple[int, int]:
        return int(math.floor(x / CELL_M)), int(math.floor(y / CELL_M))

    # ── evidence ─────────────────────────────────────────────────────────────

    def _add(self, x: float, y: float, de: float, t_s: float, src: str = "") -> None:
        """Evidence ``de`` at a point of the MAP frame. A hit on a cell that was hit a moment ago counts for little (the same stray
        return seen again from where the duck still stands is not a second witness)."""
        self._dirty = True
        key = self.cell_of(x, y)
        if self.cfg.trace_src and de > 0.0:
            d = self.sources.setdefault(key, {})
            d[src] = d.get(src, 0) + 1
        cell = self.cells.get(key)
        if cell is None:
            if len(self.cells) >= MAX_CELLS:
                return
            self.cells[key] = [max(-E_FREE_MAX, min(E_MAX, de)), t_s, t_s if de > 0.0 else -1e9]
            return
        if de > 0.0:
            if t_s - cell[2] < HIT_GAP_S:
                de *= HIT_REPEAT
            cell[2] = t_s
        cell[0] = max(-E_FREE_MAX, min(E_MAX, cell[0] + de))
        cell[1] = t_s

    def _ray_free(
        self, x: float, y: float, heading: float, length: float, de: float, t_s: float
    ) -> None:
        self._dirty = True
        c, s = math.cos(heading), math.sin(heading)
        d = NEAR_M
        seen: tuple[int, int] | None = None
        while d <= length:
            key = self.cell_of(x + d * c, y + d * s)
            if key != seen:
                seen = key
                cell = self.cells.get(key)
                if cell is None:
                    if len(self.cells) < MAX_CELLS:
                        self.cells[key] = [max(-E_FREE_MAX, de), t_s, -1e9]
                else:
                    cell[0] = max(-E_FREE_MAX, min(E_MAX, cell[0] + de))
                    cell[1] = t_s
            d += STEP_M

    def _now_s(self, t_ms: int | None = None) -> float:
        t = self._t_ms if t_ms is None else t_ms
        return 0.0 if t is None or self._t0 is None else (t - self._t0) / 1000.0

    def _pose_at(self, t_ms: int) -> tuple[float, float, float]:
        best = self._poses[-1] if self._poses else (t_ms, 0.0, 0.0, 0.0)
        for pose in reversed(self._poses):
            if pose[0] <= t_ms:
                best = pose
                break
            best = pose
        return best[1], best[2], best[3]

    def update(
        self,
        t_ms: int,
        frame: SensorFrame,
        *,
        clock_offset: int = 0,
        valid: bool = True,
        blocked_now: bool = False,
    ) -> None:
        """This tick's senses into the map. ``valid``: the trunk is on its feet and the odometry is trusted (not lifted, not fallen);
        ``blocked_now``: the legs are held by something this tick (the footprint proves nothing then)."""
        if not self.cfg.enabled or not self.cfg.learn:
            return
        body = frame.body
        odom = body.odom
        if self._t0 is None:
            self._t0 = t_ms
        t_s = (t_ms - self._t0) / 1000.0
        self._t_ms = t_ms
        self._poses.append((t_ms, odom[0], odom[1], odom[2]))
        while self._poses and t_ms - self._poses[0][0] > POSE_KEEP_MS:
            self._poses.popleft()
        if "lifted" in frame.events or "set_down" in frame.events:
            self.lost = True  # (picked up and put down: the odometry knows nothing of where)
        if "fell" in frame.events or "righted" in frame.events or frame.body.fallen:
            self._pause_ms = (
                t_ms + FALL_PAUSE_MS
            )  # (a fall: nothing is written while it lasts and for a few seconds after)
        if t_ms >= self._sweep_ms:
            self._sweep(t_s)
            self._sweep_ms = t_ms + int(SWEEP_S * 1000)
        if (
            self.lost
            or not valid
            or body.posture not in ("stand", "walk")
            or body.fallen
            or t_ms < self._pause_ms
        ):
            return
        self.updates += 1
        cam = frame.camera
        if (
            self.cfg.use_floor
            and cam.floor
            and cam.floor_t_ms is not None
            and cam.floor_t_ms != self._floor_seen
        ):
            self._floor_seen = cam.floor_t_ms
            self._look(self._pose_at(cam.floor_t_ms + clock_offset), cam.floor, t_s)
        tof = frame.tof
        if self.cfg.use_tof and tof.present:
            mx, my = self.to_map(odom[0], odom[1])
            yaw = odom[2] + self.corr[2]
            for bearing, name in TOF_BEARINGS:
                if bearing != 0.0 and not self.cfg.use_tof_side:
                    continue
                mm = getattr(tof, name)
                if mm is None or mm >= TOF_MAX_MM:
                    continue
                r = mm / 1000.0 + (self.cfg.tof_ahead_off_m if bearing == 0.0 else 0.06)
                h = yaw + bearing
                self._ray_free(mx, my, h, max(0.0, r - 0.10), TOF_FREE, t_s)
                self._add(
                    mx + r * math.cos(h),
                    my + r * math.sin(h),
                    TOF_HIT,
                    t_s,
                    "tof_ahead" if bearing == 0.0 else "tof_side",
                )
            if tof.touch:
                self._add(
                    mx + BUMP_AHEAD_M * math.cos(yaw),
                    my + BUMP_AHEAD_M * math.sin(yaw),
                    TOF_TOUCH,
                    t_s,
                    "tof_touch",
                )
        if "blocked" in frame.events:
            self._hit(odom, 0.0, BUMP_AHEAD_M, BUMP_HIT, t_s)
        if "brushed" in frame.events and abs(body.gyro[2]) > 0.3:
            side = 1.0 if body.gyro[2] < 0.0 else -1.0
            self._hit(odom, side * 1.35, BRUSH_SIDE_M, BRUSH_HIT, t_s)
        if self.cfg.use_trail and not blocked_now:
            tx, ty = odom[0], odom[1]
            if (
                self._trail_xy is None
                or math.hypot(tx - self._trail_xy[0], ty - self._trail_xy[1]) >= TRAIL_STEP_M
            ):
                self._trail_xy = (tx, ty)
                mx, my = self.to_map(tx, ty)
                r = BODY_R_M
                steps = int(r / CELL_M)
                for a in range(-steps, steps + 1):
                    for b in range(-steps, steps + 1):
                        if math.hypot(a, b) * CELL_M <= r + 0.5 * CELL_M:
                            self._add(mx + a * CELL_M, my + b * CELL_M, TRAIL_FREE, t_s)

    def _hit(
        self, odom: tuple[float, float, float], bearing: float, dist: float, de: float, t_s: float
    ) -> None:
        mx, my = self.to_map(odom[0], odom[1])
        h = odom[2] + self.corr[2] + bearing
        self._add(mx + dist * math.cos(h), my + dist * math.sin(h), de, t_s, "bump")

    def note_block(self, t_ms: int, odom: tuple[float, float, float], bearing: float = 0.0) -> None:
        """It was blocked here (the escape, ``unstick``): the width of a box ahead of the trunk is in the way."""
        if not self.cfg.enabled or self.lost:
            return
        mx, my = self.to_map(odom[0], odom[1])
        h = odom[2] + self.corr[2] + bearing
        c, s = math.cos(h), math.sin(h)
        for side in (-0.06, 0.0, 0.06):
            self._add(
                mx + 0.14 * c - side * s,
                my + 0.14 * s + side * c,
                BLOCK_HIT,
                self._now_s(t_ms),
                "block",
            )

    def _look(self, pose: tuple[float, float, float], floor: tuple[Any, ...], t_s: float) -> None:
        px, py = self.to_map(pose[0], pose[1])
        yaw = pose[2] + self.corr[2]
        for sector in floor:
            free = float(sector.free_m)
            w = float(sector.width)
            heading = yaw + float(sector.bearing)
            hit = sector.end in ("wall", "thing") and free < self.cfg.floor_hit_max_m
            reach = free - FLOOR_EDGE_M if hit else free - 0.20
            offs = (-0.3 * w, 0.3 * w) if w > 0.1 else (0.0,)
            for off in offs:
                if reach > NEAR_M:
                    self._ray_free(px, py, heading + off, reach, FLOOR_FREE, t_s)
            if hit:
                for off in (-0.3 * w, 0.0, 0.3 * w):
                    h = heading + off
                    self._add(
                        px + free * math.cos(h),
                        py + free * math.sin(h),
                        FLOOR_HIT,
                        t_s,
                        "floor_" + str(sector.end),
                    )
        self.looks += 1

    # ── forgetting ───────────────────────────────────────────────────────────

    def _sweep(self, t_s: float) -> None:
        self._dirty = True
        tau = self.cfg.tau_s
        dead = []
        for key, cell in self.cells.items():
            age = t_s - cell[1]
            if age <= 0.0:
                continue
            e = cell[0] * math.exp(-age / tau)
            cell[0] = e
            cell[1] = t_s
            if abs(e) < GONE_BELOW:
                dead.append(key)
        for key in dead:
            del self.cells[key]

    # ── asking it ────────────────────────────────────────────────────────────

    def evidence_at(self, x: float, y: float) -> float:
        """The evidence at a point of the ODOMETRY frame (0: nothing known)."""
        mx, my = self.to_map(x, y)
        cell = self.cells.get(self.cell_of(mx, my))
        return 0.0 if cell is None else cell[0]

    def occupied(self, x: float, y: float) -> bool:
        return self.evidence_at(x, y) >= self.cfg.occ_at

    def known_free(self, x: float, y: float) -> bool:
        return self.evidence_at(x, y) <= self.cfg.free_at

    def line_clear(
        self,
        a: tuple[float, float],
        b: tuple[float, float],
        radius: float = 0.18,
        skip_end_m: float = 0.0,
    ) -> bool:
        """A straight walk from ``a`` to ``b`` (ODOMETRY frame) touches nothing occupied within ``radius`` of the line (the last
        ``skip_end_m`` of it is not looked at: the place it wants to get to has things round it)."""
        ax, ay = self.to_map(*a)
        bx, by = self.to_map(*b)
        length = math.hypot(bx - ax, by - ay)
        if length < 1e-6:
            return True
        ux, uy = (bx - ax) / length, (by - ay) / length
        strong = self.strong_cells()
        reach = max(0.0, length - skip_end_m)
        n = int(radius / CELL_M)
        d = 0.0
        seen: set[tuple[int, int]] = set()
        while d <= reach:
            px, py = ax + d * ux, ay + d * uy
            ci, cj = self.cell_of(px, py)
            for i in range(ci - n, ci + n + 1):
                for j in range(cj - n, cj + n + 1):
                    if (i, j) in seen:
                        continue
                    seen.add((i, j))
                    if (i, j) in strong:
                        cx, cy = (i + 0.5) * CELL_M, (j + 0.5) * CELL_M
                        # distance from the cell centre to the segment
                        t = max(0.0, min(length, (cx - ax) * ux + (cy - ay) * uy))
                        if (
                            math.hypot(cx - (ax + t * ux), cy - (ay + t * uy))
                            <= radius + 0.5 * CELL_M
                        ):
                            return False
            d += CELL_M
        return True

    def clearance(
        self,
        odom: tuple[float, float, float],
        rel_heading: float,
        max_m: float = 1.5,
        half_m: float = 0.10,
    ) -> float:
        """How far a walk along ``rel_heading`` (rad from the trunk, + = left) is clear in a corridor ``half_m`` either side, by this
        map's counted cells (m, at most ``max_m``). The way the wall memory's ``clearance`` answers, from what lasts."""
        strong = self.strong_cells()
        if not strong:
            return max_m
        mx, my = self.to_map(odom[0], odom[1])
        h = odom[2] + self.corr[2] + rel_heading
        c, s = math.cos(h), math.sin(h)
        n = int(half_m / CELL_M)
        d = 0.05
        while d <= max_m:
            px, py = mx + d * c, my + d * s
            ci, cj = self.cell_of(px, py)
            for i in range(ci - n, ci + n + 1):
                for j in range(cj - n, cj + n + 1):
                    if (i, j) in strong:
                        cx, cy = (i + 0.5) * CELL_M, (j + 0.5) * CELL_M
                        along = (cx - mx) * c + (cy - my) * s
                        across = -(cx - mx) * s + (cy - my) * c
                        if along > 0.0 and abs(across) <= half_m + 0.5 * CELL_M:
                            return max(0.0, along - 0.5 * CELL_M)
            d += CELL_M * 0.5
        return max_m

    def free_run(
        self,
        odom: tuple[float, float, float],
        rel_heading: float,
        max_m: float = 0.6,
        half_m: float = 0.10,
    ) -> float:
        """How far a walk along ``rel_heading`` goes over ground the map KNOWS to be clear, in a corridor ``half_m`` either side (m, at
        most ``max_m``): it stops at the first cell that is occupied or that nobody has looked at. For a reverse arc, which the head's
        camera and the ToF cannot see: the duck backs only over ground it has seen clear or walked."""
        mx, my = self.to_map(odom[0], odom[1])
        h = odom[2] + self.corr[2] + rel_heading
        c, s = math.cos(h), math.sin(h)
        free = self.cfg.free_at
        n = int(half_m / CELL_M)
        d = 0.05
        while d <= max_m:
            ci, cj = self.cell_of(mx + d * c, my + d * s)
            for i in range(ci - n, ci + n + 1):
                for j in range(cj - n, cj + n + 1):
                    cell = self.cells.get((i, j))
                    if cell is None or cell[0] > free:
                        return max(0.0, d - CELL_M)
            d += CELL_M * 0.5
        return max_m

    def occupied_cells(
        self, centre: tuple[float, float], radius: float
    ) -> list[tuple[float, float]]:
        """Centres (ODOMETRY frame) of the occupied cells within ``radius`` (square) of ``centre`` (ODOMETRY frame)."""
        out = []
        occ = self.cfg.occ_at
        mc = self.to_map(*centre)
        i0, j0 = self.cell_of(mc[0] - radius - 0.2, mc[1] - radius - 0.2)
        i1, j1 = self.cell_of(mc[0] + radius + 0.2, mc[1] + radius + 0.2)
        for (i, j), cell in self.cells.items():
            if cell[0] >= occ and i0 <= i <= i1 and j0 <= j <= j1:
                out.append(self.from_map((i + 0.5) * CELL_M, (j + 0.5) * CELL_M))
        return out

    def strong_cells(self) -> set[tuple[int, int]]:
        """The occupied cells the plan counts: evidence at least ``occ_at`` and either strong (:data:`SURE_AT`, seen many times) or with
        another occupied cell beside it (a lone weak cell is one stray return, not a thing). Worked out again after any change."""
        if not self._dirty:
            return self._strong
        self._dirty = False
        occ = self.cfg.occ_at
        cells = self.cells
        out = set()
        for (i, j), c in cells.items():
            if c[0] < occ:
                continue
            if c[0] >= SURE_AT:
                out.add((i, j))
                continue
            for a in (-1, 0, 1):
                for b in (-1, 0, 1):
                    if (a or b) and cells.get((i + a, j + b), (0.0,))[0] >= occ:
                        out.add((i, j))
                        break
                else:
                    continue
                break
        self._strong = out
        return out

    def maps(
        self,
        start: tuple[float, float],
        centre: tuple[float, float],
        *,
        block_r: float = CR.BLOCK_R,
        goal: tuple[float, float] | None = None,
        goal_free_r: float = 0.5,
        discs: list[tuple[float, float, float]] | None = None,
        ring_r: float | None = None,
        ring_cost: float | None = None,
    ) -> CR.Maps:
        """The route search's grid (``clutter_route.Maps``, in the odometry frame) from this map. A cell is blocked within ``block_r``
        of an occupied cell; within ``ring_r`` it costs more, the nearer the dearer (a route keeps to the middle of a gap); it costs 1
        where the map says clear and ``UNKNOWN_COST`` where it knows nothing."""
        n = CR.N
        ring_r = self.cfg.ring_r if ring_r is None else ring_r
        ring_cost = self.cfg.ring_cost if ring_cost is None else ring_cost
        unknown = self.cfg.unknown_cost
        m = CR.Maps(centre[0], centre[1], [False] * (n * n), [unknown] * (n * n), unknown)
        free = self.cfg.free_at
        cells = self.cells
        strong = self.strong_cells()
        corr_id = self.corr == (0.0, 0.0, 0.0)
        inf = 1e9
        dist = [inf] * (n * n)
        for i in range(n):
            for j in range(n):
                cx, cy = m.centre(i, j)
                key = self.cell_of(cx, cy) if corr_id else self.cell_of(*self.to_map(cx, cy))
                if key in strong:
                    dist[i * n + j] = 0.0
                    continue
                cell = cells.get(key)
                if cell is not None and cell[0] <= free:
                    m.cost[i * n + j] = 1.0
        # the distance to the nearest occupied cell (a two-pass chamfer, 8 neighbours, in metres)
        step, diag = CR.RES, CR.RES * math.sqrt(2.0)
        for i in range(n):
            for j in range(n):
                k = i * n + j
                d = dist[k]
                if i > 0:
                    d = min(d, dist[k - n] + step)
                    if j > 0:
                        d = min(d, dist[k - n - 1] + diag)
                    if j < n - 1:
                        d = min(d, dist[k - n + 1] + diag)
                if j > 0:
                    d = min(d, dist[k - 1] + step)
                dist[k] = d
        for i in range(n - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                k = i * n + j
                d = dist[k]
                if i < n - 1:
                    d = min(d, dist[k + n] + step)
                    if j > 0:
                        d = min(d, dist[k + n - 1] + diag)
                    if j < n - 1:
                        d = min(d, dist[k + n + 1] + diag)
                if j < n - 1:
                    d = min(d, dist[k + 1] + step)
                dist[k] = d
        if self.cfg.soft:
            # (a soft margin: only the thing's own cells are shut; beside it a way costs more, the nearer the dearer, and nothing is
            # forbidden — a narrow gap that is the only way out stays plannable, a brush is allowed)
            hard_r, margin_r, margin_cost = self.cfg.hard_r, self.cfg.margin_r, self.cfg.margin_cost
            span = max(1e-6, margin_r - hard_r)
            for k in range(n * n):
                d = dist[k]
                if d <= hard_r:
                    m.blocked[k] = True
                elif d < margin_r:
                    if self.cfg.margin_exp > 0.0:
                        m.cost[k] += margin_cost * math.exp(-self.cfg.margin_exp * (d - hard_r))
                    else:
                        m.cost[k] += margin_cost * (1.0 - (d - hard_r) / span)
        else:
            span = max(1e-6, ring_r - block_r)
            for k in range(n * n):
                d = dist[k]
                if d <= block_r:
                    m.blocked[k] = True
                elif d < ring_r:
                    m.cost[k] += ring_cost * (1.0 - (d - block_r) / span)
        for x, y, r in discs or []:
            for a, b, _d in CR._disc(m, x, y, r):
                m.blocked[a * n + b] = True
        for a, b, _d in CR._disc(m, start[0], start[1], CR.START_FREE_R):
            m.blocked[a * n + b] = False
        if goal is not None:
            for a, b, _d in CR._disc(m, goal[0], goal[1], goal_free_r):
                m.blocked[a * n + b] = False
        return m

    # ── saving it with the life ──────────────────────────────────────────────

    def to_state(self, t_s: float | None = None) -> dict[str, Any]:
        """Cells (i, j, evidence, age in s) and the correction, as plain JSON values (``t_s``: the map's own clock now, default its latest)."""
        if t_s is None:
            t_s = self._now_s()
        return {
            "v": 1,
            "cell": CELL_M,
            "corr": [round(v, 5) for v in self.corr],
            "dock": None if self.dock_pose is None else [round(v, 5) for v in self.dock_pose],
            "cells": [
                [i, j, round(c[0], 2), round(max(0.0, t_s - c[1]), 1)]
                for (i, j), c in sorted(self.cells.items())
                if abs(c[0]) >= GONE_BELOW * 2
            ],
        }

    def load_state(
        self, state: dict[str, Any], t_now: float = 0.0, down_s: float = 0.0, rebase: bool = False
    ) -> None:
        """Take a saved ``to_state`` back in (before the first update: the map's clock then starts at ``t_now`` = 0). ``down_s`` seconds
        passed while nothing ran (the evidence decays over them, as it would have). A map of another cell size is not taken.
        ``rebase``: the body's odometry frame is not the one the map was saved in (a restart somewhere else): the cells are kept in the
        DOCK's frame (the saved dock pose says where it was) and are put into this boot's frame at the first dock fix (:meth:`anchor_dock`);
        a map saved with no dock pose cannot be rebased and is not taken."""
        self.cells = {}
        self._pending = []
        if not state or abs(float(state.get("cell", CELL_M)) - CELL_M) > 1e-9:
            return
        corr = state.get("corr") or (0.0, 0.0, 0.0)
        self.corr = (float(corr[0]), float(corr[1]), float(corr[2]))
        dock = state.get("dock")
        self.dock_pose = None if not dock else (float(dock[0]), float(dock[1]), float(dock[2]))
        fade = math.exp(-max(0.0, down_s) / self.cfg.tau_s)
        if rebase:
            if self.dock_pose is None:
                self.corr = (0.0, 0.0, 0.0)
                return
            inv = _inverse(self.dock_pose)
            for i, j, e, age in state.get("cells", ()):
                e = float(e) * fade
                if abs(e) >= GONE_BELOW:
                    xd, yd, _ = _compose(inv, ((i + 0.5) * CELL_M, (j + 0.5) * CELL_M, 0.0))
                    self._pending.append((xd, yd, e, float(age)))
            self.dock_pose = (
                None  # (this boot does not know where its dock is in its own frame yet)
            )
            self.corr = (0.0, 0.0, 0.0)
            return
        for i, j, e, age in state.get("cells", ()):
            e = float(e) * fade
            if abs(e) >= GONE_BELOW:
                self.cells[(int(i), int(j))] = [e, t_now - float(age), -1e9]
        self._trail_xy = None
        self._dirty = True
        self.lost = False

    def last_odom(self) -> tuple[float, float, float] | None:
        """The odometry pose of its latest update (the page's map layer), or None before the first."""
        if not self._poses:
            return None
        _t, x, y, th = self._poses[-1]
        return x, y, th

    def view(self) -> dict[str, float | int]:
        """For the telemetry."""
        occ = sum(1 for c in self.cells.values() if c[0] >= self.cfg.occ_at)
        free = sum(1 for c in self.cells.values() if c[0] <= self.cfg.free_at)
        return {"cells": len(self.cells), "occupied": occ, "free": free, "looks": self.looks}

    # ── what it knows of its home (MAP-EXPLORE-2026-10-12: the growth view, the page) ──────────

    def _open_edge(self, i: int, j: int) -> bool:
        """Cell (i, j) is known clear and has ground beside it that nobody has looked at: two cells in a row on one side never
        seen (a one-cell hole between two looks' rays is not unseen ground)."""
        cells = self.cells
        cell = cells.get((i, j))
        if cell is None or cell[0] > self.cfg.free_at:
            return False
        for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (i + a, j + b) not in cells and (i + 2 * a, j + 2 * b) not in cells:
                return True
        return False

    def _near(self, cells: set[tuple[int, int]], reach: int) -> set[tuple[int, int]]:
        out: set[tuple[int, int]] = set()
        for i, j in cells:
            for a in range(-reach, reach + 1):
                for b in range(-reach, reach + 1):
                    out.add((i + a, j + b))
        return out

    def home_view(self) -> dict[str, float | int | None]:
        """How much of its home it knows, from its own map only (never the true layout): ``known_m2`` the floor it has seen clear,
        ``places`` the 1 m squares of floor it has seen (:data:`PLACE_FREE` of :data:`PLACE_CELLS` x PLACE_CELLS), ``edges`` the
        open edges (ground known clear beside ground never seen, not beside a wall), ``share`` (0..1, or None with nothing seen)
        the floor seen over that plus a guess of the unseen floor behind the open edges (the never-seen cells a flood from them
        reaches without crossing anything seen, within :data:`HOME_MARGIN_M` of what it has seen). Rooms it has never had a
        glimpse of are not in the guess: the share says how much of the home it knows of it has looked at."""
        cells = self.cells
        free_at = self.cfg.free_at
        free = [k for k, c in cells.items() if c[0] <= free_at]
        if not free:
            return {"known_m2": 0.0, "places": 0, "edges": 0, "share": None}
        blocks: dict[tuple[int, int], int] = {}
        for i, j in free:
            key = (i // PLACE_CELLS, j // PLACE_CELLS)
            blocks[key] = blocks.get(key, 0) + 1
        wall = self._near(self.strong_cells(), EDGE_WALL_CELLS)
        edges = [k for k in free if k not in wall and self._open_edge(*k)]
        keys = list(cells)
        m = int(HOME_MARGIN_M / CELL_M)
        i0 = min(k[0] for k in keys) - m
        i1 = max(k[0] for k in keys) + m
        j0 = min(k[1] for k in keys) - m
        j1 = max(k[1] for k in keys) + m
        unseen: set[tuple[int, int]] = set()
        queue: deque[tuple[int, int]] = deque()
        for i, j in edges:
            for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (i + a, j + b)
                if n not in cells and n not in unseen:
                    unseen.add(n)
                    queue.append(n)
        while queue and len(unseen) < HOME_FLOOD_MAX:
            i, j = queue.popleft()
            for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (i + a, j + b)
                if n in unseen or n in cells or not (i0 <= n[0] <= i1 and j0 <= n[1] <= j1):
                    continue
                unseen.add(n)
                queue.append(n)
        return {
            "known_m2": round(len(free) * CELL_M * CELL_M, 2),
            "places": sum(1 for n in blocks.values() if n >= PLACE_FREE),
            "edges": len(edges),
            "share": round(len(free) / (len(free) + len(unseen)), 3),
        }

    def coarse(self) -> dict[str, Any]:
        """The page's picture of what it knows: :data:`COARSE_CELLS` x COARSE_CELLS cells a square over the map's extent, each
        ``a`` never seen, ``b`` seen clear (half or more of its cells), ``c`` a thing it learnt (a counted cell in it), ``d`` looked
        at but not sure; run-length coded (``b12a3…``, rows from the lowest j up, i across). Map frame; ``origin`` the corner (m)."""
        cells = self.cells
        if not cells:
            return {
                "cell_m": CELL_M * COARSE_CELLS,
                "origin": [0.0, 0.0],
                "w": 0,
                "h": 0,
                "rle": "",
            }
        k = COARSE_CELLS
        strong = self.strong_cells()
        free_at = self.cfg.free_at
        sq: dict[tuple[int, int], list[int]] = {}
        for (i, j), c in cells.items():
            s = sq.setdefault((i // k, j // k), [0, 0, 0])
            if (i, j) in strong:
                s[0] += 1
            elif c[0] <= free_at:
                s[1] += 1
            else:
                s[2] += 1
        i0 = min(q[0] for q in sq)
        i1 = max(q[0] for q in sq)
        j0 = min(q[1] for q in sq)
        j1 = max(q[1] for q in sq)
        half = (k * k + 1) // 2
        out: list[str] = []
        run_ch, run_n = "", 0
        for j in range(j0, j1 + 1):
            for i in range(i0, i1 + 1):
                got = sq.get((i, j))
                ch = "a" if got is None else "c" if got[0] else "b" if got[1] >= half else "d"
                if ch == run_ch:
                    run_n += 1
                else:
                    if run_n:
                        out.append(f"{run_ch}{run_n}")
                    run_ch, run_n = ch, 1
        if run_n:
            out.append(f"{run_ch}{run_n}")
        return {
            "cell_m": CELL_M * k,
            "origin": [round(i0 * k * CELL_M, 3), round(j0 * k * CELL_M, 3)],
            "w": i1 - i0 + 1,
            "h": j1 - j0 + 1,
            "rle": "".join(out),
        }


__all__ = ["CELL_M", "LASTMAP_MODE", "LastMap", "LastMapConfig", "mode_of"]
