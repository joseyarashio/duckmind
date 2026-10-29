"""害怕時躲起來 — after a big scare it runs to a shelter it remembers, then peeks out (a:social).

The owner's A list, behaviour (8): *after a big scare runs to a remembered shelter (under the
table, beside the sofa), peeks out after a while*; decided only from its own senses, emerging from
its drives and temperament (timid ducks hide sooner), physically performed, falls are real.

**Shelters it has seen** (:class:`ShelterMap`, in its own contact-odometry frame — it drifts, so a
shelter far back in time is less sure). Two things make a place a shelter, both from its senses:

* *walls close on several sides* — the camera's floor line (duck-eye FL-v1's five sectors; MODEL
  in the prototype) says where the clear floor ends at a wall or furniture; every such end is
  laid into a long-lived wall map (:data:`WALL_CELL_M` cells, fading over :data:`WALL_TAU_S`), and
  a spot's *enclosure* is the share of the eight directions around it that meet a wall within
  :data:`ENCLOSE_M` — a nook beside the sofa, a corner, the gap by the bookshelf;
* *a roof close overhead* — the camera's roof reading (``roof``: how far above the lens a surface
  covers the top of the picture; MODEL, :mod:`duckmind.world.neuro_social_senses`: a surface low
  above the head is under the table) laid into :data:`COVER_CELL_M` cells as hits and open looks.

A spot's value is ``COVER_W × cover + ENCLOSE_W × enclosure`` (≤ 1); spots worth at least
:data:`SHELTER_MIN` are its shelters (:meth:`ShelterMap.refuges`, the best few, apart).

**The flight** (:class:`HideRun`): fear past its own threshold (:func:`hide_threshold`: the timid
at a small fright, the bold only at a big one) with a shelter known makes hiding strong in the
competition (``hide_urge``; :mod:`duckmind.neuro.select`); the run picks the refuge that is good,
near and not toward where the scare came from (the appraisal's threat heading, when it knew one),
walks there at its fastest walk (the brain's locomotor rhythm steered by :meth:`HideRun.goal`),
and once in (close to the spot, or its own senses say "covered" / "walls around") it stops and
sits low. As its fear ebbs it peeks out (:meth:`HideRun.peek_due` → the ``peek_out`` moment:
the head comes up and turns toward the way out or the scare) — rarely at first, more often as it
calms — and comes out once its fear is low and nothing has scared it for a while. A refuge it
could not get to (stuck, bumping) is marked and avoided next time.

Pure and clock-injected; deterministic (the peeks' spacing comes from a seeded stream);
standard library only.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from duckmind.i18n import msg

SHELTER_VERSION = "neuro-shelter-1"

WALL_CELL_M = 0.15
WALL_TAU_S = 3600.0  # the room's walls and furniture stay put: an hour's memory
WALL_MIN = 0.7  # a wall cell counts above this weight (seen twice: one stray end is not a wall)
WALL_MAX_CELLS = 1200
FLOOR_END_MAX_M = 1.45  # a floor-line end beyond this is "unknown", not a wall
COVER_CELL_M = 0.3
COVER_TAU_S = 3600.0
ROOF_M = 0.45  # a surface this close above the lens is a roof
ROOF_OPEN = 9.9  # the roof reading's "looked up and saw open space"
ENCLOSE_M = 0.45  # a wall within this in a direction closes it …
DIRS = 8  # … of the eight around the spot
ENCLOSE_STEP_M = 0.06
ENCLOSE_SPREAD = (0.0, -0.2, 0.2)  # rad: each direction is a ±11° cone
COVER_W = 0.7
ENCLOSE_W = 1.0
# a spot worth this is a shelter: a corner (5 of 8 directions walled: 0.62), a nook, a roof —
# not a spot along a straight wall (3 of 8: 0.38; 4 of 8 with a stray wall end)
SHELTER_MIN = 0.55
SPOT_STEP_M = 0.15  # candidate spots on this grid …
SPOT_NEAR_M = 0.3  # … this near the walls it has seen …
BODY_ROOM_M = 0.22  # … with no wall this close (the body must fit there)
REFUGE_EVERY_S = 3.0
REFUGE_APART_M = 0.6
REFUGE_KEEP = 6
FLEE_MAX_M = 4.5  # it runs only to shelters it believes are this close
DIST_COST = 0.12  # value per metre away
THREAT_COST = 0.9  # value lost for running toward the scare (full when straight at it)
THREAT_CONE = 1.0  # rad: within this of the scare's heading counts as toward it
BAD_COST = 0.6  # a refuge it could not reach recently
CHOOSE_MIN = 0.05  # a refuge must score at least this to be run to
BAD_TAU_S = 600.0
DRIFT_COST_PER_MIN = 0.01  # older memories are less sure (odometry drift)
ARRIVE_M = 0.25  # this close to the spot: in
ARRIVE_SENSED_M = 0.6  # … or this close and its senses say covered / walled (or it bumped)
SENSED_ENCLOSE = 0.25
STUCK_S = 6.0  # blocked this long on the way (not moving closer): that refuge failed
PROGRESS_M = 0.08
MIN_IN_S = 6.0  # hidden at least this long
CALM_SINCE_S = 10.0  # no scare for this long …
EXIT_FEAR = (0.1, 0.15)  # … and fear under a + b × (1 − boldness): it comes out
PEEK_FIRST_S = 3.0
PEEK_MEDIAN_S = (3.0, 12.0)  # the peeks' median spacing: a + b × how afraid it still is
PEEK_SIGMA = 0.45


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def hide_threshold(boldness: float) -> float:
    """The fear at which it wants to hide: the timid at a small fright, the bold only at a big
    one (their fear rarely gets there: genome fear_gain is lower for the bold too)."""
    return 0.25 + 0.45 * _clip(boldness)


def hide_urge(fear: float, boldness: float, startled: float = 0.0) -> float:
    """0..1: how much it wants to hide now (fear past its threshold, a fresh fright)."""
    t = hide_threshold(boldness)
    return _clip((fear - t) / 0.25 + 0.5 * _clip(startled) * (1.0 - _clip(boldness)) * fear)


@dataclass(frozen=True, slots=True)
class Refuge:
    x: float  # odometry frame (m)
    y: float
    value: float  # 0..1 how good a shelter
    cover: float
    enclosure: float
    open_heading: float  # odometry heading (rad) of its most open side (the way out)
    seen_s: float  # when it last had evidence there

    def bearing_range(self, odom: tuple[float, float, float]) -> tuple[float, float]:
        dx, dy = self.x - odom[0], self.y - odom[1]
        return _wrap(math.atan2(dy, dx) - odom[2]), math.hypot(dx, dy)


@dataclass(slots=True)
class _Cover:
    hits: float = 0.0
    open: float = 0.0
    seen_s: float = 0.0


class ShelterMap:
    """Walls and roofs it has seen, in its odometry frame, and the shelters they make."""

    __slots__ = (
        "walls",
        "wall_seen",
        "floor",
        "cover",
        "bad",
        "_t_s",
        "_floor_seen",
        "_poses",
        "_refuges",
        "_ref_t",
        "looks",
    )

    def __init__(self) -> None:
        self.walls: dict[tuple[int, int], float] = {}
        self.wall_seen: dict[tuple[int, int], float] = {}
        # floor it has stood on or seen clear (COVER_CELL_M cells): a shelter must be on it —
        # the far side of a wall is not somewhere it can go
        self.floor: set[tuple[int, int]] = set()
        self.cover: dict[tuple[int, int], _Cover] = {}
        self.bad: list[tuple[float, float, float]] = []  # (x, y, when) refuges it failed
        self._t_s = 0.0
        self._floor_seen: int | None = None
        self._poses: list[tuple[int, float, float, float]] = []
        self._refuges: list[Refuge] = []
        self._ref_t = -1e9
        self.looks = 0

    # ── evidence ─────────────────────────────────────────────────────────────

    @staticmethod
    def _wcell(x: float, y: float) -> tuple[int, int]:
        return (math.floor(x / WALL_CELL_M), math.floor(y / WALL_CELL_M))

    @staticmethod
    def _ccell(x: float, y: float) -> tuple[int, int]:
        return (math.floor(x / COVER_CELL_M), math.floor(y / COVER_CELL_M))

    def _pose_at(self, t_ms: int) -> tuple[float, float, float]:
        best = self._poses[-1] if self._poses else (t_ms, 0.0, 0.0, 0.0)
        for pose in reversed(self._poses):
            if pose[0] <= t_ms:
                best = pose
                break
            best = pose
        return best[1], best[2], best[3]

    def observe(
        self,
        t_ms: int,
        odom: tuple[float, float, float],
        floor: Sequence[Any] = (),
        floor_t_ms: int | None = None,
        roof: float | None = None,
        *,
        clock_offset: int = 0,
    ) -> None:
        """This tick's floor line (the camera's free-floor sectors, relative to the trunk at
        ``floor_t_ms``) and roof reading (m above the lens; :data:`ROOF_OPEN` = open above;
        None = cannot tell) into the memory."""
        t_s = t_ms / 1000.0
        self._t_s = t_s
        self._poses.append((t_ms, odom[0], odom[1], odom[2]))
        if len(self._poses) > 150:
            del self._poses[: len(self._poses) - 150]
        self.floor.add(self._ccell(odom[0], odom[1]))
        if floor and floor_t_ms is not None and floor_t_ms != self._floor_seen:
            self._floor_seen = floor_t_ms
            self.looks += 1
            px, py, yaw = self._pose_at(floor_t_ms + clock_offset)
            for sector in floor:
                end = str(getattr(sector, "end", "unknown"))
                free = float(getattr(sector, "free_m", 9.9))
                heading = yaw + float(getattr(sector, "bearing", 0.0))
                width = float(getattr(sector, "width", 0.13))
                # clear floor along the sector: a wall remembered there is less sure; the floor
                # is somewhere it could go
                for d in (0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2):
                    if d >= free - 0.1:
                        break
                    fx, fy = px + d * math.cos(heading), py + d * math.sin(heading)
                    key = self._wcell(fx, fy)
                    if key in self.walls:
                        self.walls[key] *= 0.6
                    self.floor.add(self._ccell(fx, fy))
                if end != "wall" or free > FLOOR_END_MAX_M:
                    continue
                for frac in (-0.4, 0.0, 0.4):
                    h = heading + frac * width
                    key = self._wcell(px + free * math.cos(h), py + free * math.sin(h))
                    self.walls[key] = min(1.0, self.walls.get(key, 0.0) + 0.5)
                    self.wall_seen[key] = t_s
            if len(self.walls) > WALL_MAX_CELLS:
                for key in sorted(self.walls, key=lambda k: self.wall_seen.get(k, 0.0))[
                    : len(self.walls) - WALL_MAX_CELLS
                ]:
                    self.walls.pop(key, None)
                    self.wall_seen.pop(key, None)
        if roof is not None:
            key = self._ccell(odom[0], odom[1])
            cell = self.cover.setdefault(key, _Cover())
            if roof <= ROOF_M:
                cell.hits += 1.0
            elif roof >= ROOF_OPEN - 1e-6 or roof > 1.0:
                cell.open += 1.0
            cell.seen_s = t_s

    def mark_bad(self, x: float, y: float, t_s: float) -> None:
        self.bad.append((x, y, t_s))
        self.bad = [b for b in self.bad if t_s - b[2] < 3.0 * BAD_TAU_S][-12:]

    # ── what a spot is worth ─────────────────────────────────────────────────

    def _wall_at(self, x: float, y: float) -> bool:
        key = self._wcell(x, y)
        w = self.walls.get(key, 0.0)
        if w <= 0.0:
            return False
        age = self._t_s - self.wall_seen.get(key, self._t_s)
        return w * math.exp(-age / WALL_TAU_S) >= WALL_MIN

    def enclosure(self, x: float, y: float) -> tuple[float, float]:
        """(the share of the eight directions with a wall within ENCLOSE_M, the heading of the
        most open direction — the way out)."""
        opened: list[bool] = []
        for k in range(DIRS):
            hit = False
            for spread in ENCLOSE_SPREAD:  # a narrow cone, not a hair-thin ray
                h = 2.0 * math.pi * k / DIRS + spread
                c, s = math.cos(h), math.sin(h)
                d = ENCLOSE_STEP_M
                while d <= ENCLOSE_M + 1e-9:
                    if self._wall_at(x + d * c, y + d * s):
                        hit = True
                        break
                    d += ENCLOSE_STEP_M
                if hit:
                    break
            opened.append(not hit)
        closed = DIRS - sum(opened)
        way_out = 0.0
        if any(opened):
            # the middle of the widest run of open directions (the way out)
            best_len, best_mid = -1, 0.0
            for start in range(DIRS):
                run_start = opened[start] and (all(opened) or not opened[(start - 1) % DIRS])
                if not run_start:
                    continue
                n = 0
                while n < DIRS and opened[(start + n) % DIRS]:
                    n += 1
                if n > best_len:
                    best_len = n
                    best_mid = 2.0 * math.pi * (start + (n - 1) / 2.0) / DIRS
            way_out = _wrap(best_mid)
        return closed / DIRS, way_out

    def cover_at(self, x: float, y: float) -> float:
        cell = self.cover.get(self._ccell(x, y))
        if cell is None or cell.hits < 2.0:
            return 0.0
        age = self._t_s - cell.seen_s
        return _clip(cell.hits / (cell.hits + cell.open + 0.5)) * math.exp(-age / COVER_TAU_S)

    def value(self, x: float, y: float) -> tuple[float, float, float, float]:
        """(value, cover, enclosure, way-out heading) of a spot."""
        encl, out = self.enclosure(x, y)
        cov = self.cover_at(x, y)
        return _clip(COVER_W * cov + ENCLOSE_W * encl), cov, encl, out

    def _wall_near(self, x: float, y: float, r: float) -> bool:
        step = WALL_CELL_M
        n = int(math.ceil(r / step))
        cx, cy = self._wcell(x, y)
        for i in range(cx - n, cx + n + 1):
            for j in range(cy - n, cy + n + 1):
                wx, wy = (i + 0.5) * step, (j + 0.5) * step
                if math.hypot(wx - x, wy - y) <= r + 0.5 * step and self._wall_at(wx, wy):
                    return True
        return False

    def refuges(self, t_s: float | None = None) -> list[Refuge]:
        """Its shelters: the best spots (≥ SHELTER_MIN) where its body fits, on floor it knows,
        at least REFUGE_APART_M apart (recomputed every REFUGE_EVERY_S)."""
        now = self._t_s if t_s is None else t_s
        if now - self._ref_t < REFUGE_EVERY_S:
            return self._refuges
        self._ref_t = now
        candidates: set[tuple[int, int]] = set()  # SPOT_STEP_M grid
        g = SPOT_STEP_M
        for key, cell in self.cover.items():
            if cell.hits >= 2.0:
                x, y = (key[0] + 0.5) * COVER_CELL_M, (key[1] + 0.5) * COVER_CELL_M
                candidates.add((math.floor(x / g), math.floor(y / g)))
        reach = int(math.ceil(SPOT_NEAR_M / g))
        for key in self.walls:
            wx, wy = (key[0] + 0.5) * WALL_CELL_M, (key[1] + 0.5) * WALL_CELL_M
            if not self._wall_at(wx, wy):
                continue
            cx, cy = math.floor(wx / g), math.floor(wy / g)
            for i in range(cx - reach, cx + reach + 1):
                for j in range(cy - reach, cy + reach + 1):
                    candidates.add((i, j))
        scored: list[Refuge] = []
        for i, j in candidates:
            x, y = (i + 0.5) * g, (j + 0.5) * g
            if self._ccell(x, y) not in self.floor:
                continue  # floor it has never been on or seen: maybe the far side of a wall
            if self._wall_near(x, y, BODY_ROOM_M):
                continue  # no room for its body there
            value, cov, encl, out = self.value(x, y)
            if value < SHELTER_MIN:
                continue
            seen = max(
                self.cover.get(self._ccell(x, y), _Cover(seen_s=-1e9)).seen_s,
                max(
                    (self.wall_seen.get(self._wcell(x + dx, y + dy), -1e9) for dx, dy in _RING),
                    default=-1e9,
                ),
            )
            scored.append(Refuge(x, y, value, cov, encl, out, seen))
        scored.sort(key=lambda r: (-r.value, r.x, r.y))
        kept: list[Refuge] = []
        for r in scored:
            if all(math.hypot(r.x - k.x, r.y - k.y) >= REFUGE_APART_M for k in kept):
                kept.append(r)
            if len(kept) >= REFUGE_KEEP:
                break
        self._refuges = kept
        return kept

    def choose(
        self,
        odom: tuple[float, float, float],
        t_s: float,
        threat_heading: float | None,
    ) -> tuple[Refuge, float] | None:
        """The refuge to run to (and its score): good, near, not toward the scare, not one it
        failed to reach lately, not so old its odometry has drifted away from it."""
        best: tuple[Refuge, float] | None = None
        for r in self.refuges(t_s):
            dx, dy = r.x - odom[0], r.y - odom[1]
            dist = math.hypot(dx, dy)
            if dist > FLEE_MAX_M:
                continue
            score = r.value - DIST_COST * dist
            score -= DRIFT_COST_PER_MIN * max(0.0, t_s - r.seen_s) / 60.0
            if threat_heading is not None and dist > 0.3:
                toward = abs(_wrap(math.atan2(dy, dx) - threat_heading))
                if toward < THREAT_CONE:
                    score -= THREAT_COST * (1.0 - toward / THREAT_CONE)
            for bx, by, when in self.bad:
                if math.hypot(bx - r.x, by - r.y) < 0.4:
                    score -= BAD_COST * math.exp(-(t_s - when) / BAD_TAU_S)
            if best is None or score > best[1]:
                best = (r, score)
        return best

    def known(self, odom: tuple[float, float, float], t_s: float) -> float:
        """0..1: a shelter it could run to (the best one's value, 0 when none within reach)."""
        best = 0.0
        for r in self.refuges(t_s):
            if math.hypot(r.x - odom[0], r.y - odom[1]) <= FLEE_MAX_M:
                best = max(best, r.value)
        return best

    def view(self) -> dict[str, Any]:
        return {
            "version": SHELTER_VERSION,
            "looks": self.looks,
            "wall_cells": sum(1 for k in self.walls if self.walls[k] >= WALL_MIN),
            "cover_cells": sum(1 for c in self.cover.values() if c.hits >= 2.0),
            "refuges": [
                {
                    "x": round(r.x, 2),
                    "y": round(r.y, 2),
                    "value": round(r.value, 2),
                    "cover": round(r.cover, 2),
                    "walls": round(r.enclosure, 2),
                }
                for r in self._refuges
            ],
        }


_RING = tuple(
    (0.3 * math.cos(2.0 * math.pi * k / 8), 0.3 * math.sin(2.0 * math.pi * k / 8)) for k in range(8)
)


@dataclass(slots=True)
class HideRun:
    """One flight to a shelter and the hiding there (see the module doc). ``phase``: ``idle``
    → ``go`` → ``in`` → ``out`` (or ``failed``)."""

    boldness: float = 0.5
    seed: int = 0
    phase: str = "idle"
    refuge: Refuge | None = None
    started_s: float = 0.0
    in_s: float = 0.0
    last_scare_s: float = -1e9
    next_peek_s: float = 0.0
    peeks: int = 0
    best_dist: float = 9.9
    progress_s: float = 0.0
    threat_heading: float | None = None
    fear0: float = 0.0
    log: list[str] = field(default_factory=list)
    _rng: int = 0x5EED

    def start(
        self,
        t_s: float,
        odom: tuple[float, float, float],
        shelters: ShelterMap,
        threat_heading: float | None,
        fear: float,
    ) -> bool:
        """Begin a flight: pick the refuge. False when it knows none within reach worth going to
        (a score under :data:`CHOOSE_MIN`: far, toward the scare, or one it failed to reach)."""
        choice = shelters.choose(odom, t_s, threat_heading)
        if choice is None or choice[1] < CHOOSE_MIN:
            self.phase = "idle"
            return False
        self.refuge = choice[0]
        self.phase = "go"
        self.started_s = t_s
        self.in_s = 0.0
        self.peeks = 0
        self.best_dist = choice[0].bearing_range(odom)[1]
        self.progress_s = t_s
        self.threat_heading = threat_heading
        self.fear0 = fear
        self.last_scare_s = t_s
        self._rng = (self.seed * 0x9E3779B97F4A7C15 + int(t_s * 1000)) & 0xFFFFFFFFFFFFFFFF
        self.log.append(f"go {self.refuge.x:.2f},{self.refuge.y:.2f} v{self.refuge.value:.2f}")
        del self.log[:-8]
        return True

    def goal(self, odom: tuple[float, float, float]) -> tuple[float, float] | None:
        if self.refuge is None or self.phase != "go":
            return None
        return self.refuge.bearing_range(odom)

    def scared(self, t_s: float) -> None:
        self.last_scare_s = t_s

    def step(
        self,
        t_s: float,
        odom: tuple[float, float, float],
        *,
        fear: float,
        blocked: bool,
        roof: float | None,
        enclosure_here: float,
        shelters: ShelterMap,
        free: bool = True,
    ) -> str:
        """Advance the run; returns the phase. ``free``: its legs are its own now (no reflex
        holding the body, not getting up): only then does a lack of progress count against the
        refuge."""
        if self.phase == "go" and self.refuge is not None:
            _, dist = self.refuge.bearing_range(odom)
            if dist < self.best_dist - PROGRESS_M or not free:
                self.best_dist = min(self.best_dist, dist)
                self.progress_s = t_s
            sensed = (roof is not None and roof <= ROOF_M) or enclosure_here >= SENSED_ENCLOSE
            if dist <= ARRIVE_M or (dist <= ARRIVE_SENSED_M and (sensed or blocked)):
                self.phase = "in"
                self.in_s = t_s
                self.next_peek_s = t_s + PEEK_FIRST_S + self._draw(2.0)
                self.log.append(f"in {dist:.2f}")
                del self.log[:-8]
            elif t_s - self.progress_s > STUCK_S:
                shelters.mark_bad(self.refuge.x, self.refuge.y, t_s)
                self.phase = "failed"
                self.log.append("failed")
                del self.log[:-8]
        elif self.phase == "in":
            calm = t_s - self.last_scare_s >= CALM_SINCE_S
            low = fear <= EXIT_FEAR[0] + EXIT_FEAR[1] * (1.0 - self.boldness)
            if t_s - self.in_s >= MIN_IN_S and calm and low:
                self.phase = "out"
                self.log.append(f"out {t_s - self.in_s:.1f}s {self.peeks} peeks")
                del self.log[:-8]
        return self.phase

    def peek_due(self, t_s: float, fear: float) -> bool:
        """Whether it peeks out now (then the next peek is drawn: sooner as it calms)."""
        if self.phase != "in" or t_s < self.next_peek_s:
            return False
        self.peeks += 1
        afraid = _clip(fear / max(0.05, hide_threshold(self.boldness)))
        median = PEEK_MEDIAN_S[0] + PEEK_MEDIAN_S[1] * afraid
        self.next_peek_s = t_s + median * math.exp(PEEK_SIGMA * self._gauss())
        return True

    def peek_bearing(self, odom: tuple[float, float, float]) -> float:
        """Where it peeks (rad from the trunk now): toward the scare if it knew where it came
        from, else the shelter's way out — as far as the head turns (±1.0 rad)."""
        if self.threat_heading is not None:
            b = _wrap(self.threat_heading - odom[2])
        elif self.refuge is not None:
            b = _wrap(self.refuge.open_heading - odom[2])
        else:
            b = 0.0
        return _clip(b, -1.0, 1.0)

    def _next(self) -> int:
        self._rng = (self._rng * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return self._rng >> 11

    def _draw(self, span: float) -> float:
        return span * (self._next() / float(1 << 53))

    def _gauss(self) -> float:
        return sum(self._next() / float(1 << 53) for _ in range(4)) - 2.0

    def view(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "refuge": None
            if self.refuge is None
            else {
                "x": round(self.refuge.x, 2),
                "y": round(self.refuge.y, 2),
                "value": round(self.refuge.value, 2),
                "cover": round(self.refuge.cover, 2),
                "walls": round(self.refuge.enclosure, 2),
            },
            "peeks": self.peeks,
            "log": list(self.log[-4:]),
        }


def shelter_words(refuge: Refuge | None) -> str:
    """What kind of shelter it is, in plain words, from what the duck itself sensed there."""
    if refuge is None:
        return msg("neuro.incoming.incoming_skill_shelter.shelter_words.dac0f9")
    if refuge.cover >= 0.3 and refuge.enclosure < 0.3:
        return msg("neuro.incoming.incoming_skill_shelter.shelter_words.d6457f")
    if refuge.cover >= 0.3:
        return msg("neuro.incoming.incoming_skill_shelter.shelter_words.f0f67d")
    return msg("neuro.incoming.incoming_skill_shelter.shelter_words.6c8cca")


def walls_from(points: Iterable[tuple[float, float]], shelters: ShelterMap, t_s: float) -> None:
    """(tests) lay known wall points (odometry frame) straight into the map."""
    for x, y in points:
        key = ShelterMap._wcell(x, y)
        shelters.walls[key] = 1.0
        shelters.wall_seen[key] = t_s


__all__ = [
    "HideRun",
    "ROOF_M",
    "ROOF_OPEN",
    "Refuge",
    "SHELTER_VERSION",
    "ShelterMap",
    "hide_threshold",
    "hide_urge",
    "shelter_words",
    "walls_from",
]
