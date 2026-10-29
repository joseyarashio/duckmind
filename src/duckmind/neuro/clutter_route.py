"""A route through what the duck remembers (for ``clutter.py``): a small grid search over its own wall memory.

No world truth and no map of the home: the grid is built, each time, from what the duck itself holds in the odometry frame:

* **in the way**: the wall memory's points (the ToF's returns, the camera's floor line, the bumps), the places it was blocked, and the
  places a try just failed — each a disc the body cannot enter (:data:`BLOCK_R`), with a ring round it that costs more (:data:`RING_R`,
  so a route keeps to the middle of a gap);
* **known to be passable**: the ground it WALKED over (its trail) and the floor it has SEEN clear (the floor line's free rays): a
  cell there costs 1, a cell nobody has looked at costs :data:`UNKNOWN_COST` (the way it has not seen is a guess, not a wall).

The search is the plain cheapest-path (Dijkstra, 8 neighbours, 0.1 m cells, a 6.4 m window round the duck) to the set of places the
escape wants to reach: ``out_m`` or more from the knot's centre, and among those the ones nearest to where it was walking to
(``goal``) when it had somewhere to go. Pure and deterministic.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

RES = 0.10
N = 80  # cells a side: an 8 m window (centred where the escape put it: see ``build``)
BLOCK_R = 0.22  # the body (half a width, plus the margin the waddle needs) cannot be nearer than this to a point
RING_R = 0.38  # … and a route is dearer within this
RING_COST = 0.9
UNKNOWN_COST = 2.5
NEAR_START_M = 0.40  # an obstacle point this near the duck blocks only half the usual radius
START_FREE_R = (
    0.20  # the cells this near the duck are free (it stands where the memory says a thing is)
)
SQRT2 = math.sqrt(2.0)


@dataclass(frozen=True, slots=True)
class Route:
    points: tuple[
        tuple[float, float], ...
    ]  # world (odometry frame) points from the duck's cell to the target
    cost: float
    target: tuple[float, float]
    complete: bool = True  # the end is the place itself (or out of the knot), not just the nearest the window and the memory allow


@dataclass(slots=True)
class Maps:
    cx: float
    cy: float
    blocked: list[bool]
    cost: list[float]
    unknown: float = UNKNOWN_COST  # what a cell nobody has looked at costs (a cell dearer than this is in a ring: no place to stop)

    def cell(self, x: float, y: float) -> tuple[int, int]:
        return int(math.floor((x - self.cx) / RES + N / 2)), int(
            math.floor((y - self.cy) / RES + N / 2)
        )

    def centre(self, i: int, j: int) -> tuple[float, float]:
        return self.cx + (i + 0.5 - N / 2) * RES, self.cy + (j + 0.5 - N / 2) * RES

    def inside(self, i: int, j: int) -> bool:
        return 0 <= i < N and 0 <= j < N


def _disc(m: Maps, x: float, y: float, r: float):
    i0, j0 = m.cell(x - r, y - r)
    i1, j1 = m.cell(x + r, y + r)
    for i in range(max(0, i0), min(N - 1, i1) + 1):
        for j in range(max(0, j0), min(N - 1, j1) + 1):
            cx, cy = m.centre(i, j)
            d = math.hypot(cx - x, cy - y)
            if d <= r:
                yield i, j, d


SHADOW_M = 0.35  # a thing it has seen the front of has a back: this much behind each point (away from the duck) is in the way too


def build(
    start: tuple[float, float],
    obstacles: list[tuple[float, float]],
    discs: list[tuple[float, float, float]],
    known: list[tuple[float, float]],
    known_r: float = 0.16,
    centre: tuple[float, float] | None = None,
    shadow: float = SHADOW_M,
    goal: tuple[float, float] | None = None,
    goal_free_r: float = 0.5,
    block_r: float = BLOCK_R,
) -> Maps:
    """The grid (a window of N cells a side centred on ``centre``, default ``start``): ``obstacles`` points (BLOCK_R discs), ``discs``
    (x, y, r) extra blocked places, ``known`` points of ground walked / seen clear (corridors of ``known_r``)."""
    cx, cy = start if centre is None else centre
    m = Maps(cx, cy, [False] * (N * N), [UNKNOWN_COST] * (N * N))
    for x, y in known:
        for i, j, _d in _disc(m, x, y, known_r):
            m.cost[i * N + j] = 1.0
    for x, y in obstacles:
        if math.hypot(x - start[0], y - start[1]) <= NEAR_START_M:
            # (what is touching the duck now: it has to be able to get away from it; only the thing itself is in the way)
            for i, j, _d in _disc(m, x, y, block_r * 0.5):
                m.blocked[i * N + j] = True
            continue
        ring_r = block_r + (RING_R - BLOCK_R)
        for i, j, d in _disc(m, x, y, ring_r):
            k = i * N + j
            if d <= block_r:
                m.blocked[k] = True
            else:
                m.cost[k] += RING_COST * (1.0 - (d - block_r) / (ring_r - block_r))
        # (a point at the front of a thing: its back is behind it, seen from here)
        dx, dy = x - start[0], y - start[1]
        n = math.hypot(dx, dy)
        if n > 0.4 and shadow > 0.0:
            ux, uy = dx / n, dy / n
            for q in (0.15, 0.3):
                if q <= shadow:
                    for i, j, d in _disc(m, x + ux * q, y + uy * q, block_r * 0.8):
                        m.blocked[i * N + j] = True
    for x, y, r in discs:
        for i, j, _d in _disc(m, x, y, r):
            m.blocked[i * N + j] = True
    for i, j, _d in _disc(m, start[0], start[1], START_FREE_R):
        m.blocked[i * N + j] = False
    if goal is not None:
        # (the place it wants to be at has things round it — a dock on a wall, a toy among toys, a person: they are not in its way)
        for i, j, _d in _disc(m, goal[0], goal[1], goal_free_r):
            m.blocked[i * N + j] = False
    return m


def search(
    m: Maps,
    start: tuple[float, float],
    origin: tuple[float, float],
    out_m: float,
    goal: tuple[float, float] | None,
    goal_w: float = 3.0,
    goal_tol: float = 0.8,
) -> Route | None:
    """With a ``goal``: the cheapest way to it (to a cell within ``goal_tol`` of it). Else, or when the memory leaves no way there: the
    cheapest way to a cell at least ``out_m`` from ``origin`` — among those the one that costs least plus ``goal_w`` per metre it still
    is from ``goal`` — or None."""
    si, sj = m.cell(*start)
    if not m.inside(si, sj):
        return None
    inf = float("inf")
    dist = [inf] * (N * N)
    parent = [-1] * (N * N)
    s = si * N + sj
    dist[s] = 0.0
    heap = [(0.0, s)]
    while heap:
        d, k = heapq.heappop(heap)
        if d > dist[k]:
            continue
        i, j = divmod(k, N)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            a, b = i + di, j + dj
            if not (0 <= a < N and 0 <= b < N):
                continue
            kk = a * N + b
            if m.blocked[kk]:
                continue
            if di and dj:
                # (a diagonal may not cut a blocked corner)
                if m.blocked[(i + di) * N + j] or m.blocked[i * N + (j + dj)]:
                    continue
                step = SQRT2
            else:
                step = 1.0
            nd = d + step * RES * 0.5 * (m.cost[k] + m.cost[kk])
            if nd < dist[kk]:
                dist[kk] = nd
                parent[kk] = k
                heapq.heappush(heap, (nd, kk))
    best = None
    best_score = inf
    complete = True
    if goal is not None:
        # somewhere to go (a goal outside the window is taken at the window's edge, in its direction): the cheapest way to the place
        # itself when the memory leaves one, else to the reachable cell that costs least plus ``goal_w`` per metre it still is from it
        gx, gy = goal
        half = (N / 2 - 3) * RES
        if abs(gx - m.cx) > half or abs(gy - m.cy) > half:
            shrink = half / max(abs(gx - m.cx), abs(gy - m.cy))
            gx, gy = m.cx + (gx - m.cx) * shrink, m.cy + (gy - m.cy) * shrink
        for k in range(N * N):
            if dist[k] == inf or m.blocked[k] or m.cost[k] > m.unknown + 1e-9:
                continue  # (a cell in a ring is no place to stop)
            x, y = m.centre(*divmod(k, N))
            near = math.hypot(x - gx, y - gy)
            score = dist[k] if near <= goal_tol else dist[k] + goal_w * near + 1000.0
            if score < best_score:
                best_score, best = score, k
        if best is not None:
            x, y = m.centre(*divmod(best, N))
            complete = math.hypot(x - goal[0], y - goal[1]) <= goal_tol
    else:
        # nowhere to go: out of the knot, the cheapest place at least ``out_m`` from it
        for k in range(N * N):
            if dist[k] == inf or m.blocked[k] or m.cost[k] > m.unknown + 1e-9:
                continue
            x, y = m.centre(*divmod(k, N))
            if math.hypot(x - origin[0], y - origin[1]) < out_m:
                continue
            if dist[k] < best_score:
                best_score, best = dist[k], k
    if best is None:
        return None
    path = []
    k = best
    while k != -1:
        path.append(m.centre(*divmod(k, N)))
        k = parent[k]
    path.reverse()
    return Route(tuple(path), dist[best], path[-1], complete)


def route_cost(m: Maps, points: tuple[tuple[float, float], ...], from_index: int) -> float:
    """What the rest of an old route (from ``from_index``) costs on these maps; infinity when it runs into a blocked cell."""
    total = 0.0
    prev = None
    for x, y in points[from_index:]:
        i, j = m.cell(x, y)
        if not m.inside(i, j):
            break
        k = i * N + j
        if m.blocked[k]:
            return float("inf")
        if prev is not None:
            total += math.hypot(x - prev[0], y - prev[1]) * m.cost[k]
        prev = (x, y)
    return total


def lookahead(
    points: tuple[tuple[float, float], ...], pos: tuple[float, float], dist: float
) -> tuple[int, tuple[float, float]]:
    """(the index of the route point nearest the duck, the point ``dist`` further along the route from there — the last point when
    the route is shorter)."""
    near = min(
        range(len(points)), key=lambda k: math.hypot(points[k][0] - pos[0], points[k][1] - pos[1])
    )
    acc = 0.0
    prev = points[near]
    for k in range(near + 1, len(points)):
        acc += math.hypot(points[k][0] - prev[0], points[k][1] - prev[1])
        prev = points[k]
        if acc >= dist:
            return near, points[k]
    return near, points[-1]
