"""The home's floor as the benchmark knows it (truth, from the layout): what stands where, how much room a body has there, and
a body-wide route between two places.

Used for two things only, both outside the duck: making and labelling a click set (is a place reachable, what is on the
straight line to it, how much room has the start), and the truth-planner arm of the headless runner (:mod:`.truthplan`).
Nothing here is ever given to the brain.

The grid is :data:`CELL_M` wide. A cell is ``WALL`` (a wall, a skirting board, outside every room), ``FURNITURE`` (a fixed
piece whose body is between :data:`STEP_MM` and :data:`BODY_TOP_MM` above the floor), ``THING`` (a movable thing that tall),
``LOW`` (something lower than :data:`STEP_MM`: a door sill, a cable, a sock — a body can step over it, at a cost) or ``FREE``.
Rugs and mats are floor. Every outline is grown by half a cell, so a thin thing is never lost between cell centres.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

CELL_M = 0.02
BODY_RADIUS_M = (
    0.10  # the body's half width (0.08) plus 2 cm: what a body-wide route keeps from anything solid
)
BODY_TOP_MM = 300.0  # the duck is lower than this: what starts above it (a table top, a chair seat) is not in the way
STEP_MM = (
    25.0  # lower than this a body steps over (sills of 5 to 20 mm, cables, a sock, a magazine)
)
CLEAR_CAP_M = 0.60  # room is measured up to this far
SOFT_M = 0.30  # nearer than this to something solid a route costs more (the reflexes slow and bend a walk there)
SOFT_COST = 3.0  # … up to this many times, right at the body's width
LOW_COST_PER_MM = 0.2  # stepping over something low costs 1 + this × its height (a 20 mm sill: five times the way)
PAD_M = 0.10

FREE, LOW, THING, FURNITURE, WALL = 0, 1, 2, 3, 4
KIND_NAMES = ("clear", "low", "thing", "furniture", "wall")
WALL_KINDS = frozenset({"wall", "skirting"})
_SQRT2 = math.sqrt(2.0)


def _rot(x: Any, y: Any, rad: float) -> tuple[Any, Any]:
    c, s = math.cos(rad), math.sin(rad)
    return c * x - s * y, s * x + c * y


class FloorMap:
    """The floor of one home layout. ``kind``/``low_mm``/``owner`` are (ny, nx) arrays; ``names[owner]`` says what is there."""

    def __init__(self, layout: Any, cell: float = CELL_M) -> None:
        self.cell = float(cell)
        rects = [tuple(float(v) / 1000.0 for v in r.rect_mm) for r in layout.rooms]
        self.rooms = [str(r.id) for r in layout.rooms]
        self.x0 = min(r[0] for r in rects) - PAD_M
        self.y0 = min(r[1] for r in rects) - PAD_M
        self.nx = int(math.ceil((max(r[2] for r in rects) + PAD_M - self.x0) / self.cell))
        self.ny = int(math.ceil((max(r[3] for r in rects) + PAD_M - self.y0) / self.cell))
        xs = self.x0 + (np.arange(self.nx) + 0.5) * self.cell
        ys = self.y0 + (np.arange(self.ny) + 0.5) * self.cell
        self._xs, self._ys = xs, ys
        self.room = np.full((self.ny, self.nx), -1, dtype=np.int8)
        for i, (a, b, c, d) in enumerate(rects):
            inside = ((ys >= b) & (ys < d))[:, None] & ((xs >= a) & (xs < c))[None, :]
            self.room[inside & (self.room < 0)] = i
        self.names: list[str] = [""]
        self._static_kind = np.where(self.room >= 0, FREE, WALL).astype(np.uint8)
        self._static_low = np.zeros((self.ny, self.nx), dtype=np.float32)
        self._static_owner = np.zeros((self.ny, self.nx), dtype=np.int16)
        for e in layout.elements:
            if not e.collide or e.support:
                continue
            z0, z1 = e.z_range()
            if z1 <= 0.0 or z0 >= BODY_TOP_MM:
                continue
            kind = LOW if z1 < STEP_MM else (WALL if e.kind in WALL_KINDS else FURNITURE)
            self._stamp(
                self._static_kind,
                self._static_low,
                self._static_owner,
                self._element_mask(e),
                kind,
                z1,
                str(e.id),
            )
        # movable things: (id, layout x, layout y, footprint relative to that point, class, height)
        self.things: list[
            tuple[str, float, float, tuple[float, float, float, float], int, float]
        ] = []
        for t in layout.things:
            a, b, c, d = (float(v) / 1000.0 for v in t.footprint())
            x, y = float(t.pos_mm[0]) / 1000.0, float(t.pos_mm[1]) / 1000.0
            h = float(t.height_mm)
            self.things.append(
                (str(t.id), x, y, (a - x, b - y, c - x, d - y), LOW if h < STEP_MM else THING, h)
            )
        self.kind = self._static_kind
        self.low_mm = self._static_low
        self.owner = self._static_owner
        self.clear = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.at: dict[str, tuple[float, float]] = {}
        self.set_things({})

    # ── building it ──

    def _span(self, lo: float, hi: float, origin: float, n: int) -> tuple[int, int]:
        a = int(math.floor((lo - origin) / self.cell - 0.5))
        b = int(math.ceil((hi - origin) / self.cell + 0.5))
        return max(0, a), min(n, b + 1)

    def _element_mask(self, e: Any) -> tuple[slice, slice, np.ndarray]:
        a, b, c, d = (float(v) / 1000.0 for v in e.footprint())
        i0, i1 = self._span(a, c, self.x0, self.nx)
        j0, j1 = self._span(b, d, self.y0, self.ny)
        xs, ys = self._xs[i0:i1][None, :], self._ys[j0:j1][:, None]
        grow = 0.5 * self.cell
        x, y = float(e.pos_mm[0]) / 1000.0, float(e.pos_mm[1]) / 1000.0
        if e.shape == "capsule" and len(e.fromto_mm) == 6:
            ax, ay, _, bx, by, _ = (float(v) / 1000.0 for v in e.fromto_mm)
            ux, uy = bx - ax, by - ay
            ln2 = ux * ux + uy * uy or 1e-12
            s = np.clip(((xs - ax) * ux + (ys - ay) * uy) / ln2, 0.0, 1.0)
            mask = (
                np.hypot(xs - (ax + s * ux), ys - (ay + s * uy))
                <= float(e.radius_mm) / 1000.0 + grow
            )
        elif e.shape in ("cylinder", "sphere", "frustum"):
            r = max(float(e.radius_mm), float(e.radius_bottom_mm)) / 1000.0
            mask = np.hypot(xs - x, ys - y) <= r + grow
        else:
            lx, ly = _rot(xs - x, ys - y, -float(e.yaw_mrad) / 1000.0)
            mask = (np.abs(lx) <= float(e.size_mm[0]) / 2000.0 + grow) & (
                np.abs(ly) <= float(e.size_mm[1]) / 2000.0 + grow
            )
        return slice(j0, j1), slice(i0, i1), mask

    def _stamp(
        self,
        kind: np.ndarray,
        low: np.ndarray,
        owner: np.ndarray,
        where: tuple[slice, slice, np.ndarray],
        k: int,
        top_mm: float,
        name: str,
    ) -> None:
        sj, si, mask = where
        self.names.append(name)
        idx = len(self.names) - 1
        sub = kind[sj, si]
        take = mask & (sub < k) if k > LOW else mask & (sub <= LOW)
        if k == LOW:
            lw = low[sj, si]
            lw[take] = np.maximum(lw[take], top_mm)
        sub[take] = np.maximum(sub[take], k)
        ow = owner[sj, si]
        ow[take] = idx

    def set_things(self, at: Mapping[str, tuple[float, float]]) -> None:
        """Put the movable things where they are now (``at``: id → world x, y in m; a thing not named is where the layout
        put it) and work out the room a body has everywhere."""
        kind = self._static_kind.copy()
        low = self._static_low.copy()
        owner = self._static_owner.copy()
        del self.names[1 + int(self._static_owner.max()) :]
        self.at = {}
        for tid, x, y, rel, k, h in self.things:
            px, py = at.get(tid, (x, y))
            self.at[tid] = (float(px), float(py))
            i0, i1 = self._span(px + rel[0], px + rel[2], self.x0, self.nx)
            j0, j1 = self._span(py + rel[1], py + rel[3], self.y0, self.ny)
            if i0 >= i1 or j0 >= j1:
                self.names.append(tid)
                continue
            mask = np.ones((j1 - j0, i1 - i0), dtype=bool)
            self._stamp(kind, low, owner, (slice(j0, j1), slice(i0, i1), mask), k, h, tid)
        self.kind, self.low_mm, self.owner = kind, low, owner
        self.clear = clearance(kind >= THING, self.cell)

    # ── reading it ──

    def cell_of(self, x: float, y: float) -> tuple[int, int]:
        """(row, column) of a world point, clamped into the grid."""
        i = min(self.nx - 1, max(0, int(math.floor((x - self.x0) / self.cell))))
        j = min(self.ny - 1, max(0, int(math.floor((y - self.y0) / self.cell))))
        return j, i

    def xy_of(self, j: int, i: int) -> tuple[float, float]:
        return self.x0 + (i + 0.5) * self.cell, self.y0 + (j + 0.5) * self.cell

    def room_at(self, x: float, y: float) -> str | None:
        j, i = self.cell_of(x, y)
        r = int(self.room[j, i])
        return self.rooms[r] if r >= 0 else None

    def kind_at(self, x: float, y: float) -> int:
        j, i = self.cell_of(x, y)
        return int(self.kind[j, i])

    def clear_at(self, x: float, y: float) -> float:
        j, i = self.cell_of(x, y)
        return float(self.clear[j, i])

    def passable(self, radius: float = BODY_RADIUS_M) -> np.ndarray:
        """Where a body ``radius`` wide can be: nothing solid nearer than that."""
        return (self.kind < THING) & (self.clear >= radius)

    def step_cost(self, radius: float = BODY_RADIUS_M) -> np.ndarray:
        """What a metre of the way costs in each cell: 1 in the open, more close to something solid and over something low."""
        near = np.clip((SOFT_M - self.clear) / max(1e-6, SOFT_M - radius), 0.0, 1.0)
        return (1.0 + SOFT_COST * near * near) * (1.0 + LOW_COST_PER_MM * self.low_mm)

    def line(
        self, a: Sequence[float], b: Sequence[float], half: float = 0.07, skip: float = 0.12
    ) -> dict[str, Any]:
        """What is on the straight line from ``a`` to ``b`` for a body-wide strip (truth, for labelling): the first solid
        thing (``kind`` thing | furniture | wall, its ``id`` and how far along), else ``low`` when the strip only crosses
        something low, else ``clear``."""
        ux, uy = b[0] - a[0], b[1] - a[1]
        ln = math.hypot(ux, uy) or 1e-9
        px, py = -uy / ln, ux / ln
        n = max(2, int(ln / self.cell))
        low: dict[str, Any] | None = None
        for k in range(1, n + 1):
            s = k / n
            if s * ln < skip:
                continue
            for off in (-half, 0.0, half):
                j, i = self.cell_of(a[0] + ux * s + px * off, a[1] + uy * s + py * off)
                kind = int(self.kind[j, i])
                if kind >= THING:
                    return {
                        "kind": KIND_NAMES[kind],
                        "id": self.names[int(self.owner[j, i])] or "outside",
                        "at_m": round(s * ln, 2),
                    }
                if kind == LOW and low is None:
                    low = {
                        "kind": "low",
                        "id": self.names[int(self.owner[j, i])],
                        "at_m": round(s * ln, 2),
                    }
        return low or {"kind": "clear", "id": None, "at_m": None}


def clearance(solid: np.ndarray, cell: float = CELL_M, cap: float = CLEAR_CAP_M) -> np.ndarray:
    """Distance (m, up to ``cap``) from every cell's centre to the nearest solid cell's centre; outside the grid is solid."""
    ny, nx = solid.shape
    w = int(math.ceil(cap / cell))
    rows = np.arange(ny)[:, None]
    above = np.maximum.accumulate(np.where(solid, rows, -(10**6)), axis=0)
    below = np.minimum.accumulate(np.where(solid, rows, 10**6)[::-1], axis=0)[::-1]
    g = np.minimum(np.minimum(rows - above, below - rows), np.minimum(rows + 1, ny - rows))
    g = np.minimum(g, w).astype(np.float32)
    gp = np.pad(g, ((0, 0), (w, w)), constant_values=0.0)
    d2 = np.full((ny, nx), float(w * w), dtype=np.float32)
    for k in range(-w, w + 1):
        cand = gp[:, w + k : w + k + nx]
        np.minimum(d2, cand * cand + float(k * k), out=d2)
    return np.sqrt(d2) * cell


def cost_to_go(
    passable: np.ndarray,
    step_cost: np.ndarray,
    goal: tuple[int, int],
    cell: float = CELL_M,
    limit: float = math.inf,
) -> np.ndarray:
    """The cheapest way (in cost-metres) from every passable cell to ``goal`` (row, column), eight-connected; ``inf`` where
    there is none (or it costs more than ``limit``). The grid's border must not be passable."""
    ny, nx = passable.shape
    n = ny * nx
    ok = passable.ravel().tolist()
    sc = (step_cost.ravel() * (0.5 * cell)).tolist()
    dist = [math.inf] * n
    g = goal[0] * nx + goal[1]
    if not ok[g]:
        return np.full((ny, nx), math.inf, dtype=np.float32)
    dist[g] = 0.0
    heap = [(0.0, g)]
    steps = (
        (1, 1.0),
        (-1, 1.0),
        (nx, 1.0),
        (-nx, 1.0),
        (nx + 1, _SQRT2),
        (nx - 1, _SQRT2),
        (1 - nx, _SQRT2),
        (-nx - 1, _SQRT2),
    )
    pop, push = heapq.heappop, heapq.heappush
    while heap:
        d, u = pop(heap)
        if d > dist[u]:
            continue
        if d > limit:
            break
        su = sc[u]
        for off, w in steps:
            v = u + off
            if ok[v]:
                nd = d + w * (su + sc[v])
                if nd < dist[v]:
                    dist[v] = nd
                    push(heap, (nd, v))
    out = np.array(dist, dtype=np.float32).reshape(ny, nx)
    if limit < math.inf:
        out[out > limit] = math.inf
    return out


_NEIGHBOURS = ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1))


def descend(
    field: np.ndarray, start: tuple[int, int], max_cells: int = 100000
) -> list[tuple[int, int]]:
    """The cells of the cheapest way from ``start`` down ``field`` to its goal (empty when there is none)."""
    j, i = start
    if not math.isfinite(float(field[j, i])):
        return []
    path = [(j, i)]
    here = float(field[j, i])
    while here > 0.0 and len(path) < max_cells:
        best = None
        for dj, di in _NEIGHBOURS:
            v = float(field[j + dj, i + di])
            if v < here and (best is None or v < best[0]):
                best = (v, j + dj, i + di)
        if best is None:
            break
        here, j, i = best
        path.append((j, i))
    return path


def path_length(path: Sequence[tuple[int, int]], cell: float = CELL_M) -> float:
    return (
        sum(math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(path, path[1:], strict=False))
        * cell
    )


def nearest_passable(
    passable: np.ndarray, at: tuple[int, int], reach_cells: int, weight: np.ndarray | None = None
) -> tuple[int, int] | None:
    """The passable cell nearest ``at`` within ``reach_cells`` (``weight``: only cells where it is finite), or None."""
    ny, nx = passable.shape
    j, i = at
    j0, j1, i0, i1 = (
        max(0, j - reach_cells),
        min(ny, j + reach_cells + 1),
        max(0, i - reach_cells),
        min(nx, i + reach_cells + 1),
    )
    sub = passable[j0:j1, i0:i1]
    if weight is not None:
        sub = sub & np.isfinite(weight[j0:j1, i0:i1])
    if not sub.any():
        return None
    jj, ii = np.nonzero(sub)
    d2 = (jj + j0 - j) ** 2 + (ii + i0 - i) ** 2
    k = int(np.argmin(d2))
    if d2[k] > reach_cells * reach_cells:
        return None
    return int(jj[k] + j0), int(ii[k] + i0)


def line_free(passable: np.ndarray, fm: FloorMap, a: Sequence[float], b: Sequence[float]) -> bool:
    """Whether a body can go straight from ``a`` to ``b`` (every cell on the way passable)."""
    ln = math.hypot(b[0] - a[0], b[1] - a[1])
    n = max(1, int(ln / (0.5 * fm.cell)))
    for k in range(n + 1):
        s = k / n
        j, i = fm.cell_of(a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s)
        if not passable[j, i]:
            return False
    return True


__all__ = [
    "BODY_RADIUS_M",
    "BODY_TOP_MM",
    "CELL_M",
    "FREE",
    "FURNITURE",
    "KIND_NAMES",
    "LOW",
    "STEP_MM",
    "THING",
    "WALL",
    "FloorMap",
    "clearance",
    "cost_to_go",
    "descend",
    "line_free",
    "nearest_passable",
    "path_length",
]
