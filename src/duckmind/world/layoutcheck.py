"""Does a layout the editor made make a home the duck can live in?  (owner 2026-10-10: layout editor.)

:func:`check` takes a layout as it is written in a file (a home JSON with ``placements``) and says, with
codes and ids and no words (the page puts the words in six languages), what is wrong:

* ``bad_layout``: the file does not load (``detail`` carries the loader's words, 繁體中文, as data);
* ``bad_placement``: an item that is not in the library, a number that is not a whole number, a length
  outside what the item allows;
* ``outside``: a placed item sticks out of the home (``id`` is the placement);
* ``overlap``: a placed item is inside a wall, a piece of furniture or another placed item (``id``, ``other``);
* the loader's own generation checks (``thing_in_furniture``, ``thing_overlap``, ``thing_near_nest``,
  ``spawn_blocked`` ...), passed on as they are;
* ``room_unreachable``: a room the duck cannot get to from where it starts (``id`` is the room): the
  furniture closes every way in (a hop over a sill is a way in; a wall, a sofa or a bed is not);
* ``spawn_enclosed``: the place the duck starts is shut in with less than half of the floor.

Every problem is an error except ``warning`` ones (``warn``: true), which do not stop a save or an apply.
Plain Python; no MuJoCo. A grid walk (50 mm cells, the duck 110 mm wide each side) finds what the duck can reach.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from typing import Any

from duckmind.world import homelayout, objlib

CELL_MM = 50
DUCK_HALF_MM = 110  # the walk's clearance each side of the trunk
DUCK_HEIGHT_MM = 290
BLOCK_ABOVE_MM = (
    40  # what the duck can hop over (the library's hop class) is not a wall to the walk
)
ROOM_FLOOR_MIN_CELLS = 40
UNREACHABLE_BELOW = 0.05
ENCLOSED_BELOW = 0.5
TOL_MM = 2.0


def _flat_ok(item: objlib.Item) -> bool:
    """Floor coverings lie under everything: they never conflict with what stands on them."""
    return str(item.raw.get("ekind", "")) == "rug"


def _is_threshold(item: objlib.Item) -> bool:
    return str(item.raw.get("ekind", "")) == "threshold"


def _dist_point_poly(p: tuple[float, float], poly: list[tuple[float, float]]) -> float:
    if _inside(p, poly):
        return 0.0
    best = math.inf
    for k in range(len(poly)):
        a, b = poly[k], poly[(k + 1) % len(poly)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        t = (
            0.0
            if dx == dy == 0
            else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy)))
        )
        best = min(best, math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy)))
    return best


def _inside(p: tuple[float, float], poly: list[tuple[float, float]]) -> bool:
    x, y = p
    inside = False
    n = len(poly)
    for k in range(n):
        x1, y1 = poly[k]
        x2, y2 = poly[(k + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            inside = not inside
    return inside


def _axes(poly: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out = []
    for k in range(len(poly)):
        a, b = poly[k], poly[(k + 1) % len(poly)]
        ex, ey = b[0] - a[0], b[1] - a[1]
        n = math.hypot(ex, ey)
        if n > 1e-9:
            out.append((-ey / n, ex / n))
    return out


def _poly_overlap(
    a: list[tuple[float, float]], b: list[tuple[float, float]], grow: float = 0.0
) -> bool:
    """Separating-axis test of two convex polygons (grown by ``grow`` on every side of ``a``)."""
    for ax, ay in _axes(a) + _axes(b):
        pa = [p[0] * ax + p[1] * ay for p in a]
        pb = [p[0] * ax + p[1] * ay for p in b]
        if max(pa) + grow <= min(pb) or max(pb) <= min(pa) - grow:
            return False
    return True


def _circle_poly(c: tuple[float, float, float], n: int = 12) -> list[tuple[float, float]]:
    return [
        (c[0] + c[2] * math.cos(2 * math.pi * k / n), c[1] + c[2] * math.sin(2 * math.pi * k / n))
        for k in range(n)
    ]


def _element_shape(e: homelayout.Element) -> list[tuple[float, float]] | None:
    """A fixed element's outline on the floor, or None for what the duck is never stopped by."""
    x, y = float(e.pos_mm[0]), float(e.pos_mm[1])
    if e.shape in ("cylinder", "frustum", "sphere"):
        return _circle_poly((x, y, float(max(e.radius_mm, e.radius_bottom_mm))))
    if e.shape == "capsule":
        ax, ay, _, bx, by, _ = e.fromto_mm
        r = float(e.radius_mm)
        dx, dy = bx - ax, by - ay
        n = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / n * r, dx / n * r
        return [(ax + nx, ay + ny), (bx + nx, by + ny), (bx - nx, by - ny), (ax - nx, ay - ny)]
    if e.shape in ("box", "slab", "ellipsoid"):
        sx, sy = float(e.size_mm[0]), float(e.size_mm[1])
        c, s = math.cos(e.yaw_mrad / 1000.0), math.sin(e.yaw_mrad / 1000.0)
        return [
            (x + c * a - s * b, y + s * a + c * b)
            for a, b in ((-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2))
        ]
    return None


def _thing_shapes(t: homelayout.Thing) -> list[tuple[list[tuple[float, float]], float, float]]:
    out = []
    c, s = math.cos(t.yaw_mrad / 1000.0), math.sin(t.yaw_mrad / 1000.0)
    for p in t.parts:
        ox, oy = c * p.offset_mm[0] - s * p.offset_mm[1], s * p.offset_mm[0] + c * p.offset_mm[1]
        cx, cy = t.pos_mm[0] + ox, t.pos_mm[1] + oy
        z0 = float(t.pos_mm[2] + p.offset_mm[2])
        if p.shape in ("sphere", "cylinder"):
            poly = _circle_poly((cx, cy, float(p.radius_mm)))
        else:
            sx, sy = float(p.size_mm[0]), float(p.size_mm[1])
            poly = [
                (cx + c * a - s * b, cy + s * a + c * b)
                for a, b in (
                    (-sx / 2, -sy / 2),
                    (sx / 2, -sy / 2),
                    (sx / 2, sy / 2),
                    (-sx / 2, sy / 2),
                )
            ]
        out.append(
            (
                poly,
                z0,
                z0
                + float(
                    p.size_mm[2]
                    if p.shape == "box"
                    else (2 * p.radius_mm if p.shape == "sphere" else p.size_mm[2])
                ),
            )
        )
    return out


def check(raw: Mapping[str, Any]) -> dict[str, Any]:
    """The problems of a layout file (see the module doc), plus ``reach``: what the duck can get to."""
    problems: list[dict[str, Any]] = []
    placements = list(raw.get("placements") or [])
    bad = objlib.placement_problems(placements)
    if bad:
        return {
            "ok": False,
            "problems": [{"code": "bad_placement", "id": "", "detail": d} for d in bad],
            "reach": None,
        }
    try:
        layout = homelayout.load(dict(raw))
    except (homelayout.LayoutError, objlib.LibraryError, KeyError, TypeError, ValueError) as error:
        return {
            "ok": False,
            "problems": [{"code": "bad_layout", "id": "", "detail": str(error)}],
            "reach": None,
        }
    lib = objlib.library()
    ids = {str(p["id"]): p for p in placements}

    # a placed item's felt parts, against the home and each other
    x0, y0, x1, y1 = layout.bounds_mm
    fixed_solid = [
        (e, _element_shape(e))
        for e in layout.elements
        if e.collide
        and not e.place
        and e.kind not in ("floor", "ceiling", "mat", "window", "skirting")
        and not e.support
    ]
    shapes: dict[str, list[dict[str, Any]]] = {}
    for pid, p in ids.items():
        item = lib.item(str(p["lib"]))
        shapes[pid] = objlib.part_shapes(item, p)
        for shape in shapes[pid]:
            pts = shape["poly"] if "poly" in shape else _circle_poly(shape["circle"])
            if any(not (x0 + 40 <= px <= x1 - 40 and y0 + 40 <= py <= y1 - 40) for px, py in pts):
                problems.append({"code": "outside", "id": pid})
                break
    placed_items = {pid: lib.item(str(p["lib"])) for pid, p in ids.items()}
    for pid, p in ids.items():
        item = placed_items[pid]
        if _flat_ok(item):
            continue
        hit = False
        for shape in shapes[pid]:
            poly = shape["poly"] if "poly" in shape else _circle_poly(shape["circle"])
            z0, z1 = shape["z"]
            for e, outline in fixed_solid:
                if outline is None or e.id.startswith("nest_"):
                    continue
                if _is_threshold(item) and e.kind in ("wall", "skirting"):
                    continue
                ez0, ez1 = e.z_range()
                if min(z1, ez1) - max(z0, ez0) <= TOL_MM:
                    continue
                if _poly_overlap(poly, outline, -TOL_MM):
                    problems.append({"code": "overlap", "id": pid, "other": e.id})
                    hit = True
                    break
            if hit:
                break
    names = list(ids)
    for i, a in enumerate(names):
        ia = placed_items[a]
        for b in names[i + 1 :]:
            ib = placed_items[b]
            if _flat_ok(ia) or _flat_ok(ib):
                continue
            if ia.kind == "thing" and ib.kind == "thing":
                continue  # the loader's own thing checks say it (thing_overlap)
            clash = False
            for sa in shapes[a]:
                pa = sa["poly"] if "poly" in sa else _circle_poly(sa["circle"])
                for sb in shapes[b]:
                    if min(sa["z"][1], sb["z"][1]) - max(sa["z"][0], sb["z"][0]) <= TOL_MM:
                        continue
                    pb = sb["poly"] if "poly" in sb else _circle_poly(sb["circle"])
                    if _poly_overlap(pa, pb, -TOL_MM):
                        clash = True
                        break
                if clash:
                    break
            if clash:
                problems.append({"code": "overlap", "id": a, "other": b})
    for gen in layout.check():
        problems.append({"code": gen.code, "id": gen.subject, "detail": gen.detail})
    reach = reachability(layout)
    for room_id in reach["unreachable_rooms"]:
        problems.append({"code": "room_unreachable", "id": room_id})
    if reach["spawn_enclosed"]:
        problems.append({"code": "spawn_enclosed", "id": "spawn"})
    # (a dedup of the same pair, either way round)
    seen: set[tuple[str, str, str]] = set()
    out = []
    for p in problems:
        key = (p["code"], *sorted((p.get("id", ""), p.get("other", ""))))
        if key in seen:
            continue
        seen.add(key)
        out.append(_names(layout, p))
    return {"ok": not out, "problems": out, "reach": reach}


def _names(layout: homelayout.Layout, problem: dict[str, Any]) -> dict[str, Any]:
    """What a problem's ``id`` and ``other`` are, for the words the page puts to it: the library item and the
    placement of a piece that came from one, else the kind of the home's own piece (a wall, a sill...)."""
    elements = {e.id: e for e in layout.elements}
    things = {t.id: t for t in layout.things}
    placed = {e.place: e.lib for e in layout.elements if e.place}
    out = dict(problem)
    for field, prefix in (("id", ""), ("other", "other_")):
        ident = problem.get(field)
        if not ident:
            continue
        if ident in placed:
            out[prefix + "lib"], out[prefix + "place"] = placed[ident], ident
        elif ident in elements:
            e = elements[ident]
            if e.lib:
                out[prefix + "lib"], out[prefix + "place"] = e.lib, e.place
            else:
                out[prefix + "kind"] = e.kind
        elif ident in things:
            t = things[ident]
            if t.lib:
                out[prefix + "lib"], out[prefix + "place"] = t.lib, t.id
        elif field == "id" and ident in {r.id for r in layout.rooms}:
            out[prefix + "room"] = ident
    return out


def reachability(layout: homelayout.Layout) -> dict[str, Any]:
    """Where the duck can walk from its start: a 50 mm grid, an obstacle is anything fixed that is higher than
    a hop (40 mm) and lower than the duck is tall, and every standing thing that is a block (a basket, a bin).
    ``rooms`` is the share of each room's free floor the walk reaches."""
    x0, y0, x1, y1 = layout.bounds_mm
    nx, ny = int(math.ceil((x1 - x0) / CELL_MM)), int(math.ceil((y1 - y0) / CELL_MM))
    blocked = bytearray(nx * ny)

    def mark(poly: list[tuple[float, float]]) -> None:
        xs, ys = [p[0] for p in poly], [p[1] for p in poly]
        i0 = max(0, int((min(xs) - DUCK_HALF_MM - x0) / CELL_MM))
        i1 = min(nx - 1, int((max(xs) + DUCK_HALF_MM - x0) / CELL_MM))
        j0 = max(0, int((min(ys) - DUCK_HALF_MM - y0) / CELL_MM))
        j1 = min(ny - 1, int((max(ys) + DUCK_HALF_MM - y0) / CELL_MM))
        for j in range(j0, j1 + 1):
            for i in range(i0, i1 + 1):
                if blocked[j * nx + i]:
                    continue
                c = (x0 + (i + 0.5) * CELL_MM, y0 + (j + 0.5) * CELL_MM)
                if _dist_point_poly(c, poly) < DUCK_HALF_MM:
                    blocked[j * nx + i] = 1

    for e in layout.elements:
        if (
            not e.collide
            or e.support
            or e.kind in ("floor", "ceiling", "mat", "window", "nest_tag", "nest_led")
        ):
            continue
        if e.id.startswith("nest_") and not e.id.startswith("nest_housing"):
            continue
        z0, z1 = e.z_range()
        if z1 <= BLOCK_ABOVE_MM or z0 >= DUCK_HEIGHT_MM:
            continue
        outline = _element_shape(e)
        if outline is not None:
            mark(outline)
    for t in layout.things:
        if t.passability != "block" and t.lib:
            continue
        if not t.lib and t.mass_g < 800:
            continue
        for poly, z0, z1 in _thing_shapes(t):
            if z1 > BLOCK_ABOVE_MM and z0 < DUCK_HEIGHT_MM:
                mark(poly)
    for d in layout.drops:
        a, b, c, e_ = d.rect_mm
        mark([(a, b), (c, b), (c, e_), (a, e_)])
    # the outer wall ring is closed by the walls themselves (they are elements); outside the rooms is shut too
    for j in range(ny):
        for i in range(nx):
            cx, cy = x0 + (i + 0.5) * CELL_MM, y0 + (j + 0.5) * CELL_MM
            if layout.room_at(cx, cy) is None:
                blocked[j * nx + i] = 1
    spawn = layout.spawn
    si, sj = int((spawn.x_mm - x0) / CELL_MM), int((spawn.y_mm - y0) / CELL_MM)
    seen = bytearray(nx * ny)
    start = None
    # a spawn that sits in a blocked cell (the dock's housing) starts from the nearest free cell
    for r in range(0, 14):
        for dj in range(-r, r + 1):
            for di in range(-r, r + 1):
                i, j = si + di, sj + dj
                if 0 <= i < nx and 0 <= j < ny and not blocked[j * nx + i] and start is None:
                    start = (i, j)
        if start is not None:
            break
    if start is not None:
        queue: deque[tuple[int, int]] = deque([start])
        seen[start[1] * nx + start[0]] = 1
        while queue:
            i, j = queue.popleft()
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = i + di, j + dj
                if 0 <= a < nx and 0 <= b < ny and not blocked[b * nx + a] and not seen[b * nx + a]:
                    seen[b * nx + a] = 1
                    queue.append((a, b))
    free_total = sum(1 for k in range(nx * ny) if not blocked[k])
    reached_total = sum(1 for k in range(nx * ny) if seen[k])
    rooms: dict[str, dict[str, int]] = {r.id: {"free": 0, "reached": 0} for r in layout.rooms}
    for j in range(ny):
        for i in range(nx):
            k = j * nx + i
            if blocked[k]:
                continue
            rid = layout.room_at(x0 + (i + 0.5) * CELL_MM, y0 + (j + 0.5) * CELL_MM)
            if rid is None:
                continue
            rooms[rid]["free"] += 1
            rooms[rid]["reached"] += 1 if seen[k] else 0
    unreachable = [
        rid
        for rid, c in rooms.items()
        if c["free"] >= ROOM_FLOOR_MIN_CELLS and c["reached"] < UNREACHABLE_BELOW * c["free"]
    ]
    return {
        "cell_mm": CELL_MM,
        "free_cells": free_total,
        "reached_cells": reached_total,
        "reached_pct": round(100.0 * reached_total / free_total, 1) if free_total else 0.0,
        "rooms": {
            rid: round(100.0 * c["reached"] / c["free"], 1) if c["free"] else None
            for rid, c in rooms.items()
        },
        "unreachable_rooms": unreachable,
        "spawn_enclosed": bool(free_total)
        and (start is None or reached_total < ENCLOSED_BELOW * free_total),
    }
