"""Click sets for the go-there benchmark: each click is a start pose and a place on the floor, labelled from the layout.

A click set is a JSONL file (one click a line) with a ``.meta.json`` beside it. The committed sets are in ``data/``:

* ``real_home-v1`` — made once by :func:`generate` (seeded). Starts stand in the open (:data:`START_CLEAR_M` from anything
  solid, not on a sill or a cable); places are 0.8–1.6 m away, have room for the body (:data:`POINT_CLEAR_M`) and can be
  reached by a body-wide route (:mod:`.floor`) no longer than :data:`PATH_MAX_M`. It comes in blocks of
  :data:`BLOCK` clicks (:data:`BLOCK_SAME` in the start's room, the rest in another room), so any whole number of blocks
  is a fair sample and a comparison can stop early (:mod:`.stats`).
* ``legacy-205`` — the 205 clicks of the earlier out-of-repo rig (live clicks replayed), kept for continuity, with the
  same labels and with flags on what the new set excludes (a place on a wall, no body-wide route, a start in clutter).

Labels (truth, for scoring and tables only): ``cls`` same | other (room), ``line`` clear | low | thing | furniture | wall
(the first solid thing on the body-wide straight line, :meth:`.floor.FloorMap.line`), ``reach``, ``path_m``, ``detour``
(route over straight line), ``clear_start_m``, ``clear_point_m``, ``clear_path_m`` (the tightest spot on the route).
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from duckmind.bench import floor as F

SCHEMA = "bench-clicks-v1"
DATA = Path(__file__).resolve().parent / "data"
RANGE_M = (0.8, 1.6)  # the page's fair-scene clicks were this far from the duck
START_CLEAR_M = 0.25  # a start has this much room all round (not in clutter)
POINT_CLEAR_M = 0.12  # a place has room for the body to stand on it
PATH_MAX_M = 4.0  # a route longer than this is not a 40 s errand
ARRIVE_M = 0.18  # the errand's own arrival radius (neuro.errand.ARRIVE_M): a legacy place this near a reachable spot
BLOCK = 100
BLOCK_SAME = 75
SEEDS = 7  # the brain's seed is SEED0 + k % SEEDS, as the earlier rig's
SEED0 = 9201
FLAG_ON_SOLID = "point_on_solid"
FLAG_TIGHT = "point_tight"
FLAG_UNREACHABLE = "unreachable"
FLAG_CLUTTER = "start_in_clutter"
FLAG_FAR = "route_too_long"


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class Labeller:
    """Labels clicks on one home's floor. One body-wide cost field per start (cached for the last start)."""

    def __init__(self, layout: Any) -> None:
        self.fm = F.FloorMap(layout)
        self.passable = self.fm.passable()
        self.cost = self.fm.step_cost()
        self._field_for: tuple[int, int] | None = None
        self._field: Any = None

    def field_from(self, x: float, y: float) -> tuple[Any, tuple[int, int] | None]:
        """The cost field of every place as seen from the start (and the start's own passable cell: the nearest within
        20 cm when the start itself is tight)."""
        cell = self.fm.cell_of(x, y)
        if not self.passable[cell]:
            cell = F.nearest_passable(self.passable, cell, 10)  # type: ignore[assignment]
            if cell is None:
                return None, None
        if self._field_for != cell:
            self._field = F.cost_to_go(
                self.passable, self.cost, cell, self.fm.cell, limit=3.0 * PATH_MAX_M
            )
            self._field_for = cell
        return self._field, cell

    def label(self, pose: Sequence[float], point: Sequence[float]) -> dict[str, Any]:
        fm = self.fm
        x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
        px, py = float(point[0]), float(point[1])
        room, room_to = fm.room_at(x, y), fm.room_at(px, py)
        rng = math.hypot(px - x, py - y)
        line = fm.line((x, y), (px, py))
        out: dict[str, Any] = {
            "room": room,
            "room_to": room_to,
            "cls": "same" if room is not None and room == room_to else "other",
            "range_m": round(rng, 3),
            "bearing": round(_wrap(math.atan2(py - y, px - x) - yaw), 3),
            "line": line["kind"],
            "line_id": line["id"],
            "line_at_m": line["at_m"],
            "clear_start_m": round(fm.clear_at(x, y), 2),
            "clear_point_m": round(fm.clear_at(px, py), 2),
        }
        flags: list[str] = []
        if fm.kind_at(px, py) >= F.THING:
            flags.append(FLAG_ON_SOLID)
        elif out["clear_point_m"] < F.BODY_RADIUS_M:
            flags.append(FLAG_TIGHT)
        if out["clear_start_m"] < START_CLEAR_M:
            flags.append(FLAG_CLUTTER)
        field, _ = self.field_from(x, y)
        goal = fm.cell_of(px, py)
        reach = field is not None and math.isfinite(float(field[goal]))
        near = goal if reach else None
        if not reach and field is not None:
            near = F.nearest_passable(
                self.passable, goal, int(ARRIVE_M / fm.cell) - 2, weight=field
            )
        path = F.descend(field, near) if near is not None else []
        out["reach"] = bool(reach)
        out["reach_near"] = near is not None
        if path:
            ln = F.path_length(path, fm.cell)
            out["path_m"] = round(ln, 2)
            out["detour"] = round(ln / max(rng, 1e-6), 2)
            out["clear_path_m"] = round(min(float(fm.clear[c]) for c in path), 2)
            if ln > PATH_MAX_M:
                flags.append(FLAG_FAR)
        else:
            out["path_m"] = out["detour"] = out["clear_path_m"] = None
            flags.append(FLAG_UNREACHABLE)
        out["flags"] = flags
        return out


def _layout(home: str) -> Any:
    from duckmind.world import homelayout

    return homelayout.load(home)


def generate(
    home: str = "real_home", blocks: int = 8, seed: int = 20261110, name: str | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The click set of ``home``: ``blocks`` × :data:`BLOCK` clicks and its meta (how many candidates were dropped and why).
    Deterministic in (layout, ``seed``); a shorter set is a prefix of a longer one."""
    layout = _layout(home)
    lab = Labeller(layout)
    fm = lab.fm
    name = name or f"{home}-v1"
    ok_start = (fm.kind == F.FREE) & (fm.clear >= START_CLEAR_M) & (fm.room >= 0)
    cells = [(int(j), int(i)) for j, i in zip(*ok_start.nonzero(), strict=True)]
    clicks: list[dict[str, Any]] = []
    dropped = {
        "start_no_place": 0,
        "place_outside": 0,
        "place_wrong_room": 0,
        "place_no_room": 0,
        "place_unreachable": 0,
        "place_route_too_long": 0,
    }
    for block in range(int(blocks)):
        rng = random.Random(f"{name}/{seed}/{block}")
        order = ["same"] * BLOCK_SAME + ["other"] * (BLOCK - BLOCK_SAME)
        rng.shuffle(order)
        for want in order:
            while True:
                j, i = cells[rng.randrange(len(cells))]
                cx, cy = fm.xy_of(j, i)
                x = round(cx + rng.uniform(-0.5, 0.5) * fm.cell, 3)
                y = round(cy + rng.uniform(-0.5, 0.5) * fm.cell, 3)
                yaw = round(rng.uniform(-math.pi, math.pi), 3)
                found = None
                for _ in range(30):
                    b = rng.uniform(-math.pi, math.pi)
                    r = rng.uniform(*RANGE_M)
                    px, py = (
                        round(x + r * math.cos(yaw + b), 2),
                        round(y + r * math.sin(yaw + b), 2),
                    )
                    room_to = fm.room_at(px, py)
                    if room_to is None:
                        dropped["place_outside"] += 1
                        continue
                    if (room_to == fm.room_at(x, y)) != (want == "same"):
                        dropped["place_wrong_room"] += 1
                        continue
                    if fm.kind_at(px, py) != F.FREE or fm.clear_at(px, py) < POINT_CLEAR_M:
                        dropped["place_no_room"] += 1
                        continue
                    info = lab.label((x, y, yaw), (px, py))
                    if not info["reach"]:
                        dropped["place_unreachable"] += 1
                        continue
                    if FLAG_FAR in info["flags"]:
                        dropped["place_route_too_long"] += 1
                        continue
                    found = (px, py, info)
                    break
                if found is not None:
                    break
                dropped["start_no_place"] += 1
            px, py, info = found
            k = len(clicks)
            clicks.append(
                {
                    "k": k,
                    "set": name,
                    "block": block,
                    "pose": [x, y, yaw],
                    "point": [px, py],
                    "seed": SEED0 + k % SEEDS,
                    **info,
                }
            )
    meta = {
        "schema": SCHEMA,
        "name": name,
        "home": home,
        "layout": str(layout.fingerprint),
        "seed": seed,
        "blocks": int(blocks),
        "block": BLOCK,
        "block_same": BLOCK_SAME,
        "clicks": len(clicks),
        "rules": {
            "range_m": list(RANGE_M),
            "start_clear_m": START_CLEAR_M,
            "point_clear_m": POINT_CLEAR_M,
            "path_max_m": PATH_MAX_M,
            "body_radius_m": F.BODY_RADIUS_M,
            "cell_m": F.CELL_M,
        },
        "dropped": dropped,
    }
    return clicks, meta


def relabel(
    home: str, rows: Iterable[dict[str, Any]], name: str, source: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """An existing list of clicks (``pose``, ``point``, in order) as a labelled set: nothing is dropped, what the generated
    set would have excluded is flagged."""
    layout = _layout(home)
    lab = Labeller(layout)
    clicks = []
    for k, r in enumerate(rows):
        pose = [round(float(v), 3) for v in r["pose"]]
        point = [round(float(v), 3) for v in r["point"]]
        clicks.append(
            {
                "k": k,
                "set": name,
                "block": 0,
                "pose": pose,
                "point": point,
                "seed": SEED0 + k % SEEDS,
                **lab.label(pose, point),
            }
        )
    flags: dict[str, int] = {}
    for c in clicks:
        for f in c["flags"]:
            flags[f] = flags.get(f, 0) + 1
    meta = {
        "schema": SCHEMA,
        "name": name,
        "home": home,
        "layout": str(layout.fingerprint),
        "clicks": len(clicks),
        "source": source,
        "flags": flags,
    }
    return clicks, meta


def path_of(name: str) -> Path:
    """A set's file: a path as given, else the committed set of that name."""
    p = Path(name)
    return p if p.suffix == ".jsonl" else DATA / f"{name}.jsonl"


def load(name: str) -> list[dict[str, Any]]:
    with path_of(name).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def meta_of(name: str) -> dict[str, Any]:
    p = path_of(name)
    meta = p.parent / (p.stem + ".meta.json")
    return json.loads(meta.read_text(encoding="utf-8")) if meta.is_file() else {}


def save(
    clicks: Sequence[dict[str, Any]], meta: dict[str, Any], folder: Path | None = None
) -> Path:
    folder = Path(folder) if folder is not None else DATA
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"{meta['name']}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for c in clicks:
            f.write(json.dumps(c, ensure_ascii=False, sort_keys=True) + "\n")
    (folder / f"{meta['name']}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out


def select(
    clicks: Sequence[dict[str, Any]],
    *,
    blocks: Sequence[int] | None = None,
    limit: int = 0,
    keys: Sequence[int] | None = None,
    clean: bool = False,
) -> list[dict[str, Any]]:
    """Part of a set: whole blocks, the first ``limit``, named clicks; ``clean``: without the flagged ones."""
    out = list(clicks)
    if blocks is not None:
        out = [c for c in out if int(c.get("block", 0)) in set(blocks)]
    if keys is not None:
        out = [c for c in out if int(c["k"]) in set(keys)]
    if clean:
        out = [c for c in out if not c.get("flags")]
    return out[:limit] if limit else out


__all__ = [
    "DATA",
    "SCHEMA",
    "Labeller",
    "generate",
    "load",
    "meta_of",
    "path_of",
    "relabel",
    "save",
    "select",
]
