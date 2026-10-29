"""The object library (``world/objects/library.json``) and the placements that use it.

A home file may carry ``"placements"``: a list of ``{"id", "lib", "x_mm", "y_mm", "yaw_mrad"}`` (and,
optionally, ``"length_mm"`` for a resizable strip or cable, ``"colour"``, ``"zh"``). Each one is an
item of the library put at a place on the floor. :func:`expand` turns them into the ``elements`` (fixed
pieces: furniture, thresholds, rugs, cables) and ``things`` (free bodies the duck can push: slippers,
books, bowls) that :mod:`duckmind.world.homelayout` already understands, so the physics, the page, the
sensors and the checks all see ordinary layout pieces and need nothing new.

An item (see the file's ``doc``): ``id``, ``zh`` (the 繁體中文 name; the other five languages are the
page catalogs ``page.lib.item.<id>``), ``category``, ``kind`` (``element`` or ``thing``), ``passability``
(one of :data:`CLASSES`), ``parts`` and, for a thing, mass, friction and verbs. Its frame: x along the
long side, front = -y, the origin on the floor under the footprint centre. Heights give the classes
the duck meets (:func:`height_class`): under-foot up to 5 mm, step 5-20, hop 20-40, block above 40;
``soft`` things give under weight and ``movable`` ones are light enough to push.

Nothing here imports MuJoCo. The module is plain Python so the page, the editor and the tests use it.
"""

from __future__ import annotations

import copy
import functools
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LIBRARY_PATH = Path(__file__).with_name("objects") / "library.json"
SCHEMA = "object-library-v1"
CLASSES = ("under_foot", "step", "hop", "block", "soft", "movable")
UNDER_FOOT_MAX_MM = 5
STEP_MAX_MM = 20
HOP_MAX_MM = 40
HEIGHT_CLASSES = ("under_foot", "step", "hop", "block")
# the shapes an element part may have, and the shapes a thing part may have
ELEMENT_SHAPES = ("box", "slab", "cylinder", "frustum", "sphere", "capsule")
THING_SHAPES = ("box", "cylinder", "sphere")
CATEGORIES = ("doorway", "floor", "cable", "furniture", "clutter", "toy")


class LibraryError(ValueError):
    """A library or a placement that cannot be used, said in plain words."""


def height_class(height_mm: float) -> str:
    """The class a rigid fixed thing of this height falls in (the duck's step, hop or block)."""
    if height_mm <= UNDER_FOOT_MAX_MM:
        return "under_foot"
    if height_mm <= STEP_MAX_MM:
        return "step"
    if height_mm <= HOP_MAX_MM:
        return "hop"
    return "block"


def _rot(x: float, y: float, mrad: float) -> tuple[float, float]:
    angle = mrad / 1000.0
    c, s = math.cos(angle), math.sin(angle)
    return c * x - s * y, s * x + c * y


@dataclass(frozen=True)
class Item:
    """One library item (the file's JSON, with the numbers worked out)."""

    id: str
    zh: str
    category: str
    kind: str  # element | thing
    passability: str
    parts: tuple[Mapping[str, Any], ...]
    raw: Mapping[str, Any]

    @property
    def mass_g(self) -> int:
        return sum(int(p.get("mass_g", 0)) for p in self.parts)

    @property
    def height_mm(self) -> int:
        """The highest point of the item above the floor (parts that are not felt do not count)."""
        top = 0
        for part in self.parts:
            if part.get("collide", True) is False:
                continue
            top = max(top, int(_offset(part)[2] + _part_extent(part)[2]))
        return top

    @property
    def height_class(self) -> str:
        return height_class(self.height_mm)

    @property
    def resize_x(self) -> tuple[int, int] | None:
        value = self.raw.get("resize_x")
        return (int(value[0]), int(value[1])) if value else None

    def footprint_mm(self) -> tuple[int, int]:
        """The item's unrotated bounding box on the floor (x, y), mm."""
        x0, y0, x1, y1 = _bounds(self.parts, collide_only=True)
        return int(round(x1 - x0)), int(round(y1 - y0))

    def page_json(self) -> dict[str, Any]:
        """What the layout editor needs: the item, its parts as flat drawings, its class."""
        w, d = self.footprint_mm()
        return {
            "id": self.id,
            "zh": self.zh,
            "category": self.category,
            "kind": self.kind,
            "passability": self.passability,
            "height_mm": self.height_mm,
            "height_class": self.height_class,
            "mass_g": self.mass_g if self.kind == "thing" else 0,
            "footprint_mm": [w, d],
            "resize_x": list(self.resize_x) if self.resize_x else None,
            "contact": str(self.raw.get("contact", "rigid")),
            "friction": _item_friction(self),
            "note": str(self.raw.get("note", "")),
            "colour": _item_colour(self),
            "draw": [_draw(p) for p in self.parts if p.get("collide", True) is not False],
        }


def _offset(part: Mapping[str, Any]) -> tuple[float, float, float]:
    v = part.get("offset_mm", (0, 0, 0))
    return float(v[0]), float(v[1]), float(v[2])


def _part_extent(part: Mapping[str, Any]) -> tuple[float, float, float]:
    """The part's own size (x, y, z), mm, before it is offset."""
    shape = part["shape"]
    if shape in ("box", "slab"):
        s = part["size_mm"]
        return float(s[0]), float(s[1]), float(s[2])
    if shape in ("cylinder", "frustum"):
        r = float(max(part.get("radius_mm", 0), part.get("radius_bottom_mm", 0)))
        return 2 * r, 2 * r, float(part["height_mm"])
    if shape == "sphere":
        r = float(part["radius_mm"])
        return 2 * r, 2 * r, 2 * r
    if shape == "capsule":
        a = part["fromto_mm"]
        r = float(part["radius_mm"])
        return (
            abs(a[3] - a[0]) + 2 * r,
            abs(a[4] - a[1]) + 2 * r,
            max(a[2], a[5]) + r,
        )
    raise LibraryError(f"形狀 {shape!r} 不認得")


def _bounds(
    parts: tuple[Mapping[str, Any], ...], *, collide_only: bool = False
) -> tuple[float, float, float, float]:
    x0 = y0 = math.inf
    x1 = y1 = -math.inf
    for part in parts:
        if collide_only and part.get("collide", True) is False:
            continue
        ox, oy, _ = _offset(part)
        if part["shape"] == "capsule":
            a = part["fromto_mm"]
            r = float(part["radius_mm"])
            x0 = min(x0, ox + min(a[0], a[3]) - r)
            x1 = max(x1, ox + max(a[0], a[3]) + r)
            y0 = min(y0, oy + min(a[1], a[4]) - r)
            y1 = max(y1, oy + max(a[1], a[4]) + r)
            continue
        w, d, _ = _part_extent(part)
        x0, x1 = min(x0, ox - w / 2), max(x1, ox + w / 2)
        y0, y1 = min(y0, oy - d / 2), max(y1, oy + d / 2)
    if x0 == math.inf:
        return 0.0, 0.0, 0.0, 0.0
    return x0, y0, x1, y1


def _draw(part: Mapping[str, Any]) -> dict[str, Any]:
    """A part as the editor draws it from above."""
    ox, oy, _ = _offset(part)
    out: dict[str, Any] = {
        "shape": part["shape"],
        "at": [ox, oy],
        "z": [_offset(part)[2], _offset(part)[2] + _part_extent(part)[2]],
    }
    if part["shape"] in ("box", "slab"):
        out["size"] = [part["size_mm"][0], part["size_mm"][1]]
    elif part["shape"] in ("cylinder", "frustum", "sphere"):
        out["r"] = max(part.get("radius_mm", 0), part.get("radius_bottom_mm", 0))
    else:
        a = part["fromto_mm"]
        out["from"] = [ox + a[0], oy + a[1]]
        out["to"] = [ox + a[3], oy + a[4]]
        out["r"] = part["radius_mm"]
    if part.get("colour"):
        out["colour"] = part["colour"]
    return out


def _item_colour(item: Item) -> str:
    for part in item.parts:
        if part.get("colour"):
            return str(part["colour"])
    return str(item.raw.get("colour", "#bbbbbb"))


def _item_friction(item: Item) -> float:
    if item.kind == "thing":
        return float(item.raw.get("friction", 0.8))
    for part in item.parts:
        if part.get("friction"):
            return float(part["friction"][0])
    return 1.0


@dataclass(frozen=True)
class Library:
    items: tuple[Item, ...]
    doc: str

    def item(self, item_id: str) -> Item:
        for item in self.items:
            if item.id == item_id:
                return item
        raise LibraryError(f"物件庫裡沒有「{item_id}」")

    def __contains__(self, item_id: object) -> bool:
        return any(item.id == item_id for item in self.items)

    def page_json(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "classes": list(CLASSES),
            "categories": list(CATEGORIES),
            "items": [item.page_json() for item in self.items],
        }


def load_library(path: Path | None = None) -> Library:
    data = json.loads((path or LIBRARY_PATH).read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise LibraryError(f"物件庫的格式應該是 {SCHEMA}")
    items: list[Item] = []
    seen: set[str] = set()
    for raw in data["items"]:
        item_id = str(raw["id"])
        if item_id in seen:
            raise LibraryError(f"物件庫裡「{item_id}」出現了兩次")
        seen.add(item_id)
        items.append(
            Item(
                id=item_id,
                zh=str(raw["zh"]),
                category=str(raw["category"]),
                kind=str(raw["kind"]),
                passability=str(raw["passability"]),
                parts=tuple(raw["parts"]),
                raw=raw,
            )
        )
    return Library(tuple(items), str(data.get("doc", "")))


@functools.lru_cache(maxsize=1)
def library() -> Library:
    return load_library()


# ── placements → elements and things ───────────────────────────────────────────────────────────


def _int(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value):
        raise LibraryError(f"{what} 應該是整數（mm、mrad），這裡是 {value!r}")
    return int(value)


def placement_problems(placements: list[Any]) -> list[str]:
    """What is wrong with a list of placements, in plain words (empty: nothing)."""
    out: list[str] = []
    lib = library()
    seen: set[str] = set()
    for index, p in enumerate(placements):
        if not isinstance(p, Mapping):
            out.append(f"第 {index + 1} 件擺設不是一個物件")
            continue
        pid = str(p.get("id", ""))
        if not pid:
            out.append(f"第 {index + 1} 件擺設沒有 id")
        elif pid in seen:
            out.append(f"擺設的 id「{pid}」用了不只一次")
        seen.add(pid)
        if p.get("lib") not in lib:
            out.append(f"擺設「{pid}」用的物件「{p.get('lib')}」不在物件庫裡")
            continue
        for key in ("x_mm", "y_mm"):
            try:
                _int(p.get(key), f"擺設「{pid}」的 {key}")
            except LibraryError as error:
                out.append(str(error))
        item = lib.item(str(p["lib"]))
        if "length_mm" in p:
            if item.resize_x is None:
                out.append(f"「{item.id}」不能改長度")
            else:
                lo, hi = item.resize_x
                try:
                    length = _int(p["length_mm"], f"擺設「{pid}」的 length_mm")
                    if not lo <= length <= hi:
                        out.append(f"擺設「{pid}」的長度要在 {lo} 到 {hi} mm 之間")
                except LibraryError as error:
                    out.append(str(error))
    return out


def expand(raw: dict[str, Any]) -> dict[str, Any]:
    """Put every placement of ``raw`` into its ``elements`` and ``things`` (in place; also returned).
    Idempotent: a layout that was expanded says so (``placements_expanded``) and is left as it is."""
    placements = raw.get("placements")
    if not placements or raw.get("placements_expanded"):
        return raw
    problems = placement_problems(list(placements))
    if problems:
        raise LibraryError("；".join(problems))
    lib = library()
    elements = list(raw.get("elements", []))
    things = list(raw.get("things", []))
    for placement in placements:
        item = lib.item(str(placement["lib"]))
        if item.kind == "thing":
            things.append(thing_of(item, placement))
        else:
            elements += elements_of(item, placement)
    raw["elements"] = elements
    raw["things"] = things
    raw["placements_expanded"] = True
    return raw


def unexpand(raw: Mapping[str, Any]) -> dict[str, Any]:
    """The inverse of :func:`expand`: a layout as it is written in a file (the placements, not the pieces
    they made). The elements and things that came from a placement carry ``lib``; they are dropped and the
    ``placements_expanded`` mark with them, so loading the result expands the placements again."""
    out = copy.deepcopy(dict(raw))
    if not out.get("placements_expanded"):
        return out
    out["elements"] = [e for e in out.get("elements", []) if not e.get("place")]
    out["things"] = [t for t in out.get("things", []) if not t.get("lib")]
    out.pop("placements_expanded", None)
    return out


def _resized(part: Mapping[str, Any], item: Item, placement: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(dict(part))
    length = placement.get("length_mm")
    if length is None or item.resize_x is None:
        return out
    length = int(length)
    if out["shape"] in ("box", "slab"):
        out["size_mm"][0] = length
    elif out["shape"] == "capsule":
        out["fromto_mm"][0], out["fromto_mm"][3] = -length // 2, length // 2
    return out


def elements_of(item: Item, placement: Mapping[str, Any]) -> list[dict[str, Any]]:
    pid = str(placement["id"])
    x, y = int(placement["x_mm"]), int(placement["y_mm"])
    yaw = int(placement.get("yaw_mrad", 0))
    out: list[dict[str, Any]] = []
    many = len(item.parts) > 1
    for k, source in enumerate(item.parts):
        part = _resized(source, item, placement)
        ox, oy, oz = _offset(part)
        rx, ry = _rot(ox, oy, yaw)
        eid = f"{pid}_{part.get('id', k + 1)}" if many else pid
        shape = part["shape"]
        element: dict[str, Any] = {
            "id": eid,
            "kind": part.get("ekind") or item.raw.get("ekind") or "furniture",
            "cls": part.get("cls") or item.raw.get("cls", ""),
            "zh": placement.get("zh") or item.zh,
            "lib": item.id,
            "place": pid,
            "shape": shape,
            "colour": placement.get("colour") or part.get("colour") or "#cccccc",
            "yaw_mrad": yaw + int(part.get("yaw_mrad", 0)),
            "passability": item.passability,
        }
        if part.get("collide", True) is False:
            element["collide"] = False
        if part.get("tof") is not None:
            element["tof"] = bool(part["tof"])
        if part.get("friction"):
            element["friction"] = list(part["friction"])
        if part.get("under_mm"):
            element["under_mm"] = int(part["under_mm"])
        solref = part.get("solref") or item.raw.get("solref")
        if solref:
            element["solref"] = list(solref)
            solmix = part.get("solmix") or item.raw.get("solmix")
            if solmix:
                element["solmix"] = float(solmix)
        if shape == "capsule":
            a = part["fromto_mm"]
            r = int(part["radius_mm"])
            ax, ay = _rot(a[0] + ox, a[1] + oy, yaw)
            bx, by = _rot(a[3] + ox, a[4] + oy, yaw)
            element["radius_mm"] = r
            element["fromto_mm"] = [
                int(round(x + ax)),
                int(round(y + ay)),
                int(a[2] + oz),
                int(round(x + bx)),
                int(round(y + by)),
                int(a[5] + oz),
            ]
            element.pop("yaw_mrad", None)
        else:
            element["pos_mm"] = [int(round(x + rx)), int(round(y + ry)), int(round(oz))]
            if shape in ("box", "slab"):
                element["size_mm"] = [int(v) for v in part["size_mm"]]
                if shape == "slab":
                    element["ramp_mm"] = int(part["ramp_mm"])
            elif shape in ("cylinder", "frustum"):
                element["radius_mm"] = int(part["radius_mm"])
                element["height_mm"] = int(part["height_mm"])
                if shape == "frustum":
                    element["radius_bottom_mm"] = int(part["radius_bottom_mm"])
            elif shape == "sphere":
                element["radius_mm"] = int(part["radius_mm"])
        out.append(element)
    return out


def thing_of(item: Item, placement: Mapping[str, Any]) -> dict[str, Any]:
    raw = item.raw
    thing: dict[str, Any] = {
        "id": str(placement["id"]),
        "zh": placement.get("zh") or item.zh,
        "lib": item.id,
        "passability": item.passability,
        "cls": raw.get("cls", "unknown"),
        "pos_mm": [int(placement["x_mm"]), int(placement["y_mm"])],
        "yaw_mrad": int(placement.get("yaw_mrad", 0)),
        "mass_g": item.mass_g,
        "colour": placement.get("colour") or raw.get("colour", "#cccccc"),
        "friction": float(raw.get("friction", 0.8)),
        "roll": float(raw.get("roll", 0.0002)),
        "verbs": list(raw.get("verbs", [])),
        "parts": [
            {
                k: v
                for k, v in part.items()
                if k in ("shape", "size_mm", "radius_mm", "height_mm", "offset_mm", "mass_g")
            }
            for part in item.parts
        ],
    }
    if raw.get("opacity") is not None:
        thing["opacity"] = float(raw["opacity"])
    if raw.get("hollow"):
        thing["hollow"] = True
    if raw.get("solref"):
        thing["solref"] = list(raw["solref"])
        if raw.get("solmix"):
            thing["solmix"] = float(raw["solmix"])
    return thing


# ── shapes on the floor, for the placement checks and the editor ───────────────────────────────


def corners(item: Item, placement: Mapping[str, Any]) -> list[tuple[float, float]]:
    """The corners of the item's rotated bounding rectangle on the floor, mm (felt parts only)."""
    parts = tuple(_resized(p, item, placement) for p in item.parts)
    x0, y0, x1, y1 = _bounds(parts, collide_only=True)
    yaw = int(placement.get("yaw_mrad", 0))
    x, y = float(placement["x_mm"]), float(placement["y_mm"])
    out = []
    for a, b in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        rx, ry = _rot(a, b, yaw)
        out.append((x + rx, y + ry))
    return out


def part_shapes(item: Item, placement: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every felt part of a placed item as a floor shape: a rotated rectangle (``poly``) or a
    circle (``circle``) with the part's height range ``z`` and the item's id."""
    out: list[dict[str, Any]] = []
    yaw = int(placement.get("yaw_mrad", 0))
    x, y = float(placement["x_mm"]), float(placement["y_mm"])
    for source in item.parts:
        if source.get("collide", True) is False:
            continue
        part = _resized(source, item, placement)
        ox, oy, oz = _offset(part)
        cx, cy = _rot(ox, oy, yaw)
        z = [oz, oz + _part_extent(part)[2]]
        if part["shape"] in ("cylinder", "frustum", "sphere"):
            r = max(part.get("radius_mm", 0), part.get("radius_bottom_mm", 0))
            out.append({"circle": (x + cx, y + cy, float(r)), "z": z})
        elif part["shape"] == "capsule":
            a = part["fromto_mm"]
            r = float(part["radius_mm"])
            p0 = _rot(a[0] + ox, a[1] + oy, yaw)
            p1 = _rot(a[3] + ox, a[4] + oy, yaw)
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            n = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / n * r, dx / n * r
            poly = [
                (x + p0[0] + nx, y + p0[1] + ny),
                (x + p1[0] + nx, y + p1[1] + ny),
                (x + p1[0] - nx, y + p1[1] - ny),
                (x + p0[0] - nx, y + p0[1] - ny),
            ]
            out.append({"poly": poly, "z": z})
        else:
            w, d, _ = _part_extent(part)
            poly = []
            for a, b in ((-w / 2, -d / 2), (w / 2, -d / 2), (w / 2, d / 2), (-w / 2, d / 2)):
                rx, ry = _rot(a, b, yaw)
                poly.append((x + cx + rx, y + cy + ry))
            out.append({"poly": poly, "z": z})
    return out
