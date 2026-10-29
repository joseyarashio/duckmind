"""Phase 1's obstacle classes, put into a real home for the phase-3 gates (spec §17 期3).

Phase 1 measured six obstacles on the official flat floor (``obstacles.md`` in the measurement records
§1): a wall (3 m × 80 cm), the same wall met at ~30°, a 3 cm pole (70 cm), a 4–8 cm box
(20 × 40 cm), a 1–3 cm threshold (4 cm × 2 m) and a 5–15 cm step down. Phase 3's gates need the
same obstacles where the camera sees what it will see at home — the wood floor, the walls, the
lights — so each trial here is a **layout dict**: ``default_home`` (by default) with its
furniture and things taken out and one obstacle added, which :func:`duckmind.world.rig.Rig`
composes like any home. The stair class uses ``apartment_ungated_test`` as it is.

Geometry of a trial: the obstacle's near face (or the drop's edge) lies across the +x axis at
``FACE_X_MM``; the duck starts ``distance`` behind the point it will meet (``hit`` = the face at
lateral ``offset``), heading ``angle`` off the face's normal — so the straight walk meets the
obstacle at that incidence and offset whatever the class (a drop is axis-aligned, so the duck
turns instead of the obstacle). Every number of a trial comes from its class and index only
(:func:`trial`), seeded, so a bright and a dark run, or a ToF and a camera-only run, of the
same index are the same scene.

Evaluator helpers (truth only): :func:`clearance_mm` (a point to the obstacle, 0 inside) and
:func:`inside_drop`.
"""

from __future__ import annotations

import copy
import hashlib
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

CLASSES = ("wall", "wall30", "pole", "box", "threshold", "step", "stairs")
CLASS_ZH = {
    "wall": "正面的牆",
    "wall30": "斜 30° 的牆",
    "pole": "3 cm 細桿",
    "box": "4–8 cm 矮盒",
    "threshold": "1–3 cm 門檻",
    "step": "5–15 cm 往下的台階",
    "stairs": "公寓沒有欄杆的樓梯口",
}
FACE_X_MM = 800  # the obstacle's near face (default_home: the room spans ±3000 mm)
# How far the beak starts from the point it will meet. velstand at vx 0.40 really advances
# ~0.157 m/s and turns right ~10 °/s with no steering (phase 1 gait.md §5, HARNESS.md), so a
# trial starts close enough to meet the obstacle within a few seconds; :data:`DRIFT_TRIM_MRAD_S`
# is the yaw-rate trim that keeps a scripted walk straight (the harness's, not the duck's).
DISTANCE_M = (0.6, 1.2)
DRIFT_TRIM_MRAD_S = 180
BEAK_AHEAD_MM = 97  # the beak is ~97 mm ahead of the trunk origin (duckbody.depth docstring)
OBSTACLE_ID = "p3_obstacle"
DROP_ID = "p3_step"
# The box gate is reported per colour (owner D2): the palette the trials cycle through, the
# cardboard tone first (the phase-3 survey's hardest: 14/36 on wood).
BOX_COLOURS = {
    "cardboard": "#b08a5a",
    "white": "#eeeeea",
    "red": "#c0392b",
    "blue": "#2e6fd8",
    "green": "#3c9a4b",
    "black": "#26262a",
}
FURNITURE_KEEP = frozenset({"floor", "wall", "skirting"})  # what stays of the base home


@dataclass(frozen=True)
class Trial:
    """One scene of one class: the layout (a dict, or a home name), where the duck starts
    (world metres, radians), what it walks at, and the evaluator's view of the obstacle."""

    cls: str
    index: int
    home: str | dict[str, Any]
    start: tuple[float, float, float]
    target: dict[str, Any]
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.cls}:{self.index:03d}"


def _rng(cls: str, index: int, seed: int) -> random.Random:
    digest = hashlib.sha256(f"phase3-rig:{cls}:{index}:{seed}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def base_room(
    home: str = "default_home", *, nest: bool = True, plain_floor: bool = False
) -> dict[str, Any]:
    """The home's raw layout with its furniture and things taken out (walls, skirting and the
    floor stay; the lights stay); ``nest`` False drops the charging place too; ``plain_floor``
    paints the floor one colour (the D1 comparison)."""
    from duckmind.world import homelayout

    layout = homelayout.load(home)
    raw = copy.deepcopy(layout.to_json())
    raw.pop("variants", None)
    raw.pop("variant_applied", None)
    raw["id"] = f"{raw.get('id', home)}_p3"
    raw["elements"] = [
        e
        for e in raw.get("elements", [])
        if e.get("kind") in FURNITURE_KEEP or str(e.get("id", "")).startswith("nest_")
    ]
    raw["things"] = []
    raw["drops"] = []
    raw["zones"] = [z for z in raw.get("zones", []) if z.get("kind") == "nest"]
    raw["barriers"] = []
    if not nest:
        raw["nest"] = None
        raw["elements"] = [
            e for e in raw["elements"] if not str(e.get("id", "")).startswith("nest_")
        ]
        raw["zones"] = []
        if raw.get("spawn") == "nest":
            raw["spawn"] = "mat"
    if plain_floor:
        for e in raw["elements"]:
            if e.get("kind") == "floor":
                e.pop("texture", None)
        if isinstance(raw.get("floor"), dict):
            raw["floor"].pop("texture", None)
    raw.pop("agents", None)
    return raw


def open_floor(home: str = "default_home", half_mm: int = 20000) -> dict[str, Any]:
    """The home's floor (its look, its light) with no walls, ``2 × half_mm`` across: long
    headless walks (the pickup false-alarm hours) that never meet a wall."""
    raw = base_room(home, nest=False)
    raw["id"] = "open_floor"
    raw["elements"] = [e for e in raw["elements"] if e.get("kind") == "floor"]
    for e in raw["elements"]:
        e["size_mm"] = [2 * half_mm, 2 * half_mm, 0]
    raw["bounds_mm"] = [-half_mm, -half_mm, half_mm, half_mm]
    for room in raw.get("rooms", []):
        room["rect_mm"] = [-half_mm + 100, -half_mm + 100, half_mm - 100, half_mm - 100]
    raw.setdefault("lights", {})["windows"] = []
    raw["doorways"] = []
    return raw


def _start(hit_x: float, hit_y: float, angle: float, distance: float) -> tuple[float, float, float]:
    """The trunk's start so that walking straight at ``angle`` brings the beak to the hit point
    after ``distance`` metres."""
    back = distance + BEAK_AHEAD_MM / 1000.0
    return (hit_x - back * math.cos(angle), hit_y - back * math.sin(angle), angle)


def trial(
    cls: str,
    index: int,
    *,
    home: str = "default_home",
    seed: int = 0,
    nest: bool = True,
    plain_floor: bool = False,
) -> Trial:
    """Trial ``index`` of obstacle class ``cls`` (see :data:`CLASSES`), deterministic."""
    if cls not in CLASSES:
        raise ValueError(f"unknown obstacle class {cls!r} (known: {', '.join(CLASSES)})")
    rng = _rng(cls, index, seed)
    distance = rng.uniform(*DISTANCE_M)
    if cls == "stairs":
        return _stairs_trial(index, rng, distance)
    # a drop must keep 600 mm from the charging place (homelayout's rule): no dock in step scenes
    raw = base_room(home, nest=nest and cls != "step", plain_floor=plain_floor)
    face = FACE_X_MM / 1000.0
    params: dict[str, Any] = {"distance_m": round(distance, 3)}
    target: dict[str, Any]
    if cls in ("wall", "wall30"):
        if cls == "wall":
            angle = math.radians(rng.uniform(-15.0, 15.0))
        else:
            side = 1.0 if index % 2 == 0 else -1.0
            angle = side * math.radians(rng.uniform(22.5, 37.5))
        offset = rng.uniform(-0.3, 0.3)
        raw["elements"].append(
            {
                "id": OBSTACLE_ID,
                "kind": "wall",
                "cls": "furniture",
                "zh": CLASS_ZH[cls],
                "shape": "box",
                "pos_mm": [FACE_X_MM + 20, 0, 0],
                "size_mm": [40, 3000, 800],
                "colour": "#e6e0d4",
            }
        )
        target = {"kind": "element", "id": OBSTACLE_ID, "shape": "box"}
        params.update(angle_deg=round(math.degrees(angle), 2), offset_m=round(offset, 3))
    elif cls == "pole":
        offset = rng.uniform(-0.12, 0.12)
        angle = math.radians(rng.uniform(-10.0, 10.0))
        raw["elements"].append(
            {
                "id": OBSTACLE_ID,
                "kind": "furniture",
                "cls": "furniture",
                "zh": CLASS_ZH[cls],
                "shape": "cylinder",
                "pos_mm": [FACE_X_MM + 15, 0, 0],
                "radius_mm": 15,
                "height_mm": 700,
                "colour": "#8a6e4b",
            }
        )
        # the pole is met by the body's centre line ± offset: the hit point is the pole itself
        target = {"kind": "element", "id": OBSTACLE_ID, "shape": "cylinder"}
        params.update(angle_deg=round(math.degrees(angle), 2), offset_m=round(offset, 3))
        start = _start(face - offset * math.sin(angle), -offset * math.cos(angle), angle, distance)
        return Trial(cls, index, raw, start, target, params)
    elif cls == "box":
        heights = (40, 50, 60, 70, 80)
        height = heights[index % len(heights)]
        names = list(BOX_COLOURS)
        colour = names[(index // len(heights) + index) % len(names)]
        angle = math.radians(rng.uniform(-30.0, 30.0))
        offset = rng.uniform(-0.18, 0.18)
        raw["elements"].append(
            {
                "id": OBSTACLE_ID,
                "kind": "furniture",
                "cls": "heavy",
                "zh": f"{CLASS_ZH[cls]} {height} mm",
                "shape": "box",
                "pos_mm": [FACE_X_MM + 100, 0, 0],
                "size_mm": [200, 400, height],
                "colour": BOX_COLOURS[colour],
            }
        )
        target = {"kind": "element", "id": OBSTACLE_ID, "shape": "box", "colour": colour}
        params.update(
            angle_deg=round(math.degrees(angle), 2),
            offset_m=round(offset, 3),
            height_mm=height,
            colour=colour,
        )
    elif cls == "threshold":
        heights = (10, 15, 20, 25, 30)
        height = heights[index % len(heights)]
        angle = math.radians(rng.uniform(-45.0, 45.0))
        offset = rng.uniform(-0.3, 0.3)
        raw["elements"].append(
            {
                "id": OBSTACLE_ID,
                "kind": "threshold",
                "cls": "furniture",
                "zh": f"{CLASS_ZH[cls]} {height} mm",
                "shape": "box",
                "pos_mm": [FACE_X_MM + 20, 0, 0],
                "size_mm": [40, 2000, height],
                "colour": "#a88a66",
            }
        )
        target = {"kind": "element", "id": OBSTACLE_ID, "shape": "box"}
        params.update(
            angle_deg=round(math.degrees(angle), 2), offset_m=round(offset, 3), height_mm=height
        )
    else:  # step: a drop across the path
        depths = (50, 75, 100, 125, 150)
        depth = depths[index % len(depths)]
        angle = math.radians(rng.uniform(-50.0, 50.0))
        offset = rng.uniform(-0.3, 0.3)
        raw["drops"] = [
            {
                "id": DROP_ID,
                "zh": f"{CLASS_ZH[cls]} {depth} mm",
                "rect_mm": [FACE_X_MM, -1500, FACE_X_MM + 1500, 1500],
                "depth_mm": depth,
            }
        ]
        target = {
            "kind": "drop",
            "id": DROP_ID,
            "rect_mm": [FACE_X_MM, -1500, FACE_X_MM + 1500, 1500],
        }
        params.update(
            angle_deg=round(math.degrees(angle), 2), offset_m=round(offset, 3), depth_mm=depth
        )
    start = _start(face, offset, angle, distance)
    return Trial(cls, index, raw, start, target, params)


def _stairs_trial(index: int, rng: random.Random, distance: float) -> Trial:
    """The apartment's ungated stair hole, met from its open side (north, the landing) or its
    two flanks, straight or at up to ±40°."""
    from duckmind.world import homelayout

    layout = homelayout.load("apartment_ungated_test")
    hole = next(d for d in layout.drops if d.steps or d.id == "stairs")
    x0, y0, x1, y1 = (v / 1000.0 for v in hole.rect_mm)
    side = ("north", "west", "east")[index % 3]
    along = rng.uniform(0.15, 0.85)
    angle_off = math.radians(rng.uniform(-40.0, 40.0))
    if side == "north":  # the landing edge y1, approached heading −y
        hit = (x0 + along * (x1 - x0), y1)
        heading = -math.pi / 2 + angle_off
    elif side == "west":
        hit = (x0, y0 + along * (y1 - y0))
        heading = 0.0 + angle_off
    else:
        hit = (x1, y0 + along * (y1 - y0))
        heading = math.pi + angle_off
    start = _start(hit[0], hit[1], heading, distance)
    target = {"kind": "drop", "id": hole.id, "rect_mm": list(hole.rect_mm)}
    params = {
        "distance_m": round(distance, 3),
        "side": side,
        "angle_deg": round(math.degrees(angle_off), 2),
    }
    return Trial("stairs", index, "apartment_ungated_test", start, target, params)


# ── evaluator geometry (truth) ─────────────────────────────────────────────────


def _element(layout: Any, element_id: str) -> Any:
    for element in layout.elements:
        if element.id == element_id:
            return element
    raise KeyError(element_id)


def clearance_mm(
    layout: Any, target: Mapping[str, Any], point_m: tuple[float, float, float]
) -> int:
    """Horizontal distance (mm) from a world point (e.g. the beak tip) to the trial's obstacle:
    0 when the point is over it (or inside it)."""
    x, y = point_m[0] * 1000.0, point_m[1] * 1000.0
    if target["kind"] == "drop":
        x0, y0, x1, y1 = target["rect_mm"]
        dx = max(x0 - x, 0.0, x - x1)
        dy = max(y0 - y, 0.0, y - y1)
        return int(round(math.hypot(dx, dy)))
    element = _element(layout, str(target["id"]))
    ex, ey, _ = element.pos_mm
    if element.shape in ("cylinder", "sphere", "frustum"):
        r = max(element.radius_mm, element.radius_bottom_mm)
        return int(round(max(0.0, math.hypot(x - ex, y - ey) - r)))
    yaw = element.yaw_mrad / 1000.0
    c, s = math.cos(-yaw), math.sin(-yaw)
    lx = c * (x - ex) - s * (y - ey)
    ly = s * (x - ex) + c * (y - ey)
    dx = max(abs(lx) - element.size_mm[0] / 2.0, 0.0)
    dy = max(abs(ly) - element.size_mm[1] / 2.0, 0.0)
    return int(round(math.hypot(dx, dy)))


def inside_drop(
    target: Mapping[str, Any], point_m: tuple[float, float, float], margin_mm: float = 0.0
) -> bool:
    """Is the point over the drop's rectangle (grown by ``margin_mm``)?"""
    if target["kind"] != "drop":
        return False
    x0, y0, x1, y1 = target["rect_mm"]
    x, y = point_m[0] * 1000.0, point_m[1] * 1000.0
    return x0 - margin_mm <= x <= x1 + margin_mm and y0 - margin_mm <= y <= y1 + margin_mm


def obstacle_geoms(target: Mapping[str, Any]) -> tuple[str, ...]:
    """The MuJoCo geom names a contact with the obstacle shows up as."""
    if target["kind"] == "element":
        return (f"home_{target['id']}",)
    return ()


__all__ = [
    "BOX_COLOURS",
    "CLASSES",
    "CLASS_ZH",
    "DISTANCE_M",
    "DRIFT_TRIM_MRAD_S",
    "FACE_X_MM",
    "Trial",
    "base_room",
    "clearance_mm",
    "inside_drop",
    "obstacle_geoms",
    "trial",
]
