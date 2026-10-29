"""The official apartment, read where it lies and turned into layout elements at runtime.

Upstream's ``scene_apartment.xml`` is ``robot_allcollisions.xml`` plus ``apartment.xml``: six
rooms in x ∈ [−4, 4] m, y ∈ [−3, 3] m with 1.6 m walls, a four-step stair hole in the
corridor, furniture, a decorative dock, six visual landmarks and seven free objects. This
module parses ``apartment.xml`` with the standard XML reader (no MuJoCo, nothing copied or
cached in the repo) into ``home-layout-v2`` elements and things marked ``official``: their
ids are the official geom and body names, so :func:`duckmind.world.homelayout.compose`
finds them in the official scene and :meth:`~duckmind.world.homelayout.Layout.page_json`
draws them where the physics has them. Our additions (nest, lights, toys, railings; owner
decisions B1–B5) are in ``homes/apartment.overlay.json``.

:func:`check_pins` compares the official files with the sha256 they were reviewed at; a
difference is reported (the layout's ``warnings``), not refused — the same rule as the body
host's reviewed files. Names the overlay relies on that are gone from the official file
stop the load with a plain message.
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from duckmind.i18n import msg
from duckmind.world import homelayout

APARTMENT = "apartment.xml"
SCENE = "scene_apartment.xml"
# Reviewed 2026-09-27 at upstream microduck_rl commit cb70b792 (phase-2 survey).
PINNED_SHA256 = {
    APARTMENT: "4e6a6f5baf07ab842d9860a2e5baa72f516b864213cecb8184123d17ab23d7de",
    SCENE: "f89da842d8972db257cb9a1a7c1804e21e1d0c01cc3388d6de6ac6fa1060f6e2",
}

# Plain names for the official pieces, by name prefix (longest match first).
ZH = {
    "floor": msg("world.apartment.ZH.floor"),
    "stair": msg("world.apartment.ZH.stair"),
    "rug": msg("world.apartment.ZH.rug"),
    "wall": msg("world.apartment.ZH.wall"),
    "wA": msg("world.apartment.ZH.wA"),
    "wB": msg("world.apartment.ZH.wB"),
    "wC": msg("world.apartment.ZH.wC"),
    "wF": msg("world.apartment.ZH.wF"),
    "wG": msg("world.apartment.ZH.wG"),
    "kit_counter": msg("world.apartment.ZH.kit_counter"),
    "kit_fridge": msg("world.apartment.ZH.kit_fridge"),
    "kit_island": msg("world.apartment.ZH.kit_island"),
    "kit_stool": msg("world.apartment.ZH.kit_stool"),
    "liv_sofa": msg("world.apartment.ZH.liv_sofa"),
    "liv_table": msg("world.apartment.ZH.liv_table"),
    "liv_tv_stand": msg("world.apartment.ZH.liv_tv_stand"),
    "liv_shelf": msg("world.apartment.ZH.liv_shelf"),
    "liv_books": msg("world.apartment.ZH.liv_books"),
    "liv_lamp_pole": msg("world.apartment.ZH.liv_lamp_pole"),
    "liv_lamp_shade": msg("world.apartment.ZH.liv_lamp_shade"),
    "liv_chair": msg("world.apartment.ZH.liv_chair"),
    "liv_plant": msg("world.apartment.ZH.liv_plant"),
    "cor_console": msg("world.apartment.ZH.cor_console"),
    "cor_coat_rack": msg("world.apartment.ZH.cor_coat_rack"),
    "cor_shoes": msg("world.apartment.ZH.cor_shoes"),
    "bed_base": msg("world.apartment.ZH.bed_base"),
    "bed_pillow": msg("world.apartment.ZH.bed_pillow"),
    "bed_blanket": msg("world.apartment.ZH.bed_blanket"),
    "bed_ns": msg("world.apartment.ZH.bed_ns"),
    "bed_wardrobe": msg("world.apartment.ZH.bed_wardrobe"),
    "bed_dresser": msg("world.apartment.ZH.bed_dresser"),
    "off_desk": msg("world.apartment.ZH.off_desk"),
    "off_chair": msg("world.apartment.ZH.off_chair"),
    "off_shelf": msg("world.apartment.ZH.off_shelf"),
    "off_cabinet": msg("world.apartment.ZH.off_cabinet"),
    "off_plant": msg("world.apartment.ZH.off_plant"),
    "bth_tub": msg("world.apartment.ZH.bth_tub"),
    "bth_vanity": msg("world.apartment.ZH.bth_vanity"),
    "bth_mirror": msg("world.apartment.ZH.bth_mirror"),
    "bth_toilet": msg("world.apartment.ZH.bth_toilet"),
    "bth_rail": msg("world.apartment.ZH.bth_rail"),
    "dock": msg("world.apartment.ZH.dock"),
    "lm": msg("world.apartment.ZH.lm"),
    "obj": msg("world.apartment.ZH.obj"),
    "ball": msg("world.apartment.ZH.ball"),
}
WALL_PREFIXES = ("wall_", "wA", "wB", "wC", "wF", "wG")
FLOOR_TILES = ("floor_w", "floor_e", "floor_corridor_s", "floor_corridor_n")


def _floats(text: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if text is None:
        return default
    return tuple(float(v) for v in text.split())


def _mm(value: float) -> int:
    return int(round(value * 1000.0))


def _hex(rgb: tuple[float, ...]) -> str:
    return "#" + "".join(f"{max(0, min(255, int(round(v * 255)))):02x}" for v in rgb[:3])


def _zh(name: str) -> str:
    for prefix in sorted(ZH, key=len, reverse=True):
        if name.startswith(prefix):
            return ZH[prefix]
    return name


def _kind(name: str, collides: bool) -> str:
    if name in FLOOR_TILES:
        return "floor"
    if name.startswith("stair_"):
        return "stairs"
    if name.startswith("floor_"):
        return "floor_decor"
    if name.startswith("rug_"):
        return "rug"
    if name.startswith(WALL_PREFIXES):
        return "wall"
    if name.startswith("lm_"):
        return "landmark"
    if name.startswith("dock"):
        return "decor"
    return "furniture" if collides else "decor"


def _colours(root: ET.Element) -> dict[str, str]:
    """Material name → a flat colour (its rgba, or the mean of its texture's two colours)."""
    textures: dict[str, str] = {}
    materials: dict[str, str] = {}
    asset = root.find("asset")
    if asset is None:
        return materials
    for texture in asset.findall("texture"):
        a = _floats(texture.get("rgb1"), (0.8, 0.8, 0.8))
        b = _floats(texture.get("rgb2"), a)
        textures[texture.get("name", "")] = _hex(tuple((x + y) / 2 for x, y in zip(a, b)))
    for material in asset.findall("material"):
        name = material.get("name", "")
        if material.get("rgba"):
            materials[name] = _hex(_floats(material.get("rgba"), (0.8, 0.8, 0.8, 1)))
        elif material.get("texture") in textures:
            materials[name] = textures[material.get("texture", "")]
    return materials


def _shape(geom: ET.Element, offset: tuple[float, float, float]) -> dict[str, Any]:
    """An official geom as layout geometry (full sizes, the footprint centre and lowest z)."""
    kind = geom.get("type", "sphere")
    size = _floats(geom.get("size"), (0.0, 0.0, 0.0))
    px, py, pz = (a + b for a, b in zip(_floats(geom.get("pos"), (0.0, 0.0, 0.0)), offset))
    if geom.get("quat") or geom.get("euler") or geom.get("axisangle") or geom.get("fromto"):
        raise ValueError(
            f"upstream geom {geom.get('name')} is turned; the converter reads only upright ones"
        )
    if kind == "box":
        return {
            "shape": "box",
            "pos_mm": [_mm(px), _mm(py), _mm(pz - size[2])],
            "size_mm": [_mm(2 * size[0]), _mm(2 * size[1]), _mm(2 * size[2])],
        }
    if kind == "ellipsoid":
        return {
            "shape": "ellipsoid",
            "pos_mm": [_mm(px), _mm(py), _mm(pz - size[2])],
            "size_mm": [_mm(2 * size[0]), _mm(2 * size[1]), _mm(2 * size[2])],
        }
    if kind == "cylinder":
        return {
            "shape": "cylinder",
            "pos_mm": [_mm(px), _mm(py), _mm(pz - size[1])],
            "radius_mm": _mm(size[0]),
            "height_mm": _mm(2 * size[1]),
        }
    if kind == "sphere":
        return {
            "shape": "sphere",
            "pos_mm": [_mm(px), _mm(py), _mm(pz - size[0])],
            "radius_mm": _mm(size[0]),
        }
    if kind == "capsule":  # upright: its axis along z
        return {
            "shape": "capsule",
            "radius_mm": _mm(size[0]),
            "fromto_mm": [_mm(px), _mm(py), _mm(pz - size[1]), _mm(px), _mm(py), _mm(pz + size[1])],
        }
    raise ValueError(
        f"upstream geom {geom.get('name')} has a shape the converter does not read: {kind}"
    )


def _collides(geom: ET.Element) -> bool:
    return geom.get("contype", "1") != "0" or geom.get("conaffinity", "1") != "0"


def convert(mjcf_dir: Path | str) -> dict[str, Any]:
    """The official apartment as layout elements and things (``official: true``)."""
    path = Path(mjcf_dir).expanduser() / APARTMENT
    root = ET.parse(path).getroot()
    colours = _colours(root)
    world = root.find("worldbody")
    if world is None:
        raise ValueError(f"{path} has no worldbody")
    elements: list[dict[str, Any]] = []
    things: list[dict[str, Any]] = []

    def colour_of(geom: ET.Element) -> str:
        if geom.get("rgba"):
            return _hex(_floats(geom.get("rgba"), (0.8, 0.8, 0.8)))
        return colours.get(geom.get("material", ""), "#cccccc")

    def add_geom(geom: ET.Element, offset: tuple[float, float, float], body: str = "") -> None:
        name = geom.get("name", "")
        collides = _collides(geom)
        kind = _kind(name, collides)
        element: dict[str, Any] = {
            "id": name,
            "zh": _zh(name),
            "kind": kind,
            "cls": "furniture" if collides else "",
            "official": True,
            "colour": colour_of(geom),
            "collide": collides,
            "camera": True,
            "tof": True,
            **_shape(geom, offset),
        }
        if body:
            element["body"] = body
        if collides and element["shape"] == "cylinder" and element["radius_mm"] <= 50:
            element["cls"] = "furniture_leg"
        elements.append(element)

    for geom in world.findall("geom"):
        add_geom(geom, (0.0, 0.0, 0.0))
    for body in world.findall("body"):
        name = body.get("name", "")
        offset = _floats(body.get("pos"), (0.0, 0.0, 0.0))
        pos = (offset[0], offset[1], offset[2])
        geoms = body.findall("geom")
        if body.find("freejoint") is None and body.find("joint") is None:
            for geom in geoms:
                add_geom(geom, pos, body=name)
            continue
        if len(geoms) != 1:
            raise ValueError(f"upstream free body {name} has {len(geoms)} geoms; expected one")
        geom = geoms[0]
        shape = _shape(geom, pos)
        mass = _mm(float(geom.get("mass", "0")))  # kg → g
        if shape["shape"] == "box":
            part: dict[str, Any] = {"shape": "box", "size_mm": shape["size_mm"]}
            cls = "box"
        elif shape["shape"] == "sphere":
            part = {"shape": "sphere", "radius_mm": shape["radius_mm"]}
            cls = "ball_slow"
        else:
            raise ValueError(f"upstream free body {name} is a {shape['shape']}")
        things.append(
            {
                "id": name,
                "zh": _zh(name),
                "cls": cls,
                "official": True,
                "pos_mm": shape["pos_mm"],
                "mass_g": mass,
                "parts": [{**part, "mass_g": mass}],
                "colour": colour_of(geom),
                "verbs": ["look", "bump", "peck"],
                "registered": True,
            }
        )
    return {"elements": elements, "things": things}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_pins(mjcf_dir: Path | str, pins: dict[str, str] | None = None) -> list[str]:
    """Plain-words warnings for official files that differ from the reviewed ones."""
    folder = Path(mjcf_dir).expanduser()
    warnings = []
    for name, digest in (pins or PINNED_SHA256).items():
        path = folder / name
        if not path.is_file():
            warnings.append(f"上游的 {name} 不見了")
        elif sha256(path) != digest:
            warnings.append(f"上游的 {name} 改過了（和檢查過的版本不同），這個家的物理要重新檢查")
    return warnings


def raw_layout(mjcf_dir: Path | str) -> homelayout.Layout:
    """The official apartment exactly as upstream has it (no overlay: nothing hidden, nothing
    moved, no nest, no lights), as a :class:`~duckmind.world.homelayout.Layout` — what the
    curiosity regression test checks: the generation checks must catch the cube inside the
    kitchen island (``obj_1``) and the ball on the stair-hole edge (``ball_2``)."""
    official = convert(mjcf_dir)
    return homelayout.load(
        {
            "schema": homelayout.SCHEMA,
            "id": "apartment_official",
            "zh": msg("world.apartment.raw_layout.6c20c630"),
            "base": {
                "world": APARTMENT,
                "scenes": {"allcollisions": SCENE},
                "converted": "apartment",
            },
            "bounds_mm": [-4060, -3060, 4060, 3060],
            "ceiling": False,
            "spawn": {"x_mm": 0, "y_mm": 0, "heading_mrad": 0},
            **official,
        },
        mjcf_dir=mjcf_dir,
    )
