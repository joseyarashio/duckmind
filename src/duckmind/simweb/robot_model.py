"""Read the official Microduck MJCF so the simulator can draw the real robot.

The meshes and MJCF belong to pollen-robotics/microduck_rl (3D models licensed CC
BY-NC-SA). They are served straight from the owner's local checkout, named by
``DUCKMIND_MICRODUCK_MJCF`` (``--microduck-mjcf``), and are never copied into the
Duck Mind repository.
"""

from __future__ import annotations

import gzip
import os
import struct
import xml.etree.ElementTree as ElementTree
from functools import lru_cache
from pathlib import Path
from typing import Any

from duckmind.i18n import msg

ROOT_ENV = "DUCKMIND_MICRODUCK_MJCF"
ATTRIBUTION = msg("web.robot_model.ATTRIBUTION")
# Parts enclosed by the shells (battery, PCBs, brackets): upstream's own mesh bake drops
# them too (microduck scripts/bake-duck-mesh.py, HIDDEN). Invisible, and ~4 MB lighter.
HIDDEN_MESHES = frozenset(
    {
        "np_f970",
        "pcb__raspberry_pi_zero_2_w",
        "elec_rpi_robot_hat_pcb",
        "power_support",
        "banana_pcb_locker",
        "motor_support",
    }
)


def robot_root() -> Path | None:
    configured = os.environ.get(ROOT_ENV)
    return Path(configured).expanduser() if configured else None


def _floats(text: str | None, count: int, default: tuple[float, ...]) -> list[float]:
    if not text:
        return list(default)
    values = [float(part) for part in text.split()]
    if len(values) != count:
        raise ValueError(f"expected {count} numbers, got {text!r}")
    return values


def _body(element: ElementTree.Element, materials: dict[str, list[float]]) -> dict[str, Any]:
    joint = element.find("joint")
    geoms = []
    for geom in element.findall("geom"):
        if geom.get("class") != "visual" or geom.get("type") != "mesh":
            continue
        if geom.get("mesh") in HIDDEN_MESHES:
            continue
        geoms.append(
            {
                "mesh": geom.get("mesh"),
                "pos": _floats(geom.get("pos"), 3, (0.0, 0.0, 0.0)),
                "quat": _floats(geom.get("quat"), 4, (1.0, 0.0, 0.0, 0.0)),
                "rgba": materials.get(geom.get("material", ""), [0.8, 0.8, 0.8, 1.0]),
            }
        )
    return {
        "name": element.get("name"),
        "pos": _floats(element.get("pos"), 3, (0.0, 0.0, 0.0)),
        "quat": _floats(element.get("quat"), 4, (1.0, 0.0, 0.0, 0.0)),
        "joint": (
            {
                "name": joint.get("name"),
                "axis": _floats(joint.get("axis"), 3, (0.0, 0.0, 1.0)),
                "range": _floats(joint.get("range"), 2, (-3.14159, 3.14159)),
            }
            if joint is not None and joint.get("type", "hinge") == "hinge"
            else None
        ),
        "geoms": geoms,
        "children": [_body(child, materials) for child in element.findall("body")],
    }


def _rotate(quat: list[float], vector: tuple[float, float, float]) -> tuple[float, float, float]:
    w, x, y, z = quat
    vx, vy, vz = vector
    # v + 2w(q×v) + 2q×(q×v), with q the vector part.
    cx, cy, cz = y * vz - z * vy, z * vx - x * vz, x * vy - y * vx
    dx, dy, dz = y * cz - z * cy, z * cx - x * cz, x * cy - y * cx
    return (vx + 2 * (w * cx + dx), vy + 2 * (w * cy + dy), vz + 2 * (w * cz + dz))


def _find_body(body: dict[str, Any], name: str) -> dict[str, Any] | None:
    if body["name"] == name:
        return body
    for child in body["children"]:
        found = _find_body(child, name)
        if found is not None:
            return found
    return None


def _jaw_hinge(trunk: dict[str, Any], root: Path, meshes: dict[str, str]) -> dict[str, Any] | None:
    """Where the jaw pivots, in the head body's frame.

    The MJCF has no mouth joint (the real mouth is a servo; the jaw is a fixed geom), so
    the pin is read off the jaw mesh: its rear end is a round lobe around the pin. In the
    head body's frame the face points −z and +x is up, so the pin sits one lobe radius in
    front of the rearmost point, and turning about +y by a positive angle opens the mouth.
    """
    head = _find_body(trunk, "jaw_soft")
    geom = next((g for g in head["geoms"] if g["mesh"] == "jaw"), None) if head else None
    if geom is None or "jaw" not in meshes:
        return None
    data = (root / "assets" / meshes["jaw"]).read_bytes()
    (count,) = struct.unpack_from("<I", data, 80)
    if len(data) != 84 + count * 50:
        return None
    points = []
    for index in range(count):
        for corner in range(3):
            local = struct.unpack_from("<3f", data, 84 + index * 50 + 12 + corner * 12)
            rotated = _rotate(geom["quat"], local)
            points.append(tuple(a + b for a, b in zip(geom["pos"], rotated, strict=True)))
    rear = max(point[2] for point in points)
    depth = 0.004
    strip = [point[0] for point in points if point[2] > rear - depth]
    chord = max(strip) - min(strip)
    radius = (chord * chord / 4 + depth * depth) / (2 * depth)
    return {
        "body": "jaw_soft",
        "pos": [round(0.5 * (max(strip) + min(strip)), 5), 0.0, round(rear - radius, 5)],
        "axis": [0.0, 1.0, 0.0],
        "meshes": ["jaw", "jaw_soft"],
    }


@lru_cache(maxsize=1)
def load_model() -> dict[str, Any]:
    root = robot_root()
    if root is None:
        return {
            "available": False,
            "reason": msg("web.robot_model.load_model.074f38"),
            "attribution": ATTRIBUTION,
        }
    robot_file = root / "robot_walk.xml"
    scene_file = root / "scene.xml"
    if not robot_file.is_file():
        return {
            "available": False,
            "reason": msg("web.robot_model.load_model.dbaf79", robot_file=str(robot_file)),
            "attribution": ATTRIBUTION,
        }
    tree = ElementTree.parse(robot_file).getroot()
    asset = tree.find("asset")
    materials: dict[str, list[float]] = {}
    meshes: dict[str, str] = {}
    if asset is not None:
        for material in asset.findall("material"):
            materials[material.get("name", "")] = _floats(
                material.get("rgba"), 4, (0.8, 0.8, 0.8, 1.0)
            )
        for mesh in asset.findall("mesh"):
            file_name = mesh.get("file", "")
            meshes[mesh.get("name") or Path(file_name).stem] = file_name
    worldbody = tree.find("worldbody")
    trunk = worldbody.find("body") if worldbody is not None else None
    if trunk is None:
        return {
            "available": False,
            "reason": msg("web.robot_model.load_model.b008b4"),
            "attribution": ATTRIBUTION,
        }
    joint_order = [
        name
        for joint in tree.iter("joint")
        if (name := joint.get("name")) and joint.get("type", "hinge") == "hinge"
    ]
    keyframes: dict[str, dict[str, float]] = {}
    trunk_height: dict[str, float] = {}
    if scene_file.is_file():
        scene = ElementTree.parse(scene_file).getroot()
        for key in scene.iter("key"):
            values = [float(part) for part in (key.get("qpos") or "").split()]
            if len(values) == 7 + len(joint_order):
                keyframes[key.get("name", "")] = dict(zip(joint_order, values[7:], strict=True))
                trunk_height[key.get("name", "")] = values[2]
    body = _body(trunk, materials)
    return {
        "available": True,
        "attribution": ATTRIBUTION,
        "source": str(robot_file),
        "meshes": meshes,
        "trunk": body,
        "joints": joint_order,
        "keyframes": keyframes,
        "trunk_height": trunk_height,
        "jaw_hinge": _jaw_hinge(body, root, meshes),
    }


def mesh_path(name: str) -> Path | None:
    model = load_model()
    root = robot_root()
    if not model.get("available") or root is None:
        return None
    file_name = model["meshes"].get(name)
    if file_name is None:
        return None
    assets = (root / "assets").resolve()
    path = (assets / file_name).resolve()
    if path.parent != assets or not path.is_file():
        return None
    return path


@lru_cache(maxsize=64)
def _gzipped(path: Path) -> bytes:
    return gzip.compress(path.read_bytes(), 6)


def mesh_payload(name: str, accepts_gzip: bool) -> tuple[bytes, str | None] | None:
    """The STL bytes to send and their content encoding (binary STL gzips to ~40%)."""
    if name in HIDDEN_MESHES:
        return None
    path = mesh_path(name)
    if path is None:
        return None
    return (_gzipped(path), "gzip") if accepts_gzip else (path.read_bytes(), None)
