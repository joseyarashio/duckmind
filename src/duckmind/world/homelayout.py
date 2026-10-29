"""Home layouts (``home-layout-v2``): one JSON per home, read by the physics, the page and the
sensors alike (physical-design spec §12.1–§12.2, curiosity spec §4; phase 2).

A layout is the home the duck lives in, in the MuJoCo world frame: x and y on the floor, z up,
heading 0 = +x, counter-clockwise; integers only (millimetres, milliradians, grams), converted
to metres and radians at the edges. The same resolved layout gives

* :func:`compose` — ONE ``mujoco.MjSpec`` per home: the official scene read from the upstream
  checkout by path (never copied, never edited), plus this home's walls, furniture, nest,
  things, floor tiles and barriers added in memory;
* :meth:`Layout.page_json` — the simulator page's room: every element with its shape, pose,
  size, colour, flags and room, in the same world coordinates;
* :attr:`Layout.lights` — per-room lamps and windows for :mod:`duckmind.world.lights`;
* :attr:`Layout.nest`, :attr:`Layout.mat`, :attr:`Layout.spawn`, :attr:`Layout.frame` — where
  the charging place and the family's mat are, where the physics duck starts, and where Duck
  Core's own origin lies in the world (identity in ``default_home``).

**The charging place** (owner decision 2026-09-27) is a robot-vacuum-style dock with its back
on a wall (``nest.kind`` "dock", :mod:`duckmind.world.dock`; its ``style`` "forward" or
"backin" is how the duck docks); phase 2's open nest stays as ``kind`` "open" (the
``open_nest`` variants). A layout may name Duck Core's origin with ``frame`` ("room" keeps it at
the world's origin, as ``default_home`` does so that Duck Core's imagined walls are its walls:
there a nest duck believes its nest is at (0, 0), not at the dock, until phase 5).

Homes live in ``world/homes/``: ``default_home.json`` (the page's 6×6 m room, made physical),
``hazard_home.json`` (thresholds, a low box, drops, a stair hole and the curiosity hazard kit)
and ``apartment.overlay.json`` (the official apartment, read at runtime by
:mod:`duckmind.world.apartment`, plus our overlay: nest, lights, toys, the stair railings).

**Flags and geom groups.** Every element says whether it collides, whether the head camera
sees it and whether the ToF sees it. Collision is ``contype``/``conaffinity`` 1 or 0; what the
sensors see is the MuJoCo geom group (:func:`geom_group`):

====== ========================== ==========================================================
group  camera / ToF               used for
====== ========================== ==========================================================
0      seen by both               walls, furniture, floors, things (all official world geoms)
1      camera only                decor a ToF must not report (window glass, the mat, the
                                  official visual-only landmarks)
2, 3   —                          the official robot: visual meshes (2), colliders (3)
4      ToF only                   (reserved: invisible to the camera, seen by the ToF)
5      neither                    sensor helpers (the nest owner's electrode and probe)
====== ========================== ==========================================================

The camera renders groups 0–2 (MuJoCo's default); a ToF ray should use
:data:`TOF_GEOMGROUP` (groups 0 and 4; whether it sees the duck's own body is the ToF model's
choice). The official ``sim/tof.py`` casts with ``geomgroup=None`` and therefore sees
everything, visual-only decor included.

**Keyframes.** The official keyframes are padded with every free object at the origin, so a
reset to ``STAND`` would pile every thing onto the duck. :func:`compose` rewrites every
keyframe (:func:`fix_keyframes`): the duck at the layout's spawn, every free object and mocap
body where the layout put it. Anyone who adds free or mocap bodies after :func:`compose`
must call :func:`fix_keyframes` again before compiling (or never reset to a keyframe).

Nothing here imports MuJoCo at import time; loading, checking and the page JSON are plain
Python, so the simulator page and the standard test run need neither MuJoCo nor numpy.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import weakref
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from duckmind.i18n import msg

SCHEMA = "home-layout-v2"
PAGE_SCHEMA = "home-page-v1"
HOMES_DIR = Path(__file__).with_name("homes")
MJCF_ENV = "DUCKMIND_MICRODUCK_MJCF"  # the official microduck MJCF folder (as the tests use it)

# collision model → the official scene a home on the official floor starts from, and the robot
# file that scene includes (the same pairs as duckmind.bodyhost.COLLISIONS).
SCENES: dict[str, str] = {
    "allcollisions": "scene_allcollisions.xml",
    "groundcontact": "scene.xml",
}
ROBOT_FILES: dict[str, str] = {
    "allcollisions": "robot_allcollisions.xml",
    "groundcontact": "robot_groundcontact.xml",
}
ROBOT_ROOT = "trunk_base"
ROBOT_FREEJOINT = "trunk_base_freejoint"
OFFICIAL_FLOOR = "floor"
# The official plane's friction (MuJoCo's default: the scenes set none). Floor tiles copy it, so
# a home with drops walks exactly like the official floor.
OFFICIAL_FLOOR_FRICTION = (1.0, 0.005, 0.0001)
BODY_TIMESTEP_S = 0.005  # mjlab_microduck.sim.body_server.TIMESTEP (4 steps per robotd tick)

# Geom groups (the table above).
GROUP_SEEN = 0
GROUP_CAMERA = 1
ROBOT_VISUAL_GROUP = 2
ROBOT_COLLISION_GROUP = 3
GROUP_TOF = 4
GROUP_HIDDEN = 5
CAMERA_GEOMGROUP = (1, 1, 1, 0, 0, 0)
TOF_GEOMGROUP = (1, 0, 0, 0, 1, 0)

# MuJoCo 3.10's classic renderer on this Mac lights only the first 7 active scene lights
# (phase-2 survey, probe_lights.py): a renderer lights the camera's room and treats the rest as
# ambient. A room may hold at most LIGHT_LIMIT − 1 sources, one slot staying for the sun/moon.
LIGHT_LIMIT = 7
ROOM_LIGHT_MAX = LIGHT_LIMIT - 1

# Placement rules (curiosity spec §4.1, physical-design spec §10.5 / §12.1).
DUCK_RADIUS_MM = 140  # the standing duck's footprint (collision geoms: x −57…102, y ±93 mm)
DUCK_HEIGHT_MM = 280  # standing (STAND keyframe: 278 mm)
DROP_BUFFER_MM = 600  # things and the nest stay this far from a drop
NEST_CLEAR_MM = 300  # nothing within this of the nest pad's edge
NEST_MARGIN_MM = 20  # the nest's clearance boxes keep this far from colliders
# The open nest's clearance, in the nest frame (x toward the room, y to the left; §10.5 and
# the survey's nest_site2.py): the backboard, gap and pad; and the approach / right-arc exit.
NEST_BACK_MM = (-430, 200, -200, 200)  # x0, x1, y0, y1 (x0 follows the board: back_box)
NEST_EXIT_MM = (0, 600, -500, 200)
# Pad centre → backboard centre. Phase 2 measured the scripted right-arc exit of a duck seated
# facing the board (the way it walks in, spec §10.5 item 1): with the board at 410 mm its beak
# touched the board in 4/25 live exits (and 2/30 headless); at 480 mm 25/25 live and 30/30
# headless were clean (docs/validation/evidence/phase2/gate_nest_exit.json). §10.5 item 5's
# "≥ 0.40 m" initial value is settled at 480 mm.
NEST_BOARD_OFFSET_MM = 480
BARRIER_GAP_MM = 60  # a railing's balusters leave at most this much between them
BARRIER_REACH_MM = 60  # a barrier guards a drop edge when it stands within this of it
BARRIER_MIN_HEIGHT_MM = 400
TOLERANCE_MM = 1

NEST_EDGES = ("edge_5mm", "edge_3mm", "ramp")
# The charging place (owner decision 2026-09-27): a robot-vacuum-style dock against a wall
# (duckmind.world.dock), or phase 2's open nest (the ``open_nest`` variant, for comparison).
NEST_KINDS = ("dock", "open")
WALL_KINDS = frozenset({"wall", "skirting"})  # what a dock's back may stand against
DOCK_WALL_REACH_MM = 30  # a dock's back stands at most this far from its wall
# What a placement may not reach through (Layout.barrier_between) and plain names for pieces
# without their own (Layout.zh_of).
BARRIER_KINDS = frozenset({"wall", "railing", "gate"})
KIND_ZH = {
    "wall": msg("world.homelayout.KIND_ZH.wall"),
    "skirting": msg("world.homelayout.KIND_ZH.skirting"),
    "floor": msg("world.homelayout.KIND_ZH.floor"),
    "furniture": msg("world.homelayout.KIND_ZH.furniture"),
    "railing": msg("world.homelayout.KIND_ZH.railing"),
    "gate": msg("world.homelayout.KIND_ZH.gate"),
    "stairs": msg("world.homelayout.KIND_ZH.stairs"),
    "threshold": msg("world.homelayout.KIND_ZH.threshold"),
    "platform": msg("world.homelayout.KIND_ZH.platform"),
    "cable": msg("world.homelayout.KIND_ZH.cable"),
    "nest_pad": msg("world.homelayout.KIND_ZH.nest_pad"),
    "nest_board": msg("world.homelayout.KIND_ZH.nest_board"),
    "dock_housing": msg("world.homelayout.KIND_ZH.dock_housing"),
    "dock_face": msg("world.homelayout.KIND_ZH.dock_face"),
    "dock_plate": msg("world.homelayout.KIND_ZH.dock_plate"),
    "dock_guide": msg("world.homelayout.KIND_ZH.dock_guide"),
}
SHAPES = ("box", "cylinder", "frustum", "sphere", "ellipsoid", "capsule", "plane", "slab")
SUPPORT_KINDS = frozenset(
    {"floor", "stairs", "platform", "pit_bottom", "nest_pad", "dock_plate", "rug"}
)  # (a rug that collides is floor to stand a thing on; the drawn-only rug is not felt)
# Floor-level or high decor the placement checks ignore.
DECOR_KINDS = frozenset(
    {"rug", "mat", "floor_decor", "window", "ceiling", "wall_upper", "nest_tag", "nest_led"}
)

VERB_ZH = {
    "look": msg("world.homelayout.VERB_ZH.look"),
    "bump": msg("world.homelayout.VERB_ZH.bump"),
    "peck": msg("world.homelayout.VERB_ZH.peck"),
    "tap": msg("world.homelayout.VERB_ZH.tap"),
    "grab": msg("world.homelayout.VERB_ZH.grab"),
    "carry": msg("world.homelayout.VERB_ZH.carry"),
    "kick": msg("world.homelayout.VERB_ZH.kick"),
    "dribble": msg("world.homelayout.VERB_ZH.dribble"),
}
# Curiosity spec §4.3 THING_CLASSES: the plain name of each class, for the page's legend.
CLASS_ZH = {
    "ball_slow": msg("world.homelayout.CLASS_ZH.ball_slow"),
    "ball_fast": msg("world.homelayout.CLASS_ZH.ball_fast"),
    "handle_toy": msg("world.homelayout.CLASS_ZH.handle_toy"),
    "light_toy": msg("world.homelayout.CLASS_ZH.light_toy"),
    "box": msg("world.homelayout.CLASS_ZH.box"),
    "cup": msg("world.homelayout.CLASS_ZH.cup"),
    "tall_light": msg("world.homelayout.CLASS_ZH.tall_light"),
    "heavy": msg("world.homelayout.CLASS_ZH.heavy"),
    "furniture": msg("world.homelayout.CLASS_ZH.furniture"),
    "furniture_leg": msg("world.homelayout.CLASS_ZH.furniture_leg"),
    "tall_thin": msg("world.homelayout.CLASS_ZH.tall_thin"),
    "unknown": msg("world.homelayout.CLASS_ZH.unknown"),
    "no_touch": msg("world.homelayout.CLASS_ZH.no_touch"),
    "fragile": msg("world.homelayout.CLASS_ZH.fragile"),
    "cable": msg("world.homelayout.CLASS_ZH.cable"),
    "pet_bowl": msg("world.homelayout.CLASS_ZH.pet_bowl"),
    "food": msg("world.homelayout.CLASS_ZH.food"),
    "small_part": msg("world.homelayout.CLASS_ZH.small_part"),
}


class LayoutError(ValueError):
    """A layout that cannot be used, said in plain words."""


# ── poses ───────────────────────────────────────────────────────────────────────


def _rot(x: float, y: float, mrad: float) -> tuple[float, float]:
    angle = mrad / 1000.0
    c, s = math.cos(angle), math.sin(angle)
    return c * x - s * y, s * x + c * y


def yaw_quat(heading_mrad: float) -> list[float]:
    """MuJoCo quaternion (w, x, y, z) of a turn about z."""
    half = heading_mrad / 2000.0
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def _wrap_mrad(value: float) -> int:
    turn = 2000.0 * math.pi
    return int(round((value + turn / 2) % turn - turn / 2))


@dataclass(frozen=True)
class Pose:
    """A place and heading on the floor, in the world frame (mm, mrad)."""

    x_mm: int
    y_mm: int
    heading_mrad: int = 0
    z_mm: int = 0  # what it stands on (a nest pad lifts the duck by its thickness)

    @property
    def x(self) -> float:
        return self.x_mm / 1000.0

    @property
    def y(self) -> float:
        return self.y_mm / 1000.0

    @property
    def z(self) -> float:
        return self.z_mm / 1000.0

    @property
    def yaw(self) -> float:
        return self.heading_mrad / 1000.0

    def to_world(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        """A point given in this pose's frame, in the world (mm, not rounded)."""
        dx, dy = _rot(x_mm, y_mm, self.heading_mrad)
        return self.x_mm + dx, self.y_mm + dy

    def to_local(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        """A world point in this pose's frame (mm, not rounded)."""
        return _rot(x_mm - self.x_mm, y_mm - self.y_mm, -self.heading_mrad)

    def as_json(self) -> dict[str, int]:
        return {
            "x_mm": self.x_mm,
            "y_mm": self.y_mm,
            "heading_mrad": self.heading_mrad,
            "z_mm": self.z_mm,
        }


@dataclass(frozen=True)
class Nest(Pose):
    """The duck's charging place (spec §10.5; owner decisions C1–C4 and 2026-09-27).

    ``kind`` "dock" (the default since 2026-09-27): a robot-vacuum-style dock against a wall
    (:mod:`duckmind.world.dock`); its pose is the middle of the housing's back on the wall,
    heading into the room, ``style`` is how the duck docks ("backin", the homes' since
    2026-09-28, or "forward"), and ``side`` which side of the dock a backin duck turns round on
    ("right", or "left" where the room has a wall there: duck-body's ``duck.dock`` side).
    ``kind`` "open": phase 2's open nest; its pose is the pad centre, heading toward the room
    (the backboard is behind it, at −x in the nest frame).

    Either way :attr:`seat` is where a docked duck sits (and which way it faces) and
    :attr:`home` is where Duck Core's own nest (0, 0) lies when the layout's frame is its nest."""

    pad_radius_mm: int = 200
    edge: str = "edge_5mm"  # edge_5mm | edge_3mm | ramp (C2: all three built, measured later)
    ramp_radius_mm: int = 250  # the ramp variant's foot
    contacts_radius_mm: int = 85  # the belly contact ring (drawn; the electrode is nestpad's)
    board_offset_mm: int = NEST_BOARD_OFFSET_MM  # pad centre → backboard centre
    board_size_mm: tuple[int, int, int] = (20, 300, 220)
    board_colour: str = "#12b5c4"  # cyan (C3)
    pad_colour: str = "#e8d9b5"
    tag_mm: int = 100
    edge_tag_mm: int = 40
    backlight: Mapping[str, Any] = field(default_factory=dict)
    kind: str = "open"  # dock | open
    style: str = ""  # a dock's docking style: forward | backin
    shape_mm: Mapping[str, int] = field(default_factory=dict)  # a dock's geometry overrides
    side: str = ""  # a dock's turning side (backin's J-hook): right | left

    @property
    def pose(self) -> Pose:
        return Pose(self.x_mm, self.y_mm, self.heading_mrad, self.z_mm)

    @property
    def is_dock(self) -> bool:
        return self.kind == "dock"

    @property
    def dock_shape(self) -> Any:
        """The dock's geometry (a :class:`duckmind.world.dock.DockShape`), or None (open)."""
        if not self.is_dock:
            return None
        from duckmind.world import dock

        return dock.shape(self.style, self.shape_mm)

    @property
    def pad_height_mm(self) -> int:
        """What a docked duck sits on: the open nest's pad, or the dock's plate (0: flush)."""
        if self.is_dock:
            return int(self.dock_shape.plate_mm)
        return 3 if self.edge == "edge_3mm" else 5

    @property
    def seat(self) -> Pose:
        """Where a docked, seated duck's trunk is, the way it faces, and what it sits on (z)."""
        if not self.is_dock:
            return Pose(self.x_mm, self.y_mm, self.heading_mrad, self.pad_height_mm)
        from duckmind.world import dock

        x, y = self.to_world(dock.seat_x_mm(self.style, self.dock_shape), 0)
        heading = _wrap_mrad(self.heading_mrad + dock.seat_heading_mrad(self.style))
        return Pose(int(round(x)), int(round(y)), heading, self.pad_height_mm)

    @property
    def home(self) -> Pose:
        """Duck Core's own nest (0, 0) in the world when the layout's frame is the nest: the
        seat, heading into the room (Duck Core: "the nest's opening is toward +x")."""
        if not self.is_dock:
            return self.pose
        seat = self.seat
        return Pose(seat.x_mm, seat.y_mm, self.heading_mrad)

    @property
    def keepout_mm(self) -> tuple[float, float, float, float]:
        """World rectangle (x0, y0, x1, y1) people keep out of: the pad, or the whole dock."""
        if not self.is_dock:
            r = self.pad_radius_mm + 20
            return (self.x_mm - r, self.y_mm - r, self.x_mm + r, self.y_mm + r)
        from duckmind.world import dock

        return _local_box(self.pose, dock.box_mm(self.dock_shape))

    def zones_mm(self) -> dict[str, tuple[int, int, int, int]]:
        """The clearance boxes in the nest frame (x0, x1, y0, y1): the nest itself (``back``)
        and where the duck comes and goes (open: ``exit``; dock: ``lane`` and ``turn``)."""
        if not self.is_dock:
            return {"back": self.back_box_mm, "exit": NEST_EXIT_MM}
        from duckmind.world import dock

        return dock.clearance_mm(self.dock_shape)

    def as_json(self) -> dict[str, Any]:
        out: dict[str, Any] = dict(super().as_json())
        seat, home = self.seat, self.home
        out.update(
            kind=self.kind,
            style=self.style or None,
            side=self.side or None,
            seat=seat.as_json(),
            home=home.as_json(),
            pad_height_mm=self.pad_height_mm,
            tag_mm=self.tag_mm,
            edge_tag_mm=self.edge_tag_mm,
            backlight=dict(self.backlight),
        )
        if self.is_dock:
            zones = self.zones_mm()
            out.update(
                shape=self.dock_shape.as_json(),
                clearance_back_mm=list(zones["back"]),
                clearance_lane_mm=list(zones["lane"]),
                clearance_turn_mm=list(zones["turn"]),
                keepout_mm=[int(round(v)) for v in self.keepout_mm],
            )
            return out
        out.update(
            pad_radius_mm=self.pad_radius_mm,
            edge=self.edge,
            ramp_radius_mm=self.ramp_radius_mm,
            contacts_radius_mm=self.contacts_radius_mm,
            board_offset_mm=self.board_offset_mm,
            board_size_mm=list(self.board_size_mm),
            board_colour=self.board_colour,
            pad_colour=self.pad_colour,
            clearance_back_mm=list(self.back_box_mm),
            clearance_exit_mm=list(NEST_EXIT_MM),
        )
        return out

    @property
    def back_box_mm(self) -> tuple[int, int, int, int]:
        """The nest itself in its frame (x0, x1, y0, y1). Open: the backboard, the gap and the
        pad (NEST_BACK_MM reaching back to just behind this nest's board). Dock: the housing,
        the plate and the guides."""
        if self.is_dock:
            return self.zones_mm()["back"]
        _, x1, y0, y1 = NEST_BACK_MM
        back = -(self.board_offset_mm + self.board_size_mm[0] // 2 + 10)
        return (min(NEST_BACK_MM[0], back), x1, y0, y1)


@dataclass(frozen=True)
class Mat(Pose):
    """The family's mat on the floor, where the duck is put down (owner decision A3)."""

    radius_mm: int = 340
    colour: str = "#9fb8b0"
    drawn: bool = True  # False where a nest lies on it (default_home: the nest replaces it)

    @property
    def pose(self) -> Pose:
        return Pose(self.x_mm, self.y_mm, self.heading_mrad, self.z_mm)

    def as_json(self) -> dict[str, Any]:
        out: dict[str, Any] = dict(super().as_json())
        out.update(radius_mm=self.radius_mm, colour=self.colour, drawn=self.drawn)
        return out


# ── what a home is made of ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class Room:
    id: str
    zh: str
    rect_mm: tuple[int, int, int, int]  # x0, y0, x1, y1
    floor: str = ""

    def contains(self, x_mm: float, y_mm: float, grow_mm: float = 0.0) -> bool:
        x0, y0, x1, y1 = self.rect_mm
        return x0 - grow_mm <= x_mm <= x1 + grow_mm and y0 - grow_mm <= y_mm <= y1 + grow_mm

    def as_json(self) -> dict[str, Any]:
        return {"id": self.id, "zh": self.zh, "rect_mm": list(self.rect_mm), "floor": self.floor}


@dataclass(frozen=True)
class Element:
    """One fixed piece of the home (a wall, a sofa part, a floor tile, a nest part…).

    ``pos_mm`` is the footprint centre (x, y) and the element's lowest point (z); ``size_mm``
    the full extents of its unrotated bounding box (a cylinder: 2r, 2r, height). ``yaw_mrad``
    turns it about its vertical axis. A capsule is given by its axis (``fromto_mm``). A slab is
    a low plate whose every side is a ramp ``ramp_mm`` long: ``size_mm`` is its foot, its top
    ``ramp_mm`` smaller on every side (the dock's floor plate)."""

    id: str
    kind: str
    shape: str
    pos_mm: tuple[int, int, int]
    size_mm: tuple[int, int, int]
    zh: str = ""
    cls: str = ""
    yaw_mrad: int = 0
    radius_mm: int = 0
    radius_bottom_mm: int = 0  # a frustum's foot
    fromto_mm: tuple[int, ...] = ()
    colour: str = "#cccccc"
    opacity: float = 1.0
    texture: str = ""
    collide: bool = True
    camera: bool = True
    tof: bool = True
    page: bool = True
    room: str | None = None
    official: bool = False
    under_mm: int = 0
    friction: tuple[float, float, float] | None = None
    ramp_mm: int = 0  # a slab's ramps
    lib: str = ""  # the object library item it came from (objlib), "" for a home's own piece
    place: str = ""  # the placement it belongs to (one placed item can be several elements)
    passability: str = ""  # the library's class for it (objlib.CLASSES)
    solref: tuple[float, float] | None = None  # contact softness (MuJoCo solref), None: rigid
    solmix: float | None = None  # how much this geom's solref counts against the other geom's

    @property
    def geom(self) -> str:
        """Its MuJoCo geom name: the official name, or ``home_<id>``."""
        return self.id if self.official else f"home_{self.id}"

    @property
    def group(self) -> int:
        return geom_group(self.camera, self.tof)

    @property
    def support(self) -> bool:
        return self.collide and self.kind in SUPPORT_KINDS

    @property
    def top_mm(self) -> int:
        return self.pos_mm[2] + self.size_mm[2]

    def z_range(self) -> tuple[float, float]:
        return float(self.pos_mm[2]), float(self.pos_mm[2] + self.size_mm[2])

    def footprint(self) -> tuple[float, float, float, float]:
        """World bounding rectangle on the floor (x0, y0, x1, y1), mm."""
        x, y, _ = self.pos_mm
        if self.shape == "capsule" and len(self.fromto_mm) == 6:
            ax, ay, _, bx, by, _ = self.fromto_mm
            r = self.radius_mm
            return min(ax, bx) - r, min(ay, by) - r, max(ax, bx) + r, max(ay, by) + r
        if self.shape in ("cylinder", "sphere", "frustum"):
            r = max(self.radius_mm, self.radius_bottom_mm)
            return x - r, y - r, x + r, y + r
        return _rect_footprint(x, y, self.size_mm[0], self.size_mm[1], self.yaw_mrad)

    def contains_xy(self, x_mm: float, y_mm: float) -> bool:
        """Is the floor point (x, y) under this element's own outline (not just its box)?"""
        x, y, _ = self.pos_mm
        if self.shape in ("cylinder", "sphere", "frustum"):
            r = max(self.radius_mm, self.radius_bottom_mm)
            return math.hypot(x_mm - x, y_mm - y) <= r
        if self.shape == "capsule":
            x0, y0, x1, y1 = self.footprint()
            return x0 <= x_mm <= x1 and y0 <= y_mm <= y1
        lx, ly = _rot(x_mm - x, y_mm - y, -self.yaw_mrad)
        return abs(lx) <= self.size_mm[0] / 2 and abs(ly) <= self.size_mm[1] / 2

    def as_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "geom": self.geom,
            "zh": self.zh,
            "kind": self.kind,
            "cls": self.cls,
            "room": self.room,
            "shape": self.shape,
            "pos_mm": list(self.pos_mm),
            "yaw_mrad": self.yaw_mrad,
            "size_mm": list(self.size_mm),
            "colour": self.colour,
            "opacity": self.opacity,
            "flags": {
                "collide": self.collide,
                "camera": self.camera,
                "tof": self.tof,
                "page": self.page,
            },
            "official": self.official,
        }
        if self.shape in ("cylinder", "sphere", "frustum", "capsule"):
            out["radius_mm"] = self.radius_mm
        if self.shape == "frustum":
            out["radius_bottom_mm"] = self.radius_bottom_mm
        if self.fromto_mm:
            out["fromto_mm"] = list(self.fromto_mm)
        if self.texture:
            out["texture"] = self.texture
        if self.under_mm:
            out["under_mm"] = self.under_mm
        if self.shape == "slab":
            out["ramp_mm"] = self.ramp_mm
        if self.lib:
            out["lib"] = self.lib
            out["place"] = self.place
            out["passability"] = self.passability
            if self.friction is not None:
                out["friction"] = list(self.friction)
            if self.solref is not None:
                out["solref"] = list(self.solref)
                out["solmix"] = self.solmix
        return out


@dataclass(frozen=True)
class Part:
    """One solid of a thing, relative to the thing's footprint centre and floor contact."""

    shape: str
    size_mm: tuple[int, int, int]
    offset_mm: tuple[int, int, int] = (0, 0, 0)  # x, y, and the part's lowest point
    radius_mm: int = 0
    mass_g: int = 0

    def as_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "shape": self.shape,
            "size_mm": list(self.size_mm),
            "offset_mm": list(self.offset_mm),
            "mass_g": self.mass_g,
        }
        if self.shape in ("sphere", "cylinder"):
            out["radius_mm"] = self.radius_mm
        return out


@dataclass(frozen=True)
class Thing:
    """Something that can be moved: a toy, a cube, a glass (a free body in the physics)."""

    id: str
    zh: str
    cls: str
    pos_mm: tuple[int, int, int]  # footprint centre, and its lowest point
    parts: tuple[Part, ...]
    mass_g: int
    yaw_mrad: int = 0
    colour: str = "#cccccc"
    opacity: float = 1.0
    verbs: tuple[str, ...] = ()
    registered: bool = True
    flags: tuple[str, ...] = ()
    friction: float = 0.8
    roll: float = 0.0002  # rolling friction (MuJoCo takes the larger of the two geoms')
    hollow: bool = False
    official: bool = False
    room: str | None = None
    lib: str = ""  # the object library item it came from (objlib), "" for a home's own thing
    passability: str = ""  # the library's class for it
    solref: tuple[float, float] | None = None  # contact softness (MuJoCo solref), None: rigid
    solmix: float | None = None  # how much this geom's solref counts against the other geom's

    @property
    def body(self) -> str:
        """Its MuJoCo body name: the official name, or ``thing_<id>`` (free joint
        ``<body>_free``, geoms ``<body>_<k>``)."""
        return self.id if self.official else f"thing_{self.id}"

    @property
    def hazard(self) -> bool:
        return "hazard" in self.flags

    @property
    def height_mm(self) -> int:
        return max(p.offset_mm[2] + p.size_mm[2] for p in self.parts)

    def footprint(self) -> tuple[float, float, float, float]:
        x0 = y0 = math.inf
        x1 = y1 = -math.inf
        for part in self.parts:
            px, py = _rot(part.offset_mm[0], part.offset_mm[1], self.yaw_mrad)
            cx, cy = self.pos_mm[0] + px, self.pos_mm[1] + py
            if part.shape in ("sphere", "cylinder"):
                r = part.radius_mm
                a, b, c, d = cx - r, cy - r, cx + r, cy + r
            else:
                a, b, c, d = _rect_footprint(
                    cx, cy, part.size_mm[0], part.size_mm[1], self.yaw_mrad
                )
            x0, y0, x1, y1 = min(x0, a), min(y0, b), max(x1, c), max(y1, d)
        return x0, y0, x1, y1

    def z_range(self) -> tuple[float, float]:
        return float(self.pos_mm[2]), float(self.pos_mm[2] + self.height_mm)

    def as_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "body": self.body,
            "zh": self.zh,
            "cls": self.cls,
            "room": self.room,
            "pos_mm": list(self.pos_mm),
            "yaw_mrad": self.yaw_mrad,
            "mass_g": self.mass_g,
            "colour": self.colour,
            "opacity": self.opacity,
            "verbs": list(self.verbs),
            "registered": self.registered,
            "flags": list(self.flags),
            "friction": self.friction,
            "roll": self.roll,
            "hollow": self.hollow,
            "official": self.official,
            "parts": [p.as_json() for p in self.parts],
            **({"lib": self.lib, "passability": self.passability} if self.lib else {}),
            **(
                {"solref": list(self.solref), "solmix": self.solmix}
                if self.solref is not None
                else {}
            ),
        }


@dataclass(frozen=True)
class Drop:
    """A place the floor falls away: a pit (genkan, balcony step) or a stair hole."""

    id: str
    zh: str
    rect_mm: tuple[int, int, int, int]
    depth_mm: int
    buffer_mm: int = DROP_BUFFER_MM
    steps: int = 0  # 0: a pit; n: n equal steps down
    step_dir: str = "-y"  # the way the steps go down
    cut: bool = True  # True: this layout tiles the floor around it; False: official geometry
    gated: bool = False  # this variant must have barriers all round it
    official: tuple[str, ...] = ()

    def as_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "zh": self.zh,
            "rect_mm": list(self.rect_mm),
            "depth_mm": self.depth_mm,
            "buffer_mm": self.buffer_mm,
            "steps": self.steps,
            "step_dir": self.step_dir,
            "cut": self.cut,
            "gated": self.gated,
            "official": list(self.official),
        }


@dataclass(frozen=True)
class Zone:
    id: str
    kind: str  # forbid | drop | trip | nest | keepsake | play
    rect_mm: tuple[int, int, int, int]
    zh: str = ""

    def as_json(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "rect_mm": list(self.rect_mm), "zh": self.zh}


@dataclass(frozen=True)
class Barrier:
    """A stair railing or a baby gate, from one floor point to another (expanded into a
    bottom rail, a top rail and balusters no more than BARRIER_GAP_MM apart)."""

    id: str
    kind: str  # railing | gate
    from_mm: tuple[int, int]
    to_mm: tuple[int, int]
    height_mm: int
    zh: str = ""
    colour: str = "#f2efe8"
    drop: str = ""

    def as_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "zh": self.zh,
            "from_mm": list(self.from_mm),
            "to_mm": list(self.to_mm),
            "height_mm": self.height_mm,
            "colour": self.colour,
            "drop": self.drop,
        }


@dataclass(frozen=True)
class Lamp:
    id: str
    zh: str
    room: str
    pos_mm: tuple[int, int, int]  # the light source
    kind: str = "ceiling"  # ceiling | pendant | floor_lamp | table
    hours: tuple[int, int] = (17, 23)  # on from, off at (local hours)
    colour: str = "#ffd3a0"
    lux: int = 120  # on the floor in the middle of its room, fully on
    element: str = ""  # the geometry it belongs to (the official floor lamp's shade)

    def as_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "zh": self.zh,
            "room": self.room,
            "pos_mm": list(self.pos_mm),
            "kind": self.kind,
            "hours": list(self.hours),
            "colour": self.colour,
            "lux": self.lux,
            "element": self.element,
        }


@dataclass(frozen=True)
class Window:
    id: str
    zh: str
    room: str
    pos_mm: tuple[int, int, int]  # centre of the sill on the wall's inner face
    width_mm: int
    height_mm: int
    inward_mrad: int  # the way daylight travels into the room
    curtains: str | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "zh": self.zh,
            "room": self.room,
            "pos_mm": list(self.pos_mm),
            "width_mm": self.width_mm,
            "height_mm": self.height_mm,
            "inward_mrad": self.inward_mrad,
            "curtains": self.curtains,
        }


@dataclass(frozen=True)
class NightLight:
    id: str
    zh: str
    room: str
    pos_mm: tuple[int, int, int]
    hours: tuple[int, int] = (23, 7)
    colour: str = "#ffe2b0"

    def as_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "zh": self.zh,
            "room": self.room,
            "pos_mm": list(self.pos_mm),
            "hours": list(self.hours),
            "colour": self.colour,
        }


@dataclass(frozen=True)
class RoomLights:
    room: str
    lamps: tuple[Lamp, ...] = ()
    windows: tuple[Window, ...] = ()
    night_lights: tuple[NightLight, ...] = ()

    @property
    def count(self) -> int:
        return len(self.lamps) + len(self.windows) + len(self.night_lights)


@dataclass(frozen=True)
class Lights:
    """The home's light sources, for :mod:`duckmind.world.lights` (which decides when and how
    bright; this only says where). F5: a renderer lights the camera's room from
    :meth:`for_room` and treats the other rooms as ambient."""

    north_mrad: int
    lamps: tuple[Lamp, ...] = ()
    windows: tuple[Window, ...] = ()
    night_lights: tuple[NightLight, ...] = ()

    def for_room(self, room: str | None) -> RoomLights:
        return RoomLights(
            room=room or "",
            lamps=tuple(x for x in self.lamps if x.room == room),
            windows=tuple(x for x in self.windows if x.room == room),
            night_lights=tuple(x for x in self.night_lights if x.room == room),
        )

    @property
    def by_room(self) -> dict[str, RoomLights]:
        rooms = [x.room for x in self.lamps] + [x.room for x in self.windows]
        rooms += [x.room for x in self.night_lights]
        return {room: self.for_room(room) for room in dict.fromkeys(rooms)}

    def as_json(self) -> dict[str, Any]:
        return {
            "north_mrad": self.north_mrad,
            "lamps": [x.as_json() for x in self.lamps],
            "windows": [x.as_json() for x in self.windows],
            "night_lights": [x.as_json() for x in self.night_lights],
        }


@dataclass(frozen=True)
class Problem:
    """One generation-check failure (curiosity spec §4.1, physical-design spec §12.1)."""

    code: str
    subject: str
    detail: str

    def __str__(self) -> str:
        return f"{self.code} [{self.subject}]: {self.detail}"


# ── the layout ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Layout:
    """A resolved home: every variant applied, every macro (floor tiles, barriers, nest parts,
    window panels, wall bands, ceiling) expanded into elements."""

    id: str
    zh: str
    variant: str
    source: Mapping[str, Any]  # the resolved JSON before expansion (to_json(); round-trips)
    rooms: tuple[Room, ...]
    elements: tuple[Element, ...]
    things: tuple[Thing, ...]
    drops: tuple[Drop, ...]
    zones: tuple[Zone, ...]
    barriers: tuple[Barrier, ...]
    doorways: tuple[Mapping[str, Any], ...]
    nest: Nest | None
    mat: Mat
    spawn: Pose
    lights: Lights
    agents: Mapping[str, Any]
    ceiling_mm: int
    bounds_mm: tuple[int, int, int, int]
    north_mrad: int
    base: Mapping[str, Any]  # official scene files and their pinned sha256
    hidden: tuple[str, ...]  # official geoms and bodies taken out of the home (in memory)
    variants: tuple[str, ...]  # the variants this home offers
    mjcf_dir: str | None = None
    warnings: tuple[str, ...] = ()
    origin: Pose | None = None  # the layout's "frame" when it names one ("room", a pose)

    # ── frames ──

    @property
    def frame(self) -> Pose:
        """Where Duck Core's own origin lies in the world: what the layout's ``frame`` names
        ("room": the world's origin, so Duck Core's imagined walls are the home's walls — the
        default home), else its nest (a dock: the seat, facing into the room), else its mat."""
        if self.origin is not None:
            return self.origin
        return self.nest.home if self.nest is not None else self.mat.pose

    def plan_to_world(self, x_mm: float, y_mm: float, heading_mrad: float = 0) -> Pose:
        """A pose in Duck Core's frame (its plan), in the world."""
        wx, wy = self.frame.to_world(x_mm, y_mm)
        return Pose(
            int(round(wx)), int(round(wy)), _wrap_mrad(heading_mrad + self.frame.heading_mrad)
        )

    def world_to_plan(self, x_mm: float, y_mm: float, heading_mrad: float = 0) -> Pose:
        lx, ly = self.frame.to_local(x_mm, y_mm)
        return Pose(
            int(round(lx)), int(round(ly)), _wrap_mrad(heading_mrad - self.frame.heading_mrad)
        )

    # ── lookups ──

    def room_at(self, x_mm: float, y_mm: float) -> str | None:
        """The room a floor point is in (the camera's room, for the light set)."""
        for room in self.rooms:
            if room.contains(x_mm, y_mm):
                return room.id
        return None

    def element(self, element_id: str) -> Element:
        for item in self.elements:
            if item.id == element_id:
                return item
        raise KeyError(element_id)

    def thing(self, thing_id: str) -> Thing:
        for item in self.things:
            if item.id == thing_id:
                return item
        raise KeyError(thing_id)

    def support_top(self, x_mm: float, y_mm: float) -> int | None:
        """The height of the floor (or step, pad, platform) at a point; None over a hole."""
        tops = [e.top_mm for e in self.elements if e.support and _supports(e, x_mm, y_mm)]
        return max(tops) if tops else None

    # ── where a duck may be put down (the world's placements and the hand) ──

    def barrier_between(
        self, from_mm: tuple[float, float], to_mm: tuple[float, float]
    ) -> Element | None:
        """The first wall, railing or gate the straight way from ``from_mm`` to ``to_mm``
        passes through (goes into and comes out of between the two points), or None. A way
        that starts inside a wall (an asked spot in the wall itself) does not count as crossing
        it: whether the other end is still indoors is :meth:`place_problem`'s room test."""
        for element in self.elements:
            if element.kind in BARRIER_KINDS and element.collide:
                if _passes_through(element, from_mm, to_mm):
                    return element
        return None

    def over_drop(self, x_mm: float, y_mm: float, grow_mm: float = 0.0) -> Drop | None:
        """The drop (a pit, a stair hole) under a floor point, grown by ``grow_mm``, or None."""
        for drop in self.drops:
            x0, y0, x1, y1 = drop.rect_mm
            if x0 - grow_mm <= x_mm <= x1 + grow_mm and y0 - grow_mm <= y_mm <= y1 + grow_mm:
                return drop
        return None

    def place_problem(
        self,
        x_mm: float,
        y_mm: float,
        *,
        from_mm: tuple[float, float] | None = None,
        drop_margin_mm: float = 0.0,
    ) -> str | None:
        """Why a duck may not be put down at (x, y), in plain words, or None when it may: the
        spot must be in one of the home's rooms (never past a wall: the official floor plane
        runs on outside the house), not over a drop (kept ``drop_margin_mm`` off it) and — when
        it stands in for another spot (``from_mm``: the spot asked for, taken by furniture) —
        reachable from there without going through a wall, a railing or a gate. Whether the
        duck fits there (furniture, toys) is the physics' question, not this one."""
        bx0, by0, bx1, by1 = self.bounds_mm
        if self.room_at(x_mm, y_mm) is None or not (bx0 <= x_mm <= bx1 and by0 <= y_mm <= by1):
            return msg("world.homelayout.Layout.place_problem.84ea88d9")
        drop = self.over_drop(x_mm, y_mm, drop_margin_mm)
        if drop is not None:
            return msg("world.homelayout.Layout.place_problem.1f73af95", p=drop.zh or drop.id)
        if from_mm is not None:
            wall = self.barrier_between(from_mm, (x_mm, y_mm))
            if wall is not None:
                return msg(
                    "world.homelayout.Layout.place_problem.61adfad7",
                    p=self.zh_of(wall.geom)
                    or msg("world.homelayout.Layout.place_problem.8c8f5aa3"),
                )
        return None

    def zh_of(self, name: str) -> str | None:
        """What a MuJoCo geom or body of this home is, in plain words (for the page's notes;
        None for a name that is not the home's)."""
        base = str(name).split("/", 1)[0]
        for element in self.elements:
            if element.geom == name or element.geom == base:
                return (
                    element.zh
                    or KIND_ZH.get(element.kind)
                    or msg("world.homelayout.Layout.zh_of.112b2587")
                )
        for thing in self.things:
            body = thing.body
            if base == body or base.startswith(body + "_") or name.startswith(body + "_"):
                return thing.zh or msg("world.homelayout.Layout.zh_of.d0ec347a")
        if base.startswith("person_"):
            return msg("world.homelayout.Layout.zh_of.bf881ac8")
        if base.startswith("kit_hand"):
            return msg("world.homelayout.Layout.zh_of.a3d17df6")
        if base.startswith("home_nest") or base.startswith("nest_"):
            return (
                msg("world.homelayout.Layout.zh_of.923c19dc")
                if self.nest is not None and self.nest.is_dock
                else msg("world.homelayout.Layout.zh_of.a18153d0")
            )
        if base.startswith("duck_") and "probe" in base:
            return msg("world.homelayout.Layout.zh_of.16fd0b67")
        return None

    @property
    def fingerprint(self) -> str:
        """sha256 of the resolved layout and the official files it was built on."""
        blob = json.dumps(
            {"layout": self.source, "variant": self.variant, "base": self.base},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    # ── outputs ──

    def to_json(self) -> dict[str, Any]:
        """The resolved layout (variants applied, before expansion): ``load(to_json())`` gives
        the same layout."""
        return copy.deepcopy(dict(self.source))

    def page_json(self) -> dict[str, Any]:
        """The simulator page's room (``home-page-v1``): everything in world coordinates,
        millimetres and milliradians. three.js (x, y, z) = world (x, z, −y)."""
        legend: dict[str, dict[str, Any]] = {}
        for thing in self.things:
            entry = legend.setdefault(
                thing.cls,
                {
                    "cls": thing.cls,
                    "zh": CLASS_ZH.get(thing.cls, thing.cls),
                    "verbs": [],
                    "verbs_zh": [],
                    "count": 0,
                },
            )
            entry["count"] += 1
            for verb in thing.verbs:
                if verb not in entry["verbs"]:
                    entry["verbs"].append(verb)
                    entry["verbs_zh"].append(VERB_ZH.get(verb, verb))
        return {
            "schema": PAGE_SCHEMA,
            "id": self.id,
            "variant": self.variant,
            "zh": self.zh,
            "fingerprint": self.fingerprint,
            "units": "mm, mrad; world frame: x, y on the floor, z up; three.js (x, z, -y)",
            "bounds_mm": list(self.bounds_mm),
            "ceiling_mm": self.ceiling_mm,
            "north_mrad": self.north_mrad,
            "rooms": [r.as_json() for r in self.rooms],
            "doorways": [dict(d) for d in self.doorways],
            "nest": self.nest.as_json() if self.nest is not None else None,
            "mat": self.mat.as_json(),
            "spawn": self.spawn.as_json(),
            "frame": self.frame.as_json(),
            "elements": [e.as_json() for e in self.elements],
            "things": [t.as_json() for t in self.things],
            "drops": [d.as_json() for d in self.drops],
            "zones": [z.as_json() for z in self.zones],
            "barriers": [b.as_json() for b in self.barriers],
            "lights": self.lights.as_json(),
            "agents": copy.deepcopy(dict(self.agents)),
            "legend": list(legend.values()),
            "warnings": list(self.warnings),
        }

    def check(self) -> list[Problem]:
        """The generation checks; an empty list means the home may be built."""
        return _check(self)

    @property
    def base_scene(self) -> str:
        """The official scene file this home starts from (allcollisions): see
        :func:`base_scene`."""
        return base_scene(self)


def spawn_pose(layout: Layout, where: str) -> Pose:
    """Where the world server puts the duck for ``--spawn nest`` / ``mat``: seated in its dock
    (or on the open nest's pad) the way it docks, or on the mat. KeyError without one."""
    if where == "nest":
        if layout.nest is None:
            raise KeyError("nest")
        return layout.nest.seat
    if where == "mat":
        return layout.mat.pose
    raise KeyError(where)


def base_scene(layout: Layout, collision: str = "allcollisions") -> str:
    """The official scene file (a name in the MJCF folder) a home is composed on: its own
    official scene for that collision model (the apartment: ``scene_apartment.xml``), else the
    official floor scene of that model (``scene_allcollisions.xml`` / ``scene.xml``)."""
    if collision not in SCENES:
        raise LayoutError(f"碰撞模型 {collision!r} 不認得（{', '.join(SCENES)}）")
    scenes = dict(layout.base.get("scenes") or {})
    if layout.base.get("world") and collision in scenes:
        return str(scenes[collision])
    return SCENES[collision]


def official_files(layout: Layout, collision: str = "allcollisions") -> list[str]:
    """Every official MJCF file the composed home reads (scene, the robot it includes, and the
    official world when it is attached rather than included), for pinning."""
    scene = base_scene(layout, collision)
    files = [scene, ROBOT_FILES[collision]]
    world = layout.base.get("world")
    if world and str(world) not in files:
        files.append(str(world))
    return files


# ── loading ─────────────────────────────────────────────────────────────────────

ALIASES = {"apartment_ungated_test": ("apartment", "ungated_test")}


def available() -> list[str]:
    """The homes in ``world/homes`` (plus the spec's alias ``apartment_ungated_test``)."""
    names = []
    for path in sorted(HOMES_DIR.glob("*.json")):
        names.append(path.name.removesuffix(".json").removesuffix(".overlay"))
    return names + sorted(ALIASES)


def variants(name: str) -> list[str]:
    """The variants a home offers (``load(name, variant=…)`` or ``"name,variant"``)."""
    path = _home_path(ALIASES.get(name, (name, ""))[0])
    return sorted(_read_json(path).get("variants", {}))


def official_dir(mjcf_dir: str | os.PathLike[str] | None = None) -> Path:
    """The official microduck MJCF folder: the argument, else ``$DUCKMIND_MICRODUCK_MJCF``,
    else next to an importable ``mjlab_microduck`` (as in the body host, whose path holds
    microduck_rl/src). Found, never imported."""
    folder = mjcf_dir if mjcf_dir is not None else os.environ.get(MJCF_ENV)
    if not folder:
        try:
            found = importlib.util.find_spec("mjlab_microduck")
        except (ImportError, ValueError):
            found = None
        if found is not None and found.submodule_search_locations:
            folder = str(Path(list(found.submodule_search_locations)[0]) / "robot" / "microduck")
    if not folder:
        raise LayoutError(
            "找不到 Pollen 的小鴨模型資料夾：請給 mjcf_dir，或設定 DUCKMIND_MICRODUCK_MJCF "
            "(microduck_rl/src/mjlab_microduck/robot/microduck)"
        )
    path = Path(folder).expanduser()
    if not (path / "robot_allcollisions.xml").is_file():
        raise LayoutError(f"{path} 裡沒有上游的 robot_allcollisions.xml，不是 Pollen 的小鴨模型資料夾")
    return path


def load(
    name_or_path: str | os.PathLike[str] | Mapping[str, Any],
    *,
    variant: str | Iterable[str] | None = None,
    mjcf_dir: str | os.PathLike[str] | None = None,
) -> Layout:
    """A home by name (``"default_home"``, ``"apartment"``, ``"apartment,ungated_test"``,
    ``"apartment_ungated_test"``), by path to a layout JSON, or from a layout dict (a
    :meth:`Layout.to_json`). ``variant`` adds variants (comma-separated or a list).

    The official apartment is read from the upstream checkout (``mjcf_dir`` or
    ``$DUCKMIND_MICRODUCK_MJCF``); the other homes need it only to be composed."""
    chosen: list[str] = []
    if isinstance(name_or_path, Mapping):
        raw = copy.deepcopy(dict(name_or_path))
    else:
        text = os.fspath(name_or_path)
        path = Path(text).expanduser()
        if not (text.endswith(".json") or path.is_file()):
            name, _, rest = text.partition(",")
            name, alias_variant = ALIASES.get(name, (name, ""))
            chosen += [v for v in [alias_variant, *rest.split(",")] if v]
            path = _home_path(name)
        raw = _read_json(path)
    if isinstance(variant, str):
        chosen += [v for v in variant.split(",") if v]
    elif variant is not None:
        chosen += list(variant)
    folder = os.fspath(mjcf_dir) if mjcf_dir is not None else os.environ.get(MJCF_ENV)
    return _resolve(raw, chosen, folder)


def _home_path(name: str) -> Path:
    for candidate in (HOMES_DIR / f"{name}.json", HOMES_DIR / f"{name}.overlay.json"):
        if candidate.is_file():
            return candidate
    raise LayoutError(f"沒有叫「{name}」的家。有：{', '.join(available())}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise LayoutError(f"找不到佈置檔 {path}") from None
    except json.JSONDecodeError as error:
        raise LayoutError(f"佈置檔 {path} 不是正確的 JSON：{error}") from None
    if not isinstance(data, dict):
        raise LayoutError(f"佈置檔 {path} 應該是一個 JSON 物件")
    return data


def _resolve(raw: dict[str, Any], chosen: list[str], mjcf_dir: str | None) -> Layout:
    if raw.get("schema") != SCHEMA:
        raise LayoutError(f"佈置的格式應該是 {SCHEMA}，這份是 {raw.get('schema')!r}")
    offered = dict(raw.pop("variants", {}) or {})
    applied = list(raw.pop("variant_applied", []) or [])
    warnings: list[str] = []
    base = dict(raw.get("base") or {})
    if raw.get("placements"):
        from duckmind.world import objlib  # the library's items put where the placements say

        try:
            objlib.expand(raw)
        except objlib.LibraryError as error:
            raise LayoutError(str(error)) from None
    if base.get("convert") == "apartment":
        from duckmind.world import apartment  # the official apartment, read where it lies

        folder = official_dir(mjcf_dir)
        official = apartment.convert(folder)
        warnings += apartment.check_pins(folder, base.get("sha256") or {})
        raw = _apply_overlay(official, raw)
        base.pop("convert")
        base["converted"] = "apartment"
        raw["base"] = base
    elif base.get("converted") == "apartment" and mjcf_dir:
        from duckmind.world import apartment

        warnings += apartment.check_pins(official_dir(mjcf_dir), base.get("sha256") or {})
    for name in chosen:
        if name in applied:
            continue
        if name not in offered:
            raise LayoutError(
                f"「{raw.get('id')}」沒有「{name}」這個變體。有：{', '.join(sorted(offered))}"
            )
        raw = _apply_variant(raw, offered[name])
        applied.append(name)
    raw["variant_applied"] = applied
    source = copy.deepcopy(raw)
    source["variants"] = offered
    layout = _build(raw, mjcf_dir, warnings)
    return _replace_source(layout, source, applied, tuple(sorted(offered)))


def _replace_source(
    layout: Layout, source: dict[str, Any], applied: list[str], offered: tuple[str, ...]
) -> Layout:
    return dataclasses.replace(layout, source=source, variant=",".join(applied), variants=offered)


# Overlay operations on official elements (hide, move, regroup, meta) — also idempotent on a
# resolved layout, so load(to_json()) is the same home.


def _apply_overlay(official: Mapping[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(overlay)
    out["elements"] = list(copy.deepcopy(official.get("elements", []))) + list(
        overlay.get("elements", [])
    )
    out["things"] = list(copy.deepcopy(official.get("things", []))) + list(
        overlay.get("things", [])
    )
    return out


def _official_ops(raw: dict[str, Any]) -> None:
    hidden = set(raw.get("hide", []))
    raw["elements"] = [
        e
        for e in raw.get("elements", [])
        if not (e.get("official") and (e["id"] in hidden or e.get("body") in hidden))
    ]
    raw["things"] = [
        t for t in raw.get("things", []) if not (t.get("official") and t["id"] in hidden)
    ]
    for thing in raw["things"]:
        moved = raw.get("move", {}).get(thing["id"])
        if moved is not None and thing.get("official"):
            thing["pos_mm"] = [int(moved[0]), int(moved[1]), thing["pos_mm"][2]]
        meta = raw.get("official_things", {}).get(thing["id"])
        if meta and thing.get("official"):
            thing.update({k: v for k, v in meta.items() if k not in ("id", "pos_mm", "parts")})
    for element in raw["elements"]:
        flags = raw.get("regroup", {}).get(element["id"])
        if flags and element.get("official"):
            element.update({k: bool(v) for k, v in flags.items() if k in ("camera", "tof")})


def _apply_variant(raw: dict[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(raw)
    lists = ("elements", "things", "drops", "zones", "barriers")
    for key, value in patch.items():
        if key in ("zh", "doc"):
            continue
        if key == "add":
            for list_key, items in value.items():
                if list_key in ("lamps", "windows", "night_lights"):
                    lights = out.setdefault("lights", {})
                    lights[list_key] = list(lights.get(list_key, [])) + copy.deepcopy(items)
                else:
                    out[list_key] = list(out.get(list_key, [])) + copy.deepcopy(items)
        elif key == "remove":
            gone = set(value)
            for list_key in lists:
                out[list_key] = [x for x in out.get(list_key, []) if x.get("id") not in gone]
            lights = out.get("lights", {})
            for list_key in ("lamps", "windows", "night_lights"):
                lights[list_key] = [x for x in lights.get(list_key, []) if x["id"] not in gone]
        elif key == "patch":
            for item_id, fields in value.items():
                found = False
                for list_key in lists:
                    for item in out.get(list_key, []):
                        if item.get("id") == item_id:
                            item.update(copy.deepcopy(fields))
                            found = True
                if not found:
                    raise LayoutError(f"變體要改的「{item_id}」不在這個家裡")
        elif isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            merged = dict(out[key])
            merged.update(copy.deepcopy(dict(value)))
            out[key] = merged
        else:
            out[key] = copy.deepcopy(value)
    return out


# ── building (validation and expansion) ───────────────────────────────────────


def _int(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LayoutError(f"{what} 應該是整數（mm、mrad、g），這裡是 {value!r}")
    return value


def _ints(value: Any, n: int, what: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise LayoutError(f"{what} 應該是 {n} 個整數，這裡是 {value!r}")
    return tuple(_int(v, what) for v in value)


def _int2(value: Any, what: str) -> tuple[int, int]:
    a, b = _ints(value, 2, what)
    return a, b


def _int3(value: Any, what: str) -> tuple[int, int, int]:
    a, b, c = _ints(value, 3, what)
    return a, b, c


def _colour(value: Any, what: str) -> str:
    text = str(value)
    if len(text) != 7 or not text.startswith("#"):
        raise LayoutError(f"{what} 的顏色應該寫成 #rrggbb，這裡是 {value!r}")
    int(text[1:], 16)
    return text.lower()


def _rect(value: Any, what: str) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = _ints(value, 4, what)
    if x1 <= x0 or y1 <= y0:
        raise LayoutError(
            f"{what} 的範圍 {value!r} 不對（應該是 x0, y0, x1, y1，且 x1 > x0、y1 > y0）"
        )
    return x0, y0, x1, y1


def _element(raw: Mapping[str, Any]) -> Element:
    what = f"「{raw.get('id')}」"
    element_id = str(raw["id"])
    shape = str(raw.get("shape", "box"))
    if shape not in SHAPES:
        raise LayoutError(f"{what} 的形狀 {shape!r} 不認得（可以用 {', '.join(SHAPES)}）")
    radius = _int(raw.get("radius_mm", 0), what)
    radius_bottom = _int(raw.get("radius_bottom_mm", 0), what)
    fromto: tuple[int, ...] = ()
    pos: tuple[int, ...]
    size: tuple[int, ...]
    if shape == "capsule":
        fromto = _ints(raw["fromto_mm"], 6, what)
        ax, ay, az, bx, by, bz = fromto
        pos = ((ax + bx) // 2, (ay + by) // 2, min(az, bz) - radius)
        size = (abs(bx - ax) + 2 * radius, abs(by - ay) + 2 * radius, abs(bz - az) + 2 * radius)
    else:
        pos3 = raw.get("pos_mm")
        pos = _ints(pos3, 3, what) if pos3 is not None else (0, 0, 0)
        if shape in ("cylinder", "frustum"):
            height = _int(raw.get("height_mm", raw.get("size_mm", [0, 0, 0])[2]), what)
            r = max(radius, radius_bottom)
            size = (2 * r, 2 * r, height)
        elif shape == "sphere":
            size = (2 * radius, 2 * radius, 2 * radius)
        elif shape == "plane":
            sx, sy = _ints(raw["size_mm"][:2], 2, what)
            size = (sx, sy, 0)
        else:
            size = _ints(raw["size_mm"], 3, what)
    ramp = _int(raw.get("ramp_mm", 0), what)
    if shape == "slab" and not 0 < 2 * ramp < min(size[0], size[1]):
        raise LayoutError(f"{what} 的斜坡 ramp_mm 要大於 0，而且比底的一半短")
    friction = raw.get("friction")
    rubbing = (float(friction[0]), float(friction[1]), float(friction[2])) if friction else None
    return Element(
        id=element_id,
        kind=str(raw.get("kind", "furniture")),
        shape=shape,
        pos_mm=(int(pos[0]), int(pos[1]), int(pos[2])),
        size_mm=(int(size[0]), int(size[1]), int(size[2])),
        zh=str(raw.get("zh", "")),
        cls=str(raw.get("cls", "")),
        yaw_mrad=_int(raw.get("yaw_mrad", 0), what),
        radius_mm=radius,
        radius_bottom_mm=radius_bottom,
        fromto_mm=fromto,
        colour=_colour(raw.get("colour", "#cccccc"), what),
        opacity=float(raw.get("opacity", 1.0)),
        texture=str(raw.get("texture", "")),
        collide=bool(raw.get("collide", True)),
        camera=bool(raw.get("camera", True)),
        tof=bool(raw.get("tof", True)),
        page=bool(raw.get("page", True)),
        room=raw.get("room"),
        official=bool(raw.get("official", False)),
        under_mm=_int(raw.get("under_mm", 0), what),
        friction=rubbing,
        ramp_mm=ramp if shape == "slab" else 0,
        lib=str(raw.get("lib", "")),
        place=str(raw.get("place", "")),
        passability=str(raw.get("passability", "")),
        solref=_solref(raw.get("solref")),
        solmix=float(raw["solmix"]) if raw.get("solmix") is not None else None,
    )


def _part(raw: Mapping[str, Any], what: str) -> Part:
    shape = str(raw.get("shape", "box"))
    radius = _int(raw.get("radius_mm", 0), what)
    size: tuple[int, ...]
    if shape == "sphere":
        size = (2 * radius,) * 3
    elif shape == "cylinder":
        size = (2 * radius, 2 * radius, _int(raw["height_mm"], what))
    elif shape == "box":
        size = _ints(raw["size_mm"], 3, what)
    else:
        raise LayoutError(f"{what} 的零件形狀 {shape!r} 不認得（sphere、cylinder、box）")
    offset = _int3(raw.get("offset_mm", [0, 0, 0]), what)
    return Part(
        shape=shape,
        size_mm=(size[0], size[1], size[2]),
        offset_mm=offset,
        radius_mm=radius,
        mass_g=_int(raw.get("mass_g", 0), what),
    )


def _thing(raw: Mapping[str, Any], layout_elements: list[Element]) -> Thing:
    what = f"東西「{raw.get('id')}」"
    if "parts" in raw:
        parts = tuple(_part(p, what) for p in raw["parts"])
    else:
        parts = (_part({**raw, "offset_mm": [0, 0, 0]}, what),)
    mass = _int(raw.get("mass_g", sum(p.mass_g for p in parts)), what)
    if len(parts) == 1 and parts[0].mass_g == 0:
        parts = (dataclasses.replace(parts[0], mass_g=mass),)
    if sum(p.mass_g for p in parts) != mass:
        raise LayoutError(f"{what} 的零件重量加起來不是 {mass} g")
    pos = raw.get("pos_mm")
    if not isinstance(pos, (list, tuple)) or len(pos) not in (2, 3):
        raise LayoutError(f"{what} 的位置 pos_mm 應該是 [x, y] 或 [x, y, z]")
    x, y = _int(pos[0], what), _int(pos[1], what)
    if len(pos) == 3:
        z = _int(pos[2], what)
    else:  # it rests on whatever floor is there
        tops = [e.top_mm for e in layout_elements if e.support and _supports(e, x, y)]
        z = max(tops) if tops else 0
    return Thing(
        id=str(raw["id"]),
        zh=str(raw.get("zh", "")),
        cls=str(raw.get("cls", "unknown")),
        pos_mm=(x, y, z),
        parts=parts,
        mass_g=mass,
        yaw_mrad=_int(raw.get("yaw_mrad", 0), what),
        colour=_colour(raw.get("colour", "#cccccc"), what),
        opacity=float(raw.get("opacity", 1.0)),
        verbs=tuple(str(v) for v in raw.get("verbs", [])),
        registered=bool(raw.get("registered", True)),
        flags=tuple(str(v) for v in raw.get("flags", [])),
        friction=float(raw.get("friction", 0.8)),
        roll=float(raw.get("roll", 0.0002)),
        hollow=bool(raw.get("hollow", False)),
        official=bool(raw.get("official", False)),
        lib=str(raw.get("lib", "")),
        passability=str(raw.get("passability", "")),
        solref=_solref(raw.get("solref")),
        solmix=float(raw["solmix"]) if raw.get("solmix") is not None else None,
    )


def _solref(value: Any) -> tuple[float, float] | None:
    """MuJoCo's contact softness (time constant s, damping ratio) when a piece gives under weight."""
    if not value:
        return None
    return float(value[0]), float(value[1])


def _nest(raw: Mapping[str, Any] | None) -> Nest | None:
    if not raw:
        return None
    kind = str(raw.get("kind", "open"))
    if kind not in NEST_KINDS:
        raise LayoutError(f"充電的地方 {kind!r} 不認得（dock 充電座、open 開放式的窩）")
    what = "充電座" if kind == "dock" else "窩"
    if "x_mm" not in raw or "y_mm" not in raw:
        raise LayoutError(f"{what}的位置沒有給（x_mm、y_mm）")
    style = ""
    side = ""
    shape_mm: dict[str, int] = {}
    if kind == "dock":
        from duckmind.world import dock

        style = str(raw.get("style", dock.DEFAULT_STYLE))
        side = str(raw.get("side") or dock.DEFAULT_SIDE)
        if side not in dock.SIDES:
            raise LayoutError(f"充電座的轉身邊 {side!r} 不認得（right 右邊、left 左邊）")
        shape_mm = {str(k): v for k, v in dict(raw.get("shape", {}) or {}).items()}
        try:
            dock.shape(style, shape_mm)
        except ValueError as error:
            raise LayoutError(str(error)) from None
    elif raw.get("style"):
        raise LayoutError("停靠方式（forward、backin）只給充電座用，開放式的窩沒有")
    elif raw.get("side"):
        raise LayoutError("轉身邊（right、left）只給充電座用，開放式的窩沒有")
    edge = str(raw.get("edge", "edge_5mm"))
    if edge not in NEST_EDGES:
        raise LayoutError(f"窩的邊 {edge!r} 不認得（{', '.join(NEST_EDGES)}）")
    board = _int3(raw.get("board_size_mm", [20, 300, 220]), what)
    return Nest(
        x_mm=_int(raw["x_mm"], what),
        y_mm=_int(raw["y_mm"], what),
        heading_mrad=_int(raw.get("heading_mrad", 0), what),
        z_mm=0,
        pad_radius_mm=_int(raw.get("pad_radius_mm", 200), what),
        edge=edge,
        ramp_radius_mm=_int(raw.get("ramp_radius_mm", 250), what),
        contacts_radius_mm=_int(raw.get("contacts_radius_mm", 85), what),
        board_offset_mm=_int(raw.get("board_offset_mm", NEST_BOARD_OFFSET_MM), what),
        board_size_mm=board,
        board_colour=_colour(raw.get("board_colour", "#12b5c4"), what),
        pad_colour=_colour(raw.get("pad_colour", "#e8d9b5"), what),
        tag_mm=_int(raw.get("tag_mm", 100), what),
        edge_tag_mm=_int(raw.get("edge_tag_mm", 40), what),
        backlight=dict(raw.get("backlight", {})),
        kind=kind,
        style=style,
        shape_mm=shape_mm,
        side=side,
    )


def _nest_elements(nest: Nest) -> list[dict[str, Any]]:
    """The passive charging place. A dock: housing, cyan face panel, floor plate, the V guides,
    the two tag panels and the status light (:func:`duckmind.world.dock.elements`). The open
    nest: pad (with its edge variant), cyan backboard, the two tag panels and the charging
    light. The nest owner (:mod:`duckmind.world.nestpad`) adds the contacts, the duck's probe,
    the tag textures and the light's glow to these geoms by name (``home_nest_*``)."""
    if nest.is_dock:
        from duckmind.world import dock

        return dock.elements(nest, nest.dock_shape, nest.style)

    def at(lx: float, ly: float) -> tuple[int, int]:
        wx, wy = nest.to_world(lx, ly)
        return int(round(wx)), int(round(wy))

    h = nest.pad_height_mm
    px, py = at(0, 0)
    pad: dict[str, Any] = {
        "id": "nest_pad",
        "zh": msg("world.homelayout._nest_elements.1c67edda"),
        "kind": "nest_pad",
        "cls": "furniture",
        "pos_mm": [px, py, 0],
        "height_mm": h,
        "colour": nest.pad_colour,
    }
    if nest.edge == "ramp":
        pad.update(
            shape="frustum", radius_mm=nest.pad_radius_mm, radius_bottom_mm=nest.ramp_radius_mm
        )
    else:
        pad.update(shape="cylinder", radius_mm=nest.pad_radius_mm)
    bx, by = at(-nest.board_offset_mm, 0)
    sx, sy, sz = nest.board_size_mm
    front = -nest.board_offset_mm + sx // 2  # the board's face toward the pad
    tx, ty = at(front + 1, 0)
    ex, ey = at(nest.pad_radius_mm - nest.edge_tag_mm // 2 - 10, 0)
    lx, ly = at(front + 2, -110)
    return [
        pad,
        {
            "id": "nest_board",
            "zh": msg("world.homelayout._nest_elements.afac08d1"),
            "kind": "nest_board",
            "cls": "furniture",
            "shape": "box",
            "pos_mm": [bx, by, 0],
            "size_mm": [sx, sy, sz],
            "yaw_mrad": nest.heading_mrad,
            "colour": nest.board_colour,
        },
        {
            "id": "nest_tag",
            "zh": msg("world.homelayout._nest_elements.9d0969bf"),
            "kind": "nest_tag",
            "shape": "box",
            "pos_mm": [tx, ty, 120 - nest.tag_mm // 2],
            "size_mm": [2, nest.tag_mm, nest.tag_mm],
            "yaw_mrad": nest.heading_mrad,
            "colour": "#101018",
            "collide": False,
            "tof": False,
        },
        {
            "id": "nest_edge_tag",
            "zh": msg("world.homelayout._nest_elements.327ce1f0"),
            "kind": "nest_tag",
            "shape": "box",
            "pos_mm": [ex, ey, h],
            "size_mm": [nest.edge_tag_mm, nest.edge_tag_mm, 1],
            "yaw_mrad": nest.heading_mrad,
            "colour": "#101018",
            "collide": False,
            "tof": False,
        },
        {
            "id": "nest_led",
            "zh": msg("world.homelayout._nest_elements.42509863"),
            "kind": "nest_led",
            "shape": "sphere",
            "pos_mm": [lx, ly, 190],
            "radius_mm": 10,
            "colour": "#3a3a44",
            "collide": False,
            "tof": False,
        },
    ]


def _window_element(window: Mapping[str, Any]) -> dict[str, Any]:
    what = f"窗「{window.get('id')}」"
    x, y, sill = _ints(window["pos_mm"], 3, what)
    inward = _int(window["inward_mrad"], what)
    width, height = _int(window["width_mm"], what), _int(window["height_mm"], what)
    ux, uy = round(math.cos(inward / 1000.0)), round(math.sin(inward / 1000.0))
    if abs(ux) + abs(uy) != 1 or abs(_wrap_mrad(inward) % 1571) > 2:
        raise LayoutError(f"{what} 要貼在與 x 或 y 平行的牆上（inward_mrad 是 0、±1571 或 3142）")
    size = [10, width, height] if ux else [width, 10, height]
    return {
        "id": str(window["id"]),
        "zh": str(window.get("zh", msg("world.homelayout._window_element.72e7bfea"))),
        "kind": "window",
        "shape": "box",
        "pos_mm": [x + 5 * ux, y + 5 * uy, sill],
        "size_mm": size,
        "colour": str(window.get("colour", "#dbe9f2")),
        "collide": False,
        "tof": False,
    }


def _barrier_elements(b: Barrier) -> list[dict[str, Any]]:
    (ax, ay), (bx, by) = b.from_mm, b.to_mm
    if ax != bx and ay != by:
        raise LayoutError(f"欄杆「{b.id}」要和 x 或 y 平行")
    along_x = ay == by
    length = abs(bx - ax) if along_x else abs(by - ay)
    cx, cy = (ax + bx) // 2, (ay + by) // 2
    thick = 20
    common = {"kind": b.kind, "cls": "furniture", "shape": "box", "colour": b.colour}

    def size(long: int, wide: int, tall: int) -> list[int]:
        return [long, wide, tall] if along_x else [wide, long, tall]

    out: list[dict[str, Any]] = [
        {
            **common,
            "id": f"{b.id}_rail_low",
            "zh": b.zh,
            "pos_mm": [cx, cy, 0],
            "size_mm": size(length + thick, thick, 40),
        },
        {
            **common,
            "id": f"{b.id}_rail_top",
            "zh": b.zh,
            "pos_mm": [cx, cy, b.height_mm - 40],
            "size_mm": size(length + thick, 30, 40),
        },
    ]
    pitch = BARRIER_GAP_MM + thick
    count = max(2, math.ceil(length / pitch) + 1)
    for k in range(count):
        t = k / (count - 1)
        px = int(round(ax + (bx - ax) * t))
        py = int(round(ay + (by - ay) * t))
        out.append(
            {
                **common,
                "id": f"{b.id}_post_{k}",
                "zh": b.zh,
                "pos_mm": [px, py, 40],
                "size_mm": [thick, thick, b.height_mm - 80],
            }
        )
    return out


def _tiles(
    bounds: tuple[int, int, int, int], holes: list[tuple[int, int, int, int]]
) -> list[tuple[int, int, int, int]]:
    """Rectangles covering ``bounds`` except the holes (grid cells merged into strips)."""
    x0, y0, x1, y1 = bounds
    xs = sorted({x0, x1, *[v for h in holes for v in (h[0], h[2]) if x0 < v < x1]})
    ys = sorted({y0, y1, *[v for h in holes for v in (h[1], h[3]) if y0 < v < y1]})

    def solid(a: int, b: int, c: int, d: int) -> bool:
        mx, my = (a + c) / 2, (b + d) / 2
        return not any(h[0] <= mx <= h[2] and h[1] <= my <= h[3] for h in holes)

    rows: list[tuple[int, int, int, int]] = []
    for j in range(len(ys) - 1):
        run: list[int] | None = None
        for i in range(len(xs) - 1):
            if solid(xs[i], ys[j], xs[i + 1], ys[j + 1]):
                if run is None:
                    run = [xs[i], xs[i + 1]]
                else:
                    run[1] = xs[i + 1]
            elif run is not None:
                rows.append((run[0], ys[j], run[1], ys[j + 1]))
                run = None
        if run is not None:
            rows.append((run[0], ys[j], run[1], ys[j + 1]))
    merged: list[list[int]] = []
    for a, b, c, d in rows:
        for m in merged:
            if m[0] == a and m[2] == c and m[3] == b:
                m[3] = d
                break
        else:
            merged.append([a, b, c, d])
    return [(m[0], m[1], m[2], m[3]) for m in merged]


def _drop_elements(raw: Mapping[str, Any], drops: list[Drop]) -> list[dict[str, Any]]:
    """Floor tiles around every cut drop, pit bottoms and stair steps."""
    cut = [d for d in drops if d.cut]
    if not cut:
        return []
    floor = raw.get("floor", {})
    colour = str(floor.get("colour", "#c9a47a"))
    texture = str(floor.get("texture", ""))
    depth = max(d.depth_mm for d in cut) + 50
    bounds = _rect(raw["bounds_mm"], "bounds_mm")
    out: list[dict[str, Any]] = []
    for k, (a, b, c, d) in enumerate(_tiles(bounds, [x.rect_mm for x in cut])):
        out.append(
            {
                "id": f"floor_tile_{k}",
                "zh": str(floor.get("zh", msg("world.homelayout._drop_elements.d8fb4039"))),
                "kind": "floor",
                "shape": "box",
                "pos_mm": [(a + c) // 2, (b + d) // 2, -depth],
                "size_mm": [c - a, d - b, depth],
                "colour": colour,
                "texture": texture,
                "friction": list(OFFICIAL_FLOOR_FRICTION),
            }
        )
    for drop in cut:
        a, b, c, d = drop.rect_mm
        if drop.steps <= 0:
            out.append(
                {
                    "id": f"{drop.id}_bottom",
                    "zh": drop.zh,
                    "kind": "pit_bottom",
                    "shape": "box",
                    "pos_mm": [(a + c) // 2, (b + d) // 2, -depth],
                    "size_mm": [c - a, d - b, depth - drop.depth_mm],
                    "colour": colour,
                    "friction": list(OFFICIAL_FLOOR_FRICTION),
                }
            )
            continue
        along_y = drop.step_dir in ("-y", "+y")
        span = (d - b) if along_y else (c - a)
        for i in range(drop.steps):
            near, far = span * i // drop.steps, span * (i + 1) // drop.steps
            if drop.step_dir == "-y":
                r = (a, d - far, c, d - near)
            elif drop.step_dir == "+y":
                r = (a, b + near, c, b + far)
            elif drop.step_dir == "-x":
                r = (c - far, b, c - near, d)
            else:
                r = (a + near, b, a + far, d)
            top = -drop.depth_mm * (i + 1) // drop.steps
            out.append(
                {
                    "id": f"{drop.id}_step_{i + 1}",
                    "zh": drop.zh,
                    "kind": "stairs",
                    "shape": "box",
                    "pos_mm": [(r[0] + r[2]) // 2, (r[1] + r[3]) // 2, -depth],
                    "size_mm": [r[2] - r[0], r[3] - r[1], top + depth],
                    "colour": "#8a6a4a",
                    "friction": list(OFFICIAL_FLOOR_FRICTION),
                }
            )
    return out


def _build(raw: dict[str, Any], mjcf_dir: str | None, warnings: list[str]) -> Layout:
    raw = copy.deepcopy(raw)
    _official_ops(raw)
    home_id = str(raw.get("id", "home"))
    bounds = _rect(raw.get("bounds_mm"), "bounds_mm")
    ceiling = _int(raw.get("ceiling_mm", 2400), "ceiling_mm")
    north = _int(raw.get("north_mrad", 1571), "north_mrad")
    rooms = tuple(
        Room(
            str(r["id"]),
            str(r.get("zh", r["id"])),
            _rect(r["rect_mm"], r["id"]),
            str(r.get("floor", "")),
        )
        for r in raw.get("rooms", [])
    )
    nest = _nest(raw.get("nest"))
    drops = [
        Drop(
            id=str(d["id"]),
            zh=str(d.get("zh", msg("world.homelayout._build.00f7ec53"))),
            rect_mm=_rect(d["rect_mm"], d["id"]),
            depth_mm=_int(d["depth_mm"], d["id"]),
            buffer_mm=_int(d.get("buffer_mm", DROP_BUFFER_MM), d["id"]),
            steps=_int(d.get("steps", 0), d["id"]),
            step_dir=str(d.get("step_dir", "-y")),
            cut=bool(d.get("cut", True)),
            gated=bool(d.get("gated", False)),
            official=tuple(str(v) for v in d.get("official", [])),
        )
        for d in raw.get("drops", [])
    ]
    barriers = [
        Barrier(
            id=str(b["id"]),
            kind=str(b.get("kind", "railing")),
            zh=str(b.get("zh", msg("world.homelayout._build.6710d0d4"))),
            from_mm=_int2(b["from_mm"], b["id"]),
            to_mm=_int2(b["to_mm"], b["id"]),
            height_mm=_int(b["height_mm"], b["id"]),
            colour=_colour(b.get("colour", "#f2efe8"), b["id"]),
            drop=str(b.get("drop", "")),
        )
        for b in raw.get("barriers", [])
    ]
    element_raw: list[dict[str, Any]] = [dict(e) for e in raw.get("elements", [])]
    element_raw += _drop_elements(raw, drops)
    for barrier in barriers:
        element_raw += _barrier_elements(barrier)
    lights_raw = raw.get("lights", {}) or {}
    for window in lights_raw.get("windows", []):
        if window.get("panel", True):
            element_raw.append(_window_element(window))
    if nest is not None:
        element_raw += _nest_elements(nest)
    mat_raw = raw.get("mat") or {"x_mm": 0, "y_mm": 0}
    mat = Mat(
        x_mm=_int(mat_raw["x_mm"], "mat"),
        y_mm=_int(mat_raw["y_mm"], "mat"),
        heading_mrad=_int(mat_raw.get("heading_mrad", 0), "mat"),
        radius_mm=_int(mat_raw.get("radius_mm", 340), "mat"),
        colour=_colour(mat_raw.get("colour", "#9fb8b0"), "mat"),
    )
    if nest is not None and (
        nest.is_dock  # the dock replaces the family's mat (as the open nest on it did)
        or math.hypot(mat.x_mm - nest.x_mm, mat.y_mm - nest.y_mm)
        < mat.radius_mm + nest.board_offset_mm + 20
    ):
        mat = dataclasses.replace(mat, drawn=False)
    if mat.drawn:
        element_raw.append(
            {
                "id": "mat",
                "zh": msg("world.homelayout._build.1150b903"),
                "kind": "mat",
                "shape": "cylinder",
                "pos_mm": [mat.x_mm, mat.y_mm, 0],
                "radius_mm": mat.radius_mm,
                "height_mm": 1,
                "colour": mat.colour,
                "collide": False,
                "tof": False,
            }
        )
    elements = [_element(e) for e in element_raw]
    if raw.get("wall_bands"):
        elements += [_band(e, ceiling) for e in elements if e.kind == "wall" and e.top_mm < ceiling]
    if raw.get("ceiling", True):
        x0, y0, x1, y1 = bounds
        elements.append(
            _element(
                {
                    "id": "ceiling",
                    "zh": msg("world.homelayout._build.19c8ee9a"),
                    "kind": "ceiling",
                    "shape": "box",
                    "pos_mm": [(x0 + x1) // 2, (y0 + y1) // 2, ceiling],
                    "size_mm": [x1 - x0, y1 - y0, 20],
                    "colour": str(raw.get("ceiling_colour", "#eeeee6")),
                    "collide": False,
                    "page": False,
                }
            )
        )
    things = [_thing(t, elements) for t in raw.get("things", [])]
    ids = [e.id for e in elements] + [t.id for t in things]
    dupes = sorted({x for x in ids if ids.count(x) > 1})
    if dupes:
        raise LayoutError(f"這些名字用了不只一次：{', '.join(dupes)}")

    def room_of(x0: float, y0: float, x1: float, y1: float) -> str | None:
        """The one room an outline lies in (a wall between rooms, or the ceiling, has none)."""
        if x1 - x0 < 1 or y1 - y0 < 1:  # a point
            inside = [r.id for r in rooms if r.contains(x0, y0)]
            return inside[0] if inside else None
        area = (x1 - x0) * (y1 - y0)
        shared = []
        for r in rooms:
            a, b, c, d = r.rect_mm
            w, h = min(x1, c) - max(x0, a), min(y1, d) - max(y0, b)
            if w > 0 and h > 0 and w * h > 0.05 * area:
                shared.append(r.id)
        return shared[0] if len(shared) == 1 else None

    elements = [
        e if e.room is not None else _with_room(e, room_of(*e.footprint())) for e in elements
    ]
    things = [_with_room(t, room_of(*t.footprint())) for t in things]

    def room_at_point(item: Mapping[str, Any]) -> str:
        x, y, _ = _int3(item["pos_mm"], str(item["id"]))
        return str(item.get("room") or room_of(x, y, x, y) or "")

    lamps = tuple(
        Lamp(
            id=str(x["id"]),
            zh=str(x.get("zh", msg("world.homelayout._build.0252ab47"))),
            room=room_at_point(x),
            pos_mm=_int3(x["pos_mm"], x["id"]),
            kind=str(x.get("kind", "ceiling")),
            hours=_int2(x.get("hours", [17, 23]), x["id"]),
            colour=_colour(x.get("colour", "#ffd3a0"), x["id"]),
            lux=_int(x.get("lux", 120), x["id"]),
            element=str(x.get("element", "")),
        )
        for x in lights_raw.get("lamps", [])
    )
    windows = tuple(
        Window(
            id=str(x["id"]),
            zh=str(x.get("zh", msg("world.homelayout._build.692f5fa1"))),
            room=room_at_point(x),
            pos_mm=_int3(x["pos_mm"], x["id"]),
            width_mm=_int(x["width_mm"], x["id"]),
            height_mm=_int(x["height_mm"], x["id"]),
            inward_mrad=_int(x["inward_mrad"], x["id"]),
            curtains=x.get("curtains"),
        )
        for x in lights_raw.get("windows", [])
    )
    night_lights = tuple(
        NightLight(
            id=str(x["id"]),
            zh=str(x.get("zh", msg("world.homelayout._build.5ed15e58"))),
            room=room_at_point(x),
            pos_mm=_int3(x["pos_mm"], x["id"]),
            hours=_int2(x.get("hours", [23, 7]), x["id"]),
            colour=_colour(x.get("colour", "#ffe2b0"), x["id"]),
        )
        for x in lights_raw.get("night_lights", [])
    )
    spawn_raw = raw.get("spawn", "nest")
    if isinstance(spawn_raw, Mapping):
        spawn = Pose(
            _int(spawn_raw["x_mm"], "spawn"),
            _int(spawn_raw["y_mm"], "spawn"),
            _int(spawn_raw.get("heading_mrad", 0), "spawn"),
        )
    elif spawn_raw == "nest" and nest is not None:
        spawn = nest.seat  # seated in the dock (or on the open nest's pad), facing its way
    else:
        spawn = mat.pose
    frame = _frame(raw.get("frame"), mat)
    base = dict(raw.get("base") or {})
    return Layout(
        id=home_id,
        zh=str(raw.get("zh", home_id)),
        variant="",
        source=raw,
        rooms=rooms,
        elements=tuple(elements),
        things=tuple(things),
        drops=tuple(drops),
        zones=tuple(
            Zone(str(z["id"]), str(z["kind"]), _rect(z["rect_mm"], z["id"]), str(z.get("zh", "")))
            for z in raw.get("zones", [])
        ),
        barriers=tuple(barriers),
        doorways=tuple(dict(d) for d in raw.get("doorways", [])),
        nest=nest,
        mat=mat,
        spawn=spawn,
        origin=frame,
        lights=Lights(north, lamps, windows, night_lights),
        agents=dict(raw.get("agents", {})),
        ceiling_mm=ceiling,
        bounds_mm=bounds,
        north_mrad=north,
        base=base,
        hidden=tuple(str(v) for v in raw.get("hide", [])),
        variants=(),
        mjcf_dir=mjcf_dir,
        warnings=tuple(warnings),
    )


def _frame(raw: Any, mat: Mat) -> Pose | None:
    """The layout's ``frame`` (Duck Core's origin): absent or "nest": its nest's home, else its
    mat (None here: :attr:`Layout.frame` decides); "mat"; "room": the world's origin; or a pose
    ``{x_mm, y_mm, heading_mrad}``."""
    if raw is None or raw == "nest":
        return None
    if raw == "mat":
        return mat.pose
    if raw == "room":
        return Pose(0, 0, 0)
    if isinstance(raw, Mapping):
        return Pose(
            _int(raw.get("x_mm", 0), "frame"),
            _int(raw.get("y_mm", 0), "frame"),
            _int(raw.get("heading_mrad", 0), "frame"),
        )
    raise LayoutError(f"frame {raw!r} 不認得（nest、mat、room，或 x_mm、y_mm、heading_mrad）")


def _with_room(item: Any, room: str | None) -> Any:
    return dataclasses.replace(item, room=room)


def _band(wall: Element, ceiling: int) -> Element:
    return dataclasses.replace(
        wall,
        id=f"{wall.id}_upper",
        kind="wall_upper",
        pos_mm=(wall.pos_mm[0], wall.pos_mm[1], wall.top_mm),
        size_mm=(wall.size_mm[0], wall.size_mm[1], ceiling - wall.top_mm),
        collide=False,
        page=False,
        official=False,
        room=None,
    )


# ── geometry helpers ────────────────────────────────────────────────────────────


def geom_group(camera: bool, tof: bool) -> int:
    """The MuJoCo geom group for what sees it (collision is contype/conaffinity)."""
    if camera:
        return GROUP_SEEN if tof else GROUP_CAMERA
    return GROUP_TOF if tof else GROUP_HIDDEN


def _rect_footprint(
    x: float, y: float, sx: float, sy: float, yaw_mrad: float
) -> tuple[float, float, float, float]:
    if yaw_mrad == 0:
        return x - sx / 2, y - sy / 2, x + sx / 2, y + sy / 2
    xs, ys = [], []
    for a in (-sx / 2, sx / 2):
        for b in (-sy / 2, sy / 2):
            dx, dy = _rot(a, b, yaw_mrad)
            xs.append(x + dx)
            ys.append(y + dy)
    return min(xs), min(ys), max(xs), max(ys)


def _passes_through(
    element: Element, a: tuple[float, float], b: tuple[float, float], eps: float = 1e-6
) -> bool:
    """Whether the floor segment a→b goes into this element's outline and out again between
    its ends (Liang–Barsky against the element's own box, in its frame; round shapes and
    capsules by their bounding box)."""
    cx: float
    cy: float
    if element.shape in ("box", "plane", "ellipsoid"):
        cx, cy = float(element.pos_mm[0]), float(element.pos_mm[1])
        hx, hy = element.size_mm[0] / 2.0, element.size_mm[1] / 2.0
        ax, ay = _rot(a[0] - cx, a[1] - cy, -element.yaw_mrad)
        bx, by = _rot(b[0] - cx, b[1] - cy, -element.yaw_mrad)
    else:
        x0, y0, x1, y1 = element.footprint()
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        hx, hy = (x1 - x0) / 2.0, (y1 - y0) / 2.0
        ax, ay, bx, by = a[0] - cx, a[1] - cy, b[0] - cx, b[1] - cy
    dx, dy = bx - ax, by - ay
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, ax + hx), (dx, hx - ax), (-dy, ay + hy), (dy, hy - ay)):
        if p == 0.0:
            if q < 0.0:
                return False
            continue
        r = q / p
        if p < 0.0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
        if t0 > t1:
            return False
    return t0 > eps and t1 < 1.0 - eps


def _supports(element: Element, x_mm: float, y_mm: float) -> bool:
    if element.shape == "plane":
        x, y, _ = element.pos_mm
        return abs(x_mm - x) <= element.size_mm[0] / 2 and abs(y_mm - y) <= element.size_mm[1] / 2
    return element.contains_xy(x_mm, y_mm)


def _overlap(a: tuple[float, ...], b: tuple[float, ...], grow: float = 0.0) -> float:
    """How far two (x0, y0, x1, y1) rectangles overlap (negative: the gap between them)."""
    return min(a[2] + grow - b[0], b[2] + grow - a[0], a[3] + grow - b[1], b[3] + grow - a[1])


def _gap(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    dx = max(b[0] - a[2], a[0] - b[2], 0.0)
    dy = max(b[1] - a[3], a[1] - b[3], 0.0)
    return math.hypot(dx, dy)


def _z_overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return min(a[1], b[1]) - max(a[0], b[0])


def _local_box(pose: Pose, box: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    x0, x1, y0, y1 = box
    xs, ys = [], []
    for a in (x0, x1):
        for b in (y0, y1):
            wx, wy = pose.to_world(a, b)
            xs.append(wx)
            ys.append(wy)
    return min(xs), min(ys), max(xs), max(ys)


def _circle_rect_gap(cx: float, cy: float, rect: tuple[float, ...]) -> float:
    dx = max(rect[0] - cx, 0.0, cx - rect[2])
    dy = max(rect[1] - cy, 0.0, cy - rect[3])
    return math.hypot(dx, dy)


# ── the generation checks ───────────────────────────────────────────────────────


def _check(layout: Layout) -> list[Problem]:
    problems: list[Problem] = []
    solid = [e for e in layout.elements if e.collide and not e.support and e.kind != "mat"]
    lows = [e for e in layout.elements if e.support and e.top_mm < -10]
    drops: list[tuple[tuple[float, float, float, float], int, str]] = [
        (
            (float(d.rect_mm[0]), float(d.rect_mm[1]), float(d.rect_mm[2]), float(d.rect_mm[3])),
            d.buffer_mm,
            d.id,
        )
        for d in layout.drops
    ]
    drops += [(e.footprint(), DROP_BUFFER_MM, e.id) for e in lows]
    things = list(layout.things)
    # Things: not inside furniture, not on an edge or over a hole, not near a drop, not in a
    # trip zone, not overlapping each other (curiosity spec §4.1).
    for thing in things:
        box, zr = thing.footprint(), thing.z_range()
        for e in solid:
            if (
                _overlap(box, e.footprint()) > TOLERANCE_MM
                and _z_overlap(zr, e.z_range()) > TOLERANCE_MM
                and _hits_outline(e, box)
            ):
                problems.append(
                    Problem("thing_in_furniture", thing.id, f"「{thing.id}」放在「{e.id}」裡面")
                )
        tops = [layout.support_top(x, y) for x, y in _samples(box)]
        heights = [t for t in tops if t is not None]
        if len(heights) < len(tops):
            problems.append(Problem("thing_over_hole", thing.id, f"「{thing.id}」有一部分懸在洞上"))
        elif max(heights) - min(heights) > TOLERANCE_MM:
            problems.append(
                Problem("thing_straddles_edge", thing.id, f"「{thing.id}」跨在高低不同的地板邊上")
            )
        if not thing.hazard:
            for rect, buffer, drop_id in drops:
                if _gap(box, rect) < buffer:
                    problems.append(
                        Problem(
                            "thing_near_drop",
                            thing.id,
                            f"「{thing.id}」離落差「{drop_id}」不到 {buffer} mm",
                        )
                    )
                    break
            for zone in layout.zones:
                if zone.kind in ("trip", "drop", "forbid") and _overlap(box, zone.rect_mm) > 0:
                    problems.append(
                        Problem("thing_in_zone", thing.id, f"「{thing.id}」在「{zone.id}」區裡")
                    )
    for i, a in enumerate(things):
        for b in things[i + 1 :]:
            if (
                _overlap(a.footprint(), b.footprint()) > TOLERANCE_MM
                and _z_overlap(a.z_range(), b.z_range()) > TOLERANCE_MM
            ):
                problems.append(Problem("thing_overlap", a.id, f"「{a.id}」和「{b.id}」疊在一起"))
    problems += _check_nest(layout, solid, drops)
    # The duck starts clear of everything (standing footprint and height). A duck spawned in
    # its dock sits between the dock's own guides, which are sized around the seated duck
    # (duckmind.world.dock; the world's x_place checks the real shapes): those are left out.
    spawn = layout.spawn
    duck_z = (float(spawn.z_mm), float(spawn.z_mm + DUCK_HEIGHT_MM))
    in_dock = layout.nest is not None and layout.nest.is_dock and spawn == layout.nest.seat
    for e in solid:
        if in_dock and e.id.startswith("nest_"):
            continue
        if (
            _circle_rect_gap(spawn.x_mm, spawn.y_mm, e.footprint()) < DUCK_RADIUS_MM
            and _z_overlap(duck_z, e.z_range()) > TOLERANCE_MM
            and _circle_hits_outline(e, spawn.x_mm, spawn.y_mm, DUCK_RADIUS_MM)
        ):
            problems.append(Problem("spawn_blocked", e.id, f"小鴨一開始的位置被「{e.id}」擋住"))
    for thing in things:
        if _circle_rect_gap(spawn.x_mm, spawn.y_mm, thing.footprint()) < DUCK_RADIUS_MM:
            problems.append(Problem("spawn_blocked", thing.id, f"小鴨一開始的位置有「{thing.id}」"))
    if layout.support_top(spawn.x_mm, spawn.y_mm) is None:
        problems.append(Problem("spawn_blocked", "floor", "小鴨一開始的位置底下沒有地板"))
    # Light: every room has a lamp or a window, and no more sources than a renderer can light.
    for room in layout.rooms:
        count = layout.lights.for_room(room.id).count
        if count == 0:
            problems.append(Problem("room_without_light", room.id, f"「{room.zh}」沒有燈也沒有窗"))
        if count > ROOM_LIGHT_MAX:
            problems.append(
                Problem(
                    "room_too_many_lights",
                    room.id,
                    f"「{room.zh}」有 {count} 個光源，畫面最多只能點 {ROOM_LIGHT_MAX} 個",
                )
            )
    problems += _check_gates(layout)
    return problems


def _hits_outline(element: Element, box: tuple[float, float, float, float]) -> bool:
    """Does a thing's box touch the element's own outline (a round pot, not its square box)?"""
    if element.shape in ("cylinder", "sphere", "frustum"):
        x, y, _ = element.pos_mm
        r = max(element.radius_mm, element.radius_bottom_mm)
        return _circle_rect_gap(x, y, box) < r - TOLERANCE_MM
    if element.yaw_mrad == 0 or element.shape == "capsule":
        return True
    return any(element.contains_xy(x, y) for x, y in _samples(box))


def _circle_hits_outline(element: Element, cx: float, cy: float, r: float) -> bool:
    if element.shape in ("cylinder", "sphere", "frustum"):
        x, y, _ = element.pos_mm
        er = max(element.radius_mm, element.radius_bottom_mm)
        return math.hypot(cx - x, cy - y) < r + er
    return True


def _samples(box: tuple[float, float, float, float]) -> list[tuple[float, float]]:
    x0, y0, x1, y1 = box[0] + 1, box[1] + 1, box[2] - 1, box[3] - 1
    xs, ys = (x0, (x0 + x1) / 2, x1), (y0, (y0 + y1) / 2, y1)
    return [(x, y) for x in xs for y in ys]


def _check_nest(
    layout: Layout, solid: list[Element], drops: list[tuple[Any, int, str]]
) -> list[Problem]:
    nest = layout.nest
    if nest is None:
        return []
    problems: list[Problem] = []
    # (open: the pad and board, and the right-arc exit; dock: the dock itself, and the lane
    # and turning room in front of it)
    zones = {label: _local_box(nest.pose, box) for label, box in nest.zones_mm().items()}
    x0, y0, x1, y1 = layout.bounds_mm
    seen = [
        e
        for e in layout.elements
        if not e.collide
        and e.camera
        and e.kind not in DECOR_KINDS
        and e.top_mm > 10
        and not e.id.startswith("nest_")
    ]
    said = "充電座" if nest.is_dock else "窩"
    for label, box in zones.items():
        where = f"{said}的{_ZONE_ZH[label]}" if not (nest.is_dock and label == "back") else said
        if box[0] < x0 or box[1] < y0 or box[2] > x1 or box[3] > y1:
            problems.append(Problem("nest_blocked", "bounds", f"{said}的{label}區超出家的範圍"))
        for e in solid:
            if e.id.startswith("nest_"):
                continue
            if nest.is_dock and label == "back" and e.kind in WALL_KINDS:
                continue  # the wall (and its skirting) the dock stands against
            if (
                _overlap(box, e.footprint(), NEST_MARGIN_MM) > 0
                and e.pos_mm[2] < DUCK_HEIGHT_MM
                and _circle_or_box_touch(e, box, NEST_MARGIN_MM)
            ):
                problems.append(Problem("nest_blocked", e.id, f"{where}被「{e.id}」擋住"))
        for e in seen:
            if _overlap(box, e.footprint()) > 0 and e.pos_mm[2] < DUCK_HEIGHT_MM:
                problems.append(Problem("nest_blocked", e.id, f"{where}有看得到的「{e.id}」"))
        for rect, _buffer, drop_id in drops:
            if _gap(box, rect) < DROP_BUFFER_MM:
                problems.append(
                    Problem(
                        "nest_near_drop",
                        drop_id,
                        f"{said}離落差「{drop_id}」不到 {DROP_BUFFER_MM} mm",
                    )
                )
    if nest.is_dock:
        problems += _check_dock_wall(layout, nest)
        centre, clear = nest.seat, float(NEST_CLEAR_MM)
    else:
        centre, clear = nest.pose, float(nest.pad_radius_mm + NEST_CLEAR_MM)
    for thing in layout.things:
        box = thing.footprint()
        near = _circle_rect_gap(centre.x_mm, centre.y_mm, box) < clear
        if near or any(_overlap(box, z) > 0 for z in zones.values()):
            problems.append(Problem("thing_near_nest", thing.id, f"「{thing.id}」太靠近{said}"))
    for zone in layout.zones:  # tidying up must never put toys on the charging place
        if zone.kind == "keepsake" and any(_overlap(zone.rect_mm, z) > 0 for z in zones.values()):
            problems.append(
                Problem("zone_on_nest", zone.id, f"「{zone.zh or zone.id}」和{said}的地方重疊")
            )
    return problems


def _check_dock_wall(layout: Layout, nest: Nest) -> list[Problem]:
    """A dock stands with its back on a wall: just behind its back, across its width, there
    must be a wall (or its skirting) within DOCK_WALL_REACH_MM."""
    shape = nest.dock_shape
    walls = [e for e in layout.elements if e.kind in WALL_KINDS and e.collide]
    half = shape.width_mm / 2 - 20
    for ly in (-half, 0.0, half):
        px, py = nest.to_world(-DOCK_WALL_REACH_MM / 2, ly)
        if not any(
            _circle_rect_gap(px, py, wall.footprint()) <= DOCK_WALL_REACH_MM / 2 for wall in walls
        ):
            return [Problem("dock_not_on_wall", "nest", "充電座的背後沒有靠著牆")]
    return []


_ZONE_ZH = {
    "back": "墊子和背板",
    "exit": "出入口",
    "lane": "正前方 0.9 公尺的進出路",
    "turn": "前面轉身的地方",
}


def _circle_or_box_touch(element: Element, box: tuple[float, ...], grow: float) -> bool:
    if element.shape in ("cylinder", "sphere", "frustum"):
        x, y, _ = element.pos_mm
        r = max(element.radius_mm, element.radius_bottom_mm)
        return _circle_rect_gap(x, y, box) < r + grow
    return True


def _check_gates(layout: Layout) -> list[Problem]:
    problems: list[Problem] = []
    for drop in layout.drops:
        if not drop.gated:
            continue
        x0, y0, x1, y1 = drop.rect_mm
        sides = {
            "南": ("h", y0, -1, x0, x1),
            "北": ("h", y1, 1, x0, x1),
            "西": ("v", x0, -1, y0, y1),
            "東": ("v", x1, 1, y0, y1),
        }
        for side, (axis, line, outward, lo, hi) in sides.items():
            spans = []
            for b in layout.barriers:
                (ax, ay), (bx, by) = b.from_mm, b.to_mm
                if b.height_mm < BARRIER_MIN_HEIGHT_MM:
                    continue
                if axis == "h" and ay == by:
                    offset, span = (ay - line) * outward, (min(ax, bx), max(ax, bx))
                elif axis == "v" and ax == bx:
                    offset, span = (ax - line) * outward, (min(ay, by), max(ay, by))
                else:
                    continue
                if 0 <= offset <= BARRIER_REACH_MM:
                    spans.append(span)
            if not _covers(spans, lo, hi):
                problems.append(
                    Problem("drop_not_gated", drop.id, f"「{drop.zh}」的{side}邊沒有欄杆或門擋住")
                )
    return problems


def _covers(spans: list[tuple[int, int]], lo: int, hi: int) -> bool:
    reach = lo
    for a, b in sorted(spans):
        if a > reach:
            return False
        reach = max(reach, b)
    return reach >= hi


# ── composing the MuJoCo model ────────────────────────────────────────────────


def _mujoco() -> Any:
    import mujoco

    return mujoco


def compose(
    layout: Layout,
    collision: str = "allcollisions",
    *,
    mjcf_dir: str | os.PathLike[str] | None = None,
    nest: bool = True,
    check: bool = True,
) -> Any:
    """ONE ``mujoco.MjSpec`` for the home: the official scene read where it lies (never copied
    or edited), the hidden official pieces taken out in memory, and this home's elements and
    things added. Not compiled: other world modules (nest sensors, hand, people) add to it,
    call :func:`fix_keyframes` again if they add bodies, then compile.

    ``nest=False`` leaves the passive nest out (for a nest module that builds its own);
    ``check`` refuses a home whose generation checks fail."""
    if collision not in SCENES:
        raise LayoutError(f"碰撞模型 {collision!r} 不認得（{', '.join(SCENES)}）")
    if check:
        problems = layout.check()
        if problems:
            raise LayoutError(
                f"「{layout.id}」的佈置檢查沒過：" + "；".join(str(p) for p in problems)
            )
    mujoco = _mujoco()
    folder = official_dir(mjcf_dir if mjcf_dir is not None else layout.mjcf_dir)
    spec = _base_spec(mujoco, layout, collision, folder)
    if layout.hidden:
        # MuJoCo pads short keyframes when it compiles; deleting from a spec whose keyframes
        # are still the file's 21 values fails, so compile once first.
        spec.compile()
    for name in layout.hidden:
        _delete(spec, name, layout.id)
    for element in layout.elements:
        if element.official:
            _official_element(mujoco, spec, element)
        elif nest or not element.id.startswith("nest_"):
            _add_element(mujoco, spec, element)
    for thing in layout.things:
        if thing.official:
            body = spec.body(thing.body)
            if body is None:
                raise LayoutError(f"上游的「{thing.body}」不在上游的場景裡")
            body.pos = [thing.pos_mm[0] / 1000.0, thing.pos_mm[1] / 1000.0, float(body.pos[2])]
        else:
            _add_thing(mujoco, spec, thing)
    fix_keyframes(spec, layout)
    return spec


def _base_spec(mujoco: Any, layout: Layout, collision: str, folder: Path) -> Any:
    world = layout.base.get("world")
    scenes = dict(layout.base.get("scenes") or {})
    if world and collision in scenes:
        return mujoco.MjSpec.from_file(str(folder / scenes[collision]))
    spec = mujoco.MjSpec.from_file(str(folder / SCENES[collision]))
    if world:
        # The official world on the other robot: the scene without its plane, the world
        # attached under its own names (as scene_apartment.xml includes it).
        spec.delete(spec.geom(OFFICIAL_FLOOR))
        child = mujoco.MjSpec.from_file(str(folder / str(world)))
        spec.attach(child, frame=spec.worldbody.add_frame(), prefix="")
    return spec


def _delete(spec: Any, name: str, home: str) -> None:
    target = spec.geom(name)
    if target is None:
        target = spec.body(name)
    if target is None:
        raise LayoutError(f"「{home}」要藏起來的上游「{name}」不在上游的場景裡")
    spec.delete(target)


def _official_element(mujoco: Any, spec: Any, element: Element) -> None:
    geom = spec.geom(element.geom)
    if geom is None:
        raise LayoutError(f"上游的「{element.geom}」不在上游的場景裡")
    if int(geom.group) in (GROUP_SEEN, GROUP_CAMERA, GROUP_TOF, GROUP_HIDDEN):
        geom.group = element.group
    if not element.collide:
        geom.contype = 0
        geom.conaffinity = 0
    _look(mujoco, spec, geom, element)


# ── what a surface looks like: one layout decides it for the page and the camera ──────────
# The page draws a "wood" element with home3d.js woodTexture(): planks in these tones, 8 rows to
# a 1.2 m tile. The physics' copy for the head camera gets the same planks as a texture (the
# official plane's own blue-grey checker would otherwise be what the duck sees under a floor the
# page draws as wood). Only the look changes: collision, friction and groups stay the official.
WOOD_TONES = ("#c9a47a", "#c29b70", "#cfac84", "#bd966b", "#c6a176")
WOOD_TILE_M = 1.2
WOOD_TEXTURE = "home_wood"
WOOD_MATERIAL = "home_wood_mat"
WOOD_PX = 512


def wood_rgb(size: int = WOOD_PX) -> bytes:
    """The page's wood planks (``woodTexture`` in home3d.js, same arithmetic) as ``size`` ×
    ``size`` RGB bytes, rows top to bottom: plank rows 1/8 of the tile high, lengths and tones
    from the same formulas, a darker 2 px outline and faint grain lines."""
    return _wood_rgb(size)


def _js_mod(a: float, b: float) -> float:
    return math.fmod(a, b)  # JavaScript's % keeps the dividend's sign


def _wood_rgb_python(size: int) -> list[list[list[float]]]:
    scale = size / 512.0
    tones = [[int(t[i : i + 2], 16) for i in (1, 3, 5)] for t in WOOD_TONES]
    img = [[[0.0, 0.0, 0.0] for _ in range(size)] for _ in range(size)]

    def blend(px: int, py: int, rgb: tuple[int, int, int], alpha: float) -> None:
        if 0 <= px < size and 0 <= py < size:
            cell = img[py][px]
            for c in range(3):
                cell[c] = cell[c] * (1 - alpha) + rgb[c] * alpha

    plank = 64
    for row in range(512 // plank):
        x = -_js_mod(row * 97, 300)
        while x < 512:
            length = 180 + _js_mod(row * 53 + x, 140)
            tone = tones[int(row * 3 + abs(math.floor(x / 7 + 0.5))) % len(tones)]
            y0 = row * plank
            x0, x1 = max(0, int(x * scale)), min(size, int((x + length) * scale))
            for py in range(int(y0 * scale), int((y0 + plank) * scale)):
                for px in range(x0, x1):
                    img[py][px] = [float(v) for v in tone]
            # the outline: strokeRect(x + 1, y + 1, len − 2, plank − 2), 2 px wide
            left, right = x, x + length - 2
            top, bottom = y0, y0 + plank - 2
            for py in range(int(top * scale), int((bottom + 2) * scale)):
                for px in range(int(left * scale), int((left + 2) * scale)):
                    blend(px, py, (90, 60, 30), 0.35)
                for px in range(int(right * scale), int((right + 2) * scale)):
                    blend(px, py, (90, 60, 30), 0.35)
            for px in range(int(left * scale), int((right + 2) * scale)):
                for py in range(int(top * scale), int((top + 2) * scale)):
                    blend(px, py, (90, 60, 30), 0.35)
                for py in range(int(bottom * scale), int((bottom + 2) * scale)):
                    blend(px, py, (90, 60, 30), 0.35)
            # the grain: five faint curves along the plank
            for k in range(5):
                gy = y0 + 8 + k * 11
                for px in range(int((x + 4) * scale), int((x + length - 4) * scale)):
                    t = min(1.0, max(0.0, (px / scale - x - 4) / max(1.0, length - 8)))
                    wave = 3 * (3 * (1 - t) ** 2 * t) - 3 * (3 * (1 - t) * t**2) + t**3
                    py = int((gy + wave) * scale)
                    blend(px, py, (120, 80, 40), 0.10)
                    blend(px, py + 1, (120, 80, 40), 0.10)
            x += length
    return img


def _wood_rgb(size: int) -> bytes:
    key = int(size)
    cached = _WOOD_CACHE.get(key)
    if cached is None:
        img = _wood_rgb_python(key)
        cached = bytes(int(round(min(255.0, max(0.0, v)))) for row in img for px in row for v in px)
        _WOOD_CACHE[key] = cached
    return cached


_WOOD_CACHE: dict[int, bytes] = {}


def _look(mujoco: Any, spec: Any, geom: Any, element: Element) -> None:
    """The element's look for the camera, as the page draws it: a ``wood`` texture gets the
    page's planks at the page's scale (the colour alone is set by the caller)."""
    if element.texture != "wood":
        return
    # (an MjSpec finds what was added since its last compile only by walking its list)
    if not any(material.name == WOOD_MATERIAL for material in spec.materials):
        texture = spec.add_texture(
            name=WOOD_TEXTURE,
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            width=WOOD_PX,
            height=WOOD_PX,
            nchannel=3,
        )
        texture.data = wood_rgb(WOOD_PX)
        # sRGB bytes, as the page's canvas texture (three.js decodes it to linear reflectance
        # before lighting); the camera renders in linear light, so it decodes them too
        texture.colorspace = mujoco.mjtColorSpace.mjCOLORSPACE_SRGB
        material = spec.add_material(
            name=WOOD_MATERIAL, specular=0.0, shininess=0.0, reflectance=0.0
        )
        material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = WOOD_TEXTURE
        material.texuniform = True
        material.texrepeat = [1.0 / WOOD_TILE_M, 1.0 / WOOD_TILE_M]
    geom.material = WOOD_MATERIAL
    geom.rgba = [1.0, 1.0, 1.0, element.opacity]  # the planks' own colours


def _rgba(colour: str, opacity: float) -> list[float]:
    value = int(colour[1:], 16)
    return [
        ((value >> 16) & 255) / 255.0,
        ((value >> 8) & 255) / 255.0,
        (value & 255) / 255.0,
        opacity,
    ]


def _frustum_mesh(mujoco: Any, spec: Any, name: str, top: float, bottom: float, h: float) -> None:
    verts: list[float] = []
    for z, r in ((0.0, bottom), (h, top)):
        for i in range(32):
            a = 2 * math.pi * i / 32
            verts += [r * math.cos(a), r * math.sin(a), z]
    spec.add_mesh(name=name, uservert=verts)


def _slab_mesh(spec: Any, name: str, sx: float, sy: float, h: float, ramp: float) -> None:
    """A low plate ramped on every side (convex): its foot ``sx`` × ``sy``, its top ``ramp``
    smaller on every side, ``h`` high; in its own frame, the foot at z 0."""
    verts: list[float] = []
    for z, dx, dy in ((0.0, sx / 2, sy / 2), (h, sx / 2 - ramp, sy / 2 - ramp)):
        for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            verts += [a * dx, b * dy, z]
    spec.add_mesh(name=name, uservert=verts)


def _add_element(mujoco: Any, spec: Any, element: Element) -> None:
    g = mujoco.mjtGeom
    geom = spec.worldbody.add_geom()
    geom.name = element.geom
    x, y, z = (v / 1000.0 for v in element.pos_mm)
    sx, sy, sz = (v / 1000.0 for v in element.size_mm)
    shape = element.shape
    if shape == "box":
        geom.type = g.mjGEOM_BOX
        geom.size = [sx / 2, sy / 2, sz / 2]
        geom.pos = [x, y, z + sz / 2]
    elif shape == "ellipsoid":
        geom.type = g.mjGEOM_ELLIPSOID
        geom.size = [sx / 2, sy / 2, sz / 2]
        geom.pos = [x, y, z + sz / 2]
    elif shape == "cylinder":
        geom.type = g.mjGEOM_CYLINDER
        geom.size = [element.radius_mm / 1000.0, sz / 2, 0.0]
        geom.pos = [x, y, z + sz / 2]
    elif shape == "sphere":
        r = element.radius_mm / 1000.0
        geom.type = g.mjGEOM_SPHERE
        geom.size = [r, 0.0, 0.0]
        geom.pos = [x, y, z + r]
    elif shape == "capsule":
        geom.type = g.mjGEOM_CAPSULE
        geom.size = [element.radius_mm / 1000.0, 0.0, 0.0]
        geom.fromto = [v / 1000.0 for v in element.fromto_mm]
    elif shape == "frustum":
        mesh = f"{element.geom}_mesh"
        _frustum_mesh(
            mujoco,
            spec,
            mesh,
            element.radius_mm / 1000.0,
            element.radius_bottom_mm / 1000.0,
            sz,
        )
        geom.type = g.mjGEOM_MESH
        geom.meshname = mesh
        geom.pos = [x, y, z]
    elif shape == "slab":
        mesh = f"{element.geom}_mesh"
        _slab_mesh(spec, mesh, sx, sy, sz, element.ramp_mm / 1000.0)
        geom.type = g.mjGEOM_MESH
        geom.meshname = mesh
        geom.pos = [x, y, z]
    else:
        raise LayoutError(f"「{element.id}」的形狀 {shape} 不能加進物理（只有上游的地板是平面）")
    if element.yaw_mrad and shape != "capsule":
        geom.quat = yaw_quat(element.yaw_mrad)
    geom.rgba = _rgba(element.colour, element.opacity)
    _look(mujoco, spec, geom, element)
    geom.group = element.group
    bits = 1 if element.collide else 0
    geom.contype = bits
    geom.conaffinity = bits
    if element.friction is not None:
        geom.friction = list(element.friction)
    if element.solref is not None:
        geom.solref = list(element.solref)
        if element.solmix is not None:
            geom.solmix = element.solmix


def _add_thing(mujoco: Any, spec: Any, thing: Thing) -> None:
    g = mujoco.mjtGeom
    body = spec.worldbody.add_body()
    body.name = thing.body
    body.pos = [thing.pos_mm[0] / 1000.0, thing.pos_mm[1] / 1000.0, thing.pos_mm[2] / 1000.0]
    if thing.yaw_mrad:
        body.quat = yaw_quat(thing.yaw_mrad)
    joint = body.add_freejoint()
    joint.name = f"{thing.body}_free"
    rgba = _rgba(thing.colour, thing.opacity)
    for k, part in enumerate(thing.parts):
        geom = body.add_geom()
        geom.name = f"{thing.body}_{k}"
        ox, oy, oz = (v / 1000.0 for v in part.offset_mm)
        sx, sy, sz = (v / 1000.0 for v in part.size_mm)
        if part.shape == "sphere":
            r = part.radius_mm / 1000.0
            geom.type = g.mjGEOM_SPHERE
            geom.size = [r, 0.0, 0.0]
            geom.pos = [ox, oy, oz + r]
            geom.condim = 6
        elif part.shape == "cylinder":
            geom.type = g.mjGEOM_CYLINDER
            geom.size = [part.radius_mm / 1000.0, sz / 2, 0.0]
            geom.pos = [ox, oy, oz + sz / 2]
            geom.condim = 6
        else:
            geom.type = g.mjGEOM_BOX
            geom.size = [sx / 2, sy / 2, sz / 2]
            geom.pos = [ox, oy, oz + sz / 2]
        geom.mass = part.mass_g / 1000.0
        geom.friction = [thing.friction, 0.005, thing.roll]
        if thing.solref is not None:
            geom.solref = list(thing.solref)
            if thing.solmix is not None:
                geom.solmix = thing.solmix
        geom.rgba = rgba
        geom.group = GROUP_SEEN
    if len(thing.parts) == 1 and thing.parts[0].shape == "sphere":
        # MjSpec puts a lone sphere's centre of mass wrong unless it is given (curiosity
        # spec §4.1): explicit ipos and a hollow (2/3) or solid (2/5) m·r².
        part = thing.parts[0]
        r = part.radius_mm / 1000.0
        m = thing.mass_g / 1000.0
        inertia = (2.0 / 3.0 if thing.hollow else 2.0 / 5.0) * m * r * r
        body.explicitinertial = True
        body.mass = m
        body.ipos = [
            part.offset_mm[0] / 1000.0,
            part.offset_mm[1] / 1000.0,
            part.offset_mm[2] / 1000.0 + r,
        ]
        body.iquat = [1.0, 0.0, 0.0, 0.0]
        body.inertia = [inertia, inertia, inertia]


def fix_keyframes(spec: Any, layout: Layout | None = None) -> None:
    """Rewrite every keyframe so a reset does not teleport things (the keyframe trap): the
    duck at the layout's spawn (its keyframe posture kept), every other free body and every
    mocap body where the spec puts it. Compiles ``spec`` once to learn the state layout.

    The duck is moved to the spawn once per spec: calling it again (after adding free or
    mocap bodies, as the household does) leaves the duck where the first call put it — turning
    it and lifting it again would put a spawn facing anywhere but +x the wrong way round."""
    mujoco = _mujoco()
    model = spec.compile()
    robot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, ROBOT_FREEJOINT)
    mocap = [b for b in range(model.nbody) if model.body_mocapid[b] >= 0]
    place_duck = layout is not None and spec not in _SPAWNED
    for index, key in enumerate(spec.keys):
        qpos = [float(v) for v in model.key_qpos[index]]
        for joint in range(model.njnt):
            if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
                continue
            adr = int(model.jnt_qposadr[joint])
            if joint == robot:
                if place_duck and layout is not None:
                    spawn = layout.spawn
                    turn = yaw_quat(spawn.heading_mrad)
                    q = _quat_mul(turn, qpos[adr + 3 : adr + 7])
                    qpos[adr : adr + 3] = [spawn.x, spawn.y, qpos[adr + 2] + spawn.z]
                    qpos[adr + 3 : adr + 7] = q
                continue
            qpos[adr : adr + 7] = [float(v) for v in model.qpos0[adr : adr + 7]]
        key.qpos = qpos
        if mocap:
            order = sorted(mocap, key=lambda b: int(model.body_mocapid[b]))
            key.mpos = [float(v) for b in order for v in model.body_pos[b]]
            key.mquat = [float(v) for b in order for v in model.body_quat[b]]
    if place_duck:
        _SPAWNED[spec] = True


# The specs whose keyframes already carry the duck at its spawn (fix_keyframes).
_SPAWNED: weakref.WeakKeyDictionary[Any, bool] = weakref.WeakKeyDictionary()


def _quat_mul(a: list[float], b: list[float]) -> list[float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]
