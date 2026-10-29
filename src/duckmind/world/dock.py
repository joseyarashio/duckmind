"""The charging dock: a robot-vacuum-style dock standing against a wall (owner decision
2026-09-27: 「窩 做成充電座 樣式（類似掃地機器人充電座），公寓的窩 放置在牆邊」; physical-design spec
§10.5, §11.5). It replaces phase 2's open round pad and backboard as the home's charging place;
the open nest stays as the layout variant ``open_nest`` for comparison. What phase 2 settled is
kept: the contact is a sensor that pushes nothing and reads power-good (C1), the dock's accent is
cyan (C3), the tag is ``tag36h11`` (:mod:`duckmind.world.apriltag`, the same bits for the physics,
the camera and the page), and the status light replaces the backlight with the same meaning (C4:
on only when the dock's corner is dark, the charging colour while the duck charges).

**The dock's frame** (millimetres): the origin is the middle of the housing's back, at the floor,
on the wall; +x points into the room (the layout's ``heading_mrad``), +y is to the left of +x, z
is up. The layout gives that point (``x_mm``, ``y_mm``) and heading; everything here is built from
our own primitives through ``MjSpec`` (nothing official is read or copied):

* **housing** — a box on the wall (:attr:`DockShape.depth_mm` deep), warm white;
* **face panel** — the cyan panel on the housing's front carrying the tag (C3). In the
  ``forward`` style it stands :attr:`DockShape.panel_mm` proud of the housing as a soft front
  bumper (the duck's chest stops against it); in ``backin`` it is a flat 1 mm face;
* **floor plate** — a slab ramped on every side (4 mm over 40 mm: the duck's feet lift only
  ~1.1 cm, and phase 2's ramp walked in 30/30 where a 3 mm step caught 8/30), or flush
  (``plate_mm`` 0: a visual plate on the floor);
* **side guides** — a V: two low walls running parallel from the housing (the channel), then
  flaring to a wide mouth, so an approach that is a little off is funnelled toward the middle;
* **tags** — tag 17 on the face panel (seen from across the room) and tag 18 flat at the plate's
  front edge (read looking down on the way in);
* **status light** — a strip across the housing's face above the panel: an emitter for the
  camera (its geom name holds ``status_light``, see :mod:`duckmind.world.render_scene`), placed for
  :mod:`duckmind.world.lights` by :func:`light_point`;
* **charging contacts**, by style (sensor-only: ``contype`` 0, the duck's probes massless in
  group 5, so they push nothing — like phase 2's belly pin):

  - ``forward`` (the duck walks in facing the tag): two floor pads (+ and −) on the plate under
    the seated thighs, and a small probe at the bottom of each thigh. The contact is closed when
    each thigh's probe presses a different pad (either way round).
  - ``backin`` (the duck turns round in front of the dock and reverses in, as many robot
    vacuums do): two spring plates on the housing's face at the height of the seated duck's lower
    rear, and a probe at the trunk's rear. Closed when the probe presses the plates. The two
    plates are drawn as + and −, but one contact is modelled (like phase 2's single electrode:
    a two-pole layout of the rear is a hardware decision).

  Both report through the nest's contact path (:mod:`duckmind.world.nestpad`): the ``pg`` op,
  the world battery's charging, ``x_world`` → ``nest``, the status light's colour.

  The homes' docks are ``backin`` (:data:`DEFAULT_STYLE`, 2026-09-28: the style duck-body's
  dockexec docks by default); ``forward`` is the layout variant of the same name. A backin
  dock's layout also says which side the duck turns round on (``nest.side``, :data:`SIDES`):
  ``right`` unless the room has a wall there (the apartment's mirrored J-hook is ``left``).

**Sizing** (measured 2026-09-27 with phase 1's headless policy duck on the official floor; scratch
``dock/build``): standing, the duck spans x −47…+97 mm about its trunk (the beak's front at
z 229–256 mm, the chest's +38 mm at z 140–160 mm), ±71 mm wide at the hips (x −7…−67 mm), feet
±63 mm; seated, x −47…+77 mm and ±79 mm, the thighs on the floor at (−16…−19, ±50) mm and the feet
at (+38, ±41) mm, the trunk 61 mm up. Sitting down moves the trunk 66–73 mm back and ~15 mm to the
duck's left, turning it up to ~20°; standing up moves it ~48 mm forward. So the channel is
±130 mm (the seated duck's ±79 mm, its sideways step and its turn), which leaves a standing duck
±59 mm to be centred by; the plate carries the whole seated duck; the forward pads reach from
90 to 190 mm in front of the face (the thighs land ~130 mm out after the chest stops at the bumper)
and 6–100 mm to each side; the backin plates stand 15 mm proud with a 15 mm probe, a 29 mm window
for the seated rear. The walk drifts right (−8.5°/s at vx 0.35, −4.5°/s at vx −0.4; vx −0.3 steps
on the spot) and coasts 60–75 mm after a reverse stops: a backin duck is told to stop reversing
~0.3 m out. A forward U-turn (vx 0.30, vyaw −1.0) takes ~6 s and ends ~0.33 m to the right.

**What the duck may know.** Only the ``pg`` op (power-good through the contact). The rest — probe
positions, how far a contact is pressed, whether the bumper is touched — is evaluator truth in
``x_nest`` and ``x_world`` → ``nest``.

Nothing heavy is imported at import time; the geometry is plain Python (the layout loader uses
it), MuJoCo only inside :func:`edit_spec` and :class:`Contacts`.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from duckmind.i18n import msg
from duckmind.world import apriltag

STYLES = ("forward", "backin")
# The page's homes dock backin (owner 2026-09-28 「6. 可以」 — dock:measure's recommendation —
# and ~12:30: the homes switched at the next integration); ``forward`` stays a layout variant.
DEFAULT_STYLE = "backin"
# backin's J-hook: which side of the dock the duck turns round on (duck-body's duck.dock
# ``side``): ``right`` — the lane on the dock's right, a right U-turn — unless the room has a
# wall there (the apartment's corridor wall), where the mirror ``left`` is used
SIDES = ("right", "left")
DEFAULT_SIDE = "right"
KIND = "dock"

# Measured body numbers (see the module docstring), millimetres in the duck's heading frame.
REAR_MM = 47  # the trunk's rear behind the trunk origin (standing and seated)
CHEST_MM = 38  # the standing duck's chest ahead of the trunk origin (z 140–160 mm)
SIT_BACK_MM = 72  # sitting down moves the trunk this far back
STAND_FORWARD_MM = 48  # standing up moves it this far forward
REVERSE_COAST_MM = 65  # a reverse at vx −0.4 carries on ~60–75 mm after the stop command
FORWARD_STOP_MM = 15  # a forward walk-in is told to stop this far before the chest meets the bumper
# The duck's contact probes (their own frames, metres): the trunk's lower rear (the rear face
# spans trunk z −40…−20 mm at x −46.8 mm), and the bottom of each thigh when seated (the upper
# legs' seated floor contact, the mean of 414/454 mesh points within 3 mm of the lowest).
REAR_X_M = -0.0468
REAR_Z_M = -0.030
THIGH_M = {"left": (0.0301, 0.0417, -0.0017), "right": (-0.0301, 0.0417, -0.0017)}
PROBE_TIP_RADIUS_MM = 3
THIGH_PROBE_RADIUS_MM = 2
SENSOR_GROUP = 5  # homelayout.GROUP_HIDDEN / worldkit.SENSOR_GROUP: never a support or blocker
GROUP_CAMERA = 1  # homelayout.GROUP_CAMERA: seen by the camera, not the ToF

# MuJoCo names (the layout's elements are ``home_<id>``).
HOUSING = "home_nest_housing"
PANEL = "home_nest_panel"
PLATE = "home_nest_plate"
TAG = "home_nest_tag"
EDGE_TAG = "home_nest_edge_tag"
TAG_MOUNT = "home_nest_tag_mount"
EDGE_TAG_MOUNT = "home_nest_edge_tag_mount"
LIGHT = "home_nest_status_light"  # the camera's emitter pass picks "status_light" names
CONTACT = "home_nest_contact"  # + "_l" / "_r": the backin spring plates or the forward pads
REAR_PROBE = "duck_rear_probe"
REAR_PROBE_TIP = "duck_rear_probe_tip"
THIGH_PROBE = "duck_thigh_probe"  # + "_left" / "_right"
GUIDES = ("nest_guide_l0", "nest_guide_l1", "nest_guide_r0", "nest_guide_r1")
BUMPER_SOLREF = (0.04, 1.0)  # a soft front bumper (MuJoCo's default is 0.02)
BUMPER_SOLIMP = (0.7, 0.95, 0.01, 0.5, 2.0)
TAG_PX_PER_CELL = 16
BORDER = apriltag.TOTAL_CELLS / apriltag.BLACK_CELLS  # the white mount around a tag
# The lane and the turning room in front of the dock that nothing may stand in (the dock's
# frame, x0, x1, y0, y1): 0.9 m of clear floor straight out from the housing's face, as wide as
# the guides' mouth, and room to turn round (a forward arc of r ≈ 0.24 m sweeps ~0.4 m to one side,
# spec §10.5 / phase 1) from 0.4 m out.
LANE_MM = (0, 900, -260, 260)  # x measured from the housing's face
TURN_MM = (400, 1000, -600, 600)


@dataclass(frozen=True)
class DockShape:
    """The dock's geometry in its own frame (mm; see the module docstring). Every field can be
    overridden from the layout (``nest.shape``) for measurements."""

    depth_mm: int = 110  # housing: wall → face
    width_mm: int = 300
    height_mm: int = 190
    panel_mm: int = 10  # the face panel's protrusion (forward: the soft bumper)
    panel_width_mm: int = 200
    panel_z0_mm: int = 50  # (from the floor) the panel carries the tag's white mount exactly
    panel_z1_mm: int = 175
    plate_mm: int = 4  # 0: flush (a picture on the floor)
    ramp_mm: int = 40  # the plate's ramps on every side
    plate_length_mm: int = 240  # the plate's top, from the housing's face
    plate_width_mm: int = 260
    throat_mm: int = 260  # between the guides' inner faces, along the channel
    channel_mm: int = 120  # the guides run parallel this far out from the face…
    mouth_mm: int = 460  # …then flare to this at…
    guide_length_mm: int = 330  # …this far out
    # (dock-fall, 2026-10-11) just above the lowest of the thighs and hips they meet (z 17-120 mm), a sill the body steps over. At 80 mm
    # a duck that fell beside or in the channel lay propped on a wall it could not roll off: live, 19 % of 128 laid-down trials never
    # got up (and 2 of 10 ducks lay fallen 90 minutes); at 20 mm none of 62 stayed down, and the docking walk docked as often.
    guide_height_mm: int = 20
    guide_thick_mm: int = 12
    light_width_mm: int = 160  # the status light strip, above the panel
    light_z0_mm: int = 178
    light_z1_mm: int = 188
    tag_mm: int = 100  # the face tag's black square (its white mount: ×1.25)
    edge_tag_mm: int = 40
    # backin
    plate_travel_mm: int = 15  # the spring plates stand this far proud of the face
    plates_y0_mm: int = 4  # each plate from this far off the middle…
    plates_y1_mm: int = 70  # …to this
    plates_z0_mm: int = 8  # above the plate's top (the seated lower rear: 21–41 mm)
    plates_z1_mm: int = 46
    probe_mm: int = 15  # the rear probe's reach behind the trunk's rear
    # forward
    pads_x0_mm: int = 90  # from the housing's face
    pads_x1_mm: int = 190
    pads_y0_mm: int = 6
    pads_y1_mm: int = 100
    pad_travel_mm: int = 3  # the pads stand this far proud of the plate

    @property
    def face_mm(self) -> int:
        """Where the duck meets the dock: the front of the face panel (from the wall)."""
        return self.depth_mm + self.panel_mm

    def as_json(self) -> dict[str, int]:
        return dataclasses.asdict(self)


SHAPE_FIELDS = frozenset(f.name for f in dataclasses.fields(DockShape))


def shape(style: str, overrides: Mapping[str, Any] | None = None) -> DockShape:
    """The dock's geometry for a docking style, with the layout's overrides (integers, by
    :class:`DockShape` field name). ``backin`` has a flat face (nothing may stop the rear
    before the spring plates)."""
    if style not in STYLES:
        raise ValueError(f"充電座的停靠方式 {style!r} 不認得（{'、'.join(STYLES)}）")
    base = DockShape() if style == "forward" else DockShape(panel_mm=1)
    extra = dict(overrides or {})
    unknown = sorted(set(extra) - SHAPE_FIELDS)
    if unknown:
        raise ValueError(f"充電座沒有這些尺寸：{', '.join(unknown)}")
    for key, value in extra.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"充電座的 {key} 應該是不小於 0 的整數（mm），這裡是 {value!r}")
    return dataclasses.replace(base, **extra)


def seat_x_mm(style: str, geometry: DockShape) -> int:
    """Where a docked, seated duck's trunk is (mm from the wall along the dock's axis)."""
    if style == "forward":
        # the chest stops at the bumper (~10 mm short, measured), then the sit moves the trunk
        # back: seated trunks landed 225–257 mm from the wall (n=10, mean 239)
        return geometry.face_mm + CHEST_MM + SIT_BACK_MM + 10
    # backin: the seated rear in the middle of the plates' window — between the face and where
    # the probe just touches the plates' fronts (seated trunks 158–189 mm, n=12, mean 171)
    near = geometry.face_mm
    far = geometry.depth_mm + geometry.plate_travel_mm + geometry.probe_mm
    return (near + far) // 2 + REAR_MM


def seat_heading_mrad(style: str) -> int:
    """The docked duck's heading relative to the dock's (into the room): forward faces the dock."""
    return int(round(math.pi * 1000)) if style == "forward" else 0


def approach(style: str, geometry: DockShape) -> dict[str, int | str]:
    """The scripted docking's key places along the dock's axis (mm from the wall; for the
    measurement harness and the sanity check): where to line up (trunk, heading relative to the
    dock's), where to stop walking (the trunk), and where the seated trunk ends up."""
    seat = seat_x_mm(style, geometry)
    if style == "forward":
        return {
            "line_up_x_mm": 700,
            "line_up_heading_mrad": seat_heading_mrad(style),
            "stop_x_mm": geometry.face_mm + CHEST_MM + FORWARD_STOP_MM,
            "seat_x_mm": seat,
            "exit": "reverse",
        }
    return {
        "line_up_x_mm": 600,
        "line_up_heading_mrad": 0,
        "stop_x_mm": seat + SIT_BACK_MM + REVERSE_COAST_MM,
        "seat_x_mm": seat,
        "exit": "forward",
    }


# ── the dock in the world (plain Python: the layout loader uses these) ─────────────────


def _pose(nest: Any) -> tuple[float, float, float]:
    return float(nest.x_mm), float(nest.y_mm), float(nest.heading_mrad) / 1000.0


def to_world(nest: Any, lx: float, ly: float) -> tuple[float, float]:
    """A point of the dock's frame in the world (mm)."""
    x, y, yaw = _pose(nest)
    c, s = math.cos(yaw), math.sin(yaw)
    return x + c * lx - s * ly, y + s * lx + c * ly


def to_local(nest: Any, x_mm: float, y_mm: float) -> tuple[float, float]:
    """A world point in the dock's frame (mm)."""
    x, y, yaw = _pose(nest)
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = x_mm - x, y_mm - y
    return c * dx + s * dy, -s * dx + c * dy


def guide_segments(
    geometry: DockShape,
) -> list[tuple[str, tuple[float, float], tuple[float, float]]]:
    """Each guide wall's centre line in the dock's frame: (id, from, to)."""
    t = geometry.guide_thick_mm / 2
    out = []
    for side, tag in ((1, "l"), (-1, "r")):
        a = (float(geometry.depth_mm), side * (geometry.throat_mm / 2 + t))
        b = (float(geometry.depth_mm + geometry.channel_mm), side * (geometry.throat_mm / 2 + t))
        c = (
            float(geometry.depth_mm + geometry.guide_length_mm),
            side * (geometry.mouth_mm / 2 + t),
        )
        out.append((f"nest_guide_{tag}0", a, b))
        out.append((f"nest_guide_{tag}1", b, c))
    return out


def box_mm(geometry: DockShape) -> tuple[int, int, int, int]:
    """The dock's own footprint in its frame (x0, x1, y0, y1): housing, plate and guides."""
    half = max(
        geometry.width_mm / 2,
        geometry.mouth_mm / 2 + geometry.guide_thick_mm,
        geometry.plate_width_mm / 2 + geometry.ramp_mm,
    )
    far = max(
        geometry.depth_mm + geometry.guide_length_mm + geometry.guide_thick_mm,
        geometry.depth_mm + geometry.plate_length_mm + geometry.ramp_mm,
    )
    return (0, int(math.ceil(far)), -int(math.ceil(half)), int(math.ceil(half)))


def clearance_mm(geometry: DockShape) -> dict[str, tuple[int, int, int, int]]:
    """The dock's zones in its frame (x0, x1, y0, y1): the dock itself (``back``), and the lane
    and turning room in front of it that must stay clear (``lane``, ``turn``)."""
    face = geometry.depth_mm
    return {
        "back": box_mm(geometry),
        "lane": (face + LANE_MM[0], face + LANE_MM[1], LANE_MM[2], LANE_MM[3]),
        "turn": (face + TURN_MM[0], face + TURN_MM[1], TURN_MM[2], TURN_MM[3]),
    }


def light_point(nest: Any, geometry: DockShape) -> tuple[float, float, float]:
    """The status light's centre in the world (mm), for the lights model."""
    x, y = to_world(nest, geometry.depth_mm + 1, 0.0)
    return x, y, (geometry.light_z0_mm + geometry.light_z1_mm) / 2


def elements(nest: Any, geometry: DockShape, style: str) -> list[dict[str, Any]]:
    """The dock's passive parts as home-layout elements (world mm): housing, face panel, plate,
    guides, the two tag panels and the status light. :func:`edit_spec` adds the contacts, the
    duck's probes, the tags' faces and the soft bumper to these geoms by name."""
    heading = int(nest.heading_mrad)
    g = geometry

    def at(lx: float, ly: float) -> list[int]:
        wx, wy = to_world(nest, lx, ly)
        return [int(round(wx)), int(round(wy))]

    out: list[dict[str, Any]] = [
        {
            "id": "nest_housing",
            "zh": msg("world.dock.elements.57c792d9"),
            "kind": "dock_housing",
            "cls": "furniture",
            "shape": "box",
            "pos_mm": [*at(g.depth_mm / 2, 0), 0],
            "size_mm": [g.depth_mm, g.width_mm, g.height_mm],
            "yaw_mrad": heading,
            "colour": "#f2efe8",
        },
        {
            "id": "nest_panel",
            "zh": msg("world.dock.elements.d3aa9b93")
            if style == "forward"
            else msg("world.dock.elements.d250606d"),
            "kind": "dock_face",
            "cls": "furniture",
            "shape": "box",
            "pos_mm": [*at(g.depth_mm + g.panel_mm / 2, 0), g.panel_z0_mm],
            "size_mm": [g.panel_mm, g.panel_width_mm, g.panel_z1_mm - g.panel_z0_mm],
            "yaw_mrad": heading,
            "colour": "#12b5c4",  # cyan (C3)
        },
    ]
    top_x = g.depth_mm + g.plate_length_mm / 2
    if g.plate_mm > 0:
        out.append(
            {
                "id": "nest_plate",
                "zh": msg("world.dock.elements.43fa8684"),
                "kind": "dock_plate",
                "cls": "furniture",
                "shape": "slab",
                "pos_mm": [*at(top_x, 0), 0],
                "size_mm": [
                    g.plate_length_mm + 2 * g.ramp_mm,
                    g.plate_width_mm + 2 * g.ramp_mm,
                    g.plate_mm,
                ],
                "ramp_mm": g.ramp_mm,
                "yaw_mrad": heading,
                "colour": "#e4dfd4",
            }
        )
    else:  # flush: only a picture on the floor, the floor carries the duck
        out.append(
            {
                "id": "nest_plate",
                "zh": msg("world.dock.elements.43fa8684"),
                "kind": "floor_decor",
                "shape": "box",
                "pos_mm": [*at(top_x, 0), 0],
                "size_mm": [g.plate_length_mm, g.plate_width_mm, 1],
                "yaw_mrad": heading,
                "colour": "#e4dfd4",
                "collide": False,
                "tof": False,
            }
        )
    for guide_id, a, b in guide_segments(g):
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        if length < 1:
            continue
        flare = guide_id.endswith("1")
        angle = math.atan2(b[1] - a[1], b[0] - a[0])
        out.append(
            {
                "id": guide_id,
                "zh": msg("world.dock.elements.e2cd02b6"),
                "kind": "dock_guide",
                "cls": "furniture",
                "shape": "box",
                "pos_mm": [*at((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), 0],
                # (a flare overlaps the channel's end by its thickness: no gap at the joint)
                "size_mm": [
                    int(round(length + (g.guide_thick_mm if flare else 0))),
                    g.guide_thick_mm,
                    g.guide_height_mm,
                ],
                "yaw_mrad": int(round(heading + angle * 1000)),
                "colour": "#ddd7cb",
            }
        )
    tag_z = (g.panel_z0_mm + g.panel_z1_mm) // 2 - g.tag_mm // 2
    out += [
        {
            "id": "nest_tag",
            "zh": msg("world.dock.elements.94776399"),
            "kind": "nest_tag",
            "shape": "box",
            "pos_mm": [*at(g.face_mm + 1, 0), tag_z],
            "size_mm": [2, g.tag_mm, g.tag_mm],
            "yaw_mrad": heading,
            "colour": "#101018",
            "collide": False,
            "tof": False,
        },
        {
            "id": "nest_edge_tag",
            "zh": msg("world.dock.elements.8c4e663c"),
            "kind": "nest_tag",
            "shape": "box",
            "pos_mm": [*at(g.depth_mm + g.plate_length_mm - g.edge_tag_mm / 2 - 15, 0), g.plate_mm],
            "size_mm": [g.edge_tag_mm, g.edge_tag_mm, 1],
            "yaw_mrad": heading,
            "colour": "#101018",
            "collide": False,
            "tof": False,
        },
        {
            "id": "nest_status_light",
            "zh": msg("world.dock.elements.42236e8d"),
            "kind": "nest_led",
            "shape": "box",
            "pos_mm": [*at(g.depth_mm + 1, 0), g.light_z0_mm],
            "size_mm": [2, g.light_width_mm, g.light_z1_mm - g.light_z0_mm],
            "yaw_mrad": heading,
            "colour": "#3a3a44",  # unlit; the camera's emission pass makes it glow (C4)
            "collide": False,
            "tof": False,
        },
    ]
    return out


# ── the MuJoCo parts (sensor-only contacts, the duck's probes, the tags' faces) ────────


def edit_spec(spec: Any, nest: Any, geometry: DockShape, style: str) -> list[str]:
    """Add the contacts, the duck's probes and the tags' faces to a composed home's ``MjSpec``
    (the layout's dock elements must be in it) and make the forward bumper soft. Returns plain
    notes for anything missing."""
    import mujoco

    notes: list[str] = []
    g = mujoco.mjtGeom
    yaw = float(nest.heading_mrad) / 1000.0
    fwd = (math.cos(yaw), math.sin(yaw), 0.0)  # the dock's +x: into the room
    left = (-math.sin(yaw), math.cos(yaw), 0.0)
    up = (0.0, 0.0, 1.0)
    back = (-fwd[0], -fwd[1], 0.0)
    plate_top = geometry.plate_mm / 1000.0

    def box(name: str, x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> Any:
        geom = spec.worldbody.add_geom()
        geom.name = name
        geom.type = g.mjGEOM_BOX
        geom.size = [(x1 - x0) / 2000.0, (y1 - y0) / 2000.0, (z1 - z0) / 2000.0]
        wx, wy = to_world(nest, (x0 + x1) / 2, (y0 + y1) / 2)
        geom.pos = [wx / 1000.0, wy / 1000.0, (z0 + z1) / 2000.0]
        geom.quat = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        _sensor_look(geom, [0.78, 0.62, 0.25, 1.0], GROUP_CAMERA)  # brass
        return geom

    panel = spec.geom(PANEL)
    if panel is None:
        notes.append("這個家的充電座沒有前板（home_nest_panel）")
    elif style == "forward":
        panel.solref = list(BUMPER_SOLREF)
        panel.solimp = list(BUMPER_SOLIMP)

    # the tags' faces and their white mounts
    tag = spec.geom(TAG)
    if tag is not None:
        side = geometry.tag_mm / 1000.0
        _texture_tag(spec, tag, apriltag.NEST_TAG_ID, "board")
        tag.size = [side / 2, side / 2, 0.001]
        tag.quat = _quat_from_axes(left, up, fwd)  # reads upright from the room
        mount = spec.worldbody.add_geom()
        mount.name = TAG_MOUNT
        mount.type = g.mjGEOM_BOX
        mount.size = [side * BORDER / 2, side * BORDER / 2, 0.0005]
        mx, my = to_world(nest, geometry.face_mm + 0.5, 0.0)
        mount.pos = [mx / 1000.0, my / 1000.0, float(tag.pos[2])]
        mount.quat = _quat_from_axes(left, up, fwd)
        _sensor_look(mount, [1.0, 1.0, 1.0, 1.0], GROUP_CAMERA)
    else:
        notes.append("這個家的充電座沒有標籤面（home_nest_tag）")
    edge = spec.geom(EDGE_TAG)
    if edge is not None:
        side = geometry.edge_tag_mm / 1000.0
        _texture_tag(spec, edge, apriltag.EDGE_TAG_ID, "edge")
        edge.size = [side / 2, side / 2, 0.0005]
        edge.quat = _quat_from_axes(left, back, up)  # flat, its top toward the dock
        mount = spec.worldbody.add_geom()
        mount.name = EDGE_TAG_MOUNT
        mount.type = g.mjGEOM_BOX
        mount.size = [side * BORDER / 2, side * BORDER / 2, 0.0003]
        mount.pos = [float(edge.pos[0]), float(edge.pos[1]), plate_top + 0.0003]
        mount.quat = _quat_from_axes(left, back, up)
        _sensor_look(mount, [1.0, 1.0, 1.0, 1.0], GROUP_CAMERA)

    # the dock's contacts (visual, sensor only) and the duck's probes (massless, group 5)
    d = geometry.depth_mm
    if style == "backin":
        z0 = geometry.plate_mm + geometry.plates_z0_mm
        z1 = geometry.plate_mm + geometry.plates_z1_mm
        x1 = d + geometry.plate_travel_mm
        y0, y1 = geometry.plates_y0_mm, geometry.plates_y1_mm
        box(f"{CONTACT}_l", d, x1, y0, y1, z0, z1)
        box(f"{CONTACT}_r", d, x1, -y1, -y0, z0, z1)
        trunk = spec.body("trunk_base")
        if trunk is None:
            raise ValueError("the home has no official duck (trunk_base) for the rear probe")
        tip_x = REAR_X_M - geometry.probe_mm / 1000.0 + PROBE_TIP_RADIUS_MM / 1000.0
        shaft = trunk.add_geom()
        shaft.name = REAR_PROBE
        shaft.type = g.mjGEOM_CAPSULE
        shaft.fromto = [REAR_X_M + 0.003, 0.0, REAR_Z_M, tip_x, 0.0, REAR_Z_M]
        shaft.size = [0.0015, 0.0, 0.0]
        tip = trunk.add_geom()
        tip.name = REAR_PROBE_TIP
        tip.type = g.mjGEOM_SPHERE
        tip.size = [PROBE_TIP_RADIUS_MM / 1000.0, 0.0, 0.0]
        tip.pos = [tip_x, 0.0, REAR_Z_M]
        for geom in (shaft, tip):
            _probe(geom)
    else:
        z0, z1 = geometry.plate_mm, geometry.plate_mm + geometry.pad_travel_mm
        x0, x1 = d + geometry.pads_x0_mm, d + geometry.pads_x1_mm
        y0, y1 = geometry.pads_y0_mm, geometry.pads_y1_mm
        box(f"{CONTACT}_l", x0, x1, y0, y1, z0, z1)
        box(f"{CONTACT}_r", x0, x1, -y1, -y0, z0, z1)
        for leg, pos in THIGH_M.items():
            body = spec.body(f"upper_leg_{leg}")
            if body is None:
                raise ValueError(f"the home has no official duck (upper_leg_{leg}) for a probe")
            probe = body.add_geom()
            probe.name = f"{THIGH_PROBE}_{leg}"
            probe.type = g.mjGEOM_SPHERE
            probe.size = [THIGH_PROBE_RADIUS_MM / 1000.0, 0.0, 0.0]
            probe.pos = list(pos)
            _probe(probe)
    return notes


def _probe(geom: Any) -> None:
    geom.contype = 0
    geom.conaffinity = 0
    geom.group = SENSOR_GROUP
    geom.mass = 0.0
    geom.density = 0.0
    geom.rgba = [0.85, 0.7, 0.3, 1.0]


def _sensor_look(geom: Any, rgba: list[float], group: int) -> None:
    geom.rgba = rgba
    geom.contype = 0
    geom.conaffinity = 0
    geom.group = group


def _texture_tag(spec: Any, geom: Any, tag_id: int, where: str) -> None:
    import mujoco

    width, height, rgb = apriltag.texture_rgb(tag_id, px_per_cell=TAG_PX_PER_CELL)
    name = f"dock_tag_{where}_{tag_id}"
    texture = spec.add_texture(
        name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, width=width, height=height, nchannel=3
    )
    texture.data = rgb
    material = spec.add_material(name=f"{name}_mat", specular=0.0, shininess=0.0)
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = name
    geom.material = f"{name}_mat"
    geom.rgba = [1.0, 1.0, 1.0, 1.0]  # the texture's own colours


def _quat_from_axes(
    x: tuple[float, float, float], y: tuple[float, float, float], z: tuple[float, float, float]
) -> list[float]:
    """The orientation whose local x, y, z axes are these world vectors (orthonormal)."""
    m00, m01, m02 = x[0], y[0], z[0]
    m10, m11, m12 = x[1], y[1], z[1]
    m20, m21, m22 = x[2], y[2], z[2]
    trace = m00 + m11 + m22
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = [0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s]
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        q = [(m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s]
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        q = [(m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s]
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2
        q = [(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s]
    norm = math.sqrt(sum(v * v for v in q)) or 1.0
    return [v / norm for v in q]


# ── the contact ─────────────────────────────────────────────────────────────────────


class Contacts:
    """The dock's charging contact in a compiled world: :meth:`measure` says whether it is
    closed now and why (evaluator truth), from the duck's probes and the contact geometry."""

    def __init__(self, nest: Any, geometry: DockShape, style: str) -> None:
        self.nest = nest
        self.geometry = geometry
        self.style = style
        self.model: Any = None
        self.probes: dict[str, int] = {}
        self.panel = -1
        self.robot_bodies: frozenset[int] = frozenset()

    def ready(self, model: Any, robot_bodies: frozenset[int] = frozenset()) -> None:
        import mujoco

        def geom(name: str) -> int:
            return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))

        self.model = model
        self.robot_bodies = robot_bodies
        if self.style == "backin":
            self.probes = {"rear": geom(REAR_PROBE_TIP)}
        else:
            self.probes = {leg: geom(f"{THIGH_PROBE}_{leg}") for leg in THIGH_M}
        missing = [name for name, gid in self.probes.items() if gid < 0]
        if missing:
            raise ValueError(f"the dock's probes are missing from the compiled world: {missing}")
        self.panel = geom(PANEL)

    def measure(self, data: Any) -> tuple[bool, float | None, dict[str, Any]]:
        """(closed, the least press in metres — negative: pressed in by that much —, detail):
        the detail has each probe in the dock's frame (mm) and how far it presses."""
        if self.model is None:
            return False, None, {}
        g = self.geometry
        detail: dict[str, Any] = {}
        if self.style == "backin":
            tip = self.probes["rear"]
            px, py, pz = (float(v) * 1000.0 for v in data.geom_xpos[tip])
            lx, ly = to_local(self.nest, px, py)
            nearest = lx - PROBE_TIP_RADIUS_MM  # its point nearest the dock's face
            pressed = g.depth_mm + g.plate_travel_mm - nearest
            z0 = g.plate_mm + g.plates_z0_mm
            z1 = g.plate_mm + g.plates_z1_mm
            over = abs(ly) <= g.plates_y1_mm and z0 <= pz <= z1
            closed = pressed >= 0.0 and over
            detail["rear"] = {
                "probe_mm": [int(round(lx)), int(round(ly)), int(round(pz))],
                "pressed_mm": round(pressed, 1),
                "on_plate": over,
            }
            return closed, -pressed / 1000.0, detail
        pads: list[int] = []
        least: float | None = None
        for leg, gid in self.probes.items():
            px, py, pz = (float(v) * 1000.0 for v in data.geom_xpos[gid])
            lx, ly = to_local(self.nest, px, py)
            lowest = pz - THIGH_PROBE_RADIUS_MM
            pressed = g.plate_mm + g.pad_travel_mm - lowest
            ahead = lx - g.depth_mm
            pad = 0
            if g.pads_x0_mm <= ahead <= g.pads_x1_mm and g.pads_y0_mm <= abs(ly) <= g.pads_y1_mm:
                pad = 1 if ly > 0 else -1
            on = pad != 0 and pressed >= 0.0
            if on:
                pads.append(pad)
            least = pressed if least is None else min(least, pressed)
            detail[leg] = {
                "probe_mm": [int(round(lx)), int(round(ly)), int(round(pz))],
                "pressed_mm": round(pressed, 1),
                "pad": {1: "l", -1: "r", 0: None}[pad],
            }
        closed = len(pads) == 2 and pads[0] != pads[1]  # one thigh on each pad, either way round
        return closed, None if least is None else -least / 1000.0, detail

    def bumper(self, data: Any) -> bool:
        """Whether the duck touches the dock's face panel now (the forward bumper)."""
        if self.model is None or self.panel < 0:
            return False
        model = self.model
        for contact in data.contact[: int(data.ncon)]:
            one, two = int(contact.geom1), int(contact.geom2)
            if self.panel not in (one, two) or float(contact.dist) > 0.0:
                continue
            other = two if one == self.panel else one
            if int(model.geom_bodyid[other]) in self.robot_bodies:
                return True
        return False


# ── the page ────────────────────────────────────────────────────────────────────────


def page_json(nest: Any, geometry: DockShape, style: str) -> dict[str, Any]:
    """What the page draws of the dock beyond the layout's elements (in the dock's frame, mm;
    x into the room, z up): the two tags' cells (the physics' bits), the status light, the
    contacts (their ids match ``x_world`` → ``nest.contact_parts``), the seat and the way in."""
    g = geometry
    face_tag_z = (g.panel_z0_mm + g.panel_z1_mm) / 2
    edge_x = g.depth_mm + g.plate_length_mm - g.edge_tag_mm / 2 - 15
    if style == "backin":
        z0, z1 = g.plate_mm + g.plates_z0_mm, g.plate_mm + g.plates_z1_mm
        x0, x1 = g.depth_mm, g.depth_mm + g.plate_travel_mm
        y0, y1 = g.plates_y0_mm, g.plates_y1_mm
        what = "spring_plates"
    else:
        z0, z1 = g.plate_mm, g.plate_mm + g.pad_travel_mm
        x0, x1 = g.depth_mm + g.pads_x0_mm, g.depth_mm + g.pads_x1_mm
        y0, y1 = g.pads_y0_mm, g.pads_y1_mm
        what = "floor_pads"
    contacts = [
        {"id": "l", "kind": what, "min_mm": [x0, y0, z0], "max_mm": [x1, y1, z1]},
        {"id": "r", "kind": what, "min_mm": [x0, -y1, z0], "max_mm": [x1, -y0, z1]},
    ]
    return {
        "present": True,
        "kind": KIND,
        "style": style,
        "frame": "dock: origin at the housing's back on the wall, x into the room",
        "tags": [
            {
                **apriltag.page_json(apriltag.NEST_TAG_ID),
                "where": "board",
                "centre_mm": [g.face_mm + 2, 0, face_tag_z],
                "size_mm": g.tag_mm,
                "mount_mm": round(g.tag_mm * BORDER),
                "facing": "+x",
            },
            {
                **apriltag.page_json(apriltag.EDGE_TAG_ID),
                "where": "pad_edge",
                "centre_mm": [edge_x, 0, g.plate_mm + 1],
                "size_mm": g.edge_tag_mm,
                "mount_mm": round(g.edge_tag_mm * BORDER),
                "facing": "+z, top toward the dock",
            },
        ],
        "light": {
            "geom": LIGHT,
            "centre_mm": [g.depth_mm + 2, 0, (g.light_z0_mm + g.light_z1_mm) / 2],
            "size_mm": [g.light_width_mm, g.light_z1_mm - g.light_z0_mm],
        },
        "contacts": contacts,
        "seat": {
            "x_mm": seat_x_mm(style, g),
            "heading_mrad": seat_heading_mrad(style),
            "z_mm": g.plate_mm,
        },
        "approach": approach(style, g),
        "shape": g.as_json(),
    }


__all__ = [
    "DEFAULT_SIDE",
    "DEFAULT_STYLE",
    "KIND",
    "SIDES",
    "STYLES",
    "Contacts",
    "DockShape",
    "approach",
    "box_mm",
    "clearance_mm",
    "edit_spec",
    "elements",
    "guide_segments",
    "light_point",
    "page_json",
    "seat_heading_mrad",
    "seat_x_mm",
    "shape",
    "to_local",
    "to_world",
]
