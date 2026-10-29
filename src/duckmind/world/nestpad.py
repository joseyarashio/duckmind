"""The charging place's contact in the physics world: the dock (owner decision 2026-09-27, the
default) or phase 2's open nest (physical-design spec §10.5, §11.5; owner decisions C1–C4, B1).

**The dock.** When the layout's nest is a dock (``nest.kind`` "dock"), this part hands the
building and the contact to :mod:`duckmind.world.dock` (the contacts for its docking style, the
duck's probes, the tags' faces, the soft bumper) and reports its contact exactly as below: the
``pg`` op, ``contact_closed`` for the battery, ``x_nest`` and ``x_world`` → ``nest`` (with the dock's
``kind``, ``style``, ``seat``, each probe and whether the bumper is touched).

**The open nest** (``open_nest`` variants, for comparison).
:mod:`duckmind.world.homelayout` builds the passive nest from the layout: the collidable pad
(Ø 0.40 m; edge 5 mm, 3 mm or a ramp — C2: all three are built and measured, the owner picks),
the cyan backboard (C3), two tag panels and the charging LED. This part adds what makes it a
nest (:meth:`NestPad.edit_spec`, in memory, nothing official touched):

* **the tags' faces** — AprilTag ``tag36h11`` :data:`~duckmind.world.apriltag.NEST_TAG_ID` on
  the backboard and :data:`~duckmind.world.apriltag.EDGE_TAG_ID` flat on the pad's front edge,
  textures made from :mod:`duckmind.world.apriltag`'s bits. The layout's ``tag_mm`` is the black
  square (the tag's size); a white mount panel behind each gives its white border. The
  backboard tag reads upright to a duck in front of the nest; the edge tag reads upright to a
  duck coming in from the front and looking down (its top toward the backboard).
* **the backlight** (C4) — ``home_nest_backlight``, a 200×190 mm panel behind the tag's mount.
  It glows through the camera model's emission pass (:mod:`duckmind.world.render_scene` picks
  geoms named ``…backlight…``): on only when the nest's corner is dark, the charging colour
  while the duck charges — ``lights.home_light_state(when, home, charging=<the pg pin>)``.
* **the contact** (C1) — a disc electrode on the pad top (radius ``contacts_radius_mm``, the
  gold ring the page draws: any heading works) and a retractable spring pin under the duck's
  belly, :data:`PIN_EXTENSION_MM` below the belly's lowest point at full extension with
  :data:`PIN_TRAVEL_MM` of travel. Both are sensor-only: they collide with nothing
  (``contype`` 0) and the pin is massless, so they push nothing. The contact is closed when
  the pin tip's lowest point is at or below the electrode's top face, over the disc (the pin
  pressed in). The seated duck's belly is 17–26 mm above what it
  sits on and a standing one's about 76 mm (phase-2 survey), so it closes when the duck sits
  on the pad and opens when it stands.

**What the duck may know.** The ``pg`` op is a sensor: the nest's charger reports power-good
through the contact (spec §11.5 — "the same as a real nest's PG pin"): ``{"pg": true|false}``
and nothing else. Everything else (pin position, compression, counts) is evaluator truth in
``x_nest`` and ``x_world`` → ``nest``, never for Duck Core.

A home without a nest (``default_home,no_nest`` for 蛋黃) gets none of this, and ``pg`` reads
false, as a real duck's charge pin would with no nest to sit on.
"""

from __future__ import annotations

import math
from typing import Any

from duckmind.world import apriltag
from duckmind.world.worldkit import (
    SENSOR_GROUP,
    TRUNK,
    RobotIds,
    geom_id,
    layout_of,
    register_part,
)

ELECTRODE = "home_nest_electrode"
PIN_SHAFT = "duck_belly_pin"
PIN_TIP = "duck_belly_pin_tip"
TAG = "home_nest_tag"
EDGE_TAG = "home_nest_edge_tag"
TAG_MOUNT = "home_nest_tag_mount"
EDGE_TAG_MOUNT = "home_nest_edge_tag_mount"
BACKLIGHT = "home_nest_backlight"
STATUS_LIGHT = "home_nest_status_light"  # the dock's light (duckmind.world.dock.LIGHT)
GROUP_CAMERA = 1  # homelayout.GROUP_CAMERA: seen by the camera, not by the ToF

# The belly (trunk_base frame, metres): the lowest colliding point under the belly is 44.1 mm
# below the trunk origin near x = −20 mm (official robot_allcollisions meshes, measured
# 2026-09-27). The pin hangs there.
BELLY_X_M = -0.020
BELLY_LOW_Z_M = -0.0441
PIN_EXTENSION_MM = 25  # tip below the belly's lowest point, spring fully out
PIN_TRAVEL_MM = 15  # how far it can be pushed in
PIN_TIP_RADIUS_MM = 2
ELECTRODE_MM = 0.5  # thickness of the electrode disc on the pad
BACKLIGHT_SIZE_MM = (200, 190)  # wide, tall; centred on the tag
BORDER = apriltag.TOTAL_CELLS / apriltag.BLACK_CELLS  # white border: the mount's size
TAG_PX_PER_CELL = 16
EVERY_S = 0.02  # the contact is looked at once per robotd tick of physics time


class NestPad:
    """The nest's tags, backlight, electrode and belly pin, and its contact (see the module
    docstring). One per world."""

    name = "nestpad"

    def __init__(self, *, powered: bool = True) -> None:
        self.present = False
        self.nest: Any = None
        self.powered = powered
        self.model: Any = None
        self.robot: RobotIds | None = None
        self.tip = -1
        self.electrode = -1
        self.closed = False
        self.gap_m: float | None = None
        self.pin_local_mm: list[int] | None = None
        self.since_s: float | None = None
        self.makes = 0
        self.breaks = 0
        self._next_s = 0.0
        self._notes: list[str] = []
        self.dock: Any = None  # duckmind.world.dock.Contacts when the charging place is a dock
        self.detail: dict[str, Any] = {}  # the dock's probes at the last look

    # ── building ──────────────────────────────────────────────────────────

    def edit_spec(self, spec: Any, home: Any) -> None:
        import mujoco

        layout = layout_of(home)
        nest = getattr(layout, "nest", None) if layout is not None else None
        self.nest = nest
        self.present = nest is not None
        if nest is None:
            return
        if getattr(nest, "is_dock", False):
            from duckmind.world import dock

            self._notes += dock.edit_spec(spec, nest, nest.dock_shape, nest.style)
            self.dock = dock.Contacts(nest, nest.dock_shape, nest.style)
            return
        g = mujoco.mjtGeom
        heading = nest.heading_mrad / 1000.0
        ch, sh = math.cos(heading), math.sin(heading)

        def world(lx_mm: float, ly_mm: float) -> tuple[float, float]:
            return (
                (nest.x_mm + ch * lx_mm - sh * ly_mm) / 1000.0,
                (nest.y_mm + sh * lx_mm + ch * ly_mm) / 1000.0,
            )

        fwd = (ch, sh, 0.0)  # the nest's +x: toward the room
        left = (-sh, ch, 0.0)  # its +y
        up = (0.0, 0.0, 1.0)
        pad_top = nest.pad_height_mm / 1000.0

        # the electrode: a disc on the pad top, sensor only
        electrode = spec.worldbody.add_geom()
        electrode.name = ELECTRODE
        electrode.type = g.mjGEOM_CYLINDER
        electrode.size = [nest.contacts_radius_mm / 1000.0, ELECTRODE_MM / 2000.0, 0.0]
        cx, cy = world(0, 0)
        electrode.pos = [cx, cy, pad_top + ELECTRODE_MM / 2000.0]
        electrode.contype = 0
        electrode.conaffinity = 0
        electrode.group = GROUP_CAMERA
        electrode.rgba = [0.78, 0.62, 0.25, 1.0]  # brass

        # the tags' faces (the layout's panels) and their white mounts
        board_front_mm = -nest.board_offset_mm + nest.board_size_mm[0] / 2
        tag = spec.geom(TAG)
        if tag is not None:
            side = nest.tag_mm / 1000.0
            _texture_tag(spec, tag, apriltag.NEST_TAG_ID, "board")
            tag.size = [side / 2, side / 2, 0.001]
            # face toward the room: local x = the nest's +y (a viewer's right), y = up, z = out
            tag.quat = _quat_from_axes(left, up, fwd)
            mount = spec.worldbody.add_geom()
            mount.name = TAG_MOUNT
            mount.type = g.mjGEOM_BOX
            mount.size = [side * BORDER / 2, side * BORDER / 2, 0.0005]
            mx, my = world(board_front_mm + 0.5, 0)
            mount.pos = [mx, my, float(tag.pos[2])]
            mount.quat = _quat_from_axes(left, up, fwd)
            _plain(mount, [1.0, 1.0, 1.0, 1.0])
            backlight = spec.worldbody.add_geom()
            backlight.name = BACKLIGHT
            backlight.type = g.mjGEOM_BOX
            wide, tall = BACKLIGHT_SIZE_MM
            backlight.size = [wide / 2000.0, tall / 2000.0, 0.0005]
            bx, by = world(board_front_mm, 0)
            backlight.pos = [bx, by, float(tag.pos[2])]
            backlight.quat = _quat_from_axes(left, up, fwd)
            _plain(backlight, [0.91, 0.965, 0.97, 1.0])  # an unlit diffuser, #e8f6f7
        else:
            self._notes.append("這個家的窩沒有背板標籤面（home_nest_tag）")
        edge = spec.geom(EDGE_TAG)
        if edge is not None:
            side = nest.edge_tag_mm / 1000.0
            _texture_tag(spec, edge, apriltag.EDGE_TAG_ID, "edge")
            edge.size = [side / 2, side / 2, 0.0005]
            # flat, read looking down from the front: local x = the nest's +y, y = toward the
            # backboard (−x), z = up
            back = (-fwd[0], -fwd[1], 0.0)
            edge.quat = _quat_from_axes(left, back, up)
            mount = spec.worldbody.add_geom()
            mount.name = EDGE_TAG_MOUNT
            mount.type = g.mjGEOM_BOX
            mount.size = [side * BORDER / 2, side * BORDER / 2, 0.0003]
            mount.pos = [float(edge.pos[0]), float(edge.pos[1]), pad_top + 0.0003]
            mount.quat = _quat_from_axes(left, back, up)
            _plain(mount, [1.0, 1.0, 1.0, 1.0])

        # the belly pin on the duck (sensor only, massless)
        trunk = spec.body(TRUNK)
        if trunk is None:
            raise ValueError("the home has no official duck (trunk_base) for the belly pin")
        tip_z = BELLY_LOW_Z_M - PIN_EXTENSION_MM / 1000.0
        shaft = trunk.add_geom()
        shaft.name = PIN_SHAFT
        shaft.type = g.mjGEOM_CAPSULE
        shaft.fromto = [BELLY_X_M, 0.0, BELLY_LOW_Z_M + 0.004, BELLY_X_M, 0.0, tip_z]
        shaft.size = [0.0015, 0.0, 0.0]
        tip = trunk.add_geom()
        tip.name = PIN_TIP
        tip.type = g.mjGEOM_SPHERE
        tip.size = [PIN_TIP_RADIUS_MM / 1000.0, 0.0, 0.0]
        tip.pos = [BELLY_X_M, 0.0, tip_z + PIN_TIP_RADIUS_MM / 1000.0]
        for geom in (shaft, tip):
            geom.contype = 0
            geom.conaffinity = 0
            geom.group = SENSOR_GROUP
            geom.mass = 0.0
            geom.density = 0.0
            geom.rgba = [0.85, 0.7, 0.3, 1.0]

    def ready(self, world: Any, body: Any = None) -> None:
        self.model = world.model
        self.robot = RobotIds.of(world.model)
        if self.dock is not None:
            self.dock.ready(world.model, self.robot.bodies)
            self._look(world.data)
            return
        self.tip = geom_id(world.model, PIN_TIP)
        self.electrode = geom_id(world.model, ELECTRODE)
        if self.present and (self.tip < 0 or self.electrode < 0):
            raise ValueError("the nest's pin or electrode is missing from the compiled world")
        self._look(world.data)

    # ── the contact ───────────────────────────────────────────────────────

    def measure(self, data: Any) -> tuple[bool, float | None, list[int] | None]:
        """(closed, gap m — negative: the pin pressed in by that much —, pin tip in the nest
        frame mm) right now. The electrode is a level disc: the pin touches it when the tip's
        lowest point is at or below its top face, over the disc. A dock: its contact
        (:meth:`duckmind.world.dock.Contacts.measure`), the probe given in the dock's frame (the
        rear probe, or the left thigh's); every probe is in :attr:`detail`."""
        if self.dock is not None and self.present:
            closed, gap, detail = self.dock.measure(data)
            self.detail = detail
            first = next(iter(detail.values()), None)
            local = list(first["probe_mm"][:2]) if first else None
            return closed, gap, local
        if not self.present or self.tip < 0 or self.electrode < 0 or self.model is None:
            return False, None, None
        tip = data.geom_xpos[self.tip]
        disc = data.geom_xpos[self.electrode]
        top = float(disc[2]) + float(self.model.geom_size[self.electrode][1])
        radius = float(self.model.geom_size[self.electrode][0])
        lowest = float(tip[2]) - float(self.model.geom_size[self.tip][0])
        tx, ty = float(tip[0]), float(tip[1])
        over = math.hypot(tx - float(disc[0]), ty - float(disc[1])) <= radius
        gap = lowest - top
        heading = self.nest.heading_mrad / 1000.0
        dx, dy = tx * 1000.0 - self.nest.x_mm, ty * 1000.0 - self.nest.y_mm
        local = [
            int(round(dx * math.cos(heading) + dy * math.sin(heading))),
            int(round(-dx * math.sin(heading) + dy * math.cos(heading))),
        ]
        return over and gap <= 0.0, gap, local

    def _look(self, data: Any) -> None:
        closed, gap, local = self.measure(data)
        now = float(data.time)
        if closed and not self.closed:
            self.makes += 1
            self.since_s = now
        elif not closed and self.closed:
            self.breaks += 1
            self.since_s = now
        self.closed, self.gap_m, self.pin_local_mm = closed, gap, local

    def after_step(self, world: Any) -> None:
        if not self.present:
            return
        now = float(world.data.time)
        if now + 1e-9 < self._next_s:
            return
        self._next_s = now + EVERY_S
        self._look(world.data)

    def contact_closed(self) -> bool:
        """The charging contact as the charger sees it (the last look, ≤ 20 ms old)."""
        return bool(self.present and self.powered and self.closed)

    # ── ops ───────────────────────────────────────────────────────────────

    def ops(self) -> dict[str, Any]:
        return {"pg": self.pg_op, "x_nest": self.nest_op}

    def pg_op(self, body: Any, request: dict[str, Any]) -> dict[str, Any]:
        """The PG pin: power-good through the contact. A sensor: nothing else."""
        with body.world.lock:
            if self.present:
                self._look(body.world.data)
            return {"pg": self.contact_closed()}

    def nest_op(self, body: Any, request: dict[str, Any]) -> dict[str, Any]:
        do = str(request.get("do", "status"))
        with body.world.lock:
            if do == "power":
                on = request.get("on")
                if not isinstance(on, bool):
                    raise ValueError("x_nest power needs on: true|false")
                self.powered = on
            elif do == "page":
                return page_json(self.nest)
            elif do != "status":
                raise ValueError(f"x_nest do {do!r}: status, power or page")
            if self.present:
                self._look(body.world.data)
            return self.snapshot(body.world) or {"present": False, "pg": False}

    def snapshot(self, world: Any) -> dict[str, Any] | None:
        if not self.present:
            return None
        nest = self.nest
        pressed = None if self.gap_m is None else int(round(-self.gap_m * 1000.0))
        if self.dock is not None:
            seat = nest.seat
            return {
                "present": True,
                "kind": "dock",
                "style": nest.style,
                "pos_mm": [nest.x_mm, nest.y_mm],  # the middle of its back, on the wall
                "heading_mrad": nest.heading_mrad,  # into the room
                "seat_mm": [seat.x_mm, seat.y_mm],
                "seat_heading_mrad": seat.heading_mrad,
                "pad_top_mm": nest.pad_height_mm,  # the plate the docked duck sits on
                "contact": self.closed,
                "powered": self.powered,
                "pg": self.contact_closed(),
                "charging": self.contact_closed(),
                "pressed_mm": pressed,
                "pin_mm": self.pin_local_mm,
                "probes": {name: dict(value) for name, value in self.detail.items()},
                "bumper": self.dock.bumper(world.data) if world is not None else False,
                "since_s": None if self.since_s is None else round(self.since_s, 3),
                "makes": self.makes,
                "breaks": self.breaks,
                "tags": [
                    {"id": apriltag.NEST_TAG_ID, "geom": TAG, "size_mm": nest.dock_shape.tag_mm},
                    {
                        "id": apriltag.EDGE_TAG_ID,
                        "geom": EDGE_TAG,
                        "size_mm": nest.dock_shape.edge_tag_mm,
                    },
                ],
                "backlight": STATUS_LIGHT,  # the status light (C4: the backlight's meaning)
                "notes_zh": list(self._notes),
            }
        return {
            "present": True,
            "kind": "open",
            "edge": nest.edge,
            "pos_mm": [nest.x_mm, nest.y_mm],
            "heading_mrad": nest.heading_mrad,
            "pad_top_mm": nest.pad_height_mm,
            "contact": self.closed,
            "powered": self.powered,
            "pg": self.contact_closed(),
            "charging": self.contact_closed(),
            "pressed_mm": pressed,
            "pin_mm": self.pin_local_mm,
            "since_s": None if self.since_s is None else round(self.since_s, 3),
            "makes": self.makes,
            "breaks": self.breaks,
            "tags": [
                {"id": apriltag.NEST_TAG_ID, "geom": TAG, "size_mm": nest.tag_mm},
                {"id": apriltag.EDGE_TAG_ID, "geom": EDGE_TAG, "size_mm": nest.edge_tag_mm},
            ],
            "backlight": BACKLIGHT,
            "notes_zh": list(self._notes),
        }


def page_json(nest: Any) -> dict[str, Any]:
    """What the page draws of the nest beyond the layout's panels: the two tags' cells (the
    same bits as the physics), the white mounts, the backlight panel and the electrode, in the
    nest's frame (mm; x toward the room, z up). A dock: :func:`duckmind.world.dock.page_json`
    (its tags, status light, contacts, seat and way in, in the dock's frame)."""
    if nest is None:
        return {"present": False}
    if getattr(nest, "is_dock", False):
        from duckmind.world import dock

        return dock.page_json(nest, nest.dock_shape, nest.style)
    front = -nest.board_offset_mm + nest.board_size_mm[0] / 2
    edge_x = nest.pad_radius_mm - nest.edge_tag_mm // 2 - 10
    return {
        "present": True,
        "kind": "open",
        "tags": [
            {
                **apriltag.page_json(apriltag.NEST_TAG_ID),
                "where": "board",
                "centre_mm": [front + 2, 0, 120],
                "size_mm": nest.tag_mm,
                "mount_mm": round(nest.tag_mm * BORDER),
                "facing": "+x",
            },
            {
                **apriltag.page_json(apriltag.EDGE_TAG_ID),
                "where": "pad_edge",
                "centre_mm": [edge_x, 0, nest.pad_height_mm + 1],
                "size_mm": nest.edge_tag_mm,
                "mount_mm": round(nest.edge_tag_mm * BORDER),
                "facing": "+z, top toward the board",
            },
        ],
        "backlight": {
            "geom": BACKLIGHT,
            "centre_mm": [front, 0, 120],
            "size_mm": list(BACKLIGHT_SIZE_MM),
        },
        "electrode": {"radius_mm": nest.contacts_radius_mm, "top_mm": nest.pad_height_mm},
        "pin": {
            "extension_mm": PIN_EXTENSION_MM,
            "travel_mm": PIN_TRAVEL_MM,
            "belly_mm": [round(BELLY_X_M * 1000), 0, round(BELLY_LOW_Z_M * 1000, 1)],
        },
    }


def _plain(geom: Any, rgba: list[float]) -> None:
    geom.rgba = rgba
    geom.contype = 0
    geom.conaffinity = 0
    geom.group = GROUP_CAMERA


def _texture_tag(spec: Any, geom: Any, tag_id: int, where: str) -> None:
    import mujoco

    width, height, rgb = apriltag.texture_rgb(tag_id, px_per_cell=TAG_PX_PER_CELL)
    name = f"nest_tag_{where}_{tag_id}"
    texture = spec.add_texture(
        name=name,
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        width=width,
        height=height,
        nchannel=3,
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


def register(registry: Any) -> NestPad:
    """The nest alone as a simhome extension (``--ext duckmind.world.nestpad``)."""
    part = NestPad()
    register_part(registry, part, section="nest")
    return part


__all__ = ["NestPad", "page_json", "register"]
