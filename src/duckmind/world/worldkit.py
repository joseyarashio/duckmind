"""The world's household in one extension: the nest, the hand, the family's bodies, the battery
and the things (physical-design spec §10.5, §10.7, §11.5; phase 2; owner decisions B1–E2).

Load it into the world server (:mod:`duckmind.world.simhome`) with
``--ext duckmind.world.worldkit`` or ``DUCKMIND_SIMHOME_EXT=duckmind.world.worldkit``::

    register(registry)  # builds every part, wires them together, registers each

Each part is also its own extension (``--ext duckmind.world.hand`` …) for a test or a
measurement, but only this module wires them to each other: the battery charges when the
nest's contact closes, the hand sets the duck on the nest and never on a person, people push
the things, the things' impacts include the duck's own.

The parts (each a class with the same small interface, :class:`Part`):

========================= ======================= =====================================
part                       ops                     ``x_world`` section
========================= ======================= =====================================
:mod:`~.nestpad`           ``pg`` (sensor), x_nest nest
:mod:`~.hand`              x_hand                  hand
:mod:`~.people`            x_people                people
:mod:`~.battery`           x_battery               battery
:mod:`~.props`             x_props, x_place_prop,  things
                           x_tidy, x_roll_ball,
                           x_impacts
========================= ======================= =====================================

**The same bodies in the eye's copy.** The head camera renders its own compiled copy of the
home (:mod:`duckmind.world.render_scene`), and a world snapshot's ``qpos`` and mocap poses
only fit it if it has the same bodies in the same order. :func:`edit_spec` makes exactly the
edits the world server makes (the nest's tags, electrode, pin and backlight; the hand; the
people), deterministically, so the eye calls it on its own ``homelayout.compose(...)`` spec
before compiling. Its ``MjSpec`` holds a texture from a buffer: ``MjSpec.to_xml()`` refuses
those, so copy a spec by composing and editing it again, not through XML.

Shared helpers for the parts (:class:`RobotIds`, surfaces under a point, the free-spot probe,
quaternions) live here too. Nothing heavy is imported at import time.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

TRUNK = "trunk_base"
TRUNK_JOINT = "trunk_base_freejoint"
SENSOR_GROUP = 5  # homelayout.GROUP_HIDDEN, simhome.SENSOR_GROUP: never a support or a blocker
SPOT_PROBE = "kit_spot_probe"
SPOT_PROBE_RADIUS_M = 0.12  # the sitting duck's footprint (colliders: x −57…102, y ±93 mm)
SPOT_PROBE_HALF_M = 0.12
SPOT_CLEAR_M = 0.02  # the probe starts this far above the surface
DUCK_TOP_M = 0.16  # a sitting (or held, legs tucked) duck's top above its trunk (SIT: 158 mm)
SLIDE_STEP_MAX_M = (
    0.025  # the highest step (a rug's edge, a door strip) a hand lifts a sliding duck over
)
SLIDE_GRID_M = 0.05  # the grid the route of a slide is searched on
SLIDE_REACH_M = 1.2  # a duck is slid at most this far (a coffee table's half length is 0.72 m)
KEEP_OFF_DROP_MM = 140  # homelayout.DUCK_RADIUS_MM: a stand-in spot keeps this far off a drop
PARK_Z_M = -3.0  # where unused mocap bodies wait, under the floor (never drawn, never seen)


class Part(Protocol):
    """What a household part offers the registry (every method optional in practice)."""

    name: str

    def edit_spec(self, spec: Any, home: Any) -> None: ...

    def ready(self, world: Any, body: Any) -> None: ...

    def ops(self) -> dict[str, Callable[[Any, dict[str, Any]], dict[str, Any]]]: ...

    def snapshot(self, world: Any) -> Any: ...


def layout_of(home: Any) -> Any:
    """The home layout inside simhome's HomeInfo (or a layout itself, or None)."""
    if home is None:
        return None
    layout = getattr(home, "layout", None)
    if layout is not None:
        return layout
    return home if hasattr(home, "elements") and hasattr(home, "things") else None


def register_part(registry: Any, part: Any, *, section: str | None = None) -> None:
    """Register a part's callbacks with a simhome :class:`~duckmind.world.simhome.Registry`."""
    name = getattr(part, "name", type(part).__name__.lower())
    if hasattr(part, "edit_spec"):
        registry.edit_spec(part.edit_spec, name=f"{name}.edit_spec")
    if hasattr(part, "ready"):
        registry.on_ready(part.ready, name=f"{name}.ready")
    if hasattr(part, "before_step"):
        registry.before_step(part.before_step, name=f"{name}.before_step")
    if hasattr(part, "after_step"):
        registry.after_step(part.after_step, name=f"{name}.after_step")
    for op, fn in part.ops().items() if hasattr(part, "ops") else ():
        registry.op(op, fn)
    if section is not None and hasattr(part, "snapshot"):
        registry.snapshot(section, part.snapshot)
    if hasattr(part, "stop"):
        registry.on_stop(part.stop, name=f"{name}.stop")


# ── the household, wired ────────────────────────────────────────────────────────


@dataclass
class Kit:
    """Every part, wired to the others."""

    nest: Any
    hand: Any
    people: Any
    battery: Any
    props: Any
    parts: list[Any] = field(default_factory=list)

    def edit_spec(self, spec: Any, home: Any) -> None:
        """The world server's edits, in the world server's order (see the module docstring)."""
        add_spot_probe(spec)
        for part in self.parts:
            part.edit_spec(spec, home)


def build() -> Kit:
    from duckmind.world import battery, hand, nestpad, people, props

    nest = nestpad.NestPad()
    things = props.Props()
    folks = people.People()
    arm = hand.Hand(nest=nest, people=folks)
    pack = battery.WorldBattery(contact=nest.contact_closed)
    kit = Kit(nest=nest, hand=arm, people=folks, battery=pack, props=things)
    kit.parts = [nest, arm, folks, pack, things]
    return kit


def register(registry: Any) -> Kit:
    """The whole household as one simhome extension."""
    kit = build()
    registry.edit_spec(lambda spec, home: add_spot_probe(spec), name="worldkit.spot_probe")
    sections = {
        "nest": "nest",
        "hand": "hand",
        "people": "people",
        "battery": "battery",
        "props": "things",
    }
    for key, part in zip(("nest", "hand", "people", "battery", "props"), kit.parts, strict=True):
        register_part(registry, part, section=sections[key])
    registry.on_ready(lambda world, body: kit.people.start_thread(), name="people.planner")
    return kit


def edit_spec(spec: Any, home: Any) -> Kit:
    """Make the world server's household edits on another spec of the same home (the eye's
    render copy): returns the (unused) kit that made them."""
    kit = build()
    kit.edit_spec(spec, home)
    return kit


# ── shapes the world adds and the page draws ────────────────────────────────────


@dataclass(frozen=True)
class Prim:
    """One primitive of a body the world adds (the hand, a person), in the body's frame (m):
    MuJoCo's own sizes (half extents for a box or an ellipsoid, the radius for a sphere or a
    capsule, whose axis is ``fromto``)."""

    name: str
    shape: str  # box | sphere | capsule | ellipsoid
    rgb: tuple[float, float, float]
    size: tuple[float, ...]
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fromto: tuple[float, ...] = ()
    collide: bool = False


def add_prims(body: Any, prims: Iterable[Prim], *, group: int = 0) -> None:
    """Add these primitives to an ``MjSpec`` body (colliding: contype/conaffinity 1)."""
    mujoco = _mujoco()
    kinds = {
        "box": mujoco.mjtGeom.mjGEOM_BOX,
        "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
        "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
        "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
    }
    for prim in prims:
        geom = body.add_geom()
        geom.name = prim.name
        geom.type = kinds[prim.shape]
        geom.size = [*prim.size, *([0.0] * (3 - len(prim.size)))]
        if prim.fromto:
            geom.fromto = list(prim.fromto)
        else:
            geom.pos = list(prim.pos)
        geom.rgba = [*prim.rgb, 1.0]
        bits = 1 if prim.collide else 0
        geom.contype = bits
        geom.conaffinity = bits
        geom.group = group


def prims_json(prims: Iterable[Prim]) -> list[dict[str, Any]]:
    """The primitives for the page, in millimetres and ``#rrggbb`` (three.js: draw them in
    the body's frame, then place the frame at the mocap pose ``x_world`` gives)."""
    out = []
    for prim in prims:
        item: dict[str, Any] = {
            "name": prim.name,
            "shape": prim.shape,
            "size_mm": [round(v * 1000, 1) for v in prim.size],
            "colour": "#%02x%02x%02x" % tuple(int(round(c * 255)) for c in prim.rgb),
            "collide": prim.collide,
        }
        if prim.fromto:
            item["fromto_mm"] = [round(v * 1000, 1) for v in prim.fromto]
        else:
            item["pos_mm"] = [round(v * 1000, 1) for v in prim.pos]
        out.append(item)
    return out


# ── the robot in a model ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RobotIds:
    """Where the duck is in a compiled model."""

    trunk: int  # body id of trunk_base
    qadr: int  # its free joint's qpos address
    dof: int  # and its dof address
    bodies: frozenset[int]
    colliders: tuple[int, ...]  # the duck's colliding geoms (not sensor helpers)

    @classmethod
    def of(cls, model: Any) -> RobotIds:
        mujoco = _mujoco()
        trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TRUNK)
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, TRUNK_JOINT)
        if trunk < 0 or joint < 0:
            raise ValueError("the model has no official duck (trunk_base)")
        root = int(model.body_rootid[trunk])
        bodies = frozenset(b for b in range(int(model.nbody)) if int(model.body_rootid[b]) == root)
        colliders = tuple(
            g
            for g in range(int(model.ngeom))
            if int(model.geom_bodyid[g]) in bodies
            and (int(model.geom_contype[g]) or int(model.geom_conaffinity[g]))
            and int(model.geom_group[g]) != SENSOR_GROUP
        )
        return cls(
            trunk=int(trunk),
            qadr=int(model.jnt_qposadr[joint]),
            dof=int(model.jnt_dofadr[joint]),
            bodies=bodies,
            colliders=colliders,
        )


def _mujoco() -> Any:
    import mujoco

    return mujoco


def site_id(model: Any, name: str) -> int:
    mujoco = _mujoco()
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name))


def geom_id(model: Any, name: str) -> int:
    mujoco = _mujoco()
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))


def body_id(model: Any, name: str) -> int:
    mujoco = _mujoco()
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))


def geom_name(model: Any, geom: int) -> str:
    mujoco = _mujoco()
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
    if name:
        return str(name)
    owner = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom]))
    return f"{owner or 'body'}/geom{geom}"


def static_supports(model: Any) -> list[int]:
    """The static colliders a duck or a thing can rest on (as simhome finds them)."""
    return [
        g
        for g in range(int(model.ngeom))
        if (int(model.geom_contype[g]) or int(model.geom_conaffinity[g]))
        and int(model.geom_group[g]) != SENSOR_GROUP
        and int(model.body_weldid[int(model.geom_bodyid[g])]) == 0
        and int(model.body_mocapid[int(model.geom_bodyid[g])]) < 0
    ]


def movable_colliders(model: Any, robot: RobotIds, extra_skip: Iterable[int] = ()) -> list[int]:
    """Colliding geoms that are not static and not the duck: things, people, a hand."""
    skip = set(extra_skip)
    out = []
    for g in range(int(model.ngeom)):
        body = int(model.geom_bodyid[g])
        if g in skip or body in robot.bodies:
            continue
        if not (int(model.geom_contype[g]) or int(model.geom_conaffinity[g])):
            continue
        if int(model.geom_group[g]) == SENSOR_GROUP:
            continue
        if int(model.body_weldid[body]) == 0 and int(model.body_mocapid[body]) < 0:
            continue
        out.append(g)
    return out


def vertical_ray_may_hit(model: Any, data: Any, geom: int, x: float, y: float) -> bool:
    """Whether a ray along the vertical through (x, y) can touch this geom at all: its bounding
    sphere (``geom_rbound``; 0 for a plane, which has none) reaches the ray. The cheap test before
    a ray is cast at every collider of the home."""
    bound = float(model.geom_rbound[geom])
    if bound <= 0.0:
        return True
    centre = data.geom_xpos[geom]
    return math.hypot(float(centre[0]) - x, float(centre[1]) - y) <= bound


def surface_at(
    model: Any, data: Any, supports: Sequence[int], x: float, y: float, top: float
) -> tuple[float, int] | None:
    """The highest static collider under (x, y) below ``top``: (its z, the geom), or None
    over a hole. A probe that starts inside a collider does not count it."""
    import numpy as np

    mujoco = _mujoco()
    point = np.array([x, y, top], dtype=float)
    down = np.array([0.0, 0.0, -1.0])
    up = np.array([0.0, 0.0, 1.0])
    best: tuple[float, int] | None = None
    for geom in supports:
        if not vertical_ray_may_hit(model, data, geom, x, y):
            continue
        kind = int(model.geom_type[geom])
        if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
            dist = mujoco.mj_rayMesh(model, data, geom, point, down)
            inside = kind != int(mujoco.mjtGeom.mjGEOM_PLANE) and (
                dist >= 0.0 and mujoco.mj_rayMesh(model, data, geom, point, up) >= 0.0
            )
        elif kind in (int(mujoco.mjtGeom.mjGEOM_HFIELD), int(mujoco.mjtGeom.mjGEOM_SDF)):
            continue
        else:
            args = (data.geom_xpos[geom], data.geom_xmat[geom], model.geom_size[geom])
            dist = mujoco.mju_rayGeom(*args, point, down, kind)
            inside = kind != int(mujoco.mjtGeom.mjGEOM_PLANE) and (
                dist >= 0.0 and mujoco.mju_rayGeom(*args, point, up, kind) >= 0.0
            )
        if dist < 0.0 or inside:
            continue
        z = top - float(dist)
        if best is None or z > best[0]:
            best = (z, geom)
    return best


def add_spot_probe(spec: Any) -> None:
    """A hidden cylinder the size of a sitting duck, for :class:`SpotCheck` (once per spec)."""
    mujoco = _mujoco()
    if spec.geom(SPOT_PROBE) is not None:
        return
    geom = spec.worldbody.add_geom()
    geom.name = SPOT_PROBE
    geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    geom.size = [SPOT_PROBE_RADIUS_M, SPOT_PROBE_HALF_M, 0.0]
    geom.pos = [0.0, 0.0, PARK_Z_M]
    geom.contype = 0
    geom.conaffinity = 0
    geom.group = SENSOR_GROUP
    geom.rgba = [0.0, 0.0, 0.0, 0.0]


@dataclass(frozen=True)
class Slide:
    """A way to slide a sitting duck along the floor (:meth:`SpotCheck.slide_out`): ``route`` from
    where it sits to the spot, ``surface`` the level surface at the spot, ``top`` the highest
    surface under the duck anywhere along it (the duck is carried over that)."""

    route: list[tuple[float, float]]
    surface: float
    top: float


class SpotCheck:
    """Is there room for a sitting duck at (x, y)? A hidden probe geom is moved there in the
    data only (its geom_xpos; the next step recomputes it) and its distance to every nearby
    collider is asked. With the home's ``layout`` a spot must also be indoors, off a drop and
    not behind a wall from the spot asked for (:meth:`homelayout.Layout.place_problem`: the
    official floor plane runs on past the walls). :meth:`headroom` says how much room there is
    above a spot (a table top over it). Call with the world's lock held."""

    def __init__(
        self,
        model: Any,
        robot: RobotIds,
        *,
        skip_bodies: Iterable[int] = (),
        layout: Any = None,
    ) -> None:
        self.model = model
        self.robot = robot
        self.layout = layout
        self.probe = geom_id(model, SPOT_PROBE)
        skipped = set(skip_bodies)
        self.supports = static_supports(model)
        plane = int(_mujoco().mjtGeom.mjGEOM_PLANE)
        self.others = [
            g
            for g in self.supports + movable_colliders(model, robot)
            if int(model.geom_bodyid[g]) not in skipped and int(model.geom_type[g]) != plane
        ]
        self.overhead = [g for g in self.supports if int(model.geom_type[g]) != plane]

    def surface(self, data: Any, x: float, y: float, top: float = 0.25) -> float | None:
        hit = surface_at(self.model, data, self.supports, x, y, top)
        return None if hit is None else hit[0]

    def span(self, data: Any, x: float, y: float, top: float = 0.25) -> tuple[float, float] | None:
        """The lowest and the highest surface under the footprint at (x, y), or None when part of
        it is over a hole."""
        r = SPOT_PROBE_RADIUS_M * 0.9
        heights = [
            self.surface(data, x + dx, y + dy, top)
            for dx, dy in ((0.0, 0.0), (r, 0.0), (-r, 0.0), (0.0, r), (0.0, -r))
        ]
        values = [h for h in heights if h is not None]
        if len(values) < len(heights):
            return None
        return min(values), max(values)

    def level(self, data: Any, x: float, y: float, top: float = 0.25) -> float | None:
        """The surface under the whole footprint when it is flat (within 4 mm), else None."""
        span = self.span(data, x, y, top)
        if span is None or span[1] - span[0] > 0.004:
            return None
        return span[1]

    def blockers(self, data: Any, x: float, y: float, surface: float) -> list[int]:
        """The colliders a sitting duck at (x, y) on ``surface`` would be inside."""
        import numpy as np

        mujoco = _mujoco()
        if self.probe < 0:
            return []
        z = surface + SPOT_CLEAR_M + SPOT_PROBE_HALF_M
        saved = (data.geom_xpos[self.probe].copy(), data.geom_xmat[self.probe].copy())
        data.geom_xpos[self.probe] = [x, y, z]
        data.geom_xmat[self.probe] = [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]
        fromto = np.zeros(6)
        reach = SPOT_PROBE_RADIUS_M + 0.02
        found = []
        try:
            for g in self.others:
                gx, gy = float(data.geom_xpos[g][0]), float(data.geom_xpos[g][1])
                if math.hypot(gx - x, gy - y) > reach + float(self.model.geom_rbound[g]):
                    continue
                if mujoco.mj_geomDistance(self.model, data, self.probe, g, 0.01, fromto) < 0.0:
                    found.append(g)
        finally:
            data.geom_xpos[self.probe], data.geom_xmat[self.probe] = saved
        return found

    def headroom(self, data: Any, x: float, y: float, surface: float) -> tuple[float, int] | None:
        """The lowest static collider over a sitting duck's footprint at (x, y) — rays up from
        just above ``surface`` — as (its underside's height above the surface, the geom), or
        None with nothing overhead (open to the ceiling, or no ceiling)."""
        import numpy as np

        mujoco = _mujoco()
        r = SPOT_PROBE_RADIUS_M * 0.9
        z0 = surface + SPOT_CLEAR_M
        up = np.array([0.0, 0.0, 1.0])
        down = np.array([0.0, 0.0, -1.0])
        best: tuple[float, int] | None = None
        for dx, dy in ((0.0, 0.0), (r, 0.0), (-r, 0.0), (0.0, r), (0.0, -r)):
            point = np.array([x + dx, y + dy, z0], dtype=float)
            for g in self.overhead:
                if not vertical_ray_may_hit(self.model, data, g, x + dx, y + dy):
                    continue
                kind = int(self.model.geom_type[g])
                if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
                    dist = mujoco.mj_rayMesh(self.model, data, g, point, up)
                    below = mujoco.mj_rayMesh(self.model, data, g, point, down) if dist >= 0 else -1
                elif kind in (int(mujoco.mjtGeom.mjGEOM_HFIELD), int(mujoco.mjtGeom.mjGEOM_SDF)):
                    continue
                else:
                    args = (data.geom_xpos[g], data.geom_xmat[g], self.model.geom_size[g])
                    dist = mujoco.mju_rayGeom(*args, point, up, kind)
                    below = mujoco.mju_rayGeom(*args, point, down, kind) if dist >= 0 else -1
                if dist < 0.0 or below >= 0.0:
                    continue  # nothing above, or the point is inside it (a wall's foot beside)
                height = SPOT_CLEAR_M + float(dist)
                if best is None or height < best[0]:
                    best = (height, g)
        return best

    def problem(self, x: float, y: float, asked: tuple[float, float] | None = None) -> str | None:
        """Why the home refuses (x, y) for a duck (plain words; None: the physics decides):
        outside its rooms, over a drop, or behind a wall from ``asked``. No layout: None."""
        check = getattr(self.layout, "place_problem", None)
        if not callable(check):
            return None
        near = None if asked is None else (asked[0] * 1000.0, asked[1] * 1000.0)
        problem = check(
            x * 1000.0,
            y * 1000.0,
            from_mm=near,
            drop_margin_mm=KEEP_OFF_DROP_MM if near is not None else 0.0,
        )
        return str(problem) if problem else None

    def slide_out(
        self,
        data: Any,
        x: float,
        y: float,
        *,
        headroom_m: float,
        reach_m: float = SLIDE_REACH_M,
        step_m: float = SLIDE_GRID_M,
        skip_m: float = 0.12,
    ) -> Slide | None:
        """The shortest way to slide a sitting duck along the floor from (x, y) to the nearest
        level spot with ``headroom_m`` of room over it, or None. Dijkstra on a ``step_m`` grid
        within ``reach_m``: a cell is open when it is indoors and not behind a wall from (x, y),
        has a surface under the whole footprint that is no more than :data:`SLIDE_STEP_MAX_M`
        above the one under the duck (a rug's edge is lifted over, a sofa's seat is not) and
        has nothing in it (the first ``skip_m`` — where it sits now — is not tested for that).
        The route is straightened into as few legs as it can be."""
        start = (x, y)
        under = self.span(data, x, y)
        base = under[1] if under is not None else (self.surface(data, x, y) or 0.0)
        limit = base + SLIDE_STEP_MAX_M

        def probe(px: float, py: float, away: float) -> tuple[float, float | None] | None:
            """(the highest surface under the footprint, the surface when it is level), or None
            when a duck cannot be carried over this spot."""
            if away > 0.0 and self.problem(px, py, start) is not None:
                return None
            span = self.span(data, px, py)
            if span is None or span[1] > limit or span[1] - span[0] > SLIDE_STEP_MAX_M:
                return None
            if away >= skip_m and self.blockers(data, px, py, span[1]):
                return None
            return span[1], (span[1] if span[1] - span[0] <= 0.004 else None)

        cells: dict[tuple[int, int], tuple[float, float | None] | None] = {}

        def cell(i: int, j: int) -> tuple[float, float | None] | None:
            if (i, j) not in cells:
                cells[i, j] = probe(x + i * step_m, y + j * step_m, math.hypot(i, j) * step_m)
            return cells[i, j]

        if cell(0, 0) is None:
            return None
        best = {(0, 0): 0.0}
        parent: dict[tuple[int, int], tuple[int, int]] = {}
        done: set[tuple[int, int]] = set()
        heap = [(0.0, 0, 0)]
        goal: tuple[int, int] | None = None
        while heap and goal is None:
            dist, i, j = heapq.heappop(heap)
            if (i, j) in done:
                continue
            done.add((i, j))
            here = cells[i, j]
            assert here is not None
            if here[1] is not None:
                above = self.headroom(data, x + i * step_m, y + j * step_m, here[1])
                if above is None or above[0] >= headroom_m:
                    goal = (i, j)
                    break
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    key = (i + di, j + dj)
                    long = dist + math.hypot(di, dj) * step_m
                    if (di, dj) == (0, 0) or key in done or long > reach_m:
                        continue
                    if cell(*key) is None or long >= best.get(key, math.inf):
                        continue
                    best[key] = long
                    parent[key] = (i, j)
                    heapq.heappush(heap, (long, *key))
        if goal is None:
            return None
        nodes = [goal]
        while nodes[-1] != (0, 0):
            nodes.append(parent[nodes[-1]])
        nodes.reverse()
        # (the corners of the grid route: a run in one direction is one leg)
        corners = [nodes[0]]
        for before, here, after in zip(nodes, nodes[1:], nodes[2:], strict=False):
            if (here[0] - before[0], here[1] - before[1]) != (
                after[0] - here[0],
                after[1] - here[1],
            ):
                corners.append(here)
        if len(nodes) > 1:
            corners.append(nodes[-1])

        def spot(node: tuple[int, int]) -> tuple[float, float]:
            return x + node[0] * step_m, y + node[1] * step_m

        def open_line(a: tuple[float, float], b: tuple[float, float]) -> bool:
            samples = max(1, int(math.dist(a, b) / 0.04))
            for k in range(1, samples):
                f = k / samples
                px, py = a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f
                if probe(px, py, math.hypot(px - x, py - y)) is None:
                    return False
            return True

        # (and a corner is skipped when the straight line past it is as open as the way round)
        route = [spot(corners[0])]
        at = 0
        while at < len(corners) - 1:
            to = len(corners) - 1
            while to > at + 1 and not open_line(spot(corners[at]), spot(corners[to])):
                to -= 1
            route.append(spot(corners[to]))
            at = to
        top = max(cells[node][0] for node in nodes)  # type: ignore[index]
        landing = cells[goal]
        assert landing is not None and landing[1] is not None
        return Slide(route, landing[1], top)

    def free(
        self,
        data: Any,
        x: float,
        y: float,
        *,
        reach_m: float = 0.8,
        step_m: float = 0.04,
        headroom_m: float | None = None,
    ) -> tuple[float, float, float, list[str]] | None:
        """The nearest free, level spot to (x, y) within ``reach_m`` (a spiral of rings):
        (x, y, surface, [what was in the way at the asked spot]) or None. With a layout, only
        spots indoors, off drops and not behind a wall from (x, y); with ``headroom_m``, only
        spots with that much room above them (nothing overhead lower: a table top)."""
        in_way: list[str] = []
        rings = max(0, int(reach_m / step_m))
        for ring in range(rings + 1):
            radius = ring * step_m
            count = 1 if ring == 0 else max(6, int(2 * math.pi * radius / step_m))
            for k in range(count):
                angle = 2 * math.pi * k / count
                px, py = x + radius * math.cos(angle), y + radius * math.sin(angle)
                if self.problem(px, py, None if ring == 0 else (x, y)) is not None:
                    continue
                surface = self.level(data, px, py)
                if surface is None:
                    continue
                blocked = self.blockers(data, px, py, surface)
                if ring == 0 and blocked:
                    in_way = sorted({geom_name(self.model, g) for g in blocked})
                if blocked:
                    continue
                if headroom_m is not None:
                    above = self.headroom(data, px, py, surface)
                    if above is not None and above[0] < headroom_m:
                        if ring == 0:
                            in_way = [geom_name(self.model, above[1])]
                        continue
                return px, py, surface, in_way
        return None


# ── maths ───────────────────────────────────────────────────────────────────────


def quat_mul(a: Sequence[float], b: Sequence[float]) -> list[float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def yaw_quat(yaw: float) -> list[float]:
    """Upright, facing ``yaw`` radians about the world's z."""
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def heading_of(quat: Sequence[float]) -> float:
    """The twist about the vertical in an orientation (the heading a fallen duck still has)."""
    w, _, _, z = (float(v) for v in quat[:4])
    if abs(w) < 1e-9 and abs(z) < 1e-9:
        x, y = float(quat[1]), float(quat[2])
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return wrap(2 * math.atan2(z, w))


def tilt_of(quat: Sequence[float]) -> float:
    """The angle between the body's z axis and the world's up (radians)."""
    w, x, y, z = (float(v) for v in quat[:4])
    norm = w * w + x * x + y * y + z * z or 1.0
    up = 1.0 - 2.0 * (x * x + y * y) / norm
    return math.acos(max(-1.0, min(1.0, up)))


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def slerp(a: Sequence[float], b: Sequence[float], s: float) -> list[float]:
    qa = [float(v) for v in a]
    qb = [float(v) for v in b]
    dot = sum(x * y for x, y in zip(qa, qb, strict=True))
    if dot < 0.0:
        qb, dot = [-v for v in qb], -dot
    if dot > 0.9995:
        out = [x + s * (y - x) for x, y in zip(qa, qb, strict=True)]
    else:
        theta = math.acos(max(-1.0, min(1.0, dot)))
        sa = math.sin((1 - s) * theta) / math.sin(theta)
        sb = math.sin(s * theta) / math.sin(theta)
        out = [sa * x + sb * y for x, y in zip(qa, qb, strict=True)]
    norm = math.sqrt(sum(v * v for v in out)) or 1.0
    return [v / norm for v in out]


def rotate(quat: Sequence[float], vec: Sequence[float]) -> list[float]:
    """A vector turned by a quaternion."""
    w, x, y, z = (float(v) for v in quat)
    vx, vy, vz = (float(v) for v in vec)
    # t = 2 q_vec × v; v' = v + w t + q_vec × t
    tx, ty, tz = 2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx)
    return [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]


def min_jerk(tau: float) -> float:
    """0 → 1 with zero speed and acceleration at both ends."""
    t = max(0.0, min(1.0, tau))
    return t * t * t * (10 - 15 * t + 6 * t * t)


def mm(values: Iterable[float]) -> list[int]:
    return [int(round(float(v) * 1000.0)) for v in values]


def mrad(value: float) -> int:
    return int(round(float(value) * 1000.0))


# ── a home stepped here, without robotd (tests, measurements) ──────────────────


class Headless:
    """A home with the whole household, stepped in this process without robotd: the duck's
    actuators hold whatever ``data.ctrl`` says (a keyframe's pose unless a controller writes
    it). The same model, hooks and order as the world server (spot probe, nest, hand, people,
    battery, things), for tests and measurements; 5 ms steps."""

    def __init__(
        self,
        home: str = "default_home",
        *,
        mjcf_dir: str | None = None,
        keyframe: str = "SIT",
        collision: str = "allcollisions",
    ) -> None:
        import threading
        import types

        from duckmind.world import homelayout

        mujoco = _mujoco()
        self.layout = homelayout.load(home, mjcf_dir=mjcf_dir)
        spec = homelayout.compose(self.layout, collision, mjcf_dir=mjcf_dir)
        self.kit = build()
        self.kit.edit_spec(spec, self.layout)
        homelayout.fix_keyframes(spec, self.layout)
        self.spec = spec
        self.model = spec.compile()
        self.model.opt.timestep = 0.005  # body_server.TIMESTEP
        self.data = mujoco.MjData(self.model)
        self.world = types.SimpleNamespace(
            model=self.model, data=self.data, lock=threading.Lock(), bodies=[]
        )
        self.robot = RobotIds.of(self.model)
        self.pose(keyframe)
        for part in self.kit.parts:
            part.ready(self.world, None)
        self.before = [self.kit.hand.before_step, self.kit.people.before_step]
        self.after = [self.kit.nest.after_step, self.kit.battery.after_step]
        self.after.append(self.kit.props.after_step)

    def pose(self, keyframe: str) -> None:
        """Every joint and the duck at the layout's spawn in this keyframe, the actuators
        holding it (things stay where they are: the keyframes were rewritten for them)."""
        mujoco = _mujoco()
        key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
        if key < 0:
            raise ValueError(f"no keyframe {keyframe!r}")
        mujoco.mj_resetDataKeyframe(self.model, self.data, key)
        self.hold(keyframe)
        mujoco.mj_forward(self.model, self.data)

    def hold(self, keyframe: str) -> None:
        """The actuators hold this keyframe's pose from now on."""
        mujoco = _mujoco()
        key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
        self.data.ctrl[:] = self.model.key_ctrl[key]

    def step(self, n: int = 1) -> None:
        mujoco = _mujoco()
        for _ in range(n):
            for fn in self.before:
                fn(self.world)
            mujoco.mj_step(self.model, self.data)
            for fn in self.after:
                fn(self.world)

    def run(self, seconds: float, each: Callable[[Headless], None] | None = None) -> None:
        for _ in range(int(round(seconds / 0.005))):
            self.step()
            if each is not None:
                each(self)

    @property
    def trunk(self) -> list[float]:
        return [float(v) for v in self.data.xpos[self.robot.trunk]]

    @property
    def tilt(self) -> float:
        return tilt_of([float(v) for v in self.data.xquat[self.robot.trunk]])
