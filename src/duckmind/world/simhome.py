"""The world server: the official MuJoCo body server in a home. No upstream file is edited.

``duckmind.livebody`` starts it instead of :mod:`duckmind.bodyhost` when a home is asked for
(``LiveBody(home=...)`` or ``DUCKMIND_LIVE_HOME``); cwd = microduck_rl, PYTHONPATH =
microduck_rl/src plus this repo's src, the official arguments passed through::

    python -m duckmind.world.simhome --headless --keyframe SIT --port 7801 \
        [--home flat|<layout name>|<layout path>] [--spawn auto|nest|mat|X_MM,Y_MM,YAW_MRAD] \
        [--ext module[:function] ...] [--collision allcollisions] [--lockstep | --no-lockstep]

It is :mod:`duckmind.bodyhost` — the same guards, collision table, lockstep stepping and stall
monitor, imported, not forked (``bodyhost`` itself is unchanged for the owner's phase-0 runs)
— plus five things (physical-design spec §11.1 and §17 phase 2):

1. **A home.** ``--home`` names a layout (``src/duckmind/world/homes/*.json``, with variants
   after commas: ``default_home,no_nest`` for a duck without a nest such as 蛋黃) or a layout
   file. :mod:`duckmind.world.homelayout` turns it into ONE ``MjSpec`` built from the official
   files by path (``homelayout.load(name, mjcf_dir=...) -> Layout``,
   ``homelayout.compose(layout, collision=..., mjcf_dir=...) -> MjSpec``, both given the MJCF
   folder whose files this server checked); the compiled model goes in through
   a patched, pinned ``body_server.build_world``. ``--home flat`` (the default) is the official
   flat ``scene_allcollisions.xml``. While ``homelayout`` does not exist yet a named home falls
   back to the flat floor and says so (``x_info`` → ``home.source == "fallback"``, and a line on
   stdout); a layout that exists but is wrong refuses to start.
2. **The keyframe trap, closed.** Official keyframes list only the robot; MuJoCo pads them to the
   world's ``nq`` with every free object at the origin, so a keyframe reset piles every toy onto
   the duck. The world model here has NO keyframes (:func:`strip_keyframes`); poses are read by
   joint name from the official scene (``body_server.pose_table``) and written only into the
   robot, and free objects keep the positions the layout gave them.
3. **A spawn where the layout says.** The duck is placed at the layout's spawn (``--spawn auto``:
   the layout's ``spawn``, else its nest, else its mat, else the origin), at rest on whatever it
   stands on (:meth:`SimHome.settle`), before robotd connects.
4. **World ops** on the body server's own port (robotd never sends them; evaluator truth, never
   for Duck Core) — see *Ops* below.
5. **An extension registry**, so the sensors, hand, people, battery and props modules plug in
   without editing this file — see *Extensions* below.

Pins. Every official function this module patches, copies or relies on is pinned by the sha256
of its source (:data:`SIMHOME_UPSTREAM_SHA256`: bodyhost's five plus ``World.__init__``,
``build_world``, ``pose_table``, ``Body.__init__``, ``Body.place``, ``Body.remember`` and
``Body.restore``); a change refuses to start. Every official file a home loads (the base scene
and everything it includes) is checked against :data:`REVIEWED_SHA256` (bodyhost's table plus
``scene_apartment.xml`` and ``apartment.xml``); a changed or unlisted file starts, but the
monitor says the physics cannot be trusted until someone reviews it.

Extensions
----------
A module ``M`` passed as ``--ext M`` (or ``M:function``; also ``DUCKMIND_SIMHOME_EXT``, comma
separated) is imported before the world is built and its ``register(registry)`` is called with
the :class:`Registry`. Everything it registers runs in registration order::

    def register(registry: simhome.Registry) -> None:
        registry.edit_spec(add_hand_body)          # (spec: mujoco.MjSpec, home: HomeInfo) -> None
        registry.on_ready(find_ids)                # (world, body) -> None
        registry.before_step(drive_hand)           # (world) -> None, per substep, lock HELD
        registry.after_step(watch_contacts)        # (world) -> None, per substep, lock HELD
        registry.op("x_hand", hand_op)             # (body, request: dict) -> dict, lock NOT held
        registry.op("tof", tof_op, override=True)  # the only official op that may be replaced
        registry.snapshot("hand", hand_state)      # (world) -> JSON value, lock HELD
        registry.on_stop(close_sockets)            # () -> None, at exit

* ``edit_spec(fn)``: called on the composed home spec (after the layout, before compile; the
  keyframes are stripped afterwards). Add bodies, geoms, mocap bodies, equality welds, sites;
  never rename or change official robot elements (``Body`` finds them by name).
* ``on_ready(fn)``: once, right after the duck is spawned and before the server accepts any
  connection (single-threaded): look up ids, start threads (a render or media thread may keep
  ``world``). ``fn(world, body)``; ``world`` is ``body_server.World`` (``model``, ``data``,
  ``lock``, ``bodies``), ``body`` the duck's ``body_server.Body``. One that raises is switched
  off with its module's step hooks and named in ``x_monitor``; the world starts all the same.
* ``before_step(fn)`` / ``after_step(fn)``: around EVERY ``mj_step`` (5 ms of physics), on
  physics time (``world.data.time``), inside the world's lock — never take ``world.lock`` in a
  hook (it is not re-entrant). In lockstep the world moves only when robotd writes, so a hook
  script paced by ``data.time`` stays in step with the robot's policies whatever the wall clock
  does. Keep them to microseconds. A hook that raises is switched off and named in
  ``x_monitor`` (trust ``suspect``).
* ``op(name, fn, override=False)``: an op on the body server's port. ``x_*`` names are world or
  evaluator ops (truth, never for Duck Core); other new names are sensor ops (only what a real
  sensor could report, e.g. the nest's charging pin ``pg``). The official ops (``hello``,
  ``read``, ``write``, ``gain``, ``torque``, ``slow``, ``tof``) cannot be replaced, except
  ``tof`` with ``override=True`` (the world's multi-ray ToF; ``tofd --sim`` polls it). ``slow``
  stays official (7.4 V) until phase 5 (owner decision E1: the world battery is display only).
  ``fn(body, request) -> dict`` runs on the connection's thread WITHOUT the lock: take
  ``body.world.lock`` around any model/data access and keep it to milliseconds (robotd's
  lockstep writes wait for it). Raise ``ValueError`` for a bad request; the caller gets
  ``{"error": "..."}``.
* ``snapshot(section, fn)``: a section of ``x_world``. ``hand``, ``people``, ``battery``,
  ``lights`` and ``nest`` are always present (``null`` until a module fills them); other names
  are added; ``sim_time``, ``home``, ``duck``, ``props`` and ``mocap`` are simhome's own.
* ``on_stop(fn)``: at exit (SIGTERM included).

``registry.home`` (:class:`HomeInfo`: name, layout object, spawn, base scene) is set before the
first ``edit_spec`` call; ``registry.args`` holds this process's own arguments. Extensions take
their settings from the layout or their own environment variables: any other command-line flag
goes on to the official server, which refuses flags it does not know.

Ops
---
Newline-delimited JSON on the body port, like every official op. Units: millimetres and
milliradians as integers (the layouts' units), quaternions ``[w, x, y, z]``.

``{"op": "x_place", "x_mm": 450, "y_mm": 0, "yaw_mrad": 0, "keyframe": "SIT"}``
    A hand sets the duck down at rest: the keyframe's joints (by name, robot only — nothing else
    in the world moves), trunk at (x, y) facing ``yaw``, lowered until its lowest part is 1 mm
    above the highest static collider under it (probed from ``probe_from_mm``, default 250, so
    a table top above is not a support: raise it to put the duck on something higher), all its
    velocities zero, and held there if robotd has not enabled it yet. Optional:
    ``"keyframe"`` SIT (default) | STAND | FOLD | INIT | HOME, ``"probe_from_mm"``,
    ``"force": true`` (place even when something is in the way). If the placed duck would sit
    more than 2 mm inside anything (a wall, furniture, a toy) the placement is undone and the
    answer says what was in the way. In a home, a spot outside the layout's rooms (past a wall:
    the official floor plane runs on outside the house) is refused before any physics
    (``"refused": true``); ``"from_mm": [x, y]`` says the spot stands in for another one that
    was taken, and then a spot over a drop (kept a duck's radius off it) or behind a wall, a
    railing or a gate from it is refused too. ``force`` skips both. Answer::

        {"ok": true, "x_mm": 450, "y_mm": 0, "yaw_mrad": 0, "keyframe": "SIT",
         "trunk_z_mm": 75, "surface_mm": 0, "support": ["floor"], "blocked_by": [],
         "released": true, "sim_time": 12.34, "why_zh": []}

``{"op": "x_world"}``
    The world's truth, for evaluators and the page (never for Duck Core)::

        {"sim_time": 12.34,
         "home": {"name": "default_home", "variant": null, "source": "layout",
                  "fingerprint": "..."},
         "duck": {"pos_mm": [x, y, z], "quat": [w, x, y, z], "yaw_mrad": 0, "tilt_mrad": 3,
                  "released": true, "torque_on": true},
         "props": [{"name": "ball_0", "pos_mm": [...], "quat": [...], "yaw_mrad": 0,
                    "speed_mm_s": 0}],
         "mocap": [{"name": "hand", "pos_mm": [...], "quat": [...]}],   (parked ones left out)
         "hand": null, "people": null, "battery": null, "lights": null, "nest": null}

``{"op": "x_monitor"}``
    bodyhost's stall monitor (``trust``, ``why_zh``, realtime, stalls, robotd's rate), plus
    ``home`` and failed extension hooks as reasons.

``{"op": "x_info"}``
    What is loaded: bodyhost's fields (``robot_colliding_geoms`` counts only the robot, not the
    toys) plus ``home``, ``spawn``, ``extensions``, ``ops``, ``hooks``, ``sections``, ``world``
    (props, mocap bodies, static colliders) and the pins.

Nothing heavy is imported at import time: tests and ``duckmind.livebody`` use the registry,
the maths and :class:`WorldClient` without MuJoCo.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
import re
import signal
import sys
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

from duckmind import bodyhost
from duckmind.bodyhost import (
    COLLISIONS,
    DEFAULT_COLLISION,
    DEFAULT_LOCKSTEP,
    Host,
    HostClient,
    check_reviewed,
    check_scene_pair,
    check_upstream,
    scene_for,
)
from duckmind.i18n import lst, msg
from duckmind.i18n.boundary import attach

HOME_VERSION = "duckmind-simhome-1"
FLAT = "flat"  # the official flat floor, no layout
HOMELAYOUT = "duckmind.world.homelayout"

# bodyhost's pins plus every official function simhome patches, copies or relies on
# (checked 2026-09-27 against microduck_rl cb70b792; digests as inspect.getsource gives them).
SIMHOME_UPSTREAM_SHA256: dict[str, str] = {
    **bodyhost.UPSTREAM_SHA256,
    # relied on: builds the model through build_world and sets TIMESTEP
    "World.__init__": "f264cf3056e5bdc1b41ee64b0a5413ff9120dfe1bce263304ad1d2bf26d1d085",
    # replaced: the home's compiled model comes in here
    "build_world": "408aa6724daa09ce49da9f011b1416c5a1d73bd0e24fb1940fb1976dc4b90989",
    # called: poses by joint name from the official scene (spawn and x_place)
    "pose_table": "bc40306b882b81eee3174d417c255706388b9c09fc3515b79b6901ed9f3ff8e4",
    # relied on: trunk, trunk_dof, qpos_adr, qvel_adr, to_wire, released, held
    "Body.__init__": "05a2a18002964f3771e38fcc2afea5b6aeda255eae98fa07b73a8e94e3a6f9f4",
    # wrapped (spawn) and called (x_place): joints, velocities, ctrl, remember
    "Body.place": "2692eada8969d51e4ffa3f8f41871ed97e586e847861bb0a8303c02ae2e7dd9c",
    # called: a duck not yet enabled is held where it was placed
    "Body.remember": "d15e2b7b0a1f298306ab72f3db5d2c6225c76d0f08b7f776787d5750b5e23490",
    # called by the hooked World.step copy, as upstream does
    "Body.restore": "2f573482a0280d6af2082570b5d0543f6d9ec94fcda18008552544860c1abdf8",
}

# The official home files, reviewed 2026-09-27 (microduck_rl cb70b792): the apartment scene is
# robot_allcollisions.xml + apartment.xml; apartment.xml uses no mesh or texture files.
HOME_REVIEWED_SHA256: dict[str, str] = {
    "scene_apartment.xml": "f89da842d8972db257cb9a1a7c1804e21e1d0c01cc3388d6de6ac6fa1060f6e2",
    "apartment.xml": "4e6a6f5baf07ab842d9860a2e5baa72f516b864213cecb8184123d17ab23d7de",
}
REVIEWED_SHA256: dict[str, str] = {**bodyhost.REVIEWED_SHA256, **HOME_REVIEWED_SHA256}

OFFICIAL_OPS = frozenset({"hello", "read", "write", "gain", "torque", "slow", "tof"})
OVERRIDABLE_OPS = frozenset({"tof"})  # the world's multi-ray ToF; `slow` stays official (E1)
BUILTIN_OPS = ("x_monitor", "x_info", "x_place", "x_world")
SNAPSHOT_OWN = ("sim_time", "home", "duck", "props", "mocap")
SNAPSHOT_SECTIONS = ("hand", "people", "battery", "lights", "nest")
KEYFRAMES = ("SIT", "STAND", "FOLD", "INIT", "HOME")
OP_NAME = re.compile(r"^[a-z][a-z0-9_.]{0,63}$")
SECTION_NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")

# Sensor helpers (the nest's electrode, the belly probe) sit in this geom group
# (homelayout.GROUP_HIDDEN): never a support, never "in the way" of a placement.
SENSOR_GROUP = 5
PROBE_FROM_MM = 250  # a duck is set down on the highest static collider below this
REST_CLEARANCE_M = 0.001  # its lowest part this far above that surface
BLOCK_DEPTH_M = 0.002  # deeper than this inside anything else: something is in the way
PLACE_LIMIT_MM = 50_000
MOVING_MM_S = 5
DUCK_RADIUS_MM = 140  # homelayout.DUCK_RADIUS_MM: a stand-in spot keeps this far off a drop
PARKED_Z_M = -1.0  # a mocap body this far under the floor is parked (x_world leaves it out)


# ── the registry ────────────────────────────────────────────────────────────────


@dataclass
class Entry:
    """One registered callback, named for the monitor and x_info."""

    name: str
    fn: Callable[..., Any]
    disabled: str | None = None  # why it was switched off (a hook that raised)
    owner: str = "simhome"  # the module that registered it (--ext)


class Registry:
    """What extension modules add to the world server (see the module docstring)."""

    def __init__(self) -> None:
        self.spec_edits: list[Entry] = []
        self.ready_hooks: list[Entry] = []
        self.before: list[Entry] = []
        self.after: list[Entry] = []
        self.ops: dict[str, Entry] = {}
        self.sections: dict[str, Entry] = {}
        self.stops: list[Entry] = []
        self.extensions: list[str] = []
        self.home: HomeInfo | None = None
        self.args: argparse.Namespace | None = None
        self.owner = "simhome"  # the module registering now, for the names

    def _name(self, fn: Callable[..., Any], name: str | None) -> str:
        return name or f"{self.owner}.{getattr(fn, '__name__', 'callback')}"

    def edit_spec(self, fn: Callable[[Any, HomeInfo], None], *, name: str | None = None) -> None:
        self.spec_edits.append(Entry(self._name(fn, name), fn, owner=self.owner))

    def on_ready(self, fn: Callable[[Any, Any], None], *, name: str | None = None) -> None:
        self.ready_hooks.append(Entry(self._name(fn, name), fn, owner=self.owner))

    def before_step(self, fn: Callable[[Any], None], *, name: str | None = None) -> None:
        self.before.append(Entry(self._name(fn, name), fn, owner=self.owner))

    def after_step(self, fn: Callable[[Any], None], *, name: str | None = None) -> None:
        self.after.append(Entry(self._name(fn, name), fn, owner=self.owner))

    def on_stop(self, fn: Callable[[], None], *, name: str | None = None) -> None:
        self.stops.append(Entry(self._name(fn, name), fn, owner=self.owner))

    def op(
        self,
        name: str,
        fn: Callable[[Any, dict[str, Any]], dict[str, Any]],
        *,
        override: bool = False,
    ) -> None:
        if not OP_NAME.match(name):
            raise ValueError(f"op name {name!r}: lower case letters, digits, '_' and '.'")
        if name in BUILTIN_OPS:
            raise ValueError(f"{name!r} is simhome's own op")
        if name in OFFICIAL_OPS and not (override and name in OVERRIDABLE_OPS):
            allowed = ", ".join(sorted(OVERRIDABLE_OPS))
            raise ValueError(
                f"{name!r} is an upstream body-server op; only {allowed} may be replaced "
                "(override=True)"
            )
        if name not in OFFICIAL_OPS and override:
            raise ValueError(f"{name!r} is not an upstream op: nothing to override")
        if name in self.ops:
            raise ValueError(f"op {name!r} is already registered by {self.ops[name].name}")
        self.ops[name] = Entry(self._name(fn, f"{self.owner}:{name}"), fn)

    def snapshot(self, section: str, fn: Callable[[Any], Any]) -> None:
        if not SECTION_NAME.match(section) or section in SNAPSHOT_OWN:
            raise ValueError(f"x_world section {section!r} is simhome's own or not a plain name")
        if section in self.sections:
            raise ValueError(f"x_world section {section!r} is already filled by another module")
        self.sections[section] = Entry(self._name(fn, f"{self.owner}:{section}"), fn)

    def load(self, specs: Iterable[str]) -> None:
        """Import each ``module[:function]`` and let it register (``register`` by default)."""
        for spec in specs:
            spec = spec.strip()
            if not spec:
                continue
            module_name, _, function = spec.partition(":")
            module = importlib.import_module(module_name)
            register = getattr(module, function or "register", None)
            if not callable(register):
                raise ValueError(f"{module_name} has no {function or 'register'}(registry)")
            self.owner = module_name
            try:
                register(self)
            finally:
                self.owner = "simhome"
            self.extensions.append(spec)

    def view(self) -> dict[str, Any]:
        def names(entries: Iterable[Entry]) -> list[str]:
            return [entry.name for entry in entries]

        return {
            "extensions": list(self.extensions),
            "hooks": {
                "edit_spec": names(self.spec_edits),
                "on_ready": names(self.ready_hooks),
                "before_step": names(self.before),
                "after_step": names(self.after),
                "on_stop": names(self.stops),
            },
            "registered_ops": sorted(self.ops),
            "sections": sorted(self.sections),
        }


# ── the home ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Spawn:
    """Where the duck starts: world millimetres and milliradians, and where that came from."""

    x_mm: int = 0
    y_mm: int = 0
    yaw_mrad: int = 0
    at: str = "origin"  # "layout" | "nest" | "mat" | "asked" | "origin"


@dataclass
class HomeInfo:
    """The home this world server runs (handed to every ``edit_spec`` callback)."""

    asked: str | None
    name: str
    source: str  # "flat" | "layout" | "fallback"
    base_scene: Path
    official_files: list[str]
    spawn: Spawn
    layout: Any = None
    fingerprint: str | None = None
    notes_zh: list[str] = field(default_factory=list)
    keyframes_removed: int = 0
    variant: str | None = None  # the layout's variant(s), e.g. "no_nest"

    def brief(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "variant": self.variant,
            "source": self.source,
            "fingerprint": self.fingerprint,
        }

    def view(self) -> dict[str, Any]:
        return {
            "asked": self.asked,
            "name": self.name,
            "variant": self.variant,
            "source": self.source,
            "base_scene": self.base_scene.name,
            "official_files": list(self.official_files),
            "fingerprint": self.fingerprint,
            "spawn": {
                "x_mm": self.spawn.x_mm,
                "y_mm": self.spawn.y_mm,
                "yaw_mrad": self.spawn.yaw_mrad,
                "at": self.spawn.at,
            },
            "keyframes_removed": self.keyframes_removed,
            "notes_zh": list(self.notes_zh),
        }


@dataclass
class HomePlan:
    """What :func:`plan_home` decided, and how to make the spec (needs MuJoCo)."""

    info: HomeInfo
    make_spec: Callable[[], Any]


_FIND = object()  # plan_home: look for duckmind.world.homelayout


def find_homelayout() -> Any:
    """The layout module, or None while it does not exist (an import error inside it is not
    hidden: a home that exists but cannot load refuses to start)."""
    if importlib.util.find_spec(HOMELAYOUT) is None:
        return None
    return importlib.import_module(HOMELAYOUT)


def _get(obj: Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(round(value))
    return None


def pose_mm(obj: Any) -> tuple[int, int, int] | None:
    """(x_mm, y_mm, yaw_mrad) of a layout element: an object or mapping with ``x_mm``/``y_mm``
    and ``yaw_mrad`` or ``heading_mrad`` (0 when absent), itself or under ``pose``; or a
    sequence ``(x_mm, y_mm[, yaw_mrad])``."""
    if obj is None:
        return None
    if isinstance(obj, (list, tuple)) and len(obj) in (2, 3):
        values = [_int(v) for v in obj]
        whole = [v for v in values if v is not None]
        if len(whole) != len(values):
            return None
        return whole[0], whole[1], whole[2] if len(whole) == 3 else 0
    inner = _get(obj, "pose")
    if inner is not None and inner is not obj:
        found = pose_mm(inner)
        if found is not None:
            return found
    px, py = _int(_get(obj, "x_mm")), _int(_get(obj, "y_mm"))
    if px is None or py is None:
        return None
    heading = _int(_get(obj, "yaw_mrad"))
    if heading is None:
        heading = _int(_get(obj, "heading_mrad"))
    return px, py, heading or 0


def parse_spawn(text: str) -> tuple[str, tuple[int, int, int] | None]:
    """``auto`` | ``nest`` | ``mat`` | ``X_MM,Y_MM[,YAW_MRAD]``."""
    text = (text or "auto").strip().lower()
    if text in {"auto", "nest", "mat"}:
        return text, None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) in (2, 3):
        try:
            values = [int(p) for p in parts]
        except ValueError:
            values = []
        if values:
            return "asked", (values[0], values[1], values[2] if len(values) == 3 else 0)
    raise ValueError(f"--spawn {text!r}: auto, nest, mat or X_MM,Y_MM[,YAW_MRAD]")


def resolve_spawn(layout: Any, asked: str, layouts: Any = None) -> tuple[Spawn, list[str]]:
    """Where to put the duck at start, and plain notes when the asked place is not there."""
    kind, given = parse_spawn(asked)
    if given is not None:
        return Spawn(*given, at="asked"), []
    notes: list[str] = []
    order = {"auto": ("spawn", "nest", "mat"), "nest": ("nest",), "mat": ("mat",)}[kind]
    spawn_pose = getattr(layouts, "spawn_pose", None) if layouts is not None else None
    for where in order:
        found = None
        if layout is not None and callable(spawn_pose) and where != "spawn":
            try:
                found = pose_mm(spawn_pose(layout, where))
            except (KeyError, ValueError, LookupError):
                found = None
        if found is None and layout is not None:
            found = pose_mm(_get(layout, where))
        if found is not None:
            return Spawn(*found, at="layout" if where == "spawn" else where), notes
    if kind != "auto":
        label = {"nest": "窩", "mat": "墊子"}[kind]
        notes.append(f"這個家沒有{label}：小鴨放在原點")
    return Spawn(), notes


def official_files(mjcf_dir: Path, scene: str) -> list[str]:
    """The official files a scene loads: itself and every file it includes, recursively
    (names relative to the MJCF folder, in load order; meshes are bodyhost's mesh digest)."""
    found: list[str] = []

    def visit(name: str) -> None:
        if name in found:
            return
        found.append(name)
        path = mjcf_dir / name
        try:
            root = ET.parse(path).getroot()
        except (OSError, ET.ParseError):
            return
        for node in root.iter("include"):
            child = node.get("file")
            if child:
                visit(str((Path(name).parent / child).as_posix()))

    visit(scene)
    return found


def check_home_files(
    mjcf_dir: Path, names: Iterable[str], pinned: Mapping[str, str] | None = None
) -> list[str]:
    """One plain line per official file the home loads that is missing, changed, or not in the
    reviewed table (empty: every file is as reviewed)."""
    pinned = REVIEWED_SHA256 if pinned is None else pinned
    problems: list[str] = []
    for name in names:
        path = mjcf_dir / name
        got = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        want = pinned.get(name)
        if got is None:
            problems.append(f"上游的 {name} 找不到：這個家的物理不完整")
        elif want is None:
            problems.append(f"上游的 {name} 不在核對表裡：還沒有人核對，物理可能不是檢查過的樣子")
        elif got != want:
            problems.append(f"上游的 {name} 和核對過的版本不一樣：還沒有人核對，物理可能變了")
    return problems


def _base_sources(
    layout: Any, layouts: Any, mjcf_dir: Path, collision: str, default: str
) -> tuple[str, list[str]]:
    """The official scene a layout starts from, and every official file it loads.

    ``homelayout`` names them in ``layout.base``: ``{"world": "apartment.xml", "scenes":
    {"allcollisions": "scene_apartment.xml"}}`` (the official scene for this collision model,
    or the collision model's own scene with the world file attached), or nothing (the official
    floor). A plain ``base_scene`` / ``base`` file name is taken too."""
    base = _get(layout, "base")
    world = _get(base, "world") if base is not None and not isinstance(base, (str, Path)) else None
    scenes = _get(base, "scenes") if world is not None else None
    if isinstance(world, str) and world:
        scene = _get(scenes, collision) if scenes is not None else None
        if isinstance(scene, str) and (mjcf_dir / scene).is_file():
            return scene, official_files(mjcf_dir, scene)
        files = official_files(mjcf_dir, default)
        return default, [*files, *(f for f in official_files(mjcf_dir, world) if f not in files)]
    candidates: list[Any] = []
    getter = getattr(layouts, "base_scene", None)
    if callable(getter):
        try:
            candidates.append(getter(layout))
        except (KeyError, ValueError, LookupError, TypeError):
            pass
    for key in ("base_scene", "base"):
        value = _get(layout, key)
        candidates.append(value)
        if value is not None and not isinstance(value, (str, Path)):
            candidates.append(_get(value, "scene"))
    for value in candidates:
        if isinstance(value, (str, Path)) and str(value):
            name = Path(value).name
            if (mjcf_dir / name).is_file():
                return name, official_files(mjcf_dir, name)
    return default, official_files(mjcf_dir, default)


def _takes(fn: Any, name: str) -> bool:
    """Whether a callable accepts a keyword argument ``name``."""
    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )


def plan_home(
    asked: str | None,
    *,
    mjcf_dir: Path,
    collision: str = DEFAULT_COLLISION,
    spawn: str = "auto",
    layouts: Any = _FIND,
) -> HomePlan:
    """Decide the home: the flat official floor, a layout, or (while ``homelayout`` does not
    exist) the flat floor as a named fallback. ``layouts`` is the layout module (tests pass one,
    or None for 'not there'); the spec is made later by ``plan.make_spec()``."""
    flat_scene = scene_for(mjcf_dir, collision)

    def flat_spec() -> Any:
        import mujoco

        return mujoco.MjSpec.from_file(str(flat_scene))

    wanted = (asked or "").strip()
    if not wanted or wanted == FLAT:
        where, notes = resolve_spawn(None, spawn)
        info = HomeInfo(
            asked=asked,
            name=FLAT,
            source="flat",
            base_scene=flat_scene,
            official_files=official_files(mjcf_dir, flat_scene.name),
            spawn=where,
            notes_zh=notes,
        )
        return HomePlan(info, flat_spec)
    module = find_homelayout() if layouts is _FIND else layouts
    if module is None or not callable(getattr(module, "load", None)):
        why = (
            f"佈置程式（world/homelayout.py）還沒有好，所以「{wanted}」這個家先用上游的平地"
            f"（{flat_scene.name}）代替：牆和家具還不在物理裡"
        )
        where, notes = resolve_spawn(None, spawn)
        info = HomeInfo(
            asked=asked,
            name=FLAT,
            source="fallback",
            base_scene=flat_scene,
            official_files=official_files(mjcf_dir, flat_scene.name),
            spawn=where,
            notes_zh=[why, *notes],
        )
        return HomePlan(info, flat_spec)
    compose = getattr(module, "compose", None)
    if not callable(compose):
        raise ValueError(f"{HOMELAYOUT} has load() but no compose(layout, collision=...)")
    folder = {"mjcf_dir": str(mjcf_dir)}  # the official files this server was pinned against
    layout = module.load(wanted, **(folder if _takes(module.load, "mjcf_dir") else {}))
    base, files = _base_sources(layout, module, mjcf_dir, collision, flat_scene.name)
    where, notes = resolve_spawn(layout, spawn, module)
    fingerprint = None
    maker = getattr(module, "fingerprint", None)
    if callable(maker):
        fingerprint = str(maker(layout))
    elif isinstance(_get(layout, "fingerprint"), str):
        fingerprint = str(_get(layout, "fingerprint"))
    name = _get(layout, "id") or _get(layout, "name") or Path(wanted).stem
    variant = _get(layout, "variant")
    info = HomeInfo(
        asked=asked,
        name=str(name),
        source="layout",
        base_scene=mjcf_dir / base,
        official_files=files,
        spawn=where,
        layout=layout,
        fingerprint=fingerprint,
        notes_zh=notes,
        variant=str(variant) if variant else None,
    )
    extra = folder if _takes(compose, "mjcf_dir") else {}
    return HomePlan(info, lambda: compose(layout, collision=collision, **extra))


def strip_keyframes(spec: Any) -> int:
    """Delete every keyframe from a composed spec (the keyframe trap: official keyframes are
    padded with every free object at the origin). Returns how many were removed."""
    keys = list(spec.keys)
    for key in keys:
        spec.delete(key)
    return len(keys)


# ── pose maths (no MuJoCo) ──────────────────────────────────────────────────────


def yaw_quat(yaw_mrad: int) -> list[float]:
    """Upright, facing ``yaw`` (about world z): ``[w, x, y, z]``."""
    half = yaw_mrad / 2000.0
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def quat_yaw_mrad(quat: Iterable[float]) -> int:
    w, x, y, z = (float(v) for v in quat)
    return int(round(1000.0 * math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))))


def quat_tilt_mrad(quat: Iterable[float]) -> int:
    """The angle between the body's z axis and the world's up."""
    w, x, y, z = (float(v) for v in quat)
    norm = w * w + x * x + y * y + z * z or 1.0
    up = 1.0 - 2.0 * (x * x + y * y) / norm
    return int(round(1000.0 * math.acos(max(-1.0, min(1.0, up)))))


def wrap_mrad(value: int) -> int:
    """An angle in milliradians wrapped into [-π, π)."""
    full = 2000.0 * math.pi
    return int(round((value + full / 2) % full - full / 2))


@dataclass(frozen=True)
class PlaceArgs:
    x_mm: int
    y_mm: int
    yaw_mrad: int = 0
    keyframe: str = "SIT"
    probe_from_mm: int = PROBE_FROM_MM
    force: bool = False
    from_mm: tuple[int, int] | None = None  # the spot asked for, when this one stands in for it


def parse_place(request: Mapping[str, Any]) -> PlaceArgs:
    """The fields of an ``x_place`` request, checked (``ValueError`` names what is wrong)."""

    def whole(key: str, default: int | None, low: int, high: int) -> int:
        raw = request.get(key, default)
        value = _int(raw)
        if value is None or (isinstance(raw, float) and not raw.is_integer()):
            raise ValueError(f"x_place {key} must be a whole number, not {raw!r}")
        if not low <= value <= high:
            raise ValueError(f"x_place {key} {value} is outside [{low}, {high}]")
        return value

    x = whole("x_mm", None, -PLACE_LIMIT_MM, PLACE_LIMIT_MM)
    y = whole("y_mm", None, -PLACE_LIMIT_MM, PLACE_LIMIT_MM)
    yaw = wrap_mrad(whole("yaw_mrad", 0, -100_000, 100_000))
    keyframe = str(request.get("keyframe", "SIT")).upper()
    if keyframe not in KEYFRAMES:
        raise ValueError(f"x_place keyframe {keyframe!r}: one of {', '.join(KEYFRAMES)}")
    probe = whole("probe_from_mm", PROBE_FROM_MM, 20, 3_000)
    force = request.get("force", False)
    if not isinstance(force, bool):
        raise ValueError("x_place force must be true or false")
    asked = request.get("from_mm")
    near: tuple[int, int] | None = None
    if asked is not None:
        if not isinstance(asked, (list, tuple)) or len(asked) != 2:
            raise ValueError("x_place from_mm is [x_mm, y_mm]")
        values = [_int(value) for value in asked]
        if any(value is None or abs(value) > PLACE_LIMIT_MM for value in values):
            raise ValueError(f"x_place from_mm must be whole numbers, not {asked!r}")
        near = (int(values[0] or 0), int(values[1] or 0))
    return PlaceArgs(x, y, yaw, keyframe, probe, force, near)


def rest_shift(min_distance_m: float, clearance_m: float = REST_CLEARANCE_M) -> float:
    """How far to raise the duck (negative: lower it) so its lowest part is ``clearance``
    above the surface it is ``min_distance`` above now (negative: inside it)."""
    return clearance_m - min_distance_m


def footprint_points(
    xy: Iterable[tuple[float, float]], centre: tuple[float, float]
) -> list[tuple[float, float]]:
    """Where to probe for the surface under a duck: its centre and the corners of the box
    around its colliding parts (world x, y)."""
    points = list(xy)
    if not points:
        return [centre]
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    corners = [(x, y) for x in (min(xs), max(xs)) for y in (min(ys), max(ys))]
    return [centre, *corners]


# ── the world server ────────────────────────────────────────────────────────────


def hooked_step(mujoco: Any, home: SimHome) -> Callable[..., None]:
    """``body_server.World.step`` line for line (pinned: ``World.step`` in
    :data:`SIMHOME_UPSTREAM_SHA256`), with the extensions' hooks around every ``mj_step``."""
    before, after, run = home.registry.before, home.registry.after, home.run_hooks

    def step(self: Any, times: int = 1) -> None:
        with self.lock:
            for _ in range(times):
                if before:
                    run(before, self)
                mujoco.mj_step(self.model, self.data)
                # A duck nobody has enabled yet is put back where it was. Physics is shared, so
                # it cannot simply not be stepped — and a hand steadying one robot while another
                # walks about is an ordinary thing for a room to contain.
                for body in self.bodies:
                    if not body.released:
                        body.restore()
                if after:
                    run(after, self)

    return step


class SimHome(Host):
    """bodyhost's :class:`~duckmind.bodyhost.Host` in a home: the model, the spawn, the world
    ops and the extension registry."""

    def __init__(
        self,
        *,
        collision: str,
        lockstep: bool,
        mjcf_dir: Path,
        home: str | None = None,
        spawn: str = "auto",
        registry: Registry | None = None,
        layouts: Any = _FIND,
        monitor: bodyhost.StallMonitor | None = None,
    ) -> None:
        super().__init__(
            collision=collision,
            lockstep=lockstep,
            scene=scene_for(mjcf_dir, collision),
            monitor=monitor,
        )
        self.mjcf_dir = mjcf_dir
        self.keyframe_scene = scene_for(mjcf_dir, collision)  # poses: the official scene
        self.asked_home = home
        self.asked_spawn = spawn
        self.registry = registry if registry is not None else Registry()
        self.layouts = layouts
        self.home: HomeInfo | None = None
        self.spec: Any = None
        self.model: Any = None
        self.world: Any = None
        self.body: Any = None
        self.spawned: dict[str, Any] | None = None
        self.failures: list[str] = []
        self.ops: dict[str, Callable[[Any, dict[str, Any]], dict[str, Any]]] = {}
        self._mujoco: Any = None
        self._bs: Any = None
        self._original_place: Callable[..., None] | None = None
        self._poses: dict[str, tuple[dict[str, float] | None, float]] = {}
        self._root = 1
        self._robot_bodies: frozenset[int] = frozenset()
        self._free: list[tuple[str, int, int]] = []
        self._mocap: list[tuple[str, int]] = []
        self._supports: list[int] = []

    # ── building ──────────────────────────────────────────────────────────

    def build(self) -> Any:
        """Compose the home, let the extensions edit it, strip the keyframes and compile."""
        import mujoco

        plan = plan_home(
            self.asked_home,
            mjcf_dir=self.mjcf_dir,
            collision=self.collision,
            spawn=self.asked_spawn,
            layouts=self.layouts,
        )
        self.home = plan.info
        self.registry.home = plan.info
        spec = plan.make_spec()
        for entry in self.registry.spec_edits:
            entry.fn(spec, plan.info)
        plan.info.keyframes_removed = strip_keyframes(spec)
        self.spec = spec
        self.model = spec.compile()
        self.scene = plan.info.base_scene
        self._index(self.model, mujoco)
        return self.model

    def _index(self, model: Any, mujoco: Any) -> None:
        self._root = bodyhost.robot_root(model) or 1
        self._robot_bodies = frozenset(
            b for b in range(int(model.nbody)) if int(model.body_rootid[b]) == self._root
        )
        free, mocap, supports = [], [], []
        for joint in range(int(model.njnt)):
            body = int(model.jnt_bodyid[joint])
            if int(model.jnt_type[joint]) == int(mujoco.mjtJoint.mjJNT_FREE) and (
                body not in self._robot_bodies
            ):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or f"body{body}"
                free.append((name, int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])))
        for body in range(int(model.nbody)):
            if int(model.body_mocapid[body]) >= 0:
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or f"body{body}"
                mocap.append((name, int(model.body_mocapid[body])))
        for geom in range(int(model.ngeom)):
            body = int(model.geom_bodyid[geom])
            if (
                (int(model.geom_contype[geom]) or int(model.geom_conaffinity[geom]))
                and int(model.geom_group[geom]) != SENSOR_GROUP
                and int(model.body_weldid[body]) == 0
                and int(model.body_mocapid[body]) < 0
            ):
                supports.append(geom)
        self._free, self._mocap, self._supports = free, mocap, supports

    # ── installing ────────────────────────────────────────────────────────

    def install(self, bs: Any) -> None:
        """Patch the official module in place (only after :func:`check_upstream` passed with
        :data:`SIMHOME_UPSTREAM_SHA256`): the hooked step copy, the home's model, the spawn,
        then bodyhost's lockstep and monitor, then the world and extension ops."""
        if self.model is None:
            self.build()
        self._bs = bs
        self._mujoco = bs.mujoco
        bs.World.step = hooked_step(bs.mujoco, self)
        first = [self.model]
        spec = self.spec

        def build_world(scene: Any, count: int) -> Any:
            if count != 1:
                raise SystemExit("a home holds one duck (--ducks 1)")
            return first.pop() if first else spec.compile()

        bs.build_world = build_world
        original_place = bs.Body.place
        self._original_place = original_place
        host = self

        def place(body: Any, pose: Any, trunk_z: float, offset_y: float) -> None:
            original_place(body, pose, trunk_z, offset_y)
            if host.body is None and int(getattr(body, "index", 0)) == 0:
                host.spawn(body)

        bs.Body.place = place
        super().install(bs)  # World.step → monitor; dispatch → lockstep; run → idle in lockstep
        inner = bs.Handler.dispatch
        ops: dict[str, Callable[[Any, dict[str, Any]], dict[str, Any]]] = {
            "x_monitor": lambda body, request: self.report(),
            "x_info": lambda body, request: self.info(body),
            "x_place": self.place_op,
            "x_world": self.world_op,
        }
        ops.update({name: entry.fn for name, entry in self.registry.ops.items()})
        self.ops = ops

        def dispatch(handler: Any, body: Any, request: dict[str, Any]) -> dict[str, Any]:
            op = request.get("op")
            handle = ops.get(op) if isinstance(op, str) else None
            if handle is not None:
                # (the world's words carry their catalog messages across the socket, so the
                # page can show them in the viewer's language: docs/i18n.md)
                return attach(handle(body, request))
            return inner(handler, body, request)

        bs.Handler.dispatch = dispatch

    def run_hooks(self, entries: list[Entry], world: Any) -> None:
        for entry in entries:
            if entry.disabled is not None:
                continue
            try:
                entry.fn(world)
            except Exception as error:  # one broken extension must not stop the world
                self.disable(entry, error)

    def stop(self) -> None:
        for entry in self.registry.stops:
            try:
                entry.fn()
            except Exception as error:
                sys.stderr.write(f"== {HOME_VERSION}: stop {entry.name} failed: {error}\n")

    # ── the duck's place ──────────────────────────────────────────────────

    def spawn(self, body: Any) -> dict[str, Any]:
        """Put the duck where the home says (its joints already in the start keyframe), then
        let the extensions get ready. Runs once, before the server accepts connections."""
        assert self.home is not None
        self.world, self.body = body.world, body
        where = self.home.spawn
        with body.world.lock:
            result = self.settle(body, where.x_mm, where.y_mm, where.yaw_mrad, force=True)
        self.spawned = {**result, "at": where.at}
        if result["blocked_by"]:
            sys.stdout.write(
                f"== {HOME_VERSION}: spawn at ({where.x_mm}, {where.y_mm}) mm overlaps "
                f"{', '.join(result['blocked_by'])}\n"
            )
        for entry in self.registry.ready_hooks:
            if entry.disabled is not None:
                continue
            try:
                entry.fn(body.world, body)
            except Exception as error:  # one extension that cannot start must not stop the world
                self.disable(entry, error, when="準備時")
        return self.spawned

    def disable(self, entry: Entry, error: BaseException, *, when: str = "") -> None:
        """Switch off a hook that raised — and, when it was getting ready, the step hooks its
        module registered (they would run on a part that never got ready) — and say so in
        x_monitor (trust "suspect")."""
        entry.disabled = f"{type(error).__name__}: {error}"
        self.failures.append(f"世界擴充 {entry.name} {when}出錯，已經停用：{entry.disabled}")
        sys.stderr.write(f"== {HOME_VERSION}: hook {entry.name} failed: {error}\n")
        if when and entry.owner != "simhome":
            for other in (*self.registry.before, *self.registry.after):
                if other.owner == entry.owner and other.disabled is None:
                    other.disabled = f"{entry.name} 沒有準備好"

    def pose(self, keyframe: str) -> tuple[dict[str, float] | None, float]:
        """A keyframe as (joint name → angle, trunk z), read by joint name from the official
        scene with ``body_server.pose_table`` (so nothing but the robot is ever posed)."""
        if keyframe not in self._poses:
            try:
                self._poses[keyframe] = self._bs.pose_table(self.keyframe_scene, keyframe)
            except SystemExit as error:  # pose_table's way of saying "no such keyframe"
                raise ValueError(str(error)) from None
        return self._poses[keyframe]

    def _robot_colliders(self) -> list[int]:
        model = self.world.model
        return [
            g
            for g in range(int(model.ngeom))
            if int(model.geom_bodyid[g]) in self._robot_bodies
            and (int(model.geom_contype[g]) or int(model.geom_conaffinity[g]))
            and int(model.geom_group[g]) != SENSOR_GROUP
        ]

    def _surface(self, x: float, y: float, top: float) -> tuple[float, int] | None:
        """The highest static collider under (x, y) below ``top``: (its z, the geom)."""
        import numpy as np

        mujoco, model, data = self._mujoco, self.world.model, self.world.data
        point = np.array([x, y, top], dtype=float)
        down = np.array([0.0, 0.0, -1.0])
        best: tuple[float, int] | None = None
        for geom in self._supports:
            kind = int(model.geom_type[geom])
            if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
                dist = mujoco.mj_rayMesh(model, data, geom, point, down)
            elif kind == int(mujoco.mjtGeom.mjGEOM_HFIELD):
                dist = mujoco.mj_rayHfield(model, data, geom, point, down)
            elif kind == int(mujoco.mjtGeom.mjGEOM_SDF):
                continue
            else:
                dist = mujoco.mju_rayGeom(
                    data.geom_xpos[geom],
                    data.geom_xmat[geom],
                    model.geom_size[geom],
                    point,
                    down,
                    kind,
                )
            if dist < 0.0 or (best is not None and top - dist <= best[0]):
                continue
            if self._inside(geom, kind, point):
                continue  # the probe starts inside it: in the way, not under the duck
            best = (top - float(dist), geom)
        return best

    def _inside(self, geom: int, kind: int, point: Any) -> bool:
        """Whether a point is inside a (convex) static geom: a ray up hits it too."""
        import numpy as np

        mujoco, model, data = self._mujoco, self.world.model, self.world.data
        up = np.array([0.0, 0.0, 1.0])
        if kind == int(mujoco.mjtGeom.mjGEOM_PLANE):
            return False
        if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
            return bool(mujoco.mj_rayMesh(model, data, geom, point, up) >= 0.0)
        if kind == int(mujoco.mjtGeom.mjGEOM_HFIELD):
            return False
        return bool(
            mujoco.mju_rayGeom(
                data.geom_xpos[geom], data.geom_xmat[geom], model.geom_size[geom], point, up, kind
            )
            >= 0.0
        )

    def settle(
        self,
        body: Any,
        x_mm: int,
        y_mm: int,
        yaw_mrad: int,
        *,
        probe_from_mm: int = PROBE_FROM_MM,
        force: bool = False,
    ) -> dict[str, Any]:
        """Move the duck (joints already posed) to (x, y) facing yaw, at rest on the highest
        static collider under it, velocities zero; say what it rests on and what is in the
        way. The caller holds ``world.lock`` and undoes a refused placement."""
        import numpy as np

        mujoco, world = self._mujoco, body.world
        model, data = world.model, world.data
        x, y, yaw = x_mm / 1000.0, y_mm / 1000.0, yaw_mrad
        trunk, dof = int(body.trunk), int(body.trunk_dof)
        data.qpos[trunk : trunk + 2] = [x, y]
        data.qpos[trunk + 3 : trunk + 7] = yaw_quat(yaw)
        data.qvel[dof : dof + 6] = 0.0
        mujoco.mj_kinematics(model, data)
        colliders = self._robot_colliders()
        points = footprint_points(
            [(float(data.geom_xpos[g][0]), float(data.geom_xpos[g][1])) for g in colliders],
            (x, y),
        )
        top = probe_from_mm / 1000.0
        hits = [hit for hit in (self._surface(px, py, top) for px, py in points) if hit]
        support = sorted({geom for _, geom in hits})
        surface = max((z for z, _ in hits), default=None)
        why: list[str] = []
        if surface is None:
            why.append(msg("world.simhome.SimHome.settle.cf551e72"))
        else:
            fromto = np.zeros(6)
            gaps = [
                mujoco.mj_geomDistance(model, data, g, s, 0.5, fromto)
                for s in support
                for g in colliders
                if (int(model.geom_contype[g]) & int(model.geom_conaffinity[s]))
                or (int(model.geom_contype[s]) & int(model.geom_conaffinity[g]))
            ]
            if gaps:
                data.qpos[trunk + 2] += rest_shift(float(min(gaps)))
            else:  # nothing of the duck collides with it: the keyframe's height above it
                data.qpos[trunk + 2] += surface
        mujoco.mj_forward(model, data)
        blocked: set[str] = set()
        for contact in data.contact[: int(data.ncon)]:
            one, two = int(contact.geom1), int(contact.geom2)
            ours = [int(model.geom_bodyid[g]) in self._robot_bodies for g in (one, two)]
            if ours[0] == ours[1] or float(contact.dist) >= -BLOCK_DEPTH_M:
                continue
            if SENSOR_GROUP in (int(model.geom_group[one]), int(model.geom_group[two])):
                continue  # a sensor helper touching something is its job, not a collision
            blocked.add(self._geom_name(two if ours[0] else one))
        if blocked:
            why.append(msg("world.simhome.SimHome.settle.bce85afc", list=lst(sorted(blocked))))
        ok = surface is not None and (force or not blocked)
        if ok:
            body.remember()  # a duck not yet enabled is held here, not where it was
        return {
            "ok": ok,
            "x_mm": x_mm,
            "y_mm": y_mm,
            "yaw_mrad": yaw_mrad,
            "trunk_z_mm": int(round(float(data.qpos[trunk + 2]) * 1000.0)),
            "surface_mm": None if surface is None else int(round(surface * 1000.0)),
            "support": [self._geom_name(g) for g in support],
            "blocked_by": sorted(blocked),
            "sim_time": round(float(data.time), 4),
            "why_zh": why,
        }

    def _geom_name(self, geom: int) -> str:
        model = self.world.model
        name = self._mujoco.mj_id2name(model, self._mujoco.mjtObj.mjOBJ_GEOM, geom)
        if name:
            return str(name)
        body = int(model.geom_bodyid[geom])
        owner = self._mujoco.mj_id2name(model, self._mujoco.mjtObj.mjOBJ_BODY, body)
        return f"{owner or f'body{body}'}/geom{geom}"

    # ── the ops ───────────────────────────────────────────────────────────

    def place_problem(self, args: PlaceArgs) -> str | None:
        """Why the home refuses this placement before any physics (None: ask the physics):
        a spot outside the layout's rooms — past a wall, where the official floor plane runs on
        — and, for a spot standing in for another (``from_mm``), one over a drop or behind a
        wall, railing or gate from it. ``force`` skips this (evaluators)."""
        layout = self.home.layout if self.home is not None else None
        check = getattr(layout, "place_problem", None)
        if args.force or not callable(check):
            return None
        near = args.from_mm
        margin = float(DUCK_RADIUS_MM) if near else 0.0
        problem = check(args.x_mm, args.y_mm, from_mm=near, drop_margin_mm=margin)
        return str(problem) if problem else None

    def place_op(self, body: Any, request: dict[str, Any]) -> dict[str, Any]:
        args = parse_place(request)
        problem = self.place_problem(args)
        if problem is not None:
            world = body.world
            return {
                "ok": False,
                "x_mm": args.x_mm,
                "y_mm": args.y_mm,
                "yaw_mrad": args.yaw_mrad,
                "keyframe": args.keyframe,
                "trunk_z_mm": None,
                "surface_mm": None,
                "support": [],
                "blocked_by": [],
                "refused": True,
                "released": bool(getattr(body, "released", True)),
                "sim_time": round(float(world.data.time), 4),
                "why_zh": [problem],
            }
        pose, trunk_z = self.pose(args.keyframe)
        assert self._original_place is not None
        world = body.world
        data = world.data
        with world.lock:
            saved = (
                data.qpos.copy(),
                data.qvel.copy(),
                data.ctrl.copy(),
                data.act.copy(),
                body.held,
            )
            self._original_place(body, pose, trunk_z, 0.0)
            result = self.settle(
                body,
                args.x_mm,
                args.y_mm,
                args.yaw_mrad,
                probe_from_mm=args.probe_from_mm,
                force=args.force,
            )
            if not result["ok"]:
                data.qpos[:], data.qvel[:], data.ctrl[:], data.act[:] = saved[:4]
                body.held = saved[4]
                self._mujoco.mj_forward(world.model, data)
        return {
            **result,
            "keyframe": args.keyframe,
            "released": bool(getattr(body, "released", True)),
        }

    def world_op(self, body: Any, request: dict[str, Any]) -> dict[str, Any]:
        return self.snapshot(body.world, body)

    def snapshot(self, world: Any, body: Any) -> dict[str, Any]:
        """The world's truth (``x_world``)."""
        data = world.data
        trunk = int(body.trunk)

        def mm(values: Iterable[float]) -> list[int]:
            return [int(round(float(v) * 1000.0)) for v in values]

        def unit(values: Iterable[float]) -> list[float]:
            return [round(float(v), 5) for v in values]

        with world.lock:
            quat = data.qpos[trunk + 3 : trunk + 7]
            out: dict[str, Any] = {
                "sim_time": round(float(data.time), 4),
                "home": self.home.brief() if self.home else None,
                "duck": {
                    "pos_mm": mm(data.qpos[trunk : trunk + 3]),
                    "quat": unit(quat),
                    "yaw_mrad": quat_yaw_mrad(quat),
                    "tilt_mrad": quat_tilt_mrad(quat),
                    "released": bool(getattr(body, "released", True)),
                    "torque_on": bool(getattr(body, "torque_on", True)),
                },
                "props": [],
                "mocap": [],
            }
            for name, qadr, vadr in self._free:
                q = data.qpos[qadr + 3 : qadr + 7]
                v = data.qvel[vadr : vadr + 3]
                speed = math.sqrt(sum(float(c) ** 2 for c in v)) * 1000.0
                out["props"].append(
                    {
                        "name": name,
                        "pos_mm": mm(data.qpos[qadr : qadr + 3]),
                        "quat": unit(q),
                        "yaw_mrad": quat_yaw_mrad(q),
                        "speed_mm_s": int(round(speed)),
                    }
                )
            for name, index in self._mocap:
                if float(data.mocap_pos[index][2]) < PARKED_Z_M:
                    continue  # (parked under the floor: a hand or a person not in use)
                out["mocap"].append(
                    {
                        "name": name,
                        "pos_mm": mm(data.mocap_pos[index]),
                        "quat": unit(data.mocap_quat[index]),
                    }
                )
            for section in SNAPSHOT_SECTIONS:
                out[section] = None
            for section, entry in self.registry.sections.items():
                try:
                    out[section] = entry.fn(world)
                except Exception as error:
                    out[section] = {"error": f"{type(error).__name__}: {error}"}
        return out

    def report(self) -> dict[str, Any]:
        report = super().report()
        if self.failures:
            report["trust"] = "suspect"
            report["why_zh"] = [*report["why_zh"], *self.failures]
        report["home"] = self.home.name if self.home else None
        return report

    def info(self, body: Any) -> dict[str, Any]:
        info = super().info(body)
        info["host"] = HOME_VERSION
        info["bodyhost"] = bodyhost.HOST_VERSION
        info["upstream_sha256"] = dict(SIMHOME_UPSTREAM_SHA256)
        info["home"] = self.home.view() if self.home else None
        info["spawn"] = self.spawned
        info["ops"] = sorted(self.ops)
        info["world"] = {
            "props": [name for name, _, _ in self._free],
            "mocap": [name for name, _ in self._mocap],
            "static_colliding_geoms": len(self._supports),
            "keyframes": int(body.world.model.nkey),
        }
        info.update(self.registry.view())
        return info


class WorldClient(HostClient):
    """A connection to the world ops (``x_place``, ``x_world``, ``x_info``, ``x_monitor``, and
    any op an extension registered), apart from robotd's."""

    def place(
        self, x_mm: int, y_mm: int, yaw_mrad: int = 0, keyframe: str = "SIT", **more: Any
    ) -> dict[str, Any]:
        return self.call(
            "x_place", x_mm=x_mm, y_mm=y_mm, yaw_mrad=yaw_mrad, keyframe=keyframe, **more
        )

    def world(self) -> dict[str, Any]:
        return self.call("x_world")


# ── the process ─────────────────────────────────────────────────────────────────


def _refuse(headline: str, lines: list[str]) -> NoReturn:
    sys.stderr.write(
        "== duckmind simhome 拒絕啟動：" + headline + "\n" + "".join(f"   {x}\n" for x in lines)
    )
    sys.stderr.flush()
    raise SystemExit(3)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="The official MuJoCo body server in a home: bodyhost's guards, lockstep and "
        "monitor, a layout, world ops and extensions. Other arguments go to the official server.",
        allow_abbrev=False,
    )
    parser.add_argument("--home", default=None, help="flat (default), a layout name or path")
    parser.add_argument(
        "--spawn",
        default=os.environ.get("DUCKMIND_SIMHOME_SPAWN") or "auto",
        help="auto | nest | mat | X_MM,Y_MM[,YAW_MRAD]",
    )
    parser.add_argument(
        "--ext",
        action="append",
        default=None,
        help="module[:function] to register (repeat or comma-separate; DUCKMIND_SIMHOME_EXT)",
    )
    parser.add_argument("--collision", choices=sorted(COLLISIONS), default=DEFAULT_COLLISION)
    lock = parser.add_mutually_exclusive_group()
    lock.add_argument("--lockstep", dest="lockstep", action="store_true")
    lock.add_argument("--no-lockstep", dest="lockstep", action="store_false")
    parser.set_defaults(lockstep=DEFAULT_LOCKSTEP)
    return parser


def extension_specs(given: list[str] | None, env: str | None) -> list[str]:
    raw = given if given is not None else ([env] if env else [])
    return [part.strip() for item in raw for part in item.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> None:
    ours, rest = _parser().parse_known_args(sys.argv[1:] if argv is None else argv)
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--scene")
    peek.add_argument("--ducks", type=int, default=1)
    peek.add_argument("--cameras", default="")
    peek.add_argument("--headless", action="store_true")
    seen, _ = peek.parse_known_args(rest)
    if seen.scene is not None:
        _refuse("--scene 由 --home 和 --collision 決定", ["用 --home flat 或一個家的名字"])
    if seen.ducks != 1:
        _refuse("一個家只放一隻鴨", ["--ducks 只能是 1"])
    if ours.lockstep and (seen.cameras.strip() or not seen.headless):
        _refuse(
            "lockstep 只支援 --headless、沒有頭部鏡頭",
            ["每一次 robotd 寫入都讓整個世界前進 20 ms；要畫面或鏡頭就加 --no-lockstep"],
        )
    try:
        parse_spawn(ours.spawn)
    except ValueError as error:
        _refuse("--spawn 看不懂", [str(error)])

    from mjlab_microduck.sim import body_server as bs  # cwd/PYTHONPATH as livebody sets them

    upstream = Path(bs.__file__).resolve()
    changed = check_upstream(upstream, SIMHOME_UPSTREAM_SHA256)
    if changed:
        _refuse(
            f"上游的身體伺服器 {upstream} 的程式和核對過的不一樣，為了不讓物理悄悄改變，先不啟動",
            [*changed, "請先確認 src/duckmind/world/simhome.py 對新版還正確，再更新核對表"],
        )
    mjcf = Path(getattr(bs, "SCENES", upstream.parents[1] / "robot" / "microduck"))
    different = check_scene_pair(mjcf)
    if different:
        _refuse("上游的兩個場景不再是「只差碰撞模型」，先不啟動", different)
    registry = Registry()
    registry.args = ours
    try:
        registry.load(extension_specs(ours.ext, os.environ.get("DUCKMIND_SIMHOME_EXT")))
    except Exception as error:
        _refuse("世界擴充載入失敗", [f"{type(error).__name__}: {error}"])
    host = SimHome(
        collision=ours.collision,
        lockstep=ours.lockstep,
        mjcf_dir=mjcf,
        home=ours.home,
        spawn=ours.spawn,
        registry=registry,
    )
    try:
        host.build()
    except Exception as error:
        _refuse(f"家「{ours.home}」建不起來", [f"{type(error).__name__}: {error}"])
    assert host.home is not None
    host.unreviewed = [
        *check_reviewed(upstream, mjcf, ours.collision),
        *check_home_files(mjcf, host.home.official_files),
    ]
    host.unreviewed = list(dict.fromkeys(host.unreviewed))
    for line in [*host.home.notes_zh, *host.unreviewed]:
        print(f"== {HOME_VERSION}: {line}", flush=True)
    host.install(bs)

    def terminate(signum: int, frame: Any) -> None:
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)
    sys.argv = [sys.argv[0], *rest, "--scene", str(host.keyframe_scene)]
    spawn = host.home.spawn
    print(
        f"== {HOME_VERSION}: home {host.home.name} ({host.home.source}, "
        f"{host.home.base_scene.name}), spawn ({spawn.x_mm}, {spawn.y_mm}) mm "
        f"{spawn.yaw_mrad} mrad [{spawn.at}], {ours.collision}, "
        f"lockstep {'on' if ours.lockstep else 'off'}, extensions "
        f"{json.dumps(registry.extensions)}",
        flush=True,
    )
    try:
        bs.main()
    finally:
        host.stop()


if __name__ == "__main__":
    main()
