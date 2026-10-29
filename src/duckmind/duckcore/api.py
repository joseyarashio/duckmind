"""One JSON-in/JSON-out entry point for the simulator UI (``dispatch``).

The same function serves the local HTTP server and could serve an in-browser Python
build. Every mutating call is recorded as an input, so ``replay_check`` can rebuild the
duck from scratch and prove the L1 chain is identical.

**Engines (phase 4, design A6).** A duck names the engine it hatched with in
``Config.engine_rev``. New ducks are ``v4_1`` (this Duck Core, which learns the world's outcomes
from its world port); a saved duck without ``engine_rev`` is a v4 duck and opens on the frozen
``v4_0`` engine (``duckmind.engines.v4_0``) forever, so its history never changes. Open a saved
duck with :func:`open_session` (or :func:`restore` with its inputs); every function here that
takes a session also takes a frozen one and hands it to the frozen api untouched.

**The world (A4, A5).** A v4_1 session lives in a world built from a world header
(``Session(config, world=header)``; saved as ``{"config", "world", "inputs"}``). The outcome
rates are the world's, not the ``Config``'s: :func:`legacy_world` writes the header of the world a
v4 duck lived in (LegacyWorld, ``duckmind.world.legacy``, which registers itself when imported —
Duck Core never imports a world). The simulated family touches the duck through
:class:`LegacyActs`: a hand whose detection was a dice roll in v4 goes through the world's sensor
model first.

**Replay (A8).** ``replay_check`` (config + inputs → same chain), ``percept_replay_check`` (the
Duck Core input tape → a Duck Core with no world and no family → same chain) and
``world_replay_check`` (the world header + the world's own tape → the same percepts).
"""

from __future__ import annotations

import copy
import inspect
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from functools import cache
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

from duckmind import engines, worldport
from duckmind.duckcore import (
    bridge,
    bridge_v2,
    commands,
    content,
    daycycle,
    engine,
    excursion,
    graph,
    moments,
    nest,
    query,
)
from duckmind.duckcore.engine import ENGINE_REV, CoreState, DuckCore, HardwareConfig
from duckmind.duckcore.fixed import SCALE
from duckmind.duckcore.memory import EventLog, event_json, mapping, seq
from duckmind.duckcore.narrate import CATEGORY, category, describe
from duckmind.duckcore.routines import (
    FAMILY_KINDS,
    ROUTINES,
    FamilyProfile,
    Household,
    HouseholdV2,
    make_profile,
)
from duckmind.expression_sweep.envelope import read_caps
from duckmind.worldport import CoreTape, TapePort, WorldPort

if TYPE_CHECKING:
    from duckmind.engines.v4_0.duckcore import api as v4_0

ACT_METHODS = {
    "touch",
    "hold_head",
    "pickup",
    "hand_near",
    "say",
    "call_name",
    "tv",
    "pad_good",
    "pad_guest",
    "pad_play",
    "owner_hello",
    "owner_good",
    "owner_enough",
    "owner_not_that",
    "power_on",
    "power_off",
    "wake",
    "set_upright",
    "swap_battery",
    "end_quiet",
    "nap_now",
    "owner_walk",
    "hear_home",
    "family_leave",
    "family_return",
    "goodnight",
    "place_on_nest",
    "ack_nest_trouble",
    "set_roam_radius",
    "nest_moved",
    # Curiosity's App inputs (curiosity spec §7.2, phase-4 design §9): named here, but refused
    # (ValueError, nothing recorded) until curiosity P2 gives the engine its methods for them.
    "thing_mark",
    "home_zone",
    "family_play",
    "answer_ask",
}
PERSONA_METHODS = {
    "touch",
    "hold_head",
    "pickup",
    "hand_near",
    "say",
    "call_name",
    "pad_good",
    "pad_guest",
    "pad_play",
    "power_on",
    "power_off",
    "wake",
    "set_upright",
    "swap_battery",
    "nap_now",
    "hear_home",
    "goodnight",
    "place_on_nest",
}
MUTATING = {
    "act",
    "advance",
    "live",
    "party",
    "set_setting",
    "choose",
    "undo_promotion",
    "forget",
    "forget_by_query",
    "undo_forget",
    "command",
    # Percepts handed in from outside (the real duck's body.sock reader, a live simulator's
    # duck-body): each is checked against percept-v1 and taken in at its second.
    "sense",
    # The neural duck (growth integration G2): "bridge" attaches this Duck Core to it (from then
    # on it simulates no body and no experiences of its own), "experience" gives it what the
    # neural duck lived, each event checked against bridge-v1 and taken in at its second.
    "bridge",
    "experience",
}

# The revisions are duckmind.engines' (one registry: tests/test_engine_frozen.py checks that
# this engine's ENGINE_REV is its CURRENT).
FROZEN_REV = engines.OLD_SAVE  # the v4 engine, frozen in duckmind.engines.v4_0
ENGINE_REVS = engines.REVS
LEGACY_WORLD = "legacy"  # v4's physics, outside the mind (duckmind.world.legacy)
LEGACY_WORLD_V = 1
# Settings the world's sensor models follow too (LegacyWorld's touch detector: conservative;
# the ear is hardware, fixed in the Config and the header).
WORLD_SETTINGS = frozenset({"conservative"})
# How far ahead of Duck Core's clock a percept handed in may lie (api.sense lives the duck on to
# it): a sender whose clock is far ahead is refused, not followed for hours.
SENSE_AHEAD_S = 60
# The hands that move the duck: the world is told for its truth stream (Duck Core itself
# tells the world to stop a walk or the righting tries, boundary §3).
HANDS = ("pickup", "set_upright", "place_on_nest", "power_off", "pad_play")


@dataclass(slots=True)
class Config:
    """What the duck is (Duck Core's side only: the world's rates are in the world header)."""

    seed: int = 7
    name: str = "小鴨"
    tof: bool = False
    pickup_detect: bool = True
    ear: bool = True
    # The simulator assumes the name model has passed (like pickup_detect); the real
    # HardwareConfig keeps it off until the trained model is evaluated.
    name_ear: bool = True
    # Likewise it assumes standing up and slow walks have passed the physics sweep; the
    # real robot keeps them off until the A100 validation.
    motion_validated: bool = True
    conservative: bool = True
    # A caps file written by duckmind.expression_sweep.envelope.write_caps; empty = none.
    caps_file: str = ""
    # The simulated family swaps in a fresh pack when the battery gets too low to walk, so
    # a battery duck keeps living instead of switching off for the rest of the day.
    battery_help: bool = True
    # Duck Core v4: a charging nest it docks on by itself (simulator only; the real robot
    # keeps it off until the nest hardware exists). The page hatches new ducks with one.
    nest: bool = False
    camera: bool = True
    nest_charge_per_min: int = 125
    roam_radius_mm: int = 1500
    # Which simulated family lives with it ("legacy": the old visit routines).
    household: str = "legacy"
    # The engine it hatched with (design A6): a saved duck without one is a v4_0 duck.
    engine_rev: str = ENGINE_REV
    # When it hatched, as an ISO-8601 time with its offset ("2026-10-11T09:30:00+08:00"); "" = the
    # engine's own DEFAULT_START. A life that follows the calendar (docs/design/GROWTH-G1) hatches
    # at its real birth moment, so its local midnights, its wake-up and its growth days are the
    # family's. Left out of the saved config when empty, so every older save reads as before.
    start: str = ""
    # Curiosity (curiosity spec §7, §13.2; inert sockets until curiosity P2): on in the
    # simulator (the real robot keeps it off until gate C1), the family's toy kit, whether
    # it may carry things (off by default in a home with a toddler: owner decision 6), a
    # toddler at home (gentle mode), and how much further a ball may lead it (0 until the
    # owner decides otherwise, owner decision 3).
    curiosity: bool = True
    home_kit: str = ""
    carry_enabled: bool = True
    toddler: bool = False
    ball_leash_ext_mm: int = 0


def _config_dict(config: Config) -> dict[str, Any]:
    out = {name: getattr(config, name) for name in Config.__dataclass_fields__}
    if not out.get("start"):
        out.pop("start", None)  # (the default hatch moment is not written: saves stay as they were)
    return out


# ---------------------------------------------------------------------------- the frozen engine


@cache
def _v4_0() -> ModuleType:
    """The frozen v4 api (loaded on first use: most sessions never need it)."""
    return import_module("duckmind.engines.v4_0.duckcore.api")


@cache
def v4_world_fields() -> dict[str, Any]:
    """The v4 ``Config`` fields that are the world's now (its outcome rates), with v4's
    defaults: the frozen v4 Config's fields this Config no longer has (design A5). Read from
    the frozen engine, so Duck Core never names them."""
    frozen = _v4_0().Config()
    mine = set(Config.__dataclass_fields__)
    return {
        name: getattr(frozen, name)
        for name in type(frozen).__dataclass_fields__
        if name not in mine
    }


def legacy_world(config: Config, rates: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The header of the world a v4 duck lived in (LegacyWorld, phase-4 design §6): the
    duck's seed and content table (its draws), v4's outcome rates (``rates``: some of them
    changed, as an old v4 Config's own fields), ``camera`` as the nest-marker camera (the same
    head camera sees the nest's marker) and the sensors' starting settings."""
    defaults = v4_world_fields()
    given = dict(rates or {})
    unknown = sorted(set(given) - set(defaults))
    if unknown:
        raise ValueError(f"not a world rate: {unknown} (the world's rates: {sorted(defaults)})")
    return {
        "world": LEGACY_WORLD,
        "v": LEGACY_WORLD_V,
        "seed": int(config.seed),
        "table_hash": content.table_hash(),
        "rates": {name: int(given.get(name, value)) for name, value in sorted(defaults.items())},
        "marker": bool(config.camera),
        "sensors": {"ear": bool(config.ear), "conservative": bool(config.conservative)},
    }


def check_world(config: Config, header: Mapping[str, Any]) -> None:
    """Refuse a world header that is not this duck's (design A6: a v4_1 duck is Duck Core and
    its world together, and neither may change under a saved duck). LegacyWorld draws with the
    duck's own key, so its seed and content table must be the duck's, its nest marker the duck's
    camera and its detectors' start the duck's ear and mode; another header would silently be
    another duck. A content table that changed means the duck needs the engine it hatched with
    (a new engine revision, never this one)."""
    if header.get("world") != LEGACY_WORLD:
        return  # (another world keeps its own key and its own checks)
    problems = []
    if header.get("v") != LEGACY_WORLD_V:
        problems.append(f"LegacyWorld v{header.get('v')} (this build runs v{LEGACY_WORLD_V})")
    if header.get("seed") != int(config.seed):
        problems.append(f"seed {header.get('seed')!r}, the duck's is {config.seed}")
    table = content.table_hash()
    if header.get("table_hash") != table:
        problems.append(
            f"content table {header.get('table_hash')!r}, this engine's is {table!r} (a duck "
            "hatched with another table needs the engine it hatched with)"
        )
    if header.get("marker") != bool(config.camera):
        problems.append(
            f"nest marker {header.get('marker')!r}, the duck's camera is {config.camera}"
        )
    sensors = header.get("sensors")
    sensors = sensors if isinstance(sensors, Mapping) else {}
    if sensors.get("ear") != bool(config.ear):
        problems.append(f"sensors.ear {sensors.get('ear')!r}, the duck's ear is {config.ear}")
    if "conservative" in sensors and sensors["conservative"] != bool(config.conservative):
        problems.append(
            f"sensors.conservative {sensors['conservative']!r}, the duck starts "
            f"conservative={config.conservative}"
        )
    if problems:
        raise ValueError("this world header is not this duck's: " + "; ".join(problems))


# ---------------------------------------------------------------------------- the family's hands


class LegacyActs:
    """The simulated family's hands on a v4_1 duck (phase-4 design §10, boundary §8).

    ``routines.py`` and the page's actions call it with Duck Core's own method names. Most go
    straight to the duck. The hands whose *detection* v4 rolled as a die go through the
    world's sensor model first (:meth:`touch`, :meth:`hold_head`: LegacyWorld draws
    ``detect.touch`` / ``detect.hold`` with the duck's own key), and Duck Core learns of them
    only as a percept; the TV and the settings the sensors follow are mirrored to the world,
    and the hands that move the duck are told to it for its truth stream (the world's
    schedules stop by Duck Core's own ``duck.stop``). Reading the duck's state (``s``, ``hw``,
    ``_stand_due`` …) passes through: the family reads the duck until phase 6 makes it a
    world agent (design risk 9)."""

    def __init__(self, core: DuckCore, world: WorldPort) -> None:
        self.core = core
        self.world = world

    def __getattr__(self, name: str) -> Any:
        if name in ("core", "world") or name.startswith("__"):
            raise AttributeError(name)  # (not set up yet: never look for them on the duck)
        value = getattr(self.core, name)
        if inspect.ismethod(value):
            # (bound methods are kept: the family calls advance_seconds, hear_home … all day)
            self.__dict__[name] = value
        return value

    @property
    def s(self) -> CoreState:
        return self.core.s

    @property
    def hw(self) -> HardwareConfig:
        return self.core.hw

    @property
    def log(self) -> EventLog:
        return self.core.log

    def _sensor(self, name: str) -> Callable[..., dict[str, Any] | None]:
        model = getattr(self.world, name, None)
        if model is None:
            raise ValueError(f"this world ({self.world.src}) has no family hands to {name}")
        return cast(Callable[..., "dict[str, Any] | None"], model)

    def touch(
        self, persona: str, *, seconds: int = 5, where: str = "crown", firmness: str = "normal"
    ) -> dict[str, Any]:
        core = self.core
        if not core.needs_awake(persona, "touch"):
            return {"perceived": False, "note": "關機中"}
        part = "頭頂" if where == "crown" else "嘴巴"
        sensed = self._sensor("touch")(
            persona, seconds=seconds, where=where, firmness=firmness, now_ms=core.now_ms()
        )
        if sensed is None:
            core.journal_entry(
                persona, "touch", f"摸{part} {seconds} 秒 → 沒被偵測到（偵測器漏掉）", False, []
            )
            core.spend_input(seconds)
            return {"perceived": False, "note": "偵測器漏掉這次觸摸"}
        mark = len(core.journal)
        taken = core.on_percept(sensed)
        _journal_at(core, mark, persona, "touch", f"摸{part} {seconds} 秒・小鴨只知道有人摸", taken)
        core.spend_input(seconds)
        return {"perceived": True, "note": taken.get("note")}

    def hold_head(self, persona: str) -> dict[str, Any]:
        core = self.core
        if not core.needs_awake(persona, "hold_head"):
            return {"perceived": False, "note": "關機中"}
        s = core.s
        # The head moves under the hand unless it is asleep (v4: not napping).
        moving = not (s.napping_until is not None and s.now < s.napping_until)
        sensed = self._sensor("hold_head")(persona, moving=moving, now_ms=core.now_ms())
        if sensed is None:
            core.journal_entry(
                persona, "hold_head", "扶住頭 2 秒 → 沒偵測到（會再用小探測動作確認）", False, []
            )
            core.spend_input(3)
            return {"perceived": False, "note": "沒偵測到"}
        mark = len(core.journal)
        taken = core.on_percept(sensed)
        _journal_at(core, mark, persona, "hold_head", "扶住頭 2 秒 → 停下來", taken)
        core.spend_input(3)
        return {"perceived": True, "note": taken.get("note")}

    def tv(self, on: bool) -> dict[str, Any]:
        mirror = getattr(self.world, "tv", None)
        if mirror is not None:
            mirror(bool(on), now_ms=self.core.now_ms())
        return self.core.tv(on)

    def set_setting(self, key: str, value: bool) -> dict[str, Any]:
        result = self.core.set_setting(key, value)
        mirror = getattr(self.world, "set_setting", None)
        if key in WORLD_SETTINGS and mirror is not None:
            mirror(key, bool(value), now_ms=self.core.now_ms())
        return result

    def _hand(self, kind: str, persona: str | None) -> None:
        told = getattr(self.world, "hand", None)
        if told is not None:
            told(kind, persona=persona, now_ms=self.core.now_ms())

    def pickup(self, persona: str) -> dict[str, Any]:
        self._hand("pickup", persona)
        return self.core.pickup(persona)

    def set_upright(self, persona: str) -> dict[str, Any]:
        self._hand("set_upright", persona)
        return self.core.set_upright(persona)

    def place_on_nest(self, persona: str | None = None) -> dict[str, Any]:
        self._hand("place_on_nest", persona)
        return self.core.place_on_nest(persona)

    def power_off(self, persona: str | None, **kwargs: Any) -> dict[str, Any]:
        self._hand("power_off", persona)
        return self.core.power_off(persona, **kwargs)

    def pad_play(self, persona: str, **kwargs: Any) -> dict[str, Any]:
        self._hand("pad_play", persona)
        return self.core.pad_play(persona, **kwargs)


def _journal_at(
    core: DuckCore, mark: int, persona: str, action: str, detail: str, taken: Mapping[str, Any]
) -> None:
    """The family's line in the truth journal for a sensed hand, where v4 wrote it: before
    whatever the duck's reaction wrote (``mark``: the journal's length before the percept)."""
    event_id = taken.get("event")
    core.journal_entry(persona, action, detail, True, [str(event_id)] if event_id else [])
    if mark < len(core.journal) - 1:
        core.journal.insert(mark, core.journal.pop())


# ---------------------------------------------------------------------------- sessions


def _build_core(config: Config, port: WorldPort | None, tape: CoreTape | None) -> DuckCore:
    """The Duck Core a Config describes, on ``port`` (the same for a replay on a tape)."""
    core = DuckCore(
        seed=config.seed,
        name=config.name,
        start=datetime.fromisoformat(config.start) if config.start else None,
        hardware=HardwareConfig(
            tof=config.tof,
            pickup_detect=config.pickup_detect,
            ear=config.ear,
            name_ear=config.name_ear,
            motion_validated=config.motion_validated,
            nest=config.nest,
            camera=config.camera,
            nest_charge_per_min=config.nest_charge_per_min,
        ),
        conservative=config.conservative,
        expression_caps=read_caps(Path(config.caps_file)) if config.caps_file else None,
        battery_help=config.battery_help,
        day_aware=config.household != "legacy",
        port=port,
        tape=tape,
    )
    if config.nest and config.roam_radius_mm != 1500:
        core.s.roam_radius_mm = max(1000, min(2750, config.roam_radius_mm))
    return core


@dataclass(slots=True)
class Session:
    """A v4_1 duck, the world it lives in, its simulated family and its recorded inputs.

    ``world`` is the world header (None: :func:`legacy_world` for the config — v4's physics
    with v4's default rates). ``keep_tape`` records Duck Core's input tape and the world's own
    tape for :func:`percept_replay_check` and :func:`world_replay_check` (the replay safety
    net, about 3% of Duck Core's CPU; a population run turns it off).
    """

    config: Config = field(default_factory=Config)
    world: dict[str, Any] | None = None
    keep_tape: bool = True
    core: DuckCore = field(init=False)
    port: WorldPort = field(init=False)
    hands: LegacyActs = field(init=False)
    household: Household = field(init=False)
    inputs: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    family: HouseholdV2 | None = field(default=None, init=False)
    # A duck that moved here from an older engine (:func:`migrate`): the config it hatched
    # with there. Its inputs begin there, with a ``migrate`` input at the seam.
    origin: dict[str, Any] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.config.engine_rev != ENGINE_REV:
            raise ValueError(
                f"Session builds {ENGINE_REV} ducks; a {self.config.engine_rev!r} duck opens with "
                "api.open_session"
            )
        header = legacy_world(self.config) if self.world is None else dict(self.world)
        check_world(self.config, header)
        self.port = worldport.make_world(header)
        self.world = self.port.header()
        if not self.keep_tape and hasattr(self.port, "keep_tape"):
            self.port.keep_tape = False  # (LegacyWorld's own tape: world_replay_check's input)
        tape = (
            CoreTape(
                header={
                    "engine_rev": ENGINE_REV,
                    "config": _config_dict(self.config),
                    "world": self.world,
                }
            )
            if self.keep_tape
            else None
        )
        self.core = _build_core(self.config, self.port, tape)
        self.hands = LegacyActs(self.core, self.port)
        self.household = Household(self.config.seed ^ 0x5EED)


if TYPE_CHECKING:
    AnySession = Session | v4_0.Session


def _world_arg(world: Any) -> dict[str, Any] | None:
    if world is None:
        return None
    if not isinstance(world, Mapping):
        raise TypeError(f"world must be a world header (a JSON object), not {type(world).__name__}")
    return dict(world)


def _strict(data: Mapping[str, Any], fields: Iterable[str], rev: str) -> None:
    unknown = sorted(set(data) - set(fields))
    if unknown:
        # No silent loss: an old duck whose field is dropped would quietly become another duck.
        raise ValueError(f"engine {rev} has no Config field {unknown}")


def open_session(config: Config | Mapping[str, Any], world: Any = None) -> AnySession:
    """The duck a saved config describes, on the engine it hatched with (no inputs yet).

    * no ``engine_rev`` (every save before phase 4, 蛋黃 included) or ``"v4_0"``: the frozen v4
      engine with that Config exactly (``world`` is ignored: a v4 duck's world is its Config —
      which also lets ``replay_golden.py --factory …:open_session`` pass its corpus directory);
    * ``"v4_1"``: this engine; the v4 rate fields, if the dict has any, become the world
      header's (:func:`legacy_world`) unless a header is given. An unknown field is an error.
    """
    if isinstance(config, Config):
        if config.engine_rev == FROZEN_REV:
            raise ValueError("a v4_0 duck opens from its saved config (a dict without engine_rev)")
        return Session(config, world=_world_arg(world))
    data = dict(config)
    rev = str(data.pop("engine_rev", None) or FROZEN_REV)
    if rev == FROZEN_REV:
        frozen = _v4_0()
        _strict(data, frozen.Config.__dataclass_fields__, rev)
        return cast("v4_0.Session", frozen.Session(frozen.Config(**data)))
    if rev != ENGINE_REV:
        raise ValueError(f"unknown engine {rev!r} (this build runs {', '.join(ENGINE_REVS)})")
    rates = {key: data.pop(key) for key in list(data) if key in v4_world_fields()}
    _strict(data, Config.__dataclass_fields__, rev)
    ours = Config(engine_rev=rev, **data)
    header = _world_arg(world)
    if header is not None and rates:
        raise ValueError(f"{sorted(rates)} belong to the world header, which was given too")
    return Session(ours, world=legacy_world(ours, rates) if header is None else header)


class SaveContentMismatch(ValueError):
    """A save written on another content version of this engine: replaying it here would give
    another duck, so it is refused (never replayed silently). ``written`` is the version block
    the save carries, ``now`` this engine's (the server says it in plain words)."""

    def __init__(self, written: Mapping[str, Any], now: Mapping[str, Any]) -> None:
        super().__init__(dict(written), dict(now))
        self.written = dict(written)
        self.now = dict(now)


def content_version() -> dict[str, str]:
    """What this engine's draws and rules are made of: the engine revision, the content table's
    name and the table's hash (it is part of every random key, ``content.table_hash``). Written
    into every v4_1 save (``save_json``) and checked before a replay (``restore``): a tuning change
    to ``content.py`` changes the hash, and an older save then says so instead of replaying into
    another duck. The way forward for such a change is the one the frozen v4 engine took: copy
    the old engine into ``duckmind.engines`` first (``tools/freeze_engine.py``), so the old saves
    replay on their own constants."""
    return {
        "engine_rev": ENGINE_REV,
        "version": content.CONTENT_VERSION,
        "table_hash": content.table_hash(),
    }


class SaveBridgeMismatch(ValueError):
    """A save whose neural-duck input vocabulary is not one this build knows (written by a newer
    ``bridge-vN``): replaying it would drop or misread its experiences, so it is refused."""

    def __init__(self, written: str, now: str) -> None:
        super().__init__(written, now)
        self.written = written
        self.now = now


def check_save_bridge(saved: Mapping[str, Any]) -> None:
    """Refuse a save written with a ``bridge`` vocabulary this build does not read (a save
    without the block was never attached, and replays as it always did)."""
    block = saved.get("bridge")
    if not isinstance(block, Mapping):
        return
    if block.get("api") not in bridge_v2.APIS:
        raise SaveBridgeMismatch(str(block.get("api")), bridge_v2.BRIDGE_API)


def check_save_content(saved: Mapping[str, Any]) -> None:
    """Refuse a save whose content version is not this engine's (a save without the block is an
    older one: the world header's table hash still guards it, :func:`check_world`)."""
    written = saved.get("content")
    if not isinstance(written, Mapping):
        return
    now = content_version()
    if dict(written) != now:
        raise SaveContentMismatch(written, now)


def restore(saved: Mapping[str, Any], **config: Any) -> AnySession:
    """A saved duck (``{"config", "world"?, "inputs"}``, as :func:`save_json` writes it) with
    every input replayed and recorded again (a ``migrate`` input moves it to the newer engine
    right there, :func:`migrate`). ``config`` overrides Config fields first (the simulator
    applies today's expression caps: ``caps_file=…``). A save written on another content version
    is refused first (:func:`check_save_content`)."""
    check_save_content(saved)
    check_save_bridge(saved)
    session = open_session({**dict(saved["config"]), **config}, saved.get("world"))
    for item in saved.get("inputs", ()):
        method, params = str(item["method"]), dict(item["params"])
        if method == MIGRATE:
            session = migrate(session, str(params.get("to", ENGINE_REV)))
        else:
            dispatch(session, method, params)
    return session


def save_json(session: AnySession) -> dict[str, Any]:
    """What restores the duck: its config, its world header and its inputs (a migrated duck:
    the config it hatched with on its old engine, and every input since)."""
    origin = session.origin if isinstance(session, Session) else None
    return {
        "config": dict(origin) if origin is not None else config_json(session),
        "world": world_json(session),
        "inputs": [{"method": name, "params": args} for name, args in session.inputs],
        # (a v4_1 duck's draws depend on the content table; a frozen v4 duck's engine is its own)
        **({"content": content_version()} if isinstance(session, Session) else {}),
        # (a duck attached to the neural duck says which input vocabulary it was fed in:
        # bridge-v1 until it is given a kind only bridge-v2 has)
        **(
            {"bridge": {"api": bridge_v2.api_of(session.inputs)}}
            if isinstance(session, Session) and session.core.s.bridged
            else {}
        ),
    }


MIGRATE = "migrate"  # the input that marks where a duck moved to a newer engine


def _migrator() -> Callable[[Any, str], Session]:
    """``duckmind.engines.migrate.migrate(old_session, to) -> Session`` (phase-4 WP13)."""
    return cast("Callable[[Any, str], Session]", import_module("duckmind.engines.migrate").migrate)


def migrate(session: AnySession, to: str = ENGINE_REV) -> Session:
    """Move a duck to a newer engine, memories and all (the WP13 seam; owner question Q1, as
    recommended: 蛋黃 moves to the new mind with its memories). ``duckmind.engines.migrate``
    replays the old duck to this moment and hands its state, chain and draw counters to the new
    engine (the world's outcome counters to its world); this records the seam: the new session
    keeps the old config as its ``origin`` and its inputs gain a ``migrate`` input, so
    :func:`restore` and ``replay_check`` walk through the same seam. Returns the new session
    (callers replace theirs: the old one is left as it was)."""
    if to == FROZEN_REV or to not in ENGINE_REVS:
        raise ValueError(f"cannot move a duck to {to!r} (this build runs {', '.join(ENGINE_REVS)})")
    if isinstance(session, Session):
        raise ValueError(f"this duck already runs {session.config.engine_rev}")
    moved = _migrator()(session, to)
    if not isinstance(moved, Session):
        raise TypeError("engines.migrate must return an api.Session")
    moved.origin = config_json(session)
    # (copies: a real-time advance coalesced into the old session's last input must never
    # change the moved duck's saved inputs)
    moved.inputs = [*copy.deepcopy(session.inputs), (MIGRATE, {"to": to})]
    return moved


def _act(
    session: Session, method: str, params: dict[str, Any], persona: str | None
) -> dict[str, Any]:
    if method not in ACT_METHODS:
        raise ValueError(f"unknown action {method}")
    fn = getattr(session.hands, method, None)
    if fn is None:  # (curiosity's App inputs need the engine's sockets for them)
        raise ValueError(f"this duck's engine takes no {method} yet")
    kwargs = dict(params)
    if method in PERSONA_METHODS:
        if method in {"say", "hear_home", "goodnight", "place_on_nest"}:
            return cast(dict[str, Any], fn(persona, **kwargs))
        return cast(dict[str, Any], fn(persona or "owner", **kwargs))
    return cast(dict[str, Any], fn(**kwargs))


def _run_command(session: Session, text: str, persona: str) -> dict[str, Any]:
    results = []
    for command in commands.parse(text, persona):
        item: dict[str, Any] = {
            "clause": command.text,
            "kind": command.kind,
            "persona": command.persona,
        }
        if command.kind == "act":
            item["result"] = _act(session, command.method, command.params, command.persona)
        elif command.kind == "advance":
            minutes = (
                session.core.minutes_until(str(command.params["until"]))
                if command.params.get("until")
                else int(command.params["minutes"])
            )
            item["result"] = session.core.advance(minutes)
        elif command.kind == "live":
            item["result"] = session.household.live(
                session.hands, command.params["days"], command.params["routine"]
            )
        elif command.kind == "setting":
            item["result"] = session.hands.set_setting(
                command.params["key"], command.params["value"]
            )
        elif command.kind == "choose":
            item["result"] = session.core.choose(command.params["kind"], command.params["option"])
        elif command.kind == "query":
            item["result"] = query.search(session.core, command.params["q"])
        elif command.kind == "why":
            item["result"] = query.why(session.core, command.params["question"])
        elif command.kind == "diary":
            item["result"] = {"diary": query.diary(session.core, 7)}
        elif command.kind == "status":
            code, text_reason = session.core.still_reason()
            item["result"] = {"code": code, "text": text_reason}
        elif command.kind == "forget":
            item["result"] = _forget_by_query(session, command.params["query"])
        elif command.kind == "undo":
            item["result"] = getattr(session.core, command.method)()
        else:
            item["result"] = {
                "note": "看不懂這句話，試試看："
                + "、".join(command.params.get("suggestions", [])[:6])
            }
        results.append(item)
    return {"results": results}


def _forget_by_query(session: Session, text: str) -> dict[str, Any]:
    found = query.search(session.core, text, limit=1)
    if not found["facts"]:
        return {"ok": False, "note": "找不到符合的記憶"}
    return session.core.forget(found["facts"][0]["fact_id"])


MORNING = "07:00"
MORNING_LABEL = "跳到早上 7 點（看牠起床走走）"
# advance {"until": NAP_END}: live on until the nap it is taking ends (worked out on the
# duck's own clock when the input is applied, so a replay lands on the same moment).
NAP_END = "nap_end"


def _call(method: str, **params: Any) -> dict[str, Any]:
    return {"method": method, "params": params}


def _do(action: str, **params: Any) -> dict[str, Any]:
    return _call("act", action=action, params=params)


def _setting(key: str, value: bool) -> dict[str, Any]:
    return _call("set_setting", key=key, value=value)


def _button(label: str, *calls: dict[str, Any], primary: bool = False) -> dict[str, Any]:
    return {"label": label, "calls": list(calls), "primary": primary}


def _grow_to_first_steps(core: Any) -> dict[str, Any] | None:
    """One click to the growth day it can walk: the family's floor routine for as many
    days as it takes, then morning, on, and on the floor again."""
    s = core.s
    if s.gd >= content.FIRST_STEPS_GD:
        return None
    today = 2 if s.day_awake_s >= 15 * 60 else 1
    days = max(1, -(-(2 * content.FIRST_STEPS_GD - s.gd2 - today) // 2))
    if core.hw.nest:
        # A duck with a nest lives the whole day: a family that is home all day, then 9:00.
        return _button(
            f"過到第 {content.FIRST_STEPS_GD} 個成長日（和長輩在家的家庭快轉 {days} 天）",
            _call("live", family="elder_home", days=days),
            _call("advance", until="09:00"),
        )
    return _button(
        f"過到第 {content.FIRST_STEPS_GD} 個成長日（快轉 {days} 天）",
        _call("live", routine="floor", days=days),
        _call("advance", until="08:00"),
        _do("power_on"),
        _setting("floor", True),
    )


def _when(moment: Any, now: Any) -> str:
    """A past moment on the duck's clock, as the banner says it: 23:59, or 09/25 23:59 when
    it was another day."""
    if moment is None:
        return ""
    return moment.strftime("%H:%M" if moment.date() == now.date() else "%m/%d %H:%M")


def _nest_off_banner(core: Any, text: str, night: bool, result: Any) -> dict[str, Any]:
    """Switched off, with a nest: the battery is its own and charges only in the nest (a
    switched-on duck with an empty pack would only switch off again), so the ways forward
    go through the nest."""
    s = core.s
    battery = s.battery * 100 // SCALE
    if s.nested:
        text += f"牠坐在窩裡，關機中也在充電（電量 {battery}%）。"
    else:
        text += f"牠不在窩裡，電量 {battery}%：放回窩裡就會充電。"
    if night:
        text += "現在是晚上：開機後會在窩裡睡覺，早上 7 點自己醒來。"
    home = [] if s.nested else [_do("place_on_nest")]
    return result(
        "off",
        "關機中",
        text,
        _button(
            "開機並跳到早上 7 點" + ("" if s.nested else "（先放回窩裡充電）"),
            *home,
            _call("advance", until=MORNING),
            _do("power_on"),
            primary=True,
        )
        if night
        else {},
        _button(
            "開機（在窩裡充電）" if s.nested else "放回窩裡充電、開機",
            *home,
            _do("power_on"),
            primary=not night,
        ),
        {} if s.nested else _button("只放回窩裡充電（先不開機）", _do("place_on_nest")),
    )


def banner(session: AnySession) -> dict[str, Any]:
    """What the duck is doing and why, in plain words, with one-click ways forward.

    The page shows this over the 3D view. Every button is a list of ordinary recorded
    calls (``dispatch``), so what it does is Duck Core's own input and replays exactly.
    """
    if not isinstance(session, Session):
        return cast(dict[str, Any], _v4_0().banner(session))
    core = session.core
    s = core.s
    clock = s.now.strftime("%H:%M")
    night = core.night()
    battery = s.battery * 100 // SCALE
    tags = []
    if s.conservative:
        tags.append("保守模式")
    if s.guest_until is not None and s.now < s.guest_until:
        tags.append("訪客模式（不學新東西）")
    if s.battery_help and not s.docked and not core.hw.nest:
        tags.append("家人會幫牠換電池")  # (a duck with a nest charges itself instead)
    if s.docked:
        tags.append(f"插著電線（{battery}%）")
    if core.hw.nest:
        tags.append("有窩（自己回窩充電）")
    if core.aware:
        tags.append(daycycle.FAMILY_ZH.get(s.family, s.family))
    recent = s.last_moment
    if recent is not None and s.now - recent["at"] <= timedelta(minutes=3):
        # A clever or goofy moment just now: the banner names it for a little while.
        moment = content.MOMENT_BY_ID.get(str(recent["id"]))
        if moment is not None:
            tags.append(
                f"剛剛的{'機智' if moment.kind == 'clever' else '呆萌'}時刻：{moment.short}"
            )
    blocks = core.walk_blocks()
    code = blocks[0] if blocks else "ok"

    def result(tone: str, title: str, text: str, *buttons: dict[str, Any]) -> dict[str, Any]:
        return {
            "code": code,
            "tone": tone,
            "title": title,
            "text": text,
            "night": night,
            "actions": [button for button in buttons if button],
            "tags": tags,
            "blocks": blocks,
        }

    floor_again = [_setting("floor", True)] if s.floor_when_off else []
    back = "（放回地上）" if s.floor_when_off else ""
    if not s.awake:
        flat = s.off_reason == "battery_empty"
        why = (
            "沒電了，坐下關機"
            if flat
            else "家人讓牠關機休息了"
            if s.boot_seq > 1 or s.off_reason
            else "關機中"
        )
        # When it was switched off (not the clock now: a routine may have switched it off
        # hours ago), and the time now when that is another minute.
        off_at = _when(s.last_power_off, s.now)
        text = (
            f"{why}（{off_at} 關機{f'；現在 {clock}' if off_at != clock else ''}）。"
            if off_at
            else f"{why}（現在 {clock}）。"
        )
        if core.hw.nest:
            return _nest_off_banner(core, text, night, result)
        if flat and not s.battery_help:
            text += "沒人幫忙換電池時，一顆電池大約 1 小時就用完。"
        if night:
            text += "現在是晚上：開機後也只會坐著動頭，早上 7 點以後才會站起來走走。"
        helper = flat and not s.battery_help
        buttons = [
            _button(
                f"開機並跳到早上 7 點{back}",
                _call("advance", until=MORNING),
                _do("power_on"),
                *floor_again,
                primary=True,
            )
            if night
            else None,
            _button(
                f"讓家人幫忙換電池，開機{back}",
                _setting("battery_help", True),
                _do("power_on"),
                *floor_again,
                primary=not night,
            )
            if helper
            else None,
            _button(
                f"{'只換這次電池' if helper else '換電池'}開機{back}",
                _do("power_on"),
                *floor_again,
                primary=not night and not helper,
            ),
            _button("插電開機（不會沒電）", _setting("docked", True), _do("power_on")),
        ]
        return result("off", "關機中", text, *[b for b in buttons if b])
    if code == "righting":
        return result(
            "warn",
            "跌倒了，正在試著自己翻回來",
            "走路時跌了一跤，最多會自己試 3 次。",
            _button("扶起來（還在地上）", _do("set_upright"), primary=True),
        )
    if code == "fallen":
        return result(
            "warn",
            "跌倒了，自己翻不回來",
            "側躺著等人扶。牠沒事，偶爾會掙扎一下。",
            _button("扶起來（還在地上）", _do("set_upright"), primary=True),
        )
    if code == "hot":
        return result(
            "warn",
            "模擬過熱中",
            "太熱了：休息姿勢、很少出聲，不站不走（真機要等溫度降下來）。",
            _button("降溫", _setting("hot", False), primary=True),
        )
    if code == "yield":
        return result(
            "info",
            "手把駕駛中",
            "Duck Core 讓位，只在旁邊看。",
            _button("結束手把，讓牠自己來", _do("end_quiet"), primary=True),
        )
    if code in ("lockout", "stop", "quiet"):
        until = {"lockout": s.lockout_until, "stop": s.stop_until, "quiet": s.quiet_until}[code]
        left = core.left_min(until)
        title, text = {
            "lockout": (
                "安靜中（被叫停太多次）",
                f"10 分鐘內被扶住頭 3 次，先安靜 30 分鐘（還剩 {left} 分）。不是懲罰。",
            ),
            "stop": ("被扶住頭，停下來了", "60 秒沒碰就恢復。"),
            "quiet": (
                "安靜中（有人說「夠了」）",
                f"還剩 {left} 分：不站不走、不出聲，也不會因此學到不喜歡。",
            ),
        }[code]
        return result(
            "rest",
            title,
            text,
            _button("好了，可以動了", _do("end_quiet"), primary=True),
            _button(f"過 {left} 分鐘", _call("advance", minutes=left)) if code != "stop" else {},
        )
    if code == "nap":
        left = core.nap_left_min()
        if s.nap_reason == "away_rest":
            where = "窩裡" if s.nested else "插著電線" if s.docked else "原地"
            return result(
                "sleep",
                "家裡沒人，在休息",
                f"牠覺得家裡沒人（{clock}），就在{where}睡一覺，大約再 {left} 分鐘醒來伸個懶腰；"
                "你們回來時，牠會醒來迎接。",
                _button("家人回來了", _do("family_return"), primary=True),
                _button(f"過 {left} 分鐘", _call("advance", minutes=left)),
            )
        if s.nap_reason == "quiet_hours" and core.aware:
            where = "在窩裡" if s.nested else "插著電線" if s.docked else "在窩外（來不及回窩）"
            return result(
                "sleep",
                f"晚上{where}睡覺",
                f"現在 {clock}。晚上 9 點半牠會自己回窩、坐好說晚安，"
                "睡著時偶爾會動一動、說夢話；早上 7 點會自己醒來。想看牠動，就跳到早上。",
                _button(MORNING_LABEL, _call("advance", until=MORNING), primary=True),
                _button("叫醒牠一下（晚上只會坐著動頭）", _do("wake")),
                _button("抱牠回窩", _do("place_on_nest")) if core.hw.nest and not s.nested else {},
            )
        if s.nap_reason == "quiet_hours":
            woke = s.napping_until.strftime("%H:%M") if s.napping_until else ""
            where = "插著電" if s.docked else ""
            return result(
                "sleep",
                f"晚上{where}在睡覺",
                f"現在 {clock}。晚上 10 點到早上 7 點，{'插電的小鴨會睡覺' if s.docked else '牠在睡覺'}，"
                f"每小時醒來一下；下次約 {woke}。睡著時偶爾會動一動、說夢話；"
                "早上 7 點會自己起床走走。想看牠動，就跳到早上。",
                _button(MORNING_LABEL, _call("advance", until=MORNING), primary=True),
                _button("叫醒牠一下（晚上只會坐著動頭）", _do("wake")),
            )
        if s.nap_reason == "bedtime" and core.hw.nest:
            # (With a nest the battery charges in the nest, not by swapping or a cable.)
            return result(
                "sleep",
                "電量低，打起瞌睡了",
                f"電量 {battery}%："
                + (
                    "在窩裡充電，充飽一點就會醒來。"
                    if s.nested
                    else "放回窩裡就會充電；不管牠的話電量到 3% 會關機。"
                ),
                _button("放回窩裡充電", _do("place_on_nest"), primary=True) if not s.nested else {},
                _button(f"過 {left} 分鐘", _call("advance", minutes=left), primary=s.nested),
            )
        if s.nap_reason == "bedtime":
            return result(
                "sleep",
                "電量低，打起瞌睡了",
                f"電量 {battery}%：換新電池就會醒來；不換的話電量到 3% 會關機。",
                _button("換新電池", _do("swap_battery"), primary=True),
                _button("插電", _setting("docked", True)),
            )
        # Woken now it would still be too tired to stand (it would only sit, and doze off
        # again): the way forward is to let it sleep its fill, and the wake button says so.
        still_tired = s.energy < engine.SIT_ENERGY
        return result(
            "sleep",
            "累了，小睡中",
            f"精力 {s.energy * 100 // SCALE}%，睡飽了就會起來走走：大約再 {left} 分鐘醒；"
            "睡著時每幾分鐘會動一動。",
            _button(
                f"讓牠睡飽（過 {left} 分鐘，醒來就會走走）",
                _call("advance", until=NAP_END),
                primary=True,
            ),
            _button(
                "叫醒牠（還很累：醒來也只會坐著休息）" if still_tired else "叫醒牠", _do("wake")
            ),
        )
    if code == "night":
        text = f"現在 {clock}。晚上 10 點到早上 7 點，牠只會坐著動頭，不站起來、不走。"
        if s.docked:
            text += "插著電時，沒人陪牠就會去睡覺。"
        return result(
            "night",
            "晚上了，不走路",
            text,
            _button(MORNING_LABEL, _call("advance", until=MORNING), primary=True),
        )
    if code == "battery" and core.hw.nest and s.nested:
        # In its nest it is charging already: no battery to swap, no cable to plug in.
        gain = max(1, core.hw.nest_charge_per_min - core.hw.nest_awake_drain)
        minutes = max(1, -(-(engine.MIN_MOTION_BATTERY - s.battery) // gain))
        return result(
            "rest",
            "在窩裡充電",
            f"電量 {battery}%：低於 30% 先坐在窩裡充電，不站不走；大約 {minutes} 分鐘後能再站起來，"
            "充到一半以上就會自己出來走走。",
            _button(
                f"讓牠充電（過 {minutes} 分鐘）", _call("advance", minutes=minutes), primary=True
            ),
        )
    if code == "battery" and core.hw.nest:
        return result(
            "warn",
            "需要幫忙：抱牠回窩",
            f"電量 {battery}%，低於 30% 就不走了：牠坐下來請你們抱牠回窩（在窩裡才會充電）。",
            _button("抱牠回窩", _do("place_on_nest"), primary=True),
            _button("換新電池", _do("swap_battery")),
        )
    if code == "battery":
        return result(
            "warn",
            "電量低，不站不走",
            f"電量 {battery}%：低於 30% 只會坐著動頭。",
            _button("換新電池", _do("swap_battery"), primary=True),
            _button("插電（不走路）", _setting("docked", True)),
        )
    nest_banner = _nest_banner(core, code, result) if core.aware else None
    if nest_banner is not None:
        return nest_banner
    young = "too_young" in blocks
    grow = _grow_to_first_steps(core) if young else None
    if code == "desk":
        calls = [_setting("floor", True)]
        label = "放到地上"
        if s.docked:
            calls.append(_setting("docked", False))
            label = "放到地上、拔掉電線（用電池）"
        text = "在桌上只會動頭、站起來坐下，不會走。"
        if young:
            text += f"牠還是雛鴨，第 {content.FIRST_STEPS_GD} 個成長日才會學走路。"
        return result("info", "在桌上", text, _button(label, *calls, primary=True), grow or {})
    if code == "tethered":
        return result(
            "info",
            "在地上，但插著電線",
            f"為了安全，插著電線不會走（電量 {battery}%）。拔掉電線就用電池走走，約 1 小時一顆"
            + ("；家人會幫牠換電池。" if s.battery_help else "。"),
            _button(
                "拔掉電線（用電池）",
                _setting("docked", False),
                *([_do("swap_battery")] if battery < 30 else []),
                primary=True,
            ),
            grow or {},
        )
    if code in ("no_pickup_sensor", "unvalidated"):
        fixes = [
            _button(
                "模擬：當作已通過物理驗證",
                _setting("motion_validated", True),
                primary=True,
            )
            if not core.hw.motion_validated
            else None,
            _button(
                "模擬：裝上抱起偵測",
                _setting("pickup_detect", True),
                primary=core.hw.motion_validated,
            )
            if not core.hw.pickup_detect
            else None,
        ]
        return result(
            "info",
            "這隻鴨設定成不會自己走",
            core.walk_reason() + "（孵化時的硬體設定；可以在這裡模擬改好）",
            *[b for b in fixes if b],
        )
    if young:
        eta = core.outing_eta()
        return result(
            "info",
            "還是雛鴨，還不會走",
            f"第 {content.FIRST_STEPS_GD} 個成長日身體才準備好走第一步（現在第 {s.gd} 天）。"
            "在地上時牠會自己站起來、看看四周，再坐下"
            + (f"（下一次大約 {eta} 分鐘內）。" if eta else "。"),
            {**grow, "primary": True} if grow else {},
            _button("叫牠起來看看", _do("owner_walk")) if s.posture == "sit" else {},
        )
    if code == "tired":
        # One click forward: a nap until it is rested (not a nap you then have to wake it
        # from, which only brought it back here, still tired).
        return result(
            "rest",
            "累了，先坐著休息",
            f"精力 {s.energy * 100 // SCALE}%：太累了站不起來。旁邊有人時牠捨不得睡，"
            "沒人吵幾分鐘就會自己打個瞌睡；睡飽了就會起來走走。",
            _button(
                "讓牠睡飽再起來走走（小睡到睡飽）",
                _do("nap_now"),
                _call("advance", until=NAP_END),
                primary=True,
            ),
            _button("過 10 分鐘", _call("advance", minutes=10)),
        )
    return result(
        "ok", *_wandering(core), _button("叫牠現在去走走", _do("owner_walk"), primary=True)
    )


def _nest_banner(core: Any, code: str, result: Any) -> dict[str, Any] | None:
    """The banner for what a duck that lives the family's day is doing (nest, day phases),
    or None when the ordinary banner says it."""
    s = core.s
    clock = s.now.strftime("%H:%M")
    battery = s.battery * 100 // SCALE
    talk = _button("跟牠說話（有人在家）", _do("say", kind="voice"))
    if code == "unplaced":
        return result(
            "info",
            "剛被抱起來",
            "牠還不知道自己被放在哪裡，所以先不走：放回窩裡，或放到地上（窩前面）。",
            _button("放回窩裡", _do("place_on_nest"), primary=True),
            _button("放到地上", _setting("floor", True)),
        )
    if code == "stranded":
        why = (s.stranded or {}).get("reason")
        text = {
            "battery": f"電量 {battery}%，低於 30% 就不走了：牠坐下來請你們抱牠回窩。",
            "dock_failed": "試了三次都沒坐進窩，坐在窩旁邊等人幫忙（半小時後牠也會自己再試）。",
        }.get(str(why), "晚上來不及回窩，就在原地睡了。")
        return result(
            "warn",
            "需要幫忙：抱牠回窩",
            text,
            _button("抱牠回窩", _do("place_on_nest"), primary=True),
        )
    if code == "alone":
        return result(
            "rest",
            "家裡沒人，在窩裡休息",
            f"牠覺得家人出門了（{clock}）：回窩睡覺、偶爾醒來伸懶腰，不會自己出去走。"
            "學會你們幾點回家以後，會提早醒來看著門口。",
            _button("家人回來了", _do("family_return"), primary=True),
            _button("過 1 小時", _call("advance", minutes=60)),
        )
    if code == "family_quiet":
        return result(
            "rest",
            "最近沒聽到家人的聲音",
            "牠只在知道有人在家時才出門走走；很久沒聽到聲音，就在窩旁休息。跟牠說說話，牠就會出來。",
            _button("跟牠說話", _do("say", kind="voice"), primary=True),
            _button("家人出門了", _do("family_leave")),
        )
    if code == "door":
        return result(
            "info",
            "剛聽到開關門",
            "牠先停下來聽聽看是不是有人出門了（20 分鐘都沒聲音，就當作大家出門了）。",
            _button("跟牠說話（還有人在）", _do("say", kind="voice"), primary=True),
        )
    if code == "dawn":
        left = core.left_min(s.dawn_until)
        return result(
            "rest",
            "剛睡醒，在窩上賴一下",
            f"早上自己醒來了，先在窩上待一下（{f'大約 {left} 分鐘' if left else '快好了'}），"
            "接著就會出來走走。",
            _button(
                f"過 {max(1, left)} 分鐘", _call("advance", minutes=max(1, left)), primary=True
            ),
        )
    if code == "bedtime":
        return result(
            "night",
            "準備睡覺了",
            "晚上 9 點半（或家人說晚安以後）牠會自己走回窩、坐好，晚上 10 點前睡著；"
            "早上 7 點再自己醒來。",
            _button(MORNING_LABEL, _call("advance", until=MORNING), primary=True),
        )
    if code == "solo_done":
        return result(
            "rest",
            "家人還在睡，牠在窩裡等",
            "家人還在睡的早上，牠只在窩旁小聲地逛一逛；逛夠了就回窩等你們起床。",
            _button("家人起床了（跟牠說話）", _do("say", kind="voice"), primary=True),
        )
    if code == "nest_trouble":
        return result(
            "warn",
            "窩附近跌倒好幾次",
            "牠不會因此怕自己的窩；但在你們檢查地面前，只有家人在旁邊時才進出窩。",
            _button("地面檢查好了", _do("ack_nest_trouble"), primary=True),
            talk,
        )
    if code == "stand_backoff":
        left = core.left_min(s.stand_block_until)
        return result(
            "rest",
            "剛剛沒站起來",
            f"站起來沒站穩，先坐著休息（大約 {left} 分鐘後再試）。",
            _button(
                f"過 {max(1, left)} 分鐘", _call("advance", minutes=max(1, left)), primary=True
            ),
        )
    if code == "charging":
        return result(
            "rest",
            "在窩裡充電",
            f"電量 {battery}%：充到一半以上，牠就會自己出來走走（電越多越想出門）。",
            _button("過 10 分鐘", _call("advance", minutes=10), primary=True),
            _button("叫牠出來走走", _do("owner_walk")),
        )
    if code == "going_home":
        return result(
            "info",
            "電量快到回窩的線了",
            f"電量 {battery}%：牠會自己走回窩充電（永遠留著走回家的電）。",
            _button("抱牠回窩", _do("place_on_nest"), primary=True),
            _button("過 5 分鐘", _call("advance", minutes=5)),
        )
    if code == "ok" and core.hw.nest:
        title, text = _nest_wandering(core)
        return result("ok", title, text, _button("叫牠現在去走走", _do("owner_walk"), primary=True))
    return None


def _nest_wandering(core: Any) -> tuple[str, str]:
    s = core.s
    battery = s.battery * 100 // SCALE
    solo = (
        f"家人還在睡：只在窩旁 {excursion.leash_mm(core) / 1000:.1f} 公尺內、小小聲地逛。"
        if s.phase == "solo"
        else ""
    )
    if s.homing is not None:
        return "正在走回窩", f"{solo}自己找路、面向窩的標籤走進去、坐下充電（電量 {battery}%）。"
    if s.excursion is not None:
        reason = excursion.REASON_ZH.get(str(s.excursion.get("reason")), "走走")
        return "出門走一小趟", f"{solo}{reason}：走幾段、停下來看看，走完再決定回窩或坐一下。"
    if s.nested:
        return (
            "在窩裡",
            f"{solo}電量 {battery}%：電越多越想出門，隨時會站起來走出窩（一天進出窩好幾十次）。",
        )
    return (
        "在窩外坐著",
        f"{solo}電量 {battery}%（回窩線 {nest.threshold(core) * 100 // SCALE}%）："
        "過一兩分鐘又會起來走走；電少了或累了就自己回窩。",
    )


def _wandering(core: Any) -> tuple[str, str]:
    """The ok banner's words: what it is doing and when it will next get up, as things
    stand (DuckCore.outing_eta: the minute its restlessness makes getting up certain)."""
    s = core.s
    energy = s.energy * 100 // SCALE
    tired = f"有點累了（精力 {energy}%），起來的次數少一點；" if energy < 50 else ""
    if s.posture == "walk":
        return "正在走走", "自己起來到處走走、停下來看看；走累了會坐下，過幾分鐘又會起來。"
    if s.posture == "stand":
        return (
            "站著看看四周",
            "看一看會再走幾步，或坐下休息；坐下後過幾分鐘又會自己起來走走。",
        )
    eta = core.outing_eta()
    when = (
        "過一會兒會起來"
        if eta is None
        else "隨時會起來"
        if eta <= 1
        else f"大約 {eta} 分鐘內會起來"
    )
    return (
        "醒著，坐著看看四周",
        f"在地上、白天、電量夠：{tired}牠會自己站起來走走，下一次{when}"
        "（坐得越久越想起來）。等不及就按下面的按鈕叫牠。",
    )


def snapshot(session: AnySession) -> dict[str, Any]:
    if not isinstance(session, Session):
        return {**_v4_0().snapshot(session), "engine_rev": FROZEN_REV}
    core = session.core
    s = core.s
    stage = core.stage()
    code, reason = core.still_reason()
    last_expression = next(
        (event for event in reversed(core.log.events) if event.kind == "duck.expression"), None
    )
    flag = s.last_flag
    trying = None
    if flag is not None and s.now - flag["at"] <= timedelta(seconds=3):
        # What praise within 3 s would be for: a new phrase, a move, or a clever or goofy
        # moment (the last one's flag is a moment id, not a verb).
        if flag["kind"] == "phrase":
            text = s.phrases[flag["id"]]["text"]
        elif flag["kind"] == "moment":
            moment_flag = content.MOMENT_BY_ID.get(str(flag["id"]))
            text = moment_flag.short if moment_flag else "一段小插曲"
        else:
            verb_flag = content.VERB_BY_ID.get(str(flag["id"]))
            text = verb_flag.zh if verb_flag else "一個動作"
        trying = {"kind": flag["kind"], "id": flag["id"], "text": text}
    total_care = sum(s.care_window.values())
    compass = None
    if s.gd >= 28 or s.branch is not None:
        compass = {
            branch.branch_id: {
                "zh": branch.zh,
                "care": content.CARE_ZH[branch.care_type],
                "count": s.care_window[branch.care_type],
                "share": s.care_window[branch.care_type] * 100 // total_care if total_care else 0,
                "nature": s.nature[branch.branch_id] * 100 // SCALE,
            }
            for branch in content.BRANCHES
        }
    prefs: list[dict[str, Any]] = [
        {"item": key.split(".", 1)[1], "value": value * 100 // SCALE}
        for key, value in sorted(s.prefs.items(), key=lambda pair: (-pair[1], pair[0]))[:8]
    ]
    for pref in prefs:
        item = str(pref["item"])
        if item.startswith("verb:"):
            verb = content.VERB_BY_ID.get(item[5:])
            pref["zh"] = f"動作「{verb.zh if verb else item}」"
        elif item.startswith("moment:"):
            moment = content.MOMENT_BY_ID.get(item[7:])
            pref["zh"] = f"時刻「{moment.short if moment else item}」"
        else:
            phrase = s.phrases.get(item[7:], {})
            pref["zh"] = f"片語「{phrase.get('text', item)}」"
    milestones = [
        {
            "id": key,
            "title": info["title"],
            "status": info["status"],
            "text": info.get("text", ""),
            "gd": info.get("debuted_gd", info.get("unlocked_gd")),
        }
        for key, info in sorted(
            s.milestones.items(),
            key=lambda item: (
                item[1].get("debuted_gd", item[1].get("unlocked_gd", item[1]["gd"])),
                item[1]["order"],
                item[0],
            ),
        )
        if info["status"] != "locked"
    ]
    return {
        "identity": {
            "name": s.name,
            "seed": core.seed,
            "table_hash": core.table,
            "hatched_at": s.hatched_at.strftime("%Y-%m-%d %H:%M"),
            "quirks": [
                {"key": key, "text": text}
                for key, text in (
                    ("tilt_side", f"歪頭總是往{s.quirks['tilt_side']}邊"),
                    ("favorite_tag", f"最常說「{content.TAG_ZH[s.quirks['favorite_tag']]}」"),
                    ("habit", s.quirks["habit_text"]),
                )
                if key in s.revealed
            ],
            "hidden_quirks": 3
            - len([key for key in s.revealed if key in {"tilt_side", "favorite_tag", "habit"}]),
            "hardware": {
                "tof": core.hw.tof,
                "pickup_detect": core.hw.pickup_detect,
                "ear": core.hw.ear,
                "name_ear": core.hw.name_ear,
                "motion_validated": core.hw.motion_validated,
            },
            # The world it lives in (its rates are the world's: world_json has them).
            "world": {
                "world": str(session.world.get("world")) if session.world else None,
                "calib_hash": worldport.calib_hash(session.world) if session.world else None,
            },
        },
        "engine_rev": session.config.engine_rev,
        "clock": {
            "now": s.now.strftime("%Y-%m-%d %H:%M"),
            "iso": s.now.isoformat(),
            "weekday": "一二三四五六日"[s.now.weekday()],
            "gd": s.gd,
            "half": s.gd2 % 2 == 1,
            "stage": stage.stage_id,
            "stage_zh": stage.zh,
            "day_awake_min": s.day_awake_s // 60,
            "boot_seq": s.boot_seq,
            "lived_hours": s.lived_s // 3600,
            "ntp_ok": s.ntp_ok,
            "docked": s.docked,
        },
        "body": {
            "awake": s.awake,
            "mode": code,
            "reason": reason,
            "battery": s.battery * 100 // SCALE,
            "hot": s.hot,
            "conservative": s.conservative,
            "napping": s.napping_until is not None and s.now < s.napping_until,
            "yielding": s.yielding,
            "tv": s.tv_on,
            "guest_mode": s.guest_until is not None and s.now < s.guest_until,
            "posture": s.posture,
            "floor": s.floor,
            # Where Duck Core says it is ([x m, y m, heading rad], MuJoCo frame, mat = 0,0)
            # and the spots it learned to keep away from: the 3D view draws exactly these.
            "position": core.pose_now(),
            "avoid_spots": [
                {"x": spot["x"] / 1000, "y": spot["y"] / 1000, "r": spot["r"] / 1000}
                for spot in core.active_spots()
            ],
            "righting": s.righting is not None,
            "motion_validated": core.hw.motion_validated,
            "walk": core.walk_reason() if s.awake else None,
            "nap_reason": s.nap_reason
            if s.napping_until is not None and s.now < s.napping_until
            else None,
            "battery_help": s.battery_help,
            "night": core.night(),
            "nest": {
                "nested": s.nested,
                "charging": s.nested and s.battery < SCALE,
                "docks": s.docks,
                "threshold": nest.threshold(core) * 100 // SCALE,
                "homing": (s.homing or {}).get("reason"),
                "stranded": (s.stranded or {}).get("reason"),
                "unplaced": s.unplaced,
                "need": (s.need or {}).get("kind"),
                "dist_m": nest.dist_mm(core) / 1000,
                "roam_m": s.roam_radius_mm / 1000,
                # How far from the nest it may go right now (the family's radius, its age,
                # 1.5 m while the family sleeps): what the 3D view draws in the morning.
                "leash_m": excursion.leash_mm(core) / 1000,
            }
            if core.hw.nest
            else None,
            "day": {
                "phase": s.phase,
                "phase_zh": daycycle.PHASE_ZH.get(s.phase, s.phase),
                "family": s.family,
                "family_zh": daycycle.FAMILY_ZH.get(s.family, s.family),
            }
            if core.aware
            else None,
            "excursion": {
                "id": s.excursion.get("id"),
                "reason": s.excursion.get("reason"),
                "zh": excursion.REASON_ZH.get(str(s.excursion.get("reason")), ""),
                "legs": s.excursion.get("legs"),
            }
            if s.excursion is not None
            else None,
            "banner": banner(session),
            "last": {
                "verb": last_expression.payload.get("verb"),
                "zh": last_expression.payload.get("zh"),
                "sounds": list(seq(last_expression.payload.get("sounds"))),
                "sound_text": last_expression.payload.get("sound_text"),
                "coda": last_expression.payload.get("coda"),
                "amplitude": last_expression.payload.get("amplitude"),
                "tempo": last_expression.payload.get("tempo"),
                "event_id": last_expression.event_id,
            }
            if last_expression is not None
            else None,
            "trying": trying,
        },
        "mood": {
            "valence": s.valence * 100 // SCALE,
            "arousal": s.arousal * 100 // SCALE,
            "energy": s.energy * 100 // SCALE,
            "social": s.social * 100 // SCALE,
            "overload": s.overload * 100 // SCALE,
            "bond_owner": s.bond_owner * 100 // SCALE,
        },
        "growth": {
            "milestones": milestones,
            "unlocked": [content.VERB_BY_ID[verb_id].zh for verb_id in s.unlocked],
            "branch": content.BRANCH_BY_ID[s.branch].zh if s.branch else None,
            "branch_id": s.branch,
            "runner_up": content.BRANCH_BY_ID[s.runner_up].zh if s.runner_up else None,
            "can_rewind": s.branch is not None
            and s.gd < core._stage_day("adult")
            and not s.molt_decided,
            "compass": compass,
            "care_total": total_care,
            "adult": s.adult_name,
            "title": moments.title(core),
            "epithet": content.EPITHETS.get(s.epithet or ""),
            "stage_days": dict(s.stage_days),
            "pace_note": "每隻鴨長大的速度不太一樣：牠大約第 "
            f"{core._stage_day('juvenile')} 個成長日變成少年鴨、第 "
            f"{core._stage_day('adult')} 個成長日變成成鴨。",
            "temperament": {
                axis: int(value) * 100 // SCALE for axis, value in sorted(s.temperament.items())
            },
            "branch_odds": {
                branch: value * 100 // SCALE
                for branch, value in sorted((s.branch_detail.get("probabilities") or {}).items())
            },
            "hint": "好像快要長大了"
            if any(
                info["status"] == "locked" and info["gd"] - s.gd <= 2
                for info in s.milestones.values()
            )
            else "慢慢長大中",
        },
        "voice": {
            "first_word": content.phrase_text(content.WORDS[s.first_word])
            if s.first_word
            else None,
            "phrases": [
                {"code": code, **{key: value for key, value in item.items() if key != "tags"}}
                for code, item in sorted(
                    s.phrases.items(), key=lambda pair: (pair[1]["born_gd"], pair[0])
                )
                if item["promoted"]
                or item["liked_gds"]
                or code in set((s.song or {}).get("phrases", []))
            ][-40:],
            "liked": [
                item["text"]
                for code, item in sorted(s.phrases.items())
                if item["liked_gds"] and not item["promoted"]
            ],
            "babbles": len(s.phrases),
            "promotions": [s.phrases[code]["text"] for code in s.promotions],
            "song": {
                "version": s.song["version"],
                "text": " ／ ".join(s.phrases[code]["text"] for code in s.song["phrases"]),
            }
            if s.song
            else None,
            "molt": dict(s.molt) if s.molt and s.molt.get("status") == "open" else None,
        },
        "learning": {"prefs": prefs, "bond_owner": s.bond_owner * 100 // SCALE},
        "moments": moments.summary(core),
        "album": [
            next(
                zh
                for pool in content.RARE_POOLS.values()
                for rare_id, zh in pool
                if rare_id == item
            )
            for item in s.album
        ],
        "budgets": {
            "bids_today": s.bids_today,
            "bids_hour": len([moment for moment in s.bid_times if moment <= s.now]),
            "bids_cap_hour": stage.bids_per_hour,
            "self_phrases_hour": len(s.self_phrase_times),
            "bids_stopped": s.bids_stopped,
        },
        "stats": {
            "events": len(core.log.events),
            "chain_head": core.log.head[:12],
            "touches": s.touches,
            "pickups": s.pickups,
            "play_meters": s.play_meters,
            "walked_m": s.walked_cm / 100,
            "walks": s.walks,
            "stands": s.stands,
            "falls": s.falls,
            "avoid_spots": len(core.active_spots()),
            "facts": len(core.memory().facts),
        },
        "inputs": len(session.inputs),
    }


def _bubble(event: Any) -> str | None:
    """Short sound text for the speech bubble, never a whole sentence."""

    if event.kind == "voice.babble":
        return str(event.payload.get("text"))
    if event.kind in {"duck.bid", "duck.greeting", "duck.bedtime", "duck.reaction", "duck.stop"}:
        sounds = tuple(str(tag) for tag in event.payload.get("sounds", ()) if tag in content.TAG_ZH)
        return content.phrase_text(sounds) if sounds else None
    return None


def _cut(p: Any) -> dict[str, Any]:
    """How far a walk that was cut short got: seconds walked and where it stopped."""
    cut = mapping(p.get("walk_cut"))
    return {"walk": cut.get("walk"), "after_s": cut.get("after_s"), "pose": cut.get("pose")}


def _pose_m(pose: Any) -> Any:
    """A pose for the page, [x m, y m, heading rad]: v4 wrote that list; v4_1's commands carry
    Duck Core's predictions as ``{x_mm, y_mm, yaw_mrad}`` (the same numbers, exactly)."""
    if isinstance(pose, Mapping):
        return [
            int(pose.get("x_mm", 0)) / 1000,
            int(pose.get("y_mm", 0)) / 1000,
            int(pose.get("yaw_mrad", 0)) / 1000,
        ]
    return pose


def _motion(event: Any) -> dict[str, Any] | None:
    """What the 3D view should play for this event (clip player intent, never motor commands).

    Both engines' chains: v4_0 (frozen) and v4_1 (its commands and the world's percepts),
    so the page and the live bridge need not know which duck it is."""

    p = event.payload
    if event.kind == "duck.expression":
        return {
            "clip": str(p.get("verb")),
            "amplitude": int(p.get("amplitude", SCALE) or SCALE),
            "tempo": int(p.get("tempo", SCALE) or SCALE),
            "sounds": list(p.get("sounds", ())),
            "coda": p.get("coda"),
        }
    if event.kind == "duck.sleep_stir" and str(p.get("clip")) in content.CLIPS:
        # Asleep it still moves a little; it stays asleep afterwards.
        return {
            "clip": str(p.get("clip")),
            "amplitude": int(p.get("amplitude", 6000) or 6000),
            "tempo": SCALE,
            "sounds": list(p.get("sounds", ())),
            "stir": True,
        }
    if event.kind == "duck.reaction" and str(p.get("verb")) in content.CLIPS:
        return {
            "clip": str(p.get("verb")),
            "amplitude": 6000,
            "tempo": SCALE,
            "sounds": list(p.get("sounds", ())),
        }
    if event.kind == "owner.not_that":
        # "Not that": a small neutral dip of the head, no sound (the core's own note).
        return {"clip": "nod", "amplitude": 4000, "tempo": 8000, "sounds": []}
    if event.kind == "duck.walk":
        # Commanded vx/vyaw as duck-body would send them, plus what the official physics
        # actually does with them, so the view turns the body as far as the real duck would.
        return {
            "walk": {
                "vx": p.get("vx"),
                "vyaw": p.get("vyaw"),
                "seconds": p.get("seconds"),
                "speed": p.get("speed"),
                "yaw_rate": p.get("yaw_rate"),
                "verb": p.get("verb"),
                "id": event.event_id,
                # [x m, y m, heading rad]: where it believes it starts and expects to end
                # (v4_1: bel_from / expect, predictions; v4: from / to)
                "from": _pose_m(p.get("bel_from", p.get("from"))),
                "to": _pose_m(p.get("expect", p.get("to"))),
            }
        }
    if event.kind == "duck.posture":
        # (v4_1: a fall is Duck Core's own posture record too, with the walk it cut and where
        # it lies — v4 put both on its body.fell — so the 3D view lays the body down there)
        posture: dict[str, Any] = {"posture": str(p.get("posture"))}
        if p.get("walk_cut"):
            posture.update(stop=True, cut=_cut(p))
        if p.get("posture") == "fallen" and p.get("at") is not None:
            posture["at"] = p.get("at")
        return posture
    if event.kind == "body.posture":
        # v4_1: the stand-up act's result is the posture record (a failed try stays seated).
        return {"posture": "stand"} if p.get("got") == "stand" else None
    if event.kind == "body.fell":
        # Always a move, so the live bridge keeps treating it as an interruption; v4 put the
        # place and the cut walk here, v4_1 puts them on its duck.posture record (just after).
        fell: dict[str, Any] = {"fall": True}
        if p.get("at") is not None:
            fell["at"] = p.get("at")
        if p.get("walk_cut"):
            fell["cut"] = _cut(p)
        return fell
    if event.kind == "body.righting":
        # One try to roll back up; the posture event that follows a good try sits it up.
        return {"righting": {"try": p.get("try", p.get("tries")), "ok": bool(p.get("ok"))}}
    if event.kind == "duck.righting_end":
        # v4_1: a hand (or switching off) ended the tries (v4 wrote it as a body.righting).
        return {"righting": {"try": p.get("tries"), "ok": False}}
    if event.kind in ("body.struggle", "duck.struggle"):
        return {"righting": {"try": 0, "ok": False}}
    if event.kind == "env.floor":
        return {"place": p.get("pose"), "floor": bool(p.get("floor"))}
    if event.kind == "nest.docked":
        # In the nest's middle. Walked in, it sits the way it came (the ring contacts charge
        # at any heading, owner decision D1): no heading, so the view keeps the one its walk
        # in ended with. A hand sets it down facing the room (+x).
        walked_in = p.get("by") in ("self", "reseat")
        return {"place": [0.0, 0.0] if walked_in else [0.0, 0.0, 0.0], "floor": True, "dock": True}
    simple = {
        "sense.touch": {"touch": p.get("where", "crown")},
        "sense.pickup": {"pickup": True},
        "sense.hold_head": {"hold": True},
        "sense.hand_near": {"hand": p.get("speed", "slow")},
        "pad.play_end": {"play": list(p.get("skills", ())), "minutes": p.get("minutes")},
        "duck.nap_start": {"nap": True},
        "duck.nap_end": {"wake": True},
        "power.on": {"power": "on"},
        "power.off": {"power": "off"},
        "power.battery_swap": {"battery": True},
        "sense.nudge": {"touch": "crown"},
        "sense.set_upright": {"hold": True},
        "duck.stop": {"rest": True},
    }
    return simple.get(event.kind)


def clips(session: AnySession) -> dict[str, Any]:
    side = 1 if session.core.s.quirks.get("tilt_side") == "左" else -1
    return {
        "envelope": content.SEATED_ENVELOPE,
        "home": content.HEAD_HOME,
        "sign": content.JOINT_SIGN,
        "mouth_travel": content.MOUTH_TRAVEL_RAD,
        "cutoff_hz": content.ROBOTD_HEAD_CUTOFF_HZ,
        "side": side,
        "nap_pose": content.NAP_POSE,
        "names": content.CLIP_ZH,
        "clips": {
            name: [[t, frame] for t, frame in frames] for name, frames in content.CLIPS.items()
        },
    }


# Events without a move that the 3D view still says in words at their moment (the day's
# phases, outings and the way home, the nest, guesses and clever or goofy moments): what
# the duck is up to, so a still or walking duck on screen is never unexplained.
NOTE_KINDS = frozenset(
    {
        "duck.moment",
        "duck.predict",
        "duck.predict_outcome",
        "moment.reaction",
        "moment.gag_promoted",
        "duck.excursion",
        "nav.home_start",
        "sense.nest_seen",  # (v4: the nest in sight)
        "sense.sighting",  # (v4_1: the nest in sight, the world's answer to its look)
        "nest.dock_try",
        "nest.stranded",
        "duck.bid",
        "owner.notice",
        "duck.self_wake",
        "duck.phase",
        "env.family",
        "duck.anticipate",
        "duck.day_form",
        "duck.act_timeout",
    }
)
# Not shown in the feed: decisions and care bookkeeping (as v4), and v4_1's command and
# walk-end records, which pair one-for-one with the walk or the result the page already shows.
FEED_SKIP = frozenset({"duck.decision", "care.episode", "duck.act", "walk.done"})


def feed(
    session: AnySession, event_cursor: int, journal_cursor: int, limit: int = 120
) -> dict[str, Any]:
    if not isinstance(session, Session):
        return cast(dict[str, Any], _v4_0().feed(session, event_cursor, journal_cursor, limit))
    core = session.core
    events = core.log.events[max(0, event_cursor) :]
    journal = core.journal[max(0, journal_cursor) :]
    # A walk that a fall or a hand cut short, within this batch: show only what it walked.
    cuts = [
        mapping(event.payload.get("walk_cut"))
        for event in events[-limit:]
        if event.kind in {"body.fell", "duck.posture"}
    ]
    cut_after = {str(cut.get("walk")): cut.get("after_s") for cut in cuts if cut}
    cut_pose = {str(cut.get("walk")): cut.get("pose") for cut in cuts if cut and cut.get("pose")}
    items: list[dict[str, Any]] = []
    first = len(core.log.events) - len(events[-limit:])
    for index, event in enumerate(events[-limit:], start=first):
        if event.kind in FEED_SKIP:
            continue
        motion = _motion(event)
        if motion is not None and "walk" in motion and event.event_id in cut_after:
            motion["walk"]["cut_after"] = cut_after[event.event_id]
            if event.event_id in cut_pose:
                motion["walk"]["to"] = cut_pose[event.event_id]  # where it really stopped
        items.append(
            {
                "seq": index,
                "event_id": event.event_id,
                "at": event.occurred_at.strftime("%m/%d %H:%M"),
                "kind": event.kind,
                "category": category(event),
                "category_zh": CATEGORY.get(category(event), ""),
                "text": describe(event),
                "bubble": _bubble(event),
                "motion": motion,
                **({"note": True} if event.kind in NOTE_KINDS and motion is None else {}),
            }
        )
    return {
        "events": items,
        "journal": [
            {
                "at": entry["at"][5:16].replace("T", " "),
                "persona": entry["persona"],
                "persona_zh": entry["persona_zh"],
                "detail": entry["detail"],
                "perceived": entry["perceived"],
                "event_ids": entry["event_ids"],
            }
            for entry in journal[-limit:]
        ],
        "cursor": {"events": len(core.log.events), "journal": len(core.journal)},
    }


def config_json(session: AnySession) -> dict[str, Any]:
    """The duck's config as saved and exported (with its world header and inputs, the whole
    duck). A v4_0 duck's is the frozen Config's, named ``engine_rev: v4_0``."""
    if not isinstance(session, Session):
        return {**_v4_0().config_json(session), "engine_rev": FROZEN_REV}
    return _config_dict(session.config)


def world_json(session: AnySession) -> dict[str, Any] | None:
    """The header of the world the duck lives in (None for a v4_0 duck: its Config has it)."""
    if not isinstance(session, Session) or session.world is None:
        return None
    return dict(session.world)


# ---------------------------------------------------------------------------- percepts in


def sense(session: Session, events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Percepts handed in from outside (``dispatch(session, "sense", {"events": […]})``): the
    real duck's body.sock reader, a live simulator's duck-body (spec §7.4 "api.sense").

    Every one is checked against percept-v1 first (design A13: all of them, not a sample);
    one bad percept refuses the whole batch, so a replay never meets half of one. Then, in
    order, each is taken in at its whole second ``due_second(t_ms)``: Duck Core lives on to
    that second first when it lies ahead (the world says it is already there), and one that
    lies behind is taken in now. A percept more than :data:`SENSE_AHEAD_S` ahead of Duck
    Core's clock (or of the percept before it in the batch) is refused with the whole batch:
    a sender on another clock must not live the duck on by hours in one call."""
    checked = [worldport.check(event) for event in events]
    core = session.core
    reach = core.now_ms()
    for event in checked:
        due_ms = worldport.due_second(int(event["t_ms"])) * 1000
        if due_ms - reach > SENSE_AHEAD_S * 1000:
            raise ValueError(
                f"{event['kind']} at t_ms {event['t_ms']} lies {(due_ms - reach) // 1000} s ahead "
                f"of the duck's clock (at most {SENSE_AHEAD_S} s): another clock?"
            )
        reach = max(reach, due_ms)
    taken = []
    for event in checked:
        due_ms = worldport.due_second(int(event["t_ms"])) * 1000
        if due_ms > core.now_ms():
            core.advance_to(due_ms)
        answer = core.on_percept(event, via=worldport.VIA_INPUT)
        taken.append(
            {
                "kind": event["kind"],
                "taken": answer.get("taken"),
                "event": answer.get("event"),
                "note": answer.get("note"),
            }
        )
    return {"taken": taken, "now_ms": core.now_ms()}


def experience(session: Session, events: Any) -> dict[str, Any]:
    """What the neural duck lived (``dispatch(session, "experience", {"events": […]})``), each
    event checked against ``bridge-v2`` (bridge-v1 and the trick lessons) first (the whole batch,
    so a replay never meets half of one), then taken in, in order, at its own second
    (``DuckCore.experience``)."""
    checked = bridge_v2.check_all(events)
    core = session.core
    reach = core.now_ms()
    for event in checked:
        due_ms = int(event["t"]) * 1000
        if due_ms - reach > bridge.AHEAD_S * 1000:
            raise bridge.BridgeError(
                bridge.Code.AHEAD, kind=event["k"], t=event["t"], seconds=(due_ms - reach) // 1000
            )
        reach = max(reach, due_ms)
    taken = [core.experience(event) for event in checked]
    return {"taken": taken, "now_ms": core.now_ms()}


def dispatch(
    session: AnySession, method: str, params: dict[str, Any], *, record: bool = True
) -> dict[str, Any]:
    if not isinstance(session, Session):
        # A v4_0 duck: the frozen api, untouched (its history must never change).
        if method in (
            "sense",
            "bridge",
            "experience",
            "percept_replay_check",
            "world_replay_check",
        ):
            raise ValueError(f"{method}: a {FROZEN_REV} duck has no world port")
        return cast(dict[str, Any], _v4_0().dispatch(session, method, params, record=record))
    core = session.core
    result: dict[str, Any]
    if method == "state":
        result = {}
    elif method == "act":
        result = _act(
            session, str(params["action"]), dict(params.get("params", {})), params.get("persona")
        )
    elif method == "advance":
        if params.get("until") == NAP_END:
            # Until the nap it is taking ends (none: nothing to wait for): its whole minutes,
            # then on to the minute's own moment that wakes it (within a minute).
            minutes = core.nap_left_min()
            result = core.advance(minutes)
            extra = 0
            while core.s.awake and core.s.napping_until is not None and extra < 120:
                core.advance_seconds(1)
                extra += 1
            result["until"] = NAP_END
            result["seconds"] = minutes * 60 + extra
        elif params.get("until"):
            # "Until 07:00": the minutes are worked out on the duck's own clock, so a
            # replay of this input lands on the same moment.
            result = core.advance(core.minutes_until(str(params["until"])))
            result["until"] = str(params["until"])
        elif params.get("seconds") is not None:
            # Real time: the server's clock lives the duck forward second by second.
            result = core.advance_seconds(max(0, int(params["seconds"])))
        else:
            result = core.advance(int(params.get("minutes", 1)))
    elif method == "live" and params.get("family"):
        result = family_live(session, str(params["family"]), int(params.get("days", 7)))
    elif method == "live":
        routine = str(params.get("routine", "daily"))
        if routine not in ROUTINES:
            raise ValueError("unknown routine")
        result = session.household.live(session.hands, int(params.get("days", 7)), routine)
    elif method == "party":
        result = session.household.party(
            session.hands, guest_mode=bool(params.get("guest_mode", False))
        )
    elif method == "set_setting":
        result = session.hands.set_setting(str(params["key"]), bool(params["value"]))
    elif method == "sense":
        result = sense(session, list(params.get("events", ())))
    elif method == "bridge":
        result = core.bridge_on(str(params.get("api", bridge.BRIDGE_API)))
    elif method == "experience":
        result = experience(session, params.get("events", ()))
    elif method == "choose":
        result = core.choose(str(params["kind"]), str(params["option"]))
    elif method == "undo_promotion":
        result = core.undo_promotion()
    elif method == "forget":
        result = core.forget(str(params["fact_id"]))
    elif method == "forget_by_query":
        result = _forget_by_query(session, str(params["query"]))
    elif method == "undo_forget":
        result = core.undo_forget()
    elif method == "command":
        result = _run_command(
            session, str(params.get("text", "")), str(params.get("persona", "owner"))
        )
    elif method == "query":
        result = query.search(core, str(params.get("q", "")))
    elif method == "diary":
        result = {"diary": query.diary(core, int(params.get("limit", 30)))}
    elif method == "why":
        result = query.why(core, str(params.get("question", "為什麼不動")))
    elif method == "graph":
        result = graph.build(
            core, truth=bool(params.get("truth", False)), episodes=int(params.get("episodes", 10))
        )
    elif method == "fact":
        fact = core.memory().facts.get(str(params.get("fact_id")))
        result = {"fact": query.fact_json(core, fact, cites=12) if fact else None}
    elif method == "events":
        limit = min(500, int(params.get("limit", 100)))
        kind = params.get("kind")
        selected = [
            event for event in core.log.events if not kind or event.kind.startswith(str(kind))
        ]
        offset = (
            max(0, len(selected) - limit) if params.get("tail") else int(params.get("offset", 0))
        )
        result = {
            "total": len(selected),
            "offset": offset,
            "events": [
                dict(event_json(event), text=describe(event))
                for event in selected[offset : offset + limit]
            ],
            "chain_ok": core.log.verify(),
            "head": core.log.head,
        }
    elif method == "event":
        event = core.log.get(str(params.get("event_id")))
        result = {"event": dict(event_json(event), text=describe(event)) if event else None}
    elif method == "export":
        result = {
            **save_json(session),
            "head": core.log.head,
            "events": [event_json(event) for event in core.log.events],
        }
    elif method == "replay_check":
        result = replay_check(session)
    elif method == "percept_replay_check":
        result = percept_replay_check(session)
    elif method == "world_replay_check":
        result = world_replay_check(session)
    elif method == "clips":
        result = clips(session)
    elif method == MIGRATE:
        raise ValueError("migrate returns a new session: call api.migrate(session, to)")
    else:
        raise ValueError(f"unknown method {method}")
    if record and method in MUTATING:
        last = session.inputs[-1] if session.inputs else None
        if (
            method == "advance"
            and set(params) == {"seconds"}
            and last is not None
            and last[0] == "advance"
            and set(last[1]) == {"seconds"}
        ):
            # Real time adds a second-by-second advance all day long. Time passes the same
            # whatever the step, so back-to-back advances are one input (one line per
            # stretch of real time between two things someone did, not one per second).
            last[1]["seconds"] = int(last[1]["seconds"]) + max(0, int(params["seconds"]))
        else:
            session.inputs.append((method, json.loads(json.dumps(params))))
    return result


def family_profile(session: AnySession, kind: str) -> FamilyProfile:
    """The duck's simulated family of this kind (drawn from the duck's seed)."""
    if not isinstance(session, Session):
        return cast(FamilyProfile, _v4_0().family_profile(session, kind))
    if kind not in FAMILY_KINDS:
        raise ValueError(f"unknown family {kind}")
    return make_profile(session.config.seed ^ 0xFA11, kind)


def family_live(session: AnySession, kind: str, days: int) -> dict[str, Any]:
    """Live ``days`` with a household-v2 family of ``kind`` (one recorded input)."""
    if not isinstance(session, Session):
        return cast(dict[str, Any], _v4_0().family_live(session, kind, days))
    if session.family is None or session.family.profile.kind != kind:
        session.family = HouseholdV2(session.config.seed ^ 0xFA11, family_profile(session, kind))
    return session.family.live(session.hands, days)


# ---------------------------------------------------------------------------- replay


def _same_chain(first: DuckCore, second: DuckCore) -> bool:
    return first.log.head == second.log.head and len(first.log.events) == len(second.log.events)


def replay_check(session: AnySession) -> dict[str, Any]:
    """Rebuild the duck from its config (and world header) and recorded inputs; the chains
    must match."""
    if not isinstance(session, Session):
        return cast(dict[str, Any], _v4_0().replay_check(session))
    if session.origin is not None:
        moved = restore(save_json(session))  # (through the old engine and the seam)
        assert isinstance(moved, Session)
        fresh = moved
    else:
        fresh = Session(replace(session.config), world=session.world, keep_tape=False)
        for method, params in session.inputs:
            dispatch(fresh, method, params, record=False)
    same = _same_chain(fresh.core, session.core)
    return {
        "identical": same,
        "original_head": session.core.log.head[:16],
        "replayed_head": fresh.core.log.head[:16],
        "events": len(session.core.log.events),
        "inputs": len(session.inputs),
        "note": "同樣的種子＋同樣的輸入 → 一模一樣的事件鏈" if same else "重播結果不同！",
    }


def percept_replay_check(session: AnySession) -> dict[str, Any]:
    """Duck Core decides only from its inputs (spec §7.6, design A8): its input tape — every
    call that changed it from outside and every percept its world delivered — fed to the same
    Duck Core with no world and no family must give the same chain."""
    if not isinstance(session, Session):
        raise ValueError(f"percept_replay_check: a {FROZEN_REV} duck has no world port")
    if session.origin is not None:
        # A duck that moved here (WP13): its tape starts at the seam, so the replay starts
        # from its old self rebuilt at the seam, carried over as it was.
        moved = import_module("duckmind.engines.migrate").percept_replay_check
        return cast("dict[str, Any]", moved(session))
    tape = session.core.tape
    if tape is None:
        return {"identical": False, "note": "這隻小鴨沒有錄輸入帶（keep_tape=False）"}
    port = TapePort(tape)
    fresh = _build_core(session.config, port, None)
    try:
        calls = worldport.replay_calls(fresh, tape)
    except Exception as error:  # (a replay that went another way is a finding, not a crash)
        return {
            "identical": False,
            "original_head": session.core.log.head[:16],
            "replayed_head": fresh.log.head[:16],
            "error": f"{type(error).__name__}: {error}",
            "note": "只靠輸入帶重播時走了另一條路（和錄下來的不同）",
        }
    same = _same_chain(fresh, session.core)
    return {
        "identical": same,
        "original_head": session.core.log.head[:16],
        "replayed_head": fresh.log.head[:16],
        "events": len(session.core.log.events),
        "calls": calls,
        "percepts": sum(1 for _ in tape.percepts()),
        "all_percepts_taken": port.exhausted,
        "note": "只靠輸入帶（沒有世界、沒有家人）→ 一模一樣的事件鏈"
        if same
        else "只靠輸入帶重播的結果不同！",
    }


def world_replay_check(session: AnySession) -> dict[str, Any]:
    """The world is a pure function (spec §7.6): its header and its own tape (every command,
    poll and hand it took) fed to a fresh world give the same percepts, one for one."""
    if not isinstance(session, Session):
        raise ValueError(f"world_replay_check: a {FROZEN_REV} duck has no world port")
    port = session.port
    replay = _world_replay(port)
    tape_json = getattr(port, "tape_json", None)
    sent = getattr(port, "sent", None)
    if replay is None or tape_json is None or sent is None:
        return {"identical": None, "note": f"這個世界（{port.src}）不能重播"}
    if getattr(port, "keep_tape", True) is False:
        return {"identical": None, "note": "這隻小鴨的世界沒有錄世界帶（keep_tape=False）"}
    again = list(replay(port.header(), tape_json()))
    recorded = list(sent)
    same = worldport.canonical(again) == worldport.canonical(recorded)
    first_diff = next(
        (
            index
            for index, (a, b) in enumerate(zip(again, recorded, strict=False))
            if worldport.canonical(a) != worldport.canonical(b)
        ),
        None if len(again) == len(recorded) else min(len(again), len(recorded)),
    )
    return {
        "identical": same,
        "percepts": len(recorded),
        "replayed": len(again),
        "first_difference": first_diff,
        "calib_hash": worldport.calib_hash(port.header()),
        "note": "世界標頭＋世界收過的呼叫 → 一模一樣的感覺" if same else "世界重播的結果不同！",
    }


def _world_replay(port: WorldPort) -> Callable[..., Iterable[dict[str, Any]]] | None:
    """The world module's own ``replay(header, tape)`` (the world registered itself, so its
    module is loaded; Duck Core never imports it)."""
    module = inspect.getmodule(type(port))
    replay = getattr(module, "replay", None) if module is not None else None
    return cast("Callable[..., Iterable[dict[str, Any]]] | None", replay)
