"""Moving a duck to a newer engine with its memories (phase-4 WP13; owner decision P4-1 = a).

The owner chose that 蛋黃 moves to the new mind in phase 5 with its personality, growth, voice
and diary. This is the seam that does it, proven in phase 4 on the one move that exists today:
a v4 duck on the frozen engine (``duckmind.engines.v4_0``) moves to engine v4_1 (Duck Core as it
is now, living in LegacyWorld) and from then on behaves exactly as it would have on v4
(``tests/test_migrate.py``: the same behaviour projection, duck for duck and seam for seam).

What moves, at the moment of the move (the *seam*, after any input):

* **its state** — every ``CoreState`` field as it is (the new engine's extra fields start empty),
  its hardware (Duck Core's part of it), its expression caps, its truth journal;
* **its event chain** — every event and digest, unchanged: the new engine appends to the same
  chain, so the head only moves on (``EventLog.verify`` over the carried chain holds);
* **its draw counters** — each purpose goes on from where it stood. The ten outcome purposes
  (falls, righting, standing up, docking, touch and hold detection, the TV's false touch, the
  gamepad) are the world's in v4_1, so their counters go to the new world (LegacyWorld
  ``adopt``), Duck Core keeps the rest;
* **what the world was doing** — a walk under way (its end, or the fall v4 had already drawn for
  it) and the tries to get up after a fall are v4's schedule; the world takes them over, and the
  walk goes on as Duck Core's open act, so the result arrives when v4 would have acted;
* **its family** — the simulated household and household-v2 family with their own draw counters.

The session records the seam: its ``origin`` is the config it hatched with on v4, and its inputs
are the old ones plus a ``migrate`` input, so ``api.restore`` and ``api.replay_check`` walk
through the same seam (the old engine up to it, this move, the new engine after it), and
:func:`percept_replay_check` rebuilds the moved duck at the seam before it plays the input
tape. The old session is left as it was.

Phase 5 moves 蛋黃 to the new mind with the same seam; place memories (where it fell, favourite
spots) are relearned there, as the owner decided.

This module is part of what ``duckmind.duckcore.api`` loads (``api.migrate``), so it imports no
world (``tests/test_imports.py``): it reaches the world only through ``duckmind.worldport`` and
the port object itself.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from importlib import import_module
from typing import Any

from duckmind import worldport
from duckmind.duckcore import api
from duckmind.duckcore.acts import WALK, WALK_SLACK_MS, Act
from duckmind.duckcore.engine import CoreState, DuckCore, HardwareConfig
from duckmind.duckcore.memory import EventLog
from duckmind.engines.v4_0.duckcore import api as v4_0_api
from duckmind.provenance import Event, SourceRef
from duckmind.worldport import CoreTape, TapePort, WorldError, WorldPort

FROM_REV = "v4_0"
TO_REV = api.ENGINE_REV
FROZEN_PACKAGE = "duckmind.engines.v4_0"
# The outcome purposes that are the world's in v4_1 (phase-4 design §4, A3): their counters go
# to the world. (LegacyWorld's own list is compared at the seam when the world has one.)
WORLD_PURPOSES = (
    "fall",
    "fall.at",
    "righting",
    "stand.try",
    "dock.try",
    "detect.touch",
    "detect.hold",
    "false_alarm.tv",
    "play.meters",
    "play.skill",
)
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_SECOND = timedelta(seconds=1)


class MigrationError(ValueError):
    """A duck that cannot move (not a v4 duck, or not restorable from its inputs)."""


def _ms(moment: datetime) -> int:
    return ((moment - _EPOCH) // _SECOND) * 1000


def is_v4_0(session: Any) -> bool:
    """Is this a session of the frozen v4 engine?"""
    return isinstance(session, v4_0_api.Session)


def seam(session: Any) -> str:
    """What a v4 duck is doing at this moment, as the seam sees it: ``quiet``, ``walk`` (a walk
    under way, maybe with its fall already drawn: ``walk+fall``) or ``righting`` (lying after a
    fall, the body still trying to get up)."""
    s = session.core.s
    if s.walk_plan is not None:
        return "walk+fall" if s.walk_plan.get("fall_at") is not None else "walk"
    if s.righting is not None:
        return "righting"
    return "quiet"


# ---------------------------------------------------------------------------- the move


def migrate(old: Any, to: str = TO_REV, *, keep_tape: bool = True) -> api.Session:
    """The same duck on engine ``to`` (v4_1), memories and all, from a v4_0 session at any
    moment between inputs. ``keep_tape`` keeps the new engine's replay tapes (Duck Core's input
    tape and the world's), as a page duck does. Use ``api.migrate`` from outside: it is this."""
    if to != TO_REV:
        raise MigrationError(f"a duck moves from {FROM_REV} to {TO_REV}, not to {to!r}")
    if not is_v4_0(old):
        raise MigrationError(f"only a {FROM_REV} duck moves (this is {type(old).__module__})")
    config, rates = _config(old)
    header = api.legacy_world(config, rates)
    if header["table_hash"] != old.core.table:
        raise MigrationError("the content table changed: every draw would be another")
    if bool(header["marker"]) != bool(old.core.hw.marker):
        raise MigrationError("the nest marker is not the duck's camera: not restorable")
    port = worldport.make_world(header)
    if not keep_tape and hasattr(port, "keep_tape"):
        port.keep_tape = False  # (the world's own tape: world_replay_check's input)
    world = port.header()
    origin = v4_0_api.config_json(old)
    tape = (
        CoreTape(
            header={
                "engine_rev": to,
                "config": dataclasses.asdict(config),
                "world": world,
                "carried_over": {
                    "from": FROM_REV,
                    "origin": origin,
                    "inputs": len(old.inputs),
                    "events": len(old.core.log.events),
                    "head": old.core.log.head,
                },
            }
        )
        if keep_tape
        else None
    )
    core, handover = carry(old.core, port, tape)
    _adopt(port, core, handover)
    memo: dict[int, Any] = {}
    session = object.__new__(api.Session)  # (built from the old duck, not hatched)
    session.config = config
    session.world = world
    session.keep_tape = keep_tape
    session.core = core
    session.port = port
    session.hands = api.LegacyActs(core, port)
    session.household = _rehome(old.household, memo)
    session.family = _rehome(old.family, memo)
    session.inputs = [*copy.deepcopy(old.inputs), (api.MIGRATE, {"to": to})]
    session.origin = {**origin, "engine_rev": FROM_REV}
    return session


def _config(old: Any) -> tuple[api.Config, dict[str, int]]:
    """The v4_1 Config of an old duck and the outcome rates its world takes (v4 kept them in
    the Config and in the hardware; they must agree, or its inputs could not rebuild it)."""
    saved = dict(v4_0_api.config_json(old))
    saved.pop("engine_rev", None)
    world_fields = api.v4_world_fields()
    rates = {name: int(saved.pop(name)) for name in list(saved) if name in world_fields}
    for name, value in rates.items():
        if int(getattr(old.core.hw, name)) != value:
            raise MigrationError(
                f"its hardware {name} is {getattr(old.core.hw, name)}, its config says {value}: "
                "changed outside its inputs, so it could not be restored"
            )
    unknown = sorted(set(saved) - set(api.Config.__dataclass_fields__))
    if unknown:
        raise MigrationError(f"{TO_REV} has no Config field {unknown}")
    return api.Config(**saved, engine_rev=TO_REV), rates


def carry(
    old: Any, port: WorldPort | None, tape: CoreTape | None
) -> tuple[DuckCore, dict[str, Any]]:
    """A v4_1 Duck Core carrying a v4 Duck Core's state, chain and own counters, on ``port``;
    and what the world must take over (``counters``: the ten outcome purposes' counters;
    ``walk`` / ``righting``: v4's schedule at the seam, or None). The old core is not changed.

    A walk under way stays Duck Core's open act (its command is v4's ``duck.walk`` record):
    ``DuckCore.carried_over`` takes a quiet duck, so the walk and the righting are handed over
    right after it, as they stood."""
    names = [item.name for item in dataclasses.fields(old.s)]
    values = copy.deepcopy({name: getattr(old.s, name) for name in names})
    unknown = sorted(set(values) - set(CoreState.__dataclass_fields__))
    if unknown:
        raise MigrationError(f"{TO_REV} has no state field {unknown}")
    plan, righting = values.pop("walk_plan"), values.pop("righting")
    state = CoreState(**values)  # (the new fields start empty)
    walk_act, walk = None, None
    if plan is not None:
        walk_act, walk, plan = _walk(old, plan)
    right = None
    if righting is not None:
        fell = old.log.get(str(righting["fell"]))
        if fell is None:
            raise MigrationError(f"the fall {righting['fell']} is not in its chain")
        right = {
            "fell_ms": _ms(fell.occurred_at),
            "tries": int(righting["tries"]),
            "next_ms": _ms(righting.pop("next_at")),
        }
    counters = {name: int(count) for name, count in old.counters.items()}
    world = {name: counters.pop(name) for name in list(counters) if name in WORLD_PURPOSES}
    core = DuckCore.carried_over(
        state,
        chain(old.log),
        counters,
        hardware=HardwareConfig(
            **{item.name: getattr(old.hw, item.name) for item in dataclasses.fields(HardwareConfig)}
        ),
        expression_caps=old.caps,
        journal=copy.deepcopy(old.journal),
        port=port,
        tape=tape,
        acts_seq=1 if walk_act is not None else 0,
    )
    state.walk_plan, state.righting = plan, righting
    if walk_act is not None:
        core.acts.open[walk_act.act_id] = walk_act
    return core, {"counters": world, "walk": walk, "righting": right}


def _walk(old: Any, plan: dict[str, Any]) -> tuple[Act, dict[str, Any], dict[str, Any]]:
    """v4's walk under way as Duck Core's open act, the world's walk to finish, and the v4_1
    walk plan (v4's without the drawn fall, which is the world's now)."""
    s = old.s
    fall_at = plan.pop("fall_at", None)
    act_id = "A1"  # (the first act this duck ever had: v4 issued none)
    record = old.log.get(str(plan["event"]))
    if record is None or s.walk_start is None or s.walk_until is None:
        raise MigrationError("a walk under way without its record or its times")
    seconds = int(plan["seconds"])
    start_ms = _ms(s.walk_start)
    (fx, fy, fh), (tx, ty, th) = plan["from"], plan["to"]
    wire: dict[str, Any] = {
        "cmd": WALK,
        "act_id": act_id,
        "vx_mm": int(plan["vx"]),
        "vyaw_mrad": int(plan["vyaw"]),
        "max_ms": seconds * 1000,
        "bel_from": {"x_mm": int(fx), "y_mm": int(fy), "yaw_mrad": int(fh)},
        "expect": {"x_mm": int(tx), "y_mm": int(ty), "yaw_mrad": int(th)},
        "cm": int(plan["cm"]),
        "carried": True,  # (v4 issued it; this act took it over at the seam)
    }
    purpose = record.payload.get("purpose")
    if isinstance(purpose, str) and purpose:
        wire["purpose"] = purpose
    deadline = start_ms + seconds * 1000 + WALK_SLACK_MS
    wire["deadline_ms"] = deadline
    act = Act(act_id, WALK, record.event_id, start_ms, deadline, wire)
    walk = {
        "cmd": WALK,
        "act_id": act_id,
        "seconds": seconds,
        "vx_mm": int(plan["vx"]),
        "vyaw_mrad": int(plan["vyaw"]),
        "bel_from": [int(fx), int(fy), int(fh)],
        "expect": [int(tx), int(ty), int(th)],
        "path_mm": int(plan["cm"]) * 10,
        "start_ms": start_ms,
        "end_ms": _ms(s.walk_until),
        "fall_ms": None if fall_at is None else _ms(fall_at),
    }
    if "purpose" in wire:
        walk["purpose"] = wire["purpose"]
    return act, walk, {**plan, "act_id": act_id}


def _adopt(port: WorldPort, core: DuckCore, handover: Mapping[str, Any]) -> None:
    """The world takes over v4's dice counters and schedule, and the detectors' settings."""
    adopt = getattr(port, "adopt", None)
    if adopt is None:
        raise WorldError(f"this world ({port.src}) cannot take over a v4 duck")
    purposes = getattr(inspect.getmodule(type(port)), "PURPOSES", None)
    if purposes is not None and set(purposes) != set(WORLD_PURPOSES):
        raise MigrationError(f"the world's outcome purposes are not v4's: {sorted(purposes)}")
    s = core.s
    adopt(
        now_ms=core.now_ms(),
        counters=handover["counters"],
        walk=handover["walk"],
        righting=handover["righting"],
        conservative=s.conservative,
        tv_on=s.tv_on,
    )


def chain(old: Any) -> EventLog:
    """The same event chain in this engine's classes: every event and digest as they were (the
    payloads are immutable and shared), so the head is the same and new events append to it."""
    log = EventLog()
    refs: dict[tuple[Any, ...], SourceRef] = {}
    events = log.events
    by_id = log._by_id
    new = object.__new__
    put = object.__setattr__
    for event in old.events:
        source = event.source
        key = (source.kind, source.source_id, source.reliability, source.version)
        ref = refs.get(key)
        if ref is None:
            ref = refs[key] = SourceRef(*key)
        moved = new(Event)
        put(moved, "event_id", event.event_id)
        put(moved, "kind", event.kind)
        put(moved, "occurred_at", event.occurred_at)
        put(moved, "observed_at", event.observed_at)
        put(moved, "source", ref)
        put(moved, "trace_id", event.trace_id)
        put(moved, "payload", event.payload)
        put(moved, "parent_event_ids", event.parent_event_ids)
        events.append(moved)
        by_id[moved.event_id] = moved
    log.digests = list(old.digests)
    if log.head != old.head or len(log.events) != len(old.events):
        raise MigrationError("the chain did not carry over")
    return log


def _rehome(value: Any, memo: dict[int, Any]) -> Any:
    """A copy of ``value`` whose frozen-engine objects (the family: ``Household``,
    ``HouseholdV2``, its profile and draws) are this engine's classes, with the same fields.
    Shared objects stay shared (``memo``, as ``copy.deepcopy`` does)."""
    key = id(value)
    if key in memo:
        return memo[key]
    kind = type(value)
    module = kind.__module__
    if module == FROZEN_PACKAGE or module.startswith(FROZEN_PACKAGE + "."):
        target = getattr(
            import_module("duckmind" + module[len(FROZEN_PACKAGE) :]), kind.__qualname__
        )
        if dataclasses.is_dataclass(kind):
            mine = [item.name for item in dataclasses.fields(kind)]
            theirs = [item.name for item in dataclasses.fields(target)]
            if mine != theirs:
                raise MigrationError(f"{kind.__qualname__} changed its fields: {mine} → {theirs}")
        moved = object.__new__(target)
        memo[key] = moved
        slots = [
            name
            for cls in kind.__mro__
            for name in (
                (cls.__dict__["__slots__"],)
                if isinstance(cls.__dict__.get("__slots__"), str)
                else cls.__dict__.get("__slots__", ())
            )
        ]
        for name in slots:
            if name not in ("__dict__", "__weakref__") and hasattr(value, name):
                object.__setattr__(moved, name, _rehome(getattr(value, name), memo))
        for name, item in getattr(value, "__dict__", {}).items():
            object.__setattr__(moved, name, _rehome(item, memo))
        return moved
    if kind is dict:
        out: dict[Any, Any] = {}
        memo[key] = out
        for name, item in value.items():
            out[_rehome(name, memo)] = _rehome(item, memo)
        return out
    if kind is list:
        items: list[Any] = []
        memo[key] = items
        items.extend(_rehome(item, memo) for item in value)
        return items
    if kind in (tuple, set, frozenset):
        rebuilt = kind(_rehome(item, memo) for item in value)
        memo[key] = rebuilt
        return rebuilt
    if value is None or kind in (str, int, bool, float, datetime):
        return value
    rebuilt = copy.deepcopy(value)
    memo[key] = rebuilt
    return rebuilt


def leftovers(session: Any) -> list[str]:
    """Where a moved session still holds a frozen-engine object (none: fully moved): its
    config, state, hardware, counters, journal, open acts, world, family, and every event and
    source of its chain."""
    found: list[str] = []
    seen: set[int] = set()

    def frozen(item: Any) -> bool:
        module = type(item).__module__
        return module == FROZEN_PACKAGE or module.startswith(FROZEN_PACKAGE + ".")

    def visit(item: Any, where: str) -> None:
        if id(item) in seen or len(found) > 20:
            return
        seen.add(id(item))
        if frozen(item):
            found.append(f"{where}: {type(item).__module__}.{type(item).__qualname__}")
            return
        if isinstance(item, Mapping):
            for name, sub in item.items():
                visit(sub, f"{where}[{name!r}]")
        elif isinstance(item, (list, tuple, set, frozenset)):
            for index, sub in enumerate(item):
                visit(sub, f"{where}[{index}]")
        elif dataclasses.is_dataclass(item) and not isinstance(item, type):
            for field in dataclasses.fields(item):
                visit(getattr(item, field.name), f"{where}.{field.name}")
        elif hasattr(item, "__dict__") and not inspect.isroutine(item):
            for name, sub in vars(item).items():
                visit(sub, f"{where}.{name}")

    core = session.core
    for name, item in (
        ("config", session.config),
        ("core.s", core.s),
        ("core.hw", core.hw),
        ("core.counters", core.counters),
        ("core.caps", core.caps),
        ("core.journal", core.journal),
        ("core.acts.open", core.acts.open),
        ("port", session.port),
        ("household", session.household),
        ("family", session.family),
    ):
        visit(item, name)
    for index, event in enumerate(core.log.events):
        if type(event) is not Event or type(event.source) is not SourceRef:
            found.append(f"core.log.events[{index}]: {type(event).__module__}")
            break
    return found


# ---------------------------------------------------------------------------- replay


def at_seam(session: api.Session) -> tuple[Any, int]:
    """The moved duck's old self at the seam: its origin config on the frozen engine with every
    input before the ``migrate`` input replayed (and the index of that input)."""
    if session.origin is None:
        raise MigrationError("this duck never moved")
    inputs = session.inputs
    seam_at = next(
        (index for index, (method, _params) in enumerate(inputs) if method == api.MIGRATE), None
    )
    if seam_at is None:
        raise MigrationError("a moved duck without its migrate input")
    origin = {key: value for key, value in session.origin.items() if key != "engine_rev"}
    old = v4_0_api.Session(v4_0_api.Config(**origin))
    for method, params in inputs[:seam_at]:
        v4_0_api.dispatch(old, method, copy.deepcopy(params))
    return old, seam_at


def percept_replay_check(session: api.Session) -> dict[str, Any]:
    """``api.percept_replay_check`` for a moved duck: its old self is rebuilt at the seam (the
    frozen engine, the inputs before it) and carried over onto a port that plays the input
    tape back (no world, no family); the tape's calls must give the same chain."""
    tape = session.core.tape
    if tape is None:
        return {"identical": False, "note": "這隻小鴨沒有錄輸入帶（keep_tape=False）"}
    old, seam_at = at_seam(session)
    port = TapePort(tape)
    fresh, _handover = carry(old.core, port, None)
    try:
        calls = worldport.replay_calls(fresh, tape)
    except Exception as error:  # (a replay that went another way is a finding, not a crash)
        return {
            "identical": False,
            "original_head": session.core.log.head[:16],
            "replayed_head": fresh.log.head[:16],
            "seam_input": seam_at,
            "error": f"{type(error).__name__}: {error}",
            "note": "只靠輸入帶重播時走了另一條路（和錄下來的不同）",
        }
    same = fresh.log.head == session.core.log.head and len(fresh.log.events) == len(
        session.core.log.events
    )
    return {
        "identical": same,
        "original_head": session.core.log.head[:16],
        "replayed_head": fresh.log.head[:16],
        "events": len(session.core.log.events),
        "calls": calls,
        "percepts": sum(1 for _ in tape.percepts()),
        "all_percepts_taken": port.exhausted,
        "seam_input": seam_at,
        "note": "從搬家那一刻起只靠輸入帶（沒有世界、沒有家人）→ 一模一樣的事件鏈"
        if same
        else "只靠輸入帶重播的結果不同！",
    }


__all__ = [
    "FROM_REV",
    "TO_REV",
    "WORLD_PURPOSES",
    "MigrationError",
    "at_seam",
    "carry",
    "chain",
    "is_v4_0",
    "leftovers",
    "migrate",
    "percept_replay_check",
    "seam",
]
