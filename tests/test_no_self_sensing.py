"""Rule R2 (physical-design spec §2.2): Duck Core never writes what its senses report.

``sense.*``, ``body.*``, ``env.cue`` and ``walk.done`` are percepts. On the robot they come from
duck-body, duck-eye and duck-ear, in the simulator from the world, and they reach Duck Core only
through an input handler (today an ``api`` action, later ``api.sense``). Any other event Duck
Core signs with one of the duck's own sensors (``engine.SOURCES`` produced by duck-ear,
duck-body or robotd: ``ear``, ``joint``, ``imu``, ``tof``, ``body``) is held to the same rule:
signing its own guess with a sensor's name is the same fabrication.

Two checks, as spec §2.2 asks:

* static: every ``_emit`` call under ``duckcore/`` is read with ``ast``. A sense-like one is
  allowed only as an input handler's own record of the input it has just received
  (``INPUT_RECORDS``), and only while Duck Core never calls that handler on itself. The
  percept handler ``on_percept`` is called by Duck Core itself in exactly one place,
  ``deliver`` (``PORT_DELIVERY``), and every call of ``deliver`` hands it what a port call
  (``due``, ``command``, ``poll``) returned in that same function.
* dynamic: a short seeded life on engine v4_1 with LegacyWorld (a nest duck and a desk duck
  through a busy morning: walks, docking, falls, failed rises, the TV, the gamepad, hands, the
  family's cues; and a duck whose battery runs flat) runs with ``_emit`` watched. Every
  sense-like event must be an input record; a percept's record counts only when the very
  percept object came out of the world (its port calls, or the family's detectors).

Phase 4 emptied the exception list (``self_sensing_exceptions.json``; spec §8.10 G29–G37 and
four sensor-signed records): LegacyWorld sends the same draws in as percepts, and Duck Core's
own records (``duck.posture``, ``env.floor``, ``duck.struggle``, ``duck.righting_end``,
``duck.heard``, ``env.daily_sound``, a flat battery's ``power.off``) are signed ``self`` (the
family's floor switch is itself an input, signed ``sim`` as in v4). The list stays, empty, and
may never grow: a new self-written sense fails.
"""

from __future__ import annotations

import ast
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import FrameType
from typing import Any

import pytest
from phase4_harness import open_v41, send

from duckmind import percept
from duckmind.duckcore import acts, api, bridge_v2, engine

DUCKCORE = Path(engine.__file__).resolve().parent
EXCEPTIONS = Path(__file__).with_name("self_sensing_exceptions.json")
LISTED_AT_MOST = 0  # the list may only shrink: lower this with it, never raise it

SENSE_PREFIXES = ("sense.", "body.")
SENSE_KINDS = frozenset({"env.cue", "walk.done"})
SENSING = frozenset({"duck-ear", "duck-body", "duck-eye", "robotd", "mediad", "tofd"})
SENSOR_SOURCES = frozenset(
    key for key, (producer, _sensor, _confidence) in engine.SOURCES.items() if producer in SENSING
)
DYNAMIC = "<dynamic kind>"
UNKNOWN = "<unknown source>"

# Every percept kind (percept-v1, with phase 4's three since WP9: sense.touch, sense.hold_head,
# body.play).
PERCEPT_KINDS = frozenset(percept.SCHEMAS)
# input handler -> the kinds it writes as the record of the input it has just received. Every
# key is one of api.ACT_METHODS, or ``on_percept``: its record's kind is the percept's own
# (hence dynamic in the code; the dynamic check sees each real kind).
INPUT_RECORDS: dict[str, frozenset[str]] = {
    "on_percept": PERCEPT_KINDS | {DYNAMIC},
    "touch": frozenset({"sense.touch"}),
    "hold_head": frozenset({"sense.hold_head"}),
    "pickup": frozenset({"sense.pickup"}),
    "hand_near": frozenset({"sense.hand_near"}),
    "call_name": frozenset({"sense.name_heard"}),
    "place_on_nest": frozenset({"sense.placed"}),
    "wake": frozenset({"sense.nudge"}),
    "nap_now": frozenset({"sense.nudge"}),
    "set_upright": frozenset({"sense.set_upright"}),
    "power_on": frozenset({"power.on"}),
    "power_off": frozenset({"power.off"}),
    # growth G2: what the neural duck lived (bridge-v1, and bridge-v2's trick lessons) is an input
    # like any other; its handler writes the record of it (the kind is the experience's, so
    # dynamic here) and counts it
    "experience": bridge_v2.RECORD_KINDS | {DYNAMIC},
}
# DuckCore methods api calls with an input (a handler may pass its input on to another one).
HANDLERS = frozenset(api.ACT_METHODS) | {
    "on_percept",
    "experience",
    "set_setting",
    "choose",
    "undo_promotion",
    "forget",
    "undo_forget",
}
# The owner's unboxing: the duck hatches on arrival and is switched on (its Config, not a log
# input, is what a replay starts from).
INPUT_ROOTS = frozenset({"DuckCore.__init__", "DuckCore._hatch"})
# Where Duck Core hands its percept handler what its port delivered (pulled in its time step,
# answered to a command, answered to a poll).
PORT_DELIVERY = frozenset({"DuckCore.deliver"})
PORT_CALLS = frozenset({"due", "command", "poll"})
# Wrappers an input passes through unchanged (engine._input puts outermost calls on the
# replay tape).
WRAPPERS = frozenset({"_input.<locals>.recorded"})
# Emit sites whose kind is a variable, with every kind they may write (checked at run time).
DYNAMIC_EMITTERS: dict[str, frozenset[str]] = {"DuckCore.emit_command": acts.COMMAND_KINDS}
# Modules in the package that are not the duck's mind: the entry point, the text-command
# parser, and the simulated family (the world, which lives here for now).
NOT_MIND = frozenset({"api.py", "commands.py", "routines.py"})
LOG_WRITES = frozenset({"append", "extend", "insert"})

Key = tuple[str, str, str]  # (file under duckcore/, function qualname, event kind)


def _sense_like(kind: str, sources: frozenset[str]) -> bool:
    return (
        kind.startswith(SENSE_PREFIXES)
        or kind in SENSE_KINDS
        or kind == DYNAMIC
        or bool(sources & (SENSOR_SOURCES | {UNKNOWN}))
    )


def _handler(function: str) -> str | None:
    """``touch`` for ``DuckCore.touch`` when that is an api input handler, else None."""
    name = function.removeprefix("DuckCore.")
    return name if name != function and name in HANDLERS else None


# ---------------------------------------------------------------------------- static scan


@dataclass(frozen=True)
class _Emit:
    file: str
    function: str
    kind: str
    source: ast.expr | None
    line: int

    @property
    def key(self) -> Key:
        return self.file, self.function, self.kind


@dataclass(frozen=True)
class _Call:
    file: str
    function: str
    callee: str
    receiver: str | None
    node: ast.Call


@dataclass
class _Index:
    functions: dict[tuple[str, str], ast.FunctionDef | ast.AsyncFunctionDef]
    emits: list[_Emit]
    calls: list[_Call]
    log_writes: list[str]


class _Scan(ast.NodeVisitor):
    """Emits, calls and function definitions, each with its qualified name as Python's
    ``co_qualname`` spells it (``DuckCore._fall``, ``outer.<locals>.inner``)."""

    def __init__(self, file: str, index: _Index) -> None:
        self.file = file
        self.index = index
        self.scope: list[str] = []

    @property
    def function(self) -> str:
        names = self.scope[:-1] if self.scope[-1:] == ["<locals>"] else self.scope
        return ".".join(names) or "<module>"

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> None:
        name = node.name if not isinstance(node, ast.Lambda) else "<lambda>"
        self.scope.append(name)
        if not isinstance(node, ast.Lambda):
            self.index.functions[(self.file, self.function)] = node
        self.scope.append("<locals>")
        self.generic_visit(node)
        del self.scope[-2:]

    visit_FunctionDef = visit_AsyncFunctionDef = visit_Lambda = _function

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        callee = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        receiver = None
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            receiver = func.value.id
        if callee == "_emit":
            kind = node.args[0] if node.args else _keyword(node, "kind")
            source = node.args[2] if len(node.args) > 2 else _keyword(node, "source")
            text = kind.value if isinstance(kind, ast.Constant) else None
            if isinstance(kind, ast.Name):
                text = _constants().get(kind.id)  # a module constant (acts.TIMEOUT_KIND …)
            self.index.emits.append(
                _Emit(
                    self.file,
                    self.function,
                    text if isinstance(text, str) else DYNAMIC,
                    source,
                    node.lineno,
                )
            )
        elif isinstance(func, ast.Attribute) and func.attr in LOG_WRITES and _is_log(func.value):
            if (self.file, self.function) != ("engine.py", "DuckCore._emit"):
                self.index.log_writes.append(f"{self.file}:{node.lineno} {self.function}")
        if callee is not None:
            self.index.calls.append(_Call(self.file, self.function, callee, receiver, node))
        self.generic_visit(node)


def _keyword(node: ast.Call, name: str) -> ast.expr | None:
    return next((kw.value for kw in node.keywords if kw.arg == name), None)


def _is_log(node: ast.expr) -> bool:
    """``<x>.log`` or ``<x>.log.events``: the L1 event log."""
    if isinstance(node, ast.Attribute) and node.attr == "events":
        node = node.value
    return isinstance(node, ast.Attribute) and node.attr == "log"


@cache
def _constants() -> dict[str, str]:
    """Module-level ``NAME = "text"`` constants of Duck Core (a name defined twice with two
    values is left out: it cannot be told)."""
    found: dict[str, set[str]] = defaultdict(set)
    for path in sorted(DUCKCORE.rglob("*.py")):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        found[target.id].add(node.value.value)
    return {name: next(iter(values)) for name, values in found.items() if len(values) == 1}


@cache
def _index() -> _Index:
    index = _Index({}, [], [], [])
    for path in sorted(DUCKCORE.rglob("*.py")):
        file = path.relative_to(DUCKCORE).as_posix()
        _Scan(file, index).visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
    return index


def _parameter(
    node: ast.FunctionDef | ast.AsyncFunctionDef, name: str
) -> tuple[int | None, ast.expr | None, bool] | None:
    """(position, default, is a method) of parameter ``name``; None when it is not one."""
    args = node.args
    positional = [*args.posonlyargs, *args.args]
    method = bool(positional) and positional[0].arg == "self"
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    for position, (arg, default) in enumerate(zip(positional, defaults, strict=True)):
        if arg.arg == name:
            return position, default, method
    for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        if arg.arg == name:
            return None, default, method
    return None


def _values(
    expr: ast.expr | None, file: str, function: str, seen: frozenset[tuple[str, str, str]]
) -> frozenset[str]:
    """Every literal a source expression can take (``UNKNOWN`` where it cannot be told)."""
    if expr is None:
        return frozenset({"self"})  # _emit's default
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return frozenset({expr.value})
    if isinstance(expr, ast.IfExp):
        return _values(expr.body, file, function, seen) | _values(expr.orelse, file, function, seen)
    if isinstance(expr, ast.Name):
        return _parameter_values(file, function, expr.id, seen)
    return frozenset({UNKNOWN})


def _parameter_values(
    file: str, function: str, name: str, seen: frozenset[tuple[str, str, str]]
) -> frozenset[str]:
    """What callers pass for parameter ``name`` of ``function`` (its default included)."""
    index = _index()
    node = index.functions.get((file, function))
    found = _parameter(node, name) if node is not None else None
    if node is None or found is None or (file, function, name) in seen:
        return frozenset({UNKNOWN})
    seen = seen | {(file, function, name)}
    position, default, method = found
    values: set[str] = set()
    if default is not None:
        values |= _values(default, file, function, seen)
    short = function.rsplit(".", 1)[-1]
    calls = [call for call in index.calls if call.callee == short]
    for call in calls:
        passed = _keyword(call.node, name)
        if passed is None and position is not None:
            offset = 1 if method else 0
            if 0 <= position - offset < len(call.node.args):
                passed = call.node.args[position - offset]
        starred = any(isinstance(arg, ast.Starred) for arg in call.node.args)
        if passed is None and (starred or any(kw.arg is None for kw in call.node.keywords)):
            values.add(UNKNOWN)
        elif passed is not None:
            values |= _values(passed, call.file, call.function, seen)
    if default is None and not calls:
        values.add(UNKNOWN)
    return frozenset(values)


def _self_called() -> dict[str, list[str]]:
    """Input handlers Duck Core calls on itself (then their records are its own words)."""
    found: dict[str, list[str]] = defaultdict(list)
    for call in _index().calls:
        if (
            call.callee in INPUT_RECORDS
            and call.receiver in {"self", "core"}
            and call.file not in NOT_MIND
            and call.function not in INPUT_ROOTS
            and _handler(call.function) is None
            and not (call.callee == "on_percept" and call.function in PORT_DELIVERY)
        ):
            found[call.callee].append(f"{call.file}:{call.node.lineno} {call.function}")
    return dict(found)


@cache
def _self_emitted() -> dict[Key, list[str]]:
    """Every static emit site of a sense-like event that is not an input handler's own record
    of its input, keyed by (file, function, kind), with where it is."""
    self_called = _self_called()
    found: dict[Key, list[str]] = defaultdict(list)
    for emit in _index().emits:
        sources = _values(emit.source, emit.file, emit.function, frozenset())
        if emit.kind == DYNAMIC and emit.function in DYNAMIC_EMITTERS and sources == {"self"}:
            continue  # (its kinds are checked at run time)
        if not _sense_like(emit.kind, sources):
            continue
        handler = _handler(emit.function)
        if (
            handler is not None
            and emit.kind in INPUT_RECORDS.get(handler, frozenset())
            and handler not in self_called
        ):
            continue
        where = f"{emit.file}:{emit.line} source {'/'.join(sorted(sources))}"
        if handler in self_called:
            where += f", called by Duck Core itself at {'; '.join(self_called[handler])}"
        found[emit.key].append(where)
    return dict(found)


@cache
def _listed() -> dict[Key, dict[str, Any]]:
    entries = json.loads(EXCEPTIONS.read_text(encoding="utf-8"))["exceptions"]
    return {(entry["file"], entry["function"], entry["kind"]): entry for entry in entries}


def _show(key: Key) -> str:
    file, function, kind = key
    return f"{kind} in {file} {function}"


def _note(capsys: pytest.CaptureFixture[str], lines: list[str]) -> None:
    if lines:
        with capsys.disabled():
            print("\n[self-sensing] " + "\n[self-sensing] ".join(lines))


# ---------------------------------------------------------------------------- tests


def test_the_exception_list_is_well_formed_and_only_shrinks() -> None:
    entries = json.loads(EXCEPTIONS.read_text(encoding="utf-8"))["exceptions"]
    assert len(entries) == LISTED_AT_MOST, (
        "self_sensing_exceptions.json may only shrink: when an exception goes, lower "
        f"LISTED_AT_MOST with it ({len(entries)} listed, LISTED_AT_MOST = {LISTED_AT_MOST})"
    )
    assert len(_listed()) == len(entries), "an exception is listed twice"
    for entry in entries:
        assert set(entry) == {"spec", "file", "function", "kind", "sites", "reason"}, entry
        assert isinstance(entry["sites"], int) and entry["sites"] >= 1, entry
        assert entry["spec"].strip() and entry["reason"].strip(), entry
        assert "\n" not in entry["reason"], entry


def test_input_records_are_the_api_inputs() -> None:
    assert set(INPUT_RECORDS) - {"on_percept", "experience"} <= set(api.ACT_METHODS)
    assert "experience" in api.MUTATING  # (the neural duck's input, recorded like the others)
    assert {"ear", "joint", "imu", "tof", "body"} <= SENSOR_SOURCES
    assert not SENSOR_SOURCES & {"owner", "pad", "clock", "sim", "self"}
    # a percept is recorded signed by its src (percept-v1), which Duck Core knows as a source
    assert set(percept.SOURCES) <= set(engine.SOURCES)
    assert {"sense.touch", "sense.hold_head", "body.play", "walk.done"} <= PERCEPT_KINDS


def test_duck_core_hands_its_percept_handler_only_what_its_port_returned() -> None:
    index = _index()
    delivery = [
        call for call in index.calls if call.callee == "on_percept" and call.file not in NOT_MIND
    ]
    assert delivery, "Duck Core never hands a delivered percept to on_percept"
    assert {call.function for call in delivery} <= PORT_DELIVERY, [
        f"{call.file}:{call.node.lineno} {call.function}" for call in delivery
    ]
    problems = []
    for call in index.calls:
        if call.callee != "deliver" or call.file in NOT_MIND:
            continue
        pulled = {
            other.callee
            for other in index.calls
            if (other.file, other.function) == (call.file, call.function)
        }
        if not pulled & PORT_CALLS:
            problems.append(f"{call.file}:{call.node.lineno} {call.function}")
    assert not problems, (
        "deliver() takes only what a port call (due / command / poll) returned in the same "
        "function: " + "; ".join(problems)
    )
    for function in DYNAMIC_EMITTERS:
        assert any(emit.function == function for emit in index.emits), function


def test_only_emit_writes_the_event_log() -> None:
    writes = _index().log_writes
    assert not writes, f"the L1 log is written only through DuckCore._emit: {'; '.join(writes)}"


def test_duck_core_writes_no_new_sense_of_its_own() -> None:
    found, listed = _self_emitted(), _listed()
    problems = []
    for key, where in sorted(found.items()):
        entry = listed.get(key)
        if entry is None:
            problems.append(f"NEW: {_show(key)} ({'; '.join(where)})")
        elif len(where) > entry["sites"]:
            problems.append(
                f"MORE: {_show(key)}: {len(where)} sites, {entry['sites']} listed "
                f"({'; '.join(where)})"
            )
    assert not problems, (
        "Duck Core writes a sense of its own (spec R2). A sense enters only through an input "
        "handler; a listed exception (tests/self_sensing_exceptions.json) may only shrink.\n"
        + "\n".join(problems)
    )
    # 只准變少: an exception that shrinks is locked in the same change (it could otherwise grow
    # back up to the old listing unnoticed).
    shrank = [
        f"{entry['spec']} {_show(key)} is gone: remove it from self_sensing_exceptions.json "
        "and lower LISTED_AT_MOST by one"
        for key, entry in sorted(listed.items())
        if key not in found
    ] + [
        f"{entry['spec']} {_show(key)} has {len(found[key])} of {entry['sites']} sites left: "
        f'set its "sites" to {len(found[key])}'
        for key, entry in sorted(listed.items())
        if key in found and len(found[key]) < entry["sites"]
    ]
    assert not shrank, (
        "Fewer self-written senses — good. Lower tests/self_sensing_exceptions.json in the same "
        "change (the list only goes down):\n" + "\n".join(shrank)
    )


# ---------------------------------------------------------------------------- dynamic check


def _in_mind(frame: FrameType) -> bool:
    path = Path(frame.f_code.co_filename).resolve()
    return path.is_relative_to(DUCKCORE) and _file(frame) not in NOT_MIND


def _file(frame: FrameType) -> str:
    path = Path(frame.f_code.co_filename).resolve()
    return path.relative_to(DUCKCORE).as_posix() if path.is_relative_to(DUCKCORE) else str(path)


# ids of every percept object the world handed out (kept alive in _FROM_WORLD)
_FROM_WORLD: list[Any] = []
_FROM_WORLD_IDS: set[int] = set()


def _from_world(percepts: Any) -> Any:
    for event in [percepts] if isinstance(percepts, dict) else list(percepts or ()):
        _FROM_WORLD.append(event)
        _FROM_WORLD_IDS.add(id(event))
    return percepts


def _input_record(frame: FrameType, kind: str) -> bool:
    """Written by an input handler about the input it received (from the api, the family or
    a test, possibly passed on by another handler), not by Duck Core calling it on itself. A
    percept's record counts when its percept came from outside, or from the world through
    ``deliver``."""
    handler = _handler(frame.f_code.co_qualname)
    if handler is None or kind not in INPUT_RECORDS.get(handler, frozenset()):
        return False
    event = None
    if handler == "on_percept":  # (its first parameter after self is the percept)
        event = frame.f_locals.get(frame.f_code.co_varnames[1])
        if not isinstance(event, dict) or event.get("kind") != kind:
            return False
    caller = frame.f_back
    while caller is not None and _in_mind(caller):
        if caller.f_code.co_qualname in WRAPPERS:
            caller = caller.f_back
            continue
        if caller.f_code.co_qualname in INPUT_ROOTS:
            return True
        if caller.f_code.co_qualname in PORT_DELIVERY and handler == "on_percept":
            return id(event) in _FROM_WORLD_IDS
        if _handler(caller.f_code.co_qualname) is None:
            return False
        caller = caller.f_back
    return True


def _session(**wanted: Any) -> Any:
    """A v4_1 duck with LegacyWorld; the outcome dials go to the world's header."""
    return open_v41(wanted)


def dispatch(session: Any, method: str, params: dict[str, Any]) -> None:
    send(session, method, params)


def _act(session: Any, action: str, persona: str | None = None, **params: Any) -> None:
    if action in api.ACT_METHODS:
        dispatch(session, "act", {"action": action, "params": params, "persona": persona})


def _morning(session: Any, *, nest: bool) -> None:
    """Six busy hours from 09:00 with the TV on: the family's cues, voices, hands, a lift, the
    gamepad, its name, a hand before the beak, a nap and a nudge; then on past midnight."""
    dispatch(session, "advance", {"until": "09:00"})
    _act(session, "tv", on=True)
    for minute in range(360):
        if minute % 3 == 0:
            _act(session, "hear_home", kind="voice")
            _act(session, "hear_home", kind="person")
        if minute % 11 == 0:
            _act(session, "say", "owner", kind="voice")
        if minute % 37 == 5:
            _act(session, "touch", "child", where="crown")
        if minute % 53 == 7:
            _act(session, "hold_head", "owner")
        if minute % 71 == 9:
            _act(session, "pickup", "owner")
            _act(session, "place_on_nest" if nest else "wake", "owner")
        if minute % 29 == 3:
            _act(session, "set_upright", "owner")  # (nothing happens unless it lies there)
        if minute == 100:
            _act(session, "pad_play", "child", minutes=6)
        if minute == 150:
            _act(session, "call_name", "owner")
        if minute == 160:
            _act(session, "hand_near", "owner", speed="slow")
        if minute == 170:
            _act(session, "nap_now", "owner")
        if minute == 175:
            _act(session, "wake", "owner")
        dispatch(session, "advance", {"minutes": 1})
    _act(session, "tv", on=False)
    dispatch(session, "advance", {"minutes": 1440})


def _live_a_little() -> None:
    """Three seeded ducks. A nest duck old enough to walk, its outcome dials turned up (walks,
    homing, docking, falls, failed rises, lying stuck), and a desk duck, each through a busy
    morning; and a duck with no nest and nobody to swap its battery, left until it runs flat."""
    nest_duck = _session(
        seed=3,
        nest=True,
        tof=True,
        conservative=False,
        fall_ppm_per_s=40_000,
        righting_ppm=100_000,
        stand_ppm=600_000,
    )
    dispatch(nest_duck, "advance", {"minutes": 8 * 1440})
    _morning(nest_duck, nest=True)
    _morning(_session(seed=5, tof=True, conservative=False), nest=False)
    dispatch(_session(seed=4, battery_help=False), "advance", {"minutes": 180})


def test_a_seeded_life_writes_no_unlisted_sense(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from duckmind.world import legacy

    listed = set(_listed())
    reached: dict[Key, int] = defaultdict(int)
    records: dict[str, int] = defaultdict(int)
    wrong_kind: list[str] = []
    original = engine.DuckCore._emit
    for name in ("due", "command", "poll", "touch", "hold_head"):
        method = getattr(legacy.LegacyWorld, name)
        monkeypatch.setattr(
            legacy.LegacyWorld,
            name,
            lambda world, *a, _method=method, **k: _from_world(_method(world, *a, **k)),
        )

    def watched(
        core: engine.DuckCore,
        kind: str,
        payload: dict[str, Any],
        source: str = "self",
        parents: tuple[str, ...] | list[str] = (),
    ) -> Any:
        frame = sys._getframe(1)
        allowed = DYNAMIC_EMITTERS.get(frame.f_code.co_qualname)
        if allowed is not None and kind not in allowed:
            wrong_kind.append(f"{frame.f_code.co_qualname} wrote {kind}")
        if _sense_like(kind, frozenset({source})):
            if _input_record(frame, kind):
                records[kind] += 1
            else:
                reached[(_file(frame), frame.f_code.co_qualname, kind)] += 1
        return original(core, kind, payload, source, parents)

    monkeypatch.setattr(engine.DuckCore, "_emit", watched)
    _live_a_little()
    unlisted = sorted(key for key in reached if key not in listed)
    assert not unlisted, (
        "At run time Duck Core wrote a sense of its own that no listed exception covers:\n"
        + "\n".join(f"{_show(key)} × {reached[key]}" for key in unlisted)
    )
    assert not wrong_kind, sorted(set(wrong_kind))
    assert records.keys() >= {"sense.touch", "sense.pickup", "sense.hold_head"}, (
        "the seeded life received too few inputs: the check did not run"
    )
    from_world = {"walk.done", "body.fell", "body.righting", "body.posture", "body.dock"}
    assert records.keys() >= from_world, (
        f"the world's percepts never came in: {sorted(from_world - records.keys())}"
    )
    _note(
        capsys,
        [
            f"{_show(key)} is still in the code but the seeded life no longer reaches it"
            for key in sorted(listed & set(_self_emitted()))
            if key not in reached
        ],
    )
