"""The world port: the one way the world reaches Duck Core (phase 4 「邊界」, design A1/A4/A8).

**Duck Core pulls.** Inside its own time step (``DuckCore._set_now``) Duck Core asks its port
which percepts are due by the moment it is stepping to, and takes each one in through its input
handler (``DuckCore.on_percept``) at the percept's whole second (:func:`due_second`). At one
second the world's percepts come *before* Duck Core's own clock events (sleep stirs, midnight,
07:00). Pulling inside the step is what keeps a walk that ends inside an input's own seconds
(a 6-second touch) exactly where v4 had it; a loop outside Duck Core could not (design §2.1).

**Commands go out the other way.** Duck Core's command events (``duck.walk``, ``duck.act``,
``duck.curio``; :mod:`duckmind.duckcore.acts`) are handed to :meth:`WorldPort.command` as their
wire dicts (``{"cmd": "duck.walk", "act_id": …}``, the body.sock shapes of
:mod:`duckmind.duckbody.protocol` plus Duck Core's own fields, which a world ignores unless it
needs them). ``command`` returns the percepts the world senses *in that same call* (LegacyWorld
answers a stand, a dock try and the nest look at once: design A2); anything later comes back
through :meth:`WorldPort.due`.

**Time axis.** Duck Core's clock is a timezone-aware ``datetime`` on whole seconds; a port
speaks epoch milliseconds (:func:`epoch_ms`), always a whole second × 1000 when Duck Core
calls it. A percept's ``t_ms`` is on the sender's axis; for an in-process world that is this
same axis.

This module is neutral: standard library and :mod:`duckmind.percept` only, so Duck Core
(``duckcore/``), the worlds (``world/``), the api, the replay tools and the tests can all import
it without Duck Core ever importing a world (``tests/test_imports.py``, rule R1). A world
registers itself here when its module is imported (``duckmind.world.legacy`` does
``register_world("legacy", LegacyWorld.from_header)``); the api builds one from a header with
:func:`make_world`.

Replay (design A8): a :class:`CoreTape` records every call that changed a Duck Core from outside
(its public input methods, with their arguments, at its clock) and every percept its port
delivered (pulled, answered to a command, or returned by a poll). :class:`TapePort` plays the
delivered percepts back to a fresh Duck Core at exactly the same points, and
:func:`replay_calls` repeats the calls: no world and no family needed, the same chain head
(``percept_replay_check``).
"""

from __future__ import annotations

import hashlib
import heapq
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Any, Protocol, runtime_checkable

from duckmind import percept
from duckmind.percept import due_second

WORLDPORT_API = "worldport-v1"
TAPE_API = "coretape-v1"
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_SECOND = timedelta(seconds=1)

ADVANCE_TO = "advance_to"  # how a tape records time passing (CoreTape.passed)

# Where a delivered percept came from (tape entries, ``DuckCore.on_percept``'s ``via``).
VIA_PULL = "pull"  # due by Duck Core's time step (DuckCore._set_now)
VIA_REPLY = "reply"  # answered in the same call as a command (WorldPort.command)
VIA_POLL = "poll"  # answered to a poll (WorldPort.poll)
VIA_INPUT = "input"  # handed in from outside (api.sense, the family proxy): a tape *call*
VIAS = (VIA_PULL, VIA_REPLY, VIA_POLL, VIA_INPUT)

# Polls a phase-4 Duck Core asks (LegacyWorld only; from phase 5 no world answers them).
POLL_TV_MINUTE = "tv_minute"  # a minute with the TV on: may the ear mistake it for a touch?
POLL_PLAY_MINUTE = "play_minute"  # a minute of gamepad play (params: minute=m): body.play
POLLS = (POLL_TV_MINUTE, POLL_PLAY_MINUTE)


class WorldError(RuntimeError):
    """A world that cannot be built, or a Duck Core that needs a world and has none."""


class ReplayError(RuntimeError):
    """A tape replay that went another way than the recording (a call that raised when it
    was recorded did not, or raised something else)."""


# ── the time axis ──────────────────────────────────────────────────────────────


def epoch_ms(moment: datetime) -> int:
    """``moment`` (timezone-aware, whole seconds) in epoch milliseconds."""
    if moment.utcoffset() is None:
        raise ValueError("epoch_ms needs a timezone-aware datetime")
    return ((moment - _EPOCH) // _SECOND) * 1000


def at_ms(t_ms: int, tz: tzinfo) -> datetime:
    """The whole second Duck Core takes ``t_ms`` in (:func:`due_second`), in ``tz``."""
    return (_EPOCH + timedelta(seconds=due_second(t_ms))).astimezone(tz)


def canonical(value: Any) -> str:
    """Canonical JSON (sorted keys, no spaces, UTF-8 text kept): headers and tapes."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def calib_hash(header: Mapping[str, Any]) -> str:
    """16 hex digits over a world header: the ``l:``/``c:`` part of every percept's ``pv``."""
    return hashlib.sha256(canonical(dict(header)).encode("utf-8")).hexdigest()[:16]


# ── the port ───────────────────────────────────────────────────────────────────


@runtime_checkable
class WorldPort(Protocol):
    """What Duck Core needs from a world (design §7; appendix B for HomeWorld).

    * ``src`` — the percept source this world stamps (``legacy``, ``live``, ``surrogate``,
      ``robot``, ``replay``: :data:`duckmind.percept.SOURCES`).
    * ``header()`` — the world header (JSON): what a replay needs to build the same world.
    * ``command(commands, now_ms)`` — Duck Core's command wire dicts, issued at ``now_ms``, in
      order. Returns the percepts sensed in the same call (``t_ms == now_ms``), in order; Duck
      Core takes them in before its ``acts.start`` returns. Never returns a later percept.
    * ``next_at_ms()`` — the ``t_ms`` of the earliest pending percept (None: nothing pending).
    * ``due(now_ms)`` — removes and returns every pending percept whose
      ``due_second(t_ms) * 1000 <= now_ms``, earliest first (a world orders equal times itself:
      LegacyWorld puts ``body.fell`` before the ``walk.done`` it ends).
    * ``poll(kind, now_ms, **params)`` — a question Duck Core asks at a fixed point of its
      minute (:data:`POLLS`); returns percepts at ``now_ms``. A world that does not know the
      kind returns ``[]``.

    A port never calls Duck Core: Duck Core calls it.
    """

    src: str

    def header(self) -> dict[str, Any]: ...

    def command(
        self, commands: Sequence[Mapping[str, Any]], now_ms: int
    ) -> list[dict[str, Any]]: ...

    def next_at_ms(self) -> int | None: ...

    def due(self, now_ms: int) -> list[dict[str, Any]]: ...

    def poll(self, kind: str, now_ms: int, **params: Any) -> list[dict[str, Any]]: ...


WorldFactory = Callable[[Mapping[str, Any]], WorldPort]
_WORLDS: dict[str, WorldFactory] = {}


def register_world(name: str, factory: WorldFactory) -> None:
    """Make ``make_world({"world": name, …})`` build a world with ``factory(header)``."""
    if not re.match(r"^[a-z][a-z0-9_]{0,31}$", name):
        raise ValueError(f"bad world name {name!r}")
    _WORLDS[name] = factory


def registered_worlds() -> tuple[str, ...]:
    return tuple(sorted(_WORLDS))


def make_world(header: Mapping[str, Any]) -> WorldPort:
    """The world a header describes (``header["world"]`` names a registered world)."""
    name = header.get("world")
    factory = _WORLDS.get(str(name))
    if factory is None:
        known = ", ".join(registered_worlds()) or "none"
        raise WorldError(
            f"no world {name!r} is registered (registered: {known}); import its module first, "
            "e.g. `import duckmind.world.legacy`"
        )
    return factory(header)


# ── percept validation (percept-v1) ────────────────────────────────────────────

# Phase 4 validated three kinds here (``sense.touch``, ``sense.hold_head``, ``body.play``) until
# phase 3 committed percept.py; WP9 moved them there, so a port's percepts are percept-v1's own
# (these names stay for the callers: Duck Core, LegacyWorld, the api and the tests).
validate = percept.validate
check = percept.check
known_kind = percept.known_kind


# ── simple ports ───────────────────────────────────────────────────────────────


class QueuePort:
    """Percepts handed in from outside, taken in at their due second.

    For ``api.sense`` (the real duck's body.sock reader, a live simulator's duck-body) and for
    tests: :meth:`push` queues a percept; commands are kept in :attr:`sent` and nothing is
    answered at once. With nothing pushed it is a *silent* world (an act then runs into its
    deadline: ``duck.act_timeout``)."""

    def __init__(self, src: str = "robot", header: Mapping[str, Any] | None = None) -> None:
        if src not in percept.SOURCES:
            raise ValueError(f"src {src!r} not in {percept.SOURCES}")
        self.src = src
        self._header = dict(header or {"world": "queue", "v": 1, "src": src})
        self._queue: list[tuple[int, int, dict[str, Any]]] = []
        self._seq = 0
        self.sent: list[tuple[int, dict[str, Any]]] = []

    def header(self) -> dict[str, Any]:
        return dict(self._header)

    def push(self, event: Mapping[str, Any]) -> None:
        self._seq += 1
        heapq.heappush(self._queue, (int(event["t_ms"]), self._seq, dict(event)))

    def command(self, commands: Sequence[Mapping[str, Any]], now_ms: int) -> list[dict[str, Any]]:
        self.sent += [(now_ms, dict(command)) for command in commands]
        return []

    def next_at_ms(self) -> int | None:
        return self._queue[0][0] if self._queue else None

    def due(self, now_ms: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        while self._queue and due_second(self._queue[0][0]) * 1000 <= now_ms:
            out.append(heapq.heappop(self._queue)[2])
        return out

    def poll(self, kind: str, now_ms: int, **params: Any) -> list[dict[str, Any]]:
        return []


# ── replay: the tape and its port ──────────────────────────────────────────────


@dataclass
class CoreTape:
    """Everything that changed one Duck Core from outside, in order (design A8).

    Two kinds of entry:

    * ``{"op": "call", "t_ms", "method", "args", "kwargs"}`` — an outermost call of one of Duck
      Core's public input methods (``touch`` …, ``on_percept``, ``spend_input``,
      ``set_setting``), at Duck Core's clock when it came. Calls made from inside another input
      are not entries (the outer call repeats them). Time passing (``advance``,
      ``advance_seconds``, ``advance_to``) is ``advance_to(the second it reached)``, and
      back-to-back steps are one entry (:meth:`passed`): a page's second-by-second day is one
      line per stretch between two things someone did, as in ``session.inputs``. A call that
      raised carries ``"raised": <exception name>`` (:meth:`raised`), and a replay expects
      the same exception there.
    * ``{"op": "percept", "t_ms", "via", "n_cmd", "n_poll", "p"}`` — a percept the port
      delivered (``via`` pull / reply / poll), with how many ``command`` and ``poll`` calls Duck
      Core had made to the port by then, so a :class:`TapePort` can release it at exactly the
      same point.

    The tape is a replay tool, not the duck's memory (v4 keeps every voice and cue fact out of
    L1 on purpose). ``header`` carries what a replay needs to build the same Duck Core
    (``engine_rev``, the Config) and the world header it lived with.
    """

    header: dict[str, Any] = field(default_factory=dict)
    entries: list[dict[str, Any]] = field(default_factory=list)
    # The last call entry (None after loading: the next stretch of time is a new entry).
    _last: dict[str, Any] | None = field(default=None, init=False, repr=False, compare=False)

    def call(
        self,
        t_ms: int,
        method: str,
        args: Sequence[Any] = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        entry = {
            "op": "call",
            "t_ms": t_ms,
            "method": method,
            "args": list(args),
            "kwargs": dict(kwargs or {}),
        }
        self.entries.append(entry)
        self._last = entry
        return entry

    def passed(self, t_ms: int, to_ms: int) -> None:
        """Time passed from ``t_ms`` to ``to_ms`` in one outermost call: ``advance_to(to_ms)``,
        merged into the entry before when that was time passing too (living on is the same
        whatever the step, tests/test_step_invariance.py). No time passed: no entry."""
        last = self._last
        if (
            last is not None
            and last["method"] == ADVANCE_TO
            and not last["kwargs"]
            and "raised" not in last
        ):
            if to_ms > int(last["args"][0]):
                last["args"][0] = to_ms
            return
        if to_ms > t_ms:
            self.call(t_ms, ADVANCE_TO, [to_ms])

    def raised(self, entry: dict[str, Any], error: BaseException) -> None:
        """The call ``entry`` raised ``error`` (after whatever it did first)."""
        entry["raised"] = type(error).__name__

    def percept(
        self, t_ms: int, via: str, n_cmd: int, n_poll: int, event: Mapping[str, Any]
    ) -> None:
        if via not in (VIA_PULL, VIA_REPLY, VIA_POLL):
            raise ValueError(f"a tape percept comes by pull, reply or poll, not {via!r}")
        self.entries.append(
            {
                "op": "percept",
                "t_ms": t_ms,
                "via": via,
                "n_cmd": n_cmd,
                "n_poll": n_poll,
                "p": dict(event),
            }
        )

    def calls(self) -> Iterable[dict[str, Any]]:
        return (entry for entry in self.entries if entry["op"] == "call")

    def percepts(self) -> Iterable[dict[str, Any]]:
        return (entry for entry in self.entries if entry["op"] == "percept")

    def to_dict(self) -> dict[str, Any]:
        return {"api": TAPE_API, "header": self.header, "entries": self.entries}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CoreTape:
        if data.get("api") != TAPE_API:
            raise ValueError(f"not a {TAPE_API} tape")
        return cls(header=dict(data.get("header") or {}), entries=list(data.get("entries") or []))


class TapePort:
    """A world that plays a :class:`CoreTape`'s delivered percepts back.

    A pulled percept is released once Duck Core has made at least as many ``command`` and
    ``poll`` calls as when it was first delivered, and only at its due second; a reply is
    returned by the same-numbered ``command`` call, a poll's answer by the same-numbered
    ``poll`` call. A replayed Duck Core that behaves the same therefore takes every percept in
    at exactly the same point."""

    src = "replay"

    def __init__(self, tape: CoreTape) -> None:
        self._header = dict(tape.header.get("world") or {"world": "tape", "v": 1})
        self._pulls: list[tuple[int, int, int, dict[str, Any]]] = []
        self._replies: dict[int, list[dict[str, Any]]] = {}
        self._polls: dict[int, list[dict[str, Any]]] = {}
        for entry in tape.percepts():
            event = dict(entry["p"])
            if entry["via"] == VIA_PULL:
                self._pulls.append(
                    (int(entry["t_ms"]), int(entry["n_cmd"]), int(entry["n_poll"]), event)
                )
            elif entry["via"] == VIA_REPLY:
                self._replies.setdefault(int(entry["n_cmd"]), []).append(event)
            else:
                self._polls.setdefault(int(entry["n_poll"]), []).append(event)
        self._next = 0
        self.n_cmd = 0
        self.n_poll = 0

    def header(self) -> dict[str, Any]:
        return dict(self._header)

    def command(self, commands: Sequence[Mapping[str, Any]], now_ms: int) -> list[dict[str, Any]]:
        self.n_cmd += 1
        return self._replies.pop(self.n_cmd, [])

    def poll(self, kind: str, now_ms: int, **params: Any) -> list[dict[str, Any]]:
        self.n_poll += 1
        return self._polls.pop(self.n_poll, [])

    def _released(self) -> tuple[int, int, int, dict[str, Any]] | None:
        if self._next >= len(self._pulls):
            return None
        entry = self._pulls[self._next]
        if entry[1] > self.n_cmd or entry[2] > self.n_poll:
            return None
        return entry

    def next_at_ms(self) -> int | None:
        entry = self._released()
        return None if entry is None else entry[0]

    def due(self, now_ms: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        while (entry := self._released()) is not None and due_second(entry[0]) * 1000 <= now_ms:
            out.append(entry[3])
            self._next += 1
        return out

    @property
    def exhausted(self) -> bool:
        """Every recorded percept has been taken back in."""
        return self._next >= len(self._pulls) and not self._replies and not self._polls


def replay_calls(core: Any, tape: CoreTape) -> int:
    """Repeat a tape's calls on ``core`` (built with ``port=TapePort(tape)`` and the same
    Config). A call recorded as raising must raise the same exception again (anything else is a
    :class:`ReplayError`, or the replay's own exception). Returns how many calls were made."""
    made = 0
    for entry in tape.calls():
        raised = entry.get("raised")
        try:
            getattr(core, str(entry["method"]))(*entry["args"], **entry["kwargs"])
        except Exception as error:
            if raised != type(error).__name__:
                raise
        else:
            if raised is not None:
                raise ReplayError(
                    f"call {made} ({entry['method']}) raised {raised} when it was recorded, "
                    "and not in the replay"
                )
        made += 1
    return made


# ── the world loop ─────────────────────────────────────────────────────────────


class Steppable(Protocol):
    def now_ms(self) -> int: ...

    def advance_to(self, t_ms: int, *, stop_on_command: bool = False) -> dict[str, Any]: ...

    def drain_outbox(self) -> list[Any]: ...


def run(
    core: Steppable, until_ms: int, *, on_commands: Callable[[list[Any]], None] | None = None
) -> None:
    """spec §7.4's world loop, reduced by design A1 to a wrapper: Duck Core pulls its world
    inside its own steps, so running it is advancing it. With ``on_commands`` it stops after
    every second that issued a command and hands those commands over (a real body's
    sender, a page that shows them)."""
    if on_commands is None:
        core.advance_to(until_ms)
        return
    while core.now_ms() < due_second(until_ms) * 1000:
        core.advance_to(until_ms, stop_on_command=True)
        sent = core.drain_outbox()
        if sent:
            on_commands(sent)


__all__ = [
    "ADVANCE_TO",
    "CoreTape",
    "POLLS",
    "POLL_PLAY_MINUTE",
    "POLL_TV_MINUTE",
    "QueuePort",
    "ReplayError",
    "TAPE_API",
    "TapePort",
    "VIAS",
    "VIA_INPUT",
    "VIA_POLL",
    "VIA_PULL",
    "VIA_REPLY",
    "WORLDPORT_API",
    "WorldError",
    "WorldPort",
    "at_ms",
    "calib_hash",
    "canonical",
    "check",
    "due_second",
    "epoch_ms",
    "known_kind",
    "make_world",
    "register_world",
    "registered_worlds",
    "replay_calls",
    "run",
    "validate",
]
