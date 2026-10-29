"""Acts: Duck Core's commands to its body and the results that come back (phase 4, design
A2/A7/A11; spec §7.1–§7.3).

**What a command is (A7).** An event with an ``act_id`` that goes into the outbox:

* ``duck.walk`` — one walking segment (v4's walk record, extended with the body.sock
  ``WalkCmd`` fields ``act_id``, ``deadline_ms``, ``vx_mm``, ``vyaw_mrad``, ``max_ms``,
  ``bel_from``, ``expect``, ``purpose``);
* ``duck.act`` — every other body command, its ``cmd`` the body.sock command name:
  ``duck.posture {want}``, ``duck.dock {bel_err_mm}``, ``duck.look {kind, want, bel_pose}``,
  ``duck.stop {why, act_id?}`` (``act_id`` here names the act to stop);
* ``duck.curio {phase: step}`` — a curiosity step (curiosity spec §11), ``act_id`` ``C<n>``.

(``duck.posture`` and ``duck.stop`` stay v4's records of a posture change and of a hand holding
its head: a kind never means two things.)

**The handshake.** :meth:`Acts.start` writes the command event, puts it in the outbox, hands its
wire dict to the port and takes in, before it returns, whatever the port answered in the same
call (LegacyWorld answers a stand, a dock try and the nest look at once, A2: the caller then
continues in place exactly as v4 did). A later result arrives through the port's ``due`` inside
Duck Core's time step. Every act ends with exactly one result percept carrying its ``act_id``
(:data:`RESULT_KINDS`) — also when it was stopped (``walk.done {reason: preempt|lifted}``).

**Rules.** One body act at a time (the body.sock rule: :class:`ActBusy`). A result naming an act
no one is waiting for — late, or for an act already closed — is *stale*: Duck Core records it
and does not use it (a result naming no act is unasked, e.g. the eye seeing the nest by itself,
and taken as such). An act whose deadline second passes without a result is closed by Duck
Core's own decision ``duck.act_timeout`` (the world's percepts of that second come first), and
the body is told to stop it. Only outcome-bearing actions are acts in phase 4 (A11): walks
(end, fall), stand-up, the dock try and the nest look; sitting down stays Duck Core's own
posture record until phase 5c.

**The outbox** holds the ids of the command events a driver has not drained yet (a body that
answers later is handed them). A driver drains it every step; one that never does (the page,
whose world answers in-process) keeps only the latest :data:`OUTBOX_MAX`.

Standard library, :mod:`duckmind.provenance` and :mod:`duckmind.worldport` only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from duckmind.worldport import VIA_POLL, VIA_REPLY, WorldError, WorldPort, due_second

if TYPE_CHECKING:
    from duckmind.provenance import Event

# Mirrors duckmind.duckbody.protocol (COMMANDS, RESULT_KINDS); tests/test_command_shapes.py
# checks every command event against its parse_command.
WALK = "duck.walk"
POSTURE = "duck.posture"
DOCK = "duck.dock"
LOOK = "duck.look"
CURIO = "duck.curio"
STOP = "duck.stop"
ACT_COMMANDS = (WALK, LOOK, POSTURE, DOCK, CURIO)
RESULT_KINDS: dict[str, tuple[str, ...]] = {
    WALK: ("walk.done",),
    LOOK: ("sense.scan", "sense.sighting"),
    POSTURE: ("body.posture",),
    DOCK: ("body.dock",),
    CURIO: ("body.curio_step",),
}
RESULTS = frozenset(kind for kinds in RESULT_KINDS.values() for kind in kinds)
# The event kind a command is written as.
EVENT_KIND = {WALK: "duck.walk", CURIO: "duck.curio"}  # every other command: "duck.act"
ACT_KIND = "duck.act"
TIMEOUT_KIND = "duck.act_timeout"
COMMAND_KINDS = frozenset({"duck.walk", "duck.act", "duck.curio"})

# How long past its own planned length an act may take before Duck Core gives up on it.
# LegacyWorld always answers on time (a walk ends at its max_ms, the rest at once).
WALK_SLACK_MS = 5_000
DEADLINE_MS = {POSTURE: 10_000, DOCK: 60_000, LOOK: 5_000, CURIO: 60_000}
# Command ids kept for a driver that never drains the outbox (about ten days of commands).
OUTBOX_MAX = 4_096


class ActBusy(RuntimeError):
    """A body act was started while another one is open (body.sock refuses it: ``busy``)."""


@dataclass(slots=True)
class Act:
    """One command and, once it came, its result."""

    act_id: str
    cmd: str  # duck.walk | duck.posture | duck.dock | duck.look | duck.curio
    event_id: str  # the command event
    t_ms: int  # when it was issued
    deadline_ms: int
    wire: dict[str, Any]
    body: bool = True  # holds the body (one body act at a time)
    result: dict[str, Any] | None = None  # the result percept
    record_id: str | None = None  # Duck Core's input record of the result
    outcome: str = "open"  # open | done | timeout | failed (the port refused the command)
    extra: dict[str, Any] = field(default_factory=dict)  # the caller's own notes

    @property
    def done(self) -> bool:
        return self.outcome != "open"


class ActHost(Protocol):
    """What :class:`Acts` needs from Duck Core."""

    def now_ms(self) -> int: ...

    def emit_command(
        self, kind: str, payload: dict[str, Any], parents: tuple[str, ...]
    ) -> Event: ...

    def deliver(self, percepts: list[dict[str, Any]], via: str) -> None: ...


class Acts:
    """Duck Core's open acts, its outbox and its side of the port traffic.

    All of Duck Core's calls to its port go through here (``command`` via :meth:`start` and
    :meth:`stop`, ``poll`` via :meth:`poll`), so :attr:`n_cmd` and :attr:`n_poll` count them
    for the replay tape (:class:`duckmind.worldport.CoreTape`)."""

    def __init__(self, host: ActHost, port: WorldPort | None = None) -> None:
        self.host = host
        self.port = port
        self.seq = 0
        self.open: dict[str, Act] = {}
        self.outbox: list[str] = []  # command event ids not yet drained
        self.n_cmd = 0
        self.n_poll = 0
        self.timeouts = 0
        self.stale = 0

    # ------------------------------------------------------------------ queries

    def need_port(self) -> WorldPort:
        if self.port is None:
            raise WorldError(
                "this Duck Core has no world port: a v4_1 duck needs one to walk, stand up, "
                "dock or look (Session(config, world=…) / DuckCore(port=…))"
            )
        return self.port

    def body_act(self) -> Act | None:
        for act in self.open.values():
            if act.body:
                return act
        return None

    def busy(self) -> bool:
        """A body act is open: no other may start (``_motion_block`` → ``busy``)."""
        return self.body_act() is not None

    def open_act(self, cmd: str) -> Act | None:
        for act in self.open.values():
            if act.cmd == cmd:
                return act
        return None

    def next_deadline_ms(self) -> int | None:
        return min((act.deadline_ms for act in self.open.values()), default=None)

    # ------------------------------------------------------------------ commands

    def new_id(self) -> str:
        self.seq += 1
        return f"A{self.seq}"

    def start(
        self,
        cmd: str,
        fields: Mapping[str, Any],
        *,
        parents: tuple[str, ...] = (),
        act_id: str | None = None,
        deadline_ms: int | None = None,
        body: bool = True,
        preempt: bool = False,
    ) -> Act:
        """Issue one act: write its command event, put it in the outbox, hand it to the port
        and take in what the port answered at once. ``fields`` are the command's own fields
        (for a walk: v4's walk record plus the ``WalkCmd`` fields); ``act_id`` and
        ``deadline_ms`` are added here. Returns the act (``done`` when it was answered at once).
        """
        if cmd not in RESULT_KINDS:
            raise ValueError(f"{cmd!r} is not an act command {ACT_COMMANDS}")
        port = self.need_port()
        if body and not preempt:
            current = self.body_act()
            if current is not None:
                raise ActBusy(f"{cmd} while {current.cmd} {current.act_id} is still open")
        now = self.host.now_ms()
        act_id = act_id or self.new_id()
        if deadline_ms is None:
            if cmd == WALK:
                deadline_ms = now + int(fields["max_ms"]) + WALK_SLACK_MS
            else:
                deadline_ms = now + DEADLINE_MS[cmd]
        kind = EVENT_KIND.get(cmd, ACT_KIND)
        payload: dict[str, Any] = {} if kind != ACT_KIND else {"cmd": cmd}
        payload.update(fields)
        payload["act_id"] = act_id
        payload["deadline_ms"] = deadline_ms
        if preempt:
            payload["preempt"] = True
        event = self.host.emit_command(kind, payload, parents)
        wire = {"cmd": cmd, **payload}
        act = Act(act_id, cmd, event.event_id, now, deadline_ms, wire, body)
        self.open[act_id] = act
        self._out(event.event_id)
        try:
            replies = port.command([wire], now)
        except Exception:
            # The port refused it: the command was written, but no act is left waiting (a
            # later result for it is stale).
            del self.open[act_id]
            act.outcome = "failed"
            raise
        self.n_cmd += 1
        if replies:
            self.host.deliver(replies, VIA_REPLY)
        return act

    def stop(self, act_id: str | None, why: str, *, parents: tuple[str, ...] = ()) -> Event:
        """``duck.act {cmd: duck.stop, why, act_id?}``: stop the named act (its result comes
        back as usual, e.g. ``walk.done {reason: preempt}``), or with no act the body's own
        doings (LegacyWorld: the self-righting tries after a fall)."""
        port = self.need_port()
        now = self.host.now_ms()
        payload: dict[str, Any] = {"cmd": STOP, "why": why}
        if act_id is not None:
            payload["act_id"] = act_id
        event = self.host.emit_command(ACT_KIND, payload, parents)
        self._out(event.event_id)
        replies = port.command([{**payload}], now)
        self.n_cmd += 1
        if replies:
            self.host.deliver(replies, VIA_REPLY)
        return event

    def poll(self, kind: str, **params: Any) -> list[dict[str, Any]]:
        """Ask the port a poll (``tv_minute``, ``play_minute``) and take its answer in."""
        port = self.need_port()
        answer = port.poll(kind, self.host.now_ms(), **params)
        self.n_poll += 1
        if answer:
            self.host.deliver(answer, VIA_POLL)
        return answer

    def drain_outbox(self) -> list[str]:
        out, self.outbox = self.outbox, []
        return out

    def _out(self, event_id: str) -> None:
        outbox = self.outbox
        outbox.append(event_id)
        if len(outbox) > 2 * OUTBOX_MAX:  # (nobody drains it: keep the latest, cheaply)
            del outbox[: len(outbox) - OUTBOX_MAX]

    # ------------------------------------------------------------------ results

    def settle(self, event: Mapping[str, Any]) -> Act | None:
        """The open act this result percept ends (now closed), or None: not a result, or a
        stale one (no open act of that id wants that kind)."""
        kind = str(event.get("kind"))
        if kind not in RESULTS:
            return None
        act_id = event.get("act_id")
        if act_id is None:
            return None  # unasked (the eye saw the nest by itself): no act to close
        act = self.open.get(act_id) if isinstance(act_id, str) else None
        if act is None or kind not in RESULT_KINDS[act.cmd]:
            self.stale += 1
            return None
        del self.open[act.act_id]
        act.result = dict(event)
        act.outcome = "done"
        return act

    def is_stale(self, event: Mapping[str, Any]) -> bool:
        """A result percept naming an act no one is waiting for (checked before
        :meth:`settle`; a result naming no act is unasked, not stale)."""
        kind = str(event.get("kind"))
        if kind not in RESULTS:
            return False
        act_id = event.get("act_id")
        if act_id is None:
            return False
        act = self.open.get(act_id) if isinstance(act_id, str) else None
        return act is None or kind not in RESULT_KINDS[act.cmd]

    def expire(self, now_ms: int) -> list[Act]:
        """Close every open act whose deadline second is at or before ``now_ms`` (Duck Core
        writes ``duck.act_timeout`` for each; a result that comes later is stale)."""
        late = sorted(
            (act for act in self.open.values() if due_second(act.deadline_ms) * 1000 <= now_ms),
            key=lambda item: (item.deadline_ms, item.t_ms),
        )
        for act in late:
            del self.open[act.act_id]
            act.outcome = "timeout"
            self.timeouts += 1
        return late


__all__ = [
    "ACT_COMMANDS",
    "ACT_KIND",
    "COMMAND_KINDS",
    "CURIO",
    "DEADLINE_MS",
    "DOCK",
    "EVENT_KIND",
    "OUTBOX_MAX",
    "LOOK",
    "POSTURE",
    "RESULTS",
    "RESULT_KINDS",
    "STOP",
    "TIMEOUT_KIND",
    "WALK",
    "WALK_SLACK_MS",
    "Act",
    "ActBusy",
    "ActHost",
    "Acts",
]
