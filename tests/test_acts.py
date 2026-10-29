"""Acts: Duck Core's commands and their results (phase-4 design A2/A7/A11, spec §7.1–§7.3; test
T3).

* one body act at a time (body.sock refuses a second one: ``busy``), unless it preempts;
* every act ends with exactly one result percept carrying its ``act_id``; a second one, or a
  result nobody is waiting for, is *stale* and changes nothing;
* ``duck.stop`` is answered in the same call (``walk.done {reason: preempt}``);
* a silent world lets the deadline pass: the act is closed by Duck Core's own
  ``duck.act_timeout``, and a result that comes later is stale;
* a command is exactly an event with an ``act_id`` that goes into the outbox (A7), and what the
  port receives is that event's payload as a wire dict.

The first part drives :class:`duckmind.duckcore.acts.Acts` with a small host and hand-made
ports; the second drives a whole v4_1 Duck Core once the engine has landed (WP1).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from duckmind import worldport
from duckmind.duckcore import acts as A
from duckmind.worldport import VIA_POLL, VIA_REPLY, QueuePort, WorldError

T0 = 1_790_000_000_000  # a whole second, in epoch ms
PV = "l:00112233aabbccdd"


def _percept(kind: str, t_ms: int, **fields: Any) -> dict[str, Any]:
    return worldport.check({"kind": kind, "src": "legacy", "t_ms": t_ms, "pv": PV, **fields})


def _walk_done(act_id: str, t_ms: int, reason: str = "time", max_ms: int = 3000) -> dict[str, Any]:
    return _percept(
        "walk.done",
        t_ms,
        act_id=act_id,
        cmd={"vx": 150, "vyaw": 0, "max_ms": max_ms},
        odom={"dx_mm": 0, "dy_mm": 0, "dyaw_mrad": 0, "path_mm": 0},
        steps=0,
        reason=reason,
    )


def _posture(act_id: str, t_ms: int, got: str = "stand") -> dict[str, Any]:
    return _percept("body.posture", t_ms, want="stand", got=got, ms=0, fell=False, act_id=act_id)


@dataclass
class _Event:
    event_id: str
    kind: str
    payload: dict[str, Any]
    parents: tuple[str, ...]


@dataclass
class Host:
    """What Duck Core does for :class:`Acts`: a clock, its event log, and its input handler
    (which settles results the way ``DuckCore.on_percept`` does)."""

    t_ms: int = T0
    events: list[_Event] = field(default_factory=list)
    delivered: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    settled: list[str] = field(default_factory=list)
    stale: list[dict[str, Any]] = field(default_factory=list)
    acts: A.Acts | None = None

    def now_ms(self) -> int:
        return self.t_ms

    def emit_command(self, kind: str, payload: dict[str, Any], parents: tuple[str, ...]) -> _Event:
        event = _Event(f"E{len(self.events) + 1:06d}", kind, dict(payload), tuple(parents))
        self.events.append(event)
        return event

    def deliver(self, percepts: list[dict[str, Any]], via: str) -> None:
        assert self.acts is not None
        for event in percepts:
            self.delivered.append((via, event))
            if self.acts.is_stale(event):
                self.stale.append(event)
                continue
            act = self.acts.settle(event)
            if act is not None:
                self.settled.append(act.act_id)


class AnsweringPort(QueuePort):
    """A world that answers a stand, a dock and a look at once, schedules a walk's end, and
    answers ``duck.stop`` with the stopped walk's ``walk.done {reason: preempt}``."""

    def __init__(self) -> None:
        super().__init__(src="legacy")
        self.walk: dict[str, Any] | None = None
        self.polls: list[tuple[str, int, dict[str, Any]]] = []

    def command(self, commands: Sequence[Mapping[str, Any]], now_ms: int) -> list[dict[str, Any]]:
        super().command(commands, now_ms)
        out: list[dict[str, Any]] = []
        for wire in commands:
            cmd = wire["cmd"]
            if cmd == A.WALK:
                self.walk = dict(wire)
                self.push(_walk_done(str(wire["act_id"]), now_ms + int(wire["max_ms"])))
            elif cmd == A.POSTURE:
                out.append(_posture(str(wire["act_id"]), now_ms))
            elif cmd == A.STOP and self.walk is not None:
                self._queue.clear()  # the scheduled end will not come
                out.append(_walk_done(str(self.walk["act_id"]), now_ms, "preempt"))
                self.walk = None
        return out

    def poll(self, kind: str, now_ms: int, **params: Any) -> list[dict[str, Any]]:
        self.polls.append((kind, now_ms, params))
        if kind == worldport.POLL_PLAY_MINUTE:
            return [_percept("body.play", now_ms, policy="walk", meters=4)]
        return []


def _acts(port: worldport.WorldPort | None) -> tuple[Host, A.Acts]:
    host = Host()
    acts = A.Acts(host, port)
    host.acts = acts
    return host, acts


WALK_FIELDS = {"verb": "wander", "vx_mm": 150, "vyaw_mrad": 0, "max_ms": 3000, "purpose": "wander"}


# ---------------------------------------------------------------------------- Acts alone


def test_a_command_is_an_event_with_an_act_id_that_goes_into_the_outbox() -> None:
    port = AnsweringPort()
    host, acts = _acts(port)
    walk = acts.start(A.WALK, WALK_FIELDS, parents=("E000000",))
    assert [e.kind for e in host.events] == ["duck.walk"]
    event = host.events[0]
    assert event.payload["act_id"] == walk.act_id and walk.event_id == event.event_id
    assert event.payload["deadline_ms"] == T0 + 3000 + A.WALK_SLACK_MS == walk.deadline_ms
    assert event.parents == ("E000000",)
    assert "cmd" not in event.payload  # a walk is written as duck.walk, not duck.act
    # what the port got is the event's payload as a wire dict
    assert port.sent == [(T0, {"cmd": A.WALK, **event.payload})] and walk.wire == port.sent[0][1]
    assert acts.drain_outbox() == [event.event_id] and acts.drain_outbox() == []
    assert acts.busy() and acts.body_act() is walk and acts.open_act(A.WALK) is walk
    assert acts.next_deadline_ms() == walk.deadline_ms


def test_other_commands_are_duck_act_events_named_by_their_body_sock_command() -> None:
    port = AnsweringPort()
    host, acts = _acts(port)
    for cmd, fields in (
        (A.POSTURE, {"want": "stand"}),
        (A.DOCK, {"bel_err_mm": 120}),
        (A.LOOK, {"kind": "seek_nest", "want": "nest", "bel_pose": [0, 0, 0]}),
    ):
        act = acts.start(cmd, fields, preempt=True)
        assert host.events[-1].kind == A.ACT_KIND == "duck.act"
        assert host.events[-1].payload["cmd"] == cmd
        assert act.deadline_ms == T0 + A.DEADLINE_MS[cmd]
    assert set(A.EVENT_KIND.values()) | {A.ACT_KIND} == A.COMMAND_KINDS
    with pytest.raises(ValueError, match="not an act command"):
        acts.start(A.STOP, {})


def test_one_body_act_at_a_time_unless_it_preempts() -> None:
    port = AnsweringPort()
    _host, acts = _acts(port)
    walk = acts.start(A.WALK, WALK_FIELDS)
    with pytest.raises(A.ActBusy, match="still open"):
        acts.start(A.POSTURE, {"want": "stand"})
    assert len(port.sent) == 1  # the refused act never reached the body
    look = acts.start(A.LOOK, {"kind": "scan", "want": "none"}, body=False)  # not a body act
    assert look.act_id in acts.open and acts.body_act() is walk
    preempting = acts.start(A.POSTURE, {"want": "stand"}, preempt=True)
    assert port.sent[-1][1]["preempt"] is True and preempting.done


def test_an_answer_in_the_same_call_closes_the_act_before_start_returns() -> None:
    port = AnsweringPort()
    host, acts = _acts(port)
    stand = acts.start(A.POSTURE, {"want": "stand"})
    assert stand.done and stand.outcome == "done" and not acts.busy()
    assert stand.result is not None and stand.result["kind"] == "body.posture"
    assert host.delivered == [(VIA_REPLY, stand.result)] and host.settled == [stand.act_id]
    assert acts.n_cmd == 1


def test_every_act_has_exactly_one_result_and_a_second_is_stale() -> None:
    port = AnsweringPort()
    host, acts = _acts(port)
    walk = acts.start(A.WALK, WALK_FIELDS)
    later = port.due(T0 + 3000)
    assert [p["kind"] for p in later] == ["walk.done"]
    host.deliver(later, worldport.VIA_PULL)
    assert walk.done and walk.result == later[0] and not acts.open
    again = _walk_done(walk.act_id, T0 + 4000)
    assert acts.is_stale(again) and acts.settle(again) is None and acts.stale == 1
    assert walk.result == later[0]  # the stale one changed nothing
    # a result of the wrong kind for an open act is stale too
    stand = acts.start(A.POSTURE, {"want": "stand"}, act_id="P1")
    assert stand.done
    wrong = _walk_done("P1", T0 + 5000)
    assert acts.is_stale(wrong) and acts.settle(wrong) is None
    # a percept that answers no command is not a result at all
    assert not acts.is_stale(_percept("body.play", T0, policy="walk", meters=3))


def test_a_stop_is_answered_at_once_with_the_walk_it_ended() -> None:
    port = AnsweringPort()
    host, acts = _acts(port)
    walk = acts.start(A.WALK, WALK_FIELDS)
    host.t_ms = T0 + 1000
    stop = acts.stop(walk.act_id, "lifted", parents=(walk.event_id,))
    assert stop.kind == "duck.act" and dict(stop.payload) == {
        "cmd": A.STOP,
        "why": "lifted",
        "act_id": walk.act_id,
    }
    assert walk.done and walk.result is not None and walk.result["reason"] == "preempt"
    assert walk.result["t_ms"] == T0 + 1000 and port.next_at_ms() is None  # nothing later
    assert acts.drain_outbox() == [walk.event_id, stop.event_id] and acts.n_cmd == 2
    # a stop without an act (LegacyWorld: the self-righting tries) is a command too
    bare = acts.stop(None, "power_off")
    assert "act_id" not in bare.payload and port.sent[-1][1] == {"cmd": A.STOP, "why": "power_off"}


def test_a_silent_world_lets_the_deadline_close_the_act() -> None:
    port = QueuePort(src="legacy")  # silent: answers nothing, ever
    host, acts = _acts(port)
    walk = acts.start(A.WALK, WALK_FIELDS)
    stand_deadline = walk.deadline_ms
    assert acts.expire(stand_deadline - 1000) == [] and acts.busy()
    closed = acts.expire(stand_deadline)
    assert closed == [walk] and walk.outcome == "timeout" and acts.timeouts == 1
    assert not acts.busy()
    late = _walk_done(walk.act_id, stand_deadline + 2000)
    host.deliver([late], worldport.VIA_PULL)
    assert host.stale == [late] and walk.result is None and acts.stale == 0  # (seen first)
    assert acts.settle(late) is None and acts.stale == 1


def test_deadlines_expire_in_order_and_at_their_whole_second() -> None:
    _host, acts = _acts(QueuePort(src="legacy"))
    a = acts.start(A.LOOK, {"kind": "scan"}, body=False, deadline_ms=T0 + 2500)
    b = acts.start(A.DOCK, {"bel_err_mm": 0}, deadline_ms=T0 + 1200)
    assert acts.expire(T0 + 1000) == []
    assert acts.expire(T0 + 2000) == [b]  # due second of 1200 ms past T0 is T0 + 2 s
    assert acts.expire(T0 + 3000) == [a]


def test_polls_are_counted_and_their_answers_taken_in() -> None:
    port = AnsweringPort()
    host, acts = _acts(port)
    host.t_ms = T0 + 60_000
    answer = acts.poll(worldport.POLL_PLAY_MINUTE, minute=3)
    assert [p["kind"] for p in answer] == ["body.play"] and acts.n_poll == 1
    assert port.polls == [(worldport.POLL_PLAY_MINUTE, T0 + 60_000, {"minute": 3})]
    assert host.delivered == [(VIA_POLL, answer[0])]
    assert acts.poll(worldport.POLL_TV_MINUTE) == [] and acts.n_poll == 2


def test_a_duck_without_a_world_says_so() -> None:
    _host, acts = _acts(None)
    with pytest.raises(WorldError, match="no world port"):
        acts.start(A.WALK, WALK_FIELDS)
    with pytest.raises(WorldError):
        acts.stop(None, "stop")
    with pytest.raises(WorldError):
        acts.poll(worldport.POLL_TV_MINUTE)


def test_the_act_contract_mirrors_body_sock() -> None:
    from duckmind.duckbody import protocol

    assert set(A.ACT_COMMANDS) == set(protocol.ACT_COMMANDS)
    assert A.RESULT_KINDS == protocol.RESULT_KINDS
    assert {A.STOP, *A.ACT_COMMANDS} <= set(protocol.COMMANDS)


# ---------------------------------------------------------------------------- a whole duck


def test_a_v41_duck_keeps_one_body_act_and_one_result_each() -> None:
    """A lively v4_1 duck with LegacyWorld: every act it issued got exactly one result record,
    no act was left open past its deadline, and two body acts never overlapped."""
    from phase4_harness import lively_v41

    session = lively_v41(seed=11, nest=True, stress=True, days=1)
    events = list(session.core.log.events)
    commands = {
        e.payload["act_id"]: e
        for e in events
        if e.kind in A.COMMAND_KINDS and "act_id" in e.payload and e.payload.get("cmd") != A.STOP
    }
    results: dict[str, list[str]] = {}
    for e in events:
        if e.kind in A.RESULTS and isinstance(e.payload.get("act_id"), str):
            results.setdefault(str(e.payload["act_id"]), []).append(e.kind)
    assert commands, "the duck issued no act"
    missing = [act_id for act_id in commands if act_id not in results]
    open_now = set(session.core.acts.open)
    assert set(missing) <= open_now, f"acts without a result: {sorted(set(missing) - open_now)}"
    assert all(len(kinds) == 1 for kinds in results.values()), {
        k: v for k, v in results.items() if len(v) != 1
    }
    assert not [e for e in events if e.kind == A.TIMEOUT_KIND], "LegacyWorld is never late"


class DeafToWalks:
    """A world whose legs never answer a walk (LegacyWorld for everything else: stand-ups,
    docks and looks are still answered at once, which a phase-4 Duck Core needs). Late
    percepts can be pushed in by hand."""

    def __init__(self, world: Any) -> None:
        self.world = world
        self.src = world.src
        self.swallowed: list[dict[str, Any]] = []
        self.late = QueuePort(src="robot")

    def header(self) -> dict[str, Any]:
        return dict(self.world.header())

    def command(self, commands: Sequence[Mapping[str, Any]], now_ms: int) -> list[dict[str, Any]]:
        walks = [dict(c) for c in commands if c.get("cmd") == A.WALK]
        self.swallowed += walks
        rest = [c for c in commands if c.get("cmd") != A.WALK]
        return list(self.world.command(rest, now_ms)) if rest else []

    def next_at_ms(self) -> int | None:
        times = [t for t in (self.world.next_at_ms(), self.late.next_at_ms()) if t is not None]
        return min(times) if times else None

    def due(self, now_ms: int) -> list[dict[str, Any]]:
        out = list(self.world.due(now_ms)) + self.late.due(now_ms)
        return sorted(out, key=lambda p: int(p["t_ms"]))

    def poll(self, kind: str, now_ms: int, **params: Any) -> list[dict[str, Any]]:
        return list(self.world.poll(kind, now_ms, **params))


def test_a_walk_the_world_never_ends_is_closed_by_duck_act_timeout_and_its_late_end_is_stale() -> (
    None
):
    from phase4_harness import open_v41, send, walking_age

    session = open_v41({"seed": 5, "nest": False})
    core = session.core
    port = DeafToWalks(core.acts.port)
    core.attach_port(port)
    walking_age(session)
    before = len(core.log.events)
    for _ in range(400):  # until it has set off and given up waiting for the walk to end
        send(session, "advance", {"seconds": 30})
        if any(e.kind == A.TIMEOUT_KIND for e in core.log.events[before:]):
            break
    timeouts = [e for e in core.log.events[before:] if e.kind == A.TIMEOUT_KIND]
    assert timeouts, "no walk timed out in a world that never ends one"
    first = timeouts[0]
    act_id = str(first.payload["act_id"])
    assert first.payload["cmd"] == A.WALK
    assert first.source.kind == "duck-self"  # Duck Core's own decision, not a sensation
    command = next(
        e for e in core.log.events if e.kind == "duck.walk" and e.payload.get("act_id") == act_id
    )
    assert any(wire["act_id"] == act_id for wire in port.swallowed)
    assert first.occurred_at >= command.occurred_at
    assert (
        worldport.epoch_ms(first.occurred_at)
        == -(-int(command.payload["deadline_ms"]) // 1000) * 1000
    )
    # its end arrives late: recorded as stale, and the duck's belief does not move
    state = (core.s.posture, core.s.pos_x, core.s.pos_y, core.s.heading, core.s.walk_until)
    n = len(core.log.events)
    late = {**_walk_done(act_id, core.now_ms() + 1000), "src": "robot", "pv": "p:00112233aabbccdd"}
    port.late.push(late)
    send(session, "advance", {"seconds": 1})
    records = [e for e in core.log.events[n:] if e.payload.get("act_id") == act_id]
    assert [(e.kind, e.payload.get("stale")) for e in records] == [("walk.done", True)], records
    after = (core.s.posture, core.s.pos_x, core.s.pos_y, core.s.heading, core.s.walk_until)
    assert after == state


def test_a_phase4_duck_refuses_a_world_that_does_not_answer_a_stand_up_at_once() -> None:
    """Phase 4 continues a stand-up, a dock try and a nest look in place (A2): a world that
    answers them later is a phase-5c world, and Duck Core says so instead of guessing."""
    from phase4_harness import open_v41, walking_age

    session = open_v41({"seed": 5, "nest": False})
    session.core.attach_port(QueuePort(src="robot"))
    with pytest.raises(WorldError, match="at once"):
        walking_age(session)
