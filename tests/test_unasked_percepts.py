"""Percepts nobody asked for, and the body's own counts (phase-4 review, minor findings on the
act rules; spec §7.2).

* A result that names no act is *unasked*, not stale: phase 5's duck-eye reports the nest the
  moment it comes into view (an edge-triggered ``sense.sighting`` without ``act_id``), and the
  duck fixes its pose from it, as ``nest.on_sighting`` says. A result naming an act that is not
  open stays stale.
* The body counts its own righting tries: phase 5's duck-body reports only the last one
  (``tries: 3``), which must end the fall as stuck, not count as a first try.
* An act Duck Core gave up on is stopped at the body (a walk it never heard the end of must not
  go on, or fall, under a duck that believes it stands).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from phase4_harness import open_v41, send, walking_age

from duckmind.duckcore import acts as A
from duckmind.worldport import QueuePort

PV = "p:00112233aabbccdd"


def _sighting(t_ms: int, pv: str, **extra: Any) -> dict[str, Any]:
    return {
        "kind": "sense.sighting",
        "src": "legacy",
        "t_ms": t_ms,
        "pv": pv,
        "what": "nest",
        "b_mrad": 0,
        "conf_ppm": 1_000_000,
        "look_id": "L1",
        "edge": "new",
        "r_mm": 300,
        "r_src": "tag",
        **extra,
    }


def test_an_unasked_nest_sighting_fixes_the_pose_and_a_late_answer_is_stale() -> None:
    session = open_v41({"seed": 3, "nest": True})
    core = session.core
    core.s.pose_err_mm = 400
    pv = session.port.pv
    out = send(session, "sense", {"events": [_sighting(core.now_ms(), pv)]})
    taken = out["taken"][0]
    record = core.log.get(taken["event"])
    assert taken["taken"] is True and "stale" not in record.payload
    assert core.s.pose_err_mm <= 60, "the nest seen by itself fixes where it is"
    core.s.pose_err_mm = 400
    out = send(session, "sense", {"events": [_sighting(core.now_ms(), pv, act_id="A999")]})
    record = core.log.get(out["taken"][0]["event"])
    assert out["taken"][0]["taken"] is False and record.payload["stale"] is True
    assert core.s.pose_err_mm == 400


def test_the_bodys_own_count_of_righting_tries_is_taken() -> None:
    """Only the last try reported (tries 3, not up): the fall ends stuck, as after three."""
    session = open_v41({"seed": 5, "fall_ppm_per_s": 1_000_000, "righting_ppm": 0})
    walking_age(session)
    core = session.core
    for _ in range(3600):
        send(session, "advance", {"seconds": 1})
        if core.s.righting is not None:
            break
    assert core.s.righting is not None and core.s.righting["tries"] == 0, "no fall to get up from"
    core.attach_port(QueuePort(src="robot"))  # (the body reports only how it ended)
    last = {
        "kind": "body.righting",
        "src": "robot",
        "t_ms": core.now_ms(),
        "pv": PV,
        "ok": False,
        "ms": 9000,
        "tries": 3,
        "how": "wait",
    }
    send(session, "sense", {"events": [last]})
    assert core.s.righting is None and core.s.posture == "fallen"
    assert core.s.need is not None and core.s.need["kind"] == "help", "stuck: it asks for a hand"


class DeafToWalks:
    """LegacyWorld for everything but walks, which it never answers; keeps every stop."""

    def __init__(self, world: Any) -> None:
        self.world = world
        self.src = world.src
        self.stops: list[dict[str, Any]] = []

    def header(self) -> dict[str, Any]:
        return dict(self.world.header())

    def command(self, commands: Sequence[Mapping[str, Any]], now_ms: int) -> list[dict[str, Any]]:
        self.stops += [dict(c) for c in commands if c.get("cmd") == A.STOP]
        rest = [c for c in commands if c.get("cmd") != A.WALK]
        return list(self.world.command(rest, now_ms)) if rest else []

    def next_at_ms(self) -> int | None:
        return self.world.next_at_ms()

    def due(self, now_ms: int) -> list[dict[str, Any]]:
        return list(self.world.due(now_ms))

    def poll(self, kind: str, now_ms: int, **params: Any) -> list[dict[str, Any]]:
        return list(self.world.poll(kind, now_ms, **params))


def test_a_walk_duck_core_gave_up_on_is_stopped_at_the_body() -> None:
    session = open_v41({"seed": 5, "nest": False})
    core = session.core
    port = DeafToWalks(core.acts.port)
    core.attach_port(port)
    walking_age(session)
    before = len(core.log.events)
    for _ in range(400):
        send(session, "advance", {"seconds": 30})
        if any(e.kind == A.TIMEOUT_KIND for e in core.log.events[before:]):
            break
    timeout = next(e for e in core.log.events[before:] if e.kind == A.TIMEOUT_KIND)
    act_id = timeout.payload["act_id"]
    stops = [
        e
        for e in core.log.events[before:]
        if e.kind == "duck.act"
        and e.payload.get("cmd") == A.STOP
        and e.payload.get("act_id") == act_id
    ]
    assert stops and stops[0].payload["why"] == "timeout"
    assert timeout.event_id in stops[0].parent_event_ids
    assert any(wire.get("act_id") == act_id for wire in port.stops), "the body was never told"


def test_a_sender_on_another_clock_is_refused_not_followed() -> None:
    """api.sense lives the duck on to a percept that lies ahead, but not by more than a minute
    past its clock (or past the percept before it in the batch): the whole batch is refused."""
    import pytest

    from duckmind.duckcore import api

    session = open_v41({"seed": 3, "nest": True})
    core = session.core
    pv = session.port.pv
    now = core.now_ms()
    with pytest.raises(ValueError, match="another clock"):
        send(session, "sense", {"events": [_sighting(now + (api.SENSE_AHEAD_S + 1) * 1000, pv)]})
    assert core.now_ms() == now, "nothing of a refused batch is taken"
    steps = [_sighting(now + k * 50_000, pv) for k in (1, 2, 3)]  # (each within a minute)
    send(session, "sense", {"events": steps})
    assert core.now_ms() == now + 150_000
