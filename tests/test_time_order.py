"""Time order across Duck Core's own clock events (phase-4 review, finding 1; design A1, spec
§7.4's step-size invariance).

Duck Core pulls its world inside its own time step. What the world sensed must come in time
order with Duck Core's own clock events — a stir in its sleep, midnight's roll-over, 07:00 —
the world first only at the *same* second. LegacyWorld never has anything due across one of
them (walks never run at night, a napping duck neither walks nor lies waiting to get up), but a
queue, live or robot port does: a touch at 00:00:30 must count in the new day, after the day's
numbers were written, whatever the step size.

A nest duck sleeps through a night while a sender's percepts (touches, a head held) arrive at
random seconds, some right after midnight, some around 07:00; and ducks walk through a morning
(with a nest, without one, at 600× v4's falls) while a sender's percepts arrive at random. The
time between is passed in one piece, in random pieces, a second at a time, and by
``advance_to(stop_on_command=True)``: the chain must be the same to the byte, and its time must
never go backwards. (Whole sleeping minutes are lived at once only up to the next percept on its
way: that minute is lived as a minute, where a touch that wakes it takes the minute's rest.)
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from datetime import datetime, time, timedelta
from typing import Any

import pytest
from phase4_harness import PIECES, lively_config, open_v41, send, walking_age

from duckmind.worldport import QueuePort, epoch_ms

PV = "p:00112233aabbccdd"


class WorldAndQueue:
    """LegacyWorld for Duck Core's commands (it answers a stand-up at once, as phase 4 needs),
    plus the percepts a sender queued (a live simulator's or a robot's): both are due in time
    order, the world's first at an equal ``t_ms``."""

    def __init__(self, world: Any) -> None:
        self.world = world
        self.src = world.src
        self.queue = QueuePort(src="live")

    def header(self) -> dict[str, Any]:
        return dict(self.world.header())

    def command(self, commands: Sequence[Mapping[str, Any]], now_ms: int) -> list[dict[str, Any]]:
        return list(self.world.command(commands, now_ms))

    def next_at_ms(self) -> int | None:
        times = [t for t in (self.world.next_at_ms(), self.queue.next_at_ms()) if t is not None]
        return min(times) if times else None

    def due(self, now_ms: int) -> list[dict[str, Any]]:
        out = list(self.world.due(now_ms)) + self.queue.due(now_ms)
        return sorted(out, key=lambda p: int(p["t_ms"]))

    def poll(self, kind: str, now_ms: int, **params: Any) -> list[dict[str, Any]]:
        return list(self.world.poll(kind, now_ms, **params))


def _touch(t_ms: int, where: str = "crown") -> dict[str, Any]:
    return {
        "kind": "sense.touch",
        "src": "live",
        "t_ms": t_ms,
        "pv": PV,
        "where": where,
        "seconds": 2,
        "firmness": "light",
        "path": "ear",
        "latency_ms": 1100,
    }


def _hold(t_ms: int) -> dict[str, Any]:
    return {
        "kind": "sense.hold_head",
        "src": "live",
        "t_ms": t_ms,
        "pv": PV,
        "path": "joint+probe",
        "moving": False,
    }


def _night(seed: int, mode: str, replay: bool = False) -> tuple[str, int, list[datetime]]:
    session = open_v41({"seed": seed, "nest": True})
    send(session, "live", {"family": "elder_home", "days": 2})
    send(session, "advance", {"until": "23:50"})
    core = session.core
    port = WorldAndQueue(core.acts.port)
    core.attach_port(port)
    tz = core.s.now.tzinfo
    midnight = datetime.combine(core.s.day_date + timedelta(days=1), time(0), tz)
    seven = midnight + timedelta(hours=7)
    plan = random.Random(seed)
    moments = [midnight + timedelta(seconds=s) for s in (-30, 0, 1, 30, 95)]
    moments += [seven + timedelta(seconds=s) for s in (-1, 0, 1, 40)]
    moments += [midnight + timedelta(seconds=plan.randint(120, 7 * 3600 - 60)) for _ in range(24)]
    for index, moment in enumerate(sorted(moments)):
        make = _hold if index % 5 == 4 else _touch
        port.queue.push(make(epoch_ms(moment)))
    start = len(core.log.events)
    end = seven + timedelta(minutes=5)
    total = int((end - core.s.now).total_seconds())
    if mode == "one":
        core.advance_seconds(total)
    elif mode == "pieces":
        cut = random.Random(seed * 7 + 1)
        left = total
        while left > 0:
            piece = min(left, cut.choice(PIECES))
            core.advance_seconds(piece)
            left -= piece
    elif mode == "seconds":  # a second at a time around midnight and 07:00
        for until, fine in (
            (midnight - timedelta(seconds=40), False),
            (midnight + timedelta(seconds=100), True),
            (seven - timedelta(seconds=5), False),
            (seven + timedelta(seconds=45), True),
            (end, False),
        ):
            while core.s.now < until:
                core.advance_seconds(1 if fine else int((until - core.s.now).total_seconds()))
    else:
        target = epoch_ms(end)
        while core.now_ms() < target:
            core.advance_to(target, stop_on_command=True)
            core.drain_outbox()
    assert core.s.now == end
    if replay:  # the tape alone (no world, no sender) gives the same duck
        from duckmind.duckcore import api

        result = api.percept_replay_check(session)
        assert result["identical"] and result["all_percepts_taken"], result
    times = [event.occurred_at for event in core.log.events[start:]]
    return core.log.head, len(core.log.events), times


def _day(seed: int, mode: str, nest: bool, stress: bool) -> tuple[str, int, list[datetime]]:
    session = open_v41(lively_config(seed, nest, stress))
    walking_age(session)
    send(session, "advance", {"until": "09:00"})
    core = session.core
    port = WorldAndQueue(core.acts.port)
    core.attach_port(port)
    plan = random.Random(seed)
    begin = core.s.now
    for index in range(30):
        moment = begin + timedelta(seconds=plan.randint(5, 4 * 3600))
        port.queue.push((_hold if index % 4 == 3 else _touch)(epoch_ms(moment)))
    start = len(core.log.events)
    end = begin + timedelta(hours=4)
    total = int((end - core.s.now).total_seconds())
    if mode == "one":
        core.advance_seconds(total)
    elif mode == "pieces":
        cut = random.Random(seed * 7 + 1)
        left = total
        while left > 0:
            piece = min(left, cut.choice(PIECES))
            core.advance_seconds(piece)
            left -= piece
    elif mode == "seconds":
        for _ in range(total):
            core.advance_seconds(1)
    else:
        target = epoch_ms(end)
        while core.now_ms() < target:
            core.advance_to(target, stop_on_command=True)
            core.drain_outbox()
    assert core.s.now == end
    return core.log.head, len(core.log.events), [e.occurred_at for e in core.log.events[start:]]


def _same_whatever_the_cut(life: Any, label: str) -> None:
    head, events, times = life("one")
    assert all(b >= a for a, b in zip(times, times[1:], strict=False)), "chain time went back"
    for mode in ("pieces", "seconds", "stop"):
        other_head, other_events, other_times = life(mode)
        assert all(b >= a for a, b in zip(other_times, other_times[1:], strict=False)), mode
        assert (other_head, other_events) == (head, events), (
            f"{label}: time passed as '{mode}' gives {other_events} events, head "
            f"{other_head[:16]}; in one piece {events}, {head[:16]}"
        )


@pytest.mark.parametrize("seed", [5, 23, 101, 102, 103, 104])
def test_a_senders_percepts_come_in_time_order_with_midnight_the_stirs_and_7am(seed: int) -> None:
    _same_whatever_the_cut(lambda mode: _night(seed, mode), f"night, seed {seed}")


@pytest.mark.parametrize(
    ("seed", "nest", "stress"), [(200, True, False), (201, False, True), (202, True, True)]
)
def test_a_senders_percepts_while_it_walks_are_the_same_whatever_the_cut(
    seed: int, nest: bool, stress: bool
) -> None:
    _same_whatever_the_cut(
        lambda mode: _day(seed, mode, nest, stress), f"day, seed {seed}, nest {nest}"
    )


def test_a_touch_just_after_midnight_counts_in_the_new_day() -> None:
    """The review's case: a sender's touch at 00:00:30 comes after the day's roll-over (its
    growth.gd and duck.day_stats at 00:00:00), not before it."""
    session = open_v41({"seed": 5, "nest": True})
    send(session, "live", {"family": "elder_home", "days": 2})
    send(session, "advance", {"until": "23:59"})
    core = session.core
    port = WorldAndQueue(core.acts.port)
    core.attach_port(port)
    midnight = datetime.combine(core.s.day_date + timedelta(days=1), time(0), core.s.now.tzinfo)
    port.queue.push(_touch(epoch_ms(midnight + timedelta(seconds=30))))
    start = len(core.log.events)
    core.advance_seconds(180)
    kinds = [(e.occurred_at, e.kind) for e in core.log.events[start:]]
    rolled = next(i for i, (_at, kind) in enumerate(kinds) if kind == "growth.gd")
    touched = next(i for i, (_at, kind) in enumerate(kinds) if kind == "sense.touch")
    assert rolled < touched, kinds[:12]
    assert kinds[touched][0] == midnight + timedelta(seconds=30)


@pytest.mark.parametrize("seed", [7, 31])
def test_a_senders_night_replays_from_the_tape_alone(seed: int) -> None:
    """Whole sleeping minutes stop before a percept on its way in the replay too (its TapePort
    shows the same next percept), so the tape alone gives the same duck, cut any way."""
    for mode in ("one", "pieces"):
        _night(seed, mode, replay=True)
