"""Step-size invariance (phase-4 design §2.3, A1; test T1): how time is cut does not change the
duck.

The same inputs at the same seconds, with the time between them passed in one piece, in random
pieces of 1–600 s, or by ``advance_to(stop_on_command=True)`` (returning after every second that
issued a command, then going on): the event chain must be identical to the byte. This is what
pulling the world *inside* Duck Core's own time step buys (a walk that ends inside a 6-second
touch still ends at its second); a world loop between inputs would not have it.

Thirty seeds by default (``DUCKMIND_P4_SEEDS``), a third each: a nest duck with falls 600×
v4, weak righting and failed stand-ups; the same without a nest; a nest duck at v4's own rates.
Each is grown to the day it walks and then given a busy scripted day.
"""

from __future__ import annotations

import os

import pytest
from phase4_harness import Drive, lively, lively_config, open_v41, walking_age

SEEDS = [101 + 97 * n for n in range(int(os.environ.get("DUCKMIND_P4_SEEDS", "30")))]
KINDS = ("nest-stress", "floor-stress", "nest")


def _config(seed: int, index: int) -> dict[str, object]:
    kind = KINDS[index % 3]
    return lively_config(seed, nest=kind.startswith("nest"), stress=kind.endswith("stress"))


def _life(seed: int, index: int, mode: str) -> tuple[str, int, list[str], list[str]]:
    from duckmind.duckcore.acts import COMMAND_KINDS

    session = open_v41(_config(seed, index))
    walking_age(session)
    drive = Drive(mode, seed)
    start = len(session.core.log.events)
    session.core.drain_outbox()
    lively(session, seed, days=1, drive=drive)
    drive.drained += session.core.drain_outbox()
    log = session.core.log
    commands = [e.event_id for e in log.events[start:] if e.kind in COMMAND_KINDS]
    return log.head, len(log.events), commands, drive.drained


@pytest.mark.parametrize("index", range(len(SEEDS)), ids=[f"seed{s}" for s in SEEDS])
def test_the_chain_does_not_depend_on_how_time_is_cut(index: int) -> None:
    seed = SEEDS[index]
    head, events, commands, _ = _life(seed, index, "one")
    for mode in ("pieces", "stop"):
        other_head, other_events, other_commands, drained = _life(seed, index, mode)
        assert (other_head, other_events) == (head, events), (
            f"seed {seed} ({KINDS[index % 3]}): time passed as '{mode}' gives "
            f"{other_events} events, head {other_head[:16]}; in one piece {events}, {head[:16]}"
        )
        assert other_commands == commands
        if mode == "stop":
            # every command came out of the outbox once, in order, and nothing else did
            assert drained == commands, "the outbox and the chain's commands differ"
    ISSUED[seed] = len(commands)


# commands each seed's day issued (a duck switched off by a hand may sit its day out; the
# check below makes sure the days as a whole are not vacuous)
ISSUED: dict[int, int] = {}


def test_the_scripted_days_were_not_vacuous() -> None:
    if len(ISSUED) < len(SEEDS):
        pytest.skip("runs after the whole seed set (it reads what those days issued)")
    busy = sum(1 for count in ISSUED.values() if count > 0)
    assert busy >= len(SEEDS) * 3 // 4, f"only {busy} of {len(SEEDS)} days issued a command"
    assert sum(ISSUED.values()) >= 20 * len(SEEDS), ISSUED


def test_advance_to_stops_right_after_the_second_of_a_command() -> None:
    from duckmind.duckcore.acts import COMMAND_KINDS

    session = open_v41(_config(SEEDS[0], 0))
    walking_age(session)
    core = session.core
    core.drain_outbox()
    target = core.now_ms() + 6 * 3600 * 1000
    stops = 0
    while core.now_ms() < target and stops < 50:
        before = len(core.log.events)
        result = core.advance_to(target, stop_on_command=True)
        assert isinstance(result, dict)
        sent = core.drain_outbox()
        if core.now_ms() < target:
            stops += 1
            assert sent, "advance_to stopped early without a command"
            new = core.log.events[before:]
            assert {e.event_id for e in new if e.kind in COMMAND_KINDS} >= set(sent)
            assert all(core.log.get(event_id).occurred_at == core.s.now for event_id in sent)
    assert stops > 0, "no command in six hours of a walking duck's day"
    assert core.next_due() > core.now_ms()
