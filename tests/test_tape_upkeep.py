"""The replay tapes on a long-lived page duck (phase-4 review, findings 2 and 4; design A8).

* **Time passing is one entry per stretch.** The page lives its duck a second at a time all
  day. Duck Core's input tape records time passing as ``advance_to(the second reached)`` and
  merges back-to-back steps (living on is the same whatever the step), so an hour of real time
  between two things someone did is one line, as in ``session.inputs``, and the tape alone still
  replays the same duck.
* **The outbox never grows without bound** when no driver drains it (the page).
* **A refused call never spoils a replay.** A call that raised stays on the tape marked with
  its exception; the replay expects the same exception there. The world refuses a touch it
  could not carry before it changes (no draw, nothing on its tape).
"""

from __future__ import annotations

import pytest
from phase4_harness import act, open_v41, send, walking_age

from duckmind.duckcore import acts as acts_module
from duckmind.worldport import ADVANCE_TO, ReplayError


def _page_duck(seed: int = 7) -> object:
    session = open_v41({"seed": seed, "nest": True})
    walking_age(session)
    send(session, "advance", {"until": "09:00"})
    return session


def test_an_hour_of_real_time_is_one_tape_entry_and_replays_the_same(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from duckmind.duckcore import api

    monkeypatch.setattr(acts_module, "OUTBOX_MAX", 8)  # (the page never drains it)
    session = _page_duck()
    core = session.core  # type: ignore[attr-defined]
    calls_before = len(list(core.tape.calls()))
    commands_before = core.acts.n_cmd
    for _ in range(3600):
        send(session, "advance", {"seconds": 1})
    calls = list(core.tape.calls())
    assert len(calls) - calls_before <= 1, calls[calls_before:][:5]
    assert calls[-1]["method"] == ADVANCE_TO and calls[-1]["args"] == [core.now_ms()]
    assert core.acts.n_cmd - commands_before > 16, "a walking morning issues commands"
    assert len(core.outbox) <= 2 * acts_module.OUTBOX_MAX
    # the page's second-by-second hour is the same duck in one piece …
    assert api.percept_replay_check(session)["identical"]  # type: ignore[arg-type]
    assert api.replay_check(session)["identical"]  # type: ignore[arg-type]
    # … and a thing someone did starts a new entry
    act(session, "touch", "owner", where="crown", seconds=3)
    send(session, "advance", {"seconds": 1})
    send(session, "advance", {"seconds": 1})
    methods = [entry["method"] for entry in core.tape.calls()][-3:]
    assert methods[-1] == ADVANCE_TO and ADVANCE_TO not in methods[-2:-1]
    assert api.percept_replay_check(session)["identical"]  # type: ignore[arg-type]


def test_refused_calls_stay_on_the_tape_and_replay_as_refused() -> None:
    """The review's poison case: a setting Duck Core does not know and an action with a
    parameter it does not take both raise; the tape keeps them (marked) and replays them."""
    from duckmind.duckcore import api

    session = open_v41({"seed": 7})
    walking_age(session)
    core = session.core
    with pytest.raises(ValueError, match="unknown setting"):
        send(session, "set_setting", {"key": "ear", "value": False})
    with pytest.raises(TypeError):
        send(session, "act", {"action": "wake", "params": {"bogus": 1}, "persona": "owner"})
    marked = [entry for entry in core.tape.calls() if "raised" in entry]
    assert [(e["method"], e["raised"]) for e in marked] == [
        ("set_setting", "ValueError"),
        ("wake", "TypeError"),
    ]
    send(session, "advance", {"minutes": 30})
    result = api.percept_replay_check(session)
    assert result["identical"], result
    assert api.replay_check(session)["identical"]
    # a replay that does not raise where the recording did is caught, not waved through
    marked[0]["raised"] = "KeyError"
    try:
        wrong = api.percept_replay_check(session)
    finally:
        marked[0]["raised"] = "ValueError"
    assert wrong["identical"] is False and "ValueError" in wrong["error"]


def test_a_replay_that_does_not_raise_where_the_recording_did_is_an_error() -> None:
    from duckmind.worldport import CoreTape, replay_calls

    class Duck:
        def ping(self) -> None:
            return None

    tape = CoreTape()
    entry = tape.call(0, "ping")
    tape.raised(entry, ValueError("was refused"))
    with pytest.raises(ReplayError, match="raised ValueError"):
        replay_calls(Duck(), tape)


def test_the_world_refuses_a_touch_it_could_not_carry_before_it_changes() -> None:
    from duckmind.duckcore import api

    session = open_v41({"seed": 7})
    walking_age(session)
    world = session.port
    counters, tape = dict(world.counters), len(world.tape)
    with pytest.raises(ValueError, match="a touch of -3 seconds"):
        act(session, "touch", "owner", where="crown", seconds=-3)
    assert dict(world.counters) == counters and len(world.tape) == tape
    assert api.world_replay_check(session)["identical"]
    assert api.percept_replay_check(session)["identical"]


def test_the_ear_is_hardware_not_a_world_setting() -> None:
    from duckmind.duckcore import api

    assert "ear" not in api.WORLD_SETTINGS
