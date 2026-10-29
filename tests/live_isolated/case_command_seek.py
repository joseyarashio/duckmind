"""What the duck does when the person it heard is not in view (neuro/command_brain.py): it looks about
for them — the head sweeps, the body turns in place — within a bound, and goes on with the verb as soon
as its own eyes call one; not found, it says so and tilts its head and chirps that it heard. ``come`` and
``follow`` losing the person in the middle of the walk look about a moment before giving up. And an errand
(neuro/errand.py) holds the walk on the selector's persistent command channel until it arrives, whatever
else the duck wants."""

from __future__ import annotations

import math
from typing import Any

import pytest
from case_command_brain import _duck, _frame, _last, _person
from live_bridge_rig import person_sight, run_brain, thing_sight

from duckmind.neuro import command_brain as CB
from duckmind.neuro import errand as E
from duckmind.neuro.commands import Command
from duckmind.neuro.contract import wrap_angle

pytestmark = pytest.mark.usefixtures("no_network")

DT = 0.05  # (the rig's 20 Hz)


def _watch(brain: Any, states: list[str]) -> Any:
    """An ``on_tick`` that keeps every state the page saw."""

    def on_tick(t: float, cmd: Any) -> None:
        last = _last(brain)
        if last is not None:
            states.append(last["state"])

    return on_tick


# -- a person-verb heard with nobody in view --------------------------------------------------------


def test_a_person_verb_with_nobody_seen_looks_about_then_refuses() -> None:
    for verb in ("come", "look_at_me", "follow"):
        brain, clock = _duck(posture="stand")  # nobody in view, nobody believed
        states: list[str] = []
        ticket = brain.commands.hear(Command(verb))
        cmds = run_brain(
            brain,
            clock,
            CB.LOOK_FOR_S + 1.0,
            frames=lambda t: _frame(t, posture="stand"),
            on_tick=_watch(brain, states),
        )
        last = _last(brain)
        assert ticket.answer["state"] == "heard", verb
        assert "looking" in states, (verb, states)
        assert last["state"] == "refused" and last["reason"] == "no_one_seen", (verb, last)
        assert last["search_ms"] >= CB.LOOK_FOR_S * 1000.0 - 200.0, (verb, last)
        assert any(c.vocal == "inquire" for c in cmds), verb  # (it heard: a tilt and a chirp)


def test_a_person_behind_is_found_by_the_search_and_come_is_done() -> None:
    """The person stands behind it (the live probe: −150°, 1.58 m): the head sweeps, the body turns
    in place, and its own camera calls them within the bound."""
    brain, clock = _duck(posture="stand")
    body = {"yaw": 0.0}
    behind = math.pi

    def frames(t: float) -> Any:
        rel = wrap_angle(behind - body["yaw"])
        sight = (person_sight(1, bearing=rel, range_m=0.6),) if abs(rel) <= 0.5 else ()
        return _frame(t, posture="stand", sightings=sight, odom=(0.0, 0.0, body["yaw"]))

    def on_tick(t: float, cmd: Any) -> None:
        body["yaw"] = wrap_angle(body["yaw"] + cmd.vyaw * DT)

    brain.commands.hear(Command("come"))
    run_brain(brain, clock, CB.LOOK_FOR_S + 2.0, frames=frames, on_tick=on_tick)
    last = _last(brain)
    assert last["state"] == "done", last
    assert abs(body["yaw"]) > 2.0, body  # (it turned round to find them)
    assert last["search_ms"] < CB.LOOK_FOR_S * 1000.0, last


def test_a_seated_duck_stands_up_to_look_for_the_person_behind() -> None:
    """A seated duck cannot turn in place: it stands first, then sweeps and turns, and finds them."""
    brain, clock = _duck(posture="sit")
    body = {"yaw": 0.0, "posture": "sit"}
    behind = math.pi

    def frames(t: float) -> Any:
        rel = wrap_angle(behind - body["yaw"])
        sight = (person_sight(1, bearing=rel, range_m=0.6),) if abs(rel) <= 0.5 else ()
        return _frame(t, posture=body["posture"], sightings=sight, odom=(0.0, 0.0, body["yaw"]))

    def on_tick(t: float, cmd: Any) -> None:
        if brain.behaviour == "stand_up":
            body["posture"] = "stand"
        body["yaw"] = wrap_angle(body["yaw"] + cmd.vyaw * DT)

    brain.commands.hear(Command("come"))
    run_brain(brain, clock, CB.LOOK_FOR_S + 2.0, frames=frames, on_tick=on_tick)
    last = _last(brain)
    assert body["posture"] == "stand", body
    assert last["state"] == "done", last


GAIT_YAW = 0.365  # the live body's yaw for a commanded 1.0 (vx 0.4, measured 2026-10-23)


def test_a_person_behind_is_found_at_the_real_gait_s_turning_rate() -> None:
    """The live body turns about a third of the yaw it is asked for: the search is bounded by how
    far the body itself has turned (its odometry), not by a time that only a perfect gait fills."""
    brain, clock = _duck(posture="stand")
    body = {"yaw": 0.0}

    def frames(t: float) -> Any:
        rel = wrap_angle(math.pi - body["yaw"])
        sight = (person_sight(1, bearing=rel, range_m=0.6),) if abs(rel) <= 0.5 else ()
        return _frame(t, posture="stand", sightings=sight, odom=(0.0, 0.0, body["yaw"]))

    def on_tick(t: float, cmd: Any) -> None:
        body["yaw"] = wrap_angle(body["yaw"] + GAIT_YAW * cmd.vyaw * DT)

    brain.commands.hear(Command("come"))
    run_brain(brain, clock, CB.LOOK_FOR_S + 2.0, frames=frames, on_tick=on_tick)
    assert _last(brain)["state"] == "done", _last(brain)
    assert _last(brain)["search_ms"] > 8000.0, _last(brain)  # (longer than the old 8 s bound)


def test_a_duck_with_the_quick_turn_finds_the_person_behind_with_a_wall_close_ahead() -> None:
    """Live :8796 (2026-10-23): with a thing 0.3 m ahead the search arc had no room and the body never
    turned, so a person behind was never found. With robotd's quick turn it pivots on the spot in steps
    (owner 2026-10-24: B), looks between them, and finds them."""
    brain, clock = _duck(posture="stand")
    brain.can_pivot = True
    body = {"yaw": 0.0}
    pivots: list[float] = []

    def frames(t: float) -> Any:
        rel = wrap_angle(math.pi - body["yaw"])
        sight = (person_sight(1, bearing=rel, range_m=0.6),) if abs(rel) <= 0.5 else ()
        ahead = wrap_angle(0.0 - body["yaw"])  # (the wall stands where it began facing)
        wall = (thing_sight(2, bearing=ahead, range_m=0.3),) if abs(ahead) <= 0.5 else ()
        return _frame(t, posture="stand", sightings=sight + wall, odom=(0.0, 0.0, body["yaw"]))

    def on_tick(t: float, cmd: Any) -> None:
        pivots.append(cmd.pivot)
        rate = 2.0 * cmd.pivot if cmd.pivot and cmd.vx > 0.0 else GAIT_YAW * cmd.vyaw
        body["yaw"] = wrap_angle(body["yaw"] + rate * DT)

    brain.commands.hear(Command("come"))
    run_brain(brain, clock, CB.LOOK_FOR_S, frames=frames, on_tick=on_tick)
    assert any(pivots) and not all(pivots), "it pivots in steps and looks between them"
    last = _last(brain)
    assert last["state"] in ("doing", "done"), last
    assert last["search_ms"] < 12_000, last


def test_a_search_that_has_looked_all_round_ends_before_the_time_bound() -> None:
    brain, clock = _duck(posture="stand")
    body = {"yaw": 0.0}

    def on_tick(t: float, cmd: Any) -> None:
        body["yaw"] = wrap_angle(body["yaw"] + cmd.vyaw * DT)

    brain.commands.hear(Command("look_at_me"))
    run_brain(
        brain,
        clock,
        CB.LOOK_FOR_S + 1.0,
        frames=lambda t: _frame(t, posture="stand", odom=(0.0, 0.0, body["yaw"])),
        on_tick=on_tick,
    )
    last = _last(brain)
    assert last["state"] == "refused" and last["reason"] == "no_one_seen", last
    assert last["search_ms"] < CB.LOOK_FOR_S * 1000.0 - 2000.0, last


def test_a_person_who_comes_into_view_mid_search_is_obeyed() -> None:
    brain, clock = _duck(posture="stand")
    t0 = clock.t
    brain.commands.hear(Command("come"))
    run_brain(
        brain,
        clock,
        CB.LOOK_FOR_S,
        frames=lambda t: _frame(t, posture="stand", sightings=_person(0.6) if t > t0 + 2.0 else ()),
    )
    last = _last(brain)
    assert last["state"] == "done", last
    assert last["search_ms"] < CB.LOOK_FOR_S * 1000.0, last


def test_a_fright_mid_search_refuses_afraid() -> None:
    brain, clock = _duck(posture="stand")
    t0 = clock.t
    brain.commands.hear(Command("come"))

    def on_tick(t: float, cmd: Any) -> None:
        if t > t0 + 2.0:
            brain.drives.fear = 0.9

    run_brain(
        brain, clock, CB.LOOK_FOR_S, frames=lambda t: _frame(t, posture="stand"), on_tick=on_tick
    )
    last = _last(brain)
    assert last["state"] in ("interrupted", "refused") and last["reason"] == "afraid", last


# -- come / follow losing the person in the middle of the walk --------------------------------------


def test_come_losing_the_person_looks_about_then_continues() -> None:
    brain, clock = _duck(posture="stand", sightings=_person(2.0))
    t0 = clock.t
    states: list[str] = []
    brain.commands.hear(Command("come"))

    def frames(t: float) -> Any:
        seen = t < t0 + 2.0 or t > t0 + CB.PERSON_LOST_S + 3.0
        return _frame(t, posture="stand", sightings=_person(2.0) if seen else ())

    run_brain(
        brain,
        clock,
        CB.PERSON_LOST_S + CB.LOOK_FOR_LOST_S + 2.0,
        frames=frames,
        on_tick=_watch(brain, states),
    )
    last = _last(brain)
    assert "looking" in states, states
    assert last["state"] in ("doing", "done"), last
    assert last["search_ms"] > 0.0, last


def test_come_losing_the_person_looks_about_then_gives_up() -> None:
    brain, clock = _duck(posture="stand", sightings=_person(2.0))
    t0 = clock.t
    states: list[str] = []
    brain.commands.hear(Command("come"))

    def frames(t: float) -> Any:
        return _frame(t, posture="stand", sightings=_person(2.0) if t < t0 + 2.0 else ())

    run_brain(
        brain,
        clock,
        CB.PERSON_LOST_S + CB.LOOK_FOR_LOST_S + 2.0,
        frames=frames,
        on_tick=_watch(brain, states),
    )
    last = _last(brain)
    assert "looking" in states, states
    assert last["state"] == "gave_up" and last["reason"] == "no_one_seen", last
    assert last["search_ms"] >= CB.LOOK_FOR_LOST_S * 1000.0 - 200.0, last


# -- an errand holds the walk -----------------------------------------------------------------------


def test_an_errand_holds_the_walk_and_arrives() -> None:
    """A pointed errand survives the duck's own competing wants: the walk is held on the selector's
    persistent command channel, tick after tick, until it arrives."""
    brain, clock = _duck(posture="stand")
    body = {"x": 0.0, "y": 0.0}
    eid, _tx, _ty = brain.go_to(0.0, 0.3)  # a place 0.3 m ahead, in its own odometry frame
    index = brain.sel.names.index(E.WALK)
    held: list[float] = []

    def frames(t: float) -> Any:
        # a toy in view: the duck's own want to go and handle it is the rival the errand must survive
        return _frame(
            t,
            posture="stand",
            odom=(body["x"], body["y"], 0.0),
            sightings=(thing_sight(7, bearing=0.0, range_m=0.55),),
        )

    def on_tick(t: float, cmd: Any) -> None:
        if brain.behaviour == E.WALK:
            body["x"] += max(0.0, cmd.vx) * DT
        held.append(brain.sel._cmd_claim[index])

    run_brain(brain, clock, 25.0, frames=frames, on_tick=on_tick)
    view = brain.errand.view()
    assert view["state"] == "arrived", view
    assert view["last"]["id"] == eid and view["last"]["final_range_m"] <= E.ARRIVE_M, view
    assert max(held) >= E.ERRAND_CLAIM, held  # (the walk was held on the command channel)
