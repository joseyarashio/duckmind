"""Staying put (docs/design/neural-duck/SETTLE-2026-11-04.md): after 「停」 the duck stays where it is until it is released (a new
command, its name, a fright) or the stay's cap, looking about in place; set down by a hand it settles a moment before it walks.
Reflexes and fear stay above both."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from live_bridge_rig import frame, make_brain, person_sight, run_brain

from duckmind.neuro import brain as B
from duckmind.neuro import command_brain as CB
from duckmind.neuro import contract as C
from duckmind.neuro import select as S
from duckmind.neuro.commands import Command
from duckmind.neuro.genome import preset
from duckmind.neuro.integration import make_brain as build

pytestmark = pytest.mark.usefixtures("no_network")

SEEDS = (4107, 42, 9101, 1)  # an explorer, a songbird, a dreamer, a dreamer (tempo 0.49)
HZ = 20.0


def _fr(
    t: float,
    posture: str = "stand",
    events: tuple[str, ...] = (),
    sightings: tuple[C.Sighting, ...] = (),
    sound: str | None = None,
    level_db: float = 30.0,
    fallen: bool = False,
) -> C.SensorFrame:
    f = frame(int(round(t * 1000)), *sightings, sound_kind=sound, level_db=level_db)
    return replace(f, body=C.BodySense(posture=posture, fallen=fallen), events=events)


def _duck(seed: int, settle: float = 20.0) -> tuple[Any, Any]:
    brain, clock = make_brain(seed=seed)
    brain.commands.clock = clock
    run_brain(brain, clock, settle, frames=lambda t: _fr(t))
    return brain, clock


def _walks(cmds: list[Any]) -> list[int]:
    """The ticks where the brain asked its legs to move (a step, a turn, an arc)."""
    return [i for i, c in enumerate(cmds) if c.vx != 0.0 or c.vyaw != 0.0]


def _stop(brain: Any, clock: Any, **kw: Any) -> None:
    brain.commands.hear(Command("stop"))
    run_brain(brain, clock, 1.0, frames=lambda t: _fr(t, **kw))
    assert brain.commands.last["verb"] == "stop" and brain.commands.last["state"] == "done", brain.commands.last


def _set_down(brain: Any, clock: Any, held_s: float = 3.0) -> None:
    run_brain(brain, clock, 0.05, frames=lambda t: _fr(t, "unknown", ("lifted",)))
    run_brain(brain, clock, held_s, frames=lambda t: _fr(t, "unknown"))
    run_brain(brain, clock, 0.05, frames=lambda t: _fr(t, "stand", ("set_down",)))


# -- 「停」 ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_a_stopped_duck_stays_where_it_is_and_looks_about(seed: int) -> None:
    """Before: 「停」 held the duck for 1-6 s in this quiet room and then it wandered off by itself (every one of the four).
    Now it stays for the whole of its shortest possible stay — not a step, not a turn, no walking act — and its head looks
    about meanwhile."""
    brain, clock = _duck(seed)
    _stop(brain, clock)
    cmds = run_brain(brain, clock, 28.0, frames=lambda t: _fr(t))  # (the shortest stay is 30 s)
    assert not _walks(cmds), (seed, [(i / HZ, cmds[i].behaviour) for i in _walks(cmds)[:3]])
    assert not {c.behaviour for c in cmds} & S.LOCOMOTES
    bearings = [c.gaze.bearing for c in cmds]
    assert max(bearings) - min(bearings) > 0.3, (seed, min(bearings), max(bearings))  # (it looks about, it is no statue)


@pytest.mark.parametrize("seed", SEEDS)
def test_the_stay_has_a_cap_and_then_the_duck_goes_its_own_way(seed: int) -> None:
    """Never for ever: once the stay's cap has passed the duck's own wants have the legs again."""
    brain, clock = _duck(seed)
    _stop(brain, clock)
    cmds = run_brain(brain, clock, CB.STOP_STAY_MAX_S + 60.0, frames=lambda t: _fr(t))
    walks = _walks(cmds)
    assert walks, seed
    assert CB.STOP_STAY_MIN_S - 1.5 <= walks[0] / HZ, (seed, walks[0] / HZ)
    assert brain.commands.view()["stay"] is None and brain.commands.view()["stay_ended"] == "time"


def test_the_lively_stay_less_long_than_the_slow_but_every_one_stays() -> None:
    lively = build(preset("explorer"), 4107)  # activity 0.82, exploration 0.98
    slow = build(preset("snuggler"), 4107)  # activity 0.17, exploration 0.05
    a, b = lively.commands._stay_s(lively), slow.commands._stay_s(slow)
    assert CB.STOP_STAY_MIN_S <= a < b <= CB.STOP_STAY_MAX_S


def test_a_new_command_releases_the_stay() -> None:
    brain, clock = _duck(4107, settle=6.0)
    person = (person_sight(1, range_m=1.8),)
    _stop(brain, clock, sightings=person)
    run_brain(brain, clock, 4.0, frames=lambda t: _fr(t, sightings=person))
    brain.commands.hear(Command("come"))
    cmds = run_brain(brain, clock, 4.0, frames=lambda t: _fr(t, sightings=person))
    assert _walks(cmds) and "follow" in {c.behaviour for c in cmds}
    assert brain.commands.view()["stay_ended"] == "command"


def test_its_name_releases_the_stay() -> None:
    brain, clock = _duck(4107)
    _stop(brain, clock)
    run_brain(brain, clock, 3.0, frames=lambda t: _fr(t))
    run_brain(brain, clock, 0.3, frames=lambda t: _fr(t, sound="name", level_db=62.0))
    view = brain.commands.view()
    assert view["stay"] is None and view["stay_ended"] == "called"


def test_a_fright_releases_the_stay_and_the_duck_gets_away() -> None:
    """A bang: the startle and the flight it goes on into are never held by a stay."""
    brain, clock = _duck(42)
    _stop(brain, clock)
    run_brain(brain, clock, 3.0, frames=lambda t: _fr(t))
    run_brain(brain, clock, 0.2, frames=lambda t: _fr(t, sound="bang", level_db=95.0))
    cmds = run_brain(brain, clock, 6.0, frames=lambda t: _fr(t))
    assert brain.commands.view()["stay_ended"] == "afraid"
    assert _walks(cmds) and {c.behaviour for c in cmds} & {"flee", "retreat", "startle"}


def test_a_fall_or_being_picked_up_ends_the_stay() -> None:
    brain, clock = _duck(4107)
    _stop(brain, clock)
    run_brain(brain, clock, 0.5, frames=lambda t: _fr(t, "unknown", ("fell",), fallen=True))
    assert brain.commands.view()["stay_ended"] == "fallen"
    brain, clock = _duck(4107)
    _stop(brain, clock)
    run_brain(brain, clock, 0.1, frames=lambda t: _fr(t, "unknown", ("lifted",)))
    assert brain.commands.view()["stay_ended"] == "held"


def test_stop_said_as_the_hand_lets_go_is_taken_and_it_stays() -> None:
    """The flip rig's case: the owner says 「停」 as the duck is set down. Before: refused ``body_busy`` (what is left of the
    lifted hold), and it walked off after the settle."""
    brain, clock = _duck(4107)
    _set_down(brain, clock)
    ticket = brain.commands.hear(Command("stop"))
    run_brain(brain, clock, 2.0, frames=lambda t: _fr(t))
    assert ticket.answer["state"] != "refused", ticket.answer
    cmds = run_brain(brain, clock, 26.0, frames=lambda t: _fr(t))
    assert not _walks(cmds)


def test_a_staying_duck_does_not_shake_itself_off_standing() -> None:
    """A shake standing is a shuffle of the feet (the motor's own steps for it, repertoire.SHAPES_WALK). Before: with the
    after-effects of being righted and set down in it, the dreamer shook itself off in the middle of a 「停」 (headless rig:
    set down after a ride, 2.4 s into its settle it shuffled off). Now standing in a stay it does not; a stay's end frees it."""
    brain, clock = _duck(1)
    _stop(brain, clock)
    behaviours = []
    for _ in range(200):
        brain.appraiser._traces["righted"] = 1.0
        brain.appraiser._traces["set_down"] = 1.0
        behaviours += [c.behaviour for c in run_brain(brain, clock, 0.05, frames=lambda t: _fr(t))]
    assert brain.commands.view()["stay"] is not None, brain.commands.view()["stay_ended"]
    assert "body_shake" not in behaviours, behaviours.count("body_shake")


# -- set down by a hand ------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", (4107, 1))
def test_a_duck_set_down_settles_and_looks_about_before_it_walks(seed: int) -> None:
    """Before: set down, the explorer turned round 1.15 s later and the dreamer 0.6 s later. Now it settles first — at least
    SETTLE_MIN_S, looking around — and then goes its own way (the explorer within a few seconds more)."""
    brain, clock = _duck(seed)
    _set_down(brain, clock)
    cmds = run_brain(brain, clock, 30.0, frames=lambda t: _fr(t))
    walks = _walks(cmds)
    assert not walks or walks[0] / HZ >= 1.9, (seed, walks[0] / HZ, cmds[walks[0]].behaviour)  # (it settles 2 s at least)
    assert "look_around" in {c.behaviour for c in cmds[: int(1.9 * HZ)]}
    assert B.SETTLE_MIN_S == 2.0
    assert brain.commands.view()["stay_ended"] == "time"
    if seed == 4107:
        assert walks and walks[0] / HZ <= B.SETTLE_MAX_S + 5.0


def test_the_settle_is_short_for_the_bold_and_lively_and_longer_for_the_shy() -> None:
    bold = build(preset("explorer"), 4107)  # boldness 0.9, activity 0.82
    shy = build(preset("snuggler"), 4107)  # boldness 0.01, activity 0.17
    a, b = bold._settle_s(), shy._settle_s()
    assert B.SETTLE_MIN_S <= a < b <= B.SETTLE_MAX_S


def test_picked_up_in_a_stay_it_settles_after_the_set_down_instead() -> None:
    """Picked up during a 「停」: being held ends the stay; set down again it settles (the short one), it is not kept 30 s."""
    brain, clock = _duck(4107)
    _stop(brain, clock)
    _set_down(brain, clock)
    assert brain.commands.view()["stay"]["kind"] == "set_down"


def test_a_duck_still_afraid_from_the_carry_settles_too_and_a_new_fright_ends_it() -> None:
    """Carried by a hand the duck is often afraid when it is put down (headless: fear 0.4-0.8 after the grab and the lift):
    that fear is no new fright — it settles all the same. A bang then ends the settle."""
    brain, clock = _duck(4107)
    brain.drives.fear = 0.7  # (the grab and the lift)
    _set_down(brain, clock)
    assert brain.drives.fear >= S.CLAIM_FEAR_MAX
    run_brain(brain, clock, 1.0, frames=lambda t: _fr(t))
    assert brain.commands.view()["stay"] is not None, brain.commands.view()["stay_ended"]
    run_brain(brain, clock, 0.2, frames=lambda t: _fr(t, sound="bang", level_db=95.0))
    assert brain.commands.view()["stay_ended"] == "afraid"


@pytest.mark.parametrize("verb", ["sit", "come", "follow", "look_at_me"])
def test_other_commands_it_refuses_for_fear_leave_its_settle_alone(verb: str) -> None:
    """Afraid as it is put down, the duck refuses every word but 「停」 (the spec: a frightened duck does not obey) — and that
    refused word does not end the settle it is in (headless: 14 of 34 settles were ended so)."""
    brain, clock = _duck(4107)
    brain.drives.fear = 0.8
    _set_down(brain, clock)
    ticket = brain.commands.hear(Command(verb))
    run_brain(brain, clock, 0.3, frames=lambda t: _fr(t, sightings=(person_sight(range_m=1.2),)))
    assert ticket.answer["state"] == "refused" and ticket.answer["reason"] == "afraid", ticket.answer
    assert brain.commands.view()["stay"] is not None, brain.commands.view()["stay_ended"]


# -- afraid, 「停」 still counts (owner 2026-11-05) -------------------------------------------------------


@pytest.mark.parametrize("seed", (4107, 77))
def test_stop_said_to_an_afraid_duck_is_taken_and_it_stays(seed: int) -> None:
    """Before: 「停」 to a frightened duck was refused (``afraid``) — set down still afraid, it settled 2-5 s and could walk
    off. Now it is taken: it stays where it is, as a calm duck told 「停」 does."""
    brain, clock = _duck(seed)
    _set_down(brain, clock)
    brain.drives.fear = 0.7  # (what the grab and the lift left)
    ticket = brain.commands.hear(Command("stop"))
    run_brain(brain, clock, 1.0, frames=lambda t: _fr(t))
    assert ticket.answer["state"] != "refused", ticket.answer
    assert brain.commands.last["state"] == "done", brain.commands.last
    assert brain.commands.view()["stay"]["kind"] == "told"
    cmds = run_brain(brain, clock, 25.0, frames=lambda t: _fr(t))
    assert not _walks(cmds), (seed, [(i / HZ, cmds[i].behaviour) for i in _walks(cmds)[:3]])
    assert brain.commands.view()["stay"] is not None, brain.commands.view()["stay_ended"]


def test_an_afraid_duck_told_to_stop_on_the_move_stops_at_once() -> None:
    """Moving when it is told, a frightened duck's legs hold from the word on (the freeze a calm duck is asked for is the
    claim its fear mutes): no step after it, and the stop is done once the body is still."""
    brain, clock = _duck(4107)
    brain.drives.fear = 0.7
    ticket = brain.commands.hear(Command("stop"))
    cmds = run_brain(brain, clock, 0.5, frames=lambda t: _fr(t, "walk"))
    cmds += run_brain(brain, clock, 10.0, frames=lambda t: _fr(t))
    assert ticket.answer["state"] != "refused", ticket.answer
    assert not _walks(cmds[1:]), [(i / HZ, cmds[i].behaviour) for i in _walks(cmds)[:3]]
    assert brain.commands.last["state"] == "done", brain.commands.last
    assert brain.commands.view()["stay"]["kind"] == "told"


def test_a_new_startle_ends_the_stay_of_a_duck_told_to_stop_afraid() -> None:
    """A real fright still wins: a bang during the stay of a duck that was afraid when told ends it, and it gets away."""
    brain, clock = _duck(4107)
    brain.drives.fear = 0.7
    brain.commands.hear(Command("stop"))
    run_brain(brain, clock, 2.0, frames=lambda t: _fr(t))
    assert brain.commands.view()["stay"] is not None, brain.commands.view()["stay_ended"]
    cmds = run_brain(brain, clock, 0.2, frames=lambda t: _fr(t, sound="bang", level_db=95.0))
    cmds += run_brain(brain, clock, 3.0, frames=lambda t: _fr(t))
    assert brain.commands.view()["stay_ended"] == "afraid"
    assert {c.behaviour for c in cmds} & {"flee", "retreat", "startle"}


def test_more_fear_ends_the_stay_of_a_duck_told_to_stop_afraid() -> None:
    """A duck afraid when its stay began: fear STAY_FEAR_RISE above the least it has had since is a new fright."""
    brain, clock = _duck(4107)
    brain.drives.fear = 0.6
    brain.commands.hear(Command("stop"))
    run_brain(brain, clock, 2.0, frames=lambda t: _fr(t))
    assert brain.commands.view()["stay"] is not None, brain.commands.view()["stay_ended"]
    brain.drives.fear = min(1.0, brain.drives.fear + B.STAY_FEAR_RISE + 0.02)
    run_brain(brain, clock, 0.1, frames=lambda t: _fr(t))
    assert brain.commands.view()["stay"] is None and brain.commands.view()["stay_ended"] == "afraid"


def test_stop_is_still_refused_while_it_gets_away() -> None:
    """Startled and running from a bang, 「停」 is refused: the fright has the legs."""
    brain, clock = _duck(4107)
    run_brain(brain, clock, 0.2, frames=lambda t: _fr(t, sound="bang", level_db=95.0))
    ticket = brain.commands.hear(Command("stop"))
    run_brain(brain, clock, 0.1, frames=lambda t: _fr(t))
    assert ticket.answer["state"] == "refused" and ticket.answer["reason"] in ("afraid", "fleeing"), ticket.answer


def test_something_looming_still_sends_a_duck_told_to_stop_afraid_away() -> None:
    """Its fear acts answer a fright of now: a fast-looming thing during the stay ends it and it gets away."""
    brain, clock = _duck(4107)
    brain.drives.fear = 0.6
    brain.commands.hear(Command("stop"))
    run_brain(brain, clock, 2.0, frames=lambda t: _fr(t))
    loom = C.Sighting("looming", 0.0, 0.1, None, 0.2, 0.8, 2.5, 0, "model")
    cmds = run_brain(brain, clock, 1.5, frames=lambda t: _fr(t, sightings=(loom,)))
    cmds += run_brain(brain, clock, 2.0, frames=lambda t: _fr(t))
    assert brain.commands.view()["stay_ended"] == "afraid", brain.commands.view()
    assert {c.behaviour for c in cmds} & {"flee", "retreat", "startle", "hide"}


# -- carried by someone it knows (owner 2026-11-05) ---------------------------------------------------

LOOKS = {"family": (0.62, 0.30, 0.35, 0.80, 0.55, 0.25), "stranger": (0.20, 0.70, 0.75, 0.55, 0.15, 0.70)}


def _fear_after_carry(seed: int, who: str | None) -> tuple[float, Any, Any]:
    """A duck that has had ``who`` beside it a while (a person it knows well, a stranger, or nobody) is grabbed (the grab's
    flinch leaves fear 0.45), lifted, held 6 s and set down: its fear 3 s after."""
    brain, clock = make_brain(seed=seed)
    brain.commands.clock = clock
    sights = () if who is None else (person_sight(1, range_m=0.7, look=LOOKS[who]),)
    run_brain(brain, clock, 20.0, frames=lambda t: _fr(t, sightings=sights))
    people = brain.imprint.people.people
    if who is not None:
        assert len(people) == 1, people
    if who == "family":
        next(iter(people.values())).bond = 0.9  # (its own record: it has known them since it hatched)
    brain.drives.fear = 0.45
    _set_down(brain, clock, held_s=6.0)
    run_brain(brain, clock, 3.0, frames=lambda t: _fr(t, sightings=sights))
    return float(brain.drives.fear), brain, clock


@pytest.mark.parametrize("seed", (77, 7, 4107))
def test_a_carry_by_someone_it_knows_leaves_less_fear_than_a_strangers_or_an_unseen_hand(seed: int) -> None:
    """Before: the same lift left the same fear whoever held it (the lift's own fright, then the slow fade). Now in the hands
    of the person it knows best the lift frightens it less and its fear settles faster; a stranger's or an unseen hand's
    carry is what it always was."""
    family = _fear_after_carry(seed, "family")[0]
    stranger = _fear_after_carry(seed, "stranger")[0]
    unseen = _fear_after_carry(seed, None)[0]
    assert family < stranger - 0.05 and family < unseen - 0.05, (seed, family, stranger, unseen)


def test_a_fall_right_after_a_carry_by_someone_it_knows_still_frightens() -> None:
    """Safety stays on top: in the hands of the person it knows best, a fall right after the set-down is as frightening
    as ever (pain is pain)."""
    _, brain, clock = _fear_after_carry(77, "family")
    before = float(brain.drives.fear)
    run_brain(brain, clock, 0.05, frames=lambda t: _fr(t, "unknown", ("fell",), fallen=True))
    assert float(brain.drives.fear) > before + 0.3, (before, float(brain.drives.fear))
