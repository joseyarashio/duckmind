"""Something new that comes into view breaks into a bout that is not done (interplay fix 2026-10-04).

The finishing rule (M5) makes every bout that is not done hold on against a rival that only crept up;
the one thing that already broke into a bout, a newly tracked thing, did so only for a wander, so a
duck that was preening, foraging or looking around let a new thing appear and went on for the rest of
the bout (v7 L2, ``novel_object``, attended 0.67 -> 0.20, changed behaviour 0.44 -> 0.13). A new
thing is a surge for the curiosity acts: it is let in to any ordinary bout — not sleep, not a
reflex's act, not what is urgent business, not going home or a docking, not the handling of a thing
already in hand."""

from __future__ import annotations

import pytest
from test_neuro_brain import _genome

from duckmind.neuro import brain as B
from duckmind.neuro import contract as C
from duckmind.neuro import select as S
from duckmind.neuro.brain import NeuroBrain


def _new_thing_frame(k: int, *, posture: str = "stand") -> C.SensorFrame:
    seen = C.Sighting(kind="thing", bearing=0.2, range_m=0.8, size=0.12, track=41)
    return C.SensorFrame(
        t_ms=20 * k,
        body=C.BodySense(posture=posture, odom=(0.0, 0.0, 0.0)),
        camera=C.CameraSense(usable=True, sightings=(seen,)),
        power=C.PowerSense(battery=0.85),
    )


# the need that holds each bout on (its bout is wanted as much as the duck can want it)
HELD_BY = {
    "wander": ("restless", 0.9),
    "preen": ("groom", 0.95),
    "forage": ("peckish", 0.95),  # (a bout held on to, though a new thing does not break it off)
    "look_around": ("vigil", 0.95),
}


def _in_bout(act: str, name: str = "bold_explorer") -> NeuroBrain:
    brain = NeuroBrain(_genome(name), seed=4107)
    brain.sel.current = brain.sel.index(act)
    brain.sel.bout_age = 5.0  # (a bout in hand for a while)
    brain.behaviour = act
    if act in HELD_BY:
        need, value = HELD_BY[act]
        setattr(brain.drives, need, value)
    return brain


def _first_investigate(brain: NeuroBrain, seconds: float = 3.0) -> int | None:
    chosen = [brain.step(20 * k, _new_thing_frame(k))[0] for k in range(int(seconds * 50))]
    return next((k for k, c in enumerate(chosen) if c.behaviour == "investigate"), None)


@pytest.mark.parametrize("act", sorted(set(HELD_BY) - {"forage"}))
def test_a_new_thing_is_let_in_to_any_ordinary_bout_not_only_a_wander(act: str) -> None:
    brain = _in_bout(act)
    first = _first_investigate(brain)
    assert first is not None and first <= 5, (act, first)  # (at once: within a tenth of a second)


@pytest.mark.parametrize("act", sorted(set(HELD_BY) - {"wander", "look_around", "forage"}))
def test_the_same_new_thing_was_ignored_by_a_bout_that_holds_on_before(act: str) -> None:
    """What the test above is about: with the interrupt for a wander only (the code before this
    change), a preening or a looking-about duck goes on."""
    brain = _in_bout(act)
    brain._novelty_interrupts = lambda feat: brain.behaviour == "wander"  # type: ignore[method-assign]
    assert _first_investigate(brain) is None


@pytest.mark.parametrize(
    "act", ["nap", "doze", "return_dock", "investigate", "nibble", "peck", "forage"]
)
def test_what_a_new_thing_does_not_break_into(act: str) -> None:
    brain = _in_bout(act)
    assert not brain._novelty_interrupts(list(brain._feat))


def test_a_new_thing_does_not_break_into_urgent_business_nor_a_docking() -> None:
    brain = _in_bout("flee")
    feat = S.empty_features()
    feat[S.F["startled"]] = 0.9
    feat[S.F["fear"]] = 0.4
    assert brain.sel.urgent_now(feat) and not brain._novelty_interrupts(feat)
    brain = _in_bout("preen")
    feat = S.empty_features()
    assert brain._novelty_interrupts(feat)
    feat[S.F["docking"]] = 1.0
    assert not brain._novelty_interrupts(feat)
    feat[S.F["docking"]] = 0.0
    feat[S.F["homing"]] = 1.0
    assert not brain._novelty_interrupts(feat)


def test_a_drowsy_duck_is_not_roused_by_a_new_thing() -> None:
    brain = _in_bout("preen")
    feat = S.empty_features()
    assert brain._novelty_interrupts(feat)
    feat[S.F["sleepy"]] = B.NOVELTY_SLEEPY_MAX
    assert not brain._novelty_interrupts(feat)


@pytest.mark.parametrize("feature", ["name", "voice", "person_new", "startled", "looming"])
def test_a_person_who_has_just_called_and_a_threat_come_before_a_new_thing(feature: str) -> None:
    brain = _in_bout("turn_around")
    feat = S.empty_features()
    assert brain._novelty_interrupts(feat)
    feat[S.F[feature]] = B.NOVELTY_YIELD_AT
    assert not brain._novelty_interrupts(feat)
    feat[S.F[feature]] = B.NOVELTY_YIELD_AT * 0.5
    assert brain._novelty_interrupts(feat)


def test_a_bout_that_has_only_just_begun_is_not_broken_off_by_a_new_thing() -> None:
    brain = _in_bout("preen")
    feat = S.empty_features()
    assert brain._novelty_interrupts(feat)
    brain.sel.bout_age = B.NOVELTY_MIN_BOUT_S * 0.5
    assert not brain._novelty_interrupts(feat)
    brain.sel.bout_age = B.NOVELTY_MIN_BOUT_S
    assert brain._novelty_interrupts(feat)


def test_a_new_thing_does_not_take_the_head_from_the_instrument_or_the_dance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sound skills' acts (reflex.ENGAGED_ACTS, set while they are installed): the head has a job
    there; tests/test_neuro_skill_sound.py holds the whole scene."""
    monkeypatch.setattr("duckmind.neuro.reflex.ENGAGED_ACTS", frozenset({"preen"}))
    brain = _in_bout("preen")
    assert not brain._novelty_interrupts(S.empty_features())
    monkeypatch.setattr("duckmind.neuro.reflex.ENGAGED_ACTS", frozenset())
    assert brain._novelty_interrupts(S.empty_features())


def test_a_new_thing_far_off_is_noted_not_turned_to() -> None:
    brain = _in_bout("preen")
    feat = S.empty_features()
    brain._thing_last = (0.0, B.NOVELTY_RANGE_M * 1.5, 0.0)
    assert not brain._novelty_interrupts(feat)
    brain._thing_last = (0.0, B.NOVELTY_RANGE_M * 0.8, 0.0)
    assert brain._novelty_interrupts(feat)
