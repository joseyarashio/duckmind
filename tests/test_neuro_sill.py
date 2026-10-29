"""sill-circling (2026-10-28): the duck at the 20 mm hall–kitchen sill tripped, fell and went round in circles.

1. With no nest in the home (live ``--no-nest``, learned dock null) going home was an endless search of tight arcs: 25 of
   65 live loop captures, 53 % of its walking home. A search ends now after one whole round of looking
   (``SEEK_GIVE_UP_S``); going home is let go for ``NEST_REST_S`` (doubling after each search that found nothing) and its
   nest seen ends that rest at once.
"""

from __future__ import annotations

import math

from test_neuro_dock import _frame

from duckmind.neuro import brain as B
from duckmind.neuro import dock as D
from duckmind.neuro.brain import NeuroBrain
from duckmind.neuro.genome import preset

TICK_MS = 20


def _run(brain: NeuroBrain, k0: int, seconds: float, **frame: object) -> list[str]:
    out = []
    for k in range(k0, k0 + int(seconds * 1000 / TICK_MS)):
        cmd, _ = brain.step(TICK_MS * k, _frame(TICK_MS * k, **frame))  # type: ignore[arg-type]
        out.append(cmd.behaviour)
    return out


def _homing_brain() -> NeuroBrain:
    brain = NeuroBrain(preset("snuggler"), seed=3)
    brain.drives.hunger = 0.8
    assert brain.learn.dock is None  # no nest remembered (live: learned.dock null)
    return brain


def test_a_search_for_a_nest_it_never_sees_ends_and_it_does_something_else() -> None:
    brain = _homing_brain()
    behs = _run(brain, 0, 30.0, battery=0.36)  # the low pack pulls it home: it looks for its nest
    assert "return_dock" in behs
    first = behs.index("return_dock") * TICK_MS / 1000.0
    search_s = 4 * 9.0  # one whole round of looking: four 9 s periods of sweeps and arcs
    later = _run(brain, len(behs), first + search_s + 60.0 - 30.0, battery=0.36)
    after = later[int((first + search_s + 3.0 - 30.0) * 1000 / TICK_MS) :]
    assert after and "return_dock" not in after, sorted(set(after))  # a minute on, it has not gone back to searching


def test_its_nest_seen_ends_the_rest_at_once() -> None:
    brain = _homing_brain()
    behs = _run(brain, 0, 30.0 + B.SEEK_GIVE_UP_S + 5.0, battery=0.36)
    assert behs[-1] != "return_dock" and brain._nest_rest_until > 0.0
    pose = D.DockPose(1500.0, 100.0, math.pi - 0.1)  # its tag in view, 1.5 m ahead
    seen = _run(brain, len(behs), 10.0, battery=0.36, pose=pose)
    assert "return_dock" in seen and brain._nest_rests == 0


def test_each_search_that_finds_nothing_rests_longer() -> None:
    brain = _homing_brain()
    k = 0
    rests = []
    for _ in range(2):
        until = brain._nest_rest_until
        while brain._nest_rest_until == until:
            brain.step(TICK_MS * k, _frame(TICK_MS * k, battery=0.36))
            k += 1
            assert k < 600_000 // TICK_MS, "the search never ended"
        rests.append(brain._nest_rest_until - TICK_MS * k / 1000.0)
    assert abs(rests[0] - B.NEST_REST_S[0]) < 0.1
    assert abs(rests[1] - 2.0 * B.NEST_REST_S[0]) < 0.1
