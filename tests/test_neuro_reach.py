"""到了 means the beak can reach it (machine verification v3, finding ③, 2026-10-02).

The v3 physics run found the neural duck saying 到了 with the thing 0.2–0.45 m from its trunk: the
walk stopped at 0.22 m + half the thing's width of the eye's range, and that range starts at the
lens, 0.08 m ahead of the trunk's centre, to the thing's foot (its near edge); the measured beak
reaches 0.096 m from the trunk's centre. These tests pin the distance the brain now means: the
thing's near edge from the trunk's centre, from the duck's own senses only (what the eye sees, or
where it believes the thing is), no wider than the beak's reach and the walk's coast.
"""

from __future__ import annotations

import math

import pytest

from duckmind.neuro import brain as B
from duckmind.neuro import contract as C
from duckmind.neuro.brain import NeuroBrain
from duckmind.neuro.genome import Genome
from duckmind.neuro.perceive import lens_point
from duckmind.verify import spec as V

TICK_MS = 20
LENS_AHEAD_M = 0.081  # the head camera's lens ahead of the trunk's centre at HOME (standing)
EXPLORER = dict(
    boldness=0.9, activity=0.9, sociability=0.3, exploration=0.9, wit=0.7, chronotype=0.3, tempo=0.6
)


def _brain() -> NeuroBrain:
    brain = NeuroBrain(
        Genome.from_axes(EXPLORER, primary="explorer", branch="explorer", gd=120), 4107
    )
    brain.behaviour = "investigate"
    brain.sel.current = brain.sel.index("investigate")
    brain._attend_track = 2
    return brain


def _frame(
    k: int, sighting: C.Sighting | None, odom: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> C.SensorFrame:
    return C.SensorFrame(
        t_ms=TICK_MS * k,
        body=C.BodySense(posture="walk", odom=odom),
        camera=C.CameraSense(usable=True, sightings=() if sighting is None else (sighting,)),
        power=C.PowerSense(battery=0.85),
    )


def _thing(range_m: float, size: float = 0.12, bearing: float = 0.0) -> C.Sighting:
    return C.Sighting(kind="thing", bearing=bearing, range_m=range_m, size=size, track=2)


def _arrival_tick(seen_at: list[float | None], size: float = 0.12) -> int | None:
    """The first tick the brain says 到了 when the eye reads these lens ranges tick by tick (None:
    nothing there that tick), the duck standing still in the middle of its investigation."""
    brain = _brain()
    for k, rng in enumerate(seen_at):
        brain.step(TICK_MS * k, _frame(k, None if rng is None else _thing(rng, size)))
        if brain._bout.get("arrived"):
            return k
    return None


def test_the_arrival_distance_is_the_beaks_reach_plus_the_walks_coast() -> None:
    assert B.BEAK_REACH_M == V.BEAK_REACH_M  # the measured beak reach, as the verifier has it
    assert B.ARRIVE_M == pytest.approx(B.BEAK_REACH_M + B.STOP_COAST_M)
    assert B.REACH_M["investigate"] == B.ARRIVE_M
    # inside what the verifier calls "the beak can touch it" (reach + 0.1 m) with the odometry's
    # largest error seen in the v3 runs (0.042 m) to spare
    assert B.ARRIVE_M + 0.042 <= V.BEAK_REACH_M + V.M4_ARRIVE_MARGIN_M


def test_the_lens_is_ahead_of_the_trunks_centre() -> None:
    ahead, left = lens_point(_frame(0, None))
    assert ahead == pytest.approx(LENS_AHEAD_M, abs=0.002)
    assert left == pytest.approx(0.0, abs=1e-9)


def test_a_thing_near_the_lens_but_far_from_the_trunk_is_not_arrived_at() -> None:
    # the eye's range to its foot is 0.15 m from the lens: 0.23 m from the trunk's centre
    assert _arrival_tick([0.15] * 150) is None


def test_the_distance_is_to_the_trunk_not_the_lens() -> None:
    # 0.04 m from the lens is 0.121 m from the trunk (inside 0.126), 0.05 m is 0.131 (outside)
    assert _arrival_tick([0.04] * 100) is not None
    assert _arrival_tick([0.05] * 100) is None


def test_a_wide_thing_is_not_stopped_short_of() -> None:
    # the range is to its near edge already: half its width is nothing to add (the duck used to
    # stop 0.15 m further off a thing as wide as books)
    wide = 0.6  # angular size, rad: 0.18 m across at 0.3 m
    assert _arrival_tick([0.15] * 150, size=wide) is None
    assert _arrival_tick([0.04] * 100, size=wide) == _arrival_tick([0.04] * 100, size=0.1)


def test_it_says_it_after_it_has_been_there_a_moment_not_on_the_first_look() -> None:
    k = _arrival_tick([0.03] * 100)
    assert k is not None and k * TICK_MS / 1000.0 >= B.ARRIVE_HOLD_S - 0.05


def test_one_close_reading_in_a_row_of_far_ones_does_not_say_it() -> None:
    far = [0.4] * 100
    far[50] = 0.03  # a single noisy look
    assert _arrival_tick(far) is None


def test_a_thing_it_no_longer_sees_is_arrived_at_by_where_it_believes_it_is() -> None:
    """Seen 0.5 m from the lens (0.581 m ahead of the trunk), then out of the picture: the walk
    on from the odometry says 到了 only when the believed place is inside the beak's reach."""

    def arrived(trunk_to_thing_m: float) -> bool:
        brain = _brain()
        for k in range(25):  # half a second of looking at it
            brain.step(TICK_MS * k, _frame(k, _thing(0.5)))
        believed_x = LENS_AHEAD_M + 0.5
        odom = (believed_x - trunk_to_thing_m, 0.0, 0.0)
        t0 = 25
        for k in range(t0, t0 + 400):  # eight seconds without the thing in view
            brain.step(TICK_MS * k, _frame(k, None, odom))
            if brain._bout.get("arrived"):
                return True
        return False

    assert not arrived(0.20)  # 0.20 m from the trunk: it used to say 到了 (< 0.22)
    assert arrived(0.10)


def test_what_it_saw_is_remembered_where_the_thing_lies_not_where_the_lens_was() -> None:
    brain = _brain()
    for k in range(25):
        brain.step(TICK_MS * k, _frame(k, _thing(0.5)))
    thing = brain.learn.things[brain._thing_key]
    assert math.hypot(thing.x, thing.y) == pytest.approx(LENS_AHEAD_M + 0.5, abs=0.003)
