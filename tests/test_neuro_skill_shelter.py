"""a:social — 害怕時躲起來: shelters from its own senses, the flight, the hiding and the peeks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from duckmind.neuro import skill_shelter as H
from duckmind.neuro import skill_social_brain as SB
from duckmind.neuro.genome import preset
from duckmind.world import neuro_social_synth as W


def _walls_line(x0: float, y0: float, x1: float, y1: float, step: float = 0.05) -> list:
    n = max(1, int(math.hypot(x1 - x0, y1 - y0) / step))
    return [(x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n) for k in range(n + 1)]


def _room(sm: H.ShelterMap) -> None:
    """A 3 × 3 m room's walls (odometry frame), all of it floor it has seen."""
    pts = _walls_line(0, 0, 3, 0) + _walls_line(3, 0, 3, 3) + _walls_line(3, 3, 0, 3)
    pts += _walls_line(0, 3, 0, 0)
    H.walls_from(pts, sm, 0.0)
    for i in range(10):
        for j in range(10):
            sm.floor.add((i, j))


def test_a_corner_is_a_shelter_a_straight_wall_is_not() -> None:
    sm = H.ShelterMap()
    _room(sm)
    corner, _, encl_c, _ = sm.value(0.27, 0.27)
    wall, _, encl_w, _ = sm.value(1.5, 0.27)
    open_, _, _, _ = sm.value(1.5, 1.5)
    assert encl_c >= 5 / 8 and corner >= H.SHELTER_MIN
    assert encl_w <= 3 / 8 + 1e-9 and wall < H.SHELTER_MIN
    assert open_ == 0.0
    spots = sm.refuges(10.0)
    assert spots and all(min(r.x, r.y, 3 - r.x, 3 - r.y) < 0.6 for r in spots)  # the corners


def test_a_roof_it_sat_under_is_a_shelter() -> None:
    sm = H.ShelterMap()
    for k in range(6):  # under a table: its roof reading says 0.2 m above the lens, looked twice+
        sm.observe(1000 * k, (2.0, 2.0, 0.0), roof=0.2)
    value, cover, encl, _ = sm.value(2.0, 2.0)
    assert cover > 0.7 and encl == 0.0 and value >= H.SHELTER_MIN
    sm.observe(7000, (2.0, 2.0, 0.0))
    assert sm.refuges(10.0)


def test_it_runs_away_from_the_scare_not_toward_it() -> None:
    sm = H.ShelterMap()
    _room(sm)
    odom = (1.5, 1.5, 0.0)
    choice = sm.choose(odom, 10.0, threat_heading=math.atan2(-1.0, -1.0))  # scare from SW
    assert choice is not None
    r = choice[0]
    assert (r.x, r.y) != pytest.approx((0.27, 0.27), abs=0.3)  # not the corner the scare is in


def test_a_refuge_it_could_not_reach_is_avoided_for_a_while() -> None:
    sm = H.ShelterMap()
    _room(sm)
    odom = (0.7, 0.7, 0.0)
    first = sm.choose(odom, 10.0, None)
    assert first is not None
    sm.mark_bad(first[0].x, first[0].y, 10.0)
    second = sm.choose(odom, 11.0, None)
    assert second is not None
    assert (second[0].x, second[0].y) != (first[0].x, first[0].y)


def test_hide_threshold_timid_hide_sooner() -> None:
    assert H.hide_threshold(0.0) < H.hide_threshold(0.5) < H.hide_threshold(1.0)
    fear = 0.6
    assert H.hide_urge(fear, 0.05) > 0.9
    assert H.hide_urge(fear, 0.95) == 0.0


def test_the_run_goes_in_peeks_more_often_as_it_calms_and_comes_out() -> None:
    sm = H.ShelterMap()
    _room(sm)
    run = H.HideRun(boldness=0.1, seed=3)
    odom = (1.2, 1.2, 0.0)
    assert run.start(0.0, odom, sm, None, fear=0.95)
    assert run.phase == "go"
    target = run.refuge
    assert target is not None
    run.step(
        5.0,
        (target.x, target.y, 0.0),
        fear=0.9,
        blocked=False,
        roof=None,
        enclosure_here=0.6,
        shelters=sm,
    )
    assert run.phase == "in"
    peeks: list[float] = []
    fear = 0.9
    t = 5.0
    while t < 400.0 and run.phase == "in":
        if run.peek_due(t, fear):
            peeks.append(t)
        run.step(
            t,
            (target.x, target.y, 0.0),
            fear=fear,
            blocked=False,
            roof=None,
            enclosure_here=0.6,
            shelters=sm,
        )
        fear *= math.exp(-0.1 / 110.0)
        t += 0.1
    assert run.phase == "out"
    assert len(peeks) >= 4
    gaps = [b - a for a, b in zip(peeks, peeks[1:], strict=False)]
    assert sum(gaps[-3:]) / 3 < sum(gaps[:3]) / 3  # it peeks more often as it calms


def test_being_held_by_a_reflex_is_not_a_failed_flight() -> None:
    sm = H.ShelterMap()
    _room(sm)
    run = H.HideRun(boldness=0.1)
    assert run.start(0.0, (1.5, 1.5, 0.0), sm, None, fear=0.95)
    t = 0.0
    while t < 20.0:  # frozen by a startle: no progress, but the legs were not its own
        run.step(
            t,
            (1.5, 1.5, 0.0),
            fear=0.9,
            blocked=False,
            roof=None,
            enclosure_here=0.0,
            shelters=sm,
            free=False,
        )
        t += 0.1
    assert run.phase == "go"


@dataclass
class _Sector:
    bearing: float
    free_m: float
    end: str
    width: float = 0.13


def test_the_floor_line_lays_walls_and_known_floor() -> None:
    sm = H.ShelterMap()
    # facing a wall 0.4 m ahead, twice (a wall counts once seen twice)
    floor = tuple(_Sector(b, 0.4, "wall") for b in (0.26, 0.13, 0.0, -0.13, -0.26))
    sm.observe(1000, (1.0, 1.0, 0.0), floor, 1000)
    sm.observe(1500, (1.0, 1.0, 0.0), floor, 1500)
    assert sm.view()["wall_cells"] >= 3
    assert sm._ccell(1.2, 1.0) in sm.floor


@pytest.fixture(scope="module")
def social() -> object:
    with SB.registered():
        yield SB


def _scare_run(name: str, seed: int) -> SB.SocialBrain:
    brain = SB.SocialBrain(preset(name), seed=seed)
    world = W.SocialSynth(seed=seed, x=2.5, y=3.0, yaw=0.3)
    world.scare(1_000_000)
    W.run_social(brain, world, 1180)
    return brain


def test_synthetic_temperament_the_timid_hide_the_bold_do_not(social: object) -> None:
    # A population test, not pinned seeds. a:social's original measurement: the timid preset
    # reached a shelter and peeked 4/8, the bold 0/8. Re-measured 2026-10-02 at 6ec7f91:
    # snuggler seeds 1-24, 9/24 peeked >= 2 times (and 20/24 started a hide); explorer seeds
    # 1-16, 0/16 hid. Three hand-picked seeds are not a fair sample: whether a timid duck gets
    # into a shelter is bimodal (it either settles in with 4-10 peeks or gives up with 0) at
    # roughly 40 % per seed, so "at least 2 of 3 pinned seeds" is close to a coin flip and moves
    # with any change that shifts a trajectory.
    timid = [_scare_run("snuggler", s) for s in range(1, 25)]
    bold = [_scare_run("explorer", s) for s in range(1, 9)]
    hid = [b for b in timid if b.hide_run.peeks >= 2]
    assert len(hid) >= 5
    assert all(b.counts.get("hide", 0) == 0 for b in bold)
