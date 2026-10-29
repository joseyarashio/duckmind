"""Walking up to a thing it is interested in (LEGS-ACCEPTANCE-2026-10-04): the thing a walk is toward
stays in mind for the whole walk, the walk is not done until it has got there, it goes for a far thing
with short stops, and a near thing the camera cannot see is not "gone"."""

from __future__ import annotations

import math

import pytest
from test_neuro_brain import _genome

from duckmind.duckbody import protocol as P
from duckmind.neuro import bodylink as B_LINK
from duckmind.neuro import brain as B
from duckmind.neuro import contract as C
from duckmind.neuro import motor as M
from duckmind.neuro import select as S
from duckmind.neuro.brain import NeuroBrain
from duckmind.neuro.learn import Learner

CTX = S.Context(sitting=False, standing=True, asleep=False, can_walk=True, wakefulness=1.0)
DT = 0.05
TICK_MS = 20


def test_the_belief_of_a_thing_is_where_it_was_seen_by_the_ducks_own_odometry() -> None:
    learn = Learner(_genome("bold_explorer").p)
    key, _ = learn.see_thing(3, "thing", (0.0, 0.0, 0.0), 1.0, 0.0, 0.0, 0.1, 0.0)
    # one second after it was last seen (remembered_thing leaves out anything younger than 2 s)
    assert learn.remembered_thing((0.0, 0.0, 0.0), 1.0) is None
    k, bearing, dist, interest = learn.thing_belief(key, (0.0, 0.0, 0.0), 1.0)  # type: ignore[misc]
    assert k == key and bearing == pytest.approx(0.0, abs=1e-6) and dist == pytest.approx(1.0)
    assert 0.0 < interest <= 1.5
    # the duck has walked 0.4 m ahead and turned left 90 degrees: the thing is 0.6 m away, to its right
    _, bearing, dist, _ = learn.thing_belief(key, (0.4, 0.0, math.pi / 2), 1.0)  # type: ignore[misc]
    assert dist == pytest.approx(0.6) and bearing == pytest.approx(-math.pi / 2)
    assert learn.thing_belief(999, (0.0, 0.0, 0.0), 1.0) is None


def _sat_of_first_switch(walking_to: bool) -> float:
    """A wandering duck; at 5 s a thing comes near over 20 s. The second at which it leaves the walk
    (999: never within 100 s)."""
    sel = S.Selector(_genome("social_songbird"), seed=1)
    feat = S.empty_features()
    feat[S.F["restless"]] = 0.55
    t = 0.0
    seen = 0
    while t < 100.0:
        x = 0.0 if t < 5.0 else min(1.0, (t - 5.0) / 20.0)
        feat[S.F["thing_near"]] = x
        feat[S.F["novel"]] = 0.8 * x
        _, new = sel.step(DT, feat, CTX, False, 0.0, walking_to=walking_to)
        if new:
            seen += 1
            if seen == 2:
                return t
        t += DT
    return 999.0


def test_a_walk_to_a_thing_is_not_done_until_it_has_got_there() -> None:
    # (without it the finishing rule gives the walk up once half of it is used)
    assert _sat_of_first_switch(True) > _sat_of_first_switch(False) + 10.0


def _walking_share(pursue: float) -> float:
    p = M.LocoParams()
    s = M.LocoState()
    walking = 0
    n = 0
    for k in range(60 * 50):
        s, out = M.loco_step(
            p, s, M.LocoInput(go=0.5, bearing=0.0, range_m=1.5, pursue=pursue), k * TICK_MS
        )
        walking += out.walking
        n += 1
    return walking / n


def test_going_for_a_thing_has_short_stops_but_still_stops() -> None:
    plain, going = _walking_share(0.0), _walking_share(1.0)
    assert going > plain + 0.15, (plain, going)
    assert going < 0.98  # not a flat run: the rhythm is still its own


def _brain() -> NeuroBrain:
    brain = NeuroBrain(_genome("bold_explorer"), 4107)
    brain.behaviour = "investigate"
    brain.sel.current = brain.sel.index("investigate")
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


def test_a_thing_lost_from_sight_is_still_the_goal_of_the_walk() -> None:
    brain = _brain()
    seen = C.Sighting(kind="thing", bearing=0.3, range_m=1.0, size=0.12, track=2)
    for k in range(50):
        cmd, _ = brain.step(TICK_MS * k, _frame(k, seen))
    assert cmd.behaviour == "investigate" and brain._bout.get("thing_key") is not None
    # the eye loses it for three seconds (past the 1 s attention lock and the 2 s memory minimum)
    for k in range(50, 200):
        cmd, _ = brain.step(TICK_MS * k, _frame(k, None))
        if 105 <= k <= 125:  # the attention lock (1 s) is over and remembered_thing waits 2 s
            assert cmd.behaviour == "investigate", k
            assert cmd.target is not None and cmd.target.kind == "place", k
            assert cmd.target.bearing == pytest.approx(0.28, abs=0.05)


class _NoThing:
    thing = None
    person = None


def test_a_thing_nearer_than_the_camera_sees_is_not_gone() -> None:
    """Looking where a thing should be and not seeing it, the duck takes it for gone — unless the
    place is nearer than the camera sees the floor (the last stretch of a walk up to a thing)."""
    brain = _brain()
    brain._stepping = False
    frame = _frame(0, None)

    def used(range_m: float) -> float:
        belief = (1, 0.0, range_m, 1.0)  # straight ahead, where the camera looks
        return brain._consumed(0.1, _NoThing(), 0.0, frame, 0.0, belief, (0.0, 0.0, 0.0), 5.0)

    assert used(0.8) == pytest.approx(0.1 / B.ABSENT_S)
    assert used(B.BLIND_NEAR_M - 0.05) == 0.0
    brain._stepping = True  # while it walks, not seeing a still thing says nothing about it
    assert used(0.8) == 0.0


def test_the_far_walk_is_a_walk_the_last_stretch_is_slow_but_not_stop_and_go() -> None:
    brain = _brain()
    far = brain._pursuit("investigate", 0.1, 1.2)
    assert 0.4 <= far <= 1.0
    assert brain._pursuit("investigate", 0.9, 1.2) == 0.0  # not yet facing it
    assert brain._pursuit("wander", 0.1, 1.2) == 0.0
    assert brain._pace("investigate", B.PACE_FAST, 1.2, far) > B.PACE_FAST
    near = brain._pursuit("investigate", 0.1, B.PURSUE_FAR_M - 0.05)
    assert near == pytest.approx(B.PURSUE_NEAR * brain._go_trait())  # lined up on it: few stops …
    assert brain._pursuit("investigate", 1.5, 0.3) == 0.0  # … behind it: its own rhythm
    # … and at the slowest steps of the last stretch
    assert brain._pace("investigate", B.PACE_FAST, B.NEAR_THING_M - 0.05, near) == B.ARC_VX_MIN


# ── the body link: the ToF stop in a curiosity walk toward its thing (the owner's A1) ──────────────


def _link_twist(request: B_LINK.BodyRequest, ahead_mm: int, *, ahead_b_mrad: int = 0) -> tuple:
    from test_neuro_bodylink import run, state

    link = B_LINK.BodyLink()
    run(link, B_LINK.BodyRequest(posture="stand"), 0.0, 1.0)  # up and standing
    depth = P.DepthView(0, ahead_mm, None, None, 0, False, 0, ahead_b_mrad)
    out = link.tick(state(1.0), request, depth=depth)
    return out.twist, out.gate


def test_a_walk_up_to_a_thing_goes_on_where_a_plain_walk_stops_for_the_tof() -> None:
    plain = B_LINK.BodyRequest(vx=0.4, posture="stand")
    toward = B_LINK.BodyRequest(vx=0.4, posture="stand", toward=(0, 180))
    twist, gate = _link_twist(plain, 200)
    assert twist == (0.0, 0.0) and "reflex_near" in gate  # 0.2 m ahead: the 0.3 m stop
    twist, gate = _link_twist(toward, 200)  # the thing it is going to, 0.18 m away by its belief
    assert twist == (0.4, 0.0) and "reflex_near" not in gate


def test_the_relaxed_stop_is_for_the_thing_and_not_for_something_else_in_the_way() -> None:
    toward = B_LINK.BodyRequest(vx=0.4, posture="stand", toward=(0, 800))  # a thing 0.8 m ahead
    twist, gate = _link_twist(toward, 200)  # … and a return 0.2 m ahead: a wall, not the thing
    assert twist == (0.0, 0.0) and "reflex_near" in gate
    sideways = B_LINK.BodyRequest(vx=0.4, posture="stand", toward=(900, 200))  # off to the side
    twist, gate = _link_twist(sideways, 200, ahead_b_mrad=0)
    assert twist == (0.0, 0.0) and "reflex_near" in gate
    # and the standoff is the act's own, not nothing: a return 5 cm ahead still stops it
    close = B_LINK.BodyRequest(vx=0.4, posture="stand", toward=(0, 60))
    twist, gate = _link_twist(close, 40)
    assert twist == (0.0, 0.0) and "reflex_near" in gate


def test_other_things_in_view_do_not_take_the_walk_from_its_thing() -> None:
    brain = _brain()
    mine = C.Sighting(kind="thing", bearing=0.3, range_m=1.0, size=0.12, track=2)
    for k in range(50):
        brain.step(TICK_MS * k, _frame(k, mine))
    other = C.Sighting(
        kind="thing", bearing=-0.7, range_m=0.9, size=0.2, track=3
    )  # new, bigger, nearer
    for k in range(50, 80):
        frame = C.SensorFrame(
            t_ms=TICK_MS * k,
            body=C.BodySense(posture="walk", odom=(0.0, 0.0, 0.0)),
            camera=C.CameraSense(usable=True, sightings=(other,)),
            power=C.PowerSense(battery=0.85),
        )
        cmd, _ = brain.step(TICK_MS * k, frame)
        assert cmd.behaviour == "investigate", k
        assert cmd.target is not None and cmd.target.bearing > 0.0, (k, cmd.target)  # still its own


def test_a_far_thing_draws_less_than_a_near_one() -> None:
    from duckmind.neuro import select as SEL

    def draw(range_m: float) -> tuple[float, float]:
        brain = NeuroBrain(_genome("bold_explorer"), 4107)
        seen = C.Sighting(kind="thing", bearing=0.0, range_m=range_m, size=0.12, track=2)
        for k in range(30):
            brain.step(TICK_MS * k, _frame(k, seen))
        return brain._feat[SEL.F["thing"]], brain._feat[SEL.F["novel"]]

    near_thing, near_novel = draw(0.9)
    far_thing, far_novel = draw(3.0)
    assert near_thing == pytest.approx(1.0) and near_novel > 0.3
    assert far_thing < 0.3 * near_thing and far_novel < 0.3 * near_novel
    assert draw(B.TRAVEL_FREE_M + 0.1)[0] > 0.95  # within the near range there is no discount
