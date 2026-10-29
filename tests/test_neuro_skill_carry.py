"""a:play — 叼玩具回窩: the carry act's pure parts (duckmind.neuro.skill_carry)."""

from __future__ import annotations

import math

import pytest

from duckmind.neuro import skill_carry as K
from duckmind.neuro.skill_common import PlayThing, ThingBook


def test_drop_spot_is_beside_the_dock_on_the_side_the_j_hook_does_not_use() -> None:
    carry = K.CarryHome(turn_side="right")
    # the duck 1 m out on the dock's axis, facing the dock (heading π in the dock's frame),
    # its odometry at the origin facing +x
    spot = carry.drop_spot((1000.0, 0.0, math.pi), (0.0, 0.0, 0.0))
    assert spot is not None
    # the spot is 0.32 m out from the wall and 0.42 m to the dock's left (facing the room):
    # from the duck (facing the wall) that is 0.68 m ahead and 0.42 m to its RIGHT
    assert spot[0] == pytest.approx(1.0 - K.DROP_OUT_MM / 1000.0, abs=1e-6)
    assert spot[1] == pytest.approx(-K.DROP_SIDE_MM / 1000.0, abs=1e-6)
    left = K.CarryHome(turn_side="left").drop_spot((1000.0, 0.0, math.pi), (0.0, 0.0, 0.0))
    assert left is not None and left[1] == pytest.approx(K.DROP_SIDE_MM / 1000.0, abs=1e-6)


def test_no_dock_frame_no_spot_and_no_toy_feature() -> None:
    carry = K.CarryHome()
    assert carry.drop_spot(None, (0.0, 0.0, 0.0)) is None
    book = ThingBook()
    book.things[1] = PlayThing(1, 0.5, 0.0, 0, 0, axis=0.3)
    assert K.carry_features(book, (0.0, 0.0, 0.0), 1000, None, False)["toy"] == 0.0
    assert K.carry_features(book, (0.0, 0.0, 0.0), 1000, (-1.0, 0.0), False)["toy"] > 0.3


def test_toy_feature_skips_rollers_home_toys_and_shapeless_things() -> None:
    book = ThingBook()
    home = (-1.0, 0.0)
    book.things[1] = PlayThing(1, 0.5, 0.0, 0, 0, axis=0.3, rolled=1)  # a ball
    book.things[2] = PlayThing(2, -1.1, 0.1, 0, 0, axis=0.3)  # already beside the dock
    book.things[3] = PlayThing(3, 0.4, 0.2, 0, 0)  # no long side seen yet
    assert K.carry_features(book, (0.0, 0.0, 0.0), 1000, home, False)["toy"] == 0.0
    book.things[4] = PlayThing(4, 0.6, 0.0, 0, 0, axis=1.0)
    near = K.carry_features(book, (0.0, 0.0, 0.0), 1000, home, False)["toy"]
    book.things[4].held = 1  # it has carried it before: a known toy draws it more
    known = K.carry_features(book, (0.0, 0.0, 0.0), 1000, home, False)["toy"]
    assert 0.0 < near < known <= 1.0
    # remembered, fading over minutes
    later = K.carry_features(book, (0.0, 0.0, 0.0), int(1000 * K.TOY_MEMORY_S * 0.9), home, False)
    assert 0.0 < later["toy"] < known
    assert K.carry_features(book, (0.0, 0.0, 0.0), 0, home, True)["carrying"] == 1.0


def test_line_frame() -> None:
    # a toy straight ahead lying across the heading: the duck is on its perpendicular
    psi, e = K.line_frame(0.30, 0.0, math.pi / 2)
    assert psi == pytest.approx(0.0, abs=1e-9) and e == pytest.approx(0.0, abs=1e-9)
    # the same toy 4 cm to the left: the duck is 4 cm to the right of the line
    psi, e = K.line_frame(0.30, 0.04, math.pi / 2)
    assert psi == pytest.approx(0.0, abs=1e-9) and e == pytest.approx(-0.04, abs=1e-9)
    # turned 10° off the perpendicular
    psi, _ = K.line_frame(0.30, 0.0, math.pi / 2 + math.radians(10))
    assert abs(psi) == pytest.approx(math.radians(10), abs=1e-9)
    # no axis seen: only the side offset
    assert K.line_frame(0.3, 0.02, None) == (0.0, -0.02)


def test_axis_error() -> None:
    assert K.axis_error(None, 0.3) == 0.0
    assert K.axis_error(math.pi / 2, 0.0) == pytest.approx(0.0, abs=1e-9)
    assert K.axis_error(-math.pi / 2, 0.0) == pytest.approx(0.0, abs=1e-9)  # a line has no front
    assert abs(K.axis_error(math.pi / 2 + 0.2, 0.0)) == pytest.approx(0.2, abs=1e-9)


def _reach_until(rc: K.Reach, t: int, rel, seen, still=True, axis=math.pi / 2, n=400):  # type: ignore[no-untyped-def]
    """Step the reach until it asks for something other than a look (the legs: still unless a
    move is under way — as robotd's applied twist would say)."""
    out = None
    for _ in range(n):
        out = rc.step(t, rel, seen(t), still and rc.move is None, axis)
        if out[0] not in ("looking",):
            return t, out
        t += 20
    return t, out


def _run_move(rc: K.Reach, t: int, rel, seen, axis=math.pi / 2):  # type: ignore[no-untyped-def]
    """Let the legs take the move and run it out; returns the time it ended."""
    while rc.move is not None:
        rc.step(t, rel, seen(t), False, axis)
        t += 20
    return t


def test_reach_creeps_then_goes_blind_then_ready() -> None:
    rc = K.Reach()
    rc.reset(0)
    # a toy 0.33 m ahead, dead on its perpendicular, seen on every pass
    t, out = _reach_until(rc, 0, (0.33, 0.0), lambda t: t)
    assert out is not None and out[0] == "moving" and rc.move_name == "creep"
    assert out[1] == pytest.approx(0.40) and abs(out[2]) <= 0.4
    # the clock waits for the legs: still legs keep the move asked for, not run
    for k in range(10):
        assert rc.step(t + 20 * k, (0.33, 0.0), t, True, math.pi / 2)[0] == "moving"
    assert rc.move_t0 is None
    # run the creep out, then from 0.22 m (seen again) the blind creep
    t = _run_move(rc, t + 200, (0.33, 0.0), lambda t: t)
    t, out = _reach_until(rc, t, (0.22, 0.0), lambda t: t)
    assert out is not None and out[0] == "moving" and rc.move_name == "blind"
    t = _run_move(rc, t, (0.22, 0.0), lambda t: -1)
    t, out = _reach_until(rc, t, (0.06, 0.0), lambda t: -1)  # (it cannot see it that near)
    assert out is not None and out[0] == "ready"


def test_a_move_the_legs_never_take_is_dropped() -> None:
    rc = K.Reach()
    rc.reset(0)
    t, out = _reach_until(rc, 0, (0.33, 0.0), lambda t: t)
    assert out is not None and rc.move is not None
    for k in range(120):  # 2.4 s of still legs (the body link holding them)
        state = rc.step(t + 20 * k, (0.33, 0.0), t, True, math.pi / 2)
    assert rc.move is None and "held" in rc.notes and state[0] == "looking"


def test_reach_says_crooked_when_off_the_long_side() -> None:
    rc = K.Reach()
    rc.reset(0)
    _, out = _reach_until(rc, 0, (0.22, 0.07), lambda t: t)  # 7 cm to the side, near
    assert out is not None and out[0] == "crooked"
    rc = K.Reach()
    rc.reset(0)
    _, out = _reach_until(rc, 0, (0.22, 0.0), lambda t: t, axis=math.pi / 2 + 0.5)  # 29° off
    assert out is not None and out[0] == "crooked"


def test_reach_backs_off_when_too_near_to_see() -> None:
    rc = K.Reach()
    rc.reset(0)
    _, out = _reach_until(rc, 0, (0.08, 0.0), lambda t: -1)
    assert out is not None and out[0] == "moving" and rc.move_name == "back_arc"
    assert out[1] < 0.0 and abs(out[2]) == pytest.approx(1.0)


def test_reach_pauses_a_creep_when_a_reflex_takes_the_head() -> None:
    rc = K.Reach()
    rc.reset(0)
    t, out = _reach_until(rc, 0, (0.33, 0.0), lambda t: t)
    assert out is not None and rc.move_name == "creep"
    state = rc.step(t + 100, (0.33, 0.0), t + 100, False, math.pi / 2, True)
    assert state[0] == "looking" and rc.move is None and "paused" in rc.notes


def test_reach_gives_up_after_too_many_creeps() -> None:
    rc = K.Reach()
    rc.reset(0)
    rc.moves = K.REACH_MAX_MOVES
    _, out = _reach_until(rc, 0, (0.33, 0.0), lambda t: t)
    assert out is not None and out[0] == "gave_up"


def test_pick_prefers_a_known_near_toy_and_skips_home_ones() -> None:
    carry = K.CarryHome()
    book = ThingBook()
    book.things[1] = PlayThing(1, 1.5, 0.0, 0, 0, axis=0.0)
    book.things[2] = PlayThing(2, 0.6, 0.0, 0, 0, axis=0.0, held=1)
    book.things[3] = PlayThing(3, -0.95, 0.0, 0, 0, axis=0.0, held=2)  # beside the dock
    got = carry.pick(book, (0.0, 0.0, 0.0), 1000, (-1.0, 0.0))
    assert got is not None and got.key == 2
    assert book.things[3].home is True


def test_plan_ends_on_the_perpendicular_facing_the_toy() -> None:
    carry = K.CarryHome()
    toy = PlayThing(1, 0.6, 0.0, 0, 0, axis=0.0)  # lying along x: its perpendicular is ±y
    carry._plan(toy, (0.0, 0.0, 0.0), 0)
    assert carry.sub == "path" and carry.follow is not None and carry.line_n is not None
    end = carry.follow.points[-1]
    nx, ny = carry.line_n
    assert abs(nx) == pytest.approx(0.0, abs=1e-9) and abs(ny) == pytest.approx(1.0, abs=1e-9)
    assert math.hypot(end[0] - 0.6, end[1] - ny * K.GOAL_M) < 1e-3
    assert math.cos(end[2] - math.atan2(-ny, -nx)) == pytest.approx(1.0, abs=1e-3)
    # never nearer the toy than the keep-out on the way
    assert min(math.hypot(p[0] - 0.6, p[1]) for p in carry.follow.points) >= K.KEEP_OUT_M - 1e-6


def test_the_grab_window_is_the_measured_one() -> None:
    assert K.GRAB_WINDOW.ahead == (52, 70)
    assert K.GRAB_WINDOW.side == (-40, 40)


def test_grab_clock_waits_for_real_policy_start_reflex_and_exit() -> None:
    from duckmind.neuro.contract import BodySense, SensorFrame

    carry = K.CarryHome()
    carry.target = 1
    carry.phase = "grab"
    carry.skill_seq = 1
    carry.acted_ms = 1000
    thing = PlayThing(1, 0.06, 0.0, 1000, 1000, axis=0.0)
    book = ThingBook()
    book.things[1] = thing
    odom = (0.0, 0.0, 0.0)
    home = (-1.0, 0.0)

    def frame(t_ms: int, active_skill: str | None = None) -> SensorFrame:
        return SensorFrame(
            t_ms=t_ms,
            body=BodySense(posture="stand", active_skill=active_skill),
        )

    waiting = carry._p_grab(1000, frame(1000), book, thing, odom, home, True)
    assert waiting.phase == "grab" and waiting.skill == "ground_pick"
    assert waiting.mouth == 0.0 and waiting.event == "grab" and carry.grab_started_ms is None
    # A readiness delay must not age the grab or start the mouth program. An acknowledgement
    # after 2300 ms (the bridge readiness plus policy-start budgets) but before our 3000 ms
    # bound starts both clocks at the observed policy time.
    waiting = carry._p_grab(2200, frame(2200), book, thing, odom, home, True)
    assert waiting.phase == "grab" and waiting.mouth == 0.0
    assert carry.grab_started_ms is None and carry.closed_ms is None

    started = carry._p_grab(3300, frame(3300, "ground_pick"), book, thing, odom, home, True)
    assert started.phase == "grab" and carry.grab_started_ms == 3300
    assert started.event is None  # policy start does not duplicate the selection event
    opened = carry._p_grab(3500, frame(3500, "ground_pick"), book, thing, odom, home, True)
    assert opened.mouth == 1.0

    # Even after the minimum hold, a still-running real policy and an intervening reflex keep
    # the carry state in grab. The clock is anchored to observed policy start, not selection.
    halted = carry.step(
        6900, frame(6900, "ground_pick"), book, odom, home, None, halted=True, standing=True
    )
    assert halted.phase == "grab" and halted.skill is None
    still_running = carry._p_grab(6950, frame(6950, "ground_pick"), book, thing, odom, home, True)
    assert still_running.phase == "grab" and still_running.skill == "ground_pick"

    exited = carry._p_grab(7000, frame(7000), book, thing, odom, home, True)
    assert exited.phase == "carry" and carry.holding
    assert exited.event == "grabbed" and carry.grab_started_ms is None


def test_grab_with_no_observed_policy_start_fails_boundedly() -> None:
    from duckmind.neuro.contract import BodySense, SensorFrame

    carry = K.CarryHome()
    carry.target = 1
    carry.phase = "grab"
    carry.skill_seq = 2
    carry.acted_ms = 1000
    thing = PlayThing(1, 0.06, 0.0, 1000, 1000, axis=0.0)
    book = ThingBook()
    book.things[1] = thing
    frame = SensorFrame(t_ms=4001, body=BodySense(posture="stand"))

    expired = carry._p_grab(4001, frame, book, thing, (0.0, 0.0, 0.0), (-1.0, 0.0), True)
    assert expired.phase == "search" and expired.skill is None
    assert expired.event == "missed" and carry.last_event == "missed"
    assert carry.grab_started_ms is None and not carry.holding
    assert carry.fails == 1 and thing.grab_fails == 1
