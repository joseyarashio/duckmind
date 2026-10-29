"""Going home round a wall (owner item 1, 2026-10-04): the sensor-only wall follower of
``duckmind.neuro.detour``. Pure pieces on a hand-made boundary memory; the measured effect on the
whole duck is in tools/machine_verify/measure_unstuck.py (dock-behind scenarios)."""

from __future__ import annotations

import math
from types import SimpleNamespace

from duckmind.neuro import contract as C
from duckmind.neuro import detour as D
from duckmind.neuro.boundary import Boundary, _Point

EAST, SOUTH = 0.0, -math.pi / 2


def frame(left=None, ahead=None, right=None, present=True):
    tof = C.TofSense(
        present=present, left_mm=left, ahead_mm=ahead, right_mm=right, age_ms=0, touch=False
    )
    return SimpleNamespace(tof=tof, events=())


def wall_south(x0=-1.5, x1=1.5, y=-0.8) -> Boundary:
    """A wall seen along y = ``y`` (odometry frame)."""
    b = Boundary()
    x = x0
    while x <= x1 + 1e-9:
        b.points.append(_Point(x, y, 1.0, "wall"))
        x += 0.08
    return b


def goal_at(odom, gx, gy):
    dx, dy = gx - odom[0], gy - odom[1]
    return math.atan2(dy, dx) - odom[2], math.hypot(dx, dy)


def run(det, bound, odom, goal, seconds, *, fr=None, t0=0, escapes=0):
    """Step at 10 Hz with the nest at ``goal`` (x, y); the last waypoint."""
    wp = None
    t = t0
    for _ in range(int(seconds * 10)):
        gb, gr = goal_at(odom, *goal)
        wp = det.step(t, bound, odom, fr or frame(), gb, gr, escapes=escapes)
        t += 100
    return wp, t


def test_no_detour_while_it_is_getting_nearer_the_nest():
    det, bound = D.Detour(True), wall_south()
    odom = (0.0, 0.0, SOUTH)
    t = 0
    for i in range(80):  # 8 s, nearer by 0.05 m a step: progress, blocked or not
        gb, gr = goal_at(odom, 1.5, -3.0 + 0.05 * i * 0.0)
        wp = det.step(t, bound, odom, frame(), gb, gr - 0.05 * i)
        t += 100
        assert not wp.following
    assert not det.following


def test_a_wall_on_the_bearing_and_no_progress_begins_a_detour_along_the_wall_toward_the_nest():
    det, bound = D.Detour(True), wall_south()
    odom = (0.0, 0.0, SOUTH)
    wp, _ = run(det, bound, odom, (1.5, -3.0), 8.0)
    assert det.following and wp.following
    # the wall is south; the nest is south-east: it goes east, the wall on its right
    assert det.wall == -1
    # the walk is turned aside from the nest's bearing toward east, a quarter turn or so
    assert 0.8 <= det.theta <= D.THETA_MAX
    assert wp.bearing > 0.5  # a turn to the left, from facing south toward east


def test_it_goes_the_other_way_for_a_nest_on_the_other_side():
    det, bound = D.Detour(True), wall_south()
    odom = (0.0, 0.0, SOUTH)
    run(det, bound, odom, (-1.5, -3.0), 8.0)
    assert det.following and det.wall == 1
    assert 0.8 <= det.theta <= D.THETA_MAX and det._last.bearing < -0.5  # to the right, west


def test_a_nest_within_1_2_m_is_the_dockings_business():
    det, bound = D.Detour(True), wall_south(y=-0.5)
    odom = (0.0, 0.0, SOUTH)
    wp, _ = run(det, bound, odom, (0.0, -1.0), 10.0)
    assert not wp.following and not det.following


def test_it_ends_when_the_way_to_the_nest_is_open_again():
    det, bound = D.Detour(True), wall_south(x0=-1.5, x1=0.2)
    odom = (0.0, 0.0, SOUTH)
    _, t = run(det, bound, odom, (1.5, -3.0), 8.0)
    assert det.following
    # it has gone east past the wall's end: nothing is on the bearing, the ToF sees nothing
    clear = Boundary()
    odom = (1.0, 0.0, -1.0)
    wp, _ = run(det, clear, odom, (1.5, -3.0), 4.0, t0=t)
    assert not det.following and not wp.following


def test_it_tries_the_other_side_then_gives_up_for_a_while():
    det, bound = D.Detour(True), wall_south(x0=-4.0, x1=4.0)
    odom = (0.0, 0.0, SOUTH)
    _, t = run(det, bound, odom, (1.5, -3.0), 8.0)
    first = det.wall
    _, t = run(det, bound, odom, (1.5, -3.0), D.DETOUR_SIDE_S + 2.0, t0=t)
    assert det.wall == -first  # the other side once
    _, t = run(det, bound, odom, (1.5, -3.0), D.DETOUR_SIDE_S + 2.0, t0=t)
    assert not det.following and det.gave_up == 1
    wp, _ = run(det, bound, odom, (1.5, -3.0), 5.0, t0=t)  # the pause: the plain walk
    assert not wp.following


def test_the_episode_survives_another_act_for_a_moment_and_is_forgotten_after_a_while():
    det, bound = D.Detour(True), wall_south()
    odom = (0.0, 0.0, SOUTH)
    _, t = run(det, bound, odom, (1.5, -3.0), 8.0)
    assert det.following
    for i in range(50):  # 5 s of another act
        det.idle(t + 100 * i)
    assert det.following
    for i in range(50, 200):  # 20 s in all
        det.idle(t + 100 * i)
    assert not det.following


def test_the_walk_is_the_nests_bearing_turned_aside_and_turned_further_while_it_is_shut():
    det, bound = D.Detour(True), wall_south(x0=-4.0, x1=4.0)
    odom = (0.0, 0.0, SOUTH)
    run(det, bound, odom, (1.5, -3.0), 8.0)
    assert det.following
    gb, _ = goal_at(odom, 1.5, -3.0)
    wp, _ = run(det, bound, odom, (1.5, -3.0), 4.0, t0=100_000)
    # a long wall on the nest's bearing: the turn aside grows to its limit and stays
    assert math.isclose(det.theta, D.THETA_MAX, abs_tol=1e-6)
    assert math.isclose(abs(D.wrap_angle(wp.bearing - gb)), D.THETA_MAX, abs_tol=1e-6)
    assert (wp.bearing - gb) * det.wall < 0.0  # turned away from the wall's side


def test_the_turn_aside_is_given_back_while_the_way_to_the_nest_is_open():
    det, bound = D.Detour(True), wall_south(x0=-1.5, x1=0.2)
    odom = (0.0, 0.0, SOUTH)
    _, t = run(det, bound, odom, (1.5, -3.0), 8.0)
    assert det.following
    theta0 = det.theta
    # past the wall's end: nothing on the bearing; the turn shrinks by THETA_DOWN a second
    odom = (1.0, 0.0, -1.0)
    run(det, Boundary(), odom, (1.5, -3.0), 1.0, t0=t)
    assert det.following and det.theta < theta0 - 0.3
    run(det, Boundary(), odom, (1.5, -3.0), 4.0, t0=t + 1000)
    assert not det.following and det.theta == 0.0


def test_an_escape_toward_the_nest_counts_as_blocked():
    det, bound = D.Detour(True), Boundary()  # the memory shows nothing (a low box, a blind zone)
    odom = (0.0, 0.0, EAST)
    _, t = run(det, bound, odom, (3.0, 0.0), 6.0)
    assert not det.following
    wp, _ = run(det, bound, odom, (3.0, 0.0), 2.0, t0=t, escapes=1)
    assert det.following


def test_a_detour_that_is_off_walks_the_plain_bearing_whatever_stands_in_the_way():
    det, bound = D.Detour(False), wall_south()
    odom = (0.0, 0.0, SOUTH)
    gb, _ = goal_at(odom, 1.5, -3.0)
    wp, _ = run(det, bound, odom, (1.5, -3.0), 12.0)
    assert not det.following and not wp.following and wp.bearing == gb
    assert D.Detour().enabled is D.DETOUR_ENABLED  # unsaid: the setting
