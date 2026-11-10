"""The bench's truth planner (duckmind.bench.truthplan) on the layout's truth floor, with a point body that does exactly
what it is asked: the route goes round what is in the way, is re-made when a thing has moved onto it, and the walk
obeys the body link's rules (arcs where a straight walk is held, a back arc off a touch)."""

from __future__ import annotations

import math

import pytest

from duckmind.bench import floor as F
from duckmind.bench import truthplan as TP
from duckmind.world import homelayout

DT = 0.02


@pytest.fixture()
def fm() -> F.FloorMap:
    return F.FloorMap(homelayout.load("real_home"))


def _walk(
    pilot: TP.Pilot,
    pose: tuple[float, float, float],
    goal: tuple[float, float],
    seconds: float = 40.0,
) -> tuple[list[tuple[float, float]], float]:
    """A point body that turns and walks exactly as asked (a pivot turns on the spot): its track and the time it took."""
    x, y, yaw = pose
    track = [(x, y)]
    t = 0.0
    while t < seconds and math.hypot(goal[0] - x, goal[1] - y) > 0.10:
        pilot.update(t, (x, y, yaw), goal)
        vx, vyaw, pivot, back = pilot.drive((x, y, yaw), goal, t=t)
        assert not back
        if pivot:
            yaw += pivot * 0.9 * DT
        else:
            yaw += vyaw * DT
            x += 0.5 * vx * math.cos(yaw) * DT  # (the body walks about half of what is asked)
            y += 0.5 * vx * math.sin(yaw) * DT
        t += DT
        track.append((x, y))
    return track, t


def test_the_route_goes_round_the_sofa_and_never_through_anything(fm: F.FloorMap) -> None:
    start, goal = (
        (-0.9, -1.62, 1.57),
        (-0.9, -0.2),
    )  # from in front of the sofa to the yoga mat behind it
    assert fm.line((start[0], start[1]), goal)["kind"] == "furniture"
    pilot = TP.Pilot(fm)
    track, _ = _walk(pilot, start, goal, seconds=90.0)
    assert math.hypot(goal[0] - track[-1][0], goal[1] - track[-1][1]) <= 0.10
    assert all(
        fm.kind_at(x, y) < F.THING for x, y in track
    )  # round the sofa's end, through nothing
    assert max(abs(x + 0.9) for x, _ in track) > 0.9
    assert pilot.stats()["no_route_share"] == 0.0 and pilot.stats()["tight"] == 0


def test_a_thing_pushed_onto_the_route_makes_a_new_route(fm: F.FloorMap) -> None:
    start, goal = (0.3, -2.0, 0.0), (1.3, -2.0)
    where = {"at": dict(fm.at)}
    pilot = TP.Pilot(fm, lambda: where["at"])
    pilot.update(0.0, start, goal)
    assert pilot.replans == 1
    first = pilot.waypoint(start)
    assert (
        first is not None and abs(first[1] - start[1]) < 0.05
    )  # straight ahead: nothing in the way
    where["at"] = {**where["at"], "cushion_1": (0.8, -2.0)}  # the cushion now lies on the line
    pilot.update(1.0, start, goal)
    assert pilot.replans == 2 and fm.kind_at(0.8, -2.0) == F.THING
    track, _ = _walk(pilot, start, goal)
    assert math.hypot(goal[0] - track[-1][0], goal[1] - track[-1][1]) <= 0.10
    assert max(abs(y + 2.0) for _, y in track) > 0.25  # it went round the cushion
    assert all(fm.kind_at(x, y) < F.THING for x, y in track)


def test_where_the_link_holds_a_straight_walk_it_asks_for_arcs_and_backs_off_a_touch(
    fm: F.FloorMap,
) -> None:
    pose, goal = (0.6, -0.3, 0.0), (1.4, -0.3)
    pilot = TP.Pilot(fm)
    pilot.update(0.0, pose, goal)
    vx, vyaw, pivot, back = pilot.drive(pose, goal, t=0.0)
    assert (
        vx > 0.3 and abs(vyaw) < 0.2 and not pivot and not back
    )  # the place is straight ahead: a straight walk
    vx, vyaw, pivot, back = pilot.drive(pose, goal, t=0.1, gate=("reflex_near",))
    assert (
        vx > 0.3 and abs(vyaw) >= 0.5 and not back
    )  # held within 30 cm of something: only an arc goes on
    side = math.copysign(1.0, vyaw)
    assert (
        pilot.drive(pose, goal, t=0.5)[1] * side >= 0.5
    )  # … and it keeps to that arc for a moment
    assert abs(pilot.drive(pose, goal, t=2.0)[1]) < 0.2  # … then walks straight again
    assert not pilot.drive(pose, goal, t=2.1, gate=("reflex_touch",))[
        3
    ]  # the brain's own escapes back it off
    vx, vyaw, pivot, back = pilot.drive(pose, goal, t=2.2, gate=("stall",), may_back=True)
    assert (
        back and vx < 0.0 and abs(vyaw) >= 0.5
    )  # alone with the legs: a back arc, as the link requires
    assert (
        pilot.drive(pose, goal, t=3.0, may_back=True)[3]
        and not pilot.drive(pose, goal, t=3.6, may_back=True)[3]
    )
    assert pilot.stats()["backs"] == 1


def test_a_place_with_no_room_is_walked_to_from_the_nearest_spot_that_has(fm: F.FloorMap) -> None:
    goal = (1.4, 0.0)  # 6 cm from the hall wall's face: no room for the body there
    assert fm.kind_at(*goal) == F.FREE and fm.clear_at(*goal) < F.BODY_RADIUS_M
    pilot = TP.Pilot(fm)
    track, _ = _walk(pilot, (0.7, -0.1, 0.0), goal, seconds=30.0)
    assert math.hypot(goal[0] - track[-1][0], goal[1] - track[-1][1]) <= 0.10
