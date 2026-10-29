"""The route search over the duck's own wall memory (neuro/clutter_route.py)."""

from __future__ import annotations

import math

from duckmind.neuro import clutter_route as CR


def line(
    x0: float, y0: float, x1: float, y1: float, step: float = 0.08
) -> list[tuple[float, float]]:
    n = max(1, int(math.hypot(x1 - x0, y1 - y0) / step))
    return [(x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n) for k in range(n + 1)]


def u_pocket() -> list[tuple[float, float]]:
    """A U open toward -x: back wall at x = 0 (y -0.6 .. 0.6), arms along y = +-0.6 from x = 0 to x = -1.2."""
    return line(0.0, -0.6, 0.0, 0.6) + line(0.0, 0.6, -1.2, 0.6) + line(0.0, -0.6, -1.2, -0.6)


def test_out_of_a_pocket_goes_out_of_the_mouth_not_through_a_wall():
    obstacles = u_pocket()
    m = CR.build((-0.5, 0.0), obstacles, [], [])
    route = CR.search(m, (-0.5, 0.0), (-0.5, 0.0), 1.0, None)
    assert route is not None
    # every point of the route is clear of the wall points by the body's radius
    for x, y in route.points:
        assert min(math.hypot(x - ox, y - oy) for ox, oy in obstacles) >= CR.BLOCK_R - 0.05
    assert route.target[0] < -1.3  # it came out at the mouth
    assert math.hypot(route.target[0] + 0.5, route.target[1]) >= 1.0


def test_goal_behind_the_pocket_is_reached_round_it():
    obstacles = u_pocket()
    m = CR.build((-0.5, 0.0), obstacles, [], [])
    route = CR.search(m, (-0.5, 0.0), (-0.5, 0.0), 1.0, (1.2, 0.0))
    assert route is not None
    assert math.hypot(route.target[0] - 1.2, route.target[1]) <= 0.85
    # it goes out of the mouth first and round an arm
    xs = [p[0] for p in route.points]
    assert min(xs) < -1.2
    for x, y in route.points:
        assert min(math.hypot(x - ox, y - oy) for ox, oy in obstacles) >= CR.BLOCK_R - 0.05


def test_walked_ground_is_cheaper_than_ground_nobody_looked_at():
    # two ways out of a start with nothing in the way: the walked trail (west) and the unknown (east), equally far
    trail = line(-0.2, 0.0, -1.6, 0.0, 0.1)
    m = CR.build((0.0, 0.0), [], [], trail)
    route = CR.search(m, (0.0, 0.0), (0.0, 0.0), 1.2, None)
    assert route is not None and route.target[0] < -0.9


def test_a_failed_place_is_blocked():
    m0 = CR.build((0.0, 0.0), [], [], [])
    r0 = CR.search(m0, (0.0, 0.0), (0.0, 0.0), 1.0, (1.5, 0.0))
    assert r0 is not None and all(abs(p[1]) < 0.3 for p in r0.points)
    m1 = CR.build((0.0, 0.0), [], [(0.7, 0.0, 0.35)], [])
    r1 = CR.search(m1, (0.0, 0.0), (0.0, 0.0), 1.0, (1.5, 0.0))
    assert r1 is not None
    assert all(math.hypot(p[0] - 0.7, p[1]) > 0.34 for p in r1.points)
    assert max(abs(p[1]) for p in r1.points) > 0.3  # it goes round


def test_no_way_is_none_and_the_start_is_always_free():
    ring = [(0.5 * math.cos(a / 20 * math.pi), 0.5 * math.sin(a / 20 * math.pi)) for a in range(40)]
    m = CR.build((0.0, 0.0), ring, [], [])
    assert CR.search(m, (0.0, 0.0), (0.0, 0.0), 1.5, None) is None  # shut in on every side
    # a point where the memory says a thing is, under the duck itself, does not block the start
    m = CR.build((0.0, 0.0), [(0.05, 0.0)], [], [])
    assert CR.search(m, (0.0, 0.0), (0.0, 0.0), 1.0, None) is not None


def test_route_cost_and_lookahead():
    m = CR.build((0.0, 0.0), [], [], [])
    route = CR.search(m, (0.0, 0.0), (0.0, 0.0), 1.0, (1.5, 0.0))
    assert route is not None
    near, la = CR.lookahead(route.points, (0.0, 0.0), 0.5)
    assert near == 0 and 0.4 <= math.hypot(la[0], la[1]) <= 0.65
    assert CR.route_cost(m, route.points, 0) > 0.0
    blocked = CR.build((0.0, 0.0), [], [(0.5, 0.0, 0.25)], [])
    assert CR.route_cost(blocked, route.points, 0) == float("inf")
