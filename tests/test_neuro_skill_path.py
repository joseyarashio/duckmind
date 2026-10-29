"""a:play — arc paths for legs that only turn while they walk (duckmind.neuro.skill_path)."""

from __future__ import annotations

import math
import random

import pytest

from duckmind.neuro import skill_path as SP


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def test_every_word_lands_on_the_goal_and_the_first_is_shortest() -> None:
    rng = random.Random(7)
    words: set[str] = set()
    for _ in range(600):
        start = (rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-math.pi, math.pi))
        goal = (rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-math.pi, math.pi))
        found = SP.paths(start, goal, 0.35)
        assert found, "a Dubins path always exists"
        lengths = [p.length for p in found]
        assert lengths == sorted(lengths)
        for path in found:
            end = SP.sample(start, path)[-1]
            assert math.hypot(end[0] - goal[0], end[1] - goal[1]) < 1e-3
            assert abs(_wrap(end[2] - goal[2])) < 1e-3
            words.add(path.word)
    assert words <= set(SP.WORDS)
    assert {"LSL", "RSR", "LSR", "RSL"} <= words


def test_straight_ahead_is_a_straight() -> None:
    best = SP.paths((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.35)[0]
    assert best.length == pytest.approx(1.0, abs=1e-6)
    assert best.t == pytest.approx(0.0, abs=1e-6) and best.q == pytest.approx(0.0, abs=1e-6)


def test_samples_never_turn_tighter_than_the_radius() -> None:
    start, goal = (0.0, 0.0, 0.0), (0.2, 0.9, math.pi)
    path = SP.paths(start, goal, 0.35)[0]
    pts = SP.sample(start, path, step=0.02)
    for a, b in zip(pts, pts[1:], strict=False):
        ds = math.hypot(b[0] - a[0], b[1] - a[1])
        dh = abs(_wrap(b[2] - a[2]))
        if ds > 1e-6:
            assert dh / ds <= (1.0 / 0.35) * 1.01  # (a chord is a little shorter than its arc)


def test_plan_clear_keeps_off_the_toy() -> None:
    # the toy lies between the duck and the goal's far side: the shortest path clips it
    start = (0.0, 0.0, 0.0)
    toy = (0.6, 0.0)
    goal = (0.6, 0.6, -math.pi / 2)  # 0.6 m out on the toy's perpendicular, facing it
    got = SP.plan_clear(start, goal, toy, 0.26)
    assert got is not None
    path, pts = got
    assert SP.clearance(pts, toy) >= 0.26
    assert pts[-1][0] == pytest.approx(goal[0], abs=1e-3)


def test_plan_clear_none_when_boxed_in() -> None:
    # standing right at the toy, facing it: every path to the far goal sweeps over it
    assert SP.plan_clear((0.0, 0.0, 0.0), (0.0, 0.05, 0.0), (0.2, 0.0), 0.5) is None


def test_follower_walks_the_path_and_reports_drift() -> None:
    start, goal = (0.0, 0.0, 0.0), (1.0, 0.5, 0.0)
    path = SP.paths(start, goal, 0.35)[0]
    fol = SP.Follower(SP.sample(start, path))
    # an ideal unicycle following the commands reaches the goal
    x, y, h = start
    dt = 0.02
    for _ in range(2000):
        vx, vyaw = fol.step((x, y, h))
        if fol.done:
            break
        v = 0.3  # (the real legs walk slower than asked)
        x += v * math.cos(h) * dt
        y += v * math.sin(h) * dt
        h += vyaw * dt
    assert fol.done
    assert math.hypot(x - goal[0], y - goal[1]) < 0.08
    assert max(fol.trail) >= len(fol.points) - 3
    # a pose far off the path says so
    fol2 = SP.Follower(SP.sample(start, path))
    fol2.step((0.0, 0.5, 0.0))
    assert fol2.off > SP.OFF_PATH_M


def test_follower_commands_stay_legal() -> None:
    path = SP.paths((0, 0, 0), (-0.5, 0.2, math.pi), 0.35)[0]
    fol = SP.Follower(SP.sample((0, 0, 0), path))
    for pose in ((0, 0, 0), (0.1, 0.1, 1.0), (-0.3, 0.4, 2.5), (0.5, -0.5, -2.0)):
        vx, vyaw = fol.step(pose)
        assert vx in (0.0, 0.40)
        assert -1.0 <= vyaw <= 1.0
