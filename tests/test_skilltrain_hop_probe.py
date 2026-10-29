"""hop:feasibility part (a)/(b) helpers: the strip's shapes, the probe's outcome rules on synthetic
records, the keyframe search's schedules and flight bookkeeping — pure (numpy only; no MuJoCo)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from duckmind.skilltrain import hop_probe as P
from duckmind.skilltrain import hop_trajopt as TO


def test_sharp_strip_is_one_box_round_strip_has_curved_edges() -> None:
    sharp = P.Obstacle(0.02, "sharp")
    (g,) = sharp.geoms()
    assert g["type"] == "box" and g["size"] == pytest.approx((0.02, 1.0, 0.01))
    assert g["pos"] == pytest.approx((P.FACE_X + 0.02, 0.0, 0.01))
    low = P.Obstacle(0.01, "round")  # r = h: quarter circles from the floor, a flat top between
    kinds = sorted(x["type"] for x in low.geoms())
    assert kinds == ["box", "cylinder", "cylinder"]
    high = P.Obstacle(0.05, "round")  # r = 2 cm: a vertical face under a half-round top
    kinds = sorted(x["type"] for x in high.geoms())
    assert kinds == ["box", "box", "cylinder"]
    assert "hop_obst_0" in high.xml()


@pytest.mark.parametrize("edge", ["sharp", "round"])
def test_top_surface_reaches_the_height_and_rounds_down_at_the_edges(edge: str) -> None:
    ob = P.Obstacle(0.01, edge)
    assert ob.top_z(0.02) == pytest.approx(0.01)
    assert ob.top_z(-0.001) == 0.0 and ob.top_z(0.041) == 0.0
    if edge == "round":
        assert ob.top_z(0.0) == pytest.approx(0.0, abs=1e-9)
        assert 0.0 < ob.top_z(0.002) < 0.01
    else:
        assert ob.top_z(0.0) == pytest.approx(0.01)


def _record(n: int, *, tilt=None, fallen=None, trunk=None, feet=None) -> dict:
    t = np.arange(n) * 0.02
    return {
        "t": t,
        "tilt": np.full(n, 5.0) if tilt is None else tilt,
        "fallen": np.zeros(n) if fallen is None else fallen,
        "trunk_x": np.linspace(-0.1, 0.2, n) if trunk is None else trunk,
        "foot_x": np.stack([np.linspace(-0.1, 0.2, n)] * 2, 1) if feet is None else feet,
        "toe_hits": np.zeros(n),
        "housing": np.zeros(n),
        "tau_legs": np.full((n, 10), 0.1),
        "t_touch": float(t[10]),
        "t_walk": 0.0,
        "t_walk_end": float(t[-1]),
    }


def test_classify_crossed_blocked_fall_stuck() -> None:
    ob = P.Obstacle(0.01, "sharp")
    n = 200
    assert P.classify(_record(n), ob)["outcome"] == "crossed"
    blocked = _record(n, trunk=np.full(n, -0.03), feet=np.full((n, 2), -0.02))
    assert P.classify(blocked, ob)["outcome"] == "blocked"
    fallen = np.zeros(n)
    fallen[120:] = 1
    tilt = np.full(n, 5.0)
    tilt[110:] = 80.0
    rec = _record(n, tilt=tilt, fallen=fallen, trunk=np.full(n, -0.03), feet=np.full((n, 2), -0.02))
    out = P.classify(rec, ob)
    assert out["outcome"] == "fall" and out["stumble"] and not out["fell_after_cross"]
    tilt = np.full(n, 5.0)
    tilt[50:120] = 50.0  # 1.4 s nose-down without robotd's verdict
    rec = _record(n, tilt=tilt, trunk=np.full(n, -0.03), feet=np.full((n, 2), -0.02))
    assert P.classify(rec, ob)["outcome"] == "stuck"


def test_a_fall_after_getting_over_is_still_a_fall() -> None:
    ob = P.Obstacle(0.02, "sharp")
    n = 200
    fallen = np.zeros(n)
    fallen[180:] = 1
    out = P.classify(_record(n, fallen=fallen), ob)
    assert out["outcome"] == "fall" and out["fell_after_cross"]


def test_seeds_are_the_same_duck_on_both_servo_models() -> None:
    assert P.seed_of("sharp/20mm/v0.40", 3) == P.seed_of("sharp/20mm/v0.40", 3)
    assert P.seed_of("sharp/20mm/v0.40", 3) != P.seed_of("sharp/20mm/v0.40", 4)
    assert len({c.key for c in P.cells()}) == len(P.cells()) == 6 * 2 * 3 + 3


@pytest.mark.parametrize("family", sorted(TO.FAMILIES))
def test_schedules_stay_inside_robotd_clamp_and_start_from_home(family: str) -> None:
    params = TO.FAMILIES[family]()
    for u in (0.0, 0.5, 1.0):
        x = TO._to_x(np.full(len(params), u), params)
        ticks = TO.ticks_of(TO.schedule(family, x))
        assert ticks.shape[1] == 14
        assert np.all(np.abs(ticks) <= math.pi + 1e-9)
        assert np.allclose(ticks[:, [5, 6, 7, 8]], TO.HOME[[5, 6, 7, 8]])  # the head stays at HOME


def test_hop_legs_mirror_and_runs_and_tilt() -> None:
    q = TO._mirror(TO.HOME, (0.1, 0.2, 0.3))
    assert q[2] - TO.HOME[2] == pytest.approx(0.1) and q[11] - TO.HOME[11] == pytest.approx(-0.1)
    assert TO._runs(np.array([0, 1, 1, 0, 1])) == [(1, 3), (4, 5)]
    up = np.array([[1.0, 0.0, 0.0, 0.0]])
    side = np.array([[math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]])  # 90° roll
    assert TO._tilt_deg(up)[0] == pytest.approx(0.0)
    assert TO._tilt_deg(side)[0] == pytest.approx(90.0)


def test_gate_cells_cover_heights_edges_and_entries() -> None:
    from duckmind.skilltrain import hop_gate as G

    cells = G.cells()
    assert len(cells) == 4 * 2 * 2
    assert {c.entry for c in cells} == {"walk", "stand"}
    assert G.seed_of(cells[0].key, 1) == G.seed_of(cells[0].key, 1)
    assert cells[0].key == "sharp/10mm/walk"  # gitleaks:allow
