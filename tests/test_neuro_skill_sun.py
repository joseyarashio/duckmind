"""a:play — 曬太陽: the sun on the floor (duckmind.world.neuro_sun) and the duck's light sense and
basking act (duckmind.neuro.skill_sun). Pure: no physics."""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime

import pytest

from duckmind.neuro import contract as C
from duckmind.neuro import skill_sun as SS
from duckmind.neuro.skill_common import OdomHistory
from duckmind.world import lights as L
from duckmind.world import neuro_sun as NS


def test_solar_position_is_plausible_for_the_home() -> None:
    # Taipei (25.03 N): at the September equinox the sun culminates ~65° up, due south
    alt, az = NS.solar_position(datetime(2026, 9, 23, 11, 45))
    assert math.degrees(alt) == pytest.approx(65.0, abs=2.0)
    assert math.degrees(az) == pytest.approx(180.0, abs=15.0)
    # morning: low in the east; night: below the horizon
    alt_m, az_m = NS.solar_position(datetime(2026, 9, 28, 7, 30))
    assert 10.0 < math.degrees(alt_m) < 35.0 and 60.0 < math.degrees(az_m) < 120.0
    alt_n, _ = NS.solar_position(datetime(2026, 9, 28, 23, 0))
    assert alt_n < 0.0


def test_direct_lux_grows_with_altitude_and_is_zero_when_low() -> None:
    assert NS.direct_lux(math.radians(1.0)) == 0.0
    lows = [NS.direct_lux(math.radians(a)) for a in (5, 15, 30, 60)]
    assert lows == sorted(lows)
    # through a window at mid-morning: tens of thousands of lux on the floor (bright sun)
    assert 20_000 < NS.direct_lux(math.radians(30.0)) < 80_000


def test_sun_direction_points_east_in_the_morning() -> None:
    # north along +y in the world (north_rad = π/2): east is +x
    dx, dy, dz = NS.sun_direction(math.radians(20), math.radians(90), math.pi / 2)
    assert dx == pytest.approx(math.cos(math.radians(20)), abs=1e-9)
    assert dy == pytest.approx(0.0, abs=1e-9) and dz > 0.0


def test_window_patch_falls_on_the_floor_inside_the_room() -> None:
    # an east window (looking out along +x) at x = 3, 1 m wide, 1.2 m tall, its middle 1.3 m up
    win = L.Window("w_e", (3.0, 0.0, 1.3), (1.0, 1.2), (1.0, 0.0))
    sun = NS.sun_direction(math.radians(25), math.radians(90), math.pi / 2)
    patch = NS.window_patch(win, sun)
    assert patch is not None and len(patch) == 4
    assert all(x < 3.0 for x, _ in patch)  # on the room side
    xs = [x for x, _ in patch]
    # the sill 0.7 m up and the lintel 1.9 m up at 25°: 1.5–4.1 m into the room
    assert min(xs) == pytest.approx(3.0 - 1.9 / math.tan(math.radians(25)), abs=1e-6)
    assert max(xs) == pytest.approx(3.0 - 0.7 / math.tan(math.radians(25)), abs=1e-6)
    # the sun behind the wall: no patch
    west = NS.sun_direction(math.radians(25), math.radians(270), math.pi / 2)
    assert NS.window_patch(win, west) is None


def test_clip_polygon_and_point_in_polygon() -> None:
    square = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
    tri = [(1.0, 1.0), (3.0, 1.0), (1.0, 3.0)]
    cut = NS.clip_polygon(tri, square)
    # (the triangle's part inside the square is the square [1, 2]²: its long side x + y = 4
    # passes outside it)
    assert abs(NS.polygon_area(cut)) == pytest.approx(1.0, abs=1e-9)
    assert NS.point_in_polygon(1.2, 1.2, cut) and not NS.point_in_polygon(2.5, 1.2, cut)
    wedge = NS.clip_polygon([(1.0, 1.0), (3.0, 1.0), (1.0, 2.0)], square)
    assert abs(NS.polygon_area(wedge)) == pytest.approx(0.75, abs=1e-9)


def test_cloudy_has_no_patch() -> None:
    home = L.HomeLights("test", ())
    assert NS.SunModel(home, weather="cloudy").patches(datetime(2026, 9, 28, 9, 0)) == ()
    with pytest.raises(ValueError):
        NS.SunModel(home, weather="rain")


def _entry(lux: float, vis: int = 12, size: float = 0.3) -> dict[str, object]:
    return {
        "id": "sun:w_e:0",
        "b": 0.1,
        "e": -0.5,
        "rng": 0.8,
        "size": size,
        "lux": lux,
        "vis": vis,
    }


def test_the_camera_reports_only_light_that_stands_out() -> None:
    model = NS.SunSightModel(seed=3)
    # a patch 3× the floor it stands on is found on most still looks
    seen = sum(
        bool(model.step(t, [_entry(9000.0)], lux_under=300.0, stepping=False, saccade=False))
        for t in range(0, 20_000, 200)
    )
    assert seen >= 80  # (P_SEE_STILL 0.9 over 100 looks)
    # standing in it already: it does not stand out
    assert model.step(0, [_entry(9000.0)], lux_under=8000.0, stepping=False, saccade=False) == []
    # hidden, too small, or the head turning fast: nothing
    assert (
        model.step(0, [_entry(9000.0, vis=0)], lux_under=300.0, stepping=False, saccade=False) == []
    )
    assert (
        model.step(0, [_entry(9000.0, size=0.01)], lux_under=300.0, stepping=False, saccade=False)
        == []
    )
    assert model.step(0, [_entry(9000.0)], lux_under=300.0, stepping=False, saccade=True) == []
    got = model.step(123, [_entry(9000.0)], lux_under=300.0, stepping=False, saccade=False)
    if got:
        s = got[0]
        assert s.kind == "light" and s.lux is not None and 5000 < s.lux < 16000
        assert s.range_m is not None and 0.4 < s.range_m < 1.2


def test_meter_points_look_at_the_floor_ahead() -> None:
    cam = (0.0, 0.0, 0.2)
    level = NS.meter_points(cam, (1.0, 0.0, 0.0))  # the axis level: the rows 0, 15.5°, 31° down
    assert [round(p[0], 2) for p in level] == [
        1.5,
        round(0.2 / math.tan(math.radians(15.5)), 2),
        round(0.2 / math.tan(math.radians(31.0)), 2),
    ]
    down = NS.meter_points(cam, (math.cos(0.6), 0.0, -math.sin(0.6)))  # looking well down
    assert all(p[0] < 0.4 for p in down)
    assert NS.log_mean([100.0, 10_000.0]) == pytest.approx(1000.0)


def _frame(
    t_ms: int, lux: float | None, sightings: tuple = (), usable: bool = True
) -> C.SensorFrame:
    cam = NS.PlayCameraSense(usable=usable, sightings=sightings, age_ms=0, lux=lux)
    return C.SensorFrame(t_ms=t_ms, body=C.BodySense(posture="stand"), camera=cam, day_minute=510.0)


def test_sun_sense_needs_three_bright_readings_against_its_usual_light() -> None:
    sense = SS.SunSense()
    hist = OdomHistory()
    t = 0
    for _ in range(20):  # a while in the ordinary room light
        hist.add(t, (0.0, 0.0, 0.0))
        sense.step(t, 0.5, _frame(t, 250.0), hist)
        t += 500
    assert not sense.in_sun()
    hist.add(t, (0.0, 0.0, 0.0))
    sense.step(t, 0.5, _frame(t, 12_000.0), hist)  # one bright reading (a head turn, an edge)
    assert not sense.in_sun()
    for _ in range(3):
        t += 500
        hist.add(t, (0.0, 0.0, 0.0))
        sense.step(t, 0.5, _frame(t, 12_000.0), hist)
    assert sense.in_sun()
    # its usual light does not creep up while it sits in the sun
    assert 10**sense.log_typ < 400.0  # type: ignore[operator]


def test_sun_sense_remembers_a_sunny_spot_it_saw() -> None:
    sense = SS.SunSense()
    hist = OdomHistory()
    light = NS.LightSighting(
        kind="light",
        bearing=0.2,
        elevation=-0.4,
        range_m=0.9,
        size=0.3,
        motion=0.0,
        looming=0.0,
        track=9000,
        source="model",
        lux=9000.0,
    )
    hist.add(0, (0.0, 0.0, 0.0))
    sense.step(0, 0.05, _frame(0, 250.0, (light,)), hist)
    assert len(sense.spots) == 1
    spot = sense.best(1000, (0.0, 0.0, 0.0), 510.0)
    assert spot is not None
    gx, gy = spot.goal()
    assert math.hypot(gx, gy) > 0.9  # its goal is past the near edge, into the light
    feats = SS.sun_features(sense, 1000, (0.0, 0.0, 0.0), 510.0)
    assert feats["sunny"] > 0.5 and feats["in_sun"] == 0.0
    # a dim "light" is no sun
    dim = replace(light, lux=600.0, track=9001, bearing=-1.0)
    sense.step(500, 0.05, _frame(500, 250.0, (dim,)), hist)
    assert len(sense.spots) == 1


def test_sunbask_is_an_act_with_plain_words() -> None:
    bask = SS.SunBask()
    bask.start(0)
    assert bask.phase == "idle"
    assert SS.PHASE_ZH and all(isinstance(v, str) for v in SS.PHASE_ZH.values())


def test_walking_in_bends_toward_the_light_it_saw() -> None:
    """Walking into the sun it steers at the brightest floor it saw last (or the spot it
    remembers), never just straight on (coming up along a patch's edge, straight on kept it beside
    the sun)."""
    sense = SS.SunSense()
    frame = _frame(10_000, 250.0)
    odom = (0.0, 0.0, 0.0)
    # nothing seen and nothing remembered: straight on
    assert SS.SunBask._toward(10_000, sense, odom, frame) == 0.0
    # the bright floor seen just now 0.4 rad to its left: a left arc
    sense.best_bearing, sense.best_ms = 0.4, 9_500
    assert SS.SunBask._toward(10_000, sense, odom, frame) == pytest.approx(SS.INTO_GAIN * 0.4)
    # far off to the right: the arc walk's full rate, no more
    sense.best_bearing = -2.0
    assert SS.SunBask._toward(10_000, sense, odom, frame) == -1.0
    # that look is stale: toward the remembered spot instead (ahead and to the right here)
    sense.best_ms = 10_000 - SS.BRIGHT_FRESH_MS - 1
    sense.spots.append(SS.SunSpot(0.8, -0.8, 0.0, 20_000.0, 9_000, 510.0))
    assert SS.SunBask._toward(10_000, sense, odom, frame) < 0.0
    # in the bask machine: the "into" walk carries that turn
    bask = SS.SunBask()
    bask.phase, bask.phase_ms = "into", 9_900
    out = bask.step(10_000, 0.02, frame, sense, odom, halted=False)
    assert out.phase == "into" and out.vx == pytest.approx(0.40) and out.vyaw < 0.0


def test_it_aims_for_the_middle_of_the_bright_floor_it_saw() -> None:
    """The sighting's elevation is the bright region's middle in the picture: over a level floor
    that gives its ground range — the spot's goal is that middle, and a look down at the strip in
    front of its feet does not move a spot it already has."""
    sense = SS.SunSense()
    hist = OdomHistory()
    hist.add(0, (0.0, 0.0, 0.0))
    elevation = -math.atan2(SS.CAM_H_M, 1.0)  # the middle 1.0 m out on the floor
    light = NS.LightSighting(
        kind="light",
        bearing=0.0,
        elevation=elevation,
        range_m=0.5,
        size=0.4,
        motion=0.0,
        looming=0.0,
        track=9000,
        source="model",
        lux=20_000.0,
    )
    sense.step(0, 0.05, _frame(0, 250.0, (light,)), hist)
    spot = sense.best(100, (0.0, 0.0, 0.0), 510.0)
    assert spot is not None and spot.cx is not None
    gx, gy = spot.goal()
    assert 1.0 < gx < 1.15 and abs(gy) < 0.01  # (the lens sits ~8 cm ahead of the trunk)
    before = spot.goal()
    near = replace(light, bearing=0.5, elevation=-math.atan2(SS.CAM_H_M, 0.3), range_m=0.2)
    hist.add(600, (0.0, 0.0, 0.0))
    sense.step(600, 0.05, _frame(600, 250.0, (near,)), hist)
    assert len(sense.spots) == 1 and spot.goal() == before
    # an elevation that does not fit (the middle nearer than the near edge): the edge + INTO_M
    odd = replace(light, bearing=2.5, elevation=-0.6, range_m=0.9, track=9002)
    sense.step(1200, 0.05, _frame(1200, 250.0, (odd,)), hist)
    far = [s for s in sense.spots if s.cx is None]
    assert len(far) == 1
