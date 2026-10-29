"""The world's ToF model (:mod:`duckmind.world.tof_model`): the official sensor's geometry, and
what the multi-ray, strongest-target and status model add (spec §11.4).

MuJoCo tests (the render check); the target arithmetic runs anywhere.
"""

from __future__ import annotations

import importlib.util
import os
import threading
import types
from pathlib import Path
from typing import Any

import pytest

from duckmind.world import tof_model as tm

MJCF = os.environ.get("DUCKMIND_MICRODUCK_MJCF")


def _mjcf() -> Path:
    if not MJCF or not Path(MJCF, "scene_allcollisions.xml").is_file():
        pytest.skip("DUCKMIND_MICRODUCK_MJCF (official microduck MJCF folder) is not set")
    return Path(MJCF)


def _official_tof() -> Any:
    pytest.importorskip("mujoco")
    path = _mjcf().parents[1] / "sim" / "tof.py"
    spec = importlib.util.spec_from_file_location("official_sim_tof", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scene(
    things: list[tuple[str, int, list[float], list[float]]] = (),  # type: ignore[assignment]
    keyframe: str = "HOME",
) -> tuple[Any, Any]:
    """The official flat scene with the duck at a keyframe and extra static boxes/capsules:
    (name, group, size, pos) — a size of 2 is a vertical capsule (radius, half-height)."""
    mujoco = pytest.importorskip("mujoco")
    pytest.importorskip("numpy")
    spec = mujoco.MjSpec.from_file(str(_mjcf() / "scene_allcollisions.xml"))
    for name, group, size, pos in things:
        geom = spec.worldbody.add_geom()
        geom.name = name
        geom.group = group
        if len(size) == 2:
            geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
            geom.size = [size[0], size[1], 0.0]
        else:
            geom.type = mujoco.mjtGeom.mjGEOM_BOX
            geom.size = size
        geom.pos = pos
    model = spec.compile()
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    return model, data


def _quiet(**more: Any) -> tm.TofParams:
    """No noise and no status randomness: the geometry alone."""
    base = dict(sigma_m=0.0, sigma_share=0.0, weak_per_m=0.0, weak_incidence=0.0, phantom_p=0.0)
    base.update(more)
    return tm.TofParams(**base)


def _sensor_x(data: Any, model: Any) -> float:
    mujoco = pytest.importorskip("mujoco")
    return float(data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tof")][0])


# ── the target arithmetic ──────────────────────────────────────────────────────


def test_sub_returns_group_into_targets_nearest_first():
    hits = [(1.00, 1.0, 3), (1.02, 1.0, 3), (2.0, 0.5, 4), (1.03, 1.0, 3), (0.5, 1.0, 9)]
    found = tm.targets(hits, 9, gap_m=0.06)
    assert [round(t.distance, 3) for t in found] == [0.5, 1.017, 2.0]
    assert [round(t.share * 9) for t in found] == [1, 3, 1]
    assert found[0].geom == 9 and found[1].signal == pytest.approx((3 / 9) / 1.017**2, rel=1e-3)


def test_a_thin_thing_wins_up_close_and_loses_far_away():
    def strongest(pole_m: float, wall_m: float) -> float:
        hits = [(pole_m, 1.0, 1)] + [(wall_m, 1.0, 2)] * 8
        found = tm.targets(hits, 9, gap_m=0.06)
        return max(found, key=lambda t: t.signal).distance

    assert strongest(0.5, 2.0) == pytest.approx(0.5)  # a table leg by the duck
    assert strongest(1.2, 1.5) == pytest.approx(1.5)  # the same leg across the room


def test_weak_statuses_grow_with_range_and_obliquity_and_sunlight_shortens_the_reach():
    p = tm.TofParams()
    assert tm.weak_chance(0.25, 1.0, p) == 0.0
    assert tm.weak_chance(1.0, 1.0, p) < tm.weak_chance(3.0, 1.0, p) <= p.weak_max
    assert tm.weak_chance(1.0, 0.2, p) > tm.weak_chance(1.0, 1.0, p)
    assert tm.range_limit(0.0, p) == 4.0 and tm.range_limit(300.0, p) == pytest.approx(2.0)


def test_the_sub_beams_tile_each_zone_around_its_centre():
    dirs, zones = tm.sub_beams(3)
    assert len(dirs) == 576 and zones[:9] == [0] * 9 and zones[-1] == 63
    centres = tm.beams()
    for zone in (0, 27, 63):
        middle = dirs[zone * 9 + 4]  # the centre sub-ray is the zone's centre
        assert middle == pytest.approx(centres[zone], abs=1e-12)


# ── against the official sensor ────────────────────────────────────────────────


def test_the_zone_centres_are_the_official_sensor_s():
    official = _official_tof()
    model, _ = _scene()
    mujoco = pytest.importorskip("mujoco")
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tof")
    reference = official.Tof(model, site).directions
    assert reference.shape == (64, 3)
    for got, want in zip(tm.beams(), reference.tolist(), strict=True):
        assert got == pytest.approx(want, abs=1e-12)


def test_with_one_ray_and_no_noise_it_measures_what_the_official_one_does():
    official = _official_tof()
    np = pytest.importorskip("numpy")
    model, data = _scene([("wall", 0, [0.02, 1.5, 1.0], [1.2, 0.0, 1.0])])
    x = _sensor_x(data, model)
    mujoco = pytest.importorskip("mujoco")
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tof")
    reference = official.Tof(model, site, seed=0)
    sensor = tm.TofSensor(model, params=_quiet(sub=1), geomgroup=(1, 1, 1, 1, 1, 1))
    ours = sensor.frame_of(data)
    truth = sensor.log[-1]["truth_mm"]
    origin = data.site_xpos[site]
    rotation = data.site_xmat[site].reshape(3, 3)
    geom = np.zeros(1, dtype=np.int32)
    for zone in range(64):
        direction = np.ascontiguousarray(rotation @ reference.directions[zone])
        hit = mujoco.mj_ray(model, data, origin, direction, None, 1, -1, geom)
        if 0 <= hit <= 4.0:
            assert ours["status"][zone] == 5
            assert truth[zone] == int(hit * 1000.0)
            assert ours["distance_mm"][zone] == int(hit * 1000.0)
        else:
            assert ours["status"][zone] == 255 and ours["distance_mm"][zone] == 0
    assert 1.2 - 0.02 - x == pytest.approx(ours["distance_mm"][27] / 1000, abs=0.01)


def test_a_three_centimetre_pole_is_seen_near_but_not_by_a_single_ray():
    np = pytest.importorskip("numpy")
    model, data = _scene()
    x = _sensor_x(data, model)
    mujoco = pytest.importorskip("mujoco")
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tof")
    origin, rotation = data.site_xpos[site], data.site_xmat[site].reshape(3, 3)
    # a 3 cm pole 0.45 m ahead, 2.4° beside the centre ray of column 3 (19 mm: missed) and
    # 0.5° beside one of its sub-rays; a wall 2 m behind it
    azimuth = np.radians(2.8125 + 2.4)
    forward = rotation[:, 0] * np.cos(azimuth) + rotation[:, 1] * np.sin(azimuth)
    foot = origin + forward * 0.45
    model, data = _scene(
        [
            ("pole", 0, [0.015, 0.4], [float(foot[0]), float(foot[1]), 0.4]),
            ("wall", 0, [0.02, 2.0, 1.0], [x + 2.5, 0.0, 1.0]),
        ]
    )
    single = tm.TofSensor(model, params=_quiet(sub=1)).frame_of(data)
    multi = tm.TofSensor(model, params=_quiet()).frame_of(data)
    near = [d for d, s in zip(multi["distance_mm"], multi["status"], strict=True) if 0 < d < 600]
    assert near, "the multi-ray sensor should see the pole"
    assert all(d > 600 or d == 0 for d in single["distance_mm"]), "one ray per zone misses it"


def test_it_sees_only_what_infrared_sees():
    model, data = _scene(
        [
            ("glass", 1, [0.01, 1.0, 1.0], [0.6, 0.0, 1.0]),  # camera only: passed through
            ("helper", 5, [0.01, 1.0, 1.0], [0.7, 0.0, 1.0]),  # a sensor helper: invisible
            ("ir_only", 4, [0.01, 1.0, 1.0], [0.9, 0.0, 1.0]),  # seen by the ToF alone
        ]
    )
    x = _sensor_x(data, model)
    sensor = tm.TofSensor(model, params=_quiet())
    frame = sensor.frame_of(data)
    assert frame["distance_mm"][27] == pytest.approx((0.89 - x) * 1000, abs=15)
    assert sensor.log[-1]["geoms"][27] == "ir_only"


def test_statuses_and_noise_behave_and_draws_repeat_by_seed():
    np = pytest.importorskip("numpy")
    near_model, near_data = _scene([("wall", 0, [0.02, 2.0, 1.0], [0.5, 0.0, 1.0])])
    far_model, far_data = _scene([("wall", 0, [0.02, 3.0, 1.5], [3.3, 0.0, 1.5])])
    near = tm.TofSensor(near_model, seed=1)
    far = tm.TofSensor(far_model, seed=1)
    near_frames = [near.frame_of(near_data) for _ in range(60)]
    far_frames = [far.frame_of(far_data) for _ in range(200)]  # (σ to ±15 %: n = 200)

    def weak_share(frames: list[dict[str, Any]], zones: range) -> float:
        codes = [f["status"][z] for f in frames for z in zones]
        return sum(c in tm.WEAK_STATUSES for c in codes) / len(codes)

    upper = range(0, 32)  # the upper half looks at the wall, not the floor
    assert weak_share(far_frames, upper) > weak_share(near_frames, upper) + 0.05
    centre = [f["distance_mm"][27] for f in far_frames if f["status"][27] in (4, 5, 9, 13)]
    sigma = float(np.std(centre)) / 1000
    distance = float(np.mean(centre)) / 1000
    assert sigma == pytest.approx(0.003 + 0.005 * distance, rel=0.15)
    again = tm.TofSensor(far_model, seed=1)
    assert [again.frame_of(far_data) for _ in range(3)] == far_frames[:3]
    other = tm.TofSensor(far_model, seed=2)
    assert other.frame_of(far_data) != far_frames[0]
    why = far.log[-1]["why"]
    assert len(why) == 64 and set(why) <= set(tm.WHY.values())


def test_a_phantom_is_a_too_close_valid_return():
    model, data = _scene()
    sensor = tm.TofSensor(model, params=_quiet(phantom_p=1.0))
    frame = sensor.frame_of(data)
    assert set(frame["status"]) == {5} and all(20 <= d <= 90 for d in frame["distance_mm"])
    assert set(sensor.log[-1]["why"]) == {"p"}


def test_daylight_on_the_target_shortens_the_range():
    np = pytest.importorskip("numpy")
    model, data = _scene([("wall", 0, [0.02, 3.0, 1.5], [3.0, 0.0, 1.5])])
    dark = tm.TofSensor(model, params=_quiet()).frame_of(data)
    sunny = tm.TofSensor(model, params=_quiet(), ambient=lambda points: np.full(len(points), 300.0))
    lit = sunny.frame_of(data)
    assert dark["status"][27] == 5 and lit["status"][27] == 255
    assert sunny.log[-1]["why"][27] == "s"


def test_a_frame_of_the_world_takes_the_lock_only_to_copy():
    mujoco = pytest.importorskip("mujoco")
    model, data = _scene([("wall", 0, [0.02, 2.0, 1.0], [1.0, 0.0, 1.0])])
    held: list[bool] = []

    class Lock:
        def __init__(self) -> None:
            self.inner = threading.Lock()

        def __enter__(self) -> None:
            self.inner.acquire()
            held.append(True)

        def __exit__(self, *exc: Any) -> None:
            self.inner.release()

    world = types.SimpleNamespace(model=model, data=data, lock=Lock())
    sensor = tm.TofSensor(model)
    frame = sensor.capture(world)
    assert held == [True] and frame["rows"] == 8
    assert sensor.log[-1]["sim_time"] == pytest.approx(float(data.time))
    assert sensor.cost()["n"] == 1 and sensor.entries(limit=5)[0]["seq"] == 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tof") == sensor.site


def test_a_frame_before_the_first_forward_pass_is_empty_not_a_crash():
    mujoco = pytest.importorskip("mujoco")
    model, _ = _scene()
    fresh = mujoco.MjData(model)  # site_xmat all zeros: a zero-length ray would abort MuJoCo
    frame = tm.TofSensor(model).frame_of(fresh)
    assert set(frame["status"]) == {255}
