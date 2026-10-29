"""neuro:senses' world side (duckmind.world.neuro_world) in a real home, with MuJoCo: the head
camera's view geometry against a render of the camera itself, occlusion by the duck's own
position and by a wall, the dark, sound levels by distance, contacts only where a hand really
touches, and every page stimulus as a world event. The duck sits held in place (as the official
server holds a duck nobody enabled) with its head at robotd's HOME, so the camera looks ahead.

Skips without MuJoCo, numpy or the official MJCF folder (``DUCKMIND_MICRODUCK_MJCF``)."""

from __future__ import annotations

import math
import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from duckmind.world import neuro_senses as ns
from duckmind.world import neuro_world as nw

MJCF = os.environ.get("DUCKMIND_MICRODUCK_MJCF")
HOME_HEAD = {"neck_pitch": 0.3491, "head_pitch": 0.3491, "head_yaw": 0.0, "head_roll": 0.0}
NIGHT = datetime(2026, 12, 20, 20, 0)  # the lamps' hours: switching them matters


def _mjcf() -> str:
    if not MJCF or not Path(MJCF, "scene_allcollisions.xml").is_file():
        pytest.skip("DUCKMIND_MICRODUCK_MJCF (official microduck MJCF folder) is not set")
    pytest.importorskip("mujoco")
    pytest.importorskip("numpy")
    return MJCF


class Home:
    """A home with the household and the neuro part; the duck held seated, head at HOME."""

    def __init__(
        self,
        layout: Any = "default_home",
        *,
        x: float = 0.0,
        y: float = 0.3,
        yaw: float = 0.0,
        when: datetime = NIGHT,
        seed: int = 7,
    ) -> None:
        mjcf = _mjcf()  # skips here without MuJoCo, numpy or the official MJCF
        import mujoco

        from duckmind.world import lights, worldkit

        self.mj = mujoco
        self.w = worldkit.Headless(layout, mjcf_dir=mjcf, keyframe="SIT")
        self.m, self.d = self.w.model, self.w.data
        home = lights.HomeLights.from_layout(self.w.layout)
        self.clock = {"when": when, "lamps": True}

        def state() -> Any:
            return lights.home_light_state(
                self.clock["when"], home, lamps_on=bool(self.clock["lamps"])
            )

        def switch(on: bool) -> None:
            self.clock["lamps"] = on

        self.part = nw.attach(self.w, seed=seed, lights=state, switch_lamps=switch)
        q, dof = self.w.robot.qadr, self.w.robot.dof
        for name, value in HOME_HEAD.items():
            j = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.d.qpos[self.m.jnt_qposadr[j]] = value
            a = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            self.d.ctrl[a] = value
        self.d.qpos[q : q + 3] = [x, y, 0.075]
        self.d.qpos[q + 3 : q + 7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        self.d.qvel[:] = 0.0
        mujoco.mj_forward(self.m, self.d)
        held = self.d.qpos[q : q + 7].copy()

        def hold(world: Any) -> None:
            self.d.qpos[q : q + 7] = held
            self.d.qvel[dof : dof + 6] = 0.0

        self.w.after.insert(0, hold)
        self.hub = ns.SenseHub(seed=seed)
        self.frames: list[Any] = []

    def put(self, key: str, x: float, y: float) -> None:
        props = self.w.kit.props
        with self.w.world.lock:
            answer = props._put(self.w.world, props.things[key], x, y, 0.0, top=0.05)
        assert answer["ok"], answer

    def run(self, seconds: float, *, senses: bool = True) -> None:
        for _ in range(int(round(seconds / 0.02))):
            self.w.step(4)
            self.part.pump()
            if senses:
                with self.w.world.lock:
                    section = self.part.snapshot(self.w.world)
                doors = SimpleNamespace(
                    t_ms=int(round(self.d.time * 1000)),
                    state=None,
                    link=None,
                    events=(),
                    depth=None,
                    pg=None,
                    day_minute=1200.0,
                    world={"neuro": section},
                )
                self.frames.append(self.hub(doors))

    def seen(self, key: str) -> list[dict[str, Any]]:
        return [e for s in self.part.samples for e in s["seen"] if e["id"] == key]

    def sighted(self, kind: str) -> list[Any]:
        return [s for f in self.frames for s in f.camera.sightings if s.kind == kind]


def test_compiled_mesh_surface_samples_are_bounded_and_transformed():
    """Support samples are real compiled vertices and use the geom's current world pose."""
    model = SimpleNamespace(
        geom_type=[7],
        geom_dataid=[0],
        geom_size=[[0.0, 0.0, 0.0]],
        nmesh=1,
        mesh_vertadr=[0],
        mesh_vertnum=[4],
        mesh_vert=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
    )
    world = nw.NeuroWorld.__new__(nw.NeuroWorld)
    world.model = model
    world._surface_cache = {}
    rotation = (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    data = SimpleNamespace(geom_xpos=[(2.0, 3.0, 4.0)], geom_xmat=[rotation])
    points = world._points(data, nw.Candidate("thing:mesh", "thing", (0,), (0,)))
    assert 1 <= len(points) <= 26
    expected = {(2.0 - v, 3.0 + u, 4.0 + z) for u, v, z in model.mesh_vert}
    assert all(
        tuple(round(x, 8) for x in point)
        in {tuple(round(x, 8) for x in candidate) for candidate in expected}
        for point, _ in points
    )
    assert world._local_surface_points(0) is world._surface_cache[0]


def test_box_support_samples_cover_faces_edges_and_corners():
    world = nw.NeuroWorld.__new__(nw.NeuroWorld)
    world.model = SimpleNamespace(geom_type=[6], geom_size=[[0.2, 0.1, 0.05]])
    world._surface_cache = {}
    points = world._local_surface_points(0)
    assert len(points) == 26
    assert all(
        any(math.isclose(abs(p[k]), world.model.geom_size[0][k]) for k in range(3)) for p in points
    )
    assert {(0.2, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.05)} <= set(points)


def test_sphere_surface_samples_follow_the_primitive_surface():
    world = nw.NeuroWorld.__new__(nw.NeuroWorld)
    world.model = SimpleNamespace(geom_type=[2], geom_size=[[0.1, 0.0, 0.0]])
    world._surface_cache = {}
    points = world._local_surface_points(0)
    assert len(points) <= 26
    assert all(math.isclose(math.sqrt(sum(v * v for v in p)), 0.1, abs_tol=1e-9) for p in points)


def test_ray_sampling_uses_the_actual_first_candidate_surface_hit(monkeypatch):
    import sys

    mujoco = SimpleNamespace(mj_ray=None)
    numpy = SimpleNamespace(array=lambda values, dtype=None: list(values))
    monkeypatch.setitem(sys.modules, "mujoco", mujoco)
    monkeypatch.setitem(sys.modules, "numpy", numpy)
    world = nw.NeuroWorld.__new__(nw.NeuroWorld)
    world.model = SimpleNamespace(geom_bodyid=[7, 7, 7, 7, 7, 7, 7, 7, 8, 8])
    world._ray_hit = [-1]
    world._ray_group = None
    world.cam_body = -1
    candidate = nw.Candidate("thing:test", "thing", (7,), (3,))

    def candidate_ray(_model, _data, _start, _unit, _groups, _n, _exclude, hit):
        hit[0] = 3
        return 0.5

    mujoco.mj_ray = candidate_ray
    got = world._visible_point(None, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), candidate)
    assert got == ((0.0, 0.0, 0.5), 3)
    assert world._clear(None, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), candidate, 3)

    def wall_ray(_model, _data, _start, _unit, _groups, _n, _exclude, hit):
        hit[0] = 9
        return 0.4

    mujoco.mj_ray = wall_ray
    assert world._visible_point(None, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), candidate) is None


def test_adjacent_floor_ray_requires_a_real_floor_geom_hit(monkeypatch):
    import sys

    mujoco = SimpleNamespace(mj_ray=None)
    numpy = SimpleNamespace(array=lambda values, dtype=None: list(values))
    monkeypatch.setitem(sys.modules, "mujoco", mujoco)
    monkeypatch.setitem(sys.modules, "numpy", numpy)
    world = nw.NeuroWorld.__new__(nw.NeuroWorld)
    world.model = SimpleNamespace()
    world._ray_hit = [-1]
    world._ray_group = None
    world.cam_body = -1
    world._floor_geoms = frozenset({1})
    monkeypatch.setattr(nw, "in_frame", lambda *_args: True)
    rays = []

    def ray(_model, _data, _start, unit, _groups, _n, _exclude, hit):
        rays.append(tuple(unit))
        hit[0] = 1
        return 1.00002  # camera at z=1; adjacent ray reaches the flat floor at z≈0

    mujoco.mj_ray = ray
    point = (0.0, 0.0, 0.04)
    cam = (0.0, 0.0, 1.0)
    right, down, forward = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)
    assert world._floor_under(None, point, cam, right, down, forward) is not None
    assert len(rays) == 1
    cm = nw.cm
    raw_x = sum((point[k] - cam[k]) * right[k] for k in range(3)) / sum(
        (point[k] - cam[k]) * forward[k] for k in range(3)
    )
    raw_y = sum((point[k] - cam[k]) * down[k] for k in range(3)) / sum(
        (point[k] - cam[k]) * forward[k] for k in range(3)
    )
    shifted_x = sum(rays[0][k] * right[k] for k in range(3)) / sum(
        rays[0][k] * forward[k] for k in range(3)
    )
    shifted_y = sum(rays[0][k] * down[k] for k in range(3)) / sum(
        rays[0][k] * forward[k] for k in range(3)
    )
    xd0, yd0 = cm.distort(raw_x, raw_y, nw._DIST)
    xd1, yd1 = cm.distort(shifted_x, shifted_y, nw._DIST)
    assert abs(nw._K.fx * (xd1 - xd0) - 2.0) < 1e-5
    assert abs(nw._K.fy * (yd1 - yd0)) < 1e-5

    def wall_ray(_model, _data, _start, _unit, _groups, _n, _exclude, hit):
        hit[0] = 2
        return 1.00002

    mujoco.mj_ray = wall_ray
    assert world._floor_under(None, point, cam, right, down, forward) is None
    monkeypatch.setattr(nw, "in_frame", lambda *_args: False)
    rays.clear()
    assert world._floor_under(None, point, cam, right, down, forward) is None
    assert not rays  # never query geometry outside the delivered image


def test_floor_association_marches_across_candidate_surface_before_floor(monkeypatch):
    import sys

    mujoco = SimpleNamespace(mj_ray=None)
    numpy = SimpleNamespace(array=lambda values, dtype=None: list(values))
    monkeypatch.setitem(sys.modules, "mujoco", mujoco)
    monkeypatch.setitem(sys.modules, "numpy", numpy)
    world = nw.NeuroWorld.__new__(nw.NeuroWorld)
    world.model = SimpleNamespace(geom_bodyid=[0, 7, 8])
    world._ray_hit = [-1]
    world._ray_group = None
    world.cam_body = -1
    world._floor_geoms = frozenset({0})
    monkeypatch.setattr(nw, "in_frame", lambda *_args: True)
    candidate = nw.Candidate("thing:ball", "thing", (7,), (1,))
    cam = (0.0, 0.0, 1.0)
    point = (0.0, 0.0, 0.04)
    right, down, forward = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)
    calls = []

    def ray(_model, _data, _start, unit, _groups, _n, _exclude, hit):
        calls.append(tuple(unit))
        if len(calls) == 1:
            hit[0] = 1  # the +2-pixel ray still intersects the ball
            return 0.98
        hit[0] = 0  # the next adjacent ray reaches the actual floor
        return 1.00002

    mujoco.mj_ray = ray
    floor = world._floor_under(None, point, cam, right, down, forward, candidate)
    assert floor is not None
    assert len(calls) == 2
    cm = nw.cm
    raw0 = (point[0] - cam[0]) / (cam[2] - point[2])
    x0, y0 = cm.distort(raw0, 0.0, nw._DIST)
    x1, y1 = cm.distort(calls[0][0] / -calls[0][2], 0.0, nw._DIST)
    assert abs(nw._K.fx * (x1 - x0) - nw.FLOOR_ASSOCIATION_STEP_PX) < 1e-5
    assert abs(nw._K.fy * (y1 - y0)) < 1e-5

    calls.clear()

    def occluded(_model, _data, _start, _unit, _groups, _n, _exclude, hit):
        calls.append(True)
        hit[0] = 2  # an unrelated wall/object terminates the association
        return 0.97

    mujoco.mj_ray = occluded
    assert world._floor_under(None, point, cam, right, down, forward, candidate) is None
    assert len(calls) == 1


def test_floor_association_extends_carry_past_16px_before_floor(monkeypatch):
    """P1c A2: a tall silhouette whose face spans more than 16 px still resolves."""
    import sys

    mujoco = SimpleNamespace(mj_ray=None)
    numpy = SimpleNamespace(array=lambda values, dtype=None: list(values))
    monkeypatch.setitem(sys.modules, "mujoco", mujoco)
    monkeypatch.setitem(sys.modules, "numpy", numpy)
    world = nw.NeuroWorld.__new__(nw.NeuroWorld)
    world.model = SimpleNamespace(geom_bodyid=[0, 7, 8])
    world._ray_hit = [-1]
    world._ray_group = None
    world.cam_body = -1
    world._floor_geoms = frozenset({0})
    monkeypatch.setattr(nw, "in_frame", lambda *_args: True)
    candidate = nw.Candidate("thing:ball", "thing", (7,), (1,))
    cam = (0.0, 0.0, 1.0)
    point = (0.0, 0.0, 0.04)
    right, down, forward = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)
    calls = []

    def ray(_model, _data, _start, unit, _groups, _n, _exclude, hit):
        calls.append(tuple(unit))
        if len(calls) <= 12:
            hit[0] = 1  # twelve consecutive carry steps ride the tall face
            return 0.98
        hit[0] = 0  # then the adjacent floor appears within tolerance
        return 1.00002

    mujoco.mj_ray = ray
    floor = world._floor_under(None, point, cam, right, down, forward, candidate)
    assert floor is not None
    assert len(calls) == 13


def test_floor_association_carry_budget_stays_bounded(monkeypatch):
    """P1c A2: an endless candidate face still ends unknown, with a bounded ray count."""
    import sys

    mujoco = SimpleNamespace(mj_ray=None)
    numpy = SimpleNamespace(array=lambda values, dtype=None: list(values))
    monkeypatch.setitem(sys.modules, "mujoco", mujoco)
    monkeypatch.setitem(sys.modules, "numpy", numpy)
    world = nw.NeuroWorld.__new__(nw.NeuroWorld)
    world.model = SimpleNamespace(geom_bodyid=[0, 7, 8])
    world._ray_hit = [-1]
    world._ray_group = None
    world.cam_body = -1
    world._floor_geoms = frozenset({0})
    monkeypatch.setattr(nw, "in_frame", lambda *_args: True)
    candidate = nw.Candidate("thing:ball", "thing", (7,), (1,))
    cam = (0.0, 0.0, 1.0)
    point = (0.0, 0.0, 0.04)
    right, down, forward = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)
    calls = []

    def ray(_model, _data, _start, unit, _groups, _n, _exclude, hit):
        calls.append(tuple(unit))
        hit[0] = 1
        return 0.98

    mujoco.mj_ray = ray
    assert world._floor_under(None, point, cam, right, down, forward, candidate) is None
    assert len(calls) == nw.FLOOR_ASSOCIATION_CARRY_MAX_PX // nw.FLOOR_ASSOCIATION_STEP_PX


# ── the view geometry is the camera's ──────────────────────────────────────────


def test_the_view_geometry_matches_a_render_of_the_head_camera():
    home = Home()
    home.put("ball_felt", 0.75, 0.3)
    home.put("tissue_box", 0.9, 0.0)
    home.run(0.3, senses=False)
    import numpy as np

    from duckmind.world import camera_model as cm
    from duckmind.world import homelayout, worldkit

    spec = homelayout.compose(home.w.layout, "allcollisions", mjcf_dir=_mjcf())
    worldkit.build().edit_spec(spec, home.w.layout)
    homelayout.fix_keyframes(spec, home.w.layout)
    eye = cm.load_spec(spec, quality="720p30")
    cm.set_state(eye, home.d.qpos.copy(), mocap_pos=home.d.mocap_pos, mocap_quat=home.d.mocap_quat)
    try:
        renderer = cm.HeadCameraRenderer(eye)
    except Exception as error:  # no offscreen GL here
        pytest.skip(f"no offscreen render: {error}")
    width, height = 640, 360
    image = renderer.render(width, height, segmentation=True)
    seg = image[:, :, 0].astype(int) + image[:, :, 1].astype(int) * 256
    names = {}
    scene = renderer._scene
    for i in range(scene.ngeom):
        g = scene.geoms[i]
        if int(g.objtype) == int(home.mj.mjtObj.mjOBJ_GEOM):
            names[i + 1] = home.mj.mj_id2name(eye.model, home.mj.mjtObj.mjOBJ_GEOM, int(g.objid))
    cam, (right, down, forward) = home.part.camera(home.d)
    for key in ("ball_felt", "tissue_box"):
        ids = [sid for sid, name in names.items() if name == f"thing_{key}_0"]
        ys, xs = np.nonzero(np.isin(seg, ids))
        assert len(xs) > 50, f"{key} is not in the render"
        centre = home.d.xipos[home.w.kit.props.things[key].body]
        rel = centre - np.array(cam)
        x, y, z = float(rel @ right), float(rel @ down), float(rel @ forward)
        assert nw.in_frame(x, y, z)
        xd, yd = cm.distort(x / z, y / z)
        u = (cm.FAMILY.cx + cm.FAMILY.fx * xd) * width / cm.FAMILY.width
        v = (cm.FAMILY.cy + cm.FAMILY.fy * yd) * height / cm.FAMILY.height
        assert abs(u - xs.mean()) < 12 and abs(v - ys.mean()) < 12, (key, u, v)
    listed = {e["id"] for e in home.part.samples[-1]["seen"]}
    assert {"thing:ball_felt", "thing:tissue_box"} <= listed


# ── no truth leaks: behind, behind a wall, in the dark ─────────────────────────


def test_a_thing_behind_the_duck_is_never_in_view_and_one_ahead_is():
    home = Home()
    home.put("ball_felt", 0.7, 0.3)  # ahead
    home.put("fin_duck", -0.8, 0.3)  # behind
    home.run(3.0)
    assert home.seen("thing:ball_felt") and all(e["vis"] > 0 for e in home.seen("thing:ball_felt"))
    assert home.seen("thing:fin_duck") == []
    things = home.sighted("thing")
    assert any(abs(s.bearing) < 0.15 and s.range_m and s.range_m < 1.0 for s in things)
    # nothing from behind (the frame spans about −30°…+16° around the head's axis)
    everything = [s for f in home.frames for s in f.camera.sightings]
    assert all(abs(s.bearing) < 0.8 for s in everything)


def _walled() -> dict[str, Any]:
    from duckmind.world import homelayout

    raw = homelayout.load("default_home").to_json()
    raw.pop("variants", None)
    raw["id"] = "default_home_wall_test"
    raw["elements"] = [e for e in raw["elements"] if e.get("kind") != "furniture"] + [
        {
            "id": "wall_test",
            "kind": "wall",
            "zh": "隔間牆",
            "shape": "box",
            "pos_mm": [700, 300, 0],
            "size_mm": [80, 1400, 1200],
            "colour": "#f3efe7",
        }
    ]
    return raw


def test_a_thing_behind_a_wall_is_in_the_frame_but_hidden_and_never_sighted():
    home = Home(_walled())
    home.put("ball_felt", 1.3, 0.3)
    home.run(3.0)
    behind = home.seen("thing:ball_felt")
    assert behind and all(e["fov"] > 0 and e["vis"] == 0 for e in behind)
    assert all(set(e) == {"id", "kind", "cls", "n", "fov", "vis"} for e in behind)
    assert home.sighted("thing") == []
    home.put("ball_felt", 0.45, 0.3)
    home.run(3.0)
    assert home.sighted("thing"), "in front of the wall it is found"


def test_a_clap_behind_the_wall_is_quieter_than_in_front():
    home = Home(_walled())
    with home.w.world.lock:
        head = home.part.head(home.d)
    near = home.part.add_sound("clap", (head[0] + 0.5, head[1], 1.0), t0=0.0, seconds=0.04)
    far = home.part.add_sound("clap", (head[0] + 1.4, head[1], 1.0), t0=0.0, seconds=0.04)
    with home.w.world.lock:
        home.part._levels(home.d, float(home.d.time))
    assert far.wall and not near.wall
    assert near.db - far.db > nw.WALL_DB * 0.6


def test_in_the_dark_nothing_is_seen_and_the_light_comes_back_with_the_lamps():
    home = Home()
    home.put("ball_felt", 0.7, 0.3)
    # (neuro:fix: the camera looks through duck-eye's governor, ~2 looks a second, and a still
    # thing is found in a share of the looks — give it a few seconds, as the duck has)
    home.run(5.0)
    assert home.sighted("thing")
    answer = home.part.stimulate({"stim": "lights_off"})
    assert answer["lights_on"] is False and answer["lux_at_duck"] < 2.5
    home.frames.clear()
    home.run(3.0)
    late = home.frames[len(home.frames) // 2 :]
    assert {f.camera.light for f in late} <= {"black", "dark"}
    assert [s for f in late for s in f.camera.sightings if s.track != 0] == []
    home.part.stimulate({"stim": "lights_on"})
    home.frames.clear()
    home.run(5.0)
    assert home.frames[-1].camera.light in ("bright", "dim")
    assert home.sighted("thing")


# ── sound and touch ─────────────────────────────────────────────────────────────


def test_a_clap_is_louder_near_and_the_duck_hears_no_direction():
    home = Home()
    home.part.stimulate({"stim": "clap"})
    home.run(1.5)
    heard = [f for f in home.frames if f.sound.onset > 0]
    assert heard and all(f.sound.bearing is None for f in home.frames)
    with home.w.world.lock:
        head = home.part.head(home.d)
    assert nw.level_at(82.0, 0.5) > nw.level_at(82.0, 2.5) > nw.level_at(82.0, 6.0)
    sounds = home.part.snapshot(home.w.world)["sounds"]
    clap = [s for s in sounds if s["kind"] == "clap"][0]
    distance = math.dist([v / 1000 for v in clap["pos_mm"]], head)
    assert clap["db"] == pytest.approx(nw.level_at(clap["db1m"], distance), abs=0.2)


def test_only_a_real_contact_is_a_touch():
    home = Home()
    # the household's own "touch" hovers over the head without meeting it: nothing is felt
    with home.w.world.lock:
        home.w.kit.hand.command(home.w.world, {"do": "touch", "where": "head", "seconds": 2.0})
    home.run(4.0)
    assert list(home.part.ended) == [] and home.part.contacts == {}
    assert all(f.touch.site is None for f in home.frames)
    home.part.stimulate({"stim": "pet_head"})
    home.run(4.5)
    sites = {c.site for c in home.part.ended}
    assert "head" in sites
    felt = [f.touch.site for f in home.frames if f.touch.site]
    assert felt and set(felt) <= {"head", "beak", "back", "left", "right"}


def test_a_poke_is_felt_on_its_side():
    home = Home()
    home.part.stimulate({"stim": "poke", "side": "left"})
    home.run(1.5)
    assert {c.site for c in home.part.ended} & {"left", "back"}


def test_every_stimulus_is_a_world_event():
    home = Home()
    home.run(0.3, senses=False)
    carried = ("pick_up", "set_down")  # the household's hand: tested below, on a free duck
    # a scripted line is part of the ``say`` stimulus's contract (LV0): with no ``utterance`` (and no
    # ``live: true``) the world refuses it, which is the contract, not a fault — so the test says one
    needs = {"say": {"utterance": "hello"}}
    for stim in sorted(set(nw.STIMULI) - set(carried)):
        answer = home.part.stimulate({"stim": stim, **needs.get(stim, {})})
        assert answer["stim"] == stim and answer.get("note_zh"), stim
        home.run(0.6, senses=False)
    assert home.part.errors == []
    section = home.part.snapshot(home.w.world)
    assert section["v"] == nw.VERSION and len(section["stim"]) == 8


def test_picked_up_the_hand_is_busy_and_set_down_it_is_free_again():
    from duckmind.world import worldkit

    w = worldkit.Headless("default_home", mjcf_dir=_mjcf(), keyframe="SIT")
    part = nw.attach(w, seed=3)
    w.run(0.5)
    assert part.stimulate({"stim": "pick_up"})["hand"]["ok"]
    w.run(3.0)
    assert w.kit.hand.holding
    with pytest.raises(ValueError):
        part.stimulate({"stim": "pet_head"})  # the hand is holding the duck
    answer = part.stimulate({"stim": "set_down"})
    assert answer["hand"]["ok"]
    w.run(5.0)
    assert not w.kit.hand.holding
    assert part.stimulate({"stim": "pet_head"})["stim"] == "pet_head"


def test_the_tof_door_goes_through_duck_body_depth_summary():
    from duckmind.duckbody.robotd_client import RobotState
    from duckmind.world.rig import Rig

    rig = Rig(
        "default_home",
        tof=True,
        camera=False,
        mjcf_dir=_mjcf(),
        keyframe="STAND",
        x=2.6,
        y=0.3,
        yaw=0.0,
    )
    hub = ns.SenseHub(seed=1)
    last = None
    frame = None
    for _ in range(40):
        doors = rig.tick()
        state = RobotState.from_wire(doors.state)
        last = doors.tof or last
        frame = hub(
            SimpleNamespace(
                t_ms=doors.t_ns // 1_000_000,
                state=state,
                link=None,
                events=(),
                depth=None,
                tof=last,
                pg=None,
                day_minute=600.0,
                world=None,
            )
        )
    assert frame is not None and frame.tof.present and frame.sources["tof"] == "real"
    assert frame.tof.ahead_mm is not None and 200 < frame.tof.ahead_mm < 500  # the wall, 0.4 m


# ── the floor line (integration 2026-09-28: the duck sees the walls it faces) ─────


def test_the_floor_line_sees_the_wall_ahead_and_a_thing_on_the_floor():
    home = Home(_walled())  # a wall 0.66 m ahead of the trunk (face at x = 0.66)
    home.run(1.0)
    line = home.part.samples[-1]["floor"]
    assert line is not None and len(line["sectors"]) == 5
    mids = line["sectors"][1:4]
    assert all(s["end"] == "wall" and abs(s["free"] - 0.66) < 0.06 for s in mids), mids
    bearings = [s["b"] for s in line["sectors"]]
    assert bearings == sorted(bearings, reverse=True)  # left to right across the picture
    seen = [f.camera.floor for f in home.frames if f.camera.floor]
    assert seen and any(abs(s.free_m - 0.66) < 0.15 and s.end == "wall" for s in seen[-1])
    home.put("ball_felt", 0.35, 0.3)  # a ball on the way: a thing ends the clear floor first
    home.run(1.0)
    mid = home.part.samples[-1]["floor"]["sectors"][2]
    assert mid["end"] == "thing" and mid["free"] < 0.35


def test_in_the_dark_there_is_no_floor_line():
    home = Home(_walled())
    home.part.stimulate({"stim": "lights_off"})
    home.run(3.0)
    late = home.frames[len(home.frames) // 2 :]
    times = {f.camera.floor_t_ms for f in late}
    assert len(times) <= 1  # no new look in the dark (only what it saw before, if anything)


def test_a_family_member_never_steps_in_on_top_of_the_duck_at_the_door():
    """neuro:fix (live): the snuggler resting by the entry was knocked over twice by a family
    member appearing at the entry spot on top of it — then they come from the sofa instead."""
    home = Home(x=2.6, y=-2.4, yaw=2.3)
    home.part.stimulate({"stim": "walk_by"})
    walkers = home.part.people.walkers
    here = [w for w in walkers.values() if w.here]
    assert here
    assert all(math.dist(w.pos, (2.6, -2.4)) > 0.8 for w in here)
    far = Home(x=-1.0, y=0.5)
    far.part.stimulate({"stim": "sit_near"})
    came = [w for w in far.part.people.walkers.values() if w.here]
    assert came and all(math.dist(w.pos, (2.5, -2.5)) < 0.1 for w in came)  # the door, as ever


def test_a_sit_near_person_walks_to_a_place_and_stays_there_when_asked():
    """The fair follow scene (2026-10-24): ``sit_near``'s person left after 20 s, so a follow gave
    up when they walked out of the room. With ``to`` they walk there after ``stay_s`` and stand
    there — the person stays, then walks a path (the world's own ``x_people`` stand op)."""
    home = Home(x=0.0, y=0.3, yaw=0.0)  # looking along +x
    answer = home.part.stimulate({"stim": "sit_near", "to": [1500, 300], "stay_s": 2.0})
    who = answer["who"]
    walker = home.part.people.walkers[who]
    for _ in range(24):
        home.run(0.5)
        if walker.state == "standing":
            break
    assert walker.state == "standing" and walker.here
    for _ in range(60):
        home.run(0.5)
        if math.dist(walker.pos, (1.5, 0.3)) < 0.2:
            break
    assert math.dist(walker.pos, (1.5, 0.3)) < 0.2, walker.pos
    assert walker.state == "standing" and walker.here  # (they stay: no leave)
    home.run(4.0)
    assert walker.here and math.dist(walker.pos, (1.5, 0.3)) < 0.2


def _say(
    home: Home, who: str, place: tuple[int, int] | None, line: str = "hello"
) -> dict[str, Any]:
    request: dict[str, Any] = {"stim": "say", "utterance": line, "who": who}
    if place is not None:
        request["where_mm"] = list(place)
    return home.part.stimulate(request)


def test_a_family_member_who_speaks_from_a_place_stands_there_and_the_duck_can_see_them():
    """Live voice (2026-10-04): the voice alone never put a person in front of the duck's eyes, so
    「這是媽媽」 was refused ``no_person_in_view``. The speaker's body stands at the place (the
    family's own bodies, facing the duck) and the duck's own camera sees them."""
    home = Home(x=0.0, y=0.3, yaw=0.0)  # looking along +x
    answer = _say(home, "partner", (1000, 300))
    walker = home.part.people.walkers["partner"]
    assert walker.here and walker.state == "standing"
    assert math.dist(walker.pos, (1.0, 0.3)) < 0.2
    voice = [v / 1000.0 for v in answer["where_mm"][:2]]  # (the voice comes from the body)
    assert math.dist((walker.pos[0], walker.pos[1]), voice) < 0.01
    assert abs(math.remainder(walker.heading - math.pi, 2 * math.pi)) < 0.5  # (turned to the duck)
    home.run(1.5)
    assert home.sighted("person")  # its own eyes report a person


def test_a_place_outside_the_home_puts_the_speaker_where_there_is_room():
    """Live voice (2026-10-04): the duck by the north wall facing it, the page's "front" (1 m ahead) is
    outside the home: the speaker stands in the room near it, not nowhere."""
    from duckmind.world import people

    home = Home(x=-1.4, y=2.2, yaw=math.pi / 2)
    answer = _say(home, "owner", (-1400, 3200))
    walker = home.part.people.walkers["owner"]
    assert walker.here and walker.state == "standing"
    floor = home.part.people.floor
    assert floor.free(*walker.pos)  # (inside the room, clear of the duck)
    assert math.dist(walker.pos, (-1.4, 2.2)) >= people.STAND_KEEP_M - 1e-6
    voice = [v / 1000.0 for v in answer["where_mm"][:2]]
    assert math.dist((walker.pos[0], walker.pos[1]), voice) < 0.01


def _stand_a_sitter_near(home: Home) -> Any:
    """sit_near, run until the family member has come and stands still in front of the duck; the
    camera frames up to then are dropped (a walking person is the moving case)."""
    who = home.part.stimulate({"stim": "sit_near"})["who"]
    walker = home.part.people.walkers[who]
    for _ in range(24):
        home.run(0.5)
        if walker.state == "standing":
            break
    assert walker.state == "standing" and not walker.sway
    home.frames.clear()
    return walker


@pytest.mark.parametrize("yaw", [0.0, math.pi / 2], ids=["floor_ranged", "feet_below_the_frame"])
def test_a_family_member_standing_still_by_the_duck_is_reported_as_a_person(yaw):
    """Measured on the smoke server (stimulus sit_near): a person who stood still 0.84 m from the
    duck was never reported, so come / follow / look at me were refused ``no_one_seen`` 8/8. The
    still person is now called a person within 3 s, with the floor line's range or (the feet
    below the frame edge at this distance, as facing +y in this home) without one."""
    home = Home(x=0.0, y=0.3, yaw=yaw)
    walker = _stand_a_sitter_near(home)
    home.run(3.0)
    sights = home.sighted("person")
    assert sights, f"a person {math.dist(walker.pos, (0.0, 0.3)):.2f} m away was never reported"
    assert abs(sights[0].bearing) < 0.2
    assert all(s.source == "model" for s in sights)


@pytest.mark.parametrize(
    "where", [(0.5, 1.5), (0.5, -1.0), (-1.5, 0.3)], ids=["left", "right", "behind"]
)
def test_a_family_member_standing_outside_the_frame_is_not_reported(where):
    """The frame reaches about 26° to the right and 13° to the left of the heading (the family
    lens's principal point is off centre; the render test above checks it), so a person well
    outside it, or behind the duck, is not reported."""
    home = Home(x=0.0, y=0.3, yaw=0.0)
    answer = home.part.people.command(
        home.w.world,
        {"do": "stand", "who": "owner", "at": [int(where[0] * 1000), int(where[1] * 1000)]},
    )
    assert answer["ok"]
    walker = home.part.people.walkers["owner"]
    walker.sway = False  # (stands perfectly still: the case that was never reported)
    home.run(0.5, senses=False)
    home.frames.clear()
    home.run(4.0)
    assert walker.state == "standing"
    assert not home.sighted("person")
    assert not [e for e in home.seen("person:owner") if e.get("vis")]


def test_a_speaker_with_no_place_is_not_drawn_and_a_second_one_never_stands_on_the_first():
    home = Home(x=0.0, y=0.3, yaw=0.0)
    _say(home, "partner", None)  # (the world's own place: the family is home, just not drawn)
    assert not home.part.people.walkers["partner"].here
    _say(home, "owner", (1000, 300))
    _say(home, "partner", (1000, 300))
    walkers = home.part.people.walkers
    assert walkers["owner"].here and walkers["partner"].here
    assert math.dist(walkers["owner"].pos, walkers["partner"].pos) >= 0.5
    # a voice with no one named stands nobody up
    before = {k for k, w in walkers.items() if w.here}
    home.part.stimulate({"stim": "say", "utterance": "hello", "where_mm": [1000, -600]})
    assert {k for k, w in walkers.items() if w.here} == before


def test_a_speaker_who_says_nothing_for_a_while_walks_out_and_one_who_keeps_talking_stays():
    home = Home(x=0.0, y=0.3, yaw=0.0)
    _say(home, "partner", (1000, 300))
    _say(home, "child", (900, -300))
    walkers = home.part.people.walkers
    home.run(nw.STAND_AFTER_S - 5.0, senses=False)
    _say(home, "child", (900, -300))  # (the child speaks again: its time starts over)
    home.run(10.0, senses=False)
    assert walkers["partner"].state in ("leaving", "away") or not walkers["partner"].here
    assert walkers["child"].here and walkers["child"].state != "leaving"
    home.run(30.0, senses=False)
    assert not walkers["partner"].here


def test_integrated_world_compiled_model_sample_reaches_camera_without_swallowed_error():
    """A compiled official model catches adapter errors hidden by NeuroWorld.after_step."""
    from duckmind.world import neuro_integration as integrated
    from duckmind.world import neuro_sun as sun

    home = Home(seed=21)
    part = integrated.IntegratedWorld(seed=21)
    part.sunfield = sun.SunField(sun.SunModel.from_layout(home.w.layout, weather="sunny"))
    part._clock = lambda: home.clock["when"]
    part.bind(
        hand=home.w.kit.hand,
        people=home.w.kit.people,
        props=home.w.kit.props,
        nest=home.w.kit.nest,
        lights=home.part.lights,
        switch_lamps=home.part.switch_lamps,
    )
    part.ready(home.w.world, None)

    frame = part.sample(home.d)
    assert frame["cam"] and "lux_meter" in frame["cam"]
    assert "roof_truth" in frame

    # Exercise the production error-catching hook at t=0 without advancing physics.
    part._next_sample = 0.0
    part.after_step(home.w.world)
    assert len(part.samples) == 1
    assert part.samples[0]["cam"]["lux_meter"] >= 0
    assert part.errors == []
