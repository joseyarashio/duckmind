"""duckmind.world.simhome: the world server seam — the extension registry, the hooked step,
the x_place maths, the pins, the keyframe trap and the flat fallback. Fast; the MuJoCo cases
run the official body server in process (no robotd) and skip without MuJoCo or the official
checkout (the render check runs them); one opt-in live smoke (DUCKMIND_LIVE_SMOKE=1)."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import math
import os
import socketserver
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import pytest

from duckmind import bodyhost
from duckmind.bodyhost import HostClient, source_digests
from duckmind.livebody import LiveBody, LivePaths, live_home
from duckmind.world import simhome
from duckmind.world.simhome import (
    HOME_REVIEWED_SHA256,
    REVIEWED_SHA256,
    SIMHOME_UPSTREAM_SHA256,
    HomeInfo,
    Registry,
    SimHome,
    Spawn,
    check_home_files,
    extension_specs,
    footprint_points,
    hooked_step,
    official_files,
    parse_place,
    parse_spawn,
    plan_home,
    pose_mm,
    quat_tilt_mrad,
    quat_yaw_mrad,
    resolve_spawn,
    rest_shift,
    strip_keyframes,
    wrap_mrad,
    yaw_quat,
)

MJCF = os.environ.get("DUCKMIND_MICRODUCK_MJCF")


def _mjcf() -> Path:
    if not MJCF or not Path(MJCF, "scene_allcollisions.xml").is_file():
        pytest.skip("DUCKMIND_MICRODUCK_MJCF (official microduck MJCF folder) is not set")
    return Path(MJCF)


def _official_body_server() -> Path:
    path = _mjcf().parents[1] / "sim" / "body_server.py"
    if not path.is_file():
        pytest.skip("no official body_server.py next to the MJCF folder")
    return path


# ── the registry ────────────────────────────────────────────────────────────────


def test_the_registry_keeps_order_and_names_what_registered():
    registry = Registry()
    calls: list[str] = []

    def first(world: Any) -> None:
        calls.append("first")

    registry.before_step(first)
    registry.before_step(lambda world: calls.append("second"), name="hand.lift")
    registry.after_step(first)
    registry.edit_spec(lambda spec, home: None, name="nest.pad")
    registry.on_ready(lambda world, body: None)
    registry.on_stop(lambda: None)
    registry.op("x_hand", lambda body, request: {})
    registry.op("pg", lambda body, request: {"contact": False})  # a sensor op
    registry.op("tof", lambda body, request: {}, override=True)
    registry.snapshot("hand", lambda world: {"state": "idle"})
    registry.snapshot("toys_seen", lambda world: [])
    view = registry.view()
    assert view["hooks"]["before_step"] == ["simhome.first", "hand.lift"]
    assert view["hooks"]["after_step"] == ["simhome.first"]
    assert view["hooks"]["edit_spec"] == ["nest.pad"]
    assert view["registered_ops"] == ["pg", "tof", "x_hand"]
    assert view["sections"] == ["hand", "toys_seen"]
    for entry in registry.before:
        entry.fn(None)
    assert calls == ["first", "second"]


@pytest.mark.parametrize(
    "name, override",
    [
        ("read", False),  # official ops stay official
        ("write", True),  # ... even when asked (lockstep lives there)
        ("slow", True),  # owner decision E1: the world battery never reaches robotd
        ("tof", False),  # replacing the ToF must be said out loud
        ("x_place", False),  # simhome's own
        ("x_world", False),
        ("X_Bad", False),  # not a plain lower-case name
        ("x hand", False),
        ("x_new", True),  # nothing official to override
    ],
)
def test_what_an_extension_may_not_register_is_refused(name, override):
    with pytest.raises(ValueError):
        Registry().op(name, lambda body, request: {}, override=override)


def test_an_op_and_a_section_belong_to_one_module_only():
    registry = Registry()
    registry.op("x_hand", lambda body, request: {})
    with pytest.raises(ValueError):
        registry.op("x_hand", lambda body, request: {})
    registry.snapshot("people", lambda world: [])
    with pytest.raises(ValueError):
        registry.snapshot("people", lambda world: [])
    for own in ("duck", "props", "mocap", "home", "sim_time", "Bad Name"):
        with pytest.raises(ValueError):
            registry.snapshot(own, lambda world: None)


def test_extensions_are_loaded_by_module_name_and_register_themselves(tmp_path, monkeypatch):
    (tmp_path / "dm_ext_hand.py").write_text(
        "def register(registry):\n"
        "    registry.before_step(lambda world: None, name='hand.drive')\n"
        "    registry.op('x_hand', lambda body, request: {'hand': 'idle'})\n"
        "def other(registry):\n"
        "    registry.snapshot('hand', lambda world: None)\n"
    )
    (tmp_path / "dm_ext_empty.py").write_text("X = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    registry = Registry()
    registry.load(["dm_ext_hand", " ", "dm_ext_hand:other"])
    assert registry.extensions == ["dm_ext_hand", "dm_ext_hand:other"]
    assert registry.ops["x_hand"].name == "dm_ext_hand:x_hand"
    assert registry.sections["hand"].name == "dm_ext_hand:hand"
    assert registry.owner == "simhome"
    with pytest.raises(ValueError):
        registry.load(["dm_ext_empty"])
    assert extension_specs(None, "a, b:c,,") == ["a", "b:c"]
    assert extension_specs(["x", "y,z"], "ignored") == ["x", "y", "z"]
    assert extension_specs(None, None) == []


# ── the hooked step (a stand-in world) ──────────────────────────────────────────


class _FakeMujoco:
    """Just enough of mujoco for the step copy: mj_step moves time 5 ms."""

    def __init__(self) -> None:
        self.steps = 0

    def mj_step(self, model: Any, data: Any) -> None:
        data.time += model.opt.timestep
        self.steps += 1


class _FakeBody:
    def __init__(self, world: Any, released: bool) -> None:
        self.world = world
        self.released = released
        self.restored = 0

    def restore(self) -> None:
        self.restored += 1


def _fake_world() -> Any:
    world = types.SimpleNamespace(
        model=types.SimpleNamespace(opt=types.SimpleNamespace(timestep=0.005)),
        data=types.SimpleNamespace(time=0.0),
        lock=threading.Lock(),
        bodies=[],
    )
    return world


def _host(tmp_path: Path, registry: Registry, lockstep: bool = True) -> SimHome:
    return SimHome(
        collision="allcollisions", lockstep=lockstep, mjcf_dir=tmp_path, registry=registry
    )


def test_hooks_run_around_every_physics_step_on_physics_time(tmp_path):
    registry = Registry()
    seen: list[tuple[str, float, bool]] = []
    registry.before_step(
        lambda world: seen.append(("before", world.data.time, world.lock.locked()))
    )
    registry.after_step(lambda world: seen.append(("after", world.data.time, world.lock.locked())))
    host = _host(tmp_path, registry)
    mujoco = _FakeMujoco()
    step = hooked_step(mujoco, host)
    world = _fake_world()
    held, free = _FakeBody(world, released=False), _FakeBody(world, released=True)
    world.bodies = [held, free]
    step(world, 4)
    assert mujoco.steps == 4 and world.data.time == pytest.approx(0.020)
    assert [kind for kind, _, _ in seen] == ["before", "after"] * 4
    assert [round(t, 3) for _, t, _ in seen] == [0.0, 0.005, 0.005, 0.01, 0.01, 0.015, 0.015, 0.02]
    assert all(locked for _, _, locked in seen)  # inside the world's lock
    assert (held.restored, free.restored) == (4, 0)  # upstream's hold, per step


def test_the_step_copy_is_the_pinned_official_step_plus_hooks():
    """The copy has the official World.step's statements (pinned) plus only the two hooks."""
    copy = inspect.getsource(hooked_step)
    for line in (
        "with self.lock:",
        "for _ in range(times):",
        "mujoco.mj_step(self.model, self.data)",
        "for body in self.bodies:",
        "if not body.released:",
        "body.restore()",
    ):
        assert line in copy
    assert SIMHOME_UPSTREAM_SHA256["World.step"] == bodyhost.UPSTREAM_SHA256["World.step"]


def test_a_hook_that_raises_is_switched_off_and_the_monitor_says_so(tmp_path):
    registry = Registry()
    ticks: list[float] = []

    def broken(world: Any) -> None:
        raise RuntimeError("no hand here")

    registry.before_step(broken, name="hand.drive")
    registry.after_step(lambda world: ticks.append(world.data.time))
    host = _host(tmp_path, registry)
    step = hooked_step(_FakeMujoco(), host)
    world = _fake_world()
    step(world, 4)
    step(world, 4)
    assert len(ticks) == 8  # the world and the other hooks went on
    assert registry.before[0].disabled == "RuntimeError: no hand here"
    assert host.failures == ["世界擴充 hand.drive 出錯，已經停用：RuntimeError: no hand here"]
    report = host.report()
    assert report["trust"] == "suspect" and report["why_zh"][-1] == host.failures[0]


def _fake_official(tmp_path: Path) -> types.ModuleType:
    """A stand-in for the official module, shaped like it (World, Body, Handler, run, ...)."""
    module = types.ModuleType("fake_body_server")
    module.__file__ = str(tmp_path / "body_server.py")
    module.mujoco = _FakeMujoco()  # type: ignore[attr-defined]

    class World:
        def __init__(self, scene: Path, count: int = 1) -> None:
            self.model = module.build_world(scene, count)  # type: ignore[attr-defined]
            self.data = types.SimpleNamespace(time=0.0)
            self.lock = threading.Lock()
            self.bodies: list[Any] = []

        def step(self, times: int = 1) -> None:
            raise AssertionError("the official step is replaced by the hooked copy")

    class Body:
        def __init__(self, world: World, index: int) -> None:
            self.world, self.index, self.released = world, index, False
            self.placed: list[Any] = []

        def place(self, pose: Any, trunk_z: float, offset_y: float) -> None:
            self.placed.append((pose, trunk_z, offset_y))

        def restore(self) -> None:
            pass

        def _apply_torque(self) -> None:
            pass

    class Handler:
        def dispatch(self, body: Any, request: dict[str, Any]) -> dict[str, Any]:
            op = request.get("op")
            if op == "read":
                return {"sim_time": body.world.data.time}
            if op == "write":
                return {}
            if op == "tof":
                return {"official": True}
            raise ValueError(f"unknown op {op!r}")

    def build_world(scene: Path, count: int) -> Any:
        raise AssertionError("the official build_world is replaced")

    def run(world: Any, headless: bool) -> None:
        raise AssertionError("the official real-time loop")

    module.World, module.Body, module.Handler = World, Body, Handler  # type: ignore[attr-defined]
    module.build_world, module.run = build_world, run  # type: ignore[attr-defined]
    return module


def _installed(tmp_path: Path, registry: Registry) -> tuple[SimHome, Any, Any, Any]:
    """A SimHome on the stand-in module, spawned (spawn and settle stubbed: no MuJoCo)."""
    fake = _fake_official(tmp_path)
    host = _host(tmp_path, registry)
    model = types.SimpleNamespace(opt=types.SimpleNamespace(timestep=0.005), nkey=0)
    host.model, host.spec = model, types.SimpleNamespace(compile=lambda: model)
    host.home = HomeInfo(
        asked=None,
        name="flat",
        source="flat",
        base_scene=tmp_path / "scene_allcollisions.xml",
        official_files=[],
        spawn=Spawn(),
    )
    spawned: list[Any] = []
    host.spawn = lambda body: spawned.append(body) or {}  # type: ignore[method-assign]
    host.install(fake)
    world = fake.World(tmp_path / "scene_allcollisions.xml", 1)
    body = fake.Body(world, 0)
    body.place(None, 0.07, 0.0)
    world.bodies.append(body)
    assert spawned == [body] and body.placed == [(None, 0.07, 0.0)]
    return host, fake, world, body


def test_installed_each_robotd_write_is_one_control_period_with_the_hooks(tmp_path):
    registry = Registry()
    substeps: list[float] = []
    registry.after_step(lambda world: substeps.append(world.data.time))
    registry.op("x_hand", lambda body, request: {"asked": request.get("to")})
    registry.op("tof", lambda body, request: {"world_tof": True}, override=True)
    host, fake, world, body = _installed(tmp_path, registry)
    assert fake.run is bodyhost.idle_run
    handler = fake.Handler()
    for _ in range(5):
        assert handler.dispatch(body, {"op": "write", "targets": [0.0] * 15}) == {}
    assert len(substeps) == 20 and world.data.time == pytest.approx(0.1)
    assert handler.dispatch(body, {"op": "read"}) == {"sim_time": pytest.approx(0.1)}
    assert handler.dispatch(body, {"op": "x_hand", "to": "nest"}) == {"asked": "nest"}
    assert handler.dispatch(body, {"op": "tof"}) == {"world_tof": True}
    report = handler.dispatch(body, {"op": "x_monitor"})
    assert (report["writes"], report["advances"], report["home"]) == (5, 5, "flat")
    with pytest.raises(ValueError):
        handler.dispatch(body, {"op": "x_nothing"})
    assert world.data.time == pytest.approx(0.1)  # nothing but writes moved the world
    with pytest.raises(SystemExit):
        fake.build_world(tmp_path, 2)  # a home holds one duck
    assert sorted(host.ops) == ["tof", "x_hand", "x_info", "x_monitor", "x_place", "x_world"]


# ── the maths of a placement ────────────────────────────────────────────────────


@pytest.mark.parametrize("yaw", [0, 1571, -1571, 3141, -3000, 250])
def test_a_yaw_goes_to_a_quaternion_and_back(yaw):
    quat = yaw_quat(yaw)
    assert sum(v * v for v in quat) == pytest.approx(1.0)
    assert quat_yaw_mrad(quat) == yaw
    assert quat_tilt_mrad(quat) == 0


def test_tilt_is_the_angle_from_upright():
    half = math.radians(30) / 2
    rolled = [math.cos(half), math.sin(half), 0.0, 0.0]
    assert quat_tilt_mrad(rolled) == round(math.radians(30) * 1000)
    assert quat_tilt_mrad([0.0, 1.0, 0.0, 0.0]) == round(math.pi * 1000)  # upside down
    assert wrap_mrad(6283 + 100) == 100 and wrap_mrad(3200) == 3200 - 6283
    assert wrap_mrad(-3141) == -3141 and wrap_mrad(-3142) == 3141  # [-π, π)


def test_an_x_place_request_is_checked_field_by_field():
    args = parse_place({"x_mm": 450, "y_mm": -20, "yaw_mrad": 7000, "keyframe": "sit"})
    assert (args.x_mm, args.y_mm, args.keyframe, args.force) == (450, -20, "SIT", False)
    assert args.yaw_mrad == wrap_mrad(7000) and args.probe_from_mm == 250
    assert parse_place({"x_mm": 1.0, "y_mm": 2, "force": True}).force is True
    for bad in (
        {"y_mm": 0},
        {"x_mm": 0.5, "y_mm": 0},
        {"x_mm": True, "y_mm": 0},
        {"x_mm": "450", "y_mm": 0},
        {"x_mm": 60_000, "y_mm": 0},
        {"x_mm": 0, "y_mm": 0, "keyframe": "LIE"},
        {"x_mm": 0, "y_mm": 0, "force": "yes"},
        {"x_mm": 0, "y_mm": 0, "probe_from_mm": 5},
    ):
        with pytest.raises(ValueError):
            parse_place(bad)


def test_the_duck_is_lowered_or_raised_to_rest_just_above_its_surface():
    assert rest_shift(-0.0044) == pytest.approx(0.0054)  # the SIT keyframe is 4.4 mm deep
    assert rest_shift(0.02) == pytest.approx(-0.019)  # a hovering duck comes down
    assert rest_shift(0.001) == pytest.approx(0.0)
    points = footprint_points([(0.1, 0.0), (-0.05, 0.04), (0.02, -0.03)], (0.0, 0.0))
    assert points[0] == (0.0, 0.0)
    assert set(points[1:]) == {(-0.05, -0.03), (-0.05, 0.04), (0.1, -0.03), (0.1, 0.04)}
    assert footprint_points([], (1.0, 2.0)) == [(1.0, 2.0)]


# ── where the duck starts ───────────────────────────────────────────────────────


def test_layout_poses_are_read_in_whole_millimetres_and_milliradians():
    assert pose_mm({"x_mm": 450, "y_mm": -20, "yaw_mrad": 1571}) == (450, -20, 1571)
    assert pose_mm({"pose": {"x_mm": 1, "y_mm": 2, "heading_mrad": 3}}) == (1, 2, 3)
    assert pose_mm(types.SimpleNamespace(x_mm=5, y_mm=6)) == (5, 6, 0)
    assert pose_mm((7, 8)) == (7, 8, 0) and pose_mm([7, 8, 9]) == (7, 8, 9)
    assert pose_mm({"x": 1.0, "y": 2.0}) is None and pose_mm(None) is None
    assert parse_spawn("") == ("auto", None) and parse_spawn("Nest") == ("nest", None)
    assert parse_spawn("-3500,-2450,1571") == ("asked", (-3500, -2450, 1571))
    assert parse_spawn("450,0") == ("asked", (450, 0, 0))
    with pytest.raises(ValueError):
        parse_spawn("sofa")


def test_the_spawn_follows_the_layout_spawn_then_nest_then_mat():
    layout = {"nest": {"x_mm": -3500, "y_mm": -2450, "yaw_mrad": 1571}, "mat": [900, 0, 0]}
    assert resolve_spawn(layout, "auto") == (Spawn(-3500, -2450, 1571, at="nest"), [])
    assert resolve_spawn(layout, "mat") == (Spawn(900, 0, 0, at="mat"), [])
    assert resolve_spawn({**layout, "spawn": {"x_mm": 1, "y_mm": 2}}, "auto")[0].at == "layout"
    assert resolve_spawn(layout, "10,20,30") == (Spawn(10, 20, 30, at="asked"), [])
    spawn, notes = resolve_spawn({"mat": [900, 0]}, "nest")
    assert spawn == Spawn() and notes == ["這個家沒有窩：小鴨放在原點"]
    assert resolve_spawn(None, "auto") == (Spawn(), [])

    class Layouts:  # a layout module's own answer wins over the raw fields
        @staticmethod
        def spawn_pose(layout: Any, at: str) -> Any:
            return {"x_mm": 111, "y_mm": 222, "yaw_mrad": 333} if at == "nest" else None

    assert resolve_spawn(layout, "auto", Layouts)[0] == Spawn(111, 222, 333, at="nest")


def _scenes(folder: Path) -> None:
    (folder / "scene_allcollisions.xml").write_text(
        '<mujoco><include file="robot_allcollisions.xml"/></mujoco>'
    )
    (folder / "robot_allcollisions.xml").write_text("<mujoco/>")
    (folder / "scene_home.xml").write_text(
        '<mujoco><include file="robot_allcollisions.xml"/><include file="rooms/rooms.xml"/>'
        "</mujoco>"
    )
    (folder / "rooms").mkdir(exist_ok=True)
    (folder / "rooms" / "rooms.xml").write_text('<mujoco><include file="walls.xml"/></mujoco>')
    (folder / "rooms" / "walls.xml").write_text("<mujoco/>")


def test_a_named_home_falls_back_to_the_flat_floor_while_homelayout_is_missing(tmp_path):
    _scenes(tmp_path)
    flat = plan_home(None, mjcf_dir=tmp_path)
    assert (flat.info.name, flat.info.source, flat.info.notes_zh) == ("flat", "flat", [])
    assert plan_home("flat", mjcf_dir=tmp_path).info.source == "flat"
    plan = plan_home("default_home", mjcf_dir=tmp_path, layouts=None, spawn="nest")
    assert (plan.info.name, plan.info.source, plan.info.asked) == (
        "flat",
        "fallback",
        "default_home",
    )
    assert plan.info.base_scene == tmp_path / "scene_allcollisions.xml"
    assert "homelayout" in plan.info.notes_zh[0] and "default_home" in plan.info.notes_zh[0]
    assert plan.info.notes_zh[1] == "這個家沒有窩：小鴨放在原點"
    assert plan.info.official_files == ["scene_allcollisions.xml", "robot_allcollisions.xml"]
    half = types.SimpleNamespace(load=lambda name: {})  # load but no compose: not a fallback
    with pytest.raises(ValueError):
        plan_home("default_home", mjcf_dir=tmp_path, layouts=half)


def test_a_layout_home_names_its_base_scene_spawn_and_files(tmp_path):
    _scenes(tmp_path)
    composed: list[Any] = []
    layouts = types.SimpleNamespace(
        load=lambda name: {
            "name": "test_home",
            "base": {"scene": "scene_home.xml"},
            "nest": {"x_mm": 450, "y_mm": 0, "yaw_mrad": 0},
        },
        compose=lambda layout, collision: composed.append((layout["name"], collision)) or "spec",
        fingerprint=lambda layout: "abc123",
    )
    plan = plan_home("test_home", mjcf_dir=tmp_path, layouts=layouts)
    info = plan.info
    assert (info.name, info.source, info.fingerprint) == ("test_home", "layout", "abc123")
    assert info.base_scene == tmp_path / "scene_home.xml"
    assert info.official_files == [
        "scene_home.xml",
        "robot_allcollisions.xml",
        "rooms/rooms.xml",
        "rooms/walls.xml",
    ]
    assert info.spawn == Spawn(450, 0, 0, at="nest")
    assert plan.make_spec() == "spec" and composed == [("test_home", "allcollisions")]
    assert info.view()["spawn"] == {"x_mm": 450, "y_mm": 0, "yaw_mrad": 0, "at": "nest"}


def test_a_layout_on_an_official_world_loads_its_files_and_gets_the_mjcf_folder(tmp_path):
    """homelayout's shape: ``base = {"world": ..., "scenes": {collision: scene}}``, ``id``,
    ``variant``, ``spawn`` / ``nest`` / ``mat`` poses with ``heading_mrad``, and ``load`` /
    ``compose`` taking ``mjcf_dir`` (the folder this server checked, not the environment's)."""
    _scenes(tmp_path)
    (tmp_path / "world.xml").write_text('<mujoco><include file="rooms/rooms.xml"/></mujoco>')
    seen: dict[str, Any] = {}

    def load(name: str, *, variant: Any = None, mjcf_dir: Any = None) -> Any:
        seen["load"] = (name, mjcf_dir)
        pose = types.SimpleNamespace(x_mm=-3500, y_mm=-2450, heading_mrad=1571, z_mm=5)
        return types.SimpleNamespace(
            id="apartment",
            variant="no_nest",
            base={"world": "world.xml", "scenes": {"groundcontact": "scene_home.xml"}},
            spawn=pose,
            nest=None,
            mat=types.SimpleNamespace(pose=pose),
            fingerprint="f00d",
        )

    def compose(layout: Any, collision: str = "allcollisions", *, mjcf_dir: Any = None) -> Any:
        seen["compose"] = (layout.id, collision, mjcf_dir)
        return "spec"

    layouts = types.SimpleNamespace(load=load, compose=compose)
    plan = plan_home("apartment,no_nest", mjcf_dir=tmp_path, layouts=layouts)
    info = plan.info
    assert seen["load"] == ("apartment,no_nest", str(tmp_path))
    assert (info.name, info.variant, info.fingerprint) == ("apartment", "no_nest", "f00d")
    assert info.spawn == Spawn(-3500, -2450, 1571, at="layout")
    # no scene of its own for allcollisions: the collision scene plus the world file
    assert info.base_scene == tmp_path / "scene_allcollisions.xml"
    assert info.official_files == [
        "scene_allcollisions.xml",
        "robot_allcollisions.xml",
        "world.xml",
        "rooms/rooms.xml",
        "rooms/walls.xml",
    ]
    assert plan.make_spec() == "spec"
    assert seen["compose"] == ("apartment", "allcollisions", str(tmp_path))
    ground = plan_home("apartment", mjcf_dir=tmp_path, layouts=layouts, collision="groundcontact")
    assert ground.info.base_scene == tmp_path / "scene_home.xml"
    assert info.brief() == {
        "name": "apartment",
        "variant": "no_nest",
        "source": "layout",
        "fingerprint": "f00d",
    }


# ── pins ────────────────────────────────────────────────────────────────────────


def test_simhome_pins_everything_bodyhost_pins_and_what_it_patches_besides():
    assert bodyhost.UPSTREAM_SHA256.items() <= SIMHOME_UPSTREAM_SHA256.items()
    assert set(SIMHOME_UPSTREAM_SHA256) - set(bodyhost.UPSTREAM_SHA256) == {
        "World.__init__",
        "build_world",
        "pose_table",
        "Body.__init__",
        "Body.place",
        "Body.remember",
        "Body.restore",
    }
    assert bodyhost.REVIEWED_SHA256.items() <= REVIEWED_SHA256.items()
    assert set(HOME_REVIEWED_SHA256) == {"scene_apartment.xml", "apartment.xml"}


def test_the_new_pins_are_the_digests_inspect_getsource_gives(tmp_path):
    text = (
        "class World:\n"
        "    def __init__(self, scene, count=1):\n"
        "        self.model = build_world(scene, count)\n"
        "\n"
        "class Body:\n"
        "    def place(self, pose, trunk_z, offset_y):\n"
        "        self.remember()\n"
        "\n"
        "def pose_table(scene, keyframe):\n"
        "    return None, 0.125\n"
    )
    path = tmp_path / "fake_bs.py"
    path.write_text(text)
    spec = importlib.util.spec_from_file_location("fake_bs_pins", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    digests = source_digests(text, ["World.__init__", "Body.place", "pose_table"])
    for name, obj in {
        "World.__init__": module.World.__init__,
        "Body.place": module.Body.place,
        "pose_table": module.pose_table,
    }.items():
        assert digests[name] == hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()


def test_the_pinned_upstream_functions_are_the_ones_reviewed():
    assert bodyhost.check_upstream(_official_body_server(), SIMHOME_UPSTREAM_SHA256) == []


def test_the_official_home_files_are_the_reviewed_ones():
    folder = _mjcf()
    files = official_files(folder, "scene_apartment.xml")
    assert files == ["scene_apartment.xml", "robot_allcollisions.xml", "apartment.xml"]
    assert check_home_files(folder, files) == []
    assert check_home_files(folder, official_files(folder, "scene_allcollisions.xml")) == []


def test_a_changed_unlisted_or_missing_home_file_is_named(tmp_path):
    _scenes(tmp_path)
    digest = hashlib.sha256((tmp_path / "scene_home.xml").read_bytes()).hexdigest()
    pinned = {"scene_home.xml": digest, "robot_allcollisions.xml": "0" * 64}
    (tmp_path / "rooms" / "walls.xml").unlink()
    problems = check_home_files(
        tmp_path,
        official_files(tmp_path, "scene_home.xml") + ["rooms/walls.xml"],
        pinned,
    )
    assert problems[0] == (
        "上游的 robot_allcollisions.xml 和核對過的版本不一樣：還沒有人核對，物理可能變了"
    )
    assert "rooms/rooms.xml 不在核對表裡" in problems[1]
    assert problems[2] == "上游的 rooms/walls.xml 找不到：這個家的物理不完整"


# ── the process refuses before anything starts ──────────────────────────────────


@pytest.mark.parametrize(
    "argv",
    [
        ["--headless", "--scene", "other.xml"],
        ["--headless", "--ducks", "2", "--no-lockstep"],
        ["--headless", "--cameras", "a"],
        ["--keyframe", "SIT"],  # a viewer in lockstep
        ["--headless", "--spawn", "sofa"],
    ],
)
def test_what_a_home_cannot_do_is_refused_before_anything_starts(argv, capsys):
    with pytest.raises(SystemExit) as stopped:
        simhome.main(argv)
    assert stopped.value.code == 3
    assert "拒絕啟動" in capsys.readouterr().err


def test_a_changed_upstream_refuses_to_start_and_names_what(tmp_path, monkeypatch, capsys):
    path = tmp_path / "body_server.py"
    path.write_text("def run(world, headless):\n    pass\n")
    fake = types.ModuleType("mjlab_microduck.sim.body_server")
    fake.__file__ = str(path)
    package, sim = types.ModuleType("mjlab_microduck"), types.ModuleType("mjlab_microduck.sim")
    sim.body_server = fake  # type: ignore[attr-defined]
    package.sim = sim  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mjlab_microduck", package)
    monkeypatch.setitem(sys.modules, "mjlab_microduck.sim", sim)
    monkeypatch.setitem(sys.modules, "mjlab_microduck.sim.body_server", fake)
    with pytest.raises(SystemExit) as stopped:
        simhome.main(["--headless", "--port", "1"])
    assert stopped.value.code == 3
    said = capsys.readouterr().err
    for name in ("Body.place", "build_world", "pose_table", "World.step"):
        assert name in said


# ── livebody and the client ─────────────────────────────────────────────────────


def test_the_live_body_runs_in_the_default_home_unless_told_otherwise(tmp_path, monkeypatch):
    """Phase 2 (owner decision A1, integration): the live body runs in the world server in
    default_home by default (a duck without a nest in its no_nest variant, A2), with the
    household and the sensors loaded; ``DUCKMIND_LIVE_HOME=bodyhost`` runs phase 0's body host.
    (Before the integration the body host was the default and simhome only on request.)"""
    for name in (
        "DUCKMIND_LIVE_HOME",
        "DUCKMIND_LIVE_COLLISION",
        "DUCKMIND_LIVE_LOCKSTEP",
        "DUCKMIND_SIMHOME_EXT",
        "DUCKMIND_SENSORS_DIR",
        "DUCKMIND_TOFD",
        "DUCKMIND_SIM_TOF",
    ):
        monkeypatch.delenv(name, raising=False)
    paths = LivePaths(tmp_path, tmp_path, tmp_path, tmp_path / "robotd", tmp_path / "ort")
    plain = LiveBody(paths)
    assert plain.home == "default_home"
    assert plain._body_command(1)[1:3] == ["-m", "duckmind.world.simhome"]
    assert plain._body_command(1)[-2:] == ["--home", "default_home"]
    assert LiveBody(paths, nest=False).home == "default_home,no_nest"
    assert LiveBody(paths, home="default_home,edge_3mm", nest=False).home == "default_home,edge_3mm"
    homed = LiveBody(paths, home="default_home", tof=False)
    command = homed._body_command(7801)
    assert command[1:3] == ["-m", "duckmind.world.simhome"]
    assert command[-2:] == ["--home", "default_home"]
    assert command[command.index("--collision") + 1] == "allcollisions" and "--lockstep" in command
    env = homed._body_env(tmp_path)
    assert env["DUCKMIND_SIMHOME_EXT"] == "duckmind.world.worldkit,duckmind.world.sensors"
    assert env["DUCKMIND_SENSORS_DIR"] == str(tmp_path / "sensors")
    assert env["DUCKMIND_SIM_TOF"] == "0"  # F2: no module, tofd still runs and says so
    assert "DUCKMIND_TOFD" not in env  # (none was built next to this robotd)
    (tmp_path / "tofd").write_text("")
    assert homed._body_env(tmp_path)["DUCKMIND_TOFD"] == str(tmp_path / "tofd")
    monkeypatch.setenv("DUCKMIND_SIMHOME_EXT", "duckmind.world.hand")  # the environment wins
    assert homed._body_env(tmp_path)["DUCKMIND_SIMHOME_EXT"] == "duckmind.world.hand"
    monkeypatch.setenv("DUCKMIND_LIVE_HOME", "flat")
    assert LiveBody(paths).home == "flat" and live_home() == "flat"
    assert live_home("apartment") == "apartment"  # given beats the environment
    monkeypatch.setenv("DUCKMIND_LIVE_HOME", "  ")
    assert live_home() == "default_home"
    monkeypatch.setenv("DUCKMIND_LIVE_HOME", "bodyhost")
    body_host = LiveBody(paths)
    assert body_host.home is None and body_host._body_command(1)[1:3] == ["-m", "duckmind.bodyhost"]
    assert "--home" not in body_host._body_command(1)
    monkeypatch.delenv("DUCKMIND_SIMHOME_EXT", raising=False)
    assert "DUCKMIND_SIMHOME_EXT" not in body_host._body_env(tmp_path)  # (no world, no household)


def test_the_client_sends_an_ops_fields():
    got: list[dict[str, Any]] = []

    class Answer(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            for raw in self.rfile:
                request = json.loads(raw)
                got.append(request)
                self.wfile.write((json.dumps({"ok": True}) + "\n").encode())
                self.wfile.flush()

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Answer)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, args=(0.02,), daemon=True).start()
    try:
        client = simhome.WorldClient(server.server_address[1])
        client.place(450, -20, 1571)
        client.world()
        HostClient(server.server_address[1]).call("x_hand", to="nest")
        client.close()
    finally:
        server.shutdown()
        server.server_close()
    assert got == [
        {"op": "x_place", "x_mm": 450, "y_mm": -20, "yaw_mrad": 1571, "keyframe": "SIT"},
        {"op": "x_world"},
        {"op": "x_hand", "to": "nest"},
    ]


# ── the keyframe trap ───────────────────────────────────────────────────────────


def test_strip_keyframes_deletes_every_key():
    deleted: list[str] = []
    spec = types.SimpleNamespace(keys=["INIT", "SIT"], delete=deleted.append)
    assert strip_keyframes(spec) == 2 and deleted == ["INIT", "SIT"]


# ── the official body server in process (MuJoCo, no robotd) ─────────────────────


def _official(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A fresh copy of the official body server module (so patches never leak)."""
    pytest.importorskip("mujoco")
    pytest.importorskip("numpy")
    server = _official_body_server()
    monkeypatch.syspath_prepend(str(server.parents[2]))
    spec = importlib.util.spec_from_file_location(f"official_bs_{time.monotonic_ns()}", server)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _apartment_layouts(folder: Path) -> Any:
    """A stand-in layout module over the official apartment (until homelayout lands)."""
    mujoco = pytest.importorskip("mujoco")

    return types.SimpleNamespace(
        load=lambda name: {
            "name": "apartment_probe",
            "base_scene": "scene_apartment.xml",
            "nest": {"x_mm": -2500, "y_mm": 0, "yaw_mrad": 1571},
        },
        compose=lambda layout, collision: mujoco.MjSpec.from_file(
            str(folder / "scene_apartment.xml")
        ),
    )


def _world(
    monkeypatch: pytest.MonkeyPatch,
    *,
    home: str | None = None,
    layouts: Any = None,
    registry: Registry | None = None,
) -> tuple[SimHome, Any, Any, Any, Any]:
    bs = _official(monkeypatch)
    folder = _mjcf()
    host = SimHome(
        collision="allcollisions",
        lockstep=True,
        mjcf_dir=folder,
        home=home,
        registry=registry,
        layouts=layouts,
    )
    host.install(bs)
    world = bs.World(host.keyframe_scene, 1)
    pose, trunk_z = bs.pose_table(host.keyframe_scene, "SIT")
    body = bs.Body(world, 0)
    body.place(pose, trunk_z, offset_y=0.0)  # as body_server.main does: the spawn
    world.bodies.append(body)
    bs.mujoco.mj_forward(world.model, world.data)
    return host, bs, world, body, bs.Handler.__new__(bs.Handler)


def test_an_extension_that_cannot_get_ready_is_switched_off_and_the_world_runs(monkeypatch):
    """Phase-2 review: an exception in an extension's on_ready used to stop the world server
    before it accepted a connection. Now that extension (and its step hooks) is switched off,
    x_monitor says so (trust "suspect") and the physics runs."""
    registry = Registry()
    ran: list[str] = []
    registry.owner = "tests.broken"
    registry.on_ready(lambda world, body: (_ for _ in ()).throw(OSError("socket path too long")))
    registry.after_step(lambda world: ran.append("broken"))
    registry.owner = "tests.fine"
    registry.after_step(lambda world: ran.append("fine"))
    registry.owner = "simhome"
    host, bs, world, body, handler = _world(monkeypatch, registry=registry)
    dispatch = bs.Handler.dispatch
    for _ in range(3):
        dispatch(handler, body, {"op": "write", "targets": [0.0] * 15})
    assert "fine" in ran and "broken" not in ran  # its module's step hooks never ran
    report = dispatch(handler, body, {"op": "x_monitor"})
    assert report["trust"] == "suspect"
    assert any("準備時出錯" in line and "socket path too long" in line for line in report["why_zh"])


def test_the_flat_home_spawns_at_rest_and_lockstep_runs_the_hooks(monkeypatch):
    registry = Registry()
    substeps: list[float] = []
    registry.after_step(lambda world: substeps.append(float(world.data.time)))
    ready: list[Any] = []
    registry.on_ready(lambda world, body: ready.append((world.model.nkey, body.index)))
    host, bs, world, body, handler = _world(monkeypatch, registry=registry)
    assert ready == [(0, 0)]  # after the spawn, in a model without keyframes
    spawned = host.spawned or {}
    assert spawned["ok"] and spawned["support"] == ["floor"] and spawned["at"] == "origin"
    assert 72 <= spawned["trunk_z_mm"] <= 78  # SIT (70 mm, 4.4 mm into the floor) at rest
    dispatch = bs.Handler.dispatch
    for _ in range(5):
        dispatch(handler, body, {"op": "write", "targets": [0.0] * 15})
    assert len(substeps) == 20 and substeps[-1] == pytest.approx(0.1)
    trunk = int(body.trunk)
    assert world.data.qpos[trunk + 2] == pytest.approx(spawned["trunk_z_mm"] / 1000, abs=6e-4)
    info = dispatch(handler, body, {"op": "x_info"})
    assert info["host"] == simhome.HOME_VERSION and info["robot_colliding_geoms"] == 70
    assert info["home"]["source"] == "flat" and info["world"]["keyframes"] == 0
    assert info["upstream_sha256"] == SIMHOME_UPSTREAM_SHA256
    assert info["hooks"]["after_step"] and info["spawn"] == host.spawned


def test_x_place_sets_the_duck_down_held_where_it_was_put(monkeypatch):
    host, bs, world, body, handler = _world(monkeypatch)
    dispatch = bs.Handler.dispatch
    done = dispatch(handler, body, {"op": "x_place", "x_mm": 450, "y_mm": -200, "yaw_mrad": 1571})
    assert done["ok"] and done["blocked_by"] == [] and done["released"] is False
    assert (done["keyframe"], done["surface_mm"], done["support"]) == ("SIT", 0, ["floor"])
    for _ in range(25):  # not enabled yet: held there while the world runs
        dispatch(handler, body, {"op": "write", "targets": [0.0] * 15})
    truth = dispatch(handler, body, {"op": "x_world"})
    duck = truth["duck"]
    assert duck["pos_mm"][:2] == [450, -200] and duck["yaw_mrad"] == 1571
    assert duck["tilt_mrad"] == 0 and duck["released"] is False
    assert truth["props"] == [] and truth["mocap"] == []
    assert all(truth[section] is None for section in simhome.SNAPSHOT_SECTIONS)
    assert truth["sim_time"] == pytest.approx(0.5)
    standing = dispatch(handler, body, {"op": "x_place", "x_mm": 0, "y_mm": 0, "keyframe": "STAND"})
    assert standing["ok"] and standing["trunk_z_mm"] > done["trunk_z_mm"] + 30
    with pytest.raises(ValueError):
        dispatch(handler, body, {"op": "x_place", "x_mm": 0})


def test_in_a_home_the_toys_stay_where_they_are_and_placements_see_obstacles(monkeypatch):
    folder = _mjcf()
    mujoco = pytest.importorskip("mujoco")

    official = mujoco.MjModel.from_xml_path(str(folder / "scene_apartment.xml"))
    sit = mujoco.mj_name2id(official, mujoco.mjtObj.mjOBJ_KEY, "SIT")
    assert official.nkey == 4  # the trap: the official SIT key puts every toy at the origin
    trap = mujoco.MjData(official)
    mujoco.mj_resetDataKeyframe(official, trap, sit)
    assert trap.qpos[official.nq - 7 : official.nq - 4].tolist() == [0.0, 0.0, 0.0]

    host, bs, world, body, handler = _world(
        monkeypatch, home="apartment_probe", layouts=_apartment_layouts(folder)
    )
    dispatch = bs.Handler.dispatch
    assert host.home is not None and host.home.source == "layout"
    assert host.home.official_files == [
        "scene_apartment.xml",
        "robot_allcollisions.xml",
        "apartment.xml",
    ]
    assert host.home.keyframes_removed == 4 and world.model.nkey == 0
    assert (host.spawned or {})["x_mm"] == -2500 and (host.spawned or {})["ok"]
    before = {p["name"]: p["pos_mm"] for p in dispatch(handler, body, {"op": "x_world"})["props"]}
    assert before["obj_0"] == [-2200, -1200, 0] and before["ball_0"] == [-3000, 0, 0]
    wall = dispatch(handler, body, {"op": "x_place", "x_mm": -1000, "y_mm": 0})
    assert not wall["ok"] and wall["blocked_by"] == ["wB_m"] and "卡進" in wall["why_zh"][0]
    trunk = int(body.trunk)
    assert world.data.qpos[trunk : trunk + 2].tolist() == pytest.approx([-2.5, 0.0])  # undone
    toy = dispatch(handler, body, {"op": "x_place", "x_mm": -2200, "y_mm": -1200})
    assert not toy["ok"] and toy["blocked_by"] == ["obj_0_geom"]
    forced = dispatch(handler, body, {"op": "x_place", "x_mm": -2200, "y_mm": -1250, "force": True})
    assert forced["ok"] and forced["blocked_by"] == ["obj_0_geom"]
    outside = dispatch(handler, body, {"op": "x_place", "x_mm": 9000, "y_mm": 0})
    assert not outside["ok"] and outside["surface_mm"] is None
    step = dispatch(handler, body, {"op": "x_place", "x_mm": -200, "y_mm": -1000})
    assert step["ok"] and step["surface_mm"] == -200 and "stair_2" in step["support"]
    for yaw in (0, 1571, 3141):  # a free spot in the living room, any heading
        free = dispatch(
            handler, body, {"op": "x_place", "x_mm": -2600, "y_mm": 400, "yaw_mrad": yaw}
        )
        assert free["ok"] and free["support"] == ["floor_w"] and free["surface_mm"] == 0
    after = {p["name"]: p["pos_mm"] for p in dispatch(handler, body, {"op": "x_world"})["props"]}
    for name in ("obj_2", "obj_3", "ball_1", "ball_2"):
        assert after[name] == before[name], name  # placements never move the toys
    info = dispatch(handler, body, {"op": "x_info"})
    assert info["robot_colliding_geoms"] == 70  # the seven toys are not the robot
    assert len(info["world"]["props"]) == 7 and info["world"]["static_colliding_geoms"] == 61


def test_an_extension_edits_the_spec_and_adds_a_sensor_op_and_a_section(monkeypatch):
    mujoco = pytest.importorskip("mujoco")

    registry = Registry()

    def add_pad(spec: Any, home: HomeInfo) -> None:
        pad = spec.worldbody.add_geom(name="test_pad")
        pad.type = mujoco.mjtGeom.mjGEOM_BOX
        pad.size = [0.2, 0.2, 0.0025]
        pad.pos = [0.0, 0.0, 0.0025]

    def pg(body: Any, request: dict[str, Any]) -> dict[str, Any]:
        with body.world.lock:
            return {"contact": bool(body.world.data.ncon)}

    def add_probe(spec: Any, home: HomeInfo) -> None:  # a belly probe reaching the pad
        probe = spec.body("trunk_base").add_geom(name="test_probe")
        probe.type = mujoco.mjtGeom.mjGEOM_SPHERE
        probe.size = [0.01, 0.0, 0.0]
        probe.pos = [0.0, 0.0, -0.07]
        probe.group = simhome.SENSOR_GROUP
        probe.mass = 0.0

    registry.edit_spec(add_pad)
    registry.edit_spec(add_probe)
    registry.op("pg", pg)
    registry.snapshot("nest", lambda world: {"pad_mm": 5})
    host, bs, world, body, handler = _world(monkeypatch, registry=registry)
    spawned = host.spawned or {}
    assert spawned["support"] == ["test_pad"] and spawned["surface_mm"] == 5
    assert 77 <= spawned["trunk_z_mm"] <= 83 and spawned["blocked_by"] == []  # probe ignored
    assert set(bs.Handler.dispatch(handler, body, {"op": "pg"})) == {"contact"}
    truth = bs.Handler.dispatch(handler, body, {"op": "x_world"})
    assert truth["nest"] == {"pad_mm": 5} and truth["hand"] is None


def _homes() -> list[str]:
    layouts = simhome.find_homelayout()
    if layouts is None:
        return ["(homelayout missing)"]
    return [*layouts.available(), "default_home,no_nest"]


@pytest.mark.parametrize("name", _homes())
def test_every_home_builds_spawns_clear_and_keeps_its_things_in_place(name, monkeypatch):
    layouts = simhome.find_homelayout()
    if layouts is None:
        pytest.skip("duckmind.world.homelayout is not there yet")
    host, bs, world, body, handler = _world(monkeypatch, home=name, layouts=layouts)
    dispatch = bs.Handler.dispatch
    assert host.home is not None and host.home.source == "layout"
    assert check_home_files(_mjcf(), host.home.official_files) == []  # every file reviewed
    assert world.model.nkey == 0
    spawned = host.spawned or {}
    assert spawned["ok"] and spawned["blocked_by"] == [], spawned
    layout = host.home.layout
    expected = {thing.body: thing.pos_mm for thing in layout.things}
    for _ in range(25):  # half a second of physics, the duck held at its spawn
        dispatch(handler, body, {"op": "write", "targets": [0.0] * 15})
    truth = dispatch(handler, body, {"op": "x_world"})
    props = {p["name"]: p["pos_mm"] for p in truth["props"]}
    for thing, (x, y, _z) in expected.items():
        assert abs(props[thing][0] - x) <= 5 and abs(props[thing][1] - y) <= 5, thing
    duck = truth["duck"]
    spawn = layout.spawn
    assert abs(duck["pos_mm"][0] - spawn.x_mm) <= 1 and abs(duck["pos_mm"][1] - spawn.y_mm) <= 1
    info = dispatch(handler, body, {"op": "x_info"})
    assert info["robot_colliding_geoms"] == 70 and info["home"]["fingerprint"]
    mat = layout.mat
    back = dispatch(
        handler,
        body,
        {"op": "x_place", "x_mm": mat.x_mm, "y_mm": mat.y_mm, "yaw_mrad": mat.heading_mrad},
    )
    assert back["ok"], back


def test_in_default_home_the_walls_are_physical(monkeypatch):
    layouts = simhome.find_homelayout()
    if layouts is None:
        pytest.skip("duckmind.world.homelayout is not there yet")
    host, bs, world, body, handler = _world(monkeypatch, home="default_home", layouts=layouts)
    dispatch = bs.Handler.dispatch
    for x_mm, y_mm in ((3000, 0), (-3000, 0), (0, 3000), (0, -3000)):  # Duck Core's room edge
        wall = dispatch(handler, body, {"op": "x_place", "x_mm": x_mm, "y_mm": y_mm})
        assert not wall["ok"] and wall["blocked_by"], (x_mm, y_mm, wall)
    inside = dispatch(handler, body, {"op": "x_place", "x_mm": 450, "y_mm": 0})
    assert inside["ok"] and inside["blocked_by"] == []


# ── the real thing, on request ──────────────────────────────────────────────────


def _live_paths() -> LivePaths:
    paths = LivePaths.from_env()
    if paths is None:
        names = (
            "DUCKMIND_MICRODUCK",
            "DUCKMIND_MICRODUCK_RL",
            "DUCKMIND_POLICIES",
            "DUCKMIND_ROBOTD",
        )
        values = [os.environ.get(name) for name in names]
        if not all(values):
            pytest.skip("set the DUCKMIND_MICRODUCK* / POLICIES / ROBOTD paths")
        from duckmind.sim_live import _installed_ort_library

        paths = LivePaths(*(Path(str(v)).expanduser() for v in values), _installed_ort_library())
    if paths.missing():
        pytest.skip(f"live body needs: {paths.missing()}")
    return paths


@pytest.mark.skipif(
    os.environ.get("DUCKMIND_LIVE_SMOKE") != "1",
    reason="live smoke run (robotd + MuJoCo): set DUCKMIND_LIVE_SMOKE=1",
)
def test_live_smoke_simhome_with_robotd_rises_and_places():
    pytest.importorskip("mujoco")
    home = os.environ.get("DUCKMIND_LIVE_SMOKE_HOME", "flat")
    body = LiveBody(_live_paths(), home=home)
    try:
        body.start()  # rises from the spawn pose, then sits back down
        info = body.host_info or {}
        assert info["host"] == simhome.HOME_VERSION and info["robot_colliding_geoms"] == 70
        assert info["unreviewed"] == [] and info["world"]["keyframes"] == 0
        client = simhome.WorldClient(int(body.body_port or 0))
        spawn = info["home"]["spawn"]
        placed = client.place(spawn["x_mm"] + 300, spawn["y_mm"], spawn["yaw_mrad"])
        assert placed["ok"], placed
        time.sleep(3.0)
        duck = client.world()["duck"]
        assert abs(duck["pos_mm"][0] - (spawn["x_mm"] + 300)) < 60 and duck["tilt_mrad"] < 440
        assert body.policy == "sit" and not body.fallen
        client.close()
    finally:
        body.stop()
