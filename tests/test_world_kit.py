"""The household as one world-server extension (duckmind.world.worldkit): what it registers with
simhome, the parts wired to each other, the eye's copy getting the same bodies, and the whole
thing running inside the official body server (in process, lockstep, no robotd).

The registry cases are pure; the MuJoCo cases skip without MuJoCo or the official checkout and
run in the render check."""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from duckmind.world import worldkit
from duckmind.world.simhome import Registry, SimHome

MJCF = os.environ.get("DUCKMIND_MICRODUCK_MJCF")


def _mjcf() -> Path:
    if not MJCF or not Path(MJCF, "scene_allcollisions.xml").is_file():
        pytest.skip("DUCKMIND_MICRODUCK_MJCF (official microduck MJCF folder) is not set")
    pytest.importorskip("mujoco")
    pytest.importorskip("numpy")
    return Path(MJCF)


def test_the_household_registers_its_ops_sections_and_hooks_in_order():
    registry = Registry()
    registry.load(["duckmind.world.worldkit"])
    view = registry.view()
    assert view["extensions"] == ["duckmind.world.worldkit"]
    assert set(view["registered_ops"]) == {
        "pg",
        "x_nest",
        "x_hand",
        "x_people",
        "x_battery",
        "x_props",
        "x_place_prop",
        "x_tidy",
        "x_roll_ball",
        "x_impacts",
    }
    assert view["sections"] == ["battery", "hand", "nest", "people", "things"]
    assert view["hooks"]["edit_spec"][:3] == [
        "worldkit.spot_probe",
        "nestpad.edit_spec",
        "hand.edit_spec",
    ]
    assert view["hooks"]["before_step"] == ["hand.before_step", "people.before_step"]
    assert view["hooks"]["after_step"] == [
        "nestpad.after_step",
        "battery.after_step",
        "props.after_step",
    ]
    assert "people.planner" in view["hooks"]["on_ready"]
    assert "slow" not in registry.ops and "tof" not in registry.ops  # E1; the ToF is not ours


def test_each_part_is_also_its_own_extension():
    for module, op in (
        ("duckmind.world.nestpad", "pg"),
        ("duckmind.world.hand", "x_hand"),
        ("duckmind.world.people", "x_people"),
        ("duckmind.world.battery", "x_battery"),
        ("duckmind.world.props", "x_props"),
    ):
        registry = Registry()
        registry.load([module])
        assert op in registry.ops


def test_the_battery_charges_from_the_nests_contact():
    kit = worldkit.build()
    assert kit.battery.contact == kit.nest.contact_closed
    assert kit.hand.nest_part is kit.nest and kit.hand.people is kit.people


# ── MuJoCo ──────────────────────────────────────────────────────────────────────


def test_the_eyes_copy_gets_the_same_bodies_in_the_same_order():
    folder = str(_mjcf())
    from duckmind.world import homelayout

    world = worldkit.Headless("default_home", mjcf_dir=folder)
    layout = homelayout.load("default_home", mjcf_dir=folder)
    spec = homelayout.compose(layout, mjcf_dir=folder)
    worldkit.edit_spec(spec, layout)
    eye = spec.compile()
    assert (eye.nq, eye.nv, eye.nmocap, eye.neq) == (
        world.model.nq,
        world.model.nv,
        world.model.nmocap,
        world.model.neq,
    )
    names = [world.model.body(i).name for i in range(world.model.nbody)]
    assert [eye.body(i).name for i in range(eye.nbody)] == names
    assert eye.nmocap == 16  # the hand and five people × (body, two legs)
    from duckmind.world import render_scene

    marks = render_scene.EMITTER_MARKS
    glowing = [
        eye.geom(i).name for i in range(eye.ngeom) if any(m in eye.geom(i).name for m in marks)
    ]
    # the camera model's emitter (C4): the dock's status light (phase 2's open nest: its backlight)
    assert glowing == ["home_nest_status_light"] == render_scene.emitter_names(spec)


def _official(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> Any:
    """A fresh copy of the official body server module; the official package it imports is
    forgotten afterwards (another test looks for mjlab_microduck on its own path)."""
    server = _mjcf().parents[1] / "sim" / "body_server.py"
    if not server.is_file():
        pytest.skip("no official body_server.py next to the MJCF folder")
    monkeypatch.syspath_prepend(str(server.parents[2]))
    before = set(sys.modules)

    def forget() -> None:
        for name in set(sys.modules) - before:
            if name.split(".")[0] == "mjlab_microduck":
                sys.modules.pop(name, None)

    request.addfinalizer(forget)
    spec = importlib.util.spec_from_file_location(f"official_bs_{time.monotonic_ns()}", server)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _server(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest, home: str
) -> tuple[Any, Any, Any, Any]:
    bs = _official(monkeypatch, request)
    registry = Registry()
    registry.load(["duckmind.world.worldkit"])
    host = SimHome(
        collision="allcollisions",
        lockstep=True,
        mjcf_dir=_mjcf(),
        home=home,
        registry=registry,
    )
    host.install(bs)
    world = bs.World(host.keyframe_scene, 1)
    pose, trunk_z = bs.pose_table(host.keyframe_scene, "SIT")
    body = bs.Body(world, 0)
    body.place(pose, trunk_z, offset_y=0.0)  # body_server.main's spawn → simhome's spawn
    world.bodies.append(body)
    bs.mujoco.mj_forward(world.model, world.data)
    handler = bs.Handler.__new__(bs.Handler)
    return host, bs, body, handler


def _ticks(bs: Any, handler: Any, body: Any, n: int, targets: list[float]) -> None:
    for _ in range(n):  # robotd's writes: 4 × 5 ms of world each, in lockstep
        bs.Handler.dispatch(handler, body, {"op": "write", "targets": targets})


def test_in_the_official_body_server_the_household_runs_on_robotds_writes(monkeypatch, request):
    host, bs, body, handler = _server(monkeypatch, request, "default_home")
    dispatch = bs.Handler.dispatch
    info = dispatch(handler, body, {"op": "x_info"})
    assert info["extensions"] == ["duckmind.world.worldkit"] and info["robot_colliding_geoms"] == 70
    assert "kit_hand" in info["world"]["mocap"] and len(info["world"]["mocap"]) == 16
    assert (host.spawned or {})["ok"] and (host.spawned or {})["x_mm"] == 0  # on the nest
    body.set_torque(True)  # what robotd's enable does
    sit = [body.world.data.qpos[a] for a in body.qpos_adr]
    targets = [0.0] * 15
    for slot, wire in enumerate(body.to_wire):
        targets[wire] = float(sit[slot])
    _ticks(bs, handler, body, 50, targets)  # 1 s: seated on the pad
    truth = dispatch(handler, body, {"op": "x_world"})
    assert truth["nest"]["contact"] and truth["nest"]["pg"]
    assert truth["battery"]["pg"] and truth["battery"]["fed_to_robotd"] is False
    assert truth["hand"]["state"] == "idle" and truth["people"] == []
    assert len(truth["things"]) == 8
    assert dispatch(handler, body, {"op": "pg"}) == {"pg": True}
    assert dispatch(handler, body, {"op": "slow"})["volts"] == 7.4  # E1: robotd's reading
    started = dispatch(
        handler, body, {"op": "x_hand", "do": "place", "to": "floor", "x_mm": 1200, "y_mm": -600}
    )
    assert started["ok"]
    _ticks(bs, handler, body, 250, targets)  # 5 s of physics time
    truth = dispatch(handler, body, {"op": "x_world"})
    assert truth["hand"]["last"]["ok"] and not truth["nest"]["contact"]
    assert (
        abs(truth["duck"]["pos_mm"][0] - 1200) < 25 and abs(truth["duck"]["pos_mm"][1] + 600) < 25
    )
    assert truth["duck"]["tilt_mrad"] < 150
    report = dispatch(handler, body, {"op": "x_monitor"})
    assert not any("世界擴充" in line for line in report["why_zh"])  # no hook failed


def test_a_duck_without_a_nest_gets_no_pin_and_the_mat_is_its_desk(monkeypatch, request):
    host, bs, body, handler = _server(monkeypatch, request, "default_home,no_nest")
    dispatch = bs.Handler.dispatch
    truth = dispatch(handler, body, {"op": "x_world"})
    assert truth["nest"] is None and dispatch(handler, body, {"op": "pg"}) == {"pg": False}
    with pytest.raises(ValueError):
        dispatch(handler, body, {"op": "x_hand", "do": "place", "to": "nest"})
    answer = dispatch(handler, body, {"op": "x_hand", "do": "place", "to": "desk"})
    assert answer["target"]["to"] == "desk" and answer["target"]["trunk_mm"] == [0, 0]


@pytest.mark.parametrize("x_mm, y_mm", [(900, 2700), (1300, 2600), (600, 2750), (2990, 2990)])
def test_a_world_placement_never_puts_the_duck_outside_the_house(monkeypatch, request, x_mm, y_mm):
    """Phase-2 review (critical): start/sync placements took the nearest spot with floor under
    it, and the official floor plane runs on past the walls — a plan pose on the sofa by the
    north wall (Duck Core allows |y| ≤ 2.75 m and does not know the sofa) landed outside. Now
    x_place refuses spots outside the layout's rooms, and a stand-in spot must be reachable
    from the asked one without going through a wall."""
    from duckmind.livebody import LivePaths
    from duckmind.simweb.live import LiveBridge

    host, bs, body, handler = _server(monkeypatch, request, "default_home,no_nest")
    dispatch = bs.Handler.dispatch
    outside = dispatch(handler, body, {"op": "x_place", "x_mm": 1275, "y_mm": 3099})
    assert not outside["ok"] and outside["refused"] and "牆外" in outside["why_zh"][0]
    forced = dispatch(handler, body, {"op": "x_place", "x_mm": 1275, "y_mm": 3099, "force": True})
    assert forced["ok"]  # (an evaluator may still put it anywhere on purpose)
    behind = dispatch(
        handler, body, {"op": "x_place", "x_mm": 1200, "y_mm": 2900, "from_mm": [1200, 3200]}
    )
    assert not behind["ok"] and behind["refused"]  # (asked outside: nothing stands in for it)

    class World:
        """What LiveBridge.place needs of a LiveBody, on the in-process world server."""

        home = "default_home,no_nest"
        policy = "sit"
        error = None

        def world(self) -> dict[str, Any]:
            return dispatch(handler, body, {"op": "x_world"})

        def world_call(self, op: str, **params: Any) -> dict[str, Any]:
            answer = dispatch(handler, body, {**params, "op": op})
            if "error" in answer:
                raise RuntimeError(answer["error"])
            return answer

        def stop(self) -> None:
            return None

    root = Path(str(_mjcf())).parents[3]
    paths = LivePaths(root, root.parent, root, root / "robotd", root / "ort")
    bridge = LiveBridge(paths, eye=False)
    bridge.body = World()  # type: ignore[assignment]
    bridge.ready = True
    note = bridge.place((x_mm / 1000, y_mm / 1000, 1.9), "sync")
    duck = dispatch(handler, body, {"op": "x_world"})["duck"]["pos_mm"]
    assert abs(duck[0]) < 2960 and abs(duck[1]) < 2960, (note, duck)  # inside the walls
    if note["ok"]:
        assert not any("home_" in text or "thing_" in text for text in note["why_zh"])
