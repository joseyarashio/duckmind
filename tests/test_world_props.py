"""The home's things (duckmind.world.props): curiosity P0 — the props snapshot, placing a thing,
tidying up, rolling a ball to the duck and the knocks a sound model will hear; the apartment's
official toys kept (owner decision B5). MuJoCo cases; the render check runs them."""

from __future__ import annotations

import math
import os
import types
from pathlib import Path
from typing import Any

import pytest

MJCF = os.environ.get("DUCKMIND_MICRODUCK_MJCF")


def _mjcf() -> str:
    if not MJCF or not Path(MJCF, "scene_allcollisions.xml").is_file():
        pytest.skip("DUCKMIND_MICRODUCK_MJCF (official microduck MJCF folder) is not set")
    pytest.importorskip("mujoco")
    pytest.importorskip("numpy")
    return MJCF


def _home(name: str = "default_home") -> Any:
    from duckmind.world import worldkit

    return worldkit.Headless(name, mjcf_dir=_mjcf())


def test_every_layout_thing_is_known_by_its_id_and_lies_still():
    world = _home()
    props = world.kit.props
    assert set(props.things) == {
        "ball_felt",
        "ball_hollow",
        "fin_fish",
        "fin_duck",
        "paper_cup",
        "tissue_box",
        "bottle_empty",
        "books",
    }
    world.run(2.0)
    listed = props.list_things(world.world, {})["things"]
    assert all(t["moved_mm"] <= 2 and t["speed_mm_s"] <= 5 for t in listed)
    felt = next(t for t in listed if t["id"] == "ball_felt")
    assert felt["zh"] == "毛線球" and felt["cls"] == "ball_slow" and "kick" in felt["verbs"]
    assert felt["room"] == "living" and felt["under"] is None
    # the page's frame (the layout's: footprint centre, lowest point) though the physics frames
    # a ball at its centre
    assert abs(felt["pos_mm"][0] - 800) <= 2 and abs(felt["pos_mm"][1] + 600) <= 2
    assert abs(felt["pos_mm"][2]) <= 2 and abs(felt["centre_mm"][2] - 35) <= 2
    assert props.things["ball_felt"].radius_m == pytest.approx(0.035)
    snap = props.snapshot(world.world)
    assert [t["id"] for t in snap] == [t["id"] for t in listed] and "under" not in snap[0]


def test_the_apartment_keeps_the_official_cubes_and_balls_as_toys():
    world = _home("apartment")
    props = world.kit.props
    assert {"obj_0", "obj_1", "obj_2", "obj_3", "ball_0", "ball_1", "ball_2"} <= set(props.things)
    assert len(props.things) == 15
    obj1 = props.things["obj_1"]
    assert [round(v * 1000) for v in world.data.qpos[obj1.qadr : obj1.qadr + 2]] == [-3300, 1500]
    world.run(1.0)
    assert all(t["moved_mm"] <= 2 for t in props.list_things(world.world, {})["things"])


def test_a_thing_is_put_on_the_highest_surface_and_never_inside_furniture():
    world = _home()
    props = world.kit.props
    on_table = props.place(world.world, {"id": "paper_cup", "x_mm": 1200, "y_mm": 1300})
    assert on_table["ok"] and on_table["on"] == "home_table_top" and on_table["surface_mm"] == 440
    world.run(1.0)
    cup = props.things["paper_cup"]
    assert world.data.qpos[cup.qadr + 2] == pytest.approx(0.44, abs=0.003)  # it stays up there
    floor = props.place(world.world, {"id": "paper_cup", "x_mm": 1200, "y_mm": 1300, "on": "floor"})
    assert floor["ok"] and floor["surface_mm"] == 0
    spawn = world.layout.spawn  # (the duck sits there: in its dock)
    stuck = props.place(world.world, {"id": "books", "x_mm": spawn.x_mm, "y_mm": spawn.y_mm})
    assert not stuck["ok"] and "duck" in stuck["blocked_by"]
    books = props.things["books"]
    assert [round(v * 1000) for v in world.data.qpos[books.qadr : books.qadr + 2]] == [-1600, -300]
    with pytest.raises(ValueError):
        props.place(world.world, {"id": "unicorn", "x_mm": 0, "y_mm": 0})


def test_tidying_up_puts_the_toys_in_the_toy_corner_and_home_puts_all_back():
    world = _home()
    props = world.kit.props
    done = props.tidy(world.world, {})
    assert done["ok"] and {t["id"] for t in done["things"]} == {
        "ball_felt",
        "ball_hollow",
        "fin_fish",
        "fin_duck",
    }
    world.run(1.0)
    for t in props.list_things(world.world, {})["things"]:
        if t["id"] in {"ball_felt", "ball_hollow", "fin_fish", "fin_duck"}:
            x, y = t["pos_mm"][:2]
            assert -2900 <= x <= -1900 and -2900 <= y <= -2100
    back = props.tidy(world.world, {"to": "home"})
    assert back["ok"]
    world.run(1.0)
    assert all(t["moved_mm"] <= 3 for t in props.list_things(world.world, {})["things"])


def test_a_rolled_ball_rolls_to_the_duck_and_the_knock_is_heard():
    world = _home("default_home,no_nest")
    props = world.kit.props
    world.run(0.5)
    props.read_impacts(world.world, {})
    rolled = props.roll(world.world, {"id": "ball_hollow", "toward": "duck", "speed_mm_s": 1000})
    assert rolled["ok"] and rolled["id"] == "ball_hollow"
    ball = props.things["ball_hollow"]
    centre = world.data.xipos[ball.body]
    assert math.dist([float(centre[0]), float(centre[1])], world.trunk[:2]) == pytest.approx(
        1.2, abs=0.01
    )
    speeds = []
    for _ in range(40):  # it rolls (its centre keeps its height and speed): no hopping, no
        # speeding up (the layout's frame at the ball's lowest point did both; props moves the
        # frame to the centre)
        world.step()
        speeds.append(math.hypot(*props._velocity(world.data, ball)[:2]))  # along the floor
        assert abs(float(world.data.xipos[ball.body][2]) - 0.035) < 0.003
    assert 0.95 < min(speeds) and max(speeds) <= 1.001  # the fast ball keeps rolling
    world.run(2.5)
    knocks = props.read_impacts(world.world, {})
    assert any({k["body"], k["other"]} == {"ball_hollow", "duck"} for k in knocks["impacts"])
    assert props.read_impacts(world.world, {})["impacts"] == []  # read once
    again = props.read_impacts(world.world, {"since": 0})
    assert again["impacts"] and again["next"] == props.seq
    felt = props.roll(world.world, {"toward": [0, -2000], "speed_mm_s": 800})
    assert felt["id"] == "ball_felt"  # the first ball, from where it lies
    world.run(3.0)
    moved = props.list_things(world.world, {})["things"]
    assert next(t for t in moved if t["id"] == "ball_felt")["moved_mm"] > 150  # the slow ball
    with pytest.raises(ValueError):
        props.roll(world.world, {"id": "books"})


def test_the_prop_ops_take_the_lock():
    world = _home()
    body = types.SimpleNamespace(world=world.world)
    ops = world.kit.props.ops()
    assert set(ops) == {"x_props", "x_place_prop", "x_tidy", "x_roll_ball", "x_impacts"}
    assert len(ops["x_props"](body, {})["things"]) == 8
    assert ops["x_place_prop"](body, {"id": "fin_duck", "x_mm": 500, "y_mm": -1500})["ok"]
