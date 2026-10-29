"""The family's bodies (duckmind.world.people): owner decisions D3 (people appear only when a
button asks, walk over, then leave) and D4 (they block the duck and push toys but keep their
feet 40 cm from it).

The floor plan and the planner are plain Python on the layout and run everywhere; the walking
bodies need MuJoCo and the official checkout (the render check runs them)."""

from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Any

import pytest

from duckmind.world import homelayout, people

MJCF = os.environ.get("DUCKMIND_MICRODUCK_MJCF")


def _mjcf() -> str:
    if not MJCF or not Path(MJCF, "scene_allcollisions.xml").is_file():
        pytest.skip("DUCKMIND_MICRODUCK_MJCF (official microduck MJCF folder) is not set")
    pytest.importorskip("mujoco")
    pytest.importorskip("numpy")
    return MJCF


# ── the floor plan (no MuJoCo) ──────────────────────────────────────────────────


def test_the_personas_are_the_pages_family_and_differ():
    ids = [p.id for p in people.PERSONAS]
    assert ids == ["owner", "partner", "child", "elder", "guest"]
    assert people.persona("child").height_m < people.persona("owner").height_m
    assert len({p.shirt for p in people.PERSONAS}) == len(ids)
    with pytest.raises(ValueError):
        people.persona("dog")
    assert people.KEEP_M == 0.40


def test_the_furniture_and_the_nest_are_in_the_way_and_the_open_floor_is_not():
    floor = people.Floor(homelayout.load("default_home"))
    assert floor.free(-1.5, -0.5)  # open floor
    assert not floor.free(1.2, 1.3)  # the coffee table
    assert not floor.free(0.9, 2.45)  # the sofa
    assert not floor.free(0.0, -2.7)  # the charging dock by the south wall (its seat)
    assert not floor.free(0.2, -2.55)  # …and its guides' mouth
    assert floor.free(
        0.0, 0.0
    )  # the room's middle: open (only Duck Core believes its nest is here)
    assert not floor.free(-2.95, 0.0)  # the wall
    assert not floor.free(1.0, 0.0, duck=(1.3, 0.0))  # too near the duck
    assert not people.Floor(homelayout.load("default_home,open_nest")).free(0.0, 0.0)  # its pad


def test_a_planned_path_goes_round_the_table_and_keeps_away_from_the_duck():
    layout = homelayout.load("default_home")
    floor = people.Floor(layout)
    duck = (0.6, 1.3)  # between the entry and the far corner, beside the table
    route = floor.path((2.5, -2.5), [(-2.2, 2.2)], duck)
    assert route is not None and route[0] == (2.5, -2.5) and route[-1] == (-2.2, 2.2)
    for a, b in zip(route, route[1:], strict=False):
        for k in range(21):
            x, y = a[0] + (b[0] - a[0]) * k / 20, a[1] + (b[1] - a[1]) * k / 20
            assert floor.free(x, y), (x, y)
            assert math.dist((x, y), duck) >= people.PLAN_KEEP_M - 1e-6
    assert floor.path((2.5, -2.5), [(1.2, 1.3)], duck) is None  # inside the table: nowhere


# ── walking bodies (MuJoCo) ─────────────────────────────────────────────────────


def _home(name: str = "default_home,no_nest") -> Any:
    from duckmind.world import worldkit

    return worldkit.Headless(name, mjcf_dir=_mjcf())


def _walk(world: Any, who: str, limit_s: float = 25.0) -> list[float]:
    """Step until ``who`` stops walking; the feet's clearances seen while they moved."""
    walker = world.kit.people.walkers[who]
    seen: list[float] = []
    for _ in range(int(limit_s / 0.005)):
        world.step()
        if walker.state == "waiting" and walker.needs_plan:
            world.kit.people.pump()  # the planner thread's job in the world server
        if walker.state in ("walking", "leaving", "stepping_back") and walker.clearance is not None:
            seen.append(walker.clearance)
        if walker.state in ("standing", "away"):
            return seen
    raise AssertionError(f"{who} still {walker.state}")


def test_a_person_comes_to_the_duck_and_leaves_and_is_seen_only_while_here():
    world = _home()
    folks = world.kit.people
    assert folks.snapshot(world.world) == []
    body = world.data.mocap_pos[folks.walkers["owner"].body]
    assert body[2] < -1  # parked under the floor
    answer = folks.command(world.world, {"do": "come", "who": "owner"})
    assert answer["ok"] and answer["state"] == "walking" and answer["eta_in_s"] > 1
    seen = _walk(world, "owner")
    here = folks.snapshot(world.world)[0]
    assert here["state"] == "standing" and here["who"] == "owner" and here["zh"] == "主人"
    assert min(seen) >= 0.40 and here["clearance_mm"] >= 400
    duck = world.trunk
    assert math.dist(duck[:2], [v / 1000 for v in here["pos_mm"]]) == pytest.approx(
        people.STAND_OFF_M, abs=0.06
    )
    hand = folks.hand_point("owner")
    assert hand is not None and 0.6 < hand[2] < 1.2
    folks.command(world.world, {"do": "leave", "who": "owner"})
    _walk(world, "owner")
    assert (
        folks.snapshot(world.world) == []
        and world.data.mocap_pos[folks.walkers["owner"].body][2] < -1
    )


def test_people_keep_their_feet_40_cm_from_the_duck_wherever_it_sits():
    rng = random.Random(11)
    world = _home()
    folks = world.kit.people
    trials = 0
    worst = math.inf
    while trials < 10:
        x, y = rng.uniform(-2.3, 2.3), rng.uniform(-2.3, 1.2)
        if not folks.floor.free(x, y) or math.hypot(x - 2.5, y + 2.5) < 1.2:
            continue
        world.data.qpos[world.robot.qadr : world.robot.qadr + 2] = [x, y]
        world.run(0.3)
        who = ("owner", "child", "elder", "partner", "guest")[trials % 5]
        folks.command(world.world, {"do": "come", "who": who})
        worst = min(worst, *_walk(world, who), folks.walkers[who].clearance or math.inf)
        folks.command(world.world, {"do": "leave", "who": who})
        worst = min(worst, *(_walk(world, who) or [math.inf]))
        trials += 1
    assert worst >= people.KEEP_M


def test_a_person_steps_back_when_the_duck_comes_close_and_blocks_it():
    world = _home()
    folks = world.kit.people
    folks.command(world.world, {"do": "come", "who": "partner"})
    _walk(world, "partner")
    walker = folks.walkers["partner"]
    before = walker.pos
    # the duck walks right up to them (placed 0.45 m away)
    away = math.atan2(before[1] - world.trunk[1], before[0] - world.trunk[0])
    world.data.qpos[world.robot.qadr : world.robot.qadr + 2] = [
        before[0] - 0.45 * math.cos(away),
        before[1] - 0.45 * math.sin(away),
    ]
    world.run(2.0)
    assert walker.clearance is not None and walker.clearance >= people.KEEP_M
    assert walker.state == "standing"
    # the legs collide: the person's foot geoms are physical
    import mujoco

    foot = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_GEOM, "person_partner_leg_l_foot")
    assert world.model.geom_contype[foot] == 1 and world.model.geom_conaffinity[foot] == 1


def test_walking_feet_push_a_toy_in_the_way():
    world = _home()
    folks = world.kit.people
    toy = world.kit.props.things["tissue_box"]
    world.data.qpos[world.robot.qadr : world.robot.qadr + 2] = [-2.0, 1.8]  # out of the way
    world.run(0.3)
    folks.command(world.world, {"do": "come", "who": "owner", "to": [0, -2500]})
    route = folks.walkers["owner"].path
    a, b = route[0], route[1]
    along = math.atan2(b[1] - a[1], b[0] - a[0])
    # halfway along the first straight run, under the left foot (0.1 m left of the hips)
    mid = (
        (a[0] + b[0]) / 2 - 0.1 * math.sin(along),
        (a[1] + b[1]) / 2 + 0.1 * math.cos(along),
    )
    placed = world.kit.props.place(
        world.world, {"id": "tissue_box", "x_mm": int(mid[0] * 1000), "y_mm": int(mid[1] * 1000)}
    )
    assert placed["ok"]
    start = [float(v) for v in world.data.qpos[toy.qadr : toy.qadr + 2]]
    _walk(world, "owner")
    world.run(1.0)
    end = [float(v) for v in world.data.qpos[toy.qadr : toy.qadr + 2]]
    assert math.dist(start, end) > 0.05
    impacts = world.kit.props.read_impacts(world.world, {})["impacts"]
    assert any(e["body"] == "tissue_box" and "person_owner" in str(e["other"]) for e in impacts)


def test_people_ops_refuse_what_they_cannot_do():
    world = _home()
    folks = world.kit.people
    with pytest.raises(ValueError):
        folks.command(world.world, {"do": "come", "who": "cat"})
    with pytest.raises(ValueError):
        folks.command(world.world, {"do": "dance", "who": "owner"})
    with pytest.raises(ValueError):
        folks.command(world.world, {"do": "come", "who": "owner", "to": "moon"})
    assert folks.command(world.world, {"do": "leave", "who": "child"})["state"] == "away"
    assert folks.command(world.world, {"do": "status"}) == {"people": []}


def test_the_page_draws_people_from_the_same_shapes_as_the_physics():
    page = people.page_json()
    assert list(page) == [p.id for p in people.PERSONAS]
    child = page["child"]
    assert child["zh"] == "小孩" and child["height_mm"] == 1200
    assert list(child["bodies"]) == ["person_child", "person_child_leg_l", "person_child_leg_r"]
    legs = child["bodies"]["person_child_leg_l"]
    assert all(part["collide"] for part in legs)  # legs and shoes block the duck (D4)
    assert not any(part["collide"] for part in child["bodies"]["person_child"])
    head = next(p for p in child["bodies"]["person_child"] if p["name"].endswith("_head"))
    top = head["pos_mm"][2] + head["size_mm"][0] + people.hip_height(people.persona("child")) * 1000
    assert top == pytest.approx(1200, abs=1)  # the head's top is the person's height
