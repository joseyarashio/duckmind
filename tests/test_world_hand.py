"""The world's hand (duckmind.world.hand): owner decisions D1 (a visible hand carries the physics
duck: pickup, carry, set down on the floor, the mat or the nest) and D2 (set upright is the
everyday recovery). The duck's actuators hold a keyframe (no robotd); the policy-driven
measurements (robotd's control path re-implemented) are in the phase-2 report.

Pure maths runs everywhere; the MuJoCo cases skip without MuJoCo or the official checkout and
run in the render check."""

from __future__ import annotations

import math
import os
import types
from pathlib import Path
from typing import Any

import pytest

from duckmind.world import hand as hand_mod
from duckmind.world import worldkit

MJCF = os.environ.get("DUCKMIND_MICRODUCK_MJCF")


def _mjcf() -> str:
    if not MJCF or not Path(MJCF, "scene_allcollisions.xml").is_file():
        pytest.skip("DUCKMIND_MICRODUCK_MJCF (official microduck MJCF folder) is not set")
    pytest.importorskip("mujoco")
    pytest.importorskip("numpy")
    return MJCF


# ── maths ───────────────────────────────────────────────────────────────────────


def test_min_jerk_starts_and_ends_at_rest():
    assert worldkit.min_jerk(0.0) == 0.0 and worldkit.min_jerk(1.0) == 1.0
    assert worldkit.min_jerk(0.5) == pytest.approx(0.5)
    assert worldkit.min_jerk(-1) == 0.0 and worldkit.min_jerk(2) == 1.0
    eps = 1e-4
    assert (worldkit.min_jerk(eps) - 0) / eps < 1e-6  # no speed at the start


def test_the_heading_survives_a_fall_and_slerp_stays_a_rotation():
    yaw = 0.8
    fallen = worldkit.quat_mul(worldkit.yaw_quat(yaw), [math.cos(0.8), math.sin(0.8), 0, 0])
    assert worldkit.heading_of(fallen) == pytest.approx(yaw, abs=1e-6)
    assert worldkit.tilt_of(fallen) == pytest.approx(1.6, abs=1e-6)
    q = worldkit.slerp(fallen, worldkit.yaw_quat(yaw), 0.5)
    assert sum(v * v for v in q) == pytest.approx(1.0)
    assert worldkit.tilt_of(q) == pytest.approx(0.8, abs=1e-6)
    assert worldkit.rotate(worldkit.yaw_quat(math.pi / 2), [1, 0, 0]) == pytest.approx([0, 1, 0])


# ── the hand in a home (MuJoCo) ─────────────────────────────────────────────────


def _home(name: str = "default_home") -> Any:
    return worldkit.Headless(name, mjcf_dir=_mjcf())


def _until_done(world: Any, limit_s: float = 10.0) -> None:
    hand = world.kit.hand
    steps = 0
    while hand.script is not None and steps < limit_s / 0.005:
        world.step()
        steps += 1
    assert hand.script is None, "the hand's script did not finish"


def test_pickup_lifts_and_holds_the_duck_upright():
    world = _home()
    world.run(1.0)
    answer = world.kit.hand.command(world.world, {"do": "pickup"})
    assert answer["ok"] and answer["do"] == "pickup" and 0.5 < answer["eta_s"] < 3.0
    _until_done(world)
    world.run(0.5)
    assert world.kit.hand.holding and world.data.eq_active[world.kit.hand.weld] == 1
    assert 0.28 < world.trunk[2] < 0.33 and world.tilt < math.radians(5)
    snap = world.kit.hand.snapshot(world.world)
    assert snap["state"] == "hold" and snap["held"] and snap["shown"]
    assert not world.kit.nest.closed  # lifted off the nest: the contact opened


@pytest.mark.parametrize(
    "request_, home",
    [
        (
            {"do": "place", "to": "floor", "x_mm": 1200, "y_mm": -700, "yaw_mrad": 2000},
            "default_home",
        ),
        ({"do": "place", "to": "mat"}, "default_home,no_nest"),
        ({"do": "place", "to": "desk"}, "default_home,no_nest"),
        ({"do": "place", "to": "nest"}, "default_home"),
    ],
)
def test_a_place_carries_the_duck_there_and_sets_it_down_sitting(request_, home):
    world = _home(home)
    # (start somewhere else: the default home's duck starts in its dock by the south wall, so
    # the dock case starts 1 m from it — a carry across the room takes longer than D1's 2–4 s)
    start = [0.6, -1.8] if request_["to"] == "nest" else [1.0, -1.0]
    world.data.qpos[world.robot.qadr : world.robot.qadr + 2] = start
    world.run(1.0)
    answer = world.kit.hand.command(world.world, request_)
    assert answer["ok"] and answer["eta_s"] <= 4.5
    peak = [0.0]
    world.run(0.0)
    while world.kit.hand.script is not None:
        world.step()
        if world.kit.hand.holding:
            peak[0] = max(peak[0], world.tilt)
    world.run(1.5)
    last = world.kit.hand.last or {}
    release = last["target"]["release"]
    aim = answer["target"]["trunk_mm"]
    assert last["ok"] and not world.kit.hand.holding
    assert release["error_mm"] <= 10 and math.degrees(peak[0]) < 10
    assert math.dist([v * 1000 for v in world.trunk[:2]], aim) < 25
    assert world.tilt < math.radians(8) and world.trunk[2] < 0.09  # upright and sitting
    if request_["to"] == "nest":
        assert world.kit.nest.closed and world.kit.nest.pg_op(world, {}) == {"pg": True}
    if request_["to"] == "desk":
        assert "A3" in "".join(answer["why_zh"])


def test_a_taken_spot_moves_the_set_down_next_to_it():
    world = _home("default_home,no_nest")
    books = world.kit.props.things["books"]
    x, y = (v * 1000 for v in world.data.qpos[books.qadr : books.qadr + 2])
    world.data.qpos[world.robot.qadr : world.robot.qadr + 2] = [1.0, 1.0 - 2.2]
    world.run(0.5)
    answer = world.kit.hand.command(
        world.world, {"do": "place", "to": "floor", "x_mm": int(x), "y_mm": int(y)}
    )
    target = answer["target"]
    # (said in the owner's words: the layout's name for the thing, never a MuJoCo name)
    said = answer["why_zh"][0]
    assert target["trunk_mm"] != [int(x), int(y)] and world.layout.thing("books").zh in said
    assert "thing_" not in said and "home_" not in said
    _until_done(world)
    world.run(1.0)
    assert world.tilt < math.radians(8)
    moved = world.data.qpos[books.qadr : books.qadr + 2]
    assert math.dist([v * 1000 for v in moved], [x, y]) < 5  # the books were not shoved


def _deepest_into_furniture(world: Any) -> float:
    """How far (m) any of the duck's colliders is inside a static geom that is not the floor
    it rests on (walls, furniture, the pad's edge), this step."""
    import mujoco

    model, data = world.model, world.data
    statics = set(worldkit.static_supports(model))
    plane = int(mujoco.mjtGeom.mjGEOM_PLANE)
    deepest = 0.0
    for contact in data.contact[: int(data.ncon)]:
        one, two = int(contact.geom1), int(contact.geom2)
        a = int(model.geom_bodyid[one]) in world.robot.bodies
        b = int(model.geom_bodyid[two]) in world.robot.bodies
        if a == b:
            continue
        other = two if a else one
        if other in statics and int(model.geom_type[other]) != plane:
            deepest = max(deepest, -float(contact.dist))
    return deepest


def _run_script(world: Any, limit_s: float = 12.0) -> float:
    """Step until the hand's script ends; the deepest the duck went into furniture (m)."""
    hand = world.kit.hand
    deepest, steps = 0.0, 0
    while hand.script is not None and steps < limit_s / 0.005:
        world.step()
        deepest = max(deepest, _deepest_into_furniture(world))
        steps += 1
    assert hand.script is None, "the hand's script did not finish"
    return deepest


def _table(world: Any) -> Any:
    return world.layout.element("table_top")


def _under(element: Any, xy_m: list[float], grow_mm: float = 0.0) -> bool:
    x0, y0, x1, y1 = element.footprint()
    x, y = xy_m[0] * 1000, xy_m[1] * 1000
    return x0 - grow_mm <= x <= x1 + grow_mm and y0 - grow_mm <= y <= y1 + grow_mm


@pytest.mark.parametrize(
    "start, request_",
    [
        ([1.2, 1.3], {"do": "pickup"}),  # under the coffee table
        ([1.2, 1.3], {"do": "place", "to": "floor", "x_mm": 1200, "y_mm": 300}),  # out of it
        ([1.2, 0.2], {"do": "place", "to": "floor", "x_mm": 1200, "y_mm": 1300}),  # into it
        ([1.2, 1.3], {"do": "upright"}),
    ],
)
def test_the_hand_never_pushes_the_duck_through_furniture(start, request_):
    """Phase-2 review: a duck under the coffee table (40 cm clear: it walks under it) was lifted
    and set down straight through the top (up to 2.9 kN). Now it is slid out along the floor
    first, and a spot under the top is never a set-down spot; the duck never goes more than
    5 mm into any furniture."""
    world = _home("default_home,no_nest")
    world.data.qpos[world.robot.qadr : world.robot.qadr + 2] = start
    world.run(1.0)
    answer = world.kit.hand.command(world.world, dict(request_))
    assert answer["ok"]
    assert _run_script(world) <= 0.005
    world.run(1.0)
    assert _deepest_into_furniture(world) <= 0.005
    top = _table(world)
    if request_["do"] == "pickup":
        assert world.kit.hand.holding and not _under(top, world.trunk, grow_mm=100)
        assert "茶几" in "".join(answer["why_zh"])  # (said: slid out from under the table)
    elif request_.get("y_mm") == 1300:  # asked under the table: set down beside it
        assert not _under(top, world.trunk) and "茶几" in "".join(answer["why_zh"])
    assert world.tilt < math.radians(10)


@pytest.mark.parametrize("x_mm, y_mm", [(900, 2700), (1300, 2600), (600, 2750), (2990, 2990)])
def test_a_set_down_near_the_walls_stays_in_the_house(x_mm, y_mm):
    """Phase-2 review: a spot on the sofa by the north wall moved to the "nearest free" spot on
    the far side of the wall (the official floor plane runs on outside the house). The hand's
    spot is now indoors and reachable from the asked one — or refused, and the duck stays."""
    world = _home("default_home,no_nest")
    world.data.qpos[world.robot.qadr : world.robot.qadr + 2] = [-1.5, -1.5]
    world.run(0.5)
    try:
        answer = world.kit.hand.command(
            world.world, {"do": "place", "to": "floor", "x_mm": x_mm, "y_mm": y_mm}
        )
    except ValueError as error:
        assert "找不到" in str(error)  # nowhere indoors within reach: nothing moves
        return
    aim = answer["target"]["trunk_mm"]
    assert abs(aim[0]) < 2960 - 90 and abs(aim[1]) < 2960 - 90, aim  # (inside the walls)
    _run_script(world)
    world.run(1.0)
    assert abs(world.trunk[0]) < 2.87 and abs(world.trunk[1]) < 2.87


@pytest.mark.parametrize(
    "roll, pitch",
    [
        (math.pi / 2, 0.0),
        (-math.pi / 2, 0.0),
        (0.0, math.pi / 2),
        (0.0, -math.pi / 2),
        (math.pi, 0.0),
    ],
)
def test_set_upright_rights_a_fallen_duck_where_it_lies(roll, pitch):
    world = _home("default_home,no_nest")
    world.data.qpos[world.robot.qadr : world.robot.qadr + 2] = [0.8, 0.6]
    q = worldkit.quat_mul(
        worldkit.yaw_quat(0.7),
        worldkit.quat_mul(
            [math.cos(roll / 2), math.sin(roll / 2), 0, 0],
            [math.cos(pitch / 2), 0, math.sin(pitch / 2), 0],
        ),
    )
    world.data.qpos[world.robot.qadr + 2] = 0.12
    world.data.qpos[world.robot.qadr + 3 : world.robot.qadr + 7] = q
    world.run(2.0)
    assert world.tilt > math.radians(60)  # it lies there (nothing helps by itself: D2)
    where = world.trunk
    answer = world.kit.hand.command(world.world, {"do": "upright"})
    assert answer["ok"] and answer["eta_s"] < 4.0
    _until_done(world)
    world.run(2.0)
    assert world.tilt < math.radians(10) and world.trunk[2] < 0.09
    assert math.dist(where[:2], world.trunk[:2]) < 0.1  # where it lay


def test_touch_and_near_move_nothing_and_a_held_duck_cannot_be_patted():
    world = _home()
    world.run(1.0)
    before = world.trunk
    for request_ in (
        {"do": "touch", "where": "head", "seconds": 0.5},
        {"do": "touch", "where": "beak", "seconds": 0.5},
        {"do": "near", "speed": "fast"},
        {"do": "hold_head", "seconds": 0.4},
    ):
        answer = world.kit.hand.command(world.world, request_)
        states = set()
        while world.kit.hand.script is not None:
            world.step()
            states.add(world.kit.hand.snapshot(world.world)["state"])
        assert answer["ok"] and ({"touch", "near"} & states)
    assert math.dist(before, world.trunk) < 0.003
    assert not world.kit.hand.snapshot(world.world)["shown"]  # parked out of sight
    world.kit.hand.command(world.world, {"do": "pickup"})
    _until_done(world)
    with pytest.raises(ValueError):
        world.kit.hand.command(world.world, {"do": "touch"})
    world.kit.hand.command(world.world, {"do": "cancel"})
    _until_done(world)
    world.run(1.0)
    assert not world.kit.hand.holding and world.trunk[2] < 0.1


def test_a_duck_nobody_enabled_is_carried_and_then_held_where_it_was_put():
    world = _home("default_home,no_nest")
    remembered: list[Any] = []

    class Body:  # the official Body's hold, as simhome's world has it
        released = False
        held: Any = ("somewhere",)

        def remember(self) -> None:
            remembered.append(True)
            self.held = ("here",)

    body = Body()
    world.kit.hand.body = body
    world.run(0.5)
    world.kit.hand.command(world.world, {"do": "place", "to": "floor", "x_mm": 900, "y_mm": 900})
    while not world.kit.hand.holding:
        world.step()
    assert body.held is None  # taken out of the official hold while carried
    _until_done(world)
    assert remembered and body.held == ("here",)


def test_the_hand_lets_go_when_the_duck_is_moved_away_and_refuses_nonsense():
    world = _home()
    world.run(0.5)
    world.kit.hand.command(world.world, {"do": "pickup"})
    _until_done(world)
    world.data.qpos[world.robot.qadr : world.robot.qadr + 3] = [2.0, 2.0, 0.08]  # an x_place
    world.run(0.1)
    assert not world.kit.hand.holding and world.data.eq_active[world.kit.hand.weld] == 0
    assert world.kit.hand.last and not world.kit.hand.last["ok"]
    import mujoco

    # let go before the step: no flinging it back, no unstable step (MuJoCo would reset it)
    assert world.data.warning[int(mujoco.mjtWarning.mjWARN_BADQACC)].number == 0
    assert world.trunk[0] > 1.8 and world.trunk[1] > 1.8
    for bad in ({"do": "juggle"}, {"do": "place", "to": "roof"}, {"do": "place", "from_mm": [1]}):
        with pytest.raises(ValueError):
            world.kit.hand.command(world.world, bad)
    no_nest = _home("default_home,no_nest")
    with pytest.raises(ValueError, match="沒有充電座"):
        no_nest.kit.hand.command(no_nest.world, {"do": "place", "to": "nest"})


def test_the_hand_op_takes_the_lock_and_status_is_the_snapshot():
    world = _home()
    body = types.SimpleNamespace(world=world.world)
    status = world.kit.hand.hand_op(body, {"op": "x_hand", "do": "status"})
    assert status["state"] == "idle" and not status["holding"] and status["pos_mm"] is None
    assert hand_mod.WELD and hand_mod.HAND in [
        name for name in (world.model.body(i).name for i in range(world.model.nbody))
    ]


def test_the_page_draws_the_hand_from_the_same_shapes_as_the_physics():
    page = hand_mod.page_json()
    assert page["body"] == hand_mod.HAND and len(page["parts"]) == 7
    assert {p["shape"] for p in page["parts"]} == {"box", "capsule"}
    assert not any(p["collide"] for p in page["parts"])  # the weld carries, nothing bumps
    palm = page["parts"][0]
    assert palm["pos_mm"] == [-10.0, 0.0, 85.0] and palm["colour"].startswith("#")


# ── a duck under the coffee table of the real home (the family's battery swap was refused) ────────
#
# 蛋黃 walks under the coffee table (40 cm clear) and sits there. In the real home that table stands
# on a thick rug whose edge is 15 mm high, and the table's legs are 4 cm square. The slide-out that
# a lift needs went in straight lines over a floor level to 4 mm, so the rug's edge and the legs
# shut every way: 110 swap visits in ~50 h were refused ("小鴨在茶几底下，手沒辦法把牠拉出來抬起").
# A person reaches under the table and takes it out sideways, round a leg, over the rug's edge.


def _real_home() -> Any:
    return worldkit.Headless("real_home,no_nest", mjcf_dir=_mjcf())


def _put(world: Any, x_m: float, y_m: float, yaw: float = 0.0) -> None:
    """The duck sitting at (x, y) on the rug or the floor, as it came to sit there."""
    import mujoco

    qadr = world.robot.qadr
    surface = world.kit.hand.spots.level(world.data, x_m, y_m) or 0.0
    world.data.qpos[qadr : qadr + 3] = [x_m, y_m, surface + 0.07]
    world.data.qpos[qadr + 3 : qadr + 7] = worldkit.yaw_quat(yaw)
    world.data.qvel[:] = 0
    mujoco.mj_forward(world.model, world.data)
    world.run(1.0)


RUG_PILE_M = 0.006  # a duck at rest on the thick rug is already 4.8 mm into its pile (no hand)
UNDER_THE_TABLE = [(-1.4, -2.2), (-1.0, -2.0), (-0.8, -2.0), (-0.4, -1.8), (-1.2, -1.85)]


@pytest.mark.parametrize("x_m, y_m", UNDER_THE_TABLE)
def test_a_duck_under_the_coffee_table_is_taken_out_round_the_legs_over_the_rugs_edge(x_m, y_m):
    world = _real_home()
    _put(world, x_m, y_m)
    top = world.layout.element("coffee_table_1_top")
    x0, y0, x1, y1 = top.footprint()
    assert x0 < x_m * 1000 < x1 and y0 < y_m * 1000 < y1  # (it really is under the table top)
    answer = world.kit.hand.command(world.world, {"do": "pickup", "who": "owner"})
    assert answer["ok"] and "茶几" in "".join(answer["why_zh"])  # (said: slid out from under it)
    assert _run_script(world) <= RUG_PILE_M  # (no part of it through a leg or the top)
    world.run(0.5)
    assert world.kit.hand.holding and world.kit.hand.refused is None
    assert not _under(top, world.trunk, grow_mm=100)  # (out from under the top, upright, held up)
    assert world.tilt < math.radians(10) and world.trunk[2] > 0.25


def test_a_duck_under_the_table_can_be_lifted_swapped_and_set_down_where_it_was_taken_out():
    world = _real_home()
    _put(world, -0.8, -2.0)
    hand = world.kit.hand
    assert hand.command(world.world, {"do": "pickup", "who": "owner"})["ok"]
    _run_script(world)
    assert hand.holding
    hand.command(world.world, {"do": "place", "to": "here", "who": "owner"})
    assert _run_script(world) <= RUG_PILE_M
    world.run(1.5)
    last = hand.last or {}
    assert last["ok"] and not hand.holding
    assert world.trunk[2] < 0.1 and world.tilt < math.radians(8)  # (sitting, upright)
    assert _deepest_into_furniture(world) <= RUG_PILE_M


def test_a_slide_out_is_only_refused_when_the_way_really_is_shut():
    """No way out: a duck pushed into a pocket under the sofa (the seat is 9 cm up on four legs and
    the arms and back close it in; a rug's edge is lifted over, a sofa's seat is not)."""
    world = _real_home()
    spots = world.kit.hand.spots
    # in the middle of the sofa's footprint, the duck sits on the floor under the seat: the seat
    # is 9 cm above the floor, a sitting duck is taller, so there is no duck there at all
    surface = spots.level(world.data, -0.9, -1.0) or 0.0
    assert spots.headroom(world.data, -0.9, -1.0, surface)[0] < 0.2
    assert spots.slide_out(world.data, -0.9, -1.0, headroom_m=hand_mod.LIFT_ROOM_M) is None
    # but from under the coffee table there is one, and it is a short way
    way = spots.slide_out(world.data, -0.8, -2.0, headroom_m=hand_mod.LIFT_ROOM_M)
    assert way is not None and 0.0 < way.top - way.surface + 0.0 <= worldkit.SLIDE_STEP_MAX_M
    length = sum(math.dist(a, b) for a, b in zip(way.route, way.route[1:], strict=False))
    assert 0.3 < length < worldkit.SLIDE_REACH_M


def test_a_refused_lift_says_why_as_a_number_and_counts_in_the_snapshot():
    world = _real_home()
    hand = world.kit.hand
    _put(world, -0.9, -1.0)  # (the middle of the sofa: no way out from under it)
    with pytest.raises(hand_mod.HandRefused) as refused:
        hand.command(world.world, {"do": "pickup", "who": "owner"})
    assert refused.value.code == hand_mod.REFUSE_NO_WAY_OUT and "沙發" in str(refused.value)
    snap = hand.snapshot(world.world)
    assert snap["refused"]["code"] == hand_mod.REFUSE_NO_WAY_OUT and snap["refused"]["count"] == 1
    assert snap["script"] is None and not snap["holding"]  # (refused: nothing started)
    with pytest.raises(hand_mod.HandRefused):
        hand.command(world.world, {"do": "pickup", "who": "owner"})
    assert hand.snapshot(world.world)["refused"]["count"] == 2
    with pytest.raises(ValueError):  # (a request it does not understand is a number too)
        hand.command(world.world, {"do": "juggle"})
    assert hand.snapshot(world.world)["refused"]["code"] == hand_mod.REFUSE_OTHER


def test_the_slide_search_stays_fast_enough_to_run_under_the_worlds_lock():
    import time

    world = _real_home()
    spots = world.kit.hand.spots
    worst = 0.0
    for x_m, y_m in [*UNDER_THE_TABLE, (-0.9, -1.0), (-2.0, 2.5)]:
        started = time.perf_counter()
        spots.slide_out(world.data, x_m, y_m, headroom_m=hand_mod.LIFT_ROOM_M)
        worst = max(worst, time.perf_counter() - started)
    assert worst < 1.0  # (it holds the world's lock: the physics waits for it)
