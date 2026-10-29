"""The charging dock (duckmind.world.dock; owner decision 2026-09-27: 「窩 做成充電座 樣式（類似掃地
機器人充電座），公寓的窩 放置在牆邊」): its geometry, both docking styles, the contacts that push
nothing and report power-good through the nest's contact path, the layouts that stand it against a
wall with its lane clear, and phase 2's open nest kept as the ``open_nest`` variant.

The duck's actuators hold a keyframe here (no robotd): the policy-driven docking runs (walk in or
back in, sit) are in docs/validation/evidence/phase2/dock_sanity.json. Pure geometry and layout
tests run everywhere; the MuJoCo cases skip without MuJoCo or the official checkout and run in the
render check."""

from __future__ import annotations

import math
import os
import threading
import types
from pathlib import Path
from typing import Any

import pytest

from duckmind.world import apriltag, dock, homelayout, lights, nestpad
from duckmind.world.homelayout import load

MJCF = os.environ.get("DUCKMIND_MICRODUCK_MJCF")
# The duck's measured footprint (dock.py's docstring): standing ±71 mm at the hips, seated
# ±79 mm; sitting down moves the trunk ~72 mm back and ~15 mm sideways, turning it up to ~20°.
SEATED_HALF_MM = 79
STANDING_HALF_MM = 71


def _mjcf() -> str:
    if not MJCF or not Path(MJCF, "scene_allcollisions.xml").is_file():
        pytest.skip("DUCKMIND_MICRODUCK_MJCF (official microduck MJCF folder) is not set")
    pytest.importorskip("mujoco")
    pytest.importorskip("numpy")
    return MJCF


def _apartment(variant: str = "") -> homelayout.Layout:
    if not MJCF or not Path(MJCF, "apartment.xml").is_file():
        pytest.skip("DUCKMIND_MICRODUCK_MJCF (official microduck MJCF folder) is not set")
    return load("apartment" + variant)


# ── the geometry ────────────────────────────────────────────────────────────────


def test_the_two_styles_share_the_housing_and_differ_in_their_face_and_contacts() -> None:
    forward, backin = dock.shape("forward"), dock.shape("backin")
    # the homes dock backin since 2026-09-28 (duck-body's dockexec default); forward is a variant
    assert dock.STYLES == ("forward", "backin") and dock.DEFAULT_STYLE == "backin"
    assert dock.SIDES == ("right", "left") and dock.DEFAULT_SIDE == "right"
    same = {"depth_mm", "width_mm", "height_mm", "throat_mm", "mouth_mm", "plate_mm", "tag_mm"}
    assert all(getattr(forward, k) == getattr(backin, k) for k in same)
    # forward: a 10 mm soft bumper; backin: a flat face (nothing may stop the rear first)
    assert (forward.panel_mm, backin.panel_mm) == (10, 1)
    # the plate: a ramp on every side, no step (the feet lift ~1.1 cm; C2's ramp 30/30)
    assert 0 < forward.plate_mm <= 5 and forward.ramp_mm / forward.plate_mm >= 10
    # the tag's white mount is exactly the face panel's height; the light is above it
    assert forward.panel_z1_mm - forward.panel_z0_mm == round(forward.tag_mm * dock.BORDER)
    assert forward.light_z0_mm > forward.panel_z1_mm and forward.light_z1_mm < forward.height_mm
    # the layout may change any measure (for measurements), never with a bad value
    wide = dock.shape("forward", {"throat_mm": 300})
    assert wide.throat_mm == 300 and wide.mouth_mm == forward.mouth_mm
    with pytest.raises(ValueError, match="沒有這些尺寸"):
        dock.shape("forward", {"funnel": 3})
    with pytest.raises(ValueError, match="整數"):
        dock.shape("backin", {"plate_mm": -1})
    with pytest.raises(ValueError, match="停靠方式"):
        dock.shape("sideways")


def test_the_funnel_is_sized_from_the_ducks_footprint_and_its_sit() -> None:
    g = dock.shape("forward")
    half = g.throat_mm / 2
    # the seated duck fits with its sideways step and its turn; the standing one is centred
    # to within ±60 mm
    assert half >= SEATED_HALF_MM + 15 + 20
    assert half - STANDING_HALF_MM <= 60
    assert g.mouth_mm >= g.throat_mm + 2 * 100  # an approach ±100 mm off is caught
    assert g.plate_width_mm == g.throat_mm  # the plate fills the channel
    # above the thighs' and hips' lowest (≥17 mm); not the 60-120 mm of the first sizing: a duck that fell beside an 80 mm guide lay propped on it
    # (docs/design/neural-duck/DOCK-FALL-2026-10-11.md)
    assert 17 <= g.guide_height_mm <= 40
    segments = dict((name, (a, b)) for name, a, b in dock.guide_segments(g))
    assert set(segments) == set(dock.GUIDES)
    for side in ("l", "r"):
        (a0, b0), (a1, b1) = segments[f"nest_guide_{side}0"], segments[f"nest_guide_{side}1"]
        assert a0[1] == b0[1] and b0 == a1  # a parallel channel, then the flare from its end
        assert abs(b1[1]) > abs(a1[1])
    assert segments["nest_guide_l1"][1] == (
        segments["nest_guide_r1"][1][0],
        -segments["nest_guide_r1"][1][1],
    )


def test_the_seats_and_the_way_in_follow_the_measured_sit() -> None:
    forward, backin = dock.shape("forward"), dock.shape("backin")
    # forward: the chest stops at the bumper, the sit moves the trunk ~72 mm back, the thighs
    # (~20 mm behind the trunk) land on the pads
    seat = dock.seat_x_mm("forward", forward)
    assert seat == forward.face_mm + dock.CHEST_MM + dock.SIT_BACK_MM + 10
    thighs = seat + 18 - forward.depth_mm
    assert forward.pads_x0_mm + 30 <= thighs <= forward.pads_x1_mm - 30
    assert dock.seat_heading_mrad("forward") == 3142 and dock.seat_heading_mrad("backin") == 0
    # backin: the seated rear sits inside the plates' window (face … plate travel + probe)
    seat = dock.seat_x_mm("backin", backin)
    rear = seat - dock.REAR_MM
    assert backin.face_mm < rear < backin.depth_mm + backin.plate_travel_mm + backin.probe_mm
    way = dock.approach("backin", backin)
    assert way["stop_x_mm"] == seat + dock.SIT_BACK_MM + dock.REVERSE_COAST_MM
    assert dock.approach("forward", forward)["exit"] == "reverse" and way["exit"] == "forward"


def test_the_page_draws_the_physics_tags_and_both_styles_contacts() -> None:
    layout = load("default_home,forward")
    page = nestpad.page_json(layout.nest)
    assert page["present"] and page["kind"] == "dock" and page["style"] == "forward"
    assert [t["id"] for t in page["tags"]] == [apriltag.NEST_TAG_ID, apriltag.EDGE_TAG_ID]
    assert page["tags"][0]["cells"] == [list(r) for r in apriltag.grid(apriltag.NEST_TAG_ID)]
    assert page["light"]["geom"] == dock.LIGHT
    pads = page["contacts"]
    assert [c["id"] for c in pads] == ["l", "r"] and pads[0]["kind"] == "floor_pads"
    assert pads[0]["min_mm"][1] == -pads[1]["max_mm"][1]  # mirror images
    backin = nestpad.page_json(load("default_home").nest)  # the home's own dock since 09-28
    assert backin["style"] == "backin" and backin == nestpad.page_json(
        load("default_home,backin").nest
    )
    assert backin["contacts"][0]["kind"] == "spring_plates"
    assert backin["contacts"][0]["min_mm"][0] == dock.shape("backin").depth_mm  # on the face
    assert nestpad.page_json(load("default_home,open_nest").nest)["kind"] == "open"


# ── the layouts ─────────────────────────────────────────────────────────────────


def _clear_of_furniture(layout: homelayout.Layout) -> list[str]:
    """The colliding pieces (not the dock's own, not the wall it stands on) that reach into
    the dock's lane or turning room: the layout's clearance tools, asked directly."""
    nest = layout.nest
    assert nest is not None
    zones = {k: v for k, v in nest.zones_mm().items() if k != "back"}
    hits = []
    for element in layout.elements:
        if not element.collide or element.support or element.id.startswith("nest_"):
            continue
        for label, box in zones.items():
            area = homelayout._local_box(nest.pose, box)
            if homelayout._overlap(area, element.footprint(), homelayout.NEST_MARGIN_MM) > 0:
                if homelayout._circle_or_box_touch(element, area, homelayout.NEST_MARGIN_MM):
                    hits.append(f"{element.id}@{label}")
    return hits


def test_the_default_home_stands_its_dock_on_the_south_wall_with_its_lane_clear() -> None:
    layout = load("default_home")
    nest = layout.nest
    assert nest is not None and nest.is_dock and layout.check() == []
    assert nest.pose == homelayout.Pose(0, -2940, 1571)  # the middle of the south wall
    # 2026-09-28: it docks backin, turning round on the dock's right (the J-hook's lane)
    assert (nest.style, nest.side) == ("backin", "right")
    assert layout.page_json()["nest"]["side"] == "right"
    skirting = layout.element("skirting_s")
    assert skirting.footprint()[3] == nest.y_mm  # its back on the skirting's face
    # 0.8 m and more of clear floor straight out, and no furniture in the way in or out
    lane = nest.zones_mm()["lane"]
    assert lane[1] - lane[0] >= 800 and lane[0] == nest.dock_shape.depth_mm
    assert _clear_of_furniture(layout) == []
    # Duck Core's frame stays the room: its believed nest (0, 0) is not the dock (phase 5)
    assert layout.frame == homelayout.Pose(0, 0, 0)
    assert math.hypot(nest.seat.x_mm, nest.seat.y_mm) > 2000
    page = layout.page_json()
    assert page["nest"]["kind"] == "dock" and page["nest"]["seat"] == nest.seat.as_json()
    assert page["frame"] == {"x_mm": 0, "y_mm": 0, "heading_mrad": 0, "z_mm": 0}
    plate = next(e for e in page["elements"] if e["id"] == "nest_plate")
    assert plate["shape"] == "slab" and plate["ramp_mm"] == 40
    # 蛋黃's home is untouched: no nest, its mat at the origin
    egg = load("default_home,no_nest")
    assert egg.nest is None and egg.mat.drawn and egg.spawn == homelayout.Pose(0, 0, 0)
    assert not any(e.id.startswith("nest_") for e in egg.elements)


def test_the_apartments_dock_stands_on_the_living_rooms_south_wall() -> None:
    layout = _apartment()
    nest = layout.nest
    assert nest is not None and nest.is_dock and layout.check() == []
    assert (nest.x_mm, nest.y_mm, nest.heading_mrad) == (-1700, -2940, 1571)
    # backin on the mirrored J-hook: the corridor wall is on the dock's right (2026-09-28)
    assert (nest.style, nest.side) == ("backin", "left")
    assert _apartment(",forward").nest.style == "forward"  # type: ignore[union-attr]
    assert layout.room_at(nest.seat.x_mm, nest.seat.y_mm) == "living"
    assert _clear_of_furniture(layout) == []
    # between the TV stand and the plant, the corner shelf far off (it blocked phase 2's exit)
    tv, plant = layout.element("liv_tv_stand"), layout.element("liv_plant")
    box = homelayout._local_box(nest.pose, nest.zones_mm()["back"])
    assert tv.footprint()[2] < box[0] and plant.footprint()[0] > box[2]
    # B1: the official dock and the blue landmark stay hidden
    assert layout.hidden == ("dock", "lm_living")
    # the toy corner is not on the dock (tidying up would put toys there)
    corner = next(z for z in layout.zones if z.id == "toy_corner")
    assert all(homelayout._overlap(corner.rect_mm, z) <= 0 for z in [box])
    assert _apartment(",backin").check() == [] and _apartment(",open_nest").check() == []


def test_the_checks_keep_the_dock_on_a_wall_and_its_lane_clear() -> None:
    src = load("default_home").to_json()
    src["nest"]["y_mm"] = -2500  # 44 cm off the wall
    codes = {(p.code, p.subject) for p in load(src).check()}
    assert ("dock_not_on_wall", "nest") in codes
    src = load("default_home").to_json()
    src["elements"].append(
        {"id": "stool", "kind": "furniture", "shape": "box", "pos_mm": [300, -2300, 0],
         "size_mm": [300, 300, 400], "colour": "#8b6a47"}
    )  # fmt: skip
    codes = {(p.code, p.subject) for p in load(src).check()}
    assert ("nest_blocked", "stool") in codes
    src = load("default_home").to_json()
    src["zones"].append({"id": "pile", "kind": "keepsake", "rect_mm": [-200, -2900, 200, -2500]})
    assert ("zone_on_nest", "pile") in {(p.code, p.subject) for p in load(src).check()}
    src = load("default_home").to_json()
    src["nest"]["kind"] = "castle"
    with pytest.raises(homelayout.LayoutError, match="充電"):
        load(src)
    src = load("default_home").to_json()
    src["nest"]["side"] = "up"
    with pytest.raises(homelayout.LayoutError, match="轉身邊"):
        load(src)
    src = load("default_home").to_json()
    del src["nest"]["side"]  # a dock that does not say: the right-hand J-hook
    assert load(src).nest.side == "right"  # type: ignore[union-attr]
    src = load("default_home,open_nest").to_json()
    src["nest"]["side"] = "left"
    with pytest.raises(homelayout.LayoutError, match="只給充電座"):
        load(src)


def test_the_status_light_is_the_backlight_of_the_lights_model() -> None:
    layout = load("default_home")
    home = lights.HomeLights.from_layout(layout)
    assert home.nest is not None and home.nest.backlight
    x, y, z = dock.light_point(layout.nest, layout.nest.dock_shape)
    assert home.nest.position == pytest.approx((x / 1000, y / 1000, z / 1000))
    # the same from the page's JSON (lights reads either)
    again = lights.HomeLights.from_layout(layout.page_json())
    assert again.nest is not None
    assert again.nest.position == pytest.approx(home.nest.position, abs=1e-6)


# ── the dock in the physics (MuJoCo) ────────────────────────────────────────────


def _world(home: str) -> tuple[Any, Any, Any]:
    """A home with the household (as the world server builds it), the duck held seated at the
    layout's spawn (its dock's seat) by its actuators."""
    from duckmind.world import worldkit

    world = worldkit.Headless(home, mjcf_dir=_mjcf())
    return world, world.kit.nest, world.layout


def _look(part: Any, world: Any) -> None:
    part._look(world.data)


@pytest.mark.parametrize("home", ["default_home,forward", "default_home"])
def test_both_styles_compile_and_their_contacts_push_nothing(home: str) -> None:
    world, part, layout = _world(home)  # (skips without MuJoCo or the official checkout)
    import mujoco

    model = world.model
    style = layout.nest.style

    def gid(name: str) -> int:
        return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))

    for name in (dock.HOUSING, dock.PANEL, dock.PLATE, dock.LIGHT, dock.TAG, dock.EDGE_TAG):
        assert gid(name) >= 0, name
    for name in dock.GUIDES:
        g = gid(f"home_{name}")
        assert g >= 0 and model.geom_contype[g] == 1 and model.geom_group[g] == 0
    probes = [dock.REAR_PROBE_TIP] if style == "backin" else [
        f"{dock.THIGH_PROBE}_left",
        f"{dock.THIGH_PROBE}_right",
    ]  # fmt: skip
    for name in probes + [f"{dock.CONTACT}_l", f"{dock.CONTACT}_r"]:
        g = gid(name)
        assert g >= 0 and model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0, name
    for name in probes:
        g = gid(name)
        assert model.geom_group[g] == dock.SENSOR_GROUP  # never a support or a blocker
        body = int(model.geom_bodyid[g])
        assert model.geom_size[g][0] > 0 and body != 0  # on the duck
    # the probes add no mass: the duck weighs what the official duck weighs
    plain = homelayout.compose(layout, mjcf_dir=_mjcf()).compile()
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    assert model.body_subtreemass[trunk] == pytest.approx(plain.body_subtreemass[trunk])
    # the forward bumper is soft; the backin face is flat
    panel = gid(dock.PANEL)
    if style == "forward":
        assert tuple(model.geom_solref[panel]) == pytest.approx(dock.BUMPER_SOLREF)
    else:
        assert model.geom_size[panel][0] * 2000 == pytest.approx(1.0)


@pytest.mark.parametrize("home", ["default_home,forward", "default_home"])
def test_seated_in_the_dock_the_contact_closes_and_reports_pg(home: str) -> None:
    world, part, layout = _world(home)
    world.run(1.5)
    _look(part, world)
    assert part.closed and part.contact_closed()
    snap = part.snapshot(world.world)
    assert snap is not None and snap["kind"] == "dock" and snap["style"] == layout.nest.style
    assert snap["pg"] and snap["charging"] and snap["pressed_mm"] > 0
    assert snap["seat_mm"] == [layout.nest.seat.x_mm, layout.nest.seat.y_mm]
    if layout.nest.style == "forward":
        pads = {probe["pad"] for probe in snap["probes"].values()}
        assert pads == {"l", "r"}  # one thigh on each pad
    else:
        assert snap["probes"]["rear"]["on_plate"]
    # the battery charges on the dock's contact (the nest's path)
    body = types.SimpleNamespace(world=world.world)
    assert part.pg_op(body, {"op": "pg"}) == {"pg": True}  # a sensor: nothing else
    world.run(1.0)
    assert world.kit.battery.snapshot(world.world)["pg"]  # (a full pack does not charge more)
    off = part.nest_op(body, {"do": "power", "on": False})
    assert off["contact"] and not off["pg"]
    assert part.pg_op(body, {}) == {"pg": False}
    page = part.nest_op(body, {"do": "page"})
    assert page["kind"] == "dock" and len(page["contacts"]) == 2


@pytest.mark.parametrize("home", ["default_home,forward", "default_home"])
def test_the_contact_opens_when_the_duck_is_lifted_or_moved_out(home: str) -> None:
    world, part, layout = _world(home)
    world.run(1.0)
    _look(part, world)
    assert part.closed
    qadr = world.robot.qadr
    world.data.qpos[qadr + 2] += 0.03  # 3 cm up (standing lifts the thighs and the rear)
    import mujoco

    mujoco.mj_forward(world.model, world.data)
    _look(part, world)
    assert not part.closed
    # set back down 15 cm further into the room: off the pads, or the rear off the plates
    x, y = layout.nest.to_world(dock.seat_x_mm(layout.nest.style, layout.nest.dock_shape) + 150, 0)
    world.data.qpos[qadr : qadr + 3] = [x / 1000, y / 1000, world.data.qpos[qadr + 2] - 0.03]
    mujoco.mj_forward(world.model, world.data)
    _look(part, world)
    assert not part.closed and part.makes >= 1 and part.breaks >= 1


def test_the_hand_puts_the_duck_into_the_backin_dock_facing_the_room() -> None:
    world, part, layout = _world("default_home,backin")
    world.data.qpos[world.robot.qadr : world.robot.qadr + 2] = [0.6, -1.8]
    world.run(1.0)
    _look(part, world)
    assert not part.closed
    answer = world.kit.hand.command(world.world, {"do": "place", "to": "nest"})
    assert answer["ok"] and any("充電座" in text for text in answer["why_zh"])
    seat = layout.nest.seat
    assert answer["target"]["trunk_mm"] == [seat.x_mm, seat.y_mm]
    assert answer["target"]["yaw_mrad"] == seat.heading_mrad
    while world.kit.hand.script is not None:
        world.step()
    world.run(1.5)
    assert (world.kit.hand.last or {}).get("ok")
    _look(part, world)
    assert part.closed and part.pg_op(world, {}) == {"pg": True}
    heading = worldkit_heading(world)
    assert abs(math.remainder(heading - seat.yaw, math.tau)) < math.radians(10)


def worldkit_heading(world: Any) -> float:
    from duckmind.world import worldkit

    return worldkit.heading_of([float(v) for v in world.data.xquat[world.robot.trunk]])


def test_the_open_nest_variant_still_charges_on_its_pad() -> None:
    world, part, layout = _world("default_home,open_nest")
    world.run(1.5)
    _look(part, world)
    assert layout.nest is not None and not layout.nest.is_dock
    assert part.dock is None and part.closed and part.snapshot(world.world)["kind"] == "open"


def test_the_face_tag_reads_as_tag_17_from_the_room() -> None:
    """The camera sees the physics tag: a face-on render of the dock's face, sampled cell by
    cell, decodes to the dock's id with no bit wrong and no turn needed."""
    world, part, layout = _world("default_home")
    import mujoco

    from duckmind.world import camera_model

    camera_model.import_mujoco()  # MUJOCO_GL=cgl before the first context on this Mac
    model, data = world.model, world.data
    data.qpos[world.robot.qadr : world.robot.qadr + 2] = [2.5, 0.0]  # the duck out of the way
    mujoco.mj_forward(model, data)
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), 400)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), 400)
    try:
        renderer = mujoco.Renderer(model, 400, 400)
    except Exception as error:  # no offscreen GL here
        pytest.skip(f"no offscreen GL context: {type(error).__name__}: {error}")
    nest, g = layout.nest, layout.nest.dock_shape
    cx, cy = nest.to_world(g.face_mm + 1, 0)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [cx / 1000, cy / 1000, (g.panel_z0_mm + g.panel_z1_mm) / 2000]
    camera.distance, camera.elevation = 0.3, 0.0
    camera.azimuth = math.degrees(nest.yaw) + 180.0  # looking at the face from the room
    renderer.update_scene(data, camera=camera)
    image = renderer.render()
    renderer.close()
    gray = image.mean(axis=2)
    px_per_m = 400 / (2 * 0.3 * math.tan(math.radians(22.5)))
    side = g.tag_mm / 1000 * px_per_m
    cells = []
    for row in range(8):
        line = []
        for col in range(8):
            y = 200 - side / 2 + (row + 0.5) * side / 8
            x = 200 - side / 2 + (col + 0.5) * side / 8
            line.append(1 if gray[int(y), int(x)] > 128 else 0)
        cells.append(line)
    assert apriltag.decode(cells, max_errors=0) == (apriltag.NEST_TAG_ID, 0, 0)


def test_the_status_light_glows_in_the_dark_and_turns_green_while_charging() -> None:
    """C4 on the dock: the camera's emission pass lights the status light only in the dark, in
    the charging colour while the duck charges."""
    pytest.importorskip("numpy")
    import numpy as np

    from duckmind.world import camera_model as cm
    from duckmind.world import render_scene as rs

    folder = _mjcf()
    layout = load("default_home", mjcf_dir=folder)
    home = lights.HomeLights.from_layout(layout)
    try:
        eye = rs.HomeEye.open(homelayout.compose(layout), home=home)
    except Exception as error:
        pytest.skip(f"no offscreen GL context here: {type(error).__name__}: {error}")
    assert list(eye.eye.emitters) == [dock.LIGHT]
    try:
        import mujoco

        m, d = eye.eye.model, eye.eye.data
        free = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
        for j in range(m.njnt):
            if m.jnt_type[j] == 0 and j != free:
                d.qpos[m.jnt_qposadr[j] + 2] = -10.0
        pose = cm.keyframe_pose(eye.eye, "STAND")
        x, y = layout.nest.to_world(900, 0)
        yaw = layout.nest.yaw + math.pi
        pose["trunk"] = [x / 1000, y / 1000, pose["trunk"][2]]
        pose["quat"] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        eye.set_pose(pose)
        width, height = eye.frame_size
        light = eye.camera.renderer.segment_ids(width, height) == mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_GEOM, dock.LIGHT
        )
        assert light.sum() > 300  # the strip is in view from 0.9 m
        from datetime import datetime

        seen = {}
        for key, when, charging in (
            ("day", datetime(2026, 9, 27, 12, 0), False),
            ("moon", datetime(2026, 9, 28, 2, 0), False),
            ("moon_charging", datetime(2026, 9, 28, 2, 0), True),
        ):
            state = lights.home_light_state(when, home, charging=charging)
            eye.converge(state)
            capture = eye.capture(state)
            assert capture.glow == key.startswith("moon")
            rgb = capture.rgb.astype(np.float64)
            seen[key] = (rgb[light].mean(axis=0), rgb[~light].mean(axis=0))
        night, rest = seen["moon"]
        green, _ = seen["moon_charging"]
        assert night.mean() > 2 * rest.mean()  # the brightest thing in the dark
        assert green[1] - green[0] > 40  # charging: green, not the cool white
        assert abs(night[1] - night[0]) < 40
    finally:
        eye.close()


def test_a_duck_spawned_in_the_apartment_dock_sits_charging() -> None:
    _apartment()
    world, part, layout = _world("apartment")
    world.run(1.5)
    _look(part, world)
    seat = layout.nest.seat
    trunk = [v * 1000 for v in world.trunk[:2]]
    assert math.dist(trunk, [seat.x_mm, seat.y_mm]) < 15 and part.closed
    assert world.tilt < math.radians(8)


def test_the_x_world_dock_section_has_what_the_measurements_need() -> None:
    world, part, layout = _world("default_home,backin")
    world.run(1.0)
    _look(part, world)
    lock = threading.Lock()
    with lock:
        snap = part.snapshot(world.world)
    assert snap is not None
    for key in (
        "kind",
        "style",
        "pos_mm",
        "heading_mrad",
        "seat_mm",
        "seat_heading_mrad",
        "pad_top_mm",
        "contact",
        "pg",
        "pressed_mm",
        "probes",
        "bumper",
        "makes",
        "breaks",
        "tags",
        "backlight",
    ):
        assert key in snap, key
    assert snap["backlight"] == dock.LIGHT and snap["pad_top_mm"] == 4
    assert snap["tags"][0]["id"] == apriltag.NEST_TAG_ID
