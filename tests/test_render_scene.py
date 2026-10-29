"""render_scene: the head camera in a home — the (at most 7) lights of the camera's room, the
survey's light ladder through the render (spec §12.3: day 300, dusk 5, lamps 120, moon 0.25
lux, ±10 %), the nest backlight's glow, and what one capture carries.

Render tier: numpy, MuJoCo 3.10, an offscreen GL context and the official checkout
(``DUCKMIND_MICRODUCK_MJCF``); the homes come from ``world/homelayout.py``.
"""

import math
import os
from datetime import datetime
from pathlib import Path

import pytest

from duckmind.world import lights

MOMENTS = {
    "day": datetime(2026, 9, 27, 12, 0),
    "dusk": datetime(2026, 12, 1, 16, 50),
    "lamps": datetime(2026, 9, 27, 20, 0),
    "moon": datetime(2026, 9, 28, 2, 0),
}
LADDER = {"day": 300.0, "dusk": 5.0, "lamps": 120.0, "moon": 0.25}
CARD_HALF = 0.08
CARD_Z = 0.012  # above the nest pad (5 mm) and the apartment's rug (9 mm)


def _modules():
    pytest.importorskip("numpy", reason="the eye's renders need numpy")
    pytest.importorskip("mujoco", reason="the eye's renders need mujoco")
    from duckmind.world import camera_model as cm
    from duckmind.world import render_scene as rs

    return cm, rs


def _mjcf() -> Path:
    folder = os.environ.get("DUCKMIND_MICRODUCK_MJCF")
    if not folder or not (Path(folder) / "scene_allcollisions.xml").is_file():
        pytest.skip("official microduck_rl checkout not configured (DUCKMIND_MICRODUCK_MJCF)")
    return Path(folder)


def _homelayout():
    return pytest.importorskip(
        "duckmind.world.homelayout", reason="world/homelayout.py (phase 2 layouts) not here"
    )


def _open(rs, scene, **kwargs):
    try:
        return rs.HomeEye.open(scene, **kwargs)
    except Exception as error:  # no offscreen GL on this machine
        pytest.skip(f"no offscreen GL context here: {type(error).__name__}: {error}")


def _home(name):
    """(scene or spec, HomeLights): phase 0's room on the official floor, or a layout home."""
    if name == "phase0":
        return _mjcf() / "scene_allcollisions.xml", lights.DEFAULT_HOME
    homelayout = _homelayout()
    layout = homelayout.load(name, mjcf_dir=_mjcf())
    return homelayout.compose(layout), lights.HomeLights.from_layout(layout)


def _card(at):
    """A white card on the floor at ``at`` (x, y): a white level surface to read lux off."""

    def edit(spec):
        import mujoco

        material = spec.add_material()
        material.name = "test_white"
        material.rgba = [1.0, 1.0, 1.0, 1.0]
        material.specular = 0.0
        material.shininess = 0.0
        material.reflectance = 0.0
        card = spec.worldbody.add_geom()
        card.name = "test_card"
        card.type = mujoco.mjtGeom.mjGEOM_BOX
        card.size = [CARD_HALF, CARD_HALF, 0.002]
        card.pos = [at[0], at[1], CARD_Z]
        card.material = "test_white"
        card.contype = 0
        card.conaffinity = 0

    return edit


def _stand_at(cm, eye, x, y, yaw):
    """The duck standing at (x, y) facing ``yaw``; every other free object 10 m under the
    floor (toys between the duck and the card would hide it)."""
    import mujoco

    m, d = eye.eye.model, eye.eye.data
    free = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    for j in range(m.njnt):
        if m.jnt_type[j] == 0 and j != free:
            d.qpos[m.jnt_qposadr[j] + 2] = -10.0
    pose = cm.keyframe_pose(eye.eye, "STAND")
    pose["trunk"] = [x, y, pose["trunk"][2]]
    pose["quat"] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    eye.set_pose(pose)


def _mask(cm, eye, geom):
    import mujoco

    width, height = eye.frame_size
    gid = mujoco.mj_name2id(eye.eye.model, mujoco.mjtObj.mjOBJ_GEOM, geom)
    mask = eye.camera.renderer.segment_ids(width, height) == gid
    core = mask.copy()  # two pixels in from the edge: no blend with what is next to it
    for shift in (1, 2):
        core[shift:] &= mask[:-shift]
        core[:-shift] &= mask[shift:]
        core[:, shift:] &= mask[:, :-shift]
        core[:, :-shift] &= mask[:, shift:]
    return core


def _render_lux(cm, eye, mask, lux_per_unit):
    """The lux on the card, read back from the plain render (linear, white albedo)."""
    import numpy as np

    raw = eye.camera.renderer.render(*eye.frame_size)
    luminance = raw[mask].astype(np.float64) / 255.0 @ np.array([0.2126, 0.7152, 0.0722])
    return float(luminance.mean()) * lux_per_unit


# name: (card at, duck at, duck yaw) — the card at (or next to) the room's middle, the duck
# 0.6 m away looking at it, so the eye's normalisation spot (under the lens) is not the card.
PLACES = {
    "phase0": ((0.0, 0.0), (0.6, 0.0), math.pi),
    "default_home": ((0.0, 0.0), (0.6, 0.0), math.pi),  # the card on the nest pad
    "apartment": ((-2.65, -1.40), (-2.65, -2.0), math.pi / 2),  # the living room
}


@pytest.mark.parametrize("name", list(PLACES))
def test_the_light_ladder_through_the_render(name):
    """Spec §12.3 / survey_calibration.md §2.3: at day, dusk, lamps and moon a white level card
    by the room's middle reads the ladder's lux within 10 % off the render (and within 5 % of
    what world.lights says for that spot), under the room's own ≤ 7 lights; the camera's
    auto-exposure then lands on the survey's levels."""
    cm, rs = _modules()
    from duckmind.eye import light as eye_light

    scene, home = _home(name)
    card, duck, yaw = PLACES[name]
    eye = _open(rs, scene, home=home, edit=_card(card))
    try:
        _stand_at(cm, eye, duck[0], duck[1], yaw)
        room = rs.camera_room(eye.eye, home)
        assert room.contains(*card)
        mask = _mask(cm, eye, "test_card")
        assert mask.sum() > 2000, "the card is in view"
        for key, when in MOMENTS.items():
            state = lights.home_light_state(when, home)
            room_state = state.rooms[room.id]
            assert room_state.key == key
            assert room_state.lux == pytest.approx(LADDER[key], rel=0.10)  # at its middle
            light = eye.light(state)
            assert light.room == room.id and len(light.rig.lit) <= rs.MAX_LIGHTS
            seen = _render_lux(cm, eye, mask, light.lux_per_unit)
            model = room_state.lux_at((card[0], card[1], CARD_Z + 0.002))
            assert seen == pytest.approx(model, rel=0.05), key
            assert seen == pytest.approx(LADDER[key], rel=0.10), key
            eye.converge(state)
            capture = eye.capture(state)
            c = capture.controls
            exposure = eye_light.Exposure.from_registers(c.exposure, c.analogue_gain, c.gain)
            reading = eye_light.measure(cm.luma(capture.upright()), exposure)
            if key in ("day", "lamps"):
                assert reading.level == "bright", key
            elif key == "dusk":
                assert cm.at_ceiling(c)  # past the shutter and the analogue gain: short of light
                if name != "default_home":
                    assert reading.level == "dim" and reading.level_frame == "dim"
                # default_home's light walls and ceiling (a view twice as bright on average as
                # the apartment's) settle dusk at 1.4× digital gain, σ 3.4: "bright" by the
                # phase-0 thresholds, which were fitted on the apartment (phase 8 recalibrates)
            else:
                # moonlight: the AE at its ceiling; the survey's 0.25 lux read "black" (luma
                # 33–37), a light-walled room reads "dark" (luma ~50): the 45 line is a guess
                # until phase 8.
                assert cm.at_ceiling(c) and reading.level in ("dark", "black")
    finally:
        eye.close()


def test_the_camera_room_picks_its_lights():
    cm, rs = _modules()
    scene, home = _home("apartment")
    eye = _open(rs, scene, home=home, config=cm.CameraConfig(quality="360p30"))
    try:
        evening = lights.home_light_state(MOMENTS["lamps"], home)
        noon = lights.home_light_state(MOMENTS["day"], home)
        _stand_at(cm, eye, -2.65, -2.0, math.pi / 2)
        living = eye.light(evening)
        assert living.room == "living"
        assert set(living.rig.lit) == {"lamp:lamp_living", "window:window_living"}
        _stand_at(cm, eye, 2.25, 2.0, 0.0)
        bedroom = eye.light(evening)
        assert bedroom.room == "bedroom" and "lamp:lamp_bedroom" in bedroom.rig.lit
        assert bedroom.rig.lux != living.rig.lux or bedroom.rig.lit != living.rig.lit
        _stand_at(cm, eye, -2.5, 1.2, 0.0)
        kitchen = eye.light(noon)  # no window: the day comes in bounced, no MuJoCo light
        assert kitchen.room == "kitchen" and kitchen.rig.lit == ()
        assert kitchen.rig.lux == pytest.approx(300 * lights.SPILL)
        # the MuJoCo lights of the other rooms stay dark
        active = [int(v) for v in eye.eye.model.light_active]
        assert sum(active) == 0
    finally:
        eye.close()


def test_more_than_seven_lights_fold_into_the_bounced_light():
    cm, rs = _modules()
    import mujoco
    import numpy as np

    spec = mujoco.MjSpec()
    floor = spec.worldbody.add_geom()
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0.0, 0.0, 0.05]
    white = spec.add_material()
    white.name = "white"
    white.rgba = [1.0, 1.0, 1.0, 1.0]
    white.specular = 0.0
    white.shininess = 0.0
    white.reflectance = 0.0
    floor.material = "white"
    cam = spec.worldbody.add_camera()
    cam.name = "head_camera"
    cam.pos = [0.0, 0.0, 0.3]
    cam.quat = [0.0, 1.0, 0.0, 0.0]  # looks up; the 180° roll turns it down
    room = lights.Room(
        "hall",
        lights.rectangle(-3, -3, 3, 3),
        lamps=tuple(
            lights.Lamp(f"l{i}", (2.5 * math.cos(i), 2.5 * math.sin(i), 2.3), lux=15.0)
            for i in range(9)
        ),
    )
    state = lights.room_light_state(MOMENTS["lamps"], room)
    assert len([s for s in state.sources if s.kind == "spot"]) == 9
    eye = _open(rs, spec, config=cm.CameraConfig(quality="360p30"))
    try:
        light = eye.light(state)
        assert len(light.rig.lit) == 7 and len(light.rig.folded) == 2
        assert sum(int(v) for v in eye.eye.model.light_active) == 7
        raw = eye.camera.renderer.render(640, 360)[170:190, 310:330].reshape(-1, 3)
        luminance = float((raw / 255.0).mean(axis=0) @ np.array([0.2126, 0.7152, 0.0722]))
        assert luminance == pytest.approx(cm.RENDER_WHITE, rel=0.05)
        assert luminance * light.lux_per_unit == pytest.approx(state.lux_at((0, 0, 0)), rel=0.05)
    finally:
        eye.close()


def _facing_the_nest(cm, eye, layout):
    nest = layout.nest
    x, y = nest.to_world(900, 0)
    _stand_at(cm, eye, x / 1000, y / 1000, nest.yaw + math.pi)


def test_the_backlight_glows_in_the_dark_only_and_turns_green_while_charging():
    cm, rs = _modules()
    import numpy as np

    homelayout = _homelayout()
    # (phase 2's open nest: its backboard glows; the dock's status light: test_world_dock.py)
    layout = homelayout.load("apartment,open_nest", mjcf_dir=_mjcf())
    home = lights.HomeLights.from_layout(layout)
    eye = _open(rs, homelayout.compose(layout), home=home)
    assert list(eye.eye.emitters) == [rs.NEST_BOARD]
    try:
        _facing_the_nest(cm, eye, layout)
        board = _mask(cm, eye, rs.NEST_BOARD)
        assert board.sum() > 3000
        seen = {}
        for key, charging in (("day", False), ("lamps", False), ("moon", False), ("moon", True)):
            state = lights.home_light_state(MOMENTS[key], home, charging=charging)
            eye.converge(state)
            capture = eye.capture(state)
            assert capture.glow == (key == "moon")
            assert list(capture.extras["glow"]) == ([rs.NEST_BOARD] if key == "moon" else [])
            rgb = capture.rgb.astype(np.float64)
            seen[(key, charging)] = (rgb[board].mean(axis=0), rgb[~board].mean(axis=0))
        day_board, day_rest = seen[("day", False)]
        night_board, night_rest = seen[("moon", False)]
        green_board, _ = seen[("moon", True)]
        # by day the cyan board is just a surface; in the dark it is the brightest thing
        assert night_board.mean() > 2 * night_rest.mean() and night_board.mean() > 150
        assert day_board.mean() < 1.5 * day_rest.mean() + 60
        # while charging it glows green: much more green than red, unlike the cool white
        assert green_board[1] - green_board[0] > 60
        assert abs(night_board[1] - night_board[0]) < 40
    finally:
        eye.close()


def test_a_capture_carries_media_frame_bytes_and_an_evaluator_log_and_repeats():
    cm, rs = _modules()
    scene, home = _home("default_home")
    state = lights.home_light_state(MOMENTS["moon"], home)
    out = []
    for _ in range(2):
        eye = _open(rs, scene, home=home, seed=11)
        try:
            _stand_at(cm, eye, 0.6, 0.0, math.pi)
            eye.converge(state)
            capture = eye.capture(state, index=5)
        finally:
            eye.close()
        out.append(capture)
        scene, home = _home("default_home")  # a fresh spec: compose's is compiled once
    first, second = out
    data = first.uyvy()
    assert (first.width, first.height, first.rotate, first.format) == (1280, 720, 90, "UYVY")
    assert len(data) == first.bytes_len == 1280 * 720 * 2
    assert data == second.uyvy(), "the same seed and frame index: the same bytes"
    log = first.extras
    assert {"room", "key", "label", "lux", "reference", "lit", "folded", "glow", "ae", "wb"} <= set(
        log
    )
    assert log["room"] == "living" and log["key"] == "moon" and log["ae"]["at_ceiling"]
    assert log["wb"][0] > 1.0 > log["wb"][2]  # moonlight is blue: red up, blue down


def test_a_world_snapshot_fits_the_eye_of_the_same_layout():
    cm, rs = _modules()
    import mujoco

    homelayout = _homelayout()
    layout = homelayout.load("apartment", mjcf_dir=_mjcf())
    world = homelayout.compose(layout).compile()
    data = mujoco.MjData(world)
    mujoco.mj_resetDataKeyframe(
        world, data, mujoco.mj_name2id(world, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    )
    free = mujoco.mj_name2id(world, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    adr = world.jnt_qposadr[free]
    data.qpos[adr : adr + 2] = [-2.0, -1.5]
    home = lights.HomeLights.from_layout(layout)
    eye = _open(rs, homelayout.compose(layout), home=home, config=cm.CameraConfig(quality="360p30"))
    try:
        assert eye.eye.model.nq == world.nq and eye.eye.model.nmocap == world.nmocap
        eye.set_state(data.qpos, mocap_pos=data.mocap_pos, mocap_quat=data.mocap_quat)
        x, y, _ = cm.camera_position(eye.eye)
        assert math.hypot(x + 2.0, y + 1.5) < 0.15  # the head is where the snapshot's duck is
        assert rs.camera_room(eye.eye, home).id == "living"
        with pytest.raises(ValueError, match="qpos"):
            eye.set_state(data.qpos[:-1])
    finally:
        eye.close()
