"""Phase 2's open nest in the physics world (duckmind.world.nestpad; the ``open_nest`` variants
since the charging dock became the default, see tests/test_world_dock.py) and its tags
(duckmind.world.apriltag): owner decisions C1–C4.

Pure tests for the tag family run everywhere; the MuJoCo cases (the nest built into a home,
the contact closing when the duck sits and opening when it stands, the rendered tag decoding
to its id) skip without MuJoCo or the official checkout and run in the render check."""

from __future__ import annotations

import itertools
import math
import os
import threading
import types
from pathlib import Path
from typing import Any

import pytest

from duckmind.world import apriltag, nestpad

MJCF = os.environ.get("DUCKMIND_MICRODUCK_MJCF")


def _mjcf() -> str:
    if not MJCF or not Path(MJCF, "scene_allcollisions.xml").is_file():
        pytest.skip("DUCKMIND_MICRODUCK_MJCF (official microduck MJCF folder) is not set")
    pytest.importorskip("mujoco")
    pytest.importorskip("numpy")
    return MJCF


# ── the tag family ──────────────────────────────────────────────────────────────


def test_the_bit_layout_is_apriltag3s_four_turned_quarters():
    cells = apriltag.bit_cells()
    assert len(cells) == 36 and len(set(cells)) == 36
    assert {x for x, _ in cells} == set(range(1, 7)) == {y for _, y in cells}
    assert cells[:9] == ((1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (2, 2), (3, 2), (4, 2), (3, 3))
    for k in range(9, 36):  # each quarter is the one before, a quarter turn clockwise
        x, y = cells[k - 9]
        assert cells[k] == (7 - y, x)


def test_the_kept_codes_are_the_libraries_first_codes_and_a_real_distance_11_family():
    assert apriltag.CODES[:3] == (0xD7E00984B, 0xDDA664CA7, 0xDC4A1C821)
    assert len(apriltag.CODES) == 64 and all(0 <= c < 1 << 36 for c in apriltag.CODES)
    grids = [apriltag.grid(i, white_border=False) for i in range(len(apriltag.CODES))]

    def turned(cells: Any, turns: int) -> Any:
        for _ in range(turns):
            cells = apriltag.rotate(cells)
        return cells

    def distance(a: Any, b: Any) -> int:
        return sum(x != y for ra, rb in zip(a, b, strict=True) for x, y in zip(ra, rb, strict=True))

    worst = min(
        distance(grids[i], turned(grids[j], r))
        for i, j in itertools.product(range(len(grids)), repeat=2)
        for r in range(4)
        if not (i == j and r == 0)
    )
    assert worst >= apriltag.MIN_HAMMING


def test_a_tag_as_printed_has_a_black_square_in_a_white_border():
    cells = apriltag.grid(apriltag.NEST_TAG_ID)
    assert len(cells) == 10 and all(len(row) == 10 for row in cells)
    assert all(v == 1 for v in cells[0] + cells[9]) and all(r[0] == r[9] == 1 for r in cells)
    inner = [row[1:9] for row in cells[1:9]]
    assert all(v == 0 for v in inner[0] + inner[7]) and all(r[0] == r[7] == 0 for r in inner)
    assert apriltag.read_code(cells) == apriltag.code(apriltag.NEST_TAG_ID)


@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_a_seen_tag_decodes_to_its_id_in_any_rotation(turns):
    seen = apriltag.grid(apriltag.EDGE_TAG_ID)
    for _ in range(turns):
        seen = apriltag.rotate(seen)
    found = apriltag.decode(seen)
    assert found is not None
    tag_id, upright_turns, errors = found
    assert tag_id == apriltag.EDGE_TAG_ID and errors == 0
    assert (turns + upright_turns) % 4 == 0


def test_two_flipped_bits_still_decode_and_noise_does_not():
    cells = [list(row) for row in apriltag.grid(apriltag.NEST_TAG_ID)]
    cells[3][3] ^= 1
    cells[5][6] ^= 1
    assert apriltag.decode(cells) == (apriltag.NEST_TAG_ID, 0, 2)
    assert apriltag.decode([[1] * 10 for _ in range(10)]) is None


def test_the_texture_is_the_black_square_at_the_asked_resolution():
    width, height, rgb = apriltag.texture_rgb(apriltag.NEST_TAG_ID, px_per_cell=4)
    assert (width, height, len(rgb)) == (32, 32, 32 * 32 * 3)
    assert rgb[:3] == b"\x00\x00\x00"  # the black border's corner
    with pytest.raises(ValueError):
        apriltag.code(999)
    page = apriltag.page_json(17)
    assert page["label"] == "tag36h11:17" and len(page["cells"]) == 10


def test_the_page_draws_the_same_bits_as_the_physics():
    from duckmind.world import homelayout

    layout = homelayout.load("default_home,open_nest")
    page = nestpad.page_json(layout.nest)
    assert page["present"] and [t["id"] for t in page["tags"]] == [17, 18]
    assert page["tags"][0]["cells"] == [list(r) for r in apriltag.grid(17)]
    assert page["tags"][0]["size_mm"] == 100 and page["tags"][0]["mount_mm"] == 125
    assert page["electrode"] == {"radius_mm": 85, "top_mm": 5}
    assert nestpad.page_json(None) == {"present": False}


# ── the nest in a home (MuJoCo) ─────────────────────────────────────────────────


def _nest_world(home: str) -> tuple[Any, Any, Any, Any]:
    folder = _mjcf()  # skips without MuJoCo or the official checkout
    import mujoco

    from duckmind.world import homelayout, worldkit

    layout = homelayout.load(home, mjcf_dir=folder)
    spec = homelayout.compose(layout, mjcf_dir=folder)
    plain = spec.compile()
    part = nestpad.NestPad()
    worldkit.add_spot_probe(spec)
    part.edit_spec(spec, layout)
    homelayout.fix_keyframes(spec, layout)
    model = spec.compile()
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)
    world = types.SimpleNamespace(model=model, data=data, lock=threading.Lock(), bodies=[])
    return part, world, plain, layout


def _settle(part: Any, world: Any, keyframe: str, seconds: float) -> None:
    import mujoco

    model, data = world.model, world.data
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
    mujoco.mj_resetDataKeyframe(model, data, key)
    data.ctrl[:] = model.key_ctrl[key]
    part.ready(world, None)
    for _ in range(int(seconds / 0.005)):
        mujoco.mj_step(model, data)
        part.after_step(world)


@pytest.mark.parametrize("variant", [",open_nest", ",edge_3mm", ",ramp"])
def test_the_contact_closes_when_the_duck_sits_and_opens_when_it_stands(variant):
    part, world, _, _ = _nest_world("default_home" + variant)
    _settle(part, world, "SIT", 1.5)
    closed, gap, pin = part.measure(world.data)
    assert closed and gap is not None and -0.012 < gap < 0.0  # pressed in, within the travel
    assert pin is not None and math.hypot(*pin) < 40  # over the middle of the pad
    snap = part.snapshot(world)
    assert snap is not None and snap["contact"] and snap["pg"] and snap["makes"] >= 1
    _settle(part, world, "STAND", 0.3)
    closed, gap, _ = part.measure(world.data)
    assert not closed and gap is not None and gap > 0.03


def test_the_pin_and_the_nest_change_no_physics():
    part, world, plain, _ = _nest_world("default_home,open_nest")
    import mujoco

    model = world.model
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    assert model.body_subtreemass[trunk] == pytest.approx(plain.body_subtreemass[trunk])
    for name in (nestpad.PIN_TIP, nestpad.PIN_SHAFT, nestpad.ELECTRODE, nestpad.BACKLIGHT):
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert geom >= 0 and model.geom_contype[geom] == 0 and model.geom_conaffinity[geom] == 0
    pin = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, nestpad.PIN_TIP)
    assert model.geom_group[pin] == nestpad.SENSOR_GROUP
    # seated at the same height with and without the nest's extras
    data = mujoco.MjData(plain)
    key = mujoco.mj_name2id(plain, mujoco.mjtObj.mjOBJ_KEY, "SIT")
    mujoco.mj_resetDataKeyframe(plain, data, key)
    data.ctrl[:] = plain.key_ctrl[key]
    plain.opt.timestep = 0.005
    for _ in range(300):
        mujoco.mj_step(plain, data)
    _settle(part, world, "SIT", 1.5)
    assert world.data.qpos[2] == pytest.approx(float(data.qpos[2]), abs=2e-4)


def test_pg_is_a_sensor_and_the_nest_op_is_truth():
    part, world, _, _ = _nest_world("default_home,open_nest")
    _settle(part, world, "SIT", 1.0)
    body = types.SimpleNamespace(world=world)
    assert part.pg_op(body, {"op": "pg"}) == {"pg": True}  # nothing else: a pin reads a level
    off = part.nest_op(body, {"op": "x_nest", "do": "power", "on": False})
    assert off["contact"] and not off["pg"] and not off["charging"]
    assert part.pg_op(body, {}) == {"pg": False}
    page = part.nest_op(body, {"do": "page"})
    assert page["tags"][1]["id"] == apriltag.EDGE_TAG_ID
    with pytest.raises(ValueError):
        part.nest_op(body, {"do": "explode"})


def test_a_home_without_a_nest_has_no_pin_and_pg_reads_false():
    part, world, _, _ = _nest_world("default_home,no_nest")
    import mujoco

    assert mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_GEOM, nestpad.PIN_TIP) < 0
    _settle(part, world, "SIT", 0.5)
    assert not part.present and part.snapshot(world) is None
    assert part.pg_op(types.SimpleNamespace(world=world), {}) == {"pg": False}


def test_the_rendered_backboard_tag_reads_as_tag_17_upright():
    """The camera sees the physics tag: a face-on render, sampled cell by cell, decodes to the
    nest's id with no bit wrong and no turn needed."""
    part, world, _, layout = _nest_world("default_home,open_nest")
    import mujoco

    from duckmind.world import camera_model

    camera_model.import_mujoco()  # MUJOCO_GL=cgl before the first context on this Mac
    model, data = world.model, world.data
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "SIT")
    mujoco.mj_resetDataKeyframe(model, data, key)
    data.qpos[0] = 2.5  # the duck out of the way
    mujoco.mj_forward(model, data)
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), 400)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), 400)
    renderer = mujoco.Renderer(model, 400, 400)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    nest = layout.nest
    front = (-nest.board_offset_mm + nest.board_size_mm[0] / 2) / 1000.0
    camera.lookat[:] = [front, 0.0, 0.12]
    camera.distance, camera.azimuth, camera.elevation = 0.3, 180.0, 0.0
    renderer.update_scene(data, camera=camera)
    image = renderer.render()
    renderer.close()
    gray = image.mean(axis=2)
    # the tag's 100 mm black square is centred; fovy 45° at 0.3 m spans 0.2485 m in 400 px
    px_per_m = 400 / (2 * 0.3 * math.tan(math.radians(22.5)))
    side = 0.1 * px_per_m
    cells = []
    for row in range(8):
        line = []
        for col in range(8):
            y = 200 - side / 2 + (row + 0.5) * side / 8
            x = 200 - side / 2 + (col + 0.5) * side / 8
            line.append(1 if gray[int(y), int(x)] > 128 else 0)
        cells.append(line)
    assert apriltag.decode(cells, max_errors=0) == (apriltag.NEST_TAG_ID, 0, 0)
