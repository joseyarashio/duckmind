"""Home layouts (phase 2, part A): the page's room made physical, the hazard home and the
official apartment overlay — one layout per home for the physics, the page and the sensors.

The layout checks, the page JSON and the official apartment's conversion are plain Python (the
apartment needs the official checkout, ``DUCKMIND_MICRODUCK_MJCF``). The MuJoCo checks (every
home compiles, the duck spawns clear, things lie still, page == physics, keyframes, the stair
gate) need numpy + MuJoCo 3.10: they skip without them and run in the render check
(docs/validation/run-duck-simulator.md).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from duckmind.world import apartment, homelayout
from duckmind.world.homelayout import Layout, load

ROOT = Path(__file__).resolve().parents[1]
MJCF = os.environ.get("DUCKMIND_MICRODUCK_MJCF")
HOMES = ["default_home", "hazard_home", "apartment", "plain_home", "real_home"]
EVERY = [
    "default_home",
    "default_home,backin",
    "default_home,forward",
    "default_home,no_nest",
    "default_home,open_nest",
    "default_home,edge_3mm",
    "default_home,ramp",
    "hazard_home",
    "real_home",
    "real_home,backin",
    "real_home,forward",
    "real_home,no_nest",
    "apartment",
    "apartment,backin",
    "apartment,forward",
    "apartment_ungated_test",
    "apartment,no_nest",
    "apartment,open_nest",
    "apartment,ramp",
]


def _official() -> Path:
    if not MJCF or not (Path(MJCF) / "apartment.xml").is_file():
        pytest.skip("DUCKMIND_MICRODUCK_MJCF (official microduck MJCF folder) is not set")
    return Path(MJCF)


def _load(name: str) -> Layout:
    if name.startswith("apartment"):
        _official()
    return load(name)


# ── the layouts (no MuJoCo) ────────────────────────────────────────────────────


@pytest.mark.parametrize("name", EVERY)
def test_every_home_passes_its_generation_checks(name: str) -> None:
    layout = _load(name)
    assert layout.check() == []
    assert layout.warnings == ()  # the official files are the reviewed ones
    assert {e.id for e in layout.elements}.isdisjoint({t.id for t in layout.things})


def test_the_homes_and_their_variants_are_listed() -> None:
    assert set(homelayout.available()) >= {*HOMES, "apartment_ungated_test"}
    docks = ["backin", "edge_3mm", "forward", "no_nest", "open_nest", "ramp"]
    assert homelayout.variants("default_home") == docks
    assert homelayout.variants("apartment") == [*docks, "ungated_test"]
    with pytest.raises(homelayout.LayoutError, match="沒有叫"):
        load("castle")
    with pytest.raises(homelayout.LayoutError, match="變體"):
        load("default_home", variant="balcony")


def test_the_json_uses_integer_millimetres_and_milliradians() -> None:
    def walk(node: Any, key: str = "") -> list[str]:
        bad = []
        if isinstance(node, dict):
            for k, v in node.items():
                bad += walk(v, k)
        elif isinstance(node, list):
            for v in node:
                bad += walk(v, key)
        elif key.endswith(("_mm", "_mrad", "_g")) and (
            isinstance(node, bool) or not isinstance(node, int)
        ):
            bad.append(f"{key}={node!r}")
        return bad

    for path in sorted(homelayout.HOMES_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["schema"] == homelayout.SCHEMA
        assert walk(data) == [], path.name


# The phase-0 page's room (duck3d.js buildRoom, three.js y-up): world (x, y, z) = three
# (x, −z, y). (id, world footprint centre x, y, lowest z, full size x, y, z) — the sofa seat
# now starts 9 cm up, on legs (curiosity spec §4.2).
PAGE_ROOM = [
    ("wall_n", 0, 3000, 0, 6000, 80, 1200),
    ("wall_w", -3000, 0, 0, 80, 6000, 1200),
    ("skirting_n", 0, 2950, 0, 6000, 20, 80),
    ("skirting_w", -2950, 0, 0, 20, 6000, 80),
    ("sofa_seat", 900, 2450, 90, 2000, 850, 330),
    ("sofa_back", 900, 2780, 420, 2000, 200, 450),
    ("sofa_arm_w", -200, 2450, 420, 220, 850, 250),
    ("sofa_arm_e", 2000, 2450, 420, 220, 850, 250),
    ("table_leg_1", 550, 1600, 0, 50, 50, 400),
    ("table_leg_2", 1850, 1600, 0, 50, 50, 400),
    ("table_leg_3", 550, 1000, 0, 50, 50, 400),
    ("table_leg_4", 1850, 1000, 0, 50, 50, 400),
    ("table_top", 1200, 1300, 400, 1450, 750, 40),
    ("bookshelf", -2750, 1200, 0, 350, 1000, 1100),
    ("rug", 1200, 1300, 0, 2400, 1600, 2),
]


def test_default_home_is_the_page_room_in_the_page_orientation() -> None:
    layout = load("default_home")
    for element_id, x, y, z, sx, sy, sz in PAGE_ROOM:
        element = layout.element(element_id)
        assert element.pos_mm == (x, y, z), element_id
        assert element.size_mm == (sx, sy, sz), element_id
    # The curiosity prototype was the page's mirror image (sofa at y −2.45); this is not.
    assert layout.element("sofa_seat").pos_mm[1] > 0
    assert layout.element("plant_pot").pos_mm[:2] == (-2600, -1400)
    assert layout.thing("ball_felt").pos_mm[:2] == (800, -600)  # the page's ball
    # Physical now, all but the rug (seen, not felt: owner's list) and the leaves (soft).
    assert not layout.element("rug").collide and not layout.element("plant_leaves").collide
    assert all(layout.element(w).collide for w in ("wall_n", "wall_e", "wall_s", "wall_w"))
    assert layout.element("sofa_seat").under_mm == 90
    # The page's lamp is lights.py's lamp.
    from duckmind.world import lights

    (lamp,) = layout.lights.lamps
    assert tuple(v / 1000 for v in lamp.pos_mm) == pytest.approx(lights.LAMP_POSITION)
    assert lamp.hours == lights.LAMP_HOURS


def test_default_home_walls_are_duck_cores_room() -> None:
    """Duck Core plans in its own 6×6 m room (engine.py ROOM_*): the physical walls stand on
    its edges, and the plan's limit leaves a standing duck clear of them."""
    from duckmind.duckcore import engine

    layout = load("default_home")
    half, limit = engine.ROOM_HALF_MM, engine.ROOM_LIMIT_MM
    walls = {e.id: e for e in layout.elements if e.kind == "wall"}
    assert set(walls) == {"wall_n", "wall_s", "wall_e", "wall_w"}
    assert walls["wall_n"].pos_mm[1] == half and walls["wall_s"].pos_mm[1] == -half
    assert walls["wall_e"].pos_mm[0] == half and walls["wall_w"].pos_mm[0] == -half
    for wall in walls.values():
        x0, y0, x1, y1 = wall.footprint()
        inner = min(abs(v) for v in (x0, y0, x1, y1) if abs(v) >= half - 100)
        assert inner >= limit + homelayout.DUCK_RADIUS_MM, wall.id
        assert inner <= half
    # Duck Core's origin is the world's (the layout's frame is the room), whether the duck has
    # the dock by the south wall, phase 2's open nest at the origin, or only its mat there: its
    # imagined walls are these walls. A nest duck believes its nest is at (0, 0) — the dock is
    # not there (phase 5 teaches it) — and the physics duck starts seated in the dock.
    for name in ("default_home", "default_home,backin", "default_home,no_nest"):
        home = load(name)
        assert home.frame == homelayout.Pose(0, 0, 0)
        assert home.plan_to_world(1234, -567, 890) == homelayout.Pose(1234, -567, 890)
    nest = layout.nest
    # the homes dock backin since 2026-09-28: seated with its rear on the dock's plates,
    # facing the room, having turned round on the dock's right
    assert nest is not None and nest.is_dock and (nest.style, nest.side) == ("backin", "right")
    assert nest.pose == homelayout.Pose(0, -2940, 1571)  # its back on the south wall
    assert nest.seat == homelayout.Pose(0, -2768, 1571, 4)  # facing the room, on its plate
    assert nest.home == homelayout.Pose(0, -2768, 1571)  # where it would be Duck Core's (0, 0)
    assert layout.spawn == nest.seat
    assert load("default_home,backin").nest == nest
    forward = load("default_home,forward").nest
    assert forward is not None and forward.seat == homelayout.Pose(0, -2700, -1570, 4)
    assert forward.home == homelayout.Pose(0, -2700, 1571)  # facing the dock, on its plate
    opened = load("default_home,open_nest")
    assert opened.nest is not None and opened.nest.pose == homelayout.Pose(0, 0, 0)
    assert opened.spawn == homelayout.Pose(0, 0, 0, opened.nest.pad_height_mm)


def test_nest_and_mat_by_variant() -> None:
    with_dock, without = load("default_home"), load("default_home,no_nest")
    assert with_dock.nest is not None and with_dock.nest.is_dock and without.nest is None
    ids = {e.id for e in with_dock.elements}
    dock_parts = {"nest_housing", "nest_panel", "nest_plate", "nest_tag", "nest_edge_tag"}
    guides = {"nest_guide_l0", "nest_guide_l1", "nest_guide_r0", "nest_guide_r1"}
    assert dock_parts | guides | {"nest_status_light"} <= ids
    assert not ids & {"nest_pad", "nest_board", "nest_led"}
    assert "mat" not in ids and not with_dock.mat.drawn  # the dock replaces the mat
    assert "mat" in {e.id for e in without.elements} and without.mat.drawn
    # the cyan accent (C3) is the face panel; the plate is a slab ramped on every side
    assert with_dock.element("nest_panel").colour == "#12b5c4"
    plate = with_dock.element("nest_plate")
    assert (plate.shape, plate.kind, plate.ramp_mm, plate.size_mm[2]) == (
        "slab",
        "dock_plate",
        40,
        4,
    )
    assert plate.support and with_dock.support_top(0, -2700) == 4
    # phase 2's open nest, for comparison
    with_nest = load("default_home,open_nest")
    ids = {e.id for e in with_nest.elements}
    assert {"nest_pad", "nest_board", "nest_tag", "nest_edge_tag", "nest_led"} <= ids
    assert "mat" not in ids and not with_nest.mat.drawn  # the nest replaces the mat
    board = with_nest.element("nest_board")
    # cyan, 48 cm behind: phase 2's right-arc exit gate from a duck facing the board (§10.5)
    assert board.colour == "#12b5c4" and board.pos_mm[:2] == (-480, 0)
    assert homelayout.NEST_BOARD_OFFSET_MM == 480
    back = with_nest.nest.back_box_mm if with_nest.nest is not None else None
    assert back is not None and back[0] <= -480 - 10  # the clearance box reaches behind it
    # the apartment's open nest spot (B1) keeps 410: 480 would put the board into its south wall
    if MJCF and (Path(MJCF) / "apartment.xml").is_file():
        apartment = load("apartment,open_nest")
        assert apartment.nest is not None and apartment.nest.board_offset_mm == 410
        assert apartment.check() == []
    assert with_nest.element("nest_pad").size_mm[2] == 5
    assert load("default_home,edge_3mm").element("nest_pad").size_mm[2] == 3
    ramp = load("default_home,ramp").element("nest_pad")
    assert ramp.shape == "frustum" and (ramp.radius_mm, ramp.radius_bottom_mm) == (200, 250)
    with pytest.raises(homelayout.LayoutError, match="停靠方式"):
        load("default_home,open_nest,backin")  # a docking style is only a dock's


def test_page_json_round_trips_and_has_every_element() -> None:
    for name in ("default_home", "hazard_home"):
        layout = load(name)
        page = layout.page_json()
        assert json.loads(json.dumps(page, ensure_ascii=False)) == page
        assert (
            page["schema"] == homelayout.PAGE_SCHEMA and page["fingerprint"] == layout.fingerprint
        )
        assert len(page["elements"]) == len(layout.elements)
        assert len(page["things"]) == len(layout.things)
        for element in page["elements"]:
            assert {"id", "geom", "shape", "pos_mm", "size_mm", "colour", "flags", "room"} <= set(
                element
            )
            assert set(element["flags"]) == {"collide", "camera", "tof", "page"}
            assert all(isinstance(v, int) for v in element["pos_mm"] + element["size_mm"])
        assert page["legend"] and all(entry["zh"] for entry in page["legend"])
        # …and the layout itself: to_json() loads back to the same home.
        again = load(layout.to_json())
        assert again.fingerprint == layout.fingerprint
        assert again.page_json() == page


def test_apartment_round_trips_without_reconverting() -> None:
    layout = _load("apartment_ungated_test")
    again = load(layout.to_json())
    assert again.fingerprint == layout.fingerprint and again.variant == "ungated_test"
    assert again.page_json() == layout.page_json()


def test_curiosity_regression_the_official_apartment_has_two_placement_errors() -> None:
    """Curiosity spec §4.1: converted as upstream has it, the checks must catch the cube in
    the kitchen island (obj_1) and the ball on the stair-hole edge (ball_2) — and nothing
    else about the things; the overlay moves both and the apartment passes."""
    raw = apartment.raw_layout(_official())
    flagged = {p.subject for p in raw.check() if p.code.startswith("thing_")}
    assert flagged == {"obj_1", "ball_2"}
    codes = {(p.code, p.subject) for p in raw.check()}
    assert ("thing_in_furniture", "obj_1") in codes
    assert ("thing_near_drop", "ball_2") in codes
    fixed = load("apartment")
    assert fixed.thing("obj_1").pos_mm[:2] == (-3300, 1500)
    assert fixed.thing("ball_2").pos_mm[:2] == (-250, -2300)
    assert fixed.check() == []


def test_apartment_overlay_b1_to_b5() -> None:
    layout = _load("apartment")
    ids = {e.id for e in layout.elements}
    # B1: the decorative dock and the blue landmark are out (in memory). Our charging dock
    # (2026-09-27) stands with its back on the living room's south wall facing +y; phase 2's
    # open nest (the SW corner facing +y) is the open_nest variant.
    assert not ids & {"dock_base", "dock_panel", "dock_s1", "dock_s2", "lm_living"}
    assert layout.hidden == ("dock", "lm_living")
    assert layout.nest is not None and layout.nest.is_dock
    assert (layout.nest.x_mm, layout.nest.y_mm, layout.nest.heading_mrad) == (-1700, -2940, 1571)
    seat = layout.nest.seat
    assert layout.room_at(seat.x_mm, seat.y_mm) == "living"
    opened = load("apartment,open_nest").nest
    assert opened is not None and (opened.x_mm, opened.y_mm) == (-3500, -2450)
    # B3: windows in the living room and the bedroom; one lamp per room, 17:00–23:00; the
    # living room's is the official floor lamp.
    assert {w.room for w in layout.lights.windows} == {"living", "bedroom"}
    assert sorted(lamp.room for lamp in layout.lights.lamps) == sorted(r.id for r in layout.rooms)
    assert all(lamp.hours == (17, 23) for lamp in layout.lights.lamps)
    floor_lamp = layout.lights.for_room("living").lamps[0]
    shade = layout.element("liv_lamp_shade")
    assert floor_lamp.element == "liv_lamp_shade"
    assert floor_lamp.pos_mm[:2] == shade.pos_mm[:2]
    for room in layout.rooms:
        assert 1 <= layout.lights.for_room(room.id).count <= homelayout.ROOM_LIGHT_MAX
    # B5: the official 4 cubes and 3 balls stay, next to our toy kit.
    official = {t.id for t in layout.things if t.official}
    assert official == {"obj_0", "obj_1", "obj_2", "obj_3", "ball_0", "ball_1", "ball_2"}
    assert {"ball_felt", "fin_fish", "books"} <= {t.id for t in layout.things}
    # The landmarks are seen, not ToF-seen; the walls reach the ceiling for the camera only.
    assert not layout.element("lm_kitchen").tof and layout.element("lm_kitchen").camera
    band = layout.element("wall_south_upper")
    assert (band.pos_mm[2], band.top_mm, band.collide) == (1600, 2400, False)
    # Duck Core's frame is its nest — the dock's seat, facing into the room: a step "ahead" of
    # the nest is toward +y in the world. (The backin dock since 2026-09-28 seats the duck
    # 68 mm further back than the forward one did: −2768, not −2700.)
    assert layout.frame == homelayout.Pose(-1700, -2768, 1571)
    ahead = layout.plan_to_world(1000, 0, 0)
    assert (ahead.x_mm, ahead.y_mm, ahead.heading_mrad) == (-1700, -1768, 1571)
    back = layout.world_to_plan(ahead.x_mm, ahead.y_mm, ahead.heading_mrad)
    assert (back.x_mm, back.y_mm, back.heading_mrad) == (1000, 0, 0)


def test_nest_only_fits_with_the_dock_and_landmark_hidden() -> None:
    """The survey's finding: at (−3.50, −2.45) the open nest's clearance is blocked by the
    official decorative dock and the blue landmark unless they are hidden (B1)."""
    src = _load("apartment,open_nest").to_json()
    src["hide"] = []
    shown = load(src)
    blockers = {p.subject for p in shown.check() if p.code == "nest_blocked"}
    assert "lm_living" in blockers and blockers & {"dock_base", "dock_panel"}


def test_the_gated_variant_really_blocks_the_stair_hole() -> None:
    gated = _load("apartment")
    (stairs,) = gated.drops
    assert stairs.gated and len(gated.barriers) == 4
    assert {b.kind for b in gated.barriers} == {"railing", "gate"}
    for barrier in gated.barriers:  # every side, every opening narrower than the duck
        src = gated.to_json()
        src["barriers"] = [b for b in src["barriers"] if b["id"] != barrier.id]
        codes = {p.code for p in load(src).check()}
        assert "drop_not_gated" in codes, barrier.id
    posts = sorted(e.pos_mm[1] for e in gated.elements if e.id.startswith("railing_w_post_"))
    assert max(b - a for a, b in zip(posts, posts[1:])) - 20 <= homelayout.BARRIER_GAP_MM
    ungated = _load("apartment_ungated_test")
    assert not ungated.drops[0].gated and ungated.barriers == ()
    assert not any(e.kind in ("railing", "gate") for e in ungated.elements)


def test_the_checks_catch_bad_placements() -> None:
    src = load("default_home").to_json()
    src["things"] = [
        {**src["things"][0], "id": "in_sofa", "pos_mm": [900, 2450, 100]},
        {**src["things"][0], "id": "on_nest", "pos_mm": [100, -2300]},  # in the dock's lane
        {**src["things"][0], "id": "twin_a", "pos_mm": [-1000, -2000]},
        {**src["things"][0], "id": "twin_b", "pos_mm": [-990, -2000]},
    ]
    src["lights"]["lamps"] = []
    src["lights"]["windows"] = []
    codes = {(p.code, p.subject) for p in load(src).check()}
    assert ("thing_in_furniture", "in_sofa") in codes
    assert ("thing_near_nest", "on_nest") in codes
    assert ("thing_overlap", "twin_a") in codes
    assert ("room_without_light", "living") in codes
    src = load("default_home,no_nest").to_json()
    src["mat"] = {"x_mm": 2900, "y_mm": 0}
    src["spawn"] = "mat"
    assert ("spawn_blocked", "wall_e") in {(p.code, p.subject) for p in load(src).check()}
    src = load("hazard_home").to_json()
    src["things"].append(
        {**src["things"][1], "id": "safe_cup", "flags": [], "pos_mm": [-200, -1600]}
    )
    codes = {(p.code, p.subject) for p in load(src).check()}
    assert ("thing_near_drop", "safe_cup") in codes  # 30 cm from the stair hole
    assert not any(s in ("glass", "wine_glass") for _c, s in codes)  # hazards are exempt


def test_hazard_home_has_the_spec_hazards() -> None:
    layout = load("hazard_home")
    heights = sorted(e.size_mm[2] for e in layout.elements if e.kind == "threshold")
    assert heights == [12, 20, 30]  # 1–3 cm thresholds
    assert layout.element("step_up_15").size_mm[2] == 15
    assert layout.element("low_box").size_mm[2] == 60
    depths = sorted(d.depth_mm for d in layout.drops)
    assert depths == [50, 150, 400]  # a 5–15 cm drop each way, and the stair hole
    stairs = [e for e in layout.elements if e.kind == "stairs"]
    assert [e.top_mm for e in stairs] == [-100, -200, -300, -400]
    assert "floor" in layout.hidden  # the official plane is out; the floor is tiled
    assert layout.support_top(0, 0) == 0
    assert layout.support_top(-200, -2000) == -100  # the top step
    assert layout.support_top(2500, 2500) == -150  # the genkan
    assert layout.support_top(2500, -2500) == 15  # the step up
    kit = {"wine_glass", "glass", "photo_frame", "phone", "pet_bowl", "potted_plant", "lego"}
    assert kit | {"coin", "button_battery"} == {t.id for t in layout.things}
    assert all(t.hazard for t in layout.things)


def test_geom_groups_say_what_sees_an_element() -> None:
    g = homelayout.geom_group
    assert (g(True, True), g(True, False), g(False, True), g(False, False)) == (0, 1, 4, 5)
    layout = load("default_home")
    assert layout.element("wall_n").group == 0
    assert layout.element("window_east").group == 1  # glass: the camera sees it, the ToF not
    assert layout.element("rug").group == 1
    assert layout.element("ceiling").group == 0 and not layout.element("ceiling").page
    assert homelayout.TOF_GEOMGROUP[0] == 1 and homelayout.TOF_GEOMGROUP[1] == 0


def test_rooms_and_lights_by_room() -> None:
    layout = _load("apartment")
    assert layout.room_at(-2500, 1750) == "kitchen"
    assert layout.room_at(2250, -2000) == "bathroom"
    assert layout.room_at(9000, 0) is None
    rooms = layout.lights.by_room
    assert set(rooms) == {r.id for r in layout.rooms}
    assert [w.id for w in rooms["bedroom"].windows] == ["window_bedroom"]
    assert layout.element("kit_island").room == "kitchen"


def _repo_files() -> list[Path]:
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [ROOT / name for name in listed if (ROOT / name).is_file()]


def test_no_official_file_is_copied_into_the_repo() -> None:
    pinned = set(apartment.PINNED_SHA256.values())
    for path in homelayout.HOMES_DIR.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        named = [e["id"] for e in data.get("elements", []) if e.get("official")]
        assert named in ([], [homelayout.OFFICIAL_FLOOR]), path.name  # the plane, by name
    official: set[str] = set()
    if MJCF and Path(MJCF).is_dir():
        for path in Path(MJCF).rglob("*"):
            if path.is_file() and path.suffix.lower() in (".xml", ".stl", ".obj", ".png"):
                official.add(hashlib.sha256(path.read_bytes()).hexdigest())
    digests = official | pinned
    copied = [
        str(p.relative_to(ROOT))
        for p in _repo_files()
        if p.stat().st_size < 50_000_000 and hashlib.sha256(p.read_bytes()).hexdigest() in digests
    ]
    assert copied == []


# ── MuJoCo ────────────────────────────────────────────────────────────────────


def _mujoco() -> Any:
    pytest.importorskip("numpy", reason="the home checks in MuJoCo need numpy")
    return pytest.importorskip("mujoco", reason="the home checks need mujoco")


def _compiled(name: str, collision: str = "allcollisions") -> tuple[Any, Any, Layout]:
    mujoco = _mujoco()
    _official()
    layout = load(name)
    model = homelayout.compose(layout, collision).compile()
    model.opt.timestep = homelayout.BODY_TIMESTEP_S
    data = mujoco.MjData(model)
    stand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    mujoco.mj_resetDataKeyframe(model, data, stand)
    mujoco.mj_forward(model, data)
    return model, data, layout


def _robot_geoms(mujoco: Any, model: Any) -> set[int]:
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, homelayout.ROBOT_ROOT)
    return {g for g in range(model.ngeom) if model.body_rootid[model.geom_bodyid[g]] == root}


def _official_digests(folder: Path) -> dict[str, tuple[str, int]]:
    return {
        str(p.relative_to(folder)): (
            hashlib.sha256(p.read_bytes()).hexdigest(),
            p.stat().st_mtime_ns,
        )
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


@pytest.mark.parametrize("name", EVERY)
def test_every_home_compiles_and_the_duck_spawns_clear(name: str) -> None:
    mujoco = _mujoco()
    folder = _official()
    before = _official_digests(folder)
    model, data, layout = _compiled(name)
    assert _official_digests(folder) == before  # nothing written, nothing edited
    free = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, homelayout.ROBOT_FREEJOINT)
    trunk = data.qpos[model.jnt_qposadr[free] : model.jnt_qposadr[free] + 3]
    assert trunk[0] == pytest.approx(layout.spawn.x) and trunk[1] == pytest.approx(layout.spawn.y)
    robot = _robot_geoms(mujoco, model)
    touching = [
        (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1),
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2),
            round(float(c.dist) * 1000, 1),
        )
        for c in data.contact[: data.ncon]
        if (c.geom1 in robot or c.geom2 in robot) and c.dist < -0.001
    ]
    assert touching == []
    # Hidden official pieces are gone from the physics too.
    for hidden in layout.hidden:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, hidden) < 0
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, hidden) < 0


@pytest.mark.parametrize("name", ["default_home", "apartment"])
def test_the_other_collision_model_composes_too(name: str) -> None:
    mujoco = _mujoco()
    model, data, layout = _compiled(name, "groundcontact")
    robot = _robot_geoms(mujoco, model)
    colliding = [g for g in robot if model.geom_contype[g] or model.geom_conaffinity[g]]
    assert 0 < len(colliding) < 20  # robot_groundcontact.xml: the feet, not 70 geoms
    assert layout.nest is not None and layout.nest.is_dock
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "home_nest_plate") >= 0


@pytest.mark.parametrize("name", ["default_home", "hazard_home", "apartment", "real_home"])
def test_things_lie_still_for_five_seconds(name: str) -> None:
    """Curiosity spec §14: every thing, resting where the layout puts it, moves < 1 mm in
    5 s of physics at the body server's timestep (the duck held where it spawned)."""
    mujoco = _mujoco()
    model, data, _layout = _compiled(name)
    robot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, homelayout.ROBOT_FREEJOINT)
    frees = [j for j in range(model.njnt) if model.jnt_type[j] == 0 and j != robot]
    assert frees
    start = {j: data.qpos[model.jnt_qposadr[j] : model.jnt_qposadr[j] + 3].copy() for j in frees}
    adr = model.jnt_qposadr[robot]
    held = data.qpos[adr : adr + 7].copy()
    for _ in range(int(round(5.0 / model.opt.timestep))):
        mujoco.mj_step(model, data)
        data.qpos[adr : adr + 7] = held
        data.qvel[model.jnt_dofadr[robot] : model.jnt_dofadr[robot] + 6] = 0
    for j in frees:
        now = data.qpos[model.jnt_qposadr[j] : model.jnt_qposadr[j] + 3]
        moved = math.dist(now, start[j]) * 1000
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.jnt_bodyid[j])
        assert moved < 1.0, f"{body} moved {moved:.2f} mm"


@pytest.mark.parametrize("name", ["default_home", "hazard_home", "apartment", "real_home"])
def test_page_json_is_the_physics(name: str) -> None:
    """Every element the page draws is the geom the physics has (same name, place, size,
    collision and group), and every colliding world geom is on the page."""
    mujoco = _mujoco()
    model, data, layout = _compiled(name)
    page = layout.page_json()
    robot = _robot_geoms(mujoco, model)
    thing_bodies = {t.body for t in layout.things}
    drawn = {e["geom"] for e in page["elements"]}
    for element in page["elements"]:
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, element["geom"])
        assert g >= 0, element["id"]
        flags = element["flags"]
        assert bool(model.geom_contype[g]) == flags["collide"], element["id"]
        assert model.geom_group[g] == homelayout.geom_group(flags["camera"], flags["tof"])
        box = model.geom_aabb[g]
        centre = data.geom_xpos[g] + data.geom_xmat[g].reshape(3, 3) @ box[:3]
        half = abs(data.geom_xmat[g].reshape(3, 3)) @ box[3:]
        if element["shape"] == "plane":
            continue
        if element["shape"] == "capsule":  # its axis and radius (a slanted box is looser)
            ax, ay, az, bx, by, bz = (v / 1000 for v in element["fromto_mm"])
            assert list(data.geom_xpos[g]) == pytest.approx(
                [(ax + bx) / 2, (ay + by) / 2, (az + bz) / 2], abs=1e-3
            )
            length = math.dist((ax, ay, az), (bx, by, bz))
            assert model.geom_size[g][:2] == pytest.approx(
                [element["radius_mm"] / 1000, length / 2], abs=1e-3
            )
            continue
        low = [float(centre[i] - half[i]) * 1000 for i in range(3)]
        high = [float(centre[i] + half[i]) * 1000 for i in range(3)]
        x0, y0, x1, y1 = layout.element(element["id"]).footprint()
        z0, z1 = layout.element(element["id"]).z_range()
        tol = 1.5 if element["shape"] != "frustum" else 3.0
        assert low == pytest.approx([x0, y0, z0], abs=tol), element["id"]
        assert high == pytest.approx([x1, y1, z1], abs=tol), element["id"]
    for g in range(model.ngeom):
        if g in robot or not model.geom_contype[g]:
            continue
        body = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, model.body_rootid[model.geom_bodyid[g]]
        )
        if body in thing_bodies:
            continue
        name_g = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        assert name_g in drawn, name_g


@pytest.mark.parametrize("name", ["default_home", "apartment", "real_home"])
def test_keyframes_keep_things_where_they_are(name: str) -> None:
    """The keyframe trap: a reset to any keyframe leaves every thing at its layout place (not
    piled at the origin) and puts the duck at its spawn."""
    mujoco = _mujoco()
    model, data, layout = _compiled(name)
    robot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, homelayout.ROBOT_FREEJOINT)
    for key in range(model.nkey):
        mujoco.mj_resetDataKeyframe(model, data, key)
        for j in range(model.njnt):
            if model.jnt_type[j] != 0:
                continue
            adr = model.jnt_qposadr[j]
            if j == robot:
                assert data.qpos[adr] == pytest.approx(layout.spawn.x)
                assert data.qpos[adr + 1] == pytest.approx(layout.spawn.y)
            else:
                assert list(data.qpos[adr : adr + 7]) == pytest.approx(
                    list(model.qpos0[adr : adr + 7])
                )
                assert abs(data.qpos[adr]) + abs(data.qpos[adr + 1]) > 0.1


def test_the_stair_gate_stops_a_ray_at_duck_height() -> None:
    mujoco = _mujoco()
    import numpy as np

    for name, blocked in (("apartment", True), ("apartment_ungated_test", False)):
        model, data, _layout = _compiled(name)
        hits = []
        for direction in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)):
            geom = np.array([-1], dtype=np.int32)
            start = np.array([-0.2, -1.05, 0.08])  # the middle of the hole, 8 cm up
            distance = mujoco.mj_ray(
                model, data, start, np.array(direction, dtype=float), None, 1, -1, geom
            )
            hit = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom[0])) or ""
            hits.append(distance < 0.5 and hit.startswith(("home_railing", "home_gate")))
        assert all(hits) if blocked else not any(hits), (name, hits)


def test_the_official_scene_each_home_starts_from() -> None:
    assert homelayout.base_scene(load("default_home")) == "scene_allcollisions.xml"
    assert homelayout.base_scene(load("hazard_home"), "groundcontact") == "scene.xml"
    flat = homelayout.official_files(load("default_home"))
    assert flat == ["scene_allcollisions.xml", "robot_allcollisions.xml"]
    home = _load("apartment")
    assert home.base_scene == "scene_apartment.xml"
    assert homelayout.official_files(home) == [
        "scene_apartment.xml",
        "robot_allcollisions.xml",
        "apartment.xml",
    ]
    assert homelayout.official_files(home, "groundcontact") == [
        "scene.xml",
        "robot_groundcontact.xml",
        "apartment.xml",
    ]


def test_the_official_folder_is_found_or_named(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(homelayout.MJCF_ENV, raising=False)
    package = tmp_path / "mjlab_microduck"
    (package / "robot" / "microduck").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "robot" / "microduck" / "robot_allcollisions.xml").write_text("<mujoco/>")
    monkeypatch.syspath_prepend(str(tmp_path))
    # (another test in this session may have imported the real mjlab_microduck: find_spec
    # would then answer from sys.modules instead of the path)
    for name in [n for n in sys.modules if n.split(".")[0] == "mjlab_microduck"]:
        monkeypatch.delitem(sys.modules, name)
    assert homelayout.official_dir() == package / "robot" / "microduck"
    with pytest.raises(homelayout.LayoutError, match="不是 Pollen 的小鴨模型"):
        homelayout.official_dir(tmp_path)


def test_a_placement_stays_indoors_off_drops_and_on_this_side_of_the_walls() -> None:
    """Layout.place_problem (phase-2 review, critical): the official floor plane runs on past
    the walls, so "the nearest spot with floor under it" can be outside the house."""
    home = load("default_home,no_nest")
    assert home.place_problem(900, 1900) is None  # in front of the sofa
    assert "牆外" in (home.place_problem(1275, 3099) or "")  # past the north wall
    assert "牆外" in (home.place_problem(3190, 2990) or "")  # past the east wall
    wall = home.barrier_between((900, 2700), (1010, 3167))
    assert wall is not None and wall.kind == "wall"
    assert home.barrier_between((900, 2700), (900, 1900)) is None  # the sofa is no wall
    assert home.barrier_between((2990, 2990), (2800, 2800)) is None  # starts inside the wall
    assert "隔著" in (home.place_problem(900, 2900, from_mm=(900, 3200)) or "")
    assert (
        home.zh_of("home_wall_n") and home.zh_of("home_sofa_seat") == home.element("sofa_seat").zh
    )
    assert home.zh_of("thing_books_0") == home.thing("books").zh and home.zh_of("nothing") is None
    hazard = load("hazard_home")
    pit = hazard.drops[0]
    x0, y0, x1, y1 = pit.rect_mm
    middle = ((x0 + x1) / 2, (y0 + y1) / 2)
    assert "掉下去" in (hazard.place_problem(*middle) or "")
    beside = (x0 - 100, (y0 + y1) / 2)  # 10 cm off the edge: fine asked, not as a stand-in
    assert hazard.place_problem(*beside) is None
    assert "掉下去" in (hazard.place_problem(*beside, from_mm=beside, drop_margin_mm=140) or "")


def test_the_camera_sees_the_floor_the_page_draws() -> None:
    """One layout decides the floor (phase-2 gate finding): the page draws default_home's
    official floor as wood planks (home3d.js woodTexture), so the physics' copy that the head
    camera renders gets the same planks — not the official blue-grey checker. Only the look:
    collision, friction and the geom group stay the official plane's."""
    folder = str(_official())
    mujoco = pytest.importorskip("mujoco")
    planks = homelayout.wood_rgb(64)
    assert len(planks) == 64 * 64 * 3
    mean = [sum(planks[c::3]) / (64 * 64) for c in range(3)]
    tones = [[int(t[i : i + 2], 16) for i in (1, 3, 5)] for t in homelayout.WOOD_TONES]
    for c in range(3):  # the page's tones, a little darker for the outlines and grain
        average = sum(t[c] for t in tones) / len(tones)
        assert average * 0.85 < mean[c] <= average + 1
    for name, geoms in (("default_home", ["floor"]), ("hazard_home", None)):
        layout = load(name, mjcf_dir=folder)
        model = homelayout.compose(layout, mjcf_dir=folder).compile()
        wood = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, homelayout.WOOD_MATERIAL)
        assert wood >= 0
        drawn = [e.geom for e in layout.elements if e.texture == "wood"]
        assert drawn and (geoms is None or drawn == geoms)
        for geom in drawn:
            g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
            assert int(model.geom_matid[g]) == wood, geom
        texture = int(model.mat_texid[wood][int(mujoco.mjtTextureRole.mjTEXROLE_RGB)])
        assert int(model.tex_colorspace[texture]) == int(mujoco.mjtColorSpace.mjCOLORSPACE_SRGB)
    official = mujoco.MjModel.from_xml_path(str(Path(folder) / "scene_allcollisions.xml"))
    home = load("default_home", mjcf_dir=folder)
    model = homelayout.compose(home, mjcf_dir=folder).compile()
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    plain = mujoco.mj_name2id(official, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert int(model.geom_group[floor]) == int(official.geom_group[plain])
    assert list(model.geom_friction[floor]) == list(official.geom_friction[plain])
    assert int(model.geom_contype[floor]) == int(official.geom_contype[plain])
