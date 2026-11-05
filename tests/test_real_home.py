"""The realistic home (owner 2026-10-10) and the editor's checks of a layout."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from duckmind.world import homelayout, layoutcheck, objlib

HOME = Path(homelayout.HOMES_DIR) / "real_home.json"


def _raw() -> dict:
    return json.loads(HOME.read_text(encoding="utf-8"))


def test_real_home_is_a_small_apartment_with_rooms_doorways_and_things_to_cross() -> None:
    layout = homelayout.load("real_home")
    assert [r.id for r in layout.rooms] == ["living", "hall", "bedroom", "kitchen"]
    assert len(layout.doorways) == 4
    assert layout.check() == []
    items = {e.lib for e in layout.elements if e.lib} | {t.lib for t in layout.things if t.lib}
    for wanted in (
        "threshold_5", "threshold_10", "threshold_20", "rug_thick", "rug_thin", "yoga_mat", "cable_round_8",
        "cable_round_6", "cable_flat_3", "slipper_pair", "shoes_pair", "sofa_3seat", "table_dining",
        "chair_dining", "bed_double", "bowl_water", "cushion", "sock",
    ):  # fmt: skip
        assert wanted in items, wanted
    # a cable near the TV, slippers and shoes at the entry
    cable = layout.element("cable_round_8_1")
    tv = layout.element("tv_stand_1_cabinet")
    assert abs(cable.pos_mm[0] - tv.pos_mm[0]) < 1500 and abs(cable.pos_mm[1] - tv.pos_mm[1]) < 600
    hall = next(r for r in layout.rooms if r.id == "hall")
    for t in layout.things:
        if t.lib in ("slipper_pair", "shoes_pair") and hall.contains(*t.pos_mm[:2]):
            break
    else:
        raise AssertionError("no slippers or shoes in the entry hall")


def test_real_home_variants_and_the_old_home_stays() -> None:
    assert {"no_nest", "forward", "backin"} <= set(homelayout.variants("real_home"))
    assert homelayout.load("real_home,no_nest").nest is None
    assert homelayout.load("real_home").nest is not None
    assert homelayout.load("default_home").check() == []  # the old measures stay comparable
    assert "real_home" in homelayout.available()


def test_every_doorway_is_open_and_has_a_sill_of_its_own() -> None:
    layout = homelayout.load("real_home")
    sills = {e.place: e for e in layout.elements if e.kind == "threshold"}
    assert len(sills) == 4
    for d in layout.doorways:
        assert d["sill"].startswith("threshold_")
        mid = ((d["from_mm"][0] + d["to_mm"][0]) / 2, (d["from_mm"][1] + d["to_mm"][1]) / 2)
        walls = [e for e in layout.elements if e.kind == "wall" and e.contains_xy(*mid)]
        assert walls == [], d["id"]  # nothing in the doorway but its sill


def test_the_checks_pass_for_every_home_and_catch_what_the_editor_can_do_wrong() -> None:
    assert layoutcheck.check(_raw())["ok"]
    for name in ("default_home", "hazard_home", "plain_home"):
        raw = json.loads((Path(homelayout.HOMES_DIR) / f"{name}.json").read_text(encoding="utf-8"))
        assert layoutcheck.check(raw)["ok"], name

    def with_(*extra: dict) -> list[dict]:
        raw = _raw()
        raw["placements"] += list(extra)
        return layoutcheck.check(raw)["problems"]

    overlap = with_({"id": "w", "lib": "wardrobe", "x_mm": -900, "y_mm": -1000, "yaw_mrad": 0})
    assert any(p["code"] == "overlap" and p["id"] == "w" for p in overlap) or any(
        p.get("other") == "w" for p in overlap
    )
    assert any(
        p["code"] == "outside"
        for p in with_({"id": "b", "lib": "bed_double", "x_mm": 2950, "y_mm": 0})
    )
    assert any(
        p["code"] == "overlap"
        for p in with_({"id": "t", "lib": "table_dining", "x_mm": 0, "y_mm": 600})
    )  # in a wall
    assert any(
        p["code"] == "bad_placement"
        for p in with_({"id": "q", "lib": "nope", "x_mm": 0, "y_mm": 0})
    )
    # a rug may lie under what stands on it
    assert with_({"id": "r", "lib": "rug_thin", "x_mm": -900, "y_mm": -1000}) == []


def test_a_turned_cushion_beside_a_sock_can_be_applied_and_a_real_overlap_names_both() -> None:
    """Owner 2026-11-04: 「套用到模擬器」 stayed off on real_home because the check said the floor cushion
    overlapped 「家裡本來就有的東西」. The cushion is turned (400 mrad): its bounding box reached the sock beside
    it, its outline does not; and the words had no name for the sock because the problem did not say it."""

    def cushion_at(x: int, y: int) -> dict:
        raw = _raw()
        for p in raw["placements"]:
            if p["id"] == "cushion_1":
                p["x_mm"], p["y_mm"] = x, y
        return layoutcheck.check(raw)

    beside = cushion_at(450, -1200)
    assert beside["ok"], beside["problems"]
    on_the_ball = cushion_at(700, -900)  # right over the felt ball
    overlap = [p for p in on_the_ball["problems"] if p["code"] == "thing_overlap"]
    assert overlap and {(p["lib"], p["other_lib"]) for p in overlap} == {("cushion", "ball_felt")}


def test_a_shut_in_room_is_found() -> None:
    raw = _raw()
    # wardrobes across the bedroom doorway and the sofa-side opening shut the bedroom in
    raw["placements"] += [
        {"id": "w1", "lib": "wardrobe", "x_mm": -1450, "y_mm": 700, "yaw_mrad": 0},
        {"id": "w2", "lib": "wardrobe", "x_mm": -1450, "y_mm": 1250, "yaw_mrad": 0},
    ]
    verdict = layoutcheck.check(raw)
    assert "bedroom" in verdict["reach"]["unreachable_rooms"]
    assert any(
        p["code"] == "room_unreachable" and p["id"] == "bedroom" for p in verdict["problems"]
    )
    assert verdict["reach"]["reached_pct"] < 100


def test_a_hop_over_a_sill_is_a_way_in_but_a_curb_is_not() -> None:
    def reach(sill: str) -> dict:
        raw = copy.deepcopy(_raw())
        raw["placements"] = [p for p in raw["placements"] if p["lib"] not in ("threshold_10",)]
        raw["placements"] += [
            {"id": "s", "lib": sill, "x_mm": -1450, "y_mm": 600, "yaw_mrad": 0, "length_mm": 900}
        ]
        # the bedroom's only other way in is walled off
        return layoutcheck.check(raw)["reach"]

    assert reach("threshold_30")["rooms"]["bedroom"] > 90  # 30 mm: a hop
    assert objlib.library().item("curb_80").passability == "block"
