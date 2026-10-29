"""The object library (world/objects/library.json) and the placements that use it (owner 2026-10-10)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from duckmind.world import homelayout, objlib

STATIC = Path(homelayout.__file__).parents[1] / "simweb" / "static"
LANGS = ("zh-Hant", "en", "ja", "es", "zh-Hans", "ko")


def _catalog(lang: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(STATIC.glob(f"i18n.{lang}.*.js")):
        body = path.read_text(encoding="utf-8")
        out.update(
            json.loads(body[body.index("{", body.index("DuckI18n.load(")) : body.rindex("}") + 1])
        )
    return out


def test_the_library_has_the_items_the_owner_asked_for() -> None:
    ids = {item.id for item in objlib.library().items}
    wanted = {
        "threshold_5", "threshold_10", "threshold_15", "threshold_20",  # door sills
        "rug_thin", "rug_thick", "doormat", "yoga_mat",  # rugs, rug edges, mat edge
        "cable_round_6", "cable_round_8", "cable_flat_3",  # cables
        "slipper_pair", "sock", "paperback_book", "magazine", "bowl_food", "bowl_water",
        "charger_brick", "toy_block", "cushion", "laundry_basket", "chair_dining",
        "table_dining", "sofa_3seat", "shoes_pair", "ball_felt", "books",
    }  # fmt: skip
    assert wanted <= ids


def test_every_item_is_well_formed_and_classed_by_what_the_duck_meets() -> None:
    lib = objlib.library()
    assert lib.items
    for item in lib.items:
        assert item.passability in objlib.CLASSES, item.id
        assert item.category in objlib.CATEGORIES, item.id
        assert item.kind in ("element", "thing"), item.id
        assert item.parts, item.id
        if item.kind == "element" and item.passability in objlib.HEIGHT_CLASSES:
            assert item.height_class == item.passability, (item.id, item.height_mm)
        if item.kind == "thing":
            assert item.mass_g > 0, item.id
            assert item.passability in ("movable", "soft", "block"), item.id
        for part in item.parts:
            assert part["shape"] in (
                objlib.ELEMENT_SHAPES if item.kind == "element" else objlib.THING_SHAPES
            )
    # every class the owner named has an item
    assert {item.passability for item in lib.items} >= set(objlib.CLASSES)


def test_the_height_classes() -> None:
    assert [objlib.height_class(h) for h in (3, 5, 6, 20, 21, 40, 41, 300)] == [
        "under_foot", "under_foot", "step", "step", "hop", "hop", "block", "block",
    ]  # fmt: skip


def test_real_sizes_for_the_named_items() -> None:
    lib = objlib.library()
    assert [lib.item(f"threshold_{h}").height_mm for h in (5, 10, 15, 20)] == [5, 10, 15, 20]
    assert lib.item("cable_round_6").height_mm == 6 and lib.item("cable_round_8").height_mm == 8
    assert (
        6 <= lib.item("cable_flat_3").footprint_mm()[1] <= 20
        and lib.item("cable_flat_3").height_mm <= 5
    )
    assert lib.item("rug_thin").height_mm <= 5 < lib.item("rug_thick").height_mm
    assert lib.item("rug_thick").page_json()["contact"] == "soft"
    # a rug grips more than the wood floor (the contact takes the larger friction)
    assert lib.item("rug_thin").page_json()["friction"] > homelayout.OFFICIAL_FLOOR_FRICTION[0]


def _home(placements: list[dict]) -> dict:
    raw = json.loads((Path(homelayout.HOMES_DIR) / "default_home.json").read_text(encoding="utf-8"))
    raw["placements"] = placements
    return raw


def test_placements_become_elements_and_things_with_their_class() -> None:
    layout = homelayout.load(
        _home(
            [
                {
                    "id": "sill",
                    "lib": "threshold_15",
                    "x_mm": -2000,
                    "y_mm": 1500,
                    "yaw_mrad": 1571,
                    "length_mm": 700,
                },
                {"id": "slip", "lib": "slipper_pair", "x_mm": -2000, "y_mm": 600, "yaw_mrad": 300},
                {
                    "id": "cab",
                    "lib": "cable_round_8",
                    "x_mm": -2000,
                    "y_mm": -600,
                    "yaw_mrad": 0,
                    "length_mm": 500,
                },
            ]
        )
    )
    sill = layout.element("sill")
    assert sill.lib == "threshold_15" and sill.place == "sill" and sill.passability == "step"
    assert sill.size_mm == (700, 60, 15) and sill.yaw_mrad == 1571
    slip = layout.thing("slip")
    assert slip.lib == "slipper_pair" and slip.mass_g == 300 and slip.passability == "movable"
    cable = layout.element("cab")
    assert cable.shape == "capsule" and cable.fromto_mm[0] == -2250 and cable.fromto_mm[3] == -1750
    assert layout.check() == []
    page = layout.page_json()
    assert next(e for e in page["elements"] if e["id"] == "sill")["lib"] == "threshold_15"


def test_a_layout_round_trips_through_to_json_and_unexpand() -> None:
    raw = _home(
        [
            {"id": "a", "lib": "sock", "x_mm": 0, "y_mm": 0},
            {"id": "b", "lib": "sofa_3seat", "x_mm": -1500, "y_mm": -1500},
        ]
    )
    layout = homelayout.load(raw)
    again = homelayout.load(layout.to_json())  # already expanded: not expanded twice
    assert {e.id for e in again.elements} == {e.id for e in layout.elements}
    assert [t.id for t in again.things] == [t.id for t in layout.things]
    plain = objlib.unexpand(layout.to_json())
    assert [p["id"] for p in plain["placements"]] == ["a", "b"]
    assert not any(e.get("place") for e in plain["elements"]) and not plain.get(
        "placements_expanded"
    )
    assert homelayout.load(plain).fingerprint == layout.fingerprint or {
        e.id for e in homelayout.load(plain).elements
    } == {e.id for e in layout.elements}


def test_bad_placements_are_refused_in_plain_words() -> None:
    for bad in (
        {"id": "x", "lib": "nothing_like_it", "x_mm": 0, "y_mm": 0},
        {"id": "x", "lib": "sock", "x_mm": 1.5, "y_mm": 0},
        {"id": "x", "lib": "sock", "x_mm": 0, "y_mm": 0, "length_mm": 400},
        {"id": "x", "lib": "threshold_5", "x_mm": 0, "y_mm": 0, "length_mm": 100},
    ):
        with pytest.raises(homelayout.LayoutError):
            homelayout.load(_home([bad]))
    with pytest.raises(homelayout.LayoutError):
        homelayout.load(
            _home(
                [
                    {"id": "x", "lib": "sock", "x_mm": 0, "y_mm": 0},
                    {"id": "x", "lib": "sock", "x_mm": 300, "y_mm": 0},
                ]
            )
        )


def test_the_library_speaks_six_languages() -> None:
    catalogs = {lang: _catalog(lang) for lang in LANGS}
    for item in objlib.library().items:
        key = f"page.lib.item.{item.id}"
        assert catalogs["zh-Hant"][key] == item.zh, item.id  # the file's name is the source
        for lang in LANGS[1:]:
            assert catalogs[lang].get(key), (lang, item.id)
    for cls in objlib.CLASSES:
        for lang in LANGS:
            assert catalogs[lang].get(f"page.lib.class.{cls}"), (lang, cls)
    for cat in objlib.CATEGORIES:
        for lang in LANGS:
            assert catalogs[lang].get(f"page.lib.cat.{cat}"), (lang, cat)


def test_the_page_names_of_pieces_from_the_library_use_the_library_keys() -> None:
    from duckmind.i18n import boundary

    layout = homelayout.load(_home([{"id": "s", "lib": "sock", "x_mm": 0, "y_mm": 0}]))
    page = layout.page_json()
    answer = boundary.annotate_home(copy.deepcopy(page), layout.id)
    sock = next(t for t in answer["things"] if t["id"] == "s")
    assert sock["_i18n"]["zh"]["k"] == "page.lib.item.sock"
