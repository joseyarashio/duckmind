"""Writes ``src/duckmind/world/homes/real_home.json``: a small apartment with a living room, a bedroom,
a kitchen and an entry hall, doors with sills, rugs, cables near the TV, slippers and shoes at the
entry and a few small things on the floor (owner, 2026-10-10: 「家中環境採真實一點的設置」).

The structure (walls, rooms, doorways, lights) is written here; everything that stands in the rooms is a
*placement* of an item from the object library (``world/objects/library.json``), so the layout editor
can move it. The result is plain JSON: edit it by hand or in the editor.

    python tools/make_real_home.py            # rewrites real_home.json
    python tools/make_real_home.py OUT.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HOMES = Path(__file__).resolve().parents[1] / "src/duckmind/world/homes"
DEFAULT = HOMES / "default_home.json"
HALF = 3000
WALL_T = 80
WALL_H = 1200

# (x0, y0, x1, y1) of each room, wall centre lines to wall centre lines; no two overlap
ROOMS = {
    "living": ("客廳", (-3000, -3000, 1500, 600)),
    "hall": ("玄關", (1500, -3000, 3000, 600)),
    "bedroom": ("臥室", (-3000, 600, 0, 3000)),
    "kitchen": ("廚房", (0, 600, 3000, 3000)),
}
# doorways: id, rooms, from, to, threshold item
DOORS = (
    ("living_bedroom", ("living", "bedroom"), (-1900, 600), (-1000, 600), "threshold_10"),
    ("living_kitchen", ("living", "kitchen"), (200, 600), (1200, 600), "threshold_5"),
    ("hall_kitchen", ("hall", "kitchen"), (1900, 600), (2800, 600), "threshold_20"),
    ("living_hall", ("living", "hall"), (1500, -1500), (1500, -600), "threshold_5"),
)


def wall(wid: str, a: tuple[int, int], b: tuple[int, int]) -> dict:
    (ax, ay), (bx, by) = a, b
    if ay == by:
        size, pos = [bx - ax, WALL_T, WALL_H], [(ax + bx) // 2, ay, 0]
    else:
        size, pos = [WALL_T, by - ay, WALL_H], [ax, (ay + by) // 2, 0]
    return {
        "id": wid,
        "kind": "wall",
        "zh": "牆",
        "shape": "box",
        "pos_mm": pos,
        "size_mm": size,
        "colour": "#f3efe7",
    }


def placements() -> list[dict]:
    out: list[dict] = []
    counter: dict[str, int] = {}

    def put(lib: str, x: int, y: int, yaw: int = 0, **extra: object) -> None:
        counter[lib] = counter.get(lib, 0) + 1
        out.append(
            {
                "id": f"{lib}_{counter[lib]}",
                "lib": lib,
                "x_mm": x,
                "y_mm": y,
                "yaw_mrad": yaw,
                **extra,
            }
        )

    # doorway sills
    for _, _, (ax, ay), (bx, by), item in DOORS:
        if ay == by:
            put(item, (ax + bx) // 2, ay, 0, length_mm=bx - ax)
        else:
            put(item, ax, (ay + by) // 2, 1571, length_mm=by - ay)
    # living room: TV on the south wall, the sofa facing it, a shag rug and a table between, cables
    put("tv_stand", -900, -2760, 3142)
    put("sofa_3seat", -900, -1000, 0)
    put("rug_thick", -900, -2000, 0)
    put("coffee_table", -900, -2000, 0)
    put("power_strip", -1850, -2870, 0)
    put(
        "cable_round_8", -1850, -2500, 1571, length_mm=600
    )  # a power cord from the strip, along the floor
    put("cable_flat_3", -1650, -2250, 1571, length_mm=800)  # a network cable from the stand
    put("bookshelf", -2400, 385, 0)
    put("plant_pot", -2700, -250, 0)
    put("yoga_mat", -1000, -150, 0)
    put("cushion", 700, -1300, 400)
    put("laundry_basket", -600, 360, 0)
    # a few things on the floor
    put("ball_felt", 900, -900)
    put("ball_hollow", 400, -650)
    put("fin_fish", 1200, -1250)
    put("fin_duck", 1100, -100)
    put("paper_cup", 300, -1600)
    put("tissue_box", 1300, -500)
    put("bottle_empty", 200, 300)
    put("books", 200, -250)
    put("magazine", -1500, -100, 200)
    put("sock", 300, -900, 700)
    # entry hall: the front door, a mat, a shoe cabinet, shoes and slippers, a console
    put("door_panel", 2956, -2250, 1571)
    put("doormat", 2700, -2250, 1571)
    put("shoe_rack", 2200, -2790, 3142)
    put("shoes_pair", 1900, -2450, 300)
    put("shoes_pair", 2250, -2450, -300)
    put("slipper_pair", 2050, -1950, 500)
    put("slipper_pair", 2400, -1650, -200)
    put("console_cabinet", 1715, -200, 1571)
    put("umbrella_stand", 1700, -2750, 0)
    # bedroom: the bed head at the west wall, a nightstand with a phone charger, a wardrobe, a mat
    put("bed_double", -1960, 2150, 1571)
    put("nightstand", -2750, 1100, 0)
    put("cable_round_6", -2270, 950, -540, length_mm=600)  # the phone charger's cord
    put("charger_brick", -1940, 770, 0)
    put("wardrobe", -315, 2200, -1571)
    put("kitchen_mat", -1500, 1050, 0)
    put("slipper_pair", -2300, 1260, 100)
    put("paperback_book", -700, 2000, 600)
    put("laundry_basket", -800, 900, 0)
    put("sock", -700, 1500, 1900)
    # kitchen: the counter and the fridge, a table with four chairs, the pet's mat and bowls
    put("kitchen_counter", 1150, 2660, 0)
    put("fridge", 2350, 2635, 0)
    put("trash_bin", 2800, 2750)
    put("rug_thin", 1500, 1500, 0)  # under the table and its chairs
    put("table_dining", 1500, 1500, 0)
    put("chair_dining", 1250, 1000, 3142)
    put("chair_dining", 1750, 1000, 3142)
    put("chair_dining", 1250, 2000, 0)
    put("chair_dining", 1750, 2000, 0)
    put("kitchen_mat", 2700, 1500, 1571)
    put("bowl_food", 2750, 1300)
    put("bowl_water", 2750, 1700)
    put("toy_block", 500, 900, 100, colour="#e2523c")
    put("toy_block", 590, 960, 400, colour="#2f6fe0")
    put("toy_block", 540, 1060, 900, colour="#f2c91e")
    return out


def build() -> dict:
    base = json.loads(DEFAULT.read_text())
    elements = [
        e
        for e in base["elements"]
        if e["id"] in ("floor", "wall_n", "wall_w", "wall_s", "wall_e") or e["kind"] == "skirting"
    ]
    # inner walls, with the doorways left open
    elements += [
        wall("wall_h1", (-3000, 600), (-1900, 600)),
        wall("wall_h2", (-1000, 600), (200, 600)),
        wall("wall_h3", (1200, 600), (1900, 600)),
        wall("wall_h4", (2800, 600), (3000, 600)),
        wall("wall_v_bed_kitchen", (0, 600), (0, 3000)),
        wall("wall_v_hall1", (1500, -3000), (1500, -1500)),
        wall("wall_v_hall2", (1500, -600), (1500, 600)),
    ]
    spawn_mat = {
        "x_mm": 600,
        "y_mm": -300,
        "heading_mrad": 0,
        "radius_mm": 340,
        "colour": "#9fb8b0",
    }
    dock = dict(base["nest"])
    dock.update({"x_mm": 700, "y_mm": -2940})
    rooms = [
        {"id": rid, "zh": zh, "rect_mm": list(rect), "floor": "wood"}
        for rid, (zh, rect) in ROOMS.items()
    ]
    doorways = [
        {"id": did, "rooms": list(rs), "from_mm": list(a), "to_mm": list(b), "sill": item}
        for did, rs, a, b, item in DOORS
    ]
    home = {
        "schema": "home-layout-v2",
        "id": "real_home",
        "zh": "真實的家：客廳、臥室、廚房和玄關（門檻、地毯、電線、拖鞋）",
        "doc": (
            "A small apartment, 6 x 6 m: the living room (with the TV, the sofa, a shag rug, a yoga mat and a few toys), "
            "the bedroom, the kitchen and the entry hall, joined by four doorways with sills of 5, 5, 10 and 20 mm. "
            "Everything that stands in the rooms is a placement of an object library item (world/objects/library.json): "
            "furniture, rugs, cables near the TV and the bed, slippers and shoes at the entry, small things on the floor. "
            "Written by tools/make_real_home.py; edit placements by hand or in the layout editor (資料 page). The walls, "
            "rooms and lights are as in default_home (the same 6.08 m square, Duck Core's frame stays the room). "
            "The charging dock stands with its back on the south wall of the living room; no_nest is the variant for 蛋黃. "
            "Units: mm, mrad, g. docs/design/HOME-FORMAT.md."
        ),
        "base": {},
        "bounds_mm": base["bounds_mm"],
        "ceiling_mm": base["ceiling_mm"],
        "north_mrad": base["north_mrad"],
        "wall_bands": True,
        "floor": base["floor"],
        "rooms": rooms,
        "doorways": doorways,
        "spawn": "nest",
        "frame": "room",
        "nest": dock,
        "mat": spawn_mat,
        "elements": elements,
        "things": [],
        "placements": placements(),
        "drops": [],
        "zones": [
            {
                "id": "toy_corner",
                "kind": "keepsake",
                "zh": "玩具角（收玩具放這裡）",
                "rect_mm": [400, -1000, 1400, -200],
            }
        ],
        "barriers": [],
        "lights": {
            "lamps": [
                {
                    "id": "lamp_living",
                    "zh": "客廳的燈",
                    "room": "living",
                    "kind": "pendant",
                    "pos_mm": [-600, -800, 1900],
                    "hours": [17, 23],
                    "colour": "#ffd3a0",
                    "lux": 120,
                },
                {
                    "id": "lamp_hall",
                    "zh": "玄關的燈",
                    "room": "hall",
                    "kind": "ceiling",
                    "pos_mm": [2250, -1200, 2300],
                    "hours": [17, 23],
                    "colour": "#ffd3a0",
                    "lux": 120,
                },
                {
                    "id": "lamp_bedroom",
                    "zh": "臥室的燈",
                    "room": "bedroom",
                    "kind": "ceiling",
                    "pos_mm": [-1500, 1800, 2300],
                    "hours": [17, 23],
                    "colour": "#ffd3a0",
                    "lux": 120,
                },
                {
                    "id": "lamp_kitchen",
                    "zh": "廚房的燈",
                    "room": "kitchen",
                    "kind": "ceiling",
                    "pos_mm": [1500, 1600, 2300],
                    "hours": [17, 23],
                    "colour": "#ffd3a0",
                    "lux": 120,
                },
            ],
            "windows": [
                {
                    "id": "window_living",
                    "zh": "客廳的落地窗",
                    "room": "living",
                    "pos_mm": [-2960, -1200, 0],
                    "width_mm": 1600,
                    "height_mm": 2000,
                    "inward_mrad": 0,
                },
                {
                    "id": "window_bedroom",
                    "zh": "臥室的窗",
                    "room": "bedroom",
                    "pos_mm": [-1500, 2960, 900],
                    "width_mm": 1200,
                    "height_mm": 1200,
                    "inward_mrad": -1571,
                },
                {
                    "id": "window_kitchen",
                    "zh": "廚房的窗",
                    "room": "kitchen",
                    "pos_mm": [1500, 2960, 1100],
                    "width_mm": 1000,
                    "height_mm": 900,
                    "inward_mrad": -1571,
                },
            ],
            "night_lights": [],
        },
        "agents": {
            "spots": [
                {"id": "entry", "zh": "大門口", "x_mm": 2450, "y_mm": -2250, "heading_mrad": 3142},
                {"id": "sofa", "zh": "沙發", "x_mm": -900, "y_mm": -1000, "heading_mrad": -1571},
            ]
        },
        "variants": {
            "no_nest": {
                "zh": "沒有充電窩的小鴨（例如蛋黃）：墊子在客廳",
                "nest": None,
                "spawn": "mat",
            },
            "forward": base["variants"]["forward"],
            "backin": base["variants"]["backin"],
        },
    }
    return home


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HOMES / "real_home.json"
    out.write_text(json.dumps(build(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("wrote", out)
