"""world.lights: the page's day/night logic as one light state, on the survey's light ladder."""

import math
from datetime import date, datetime

import pytest

from duckmind.world import lights

# The survey's light ladder (survey_calibration.md §2.3, spec §12.3), lux in the room.
LADDER = {"day": 300.0, "dusk": 5.0, "lamps": 120.0, "moon": 0.25, "black": 0.03}

# The four times of the phase-0 gate, with the label each must carry.
FOUR = [
    ("day", datetime(2026, 9, 27, 12, 0), {}, "白天"),
    # Lamps come on at 17:00, so a lamp-free dusk happens where the sun sets before 17:00.
    ("dusk", datetime(2026, 12, 1, 16, 50), {}, "黃昏"),
    ("lamps", datetime(2026, 9, 27, 20, 0), {}, "晚上開燈"),
    ("moon", datetime(2026, 9, 28, 2, 0), {}, "月光"),
]


def js_twilight(minute: float, rise: float, sunset: float) -> float:
    """duck3d.js lightFrame's twilight line, verbatim (see lights.py: it is 0.12 all day)."""
    return (
        max(lights.smooth(1 - (rise - minute) / 30), lights.smooth(1 - (minute - sunset) / 30))
        * 0.12
    )


@pytest.mark.parametrize("key,when,kwargs,label", FOUR)
def test_the_four_times_carry_their_label_and_the_ladder_level(key, when, kwargs, label):
    state = lights.light_state(when, **kwargs)
    assert state.label == label
    assert state.key == key
    assert state.lux == pytest.approx(LADDER[key], rel=0.10)


def test_pitch_black_night_and_dawn():
    black = lights.light_state(datetime(2026, 9, 28, 2, 0), moon=0.0)
    assert (black.key, black.label) == ("black", "全黑")
    assert black.lux == pytest.approx(LADDER["black"], rel=0.10)
    dawn = lights.light_state(datetime(2026, 9, 27, 5, 45))
    assert (dawn.key, dawn.label) == ("dawn", "清晨")
    assert dawn.lux == pytest.approx(LADDER["dusk"], rel=0.10)
    # An evening the family leaves the lamps off: dusk, then about 1 lux a quarter hour later.
    evening = lights.light_state(datetime(2026, 9, 27, 17, 40), lamps_on=False)
    assert evening.label == "黃昏" and evening.lux == pytest.approx(5.0, rel=0.10)
    later = lights.light_state(datetime(2026, 9, 27, 18, 1), lamps_on=False)
    assert 0.5 < later.lux < 2.0 and later.label == "黃昏"


def test_sun_times_match_the_page():
    # duck3d.js sunTimes() evaluated in node for the same dates.
    page = {
        date(2026, 9, 27): (343.4372979173018, 1065.903338638709),
        date(2026, 6, 21): (304.23830731787535, 1125.9381662825697),
        date(2026, 12, 1): (381.8462206454188, 1024.2558254051837),
        date(2026, 3, 20): (359.10780829891775, 1084.7403812987434),
    }
    for day, (rise, sunset) in page.items():
        got = lights.sun_times(day)
        assert got[0] == pytest.approx(rise, abs=1e-6)
        assert got[1] == pytest.approx(sunset, abs=1e-6)


def test_daytime_factors_match_the_page():
    rise, sunset = lights.sun_times(date(2026, 9, 27))
    # duck3d.js lightFrame targets evaluated in node (4 decimals).
    page = {
        720: (0.9978, 1.0, 0.0, 0.0),
        1020: (0.1983, 0.4049, 0.6958, 0.5951),
        1050: (0.0691, 0.12, 0.8579, 0.88),
        360: (0.072, 0.12, 0.8724, 0.0),
    }
    for minute, (elev, daylight, glow, lamps) in page.items():
        f = lights.page_factors(minute, rise, sunset)
        assert (f.elevation, f.daylight, f.glow, f.lamps) == pytest.approx(
            (elev, daylight, glow, lamps), abs=1e-4
        )


def test_twilight_is_limited_to_its_half_hours():
    rise, sunset = lights.sun_times(date(2026, 9, 27))
    # The page's line is 0.12 at every minute (one of its two terms is always 1)...
    assert js_twilight(120, rise, sunset) == pytest.approx(0.12)
    assert js_twilight(1200, rise, sunset) == pytest.approx(0.12)
    # ...here it rises in the half hour before sunrise and fades in the half hour after sunset.
    assert lights.page_factors(120, rise, sunset).daylight == 0.0
    assert lights.page_factors(1200, rise, sunset).daylight == 0.0
    assert lights.page_factors(rise - 15, rise, sunset).twilight == pytest.approx(0.06)
    assert lights.page_factors(sunset + 15, rise, sunset).twilight == pytest.approx(0.06)
    assert lights.page_factors(sunset + 31, rise, sunset).twilight == 0.0


def test_lamps_follow_the_family_hours_and_the_daylight():
    assert lights.light_state(datetime(2026, 9, 27, 16, 59)).lamps == 0.0
    assert lights.light_state(datetime(2026, 9, 27, 17, 0)).lamps > 0.5
    assert lights.light_state(datetime(2026, 9, 27, 22, 59)).lamps == 1.0
    assert lights.light_state(datetime(2026, 9, 27, 23, 0)).lamps == 0.0
    assert lights.light_state(datetime(2026, 9, 27, 20, 0), lamps_on=False).label == "月光"
    # A long June day: at 17:00 there is still too much daylight for lamps.
    assert lights.light_state(datetime(2026, 6, 21, 17, 0)).lamps == 0.0


def test_light_over_a_day_has_no_jumps():
    previous = None
    for minute in range(0, 1440):
        state = lights.light_state(datetime(2026, 9, 27, minute // 60, minute % 60))
        assert state.lux > 0
        if previous is not None and minute not in (17 * 60, 23 * 60):  # lamps switch
            assert 0.8 < state.lux / previous < 1.25, (minute, previous, state.lux)
        previous = state.lux


def test_sources_are_ready_for_a_renderer():
    day = lights.light_state(datetime(2026, 9, 27, 12, 0))
    (sun,) = day.sources
    assert sun.name == "sun" and sun.kind == "directional"
    assert math.isclose(sum(v * v for v in sun.toward), 1.0) and sun.toward[2] > 0.5
    night = lights.light_state(datetime(2026, 9, 27, 20, 0))
    by_name = {s.name: s for s in night.sources}
    assert set(by_name) == {"lamp", "moon"}
    assert by_name["lamp"].position == lights.LAMP_POSITION
    for source in night.sources:
        r, g, b = source.color
        assert 0.2126 * r + 0.7152 * g + 0.0722 * b == pytest.approx(1.0)
        assert 0 < source.indirect < 1
    assert sum(s.lux for s in night.sources) == pytest.approx(night.lux)


def test_page_parameters_match_light_frame():
    noon = lights.light_state(datetime(2026, 9, 27, 12, 0)).three()
    assert noon["hemi"] == {"intensity": pytest.approx(1.25), "color": "#f4f7f2"}
    assert noon["sun"]["intensity"] == pytest.approx(1.72)
    assert noon["sun"]["color"] == "#ffffff"
    assert noon["fill"]["intensity"] == pytest.approx(0.5)
    assert noon["lamp"]["intensity"] == 0.0 and noon["lamp"]["position"] == [0.9, 1.9, -1.6]
    assert noon["background"] == "#e9eef0"
    evening = lights.light_state(datetime(2026, 9, 27, 20, 0)).three()
    assert evening["hemi"]["intensity"] == pytest.approx(0.53)
    assert evening["lamp"] == {
        "intensity": pytest.approx(2.2),
        "color": "#ffd3a0",
        "position": [0.9, 1.9, -1.6],
    }
    assert evening["background"] == "#2b3446"
    assert evening["sun"]["offset"] == [0.0, 0.735, 0.5]  # "the moon sits high"


def test_state_serialises_for_the_page():
    import json

    data = lights.light_state(datetime(2026, 9, 27, 20, 0)).to_dict()
    text = json.dumps(data, ensure_ascii=False)
    assert "晚上開燈" in text and data["lux_by_source"]["lamp"] == pytest.approx(120.0)
    assert data["when"] == "2026-09-27T20:00:00"


# ── phase 2: homes (rooms, lamps, windows, the nest's backlight) ──────────────

MOMENTS = {
    "day": datetime(2026, 9, 27, 12, 0),
    "dusk": datetime(2026, 12, 1, 16, 50),
    "lamps": datetime(2026, 9, 27, 20, 0),
    "moon": datetime(2026, 9, 28, 2, 0),
}

# A small flat in the layouts' own units (mm, mrad): a living room with a window and a floor
# lamp, a bedroom with a window, a corridor with neither window nor lamp but a night light.
FLAT = {
    "id": "flat",
    "ceiling_mm": 2400,
    "rooms": [
        {"id": "living", "rect_mm": [-4000, -3000, -1000, 500]},
        {"id": "bedroom", "rect_mm": [500, 1000, 4000, 3000]},
        {"id": "corridor", "rect_mm": [-1000, -3000, 500, 3000]},
    ],
    "lights": {
        "lamps": [
            {"id": "ceiling", "room": "living", "pos_mm": [-2500, -1250, 2300], "lux": 80},
            {
                "id": "floor_lamp",
                "room": "living",
                "kind": "floor_lamp",
                "pos_mm": [-3800, -300, 950],
                "hours": [18, 22],
                "colour": "#ffd3a0",
                "lux": 40,
            },
            {"id": "bed", "pos_mm": [2250, 2000, 2300]},  # no room: placed where it stands
        ],
        "windows": [
            # the layouts' way: the sill's middle on the wall, width, height, and the way
            # daylight travels in (+x: a window in the west wall)
            {
                "id": "west",
                "room": "living",
                "pos_mm": [-3940, -1000, 900],
                "width_mm": 1600,
                "height_mm": 1200,
                "inward_mrad": 0,
            },
            {
                "id": "north",
                "room": "bedroom",
                "pos_mm": [2600, 2940, 900],
                "width_mm": 1400,
                "height_mm": 1200,
                "inward_mrad": -1571,
            },
        ],
        "night_lights": [{"id": "hall", "room": "corridor", "pos_mm": [-250, 0, 300]}],
    },
    "nest": {
        "x_mm": -3500,
        "y_mm": -2450,
        "heading_mrad": 1571,
        "backlight": {"when": "dark", "colour": "#e8f6f7", "charging_colour": "#1f9d57"},
    },
}


def test_the_default_room_is_phase_zeros_room_value_for_value():
    for minute in range(0, 1440, 7):
        when = datetime(2026, 9, 27, minute // 60, minute % 60)
        old = lights.light_state(when)
        room = lights.room_light_state(when, lights.DEFAULT_ROOM)
        assert room == old
        home = lights.home_light_state(when)
        assert home.rooms["home"] == old
    assert lights.DEFAULT_ROOM.reference == lights.REFERENCE_SPOT


def test_a_layout_reads_into_rooms_lamps_windows_and_the_nest():
    home = lights.HomeLights.from_layout(FLAT)
    assert [r.id for r in home.rooms] == ["living", "bedroom", "corridor"]
    living = home.room("living")
    assert living.reference == pytest.approx((-2.5, -1.25, 0.0))
    assert living.top == pytest.approx(2.4)
    by_id = {lamp.id: lamp for lamp in living.lamps}
    assert by_id["ceiling"].position == pytest.approx((-2.5, -1.25, 2.3))
    assert by_id["floor_lamp"].on_hours == (18.0, 22.0) and by_id["floor_lamp"].lux == 40
    assert home.room("bedroom").lamps[0].id == "bed"  # placed by where it stands
    (west,) = living.windows
    assert west.facing == pytest.approx((-1.0, 0.0), abs=1e-3)  # it looks out west
    assert west.position == pytest.approx((-3.94, -1.0, 1.5))  # the sill + half its height
    assert west.size == pytest.approx((1.6, 1.2))
    toward = west.toward()
    assert toward[0] < 0 and toward[2] == pytest.approx(math.sin(math.radians(40)))
    (night,) = home.room("corridor").lamps
    assert night.kind == "night" and night.on_hours == (23.0, 7.0)
    nest = home.nest
    assert nest is not None and nest.backlight
    # the backboard's face, behind the pad (-x in the nest frame, which faces +y)
    assert nest.position[:2] == pytest.approx((-3.5, -2.85), abs=1e-3)
    assert nest.charging_color == 0x1F9D57
    assert home.room_at(-2.0, 0.0).id == "living"
    assert home.room_at(-1.0 - 0.001, 2.0).id == "corridor"  # a wall gap: the nearest room


def test_every_room_is_on_the_ladder_at_its_middle():
    home = lights.HomeLights.from_layout(FLAT)
    ladder = {"day": 300.0, "dusk": 5.0, "moon": 0.25}
    for key, when in MOMENTS.items():
        state = lights.home_light_state(when, home)
        for room_id in ("living", "bedroom"):  # rooms with windows
            room = state.rooms[room_id]
            if key == "lamps":
                assert room.key == "lamps" and room.label == "晚上開燈"
            else:
                assert room.key == key
                assert room.lux == pytest.approx(ladder[key], rel=0.10)
        hall = state.rooms["corridor"]  # no window: a little of the day, all bounced
        if key == "day":
            assert hall.lux == pytest.approx(300 * lights.SPILL) and hall.key == "day"
            (spill,) = hall.sources
            assert spill.kind == "ambient" and spill.indirect == 1.0
    evening = lights.home_light_state(MOMENTS["lamps"], home)
    # both living-room lamps on at 20:00 (each its own lux at the room's middle)
    assert evening.rooms["living"].lux == pytest.approx(80 + 40, abs=0.5)
    assert evening.rooms["bedroom"].lux == pytest.approx(120.0, abs=0.5)
    at_19 = lights.home_light_state(datetime(2026, 9, 27, 17, 30), home).rooms["living"]
    assert {s.id for s in at_19.sources if s.kind == "spot"} == {"ceiling"}  # floor lamp at 18
    late = lights.home_light_state(datetime(2026, 9, 28, 2, 0), home).rooms["corridor"]
    assert late.lux == pytest.approx(lights.NIGHT_LIGHT_LUX + 0.25 * lights.SPILL, rel=1e-6)


def test_lux_anywhere_follows_the_lamps_and_matches_at_the_middle():
    home = lights.HomeLights.from_layout(FLAT)
    room = lights.home_light_state(MOMENTS["lamps"], home).rooms["living"]
    assert room.lux_at(room.reference) == pytest.approx(room.lux, rel=1e-9)
    far_corner = room.lux_at((-1.1, -2.9, 0.0))
    assert room.lux > far_corner
    for source in room.sources:
        if source.kind == "spot":
            # drawn from just above the ceiling over where the lamp stands
            assert source.fixture == pytest.approx((source.position[0], source.position[1], 2.45))
            assert source.direct_at(room.reference) == pytest.approx(
                source.lux * (1 - source.indirect)
            )
            below = (source.position[0], source.position[1], 0.0)
            assert source.direct_at(below) >= source.direct_at(room.reference)
    # the one light state serves HomeWorld too: the home answers for any point
    state = lights.home_light_state(MOMENTS["lamps"], home)
    assert state.lux_at((-3.7, -0.4, 0.0)) == pytest.approx(room.lux_at((-3.7, -0.4, 0.0)))
    assert state.at(2.0, 2.0).room == "bedroom"


def test_the_nest_backlight_glows_only_in_the_dark_and_changes_colour_while_charging():
    home = lights.HomeLights.from_layout(FLAT)
    on = {}
    for key, when in MOMENTS.items():
        backlight = lights.home_light_state(when, home).backlight
        assert backlight is not None and backlight.room == "living"
        on[key] = backlight.on
    assert on == {"day": False, "dusk": True, "lamps": False, "moon": True}
    idle = lights.home_light_state(MOMENTS["moon"], home).backlight
    charging = lights.home_light_state(MOMENTS["moon"], home, charging=True).backlight
    assert idle is not None and charging is not None
    assert idle.hex == "#e8f6f7" and charging.hex == "#1f9d57"
    assert max(idle.color) == pytest.approx(1.0) and max(charging.color) == pytest.approx(1.0)
    # an absolute brightness: π × cd/m² on the camera's lux × albedo scale, day or night
    assert idle.scene_lux == pytest.approx(math.pi * lights.BACKLIGHT_NITS)
    no_nest = lights.HomeLights.from_layout({**FLAT, "nest": None})
    assert lights.home_light_state(MOMENTS["moon"], no_nest).backlight is None
    unlit = {**FLAT, "nest": {**FLAT["nest"], "backlight": False}}
    assert (
        lights.home_light_state(MOMENTS["moon"], lights.HomeLights.from_layout(unlit)).backlight
        is None
    )


def test_the_page_gets_every_lamp_window_and_the_backlight():
    import json

    home = lights.HomeLights.from_layout(FLAT)
    page = lights.home_light_state(MOMENTS["lamps"], home).to_page()
    text = json.dumps(page, ensure_ascii=False)
    assert "晚上開燈" in text
    lamps = {lamp["id"]: lamp for lamp in page["lamps"]}
    assert lamps["floor_lamp"]["on"] and lamps["floor_lamp"]["position"] == [-3.8, 0.95, 0.3]
    assert lamps["floor_lamp"]["intensity"] == pytest.approx(2.2 * 40 / 120, abs=1e-3)
    windows = {w["id"]: w for w in page["windows"]}
    assert windows["west"]["toward"][0] < 0  # three.js x = world x
    assert page["backlight"]["on"] is False and page["rooms"]["corridor"]["key"] == "black"


def test_a_home_layout_from_homelayout_reads_if_it_is_here():
    homelayout = pytest.importorskip(
        "duckmind.world.homelayout", reason="world/homelayout.py (phase 2 layouts) not here yet"
    )
    layout = homelayout.load("default_home")
    home = lights.HomeLights.from_layout(layout)
    assert [room.id for room in home.rooms] == [room.id for room in layout.rooms]
    for key, when in MOMENTS.items():
        state = lights.home_light_state(when, home)
        for room in home.rooms:
            if room.windows:
                assert state.rooms[room.id].key == key
    assert home.nest is not None
