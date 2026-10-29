"""The light in the duck's home, as one state the page, the camera and later HomeWorld read.

Ported from the simulator page's day/night logic (``duck3d.js``: ``sunTimes`` and the target
values of ``lightFrame``): Taipei sunrise and sunset for the duck's date, civil twilight, the
family's lamps from 17:00 to 23:00, moonlight after. Everything here is a pure function of the
duck's local date and time (and of the home's rooms, lamps and windows), so the page, the MuJoCo
head camera (:mod:`duckmind.world.camera_model`, through :mod:`duckmind.world.render_scene`) and
HomeWorld can never disagree about how bright a room is. The page's easing between states is a
drawing matter and stays on the page.

Two entry points:

* :func:`light_state` — phase 0's one room (the page's 6×6 room: its sun, its one lamp, the
  moon). Unchanged, value for value.
* :func:`home_light_state` — phase 2's homes (spec §12.3; owner decisions B3, C4, F5): every room
  of a :class:`HomeLights` (built from a home layout by :meth:`HomeLights.from_layout`) gets its
  own :class:`LightState` — daylight and moonlight through its windows (a room without windows
  gets a little of it second-hand), its lamps (the floor lamp too) on their hours, night lights —
  and the nest's backlight is switched on when the nest's corner is dark, in another colour while
  the duck charges.

What a room's state holds:

* the page's own factors (``daylight``, ``glow``, ``lamps``, the sun's ``elevation``) and the
  Three.js light parameters computed from them exactly as ``lightFrame`` does
  (:meth:`LightState.three`, colours as ``#rrggbb``, blended in linear light like three.js);
* the light sources for a renderer (:class:`LightSource`: the sun, the moon or a window as
  directional light, a lamp as a spot hanging just above the ceiling over where the lamp stands,
  light that only arrives bounced as ``ambient``, and for each how much arrives as direction-less
  room light). :meth:`LightState.lux_at` gives the lux on a level surface anywhere in the room
  with the same fall-off the renderer draws, so the camera and HomeWorld agree everywhere, not
  only at the room's middle;
* an illuminance estimate in lux at the room's reference spot (the middle of its floor), on the
  survey's light ladder (``survey_calibration.md`` §2.3 and spec §12.3: day 300, lamps 120, dusk
  5, twilight about 1, moonlight 0.25, pitch black 0.03 lux) — an assumption about a Taipei flat
  until the camera's light constant is measured on a board (spec phase 8);
* a plain label: 白天／清晨／黃昏／晚上開燈／月光／全黑.

One correction to the page. ``lightFrame`` computes civil twilight as
``max(smooth(1 - (rise - m) / 30), smooth(1 - (m - set) / 30)) * 0.12``; one of the two terms
is 1 at every minute of the day (the first after sunrise, the second before sunset), so the
page never gets darker than "daylight 0.12" at night and its twilight never fades. The comment
beside it says what was meant ("the sky stays light for about half an hour past the
horizon"), and that is what :func:`page_factors` does: twilight rises over the half hour
before sunrise and fades over the half hour after sunset. When the page switches to reading
this state its nights become moonlit instead of dusky.

A morning twilight is labelled 清晨 rather than 黃昏: with lamps coming on at 17:00 the only
lamp-free dim light in most of the year is the dawn, and calling 05:40 "dusk" would be wrong.

What the rooms approximate (F5): MuJoCo draws no shadows here, so light does not stop at walls.
The renderer therefore lights only the room the camera is in with its own windows and lamps
(at most 7 lights) and gives the rest of the home that room's bounced light;
:mod:`duckmind.world.render_scene` does that selection.
"""

from __future__ import annotations

import math
import types
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any

from duckmind.i18n import msg

# The duck's home (duck3d.js HOME_LAT / HOME_LON / HOME_UTC): Taipei, UTC+8, no daylight saving.
HOME_LAT = 25.03
HOME_LON = 121.56
HOME_UTC = 8

# duck3d.js lightFrame.
TWILIGHT_MIN = 30.0  # civil twilight: the sky stays light about half an hour past the horizon
TWILIGHT_LEVEL = 0.12  # the page's daylight factor at the horizon
LAMP_HOURS = (17, 23)  # the family's lamps: on from 17:00 while daylight < 0.7, off at 23:00

# The survey's light ladder (lux on a level surface in the middle of the room).
DAY_LUX = 300.0  # daylight through the windows at daylight factor 1
DUSK_LUX = 5.0  # at the horizon (daylight factor 0.12), lamps not yet on
LAMP_LUX = 120.0  # evening lamps, fully on
MOON_LUX = 0.25  # moon and city glow through the windows
BLACK_LUX = 0.03  # no moon: what is left of the city glow
# Daylight between the ladder's two anchors: lux = DAY_LUX * daylight ** p, through (1, 300)
# and (0.12, 5). It puts "twilight, about 1 lux" a quarter hour past sunset, as the survey has it.
DAYLIGHT_EXPONENT = math.log(DAY_LUX / DUSK_LUX) / math.log(1.0 / TWILIGHT_LEVEL)

# Label thresholds (lux).
DAY_MIN_LUX = 30.0  # daylight below this is twilight: the camera needs gain beyond the shutter
MOON_MIN_LUX = 0.1  # between moonlight (0.25) and pitch black (0.03)

# How much of each source reaches the room as direction-less light (bounced off walls,
# ceiling and sky) rather than straight from where it is. MuJoCo draws no bounced light, so a
# renderer adds this part as ambient light.
SUN_INDIRECT = 0.6  # daylight indoors is mostly sky and wall light
LAMP_INDIRECT = 0.3
MOON_INDIRECT = 0.5

# Where things are, in the MuJoCo world frame (metres, z up). The page draws the same lamp at
# three.js (0.9, 1.9, -1.6): three.js is y-up, and the page's world group maps MuJoCo (x, y, z)
# to three.js (x, z, -y).
LAMP_POSITION = (0.9, 1.6, 1.9)
REFERENCE_SPOT = (0.0, 0.0, 0.0)  # where the lux figures apply: the middle of the room

# How a lamp's direct light falls off (the same numbers the camera's MuJoCo lamps carry). A lamp
# is drawn as a ceiling fixture over where it stands: MuJoCo 3.10 on this Mac (CGL) renders a
# surface that faces a positional light from below wrongly (measured in phase 0; see
# camera_model.SPOT_CUTOFF_DEG), so every lamp shines down from just above the ceiling, and the
# lux model here uses that same point so the page, HomeWorld and the camera agree.
DEFAULT_CEILING_M = 2.4
LAMP_ABOVE_CEILING_M = 0.05
LAMP_ATTENUATION = (0.3, 0.0, 0.9)  # constant, linear, quadratic

# Homes (spec §12.3, owner decision B3).
WINDOW_SKY_ELEVATION_DEG = 40.0  # daylight comes in through a window from the sky above it
SPILL = 0.15  # a room without windows gets this much of the day and the night, all bounced
NIGHT_LIGHT_LUX = 2.0  # a plug-in night light, at the room's middle
# The nest's backlight (spec §10.5, §11.5; owner decisions C3, C4): a dim glowing panel behind
# the tag, on only when the nest's corner is dark, another colour while the duck charges. Its
# brightness is absolute (cd/m²), so the camera sees it the same by day and by night — the dark
# makes it stand out, not a brighter lamp.
BACKLIGHT_NITS = 3.0  # "very dim" (spec D2): brightest thing at night, not a lamp; colour kept
BACKLIGHT_ON_BELOW_LUX = 10.0  # lux at the nest: dusk (5 lux) and moonlight switch it on
BACKLIGHT_COLOR = 0xE8F6F7  # a cool white by the cyan backboard (C3; the layouts' default)
BACKLIGHT_CHARGING_COLOR = 0x1F9D57  # green while charging (the layouts' default)

# duck3d.js colours (sRGB hex).
DAY_SKY = 0xE9EEF0
NIGHT_SKY = 0x2B3446
DAY_LIGHT = 0xF4F7F2
NIGHT_LIGHT = 0x9FB0D8  # moonlight
DUSK_SKY = 0xF0C7A0  # sunrise / sunset glow
SUN_WARM = 0xFFB46E
SUN_DAY = 0xFFFFFF
LAMP_COLOR = 0xFFD3A0

LABELS = {
    "day": msg("world.lights.LABELS.day"),
    "dawn": msg("world.lights.LABELS.dawn"),
    "dusk": msg("world.lights.LABELS.dusk"),
    "lamps": msg("world.lights.LABELS.lamps"),
    "moon": msg("world.lights.LABELS.moon"),
    "black": msg("world.lights.LABELS.black"),
}

RGB = tuple[float, float, float]
Vec3 = tuple[float, float, float]
Point = tuple[float, float]


# ── the sun ────────────────────────────────────────────────────────────


def sun_times(day: date) -> tuple[float, float]:
    """Sunrise and sunset at home that day, in minutes after local midnight.

    duck3d.js ``sunTimes``: the sun's declination and the equation of time, within a few
    minutes of an almanac.
    """
    n = day.timetuple().tm_yday
    g = 2 * math.pi * (n - 1) / 365
    decl = (
        0.006918
        - 0.399912 * math.cos(g)
        + 0.070257 * math.sin(g)
        - 0.006758 * math.cos(2 * g)
        + 0.000907 * math.sin(2 * g)
        - 0.002697 * math.cos(3 * g)
        + 0.00148 * math.sin(3 * g)
    )
    eqt = 229.18 * (
        0.000075
        + 0.001868 * math.cos(g)
        - 0.032077 * math.sin(g)
        - 0.014615 * math.cos(2 * g)
        - 0.040849 * math.sin(2 * g)
    )
    lat = math.radians(HOME_LAT)
    zenith = math.radians(90.833)  # the sun's rim at the horizon, with refraction
    cos_h = math.cos(zenith) / (math.cos(lat) * math.cos(decl)) - math.tan(lat) * math.tan(decl)
    half = math.degrees(math.acos(min(1.0, max(-1.0, cos_h)))) * 4
    noon = 720 - 4 * HOME_LON - eqt + HOME_UTC * 60
    return noon - half, noon + half


def smooth(x: float) -> float:
    """duck3d.js ``smooth``: smoothstep on [0, 1]."""
    t = min(1.0, max(0.0, x))
    return t * t * (3 - 2 * t)


@dataclass(frozen=True)
class PageFactors:
    """duck3d.js ``lightFrame`` targets, twilight limited to its two half hours."""

    elevation: float  # sin of the day's progress, 0 at night (not the true solar altitude)
    twilight: float
    daylight: float
    glow: float
    lamps: float
    phase: float  # where the sun is on its east-to-west arc; 0.5 at night (the moon sits high)


def _in_hours(hour: float, hours: tuple[float, float]) -> bool:
    """Whether ``hour`` (0–24) falls in ``[on, off)``; a span past midnight wraps (23 → 7)."""
    on, off = float(hours[0]), float(hours[1])
    if on <= off:
        return on <= hour < off
    return hour >= on or hour < off


def page_factors(
    minute: float,
    sunrise: float,
    sunset: float,
    *,
    lamps_on: bool = True,
    lamp_hours: tuple[float, float] = LAMP_HOURS,
) -> PageFactors:
    day_len = sunset - sunrise
    elevation = (
        math.sin(math.pi * (minute - sunrise) / day_len) if sunrise < minute < sunset else 0.0
    )
    if minute < sunrise:
        rim = smooth(1 - (sunrise - minute) / TWILIGHT_MIN)
    elif minute > sunset:
        rim = smooth(1 - (minute - sunset) / TWILIGHT_MIN)
    else:
        rim = 1.0
    twilight = rim * TWILIGHT_LEVEL
    daylight = max(smooth(elevation * 2.2), TWILIGHT_LEVEL if elevation > 0 else twilight)
    glow = math.exp(-(((elevation - 0.12) / 0.13) ** 2)) if elevation > 0 else 0.0
    lamps = _lamp_factor(minute / 60, daylight, lamps_on, lamp_hours)
    phase = (minute - sunrise) / day_len if elevation > 0 else 0.5
    return PageFactors(elevation, twilight, daylight, glow, lamps, phase)


def _lamp_factor(hour: float, daylight: float, lamps_on: bool, hours: tuple[float, float]) -> float:
    """lightFrame's lamps: on in their hours while daylight < 0.7, dimmed by what is left of it."""
    return 1 - daylight if lamps_on and daylight < 0.7 and _in_hours(hour, hours) else 0.0


# ── colours, the way three.js blends them ─────────────────────────────────


def _to_linear(c: float) -> float:
    return c * 0.0773993808 if c < 0.04045 else (c * 0.9478672986 + 0.0521327014) ** 2.4


def _to_srgb(c: float) -> float:
    return c * 12.92 if c < 0.0031308 else 1.055 * c**0.41666 - 0.055


def linear_rgb(hex_color: int) -> RGB:
    """An sRGB hex colour in linear light (three.js ``new Color(hex)``)."""
    return (
        _to_linear((hex_color >> 16 & 255) / 255),
        _to_linear((hex_color >> 8 & 255) / 255),
        _to_linear((hex_color & 255) / 255),
    )


def hex_string(rgb: RGB) -> str:
    """Linear light back to ``#rrggbb`` (three.js ``getHexString``)."""
    parts = (round(min(255.0, max(0.0, _to_srgb(c) * 255))) for c in rgb)
    return "#" + "".join(f"{part:02x}" for part in parts)


def _lerp(a: RGB, b: RGB, t: float) -> RGB:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


def luminance(rgb: RGB) -> float:
    """Rec. 709 luminance of a linear colour."""
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def unit_luminance(rgb: RGB) -> RGB:
    """Scaled so its luminance (Rec. 709, linear) is 1: a colour that carries no brightness."""
    y = luminance(rgb)
    return (rgb[0] / y, rgb[1] / y, rgb[2] / y) if y > 0 else (1.0, 1.0, 1.0)


def _unit(v: Vec3) -> Vec3:
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return (v[0] / n, v[1] / n, v[2] / n)


def _sun_offset(elevation: float, phase: float) -> Vec3:
    """duck3d.js: where the page puts its sun relative to the duck, in three.js axes."""
    return (math.cos(math.pi * phase) * 1.1, 0.35 + 1.1 * max(elevation, 0.35), 0.5)


def _mujoco(three: Vec3) -> Vec3:
    """three.js (x, y up, z) → MuJoCo (x, y, z up)."""
    return (three[0], -three[2], three[1])


def _three(mujoco: Vec3) -> list[float]:
    """MuJoCo (x, y, z up) → three.js (x, y up, z), rounded for the page."""
    return [round(mujoco[0], 4), round(mujoco[2], 4), round(-mujoco[1], 4)]


# ── how a lamp's light falls off ───────────────────────────────────────────


def fixture(position: Vec3, ceiling: float) -> Vec3:
    """Where a lamp standing at ``position`` shines from: just above the ceiling over it."""
    return (position[0], position[1], max(position[2], ceiling + LAMP_ABOVE_CEILING_M))


def reach(source: Vec3, point: Vec3, attenuation: Sequence[float] = LAMP_ATTENUATION) -> float:
    """Direct light from a point source at ``source`` on a level surface at ``point``, per unit
    of the source's strength: the incidence cosine over MuJoCo's attenuation polynomial (the
    renderer's own lighting equation, so a render matches this everywhere)."""
    dx, dy, dz = source[0] - point[0], source[1] - point[1], source[2] - point[2]
    r = math.sqrt(dx * dx + dy * dy + dz * dz)
    if r <= 1e-9:
        return 0.0
    c0, c1, c2 = (float(v) for v in attenuation)
    return max(dz / r, 0.05) / (c0 + c1 * r + c2 * r * r)


# ── the state ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LightSource:
    """One light for a renderer, in the MuJoCo world frame (z up).

    ``kind``: ``directional`` (the sun, the moon, a window's sky), ``spot`` (a lamp shining down
    over the half-space below its fixture) or ``ambient`` (light that only arrives bounced, such
    as the daylight a windowless corridor gets from the rooms around it).
    """

    name: str  # "sun", "lamp", "moon", or "window:<id>", "lamp:<id>", "spill" in a home
    kind: str
    lux: float  # at the reference spot on a level surface, direct and bounced together
    indirect: float  # the share of `lux` that arrives as direction-less room light
    color: RGB  # linear RGB with luminance 1
    toward: Vec3 | None = None  # directional: unit vector from the room toward the light
    position: Vec3 | None = None  # spot: where the lamp stands (m)
    fixture: Vec3 | None = None  # spot: where its light is drawn from (above the ceiling)
    intensity: float | None = None  # spot: direct lux = intensity × reach(fixture, point)
    room: str | None = None
    id: str | None = None  # the layout's lamp or window id

    def direct_at(self, point: Vec3) -> float:
        """Direct lux on a level surface at ``point`` (bounced light not included)."""
        if self.kind == "spot" and self.fixture is not None and self.intensity is not None:
            return self.intensity * reach(self.fixture, point)
        if self.kind == "ambient":
            return 0.0
        return self.lux * (1.0 - self.indirect)  # directional: the same on every level surface


@dataclass(frozen=True)
class LightState:
    when: str  # the duck's local date and time, ISO
    minute: float  # of the day
    sunrise: float  # minutes after midnight
    sunset: float
    elevation: float  # the page's sun factor (0 at night)
    daylight: float
    glow: float
    lamps: float
    phase: float
    lux: float  # total at the reference spot
    key: str  # day / dawn / dusk / lamps / moon / black
    label: str  # plain words for the owner
    sources: tuple[LightSource, ...]
    room: str | None = None  # a home's room; None for phase 0's one room
    reference: Vec3 = REFERENCE_SPOT  # where `lux` and each source's `lux` apply
    lamp_position: Vec3 = LAMP_POSITION  # the room's main lamp, for the page

    def lux_by_source(self) -> dict[str, float]:
        return {source.name: source.lux for source in self.sources}

    def lux_at(self, point: Vec3) -> float:
        """Lux on a level surface at ``point`` in this room: every source's direct light with
        its own fall-off, plus the bounced light, which is the same all over the room."""
        return sum(s.direct_at(point) + s.lux * s.indirect for s in self.sources)

    def three(self) -> dict[str, Any]:
        """The page's Three.js lights, as ``lightFrame`` sets them (before its easing)."""
        d, glow, lamps = self.daylight, self.glow, self.lamps
        hemi = _lerp(linear_rgb(NIGHT_LIGHT), linear_rgb(DAY_LIGHT), max(d, lamps * 0.8))
        sun = _lerp(linear_rgb(SUN_DAY), linear_rgb(SUN_WARM), glow * 0.8)
        sky = _lerp(linear_rgb(NIGHT_SKY), linear_rgb(DAY_SKY), d)
        sky = _lerp(sky, linear_rgb(DUSK_SKY), glow * 0.55)
        offset = _sun_offset(self.elevation, self.phase)
        lamp = self.lamp_position
        return {
            "hemi": {"intensity": 0.28 + 0.97 * d + 0.25 * lamps, "color": hex_string(hemi)},
            "sun": {
                "intensity": 1.6 * d + 0.12,
                "color": hex_string(sun),
                # relative to the point the view follows (the duck), three.js axes
                "offset": [round(value, 4) for value in offset],
            },
            "fill": {"intensity": 0.15 + 0.35 * d},
            "lamp": {
                "intensity": 2.2 * lamps,
                "color": f"#{LAMP_COLOR:06x}",
                "position": [lamp[0], lamp[2], -lamp[1]],
            },
            "background": hex_string(sky),
        }

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["sources"] = [asdict(source) for source in self.sources]
        out["lux_by_source"] = self.lux_by_source()
        out["three"] = self.three()
        return out


def daylight_lux(daylight: float) -> float:
    return DAY_LUX * daylight**DAYLIGHT_EXPONENT if daylight > 0 else 0.0


def _label(minute: float, day: float, lamp: float, moon: float) -> str:
    if lamp > 0 and lamp >= day and lamp >= moon:
        return "lamps"
    if day > 0 and day >= moon:
        if day >= DAY_MIN_LUX:
            return "day"
        return "dusk" if minute >= 12 * 60 else "dawn"
    return "moon" if day + lamp + moon >= MOON_MIN_LUX else "black"


# ── a home: rooms, lamps, windows, the nest ───────────────────────────────


@dataclass(frozen=True)
class Lamp:
    """A lamp the family switches on in its hours (spec §12.1 ``lamps``, ``night_lights``)."""

    id: str
    position: Vec3  # where it stands (m); its light is drawn from the ceiling above
    color: int = LAMP_COLOR  # sRGB hex
    on_hours: tuple[float, float] = LAMP_HOURS  # [on, off) local hours; may wrap midnight
    # Its lux on the floor in the middle of its room when fully on (a layout's ``lux``); None:
    # its ``power`` share of LAMP_LUX, so the lamps of a room add to 120 lux together.
    lux: float | None = None
    power: float = 1.0
    kind: str = "lamp"  # "ceiling", "pendant", "floor_lamp", "table", … or "night"


@dataclass(frozen=True)
class Window:
    """A window: daylight and moonlight come in through it (spec §12.1 ``windows``)."""

    id: str
    position: Vec3  # its middle (m)
    size: tuple[float, float]  # width, height (m)
    facing: Point  # unit vector of the way it looks out (its outward normal), level

    def toward(self, elevation_deg: float = WINDOW_SKY_ELEVATION_DEG) -> Vec3:
        """Unit vector from the room toward the sky light that comes in through it."""
        e = math.radians(elevation_deg)
        return _unit((self.facing[0] * math.cos(e), self.facing[1] * math.cos(e), math.sin(e)))

    @property
    def area(self) -> float:
        return max(0.0, self.size[0]) * max(0.0, self.size[1])


def _point_in_polygon(x: float, y: float, polygon: Sequence[Point]) -> bool:
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            if x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                inside = not inside
    return inside


def _segment_distance(x: float, y: float, a: Point, b: Point) -> float:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length = dx * dx + dy * dy
    t = 0.0 if length <= 0 else max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / length))
    return math.hypot(x - (ax + t * dx), y - (ay + t * dy))


def rectangle(x0: float, y0: float, x1: float, y1: float) -> tuple[Point, ...]:
    lo_x, hi_x = min(x0, x1), max(x0, x1)
    lo_y, hi_y = min(y0, y1), max(y0, y1)
    return ((lo_x, lo_y), (hi_x, lo_y), (hi_x, hi_y), (lo_x, hi_y))


@dataclass(frozen=True)
class Room:
    """A room's footprint and its light fittings."""

    id: str
    polygon: tuple[Point, ...]  # footprint in the world frame (m), in order around the room
    ceiling: float = DEFAULT_CEILING_M  # height above the floor (m)
    floor: float = 0.0  # floor height (m)
    lamps: tuple[Lamp, ...] = ()
    windows: tuple[Window, ...] = ()
    # Phase 0's room: daylight from the page's moving sun instead of through windows.
    sky: bool = False
    zh: str = ""

    def contains(self, x: float, y: float) -> bool:
        return _point_in_polygon(x, y, self.polygon)

    def distance(self, x: float, y: float) -> float:
        """0 inside, else how far the point is from the room's outline (m)."""
        if self.contains(x, y):
            return 0.0
        n = len(self.polygon)
        return min(
            _segment_distance(x, y, self.polygon[i], self.polygon[(i + 1) % n]) for i in range(n)
        )

    @property
    def centre(self) -> Point:
        """The middle of the footprint (area centroid; a rectangle's centre)."""
        area = cx = cy = 0.0
        n = len(self.polygon)
        for i in range(n):
            x1, y1 = self.polygon[i]
            x2, y2 = self.polygon[(i + 1) % n]
            cross = x1 * y2 - x2 * y1
            area += cross
            cx += (x1 + x2) * cross
            cy += (y1 + y2) * cross
        if abs(area) < 1e-12:
            xs = [p[0] for p in self.polygon]
            ys = [p[1] for p in self.polygon]
            return ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
        return (cx / (3 * area), cy / (3 * area))

    @property
    def reference(self) -> Vec3:
        """Where the room's lux figures apply: the middle of its floor."""
        cx, cy = self.centre
        return (cx, cy, self.floor)

    @property
    def top(self) -> float:
        """The ceiling's height in the world (m)."""
        return self.floor + self.ceiling


@dataclass(frozen=True)
class NestLight:
    """Where the nest's backlight is (the backboard's face), whether this nest has one, and its
    colours (sRGB hex): lit, and while the duck charges."""

    position: Vec3
    backlight: bool = True
    color: int = BACKLIGHT_COLOR
    charging_color: int = BACKLIGHT_CHARGING_COLOR
    nits: float = BACKLIGHT_NITS


@dataclass(frozen=True)
class HomeLights:
    """A home's rooms with their lamps and windows, and the nest: what :func:`home_light_state`
    needs. Build it from a home layout with :meth:`from_layout`."""

    id: str
    rooms: tuple[Room, ...]
    nest: NestLight | None = None

    def room(self, room_id: str) -> Room:
        for room in self.rooms:
            if room.id == room_id:
                return room
        raise KeyError(f"no room {room_id!r} in {self.id}")

    def room_at(self, x: float, y: float) -> Room:
        """The room the point is in; outside every room (a doorway gap, a wall), the nearest."""
        inside = [room for room in self.rooms if room.contains(x, y)]
        if inside:
            return inside[0]
        return min(self.rooms, key=lambda room: room.distance(x, y))

    @property
    def ceiling(self) -> float:
        """The highest ceiling in the world frame (m)."""
        return max(room.top for room in self.rooms)

    @classmethod
    def from_layout(cls, layout: Any) -> HomeLights:
        """A home layout's rooms, lamps, windows and nest (see :func:`home_lights_from_layout`)."""
        return home_lights_from_layout(layout)


DEFAULT_ROOM = Room(
    id="home",
    polygon=rectangle(-3.0, -3.0, 3.0, 3.0),
    ceiling=DEFAULT_CEILING_M,
    lamps=(Lamp("lamp", LAMP_POSITION),),
    sky=True,
    zh=msg("world.lights.DEFAULT_ROOM.Room_zh"),
)
# Phase 0's one room, as the page draws it: 6×6 m, one lamp, daylight from the page's sun.
DEFAULT_HOME = HomeLights("default_home", (DEFAULT_ROOM,))


# ── a room's light at a time ────────────────────────────────────────────────


def _clock(when: datetime) -> tuple[float, float, float]:
    minute = (when.hour * 60 + when.minute + when.second / 60 + when.microsecond / 6e7) % 1440
    sunrise, sunset = sun_times(when.date())
    return minute, sunrise, sunset


def _lamp_source(room: Room, lamp: Lamp, lux: float, reference: Vec3) -> LightSource:
    at = fixture(lamp.position, room.top)
    room_id = None if room is DEFAULT_ROOM else room.id
    direct = lux * (1.0 - LAMP_INDIRECT)
    return LightSource(
        "lamp" if lamp.id == "lamp" else f"lamp:{lamp.id}",
        "spot",
        lux,
        LAMP_INDIRECT,
        unit_luminance(linear_rgb(lamp.color)),
        position=lamp.position,
        fixture=at,
        intensity=direct / max(reach(at, reference), 1e-12),
        room=room_id,
        id=lamp.id,
    )


def _mix(parts: Sequence[tuple[float, RGB]]) -> RGB:
    """Lux-weighted mean colour, back to unit luminance."""
    total = sum(w for w, _ in parts)
    if total <= 0:
        return (1.0, 1.0, 1.0)
    rgb = tuple(sum(w * c[i] for w, c in parts) / total for i in range(3))
    return unit_luminance((rgb[0], rgb[1], rgb[2]))


def room_light_state(
    when: datetime,
    room: Room = DEFAULT_ROOM,
    *,
    lamps_on: bool = True,
    moon: float = 1.0,
    lamp_hours: tuple[float, float] | None = None,
) -> LightState:
    """One room's light at the duck's local time ``when``.

    Daylight and moonlight come through the room's windows (shared by window area; a room
    without windows gets :data:`SPILL` of them as bounced light only), its lamps give
    :data:`LAMP_LUX` together at the room's middle when all are on (each its ``power`` share),
    a night light :data:`NIGHT_LIGHT_LUX`. ``lamp_hours`` overrides every lamp's own hours.
    """
    minute, sunrise, sunset = _clock(when)
    hours = lamp_hours or LAMP_HOURS
    f = page_factors(minute, sunrise, sunset, lamps_on=lamps_on, lamp_hours=hours)
    moon = min(1.0, max(0.0, moon))
    reference = room.reference
    day_lux = daylight_lux(f.daylight)
    night = 1.0 - min(1.0, f.daylight / TWILIGHT_LEVEL)
    moon_lux = (BLACK_LUX + moon * (MOON_LUX - BLACK_LUX)) * night
    sun_color = unit_luminance(_lerp(linear_rgb(SUN_DAY), linear_rgb(SUN_WARM), f.glow * 0.8))
    moon_color = unit_luminance(linear_rgb(NIGHT_LIGHT))
    sources: list[LightSource] = []
    windows = [w for w in room.windows if w.area > 0]
    spill = not room.sky and not windows
    day_in = day_lux * (SPILL if spill else 1.0)
    moon_in = moon_lux * (SPILL if spill else 1.0)

    if room.sky and day_lux > 0:
        sources.append(
            LightSource(
                "sun",
                "directional",
                day_lux,
                SUN_INDIRECT,
                sun_color,
                toward=_unit(_mujoco(_sun_offset(f.elevation, f.phase))),
                room=room.id if room is not DEFAULT_ROOM else None,
            )
        )
    elif windows and day_lux + moon_lux > 0:
        area = sum(w.area for w in windows)
        for w in windows:
            share = w.area / area
            day, night_part = day_lux * share, moon_lux * share
            sources.append(
                LightSource(
                    f"window:{w.id}",
                    "directional",
                    day + night_part,
                    (day * SUN_INDIRECT + night_part * MOON_INDIRECT) / (day + night_part),
                    _mix(((day, sun_color), (night_part, moon_color))),
                    toward=w.toward(),
                    room=room.id,
                    id=w.id,
                )
            )
    elif spill and day_in + moon_in > 0:
        sources.append(
            LightSource(
                "spill",
                "ambient",
                day_in + moon_in,
                1.0,
                _mix(((day_in, sun_color), (moon_in, moon_color))),
                room=room.id,
            )
        )

    lamp_total = 0.0
    lamp_factor = 0.0
    day_power = sum(max(0.0, lamp.power) for lamp in room.lamps if lamp.kind != "night")
    for lamp in room.lamps:
        if lamp.kind == "night":
            on = f.daylight < TWILIGHT_LEVEL and _in_hours(minute / 60, lamp.on_hours)
            lux = NIGHT_LIGHT_LUX * max(0.0, lamp.power) if on else 0.0
        else:
            factor = _lamp_factor(minute / 60, f.daylight, lamps_on, lamp_hours or lamp.on_hours)
            lamp_factor = max(lamp_factor, factor)
            if lamp.lux is not None:
                lux = max(0.0, lamp.lux) * factor
            else:
                share = max(0.0, lamp.power) / day_power if day_power > 0 else 0.0
                lux = LAMP_LUX * factor * share
        if lux > 0:
            lamp_total += lux
            sources.append(_lamp_source(room, lamp, lux, reference))

    if room.sky and moon_lux > 0:
        sources.append(
            LightSource(
                "moon",
                "directional",
                moon_lux,
                MOON_INDIRECT,
                moon_color,
                toward=_unit(_mujoco(_sun_offset(0.0, 0.5))),
                room=room.id if room is not DEFAULT_ROOM else None,
            )
        )

    key = _label(minute, day_in, lamp_total, moon_in)
    main_lamp = next((lamp.position for lamp in room.lamps if lamp.kind != "night"), LAMP_POSITION)
    return LightState(
        when=when.replace(tzinfo=None).isoformat(timespec="seconds"),
        minute=minute,
        sunrise=sunrise,
        sunset=sunset,
        elevation=f.elevation,
        daylight=f.daylight,
        glow=f.glow,
        lamps=lamp_factor if room.lamps else f.lamps,
        phase=f.phase,
        lux=day_in + lamp_total + moon_in,
        key=key,
        label=LABELS[key],
        sources=tuple(sources),
        room=room.id if room is not DEFAULT_ROOM else None,
        reference=reference,
        lamp_position=main_lamp,
    )


def light_state(
    when: datetime,
    *,
    lamps_on: bool = True,
    moon: float = 1.0,
    lamp_hours: tuple[int, int] = LAMP_HOURS,
) -> LightState:
    """The light at home at the duck's local time ``when`` (naive or any tz: its wall clock).

    Phase 0's one room (:data:`DEFAULT_ROOM`). ``lamps_on=False`` is an evening the family
    leaves the lamps off; ``moon`` scales the night light between pitch black (0) and the page's
    moonlight (1, the default).
    """
    return room_light_state(when, DEFAULT_ROOM, lamps_on=lamps_on, moon=moon, lamp_hours=lamp_hours)


# ── the nest's backlight ────────────────────────────────────────────────────


@dataclass(frozen=True)
class BacklightState:
    """The nest's backlight at a moment: an emitter of absolute brightness (C4)."""

    on: bool
    charging: bool
    color: RGB  # linear RGB, brightest channel 1
    nits: float  # luminance while on (cd/m²)
    lux_at_nest: float  # the room's light where the nest's own light sensor is
    position: Vec3 | None = None
    room: str | None = None
    srgb: int = BACKLIGHT_COLOR  # the colour as the layout gives it (sRGB hex)

    @property
    def scene_lux(self) -> float:
        """Its brightness on the camera's scale (lux × albedo): a surface of luminance L looks
        like a white card under π·L lux."""
        return math.pi * self.nits if self.on else 0.0

    @property
    def hex(self) -> str:
        return f"#{self.srgb:06x}"


def backlight_state(
    lux_at_nest: float,
    *,
    charging: bool = False,
    on_below_lux: float = BACKLIGHT_ON_BELOW_LUX,
    nits: float = BACKLIGHT_NITS,
    position: Vec3 | None = None,
    room: str | None = None,
    color: int = BACKLIGHT_COLOR,
    charging_color: int = BACKLIGHT_CHARGING_COLOR,
) -> BacklightState:
    """C4: on only when the nest's corner is dark; another colour while the duck charges."""
    srgb = charging_color if charging else color
    rgb = linear_rgb(srgb)
    peak = max(rgb)
    return BacklightState(
        on=lux_at_nest < on_below_lux,
        charging=charging,
        color=(rgb[0] / peak, rgb[1] / peak, rgb[2] / peak),
        nits=nits,
        lux_at_nest=lux_at_nest,
        position=position,
        room=room,
        srgb=srgb,
    )


# ── the whole home at a time ────────────────────────────────────────────────


@dataclass(frozen=True)
class HomeLightState:
    """Every room's light at one moment, and the nest's backlight."""

    when: str
    home: str
    rooms: dict[str, LightState] = field(default_factory=dict)
    backlight: BacklightState | None = None
    lights: HomeLights | None = None

    def room(self, room_id: str) -> LightState:
        return self.rooms[room_id]

    def at(self, x: float, y: float) -> LightState:
        """The state of the room the point is in (or the nearest room)."""
        if self.lights is None:
            return next(iter(self.rooms.values()))
        return self.rooms[self.lights.room_at(x, y).id]

    def lux_at(self, point: Vec3) -> float:
        """Lux on a level surface at ``point`` (for HomeWorld and evaluators)."""
        return self.at(point[0], point[1]).lux_at(point)

    def to_page(self) -> dict[str, Any]:
        """What the page draws: each room's key, label, lux and Three.js factors; every lamp
        (three.js axes) with its colour, whether it is on and a Three.js intensity like
        ``lightFrame``'s lamp; every window with its daylight; the backlight."""
        rooms: dict[str, Any] = {}
        lamps: list[dict[str, Any]] = []
        windows: list[dict[str, Any]] = []
        for room_id, state in self.rooms.items():
            rooms[room_id] = {
                "key": state.key,
                "label": state.label,
                "lux": round(state.lux, 3),
                "three": state.three(),
            }
            by_id = {s.id: s for s in state.sources if s.id is not None}
            room = self.lights.room(room_id) if self.lights is not None else None
            for lamp in room.lamps if room is not None else ():
                source = by_id.get(lamp.id)
                share = source.lux / LAMP_LUX if source is not None else 0.0
                lamps.append(
                    {
                        "id": lamp.id,
                        "room": room_id,
                        "kind": lamp.kind,
                        "position": _three(lamp.position),
                        "color": f"#{lamp.color:06x}",
                        "on": source is not None,
                        "intensity": round(2.2 * share, 4),
                        "lux": round(source.lux, 3) if source is not None else 0.0,
                    }
                )
            for window in room.windows if room is not None else ():
                source = by_id.get(window.id)
                windows.append(
                    {
                        "id": window.id,
                        "room": room_id,
                        "position": _three(window.position),
                        "size": [round(window.size[0], 4), round(window.size[1], 4)],
                        "toward": _three(window.toward()),
                        "lux": round(source.lux, 3) if source is not None else 0.0,
                        "color": hex_string(source.color) if source is not None else None,
                    }
                )
        backlight = None
        if self.backlight is not None:
            b = self.backlight
            backlight = {
                "on": b.on,
                "charging": b.charging,
                "color": b.hex,
                "nits": b.nits,
                "position": _three(b.position) if b.position is not None else None,
            }
        return {
            "when": self.when,
            "home": self.home,
            "rooms": rooms,
            "lamps": lamps,
            "windows": windows,
            "backlight": backlight,
        }


def home_light_state(
    when: datetime,
    home: HomeLights = DEFAULT_HOME,
    *,
    lamps_on: bool = True,
    moon: float = 1.0,
    charging: bool = False,
    lamp_hours: tuple[float, float] | None = None,
) -> HomeLightState:
    """Every room of ``home`` at the duck's local time ``when``, and the nest's backlight
    (``charging``: the duck sits on its charging contact — the nest or battery owner says)."""
    rooms = {
        room.id: room_light_state(when, room, lamps_on=lamps_on, moon=moon, lamp_hours=lamp_hours)
        for room in home.rooms
    }
    backlight = None
    if home.nest is not None and home.nest.backlight:
        where = home.nest.position
        room = home.room_at(where[0], where[1])
        floor = (where[0], where[1], room.floor)
        backlight = backlight_state(
            rooms[room.id].lux_at(floor),
            charging=charging,
            position=where,
            room=room.id,
            color=home.nest.color,
            charging_color=home.nest.charging_color,
            nits=home.nest.nits,
        )
    return HomeLightState(
        when=when.replace(tzinfo=None).isoformat(timespec="seconds"),
        home=home.id,
        rooms=rooms,
        backlight=backlight,
        lights=home,
    )


# ── reading a home layout ───────────────────────────────────────────────────

_MISSING = object()


def _get(obj: Any, *names: str, default: Any = _MISSING) -> Any:
    """``obj.name`` or ``obj[name]`` for the first name present (a no-argument method is
    called)."""
    for name in names:
        if isinstance(obj, Mapping):
            if name in obj:
                return obj[name]
        elif hasattr(obj, name):
            value = getattr(obj, name)
            if callable(value) and not isinstance(value, type):
                value = value()
            return value
    if default is _MISSING:
        raise KeyError(names[0])
    return default


def _keyed(obj: Any, names: Sequence[str]) -> tuple[str, Any] | None:
    """The first of ``names`` that ``obj`` has, with its value."""
    absent = object()
    for name in names:
        value = _get(obj, name, default=absent)
        if value is not absent:
            return name, value
    return None


def _length(values: Any, key: str) -> list[float]:
    """Lengths in metres: ``*_mm`` keys are millimetres, ``*_m`` metres; an unmarked key is
    read as millimetres when any value is larger than 20 (no room is 20 m across)."""
    if isinstance(values, (int, float)):
        items = [float(values)]
    else:
        items = [float(v) for v in values]
    if key.endswith("_mm"):
        return [v / 1000 for v in items]
    if key.endswith("_m"):
        return items
    return [v / 1000 for v in items] if any(abs(v) > 20 for v in items) else items


def _points(values: Any, key: str) -> list[Point]:
    flat: list[float] = []
    for p in values:
        flat.extend(float(v) for v in (p if not isinstance(p, Mapping) else (p["x"], p["y"])))
    metres = _length(flat, key)
    return [(metres[i], metres[i + 1]) for i in range(0, len(metres), 2)]


def _angle(value: Any, key: str) -> float:
    v = float(value)
    if key.endswith("_mrad"):
        return v / 1000
    if key.endswith("_deg"):
        return math.radians(v)
    if key.endswith("_rad"):
        return v
    return v / 1000 if abs(v) > 2 * math.pi + 1e-9 else v


def _color(value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip().lstrip("#")
        return int(text, 16)
    parts = [float(v) for v in value]
    if all(p <= 1.0 for p in parts):
        parts = [p * 255 for p in parts]
    r, g, b = (max(0, min(255, round(p))) for p in parts[:3])
    return (r << 16) | (g << 8) | b


def _hours(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        return default

    def one(v: Any) -> float:
        if isinstance(v, str) and ":" in v:
            hh, mm = v.split(":", 1)
            return int(hh) + int(mm) / 60
        return float(v)

    items = list(value)
    return (one(items[0]), one(items[1]))


def _footprint(room: Any) -> tuple[Point, ...]:
    found = _keyed(room, ("polygon_mm", "poly_mm", "polygon_m", "polygon", "poly"))
    if found:
        return tuple(_points(found[1], found[0]))
    found = _keyed(room, ("rect_mm", "bounds_mm", "rect_m", "bounds_m", "rect", "bounds"))
    if found:
        x0, y0, x1, y1 = _length(found[1], found[0])
        return rectangle(x0, y0, x1, y1)
    lo = _keyed(room, ("min_mm", "min_m", "min"))
    hi = _keyed(room, ("max_mm", "max_m", "max"))
    if lo and hi:
        x0, y0 = _length(lo[1], lo[0])[:2]
        x1, y1 = _length(hi[1], hi[0])[:2]
        return rectangle(x0, y0, x1, y1)
    mid = _keyed(room, ("center_mm", "centre_mm", "pos_mm", "center", "centre", "pos"))
    size = _keyed(room, ("size_mm", "size_m", "size"))
    if mid and size:
        cx, cy = _length(mid[1], mid[0])[:2]
        w, h = _length(size[1], size[0])[:2]
        return rectangle(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    raise ValueError(f"room {_get(room, 'id', default='?')!r} has no footprint")


def _position(item: Any, default_z: float = DEFAULT_CEILING_M - 0.1) -> Vec3:
    found = _keyed(item, ("pos_mm", "position_mm", "xyz_mm", "pos_m", "pos", "position", "xyz"))
    if found is None:
        raise ValueError(f"{_get(item, 'id', default='?')!r} has no position")
    values = _length(found[1], found[0])
    if len(values) == 2:  # no height given: a ceiling lamp, a window at chest height
        return (values[0], values[1], default_z)
    return (values[0], values[1], values[2])


def _lamp(item: Any, ceiling: float, default_kind: str = "lamp") -> Lamp:
    kind = str(_get(item, "kind", "type", default=default_kind) or default_kind)
    night = kind == "night" or default_kind == "night"
    lux = _get(item, "lux", default=None)
    return Lamp(
        id=str(_get(item, "id", "name")),
        position=_position(item, ceiling - 0.1),
        color=_color(_get(item, "colour", "color", "rgb", default=None), LAMP_COLOR),
        on_hours=_hours(
            _get(item, "on_hours", "hours", "schedule", default=None),
            (23.0, 7.0) if night else LAMP_HOURS,
        ),
        lux=float(lux) if lux is not None else None,
        power=float(_get(item, "power", "share", "weight", default=1.0)),
        kind="night" if night else kind,
    )


def _window(item: Any, centre: Point, floor: float = 0.0) -> Window:
    position = _position(item, floor + 1.2)
    size_key = _keyed(item, ("size_mm", "size_m", "size"))
    width = _keyed(item, ("width_mm", "width_m", "width"))
    height = _keyed(item, ("height_mm", "height_m", "height"))
    if size_key:
        size = _length(size_key[1], size_key[0])[:2]
    elif width and height:
        size = [_length(width[1], width[0])[0], _length(height[1], height[0])[0]]
        # home-layout-v2 gives the sill (the window's foot) with its width and height
        position = (position[0], position[1], position[2] + size[1] / 2)
    else:
        size = [1.2, 1.2]
    found = _keyed(item, ("facing_mrad", "facing_deg", "facing_rad", "azimuth_mrad", "facing"))
    inward = _keyed(item, ("inward_mrad", "inward_deg", "inward_rad", "inward"))
    if found is None and inward is None:
        found = _keyed(item, ("normal", "outward"))
    if inward is not None:  # the way daylight travels in: the window looks the other way
        a = _angle(inward[1], inward[0]) + math.pi
        fx, fy = math.cos(a), math.sin(a)
    elif found is None:  # look away from the room's middle
        fx, fy = position[0] - centre[0], position[1] - centre[1]
    elif isinstance(found[1], (int, float)):
        a = _angle(found[1], found[0])
        fx, fy = math.cos(a), math.sin(a)
    else:
        fx, fy = (float(v) for v in list(found[1])[:2])
    n = math.hypot(fx, fy) or 1.0
    fx, fy = fx / n, fy / n
    # "facing" is the way the window looks out; one that points into the room is turned round.
    if (position[0] - centre[0]) * fx + (position[1] - centre[1]) * fy < 0:
        fx, fy = -fx, -fy
    return Window(
        id=str(_get(item, "id", "name")),
        position=position,
        size=(size[0], size[1]),
        facing=(fx, fy),
    )


def _items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return list(value.values())
    return list(value)


def home_lights_from_layout(layout: Any) -> HomeLights:
    """A :class:`HomeLights` from a home layout (``world/homelayout.py``'s ``Layout`` or its
    JSON), read by name so it works on an object or a dict:

    * ``id``; ``rooms``: each with ``id`` and a footprint (``polygon_mm`` / ``rect_mm`` =
      [x0, y0, x1, y1] / ``min_mm`` + ``max_mm`` / ``center_mm`` + ``size_mm``), optional
      ``ceiling_mm`` and ``floor_mm``;
    * ``lights``: per room (a mapping room id → {lamps, windows, night_lights}, or a list of
      {room, lamps, windows, night_lights}), or ``lamps`` / ``windows`` lists on the rooms or
      on the layout (each with ``room``, or placed by where it stands). A lamp is
      {id, pos_mm, colour, on_hours, power?, kind?}; a window {id, pos_mm, size_mm,
      facing_mrad (the way it looks out)};
    * ``nest``: {x_mm, y_mm | pose_mm, backlight?} (optional).

    Units: ``*_mm`` millimetres, ``*_mrad`` milliradians (integers, as the layouts keep them).
    """
    if isinstance(layout, HomeLights):
        return layout
    maybe = _get(layout, "lights", default=None)
    if isinstance(maybe, HomeLights):
        return maybe
    home_id = str(_get(layout, "id", "name", default="home"))
    ceiling_key = _keyed(layout, ("ceiling_mm", "ceiling_m", "ceiling"))
    home_ceiling = _length(ceiling_key[1], ceiling_key[0])[0] if ceiling_key else None
    raw_rooms = _items(_get(layout, "rooms", default=None))
    if not raw_rooms:
        raise ValueError(f"the layout {home_id!r} has no rooms")
    shells: list[dict[str, Any]] = []
    for raw in raw_rooms:
        c_key = _keyed(raw, ("ceiling_mm", "ceiling_m", "ceiling"))
        f_key = _keyed(raw, ("floor_mm", "floor_m", "floor_z_mm"))
        floor = _length(f_key[1], f_key[0])[0] if f_key and not isinstance(f_key[1], str) else 0.0
        ceiling = (
            _length(c_key[1], c_key[0])[0]
            if c_key
            else home_ceiling
            if home_ceiling is not None
            else DEFAULT_CEILING_M
        )
        shells.append(
            {
                "id": str(_get(raw, "id", "name")),
                "zh": str(_get(raw, "zh", default="") or ""),
                "polygon": _footprint(raw),
                "ceiling": ceiling,
                "floor": floor,
                "lamps": list(_items(_get(raw, "lamps", default=None))),
                "night": list(_items(_get(raw, "night_lights", default=None))),
                "windows": list(_items(_get(raw, "windows", default=None))),
            }
        )
    by_id = {shell["id"]: shell for shell in shells}

    def place(item: Any, bucket: str, room_id: Any = None) -> None:
        rid = room_id if room_id is not None else _get(item, "room", default=None)
        if rid is None:
            x, y, _ = _position(item)
            probe = [Room(s["id"], s["polygon"], s["ceiling"], s["floor"]) for s in shells]
            rid = min(probe, key=lambda r: r.distance(x, y)).id
        if str(rid) not in by_id:
            raise ValueError(f"light {_get(item, 'id', default='?')!r} is in unknown room {rid!r}")
        by_id[str(rid)][bucket].append(item)

    lights = _get(layout, "lights", default=None)
    flat = lights is None or any(
        _get(lights, key, default=None) is not None for key in ("lamps", "windows", "night_lights")
    )
    if isinstance(lights, Mapping) and not flat:
        for rid, entry in lights.items():
            for item in _items(_get(entry, "lamps", default=None)):
                place(item, "lamps", rid)
            for item in _items(_get(entry, "night_lights", default=None)):
                place(item, "night", rid)
            for item in _items(_get(entry, "windows", default=None)):
                place(item, "windows", rid)
    elif flat:
        source = lights if lights is not None else layout
        for item in _items(_get(source, "lamps", default=None)):
            place(item, "lamps")
        for item in _items(_get(source, "night_lights", default=None)):
            place(item, "night")
        for item in _items(_get(source, "windows", default=None)):
            place(item, "windows")
    else:
        for entry in _items(lights):
            rid = _get(entry, "room", "id", default=None)
            for item in _items(_get(entry, "lamps", default=None)):
                place(item, "lamps", rid)
            for item in _items(_get(entry, "night_lights", default=None)):
                place(item, "night", rid)
            for item in _items(_get(entry, "windows", default=None)):
                place(item, "windows", rid)

    rooms: list[Room] = []
    for s in shells:
        probe = Room(s["id"], s["polygon"], s["ceiling"], s["floor"])
        centre = probe.centre
        rooms.append(
            Room(
                id=s["id"],
                polygon=s["polygon"],
                ceiling=s["ceiling"],
                floor=s["floor"],
                lamps=tuple(
                    [_lamp(item, probe.top) for item in s["lamps"]]
                    + [_lamp(item, probe.top, "night") for item in s["night"]]
                ),
                windows=tuple(_window(item, centre, s["floor"]) for item in s["windows"]),
                zh=s["zh"],
            )
        )
    return HomeLights(home_id, tuple(rooms), _nest(layout, rooms))


def _nest(layout: Any, rooms: Iterable[Room]) -> NestLight | None:
    nest = _get(layout, "nest", default=None)
    if not nest:
        return None
    xs = _keyed(nest, ("x_mm", "x_m", "x"))
    ys = _keyed(nest, ("y_mm", "y_m", "y"))
    if xs is not None and ys is not None:
        key = xs[0] if xs[0] != "x" else ("x_mm" if max(abs(xs[1]), abs(ys[1])) > 20 else "x")
        x = _length(xs[1], key)[0]
        y = _length(ys[1], key.replace("x", "y"))[0]
    else:
        pose = _keyed(nest, ("pose_mm", "pose"))
        if pose is None:
            return None
        values = [float(v) for v in list(pose[1])[:3]]
        x, y = _length(values[:2], pose[0] if pose[0].endswith("_mm") else "pose")
    heading = _keyed(nest, ("heading_mrad", "heading_rad", "heading"))
    yaw = _angle(heading[1], heading[0]) if heading is not None else 0.0
    if str(_get(nest, "kind", default="open") or "open") == "dock":
        # A dock (owner decision 2026-09-27): its status light, the strip across the housing's
        # face above the tag panel, has the backlight's meaning (C4).
        from duckmind.world import dock

        style = str(_get(nest, "style", default="") or dock.DEFAULT_STYLE)
        given = _get(nest, "shape_mm", "shape", default=None) or {}
        geometry = dock.shape(
            style, {k: v for k, v in dict(given).items() if k in dock.SHAPE_FIELDS}
        )
        where = types.SimpleNamespace(x_mm=x * 1000.0, y_mm=y * 1000.0, heading_mrad=yaw * 1000.0)
        lx, ly, lz = dock.light_point(where, geometry)
        bx, by, mid = lx / 1000.0, ly / 1000.0, lz / 1000.0
    else:
        # The backboard stands behind the pad (−x in the nest's frame); the glow is on its face.
        offset = _keyed(nest, ("board_offset_mm", "board_offset_m"))
        back = _length(offset[1], offset[0])[0] if offset is not None else 0.41
        board = _keyed(nest, ("board_size_mm", "board_size_m"))
        sx, _, sz = _length(board[1], board[0])[:3] if board is not None else (0.02, 0.3, 0.22)
        face = -back + sx / 2
        bx, by = x + face * math.cos(yaw), y + face * math.sin(yaw)
        mid = sz / 2
    backlight = _get(nest, "backlight", default=True)
    color, charging_color, nits = BACKLIGHT_COLOR, BACKLIGHT_CHARGING_COLOR, BACKLIGHT_NITS
    if isinstance(backlight, Mapping):
        nits = float(_get(backlight, "nits", default=BACKLIGHT_NITS))
        color = _color(_get(backlight, "colour", "color", default=None), BACKLIGHT_COLOR)
        charging_color = _color(
            _get(backlight, "charging_colour", "charging_color", default=None),
            BACKLIGHT_CHARGING_COLOR,
        )
        when = str(_get(backlight, "when", default="dark") or "dark")
        backlight = bool(_get(backlight, "on", "enabled", default=True)) and when != "never"
    floor = next((r.floor for r in rooms if r.contains(bx, by)), 0.0)
    return NestLight((bx, by, floor + mid), bool(backlight), color, charging_color, nits)
