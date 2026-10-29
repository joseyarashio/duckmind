"""The sun through the windows (a:play, 2026-09-28): the patch of direct sunlight on the floor,
as the world's truth, and what the duck's head camera could honestly make of it.

**World truth** (never the brain's): the home's windows (:class:`~duckmind.world.lights.
HomeLights`: centre, width × height, the way each looks out) and the sun's place in the sky at the
duck's local date and time in Taipei (:func:`solar_position`, the same NOAA formulas as
:func:`duckmind.world.lights.sun_times`). A window the sun shines into casts its opening onto the
floor along the sun's rays (:func:`window_patch`), clipped to its room; inside that patch the
floor gets the sun's direct light through the glass (:func:`direct_lux`: the Kasten–Young air
mass, Meinel's clear-sky beam, 93 lm/W, ~75 % through the glass, on a level floor) on top of the
room's own daylight (:mod:`duckmind.world.lights` models daylight through a window as the same
on every level surface: it has no patch). A patch moves across the floor as the sun moves: the
east window of ``default_home`` lights a long strip far into the room at 07:00, a short one near
the wall by 10:00, nothing after about 11:00 (September); the apartment's west window, the
afternoon. Furniture in the way shades it (:class:`SunShade`: a ray from each floor cell toward
the sun that hits something other than a wall — the window's own wall is solid in the physics
model, the opening is not modelled — is in shade; rebuilt every :data:`REFRESH_S` of sim time).
``weather`` "cloudy" (or a patch darker than :data:`MIN_PATCH_LUX`) casts none.

**What the camera could report** (:class:`SunSightModel`, MODEL, like the rest of the prototype's
camera): a *bright floor region* in the picture — a ``light`` sighting (the contract's
VISUAL_KINDS "light") with its bearing, the ground-plane range to its nearest visible part, its
angular size and its measured brightness — only for the part of a patch that is inside the
delivered frame and not hidden (a ray from the lens reaches the floor there), only when it stands
out (≥ :data:`MIN_CONTRAST` × the lux under the lens: a patch in a bright room at noon is
harder to tell), with the look's noise; and the camera's light meter over its full range
(:class:`PlayCameraSense` ``lux``: the auto-exposure reading, 2 Hz, the same meter the light
bands come from — the bands saturate at 30 lux, a sunny floor reads 5,000–30,000). A duck on the
real robot has exactly these: the exposure of its own camera and a bright blob in its picture.
Nothing here gives a position in the room, the time of day or the window.

Pure except :class:`SunShade` and :func:`view_entries` (MuJoCo rays, world side only).
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from duckmind.neuro import contract as C
from duckmind.world import lights as L

VERSION = "neuro-sun-1"
GLASS_T = 0.75  # through ordinary window glass
EFFICACY_LM_W = 93.0  # luminous efficacy of direct sunlight
SOLAR_W_M2 = 1361.0
MIN_ALT_DEG = 2.0  # lower than this the sun is behind the buildings across the street
MIN_PATCH_LUX = 300.0  # a patch dimmer than this is not "sunny"
REFRESH_S = 30.0  # the sun moves ~0.12° in 30 s: the patch and its shade are rebuilt this often
CELL_M = 0.08  # the shade mask's floor grid
VIEW_CELL_M = 0.12  # the camera view's floor samples (a coarser grid: ≤ VIEW_MAX points a look)
VIEW_MAX = 90
WALLISH = ("wall", "skirting")  # geoms a ray to the sun may leave the room through (the window)


# ── the sun ────────────────────────────────────────────────────────────────────


def solar_position(
    when: datetime, lat: float = L.HOME_LAT, lon: float = L.HOME_LON, utc: float = L.HOME_UTC
) -> tuple[float, float]:
    """(altitude, azimuth) of the sun at the local time ``when`` (naive, the duck's clock),
    radians; azimuth from north, clockwise (east = π/2). NOAA's general solar position
    formulas (the same as :func:`duckmind.world.lights.sun_times`), a few tenths of a degree."""
    n = when.timetuple().tm_yday
    hour = when.hour + when.minute / 60.0 + when.second / 3600.0
    g = 2.0 * math.pi / 365.0 * (n - 1 + (hour - 12.0) / 24.0)
    eqt = 229.18 * (
        0.000075
        + 0.001868 * math.cos(g)
        - 0.032077 * math.sin(g)
        - 0.014615 * math.cos(2 * g)
        - 0.040849 * math.sin(2 * g)
    )
    decl = (
        0.006918
        - 0.399912 * math.cos(g)
        + 0.070257 * math.sin(g)
        - 0.006758 * math.cos(2 * g)
        + 0.000907 * math.sin(2 * g)
        - 0.002697 * math.cos(3 * g)
        + 0.00148 * math.sin(3 * g)
    )
    true_solar_min = hour * 60.0 + eqt + 4.0 * lon - 60.0 * utc
    ha = math.radians(true_solar_min / 4.0 - 180.0)
    phi = math.radians(lat)
    cos_zen = math.sin(phi) * math.sin(decl) + math.cos(phi) * math.cos(decl) * math.cos(ha)
    zen = math.acos(max(-1.0, min(1.0, cos_zen)))
    az = (
        math.atan2(math.sin(ha), math.cos(ha) * math.sin(phi) - math.tan(decl) * math.cos(phi))
        + math.pi
    )
    return math.pi / 2.0 - zen, az % (2.0 * math.pi)


def sun_direction(alt: float, az: float, north_rad: float) -> tuple[float, float, float]:
    """Unit vector toward the sun in the world frame (z up); ``north_rad``: the heading of north
    in the world (a layout's ``north_mrad`` / 1000). Azimuth turns clockwise from north."""
    heading = north_rad - az
    c = math.cos(alt)
    return (c * math.cos(heading), c * math.sin(heading), math.sin(alt))


def direct_lux(alt: float, *, clear: float = 1.0, glass: float = GLASS_T) -> float:
    """Direct sunlight on a level floor through a window, lux: the beam from Meinel's clear-sky
    model at the Kasten–Young air mass, × the luminous efficacy, × the glass, × sin(altitude)."""
    if alt <= math.radians(MIN_ALT_DEG):
        return 0.0
    deg = math.degrees(alt)
    air_mass = 1.0 / (math.sin(alt) + 0.50572 * (deg + 6.07995) ** -1.6364)
    beam = SOLAR_W_M2 * 0.7 ** (air_mass**0.678)
    return EFFICACY_LM_W * beam * clear * glass * math.sin(alt)


# ── the patch ─────────────────────────────────────────────────────────────────

Point = tuple[float, float]


def window_patch(window: L.Window, sun: Sequence[float], floor: float = 0.0) -> list[Point] | None:
    """The window's opening cast onto the floor along the sun's rays: four floor points (m), or
    None when the sun does not shine in (behind the window's wall, or too low)."""
    sx, sy, sz = (float(v) for v in sun)
    fx, fy = window.facing
    if sz <= math.sin(math.radians(MIN_ALT_DEG)) or sx * fx + sy * fy <= 1e-3:
        return None
    cx, cy, cz = window.position
    ux, uy = -fy, fx  # along the wall
    half_w, half_h = window.size[0] / 2.0, window.size[1] / 2.0
    out: list[Point] = []
    for du, dz in ((-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)):
        px, py, pz = cx + ux * du, cy + uy * du, cz + dz
        t = (pz - floor) / sz
        if t <= 0.0:
            return None
        out.append((px - t * sx, py - t * sy))
    return out


def polygon_area(poly: Sequence[Point]) -> float:
    area = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def point_in_polygon(x: float, y: float, poly: Sequence[Point]) -> bool:
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def clip_polygon(subject: Sequence[Point], clip: Sequence[Point]) -> list[Point]:
    """Sutherland–Hodgman: ``subject`` clipped to the convex polygon ``clip``."""
    if not subject or not clip:
        return []
    ccw = polygon_area(clip) > 0.0
    out = list(subject)
    n = len(clip)
    for i in range(n):
        ax, ay = clip[i]
        bx, by = clip[(i + 1) % n]

        def inside(
            p: Point, ax: float = ax, ay: float = ay, bx: float = bx, by: float = by
        ) -> bool:
            cross = (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax)
            return cross >= 0.0 if ccw else cross <= 0.0

        def cut(
            p: Point, q: Point, ax: float = ax, ay: float = ay, bx: float = bx, by: float = by
        ) -> Point:
            dx1, dy1 = q[0] - p[0], q[1] - p[1]
            dx2, dy2 = bx - ax, by - ay
            den = dx1 * dy2 - dy1 * dx2
            if abs(den) < 1e-12:
                return q
            t = ((ax - p[0]) * dy2 - (ay - p[1]) * dx2) / den
            return (p[0] + t * dx1, p[1] + t * dy1)

        points, out = out, []
        if not points:
            break
        prev = points[-1]
        for cur in points:
            if inside(cur):
                if not inside(prev):
                    out.append(cut(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(cut(prev, cur))
            prev = cur
    return out


@dataclass(frozen=True)
class Patch:
    """One window's sunlight on the floor (world truth)."""

    window: str
    room: str
    polygon: tuple[Point, ...]
    lux: float  # direct sunlight on the floor inside it (on top of the room's light)
    alt: float  # the sun, rad
    az: float

    @property
    def area(self) -> float:
        return abs(polygon_area(self.polygon))

    @property
    def centre(self) -> Point:
        n = len(self.polygon)
        return (sum(p[0] for p in self.polygon) / n, sum(p[1] for p in self.polygon) / n)

    def contains(self, x: float, y: float) -> bool:
        return point_in_polygon(x, y, self.polygon)

    def view(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "room": self.room,
            "polygon_mm": [[round(x * 1000), round(y * 1000)] for x, y in self.polygon],
            "lux": round(self.lux),
            "alt_deg": round(math.degrees(self.alt), 1),
            "az_deg": round(math.degrees(self.az), 1),
            "area_m2": round(self.area, 3),
        }


class SunModel:
    """The sun of one home: which windows it shines through at a time and the patches it casts.
    ``weather``: "sunny" (a clear sky) or "cloudy" (no direct sun: no patch)."""

    def __init__(
        self, home: L.HomeLights, north_rad: float = math.pi / 2, *, weather: str = "sunny"
    ) -> None:
        if weather not in ("sunny", "cloudy"):
            raise ValueError("weather: sunny or cloudy")
        self.home = home
        self.north = float(north_rad)
        self.weather = weather

    @classmethod
    def from_layout(cls, layout: Any, *, weather: str = "sunny") -> SunModel:
        north = float(getattr(layout, "north_mrad", 1571)) / 1000.0
        return cls(L.HomeLights.from_layout(layout), north, weather=weather)

    def patches(self, when: datetime) -> tuple[Patch, ...]:
        if self.weather != "sunny":
            return ()
        alt, az = solar_position(when)
        lux = direct_lux(alt)
        if lux < MIN_PATCH_LUX:
            return ()
        sun = sun_direction(alt, az, self.north)
        out: list[Patch] = []
        for room in self.home.rooms:
            for window in room.windows:
                cast = window_patch(window, sun, room.floor)
                if cast is None:
                    continue
                clipped = clip_polygon(cast, room.polygon)
                if len(clipped) < 3 or abs(polygon_area(clipped)) < 0.01:
                    continue
                out.append(Patch(window.id, room.id, tuple(clipped), lux, alt, az))
        return tuple(out)


# ── shade (MuJoCo rays; world side) ───────────────────────────────────────────


class SunShade:
    """Which floor cells of each patch the sun really reaches: a ray from the cell toward the
    sun that meets furniture, a thing or a person first is in shade; one that meets a wall (the
    window's own wall: the opening is not in the physics model) or nothing is lit. Static at a
    refresh (people and toys move: a ray at the query point too, for the camera's meter)."""

    def __init__(self) -> None:
        self.model: Any = None
        self.skip_bodies: set[int] = set()
        self.wallish: set[int] = set()
        self._hit: Any = None

    def ready(self, model: Any, skip_bodies: Sequence[int] = ()) -> None:
        import mujoco
        import numpy as np

        self.model = model
        self.skip_bodies = {int(b) for b in skip_bodies}
        self.wallish = set()
        for g in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if any(w in name for w in WALLISH):
                self.wallish.add(g)
        self._hit = np.zeros(1, dtype=np.int32)

    def lit(self, data: Any, x: float, y: float, sun: Sequence[float], z: float = 0.004) -> bool:
        """Does the sun reach the floor at (x, y)?"""
        import mujoco
        import numpy as np

        if self.model is None:
            return True
        start = np.array([x, y, z], dtype=float)
        vec = np.array(sun, dtype=float)
        for _ in range(6):
            self._hit[0] = -1
            got = float(mujoco.mj_ray(self.model, data, start, vec, None, 1, -1, self._hit))
            hit = int(self._hit[0])
            if got < 0.0 or hit < 0:
                return True
            if hit in self.wallish:
                return True
            body = int(self.model.geom_bodyid[hit])
            if body in self.skip_bodies:
                start = start + vec * (got + 0.005)
                continue
            if (
                int(self.model.geom_contype[hit]) == 0
                and int(self.model.geom_conaffinity[hit]) == 0
            ):
                # a drawn-only geom (a rug, leaves, a lamp shade) casts no shadow here
                start = start + vec * (got + 0.005)
                continue
            return False
        return True


@dataclass
class _Mask:
    patch: Patch
    cells: dict[tuple[int, int], bool]  # floor cell (CELL_M grid) → lit


class SunField:
    """The patches of a home at the world's time, with their shade masks, refreshed every
    :data:`REFRESH_S` of sim time (or when the clock jumps): :meth:`extra_lux` is the direct sun
    at a floor point on top of the room's light."""

    def __init__(self, model_: SunModel, shade: SunShade | None = None) -> None:
        self.sun = model_
        self.shade = shade or SunShade()
        self.masks: list[_Mask] = []
        self.when: datetime | None = None
        self.refreshed_s: float | None = None
        self.sun_dir: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def refresh(self, data: Any, when: datetime, sim_s: float, *, force: bool = False) -> None:
        jump = self.when is None or abs((when - self.when).total_seconds()) > 2.0 * REFRESH_S
        due = self.refreshed_s is None or sim_s - self.refreshed_s >= REFRESH_S
        if not (force or jump or due):
            return
        self.when, self.refreshed_s = when, sim_s
        alt, az = solar_position(when)
        self.sun_dir = sun_direction(alt, az, self.sun.north)
        self.masks = []
        for patch in self.sun.patches(when):
            cells: dict[tuple[int, int], bool] = {}
            xs = [p[0] for p in patch.polygon]
            ys = [p[1] for p in patch.polygon]
            i0, i1 = math.floor(min(xs) / CELL_M), math.floor(max(xs) / CELL_M)
            j0, j1 = math.floor(min(ys) / CELL_M), math.floor(max(ys) / CELL_M)
            for i in range(i0, i1 + 1):
                for j in range(j0, j1 + 1):
                    cx, cy = (i + 0.5) * CELL_M, (j + 0.5) * CELL_M
                    if patch.contains(cx, cy):
                        cells[(i, j)] = self.shade.lit(data, cx, cy, self.sun_dir)
            self.masks.append(_Mask(patch, cells))

    def extra_lux(self, x: float, y: float) -> float:
        """Direct sun on the floor at (x, y) now (0 outside every patch or in shade)."""
        key = (math.floor(x / CELL_M), math.floor(y / CELL_M))
        best = 0.0
        for mask in self.masks:
            if not mask.patch.contains(x, y):
                continue
            if mask.cells.get(key, True):
                best = max(best, mask.patch.lux)
        return best

    def lit_share(self, points: Sequence[Point]) -> float:
        """The share of these floor points in the sun (evaluators: the duck's footprint)."""
        if not points:
            return 0.0
        return sum(1 for x, y in points if self.extra_lux(x, y) > 0.0) / len(points)

    def view(self) -> dict[str, Any]:
        out = []
        for mask in self.masks:
            lit = sum(1 for v in mask.cells.values() if v)
            item = mask.patch.view()
            item["lit_share"] = round(lit / max(1, len(mask.cells)), 3)
            out.append(item)
        return {
            "version": VERSION,
            "weather": self.sun.weather,
            "when": None if self.when is None else self.when.isoformat(timespec="seconds"),
            "sun_dir": [round(v, 4) for v in self.sun_dir],
            "patches": out,
        }


def view_entries(
    sunfield: SunField,
    data: Any,
    cam: Sequence[float],
    axes: Sequence[Sequence[float]],
    yaw: float,
    in_frame: Callable[[float, float, float], bool],
    clear: Callable[[Sequence[float]], bool],
    room_lux: Callable[[Sequence[float]], float],
) -> list[dict[str, Any]]:
    """The camera's view of the sunny floor (world truth for :class:`SunSightModel`): per patch,
    the floor samples (a :data:`VIEW_CELL_M` grid, at most :data:`VIEW_MAX`) inside the delivered
    frame that a ray from the lens reaches (``clear``) and the sun reaches (the shade mask); their
    centroid's bearing and elevation from the lens in the duck's heading frame (+ = left / up),
    the level range from the lens to the nearest of them, their angular spread, and the lux
    there (room + sun) with the room's own lux beside it."""
    right, down, forward = (tuple(float(v) for v in a) for a in axes)
    out: list[dict[str, Any]] = []
    c, s = math.cos(yaw), math.sin(yaw)
    for index, mask in enumerate(sunfield.masks):
        patch = mask.patch
        xs = [p[0] for p in patch.polygon]
        ys = [p[1] for p in patch.polygon]
        pts: list[tuple[float, float]] = []
        x = math.floor(min(xs) / VIEW_CELL_M) * VIEW_CELL_M + VIEW_CELL_M / 2
        while x <= max(xs):
            y = math.floor(min(ys) / VIEW_CELL_M) * VIEW_CELL_M + VIEW_CELL_M / 2
            while y <= max(ys):
                if patch.contains(x, y) and sunfield.extra_lux(x, y) > 0.0:
                    pts.append((x, y))
                y += VIEW_CELL_M
            x += VIEW_CELL_M
        if not pts:
            continue
        inside: list[tuple[float, float]] = []
        for px, py in pts:
            rel = (px - cam[0], py - cam[1], 0.002 - cam[2])
            r = rel[0] * right[0] + rel[1] * right[1] + rel[2] * right[2]
            d = rel[0] * down[0] + rel[1] * down[1] + rel[2] * down[2]
            f = rel[0] * forward[0] + rel[1] * forward[1] + rel[2] * forward[2]
            if in_frame(r, d, f):
                inside.append((px, py))
        if not inside:
            continue
        if len(inside) > VIEW_MAX:
            step = len(inside) / VIEW_MAX
            inside = [inside[int(k * step)] for k in range(VIEW_MAX)]
        visible = [p for p in inside if clear((p[0], p[1], 0.002))]
        if not visible:
            continue
        mx = sum(p[0] for p in visible) / len(visible)
        my = sum(p[1] for p in visible) / len(visible)
        dx, dy, dz = mx - cam[0], my - cam[1], 0.002 - cam[2]
        hx, hy = c * dx + s * dy, -s * dx + c * dy
        bearings = []
        near = math.inf
        for px, py in visible:
            ex, ey = px - cam[0], py - cam[1]
            bearings.append(math.atan2(-s * ex + c * ey, c * ex + s * ey))
            near = min(near, math.hypot(ex, ey))
        spread = max(bearings) - min(bearings) if len(bearings) > 1 else 0.0
        lux_room = room_lux((mx, my, 0.0))
        out.append(
            {
                "id": f"sun:{patch.window}:{index}",
                "b": round(math.atan2(hy, hx), 4),
                "e": round(math.atan2(dz, math.hypot(dx, dy)), 4),
                "rng": round(near, 3),
                "rng_c": round(math.hypot(dx, dy), 3),
                "size": round(max(spread, 2.0 * math.atan(VIEW_CELL_M / max(near, 0.1))), 4),
                "lux": round(lux_room + patch.lux, 1),
                "lux_room": round(lux_room, 1),
                "vis": len(visible),
                "fov": len(inside),
                "n": len(pts),
            }
        )
    return out


# ── the sensor model (what a camera could report) ─────────────────────────────

MIN_CONTRAST = 3.0  # a region this many times brighter than the floor under the lens stands out
P_SEE_STILL = 0.9  # a big bright blob, the camera still
P_SEE_STEPPING = 0.55  # … while the legs step (motion blur hides edges, not a sunlit floor)
BEARING_SIGMA = 0.03
RANGE_SIGMA = 0.10
LUX_SIGMA_DEX = 0.06  # the meter's scatter (neuro_senses.METER_NOISE_DEX)
MIN_SIZE_RAD = 0.04


# Backwards-compatible names for callers from before these readings moved into contract.py.
LightSighting = C.Sighting
PlayCameraSense = C.CameraSense


@dataclass
class SunSightModel:
    """Bright floor regions in the picture → ``light`` sightings (see the module docstring)."""

    seed: int = 7
    tracks: dict[str, int] = field(default_factory=dict)
    next_tid: int = 9000

    def step(
        self,
        t_ms: int,
        entries: Sequence[Mapping[str, Any]],
        *,
        lux_under: float,
        stepping: bool,
        saccade: bool,
    ) -> list[LightSighting]:
        out: list[LightSighting] = []
        if saccade:
            return out
        for e in entries:
            if int(e.get("vis", 0)) <= 0:
                continue
            lux = float(e.get("lux", 0.0))
            if lux < MIN_CONTRAST * max(lux_under, 1.0):
                continue  # it does not stand out from where the duck is
            size = float(e.get("size", 0.0))
            if size < MIN_SIZE_RAD:
                continue
            key = str(e.get("id"))
            rng = random.Random(f"sun:{self.seed}:{key}:{t_ms}")
            if rng.random() >= (P_SEE_STEPPING if stepping else P_SEE_STILL):
                continue
            tid = self.tracks.get(key)
            if tid is None:
                tid = self.tracks[key] = self.next_tid
                self.next_tid += 1
            near = float(e.get("rng", 1.0))
            out.append(
                LightSighting(
                    kind="light",
                    bearing=C.wrap_angle(float(e.get("b", 0.0)) + rng.gauss(0.0, BEARING_SIGMA)),
                    elevation=float(e.get("e", 0.0)) + rng.gauss(0.0, BEARING_SIGMA),
                    range_m=round(max(0.05, near * (1.0 + rng.gauss(0.0, RANGE_SIGMA))), 3),
                    size=max(0.01, size * (1.0 + rng.gauss(0.0, 0.1))),
                    motion=0.0,
                    looming=0.0,
                    track=tid,
                    source="model",
                    lux=round(
                        10.0 ** (math.log10(max(lux, 1e-3)) + rng.gauss(0.0, LUX_SIGMA_DEX)), 1
                    ),
                )
            )
        return out


METER_ROWS_RAD = (
    0.0,
    math.radians(15.5),
    math.radians(31.0),
)  # the picture's centre, 1/4 down, bottom
METER_FAR_M = 1.5


def meter_points(cam: Sequence[float], forward: Sequence[float]) -> list[tuple[float, float]]:
    """Where the camera's light meter reads the floor: the lower half of the picture (where the
    meter's weight is) — the floor where its centre row, the row a quarter down and its bottom row
    meet the floor (≤ 1.5 m) along the lens's level heading. At HOME (the axis 8–15° down) that is
    ~0.22, 0.4 and 1.2 m ahead; with the head pitched right down, ~0.1–0.35 m: the floor right in
    front of its chest."""
    fx, fy, fz = float(forward[0]), float(forward[1]), float(forward[2])
    flat = math.hypot(fx, fy) or 1.0
    ux, uy = fx / flat, fy / flat
    down = math.atan2(-fz, flat)  # the axis's depression below level
    out: list[tuple[float, float]] = []
    for row in METER_ROWS_RAD:
        dep = down + row
        d = (
            METER_FAR_M
            if dep <= math.radians(2.0)
            else min(METER_FAR_M, float(cam[2]) / math.tan(dep))
        )
        out.append((float(cam[0]) + ux * d, float(cam[1]) + uy * d))
    return out


def log_mean(values: Sequence[float]) -> float:
    """The exposure's average of several readings (in log: a camera's meter)."""
    if not values:
        return 0.0
    return 10.0 ** (sum(math.log10(max(v, 1e-3)) for v in values) / len(values))


__all__ = [
    "LightSighting",
    "Patch",
    "PlayCameraSense",
    "SunField",
    "SunModel",
    "SunShade",
    "SunSightModel",
    "clip_polygon",
    "direct_lux",
    "log_mean",
    "meter_points",
    "point_in_polygon",
    "solar_position",
    "sun_direction",
    "view_entries",
    "window_patch",
]
