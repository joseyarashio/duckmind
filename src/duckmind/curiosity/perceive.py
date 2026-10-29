"""perceive: the Act Runner's thin adapter over duck-eye (curiosity spec §6.1–6.2; physical-design
spec §15.1: ``perceive.py`` becomes a call into duck-eye's ``things`` task).

The Act Runner never sees the world's truth. What it knows about a thing comes from looks:

1. :func:`sightings` reads one ``eye.result`` (a dict as it came off ``eye.sock``, or an
   :class:`~duckmind.eye.protocol.EyeResult`) and returns what the ``things`` task saw, in the
   trunk frame at the capture (x ahead, y left, millimetres on the floor);
2. :class:`Tracker` keeps the target in the **odometry frame** (the pose tap's ``odom`` when the
   look was taken — the duck stands still while it looks), so the target stays put while the
   duck walks, and says where it is now in the current trunk frame (:meth:`Tracker.relative`);
3. two looks *agree* when they are at least :data:`AGREE_GAP_MS` apart and the target moved at
   most :data:`AGREE_MM` between them (§5.2 step 4): only then may the runner act.

A memory ages out (§6.2): a thing that stood still is remembered :data:`KEEP_STILL_MS`, one that
moved :data:`KEEP_MOVING_MS`; after that the runner looks for it again.

The ``things`` task's entries are read tolerantly (the task's owner, duck-eye, defines them):
identity from ``id`` (a registered toy) or ``colour`` (``label``/``cls`` too); position from
``x_mm``/``y_mm`` (a centre) or ``r_mm``/``b_mrad`` — duck-eye's ``things`` gives the range to the
thing's *foot* on the floor (its near edge, :mod:`duckmind.eye.floorline`), so such a sighting is
marked ``foot`` and :func:`centre_of` moves it half the thing's registered depth outward; size from
``w_mm``/``h_mm``/``d_mm`` (or ``size_mm``); ``conf_ppm``.

Pure: standard library only (no world, no clocks).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

AGREE_MM = 12
AGREE_GAP_MS = 200
KEEP_STILL_MS = 10_000
KEEP_MOVING_MS = 3_000
MOVED_MM = 300  # a sighting this far from the memory is "moved" (sense.thing edge)
ID_KEYS = ("id", "thing", "label", "colour", "color", "cls", "class")


@dataclass(frozen=True)
class Seen:
    """One thing in one look. ``x_mm``/``y_mm``: trunk frame at the capture, on the floor
    (the thing's centre, or its near-face middle when ``near`` says so)."""

    look_id: str
    t_ms: int
    key: str
    x_mm: int
    y_mm: int
    conf_ppm: int = 1_000_000
    w_mm: int | None = None
    h_mm: int | None = None
    d_mm: int | None = None
    labels: tuple[str, ...] = ()
    foot: bool = False  # the range is to its near edge on the floor, not its centre

    @property
    def r_mm(self) -> int:
        return int(round(math.hypot(self.x_mm, self.y_mm)))

    @property
    def b_mrad(self) -> int:
        return int(round(math.atan2(self.y_mm, self.x_mm) * 1000.0))


def _int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return int(round(value))


def _entry(entry: Mapping[str, Any], look_id: str, t_ms: int) -> Seen | None:
    labels = tuple(str(entry[k]) for k in ID_KEYS if isinstance(entry.get(k), str) and entry.get(k))
    if not labels:
        return None
    x, y = _int(entry.get("x_mm")), _int(entry.get("y_mm"))
    foot = False
    if x is None or y is None:
        r, b = _int(entry.get("r_mm")), _int(entry.get("b_mrad"))
        if r is None or b is None:
            return None
        x = int(round(r * math.cos(b / 1000.0)))
        y = int(round(r * math.sin(b / 1000.0)))
        foot = True
    size = entry.get("size_mm")
    w, h, d = _int(entry.get("w_mm")), _int(entry.get("h_mm")), _int(entry.get("d_mm"))
    if isinstance(size, Sequence) and not isinstance(size, str) and len(size) >= 2:
        w = w if w is not None else _int(size[0])
        h = h if h is not None else _int(size[-1])
    conf = _int(entry.get("conf_ppm"))
    return Seen(
        look_id=look_id,
        t_ms=t_ms,
        key=labels[0],
        x_mm=x,
        y_mm=y,
        conf_ppm=1_000_000 if conf is None else max(0, min(1_000_000, conf)),
        w_mm=w,
        h_mm=h,
        d_mm=d,
        labels=labels,
        foot=foot,
    )


def _as_dict(result: Any) -> Mapping[str, Any]:
    if isinstance(result, Mapping):
        return result
    to_json = getattr(result, "to_json", None)
    if callable(to_json):
        value = to_json()
        if isinstance(value, Mapping):
            return value
    raise TypeError("an eye.result dict or EyeResult")


def result_ok(result: Any) -> bool:
    """The look produced pose-paired results (``ok``); anything else is no sight."""
    return _as_dict(result).get("status") == "ok"


def result_light(result: Any) -> str | None:
    light = (_as_dict(result).get("results") or {}).get("light") or {}
    level = light.get("level") if isinstance(light, Mapping) else None
    return level if isinstance(level, str) else None


def sightings(result: Any, *, t_ms: int | None = None) -> list[Seen]:
    """What the ``things`` task saw in one look (empty unless the look is ``ok``)."""
    message = _as_dict(result)
    if message.get("status") != "ok":
        return []
    look_id = str(message.get("look_id", "?"))
    when = t_ms
    if when is None:
        captured = message.get("captured_at_ns")
        when = int(captured) // 1_000_000 if isinstance(captured, int) else 0
    things = (message.get("results") or {}).get("things") or {}
    entries: Iterable[Any]
    if isinstance(things, Mapping):
        entries = things.get("things") or things.get("items") or things.get("seen") or []
    elif isinstance(things, list):
        entries = things
    else:
        entries = []
    out = []
    for entry in entries:
        if isinstance(entry, Mapping):
            seen = _entry(entry, look_id, when)
            if seen is not None:
                out.append(seen)
    return out


def centre_of(seen: Seen, depth_mm: int | None) -> Seen:
    """A foot sighting moved half the thing's depth outward along its bearing (a centre
    sighting, or an unknown depth, is returned as it is)."""
    half = (depth_mm if depth_mm is not None else seen.d_mm or 0) / 2.0
    if not seen.foot or half <= 0:
        return seen
    b = math.atan2(seen.y_mm, seen.x_mm)
    return Seen(
        look_id=seen.look_id,
        t_ms=seen.t_ms,
        key=seen.key,
        x_mm=int(round(seen.x_mm + half * math.cos(b))),
        y_mm=int(round(seen.y_mm + half * math.sin(b))),
        conf_ppm=seen.conf_ppm,
        w_mm=seen.w_mm,
        h_mm=seen.h_mm,
        d_mm=seen.d_mm,
        labels=seen.labels,
        foot=False,
    )


# ── the odometry frame ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Odom:
    """The duck in its odometry frame: millimetres and milliradians (the pose tap's odom)."""

    x_mm: int
    y_mm: int
    yaw_mrad: int

    @classmethod
    def from_pose(cls, odom: Sequence[float]) -> Odom:
        """From the pose tap's ``odom`` (x m, y m, yaw rad)."""
        return cls(
            int(round(odom[0] * 1000.0)), int(round(odom[1] * 1000.0)), int(round(odom[2] * 1000.0))
        )

    def to_world(self, x_mm: float, y_mm: float) -> tuple[int, int]:
        """A trunk-frame point (at this pose) in the odometry frame."""
        c, s = math.cos(self.yaw_mrad / 1000.0), math.sin(self.yaw_mrad / 1000.0)
        return (
            int(round(self.x_mm + x_mm * c - y_mm * s)),
            int(round(self.y_mm + x_mm * s + y_mm * c)),
        )

    def to_trunk(self, x_mm: float, y_mm: float) -> tuple[int, int]:
        """An odometry-frame point in the trunk frame at this pose."""
        c, s = math.cos(self.yaw_mrad / 1000.0), math.sin(self.yaw_mrad / 1000.0)
        dx, dy = x_mm - self.x_mm, y_mm - self.y_mm
        return int(round(dx * c + dy * s)), int(round(-dx * s + dy * c))

    def delta(self, before: Odom) -> tuple[int, int, int]:
        """This pose relative to ``before``, in ``before``'s trunk frame (dx, dy, dyaw)."""
        dx, dy = before.to_trunk(self.x_mm, self.y_mm)
        dyaw = (self.yaw_mrad - before.yaw_mrad + 3142) % 6283 - 3142
        return dx, dy, dyaw


@dataclass(frozen=True)
class Fix:
    """The target in the odometry frame, from one look."""

    x_mm: int
    y_mm: int
    t_ms: int
    look_id: str
    key: str
    w_mm: int | None = None
    h_mm: int | None = None
    d_mm: int | None = None


def fix_of(seen: Seen, at: Odom) -> Fix:
    x, y = at.to_world(seen.x_mm, seen.y_mm)
    return Fix(x, y, seen.t_ms, seen.look_id, seen.key, seen.w_mm, seen.h_mm, seen.d_mm)


def pick(
    seen: Sequence[Seen],
    key: str | None,
    near: tuple[int, int] | None = None,
    *,
    gate_mm: int = 400,
) -> Seen | None:
    """The sighting that is the target: its label (any of the entry's labels) and, when a
    believed trunk-frame position is known, the nearest one within ``gate_mm`` of it."""
    candidates = [s for s in seen if key is None or key in (s.labels or (s.key,))]
    if near is not None:
        candidates = [
            s for s in candidates if math.hypot(s.x_mm - near[0], s.y_mm - near[1]) <= gate_mm
        ]
        candidates.sort(key=lambda s: math.hypot(s.x_mm - near[0], s.y_mm - near[1]))
    else:
        candidates.sort(key=lambda s: (-s.conf_ppm, s.r_mm))
    return candidates[0] if candidates else None


class Tracker:
    """One target, remembered in the odometry frame (§6.2's small tracker)."""

    def __init__(self, key: str | None) -> None:
        self.key = key
        self.fixes: list[Fix] = []
        self.moving = False
        self.lost_at: int | None = None

    @property
    def last(self) -> Fix | None:
        return self.fixes[-1] if self.fixes else None

    def update(self, fix: Fix) -> None:
        last = self.last
        if last is not None and math.hypot(fix.x_mm - last.x_mm, fix.y_mm - last.y_mm) > 30:
            self.moving = True
        elif last is not None:
            self.moving = False
        if self.key is None:
            self.key = fix.key
        self.fixes.append(fix)
        del self.fixes[:-8]
        self.lost_at = None

    def fresh(self, now_ms: int) -> bool:
        last = self.last
        if last is None:
            return False
        keep = KEEP_MOVING_MS if self.moving else KEEP_STILL_MS
        return now_ms - last.t_ms <= keep

    def relative(self, at: Odom) -> tuple[int, int] | None:
        """Where the target is now in the trunk frame (from the last fix and odometry)."""
        last = self.last
        return None if last is None else at.to_trunk(last.x_mm, last.y_mm)

    def agreed(self, *, agree_mm: int = AGREE_MM, gap_ms: int = AGREE_GAP_MS) -> bool:
        """The last two fixes agree (§5.2 step 4): ≥ gap apart, ≤ agree_mm apart."""
        if len(self.fixes) < 2:
            return False
        a, b = self.fixes[-2], self.fixes[-1]
        return (
            b.t_ms - a.t_ms >= gap_ms and math.hypot(b.x_mm - a.x_mm, b.y_mm - a.y_mm) <= agree_mm
        )


__all__ = [
    "AGREE_GAP_MS",
    "AGREE_MM",
    "Fix",
    "KEEP_MOVING_MS",
    "KEEP_STILL_MS",
    "MOVED_MM",
    "Odom",
    "Seen",
    "Tracker",
    "centre_of",
    "fix_of",
    "pick",
    "result_light",
    "result_ok",
    "sightings",
]
