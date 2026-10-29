"""Folding what duck-body senses into percept-v1 events (spec §4.3 step 10, §7.2, §7.7).

:class:`Aggregator` stamps every event (``src``, ``t_ms`` = when it was *sensed*, ``pv``),
validates it against :mod:`duckmind.percept` and applies the folding rules that keep the event
budget (spec §7.7):

* ``sense.light`` — on a level change (duck-eye already applies the 2-look hysteresis) and every
  10 minutes;
* ``sense.sighting`` — edges only: ``new``, ``moved`` (≥ 300 mm), ``lost``; at most one per
  thing per 5 s;
* ``sense.depth`` — only while walking, when the ahead range crosses 800 / 500 / 300 / 150 mm,
  the drop test starts to hold, or touch starts;
* ``sense.contact`` — edges, and not during a walk (those fold into ``walk.done.reason``);
* ``body.battery`` — each 5 % step; ``body.hot`` — each thermal tier change;
* ``sense.lifted``/``sense.set_down``, ``body.fell``/``body.righting``, ``body.posture``,
  ``body.dock``, ``walk.done``, ``sense.scan`` — as they happen (:meth:`Aggregator.emit`).

:meth:`Aggregator.counts` is the budget counter per kind.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from duckmind.duckbody.protocol import DepthView
from duckmind.percept import make

LIGHT_PERIOD_MS = 600_000
SIGHTING_GAP_MS = 5000
MOVED_MM = 300
DEPTH_LINES_MM = (800, 500, 300, 150)
BATTERY_STEP = 5
HOT_TIERS_C = (70, 78, 85)  # spec §4.6: < 70 → 0, 70–78 → 1, 78–85 → 2, ≥ 85 → 3


def hot_tier(soc_c: int, throttled: bool = False) -> int:
    if throttled or soc_c >= HOT_TIERS_C[2]:
        return 3
    if soc_c >= HOT_TIERS_C[1]:
        return 2
    if soc_c >= HOT_TIERS_C[0]:
        return 1
    return 0


def depth_band(ahead_mm: int | None) -> int:
    """How many of the depth lines the ahead range is at or inside (0 = beyond 800 mm)."""
    if ahead_mm is None:
        return 0
    return sum(1 for line in DEPTH_LINES_MM if ahead_mm <= line)


@dataclass
class _Seen:
    t_ms: int
    b_mrad: int
    r_mm: int | None
    sent_ms: int


@dataclass
class Aggregator:
    src: str
    pv: str
    counter: Counter[str] = field(default_factory=Counter)

    def __post_init__(self) -> None:
        self.light_level: str | None = None
        self.light_sent_ms: int | None = None
        self.things: dict[tuple[str, str], _Seen] = {}
        self.depth_state: tuple[int, bool, bool] = (0, False, False)
        self.battery_step: int | None = None
        self.tier: int | None = None

    # ── every event goes through here ──

    def emit(self, kind: str, t_ms: int, **fields: Any) -> dict[str, Any]:
        event = make(kind, src=self.src, t_ms=int(t_ms), pv=self.pv, **fields)
        self.counter[kind] += 1
        return event

    def emit_body(self, body: dict[str, Any], t_ms: int) -> dict[str, Any]:
        """An event body with its ``kind`` (from righting, executors …) stamped. A ``t_ms`` in
        the body is when it was sensed and wins over ``t_ms``; ``src``/``pv`` are always ours."""
        fields = {k: v for k, v in body.items() if k not in ("kind", "src", "t_ms", "pv")}
        sensed = body.get("t_ms")
        when = sensed if isinstance(sensed, int) and not isinstance(sensed, bool) else t_ms
        return self.emit(str(body["kind"]), when, **fields)

    def counts(self) -> dict[str, int]:
        return dict(self.counter)

    # ── folding rules ──

    def light(self, t_ms: int, level: str, luma: int, noise_q: int) -> dict[str, Any] | None:
        changed = level != self.light_level
        periodic = self.light_sent_ms is not None and t_ms - self.light_sent_ms >= LIGHT_PERIOD_MS
        if not changed and not periodic:
            return None
        self.light_level, self.light_sent_ms = level, t_ms
        return self.emit(
            "sense.light",
            t_ms,
            level=level,
            luma=max(0, min(255, int(luma))),
            noise_q=max(0, int(noise_q)),
            why="change" if changed else "periodic",
        )

    def sighting(
        self,
        t_ms: int,
        *,
        what: str,
        thing_id: str | None,
        b_mrad: int,
        r_mm: int | None,
        conf_ppm: int,
        look_id: str,
        light: str,
        r_src: str | None = None,
        e_mrad: int | None = None,
        yaw_mrad: int | None = None,
        act_id: str | None = None,
    ) -> dict[str, Any] | None:
        """A thing seen in a look: an event only when it is new or moved ≥ 300 mm (and not
        within 5 s of the last event for it)."""
        key = (what, thing_id or what)
        seen = self.things.get(key)
        edge = None
        if seen is None:
            edge = "new"
        elif (
            r_mm is not None
            and seen.r_mm is not None
            and _moved(seen.b_mrad, seen.r_mm, b_mrad, r_mm) >= MOVED_MM
        ):
            edge = "moved"
        if edge is not None and seen is not None and t_ms - seen.sent_ms < SIGHTING_GAP_MS:
            edge = None
        sent_ms = t_ms if edge is not None else (seen.sent_ms if seen else t_ms)
        if edge is not None or seen is None:
            self.things[key] = _Seen(t_ms, b_mrad, r_mm, sent_ms)
        else:
            seen.t_ms = t_ms
        if edge is None:
            return None
        fields: dict[str, Any] = {
            "what": what,
            "b_mrad": int(b_mrad),
            "conf_ppm": int(conf_ppm),
            "look_id": look_id,
            "edge": edge,
            "light": light,
            "r_mm": None if r_mm is None else int(r_mm),
        }
        for name, value in (
            ("id", thing_id),
            ("r_src", r_src),
            ("e_mrad", e_mrad),
            ("yaw_mrad", yaw_mrad),
            ("act_id", act_id),
        ):
            if value is not None:
                fields[name] = value
        return self.emit("sense.sighting", t_ms, **fields)

    def lost(
        self, t_ms: int, *, what: str, thing_id: str | None, look_id: str, light: str
    ) -> dict[str, Any] | None:
        key = (what, thing_id or what)
        seen = self.things.pop(key, None)
        if seen is None:
            return None
        fields: dict[str, Any] = {
            "what": what,
            "b_mrad": seen.b_mrad,
            "conf_ppm": 0,
            "look_id": look_id,
            "edge": "lost",
            "light": light,
            "r_mm": seen.r_mm,
        }
        if thing_id is not None:
            fields["id"] = thing_id
        return self.emit("sense.sighting", t_ms, **fields)

    def depth(self, t_ms: int, view: DepthView, *, walking: bool) -> dict[str, Any] | None:
        """``sense.depth`` while walking, on a line crossing, a new drop or a new touch."""
        state = (depth_band(view.ahead_mm), view.drop_zones >= 4, view.touch)
        before = self.depth_state
        self.depth_state = state
        if not walking:
            return None
        crossed = state[0] > before[0]
        new_drop = state[1] and not before[1]
        new_touch = state[2] and not before[2]
        if not (crossed or new_drop or new_touch):
            return None
        return self.emit(
            "sense.depth",
            t_ms,
            ahead_mm=view.ahead_mm,
            left_mm=view.left_mm,
            right_mm=view.right_mm,
            drop_zones=int(view.drop_zones),
            touch=bool(view.touch),
            head_pitch_mrad=int(view.head_pitch_mrad),
        )

    def contact(
        self, t_ms: int, *, kind: str, side: str, ev: list[str], walking: bool
    ) -> dict[str, Any] | None:
        if walking:
            return None  # folded into walk.done.reason / contact
        return self.emit("sense.contact", t_ms, contact=kind, side=side, ev=list(ev)[:8])

    def battery(self, t_ms: int, pct: int, trend: str) -> dict[str, Any] | None:
        step = max(0, min(100, int(pct))) // BATTERY_STEP
        if step == self.battery_step:
            return None
        self.battery_step = step
        return self.emit("body.battery", t_ms, pct=max(0, min(100, int(pct))), trend=trend)

    def hot(
        self, t_ms: int, soc_c: int, ceiling_mhz: int | None, throttled: bool = False
    ) -> dict[str, Any] | None:
        tier = hot_tier(soc_c, throttled)
        if tier == self.tier:
            return None
        first = self.tier is None
        self.tier = tier
        if first and tier == 0:
            return None  # a cool start is not news
        return self.emit("body.hot", t_ms, soc_c=int(soc_c), tier=tier, ceiling_mhz=ceiling_mhz)


def _moved(b0: int, r0: int, b1: int, r1: int) -> int:
    """Distance between two (bearing mrad, range mm) points, mm (law of cosines). duck-body
    does not import Duck Core (its package loads the engine; tests/test_imports.py)."""
    import math

    d2 = r0 * r0 + r1 * r1 - 2.0 * r0 * r1 * math.cos((b1 - b0) / 1000.0)
    return int(round(math.sqrt(max(0.0, d2))))


__all__ = ["Aggregator", "DEPTH_LINES_MM", "depth_band", "hot_tier"]
