"""The Act Runner's numbers, measured under the robotd path (curiosity spec §5.2 "可達地圖";
phase-3 re-measurement, 2026-09-27).

Every curiosity number before phase 3 (M1–M5, J1–J4) came from duckphys' raw control law —
no robotd command smoothing, no standing tuning, no filters. Phase 3 measured them again live:
official ``robotd --sim`` (lockstep, allcollisions) driving the official policies in MY world
server on default_home's wood floor, with the provisional jaw (owner decision A4). The script
and raw data are measurement records, not in the repo (the phase-3 curiosity ``measure.py`` and
``live_*.jsonl``); the summary is ``docs/validation/evidence/phase3/curiosity_calib.json``.
Distances are in the trunk frame **standing, before the head goes down** (x ahead of the trunk
origin, y left), millimetres.

What the numbers are, briefly (n and intervals in the evidence file):

* ``ground_pick`` (n = 5) — the beak tip's lowest point is 78 mm ahead, 14 mm left and 21 mm up,
  1.16 s in, while the trunk itself rocks 29 mm back and yaws +9.5°. So the beak reaches a thing
  whose body covers ~70–90 mm ahead; demo.py's "10.8 cm" and curiosity J4's "5.7 cm" (the jaw's
  grip on a 5 cm fin) measure different things. A 70 mm ball centred 70–130 mm ahead was touched
  by the beak 8/8 and pushed 32–47 mm.
* grab (jaw, provisional A4) — the fin 52–70 mm ahead (±20 mm at 57/67): lifted 36/36, the
  mouth then reads +10° (empty −5°); 75 mm: 0/6; 47 mm cannot even be placed (inside the duck's
  feet). A 1.0 s relay leg kept it 24/24 (carried ~0.24 m); a 1.7 s leg dropped it 9/9 near the
  leg's end (M5 said 1.6–1.8 s).
* steps from standing (n = 10–20 each, 0 falls in 215): creep 0.5 s 42 mm; 0.4 s 29 mm; trim
  0.3 s only 6 mm (it mostly turns: −6°); bump 0.6 s 61 mm; leg 1.0 s 125 mm; head-down creep
  34 mm; reversing spins up slowly: back 0.6 s −2 mm, 1.0 s −34 mm, 1.5 s −123 mm; a 1.0 s
  reverse arc turns +36° (left) / −32° (right) while backing 75–115 mm.
* bump — the near face 50 mm ahead: a 0.6 s creep touched and moved it 8/8 (box 29 mm, cup
  38 mm, ball 85 mm); at 100 mm: 0/12 touched (the demo's standoff never reaches).
* kick (ball_felt on wood, n = 72 spots) — rolls ≥ 30 cm with the ball 65–95 mm ahead and
  25–70 mm toward the foot (23/24); none from 125 mm; the ball goes +24° (left foot) / −20°
  (right) off the heading, 0.4–0.9 m.

Constants marked ``provisional`` depend on the guessed jaw pivot (A4) or on a small n; they are
re-measured when the pivot is measured or the headless skill path exists.

Pure: standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Window:
    """Where the thing must be (trunk frame, mm) for a verb to work: ``ahead`` and ``side``
    ranges; ``side`` is toward the kicking foot for kicks (positive), plain y otherwise."""

    ahead: tuple[int, int]
    side: tuple[int, int]

    @property
    def centre(self) -> tuple[int, int]:
        return ((self.ahead[0] + self.ahead[1]) // 2, (self.side[0] + self.side[1]) // 2)

    def contains(self, ahead: int, side: int, tol_pct: int = 100) -> bool:
        ca, cs = self.centre
        ha = (self.ahead[1] - self.ahead[0]) * tol_pct // 200
        hs = (self.side[1] - self.side[0]) * tol_pct // 200
        return abs(ahead - ca) <= ha and abs(side - cs) <= hs


@dataclass(frozen=True)
class Calib:
    """The Act Runner's calibration (see the module docstring)."""

    source: str
    # distance the trunk moves forward for one primitive from standing (median, mm)
    step_mm: dict[str, int] = field(default_factory=dict)
    # the peck: where the beak tip lands (trunk frame at the start)
    beak_tip_mm: tuple[int, int, int] = (78, 14, 21)
    beak_bottom_ms: int = 1160
    peck: Window = Window((75, 125), (-20, 20))
    grab: Window = Window((52, 70), (-20, 20))
    relay_ms_max: int = 1300  # a longer relay leg drops the fin (1.7 s: 9/9 dropped)
    bump_gap_mm: int = 50  # the near face this far ahead of the trunk origin before the bump creep
    front_mm: int = (
        38  # the duck's front (chest and feet) ahead of the trunk origin, at a toy's height
    )
    back_ms: int = 1000  # a reverse step (robotd spins up backwards in ~1 s: 0.6 s moves 2 mm)
    back_arc_ms: int = 1000
    tap_gap_mm: int = 58
    kick_window: Window = Window((65, 90), (28, 70))
    kick_dir_mrad: dict[str, int] = field(default_factory=lambda: {"left": 414, "right": -344})
    kick_cone_half_mrad: int = 524  # ±30°: the live directions spread wider than duckphys' ±20°
    # a ball's roll on this floor after a clean kick: p10, p90 (mm), for the kick cone
    roll_mm: dict[str, tuple[int, int]] = field(default_factory=lambda: {"ball_slow": (385, 870)})
    # straight steps from standing, (command ms, forward mm incl. the tail), live robotd medians
    fwd_table: tuple[tuple[int, int], ...] = (
        (300, 6),
        (400, 29),
        (500, 42),
        (600, 61),
        (1000, 125),
        (2000, 290),
    )
    back_table: tuple[tuple[int, int], ...] = ((600, -2), (1000, -34), (1500, -123))
    look_ahead_mm: tuple[int, int] = (250, 400)
    eye_near_mm: int = 95  # the full-down look sees the floor from ~95 mm ahead (spec §4.1)
    skill_ms: dict[str, int] = field(
        default_factory=lambda: {"ground_pick": 2850, "kick_left": 1000, "kick_right": 1000}
    )
    mouth_open_ms: int = 200  # grab: open at 0.2 s into ground_pick …
    mouth_close_ms: int = 1400  # … close at 1.4 s (curiosity M5)
    provisional: tuple[str, ...] = (
        "grab",
        "carry",
        "kick_dir_mrad",
        "roll_mm",
        "kick_cone_half_mrad",
    )


CALIB = Calib(
    source="live robotd --sim, default_home wood floor, allcollisions, provisional jaw (A4); "
    "phase 3, 2026-09-27; measurement records live_v1.jsonl + live_v2.jsonl (not in the repo); "
    "docs/validation/evidence/phase3/curiosity_calib.json",
    step_mm={
        "creep": 42,
        "creep04": 29,
        "trim": 6,
        "bump": 61,
        "leg1": 125,
        "creep_down": 34,
        "back": -34,
        "back_arc": -95,
    },
)


def step_ms(table: tuple[tuple[int, int], ...], want_mm: float) -> int:
    """The command duration whose measured displacement is nearest ``want_mm`` (piecewise linear
    in the table, clamped to its ends)."""
    points = sorted(table, key=lambda p: p[1])
    if want_mm <= points[0][1]:
        return points[0][0]
    if want_mm >= points[-1][1]:
        return points[-1][0]
    for (ms0, d0), (ms1, d1) in zip(points, points[1:], strict=False):
        if d0 <= want_mm <= d1:
            frac = 0.0 if d1 == d0 else (want_mm - d0) / (d1 - d0)
            return int(round(ms0 + frac * (ms1 - ms0)))
    return points[-1][0]


@dataclass(frozen=True)
class Patience:
    """The patience parameters one step runs with (curiosity spec §7.7: wits → patience is the
    only way personality reaches the physics). Tiers: ``hasty``, ``normal``, ``careful``."""

    tier: str
    tol_pct: int
    settle_ms: int
    align_budget_ms: int
    relay_ms: int

    @classmethod
    def from_wits(cls, wits: int) -> Patience:
        w = max(1500, min(8500, int(wits))) - 1500
        tol = 130 - w * 50 // 7000
        settle = 300 + w * 700 // 7000
        budget = 8000 + w * 12000 // 7000
        relay = 1700 - w * 700 // 7000
        tier = "hasty" if tol >= 115 else "careful" if tol <= 95 else "normal"
        return cls(tier, tol, settle, budget, relay)

    def frustrated(self) -> Patience:
        """After a miss: tol −10, settle +200 (within bounds, §7.7)."""
        return Patience(
            self.tier,
            max(80, self.tol_pct - 10),
            min(1000, self.settle_ms + 200),
            self.align_budget_ms,
            self.relay_ms,
        )


TIERS = {
    "hasty": Patience("hasty", 130, 300, 8000, 1700),
    "normal": Patience("normal", 100, 600, 14000, 1300),
    "careful": Patience("careful", 80, 1000, 20000, 1000),
}


__all__ = ["CALIB", "Calib", "Patience", "TIERS", "Window", "step_ms"]
