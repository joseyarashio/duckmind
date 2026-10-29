"""What happened, from what the duck sensed (curiosity spec §5.3–5.8), and the outcome tables
(curiosity spec §13 ``curiosity-outcomes-v1``; physical-design spec §15.2 item 3).

**Classification** uses perception and the body's own signals only — the thing before and after
(two looks, odometry frame), the mouth joint (robotd joint 9), odometry progress, the legs'
currents — never the world's truth:

* ``bump``: ``moved`` (≥ :data:`MOVED_MM`), ``rolled`` (≥ :data:`ROLLED_MM`), ``tipped`` (its
  height class fell), ``stuck`` (forward progress < 30 % of the command's for ≥ 0.6 s, with the
  legs' load up as the second witness), ``nothing`` (it was within reach and did not move),
  ``missed`` (it was out of reach at the end: the duck stopped short);
* ``peck``/``nibble``: ``in_beak`` (the mouth stayed ≥ 3° above closed after closing),
  ``squirted`` (it shot ≥ :data:`SQUIRT_MM` away), ``moved``, ``nothing``, ``unreachable``
  (lower than :data:`PECK_MIN_H_MM`: the beak stops 2 cm above the floor);
* ``grab`` (the relay): ``beak`` ∈ ``held``/``slipped``/``missed``/``tethered``;
* ``kick``: ``rolled``/``moved``/``missed`` with ``dir_mrad``;
* ``tap``: ``moved``/``tipped``/``nothing``; ``look``: ``seen``/``lost``.

The knock's sound is always ``unknown`` in phase 3 (owner decision A5: only the world's truth
knows how loud a knock was; the ear's features come later).

**Tables** (:class:`OutcomeTable`): per verb × thing class × patience tier, the count ``n``, each
outcome's share in ppm with its Wilson 95 % interval, displacement and duration quantiles, and
the fall count with its Wilson upper bound. The same JSON shape whether the rows came from live
runs or a headless sweep; ``physics`` says which.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

MOVED_MM = 15
ROLLED_MM = 300
SQUIRT_MM = 50
KICK_MISS_MM = 30
PECK_MIN_H_MM = 20
TIPPED_HEIGHT_RATIO = 0.6
STUCK_PROGRESS = 0.30
STUCK_MS = 600
IN_BEAK_MARGIN_MRAD = 52  # 3° (curiosity E10: holding 8–13°, empty −4.6…−5°)
MOUTH_CLOSED_MRAD = -87  # robotd's MOUTH_CLOSED, −5°
REACH_SLACK_MM = 40
SOUND_UNKNOWN = "unknown"

VERBS = (
    "look",
    "approach",
    "bump",
    "peck",
    "nibble",
    "tap",
    "grab",
    "carry",
    "put_down",
    "kick",
    "dribble",
)
OUTCOMES: dict[str, tuple[str, ...]] = {
    "look": ("seen", "lost"),
    "approach": ("arrived", "lost", "gave_up", "blocked"),
    "bump": ("moved", "rolled", "tipped", "stuck", "nothing", "missed"),
    "peck": ("in_beak", "pressed", "squirted", "moved", "nothing", "unreachable", "missed"),
    "nibble": ("in_beak", "pressed", "squirted", "moved", "nothing", "unreachable", "missed"),
    "tap": ("moved", "tipped", "nothing", "missed"),
    "grab": ("held", "slipped", "missed", "tethered"),
    "carry": ("held", "slipped", "tethered"),
    "put_down": ("placed", "missed"),
    "kick": ("rolled", "moved", "missed"),
    "dribble": ("rolled", "moved", "missed"),
}
ENDS = (
    "done",
    "lost",
    "gave_up",
    "dark",
    "lifted",
    "fell",
    "preempt",
    "deadline",
    "guard",
    "cone_blocked",
    "not_allowed",
)


def moved_mm(before: tuple[int, int] | None, after: tuple[int, int] | None) -> int | None:
    if before is None or after is None:
        return None
    return int(round(math.hypot(after[0] - before[0], after[1] - before[1])))


def direction_mrad(before: tuple[int, int], after: tuple[int, int], heading_mrad: int) -> int:
    """The thing's travel direction relative to the duck's heading (odometry frame points)."""
    a = math.atan2(after[1] - before[1], after[0] - before[0]) - heading_mrad / 1000.0
    return int(round(((a + math.pi) % (2 * math.pi) - math.pi) * 1000.0))


def tipped(h_before: int | None, h_after: int | None) -> bool:
    return (
        h_before is not None
        and h_after is not None
        and h_before > 0
        and h_after < h_before * TIPPED_HEIGHT_RATIO
    )


def stuck(progress_mm: int, expected_mm: int, *, witnessed: bool) -> bool:
    """Forward progress under 30 % of what the command should have made, and a second witness
    (the legs' load or the IMU): odometry alone mistakes slipping feet for progress (J1)."""
    return expected_mm > 0 and progress_mm < STUCK_PROGRESS * expected_mm and witnessed


def in_beak(mouth_mrad: int | None, *, commanded_open: float = 0.0) -> bool:
    if mouth_mrad is None or commanded_open > 0.0:
        return False
    return mouth_mrad - MOUTH_CLOSED_MRAD >= IN_BEAK_MARGIN_MRAD


def classify_bump(
    *,
    moved: int | None,
    h_before: int | None,
    h_after: int | None,
    progress_mm: int,
    expected_mm: int,
    witnessed: bool,
    gap_after_mm: int | None,
) -> str:
    if moved is not None and moved >= ROLLED_MM:
        return "rolled"
    if tipped(h_before, h_after):
        return "tipped"
    if moved is not None and moved >= MOVED_MM:
        return "moved"
    if stuck(progress_mm, expected_mm, witnessed=witnessed):
        return "stuck"
    if gap_after_mm is not None and gap_after_mm > REACH_SLACK_MM:
        return "missed"
    return "nothing"


def classify_peck(
    *, mouth_mrad: int | None, moved: int | None, reachable: bool, touched_zone: bool = True
) -> str:
    if not reachable:
        return "unreachable"
    if in_beak(mouth_mrad):
        return "in_beak"
    if moved is not None and moved >= SQUIRT_MM:
        return "squirted"
    if moved is not None and moved >= MOVED_MM:
        return "moved"
    return "nothing" if touched_zone else "missed"


def classify_kick(moved: int | None) -> str:
    if moved is None or moved < KICK_MISS_MM:
        return "missed"
    return "rolled" if moved >= ROLLED_MM else "moved"


def classify_tap(*, moved: int | None, h_before: int | None, h_after: int | None) -> str:
    if tipped(h_before, h_after):
        return "tipped"
    if moved is not None and moved >= MOVED_MM:
        return "moved"
    return "nothing"


def gap_bin(mouth_mrad: int | None) -> int | None:
    """How far open the beak stays on the thing, in 3° bins above closed (E10's 8–13° → 3–4)."""
    if mouth_mrad is None:
        return None
    return max(0, (mouth_mrad - MOUTH_CLOSED_MRAD) // IN_BEAK_MARGIN_MRAD)


# ── statistics ──────────────────────────────────────────────────────────────────


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """The Wilson 95 % interval for k successes in n (0..1)."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def quantiles(values: Sequence[float], qs: Sequence[float] = (0.1, 0.5, 0.9)) -> list[int] | None:
    data = sorted(values)
    if not data:
        return None
    out = []
    for q in qs:
        pos = q * (len(data) - 1)
        lo, hi = int(math.floor(pos)), int(math.ceil(pos))
        value = data[lo] + (data[hi] - data[lo]) * (pos - lo)
        out.append(int(round(value)))
    return out


def rate(k: int, n: int) -> dict[str, Any]:
    lo, hi = wilson(k, n)
    return {
        "k": k,
        "n": n,
        "ppm": int(round(1_000_000 * k / n)) if n else None,
        "wilson95_ppm": [int(round(lo * 1_000_000)), int(round(hi * 1_000_000))],
    }


@dataclass(frozen=True)
class Row:
    """One step's result for a table (from a body.curio_step, live or headless)."""

    verb: str
    cls: str
    tier: str
    outcome: str
    moved_mm: int | None = None
    ms: int | None = None
    fell: bool = False

    @classmethod
    def from_step(cls, step: Mapping[str, Any], *, cls_name: str, tier: str) -> Row:
        return cls(
            verb=str(step.get("verb")),
            cls=cls_name,
            tier=tier,
            outcome=str(step.get("outcome")),
            moved_mm=step.get("moved_mm") if isinstance(step.get("moved_mm"), int) else None,
            ms=step.get("ms") if isinstance(step.get("ms"), int) else None,
            fell=bool(step.get("fell")),
        )


class OutcomeTable:
    """``curiosity-outcomes-v1``: outcome shares per verb × class × patience tier."""

    SCHEMA = "curiosity-outcomes-v1"

    def __init__(self, *, physics: str, perception: str, note: str = "") -> None:
        self.physics = physics
        self.perception = perception
        self.note = note
        self.rows: list[Row] = []

    def add(self, row: Row) -> None:
        if row.verb in OUTCOMES and row.outcome not in OUTCOMES[row.verb] + ENDS:
            raise ValueError(f"{row.verb}: unknown outcome {row.outcome!r}")
        self.rows.append(row)

    def extend(self, rows: Iterable[Row]) -> None:
        for row in rows:
            self.add(row)

    def to_json(self) -> dict[str, Any]:
        groups: dict[tuple[str, str, str], list[Row]] = defaultdict(list)
        for row in self.rows:
            groups[(row.verb, row.cls, row.tier)].append(row)
        cells = []
        for (verb, cls_name, tier), rows in sorted(groups.items()):
            n = len(rows)
            counts = Counter(r.outcome for r in rows)
            falls = sum(1 for r in rows if r.fell)
            cells.append(
                {
                    "verb": verb,
                    "cls": cls_name,
                    "tier": tier,
                    "n": n,
                    "outcomes": {name: rate(k, n) for name, k in sorted(counts.items())},
                    "moved_mm_p10_p50_p90": quantiles(
                        [r.moved_mm for r in rows if r.moved_mm is not None]
                    ),
                    "ms_p10_p50_p90": quantiles([r.ms for r in rows if r.ms is not None]),
                    "falls": rate(falls, n),
                }
            )
        return {
            "schema": self.SCHEMA,
            "physics": self.physics,
            "perception": self.perception,
            "note": self.note,
            "n": len(self.rows),
            "cells": cells,
        }


__all__ = [
    "ENDS",
    "KICK_MISS_MM",
    "MOVED_MM",
    "OUTCOMES",
    "OutcomeTable",
    "ROLLED_MM",
    "Row",
    "SOUND_UNKNOWN",
    "VERBS",
    "classify_bump",
    "classify_kick",
    "classify_peck",
    "classify_tap",
    "direction_mrad",
    "gap_bin",
    "in_beak",
    "moved_mm",
    "quantiles",
    "rate",
    "stuck",
    "tipped",
    "wilson",
]
