"""cornered: may the duck take a small step back? (owner 2026-09-28 「2. 可以」; phase-3 verify
finding (f): within 30 cm of a small thing no move was allowed — no turning in place, no
reversing — so the duck stayed stuck: 511 ToF stops and 27 hand placements in the live hour.)

The duck is **cornered** when duck-body's own senses, at the spot where it stands now, say every
way forward is shut — ahead, to the left and to the right:

* **ahead** — the ToF's ahead corridor within the stop distance (the near reflex would halt any
  forward walk, straight or arc) or its touch; a front contact (proprio's ``blocked`` or
  ``stumbled``); a walk straight on that a reflex or a contact ended before it went
  :data:`MOVED_MM`; a look that saw a wall or a thing end the floor within :data:`AHEAD_MM`
  straight ahead;
* **left / right** — the ToF's side corridor within :data:`SIDE_MM`; a contact on that side
  (``blocked`` or ``brushed``); an arc to that side that a reflex or a contact ended before it
  went :data:`MOVED_MM`; a look that saw a wall or a thing within :data:`SIDE_MM` on that side.

A mark holds while the duck stays where it was set — within :data:`STAY_MM` and
:data:`STAY_MRAD` by its own odometry — and for at most :data:`HOLD_MS`; a new odometry epoch
(lifted, set down) forgets them all. The back step itself moves the duck out of its marks, so a
second one needs the corner found again.

What it grants (walkexec and the guard, :mod:`duckmind.duckbody.reflex`): one ``duck.walk`` back
step or back arc at −0.30…−0.40 m/s and at most :data:`~duckmind.duckbody.reflex.BACK_STEP_MAX_MM`.

Pure and integer (mm, mrad, ms); no clock, no I/O.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from duckmind.duckbody.fixedmath import isqrt
from duckmind.duckbody.protocol import STOP_AHEAD_MM, DepthView

WAYS = ("ahead", "left", "right")
AHEAD_MM = STOP_AHEAD_MM  # a look's middle sector ending this near shuts the way ahead
SIDE_MM = 350  # a ToF side corridor or a look's side sectors this near shut that side
MOVED_MM = 60  # a walk a reflex or a contact ended before it went this far: its way is shut
ARC_MRAD = 500  # walkexec's arc rule: |vyaw| ≥ this is an arc to that side
STAY_MM = 80  # a mark holds while the duck stays this close to where it was set …
STAY_MRAD = 350  # … turned no more than this (20°) …
HOLD_MS = 60_000  # … and for at most a minute
SIDE_BEARING_MRAD = 200  # a scan's sector this far off the heading (either way) is that side
HALTS = ("reflex_near", "reflex_touch", "reflex_drop", "stall", "stumble")
ENDS = ("wall", "thing")  # what a look saw end the floor (unknown and trip lines shut nothing)


@dataclass(frozen=True)
class Spot:
    """The trunk in duck-body's odometry frame (mm, mrad)."""

    x_mm: int
    y_mm: int
    h_mrad: int

    @classmethod
    def of(cls, odom: tuple[float, float, float]) -> Spot:
        """From the posture tap's ``odom`` (metres, radians)."""
        x, y, h = odom
        return cls(int(round(x * 1000.0)), int(round(y * 1000.0)), int(round(h * 1000.0)))


@dataclass(frozen=True)
class Mark:
    t_ms: int
    at: Spot
    why: str


def _turn(a: int, b: int) -> int:
    return (a - b + 3141) % 6283 - 3141


def _way_of_side(side: object) -> str:
    """A contact side as a way: ``front`` and ``unknown`` shut the way ahead."""
    return str(side) if side in ("left", "right") else "ahead"


@dataclass
class Cornered:
    """Where the duck found its ways shut (see the module docstring). Feed it every tick
    (:meth:`sense`), every ``walk.done`` (:meth:`walked`) and every ``sense.scan``
    (:meth:`scanned`); ask :meth:`now`."""

    marks: dict[str, Mark] = field(default_factory=dict)
    epoch: int = 0

    def sync(self, epoch: int) -> None:
        if epoch != self.epoch:
            self.marks.clear()
            self.epoch = epoch

    def mark(self, way: str, t_ms: int, at: Spot, why: str) -> None:
        if way in WAYS:
            self.marks[way] = Mark(t_ms, at, why)

    def holds(self, way: str, t_ms: int, at: Spot) -> bool:
        m = self.marks.get(way)
        if m is None or t_ms < m.t_ms or t_ms - m.t_ms > HOLD_MS:
            return False
        if isqrt((at.x_mm - m.at.x_mm) ** 2 + (at.y_mm - m.at.y_mm) ** 2) > STAY_MM:
            return False
        return abs(_turn(at.h_mrad, m.at.h_mrad)) <= STAY_MRAD

    def now(self, t_ms: int, at: Spot) -> bool:
        """Every way forward shut at this spot."""
        return all(self.holds(way, t_ms, at) for way in WAYS)

    def shut(self, t_ms: int, at: Spot) -> dict[str, str]:
        """The ways that hold now and why (for the record)."""
        return {w: self.marks[w].why for w in WAYS if self.holds(w, t_ms, at)}

    # ── what the senses say ─────────────────────────────────────────────────

    def sense(
        self,
        t_ms: int,
        at: Spot,
        depth: DepthView | None,
        edges: tuple[str, ...] = (),
        side: str = "unknown",
    ) -> None:
        """One tick: the ToF's corridors and proprio's contact edges (``side`` is proprio's)."""
        if depth is not None:
            if depth.touch or (depth.ahead_mm is not None and depth.ahead_mm <= STOP_AHEAD_MM):
                self.mark("ahead", t_ms, at, "tof")
            if depth.left_mm is not None and depth.left_mm <= SIDE_MM:
                self.mark("left", t_ms, at, "tof")
            if depth.right_mm is not None and depth.right_mm <= SIDE_MM:
                self.mark("right", t_ms, at, "tof")
        for edge in edges:
            if edge in ("blocked", "stumbled"):
                self.mark(_way_of_side(side), t_ms, at, f"contact:{edge}")
            elif edge == "brushed" and side in ("left", "right"):
                self.mark(side, t_ms, at, "contact:brushed")

    def walked(self, t_ms: int, at: Spot, event: Mapping[str, Any]) -> None:
        """A forward ``walk.done`` that ended here by a reflex or a contact: stopped at once (under
        :data:`MOVED_MM`), it shuts the way it tried (straight: ahead; an arc: its side); stopped
        after going on, the way ahead of where it stands now. Its contact shuts the side it
        names; its end look (taken here) adds what it saw."""
        cmd = event.get("cmd") or {}
        odom = event.get("odom") or {}
        vx = int(cmd.get("vx", 0) or 0)
        vyaw = int(cmd.get("vyaw", 0) or 0)
        path = int(odom.get("path_mm", 0) or 0)
        reason = event.get("reason")
        if vx <= 0:
            return  # a back step says nothing about the ways forward
        if reason in HALTS:
            way = "ahead"
            if path < MOVED_MM:
                way = "left" if vyaw >= ARC_MRAD else "right" if vyaw <= -ARC_MRAD else "ahead"
            self.mark(way, t_ms, at, f"walk:{reason}")
        contact = event.get("contact")
        if isinstance(contact, Mapping) and contact.get("kind") in ("blocked", "stumbled"):
            self.mark(_way_of_side(contact.get("side")), t_ms, at, f"walk:{contact.get('kind')}")
        elif isinstance(contact, Mapping) and contact.get("side") in ("left", "right"):
            self.mark(str(contact["side"]), t_ms, at, f"walk:{contact.get('kind')}")
        end = event.get("end")
        if isinstance(end, Mapping):
            self.looked(t_ms, at, end)

    def looked(self, t_ms: int, at: Spot, fold: Mapping[str, Any]) -> None:
        """A folded look taken here (``free_mm[5]``, ``ends[5]``; sector 0 is the picture's
        left): a wall or thing near in the middle shuts ahead, in the outer sectors a side."""
        if fold.get("light") not in ("bright", "dim"):
            return  # a dark frame sees no floor
        free = list(fold.get("free_mm") or [])
        ends = list(fold.get("ends") or [])
        for i, (mm, end) in enumerate(zip(free, ends, strict=False)):
            if end not in ENDS:
                continue
            if i == 2 and int(mm) <= AHEAD_MM:
                self.mark("ahead", t_ms, at, f"look:{end}")
            elif i < 2 and int(mm) <= SIDE_MM:
                self.mark("left", t_ms, at, f"look:{end}")
            elif i > 2 and int(mm) <= SIDE_MM:
                self.mark("right", t_ms, at, f"look:{end}")

    def scanned(self, t_ms: int, at: Spot, event: Mapping[str, Any]) -> None:
        """A ``sense.scan`` taken here: its sectors carry their bearing (trunk frame, left +)."""
        if event.get("light") not in ("bright", "dim"):
            return
        for sector in event.get("sectors") or []:
            if not isinstance(sector, Mapping) or sector.get("end") not in ENDS:
                continue
            b = int(sector.get("b_mrad", 0) or 0)
            mm = int(sector.get("free_mm", 0) or 0)
            if abs(b) < SIDE_BEARING_MRAD:
                if mm <= AHEAD_MM:
                    self.mark("ahead", t_ms, at, f"scan:{sector['end']}")
            elif mm <= SIDE_MM:
                self.mark("left" if b > 0 else "right", t_ms, at, f"scan:{sector['end']}")


__all__ = [
    "AHEAD_MM",
    "HOLD_MS",
    "MOVED_MM",
    "SIDE_MM",
    "STAY_MM",
    "STAY_MRAD",
    "WAYS",
    "Cornered",
    "Mark",
    "Spot",
]
