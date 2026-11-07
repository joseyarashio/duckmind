"""How the legs of the loaded networks are loaded at rest — the sit-stand network's seat and the walk network's standing
stance — and proprio's lift numbers for them.

proprio claims a seated duck is in a hand when its legs go light: under ``unsure_load_sit`` for a moment, then under
``lift_load_sit`` for ``lift_hold_sit_s`` (and a claim ends when the load is back in ``floor_band_sit``). Those numbers
were set on Pollen's sit-stand network of policy set v5, whose seat loads the legs ≈ 26 (the simulator's leg-force proxy).
The network of set v8 (2026-11-05) sits lower and lands its seat in a bounce: its seat loads the legs ≈ 23, and about
1-1.5 s after robotd's ``sit`` label the legs go light for a few tenths of a second as the trunk settles. On v8 the v5
numbers claimed a pickup 11 times an hour of free roam (v5: 2), most of them right after sitting down.

So proprio's seated numbers follow the loaded seat: a measured table per sit-stand file (keyed by its sha256,
``seat_load.json`` next to this module) holds the seat's leg load (the median over 3-12 s after the ``sit`` label, on the
floor) and how long its seat-down unload lasts, measured on the live robotd in the simulator with its servo command
delay. One entry is the reference (v5: the seat the numbers were set on). For any other measured seat the seated claim
levels (``unsure_load_sit``, ``lift_load_sit`` and the floor band's lower edge) are scaled by the seat's load over the
reference's, and the seated light-legs claim waits out the seat-down unload (``seat_settle_s``); a duck held in the hand
carries nothing on its legs (≈ 14 seated, ≈ 9 standing), well under either. The reference seat, or a seat without a
table, keeps proprio's numbers unchanged.

The standing claim (legs under ``lift_load`` for ``lift_hold_s``, stopped) was set on v5's walk network, whose standing
stance loads the legs ≈ 37 and stayed above 19 in every 0.3 s of 8 h of v5 records. v8's walk network (with the
backlash body) stands at ≈ 32, but for one to three seconds after a stop, a rise or a turn it settles into a stance that
loads the legs only 15.6-19.7 (2026-11-06: 7 such stances held under 20 for 0.3 s in 4.2 h of V8Fix records, 12 in 5.8 h
of V8Brush / BrushModel records) — each a pickup claim, and the claim sits the duck down: the 4 h 15 min soak's five
"seated" false lifts on v8 were all these standing claims (robotd's ``sit`` 0.5-2 s before the soak saw the claim was the
claim's own sit-down). A duck standing in the hand carries ≈ 10 (median of 480 in-hand ticks on v8; ≤ 12.5 once the
lift's swing is over). So the walk network has its own table (``stances``, keyed by the walk file's sha256): the measured
standing level for a walk other than the reference walk (v5) is the claim's ``lift_load`` (never higher than proprio's
own); the reference walk, or a walk without an entry, keeps proprio's number.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from duckmind.duckbody.proprio import ProprioConfig

DATA = Path(__file__).with_name("seat_load.json")
SEAT_FILE = "alpha_sitstand.onnx"


@dataclass(frozen=True)
class SeatLoad:
    """One sit-stand network's measured seat: ``load`` (the legs' load seated on the floor, proprio's raw units) and
    ``settle_s`` (its seat-down unload is over this long after robotd's ``sit`` label), on the ``body`` collision
    model with a ``delay_ms`` servo command delay."""

    sitstand_sha256: str
    name: str
    body: str
    delay_ms: float
    load: float
    settle_s: float

    @staticmethod
    def of(entry: Mapping[str, object]) -> SeatLoad:
        out = SeatLoad(
            sitstand_sha256=str(entry["sitstand_sha256"]),
            name=str(entry["name"]),
            body=str(entry["body"]),
            delay_ms=float(entry["delay_ms"]),  # type: ignore[arg-type]
            load=float(entry["load"]),  # type: ignore[arg-type]
            settle_s=float(entry["settle_s"]),  # type: ignore[arg-type]
        )
        if not out.load > 0.0 or out.settle_s < 0.0:
            raise ValueError(f"seat load {out.name}: load must be > 0 and settle_s >= 0")
        return out


@dataclass(frozen=True)
class StanceLoad:
    """One walk network's measured standing stance on the ``body`` collision model with a ``delay_ms`` servo command
    delay: ``floor_low`` (the lowest load its legs stayed under for 0.3 s standing stopped on a floor), ``held`` (the
    median load standing in a hand) and ``lift_load`` (the standing claim's level between them, proprio's raw units)."""

    walk_sha256: str
    name: str
    body: str
    delay_ms: float
    floor_low: float
    held: float
    lift_load: float

    @staticmethod
    def of(entry: Mapping[str, object]) -> StanceLoad:
        out = StanceLoad(
            walk_sha256=str(entry["walk_sha256"]),
            name=str(entry["name"]),
            body=str(entry["body"]),
            delay_ms=float(entry["delay_ms"]),  # type: ignore[arg-type]
            floor_low=float(entry["floor_low"]),  # type: ignore[arg-type]
            held=float(entry["held"]),  # type: ignore[arg-type]
            lift_load=float(entry["lift_load"]),  # type: ignore[arg-type]
        )
        if not out.held < out.lift_load < out.floor_low:
            raise ValueError(f"stance load {out.name}: held < lift_load < floor_low")
        return out


@dataclass(frozen=True)
class SeatTable:
    """Every measured seat (by sha256) and which one is the reference; every measured standing stance (by the walk
    file's sha256) and the reference walk."""

    reference: str
    seats: Mapping[str, SeatLoad]
    stance_reference: str | None = None
    stances: Mapping[str, StanceLoad] = field(default_factory=dict)

    def seat_for(self, sha256: str | None, body: str | None, delay_ms: float) -> SeatLoad | None:
        """The measured seat for a body running the sit-stand network ``sha256`` on the collision model ``body`` with a
        ``delay_ms`` servo command delay; None for the reference seat, an unmeasured one, or one measured on another
        body or with another delay."""
        if sha256 is None or sha256 == self.reference:
            return None
        loaded, ref = self.seats.get(sha256), self.seats.get(self.reference)
        if loaded is None or ref is None or (body is not None and loaded.body != body):
            return None
        if abs(loaded.delay_ms - delay_ms) > 1e-6 or abs(ref.delay_ms - delay_ms) > 1e-6:
            return None
        return loaded

    def proprio_for(
        self, cfg: ProprioConfig, sha256: str | None, body: str | None, delay_ms: float
    ) -> ProprioConfig:
        """``cfg`` with its seated lift numbers for this seat (module doc); ``cfg`` itself when :meth:`seat_for` has
        none."""
        seat = self.seat_for(sha256, body, delay_ms)
        if seat is None:
            return cfg
        scale = seat.load / self.seats[self.reference].load
        lo, hi = cfg.floor_band_sit
        return replace(
            cfg,
            unsure_load_sit=cfg.unsure_load_sit * scale,
            lift_load_sit=cfg.lift_load_sit * scale,
            floor_band_sit=(lo * scale, hi),
            seat_settle_s=seat.settle_s,
        )

    def stance_for(
        self, walk_sha256: str | None, body: str | None, delay_ms: float
    ) -> StanceLoad | None:
        """The measured standing stance for a body running the walk network ``walk_sha256`` on the collision model
        ``body`` with a ``delay_ms`` servo command delay; None for the reference walk, an unmeasured one, or one measured
        on another body or with another delay."""
        if walk_sha256 is None or walk_sha256 == self.stance_reference:
            return None
        loaded = self.stances.get(walk_sha256)
        if loaded is None or (body is not None and loaded.body != body):
            return None
        if abs(loaded.delay_ms - delay_ms) > 1e-6:
            return None
        return loaded

    def stand_proprio_for(
        self, cfg: ProprioConfig, walk_sha256: str | None, body: str | None, delay_ms: float
    ) -> ProprioConfig:
        """``cfg`` with the standing claim's level for this walk's stance (module doc); ``cfg`` itself when
        :meth:`stance_for` has none or its level is not lower."""
        stance = self.stance_for(walk_sha256, body, delay_ms)
        if stance is None or stance.lift_load >= cfg.lift_load:
            return cfg
        return replace(cfg, lift_load=stance.lift_load)


def load(path: Path = DATA) -> SeatTable:
    data = json.loads(path.read_text(encoding="utf-8"))
    seats = {s.sitstand_sha256: s for s in (SeatLoad.of(e) for e in data["seats"])}
    if data["reference"] not in seats:
        raise ValueError("seat load table: the reference seat has no entry")
    stances = {s.walk_sha256: s for s in (StanceLoad.of(e) for e in data.get("stances", ()))}
    ref_walk = data.get("stance_reference")
    return SeatTable(
        reference=str(data["reference"]),
        seats=seats,
        stance_reference=None if ref_walk is None else str(ref_walk),
        stances=stances,
    )


def seat_sha256(policies: Path | None) -> str | None:
    """sha256 of the sit-stand file in the policy folder ``policies`` (None when there is none)."""
    if policies is None:
        return None
    try:
        return hashlib.sha256((Path(policies) / SEAT_FILE).read_bytes()).hexdigest()
    except OSError:
        return None


__all__ = ["DATA", "SEAT_FILE", "SeatLoad", "SeatTable", "StanceLoad", "load", "seat_sha256"]
