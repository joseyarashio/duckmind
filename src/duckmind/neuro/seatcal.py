"""How the legs of the loaded sit-stand network's seat are loaded, and proprio's seated lift numbers for it.

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
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
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
class SeatTable:
    """Every measured seat (by sha256) and which one is the reference."""

    reference: str
    seats: Mapping[str, SeatLoad]

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


def load(path: Path = DATA) -> SeatTable:
    data = json.loads(path.read_text(encoding="utf-8"))
    seats = {s.sitstand_sha256: s for s in (SeatLoad.of(e) for e in data["seats"])}
    if data["reference"] not in seats:
        raise ValueError("seat load table: the reference seat has no entry")
    return SeatTable(reference=str(data["reference"]), seats=seats)


def seat_sha256(policies: Path | None) -> str | None:
    """sha256 of the sit-stand file in the policy folder ``policies`` (None when there is none)."""
    if policies is None:
        return None
    try:
        return hashlib.sha256((Path(policies) / SEAT_FILE).read_bytes()).hexdigest()
    except OSError:
        return None


__all__ = ["DATA", "SEAT_FILE", "SeatLoad", "SeatTable", "load", "seat_sha256"]
