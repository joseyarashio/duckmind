"""The ToF touch band of one sensor, from the guided measurement (a flat board at known distances).

depth's ``touch`` is a count: ``touch_zones`` or more of the 64 zones inside the too-close band (``too_close_mm`` of
levelled reach from the sensor). Both were set on the simulator's sensor model. On a board the VL53L8CX reads a pressed
face its own way (short ranges, or statuses that are no return at all), so the count is measured: at every distance
the recording holds, how many zones fall inside each candidate band.

The band is right when a face **on** the board (``touch_max_mm`` and nearer) always counts above a board that is
**clear** (``clear_min_mm`` and farther), with room between (:data:`MIN_GAP` zones). The simulator's numbers are kept
when they fit this sensor; else the band with the widest gap is taken and the count set in its middle. When no band
separates the two, a touch cannot be a too-close count on this sensor: nothing is fitted, and the report shows what
the sensor read at each distance.

Pure (the recording's ToF lines in, numbers out).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from duckmind.bringup.fit import quantile
from duckmind.bringup.record import segments
from duckmind.duckbody import depth as D
from duckmind.duckbody import tof_reproject as tr
from duckmind.duckbody.calibration import SIMULATOR, TofBands
from duckmind.duckbody.robotd_client import RobotState

TOUCH_MAX_MM = 30
CLEAR_MIN_MM = 150
BANDS_MM = tuple(range(40, 301, 10))
MIN_GAP = 16  # zones between the lowest count on the board and the highest count clear of it
STANDING_HEIGHT_M = 0.12


@dataclass(frozen=True)
class Row:
    """One distance: the frames' zone counts inside each candidate band, and what the sensor returned."""

    mm: int
    frames: int
    valid_p50: float
    nearest_mm_p50: float | None
    counts: Mapping[int, tuple[int, ...]]  # band → the too-close count of every frame

    def to_json(self, band: int) -> dict[str, Any]:
        counts = self.counts[band]
        return {
            "board_mm": self.mm,
            "frames": self.frames,
            "valid_zones_p50": self.valid_p50,
            "nearest_return_mm_p50": self.nearest_mm_p50,
            "too_close_zones": {
                "min": min(counts),
                "p50": quantile(counts, 0.5),
                "max": max(counts),
            },
        }


@dataclass
class TofFit:
    bands: TofBands | None = None
    rows: list[Row] = field(default_factory=list)
    why: str = ""  # why nothing was fitted
    gap: int | None = None
    kept_simulator: bool = False

    def to_json(self) -> dict[str, Any]:
        band = self.bands.too_close_mm if self.bands is not None else SIMULATOR.tof.too_close_mm  # type: ignore[union-attr]
        return {
            "bands": None if self.bands is None else self.bands.to_json(),
            "kept_the_simulators": self.kept_simulator,
            "gap_zones": self.gap,
            "why_not": self.why,
            "rule": {
                "touch_max_mm": TOUCH_MAX_MM,
                "clear_min_mm": CLEAR_MIN_MM,
                "min_gap_zones": MIN_GAP,
            },
            "shown_band_mm": band,
            "rows": [row.to_json(band) for row in self.rows],
        }


def _row(mm: int, group: Sequence[Mapping[str, Any]]) -> Row | None:
    counts: dict[int, list[int]] = {band: [] for band in BANDS_MM}
    valid: list[int] = []
    nearest: list[float] = []
    previous: list[int | None] | None = None
    for line in group:
        state = RobotState.from_wire(line["state"])
        raw = line["tof"]
        if state.tof is None:
            raise ValueError("robot.state carries no frames.tof: the sensor's pose is unknown")
        frame = D.ToFFrame.of(
            int(raw.get("t_ns") or state.t_ns),
            raw.get("distance_mm") or [],
            raw.get("status") or [],
        )
        pos, quat = state.tof
        pose = D.pose_from_frames(
            pos, quat, state.down or (0.0, 0.0, -1.0), D.trunk_height_mm(state.odom_pos)
        )
        ranges, previous, _ = D.accept_weak(frame, previous, D.DEFAULT.weak_window_mm)
        posture = tr.Posture(
            gravity=pose.gravity,
            trunk_height_m=None if pose.trunk_height_mm is None else pose.trunk_height_mm / 1000.0,
        )
        got = [r for r in ranges if r is not None]
        valid.append(len(got))
        if got:
            nearest.append(min(got) * 1000.0)
        for band in BANDS_MM:
            zones = tr.project(
                ranges, pose.sensor, posture, STANDING_HEIGHT_M, min_range_m=band / 1000.0
            )
            counts[band].append(sum(zone.kind == "too_close" for zone in zones))
    if not valid:
        return None
    return Row(
        mm=mm,
        frames=len(valid),
        valid_p50=quantile(valid, 0.5),
        nearest_mm_p50=round(quantile(nearest, 0.5), 1) if nearest else None,
        counts={band: tuple(values) for band, values in counts.items()},
    )


def fit(lines: Sequence[Mapping[str, Any]]) -> TofFit | None:
    """The touch band from the recording's ``tof_<mm>`` segments; None when the recording has none."""
    by_mm: dict[int, list[Mapping[str, Any]]] = {}
    for name, _, group in segments(lines):
        if name.startswith("tof_"):
            by_mm.setdefault(int(name[4:]), []).extend(line for line in group if "tof" in line)
    if not by_mm:
        return None
    out = TofFit()
    for mm in sorted(by_mm):
        row = _row(mm, by_mm[mm])
        if row is not None:
            out.rows.append(row)
    on = [row for row in out.rows if row.mm <= TOUCH_MAX_MM]
    clear = [row for row in out.rows if row.mm >= CLEAR_MIN_MM]
    if not on or not clear:
        out.why = (
            f"the measurement needs the board on the face (≤ {TOUCH_MAX_MM} mm) and clear of it "
            f"(≥ {CLEAR_MIN_MM} mm)"
        )
        return out

    def gap(band: int) -> tuple[int, int]:
        low = min(min(row.counts[band]) for row in on)  # the fewest zones a pressed face gave
        high = max(max(row.counts[band]) for row in clear)  # the most a clear board gave
        return low, high

    sim = SIMULATOR.tof
    assert sim is not None
    low, high = gap(sim.too_close_mm)
    if high + MIN_GAP // 2 <= sim.touch_zones <= low - MIN_GAP // 2:
        out.bands, out.gap, out.kept_simulator = sim, low - high, True
        return out
    best = max(
        BANDS_MM, key=lambda band: (gap(band)[0] - gap(band)[1], -abs(band - sim.too_close_mm))
    )
    low, high = gap(best)
    out.gap = low - high
    if low - high < MIN_GAP:
        out.why = (
            f"no too-close band tells a face on the board from a clear board: the best ({best} mm) counts "
            f"{low} zones at the least on the board and {high} at the most clear of it"
        )
        return out
    out.bands = TofBands(touch_zones=max(1, (low + high + 1) // 2), too_close_mm=best)
    return out


__all__ = ["BANDS_MM", "CLEAR_MIN_MM", "MIN_GAP", "TOUCH_MAX_MM", "Row", "TofFit", "fit"]
