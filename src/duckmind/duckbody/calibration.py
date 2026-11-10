"""The body-sense calibration document: the numbers the duck's body sense reads off one particular body.

Two of the body's senses compare a raw reading with a level that only that body can give:

* **the leg load** (:mod:`duckmind.duckbody.proprio`): the sum of the ten leg servos' currents. Picked up, set down,
  legs flailing in the air and "the floor is back" are all bands of it. In the simulator it is a leg-force proxy
  (|actuator force| × 100); on a board it is milliamps of ten real servos — another quantity on another scale;
* **the ToF's too-close band** (:mod:`duckmind.duckbody.depth`): a return whose levelled horizontal reach from the
  sensor is under ``too_close_mm`` is the cover glass's crosstalk, and ``touch_zones`` or more of the 64 zones inside it is a face pressed on something.

Both come from one document (schema :data:`SCHEMA`), keyed by **the board and the two networks the body runs** (the walk
and sit-stand files' sha256: another network stands and sits with other leg loads). The simulator's numbers ship next to
this module (``body_calibration.simulator.json``) and are what :class:`~duckmind.duckbody.proprio.ProprioConfig` and
:class:`~duckmind.duckbody.depth.DepthConfig` default to, so a simulated body behaves as it always did. A real body's
document is measured on the bench (``python -m duckmind.bringup``, docs/deploy/bring-up.md) and kept in
:func:`folder`; without one for the networks it runs, the brain does not walk it (:func:`find` says why).

Pure but for :func:`folder` / :func:`find` / :func:`save` (files).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "duckmind.body-calibration/1"
SIMULATOR_BOARD = "simulator"
REHEARSAL_BOARD = "rehearsal"  # ``sim-body --as-real``: a simulated duck that plays a real one
DATA = Path(__file__).with_name("body_calibration.simulator.json")
SHA256 = re.compile(r"[0-9a-f]{64}")
ZONES = 64
# why a real body has no calibration (:class:`Found`): the codes the page's words are chosen by
WHY_NONE = "none"  # nothing was ever measured for this board
WHY_OTHER = "other_networks"  # measured, but with other walk / sit-stand networks than it runs now
WHY_INVALID = "invalid"  # a document for it is there and cannot be read (schema, missing or impossible numbers)
WHY_UNKNOWN_BODY = (
    "unknown_body"  # the bridge did not say the board or the networks: nothing can be matched
)
WHYS = (WHY_NONE, WHY_OTHER, WHY_INVALID, WHY_UNKNOWN_BODY)


class CalibrationError(ValueError):
    """A calibration document that cannot be used, with the reason in plain English."""


@dataclass(frozen=True)
class LoadBands:
    """proprio's leg-load levels (the raw units of ``RobotState.leg_load`` on that body). A seated level of 0 turns the
    seated light-legs claim off (a body whose seat does not load the legs more than a hand does)."""

    unsure_load: float
    unsure_load_sit: float
    unsure_load_reverse: float
    lift_load: float
    lift_load_sit: float
    flail_load: float
    floor_band: tuple[float, float]
    floor_band_sit: tuple[float, float]

    def to_json(self) -> dict[str, Any]:
        return {
            "unsure_load": self.unsure_load,
            "unsure_load_sit": self.unsure_load_sit,
            "unsure_load_reverse": self.unsure_load_reverse,
            "lift_load": self.lift_load,
            "lift_load_sit": self.lift_load_sit,
            "flail_load": self.flail_load,
            "floor_band": list(self.floor_band),
            "floor_band_sit": list(self.floor_band_sit),
        }


@dataclass(frozen=True)
class TofBands:
    """depth's too-close band (mm of a return's levelled horizontal reach from the sensor) and how many zones inside it are a
    touch."""

    touch_zones: int
    too_close_mm: int

    def to_json(self) -> dict[str, Any]:
        return {"touch_zones": self.touch_zones, "too_close_mm": self.too_close_mm}


@dataclass(frozen=True)
class Calibration:
    """One body's numbers. ``tof`` None: a duck without a ToF module (depth keeps the simulator's band, unused)."""

    board: str
    walk_sha256: str | None
    sitstand_sha256: str | None
    source: str  # "simulator" (the shipped numbers) | "bench" (measured on that body)
    proprio: LoadBands
    tof: TofBands | None
    recorded: str = ""
    unit: str = (
        ""  # the owner's own label for that one duck (a sticker, a name): for people, not matched
    )
    units: str = ""
    evidence: Mapping[str, Any] | None = None

    def matches(
        self, board: str | None, walk_sha256: str | None, sitstand_sha256: str | None
    ) -> bool:
        return (
            board is not None
            and walk_sha256 is not None
            and sitstand_sha256 is not None
            and (self.board, self.walk_sha256, self.sitstand_sha256)
            == (board, walk_sha256, sitstand_sha256)
        )

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema": SCHEMA,
            "board": self.board,
            "unit": self.unit,
            "walk_sha256": self.walk_sha256,
            "sitstand_sha256": self.sitstand_sha256,
            "source": self.source,
            "recorded": self.recorded,
            "units": self.units,
            "proprio": self.proprio.to_json(),
            "tof": None if self.tof is None else self.tof.to_json(),
        }
        if self.evidence is not None:
            out["evidence"] = dict(self.evidence)
        return out


def _number(table: Mapping[str, Any], key: str, where: str) -> float:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise CalibrationError(f"{where}.{key} is not a number")
    if value < 0:
        raise CalibrationError(f"{where}.{key} is negative")
    return float(value)


def _band(table: Mapping[str, Any], key: str, where: str) -> tuple[float, float]:
    value = table.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise CalibrationError(f"{where}.{key} is not a [low, high] pair")
    pair = {"low": value[0], "high": value[1]}
    low, high = _number(pair, "low", f"{where}.{key}"), _number(pair, "high", f"{where}.{key}")
    if not low < high:
        raise CalibrationError(f"{where}.{key}: low is not under high")
    return (low, high)


def _load_bands(table: Any) -> LoadBands:
    if not isinstance(table, Mapping):
        raise CalibrationError("proprio is missing")
    bands = LoadBands(
        unsure_load=_number(table, "unsure_load", "proprio"),
        unsure_load_sit=_number(table, "unsure_load_sit", "proprio"),
        unsure_load_reverse=_number(table, "unsure_load_reverse", "proprio"),
        lift_load=_number(table, "lift_load", "proprio"),
        lift_load_sit=_number(table, "lift_load_sit", "proprio"),
        flail_load=_number(table, "flail_load", "proprio"),
        floor_band=_band(table, "floor_band", "proprio"),
        floor_band_sit=_band(table, "floor_band_sit", "proprio"),
    )
    # the order the claims rely on: a hand's level < "unsure" < the floor's band (< the flailing level)
    if not 0.0 < bands.lift_load <= bands.unsure_load <= bands.floor_band[0]:
        raise CalibrationError("proprio: not 0 < lift_load <= unsure_load <= floor_band low")
    if bands.unsure_load_reverse > bands.unsure_load:
        raise CalibrationError("proprio: unsure_load_reverse is above unsure_load")
    if not bands.lift_load_sit <= bands.unsure_load_sit <= bands.floor_band_sit[0]:
        raise CalibrationError(
            "proprio: not lift_load_sit <= unsure_load_sit <= floor_band_sit low"
        )
    if bands.flail_load <= bands.floor_band[0]:
        raise CalibrationError("proprio: flail_load is not above the floor band's low edge")
    return bands


def _tof_bands(table: Any) -> TofBands | None:
    if table is None:
        return None
    if not isinstance(table, Mapping):
        raise CalibrationError("tof is neither a table nor null")
    zones, band = _number(table, "touch_zones", "tof"), _number(table, "too_close_mm", "tof")
    if zones != int(zones) or not 1 <= zones <= ZONES:
        raise CalibrationError(f"tof.touch_zones is not a whole number from 1 to {ZONES}")
    if band != int(band) or not 20 <= band <= 500:
        raise CalibrationError("tof.too_close_mm is not a whole number from 20 to 500")
    return TofBands(touch_zones=int(zones), too_close_mm=int(band))


def _sha(data: Mapping[str, Any], key: str, *, required: bool) -> str | None:
    value = data.get(key)
    if value is None and not required:
        return None
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise CalibrationError(f"{key} is not a sha256 (64 lowercase hex digits)")
    return value


def parse(data: Any) -> Calibration:
    """The document as a :class:`Calibration`, or :class:`CalibrationError`. A bench document names both networks; only
    the shipped simulator document (board ``simulator``) may leave them open."""
    if not isinstance(data, Mapping):
        raise CalibrationError("not a JSON object")
    if data.get("schema") != SCHEMA:
        raise CalibrationError(f"schema is {data.get('schema')!r}, this program reads {SCHEMA!r}")
    board, source = data.get("board"), data.get("source")
    if not isinstance(board, str) or not board:
        raise CalibrationError("board is missing")
    if source not in ("simulator", "bench"):
        raise CalibrationError("source is neither 'simulator' nor 'bench'")
    if (source == "simulator") != (board == SIMULATOR_BOARD):
        raise CalibrationError("only the board 'simulator' carries the simulator's numbers")
    named = source == "bench"
    evidence = data.get("evidence")
    return Calibration(
        board=board,
        walk_sha256=_sha(data, "walk_sha256", required=named),
        sitstand_sha256=_sha(data, "sitstand_sha256", required=named),
        source=str(source),
        proprio=_load_bands(data.get("proprio")),
        tof=_tof_bands(data.get("tof")),
        recorded=str(data.get("recorded") or ""),
        unit=str(data.get("unit") or ""),
        units=str(data.get("units") or ""),
        evidence=dict(evidence) if isinstance(evidence, Mapping) else None,
    )


def load(path: Path) -> Calibration:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise CalibrationError(f"cannot be read: {error}") from error
    return parse(data)


# The simulator's numbers (set on Pollen's v5 networks in the simulator, 2026-09: proprio's module doc; the loaded seat
# and stance then scale the seated and standing claims, :mod:`duckmind.neuro.seatcal`).
SIMULATOR = load(DATA)


def key_of(info: Mapping[str, Any] | None) -> tuple[str | None, str | None, str | None]:
    """(board, walk sha256, sit-stand sha256) of the body a bridge described at the handshake
    (:func:`duckmind.remotebody.bridge.real_body_info`): the board is its ``robotd.toml``'s ``[board] version``; a
    rehearsal (``sim-body --as-real``) has no ``robotd.toml`` and is the board :data:`REHEARSAL_BOARD`. None for what
    the bridge did not say."""
    info = info or {}
    config, policies = info.get("robotd_config"), info.get("policies")
    board = config.get("board") if isinstance(config, Mapping) else None
    if not isinstance(board, str) or not board:
        board = REHEARSAL_BOARD if info.get("rehearsal") else None
    walk = policies.get("walk_sha256") if isinstance(policies, Mapping) else None
    seat = policies.get("sitstand_sha256") if isinstance(policies, Mapping) else None
    return (
        board,
        walk if isinstance(walk, str) and SHA256.fullmatch(walk) else None,
        seat if isinstance(seat, str) and SHA256.fullmatch(seat) else None,
    )


def folder() -> Path:
    """Where this computer keeps the bodies' measured calibrations (``<duckmind home>/body-calibration``)."""
    from duckmind import paths

    return paths.duckmind_home() / "body-calibration"


def file_name(board: str, walk_sha256: str, sitstand_sha256: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", board)
    return f"{safe}-{walk_sha256[:12]}-{sitstand_sha256[:12]}.json"


@dataclass(frozen=True)
class Found:
    """What :func:`find` found for a body: its ``calibration``, or None and ``why`` (one of :data:`WHYS`); ``path`` the
    document it is about and ``detail`` an unreadable document's fault, for the console."""

    calibration: Calibration | None
    why: str = ""
    path: Path | None = None
    detail: str = ""


def find(
    board: str | None,
    walk_sha256: str | None,
    sitstand_sha256: str | None,
    where: Path | None = None,
) -> Found:
    """The measured calibration for that board running those networks among the documents in ``where`` (default
    :func:`folder`); the newest ``recorded`` when several match. Never the simulator's: a real body walks on its own
    numbers or not at all."""
    if board is None or walk_sha256 is None or sitstand_sha256 is None:
        return Found(None, WHY_UNKNOWN_BODY)
    where = folder() if where is None else where
    matches: list[tuple[str, str, Calibration, Path]] = []
    other: Path | None = None
    broken: tuple[Path, str] | None = None
    for path in sorted(where.glob("*.json")) if where.is_dir() else ():
        try:
            cal = load(path)
        except CalibrationError as error:
            # (a broken file counts against this body only when it is named as this body's)
            if path.name == file_name(board, walk_sha256, sitstand_sha256):
                broken = (path, str(error))
            continue
        if cal.source != "bench":
            continue
        if cal.matches(board, walk_sha256, sitstand_sha256):
            matches.append((cal.recorded, path.name, cal, path))
        elif cal.board == board:
            other = path
    if matches:
        _, _, cal, path = max(matches, key=lambda m: (m[0], m[1]))
        return Found(cal, path=path)
    if broken is not None:
        return Found(None, WHY_INVALID, broken[0], broken[1])
    if other is not None:
        return Found(None, WHY_OTHER, other)
    return Found(None, WHY_NONE)


def save(calibration: Calibration, where: Path | None = None) -> Path:
    """Write a bench calibration to ``where`` (default :func:`folder`) under its body's name; the document is parsed
    back first, so nothing unusable is ever written."""
    if calibration.source != "bench":
        raise CalibrationError("only a bench calibration is saved")
    data = calibration.to_json()
    parse(data)
    assert calibration.walk_sha256 is not None and calibration.sitstand_sha256 is not None
    where = folder() if where is None else where
    where.mkdir(parents=True, exist_ok=True)
    path = where / file_name(
        calibration.board, calibration.walk_sha256, calibration.sitstand_sha256
    )
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


__all__ = [
    "DATA",
    "REHEARSAL_BOARD",
    "SCHEMA",
    "SIMULATOR",
    "SIMULATOR_BOARD",
    "WHYS",
    "WHY_INVALID",
    "WHY_NONE",
    "WHY_OTHER",
    "WHY_UNKNOWN_BODY",
    "Calibration",
    "CalibrationError",
    "Found",
    "LoadBands",
    "TofBands",
    "file_name",
    "find",
    "folder",
    "key_of",
    "load",
    "parse",
    "save",
]
