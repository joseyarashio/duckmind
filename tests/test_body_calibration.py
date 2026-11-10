"""The body-sense calibration document (duckbody/calibration.py, docs/deploy/bring-up.md): the leg-load bands and the
ToF touch band are read from one document per body; the simulator's ship as the built-in one, so a simulated body
senses exactly as before; a real body's is found by its board and the two networks it runs."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import pytest
from test_duckbody_depth import frame, sensor
from test_duckbody_proprio import tick

from duckmind.duckbody import calibration as C
from duckmind.duckbody import depth as D
from duckmind.duckbody import tof_reproject as tr
from duckmind.duckbody.proprio import Proprio, ProprioConfig

WALK, SEAT, OTHER = "a" * 64, "b" * 64, "c" * 64
# the numbers proprio and depth carried as literals before the document existed (2026-11-10, main 8089fb4)
BEFORE = {
    "unsure_load": 24.0,
    "unsure_load_sit": 20.5,
    "unsure_load_reverse": 18.0,
    "lift_load": 20.0,
    "lift_load_sit": 18.5,
    "flail_load": 100.0,
    "floor_band": (28.0, 120.0),
    "floor_band_sit": (21.5, 70.0),
}


def bench(**changes: object) -> dict[str, object]:
    """A measured body's document: a board whose legs draw about 12× the simulator's proxy."""
    data: dict[str, object] = {
        "schema": C.SCHEMA,
        "board": "face-board-v1",
        "unit": "bench duck",
        "walk_sha256": WALK,
        "sitstand_sha256": SEAT,
        "source": "bench",
        "recorded": "2026-11-20T10:00:00Z",
        "units": "mA, ten leg servos",
        "proprio": {
            "unsure_load": 290.0,
            "unsure_load_sit": 250.0,
            "unsure_load_reverse": 215.0,
            "lift_load": 240.0,
            "lift_load_sit": 220.0,
            "flail_load": 1200.0,
            "floor_band": [335.0, 1440.0],
            "floor_band_sit": [260.0, 840.0],
        },
        "tof": {"touch_zones": 40, "too_close_mm": 150},
    }
    data.update(changes)
    return data


def test_the_simulators_numbers_ship_as_a_document_and_are_the_defaults() -> None:
    sim = C.SIMULATOR
    assert (sim.board, sim.source, sim.walk_sha256, sim.sitstand_sha256) == (
        "simulator",
        "simulator",
        None,
        None,
    )
    assert sim.proprio.to_json() == {
        k: list(v) if isinstance(v, tuple) else v for k, v in BEFORE.items()
    }
    assert sim.tof == C.TofBands(touch_zones=32, too_close_mm=100)
    cfg = ProprioConfig()
    assert {k: getattr(cfg, k) for k in BEFORE} == BEFORE
    assert (D.DEFAULT.touch_zones, D.DEFAULT.too_close_mm) == (32, 100)
    assert (
        D.DEFAULT.too_close_mm / 1000.0 == tr.MIN_RANGE_M
    )  # the Reprojector's own band, to the bit
    # applying the simulator's document to the defaults changes nothing
    assert cfg.calibrated(sim.proprio) == cfg and D.DEFAULT.calibrated(sim.tof) == D.DEFAULT


def test_a_measured_bodys_levels_replace_only_the_load_bands() -> None:
    cal = C.parse(bench())
    base = ProprioConfig(seat_settle_s=0.7, turn_dead_s=0.3)
    cfg = base.calibrated(cal.proprio)
    changed = {f.name for f in fields(cfg) if getattr(cfg, f.name) != getattr(base, f.name)}
    assert changed == set(BEFORE)
    assert cfg.lift_load == 240.0 and cfg.floor_band == (335.0, 1440.0)
    depth = D.DEFAULT.calibrated(cal.tof)
    assert (depth.touch_zones, depth.too_close_mm) == (40, 150)
    assert D.DEFAULT.calibrated(None) is D.DEFAULT  # a duck without a ToF


def test_a_duck_in_the_hand_is_claimed_at_its_own_bodys_level() -> None:
    """A board whose standing legs draw 420 and whose legs in the hand draw 110: the simulator's level (20) never
    sees the pickup; the measured one does, and does not claim it on the floor."""
    cal = C.parse(bench())

    def lifted(cfg: ProprioConfig, load: float) -> bool:
        proprio = Proprio(cfg)
        edges: list[str] = []
        for k in range(150):  # 3 s standing on the floor, then 3 s at ``load``
            edges += proprio.step(tick(k * 0.02, policy="stand", load=(210.0, 210.0))).edges
        for k in range(150, 300):
            edges += proprio.step(tick(k * 0.02, policy="stand", load=(load / 2, load / 2))).edges
        return "lifted" in edges

    assert not lifted(ProprioConfig(), 110.0)
    assert lifted(ProprioConfig().calibrated(cal.proprio), 110.0)
    assert not lifted(ProprioConfig().calibrated(cal.proprio), 420.0)


def test_the_touch_band_is_the_measured_sensors() -> None:
    """Every zone returning 12 cm: no touch in the simulator's 10 cm band, a touch for a sensor measured to read a
    pressed face out to 15 cm."""
    pose = sensor(0)
    mm, st = [120] * 64, [5] * 64
    plain = D.DepthTracker(standing_height_m=0.125)
    reading, _ = plain.step(frame(0, mm, st), D.SensorPose(pose))
    assert not reading.touch and reading.too_close == 0
    cal = C.parse(bench(tof={"touch_zones": 40, "too_close_mm": 150}))
    measured = D.DepthTracker(standing_height_m=0.125, config=D.DEFAULT.calibrated(cal.tof))
    reading, _ = measured.step(frame(0, mm, st), D.SensorPose(pose))
    assert reading.touch and reading.too_close >= 40
    few = C.parse(bench(tof={"touch_zones": 64, "too_close_mm": 150}))
    strict = D.DepthTracker(standing_height_m=0.125, config=D.DEFAULT.calibrated(few.tof))
    part = [120] * 40 + [900] * 24
    reading, _ = strict.step(frame(0, part, st), D.SensorPose(pose))
    assert not reading.touch  # (40 of 64 inside the band: under this sensor's 64)


@pytest.mark.parametrize(
    ("change", "fault"),
    [
        ({"schema": "duckmind.body-calibration/2"}, "schema"),
        ({"walk_sha256": None}, "walk_sha256"),
        ({"sitstand_sha256": "xyz"}, "sitstand_sha256"),
        ({"board": ""}, "board"),
        ({"board": "simulator"}, "simulator"),
        ({"source": "guess"}, "source"),
        ({"proprio": None}, "proprio"),
        ({"tof": {"touch_zones": 0, "too_close_mm": 100}}, "touch_zones"),
        ({"tof": {"touch_zones": 32, "too_close_mm": 5}}, "too_close_mm"),
    ],
)
def test_a_document_that_cannot_be_used_is_refused_with_its_fault(
    change: dict[str, object], fault: str
) -> None:
    with pytest.raises(C.CalibrationError, match=fault):
        C.parse(bench(**change))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("lift_load", 300.0),  # above "unsure": the claim would come before the doubt
        ("lift_load", 0.0),  # a standing claim that can never be made
        ("unsure_load", 400.0),  # above the floor band: a duck on the floor would be "unsure"
        ("unsure_load_reverse", 300.0),
        ("flail_load", 300.0),  # inside the floor's own loads
        ("floor_band", [500.0, 400.0]),
        ("lift_load_sit", float("nan")),
        ("unsure_load", "24"),
    ],
)
def test_load_bands_out_of_order_are_refused(key: str, value: object) -> None:
    data = bench()
    data["proprio"] = {**data["proprio"], key: value}  # type: ignore[dict-item]
    with pytest.raises(C.CalibrationError):
        C.parse(json.loads(json.dumps(data)))


def test_a_body_is_keyed_by_its_board_and_its_two_networks() -> None:
    real = {
        "backend": "real",
        "policies": {"set": "v8", "walk_sha256": WALK, "sitstand_sha256": SEAT},
        "robotd_config": {"board": "face-board-v1", "idle": False},
    }
    assert C.key_of(real) == ("face-board-v1", WALK, SEAT)
    rehearsal = {"backend": "real", "rehearsal": True, "policies": real["policies"]}
    assert C.key_of(rehearsal) == (C.REHEARSAL_BOARD, WALK, SEAT)
    # what the bridge did not say stays unknown (and matches nothing)
    assert C.key_of({"backend": "real", "policies": real["policies"]}) == (None, WALK, SEAT)
    assert C.key_of({"backend": "real", "robotd_config": {"board": "zero3"}}) == (
        "zero3",
        None,
        None,
    )
    assert C.key_of(None) == (None, None, None)


def test_find_gives_the_newest_measurement_for_that_board_and_networks_or_says_why(
    tmp_path: Path,
) -> None:
    key = ("face-board-v1", WALK, SEAT)
    assert C.find(*key, where=tmp_path / "absent") == C.Found(None, C.WHY_NONE)
    assert C.find(None, WALK, SEAT, where=tmp_path).why == C.WHY_UNKNOWN_BODY
    assert C.find("face-board-v1", None, SEAT, where=tmp_path).why == C.WHY_UNKNOWN_BODY
    # measured with other networks (the duck was updated since): not this body's
    old = C.save(C.parse(bench(walk_sha256=OTHER)), tmp_path)
    found = C.find(*key, where=tmp_path)
    assert (found.calibration, found.why, found.path) == (None, C.WHY_OTHER, old)
    # another board's document says nothing about this one
    assert C.find("zero3", WALK, SEAT, where=tmp_path).why == C.WHY_NONE
    # this body's file, broken
    mine = tmp_path / C.file_name(*key)
    mine.write_text("{not json", encoding="utf-8")
    broken = C.find(*key, where=tmp_path)
    assert broken.why == C.WHY_INVALID and broken.path == mine and broken.detail
    mine.write_text(json.dumps(bench(schema="duckmind.body-calibration/0")), encoding="utf-8")
    assert C.find(*key, where=tmp_path).why == C.WHY_INVALID
    # measured: found, and the newest wins
    path = C.save(C.parse(bench()), tmp_path)
    assert path == mine and C.load(path) == C.parse(bench())
    (tmp_path / "later.json").write_text(
        json.dumps(bench(recorded="2026-12-01T09:00:00Z", unit="re-measured")), encoding="utf-8"
    )
    found = C.find(*key, where=tmp_path)
    assert (
        found.calibration is not None and found.calibration.unit == "re-measured" and not found.why
    )
    # the simulator's numbers are never a real body's, whatever a file claims
    (tmp_path / "later.json").unlink()
    mine.unlink()
    (tmp_path / "sim.json").write_text(C.DATA.read_text(encoding="utf-8"), encoding="utf-8")
    assert C.find(*key, where=tmp_path).why == C.WHY_OTHER
    with pytest.raises(C.CalibrationError):
        C.save(C.SIMULATOR, tmp_path)


def test_the_measured_calibrations_live_under_the_duckmind_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DUCKMIND_HOME", str(tmp_path))
    assert C.folder() == tmp_path / "body-calibration"
    C.save(C.parse(bench()))
    assert C.find("face-board-v1", WALK, SEAT).calibration == C.parse(bench())
