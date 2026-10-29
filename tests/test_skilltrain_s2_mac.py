"""The Stage-2 Mac harness (:mod:`duckmind.skilltrain.s2_mac`): delay labels, the stride-band and sine
fits, the slow heading frame, the condition sets, the gate verdicts from a summary; and one physics
trial when the official MJCF, the v5 policies and MuJoCo / onnxruntime are present."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

from duckmind import paths
from duckmind.skilltrain import s2_contract as C

S = pytest.importorskip("duckmind.skilltrain.s2_mac")


def test_delay_labels() -> None:
    assert S.delay_of("d0") == ("fixed", 0, 0)
    assert S.delay_of("d20") == ("fixed", 4, 4)
    assert S.delay_of("d40") == ("fixed", 8, 8)
    assert S.delay_of("j3-6") == ("jitter", 3, 6)
    with pytest.raises(ValueError):
        S.delay_of("d12")


def test_highpass_keeps_the_stride_and_drops_the_drift() -> None:
    t = np.arange(0, 9, 0.02)
    stride = 0.05 * np.sin(2 * math.pi * 2.0 * t)
    drift = 0.3 * t  # a slow turn
    sd = S.highpass_sd(stride + drift)
    assert abs(sd - 0.05 / math.sqrt(2)) < 0.01
    assert S.highpass_sd(drift) < 0.002


def test_lsq_sine_recovers_amplitude_and_lag() -> None:
    t = np.arange(0, 10, 0.02)
    y = 0.12 * np.sin(2 * math.pi * 0.25 * (t - 0.2)) + 0.03
    amp, lag = S.lsq_sine(y, t, 0.25)
    assert abs(amp - 0.12) < 1e-6 and abs(lag - 0.2) < 1e-4


def test_the_heading_frame_follows_an_arc_and_filters_the_wobble() -> None:
    ref = S.HeadingRef(0.0)
    errs = []
    for k in range(600):
        t = k * 0.02
        yaw = 0.4 * t + math.radians(3) * math.sin(2 * math.pi * 2.0 * t)
        r = ref.step(yaw)
        if t > 6:
            errs.append(math.atan2(math.sin(yaw - r), math.cos(yaw - r)))
    arr = np.array(errs)
    assert abs(arr.mean()) < math.radians(1.0)  # no steady-state lag in a steady arc
    assert arr.std() > math.radians(1.5)  # the stride wobble stays out of the reference


def test_condition_sets() -> None:
    gates = S.condition_set("gates")
    keys = {c.key for c in gates}
    for need in ("lean/roll+0.2r", "lean/chest-0.15", "lean/ws_roll+0.15", "lean/z-20mm", "sway/roll0.2@0.1Hz",
                 "head/pullback", "head/micro", "walk/0.40/home", "walk/0.55/home", "stopgo/0.40/pose"):  # fmt: skip
        assert need in keys
    oracle = S.condition_set("oracle")
    assert all(c.v5_only for c in oracle if c.key.startswith("oracle/"))
    assert {c.key for c in S.condition_set("all")} >= keys | {
        c.key for c in S.regression_conditions()
    }


def _cell(**nums: float) -> dict:
    return {k: {"n": 30, "p50": v, "p10": v, "p90": v} for k, v in nums.items()}


def test_verdicts_from_a_summary() -> None:
    cand, base = "mjcf/d0/cand", "mjcf/d0/v5"
    cells = {
        "lean/roll+0.2r": {cand: _cell(achieved_abs=8.0)},
        "lean/roll-0.2r": {cand: _cell(achieved_abs=7.5)},
        "lean/z-20mm": {cand: _cell(achieved=-13.0)},
        "lean/chest-0.15": {cand: _cell(achieved=-6.5)},
        "walk/0.40/home": {
            cand: _cell(cam_perp_dps=40.0, speed=0.18),
            base: _cell(cam_perp_dps=80.0, speed=0.17),
        },
        "walk/0.55/home": {
            cand: _cell(cam_perp_dps=60.0, speed=0.23),
            base: _cell(cam_perp_dps=110.0, speed=0.24),
        },
        "stopgo/0.40": {cand: {"n": 30, "near_fall": {"k": 0, "n": 30}}},
    }
    v = S.verdicts({"cells": cells}, "gaze", "cand", "mjcf", "d0")
    assert v["lean_roll_deg"]["verdict"] == "PASS" and v["lean_roll_deg"]["value"] == 7.5
    assert v["crouch_mm"]["verdict"] == "PASS" and v["chest_out_deg"]["verdict"] == "PASS"
    assert (
        v["cam_perp_ratio"]["value"] == pytest.approx(0.545, abs=1e-3)
        and v["cam_perp_ratio"]["verdict"] == "PASS"
    )
    assert v["walk_speed_ratio"]["verdict"] == "PASS"
    assert v["stopgo_near_falls"]["verdict"] == "UNRESOLVED"  # n = 30 cannot resolve ≤ 0.3 %
    assert v["neck_pullback_gain"]["verdict"] == "NOT MEASURED"


def _physics_ready() -> bool:
    try:
        import mujoco  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    mjcf = os.environ.get("DUCKMIND_MICRODUCK_MJCF")
    pol = paths.policy_dir()
    return bool(mjcf) and Path(mjcf).is_dir() and (pol / "velstand.onnx").is_file()


@pytest.mark.skipif(
    not _physics_ready(), reason="needs MuJoCo, onnxruntime, the official MJCF and the v5 policies"
)
def test_one_trial_on_the_xml_servos_with_a_20ms_delay() -> None:
    cond = next(c for c in S.condition_set("gates") if c.key == "head/micro")
    row = S.run_trial(cond, servo="mjcf", delay="d20", rep=0, walk=None)
    assert row["excluded"] is None, row["excluded"]
    assert row["duck"]["delay"]["model"] == ["fixed", 4, 4]
    m = row["metrics"]
    assert set(m) >= {"gain", "lag_ms", "fell"} and not m["fell"]
    assert 0.0 < m["gain"] < 2.0
    assert C.GATE_BY_KEY["yaw_micro_gain"].threshold == 0.6
