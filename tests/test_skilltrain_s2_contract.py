"""Stage-2 numbers (:mod:`duckmind.skilltrain.s2_contract`): consistency, the gates the task names,
the verdict rules (Wilson 95 %), the envelope and the unit estimate. Pure standard library."""

from __future__ import annotations

import math

import pytest

from duckmind.skilltrain import s2_contract as C


def test_the_contract_is_internally_consistent() -> None:
    assert C._check() == []
    assert C.TASK_IDS == {
        "pose": "Mjlab-DuckWalkPose-Flat-MicroDuck",
        "gaze": "Mjlab-DuckWalkGaze-Flat-MicroDuck",
    }
    assert (C.BODY_Z_SLOT, C.BODY_ROLL_SLOT, C.BODY_PITCH_SLOT) == (
        57,
        58,
        59,
    )  # duck-control obs.rs
    assert C.WALK_ACTION_SCALE == 0.9 and C.GAIN_RATIO == 1.0 and C.CMD_ALPHA == 0.2


def test_every_gate_the_task_names_exists() -> None:
    keys = set(C.GATE_BY_KEY)
    visibility = {"lean_roll_deg", "lean_pitch_deg", "sway_amp_deg", "sway_lag_ms", "weight_shift_ratio",
                  "chest_out_deg", "crouch_mm", "neck_pullback_gain", "yaw_micro_gain", "cam_perp_ratio",
                  "head_yaw_stride_sd_ratio", "head_yaw_delivered", "release_rate_dps", "pose_stepped"}  # fmt: skip
    regression = {"walk_speed_ratio", "walk_falls", "stopgo_near_falls", "push_righting_8s", "righting_p50_extra_s",
                  "sitstand_falls", "turn_gain_ratio", "stop_s_p50", "cam_perp_regress_ratio", "walkoffs",
                  "stand_leg_rms_ratio", "head_motion_falls"}  # fmt: skip
    assert visibility <= keys and regression <= keys
    pose = {g.key for g in C.gates_for("pose")}
    gaze = {g.key for g in C.gates_for("gaze")}
    assert "cam_perp_ratio" not in pose and "cam_perp_regress_ratio" in pose
    assert "cam_perp_ratio" in gaze and "lean_roll_deg" in gaze  # R4 keeps R3's standing abilities
    for g in C.GATES:
        assert g.zh and g.kind in (
            "min",
            "max",
            "rate_max",
            "rate_min",
            "ratio_max",
            "within",
            "max_vs_v5",
            "report",
        )


def test_plan_thresholds() -> None:
    g = C.GATE_BY_KEY
    assert g["lean_roll_deg"].threshold == 7.0 and g["chest_out_deg"].threshold == 6.0
    assert g["crouch_mm"].threshold == 12.0 and g["weight_shift_ratio"].threshold == 0.65
    assert g["sway_lag_ms"].threshold == 300.0 and g["release_rate_dps"].threshold == 45.0
    assert g["stopgo_near_falls"].threshold == 0.003 and g["walk_falls"].threshold == 0.001
    assert g["sitstand_falls"].threshold == 0.018 and g["push_righting_8s"].threshold == 0.75
    # lowered from the plan's 0.50 by the P0 probe; the stretch target stays documented
    assert g["cam_perp_ratio"].threshold == 0.60 and "0.50" in g["cam_perp_ratio"].note


def test_verdict_rules() -> None:
    g = C.GATE_BY_KEY
    assert (
        C.verdict(g["lean_roll_deg"], 7.2) == "PASS"
        and C.verdict(g["lean_roll_deg"], 6.9) == "FAIL"
    )
    assert C.verdict(g["lean_roll_deg"], None) == "NOT MEASURED"
    assert C.verdict(g["sway_lag_ms"], 250.0) == "PASS"
    # rates at Wilson 95 %: 0/2048 passes ≤ 0.3 %, 0/30 cannot resolve it, 20/2048 fails
    assert C.verdict(g["stopgo_near_falls"], (0, 2048)) == "PASS"
    assert C.verdict(g["stopgo_near_falls"], (0, 30)) == "UNRESOLVED"
    assert C.verdict(g["stopgo_near_falls"], (20, 2048)) == "FAIL"
    assert C.verdict(g["push_righting_8s"], (360, 414)) == "PASS"
    assert (
        C.verdict(g["walk_speed_ratio"], 1.08) == "PASS"
        and C.verdict(g["walk_speed_ratio"], 0.88) == "FAIL"
    )
    assert C.verdict(g["cam_perp_ratio"], 0.55) == "PASS"
    assert C.verdict(g["walkoffs"], (1, 512)) == "NO BASELINE"
    assert C.verdict(g["walkoffs"], (1, 512), v5=(2, 512)) == "PASS"
    assert C.verdict(g["walkoffs"], (60, 512), v5=(2, 512)) == "FAIL"
    assert C.verdict(g["neck_travel_rad_min"], 50.0) == "REPORT"


def test_n_for_zero_matches_the_plan_sizes() -> None:
    assert C.n_for_zero(0.003) <= 2048  # "≤ 0.3 % at n ≥ 2,048" is resolvable with 0 events
    assert C.n_for_zero(0.001) > 2048  # walking falls ≤ 0.10 % needs the pooled programs


def test_pose_envelope() -> None:
    assert C.POSE.clamp(-0.05, 0.3, -0.3) == (-0.025, 0.2, -0.2)
    assert C.POSE.clamp(0.02, 0.0, 0.0)[0] == 0.008
    assert C.POSE.rate_rad_s == 0.5 and C.POSE.rate_m_s == 0.060


def test_bc_specs() -> None:
    pose, gaze = C.BC["pose"], C.BC["gaze"]
    assert (
        pose.anchor == "v5" and pose.anchor_dims == tuple(range(14)) and pose.coef_at(2000) == 1.0
    )
    assert gaze.anchor == "start" and gaze.anchor_dims == C.LEG_IDX and len(gaze.anchor_dims) == 10
    assert gaze.coef_at(0) == 1.0 and gaze.coef_at(1499) == 0.45 and gaze.coef_at(1500) == 0.3
    assert (
        pose.fallen_tilt_deg == 35.0
        and pose.anchor_tilt_deg == 25.0
        and pose.anchor_min_twist == 0.1
    )


def test_the_delay_range_covers_the_owners_choices() -> None:
    lo, hi = C.DELAY_STEPS
    assert lo * 5 == 0 and hi * 5 == 40  # 0 / 20 / 40 ms all inside
    assert C.DELAY_JITTER_STEPS == (3, 6)  # the recipe's 15–30 ms


def test_the_unit_estimate_fits_the_cap() -> None:
    est = C.estimate()
    likely = next(r for r in est["rows"] if r["case"] == "likely")
    assert est["fits_cap"] and likely["units"] <= C.STAGE_CAP_UNITS
    slow = C.estimate(s_per_it={"pose": 6.0, "gaze": 6.0})
    assert next(r for r in slow["rows"] if r["case"] == "likely")["units"] > likely["units"]
    assert math.isclose(C.units(60.0), C.UNITS_PER_HOUR)


@pytest.mark.parametrize("stage", C.STAGES)
def test_program_mix(stage: str) -> None:
    mix = C.PROGRAM_MIX[stage]
    assert set(mix) == set(C.PROGRAMS) and abs(sum(mix.values()) - 1.0) < 1e-9
    if stage == "pose":
        assert mix["stand"] + mix["pose"] >= 0.5 and mix["stopgo"] == 0.20
    else:
        assert mix["walk_neuro"] >= 0.4
