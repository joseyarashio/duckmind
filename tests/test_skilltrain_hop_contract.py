"""Forward hop-over contract (hop:feasibility, 2026-09-28): the table entries, the command encoding,
the strip shapes and heights, the physics bound's arithmetic — pure (standard library + numpy)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from duckmind.skilltrain import hop_ceiling as HC
from duckmind.skilltrain import hop_contract as H
from duckmind.skilltrain import table as T


def test_four_entries_walk_like_commands_one_per_height_class() -> None:
    vs = H.variants()
    assert [v.name for v in vs] == ["hop_over_1", "hop_over_2", "hop_over_3", "hop_over_4"]
    assert [v.command for v in vs] == [
        (0.35, 0.0, 0.0),
        (0.40, 0.0, 0.0),
        (0.45, 0.0, 0.0),
        (0.50, 0.0, 0.0),
    ]
    for v in vs:
        assert v.unwind == (0.0, 0.0, 0.0)
        assert (v.hold_s, v.unwind_s) == (H.HOLD_S, H.UNWIND_S)
        assert v.postures == ("stand", "walk")
        assert v.action_scale == pytest.approx(0.9)  # velstand's walk-slot scale (the warm start's)
    # commands stay inside the gait envelope the v5 walk was trained on (vx 0.35–0.55)
    assert all(0.30 <= v.command[0] <= 0.55 for v in vs)


def test_height_class_rounds_the_estimate_up_to_whole_cm() -> None:
    assert H.height_class(0.004) == 1
    assert H.height_class(0.010) == 1
    assert H.height_class(0.0101) == 2
    assert H.height_class(0.030) == 3
    assert H.height_class(0.035) == 4
    assert H.height_class(0.090) == 4  # the highest class; the brain does not try taller ones


def test_skill_defs_are_valid_robotd_entries(tmp_path) -> None:
    onnx = tmp_path / "hopover.onnx"
    onnx.write_bytes(b"")
    defs = [T.SkillDef(**d) for d in H.skill_defs(str(onnx))]
    assert [d.name for d in defs] == [v.name for v in H.variants()]
    text = T.toml_section(defs)
    assert text.count("[[policy.skill]]") == 4
    assert "chain = false" in text and "unwind_s = 0.5" in text
    problems = T.problems(defs)
    assert problems == [], problems


def test_curriculum_rows_are_half_centimetre_steps() -> None:
    rows = [round(H.height_of_row(r) * 1000, 3) for r in range(H.TERRAIN_ROWS)]
    assert rows == [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
    assert H.round_radius(0.005, 0.04) == pytest.approx(0.005)
    assert H.round_radius(0.05, 0.04) == pytest.approx(0.02)


def test_a_true_hop_over_a_strip_needs_far_more_than_the_body_has() -> None:
    need = H.hop_over_needs(0.30)  # the walk's fastest forward speed, a 4 cm strip + one sole
    assert need["flight_ms"] == pytest.approx(313.3, abs=0.5)
    assert need["com_rise_mm"] > 100
    best = max(v["com_rise_mm"] for v in H.CEILING.values())
    assert need["com_rise_mm"] / best > 5


def test_servo_curves_brake_above_their_no_load_speed() -> None:
    w = np.array([0.0, 5.0, 10.0, 20.0, 25.0])
    bam = HC.servo_torque("bam", w, 7.4)
    xml = HC.servo_torque("mjcf", w)
    # at stall: the 1.75 A firmware limit (0.641 N·m) less the gearbox's load friction
    assert bam[0] == pytest.approx(
        HC.KT * HC.I_MAX * (1 - HC.F_LOAD_MOTOR) - HC.F_BASE - HC.F_STRIBECK, abs=1e-3
    )
    assert xml[0] == pytest.approx(HC.XML_FORCE - HC.XML_FRICTION)
    assert np.all(np.diff(bam) <= 1e-12) and np.all(np.diff(xml) < 0)
    assert bam[-1] < 0 and xml[-1] < 0  # beyond the no-load speed the motor brakes
    no_load = 7.4 / HC.KT
    assert 19.5 < no_load < 20.5


def test_wilson_matches_the_stage1_helper() -> None:
    from duckmind.skilltrain import rl_contract as C

    for k, n in ((0, 30), (3, 30), (30, 30), (14, 90)):
        assert H.wilson(k, n) == pytest.approx(C.wilson(k, n))
    lo, hi = H.wilson(0, 30)
    assert lo == 0.0 and math.isclose(hi, 0.1135, abs_tol=5e-4)
