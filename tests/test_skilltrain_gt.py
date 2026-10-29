"""tr:gate: the verdict rules and the delayed-servo runner option (pure; no simulator)."""

from __future__ import annotations

from typing import Any

import pytest

from duckmind.skilltrain import gt_gate as GG
from duckmind.skilltrain import gt_verdict as V
from duckmind.skilltrain import metrics as M


def _metrics(**over: Any) -> dict[str, Any]:
    m: dict[str, Any] = {
        "fell": False,
        "fell_in_skill": False,
        "handback_fall": False,
        "t45": False,
        "righted_8s": None,
        "tau_leg": [0.1] * 60,
        "tau_head": [0.1] * 60,
        "acc": [50.0] * 60,
        "joint_ticks": 840,
        "stall_ticks": 0,
        "strain_ticks": 0,
        "servo_hits_upright": 0,
        "back_070_cm": 0.0,
        "yaw_change_deg": 0.0,
        "planted_end": True,
        "still_s": 0.3,
        "travel_cm": 2.0,
    }
    m.update(over)
    return m


def _row(key: str, family: str, posture: str, performed: bool | None, **metrics: Any) -> dict:
    return {
        "key": f"{key}/mjcf/0",
        "cond": {"key": key, "family": family, "posture": posture, "level": 1.0},
        "metrics": _metrics(**metrics),
        "visibility": {"performed": performed, "checks": {}},
    }


def test_delay_options() -> None:
    assert GG.delay_steps("none", 1) == 0
    assert GG.delay_steps("4", 1) == 4
    assert all(3 <= GG.delay_steps("dr", s) <= 6 for s in range(40))
    with pytest.raises(ValueError):
        GG.delay_steps("-1", 0)


def test_hop_back_from_walking_is_judged_by_the_hop_checks() -> None:
    # the harness's walking-stop checks said no; the hop checks say yes
    row = _row("skill/hop_back/walk0.40", "hop_back", "walk", False,
               back_070_cm=9.0, yaw_change_deg=5.0, planted_end=True)  # fmt: skip
    assert V.performed(row) is True
    turned = _row("skill/hop_back/walk0.40", "hop_back", "walk", False,
                  back_070_cm=9.0, yaw_change_deg=30.0, planted_end=True)  # fmt: skip
    assert V.performed(turned) is False
    # every other family keeps the harness's own verdict
    assert V.performed(_row("skill/startle_1/walk0.40", "startle", "walk", False)) is False


def _cell(k: int, n: int, falls: int = 0) -> dict[str, Any]:
    return {"n": n, "performed": M.rate(k, n), "falls": M.rate(falls, n)}


def test_cell_status() -> None:
    assert V.status(_cell(27, 30)) == "ok"
    assert V.status(_cell(30, 30, falls=1)) == "ok"
    assert V.status(_cell(30, 30, falls=2)) == "weak"
    assert V.status(_cell(22, 30)) == "weak"
    assert V.status(_cell(20, 30)) == "miss"
    assert V.status(_cell(0, 30, falls=24)) == "falls"
    assert V.status(None) == "none"


def test_judge_limits_a_skill_to_the_postures_that_passed() -> None:
    entry = {"name": "hop_back", "family": "hop_back", "postures": ["stand", "walk"]}
    stand, walk = "skill/hop_back/stand", "skill/hop_back/walk0.40"
    cols = {
        "sim_delay": {stand: _cell(30, 30), walk: _cell(26, 30)},
        "mjcf_dr_d": {stand: _cell(24, 30), walk: _cell(16, 30)},
        "bam_dr": {stand: _cell(27, 30), walk: _cell(17, 30, falls=2)},
        "sim_today": {stand: _cell(0, 30, falls=30), walk: _cell(0, 30, falls=30)},
        "mjcf_dr_0": {stand: _cell(1, 30, falls=24), walk: _cell(0, 30, falls=26)},
    }
    d = V.judge(entry, cols, "delay")
    assert d["verdict"] == "pass_with_limits" and d["allowed"] == ["stand"]
    assert V.judge(entry, cols, "today")["verdict"] == "fail"
    good = {c: {stand: _cell(30, 30), walk: _cell(30, 30)} for c in cols}
    assert V.judge(entry, good, "delay")["verdict"] == "pass"


def test_caps_halve_on_marginal_strain() -> None:
    from duckmind.skilltrain import guard as G

    entry = {"name": "shake_2", "family": "shake", "postures": ["stand"]}
    strain = {
        "tau_leg_p95": 0.28,
        "tau_head_p95": 0.31,
        "stall_frac": 0.0,
        "acc_p99": 150.0,
        "servo_hits_upright_nofall": M.rate(0, 30),
    }
    cols = {"bam_dr": {"skill/shake_2/stand": {"n": 30, "strain": strain}}}
    caps = V.caps([entry], cols, G.DEFAULT_RULES)
    assert caps["shake_2"]["per_hour"] == (G.DEFAULT_RULES["shake_2"].per_hour or 0) // 2
    strain["tau_head_p95"] = 0.2
    assert V.caps([entry], cols, G.DEFAULT_RULES)["shake_2"]["per_hour"] == 4
