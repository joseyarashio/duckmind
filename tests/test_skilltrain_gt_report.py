"""s1b:fix honest reporting (:mod:`duckmind.skilltrain.gt_report`) and the latency map's pure
helpers (:mod:`duckmind.skilltrain.gt_latency`), on synthetic rows (no physics)."""

from __future__ import annotations

from typing import Any

import pytest

from duckmind.skilltrain import gt_latency as L
from duckmind.skilltrain import gt_report as R


def row(key: str, *, family: str = "startle", posture: str = "stand", level: float = 0.5,
        fell: bool = False, kind: str = "skill", speed: float = 0.0, righted: bool | None = None,
        dip: float = 16.0, neck: float = 0.30, **extra: Any) -> dict[str, Any]:  # fmt: skip
    metrics = {
        "fell": fell,
        "fell_in_skill": fell,
        "righted_8s": righted,
        "righted_s": 1.2 if righted else None,
        "first_motion_s": 0.05,
        "retract_025_rad": neck,
        "dip_030_mm": dip,
        "hold_dev_mm": 0.8,
        "hold_footfalls_per_2s": 0.0,
        "still_s": 0.4,
        "travel_cm": 4.0,
        "yaw_change_deg": 5.0,
        "tau_leg": [0.2, 0.25],
        "tau_head": [0.1, 0.1],
        "acc": [100.0, 120.0],
        "stall_ticks": 0,
        "joint_ticks": 28,
        **extra,
    }
    return {
        "cond": {
            "key": key,
            "kind": kind,
            "family": family,
            "posture": posture,
            "level": level,
            "speed": speed,
            "skill": key.split("/")[1],
        },  # fmt: skip
        "metrics": metrics,
        "excluded": None,
    }


def test_pooled_fall_gates_say_fail_when_it_fails() -> None:
    rows = [row("skill/startle_2/stand", fell=(i < 3)) for i in range(300)]
    rows += [row("skill/startle_4/stand", level=1.0) for _ in range(800)]
    g = R.pooled_gates(rows)
    assert g["low"]["k"] == 3 and g["low"]["verdict"] == "FAIL"  # 1 % against ≤ 0.2 %
    assert g["high"]["k"] == 0 and g["high"]["verdict"] == "PASS"  # 0/800 ≤ 0.5 % (needs 765)
    # only the allowed postures count when asked
    walk = [row("skill/freeze_4/walk", family="freeze", posture="walk", speed=0.55, fell=True)]
    assert R.pooled_gates(walk, {"freeze_4": ["stand", "walk0.40"]})["high"]["n"] == 0
    assert R.pooled_gates(walk)["high"]["k"] == 1


def test_a_nominal_duck_is_one_duck() -> None:
    rows = [row("skill/hop_back/stand", family="hop_back", fell=True) for _ in range(30)]
    rows += [row("skill/skill_abort/stand", family="abort") for _ in range(30)]
    n = R.nominal(rows)
    assert n["ducks"] == 1 and n["conditions"] == 2 and n["conditions_with_a_fall"] == 1
    assert n["per_condition"]["skill/hop_back/stand"]["outcome"] == "fell"
    assert n["per_condition"]["skill/skill_abort/stand"]["outcome"] == "no visibility gate"
    assert all("wilson95" not in v for v in n["per_condition"].values())  # no interval


def test_a_walking_startle_beside_v5s_stop_and_the_labels() -> None:
    rows = [row("skill/startle_2/walk0.40", posture="walk", speed=0.40) for _ in range(10)]
    rows += [row("v5/stop/walk0.40", kind="v5", posture="walk", speed=0.40, level=1.0, dip=4.0,
                 neck=0.02) for _ in range(10)]  # fmt: skip
    w = R.walking_vs_v5(rows)
    assert len(w) == 1 and w[0]["skill"]["performed"]["k"] == 10
    assert w[0]["v5_stop"]["performed"]["k"] == 0  # today's stop fails the walking gate now
    assert R.strain_label("pass", "over").startswith("motion pass / strain FAIL")
    assert R.strain_label("pass", "ok") == "motion pass"
    assert (
        R.strain_label("pass_with_limits", "marginal")
        == "motion pass with limits / strain marginal"
    )


def test_righting_is_reported() -> None:
    rows = [row("skill/hop_back/stand", family="hop_back", fell=True, righted=(i < 4))
            for i in range(5)]  # fmt: skip
    r = R.righting(rows)
    assert r["falls"] == 5 and r["righted_8s"]["k"] == 4 and r["not_righted"]


def test_latency_map_helpers() -> None:
    assert L.steps_of(20) == 4 and L.steps_of(0) == 0
    with pytest.raises(ValueError):
        L.steps_of(7)
    assert L.label_of("fixed", 4, 4) == "d20" and L.label_of("jitter", 3, 6) == "j3-6"
    ok = {"n": 30, "performed": {"k": 28, "n": 30}, "falls": {"k": 0, "n": 30}}
    weak = {"n": 30, "performed": {"k": 22, "n": 30}, "falls": {"k": 1, "n": 30}}
    falls = {"n": 30, "performed": {"k": 1, "n": 30}, "falls": {"k": 24, "n": 30}}
    assert [L.status(c) for c in (ok, weak, falls, None)] == ["ok", "weak", "falls", "none"]
    assert L._span([15.0, 20.0, 25.0, 40.0]) == [15.0, 25.0]
    assert L._span([]) is None
