"""b:gate (2026-09-28): the full B gate's cells and scoring (no physics)."""

from __future__ import annotations

from typing import Any

from duckmind.skilltrain import bsk_contract as B
from duckmind.skilltrain import bsk_gatex as X


def _cell(name: str, entry: str = "stand", **kw: Any) -> dict[str, Any]:
    return {"set": kw.pop("set", "trained"), "name": name, "entry": entry, "height_mm": kw.pop("h", 0),
            "soft": kw.pop("soft", False), "aim_deg": kw.pop("aim", None)}  # fmt: skip


def _turn(**kw: Any) -> dict[str, Any]:
    r = {"cell": _cell("turn_left"), "dyaw_deg": 290.0, "rate_last2_dps": 72.0, "excursion_m": 0.03,
         "footfalls": 17, "fell": False, "fell_any": False, "refused": False, "leg_tau_p95": 0.24}  # fmt: skip
    r.update(kw)
    return r


def test_cells_cover_the_owner_gate_list() -> None:
    keys = {c.key for c in X.trained_cells()}
    for h in (30, 40, 50):
        for soft in ("", "-soft"):
            assert f"trained/step_up/stand/{h}mm{soft}" in keys
            assert f"trained/step_down/stand/{h}mm{soft}" in keys
    for a in ("0", "15", "25"):
        assert f"trained/step_up/stand/40mm/aim{a}" in keys
    for n in ("turn_left", "turn_right", "march", "bow", "lie_low"):
        assert {f"trained/{n}/stand", f"trained/{n}/walk"} <= keys
    today = {c.key for c in X.today_cells()}
    assert {
        "today/pivot/stand",
        "today/arc/stand",
        "today/ground_pick/stand",
        "today/sit_head_down/stand",
    } <= today
    assert len(keys) == len(X.trained_cells()) and len(today) == len(X.today_cells())  # unique keys


def test_every_cell_name_maps_to_a_contract_gate() -> None:
    for c in X.trained_cells() + X.today_cells():
        assert c.behaviour in (*B.GATES, "arc")
    names = {e.name for e in B.ENTRIES}
    assert {c.name for c in X.trained_cells()} <= names


def test_core_is_the_training_envelope() -> None:
    assert X.GCell("trained", "step_up", "stand", 40).core
    assert X.GCell("trained", "step_up", "stand", 40, aim_deg=0.0).core
    assert not X.GCell("trained", "step_up", "stand", 40, aim_deg=25.0).core
    assert not X.GCell("trained", "step_up", "walk", 40).core
    assert X.GCell("trained", "march", "walk").core
    assert not X.GCell("trained", "bow", "walk").core


def test_seeds_are_shared_across_models_and_delays_but_not_cells() -> None:
    a = X.GCell("trained", "turn_left", "stand")
    b = X.GCell("today", "pivot", "stand")
    assert X.seed_of(a, 0) == X.seed_of(a, 0)
    assert X.seed_of(a, 0) != X.seed_of(a, 1)
    assert X.seed_of(a, 0) != X.seed_of(b, 0)


def test_outcomes_success_refused_tripped_fell() -> None:
    assert X.outcome(_turn()) == "success"
    assert X.outcome(_turn(refused=True)) == "refused"
    assert X.outcome(_turn(fell_any=True)) == "fell"
    assert X.outcome(_turn(excursion_m=0.09)) == "tripped"
    assert X.judge(_turn(excursion_m=0.09))["missed"] == ["excursion_5cm"]


def test_mat_fall_is_fell_and_bow_needs_return_to_stand() -> None:
    mat = {
        "cell": _cell("step_up", h=40),
        "outcome": "fall",
        "max_tilt_deg": 70.0,
        "fell_any": True,
    }
    assert X.outcome(mat) == "fell"
    done = {
        "cell": _cell("step_up", h=40),
        "outcome": "done",
        "max_tilt_deg": 12.0,
        "fell_any": False,
    }
    assert X.outcome(done) == "success"
    bow = {"cell": _cell("bow"), "t_bow_s": 0.5, "hold_bow_s": 0.9, "footfalls": 0, "fell_in": False,
           "fell_after": False, "fell_any": False, "returned": True}  # fmt: skip
    assert X.outcome(bow) == "success"
    assert X.outcome({**bow, "returned": False}) == "tripped"


def test_score_levels_and_verdicts() -> None:
    cols = {}
    for col in X.COLS:
        rows = [_turn() for _ in range(29)] + [_turn(fell_any=True, righted_8s=True, righted_s=1.2)]
        cols[col] = [{**r, **X.judge(r)} for r in rows]
    res = X.score(cols)
    c = res["cells"]["trained/turn_left/stand"]
    assert c["columns"]["mjcf_d4"]["level"] == "ok"
    assert c["columns"]["mjcf_d4"]["righting"]["righted_8s"]["k"] == 1
    assert res["skills"]["trained/turn_left"]["verdict"] == "pass"
    bad = {col: [{**r, **X.judge(r)} for r in [_turn(excursion_m=0.2)] * 30] for col in X.COLS}
    assert X.score(bad)["skills"]["trained/turn_left"]["verdict"] == "fail"
    assert "trained/turn_left" in X.report(res, "t")
