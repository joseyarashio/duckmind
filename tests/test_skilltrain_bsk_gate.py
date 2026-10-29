"""b:train (2026-09-28): the B-skill verdicts re-scored from Mac gate rows (no physics)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from duckmind.skilltrain import bsk_contract as B
from duckmind.skilltrain import bsk_export as E
from duckmind.skilltrain import bsk_gate as G


def _turn(ok: bool = True, fell: bool = False, **kw: Any) -> dict[str, Any]:
    r = {"cell": {"family": "turn", "entry": "stand", "skill": "turn_left", "height_m": 0.0, "soft": False},
         "dyaw_deg": 280.0 if ok else 40.0, "rate_last2_dps": 70.0, "excursion_m": 0.03, "footfalls": 17,
         "fell": fell, "leg_tau_p95": 0.24, "key": "k"}  # fmt: skip
    r.update(kw)
    return r


def _mat(outcome: str, tilt: float = 12.0) -> dict[str, Any]:
    return {"cell": {"family": "mat_up", "entry": "stand", "skill": "step_up", "height_m": 0.04, "soft": True},
            "outcome": outcome, "max_tilt_deg": tilt, "leg_tau_p95": 0.3, "key": "m"}  # fmt: skip


def test_turn_checks_follow_the_contract() -> None:
    assert G.performed(_turn())
    assert not G.performed(_turn(ok=False))
    assert not G.performed(_turn(fell=True))
    assert not G.performed(_turn(excursion_m=B.GATES["turn"]["excursion_m"] + 0.001))
    assert not G.performed(_turn(footfalls=5))  # < 1.5 steps/s over the 4 s hold


def test_mat_performed_needs_the_level_a_small_tilt_and_no_fall() -> None:
    assert G.performed(_mat("done"))
    assert not G.performed(_mat("done", tilt=31.0))
    assert not G.performed(_mat("blocked"))
    assert G.fell(_mat("fall")) and not G.performed(_mat("fall"))
    assert G.cell_key(_mat("done")) == "step_up/40mm-soft"


def test_bow_and_lie_use_the_timed_pose_fields() -> None:
    bow = {"cell": {"family": "bow", "entry": "stand", "skill": "bow"}, "t_bow_s": 0.5, "hold_bow_s": 0.9,
           "footfalls": 0, "fell_in": False, "fell_after": False}  # fmt: skip
    assert G.performed(bow)
    assert not G.performed({**bow, "t_bow_s": None})  # rows without the new fields never pass
    assert not G.performed({**bow, "fell_after": True})
    lie = {"cell": {"family": "lie", "entry": "stand", "skill": "lie_low"}, "hold_lie_s": 2.2, "footfalls": 1,
           "fell_in": False, "fell_after": False}  # fmt: skip
    assert G.performed(lie) and not G.performed({**lie, "hold_lie_s": 1.5})


def test_levels_are_stage1s() -> None:
    assert G.level(27, 1, 30) == "ok"
    assert G.level(27, 2, 30) == "weak"
    assert G.level(21, 3, 30) == "weak"
    assert G.level(20, 0, 30) == "miss"
    assert G.level(10, 15, 30) == "falls"


def test_score_passes_only_with_every_column(tmp_path: Path) -> None:
    for d in G.DELAYS:
        (tmp_path / d).mkdir()
        for servo in ("mjcf", "bam"):
            rows = [{**_turn(), "key": f"{d}/{servo}/{i}"} for i in range(30)]
            (tmp_path / d / f"rows-{servo}.jsonl").write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n"
            )
    res = G.score(G.load_rows(tmp_path))
    assert res["skills"]["turn_left"]["verdict"] == "pass"
    assert res["cells"]["turn_left/stand"]["strain_bam"] == "ok"
    bad = [
        {**_turn(ok=i >= 10), "key": f"x/{i}"} for i in range(30)
    ]  # 20/30 performed on BAM at 40 ms
    (tmp_path / "d8" / "rows-bam.jsonl").write_text("\n".join(json.dumps(r) for r in bad) + "\n")
    res = G.score(G.load_rows(tmp_path))
    assert res["skills"]["turn_left"]["verdict"] == "pass with limits"
    assert any("bam_d8 miss" in x for x in res["skills"]["turn_left"]["limits"])
    assert "| turn_left/stand |" in G.report(res)


def test_export_metadata_matches_the_trainers() -> None:
    m = E.metadata("bskbow", 700, "latest.pt")
    assert m["skills"] == [e.name for e in B.entries("bskbow")]
    assert m["action_scale"] == B.ACTION_SCALE and m["gain_ratio"] == B.GAIN_RATIO
    assert m["servo_delay_ms"] == [0, 40]
    assert m["command_encoding"]["lie_low"] == [0.0, 0.2, 0.0]


def test_a100_eval_cells_cover_every_trained_entry() -> None:
    from duckmind.skilltrain import bsk_eval as V

    for net in B.NETS:
        cells = V.cells(net)
        names = {e.name for e in B.entries(net) if e.behaviour != "release"}
        assert {c[0] for c in cells} == names
        assert len(cells) == len(set(cells))
    assert ("turn_left", "walk") in V.cells("bskstep") and ("bow", "stand") in V.cells("bskbow")
    assert V.MAT_LEVELS == (2, 4, 6)  # 3 / 4 / 5 cm with 7 rows over 2–5 cm
