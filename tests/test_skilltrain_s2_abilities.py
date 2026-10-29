"""Stage-2 abilities (:mod:`duckmind.skilltrain.s2_abilities`): the rules that turn the gate verdicts of
the A100 battery and the Mac harness into the ``abilities.json`` the neural demo's wiring reads.
Pure standard library."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from duckmind.skilltrain import s2_abilities as A
from duckmind.skilltrain import s2_contract as C

SHA = "ab" * 32


def _gates(stage: str, verdict: str = "PASS", **over: str) -> dict:
    return {g.key: {"verdict": over.get(g.key, verdict)} for g in C.gates_for(stage)}


def _src(kind: str, servo: str, delay: str, gates: dict) -> dict:
    return {
        "kind": kind,
        "servo": servo,
        "delay": delay,
        "file": "x",
        "walk_sha256": SHA,
        "gates": gates,
    }


def _all_pass(stage: str) -> list[dict]:
    return [_src("a100", "bam", "d20", _gates(stage)), _src("mac", "mjcf", "d20", _gates(stage)),
            _src("mac", "bam", "d20", _gates(stage))]  # fmt: skip


def test_everything_passed_grants_everything() -> None:
    res = A.abilities(_all_pass("gaze"), "gaze", SHA)
    assert res["usable"] and res["neck_standing"] and res["gaze_stable"]
    assert res["pose"] == {"lean": True, "sway": True, "chest_out": True, "crouch": True}
    assert res["walk_sha256"] == SHA and res["why_zh"] == []


def test_a_pose_network_never_claims_the_steady_gaze() -> None:
    res = A.abilities(_all_pass("pose"), "pose", SHA)
    assert res["usable"] and not res["gaze_stable"] and res["pose"]["lean"]


def test_one_fail_anywhere_withholds_the_group() -> None:
    src = _all_pass("gaze")
    src[1]["gates"]["crouch_mm"] = {"verdict": "FAIL"}  # the XML servos do not crouch enough
    res = A.abilities(src, "gaze", SHA)
    assert res["pose"]["crouch"] is False and res["pose"]["lean"] is True
    assert any("蹲低" in w for w in res["why_zh"])


def test_visibility_gates_need_both_servo_models() -> None:
    src = [s for s in _all_pass("gaze") if s["servo"] == "bam"]  # no mjcf run at all
    res = A.abilities(src, "gaze", SHA)
    assert not res["usable"] and not any(res["pose"].values()) and not res["gaze_stable"]
    assert res["groups"]["lean"]["gates"]["lean_roll_deg"]["missing"] == ["mjcf"]


def test_rate_gates_are_resolved_by_the_a100_and_unresolved_mac_cells_are_allowed() -> None:
    src = _all_pass("gaze")
    for s in src[1:]:
        for key in ("walk_falls", "stopgo_near_falls", "pose_falls", "sitstand_falls"):
            s["gates"][key] = {"verdict": "UNRESOLVED"}
    assert A.abilities(src, "gaze", SHA)["usable"]
    src[0]["gates"]["walk_falls"] = {"verdict": "UNRESOLVED"}  # nobody resolved it
    res = A.abilities(src, "gaze", SHA)
    assert not res["usable"] and not res["gaze_stable"] and not any(res["pose"].values())


def test_pose_programs_need_the_pose_safety_gates() -> None:
    src = _all_pass("gaze")
    src[0]["gates"]["release_rate_dps"] = {"verdict": "FAIL"}
    res = A.abilities(src, "gaze", SHA)
    assert res["usable"] and not any(res["pose"].values())
    assert res["neck_standing"] and res["gaze_stable"]  # the head's abilities do not need it


def test_the_cli_checks_the_evaluated_file(tmp_path: Path) -> None:
    onnx = tmp_path / "velstand_duck.onnx"
    onnx.write_bytes(b"net")
    sha = A.sha256(onnx)
    battery = {"complete": True, "meta": {"arms": {"cand": {"sha256": sha}}},
               "gates": {"d20": _gates("gaze"), "d0": _gates("gaze")}}  # fmt: skip
    mac = {"walk_sha256": {"mjcf/d20/r4": sha, "bam/d20/r4": sha},
           "verdicts": {"mjcf/d20/r4": _gates("gaze"), "bam/d20/r4": _gates("gaze")}}  # fmt: skip
    (tmp_path / "b.json").write_text(json.dumps(battery))
    (tmp_path / "m.json").write_text(json.dumps(mac))
    out = tmp_path / "abilities.json"
    args = ["--walk", str(onnx), "--stage", "gaze", "--battery", str(tmp_path / "b.json"), "--mac",
            str(tmp_path / "m.json"), "--arm", "r4", "--out", str(out)]  # fmt: skip
    assert A.main(args) == 0
    res = json.loads(out.read_text())
    assert res["usable"] and res["walk_sha256"] == sha and len(res["sources"]) == 4
    mac["walk_sha256"]["mjcf/d20/r4"] = "0" * 64
    (tmp_path / "m.json").write_text(json.dumps(mac))
    with pytest.raises(SystemExit):
        A.main(args)
