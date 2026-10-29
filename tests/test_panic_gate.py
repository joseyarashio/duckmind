"""The panic-gait contract (pure numbers) and the gate's verdict rules on synthetic rows (no MuJoCo, no torch)."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "tools" / "panic_gate").is_dir():
    pytest.skip("tools/panic_gate is not in this checkout", allow_module_level=True)
sys.path.insert(0, str(ROOT / "tools" / "panic_gate"))

import gate as G  # noqa: E402

from duckmind.skilltrain import pg_contract as C  # noqa: E402


def test_reward_revisions_name_known_terms_only() -> None:
    known = {
        "track_vel_sharp",
        "track_vel_forward",
        "track_yaw_sharp",
        "track_vel_coarse",
        "heading_hold",
        "yaw_mean",
    }
    assert C.DEFAULT_REWARD_REV in C.REWARD_REVS
    for rev, changes in C.REWARD_REVS.items():
        assert set(changes) <= known, rev
        for ch in changes.values():
            assert set(ch) <= {"weight", "params"}


def test_numbers_are_consistent() -> None:
    its = [i for i, _ in C.VMAX_STAGES]
    assert its == sorted(its) and C.VMAX_STAGES[-1][1] == 1.0
    assert C.TARGET_K0 * C.TARGET_KNEE < C.TARGET_TOP_VX
    assert len(C.BC["anchor_dims"]) == C.ACTION_LEN == 14
    assert C.P_MID + C.P_FAST < 1.0
    assert C.RECONDITION_STD == {48: 0.45}


def test_trainer_modules_parse() -> None:
    for name in ("pg_mdp", "pg_bc", "pg_tasks", "pg_train"):
        ast.parse((ROOT / "src" / "duckmind" / "skilltrain" / f"{name}.py").read_text())


def _run_row(
    rep: int,
    servo: str,
    vx: float,
    vyaw: float,
    chord: float,
    yaw_rate: float = 0.0,
    fell: bool = False,
) -> dict:
    return {
        "prog": "run", "vx": vx, "vyaw": vyaw, "servo": servo, "delay_ms": 20.0, "rep": rep, "excluded": None,
        "fell": fell, "t_stop05": 0.6,
        "m": {"chord10": chord, "yaw_rate": yaw_rate, "t50": 0.3, "path10": chord},
        "tau_leg_p95": 0.5, "stall_frac": 0.0,
    }  # fmt: skip


def _rows(chord_fast: float, arc: float) -> list[dict]:
    rows = []
    for servo in ("bam", "mjcf"):
        for rep in range(5):
            rows.append(_run_row(rep, servo, 1.0, 0.0, chord_fast))
            rows.append(_run_row(rep, servo, 0.8, 1.0, 0.2, yaw_rate=arc))
            rows.append(_run_row(rep, servo, 0.8, -1.0, 0.2, yaw_rate=-arc))
            for v in (0.35, 0.45, 0.55):
                rows.append(_run_row(rep, servo, v, 0.0, 0.2))
            rows.append({"prog": "still", "vx": 0.0, "vyaw": 0.0, "servo": servo, "delay_ms": 20.0, "rep": rep,
                         "excluded": None, "fell": False, "drift_cm": 0.3})  # fmt: skip
            for prog in ("walkin", "turnin"):
                rows.append({"prog": prog, "vx": 1.0, "vyaw": 0.0, "servo": servo, "delay_ms": 20.0, "rep": rep,
                             "excluded": None, "fell": False, "tau_leg_p95": 0.5, "stall_frac": 0.0})  # fmt: skip
    return rows


def test_gate_thresholds() -> None:
    base = _rows(0.31, 0.45)
    good = _rows(0.60, 0.55)
    _, res = G.verdicts(good, base, delays=(20.0,))
    flags = G.flags(res, 20)
    assert flags["G1 bam"] and flags["G1 mjcf"] and flags["G2 bam"] and flags["G5 mjcf"]
    _, res = G.verdicts(_rows(0.549, 0.55), base, delays=(20.0,))
    assert not G.flags(res, 20)["G1 bam"]  # 0.549 < 0.55: a miss is a miss
    _, res = G.verdicts(_rows(0.6, 0.49), base, delays=(20.0,))
    assert not G.flags(res, 20)["G2 bam"]  # yaw rate 0.49 < 0.5 of the command
    text = G.render(res, "x", "v5")
    assert "| G1 chord10" in text and "MISS" in text


def test_verdict_scales_fall_tolerance_with_n(tmp_path) -> None:
    import json

    import verdict as V

    def write(label: str, falls_per_cell: int) -> None:
        rows = _rows(0.6, 0.55)
        for r in rows:
            r["rep"] += 30
        for r in rows:
            if r["prog"] == "walkin" and r["servo"] == "mjcf" and r["rep"] < 30 + falls_per_cell:
                r["fell"] = True
        (tmp_path / f"rows-{label}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    write("v5", 0)
    write("c", 0)
    ok = V.verdict(tmp_path, "c", 30, delays=(20.0,))
    assert "G1 bam" not in ok["misses"] and "G5 mjcf@20" not in ok["misses"]
    write("c", 1)
    bad = V.verdict(tmp_path, "c", 30, delays=(20.0,))
    assert "G5 mjcf@20" in bad["misses"]  # 0 falls are allowed on the XML servo at 20 ms
