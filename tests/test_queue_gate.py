"""The queue's contract (pure numbers) and the gate tables' verdict rules on synthetic rows (no MuJoCo, no torch)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "tools" / "queue_gate").is_dir():
    pytest.skip("tools/queue_gate is not in this checkout", allow_module_level=True)
sys.path.insert(0, str(ROOT / "tools" / "queue_gate"))

import qtable as Q  # noqa: E402

from duckmind.skilltrain import qt_contract as C  # noqa: E402


def test_emotion_ranges_match_the_keyframes() -> None:
    data = json.loads((ROOT / "src/duckmind/neuro/emotions_data.json").read_text())
    cols = data["columns"]
    for name, key in (("neck", "neck"), ("head_pitch", "head_pitch"), ("head_yaw", "head_yaw"),
                      ("head_roll", "head_roll"), ("body_pitch", "body_pitch")):  # fmt: skip
        vals = [row[cols.index(key)] for e in data["emotions"].values() for row in e["keyframes"]]
        lo, hi = C.EMOTION_RANGES[name]
        assert abs(min(vals) - lo) < 1e-6 and abs(max(vals) - hi) < 1e-6, name


def test_training_ranges_cover_the_emotions() -> None:
    names = ("neck", "head_pitch", "head_yaw", "head_roll")
    for (lo, hi), n in zip(C.EX_HEAD_RANGES, names, strict=True):
        elo, ehi = C.EMOTION_RANGES[n]
        assert lo <= elo and hi >= ehi, n
    assert (
        C.EX_BODY_RANGES[4][0] <= 0.0
        and C.EX_BODY_RANGES[4][1] >= C.EMOTION_RANGES["body_pitch"][1]
    )


def test_twist_mixtures_are_distributions() -> None:
    for mix in (C.EX_TWIST_MIX, C.TURN_TWIST_MIX, C.CALM_TWIST_MIX):
        assert abs(sum(b[0] for b in mix) - 1.0) < 1e-9
    assert C.TURN_YAW[0] <= 0.5 and C.TURN_YAW[1] >= 2.0


def test_style_codes_are_distinct_and_targets_separate_the_styles() -> None:
    assert len(set(C.STYLE_CODE.values())) == 4
    assert set(C.STYLE_TARGETS) == set(C.STYLES)
    zs = sorted(t["z"] for t in C.STYLE_TARGETS.values())
    assert all(b - a >= 0.004 for a, b in zip(zs, zs[1:], strict=False))
    assert C.STYLE_TARGETS["sneak"]["speed"] == (0.15, 0.30) and C.STYLE_TARGETS["strut"][
        "speed"
    ] == (0.35, 0.55)


def _row(prog, vx, vyaw, arg, servo, delay, rep, **kw):
    return {
        "prog": prog,
        "vx": vx,
        "vyaw": vyaw,
        "arg": arg,
        "servo": servo,
        "delay_ms": float(delay),
        "rep": rep,
        "excluded": None,
        **kw,
    }


def test_fall_rule_is_v5_plus_one_per_30_and_zero_at_20ms_xml() -> None:
    cand = [_row("run", 0.45, 0.0, "", "mjcf", 20, i, fell=(i == 0)) for i in range(30)]
    base = [_row("run", 0.45, 0.0, "", "mjcf", 20, i, fell=False) for i in range(30)]
    v = Q.Verdict()
    Q.falls_cells(cand, base, "mjcf", 20, v, ("run",), 1, "t")
    assert v.misses == ["falls mjcf@20"]
    cand_bam = [_row("run", 0.45, 0.0, "", "bam", 20, i, fell=(i == 0)) for i in range(30)]
    base_bam = [_row("run", 0.45, 0.0, "", "bam", 20, i, fell=False) for i in range(30)]
    v = Q.Verdict()
    Q.falls_cells(cand_bam, base_bam, "bam", 20, v, ("run",), 1, "t")
    assert v.misses == []


def test_turn_table_gates_rate_translation_and_the_180_time() -> None:
    def rows(rate, disp, t180):
        out = []
        for i in range(30):
            out.append(
                _row(
                    "turn",
                    0.0,
                    2.0,
                    "skill-stand",
                    "bam",
                    20,
                    i,
                    yaw_rate_steady=rate,
                    disp_cm_per90=disp,
                    t180=t180,
                    fell=False,
                )
            )
            out.append(
                _row(
                    "turn",
                    0.0,
                    -2.0,
                    "skill-stand",
                    "bam",
                    20,
                    i,
                    yaw_rate_steady=rate,
                    disp_cm_per90=disp,
                    t180=t180,
                    fell=False,
                )
            )
        return out

    good = Q.table_turn(rows(1.6, 3.0, 1.3), rows(1.0, 9.0, 3.0), [20.0])
    assert not [m for m in good.misses if "skill-stand w=2.0" in m]
    bad = Q.table_turn(rows(1.0, 8.0, 2.0), rows(1.0, 9.0, 3.0), [20.0])
    assert {
        "skill-stand w=2.0 rate",
        "skill-stand w=2.0 translation",
        "skill-stand w=2.0 180 deg time",
    } <= set(bad.misses)
