"""s1b:fix: honest gate reports for the Stage-1 skills (tr:review's findings on gate honesty).

Everything is re-scored from the harness's per-trial rows (``rows-*.jsonl`` of
:mod:`~duckmind.skilltrain.gate`, :mod:`~duckmind.skilltrain.gt_gate`,
:mod:`~duckmind.skilltrain.gt_latency`) with the **current** visibility rules
(:func:`duckmind.skilltrain.metrics.visibility`: a startle from a walk must now also show the
neck pulled back and the trunk down — a plain stop no longer passes). Nothing here simulates.

What it says, and how:

* **pooled fall-rate gates** (plan §6: ≤ 0.2 % for startle_1–3, the shakes and stretches; ≤ 0.5 %
  for the deepest startle, the freezes, the hop-back, the abort) per DR column: k/n over every
  trial of the group in its allowed postures, Wilson 95 %, and PASS / FAIL / UNRESOLVED
  (:func:`duckmind.skilltrain.metrics.rate_verdict`) — a FAIL is written FAIL, whatever the
  simulator does with it (the owner's rule: falls are real outcomes, reported, not hidden);
* **independent ducks only**: a column of the nominal duck (no DR) is *one duck* per condition:
  its 30 replays differ only in the start yaw and timing, so it is reported as that duck's
  outcome (performed / fell in how many replays) — never as n = 30 with a Wilson interval;
* **walking cells beside v5's own stop** at the same speed (performed under the same rule, the
  crouch and neck numbers, the heading change and the share that spun > 20°);
* **strain** labels a verdict ``motion pass / strain FAIL (waived in the simulator)`` — never a
  plain pass — when the bam strain gate fails;
* **righting** after every fall (righted within 8 s, k/n).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from duckmind.skilltrain import metrics as M

FALL_GATE = M.FALL_GATE  # {"low": 0.002, "high": 0.005}
SPIN_DEG = 20.0


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    p = Path(path)
    if not p.is_file():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def posture_label(cond: Mapping[str, Any]) -> str:
    """stand | walk0.40 | walk0.55 | sit | risen | coast (the verdicts' posture names)."""
    posture = str(cond.get("posture") or "")
    if posture == "walk":
        return f"walk{float(cond.get('speed') or 0.0):.2f}"
    return posture


def skill_of(cond: Mapping[str, Any]) -> str | None:
    """The table entry a row judges (the held entry of a continuation; None for v5)."""
    if cond.get("kind") != "skill":
        return None
    return str(cond.get("cont") or cond.get("skill") or "")


def performed_now(row: Mapping[str, Any]) -> bool | None:
    """The trial's visibility verdict under the current rules (hop_back from a walk by the hop
    checks, as gt_verdict judges it)."""
    cond = row["cond"]
    fam, posture = str(cond.get("family", "")), str(cond.get("posture", ""))
    judged = "stand" if posture in ("risen", "coast") else posture
    if fam == "hop_back" and judged == "walk":
        return M.visibility("hop_back", cond.get("level"), "stand", row["metrics"])["performed"]
    gate_fam = M.gate_family(fam, judged) if cond.get("kind") == "skill" else fam
    return M.visibility(gate_fam, cond.get("level"), judged, row["metrics"])["performed"]


def fall_group(cond: Mapping[str, Any]) -> str | None:
    """The plan's fall gate a skill trial counts toward (None: a v5 baseline)."""
    if cond.get("kind") != "skill":
        return None
    fam = str(cond.get("family", ""))
    level = float(cond.get("level") or 0.0)
    high = fam in ("hop_back", "freeze", "abort", "continuation") or (
        fam == "startle" and level >= 1.0
    )
    return "high" if high else "low"


def valid(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [r for r in rows if not r.get("excluded") and r.get("metrics")]


def pooled_gates(
    rows: Iterable[Mapping[str, Any]],
    allowed: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, Any]:
    """The plan's fall gates over every trial of each group (in the allowed postures, when
    ``allowed`` maps an entry to its allowed posture labels)."""
    groups: dict[str, list[bool]] = {"low": [], "high": []}
    for r in valid(rows):
        cond = r["cond"]
        group = fall_group(cond)
        if group is None:
            continue
        name = skill_of(cond)
        if allowed is not None and name is not None and name in allowed:
            if posture_label(cond) not in set(allowed[name]):
                continue
        groups[group].append(bool(r["metrics"]["fell"]))
    out: dict[str, Any] = {}
    for group, falls in groups.items():
        k, n = sum(falls), len(falls)
        gate = FALL_GATE[group]
        out[group] = {
            **M.rate(k, n),
            "gate": gate,
            "verdict": M.rate_verdict(k, n, gate) if n else "no trials",
            "n_for_zero": M.n_for_zero(gate),
        }
    return out


def nominal(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """A nominal-duck column as what it is: one duck per condition. Per condition the outcome
    of its replays (performed / fell in how many) and the spread of its key number; pooled, the
    number of conditions (= ducks × conditions) whose replays fell at all."""
    by: dict[str, list[Mapping[str, Any]]] = {}
    for r in valid(rows):
        by.setdefault(r["cond"]["key"], []).append(r)
    per: dict[str, Any] = {}
    fell_conditions = 0
    for key, rs in sorted(by.items()):
        perf = [performed_now(r) for r in rs]
        fell = sum(1 for r in rs if r["metrics"]["fell"])
        fell_conditions += 1 if fell else 0
        gated = [p for p in perf if p is not None]
        if fell > len(rs) / 2:
            outcome = "fell"
        elif not gated:
            outcome = "no visibility gate"  # release / abort: judged by falls only
        elif sum(1 for p in gated if p) > len(gated) / 2:
            outcome = "performed"
        else:
            outcome = "not performed"
        per[key] = {
            "replays": len(rs),
            "performed_replays": sum(1 for p in perf if p),
            "fell_replays": fell,
            "outcome": outcome,
            "kind": rs[0]["cond"].get("kind"),
        }
    return {
        "ducks": 1,
        "conditions": len(per),
        "conditions_with_a_fall": fell_conditions,
        "per_condition": per,
        "note": "one nominal duck: its replays differ only in the start yaw and timing — not "
        "independent trials, no Wilson interval",
    }


def _dist(values: Iterable[float | None]) -> dict[str, Any]:
    return M.dist(values)


def walking_vs_v5(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Every walking startle / freeze cell next to v5's own stop at the same speed."""
    by: dict[str, list[Mapping[str, Any]]] = {}
    for r in valid(rows):
        by.setdefault(r["cond"]["key"], []).append(r)
    v5 = {k: rs for k, rs in by.items() if k.startswith("v5/stop/")}
    out: list[dict[str, Any]] = []
    for key, rs in sorted(by.items()):
        cond = rs[0]["cond"]
        if cond.get("kind") != "skill" or cond.get("posture") != "walk":
            continue
        if cond.get("family") not in ("startle", "freeze"):
            continue
        speed = posture_label(cond)
        base = v5.get(f"v5/stop/{speed}", [])

        def block(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
            if not group:
                return {"n": 0}
            ms = [g["metrics"] for g in group]
            perf = [performed_now(g) for g in group]
            yaw = [m.get("yaw_change_deg") for m in ms]
            return {
                "n": len(group),
                "performed": M.rate(sum(1 for p in perf if p), len(perf)),
                "dip_030_mm": _dist(m.get("dip_030_mm") for m in ms),
                "retract_025_rad": _dist(m.get("retract_025_rad") for m in ms),
                "still_s": _dist(m.get("still_s") for m in ms),
                "yaw_change_deg": _dist(yaw),
                "spun_gt_20deg": M.rate(
                    sum(1 for y in yaw if y is not None and y > SPIN_DEG), len(yaw)
                ),
            }

        out.append({"key": key, "speed": speed, "skill": block(rs), "v5_stop": block(base)})
    return out


def righting(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Falls among skill trials and whether each was up again within 8 s."""
    fell = [r for r in valid(rows) if r["cond"].get("kind") == "skill" and r["metrics"]["fell"]]
    up = [r for r in fell if r["metrics"].get("righted_8s")]
    return {
        "falls": len(fell),
        "righted_8s": M.rate(len(up), len(fell)),
        "verdict": M.rate_verdict(len(up), len(fell), M.RIGHTING_GATE, at_most=False)
        if fell
        else "no falls",
        "not_righted": sorted(r["cond"]["key"] for r in fell if not r["metrics"].get("righted_8s")),
        "righted_s": _dist(r["metrics"].get("righted_s") for r in up),
    }


def strain_label(motion: str, strain: str) -> str:
    """The verdict as it may be said: a strain FAIL is never folded into a plain pass."""
    words = {
        "pass": "motion pass",
        "pass_with_limits": "motion pass with limits",
        "fail": "motion fail",
    }
    head = words.get(motion, motion)
    if strain == "over":
        return f"{head} / strain FAIL (waived in the simulator; closed for the real duck)"
    if strain == "marginal":
        return f"{head} / strain marginal"
    return head


def cells(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per condition: performed (current rules) and falls with Wilson, key numbers, strain."""
    by: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        by.setdefault(r["cond"]["key"], []).append(r)
    out: dict[str, dict[str, Any]] = {}
    for key, rs in sorted(by.items()):
        ok = valid(rs)
        c: dict[str, Any] = {"n": len(ok), "excluded": len(rs) - len(ok)}
        if ok:
            perf = [performed_now(r) for r in ok]
            gated = [p for p in perf if p is not None]
            ms = [r["metrics"] for r in ok]
            c["performed"] = M.rate(sum(1 for p in gated if p), len(gated)) if gated else None
            c["falls"] = M.rate(sum(1 for m in ms if m["fell"]), len(ms))
            c["falls_in_skill"] = M.rate(sum(1 for m in ms if m["fell_in_skill"]), len(ms))
            leg = np.concatenate([np.asarray(m["tau_leg"], float) for m in ms])
            head = np.concatenate([np.asarray(m["tau_head"], float) for m in ms])
            c["strain"] = {
                "tau_leg_p95": round(float(np.percentile(leg, 95)), 4) if leg.size else None,
                "tau_head_p95": round(float(np.percentile(head, 95)), 4) if head.size else None,
                "stall_frac": round(
                    sum(m["stall_ticks"] for m in ms) / max(1, sum(m["joint_ticks"] for m in ms)), 5
                ),
                "acc_p99": round(float(np.percentile(np.concatenate(
                    [np.asarray(m["acc"], float) for m in ms]), 99)), 1),
            }  # fmt: skip
            c["numbers"] = {
                k: _dist(m.get(k) for m in ms)
                for k in (
                    "dip_030_mm",
                    "retract_025_rad",
                    "dip_max_mm",
                    "bob_mm",
                    "hold_dev_mm",
                    "hold_footfalls_per_2s",
                    "still_s",
                    "yaw_change_deg",
                    "back_070_cm",
                    "footfalls_skill",
                    "roll_amp_deg",
                    "roll_osc_s",
                    "yaw_amp_rad",
                    "rise_max_mm",
                    "cam_pitch_max_deg",
                    "first_motion_s",
                    "skill_s",
                    "tilt_max_deg",
                )
                if any(m.get(k) is not None for m in ms)
            }
        out[key] = c
    return out


def strain_status(c: Mapping[str, Any] | None, *, seated: bool = False) -> str:
    """ok | marginal | over (bam cells), the plan's strain gate (gt_verdict's rule)."""
    if not c or not c.get("n"):
        return "none"
    s = c["strain"]
    tau = max(float(s["tau_leg_p95"] or 0.0), float(s["tau_head_p95"] or 0.0))
    if (
        tau > M.TAU_P95_GATE
        or float(s["stall_frac"]) > M.STALL_GATE
        or (float(s["acc_p99"]) > M.ACC_P99_GATE)
    ):
        return "over"
    if tau > 0.30 or float(s["stall_frac"]) > 0.003:
        return "marginal"
    return "ok"


def dr_factor_table(
    rows: Iterable[Mapping[str, Any]], factors: Sequence[tuple[str, str, float]]
) -> list[dict[str, Any]]:
    """Falls among skill trials split by DR draws (``(label, dr key, threshold)``: below vs at
    or above) — which part of the randomisation the falls come with."""
    rs = [r for r in valid(rows) if r["cond"].get("kind") == "skill"]
    out: list[dict[str, Any]] = []
    for label, key, cut in factors:
        lo = [r for r in rs if float(r["dr"].get(key) or 0.0) < cut]
        hi = [r for r in rs if float(r["dr"].get(key) or 0.0) >= cut]
        out.append(
            {
                "factor": label,
                "cut": cut,
                "below": M.rate(sum(1 for r in lo if r["metrics"]["fell"]), len(lo)),
                "at_or_above": M.rate(sum(1 for r in hi if r["metrics"]["fell"]), len(hi)),
            }
        )
    return out


def fmt_rate(r: Mapping[str, Any] | None, pct: int = 1) -> str:
    if not r or not r.get("n"):
        return "—"
    lo, hi = r["wilson95"]
    return f"{r['k']}/{r['n']} [{lo * 100:.{pct}f}–{hi * 100:.{pct}f} %]"


__all__ = [
    "cells",
    "dr_factor_table",
    "fall_group",
    "fmt_rate",
    "load_rows",
    "nominal",
    "performed_now",
    "pooled_gates",
    "posture_label",
    "righting",
    "skill_of",
    "strain_label",
    "strain_status",
    "walking_vs_v5",
]
