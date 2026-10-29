"""tr:gate: one verdict per Stage-1 skill from the Mac gate runs (tier 2) and the A100 evaluation.

Everything is re-scored from the harness's per-trial rows (``rows-<servo>.jsonl``, written by
:mod:`duckmind.skilltrain.gate` / :mod:`duckmind.skilltrain.gt_gate`) with
:mod:`duckmind.skilltrain.metrics`; nothing here simulates. The runs ("columns") a verdict reads:

==============  ==========================================================================
``sim_today``   the owner's simulator as it runs today: XML servos (``mjcf``), the nominal
                duck, **no** servo command delay (lockstep applies targets at once)
``sim_delay``   the same simulator with training's servo command delay (20 ms fixed)
``mjcf_dr_0``   XML servos, a new DR duck per trial (CoM, mass, foot μ), no delay
``mjcf_dr_d``   XML servos, DR ducks, each with training's drawn delay (15–30 ms)
``bam_dr``      training's servo model (BAM XL330, the real duck's), full recipe DR
``a100``        the training lane's final evaluation (BAM on MuJoCo Warp, n = 128 per cell)
==============  ==========================================================================

**Rules** (tr:gate 2026-09-28; the plan's §6 gates, with the owner's 2026-09-28 relaxation of
fall gates — "even falling is realistic simulation: report falls, do not cap for them"):

* a trial is *performed* when every visibility check of its gate family passes
  (:func:`duckmind.skilltrain.metrics.visibility`), except that **hop_back from walking is judged
  by the hop checks** (≥ 6 cm back along the start heading by 0.7 s, ≤ 15° yaw, planted at the
  end) — the walking-stop checks cannot apply to a motion that is still backing up at 0.7 s;
* a cell (skill × posture × column) is ``ok`` when performed ≥ 90 % and falls ≤ 1 per 30
  (3.4 %), ``weak`` when performed ≥ 70 % and falls ≤ 10 %, ``falls`` when half or more fell,
  else ``miss``; release / abort cells (no visibility gate) are judged by falls only;
* a skill **passes** in a simulator configuration when every posture it was trained for is
  ``ok`` in that configuration's nominal column (what the owner's simulator shows) and at least
  ``weak`` in both DR columns (that configuration's ``mjcf`` DR and ``bam_dr``; n = 30 per cell,
  so a true 95 % reads 26/30 or less 6 % of the time, and the Mac ``bam`` port is harsher than the
  A100 — the harness lane's note 5); it **passes with limits** when only some postures pass
  (the others are not allowed) or when a DR column is below ``weak`` somewhere (named as a
  limit); it **fails** when no posture passes;
* strain is judged on ``bam_dr`` per posture (the plan: torque p95 ≤ 0.35 N·m per joint group,
  stalls ≤ 0.5 %, joint acceleration p99 ≤ 300 rad/s², no housing impact while upright; a seated
  skill's resting hip housings are the seat, see :data:`SEATED_HOUSING_NOTE`): ``ok``,
  ``marginal`` (p95 > 0.30 N·m or stalls > 0.3 %) or ``over``. The hourly caps are the harness's
  provisional ones (:data:`duckmind.skilltrain.guard.DEFAULT_RULES`), halved for a skill that is
  marginal or over in any posture (never zero: the stall gate passes everywhere and the load
  lasts under a second — :func:`caps`).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from duckmind.skilltrain import metrics as M

OK_RATE, WEAK_RATE = 0.90, 0.70
OK_FALL_RATE, WEAK_FALL_RATE, FALLS_RATE = 1.0 / 29.0, 0.10, 0.5
STRAIN_MARGIN_NM, STALL_MARGIN = 0.30, 0.003
SEATED_HOUSING_NOTE = (
    "a seated duck rests on its hip housings (the seat); the impact detector counts the load "
    "shifting on them as the stretch rises and settles (peak 9–13 N) — a normal sit-down lands "
    "on them at 24–58 N (rep-0 traces, both servo models)"
)
COLUMNS = ("sim_today", "sim_delay", "mjcf_dr_0", "mjcf_dr_d", "bam_dr")
CONFIGS = {
    "delay": ("sim_delay", "mjcf_dr_d"),
    "today": ("sim_today", "mjcf_dr_0"),
}
V5_BEFORE = {
    ("startle", "stand"): "v5/head_startle/stand",
    ("freeze", "stand"): "v5/head_startle/stand",
    ("startle", "walk0.40"): "v5/stop/walk0.40",
    ("startle", "walk0.55"): "v5/stop/walk0.55",
    ("freeze", "walk0.40"): "v5/stop/walk0.40",
    ("freeze", "walk0.55"): "v5/stop/walk0.55",
    ("hop_back", "stand"): "v5/back_burst/stand",
    ("hop_back", "walk0.40"): "v5/stop/walk0.40",
    ("shake", "stand"): "v5/head_shake/stand",
    ("stretch_tall", "stand"): "v5/look_up/stand",
    ("stretch_sit", "sit"): "v5/look_up/sit",
}
KEY_NUMBERS = {
    "startle": ("first_motion_s", "retract_025_rad", "dip_030_mm", "hold_dev_mm"),
    "freeze": ("first_motion_s", "retract_025_rad", "dip_030_mm", "hold_dev_mm"),
    "walk": ("still_s", "travel_cm", "dip_max_mm"),
    "hop_back": ("back_070_cm", "yaw_change_deg"),
    "shake": ("roll_amp_deg", "yaw_amp_rad", "yaw_hz", "footfalls_skill", "drift_cm"),
    "stretch_tall": ("rise_max_mm", "cam_pitch_max_deg", "stretch_tall_hold_s"),
    "stretch_sit": ("rise_max_mm", "cam_pitch_max_deg", "reseat_s"),
    "release": ("still_s", "travel_cm"),
    "abort": ("still_s", "travel_cm"),
}


# ── rows ─────────────────────────────────────────────────────────────────────────────────────


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    """A harness ``rows-<servo>.jsonl`` (a torn last line is skipped)."""
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


def performed(row: Mapping[str, Any]) -> bool | None:
    """The trial's visibility verdict under this module's rules (None: no visibility gate),
    re-scored from its metrics with the **current** definitions (s1b:fix: a startle from a walk
    must also pull the neck back and dip — a stored verdict from before that rule is ignored)."""
    cond = row["cond"]
    fam, posture = cond.get("family", ""), cond.get("posture", "")
    if fam == "hop_back" and posture == "walk":
        v = M.visibility("hop_back", cond.get("level"), "stand", row["metrics"])
        return v["performed"]
    judged = "stand" if posture in ("risen", "coast") else posture
    gate_fam = M.gate_family(fam, judged) if cond.get("kind") == "skill" else fam
    return M.visibility(gate_fam, cond.get("level"), judged, row["metrics"])["performed"]


def cell(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """One condition in one column: rates with Wilson 95 %, key numbers, strain."""
    ok = [r for r in rows if not r.get("excluded")]
    n = len(ok)
    out: dict[str, Any] = {"n": n, "excluded": len(rows) - n}
    if not n:
        return out
    ms = [r["metrics"] for r in ok]
    perf = [performed(r) for r in ok]
    gated = [p for p in perf if p is not None]
    out["performed"] = M.rate(sum(1 for p in gated if p), len(gated)) if gated else None
    out["falls"] = M.rate(sum(1 for m in ms if m["fell"]), n)
    out["falls_in_skill"] = M.rate(sum(1 for m in ms if m["fell_in_skill"]), n)
    out["handback_falls"] = M.rate(sum(1 for m in ms if m["handback_fall"]), n)
    out["t45"] = M.rate(sum(1 for m in ms if m["t45"]), n)
    fallen = [m for m in ms if m["fell"]]
    out["righted_8s"] = M.rate(sum(1 for m in fallen if m.get("righted_8s")), len(fallen))
    cond = ok[0]["cond"]
    fam = "walk" if cond.get("posture") == "walk" else cond.get("family", "")
    if cond.get("family") == "hop_back":
        fam = "hop_back"
    keys = KEY_NUMBERS.get(fam, ())
    if cond.get("posture") == "walk" and cond.get("family") == "hop_back":
        keys = (*KEY_NUMBERS["hop_back"], *KEY_NUMBERS["walk"])
    out["numbers"] = {k: M.dist(m.get(k) for m in ms) for k in keys}
    leg = np.concatenate([np.asarray(m["tau_leg"], float) for m in ms])
    head = np.concatenate([np.asarray(m["tau_head"], float) for m in ms])
    first = np.concatenate([np.asarray(m["tau_leg"][:35], float) for m in ms])
    acc = np.concatenate([np.asarray(m["acc"], float) for m in ms])
    cells_n = sum(m["joint_ticks"] for m in ms)
    hits_nofall = sum(1 for m in ms if m["servo_hits_upright"] > 0 and not m["fell"])
    out["strain"] = {
        "tau_leg_p95": _r(np.percentile(leg, 95)),
        "tau_head_p95": _r(np.percentile(head, 95)),
        "tau_leg_p95_first_0.7s": _r(np.percentile(first, 95)),
        "stall_frac": _r(sum(m["stall_ticks"] for m in ms) / cells_n, 5) if cells_n else None,
        "over_0.4Nm_frac": _r(sum(m["strain_ticks"] for m in ms) / cells_n, 5) if cells_n else None,
        "acc_p99": _r(np.percentile(acc, 99), 1),
        "servo_hits_upright": M.rate(sum(1 for m in ms if m["servo_hits_upright"] > 0), n),
        # the gate's "0 housing impacts outside falls": trials that did not fall
        "servo_hits_upright_nofall": M.rate(hits_nofall, n),
    }
    return out


def _r(x: float, digits: int = 4) -> float:
    return round(float(x), digits)


def columns(runs: Mapping[str, str | Path]) -> dict[str, dict[str, dict[str, Any]]]:
    """``{column: {condition key: cell}}`` from ``{column: rows file}``."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for name, path in runs.items():
        by: dict[str, list[dict[str, Any]]] = {}
        for r in load_rows(path):
            by.setdefault(r["cond"]["key"], []).append(r)
        out[name] = {k: cell(v) for k, v in sorted(by.items())}
    return out


# ── judging ──────────────────────────────────────────────────────────────────────────────────


def status(c: Mapping[str, Any] | None) -> str:
    """ok | weak | miss | falls | none (no trials) for one cell."""
    if not c or not c.get("n"):
        return "none"
    falls = c["falls"]["k"] / c["falls"]["n"]
    if falls >= FALLS_RATE:
        return "falls"
    perf = c.get("performed")
    rate = 1.0 if perf is None or not perf.get("n") else perf["k"] / perf["n"]
    if rate >= OK_RATE and falls <= OK_FALL_RATE:
        return "ok"
    if rate >= WEAK_RATE and falls <= WEAK_FALL_RATE:
        return "weak"
    return "miss"


def strain_status(c: Mapping[str, Any] | None, *, seated: bool = False) -> str:
    """ok | marginal | over (bam cells), per the plan's strain gate."""
    if not c or not c.get("n"):
        return "none"
    s = c["strain"]
    tau = max(float(s["tau_leg_p95"]), float(s["tau_head_p95"]))
    stall = float(s["stall_frac"] or 0.0)
    hits = 0 if seated else int(s["servo_hits_upright_nofall"]["k"])
    if tau > M.TAU_P95_GATE or stall > M.STALL_GATE or float(s["acc_p99"]) > M.ACC_P99_GATE or hits:
        return "over"
    if tau > STRAIN_MARGIN_NM or stall > STALL_MARGIN:
        return "marginal"
    return "ok"


@dataclass(frozen=True)
class Posture:
    label: str  # stand | walk0.40 | walk0.55 | sit
    key: str  # the harness condition key


def postures_of(entry: Mapping[str, Any]) -> list[Posture]:
    """The gate conditions a table entry was tried in (the harness's own keys)."""
    name, fam = entry["name"], entry.get("family", "")
    if fam == "release":
        return [Posture("stand", "seq/freeze_4_release/stand"),
                Posture("walk0.40", "seq/freeze_4_release/walk0.40")]  # fmt: skip
    if fam == "abort":
        return [Posture("stand", "seq/startle_4_abort/stand"),
                Posture("walk0.40", "seq/startle_4_abort/walk0.40")]  # fmt: skip
    out = []
    for p in entry.get("postures") or ("stand",):
        if p == "walk":
            speeds = (0.40, 0.55) if fam in ("startle", "freeze") else (0.40,)
            out += [Posture(f"walk{v:.2f}", f"skill/{name}/walk{v:.2f}") for v in speeds]
        else:
            out.append(Posture(p, f"skill/{name}/{p}"))
    return out


def judge(
    entry: Mapping[str, Any], cols: Mapping[str, Mapping[str, Mapping[str, Any]]], config: str
) -> dict[str, Any]:
    """The verdict of one skill in one simulator configuration (``delay`` or ``today``)."""
    sim_col, dr_col = CONFIGS[config]
    rows = []
    limits: list[str] = []
    for p in postures_of(entry):
        st = {c: status(cols.get(c, {}).get(p.key)) for c in (sim_col, dr_col, "bam_dr")}
        passes = st[sim_col] == "ok"
        robust = all(st[c] in ("ok", "weak") for c in (dr_col, "bam_dr"))
        if not passes:
            limits.append(f"{p.label}: not allowed ({sim_col} {st[sim_col]})")
        for c in (dr_col, "bam_dr"):
            if passes and st[c] not in ("ok", "weak"):
                limits.append(f"{p.label}: {c} {st[c]}")
        rows.append({"posture": p.label, "key": p.key, "status": st, "passes": passes,
                     "robust": robust})  # fmt: skip
    good = [r["posture"] for r in rows if r["passes"]]
    if good and len(good) == len(rows) and all(r["robust"] for r in rows):
        verdict = "pass"
    elif good:
        verdict = "pass_with_limits"
    else:
        verdict = "fail"
    return {"config": config, "verdict": verdict, "allowed": good, "limits": limits,
            "postures": rows}  # fmt: skip


def caps(
    entries: Iterable[Mapping[str, Any]],
    cols: Mapping[str, Mapping[str, Mapping[str, Any]]],
    base: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Hourly caps: ``base`` (``guard.DEFAULT_RULES``), halved when the skill's bam strain is
    marginal or over in any posture it is allowed in."""
    out: dict[str, dict[str, Any]] = {}
    bam = cols.get("bam_dr", {})
    for e in entries:
        rule = base.get(e["name"])
        if rule is None:
            continue
        per_posture = {
            p.label: strain_status(bam.get(p.key), seated=(p.label == "sit"))
            for p in postures_of(e)
        }
        worst = max(per_posture.values(), key=("none", "ok", "marginal", "over").index)
        per_hour = rule.per_hour
        if per_hour is not None and worst in ("marginal", "over"):
            per_hour = max(1, per_hour // 2)
        out[e["name"]] = {
            "per_hour": per_hour,
            "min_gap_s": rule.min_gap_s,
            "provisional_per_hour": rule.per_hour,
            "strain": per_posture,
            "why": "halved: strain marginal/over on bam" if per_hour != rule.per_hour else "kept",
        }
    return out


def a100_cells(paths: Iterable[str | Path]) -> dict[str, dict[str, Any]]:
    """``{skill/kind: cell}`` from the training lane's ``eval.json`` files."""
    out: dict[str, dict[str, Any]] = {}
    for path in paths:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        out.update(doc.get("cells") or {})
    return out


def a100_summary(name: str, cells: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Pooled A100 numbers of a skill: performed and falls over its standing/walking/seated
    cells (the 'crouched' entry kind is left out: a crouch cannot be measured from a crouch)."""
    kinds = {}
    perf_k = perf_n = fall_k = fall_n = 0
    for key, c in cells.items():
        skill, _, kind = key.partition("/")
        if skill != name or kind == "crouched":
            continue
        kinds[kind] = {
            "performed": c.get("performed"),
            "fall_skill": c.get("fall_skill"),
            "fall_handback": c.get("fall_handback"),
            "tau_leg_p95": c.get("tau_leg_p95"),
        }
        if c.get("performed"):
            perf_k += int(c["performed"][0])
            perf_n += int(c["performed"][1])
        fall_k += int(c["fall_skill"][0])
        fall_n += int(c["fall_skill"][1])
    return {
        "kinds": kinds,
        "performed": M.rate(perf_k, perf_n) if perf_n else None,
        "falls_in_skill": M.rate(fall_k, fall_n) if fall_n else None,
    }


def build(
    table: str | Path,
    runs: Mapping[str, str | Path],
    a100: Iterable[str | Path] = (),
    base_rules: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Every skill's verdicts (both simulator configurations), cells, v5 before, caps."""
    from duckmind.skilltrain import guard as G

    doc = json.loads(Path(table).read_text(encoding="utf-8"))
    entries = doc.get("skills") or []
    cols = columns(runs)
    a100c = a100_cells(a100)
    skills: dict[str, Any] = {}
    for e in entries:
        name, fam = e["name"], e.get("family", "")
        per = {}
        for p in postures_of(e):
            before_key = V5_BEFORE.get((fam, p.label))
            per[p.label] = {
                "key": p.key,
                "cells": {c: cols.get(c, {}).get(p.key) for c in COLUMNS},
                "status": {c: status(cols.get(c, {}).get(p.key)) for c in COLUMNS},
                "strain_bam": strain_status(
                    cols.get("bam_dr", {}).get(p.key), seated=(p.label == "sit")
                ),
                "v5_before": {c: cols.get(c, {}).get(before_key) for c in COLUMNS if before_key},
                "v5_before_key": before_key,
            }
        verdict = judge(e, cols, "delay")
        today = judge(e, cols, "today")
        worst = str(
            max(
                (str(per[p]["strain_bam"]) for p in verdict["allowed"] if p in per),
                key=("none", "ok", "marginal", "over").index,
                default="none",
            )
        )
        skills[name] = {
            "family": fam,
            "level": e.get("level"),
            "trained_postures": e.get("postures"),
            "verdict": verdict,
            "verdict_today": today,
            # (s1b:fix) never a plain "pass" over a strain failure
            "label": strain_label(verdict["verdict"], worst),
            "label_today": strain_label(today["verdict"], worst),
            "strain_bam_worst_allowed": worst,
            "postures": per,
            "a100": a100_summary(name, a100c),
        }
    return {
        "format": "duckmind-skill-verdicts-1",
        "table": str(table),
        "runs": {k: str(v) for k, v in runs.items()},
        "a100": [str(p) for p in a100],
        "rules": {
            "ok": f"performed ≥ {OK_RATE:.0%} and falls ≤ 1/30",
            "weak": f"performed ≥ {WEAK_RATE:.0%} and falls ≤ {WEAK_FALL_RATE:.0%}",
            "falls": f"falls ≥ {FALLS_RATE:.0%}",
            "pass": "every trained posture ok in the configuration's nominal column and at least "
            "weak in its mjcf DR column and in bam_dr",
            "pass_with_limits": "only some postures ok in the nominal column (only those "
            "allowed), or a DR column below weak somewhere (named)",
            "hop_back_walk": "judged by the hop checks (back ≥ 6 cm by 0.7 s, yaw ≤ 15°, planted)",
            "strain": "bam; p95 ≤ 0.35 N·m, stalls ≤ 0.5 %, acc p99 ≤ 300 rad/s², no housing "
            "impact while upright (seated: the seat excluded); marginal > 0.30 N·m",
            "seated_housing": SEATED_HOUSING_NOTE,
        },
        "skills": skills,
        "caps": caps(entries, cols, base_rules if base_rules is not None else G.DEFAULT_RULES),
    }


def strain_label(verdict: str, strain: str) -> str:
    """(s1b:fix) The verdict as it may be said: a strain failure on the training servo model
    is written next to it, never folded into a plain pass."""
    words = {"pass": "motion pass", "pass_with_limits": "motion pass with limits",
             "fail": "motion fail"}  # fmt: skip
    head = words.get(verdict, verdict)
    if strain == "over":
        return f"{head} / strain FAIL (waived in the simulator; closed for the real duck)"
    if strain == "marginal":
        return f"{head} / strain marginal"
    return head


def _fmt_rate(r: Mapping[str, Any] | None) -> str:
    if not r or not r.get("n"):
        return "—"
    lo, hi = r["wilson95"]
    return f"{r['k']}/{r['n']} [{lo * 100:.0f}–{hi * 100:.0f}%]"


def _fmt_num(d: Mapping[str, Any] | None, digits: int = 1) -> str:
    if not d or not d.get("n"):
        return "—"
    return f"{d['p50']:.{digits}f} ({d['p10']:.{digits}f}–{d['p90']:.{digits}f})"


def markdown(v: Mapping[str, Any]) -> str:
    """The verdict tables (English, for the orchestrator)."""
    lines = ["# Stage-1 skill verdicts (tr:gate)", ""]
    lines.append(
        "Cells: performed k/n [Wilson 95 %] · falls k/n. Columns: sim today = XML servos, "
        "nominal duck, 0 ms command delay (the owner's simulator now); sim +20 ms = the same "
        "with training's servo command delay; mjcf DR 0 / DR+delay = XML servos with DR ducks; "
        "bam DR = training's servo model with the recipe's DR and delay."
    )
    lines.append("")
    lines.append(
        "| skill | verdict (+20 ms delay) | allowed | limits | verdict (today, 0 ms) | allowed "
        "today |"
    )
    lines.append("|---|---|---|---|---|---|")
    for name, s in v["skills"].items():
        d, t = s["verdict"], s["verdict_today"]
        lines.append(
            f"| {name} | {s.get('label', d['verdict'])} | {', '.join(d['allowed']) or '—'} | "
            f"{'; '.join(d['limits']) or '—'} | {s.get('label_today', t['verdict'])} | "
            f"{', '.join(t['allowed']) or '—'} |"
        )
    lines.append("")
    head = "| skill | posture | " + " | ".join(COLUMNS) + " | strain (bam) |"
    lines += [head, "|" + "---|" * (len(COLUMNS) + 3)]
    for name, s in v["skills"].items():
        for label, p in s["postures"].items():
            cells = []
            for c in COLUMNS:
                ce = p["cells"].get(c)
                if not ce or not ce.get("n"):
                    cells.append("—")
                    continue
                cells.append(
                    f"{_fmt_rate(ce.get('performed'))} · falls {ce['falls']['k']}/{ce['falls']['n']}"
                    f" ({p['status'][c]})"
                )
            lines.append(f"| {name} | {label} | " + " | ".join(cells) + f" | {p['strain_bam']} |")
    lines.append("")
    lines.append("## Hourly caps (bam strain)")
    lines.append("")
    lines.append("| skill | per hour | provisional | strain by posture | why |")
    lines.append("|---|---|---|---|---|")
    for name, c in v["caps"].items():
        lines.append(
            f"| {name} | {c['per_hour']} | {c['provisional_per_hour']} | "
            f"{', '.join(f'{k} {s}' for k, s in c['strain'].items())} | {c['why']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m duckmind.skilltrain.gt_verdict")
    ap.add_argument("--table", required=True, help="the gated skills.json")
    ap.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="COLUMN=ROWS.jsonl",
        help=f"a column's rows file ({', '.join(COLUMNS)})",
    )
    ap.add_argument("--a100", action="append", default=[], help="an eval.json (repeatable)")
    ap.add_argument("--out", required=True, help="writes OUT.json and OUT.md")
    args = ap.parse_args(argv)
    runs = {}
    for item in args.run:
        name, _, path = item.partition("=")
        if name not in COLUMNS:
            raise SystemExit(f"unknown column {name!r}; use {COLUMNS}")
        runs[name] = path
    v = build(args.table, runs, args.a100)
    out = Path(args.out)
    out.with_suffix(".json").write_text(json.dumps(v, indent=1, ensure_ascii=False), "utf-8")
    out.with_suffix(".md").write_text(markdown(v), "utf-8")
    for name, s in v["skills"].items():
        print(
            f"{name:16s} +delay {s['verdict']['verdict']:17s} today "
            f"{s['verdict_today']['verdict']:17s} allowed {s['verdict']['allowed']}"
        )
    return 0


__all__ = [
    "COLUMNS",
    "CONFIGS",
    "build",
    "caps",
    "cell",
    "columns",
    "judge",
    "markdown",
    "performed",
    "postures_of",
    "status",
    "strain_label",
    "strain_status",
]

if __name__ == "__main__":
    raise SystemExit(main())
