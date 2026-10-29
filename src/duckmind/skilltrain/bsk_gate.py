"""b:train: one verdict per B skill from the Mac gate rows (``bsk_probe run --onnx-dir … --delay-steps d``).

Nothing here simulates: every trial row is re-scored against ``bsk_contract.GATES``. The runs ("columns") are
the two servo models × the fixed servo command delays of the gate (``d0`` / ``d4`` / ``d8`` = 0 / 20 / 40 ms),
each with a fresh DR duck per trial:

* ``mjcf_d4`` — the owner's simulator as it will run the skills (XML servos, a fixed 20 ms delay): the
  **nominal** column;
* ``mjcf_d0`` / ``mjcf_d8``, ``bam_d0`` / ``bam_d4`` / ``bam_d8`` — the DR columns (the BAM XL330 model is the
  real servo's, the one training used).

**Rules** (Stage 1's tr:gate levels, the owner's 2026-09-28 relaxation — falls are reported, not capped):

* a trial is *performed* when every check of its skill passes (:func:`performed`);
* a cell (entry × posture/height × column) is ``ok`` when performed ≥ 90 % and falls ≤ 1 per 29,
  ``weak`` when performed ≥ 70 % and falls ≤ 10 %, ``falls`` when half or more fell, else ``miss``;
* a skill **passes** when every cell it was trained for is ``ok`` in the nominal column and at least ``weak``
  in every DR column; **passes with limits** when every cell is at least ``weak`` in the nominal column (the
  limits are named); **fails** otherwise;
* strain is judged on BAM (the real servo's model) per cell: leg torque p95 (the trials' median) ≤ 0.30 N·m
  ``ok``, ≤ 0.35 ``marginal``, else ``over``; the XML servos' number is reported beside it.

    python -m duckmind.skilltrain.bsk_gate --gate DIR   # DIR/d0, DIR/d4, DIR/d8 with rows-{mjcf,bam}.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from duckmind.skilltrain import bsk_contract as B

OK_RATE, WEAK_RATE = 0.90, 0.70
OK_FALL_RATE, WEAK_FALL_RATE, FALLS_RATE = 1.0 / 29.0, 0.10, 0.5
STRAIN_OK_NM, STRAIN_MAX_NM = 0.30, B.GATES["safety"]["torque_p95_nm"]
NOMINAL = "mjcf_d4"
DELAYS = {"d0": 0, "d4": 20, "d8": 40}
SKILL_OF = {  # bsk_probe's trained cell → the contract entry
    "turn_left": "turn",
    "turn_right": "turn",
    "march": "march",
    "step_up": "step_up",
    "step_down": "step_down",
    "bow": "bow",
    "lie_low": "lie",
}


def _num(r: Mapping[str, Any], k: str) -> float | None:
    v = r.get(k)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return float(v)


def fell(r: Mapping[str, Any]) -> bool:
    beh = SKILL_OF[r["cell"]["skill"]]
    if beh in ("step_up", "step_down"):
        return r.get("outcome") == "fall"
    if beh in ("bow", "lie"):
        return bool(r.get("fell_in")) or bool(r.get("fell_after"))
    return bool(r.get("fell"))


def checks(r: Mapping[str, Any]) -> dict[str, bool]:
    """Each gate check of the row's skill (bsk_contract.GATES) → passed?"""
    beh = SKILL_OF[r["cell"]["skill"]]
    g = B.GATES[beh]
    hold_s = 4.0  # bsk_probe's held window for turn / march

    def ge(k: str, lim: float) -> bool:
        v = _num(r, k)
        return v is not None and v >= lim

    def le(k: str, lim: float) -> bool:
        v = _num(r, k)
        return v is not None and v <= lim

    if beh == "turn":
        return {"yaw_90_in_4s": ge("dyaw_deg", g["yaw_deg_in_4s"]), "rate_30dps": ge("rate_last2_dps", g["rate_dps"]),
                "excursion_5cm": le("excursion_m", g["excursion_m"]),
                "steps_1.5_per_s": ge("footfalls", g["steps_per_s"] * hold_s), "no_fall": not fell(r)}  # fmt: skip
    if beh == "march":
        dy = _num(r, "dyaw_deg")
        return {"footfalls_2_per_s": ge("footfalls", g["footfalls_per_s"] * hold_s),
                "lift_20mm": ge("foot_lift_mm", g["lift_mm"]), "travel_5cm": le("net_m", g["travel_m_4s"]),
                "yaw_15deg": dy is not None and abs(dy) <= g["yaw_deg_4s"], "no_fall": not fell(r)}  # fmt: skip
    if beh in ("step_up", "step_down"):
        return {"on_the_level": r.get("outcome") == "done", "tilt_30deg": le("max_tilt_deg", g["tilt_max_deg"]),
                "no_fall": not fell(r)}  # fmt: skip
    if beh == "bow":
        return {"pose_by_0.6s": le("t_bow_s", g["by_s"]), "held_0.8s": ge("hold_bow_s", g["hold_s"]),
                "footfalls_le_1": le("footfalls", g["footfalls"]), "no_fall": not fell(r)}  # fmt: skip
    return {"held_2s": ge("hold_lie_s", g["hold_s"]), "footfalls_le_1": le("footfalls", g["footfalls"]),
            "no_fall": not fell(r)}  # fmt: skip


def performed(r: Mapping[str, Any]) -> bool:
    return all(checks(r).values())


def level(perf: int, falls: int, n: int) -> str:
    if n == 0:
        return "none"
    if falls / n >= FALLS_RATE:
        return "falls"
    if perf / n >= OK_RATE and falls / n <= OK_FALL_RATE:
        return "ok"
    if perf / n >= WEAK_RATE and falls / n <= WEAK_FALL_RATE:
        return "weak"
    return "miss"


def _rate(k: int, n: int) -> dict[str, Any]:
    lo, hi = B.wilson(k, n)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "wilson95": [round(lo, 4), round(hi, 4)],
            "text": f"{k}/{n} [{100 * lo:.0f}–{100 * hi:.0f} %]"}  # fmt: skip


def _median(vals: Sequence[float | None]) -> float | None:
    v = sorted(x for x in vals if x is not None)
    if not v:
        return None
    m = len(v) // 2
    return round(v[m] if len(v) % 2 else 0.5 * (v[m - 1] + v[m]), 4)


def cell_key(r: Mapping[str, Any]) -> str:
    c = r["cell"]
    if c["family"].startswith("mat"):
        return f"{c['skill']}/{round(c['height_m'] * 1000):d}mm{'-soft' if c.get('soft') else ''}"
    return f"{c['skill']}/{c['entry']}"


def load_rows(gate_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    """{column: rows} from DIR/d0, DIR/d4, DIR/d8 (each with rows-mjcf.jsonl / rows-bam.jsonl); a row seen twice
    (a resumed run) counts once."""
    out: dict[str, list[dict[str, Any]]] = {}
    for d in DELAYS:
        for servo in ("mjcf", "bam"):
            p = Path(gate_dir) / d / f"rows-{servo}.jsonl"
            if not p.is_file():
                continue
            seen: dict[str, dict[str, Any]] = {}
            for line in p.read_text().splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("excluded") or not (r.get("cell") or {}).get("skill"):
                    continue
                seen[r["key"]] = r
            out[f"{servo}_{d}"] = list(seen.values())
    return out


def score(columns: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    cells: dict[str, dict[str, Any]] = {}
    for col, rows in columns.items():
        groups: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            groups.setdefault(cell_key(r), []).append(r)
        for key, rs in sorted(groups.items()):
            n = len(rs)
            perf = sum(performed(r) for r in rs)
            falls = sum(fell(r) for r in rs)
            names = list(checks(rs[0]))
            per_check = {c: _rate(sum(checks(r)[c] for r in rs), n)["text"] for c in names}
            tau = _median([_num(r, "leg_tau_p95") for r in rs])
            cells.setdefault(key, {"skill": rs[0]["cell"]["skill"], "columns": {}})["columns"][col] = {
                "n": n, "performed": _rate(perf, n), "falls": _rate(falls, n), "level": level(perf, falls, n),
                "checks": per_check, "leg_tau_p95_median": tau,
            }  # fmt: skip
    skills: dict[str, dict[str, Any]] = {}
    for key, c in cells.items():
        s = skills.setdefault(c["skill"], {"cells": [], "limits": []})
        s["cells"].append(key)
        cols = c["columns"]
        nominal = cols.get(NOMINAL, {}).get("level", "none")
        dr = {k: v["level"] for k, v in cols.items() if k != NOMINAL}
        c["nominal"] = nominal
        c["dr_levels"] = dr
        bam_tau = [
            v["leg_tau_p95_median"]
            for k, v in cols.items()
            if k.startswith("bam") and v["leg_tau_p95_median"]
        ]
        worst = max(bam_tau) if bam_tau else None
        c["strain_bam"] = (None if worst is None else "ok" if worst <= STRAIN_OK_NM else
                           "marginal" if worst <= STRAIN_MAX_NM else "over")  # fmt: skip
        c["strain_bam_nm"] = worst
        mj = [
            v["leg_tau_p95_median"]
            for k, v in cols.items()
            if k.startswith("mjcf") and v["leg_tau_p95_median"]
        ]
        c["strain_mjcf_nm"] = max(mj) if mj else None
        c["pass"] = (
            nominal == "ok" and all(v in ("ok", "weak") for v in dr.values()) and len(dr) >= 5
        )
        c["pass_limits"] = nominal in ("ok", "weak")
        if not c["pass"]:
            bad = [f"{k} {v}" for k, v in {NOMINAL: nominal, **dr}.items() if v not in ("ok",)]
            s["limits"].append(f"{key}: " + ", ".join(bad))
    for name, s in skills.items():
        cs = [cells[k] for k in s["cells"]]
        s["verdict"] = ("pass" if all(c["pass"] for c in cs) else "pass with limits"
                        if all(c["pass_limits"] for c in cs) else "fail")  # fmt: skip
        s["strain_bam"] = max(
            (c["strain_bam"] or "ok" for c in cs), key=("ok", "marginal", "over").index
        )
    return {"format": "duckmind-bsk-gate-1", "nominal": NOMINAL, "cells": cells, "skills": skills}


def report(res: Mapping[str, Any]) -> str:
    cols = ["mjcf_d0", "mjcf_d4", "mjcf_d8", "bam_d0", "bam_d4", "bam_d8"]
    lines = ["| cell | " + " | ".join(cols) + " | strain bam / mjcf (N·m) |",
             "|---|" + "---|" * len(cols) + "---|"]  # fmt: skip
    for key, c in sorted(res["cells"].items()):
        row = []
        for col in cols:
            v = c["columns"].get(col)
            row.append(
                "—"
                if v is None
                else f"{v['level']} · done {v['performed']['text']} · falls {v['falls']['text']}"
            )
        lines.append(
            f"| {key} | " + " | ".join(row) + f" | {c['strain_bam_nm']} / {c['strain_mjcf_nm']} |"
        )
    lines.append("")
    for name, s in sorted(res["skills"].items()):
        lim = ("; limits: " + "; ".join(s["limits"])) if s["limits"] else ""
        lines.append(f"- **{name}**: {s['verdict']} (strain on BAM {s['strain_bam']}){lim}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bsk_gate")
    ap.add_argument("--gate", required=True, help="a folder with d0/ d4/ d8/ bsk_probe outputs")
    a = ap.parse_args(argv)
    res = score(load_rows(a.gate))
    Path(a.gate, "verdict.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    text = report(res)
    Path(a.gate, "verdict.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
