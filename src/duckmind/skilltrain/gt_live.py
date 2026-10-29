"""s1b:fix: the live skill table — built, judged, pinned.

    python -m duckmind.skilltrain.gt_live build --startle S.onnx --comfort C.onnx \\
        --walk velstand.onnx --sitstand alpha_sitstand.onnx --out skills.live.json \\
        [--delay-ms 20] [--verdicts live-verdicts.json] [--shake-s 1.0]
    python -m duckmind.skilltrain.gt_live judge --table skills.live.json \\
        --sim mjcf=ROWS.jsonl --train bam=ROWS.jsonl --out live-verdicts

``build`` writes :func:`duckmind.skilltrain.table.stage1_live` as a ``skills.json`` with

* ``files`` — every network the table runs **or hands back to**, with its sha256: the two skill
  networks, the walk network (``skill_abort`` runs it; every skill hands back to it) and the
  sitstand network (it re-seats the seated stretch). :func:`gt_wiring.verified_section`
  refuses the table when any of them changed (an OTA update, a re-export);
* ``body.command_delay_ms`` — the servo command delay the owner's simulator should run with;
* ``guard`` — duck-body's rules per entry (:mod:`duckmind.skilltrain.guard`): the postures its
  verdict allows, **the fastest walk that passed** (``max_walk_mps``), the settle times, the
  held crouches, the hourly caps (halved when the bam strain is marginal or over);
* ``gate`` per entry — the verdict it was installed under.

``judge`` gives each entry × posture a verdict from two **DR** columns (independent ducks):
the proposed simulator (the XML servos with the chosen delay) and training's servo model (BAM
with training's own per-step 15–30 ms delay). A posture is allowed when the simulator column
is ``ok`` (performed ≥ 90 %, falls ≤ 1/30) and training's column at least ``weak`` (≥ 70 %,
falls ≤ 10 %). Strain on BAM labels, never folds into a plain pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from duckmind.skilltrain import guard as G
from duckmind.skilltrain import table as T

STAGE = "stage1-2026-09-28"
POSTURE_BASE = {"stand": "stand", "sit": "sit", "walk0.40": "walk", "walk0.55": "walk"}


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(
    *,
    startle: str,
    comfort: str,
    walk: str,
    sitstand: str,
    delay_ms: float | None,
    verdicts: Mapping[str, Any] | None = None,
    shake_s: float = T.LIVE_SHAKE_S,
    scale: tuple[float, float] = (1.0, 1.0),
) -> dict[str, Any]:
    entries = T.stage1_live(
        str(Path(startle).resolve()),
        str(Path(comfort).resolve()),
        str(Path(walk).resolve()),
        startle_scale=scale[0],
        startle_gain_ratio=scale[1],
        comfort_scale=scale[0],
        comfort_gain_ratio=scale[1],
        shake_s=shake_s,
    )
    problems = [p for p in T.problems(entries) if "never chains" not in p]
    if problems:
        raise ValueError("robotd would refuse this table: " + "; ".join(problems))
    per = (verdicts or {}).get("skills") or {}
    skills = []
    rules: dict[str, Any] = {}
    for e in entries:
        raw = e.to_json()
        v = per.get(e.name)
        if v is not None:
            raw["gate"] = {k: v.get(k) for k in ("verdict", "label", "allowed", "limits")}
        skills.append(raw)
        rule = G.DEFAULT_RULES.get(e.name)
        if rule is None:
            continue
        allowed = list(v.get("allowed") or []) if v is not None else None
        spec: dict[str, Any] = {
            "per_hour": rule.per_hour,
            "min_gap_s": rule.min_gap_s,
            "settle_s": rule.settle_s,
            "rise_settle_s": rule.rise_settle_s,
            "hold": rule.hold,
            "continues": sorted(rule.continues),
            "max_walk_mps": rule.max_walk_mps,
            "postures": sorted(rule.postures),
            "note": rule.note,
        }
        if allowed is not None and e.family not in ("release", "abort"):
            spec["postures"] = sorted({POSTURE_BASE[a] for a in allowed if a in POSTURE_BASE})
            walks = [float(a[4:]) for a in allowed if a.startswith("walk")]
            spec["max_walk_mps"] = max(walks) if walks else None
            strain = str(v.get("strain_bam") or "ok") if v is not None else "ok"
            if strain in ("marginal", "over") and rule.per_hour is not None:
                spec["per_hour"] = max(1, rule.per_hour // 2)
                spec["note"] = (rule.note + " — cap halved: bam strain " + strain).strip(" —")
        rules[e.name] = spec
    doc: dict[str, Any] = {
        "format": T.FORMAT,
        "stage": STAGE,
        "table": "stage1_live (s1b:fix): held crouches, a finer startle grid",
        "files": {
            "startle": {"path": str(Path(startle).resolve()), "sha256": sha256(startle)},
            "comfort": {"path": str(Path(comfort).resolve()), "sha256": sha256(comfort)},
            "walk": {
                "path": str(Path(walk).resolve()),
                "sha256": sha256(walk),
                "role": "skill_abort runs it; every skill hands back to it",
            },
            "sitstand": {
                "path": str(Path(sitstand).resolve()),
                "sha256": sha256(sitstand),
                "role": "re-seats the seated stretch",
            },
        },  # fmt: skip
        "guard": {"global_per_hour": G.GLOBAL_PER_HOUR, "rules": rules},
        "skills": skills,
    }
    if delay_ms is not None:
        doc["body"] = {
            "command_delay_ms": float(delay_ms),
            "why": "the skill networks were trained with training's 15–30 ms servo command "
            "delay; the latency map (s1b:fix, s1b/latency-a3c3/map.md): on the XML servos 0 "
            "falls from 15 to 30 ms, the hop-back performs only at 20–25 ms",
        }
    if verdicts is not None:
        doc["gate"] = {"verdicts": verdicts.get("source"), "rules": verdicts.get("rules")}
    return doc


# ── the live verdicts ────────────────────────────────────────────────────────────────────────


def _status(c: Mapping[str, Any] | None) -> str:
    from duckmind.skilltrain import gt_latency as L

    return L.status(c)


def judge(
    table: Sequence[Mapping[str, Any]],
    sim: Mapping[str, Mapping[str, Any]],
    train: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Per entry: the postures allowed (see the module doc), the limits, the bam strain. ``sim``
    / ``train``: :func:`duckmind.skilltrain.gt_report.cells` of the two columns."""
    from duckmind.skilltrain import gt_report as R

    out: dict[str, Any] = {}
    for e in table:
        name, fam = str(e["name"]), str(e.get("family", ""))
        keys: list[tuple[str, str]] = []
        if e.get("hold"):
            keys = [("stand", f"hold/{name}/stand/3s"), ("walk0.40", f"hold/{name}/walk0.40/3s"),
                    ("walk0.55", f"hold/{name}/walk0.55/3s")]  # fmt: skip
        elif fam == "release":
            keys = [("stand", "seq/freeze_4_release/stand"),
                    ("walk0.40", "seq/freeze_4_release/walk0.40")]  # fmt: skip
        elif fam == "abort":
            keys = [("stand", "seq/startle_4_abort/stand"),
                    ("walk0.40", "seq/startle_4_abort/walk0.40")]  # fmt: skip
        else:
            for p in e.get("postures") or ("stand",):
                if p == "walk":
                    speeds = ("0.40", "0.55") if fam in ("startle", "freeze") else ("0.40",)
                    keys += [(f"walk{s}", f"skill/{name}/walk{s}") for s in speeds]
                else:
                    keys.append((p, f"skill/{name}/{p}"))
        rows = []
        for label, key in keys:
            a, b = sim.get(key), train.get(key)
            sa, sb = _status(a), _status(b)
            rows.append({"posture": label, "key": key, "sim": sa, "train": sb,
                         "passes": sa == "ok" and sb in ("ok", "weak"),
                         "strain_bam": R.strain_status(b, seated=(label == "sit"))})  # fmt: skip
        allowed = [r["posture"] for r in rows if r["passes"]]
        limits = [f"{r['posture']}: sim {r['sim']}, train {r['train']}" for r in rows
                  if not r["passes"]]  # fmt: skip
        verdict = "pass" if rows and len(allowed) == len(rows) else (
            "pass_with_limits" if allowed else "fail")  # fmt: skip
        worst = max((r["strain_bam"] for r in rows if r["passes"]),
                    key=("none", "ok", "marginal", "over").index, default="none")  # fmt: skip
        out[name] = {
            "verdict": verdict,
            "label": R.strain_label(verdict, str(worst)),
            "allowed": allowed,
            "limits": limits,
            "strain_bam": worst,
            "postures": rows,
        }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m duckmind.skilltrain.gt_live")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--startle", required=True)
    b.add_argument("--comfort", required=True)
    b.add_argument("--walk", required=True)
    b.add_argument("--sitstand", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--delay-ms", type=float)
    b.add_argument("--verdicts")
    b.add_argument("--shake-s", type=float, default=T.LIVE_SHAKE_S)
    j = sub.add_parser("judge")
    j.add_argument("--table", required=True)
    j.add_argument("--sim", required=True, help="the simulator column's rows file")
    j.add_argument("--train", required=True, help="training's column's rows file")
    j.add_argument("--out", required=True, help="writes OUT.json")
    args = ap.parse_args(argv)
    if args.cmd == "build":
        verdicts = json.loads(Path(args.verdicts).read_text("utf-8")) if args.verdicts else None
        doc = build(
            startle=args.startle,
            comfort=args.comfort,
            walk=args.walk,
            sitstand=args.sitstand,
            delay_ms=args.delay_ms,
            verdicts=verdicts,
            shake_s=args.shake_s,
        )
        Path(args.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", "utf-8")
        print(f"wrote {args.out}: {len(doc['skills'])} entries")
        return 0
    from duckmind.skilltrain import gt_report as R

    table = json.loads(Path(args.table).read_text("utf-8"))["skills"]
    sim = R.cells(R.load_rows(args.sim))
    train = R.cells(R.load_rows(args.train))
    v = {"source": {"sim": args.sim, "train": args.train}, "skills": judge(table, sim, train),
         "rules": {"allowed": "sim column ok (performed ≥ 90 %, falls ≤ 1/30) and training's "
                   "column at least weak (≥ 70 %, falls ≤ 10 %); both DR"}}  # fmt: skip
    Path(args.out).with_suffix(".json").write_text(json.dumps(v, indent=1, ensure_ascii=False))
    for name, s in v["skills"].items():
        print(f"{name:16s} {s['label']:60s} allowed {s['allowed']}")
    return 0


__all__ = ["build", "judge", "sha256"]

if __name__ == "__main__":
    sys.exit(main())
