"""s1b:fix: the Stage-1 skills across servo command delays — the latency map.

The owner decides whether our simulator gets training's servo command delay (the proposal is
20 ms). The map says whether the skills are robust to the delay or work only in a narrow window.
Each cell is one table entry × posture × servo model × delay, n DR ducks (a new domain-
randomisation draw per trial; the same draw at every delay, so the delays are compared on the
same ducks: :func:`duckmind.skilltrain.programs.seed_of` does not depend on the delay).

**The delay.** Robotd's joint targets reach the servos ``d`` ms late (whole 5-ms physics steps):

* ``fixed`` — every target exactly ``d`` ms late (what a simulator setting would do; the
  ``bam`` profile's own drawn 15–30 ms is replaced by ``d``);
* ``jitter`` — training's model: mjlab's ``DelayBuffer`` with ``update_period = 0`` and
  ``hold_prob = 0`` draws a new lag for every physics step, uniform in ``[lo, hi]`` steps (the
  velocity recipe's ``delay_min_lag = 3``, ``delay_max_lag = 6``: 15–30 ms, mean 22.5 ms). The
  Mac gates so far fixed one lag per duck; this is the reproduction of training's own model.

Both servo models (``mjcf`` = the owner's simulator's XML servos, physical DR; ``bam`` =
training's BAM XL330 with the recipe's DR, observation noise and IMU error), the v5 baselines
too (v5 across the delays: the same draws, the same metrics), and every trial's seated-boot rise
(sitstand across the delays: a boot rise that fails is an excluded row with its reason).

    python -m duckmind.skilltrain.gt_latency run --servo mjcf --table skills.json --out DIR \\
        [--delays 0,5,10,15,20,25,30,40] [--jitter 3-6] [--n 30] [--baselines] [--only ...]
    python -m duckmind.skilltrain.gt_latency analyze --out DIR [--table skills.json]

``run`` appends one row per trial to ``DIR/rows-<servo>-<label>.jsonl`` (``label`` = ``d20`` for
a fixed 20 ms, ``j3-6`` for the jitter) and skips keys already there (resumable). ``analyze``
writes ``DIR/map.json`` and ``DIR/map.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from duckmind import paths
from duckmind.skilltrain import gate as GATE
from duckmind.skilltrain import metrics as M
from duckmind.skilltrain import programs as P
from duckmind.skilltrain import table as T
from duckmind.skilltrain.skillduck import SkillDuck

PHYSICS_STEP_MS = 5.0
DELAYS_MS = (0, 5, 10, 15, 20, 25, 30, 40)
TRAINING_JITTER = (3, 6)  # microduck_constants _BAM_ACTUATOR_KWARGS delay_min_lag / max_lag
OK_RATE, OK_FALLS = 0.90, 1.0 / 29.0  # gt_verdict's "ok" cell (performed ≥ 90 %, falls ≤ 1/30)
WEAK_RATE, WEAK_FALLS = 0.70, 0.10


def steps_of(ms: float) -> int:
    steps = float(ms) / PHYSICS_STEP_MS
    if steps < 0 or abs(steps - round(steps)) > 1e-9:
        raise ValueError(f"a command delay of {ms} ms is not a whole number of 5-ms steps")
    return int(round(steps))


def label_of(mode: str, lo: int, hi: int) -> str:
    return f"d{int(round(hi * PHYSICS_STEP_MS))}" if mode == "fixed" else f"j{lo}-{hi}"


def latency_duck(mode: str, lo: int, hi: int) -> type[SkillDuck]:
    """A :class:`SkillDuck` whose servos (either model) act on robotd's targets late: exactly
    ``hi`` physics steps (``fixed``), or a lag redrawn every physics step in ``[lo, hi]``
    (``jitter``, training's DelayBuffer)."""
    if mode not in ("fixed", "jitter") or lo < 0 or hi < lo:
        raise ValueError(f"delay model {mode} {lo}-{hi}")

    class LatencyDuck(SkillDuck):
        lag_spec = (mode, lo, hi)

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.lag_steps = hi
            self._lag = deque(maxlen=hi + 1)
            self._lag_rng = np.random.default_rng(int(self.draw.seed) + 104_729)

        def _delayed(self) -> np.ndarray:
            if mode == "fixed" or len(self._lag) <= 1:
                return self._lag[0]
            lag = int(self._lag_rng.integers(lo, hi + 1))
            return self._lag[-1 - min(lag, len(self._lag) - 1)]

        def info(self) -> dict[str, Any]:
            out = super().info()
            out["lag_model"] = {"mode": mode, "lo_steps": lo, "hi_steps": hi}
            return out

    LatencyDuck.__name__ = f"LatencyDuck[{label_of(mode, lo, hi)}]"
    return LatencyDuck


def _conditions(entries: Sequence[T.Entry], args: argparse.Namespace) -> list[P.Condition]:
    conds = P.conditions_for(entries) + P.sequence_conditions(entries)
    if getattr(args, "live", False):
        conds += P.live_conditions(entries)
    if args.baselines:
        conds += P.baseline_conditions()
    if args.only:
        conds = [c for c in conds if any(s in c.key for s in args.only)]
    return conds


def cmd_run(args: argparse.Namespace) -> int:
    policies = Path(args.policies).expanduser() if args.policies else paths.policy_dir()
    entries = T.load(args.table)
    defs = [e.skill for e in entries]
    conds = _conditions(entries, args)
    models: list[tuple[str, int, int]] = [
        ("fixed", steps_of(ms), steps_of(ms)) for ms in _ints(args.delays)
    ]
    if args.jitter:
        lo, _, hi = args.jitter.partition("-")
        models.append(("jitter", int(lo), int(hi)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    guard = GATE.Guard(not args.no_guard)
    began = time.monotonic()
    total = len(models) * len(conds) * args.n
    ran = 0
    meta = {
        "servo": args.servo,
        "n": args.n,
        "table": str(args.table),
        "skills": [{"name": d.name, "path": d.path, "sha256": GATE._sha(d.path)} for d in defs],
        "velstand_sha256": GATE._sha(str(policies / "velstand.onnx")),
        "sitstand_sha256": GATE._sha(str(policies / "alpha_sitstand.onnx")),
        "models": [
            {"mode": m, "lo": lo, "hi": hi, "label": label_of(m, lo, hi)} for m, lo, hi in models
        ],
        "conditions": [c.key for c in conds],
        "dr_preset": "physical" if args.servo == "mjcf" else "recipe",
    }
    (out / f"meta-{args.servo}.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    for mode, lo, hi in models:
        label = label_of(mode, lo, hi)
        P.SkillDuck = latency_duck(mode, lo, hi)  # type: ignore[misc]
        rows_path = out / f"rows-{args.servo}-{label}.jsonl"
        done = GATE._done(rows_path)
        todo = [
            (c, rep)
            for c in conds
            for rep in range(args.n)
            if f"{c.key}/{args.servo}/{label}/{rep}" not in done
        ]
        print(f"{args.servo} {label}: {len(todo)} trials to run", flush=True)
        for c, rep in todo:
            guard.wait()
            try:
                row = P.run_trial(
                    c,
                    servo=args.servo,
                    rep=rep,
                    skills=defs,
                    policies=str(policies),
                    mjcf_dir=args.mjcf_dir,
                    keep_trace=False,
                )
            except Exception as error:  # one broken trial must not stop the map
                row = {
                    "cond": c.to_json(),
                    "servo": args.servo,
                    "rep": rep,
                    "excluded": f"error: {type(error).__name__}: {error}",
                }
            row["key"] = f"{c.key}/{args.servo}/{label}/{rep}"
            row["delay"] = {"mode": mode, "lo_steps": lo, "hi_steps": hi, "label": label}
            with rows_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            ran += 1
            if ran % 50 == 0:
                el = time.monotonic() - began
                print(f"  {ran}/{total} {row['key']} ({el:.0f} s)", flush=True)
    print(
        f"done: {ran} trials in {time.monotonic() - began:.0f} s; guard pauses {guard.pauses}, "
        f"realtime min {min(guard.readings, default=None)}",
        flush=True,
    )
    return 0


def _ints(text: str) -> list[int]:
    return [int(x) for x in str(text).split(",") if x.strip()]


# ── analysis ─────────────────────────────────────────────────────────────────────────────────

FAMILY_NUMBERS = {
    "startle": ("dip_030_mm", "retract_025_rad", "first_motion_s", "hold_dev_mm"),
    "freeze": ("dip_030_mm", "retract_025_rad", "hold_dev_mm", "bob_mm"),
    "walk": ("still_s", "travel_cm", "dip_030_mm", "retract_025_rad", "yaw_change_deg"),
    "hop_back": ("back_070_cm", "yaw_change_deg", "footfalls_skill"),
    "shake": ("roll_amp_deg", "roll_osc_s", "yaw_amp_rad", "yaw_hz"),
    "stretch_tall": ("rise_max_mm", "cam_pitch_max_deg", "stretch_tall_hold_s"),
    "stretch_sit": ("rise_max_mm", "cam_pitch_max_deg", "reseat_s"),
    "release": ("still_s", "bob_mm"),
    "abort": ("still_s", "bob_mm"),
}


def performed(row: Mapping[str, Any]) -> bool | None:
    """The trial's visibility verdict under the current definitions (re-scored from its metrics:
    hop_back from walking by the hop checks, as gt_verdict judges it)."""
    cond = row["cond"]
    fam, posture = cond.get("family", ""), cond.get("posture", "")
    if fam == "hop_back" and posture == "walk":
        return M.visibility("hop_back", cond.get("level"), "stand", row["metrics"])["performed"]
    gate_fam = M.gate_family(fam, posture) if cond.get("kind") == "skill" else fam
    return M.visibility(gate_fam, cond.get("level"), posture, row["metrics"])["performed"]


def cell(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if not r.get("excluded")]
    n = len(ok)
    out: dict[str, Any] = {
        "n": n,
        "excluded": len(rows) - n,
        "boot_rise_failed": sum(
            1 for r in rows if "did not stand up" in str(r.get("excluded") or "")
        ),
    }
    if not n:
        return out
    ms = [r["metrics"] for r in ok]
    perf = [performed(r) for r in ok]
    gated = [p for p in perf if p is not None]
    out["performed"] = M.rate(sum(1 for p in gated if p), len(gated)) if gated else None
    out["falls"] = M.rate(sum(1 for m in ms if m["fell"]), n)
    out["falls_in_skill"] = M.rate(sum(1 for m in ms if m["fell_in_skill"]), n)
    fallen = [m for m in ms if m["fell"]]
    out["righted_8s"] = M.rate(sum(1 for m in fallen if m.get("righted_8s")), len(fallen))
    cond = ok[0]["cond"]
    fam = cond.get("family", "")
    key = "walk" if cond.get("posture") == "walk" and fam in ("startle", "freeze") else fam
    if cond.get("kind") == "v5":
        key = "walk" if cond.get("posture") == "walk" else fam
    numbers = FAMILY_NUMBERS.get(key, ())
    out["numbers"] = {k: M.dist(m.get(k) for m in ms) for k in (*numbers, "tilt_max_deg")}
    leg = np.concatenate([np.asarray(m["tau_leg"], float) for m in ms])
    head = np.concatenate([np.asarray(m["tau_head"], float) for m in ms])
    out["strain"] = {
        "tau_leg_p95": round(float(np.percentile(leg, 95)), 4) if leg.size else None,
        "tau_head_p95": round(float(np.percentile(head, 95)), 4) if head.size else None,
    }
    return out


def status(c: Mapping[str, Any] | None) -> str:
    """ok | weak | miss | falls | none — gt_verdict's cell rule."""
    if not c or not c.get("n"):
        return "none"
    falls = c["falls"]["k"] / c["falls"]["n"]
    if falls >= 0.5:
        return "falls"
    perf = c.get("performed")
    rate = 1.0 if perf is None or not perf.get("n") else perf["k"] / perf["n"]
    if rate >= OK_RATE and falls <= OK_FALLS:
        return "ok"
    if rate >= WEAK_RATE and falls <= WEAK_FALLS:
        return "weak"
    return "miss"


def load_all(out: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """``{(servo, label): rows}`` from every ``rows-<servo>-<label>.jsonl`` in ``out``."""
    found: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for path in sorted(out.glob("rows-*-*.jsonl")):
        _, servo, label = path.stem.split("-", 2)
        found[(servo, label)] = GATE._rows(path)
    return found


def _label_ms(label: str) -> float:
    if label.startswith("d"):
        return float(label[1:])
    lo, _, hi = label[1:].partition("-")
    return (float(lo) + float(hi)) / 2.0 * PHYSICS_STEP_MS


def analyze(out: Path) -> dict[str, Any]:
    data = load_all(out)
    labels = sorted({lab for _, lab in data}, key=lambda s: (s.startswith("j"), _label_ms(s)))
    servos = sorted({s for s, _ in data})
    cells: dict[str, dict[str, dict[str, Any]]] = {}
    for (servo, label), rows in data.items():
        by: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            by.setdefault(r["cond"]["key"], []).append(r)
        for key, rs in by.items():
            cells.setdefault(key, {})[f"{servo}/{label}"] = cell(rs)
    windows: dict[str, dict[str, Any]] = {}
    for key, per in sorted(cells.items()):
        fixed = [lab for lab in labels if lab.startswith("d")]
        ok_all = [
            lab for lab in fixed if all(status(per.get(f"{s}/{lab}")) == "ok" for s in servos)
        ]
        by_servo = {
            s: [lab for lab in fixed if status(per.get(f"{s}/{lab}")) == "ok"] for s in servos
        }
        windows[key] = {
            "ok_both": [_label_ms(lab) for lab in ok_all],
            "ok_by_servo": {s: [_label_ms(lab) for lab in v] for s, v in by_servo.items()},
            "window_ms": _span([_label_ms(lab) for lab in ok_all]),
        }
    pooled = pooled_falls(data)
    return {
        "format": "duckmind-latency-map-1",
        "out": str(out),
        "labels": labels,
        "servos": servos,
        "cells": cells,
        "windows": windows,
        "pooled": pooled,
        "rules": {
            "ok": "performed ≥ 90 % and falls ≤ 1/30 (gt_verdict's cell rule)",
            "weak": "performed ≥ 70 % and falls ≤ 10 %",
            "window": "the contiguous fixed delays at which the cell is ok on both servo models",
        },
    }


def _span(values: Iterable[float]) -> list[float] | None:
    """The longest run of consecutive map delays in ``values`` as [lo, hi] ms."""
    vals = sorted(set(values))
    if not vals:
        return None
    grid = list(DELAYS_MS)
    best: list[float] = []
    run: list[float] = []
    for v in grid:
        if v in vals:
            run.append(v)
            if len(run) > len(best):
                best = list(run)
        else:
            run = []
    return [best[0], best[-1]] if best else [vals[0], vals[0]]


def pooled_falls(data: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Falls over every skill trial per servo × delay (and the v5 baselines separately), plus
    boot-rise failures (sitstand) per servo × delay."""
    out: dict[str, Any] = {}
    for (servo, label), rows in data.items():
        skill = [r for r in rows if r["cond"].get("kind") == "skill" and not r.get("excluded")]
        v5 = [r for r in rows if r["cond"].get("kind") == "v5" and not r.get("excluded")]
        boots = len(rows) - sum(1 for r in rows if "error" in str(r.get("excluded") or ""))
        failed = sum(1 for r in rows if "did not stand up" in str(r.get("excluded") or ""))
        out[f"{servo}/{label}"] = {
            "skill_falls": M.rate(sum(1 for r in skill if r["metrics"]["fell"]), len(skill)),
            "v5_falls": M.rate(sum(1 for r in v5 if r["metrics"]["fell"]), len(v5)),
            "boot_rise_failed": M.rate(failed, boots),
        }
    return out


def markdown(m: Mapping[str, Any]) -> str:
    labels = [lab for lab in m["labels"]]
    servos = m["servos"]
    head = "| condition | servo | " + " | ".join(labels) + " | ok window (both) |"
    lines = [
        "# Stage-1 latency map (s1b:fix)",
        "",
        "Each cell: status (ok / weak / miss / FALLS) · performed k/n · falls k/n. "
        "Fixed delays `dN` = N ms; `j3-6` = training's per-step 15–30 ms jitter. "
        "DR ducks (mjcf physical DR; bam the recipe's DR), the same draws at every delay.",
        "",
        head,
        "|" + "---|" * (len(labels) + 3),
    ]
    for key, per in m["cells"].items():
        for s in servos:
            row = []
            for lab in labels:
                c = per.get(f"{s}/{lab}")
                if not c or not c.get("n"):
                    row.append("—")
                    continue
                st = status(c)
                perf = c.get("performed")
                p = f"{perf['k']}/{perf['n']}" if perf else "–"
                row.append(f"{st.upper() if st == 'falls' else st} {p} · {c['falls']['k']}")
            win = m["windows"].get(key, {}).get("window_ms")
            lines.append(
                f"| {key} | {s} | "
                + " | ".join(row)
                + f" | {'—' if not win else f'{win[0]:g}–{win[1]:g} ms'} |"
            )
    lines += ["", "## Pooled falls (all skill trials) and boot rises", ""]
    lines.append("| servo/delay | skill falls | v5 falls | seated-boot rise failed |")
    lines.append("|---|---|---|---|")
    for k, v in sorted(m["pooled"].items()):
        lines.append(
            f"| {k} | {_fmt(v['skill_falls'])} | {_fmt(v['v5_falls'])} | "
            f"{_fmt(v['boot_rise_failed'])} |"
        )
    return "\n".join(lines) + "\n"


def _fmt(r: Mapping[str, Any] | None) -> str:
    if not r or not r.get("n"):
        return "—"
    lo, hi = r["wilson95"]
    return f"{r['k']}/{r['n']} [{lo * 100:.1f}–{hi * 100:.1f} %]"


def cmd_analyze(args: argparse.Namespace) -> int:
    out = Path(args.out)
    m = analyze(out)
    (out / "map.json").write_text(json.dumps(m, indent=1, ensure_ascii=False), "utf-8")
    (out / "map.md").write_text(markdown(m), "utf-8")
    print(f"wrote {out / 'map.json'} and {out / 'map.md'}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m duckmind.skilltrain.gt_latency")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run the map for one servo model")
    run.add_argument("--servo", choices=("mjcf", "bam"), required=True)
    run.add_argument("--table", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--n", type=int, default=30)
    run.add_argument("--delays", default=",".join(str(d) for d in DELAYS_MS))
    run.add_argument("--jitter", help="also training's per-step jitter LO-HI steps, e.g. 3-6")
    run.add_argument("--baselines", action="store_true")
    run.add_argument(
        "--live", action="store_true", help="also the live table's holds, continuations, entries"
    )
    run.add_argument("--only", nargs="*")
    run.add_argument("--policies")
    run.add_argument("--mjcf-dir")
    run.add_argument("--no-guard", action="store_true")
    run.set_defaults(func=cmd_run)
    an = sub.add_parser("analyze", help="map.json / map.md from the rows")
    an.add_argument("--out", required=True)
    an.set_defaults(func=cmd_analyze)
    args = ap.parse_args(argv)
    return int(args.func(args))


__all__ = [
    "DELAYS_MS",
    "TRAINING_JITTER",
    "analyze",
    "cell",
    "label_of",
    "latency_duck",
    "performed",
    "status",
    "steps_of",
]

if __name__ == "__main__":
    sys.exit(main())
