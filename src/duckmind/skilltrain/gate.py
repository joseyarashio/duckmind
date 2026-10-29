"""The Mac gate runner (tier 2 of the training plan §6): resumable, one servo model per process.

    python -m duckmind.skilltrain.gate run --servo mjcf --n 30 --out DIR \\
        [--table skills.json | --dummy] [--baselines] [--semantics] [--only SUBSTR ...]
    python -m duckmind.skilltrain.gate summarize --out DIR
    python -m duckmind.skilltrain.gate toml --table skills.json     # the robotd.toml section
    python -m duckmind.skilltrain.gate caps --out DIR               # hourly caps from strain

``run`` appends one JSON row per trial to ``DIR/rows-<servo>.jsonl`` (a key already there is
skipped, so a stopped run resumes), keeps the compacted trace of rep 0 of every condition in
``DIR/traces-<servo>.jsonl``, and rewrites ``DIR/summary-<servo>.json``. Before a trial and
every 30 s of wall time it reads the owner's :8765 realtime (``GET /api/body`` only) and pauses
5 minutes while it is under 0.97 (WORK-RULES §5). ``summarize`` merges both servo models into
``DIR/summary.json`` and ``DIR/report.md``.

Run with ``uv run --python 3.12 --with onnxruntime==1.27.0 --with mujoco==3.10.0 --with numpy``
plus, for ``--servo bam``, ``--with "better-actuator-models @
git+https://github.com/Rhoban/bam.git@62bd8ce12154340be97e06f7f41a0ca8f116d967"``, and
``DUCKMIND_MICRODUCK_MJCF`` pointing at the official MJCF folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from duckmind import paths
from duckmind.skilltrain import metrics as M
from duckmind.skilltrain import programs as P
from duckmind.skilltrain import table as T

OWNER_BODY = "http://127.0.0.1:8765/api/body"
REALTIME_MIN = 0.97
PAUSE_S = 300.0
CHECK_EVERY_S = 30.0


def owner_realtime() -> float | None:
    """The owner's simulator's realtime factor, read the way WORK-RULES allows (GET only)."""
    try:
        with urllib.request.urlopen(OWNER_BODY, timeout=3) as answer:
            body = json.load(answer)
        value = (body.get("monitor") or {}).get("realtime")
        return None if value is None else float(value)
    except Exception:
        return None


class Guard:
    def __init__(self, enabled: bool, log: Any = print) -> None:
        self.enabled = enabled
        self.log = log
        self.checked = -1e9
        self.pauses = 0
        self.readings: list[float] = []

    def wait(self) -> None:
        if not self.enabled or time.monotonic() - self.checked < CHECK_EVERY_S:
            return
        while True:
            rt = owner_realtime()
            self.checked = time.monotonic()
            if rt is not None:
                self.readings.append(rt)
            if rt is None or rt >= REALTIME_MIN:
                return
            self.pauses += 1
            self.log(f"[guard] :8765 realtime {rt:.3f} < {REALTIME_MIN} — pausing {PAUSE_S:.0f} s")
            time.sleep(PAUSE_S)


def _sha(path: str | None) -> str | None:
    if not path or not Path(path).is_file():
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _entries(args: argparse.Namespace, policies: Path) -> list[T.Entry]:
    walk = str(policies / "velstand.onnx")
    if args.table:
        return T.load(args.table)
    return T.dummy(walk)


def _done(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            keys.add(json.loads(line)["key"])
        except (ValueError, KeyError):
            continue  # a torn last line: the trial reruns
    return keys


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def cmd_run(args: argparse.Namespace) -> int:
    policies = Path(args.policies).expanduser() if args.policies else paths.policy_dir()
    entries = _entries(args, policies)
    bad = [p for p in T.problems(entries) if "never chains" not in p]
    if bad:
        print("table problems:\n  " + "\n  ".join(bad), file=sys.stderr)
        return 2
    defs = [e.skill for e in entries]
    conds: list[P.Condition] = []
    if not args.no_skills:
        conds += P.conditions_for(entries, head_alpha=args.head_alpha)
        if not args.table:
            # the dummy table: the gate conditions for the startle-window stand-in only
            conds = [c for c in conds if c.skill == "dummy_stand"]
        if args.table:
            conds += P.sequence_conditions(entries)
    if args.baselines:
        conds += P.baseline_conditions(head_alpha=args.head_alpha)
    if args.semantics:
        conds += P.semantics_conditions(n=args.semantics_n)
    if args.only:
        conds = [c for c in conds if any(s in c.key for s in args.only)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / f"rows-{args.servo}.jsonl"
    traces_path = out / f"traces-{args.servo}.jsonl"
    done = _done(rows_path)
    guard = Guard(not args.no_guard)
    plan = [(c, rep) for c in conds for rep in range(c.n if c.n is not None else args.n)]
    todo = [(c, rep) for c, rep in plan if f"{c.key}/{args.servo}/{rep}" not in done]
    meta = {
        "servo": args.servo,
        "n": args.n,
        "head_alpha": args.head_alpha,
        "policies": str(policies),
        "velstand_sha256": _sha(str(policies / "velstand.onnx")),
        "sitstand_sha256": _sha(str(policies / "alpha_sitstand.onnx")),
        "skills": [{"name": d.name, "path": d.path, "sha256": _sha(d.path)} for d in defs],
        "conditions": [c.to_json() for c in conds],
        "planned": len(plan),
        "already_done": len(plan) - len(todo),
    }
    (out / f"meta-{args.servo}.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(
        f"{args.servo}: {len(todo)} of {len(plan)} trials to run ({len(conds)} conditions)",
        flush=True,
    )
    began = time.monotonic()
    stats = {"wall_trials": 0.0, "ran": 0}

    def run_list(items: list[tuple[P.Condition, int]], label: str) -> None:
        for i, (c, rep) in enumerate(items):
            guard.wait()
            try:
                row = P.run_trial(
                    c,
                    servo=args.servo,
                    rep=rep,
                    skills=defs,
                    policies=str(policies),
                    mjcf_dir=args.mjcf_dir,
                    dr_preset=args.dr_preset,
                    keep_trace=(rep == 0),
                )
            except Exception as error:  # one broken trial must not stop the others
                row = {
                    "key": f"{c.key}/{args.servo}/{rep}",
                    "cond": c.to_json(),
                    "servo": args.servo,
                    "rep": rep,
                    "excluded": f"error: {type(error).__name__}: {error}",
                }
            trace = row.pop("trace", None)
            with rows_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if trace is not None:
                with traces_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": row["key"], "trace": trace}) + "\n")
            stats["wall_trials"] += float(row.get("wall_s") or 0.0)
            stats["ran"] += 1
            if (i + 1) % 10 == 0 or i + 1 == len(items):
                el = time.monotonic() - began
                left = el / max(stats["ran"], 1) * (len(items) - i - 1)
                m = row.get("metrics") or {}
                print(
                    f"  {label} {i + 1}/{len(items)}  {row['key']}  fell={m.get('fell')}  "
                    f"elapsed {el:.0f} s, ~{left:.0f} s left",
                    flush=True,
                )

    run_list(todo, "run")
    # top up: a duck excluded before the skill (did not stand up, down before it) is replaced
    # by the next rep of its condition, until each condition has n valid trials (≤ 10 extra)
    for _ in range(10):
        valid: dict[str, int] = {}
        reps: dict[str, int] = {}
        for r in _rows(rows_path):
            key = r["cond"]["key"]
            reps[key] = max(reps.get(key, -1), int(r["rep"]))
            valid[key] = valid.get(key, 0) + (0 if r.get("excluded") else 1)
        extra = []
        for c in conds:
            want = c.n if c.n is not None else args.n
            short = want - valid.get(c.key, 0)
            first = reps.get(c.key, -1) + 1
            if short > 0 and first < want + 10:
                extra += [(c, rep) for rep in range(first, min(first + short, want + 10))]
        if not extra:
            break
        run_list(extra, "top-up")
    summary = summarize_rows(_rows(rows_path))
    summary["guard"] = {
        "pauses": guard.pauses,
        "realtime_min": min(guard.readings, default=None),
        "readings": len(guard.readings),
    }
    summary["meta"] = meta
    (out / f"summary-{args.servo}.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False)
    )
    print(
        f"done in {time.monotonic() - began:.0f} s (trials {stats['wall_trials']:.0f} s); "
        f"guard pauses "
        f"{guard.pauses}",
        flush=True,
    )
    return 0


def summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(r["cond"]["key"], []).append(r)
    out: dict[str, Any] = {"conditions": {}}
    for key, rs in sorted(by.items()):
        cond = rs[0]["cond"]
        fam = cond.get("family", "")
        level = cond.get("level")
        high = fam in ("hop_back", "freeze", "abort") or (fam == "startle" and (level or 0) >= 1.0)
        s = M.summarize(rs, fall_gate=M.FALL_GATE["high" if high else "low"])
        s["cond"] = cond
        s["excluded_reasons"] = sorted({str(r.get("excluded")) for r in rs if r.get("excluded")})
        s["sequences"] = sorted(
            {" → ".join(r.get("sequence") or []) for r in rs if r.get("sequence")}
        )
        s["refreshed_mean"] = (
            round(sum(r.get("refreshed", 0) for r in rs) / len(rs), 2) if rs else None
        )
        s["wall_s_mean"] = round(sum(float(r.get("wall_s") or 0) for r in rs) / len(rs), 3)
        s["sim_s_mean"] = round(sum(float(r.get("sim_s") or 0) for r in rs) / len(rs), 3)
        out["conditions"][key] = s
    return out


def cmd_summarize(args: argparse.Namespace) -> int:
    out = Path(args.out)
    merged: dict[str, Any] = {"servo": {}}
    for servo in ("mjcf", "bam"):
        rows = _rows(out / f"rows-{servo}.jsonl")
        if rows:
            merged["servo"][servo] = summarize_rows(rows)
    (out / "summary.json").write_text(json.dumps(merged, indent=1, ensure_ascii=False))
    (out / "report.md").write_text(report_md(merged), encoding="utf-8")
    print(f"wrote {out / 'summary.json'} and {out / 'report.md'}")
    return 0


def _fmt_rate(r: dict[str, Any] | None) -> str:
    if not r or not r.get("n"):
        return "—"
    lo, hi = r["wilson95"]
    return f"{r['k']}/{r['n']} [{lo * 100:.1f}–{hi * 100:.1f}%]"


def _fmt_dist(d: dict[str, Any] | None, digits: int = 2, of: int | None = None) -> str:
    """p50 (p10–p90); "[k of n]" when only k of the n ducks have the number (the others never
    reached it: no motion, never still, never re-seated)."""
    if not d or not d.get("n"):
        return "— (none)" if of else "—"
    text = f"{d['p50']:.{digits}f} ({d['p10']:.{digits}f}–{d['p90']:.{digits}f})"
    if of is not None and d["n"] < of:
        text += f" [{d['n']} of {of}]"
    return text


def report_md(merged: dict[str, Any]) -> str:
    lines = ["# Stage-1 skill gates — Mac headless (tier 2)", ""]
    lines.append(
        "Rates k/n [Wilson 95 %]; distributions p50 (p10–p90) over ducks. "
        "Visibility = the plan's §6 checks for the condition's gate family."
    )
    lines.append("")
    for servo, s in merged.get("servo", {}).items():
        lines.append(f"## servo model: {servo}")
        lines.append("")
        lines.append(
            "| condition | n | performed | falls | t45 | servo hits upright | "
            "tau p95 leg/head N·m | stall | acc p99 | key numbers |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for key, c in s.get("conditions", {}).items():
            if not c.get("n"):
                lines.append(
                    f"| {key} | 0 | excluded: {'; '.join(c.get('excluded_reasons', []))} "
                    "| | | | | | | |"
                )
                continue
            d = c["dist"]
            st = c["strain"]
            fam = c["cond"].get("family")
            posture = c["cond"].get("posture")
            if posture == "walk":
                key_nums = (
                    f"still {_fmt_dist(d['still_s'], of=c['n'])} s, travel "
                    f"{_fmt_dist(d['travel_cm'], 1)} cm, resume ±10% "
                    f"{_fmt_rate(c.get('resume_within_10pct'))}"
                )
            elif fam in ("shake",):
                key_nums = (
                    f"roll osc {_fmt_dist(d['roll_osc_s'])} s, roll amp "
                    f"{_fmt_dist(d['roll_amp_deg'], 1)}°, yaw amp "
                    f"{_fmt_dist(d['yaw_amp_rad'])} rad @ {_fmt_dist(d['yaw_hz'], 1)} Hz"
                )
            elif fam in ("stretch_tall",):
                key_nums = (
                    f"rise {_fmt_dist(d['rise_max_mm'], 1)} mm, cam pitch max "
                    f"{_fmt_dist(d['cam_pitch_max_deg'], 1)}°, held "
                    f"{_fmt_dist(d['stretch_tall_hold_s'])} s"
                )
            elif fam in ("stretch_sit",) or posture == "sit":
                key_nums = (
                    f"rise {_fmt_dist(d['rise_max_mm'], 1)} mm, cam pitch max "
                    f"{_fmt_dist(d['cam_pitch_max_deg'], 1)}°, reseat "
                    f"{_fmt_dist(d['reseat_s'], of=c['n'])} s"
                )
            elif fam == "hop_back":
                key_nums = (
                    f"back in 0.7 s {_fmt_dist(d['back_070_cm'], 1)} cm, yaw "
                    f"{_fmt_dist(d['yaw_change_deg'], 1)}°"
                )
            else:
                key_nums = (
                    f"first motion {_fmt_dist(d['first_motion_s'], of=c['n'])} s, neck back "
                    f"{_fmt_dist(d['retract_025_rad'], 3)} rad, dip "
                    f"{_fmt_dist(d['dip_030_mm'], 1)} mm, hold ±"
                    f"{_fmt_dist(d['hold_dev_mm'], 1)} mm"
                )
            lines.append(
                f"| {key} | {c['n']} | {_fmt_rate(c['performed'])} | {_fmt_rate(c['falls'])} | "
                f"{_fmt_rate(c['t45'])} | {_fmt_rate(c['servo_hits_upright'])} | "
                f"{st['tau_leg_p95']}/{st['tau_head_p95']} | {st['stall_frac']} | "
                f"{st['acc_p99']} | {key_nums} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def cmd_caps(args: argparse.Namespace) -> int:
    """The hourly caps from a trained table's strain gates (bam; the worst posture per skill)
    → ``DIR/caps.json`` (duck-body's :class:`~duckmind.skilltrain.guard.SkillGuard` rules)."""
    from duckmind.skilltrain import guard as G

    out = Path(args.out)
    merged = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    bam = (merged.get("servo", {}).get("bam") or {}).get("conditions", {})
    worst: dict[str, dict[str, float]] = {}
    for key, c in bam.items():
        cond = c.get("cond") or {}
        name = cond.get("skill")
        if (
            cond.get("kind") != "skill"
            or not name
            or not key.startswith("skill/")
            or not c.get("n")
        ):
            continue
        cur = {
            "tau_leg_p95": float(c["strain"]["tau_leg_p95"] or 0.0),
            "tau_head_p95": float(c["strain"]["tau_head_p95"] or 0.0),
            "stall_frac": float(c["strain"]["stall_frac"] or 0.0),
            "servo_hits_upright_k": float(c["servo_hits_upright"]["k"]),
        }
        prev = worst.get(name)
        worst[name] = cur if prev is None else {k: max(prev[k], cur[k]) for k in cur}
    rules = G.caps_from_strain(G.DEFAULT_RULES, worst)
    caps = {
        name: {
            "per_hour": r.per_hour,
            "min_gap_s": r.min_gap_s,
            "note": r.note,
            "strain": worst.get(name),
        }
        for name, r in rules.items()
    }
    (out / "caps.json").write_text(json.dumps(caps, indent=1, ensure_ascii=False))
    for name, c in caps.items():
        print(f"{name:16s} {c['per_hour']!s:>4}/h  gap {c['min_gap_s']:5.0f} s  {c['note']}")
    return 0


def cmd_toml(args: argparse.Namespace) -> int:
    entries = T.load(args.table)
    for p in T.problems(entries):
        print("# note: " + p, file=sys.stderr)
    sys.stdout.write(T.toml_section(entries))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m duckmind.skilltrain.gate")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run gate trials for one servo model")
    run.add_argument("--servo", choices=("mjcf", "bam"), required=True)
    run.add_argument("--n", type=int, default=30)
    run.add_argument("--out", required=True)
    run.add_argument("--table", help="a skills.json (duckmind-skill-table-1); default: the dummy")
    run.add_argument("--dummy", action="store_true", help="the v5 stand-in table (default)")
    run.add_argument("--baselines", action="store_true", help="the v5 baselines too")
    run.add_argument("--semantics", action="store_true", help="robotd skill-semantics checks")
    run.add_argument("--semantics-n", type=int, default=5)
    run.add_argument("--no-skills", action="store_true", help="skip the table's gate conditions")
    run.add_argument("--only", nargs="*", help="keep conditions whose key contains any of these")
    run.add_argument("--head-alpha", type=float, default=1.0)
    run.add_argument("--dr-preset", choices=("none", "physical", "recipe"))
    run.add_argument("--policies")
    run.add_argument("--mjcf-dir")
    run.add_argument("--no-guard", action="store_true")
    run.set_defaults(func=cmd_run)
    summ = sub.add_parser("summarize", help="merge both servo models into summary.json/report.md")
    summ.add_argument("--out", required=True)
    summ.set_defaults(func=cmd_summarize)
    caps = sub.add_parser("caps", help="hourly caps from a trained run's strain gates (bam)")
    caps.add_argument("--out", required=True)
    caps.set_defaults(func=cmd_caps)
    toml = sub.add_parser("toml", help="print the robotd.toml skill section of a table")
    toml.add_argument("--table", required=True)
    toml.set_defaults(func=cmd_toml)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
