"""B-list skills, feasibility part (b): are the trained postures' targets physically holdable? (b:feasibility)

A scripted, open-loop joint trajectory through robotd's own path (the Stage-1 gate duck, both servo
models, DR ducks, the low-pass and voltage adaptation, the pinned scale 0.9): HOME → the target posture
in ``ramp`` s (smoothstep), held ``hold`` s, back to HOME in 0.5 s, then v5 velstand takes over (a
one-shot skill window whose "network" plays the script). No feedback at all — if the target is
holdable open-loop, a trained network (with feedback) has room; if it tips over open-loop, the target
must move. Postures (``bsk_contract``): ``BOW`` (25° nose-down, 15 mm lower, head low looking ahead)
and ``LIE`` (30°, 30 mm, beak toward the floor).

    python -m duckmind.skilltrain.bsk_pose run --servo mjcf --n 30 --out DIR
    python -m duckmind.skilltrain.bsk_pose summarize --out DIR
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from duckmind.skilltrain import bsk_contract as B
from duckmind.skilltrain import bsk_probe as P

POSTURES = {"bow": B.BOW, "lie": B.LIE}
HOLD_S = {"bow": 1.2, "lie": 3.0}
RETURN_S = 0.5


class ScriptNet:
    """Stands in for an ONNX network in robotd's port: returns the raw action that makes
    robotd's target = the scripted joint target (before the low-pass)."""

    def __init__(self, duck: Any, pose: tuple[float, ...], ramp: float, hold: float) -> None:
        self.duck, self.pose = duck, np.asarray(pose, float)
        self.ramp, self.hold = ramp, hold
        self.t0: float | None = None
        self.recurrent = False

    def reset(self) -> None:
        self.t0 = None

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        now = self.duck.now()
        if self.t0 is None:
            self.t0 = now
        t = now - self.t0
        home = np.asarray(B.HOME, float)
        if t < self.ramp:
            a = t / self.ramp
        elif t < self.ramp + self.hold:
            a = 1.0
        else:
            a = max(0.0, 1.0 - (t - self.ramp - self.hold) / RETURN_S)
        a = a * a * (3 - 2 * a)  # smoothstep
        q = home + a * (self.pose - home)
        mult = 7.4 / min(max(float(self.duck.battery_v), 6.0), 9.5)
        return ((q - home) / (B.ACTION_SCALE * mult)).astype(np.float32)


def run_trial(name: str, *, servo: str, rep: int, delay: int | None = None) -> dict[str, Any]:
    from duckmind.skilltrain import dr as DR
    from duckmind.skilltrain.table import SkillDef
    from duckmind.world.headless_duck import policy_dir

    began = time.perf_counter()
    post = POSTURES[name]
    hold = HOLD_S[name]
    seed = P.seed_of(f"pose/{name}", rep)
    draw = DR.draw(seed, "recipe" if servo == "bam" else "physical")
    rng = np.random.default_rng(seed + 11)
    pol = policy_dir()
    window = post.ramp_s + hold + RETURN_S
    skill = SkillDef("scripted", str(pol / "velstand.onnx"), window, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0, False,
                     B.ACTION_SCALE, B.GAIN_RATIO)  # fmt: skip
    Duck = P._duck_class()
    duck = Duck(
        [skill],
        servo=servo,
        draw=draw,
        head_alpha=1.0,
        keyframe="SIT",
        yaw=float(rng.uniform(-math.pi, math.pi)),
    )
    if delay is not None:
        from collections import deque

        duck.lag_steps = int(delay)
        duck._lag = deque(maxlen=duck.lag_steps + 1)
    duck.port.controller.policy.skills[0] = ScriptNet(duck, post.pose(), post.ramp_s, hold)
    row: dict[str, Any] = {"key": f"pose/{name}/{servo}/{rep}", "posture": name, "servo": servo, "rep": rep, "seed": seed,
                           "dr": draw.to_json(), "delay_steps": None, "excluded": None}  # fmt: skip
    ok, _ = duck.stand(timeout=10.0, hold=1.0)
    if not ok:
        row["excluded"] = "did not stand up after the seated boot"
        return row
    row["delay_steps"] = int(duck.lag_steps)
    P._run(duck, float(rng.uniform(1.0, 1.5)), lambda _t: duck.move(0.0))
    t0 = duck.now()
    duck.do("scripted")
    t_end = P._until_label_leaves(duck, ("scripted",), window + 2.0)
    if t_end is None:
        row["excluded"] = "the script did not hand back"
        return row
    P._run(duck, P.HANDBACK_S, lambda _t: duck.move(0.0))
    row.update(P.bow_metrics(duck, t0, t_end))
    tr, ex = P._arrays(duck)
    hold_sel = (tr["t"] > t0 + post.ramp_s + 0.2) & (tr["t"] <= t0 + post.ramp_s + hold)
    pre = np.flatnonzero(tr["t"] <= t0)[-5:]
    row["hold_pitch_deg"] = round(float(np.median(ex[hold_sel, 7]) - np.median(ex[pre, 7])), 2)
    row["hold_drop_mm"] = round(
        1000 * float(np.median(tr["pos"][pre, 2]) - np.median(tr["pos"][hold_sel, 2])), 1
    )
    row["hold_beak_mm"] = round(1000 * float(np.median(ex[hold_sel, 1])), 1)
    row["hold_leg_tau_p95"] = round(
        float(np.percentile(np.abs(tr["tau_pk"][hold_sel][:, list(P.LEGS)]), 95)), 4
    )
    row["hold_footfalls"] = int(tr["td"][hold_sel].sum())
    row["target"] = {"pitch_deg": post.pitch_deg, "drop_mm": 1000 * post.drop_m}
    row["wall_s"] = round(time.perf_counter() - began, 3)
    return row


def run(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"rows-{args.servo}.jsonl"
    done = P._done(path)
    with path.open("a") as f:
        for name in POSTURES:
            valid, rep = 0, 0
            while valid < args.n and rep < 3 * args.n:
                key = f"pose/{name}/{args.servo}/{rep}"
                if key not in done:
                    row = run_trial(name, servo=args.servo, rep=rep, delay=args.delay_steps)
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    done[key] = row
                valid += 0 if done[key]["excluded"] else 1
                rep += 1
            print(f"[{args.servo}] {name}: {valid} valid", flush=True)
    return 0


def summarize(args: argparse.Namespace) -> int:
    out = Path(args.out)
    lines = []
    for servo in ("mjcf", "bam"):
        rows = [r for r in P._done(out / f"rows-{servo}.jsonl").values() if not r["excluded"]]
        for name in POSTURES:
            rs = [r for r in rows if r["posture"] == name]
            if not rs:
                continue
            n = len(rs)
            parts = [
                f"falls in {P.rate(sum(r['fell_in'] for r in rs), n)}",
                f"hand-back {P.rate(sum(r['fell_after'] for r in rs), n)}",
            ]
            for k in (
                "hold_pitch_deg",
                "hold_drop_mm",
                "hold_beak_mm",
                "hold_leg_tau_p95",
                "hold_footfalls",
                "head_tau_p95",
                "drift_m",
            ):
                parts.append(f"{k} {P._q([r.get(k) for r in rs])}")
            lines.append(f"{name} · {servo} · n = {n}: " + "; ".join(parts))
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bsk_pose")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--servo", choices=("mjcf", "bam"), required=True)
    r.add_argument("--n", type=int, default=30)
    r.add_argument("--out", required=True)
    r.add_argument("--delay-steps", type=int, default=None)
    s = sub.add_parser("summarize")
    s.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return run(args) if args.cmd == "run" else summarize(args)


if __name__ == "__main__":
    raise SystemExit(main())
