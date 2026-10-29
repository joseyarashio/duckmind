"""Mac gate (tier 2) of the forward hop-over network through robotd's own skill path.

hop:feasibility (2026-09-28). The same strips as the v5 probe (:mod:`.hop_probe`: 1–4 cm, sharp /
round, 4 cm deep), the same DR ducks (Stage-1 gate draws; the same seed = the same physical duck on
both servo models), and the network run the way the simulator will run it: a robotd
``[[policy.skill]]`` table (``hop_over_1..4`` → ``hopover.onnx``, :func:`hop_contract.skill_defs`)
through :class:`~duckmind.skilltrain.robotd_port.Robotd` (windows, unwind, the filter, the hand-back
to velstand), with an optional servo command delay (``--delay-ms``; the simulator's fixed 20 ms).

A trial: boot seated and rise; **walk** entry — set down 0.35–0.50 m before the strip and walk at
0.40 with the IMU heading hold until the trunk centre is 8–16 cm before the near face (the brain's
trigger point from camera + odometry, errors included in the spread), then ``robot.do
hop_over_<class>`` and a zero twist while it runs, then walking resumed for 2 s and a stop; **stand**
entry — set down 8–16 cm before the face, stand 0.5–1.0 s, the skill, then 2 s standing. Outcomes:
:func:`hop_probe.classify` over the whole trial (crossed / blocked / stuck / fall), plus falls split
into during-the-skill and after the hand-back, the skill's own window, leg torque and housing impacts
during the skill.

    python -m duckmind.skilltrain.hop_gate run --onnx hopover.onnx --servo bam --n 30 --out DIR
    python -m duckmind.skilltrain.hop_gate summarize --out DIR
"""

from __future__ import annotations

import argparse
import json
import math
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from duckmind.skilltrain import hop_contract as H
from duckmind.skilltrain import hop_probe as P

HEIGHTS_M = (0.01, 0.02, 0.03, 0.04)
ENTRIES = ("walk", "stand")
TRIGGER_D = (0.08, 0.16)
WALK_START_D = (0.35, 0.50)
APPROACH_VX = 0.40
AFTER_S = 2.0
NO_TRIGGER_S = 10.0


@dataclass(frozen=True)
class GateCell:
    height_m: float
    edge: str
    entry: str

    @property
    def key(self) -> str:
        return f"{self.edge}/{round(self.height_m * 1000):d}mm/{self.entry}"


def seed_of(key: str, rep: int) -> int:
    return zlib.crc32(f"hop-gate/{key}/{rep}".encode()) & 0x7FFFFFFF


def run_trial(cell: GateCell, *, servo: str, rep: int, scenes: Path, official: Path, onnx: str,
              delay_steps: int | None = None) -> dict[str, Any]:  # fmt: skip
    import mujoco
    import numpy as np

    from duckmind.skilltrain import dr as DR
    from duckmind.skilltrain.table import SkillDef

    if P._DUCK_CLASS is None:
        P._DUCK_CLASS = P._make_duck_class()
    began = time.perf_counter()
    seed = seed_of(cell.key, rep)
    draw = DR.draw(seed, "recipe" if servo == "bam" else "physical")
    rng = np.random.default_rng(seed + 101)
    obstacle = P.Obstacle(cell.height_m, cell.edge)
    folder = P.scene_dir(obstacle, scenes, official)
    skills = [SkillDef(**d) for d in H.skill_defs(str(Path(onnx).resolve()))]
    aim = math.radians(float(rng.uniform(-P.AIM_DEG, P.AIM_DEG)))
    d_trig = float(rng.uniform(*TRIGGER_D))
    d_start = float(rng.uniform(*WALK_START_D)) if cell.entry == "walk" else d_trig
    y0 = float(rng.uniform(-0.3, 0.3))
    duck = P._DUCK_CLASS(skills, servo=servo, draw=draw, head_alpha=1.0, keyframe="SIT", mjcf_dir=folder,
                         x=P.FACE_X - 0.9, y=y0, yaw=float(rng.uniform(-math.pi, math.pi)))  # fmt: skip
    if delay_steps is not None:
        from collections import deque

        duck.lag_steps = int(delay_steps)
        duck._lag = deque(maxlen=duck.lag_steps + 1)
    m = duck.model
    obst = [g for g in range(m.ngeom)
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("hop_obst_")]  # fmt: skip
    duck.setup_probe(obst)
    cls = H.height_class(cell.height_m)
    skill = f"hop_over_{cls}"
    row: dict[str, Any] = {"key": f"{cell.key}/{servo}/{rep}", "cell": asdict(cell), "servo": servo, "rep": rep,
                           "seed": seed, "dr": draw.to_json(), "aim_deg": round(math.degrees(aim), 3),
                           "trigger_d_m": round(d_trig, 4), "skill": skill, "lag_steps": duck.lag_steps,
                           "excluded": None}  # fmt: skip

    def excluded(why: str) -> dict[str, Any]:
        row["excluded"] = why
        row["wall_s"] = round(time.perf_counter() - began, 3)
        return row

    ok, _ = duck.stand(timeout=10.0, hold=1.0)
    if not ok:
        return excluded("did not stand up after the seated boot")
    duck.teleport(P.FACE_X - d_start, y0, aim)
    for _ in range(int(float(rng.uniform(0.5, 1.0)) * P.CONTROL_HZ)):
        duck.move(0.0)
        duck.tick()
    if duck.safety.fallen or duck.tilt_deg() > 30.0:
        return excluded("down before the approach")
    duck.robot_touch_any = False
    rec: dict[str, list[Any]] = {k: [] for k in ("t", "tilt", "fallen", "trunk_x", "foot_x", "toe_hits",
                                                 "housing", "tau_legs")}  # fmt: skip
    legs = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
    feet = [int(g) for g in duck.foot_geoms]
    t0 = duck.now()
    phase = "approach" if cell.entry == "walk" else "request"
    t_req = t_skill_start = t_skill_end = t_end = None
    touched_before = False

    def hold_heading() -> float:
        yaw = P._yaw(duck.data.qpos[duck.qadr + 3 : duck.qadr + 7])
        return max(-P.HOLD_MAX, min(P.HOLD_MAX, -P.HOLD_K * P._wrap(yaw - aim)))

    while True:
        now = duck.now()
        trunk_x = float(duck.data.qpos[duck.qadr]) - P.FACE_X
        if phase == "approach":
            if trunk_x >= -d_trig:
                phase = "request"
            elif now - t0 > NO_TRIGGER_S:
                return excluded("did not reach the trigger point within 10 s")
            else:
                duck.move(APPROACH_VX, 0.0, hold_heading())
        if phase == "request":
            touched_before = bool(duck.robot_touch_any)
            ack = duck.do(skill)
            if not ack.get("accepted"):
                return excluded(f"robotd refused {skill}: {ack}")
            t_req = now
            phase = "skill"
        if phase == "skill":
            duck.move(0.0)
        elif phase == "after":
            if cell.entry == "walk":
                duck.move(APPROACH_VX, 0.0, hold_heading())
            else:
                duck.move(0.0)
        elif phase == "watch":
            duck.move(0.0)
        duck.toe_tick = 0
        duck.tick()
        now = duck.now()
        d = duck.data
        tr = duck.trace
        rec["t"].append(now)
        rec["tilt"].append(float(tr.data["tilt"][-1]))
        rec["fallen"].append(float(tr.data["fallen"][-1]))
        rec["trunk_x"].append(float(d.qpos[duck.qadr]) - P.FACE_X)
        rec["foot_x"].append([float(d.geom_xpos[g][0]) - P.FACE_X for g in feet])
        rec["toe_hits"].append(duck.toe_tick)
        rec["housing"].append(
            float(np.max(tr.data["f_housing"][-1])) if len(tr.data["f_housing"][-1]) else 0.0
        )
        rec["tau_legs"].append([float(tr.data["tau_pk"][-1][i]) for i in legs])
        label = str(duck.policy)
        if phase == "skill":
            mine = label.startswith(skill)
            if mine and t_skill_start is None:
                t_skill_start = now
            if t_skill_start is not None and not mine:
                t_skill_end = now
                phase = "after"
            elif t_req is not None and now - t_req > H.HOLD_S + H.UNWIND_S + 2.0:
                return excluded("the skill did not hand back")
        elif phase == "after" and t_skill_end is not None and now >= t_skill_end + AFTER_S:
            phase = "watch"
            t_end = now
        elif (
            phase == "watch"
            and t_end is not None
            and now >= t_end + (P.WATCH_FALL_S if duck.safety.fallen else 1.0)
        ):
            break
        if duck.safety.fallen and phase in ("skill", "after"):
            # robotd's fall reflex is off inside a skill; after it, velstand rights (followed 3 s)
            if phase == "after":
                phase, t_end = "watch", now
    t = np.asarray(rec["t"])
    fallen = np.asarray(rec["fallen"]) > 0.5
    skill_mask = (t >= (t_skill_start or 0)) & (t <= (t_skill_end or t[-1]))
    after_mask = t > (t_skill_end or t[-1])
    rec_all = {**rec, "t_touch": t_req, "t_walk": t0, "t_walk_end": t_end or float(t[-1])}
    row.update(P.classify(rec_all, obstacle))
    tau = np.asarray(rec["tau_legs"])[skill_mask]
    housing = np.asarray(rec["housing"])[skill_mask]
    tilt = np.asarray(rec["tilt"])[skill_mask]
    row.update({
        "fall_in_skill": bool(fallen[skill_mask].any()),
        "fall_after": bool(fallen[after_mask].any()) and not bool(fallen[skill_mask].any()),
        "touched_before_request": touched_before,
        "skill_s": None if t_skill_start is None or t_skill_end is None else round(t_skill_end - t_skill_start, 3),
        "skill_tau_p95": round(float(np.percentile(tau, 95)), 4) if tau.size else None,
        "skill_tau_max": round(float(tau.max()), 4) if tau.size else None,
        "skill_housing_impact": bool(((housing > P.HOUSING_N) & (tilt < 45.0)).any()),
        "crossed_by_skill_end": bool(((np.asarray(rec["trunk_x"]) > H.DEPTH_M + H.PAST_TRUNK_M)
                                      & (np.min(np.asarray(rec["foot_x"]), axis=1) > H.DEPTH_M + P.PAST_FOOT_M)
                                      & skill_mask & ~fallen).any()),
        "sim_s": round(duck.now(), 3),
        "wall_s": round(time.perf_counter() - began, 3),
    })  # fmt: skip
    return row


def cells() -> list[GateCell]:
    return [GateCell(h, e, k) for h in HEIGHTS_M for e in P.EDGES for k in ENTRIES]


def run(args: argparse.Namespace) -> int:
    from duckmind.skilltrain.gate import Guard
    from duckmind.world.headless_duck import official_dir

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    official = official_dir(None)
    scenes = out / "scenes"
    tag = args.servo + (f"-d{args.delay_ms}" if args.delay_ms is not None else "")
    path = out / f"rows-{tag}.jsonl"
    done = P._done(path)
    guard = Guard(not args.no_guard, log=lambda s: print(s, flush=True))
    delay = None if args.delay_ms is None else int(round(args.delay_ms / 5.0))
    began = time.perf_counter()
    with path.open("a") as f:
        for cell in cells():
            valid = sum(1 for r in done.values() if r["cell"] == asdict(cell) and not r["excluded"])
            rep = 0
            while valid < args.n and rep < 3 * args.n:
                key = f"{cell.key}/{args.servo}/{rep}"
                if key in done:
                    rep += 1
                    continue
                guard.wait()
                row = run_trial(cell, servo=args.servo, rep=rep, scenes=scenes, official=official,
                                onnx=args.onnx, delay_steps=delay)  # fmt: skip
                f.write(json.dumps(row) + "\n")
                f.flush()
                done[key] = row
                valid += 0 if row["excluded"] else 1
                rep += 1
            print(
                f"[{tag}] {cell.key}: {valid} valid, {time.perf_counter() - began:.0f} s",
                flush=True,
            )
    return 0


def summarize(args: argparse.Namespace) -> int:
    out = Path(args.out)
    rows = []
    for p in sorted(out.glob("rows-*.jsonl")):
        tag = p.stem.removeprefix("rows-")
        rows += [dict(r, tag=tag) for r in P._done(p).values()]
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        if r["excluded"]:
            continue
        c = r["cell"]
        groups.setdefault(
            (r["tag"], c["edge"], round(c["height_m"] * 1000), c["entry"]), []
        ).append(r)
    res = []
    for (tag, edge, hmm, entry), rs in sorted(groups.items()):
        n = len(rs)

        def rate(k: int) -> list:
            lo, hi = H.wilson(k, n)
            return [k, n, round(100 * lo, 1), round(100 * hi, 1)]

        res.append({"tag": tag, "edge": edge, "height_mm": hmm, "entry": entry, "n": n,
                    "crossed": rate(sum(r["outcome"] == "crossed" for r in rs)),
                    "crossed_by_skill_end": rate(sum(r["crossed_by_skill_end"] for r in rs)),
                    "fall_in_skill": rate(sum(r["fall_in_skill"] for r in rs)),
                    "fall_after": rate(sum(r["fall_after"] for r in rs)),
                    "stuck": rate(sum(r["outcome"] == "stuck" for r in rs)),
                    "stumble": rate(sum(r["stumble"] for r in rs)),
                    "housing": rate(sum(r["skill_housing_impact"] for r in rs)),
                    "tau_p95_median": sorted(r["skill_tau_p95"] or 0 for r in rs)[n // 2]})  # fmt: skip
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    for c in res:
        print(c)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="hop_gate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--onnx", required=True)
    r.add_argument("--servo", choices=("mjcf", "bam"), required=True)
    r.add_argument("--n", type=int, default=30)
    r.add_argument("--delay-ms", type=int, default=None)
    r.add_argument("--out", required=True)
    r.add_argument("--no-guard", action="store_true")
    s = sub.add_parser("summarize")
    s.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return run(args) if args.cmd == "run" else summarize(args)


if __name__ == "__main__":
    raise SystemExit(main())
