"""The A100 evaluation battery of the B networks (b:train, 2026-09-28).

The final actor (a checkpoint's deterministic mean action) on the training physics — the BAM XL330 on MuJoCo Warp,
the recipe's DR on every duck (final CoM ranges, foot μ 0.7–1.3, no pushes) — at a **fixed** servo command delay
(0 / 20 / 40 ms = 0 / 4 / 8 physics steps), every trained entry × the gate's entry kinds (turn / march: standing
and walking; bow / lie: standing; the mats: standing at 3 / 4 / 5 cm, rigid and soft), nominal windows (held
entries 3.0–3.5 s), one episode per duck. What it cannot see: robotd's hand-back to velstand after the window
(the Mac gate runs that), the XML servos (the Mac gate).

Per duck, at the end of its window (the tick before the reset): fell (tilt > 60° at any time), max tilt, and the
skill's own numbers — turn: yaw turned in the hold (signed, deg), rate, max excursion from the entry, real steps;
march: footfalls, max excursion, |heading change|; mat: on the target level with tilt < 30° (success) and
success without a fall (clean); bow / lie: the deepest nose-down pitch and trunk drop in the hold, seconds held at
the gate's pose (bow ≥ 15°; lie ≥ 20° and ≥ 20 mm). Rates with Wilson 95 % per cell.

    python -m duckmind.skilltrain.bsk_eval --net bskmat --ckpt latest.pt --library v5_entries.npz \
        --n 512 --delays 0,4,8 --device cuda:0 --out eval_bskmat.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from duckmind.skilltrain import bsk_contract as B

KINDS = {"turn": ("stand", "walk"), "march": ("stand", "walk"), "step_up": ("stand",), "step_down": ("stand",),
         "bow": ("stand",), "lie": ("stand",)}  # fmt: skip
MAT_LEVELS = (2, 4, 6)  # rows → 3.0 / 4.0 / 5.0 cm (bsk_mdp.mat_height_of with 7 rows 2–5 cm)


def cells(net: str) -> list[tuple[str, str]]:
    return [
        (e.name, k) for e in B.entries(net) if e.behaviour != "release" for k in KINDS[e.behaviour]
    ]


def _rate(k: int, n: int) -> dict[str, Any]:
    lo, hi = B.wilson(k, n)
    return {"k": int(k), "n": int(n), "rate": round(k / n, 4) if n else None,
            "wilson95": [round(lo, 4), round(hi, 4)]}  # fmt: skip


def _q(v: list[float]) -> dict[str, Any] | None:
    import numpy as np

    a = np.asarray([x for x in v if x is not None and math.isfinite(x)], float)
    if not len(a):
        return None
    return {"p10": round(float(np.percentile(a, 10)), 3), "p50": round(float(np.median(a)), 3),
            "p90": round(float(np.percentile(a, 90)), 3)}  # fmt: skip


def run_delay(
    net: str, ckpt: str, library: str | None, n: int, delay: int, device: str
) -> dict[str, Any]:
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper

    from duckmind.skilltrain import bsk_mdp as M
    from duckmind.skilltrain import bsk_tasks as T
    from duckmind.skilltrain import rl_mdp as RM
    from duckmind.skilltrain.rl_eval import checkpoint_policy

    cl = cells(net)
    per = len(cl) * (len(MAT_LEVELS) * 2 if net == "bskmat" else 1)
    num = n * per
    cfg = T.make_env_cfg(net, library_path=library, eval_cells=cl, delay_steps=(delay, delay))
    cfg.scene.num_envs = num
    cfg.seed = 1000 + delay
    t0 = time.time()
    env = ManagerBasedRlEnv(cfg, device=device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    fn, info = checkpoint_policy(ckpt, device)
    dev = env.device
    ids = torch.arange(num, device=dev)
    mat_row = torch.full((num,), -1, dtype=torch.long, device=dev)
    soft = torch.zeros(num, dtype=torch.bool, device=dev)
    if net == "bskmat":
        terrain = env.scene.terrain
        g = ids // len(cl)  # the cell is ids % len(cl) (reset_bsk_episode's eval_cells)
        mat_row = torch.tensor(MAT_LEVELS, device=dev)[g % len(MAT_LEVELS)]
        types = (g // len(MAT_LEVELS)) % terrain.terrain_origins.shape[1]
        terrain.terrain_levels[:] = mat_row
        terrain.terrain_types[:] = types
        terrain.env_origins[:] = terrain.terrain_origins[mat_row, types]
        soft = (
            types == 1
        )  # MatTerrainCfg: column 0 rigid, column 1 soft (SOFT_SHARE 0.4 → 1 of 2 columns)
    env.reset()
    obs = wrapped.get_observations()
    sp, bp = RM.plan_of(env), M.bsk_plan(env)
    robot = env.scene["robot"]
    cell_of = (ids % len(cl)).clone()
    beh0 = bp.behaviour.clone()
    hold_s = sp.hold_ticks.float() * env.step_dt
    ends = sp.end_ticks.clone()
    z_entry = sp.entry_z.clone()
    xy_entry = sp.entry_xy.clone()
    yaw_entry = sp.entry_yaw.clone()
    zeros = lambda: torch.zeros(num, device=dev)  # noqa: E731
    run = {
        k: zeros() for k in ("exc", "pitch_max", "drop_max", "t_bow", "t_lie")
    }  # this script's running values
    plan_keys = ("fell", "max_tilt", "success", "yaw", "steps", "dyaw")
    final = {k: zeros() for k in (*run, *plan_keys)}
    snap: dict[str, torch.Tensor] = {}
    done = torch.zeros(num, dtype=torch.bool, device=dev)
    steps_needed = int(ends.max().item()) + 2
    with torch.inference_mode():
        for _ in range(steps_needed):
            act = fn(obs["actor"])
            obs, _, dones, _ = wrapped.step(act.to(wrapped.device))
            ended = dones.bool().to(dev) & ~done
            live = ~done & ~ended
            # an ended duck was reset inside the step: its episode's values are the previous tick's (snap / run)
            for k in plan_keys:
                if k in snap:
                    final[k] = torch.where(ended, snap[k].float(), final[k])
            for k in run:
                final[k] = torch.where(ended, run[k], final[k])
            done |= ended
            hold = (RM.skill_term(env).phase == RM.HOLD) & live
            pos = robot.data.root_link_pos_w
            d = (pos[:, :2] - xy_entry).norm(dim=1)
            run["exc"] = torch.where(live, torch.maximum(run["exc"], d), run["exc"])
            _, pitch = RM._roll_pitch(robot.data.root_link_quat_w)
            pdeg = torch.rad2deg(torch.nan_to_num(pitch))
            drop = z_entry - (pos[:, 2] - env.scene.env_origins[:, 2])
            run["pitch_max"] = torch.where(
                hold, torch.maximum(run["pitch_max"], pdeg), run["pitch_max"]
            )
            run["drop_max"] = torch.where(
                hold, torch.maximum(run["drop_max"], drop), run["drop_max"]
            )
            run["t_bow"] = (
                run["t_bow"] + (hold & (pdeg >= B.GATES["bow"]["pitch_deg"])).float() * env.step_dt
            )
            lie_pose = hold & (pdeg >= B.GATES["lie"]["pitch_deg"]) & (drop >= 0.020)
            run["t_lie"] = run["t_lie"] + lie_pose.float() * env.step_dt
            snap = {"fell": bp.fell.clone(), "max_tilt": bp.max_tilt.clone(), "success": bp.success.clone(),
                    "yaw": bp.yaw_acc.clone(), "steps": bp.steps.clone(),
                    "dyaw": RM._wrap(RM._yaw(robot.data.root_link_quat_w) - yaw_entry).abs()}  # fmt: skip
            if bool(done.all()):
                break
    build_run_s = round(time.time() - t0, 1)
    cpu = {k: v.detach().cpu().numpy() for k, v in final.items()}
    cell_np, row_np, soft_np = cell_of.cpu().numpy(), mat_row.cpu().numpy(), soft.cpu().numpy()
    beh_np, hold_np, done_np = beh0.cpu().numpy(), hold_s.cpu().numpy(), done.cpu().numpy()
    out_cells = []
    groups: dict[tuple, list[int]] = {}
    for i in range(num):
        if not done_np[i]:
            continue
        key = (int(cell_np[i]), int(row_np[i]), bool(soft_np[i]))
        groups.setdefault(key, []).append(i)
    for (c, row, sft), members in sorted(groups.items()):
        name, kind = cl[c]
        beh = B.BEHAVIOURS[int(beh_np[members[0]])]
        m = len(members)
        fell = [bool(cpu["fell"][i]) for i in members]
        res: dict[str, Any] = {"entry": name, "kind": kind, "n": m, "falls": _rate(sum(fell), m),
                               "max_tilt_deg": _q([float(cpu["max_tilt"][i]) for i in members])}  # fmt: skip
        if beh == "turn":
            sign = 1.0 if name == "turn_left" else -1.0
            yaw = [sign * math.degrees(float(cpu["yaw"][i])) for i in members]
            rate = [y / max(1e-3, float(hold_np[i])) for y, i in zip(yaw, members, strict=True)]
            res.update(yaw_deg=_q(yaw), rate_dps=_q(rate), excursion_m=_q([float(cpu["exc"][i]) for i in members]),
                       steps=_q([float(cpu["steps"][i]) for i in members]),
                       performed=_rate(sum(1 for y, r, i, f in zip(yaw, rate, members, fell, strict=True)
                                           if r >= B.GATES["turn"]["rate_dps"] and cpu["exc"][i] <= B.GATES["turn"]["excursion_m"]
                                           and not f), m))  # fmt: skip
        elif beh == "march":
            dy = [math.degrees(float(cpu["dyaw"][i])) for i in members]
            fps = [float(cpu["steps"][i]) / max(1e-3, float(hold_np[i])) for i in members]
            res.update(heading_change_deg=_q(dy), excursion_m=_q([float(cpu["exc"][i]) for i in members]),
                       footfalls_per_s=_q(fps),
                       performed=_rate(sum(1 for a, s, i, f in zip(dy, fps, members, fell, strict=True)
                                           if a <= B.GATES["march"]["yaw_deg_4s"] and s >= B.GATES["march"]["footfalls_per_s"]
                                           and cpu["exc"][i] <= B.GATES["march"]["travel_m_4s"] and not f), m))  # fmt: skip
        elif beh in ("step_up", "step_down"):
            succ = [bool(cpu["success"][i]) for i in members]
            res.update(height_mm=round(1000 * float(M.mat_height_of(row)), 1), soft=sft, success=_rate(sum(succ), m),
                       performed=_rate(sum(s and not f for s, f in zip(succ, fell, strict=True)), m))  # fmt: skip
        else:
            key_t = "t_bow" if beh == "bow" else "t_lie"
            need = B.GATES[beh]["hold_s"]
            held = [float(cpu[key_t][i]) for i in members]
            res.update(pitch_max_deg=_q([float(cpu["pitch_max"][i]) for i in members]),
                       drop_max_mm=_q([1000 * float(cpu["drop_max"][i]) for i in members]), held_s=_q(held),
                       performed=_rate(sum(1 for h, f in zip(held, fell, strict=True) if h >= need and not f), m))  # fmt: skip
        out_cells.append(res)
    try:
        env.close()
    except Exception:  # noqa: BLE001
        pass
    return {"delay_steps": delay, "delay_ms": 5 * delay, "ducks": num, "finished": int(done_np.sum()),
            "build_run_s": build_run_s, "ckpt": info["ckpt"], "cells": out_cells}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bsk_eval")
    ap.add_argument("--net", choices=B.NETS, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--library", default=None)
    ap.add_argument(
        "--n", type=int, default=512, help="ducks per cell (per mat height × top for bskmat)"
    )
    ap.add_argument(
        "--delays", default="0,4,8", help="fixed servo command delays in 5 ms physics steps"
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = {"format": "duckmind-bsk-a100-eval-1", "net": a.net, "n_per_cell": a.n, "started": time.strftime("%H:%M:%S"),
           "delays": []}  # fmt: skip
    for d in (int(x) for x in a.delays.split(",") if x.strip()):
        r = run_delay(a.net, a.ckpt, a.library, a.n, d, a.device)
        res["delays"].append(r)
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
        print(
            f"[bsk_eval] {a.net} delay {5 * d} ms: {r['finished']}/{r['ducks']} ducks in {r['build_run_s']} s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
