"""Evaluation of the forward hop-over skill on the training physics, with the hand-back to v5.

hop:feasibility (2026-09-28). A grid of cells (strip row = height, edge column, entry kind), ``n``
ducks per cell under the recipe's final DR (no pushes; servo command delay 0–40 ms), each running
the entry's nominal window (3.0 s hold + 0.5 s unwind) and then **v5 velstand** (scale 0.9, zero
command) for ``handback_s`` — robotd's filter state carries over, as in robotd.

Policies: an exported ``hopover.onnx`` / a training actor (``policy``), or **the v5 baseline**
(``policy="v5walk"``): velstand itself walking at ``baseline_vx`` through the same window — what the
duck does today when it just walks on. Per duck (the first episode only), from MuJoCo truth:
crossed (both soles and the trunk past the far face, upright), the time to cross, robotd's fall
rule during the skill and during the hand-back, max tilt, toe stubs, leg |τ| p95 / max, servo
housing impacts > 2 N while upright. Rates are k/n with Wilson 95 %.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from duckmind.skilltrain import hop_contract as H
from duckmind.skilltrain import rl_contract as C

TRACE = (
    "tilt",
    "relx",
    "crossed",
    "fell",
    "stubs",
    "tau_leg",
    "servo_f",
    "speed",
    "phase",
    "foot_fz",
)


def evaluate(
    policy: Any,
    *,
    policies_dir: str | Path | None,
    n_per_cell: int = 32,
    cells: list[tuple[int, int, str]] | None = None,
    device: str = "cpu",
    library_path: str | None = None,
    handback_s: float = 2.0,
    baseline_vx: float = 0.40,
    seed: int = 11,
    delay: str | int = "train",
    log: Any = print,
) -> dict[str, Any]:
    """``delay``: the servo command delay (:func:`hop_contract.delay_fields`): ``"train"`` (training's
    piecewise-constant 0–40 ms), ``"jitter"`` (the calibration's per-step 0–40 ms) or a fixed lag in
    physics steps (0 / 4 / 8 = 0 / 20 / 40 ms)."""
    import torch
    from mjlab.envs import ManagerBasedRlEnv

    from duckmind.skilltrain import hop_mdp as M
    from duckmind.skilltrain import hop_tasks as T
    from duckmind.skilltrain import rl_mdp as RM
    from duckmind.skilltrain import rl_onnx

    cells = cells or T.eval_grid()
    n_env = n_per_cell * len(cells)
    cfg = T.make_env_cfg(library_path=library_path, eval_cells=cells, handback_s=handback_s,
                         delay_steps=delay)  # fmt: skip
    cfg.scene.num_envs = n_env
    cfg.seed = seed
    t0 = time.time()
    env = ManagerBasedRlEnv(cfg, device=device)
    obs, _ = env.reset(seed=seed)
    sp, plan = env.skill_plan, env.hop_plan
    term = env.command_manager.get_term("twist")
    action_term = env.action_manager.get_term("joint_pos")
    robot = env.scene["robot"]
    walk = (
        rl_onnx.torch_policy(Path(policies_dir) / "velstand.onnx", device) if policies_dir else None
    )
    baseline = isinstance(policy, str) and policy == "v5walk"
    if baseline and walk is None:
        raise ValueError("the v5 baseline needs the policies folder (velstand.onnx)")
    end = sp.end_ticks.clone()
    t_max = int(end.max().item())
    tr = {k: torch.zeros(t_max, n_env, device=device) for k in TRACE}
    done = torch.zeros(n_env, dtype=torch.bool, device=device)
    servo_sensor = env.scene.sensors["servo_ground_contact"]
    feet_sensor = env.scene.sensors["feet_ground_contact"]
    info = {
        "row": env.scene.terrain.terrain_levels.clone(), "col": env.scene.terrain.terrain_types.clone(),
        "kind": plan.kind.clone(), "height": plan.height.clone(), "start_d": plan.start_d.clone(),
        "hold": sp.hold_ticks.clone(), "unwind": sp.unwind_ticks.clone(), "cls": plan.cmd_height.clone(),
    }  # fmt: skip
    with torch.no_grad():
        for k in range(t_max):
            x = obs["actor"]
            phase = term.phase
            hb = phase >= RM.HANDBACK
            if baseline:
                xw = x.clone()
                xw[:, 48:51] = torch.tensor([baseline_vx, 0.0, 0.0], device=device)
                raw = walk(xw)
                scale = torch.full((n_env,), C.WALK_ACTION_SCALE, device=device)
            else:
                raw = policy(x)
                scale = torch.full((n_env,), H.ACTION_SCALE, device=device)
            if walk is not None:
                raw = torch.where(hb.unsqueeze(1), walk(x), raw)
                scale = torch.where(hb, torch.full_like(scale, C.WALK_ACTION_SCALE), scale)
            action_term.set_runtime_scale(scale)
            obs, _r, terminated, truncated, _ = env.step(raw)
            ended = terminated | truncated
            d = robot.data
            force = torch.nan_to_num(d.actuator_force.abs(), nan=0.0)
            sf = servo_sensor.data.force
            sf = (
                sf.reshape(n_env, -1, 3).norm(dim=2).max(dim=1).values
                if sf is not None
                else torch.zeros(n_env, device=device)
            )
            ff = feet_sensor.data.force
            fz = (
                torch.nan_to_num(ff[:, :2, 2], nan=0.0).abs().max(dim=1).values
                if ff is not None
                else torch.zeros(n_env, device=device)
            )
            vals = {
                "tilt": M.tilt_deg(env), "relx": M.rel_x(env), "crossed": plan.crossed.float(),
                "fell": plan.fell.float(), "stubs": plan.stubs, "tau_leg": force[:, list(C.LEG_IDX)].max(dim=1).values,
                "servo_f": sf, "speed": d.root_link_lin_vel_w[:, :2].norm(dim=1), "phase": phase.float(),
                "foot_fz": fz,
            }  # fmt: skip
            write = ~done & ~ended
            for key, v in vals.items():
                tr[key][k] = torch.where(write, torch.nan_to_num(v.float(), nan=0.0), tr[key][k])
            if bool(ended.any()) and k > 0:
                for key in TRACE:
                    tr[key][k] = torch.where(ended & ~done, tr[key][k - 1], tr[key][k])
            done |= ended
            if bool(done.all()):
                break
    wall = time.time() - t0
    traces = {k: v.cpu().numpy() for k, v in tr.items()}
    meta_info = {k: v.cpu().numpy() for k, v in info.items()}
    try:
        env.close()
    except Exception:  # noqa: BLE001
        pass
    res = summarise(
        cells, n_per_cell, traces, meta_info, handback_ticks=int(round(handback_s / C.DT))
    )
    res["meta"] = {"policy": "v5walk" if baseline else "trained", "baseline_vx": baseline_vx if baseline else None,
                   "n_per_cell": n_per_cell, "cells": len(cells), "ducks": n_env, "device": device,
                   "handback_s": handback_s, "seed": seed, "wall_s": round(wall, 1), "library": library_path,
                   "delay": H.delay_label(delay), "delay_fields": H.delay_fields(delay),
                   "physics": "official VelStand recipe (allcollisions, BAM XL330), final DR, no pushes; "
                              "strip terrain (hop_terrain); servo command delay as 'delay'"}  # fmt: skip
    log(
        f"[hop] eval {'v5walk' if baseline else 'trained'} ({H.delay_label(delay)}): {n_env} ducks, "
        f"{len(cells)} cells in {wall:.0f} s"
    )
    return res


def _fall(tilt: np.ndarray, start: int, stop: int) -> bool:
    run = 0
    for t in tilt[start:stop]:
        run = run + 1 if t > H.FALL_TILT_DEG else 0
        if run >= int(round(H.FALL_DEBOUNCE_S / C.DT)):
            return True
    return False


def duck_outcome(
    i: int, tr: dict[str, np.ndarray], hold: int, unwind: int, handback: int
) -> dict[str, Any]:
    skill_end = hold + unwind
    tilt = tr["tilt"][:, i]
    fell = _fall(tilt, 0, skill_end)
    crossed_seq = tr["crossed"][:skill_end, i] > 0.5
    crossed = bool(crossed_seq.any()) and not fell
    t_cross = float(np.argmax(crossed_seq)) * C.DT if crossed_seq.any() else None
    fell_hb = None if (handback <= 0 or fell) else _fall(tilt, skill_end, skill_end + handback)
    upright = tilt[:skill_end] < 45.0
    impacts = int(np.sum((tr["servo_f"][:skill_end, i] > 2.0) & upright))
    tau = tr["tau_leg"][:skill_end, i]
    fz = tr["foot_fz"][:skill_end, i] if "foot_fz" in tr else np.zeros(1)
    return {
        "foot_peak_bw": round(float(fz.max()) / H.BODY_WEIGHT_N, 3),
        "crossed": crossed,
        "fall_skill": fell,
        "fall_handback": fell_hb,
        "t_cross_s": t_cross,
        "max_tilt": float(tilt[:skill_end].max()),
        "stumble": bool(tilt[:skill_end].max() >= H.STUMBLE_DEG),
        "stubs": float(tr["stubs"][skill_end - 1, i]),
        "tau_leg_p95": float(np.percentile(tau, 95)),
        "tau_leg_max": float(tau.max()),
        "housing_impact": impacts > 0,
        "end_relx_m": float(tr["relx"][skill_end - 1, i]),
        "end_speed": float(tr["speed"][skill_end - 1, i]),
    }


def _rate(k: int, n: int) -> list[float]:
    lo, hi = H.wilson(k, n)
    return [int(k), int(n), round(100 * lo, 1), round(100 * hi, 1)]


def summarise(cells: list[tuple[int, int, str]], n_per_cell: int, tr: dict[str, np.ndarray],
              info: dict[str, np.ndarray], handback_ticks: int) -> dict[str, Any]:  # fmt: skip
    per_cell: dict[str, Any] = {}
    rows_out = []
    for c_idx, (row, col, kind) in enumerate(cells):
        ducks = [i for i in range(len(info["row"])) if i % len(cells) == c_idx]
        outs = []
        for i in ducks:
            o = duck_outcome(i, tr, int(info["hold"][i]), int(info["unwind"][i]), handback_ticks)
            o.update(
                {"start_d": float(info["start_d"][i]), "height_mm": float(info["height"][i] * 1000)}
            )
            outs.append(o)
            rows_out.append({"cell": c_idx, **o})
        n = len(outs)
        key = f"h{round(float(info['height'][ducks[0]]) * 1000)}mm/{H.EDGES[col]}/{kind}"
        hb = [o for o in outs if o["fall_handback"] is not None]
        per_cell[key] = {
            "n": n,
            "crossed": _rate(sum(o["crossed"] for o in outs), n),
            "fall_skill": _rate(sum(o["fall_skill"] for o in outs), n),
            "fall_handback": _rate(sum(bool(o["fall_handback"]) for o in hb), len(hb)),
            "stumble": _rate(sum(o["stumble"] for o in outs), n),
            "housing_impact": _rate(sum(o["housing_impact"] for o in outs), n),
            "t_cross_p50": float(np.median([o["t_cross_s"] for o in outs if o["t_cross_s"] is not None]))
            if any(o["t_cross_s"] is not None for o in outs) else None,
            "tau_leg_p95_p50": round(float(np.median([o["tau_leg_p95"] for o in outs])), 3),
            "tau_leg_max_p90": round(float(np.percentile([o["tau_leg_max"] for o in outs], 90)), 3),
            "stubs_p50": float(np.median([o["stubs"] for o in outs])),
            "foot_peak_bw_p50": round(float(np.median([o["foot_peak_bw"] for o in outs])), 2),
            "foot_peak_bw_p90": round(float(np.percentile([o["foot_peak_bw"] for o in outs], 90)), 2),
        }  # fmt: skip
    # pooled by height (the gates are per height class)
    by_h: dict[str, Any] = {}
    pooled: dict[str, list[dict]] = {}
    for key, v in per_cell.items():
        hkey = key.split("/")[0]
        agg = by_h.setdefault(
            hkey,
            {
                "crossed": [0, 0],
                "fall_skill": [0, 0],
                "fall_handback": [0, 0],
                "housing_impact": [0, 0],
            },  # fmt: skip
        )
        for m in ("crossed", "fall_skill", "fall_handback", "housing_impact"):
            agg[m][0] += v[m][0]
            agg[m][1] += v[m][1]
    for r in rows_out:
        pooled.setdefault(f"h{round(r['height_mm'])}mm", []).append(r)
    for hkey, agg in by_h.items():
        for m in list(agg):
            agg[m] = _rate(agg[m][0], agg[m][1])
        rs = pooled.get(hkey, [])
        if rs:
            agg["tau_leg_p95_p50"] = round(float(np.median([o["tau_leg_p95"] for o in rs])), 3)
            agg["foot_peak_bw_p50"] = round(float(np.median([o["foot_peak_bw"] for o in rs])), 2)
            agg["foot_peak_bw_p90"] = round(
                float(np.percentile([o["foot_peak_bw"] for o in rs], 90)), 2
            )
            agg["landing_over_target"] = _rate(
                sum(o["foot_peak_bw"] > H.LANDING_BW_TARGET for o in rs), len(rs)
            )
    return {"cells": per_cell, "by_height": by_h, "ducks": rows_out}


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="hop_eval")
    ap.add_argument(
        "--onnx", default="v5walk", help="an exported hopover.onnx, or v5walk (the baseline)"
    )
    ap.add_argument("--policies", required=True, help="the v5 folder (velstand.onnx)")
    ap.add_argument("--library", default=None)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--vx", type=float, default=0.40)
    ap.add_argument("--kinds", default="walk,stand")
    ap.add_argument(
        "--delay", default="train", help="train | jitter | a fixed delay (e.g. 20ms or 4 steps)"
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    from duckmind.skilltrain import hop_tasks as T
    from duckmind.skilltrain import rl_onnx

    policy: Any = (
        "v5walk" if args.onnx == "v5walk" else rl_onnx.torch_policy(args.onnx, args.device)
    )
    res = evaluate(policy, policies_dir=args.policies, n_per_cell=args.n, device=args.device,
                   library_path=args.library, baseline_vx=args.vx,
                   cells=T.eval_grid(tuple(args.kinds.split(","))),
                   delay=H.parse_delay(args.delay))  # fmt: skip
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    for key, v in res["by_height"].items():
        print(key, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
