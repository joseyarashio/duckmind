"""Checkpoint evaluation on the training physics (the plan's tier 3 "A100 DR battery", §6).

For each (table entry, entry kind) cell, ``n`` ducks under the recipe's final domain
randomisation (trunk CoM ±15 mm, head CoM ±10 mm, foot μ 0.7–1.3 — or the training floor's
0.5–1.3 — mass ±5 %, BAM friction/armature ±10 %, battery 6.5–8.2 V with sag, 15–30 ms delay,
encoder bias, IMU tilt, noise; no pushes) run the entry's nominal window through robotd's path,
then **hand back**: v5 velstand (scale 0.9, zero command) for standing skills, sitstand (the sit
flag, scale 1.0) for ``stretch_sit`` — the filter state carries over, as in robotd.

Per duck, from MuJoCo truth (the gates double as the "performed" detectors):

* **falls** — robotd's rule (tilt > 60° for 0.2 s), during the skill and in the hand-back;
* **crouch** — first visible motion (≥ 3 mm or 3°) time, neck pull-back at 0.25 s, trunk dip at
  0.3 s, the hold band (± 3 mm) and footfalls per 2 s (holds ≥ 1 s), from walking: speed at
  0.6–0.8 s and travel;
* **hop** — backward displacement at 0.7 s along the entry heading, yaw change, planted at the end;
* **shake** — any 0.5 s window with trunk roll ≥ 4° at 2.5–4 Hz, head yaw ≥ 0.15 rad at ≥ 3 Hz;
  footfalls; drift;
* **stretch** — ≥ 0.8 s continuously with the trunk ≥ +8 mm (seated: ≥ +10 mm over the seat)
  and the camera ≥ +15°; seated: re-seated (z ≤ seat + 5 mm) within 1.5 s of the unwind;
* **strain** (skill phase) — per-duck |τ| p95 (legs, head), stall fraction (> 0.4 N·m at
  < 0.5 rad/s), |joint acceleration| p99, servo-housing floor contacts > 2 N while upright.

Rates are k/n with Wilson 95 %; distributions are [p50, p95, max].
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from duckmind.skilltrain import rl_contract as C

# A footfall is a re-contact after ≥ 40 ms of air — the Mac gate's definition (skillduck.MIN_AIR_S, phase 1
# touchdowns()); contact chatter under that is not a step (tr:train 2026-09-28; it counted every re-contact).
MIN_AIR_S = 0.04
TRACE = (
    "z", "x", "y", "yaw", "tilt", "roll", "pitch", "neck", "head_yaw", "head_yaw_rate", "roll_rate",
    "cam", "speed", "td", "feet", "tau_leg95", "tau_head95", "stall", "jacc99", "servo_f",
)  # fmt: skip


def default_cells(net: str) -> list[tuple[str, str]]:
    """Every table entry × every entry kind its postures allow (crouched / coast / risen
    included), the V1 grid."""
    cells = []
    for v in C.variants(net):
        kinds = C.entry_mix(v.behaviour, v.postures)
        for kind in C.ENTRY_KINDS:
            if kind in kinds:
                cells.append((v.name, kind))
    return cells


def evaluate(
    net: str,
    policy: Any,
    *,
    policies_dir: str | Path | None,
    n_per_cell: int = 64,
    cells: list[tuple[str, str]] | None = None,
    device: str = "cpu",
    library_path: str | None = None,
    handback_s: float = C.GATES["safety"]["handback_s"],  # type: ignore[index]
    foot_mu: tuple[float, float] = (0.7, 1.3),
    seed: int = 11,
    log: Any = print,
    servo_lag: str | int | None = None,
) -> dict[str, Any]:
    """Run the grid; ``policy`` maps a (N, 61) obs tensor to raw actions (the trained actor, or
    ``rl_onnx.torch_policy`` of an exported file). ``servo_lag`` (s1b:retrain): the servo command delay —
    ``None``/``"recipe"`` the recipe's 15–30 ms redrawn every physics step, an int a fixed lag in 5-ms physics
    steps (0 = none), ``"train"`` training's randomised 0–40 ms."""
    import torch
    from mjlab.envs import ManagerBasedRlEnv

    from duckmind.skilltrain import rl_mdp as M
    from duckmind.skilltrain import rl_onnx
    from duckmind.skilltrain import rl_tasks as T

    cells = cells or default_cells(net)
    if not policies_dir:
        handback_s = 0.0
    n_env = n_per_cell * len(cells)
    cfg = T.make_env_cfg(
        net, library_path=library_path, eval_cells=cells, handback_s=handback_s, foot_mu=foot_mu,
        servo_lag=servo_lag,
    )  # fmt: skip
    cfg.scene.num_envs = n_env
    cfg.seed = seed
    t0 = time.time()
    env = ManagerBasedRlEnv(cfg, device=device)
    obs, _ = env.reset(seed=seed)
    plan = env.skill_plan
    term = env.command_manager.get_term("twist")
    action_term = env.action_manager.get_term("joint_pos")
    robot = env.scene["robot"]
    walk: Any = None
    sit: Any = None
    if handback_s > 0 and policies_dir:
        walk = rl_onnx.torch_policy(Path(policies_dir) / "velstand.onnx", device)
        sit = rl_onnx.torch_policy(Path(policies_dir) / "alpha_sitstand.onnx", device)
    skill_scale, _gain = C.pinned(net)
    end = plan.end_ticks.clone()
    T_max = int(end.max().item())
    tr = {k: torch.zeros(T_max, n_env, device=device) for k in TRACE}
    done = torch.zeros(n_env, dtype=torch.bool, device=device)
    servo_ids = __import__("mjlab_microduck.tasks.mdp", fromlist=["x"])._servo_joint_ids(env, robot)
    sensors = env.scene.sensors
    feet = sensors["feet_ground_contact"]
    servo_sensor = (
        sensors.get("servo_ground_contact")
        if hasattr(sensors, "get")
        else sensors["servo_ground_contact"]
    )
    leg_idx, head_idx = list(C.LEG_IDX), list(C.HEAD_IDX)
    prev_qd = robot.data.joint_vel[:, servo_ids].clone()
    record_plan = {
        "variant": plan.variant.clone(), "kind": plan.kind.clone(), "level": plan.level.clone(),
        "hold": plan.hold_ticks.clone(), "unwind": plan.unwind_ticks.clone(), "entry_yaw": plan.entry_yaw.clone(),
        "entry_xy": plan.entry_xy.clone(), "entry_z": plan.entry_z.clone(), "seated": plan.seated.clone(),
    }  # fmt: skip
    with torch.no_grad():
        for k in range(T_max):
            x = obs["actor"]
            raw = policy(x)
            phase = term.phase
            if walk is not None:
                hb = phase >= M.HANDBACK
                seated = plan.seated
                raw = torch.where((hb & ~seated).unsqueeze(1), walk(x), raw)
                raw = torch.where((hb & seated).unsqueeze(1), sit(x), raw)
                scale = torch.where(
                    hb, torch.where(seated, torch.full_like(plan.level, C.SITSTAND_ACTION_SCALE),
                                    torch.full_like(plan.level, C.WALK_ACTION_SCALE)),
                    torch.full_like(plan.level, skill_scale),
                )  # fmt: skip
                action_term.set_runtime_scale(scale)
            obs, _r, terminated, truncated, _ = env.step(raw)
            ended = terminated | truncated
            d = robot.data
            live = ~done
            q = d.root_link_quat_w
            roll, pitch = M._roll_pitch(q)
            qd = d.joint_vel[:, servo_ids]
            acc = ((qd - prev_qd) / env.step_dt).abs()
            prev_qd = qd.clone()
            force = torch.nan_to_num(d.actuator_force.abs(), nan=0.0)
            first = feet.compute_first_contact(env.step_dt)[:, :2]
            first = first & (feet.data.last_air_time[:, :2] >= MIN_AIR_S)
            contact = feet.data.found[:, :2] > 0
            sf = servo_sensor.data.force
            sf = (
                sf.reshape(n_env, -1, 3).norm(dim=2).max(dim=1).values
                if sf is not None
                else torch.zeros(n_env, device=device)
            )
            vals = {
                "z": d.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2],
                "x": d.root_link_pos_w[:, 0], "y": d.root_link_pos_w[:, 1], "yaw": M._yaw(q),
                "tilt": torch.rad2deg(torch.acos((1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)).clamp(-1, 1))),
                "roll": roll, "pitch": pitch,
                "neck": d.joint_pos[:, servo_ids][:, C.NECK_PITCH],
                "head_yaw": d.joint_pos[:, servo_ids][:, C.HEAD_YAW],
                "head_yaw_rate": qd[:, C.HEAD_YAW], "roll_rate": d.root_link_ang_vel_b[:, 0],
                "cam": M.camera_pitch_deg(env), "speed": d.root_link_lin_vel_w[:, :2].norm(dim=1),
                "td": first.float().sum(dim=1), "feet": contact.float().sum(dim=1),
                "tau_leg95": force[:, leg_idx].max(dim=1).values, "tau_head95": force[:, head_idx].max(dim=1).values,
                "stall": ((force > 0.4) & (qd.abs() < 0.5)).float().sum(dim=1),
                "jacc99": acc.max(dim=1).values, "servo_f": sf,
            }  # fmt: skip
            # an env that just ended auto-reset: its post-step state is the NEXT episode's
            write = live & ~ended
            for key, v in vals.items():
                tr[key][k] = torch.where(write, torch.nan_to_num(v.float(), nan=0.0), tr[key][k])
            if bool(ended.any()):
                # the last tick of the episode: keep the state before the reset (x,y,…) from the
                # previous row so the trace stays continuous
                if k > 0:
                    for key in TRACE:
                        tr[key][k] = torch.where(ended & live, tr[key][k - 1], tr[key][k])
            done |= ended
            if bool(done.all()):
                break
    sim_s = time.time() - t0
    traces = {k: v.cpu().numpy() for k, v in tr.items()}
    info = {k: v.cpu().numpy() for k, v in record_plan.items()}
    try:
        env.close()
    except Exception:  # noqa: BLE001
        pass
    result = summarise(
        net, cells, n_per_cell, traces, info, handback_ticks=int(round(handback_s / C.DT))
    )
    result["meta"] = {
        "net": net, "n_per_cell": n_per_cell, "cells": len(cells), "ducks": n_env, "device": device,
        "handback_s": handback_s, "foot_mu": list(foot_mu), "seed": seed, "wall_s": round(sim_s, 1),
        "library": library_path, "physics": "official VelStand recipe (allcollisions, BAM XL330), final DR, no pushes",
        "servo_lag": servo_lag if servo_lag is not None else "recipe",
        "servo_lag_ms": (int(servo_lag) * 5 if isinstance(servo_lag, int) else
                         "15–30 per physics step" if servo_lag in (None, "recipe") else "0–40 piecewise"),
    }  # fmt: skip
    log(f"[skilltrain] eval {net}: {n_env} ducks, {len(cells)} cells in {sim_s:.0f} s")
    return result


def _fall_mask(tilt: np.ndarray, start: int, stop: int) -> bool:
    run = 0
    for t in tilt[start:stop]:
        run = run + 1 if t > C.FALL_TILT_DEG else 0
        if run >= int(round(C.FALL_DEBOUNCE_S / C.DT)):
            return True
    return False


def _window_osc(x: np.ndarray, rate: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-window (amplitude = √2·std, frequency = rms(rate)/(2π·std))."""
    if len(x) < w:
        return np.zeros(0), np.zeros(0)
    amps, freqs = [], []
    for s in range(0, len(x) - w + 1):
        seg, rseg = x[s : s + w], rate[s : s + w]
        sd = float(np.std(seg))
        amps.append(math.sqrt(2) * sd)
        freqs.append(float(np.sqrt(np.mean(rseg**2))) / (2 * math.pi * max(sd, 1e-6)))
    return np.array(amps), np.array(freqs)


def _held(ok: np.ndarray, ticks: int) -> bool:
    run = 0
    for v in ok:
        run = run + 1 if v else 0
        if run >= ticks:
            return True
    return False


def duck_outcome(net: str, v: C.Variant, kind: str, level: float, i: int, tr: dict, hold: int,
                 unwind: int, handback: int, entry_yaw: float) -> dict[str, Any]:  # fmt: skip
    """Everything one duck's first episode shows (the gates as detectors)."""
    dt = C.DT
    col = {k: tr[k][:, i] for k in TRACE}
    skill_end = hold + unwind
    fell = _fall_mask(col["tilt"], 0, skill_end)
    out: dict[str, Any] = {
        "fall_skill": fell,
        # the hand-back test: excess falls of ducks that were up when the skill handed back
        "fall_handback": None
        if (handback <= 0 or fell)
        else _fall_mask(col["tilt"], skill_end, skill_end + handback),
    }
    z0, neck0 = col["z"][0], col["neck"][0]
    tick = lambda s: min(max(0, int(round(s / dt))), max(0, hold - 1))  # noqa: E731
    g = C.GATES
    beh = v.behaviour
    if beh in ("crouch", "hop"):
        gc = g["crouch"]
        moved = (
            (np.abs(col["z"][:hold] - z0) * 1000 >= gc["first_motion_mm"])
            | (np.degrees(np.abs(col["neck"][:hold] - neck0)) >= gc["first_motion_deg"])
            | (np.degrees(np.abs(col["pitch"][:hold] - col["pitch"][0])) >= gc["first_motion_deg"])
        )
        first = float(np.argmax(moved) * dt) if moved.any() else None
        retract = float(col["neck"][tick(gc["retract_by_s"])] - neck0)
        dip_mm = float((z0 - col["z"][tick(gc["dip_by_s"])]) * 1000)
        k = C.HOP_K if beh == "hop" else level
        out.update(
            first_motion_s=first,
            retract_rad=retract,
            dip_mm=dip_mm,
            first_ok=first is not None and first <= gc["first_motion_s"],
            retract_ok=retract >= C.gate_interp(gc["retract_rad"], k),
            dip_ok=dip_mm >= C.gate_interp(gc["dip_mm"], k),
        )
        if hold * dt >= 1.0 and beh == "crouch":
            a = tick(0.4)
            band = float((col["z"][a:hold].max() - col["z"][a:hold].min()) * 1000)
            falls = float(col["td"][a:hold].sum()) * 2.0 / max((hold - a) * dt, 1e-6)
            out.update(hold_band_mm=band, footfalls_per_2s=falls,
                       hold_ok=band <= gc["hold_band_mm"] and falls <= gc["hold_footfalls_per_2s"])  # fmt: skip
        if kind in ("walk", "coast"):  # s1b:retrain: the heading while stopping (rev a4's walk_yaw)
            i_e = max(0, min(skill_end, len(col["yaw"])) - 1)
            out["yaw_deg"] = abs(math.degrees(math.atan2(math.sin(col["yaw"][i_e] - col["yaw"][0]),
                                                         math.cos(col["yaw"][i_e] - col["yaw"][0]))))  # fmt: skip
        if kind == "walk":
            s0, s1 = tick(0.6), tick(0.8)
            spd = float(col["speed"][s0 : s1 + 1].mean()) if s1 >= s0 else float(col["speed"][s0])
            path = float(np.sum(np.hypot(np.diff(col["x"][:hold]), np.diff(col["y"][:hold]))))
            out.update(stop_speed=spd, travel_m=path,
                       stop_ok=spd <= gc["walk_speed_mps"] and path <= gc["walk_travel_m"])  # fmt: skip
        out["performed"] = bool(out["retract_ok"] and out["dip_ok"])
        if beh == "hop":
            gh = g["hop"]
            t7 = tick(gh["by_s"])
            hx, hy = math.cos(entry_yaw), math.sin(entry_yaw)
            back = -((col["x"][t7] - col["x"][0]) * hx + (col["y"][t7] - col["y"][0]) * hy)
            # yaw and "ends planted" at the END OF THE SKILL (hold + unwind), both feet down and the
            # trunk < 0.02 m/s — the Mac gate's definition (skilltrain.metrics: planted_end at
            # t_end); tr:train 2026-09-28 (it was the hold's last tick, stricter than plan §6)
            i_end = max(0, min(skill_end, len(col["yaw"])) - 1)
            dyaw = abs(math.degrees(math.atan2(math.sin(col["yaw"][i_end] - col["yaw"][0]),
                                               math.cos(col["yaw"][i_end] - col["yaw"][0]))))  # fmt: skip
            planted = (
                col["feet"][i_end] >= 2
                and col["speed"][i_end] < C.GATES["crouch"]["walk_speed_mps"]
            )
            planted_hold = col["feet"][hold - 1] >= 2
            # s1b:retrain (rev a4): the hop's footfalls in its 0.8 s window and the mean back-step length
            n_win = int(col["td"][: tick(C.HOP_WINDOW_S) + 1].sum())
            out.update(
                hop_footfalls=n_win,
                hop_step_cm=float(100 * back / max(n_win, 1)) if back > 0 else 0.0,
            )
            out.update(back_m=float(back), yaw_deg=dyaw, planted_end=bool(planted),
                       planted_hold_end=bool(planted_hold),
                       performed=bool(back >= gh["back_m"] and dyaw <= gh["yaw_deg"] and planted))  # fmt: skip
    elif beh == "shake":
        gs = g["shake"]
        w = int(round(gs["for_s"] / dt))
        a = tick(0.1)
        amp, fr = _window_osc(col["roll"][a:hold], col["roll_rate"][a:hold], w)
        ok_roll = (
            (np.degrees(amp) >= gs["roll_deg"])
            & (fr >= gs["roll_hz"][0])
            & (fr <= gs["roll_hz"][1])
        )
        hamp, hfr = _window_osc(col["head_yaw"][a:hold], col["head_yaw_rate"][a:hold], w)
        ok_head = (hamp >= gs["head_yaw_rad"]) & (hfr >= gs["head_hz_min"])
        falls = int(col["td"][:hold].sum())
        drift = float(
            math.hypot(col["x"][hold - 1] - col["x"][0], col["y"][hold - 1] - col["y"][0])
        )
        out.update(
            roll_amp_deg=float(np.degrees(amp).max()) if len(amp) else 0.0,
            roll_hz_at_max=float(fr[np.argmax(amp)]) if len(amp) else 0.0,
            head_amp_rad=float(hamp.max()) if len(hamp) else 0.0,
            footfalls=falls, drift_m=drift,
            performed=bool(ok_roll.any() and ok_head.any() and falls <= gs["footfalls"] and drift <= gs["drift_m"]),
        )  # fmt: skip
    elif beh in ("stretch_tall", "stretch_sit"):
        gs = g[beh]
        rise = (col["z"][:hold] - z0) * 1000
        ok = (rise >= gs["rise_mm"]) & (col["cam"][:hold] >= gs["camera_deg"])
        out.update(rise_mm_max=float(rise.max()), cam_deg_max=float(col["cam"][:hold].max()),
                   performed=_held(ok, int(round(gs["hold_s"] / dt))))  # fmt: skip
        if beh == "stretch_sit":
            by = hold + int(round(gs["reseat_by_s"] / dt))
            seg = col["z"][hold : min(by, len(col["z"]))]
            reseated = bool(len(seg) and (seg <= C.GATES["safety"]["reseated_z_m"]).any())  # type: ignore[index]
            out["reseated"] = reseated
            out["performed"] = bool(out["performed"] and reseated)
    else:
        out["performed"] = None
    se = slice(0, skill_end)
    upright = col["tilt"][se] < 45.0
    out.update(
        tau_leg_p95=float(np.percentile(col["tau_leg95"][se], 95)),
        tau_head_p95=float(np.percentile(col["tau_head95"][se], 95)),
        stall_frac=float(col["stall"][se].mean() / 14.0),
        jacc_p99=float(np.percentile(col["jacc99"][se], 99)),
        servo_hits=int(((col["servo_f"][se] > 2.0) & upright).sum()),
    )
    return out


def summarise(
    net: str, cells: list[tuple[str, str]], n: int, traces: dict, info: dict, *, handback_ticks: int
) -> dict:
    vs = {v.name: v for v in C.variants(net)}
    names = [v.name for v in C.variants(net)]
    rows: dict[str, list[dict]] = {}
    n_env = traces["z"].shape[1]
    for i in range(n_env):
        vname = names[int(info["variant"][i])]
        kind = C.ENTRY_KINDS[int(info["kind"][i])]
        o = duck_outcome(net, vs[vname], kind, float(info["level"][i]), i, traces, int(info["hold"][i]),
                         int(info["unwind"][i]), handback_ticks, float(info["entry_yaw"][i]))  # fmt: skip
        rows.setdefault(f"{vname}/{kind}", []).append(o)
    out: dict[str, Any] = {"format": "duckmind-skill-eval-1", "net": net, "cells": {}}
    worst_fall = 0.0
    for key, rs in sorted(rows.items()):
        vname = key.split("/")[0]
        cell: dict[str, Any] = {"n": len(rs)}
        for flag in ("fall_skill", "fall_handback", "performed", "first_ok", "retract_ok", "dip_ok",
                     "hold_ok", "stop_ok", "reseated", "planted_end", "planted_hold_end"):  # fmt: skip
            vals = [r[flag] for r in rs if r.get(flag) is not None]
            if vals:
                cell[flag] = C.rate(sum(bool(v) for v in vals), len(vals))
        for num in ("first_motion_s", "retract_rad", "dip_mm", "hold_band_mm", "footfalls_per_2s", "stop_speed",
                    "travel_m", "back_m", "yaw_deg", "hop_footfalls", "hop_step_cm", "roll_amp_deg", "roll_hz_at_max",
                    "head_amp_rad",
                    "footfalls", "drift_m", "rise_mm_max", "cam_deg_max", "tau_leg_p95", "tau_head_p95", "stall_frac",
                    "jacc_p99", "servo_hits"):  # fmt: skip
            arr = np.array([r[num] for r in rs if r.get(num) is not None], dtype=float)
            if len(arr):
                cell[num] = [round(float(np.percentile(arr, 50)), 4), round(float(np.percentile(arr, 95)), 4),
                             round(float(arr.max()), 4)]  # fmt: skip
        gate = C.fall_rate_gate(vs[vname])
        k, n_ = cell["fall_skill"][0], cell["fall_skill"][1]
        cell["fall_gate"] = gate
        cell["fall_gate_ok"] = bool(k / max(n_, 1) <= gate)
        worst_fall = max(worst_fall, k / max(n_, 1))
        out["cells"][key] = cell
    out["worst_fall_rate"] = round(worst_fall, 5)
    return out


def checkpoint_policy(path: str | Path, device: str) -> tuple[Any, dict[str, Any]]:
    """The deterministic actor of an rsl_rl checkpoint (``actor_state_dict``, the recipe's
    61-512-256-128-14 MLP with its normaliser) as a batched policy function."""
    import torch

    from duckmind.skilltrain import rl_onnx

    ck = torch.load(str(path), map_location=device, weights_only=False)
    actor = rl_onnx.build_actor().to(device)
    actor.load_state_dict(ck["actor_state_dict"], strict=True)
    actor.eval()

    def fn(x: Any) -> Any:
        return rl_onnx.actor_forward(actor, x.to(device))

    info = {
        "ckpt": {
            "file": Path(path).name,
            "iteration": int(ck.get("iter", -1)),
            "sha256": rl_onnx.sha256(path),
            "std": float(actor.distribution.std_param.detach().mean()),
        }
    }
    return fn, info


def main(argv: list[str] | None = None) -> int:
    import torch

    from duckmind.skilltrain import rl_onnx

    ap = argparse.ArgumentParser(
        description="Evaluate a skill ONNX on the training physics (A100 tier 3)"
    )
    ap.add_argument("--net", choices=C.NETS, required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--onnx", help="an exported skill network")
    src.add_argument(
        "--ckpt",
        help="an rsl_rl training checkpoint (model_<it>.pt): its actor, deterministic — the "
        "250-iteration checks of a running attempt, no export needed",
    )
    ap.add_argument(
        "--policies", default=None, help="v5 folder for the hand-back (velstand, alpha_sitstand)"
    )
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--library", default=None)
    ap.add_argument("--foot-mu", default="0.7,1.3")
    ap.add_argument(
        "--servo-lag",
        default="recipe",
        help="recipe (15–30 ms per step) | train (0–40 ms piecewise) | an int: fixed physics steps of 5 ms",
    )
    ap.add_argument("--kinds", default=None, help="only these entry kinds, e.g. walk,coast")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    import mjlab.tasks  # noqa: F401
    import mjlab_microduck.tasks  # noqa: F401

    if args.onnx:
        fn = rl_onnx.torch_policy(args.onnx, args.device)
        source = {"onnx": rl_onnx.describe(args.onnx)}
    else:
        fn, source = checkpoint_policy(args.ckpt, args.device)
    mu = tuple(float(v) for v in args.foot_mu.split(","))
    lag: str | int = (
        args.servo_lag if args.servo_lag in ("recipe", "train") else int(args.servo_lag)
    )
    cells = default_cells(args.net)
    if args.kinds:
        keep = set(args.kinds.split(","))
        cells = [c for c in cells if c[1] in keep]
    res = evaluate(args.net, fn, policies_dir=args.policies, n_per_cell=args.n, device=args.device,
                   library_path=args.library, foot_mu=mu, servo_lag=lag, cells=cells,
                   seed=args.seed)  # type: ignore[arg-type]  # fmt: skip
    res.update(source)
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "cells"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
