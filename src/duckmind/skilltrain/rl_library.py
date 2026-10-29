"""Entry states for the skill tasks: where the previous network leaves the duck (training plan
§3.1: "a state library recorded once from v5 velstand under DR").

**Recorder** (:func:`record`): the official VelStand task physics (``robot_allcollisions``, BAM
XL330, the recipe's final domain randomisation with the floor friction widened to 0.5–1.3, no
pushes), driven by the shipped v5 networks through robotd's path — twist smoothing 0.2, head
smoothing (the neural duck's 1.0), the sit machine (sitstand with the sit flag, the 1 s rise,
then the walk slot), action scale 0.9 walking / 1.0 sitstand, voltage adaptation, low-pass head
0.5 / legs 0.7, the raw previous output fed back. Programs per duck:

======== ================================================================================
stand    zero command (half the ducks with small head offsets), after 2 s
walk     0.35–0.55 m/s, 30 % straight, else a yaw rate of ±0.2–0.6 rad/s (arcs), after 2.5 s
coast    walks 4 s, then the zero command: 0.04–0.7 s after the stop
head     standing, head offsets inside the A100 envelope, new every 0.8–1.5 s
seated   the sit toggle at 0.5 s: seated from 3.5 s
risen    sits, then stands at 3.5 s (1 s sitstand rise): 0.1–1.0 s after velstand takes over
======== ================================================================================

Each snapshot keeps what a skill's first tick needs: the trunk height (above the floor),
orientation, velocities (world), the 14 servo positions and velocities, the network's last raw
output and robotd's filtered previous target. Only upright (tilt < 30°), finite states are kept.
The ``crouched`` kind is synthetic (the skill's own crouch, part-way).

**Loader / sampler** (:func:`cached`, :func:`sample_states`) and **synthetic fallbacks**
(:func:`synthetic_states`, used for any kind the library lacks — and for the CPU smoke test
before a library exists).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, cast

import numpy as np

from duckmind.skilltrain import rl_contract as C

FORMAT = "duckmind-skill-entries-1"
FIELDS = (
    "root_z",
    "root_quat",
    "lin_vel",
    "ang_vel",
    "q",
    "qd",
    "last_raw",
    "prev_target",
    "has_prev",
)
RECORD_KINDS = ("stand", "walk", "coast", "head", "seated", "risen")
PROGRAM_SHARE = {
    "stand": 0.22,
    "walk": 0.30,
    "coast": 0.16,
    "head": 0.14,
    "seated": 0.10,
    "risen": 0.08,
}
HEAD_ALPHA = 1.0  # the neural duck's simulator (fid:motor, owner-approved 「加快」)
CMD_ALPHA = 0.2
SIT_RISE_S = 1.0
TILT_KEEP_DEG = 30.0


# ── loading and sampling ────────────────────────────────────────────────────────────────────


def load(path: str | Path) -> dict[str, Any]:
    """One library file, or several joined by commas (s1b:retrain: the A100 library plus the Mac-like walks)."""
    parts = [p for p in str(path).split(",") if p]
    if len(parts) > 1:
        loaded = [load(p) for p in parts]
        out = {k: np.concatenate([d[k] for d in loaded]) for k in (*FIELDS, "kind")}
        out["meta"] = {**loaded[0]["meta"], "joined": [d["meta"] for d in loaded], "paths": parts}
        return out
    data = np.load(str(path), allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    if meta.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} file")
    out = {k: data[k] for k in (*FIELDS, "kind")}
    out["meta"] = meta
    return out


def cached(env: Any, path: str | None) -> dict[str, Any] | None:
    """The library as device tensors, loaded once per env (``None`` without a file)."""
    if not path:
        return None
    lib = getattr(env, "_skill_library", None)
    if lib is not None and lib.get("path") == str(path):
        return lib
    import torch

    p = Path(path)
    missing = [q for q in str(path).split(",") if q and not Path(q).is_file()]
    if missing:
        print(f"[skilltrain] entry library {missing} not found: synthetic entries only", flush=True)
        env._skill_library = {"path": str(path), "missing": True}
        return env._skill_library
    raw = load(str(path))
    dev = env.device
    lib = {"path": str(path), "missing": False, "meta": raw["meta"]}
    for key in FIELDS:
        arr = raw[key]
        lib[key] = torch.as_tensor(
            arr, device=dev, dtype=torch.bool if key == "has_prev" else torch.float32
        )
    kinds = torch.as_tensor(raw["kind"].astype(np.int64), device=dev)
    lib["by_kind"] = {k: (kinds == i).nonzero().flatten() for i, k in enumerate(C.ENTRY_KINDS)}
    counts = {k: int(v.numel()) for k, v in lib["by_kind"].items()}
    print(
        f"[skilltrain] entry library {p.name if ',' not in str(path) else path}: {sum(counts.values())} states "
        f"{counts}",
        flush=True,
    )
    env._skill_library = lib
    return lib


def sample_states(lib: dict[str, Any] | None, kind: Any, device: str) -> dict[str, Any] | None:
    """One library row per env of the requested kind; ``missing`` marks kinds it lacks."""
    import torch

    if lib is None or lib.get("missing"):
        return None
    n = int(kind.numel())
    rows = torch.zeros(n, dtype=torch.long, device=device)
    missing = torch.ones(n, dtype=torch.bool, device=device)
    for i, name in enumerate(C.ENTRY_KINDS):
        sel = (kind == i).nonzero().flatten()
        pool = lib["by_kind"].get(name)
        if len(sel) == 0 or pool is None or pool.numel() == 0:
            continue
        pick = torch.randint(0, int(pool.numel()), (len(sel),), device=device)
        rows[sel] = pool[pick]
        missing[sel] = False
    out = {key: lib[key][rows].clone() for key in FIELDS}
    out["missing"] = missing
    return out


def synthetic_states(kind: Any, device: str) -> dict[str, Any]:
    """Approximate entries without a library: standing HOME (+noise), a HOME pose sliding at
    walking speed (crude), coasting, head offsets, the SIT keyframe (as SitStand spawns it),
    just risen, and part-way crouches from :mod:`rl_contract`'s kinematic table (feet on the
    floor by construction). The filter state starts at the pose (robotd would have it there)."""
    import torch

    n = int(kind.numel())
    kid = {k: i for i, k in enumerate(C.ENTRY_KINDS)}
    home = torch.tensor(C.HOME, device=device).repeat(n, 1)
    sit = torch.tensor(C.sit_pose(), device=device).repeat(n, 1)
    q = home + torch.randn(n, 14, device=device) * 0.02
    z = C.STAND_Z + 0.002 + torch.rand(n, device=device) * 0.004
    tilt = (torch.rand(n, 2, device=device) * 2 - 1) * math.radians(2.0)
    lin = torch.zeros(n, 3, device=device)
    ang = torch.zeros(n, 3, device=device)

    walk = (kind == kid["walk"]) | (kind == kid["coast"])
    speed = torch.where(
        kind == kid["walk"],
        0.35 + 0.2 * torch.rand(n, device=device),
        0.05 + 0.25 * torch.rand(n, device=device),
    )
    lin[:, 0] = torch.where(walk, speed, lin[:, 0])  # along +x before the event's yaw rotation
    ang[:, 2] = torch.where(
        kind == kid["walk"], (torch.rand(n, device=device) * 2 - 1) * 0.5, ang[:, 2]
    )

    head = kind == kid["head"]
    offs = torch.stack(
        [
            (torch.rand(n, device=device) * 2 - 1) * 0.3,
            (torch.rand(n, device=device) * 2 - 1) * 0.4,
            (torch.rand(n, device=device) * 2 - 1) * 1.0,
            torch.zeros(n, device=device),
        ],  # fmt: skip
        dim=1,
    )
    q[:, list(C.HEAD_IDX)] = torch.where(
        head.unsqueeze(1), home[:, list(C.HEAD_IDX)] + offs, q[:, list(C.HEAD_IDX)]
    )

    seated = kind == kid["seated"]
    q = torch.where(seated.unsqueeze(1), sit + torch.randn(n, 14, device=device) * 0.05, q)
    z = torch.where(seated, C.SIT_Z + 0.002 + torch.rand(n, device=device) * 0.008, z)
    tilt = torch.where(
        seated.unsqueeze(1), (torch.rand(n, 2, device=device) * 2 - 1) * math.radians(5.0), tilt
    )

    risen = kind == kid["risen"]
    risen_q = torch.tensor(C.home_dip_pose(0.004), device=device).repeat(n, 1)
    q = torch.where(risen.unsqueeze(1), risen_q + torch.randn(n, 14, device=device) * 0.02, q)
    z = torch.where(risen, torch.full_like(z, C.STAND_Z - 0.004 + 0.002), z)

    crouched = kind == kid["crouched"]
    lam = 0.3 + 0.7 * torch.rand(n, device=device)
    dip = lam * C.crouch_target(1.0).dip_m
    crouch_q = torch.tensor(C.crouch_target(1.0).pose(), device=device).repeat(n, 1)
    cq = home + lam.unsqueeze(1) * (crouch_q - home) + torch.randn(n, 14, device=device) * 0.02
    q = torch.where(crouched.unsqueeze(1), cq, q)
    z = torch.where(crouched, C.STAND_Z - dip + 0.002, z)

    roll, pitch = tilt[:, 0], tilt[:, 1]
    cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
    cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
    quat = torch.stack([cr * cp, sr * cp, cr * sp, -sr * sp], dim=1)
    return {
        "root_z": z,
        "root_quat": quat,
        "lin_vel": lin,
        "ang_vel": ang,
        "q": q,
        "qd": torch.zeros(n, 14, device=device),
        "last_raw": torch.zeros(n, 14, device=device),
        "prev_target": q.clone(),
        "has_prev": torch.ones(n, dtype=torch.bool, device=device),
        "missing": torch.zeros(n, dtype=torch.bool, device=device),
    }


# ── recording (the official VelStand physics, v5 through robotd's path) ─────────────────────


def build_record_env(
    envs: int, device: str, seed: int, foot_mu: tuple[float, float] = (0.5, 1.3)
) -> Any:
    """The official VelStand play env with the recipe's final DR, no pushes, no prone spawns,
    no curricula, only the NaN guard (as the A100 neuro validation built it)."""
    from copy import deepcopy

    import mjlab.tasks  # noqa: F401
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg("Mjlab-VelStand-Flat-MicroDuck", play=True)
    cfg.scene.num_envs = envs
    cfg.seed = seed
    cfg.episode_length_s = 3600.0
    cfg.curriculum = {}
    for name in ("push_robot", "topple_push", "random_prone_init"):
        cfg.events.pop(name, None)
    for name, command in cfg.commands.items():
        command.resampling_time_range = (1e9, 1e9)
        if name in ("head_pose", "body_pose"):
            command.ranges = ((0.0, 0.0),) * len(command.ranges)
    twist = cfg.commands["twist"]
    twist.rel_standing_envs = 0.0
    twist.rel_heading_envs = twist.rel_forward_envs = twist.rel_turn_in_place_envs = 0.0
    twist.ranges.lin_vel_x = twist.ranges.lin_vel_y = twist.ranges.ang_vel_z = (0.0, 0.0)
    if "set_ground_state" not in cfg.events:
        cfg.events["set_ground_state"] = deepcopy(
            load_env_cfg("Mjlab-SitStand-Flat-MicroDuck", play=True).events["set_ground_state"]
        )
    cfg.events["set_ground_state"].params.update(sitting_prob=0.0, standing_prob=1.0)
    if "randomize_com" in cfg.events:
        cfg.events["randomize_com"].params["ranges"] = (-0.015, 0.015)
    if "randomize_head_com" in cfg.events:
        cfg.events["randomize_head_com"].params["ranges"] = (-0.010, 0.010)
    if "foot_friction" in cfg.events:
        cfg.events["foot_friction"].params["ranges"] = foot_mu
    for name in list(cfg.terminations):
        if name != "nan_state":
            cfg.terminations.pop(name)
    return ManagerBasedRlEnv(cfg, device=device)


def _programs(n: int, gen: Any, device: str, share: dict[str, float] | None = None) -> Any:
    import torch

    kinds = list(PROGRAM_SHARE)
    share = share or PROGRAM_SHARE
    probs = torch.tensor([float(share.get(k, 0.0)) for k in kinds])
    idx = torch.multinomial(probs, n, replacement=True, generator=gen)
    return idx.to(device), kinds


def record(
    out: str | Path,
    policies: str | Path,
    *,
    envs: int = 1024,
    device: str = "cuda:0",
    seconds: float = 8.0,
    seed: int = 1,
    every_s: float = 0.1,
    max_states: int = 200_000,
    log: Any = print,
    programs: dict[str, float] | None = None,
    walk_vx: tuple[float, ...] | None = None,
    walk_straight: float = 0.3,
    walk_keep_s: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Record the entry-state library to ``out`` (npz). Returns the metadata.

    s1b:retrain options (the Mac gate's walking entries: v5 at a commanded 0.40 or 0.55 m/s, straight, the skill
    3.0–3.5 s after the walk command): ``programs`` (program shares, e.g. ``{"walk": 1.0}``), ``walk_vx`` (the
    commanded speeds to pick from instead of 0.35–0.55), ``walk_straight`` (the straight share; the others arc at
    ±0.2–0.6 rad/s), ``walk_keep_s`` (the snapshot window of a walk, s after its command; default from 2.5 s)."""
    import torch

    from duckmind.skilltrain import rl_onnx

    policies = Path(policies)
    for name in ("velstand.onnx", "alpha_sitstand.onnx"):
        got = rl_onnx.sha256(policies / name)
        if got != C.V5_SHA256[name]:
            raise RuntimeError(f"{name}: sha256 {got[:16]} is not the v5 {C.V5_SHA256[name][:16]}")
    t0 = time.time()
    env = build_record_env(envs, device, seed)
    walk_net = rl_onnx.torch_policy(policies / "velstand.onnx", device)
    sit_net = rl_onnx.torch_policy(policies / "alpha_sitstand.onnx", device)
    n = env.num_envs
    gen = torch.Generator().manual_seed(seed)
    prog, kinds = _programs(n, gen, device, programs)
    kid = {k: i for i, k in enumerate(kinds)}
    robot = env.scene["robot"]
    act = robot.actuators[0]
    home = torch.tensor(C.HOME, device=device)
    alpha = torch.tensor(
        [C.LOWPASS_HEAD if i in C.HEAD_IDX else C.LOWPASS_LEGS for i in range(14)], device=device
    )
    vin = getattr(act, "vin_tensor", None)
    vf = (
        (C.NOMINAL_V / vin.reshape(n, 1).clamp(*C.VIN_CLAMP))
        if vin is not None
        else torch.ones(n, 1, device=device)
    )
    twist_cmd = env.command_manager.get_term("twist").command
    head_cmd = env.command_manager.get_term("head_pose").command
    body_cmd = env.command_manager.get_term("body_pose").command
    dt = env.step_dt
    ticks = int(round(seconds / dt))
    every = max(1, int(round(every_s / dt)))
    r = lambda lo, hi: lo + (hi - lo) * torch.rand(n, generator=gen).to(device)  # noqa: E731
    vx = r(0.35, 0.55)
    if walk_vx:
        pick = torch.randint(0, len(walk_vx), (n,), generator=gen).to(device)
        vx = torch.tensor([float(v) for v in walk_vx], device=device)[pick]
    wz = torch.where(torch.rand(n, generator=gen).to(device) < walk_straight, torch.zeros(n, device=device),
                     torch.sign(r(-1, 1)) * r(0.2, 0.6))  # fmt: skip
    stop_t = 4.0
    head_offsets = torch.zeros(n, 4, device=device)
    next_head = torch.zeros(n, device=device)
    small_head = torch.rand(n, generator=gen).to(device) < 0.5
    twist_ema = torch.zeros(n, 3, device=device)
    head_ema = torch.zeros(n, 4, device=device)
    SIT_UP, SIT_SITTING, SIT_RISING = 0, 1, 2
    sit = torch.zeros(n, dtype=torch.long, device=device)
    rise_left = torch.zeros(n, device=device)
    handover_t = torch.full((n,), -1.0, device=device)
    prev_target: Any = None
    last_raw = torch.zeros(n, 14, device=device)
    obs, _ = env.reset(seed=seed)
    rows: dict[str, list] = {k: [] for k in (*FIELDS, "kind")}
    keep_kind = torch.tensor([C.ENTRY_KINDS.index(k) for k in kinds], device=device)[prog]
    for k in range(ticks + 1):
        t = k * dt
        # client intents for this tick
        target_twist = torch.zeros(n, 3, device=device)
        walking = (prog == kid["walk"]) | ((prog == kid["coast"]) & (t < stop_t))
        target_twist[:, 0] = torch.where(walking, vx, 0.0)
        target_twist[:, 2] = torch.where(walking, wz, 0.0)
        new_head = (prog == kid["head"]) & (t >= next_head)
        if bool(new_head.any()):
            head_offsets[new_head] = torch.stack(
                [r(-0.3, 0.3), r(-0.4, 0.4), r(-1.0, 1.0), r(-0.1, 0.1)], dim=1
            )[new_head]
            next_head = torch.where(new_head, t + r(0.8, 1.5), next_head)
        stand_small = (prog == kid["stand"]) & small_head & (k == 0)
        if bool(stand_small.any()):
            head_offsets[stand_small] = torch.stack(
                [r(-0.08, 0.08), r(-0.08, 0.08), r(-0.15, 0.15), torch.zeros(n, device=device)],
                dim=1,
            )[stand_small]
        # the sit machine (robotd: requests first, then the rise timer)
        want_sit = ((prog == kid["seated"]) | (prog == kid["risen"])) & (abs(t - 0.5) < dt / 2)
        want_stand = (prog == kid["risen"]) & (abs(t - 3.5) < dt / 2) & (sit == SIT_SITTING)
        sit = torch.where(want_sit, SIT_SITTING, sit)
        rise_left = torch.where(want_stand, torch.full_like(rise_left, SIT_RISE_S), rise_left)
        sit = torch.where(want_stand, SIT_RISING, sit)
        done_rise = (sit == SIT_RISING) & (rise_left <= 0)
        handover_t = torch.where(done_rise, torch.full_like(handover_t, t), handover_t)
        sit = torch.where(done_rise, SIT_UP, sit)
        twist_ema += CMD_ALPHA * (target_twist - twist_ema)
        head_ema += HEAD_ALPHA * (head_offsets - head_ema)
        eff = torch.where((sit == SIT_UP).unsqueeze(1), twist_ema, torch.zeros_like(twist_ema))
        eff[:, 0] = torch.where(sit == SIT_SITTING, 1.0, eff[:, 0])
        twist_cmd[:] = eff
        head_cmd[:] = head_ema
        body_cmd[:] = 0.0
        rise_left = torch.where(sit == SIT_RISING, rise_left - dt, rise_left)
        if k == 0:
            obs["actor"][:, 48:61] = torch.cat([twist_cmd, head_cmd, body_cmd], dim=1)
        x = obs["actor"]
        on_sitstand = sit != SIT_UP
        raw = torch.where(on_sitstand.unsqueeze(1), sit_net(x), walk_net(x))
        scale = torch.where(on_sitstand, C.SITSTAND_ACTION_SCALE, C.WALK_ACTION_SCALE).unsqueeze(1)
        target = home + scale * vf * raw
        if prev_target is not None:
            target = alpha * target + (1 - alpha) * prev_target
        target = target.clamp(-C.ACTUATOR_CLAMP, C.ACTUATOR_CLAMP)
        # snapshot the state this tick's network acted from (the "last" values are the tick before)
        if k > 0 and k % every == 0:
            tsnap = t
            ok = _keep_mask(prog, kid, tsnap, stop_t, handover_t, sit, SIT_SITTING, walk_keep_s)
            ok &= _upright_finite(env)
            if bool(ok.any()):
                _snapshot(env, rows, ok, last_raw, prev_target, keep_kind)
        prev_target = target
        last_raw = raw
        obs, _rew, _term, _trunc, _ = env.step(target - home)
        obs["actor"][:, 34:48] = raw
        if sum(len(v) for v in rows["kind"]) >= max_states:
            break
    arrays = {k: np.concatenate(v) if v else np.zeros((0,)) for k, v in rows.items()}
    counts = {
        name: int((arrays["kind"] == C.ENTRY_KINDS.index(name)).sum()) for name in RECORD_KINDS
    }
    meta = {
        "format": FORMAT,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": "Mjlab-VelStand-Flat-MicroDuck (play cfg), v5 velstand + alpha_sitstand via robotd's path",
        "v5_sha256": {k: C.V5_SHA256[k] for k in ("velstand.onnx", "alpha_sitstand.onnx")},
        "microduck_rl_commit": C.MICRODUCK_RL_COMMIT,
        "envs": n,
        "seconds": seconds,
        "seed": seed,
        "device": device,
        "dr": "trunk CoM ±15 mm, head CoM ±10 mm, foot μ 0.5–1.3, mass ±5 %, BAM friction/armature ±10 %, "
        "battery 6.5–8.2 V with sag, 15–30 ms delay, encoder bias, IMU tilt, obs noise",
        "robotd": {
            "cmd_alpha": CMD_ALPHA,
            "head_alpha": HEAD_ALPHA,
            "scale_walk": C.WALK_ACTION_SCALE,
            "scale_sitstand": C.SITSTAND_ACTION_SCALE,
            "lowpass": [C.LOWPASS_HEAD, C.LOWPASS_LEGS],
            "voltage_adapt": True,
        },  # fmt: skip
        "counts": counts,
        "wall_s": round(time.time() - t0, 1),
        "programs": programs or PROGRAM_SHARE,
        "walk": {
            "vx": list(walk_vx) if walk_vx else [0.35, 0.55],
            "straight": walk_straight,
            "keep_s": list(walk_keep_s) if walk_keep_s else [2.5, seconds],
        },  # fmt: skip
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out),
        meta=np.array(json.dumps(meta)),
        kind=arrays["kind"].astype(np.int8),
        **cast(
            dict[str, Any],
            {k: arrays[k].astype(bool if k == "has_prev" else np.float32) for k in FIELDS},
        ),
    )
    log(
        f"[skilltrain] recorded {sum(counts.values())} entry states {counts} in {meta['wall_s']} s → {out}"
    )
    try:
        env.close()
    except Exception:  # noqa: BLE001
        pass
    return meta


def _keep_mask(prog, kid, t, stop_t, handover_t, sit, sitting, walk_keep_s=None) -> Any:
    import torch

    after = lambda a: torch.full_like(handover_t, t) >= a  # noqa: E731
    keep = torch.zeros_like(prog, dtype=torch.bool)
    keep |= (prog == kid["stand"]) & (t >= 2.0)
    wa, wb = walk_keep_s or (2.5, float("inf"))
    keep |= (prog == kid["walk"]) & (t >= wa) & (t <= wb)
    keep |= (prog == kid["coast"]) & (t >= stop_t + 0.04) & (t <= stop_t + 0.7)
    keep |= (prog == kid["head"]) & (t >= 2.0)
    keep |= (prog == kid["seated"]) & (t >= 3.5) & (sit == sitting)
    since = t - handover_t
    keep |= (prog == kid["risen"]) & (handover_t >= 0) & (since >= 0.1) & (since <= 1.0)
    del after
    return keep


def _upright_finite(env: Any) -> Any:
    import torch

    robot = env.scene["robot"]
    q = robot.data.root_link_quat_w
    cos_tilt = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
    finite = torch.isfinite(robot.data.joint_pos).all(dim=1) & torch.isfinite(q).all(dim=1)
    return finite & (cos_tilt > math.cos(math.radians(TILT_KEEP_DEG)))


def _snapshot(env, rows, ok, last_raw, prev_target, keep_kind) -> None:
    import torch
    from mjlab_microduck.tasks import mdp as microduck_mdp

    robot = env.scene["robot"]
    d = robot.data
    servo = microduck_mdp._servo_joint_ids(env, robot)
    z = d.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    pieces = {
        "root_z": z,
        "root_quat": d.root_link_quat_w,
        "lin_vel": d.root_link_lin_vel_w,
        "ang_vel": d.root_link_ang_vel_w,
        "q": d.joint_pos[:, servo],
        "qd": d.joint_vel[:, servo],
        "last_raw": last_raw,
        "prev_target": prev_target if prev_target is not None else d.joint_pos[:, servo],
        "has_prev": torch.full_like(z, prev_target is not None, dtype=torch.bool),
        "kind": keep_kind,
    }
    for key, value in pieces.items():
        rows[key].append(value[ok].detach().cpu().numpy())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Record the skill entry-state library (v5 on the official physics)"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--policies", required=True, help="folder with velstand.onnx and alpha_sitstand.onnx (v5)"
    )
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--every-s", type=float, default=0.1)
    ap.add_argument("--max-states", type=int, default=200_000)
    ap.add_argument(
        "--programs", default=None, help="program shares, e.g. walk=1.0 (s1b: the Mac-like walks)"
    )
    ap.add_argument(
        "--walk-vx", default=None, help="commanded walking speeds to pick from, e.g. 0.40,0.55"
    )
    ap.add_argument("--walk-straight", type=float, default=0.3)
    ap.add_argument(
        "--walk-keep",
        default=None,
        help="snapshot window of a walk in s after its command, e.g. 2.8,3.8",
    )
    args = ap.parse_args(argv)
    programs = None
    if args.programs:
        programs = {k: float(v) for k, v in (kv.split("=") for kv in args.programs.split(","))}
    walk_vx = tuple(float(v) for v in args.walk_vx.split(",")) if args.walk_vx else None
    keep = tuple(float(v) for v in args.walk_keep.split(",")) if args.walk_keep else None
    record(args.out, args.policies, envs=args.envs, device=args.device, seconds=args.seconds, seed=args.seed,
           every_s=args.every_s, max_states=args.max_states, programs=programs, walk_vx=walk_vx,
           walk_straight=args.walk_straight, walk_keep_s=keep)  # type: ignore[arg-type]  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
