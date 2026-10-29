"""MDP pieces of the three B-skill tasks (b:feasibility, 2026-09-28): ``bskstep`` (turn on the spot,
march in place), ``bskmat`` (step up onto / down off a 2–5 cm mat), ``bskbow`` (play-bow, lie low).

One episode = one robotd skill window as robotd runs it, on Stage 1's machinery (``rl_mdp``'s
:class:`~duckmind.skilltrain.rl_mdp.SkillCommand` phases hold → unwind; ``rl_mdp.SkillPlan`` for the
posture blend; ``rl_library`` entries from v5 with robotd's filter state; ``rl_action``'s robotd path):

* **held entries** (``turn_left``/``turn_right``/``march``/``lie_low``: chain, re-requested): a hold of
  1–6 s; the stepping skills then end (the brain's ``skill_abort`` hands to v5 velstand — the Mac gate
  measures that hand-back); ``lie_low`` ends with ``bow_release`` (the zero command, 0.8 s) and the
  hand-back bonus;
* **one-shots** (``step_up``/``step_down``/``bow``): the entry's window, then robotd's unwind (zero
  command) and the hand-back bonus at the last tick.

Sign convention (Stage 1's, checked by the trainer): reward functions return ≥ 0 with positive
weights, cost functions ≥ 0 with negative weights; the mat progress potential is signed by design.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from mjlab_microduck.tasks import mdp as microduck_mdp

from duckmind.skilltrain import bsk_contract as B
from duckmind.skilltrain import rl_mdp as RM

HOLD, UNWIND, HANDBACK, DONE = RM.HOLD, RM.UNWIND, RM.HANDBACK, RM.DONE
BEH_ID = {b: i for i, b in enumerate(B.BEHAVIOURS)}


def _kind_ids() -> dict[str, int]:
    from duckmind.skilltrain import rl_contract as C  # the library's own kind order (Stage 1's)

    return {k: i for i, k in enumerate(C.ENTRY_KINDS)}


KIND_ID = _kind_ids()
G = 9.81


class BskPlan:
    """Per-env B-skill state (the posture blend lives in ``rl_mdp.SkillPlan``)."""

    def __init__(self, n: int, device: str):
        z = lambda: torch.zeros(n, device=device)  # noqa: E731
        zb = lambda: torch.zeros(n, dtype=torch.bool, device=device)  # noqa: E731
        self.behaviour = torch.zeros(n, dtype=torch.long, device=device)
        self.variant = torch.zeros(n, dtype=torch.long, device=device)
        self.sign = z()  # turn direction (+1 left, −1 right), 0 march
        self.lift_target = z()
        self.ground_ref = torch.full((n, 2), float("nan"), device=device)
        self.mat_h, self.edge_x, self.down = z(), z(), zb()
        self.z0 = z()
        self.phi = z()
        self.success, self.fell, self.done_flag = zb(), zb(), zb()
        self.max_tilt = z()
        self.prev_max_tilt = z()  # rev b2: the episode's max tilt before this tick (fall_event)
        self.steps = z()
        self.yaw_acc = z()
        self.prev_yaw = z()
        self.beak_entry_z = torch.full((n,), float("nan"), device=device)
        self.since_td = torch.zeros(
            n, 2, device=device
        )  # s since each foot's last real touchdown (rev b1)


def bsk_plan(env: Any) -> BskPlan:
    plan = getattr(env, "bsk_plan", None)
    if plan is None:
        plan = BskPlan(env.num_envs, env.device)
        env.bsk_plan = plan
    return plan


def _robot(env: Any):
    return env.scene["robot"]


def _feet_sites(env: Any) -> list[int]:
    cache = env.__dict__.setdefault("_bsk_site_cache", {})
    if "feet" not in cache:
        ids, _ = _robot(env).find_sites(["left_foot", "right_foot"], preserve_order=True)
        cache["feet"] = ids
    return cache["feet"]


def _phase(env: Any) -> torch.Tensor:
    return RM.skill_term(env).phase


def _hold(env: Any) -> torch.Tensor:
    return (_phase(env) == HOLD).float()


def _in_skill(env: Any) -> torch.Tensor:
    ph = _phase(env)
    return ((ph == HOLD) | (ph == UNWIND)).float()


def _t_hold(env: Any) -> torch.Tensor:
    return RM.skill_term(env).tick.float() * env.step_dt


def _is(env: Any, *names: str) -> torch.Tensor:
    beh = bsk_plan(env).behaviour
    out = torch.zeros_like(beh, dtype=torch.bool)
    for n in names:
        out |= beh == BEH_ID[n]
    return out


def _yaw(env: Any) -> torch.Tensor:
    return RM._yaw(_robot(env).data.root_link_quat_w)


def _tilt_deg(env: Any) -> torch.Tensor:
    q = _robot(env).data.root_link_quat_w
    c = torch.nan_to_num(1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2), nan=1.0)
    return torch.rad2deg(torch.acos(c.clamp(-1.0, 1.0)))


def _feet_contact(env: Any) -> torch.Tensor:
    return env.scene.sensors["feet_ground_contact"].data.found[:, :2] > 0


def _feet_z(env: Any) -> torch.Tensor:
    """(N, 2): the foot sites' height above the env's floor (the site is at the sole)."""
    p = _robot(env).data.site_pos_w[:, _feet_sites(env)]
    return p[:, :, 2] - env.scene.env_origins[:, 2].unsqueeze(1)


def _feet_x(env: Any) -> torch.Tensor:
    return _robot(env).data.site_pos_w[:, _feet_sites(env), 0]


# ── the episode ─────────────────────────────────────────────────────────────────────────────


def reset_bsk_episode(
    env: Any,
    env_ids: Any,
    net: str = "bskstep",
    mix_stage: str = "full",
    library_path: str | None = None,
    eval_cells: list[tuple[str, str]] | None = None,
    nominal_windows: bool = False,
    entry_mix_overrides: dict[str, dict[str, float]] | None = None,
) -> None:
    """Choose each env's entry (table entry, entry kind, windows, targets) and write its state."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    ids = env_ids.to(env.device, dtype=torch.long)
    n = len(ids)
    if n == 0:
        return
    dev = env.device
    es = B.entries(net)
    trained = [e for e in es if e.behaviour != "release"]
    names = [e.name for e in trained]
    if eval_cells:
        cells = [(names.index(v), KIND_ID[k]) for v, k in eval_cells]
        c = ids % len(cells)
        vidx = torch.tensor([x[0] for x in cells], device=dev)[c]
        kind = torch.tensor([x[1] for x in cells], device=dev)[c]
    else:
        w = torch.tensor([B.VARIANT_WEIGHTS[net][nm] for nm in names], device=dev)
        vidx = torch.multinomial(w / w.sum(), n, replacement=True)
        kind = torch.zeros(n, dtype=torch.long, device=dev)
        for i, e in enumerate(trained):
            sel = (vidx == i).nonzero().flatten()
            if len(sel) == 0:
                continue
            mix = B.entry_mix(e.behaviour)
            if mix_stage == "early" and "stand" in mix:
                mix = {k: (0.8 if k == "stand" else 0.2 * v) for k, v in mix.items()}
            if eval_cells is None and entry_mix_overrides and e.behaviour in entry_mix_overrides:
                mix = dict(entry_mix_overrides[e.behaviour])
            ks = list(mix)
            probs = torch.tensor([mix[k] for k in ks], device=dev)
            pick = torch.multinomial(probs / probs.sum(), len(sel), replacement=True)
            kind[sel] = torch.tensor([KIND_ID[k] for k in ks], device=dev)[pick]

    sp, bp, dt = RM.plan_of(env), bsk_plan(env), env.step_dt
    RM.entry_of(env)
    home = torch.tensor(B.HOME, device=dev)
    beh = torch.zeros(n, dtype=torch.long, device=dev)
    command = torch.zeros(n, 3, device=dev)
    unwind = torch.zeros(n, 3, device=dev)
    hold_s = torch.zeros(n, device=dev)
    unwind_s = torch.zeros(n, device=dev)
    sign = torch.zeros(n, device=dev)
    lift = torch.zeros(n, device=dev)
    q_full = home.repeat(n, 1)
    z_full = torch.full((n,), B.STAND_Z, device=dev)
    pitch_full = torch.zeros(n, device=dev)
    goal = torch.zeros(n, device=dev)
    ramp = torch.full((n,), 0.3, device=dev)
    for i, e in enumerate(trained):
        sel = (vidx == i).nonzero().flatten()
        m = len(sel)
        if m == 0:
            continue
        beh[sel] = BEH_ID[e.behaviour]
        command[sel] = torch.tensor(e.command, device=dev)
        if e.behaviour in B.HOLD_RANGE_S:
            lo, hi = B.HOLD_RANGE_S[e.behaviour]
            h = (
                torch.full((m,), 0.5 * (lo + hi), device=dev)
                if nominal_windows
                else lo + (hi - lo) * torch.rand(m, device=dev)
            )
        else:
            h = torch.full((m,), e.duration, device=dev)
            if not nominal_windows:
                jit = torch.rand(m, device=dev) < 0.3
                h = torch.where(
                    jit,
                    h
                    * (
                        B.HOLD_JITTER[0]
                        + (B.HOLD_JITTER[1] - B.HOLD_JITTER[0]) * torch.rand(m, device=dev)
                    ),
                    h,
                )
        hold_s[sel] = h
        if e.behaviour == "lie":
            unwind_s[sel] = 0.8  # bow_release: the zero command
        else:
            unwind_s[sel] = e.unwind_s
        unwind[sel] = torch.tensor(e.unwind, device=dev)
        sign[sel] = float(e.level) if e.behaviour == "turn" else 0.0
        lift[sel] = B.LIFT_TARGET_M.get(e.behaviour, 0.0)
        if e.behaviour in ("bow", "lie"):
            post = B.BOW if e.behaviour == "bow" else B.LIE
            q_full[sel] = torch.tensor(post.pose(), device=dev)
            z_full[sel] = B.STAND_Z - post.drop_m
            pitch_full[sel] = -math.radians(post.pitch_deg)  # rl_mdp.pitch_track: + = nose up
            goal[sel] = 1.0
            ramp[sel] = post.ramp_s

    sp.variant[ids], sp.behaviour[ids], sp.kind[ids] = vidx, beh, kind
    sp.level[ids] = 1.0
    sp.command[ids], sp.unwind_cmd[ids] = command, unwind
    sp.hold_ticks[ids] = torch.round(hold_s / dt).long()
    sp.unwind_ticks[ids] = torch.round(unwind_s / dt).long()
    sp.handback_ticks[ids] = 0
    sp.cut[ids] = False
    sp.seated[ids] = False
    sp.handback_cmd[ids] = 0.0
    sp.q_base[ids] = home.repeat(n, 1)
    sp.q_full[ids] = q_full
    sp.z_base[ids] = B.STAND_Z
    sp.z_full[ids] = z_full
    sp.pitch_full[ids] = pitch_full
    sp.cam_full[ids] = 0.0
    sp.goal_hold[ids] = goal
    sp.ramp_up[ids] = ramp
    sp.ramp_down[ids] = 0.5
    sp.bands[ids] = 0.0
    sp.head_w[ids] = 1.0
    sp.hop[ids] = False
    sp.shake[ids] = False
    sp.stretch[ids] = False

    bp.behaviour[ids], bp.variant[ids] = beh, vidx
    bp.sign[ids], bp.lift_target[ids] = sign, lift
    bp.ground_ref[ids] = float("nan")
    bp.phi[ids] = 0.0
    bp.success[ids] = False
    bp.fell[ids] = False
    bp.done_flag[ids] = False
    bp.max_tilt[ids] = 0.0
    bp.prev_max_tilt[ids] = 0.0
    bp.steps[ids] = 0.0
    bp.yaw_acc[ids] = 0.0
    bp.beak_entry_z[ids] = float("nan")

    if net == "bskmat":
        _write_mat_entry(env, ids, kind, library_path, beh)
    else:
        RM._write_entry_state(env, ids, kind, library_path, 0.3, sp.level[ids])
    sp.alpha0[ids] = 0.0
    bp.prev_yaw[ids] = sp.entry_yaw[ids]
    bp.z0[ids] = sp.entry_z[ids]


def mat_height_of(levels: torch.Tensor | int, rows: int = B.MAT_ROWS) -> Any:
    lo, hi = B.MAT_HEIGHT_RANGE
    return lo + (hi - lo) * (levels / max(1, rows - 1))


def _write_mat_entry(env: Any, ids: torch.Tensor, kind: torch.Tensor, library_path: str | None,
                     beh: torch.Tensor) -> None:  # fmt: skip
    """A library (or synthetic) standing state, turned to face the mat ±8°, its trunk centre 7–14 cm
    before the near edge (step_up, on the floor) or before the far edge on the mat (step_down)."""
    from duckmind.skilltrain import rl_library

    dev, n = env.device, len(ids)
    robot = _robot(env)
    sp, bp, entry = RM.plan_of(env), bsk_plan(env), RM.entry_of(env)
    terrain = env.scene.terrain
    levels = (
        terrain.terrain_levels[ids].float()
        if getattr(terrain, "terrain_levels", None) is not None
        else torch.zeros(n, device=dev)
    )
    h = mat_height_of(levels)
    lib = rl_library.cached(env, library_path)
    state = rl_library.sample_states(lib, kind, dev) if lib is not None else None
    synth = rl_library.synthetic_states(kind, dev)
    if state is None:
        state = synth
    else:
        miss = state["missing"]
        for key in (
            "root_z",
            "root_quat",
            "lin_vel",
            "ang_vel",
            "q",
            "qd",
            "last_raw",
            "prev_target",
            "has_prev",
        ):
            a, b = state[key], synth[key]
            state[key] = torch.where(miss.view(-1, *([1] * (a.dim() - 1))), b, a)
    down = beh == BEH_ID["step_down"]
    aim = math.radians(B.MAT_AIM_DEG) * (2 * torch.rand(n, device=dev) - 1)
    # the library state's own yaw is replaced: heading = aim (facing +x, the mat's normal)
    q = state["root_quat"]
    own = RM._yaw(q)
    quat = RM._quat_mul(RM._quat_yaw(aim - own), q)
    quat = quat / quat.norm(dim=-1, keepdim=True)
    origin = env.scene.env_origins[ids]  # on the near edge, floor level (hop_terrain)
    d = B.MAT_START_D[0] + (B.MAT_START_D[1] - B.MAT_START_D[0]) * torch.rand(n, device=dev)
    edge = origin[:, 0] + torch.where(down, torch.full_like(d, B.MAT_DEPTH_M), torch.zeros_like(d))
    x = edge - d
    y = origin[:, 1] + 0.3 * (2 * torch.rand(n, device=dev) - 1)
    z = state["root_z"] + origin[:, 2] + torch.where(down, h, torch.zeros_like(h)) + 0.002
    pose = torch.cat([x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1), quat], dim=1)
    rot = aim - own
    vel = torch.cat([RM._rot_z(state["lin_vel"], rot), RM._rot_z(state["ang_vel"], rot)], dim=1)
    robot.write_root_link_pose_to_sim(pose, env_ids=ids)
    robot.write_root_link_velocity_to_sim(vel, env_ids=ids)
    servo = microduck_mdp._servo_joint_ids(env, robot)
    q_all = robot.data.default_joint_pos[ids].clone()
    qd_all = torch.zeros_like(q_all)
    q_all[:, servo] = state["q"]
    qd_all[:, servo] = state["qd"]
    robot.write_joint_state_to_sim(q_all, qd_all, env_ids=ids)
    entry.prev_target[ids] = state["prev_target"]
    entry.has_prev[ids] = state["has_prev"]
    entry.last_raw[ids] = state["last_raw"]
    sp.entry_xy[ids] = torch.stack([x, y], dim=1)
    sp.entry_yaw[ids] = RM._yaw(quat)
    sp.entry_z[ids] = z - origin[:, 2]
    bp.mat_h[ids] = h
    bp.edge_x[ids] = edge
    bp.down[ids] = down


# ── per-step bookkeeping (called by the first reward term) ────────────────────────────────────


def bookkeeping(env: Any) -> torch.Tensor:
    """Updates the foot ground references, tilt, yaw accumulation and footfall counts; returns 0."""
    bp = bsk_plan(env)
    contact = _feet_contact(env)
    fz = _feet_z(env)
    ref = torch.where(contact, fz, bp.ground_ref)
    bp.ground_ref = torch.where(torch.isnan(bp.ground_ref) & ~contact, fz, ref)
    tilt = _tilt_deg(env)
    fresh = env.episode_length_buf <= 1
    bp.prev_max_tilt = torch.where(fresh, torch.zeros_like(tilt), bp.max_tilt)
    bp.max_tilt = torch.where(fresh, tilt, torch.maximum(bp.max_tilt, tilt))
    yaw = _yaw(env)
    step = RM._wrap(yaw - bp.prev_yaw)
    bp.yaw_acc = torch.where(
        fresh, torch.zeros_like(step), bp.yaw_acc + step * (_phase(env) == HOLD)
    )
    bp.prev_yaw = yaw
    bp.fell |= RM._cos_tilt(env) < math.cos(math.radians(60.0))
    sensor = env.scene.sensors["feet_ground_contact"]
    first = sensor.compute_first_contact(env.step_dt)[:, :2]
    real = first & (sensor.data.last_air_time[:, :2] >= B.STEP_AIR_S[0])
    bp.since_td = torch.where(
        real | fresh.unsqueeze(1), torch.zeros_like(bp.since_td), bp.since_td + env.step_dt
    )
    return torch.zeros(env.num_envs, device=env.device)


def _real_steps(env: Any) -> torch.Tensor:
    sensor = env.scene.sensors["feet_ground_contact"]
    first = sensor.compute_first_contact(env.step_dt)[:, :2]
    air = sensor.data.last_air_time[:, :2]
    lo, hi = B.STEP_AIR_S
    return (first & (air >= lo) & (air <= hi)).float().sum(dim=1)


def _lift(env: Any) -> torch.Tensor:
    """(N, 2) each foot's height above where it last stood (0 while in contact)."""
    bp = bsk_plan(env)
    lift = (_feet_z(env) - torch.nan_to_num(bp.ground_ref, nan=0.0)).clamp(min=0.0)
    return torch.where(_feet_contact(env), torch.zeros_like(lift), lift)


# ── rewards: stepping in place (bskstep) ─────────────────────────────────────────────────────


def yaw_track(env: Any, std: float = 0.35) -> torch.Tensor:
    """Trunk yaw rate toward the entry's (turn: ±TURN_RATE after a 0.4 s ramp; march: 0). The
    stepping task's first reward term: it runs the per-step bookkeeping."""
    bookkeeping(env)
    bp = bsk_plan(env)
    ramp = (_t_hold(env) / B.TURN_RAMP_S).clamp(0.0, 1.0)
    target = bp.sign * B.TURN_RATE * ramp
    wz = torch.nan_to_num(_robot(env).data.root_link_ang_vel_w[:, 2], nan=0.0)
    return torch.exp(-(((wz - target) / std) ** 2)) * _hold(env) * _is(env, "turn", "march")


def stay(env: Any, std: float = 0.03) -> torch.Tensor:
    """The trunk centre near where the skill began (in place)."""
    sp = RM.plan_of(env)
    d = (_robot(env).data.root_link_pos_w[:, :2] - sp.entry_xy).norm(dim=1)
    return torch.exp(-((d / std) ** 2)) * _hold(env) * _is(env, "turn", "march")


def travel(env: Any) -> torch.Tensor:
    """Cost: travel from the entry point beyond STAY_FREE_M (m)."""
    sp = RM.plan_of(env)
    d = (_robot(env).data.root_link_pos_w[:, :2] - sp.entry_xy).norm(dim=1)
    return (d - B.STAY_FREE_M).clamp(min=0.0) * _in_skill(env) * _is(env, "turn", "march")


def trunk_speed(env: Any, std: float = 0.08) -> torch.Tensor:
    v = torch.nan_to_num(_robot(env).data.root_link_lin_vel_w[:, :2], nan=0.0).norm(dim=1)
    return torch.exp(-((v / std) ** 2)) * _hold(env) * _is(env, "turn", "march")


def steps(env: Any) -> torch.Tensor:
    """A real step (a foot airborne 0.08–0.45 s) pays, in the hold of the stepping skills."""
    s = _real_steps(env)
    bsk_plan(env).steps += s * _hold(env)
    return s * _hold(env) * _is(env, "turn", "march")


def clearance(env: Any) -> torch.Tensor:
    """Airborne feet high above where they stood, up to the behaviour's lift target."""
    bp = bsk_plan(env)
    tgt = bp.lift_target.clamp(min=1e-3).unsqueeze(1)
    lift = _lift(env)
    airborne = ~_feet_contact(env)
    score = ((lift / tgt).clamp(0.0, 1.0) * airborne).sum(dim=1)
    return score * _hold(env) * _is(env, "turn", "march")


def _stepping(env: Any, window_s: float = B.STEP_WINDOW_S) -> torch.Tensor:
    """1 when both feet made a real step within ``window_s`` (rev b1: in-place rewards need stepping)."""
    return (bsk_plan(env).since_td < window_s).all(dim=1).float()


def stay_stepping(env: Any, std: float = 0.03) -> torch.Tensor:
    """rev b1: ``stay`` only while stepping (the b0 calibration run learned to stand still and slide)."""
    return stay(env, std) * _stepping(env)


def speed_stepping(env: Any, std: float = 0.08) -> torch.Tensor:
    return trunk_speed(env, std) * _stepping(env)


def yaw_progress(env: Any) -> torch.Tensor:
    """rev b1: turn rate in the commanded direction / TURN_RATE, clipped to [0, 1] — a gradient from a
    standstill (the b0 Gaussian σ 0.35 rad/s gave none at 0.2 rad/s) — paid only while stepping."""
    bp = bsk_plan(env)
    wz = torch.nan_to_num(_robot(env).data.root_link_ang_vel_w[:, 2], nan=0.0)
    return (
        (bp.sign * wz / B.TURN_RATE).clamp(0.0, 1.0)
        * _stepping(env)
        * _hold(env)
        * _is(env, "turn")
    )


def cadence(env: Any) -> torch.Tensor:
    """rev b1 cost: seconds a foot has gone without a real step beyond CADENCE_S, in the hold after the
    first 0.5 s (a march or a turn on the spot is made of steps)."""
    late = (bsk_plan(env).since_td - B.CADENCE_S).clamp(min=0.0).sum(dim=1)
    return late * _hold(env) * (_t_hold(env) > 0.5).float() * _is(env, "turn", "march")


def foot_slide(env: Any) -> torch.Tensor:
    """rev b1 cost: horizontal speed of feet in contact (m/s) — turning by sliding the soles is not a step."""
    v = _robot(env).data.site_lin_vel_w[:, _feet_sites(env), :2].norm(dim=2)
    return (v * _feet_contact(env)).sum(dim=1) * _hold(env) * _is(env, "turn", "march")


def double_flight(env: Any) -> torch.Tensor:
    """Cost: both feet off the floor while stepping in place (a march is not a hop)."""
    both = (~_feet_contact(env)).all(dim=1).float()
    return both * _in_skill(env) * _is(env, "turn", "march", "bow", "lie")


def heading_drift(env: Any) -> torch.Tensor:
    """Cost: |heading change| (rad) during a march (a turn is paid by yaw_track)."""
    sp = RM.plan_of(env)
    d = RM._wrap(_yaw(env) - sp.entry_yaw).abs()
    return d * _hold(env) * _is(env, "march")


def march_heading(env: Any, std_deg: float = 8.0) -> torch.Tensor:
    """Reward keeping the entry heading; turn commands are deliberately excluded."""
    d = RM._wrap(_yaw(env) - RM.plan_of(env).entry_yaw)
    return torch.exp(-((d / math.radians(std_deg)) ** 2)) * _hold(env) * _is(env, "march")


def touchdown_impact(env: Any) -> torch.Tensor:
    """Cost: a foot's touchdown force above IMPACT_BW × body weight (N/(mg)): a stomp, not a slam."""
    sensor = env.scene.sensors["feet_ground_contact"]
    first = sensor.compute_first_contact(env.step_dt)[:, :2]
    f = torch.nan_to_num(sensor.data.force, nan=0.0).reshape(env.num_envs, 2, -1).norm(dim=-1)
    bw = B.MASS_KG * G
    excess = ((f / bw) - B.IMPACT_BW).clamp(min=0.0)
    return (excess * first).sum(dim=1) * _in_skill(env)


def fall_event(env: Any, deg: float = 45.0) -> torch.Tensor:
    """rev b2 cost: 1 at the first tick of the episode the trunk tilts past ``deg`` — the fall is paid at
    once and in full (S0-B: the mat network still fell in 24–28 % of its episodes at 137 it while the
    per-tick ``fallen`` cost and the lost progress were discounted away; on the Mac 23 of 27 BAM falls
    started 0.5–2.3 s into the skill window itself). Needs the bookkeeping of this tick."""
    bp = bsk_plan(env)
    return ((bp.max_tilt > deg) & (bp.prev_max_tilt <= deg)).float()


def leg_torque(env: Any) -> torch.Tensor:
    """Cost: Σ over the ten leg servos of |τ| above TORQUE_FREE_NM (N·m)."""
    from duckmind.skilltrain import rl_contract as C

    tau = torch.nan_to_num(_robot(env).data.actuator_force, nan=0.0).abs()
    return (tau[:, list(C.LEG_IDX)] - B.TORQUE_FREE_NM).clamp(min=0.0).sum(dim=1) * _in_skill(env)


# ── rewards: the mat (bskmat) ────────────────────────────────────────────────────────────────


def _mat_state(env: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """(feet on the target level (N,), trunk progress 0..1 (N,))."""
    bp = bsk_plan(env)
    fz, fx = _feet_z(env), _feet_x(env)
    h = bp.mat_h.unsqueeze(1)
    contact = _feet_contact(env)
    past = fx > bp.edge_x.unsqueeze(1) + 0.02
    up_ok = contact & past & (fz > 0.6 * h)
    down_ok = contact & past & (fz < 0.4 * h)
    on = torch.where(bp.down.unsqueeze(1), down_ok, up_ok).float().sum(dim=1)
    tz = _robot(env).data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    rise = ((tz - bp.z0) / bp.mat_h.clamp(min=1e-3)).clamp(0.0, 1.0)
    drop = ((bp.z0 - tz) / bp.mat_h.clamp(min=1e-3)).clamp(0.0, 1.0)
    prog = torch.where(bp.down, drop, rise)
    return on, prog


def mat_progress(env: Any) -> torch.Tensor:
    """Potential-based: Φ = 0.6 · trunk rise (or drop) / h + 0.2 · feet on the target level, paid as
    (Φ_t − Φ_{t−1}) / dt in the skill window (signed: going back costs)."""
    bookkeeping(env)
    bp = bsk_plan(env)
    on, prog = _mat_state(env)
    phi = 0.6 * prog + 0.2 * on
    fresh = env.episode_length_buf <= 1
    prev = torch.where(fresh, phi, bp.phi)
    bp.phi = phi
    ok = (on >= 2) & (_tilt_deg(env) < 30.0)
    bp.success |= ok & (_phase(env) >= HOLD)
    return (phi - prev) / env.step_dt * _in_skill(env) * _is(env, "step_up", "step_down")


def mat_clearance(env: Any) -> torch.Tensor:
    """step_up: airborne feet lifted above the mat top (+12 mm margin) before both are on it."""
    bp = bsk_plan(env)
    on, _ = _mat_state(env)
    tgt = (bp.mat_h + 0.012).unsqueeze(1)
    lift = (_feet_z(env)).clamp(min=0.0)
    airborne = ~_feet_contact(env)
    score = ((lift / tgt).clamp(0.0, 1.0) * airborne).sum(dim=1)
    return score * _hold(env) * _is(env, "step_up") * (on < 2).float()


def mat_settle(env: Any, z_std: float = 0.012, v_std: float = 0.06) -> torch.Tensor:
    """One-shot at the skill's last tick: both feet on the target level, upright, standing height
    over it, still — ready for velstand."""
    sp, bp = RM.plan_of(env), bsk_plan(env)
    last = env.episode_length_buf == sp.skill_ticks
    on, _ = _mat_state(env)
    tz = _robot(env).data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    target = B.STAND_Z + torch.where(bp.down, torch.zeros_like(bp.mat_h), bp.mat_h)
    h = torch.exp(-(((tz - target) / z_std) ** 2))
    v = _robot(env).data.root_link_lin_vel_w.norm(dim=1)
    still = torch.exp(-((v / v_std) ** 2))
    up = RM._cos_tilt(env).clamp(min=0.0)
    return (on >= 2).float() * h * still * up * last * _is(env, "step_up", "step_down")


def after_speed(env: Any) -> torch.Tensor:
    """Cost: trunk speed above 0.05 m/s once both feet are on the target level (stop there)."""
    on, _ = _mat_state(env)
    v = torch.nan_to_num(_robot(env).data.root_link_lin_vel_w[:, :2], nan=0.0).norm(dim=1)
    return (
        (v - 0.05).clamp(min=0.0)
        * (on >= 2).float()
        * _in_skill(env)
        * _is(env, "step_up", "step_down")
    )


def heading_keep(env: Any, std: float = 0.25) -> torch.Tensor:
    sp = RM.plan_of(env)
    d = RM._wrap(_yaw(env) - sp.entry_yaw)
    return torch.exp(-((d / std) ** 2)) * _in_skill(env) * _is(env, "step_up", "step_down")


def mat_levels(env: Any, env_ids: Any) -> torch.Tensor:
    """Curriculum: a mat one row higher after a clean success, one lower after a fall or a failure.
    rev b2: a success followed by a fall no longer promotes (S0-B's curriculum reached 4 cm mean while
    24–28 % of the episodes fell — most of them after touching the top)."""
    terrain = env.scene.terrain
    if getattr(terrain, "terrain_origins", None) is None or env_ids is None:
        return torch.tensor(0.0)
    bp = bsk_plan(env)
    ids = env_ids if torch.is_tensor(env_ids) else torch.as_tensor(env_ids, device=env.device)
    clean = bp.success[ids] & ~bp.fell[ids]
    up = clean
    down = ~clean
    terrain.update_env_origins(ids, up, down)
    return terrain.terrain_levels.float().mean()


# ── rewards: the bow (bskbow) ────────────────────────────────────────────────────────────────


def bow_composite(env: Any, height_std: float = 0.012, orient_std: float = 0.12,
                  legs_std: float = 0.35, head_std: float = 0.25) -> torch.Tensor:  # fmt: skip
    """Multiplicative goal score for the front-down postures: trunk height · trunk orientation vs the
    blended target pitch (roll level) · legs · head (the bow's own ``rl_mdp.composite``, whose upright
    factor would fight a 25–30° pitch). The bow task's first reward term: it runs the bookkeeping."""
    bookkeeping(env)
    q_t, z_t, a = RM._blend(env)
    sp = RM.plan_of(env)
    q = RM._servo_q(env)
    from duckmind.skilltrain import rl_contract as C

    h = torch.exp(-(((RM._trunk_z(env) - z_t) / height_std) ** 2))
    roll, pitch = RM._roll_pitch(_robot(env).data.root_link_quat_w)
    target_pitch = -a * sp.pitch_full
    o = torch.exp(-((pitch - target_pitch) ** 2 + roll**2) / orient_std**2)
    legs = list(C.LEG_IDX)
    lg = torch.exp(-((q[:, legs] - q_t[:, legs]) ** 2).mean(dim=1) / legs_std**2)
    hd = torch.exp(
        -((q[:, list(C.HEAD_IDX)] - q_t[:, list(C.HEAD_IDX)]) ** 2).mean(dim=1) / head_std**2
    )
    return h * o * lg * hd * _in_skill(env) * _is(env, "bow", "lie")


# ── critic observation and metrics ───────────────────────────────────────────────────────────


def pitch_progress(env: Any) -> torch.Tensor:
    """rev b1: nose-down pitch / the blended target pitch, clipped to [0, 1] — a linear gradient from standing
    (the b0 Gaussians σ 5° / 0.12 rad paid ~0 at a 25–33° error: the b0 bow only dropped 11 mm, pitch ~0°)."""
    sp = RM.plan_of(env)
    _, _, a = RM._blend(env)
    _, pitch = RM._roll_pitch(_robot(env).data.root_link_quat_w)  # + = nose down
    target = (-sp.pitch_full * a).clamp(min=1e-3)
    # rev b2: the hold only. In the unwind the blended target shrinks toward 0 with alpha, so b1 paid the
    # full score for STAYING pitched while the duck should rise (it fought the hand-back bonus: 0.32 → 0.07).
    return (
        (pitch / target).clamp(0.0, 1.0) * (a > 0.05).float() * _hold(env) * _is(env, "bow", "lie")
    )


def bow_gate_pose(env: Any, drop_m: float = 0.050) -> torch.Tensor:
    """Dense bow-only signal with margin above the gate's 40 mm / 15 degree pose.

    Paid throughout HOLD so touching the target briefly is less valuable than holding it.
    The entry is sampled once after reset; never re-zero at the lowest pose or in unwind.
    """
    cache = env.__dict__.setdefault("_bsk_site_cache", {})
    if "beak" not in cache:
        ids, _ = _robot(env).find_sites(["mouth_tip"], preserve_order=True)
        if len(ids) != 1:
            raise RuntimeError("bow gate reward requires exactly one mouth_tip site")
        cache["beak"] = ids[0]
    z = _robot(env).data.site_pos_w[:, cache["beak"], 2]
    bp = bsk_plan(env)
    bp.beak_entry_z = torch.where(torch.isnan(bp.beak_entry_z), z, bp.beak_entry_z)
    drop = ((bp.beak_entry_z - z) / drop_m).clamp(0.0, 1.0)
    _, pitch = RM._roll_pitch(_robot(env).data.root_link_quat_w)
    orient = (pitch / math.radians(20.0)).clamp(0.0, 1.0)
    return drop * orient * _hold(env) * _is(env, "bow")


def bsk_state_obs(env: Any) -> torch.Tensor:
    """Phase (4), seconds into the hold, hold left, behaviour one-hot (7), turn sign, displacement
    from the entry (2, heading frame), yaw change, mat height and edge distance (×10): 20 dims."""
    term, sp, bp = RM.skill_term(env), RM.plan_of(env), bsk_plan(env)
    dt = env.step_dt
    tick = term.tick.float()
    phase = torch.nn.functional.one_hot(term.phase.clamp(0, 3), 4).float()
    hold_left = ((sp.hold_ticks.float() - tick) * dt).clamp(min=0.0)
    beh = torch.nn.functional.one_hot(bp.behaviour, len(B.BEHAVIOURS)).float()
    d = _robot(env).data.root_link_pos_w[:, :2] - sp.entry_xy
    local = RM._rot_z(torch.cat([d, torch.zeros_like(d[:, :1])], dim=1), -sp.entry_yaw)[:, :2]
    dyaw = RM._wrap(_yaw(env) - sp.entry_yaw)
    mat = _is(env, "step_up", "step_down").float()
    edge = (bp.edge_x - _robot(env).data.root_link_pos_w[:, 0]) * 10.0 * mat
    return torch.cat([phase, (tick * dt).unsqueeze(1), hold_left.unsqueeze(1), beh, bp.sign.unsqueeze(1),
                      local * 10.0, dyaw.unsqueeze(1), (bp.mat_h * 10.0).unsqueeze(1), edge.unsqueeze(1)], dim=1)  # fmt: skip


def bsk_done(env: Any) -> torch.Tensor:
    return env.episode_length_buf >= RM.plan_of(env).end_ticks


def m_turn_rate_dps(env: Any) -> torch.Tensor:
    """Signed yaw rate in the turn direction (deg/s), turn holds only."""
    bp = bsk_plan(env)
    wz = torch.nan_to_num(_robot(env).data.root_link_ang_vel_w[:, 2], nan=0.0)
    return torch.rad2deg(wz * bp.sign) * _hold(env) * _is(env, "turn")


def m_travel_cm(env: Any) -> torch.Tensor:
    sp = RM.plan_of(env)
    d = (_robot(env).data.root_link_pos_w[:, :2] - sp.entry_xy).norm(dim=1)
    return 100.0 * d * _is(env, "turn", "march")


def m_steps(env: Any) -> torch.Tensor:
    return bsk_plan(env).steps * _is(env, "turn", "march")


def m_lift_mm(env: Any) -> torch.Tensor:
    return 1000.0 * _lift(env).max(dim=1).values * _hold(env) * _is(env, "turn", "march")


def m_mat_success(env: Any) -> torch.Tensor:
    return bsk_plan(env).success.float() * _is(env, "step_up", "step_down")


def m_mat_clean(env: Any) -> torch.Tensor:
    """rev b2: on the target level with the tilt < 30° AND no fall in the episode (the curriculum's rule)."""
    bp = bsk_plan(env)
    return (bp.success & ~bp.fell).float() * _is(env, "step_up", "step_down")


def m_mat_up_success(env: Any) -> torch.Tensor:
    return bsk_plan(env).success.float() * _is(env, "step_up")


def m_mat_mm(env: Any) -> torch.Tensor:
    return 1000.0 * bsk_plan(env).mat_h * _is(env, "step_up", "step_down")


def m_fell(env: Any) -> torch.Tensor:
    return bsk_plan(env).fell.float()


def m_bow_pitch_deg(env: Any) -> torch.Tensor:
    _, pitch = RM._roll_pitch(_robot(env).data.root_link_quat_w)
    return torch.rad2deg(pitch) * _hold(env) * _is(env, "bow", "lie")


def m_bow_drop_mm(env: Any) -> torch.Tensor:
    return 1000.0 * (B.STAND_Z - RM._trunk_z(env)) * _hold(env) * _is(env, "bow", "lie")


def m_share(name: str):
    def f(env: Any) -> torch.Tensor:
        return _is(env, name).float()

    f.__name__ = f"m_share_{name}"
    return f


__all__ = [
    "bow_composite",
    "bsk_done",
    "bsk_plan",
    "bsk_state_obs",
    "mat_levels",
    "mat_progress",
    "reset_bsk_episode",
]
