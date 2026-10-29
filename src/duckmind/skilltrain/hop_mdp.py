"""MDP pieces of the forward hop-over task (hop:feasibility, 2026-09-28).

One episode = one robotd skill window, as robotd runs it (Stage 1's machinery, ``rl_mdp``'s
:class:`~duckmind.skilltrain.rl_mdp.SkillCommand` for the phases hold → unwind → [hand-back]):

1. **Entry** (:func:`reset_hop_episode`): the tile's strip (height = the curriculum row, edge =
   the column); the duck from the v5 state library (walking 0.35–0.55, standing, coasting;
   ``rl_library``) turned to face the strip ±12° with its trunk centre 6–18 cm before the near face;
   robotd's filter state and the walk network's last raw action carried over (``rl_action``).
2. **Hold**: the entry's constant command (0.30 + 0.05·class, 0, 0) for 3.0 s (±jitter): the
   network walks up, finds the edge by feel and steps (or hops) over; the potential pays forward
   progress up to 10 cm past the far face.
3. **Unwind**: 0.5 s of the zero command: settle for velstand (the hand-back bonus at the last tick).

Sign convention (Stage 1's): reward functions ≥ 0 with positive weights, costs ≥ 0 with negative
weights; the progress potential is signed by design (moving back costs) and exempt.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from mjlab_microduck.tasks import mdp as microduck_mdp

from duckmind.skilltrain import hop_contract as H
from duckmind.skilltrain import rl_contract as C
from duckmind.skilltrain import rl_mdp as RM
from duckmind.skilltrain.hop_terrain import height_of

HOLD, UNWIND, HANDBACK = RM.HOLD, RM.UNWIND, RM.HANDBACK
PAST_FOOT_SITE_M = 0.021  # the foot site is the ankle: its heel is 2.0 cm behind, the toe 3.4 ahead
TOE_AHEAD_M = 0.034
KIND_IDS = {k: C.ENTRY_KINDS.index(k) for k in ("stand", "walk", "coast")}


class HopPlan:
    """Per-env strip and progress state."""

    def __init__(self, n: int, device: str):
        z = lambda: torch.zeros(n, device=device)  # noqa: E731
        zb = lambda: torch.zeros(n, dtype=torch.bool, device=device)  # noqa: E731
        self.height, self.depth, self.edge = (
            z(),
            z(),
            torch.zeros(n, dtype=torch.long, device=device),
        )
        self.face_x, self.mid_y = z(), z()
        self.start_d, self.cmd_height = z(), z()
        self.kind = torch.zeros(n, dtype=torch.long, device=device)
        self.phi = z()
        self.crossed, self.fell, self.first_cross = zb(), zb(), zb()
        self.first_fall = zb()
        self.cross_tick = torch.full((n,), -1, dtype=torch.long, device=device)
        self.down_ticks = torch.zeros(n, dtype=torch.long, device=device)
        self.max_tilt = z()
        self.stubs = z()
        self.entry_yaw = z()
        self.has_outcome = zb()


def hop_plan(env: Any) -> HopPlan:
    plan = getattr(env, "hop_plan", None)
    if plan is None:
        plan = HopPlan(env.num_envs, env.device)
        env.hop_plan = plan
    return plan


def _robot(env: Any):
    return env.scene["robot"]


def _sites(env: Any) -> list[int]:
    cache = env.__dict__.setdefault("_hop_site_cache", {})
    if "feet" not in cache:
        ids, _ = _robot(env).find_sites(["left_foot", "right_foot"], preserve_order=True)
        cache["feet"] = ids
    return cache["feet"]


def rel_x(env: Any) -> torch.Tensor:
    """Trunk centre's distance past the near face (m, + = over / beyond)."""
    return _robot(env).data.root_link_pos_w[:, 0] - hop_plan(env).face_x


def feet_rel(env: Any) -> torch.Tensor:
    """(N, 2, 3): the foot sites relative to the near face (x) and the floor (z)."""
    p = _robot(env).data.site_pos_w[:, _sites(env)]
    plan = hop_plan(env)
    out = p.clone()
    out[:, :, 0] = p[:, :, 0] - plan.face_x.unsqueeze(1)
    out[:, :, 2] = p[:, :, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    return out


def tilt_deg(env: Any) -> torch.Tensor:
    q = _robot(env).data.root_link_quat_w
    c = torch.nan_to_num(1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2), nan=1.0)
    return torch.rad2deg(torch.acos(c.clamp(-1.0, 1.0)))


def crossed_now(env: Any) -> torch.Tensor:
    plan = hop_plan(env)
    f = feet_rel(env)
    far = plan.depth.unsqueeze(1)
    feet_past = (f[:, :, 0] > far + PAST_FOOT_SITE_M).all(dim=1)
    return feet_past & (rel_x(env) > plan.depth + H.PAST_TRUNK_M) & (tilt_deg(env) < 45.0)


def _in_skill(env: Any) -> torch.Tensor:
    ph = RM.skill_term(env).phase
    return (ph == HOLD) | (ph == UNWIND)


# ── the episode ─────────────────────────────────────────────────────────────────────────────


def reset_hop_episode(
    env: Any,
    env_ids: Any,
    mix_stage: str = "full",
    library_path: str | None = None,
    eval_cells: list[tuple[int, int, str]] | None = None,
    handback_s: float = 0.0,
    nominal_windows: bool = False,
) -> None:
    """Choose each env's entry, windows and command, and place the duck before its strip.

    ``eval_cells`` (evaluation): (terrain row, edge column, entry kind) round-robin over envs."""
    from duckmind.skilltrain import rl_library

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    ids = env_ids.to(env.device, dtype=torch.long)
    n = len(ids)
    if n == 0:
        return
    dev = env.device
    plan, sp, entry = hop_plan(env), RM.plan_of(env), RM.entry_of(env)
    terrain = env.scene.terrain
    rows = int(terrain.terrain_origins.shape[0])
    if eval_cells:
        cell = ids % len(eval_cells)
        row = torch.tensor([c[0] for c in eval_cells], device=dev)[cell]
        col = torch.tensor([c[1] for c in eval_cells], device=dev)[cell]
        kind = torch.tensor([KIND_IDS[c[2]] for c in eval_cells], device=dev)[cell]
        terrain.terrain_levels[ids] = row
        terrain.terrain_types[ids] = col
        terrain.env_origins[ids] = terrain.terrain_origins[row, col]
    else:
        row = terrain.terrain_levels[ids]
        col = terrain.terrain_types[ids]
        mix = H.ENTRY_MIX_EARLY if mix_stage == "early" else H.ENTRY_MIX
        names = list(mix)
        probs = torch.tensor([mix[k] for k in names], device=dev)
        pick = torch.multinomial(probs, n, replacement=True)
        kind = torch.tensor([KIND_IDS[k] for k in names], device=dev)[pick]
    origin = env.scene.env_origins[ids]
    height = height_of(row.float(), rows)
    depth = torch.full((n,), H.DEPTH_M, device=dev)
    # the brain's class: its camera estimate (± 5 mm) rounded up to whole cm, 1–4
    if eval_cells or nominal_windows:
        est = height
    else:
        est = height + (torch.rand(n, device=dev) * 2 - 1) * H.HEIGHT_EST_NOISE
    cls = torch.ceil((est * 100.0).clamp(min=0.5) - 1e-6).clamp(1, 4)
    command = torch.zeros(n, 3, device=dev)
    command[:, 0] = (
        H.CMD_BASE + H.CMD_STEP * cls
    )  # a walk-like command: velstand warm-starts from it
    hold_s = torch.full((n,), H.HOLD_S, device=dev)
    if not (eval_cells or nominal_windows):
        jit = H.HOLD_JITTER[0] + (H.HOLD_JITTER[1] - H.HOLD_JITTER[0]) * torch.rand(n, device=dev)
        hold_s = hold_s * jit
    dt = env.step_dt
    sp.hold_ticks[ids] = torch.round(hold_s / dt).long()
    sp.unwind_ticks[ids] = int(round(H.UNWIND_S / dt))
    sp.handback_ticks[ids] = int(round(handback_s / dt))
    sp.command[ids] = command
    sp.unwind_cmd[ids] = 0.0
    sp.handback_cmd[ids] = 0.0
    sp.goal_hold[ids] = 0.0
    sp.alpha0[ids] = 0.0
    sp.seated[ids] = False
    sp.kind[ids] = kind
    sp.level[ids] = cls / 4.0
    sp.q_base[ids] = torch.tensor(C.HOME, device=dev)
    sp.q_full[ids] = torch.tensor(C.HOME, device=dev)
    sp.z_base[ids] = C.STAND_Z
    sp.z_full[ids] = C.STAND_Z
    # the entry state
    lib = rl_library.cached(env, library_path)
    state = rl_library.sample_states(lib, kind, dev) if lib is not None else None
    synth = rl_library.synthetic_states(kind, dev)
    if state is None:
        state = synth
    else:
        for key in ("root_z", "root_quat", "lin_vel", "ang_vel", "q", "qd", "last_raw",
                    "prev_target", "has_prev"):  # fmt: skip
            a, b = state[key], synth[key]
            mask = state["missing"].view(-1, *([1] * (a.dim() - 1)))
            state[key] = torch.where(mask, b, a)
    aim = torch.deg2rad((torch.rand(n, device=dev) * 2 - 1) * H.AIM_DEG)
    start_d = H.START_D[0] + (H.START_D[1] - H.START_D[0]) * torch.rand(n, device=dev)
    if eval_cells:
        # evaluation: a fixed grid of distances (every cell sees the same spread)
        start_d = (
            H.START_D[0]
            + (H.START_D[1] - H.START_D[0]) * ((ids // len(eval_cells)) % 7).float() / 6.0
        )
        aim = torch.deg2rad(
            ((ids // (7 * len(eval_cells))) % 5).float() / 4.0 * 2 * H.AIM_DEG - H.AIM_DEG
        )
    # the library keeps each state's recorded heading: turn it (and its world-frame velocities)
    # so the duck faces the strip's normal (+x) plus the aim
    turn = aim - RM._yaw(state["root_quat"])
    quat = RM._quat_mul(RM._quat_yaw(turn), state["root_quat"])
    quat = quat / quat.norm(dim=-1, keepdim=True)
    lateral = (torch.rand(n, device=dev) * 2 - 1) * 0.3
    x = origin[:, 0] - start_d
    y = origin[:, 1] + lateral
    z = state["root_z"] + origin[:, 2]
    robot = _robot(env)
    pose = torch.stack([x, y, z], dim=1)
    robot.write_root_link_pose_to_sim(torch.cat([pose, quat], dim=1), env_ids=ids)
    vel = torch.cat([RM._rot_z(state["lin_vel"], turn), RM._rot_z(state["ang_vel"], turn)], dim=1)
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
    sp.entry_xy[ids] = pose[:, :2]
    sp.entry_yaw[ids] = aim
    sp.entry_z[ids] = state["root_z"]
    # the strip
    plan.height[ids], plan.depth[ids], plan.edge[ids] = height, depth, col
    plan.face_x[ids], plan.mid_y[ids] = origin[:, 0], origin[:, 1]
    plan.start_d[ids], plan.cmd_height[ids] = start_d, cls / 100.0
    plan.kind[ids] = kind
    plan.phi[ids] = (-start_d).clamp(max=float(H.DEPTH_M + H.GOAL_TRUNK_M))
    plan.crossed[ids] = False
    plan.first_cross[ids] = False
    plan.fell[ids] = False
    plan.first_fall[ids] = False
    plan.cross_tick[ids] = -1
    plan.down_ticks[ids] = 0
    plan.max_tilt[ids] = 0.0
    plan.stubs[ids] = 0.0
    plan.entry_yaw[ids] = aim
    plan.has_outcome[ids] = True


def strip_levels(
    env: Any, env_ids: Any, up_rate: float = 1.0, max_level: int | None = None
) -> dict[str, torch.Tensor]:
    """Curriculum (runs before the reset): a duck that crossed without falling moves one row up
    (higher strip), a duck that fell or did not cross moves one row down."""
    terrain = env.scene.terrain
    plan = hop_plan(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    ids = env_ids.to(env.device, dtype=torch.long)
    known = plan.has_outcome[ids]
    good = plan.crossed[ids] & ~plan.fell[ids] & known
    bad = (plan.fell[ids] | ~plan.crossed[ids]) & known
    if up_rate < 1.0:
        good = good & (torch.rand(len(ids), device=env.device) < up_rate)
    terrain.update_env_origins(ids, good, bad)
    if max_level is not None:
        # Keep both the row indices and actual origins aligned (not merely the logged height).
        terrain.terrain_levels[ids] = terrain.terrain_levels[ids].clamp(max=max_level)
        terrain.env_origins[ids] = terrain.terrain_origins[
            terrain.terrain_levels[ids], terrain.terrain_types[ids]
        ]
    levels = terrain.terrain_levels.float()
    return {"mean": levels.mean(), "max": levels.max()}


# ── per-step bookkeeping + rewards (≥ 0, positive weight) ───────────────────────────────────


def progress(env: Any, max_mps: float = 2.0) -> torch.Tensor:
    """Potential Φ = clamp(trunk past the near face, ·, depth + 10 cm): (ΔΦ)/dt during the hold
    (signed: stepping back costs), paid up to ``max_mps`` (rev h2: 0.35 — a dash over the strip
    earns no more than a brisk walk). Also updates the crossing / fall bookkeeping (called once
    per env step, first in the reward list)."""
    plan = hop_plan(env)
    x = rel_x(env)
    phi = x.clamp(max=float(H.DEPTH_M + H.GOAL_TRUNK_M))
    phi = torch.minimum(phi, plan.depth + H.GOAL_TRUNK_M)
    r = (phi - plan.phi) / env.step_dt
    plan.phi = phi
    # bookkeeping: crossing, falls (robotd's rule), max tilt
    tilt = tilt_deg(env)
    now = crossed_now(env)
    plan.first_cross = now & ~plan.crossed
    plan.cross_tick = torch.where(plan.first_cross, env.episode_length_buf, plan.cross_tick)
    plan.crossed = plan.crossed | now
    plan.down_ticks = torch.where(
        tilt > H.FALL_TILT_DEG, plan.down_ticks + 1, torch.zeros_like(plan.down_ticks)
    )
    down = plan.down_ticks >= int(round(H.FALL_DEBOUNCE_S / env.step_dt))
    plan.first_fall = down & ~plan.fell
    plan.fell = plan.fell | down
    plan.max_tilt = torch.maximum(plan.max_tilt, tilt)
    gate = (RM.skill_term(env).phase == HOLD).float()
    return torch.nan_to_num(r, nan=0.0).clamp(-2.0, float(max_mps)) * gate


def cross_bonus(env: Any) -> torch.Tensor:
    """One-shot when both soles and the trunk are first past the far face, upright."""
    return hop_plan(env).first_cross.float()


def fall_event(env: Any) -> torch.Tensor:
    """One cost per debounced fall, including the post-skill hand-back phase."""
    return hop_plan(env).first_fall.float()


def upright(env: Any, std_deg: float = 20.0) -> torch.Tensor:
    return torch.exp(-((tilt_deg(env) / std_deg) ** 2)) * _in_skill(env)


def heading(env: Any, free_deg: float = 10.0, std_deg: float = 15.0) -> torch.Tensor:
    """Keep the entry heading (cross square, end facing on)."""
    sp = RM.plan_of(env)
    yaw = RM._yaw(_robot(env).data.root_link_quat_w)
    err = torch.rad2deg(RM._wrap(yaw - sp.entry_yaw).abs())
    return torch.exp(-((((err - free_deg).clamp(min=0.0)) / std_deg) ** 2)) * _in_skill(env)


def swing_clearance(env: Any, cap_m: float = 0.015) -> torch.Tensor:
    """A foot off the floor above the strip's span (toe ahead of the near face, heel short of
    the far face) is paid its height over the strip top, up to ``cap_m``: lift, don't stub."""
    plan = hop_plan(env)
    f = feet_rel(env)
    toe = f[:, :, 0] + TOE_AHEAD_M
    heel = f[:, :, 0] - 0.020
    over = (toe > -0.01) & (heel < plan.depth.unsqueeze(1) + 0.01)
    contact = env.scene.sensors["feet_ground_contact"].data.found[:, :2] > 0
    lift = (f[:, :, 2] - plan.height.unsqueeze(1)).clamp(0.0, cap_m) / cap_m
    return ((lift * over * ~contact).sum(dim=1)) * (RM.skill_term(env).phase == HOLD)


def settle_bonus(env: Any, v_std: float = 0.08) -> torch.Tensor:
    """One-shot at the skill's last tick: over the strip, upright and slow — ready for velstand."""
    sp, plan = RM.plan_of(env), hop_plan(env)
    last = env.episode_length_buf == sp.skill_ticks
    v = _robot(env).data.root_link_lin_vel_w[:, :2].norm(dim=1)
    up = torch.exp(-((tilt_deg(env) / 10.0) ** 2))
    z = RM._trunk_z(env)
    h = torch.exp(-(((z - C.STAND_Z) / 0.012) ** 2))
    return last * plan.crossed * ~plan.fell * up * h * torch.exp(-((v / v_std) ** 2))


# ── costs (≥ 0, negative weight) ────────────────────────────────────────────────────────────


def toe_stub(env: Any, ratio: float = 0.5) -> torch.Tensor:
    """Horizontal floor/strip force on a sole beyond ``ratio`` × its vertical force (N): a toe
    pushing into the strip's face (world-frame net force, both feet), near the strip."""
    plan = hop_plan(env)
    force = env.scene.sensors["feet_ground_contact"].data.force
    if force is None:
        return torch.zeros(env.num_envs, device=env.device)
    force = torch.nan_to_num(force[:, :2], nan=0.0)
    horiz = force[:, :, :2].norm(dim=2)
    vert = force[:, :, 2].abs()
    near = (rel_x(env) > -0.12) & (rel_x(env) < plan.depth + 0.10)
    stub = (horiz - ratio * vert).clamp(min=0.0).sum(dim=1) * near
    plan.stubs = plan.stubs + (stub > 1.0).float()
    return stub * _in_skill(env)


def foot_impact(env: Any, free_bw: float = 2.5) -> torch.Tensor:
    """Vertical floor/strip force on the soles above ``free_bw`` × body weight (N), both feet — the
    calibration's leap-step landed at 3.6–5.7 × body weight p50 (8.6 × peak): the servos and the
    gearboxes take that (rev h2)."""
    force = env.scene.sensors["feet_ground_contact"].data.force
    if force is None:
        return torch.zeros(env.num_envs, device=env.device)
    cache = env.__dict__.setdefault("_hop_weight", {})
    if "w" not in cache:
        robot = _robot(env)
        cache["w"] = float(robot.data.default_mass.sum(dim=-1)[0].item()) * 9.81 if hasattr(
            robot.data, "default_mass") else 0.737 * 9.81  # fmt: skip
    fz = torch.nan_to_num(force[:, :2, 2], nan=0.0).abs()
    return (fz - free_bw * cache["w"]).clamp(min=0.0).sum(dim=1) * _in_skill(env)


def fallen(env: Any, tilt: float = 45.0) -> torch.Tensor:
    return (tilt_deg(env) > tilt).float()


def leg_torque(env: Any, free_nm: float = 0.30) -> torch.Tensor:
    """Σ leg |τ| above ``free_nm`` (strain: the gate is p95 ≤ 0.35 N·m)."""
    force = torch.nan_to_num(_robot(env).data.actuator_force, nan=0.0).abs()
    return (force[:, list(C.LEG_IDX)] - free_nm).clamp(min=0.0).sum(dim=1) * _in_skill(env)


def after_cross_speed(env: Any, free_mps: float = 0.10) -> torch.Tensor:
    """After crossing: trunk speed above ``free_mps`` (stop past the strip, don't run on)."""
    plan = hop_plan(env)
    v = _robot(env).data.root_link_lin_vel_w[:, :2].norm(dim=1)
    return (v - free_mps).clamp(min=0.0) * plan.crossed * _in_skill(env)


def housing_touch(env: Any, threshold: float = 2.0) -> torch.Tensor:
    """1 while any servo housing presses on the floor or the strip with more than ``threshold`` N
    (upright, in the skill) — the gate's own event (a housing contact > 2 N), where ``servo_impact``
    pays only the force above it (hop:train; weight 0 unless switched on mid-run)."""
    sensor = env.scene.sensors["servo_ground_contact"]
    force = sensor.data.force
    if force is None:
        return torch.zeros(env.num_envs, device=env.device)
    mag = (
        torch.nan_to_num(force, nan=0.0).reshape(env.num_envs, -1, 3).norm(dim=2).max(dim=1).values
    )
    return (mag > threshold).float() * (tilt_deg(env) < 45.0) * _in_skill(env)


def dash_speed(env: Any, free_mps: float = 0.45) -> torch.Tensor:
    """Trunk speed above ``free_mps`` during the hold (the calibration dashed at 0.7–0.8 m/s; hop:train;
    weight 0 unless switched on mid-run)."""
    v = _robot(env).data.root_link_lin_vel_w[:, :2].norm(dim=1)
    return (v - free_mps).clamp(min=0.0) * (RM.skill_term(env).phase == HOLD)


# ── metrics (per-step averages over an episode; ``reduce='last'`` where noted) ──────────────


def m_crossed(env: Any) -> torch.Tensor:
    return hop_plan(env).crossed.float()


def m_fell(env: Any) -> torch.Tensor:
    return hop_plan(env).fell.float()


def m_height_mm(env: Any) -> torch.Tensor:
    return hop_plan(env).height * 1000.0


def m_progress_mm(env: Any) -> torch.Tensor:
    return rel_x(env) * 1000.0


def m_stubs(env: Any) -> torch.Tensor:
    return hop_plan(env).stubs


def m_max_tilt(env: Any) -> torch.Tensor:
    return hop_plan(env).max_tilt


def hop_state_obs(env: Any) -> torch.Tensor:
    """The critic's privileged extras (14): phase one-hot (4), seconds into the phase, strip height
    and depth (×10), edge, trunk past the near face (×5), both feet relative (x ×5, z ×20), crossed."""
    term, plan = RM.skill_term(env), hop_plan(env)
    phase = torch.nn.functional.one_hot(term.phase.clamp(0, 3), 4).float()
    f = feet_rel(env)
    return torch.cat(
        [
            phase,
            RM._t_phase(env).unsqueeze(1),
            (plan.height * 10.0).unsqueeze(1),
            (plan.depth * 10.0).unsqueeze(1),
            plan.edge.float().unsqueeze(1),
            (rel_x(env) * 5.0).unsqueeze(1),
            f[:, :, 0] * 5.0,
            f[:, :, 2] * 20.0,
            plan.crossed.float().unsqueeze(1),
        ],
        dim=1,
    )


COST_FUNCS = (
    toe_stub,
    fallen,
    leg_torque,
    after_cross_speed,
    foot_impact,
    housing_touch,
    dash_speed,
)
SIGNED_TERMS = ("progress",)


def radians(deg: float) -> float:
    return math.radians(deg)


__all__ = [
    "COST_FUNCS",
    "HopPlan",
    "SIGNED_TERMS",
    "cross_bonus",
    "crossed_now",
    "hop_plan",
    "hop_state_obs",
    "progress",
    "reset_hop_episode",
    "settle_bonus",
    "strip_levels",
    "swing_clearance",
    "toe_stub",
]
