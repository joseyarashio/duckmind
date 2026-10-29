"""MDP pieces of the two Stage-1 skill tasks (training plan §3.1 R1 startle, §3.2 R2 comfort).

An episode is one robotd skill window, as robotd runs it:

1. **Entry** (:func:`reset_skill_episode`): a state the previous network left the duck in —
   v5 velstand standing, walking 0.35–0.55 m/s with arcs, coasting after a stop, head turned,
   just risen, or sitstand seated — sampled from a state library recorded on v5 under domain
   randomisation (``rl_library``), rotated to a random heading; plus synthetic crouches (a new
   startle while the last one is still unwinding). The entry also carries robotd's filter state
   and the previous network's last raw action (``rl_action``).
2. **Hold** (:class:`SkillCommand`): the table entry's constant ``command`` for its
   ``duration`` (optionally jittered, or cut short by ``startle_release``); head and body slots
   zero (zero-padded observations).
3. **Unwind**: the entry's ``unwind`` command for ``unwind_s`` (the release's zero command for
   0.8 s when it cut the hold), then the episode ends (:func:`skill_done`, a true terminal: what
   follows is the walk network's). The evaluator adds a **hand-back** phase that runs v5.

Rewards follow ``microduck_rl/AGENTS.md``: single fixed targets with a slewed blend (the command
term's ``alpha``: arriving early pays nothing), a multiplicative composite at the goal,
potential-based progress terms (the back-steps), positive terms never gated on a bad state,
smoothness taxes only after discovery (curricula in ``rl_tasks``). **Sign convention (one for
every term here):** reward functions return ≥ 0 and get a positive weight; cost functions return
≥ 0 and get a negative weight — so every ``Episode_Reward/<cost>`` must log ≤ 0 (the trainer
aborts otherwise). Upstream's self-negating helpers are wrapped (:func:`vertical_accel_cost`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from mjlab.managers import CommandTermCfg
from mjlab.managers.command_manager import CommandTerm
from mjlab_microduck.tasks import mdp as microduck_mdp

from duckmind.skilltrain import rl_contract as C

HOLD, UNWIND, HANDBACK, DONE = 0, 1, 2, 3
PHASES = ("hold", "unwind", "handback", "done")
BEHAVIOUR_ID = {name: i for i, name in enumerate(C.BEHAVIOURS)}
KIND_ID = {name: i for i, name in enumerate(C.ENTRY_KINDS)}


# ── per-env state the managers share ────────────────────────────────────────────────────────


class SkillEntry:
    """robotd's carried-over state at the skill's first tick (read by the action term's reset)."""

    def __init__(self, n: int, device: str):
        self.prev_target = torch.tensor(C.HOME, device=device).repeat(n, 1)
        self.has_prev = torch.zeros(n, dtype=torch.bool, device=device)
        self.last_raw = torch.zeros(n, C.ACTION_LEN, device=device)


class SkillPlan:
    """Everything one episode's variant, windows and targets are (per env)."""

    def __init__(self, n: int, device: str):
        z = lambda *shape: torch.zeros(*shape, device=device)  # noqa: E731
        zl = lambda: torch.zeros(n, dtype=torch.long, device=device)  # noqa: E731
        zb = lambda: torch.zeros(n, dtype=torch.bool, device=device)  # noqa: E731
        home = torch.tensor(C.HOME, device=device).repeat(n, 1)
        self.variant, self.behaviour, self.kind = zl(), zl(), zl()
        self.command, self.unwind_cmd, self.handback_cmd = z(n, 3), z(n, 3), z(n, 3)
        self.hold_ticks, self.unwind_ticks, self.handback_ticks = zl(), zl(), zl()
        self.level = z(n)
        self.seated, self.hop, self.shake, self.stretch, self.cut = zb(), zb(), zb(), zb(), zb()
        self.q_base, self.q_full = home.clone(), home.clone()
        self.z_base = torch.full((n,), C.STAND_Z, device=device)
        self.z_full = self.z_base.clone()
        self.pitch_full = z(n)  # rad, + = nose up
        self.cam_full = z(n)  # deg
        self.alpha0, self.goal_hold = z(n), z(n)
        self.ramp_up = torch.full((n,), 0.2, device=device)
        self.ramp_down = torch.full((n,), C.UNWIND_RAMP_S, device=device)
        self.bands = z(n, 8)  # shake: roll lo/hi deg, roll hz lo/hi, yaw lo/hi rad, yaw hz lo/hi
        self.head_w = torch.ones(n, 4, device=device)  # per head joint weight in the pose terms
        self.entry_xy, self.entry_yaw, self.entry_z = z(n, 2), z(n), z(n)

    @property
    def skill_ticks(self) -> torch.Tensor:
        return self.hold_ticks + self.unwind_ticks

    @property
    def end_ticks(self) -> torch.Tensor:
        return self.hold_ticks + self.unwind_ticks + self.handback_ticks


def plan_of(env: Any) -> SkillPlan:
    plan = getattr(env, "skill_plan", None)
    if plan is None:
        plan = SkillPlan(env.num_envs, env.device)
        env.skill_plan = plan
    return plan


def entry_of(env: Any) -> SkillEntry:
    entry = getattr(env, "skill_entry", None)
    if entry is None:
        entry = SkillEntry(env.num_envs, env.device)
        env.skill_entry = entry
    return entry


def skill_term(env: Any) -> SkillCommand:
    return env.command_manager.get_term("twist")  # type: ignore[return-value]


# ── small maths ─────────────────────────────────────────────────────────────────────────────


def _yaw(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _roll_pitch(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w, x, y, z = q.unbind(-1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin((2 * (w * y - z * x)).clamp(-1.0, 1.0))
    return roll, pitch


def _quat_yaw(yaw: torch.Tensor) -> torch.Tensor:
    h = 0.5 * yaw
    zero = torch.zeros_like(h)
    return torch.stack([torch.cos(h), zero, zero, torch.sin(h)], dim=-1)


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def _rot_z(v: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(yaw), torch.sin(yaw)
    x, y, z = v.unbind(-1)
    return torch.stack([c * x - s * y, s * x + c * y, z], dim=-1)


def _wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def _smooth_band(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, soft: torch.Tensor):
    """1 inside [lo, hi]; (x/lo)² below (a gradient from zero: discovery); Gaussian above."""
    below = (x / lo.clamp(min=1e-6)).clamp(0.0, 1.0) ** 2
    above = torch.exp(-((((x - hi).clamp(min=0.0)) / soft.clamp(min=1e-6)) ** 2))
    return torch.where(x < lo, below, above)


def _freq_band(f: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, sigma: float):
    dist = torch.where(f < lo, lo - f, torch.where(f > hi, f - hi, torch.zeros_like(f)))
    return torch.exp(-((dist / max(sigma, 1e-6)) ** 2))


def _robot(env: Any):
    return env.scene["robot"]


def _servo_q(env: Any) -> torch.Tensor:
    robot = _robot(env)
    return robot.data.joint_pos[:, microduck_mdp._servo_joint_ids(env, robot)]


def _trunk_z(env: Any) -> torch.Tensor:
    robot = _robot(env)
    return torch.nan_to_num(robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2], nan=0.0)


def _cos_tilt(env: Any) -> torch.Tensor:
    q = _robot(env).data.root_link_quat_w
    return torch.nan_to_num(1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2), nan=1.0)


def camera_pitch_deg(env: Any) -> torch.Tensor:
    """The head camera's optical-axis pitch, world, + = up (the v3 sweep's definition)."""
    robot = _robot(env)
    cache = env.__dict__.setdefault("_skill_site_cache", {})
    if "cam" not in cache:
        ids, _ = robot.find_sites([C.CAMERA_SITE], preserve_order=True)
        cache["cam"] = ids[0]
    q = robot.data.site_quat_w[:, cache["cam"]]
    w, x, y, z = q.unbind(-1)
    return torch.rad2deg(torch.asin((2.0 * (x * z - w * y)).clamp(-1.0, 1.0)))


def _blend(env: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(joint target (N,14), trunk-height target (N,), alpha (N,)) at this tick."""
    plan, term = plan_of(env), skill_term(env)
    a = term.alpha
    q_t = plan.q_base + a.unsqueeze(1) * (plan.q_full - plan.q_base)
    z_t = plan.z_base + a * (plan.z_full - plan.z_base)
    return q_t, z_t, a


def _phase(env: Any) -> torch.Tensor:
    return skill_term(env).phase


def _t_phase(env: Any) -> torch.Tensor:
    """Seconds into the current phase of the action being rewarded."""
    term, plan = skill_term(env), plan_of(env)
    tick = term.tick.float()
    start = torch.where(
        term.phase == HOLD,
        torch.zeros_like(tick),
        torch.where(term.phase == UNWIND, plan.hold_ticks.float(), plan.skill_ticks.float()),
    )
    return (tick - start) * env.step_dt


def _in_skill(env: Any) -> torch.Tensor:
    ph = _phase(env)
    return (ph == HOLD) | (ph == UNWIND)


def _hop_window(env: Any) -> torch.Tensor:
    plan = plan_of(env)
    return plan.hop & (_phase(env) == HOLD) & (_t_phase(env) < C.HOP_WINDOW_S)


# ── the episode: sampling a plan and an entry ───────────────────────────────────────────────


@dataclass
class _Net:
    name: str
    variants: tuple[C.Variant, ...]
    weights: torch.Tensor


def _net_info(env: Any, net: str) -> _Net:
    cache = env.__dict__.setdefault("_skill_net_cache", {})
    if net not in cache:
        robot = _robot(env)
        names = [robot.joint_names[i] for i in microduck_mdp._servo_joint_ids(env, robot)]
        if tuple(names) != C.JOINTS:
            raise RuntimeError(f"servo order {names} is not robotd's policy order {C.JOINTS}")
        vs = C.variants(net)
        w = C.variant_weights(net)
        cache[net] = _Net(net, vs, torch.tensor([w[v.name] for v in vs], device=env.device))
    return cache[net]


def _uniform(n: int, lo: float, hi: float, device: str) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(n, device=device)


def reset_skill_episode(
    env: Any,
    env_ids: Any,
    net: str = "startle",
    mix_stage: str = "full",
    library_path: str | None = None,
    xy_range: float = 0.3,
    eval_cells: list[tuple[str, str]] | None = None,
    handback_s: float = 0.0,
    nominal_windows: bool = False,
) -> None:
    """Reset event: choose each env's table entry, windows, targets and entry state.

    ``eval_cells`` (evaluation only): a list of (variant name, entry kind) assigned round-robin
    over envs, with the table's nominal windows and no random cuts; ``handback_s`` adds the
    evaluator's hand-back phase (v5 afterwards)."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    env_ids = env_ids.to(env.device, dtype=torch.long)
    n = len(env_ids)
    if n == 0:
        return
    dev = env.device
    info = _net_info(env, net)
    plan = plan_of(env)
    entry_of(env)  # make sure the action term finds the entry state
    dt = env.step_dt
    vs = info.variants

    # 1. variant and entry kind
    if eval_cells:
        names = [v.name for v in vs]
        cells = [(names.index(v), C.ENTRY_KINDS.index(k)) for v, k in eval_cells]
        cell = env_ids % len(cells)
        vidx = torch.tensor([c[0] for c in cells], device=dev)[cell]
        kind = torch.tensor([c[1] for c in cells], device=dev)[cell]
    else:
        vidx = torch.multinomial(info.weights, n, replacement=True)
        kind = torch.zeros(n, dtype=torch.long, device=dev)
        for i, v in enumerate(vs):
            sel = (vidx == i).nonzero().flatten()
            if len(sel) == 0:
                continue
            mix = C.entry_mix(v.behaviour, v.postures, early=(mix_stage == "early"))
            kinds = list(mix)
            probs = torch.tensor([mix[k] for k in kinds], device=dev)
            pick = torch.multinomial(probs, len(sel), replacement=True)
            kind[sel] = torch.tensor([C.ENTRY_KINDS.index(k) for k in kinds], device=dev)[pick]

    # 2. windows, commands, targets per variant
    beh = torch.zeros(n, dtype=torch.long, device=dev)
    level = torch.zeros(n, device=dev)
    command = torch.zeros(n, 3, device=dev)
    unwind = torch.zeros(n, 3, device=dev)
    hold_s = torch.zeros(n, device=dev)
    unwind_s = torch.zeros(n, device=dev)
    cut = torch.zeros(n, dtype=torch.bool, device=dev)
    release = C.release_variant() if net == "startle" else None
    for i, v in enumerate(vs):
        sel = (vidx == i).nonzero().flatten()
        m = len(sel)
        if m == 0:
            continue
        beh[sel] = BEHAVIOUR_ID[v.behaviour]
        lv = torch.full((m,), v.level, device=dev)
        cmd = torch.tensor(v.command, device=dev).repeat(m, 1)
        if not eval_cells and v.behaviour in ("crouch", "shake"):
            cont = torch.rand(m, device=dev) < C.CONTINUOUS_LEVEL_PROB
            lo, hi = (0.25, 1.0) if v.behaviour == "crouch" else (0.5, 1.0)
            lv = torch.where(cont, _uniform(m, lo, hi, dev), lv)
            if v.behaviour == "crouch":
                cmd[:, 0] = lv
            else:
                cmd[:, 2] = lv
        level[sel] = lv
        command[sel] = cmd
        unwind[sel] = torch.tensor(v.unwind, device=dev)
        h = torch.full((m,), v.hold_s, device=dev)
        u = torch.full((m,), v.unwind_s, device=dev)
        if not eval_cells and not nominal_windows:
            jit = torch.rand(m, device=dev) < C.HOLD_JITTER_PROB
            h = torch.where(jit, h * _uniform(m, *C.HOLD_JITTER, dev), h)
            p_long = C.HOLD_LONG_PROB.get(v.behaviour, 0.0)
            if (
                p_long > 0
            ):  # rev a4: held crouches (the live table re-requests a hold while the fear lasts)
                long = torch.rand(m, device=dev) < p_long
                h = torch.where(long, _uniform(m, *C.HOLD_LONG_S, dev), h)
            p_cut = C.RELEASE_CUT_PROB.get(v.behaviour, 0.0)
            if release is not None and p_cut > 0:
                c = torch.rand(m, device=dev) < p_cut
                h = torch.where(c, _uniform(m, 0.15, 1.0, dev) * h.clamp(min=0.2), h)
                u = torch.where(c, torch.full_like(u, release.hold_s), u)
                unwind[sel] = torch.where(
                    c.unsqueeze(1),
                    torch.tensor(release.command, device=dev).expand(m, 3),
                    unwind[sel],
                )
                cut[sel] = c
        hold_s[sel] = h
        unwind_s[sel] = u

    ids = env_ids
    plan.variant[ids], plan.behaviour[ids], plan.kind[ids] = vidx, beh, kind
    plan.level[ids] = level
    plan.command[ids], plan.unwind_cmd[ids] = command, unwind
    plan.hold_ticks[ids] = torch.round(hold_s / dt).long()
    plan.unwind_ticks[ids] = torch.round(unwind_s / dt).long()
    plan.handback_ticks[ids] = int(round(handback_s / dt))
    plan.cut[ids] = cut
    seated = kind == KIND_ID["seated"]
    plan.seated[ids] = seated
    plan.handback_cmd[ids] = 0.0
    plan.handback_cmd[ids, 0] = seated.float()  # robotd re-seats: sitstand with the sit flag
    _set_targets(env, ids, beh, level, seated)
    _write_entry_state(env, ids, kind, library_path, xy_range, level)


def _table_interp(x: torch.Tensor, table) -> torch.Tensor:
    """Linear interpolation of a (x, (a, b, c)) table at each x (clamped at the ends): (N, 3)."""
    xs = torch.tensor([row[0] for row in table], device=x.device)
    ys = torch.tensor([row[1] for row in table], device=x.device)
    xc = x.clamp(float(xs[0]), float(xs[-1]))
    hi = torch.searchsorted(xs, xc.contiguous()).clamp(1, len(xs) - 1)
    lo = hi - 1
    t = ((xc - xs[lo]) / (xs[hi] - xs[lo])).unsqueeze(1)
    return ys[lo] + t * (ys[hi] - ys[lo])


def _mirror_legs_torch(base: torch.Tensor, left: torch.Tensor) -> torch.Tensor:
    q = base.clone()
    q[:, list(C.LEFT_PITCH_CHAIN)] += left
    q[:, list(C.RIGHT_PITCH_CHAIN)] -= left
    return q


def _lerp_t(a: float, b: float, t: torch.Tensor) -> torch.Tensor:
    return a + (b - a) * t.clamp(0.0, 1.0)


# The shake's training bands, (a = 0.5, a = 1.0) endpoints. rev c2 and earlier: roll_lo (4, 6), roll_hi (6, 9) deg,
# roll_hz (2.5, 4), head_lo (0.15, 0.22), head_hi (0.22, 0.30) rad, head_hz (3, 5). rev c3 (tr:train 2026-09-28):
# roll_lo (4.5, 5.0), roll_hi (5.5, 6.5). rev c4 (s1b:retrain 2026-09-28): a bigger shake-off — the trunk roll
# +1° (c3 reached 5.4–6.0° p50 at leg |τ| p95 0.23–0.26 N·m on the A100: the legs have margin to the 0.35 gate);
# the head yaw bands stay (head |τ| p95 0.24–0.28 on the A100, 0.30–0.36 on the Mac: no margin), and the window
# is 1.5 s (rl_contract.TRAIN_HOLD_S).
SHAKE_BANDS = {
    "roll_lo": (5.5, 6.0), "roll_hi": (6.5, 7.5), "roll_hz": (2.7, 4.0),
    "head_lo": (0.17, 0.19), "head_hi": (0.21, 0.24), "head_hz": (3.2, 5.0),
}  # fmt: skip


def _set_targets(env, ids, beh, level, seated) -> None:
    """Per-env blend endpoints (vectorised versions of rl_contract's targets)."""
    plan, dev = plan_of(env), env.device
    n = len(ids)
    home = torch.tensor(C.HOME, device=dev).expand(n, -1)
    sit = torch.tensor(C.sit_pose(), device=dev).expand(n, -1)
    q_base = torch.where(seated.unsqueeze(1), sit, home).clone()
    z_base = torch.where(seated, torch.full_like(level, C.SIT_Z), torch.full_like(level, C.STAND_Z))
    is_ = lambda name: beh == BEHAVIOUR_ID[name]  # noqa: E731
    crouchy = is_("crouch") | is_("hop") | is_("rest")
    # crouch / hop / rest: the startle posture (rest keeps it only as the entry's reference)
    k = torch.where(is_("crouch"), level, torch.full_like(level, C.HOP_K))
    t = (k.clamp(0.25, 1.0) - 0.25) / 0.75
    # rev a3: the hop steps back at a shallower trunk dip (servo strain); its neck pull-back stays k = HOP_K
    k_dip = torch.where(is_("hop"), torch.full_like(level, C.HOP_DIP_K), k)
    dip = _lerp_t(0.008, 0.025, (k_dip.clamp(0.25, 1.0) - 0.25) / 0.75)
    neck = _lerp_t(0.15, 0.35, t)
    crouch_q = _mirror_legs_torch(home, _table_interp(dip, C.HOME_DIP_TABLE))
    crouch_q[:, C.NECK_PITCH] += neck
    crouch_q[:, C.HEAD_PITCH] -= neck * 2.0 / 3.0
    # stretches
    tall, st = C.STRETCH_TALL, C.STRETCH_SIT
    tall_q = torch.tensor(tall.pose(), device=dev).expand(n, -1)
    sit_q = torch.tensor(st.pose(), device=dev).expand(n, -1)
    is_tall, is_sit = is_("stretch_tall"), is_("stretch_sit")
    q_full = torch.where(crouchy.unsqueeze(1), crouch_q, q_base)
    q_full = torch.where(is_tall.unsqueeze(1), tall_q, q_full)
    q_full = torch.where(is_sit.unsqueeze(1), sit_q, q_full)
    z_full = torch.where(crouchy, C.STAND_Z - dip, z_base)
    z_full = torch.where(is_tall, torch.full_like(z_full, tall.base_z + tall.rise_m), z_full)
    z_full = torch.where(is_sit, torch.full_like(z_full, st.base_z + st.rise_m), z_full)
    pitch = torch.where(crouchy, torch.deg2rad(_lerp_t(1.5, 4.0, t)), torch.zeros_like(t))
    cam = torch.where(is_tall, torch.full_like(t, tall.camera_deg), torch.zeros_like(t))
    cam = torch.where(is_sit, torch.full_like(t, st.camera_deg), cam)
    ramp_up = torch.where(crouchy, _lerp_t(0.15, 0.25, t), torch.full_like(t, 0.2))
    ramp_up = torch.where(is_tall | is_sit, torch.full_like(t, tall.ramp_s), ramp_up)
    goal = (is_("crouch") | is_("hop") | is_tall | is_sit).float()
    # shake bands (linear in a, rl_contract.shake_target)
    a = (level.clamp(0.5, 1.0) - 0.5) / 0.5
    # rev c3: training bands with margin over the gate (roll ≥ 4° at 2.5–4 Hz, head yaw ≥ 0.15 rad at ≥ 3 Hz) and
    # smaller amplitudes than rev c2 (c2's shake_2 ran head |τ| p95 0.44, legs 0.37 N·m vs the 0.35 strain gate)
    b = SHAKE_BANDS
    bands = torch.stack(
        [
            _lerp_t(*b["roll_lo"], a),
            _lerp_t(*b["roll_hi"], a),
            torch.full_like(a, b["roll_hz"][0]),
            torch.full_like(a, b["roll_hz"][1]),
            _lerp_t(*b["head_lo"], a),
            _lerp_t(*b["head_hi"], a),
            torch.full_like(a, b["head_hz"][0]),
            torch.full_like(a, b["head_hz"][1]),
        ],  # fmt: skip
        dim=1,
    ) * is_("shake").unsqueeze(1)
    head_w = torch.ones(n, 4, device=dev)
    head_w[:, 2] = torch.where(is_("shake"), 0.0, 1.0)  # the shake swings the head's yaw
    plan.q_base[ids], plan.q_full[ids] = q_base, q_full
    plan.z_base[ids], plan.z_full[ids] = z_base, z_full
    plan.pitch_full[ids], plan.cam_full[ids] = pitch, cam
    plan.goal_hold[ids], plan.ramp_up[ids] = goal, ramp_up
    plan.ramp_down[ids] = C.UNWIND_RAMP_S
    plan.bands[ids] = bands
    plan.head_w[ids] = head_w
    plan.hop[ids] = is_("hop")
    plan.shake[ids] = is_("shake")
    plan.stretch[ids] = is_tall | is_sit


def _write_entry_state(env, ids, kind, library_path, xy_range, level) -> None:
    """Sample each env's entry state (library first, synthetic otherwise) and write it."""
    from duckmind.skilltrain import rl_library

    dev, n = env.device, len(ids)
    robot = _robot(env)
    plan, entry = plan_of(env), entry_of(env)
    lib = rl_library.cached(env, library_path)
    state = rl_library.sample_states(lib, kind, dev) if lib is not None else None
    synth = rl_library.synthetic_states(kind, dev)
    if state is None:
        state = synth
    else:
        use_synth = state["missing"] | (kind == KIND_ID["crouched"])
        for key in ("root_z", "root_quat", "lin_vel", "ang_vel", "q", "qd", "last_raw",
                    "prev_target", "has_prev"):  # fmt: skip
            a, b = state[key], synth[key]
            mask = use_synth.view(-1, *([1] * (a.dim() - 1)))
            state[key] = torch.where(mask, b, a)
    yaw = _uniform(n, -math.pi, math.pi, dev)
    quat = _quat_mul(_quat_yaw(yaw), state["root_quat"])
    quat = quat / quat.norm(dim=-1, keepdim=True)
    xy = env.scene.env_origins[ids, :2] + (torch.rand(n, 2, device=dev) * 2 - 1) * xy_range
    z = state["root_z"] + env.scene.env_origins[ids, 2]
    pose = torch.cat([xy, z.unsqueeze(1), quat], dim=1)
    vel = torch.cat([_rot_z(state["lin_vel"], yaw), _rot_z(state["ang_vel"], yaw)], dim=1)
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
    plan.entry_xy[ids] = xy
    plan.entry_yaw[ids] = _yaw(quat)
    plan.entry_z[ids] = state["root_z"]
    # the blend starts where the entry already is (a crouched entry is part-way down)
    dip_full = (plan.z_base[ids] - plan.z_full[ids]).clamp(min=1e-4)
    already = ((plan.z_base[ids] - state["root_z"]) / dip_full).clamp(0.0, 1.0)
    crouchy = (
        (plan.behaviour[ids] == BEHAVIOUR_ID["crouch"])
        | (plan.behaviour[ids] == BEHAVIOUR_ID["hop"])
        | (plan.behaviour[ids] == BEHAVIOUR_ID["rest"])
    )
    plan.alpha0[ids] = torch.where(
        crouchy & (kind == KIND_ID["crouched"]), already, torch.zeros_like(already)
    )


# ── the command term ────────────────────────────────────────────────────────────────────────


class SkillCommand(CommandTerm):
    """The skill's twist as robotd feeds it (constant, unsmoothed), plus the phase and the slewed
    target blend the rewards track. Registered as ``twist`` so the observation layout and the
    exported metadata match every microduck policy."""

    cfg: SkillCommandCfg

    def __init__(self, cfg: SkillCommandCfg, env: Any):
        super().__init__(cfg, env)
        n, dev = self.num_envs, self.device
        self._cmd = torch.zeros(n, 3, device=dev)
        self.phase = torch.zeros(n, dtype=torch.long, device=dev)
        self.alpha = torch.zeros(n, device=dev)
        self.tick = torch.zeros(n, dtype=torch.long, device=dev)
        self._last_n = torch.zeros(n, dtype=torch.long, device=dev)

    @property
    def command(self) -> torch.Tensor:
        return self._cmd

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        plan = plan_of(self._env)
        self.alpha[env_ids] = plan.alpha0[env_ids]
        self._last_n[env_ids] = 0
        self.tick[env_ids] = 0
        self._update_command()

    def _update_command(self) -> None:
        env, plan = self._env, plan_of(self._env)
        n = env.episode_length_buf
        inc = (n - self._last_n).clamp(min=0).float()
        self._last_n = n.clone()
        hold = plan.hold_ticks
        unw = hold + plan.unwind_ticks
        hb = unw + plan.handback_ticks
        phase = torch.where(
            n < hold,
            HOLD,
            torch.where(n < unw, UNWIND, torch.where(n < hb, HANDBACK, DONE)),
        )
        goal = torch.where(phase == HOLD, plan.goal_hold, torch.zeros_like(plan.goal_hold))
        ramp = torch.where(phase == HOLD, plan.ramp_up, plan.ramp_down).clamp(min=1e-3)
        step = inc * env.step_dt / ramp
        self.alpha = self.alpha + (goal - self.alpha).clamp(-1.0, 1.0).clamp(-step, step)
        cmd = torch.where(
            (phase == HOLD).unsqueeze(1),
            plan.command,
            torch.where(
                (phase == UNWIND).unsqueeze(1),
                plan.unwind_cmd,
                torch.where(
                    (phase == HANDBACK).unsqueeze(1),
                    plan.handback_cmd,
                    torch.zeros_like(plan.command),
                ),
            ),
        )
        self.phase = phase
        self.tick = n.clone()
        self._cmd[:] = cmd


@dataclass(kw_only=True)
class SkillCommandCfg(CommandTermCfg):
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    def build(self, env: Any) -> SkillCommand:
        return SkillCommand(self, env)


# ── observations (the critic's privileged extras; the actor stays the 61-D contract) ────────


def skill_state_obs(env: Any) -> torch.Tensor:
    """Phase one-hot (4), seconds into the phase, seconds left in hold and unwind, alpha,
    behaviour one-hot (6), level, entry kind one-hot (7), height error to the blend target (×10),
    backward displacement since entry: 24 dims."""
    term, plan = skill_term(env), plan_of(env)
    dt = env.step_dt
    tick = term.tick.float()
    phase = torch.nn.functional.one_hot(term.phase.clamp(0, 3), 4).float()
    hold_left = ((plan.hold_ticks.float() - tick) * dt).clamp(min=0.0)
    unwind_left = ((plan.skill_ticks.float() - tick) * dt).clamp(min=0.0)
    beh = torch.nn.functional.one_hot(plan.behaviour, len(C.BEHAVIOURS)).float()
    kind = torch.nn.functional.one_hot(plan.kind, len(C.ENTRY_KINDS)).float()
    _, z_t, a = _blend(env)
    z_err = (_trunk_z(env) - z_t).unsqueeze(1) * 10.0
    back = _back_disp(env).unsqueeze(1)
    return torch.cat(
        [
            phase,
            _t_phase(env).unsqueeze(1),
            hold_left.unsqueeze(1),
            unwind_left.unsqueeze(1),
            a.unsqueeze(1),
            beh,
            plan.level.unsqueeze(1),
            kind,
            z_err,
            back,
        ],  # fmt: skip
        dim=1,
    )


# ── rewards: task terms (≥ 0, positive weight) ──────────────────────────────────────────────


def pose_legs(env: Any, std: float = 0.3) -> torch.Tensor:
    """Leg joints toward the blended posture — not in the hop window (rev a1): the back-steps
    need the legs; the trunk height, pitch and the head stay tracked there."""
    q_t, _, _ = _blend(env)
    q = _servo_q(env)
    idx = list(C.LEG_IDX)
    score = torch.exp(-(((q[:, idx] - q_t[:, idx]) / std) ** 2)).mean(dim=1)
    return score * (_in_skill(env) & ~_hop_window(env))


def _head_err2(env: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-joint squared head error vs the blend target and the per-env joint weights (the shake
    leaves the head's yaw free)."""
    q_t, _, _ = _blend(env)
    idx = list(C.HEAD_IDX)
    return (_servo_q(env)[:, idx] - q_t[:, idx]) ** 2, plan_of(env).head_w


def pose_head(env: Any, std: float = 0.12) -> torch.Tensor:
    err2, w = _head_err2(env)
    score = (torch.exp(-err2 / std**2) * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
    return score * _in_skill(env)


def height_track(env: Any, std: float = 0.01) -> torch.Tensor:
    _, z_t, _ = _blend(env)
    return torch.exp(-(((_trunk_z(env) - z_t) / std) ** 2)) * _in_skill(env)


def pitch_track(env: Any, std_deg: float = 4.0) -> torch.Tensor:
    """Trunk pitch toward the blended pitch-back (crouches only; others target level)."""
    _, _, a = _blend(env)
    _, pitch = _roll_pitch(_robot(env).data.root_link_quat_w)
    # mjlab/MuJoCo pitch about +y is nose DOWN for this asin convention → nose-up target = −
    target = -a * plan_of(env).pitch_full
    return torch.exp(-(((pitch - target) / math.radians(std_deg)) ** 2)) * _in_skill(env)


def composite(
    env: Any,
    height_std: float = 0.015,
    upright_std: float = 0.35,
    legs_std: float = 0.35,
    head_std: float = 0.25,
) -> torch.Tensor:
    """Multiplicative goal score (height · upright · legs · head) vs the blended target: a lean,
    a sit or a dangling head collapses it (the SitStand lesson). In the hop window the leg factor
    is 1 (rev a1: the legs step there)."""
    q_t, z_t, _ = _blend(env)
    q = _servo_q(env)
    legs = list(C.LEG_IDX)
    h = torch.exp(-(((_trunk_z(env) - z_t) / height_std) ** 2))
    quat = _robot(env).data.root_link_quat_w
    up = torch.exp(-(2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)) / (upright_std**2))
    lg = torch.exp(-((q[:, legs] - q_t[:, legs]) ** 2).mean(dim=1) / (legs_std**2))
    lg = torch.where(_hop_window(env), torch.ones_like(lg), lg)
    err2, w = _head_err2(env)
    hd = torch.exp(-(err2 * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0) / (head_std**2))
    return h * up * lg * hd * _in_skill(env)


def still(env: Any, vel_std: float = 0.05) -> torch.Tensor:
    """Trunk still in the ground plane (the freeze), outside the back-step window."""
    v = _robot(env).data.root_link_lin_vel_w[:, :2].norm(dim=1)
    shaking = plan_of(env).shake & (_phase(env) == HOLD)  # the shake moves the trunk: drift rules
    gate = _in_skill(env) & ~_hop_window(env) & ~shaking
    return torch.exp(-((v / vel_std) ** 2)) * gate


def planted(env: Any, sensor_name: str = "feet_ground_contact") -> torch.Tensor:
    gate = _in_skill(env) & ~_hop_window(env)
    return microduck_mdp.feet_grounded_reward(env, sensor_name) * gate


def _back_disp(env: Any) -> torch.Tensor:
    plan = plan_of(env)
    xy = _robot(env).data.root_link_pos_w[:, :2]
    d = xy - plan.entry_xy
    heading = torch.stack([torch.cos(plan.entry_yaw), torch.sin(plan.entry_yaw)], dim=1)
    return -(d * heading).sum(dim=1)


def hop_progress(env: Any, cap_m: float = C.HOP_BACK_M) -> torch.Tensor:
    """Potential-based back-steps: (Φ_t − Φ_{t−1}) / dt with Φ = min(backward displacement
    along the entry heading, cap), paid inside the hop window only (then planted). Moving
    forward costs, back-and-forth nets zero, nothing past the cap pays."""
    phi = torch.clamp(_back_disp(env), max=cap_m)
    prev = getattr(env, "_skill_hop_phi", None)
    fresh = env.episode_length_buf <= 1
    if prev is None or prev.shape != phi.shape:
        prev = phi.clone()
    prev = torch.where(fresh, phi, prev)
    out = (phi - prev) / env.step_dt
    env._skill_hop_phi = phi.clone()
    return out * _hop_window(env)


def hop_steps(
    env: Any, sensor_name: str = "feet_ground_contact", lo: float = 0.04, hi: float = 0.3
):
    """Real back-steps, not a shuffle: a touchdown after an air time in [lo, hi] s pays, inside
    the hop window."""
    sensor = env.scene.sensors[sensor_name]
    first = sensor.compute_first_contact(env.step_dt)[:, :2]
    air = sensor.data.last_air_time[:, :2]
    good = first & (air >= lo) & (air <= hi)
    return good.float().sum(dim=1) * _hop_window(env)


def shake_servo_torque(env: Any, free_nm: float = 0.30) -> torch.Tensor:
    """rev c3: Σ over all 14 servos of |torque| above ``free_nm`` (N·m) through a shake's window (the rev-c2
    shake_2 ran head |τ| p95 0.44 and legs 0.37 N·m against the 0.35 strain gate)."""
    tau = torch.nan_to_num(_robot(env).data.actuator_force, nan=0.0).abs()
    excess = (tau - free_nm).clamp(min=0.0).sum(dim=1)
    return excess * (plan_of(env).shake & _in_skill(env))


def hop_leg_torque(env: Any, free_nm: float = 0.30) -> torch.Tensor:
    """rev a3: Σ over the ten leg servos of |torque| above ``free_nm`` (N·m) through a hop-back's window
    (the plan's strain gate is a p95 ≤ 0.35 N·m; the a2 hop ran 0.49–0.62)."""
    tau = torch.nan_to_num(_robot(env).data.actuator_force, nan=0.0).abs()
    excess = (tau[:, list(C.LEG_IDX)] - free_nm).clamp(min=0.0).sum(dim=1)
    return excess * (plan_of(env).hop & _in_skill(env))


def yaw_hold(env: Any, std: float = 0.15) -> torch.Tensor:
    plan = plan_of(env)
    dyaw = _wrap(_yaw(_robot(env).data.root_link_quat_w) - plan.entry_yaw)
    return torch.exp(-((dyaw / std) ** 2)) * _in_skill(env)


def hop_yaw(env: Any) -> torch.Tensor:
    """rev a2: |heading change since the entry| (rad) through a hop-back's window — a linear cost.
    The Gaussian ``yaw_hold`` (σ 0.15 rad) has no gradient at the 39–55° turns the rev-a1 hop
    learned by iteration 250 (the gate allows ≤ 15°)."""
    plan = plan_of(env)
    dyaw = _wrap(_yaw(_robot(env).data.root_link_quat_w) - plan.entry_yaw).abs()
    return dyaw * (plan.hop & _in_skill(env))


# ── rev a4 / c4 (s1b:retrain, 2026-09-28) ───────────────────────────────────────────────────


def _walk_entry(env: Any) -> torch.Tensor:
    """Episodes entered from a walk or a coast (the startle network's stop-and-flinch)."""
    kind = plan_of(env).kind
    return (kind == KIND_ID["walk"]) | (kind == KIND_ID["coast"])


def walk_stop_torque(
    env: Any, free_nm: float = C.WALK_STOP_FREE_NM, window_s: float = C.WALK_STOP_WINDOW_S
) -> torch.Tensor:
    """rev a4: Σ over the ten leg servos of |torque| above ``free_nm`` (N·m) in the first ``window_s`` of a
    skill entered from a walk or a coast (the Mac: startle_4 stopping from 0.55 m/s at leg |τ| p95 0.56–0.64
    against v5's own stop 0.33–0.40). The hop-back has its own torque term."""
    plan = plan_of(env)
    tau = torch.nan_to_num(_robot(env).data.actuator_force, nan=0.0).abs()
    excess = (tau[:, list(C.LEG_IDX)] - free_nm).clamp(min=0.0).sum(dim=1)
    gate = _walk_entry(env) & (_phase(env) == HOLD) & (_t_phase(env) < window_s) & ~plan.hop
    return excess * gate


def walk_yaw(env: Any, free_rad: float = C.WALK_YAW_FREE_RAD) -> torch.Tensor:
    """rev a4: |heading change since the entry| above ``free_rad`` through the skill of a walking or coasting
    entry — the stop holds the heading (BAM spun > 20° in up to 30 % of the 0.55 m/s Mac cells)."""
    plan = plan_of(env)
    dyaw = _wrap(_yaw(_robot(env).data.root_link_quat_w) - plan.entry_yaw).abs()
    return (dyaw - free_rad).clamp(min=0.0) * (_walk_entry(env) & _in_skill(env) & ~plan.hop)


def _feet_xy(env: Any) -> torch.Tensor:
    robot = _robot(env)
    cache = env.__dict__.setdefault("_skill_site_cache", {})
    if "feet" not in cache:
        ids, _ = robot.find_sites(["left_foot", "right_foot"], preserve_order=True)
        cache["feet"] = ids
    return robot.data.site_pos_w[:, cache["feet"], :2]


def _hop_steps(env: Any, sensor_name: str = "feet_ground_contact") -> dict[str, torch.Tensor]:
    """Per env step (computed once): each foot's touchdown this tick (≥ 40 ms of air) with its step's backward
    length along the entry heading (from where the foot left the floor), and the hop window's step count.

    A foot's anchor follows it while it touches the floor and stays where it lifted off while it is in the air,
    so at the touchdown ``foot − anchor`` is the step (a slide in contact is not a step)."""
    step = int(env.common_step_counter)
    st = getattr(env, "_skill_hop_steps", None)
    if st is not None and st.get("step") == step:
        return st
    n, dev = env.num_envs, env.device
    feet = _feet_xy(env)
    if st is None or st["anchor"].shape[0] != n:
        st = {"anchor": feet.clone(), "count": torch.zeros(n, device=dev)}
        env._skill_hop_steps = st
    sensor = env.scene.sensors[sensor_name]
    contact = sensor.data.found[:, :2] > 0
    first = sensor.compute_first_contact(env.step_dt)[:, :2]
    air = sensor.data.last_air_time[:, :2]
    fresh = env.episode_length_buf <= 1
    anchor = torch.where(fresh.view(-1, 1, 1), feet, st["anchor"])
    count = torch.where(fresh, torch.zeros_like(st["count"]), st["count"])
    plan = plan_of(env)
    heading = torch.stack([torch.cos(plan.entry_yaw), torch.sin(plan.entry_yaw)], dim=1)
    back = -((feet - anchor) * heading.unsqueeze(1)).sum(dim=2)
    td = first & (air >= C.MIN_STEP_AIR_S) & ~fresh.unsqueeze(1)
    win = _hop_window(env)
    prev = count
    count = prev + (td & win.unsqueeze(1)).float().sum(dim=1)
    st.update(
        anchor=torch.where(contact.unsqueeze(2), feet, anchor),
        count=count,
        prev_count=prev,
        td=td,
        real=td & (air >= C.HOP_STEP_AIR_S),
        back=back,
        win=win,
        both_air=~contact.any(dim=1),
        step=step,
    )
    return st


def hop_long_steps(env: Any) -> torch.Tensor:
    """rev a4: real back-steps (a foot airborne ≥ 0.06 s) inside the hop window, paid by their backward length —
    0 at ≤ 1.5 cm, 1 at ≥ 3 cm (``C.HOP_STEP_M``) — for the first ``C.HOP_MAX_STEPS`` steps: two to three real
    steps, not rev a3's 7–8 shuffles of ≈ 1.2 cm."""
    st = _hop_steps(env)
    lo, hi = C.HOP_STEP_M
    pay = ((st["back"] - lo) / (hi - lo)).clamp(0.0, 1.0) * st["real"].float()
    under = (st["prev_count"] < C.HOP_MAX_STEPS).float()
    return pay.sum(dim=1) * under * st["win"]


def hop_extra_steps(env: Any) -> torch.Tensor:
    """rev a4: every touchdown (≥ 40 ms of air) beyond the ``C.HOP_MAX_STEPS``-th inside the hop window."""
    st = _hop_steps(env)
    cap = float(C.HOP_MAX_STEPS)
    over = (st["count"] - cap).clamp(min=0.0) - (st["prev_count"] - cap).clamp(min=0.0)
    return over * st["win"]


def hop_flight(env: Any) -> torch.Tensor:
    """rev a4 (plan D5, back-steps instead of a jump): both feet off the floor during a hop-back's hold."""
    st = _hop_steps(env)
    return st["both_air"].float() * (plan_of(env).hop & (_phase(env) == HOLD))


def stretch_acc(env: Any, free: float = C.STRETCH_ACC_FREE) -> torch.Tensor:
    """rev c4: Σ over the servos of |joint acceleration| (control-rate difference) above ``free`` rad/s² through
    a stretch's window (the Mac: stretch_tall's acceleration p99 316 rad/s² against the 300 gate)."""
    robot = _robot(env)
    qd = torch.nan_to_num(
        robot.data.joint_vel[:, microduck_mdp._servo_joint_ids(env, robot)], nan=0.0
    )
    prev = getattr(env, "_skill_stretch_prev_qd", None)
    if prev is None or prev.shape != qd.shape:
        prev = qd.clone()
    acc = (qd - prev) / env.step_dt
    acc = torch.where((env.episode_length_buf <= 1).unsqueeze(1), torch.zeros_like(acc), acc)
    env._skill_stretch_prev_qd = qd.clone()
    return (acc.abs() - free).clamp(min=0.0).sum(dim=1) * (plan_of(env).stretch & _in_skill(env))


def randomize_servo_lag(
    env: Any, env_ids: Any, min_lag: int = 0, max_lag: int = 8, fixed: int | None = None
) -> None:
    """rev a4/c4 reset event: each reset env starts its episode with a servo command lag (physics steps) drawn
    from ``min_lag``–``max_lag`` (or ``fixed``); the actuator's delay buffer re-draws it during the episode as
    its config says (``rl_tasks``: every 0.5 s with probability 0.5 in training, never in a fixed-lag
    evaluation). Runs after the scene reset that zeroes the buffer's lags."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    env_ids = env_ids.to(env.device, dtype=torch.long)
    if len(env_ids) == 0:
        return
    if fixed is not None:
        lags = torch.full((len(env_ids),), int(fixed), dtype=torch.long, device=env.device)
    else:
        lags = torch.randint(int(min_lag), int(max_lag) + 1, (len(env_ids),), device=env.device)
    for act in _robot(env).actuators:
        if getattr(act, "has_delay", False):
            act.set_lags(lags, env_ids)


def servo_lags(env: Any) -> torch.Tensor:
    """The current servo command lag per env (physics steps; 0 without a delay buffer)."""
    for act in _robot(env).actuators:
        buf = getattr(act, "_delay_buffer", None)
        if buf is not None:
            return buf.current_lags
    return torch.zeros(env.num_envs, dtype=torch.long, device=env.device)


def upright(env: Any) -> torch.Tensor:
    return _cos_tilt(env).clamp(min=0.0)


SHAKE_TD_DECAY = 0.7  # rev a1: the shake bands × exp(−0.7 · steps in the window) ≈ 0.5ⁿ
# rev c2 (tr:train, 2026-09-28): only a real step counts — a foot airborne ≥ 0.06 s. Counting every first
# contact (rev a1) let the exploration noise's contact chatter (≈ 9 per 0.5 s window at iteration 148) drive
# the shake reward to ≈ 0 (0.0015 vs 0.285 without the factor in S0): the shake could not be discovered.
SHAKE_STEP_AIR_S = 0.06


def _osc(env: Any) -> dict[str, torch.Tensor]:
    """Trunk-roll and head-yaw oscillation statistics over the last SHAKE_WINDOW_S of the hold
    (updated once per env step): amplitude = √2·std, frequency = std(rate)/(2π·std); plus the
    feet's touchdowns inside the same window (a shake is done with the feet planted)."""
    step = int(env.common_step_counter)
    cache = getattr(env, "_skill_osc", None)
    w = max(4, int(round(C.SHAKE_WINDOW_S / env.step_dt)))
    n, dev = env.num_envs, env.device
    if cache is None:
        cache = {
            "buf": torch.zeros(n, 4, w, device=dev),
            "valid": torch.zeros(n, w, device=dev),
            "td": torch.zeros(n, w, device=dev),
            "i": 0,
            "step": -1,
        }
        env._skill_osc = cache
    if cache["step"] == step:
        return cache["stats"]
    robot = _robot(env)
    roll, _ = _roll_pitch(robot.data.root_link_quat_w)
    roll_rate = robot.data.root_link_ang_vel_b[:, 0]
    q = _servo_q(env)
    servo = microduck_mdp._servo_joint_ids(env, robot)
    qd = robot.data.joint_vel[:, servo]
    sample = torch.stack([roll, roll_rate, q[:, C.HEAD_YAW], qd[:, C.HEAD_YAW]], dim=1)
    fresh = env.episode_length_buf <= 1
    cache["buf"][fresh] = 0.0
    cache["valid"][fresh] = 0.0
    cache["td"][fresh] = 0.0
    i = cache["i"] % w
    cache["buf"][:, :, i] = torch.nan_to_num(sample)
    ok = (plan_of(env).shake & (_phase(env) == HOLD)).float()
    cache["valid"][:, i] = ok
    feet = env.scene.sensors["feet_ground_contact"]
    first = feet.compute_first_contact(env.step_dt)[:, :2]
    steps = (first & (feet.data.last_air_time[:, :2] >= SHAKE_STEP_AIR_S)).float().sum(dim=1)
    cache["td"][:, i] = steps * ok
    cache["i"] = i + 1
    cache["step"] = step
    m = cache["valid"]
    cnt = m.sum(dim=1).clamp(min=1.0)

    def std(x: torch.Tensor) -> torch.Tensor:
        mean = (x * m).sum(dim=1) / cnt
        return torch.sqrt((((x - mean.unsqueeze(1)) ** 2) * m).sum(dim=1) / cnt + 1e-12)

    def rms(x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(((x**2) * m).sum(dim=1) / cnt + 1e-12)

    buf = cache["buf"]
    s_roll, s_yaw = std(buf[:, 0]), std(buf[:, 2])
    stats = {
        "roll_amp_deg": torch.rad2deg(math.sqrt(2.0) * s_roll),
        "roll_hz": rms(buf[:, 1]) / (2 * math.pi * s_roll.clamp(min=1e-3)),
        "yaw_amp": math.sqrt(2.0) * s_yaw,
        "yaw_hz": rms(buf[:, 3]) / (2 * math.pi * s_yaw.clamp(min=1e-3)),
        "samples": m.sum(dim=1),
        "touchdowns": cache["td"].sum(dim=1),
    }
    cache["stats"] = stats
    return stats


def _shake_gate(env: Any) -> torch.Tensor:
    plan = plan_of(env)
    return plan.shake & (_phase(env) == HOLD) & (_t_phase(env) >= C.SHAKE_VALID_AFTER_S)


def _planted_factor(s: dict[str, torch.Tensor]) -> torch.Tensor:
    """rev a1: a shake-off is done with the feet planted (the gate allows ≤ 1 footfall)."""
    return torch.exp(-SHAKE_TD_DECAY * s["touchdowns"])


def shake_roll(env: Any, sigma_hz: float = 1.5) -> torch.Tensor:
    s, b = _osc(env), plan_of(env).bands
    amp = _smooth_band(s["roll_amp_deg"], b[:, 0], b[:, 1], 0.5 * (b[:, 1] - b[:, 0]) + 1.0)
    freq = _freq_band(s["roll_hz"], b[:, 2], b[:, 3], sigma_hz)
    return amp * freq * _planted_factor(s) * _shake_gate(env)


def shake_head(env: Any, sigma_hz: float = 1.5) -> torch.Tensor:
    s, b = _osc(env), plan_of(env).bands
    amp = _smooth_band(s["yaw_amp"], b[:, 4], b[:, 5], 0.5 * (b[:, 5] - b[:, 4]) + 0.02)
    freq = _freq_band(s["yaw_hz"], b[:, 6], b[:, 7], sigma_hz)
    return amp * freq * _planted_factor(s) * _shake_gate(env)


def camera_pitch(env: Any, std_deg: float = 6.0) -> torch.Tensor:
    """Head-camera pitch toward the stretch's target, weighted by the blend (stretches only)."""
    plan = plan_of(env)
    _, _, a = _blend(env)
    err = camera_pitch_deg(env) - plan.cam_full
    return a * torch.exp(-((err / std_deg) ** 2)) * plan.stretch * _in_skill(env)


def handback_bonus(env: Any, z_std: float = 0.008, q_std: float = 0.2, v_std: float = 0.05):
    """One-shot at the skill's last tick: how ready the duck is for the network that takes over
    (velstand: HOME at 0.115 m, upright, still; sitstand: the seat) — the hand-back test."""
    plan = plan_of(env)
    last = env.episode_length_buf == plan.skill_ticks
    q = _servo_q(env)
    h = torch.exp(-(((_trunk_z(env) - plan.z_base) / z_std) ** 2))
    pose = torch.exp(-((q - plan.q_base) ** 2).mean(dim=1) / (q_std**2))
    quat = _robot(env).data.root_link_quat_w
    up = torch.exp(-(2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)) / 0.3**2)
    v = _robot(env).data.root_link_lin_vel_w.norm(dim=1)
    still_ = torch.exp(-((v / v_std) ** 2))
    return h * pose * up * still_ * last


# ── rewards: costs (≥ 0, negative weight) ───────────────────────────────────────────────────


def pose_l1(env: Any) -> torch.Tensor:
    q_t, _, _ = _blend(env)
    return (_servo_q(env) - q_t).abs().mean(dim=1) * _in_skill(env)


def height_l1(env: Any) -> torch.Tensor:
    _, z_t, _ = _blend(env)
    return (_trunk_z(env) - z_t).abs() * _in_skill(env)


def steps_when_still(env: Any, sensor_name: str = "feet_ground_contact", speed: float = 0.15):
    """A foot touching down while the trunk is (nearly) still, outside the back-step window: a
    shuffle. Catching a stumble (trunk moving) is free."""
    sensor = env.scene.sensors[sensor_name]
    first = sensor.compute_first_contact(env.step_dt)[:, :2].float().sum(dim=1)
    v = _robot(env).data.root_link_lin_vel_w[:, :2].norm(dim=1)
    # a shaking trunk moves fast, and a step there is always a shuffle (shake by stepping)
    gate = _in_skill(env) & ~_hop_window(env) & ((v < speed) | plan_of(env).shake)
    return first * gate


def foot_slip(env: Any, sensor_name: str = "feet_ground_contact") -> torch.Tensor:
    """Feet sliding while in contact during the back-step window (steps, not a skate)."""
    robot = _robot(env)
    cache = env.__dict__.setdefault("_skill_site_cache", {})
    if "feet" not in cache:
        ids, _ = robot.find_sites(["left_foot", "right_foot"], preserve_order=True)
        cache["feet"] = ids
    v = robot.data.site_lin_vel_w[:, cache["feet"], :2].norm(dim=2)
    contact = env.scene.sensors[sensor_name].data.found[:, :2] > 0
    return (v * contact).sum(dim=1) * _hop_window(env)


def drift(env: Any, free_m: float = 0.01) -> torch.Tensor:
    """Horizontal travel from the entry beyond ``free_m`` (shake and stretch stay in place)."""
    plan = plan_of(env)
    d = (_robot(env).data.root_link_pos_w[:, :2] - plan.entry_xy).norm(dim=1)
    return (d - free_m).clamp(min=0.0) * (plan.shake | plan.stretch) * _in_skill(env)


def rise_speed(env: Any, cap: float = C.RISE_SPEED_CAP) -> torch.Tensor:
    """Trunk vz above sitstand's rise cap during unwinds and rests (a gentle return)."""
    vz = torch.nan_to_num(_robot(env).data.root_link_lin_vel_w[:, 2], nan=0.0)
    ph = _phase(env)
    gate = (ph == UNWIND) | ((ph == HOLD) & (plan_of(env).goal_hold < 0.5) & ~plan_of(env).shake)
    return (vz - cap).clamp(min=0.0) * gate


def fallen(env: Any, tilt_deg: float = 45.0) -> torch.Tensor:
    return (_cos_tilt(env) < math.cos(math.radians(tilt_deg))).float()


def vertical_accel_cost(env: Any) -> torch.Tensor:
    """|a_z| of the trunk (upstream's self-negating trunk_vertical_accel_penalty, as a cost)."""
    return -microduck_mdp.trunk_vertical_accel_penalty(env)


def torque_sq(env: Any) -> torch.Tensor:
    """Σ τ² over the servos (wear; the shake's torque-RMS cost)."""
    return torch.nan_to_num(_robot(env).data.actuator_force, nan=0.0).pow(2).sum(dim=1)


# ── termination ─────────────────────────────────────────────────────────────────────────────


def skill_done(env: Any) -> torch.Tensor:
    """The window (hold + unwind [+ the evaluator's hand-back]) is over: a true terminal."""
    return env.episode_length_buf >= plan_of(env).end_ticks


# ── metrics (per-step averages over an episode) ─────────────────────────────────────────────


def m_fallen(env: Any) -> torch.Tensor:
    return fallen(env, 45.0)


def m_height_err_mm(env: Any) -> torch.Tensor:
    _, z_t, _ = _blend(env)
    return 1000.0 * (_trunk_z(env) - z_t).abs() * _in_skill(env)


def m_speed(env: Any) -> torch.Tensor:
    return _robot(env).data.root_link_lin_vel_w[:, :2].norm(dim=1) * (_phase(env) == HOLD)


def m_back_mm(env: Any) -> torch.Tensor:
    return 1000.0 * _back_disp(env) * plan_of(env).hop


def m_roll_amp_deg(env: Any) -> torch.Tensor:
    return _osc(env)["roll_amp_deg"] * _shake_gate(env)


def m_shake_touchdowns(env: Any) -> torch.Tensor:
    """Real steps (airborne ≥ 0.06 s) inside the shake's statistics window (a shake by stepping shows here)."""
    return _osc(env)["touchdowns"] * _shake_gate(env)


def m_camera_deg(env: Any) -> torch.Tensor:
    return camera_pitch_deg(env) * plan_of(env).stretch * (_phase(env) == HOLD)


def m_neck_delta(env: Any) -> torch.Tensor:
    return (_servo_q(env)[:, C.NECK_PITCH] - C.HOME[C.NECK_PITCH]) * (_phase(env) == HOLD)


# Terms whose weight is negative (costs). The trainer checks Episode_Reward/<cost> ≤ 0.
COST_FUNCS = (
    pose_l1, height_l1, steps_when_still, foot_slip, drift, rise_speed, fallen,
    vertical_accel_cost, torque_sq, hop_yaw, hop_leg_torque, shake_servo_torque,
    walk_stop_torque, walk_yaw, hop_extra_steps, hop_flight, stretch_acc,
)  # fmt: skip


__all__ = [
    "HANDBACK",
    "HOLD",
    "SkillCommand",
    "SkillCommandCfg",
    "SkillEntry",
    "SkillPlan",
    "UNWIND",
    "camera_pitch",
    "camera_pitch_deg",
    "composite",
    "reset_skill_episode",
    "skill_done",
    "skill_state_obs",
]
