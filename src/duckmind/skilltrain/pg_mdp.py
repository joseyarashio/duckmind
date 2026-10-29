"""MDP pieces of the panic-gait fine-tune: the command mixture (with robotd's twist EMA), the sharper
tracking / capped forward-progress / fall-event terms, and the speed metrics.

Sign convention (Stage 1's): reward functions return ≥ 0; costs get a negative weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from mjlab_microduck.tasks.mdp import VelocityCommandCommandOnly, VelocityCommandCommandOnlyCfg

from duckmind.skilltrain import pg_contract as C


def _robot(env: Any) -> Any:
    return env.scene["robot"]


def _cmd(env: Any) -> torch.Tensor:
    return env.command_manager.get_command("twist")


def _tilt_deg(env: Any) -> torch.Tensor:
    q = _robot(env).data.root_link_quat_w
    cos = torch.nan_to_num(1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2), nan=1.0)
    return torch.rad2deg(torch.acos(cos.clamp(-1.0, 1.0)))


# ── the command ─────────────────────────────────────────────────────────────────────────────


class PgTwistCommand(VelocityCommandCommandOnly):
    """Upstream's velocity command + a strolling band + a panic band, seen through robotd's twist EMA."""

    cfg: PgTwistCommandCfg

    def __init__(self, cfg: PgTwistCommandCfg, env: Any) -> None:
        super().__init__(cfg, env)
        self.ema = torch.zeros(self.num_envs, 3, device=self.device)
        self.vmax = float(cfg.vmax)

    @property
    def command(self) -> torch.Tensor:
        return self.ema

    def reset(self, env_ids: Any) -> dict[str, float]:
        out = super().reset(env_ids)
        self.ema[env_ids] = 0.0  # the duck starts standing: robotd's EMA starts from the stand
        return out

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        m = len(env_ids)
        if m == 0:
            return
        dev = self.device
        cmd = self.vel_command_b
        # standing envs are zeroed every step; the turn-in-place bucket has vx == 0 exactly
        active = (~self.is_standing_env[env_ids]) & (cmd[env_ids, 0] != 0.0)
        u = torch.rand(m, device=dev)
        mid = active & (u < self.cfg.p_mid)
        fast = active & (u >= self.cfg.p_mid) & (u < self.cfg.p_mid + self.cfg.p_fast)
        fwd_only = self.is_forward_env[env_ids]
        lo, hi = self.cfg.mid_vx
        cmd[env_ids[mid], 0] = lo + (hi - lo) * torch.rand(int(mid.sum()), device=dev)
        vmax = max(self.vmax, self.cfg.fast_vx_min + 0.01)
        n = int(fast.sum())
        if n:
            ids = env_ids[fast]
            cmd[ids, 0] = self.cfg.fast_vx_min + (vmax - self.cfg.fast_vx_min) * torch.rand(
                n, device=dev
            )
            keep0 = fwd_only[fast]  # upstream's forward-only envs keep vy = yaw = 0
            vy = (torch.rand(n, device=dev) * 2 - 1) * self.cfg.fast_vy
            straight = torch.rand(n, device=dev) < self.cfg.fast_straight_prob
            yaw_s = (torch.rand(n, device=dev) * 2 - 1) * self.cfg.fast_straight_yaw
            yaw_a = (torch.rand(n, device=dev) * 2 - 1) * self.cfg.yaw_max
            yaw = torch.where(straight, yaw_s, yaw_a)
            cmd[ids, 1] = torch.where(keep0, torch.zeros_like(vy), vy)
            cmd[ids, 2] = torch.where(keep0, torch.zeros_like(yaw), yaw)
        self.vel_command_w[env_ids] = cmd[env_ids]

    def _update_command(self) -> None:
        super()._update_command()
        self.ema += C.CMD_ALPHA * (self.vel_command_b - self.ema)


@dataclass(kw_only=True)
class PgTwistCommandCfg(VelocityCommandCommandOnlyCfg):
    p_mid: float = C.P_MID
    mid_vx: tuple[float, float] = C.MID_VX
    p_fast: float = C.P_FAST
    fast_vx_min: float = C.FAST_VX_MIN
    fast_straight_prob: float = C.FAST_STRAIGHT_PROB
    fast_straight_yaw: float = C.FAST_STRAIGHT_YAW
    fast_vy: float = C.FAST_VY
    yaw_max: float = C.YAW_MAX
    vmax: float = C.VMAX_STAGES[0][1]

    def build(self, env: Any) -> PgTwistCommand:
        return PgTwistCommand(self, env)


def set_vmax(env: Any, env_ids: Any, vmax_stages: list[dict]) -> torch.Tensor:
    """Curriculum: the panic band's upper vx by env step."""
    del env_ids
    term = env.command_manager.get_term("twist")
    value = vmax_stages[0]["vmax"]
    for s in vmax_stages:
        if env.common_step_counter >= s["step"]:
            value = s["vmax"]
    term.vmax = float(value)
    return torch.tensor(float(value))


# ── rewards ─────────────────────────────────────────────────────────────────────────────────


def target_vx(cmd_x: torch.Tensor) -> torch.Tensor:
    """The forward speed the reward asks for (``pg_contract.TARGET_*``): K0 × |cmd| up to the knee, then rising
    to TARGET_TOP_VX at TARGET_TOP_CMD; odd in the command."""
    a = cmd_x.abs()
    k0 = C.TARGET_OVERRIDE.get("k0", C.TARGET_K0)
    top = C.TARGET_OVERRIDE.get("top", C.TARGET_TOP_VX)
    knee_v = k0 * C.TARGET_KNEE
    slope = (top - knee_v) / (C.TARGET_TOP_CMD - C.TARGET_KNEE)
    t = torch.where(a <= C.TARGET_KNEE, k0 * a, knee_v + (a - C.TARGET_KNEE) * slope)
    return torch.sign(cmd_x) * t


def _goal_x(cmd: torch.Tensor, mapped: bool, arc_k: float) -> torch.Tensor:
    """The forward goal; ``arc_k`` > 0 lowers it with the yaw command (an arc may slow down to turn)."""
    gx = target_vx(cmd[:, 0]) if mapped else cmd[:, 0]
    if arc_k:
        gx = gx * (1.0 - arc_k * cmd[:, 2].abs().clamp(max=1.0))
    return gx


def _goal_xy(env: Any, mapped: bool, arc_k: float = 0.0) -> torch.Tensor:
    cmd = _cmd(env)
    return torch.stack([_goal_x(cmd, mapped, arc_k), cmd[:, 1]], dim=1)


def track_vel_sharp(
    env: Any, std: float = 0.18, mapped: bool = False, arc_k: float = 0.0
) -> torch.Tensor:
    """exp(-|v_xy − goal|² / std²), a sharper twin of upstream's std √0.1 term (so a 0.4× gait stops paying)."""
    lin = _robot(env).data.root_link_lin_vel_b
    err = torch.sum((_goal_xy(env, mapped, arc_k) - lin[:, :2]) ** 2, dim=1) + lin[:, 2] ** 2
    return torch.exp(-err / std**2)


def track_vel_coarse(
    env: Any, std: float = 0.31622776601683794, mapped: bool = True, arc_k: float = 0.0
) -> torch.Tensor:
    """Upstream's ``track_linear_velocity`` (σ √0.1), against the mapped goal."""
    return track_vel_sharp(env, std=std, mapped=mapped, arc_k=arc_k)


def track_vel_forward(
    env: Any, gate: float = 0.3, mapped: bool = False, arc_k: float = 0.0
) -> torch.Tensor:
    """Forward progress (m/s) capped at the goal: min(max(v_x, 0), goal_x) while the command's x > gate."""
    cmd, lin = _cmd(env), _robot(env).data.root_link_lin_vel_b
    goal = _goal_x(cmd, mapped, arc_k)
    return torch.minimum(lin[:, 0].clamp(min=0.0), goal.clamp(min=0.0)) * (cmd[:, 0] > gate)


def track_yaw_sharp(env: Any, std: float = 0.25, widen: float = 0.0) -> torch.Tensor:
    """exp(-err²/σ²) on the yaw rate; σ = std + widen·|yaw command| (tight on straight runs, forgiving in arcs
    where v5 reaches only 0.5–0.65 of the command)."""
    cmd, ang = _cmd(env), _robot(env).data.root_link_ang_vel_b
    err = (cmd[:, 2] - ang[:, 2]) ** 2 + torch.sum(ang[:, :2] ** 2, dim=1)
    sigma = std + widen * cmd[:, 2].abs()
    return torch.exp(-err / sigma**2)


def fall_event(env: Any, tilt_deg: float = 60.0) -> torch.Tensor:
    """1 on the step the trunk first tips past ``tilt_deg`` (robotd's fall line is 60°); weight it negative."""
    down = _tilt_deg(env) > tilt_deg
    prev = getattr(env, "_pg_prev_down", None)
    if prev is None:
        prev = torch.zeros_like(down)
    prev = torch.where(
        env.episode_length_buf <= 1, down, prev
    )  # a duck spawned lying down is not a fall event
    event = down & ~prev
    env._pg_prev_down = down
    return event.float()


# ── metrics (Episode_Metrics/…; means over the episode's steps) ─────────────────────────────


def m_fast_n(env: Any) -> torch.Tensor:
    return (_cmd(env)[:, 0] >= 0.8).float()


def m_fast_cmd(env: Any) -> torch.Tensor:
    c = _cmd(env)[:, 0]
    return torch.where(c >= 0.8, c, torch.zeros_like(c))


def m_fast_vx(env: Any) -> torch.Tensor:
    c = _cmd(env)[:, 0]
    return torch.where(c >= 0.8, _robot(env).data.root_link_lin_vel_b[:, 0], torch.zeros_like(c))


def m_ord_n(env: Any) -> torch.Tensor:
    c = _cmd(env)[:, 0]
    return ((c > 0.3) & (c < 0.6)).float()


def m_ord_cmd(env: Any) -> torch.Tensor:
    c = _cmd(env)[:, 0]
    return torch.where((c > 0.3) & (c < 0.6), c, torch.zeros_like(c))


def m_ord_vx(env: Any) -> torch.Tensor:
    c = _cmd(env)[:, 0]
    return torch.where(
        (c > 0.3) & (c < 0.6), _robot(env).data.root_link_lin_vel_b[:, 0], torch.zeros_like(c)
    )


def m_arc_n(env: Any) -> torch.Tensor:
    cmd = _cmd(env)
    return ((cmd[:, 0] >= 0.6) & (cmd[:, 2].abs() > 0.5)).float()


def m_arc_cmd(env: Any) -> torch.Tensor:
    cmd = _cmd(env)
    on = (cmd[:, 0] >= 0.6) & (cmd[:, 2].abs() > 0.5)
    return torch.where(on, cmd[:, 2].abs(), torch.zeros_like(cmd[:, 2]))


def m_arc_yaw(env: Any) -> torch.Tensor:
    cmd = _cmd(env)
    on = (cmd[:, 0] >= 0.6) & (cmd[:, 2].abs() > 0.5)
    got = _robot(env).data.root_link_ang_vel_b[:, 2] * torch.sign(cmd[:, 2])
    return torch.where(on, got, torch.zeros_like(got))


def m_fell60(env: Any) -> torch.Tensor:
    return (_tilt_deg(env) > 60.0).float()


def m_str_n(env: Any) -> torch.Tensor:
    cmd = _cmd(env)
    return ((cmd[:, 0] >= 0.5) & (cmd[:, 2].abs() < 0.15)).float()


def m_str_yawerr(env: Any) -> torch.Tensor:
    """|ω_z − cmd| while commanded straight and fast: the heading drift's source."""
    cmd = _cmd(env)
    on = (cmd[:, 0] >= 0.5) & (cmd[:, 2].abs() < 0.15)
    err = (_robot(env).data.root_link_ang_vel_b[:, 2] - cmd[:, 2]).abs()
    return torch.where(on, err, torch.zeros_like(err))


# ── heading hold (g8): the accumulated drift, not only the yaw-rate error ────────────────────


def _heading_err(env: Any) -> torch.Tensor:
    """Heading minus its reference, rad: the reference integrates the yaw command and is re-anchored when the
    episode starts or any command component moves by more than 0.04 (the EMA's slew after a resample counts as
    movement). Updated once per env step (the reward and the critic's observation both read it)."""
    st = getattr(env, "_pg_head", None)
    if st is None:
        z = torch.zeros(env.num_envs, device=env.device)
        st = {
            "step": -1,
            "ref": z.clone(),
            "prev_cmd": torch.zeros(env.num_envs, 3, device=env.device),
            "err": z.clone(),
        }
        env._pg_head = st
    if st["step"] == env.common_step_counter:
        return st["err"]
    st["step"] = env.common_step_counter
    q = _robot(env).data.root_link_quat_w
    yaw = torch.atan2(
        2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2)
    )
    yaw = torch.nan_to_num(yaw)
    cmd = _cmd(env)
    moved = (cmd - st["prev_cmd"]).abs().amax(dim=1) > 0.04
    fresh = env.episode_length_buf <= 1
    st["ref"] = torch.where(fresh | moved, yaw, st["ref"] + cmd[:, 2] * C.DT)
    st["prev_cmd"] = cmd.clone()
    st["err"] = torch.atan2(torch.sin(yaw - st["ref"]), torch.cos(yaw - st["ref"]))
    return st["err"]


def heading_hold(
    env: Any, sigma: float = 0.25, min_vx: float = 0.3, max_yaw: float = 0.2
) -> torch.Tensor:
    """exp(-ψ_err² / σ²) while commanded forward and (nearly) straight: pays for not drifting."""
    cmd = _cmd(env)
    on = (cmd[:, 0] >= min_vx) & (cmd[:, 2].abs() < max_yaw)
    return torch.exp(-(_heading_err(env) ** 2) / sigma**2) * on


def heading_err_obs(env: Any) -> torch.Tensor:
    """Privileged (critic): the heading error, rad."""
    return _heading_err(env).unsqueeze(1)


def _yaw_lowpass(env: Any, tau_s: float = 0.8) -> torch.Tensor:
    """The yaw rate averaged over ~tau_s (the stride's ±0.4 rad/s waddle averages out, the per-duck bias stays);
    re-anchored at the episode start and when a command component moves. Updated once per env step."""
    st = getattr(env, "_pg_yawlp", None)
    if st is None:
        z = torch.zeros(env.num_envs, device=env.device)
        st = {
            "step": -1,
            "lp": z.clone(),
            "prev_cmd": torch.zeros(env.num_envs, 3, device=env.device),
        }
        env._pg_yawlp = st
    if st["step"] == env.common_step_counter:
        return st["lp"]
    st["step"] = env.common_step_counter
    w = torch.nan_to_num(_robot(env).data.root_link_ang_vel_b[:, 2])
    cmd = _cmd(env)
    moved = (cmd - st["prev_cmd"]).abs().amax(dim=1) > 0.04
    fresh = env.episode_length_buf <= 1
    a = C.DT / tau_s
    st["lp"] = torch.where(fresh | moved, w, st["lp"] + a * (w - st["lp"]))
    st["prev_cmd"] = cmd.clone()
    return st["lp"]


def yaw_mean(
    env: Any, sigma: float = 0.07, tau_s: float = 0.8, min_vx: float = 0.3, max_yaw: float = 0.2
) -> torch.Tensor:
    """exp(-(lowpassed yaw rate − command)² / σ²) while commanded forward and (nearly) straight: the drift's bias,
    which the per-step yaw tracking cannot see under the stride's own yaw oscillation."""
    cmd = _cmd(env)
    on = (cmd[:, 0] >= min_vx) & (cmd[:, 2].abs() < max_yaw)
    return torch.exp(-((_yaw_lowpass(env, tau_s) - cmd[:, 2]) ** 2) / sigma**2) * on


def m_yawlp_abs(env: Any) -> torch.Tensor:
    """|lowpassed yaw rate − command| while commanded fast and straight (the metric twin of yaw_mean)."""
    cmd = _cmd(env)
    on = (cmd[:, 0] >= 0.5) & (cmd[:, 2].abs() < 0.15)
    return torch.where(on, (_yaw_lowpass(env) - cmd[:, 2]).abs(), torch.zeros_like(cmd[:, 2]))
