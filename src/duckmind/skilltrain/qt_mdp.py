"""MDP pieces of the 2026-10-09 training queue (``qt_*``): command terms with robotd's smoothing, the
tracking / turning / calm-camera / style rewards, and the metrics the quick gate reads.

Built on the panic-gait round's pieces (``pg_mdp``: the robotd twist EMA, the fall-event term) and Stage 2's
(``s2_mdp``: the kinematic crouch, the camera). Sign convention (Stage 1's): reward functions return >= 0;
costs get a negative weight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
from mjlab.managers import CommandTermCfg
from mjlab.managers.command_manager import CommandTerm
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.mdp import VelocityCommandCommandOnly, VelocityCommandCommandOnlyCfg

from duckmind.skilltrain import pg_contract as P
from duckmind.skilltrain import qt_contract as C
from duckmind.skilltrain import s2_mdp as S2M
from duckmind.skilltrain.pg_mdp import fall_event  # noqa: F401  (re-exported for the task builder)
from duckmind.skilltrain.pg_xml import PgAction, PgActionCfg

LEG_STD_STANDING = (
    0.1,
    0.05,
    0.15,
    0.15,
    0.1,
) * 2  # the velocity recipe's std_standing (hip_yaw ... ankle)
LEG_STD_WALKING = (0.3, 0.05, 0.4, 0.4, 0.25) * 2
LEG_STD_RUNNING = (0.35, 0.08, 0.55, 0.55, 0.35) * 2
WALKING_THRESHOLD = 0.01  # the recipe's pose reward: below this |twist| the duck is standing


def _robot(env: Any) -> Any:
    return env.scene["robot"]


def _twist(env: Any) -> torch.Tensor:
    return env.command_manager.get_command("twist")


def _head_cmd(env: Any) -> torch.Tensor:
    return env.command_manager.get_command("head_pose")


def _body_cmd(env: Any) -> torch.Tensor:
    return env.command_manager.get_command("body_pose")


def _upright(env: Any, deg: float = 40.0) -> torch.Tensor:
    return S2M._cos_tilt(env) > math.cos(math.radians(deg))


def _u(n: int, lo: float, hi: float, dev: Any) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(n, device=dev)


# ── robotd's standing tuning ────────────────────────────────────────────────────────────────


@dataclass(kw_only=True)
class QtActionCfg(PgActionCfg):
    """The panic round's action term + robotd's *standing tuning* in a share of the episodes: a network in robotd's
    ``stand`` slot runs at action scale 1.0 and gain ratio 0.8 whenever |twist| <= 0.05 (``control.rs``: the walk slot's 0.9 / 1.0
    otherwise). With ``p_stand_tune`` the standing frames of such an episode get that tuning, so the network works in either slot."""

    p_stand_tune: float = 0.0

    def build(self, env: Any) -> QtAction:
        return QtAction(self, env)


class QtAction(PgAction):
    cfg: QtActionCfg

    def __init__(self, cfg: QtActionCfg, env: Any) -> None:
        super().__init__(cfg, env)
        self.tune = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._kp = [a for a in self._entity.actuators if getattr(a, "kp_scale", None) is not None]

    def reset(self, env_ids: Any = None) -> None:
        super().reset(env_ids)
        if self.cfg.p_stand_tune <= 0:
            return
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None or isinstance(env_ids, slice)
            else env_ids
        )
        if len(ids):
            self.tune[ids] = torch.rand(len(ids), device=self.device) < self.cfg.p_stand_tune

    def process_actions(self, actions: torch.Tensor) -> None:
        if self.cfg.p_stand_tune > 0:
            t = self._env.command_manager.get_command("twist")
            use = self.tune & (t.norm(dim=1) <= C.STANDING_THRESHOLD)
            self._scale_now[:] = torch.where(
                use, 1.0, float(self.cfg.skill_action_scale)
            ).unsqueeze(1)
            ratio = torch.where(use, 0.8, 1.0).unsqueeze(1)
            for act in self._kp:
                act.kp_scale[:] = act.default_kp_scale * ratio
        super().process_actions(actions)

    def m_stand_tuned(self) -> torch.Tensor:
        return self.tune.float()


def m_stand_tuned(env: Any) -> torch.Tensor:
    term = env.action_manager.get_term("joint_pos")
    return (
        term.tune.float() if hasattr(term, "tune") else torch.zeros(env.num_envs, device=env.device)
    )


# ── the twist command ───────────────────────────────────────────────────────────────────────


class QtTwistCommand(VelocityCommandCommandOnly):
    """Upstream's velocity command + a bucket mixture, seen through robotd's twist EMA (or, in a share of the
    episodes, unsmoothed: a skill's constant command)."""

    cfg: QtTwistCommandCfg

    def __init__(self, cfg: QtTwistCommandCfg, env: Any) -> None:
        super().__init__(cfg, env)
        self.ema = torch.zeros(self.num_envs, 3, device=self.device)
        self.alpha = torch.full((self.num_envs,), P.CMD_ALPHA, device=self.device)
        self._probs = torch.tensor([b[0] for b in cfg.mix], device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.ema

    def reset(self, env_ids: Any) -> dict[str, float]:
        out = super().reset(env_ids)
        self.ema[env_ids] = 0.0  # the duck starts standing: robotd's EMA starts from the stand
        inst = torch.rand(len(env_ids), device=self.device) < self.cfg.p_instant
        self.alpha[env_ids] = torch.where(inst, 1.0, P.CMD_ALPHA)
        return out

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        m = len(env_ids)
        if m == 0 or not self.cfg.mix:
            return
        dev = self.device
        cmd = self.vel_command_b
        pick = torch.multinomial(self._probs, m, replacement=True)
        for k, (_, vx0, vx1, vy0, vy1, w0, w1, sym) in enumerate(self.cfg.mix):
            if vx0 is None:  # upstream's own draw stays
                continue
            ids = env_ids[pick == k]
            n = len(ids)
            if n == 0:
                continue
            vx = _u(n, vx0, vx1, dev)
            vy = _u(n, vy0, vy1, dev)
            w = _u(n, w0, w1, dev)
            if sym:
                w = w * torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
            cmd[ids] = torch.stack([vx, vy, w], dim=1)
            self.is_standing_env[ids] = bool(vx1 == 0.0 and w1 == 0.0)
            self.is_forward_env[ids] = False
        self.vel_command_w[env_ids] = cmd[env_ids]

    def _update_command(self) -> None:
        super()._update_command()
        self.ema += self.alpha.unsqueeze(1) * (self.vel_command_b - self.ema)


@dataclass(kw_only=True)
class QtTwistCommandCfg(VelocityCommandCommandOnlyCfg):
    mix: tuple = ()
    p_instant: float = 0.0

    def build(self, env: Any) -> QtTwistCommand:
        return QtTwistCommand(self, env)


# ── the head and body pose commands ─────────────────────────────────────────────────────────


class QtPoseCommand(CommandTerm):
    """A pose command as a client + robotd would deliver it: a target resampled every ~1-3 s inside the ranges
    (with an all-zero share and per-axis off shares), moved by the client at a bounded rate, then smoothed by robotd's
    EMA (``head_alpha`` 1.0 or 0.2 for the head, 0.2 for the body; the body block snaps to 0 when pose mode ends)."""

    cfg: QtPoseCommandCfg

    def __init__(self, cfg: QtPoseCommandCfg, env: Any) -> None:
        super().__init__(cfg, env)
        n, d = self.num_envs, len(cfg.ranges)
        self.dim = d
        self.target = torch.zeros(n, d, device=self.device)
        self.cur = torch.zeros(n, d, device=self.device)
        self.alpha = torch.full((n,), float(cfg.alphas[0][0]), device=self.device)
        self.rate = torch.full((n,), float(cfg.rate[1]), device=self.device)
        self.scale = float(cfg.start_scale)
        lo = torch.tensor([r[0] for r in cfg.ranges], device=self.device)
        hi = torch.tensor([r[1] for r in cfg.ranges], device=self.device)
        self._lo, self._hi = lo, hi
        # per-axis rate weight: metres move at a tenth of a radian's rate
        self._axis_rate = torch.tensor(cfg.axis_rate or (1.0,) * d, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.cur

    def _update_metrics(self) -> None:
        pass

    def reset(self, env_ids: Any) -> dict[str, float]:
        out = super().reset(env_ids)
        self.cur[env_ids] = 0.0
        p = torch.tensor([a[1] for a in self.cfg.alphas], device=self.device)
        vals = torch.tensor([a[0] for a in self.cfg.alphas], device=self.device)
        self.alpha[env_ids] = vals[torch.multinomial(p, len(env_ids), replacement=True)]
        self.rate[env_ids] = _u(len(env_ids), self.cfg.rate[0], self.cfg.rate[1], self.device)
        return out

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        cfg, dev = self.cfg, self.device
        draw = self._lo + (self._hi - self._lo) * torch.rand(n, self.dim, device=dev)
        draw = draw * self.scale
        off = torch.rand(n, self.dim, device=dev) < cfg.p_axis_off
        draw = torch.where(off, torch.zeros_like(draw), draw)
        if cfg.carrier:  # the style code and similar: written as is, not drawn
            draw[:, list(cfg.carrier)] = 0.0
        zero = torch.rand(n, device=dev) < cfg.p_zero
        draw[zero] = 0.0
        self.target[env_ids] = draw
        # leaving pose mode: robotd snaps the body block to 0 (the B-button exit)
        snap = zero & (torch.rand(n, device=dev) < cfg.p_snap)
        if bool(snap.any()):
            self.cur[env_ids[snap]] = 0.0

    def _update_command(self) -> None:
        delta = self.target - self.cur
        step = self.alpha.unsqueeze(1) * delta
        lim = (self.rate.unsqueeze(1) * self._axis_rate.unsqueeze(0)) * P.DT
        self.cur += torch.maximum(torch.minimum(step, lim), -lim)


@dataclass(kw_only=True)
class QtPoseCommandCfg(CommandTermCfg):
    ranges: tuple[tuple[float, float], ...] = ()
    alphas: tuple[tuple[float, float], ...] = ((0.2, 1.0),)  # (alpha, probability)
    rate: tuple[float, float] = (1.0, 4.0)
    axis_rate: tuple[float, ...] | None = None
    p_zero: float = 0.25
    p_axis_off: float = 0.25
    p_snap: float = 0.0
    start_scale: float = 1.0
    carrier: tuple[int, ...] = ()

    def build(self, env: Any) -> QtPoseCommand:
        return QtPoseCommand(self, env)


def set_pose_scale(env: Any, env_ids: Any, stages: list[dict]) -> torch.Tensor:
    """Curriculum: the pose ranges' share by env step (applied to both pose terms)."""
    del env_ids
    value = stages[0]["scale"]
    for s in stages:
        if env.common_step_counter >= s["step"]:
            value = s["scale"]
    for name in ("head_pose", "body_pose"):
        env.command_manager.get_term(name).scale = float(value)
    return torch.tensor(float(value))


class StyleCommand(QtPoseCommand):
    """body_pose with the style code in the x / y carrier slots: one style per episode (resampled with the
    command), the rest of the block (z, roll, pitch) zero. ``cfg.style_probs`` = (normal, sneak, drag, strut)."""

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        probs = torch.tensor(self.cfg.style_probs, device=self.device)
        pick = torch.multinomial(probs, n, replacement=True)
        codes = torch.tensor([C.STYLE_CODE[s] for s in C.STYLES], device=self.device)
        self.target[env_ids] = 0.0
        self.target[env_ids, 0:2] = codes[pick]
        self.style_id[env_ids] = pick

    def __init__(self, cfg: Any, env: Any) -> None:
        super().__init__(cfg, env)
        self.style_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def reset(self, env_ids: Any) -> dict[str, float]:
        out = super().reset(env_ids)
        self.cur[env_ids] = self.target[
            env_ids
        ]  # the style is there from the first step (a skill's baked constant)
        return out

    def _update_command(self) -> None:
        self.cur[:] = self.target


@dataclass(kw_only=True)
class StyleCommandCfg(QtPoseCommandCfg):
    style_probs: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)

    def build(self, env: Any) -> StyleCommand:
        return StyleCommand(self, env)


# ── item 1: tracking the head and body pose commands ────────────────────────────────────────


def _pose_targets(env: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(z, roll, pitch) the trunk is asked for: robot.pose's block, or the walking style's targets (item style)."""
    term = env.command_manager.get_term("body_pose")
    if isinstance(term, StyleCommand):
        tab = torch.tensor(
            [[C.STYLE_TARGETS[s]["z"], C.STYLE_TARGETS[s]["pitch"]] for s in C.STYLES],
            device=env.device,
        )[term.style_id]
        return tab[:, 0], torch.zeros_like(tab[:, 0]), tab[:, 1]
    b = _body_cmd(env)
    # v5 walks with the trunk ~4 mm above its standing height (measured: standing 116.2, walking 117-122 mm): the height asked
    # for while walking is the walk's own, plus robot.pose's z
    t = _twist(env)
    walking = ((t[:, :2].norm(dim=1) + t[:, 2].abs() - 0.05) / 0.25).clamp(0.0, 1.0)
    return b[:, 2] + C.WALK_LIFT_M * walking, b[:, 3], b[:, 4]


def _rp(env: Any) -> tuple[torch.Tensor, torch.Tensor]:
    roll, pitch, _ = S2M._rpy(_robot(env).data.root_link_quat_w)
    return torch.nan_to_num(roll), torch.nan_to_num(pitch)


def _head_err(env: Any) -> torch.Tensor:
    asset = _robot(env)
    microduck_mdp.head_pose_tracking(
        env, command_name="head_pose"
    )  # builds the joint-id cache on first use
    ids = env._head_pose_neck_ids
    jp = asset.data.joint_pos
    measured = jp[:, ids] + jp[:, env._head_pose_bl_ids] * env._head_pose_bl_mask
    return measured - asset.data.default_joint_pos[:, ids] - _head_cmd(env)


def ex_head(
    env: Any,
    std: float = 0.5,
    mid_std: float = 0.2,
    fine_std: float = 0.08,
    w: tuple = (0.3, 0.35, 0.35),
) -> torch.Tensor:
    """Per-joint head tracking, mean over neck / head pitch / yaw / roll of three Gaussians (wide, middle, fine): the
    undershoot of a large command (v5 reaches ~0.8 of it) still has a gradient at sigma 0.2."""
    e = _head_err(env)
    per = (
        w[0] * torch.exp(-((e / std) ** 2))
        + w[1] * torch.exp(-((e / mid_std) ** 2))
        + w[2] * torch.exp(-((e / fine_std) ** 2))
    )
    return per.mean(dim=1) * _upright(env, 45.0)


def ex_neck(env: Any, std: float = 0.15) -> torch.Tensor:
    """The neck and the head pitch (the joints the head's weight sags) at one fine scale: exp(-(e/std)^2) averaged; the
    undershoot of a deep neck command (quick gate: 0.17-0.23 rad) needs its own push beyond ``ex_head``."""
    e = _head_err(env)[:, :2]
    return torch.exp(-((e / std) ** 2)).mean(dim=1) * _upright(env, 45.0)


def ex_trunk(
    env: Any, pitch_std: float = 0.045, z_std: float = 0.007, roll_std: float = 0.06
) -> torch.Tensor:
    """Trunk pitch, height and roll toward robot.pose's block (pitch + = nose down; height = the v5 stance + z)."""
    zt, rt, pt = _pose_targets(env)
    roll, pitch = _rp(env)
    ez = S2M._trunk_z(env) - (C.NOMINAL_Z + zt)
    r = (
        torch.exp(-(((pitch - pt) / pitch_std) ** 2))
        + torch.exp(-((ez / z_std) ** 2))
        + torch.exp(-(((roll - rt) / roll_std) ** 2))
    )
    return r / 3.0 * _upright(env, 40.0)


def ex_upright(env: Any, std: float = math.sqrt(0.05)) -> torch.Tensor:
    """Upstream's ``upright`` measured against the commanded lean."""
    _, rt, pt = _pose_targets(env)
    roll, pitch = _rp(env)
    err2 = torch.sin(roll - rt) ** 2 + torch.sin(pitch - pt) ** 2
    return torch.exp(-torch.nan_to_num(err2, nan=1.0) / std**2)


def _speed_class(env: Any) -> torch.Tensor:
    """0 standing, 1 walking, 2 running: upstream's pose reward gate on |v_xy| + |yaw|."""
    t = _twist(env)
    speed = t[:, :2].norm(dim=1) + t[:, 2].abs()
    return (speed >= WALKING_THRESHOLD).long() + (speed >= P.RUNNING_THRESHOLD).long()


def posture(env: Any, widen_lean: float = 4.0, widen_z: float = 3.0) -> torch.Tensor:
    """Upstream's ``pose`` (legs toward HOME, tolerance by speed class) made pose-aware: the target is HOME with the
    legs bent for the commanded z (the kinematic crouch) and the tolerance widens with the commanded lean and crouch."""
    zt, rt, pt = _pose_targets(env)
    legs = list(C.LEG_IDX)
    dev = env.device
    q = _robot(env).data.joint_pos[:, S2M._servo_ids(env)][:, legs]
    target = S2M._dip_legs(-zt)[:, legs]
    cls = _speed_class(env)
    stds = torch.tensor([LEG_STD_STANDING, LEG_STD_WALKING, LEG_STD_RUNNING], device=dev)[cls]
    lean = (torch.maximum(rt.abs(), pt.abs()) / 0.2).clamp(0, 1)
    crouch = (zt.abs() / 0.02).clamp(0, 1)
    stds = stds * (1.0 + widen_lean * lean + widen_z * crouch).unsqueeze(1)
    return torch.exp(-torch.mean(((q - target) / stds) ** 2, dim=1))


def walk_progress(env: Any, k: float = 0.40, lo: float = 0.3, hi: float = 0.7) -> torch.Tensor:
    """Forward speed (m/s) up to ``k`` x the command (v5 walks at ~0.4 x), while the command is a stroll (lo < vx < hi):
    a pose command must not cost the walk its speed."""
    c = _twist(env)[:, 0]
    vx = torch.nan_to_num(_robot(env).data.root_link_lin_vel_b[:, 0])
    return torch.minimum(vx.clamp(min=0.0), k * c) * ((c > lo) & (c < hi)) * _upright(env, 40.0)


# ── item 2: quick turn ──────────────────────────────────────────────────────────────────────


def turn_rate(env: Any, goal_k: float = C.TURN_GOAL_K, gate: float = 0.4) -> torch.Tensor:
    """Yaw rate in the commanded direction (rad/s), capped at ``goal_k`` x the command, while the yaw command > gate."""
    cmd = _twist(env)
    w = torch.nan_to_num(_robot(env).data.root_link_ang_vel_b[:, 2])
    sgn = torch.sign(cmd[:, 2])
    return torch.minimum((w * sgn).clamp(min=0.0), goal_k * cmd[:, 2].abs()) * (
        cmd[:, 2].abs() > gate
    )


def turn_low(
    env: Any, goal_k: float = C.TURN_GOAL_K, gate: float = 0.35, below: float = 1.1
) -> torch.Tensor:
    """The commanded yaw rate reached, as a share (0..1) of the goal, for the slow commands (gate < |cmd| < below): a
    0.5 rad/s command pays as much as a 2 rad/s one (turn_rate pays the rate itself, so the slow ones gave no push to start)."""
    cmd = _twist(env)
    w = torch.nan_to_num(_robot(env).data.root_link_ang_vel_b[:, 2])
    goal = goal_k * cmd[:, 2].abs()
    got = torch.minimum((w * torch.sign(cmd[:, 2])).clamp(min=0.0), goal)
    on = (cmd[:, 2].abs() > gate) & (cmd[:, 2].abs() < below) & (cmd[:, :2].norm(dim=1) < 0.3)
    return got / goal.clamp(min=0.2) * on


def turn_track(env: Any, std: float = 0.3, widen: float = 0.12) -> torch.Tensor:
    """exp(-(w - command)^2 / sigma^2), sigma widening with the command."""
    cmd = _twist(env)
    w = torch.nan_to_num(_robot(env).data.root_link_ang_vel_b[:, 2])
    sigma = std + widen * cmd[:, 2].abs()
    return torch.exp(-((w - cmd[:, 2]) ** 2) / sigma**2)


def turn_still(env: Any, std: float = 0.06) -> torch.Tensor:
    """exp(-|v_xy|^2 / sigma^2) while commanded to turn without walking: turning on the spot."""
    cmd = _twist(env)
    on = (cmd[:, :2].norm(dim=1) < 0.05) & (cmd[:, 2].abs() > 0.3)
    v = torch.nan_to_num(_robot(env).data.root_link_lin_vel_b[:, :2]).norm(dim=1)
    return torch.exp(-((v / std) ** 2)) * on


# ── item 4: a steady camera ─────────────────────────────────────────────────────────────────


def _head_quiet(env: Any, tol: float = 0.05) -> torch.Tensor:
    return _head_cmd(env).abs().amax(dim=1) < tol


def cam_rate(env: Any) -> torch.Tensor:
    """The head camera's world angular speed (rad/s)."""
    _, omega = S2M._cam(env)
    return torch.nan_to_num(omega).norm(dim=1)


def calm_cam(env: Any, floor_dps: float = 8.0) -> torch.Tensor:
    """(camera world angular speed - floor)^2 (rad/s)^2 while walking with no head command and upright."""
    ex = (cam_rate(env) - math.radians(floor_dps)).clamp(min=0.0)
    t = _twist(env)
    walking = t[:, :2].norm(dim=1) + t[:, 2].abs() > 0.1
    return ex**2 * walking * _head_quiet(env) * _upright(env, 30.0)


def calm_head_vel(env: Any) -> torch.Tensor:
    """Sum of head joint velocity^2 while walking with no head command (wear and smear alike)."""
    qd = torch.nan_to_num(_robot(env).data.joint_vel[:, S2M._servo_ids(env)])[:, list(C.HEAD_IDX)]
    t = _twist(env)
    walking = t[:, :2].norm(dim=1) + t[:, 2].abs() > 0.1
    return (qd**2).sum(dim=1) * walking * _head_quiet(env)


# ── metrics (Episode_Metrics/...: means over the episode's steps) ───────────────────────────


def m_head_err(env: Any) -> torch.Tensor:
    """Mean |head joint error| over the four joints (rad)."""
    return _head_err(env).abs().mean(dim=1) * _upright(env, 45.0)


def m_pitch_err_deg(env: Any) -> torch.Tensor:
    _, pitch = _rp(env)
    return torch.rad2deg((pitch - _pose_targets(env)[2]).abs()) * _upright(env, 45.0)


def m_z_err_mm(env: Any) -> torch.Tensor:
    return (
        (S2M._trunk_z(env) - (C.NOMINAL_Z + _pose_targets(env)[0])).abs()
        * 1000.0
        * _upright(env, 45.0)
    )


def m_pitch_cmd_on(env: Any) -> torch.Tensor:
    return (_body_cmd(env)[:, 4] > 0.05).float()


def m_pitch_got(env: Any) -> torch.Tensor:
    """Trunk pitch (rad) while a bow > 0.05 is commanded (the gain is this / the command, see m_pitch_cmd)."""
    _, pitch = _rp(env)
    on = _body_cmd(env)[:, 4] > 0.05
    return torch.where(on, pitch, torch.zeros_like(pitch))


def m_pitch_cmd(env: Any) -> torch.Tensor:
    b = _body_cmd(env)[:, 4]
    return torch.where(b > 0.05, b, torch.zeros_like(b))


def m_head_cmd_on(env: Any) -> torch.Tensor:
    return (~_head_quiet(env, 0.02)).float()


def m_walk_n(env: Any) -> torch.Tensor:
    c = _twist(env)[:, 0]
    return ((c > 0.3) & (c < 0.6)).float()


def m_walk_cmd(env: Any) -> torch.Tensor:
    c = _twist(env)[:, 0]
    return torch.where((c > 0.3) & (c < 0.6), c, torch.zeros_like(c))


def m_walk_vx(env: Any) -> torch.Tensor:
    c = _twist(env)[:, 0]
    return torch.where(
        (c > 0.3) & (c < 0.6), _robot(env).data.root_link_lin_vel_b[:, 0], torch.zeros_like(c)
    )


def m_walk_zero_pose_n(env: Any) -> torch.Tensor:
    c = _twist(env)[:, 0]
    quiet = _head_quiet(env, 0.02) & (_body_cmd(env).abs().amax(dim=1) < 0.003)
    return ((c > 0.3) & (c < 0.6) & quiet).float()


def m_walk_zero_pose_vx(env: Any) -> torch.Tensor:
    c = _twist(env)[:, 0]
    quiet = _head_quiet(env, 0.02) & (_body_cmd(env).abs().amax(dim=1) < 0.003)
    on = (c > 0.3) & (c < 0.6) & quiet
    return torch.where(on, _robot(env).data.root_link_lin_vel_b[:, 0], torch.zeros_like(c))


def m_walk_zero_pose_cmd(env: Any) -> torch.Tensor:
    c = _twist(env)[:, 0]
    quiet = _head_quiet(env, 0.02) & (_body_cmd(env).abs().amax(dim=1) < 0.003)
    return torch.where((c > 0.3) & (c < 0.6) & quiet, c, torch.zeros_like(c))


def m_fell60(env: Any) -> torch.Tensor:
    return (S2M._tilt_deg(env) > 60.0).float()


# turn
def m_turn_n(env: Any) -> torch.Tensor:
    c = _twist(env)
    return ((c[:, :2].norm(dim=1) < 0.05) & (c[:, 2].abs() > 0.4)).float()


def m_turn_cmd(env: Any) -> torch.Tensor:
    c = _twist(env)
    on = (c[:, :2].norm(dim=1) < 0.05) & (c[:, 2].abs() > 0.4)
    return torch.where(on, c[:, 2].abs(), torch.zeros_like(c[:, 2]))


def m_turn_rate(env: Any) -> torch.Tensor:
    c = _twist(env)
    on = (c[:, :2].norm(dim=1) < 0.05) & (c[:, 2].abs() > 0.4)
    w = torch.nan_to_num(_robot(env).data.root_link_ang_vel_b[:, 2]) * torch.sign(c[:, 2])
    return torch.where(on, w, torch.zeros_like(w))


def m_turn_drift(env: Any) -> torch.Tensor:
    """Trunk ground speed (m/s) while turning on the spot."""
    c = _twist(env)
    on = (c[:, :2].norm(dim=1) < 0.05) & (c[:, 2].abs() > 0.4)
    v = torch.nan_to_num(_robot(env).data.root_link_lin_vel_b[:, :2]).norm(dim=1)
    return torch.where(on, v, torch.zeros_like(v))


# calm
def m_cam_dps(env: Any) -> torch.Tensor:
    t = _twist(env)
    on = (t[:, :2].norm(dim=1) + t[:, 2].abs() > 0.1) & _head_quiet(env)
    return torch.where(on, torch.rad2deg(cam_rate(env)), torch.zeros_like(t[:, 0]))


def m_cam_n(env: Any) -> torch.Tensor:
    t = _twist(env)
    return ((t[:, :2].norm(dim=1) + t[:, 2].abs() > 0.1) & _head_quiet(env)).float()


__all__ = [
    "QtPoseCommandCfg",
    "QtTwistCommandCfg",
    "StyleCommandCfg",
    "calm_cam",
    "calm_head_vel",
    "ex_head",
    "ex_neck",
    "ex_trunk",
    "ex_upright",
    "fall_event",
    "posture",
    "set_pose_scale",
    "turn_low",
    "turn_rate",
    "turn_still",
    "turn_track",
    "walk_progress",
]
_ = field
