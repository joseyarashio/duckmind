"""MDP pieces of the Stage-2 walk-network fine-tunes (s2:build): R3 ``pose`` and R4 ``gaze``.

Built on the official VelStand task (``s2_tasks``); what this module adds:

* **The command program** (:class:`S2Program`, driven by :class:`S2TwistCommand` registered as
  ``twist``): per episode one of the programs of :data:`s2_contract.PROGRAMS` — plain standing with
  the fine head, standing robot.pose segments (steps / sines 0.05–0.5 Hz / OU inside the envelope,
  slew-limited as duck-body sends them, released by a ramp or robotd's snap), stop-go (walk →
  stop → pose, entered 0–1.5 s after the stop, i.e. also while still coasting → release → walk),
  v5's own velocity-command distribution, and the neural duck's walking (OU speed, arcs, bursts and
  pauses, the owner's 0.70 flee bursts) with head saccades/scans/offsets inside the A100 envelope.
  Then **robotd's semantics** turn the client's intents into the command block the network sees:
  twist EMA 0.2 (zeroed while pose mode is held), head EMA ``head_alpha`` (1.0 or 0.2 per episode),
  body EMA 0.2 while pose mode is held, snapped to 0 on release. ``head_pose`` / ``body_pose`` are
  views of the same state, so every upstream reward that reads them sees robotd's block.
* **The delay model** (:class:`S2ActionCfg`): robotd's action path of Stage 1 (scale 0.9 ×
  voltage adaptation, low-pass 0.5 / 0.7) plus a per-episode servo command delay of 0–40 ms held
  for the episode, and in a quarter of the episodes the recipe's 15–30 ms redrawn every tick (the
  BAM actuator's delay buffer with ``hold_prob = 1``: only we set its lags).
* **Rewards** (sign convention of Stage 1: functions return ≥ 0; costs get a negative weight):
  R3 — ``pose_track`` (z / roll / pitch toward the v5 stance + robotd's body block), a pose-aware
  ``posture`` (legs toward the kinematic crouch of the commanded z, tolerance widened with the
  commanded lean; replaces upstream's ``pose``), ``upright_pose`` (level relative to the commanded
  lean; replaces ``upright``), ``still_stand``, ``steps_standing``, ``stand_leg_torque`` (leg RMS
  above v5 + 10 %), ``release_rate`` (trunk rate above 40 °/s in the second after a release),
  ``head_fine`` (the head command at σ 0.15 rad, standing only); R4 — ``cam_smear`` (the head
  camera's ⊥ world rate above 10 °/s, walking), ``gaze_yaw`` (the camera's heading against the slow
  heading frame + the yaw command), ``head_torque``; plus upstream's ``neck_action_acceleration_l2``.
* **Metrics** (``Episode_Metrics/…``) and the critic's privileged ``s2_state``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from mjlab.managers import CommandTermCfg
from mjlab.managers.command_manager import CommandTerm
from mjlab_microduck.tasks import mdp as microduck_mdp

from duckmind.skilltrain import rl_contract as C1
from duckmind.skilltrain import s2_contract as C
from duckmind.skilltrain.rl_action import RobotdJointPositionAction, RobotdJointPositionActionCfg

PROG_ID = {name: i for i, name in enumerate(C.PROGRAMS)}
SEG_STAND, SEG_V5, SEG_NEURO, SEG_POSE, SEG_RELEASE, SEG_STOPWAIT, SEG_RESUMEWAIT = range(7)
N_SEG = 7
HEAD_HOME, HEAD_V5, HEAD_PULLBACK, HEAD_MICRO, HEAD_LIVELY, HEAD_STATIC, HEAD_SACC, HEAD_SCAN = (
    range(8)
)
HEAD_KINDS = {"home": HEAD_HOME, "v5": HEAD_V5, "pullback": HEAD_PULLBACK, "micro": HEAD_MICRO,
              "lively": HEAD_LIVELY, "static": HEAD_STATIC, "sacc": HEAD_SACC, "scan": HEAD_SCAN}  # fmt: skip
AX_STEP, AX_SINE, AX_OU, AX_OFF = range(4)
FOREVER = 10**9
LEG_STD_STANDING = (0.1, 0.05, 0.15, 0.15, 0.1) * 2  # velocity recipe std_standing (hip_yaw…ankle)
LEG_STD_WALKING = (0.3, 0.05, 0.4, 0.4, 0.25) * 2
POSTURE_WALKING_THRESHOLD = 0.01  # the recipe's walking_threshold
RELEASE_WINDOW_S = 1.0
RELEASE_FREE_DPS = 40.0
STAND_LEG_RMS_CAP = C.V5["stand_leg_rms_nm"] * 1.10


# ── small maths ─────────────────────────────────────────────────────────────────────────────


def _u(shape: Any, lo: float | torch.Tensor, hi: float | torch.Tensor, device: str) -> torch.Tensor:
    if isinstance(shape, int):
        shape = (shape,)
    return lo + (hi - lo) * torch.rand(*shape, device=device)


def _ticks(seconds: torch.Tensor) -> torch.Tensor:
    return torch.clamp((seconds * C.CONTROL_HZ).round().long(), min=1)


def _wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def _rpy(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    w, x, y, z = q.unbind(-1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin((2 * (w * y - z * x)).clamp(-1.0, 1.0))
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def _choice(weights: tuple[tuple[Any, float], ...], n: int, device: str) -> torch.Tensor:
    p = torch.tensor([w for _, w in weights], device=device)
    return torch.multinomial(p, n, replacement=True)


def _robot(env: Any) -> Any:
    return env.scene["robot"]


def _cos_tilt(env: Any) -> torch.Tensor:
    q = _robot(env).data.root_link_quat_w
    return torch.nan_to_num(1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2), nan=1.0)


def _tilt_deg(env: Any) -> torch.Tensor:
    return torch.rad2deg(torch.acos(_cos_tilt(env).clamp(-1.0, 1.0)))


def _trunk_z(env: Any) -> torch.Tensor:
    r = _robot(env)
    return torch.nan_to_num(r.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2], nan=0.0)


def _servo_ids(env: Any) -> list[int]:
    return microduck_mdp._servo_joint_ids(env, _robot(env))


def _cam(env: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """(the head camera's optical axis, world (N, 3); its world angular velocity (N, 3))."""
    robot = _robot(env)
    cache = env.__dict__.setdefault("_s2_cache", {})
    if "cam" not in cache:
        ids, _ = robot.find_sites([C1.CAMERA_SITE], preserve_order=True)
        cache["cam"] = ids[0]
    q = robot.data.site_quat_w[:, cache["cam"]]
    w, x, y, z = q.unbind(-1)
    axis = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], dim=1)
    omega = torch.nan_to_num(robot.data.site_ang_vel_w[:, cache["cam"]], nan=0.0)
    return axis, omega


# ── the program state ───────────────────────────────────────────────────────────────────────


class S2State:
    """Everything the command program and robotd's smoothing keep per env."""

    def __init__(self, n: int, device: str) -> None:
        z = lambda *s: torch.zeros(*s, device=device)  # noqa: E731
        zl = lambda *s: torch.zeros(*s, dtype=torch.long, device=device)  # noqa: E731
        zb = lambda *s: torch.zeros(*s, dtype=torch.bool, device=device)  # noqa: E731
        self.n, self.device = n, device
        self.program, self.seg, self.seg_left, self.tick = zl(n), zl(n), zl(n), zl(n)
        # the client (duck-body) intents
        self.twist_t, self.head_t, self.pose_t, self.pose_goal = z(n, 3), z(n, 4), z(n, 3), z(n, 3)
        self.pose_on = zb(n)
        # robotd
        self.twist_ema, self.head_ema, self.body_ema, self.twist_eff = (
            z(n, 3),
            z(n, 4),
            z(n, 3),
            z(n, 3),
        )
        self.head_alpha = torch.ones(n, 1, device=device)
        # pose generator (per axis z, roll, pitch)
        self.ax_mode = torch.full((n, 3), AX_OFF, dtype=torch.long, device=device)
        self.ax_amp, self.ax_hz, self.ax_phase = z(n, 3), z(n, 3), z(n, 3)
        self.ax_center, self.ax_level, self.ax_ou, self.ax_tau = (
            z(n, 3),
            z(n, 3),
            z(n, 3),
            torch.ones(n, 3, device=device),
        )
        self.ax_timer = zl(n, 3)
        self.seg_t = zl(n)  # ticks into the current segment
        self.release_ramp = zb(n)
        self.release_ticks = torch.full((n,), FOREVER, dtype=torch.long, device=device)
        self.range_scale = 1.0
        # walking
        self.v5_twist, self.v5_left = z(n, 3), zl(n)
        self.ou_vx, self.ou_wz, self.ou_tau = z(n), z(n), torch.ones(n, device=device)
        self.burst_on, self.burst_left = torch.ones(n, dtype=torch.bool, device=device), zl(n)
        self.flee_left = zl(n)
        # the head
        self.head_kind, self.head_left = zl(n), zl(n)
        self.head_hold = z(n, 4)
        self.head_phase, self.head_hz, self.head_amp = z(n), z(n), z(n)
        self.head_ou = z(n, 4)
        self.sacc_target, self.sacc_left, self.sacc_sign = z(n), zl(n), torch.ones(n, device=device)
        self.pull_on = zb(n)
        # gaze (R4): the slow heading frame
        self.heading_ref, self.yaw_rate_lp, self.prev_yaw = z(n), z(n), z(n)
        self.fresh = torch.ones(n, dtype=torch.bool, device=device)
        self.recent_fall = zl(n)


def state_of(env: Any) -> S2State:
    st = getattr(env, "s2_state", None)
    if st is None:
        st = S2State(env.num_envs, env.device)
        env.s2_state = st
    return st


# ── the command terms ───────────────────────────────────────────────────────────────────────


class S2TwistCommand(CommandTerm):
    """The master term: runs the program and robotd's smoothing each env step; its command is the
    effective twist robotd feeds the walk network."""

    cfg: S2TwistCommandCfg

    def __init__(self, cfg: S2TwistCommandCfg, env: Any) -> None:
        super().__init__(cfg, env)
        self.st = state_of(env)
        self.stage = cfg.stage
        mix = C.PROGRAM_MIX[cfg.stage]
        self._prog_p = torch.tensor([mix[p] for p in C.PROGRAMS], device=self.device)
        self.fixed_program = cfg.fixed_program

    @property
    def command(self) -> torch.Tensor:
        return self.st.twist_eff

    def _update_metrics(self) -> None:
        pass

    # ── reset ──
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        st, dev = self.st, self.device
        ids = env_ids
        m = len(ids)
        if m == 0:
            return
        if self.fixed_program is not None:
            prog = torch.full((m,), PROG_ID[self.fixed_program], dtype=torch.long, device=dev)
        else:
            prog = torch.multinomial(self._prog_p, m, replacement=True)
        st.program[ids] = prog
        alpha = torch.where(torch.rand(m, device=dev) < C.HEAD_ALPHA_MIX[0][1], C.HEAD_ALPHA_MIX[0][0],
                            C.HEAD_ALPHA_MIX[1][0])  # fmt: skip
        st.head_alpha[ids] = alpha.unsqueeze(1)
        for buf in (st.twist_t, st.head_t, st.pose_t, st.pose_goal, st.twist_ema, st.head_ema,
                    st.body_ema, st.twist_eff, st.v5_twist, st.head_hold, st.head_ou):  # fmt: skip
            buf[ids] = 0.0
        st.pose_on[ids] = False
        st.release_ticks[ids] = FOREVER
        st.tick[ids] = 0
        st.fresh[ids] = True
        st.recent_fall[ids] = 0
        st.ax_mode[ids] = AX_OFF
        st.ou_vx[ids] = 0.45
        st.ou_wz[ids] = 0.0
        st.ou_tau[ids] = _u(m, *C.NEURO_WALK["vx_tau_s"], dev)
        st.burst_on[ids] = True
        st.burst_left[ids] = _ticks(_u(m, *C.NEURO_WALK["burst_s"], dev))
        st.flee_left[ids] = 0
        st.v5_left[ids] = 0
        seg = torch.full((m,), SEG_STAND, dtype=torch.long, device=dev)
        left = torch.full((m,), FOREVER, dtype=torch.long, device=dev)
        is_ = lambda p: prog == PROG_ID[p]  # noqa: E731
        seg = torch.where(is_("walk_v5"), SEG_V5, seg)
        seg = torch.where(is_("walk_neuro") | is_("stopgo"), SEG_NEURO, seg)
        left = torch.where(is_("pose"), _ticks(_u(m, 1.0, 2.0, dev)), left)
        left = torch.where(is_("stopgo"), _ticks(_u(m, *C.WALK_SEGMENT_S, dev)), left)
        st.seg[ids] = seg
        st.seg_left[ids] = left
        st.seg_t[ids] = 0
        mask = torch.zeros(st.n, dtype=torch.bool, device=dev)
        mask[ids] = True
        self._new_head(mask)

    # ── one env step ──
    def _update_command(self) -> None:
        st, env = self.st, self._env
        dev = self.device
        st.tick += 1
        st.seg_left -= 1
        st.seg_t += 1
        tilt = _tilt_deg(env)
        st.recent_fall = torch.where(tilt > C.POSE.abort_tilt_deg, torch.full_like(st.recent_fall, C.CONTROL_HZ),
                                     (st.recent_fall - 1).clamp(min=0))  # fmt: skip
        prev_on = st.pose_on.clone()
        self._transitions(tilt)
        self._walk_targets()
        self._pose_targets(tilt)
        self._head_targets()
        # robotd (main.rs order): twist EMA, head EMA, body EMA while pose mode is held
        st.twist_ema += C.CMD_ALPHA * (st.twist_t - st.twist_ema)
        st.head_ema += st.head_alpha * (st.head_t - st.head_ema)
        body_next = st.body_ema + C.CMD_ALPHA * (st.pose_t - st.body_ema)
        st.body_ema = torch.where(st.pose_on.unsqueeze(1), body_next, torch.zeros_like(body_next))
        st.twist_eff = torch.where(
            st.pose_on.unsqueeze(1), torch.zeros_like(st.twist_ema), st.twist_ema
        )
        released = prev_on & ~st.pose_on
        st.release_ticks = torch.where(released, torch.zeros_like(st.release_ticks),
                                       (st.release_ticks + 1).clamp(max=FOREVER))  # fmt: skip
        self._heading()
        del dev

    def _transitions(self, tilt: torch.Tensor) -> None:
        st, dev = self.st, self.device
        prog = st.program
        is_p = lambda p: prog == PROG_ID[p]  # noqa: E731
        # a duck tipping past the guard's limit is released at once (duck-body's rule)
        abort = (st.seg == SEG_POSE) | (st.seg == SEG_RELEASE)
        abort &= tilt > C.POSE.abort_tilt_deg
        if bool(abort.any()):
            st.pose_on = st.pose_on & ~abort
            st.pose_goal[abort] = 0.0
            st.pose_t[abort] = 0.0
            self._enter(abort & is_p("pose"), SEG_STAND, C.STAND_GAP_S)
            self._enter(abort & ~is_p("pose"), SEG_RESUMEWAIT, C.RESUME_WAIT_S)
        # the release ends when the ramp reached zero (or at once for a snap)
        rel = st.seg == SEG_RELEASE
        done_rel = rel & (
            ~st.release_ramp | (st.pose_t.abs().amax(dim=1) < 1e-4) | (st.seg_left <= 0)
        )
        if bool(done_rel.any()):
            st.pose_on = st.pose_on & ~done_rel
            st.pose_t[done_rel] = 0.0
            self._enter(done_rel & is_p("pose"), SEG_STAND, C.STAND_GAP_S)
            self._enter(done_rel & ~is_p("pose"), SEG_RESUMEWAIT, C.RESUME_WAIT_S)
        due = st.seg_left <= 0
        if not bool(due.any()):
            return
        seg = st.seg
        # pose: STAND gap → POSE → RELEASE (→ STAND); stopgo: NEURO → STOPWAIT → POSE → RELEASE → RESUMEWAIT → NEURO
        to_pose = due & (((seg == SEG_STAND) & is_p("pose")) | (seg == SEG_STOPWAIT))
        to_release = due & (seg == SEG_POSE)
        to_stopwait = due & (seg == SEG_NEURO) & is_p("stopgo")
        to_walk = due & (seg == SEG_RESUMEWAIT) & is_p("stopgo")
        if bool(to_pose.any()):
            self._start_pose(to_pose)
        if bool(to_release.any()):
            ramp = torch.rand(st.n, device=dev) < C.RELEASE_RAMP_PROB
            st.release_ramp = torch.where(to_release, ramp, st.release_ramp)
            st.pose_goal[to_release] = 0.0
            snap = to_release & ~ramp
            st.pose_on = st.pose_on & ~snap
            st.pose_t[snap] = 0.0
            self._enter(to_release, SEG_RELEASE, C.RELEASE_RAMP_S)
            # a ramp ends when the target reaches 0 (the rate limit), at the latest after 2 s
            st.seg_left[to_release] = 2 * C.CONTROL_HZ
        self._enter(to_stopwait, SEG_STOPWAIT, C.STOP_WAIT_S)
        self._enter(to_walk, SEG_NEURO, C.WALK_SEGMENT_S)
        if bool((to_stopwait | to_walk).any()):
            self._new_head(to_stopwait | to_walk)

    def _enter(self, mask: torch.Tensor, seg: int, span_s: tuple[float, float]) -> None:
        if not bool(mask.any()):
            return
        st = self.st
        st.seg = torch.where(mask, torch.full_like(st.seg, seg), st.seg)
        st.seg_left = torch.where(mask, _ticks(_u(st.n, *span_s, self.device)), st.seg_left)
        st.seg_t = torch.where(mask, torch.zeros_like(st.seg_t), st.seg_t)
        if seg == SEG_STAND:
            self._new_head(mask)

    # ── robot.pose ──
    def _start_pose(self, mask: torch.Tensor) -> None:
        st, dev, n = self.st, self.device, self.st.n
        self._enter(mask, SEG_POSE, C.POSE_SEGMENT_S)
        lo, hi = self._envelope()
        active = torch.stack(
            [torch.rand(n, device=dev) < C.POSE_AXIS_PROB[a] for a in ("z", "roll", "pitch")], dim=1
        )
        none = ~active.any(dim=1)
        active[:, 1] |= none & (torch.rand(n, device=dev) < 0.5)  # at least one axis mostly
        zero = torch.rand(n, device=dev) < C.POSE_ZERO_PROB
        active &= ~zero.unsqueeze(1)
        mode = _choice(C.POSE_MODES, n * 3, dev).reshape(n, 3)
        mode = torch.where(active, mode, torch.full_like(mode, AX_OFF))
        st.ax_mode = torch.where(mask.unsqueeze(1), mode, st.ax_mode)
        center = torch.where(
            lo.unsqueeze(0).expand(n, 3) < 0, (lo + hi) / 2, torch.zeros(3, device=dev)
        ).expand(n, 3)
        half = ((hi - lo) / 2).expand(n, 3)
        amp = half * _u((n, 3), 0.3, 1.0, dev)
        hz = _u((n, 3), *C.POSE_SINE_HZ, dev)
        phase = _u((n, 3), 0.0, 2 * math.pi, dev)
        tau = _u((n, 3), *C.POSE_OU_TAU_S, dev)
        level = lo + (hi - lo) * torch.rand(n, 3, device=dev)
        m3 = mask.unsqueeze(1)
        st.ax_center = torch.where(m3, center, st.ax_center)
        st.ax_amp = torch.where(m3, amp, st.ax_amp)
        st.ax_hz = torch.where(m3, hz, st.ax_hz)
        st.ax_phase = torch.where(m3, phase, st.ax_phase)
        st.ax_tau = torch.where(m3, tau, st.ax_tau)
        st.ax_level = torch.where(m3, level, st.ax_level)
        st.ax_ou = torch.where(m3, torch.zeros_like(st.ax_ou), st.ax_ou)
        st.ax_timer = torch.where(m3, _ticks(_u((n, 3), *C.POSE_STEP_HOLD_S, dev)), st.ax_timer)
        self._new_head(mask)

    def _envelope(self) -> tuple[torch.Tensor, torch.Tensor]:
        s = float(self.st.range_scale)
        a = C.POSE.angle * s
        lo = torch.tensor([C.POSE.z[0] * s, -a, -a], device=self.device)
        hi = torch.tensor([C.POSE.z[1] * s, a, a], device=self.device)
        return lo, hi

    def _pose_targets(self, tilt: torch.Tensor) -> None:
        st, dev = self.st, self.device
        dt = C.DT
        posing = st.seg == SEG_POSE
        lo, hi = self._envelope()
        t = st.seg_t.float().unsqueeze(1) * dt
        sine = st.ax_center + st.ax_amp * torch.sin(2 * math.pi * st.ax_hz * t + st.ax_phase)
        st.ax_timer -= 1
        redraw = (st.ax_timer <= 0) & (st.ax_mode == AX_STEP)
        new_level = lo + (hi - lo) * torch.rand(st.n, 3, device=dev)
        st.ax_level = torch.where(redraw, new_level, st.ax_level)
        st.ax_timer = torch.where(
            redraw, _ticks(_u((st.n, 3), *C.POSE_STEP_HOLD_S, dev)), st.ax_timer
        )
        half = (hi - lo) / 2
        noise = torch.randn(st.n, 3, device=dev)
        st.ax_ou = (
            st.ax_ou
            - st.ax_ou * dt / st.ax_tau
            + 0.5 * half * torch.sqrt(2 * dt / st.ax_tau) * noise
        )
        ou = (st.ax_center + st.ax_ou).clamp(lo, hi)
        goal = torch.zeros(st.n, 3, device=dev)
        goal = torch.where(st.ax_mode == AX_STEP, st.ax_level, goal)
        goal = torch.where(st.ax_mode == AX_SINE, sine, goal)
        goal = torch.where(st.ax_mode == AX_OU, ou, goal)
        goal = torch.maximum(torch.minimum(goal, hi), lo)
        st.pose_goal = torch.where(posing.unsqueeze(1), goal, torch.where((st.seg == SEG_RELEASE).unsqueeze(1),
                                   torch.zeros_like(goal), st.pose_goal))  # fmt: skip
        rate = (
            torch.tensor([C.POSE.rate_m_s, C.POSE.rate_rad_s, C.POSE.rate_rad_s], device=dev) * dt
        )
        st.pose_t = st.pose_t + (st.pose_goal - st.pose_t).clamp(-rate, rate)
        # pose mode: during POSE when upright and not just fallen (duck-body's guard), and while a
        # ramp release runs; never while walking segments run
        ok = (tilt < C.POSE.enter_tilt_deg) & (st.recent_fall == 0)
        entering = posing & ~st.pose_on
        st.pose_on = torch.where(entering, ok, st.pose_on)
        on_segment = posing | ((st.seg == SEG_RELEASE) & st.release_ramp)
        st.pose_on &= on_segment
        st.pose_t = torch.where(st.pose_on.unsqueeze(1) | (st.seg == SEG_RELEASE).unsqueeze(1), st.pose_t,
                                torch.zeros_like(st.pose_t))  # fmt: skip

    # ── walking ──
    def _walk_targets(self) -> None:
        st, dev, n = self.st, self.device, self.st.n
        dt = C.DT
        v5 = st.seg == SEG_V5
        st.v5_left -= 1
        redraw = v5 & (st.v5_left <= 0)
        if bool(redraw.any()):
            p = C.V5_TWIST
            r = torch.rand(n, device=dev)
            standing = r < p["standing"]
            tip = (r >= p["standing"]) & (r < p["standing"] + p["turn_in_place"])
            vx = _u(n, *p["vx"], dev)
            vy = _u(n, *p["vy"], dev)
            wz = _u(n, *p["wz"], dev)
            tip_wz = torch.sign(_u(n, -1.0, 1.0, dev)) * _u(n, *p["turn_in_place_wz"], dev)
            new = torch.stack([vx, vy, wz], dim=1)
            new = torch.where(standing.unsqueeze(1), torch.zeros_like(new), new)
            new = torch.where(
                tip.unsqueeze(1),
                torch.stack([torch.zeros_like(vx), torch.zeros_like(vx), tip_wz], 1),
                new,
            )
            st.v5_twist = torch.where(redraw.unsqueeze(1), new, st.v5_twist)
            st.v5_left = torch.where(redraw, _ticks(_u(n, *p["resample_s"], dev)), st.v5_left)
        # the neural duck's walk: OU speed and turn, arcs slower, bursts and pauses, flee bursts
        q = C.NEURO_WALK
        noise = torch.randn(n, 2, device=dev)
        st.ou_vx = (
            st.ou_vx
            + (0.45 - st.ou_vx) * dt / st.ou_tau
            + q["vx_sd"] * torch.sqrt(2 * dt / st.ou_tau) * noise[:, 0]
        )
        st.ou_vx = st.ou_vx.clamp(*q["vx"])
        st.ou_wz = (
            st.ou_wz
            - st.ou_wz * dt / q["wz_tau_s"]
            + q["wz_sd"] * math.sqrt(2 * dt / q["wz_tau_s"]) * noise[:, 1]
        )
        st.ou_wz = st.ou_wz.clamp(-q["wz_max"], q["wz_max"])
        neuro = st.seg == SEG_NEURO
        bursty = neuro & (st.program == PROG_ID["walk_neuro"])
        st.burst_left = torch.where(bursty, st.burst_left - 1, st.burst_left)
        flip = bursty & (st.burst_left <= 0)
        if bool(flip.any()):
            st.burst_on = torch.where(flip, ~st.burst_on, st.burst_on)
            span_on = _ticks(_u(n, *q["burst_s"], dev))
            span_off = _ticks(_u(n, *q["pause_s"], dev))
            st.burst_left = torch.where(
                flip, torch.where(st.burst_on, span_on, span_off), st.burst_left
            )
            flee = flip & st.burst_on & (torch.rand(n, device=dev) < q["flee_prob"])
            st.flee_left = torch.where(flee, _ticks(_u(n, *q["flee_s"], dev)), st.flee_left)
            self._new_head(flip)
        st.flee_left = (st.flee_left - 1).clamp(min=0)
        vx = torch.where(
            st.ou_wz.abs() > q["arc_wz"], st.ou_vx.clamp(max=q["arc_vx_max"]), st.ou_vx
        )
        vx = torch.where(st.flee_left > 0, torch.full_like(vx, q["flee_vx"]), vx)
        wz = torch.where(st.flee_left > 0, torch.zeros_like(vx), st.ou_wz)
        walking_now = neuro & (st.burst_on | (st.program != PROG_ID["walk_neuro"]))
        neuro_twist = torch.stack([vx, torch.zeros_like(vx), wz], dim=1)
        twist = torch.zeros(n, 3, device=dev)
        twist = torch.where(v5.unsqueeze(1), st.v5_twist, twist)
        twist = torch.where(walking_now.unsqueeze(1), neuro_twist, twist)
        st.twist_t = twist

    # ── the head ──
    def _new_head(self, mask: torch.Tensor) -> None:
        st, dev, n = self.st, self.device, self.st.n
        if not bool(mask.any()):
            return
        walking = (st.seg == SEG_V5) | (st.seg == SEG_NEURO)
        k_stand = torch.tensor([HEAD_KINDS[k] for k, _ in C.HEAD_MIX_STAND], device=dev)[
            _choice(C.HEAD_MIX_STAND, n, dev)
        ]
        k_walk = torch.tensor([HEAD_KINDS[k] for k, _ in C.HEAD_MIX_WALK], device=dev)[
            _choice(C.HEAD_MIX_WALK, n, dev)
        ]
        kind = torch.where(walking, k_walk, k_stand)
        st.head_kind = torch.where(mask, kind, st.head_kind)
        st.head_left = torch.where(mask, _ticks(_u(n, *C.HEAD_V5_RESAMPLE_S, dev)), st.head_left)
        r = torch.tensor(C.HEAD_V5_RANGES, device=dev)
        v5 = (torch.rand(n, 4, device=dev) * 2 - 1) * r
        # a static walking gaze: a yaw offset, the look-up or the half-down (A100 envelope)
        wh = C.WALK_HEAD
        pick = torch.randint(0, 3, (n,), device=dev)
        yaw = torch.sign(_u(n, -1.0, 1.0, dev)) * _u(n, 0.3, wh["yaw_max"], dev)
        static = torch.zeros(n, 4, device=dev)
        static[:, 2] = torch.where(pick == 0, yaw, 0.0)
        static[:, 0] = torch.where(
            pick == 1, wh["look_up"][0], torch.where(pick == 2, wh["half_down"][0], 0.0)
        )
        static[:, 1] = torch.where(
            pick == 1, wh["look_up"][1], torch.where(pick == 2, wh["half_down"][1], 0.0)
        )
        neck = _u(n, *C.HEAD_PULLBACK["neck"], dev)
        pull = torch.stack([neck, neck * C.HEAD_PULLBACK["head_per_neck"], torch.zeros_like(neck),
                            torch.zeros_like(neck)], dim=1)  # fmt: skip
        hold = torch.where((st.head_kind == HEAD_V5).unsqueeze(1), v5,
                           torch.where((st.head_kind == HEAD_STATIC).unsqueeze(1), static, pull))  # fmt: skip
        st.head_hold = torch.where(mask.unsqueeze(1), hold, st.head_hold)
        st.head_amp = torch.where(mask, _u(n, *C.HEAD_MICRO["amp_rad"], dev), st.head_amp)
        micro_hz = _u(n, *C.HEAD_MICRO["hz"], dev)
        scan_hz = _u(n, *C.WALK_HEAD["scan_hz"], dev)
        st.head_hz = torch.where(
            mask, torch.where(st.head_kind == HEAD_SCAN, scan_hz, micro_hz), st.head_hz
        )
        st.head_phase = torch.where(mask, _u(n, 0.0, 2 * math.pi, dev), st.head_phase)
        st.pull_on = torch.where(mask, torch.rand(n, device=dev) < 0.6, st.pull_on)
        st.sacc_left = torch.where(mask, torch.zeros_like(st.sacc_left), st.sacc_left)

    def _head_targets(self) -> None:
        st, dev, n = self.st, self.device, self.st.n
        dt = C.DT
        st.head_left -= 1
        due = st.head_left <= 0
        # the pull-back alternates hold / release; v5 draws re-sample; others just renew
        if bool(due.any()):
            renew_v5 = due & (st.head_kind == HEAD_V5)
            r = torch.tensor(C.HEAD_V5_RANGES, device=dev)
            st.head_hold = torch.where(
                renew_v5.unsqueeze(1), (torch.rand(n, 4, device=dev) * 2 - 1) * r, st.head_hold
            )
            flip_pull = due & (st.head_kind == HEAD_PULLBACK)
            st.pull_on = torch.where(flip_pull, ~st.pull_on, st.pull_on)
            span = torch.where(
                st.pull_on, _u(n, *C.HEAD_PULLBACK["hold_s"], dev), _u(n, 0.5, 2.0, dev)
            )
            st.head_left = torch.where(due, torch.where(flip_pull, _ticks(span),
                                       _ticks(_u(n, *C.HEAD_V5_RESAMPLE_S, dev))), st.head_left)  # fmt: skip
            # every other head program is re-drawn (kind and parameters) when its time is up
            others = due & ~renew_v5 & ~flip_pull
            if bool(others.any()):
                self._new_head(others)
        t = st.seg_t.float() * dt
        head = torch.zeros(n, 4, device=dev)
        k = st.head_kind
        head = torch.where(((k == HEAD_V5) | (k == HEAD_STATIC)).unsqueeze(1), st.head_hold, head)
        head = torch.where(((k == HEAD_PULLBACK) & st.pull_on).unsqueeze(1), st.head_hold, head)
        micro = st.head_amp * torch.sin(2 * math.pi * st.head_hz * t + st.head_phase)
        micro_axis = (st.head_phase > math.pi).long()  # yaw or head pitch
        mh = torch.zeros(n, 4, device=dev)
        mh[:, 2] = torch.where(micro_axis == 0, micro, 0.0)
        mh[:, 1] = torch.where(micro_axis == 1, micro, 0.0)
        head = torch.where((k == HEAD_MICRO).unsqueeze(1), mh, head)
        scan = torch.zeros(n, 4, device=dev)
        scan[:, 2] = C.WALK_HEAD["scan_amp"] * torch.sin(
            2 * math.pi * st.head_hz * t + st.head_phase
        )
        head = torch.where((k == HEAD_SCAN).unsqueeze(1), scan, head)
        # lively: OU on all four + saccades on the yaw
        noise = torch.randn(n, 4, device=dev)
        sd = torch.tensor([0.1, 0.2, 0.25, 0.12], device=dev)
        lim = torch.tensor([0.25, 0.4, 1.2, 0.3], device=dev)
        st.head_ou = (
            st.head_ou - st.head_ou * dt / 1.5 + sd * math.sqrt(2 * dt / 1.5) * noise
        ).clamp(-lim, lim)
        st.sacc_left -= 1
        fire = st.sacc_left <= 0
        switch = torch.rand(n, device=dev) < 0.7
        st.sacc_sign = torch.where(fire & switch, -st.sacc_sign, st.sacc_sign)
        st.sacc_target = torch.where(
            fire, st.sacc_sign * _u(n, *C.WALK_HEAD["sacc_amp"], dev), st.sacc_target
        )
        st.sacc_left = torch.where(
            fire, _ticks(_u(n, *C.WALK_HEAD["sacc_dwell_s"], dev)), st.sacc_left
        )
        lively = st.head_ou.clone()
        lively[:, 2] = (lively[:, 2] + st.sacc_target).clamp(-1.2, 1.2)
        head = torch.where((k == HEAD_LIVELY).unsqueeze(1), lively, head)
        sacc = torch.zeros(n, 4, device=dev)
        sacc[:, 2] = st.sacc_target
        head = torch.where((k == HEAD_SACC).unsqueeze(1), sacc, head)
        st.head_t = head

    # ── gaze (R4): the slow heading frame ──
    def _heading(self) -> None:
        st = self.st
        _, _, yaw = _rpy(_robot(self._env).data.root_link_quat_w)
        yaw = torch.nan_to_num(yaw)
        st.heading_ref = torch.where(st.fresh, yaw, st.heading_ref)
        st.prev_yaw = torch.where(st.fresh, yaw, st.prev_yaw)
        st.yaw_rate_lp = torch.where(st.fresh, torch.zeros_like(yaw), st.yaw_rate_lp)
        st.fresh[:] = False
        dt = C.DT
        rate = _wrap(yaw - st.prev_yaw) / dt
        st.prev_yaw = yaw
        st.yaw_rate_lp += (dt / C.HEADING_RATE_TAU_S) * (rate - st.yaw_rate_lp)
        st.heading_ref = _wrap(
            st.heading_ref + (st.yaw_rate_lp + _wrap(yaw - st.heading_ref) / C.HEADING_TAU_S) * dt
        )


@dataclass(kw_only=True)
class S2TwistCommandCfg(CommandTermCfg):
    stage: str = "pose"
    fixed_program: str | None = None  # evaluation: every env runs this program
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    def build(self, env: Any) -> S2TwistCommand:
        return S2TwistCommand(self, env)


class _ViewCommand(CommandTerm):
    """A command that only exposes the program state (updated by the twist term)."""

    def __init__(self, cfg: Any, env: Any) -> None:
        super().__init__(cfg, env)
        self.st = state_of(env)

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        pass

    def _update_command(self) -> None:
        pass


class S2HeadCommand(_ViewCommand):
    @property
    def command(self) -> torch.Tensor:
        return self.st.head_ema


class S2BodyCommand(_ViewCommand):
    def __init__(self, cfg: Any, env: Any) -> None:
        super().__init__(cfg, env)
        self._block = torch.zeros(env.num_envs, 6, device=env.device)

    @property
    def command(self) -> torch.Tensor:
        self._block[:, 2:5] = self.st.body_ema
        return self._block


@dataclass(kw_only=True)
class S2HeadCommandCfg(CommandTermCfg):
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    def build(self, env: Any) -> S2HeadCommand:
        return S2HeadCommand(self, env)


@dataclass(kw_only=True)
class S2BodyCommandCfg(CommandTermCfg):
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    def build(self, env: Any) -> S2BodyCommand:
        return S2BodyCommand(self, env)


def set_pose_range(env: Any, env_ids: Any, scale_stages: list[dict]) -> torch.Tensor:
    """Curriculum: the pose envelope's share (0.5 → 1.0)."""
    del env_ids
    value = scale_stages[0]["scale"]
    for s in scale_stages:
        if env.common_step_counter >= s["step"]:
            value = s["scale"]
    state_of(env).range_scale = float(value)
    return torch.tensor(float(value))


# ── robotd's action path + the servo command delay ──────────────────────────────────────────


@dataclass(kw_only=True)
class S2ActionCfg(RobotdJointPositionActionCfg):
    delay_steps: tuple[int, int] = C.DELAY_STEPS
    jitter_prob: float = C.DELAY_JITTER_PROB
    jitter_steps: tuple[int, int] = C.DELAY_JITTER_STEPS
    fixed_lag: int | None = None  # evaluation: every duck at this lag (physics steps)

    def build(self, env: Any) -> S2Action:
        return S2Action(self, env)


class S2Action(RobotdJointPositionAction):
    """Stage 1's robotd action term + a per-episode servo command delay (the BAM actuator's delay
    buffer is built with ``hold_prob = 1``, so its lags change only here)."""

    cfg: S2ActionCfg

    def __init__(self, cfg: S2ActionCfg, env: Any) -> None:
        super().__init__(cfg, env)
        n = self.num_envs
        self.lag = torch.zeros(n, dtype=torch.long, device=self.device)
        self.jitter = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._acts = [
            a for a in self._entity.actuators if getattr(a, "_delay_buffer", None) is not None
        ]

    def _set(self, lags: torch.Tensor, ids: Any) -> None:
        for act in self._acts:
            act.set_lags(lags, ids)

    def reset(self, env_ids: Any = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_ids, slice):
            ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        else:
            ids = env_ids
        m = len(ids)
        if m == 0 or not self._acts:
            return
        cfg = self.cfg
        if cfg.fixed_lag is not None:
            lag = torch.full((m,), int(cfg.fixed_lag), dtype=torch.long, device=self.device)
            jit = torch.zeros(m, dtype=torch.bool, device=self.device)
        else:
            lag = torch.randint(
                cfg.delay_steps[0], cfg.delay_steps[1] + 1, (m,), device=self.device
            )
            jit = torch.rand(m, device=self.device) < cfg.jitter_prob
            jl = torch.randint(
                cfg.jitter_steps[0], cfg.jitter_steps[1] + 1, (m,), device=self.device
            )
            lag = torch.where(jit, jl, lag)
        self.lag[ids] = lag
        self.jitter[ids] = jit
        self._set(lag, ids)

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        if self._acts and bool(self.jitter.any()):
            ids = self.jitter.nonzero().flatten()
            lo, hi = self.cfg.jitter_steps
            lag = torch.randint(lo, hi + 1, (len(ids),), device=self.device)
            self.lag[ids] = lag
            self._set(lag, ids)


# ── gates shared by the rewards ─────────────────────────────────────────────────────────────


def _standing(env: Any, threshold: float = C.STANDING_THRESHOLD) -> torch.Tensor:
    return state_of(env).twist_eff.norm(dim=1) <= threshold


def _walking(env: Any) -> torch.Tensor:
    return state_of(env).twist_eff.norm(dim=1) > C.WALKING_TWIST


def _upright(env: Any, deg: float = 40.0) -> torch.Tensor:
    return _cos_tilt(env) > math.cos(math.radians(deg))


def _targets(env: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(z, roll, pitch) the trunk is asked for: the v5 stance + robotd's body block, standing."""
    st = state_of(env)
    b = st.body_ema
    stand = _standing(env).float()
    z = C.NOMINAL_Z + b[:, 0]
    roll = C.NOMINAL_ROLL + b[:, 1]
    pitch = C.NOMINAL_PITCH * stand + b[:, 2]
    return z, roll, pitch


# ── rewards: R3 ─────────────────────────────────────────────────────────────────────────────


def pose_track(
    env: Any, z_std: float = 0.010, angle_std: float = math.radians(5.0), z_std_coarse: float = 0.0
) -> torch.Tensor:
    """Mean of the z / roll / pitch Gaussians toward the stance + robotd's body block (upstream
    StandUp's z/roll/pitch tracking), standing and upright only. ``z_std_coarse`` > 0 (reward rev p2)
    averages the z Gaussian with a wider one: at σ 10 mm a −20 mm crouch not yet taken scores e⁻⁴ and
    gives no gradient (p1's crouch stayed at 1–2 mm)."""
    z_t, r_t, p_t = _targets(env)
    roll, pitch, _ = _rpy(_robot(env).data.root_link_quat_w)
    ez = _trunk_z(env) - z_t
    rz = torch.exp(-((ez / z_std) ** 2))
    if z_std_coarse > 0:
        rz = 0.5 * (rz + torch.exp(-((ez / z_std_coarse) ** 2)))
    rr = torch.exp(-(((roll - r_t) / angle_std) ** 2))
    rp = torch.exp(-(((pitch - p_t) / angle_std) ** 2))
    return (rz + rr + rp) / 3.0 * _standing(env) * _upright(env)


def pose_composite(
    env: Any, z_std: float = 0.015, angle_std: float = math.radians(8.0)
) -> torch.Tensor:
    """All three at once (a lean that ignores the height, or the reverse, scores little), during
    pose mode only."""
    st = state_of(env)
    z_t, r_t, p_t = _targets(env)
    roll, pitch, _ = _rpy(_robot(env).data.root_link_quat_w)
    s = torch.exp(-(((_trunk_z(env) - z_t) / z_std) ** 2) - ((roll - r_t) / angle_std) ** 2
                  - ((pitch - p_t) / angle_std) ** 2)  # fmt: skip
    return s * st.pose_on * _upright(env)


def _dip_legs(dip: torch.Tensor) -> torch.Tensor:
    """HOME's legs bent (dip > 0) / straightened for a trunk ``dip`` m lower (the kinematic table)."""
    from duckmind.skilltrain.rl_mdp import _table_interp

    left = _table_interp(dip, C1.HOME_DIP_TABLE)
    q = torch.tensor(C.HOME, device=dip.device).repeat(len(dip), 1)
    q[:, list(C1.LEFT_PITCH_CHAIN)] += left
    q[:, list(C1.RIGHT_PITCH_CHAIN)] -= left
    return q


def posture(env: Any, widen: float = 4.0) -> torch.Tensor:
    """upstream's ``variable_posture`` (legs toward HOME, tight when standing), made pose-aware:
    standing, the target is HOME with the legs bent for the commanded z (the kinematic crouch) and
    the tolerance widens with the commanded lean (×(1 + widen·|lean|/0.2))."""
    st = state_of(env)
    cmd = st.twist_eff
    speed = cmd[:, :2].norm(dim=1) + cmd[:, 2].abs()
    standing = speed < POSTURE_WALKING_THRESHOLD
    legs = list(C.LEG_IDX)
    dev = env.device
    q = _robot(env).data.joint_pos[:, _servo_ids(env)][:, legs]
    target = _dip_legs(-st.body_ema[:, 0])[:, legs]
    home = torch.tensor(C.HOME, device=dev)[legs]
    target = torch.where(standing.unsqueeze(1), target, home)
    lean = st.body_ema[:, 1:].abs().amax(dim=1) / C.POSE.angle
    s_std = torch.tensor(LEG_STD_STANDING, device=dev).unsqueeze(0) * (
        1.0 + widen * lean.clamp(0, 1)
    ).unsqueeze(1)
    w_std = torch.tensor(LEG_STD_WALKING, device=dev).unsqueeze(0)
    std = torch.where(standing.unsqueeze(1), s_std, w_std)
    return torch.exp(-torch.mean(((q - target) / std) ** 2, dim=1))


def upright_pose(env: Any, std: float = math.sqrt(0.05)) -> torch.Tensor:
    """upstream's ``upright`` (weight 2, std √0.05) measured against the commanded lean."""
    _, r_t, p_t = _targets(env)
    roll, pitch, _ = _rpy(_robot(env).data.root_link_quat_w)
    err2 = torch.sin(roll - r_t) ** 2 + torch.sin(pitch - p_t) ** 2
    return torch.exp(-torch.nan_to_num(err2, nan=1.0) / std**2)


def still_stand(env: Any, vel_std: float = 0.1) -> torch.Tensor:
    """exp(−v²/σ²) of the trunk's ground speed at zero effective twist (upstream's
    ``stillness_at_zero_command`` on robotd's twist)."""
    v = _robot(env).data.root_link_lin_vel_w[:, :2].norm(dim=1)
    return torch.exp(-((v / vel_std) ** 2)) * _standing(env, 0.01)


def steps_standing(env: Any, air_s: float = 0.05, pushed_mps: float = 0.2) -> torch.Tensor:
    """Touchdowns (after ≥ ``air_s`` of air) at zero effective twist while the trunk is not being
    pushed: stepping in place / walking off. A stumble caught after a push is free."""
    sensor = env.scene.sensors["feet_ground_contact"]
    first = sensor.compute_first_contact(env.step_dt)[:, :2]
    air = sensor.data.last_air_time[:, :2]
    steps = (first & (air >= air_s)).float().sum(dim=1)
    v = _robot(env).data.root_link_lin_vel_w[:, :2].norm(dim=1)
    return steps * _standing(env, 0.01) * (v < pushed_mps) * _upright(env)


def stand_leg_torque(env: Any, cap_nm: float = STAND_LEG_RMS_CAP) -> torch.Tensor:
    """Leg RMS torque above v5 + 10 % (0.047 → 0.052 N·m), plain standing (no pose, not just
    released) — the all-day battery and heat of standing."""
    st = state_of(env)
    tau = torch.nan_to_num(_robot(env).data.actuator_force, nan=0.0)[:, list(C.LEG_IDX)]
    rms = torch.sqrt((tau**2).mean(dim=1))
    plain = _standing(env) & ~st.pose_on & (st.release_ticks > C.CONTROL_HZ) & _upright(env, 20.0)
    return (rms - cap_nm).clamp(min=0.0) * plain


def release_rate(env: Any, free_dps: float = RELEASE_FREE_DPS) -> torch.Tensor:
    """Trunk roll/pitch rate above 40 °/s in the second after pose mode ends (robotd snaps the body
    block to 0; alpha_stand swung back at 79–93 °/s)."""
    st = state_of(env)
    w = _robot(env).data.root_link_ang_vel_b[:, :2].norm(dim=1)
    excess = (torch.nan_to_num(w) - math.radians(free_dps)).clamp(min=0.0)
    window = st.release_ticks < int(RELEASE_WINDOW_S * C.CONTROL_HZ)
    return excess**2 * window * _upright(env)


def head_fine(env: Any, std: float = 0.15, std_coarse: float = 0.0) -> torch.Tensor:
    """The head command at a fine tolerance (σ 0.15 rad), standing only: the neck pull-back and
    ±3° micro-motions get a gradient (upstream found a fine σ while walking stops the walk).
    ``std_coarse`` > 0 (reward rev p2) averages it with a wider Gaussian: the pull-back (neck +0.20…0.35,
    head −⅔ of it) is 0.3–0.4 rad from HOME, where σ 0.15 is flat (p1's neck gain stayed 0)."""
    r = microduck_mdp.head_pose_tracking(env, command_name="head_pose", std=std)
    if std_coarse > 0:
        r = 0.5 * (
            r + microduck_mdp.head_pose_tracking(env, command_name="head_pose", std=std_coarse)
        )
    return r * _standing(env) * _upright(env, 25.0)


# ── rewards: R4 ─────────────────────────────────────────────────────────────────────────────


def cam_perp_rate(env: Any) -> torch.Tensor:
    axis, omega = _cam(env)
    perp = omega - (omega * axis).sum(dim=1, keepdim=True) * axis
    return perp.norm(dim=1)


def cam_smear(env: Any, floor_dps: float = C.CAM_FLOOR_DPS) -> torch.Tensor:
    """(the head camera's world angular rate ⊥ the optical axis − 10 °/s)², walking and upright:
    what smears the image centre, only the escapable part."""
    excess = (cam_perp_rate(env) - math.radians(floor_dps)).clamp(min=0.0)
    return excess**2 * _walking(env) * _upright(env, 30.0)


def gaze_yaw(env: Any, std: float = C.GAZE_YAW_STD) -> torch.Tensor:
    """The camera's heading toward the slow heading frame + the yaw command, walking."""
    st = state_of(env)
    axis, _ = _cam(env)
    cam_yaw = torch.atan2(axis[:, 1], axis[:, 0])
    err = _wrap(cam_yaw - (st.heading_ref + st.head_ema[:, 2]))
    return torch.exp(-((err / std) ** 2)) * _walking(env) * _upright(env, 30.0)


def head_torque(env: Any, free_nm: float = 0.25) -> torch.Tensor:
    tau = torch.nan_to_num(_robot(env).data.actuator_force, nan=0.0).abs()[:, list(C.HEAD_IDX)]
    return (tau - free_nm).clamp(min=0.0).sum(dim=1)


def head_vel(env: Any) -> torch.Tensor:
    """Σ head joint velocity² (rad/s)², walking: v5's own head wobble is half its camera shake on
    BAM (P0: holding the head joints still cut 61.8 → 30.2 °/s at 0.40) — wear and smear alike."""
    qd = torch.nan_to_num(_robot(env).data.joint_vel[:, _servo_ids(env)], nan=0.0)[
        :, list(C.HEAD_IDX)
    ]
    return (qd**2).sum(dim=1) * _walking(env)


def neck_accel(env: Any) -> torch.Tensor:
    """upstream's ``neck_action_acceleration_l2`` (≥ 0), walking only (the stabiliser's wear)."""
    return microduck_mdp.neck_action_acceleration_l2(env) * _walking(env)


# ── observations (the critic's privileged extras) ───────────────────────────────────────────


def s2_state_obs(env: Any) -> torch.Tensor:
    """pose mode, robotd's body block (×10 / ×1 / ×1), the client's goal, seconds since the last
    release (≤ 2), the program and segment one-hots, the gaze yaw error, the camera ⊥ rate (rad/s,
    ÷ 5), the servo delay (steps ÷ 8), head_alpha: 25 dims."""
    st = state_of(env)
    scale = torch.tensor([10.0, 1.0, 1.0], device=env.device)
    prog = torch.nn.functional.one_hot(
        st.program.clamp(0, len(C.PROGRAMS) - 1), len(C.PROGRAMS)
    ).float()
    seg = torch.nn.functional.one_hot(st.seg.clamp(0, N_SEG - 1), N_SEG).float()
    axis, _ = _cam(env)
    cam_yaw = torch.atan2(axis[:, 1], axis[:, 0])
    gaze = _wrap(cam_yaw - (st.heading_ref + st.head_ema[:, 2])).unsqueeze(1)
    rel = (st.release_ticks.float() * C.DT).clamp(max=2.0).unsqueeze(1)
    act = env.action_manager.get_term("joint_pos")
    lag = (
        getattr(act, "lag", torch.zeros(env.num_envs, device=env.device)).float() / 8.0
    ).unsqueeze(1)
    return torch.cat([st.pose_on.float().unsqueeze(1), st.body_ema * scale, st.pose_goal * scale, rel, prog, seg,
                      gaze, (cam_perp_rate(env) / 5.0).unsqueeze(1), lag, st.head_alpha], dim=1)  # fmt: skip


# ── metrics (per-step averages over an episode) ─────────────────────────────────────────────


def m_pose_on(env: Any) -> torch.Tensor:
    return state_of(env).pose_on.float()


def m_pose_err_deg(env: Any) -> torch.Tensor:
    st = state_of(env)
    _, r_t, p_t = _targets(env)
    roll, pitch, _ = _rpy(_robot(env).data.root_link_quat_w)
    err = torch.maximum((roll - r_t).abs(), (pitch - p_t).abs())
    return torch.rad2deg(torch.nan_to_num(err)) * st.pose_on


def m_lean_deg(env: Any) -> torch.Tensor:
    """|achieved roll/pitch − stance| during pose mode (does it lean at all?)."""
    st = state_of(env)
    roll, pitch, _ = _rpy(_robot(env).data.root_link_quat_w)
    lean = torch.maximum(roll.abs(), (pitch - C.NOMINAL_PITCH).abs())
    return torch.rad2deg(torch.nan_to_num(lean)) * st.pose_on


def m_z_err_mm(env: Any) -> torch.Tensor:
    z_t, _, _ = _targets(env)
    return 1000.0 * (_trunk_z(env) - z_t).abs() * state_of(env).pose_on


def m_cam_perp_dps(env: Any) -> torch.Tensor:
    return torch.rad2deg(cam_perp_rate(env)) * _walking(env) * _upright(env, 30.0)


def m_gaze_err_deg(env: Any) -> torch.Tensor:
    st = state_of(env)
    axis, _ = _cam(env)
    err = _wrap(torch.atan2(axis[:, 1], axis[:, 0]) - (st.heading_ref + st.head_ema[:, 2])).abs()
    return torch.rad2deg(err) * _walking(env) * _upright(env, 30.0)


def m_stand_leg_rms(env: Any) -> torch.Tensor:
    tau = torch.nan_to_num(_robot(env).data.actuator_force, nan=0.0)[:, list(C.LEG_IDX)]
    st = state_of(env)
    return torch.sqrt((tau**2).mean(dim=1)) * _standing(env) * ~st.pose_on


def m_steps_standing(env: Any) -> torch.Tensor:
    return steps_standing(env)


def m_fallen(env: Any) -> torch.Tensor:
    return (_cos_tilt(env) < math.cos(math.radians(45.0))).float()


def m_walking(env: Any) -> torch.Tensor:
    return _walking(env).float()


def m_lag_ms(env: Any) -> torch.Tensor:
    act = env.action_manager.get_term("joint_pos")
    return getattr(act, "lag", torch.zeros(env.num_envs, device=env.device)).float() * 5.0


def m_neck_dev(env: Any) -> torch.Tensor:
    """|neck_pitch − HOME − command| standing (the droop / the pull-back)."""
    q = _robot(env).data.joint_pos[:, _servo_ids(env)][:, C.NECK_PITCH]
    cmd = state_of(env).head_ema[:, 0]
    return (q - C.HOME[C.NECK_PITCH] - cmd).abs() * _standing(env)


# Terms whose weight is negative (costs): every Episode_Reward/<cost> must log ≤ 0.
COST_FUNCS = (
    steps_standing,
    stand_leg_torque,
    release_rate,
    cam_smear,
    head_torque,
    head_vel,
    neck_accel,
)

__all__ = [
    "S2Action",
    "S2ActionCfg",
    "S2BodyCommandCfg",
    "S2HeadCommandCfg",
    "S2TwistCommandCfg",
    "S2State",
    "cam_smear",
    "gaze_yaw",
    "pose_track",
    "posture",
    "s2_state_obs",
    "state_of",
    "upright_pose",
]
