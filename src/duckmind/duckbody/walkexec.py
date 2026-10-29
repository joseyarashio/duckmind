"""walkexec: one ``duck.walk`` segment, start to ``walk.done`` (physical-design spec §4.2, §5, §7.1–
§7.3; owner decisions A2, A3, 2026-09-27).

A pure step machine (:class:`WalkExec`, an :class:`~duckmind.duckbody.protocol.ActExecutor`):
each tick the body loop hands it an :class:`~duckmind.duckbody.protocol.ActContext` and sends
what its :class:`~duckmind.duckbody.protocol.ActOutput` asks for through the reflexes and the
forbidden-command guard. It never talks to robotd, the eye or a clock itself.

The segment, in order:

1. **stand** — ``sit_toggle`` until the policy stands (re-sent every 3 s, 6 s at most: the
   livebody pattern);
2. **pre-look** — ``glance_down`` when asked (``pre_look: glance``) or when the rules want it
   (``auto``: :func:`~duckmind.duckbody.lookplan.need_glance`, by the caller's ``attention`` —
   owner 2026-09-28 ~12:30 「保留鴨子偶爾不注意撞到東西的呆萌感」: a duck whose attention is
   elsewhere does not glance), aimed at the route's 0.3 m point; only standing still, legs
   quiet, head still (look only when still);
3. **veto** — :func:`~duckmind.duckbody.lookplan.look_veto` on duck-body's free-cell memory
   and the dark gate. Floor no look has seen free only shortens or slows a walk (owner
   2026-09-28 「3. 一律放寬，即便跌倒才是真實模擬」): an eye that has not learned its floor yet
   sees only 35 cm, so the walk takes one short step at the approach speed into the unseen
   floor and the next attentive walk glances first. A wall or a thing a recent look saw end
   the floor still stops the walk before it (right there: ``veto_look``, turn another way) —
   only what the duck really looked at: floor it did not look at stays unseen, and a duck whose
   attention is not on its way (``low``, ``elsewhere``) walks there as asked, not slowly — it
   does not know what it has not seen — and may bump into what is there. The dark refuses
   and shortens nothing any more (owner 2026-09-28 ~12:30 「2. 放寬」, C2 reversed; the 3 s dark
   segment went at the phase-4 landing, 「一律放寬」): the eye sees less, physics decides;
4. **head** — HOME, settled ≥ 0.3 s (rule 3), or the head-down walking pose for a curiosity
   approach toward a just-seen thing (owner A2);
5. **walk** — the twist streamed every tick (the body loop keeps robotd's 500 ms deadman fed),
   at the speed policy's vx (:func:`speed_policy`, owner A3) until ``max_ms``, a reflex
   (``halted_by``), a body sense (fell, lifted, blocked, stumbled), the deadline or a preempt;
   a head-down walk that makes no progress stops, raises the head, settles and walks on head up
   (:class:`~duckmind.duckbody.lookplan.HeadDownWatch`);
6. **tail** — zero twist for 0.45 s (the legs keep stepping; odometry keeps counting);
7. **end look** — the segment's last frame (head at HOME, still); not when the caller's
   attention is ``elsewhere`` (its gaze is on something else);
8. **walk.done** — odometry in the start's heading frame (dx forward, dy left), path, steps,
   the reason, contact, the two looks folded in, battery, ToF range at the stop (and the
   caller's ``attention`` when it was not ``on``).

**Speed policy (owner A3).** A walk toward a specific thing (``toward`` set) goes at most
:attr:`WalkConfig.approach_vx_mm` (0.40 m/s). Arcs (``|vyaw| ≥`` :attr:`WalkConfig.arc_vyaw_mrad`)
go at :attr:`WalkConfig.arc_vx_mm` (0.40): faster arcs turn much wider (phase 3 A3: the right
arc's radius 0.26 m at 0.40, 0.45 m at 0.50), and Duck Core plans with the tight ones. Other
walks go up to :attr:`WalkConfig.normal_vx_mm`, the measured normal speed. Nothing under
:attr:`WalkConfig.min_vx_mm` (0.35: 0.30 steps on the spot under the XL330 model, phase 2 E2).

**Back step (owner 2026-09-28 「2. 可以」).** A negative ``vx_mm`` asks for a small step back (a
back arc with ``vyaw_mrad``). It goes only when the body loop says the duck is cornered
(``ActContext.cornered``, :mod:`duckmind.duckbody.cornered`: ahead, left and right all shut by
its own senses) — otherwise it ends ``veto_look`` without moving. Granted, it skips the
pre-look and the look veto (the camera cannot see behind; the ground behind is where it came
from), keeps a stricter dark gate's cap if one is given, settles the head at HOME and backs at
−:attr:`WalkConfig.back_vx_min_mm`…−:attr:`WalkConfig.back_vx_max_mm` (−0.30…−0.40 m/s)
until its odometry path plus the reverse's coast reaches :attr:`WalkConfig.back_max_mm`
(0.2 m) or :attr:`WalkConfig.back_max_ms`; the guard stops it at 0.2 m whatever happens.
What is ahead of the duck never halts it (the body loop judges a backing move by the body's own
reflexes only); a contact behind ends it ``stall``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from duckmind.duckbody import lookplan as L
from duckmind.duckbody.fixedmath import SCALE, atan2_mrad, cos_mrad, isqrt, signed_mrad, sin_mrad
from duckmind.duckbody.protocol import ActContext, ActOutput, WalkCmd
from duckmind.duckbody.reflex import A2_WALK_MAX_MM, FRESH_TARGET_MS, HEAD_SETTLE_MS
from duckmind.eye.protocol import EyeLearn
from duckmind.percept import WALK_REASONS

RESULT_KIND = "walk.done"


@dataclass(frozen=True)
class WalkConfig:
    """walkexec's numbers. ``normal_vx_mm`` is the A3 choice (``docs/validation/evidence/phase3/
    a3_walk_speed.json``: the owner's rule — the fastest of 0.45/0.50/0.55 with 0/30 falls — picks
    0.55; its caveats are there); the rest are spec §4.2/§5 and phase 1–2 measurements."""

    normal_vx_mm: int = 550  # A3: 0/150 falls at 0.45, 0.50 and 0.55 in both servo models
    approach_vx_mm: int = 400
    arc_vx_mm: int = 400
    arc_vyaw_mrad: int = 500
    min_vx_mm: int = 350
    vyaw_max_mrad: int = 1000
    stand_retry_ms: int = 3000
    stand_timeout_ms: int = 6000
    tail_ms: int = L.TAIL_MS
    head_settle_ms: int = L.HOME_SETTLE_MS
    head_timeout_ms: int = 3000
    head_down_fallback: bool = True  # owner A2: raise the head when the head-down walk stalls
    head_down: L.HeadDownConfig = L.HeadDownConfig()
    use_look_veto: bool = True
    end_look: bool = True
    learn: bool = True  # send eye.learn (owner D1: the floor learnt from walked-safe ground)
    # owner B3 stops a walk on an unsure lift and claims nothing; wait this long after the stop
    # for the body to know: the floor explains it (a wall or a thing held the trunk still — the
    # walk ends ``stall`` with a contact) or it was a lift (the body loop ends it ``lifted``)
    unsure_resolve_ms: int = 3000
    # the cornered back step (owner 「2. 可以」): −0.30…−0.40 m/s (0.30 is the least that steps;
    # −0.40 is the reversing speed phase 1 measured, 0/120 falls), 0.2 m at most with the coast
    # (a reverse at −0.40 carries on 60–75 mm after the stop: world.dock), and a time cap
    # (velstand spins up backwards in ~1 s)
    back_vx_min_mm: int = 300
    back_vx_max_mm: int = 400
    back_max_mm: int = 200
    back_coast_mm: int = 65
    back_max_ms: int = 2500
    # a reverse start dips the leg load to ~20–23 (the unsure mark is 24: 4 of 48 open-floor
    # starts and 9 of 24 cornered back steps went "unsure", 2026-09-28 rig probe) and spins up in
    # ~1 s: owner B3 says stop and wait to know — the back step holds, and goes on when the floor
    # explains it (at most this many times); a stall before it has backed this far is the
    # spin-up, not a bump behind
    back_resumes: int = 2
    back_block_min_mm: int = 40
    # leaving the dock along its axis (the dock recommendation's step 8): out to here (mm from
    # the wall), steering at a carrot on the axis (PI-less: kp + the gait's drift feed-forward)
    leave_out_mm: int = 1000
    leave_carrot_mm: int = 250
    leave_kp_milli: int = 2500
    leave_ff_mrad: int = 150


DEFAULT = WalkConfig()


def speed_policy(cmd: WalkCmd, config: WalkConfig = DEFAULT) -> tuple[int, int, str]:
    """(vx mm/s, vyaw mrad/s, which rule) for a walk command (owner A3). A negative ``vx_mm`` is
    a back step (rule ``back``: −0.30…−0.40 m/s, walkexec takes it only when cornered); 0 is no
    walk (vx 0)."""
    vyaw = max(-config.vyaw_max_mrad, min(config.vyaw_max_mrad, cmd.vyaw_mrad + cmd.trim_mrad))
    if cmd.vx_mm < 0:
        back = max(config.back_vx_min_mm, min(config.back_vx_max_mm, -cmd.vx_mm))
        return -back, vyaw, "back"
    if cmd.vx_mm == 0:
        return 0, 0, "none"
    if cmd.approach:
        top, rule = config.approach_vx_mm, "approach"
    elif abs(cmd.vyaw_mrad) >= config.arc_vyaw_mrad:
        top, rule = config.arc_vx_mm, "arc"
    else:
        top, rule = config.normal_vx_mm, "normal"
    vx = max(config.min_vx_mm, min(top, cmd.vx_mm))
    return min(vx, top), vyaw, rule


# the reflex / sense that ended a walk → walk.done.reason
_HALT_REASONS = (
    ("unloaded", "reflex_unloaded"),
    ("lift", "lifted"),
    ("fell", "fell"),
    ("fall", "fell"),
    ("drop", "reflex_drop"),
    ("touch", "reflex_touch"),
    ("dark", "reflex_dark"),
    ("stumble", "stumble"),
    ("stall", "stall"),
    ("block", "stall"),
    ("near", "reflex_near"),
    ("deadline", "deadline"),
    ("preempt", "preempt"),
)


AHEAD_REASONS = ("reflex_near", "reflex_touch", "reflex_drop")  # the ToF's, about what is ahead


def reason_of(halted_by: str) -> str:
    """A reflex's name as a ``walk.done.reason``."""
    if halted_by in WALK_REASONS:
        return halted_by
    for needle, reason in _HALT_REASONS:
        if needle in halted_by:
            return reason
    return "reflex_near"


DarkGate = Callable[[ActContext, WalkCmd], tuple[bool, int | None, str]]


def default_dark_gate(ctx: ActContext, cmd: WalkCmd) -> tuple[bool, int | None, str]:
    """Spec §4.5 without the free-cell part (that is :func:`~duckmind.duckbody.lookplan.look_veto`),
    as the owner relaxed it (2026-09-28 ~12:30 「2. 放寬」, and 「一律放寬」 at the phase-4
    landing): no light refuses or caps a walk — dark, black or the eye down only say ``dark``.
    A stricter gate may still refuse (``veto_dark``) or cap the segment."""
    if ctx.light in ("dark", "black", "none"):
        return True, None, "dark"
    return True, None, "ok"


def _pose2(ctx: ActContext) -> L.Pose2 | None:
    if ctx.pose is None:
        return None
    x, y, h = ctx.pose.odom
    return L.Pose2(int(round(x * 1000)), int(round(y * 1000)), int(round(h * 1000)))


def _frame_of(start: L.Pose2, p: L.Pose2) -> tuple[int, int]:
    """``p`` in ``start``'s heading frame (x forward, y left), mm."""
    dx, dy = p.x_mm - start.x_mm, p.y_mm - start.y_mm
    c, s = cos_mrad(start.h_mrad), sin_mrad(start.h_mrad)
    return (dx * c + dy * s) // SCALE, (-dx * s + dy * c) // SCALE


@dataclass
class WalkExec:
    """One ``duck.walk`` act (see the module docstring). ``grid`` is duck-body's shared
    free-cell memory (it outlives the act); ``dark_gate`` is darkgate's ``allow_walk`` when the
    body loop has it."""

    command: WalkCmd
    start_ms: int
    config: WalkConfig = DEFAULT
    grid: L.VetoGrid | None = None
    dark_gate: DarkGate = default_dark_gate
    first_leg: bool = False
    after_arc: bool = False
    set_down: bool = False
    after_contact: bool = False
    after_unseen: bool = False  # the last walk stepped into floor no look had seen free
    last_glance: tuple[int, int] | None = None  # (t_ms, heading mrad) of the last glance_down
    look_prefix: str = ""
    act_id: str = field(init=False, default="")
    # results the body loop may read when the act is done
    percept: dict[str, Any] | None = None
    learn: EyeLearn | None = None
    glance_at: tuple[int, int] | None = None  # (t_ms, heading) if this walk glanced down
    # internal state
    phase: str = "stand"
    phase_ms: int = 0
    vx_mm: int = 0
    vyaw_mrad: int = 0
    rule: str = ""
    max_ms: int = 0
    head_mode: str = "home"
    reason: str | None = None
    contact: dict[str, str] | None = None
    stop_mm: int | None = None
    pre: dict[str, Any] | None = None
    end: dict[str, Any] | None = None
    start_pose: L.Pose2 | None = None
    look_pose: L.Pose2 | None = None
    pre_look_id: str | None = None
    last_pose: L.Pose2 | None = None
    path_mm: int = 0
    path_samples: list[L.Pose2] = field(default_factory=list)
    steps0: int | None = None
    walk_started_ms: int | None = None
    walked_ms: int = 0
    last_move_ms: int | None = None
    toggled_ms: int | None = None
    settle: L.HomeSettle = field(default_factory=L.HomeSettle)
    look: L.LookRun | None = None
    watch: L.HeadDownWatch | None = None
    raised: bool = False
    back_granted: bool = False  # a back step the body's senses allowed (the duck is cornered)
    back_holds: int = 0  # times the back step held for an unsure body (owner B3) and went on
    veto_why: str | None = None  # look_veto's answer for this walk
    leave: Any = None  # the dock's frame (dockexec.DockFrame) for a leave_nest walk from the dock
    leaving: bool = False  # this walk steers out along the dock's axis

    def __post_init__(self) -> None:
        self.act_id = self.command.act_id
        self.phase_ms = self.start_ms
        self.vx_mm, self.vyaw_mrad, self.rule = speed_policy(self.command, self.config)
        self.max_ms = self.command.max_ms
        toward = self.command.toward
        # owner A2's final approach, as the guard allows it: near and just seen
        down_ok = (
            self.command.head == "down"
            and toward is not None
            and toward.r_mm <= A2_WALK_MAX_MM
            and 0 <= self.start_ms - toward.seen_t_ms <= FRESH_TARGET_MS
            and self.rule != "back"
        )
        self.head_mode = "down" if down_ok else "home"

    # ── the step ─────────────────────────────────────────────────────────────

    def step(self, ctx: ActContext) -> ActOutput:
        now = ctx.t_ms
        pose = _pose2(ctx)
        if pose is not None:
            if self.last_pose is not None and self.phase in ("walk", "raise", "tail"):
                self.path_mm += isqrt(
                    (pose.x_mm - self.last_pose.x_mm) ** 2 + (pose.y_mm - self.last_pose.y_mm) ** 2
                )
                if self.phase == "walk" and (
                    not self.path_samples or self.path_samples[-1] != pose
                ):
                    self.path_samples.append(pose)
            self.last_pose = pose
        if self.grid is not None and ctx.pose is not None:
            self.grid.sync(ctx.pose.epoch)
        # a back step's unsure body: hold still until it knows (owner B3), then go on
        if (
            self.rule == "back"
            and self.phase == "walk"
            and (ctx.proprio.unloaded or ctx.halted_by == "reflex_unloaded")
            and not ctx.proprio.lifted
            and self.back_holds < self.config.back_resumes
        ):
            self._pause_walk(now)
            self._to("back_hold", now)
            return ActOutput(twist=None, head=None)
        # things that end any phase
        if self.phase not in ("tail", "end_look", "done", "back_hold"):
            ended = self._body_ends(ctx, now)
            if ended is not None:
                return self._finish(ctx, ended, skip_end_look=ended in ("fell", "lifted"))
        handler = getattr(self, f"_phase_{self.phase}")
        out: ActOutput = handler(ctx, now, pose)
        return out

    def preempt(self, why: str, ctx: ActContext) -> ActOutput:
        reason = why if why in WALK_REASONS else "preempt"
        return self._finish(ctx, reason, skip_end_look=True, immediate=True)

    # ── phases ───────────────────────────────────────────────────────────────

    def _to(self, phase: str, now: int) -> None:
        self.phase = phase
        self.phase_ms = now

    def _phase_stand(self, ctx: ActContext, now: int, pose: L.Pose2 | None) -> ActOutput:
        if self.vx_mm == 0:
            return self._finish(ctx, "stall", skip_end_look=True, immediate=True)
        posture = ctx.proprio.posture
        if posture == "stand" and ctx.proprio.policy in ("walk", "stand", ""):
            self.steps0 = ctx.proprio.steps
            self.start_pose = pose
            if self.rule == "back":
                self._to("back_check", now)
                return self._phase_back_check(ctx, now, pose)
            self._to("pre_look", now)
            return self._phase_pre_look(ctx, now, pose)
        if now - self.phase_ms > self.config.stand_timeout_ms:
            return self._finish(ctx, "stall", skip_end_look=True, immediate=True)
        if self.toggled_ms is None or now - self.toggled_ms >= self.config.stand_retry_ms:
            if posture == "sit":  # never toggle an unknown posture: it may be standing
                self.toggled_ms = now
                return ActOutput(do="sit_toggle")
        return ActOutput()

    def _phase_pre_look(self, ctx: ActContext, now: int, pose: L.Pose2 | None) -> ActOutput:
        cmd = self.command
        if self.look is None:
            heading = pose.h_mrad if pose is not None else 0
            want = cmd.pre_look == "glance" or (
                cmd.pre_look == "auto"
                and L.need_glance(
                    now_ms=now,
                    heading_mrad=heading,
                    last_glance_ms=None if self.last_glance is None else self.last_glance[0],
                    last_glance_heading_mrad=None
                    if self.last_glance is None
                    else self.last_glance[1],
                    light=ctx.light,
                    first_leg=self.first_leg,
                    after_arc=self.after_arc,
                    set_down=self.set_down,
                    after_contact=self.after_contact,
                    after_unseen=self.after_unseen,
                    attention=cmd.attention,
                )
            )
            if not want or ctx.light == "black" and cmd.pre_look != "glance":
                self._to("veto", now)
                return self._phase_veto(ctx, now, pose)
            aim = self._route_point_yaw(pose)
            self.pre_look_id = f"{self.look_prefix}{cmd.act_id}.pre"
            self.look = L.LookRun(
                L.CATALOGUE["glance_down"], self.pre_look_id, now, yaw_mrad=aim, mode="outing"
            )
            self.look_pose = pose
        head, asks, finished = self.look.step(
            now,
            still=bool(ctx.pose and ctx.pose.still),
            legs_quiet=L.legs_quiet(now, self.last_move_ms),
            results=ctx.eye,
        )
        if finished:
            result = self.look.results[-1] if self.look.results else None
            self.pre = L.floor_fold(result)
            if result is not None and self.pre is not None and self.pre.get("light") is not None:
                depth = ctx.depth
                if depth is not None and depth.drop_zones:
                    self.pre["drop"] = min(64, int(depth.drop_zones))
                self._remember_view(
                    result, self.look_pose, now, self.look.heads[-1] if self.look.heads else 0
                )
            if pose is not None:
                self.glance_at = (now, pose.h_mrad)
            self.look = None
            self._to("veto", now)
            return self._phase_veto(ctx, now, pose)
        return ActOutput(head=L.to_radians(head), looks=tuple(asks))

    def _phase_veto(self, ctx: ActContext, now: int, pose: L.Pose2 | None) -> ActOutput:
        ok, cap, why = self.dark_gate(ctx, self.command)
        if not ok:
            return self._finish(ctx, "veto_dark", skip_end_look=True, immediate=True)
        if cap is not None:
            self.max_ms = min(self.max_ms, cap)
        epoch = ctx.pose.epoch if ctx.pose is not None else None
        if self.leave is not None and self.rule != "back" and epoch == self.leave.epoch:
            # out of the dock along its axis: the dock's own lane, which the home keeps clear
            # (the layout's lane rule) — no look veto there; a stricter gate's cap above holds
            self.leaving = True
            self.veto_why = "dock_lane"
            self._to("head", now)
            self.settle = L.HomeSettle()
            return self._phase_head(ctx, now, pose)
        if self.config.use_look_veto and self.grid is not None and pose is not None:
            # attention not on its way (owner 「呆萌感」): it walks as asked into what it did not
            # look at — only what a recent look saw still stops it
            veto = L.look_veto(
                self.grid,
                pose,
                self.vx_mm,
                self.vyaw_mrad,
                self.max_ms,
                now,
                ctx.light,
                careful=self.command.attention == "on",
            )
            self.veto_why = veto.why
            if not veto.ok:
                # a wall or a thing a recent look saw is right there (the only refusal left:
                # the dark refuses nothing since the owner's 「2. 放寬」)
                return self._finish(ctx, "veto_look", skip_end_look=True, immediate=True)
            self.max_ms = veto.max_ms
            if veto.vx_cap_mm is not None and self.vx_mm > veto.vx_cap_mm:
                # into unseen floor: the approach speed (never under the least that steps)
                self.vx_mm = max(self.config.min_vx_mm, veto.vx_cap_mm)
        self._to("head", now)
        self.settle = L.HomeSettle()
        return self._phase_head(ctx, now, pose)

    def _phase_back_check(self, ctx: ActContext, now: int, pose: L.Pose2 | None) -> ActOutput:
        """A back step goes only when the duck is cornered (owner 「2. 可以」); a stricter dark
        gate than the default may still refuse it or cap its segment."""
        if not ctx.cornered:
            return self._finish(ctx, "veto_look", skip_end_look=True, immediate=True)
        ok, cap, _why = self.dark_gate(ctx, self.command)
        if not ok:
            return self._finish(ctx, "veto_dark", skip_end_look=True, immediate=True)
        self.back_granted = True
        self.max_ms = min(self.max_ms, self.config.back_max_ms, cap or self.max_ms)
        self._to("head", now)
        self.settle = L.HomeSettle()
        return self._phase_head(ctx, now, pose)

    def _walk_head(self) -> L.Offsets:
        return L.HEAD_DOWN_WALK if self.head_mode == "down" and not self.raised else L.HOME

    def _head_there(self, ctx: ActContext, target: L.Offsets) -> bool:
        if ctx.pose is None:
            return True  # no tap (a test rig without joints): trust the command
        return L.head_at(ctx.pose.head, L.V5_HOME_HEAD, target) and (
            ctx.pose.head_vel is None or ctx.pose.head_vel < 0.2
        )

    def _guard_settled(self, ctx: ActContext, target: L.Offsets) -> bool:
        """The body loop's guard counts the head settled at HOME (its rule 3) when it says —
        else a first move would be refused and zeroed. A head-down approach is the guard's A2
        case and needs no HOME count."""
        if target != L.HOME or ctx.head_home_ms is None:
            return True
        return ctx.head_home_ms >= HEAD_SETTLE_MS

    def _phase_head(self, ctx: ActContext, now: int, pose: L.Pose2 | None) -> ActOutput:
        target = self._walk_head()
        settled = self.settle.step(now, self._head_there(ctx, target))
        if settled and self._guard_settled(ctx, target):
            self._to("walk", now)
            self.walk_started_ms = now
            if self.head_mode == "down" and not self.raised and pose is not None:
                self.watch = L.HeadDownWatch(self.vx_mm, now, self.config.head_down)
            return self._phase_walk(ctx, now, pose)
        if now - self.phase_ms > self.config.head_timeout_ms:
            if self.head_mode == "down" and not self.raised:
                self.raised = True  # the head never got down: walk head up
                self.settle = L.HomeSettle()
                self.phase_ms = now
            else:
                return self._finish(ctx, "stall", skip_end_look=True, immediate=True)
        return ActOutput(head=None if target == L.HOME else L.to_radians(target))

    def _phase_walk(self, ctx: ActContext, now: int, pose: L.Pose2 | None) -> ActOutput:
        started = self.walk_started_ms if self.walk_started_ms is not None else now
        walked = self.walked_ms + (now - started)
        if walked >= self.max_ms:
            self.walked_ms = walked
            return self._finish(ctx, "time")
        if (
            self.rule == "back"
            and self.path_mm + self.config.back_coast_mm >= self.config.back_max_mm
        ):
            self.walked_ms = walked
            return self._finish(ctx, "time")  # backed far enough (with the coast still to come)
        if (
            self.watch is not None
            and self.start_pose is not None
            and pose is not None
            and not self.raised
        ):
            if self.watch.step(now, self.start_pose, pose):
                if not self.config.head_down_fallback:
                    self.walked_ms = walked
                    return self._finish(ctx, "veto_head")
                self.raised = True
                self.walked_ms = walked
                self.walk_started_ms = None
                self.settle = L.HomeSettle()
                self._to("raise", now)
                return ActOutput(twist=None, head=None)
        self.last_move_ms = now
        head = self._walk_head()
        vyaw = self.vyaw_mrad
        if self.leaving and ctx.pose is not None:
            x, y, h = ctx.pose.odom
            p = self.leave.pose((x * 1000.0, y * 1000.0, h))
            if p.x >= self.config.leave_out_mm:
                self.walked_ms = walked
                return self._finish(ctx, "time")  # out of the dock's lane
            vyaw = self._leave_vyaw(p)
        return ActOutput(
            twist=(self.vx_mm, vyaw), head=None if head == L.HOME else L.to_radians(head)
        )

    def _leave_vyaw(self, p: Any) -> int:
        """Along the dock's axis, facing the room: a carrot on the axis ahead (farther when off
        it), P on the heading error plus the gait's rightward drift feed-forward."""
        cfg = self.config
        reach = max(float(cfg.leave_carrot_mm), 1.5 * abs(p.y))
        target = math.atan2(-p.y, reach)
        err = (target - p.h + math.pi) % (2 * math.pi) - math.pi
        vyaw = cfg.leave_kp_milli / 1000.0 * err + cfg.leave_ff_mrad / 1000.0
        return int(round(max(-1.0, min(1.0, vyaw)) * 1000))

    def _phase_raise(self, ctx: ActContext, now: int, pose: L.Pose2 | None) -> ActOutput:
        """A2 fallback: stopped, head coming up; walk on once it has been HOME 0.3 s."""
        quiet = L.legs_quiet(now, self.last_move_ms)
        settled = self.settle.step(now, self._head_there(ctx, L.HOME))
        if settled and quiet and self._guard_settled(ctx, L.HOME):
            self.walk_started_ms = now
            self._to("walk", now)
            return self._phase_walk(ctx, now, pose)
        if now - self.phase_ms > self.config.head_timeout_ms + self.config.tail_ms:
            return self._finish(ctx, "veto_head")
        return ActOutput(twist=None, head=None)

    def _phase_tail(self, ctx: ActContext, now: int, pose: L.Pose2 | None) -> ActOutput:
        if self.reason == "reflex_unloaded":
            # owner B3: the walk stopped on an unsure lift; wait for the body to know
            if ctx.proprio.unloaded and now - self.phase_ms < self.config.unsure_resolve_ms:
                return ActOutput(twist=None, head=None)
            if not ctx.proprio.unloaded and not ctx.proprio.lifted:
                # the floor explains it: something held the trunk still — a bump
                self.reason = "stall"
                if self.contact is None:
                    self.contact = {"kind": "blocked", "side": "front"}
        if now - self.phase_ms >= self.config.tail_ms:
            self._to("end_look", now)
            if not self._wants_end_look(ctx):
                return self._done(ctx, now)
            self.look = L.LookRun(
                L.CATALOGUE["end_look"], f"{self.look_prefix}{self.act_id}.end", now, mode="outing"
            )
            return self._phase_end_look(ctx, now, pose)
        return ActOutput(twist=None, head=None)

    def _phase_end_look(self, ctx: ActContext, now: int, pose: L.Pose2 | None) -> ActOutput:
        if self.look is None:
            return self._done(ctx, now)
        head, asks, finished = self.look.step(
            now,
            still=bool(ctx.pose and ctx.pose.still),
            legs_quiet=L.legs_quiet(now, self.last_move_ms),
            results=ctx.eye,
        )
        if finished:
            result = self.look.results[-1] if self.look.results else None
            self.end = L.floor_fold(result)
            if result is not None and pose is not None:
                self._remember_view(result, pose, now, 0)
            self.look = None
            return self._done(ctx, now)
        return ActOutput(head=L.to_radians(head) if head != L.HOME else None, looks=tuple(asks))

    def _phase_done(self, ctx: ActContext, now: int, pose: L.Pose2 | None) -> ActOutput:
        return ActOutput(done=True)

    # ── endings ──────────────────────────────────────────────────────────────

    def _body_ends(self, ctx: ActContext, now: int) -> str | None:
        p = ctx.proprio
        if p.fallen:
            return "fell"
        if p.lifted:
            return "lifted"
        if self.command.deadline_ms and now >= self.command.deadline_ms:
            return "deadline"
        if self.phase in ("walk", "raise"):
            back = self.rule == "back"
            spinning_up = back and self.path_mm < self.config.back_block_min_mm
            if ctx.halted_by:
                reason = reason_of(ctx.halted_by)
                # what is ahead never ends a back step (it backs away from it), nor does a
                # "stall" while the reverse is still spinning up
                if not (back and reason in AHEAD_REASONS) and not (
                    spinning_up and reason == "stall"
                ):
                    return reason
            if p.unloaded:
                return "reflex_unloaded"
            if p.stumbled:
                self.contact = {"kind": "stumbled", "side": "unknown" if back else "front"}
                return "stumble"
            if p.blocked and not spinning_up:
                self.contact = {"kind": "blocked", "side": "unknown" if back else "front"}
                return "stall"
        return None

    def _pause_walk(self, now: int) -> None:
        """Bank the walked time (a paused walk keeps its budget) and restart the head settle."""
        if self.walk_started_ms is not None:
            self.walked_ms += now - self.walk_started_ms
        self.walk_started_ms = None
        self.settle = L.HomeSettle()

    def _phase_back_hold(self, ctx: ActContext, now: int, pose: L.Pose2 | None) -> ActOutput:
        """Owner B3 for a back step: unsure only stops. Lifted — the body loop ends the act;
        the floor explains it (not unsure any more) — settle the head and back on; still unsure
        after :attr:`WalkConfig.unsure_resolve_ms` — end ``reflex_unloaded``."""
        if ctx.proprio.lifted or ctx.proprio.fallen:
            return self._finish(ctx, "lifted" if ctx.proprio.lifted else "fell", skip_end_look=True)
        if ctx.proprio.unloaded or ctx.halted_by == "reflex_unloaded":
            if now - self.phase_ms >= self.config.unsure_resolve_ms:
                return self._finish(ctx, "reflex_unloaded")
            return ActOutput(twist=None, head=None)
        self.back_holds += 1
        self._to("head", now)
        return self._phase_head(ctx, now, pose)

    def _wants_end_look(self, ctx: ActContext) -> bool:
        if self.command.attention == "elsewhere":
            return False  # its gaze is on something else (owner 2026-09-28 「呆萌感」)
        return self.config.end_look and self.reason not in ("fell", "lifted", "preempt", "deadline")

    def _finish(
        self, ctx: ActContext, reason: str, *, skip_end_look: bool = False, immediate: bool = False
    ) -> ActOutput:
        """Stop walking: record why, then the tail (unless nothing moved) and the end look."""
        if self.reason is None:
            self.reason = reason
            if ctx.depth is not None:
                self.stop_mm = ctx.depth.ahead_mm
            if ctx.proprio.brushed and self.contact is None:
                self.contact = {"kind": "brushed", "side": "unknown"}
            if reason == "reflex_touch" and self.contact is None:
                self.contact = {"kind": "blocked", "side": "front"}
        now = ctx.t_ms
        self.look = None
        if immediate or self.last_move_ms is None:
            return self._done(ctx, now)
        if skip_end_look:
            self.config = _without_end_look(self.config)
        self._to("tail", now)
        return ActOutput(twist=None, head=None)

    def _done(self, ctx: ActContext, now: int) -> ActOutput:
        if self.phase == "done" and self.percept is not None:
            return ActOutput(done=True)
        self._to("done", now)
        self.percept = self._walk_done(ctx, now)
        self.learn = self._learn()
        return ActOutput(twist=None, head=None, percepts=(self.percept,), done=True)

    def _walk_done(self, ctx: ActContext, now: int) -> dict[str, Any]:
        start = self.start_pose
        end = self.last_pose
        dx = dy = dyaw = 0
        if start is not None and end is not None:
            dx, dy = _frame_of(start, end)
            dyaw = (end.h_mrad - start.h_mrad + 3141) % 6283 - 3141
        steps = 0
        if self.steps0 is not None:
            steps = max(0, ctx.proprio.steps - self.steps0)
        event: dict[str, Any] = {
            "kind": RESULT_KIND,
            "t_ms": now,
            "act_id": self.act_id,
            "cmd": {"vx": self.vx_mm, "vyaw": self.vyaw_mrad, "max_ms": self.max_ms},
            "pre": self.pre,
            "end": self.end,
            "odom": {"dx_mm": dx, "dy_mm": dy, "dyaw_mrad": dyaw, "path_mm": self.path_mm},
            "steps": steps,
            "reason": self.reason or "time",
            "contact": self.contact,
            "battery_pct": ctx.proprio.battery_pct,
            "purpose": self.command.purpose,
            "head": "down" if self.head_mode == "down" and not self.raised else "home",
            "stop_mm": self.stop_mm,
        }
        if self.command.attention != "on":
            event["attention"] = self.command.attention
        return event

    def _learn(self) -> EyeLearn | None:
        """eye.learn for the pre-look's view (owner D1): the walked ground in the frame of the
        pre-look's capture, if the walk went through it."""
        if (
            not self.config.learn
            or self.pre is None
            or self.pre_look_id is None
            or self.look_pose is None
        ):
            return None
        outcome = {
            "time": "clear",
            "reflex_near": "clear",
            "reflex_dark": "clear",
            "veto_head": "clear",
            "stall": "contact",
            "reflex_touch": "contact",
            "stumble": "stumble",
            "fell": "fell",
            "lifted": "lifted",
        }.get(self.reason or "time", "stopped")
        points = []
        for p in self.path_samples[::3][:512]:
            points.append(_frame_of(self.look_pose, p))
        contact = points[-1] if outcome in ("contact", "stumble") and points else None
        try:
            return EyeLearn(
                walk_id=self.act_id,
                look_id=self.pre_look_id,
                outcome=outcome,
                floor=tuple(points),
                width_mm=2 * L.BODY_HALF_MM,
                contact=contact,
            )
        except ValueError:
            return None

    # ── helpers ──────────────────────────────────────────────────────────────

    def _route_point_yaw(self, pose: L.Pose2 | None) -> int:
        """The head yaw that looks at the route's 0.3 m point (spec §4.2 glance_down)."""
        if pose is None or self.vyaw_mrad == 0:
            return 0
        route = L.predict_route(L.Pose2(0, 0, 0), self.vx_mm, self.vyaw_mrad, 3000)
        for p in route:
            if isqrt(p.x_mm * p.x_mm + p.y_mm * p.y_mm) >= 300:
                return signed_mrad(atan2_mrad(p.y_mm, p.x_mm))
        return 0

    def _remember_view(
        self, result: Mapping[str, Any], pose: L.Pose2 | None, now: int, head_yaw: int
    ) -> None:
        if self.grid is None or pose is None:
            return
        level = L.light_of(result)
        self.grid.mark_view(pose, L.view_marks(result, head_yaw), now, level)


def _without_end_look(config: WalkConfig) -> WalkConfig:
    return replace(config, end_look=False)


@dataclass
class WalkMemory:
    """What walkexec keeps between walks (one per body loop): the free-cell grid, the last glance
    (for :func:`~duckmind.duckbody.lookplan.need_glance`) and what the last walk was."""

    grid: L.VetoGrid = field(default_factory=L.VetoGrid)
    last_glance: tuple[int, int] | None = None
    last_was_arc: bool = False
    last_contact: bool = False
    last_unseen: bool = False  # the last walk stepped into floor no look had seen free
    dock: Any = None  # the dock's frame from the last docking that closed PG (dockexec)
    walks: int = 0
    learn: list[EyeLearn] = field(default_factory=list)  # eye.learn lines for the server to send

    def done(self, ex: WalkExec) -> None:
        if ex.glance_at is not None:
            self.last_glance = ex.glance_at
        self.last_was_arc = abs(ex.vyaw_mrad) >= ex.config.arc_vyaw_mrad
        self.last_contact = ex.reason in ("stall", "stumble", "reflex_touch")
        self.last_unseen = ex.veto_why == "unseen" and ex.path_mm > 0
        if ex.path_mm > 0 and not ex.leaving:
            self.dock = None  # walked off somewhere: the dock's frame is for leaving it only
        if ex.learn is not None:
            self.learn.append(ex.learn)
            del self.learn[:-8]


def make_factory(
    config: WalkConfig = DEFAULT,
    memory: WalkMemory | None = None,
    dark_gate: DarkGate | None = None,
) -> Callable[[Any, int, Any], WalkExec]:
    """The body loop's executor factory for ``duck.walk`` (``BodyConfig.executors``): every walk
    shares one :class:`WalkMemory`; ``dark_gate`` is darkgate's ``walk_gate``."""
    mem = memory if memory is not None else WalkMemory()

    def build(cmd: Any, t_ms: int, loop: Any) -> WalkExec:
        mem.walks += 1
        return _Remembered(
            command=cmd,
            start_ms=t_ms,
            config=config,
            grid=mem.grid,
            dark_gate=dark_gate or default_dark_gate,
            first_leg=cmd.purpose in ("leave_nest", "first"),
            after_arc=mem.last_was_arc,
            after_contact=mem.last_contact,
            after_unseen=mem.last_unseen,
            last_glance=mem.last_glance,
            leave=mem.dock if cmd.purpose == "leave_nest" else None,
            memory=mem,
        )

    return build


@dataclass
class _Remembered(WalkExec):
    memory: WalkMemory | None = None

    def _done(self, ctx: ActContext, now: int) -> ActOutput:
        out = super()._done(ctx, now)
        if (
            self.memory is not None
            and self.percept is not None
            and not getattr(self, "_told", False)
        ):
            self._told = True
            self.memory.done(self)
        return out
