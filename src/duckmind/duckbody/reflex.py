"""Reflexes and the forbidden-command guard (spec §5.3; owner decisions A1, A2, A3, B3).

**Reflexes** (:func:`verdict`) — pure integer functions (mm, mrad, ms, ppm) so HomeWorld can call
the same ones (spec §14.2). Each tick the body loop asks whether the body must hold still; the
answer is the ``walk.done.reason`` of the first reflex that holds, in this order:

``fell`` → ``lifted`` → ``reflex_unloaded`` → ``reflex_drop`` → ``reflex_touch`` → ``stall`` →
``reflex_near`` → ``reflex_dark`` (a dark segment over its cap).

* ``stop_near`` — the ToF straight ahead at or under the walk's ``stop_ahead_mm`` (default
  300). **Owner A1**: relaxed *only* in a curiosity act walking toward the thing it has just
  seen (:func:`relaxed_near`): purpose ``curio``, a ``toward`` seen at most
  :data:`FRESH_TARGET_MS` ago, the return where the target should be (bearing within
  :data:`TARGET_BEARING_MRAD`, range consistent with the target's) — then the stop distance is the
  act's own standoff (never under :data:`MIN_NEAR_MM`). Every other reflex is unchanged.
* ``no_drop`` — drop zones ahead ≥ 4; ``touch`` — the ToF touch; ``stall`` — proprio's blocked.
* ``lifted``/``unloaded`` — proprio's two lift stages (owner B3: an unsure lift stops the walk
  but claims nothing).
* a move that backs away (``reversing``: a dock's reverse leg, a cornered back step) is judged by
  the body's own reflexes only — the ToF looks ahead, away from where it goes.

**Forbidden commands** (:func:`check_intent`) — the one chokepoint every robotd intent passes;
a property test drives it with random intents and states. Refused (reason returned):

* any method outside the allowlist (no servo, gain, torque, policy, reboot, enable…);
* turning in place: ``vx = 0`` with ``vyaw ≠ 0`` (6 s turns only 0–11°);
* ``0 < |vx| < 0.30`` (velstand does not step);
* ``|vx|`` over the cap: :attr:`Limits.vx_max_mm` for a normal walk (owner A3: the fastest of
  0.45/0.50/0.55 with 0/30 falls — 0.55, measured by body-motion, ``a3_walk_speed.json``) and
  :data:`~duckmind.duckbody.protocol.APPROACH_VX_MM` (0.40) when approaching a specific thing;
* ``|vyaw| > 1.0``;
* ``vx < 0`` except (a) the curiosity ``back_arc`` inside a curiosity act, (b) inside a
  ``duck.dock`` act — **owner 2026-09-28 「1. 可以」**: both dock styles reverse (phase 1:
  −0.40 m/s reversing 0/120 falls), docking only — and (c) a small back step or back arc when
  the duck is cornered — **owner 「2. 可以」** (phase-3 verify finding (f): within 30 cm of a
  small thing no move was allowed and the duck stayed stuck): the body loop sets
  :attr:`GuardContext.back_step_ok` only while a ``duck.walk`` back step its own senses granted
  (:mod:`duckmind.duckbody.cornered`) has moved less than :data:`BACK_STEP_MAX_MM`; never
  faster than −0.40 (:data:`REVERSE_VX_MAX_MM`: the only reversing speed measured) and never
  slower than the −0.30 every walk needs;
* rule 3 (spec §4.2: "*before* a walk the head is back at HOME and has stayed there ≥ 0.3 s"):
  a move that *starts* a walk (no twist streaming yet) while the head is not at HOME and settled
  ≥ 0.3 s; once the walk streams, a refresh while the head is commanded away from HOME or the
  measured head is more than :data:`HEAD_MOVING_MRAD` off it (a walking duck's head swings with
  the gait — up to 0.40 rad at 0.55 m/s in the A3 traces, 1,200 walks — so judging the 0.3 s
  settle on every 50 Hz refresh zeroed nearly every walk, phase-3 review) — **owner A2**: except
  the final approach toward a just-seen thing, which may walk head down (at approach speed);
* a non-zero move while lifted, unsure, fallen, righting or capturing a frame;
* ``robot.do`` outside ``sit_toggle``/``ground_pick``/``kick_left``/``kick_right``, anything
  but ``sit_toggle`` outside a curiosity act, and the skills while lifted, unsure (maybe being
  picked up), fallen or righting;
* ``robot.look`` (it takes absolute targets; not a gameplay primitive yet, ``robotd.py``);
* on the world port, any op but ``pg`` (:func:`check_world_op`).

A zero ``robot.move`` and ``robot.stop`` are always allowed: stopping is never forbidden.
"""

from __future__ import annotations

from dataclasses import dataclass

from duckmind.duckbody.protocol import (
    APPROACH_VX_MM,
    MIN_STEP_VX_MM,
    NORMAL_VX_MAX_MM,
    STOP_AHEAD_MM,
    VYAW_MAX_MRAD,
    DepthView,
    Intent,
    ProprioView,
    WalkCmd,
)
from duckmind.duckbody.robotd_client import ALLOWED_METHODS, FORBIDDEN_METHODS

DROP_ZONES = 4
FRESH_TARGET_MS = 3000  # "just seen": the target's look is at most this old (A1)
TARGET_BEARING_MRAD = 250  # the ToF return lies within this of where the target should be
TARGET_RANGE_SLACK_MM = 150
MIN_NEAR_MM = 50  # A1 never lets the ToF stop go under this
HEAD_SETTLE_MS = 300
# Once a walk streams, the measured head may swing this far from HOME with the gait (the A3
# traces' largest swing: 0.396 rad, simulator servos, 0.55 m/s right arc; XL330 model ≤ 0.25).
HEAD_MOVING_MRAD = 450
REVERSE_VX_MAX_MM = 400  # the only reversing speed measured (phase 1: −0.40, 0/120 falls)
# owner 「2. 可以」: a cornered duck may back off this far at most (the command and the coast
# together); past it the guard refuses every further reverse twist of that step
BACK_STEP_MAX_MM = 200
# Owner A2 is the *final* approach: a plain duck.walk may go head down only toward a target this
# near, seen at most FRESH_TARGET_MS before the walk began (a curiosity step says it itself).
A2_WALK_MAX_MM = 600
ROBOT_DO_SKILLS = ("sit_toggle", "ground_pick", "kick_left", "kick_right")
WORLD_OPS = ("pg",)


@dataclass(frozen=True)
class Limits:
    """Speed caps in mm/s and mrad/s. ``vx_max_mm`` is the normal walking cap (owner A3)."""

    vx_max_mm: int = NORMAL_VX_MAX_MM
    approach_vx_mm: int = APPROACH_VX_MM
    min_step_vx_mm: int = MIN_STEP_VX_MM
    vyaw_max_mrad: int = VYAW_MAX_MRAD


# ── reflexes ──────────────────────────────────────────────────────────────────


def stop_near(ahead_mm: int | None, stop_ahead_mm: int = STOP_AHEAD_MM) -> bool:
    return ahead_mm is not None and ahead_mm <= stop_ahead_mm


def no_drop(drop_zones: int) -> bool:
    return drop_zones >= DROP_ZONES


def relaxed_near(
    cmd: WalkCmd | None,
    depth: DepthView | None,
    now_ms: int,
    walked_mm: int = 0,
) -> int | None:
    """Owner A1: the ToF stop distance for this walk when it may be relaxed, else ``None``.

    Relaxed only when *all* hold: a curiosity walk (``purpose == "curio"``) toward a target it
    saw at most :data:`FRESH_TARGET_MS` ago; the nearest ahead return within
    :data:`TARGET_BEARING_MRAD` of the target's bearing; and its range consistent with the
    target's (not farther than it was, not much nearer than it should be after ``walked_mm``)."""
    if cmd is None or depth is None or cmd.purpose != "curio" or cmd.toward is None:
        return None
    target = cmd.toward
    if now_ms - target.seen_t_ms > FRESH_TARGET_MS or now_ms < target.seen_t_ms:
        return None
    if depth.ahead_mm is None:
        return None
    if abs(depth.ahead_b_mrad - target.b_mrad) > TARGET_BEARING_MRAD:
        return None
    expected = max(0, target.r_mm - max(0, walked_mm))
    if depth.ahead_mm > target.r_mm + TARGET_RANGE_SLACK_MM:
        return None
    if depth.ahead_mm < expected - TARGET_RANGE_SLACK_MM:
        return None
    return max(MIN_NEAR_MM, min(cmd.stop_ahead_mm, STOP_AHEAD_MM))


@dataclass(frozen=True)
class Verdict:
    """``halt``: the body must not move this tick; ``reason``: the walk.done reason."""

    halt: bool
    reason: str | None = None
    relaxed_mm: int | None = None


CLEAR = Verdict(False)


def verdict(
    *,
    proprio: ProprioView,
    depth: DepthView | None,
    cmd: WalkCmd | None,
    now_ms: int,
    walked_mm: int = 0,
    touch_counts: bool = True,
    dark_over_cap: bool = False,
    relax: bool = True,
    reversing: bool = False,
) -> Verdict:
    """The first reflex that holds (module doc order), or :data:`CLEAR`. ``relax`` False: the
    running act is not a curiosity act, so owner A1 does not apply whatever ``cmd`` says (a plain
    ``duck.walk`` naming purpose ``curio`` gets no relaxed stop). ``reversing``: the move backs
    away — the ToF looks ahead, so its drop, touch and near stops say nothing about where a
    backing duck goes (a cornered back step, a dock's reverse leg); the body's own reflexes
    (fell, lifted, unsure, stall) still hold."""
    if proprio.fallen:
        return Verdict(True, "fell")
    if proprio.lifted:
        return Verdict(True, "lifted")
    if proprio.unloaded:
        return Verdict(True, "reflex_unloaded")
    if reversing:
        if proprio.blocked:
            return Verdict(True, "stall")
        return Verdict(True, "reflex_dark") if dark_over_cap else CLEAR
    if depth is not None:
        if no_drop(depth.drop_zones):
            return Verdict(True, "reflex_drop")
        if depth.touch and touch_counts:
            return Verdict(True, "reflex_touch")  # A1 relaxes the 30 cm stop only
    if proprio.blocked:
        return Verdict(True, "stall")
    if depth is not None:
        stop_at = STOP_AHEAD_MM if cmd is None else cmd.stop_ahead_mm
        relaxed = relaxed_near(cmd, depth, now_ms, walked_mm) if relax else None
        if relaxed is not None:
            if stop_near(depth.ahead_mm, relaxed):
                return Verdict(True, "reflex_near", relaxed)
        elif stop_near(depth.ahead_mm, max(stop_at, STOP_AHEAD_MM)):
            return Verdict(True, "reflex_near")
    if dark_over_cap:
        return Verdict(True, "reflex_dark")
    return CLEAR


# ── the forbidden-command guard ───────────────────────────────────────────────


@dataclass(frozen=True)
class GuardContext:
    """What the guard needs to know about the body and the act this tick."""

    head_home_ms: int = 0  # how long the measured head has been at HOME (0: not at HOME)
    act: str | None = None  # the running act's command (duck.walk, duck.curio, …)
    approach: bool = False  # a walk toward a specific, just-seen thing (A2/A3)
    head_down_ok: bool = False  # this walk may go head down (A2: the final approach)
    back_arc_ok: bool = False  # curiosity's back_arc (reversing measured 0/120 falls)
    # a duck.walk back step granted because the duck is cornered (owner 「2. 可以」), still
    # under BACK_STEP_MAX_MM of backing (the body loop measures it by odometry)
    back_step_ok: bool = False
    lifted: bool = False
    unloaded: bool = False
    fallen: bool = False
    righting: bool = False
    capturing: bool = False
    moving: bool = False  # a non-zero twist already streams (rule 3 was met when it started)
    head_cmd_home: bool = True  # the head is commanded HOME (no look, no head-down pose)
    head_off_mrad: int = 0  # the measured head's largest joint offset from HOME
    # Stage-1 trained one-shot skills (tr:harness wiring): the names this robotd loaded that
    # duck-body's skill guard (duckmind.skilltrain.guard) let start this tick; empty = none
    trained_skills: frozenset[str] = frozenset()


def _mm(value: object) -> int:
    return int(round(float(value) * 1000.0)) if isinstance(value, (int, float)) else 0


def reverse_allowed(ctx: GuardContext) -> bool:
    """May this act reverse at all (the speed caps apart)? Curiosity's back_arc, a ``duck.dock``
    act (owner 2026-09-28 「1. 可以」) or a granted cornered back step (owner 「2. 可以」)."""
    if ctx.act == "duck.curio":
        return ctx.back_arc_ok
    if ctx.act == "duck.dock":
        return True
    return ctx.act == "duck.walk" and ctx.back_step_ok


def check_intent(intent: Intent, ctx: GuardContext, limits: Limits = Limits()) -> str | None:
    """``None`` when ``intent`` may go to robotd, else why it may not."""
    method = intent.method
    if method in FORBIDDEN_METHODS or method not in ALLOWED_METHODS:
        return f"method {method} is never sent by duck-body"
    if method == "robot.stop":
        return None
    if method == "robot.look":
        return "robot.look takes absolute targets (not a gameplay primitive yet)"
    if method == "robot.do":
        skill = intent.params.get("skill")
        if skill in ctx.trained_skills:
            # the skill guard already checked posture, tilt, hand, dock and the wear caps;
            # skill_abort only hands a running skill back to velstand, so it is never refused
            if skill != "skill_abort" and (ctx.lifted or ctx.fallen or ctx.righting):
                return f"robot.do {skill} while lifted/fallen/righting"
            return None
        if skill not in ROBOT_DO_SKILLS:
            return f"robot.do {skill!r} is not allowed"
        if skill != "sit_toggle" and ctx.act != "duck.curio":
            return f"robot.do {skill} only inside a curiosity act"
        if skill != "sit_toggle" and (ctx.lifted or ctx.unloaded or ctx.fallen or ctx.righting):
            return f"robot.do {skill} while lifted/unsure/fallen/righting"
        return None
    if method != "robot.move":
        return None  # robot.head / robot.mouth: expressions and looks
    vx = _mm(intent.params.get("vx", 0.0))
    vyaw = _mm(intent.params.get("vyaw", 0.0))
    vy = _mm(intent.params.get("vy", 0.0))
    if vx == 0 and vyaw == 0 and vy == 0:
        return None  # stopping is always allowed
    if vy != 0:
        return "sideways steps are not used"
    if ctx.lifted or ctx.unloaded or ctx.fallen or ctx.righting:
        return "no walking while lifted, unsure, fallen or righting"
    if ctx.capturing:
        return "no walking while a frame is being captured"
    if vx == 0:
        return "turning in place (vx 0, vyaw ≠ 0) is forbidden"
    if abs(vyaw) > limits.vyaw_max_mrad:
        return f"|vyaw| {abs(vyaw)} mrad/s over {limits.vyaw_max_mrad}"
    if 0 < abs(vx) < limits.min_step_vx_mm:
        return f"|vx| {abs(vx)} mm/s under {limits.min_step_vx_mm}: velstand does not step"
    cap = limits.approach_vx_mm if ctx.approach else limits.vx_max_mm
    if abs(vx) > cap:
        return f"|vx| {abs(vx)} mm/s over {cap}"
    if vx < 0 and -vx > REVERSE_VX_MAX_MM:
        return f"reversing at {-vx} mm/s: only {REVERSE_VX_MAX_MM} was measured"
    if vx < 0 and not reverse_allowed(ctx):
        return "reversing only as curiosity's back_arc, inside duck.dock or a cornered back step"
    if ctx.head_down_ok and ctx.approach and abs(vx) <= limits.approach_vx_mm:
        return None  # owner A2: the final approach may walk head down
    if not ctx.moving:
        if ctx.head_home_ms < HEAD_SETTLE_MS:
            return "the head must be at HOME and settled 0.3 s before walking"
    elif not ctx.head_cmd_home:
        return "walking with the head commanded away from HOME"
    elif ctx.head_off_mrad > HEAD_MOVING_MRAD:
        return f"walking with the head {ctx.head_off_mrad} mrad off HOME"
    return None


def check_world_op(op: object) -> str | None:
    """duck-body asks the simulated world for nothing but the pad contact (``pg``)."""
    return None if op in WORLD_OPS else f"world op {op!r} is not a sensor duck-body may read"


__all__ = [
    "A2_WALK_MAX_MM",
    "BACK_STEP_MAX_MM",
    "CLEAR",
    "DROP_ZONES",
    "FRESH_TARGET_MS",
    "GuardContext",
    "HEAD_MOVING_MRAD",
    "HEAD_SETTLE_MS",
    "Limits",
    "MIN_NEAR_MM",
    "REVERSE_VX_MAX_MM",
    "ROBOT_DO_SKILLS",
    "Verdict",
    "check_intent",
    "check_world_op",
    "no_drop",
    "relaxed_near",
    "reverse_allowed",
    "stop_near",
    "verdict",
]
