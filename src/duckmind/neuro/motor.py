"""The neural duck's motor system (neuro:motor): from the brain's command to robotd's intents.

Three pure parts, each a step function on frozen state (the clock is the frame's ``t_ms``):

* :func:`loco_step` — the **locomotor rhythm**. Walking comes in bursts and pauses like an
  animal's (intermittent locomotion): a pause ends when a noisy *go* integrator driven by the
  brain's drive to walk reaches threshold; a burst ends when its own satiation integrator does —
  never a timer, so bout lengths come out heavy-tailed and different every time. Speed and turn
  are smooth structured signals: an OU speed drift around the duck's own cruising speed, a slow
  meander oscillator plus three-scale (pink) curvature noise, the innate side's bias, steering
  toward a goal, the reflexes' veer, and a course-keeping reflex that uses the gyro to cancel the
  walk's measured drift (~10°/s to the right). Always inside robotd's and the owner's walking
  rules (:func:`duckmind.neuro.contract.legal_walk`): 0.35–0.55 m/s walking, ≤ 0.40 in arcs and
  near targets, |vyaw| ≤ 1.0, arcs not spins.
* :func:`gait_step` — the **gait phase** (0..1) from a leg joint's oscillation, for the head bob
  (the senses lane can fill ``BodySense.step_phase`` with it).
* :func:`step` — the **motor integration** every tick: the behaviour's parametric program
  (:mod:`duckmind.neuro.repertoire`), its head carriage, the brain's gaze (a speed-limited
  saccade system), the reflex overlay, liveliness (OU micro-motion, micro-saccades, breathing,
  a hatchling's wobble), while walking the look into turns, a slow (< 0.2 Hz) compensation of
  the trunk's pitch and roll and a slow (≤ 1 Hz) head sway at the waddle's tempo; clamped to the
  measured envelopes; then — for a robotd left at its default head smoothing (EMA 0.2 per
  50 Hz tick) — the smoothing is pre-compensated (an efference copy of the EMA, inverted, within
  limits). The prototype's robotd runs ``[control] head_alpha`` 1.0 (owner 2026-09-28, A100-
  validated: fid:motor), where that is the identity; the sent command's rate is limited instead
  (~5 rad/s seated), so reflexes are quick and big seated moves stay gentle on the servos.

Everything a duck does here comes from Duck Core's diversity design through the brain's
:class:`~duckmind.neuro.genome.Genome` — the prototype's ONE diversity mapping
(:func:`profile_from_genome`: the seven temperament axes as 0..1, the primary nature or the
branch, the innate side, the favourite sound, the quirk habit and the stage at its own pace,
all Duck Core's own hatching draws for the seed): two ducks with different temperaments walk,
look and startle differently; the same duck stays itself.

Floats in the prototype. Phase 5 (fixed point): OU/leaky steps become multiply-adds with
per-rate constants, the pre-compensation divides by the constant α (×5), sines use
``duckbody.fixedmath``; nothing needs more than +, ×, a table sine and atan2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

from duckmind.neuro import reflex as R
from duckmind.neuro import repertoire as P
from duckmind.neuro import signals as S
from duckmind.neuro.contract import (
    APPROACH_VX,
    ARC_VX_MIN,
    BEHAVIOURS,
    NORMAL_VX_MAX,
    HeadPose,
    MotorCommand,
    SensorFrame,
    dock_walk,
    legal_walk,
)

MOTOR_VERSION = "neuro-motor-1"

# ── the body's numbers ────────────────────────────────────────────────────────

ENVELOPE = (0.25, 0.35, 0.6, 0.25)  # rad per clip unit: content.SEATED_ENVELOPE (np, hp, hy, hr)
# Standing, the walking network owns the neck: a pull-back (np < 0) is not performed at all
# (pet:body: gain 0.00 bam / 0.04 mjcf; seated 0.54). The same "head back and up" is made with
# the head pitch instead (motor equivalence), radian for radian: 0.25 rad of neck per clip unit
# becomes 0.25 / 0.35 clip units of head pitch (fid:motor: it was half of that, the startle's
# snap standing reached ~7–10°).
NECK_TO_PITCH = ENVELOPE[0] / ENVELOPE[1]
JOINT_SIGN = (-1.0, -1.0, 1.0, 1.0)  # content.JOINT_SIGN (np, hp, hy, hr)
MOUTH_ENVELOPE = 0.5  # content.SEATED_ENVELOPE["mouth"]: robot.mouth "open" = m × 0.5
HEAD_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
ROBOTD_TICK_MS = 20.0
ROBOTD_HEAD_ALPHA = 0.2  # robotd's own default control.head_alpha: 1.78 Hz
# fid:motor 2026-09-28 — the owner approved robotd's [control] head_alpha 1.0 for the neural duck
# (「加快」): the head command goes to the policy unsmoothed (A100 2026-09-28, 102 conditions ×
# 512 DR ducks: standing t90 0.46 → 0.34 s, seated 0.24 → 0.16 s, 0/10,214 falls; walking at
# 1.0 0/2,048). The motor's efference-copy pre-compensation below follows the profile's
# ``head_alpha`` and is the identity at 1.0 (it only matters for a robotd left at 0.2). What 1.0
# costs is in large or fast SEATED moves (a 1-rad step: head torque 0.28 → 0.51 N·m, foot
# shuffles 10 → 26.5 %; seated 3 Hz sines strain): so the sent command's rate is limited —
# HEAD_SLEW_SIT seated (the A100 recommendation, ~5 rad/s), HEAD_SLEW otherwise (standing 1-rad
# steps peak at 4.7 rad/s on the joint at 1.0, ≤ 0.29 N·m: no limit needed, this only stops a
# pathological jump).
HEAD_ALPHA_FAST = 1.0
HEAD_SLEW_SIT = 5.0  # rad/s per joint, seated (and getting up / sitting down)
HEAD_SLEW = 12.0  # rad/s per joint, standing or walking
# The head's yaw range (fid:motor): the A100 validated head yaw offsets to ±1.0 rad while
# walking at 0.40–0.55 (1/1,023 and 0/1,024 falls, like HOME) and 1-rad steps standing and seated
# (0 falls) — an orienting turn may use all of it (the programs' clip units stop at ±0.6 rad).
YAW_MAX_RAD = 1.0
YAW_CLIP = YAW_MAX_RAD / ENVELOPE[2]  # … in clip units (±1.67)
# While walking the yaw goal keeps WALK_MARGIN inside that (the body link's walking gate reads
# the MEASURED head against ±1.0 rad), the pitch up to the validated look-up (−0.35 rad).
WALK_HP_UP = 0.9  # clip units of head pitch up while walking (0.315 rad)
# Pre-compensation limits: how far the sent command may run ahead of the goal (rad), and the
# absolute command range (inside velstand/sitstand's trained head range ±1.10/±1.10/±1.40/±0.31).
LEAD_MAX = (0.15, 0.20, 0.35, 0.10)
COMMAND_MAX = (0.40, 0.55, YAW_MAX_RAD, 0.31)
CAM_ELEVATION_HOME = R.CAM_ELEVATION_HOME
STRIDE_HZ = 2.4  # the walking network's stride (headless: 4.8 touchdowns/s at 0.40–0.55 m/s)
# The walking head SWAY (2026-09-28 integration; contract MotorCommand.bob): ducks waddle, they
# do not head-bob like pigeons (pet:etho), and a stride-rate (~2–2.4 Hz) head motion is beyond
# the head's bandwidth (pet:body: yaw −3 dB 0.87 Hz standing). So the head sways side to side —
# roll with a little yaw — slowly, locked to every second (or third, for a fast stride) cycle of
# the waddle, never faster than SWAY_MAX_HZ; sized above the gearbox deadband (yaw ≥ ±3°).
SWAY_MAX_HZ = 1.0
# (review 2026-09-28: the sway was phase-locked to a noisy measured gait phase and came out
# smeared, ~1° at 0.8 Hz — invisible. Now a clean oscillator at a third of the measured stride
# rate (the waddle sets its tempo, not its every jitter), ±3.5–4.5° of head roll with a little
# yaw: above the gearboxes' ±3° deadband, ≤ 1 Hz — inside the head's bandwidth.)
SWAY_HZ = (0.6, 0.95)  # the sway's rate: a third of the stride, kept in this band
SWAY_DEFAULT_HZ = 0.8
SWAY_ROLL = (0.18, 0.2)  # clip units of head roll (×0.25 rad): a + b × cmd.bob
SWAY_YAW_PER_ROLL = 0.4  # clip units of yaw per clip unit of roll
ARC_BIAS = 0.6  # rad/s: a reflex veer at least this strong (reflex.TURN_AWAY_VYAW) makes an arc
# The walking head stabiliser (vestibulo-collic feed-forward) is OFF by default (profile.vcr 0):
# review 2026-09-28 — at the stride rate (2.4 Hz) it made the sent head command's main motion a
# 2.4-Hz oscillation (0.13 rad roll, 0.22 rad yaw) the head cannot follow (pet:body: standing
# yaw −3 dB at 0.87 Hz), and by pet:probe's own tuning table (tune_stab2: gain 1.5, yaw lead
# 0.2 s, roll lead 0.1 s) it made the live simulator's (mjcf) camera shake WORSE (+6 % at 0.40,
# +10 % at 0.55 m/s; bam roll rate +16 %) — it helped only the bam model's yaw. Only the slow
# trunk-tilt compensation stays (below ~0.2 Hz: pet:etho A3, stabilise only below ~0.3 Hz).
WALK_MARGIN = 0.9  # the motor keeps its walking head goal at 90 % of the body link's bound
TRANSITION_ENV = 0.25  # |clip units| while the body rises or sits down
TRIM_GAIN = 0.5  # 1/s: how fast the course-keeping trim learns (loop τ ≈ 4 s at gain 0.5)
TRIM_TAU_S = 1.5  # the drift estimate's average (over ~3 strides)
TRIM_SETTLE_MS = 1200  # learn only once the gait is steady
TRIM_STRAIGHT = 0.15  # … and while the intended curvature is small, rad/s
TRIM_MAX = 0.45  # rad/s
MIN_BURST_MS = 800
MIN_PAUSE_MS = 450
# (neuro:fix 09:00, tried and left at 1.0: at 0.75 a calm duck's wander moves 38–39 % of its
# time instead of 28–31 %, but the bold explorer, walking more, corners itself in the brain-only
# synthetic room — turn_around 33 % of a 30-min run, stuck where no arc has room and no reverse
# is allowed; see PROTOTYPE.md, neuro:fix, known gaps)
# (fid:motor 2026-09-28: the owner allowed reversing a small arc when cornered, and the brain now
# backs out of corners — so the lower threshold is back: synthetic, brain + motor, 30 min × 2 seeds
# per preset: walking explorer 38 → 42–43 %, snuggler 12–13 → 13–15 %, dreamer 13 → 14–15 %,
# hatchling 18–20 → 20–22 %, turn_around ≤ 0.9 % of the time, never cornered for long)
PAUSE_THETA_MEDIAN = 0.75  # a pause's go-threshold, median (the go integrator's units)
PAUSE_THETA_SIGMA = 0.45  # the spread of a pause's threshold (log-normal): a heavier tail …
BURST_THETA_SIGMA = 0.35
THETA_RANGE = (0.6, 1.5)  # … kept where the leaky go-integrator can still reach it
PANIC_VX_MIN = 0.52  # a flight in panic runs between this and NORMAL_VX_MAX (never one value)
PANIC_ACTS = frozenset({"flee", "retreat"})  # the acts whose walk may be a panic (see LocoInput)
PURSUE_GO = 2.0  # 1/s: what a walk toward a thing adds to the go integrator while it rests
PURSUE_TIRE_CUT = 0.85  # … and the share of a burst's tiring it takes away
SOFT_FLOOR = 0.39  # below this the cruising speed's drift is squeezed softly toward 0.35
VCR_GAIN = 1.5  # (pet:probe's feed-forward tuning; used only with profile.vcr > 0)
VCR_TAU_S = 0.6
TRUNK_LP_S = 0.8  # the trunk tilt the head compensates: slower than ~0.2 Hz only
VCR_YAW_LEAD_S = 0.2
VCR_ROLL_LEAD_S = 0.1


def pose_to_offsets(pose: HeadPose) -> tuple[float, float, float, float]:
    """Clip units (absolute side) → robot.head offsets from HOME, rad (content.head_targets)."""
    values = (pose.np, pose.hp, pose.hy, pose.hr)
    return (
        JOINT_SIGN[0] * ENVELOPE[0] * values[0],
        JOINT_SIGN[1] * ENVELOPE[1] * values[1],
        JOINT_SIGN[2] * ENVELOPE[2] * values[2],
        JOINT_SIGN[3] * ENVELOPE[3] * values[3],
    )


def offsets_to_pose(offsets: tuple[float, float, float, float], mouth: float = 0.0) -> HeadPose:
    return HeadPose(
        np=offsets[0] / (JOINT_SIGN[0] * ENVELOPE[0]),
        hp=offsets[1] / (JOINT_SIGN[1] * ENVELOPE[1]),
        hy=offsets[2] / (JOINT_SIGN[2] * ENVELOPE[2]),
        hr=offsets[3] / (JOINT_SIGN[3] * ENVELOPE[3]),
        mouth=mouth,
    )


# ── the locomotor rhythm ──────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class LocoParams:
    vx_base: float = 0.46  # its cruising command, m/s
    vx_spread: float = 0.045  # OU std of the speed drift
    vx_tau: float = 1.8  # s
    go_gain: float = 0.9  # 1/s: go-integrator fill rate at go = 1 (pause → burst)
    go_leak: float = 0.3  # 1/s
    go_noise: float = 0.35
    tire: float = 0.2  # 1/s: a burst's satiation fill rate at go = 0.5 (burst → pause)
    tire_noise: float = 0.25
    min_burst_s: float = 1.6  # the body needs ~0.5–1.3 s just to start stepping
    min_pause_s: float = 0.5  # the legs keep stepping ~0.45 s after a stop
    meander_amp: float = 0.3  # rad/s, the slow curvature oscillation
    meander_hz: float = 0.11
    turn_sigma: float = 0.22  # rad/s, multi-scale curvature noise
    side_bias: float = 0.04  # rad/s toward the innate side
    steer_gain: float = 1.5  # rad/s per rad of goal bearing
    course_gain: float = 0.0  # 1/s: its own course keeping (off: the motor's learned trim does it)
    arc_threshold: float = 0.3  # |vyaw| above this is an arc (vx ≤ 0.40, ≥ 0.35)


@dataclass(frozen=True, slots=True)
class LocoInput:
    """What drives the rhythm this tick (the brain's; the senses' through the reflexes)."""

    go: float = 0.0  # 0..1 the drive to walk (restless, curious, social pull…)
    bearing: float | None = None  # a goal to steer toward (rad, + = left); None = wander
    range_m: float | None = None
    approach: bool = False  # toward a specific thing (≤ 0.40)
    halt: bool = False  # a reflex holds the body
    vx_cap: float | None = None  # a reflex's speed cap (near things)
    vyaw_bias: float = 0.0  # a reflex's veer
    arousal: float = 0.5
    energy: float = 0.8
    yaw_rate: float | None = None  # gyro z (rad/s), for course keeping
    turn: float = 0.0  # an extra curvature the behaviour wants (turn_around, retreat arcs)
    wander_scale: float = 1.0  # × the meander and the pink curvature noise (forage ×1.3,
    # zoomies ×1.8, flee/retreat/approach ×0.3–0.5): how much the path wanders in this behaviour
    # a flight in panic (2026-10-04, owner: 「驚慌失措，全速逃離」): the rhythm does not rest — no
    # pauses, no tiring, the speed at the top of the band (``PANIC_VX_MIN``…max) wherever it is not
    # in an arc (an arc is never faster than 0.40, measured: the body's limit)
    panic: bool = False
    # a walk straight up to a thing it wants (2026-10-05, legs): 0..1, how much it goes for it. The
    # walk tires slowly and a stop is short — a duck that goes for something does not stop every
    # second step — but the rhythm is still its own (no flat run: the speed and the bursts vary)
    pursue: float = 0.0


@dataclass(frozen=True, slots=True)
class LocoState:
    t_ms: int = -1
    walking: bool = False
    since_ms: int = 0
    acc: float = 0.0
    speed: float = 0.0
    turn: S.Pink = field(default_factory=S.Pink)
    phase: float = 0.0
    course_i: float = 0.0
    yaw_lp: float = 0.0
    last_vyaw: float = 0.0
    rng: int = 0x10C0
    bursts: int = 0
    theta: float = 1.0  # this pause's (or burst's) own threshold: heavy-tailed, never a timer


@dataclass(frozen=True, slots=True)
class LocoOut:
    vx: float
    vyaw: float
    walking: bool
    burst_ms: int
    pause_ms: int
    drive: float
    curvature: float
    bursts: int

    def telemetry(self) -> dict[str, float]:
        return {
            "go": round(self.drive, 3),
            "burst": 1.0 if self.walking else 0.0,
            "burst_s": round(self.burst_ms / 1000.0, 2),
            "pause_s": round(self.pause_ms / 1000.0, 2),
            "speed": round(self.vx, 3),
            "curvature": round(self.curvature, 3),
            "bursts": float(self.bursts),
        }


def loco_params_from(style: P.Style) -> LocoParams:
    """The rhythm of this duck (Duck Core's axes and branch, the stage: a hatchling walks in
    short wobbly spurts, an explorer in long meanders, a snuggler sets off less and stops sooner;
    activity sets its cruising speed; never still — the owner's rule)."""
    a, e, b, tp, y = style.activity, style.exploration, style.boldness, style.tempo, style.young
    wx, ws, wd = style.w("explorer"), style.w("snuggler"), style.w("dreamer")
    return LocoParams(
        # (fix 2026-09-28: a calm duck's cruising speed sat at the 0.35 floor — the OU drift
        # below it was clipped onto one value; now every duck cruises above the band's floor)
        vx_base=S.clamp(
            0.43 + 0.12 * (a - 0.2) + 0.03 * (b - 0.5) - 0.03 * y + 0.02 * wx - 0.01 * ws,
            0.41,
            0.51,
        ),
        vx_spread=0.07 + 0.04 * a,  # never one speed, calm ducks too (pet:probe σ 0.05–0.12)
        # (integration 2026-09-28, pet:etho D1: moves log-normal, median ~2.5 s, a traveller's
        # longer; pauses median ~1.5 s with a heavy tail; the pauses 30–60 % of a bout — the
        # old numbers gave the explorer 7-s moves and 1-s pauses, 84 % walking)
        # (integration 2026-09-28, later: calm ducks' walks were 1.7-s spurts between 3-s
        # stops — 27 % of a wander bout — the owner's 走太少; activity now shapes the rhythm
        # less and the snuggler's moves are ~2.5 s, the explorer's stay ~3 s)
        # (fix 2026-09-28: a calm duck's go integrator hardly reached its threshold — only its
        # noise set it off — and its walks tired fast: it walked 7–9 % of the day)
        go_gain=0.7 + 0.25 * a + 0.05 * wx - 0.05 * wd,
        tire=(0.36 + 0.1 * y) * (1.05 - 0.15 * a) * (1.0 + 0.1 * ws + 0.05 * wd - 0.15 * wx),
        min_burst_s=1.6 - 0.2 * y,
        # (review 2026-09-28: the path was a rosette of ~0.8-m loops, 63–97 % of the walking in
        # arcs — capped at 0.40 m/s, one speed; now mostly straighter stretches that bend, the
        # explorer's still the most wandering)
        meander_amp=0.06 + 0.14 * e + 0.05 * wx,
        meander_hz=(0.07 + 0.08 * tp) * (1.0 - 0.25 * wd),
        turn_sigma=0.06 + 0.1 * e + 0.1 * y,
        side_bias=0.04 * style.side,
    )


def loco_step(p: LocoParams, s: LocoState, inp: LocoInput, t_ms: int) -> tuple[LocoState, LocoOut]:
    """One tick of the locomotor rhythm at ``t_ms``."""
    dt = 0.02 if s.t_ms < 0 else max(0.0, (t_ms - s.t_ms) / 1000.0)
    rng = s.rng
    rng, g_speed = S.gauss(rng)
    rng, g_acc = S.gauss(rng)
    rng, turn = S.pink_exact(s.turn, rng, dt)  # exact for any tick (5–50 Hz)
    speed = S.ou_exact(s.speed, 0.0, 1.0, p.vx_tau, dt, g_speed)
    phase = (s.phase + 2.0 * math.pi * p.meander_hz * dt) % (2.0 * math.pi)
    walking, since, acc, bursts = s.walking, s.since_ms, s.acc, s.bursts
    theta = s.theta
    go = S.clamp(inp.go, 0.0, 1.0)
    if s.t_ms < 0:
        since = t_ms
    held = inp.halt or go < 0.03 or (inp.vx_cap is not None and inp.vx_cap < 0.30)
    # each pause and each burst gets its own integrate-to threshold, log-normal: pauses come out
    # heavy-tailed (pet:etho D1: median ~1.5 s, a tail to 15 s; review 2026-09-28: burstiness
    # B < 0, a regular stop–go), bursts less so
    if held:
        if walking:
            walking, since = False, t_ms
            rng, theta = S.lognormal(rng, PAUSE_THETA_MEDIAN, PAUSE_THETA_SIGMA)
            theta = S.clamp(theta, *THETA_RANGE)
        acc = 0.0 if inp.halt else max(0.0, acc - dt)
    elif inp.panic:
        # a flight does not stop to rest or tire: it goes from the first tick and goes on
        if not walking:
            walking, since, acc, bursts = True, t_ms, 0.0, bursts + 1
        acc = 0.0
    elif not walking:
        acc += dt * (p.go_gain * go - p.go_leak * acc) + p.go_noise * math.sqrt(dt) * g_acc
        acc += dt * PURSUE_GO * S.clamp(inp.pursue, 0.0, 1.0)
        acc = max(0.0, acc)
        if acc >= theta and t_ms - since >= p.min_pause_s * 1000.0:
            walking, since, acc, bursts = True, t_ms, 0.0, bursts + 1
            rng, theta = S.lognormal(rng, 1.0, BURST_THETA_SIGMA)
            theta = S.clamp(theta, *THETA_RANGE)
    else:
        tire = (
            p.tire * (0.55 + 0.9 * (1.0 - go)) * (0.8 + 0.8 * (1.0 - S.clamp(inp.energy, 0.0, 1.0)))
        )
        tire *= 1.0 - PURSUE_TIRE_CUT * S.clamp(inp.pursue, 0.0, 1.0)
        acc = max(0.0, acc + dt * tire + p.tire_noise * math.sqrt(dt) * g_acc)
        if acc >= theta and t_ms - since >= p.min_burst_s * 1000.0:
            walking, since, acc = False, t_ms, 0.0
            rng, theta = S.lognormal(rng, PAUSE_THETA_MEDIAN, PAUSE_THETA_SIGMA)
            theta = S.clamp(theta, *THETA_RANGE)
    yaw_lp = s.yaw_lp
    if inp.yaw_rate is not None:
        yaw_lp = S.leaky(yaw_lp, inp.yaw_rate, 1.0, dt)
    course_i = s.course_i
    vx = vyaw = curvature = 0.0
    if walking:
        vx = (
            p.vx_base
            + p.vx_spread * speed
            + 0.08 * (S.clamp(inp.arousal, 0.0, 1.0) - 0.5)
            - 0.05 * (1.0 - S.clamp(inp.energy, 0.0, 1.0))
        )
        if inp.panic:
            # at the top of the band, with only a little variation (never one value)
            vx = PANIC_VX_MIN + (NORMAL_VX_MAX - PANIC_VX_MIN) * S.clamp(
                0.5 + 0.25 * speed, 0.0, 1.0
            )
        if vx < SOFT_FLOOR:
            # a slow duck's slow drift stays a drift near the gait's floor, not one clipped value
            vx = ARC_VX_MIN + (SOFT_FLOOR - ARC_VX_MIN) * math.exp(
                (vx - SOFT_FLOOR) / (SOFT_FLOOR - ARC_VX_MIN)
            )
        scale = max(0.0, inp.wander_scale)
        wander = scale * (p.meander_amp * math.sin(phase) + p.turn_sigma * turn.value) + p.side_bias
        if inp.bearing is not None:
            curvature = p.steer_gain * S.wrap(inp.bearing) + 0.3 * wander
            if inp.range_m is not None and inp.range_m < 0.5:
                vx = min(vx, APPROACH_VX)
        else:
            curvature = wander
        curvature += inp.turn + inp.vyaw_bias
        if abs(inp.vyaw_bias) >= ARC_BIAS:
            # the reflexes are turning it away (a wall, a stall, a stumble): an arc in their
            # direction, never straight on (the body link passes only |vyaw| ≥ 0.5 then)
            sign = 1.0 if inp.vyaw_bias > 0.0 else -1.0
            curvature = sign * max(ARC_BIAS, sign * curvature)
        straight = abs(s.last_vyaw) < 0.2 and inp.yaw_rate is not None
        if straight and t_ms - since > 1500:  # course keeping: cancel the measured drift
            course_i = S.clamp(course_i - p.course_gain * yaw_lp * dt, -0.35, 0.35)
        curvature += course_i
        cap = APPROACH_VX if inp.approach else NORMAL_VX_MAX
        if inp.vx_cap is not None:
            cap = min(cap, inp.vx_cap)
        if abs(curvature) > p.arc_threshold:
            cap = min(cap, APPROACH_VX)
        if cap < NORMAL_VX_MAX:
            # a cap (an arc, an approach, a reflex): the speed keeps its own variation inside
            # the band under the cap instead of being flattened onto it (review 2026-09-28: 59–98 %
            # of the walking at exactly one speed, the cap)
            u = S.clamp((vx - ARC_VX_MIN) / (NORMAL_VX_MAX - ARC_VX_MIN), 0.0, 1.0)
            vx = ARC_VX_MIN + u * max(0.0, cap - ARC_VX_MIN)
        else:
            vx = S.clamp(vx, ARC_VX_MIN, cap)
        vx, vyaw = legal_walk(
            vx, curvature, approach=inp.approach or abs(curvature) > p.arc_threshold
        )
    state = LocoState(
        t_ms=t_ms,
        walking=walking,
        since_ms=since,
        acc=acc,
        speed=speed,
        turn=turn,
        phase=phase,
        course_i=course_i,
        yaw_lp=yaw_lp,
        last_vyaw=vyaw,
        rng=rng,
        bursts=bursts,
        theta=theta,
    )
    span = t_ms - since
    return state, LocoOut(
        vx, vyaw, walking, span if walking else 0, 0 if walking else span, go, curvature, bursts
    )


# ── the gait phase ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class GaitState:
    mean: float = 0.0
    prev: float | None = None
    amp: float = 0.0
    phase: float = 0.0


def gait_step(
    g: GaitState, joint: float, dt: float, hz: float = STRIDE_HZ
) -> tuple[GaitState, float | None]:
    """The gait phase (0..1, one cycle per stride) from one leg joint's angle (e.g. the left hip
    pitch in robot.state): the joint's oscillation about its slow mean and its rate give a
    quadrature pair. ``None`` while it is not stepping (oscillation under 0.03 rad)."""
    if dt <= 0.0:
        return g, None
    mean = S.leaky(g.mean if g.prev is not None else joint, joint, 1.0, dt)
    x = joint - mean
    prev = g.prev if g.prev is not None else joint
    v = (joint - prev) / dt
    omega = 2.0 * math.pi * hz
    amp = S.leaky(g.amp, math.hypot(x, v / omega), 0.4, dt)
    phase = (math.atan2(-v / omega, x) / (2.0 * math.pi)) % 1.0
    out = GaitState(mean, joint, amp, phase)
    return out, (phase if amp > 0.03 else None)


# ── the motor integration ─────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class MotorProfile:
    """One duck's motor settings: its style (repertoire), rhythm (locomotor) and reflexes, plus
    the measured envelopes and the plant compensation."""

    style: P.Style = field(default_factory=P.Style)
    loco: LocoParams = field(default_factory=LocoParams)
    reflex: R.ReflexParams = field(default_factory=R.ReflexParams)
    # robotd's [control] head_alpha on the body this motor drives (the simulator's robotd runs
    # 1.0 in prototype mode; the live driver sets what the body's robotd really runs)
    head_alpha: float = HEAD_ALPHA_FAST
    head_slew_sit: float = HEAD_SLEW_SIT  # rad/s: the sent head command's rate limit, seated
    head_slew: float = HEAD_SLEW  # rad/s: … standing or walking
    lead: float = 1.0  # robotd EMA pre-compensation: 0 = raw, 1 = the full inverse (clamped)
    walk_env: float = 0.8  # |clip units| per joint while walking (measured: see PROTOTYPE notes)
    stand_env: float = 1.0  # standing or seated: the A100-validated seated envelope
    head_down_np: float = 0.5  # owner A2: the head-down approach (HALF_DOWN) at ≤ 0.40 m/s
    bob_amp: float = 0.35  # clip units of neck (np) at bob = 1
    bob_cycles: int = 1  # bob cycles per stride
    bob_phase: float = 0.0  # rad, lead to meet the plant's lag
    look_into_turn: float = 0.4  # head yaw (clip units) per rad/s of turning
    vcr: float = 0.0  # the stride-rate head stabiliser (× cmd.stabilise); 0 = off (see VCR_GAIN)
    trim_gain: float = 1.0  # the course-keeping trim's learning rate (×); 0 = off
    seed: int = 0


def profile_from_style(
    style: P.Style,
    reflex: R.ReflexParams | None = None,
    *,
    seed: int = 0,
    head_alpha: float = HEAD_ALPHA_FAST,
) -> MotorProfile:
    """A motor profile around a :class:`~duckmind.neuro.repertoire.Style` (tests and tools;
    a real duck's comes from :func:`profile_from_genome`). ``head_alpha``: robotd's
    ``[control] head_alpha`` on the body it drives."""
    return MotorProfile(
        head_alpha=S.clamp(head_alpha, 0.01, 1.0),
        style=style,
        loco=loco_params_from(style),
        reflex=reflex
        if reflex is not None
        else R.params_from(
            {
                "boldness": style.boldness,
                "wit": style.wit,
                "exploration": style.exploration,
                "sociability": style.sociability,
            },
            stage=style.stage,
            side=style.side,
            habit=style.habit,
        ),
        # subtle: ducks waddle, they do not head-bob like pigeons (pet:etho: geese 1–4.6 cm)
        bob_amp=0.15 + 0.1 * style.activity - 0.05 * style.w("snuggler"),
        look_into_turn=0.3 + 0.25 * style.exploration,
        seed=seed,
    )


def profile_from_genome(genome: Any, *, head_alpha: float = HEAD_ALPHA_FAST) -> MotorProfile:
    """The motor profile of a brain :class:`~duckmind.neuro.genome.Genome` — the prototype's ONE
    diversity mapping (Duck Core's own hatching draws for the seed: the seven temperament axes
    as 0..1, the nature or the branch, the stage at its own pace, the innate side, the favourite
    sound, the quirk habit). The same duck the brain runs; ``Genome.from_seed(seed, gd=…)`` for a
    seed. ``head_alpha``: robotd's ``[control] head_alpha`` on the body it drives."""
    style = P.style_from_genome(genome)
    seed = int(genome.seed) if int(genome.seed) >= 0 else 0
    return profile_from_style(style, R.params_from_genome(genome), seed=seed, head_alpha=head_alpha)


Quad = tuple[float, float, float, float]
ZERO4: Quad = (0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class MotorState:
    t_ms: int = -1
    program: P.ProgramState | None = None
    rng: int = 0x0D0C
    ema: Quad = ZERO4  # efference copy of robotd's smoothed head command (rad from HOME)
    sent: Quad = ZERO4  # the head command sent last tick (rad from HOME): its rate is limited
    gaze: tuple[float, float] = (0.0, 0.0)  # where the gaze system has the head (hy, hp clip)
    ou: Quad = ZERO4  # idle micro-motion (clip units)
    micro: tuple[float, float] = (0.0, 0.0)  # the current micro-saccade offset (hy, hp)
    micro_next: int = 0
    breath: float = 0.0
    gait_osc: float = 0.0  # a free-running stride oscillator when no gait phase is sensed
    # the walking head sway (≤ 1 Hz): its phase (rad), the gait cycles counted (it locks to
    # every SWAY_EVERY-th waddle cycle) and the gait phase seen last tick
    sway: float = 0.0
    sway_cycles: int = 0
    gait_prev: float = 0.0
    gait_hz: float = 0.0  # the measured stride rate (from the gait phase's wraps), 0 = unknown
    gait_wrap_ms: int = -1  # when the gait phase last wrapped
    trunk: Quad = ZERO4  # trunk pitch, roll: fast (0.3 s) and carriage (8 s) averages, rad
    vcr: tuple[float, float, float] = (0.0, 0.0, 0.0)  # heading ref, steady yaw rate, roll ref
    trim: float = 0.0  # the learned course-keeping yaw trim, rad/s (a body property: kept)
    drift: float = 0.0  # the low-passed measured yaw rate while walking, rad/s
    walk_t0: int = -1  # when the current burst began (-1: not walking)
    stop_t0: int = -1_000_000  # when the current pause began
    last_twist: tuple[float, float] = (0.0, 0.0)
    posture: str = "sit"
    posture_bout: int = -1
    vocal_seq: int = 0
    sound_ms: int = -10_000
    sound_level: float = 0.0
    pose: HeadPose = field(default_factory=HeadPose)
    # a moment (contract.MOMENTS) playing over the current program, and the one last started
    moment: P.ProgramState | None = None
    moment_seq: int = 0


@dataclass(frozen=True, slots=True)
class MotorOutput:
    """robotd intents for this tick, and what they mean (for the page and the evaluators)."""

    t_ms: int
    head: Quad  # robot.head offsets from HOME to send now (rad; pre-compensated)
    head_goal: Quad  # the intended head pose (rad from HOME), before pre-compensation
    pose: HeadPose  # the intended pose in clip units (absolute side)
    mouth: float  # robot.mouth "open" (0..0.5 of the jaw travel, as LiveBody sends it)
    vx: float
    vyaw: float
    posture: str  # "sit" | "stand": the posture to hold (LiveBody.request_posture)
    sound: str | None  # a vocal tag to play now (once)
    sound_level: float
    program: str
    label: str  # what the program is doing now (plain Chinese)
    unit: int
    walking: bool
    notes: tuple[str, ...] = ()
    trim: float = 0.0  # the learned course-keeping yaw trim (rad/s)

    @property
    def vy(self) -> float:
        return 0.0

    def head_params(self) -> dict[str, float]:
        """``robot.head`` params (offsets from HOME, rad)."""
        return {name: round(value, 5) for name, value in zip(HEAD_JOINTS, self.head, strict=True)}

    def move_params(self) -> dict[str, float]:
        return {"vx": round(self.vx, 4), "vy": 0.0, "vyaw": round(self.vyaw, 4)}

    def to_json(self) -> dict[str, Any]:
        return {
            "t_ms": self.t_ms,
            "head": list(self.head),
            "head_goal": list(self.head_goal),
            "pose": {
                "np": self.pose.np,
                "hp": self.pose.hp,
                "hy": self.pose.hy,
                "hr": self.pose.hr,
                "mouth": self.pose.mouth,
            },
            "mouth": self.mouth,
            "vx": self.vx,
            "vyaw": self.vyaw,
            "posture": self.posture,
            "sound": self.sound,
            "sound_level": self.sound_level,
            "program": self.program,
            "label": self.label,
            "unit": self.unit,
            "walking": self.walking,
            "trim": self.trim,
            "notes": list(self.notes),
        }


def _mix(a: HeadPose, b: HeadPose, w: float) -> HeadPose:
    w = S.clamp(w, 0.0, 1.0)
    return HeadPose(
        np=a.np + (b.np - a.np) * w,
        hp=a.hp + (b.hp - a.hp) * w,
        hy=a.hy + (b.hy - a.hy) * w,
        hr=a.hr + (b.hr - a.hr) * w,
        mouth=a.mouth + (b.mouth - a.mouth) * w,
    )


def _call_opening(since_ms: int, level: float) -> float:
    """How far a call opens the beak ``since_ms`` after it started (0 outside the call)."""
    if since_ms < 0 or since_ms >= 420:
        return 0.0
    call = 0.35 + 0.5 * level
    if since_ms < 120:
        return call * S.ease(since_ms / 120.0, "snap")
    if since_ms <= 220:
        return call
    return call * (1.0 - S.min_jerk((since_ms - 220) / 200.0))


def _toward(value: float, target: float, rate: float, dt: float) -> float:
    step = rate * dt
    if abs(target - value) <= step:
        return target
    return value + step if target > value else value - step


QUIET_REFLEXES = frozenset({"freeze", "startle", "righting"})
STOP_ACTS = frozenset({"freeze", "startle", "still"})  # acts that hold the legs still at once
CUED_POSTURE = frozenset({"stand_up", "sit_down"})  # the program says when (after its head prep)


def step(
    profile: MotorProfile, state: MotorState, cmd: MotorCommand, frame: SensorFrame
) -> tuple[MotorState, MotorOutput]:
    """One motor tick at ``frame.t_ms``: the command's program, overlays and clamps, and the
    robotd intents (pre-compensated head, mouth, legal twist, posture, sound)."""
    t = frame.t_ms
    dt = 0.02 if state.t_ms < 0 else max(0.0, (t - state.t_ms) / 1000.0)
    style = profile.style
    notes: list[str] = []
    rng = state.rng
    body = frame.body
    walking = body.posture == "walk" or abs(body.vx) > 0.2
    standing = body.posture in ("stand", "walk")

    # 1. the behaviour's program
    prog_state, prog = P.step(state.program, cmd, style, frame, t)
    pose = prog.pose
    # 1b. a moment (contract.MOMENTS): a short head act played over the program — it starts
    # from where the head is and hands back to the program when it ends (review 2026-09-28:
    # glances, tilts and double takes are events inside an act, not acts of their own)
    moment_state = state.moment
    moment_vocals: tuple[tuple[str, float], ...] = ()
    moment_alive: float | None = None
    if cmd.moment and cmd.moment_seq != state.moment_seq and cmd.moment in P.BUILDERS:
        start_cmd = replace(
            cmd,
            behaviour=cmd.moment,
            bout=-cmd.moment_seq,
            variant=(cmd.variant * 31 + cmd.moment_seq * 7919) & 0x7FFFFFFF,
        )
        last = prog_state.last
        moment_state = P.start(start_cmd, style, frame, t, last)
    if moment_state is not None:
        elapsed = float(t - moment_state.unit_t0)
        score = moment_state.score
        if elapsed >= score.length or cmd.reflex in QUIET_REFLEXES or cmd.asleep:
            moment_state = None
        else:
            m_pose = score.pose_at(elapsed)
            fade_in = S.clamp(elapsed / 120.0, 0.0, 1.0)
            fade_out = S.clamp((score.length - elapsed) / 250.0, 0.0, 1.0)
            w_m = min(fade_in, fade_out)
            got: list[tuple[str, float]] = []
            cue_i = moment_state.cue_i
            while cue_i < len(score.cues) and score.cues[cue_i].t <= elapsed:
                cue = score.cues[cue_i]
                if cue.kind == "vocal":
                    got.append((cue.tag, cue.a))
                cue_i += 1
            moment_state = replace(moment_state, cue_i=cue_i)
            moment_vocals = tuple(got)
            moment_alive = score.alive
            pose = _mix(
                pose,
                HeadPose(np=m_pose[0], hp=m_pose[1], hy=m_pose[2], hr=m_pose[3], mouth=m_pose[4]),
                w_m,
            )
    carriage = style.carriage
    pose = HeadPose(
        np=pose.np + carriage.np, hp=pose.hp + carriage.hp, hy=pose.hy, hr=pose.hr, mouth=pose.mouth
    )

    # 2. the brain's gaze: a speed-limited saccade system that takes over yaw and pitch
    gaze = cmd.gaze
    gy, gp = state.gaze
    reflexive = gaze.speed >= 0.8 or cmd.reflex in ("orient", "freeze", "startle")
    w = gaze.gain if reflexive else gaze.gain * prog.gaze
    if w > 0.01:
        # (fid:motor: an orienting look may turn the head the whole validated ±1.0 rad)
        target_y = S.clamp(gaze.bearing / ENVELOPE[2], -YAW_CLIP, YAW_CLIP)
        target_p = S.clamp((gaze.elevation - CAM_ELEVATION_HOME) / ENVELOPE[1], -1.0, 1.0)
        rate = 2.0 + 10.0 * S.clamp(
            gaze.speed, 0.0, 1.0
        )  # clip units/s: 0.1 s for a reflex saccade
        gy = _toward(gy, target_y, rate, dt)
        gp = _toward(gp, target_p, rate * 0.8, dt)
        pose = HeadPose(
            np=pose.np,
            hp=pose.hp + (gp - pose.hp) * w,
            hy=pose.hy + (gy - pose.hy) * w,
            hr=pose.hr,
            mouth=pose.mouth,
        )
    else:
        gy = _toward(gy, pose.hy, 6.0, dt)
        gp = _toward(gp, pose.hp, 6.0, dt)

    # 3. the reflex overlay (an orienting or freezing reflex's pose carries the yaw toward the
    # source itself, so its lift never pulls the turned head back to the middle)
    if cmd.head is not None and cmd.head_blend > 0.0:
        pose = _mix(pose, cmd.head, cmd.head_blend)

    # 4. liveliness: breathing and the young one's wobble. (Review 2026-09-28: a second,
    # motor-side saccade generator and a continuous OU drift on the held head meant the head
    # never held still — 92 % of standing half-seconds moving, the longest still run 1.5 s; a
    # bird shifts its gaze and HOLDS it (pet:etho A1). The one saccade system is the brain's
    # gaze cycle; a held fixation is still. What is left here is slow breathing on the head
    # pitch while seated (pet:body: seated the head follows small commands, gain 0.6–0.9 at
    # ±1–3°; standing they vanish in the gearboxes) and a hatchling's unsteady carriage.)
    alive = S.clamp(cmd.alive, 0.0, 1.0) * (prog.alive if moment_alive is None else moment_alive)
    if cmd.reflex in QUIET_REFLEXES:
        alive *= 0.1
    if cmd.asleep:
        alive *= 0.3
    breath_amp = 0.0 if standing else 0.06
    young = style.young
    sig = (0.0, 0.05 * young, 0.08 * young, 0.05 * young)  # a hatchling wobbles: slow, small
    ou: list[float] = []
    for i, value in enumerate(state.ou):
        rng, g = S.gauss(rng)
        ou.append(S.ou_exact(value, 0.0, sig[i], 1.2, dt, g))
    micro, micro_next = state.micro, state.micro_next
    breath = (state.breath + 2.0 * math.pi * 0.28 * dt) % (2 * math.pi)
    pose = HeadPose(
        np=pose.np + alive * ou[0],
        hp=pose.hp + alive * (ou[1] + breath_amp * math.sin(breath)),
        hy=pose.hy + alive * ou[2],
        hr=pose.hr + alive * ou[3],
        mouth=pose.mouth,
    )
    if standing and (pose.np < 0.0 or (pose.np > 0.0 and not walking)):
        # motor equivalence: standing, the neck will not pull back (pet:body: gain 0.00 bam), so
        # a raised head is made with the head pitch instead (inside the envelope, clamped below)
        # — and the pull-back is not asked for at all (fid:motor: robotd's head command is what
        # the page's witness compares with the body; a neck command the walking network never
        # performs would only claim a motion the body does not make). (fid:integrate: standing
        # still the neck barely goes forward/down either — pet:body glance-down gain ≤ 0.65,
        # ±7° → 0.22; fid:audit 4–8° commanded → 1–3° in 16 act cells — so a look down is the
        # head pitch's too. Walking, a forward neck stays: owner A2's head-down approach.)
        pose = replace(pose, np=0.0, hp=pose.hp - NECK_TO_PITCH * pose.np)

    # 5. walking: look into the turn, stabilise against the trunk, bob with the gait
    lp_pitch, lp_roll, base_pitch, base_roll = state.trunk
    lp_pitch = S.leaky(lp_pitch, body.pitch, TRUNK_LP_S, dt)  # slow tilts only (< ~0.2 Hz)
    lp_roll = S.leaky(lp_roll, body.roll, TRUNK_LP_S, dt)
    base_pitch = S.leaky(base_pitch, body.pitch, 8.0, dt)  # the trunk's own carriage
    base_roll = S.leaky(base_roll, body.roll, 8.0, dt)
    # the vestibulo-collic reflex's references: the heading and roll the gait wobbles about
    # (0.6 s averages) and the steady yaw rate (so an intended turn is not "stabilised" away)
    yaw_now = body.odom[2]
    yaw_ref, rate_lp, roll_ref = state.vcr
    if state.t_ms < 0:
        yaw_ref, roll_ref = yaw_now, body.roll
    yaw_ref = yaw_ref + S.wrap(yaw_now - yaw_ref) * min(1.0, dt / VCR_TAU_S)
    rate_lp = S.leaky(rate_lp, body.gyro[2], VCR_TAU_S, dt)
    roll_ref = S.leaky(roll_ref, body.roll, VCR_TAU_S, dt)
    gait_osc = (state.gait_osc + STRIDE_HZ * dt) % 1.0
    gait_now = body.step_phase if body.step_phase is not None else gait_osc
    sway, sway_cycles = state.sway, state.sway_cycles
    gait_hz, gait_wrap_ms = state.gait_hz, state.gait_wrap_ms
    if gait_now < state.gait_prev - 0.5:
        sway_cycles += 1  # one more waddle cycle (the phase wrapped)
        if body.step_phase is not None and gait_wrap_ms >= 0:
            period = (t - gait_wrap_ms) / 1000.0
            if 0.25 < period < 2.0:  # the stride's own rate, smoothed over a few strides
                gait_hz = (
                    1.0 / period if gait_hz <= 0.0 else gait_hz + 0.3 * (1.0 / period - gait_hz)
                )
        gait_wrap_ms = t
    reflex_hold = cmd.head_blend if cmd.head is not None else 0.0
    if walking:
        # the look into the turn follows the intended curvature (the applied yaw rate minus
        # the learned course trim, which only straightens the walk)
        lead = profile.look_into_turn * (body.vyaw - state.trim) * (1.0 - reflex_hold)
        stab = S.clamp(cmd.stabilise, 0.0, 1.0) * (1.0 - reflex_hold)
        # feed-forward from the trunk IMU only (pet:probe's tuning, open loop w.r.t. the head so
        # it cannot ring through it): yaw against the gait's wobble with a 0.2 s lead, roll
        # against the trunk's roll with the empirically signed 0.1 s gyro term
        wobble = S.wrap(yaw_now - yaw_ref) - VCR_TAU_S * rate_lp
        vcr_yaw = (
            -VCR_GAIN * profile.vcr * stab * (wobble + VCR_YAW_LEAD_S * (body.gyro[2] - rate_lp))
        )
        vcr_roll = (
            VCR_GAIN
            * profile.vcr
            * stab
            * ((body.roll - roll_ref) + VCR_ROLL_LEAD_S * body.gyro[0])
        )
        pose = HeadPose(
            np=pose.np,
            hp=pose.hp + stab * (lp_pitch - base_pitch) / ENVELOPE[1],
            hy=pose.hy + lead + S.clamp(vcr_yaw, -0.4, 0.4) / ENVELOPE[2],
            hr=pose.hr
            - 0.7 * stab * (lp_roll - base_roll) / ENVELOPE[3]
            + S.clamp(vcr_roll, -0.15, 0.15) / ENVELOPE[3],
            mouth=pose.mouth,
        )
        bob = S.clamp(cmd.bob, 0.0, 1.0) * (1.0 - reflex_hold)
        if bob > 0.0:
            # a slow, clean sway at a third of the stride rate (the waddle's tempo), ≤ 1 Hz
            hz = S.clamp(gait_hz / 3.0, *SWAY_HZ) if gait_hz > 0.0 else SWAY_DEFAULT_HZ
            sway = sway + dt * 2.0 * math.pi * min(SWAY_MAX_HZ, hz)
            roll = (SWAY_ROLL[0] + SWAY_ROLL[1] * bob) * (0.8 + profile.bob_amp)
            pose = HeadPose(
                np=pose.np,
                hp=pose.hp,
                hy=pose.hy + SWAY_YAW_PER_ROLL * roll * math.sin(sway + 0.5),
                hr=pose.hr + roll * math.sin(sway),
                mouth=pose.mouth,
            )

    # 6. sound: the brain's calls and the program's, one at a time; a call opens the beak
    sound: str | None = None
    level = 0.0
    if cmd.vocal and cmd.vocal_seq != state.vocal_seq:
        sound, level = cmd.vocal, S.clamp(cmd.vocal_level, 0.0, 1.0)
    elif moment_vocals and t - state.sound_ms >= 250:
        sound, level = moment_vocals[0]
    elif prog.vocals and t - state.sound_ms >= 250:
        sound, level = prog.vocals[0]
    sound_ms, sound_level = state.sound_ms, state.sound_level
    if sound is not None:
        sound_ms, sound_level = t, level
    elif cmd.vocal is None and cmd.vocal_seq != state.vocal_seq:
        # (live voice, LV2) a new call number with no call is the stop of the one that sounds: a
        # person spoke over it; the beak closes now instead of when the call would have ended
        sound_ms = -1_000_000
    opening = _call_opening(t - sound_ms, sound_level)
    if opening > 0.0:
        pose = replace(pose, mouth=max(pose.mouth, opening))

    # 7. the twist: the brain's (already legal) or the program's own shaping, re-checked
    spec = BEHAVIOURS.get(prog_state.behaviour)
    vx, vyaw = 0.0, 0.0
    body_ok = standing and not body.fallen and not cmd.asleep
    # (dock:build 2026-09-28) the docking choreography's own twist (neuro.dock: its U-turn and
    # tight arcs at 0.30, the reverse at −0.40, its own PI steering with the gait's drift
    # feed-forward — as phase 3 measured it): passed on as it is — no raise to 0.35, no
    # course-keeping trim on top, no burst/pause holds (it times its own stops and looks)
    dock_twist = body_ok and cmd.docking
    if dock_twist:
        vx, vyaw = dock_walk(cmd.vx, cmd.vyaw)
    elif body_ok and prog.twist is not None and prog_state.behaviour in P.SHAPES_WALK:
        if prog_state.behaviour == "zoomies":
            if cmd.vx > 0.0:  # the brain's rhythm says go: the program says how
                vx, vyaw = legal_walk(prog.twist[0], prog.twist[1])
        else:
            vx, vyaw = legal_walk(prog.twist[0], prog.twist[1], approach=True)
    elif body_ok and (
        spec is None or spec.locomotes or cmd.vx > 0.0 or cmd.docking or cmd.back_arc
    ):
        # at the nest the brain may back out (contract 23:58: MotorCommand.docking, ≥ −0.40), and
        # beside a thing it investigated take curiosity's back_arc (MotorCommand.back_arc)
        vx, vyaw = legal_walk(
            cmd.vx, cmd.vyaw, approach=cmd.approach, docking=cmd.docking, back_arc=cmd.back_arc
        )
    if 0.0 < vx < ARC_VX_MIN and not dock_twist:
        vx = (
            ARC_VX_MIN  # pet:probe: 0.30 steps only 40 % (bam) / 80 % (mjcf) of starts; 0.35 always
        )
    reversing = vx < 0.0  # (docking, or a back_arc: the brain times these itself)
    if abs(vyaw) > 0.3 and vx > APPROACH_VX and not dock_twist:
        vx = APPROACH_VX  # an arc: never faster than 0.40
        notes.append("arc≤0.40")
    # the body's own rhythm limits: a burst shorter than the body needs to get going and a stop
    # shorter than its legs' tail are not a walk and not a pause (pet:probe: bursts < 0.4 s
    # barely move, pauses < 0.5 s are coasted through), so the motor holds a burst ≥ 0.8 s and
    # a pause ≥ 0.45 s — unless a reflex, the posture or sleep stops the body at once
    still_moment = moment_state is not None and moment_state.behaviour == "still"
    hard_stop = (
        not body_ok
        or cmd.asleep
        or still_moment
        or (cmd.reflex in R.HALTS and cmd.reflex not in ("withdraw",))
        # (fid:motor) a freeze is a freeze whether a reflex or the brain chose it: the legs stop
        # at once, never "a burst ≥ 0.8 s first"
        or prog_state.behaviour in STOP_ACTS
    )
    walk_t0, stop_t0, last_twist = state.walk_t0, state.stop_t0, state.last_twist
    if hard_stop:
        vx, vyaw = 0.0, 0.0
        if walk_t0 >= 0:
            walk_t0, stop_t0 = -1, t
    elif reversing or dock_twist:
        pass
    elif vx > 0.0 and walk_t0 < 0:
        if t - stop_t0 < MIN_PAUSE_MS:
            vx, vyaw = 0.0, 0.0
            notes.append("pause≥0.45s")
        else:
            walk_t0 = t
    elif vx == 0.0 and walk_t0 >= 0:
        if t - walk_t0 < MIN_BURST_MS:
            vx, vyaw = last_twist
            notes.append("burst≥0.8s")
        else:
            walk_t0, stop_t0 = -1, t
    last_twist = (vx, vyaw) if vx > 0.0 else last_twist
    # course keeping, learned like a cerebellum calibrates a reflex: every body drifts right
    # when told to walk straight (pet:probe: −10…−15 °/s, stronger the lighter the body), so
    # while the intended curvature is small the motor learns a yaw trim from the gyro's
    # low-passed yaw rate and adds it to every walk (the arc rule still reads the intent)
    trim, drift = state.trim, state.drift
    if vx > 0.0 and walking and not dock_twist:
        drift = S.leaky(drift, body.gyro[2], TRIM_TAU_S, dt)
        steady = walk_t0 >= 0 and t - walk_t0 >= TRIM_SETTLE_MS
        if steady and abs(vyaw) < TRIM_STRAIGHT:
            trim = S.clamp(trim - TRIM_GAIN * profile.trim_gain * drift * dt, -TRIM_MAX, TRIM_MAX)
    if vx > 0.0 and not dock_twist:
        vyaw = S.clamp(vyaw + trim, -1.0, 1.0)

    if still_moment and cmd.reflex not in R.HALTS and state.t_ms >= 0:
        # (fid:fix 2026-09-28) 「定格」: the head holds the pose it was last sent — the gaze, the
        # walk's look into its turn and its sway would otherwise swing it on (headless: 0/12
        # 定格 held the head within 3°, 18–30° of motion inside, most while it had been turning)
        pose = replace(state.pose, mouth=0.0)

    # 8. the envelope: the A100-validated seated envelope, or — walking or about to (the body
    # link lets the legs start only once the measured head is inside it) — the walking one;
    # owner A2's head-down approach at ≤ 0.40 m/s
    walk_mode = walking or vx > 0.0
    # a margin inside the link's walking bound, so the measured head (which lags and may
    # overshoot a little) sits inside it and the legs are not held back ("head" gate, live)
    env = profile.walk_env * WALK_MARGIN if walk_mode else profile.stand_env
    # the yaw's own range: the validated ±1.0 rad (walking: WALK_MARGIN inside it), the look-up
    # while walking to WALK_HP_UP (fid:motor: orienting turns and the startle's snap need them)
    yaw_env = YAW_CLIP * (WALK_MARGIN if walk_mode else 1.0)
    hp_high = max(env, WALK_HP_UP) if walk_mode else env
    if body.posture == "transition":
        # getting up or sitting down is a balance manoeuvre: the head stays near HOME whatever
        # asks for it (live :8891, neuro:brain: a startle head snap during a rise tipped the duck)
        env = min(env, TRANSITION_ENV)
        yaw_env = hp_high = env
        notes.append("transition")
    np_low, np_high = -env, env
    hp_low = -env
    if walk_mode and cmd.approach and vx <= APPROACH_VX + 1e-9:
        np_high = max(np_high, profile.head_down_np)
        hp_low = min(hp_low, -profile.head_down_np)
    clamped = HeadPose(
        np=S.clamp(pose.np, np_low, np_high),
        hp=S.clamp(pose.hp, hp_low, hp_high),
        hy=S.clamp(pose.hy, -yaw_env, yaw_env),
        hr=S.clamp(pose.hr, -env, env),
        mouth=S.clamp(pose.mouth, 0.0, 1.0),
    )
    if clamped != pose:
        notes.append("envelope")
    pose = clamped
    goal = pose_to_offsets(pose)

    # 9. robotd's head smoothing, pre-compensated (an efference copy of its EMA, inverted) —
    # the identity at head_alpha 1.0 (fid:motor: robotd then passes the command straight on) —
    # and the sent command's rate limited: seated HEAD_SLEW_SIT (A100: large fast seated moves
    # at 1.0 strain the head servos and shuffle the feet), else HEAD_SLEW
    ticks = max(1, int(round(dt * 1000.0 / ROBOTD_TICK_MS))) if state.t_ms >= 0 else 1
    alpha = 1.0 - (1.0 - S.clamp(profile.head_alpha, 0.01, 1.0)) ** ticks
    lead_k = S.clamp(profile.lead, 0.0, 1.0)
    seated_head = body.posture in ("sit", "transition") or not standing
    slew = (profile.head_slew_sit if seated_head else profile.head_slew) * max(dt, 0.02)
    sent: list[float] = []
    ema: list[float] = []
    for i in range(4):
        y = state.ema[i]
        want = y + (goal[i] - y) / alpha
        u = goal[i] + lead_k * (want - goal[i])
        u = S.clamp(u, goal[i] - LEAD_MAX[i], goal[i] + LEAD_MAX[i])
        u = S.clamp(u, -COMMAND_MAX[i], COMMAND_MAX[i])
        if state.t_ms >= 0:
            u = S.clamp(u, state.sent[i] - slew, state.sent[i] + slew)
        sent.append(u)
        ema.append(y + alpha * (u - y))

    # 10. posture: the program's cue latches for its bout; else the behaviour's; else the brain's
    posture = state.posture
    posture_bout = state.posture_bout
    if (cmd.docking or cmd.behaviour == "return_dock") and cmd.posture in ("sit", "stand"):
        # (dock:build) the docking program's own stand and sit, and home on its contacts it
        # sits: the brain's posture, over the act's default need to stand
        posture = cmd.posture
    elif prog.posture is not None:  # the program's cue (latched for its bout) or its need
        posture, posture_bout = prog.posture, prog_state.bout
    elif prog_state.behaviour in CUED_POSTURE:
        pass  # before its cue (the head's preparation) or latched after it
    elif cmd.posture in ("sit", "stand"):
        posture = cmd.posture
    if cmd.asleep:
        posture = "sit"
    if body.fallen:
        notes.append("fallen")

    new_state = MotorState(
        t_ms=t,
        program=prog_state,
        rng=rng,
        ema=(ema[0], ema[1], ema[2], ema[3]),
        sent=(sent[0], sent[1], sent[2], sent[3]),
        gaze=(gy, gp),
        ou=(ou[0], ou[1], ou[2], ou[3]),
        micro=micro,
        micro_next=micro_next,
        breath=breath,
        gait_osc=gait_osc,
        sway=sway % (2.0 * math.pi),
        sway_cycles=sway_cycles % 6,
        gait_prev=gait_now,
        gait_hz=gait_hz,
        gait_wrap_ms=gait_wrap_ms,
        trunk=(lp_pitch, lp_roll, base_pitch, base_roll),
        vcr=(yaw_ref, rate_lp, roll_ref),
        trim=trim,
        drift=drift,
        walk_t0=walk_t0,
        stop_t0=stop_t0,
        last_twist=last_twist,
        posture=posture,
        posture_bout=posture_bout,
        vocal_seq=cmd.vocal_seq,
        sound_ms=sound_ms,
        sound_level=sound_level,
        pose=pose,
        moment=moment_state,
        moment_seq=cmd.moment_seq,
    )
    return new_state, MotorOutput(
        t_ms=t,
        head=(sent[0], sent[1], sent[2], sent[3]),
        head_goal=goal,
        pose=pose,
        mouth=pose.mouth * MOUTH_ENVELOPE,
        vx=vx,
        vyaw=vyaw,
        posture=posture,
        sound=sound,
        sound_level=round(level, 3),
        program=prog_state.behaviour,
        label=prog.label if moment_state is None else f"{prog.label}（{moment_state.label}）",
        unit=prog.unit,
        walking=walking,
        notes=tuple(notes),
        trim=round(trim, 4),
    )


class MotorSystem:
    """A convenience holder for the driver: ``MotorSystem(profile).step(cmd, frame)``."""

    def __init__(self, profile: MotorProfile, seed: int = 0) -> None:
        self.profile = profile
        self.state = MotorState(rng=S.seed_from("neuro.motor", seed or profile.seed))

    def step(self, cmd: MotorCommand, frame: SensorFrame) -> MotorOutput:
        self.state, out = step(self.profile, self.state, cmd, frame)
        return out


__all__ = [
    "COMMAND_MAX",
    "ENVELOPE",
    "GaitState",
    "LEAD_MAX",
    "LocoInput",
    "LocoOut",
    "LocoParams",
    "LocoState",
    "MOTOR_VERSION",
    "MotorOutput",
    "MotorProfile",
    "MotorState",
    "MotorSystem",
    "gait_step",
    "loco_params_from",
    "loco_step",
    "offsets_to_pose",
    "pose_to_offsets",
    "profile_from_genome",
    "profile_from_style",
    "step",
]
