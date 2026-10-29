"""bodylink: the neural duck's motor output onto the real body path, under duck-body's safety.

Pure and clock-injected: one :meth:`BodyLink.tick` per robotd ``robot.state`` tick (50 Hz; in
the live simulator the physics steps in lockstep with robotd's writes, so this *is* lockstep
with the physics). The live driver (:mod:`duckmind.simweb.neuro_live`) owns the sockets and
sends what comes out; the same code can run against the headless rig.

What goes in is a :class:`BodyRequest` — what the motor programs want from the body this tick
(the twist, the head offsets from robotd's HOME, the beak, the posture, and whether the walk is an
approach or a docking reverse). What comes out (:class:`LinkOut`) is the robotd intents that may
be sent, and why the request was changed (``gate``), from phase 3's duck-body pieces:

* **falls → righting** (spec §5.4, owner B1/B2): :class:`~duckmind.duckbody.proprio.Proprio`'s
  fall episode and :class:`~duckmind.duckbody.righting.Righting` (``sit_toggle`` into velstand,
  the sit-2-s-stand second try). While an episode is open the body holds still and the head goes
  HOME; posture requests wait.
* **pickup** (owner B3/B4): the duck's own sense of being lifted (proprio, not the world's
  truth): no walking while unsure or lifted, and it sits once it is sure.
* **a lift claim that does not end** (docs/design/neural-duck/DOCK-FALL-2026-10-11.md §7.6): a seated duck resting partly on a ledge (the
  dock's guide, a door sill) reads legs as light as a hand's, and nothing ever sets it down. A claim that has stood
  ``retest_after_s`` while seated is re-tested from the duck's own senses, at most once per ``retest_every_s``: the head goes up (the
  seated envelope's edge) and the legs answer or not (:class:`~duckmind.neuro.liftcheck.FloorProbe`). Legs that carry the trunk twice
  in a row end the claim as a set-down does; legs that carry nothing leave it standing. A duck held in a hand is never released by it.
  A trunk tilted past proprio's own lift limit (a duck lying on its beak at 62°, its neck pinned by the trunk's weight, cannot be asked for the
  HOME pose) is read with the head at rest instead, and counts as borne by something only when the trunk itself also rocks as the head's
  weight moves (a hand does not let it); the fall its claim had held back begins when the claim ends
  (docs/design/neural-duck/LIFT-LOCK-RIGHTING-2026-10-18.md).
* **battery**: no standing up or walking under ``content.LOCOMOTION["min_battery"]`` (30 %).
* **reflex stops** (spec §5.3, :func:`~duckmind.duckbody.reflex.verdict`): a drop ahead stops
  the walk; a touch, a stall (proprio's ``blocked``), a stumble or a ToF return within 30 cm stops
  straight walking — after a short hold only an arc (|vyaw| ≥ ``veer_vyaw``) may go on, turning
  away from the side that was hit when proprio knows it, so the duck turns away instead of
  standing nose to the wall forever (a prototype choice; bumping a wall is harmless, spec §5.3).
  Proprio's ``brushed`` is not used: under the neural motor's continuously changing steering it
  fired 7–8 times a minute, 67 of 72 away from any wall (live 2026-09-28), so it is not a contact.
* **walking rules** (robotd, velstand; owner A3): speeds are 0 or 0.35–0.55 m/s (0.40 when
  approaching a thing), turns only while stepping (an in-place turn becomes the slowest arc:
  owner "turns are arcs"), no reversing except a docking reverse (≥ −0.40, owner phase 3).
  **Docking** (dock:build 2026-09-28, :mod:`duckmind.neuro.dock`): the choreography's own twist
  goes through as phase 3 measured it — forward from 0.30 (its U-turn and tight arcs), up to
  0.40, reversing to −0.40 — and while it backs in, a stall is the dock's face (the choreography
  says "blocked = arrived") and a ToF return ahead is behind where it goes: neither stops nor
  turns it away; after any reflex stop while docking only the hold applies (no arcs-only turning
  away: the choreography knows where it goes). Everything else is the same.
* **the forbidden-command guard** (:func:`~duckmind.duckbody.reflex.check_intent`): every intent
  passes it; a refused move becomes a zero move.
* **the deadman**: the twist is streamed every tick while non-zero
  (:class:`~duckmind.duckbody.robotd_client.MoveStreamer`: robotd zeroes a twist that is not
  refreshed within 500 ms), the head and the beak on change and every 0.5 s.

**One deliberate change to phase 3's guard (owner feedback 2026-09-27, 「不像寵物」): the head
stays alive while walking.** duck-body's rule 3 forbids walking unless the head sits at HOME
(settled 0.3 s). The neural duck walks with a moving head (orienting, looking around, looking
down), inside the walking envelope :data:`WALK_HEAD`: a measured head inside it counts as
"settled" for the guard, and commanded offsets are clamped into it while walking. The envelope is
MEASURED (2026-09-27, headless, official allcollisions MJCF + shipped velstand, both servo
models, floor friction 0.5–1.0, mass ±8 %): pet:probe 0/1,080 walks fell with the head anywhere
in yaw ±1.0, look-up to head_pitch −0.35 and look-down to (neck −0.5, head_pitch +0.7); pet:body
0/720 (yaw ±0.3/±0.6/+1.0, look-up, scanning) — the rule was a perception rule, not a stability
one. Posture changes still keep the head at HOME (a rise is a balance manoeuvre).

Floats in the prototype; phase 5 converts to fixed point (the pieces from duck-body already are).
"""

from __future__ import annotations

import math
import os
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from duckmind.duckbody import protocol as P
from duckmind.duckbody.proprio import Proprio, ProprioConfig, ProprioOut
from duckmind.duckbody.reflex import GuardContext, Limits, check_intent, verdict
from duckmind.duckbody.righting import Righting, RightingConfig, RightingOut
from duckmind.duckbody.robotd_client import JOINT_NAMES, MoveStreamer, RobotState
from duckmind.duckcore import content
from duckmind.i18n import msg
from duckmind.neuro.liftcheck import FloorProbe
from duckmind.skilltrain import guard as SG

# the ToF stop in a walk up to a thing (the owner's A1: the act's own standoff, never under 50 mm): the
# ToF is 0.1 m or so ahead of the trunk, the beak reaches 0.096 m
CURIO_STOP_MM = 60
NS = 1_000_000_000
MS = 1_000_000
# the roll (the flip trick): held as a lie-down on purpose from the skill's start until the duck has been upright again this long …
ROLL_UP_S = 1.0
ROLL_UP_TILT_DEG = 25.0  # … (proprio's own "up")
ROLL_HOLD_MAX_S = 30.0  # … at most this long after the last tick robotd ran the skill (a duck still down then falls as any does)
Offsets = tuple[
    float, float, float, float
]  # robot.head: neck_pitch, head_pitch, head_yaw, head_roll
Bounds = tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]
HOME: Offsets = (0.0, 0.0, 0.0, 0.0)
# the proprio edges a duck that lies down on purpose does not claim (bodylink.BodyRequest.lying)
LYING_MASKED = (
    "fell",
    "lifted",
    "unloaded",
    "unloaded_clear",
    "set_down",
    "blocked",
    "brushed",
    "stumbled",
)
HEAD_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")


def _envelope(scale: float = 1.0) -> Bounds:
    """± content.SEATED_ENVELOPE per head joint (the clip system's validated targets), rad."""
    env = content.SEATED_ENVELOPE
    return (
        (-env["neck_pitch"] * scale, env["neck_pitch"] * scale),
        (-env["head_pitch"] * scale, env["head_pitch"] * scale),
        (-env["head_yaw"] * scale, env["head_yaw"] * scale),
        (-env["head_roll"] * scale, env["head_roll"] * scale),
    )


# The walking head envelope (offsets from HOME, rad), measured safe (module doc: pet:probe
# 0/1,080, pet:body 0/720 walks fell): neck down to −0.5 (standing, the walking network owns the
# neck: a pull-back is not performed, pet:body), head_pitch −0.35 (up) … +0.7 (down), yaw ±1.0,
# roll the trained ±0.31.
WALK_HEAD: Bounds = ((-0.50, 0.25), (-0.35, 0.70), (-1.0, 1.0), (-0.31, 0.31))
STILL_HEAD: Bounds = _envelope()
# An emotion's head (Pollen's microduck_emotions, 2026-10-08): further than the walking envelope and the seated range,
# as the workshop played it (neck down to -1.5 rad, the head 1.0 rad, yaw 1.0): neck_pitch, head_pitch, head_yaw, head_roll
EMOTION_HEAD: Bounds = ((-1.60, 0.30), (-1.10, 1.10), (-1.10, 1.10), (-0.60, 0.60))


# The simulator's servo command delay (``bodyhost.COMMAND_DELAY_ENV``, 20 ms since 2026-10-05) makes the walking network
# walk FASTER at the pace commands (+21 % ground speed at 0.35–0.50 m/s: measured headless and on the live body, same
# network) and stop LATER (0.021 → 0.035 m after a stop at 0.40), while every distance rule of the brain (approach and
# arrival, the stuck and unstick models, the speeds of the designed paces) was tuned without it. The link therefore asks for
# the command that gives, with the delay, the ground speed the pace was designed to have: ``(command, command that walks
# as that command did at 0 ms)``, measured on the walking network with a 20 ms delay (docs/design/neural-duck/
# BUGS-2026-10-09.md, ``gait_grid.json``). From 0.525 up the delayed gait is the slower one (0.2435 against 0.2499 at
# 0.525, 0.258 against 0.296 at 0.55): no compensation there, a flight is never slowed. The table is for 20 ms; a shorter
# delay takes the same share of the correction (delay / 20), a longer one the whole.
DELAY_PACE_TABLE: tuple[tuple[float, float], ...] = (
    (0.35, 0.309),
    (0.375, 0.330),
    (0.40, 0.359),
    (0.425, 0.374),
    (0.45, 0.394),
    (0.475, 0.414),
    (0.50, 0.443),
    (0.525, 0.525),
)
DELAY_TABLE_MS = 20.0

# The hand-back of a flight (2026-10-09). The panic skill ends when the brain stops asking for the flight (its motor's
# burst ends, a look, a reflex) and robotd hands the body back to v5 with ``skill_abort``. With the walk request at zero
# that is v5 under a ZERO command at 0.4-0.6 m/s with the trunk leaned 15 degrees: it tipped (live, fresh boots after a bang:
# 3 of 16 fell, 1 of 8 more in the next batch; every one at the abort of a flight whose request had just dropped to 0).
# The panic gate's own hand-off check (H1) hands back to v5 at 0.40 and walks on; the link therefore keeps asking v5 for
# HANDBACK_VX for HANDBACK_S after a flight that ends on its own, and only then stops it as any walk is stopped. A reflex
# stop, a lift, a fall or a low battery still stops it at once.
HANDBACK_VX = 0.40
HANDBACK_S = 0.6
# … and the legs of a flight that has just ended still swing at the flight's tempo (10.8 rad/s at the hand-back, 0.5 m/s: live, a fresh
# boot after a bang): the "legs swinging free" test (``_airborne``, 10 rad/s) read that as a pick-up, the stop rule sat the duck down at
# 0.54 m/s and it tipped (tilt 56 degrees, fell 1.5 s later). For AFTER_FLIGHT_S after the last tick of a flight the leg speed is not
# read as a pick-up; the lift sensors (``view.lifted``, ``unloaded``) still are.
AFTER_FLIGHT_S = 1.2
# … and a stumble in the middle of a flight (the flight's own lean is 11-15 degrees, so a footfall's jolt reads as a trip): stopping
# dead at 0.3-0.5 m/s tipped the duck over (live, 2 of 20 fresh boots after a bang, 9.0 s and 4.3 s into the flight, both at the stumble
# reflex's zero twist). A stumble in a flight ends it (skill_abort) and v5 walks on at HANDBACK_VX for HANDBACK_S as after any flight.
# … and the flights that fell were the ones that ended EARLY (live fresh boots after a bang, 54 flights: the two that were handed back
# after 0.49 and 0.63 s, at 0.18 and 0.27 m/s, still in the run-up, tilted 5-11 degrees by the lean it takes, fell; 51 flights of 1.1-1.4 s
# at 0.6 m/s did not): the brain's request ended in the run-up (its stabilise reflex reads the lean as a lurch, a look, a pause). A flight
# the link has started is therefore held for MIN_FLIGHT_S whatever the request does; the stop rules (reflex stop, lift, fall, low battery) and
# the tilt watch still end it at once.
MIN_FLIGHT_S = 1.0


def command_delay_ms() -> float:
    """The servo command delay this process's body runs with (0 on a real robot: the variable is the simulator's)."""
    try:
        return max(
            0.0, float((os.environ.get("DUCKMIND_BODY_COMMAND_DELAY_MS") or "0").strip() or 0)
        )
    except ValueError:
        return 0.0


# … and it turns faster too: the same yaw command gives ~25 % more turning rate (cruise at 0.35 m/s, 20 ms against 0 ms:
# command ±1.0 → 0.59 / −0.84 rad/s against 0.42 / −0.73, ±0.6 → 0.19 / −0.57 against 0.14 / −0.53), which tightens the
# arcs the brain turns away from things with. A yaw command is scaled by YAW_DELAY_SCALE (1 / 1.25) at the full table delay.
YAW_DELAY_SCALE = 0.8


def delay_compensated_vyaw(vyaw: float, delay_ms: float) -> float:
    """``vyaw`` as the command that turns, with a servo command delay of ``delay_ms``, at the rate it turns at without it."""
    if delay_ms <= 0.0:
        return vyaw
    share = min(1.0, delay_ms / DELAY_TABLE_MS)
    return vyaw * (1.0 + share * (YAW_DELAY_SCALE - 1.0))


def delay_compensated_vx(vx: float, delay_ms: float) -> float:
    """The forward command that walks, with a servo command delay of ``delay_ms``, at the ground speed ``vx`` walks at
    without it (see :data:`DELAY_PACE_TABLE`). Commands under the table's first row or from its last row on, and a
    zero delay, are as they are."""
    if delay_ms <= 0.0 or vx < DELAY_PACE_TABLE[0][0] or vx >= DELAY_PACE_TABLE[-1][0]:
        return vx
    share = min(1.0, delay_ms / DELAY_TABLE_MS)
    for (a, fa), (b, fb) in zip(DELAY_PACE_TABLE, DELAY_PACE_TABLE[1:], strict=False):
        if a <= vx <= b:
            target = fa + (fb - fa) * (vx - a) / (b - a)
            return vx + share * (target - vx)
    return vx


@dataclass(frozen=True)
class LinkConfig:
    """bodylink's numbers (m/s, rad/s, rad, s)."""

    min_vx: float = 0.35  # walkexec min_vx: 0.30 steps on the spot under the XL330 model
    max_vx: float = 0.55  # owner A3: the normal walking cap
    approach_vx: float = 0.40  # owner A3: toward a specific thing
    reverse_vx: float = -0.40  # owner phase 3: reversing only while docking
    dock_min_vx: float = 0.30  # (dock:build) the docking choreography's U-turn and tight arcs
    pause_below: float = 0.15  # a wanted speed under this is a pause, not a walk
    turn_below: float = 0.05  # |vyaw| under this is straight
    vyaw_max: float = 1.0
    still_head: Bounds = STILL_HEAD  # commanded offsets while not walking
    emotion_head: Bounds = EMOTION_HEAD  # … while an emotion plays (BodyRequest.emotion)
    walk_head: Bounds = WALK_HEAD  # the MEASURED head must be inside this to walk
    walk_cmd_head: Bounds | None = None  # commanded offsets while walking (None: walk_head)
    head_start_tol: float = 0.03  # the measured head must be this close to the walking envelope
    head_keep_tol: float = 0.10  # … and stay this close while walking (hysteresis)
    head_resend_s: float = 0.5
    head_change: float = 0.002  # rad: a smaller change is not re-sent before the resend period
    mouth_max: float = 0.5  # content.SEATED_ENVELOPE["mouth"]: the validated beak opening
    mouth_resend_s: float = 0.5
    posture_retry_s: float = 3.0  # re-send sit_toggle when the posture has not changed by then
    posture_head_s: float = 2.0  # the head stays HOME this long after a sit_toggle
    # A measured head this close to HOME counts as at HOME (neck_pitch, head_pitch, head_yaw,
    # head_roll; rad). The neck is loose: standing, the walking network holds it ~0.13 rad
    # forward of HOME whatever robot.head asks (live 2026-09-27; pet:body "neck X standing"); the
    # head pitch it holds ~0.10 rad down (live :8783, 2026-10-24: 0.105 steady while walking, 0.02
    # still), so the pitch's bound is 0.13 — a head 0.10 down waited the whole home_wait_s 3 s and
    # a sit-down took 3.6 s (over the spec's 3 s). A head looking far down still waits.
    home_tol: tuple[float, float, float, float] = (0.25, 0.13, 0.10, 0.10)
    home_settle_s: float = 0.3  # … for this long before a posture change (rule 3)
    home_wait_s: float = 3.0  # … but a posture change waits no longer than this for it
    battery_min_pct: int = int(str(content.LOCOMOTION["min_battery"])) // 100  # 30 %
    command_delay_ms: float | None = (
        None  # None: the process's own (:func:`command_delay_ms`), read once per link
    )
    reflex_hold_s: float = 0.5  # a reflex stop holds the body still this long
    # proprio's "stumbled" (a gyro spike with doubled leg load) stops the legs only when the
    # trunk really tips (the IMU): live 2026-09-28 it fired 107 times in 35 min for the slow,
    # shy duck (every ~3 s of walking, max tilt 7°) and cut its walks into 0.9-s spurts; a trip
    # tilts the trunk past this within the window
    stumble_tilt_deg: float = 10.0
    stumble_window_s: float = 0.5
    # … then only arcs until the heading has turned this far (odometry yaw) — a consequence,
    # not a set time (review 2026-09-28) — with a long cap only for safety
    veer_turn_rad: float = 0.5
    veer_max_s: float = 5.0
    veer_vyaw: float = 0.5  # an arc at least this tight counts as turning away
    # (2026-10-11, WALL-LOOPS-2026-10-11.md) with the table's quick-turn entries, a duck held at a wall (the ToF stop, a touch, a stall)
    # that is asked to walk on turns away ON THE SPOT with the quick turn instead of sweeping a forward arc back into the wall: the
    # turn runs until it has turned ``wall_pivot_min_rad`` and the ToF ahead is clear (``wall_pivot_clear_mm``), at most
    # ``wall_pivot_max_rad`` / ``wall_pivot_max_s``. At most ``wall_pivot_budget`` of them per ``wall_pivot_window_s`` within
    # ``wall_pivot_radius_m`` (then the older stop-and-veer logic is all there is). Without the table's entries nothing changes.
    wall_pivot: bool = True
    wall_pivot_min_rad: float = 1.0
    wall_pivot_max_rad: float = 2.2
    wall_pivot_max_s: float = 1.8
    wall_pivot_clear_mm: int = 600
    wall_pivot_pin_s: float = 0.7
    # a walk toward a goal (an approach to a thing, the dock) is not turned away from a wall at once: only when the duck has been held there this long
    wall_pivot_long_s: float = 2.0
    wall_pivot_budget: int = 3
    wall_pivot_window_s: float = 25.0
    wall_pivot_radius_m: float = 0.7
    wall_pivot_cooldown_s: float = 0.3
    wall_pivot_start_s: float = (
        0.6  # robotd must show the quick turn this soon after the ask, else it was refused
    )
    # A lift claim this soon after pushing into something (proprio's blocked/stumbled, a ToF
    # touch, a stall stop) must hold this long before it is believed: pressing into a wall
    # unloads the legs for a moment the way a pickup does (live 2026-09-27: blocked at 39.1 s,
    # a false "lifted" at 39.9 s; the sit and rise that followed ended in a fall).
    lift_after_contact_s: float = 3.0
    lift_confirm_s: float = 1.5
    # A lift claim that has stood this long while the duck sits is re-tested (FloorProbe), at most once per ``retest_every_s``; the
    # claim ends when ``retest_votes`` re-tests in a row find that something under the duck bears its weight. A refuted claim is not
    # made again until the seat changes (a posture change, a fall) or a re-test finds the legs carry nothing (a hand has lifted it).
    retest_after_s: float = 10.0
    retest_every_s: float = 20.0
    retest_votes: int = 2
    # (fid:fix 2026-09-28) Picked up while stepping, the walking network keeps stepping in the
    # air — the legs swing free, much faster than any step on a floor — and the head whips (the
    # review, live: up to 16.5 rad/s for ~2 s while the walk command went on; proprio's lift
    # claim came 2.6–3.1 s after lift-off: its leg-load test cannot see a stepping duck lifted).
    # The legs' own speed sees it at once: headless (the whole duck, the world's hand, 24 walking
    # pickups, the same seeds before/after) it fired in 15 of the 19 lifted mid-step, a median
    # 0.26 s after lift-off (the other 4 had just been told to stop); the head's peak speed fell
    # from 16.6 rad/s (median; 0.57 s above 8) to 4.4 (max 6.8, never above 8); no false alarm in
    # 720 s of free walking, stopping, sitting down and getting up (4 ducks) nor in 48 pushes.
    # Then: no twist, the head HOME and sit (owner B4: held in the air it sits with its legs
    # tucked) — at once.
    air_leg_speed: float = 10.0  # rad/s: the norm of the ten leg joints' speeds
    air_ticks: int = 3  # … this many ticks in a row
    air_gyro: float = 1.5  # rad/s: … with the trunk not thrown about (a fall or a trip rocks it)
    air_hold_s: float = 0.8  # it stays "in the air" this long after the last such tick
    limits: Limits = field(default_factory=Limits)
    proprio: ProprioConfig = field(default_factory=ProprioConfig)
    righting: RightingConfig = field(default_factory=RightingConfig)


DEFAULT = LinkConfig()


@dataclass(frozen=True)
class BodyRequest:
    """What the motor programs want from the body this tick.

    ``vx``/``vyaw`` m/s and rad/s (+ = left); ``head`` robot.head offsets from robotd's v5 HOME
    (neck_pitch, head_pitch, head_yaw, head_roll; rad; None = HOME); ``mouth`` robot.mouth's
    ``open`` (0..1 of the beak's travel); ``posture`` "stand" | "sit" | None (keep); ``approach``
    a walk toward a specific thing (cap 0.40, owner A3); ``docking`` may reverse (≥ −0.40)."""

    vx: float = 0.0
    vyaw: float = 0.0
    head: Offsets | None = None
    mouth: float = 0.0
    posture: str | None = None
    approach: bool = False
    docking: bool = False
    back_arc: bool = False  # curiosity's reverse arc beside a thing (owner phase 3)
    skill: str | None = None  # a trained one-shot skill to start (Stage 1; skilltrain.guard)
    # a flight in panic (flee / retreat): with the table's panic entries loaded the link runs it as robotd's
    # panic skill (held while this stays set, handed back to v5 with skill_abort when it stops or a stop rule
    # fires); otherwise v5 walks it at ``vx`` (the brain's panic range, <= 0.55)
    panic: bool = False
    # (2026-10-09) pivot on the spot toward this side (+ left, - right, 0 none): with the table's quick-turn entries the link runs it as
    # robotd's quick-turn skill (held while this stays set, handed back to v5 with skill_abort the tick it clears); the twist above
    # (the brain's arc toward it) is what v5 walks otherwise and after the pivot
    pivot: float = 0.0
    skill_manual: bool = False  # … pressed on the page: the wear caps do not apply (guard.request)
    # (startle-hop, 2026-10-10) the skill asked is the hop-back that answers a scare: it waits for the stop (trunk slow, upright) instead
    # of the posture's settle time, counts at half against the hourly caps (guard.request ``chain``), and is not made on a low battery
    skill_chain: bool = False
    # the thing a curiosity walk goes up to — (bearing mrad, range mm) from the trunk, the brain's
    # current belief of it: the owner's A1 (the ToF stop is relaxed in a curiosity walk toward the
    # thing it has just seen; reflex.relaxed_near) — the ToF return must lie where the thing is
    toward: tuple[int, int] | None = None
    # an emotion plays (neuro.emotions): the head may go to the emotion envelope, the standing body may bow by ``pose``
    # (robot.pose pitch, rad; None: the pose is not driven). Nothing else sets either.
    emotion: bool = False
    pose: float | None = None
    # (play dead, 2026-10-11) the duck lies down ON PURPOSE (a robotd scripted pose, then it gets itself up): the fall, the lift, the
    # legs-light claims and the righting that a duck on its back would start are held back — it is no fall and no pick-up — and
    # the beak stays live. When the request stops saying so and the duck is still down, the ordinary fall (and its righting) begins.
    lying: bool = False


@dataclass(frozen=True)
class LinkOut:
    """One tick's result: the intents to send (in order), the body sense, the events that
    happened (duck-body percept bodies: ``body.fell``, ``body.righting``, proprio edges), why
    the request was changed (``gate`` codes, see :data:`GATE_ZH`) and what was sent."""

    intents: tuple[P.Intent, ...]
    prox: ProprioOut
    events: tuple[dict[str, Any], ...] = ()
    gate: tuple[str, ...] = ()
    twist: tuple[float, float] = (0.0, 0.0)
    head: Offsets = HOME
    mouth: float = 0.0
    posture_want: str | None = None
    righting: str = "idle"
    refused: tuple[tuple[str, str], ...] = ()


GATE_ZH: dict[str, str] = {
    "fallen": msg("neuro.gate.fallen"),
    "lifted": msg("neuro.gate.lifted"),
    "unloaded": msg("neuro.gate.unloaded"),
    "battery": msg("neuro.gate.battery"),
    "posture": msg("neuro.gate.posture"),
    "posture_head": msg("neuro.gate.posture_head"),
    "head": msg("neuro.gate.head"),
    "reflex_drop": msg("neuro.gate.reflex_drop"),
    "reflex_touch": msg("neuro.gate.reflex_touch"),
    "reflex_near": msg("neuro.gate.reflex_near"),
    "stall": msg("neuro.gate.stall"),
    "stumble": msg("neuro.gate.stumble"),
    "airborne": msg("neuro.gate.airborne"),
    "veer": msg("neuro.gate.veer"),
    "arc": msg("neuro.gate.arc"),
    "min_speed": msg("neuro.gate.min_speed"),
    "cap": msg("neuro.gate.cap"),
    "no_reverse": msg("neuro.gate.no_reverse"),
    "back_arc": msg("neuro.gate.back_arc"),
    "refused": msg("neuro.gate.refused"),
    "skill": msg("neuro.gate.skill"),
}


LEG_JOINTS = tuple(
    i for i, name in enumerate(JOINT_NAMES) if any(k in name for k in ("hip", "knee", "ankle"))
)


def _leg_speed(state: RobotState) -> float | None:
    """The norm of the ten leg joints' measured speeds (rad/s), None when robotd sends none."""
    vel = state.velocities
    if not vel:
        return None
    return math.sqrt(sum(vel[i] * vel[i] for i in LEG_JOINTS if i < len(vel)))


def _away(side: str) -> int:
    """The turn that leads away from a contact on ``side`` (+1 left, −1 right, 0 unknown)."""
    return -1 if side == "left" else (1 if side == "right" else 0)


def clamp_head(head: Offsets | None, bounds: Bounds) -> Offsets:
    """``head`` (None: HOME) clamped joint by joint into ``bounds``."""
    if head is None:
        return HOME
    out = [min(hi, max(lo, float(v))) for v, (lo, hi) in zip(head, bounds, strict=True)]
    return (out[0], out[1], out[2], out[3])


def head_inside(measured: Sequence[float], bounds: Bounds, tol: float) -> bool:
    """Whether measured head offsets lie within ``bounds`` widened by ``tol``."""
    return all(lo - tol <= v <= hi + tol for v, (lo, hi) in zip(measured, bounds, strict=True))


def shape_twist(
    vx: float,
    vyaw: float,
    *,
    approach: bool = False,
    docking: bool = False,
    back_arc: bool = False,
    cfg: LinkConfig = DEFAULT,
) -> tuple[float, float, tuple[str, ...]]:
    """The nearest twist robotd and velstand accept (and why it changed): 0 or
    ``min_vx``…cap forward, an in-place turn becomes the slowest arc, reversing only docking or
    as curiosity's back_arc (a reverse ARC, |vyaw| ≥ 0.5)."""
    notes: list[str] = []
    vyaw = max(-cfg.vyaw_max, min(cfg.vyaw_max, float(vyaw)))
    vx = float(vx)
    if vx < 0.0 and docking:
        return max(cfg.reverse_vx, vx), vyaw, ()
    if vx > 0.0 and docking:
        # (dock:build) the docking choreography's own forward twist, as phase 3 measured it
        if vx < cfg.pause_below:
            return 0.0, 0.0, ()
        return min(cfg.approach_vx, max(cfg.dock_min_vx, vx)), vyaw, ()
    if vx < 0.0 and back_arc and abs(vyaw) >= cfg.veer_vyaw:
        return max(cfg.reverse_vx, vx), vyaw, ("back_arc",)
    if vx < 0.0:
        # a refused reverse is a stop — never turned into a forward arc into what it backs from
        notes.append("no_reverse")
        return 0.0, 0.0, tuple(notes)
    if vx < cfg.pause_below:
        if abs(vyaw) < cfg.turn_below:
            return 0.0, 0.0, tuple(notes)
        notes.append("arc")  # owner: turns are arcs (6 s in place turns only 0–11°)
        return cfg.min_vx, vyaw, tuple(notes)
    if vx < cfg.min_vx:
        notes.append("min_speed")
        vx = cfg.min_vx
    cap = cfg.approach_vx if approach else cfg.max_vx
    if vx > cap:
        notes.append("cap")
        vx = cap
    return vx, vyaw, tuple(notes)


class BodyLink:
    """The per-tick safety and streaming layer (see the module doc)."""

    def __init__(self, cfg: LinkConfig = DEFAULT) -> None:
        self.cfg = cfg
        self.delay_ms = (
            command_delay_ms() if cfg.command_delay_ms is None else float(cfg.command_delay_ms)
        )
        self.proprio = Proprio(cfg.proprio)
        self.righting = Righting(cfg.righting)
        self.streamer = MoveStreamer()
        self.state: RobotState | None = None
        self.posture_want: str | None = None
        self.toggled_ns: int | None = None
        self.home_since_ns: int | None = None  # the measured head at HOME since
        self.toggle_wait_ns: int | None = None  # a posture change waiting for the head since
        self.head_sent: Offsets | None = None
        self.head_sent_ns: int | None = None
        self.mouth_sent: float | None = None
        self.mouth_sent_ns: int | None = None
        self.pose_on = False  # robot.pose is active (an emotion bows the standing body)
        self.was_lying = False  # the last tick's request said the duck lies down on purpose
        self.rolling_until_ns = 0  # a roll the duck asked for is on (or it has not stood up since): no fall, no righting
        self.roll_up_ns: int | None = None  # … it has stood upright again since
        self.lay_downs = 0  # lie-downs on purpose seen (for the evaluators)
        self.walking = False  # the last tick streamed a non-zero twist
        self.flight_t0_ns: int | None = (
            None  # the panic skill has been running since (MIN_FLIGHT_S)
        )
        self.flight_seen_ns = 0
        self._odom_hist: deque[tuple[int, float, float]] = (
            deque()
        )  # the trunk's recent odometry (startle-hop)
        self.flight_cut_until_ns = (
            0  # a stumble ended the flight: the hand-back, whatever the brain asks
        )
        self.flight_twist: tuple[float, float] = (
            HANDBACK_VX,
            0.0,
        )  # what the flight last asked for
        self.handback_until_ns = (
            0  # a flight just ended on its own: v5 keeps HANDBACK_VX until then (no zero twist)
        )
        self.hold_until_ns = 0  # a reflex stop holds the body still until then …
        self.veer_until_ns = 0  # … then only arcs (at most until then) …
        self.veer_sign = 0  # … turning this way (+1 left, −1 right, 0 either): away from a hit
        self.veer_yaw0: float | None = None  # … until the heading has turned from this
        # the wall pivot (cfg.wall_pivot): the episode running now = (side +1 left / -1 right, odom yaw at the start, start ns), the
        # starts of the last few (ns, x, y), the last tick a wall held the body, the end of the cool-down
        self.wp: tuple[int, float, int] | None = None
        self.wp_starts: deque[tuple[int, float, float]] = deque()
        self.wp_pin_ns = -(10**18)
        self.wp_held_ns: int | None = None  # held at a wall without a break since
        self.wp_cool_ns = 0
        self.wp_side_last = 0
        self.wall_pivots = 0  # episodes started
        self.wall_pivots_refused = 0  # … that robotd / the skill guard did not run
        self.wall_pivots_spent = 0  # ticks the budget said no
        self.refused_total = 0
        self.ticks = 0
        self.contact_ns = -(10**18)  # when it last pushed into something (its own body's sense)
        self.stumble_ns = -(10**18)  # proprio's last "stumbled" edge (a trip only if it tips)
        self.stumbles_ignored = 0  # "stumbled" edges the trunk's tilt did not back
        self.lift_pending_ns: int | None = None  # a lift claim right after a push, unconfirmed
        self.lift_told = False  # the rest of the duck was told it is lifted
        self.lifts_dropped = 0  # claims that turned out to be a push (for the evaluators)
        # a lift claim that persists while seated is re-tested (module doc, liftcheck.FloorProbe)
        self.claim_ns: int | None = None  # the standing claim has stood (seated) since
        self.probe: FloorProbe | None = None  # the re-test running now
        self.probe_next_ns = 0  # no re-test starts before this (the rate limit)
        self.probe_votes = (
            0  # re-tests in a row that found something under the duck bears its weight
        )
        self.probe_blocked = False  # the last tick wanted the head or the legs for something else
        self.refuted = False  # a claim was refuted: light legs claim no more while the seat stays
        self.refuted_policy = ""  # the posture it was refuted in
        self.floor_probes = 0  # re-tests started (for the evaluators)
        self.lifts_refuted = 0  # claims a re-test ended (for the evaluators)
        self.was_unloaded = False
        self.lean_unsure = False  # the legs went light while pressing into something
        self.air_run = 0  # (fid:fix) ticks in a row the legs swung free
        self.air_until_ns = 0  # … it counts as in the air until then
        self.airs = 0  # times it was found in the air (for the evaluators)
        # Stage-1 trained one-shot skills: set by the driver when robotd lists them (None: none)
        self.skills: SG.SkillGuard | None = None
        self._skill_said: tuple[str, str] | None = None  # (s1b:fix) the last refusal said
        # the guard's last answer to a request: (skill, decision, the guard's clock) — the page's
        # 「動作試玩」 reads why a press was refused
        self.last_skill_decision: tuple[str, SG.Decision, float] | None = None

    # ── helpers ──

    def _confirm_lift(self, prox: ProprioOut, t_ns: int, touch: bool) -> ProprioOut:
        """Proprio's lift claim, believed at once — unless it came right after the body pushed
        into something (``lift_after_contact_s``): then it must hold ``lift_confirm_s`` first
        (meanwhile only "unsure": no walking, no sitting down), and a claim the floor ends
        before that is dropped with its set-down. Only its own body's senses decide this."""
        cfg = self.cfg
        view = prox.view
        edges = list(prox.edges)
        if touch or view.blocked or "blocked" in edges or "stumbled" in edges:
            self.contact_ns = t_ns
        if self.hold_until_ns > t_ns:
            self.contact_ns = t_ns  # a stall stop is holding it: a push, whatever else it is
        recent = t_ns - self.contact_ns < int(cfg.lift_after_contact_s * NS)
        if "lifted" in edges and recent and not self.lift_told:
            self.lift_pending_ns = t_ns
        if self.lift_pending_ns is not None:
            if not view.lifted:  # the floor ended it: it was the wall, not a hand
                self.lift_pending_ns = None
                self.lifts_dropped += 1
                kept = [e for e in edges if e not in ("lifted", "set_down")]
                return replace(prox, edges=tuple(kept))
            if t_ns - self.lift_pending_ns >= int(cfg.lift_confirm_s * NS):
                self.lift_pending_ns = None
                self.lift_told = True
                return replace(prox, edges=tuple(dict.fromkeys([*edges, "lifted"])))
            kept = [e for e in edges if e != "lifted"]
            return replace(prox, edges=tuple(kept), view=replace(view, lifted=False, unloaded=True))
        if "lifted" in edges:
            self.lift_told = True
        if not view.lifted:
            self.lift_told = False
        return prox

    def _retest_lift(
        self, prox: ProprioOut, state: RobotState, t_ns: int, lying: bool
    ) -> ProprioOut:
        """A lift claim that has stood ``retest_after_s`` while the duck sits, re-tested (module doc): the legs' load is read with the
        head at HOME and with it raised (:class:`~duckmind.neuro.liftcheck.FloorProbe`), from the duck's own senses only. Only
        ``retest_votes`` re-tests in a row that find the legs carry the trunk end the claim, through the ``set_down`` edge every
        listener already takes; a claim that is refuted stays down while the seat stays (:attr:`refuted`) and is re-tested the same way,
        so a hand that lifts the duck afterwards is found within one more re-test (the legs then carry nothing)."""
        cfg = self.cfg
        view = prox.view
        t_ms = t_ns // MS
        seated = view.posture == "sit" and not view.fallen and not lying
        if self.refuted and not (seated and state.policy == self.refuted_policy):
            self.refuted = False  # the seat changed: judged afresh
        if view.lifted and seated:
            if self.claim_ns is None:
                self.claim_ns = t_ns
        else:
            self.claim_ns = None
            if not self.refuted:
                self.probe_votes = 0
        watching = seated and (view.lifted or self.refuted)
        if self.probe is None:
            stood = self.claim_ns is not None and t_ns - self.claim_ns >= int(
                cfg.retest_after_s * NS
            )
            if (
                watching
                and (stood or self.refuted)
                and t_ns >= self.probe_next_ns
                and not self.probe_blocked
            ):
                tilt = state.tilt_rad
                # (a trunk past proprio's own lift limit lies on something with its neck pinned: the head is read at rest, liftcheck)
                tilted = (
                    tilt is not None and math.degrees(tilt) >= self.proprio.cfg.lift_max_tilt_deg
                )
                self.probe = FloorProbe(t_ms, tilted=tilted)
                self.probe_next_ns = t_ns + int(cfg.retest_every_s * NS)
                self.floor_probes += 1
        if self.probe is not None:
            g = state.gyro
            gyro = None if g is None else math.sqrt(g[0] ** 2 + g[1] ** 2 + g[2] ** 2)
            verdict = self.probe.step(
                t_ms,
                state.leg_load,
                state.head_offsets,
                gyro,
                ended=not watching or self.probe_blocked,
                down=state.down,
            )
            if verdict is not None:
                self.probe = None
                if verdict == "held":
                    self.probe_votes = 0
                    self.refuted = False  # nothing bears it: a hand holds it (the legs claim it as they always did)
                elif verdict == "supported" and not self.refuted:
                    self.probe_votes += 1
                    if self.probe_votes >= cfg.retest_votes:
                        self.probe_votes = 0
                        self.refuted, self.refuted_policy = True, state.policy
                        self.lifts_refuted += 1
                        self.claim_ns = None
                        self.lift_told = False
                        self.lift_pending_ns = None
                        edges = tuple(dict.fromkeys([*prox.edges, "set_down"]))
                        prox = replace(
                            prox, edges=edges, view=replace(view, lifted=False, unloaded=False)
                        )
        if self.refuted:
            self.proprio.drop_claim()  # (every tick: its light legs would claim again after their holds)
        return prox

    def _lying(self, prox: ProprioOut, events: list[dict[str, Any]]) -> ProprioOut:
        """The duck lies down on purpose: no fall, no lift, no legs-light claim — the edges that say so are dropped and the
        view keeps the duck "up" and "loaded" — and one ``body.lay_down`` event says why the claims were not made."""
        masked = [e for e in prox.edges if e in LYING_MASKED]
        if masked or not self.was_lying:
            if not self.was_lying:
                self.lay_downs += 1
            events.append({"kind": "body.lay_down", "held_back": masked})
        self.proprio.lift = (
            "ground"  # (a claim of the legs the proprio began to believe is not carried on)
        )
        self.lift_pending_ns = None
        self.lift_told = False
        view = replace(prox.view, fallen=False, lifted=False, unloaded=False)
        return replace(prox, view=view, edges=tuple(e for e in prox.edges if e not in LYING_MASKED))

    def _rolling(self, state: RobotState, t_ns: int) -> bool:
        """The roll the duck asked for (robotd runs the roulade; a roll is a fall to the gravity filter): it lies down on
        purpose, so no fall edge, no righting and no lift claim, from the start of the skill until it has stood upright
        again for ``ROLL_UP_S`` (the hold is let go only with the trunk up: a duck still down after that falls as any does)."""
        label = (state.policy or "").split(":", 1)[0]
        if label == SG.ROLL_ENTRY:
            self.rolling_until_ns = t_ns + int(ROLL_HOLD_MAX_S * NS)
            self.roll_up_ns = None
            return True
        if t_ns >= self.rolling_until_ns:
            return False
        tilt = state.tilt_rad
        up = (
            label in ("walk", "stand")
            and not state.fallen
            and tilt is not None
            and math.degrees(tilt) < ROLL_UP_TILT_DEG
        )
        if not up:
            self.roll_up_ns = None
            return True
        if self.roll_up_ns is None:
            self.roll_up_ns = t_ns
        if t_ns - self.roll_up_ns >= int(ROLL_UP_S * NS):
            self.rolling_until_ns = 0
            self.roll_up_ns = None
            return False
        return True

    def _lying_over(self, prox: ProprioOut) -> ProprioOut:
        """It no longer lies down on purpose: a duck that is still down did not get up by itself — the ordinary fall (and its
        righting) begins here, as the fall edge that was held back would have."""
        if prox.view.fallen and "fell" not in prox.edges:
            return replace(
                prox, edges=(*prox.edges, "fell"), fell_trigger=prox.fell_trigger or "fallen"
            )
        return prox

    def _airborne(
        self, state: RobotState, t_ns: int, righting: bool, events: list[dict[str, Any]]
    ) -> bool:
        """(fid:fix) Whether the legs are swinging free — picked up while stepping — from its own
        joints and gyro (robot.state): the ten leg joints faster than any step on a floor
        (``air_leg_speed``) for ``air_ticks`` ticks, the trunk not thrown about; it stays so for
        ``air_hold_s`` after the last such tick. The first tick of it adds a ``body.airborne``
        event."""
        cfg = self.cfg
        speed = _leg_speed(state)
        gyro = state.gyro
        calm = gyro is None or math.sqrt(gyro[0] ** 2 + gyro[1] ** 2 + gyro[2] ** 2) < cfg.air_gyro
        after_flight = self.flight_seen_ns > 0 and t_ns - self.flight_seen_ns < int(
            AFTER_FLIGHT_S * NS
        )
        if (
            speed is not None
            and not after_flight
            and speed >= cfg.air_leg_speed
            and calm
            and not righting
            and state.policy in ("walk", "stand")
        ):
            self.air_run += 1
        else:
            self.air_run = 0
        if self.air_run >= cfg.air_ticks:
            if t_ns >= self.air_until_ns:
                self.airs += 1
                events.append({"kind": "body.airborne", "leg_speed": round(speed or 0.0, 2)})
            self.air_until_ns = t_ns + int(cfg.air_hold_s * NS)
        return t_ns < self.air_until_ns

    def _trunk_speed(self, state: RobotState, t_ns: int) -> float:
        """The trunk's speed from its own odometry over the last 0.15-0.4 s (inf: not enough history yet)."""
        hist = self._odom_hist
        x, y = float(state.odom_pos[0]), float(state.odom_pos[1])
        if hist and t_ns < hist[-1][0]:
            hist.clear()  # (the clock started over)
        if not hist or t_ns - hist[-1][0] >= int(0.02 * NS):
            hist.append((t_ns, x, y))
        while hist and t_ns - hist[0][0] > int(0.4 * NS):
            hist.popleft()
        old = [h for h in hist if t_ns - h[0] >= int(0.15 * NS)]
        if not old:
            return math.inf
        t0, x0, y0 = old[-1]
        return math.hypot(x - x0, y - y0) / ((t_ns - t0) / NS)

    def _flying(self, state: RobotState) -> bool:
        """robotd runs one of the panic skill entries now."""
        if self.skills is None:
            return False
        return SG.panic_running(self.skills.names, state.policy)

    def _qturning(self, state: RobotState) -> bool:
        """robotd runs one of the quick-turn entries now."""
        if self.skills is None:
            return False
        return SG.quick_turn_running(self.skills.names, state.policy)

    def _wall_side(self, vyaw: float, depth: P.DepthView | None, t_ns: int) -> int:
        """Which way a wall pivot turns (+1 left, -1 right): away from the side that was hit when proprio knows it, else the way
        the brain's arc turns (the wall memory's freer side) unless the ToF sees that side shut and the other open, else the side the
        ToF sees more room on, else the other way than the last time."""
        cfg = self.cfg
        if self.veer_sign != 0 and t_ns < self.veer_until_ns:
            return self.veer_sign
        left = 9999 if depth is None or depth.left_mm is None else int(depth.left_mm)
        right = 9999 if depth is None or depth.right_mm is None else int(depth.right_mm)
        tof = 0 if abs(left - right) < 150 else (1 if left > right else -1)
        if abs(vyaw) >= cfg.veer_vyaw:
            brain = 1 if vyaw > 0.0 else -1
            room = left if brain > 0 else right
            if tof and tof != brain and room < 350:
                return tof
            return brain
        if tof:
            return tof
        return -self.wp_side_last or 1

    def _wall_pivot_step(
        self,
        state: RobotState,
        request: BodyRequest,
        depth: P.DepthView | None,
        t_ns: int,
        *,
        reflex: str | None,
        vyaw: float,
        walk: bool,
        hard: bool,
        busy: bool,
        goal: bool,
        qturning: bool,
    ) -> int:
        """The wall pivot (see :attr:`LinkConfig.wall_pivot`): +1 / -1 while the duck should be turning away from a wall on the spot
        with the quick-turn skill (left / right), else 0. ``walk``: the brain asks for a forward walk; ``hard``: a stop that no skill
        may override (a fall, a lift, the battery, a drop); ``busy``: another program has the legs (a flight, a skill, the brain's own pivot,
        a posture change); ``goal``: the walk is toward a goal (an approach to a thing, the dock): turned away from only after
        :attr:`LinkConfig.wall_pivot_long_s` of being held."""
        cfg = self.cfg
        if not cfg.wall_pivot or self.skills is None or not SG.quick_turn_names(self.skills.names):
            self.wp = None
            return 0
        # held at a wall: the nose on something (the ToF touch), a stall, or the ToF stop in front of a straight walk (an arc is let through
        # at the ToF stop: the duck walks it away; only when that ends in a touch is it held)
        held = reflex in ("reflex_touch", "stall") or (
            reflex == "reflex_near" and abs(vyaw) < cfg.veer_vyaw
        )
        if held:
            self.wp_pin_ns = t_ns
            if self.wp_held_ns is None:
                self.wp_held_ns = t_ns
        elif t_ns - self.wp_pin_ns > int(0.8 * NS):
            self.wp_held_ns = None
        long_held = self.wp_held_ns is not None and t_ns - self.wp_held_ns >= int(
            cfg.wall_pivot_long_s * NS
        )
        if self.wp is not None:
            side, yaw0, t0 = self.wp
            turned = abs(wrap(float(state.odom_yaw) - yaw0))
            elapsed = (t_ns - t0) / NS
            clear = depth is None or (
                not depth.touch
                and (depth.ahead_mm is None or depth.ahead_mm >= cfg.wall_pivot_clear_mm)
            )
            lost = elapsed >= cfg.wall_pivot_start_s and not qturning
            done = (
                hard
                or busy
                or lost
                or elapsed >= cfg.wall_pivot_max_s
                or turned >= cfg.wall_pivot_max_rad
                or (turned >= cfg.wall_pivot_min_rad and clear)
            )
            if not done:
                return side
            self.wp = None
            self.wp_cool_ns = t_ns + int(cfg.wall_pivot_cooldown_s * NS)
            self.wp_pin_ns = -(10**18)
            self.wp_held_ns = None
            return 0
        if hard or busy or not walk or (goal and not long_held) or t_ns < self.wp_cool_ns:
            return 0
        if not held and t_ns - self.wp_pin_ns > int(cfg.wall_pivot_pin_s * NS):
            return 0
        x, y = float(state.odom_pos[0]), float(state.odom_pos[1])
        starts = self.wp_starts
        while starts and t_ns - starts[0][0] > int(cfg.wall_pivot_window_s * NS):
            starts.popleft()
        near = sum(
            1 for _, sx, sy in starts if math.hypot(sx - x, sy - y) <= cfg.wall_pivot_radius_m
        )
        if near >= cfg.wall_pivot_budget:
            self.wall_pivots_spent += 1
            return 0
        side = self._wall_side(vyaw, depth, t_ns)
        self.wp = (side, float(state.odom_yaw), t_ns)
        self.wp_side_last = side
        starts.append((t_ns, x, y))
        self.wall_pivots += 1
        return side

    def _toggle_due(self, t_ns: int) -> bool:
        return self.toggled_ns is None or t_ns - self.toggled_ns >= int(
            self.cfg.posture_retry_s * NS
        )

    def _send(
        self,
        intent: P.Intent,
        guard: GuardContext,
        out: list[P.Intent],
        refused: list[tuple[str, str]],
        t_ns: int,
    ) -> bool:
        why = check_intent(intent, guard, self.cfg.limits)
        if why is None:
            out.append(intent)
            return True
        refused.append((intent.method, why))
        self.refused_total += 1
        if intent.method == "robot.move":
            out.append(self.streamer.zero_now(t_ns))
        return False

    # ── the tick ──

    def tick(
        self,
        state: RobotState,
        request: BodyRequest,
        *,
        depth: P.DepthView | None = None,
        battery_pct: int | None = None,
    ) -> LinkOut:
        """One robot.state tick: the request through the safety layer (module doc)."""
        cfg = self.cfg
        self.state = state
        self.ticks += 1
        t_ns = state.t_ns
        t_ms = t_ns // MS
        sent = self.streamer.last
        touch = bool(depth is not None and depth.touch)
        prox = self.proprio.step(state, cmd=(sent[0], sent[1]), touch=touch)
        prox = self._confirm_lift(prox, t_ns, touch)  # a push into a wall is not a pickup
        events: list[dict[str, Any]] = []
        lying = bool(request.lying) or self._rolling(state, t_ns)
        prox = self._retest_lift(prox, state, t_ns, lying)
        if lying:
            prox = self._lying(prox, events)
        elif self.was_lying:
            prox = self._lying_over(prox)
        self.was_lying = lying
        view = prox.view
        during = "walk" if self.walking else "still"
        rout = RightingOut() if lying else self.righting.step(state, prox, during=during)
        events.extend(dict(body) for body in rout.events)
        for edge in prox.edges:
            events.append({"kind": f"proprio.{edge}", "side": prox.contact_side})
        gate: list[str] = []
        intents: list[P.Intent] = []
        refused: list[tuple[str, str]] = []
        righting = self.righting.active or view.fallen
        # (fid:fix) the legs swinging free while it steps: it is in the air (see LinkConfig)
        airborne = False if lying else self._airborne(state, t_ns, righting, events)

        # the head's measured offsets and whether they fit the walking envelope
        measured = state.head_offsets
        tol = cfg.head_keep_tol if self.walking else cfg.head_start_tol
        head_ok = head_inside(measured, cfg.walk_head, tol)
        # a flight, and the second after it, are not held to the walking head envelope (the panic gait swings the head far outside
        # it, and it takes a moment to come back): the guard's "head must be at HOME and settled" rule refused each flight move and
        # zeroed the twist mid-flight (live, round 4: 9 % of the flight ticks), the flight decayed from 0.53 to 0.03 m/s and tipped
        recent_flight = self._flying(state) or (
            self.flight_seen_ns > 0 and t_ns - self.flight_seen_ns < int(AFTER_FLIGHT_S * NS)
        )
        guard = GuardContext(
            head_home_ms=10_000
            if (head_ok or recent_flight)
            else 0,  # the one change: a live head may walk
            act=None,
            approach=request.approach,
            lifted=view.lifted,
            unloaded=view.unloaded,
            fallen=view.fallen,
            righting=self.righting.active,
        )
        if rout.do is not None:
            self._send(P.Intent.do(rout.do), guard, intents, refused, t_ns)

        # ── the twist ──
        vx, vyaw, notes = shape_twist(
            request.vx,
            request.vyaw,
            approach=request.approach,
            docking=request.docking,
            back_arc=request.back_arc,
            cfg=cfg,
        )
        gate.extend(notes)
        wants_walk = vx != 0.0
        want_posture = request.posture if request.posture in ("stand", "sit") else None
        if wants_walk:
            want_posture = "stand"
        low_battery = battery_pct is not None and battery_pct < cfg.battery_min_pct
        if righting:
            gate.append("fallen")
        elif view.lifted:
            gate.append("lifted")
            want_posture = "sit"  # owner B4: it sits once it senses the pickup
        elif airborne:
            gate.append("airborne")
            want_posture = "sit"  # (fid:fix) in the air: sit at once (the legs tuck, stop kicking)
        elif view.unloaded:
            gate.append("unloaded")
        if low_battery and (wants_walk or want_posture == "stand"):
            gate.append("battery")
            want_posture = "sit"
        # reflex stops (duck-body's verdict; fell/lifted/unloaded are handled above)
        # a back arc backs away: the ToF looks ahead, so its touch/near/drop stops say nothing about
        # where a backing duck goes (duck-body's verdict says so: ``reversing``). Without it a
        # ToF "touch" re-armed the 0.5 s hold every time it ended, and a duck pressed on something
        # could not back off at all (2026-10-04)
        backing_arc = request.back_arc and not request.docking and request.vx < 0.0
        walk_cmd = None
        if request.toward is not None and request.vx > 0.0 and not request.docking:
            walk_cmd = P.WalkCmd(
                act_id="neuro",
                deadline_ms=t_ms + 1000,
                vx_mm=int(round(request.vx * 1000.0)),
                purpose="curio",
                toward=P.Toward(
                    look_id="neuro",
                    b_mrad=request.toward[0],
                    r_mm=request.toward[1],
                    seen_t_ms=t_ms,  # the belief of where it is is as current as its odometry
                ),
                stop_ahead_mm=CURIO_STOP_MM,
            )
        v = verdict(proprio=view, depth=depth, cmd=walk_cmd, now_ms=t_ms, reversing=backing_arc)
        reflex = (
            v.reason if v.halt and v.reason not in ("fell", "lifted", "reflex_unloaded") else None
        )
        backing_in = request.docking and request.vx < 0.0
        if backing_in and reflex in ("stall", "reflex_near", "reflex_touch", "stop_near"):
            # (dock:build) backing into the dock: the stall is its face ("blocked = arrived",
            # the choreography stops on it) and the ToF looks ahead, away from where it goes
            reflex = None
        # (startle-hop, 2026-10-10) the back-steps lean the trunk 10-13 degrees and jolt the gyro BY DESIGN: read as a trip (live, 7 of 12 hops) the
        # stumble reflex armed `veer` (arcs only, up to 5 s) and the flight 1.5 s later was held back by it (stalled 0.5-2.3 s, one never started)
        hopping = (state.policy or "").split(":", 1)[0] == SG.HOP_ENTRY
        if hopping:
            self.stumble_ns = -(10**18)
        elif "stumbled" in prox.edges:
            self.stumble_ns = t_ns
            self.stumbles_ignored += 1  # (taken back below if the trunk tips)
        tilt_deg = math.degrees(state.tilt_rad) if state.tilt_rad is not None else 0.0
        recent = t_ns - self.stumble_ns <= int(cfg.stumble_window_s * NS)
        if recent and tilt_deg >= cfg.stumble_tilt_deg and reflex is None:
            reflex = "stumble"  # tripped (spec §8.9 E5): stop, then turn away
            if self.stumble_ns > 0:
                self.stumbles_ignored = max(0, self.stumbles_ignored - 1)
                self.stumble_ns = -(10**18)
        cut = reflex == "stumble" and self._flying(state) and not righting and not request.docking
        if cut:
            self.flight_cut_until_ns = t_ns + int(HANDBACK_S * NS)
        if not cut and (
            reflex in ("reflex_touch", "stall", "stumble")
            or (reflex == "reflex_near" and abs(vyaw) < cfg.veer_vyaw)
        ):
            if t_ns >= self.hold_until_ns:
                self.hold_until_ns = t_ns + int(cfg.reflex_hold_s * NS)
                # (dock:build) docking: the hold only — the choreography turns where it must
                veer_s = 0.0 if request.docking else cfg.veer_max_s
                self.veer_until_ns = self.hold_until_ns + int(veer_s * NS)
                self.veer_sign = _away(prox.contact_side) if reflex == "stall" else 0
                self.veer_yaw0 = None
        if reflex is not None:
            gate.append(reflex)
        # Light legs right after pushing into something are the chest leaning on it, not a hand
        # (live 2026-09-28: a duck pressed into a corner stayed "unsure" and could never walk
        # off): then an arc away may go on; straight on stays stopped.
        if view.unloaded and not self.was_unloaded:  # an "unsure" that began at a push
            self.lean_unsure = t_ns - self.contact_ns < int(cfg.lift_after_contact_s * NS)
        if not view.unloaded:
            self.lean_unsure = False
        self.was_unloaded = view.unloaded
        leaning = view.unloaded and self.lean_unsure and abs(vyaw) >= cfg.veer_vyaw
        stop = righting or view.lifted or airborne or (view.unloaded and not leaning) or low_battery
        if reflex == "reflex_drop":
            stop = True
        if t_ns < self.hold_until_ns:
            stop = True
        elif t_ns < self.veer_until_ns:
            # after the stop: only an arc, and away from the side that was hit when it is known,
            # until the heading has turned far enough (its own odometry yaw)
            if self.veer_yaw0 is None:
                self.veer_yaw0 = float(state.odom_yaw)
            if abs(wrap(float(state.odom_yaw) - self.veer_yaw0)) >= cfg.veer_turn_rad:
                self.veer_until_ns = 0
            elif wants_walk and not backing_arc:  # (a back arc backs off the hit: not gated)
                toward_hit = self.veer_sign != 0 and vyaw * self.veer_sign < 0
                if abs(vyaw) < cfg.veer_vyaw or toward_hit:
                    gate.append("veer")
                    stop = True
        if reflex == "reflex_near" and abs(vyaw) < cfg.veer_vyaw:
            stop = True

        posture_now = view.posture  # stand (walk/stand policy) | sit | unknown
        transitioning = view.policy in ("rise", "held", "")
        flying = self._flying(state)
        qturning = self._qturning(state)
        if flying or qturning or view.policy == SG.ABORT:
            # the panic / quick-turn skill is running upright; and the tick or two of robotd's own ``skill_abort`` label (the
            # hand-back to v5) is not a posture change either: read as one (live, round 4) it stopped the legs at the abort
            # ("posture" gate), wiped the hand-back window, and the duck walked on at the flight's own arc (0.35 m/s,
            # 0.99 rad/s) and tipped
            posture_now, transitioning = "stand", False
        # the wall pivot: held at a wall and asked to walk on, a duck with the quick turn turns away on the spot, not in an arc back into it
        wall_side = self._wall_pivot_step(
            state,
            request,
            depth,
            t_ns,
            reflex=reflex,
            vyaw=vyaw,
            walk=wants_walk and vx > 0.0,
            hard=righting
            or view.lifted
            or airborne
            or view.unloaded
            or low_battery
            or reflex == "reflex_drop",
            busy=flying
            or backing_arc
            or request.panic
            or request.skill is not None
            or request.pivot != 0.0
            or posture_now != "stand"
            or transitioning,
            goal=request.docking or request.approach or request.toward is not None,
            qturning=qturning,
        )
        if wall_side:
            stop = True
        if wants_walk and not stop:
            if posture_now != "stand" or transitioning:
                gate.append("posture")
                stop = True
            elif not head_ok and not recent_flight:
                # (the panic gait swings the head well outside the walking envelope — neck -0.6, yaw
                # -1.1 rad at 1 m/s, robotd's skill zeroes the head command: measured 2026-10-09 — and
                # that envelope was measured for v5 at <= 0.55: a flight is not stopped by it; the
                # walk that follows waits for the head to be back inside)
                gate.append("head")
                stop = True
        if flying:
            if self.flight_t0_ns is None:
                self.flight_t0_ns = t_ns
            self.flight_seen_ns = t_ns
        elif self.flight_t0_ns is not None and t_ns - self.flight_seen_ns > int(0.3 * NS):
            self.flight_t0_ns = None
        cut_now = t_ns < self.flight_cut_until_ns
        asked_flight = (
            bool(request.panic) and wants_walk and not stop and not righting and not cut_now
        )
        held_flight = (
            flying
            and not cut_now
            and not asked_flight
            and not stop
            and not righting
            and not request.docking
            and self.flight_t0_ns is not None
            and t_ns - self.flight_t0_ns < int(MIN_FLIGHT_S * NS)
        )
        asking_flight = asked_flight or held_flight
        if flying and not asking_flight and not stop and self.handback_until_ns <= t_ns:
            self.handback_until_ns = t_ns + int(
                HANDBACK_S * NS
            )  # (the flight ends on its own: coast, do not drop)
        coasting = (
            (not wants_walk or cut_now)
            and not stop
            and not righting
            and not request.docking
            and t_ns < self.handback_until_ns
        )
        if stop:
            self.handback_until_ns = 0
        if held_flight:
            twist = self.flight_twist
            wants_walk = True
            want_posture = "stand"
        elif coasting:
            twist = (HANDBACK_VX, 0.0)
            wants_walk = True
            want_posture = "stand"
        else:
            twist = (0.0, 0.0) if stop or not wants_walk else (vx, vyaw)
        if asked_flight:
            self.flight_twist = twist
        if (
            twist[0] > 0.0
            and not held_flight
            and not coasting
            and not request.docking
            and not request.panic
            and not request.back_arc
        ):
            # (the simulator's servo command delay: the commands that walk and turn as they were designed to)
            twist = (
                delay_compensated_vx(twist[0], self.delay_ms),
                delay_compensated_vyaw(twist[1], self.delay_ms),
            )

        # ── posture (sit_toggle until robotd's policy says it) ──
        # A stand-up or sit-down is a balance manoeuvre (spec §10.3: 4.7–6.3 % of sit–stand
        # cycles tip over; live, a rise begun with the head looking down tipped over): the head
        # first settles at HOME (≥ home_settle_s, rule 3 "move only when still") and stays there
        # through it. Held up in the air (B4) it sits at once: nothing to balance there.
        if want_posture is not None:
            self.posture_want = want_posture
        wanted = self.posture_want
        home_now = all(abs(v) <= tol for v, tol in zip(measured, cfg.home_tol, strict=True))
        if not home_now:
            self.home_since_ns = None
        elif self.home_since_ns is None:
            self.home_since_ns = t_ns
        settled = self.home_since_ns is not None and t_ns - self.home_since_ns >= int(
            cfg.home_settle_s * NS
        )
        needs_toggle = (
            wanted is not None
            and not righting
            and not transitioning
            and posture_now in ("stand", "sit")
            and posture_now != wanted
            and twist == (0.0, 0.0)
            and (not self.walking or airborne)
            and not (
                wanted == "stand" and (view.lifted or view.unloaded or airborne or low_battery)
            )
        )
        if not needs_toggle:
            self.toggle_wait_ns = None
        elif self.toggle_wait_ns is None:
            self.toggle_wait_ns = t_ns
        waited = self.toggle_wait_ns is not None and t_ns - self.toggle_wait_ns >= int(
            cfg.home_wait_s * NS
        )
        held_up = view.lifted or airborne  # (nothing to balance in the air: no head settle first)
        if needs_toggle and not (settled or held_up or waited):
            gate.append("posture_head")
        if needs_toggle and (settled or held_up or waited) and self._toggle_due(t_ns):
            self.toggled_ns = t_ns
            self._send(P.Intent.do("sit_toggle"), guard, intents, refused, t_ns)

        # ── the head (HOME while righting or changing posture; the walking envelope while
        # walking) ──
        moving = twist != (0.0, 0.0)
        toggling = self.toggled_ns is not None and t_ns - self.toggled_ns < int(
            cfg.posture_head_s * NS
        )
        # (a lift claim that persists is re-tested: the head goes up. A posture change, a walk, a fall or a flight come first.)
        self.probe_blocked = (
            righting or transitioning or toggling or airborne or needs_toggle or moving
        )
        if righting or transitioning or toggling or airborne or (needs_toggle and not view.lifted):
            head = HOME
        elif self.probe is not None and not self.probe_blocked:
            head = clamp_head(self.probe.head(), cfg.still_head)
        else:
            walk_cmd = cfg.walk_cmd_head if cfg.walk_cmd_head is not None else cfg.walk_head
            bounds = walk_cmd if (moving or wants_walk) else cfg.still_head
            if request.emotion and not (moving or wants_walk):
                bounds = cfg.emotion_head
            head = clamp_head(request.head, bounds)
        resend = self.head_sent_ns is None or t_ns - self.head_sent_ns >= int(
            cfg.head_resend_s * NS
        )
        changed = self.head_sent is None or any(
            abs(a - b) > cfg.head_change for a, b in zip(head, self.head_sent, strict=True)
        )
        if changed or resend:
            self.head_sent, self.head_sent_ns = head, t_ns
            self._send(P.Intent.head(head), guard, intents, refused, t_ns)

        # ── the body pose (an emotion's bow): standing, nothing rising or righting ──
        bow = None
        if (
            request.emotion
            and request.pose is not None
            and not (righting or transitioning or toggling or airborne or view.lifted)
            and posture_now == "stand"
        ):
            bow = max(0.0, float(request.pose))
        if bow is not None:
            self.pose_on = True
            self._send(P.Intent.pose(bow), guard, intents, refused, t_ns)
        elif self.pose_on:
            self.pose_on = False  # (robotd snaps the pose back to nominal)
            self._send(P.Intent.pose(0.0, active=False), guard, intents, refused, t_ns)

        # ── the beak ──
        mouth = 0.0 if righting else min(cfg.mouth_max, max(0.0, float(request.mouth)))
        mouth_due = self.mouth_sent_ns is None or t_ns - self.mouth_sent_ns >= int(
            cfg.mouth_resend_s * NS
        )
        if self.mouth_sent is None or abs(mouth - self.mouth_sent) > 0.01 or mouth_due:
            self.mouth_sent, self.mouth_sent_ns = mouth, t_ns
            self._send(P.Intent("robot.mouth", {"open": mouth}), guard, intents, refused, t_ns)

        # ── a trained one-shot skill (Stage 1): robot.do into this robotd's own table, through
        # the skill guard (the postures it was gated for, upright, no hand, no dock, the hourly
        # wear caps); a running skill that tips past 30° goes back to velstand (skill_abort) ──
        if self.skills is not None:
            gyro = state.gyro
            # (dock-fall) no whole-body skill from a reverse: the skills were trained and gated from a forward walk or a stand, and a
            # scare that ends the docking walk's reverse (the brain's startle is no docking request) used to start `freeze_3` at -0.4 m/s
            # right at the dock: the stumble that followed tipped the duck between the dock's guides (2 of 10 ducks, live)
            in_skill = (state.policy or "").split(":", 1)[0] in self.skills.names
            backing = (state.applied[0] < -0.05 or sent[0] < -0.05) and not in_skill
            speed_chain = self._trunk_speed(state, t_ns)  # (kept up every tick: the history)
            speed_chain = speed_chain if request.skill_chain else None
            sview = SG.BodyView(
                label=state.policy,
                twist=math.hypot(state.applied[0], state.applied[2]),
                tilt_deg=tilt_deg,
                fallen=view.fallen,
                righting=self.righting.active,
                held=view.lifted,
                # (a wall pivot is no skill on the dock: the docking choreography's forward legs, held at a door jamb, are turned away from too)
                docked=(request.docking and not wall_side) or backing,
                # (s1b:fix) how fast it walks (the guard's per-skill speed limit) and whether
                # the trunk is still (its settle rule after a rise, a stop or another skill)
                vx=float(state.applied[0]),
                gyro_rad_s=None if gyro is None else math.sqrt(sum(g * g for g in gyro)),
                # (startle-hop) the trunk's own speed (its odometry), given only for the hop-back that answers a scare
                speed_mps=speed_chain,
            )
            now_s = t_ns / NS
            abort = self.skills.watch(sview, now_s)
            # the panic skill (2026-10-09): held while the brain flees and no stop rule fires, handed back to
            # v5 (skill_abort) the tick that stops; another skill asked for meanwhile waits its turn
            flight = SG.panic_names(self.skills.names)
            ask = request.skill
            if request.skill_chain and low_battery:
                ask = None  # (startle-hop) no back-steps on a low battery: the scare gets its plain startle next
            if flight:
                asking = asking_flight
                if flying and not asking:
                    ask = SG.ABORT
                elif asking and ask is None:
                    ask = SG.panic_entry(request.vyaw, self.skills.names, state.policy)
            # the quick turn (2026-10-09): a pivot the brain asks for (turning toward something it is looking at, far from facing it)
            # runs as robotd's quick-turn skill, held while it is asked for and handed back to v5 the tick it is not
            if SG.quick_turn_names(self.skills.names) and not flying:
                pivoting = (
                    request.pivot != 0.0 and wants_walk and not stop and not righting
                    and not request.panic and not request.docking
                )  # fmt: skip
                if wall_side and ask is None:
                    ask = SG.quick_turn_entry(float(wall_side), self.skills.names)
                elif pivoting and ask is None:
                    ask = SG.quick_turn_entry(request.pivot, self.skills.names)
                elif qturning and not pivoting and not wall_side and ask is None:
                    ask = SG.ABORT
            if abort is not None:
                decision = self.skills.request(abort, sview, now_s)
            else:
                decision = SG.request_with_fallback(
                    self.skills,
                    ask,
                    sview,
                    now_s,
                    manual=request.skill_manual,
                    chain=request.skill_chain and ask == request.skill,
                )
            if wall_side and decision.skill is not None and not decision.ok:
                # the skill guard (or the intent guard below) will not run it: the old stop-and-veer is all there is for a while
                self.wp = None
                self.wp_cool_ns = t_ns + int(2.0 * NS)
                self.wall_pivots_refused += 1
            if decision.skill is not None and decision.reason != "hold":
                self.last_skill_decision = (decision.skill, decision, now_s)
            if decision.skill is not None:
                if decision.ok:
                    skill_guard = replace(guard, trained_skills=self.skills.names)
                    self._send(P.Intent.do(decision.skill), skill_guard, intents, refused, t_ns)
                    if decision.reason != "hold":  # (a held crouch's re-requests: no event)
                        events.append(
                            {"kind": "skill.started", "skill": decision.skill,
                             "how": decision.reason}
                        )  # fmt: skip
                    self._skill_said = None
                else:
                    gate.append("skill")
                    said = (decision.skill, decision.reason)
                    if said != self._skill_said:  # (said once, not every tick it is asked)
                        self._skill_said = said
                        events.append(
                            {
                                "kind": "skill.refused",
                                "skill": decision.skill,
                                "why": decision.reason,
                                "why_zh": decision.reason_zh,
                            }
                        )

        # ── the twist, streamed at 50 Hz while non-zero (robotd's 500 ms deadman) ──
        move = self.streamer.step(t_ns, twist if moving else None)
        if move is not None:
            move_guard = guard
            if move.twist[0] < 0.0 and (request.docking or request.back_arc):
                # owner (phase 3): reversing only while docking and as curiosity's back_arc —
                # through the guard's own reverse permission (every other rule still applies)
                move_guard = replace(guard, act="duck.curio", back_arc_ok=True)
            if not self._send(move, move_guard, intents, refused, t_ns):
                twist = (0.0, 0.0)
                moving = False
        self.walking = moving
        if refused:
            gate.append("refused")
        return LinkOut(
            intents=tuple(intents),
            prox=prox,
            events=tuple(events),
            gate=tuple(dict.fromkeys(gate)),
            twist=twist,
            head=head,
            mouth=mouth,
            posture_want=self.posture_want,
            righting=self.righting.phase,
            refused=tuple(refused),
        )

    def stop_intents(self, t_ns: int) -> list[P.Intent]:
        """What to send when the link lets go of the body: a zero twist, the head HOME and the
        beak closed (a stop is never refused)."""
        self.walking = False
        out = [
            self.streamer.zero_now(t_ns),
            P.Intent.head(HOME),
            P.Intent("robot.mouth", {"open": 0.0}),
        ]
        if self.pose_on:
            self.pose_on = False
            out.append(P.Intent.pose(0.0, active=False))
        return out


def symmetric(limits: Sequence[float]) -> Bounds:
    """± per-joint limits (neck_pitch, head_pitch, head_yaw, head_roll) as bounds."""
    a, b, c, d = (abs(float(v)) for v in limits)
    return ((-a, a), (-b, b), (-c, c), (-d, d))


def config_for_motor(command_max: Sequence[float], *, base: LinkConfig = DEFAULT) -> LinkConfig:
    """The link's head bounds for a motor system whose commands (robotd's head smoothing
    pre-compensated: they run ahead of the pose they mean) never leave ``± command_max`` rad:
    standing or seated, those; walking, those inside the measured :data:`WALK_HEAD`. The
    measured head must stay inside :data:`WALK_HEAD` to walk."""
    cmd = symmetric(command_max)
    walk_cmd = tuple(
        (max(lo, clo), min(hi, chi)) for (lo, hi), (clo, chi) in zip(WALK_HEAD, cmd, strict=True)
    )
    return replace(
        base,
        walk_head=WALK_HEAD,
        walk_cmd_head=(walk_cmd[0], walk_cmd[1], walk_cmd[2], walk_cmd[3]),
        still_head=cmd,
    )


def with_head(request: BodyRequest, head: Offsets | None) -> BodyRequest:
    return replace(request, head=head)


def head_offsets_of(pose: dict[str, float], side: int = 1) -> Offsets:
    """Clip units (``np``/``hp``/``hy``/``hr``, fractions of the seated envelope; ``side`` +1
    when positive yaw/roll mean the duck's left) → robot.head offsets from HOME, rad."""
    goal = content.head_targets(pose, side)
    return (
        goal["neck_pitch"] - content.HEAD_HOME["neck_pitch"],
        goal["head_pitch"] - content.HEAD_HOME["head_pitch"],
        goal["head_yaw"] - content.HEAD_HOME["head_yaw"],
        goal["head_roll"] - content.HEAD_HOME["head_roll"],
    )


def mouth_open_of(m: float) -> float:
    """Clip mouth (0..1 of the validated opening) → robot.mouth ``open`` (0..1 of the travel)."""
    return max(0.0, min(1.0, float(m))) * content.SEATED_ENVELOPE["mouth"]


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


__all__ = [
    "DEFAULT",
    "GATE_ZH",
    "HOME",
    "STILL_HEAD",
    "WALK_HEAD",
    "BodyLink",
    "BodyRequest",
    "LinkConfig",
    "LinkOut",
    "clamp_head",
    "command_delay_ms",
    "config_for_motor",
    "delay_compensated_vx",
    "delay_compensated_vyaw",
    "head_inside",
    "head_offsets_of",
    "mouth_open_of",
    "delay_compensated_vx",
    "shape_twist",
    "symmetric",
]
