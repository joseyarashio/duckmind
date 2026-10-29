"""The neural duck's contract: what the brain senses, what it commands, what it reports.

This file is the meeting point of the four prototype lanes:

* **neuro:senses** builds a :class:`SensorFrame` every brain tick from the real sensor doors
  (robotd ``robot.state``: IMU, joints, odometry, ``fallen``; tofd's ToF frame; the camera
  through phase-3 duck-eye where usable) plus *world-side sensor models* for what the prototype
  still lacks (sound, touch, dock contact, and the camera's motion/looming salience).
* **neuro:brain** (``brain.py``) turns frames into a :class:`MotorCommand` and a
  :class:`BrainTelemetry`: ``NeuroBrain(genome, seed).step(t_ms, frame) -> (cmd, telemetry)``.
* **neuro:motor** turns a :class:`MotorCommand` into robotd intents: a *parametric motor
  program* per behaviour in :data:`BEHAVIOURS` (head, neck, mouth, posture, vocal timing),
  overlaid with the brain's gaze/reflex head intent, executed differently each bout
  (``variant``) and clamped to the safe envelopes (pet:probe) and robotd's walking rules.
* **neuro:web** shows the telemetry on the simulator page and injects the scenario
  :data:`STIMULI` into the *world* (never into the brain): the senses lane reports what the
  duck's sensors would honestly pick up from them.

Honest senses (owner, 2026-09-26: decide only from what the duck's sensors could report). The
brain never reads a world position: every place in a frame is egocentric (bearing from the
trunk's forward axis, ``+`` = the duck's left, radians; ranges in metres) or in the duck's own
odometry frame (contact odometry from boot, which drifts). Every channel says where it came
from (:data:`SOURCES`): ``real`` is a real sensor door (in the simulator: robotd/tofd ``--sim``
on MuJoCo physics, the same code path as the board), ``eye`` is the phase-3 duck-eye pipeline
on a rendered head-camera frame, ``model`` is a world-side sensor model standing in for a
sensor or detector the prototype does not have yet. :data:`CHANNEL_SOURCE` lists the default of
each channel; a frame carries its own ``sources`` so a page can mark modelled readings.

Units: time ``t_ms`` (int, the brain's monotonic clock), angles in radians, distances in metres
(ToF in millimetres, as tofd reports), speeds m/s and rad/s, levels 0..1 unless noted.
Floats are fine in the prototype; phase 5 converts the brain to fixed point (see brain.py).

Pure data: standard library only. The brain never imports :mod:`duckmind.world`.

Changes (additive only; each with its time):

* 2026-09-27 23:15 neuro:motor — ONE module per concern (orchestrator 22:52). Nothing in this
  file changed. Survivors: ``genome.py`` is the ONE diversity mapping (the motor side builds its
  profile with ``motor.profile_from_genome(genome)``; ``motor.hatch_traits``,
  ``profile_from_traits`` and ``profile_from_seed`` are gone — the live seam's ``LiveMotor`` now
  calls ``profile_from_genome``, with ``Genome.from_seed(seed)`` under the stub brain);
  ``reflex.py`` is the ONE reflex module ("neuro-reflex-2": the brain passes ``fear``,
  ``comfort``, ``walking`` and its own habituation ``hab`` / sensitisation ``sens``, and gets
  ``ReflexOutput.presented`` / ``hold`` / ``fallen`` / ``lifted``); ``signals.py`` is the ONE
  structured-noise module (``Stream``, ``OU``, ``Oscillator``, ``ou_exact`` added); the
  locomotor rhythm is ``motor.loco_step`` (``LocoInput.wander_scale`` added). The brain lane
  retires ``reflexes.py``, ``loco.py`` and ``noise.py``. Details: ``NOTES-brain-motor.md``.
* 2026-09-27 23:40 neuro:brain — nothing in the contract changed. For **neuro:senses** (live run
  of the merged brain on :8891, explorer 4107:60): ``TouchSense`` reported contact on 27 % of 50 Hz
  ticks with no hand, person or page stimulus (sites right/left = the LEG bodies, back, head),
  flickering on/off every ~0.1–0.2 s; the reflex arcs withdrew each time and the duck stood still
  for minutes. The robot has no leg touch sensor (touch = pet-detect on the head + IMU/joint
  disturbance): leg contacts with things belong in proprio (``blocked`` / ``brushed``), not in
  ``TouchSense``; and a flicker shorter than ~0.4 s is one touch. Details: NOTES-brain-motor.md.
* 2026-09-27 23:58 neuro:brain — additive: ``MotorCommand.docking`` (default False). The brain sets
  it while returning to the nest AND while backing off it (the nest is a dead end; owner phase 3:
  reversing only at the nest) — then ``vx`` may be negative (≥ ``REVERSE_VX``). **neuro:motor**:
  pass ``docking=cmd.docking`` to ``legal_walk`` in ``motor.step`` (today a reverse is clipped to
  0); **neuro:live**: ``BodyRequest.docking = cmd.docking or cmd.behaviour == "return_dock"``.
* 2026-09-28 00:27 neuro:brain — the brain side of the one design is done: ``brain.py`` runs on
  ``reflex.step`` (the one onset detector: its ``presented`` onsets are habituated once by
  ``habituation.py``), ``motor.loco_step`` and ``signals``; ``reflexes.py``, ``loco.py``,
  ``noise.py`` are deleted; new ``appraisal.py`` (what reflex events mean to drives, selection
  and learning); ``perceive.py`` keeps levels only. Nothing else in this file changed.
* 2026-09-28 01:00 neuro:integrate — additive: the camera's **floor line**
  (:class:`FreeSector`, ``CameraSense.floor`` / ``floor_t_ms``: duck-eye FL-v1's five free-floor
  sectors, MODEL from view geometry and light) so the duck sees the walls and furniture it faces
  and steers away before touching them (``boundary.py``); the world side makes ``wave``,
  ``roll_ball``, ``novel_object``, ``tv``, ``hello`` and ``knock`` too (:data:`STIMULI`);
  ``MotorCommand.bob`` is a slow (≤ 1 Hz) head sway locked to the waddle, not a stride-rate bob.
* 2026-09-28 05:50 neuro:integrate (continuation) — additive: the stimulus ``swap_battery`` (a
  person swaps in a charged pack: the world's ``x_battery``) and the names of the maintenance
  drives ``groom`` / ``peckish`` / ``vigil`` in :data:`DRIVES`.
* 2026-09-28 07:30 neuro:fix — additive: **moments** (:data:`MOMENTS`): the short head acts
  (a glance, a tilt, a double take, a look down, a moment of stillness, a head shake…) are
  point events that play *inside* the current behaviour, not behaviours that replace it (the
  review: 18–27 act changes a minute, most shorter than the body can render) —
  ``MotorCommand.moment`` / ``moment_seq`` (the motor overlays the moment's program on the
  current one) and ``BrainTelemetry.moment``; ``MotorCommand.back_arc``: curiosity's reverse
  arc beside a thing it investigated (owner phase 3: reversing only at the nest and as
  curiosity's back_arc; vx ≥ ``REVERSE_VX``, an arc). Also additive: the touch site ``body``
  (a touch the model cannot place better: the review found the touch model named sites it
  could not know), the stimulus ``carry_to_mat`` (a person carries it to its floor mat: the
  world's hand), ``legal_walk(..., back_arc=True)`` and ``BACK_ARC_VYAW``.
* 2026-09-28 20:00 dock:build — additive: the neural duck backs into its dock by itself
  (:mod:`duckmind.neuro.dock`, phase 3's measured J-hook). ``Sighting.facing``: a ``nest``
  sighting read as a tag (``range_m`` set) says the duck's heading off square-on to the tag
  (rad, + = its left; duck-eye NB-v1's tag pose; None: not read as a tag, or the mirror pose fits
  as well), and for such a reading ``range_m`` / ``bearing`` are from the trunk origin in the
  level frame, as the tag reader gives them. ``MotorCommand.docking`` now means: the twist is the
  docking choreography's own (:func:`dock_walk`: forward 0.30–0.40 — the U-turn and the tight
  arcs at 0.30 — reversing −0.40…−0.30), which the motor passes on as it is (no course-keeping
  trim, no burst/pause holds, no raise to 0.35) and the body link lets through (a stall while
  backing in is the dock's face, not a reflex stop); **neuro:live**: ``BodyRequest.docking =
  cmd.docking`` (no longer ``or behaviour == "return_dock"``). ``BrainTelemetry.dock``: the
  docking program's state (mode, phase and its plain words, PG, charging, tries, the duck's own
  pose belief in the dock's frame). ``return_dock`` is now 「自己回去充電」 (the body backs in).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from duckmind.i18n import msg

CONTRACT_VERSION = "neuro-contract-1"

# ── provenance ────────────────────────────────────────────────────────────────

SOURCES = ("real", "eye", "model", "none")
SOURCE_ZH = {
    "real": msg("neuro.sense_source.real"),
    "eye": msg("neuro.sense_source.eye"),
    "model": msg("neuro.sense_source.model"),
    "none": msg("neuro.sense_source.none"),
}
# What each SensorFrame channel is by default in the live prototype.
CHANNEL_SOURCE: dict[str, str] = {
    "body": "real",  # robotd robot.state: IMU, joints, odometry, fallen, applied move
    "proprio": "real",  # duck-body proprio events computed from robot.state (blocked, lifted…)
    "tof": "real",  # tofd tof.frame (most ducks have no ToF: then "none")
    "camera": "eye",  # light level and sightings from duck-eye; motion/looming: "model"
    "sound": "model",  # the mono microphone + duck-ear classes: modelled from world events
    "touch": "model",  # where the hand really touches (physics contact), as a touch sense
    "power": "model",  # battery level, dock contact, charging (the world's battery model)
    "clock": "real",  # local time of day (NTP on the board, the sim clock in the simulator)
}

# ── shared vocabularies ───────────────────────────────────────────────────────

POSTURES = ("sit", "stand", "walk", "transition", "unknown")
# duck-body proprio events (spec §5.2) and the fall/righting cycle (spec §5.4).
BODY_EVENTS = (
    "blocked",  # walked into something: little progress + heading jerk / ToF touch / leg load
    "brushed",  # heading kicked 15–30° without stopping (a thin thing)
    "stumbled",  # tripped: yaw rate > 3 rad/s with leg load ×2 or the trunk dipping
    "fell",  # safety.fallen rising edge
    "righted",  # up again by itself (tilt < 25°, fallen clear for 1 s)
    "righted_help",  # up again because someone helped
    "righting_failed",  # not up after the tries: needs help
    "lifted",  # picked up (legs unloaded, gravity/gyro transient)
    "set_down",  # put back on the floor
    "lay_down",  # it lay down on purpose (play dead): the fall / lift claims of that are not told (bodylink)
)
LIGHT_LEVELS = ("bright", "dim", "dark", "black")  # duck-eye's light classes (eye-v1)
VISUAL_KINDS = (
    "motion",  # something moved in view (no identity)
    "looming",  # something is coming at the face fast (expansion rate)
    "thing",  # an object on the floor (duck-eye things / curiosity cone)
    "toy",  # a thing the duck has learned is a toy (only from its own experience)
    "person",  # a person or part of one (MODEL until the person detector exists, spec §8.8)
    "hand",  # a hand close to the head (MODEL)
    "nest",  # the dock's AprilTag (duck-eye nesttag: real pipeline)
    "duck",  # another duck (MODEL)
    "light",  # a lamp or bright window
)
SOUND_KINDS = (
    "voice",  # someone talking
    "name",  # its name (the name KWS)
    "hello",  # a greeting word
    "clap",  # hand clap
    "bang",  # something dropped / door slam / a loud crack
    "door",  # a door opening or closing
    "footsteps",
    "music",
    "tv",
    "duck_call",  # another duck's call
    "noise",  # anything else loud enough to notice
)
# (2026-09-28 neuro:fix, additive: "body" — a touch on the body it cannot place: the robot has no
# skin; it feels the head and beak through the head joints (fast, placed) and the rest through
# pet-detect on its microphone (slow, unplaced) or, for a firm push, the IMU's jolt)
TOUCH_SITES = ("head", "back", "left", "right", "tail", "beak", "body")
VOCAL_TAGS = ("chirp", "inquire", "coo", "greet", "wheee", "alarm")  # content.TAGS + falls
LOOKS = ("scan", "look_up", "seek_nest", "fixate", "idle", "glance_down")  # duck-body looks

# ── walking rules (duck-body protocol, owner A2/A3; robotd limits) ────────────

MIN_STEP_VX = 0.30  # velstand does not step below this: vx is 0 or at least this
NORMAL_VX_MAX = 0.55  # owner A3 (measured 0/1,200 falls at 0.35–0.55)
APPROACH_VX = 0.40  # owner A3: approaching a specific thing, and arcs
ARC_VX_MIN = 0.35  # an arc is never slower than this
VYAW_MAX = 1.0  # |yaw rate| cap; turning happens only while walking (no in-place turns)
REVERSE_VX = -0.40  # reversing: only while docking and as curiosity's back_arc (owner, phase 3)
BACK_ARC_VYAW = 1.0  # curiosity's back_arc turns at the full yaw rate (curiosity.prims.back_arc)
# (dock:build) the docking choreography's own slowest forward step: dock:measure's U-turn and
# tight arcs (vx 0.30, vyaw ±1.0), measured in phase 3 (duck-body dockexec backin)
DOCK_VX_MIN = 0.30


def dock_walk(vx: float, vyaw: float) -> tuple[float, float]:
    """The docking choreography's own twist (:mod:`duckmind.neuro.dock`; owner 2026-09-28 「6.
    可以」 back-in, 「1. 可以」 reversing while docking): forward 0 or :data:`DOCK_VX_MIN`…
    :data:`APPROACH_VX`, reversing 0 or :data:`REVERSE_VX`…−:data:`MIN_STEP_VX`, |vyaw| ≤
    :data:`VYAW_MAX` and turning only while stepping."""
    if vx >= 0.5 * MIN_STEP_VX:
        vx = min(APPROACH_VX, max(DOCK_VX_MIN, vx))
    elif vx <= -0.5 * MIN_STEP_VX:
        vx = max(REVERSE_VX, min(-MIN_STEP_VX, vx))
    else:
        return 0.0, 0.0
    return float(vx), float(max(-VYAW_MAX, min(VYAW_MAX, vyaw)))


def legal_walk(
    vx: float,
    vyaw: float,
    *,
    approach: bool = False,
    docking: bool = False,
    back_arc: bool = False,
) -> tuple[float, float]:
    """The nearest command robotd and duck-body accept: ``vx`` is 0 or in
    [:data:`MIN_STEP_VX`, cap]; turning only while stepping (``vx = 0`` ⇒ ``vyaw = 0``); an arc
    (``|vyaw| > 0.3``) at least :data:`ARC_VX_MIN`; reversing only when ``docking`` — or as
    curiosity's reverse ARC beside a thing (``back_arc``: vx ≥ :data:`REVERSE_VX` with the full
    yaw rate, never straight back)."""
    cap = APPROACH_VX if approach else NORMAL_VX_MAX
    if vx < 0.0 and back_arc and not docking:
        if abs(vyaw) < 0.3:
            return 0.0, 0.0  # a back_arc turns; a straight reverse is not one
        return max(REVERSE_VX, vx), BACK_ARC_VYAW if vyaw > 0.0 else -BACK_ARC_VYAW
    if vx < 0.0:
        vx = max(REVERSE_VX, vx) if docking else 0.0
    elif vx > 0.0:
        vx = min(cap, max(MIN_STEP_VX, vx)) if vx >= 0.5 * MIN_STEP_VX else 0.0
    if vx == 0.0:
        return 0.0, 0.0
    vyaw = max(-VYAW_MAX, min(VYAW_MAX, vyaw))
    if vx > 0.0 and abs(vyaw) > 0.3:
        vx = max(vx, min(cap, ARC_VX_MIN))
    return vx, vyaw


# ── the sensor frame ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class BodySense:
    """REAL: robotd ``robot.state`` (50 Hz), as duck-body reads it."""

    posture: str = "sit"  # POSTURES: robotd policy + joints, confirmed by duck-body
    fallen: bool = False  # safety.fallen (60° for 0.2 s)
    limp: bool = False
    tilt: float = 0.0  # trunk tilt from vertical, rad (IMU gravity)
    roll: float = 0.0  # trunk roll, rad (+ = leaning to its left)
    pitch: float = 0.0  # trunk pitch, rad (+ = nose down)
    gyro: tuple[float, float, float] = (0.0, 0.0, 0.0)  # trunk rates rad/s (roll, pitch, yaw)
    head: tuple[float, float, float, float] = (0.349, 0.349, 0.0, 0.0)  # measured neck_pitch,
    # head_pitch, head_yaw, head_roll joint angles, rad (robotd's joint frame, HOME = 0.349 ×2)
    vx: float = 0.0  # move.applied forward, m/s (what robotd passed on, not the true speed)
    vyaw: float = 0.0  # move.applied yaw rate, rad/s
    odom: tuple[float, float, float] = (0.0, 0.0, 0.0)  # contact odometry x, y (m), yaw (rad),
    # in the duck's own frame from boot — it drifts; the brain may learn places in it
    leg_load: float | None = None  # leg current relative to quiet standing (1.0), None if unknown
    step_phase: float | None = None  # gait phase 0..1 from the leg joints while stepping
    mouth: float | None = None  # REAL robot.state joint 9, rad; None when unavailable
    active_skill: str | None = None  # REAL robot.state policy when a known one-shot is running


@dataclass(frozen=True, slots=True)
class TofSense:
    """REAL when fitted: tofd ``tof.frame`` (8×8, 45°×45°, 15 Hz) through duck-body's depth
    summary. Most ducks have none (``present`` False): every rule must work without it."""

    present: bool = False
    left_mm: int | None = None  # nearest valid non-floor return, left third of the view
    ahead_mm: int | None = None  # … centre third
    right_mm: int | None = None  # … right third
    drop_zones: int = 0  # cells in the heading where the floor should be but is not (≥ 4: drop)
    touch: bool = False  # ≥ 32 cells too near (nose against something)
    age_ms: int = 0
    # the height above the floor (mm) of what the ahead return is on, from the hit points (0: unknown); and whether the
    # ahead / side return is a LOW EDGE (a curb, a step of about 30 mm and more) the duck saw from afar and now carries in its
    # odometry because the beams pass over it inside ~0.4 m (duckbody.depth)
    ahead_top_mm: int = 0
    virtual: bool = False
    low_edge_mm: int = 0  # the height estimate (mm) of a wide low EDGE in view, else 0


@dataclass(frozen=True, slots=True)
class Sighting:
    """One thing in the camera's field of view (37.4° wide × 62.2° tall, where the head points).
    Only what is inside the view *and* visible in the current light is ever reported."""

    kind: str  # VISUAL_KINDS
    bearing: float  # rad from the trunk's forward axis, + = the duck's left
    elevation: float = 0.0  # rad, + = up (from the trunk's horizontal)
    range_m: float | None = None  # None when the eye cannot tell
    size: float = 0.05  # angular size, rad
    motion: float = 0.0  # 0..1 image motion of this blob
    looming: float = 0.0  # expansion rate 1/s (dθ/dt ÷ θ); > 0 = coming closer
    track: int = 0  # stable id while it stays tracked (0 = untracked)
    source: str = "model"  # "eye" or "model"
    # (dock:build 2026-09-28, additive) a ``nest`` sighting read as a tag: the duck's heading off
    # square-on to the tag (rad, + = its left), as duck-eye NB-v1's tag pose gives it; None when
    # not read as a tag or when the mirror pose fits as well (the range and bearing still hold)
    facing: float | None = None
    axis: float | None = None  # visible floor object's long axis, rad modulo π; None if unknown
    look: tuple[float, ...] | None = None  # MODEL person appearance signature (six dimensions)
    lux: float | None = None  # MODEL brightness of this visible patch, lux
    # (telemetry only, 2026-10-03) the loom watch reports a looming with no identity (``track`` 0: a
    # watch report never poses as a new thing); this is the eye's current track of the same visible
    # object when the camera model has one (the object association the looks use), 0 = none. Nothing
    # in the brain decides by it: the reflex arc copies it into the reflex's ``reflex_track`` record
    source_track: int = 0


FLOOR_ENDS = ("wall", "thing", "unknown")  # duck-eye FL-v1's ends (a drop is the ToF's)
FLOOR_RANGE_M = 1.5  # FL-v1 looks no further (MAX_RANGE_MM): "unknown" at this distance


@dataclass(frozen=True, slots=True)
class FreeSector:
    """One sector of the camera's floor line — the same shape as duck-eye FL-v1's ``sectors``
    (:mod:`duckmind.eye.floorline`): how far the floor is clear in that direction and what ends
    it. Only what the lens sees in its light: none in the dark, and only inside the picture
    (about 37° wide, where the head points). MODEL in the prototype (view geometry and light,
    :mod:`duckmind.world.neuro_senses`); the eye's own floor line is phase 5's."""

    bearing: float  # rad from the trunk's forward axis at the look, + = the duck's left
    width: float = 0.13  # rad, the sector's width
    free_m: float = FLOOR_RANGE_M  # clear floor from the trunk origin (level frame), m
    end: str = "unknown"  # FLOOR_ENDS: what ends the clear stretch ("unknown": nothing seen)


@dataclass(frozen=True, slots=True)
class CameraSense:
    """The head camera: ``eye`` for light/things/nest (phase-3 duck-eye, ~2 looks/s),
    ``model`` for motion, looming, person, hand and duck (no validated detector yet).

    ``floor`` is the latest floor line (:class:`FreeSector` × 5, left to right as the picture
    goes, i.e. bearings falling) and ``floor_t_ms`` the brain time of the look it came from
    (``None``: no floor line yet, or too dark for one) — the bearings are relative to the trunk
    *at that look*, so a brain that keeps its own odometry can place them."""

    usable: bool = False  # a frame was analysed recently (eye up, not throttled)
    light: str = "bright"  # LIGHT_LEVELS
    brightness: float = 0.7  # 0..1 frame brightness (the camera as a light sensor)
    motion: float = 0.0  # 0..1 global image motion in view
    sightings: tuple[Sighting, ...] = ()
    age_ms: int = 0  # how old the newest analysed frame is
    source: str = "eye"
    floor: tuple[FreeSector, ...] = ()
    floor_t_ms: int | None = None
    lux: float | None = None  # MODEL log-meter reading in lux, None before a fresh pass
    roof: float | None = None  # MODEL roof distance above camera in metres; 9.9 means open sky
    roof_t_ms: int | None = None  # time of roof reading; None means no valid roof observation


@dataclass(frozen=True, slots=True)
class SoundSense:
    """MODEL: the duck's mono microphone and duck-ear's classes. ``bearing`` stays None on the
    real duck (a mono mic hears no direction); a model may give a noisy one only when it
    assumes a microphone array, and must mark it so in ``bearing_source``."""

    level_db: float = 30.0  # sound level at the duck, dB (a quiet room ~30, a clap near ~75)
    onset: float = 0.0  # 0..1 onset strength this tick (a sudden rise over the background)
    kind: str | None = None  # SOUND_KINDS of the current sound event, None = nothing notable
    bearing: float | None = None  # rad, + = left; None = unknown (the real duck)
    bearing_source: str = "none"  # "none" | "model"
    source: str = "model"
    voice: tuple[float, float] | None = None  # MODEL speaker signature, valid for this speech pass
    voice_t_ms: int | None = None  # time of the speaker reading; None when voice is invalid
    # WHAT made the sound, as a kind only (never a name or an id): "self" = the duck's own contact
    # (it bumped something, something it had pushed fell, a thing hit it), "thing" = a thing it did
    # not touch, "person", "page" (the page's bang or clap), "" = not told (the synthetic world, a
    # real microphone, old recordings). The brain's reflexes do not read it; the experience observer
    # does, so that a sound the duck made itself is never a "noise" to Duck Core's growth
    # (docs/design/SELF-NOISE-2026-10-11.md).
    by: str = ""


@dataclass(frozen=True, slots=True)
class TouchSense:
    """MODEL: where the hand is really in contact with the body (physics contact in the
    simulator). On the robot: pet-detect's head scratch + IMU/joint disturbance."""

    site: str | None = None  # TOUCH_SITES, None = no contact
    pressure: float = 0.0  # 0..1
    stroke: float = 0.0  # 0..1 how much it is a stroking (sliding) contact
    source: str = "model"
    # WHO touched it, the kind of the world's contact record ("hand": the kit hand a person's pat
    # or poke comes through, "person": a family member's body, "thing": a movable object such as
    # a slipper or a toy; never a name or an id; "" = not told: the synthetic world and old
    # recordings). The brain's reflexes do not read it; the experience observer does, so that
    # only a hand is ever a "touch" or a "poke" to Duck Core's growth
    # (docs/design/REAL-HOME-FIX-2026-10-10.md).
    by: str = ""


@dataclass(frozen=True, slots=True)
class PowerSense:
    """MODEL in the prototype (robot.health voltage EMA and the dock contact on the robot)."""

    battery: float = 0.8  # 0..1 state of charge
    on_dock: bool = False  # sitting on the charging nest's contacts
    charging: bool = False
    hot: bool = False  # SoC ≥ 85 °C: look less, rest
    source: str = "model"


@dataclass(frozen=True, slots=True)
class SensorFrame:
    """Everything the brain may know at ``t_ms``. Built by neuro:senses."""

    t_ms: int
    body: BodySense = field(default_factory=BodySense)
    events: tuple[str, ...] = ()  # BODY_EVENTS that happened since the last frame
    tof: TofSense = field(default_factory=TofSense)
    camera: CameraSense = field(default_factory=CameraSense)
    sound: SoundSense = field(default_factory=SoundSense)
    touch: TouchSense = field(default_factory=TouchSense)
    power: PowerSense = field(default_factory=PowerSense)
    day_minute: float = 600.0  # local time of day, minutes 0..1440 (the duck's clock)
    sources: Mapping[str, str] = field(default_factory=lambda: dict(CHANNEL_SOURCE))

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out["sources"] = dict(self.sources)
        return out

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> SensorFrame:
        cam = dict(data.get("camera") or {})
        sightings = []
        for item in cam.get("sightings") or ():
            sighting = dict(item)
            if sighting.get("look") is not None:
                sighting["look"] = tuple(sighting["look"])
            sightings.append(Sighting(**sighting))
        cam["sightings"] = tuple(sightings)
        cam["floor"] = tuple(FreeSector(**s) for s in cam.get("floor") or ())
        body = dict(data.get("body") or {})
        for key in ("gyro", "head", "odom"):
            if key in body:
                body[key] = tuple(body[key])
        sound = dict(data.get("sound") or {})
        if sound.get("voice") is not None:
            sound["voice"] = tuple(sound["voice"])
        return cls(
            t_ms=int(data["t_ms"]),
            body=BodySense(**body),
            events=tuple(data.get("events") or ()),
            tof=TofSense(**(data.get("tof") or {})),
            camera=CameraSense(**cam),
            sound=SoundSense(**sound),
            touch=TouchSense(**(data.get("touch") or {})),
            power=PowerSense(**(data.get("power") or {})),
            day_minute=float(data.get("day_minute", 600.0)),
            sources=dict(data.get("sources") or CHANNEL_SOURCE),
        )


# ── the motor command ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class HeadPose:
    """A head pose in Duck Core's clip units: fractions of ``content.SEATED_ENVELOPE`` added to
    HOME, with an ABSOLUTE side (``hy``/``hr`` + = toward the duck's left). Convert with
    ``content.head_targets({"np": np, "hp": hp, "hy": hy, "hr": hr, "m": mouth}, side=+1)``.
    ``np`` + leans the neck forward/down, ``hp`` + looks up, ``mouth`` 0..1 opening."""

    np: float = 0.0
    hp: float = 0.0
    hy: float = 0.0
    hr: float = 0.0
    mouth: float = 0.0


@dataclass(frozen=True, slots=True)
class Gaze:
    """Where attention points (the orienting system's output), in the trunk frame. The motor
    turns the head toward it inside the envelope; the brain turns the body when it is beyond."""

    bearing: float = 0.0  # rad, + = the duck's left
    elevation: float = 0.0  # rad, + = up
    gain: float = 0.0  # 0..1 how much it overrides the program's own head yaw/pitch
    speed: float = 0.5  # 0..1 urgency: 1 = a reflex saccade, 0.2 = a slow drift


@dataclass(frozen=True, slots=True)
class Target:
    """What a behaviour is about, as the senses gave it (egocentric) or as learned (odometry)."""

    kind: str  # VISUAL_KINDS, "sound", "touch", "place", "nest", "threat"
    bearing: float  # rad, + = left (from the trunk now)
    range_m: float | None = None
    # the eye's track of the sighting the target is about (thing, person, nest); of what scared the duck
    # for a ``threat``, and of the remembered thing for a ``place`` that is one's believed place
    # (telemetry: nothing decides by it); 0 = none
    track: int = 0
    site: str | None = None  # a touch site when kind == "touch"


@dataclass(frozen=True, slots=True)
class MotorCommand:
    """The brain's output each tick. Discrete part: which behaviour (a motor program the motor
    lane owns) with this bout's parameters. Continuous part: locomotion, posture, gaze, reflex
    head overlay, liveliness and vocal events."""

    t_ms: int
    behaviour: str  # BEHAVIOURS key
    bout: int  # increments at every new bout: restart the program with the new parameters
    variant: int  # per-bout seed for the program's own variation (never the same twice)
    intensity: float = 0.5  # 0..1 how strongly (scales with the stimulus/drive behind it)
    tempo: float = 1.0  # × the program's nominal speed (arousal, temperament, stage), 0.5..1.6
    amplitude: float = 0.6  # 0..1 of the stage's envelope (STAGES amplitude cap × caps)
    side: int = 1  # +1 left / −1 right: the innate side, or the stimulus's side this bout
    target: Target | None = None
    phase: float = 0.0  # 0..1 how far the bout's motivation is consumed (satiation)
    vx: float = 0.0  # m/s, already legal (legal_walk)
    vyaw: float = 0.0  # rad/s, already legal
    approach: bool = False  # walking toward a specific thing (A2/A3: 0.40, head may go down)
    # a flight in panic (flee / retreat, legs in working order): the body link may run it as robotd's panic
    # skill (the panic network, 2026-10-09) when the table has one; v5 walks it at vx otherwise
    panic: bool = False
    # (2026-10-09) turn the body toward something it is looking at by pivoting on the spot (+ left, - right, 0 none) while the
    # heading still has far to go: the body link runs it as robotd's quick-turn skill when the table has one; the arc in
    # vx / vyaw above is what v5 walks otherwise (and finishes the last stretch either way)
    pivot: float = 0.0
    posture: str = "sit"  # "sit" | "stand" wanted (the motor toggles through robotd)
    gaze: Gaze = field(default_factory=Gaze)
    head: HeadPose | None = None  # a reflex pose (startle, withdraw, freeze…) or None
    head_blend: float = 0.0  # 0..1 weight of ``head`` over the program's pose
    stabilise: float = 0.0  # 0..1 head stabilisation gain against trunk motion (gyro)
    bob: float = 0.0  # 0..1 the walking head SWAY: a slow (≤ 1 Hz) side-to-side sway locked
    # to every other step of the waddle — ducks waddle, they do not head-bob like pigeons, and a
    # stride-rate (~2 Hz) bob is beyond the head's bandwidth (pet:body); 2026-09-28 integration
    alive: float = 0.5  # 0..1 amplitude of the idle micro-motion (breathing, small glances)
    vocal: str | None = None  # VOCAL_TAGS, emitted once when vocal_seq changes
    vocal_seq: int = 0
    vocal_level: float = 0.5  # 0..1 loudness/energy of the call
    look: str | None = None  # LOOKS: a camera look the brain asks for (a hint to duck-eye)
    reflex: str | None = None  # REFLEXES active now (drives the reflex overlay)
    # the eye's track of the sighting whose onset started that reflex (0: none; telemetry only: the
    # motor lane ignores it), so a record can say what the reflex was about
    reflex_track: int = 0
    asleep: bool = False  # sleeping: eyes shut, head low, no looks
    # (dock:build 2026-09-28) the twist is the docking choreography's own (dock_walk): going in
    # backwards and walking out of the dock; the motor and the body link pass it on as it is
    docking: bool = False
    # (2026-09-28 neuro:fix, additive) a moment (MOMENTS): a short head act played inside the
    # current behaviour, started once when ``moment_seq`` changes
    moment: str | None = None
    moment_seq: int = 0
    # curiosity's back_arc (owner phase 3): backing away beside a thing it investigated, an arc
    back_arc: bool = False

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


# ── telemetry ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Candidate:
    behaviour: str
    salience: float  # the basal-ganglia input (drive × stimulus × temperament × learning)
    activation: float  # its channel's activation after competition (0..1)
    why_zh: str = ""  # its strongest reason, plain Chinese


@dataclass(frozen=True, slots=True)
class ReflexEvent:
    t_ms: int
    reflex: str  # REFLEXES
    stimulus: str  # a stimulus class, e.g. "sound:clap", "touch:back", "visual:looming"
    strength: float  # the response (after habituation × sensitisation × arousal gain)
    gain: float  # habituation × sensitisation at that moment (1 = fresh)
    zh: str = ""
    # (fid:fix 2026-09-28, additive) the visible motions the words name, planned from where the
    # MEASURED head is now (REFLEX_PARTS): the page's witness judges each one on the body
    parts: tuple[str, ...] = ()
    # the eye's track of the sighting whose onset started it (0: none — a sound, a touch, a light
    # change); telemetry, so a record can say what the reflex was about
    track: int = 0


# The motions a reflex row may name (ReflexEvent.parts) and what each one means on the body
# (the page's witness judges them in the reflex's own time: a flinch 0.4 s, an orient 0.6 s, a
# lean 1.5 s): the nose thrown up/back ≥ 10°, jerked down ≥ 10°, the neck pulled back ≥ 8°, the
# head turned ≥ 10° to the duck's left / right, the nose lowered ≥ 10°.
REFLEX_PARTS: dict[str, str] = {
    "up": msg("neuro.body.part.up"),
    "dip": msg("neuro.body.part.dip"),
    "neck": "脖子一縮",
    "yaw_left": "頭轉向左邊",
    "yaw_right": "頭轉向右邊",
    "down": msg("neuro.body.part.down"),
}


@dataclass(frozen=True, slots=True)
class BrainTelemetry:
    t_ms: int
    behaviour: str
    behaviour_zh: str
    reason_zh: str  # why it is doing this, plain Chinese
    bout: int
    bout_ms: int  # how long this bout has run
    satiation: float  # 0..1 of the current bout
    drives: Mapping[str, float]  # DRIVES → 0..1
    arousal: float  # the neuromodulator, 0..1
    valence: float  # mood, −1..1
    reflex: str | None  # the reflex holding the body now, if any
    reflexes: tuple[ReflexEvent, ...]  # the most recent reflex events (newest last)
    competition: tuple[Candidate, ...]  # the top candidates, strongest first
    habituation: Mapping[str, float]  # stimulus class → response gain (1 fresh … → 0)
    sensitisation: Mapping[str, float]  # stimulus class → extra gain (0 = none)
    learned: Mapping[str, Any]  # places, avoided spots, liked spots, things, cue values
    locomotion: Mapping[str, float]  # go-drive, burst state, speed, curvature
    stage: str  # growth stage id
    counts: Mapping[str, int]  # bouts started per behaviour since boot (moments: times played)
    moment: str | None = None  # the moment playing now (MOMENTS), if any
    # (dock:build 2026-09-28, additive) the docking program: mode (idle | docking | home |
    # leaving), phase and its plain words, pg, charging, tries, its own pose belief in the dock's
    # frame [x mm, y mm, heading °] (see neuro.dock.DockRun.view)
    dock: Mapping[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        for key in ("drives", "habituation", "sensitisation", "locomotion", "counts"):
            out[key] = dict(getattr(self, key))
        out["learned"] = dict(self.learned)
        out["dock"] = None if self.dock is None else dict(self.dock)
        return out


# ── drives, reflexes, stimulus classes ────────────────────────────────────────

DRIVES: dict[str, str] = {
    "energy": msg("neuro.drive.energy"),  # 1 rested … 0 exhausted (fatigue = 1 − energy)
    "sleep_pressure": msg("neuro.drive.sleep_pressure"),
    "alertness": "生理時鐘（清醒度）",  # the circadian phase, chronotype-shifted
    "curiosity": msg("neuro.drive.curiosity"),
    "fear": msg("neuro.drive.fear"),
    "social": msg("neuro.drive.social"),
    "comfort": msg("neuro.drive.comfort"),
    "boredom": msg("neuro.drive.boredom"),
    "restless": msg("neuro.drive.restless"),
    "hunger": msg("neuro.drive.hunger"),  # battery hunger: the pull back to the nest
    "play": msg("neuro.drive.play"),
    # (2026-09-28 integration, additive: the maintenance needs, drives.py)
    "groom": msg("neuro.drive.groom"),
    "peckish": msg("neuro.drive.peckish"),
    "vigil": msg("neuro.drive.vigil"),
}
REFLEXES: dict[str, str] = {
    "orient": msg("neuro.reflex.orient"),
    "startle": msg("neuro.reflex.startle"),
    "freeze": msg("neuro.reflex.freeze"),
    "withdraw": msg("neuro.reflex.withdraw"),
    "lean_in": msg("neuro.reflex.lean_in"),  # an expected gentle touch: lean into the hand
    "veer": msg("neuro.reflex.veer"),
    "drop_stop": msg("neuro.reflex.drop_stop"),
    "bump_stop": msg("neuro.reflex.bump_stop"),
    "righting": msg("neuro.reflex.righting"),
    "lifted": msg(
        "neuro.reflex.lifted"
    ),  # (fid:fix: its own sense — the page's body line says if it was)
    "stabilise": msg("neuro.reflex.stabilise"),
}
# Stimulus classes the habituation system keeps a level for ("channel:kind").
STIMULUS_CLASSES = (
    tuple(f"sound:{kind}" for kind in SOUND_KINDS)
    + tuple(f"touch:{site}" for site in TOUCH_SITES)
    + ("visual:motion", "visual:looming", "visual:thing", "visual:person", "visual:hand")
    + ("light:on", "light:off", "body:bump", "body:lifted")
)

# ── the page's scenario stimuli (into the WORLD; the senses lane reports what the duck senses)


@dataclass(frozen=True, slots=True)
class StimulusSpec:
    zh: str
    channels: tuple[str, ...]  # which senses can pick it up (if it reaches them)
    note: str


STIMULI: dict[str, StimulusSpec] = {
    "clap": StimulusSpec(
        msg("neuro.stimulus.clap.label"), ("sound",), "a clap ~1 m away: onset, ~75 dB"
    ),
    "call_name": StimulusSpec(
        msg("neuro.stimulus_class.sound.name"), ("sound",), "its name, KWS class 'name'"
    ),
    "voice": StimulusSpec(msg("neuro.stimulus.voice.label"), ("sound",), "a few seconds of talk"),
    "bang": StimulusSpec(msg("neuro.stimulus.bang.label"), ("sound",), "a sudden ~90 dB bang"),
    "door": StimulusSpec(msg("neuro.stimulus.door.label"), ("sound",), "a door"),
    "music": StimulusSpec(msg("neuro.stimulus.music.label"), ("sound",), "music for a while"),
    "pet_head": StimulusSpec("摸摸頭", ("touch",), "gentle stroke on the head"),
    "stroke_back": StimulusSpec(
        msg("neuro.stimulus.stroke_back.label"), ("touch",), "gentle stroke along the back"
    ),
    "poke": StimulusSpec(
        msg("neuro.stimulus.poke.label"), ("touch",), "a sharp poke on its side, unannounced"
    ),
    "hand_loom": StimulusSpec(
        msg("neuro.stimulus.hand_loom.label"), ("camera",), "a hand coming fast at the face"
    ),
    "toy_drop": StimulusSpec(
        msg("neuro.stimulus.toy_drop.label"), ("sound", "camera"), "a toy lands nearby"
    ),
    "walk_by": StimulusSpec(
        msg("neuro.stimulus.family_walk.label"), ("sound", "camera"), "someone walks past"
    ),
    "sit_near": StimulusSpec(
        msg("neuro.stimulus.sit_near.label"), ("sound", "camera"), "someone sits close by"
    ),
    "lights_off": StimulusSpec(
        msg("neuro.stimulus.lights_off.label"), ("camera",), "room light off"
    ),
    "lights_on": StimulusSpec(msg("neuro.stimulus.lights_on.label"), ("camera",), "room light on"),
    "pick_up": StimulusSpec(
        msg("neuro.stimulus.pick_up.label"), ("body", "touch"), "lifted off the floor"
    ),
    "set_down": StimulusSpec(
        msg("neuro.stimulus.set_down.label"), ("body", "touch"), "put back on the floor"
    ),
    "push": StimulusSpec(
        msg("neuro.stimulus.push.label"), ("body", "touch"), "a push (it may fall)"
    ),
    # (2026-09-28 integration, additive: the world side makes these too — neuro:senses)
    "wave": StimulusSpec("在牠前面揮手", ("camera",), "a hand waving in front of the lens"),
    "roll_ball": StimulusSpec(
        msg("neuro.stimulus.roll_ball.label"), ("camera", "touch"), "a ball rolled toward it"
    ),
    "novel_object": StimulusSpec(
        msg("neuro.stimulus.novel_object.label"), ("camera",), "a thing it has not seen, nearby"
    ),
    "tv": StimulusSpec(
        msg("neuro.stimulus.tv.label"), ("sound",), "the TV on or off (it masks voices)"
    ),
    "hello": StimulusSpec(msg("neuro.stimulus.hello.label"), ("sound",), "a greeting word"),
    "knock": StimulusSpec(msg("neuro.stimulus.knock.label"), ("sound",), "a knock at the door"),
    # (2026-09-28 integration, additive: a person swaps in a charged pack — the world's battery
    # lasts ~30–60 min; the duck feels it only through its power sense)
    "swap_battery": StimulusSpec(
        msg("neuro.stimulus.swap_battery.label"), ("power",), "a charged pack swapped in"
    ),
    # (2026-09-28 neuro:fix, additive: the family carries it out of a tight spot to the mat — the
    # world's hand; it feels itself lifted and set down)
    "carry_to_mat": StimulusSpec(
        msg("neuro.stimulus.carry_to_mat.label"),
        ("body", "touch"),
        "picked up and set down on the floor mat",
    ),
}

# ── the repertoire ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class BehaviourSpec:
    """One discrete behaviour. The brain decides *when* and with what parameters; the motor
    lane owns *how* (a parametric program: never the identical movement twice)."""

    zh: str
    family: str  # verb | signature | pause | moment | ethogram | locomotion | rest | reflex
    unlock_gd: int  # base growth day (the duck's own pace applies, as in Duck Core)
    posture: str  # "sit" | "stand" | "any": what the program needs
    locomotes: bool  # it walks (vx/vyaw come from the brain's locomotor rhythm)
    sounds: tuple[str, ...]  # vocal tags it may use ("" = may be silent)
    body_zh: str  # what the body does (for the motor lane and the page)
    verb: str | None = None  # the Duck Core verb it is (content.VERB_BY_ID), if any
    branch: str | None = None  # a branch signature: only a duck of this branch
    sleep: bool = False  # performed asleep
    # it turns or works on one spot on purpose (sunbathing: into the light and a look down, again;
    # the settling circle; going round a toy): its steps inside a small circle are the act, not a
    # failure to get anywhere. The unstick sense (``unstick.py``) does not count them toward
    # "confined"; pushing into something is still pushing, in these acts too.
    stationary: bool = False


def _b(
    zh: str,
    family: str,
    unlock_gd: int,
    posture: str,
    body_zh: str,
    sounds: tuple[str, ...] = ("",),
    *,
    locomotes: bool = False,
    verb: str | None = None,
    branch: str | None = None,
    sleep: bool = False,
    stationary: bool = False,
) -> BehaviourSpec:
    return BehaviourSpec(
        zh, family, unlock_gd, posture, locomotes, sounds, body_zh, verb, branch, sleep, stationary
    )


BEHAVIOURS: dict[str, BehaviourSpec] = {
    # Duck Core v4 verbs (content.VERBS): same ids, same unlock days, same sounds.
    "look_around": _b(
        msg("neuro.act.look_around.name"),
        "verb",
        0,
        "any",
        msg("neuro.act.look_around.body"),
        ("", "chirp", "inquire"),
        verb="look_around",
    ),
    "perk_tilt": _b(
        msg("neuro.act.perk_tilt.name"),
        "verb",
        0,
        "any",
        msg("neuro.act.perk_tilt.body"),
        ("inquire",),
        verb="perk_tilt",
    ),
    "stretch_yawn": _b(
        msg("neuro.act.stretch_yawn.name"),
        "verb",
        0,
        "any",
        msg("neuro.act.stretch_yawn.body"),
        ("coo",),
        verb="stretch_yawn",
    ),
    "preen": _b(
        msg("neuro.act.preen.name"), "verb", 0, "any", msg("neuro.act.preen.body"), verb="preen"
    ),
    "happy_sway": _b(
        msg("neuro.act.happy_sway.name"),
        "verb",
        0,
        "any",
        msg("neuro.act.happy_sway.body"),
        ("chirp",),
        verb="happy_sway",
    ),
    "tuck_away": _b(
        msg("neuro.act.tuck_away.name"),
        "verb",
        0,
        "any",
        msg("neuro.act.tuck_away.body"),
        ("", "coo"),
        verb="tuck_away",
    ),
    "nap": _b(
        msg("neuro.act.nap.name"),
        "rest",
        0,
        "sit",
        msg("neuro.act.nap.body"),
        ("", "chirp"),
        verb="nap",
        sleep=True,
    ),
    "wake_ritual": _b(
        msg("neuro.act.wake_ritual.name"),
        "verb",
        3,
        "any",
        msg("neuro.act.wake_ritual.body"),
        ("coo", "chirp"),
        verb="wake_ritual",
    ),
    "nod": _b(msg("neuro.act.nod.name"), "verb", 14, "any", msg("neuro.act.nod.body"), verb="nod"),
    "peekaboo": _b(
        msg("neuro.act.peekaboo.name"),
        "verb",
        14,
        "any",
        msg("neuro.act.peekaboo.body"),
        ("inquire", "chirp"),
        verb="peekaboo",
    ),
    "call_response": _b(
        msg("neuro.act.call_response.name"),
        "verb",
        18,
        "any",
        msg("neuro.act.call_response.body"),
        ("inquire", "chirp"),
        verb="call_response",
    ),
    "snuggle": _b(
        msg("neuro.act.snuggle.name"),
        "verb",
        28,
        "any",
        msg("neuro.act.snuggle.body"),
        ("coo",),
        verb="snuggle",
    ),
    "little_dance": _b(
        msg("neuro.act.little_dance.name"),
        "verb",
        28,
        "any",
        msg("neuro.act.little_dance.body"),
        ("chirp", "wheee"),
        verb="little_dance",
    ),
    "not_now": _b(
        msg("neuro.act.not_now.name"),
        "verb",
        60,
        "any",
        msg("neuro.act.not_now.body"),
        verb="not_now",
    ),
    "little_song": _b(
        msg("neuro.act.little_song.name"),
        "signature",
        45,
        "any",
        msg("neuro.act.little_song.body"),
        ("chirp", "greet"),
        verb="little_song",
        branch="songbird",
    ),
    "self_duet": _b(
        msg("neuro.act.self_duet.name"),
        "signature",
        75,
        "any",
        msg("neuro.act.self_duet.body"),
        ("chirp", "inquire"),
        verb="self_duet",
        branch="songbird",
    ),
    "survey": _b(
        msg("neuro.act.survey.name"),
        "signature",
        45,
        "any",
        msg("neuro.act.survey.body"),
        ("inquire",),
        verb="survey",
        branch="explorer",
    ),
    "peekaboo_master": _b(
        msg("neuro.act.peekaboo_master.name"),
        "signature",
        75,
        "any",
        msg("neuro.act.peekaboo_master.body"),
        ("inquire", "chirp"),
        verb="peekaboo_master",
        branch="explorer",
    ),
    "melt": _b(
        msg("neuro.act.melt.name"),
        "signature",
        45,
        "sit",
        msg("neuro.act.melt.body"),
        ("coo",),
        verb="melt",
        branch="snuggler",
    ),
    "nuzzle_tilt": _b(
        msg("neuro.act.nuzzle_tilt.name"),
        "signature",
        75,
        "any",
        msg("neuro.act.nuzzle_tilt.body"),
        ("coo",),
        verb="nuzzle_tilt",
        branch="snuggler",
    ),
    "moon_gaze": _b(
        msg("neuro.act.moon_gaze.name"),
        "signature",
        45,
        "any",
        msg("neuro.act.moon_gaze.body"),
        verb="moon_gaze",
        branch="dreamer",
    ),
    "sleep_talk": _b(
        msg("neuro.act.sleep_talk.name"),
        "signature",
        75,
        "sit",
        msg("neuro.act.sleep_talk.body"),
        ("coo",),
        verb="sleep_talk",
        branch="dreamer",
        sleep=True,
    ),
    "stand_up": _b(
        msg("neuro.act.stand_up.name"),
        "verb",
        0,
        "any",
        msg("neuro.act.stand_up.body"),
        ("", "chirp"),
        verb="stand_up",
    ),
    "sit_down": _b(
        msg("neuro.act.sit_down.name"),
        "verb",
        0,
        "any",
        msg("neuro.act.sit_down.body"),
        ("", "coo"),
        verb="sit_down",
    ),
    "wander": _b(
        msg("neuro.act.wander.name"),
        "locomotion",
        5,
        "stand",
        msg("neuro.act.wander.body"),
        ("", "inquire", "chirp"),
        locomotes=True,
        verb="wander",
    ),
    "turn_around": _b(
        msg("neuro.act.turn_around.name"),
        "locomotion",
        5,
        "stand",
        msg("neuro.act.turn_around.body"),
        ("", "inquire"),
        locomotes=True,
        verb="turn_around",
    ),
    # Duck Core pause and moment verbs.
    "peek_down": _b(
        msg("neuro.act.peek_down.name"),
        "pause",
        0,
        "any",
        msg("neuro.act.peek_down.body"),
        verb="peek_down",
    ),
    "glance": _b(
        msg("neuro.act.glance.name"), "pause", 0, "any", msg("neuro.act.glance.body"), verb="glance"
    ),
    "tilt": _b(
        msg("neuro.act.tilt.name"), "pause", 0, "any", msg("neuro.act.tilt.body"), verb="tilt"
    ),
    "dabble": _b(
        msg("neuro.act.dabble.name"),
        "moment",
        0,
        "any",
        msg("neuro.act.dabble.body"),
        verb="dabble",
    ),
    "double_take": _b(
        msg("neuro.act.double_take.name"),
        "moment",
        0,
        "any",
        msg("neuro.act.double_take.body"),
        verb="double_take",
    ),
    "proud_puff": _b(
        msg("neuro.act.proud_puff.name"),
        "moment",
        0,
        "any",
        msg("neuro.act.proud_puff.body"),
        verb="proud_puff",
    ),
    "still": _b(
        msg("neuro.act.still.name"), "moment", 0, "any", msg("neuro.act.still.body"), verb="still"
    ),
    "freeze": _b(
        msg("neuro.act.freeze.name"),
        "moment",
        0,
        "any",
        msg("neuro.act.freeze.body"),
        verb="freeze",
    ),
    "glance_away": _b(
        msg("neuro.act.glance_away.name"),
        "moment",
        0,
        "any",
        msg("neuro.act.glance_away.body"),
        verb="glance_away",
    ),
    # Duck ethogram acts this body can do.
    "peck": _b(msg("neuro.act.peck.name"), "ethogram", 0, "any", msg("neuro.act.peck.body"), ("",)),
    "forage": _b(
        msg("neuro.act.forage.name"),
        "ethogram",
        5,
        "stand",
        msg("neuro.act.forage.body"),
        ("", "chirp"),
        locomotes=True,
    ),
    "nibble": _b(
        msg("neuro.act.nibble.name"),
        "ethogram",
        0,
        "any",
        msg("neuro.act.nibble.body"),
        ("", "inquire"),
    ),
    "head_shake": _b(
        msg("neuro.act.head_shake.name"),
        "ethogram",
        0,
        "any",
        msg("neuro.act.head_shake.body"),
        ("",),
    ),
    "body_shake": _b(
        msg("neuro.act.body_shake.name"),
        "ethogram",
        5,
        "stand",
        msg("neuro.act.body_shake.body"),
        ("",),
    ),
    # The tricks a person can teach (phase 2 「學會聽話」, neuro.tricks): channels the selector only
    # takes up when a command claims them (no wiring of their own: nothing in the duck's day asks
    # for a bow or a spin). ``shake`` is ``body_shake``, which the duck also does by itself.
    "bow": _b(
        msg("page.cmd.verb.bow"),
        "ethogram",
        5,
        "stand",
        msg("neuro.skill.doing.bow"),
        ("greet", "chirp"),
    ),
    "lie_low": _b(
        msg("page.motion.name.lie_low"),
        "ethogram",
        5,
        "stand",
        msg("neuro.skill.doing.lie"),
        ("", "coo"),
    ),
    "spin": _b(
        msg("page.cmd.verb.spin"),
        "ethogram",
        5,
        "stand",
        msg("neuro.skill.doing.turn"),
        ("", "wheee"),
        locomotes=True,
        stationary=True,  # (it turns on one spot on purpose: the unstick sense does not count it)
    ),
    "flip": _b(
        msg("page.cmd.verb.flip"),
        "ethogram",
        45,
        "stand",
        msg("page.cmd.verb.flip"),
        ("", "wheee"),
    ),
    "investigate": _b(
        msg("neuro.act.investigate.name"),
        "ethogram",
        5,
        "stand",
        msg("neuro.act.investigate.body"),
        ("", "inquire"),
        locomotes=True,
    ),
    "follow": _b(
        msg("neuro.act.follow.name"),
        "ethogram",
        5,
        "stand",
        msg("neuro.act.follow.body"),
        ("", "chirp", "greet"),
        locomotes=True,
    ),
    "greet": _b(
        msg("neuro.act.greet.name"),
        "ethogram",
        0,
        "any",
        msg("neuro.act.greet.body"),
        ("greet", "chirp"),
    ),
    "call_out": _b(
        msg("neuro.act.call_out.name"),
        "ethogram",
        0,
        "any",
        msg("neuro.act.call_out.body"),
        ("chirp", "greet", "inquire"),
    ),
    "watch": _b(
        msg("neuro.act.watch.name"), "ethogram", 0, "any", msg("neuro.act.watch.body"), ("",)
    ),
    "retreat": _b(
        msg("neuro.act.retreat.name"),
        "ethogram",
        5,
        "stand",
        msg("neuro.act.retreat.body"),
        ("",),
        locomotes=True,
    ),
    "flee": _b(
        msg("neuro.act.flee.name"),
        "ethogram",
        5,
        "stand",
        msg("neuro.act.flee.body"),
        ("", "alarm"),
        locomotes=True,
    ),
    "doze": _b(
        msg("neuro.act.doze.name"), "rest", 0, "sit", msg("neuro.act.doze.body"), ("",), sleep=True
    ),
    "rest_sit": _b(
        msg("neuro.act.rest_sit.name"),
        "rest",
        0,
        "sit",
        msg("neuro.act.rest_sit.body"),
        ("", "coo"),
    ),
    "return_dock": _b(
        msg("neuro.act.return_dock.name"),
        "locomotion",
        5,
        "stand",
        msg("neuro.act.return_dock.body"),
        ("", "coo"),
        locomotes=True,
    ),
    "zoomies": _b(
        msg("neuro.act.zoomies.name"),
        "locomotion",
        10,
        "stand",
        msg("neuro.act.zoomies.body"),
        ("wheee", "chirp"),
        locomotes=True,
    ),
    # Reflex states (they interrupt; the reflex layer starts and ends them).
    "startle": _b(
        msg("neuro.act.startle.name"),
        "reflex",
        0,
        "any",
        msg("neuro.act.startle.body"),
        ("", "alarm"),
    ),
    "withdraw": _b(
        msg("neuro.act.withdraw.name"), "reflex", 0, "any", msg("neuro.act.withdraw.body"), ("",)
    ),
    "righting": _b(
        msg("neuro.act.righting.name"),
        "reflex",
        0,
        "any",
        msg("neuro.act.righting.body"),
        ("alarm",),
    ),
    "held": _b(
        msg("neuro.act.held.name"), "reflex", 0, "any", msg("neuro.act.held.body"), ("", "inquire")
    ),
}
FAMILIES = ("verb", "signature", "pause", "moment", "ethogram", "locomotion", "rest", "reflex")
# Short head acts that play INSIDE the current behaviour (a point event, not a bout): the brain
# sets them off (MotorCommand.moment), the motor overlays their program on the current one.
MOMENTS = frozenset(
    {
        "glance",
        "tilt",
        "perk_tilt",
        "double_take",
        "peek_down",
        "still",
        "head_shake",
        "glance_away",
        "proud_puff",
    }
)
# … and the ones that may play while it walks (the others wait for a stop)
MOMENTS_WALKING = frozenset({"glance", "peek_down", "double_take", "head_shake", "still"})


def behaviour_zh(behaviour: str) -> str:
    spec = BEHAVIOURS.get(behaviour)
    return spec.zh if spec is not None else behaviour


# ── the brain's step API ──────────────────────────────────────────────────────


class Brain(Protocol):
    """``NeuroBrain(genome, seed)`` (brain.py) implements this. Deterministic: the same genome,
    seed and frame stream give the same outputs. ``t_ms`` never goes backwards."""

    def step(self, t_ms: int, frame: SensorFrame) -> tuple[MotorCommand, BrainTelemetry]: ...


def wrap_angle(angle: float) -> float:
    """An angle in (−π, π]."""
    return math.atan2(math.sin(angle), math.cos(angle))
