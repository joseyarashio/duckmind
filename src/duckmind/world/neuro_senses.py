"""neuro:senses — what the neural duck's sensors could honestly report, as a
:class:`~duckmind.neuro.contract.SensorFrame` each brain tick.

The brain (:mod:`duckmind.neuro`) receives nothing else. This module sits on the world side
(it reads the simulator's truth for the sensor models below; the brain never imports it) and is
pure: :class:`SenseHub` is a step function over one tick's doors, deterministic for a seed and an
input stream, with no clock, socket or thread of its own (the simweb driver
:mod:`duckmind.simweb.neuro_live` owns those and calls :func:`live_senses`).

Where each channel comes from (``frame.sources``; the page marks every modelled reading):

========= ======== ==========================================================================
channel   source   what
========= ======== ==========================================================================
body      real     robotd ``robot.state`` (50 Hz): posture from the policy, ``safety.fallen``,
                   tilt/roll/pitch from robotd's projected gravity, the gyro, the four measured
                   head joints, ``move.applied``, contact odometry, leg load (the sim's
                   |force| × 100 proxy, owner B3), the gait phase from the hip joints
proprio   real     duck-body's edges (``blocked``, ``lifted``, ``fell``…) as the body link ran
                   them on the same state (:mod:`duckmind.neuro.bodylink`)
tof       real     tofd's frame through duck-body's DS-v1 depth summary (none on most ducks)
camera    model    the head camera: its light level (lux on the floor under the lens through
                   owner C3's bands, 2 Hz, two readings to switch) and a salience detector
                   (onset, motion, looming, things on the floor, the dock's tag) over the
                   world's view geometry (:mod:`duckmind.world.neuro_world`: family K, the
                   lens, ray-cast occlusion, the light on each thing) — only what is inside the
                   frame, not hidden, and lit well enough; person and hand labels are MODEL
                   (no validated detector, spec §4.7/§8.8; a person is called per look until
                   called, still or moving, :data:`P_PERSON`). The pass runs at
                   :data:`VISION_HZ` (a knob: the board's look budget today is ≤ 2 looks/s,
                   bursts of 4/s), blind during a saccade, and while the legs step only
                   :data:`P_SHARP_STEPPING` of the frames are sharp enough (pet:probe); and
                   the floor line (:class:`FloorLineModel`: duck-eye FL-v1's five free-floor
                   sectors — how far the floor is clear toward the walls, furniture and things
                   in the picture — from the world's rays, bright or dim light only). One
                   result is NOT on the look clock: the loom watch (:class:`LoomWatch`), a
                   cheap per-frame expansion detector that reports only looming
sound     model    the mono microphone and duck-ear's classes over the world's sound events:
                   one level, onsets, a class after the classifier's delay (masked by every
                   other sound playing — the TV hides a name call; the name KWS only when loud
                   enough over that), never a direction (spec §3.1: a mono mic)
touch     model    where the world's hand (or a person, a thing) really meets the shell, felt
                   as the robot would (joint deflection 40–80 ms, pet:body; pet-detect) with
                   misses, and adapting: felt at the onset and while it presses or slides, a
                   steady press fades, a toy resting on a foot is soon not felt
power     model    the world battery (robotd still reads 7.4 V, owner E1); ``on_dock`` is the
                   PG pin (a real sensor op)
clock     real     the duck's local time of day
========= ======== ==========================================================================

What the brain can never get from here: a world position, a world id, a sound's direction, a
thing it cannot see (behind it, behind a wall, in the dark, outside the frame), a touch the hand
did not really make. Truth enters only through the ``neuro`` section of ``x_world`` (the world's
own evaluator view) and the models turn it into sensor readings with the sensors' limits,
latencies and noise.

Floats in the prototype. Phase 5 must convert the models that stay on the robot (the light
meter's bands, the onset detector, the salience tracker) to integers (mm, mrad, ppm, centi-dB).
"""

from __future__ import annotations

import math
import random
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from duckmind.i18n import msg
from duckmind.neuro import contract as C

VERSION = "neuro-senses-1"

# ── camera ─────────────────────────────────────────────────────────────────────

# Owner C3 (2026-09-27): the light gate's ground truth, lux on the floor under the duck.
LUX_BANDS = ((30.0, "bright"), (2.5, "dim"), (0.5, "dark"))
METER_PERIOD_MS = 500  # mediad's AE meters twice a second
METER_NOISE_DEX = 0.06  # the meter's reading scatter (log10 lux): ~97 % agreement off the edges
DETECT_LATENCY_MS = (70, 140)  # capture → a salience result (a low-resolution difference pass)
TRACK_DROP_MS = 800  # a track not seen for this long is gone (seen again: a new track)
MIN_SIZE_RAD = {"bright": 0.017, "dim": 0.035, "dark": 0.10}  # smallest blob the pass finds
P_SEE = {"bright": 0.97, "dim": 0.85, "dark": 0.35, "black": 0.0}  # per pass, a moving blob
P_STATIC = 0.25  # a still thing found per pass while the camera is still (≈ 2.5 looks/s)
# MODEL: a person standing still is a large blob (legs and body), far surer to find than a toy:
# found per pass while the camera is still at the rate a sharp walking frame finds a near ranged
# object (P_WALK_STATIC_RANGED), times the light band's P_SEE. The floor line is not needed (the
# feet may be below the frame edge, then there is no range); the person call itself is P_PERSON,
# rolled per look until it is made.
P_STATIC_PERSON = 0.65
P_KEEP = 0.98  # an existing track followed per pass
P_PERSON = {"bright": 0.8, "dim": 0.5}  # MODEL: legs/feet called a person (per look, until called)
MOTION_RAD_S = 0.035  # own image motion that counts as moving (≈ 2°/s)
MOTION_STEPPING_RAD_S = 0.21  # …while the legs step and the head bobs (≈ 12°/s)
LOOM_PER_S = 0.35  # expansion rate (closing speed ÷ distance) that counts as coming nearer
LOOMING_PER_S = 0.6  # …and that makes it "looming" (with LOOMING_SIZE_RAD)
LOOMING_SIZE_RAD = 0.05
HAND_NEAR_M = 0.6  # MODEL: a hand this close is called a hand
# A floor-associated static thing this close can be acquired in one of the sharp frames while
# walking. This is deliberately not a motion-threshold relaxation: it needs a measured range,
# bright/dim light, and no saccade.
WALK_STATIC_RANGE_M = 2.0
# A visible, floor-ranged object inside WALK_STATIC_RANGE_M is a stronger cue than an
# un-ranged image blob. The rate is deliberately below a certain detection, but high enough
# to be useful when the look governor supplies only one sharp eligible view of a close object.
P_WALK_STATIC_RANGED = 0.65
P_SHARP_STEPPING = 0.35  # pet:probe: quarter-resolution frames sharp 36–60 % walking (bam)
VISION_HZ = 10.0  # the most passes a second the samples allow (the look budget decides below)
# The LOOK BUDGET (review 2026-09-28: the model saw 5–6× what the robot's own look governor
# allows). Every salience pass and floor line is a camera look and goes through duck-eye's own
# governor (duckmind.eye.governor.admit: ≤ 2 looks/s on average, bursts of 4/s for ≤ 5 s,
# tier 0 — the prototype does not model the SoC's heat): a look about every LOOK_GAP_S when
# nothing happens, faster (LOOK_BURST_GAP_S) for LOOK_BURST_S after a sound onset or a touch.
LOOK_GAP_S = 0.5
LOOK_BURST_GAP_S = 0.25
LOOK_BURST_S = 2.0
LOOK_KIND = "scan"
# The LOOM WATCH (machine verification v3, finding ②, 2026-10-02): a hand coming at the face
# lasts ~0.45 s, the look clock above gives a result every ~0.55 s, so a fast approach fell
# between two looks (non-bold startle 1/8–2/8 on the physics world). The expansion of the
# nearest large blob is therefore measured on EVERY frame the world samples (10 Hz, the camera's
# own frame rate in the model), outside the look governor; labels, the floor line, tracks and
# the dock tag stay on the look budget. This REOPENS the 2026-09-28 look-budget decision for
# this one measure: the model assumes the board can compute it per frame (a frame difference
# and the growth of the biggest changed blob on a low-resolution luma frame, no network, no
# NPU job). That assumption is not measured: check the board's compute before hardware.
LOOM_WATCH_HOLD_MS = 200  # a report is the brain's for this long (two frames), then it is stale
# EGO-MOTION DISCOUNT (the loom watch and the looks' looming label; next to the reach fix, 2026-10-03):
# the duck walks up to a thing to within the beak's reach (~0.05 m lens to thing), and every still
# thing it closes on expands in the picture at v·cos(a)/d — a "loom" its own walk makes. Both
# detectors read an expansion that the duck's own forward speed alone can explain for a STILL
# thing as no loom (the watch reports nothing; a look still finds and tracks the thing, as moving,
# but gives it no ``looming`` label or value); one only something coming at the duck can give is
# read as measured. Only what the duck knows is used: its own applied forward speed
# (BodySense.vx, kept LOOM_EGO_HOLD_MS after it drops: the body coasts a few cm after a stop and
# the camera still closes), and the blob's measured bearing, elevation and angular size — never
# the world's distance, closing speed or kind. For a still thing of width W at angle a off the
# heading, d = W / (2 tan(s/2)), so its own-walk expansion is v·cos(a)·2·tan(s/2) / W: at most that
# for any W >= LOOM_STILL_THING_M (see walk_explains). Standing (v = 0): nothing is discounted, a
# hand coming at a still duck is as before.
LOOM_EGO_HOLD_MS = 500
LOOM_EGO_MIN_M_S = 0.02  # slower than this the walk explains nothing
LOOM_STILL_THING_M = 0.02  # the narrowest still thing (a patch of it) the duck walks up to
SACCADE_GYRO = 1.2  # rad/s of trunk rotation: the image smears, the pass is blind
SACCADE_HEAD = 1.5  # rad/s of a head joint
BEARING_SIGMA = {"bright": 0.02, "dim": 0.035, "dark": 0.06, "black": 0.08}  # rad (+ FK, IMU)
SIZE_SIGMA = 0.08
RANGE_SIGMA = 0.08  # ground-plane range from the foot's image row (monocular)
TAG_RANGE_SIGMA = 0.03
TAG_READ_M = {"bright": 4.4, "dim": 2.2}  # a 10 cm tag's reading distance (full / half res)
TAG_FACE_MAX = math.radians(60.0)
# (dock:build 2026-09-28) The dock's face tag read as a tag — duck-eye NB-v1's pose, as phase 3
# measured it on the renders (main tree docs/validation/evidence/phase3/eye_nesttag.json,
# errors_when_read, ~90 looks per distance): range |error| median 3.5 / 10 / 25 / 50 / 69 mm at
# 1 / 1.5 / 2 / 2.5 / 3 m (σ ≈ 5 mm × d³), bearing 0.1–0.3 mrad (+ the head's joint readings and
# the IMU here: σ 3 mrad), the facing 6–78 mrad (σ 10 mrad + 30 mrad per metre beyond 1 m) and
# "not trusted" (the mirror pose fits as well) 0 / 14 / 2 / 48 / 38 % (0 below 1.2 m, rising to
# 45 % at 2.5 m). Range and bearing are from the trunk origin in the level frame, as the reader's.
TAG_BEARING_SIGMA = 0.003


def tag_range_sigma(d: float) -> float:
    return 0.005 * d * d * d


def tag_facing_sigma(d: float) -> float:
    return 0.010 + 0.030 * max(0.0, d - 1.0)


def tag_unsure(d: float) -> float:
    return max(0.0, min(0.45, 0.45 * (d - 1.2) / 1.3))


def tag_read_p(face: float) -> float:
    """A readable tag in a sharp still frame is read at NB-v1's measured rate (phase 3: 180/180
    bright and 90/90 dim within 30° of the tag's normal, 1–3 m; 92/120 bright at 45°): 0.97 up
    to 35° off the normal, 0.77 at 45°, ~0.5 at 60°. The tag reader runs on the look itself —
    not the salience pass's slow pick-up of a still new thing."""
    return 0.97 * max(0.0, min(1.0, 1.0 - max(0.0, face - 0.61) / 0.52 * 0.6))


FALSE_PER_S = {"bright": 0.01, "dim": 0.04, "dark": 0.15, "black": 0.05}  # noise blobs
CAMERA_STALE_MS = 1500  # no view for this long: the camera is not usable
# The floor line (MODEL of duck-eye FL-v1, 2026-09-28 integration): five free-floor sectors
# from the world's view geometry, only in bright or dim light (FL-v1: dark/black give none; dim
# multiplies the distances by 0.8), at most FLOOR_HZ looks a second (the look budget), blind in
# a saccade and only sharp enough in P_SHARP_STEPPING of the frames while stepping; FL-v1's
# measured scatter (median |error| 19 mm, p90 174 mm) and false stops (16 % of clear columns;
# a sector is the nearest of its columns, so a false stop here shortens a sector a little and
# rarely: FLOOR_FALSE_STOP) as noise.
FLOOR_HZ = 4.0  # (the look budget paces it: a floor line comes with a look)
FLOOR_SIGMA_M = (0.015, 0.06)  # scatter: a + b × distance (m)
FLOOR_FALSE_STOP = 0.04  # per sector per look: a spurious end at 40–90 % of the true one
FLOOR_DIM = 0.8  # FL-v1 in dim light

# ── microphone ─────────────────────────────────────────────────────────────────

ROOM_DB = 30.0  # a quiet flat, A-weighted
LATE_S = 1.0  # a world event that reaches the models this late is still perceived (late)
LEVEL_TAU_MS = 125.0  # the level's "fast" time constant
BACKGROUND_TAU_MS = 3000.0
ONSET_RISE_DB = 9.0  # over the background, and…
ONSET_JUMP_DB = 6.0  # …this much above the level a moment ago
ONSET_LATENCY_MS = (20, 40)
ONSET_REFRACTORY_MS = 150
CLASS_MIN_SNR_DB = 3.0  # quieter than this over the noise (and the other sounds): unclassified
STEADY_KIND_S = 4.0  # a steady sound (TV, music) is named this long after it is recognised
CLASS_HOLD_MS = 400
CLASS_LATENCY_MS = {"impulse": (80, 160), "speech": (250, 450), "steady": (600, 1200)}
# duck-ear's confusions (MODEL): what each world sound is heard as, with probabilities.
HEARD_AS: dict[str, tuple[tuple[str, float], ...]] = {
    "clap": (("clap", 0.80), ("bang", 0.12), ("noise", 0.08)),
    "bang": (("bang", 0.80), ("clap", 0.10), ("door", 0.10)),
    "door": (("door", 0.75), ("bang", 0.15), ("noise", 0.10)),
    "knock": (("door", 0.60), ("bang", 0.25), ("noise", 0.15)),
    "voice": (("voice", 0.87), ("tv", 0.06), ("name", 0.02), ("noise", 0.05)),
    "hello": (("hello", 0.80), ("voice", 0.20)),
    "footsteps": (("footsteps", 0.80), ("noise", 0.20)),
    "music": (("music", 0.80), ("tv", 0.20)),
    "tv": (("tv", 0.80), ("voice", 0.15), ("music", 0.05)),
    "noise": (("noise", 1.0),),
}
NAME_KWS_MAX = 0.92  # the name keyword spotter at its best
NAME_KWS_SNR_DB = 8.0  # …half of that at this SNR

# ── touch ──────────────────────────────────────────────────────────────────────

TOUCH_LATENCY_MS = (40, 80)  # the head and beak: the head joints' deflection (pet:body)
# The rest of the body has no skin (review 2026-09-28: the model had reported back, side and tail
# touches placed and fast): a firm push or poke is felt through the trunk IMU's jolt, placed
# left or right, as fast as the head; a gentle touch or stroke only through pet-detect on the
# microphone (1-s windows, 4 a second: ~0.25–1 s) and unplaced ("body").
BODY_SITES = frozenset({"back", "left", "right", "tail"})
PET_DETECT_MS = (250, 1000)
FIRM_CLOSING_M_S = 0.3  # an approach this fast, or …
FIRM_DEPTH_MM = 3.0  # … a press this deep, jolts the trunk (the IMU feels it)
TOUCH_HOLD_MS = 80
# Adaptation (MODEL, like skin and joint-load detectors): a contact is felt at its onset, while it
# presses or slides, and a steady press fades; a toy resting against a foot is soon not felt.
TOUCH_ONSET_TAU_S = 0.4
TOUCH_ADAPT_TAU_S = 2.0
TOUCH_FELT_MIN = 0.15
TOUCH_SHARP_M_S = 1.0  # an approach this fast adds a full unit of pressure (a poke, a thrown ball)
TOUCH_IMPACT_TAU_S = 0.3  # …and that part fades: a held press reads as its depth
P_FEEL_FIRM = 0.92
P_FEEL_LIGHT = 0.75  # a light touch (under 3 mm, slow) is missed a quarter of the time
SITE_MIXUP = 0.08
SITE_NEIGHBOURS = {
    "head": ("beak",),
    "beak": ("head",),
    "back": ("left", "right", "tail"),
    "left": ("back",),
    "right": ("back",),
    "tail": ("back",),
}

# ── body ───────────────────────────────────────────────────────────────────────

QUIET_LEG_LOAD = 35.0  # duck-body's floor band for a duck standing still (sim leg-force proxy)
HIP_PITCH = (2, 12)  # wire indices of the left and right hip pitch
STRIDE_HZ = 1.25  # a typical stride (two steps) until the gait's own is measured
BATTERY_NOISE = 0.004


def light_band(lux: float) -> str:
    """Owner C3's level for a lux reading."""
    for floor, level in LUX_BANDS:
        if lux >= floor:
            return level
    return "black"


def brightness_of(lux: float) -> float:
    """The camera as a light sensor, 0..1 on a log scale (0.03 lux → 0, 300 lux → ~1)."""
    return max(0.0, min(1.0, (math.log10(max(lux, 1e-3)) + 1.5) / 4.0))


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def _rng(*key: Any) -> random.Random:
    """A generator for one decision, seeded by what it is about (order independent)."""
    return random.Random(":".join(str(k) for k in key))


def _pick(rng: random.Random, table: Sequence[tuple[str, float]]) -> str:
    x = rng.random()
    acc = 0.0
    for kind, p in table:
        acc += p
        if x < acc:
            return kind
    return table[-1][0]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _db_sum(levels: Sequence[float]) -> float:
    power = sum(10.0 ** (v / 10.0) for v in levels if v > -100.0)
    return 10.0 * math.log10(power) if power > 0 else -100.0


# ── the light meter ──────────────────────────────────────────────────────────────


@dataclass
class LightMeter:
    """The camera's own light level (spec §4.5): a reading every :data:`METER_PERIOD_MS` of the
    lux on the floor under the lens with :data:`METER_NOISE_DEX` scatter, the band switched after
    two readings agree (hysteresis). Owner C1: black with the dock's backlight in view is dark."""

    seed: int = 7
    level: str = "bright"
    lux: float = 300.0
    candidate: str | None = None
    next_ms: int | None = None
    readings: int = 0

    def read(self, t_ms: int, lux: float, *, glow_in_view: bool = False) -> None:
        if self.next_ms is not None and t_ms < self.next_ms:
            return
        self.next_ms = t_ms + METER_PERIOD_MS
        rng = _rng("meter", self.seed, self.readings)
        self.readings += 1
        measured = 10.0 ** (math.log10(max(lux, 1e-3)) + rng.gauss(0.0, METER_NOISE_DEX))
        self.lux = measured
        band = light_band(measured)
        if band == "black" and glow_in_view:
            band = "dark"
        if self.readings == 1:
            self.level, self.candidate = band, None
            return
        if band == self.level:
            self.candidate = None
        elif band == self.candidate:
            self.level, self.candidate = band, None
        else:
            self.candidate = band


def walk_explains(
    ego_v: float, bearing: float, elevation: float, size: float, looming: float
) -> bool:
    """Whether the duck's own forward speed ``ego_v`` (m/s) can give a still thing, seen at this
    bearing and elevation (rad from its heading) with this angular size (rad), an expansion of
    ``looming`` (1/s) — for any still thing no narrower than :data:`LOOM_STILL_THING_M`. For a
    still thing of width W at distance d, s = 2·atan(W/2d) and the walk gives ego_v·cos(a)/d, so
    the walk gives at most ego_v·cos(a)·2·tan(s/2) / W for W at least that."""
    if ego_v < LOOM_EGO_MIN_M_S:
        return False
    ahead = math.cos(bearing) * math.cos(elevation)
    if ahead <= 0.0:
        return False  # off to the side or behind: walking forward does not close on it
    most = ego_v * ahead * 2.0 * math.tan(min(size, 3.0) / 2.0) / LOOM_STILL_THING_M
    return looming <= most


# ── the salience detector ─────────────────────────────────────────────────────────


@dataclass
class _Track:
    tid: int
    key: str  # the world's candidate id: never leaves this module
    kind: str
    first_ms: int
    seen_ms: int
    person: bool = False  # MODEL person call: once made it is kept for the track
    # C1 (range-first, root 批 2026-10-01)：同 track 的 range 用指數平均，不用單幀值。
    # 新 track 前 3 次權重遞增（0.3/0.4/0.5），之後固定 0.5——第一次不急著信，越看越準。
    rng_est: float | None = None  # smoothed range (m), None until first ranged sighting
    rng_n: int = 0  # ranged sightings so far


@dataclass
class VisionModel:
    """Salience from the world's view geometry (see the module docstring). ``step`` takes one
    view sample and returns the sightings a detection pass would give, and when."""

    seed: int = 7
    tracks: dict[str, _Track] = field(default_factory=dict)
    next_tid: int = 1
    passes: int = 0

    def step(
        self,
        t_ms: int,
        sample: Mapping[str, Any],
        *,
        stepping: bool,
        saccade: bool,
        ego_v: float = 0.0,
    ) -> list[C.Sighting]:
        self.passes += 1
        out: list[C.Sighting] = []
        seen_now: set[str] = set()
        for entry in sample.get("seen") or ():
            sighting = self._one(t_ms, entry, stepping=stepping, saccade=saccade, ego_v=ego_v)
            if sighting is not None:
                out.append(sighting)
                seen_now.add(str(entry.get("id")))
        for key in [k for k, tr in self.tracks.items() if t_ms - tr.seen_ms > TRACK_DROP_MS]:
            del self.tracks[key]
        # tracks not re-seen this pass but still young stay silent (no sighting) until dropped
        out += self._noise(t_ms, sample)
        return out

    def _one(
        self,
        t_ms: int,
        e: Mapping[str, Any],
        *,
        stepping: bool,
        saccade: bool,
        ego_v: float = 0.0,
    ) -> C.Sighting | None:
        key = str(e.get("id"))
        kind = str(e.get("kind"))
        vis = int(e.get("vis", 0))
        if vis <= 0:
            return None  # inside the frame but hidden behind something
        lux = float(e.get("lux", 0.0))
        band = light_band(lux)
        glow = bool(e.get("glow", False))
        size = float(e.get("size", 0.0))
        dist = max(float(e.get("dist", 1.0)), 0.05)
        om = float(e.get("om", 0.0))
        loom = max(0.0, float(e.get("vr", 0.0)) / dist)
        rng = _rng("see", self.seed, key, t_ms)
        track = self.tracks.get(key)
        if band == "black" and not (kind == "nest" and glow):
            return None
        if band != "black" and size < MIN_SIZE_RAD.get(band, 1.0):
            return None
        threshold = MOTION_STEPPING_RAD_S if stepping else MOTION_RAD_S
        moving = not saccade and (om >= threshold or loom >= LOOM_PER_S)
        still_camera = not stepping and not saccade
        p_see = P_SEE[band] if band != "black" else 0.9
        if stepping:
            p_see *= P_SHARP_STEPPING  # most frames smear while the legs step (pet:probe)
        tag_ok = (
            kind == "nest"
            and band in TAG_READ_M
            and dist <= TAG_READ_M[band]
            and float(e.get("face", 0.0)) <= TAG_FACE_MAX
            and e.get("tx") is not None
        )
        if tag_ok:
            # (dock:build) the dock's tag: the tag reader's own rate on a sharp frame
            p = tag_read_p(float(e.get("face", 0.0)))
            if stepping:
                p *= P_SHARP_STEPPING
            if saccade:
                p *= 0.5
        elif track is not None:
            p = P_KEEP if band in ("bright", "dim") else p_see
            if stepping:
                p *= P_SHARP_STEPPING  # a smeared frame keeps no track either (review 2026-09-28)
            if saccade:
                p *= 0.5
        elif moving:
            p = p_see
        elif still_camera:
            p = (P_STATIC_PERSON if kind == "person" else P_STATIC) * p_see
        elif (
            stepping
            and not saccade
            and kind in ("thing", "person")
            and band in ("bright", "dim")
            and (e.get("rng") is not None or kind == "person")
            and dist <= WALK_STATIC_RANGE_M
        ):
            # Walking still has its own image-motion gate. This separate near/floor-associated
            # branch lets a sharp frame notice a stationary object at the duck's feet (a person
            # standing still is legs and feet on the floor, found with no range too when they are
            # so near the feet fall below the frame) without treating far, dark, un-ranged, or
            # saccaded blobs as stable things.
            p = P_WALK_STATIC_RANGED
        else:
            p = 0.0
        if rng.random() >= p:
            return None
        # the duck's own walk is not a thing coming at it: the expansion it explains is not read as
        # looming (the thing is still found and tracked, as moving); same rule as the loom watch
        loom_read = loom
        if ego_v >= LOOM_EGO_MIN_M_S and loom >= LOOMING_PER_S:
            gate = _rng("walkgate", self.seed, key, t_ms)
            sig = BEARING_SIGMA.get(band, 0.08)
            if walk_explains(
                ego_v,
                C.wrap_angle(float(e.get("b", 0.0)) + gate.gauss(0.0, sig)),
                float(e.get("e", 0.0)) + gate.gauss(0.0, sig),
                max(0.005, size * (1.0 + gate.gauss(0.0, SIZE_SIGMA))),
                max(0.0, loom + gate.gauss(0.0, 0.05)),
            ):
                loom_read = 0.0
        label = self._label(kind, e, band, moving=moving, loom=loom_read, size=size, dist=dist)
        if label is None:
            return None
        called = kind == "person" and bool(track is not None and track.person)
        if kind == "person" and label in ("motion", "thing") and not called:
            # MODEL: the legs and feet are called a person per look until they are (a call made is
            # kept for the track), so a person who stays in view is found out within a few looks
            # instead of, 20-50 % of the time, never
            called = _rng("person", self.seed, key, t_ms).random() < P_PERSON.get(band, 0.0)
            if not called and label == "thing":
                return None  # still legs not called a person: nothing is reported, no track
        if called and label in ("motion", "thing"):
            label = "person"
        if track is None:
            track = _Track(self.next_tid, key, label, t_ms, t_ms)
            self.next_tid += 1
            self.tracks[key] = track
        track.seen_ms = t_ms
        track.person = called
        track.kind = label
        sigma = BEARING_SIGMA[band]
        rng_range: float | None = None
        tag_bearing: float | None = None
        facing: float | None = None
        if label == "nest" and band in TAG_READ_M:
            if e.get("tx") is not None and e.get("tn") is not None:
                # read as a tag (dock:build): the reader's range, bearing and facing, from the
                # trunk origin in the level frame, with its measured errors
                tx, ty = float(e["tx"]), float(e.get("ty", 0.0))
                d = math.hypot(tx, ty)
                rng_range = d + rng.gauss(0.0, tag_range_sigma(d))
                tag_bearing = C.wrap_angle(math.atan2(ty, tx) + rng.gauss(0.0, TAG_BEARING_SIGMA))
                if rng.random() >= tag_unsure(d):
                    facing = C.wrap_angle(
                        math.pi - float(e["tn"]) + rng.gauss(0.0, tag_facing_sigma(d))
                    )
            else:
                rng_range = dist * (1.0 + rng.gauss(0.0, TAG_RANGE_SIGMA))
        elif e.get("rng") is not None and label in ("thing", "person", "motion"):
            _raw = float(e["rng"]) * (1.0 + rng.gauss(0.0, RANGE_SIGMA))
            # C1 (range-first, root 批 2026-10-01)：同 track 指數平均，第一次不急著信。
            # track 剛建時在下面；這裡 track 一定存在（_one 上方已建）。
            _w = (0.3, 0.4, 0.5)[min(track.rng_n, 2)] if track is not None else 0.5
            if track is not None and track.rng_est is not None:
                rng_range = (1.0 - _w) * track.rng_est + _w * _raw
            else:
                rng_range = _raw
            if track is not None:
                track.rng_est = rng_range
                track.rng_n += 1
        bearing = C.wrap_angle(float(e.get("b", 0.0)) + rng.gauss(0.0, sigma))
        if tag_bearing is not None:
            bearing = tag_bearing
        return C.Sighting(
            kind=label,
            bearing=bearing,
            elevation=float(e.get("e", 0.0)) + rng.gauss(0.0, sigma),
            range_m=None if rng_range is None else round(max(0.05, rng_range), 3),
            size=max(0.005, size * (1.0 + rng.gauss(0.0, SIZE_SIGMA))),
            motion=_clip(om / 1.0) if moving else 0.0,
            looming=max(0.0, loom_read + rng.gauss(0.0, 0.05)) if loom_read > 0.05 else 0.0,
            track=track.tid,
            source="model",
            facing=None if facing is None else round(facing, 4),
        )

    def _label(
        self,
        kind: str,
        e: Mapping[str, Any],
        band: str,
        *,
        moving: bool,
        loom: float,
        size: float,
        dist: float,
    ) -> str | None:
        if kind == "nest":
            if band in TAG_READ_M:
                readable = dist <= TAG_READ_M[band] and float(e.get("face", 0.0)) <= TAG_FACE_MAX
                return "nest" if readable or bool(e.get("glow")) else None
            return "nest" if bool(e.get("glow")) else None
        if loom >= LOOMING_PER_S and size >= LOOMING_SIZE_RAD:
            return "looming"
        if kind == "hand":
            if dist <= HAND_NEAR_M and band in ("bright", "dim"):
                return "hand"
            return "motion" if moving else None
        if kind == "person":
            # MODEL: a person is found by their legs and body in the picture, not by a floor line,
            # so standing still, near enough that the feet fall below the frame, they are found
            # too (no range then); one already followed stays in the picture, as ever
            if moving or str(e.get("id")) in self.tracks:
                return "motion"
            return "thing" if band in ("bright", "dim") else None
        if kind == "thing":
            if moving:
                return "motion"
            if e.get("rng") is not None and band in ("bright", "dim"):
                return "thing"  # duck-eye's floor line: no things in the dark (spec §15.2)
            return None
        return "motion" if moving else None

    def _noise(self, t_ms: int, sample: Mapping[str, Any]) -> list[C.Sighting]:
        cam = sample.get("cam") or {}
        band = light_band(float(cam.get("lux", 300.0)))
        rng = _rng("noise", self.seed, t_ms)
        if rng.random() >= FALSE_PER_S.get(band, 0.0) * 0.1:
            return []
        axis = float(sample.get("cam_b", 0.0))
        return [
            C.Sighting(
                kind="motion",
                bearing=C.wrap_angle(axis + rng.uniform(-0.25, 0.45)),
                elevation=rng.uniform(-0.4, 0.2),
                range_m=None,
                size=rng.uniform(0.02, 0.05),
                motion=rng.uniform(0.1, 0.3),
                looming=0.0,
                track=0,
                source="model",
            )
        ]


@dataclass
class LoomWatch:
    """The cheap per-frame looming detector (see :data:`LOOM_WATCH_HOLD_MS` for why it exists).

    It runs on every world sample (a frame), not on the look clock, and reports one thing: the
    expansion rate of the nearest large blob (``looming`` = d ln(size)/dt = 1 / time to contact)
    as a ``looming`` sighting without range, track, label or image motion (``motion`` 0: a
    watch report never poses as a new moving thing to the reflex arc's motion/orient branches,
    only its looming branch and the brain's looming percept read it) — the same quantity, thresholds
    (:data:`LOOMING_PER_S`, :data:`LOOMING_SIZE_RAD`), light bands, noise and detection rates
    (:data:`P_SEE`; :data:`P_SHARP_STEPPING` while the legs step; blind in a saccade) as a look's
    ``looming`` label, so the reflex layer and the thresholds are unchanged. Per frame it takes
    the blob with the highest expansion rate among those that pass, once.

    Ego-motion discount (see :data:`LOOM_EGO_HOLD_MS`): a blob whose measured expansion is no
    more than the duck's own forward speed can give a still thing of that angular size and
    bearing (``v·cos(a)·2·tan(s/2) / LOOM_STILL_THING_M``) is the duck walking up to something,
    not something coming at the duck, and is withheld. It uses the blob's measured bearing,
    elevation, size and expansion and the duck's own applied speed, nothing else of the world.

    Cost assumption (NOT measured, to be checked on the board before hardware): the absolute
    difference of two consecutive low-resolution luma frames (about 160×90), a threshold, the
    bounding box of the biggest changed blob and its growth against the previous frame — tens of
    thousands of integer operations a frame, no model and no ISP/NPU job, so it needs no look from
    the governor; the discount adds a few multiplications per blob."""

    seed: int = 7
    frames: int = 0  # frames looked at
    seen: int = 0  # frames with something large and expanding in the picture
    explained: int = 0  # blobs withheld because the duck's own walk explains their expansion
    reports: int = 0  # looming sightings given

    walk_explains = staticmethod(walk_explains)

    def step(
        self,
        t_ms: int,
        sample: Mapping[str, Any],
        *,
        stepping: bool,
        saccade: bool,
        ego_v: float = 0.0,
        tracks: Mapping[str, Any] | None = None,
    ) -> C.Sighting | None:
        self.frames += 1
        best: tuple[Any, ...] | None = None
        seen = False
        for e in sample.get("seen") or ():
            if str(e.get("kind")) == "nest" or int(e.get("vis", 0)) <= 0:
                continue
            band = light_band(float(e.get("lux", 0.0)))
            size = float(e.get("size", 0.0))
            if band == "black" or size < max(LOOMING_SIZE_RAD, MIN_SIZE_RAD.get(band, 1.0)):
                continue
            loom = max(0.0, float(e.get("vr", 0.0)) / max(float(e.get("dist", 1.0)), 0.05))
            if loom < LOOMING_PER_S:
                continue
            seen = True
            # what the detector measures of this blob (each with the sensor's noise)
            rng = _rng("loomwatch", self.seed, e.get("id"), t_ms)
            sigma = BEARING_SIGMA[band]
            bearing = C.wrap_angle(float(e.get("b", 0.0)) + rng.gauss(0.0, sigma))
            elevation = float(e.get("e", 0.0)) + rng.gauss(0.0, sigma)
            seen_size = max(0.005, size * (1.0 + rng.gauss(0.0, SIZE_SIGMA)))
            looming = max(0.0, loom + rng.gauss(0.0, 0.05))
            if self.walk_explains(ego_v, bearing, elevation, seen_size, looming):
                self.explained += 1
                continue
            if best is None or looming > best[0]:
                best = (looming, bearing, elevation, seen_size, band, rng, str(e.get("id")))
        if seen:
            self.seen += 1
        if best is None or saccade:
            return None  # nothing left, or the image smears: no expansion can be measured
        looming, bearing, elevation, seen_size, band, rng, key = best
        p = P_SEE[band] * (P_SHARP_STEPPING if stepping else 1.0)
        if rng.random() >= p:
            return None
        self.reports += 1
        # (telemetry only) the eye's track of the same object, if the looks have one: the report itself
        # stays without identity (``track`` 0), so it never poses as a new thing
        known = None if tracks is None else tracks.get(key)
        source_track = 0 if known is None else int(known.tid)
        return C.Sighting(
            kind="looming",
            bearing=bearing,
            elevation=elevation,
            range_m=None,
            size=seen_size,
            motion=0.0,  # expansion only: image motion is the difference pass's, on the looks
            looming=looming,
            track=0,
            source="model",
            source_track=source_track,
        )


# ── the microphone ─────────────────────────────────────────────────────────────────


@dataclass
class _Heard:
    """One world sound as the microphone and duck-ear take it."""

    id: int
    kind: str
    t0: float  # physics seconds
    t1: float
    db: float  # at the head
    env: str
    name: bool
    by: str = ""  # the world's label of what made it (a kind only; "" = not told)
    decided: bool = False
    heard_as: str | None = None
    class_at: float = 0.0  # physics seconds when duck-ear says what it was
    snr: float = 0.0
    shift: float = 0.0  # it reached this model late (the driver's world copy): heard from arrival


@dataclass
class HearingModel:
    """The mono microphone (see the module docstring)."""

    seed: int = 7
    events: dict[int, _Heard] = field(default_factory=dict)
    power_fast: float = 10.0 ** (ROOM_DB / 10.0)
    background: float = ROOM_DB
    last_inst: float = ROOM_DB
    last_s: float | None = None
    onset_pending: list[tuple[float, float]] = field(default_factory=list)  # (due s, strength)
    last_onset_s: float = -1e9

    def update(self, sounds: Sequence[Mapping[str, Any]], now_s: float | None = None) -> None:
        """New and updated world sounds. One that started less than :data:`LATE_S` before it
        reached this model (the driver copies the world a few times a second) is heard from its
        arrival, whole: late, never lost; an older one (this model just started) as it is."""
        for item in sounds:
            try:
                sid = int(item["id"])
            except (KeyError, TypeError, ValueError):
                continue
            ev = self.events.get(sid)
            if ev is None:
                t0 = float(item.get("t0", 0.0))
                shift = 0.0
                if now_s is not None and now_s - LATE_S < t0 < now_s - 0.005:
                    shift = now_s - t0
                ev = _Heard(
                    sid,
                    str(item.get("kind", "noise")),
                    t0 + shift,
                    float(item.get("t1", 0.0)) + shift,
                    float(item.get("db", 0.0)),
                    str(item.get("env", "impulse")),
                    bool(item.get("name", False)),
                    by=str(item.get("by", "") or "").split(":", 1)[0],  # (the kind only: no ids)
                    shift=shift,
                )
                self.events[sid] = ev
            else:
                ev.t1 = float(item.get("t1", ev.t1)) + ev.shift
                ev.db = float(item.get("db", ev.db))

    def _level(self, ev: _Heard, s: float) -> float:
        if s < ev.t0:
            return -100.0
        if ev.env == "impulse":
            if s <= ev.t1:
                return ev.db
            return ev.db - 160.0 * (s - ev.t1)  # a room's decay, ~0.2 s to vanish
        if s > ev.t1:
            return ev.db - 120.0 * (s - ev.t1)
        if ev.env == "speech":
            phase = (s - ev.t0) * 4.0 + (ev.id % 7) * 0.13
            gap = 0.5 + 0.5 * math.sin(2.0 * math.pi * phase)
            return ev.db - 12.0 * (1.0 - gap) ** 2
        return ev.db + 1.5 * math.sin(2.0 * math.pi * 0.3 * (s - ev.t0) + ev.id)

    def step(self, s: float, self_db: float) -> C.SoundSense:
        """The microphone at physics second ``s`` (``self_db``: the duck's own motor noise)."""
        dt = 0.02 if self.last_s is None else max(0.001, min(0.2, s - self.last_s))
        # the loudest moment in the last dt (a clap is shorter than a tick)
        probes = [s - dt * k / 4.0 for k in range(4, -1, -1)]
        inst = ROOM_DB
        for p in probes:
            levels = [ROOM_DB, self_db] + [self._level(ev, p) for ev in self.events.values()]
            inst = max(inst, _db_sum(levels)) if p == probes[0] else max(inst, _db_sum(levels))
        power = 10.0 ** (inst / 10.0)
        a = 1.0 - math.exp(-dt * 1000.0 / LEVEL_TAU_MS)
        self.power_fast += a * (power - self.power_fast)
        level = 10.0 * math.log10(self.power_fast)
        rise = inst - self.background
        jump = inst - self.last_inst
        if (
            rise >= ONSET_RISE_DB
            and jump >= ONSET_JUMP_DB
            and s - self.last_onset_s > ONSET_REFRACTORY_MS / 1000.0
        ):
            self.last_onset_s = s
            lag = _rng("onset", self.seed, round(s, 3)).uniform(*ONSET_LATENCY_MS) / 1000.0
            self.onset_pending.append((s + lag, _clip((rise - 6.0) / 24.0)))
        if inst < self.background + 6.0 or inst < self.background:
            b = 1.0 - math.exp(-dt * 1000.0 / BACKGROUND_TAU_MS)
            bg_power = 10.0 ** (self.background / 10.0)
            bg_power += b * (power - bg_power)
            self.background = 10.0 * math.log10(max(bg_power, 1e-3))
        self.last_inst = inst
        self.last_s = s
        onset = 0.0
        due = [o for o in self.onset_pending if o[0] <= s]
        if due:
            onset = max(o[1] for o in due)
            self.onset_pending = [o for o in self.onset_pending if o[0] > s]
        quiet = _db_sum([ROOM_DB, self_db])
        for ev in self.events.values():
            if not ev.decided and s >= ev.t0:
                # masked by everything else sounding while it does (the TV masks a name call)
                others = [self._level(o, ev.t0 + 0.02) for o in self.events.values() if o is not ev]
                self._decide(ev, _db_sum([quiet, *others]))
        # the newest classified sound is the one duck-ear reports; a steady one only a while
        best: _Heard | None = None
        for ev in self.events.values():
            if ev.heard_as is None or s < ev.class_at:
                continue
            if ev.env == "steady":
                until = min(ev.t1, ev.class_at + STEADY_KIND_S)
            else:
                until = max(ev.t1, ev.class_at)
            if s > until + CLASS_HOLD_MS / 1000.0:
                continue
            if best is None or ev.class_at > best.class_at:
                best = ev
        kind = None if best is None else best.heard_as
        for sid in [k for k, ev in self.events.items() if ev.t1 < s - 5.0 and ev.decided]:
            del self.events[sid]
        noise = _rng("level", self.seed, round(s, 2)).gauss(0.0, 0.5)
        return C.SoundSense(
            level_db=round(level + noise, 1),
            onset=round(onset, 3),
            kind=kind,
            bearing=None,
            bearing_source="none",
            source="model",
            by="" if best is None else best.by,
        )

    def _decide(self, ev: _Heard, noise_floor: float) -> None:
        ev.decided = True
        ev.snr = ev.db - noise_floor
        if ev.snr < CLASS_MIN_SNR_DB:
            return  # heard as a level, never classified
        rng = _rng("class", self.seed, ev.id)
        lo, hi = CLASS_LATENCY_MS.get(ev.env, (200, 400))
        wait = rng.uniform(lo, hi) / 1000.0
        if ev.env == "speech":
            ev.class_at = ev.t1 + wait  # a word is known when it has been said
        else:
            ev.class_at = ev.t0 + wait
        if ev.kind == "name" or ev.name:
            p = NAME_KWS_MAX * _sigmoid((ev.snr - NAME_KWS_SNR_DB) / 3.0)
            ev.heard_as = "name" if rng.random() < p else "voice"
            return
        ev.heard_as = _pick(rng, HEARD_AS.get(ev.kind, (("noise", 1.0),)))


# ── touch ──────────────────────────────────────────────────────────────────────────


@dataclass
class _Felt:
    key: str
    site: str
    t0: float
    t1: float
    active: bool
    felt: bool
    at: float  # physics second it is felt
    depth_mm: float = 0.0
    closing: float = 0.0
    slide: float = 0.0
    by: str = ""  # who touched it: the kind of the world's contact record (hand, person, thing)
    world_site: str = ""  # the site the world last gave (a hand moving on: the site follows)
    impact: float = 0.0  # how fast it came in (m/s, the first report of the episode)
    onset: float = 0.0  # the first moment's strength
    dynamic: float = 0.0  # pressing or sliding now
    moved_s: float = 0.0  # when it last pressed or slid (the steady part fades from here)


@dataclass
class TouchModel:
    """Contact episodes felt as the robot would (see the module docstring)."""

    seed: int = 7
    episodes: dict[str, _Felt] = field(default_factory=dict)

    def update(self, touches: Sequence[Mapping[str, Any]], now_s: float | None = None) -> None:
        for item in touches:
            key = f"{item.get('key')}@{item.get('t0')}"
            ep = self.episodes.get(key)
            depth = float(item.get("now_depth_mm", item.get("depth_mm", 0.0)))
            closing = float(item.get("closing", 0.0))
            if ep is None:
                rng = _rng("touch", self.seed, key)
                sliding = float(item.get("slide", 0.0)) >= 0.05  # a stroke is plain to feel
                firm = float(item.get("depth_mm", 0.0)) >= 3.0 or closing >= 0.1 or sliding
                felt = rng.random() < (P_FEEL_FIRM if firm else P_FEEL_LIGHT)
                site = str(item.get("site", "back"))
                if rng.random() < SITE_MIXUP and site in SITE_NEIGHBOURS:
                    site = rng.choice(SITE_NEIGHBOURS[site])
                t0 = float(item.get("t0", 0.0))
                jolt = (
                    closing >= FIRM_CLOSING_M_S or float(item.get("depth_mm", 0.0)) >= FIRM_DEPTH_MM
                )
                if site in BODY_SITES and not jolt:
                    # a gentle touch on the body: only pet-detect hears it — late, unplaced
                    site = "body"
                    at = t0 + rng.uniform(*PET_DETECT_MS) / 1000.0
                else:
                    if site in ("back", "tail"):
                        site = "body"  # the IMU's jolt says it was pushed, not where on the back
                    at = t0 + rng.uniform(*TOUCH_LATENCY_MS) / 1000.0
                if now_s is not None and now_s - LATE_S < t0 and at < now_s:
                    at = now_s  # reached this model late: felt from its arrival
                ep = _Felt(
                    key,
                    site,
                    t0,
                    float(item.get("t1", t0)),
                    bool(item.get("active", False)),
                    felt,
                    at,
                )
                ep.by = str(item.get("by", "")).split(":", 1)[0]  # (the kind only: no ids)
                ep.moved_s = t0
                ep.impact = closing
                self.episodes[key] = ep
            now_closing = float(item.get("now_closing", 0.0))
            now_slide = float(item.get("now_slide", 0.0))
            ep.onset = max(  # an impact is felt by how fast it came and how hard it presses
                ep.onset,
                _clip(float(item.get("closing", 0.0)) / 0.15 + depth / 6.0),
            )
            ep.dynamic = _clip(now_closing / 0.25 + now_slide / 0.08) if item.get("active") else 0.0
            if ep.dynamic >= 0.2:
                ep.moved_s = float(item.get("t1", ep.t1))
            site_now = str(item.get("site", ep.site))
            if ep.world_site and site_now != ep.world_site:
                # the contact moved on (a stroke from the head to the back): placed on the head
                # and beak, unplaced on the body
                ep.site = site_now if site_now in ("head", "beak") else "body"
            ep.world_site = site_now
            ep.t1 = float(item.get("t1", ep.t1))
            ep.active = bool(item.get("active", False))
            ep.depth_mm = depth if ep.active else 0.0
            ep.closing = closing
            ep.slide = float(item.get("now_slide", item.get("slide", 0.0))) if ep.active else 0.0

    def step(self, s: float) -> C.TouchSense:
        best: _Felt | None = None
        best_p = -1.0
        for ep in self.episodes.values():
            if not ep.felt or s < ep.at or s > ep.t1 + TOUCH_HOLD_MS / 1000.0 + (ep.at - ep.t0):
                continue
            p = self._felt(ep, s - (ep.at - ep.t0))
            if p >= TOUCH_FELT_MIN and p > best_p:
                best, best_p = ep, p
        for key in [k for k, ep in self.episodes.items() if ep.t1 < s - 3.0]:
            del self.episodes[key]
        if best is None:
            return C.TouchSense(site=None, pressure=0.0, stroke=0.0, source="model")
        # how hard it presses and how fast it came (a poke, a push, a ball: sharp; a slow
        # pat: gentle), fading as the contact adapts
        impact = best.impact * math.exp(-max(0.0, s - best.at) / TOUCH_IMPACT_TAU_S)
        press = _clip(best.depth_mm / 8.0 + impact / TOUCH_SHARP_M_S)
        return C.TouchSense(
            site=best.site,
            pressure=round(press * min(1.0, best_p / 0.4), 3),
            stroke=round(_clip(best.slide / 0.08), 3),
            source="model",
            by=best.by,
        )

    @staticmethod
    def _felt(ep: _Felt, s: float) -> float:
        """How strongly it is felt at physics second ``s`` (the contact's own time, before the
        sensing delay): the onset, the pressing or sliding, and the steady press fading."""
        onset = ep.onset * math.exp(-max(0.0, s - ep.t0) / TOUCH_ONSET_TAU_S)
        steady = _clip(ep.depth_mm / 8.0) * math.exp(-max(0.0, s - ep.moved_s) / TOUCH_ADAPT_TAU_S)
        return _clip(max(onset, ep.dynamic, steady))


# ── the body ───────────────────────────────────────────────────────────────────────

PANIC_LABELS = frozenset(
    {"panic_run", "panic_arc_left", "panic_arc_right", "quick_turn_left", "quick_turn_right"}
)  # skilltrain.guard.PANIC_ENTRIES + QUICK_TURN_ENTRIES (both run upright and moving)


def _posture(state: Any, view: Any) -> str:
    """robotd's policy (and duck-body's proprio view) as the contract's posture."""
    if bool(getattr(state, "fallen", False)) or bool(getattr(view, "fallen", False)):
        return "unknown"
    policy = str(getattr(state, "policy", ""))
    applied = getattr(state, "applied", (0.0, 0.0, 0.0))
    base = policy.split(":", 1)[0]
    if base in PANIC_LABELS:
        return "walk"  # the panic network runs the flight (a robotd skill): the duck is walking
    if base == "skill_abort":
        return "stand"  # its one tick of v5 under a zero command, ending a skill
    if policy == "walk" and (abs(applied[0]) > 0.05 or abs(applied[2]) > 0.2):
        return "walk"
    if policy in ("walk", "stand"):
        return "stand"
    if policy == "sit":
        return "sit"
    if policy in ("rise", "held", "sitting", "homing"):
        return "transition"
    return "unknown"


@dataclass
class GaitPhase:
    """The stride's phase 0..1 from the hip pitches (left minus right), while stepping."""

    hz: float = STRIDE_HZ
    mean: float = 0.0
    last_sign: int = 0
    last_cross_s: float | None = None

    def step(self, s: float, joints: Sequence[float], vel: Sequence[float] | None) -> float | None:
        x = float(joints[HIP_PITCH[0]]) - float(joints[HIP_PITCH[1]])
        self.mean += 0.02 * (x - self.mean)
        x -= self.mean
        sign = 1 if x >= 0 else -1
        if sign > 0 and self.last_sign < 0:
            if self.last_cross_s is not None:
                period = s - self.last_cross_s
                if 0.3 < period < 3.0:
                    self.hz += 0.3 * (1.0 / period - self.hz)
            self.last_cross_s = s
        self.last_sign = sign
        w = 2.0 * math.pi * self.hz
        v = 0.0
        if vel is not None:
            v = float(vel[HIP_PITCH[0]]) - float(vel[HIP_PITCH[1]])
        phase = math.atan2(-v / w, x) / (2.0 * math.pi)
        return phase % 1.0


def body_sense(state: Any, view: Any = None, phase: float | None = None) -> C.BodySense:
    """REAL: ``robot.state`` as the contract's body (see :func:`_posture`)."""
    down = getattr(state, "down", None) or (0.0, 0.0, -1.0)
    tilt = getattr(state, "tilt_rad", None) or 0.0
    roll = math.atan2(down[1], -down[2]) if down[2] < 0 else 0.0
    pitch = math.atan2(down[0], -down[2]) if down[2] < 0 else 0.0
    load = getattr(state, "leg_load", None)
    applied = getattr(state, "applied", (0.0, 0.0, 0.0))
    gyro = getattr(state, "gyro", None) or (0.0, 0.0, 0.0)
    odom = getattr(state, "odom_pos", (0.0, 0.0, 0.0))
    return C.BodySense(
        posture=_posture(state, view),
        fallen=bool(getattr(state, "fallen", False)),
        limp=bool(getattr(state, "limp", False)),
        tilt=float(tilt),
        roll=float(roll),
        pitch=float(pitch),
        gyro=(float(gyro[0]), float(gyro[1]), float(gyro[2])),
        head=tuple(float(v) for v in getattr(state, "head", (0.349, 0.349, 0.0, 0.0))),  # type: ignore[arg-type]
        vx=float(applied[0]),
        vyaw=float(applied[2]),
        odom=(float(odom[0]), float(odom[1]), float(getattr(state, "odom_yaw", 0.0))),
        leg_load=None if load is None else float(load) / QUIET_LEG_LOAD,
        step_phase=phase,
    )


def self_noise_db(state: Any) -> float:
    """The duck's own servos at its microphone (MODEL): ~30 dB still, ~40 dB walking."""
    vel = getattr(state, "velocities", None)
    if not vel:
        return 28.0
    total = sum(abs(float(v)) for v in vel)
    return 28.0 + 10.0 * math.log10(1.0 + total / 1.5)


# ── ToF ────────────────────────────────────────────────────────────────────────────


def tof_sense(depth: Any, now_ns: int | None = None) -> C.TofSense:
    """REAL: duck-body's DS-v1 view (``DepthView``) as the contract's ToF."""
    if depth is None:
        return C.TofSense(present=False)
    age = 0
    t_ns = getattr(depth, "t_ns", None)
    if now_ns is not None and t_ns:
        age = max(0, int((now_ns - int(t_ns)) // 1_000_000))
    return C.TofSense(
        present=True,
        left_mm=getattr(depth, "left_mm", None),
        ahead_mm=getattr(depth, "ahead_mm", None),
        right_mm=getattr(depth, "right_mm", None),
        drop_zones=int(getattr(depth, "drop_zones", 0) or 0),
        touch=bool(getattr(depth, "touch", False)),
        age_ms=age,
        ahead_top_mm=int(getattr(depth, "ahead_top_mm", 0) or 0),
        virtual=bool(getattr(depth, "virtual", False)),
        low_edge_mm=int(getattr(depth, "low_edge_mm", 0) or 0),
    )


# ── the hub ────────────────────────────────────────────────────────────────────────


class FloorLineModel:
    """MODEL of duck-eye FL-v1's floor line (:mod:`duckmind.eye.floorline`): the world's
    free-floor truth per sector (``neuro_world``: rays at the body's height inside the picture,
    capped by what the picture shows of the floor) → the five :class:`~duckmind.neuro.contract.
    FreeSector` a look would report — only in bright or dim light (dim ×0.8, FL-v1), at most
    :data:`FLOOR_HZ` looks a second, none in a saccade, and while stepping only when the frame
    is sharp enough (:data:`P_SHARP_STEPPING`); FL-v1's scatter and occasional false stops.
    Deterministic for a seed and the sample times."""

    def __init__(self, seed: int = 7) -> None:
        self.seed = int(seed)
        self.last_t: float | None = None
        self.looks = 0

    def look(
        self,
        t: float,
        truth: Any,
        light: str,
        *,
        stepping: bool,
        saccade: bool,
    ) -> tuple[C.FreeSector, ...] | None:
        """One sample's floor line, or None when this sample gives none."""
        if not isinstance(truth, Mapping) or light not in ("bright", "dim") or saccade:
            return None
        if self.last_t is not None and t < self.last_t + 1.0 / FLOOR_HZ - 1e-6:
            return None
        rng = _rng("floor", self.seed, round(t, 3))
        if stepping and rng.random() >= P_SHARP_STEPPING:
            return None  # a smeared frame: no floor line this time
        sectors = truth.get("sectors") or ()
        if len(sectors) == 0:
            return None
        self.last_t = t
        self.looks += 1
        scale = FLOOR_DIM if light == "dim" else 1.0
        out = []
        for item in sectors:
            free = float(item.get("free", C.FLOOR_RANGE_M))
            end = str(item.get("end") or "unknown")
            if end not in C.FLOOR_ENDS:
                end = "unknown"
            sigma = FLOOR_SIGMA_M[0] + FLOOR_SIGMA_M[1] * free
            free = free + rng.gauss(0.0, sigma)
            if rng.random() < FLOOR_FALSE_STOP:
                free, end = free * rng.uniform(0.4, 0.9), "unknown"  # a seam, a shadow, glare
            free = min(C.FLOOR_RANGE_M, max(0.05, free * scale))
            out.append(
                C.FreeSector(
                    bearing=round(float(item.get("b", 0.0)) + rng.gauss(0.0, 0.01), 4),
                    width=round(float(item.get("w", 0.13)), 4),
                    free_m=round(free, 3),
                    end=end,
                )
            )
        return tuple(out)


@dataclass
class _Pending:
    due_ms: int
    camera: C.CameraSense
    capture_ms: int = 0


@dataclass
class _PendingLoom:
    due_ms: int
    capture_ms: int
    sighting: C.Sighting


@dataclass
class SenseHub:
    """Every channel for one duck: ``hub(doors) -> SensorFrame`` (the driver's ``senses``).

    ``doors`` is the driver's :class:`~duckmind.simweb.neuro_live.Doors` (or anything with its
    fields): ``t_ms``, ``state`` (a :class:`~duckmind.duckbody.robotd_client.RobotState`),
    ``link`` (the body link's last output: its proprio view), ``events``, ``tof``, ``depth``,
    ``pg``, ``day_minute`` and ``world`` (``x_world``; only its ``neuro`` and ``battery``
    sections are read, by the models)."""

    seed: int = 7
    vision_hz: float = VISION_HZ  # salience passes per second (a sample is skipped between)
    meter: LightMeter = field(init=False)
    vision: VisionModel = field(init=False)
    hearing: HearingModel = field(init=False)
    touch: TouchModel = field(init=False)
    gait: GaitPhase = field(default_factory=GaitPhase)
    offset_ms: float | None = None  # brain ms − physics ms (the smallest seen, creeping up)
    last_t_ms: int | None = None
    last_sample_t: float = -1.0
    camera: C.CameraSense = field(
        default_factory=lambda: C.CameraSense(usable=False, source="model")
    )
    camera_ms: int | None = None
    pending: deque[_Pending] = field(default_factory=deque)
    battery: float | None = None
    battery_ms: int = -10_000
    frames: int = 0
    last_view: dict[str, Any] = field(default_factory=dict)
    cam_bearing: float = 0.0
    last_pass_t: float | None = None
    depth_tracker: Any = None
    depth_seen: int | None = None
    depth_view: Any = None
    floor_model: FloorLineModel = field(init=False)
    floor: tuple[C.FreeSector, ...] = ()
    floor_t_ms: int | None = None
    look_budget: Any = None  # duck-eye's governor bucket (duckmind.eye.governor.Budget)
    next_look_t: float = -1.0  # physics s: the next look is due
    burst_until: float = -1.0  # physics s: looking faster after an onset until then
    looks: int = 0
    looks_refused: int = 0
    look_times: deque[float] = field(default_factory=lambda: deque(maxlen=64))
    # the loom watch: the one camera result that is not on the look clock (LoomWatch)
    loom_watch_on: bool = True  # False: looms are found by the looks only (the old behaviour)
    loom_watch: LoomWatch = field(init=False)
    loom_t: float = -1.0  # physics s: the newest frame the watch has looked at
    loom_queue: list[_PendingLoom] = field(default_factory=list)
    loom_live: _PendingLoom | None = None
    camera_capture_ms: int | None = None  # when the look now held was captured
    own_speed: deque[tuple[int, float]] = field(default_factory=lambda: deque(maxlen=64))

    def __post_init__(self) -> None:
        self.meter = LightMeter(seed=self.seed)
        self.loom_watch = LoomWatch(seed=self.seed)
        self.vision = VisionModel(seed=self.seed)
        self.hearing = HearingModel(seed=self.seed)
        self.touch = TouchModel(seed=self.seed)
        self.floor_model = FloorLineModel(seed=self.seed)
        from duckmind.eye.governor import Budget as LookBudget

        self.look_budget = LookBudget()

    def __call__(self, doors: Any) -> C.SensorFrame:
        return self.step(doors)

    # the clocks

    def _sync(self, t_ms: int, sim_s: float | None) -> None:
        if self.last_t_ms is not None and self.offset_ms is not None:
            self.offset_ms += 0.01 * max(0, t_ms - self.last_t_ms)  # let a drift show
        if sim_s is not None:
            seen = t_ms - sim_s * 1000.0
            if self.offset_ms is None or seen < self.offset_ms:
                self.offset_ms = seen
        self.last_t_ms = t_ms

    def sim_s(self, t_ms: int) -> float:
        """Physics seconds at brain ms ``t_ms`` (the world's events are stamped in them)."""
        return (t_ms - (self.offset_ms or 0.0)) / 1000.0

    def brain_ms(self, sim_s: float) -> int:
        return int(round(sim_s * 1000.0 + (self.offset_ms or 0.0)))

    # the step

    def step(self, doors: Any) -> C.SensorFrame:
        t_ms = int(getattr(doors, "t_ms", 0))
        state = getattr(doors, "state", None)
        link = getattr(doors, "link", None)
        view = getattr(getattr(link, "prox", None), "view", None)
        world = getattr(doors, "world", None) or {}
        section = world.get("neuro") if isinstance(world, Mapping) else None
        sim_now = None
        if isinstance(section, Mapping):
            samples = section.get("samples") or ()
            if samples:
                sim_now = float(samples[-1].get("t", 0.0))
            sim_now = float(section.get("sim_time", sim_now or 0.0))
        self._sync(t_ms, sim_now)
        s = self.sim_s(t_ms)
        sources = dict(C.CHANNEL_SOURCE)
        # this prototype's camera is modelled end to end (light and salience alike)
        sources["camera"] = "model"
        # body (real)
        if state is not None:
            phase = None
            joints = getattr(state, "joints", None)
            if joints is not None and len(joints) >= 13:
                phase = self.gait.step(s, joints, getattr(state, "velocities", None))
            body = body_sense(state, view, phase)
            if body.posture != "walk":
                body = C.BodySense(**{**_fields(body), "step_phase": None})
        else:
            body = C.BodySense()
            sources["body"] = "none"
        stepping = body.posture == "walk"
        self.own_speed.append((t_ms, body.vx))
        head_speed = getattr(state, "head_speed", None) if state is not None else None
        gyro_norm = math.sqrt(sum(g * g for g in body.gyro))
        saccade = gyro_norm > SACCADE_GYRO or (head_speed or 0.0) > SACCADE_HEAD
        # the world's truth → the models
        if isinstance(section, Mapping):
            self._see(t_ms, section, stepping=stepping, saccade=saccade)
            self.hearing.update(section.get("sounds") or (), now_s=s)
            self.touch.update(section.get("touches") or (), now_s=s)
            self.cam_bearing = self._cam_axis(section)
        camera = self._camera(t_ms, gyro_norm + (head_speed or 0.0))
        sound = self.hearing.step(s, self_noise_db(state) if state is not None else 28.0)
        touch = self.touch.step(s)
        if sound.onset > 0.0 or touch.site is not None:
            self.burst_until = s + LOOK_BURST_S  # something happened: it looks faster a moment
        if not isinstance(section, Mapping):
            sources.update(camera="none", sound="none", touch="none")
            sound = C.SoundSense(source="none")
            touch = C.TouchSense(source="none")
        # ToF (real, when fitted)
        depth = getattr(doors, "depth", None)
        if depth is None:
            depth = self._depth(getattr(doors, "tof", None), state, stepping)
        tof = tof_sense(depth, getattr(state, "t_ns", None))
        if not tof.present:
            sources["tof"] = "none"
        # power (model; the PG pin is real)
        power = self._power(t_ms, world, getattr(doors, "pg", None))
        if self.battery is None:
            sources["power"] = "none"
        self.frames += 1
        frame = C.SensorFrame(
            t_ms=t_ms,
            body=body,
            events=tuple(getattr(doors, "events", ()) or ()),
            tof=tof,
            camera=camera,
            sound=sound,
            touch=touch,
            power=power,
            day_minute=float(getattr(doors, "day_minute", 600.0)),
            sources=sources,
        )
        return frame

    def _depth(self, frame: Any, state: Any, walking: bool) -> Any:
        """A raw ``tof.frame`` through duck-body's own DS-v1 summary (when the driver hands the
        frame, not the body link's view): the sensor's pose from ``robot.state.frames.tof``."""
        if not isinstance(frame, Mapping) or state is None:
            return None
        pose = getattr(state, "tof", None)
        gravity = getattr(state, "down", None)
        distances, statuses = frame.get("distance_mm"), frame.get("status")
        if pose is None or gravity is None or not distances or not statuses:
            return None
        from duckmind.duckbody import depth as ds

        t_ns = int(frame.get("t_ns") or getattr(state, "t_ns", 0) or 0)
        if self.depth_seen == t_ns and self.depth_view is not None:
            return self.depth_view
        if self.depth_tracker is None:
            self.depth_tracker = ds.DepthTracker(standing_height_m=0.12)
        reading, _ = self.depth_tracker.step(
            ds.ToFFrame.of(t_ns, distances, statuses),
            ds.pose_from_frames(pose[0], pose[1], gravity),
            walking=walking,
            odom=(
                float(getattr(state, "odom_pos", (0.0, 0.0, 0.0))[0]),
                float(getattr(state, "odom_pos", (0.0, 0.0, 0.0))[1]),
                float(getattr(state, "odom_yaw", 0.0)),
            ),
            upright=ds.upright(getattr(state, "odom_pos", None)),
        )
        self.depth_seen, self.depth_view = t_ns, reading.view()
        return self.depth_view

    def _cam_axis(self, section: Mapping[str, Any]) -> float:
        samples = section.get("samples") or ()
        if not samples:
            return self.cam_bearing
        last = samples[-1]
        fwd = (last.get("cam") or {}).get("fwd")
        duck = last.get("duck") or {}
        if not fwd:
            return self.cam_bearing
        return C.wrap_angle(math.atan2(float(fwd[1]), float(fwd[0])) - float(duck.get("yaw", 0.0)))

    def _see(self, t_ms: int, section: Mapping[str, Any], *, stepping: bool, saccade: bool) -> None:
        from duckmind.eye.governor import admit

        for sample in section.get("samples") or ():
            t = float(sample.get("t", 0.0))
            if self.loom_watch_on and t > self.loom_t + 1e-6:
                self.loom_t = t
                self._watch_loom(t_ms, t, sample, stepping=stepping, saccade=saccade)
            if t <= self.last_sample_t + 1e-6:
                continue
            if self.last_pass_t is not None and t < self.last_pass_t + 1.0 / self.vision_hz - 1e-6:
                continue  # no pass on this frame (the pass rate is a knob)
            if t < self.next_look_t - 1e-6:
                continue  # no look due yet
            decision = admit(LOOK_KIND, "", int(t * 1e9), self.look_budget, None)
            self.look_budget = decision.budget
            if not decision.ok:  # the governor says not yet
                self.looks_refused += 1
                self.next_look_t = t + max(0.05, decision.retry_in_ms / 1000.0)
                continue
            gap = LOOK_BURST_GAP_S if t < self.burst_until else LOOK_GAP_S
            jitter = _rng("look", self.seed, round(t, 3)).uniform(0.85, 1.15)
            self.next_look_t = t + gap * jitter
            self.looks += 1
            self.look_times.append(t)
            self.last_sample_t = self.last_pass_t = t
            capture_ms = self.brain_ms(t)
            cam = sample.get("cam") or {}
            glow = any(
                e.get("kind") == "nest" and e.get("glow") and int(e.get("vis", 0)) > 0
                for e in sample.get("seen") or ()
            )
            self.meter.read(capture_ms, float(cam.get("lux", 300.0)), glow_in_view=glow)
            sample = {**sample, "cam_b": self._cam_axis({"samples": [sample]})}
            sightings = self.vision.step(
                capture_ms, sample, stepping=stepping, saccade=saccade, ego_v=self._own_speed(t_ms)
            )
            lag = _rng("lag", self.seed, round(t, 3)).uniform(*DETECT_LATENCY_MS)
            lux = self.meter.lux
            motion = 0.0
            for sighting in sightings:
                motion += sighting.size * sighting.size * sighting.motion
            line = self.floor_model.look(
                t, sample.get("floor"), self.meter.level, stepping=stepping, saccade=saccade
            )
            if line is not None:
                self.floor, self.floor_t_ms = line, capture_ms
            camera = C.CameraSense(
                usable=True,
                light=self.meter.level,
                brightness=round(brightness_of(lux), 3),
                motion=_clip(motion / 0.02),
                sightings=tuple(sightings),
                age_ms=0,
                source="model",
                floor=self.floor,
                floor_t_ms=self.floor_t_ms,
            )
            self.pending.append(_Pending(max(t_ms, capture_ms + int(lag)), camera, capture_ms))
            self.last_view = {
                "t": t,
                "in_view": [
                    {"id": e.get("id"), "vis": e.get("vis"), "fov": e.get("fov"), "n": e.get("n")}
                    for e in sample.get("seen") or ()
                ],
                "sighted": [s_.kind for s_ in sightings],
                "light": self.meter.level,
            }

    def _watch_loom(
        self, t_ms: int, t: float, sample: Mapping[str, Any], *, stepping: bool, saccade: bool
    ) -> None:
        """One world frame through the loom watch: a report (if any) reaches the brain after the
        detector's own latency, like a look's result."""
        capture_ms = self.brain_ms(t)
        sighting = self.loom_watch.step(
            capture_ms,
            sample,
            stepping=stepping,
            saccade=saccade,
            ego_v=self._own_speed(t_ms),
            tracks=self.vision.tracks,
        )
        if sighting is None:
            return
        lag = _rng("loomlag", self.seed, round(t, 3)).uniform(*DETECT_LATENCY_MS)
        self.loom_queue.append(_PendingLoom(max(t_ms, capture_ms + int(lag)), capture_ms, sighting))

    def _own_speed(self, t_ms: int) -> float:
        """The duck's own forward speed for the loom watch's ego-motion discount: the largest
        forward speed it applied in the last :data:`LOOM_EGO_HOLD_MS` (the body coasts after a
        stop and the camera still closes on what it walked up to)."""
        return max(
            (v for t, v in self.own_speed if t_ms - t <= LOOM_EGO_HOLD_MS and v > 0.0), default=0.0
        )

    def _loom_report(self, t_ms: int, held: Sequence[C.Sighting]) -> C.Sighting | None:
        """The newest loom-watch report that has arrived, while it is fresh; None when a look
        captured at or after the same frame already says ``looming`` (its own result is the
        better measurement and replaces the watch's for that frame)."""
        live = self.loom_live
        if live is None or t_ms - live.due_ms > LOOM_WATCH_HOLD_MS:
            return None
        if (
            self.camera_capture_ms is not None
            and self.camera_capture_ms >= live.capture_ms
            and any(s.kind == "looming" for s in held)
        ):
            return None
        return live.sighting

    def _camera(self, t_ms: int, ego: float) -> C.CameraSense:
        while self.pending and self.pending[0].due_ms <= t_ms:
            item = self.pending.popleft()
            self.camera, self.camera_ms = item.camera, item.due_ms
            self.camera_capture_ms = item.capture_ms
        due = [p for p in self.loom_queue if p.due_ms <= t_ms]
        if due:  # loom-watch reports that have arrived: the newest frame's is the live one
            self.loom_queue = [p for p in self.loom_queue if p.due_ms > t_ms]
            self.loom_live = max(due, key=lambda p: p.capture_ms)
        if self.camera_ms is None:
            return self.camera
        age = t_ms - self.camera_ms
        camera = self.camera
        usable = age <= CAMERA_STALE_MS
        sightings = camera.sightings if age <= TRACK_DROP_MS else ()
        report = self._loom_report(t_ms, sightings) if usable else None
        if report is not None:
            sightings = (*sightings, report)
        return C.CameraSense(
            usable=usable,
            light=camera.light,
            brightness=camera.brightness,
            motion=_clip(max(camera.motion, ego / 3.0)),
            sightings=sightings,
            age_ms=int(age),
            source="model",
            floor=camera.floor,
            floor_t_ms=camera.floor_t_ms,
        )

    def _power(self, t_ms: int, world: Any, pg: bool | None) -> C.PowerSense:
        battery = world.get("battery") if isinstance(world, Mapping) else None
        charging = False
        if isinstance(battery, Mapping) and "percent" in battery:
            if t_ms - self.battery_ms >= 1000 or self.battery is None:
                self.battery_ms = t_ms
                soc = float(battery.get("soc_ppm", float(battery["percent"]) * 10_000)) / 1e6
                noise = _rng("battery", self.seed, t_ms // 1000).gauss(0.0, BATTERY_NOISE)
                self.battery = _clip(soc + noise)
            charging = bool(battery.get("charging"))
        on_dock = bool(pg) if pg is not None else False
        return C.PowerSense(
            battery=0.8 if self.battery is None else round(self.battery, 3),
            on_dock=on_dock,
            charging=on_dock and charging,
            hot=False,
            source="model" if self.battery is not None else "none",
        )

    def view(self) -> dict[str, Any]:
        """For the page and the evaluators: what the camera had in view and what it reported,
        the meter, the microphone's background, the clocks."""
        return {
            "version": VERSION,
            "frames": self.frames,
            "offset_ms": None if self.offset_ms is None else round(self.offset_ms, 1),
            "light": {"level": self.meter.level, "lux": round(self.meter.lux, 2)},
            "view": self.last_view,
            "background_db": round(self.hearing.background, 1),
            "tracks": len(self.vision.tracks),
            "looks": self.looks,
            "looks_refused": self.looks_refused,
            "looks_per_s": self.looks_per_s(),
        }

    def looks_per_s(self) -> float:
        """The camera's look rate over its last looks (the look budget: ≤ 2/s on average)."""
        times = self.look_times
        if len(times) < 2 or times[-1] - times[0] <= 0.0:
            return 0.0
        return round((len(times) - 1) / (times[-1] - times[0]), 2)

    def new_world(self) -> None:
        """The body was replaced (the layout editor's Apply, a fresh physics world): its physics clock
        starts again at 0 while the brain's clock goes on, and the models' state (touch episodes,
        sounds, tracks, the camera's pending looks) is in the old world's seconds. ``offset_ms`` only
        ever creeps up, so it would keep the old gap: every contact and sound of the new world would
        land in the past and be dropped, for as long as the old world had run. Start every model
        afresh, as at boot; what counts the duck's looks and frames goes on."""
        fresh = type(self)(seed=self.seed, vision_hz=self.vision_hz)
        keep = {"seed", "vision_hz", "loom_watch_on", "looks", "looks_refused", "frames"}
        for name, value in vars(fresh).items():
            if name not in keep:
                setattr(self, name, value)


def _fields(obj: Any) -> dict[str, Any]:
    return {name: getattr(obj, name) for name in obj.__slots__}


def live_senses(seed: int = 7, vision_hz: float = VISION_HZ) -> SenseHub:
    """The driver's senses lane (``neuro_live``'s ``senses(doors) -> SensorFrame``)."""
    return SenseHub(seed=int(seed), vision_hz=float(vision_hz))


# Page stimuli the world can make (``x_neuro_stim``): the contract's scenario keys plus the
# page's own (wave, roll_ball, novel_object, tv, knock). A page button goes to the WORLD only.
SUPPORTED_STIMULI = frozenset(
    {
        "clap",
        "call_name",
        "voice",
        "hello",
        "say",
        "bang",
        "door",
        "knock",
        "music",
        "tv",
        "wave",
        "hand_loom",
        "pet_head",
        "stroke_back",
        "poke",
        "push",
        "roll_ball",
        "toy_drop",
        "novel_object",
        "walk_by",
        "sit_near",
        "lights_off",
        "lights_on",
        "pick_up",
        "set_down",
    }
)


def make_injector(world_call: Any) -> Any:
    """``inject(name, params) -> answer`` for the page's stimulus buttons: the world op
    ``x_neuro_stim`` through the driver's ``world_call(op, **params)`` (never the brain)."""

    def inject(name: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if name not in SUPPORTED_STIMULI:
            raise ValueError(msg("neuro.world_refusal.dc0045", given_id=name))
        extra = {k: v for k, v in dict(params or {}).items() if k not in ("stim", "kind", "op")}
        return dict(world_call("x_neuro_stim", stim=name, **extra))

    return inject
