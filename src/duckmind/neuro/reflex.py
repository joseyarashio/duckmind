"""Reflex arcs straight from the senses (neuro:motor) — the ONE reflex module of the prototype.

The fast layer under the brain's choices: a stimulus the duck's own sensors report reaches a
motor response after a short neural latency, without deliberation — and the same stimulus never
gives quite the same response, because the response is scaled by the stimulus, by the arousal
neuromodulator, by habituation (the tenth clap gets a weaker turn of the head) and by
sensitisation (after a scare the duck is jumpier for a while), and drawn with its own seeded
variation.

=============  ==========================================  ===========================================
reflex         stimulus (SensorFrame channel)              response
=============  ==========================================  ===========================================
``orient``     a sound onset (``sound``), motion or a new  gaze toward the source (a saccade, speed ∝
               thing in view (``camera`` sightings),       strength) that stays on the source while
               the lights coming on                        the body turns; with no bearing (the real
                                                           mono mic) a *listening* pose: head up,
                                                           cocked; habit ``double_look``: it looks
                                                           once more as the first look fades
``startle``    a sudden loud sound, something looming at   the head snaps back and up, the walk stops;
               the face                                    then ``freeze`` (always when afraid)
``freeze``     after a startle; the lights going out       head raised and still, looking at it, for
                                                           as long as the brain's fear holds it
``withdraw``   an unexpected touch (a poke); a flickering  the head pulls away from the touched side,
               contact is one touch (1.2 s gap)            sometimes a quick step away; a touch it
                                                           knows well (habituated) gets a look only
``lean_in``    a stroke, or a touch it saw coming          the head leans into the hand, slowly
``veer``       ToF: something ahead within ~0.7 m; a       yaw bias toward the freer side, speed cap;
               brush (proprio ``brushed``); its own leg    a short swerve away (reafference: its own
               or beak meeting something while stepping    contact is never a withdrawal)
(turn away)    ToF ≤ 300 mm, a stall (``blocked``)         stop 0.3–1.1 s, then 2–3.5 s of arcs away
``drop_stop``  ToF: the floor ahead is missing (≥ 4 cells) stop, look down at the edge
``bump_stop``  ToF touch / proprio ``blocked``             stop, pull the head back ("oops": never a
                                                           scare, spec §8.9 E4)
``righting``   ``fell`` (robotd's fall verdict)            stop everything, head still at HOME until up
``lifted``     proprio ``lifted``                          stop, quiet, until ``set_down``
``stabilise``  trunk tilt/gyro jolt while walking (IMU)    brace: stop the twist, head toward HOME
=============  ==========================================  ===========================================

Latencies are the neural part (startle ~60 ms, withdrawal ~90, orienting ~140, leaning ~350) and
shorten with arousal; the body adds its own: robotd smooths ``robot.head`` at 1.78 Hz (EMA 0.2
per 50 Hz tick) and the walking network tracks the head command with a further ~0.1 s — measured
headless (neuro:motor head probe, n = 5 per posture × joint × servo model): head pitch, yaw and
roll reach 10 % of a step after 60–140 ms, 50 % after 140–260 ms, 90 % after 220–540 ms; the
neck is slower and weaker, above all standing (bam: 50 % after 0.3–0.7 s, gain 0.08–0.5), so the
reflex poses lean on head pitch and yaw. :mod:`duckmind.neuro.motor` pre-compensates robotd's
EMA to win some of that back.

**Habituation and sensitisation.** In the prototype they come from the BRAIN
(``habituation.Habituation``: per stimulus class, lazy exact relaxation, familiarity,
dishabituation): :func:`step` takes ``hab`` (the level per class, 1 = fresh) and ``sens`` (the
extra gain per class) and reports every onset it processed in ``ReflexOutput.presented``; the
brain then habituates (``present``) and sensitises (``sensitise``) each once. Without them (the
motor lane's own tests and headless runs) the module keeps a simple version of its own: each
event depresses its class by ``hab_rate × strength`` (floor ``hab_floor``), the gain recovers
toward 1 with ``hab_recover_s``, a scare dishabituates the others and raises a leaky
sensitisation (general and per sense) that multiplies every defensive response until it decays.
The brain's ``fear`` (0..1) sets how long a freeze holds; ``comfort`` lowers withdrawal.

Honest senses: this module reads only a :class:`~duckmind.neuro.contract.SensorFrame` (the real
doors plus clearly marked sensor models); a bearing is used only when the frame has one, and a
remembered bearing is carried along by the duck's own odometry yaw (never a world position).
Pure: :func:`step` takes and returns frozen state; the clock is the frame's ``t_ms``.

Phase 5 (fixed point): the timings are already integer milliseconds; every gain, level and pose is
a product of clamps and sums (SCALE 10_000 integers); the own habituation/sensitisation leaks
become per-tick-rate multipliers; ``S.ease``/``S.min_jerk`` are polynomials; the seeded draws are
splitmix64 integers (the lognormal hold needs a small exp table); ``S.wrap`` needs atan2
(``duckbody.fixedmath.atan2_mrad``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from duckmind.i18n import join, lst, msg
from duckmind.neuro import signals as S
from duckmind.neuro.contract import (
    REFLEXES,
    Gaze,
    HeadPose,
    ReflexEvent,
    SensorFrame,
    Sighting,
)
from duckmind.neuro.touch import is_stroking

REFLEX_VERSION = "neuro-reflex-2"

PRIORITY = {
    "righting": 100,
    "lifted": 90,
    "drop_stop": 80,
    "startle": 70,
    "withdraw": 60,
    "bump_stop": 55,
    "stabilise": 50,
    "freeze": 45,
    "orient": 30,
    "lean_in": 20,
}
HALTS = frozenset(
    {"righting", "lifted", "drop_stop", "startle", "withdraw", "bump_stop", "stabilise", "freeze"}
)
INTERRUPTS = frozenset({"righting", "lifted", "drop_stop", "startle", "withdraw", "bump_stop"})
# The reflexes that hold the body (the brain shows them as the behaviour); orient, lean_in,
# veer and stabilise only overlay what the behaviour does.
HOLDS = frozenset({"righting", "lifted", "drop_stop", "startle", "withdraw", "bump_stop", "freeze"})
# Onset classes that never habituate (a fall is a fall; the floor edge is always an edge).
NO_HABITUATION = frozenset({"body:fall", "tof:drop", "tof:touch", "body:tilt", "body:brushed"})
# How strongly each sound kind can startle (1) or only draw the head (small). The family's voice,
# a greeting and its own name are safety signals to a duckling (it follows its mother's call;
# review 2026-09-28: the "very clingy" shy duck startled at its own name and fled): they draw
# the head, they do not startle.
SOUND_STARTLE = {
    "bang": 1.0,
    "noise": 0.9,
    "clap": 0.85,
    "door": 0.6,
    "footsteps": 0.25,  # (the family's own footsteps are familiar)
    "duck_call": 0.12,
    "music": 0.25,
    "tv": 0.25,
    "voice": 0.06,
    "hello": 0.04,
    "name": 0.0,
}
# … and how strongly they draw the head (× the duck's orienting gain)
SOUND_ORIENT = {"name": 1.6, "hello": 1.3, "voice": 1.1, "duck_call": 1.2}
# A sound the ear has not named yet. The onset detector reports a sound 20–40 ms after it starts;
# duck-ear names it later: 80–160 ms for an impulse, about a second for a steady sound, and a
# spoken word (its own name included) only once it has been said. So at the onset the reflexes
# usually hold a sound with NO kind yet — and "no kind yet" is not "a noise" (a noise is a class
# duck-ear gives): it startles on its loudness alone only when it is loud enough that nothing
# needs to be recognised. UNNAMED_STARTLE_DB is the level the microphone reads on the onset's
# first ticks (dB; no weight below the first, the full weight above the second): a name called
# from 1.5–2 m reads 60–61 dB (from 1 m 65–66), a clap 71–74, a bang 80 and more (neuro_senses'
# HearingModel: its fast level is ~4.7 dB under the sound's own at the onset's first tick). A
# quieter one gets its look at once and its class decides whether it startles when the ear
# names it (within SOUND_NAME_WAIT_MS; a name never does).
UNNAMED_STARTLE_DB = (64.0, 76.0)
SOUND_NAME_WAIT_MS = 600
# A track the duck has seen this recently is not new when it is listed again: the eye gives a
# thing that leaves the picture for a moment (the head's own nod or turn, a pass that did not
# re-find it) the same track id when it is found again (neuro_senses.TRACK_DROP_MS, 0.8 s), and
# a thing that was already looked at is not looked at again as if it had just appeared
# (found 2026-10-04: 6 of 14 orienting looks in a 40 s dance were this).
SEEN_KEEP_MS = 2000
VISUAL_CLASS = {
    "person": "visual:person",
    "hand": "visual:hand",
    "thing": "visual:thing",
    "toy": "visual:thing",
}
EVENTS_KEPT = 12
CAM_ELEVATION_HOME = -0.15  # rad: where the camera looks at HOME (headless probe: −8…−15°)
FREEZE_FEAR_HOLD = 0.35  # while the brain's fear is above this, a freeze keeps holding …
FREEZE_MAX_MS = 12_000  # … but never longer than this from its start
UP_TILT = 0.44  # rad: robotd's fall verdict cleared and the trunk this upright = up again
# A startle floor (owner 2026-10-05: 「嚇一跳也是」): a sudden loud sound — its raw size at least
# STARTLE_FLOOR_RAW (a bang at the head's 92-96 dB reads 1.0; a clap at 80 dB 0.89) — that the duck
# is not yet used to (its habituation level still at least STARTLE_FLOOR_HAB: the sharp sounds of
# its home habituate the group a little, a fresh boot reads 0.77-0.79) startles EVERY
# temperament, whatever its arousal and this onset's own variation did to the response (the live
# bold explorer's single page bang read 0.94 x 0.83 x (0.75 + 0.5 x arousal) x 0.874 x jitter
# against its 0.683 bar and jumped about half the time). Only the sound kinds that can startle at
# all (SOUND_STARTLE at least STARTLE_FLOOR_KIND: bang, noise, clap, door; never the family's voice
# or its own name). A startle the floor fired is a strong scare too (``Presented.floor``: the brain
# makes it a flight for every temperament). Below the floor, and for a bang it has got used to,
# nothing changes.
STARTLE_FLOOR_RAW = 0.9
STARTLE_FLOOR_HAB = 0.6
STARTLE_FLOOR_KIND = 0.6
HAB_LOUD_KEY = "sound:loud"  # habituation.LOUD_KEY: the level of loud sudden sounds alone (the floor asks it, not only the class)
SOUND_REFRACTORY_MS = 180  # one sound class at most every 0.18 s: a 4/s call train is seen
TOUCH_GAP_MS = 1200  # a touch site must have been free this long for a new touch onset
REAFFERENT_MS = 1000  # a leg/beak contact this soon after stepping is its own doing (a brush)
# A gentle touch while it steps — or soon after its body felt a stall — with no hand in view and
# nothing sliding on it is a thing it walked into (a box, a book), not a hand patting it: a
# box leant into rewarded the duck for pushing it (owner 2026-10-04, 「頂著東西往前走」)
OWN_CONTACT_MS = 4000
OWN_CONTACT_PRESSURE = 0.45  # a firm touch (a poke, a push) stays a touch
HAND_CUE = 0.5  # a hand seen near or a voice with it: a touch now may be a hand's
HABITUATED_TOUCH = 0.3  # a touch class habituated below this gets a look, not a withdrawal
TOUCH_BEARING = {"head": 0.0, "beak": 0.0, "left": 1.0, "right": -1.0}  # where to look (rad)
PERSON_LOOM = 2.0  # 1/s: a person walking up startles only when coming this fast
# The same stimulus never gives quite the same size of response: each onset's response is drawn
# with this log-normal spread (pet:etho: every execution re-draws its fine structure, CV 0.1–0.3;
# review 2026-09-28: the sizes repeated to two decimals)
RESP_SIGMA = 0.2
TURN_AWAY_VYAW = 0.9  # rad/s: the turning-away bias; ≥ motor.ARC_BIAS makes the walk an arc
# (duck-body's body link lets only arcs with |vyaw| ≥ 0.5 through after a stall or a wall)
# Turning away ends on its consequences, not on a timer (review 2026-09-28): the stop lasts until
# the legs have stopped (the policy out of its walk, the applied twist ~0) and at least the
# legs' tail; the arc until it has turned as far as it needs (what it saw open, else ~1 rad) or
# it sees the way ahead clear — with a long safety cap.
TURN_STOP_MIN_MS = 400
TURN_NEED_RAD = (0.6, 2.6)
TURN_DEFAULT_RAD = 1.0
TURN_CLEAR_RAD = 0.35  # turned at least this much and the way is seen clear: done
TURN_ARC_MIN_MS = 600
TURN_ARC_MAX_MS = 7000
TURN_FLIP_RAD = 0.2  # an arc that has turned less than this may still change side
# fid:motor 2026-09-28 (the owner: 「嚇一跳…我看3D模擬並沒有呈現出來」 — only what the physical body
# visibly does counts). A startle while walking stops the legs and keeps them stopped long enough
# to READ as a stop: the stop itself takes ~0.6 s (A100 D.startle from 0.55: still in 0.62 s p50,
# 0.84 s p95, 5.4 cm) and a stop is seen as one only when it lasts ≥ 0.8 s (A100 D.bursts,
# pet:probe: pauses < 0.4 s are coasted through) — so ≥ 1.5 s from the onset, whatever follows.
STARTLE_STOP_MS = 1500
# The startle's beak snap: the mouth is the one raw actuator (robotd writes it every tick, no
# smoothing) — it gapes at once and shuts again (the rest of the startle is the head's snap).
BEAK_SNAP_MS = 140
BEAK_SHUT_MS = 90
# A withdrawal's step away: from standing the legs need ~0.5 s before the first footfall (pet:body:
# 2 cm after 0.52 s at 0.40), so a step that is SEEN lasts 1.1–1.5 s. A touch on the head or the
# beak comes from in front — the hand is where it would walk — so it backs off in a small arc
# instead (the owner, 2026-09-28: reversing a small step/arc when cornered is allowed; curiosity
# calibration, phase 3: vx −0.40, 1.0 rad/s, 1.0 s from standing ≈ 8–12 cm back, ~35°, 0/40
# falls, tilt ≤ 5.4°).
WITHDRAW_STEP_MS = (1100, 1500)
WITHDRAW_BACK_MS = 1000
WITHDRAW_BACK_VX = -0.40
YAW_PER_CLIP = 0.6  # rad of head yaw per clip unit (content.SEATED_ENVELOPE["head_yaw"])
YAW_CLIP_MAX = 1.0 / YAW_PER_CLIP  # the validated ±1.0 rad (motor.YAW_MAX_RAD), in clip units
PITCH_PER_CLIP = 0.35  # rad of head pitch per clip unit (content.SEATED_ENVELOPE["head_pitch"])
NECK_PER_CLIP = 0.25  # rad of neck pitch per clip unit
HOME_PITCH = 0.3491  # robotd's v5 HOME of the neck and the head pitch (duckbody V5_HOME_HEAD)
# ── fid:fix 2026-09-28: every reflex plans its motion from where the MEASURED head is now ─────
# The review watched the page as the owner does: a second clap while the duck was still afraid
# (its head already raised in the freeze) was logged 「嚇一跳，頭猛往後仰」 while the nose went up
# 0.4–1.2° (the look-up had no room left) and only the same ~20° sideways jerk to its own side
# showed, every time; 「頭轉向右邊」 came with no turn at all (the head was already there) or a
# turn the other way, 34 % of the time as named. So each reflex reads the head from its own
# proprioception (robot.state's joints: BodySense.head), plans a motion that WILL show from
# there, and names only those parts (contract.REFLEX_PARTS) — the page's witness then judges each
# named part on the body.
SEEN_RAD = math.radians(10.0)  # fid:audit's V = 1: a part is named only when it moves this much
TURN_SEEN_RAD = math.radians(12.0)  # … a head turn is planned this big to be seen ≥ 10°
# How far up/back the nose gets on this body (rad above HOME; standing and walking the head pitch
# alone — the walking network holds the neck —, seated the neck and the head pitch together: the
# page's own measure), measured: standing a startle from rest 17.3° p50 (fid:motor, n = 14,
# headless DR) and 14.7° live, a freeze HELD at 12.4–13.1° with the command at the envelope
# (review, n = 4: a second startle then made 0.4–1.2°); walking 14.2° p50 (n = 21); seated head
# 14.4° + neck 8.7° (n = 20). The room a snap up has is this minus where the nose is now.
NOSE_REACH = {"stand": 0.24, "walk": 0.22, "sit": 0.40}
# No room above (the head already raised): the startle jerks the head DOWN first — a flinch the
# body can show from there (the standing head pitch reaches ~74 % of a step in 0.2 s at robotd
# head_alpha 1.0: A100 t90 0.34 s) — and snaps it back up: 「頭猛一沉」.
DIP_RAD = 0.42  # the dip's command below the nose now (rad), × the onset's own spread
DIP_MS = (170, 230)  # the dip, its snap included …
SNAP_UP_MS = 90  # … then back up to the startle's raised pose in this long
NECK_BACK_SEEN = (
    0.05  # seated, the neck is named (「脖子一縮」) only when it starts this far forward
)
# (rad): from HOME the seated neck makes 8.7° p50 of its whole 14° command — too near the line
OWN_SIDE_P = 0.65  # a sound with no direction: the head goes to its own side this often (65/35)
JERK_SPREAD = 0.3  # a startle's sideways jerk: this log-normal spread on its size (±30 %)
# Orienting: a look that turns the head less than TURN_SEEN_RAD is a re-fixation — the head is
# already on it: no row, no motion. A thing straight ahead (within AHEAD_RAD) with the head already
# looking ahead is viewed the bird's way, with one eye: the head turned ONE_EYE_RAD to one side.
AHEAD_RAD = math.radians(15.0)
ONE_EYE_RAD = (0.30, 0.40)
LISTEN_YAW_RAD = (0.30, 0.48)  # a sound with no direction: the head cocked this far aside
ORIENT_MERGE_MS = 3000  # a new look at the same place within this long is the same look
ORIENT_MERGE_RAD = math.radians(12.0)
# While the head is being HELD still — a freeze, a startle's hold, the acts freeze / startle /
# still — an orienting onset does not move the head and gets no row (the review: 「愣住 40 秒」
# with 8 visible moves inside; rows under a freeze that the head never made): it is still heard
# and seen (habituation, arousal): a covert look.
HOLDING_STILL = frozenset({"freeze", "startle", "still"})
# Acts whose job the head must keep doing against a look: the sound skills' — the dance's nod on
# the beat, the instrument's hold on the hand's beam (``skill_sound.install`` adds them). While
# one is the brain's act, a thing or an unnamed movement newly SEEN is noted (habituation,
# arousal) but does not take the head: a covert look, as under HOLDING_STILL. A person, a hand, a
# sound, a touch, anything defensive still does. (Without it the looks at props that came back
# into the picture as the head nodded owned the head for 45-79 % of a dance's or an instrument's
# ticks, and a tick a reflex owns has no nod and no note: skill_sound.SoundSkills.post.)
ENGAGED_ACTS: frozenset[str] = frozenset()
ENGAGED_LOOKS = frozenset({"visual:thing", "visual:motion"})
# A freeze holds at least this long after its rise (ms): the head needs ~0.5 s to settle into the
# freeze pose, so a shorter hold is never SEEN still (fid:fix, headless: a bold duck's lights-off
# freeze held 0.2–0.5 s of stillness before it went back to pecking; was 300 ms)
FREEZE_MIN_MS = 1100.0


@dataclass(frozen=True, slots=True)
class ReflexParams:
    """One duck's reflex settings (from its genome: :func:`params_from_genome`)."""

    startle_gain: float = 1.0
    orient_gain: float = 1.0
    startle_threshold: float = 0.42
    orient_threshold: float = 0.07
    withdraw_threshold: float = 0.12
    latency_scale: float = 1.0
    freeze_ms: float = 1400.0
    hab_rate: float = 0.22
    hab_recover_s: float = 150.0
    hab_floor: float = 0.04
    sens_gain: float = 0.5
    sens_decay_s: float = 90.0
    veer_gain: float = 1.0
    boldness: float = 0.5
    side: int = 1
    young: float = 0.25
    habit: str = ""  # content.QUIRK_HABITS id ("double_look": the orienting look comes back once)


def params_from(
    axes: Mapping[str, float | int] | None = None,
    *,
    stage: str = "juvenile",
    side: int = 1,
    habit: str = "",
) -> ReflexParams:
    """Reflex settings from temperament axes 0..1 (``genome.Genome.axes``): boldness lowers the
    startle and shortens the freeze, wit quickens latencies and learning (faster habituation),
    exploration and sociability draw the head more readily; the young startle more easily and
    freeze longer. Use :func:`params_from_genome` for a real duck (one diversity mapping)."""
    a = dict(axes or {})
    bold = S.norm_axis(a.get("boldness"))
    wit = S.norm_axis(a.get("wit"))
    explore = S.norm_axis(a.get("exploration"))
    social = S.norm_axis(a.get("sociability"))
    young = {"hatchling": 1.0, "duckling": 0.6, "juvenile": 0.25, "adult": 0.0}.get(stage, 0.25)
    return ReflexParams(
        startle_gain=(1.25 - 0.6 * (bold - 0.2)) * (1.0 + 0.2 * young),
        orient_gain=0.8 + 0.4 * explore + 0.2 * social,
        # a bold duck shrugs off a moderate clap, a timid one jumps (the first brain's live
        # measurement: with one threshold for all, every temperament startled at the first clap)
        startle_threshold=0.30 + 0.45 * bold - 0.08 * young,
        latency_scale=1.15 - 0.3 * wit,
        freeze_ms=2200.0 * (1.35 - bold) * (1.0 + 0.3 * young),
        hab_rate=0.16 + 0.14 * wit,
        hab_recover_s=110.0 + 90.0 * (1.0 - wit),
        sens_gain=0.65 - 0.35 * bold,
        sens_decay_s=60.0 + 60.0 * (1.0 - bold),
        boldness=bold,
        side=1 if side >= 0 else -1,
        young=young,
        habit=habit,
    )


def params_from_genome(genome: Any) -> ReflexParams:
    """The reflex settings of a brain :class:`~duckmind.neuro.genome.Genome` (Duck Core's own
    draws: its axes 0..1, growth stage, innate side and quirk habit). The own-habituation
    constants are the genome's (``hab_rate``, ``hab_recover_tau_s``, ``hab_floor``,
    ``sens_gain``, ``sens_tau_s``), the same ones the brain's habituation uses."""
    base = params_from(
        dict(genome.axes), stage=str(genome.stage), side=int(genome.side), habit=str(genome.habit)
    )
    p = genome.p
    return replace(
        base,
        hab_rate=float(p.hab_rate),
        hab_recover_s=float(p.hab_recover_tau_s),
        hab_floor=float(p.hab_floor),
        sens_gain=float(p.sens_gain),
        sens_decay_s=float(p.sens_tau_s),
    )


@dataclass(frozen=True, slots=True)
class Active:
    """One reflex response in progress: onset after the latency, rise, hold, release."""

    reflex: str
    stimulus: str
    t0: int
    rise: int
    hold: int
    release: int
    pose: HeadPose
    blend: float
    strength: float
    side: int = 0
    bearing: float | None = None  # at onset, from the trunk (+ = left)
    elevation: float = 0.0
    then: str | None = None  # "freeze" after a startle, "double" for a double look
    escape: tuple[float, float, int] | None = None  # (vx, vyaw, ms) a quick step away
    yaw0: float = 0.0  # the odometry yaw at onset: the bearing is carried along the turn
    # (fid:fix) planned from the measured head: the words and the parts they name
    # (contract.REFLEX_PARTS); a dip the head is jerked into first (a startle with no room
    # above: ``dip`` for ``dip_ms``, then SNAP_UP_MS up to ``pose``); the pose it takes over
    # from (``start``: a freeze right after a startle goes on from the startle's pose, never
    # back through the act's own pose first); the look's aim off the bearing (one eye)
    zh: str = ""
    parts: tuple[str, ...] = ()
    dip: HeadPose | None = None
    dip_ms: int = 0
    start: HeadPose | None = None
    aim: float = 0.0
    # the eye's track of the sighting whose onset started this reflex (0: none — a sound, a touch, a
    # light change). Telemetry: ``ReflexOutput.track`` / ``MotorCommand.reflex_track``; a freeze or a
    # double look that follows a reflex carries the same track
    track: int = 0

    @property
    def end(self) -> int:
        return self.t0 + self.rise + self.hold + self.release


@dataclass(frozen=True, slots=True)
class Presented:
    """One onset the arcs processed this tick (fired or not), for the brain's habituation:
    ``Habituation.present(stimulus, raw)`` when ``depress``, ``sensitise(stimulus, scare)``."""

    stimulus: str  # "sound:clap", "touch:left", "visual:looming", "light:off", "body:bump"…
    raw: float  # 0..1 before habituation
    response: float  # after habituation × sensitisation × arousal × the duck's gain
    scare: float  # 0..1 the fright it carried (already × habituation²)
    reflex: str | None  # the reflex it fired, None below threshold
    bearing: float | None  # rad, + = left; None = unknown (the mono mic)
    depress: bool = True  # False: this class never habituates (falls, drops, …)
    track: int = 0  # the eye's track of the sighting this onset is about; 0 = none (sound, touch…)
    floor: bool = False  # the startle was fired by the startle floor (a sudden loud sound it is
    # not used to): a strong scare whatever the response read


@dataclass(frozen=True, slots=True)
class ReflexState:
    t_ms: int = -1
    rng: int = 0x5EED
    hab: tuple[tuple[str, float], ...] = ()
    sens: float = 0.0  # general sensitisation (jumpiness), decays with sens_decay_s
    sens_by: tuple[tuple[str, float], ...] = ()  # per sense (sound, visual, touch, …): the
    # sensitisation a scare left in that sense (a bang makes it jumpier to sounds most)
    active: Active | None = None
    lifted: bool = False
    fallen: bool = False  # a fall episode is open (until it is up again)
    refractory: tuple[tuple[str, int], ...] = ()
    prev_touch: str | None = None
    touch_last: tuple[tuple[str, int], ...] = ()  # site → last t_ms it was in contact
    walk_last: int = -1_000_000  # the last t_ms it was stepping (for reafference)
    bump_last: int = -1_000_000  # the last t_ms its body felt a brush or a stall (proprio)
    # track, motion, looming, the t_ms it was last in view — kept SEEN_KEEP_MS, not just the last
    # tick: a track that was out of the list for a moment is the same thing coming back
    prev_seen: tuple[tuple[int, float, float, int], ...] = ()
    prev_light: str = ""
    veer_side: int = 0
    swerve: tuple[float, int] = (0.0, 0)  # a brush's swerve: (vyaw bias, until t_ms)
    # turning away: (side, phase 0 none / 1 the stop / 2 the arc, when the phase began, the
    # odometry yaw when the arc began)
    avoid: tuple[int, int, int, float] = (0, 0, -1, 0.0)
    events: tuple[ReflexEvent, ...] = ()
    count: int = 0
    stop_until: int = -1  # a startle (or freeze) caught it walking: the legs stay still until then
    # (fid:fix) the last named look: when, and where it pointed in the duck's own odometry frame
    # (a new look at the same place within ORIENT_MERGE_MS is the same look: no second row)
    last_look: tuple[int, float, str] = (-1_000_000, 0.0, "")
    # the last sound onset the ear had not named and that did not startle: (t_ms, raw). When
    # duck-ear names it (SOUND_NAME_WAIT_MS at most), the class decides whether it startles.
    unnamed: tuple[int, float] | None = None


@dataclass(frozen=True, slots=True)
class ReflexOutput:
    """What the reflexes want this tick. The brain puts ``active``/``head``/``head_blend``/
    ``gaze`` into its :class:`~duckmind.neuro.contract.MotorCommand` (``reflex``, ``head``,
    ``head_blend``, ``gaze``), stops or limits locomotion (``halt``, ``vx_cap``), adds
    ``vyaw_bias`` to its turning, ends the current bout when ``interrupt``, feeds
    ``arousal_kick``/``fear_kick`` to its neuromodulators and ``presented`` to its habituation."""

    active: str | None = None
    stimulus: str | None = None
    head: HeadPose | None = None
    head_blend: float = 0.0
    gaze: Gaze | None = None
    halt: bool = False
    vx_cap: float | None = None
    vyaw_bias: float = 0.0
    interrupt: bool = False
    escape: tuple[float, float] | None = None
    stabilise: float = 0.0
    arousal_kick: float = 0.0
    fear_kick: float = 0.0
    events: tuple[ReflexEvent, ...] = ()
    recent: tuple[ReflexEvent, ...] = ()
    habituation: Mapping[str, float] = field(default_factory=dict)
    sensitisation: Mapping[str, float] = field(default_factory=dict)
    presented: tuple[Presented, ...] = ()
    hold: str | None = None  # the reflex holding the body (HOLDS), if any
    fallen: bool = False  # a fall episode is open
    lifted: bool = False  # it is being held off the floor
    turning_away: bool = False  # after a stall, a stumble or a wall: arcing away (vyaw_bias)
    track: int = 0  # the eye's track of the sighting that started the active reflex (0: none)


# ── helpers ───────────────────────────────────────────────────────────────────


def _walking(frame: SensorFrame) -> bool:
    return frame.body.posture == "walk" or abs(frame.body.vx) > 0.2


def _standing(frame: SensorFrame) -> bool:
    return frame.body.posture in ("stand", "walk")


def _side_of(bearing: float | None) -> int:
    if bearing is None:
        return 0
    return 1 if bearing >= 0.0 else -1


_SITE_ZH = {
    "head": msg("neuro.touch_site.head"),
    "back": msg("neuro.touch_site.back"),
    "body": msg("neuro.touch_site.body"),
    "left": msg("neuro.touch_site.left"),
    "right": msg("neuro.touch_site.right"),
    "tail": msg("neuro.touch_site.tail"),
    "beak": msg("neuro.touch_site.beak"),
}


def _side_zh(bearing: float | None) -> str:
    """Which way, in plain words (+ = the duck's left)."""
    if bearing is None:
        return ""
    if abs(bearing) < 0.2:
        return msg("neuro.side.ahead")
    return msg("neuro.side.left") if bearing > 0.0 else msg("neuro.side.right")


Head4 = tuple[float, float, float, float]
HEAD_HOME: Head4 = (0.0, 0.0, 0.0, 0.0)


def head_now(frame: SensorFrame) -> Head4:
    """The MEASURED head (its own proprioception: robot.state's joints, BodySense.head) as offsets
    from robotd's HOME, rad: the neck (+ leaning back), the head pitch (− looking up), the yaw and
    the roll (+ the duck's left)."""
    h = frame.body.head
    if len(h) < 4:
        return HEAD_HOME
    return (h[0] - HOME_PITCH, h[1] - HOME_PITCH, h[2], h[3])


def _pose_key(posture: str) -> str:
    return posture if posture in NOSE_REACH else "stand"


def _up_now(head: Head4, posture: str) -> float:
    """How far up/back the nose points now, rad: seated the neck back and the head pitch up
    together (the page's own 「往後仰」 measure); standing and walking the head pitch alone — the
    walking network holds the neck (~0.13 rad forward, whatever it is asked: pet:body), so a
    look up there is the head pitch's (the motor's neck→pitch conversion)."""
    if _pose_key(posture) == "sit":
        return head[0] - head[1]
    return -head[1]


# How far down the nose gets (rad below HOME; a look down standing is the head pitch's: pet:body
# glance-down gain ≤ 0.65 of the envelope's −20°; seated neck and head together)
NOSE_DOWN_REACH = {"stand": 0.22, "walk": 0.22, "sit": 0.45}
# The part of a steady nose command the body holds (a freeze HELD at 12.4–13.1° up for a 19.5°
# command standing: ~0.65; a snap from rest peaks higher, 17.3° of 20°; seated ~0.75)
NOSE_GAIN = {"stand": 0.68, "walk": 0.68, "sit": 0.75}


def nose_room(frame: SensorFrame) -> tuple[float, float]:
    """(fid:fix) How far the nose can still go up and how far down from where the MEASURED head
    points now (rad): for the brain's moments (a look up or down that has no room is not seen)."""
    key = _pose_key(frame.body.posture)
    up = _up_now(head_now(frame), frame.body.posture)
    return NOSE_REACH[key] - up, NOSE_DOWN_REACH[key] + up


def _up_of(pose: HeadPose, posture: str) -> float:
    """Where a reflex pose would hold the nose on this body (rad, + up), inside its reach: the
    command (the motor keeps 0.35·hp − 0.25·np through its standing neck→pitch conversion) × the
    body's gain, clipped to what it reaches."""
    key = _pose_key(posture)
    cmd = PITCH_PER_CLIP * pose.hp - NECK_PER_CLIP * pose.np
    if key != "sit":
        cmd = S.clamp(cmd, -PITCH_PER_CLIP, PITCH_PER_CLIP)  # (the standing envelope: hp ±1)
    return S.clamp(cmd * NOSE_GAIN[key], -NOSE_DOWN_REACH[key], NOSE_REACH[key])


def _yaw_max(posture: str) -> float:
    """The head yaw's range (rad): the validated ±1.0, walking kept inside the motor's margin."""
    return 0.9 if posture == "walk" else 1.0


def _turn_part(delta: float) -> str | None:
    """A head turn big enough to be seen, by the way it goes (+ = the duck's left)."""
    if abs(delta) < TURN_SEEN_RAD:
        return None
    return "yaw_left" if delta > 0.0 else "yaw_right"


def _way(lean: int, d: _Draw) -> int:
    """Which way when the senses do not say: its own side (the innate side's slow drift) about
    two times in three, the other side otherwise (the review: 24/33 to its own side, all alike)."""
    own = 1 if lean >= 0 else -1
    return own if d.u(0.0, 1.0) < OWN_SIDE_P else -own


def _planned_parts(pose: HeadPose, head: Head4, posture: str) -> list[str]:
    """The visible parts a reflex pose makes from where the head is now: the nose up/down (its
    reach on this body), the neck pulled back (seated), the head turned."""
    parts: list[str] = []
    nose_delta = _up_of(pose, posture) - _up_now(head, posture)
    if nose_delta >= SEEN_RAD:
        parts.append("up")
    elif nose_delta <= -SEEN_RAD:
        parts.append("down")
    turn = _turn_part(pose.hy * YAW_PER_CLIP - head[2])
    if turn is not None:
        parts.append(turn)
    return parts


_TURN_ZH = {"yaw_left": msg("neuro.side.left"), "yaw_right": msg("neuro.side.right")}


def _startle_zh(parts: tuple[str, ...], *, stop: bool = False) -> str:
    """「嚇一跳，…」 with exactly the parts it will make: 「頭猛往後仰」 (the nose thrown up/back),
    seated with the neck 「頭猛往後一縮」, from a raised head 「頭猛一沉」 (jerked down, then up
    again), and 「甩向右邊」 when the sideways jerk is big enough to see; walking 「頭猛甩向右邊，
    停下來」 (the legs' stop is judged by the page's witness)."""
    if stop and not ({"up", "dip", "neck"} & set(parts)):
        turns = [side for key, side in _TURN_ZH.items() if key in parts]
        if turns:
            return msg("neuro.rx.startle.walking_jerk_stop", side=turns[0])
        return msg("neuro.rx.startle.walking_stop")
    bits: list[str] = []
    if "dip" in parts:
        bits.append(msg("neuro.rx.startle.part.dip"))
    elif "up" in parts and "neck" in parts:
        bits.append(msg("neuro.rx.startle.part.up_neck"))
    elif "up" in parts:
        bits.append(msg("neuro.rx.startle.part.up"))
    elif "neck" in parts:
        bits.append(msg("neuro.rx.startle.part.neck"))
    for key, side in _TURN_ZH.items():
        if key in parts:
            bits.append(msg("neuro.rx.startle.part.jerk_side", side=side))
    return (
        join(msg("neuro.rx.startle.with_parts.prefix"), lst(bits))
        if bits
        else msg("neuro.rx.startle.plain")
    )


def _zh(
    reflex: str,
    stimulus: str,
    bearing: float | None = None,
    *,
    posture: str = "",
    escape: tuple[float, float, int] | None = None,
) -> str:
    """The reflex in the owner's plain words (the reflexes that plan their motion — startle,
    orient — carry their own, :attr:`Active.zh`). Durations are said by the page when the hold
    ends, from the body (neuro_witness: 「定住 1.2 秒」). It says only what this body does
    (fid:motor 2026-09-28)."""
    channel, _, kind = stimulus.partition(":")
    side = _side_zh(bearing)
    if reflex == "orient":
        if bearing is None:
            return (
                msg("neuro.rx.orient.listen_no_direction")
                if channel == "sound"
                else msg("neuro.rx.orient.look")
            )
        return (
            msg("neuro.rx.orient.back_to_front")
            if side == "正前方"
            else msg("neuro.rx.orient.turn_to.57aec9", side=side)
        )
    if reflex == "startle":
        return msg("neuro.rx.startle.default")
    if reflex == "freeze":
        return join(
            msg("neuro.rx.freeze.plain"),
            msg("neuro.rx.freeze.staring_suffix", side=side) if side else "",
        )
    if reflex == "withdraw":
        where = _SITE_ZH.get(kind, side)
        text = (
            msg("neuro.rx.withdraw.from_site", site=where)
            if where
            else msg("neuro.rx.withdraw.plain")
        )
        if escape is not None:
            text += (
                msg("neuro.rx.withdraw.step_back_suffix")
                if escape[0] < 0.0
                else msg("neuro.rx.withdraw.step_away_suffix")
            )
        return text
    if reflex == "lean_in":
        where = _SITE_ZH.get(kind, "")
        return (
            msg("neuro.rx.lean_in.stroked_at", site=where)
            if where
            else msg("neuro.rx.lean_in.plain")
        )
    if reflex == "veer":
        turn = (
            ""
            if bearing is None
            else (msg("neuro.rx.veer.left") if bearing > 0 else msg("neuro.rx.veer.right"))
        )
        return msg("neuro.rx.veer.turning", turn=turn) if turn else msg("neuro.rx.veer.plain")
    if reflex == "lifted":
        return REFLEXES["lifted"]
    if reflex == "stabilise":
        # (fid:fix: a jolt of the trunk while walking — a push, a trip, a bump: the body STOPS and
        # stands steady; holding the head steady while walking is not something this body does)
        return msg("neuro.rx.stabilise")
    what = {
        "sound": msg("neuro.rx.channel.sound"),
        "visual": msg("neuro.rx.channel.visual"),
        "touch": msg("neuro.rx.channel.touch"),
        "light": msg("neuro.rx.channel.light"),
        "body": msg("neuro.rx.channel.body"),
        "tof": msg("neuro.rx.channel.tof"),
        "habit": msg("neuro.rx.channel.habit"),
    }.get(channel, "")
    return (
        msg("neuro.rx.generic_with_channel", reflex_name=REFLEXES.get(reflex, reflex), cause=what)
        if what
        else REFLEXES.get(reflex, reflex)
    )


class _Draw:
    def __init__(self, rng: int) -> None:
        self.rng = rng

    def u(self, low: float, high: float) -> float:
        self.rng, value = S.between(self.rng, low, high)
        return value

    def g(self) -> float:
        self.rng, value = S.gauss(self.rng)
        return value

    def ln(self, median: float, sigma: float) -> float:
        self.rng, value = S.lognormal(self.rng, median, sigma)
        return value


def _latency(base_ms: float, params: ReflexParams, arousal: float, d: _Draw) -> int:
    factor = params.latency_scale * (1.35 - 0.6 * S.clamp(arousal, 0.0, 1.0))
    return max(15, int(base_ms * factor * (1.0 + 0.15 * d.g())))


def _startle_floor(raw: float, kind_k: float, level: float) -> bool:
    """A sudden loud sound the duck is not yet used to startles it whatever else it is (see
    :data:`STARTLE_FLOOR_RAW`)."""
    return kind_k >= STARTLE_FLOOR_KIND and raw >= STARTLE_FLOOR_RAW and level >= STARTLE_FLOOR_HAB


def _gain_a(arousal: float) -> float:
    """The arousal neuromodulator's gain on every response (0.75 … 1.25): strong enough to show,
    not so strong that a climbing arousal swamps habituation (live, 2026-09-27)."""
    return 0.75 + 0.5 * S.clamp(arousal, 0.0, 1.0)


# Reflex head poses are scaled by what the body is doing NOW (live physics 2026-09-27, the first
# brain agent on :8891: a startle head snap during a rise made the duck fall): while getting up,
# sitting down or not knowing (transition/unknown) no reflex overlay at all — the head stays with
# what it was doing — except a withdrawal's turn away, kept at 40 %. Seated, standing and walking
# the whole pose (fid:motor 2026-09-28: standing and walking it was cut to 50–60 %, and the
# owner saw no startle at all — ~7–10° of head pitch, the neck not moving; the A100 validated the
# head's whole range while walking and standing — yaw ±1.0 rad, look-up −0.35 rad, A2 half-down,
# saccades, 1-rad steps: 0 falls beyond the background — and the motor clamps every pose into
# those envelopes and converts a standing neck pull-back into head pitch). Righting, lifted and
# the brace hold the head at or near HOME and are not scaled.
UNSCALED = frozenset({"righting", "lifted", "stabilise"})


def posture_scale(posture: str, reflex: str) -> tuple[float, float]:
    """(pose scale, yaw scale) of ``reflex``'s head pose in ``posture`` (see the note above)."""
    if reflex in UNSCALED or posture in ("sit", "stand", "walk"):
        return 1.0, 1.0
    return (0.2, 0.4) if reflex == "withdraw" else (0.0, 0.0)


def _scaled(pose: HeadPose, k: float, k_yaw: float) -> HeadPose:
    return HeadPose(
        np=pose.np * k, hp=pose.hp * k, hy=pose.hy * k_yaw, hr=pose.hr * k, mouth=pose.mouth
    )


def _mix_pose(a: HeadPose, b: HeadPose, w: float) -> HeadPose:
    w = S.clamp(w, 0.0, 1.0)
    return HeadPose(
        np=a.np + (b.np - a.np) * w,
        hp=a.hp + (b.hp - a.hp) * w,
        hy=a.hy + (b.hy - a.hy) * w,
        hr=a.hr + (b.hr - a.hr) * w,
        mouth=a.mouth + (b.mouth - a.mouth) * w,
    )


# ── the responses ─────────────────────────────────────────────────────────────


def _startle(
    t: int,
    stim: str,
    s: float,
    bearing: float | None,
    params: ReflexParams,
    arousal: float,
    d: _Draw,
    *,
    fear: float = 0.0,
    yaw0: float = 0.0,
    lean: int = 1,
    head: Head4 = HEAD_HOME,
    posture: str = "stand",
) -> Active:
    """The startle (fid:motor 2026-09-28; fid:fix: planned from where the head IS). The head
    snaps back and UP at once — the validated look-up (head pitch −0.35 rad) and, seated, the neck
    pulled back (standing the motor turns that into more head pitch) — when the body has the room
    to show it (:data:`NOSE_REACH`); with the head already raised (a second clap while it is still
    afraid) it is jerked DOWN first and snapped back up (:data:`DIP_RAD`: 「頭猛一沉」). The head is
    jerked sideways from where it points — away from a looming thing it saw; for a sound with no
    direction its own side about two times in three, ±30 % in size (the review: the same ~20° to
    the same side every time looked mechanical). The beak snaps open for a moment
    (:data:`BEAK_SNAP_MS`), the legs stop (the brain's halt; :data:`STARTLE_STOP_MS` walking)."""
    side = _side_of(bearing)
    key = _pose_key(posture)
    k = 0.8 + 0.2 * S.clamp(s, 0.0, 1.0)
    pose = HeadPose(
        np=S.clamp(-(0.8 + 0.2 * s) * (1.0 + 0.08 * d.g()), -1.0, 0.0),
        hp=S.clamp(0.9 * k + 0.06 * d.g(), 0.7, 1.0),
        hy=0.0,
        hr=S.clamp(0.25 * d.g(), -0.5, 0.5),
        mouth=S.clamp(0.8 + 0.2 * s, 0.0, 1.0),
    )
    up_now = _up_now(head, posture)
    parts: list[str] = []
    dip: HeadPose | None = None
    dip_ms = 0
    walking = key == "walk"
    if NOSE_REACH[key] - up_now >= SEEN_RAD:
        parts.append("up")
    elif walking:
        pass  # (walking the look-up comes as the legs stop; a dip does not show under the gait)
    else:
        # no room above: down first (from where the nose is, its own depth), then up again
        depth = DIP_RAD * S.clamp(math.exp(0.2 * d.g()), 0.8, 1.25)
        down = S.clamp((up_now - depth) / PITCH_PER_CLIP, -1.0, 1.0)
        dip = HeadPose(np=0.0, hp=down, hy=0.0, hr=0.0, mouth=pose.mouth)
        dip_ms = int(d.u(*DIP_MS))
        parts.append("dip")
    if key == "sit" and head[0] <= -NECK_BACK_SEEN and dip is None:
        parts.append("neck")  # seated, from a neck leaning forward: the pull-back shows
    # the sideways jerk, from where the head points now
    if bearing is not None and side != 0:
        way = -side  # away from what it saw coming
    else:
        way = _way(lean, d)
    size = S.clamp((0.30 + 0.25 * s) * math.exp(JERK_SPREAD * d.g()), 0.2, 0.62)
    ymax = _yaw_max(posture)
    yaw_now = S.clamp(head[2], -ymax, ymax)
    target = yaw_now + way * size
    if abs(target) > ymax:  # no room that way: the other way
        target = yaw_now - way * size
    target = S.clamp(target, -ymax, ymax)
    pose = replace(pose, hy=target / YAW_PER_CLIP)
    if dip is not None:
        dip = replace(dip, hy=pose.hy)  # the jerk aside comes with the dip
    turn = _turn_part(target - head[2])
    if turn is not None:
        parts.append(turn)
    if walking and turn is not None and "up" in parts:
        # (fid:fix, headless 8 × 3 walking claps: the walking network lifts the head only as the
        # legs stop — 「頭往後仰」 reached 10° after 0.5–0.7 s, 0/12 within the flinch's 0.4 s; the
        # sideways jerk came in 0.22–0.36 s, 13/14) — walking it names the jerk and the stop
        parts.remove("up")
    snap = dip_ms + (SNAP_UP_MS if dip is not None else 0)
    words = _startle_zh(tuple(parts), stop=walking)
    if posture not in ("sit", "stand", "walk"):
        # getting up or sitting down: the head stays where it is (posture_scale), so nothing is
        # named — the legs' balance first
        parts, dip, dip_ms, snap = [], None, 0, 0
        words = msg("neuro.rx.startle.changing_posture")
    return Active(
        "startle",
        stim,
        t + _latency(60.0, params, arousal, d),
        int(d.u(50, 80)),
        int(d.u(350, 600)) + snap,
        int(d.u(350, 550)),
        pose,
        1.0,
        s,
        side,
        bearing,
        then="freeze" if (s > 0.3 or fear > 0.15) else None,
        yaw0=yaw0,
        zh=words,
        parts=tuple(parts),
        dip=dip,
        dip_ms=dip_ms,
    )


def _freeze(
    t: int,
    stim: str,
    s: float,
    bearing: float | None,
    params: ReflexParams,
    d: _Draw,
    *,
    fear: float = 0.0,
    yaw0: float = 0.0,
    head: Head4 | None = None,
    start: HeadPose | None = None,
) -> Active:
    """Freezing: the head raised high and STILL (fid:motor: ~12–16° of head pitch up, the neck
    too seated; the whole pose, nothing of the act underneath leaks). Looking at what it saw; with
    no direction it keeps looking where the head points (fid:fix: the old pose swung the head
    back to the middle first — a visible move inside 「定住」). After a startle it goes on from the
    startle's own pose (``start``), never back through the act's pose first. (fid:fix) Its words
    claim the stillness only (「定住不動」); how far the head first moved into the pose depends on
    the walking network's neck (standing it drifts forward as the head pitch rises: headless, a
    planned 「抬起頭」 made < 10° of nose in 8 of 15), so the page's witness says it, measured
    (「身體：頭先抬高 19°、頭定住不動」)."""
    side = _side_of(bearing)
    if bearing is not None:
        look = S.clamp(bearing / 0.6, -1.0, 1.0) * 0.8
    elif start is not None:
        look = start.hy
    elif head is not None:
        look = head[2] / YAW_PER_CLIP
    else:
        look = 0.0
    pose = HeadPose(
        np=-0.45 + 0.05 * d.g(), hp=S.clamp(0.65 + 0.08 * d.g(), 0.5, 0.85), hy=look, hr=0.0
    )
    median = params.freeze_ms * (0.5 + s) * (0.6 + 1.4 * S.clamp(fear, 0.0, 1.0))
    hold = int(S.clamp(d.ln(median, 0.35), FREEZE_MIN_MS, 6000.0))
    return Active(
        "freeze",
        stim,
        t,
        int(d.u(150, 250)),
        hold,
        int(d.u(400, 700)),
        pose,
        1.0,
        s,
        side,
        bearing,
        yaw0=yaw0,
        start=start,
    )


def _look_zh(target: float, delta: float) -> str:
    """An orienting turn in words, by the way the head goes and where it ends (+ = left)."""
    if abs(target) < 0.12:
        return msg("neuro.rx.orient.back_to_front")
    where = msg("neuro.side.left") if target > 0.0 else msg("neuro.side.right")
    if target * delta > 0.0:
        return msg("neuro.rx.orient.turn_to.8566fa", site=where)
    return msg("neuro.rx.orient.turn_back_a_little", site=where)  # (it was turned past it)


def _orient(
    t: int,
    stim: str,
    r: float,
    bearing: float | None,
    elevation: float,
    params: ReflexParams,
    arousal: float,
    d: _Draw,
    *,
    yaw0: float = 0.0,
    again: bool = False,
    cock: int = 0,
    head: Head4 = HEAD_HOME,
    posture: str = "stand",
) -> Active | None:
    """Orienting (fid:fix 2026-09-28: planned from where the MEASURED head points). A look that
    would turn the head less than :data:`TURN_SEEN_RAD` is a re-fixation — the head is already on
    it: None (no row, no motion; the brain still hears/sees it). A thing straight ahead with the
    head already looking ahead is viewed with one eye (the head turned :data:`ONE_EYE_RAD` aside:
    a bird's lateral look). The words say the way the head goes (「頭轉回來一點，看右邊」 when it
    was turned past the thing) and name that turn (``parts``) for the page's witness."""
    hold = int(d.u(600, 1600) * (1.0 + r))
    if posture not in ("sit", "stand", "walk"):
        return None  # getting up or sitting down: the head stays (posture_scale): a covert look
    ymax = _yaw_max(posture)
    yaw_now = S.clamp(head[2], -ymax, ymax)
    if bearing is None:
        # a mono microphone hears no direction: it listens, the head cocked aside — its own side
        # two times in three — from where it points, and up a little
        way = _way(cock if cock in (-1, 1) else params.side, d)
        size = d.u(*LISTEN_YAW_RAD) * (0.85 + 0.3 * S.clamp(r, 0.0, 1.0))
        target = yaw_now + way * size
        if abs(target) > ymax:
            target, way = yaw_now - way * size, -way
        target = S.clamp(target, -ymax, ymax)
        turn = _turn_part(target - head[2])
        if turn is None:
            return None
        pose = HeadPose(
            np=-0.1 - 0.15 * r,
            hp=S.clamp(0.4 + 0.5 * r + 0.06 * d.g(), 0.3, 1.0),
            hy=target / YAW_PER_CLIP,
            hr=way * S.clamp(0.5 + 0.4 * r, 0.0, 0.95),
        )
        return Active(
            "orient",
            stim,
            t + _latency(160.0, params, arousal, d),
            int(d.u(200, 320)),
            hold,
            int(d.u(400, 700)),
            pose,
            1.0,
            r,
            0,
            None,
            yaw0=yaw0,
            zh=msg("neuro.rx.orient.listen_no_direction"),
            parts=(turn,),
        )
    # With a direction (the camera, a touch): the gaze turns the head toward it (up to the
    # validated ±1.0 rad — the motor's gaze system); the overlay adds an alert lift and a cock of
    # the head toward that side
    target = S.clamp(bearing, -ymax, ymax)
    aim = 0.0
    if abs(target - head[2]) >= TURN_SEEN_RAD:
        zh = _look_zh(target, target - head[2])
    elif abs(bearing) < AHEAD_RAD and abs(head[2]) < TURN_SEEN_RAD:
        # straight ahead and already looking ahead: one eye on it
        way = _way(cock if cock in (-1, 1) else params.side, d)
        aim = way * d.u(*ONE_EYE_RAD)
        target = S.clamp(bearing + aim, -ymax, ymax)
        zh = msg("neuro.rx.orient.one_eye_ahead")
    else:
        return None  # the head is already on it: a re-fixation
    turn = _turn_part(target - head[2])
    if turn is None:
        return None
    toward = 1 if target >= 0.0 else -1
    pose = HeadPose(
        np=-0.1,
        hp=S.clamp(0.3 + 0.2 * r + 0.05 * d.g(), 0.2, 0.6),
        hy=target / YAW_PER_CLIP,
        hr=toward * S.clamp(0.3 + 0.3 * r, 0.0, 0.6),
    )
    double = params.habit == "double_look" and r > 0.25 and not again
    return Active(
        "orient",
        stim,
        t + (0 if again else _latency(140.0, params, arousal, d)),
        int(d.u(120, 220)),
        hold,
        int(d.u(400, 700)),
        pose,
        0.6,
        r,
        _side_of(bearing),
        bearing,
        elevation,
        then="double" if double else None,
        yaw0=yaw0,
        zh=zh,
        parts=(turn,),
        aim=aim,
    )


def _withdraw(
    t: int,
    stim: str,
    site: str,
    s: float,
    standing: bool,
    params: ReflexParams,
    arousal: float,
    d: _Draw,
    look_side: int = 1,
    open_side: int = 0,
    *,
    walking: bool = False,
    head: Head4 = HEAD_HOME,
    posture: str = "stand",
) -> Active:
    """Withdrawal (fid:motor 2026-09-28): the head pulls away from the touch at once — big
    enough to be seen standing (yaw 25–43° away from a side touch; a touch on the head or the
    beak throws the head back and up, the validated look-up, and turns the face aside) — and a
    standing duck steps away: a forward arc away from a side, a step forward from behind, and
    from a touch in front (the hand is where it would walk) a small back arc toward the side it
    saw open (reversing a small arc when cornered: owner 2026-09-28). The steps last long enough
    for the legs to start (:data:`WITHDRAW_STEP_MS`)."""
    esc: tuple[float, float, int] | None = None
    step_ms = int(d.u(*WITHDRAW_STEP_MS))
    if site in ("left", "right"):
        away = -1 if site == "left" else 1
        pose = HeadPose(
            np=-0.3 - 0.3 * s,
            hp=S.clamp(0.3 + 0.2 * s, 0.0, 0.6),
            hy=away * S.clamp(0.7 + 0.5 * s + 0.05 * d.g(), 0.6, 1.25),
            hr=away * S.clamp(0.3 + 0.4 * s, 0.0, 0.8),
        )
        if standing and s > 0.4:
            esc = (0.40, away * S.clamp(0.6 + 0.3 * s, 0.0, 0.9), step_ms)
    elif site in ("head", "beak"):
        aside = open_side if open_side in (-1, 1) else (1 if d.g() >= 0.0 else -1)
        pose = HeadPose(
            np=-S.clamp(0.7 + 0.3 * s, 0.0, 1.0),
            hp=S.clamp(0.8 + 0.2 * s, 0.6, 1.0),
            hy=aside * S.clamp(0.4 + 0.3 * s + 0.05 * d.g(), 0.3, 0.8),
            hr=0.0,
        )
        if standing and not walking and s > 0.35:  # (walking: it stops — the brain's halt)
            esc = (WITHDRAW_BACK_VX, float(aside) * 1.0, WITHDRAW_BACK_MS)
    else:  # back, tail, body: ducks down, looks round at it (over its own side's shoulder)
        look = 1 if look_side >= 0 else -1
        pose = HeadPose(
            np=0.6,
            hp=-0.3,
            hy=look * (1.2 if site == "tail" else 0.95),
            hr=look * 0.3,
        )
        if standing and s > 0.45:
            esc = (0.40, -look * 0.5, step_ms)
    return Active(
        "withdraw",
        stim,
        t + _latency(90.0, params, arousal, d),
        int(d.u(70, 120)),
        int(d.u(450, 800)),
        int(d.u(400, 700)),
        pose,
        1.0,
        s,
        0,
        None,
        escape=esc,
        parts=tuple(_planned_parts(pose, head, posture)),
    )


def _lean_in(
    t: int,
    stim: str,
    site: str,
    s: float,
    params: ReflexParams,
    arousal: float,
    d: _Draw,
    *,
    head: Head4 = HEAD_HOME,
    posture: str = "stand",
) -> Active:
    """Leaning into a touch (fid:motor 2026-09-28): slowly, the head goes forward and down
    toward the hand with a cosy tilt — seen standing too (it was ~1–3° standing: the neck barely
    follows small standing commands, so the head pitch and roll carry it: 10–18° down, 8–13°
    of tilt, the head turned toward a side touch)."""
    tilt = params.side if params.side in (-1, 1) else 1
    if site in ("left", "right"):
        toward = 1 if site == "left" else -1
        pose = HeadPose(
            np=0.45,
            hp=-S.clamp(0.45 + 0.2 * s + 0.05 * d.g(), 0.3, 0.75),
            hy=toward * S.clamp(0.55 + 0.2 * s, 0.4, 0.85),
            hr=toward * S.clamp(0.6 + 0.2 * s, 0.4, 0.9),
        )
    elif site in ("back", "tail", "body"):
        pose = HeadPose(
            np=0.4,
            hp=-S.clamp(0.4 + 0.2 * s + 0.05 * d.g(), 0.3, 0.7),
            hy=tilt * 0.45,
            hr=tilt * S.clamp(0.5 + 0.2 * s, 0.4, 0.8),
        )
    else:  # the head or the beak: it pushes its head into the hand, eyes half closed
        pose = HeadPose(
            np=S.clamp(0.55 + 0.2 * s, 0.4, 0.8),
            hp=-S.clamp(0.45 + 0.25 * s + 0.05 * d.g(), 0.3, 0.8),
            hy=0.1 * d.g(),
            hr=tilt * S.clamp(0.6 + 0.25 * s, 0.4, 0.95),
        )
    return Active(
        "lean_in",
        stim,
        t + _latency(350.0, params, arousal, d),
        int(d.u(550, 850)),
        int(d.u(900, 1500)),
        int(d.u(700, 1000)),
        pose,
        1.0,
        s,
        0,
        None,
        parts=tuple(_planned_parts(pose, head, posture)),
    )


def _simple(
    reflex: str,
    t: int,
    stim: str,
    pose: HeadPose,
    blend: float,
    rise: int,
    hold: int,
    release: int,
    s: float = 1.0,
) -> Active:
    return Active(reflex, stim, t, rise, hold, release, pose, blend, s)


# ── the step ──────────────────────────────────────────────────────────────────


def _outranks(new: str, current: Active | None, t: int) -> bool:
    if current is None or t >= current.end:
        return True
    if t >= current.t0 + current.rise + current.hold:  # the current one is letting go
        return PRIORITY[new] >= PRIORITY.get(current.reflex, 0) - 40
    return PRIORITY[new] >= PRIORITY.get(current.reflex, 0)


def step(
    params: ReflexParams,
    state: ReflexState,
    frame: SensorFrame,
    *,
    arousal: float = 0.5,
    asleep: bool = False,
    expect_touch: float = 0.0,
    fear: float = 0.0,
    comfort: float = 0.5,
    walking: bool | None = None,
    hab: Mapping[str, float] | None = None,
    sens: Mapping[str, float] | None = None,
    lean: float = 0.0,
    free_side: int = 0,
    way_clear: bool = False,
    turn_need: float | None = None,
    still_act: bool = False,
    engaged: bool = False,
    expect_hand: float | None = None,
    commit_side: int = 0,
    flight: bool = False,
) -> tuple[ReflexState, ReflexOutput]:
    """One reflex tick at ``frame.t_ms``.

    ``flight`` (2026-10-04, owner: 「嚇一跳也是全速逃離」): the brain has decided the fright is a
    flight (it has not gone and the duck can run): the freeze the startle goes on into is only the
    beat it needs to be seen (:data:`FREEZE_MIN_MS`) and the fear does not hold it on — then the
    body is let go and the brain runs.

    ``still_act`` (fid:fix): the brain's act holds the head still (freeze, startle, still) —
    an orienting onset then moves nothing and gets no row (a covert look). ``engaged``: the
    brain's act is one of :data:`ENGAGED_ACTS`: only a thing or a movement newly seen (:data:`
    ENGAGED_LOOKS`) is covert then.

    ``arousal`` is the brain's neuromodulator (0..1); ``asleep`` raises every threshold (a
    sleeper wakes only to strong stimuli); ``expect_touch`` (0..1) is how much the brain saw a
    touch coming (a hand in view): an expected touch is leaned into, an unexpected one withdrawn
    from. ``fear`` (0..1, the brain's) makes a startle end in a freeze and the freeze hold as
    long as the fear does; ``comfort`` (0..1) lowers withdrawal (neutral at 0.5). ``walking``:
    the brain's own view of whether it is stepping (default: from the frame). ``hab`` / ``sens``:
    the brain's habituation level (1 = fresh) and extra gain (≥ 0) per stimulus class — given,
    the module keeps none of its own and reports each processed onset in ``presented``.
    ``lean`` (−1..1): the brain's slowly drifting side preference (its innate side's OU) — which
    way to turn, cock or look when the senses do not say; ``free_side`` (+1 left, −1 right, 0
    unknown): the side the camera's floor line saw more open; ``way_clear``: the way ahead is
    seen clear; ``turn_need``: how far (rad) a turn-away should go (the open heading it saw).

    ``expect_hand`` (0..1): the part of ``expect_touch`` that comes from a hand seen or a voice
    heard — not from an earlier touch (default: all of ``expect_touch``). ``commit_side`` (+1
    left, −1 right, 0 none): an escape (unstick.py) has chosen the side to turn away to; every
    turn away goes that way, a running one is turned to it."""
    t = frame.t_ms
    dt = 0.02 if state.t_ms < 0 else max(0.0, (t - state.t_ms) / 1000.0)
    d = _Draw(state.rng)
    own_hab = hab is None
    own_sens = sens is None
    hab_own = dict(state.hab) if own_hab else {}
    sens_by = dict(state.sens_by) if own_sens else {}
    general = state.sens * (1.0 - min(1.0, dt / params.sens_decay_s)) if own_sens else 0.0
    lifted = state.lifted
    fallen = state.fallen
    fear = S.clamp(fear, 0.0, 1.0)
    yaw_now = frame.body.odom[2]
    for key in list(hab_own):
        hab_own[key] = hab_own[key] + (1.0 - hab_own[key]) * min(1.0, dt / params.hab_recover_s)
    for key in list(sens_by):
        sens_by[key] = sens_by[key] * (1.0 - min(1.0, dt / params.sens_decay_s))
    refractory = {k: v for k, v in state.refractory if v > t}
    events: list[ReflexEvent] = []
    presented: list[Presented] = []
    active = state.active
    stop_until = state.stop_until
    last_look = state.last_look
    head = head_now(frame)  # (fid:fix) its own proprioception: where the head IS
    posture = frame.body.posture
    if active is not None and active.reflex == "freeze" and t >= active.t0 + active.rise:
        # a freeze holds while the brain's fear does (a leaky integrator ends it, not a timer)
        if fear >= FREEZE_FEAR_HOLD and t - active.t0 < FREEZE_MAX_MS and not flight:
            local_end = active.t0 + active.rise + active.hold
            if t >= local_end - 20:
                active = replace(active, hold=t - active.t0 - active.rise + 40)
    if active is not None and t >= active.end:
        if active.then == "freeze":
            # (fid:fix) a startle goes on into its freeze from its own pose — it is not released
            # back to the act's pose first (that was a visible down-and-up inside 「定住」)
            carried = active.track
            active = _freeze(
                t,
                active.stimulus,
                active.strength,
                active.bearing,
                params,
                d,
                fear=fear,
                yaw0=active.yaw0,
                head=head,
                start=active.pose,
            )
            if active is not None and carried:
                active = replace(active, track=carried)
            if active is not None and flight:
                active = replace(active, hold=int(FREEZE_MIN_MS))
        elif active.then == "double" and active.bearing is not None:
            carried = active.track
            again = S.clamp(0.6 * active.strength, 0.0, 1.0)
            second = _orient(
                t,
                active.stimulus,
                again,
                active.bearing,
                active.elevation,
                params,
                arousal,
                d,
                yaw0=active.yaw0,
                again=True,
                head=head,
                posture=posture,
            )
            active = second if second is None or not carried else replace(second, track=carried)
            if second is not None:
                events.append(
                    ReflexEvent(
                        t,
                        "orient",
                        "habit:double_look",
                        round(again, 3),
                        1.0,
                        msg("neuro.rx.orient.double_look", look=second.zh),
                        second.parts,
                        carried,
                    )
                )
        else:
            active = None
    arousal_kick = 0.0
    fear_kick = 0.0
    gain_a = _gain_a(arousal) * (0.4 if asleep else 1.0)
    standing = _standing(frame)
    walking_now = _walking(frame) if walking is None else bool(walking)
    # its own side when the senses do not say which way: the innate side, drifting slowly
    lean_side = 1 if 0.45 * params.side + S.clamp(lean, -1.0, 1.0) >= 0.0 else -1
    open_side = free_side if free_side in (-1, 1) else lean_side
    commit = commit_side if commit_side in (-1, 1) else 0
    if commit:
        open_side = commit
    hand_cue = expect_touch if expect_hand is None else expect_hand
    need = TURN_DEFAULT_RAD if turn_need is None else turn_need
    need = S.clamp(need, TURN_NEED_RAD[0], TURN_NEED_RAD[1])

    def sense_of(stim: str) -> str:
        return stim.split(":", 1)[0]

    def level_of(stim: str) -> float:
        if hab is not None:
            return S.clamp(float(hab.get(stim, 1.0)), 0.0, 1.0)
        return hab_own.get(stim, 1.0)

    def floor_level_of(stim: str) -> float:
        """The habituation the startle floor asks about: with the brain's habituation it is the larger of the class's own
        level and the loud-sound level (a duck that has only heard quiet sounds of a class is not used to a loud one);
        on its own habituation (no brain) the class's level, as before."""
        own = level_of(stim)
        if hab is None or HAB_LOUD_KEY not in hab:
            return own  # (no loud-sound level to ask: a plain mapping of class levels, or its own habituation)
        return max(own, level_of(HAB_LOUD_KEY))

    def extra_of(stim: str) -> float:
        """The sensitisation on this stimulus's defensive response: the brain's (per class)
        or the own general level plus half of what scares left in its own sense."""
        if sens is not None:
            return max(0.0, float(sens.get(stim, 0.0)))
        return general + 0.5 * sens_by.get(sense_of(stim), 0.0)

    def present(
        stim: str,
        raw: float,
        response: float,
        *,
        reflex: str | None,
        bearing: float | None,
        scare: float = 0.0,
        depress: bool = True,
        track: int = 0,
        floor: bool = False,
    ) -> float:
        """Record one processed onset for the brain; returns the fright it carried."""
        h = level_of(stim)
        fright_now = S.clamp(scare, 0.0, 1.0) * h * h
        presented.append(
            Presented(
                stim,
                round(S.clamp(raw, 0.0, 1.0), 4),
                round(response, 4),
                round(fright_now, 4),
                reflex,
                bearing,
                depress and stim not in NO_HABITUATION,
                track,
                floor,
            )
        )
        return fright_now

    def fire(
        new: Active,
        stim: str,
        raw: float,
        response: float,
        *,
        depress: bool = True,
        scare: float = 0.0,
        bearing: float | None = None,
        track: int = 0,
        source_track: int = 0,
        floor: bool = False,
    ) -> None:
        """Record a reflex event: habituate its class, let a scare sensitise (a habituated
        stimulus scares little: × gain²; the level saturates), dishabituate the others.
        ``source_track``: the eye's track of the same object when ``track`` is 0 (a loom-watch
        report has no identity of its own): telemetry for the reflex's record only."""
        nonlocal active, general, arousal_kick, fear_kick, stop_until
        reflex_track = track or source_track
        if reflex_track:
            new = replace(new, track=reflex_track)
        h = level_of(stim)
        gain_now = h * (1.0 + extra_of(stim))
        if new.reflex in ("startle", "freeze") and walking_now:
            # caught walking: the legs stop and STAY stopped long enough to be seen as a stop
            stop_until = max(stop_until, new.t0 + STARTLE_STOP_MS)
        fright = present(
            stim,
            raw,
            response,
            reflex=new.reflex,
            bearing=bearing,
            scare=scare,
            depress=depress,
            track=track,
            floor=floor,
        )
        if own_hab and depress and stim not in NO_HABITUATION:
            hab_own[stim] = max(
                params.hab_floor, h * (1.0 - params.hab_rate * S.clamp(raw, 0.0, 1.0))
            )
        if fright > 0.0:
            if own_sens:
                general = min(1.5, general + params.sens_gain * fright * (1.0 - general / 1.5))
                key = sense_of(stim)
                sens_by[key] = min(1.5, sens_by.get(key, 0.0) + params.sens_gain * fright)
            if own_hab:
                for other in list(hab_own):  # a real scare dishabituates the rest
                    if other != stim:
                        hab_own[other] = hab_own[other] + 0.25 * (1.0 - hab_own[other]) * fright
            fear_kick += 0.4 * fright * (1.2 - params.boldness)
        arousal_kick += 0.08 + 0.3 * response
        if _outranks(new.reflex, active, t):
            active = new
        side = bearing if bearing is not None else new.bearing
        events.append(
            ReflexEvent(
                t,
                new.reflex,
                stim,
                round(response, 3),
                round(gain_now, 3),
                new.zh or _zh(new.reflex, stim, side, posture=posture, escape=new.escape),
                new.parts,
                reflex_track,
            )
        )

    def held_still() -> bool:
        """(fid:fix) The head is being held still now: the brain's act (freeze, startle,
        still) or a holding reflex before its release."""
        if still_act:
            return True
        if (
            active is not None
            and active.reflex == "lean_in"
            and active.t0 <= t < active.t0 + active.rise + active.hold
        ):
            return True  # (being stroked: the hand it sees moving is not a new look)
        if active is None or active.reflex not in HOLDS or active.reflex == "lifted":
            return False
        # (a hold letting go may be taken over by a look — a fast train's claps end a startle
        # before it becomes a freeze: Miller's rule — and then the head does turn)
        return t >= active.t0 and not _outranks("orient", active, t)

    def look(
        new: Active | None,
        stim: str,
        raw: float,
        response: float,
        *,
        bearing: float | None,
        track: int = 0,
        source_track: int = 0,
    ) -> None:
        """(fid:fix) An orienting onset: a named look that moves the head (and gets a row), or —
        the head held still, the head already on it, the same look again within
        ORIENT_MERGE_MS — a covert one: heard/seen (habituation, arousal), nothing moves."""
        nonlocal arousal_kick, last_look
        covert = new is None or held_still() or (engaged and stim in ENGAGED_LOOKS)
        if new is not None and not covert:
            where = yaw_now + new.pose.hy * YAW_PER_CLIP  # its direction in the odometry frame
            same = (
                stim == last_look[2]  # (a sound with no direction: the same sound again)
                if new.bearing is None
                else abs(S.wrap(where - last_look[1])) < ORIENT_MERGE_RAD
            )
            if t - last_look[0] < ORIENT_MERGE_MS and same:
                covert = True  # the same look again: no second row
            else:
                last_look = (t, where, stim)
        if covert:
            present(stim, raw, response, reflex="orient", bearing=bearing, track=track)
            arousal_kick += 0.04 + 0.15 * response
            return
        assert new is not None
        fire(new, stim, raw, response, bearing=bearing, track=track, source_track=source_track)

    jitter = [1.0]  # this onset's own draw (set per onset: RESP_SIGMA)

    def new_onset() -> None:
        jitter[0] = S.clamp(math.exp(RESP_SIGMA * d.g()), 0.5, 2.0)

    def response_of(stim: str, raw: float, gain: float, *, defensive: bool = True) -> float:
        """raw × habituation × sensitisation × arousal × the duck's own gain × this onset's own
        variation. Sensitisation mostly raises the defensive responses (startle, withdrawal);
        orienting a little."""
        h = level_of(stim)
        extra = extra_of(stim) if defensive else 0.3 * extra_of(stim)
        return raw * h * (1.0 + extra) * gain_a * gain * jitter[0]

    def fright_of(r: float) -> float:
        """How much a startle frightens: only its excess over a mild startle."""
        return S.clamp((r - 0.5) / 0.5, 0.0, 1.0)

    def bare_of(stim: str, raw: float, gain: float) -> float:
        """The response WITHOUT sensitisation: what a scare's size is judged by. (Judged by the
        sensitised response, a repeated harmless clap fed its own jumpiness — the first brain's
        measurement on the merged code: the 3rd clap ×1.4 the 1st, 10th < 60 % in only 39/80.)
        The scare is the stimulus's (how loud and sudden, how new), not this flinch's own
        variation: no jitter here."""
        return raw * level_of(stim) * gain_a * gain

    # 1. the body: falls, lifts, bumps, brushes, stumbles (robotd / duck-body proprio: REAL)
    ev = set(frame.events)
    body = frame.body
    if body.fallen or "fell" in ev:
        if not fallen or active is None or active.reflex != "righting":
            new = _simple("righting", t, "body:fall", HeadPose(), 1.0, 250, 1000, 400)
            if not fallen:
                fire(new, "body:fall", 1.0, 1.0, depress=False, scare=0.6)
            elif _outranks("righting", active, t):
                active = new
            fallen = True
        elif active is not None:
            active = replace(active, hold=max(active.hold, t - active.t0 + 1000))
        if "righting_failed" in ev:
            events.append(
                ReflexEvent(
                    t, "righting", "body:righting_failed", 1.0, 1.0, msg("neuro.rx.righting.failed")
                )
            )
    elif fallen:
        # up again only when the trunk IS upright (the IMU): live :8892 (2026-09-28 00:04) the
        # duck ended wedged at 45° after a fall — robotd's 60° verdict cleared and proprio said
        # `righted`, yet it could not stand or walk for minutes. Still down = still righting
        # (head HOME, no walking; the brain calls for help), however the labels read.
        up = body.tilt < UP_TILT
        if not up and active is not None and active.reflex == "righting":
            active = replace(active, hold=max(active.hold, t - active.t0 + 1000))
        elif not up and _outranks("righting", active, t):
            active = _simple("righting", t, "body:fall", HeadPose(), 1.0, 250, 1000, 400)
        if up:
            fallen = False
            helped = "righted_help" in ev
            stim = "body:righted_help" if helped else "body:righted"
            events.append(
                ReflexEvent(
                    t,
                    "righting",
                    stim,
                    0.6,
                    1.0,
                    msg("neuro.rx.righting.helped")
                    if helped
                    else msg("neuro.rx.righting.by_itself"),
                )
            )
    if "lifted" in ev and not lifted:
        lifted = True
        new = _simple("lifted", t, "body:lifted", HeadPose(hp=0.2), 0.35, 400, 2000, 600)
        fire(new, "body:lifted", 0.6, 0.6, scare=0.3)
    if "set_down" in ev:
        lifted = False
        if active is not None and active.reflex == "lifted":
            active = replace(active, hold=max(0, t - active.t0 - active.rise))
    elif lifted and active is not None and active.reflex == "lifted":
        active = replace(active, hold=max(active.hold, t - active.t0 + 1000))
    if "blocked" in ev:
        new = _simple(
            "bump_stop",
            t + 40,
            "body:bump",
            HeadPose(np=-0.3 + 0.05 * d.g(), hp=0.1),
            0.7,
            120,
            int(d.u(500, 900)),
            500,
        )
        fire(new, "body:bump", 0.6, response_of("body:bump", 0.6, 1.0))  # "oops": no scare
    # turning away (spec §5.3 / §8.9 E4, duck-body's body link: stop, then arcs only): after a
    # stall or with a wall right ahead the duck stops for a moment and then walks off in an arc —
    # never nose to the wall for ever (no in-place turns on this body). Proprio's `stumbled` is
    # not used here or for the brace: in live physics it fires every few seconds of walking
    # (90 in 950 s on :8895, 2026-09-27) — the body link already stops the legs on it.
    avoid = state.avoid
    if commit and avoid[1] != 0 and avoid[0] != commit:
        avoid = (commit, avoid[1], avoid[2], avoid[3])  # the escape's side, not the one it began on

    def turn_away(side: int) -> None:
        """Stop, then arc away to ``side`` (unless it is already turning away)."""
        nonlocal avoid
        if commit:
            side = commit
        if avoid[1] == 0:
            avoid = (1 if side >= 0 else -1, 1, t, yaw_now)

    if "blocked" in ev:
        kick = body.gyro[2]
        if abs(kick) > 0.3:
            side = -1 if kick > 0.0 else 1  # the thing is where the heading was kicked from
        else:
            side = open_side  # the side it saw open, else its own side
        turn_away(side)
    swerve = state.swerve if state.swerve[1] > t else (0.0, 0)
    # proprio's "brushed" swerves only when something backs it (a contact felt now, the ToF
    # close): under continuous steering the label alone is mostly noise (live 2026-09-28: 67 of
    # 72 away from any wall; 57 in 3 min turned the duck at random)
    tof_near = (
        frame.tof.present
        and min(
            v
            for v in (frame.tof.left_mm, frame.tof.ahead_mm, frame.tof.right_mm, 9999)
            if v is not None
        )
        < 300
    )
    backed = frame.touch.site is not None or tof_near
    if "brushed" in ev and "brushed" not in refractory and backed:
        away = -1.0 if body.gyro[2] > 0.0 else 1.0  # kicked to its left: the thing is on its left
        swerve = (0.5 * away, t + int(d.u(450, 750)))
        events.append(ReflexEvent(t, "veer", "body:brushed", 0.5, 1.0, _zh("veer", "body:brushed")))
        present("body:brushed", 0.5, 0.5, reflex="veer", bearing=None, depress=False)
        refractory["brushed"] = t + 600
    if walking_now and (body.tilt > 0.30 or max(abs(body.gyro[0]), abs(body.gyro[1])) > 3.0):
        if "stabilise:tilt" not in refractory:  # a real jolt of the trunk (IMU), not a label
            new = _simple(
                "stabilise", t + 30, "body:tilt", HeadPose(), 0.7, 150, int(d.u(400, 800)), 300
            )
            fire(new, "body:tilt", 0.5, 0.5, depress=False, scare=0.1)
            refractory["stabilise:tilt"] = t + 1500

    # 2. ToF (REAL when fitted): drop, touch, veer
    tof = frame.tof
    vx_cap: float | None = None
    vyaw_bias = swerve[0] if swerve[1] > t else 0.0
    veer_side = state.veer_side
    veer_gaze: Gaze | None = None
    if tof.present and tof.age_ms <= 500:
        if tof.drop_zones >= 4 and (walking_now or standing) and "drop" not in refractory:
            new = _simple(
                "drop_stop",
                t + _latency(60.0, params, arousal, d),
                "tof:drop",
                HeadPose(np=0.6, hp=-0.8),
                0.8,
                250,
                int(d.u(1000, 1800)),
                500,
            )
            fire(new, "tof:drop", 0.8, 0.8, depress=False, scare=0.2)
            refractory["drop"] = t + 2500
            # then away from the edge in an arc (it cannot turn in place); toward the side the
            # ToF sees more floor on, else its innate side
            left_mm = tof.left_mm if tof.left_mm is not None else 9999
            right_mm = tof.right_mm if tof.right_mm is not None else 9999
            side = (1 if left_mm > right_mm else -1) if left_mm != right_mm else open_side
            turn_away(side)
        if tof.touch and walking_now and "tof_touch" not in refractory:
            new = _simple(
                "bump_stop",
                t + 40,
                "tof:touch",
                HeadPose(np=-0.3, hp=0.1),
                0.7,
                120,
                int(d.u(500, 900)),
                500,
            )
            fire(new, "tof:touch", 0.5, 0.5, depress=False)
            refractory["tof_touch"] = t + 1500
        ahead = tof.ahead_mm
        left = tof.left_mm if tof.left_mm is not None else 9999
        right = tof.right_mm if tof.right_mm is not None else 9999
        if walking_now or standing:
            push = 0.0
            if ahead is not None and ahead <= 700:
                push = S.clamp((700 - ahead) / 400.0, 0.0, 1.0)
                vx_cap = 0.40
                # the freer side by the ToF; equal ranges (a wall square ahead) is no difference —
                # then the side the turn away will take (2026-10-04: ``left > right`` of two equal
                # ranges said right, the turn away said left: the veer and the arc cancelled)
                freer = (1 if left > right else -1) if abs(left - right) > 60 else open_side
                if veer_side == 0 or (freer != veer_side and abs(left - right) > 150):
                    veer_side = freer
                if ahead <= 300:  # a wall right ahead: stop, then arc away
                    turn_away(veer_side if abs(left - right) > 60 else open_side)
            side_push = 0.0
            if left <= 400:
                side_push -= S.clamp((400 - left) / 300.0, 0.0, 1.0)
            if right <= 400:
                side_push += S.clamp((400 - right) / 300.0, 0.0, 1.0)
            vyaw_bias = S.clamp(
                vyaw_bias + params.veer_gain * (0.9 * push * veer_side + 0.6 * side_push),
                -0.9,
                0.9,
            )
            if push > 0.0:
                veer_gaze = Gaze(bearing=0.35 * veer_side, elevation=0.0, gain=0.3, speed=0.4)
            else:
                veer_side = 0
    turning_away = False
    tof_close = tof.present and tof.age_ms <= 500 and (tof.ahead_mm or 9999) <= 450
    if avoid[1] == 1:
        # the stop: until the legs have stopped (the policy out of its walk, the twist at 0)
        legs_still = body.posture != "walk" and abs(body.vx) < 0.05
        if legs_still and t - avoid[2] >= TURN_STOP_MIN_MS:
            avoid = (avoid[0], 2, t, yaw_now)
    elif avoid[1] == 2:
        # the arc: until it has turned as far as it needs, or it sees the way clear (and has
        # turned a little); a long cap only for safety
        turned = abs(S.wrap(yaw_now - avoid[3]))
        long_enough = t - avoid[2] >= TURN_ARC_MIN_MS
        clear = way_clear and not tof_close and turned >= TURN_CLEAR_RAD
        if (long_enough and (turned >= need or clear)) or t - avoid[2] >= TURN_ARC_MAX_MS:
            avoid = (0, 0, -1, 0.0)
        elif free_side in (-1, 1) and free_side != avoid[0] and turned < TURN_FLIP_RAD:
            # (2026-10-04, headless) the arc began before the wall memory knew which way is open
            # (the first tick of a wall in view: its innate side) and the memory now says the
            # other way — with no turn made yet (a wall close ahead holds the legs) the arc goes
            # the way the memory says, not against the boundary's bend forever
            avoid = (free_side, 2, avoid[2], avoid[3])
    if avoid[1] == 1:
        vx_cap = 0.0  # the stop
    elif avoid[1] == 2:  # the arc away: a bias this strong makes the rhythm arc (loco_step)
        turning_away = True
        vx_cap = 0.40
        vyaw_bias = float(avoid[0]) * TURN_AWAY_VYAW

    # 3. touch (MODEL: where something really touches the shell — a hand, or a wall it rubs).
    # A contact that flickers off and on (physics contacts do, ~every 0.2 s along a wall: live
    # :8892, 2026-09-27) is ONE touch: a site counts as a new onset only after it has been free
    # for TOUCH_GAP_MS — else the duck withdraws five times a second and cannot walk away.
    site = frame.touch.site
    touch_last = dict(state.touch_last)
    fresh_touch = site is not None and t - touch_last.get(site, -TOUCH_GAP_MS - 1) > TOUCH_GAP_MS
    if site is not None:
        touch_last[site] = t
    touch_last = {k: v for k, v in touch_last.items() if t - v <= TOUCH_GAP_MS}
    # Reafference: a contact that comes with its own walking is its own doing — it walked into
    # something, not a hand touching it: the beak while stepping (nose first into a thing), a
    # side while stepping AND its body just felt the brush or the stall (proprio). A poke on
    # its side while it walks, with no such body event, is still a touch.
    bump_last = t if ("blocked" in ev or "brushed" in ev) else state.bump_last
    stepping = walking_now or t - state.walk_last <= REAFFERENT_MS
    reafferent = stepping and (
        site == "beak" or (site in ("left", "right") and t - bump_last <= REAFFERENT_MS)
    )
    # (owner 2026-10-04) a gentle touch it walked into: while it steps or soon after a stall, no
    # hand in view, nothing sliding on it, not firm — the thing is where it was going
    own_contact = (
        not reafferent
        and site is not None
        and (stepping or t - bump_last <= OWN_CONTACT_MS)
        and hand_cue < HAND_CUE
        and not is_stroking(frame.touch.stroke)
        and frame.touch.pressure < OWN_CONTACT_PRESSURE
    )
    if (
        site is not None
        and site != state.prev_touch
        and fresh_touch
        and (reafferent or own_contact)
    ):
        if site in ("beak", "head"):  # nose into something: stop, then arc away (like a stall)
            turn_away(open_side)
        elif site in ("left", "right"):  # its side along something: a short swerve away
            away = -1.0 if site == "left" else 1.0
            swerve = (0.5 * away, t + int(d.u(450, 750)))
            if not turning_away:
                vyaw_bias = S.clamp(vyaw_bias + swerve[0], -0.9, 0.9)
        else:  # felt on the body, unplaced: it bends away to the side it saw open
            swerve = (0.5 * open_side, t + int(d.u(450, 750)))
            if not turning_away:
                vyaw_bias = S.clamp(vyaw_bias + swerve[0], -0.9, 0.9)
        events.append(ReflexEvent(t, "veer", "body:brushed", 0.5, 1.0, _zh("veer", "body:brushed")))
        present("body:brushed", 0.5, 0.5, reflex="veer", bearing=None, depress=False)
        if own_contact:
            bump_last = t  # (its own contact goes on counting as the body's: the next one too)
    elif site is not None and site != state.prev_touch and fresh_touch:
        new_onset()
        stim = f"touch:{site}"
        p = S.clamp(frame.touch.pressure, 0.0, 1.0)
        stroking = is_stroking(frame.touch.stroke)
        if stroking or expect_touch >= 0.5:
            raw = 0.3 + 0.4 * p
            lean_to = _lean_in(t, stim, site, raw, params, arousal, d, head=head, posture=posture)
            # (fid:fix) said as it feels: stroked (a sliding touch) or only touched — a push the
            # touch sense felt is not 「被摸」 (the review: a push said 「右邊被摸，靠過去」)
            where = _SITE_ZH.get(site, "")
            felt = (
                msg("neuro.rx.lean_in.felt.stroked")
                if stroking
                else msg("neuro.rx.lean_in.felt.touched")
            )
            lean_to = replace(
                lean_to,
                zh=msg("neuro.rx.lean_in.felt_at", site=where, felt=felt)
                if where
                else msg("neuro.rx.lean_in.plain"),
            )
            fire(lean_to, stim, raw * 0.5, response_of(stim, raw, 1.0))
        else:
            raw = S.clamp(0.4 + 0.6 * p, 0.0, 1.0)
            mood = (1.0 + 0.5 * (0.5 - S.clamp(comfort, 0.0, 1.0))) * (1.0 + fear)
            r = response_of(stim, raw, 1.0) * mood
            if r >= params.withdraw_threshold and level_of(stim) >= HABITUATED_TOUCH:
                fire(
                    _withdraw(
                        t,
                        stim,
                        site,
                        min(1.0, r),
                        standing,
                        params,
                        arousal,
                        d,
                        lean_side,
                        open_side if free_side in (-1, 1) else 0,
                        walking=walking_now,
                        head=head,
                        posture=posture,
                    ),
                    stim,
                    raw,
                    r,
                    scare=0.6 * min(1.0, bare_of(stim, raw, 1.0)),
                )
            elif r >= params.withdraw_threshold:  # a touch it knows well: only a look at it
                toward = TOUCH_BEARING.get(site, float(params.side))  # back/tail: over a shoulder
                look(
                    _orient(
                        t,
                        stim,
                        min(1.0, r),
                        toward,
                        -0.1,
                        params,
                        arousal,
                        d,
                        head=head,
                        posture=posture,
                        cock=lean_side,
                    ),
                    stim,
                    raw,
                    r,
                    bearing=toward,
                )
            else:
                present(stim, raw, r, reflex=None, bearing=None)
    elif site is not None and active is not None and active.reflex == "lean_in":
        active = replace(active, hold=max(active.hold, t - active.t0 - active.rise + 600))

    # 4. sound (MODEL: the mono microphone; a bearing only if the frame has one)
    snd = frame.sound
    unnamed = state.unnamed
    if unnamed is not None and t - unnamed[0] > SOUND_NAME_WAIT_MS:
        unnamed = None  # nobody named it in time: it stays what it was
    if snd.onset >= 0.08:
        named = snd.kind is not None
        kind = snd.kind or "noise"
        stim = f"sound:{kind}"
        if stim not in refractory:
            new_onset()
            loud = S.clamp((snd.level_db - 45.0) / 45.0, 0.0, 1.0)
            raw = S.clamp(0.5 * snd.onset + 0.5 * loud, 0.0, 1.0)
            k = SOUND_STARTLE.get(kind, 0.8)
            if not named:
                quiet_db, loud_db = UNNAMED_STARTLE_DB
                k *= S.clamp((snd.level_db - quiet_db) / (loud_db - quiet_db), 0.0, 1.0)
            r_startle = response_of(stim, raw, params.startle_gain * k)
            r_orient = response_of(
                stim, raw, params.orient_gain * SOUND_ORIENT.get(kind, 1.0), defensive=False
            )
            floored = _startle_floor(raw, k, floor_level_of(stim))
            if floored:
                r_startle = max(r_startle, params.startle_threshold)
            unnamed = None if named or r_startle >= params.startle_threshold else (t, raw)
            if r_startle >= params.startle_threshold:
                fire(
                    _startle(
                        t,
                        stim,
                        min(1.0, r_startle),
                        snd.bearing,
                        params,
                        arousal,
                        d,
                        fear=fear,
                        yaw0=yaw_now,
                        lean=lean_side,
                        head=head,
                        posture=posture,
                    ),
                    stim,
                    raw,
                    r_startle,
                    scare=fright_of(bare_of(stim, raw, params.startle_gain * k)),
                    bearing=snd.bearing,
                    floor=floored,
                )
            elif r_orient >= params.orient_threshold:
                look(
                    _orient(
                        t,
                        stim,
                        min(1.0, r_orient),
                        snd.bearing,
                        0.1,
                        params,
                        arousal,
                        d,
                        yaw0=yaw_now,
                        cock=lean_side,
                        head=head,
                        posture=posture,
                    ),
                    stim,
                    raw,
                    r_orient,
                    bearing=snd.bearing,
                )
            else:
                present(stim, raw, r_orient, reflex=None, bearing=snd.bearing)
            refractory[stim] = t + SOUND_REFRACTORY_MS
    elif snd.kind is not None and unnamed is not None:
        # duck-ear names the sound the last onset reported (the onset already got its look): the
        # class decides if it startles — a door or a noise does, its own name never does
        stim = f"sound:{snd.kind}"
        raw = unnamed[1]
        unnamed = None  # (once per onset: no refractory check, the onset's own was noise's)
        new_onset()
        k = SOUND_STARTLE.get(snd.kind, 0.8)
        r_startle = response_of(stim, raw, params.startle_gain * k)
        floored = _startle_floor(raw, k, floor_level_of(stim))
        if floored:
            r_startle = max(r_startle, params.startle_threshold)
        if r_startle >= params.startle_threshold:
            fire(
                _startle(
                    t,
                    stim,
                    min(1.0, r_startle),
                    snd.bearing,
                    params,
                    arousal,
                    d,
                    fear=fear,
                    yaw0=yaw_now,
                    lean=lean_side,
                    head=head,
                    posture=posture,
                ),
                stim,
                raw,
                r_startle,
                scare=fright_of(bare_of(stim, raw, params.startle_gain * k)),
                bearing=snd.bearing,
                floor=floored,
            )
            refractory[stim] = t + SOUND_REFRACTORY_MS

    # 5. vision (camera: eye / MODEL for motion, looming, person, hand)
    cam = frame.camera
    seen: list[tuple[int, float, float]] = []
    previous = {
        track: (m, lo) for track, m, lo, seen_t in state.prev_seen if t - seen_t <= SEEN_KEEP_MS
    }
    best: tuple[float, Sighting, str, str] | None = None
    in_view = {sight.track for sight in cam.sightings}
    for sight in cam.sightings:
        seen.append((sight.track, sight.motion, sight.looming))
        before = previous.get(sight.track) if sight.track else None
        person = sight.kind == "person"
        if sight.looming >= (PERSON_LOOM if person else 1.2):
            # something coming at the face; a person it knows walking up is no threat unless
            # they come fast (a big familiar shape: half the fright)
            raw = S.clamp((sight.looming - 1.2) / 2.5 + 0.3 + sight.size, 0.0, 1.0)
            if person:
                raw *= 0.5
            candidate = (raw, sight, "visual:looming", "startle")
        elif sight.motion >= 0.3 and (before is None or before[0] < 0.15):
            stim = VISUAL_CLASS.get(sight.kind, "visual:motion")
            raw = S.clamp(sight.motion * S.clamp(sight.size / 0.2, 0.3, 1.0), 0.0, 1.0)
            candidate = (raw, sight, stim, "orient")
        elif before is None and sight.track and sight.kind in ("thing", "toy", "person", "hand"):
            candidate = (0.4, sight, VISUAL_CLASS.get(sight.kind, "visual:motion"), "orient")
        else:
            continue
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is not None:
        raw, sight, stim, kind = best
        if stim not in refractory:
            new_onset()
            if kind == "startle":
                r = response_of(stim, raw, params.startle_gain)
                if r >= params.startle_threshold * 0.8:
                    fire(
                        _startle(
                            t,
                            stim,
                            min(1.0, r),
                            sight.bearing,
                            params,
                            arousal,
                            d,
                            fear=fear,
                            yaw0=yaw_now,
                            lean=lean_side,
                            head=head,
                            posture=posture,
                        ),
                        stim,
                        raw,
                        r,
                        scare=fright_of(bare_of(stim, raw, params.startle_gain)) + 0.2,
                        bearing=sight.bearing,
                        track=sight.track,
                        source_track=sight.source_track,
                    )
                else:
                    look(
                        _orient(
                            t,
                            stim,
                            min(1.0, r),
                            sight.bearing,
                            sight.elevation,
                            params,
                            arousal,
                            d,
                            yaw0=yaw_now,
                            cock=lean_side,
                            head=head,
                            posture=posture,
                        ),
                        stim,
                        raw,
                        r,
                        bearing=sight.bearing,
                        track=sight.track,
                        source_track=sight.source_track,
                    )
            else:
                r = response_of(stim, raw, params.orient_gain, defensive=False)
                if r >= params.orient_threshold:
                    look(
                        _orient(
                            t,
                            stim,
                            min(1.0, r),
                            sight.bearing,
                            sight.elevation,
                            params,
                            arousal,
                            d,
                            yaw0=yaw_now,
                            cock=lean_side,
                            head=head,
                            posture=posture,
                        ),
                        stim,
                        raw,
                        r,
                        bearing=sight.bearing,
                        track=sight.track,
                        source_track=sight.source_track,
                    )
                else:
                    present(stim, raw, r, reflex=None, bearing=sight.bearing, track=sight.track)
            refractory[stim] = t + 400
    light = cam.light if cam.usable else state.prev_light
    if cam.usable and state.prev_light and light != state.prev_light:
        dark = ("dark", "black")
        if light in dark and state.prev_light not in dark:
            stim = "light:off"
            r = response_of(stim, 0.5, params.startle_gain)
            new = _freeze(
                t + _latency(120.0, params, arousal, d),
                stim,
                min(1.0, r),
                None,
                params,
                d,
                fear=fear,
                yaw0=yaw_now,
                head=head,
            )
            fire(new, stim, 0.5, r, scare=0.2 * min(1.0, bare_of(stim, 0.5, params.startle_gain)))
        elif light not in dark and state.prev_light in dark:
            stim = "light:on"
            r = response_of(stim, 0.5, params.orient_gain, defensive=False)
            lit = _orient(
                t,
                stim,
                min(1.0, r),
                0.0,
                0.5,
                params,
                arousal,
                d,
                yaw0=yaw_now,
                cock=lean_side,
                head=head,
                posture=posture,
            )
            look(lit, stim, 0.5, r, bearing=0.0)

    # 6. what the active reflex asks for now
    out_head: HeadPose | None = None
    blend = 0.0
    gaze: Gaze | None = veer_gaze
    halt = False
    interrupt = False
    escape: tuple[float, float] | None = None
    name: str | None = None
    stim_out: str | None = None
    track_out = 0
    if active is not None and t >= active.t0:
        local = t - active.t0
        if local < active.rise:
            w = S.ease(local / max(1, active.rise), "snap")
        elif local < active.rise + active.hold or active.then == "freeze":
            w = 1.0  # (fid:fix: a startle that goes on into a freeze keeps its pose to the end)
        else:
            w = 1.0 - S.min_jerk((local - active.rise - active.hold) / max(1, active.release))
        name = active.reflex
        stim_out = active.stimulus
        track_out = active.track
        k_pose, k_yaw = posture_scale(frame.body.posture, active.reflex)
        pose_now = active.pose
        if active.dip is not None and local < active.dip_ms:
            pose_now = active.dip  # (fid:fix) a startle with no room above: down first …
        elif active.dip is not None and local < active.dip_ms + SNAP_UP_MS:
            u = S.ease((local - active.dip_ms) / SNAP_UP_MS, "snap")  # … then snapped back up
            pose_now = _mix_pose(active.dip, active.pose, u)
        if active.start is not None and local < active.rise:
            # (fid:fix) from the pose it takes over from (a startle's), straight into its own
            pose_now = _mix_pose(active.start, pose_now, w)
            w = 1.0
        out_head = _scaled(pose_now, k_pose, k_yaw)
        blend = active.blend * w if max(k_pose, k_yaw) > 0.0 else 0.0
        halt = (
            active.reflex in HALTS
            and local < active.rise + active.hold + active.release // 2
            and not (active.escape is not None and local < active.escape[2] and standing)
        )
        interrupt = active.reflex in INTERRUPTS and local < active.rise + active.hold
        if active.bearing is not None:
            # the source stays where it was in the room: the body's own turn (odometry yaw)
            # since the onset is taken off the remembered bearing (+ a one-eye look's aim)
            bearing_now = S.wrap(active.bearing - (yaw_now - active.yaw0))
            aimed = bearing_now + active.aim
            if active.reflex == "orient":
                # (fid:fix) the whole planned turn, quickly: the row names it (a weak look used
                # to go 0.6–0.8 of the way, slowly — a named 12° turn came out 8°)
                gaze = Gaze(
                    bearing=aimed,
                    elevation=active.elevation,
                    gain=S.clamp(w, 0.0, 1.0),
                    speed=S.clamp(0.6 + 0.4 * active.strength, 0.6, 1.0),
                )
            elif active.reflex == "freeze":
                gaze = Gaze(bearing=bearing_now, elevation=0.1, gain=w, speed=0.6)
            if active.reflex in ("orient", "freeze") and out_head is not None:
                # the overlay's own yaw points where the saccade points (up to the validated
                # ±1.0 rad), so its lift never pulls the turned head back to the middle
                out_head = replace(
                    out_head,
                    hy=S.clamp(aimed / YAW_PER_CLIP, -YAW_CLIP_MAX, YAW_CLIP_MAX) * k_yaw,
                )
            if gaze is not None and frame.body.posture not in ("sit", "stand", "walk"):
                # getting up or sitting down: the eyes may drift toward it, never snap
                gaze = Gaze(gaze.bearing, gaze.elevation, 0.5 * gaze.gain, 0.3 * gaze.speed)
        if active.reflex == "startle" and out_head is not None:
            # the beak snaps open and shut again: the raw actuator, quicker than the head
            if local >= BEAK_SNAP_MS:
                shut = S.clamp((local - BEAK_SNAP_MS) / BEAK_SHUT_MS, 0.0, 1.0)
                out_head = replace(out_head, mouth=out_head.mouth * (1.0 - shut))
        if active.escape is not None and local < active.escape[2] and standing:
            escape = (active.escape[0], active.escape[1])
    if t < stop_until and standing:
        halt = True  # caught walking by a startle: still, long enough to be seen (fid:motor)
    stabilise = 0.7 if walking_now else (0.3 if standing else 0.0)
    if halt:
        vx_cap = 0.0
    recent = (state.events + tuple(events))[-EVENTS_KEPT:]
    if own_sens:
        sens_map = {key: round(extra_of(key), 3) for key in hab_own if extra_of(key) > 0.005}
        if general > 0.005:
            sens_map["*"] = round(general, 3)
    else:
        sens_map = {}
    new_state = ReflexState(
        t_ms=t,
        rng=d.rng,
        hab=tuple(sorted(hab_own.items())),
        sens=general,
        sens_by=tuple(sorted((k, v) for k, v in sens_by.items() if v > 1e-4)),
        active=active,
        lifted=lifted,
        fallen=fallen,
        refractory=tuple(sorted(refractory.items())),
        prev_touch=site,
        touch_last=tuple(sorted(touch_last.items())),
        walk_last=t if walking_now else state.walk_last,
        bump_last=bump_last,
        prev_seen=tuple((track, m, lo, t) for track, m, lo in seen if track)
        + tuple(
            (track, m, lo, seen_t)
            for track, m, lo, seen_t in state.prev_seen
            if t - seen_t <= SEEN_KEEP_MS and track not in in_view
        ),
        prev_light=light,
        veer_side=veer_side,
        swerve=swerve,
        avoid=avoid,
        events=recent,
        count=state.count + len(events),
        stop_until=stop_until,
        last_look=last_look,
        unnamed=unnamed,
    )
    return new_state, ReflexOutput(
        active=name,
        stimulus=stim_out,
        head=out_head,
        head_blend=blend,
        gaze=gaze,
        halt=halt,
        vx_cap=vx_cap,
        vyaw_bias=vyaw_bias,
        interrupt=interrupt,
        escape=escape,
        stabilise=stabilise,
        arousal_kick=S.clamp(arousal_kick, 0.0, 1.0),
        fear_kick=S.clamp(fear_kick, 0.0, 1.0),
        events=tuple(events),
        recent=recent,
        habituation={k: round(v, 3) for k, v in hab_own.items()},
        sensitisation=sens_map,
        presented=tuple(presented),
        hold=name if name in HOLDS else None,
        fallen=fallen,
        lifted=lifted,
        turning_away=turning_away,
        track=track_out,
    )


def initial(seed: int = 0) -> ReflexState:
    """A fresh reflex state for a duck (its own variation stream)."""
    return ReflexState(rng=S.seed_from("neuro.reflex", seed))


__all__ = [
    "Active",
    "HABITUATED_TOUCH",
    "HALTS",
    "HOLDS",
    "INTERRUPTS",
    "NO_HABITUATION",
    "PRIORITY",
    "Presented",
    "REAFFERENT_MS",
    "REFLEX_VERSION",
    "ReflexOutput",
    "ReflexParams",
    "ReflexState",
    "SOUND_REFRACTORY_MS",
    "TOUCH_GAP_MS",
    "TURN_AWAY_VYAW",
    "initial",
    "params_from",
    "params_from_genome",
    "posture_scale",
    "step",
]
