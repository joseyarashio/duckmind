"""The neural duck's brain: ``NeuroBrain(genome, seed).step(t_ms, frame) -> (cmd, telemetry)``.

One tick (designed for 10–50 Hz; the drives, traces and noise are exact for any step and every
event counts once):

1. **perceive** — ongoing levels from the frame: voices, music, the nearest person, the most
   salient thing, the nest, how dark it is, a gentle touch, whether a touch was seen or heard
   coming (``perceive.py``);
2. **reflex arcs** — orienting, startle → freeze, withdrawal / leaning in, the ToF veer, drop and
   bump stops, bracing while walking, righting, being held: neuro:motor's :mod:`reflex` module
   (the one reflex module), run here with this brain's arousal, fear, comfort, habituation and
   sensitisation (``habituation.py``: the 10th clap gets a weaker response; a scare makes it
   jumpier for a while);
3. **appraisal** — what the reflexes mean to the slower systems: drive impulses, after-effect
   traces for selection, place and cue learning, the owner's 3-hard-falls rule
   (``appraisal.py``, ``learn.py``);
4. **drives** — energy, sleep pressure × circadian, curiosity, fear, social need, comfort,
   boredom, the urge to walk, battery hunger, play; arousal and mood (``drives.py``);
5. **selection** — the basal-ganglia-like competition over the whole repertoire with
   commitment and satiation (``select.py``), interrupted by reflexes;
6. **motor intent** — the locomotor rhythm (neuro:motor's :func:`motor.loco_step`: bursts and
   pauses, OU speed drift, meandering arcs) steered by the behaviour's goal; posture; gaze (the
   reflex's saccade, the behaviour's target, idle scanning); liveliness, head bob and
   stabilisation; calls.

Honest senses: the brain reads only the SensorFrame (egocentric sightings, its own odometry);
it never imports :mod:`duckmind.world`. Deterministic: the same genome, seed and frames give the
same outputs; only differences of ``t_ms`` matter (no absolute clock), plus ``day_minute``.

Phase 5 (fixed point) must convert: every float state to SCALE 10_000 integers; ``exp(-dt/τ)``
to per-tick-rate constants (the board's tick is fixed); ``sin`` to duckbody.fixedmath's table;
the splitmix64 streams are already integer; the Irwin–Hall normal needs no log/sqrt.
"""

# NOTE to other writers (neuro:brain continuation agent, 2026-09-27 23:20): brain.py now runs on
# reflex.py + motor.loco_step + signals + appraisal.py (the orchestrator's one design, 22:52);
# reflexes.py, loco.py and noise.py are retired. Please write change requests in
# NOTES-brain-motor.md instead of editing this file directly — two writers clobber each other.

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from duckmind.duckcore import content
from duckmind.i18n import msg
from duckmind.neuro import dock as D
from duckmind.neuro import motor as M
from duckmind.neuro import reflex as R
from duckmind.neuro import select as S
from duckmind.neuro import signals as N
from duckmind.neuro.appraisal import Appraiser
from duckmind.neuro.boundary import (
    GOAL_CLEAR_M,
    GOAL_CLEAR_THING_M,
    NEAR_M,
    SLOW_M,
    STOP_M,
    Boundary,
)
from duckmind.neuro.clutter import CLUTTER_ENABLED, Clutter, ClutterConfig
from duckmind.neuro.contract import (
    APPROACH_VX,
    ARC_VX_MIN,
    BEHAVIOURS,
    NORMAL_VX_MAX,
    REFLEXES,
    BrainTelemetry,
    Candidate,
    Gaze,
    MotorCommand,
    SensorFrame,
    Sighting,
    Target,
    behaviour_zh,
    legal_walk,
    wrap_angle,
)
from duckmind.neuro.contract import REVERSE_VX as C_REVERSE_VX
from duckmind.neuro.detour import FOLLOW_CAP, FOLLOW_GO, Detour
from duckmind.neuro.drives import HUNGER_BATTERY_AT, DriveInputs, Drives
from duckmind.neuro.errand import ERRAND_CLAIM, WALK, Errand
from duckmind.neuro.habituation import Habituation
from duckmind.neuro.lastmap import LastMap, LastMapConfig, mode_of, overrides_from_env
from duckmind.neuro.learn import Learner
from duckmind.neuro.liftcheck import LIFT_DROP_EVENTS, LIFT_SHORT_MS, LiftWitness
from duckmind.neuro.perceive import (
    Perceiver,
    bearing_to,
    lens_point,
    range_from_trunk,
    sighting_odom,
    to_odom,
)
from duckmind.neuro.telemetry import stimulus_zh
from duckmind.neuro.unstick import Escape, Unstick

# While a reflex holds the whole body, the command names the reflex's own program.
HOLD_BEHAVIOUR = {
    "startle": "startle",
    "freeze": "freeze",
    "withdraw": "withdraw",
    "righting": "righting",
    "lifted": "held",
    "drop_stop": "still",
    "bump_stop": "still",
}
TARGETED = {
    "investigate": "thing",
    "nibble": "thing",
    "peck": "thing",
    "watch": "motion",
    "follow": "person",
    "greet": "person",
    "snuggle": "person",
    "nuzzle_tilt": "person",
    "peekaboo": "person",
    "peekaboo_master": "person",
    "nod": "person",
    "call_response": "person",
    "retreat": "threat",
    "flee": "threat",
    "return_dock": "nest",
    "glance_away": "person",
}
GAZE_ELEVATION = {
    "peck": -0.55,
    "nibble": -0.35,
    "forage": -0.4,
    "peek_down": -0.6,
    "investigate": -0.15,
}
EXERTION = {"zoomies": 0.8, "flee": 1.0, "little_dance": 0.5, "happy_sway": 0.3, "body_shake": 0.6}
# Acts whose motor program moves the head itself (a look around, a preen, a peck…): the idle gaze
# cycle does not add its own shifts on top (review 2026-09-28: two saccade systems at once and
# the head never held still)
HEAD_ACTS = frozenset(
    {
        "look_around",
        "survey",
        "preen",
        "peck",
        "nibble",
        "dabble",
        "stretch_yawn",
        "happy_sway",
        "little_dance",
        "not_now",
        "nod",
        "peekaboo",
        "peekaboo_master",
        "self_duet",
        "little_song",
        "snuggle",
        "melt",
        "nuzzle_tilt",
        "moon_gaze",
        "call_out",
        "greet",
        "call_response",
        "tuck_away",
        "wake_ritual",
        "body_shake",
        "bow",
        "lie_low",
        "spin",
        "flip",
        "zoomies",
    }
)
GAZE_MEDIAN_S = 2.0  # the calm, standing gaze cycle's median interval (seated a little longer):
# pet:etho A1 has 1.2 s for a bird's eye; this head needs 0.3–0.5 s per shift (pet:body), so a
# shift and a visible HOLD take about twice as long
ONE_EYE_RAD = 0.12  # at rest it looks with one eye, a little to the side (pet:etho A2)
# How fast it walks, by what it is doing (owner 2026-10-04: 「鴨子各項反應速率應該是有速率差別」).
# The body takes 0 or 0.35–0.55 m/s; the duck uses the band by its state:
#   · an ordinary walk (wander, explore, forage, going home, a person, a thing far off) runs at
#     PACE_SLOW…PACE_FAST at the top, by temperament (the bold and the lively toward the fast end,
#     the lazy toward the slow end), with the rhythm's own drift below it (motor.loco_step);
#   · close to the thing it investigates it goes at the gait's slowest (ARC_VX_MIN), stopping and
#     going by its rhythm; behind the person it follows it speeds up to the top of the band;
#   · a flight in panic (flee, retreat) runs at the top at once and turns away in the tightest arc
#     (PANIC_*), no wandering, no pauses, for as long as the flight lasts.
PACE_SLOW = 0.40
PACE_FAST = 0.45
NEAR_THING_M = 0.3  # closer than this to the thing it investigates: the slowest steps
FOLLOW_CATCH_UP_M = (0.8, 2.0)  # a person this far ahead: from ordinary pace to the top of the band
PANIC_TURN_GAIN = 2.5  # rad/s per rad of the threat's reverse bearing: the full yaw rate while
# it is more than 0.4 rad off facing away …
PANIC_FACING_RAD = 0.25  # … and within this of facing away it runs straight (an arc is ≤ 0.40 m/s)
# A startle whose fright has not gone is a flight (select.Selector.commit_next): the fright still
# up (its fear drive, or the thing still looming) once the reflex holds (the startle and the
# freeze it goes on into) are over, within STARTLE_FLIGHT_S of that (the bolder the duck, the
# more fear it takes).
STARTLE_FLIGHT_S = 6.0
STARTLE_FLIGHT_FEAR = 0.35  # + 0.1 × boldness
STARTLE_FLIGHT_LOOM = 0.3
# A strong scare — a very loud sudden sound close by, a hand rushing at the face: a startle whose
# stimulus is this big (its raw size × the habituation it had left, 0..1; a bang at the head's
# 92–96 dB or a hand at the beak read 0.9 and over, fresh) — is a flight for EVERY temperament, the
# bold included (owner 2026-10-05: 「逃跑應該是驚慌失措，全速逃離，嚇一跳也是」). The bold differ in
# how much it takes: a mild or medium scare they stand their ground to (their fear drive stays under
# the bar above), and they run shorter (the flee act's gain falls with boldness, genome:
# 1.45 − 0.9 × boldness) and settle sooner.
STRONG_SCARE = 0.8
# … and so is the startle the reflex layer's startle floor fired (a sudden loud sound at least
# reflex.STARTLE_FLOOR_RAW big that it is not used to; ``Presented.floor``): the sharp sounds of
# its home habituate the group a little, so a fresh bang reads raw 1.0 × level ~0.78, just under
# the bar above, and the first bang on the page would otherwise be a jump and no flight.
# Being pushed over is a strong scare too: a fall that comes within PUSH_FALL_S of something
# touching it, once it is up again, is a flight (it was startled by the push and the fall; the
# righting is the beat it needs to be seen).
PUSH_FALL_S = 2.5
# The locomotor behaviours' walking style: (top speed m/s, go offset, meander scale). The rhythm
# itself (bursts, pauses, speed drift, curvature noise) is the duck's own (motor.loco_step).
LOCO: dict[str, tuple[float, float, float]] = {
    "wander": (PACE_FAST, 0.0, 1.0),
    "forage": (0.38, -0.12, 1.3),
    "investigate": (PACE_FAST, 0.15, 0.5),
    "follow": (PACE_FAST, 0.2, 0.5),
    "retreat": (NORMAL_VX_MAX, 0.3, 0.4),
    "flee": (NORMAL_VX_MAX, 0.5, 0.3),
    "return_dock": (PACE_FAST, 0.25, 0.4),
    "zoomies": (NORMAL_VX_MAX, 0.45, 1.8),
    "turn_around": (APPROACH_VX, 0.5, 0.0),
}
# How much of a reflex's head throw the body can take in each posture (live physics 2026-09-27:
# a startle snap during a rise tipped the duck over): seated all of it, balancing about half,
# getting up or sitting down none — the head stays where it is.
REFLEX_HEAD_BY_POSTURE = {"sit": 1.0, "stand": 0.55, "walk": 0.55}
# Where a goal-directed walk stops short: in reach of a thing (then nibble or peck), a little
# way from a person (it does not walk into legs), m.
#
# 到了 at a thing means the beak can reach it (machine verification v3 finding ③, 2026-10-02: the
# duck said 到了 with the thing 0.2–0.45 m from its trunk, 5/28 .. 1/21 inside the beak's reach
# + 0.1 m). The distance is the thing's NEAR EDGE from the TRUNK's centre — what the beak's reach
# is measured against — from the duck's own senses: the eye's ground range is to the thing's foot
# on the floor (its near edge, nothing to add for the thing's width) and starts at the lens, which
# sits ~0.08 m ahead of the trunk's centre (perceive.lens_point); a thing it no longer sees is
# the place it believes it is at (odometry, stored the same way). Measured on the official MJCF:
#   BEAK_REACH_M  the beak tip's farthest horizontal distance from the trunk centre over the
#                 neck/head pitch grid (tools/machine_verify/measure_reach.py): 0.0958 → 0.096;
#   STOP_COAST_M  how far the body still goes after the walk command is zero: 75 stops in the v3
#                 L2 traces, 0.026 m median, 0.037 m largest within 0.2 s → 0.03.
# The walk stops at BEAK_REACH_M + STOP_COAST_M, so the beak ends up where it can touch; 到了 is
# said when it has been that close, and stopped, for ARRIVE_HOLD_S (a look that is wrong by one
# noisy sample does not say it).
BEAK_REACH_M = 0.096
STOP_COAST_M = 0.03
ARRIVE_M = BEAK_REACH_M + STOP_COAST_M
ARRIVE_HOLD_S = 0.2
REACH_M = {"investigate": ARRIVE_M, "follow": 0.5}
# Curiosity's back_arc (owner phase 3: reversing only at the nest and as curiosity's back_arc,
# curiosity.prims: vx −0.40, wz ±1.0, 1.0 s ≈ 95 mm back and ~33°): done with a thing right in
# front of it — or pushing into it — it backs off in a small arc before it turns away (review
# 2026-09-28: walked to 0.19 m of the books, then its forward arcs kept sweeping into them)
THING_ACTS = frozenset({"investigate", "nibble", "peck"})
BACK_ARC_MS = 1000
BACK_NEAR_M = 0.4  # the thing this close in front …
BACK_AHEAD_RAD = 0.7  # … and this nearly straight ahead
BACKS_PER_PLACE = 2  # at most this many back arcs before it has moved on (curiosity: 2)
BACK_RESET_M = 0.5
# Cornered (fid:motor; owner 2026-09-28: reversing a small step/arc when cornered is ALLOWED): no
# forward arc has room, so it backs out in a small reverse arc turning toward the way it saw
# open, then walks on — a few times at one place before it gives up there (then, as before, it
# stops, sits and calls its family). Phase-3 calibration from standing (curiosity_calib, n = 10
# per condition, 0 falls, tilt ≤ 5.4°): vx −0.40 with ±1.0 rad/s for 1.0 s ≈ 8–12 cm back and
# 32–36°; 1.5 s ≈ 15–19 cm and 52–59°.
CORNER_BACK_MS = 1300
CORNER_BACKS = 3
CORNER_BACK_ROOM_M = 0.3  # … only where the walls it remembers leave this much room behind
# Orienting, then the body follows (fid:motor; pet:etho A4 "the head leads the turn"): a thing it
# sees (or the thing it watches) far to one side — beyond what a comfortable head turn keeps in
# view — and still interesting after the head has turned: it turns its body toward it, in an arc
# (this body cannot pivot: owner "turns are arcs"). The A100 validated the orient-then-walk
# program (D.orient.stand / .walk: 0/512 falls each; head t50 0.26 s, the body aligned in
# 2.4–3.6 s); it stops as soon as it faces the thing (or after TURN_TOWARD_MAX_MS).
TURN_TOWARD_RAD = 0.8  # |bearing| beyond this, rad
TURN_TOWARD_AFTER_MS = 700  # … still interesting this long after the look began
TURN_TOWARD_DONE_RAD = 0.3  # facing it within this: done
TURN_TOWARD_MAX_MS = 3000
TURN_TOWARD_VX = 0.37
TURN_TOWARD_VYAW = 0.9
PIVOT_MIN_RAD = 0.6  # (2026-10-09) the quick turn pivots while the heading still has at least this far to go; the arc finishes
TURN_TOWARD_ROOM_M = 0.45  # the arc's first sweep must be this clear of walls it has seen
TURN_TOWARD_ACTS = frozenset({"watch", "greet", "call_response", "nod"})
# … the acts in which its eyes pursue the person they are about (the head follows a person who
# walks, as far as the neck turns — then the body follows); a person it saw is believed where it
# went for this long after it left the picture
PURSUE_ACTS = frozenset({"watch", "greet", "call_response", "nod", "follow"})
PURSUE_BELIEF_S = 2.5
# … a person it saw this recently is still where it knows them to be when its name is called
PERSON_KNOWN_S = 12.0
COMES_WHEN_CALLED_SOCIABILITY = 0.6
# (acts during which it stays put whatever it sees: resting, sleeping, grooming, eating)
TURN_TOWARD_NEVER = frozenset(
    {
        "rest_sit",
        "doze",
        "nap",
        "melt",
        "sleep_talk",
        "preen",
        "snuggle",
        "peck",
        "nibble",
        "dabble",
    }
)
STUCK_TAU_S = 60.0
# the acts a new thing does not break into: going home, and the handling or the search of the things
# on the floor it has in hand (the beak down: investigate, nibble, peck, forage — M5, a forage broken
# off by a new thing 3 times in the bold explorer's 111 response bouts took its natural endings to
# 89.2 %, 93.6 % without)
NOVELTY_KEEPS = frozenset({"return_dock", "investigate", "nibble", "peck", "forage"})
# a duck this drowsy (the wakefulness gate is down to 0.3) is not roused by it
NOVELTY_SLEEPY_MAX = 0.6
# a bout that began less than this long ago (a look, a survey: often the duck's own first answer to
# the thing that has just appeared) is not broken off by it
NOVELTY_MIN_BOUT_S = 1.5
# a thing newly seen farther off than this is noted, not turned to (what the sound skills do with the
# props of their scenes: a hovering hand outweighs a new thing far away); the L1 and L2 objects are put
# down 1.2-1.4 m from the duck
NOVELTY_RANGE_M = 1.6
# … and a person who has just called or greeted it (or holds a hand over it, or plays music to it:
# the sound skills' features, there once they are installed), or anything that threatens it, comes first
NOVELTY_YIELDS_TO = ("name", "voice", "person_new", "hand_hover", "beat", *S.URGENT_FEATURES)
NOVELTY_YIELD_AT = 0.2
ERRAND_GO = 0.9  # a walk to a place somebody pointed at: how much of the time it walks (0..1; the rhythm's own `go`)
# Getting unstuck (owner 2026-10-04: pressing into things, stuck in corners; unstick.py): the
# ToF looks over a low box and the camera is blind below ~0.2 m, so what tells it it is pressing
# is its own request against its own progress. These are only the cues that something is there.
LOOSEN_ACTS = (
    "investigate",
    "return_dock",
    "follow",
)  # the walks toward something: the clutter escape may let the pull go
WAYPOINT_ACTS = (
    "return_dock",
    "investigate",
    "wander",
)  # the walks a map route may head (neuro.lastmap)
# the walks a route over the map may take the legs in; every other act (eating, playing, looking, a thing it came to see close by, a flight) is
# not a knot: contacts there are the act, and a route that pulls the duck off a toy it is playing with only makes it come back (measured, 15 of
# 15 "stuck" episodes of the first free real-home runs were with a toy, a pair of slippers or a book)
ROUTE_ACTS = ("wander", "turn_around", "return_dock")
ROUTE_NEAR_THING_M = (
    0.9  # investigate / follow count while the thing or person is farther than this
)
CLUTTER_NEAR_MM = 320  # the ToF this near ahead holds a straight walk (the link's ToF stop): the clutter escape reads it as held
UNSTICK_TOF_MM = 450  # a ToF return this close ahead is "something is there"
UNSTICK_FREE_M = 0.35  # … or the wall memory leaves this little way ahead
CORNER_AT = 3  # the third escape at one place is a corner (unstick.Escape.count)
# Touching a thing it walks past (2026-10-09, the tissue box: a wander burst that starts as a tight arc towards a thing
# standing against its side sweeps it along, burst after burst): after a touch it felt while stepping, a free walk or a
# flight (no other goal, no docking) is biased away from the side the thing is on for TOUCH_AVOID_S. The side is what
# the duck itself knows: a left / right touch says it; a touch it cannot place (the robot has no skin map: "body",
# head, beak) is on the side it was turning towards (its own applied yaw rate, at least TOUCH_TURN_MIN rad/s).
TOUCH_AVOID_S = 10.0
TOUCH_AVOID_BIAS = 0.8  # rad/s of yaw bias away from the touched side (the reflex swerve's size)
TOUCH_TURN_MIN = 0.25
# A LOW EDGE ahead (a curb, a high sill, a step of about 30 mm and more) is not a wall to scrape along: the toes climb it before the
# beak touches it and the trunk pitches over it (measured 2026-10-10, the obstacle course: 60 and 80 mm curbs in a doorway, 61 falls in
# 80 trials). Its face is seen from afar and remembered once the beams pass over it inside ~0.4 m (duckbody.depth, ``low_edge_mm``);
# the walk is stopped at 300 mm by the link's near stop, and the forward arc away from there needs 0.4-0.5 m of turning radius it does
# not have. So the duck backs off in a small reverse arc first (the unstick escape, counted per place like any other) and then turns away.
# ``low_edge_mm`` is the ToF's own height estimate of the face. It reads high by 20-30 mm at 0.75-0.95 m (a zone that grazes the floor
# returns its nearest part): a 25 mm sill reads at most 57 mm (27 % of its frames above 55), a 30 mm one up to 82, a 60 mm curb 55-95;
# 55 mm therefore means a step of about 30 mm and more.
LOW_BLOCK_MM = (55, 130)
LOW_BACK_MM = 340  # the edge is this close ahead (the link's near stop is 300) …
LOW_BACK_GAP_MS = 4000  # … at most one such back-off this often …
LOW_BACK_MS = (
    1300,
    2200,
)  # … a reverse arc this long, then this long turning away (unstick.UnstickConfig.back_ms / turn_ms)
ESCAPE_WAIT_MS = 3000  # an escape waits at most this long for a reflex stop to end
ESCAPE_BACK_SLACK_MS = 1500  # … and its back arc may take this much longer (the link's holds)
BACK_M_PER_S = 0.10  # the reverse arc goes back ~8–12 cm in 1.0 s (phase-3 calibration)
ESCAPE_BACK_TAIL_M = 0.09  # the tail (7 cm behind the trunk's origin) and a hand's breadth
ESCAPE_BACK_MIN_M = 0.04  # a reverse shorter than this is no reverse
GIVE_UP_AT = 2  # the second escape at one place: it gives up what it was walking toward …
GIVE_UP_USE = 0.8  # … (the bout is used up this much: the selector looks for something else)
DOCK_REST_M = 0.45  # it comes home to rest this close beside its nest (not onto it: see step)
LOOK_STOP_MIN_MS = 400  # a stop to look lasts at least this (the legs' tail, a sharp frame) …
LOOK_STOP_MAX_MS = 2500  # … and at most this when no floor line comes
BY_NEST_S = 600.0  # after coming home it counts as at its nest this long (unless it walks off)
STUCK_AT = 3.0  # this many stalls (leaky, τ 60 s) at one spot: it is stuck
CORNERED_STUCK_S = 4.0  # … and every this many seconds cornered counts as one
# how close it comes to a person: a + b × (1 − boldness) × (1 − ½ sociability) — the shy stop
# further off, the clingy come close whatever their shyness (review 2026-09-28)
FOLLOW_REACH_M = (0.35, 0.35)
COO_ACTS = frozenset({"rest_sit", "snuggle", "melt", "nuzzle_tilt", "doze", "preen"})
# Acts that keep the beak busy or the body still: no contact calls during them.
BEAK_BUSY = frozenset(
    {"preen", "peck", "nibble", "dabble", "forage", "freeze", "still", "startle", "withdraw"}
)
# What each act does for the maintenance needs (drives.py: groom, peckish, vigil).
GROOMING = {"preen": 1.0, "body_shake": 0.6, "head_shake": 0.4, "stretch_yawn": 0.5}
PECKING = {"peck": 1.0, "nibble": 0.8, "dabble": 1.0, "forage": 0.5}
SCANNING = {"look_around": 1.0, "survey": 1.0, "glance": 0.6, "perk_tilt": 0.5, "watch": 0.4}
POSTURE_TIRED_TAU_S = 90.0  # how long a sit-down or a rise makes the next change costlier
PAUSE_SCANNING = 0.6  # a walking act's pause scans (drives.vigil) at this rate
# The need each act satisfies (drives.py): the act's bout is used up as the need is (its
# satiation follows the need's fall — a consequence, not a clock; review 2026-09-28)
NEED_OF = {
    "preen": "groom",
    "body_shake": "groom",
    "peck": "peckish",
    "nibble": "peckish",
    "forage": "peckish",
    "dabble": "peckish",
    "look_around": "vigil",
    "watch": "vigil",
    "wander": "restless",
}
NEED_FLOOR = 0.15  # a bout begun on less need than this is used up as if by this much
NEED_SATIATES = 0.35  # the share of the bout's satiation its need's fall gives (the rest: time)
MOMENT_BUSY_TAU_S = 2.0  # a moment playing holds off the next (a leaky trace, ~2.4 s over 0.3)
# (fid:fix) a moment that looks up or down needs this much room for its nose from where the head
# is (rad: 12°, so that ≥ 10° is seen); the acts that keep the head low have none for a look down
MOMENT_ROOM_RAD = 0.21
HEAD_DOWN_ACTS = frozenset({"forage", "peck", "preen", "nibble", "dabble", "snuggle", "melt"})
# (dock:build 2026-09-28) backing into its dock and sitting in it, its body meets the dock's face
# and guides: those contacts are its own doing (reafference) — no bump reflex, no wall marked,
# no bad place learned; the docking program reads them itself (blocked = arrived)
DOCK_CONTACT_EVENTS = frozenset({"blocked", "brushed", "stumbled"})
DOCK_CAPTURE_MS = 100  # a camera pass is captured ~70–140 ms before it is delivered
# Looking for its nest (return_dock, the nest neither seen nor remembered): every SEEK_PERIOD_S it
# stops and sweeps its head across its whole reach — left, right, ahead (SEEK_SWEEP: bearing,
# until s into the scan) — then walks on in a search arc. The lens sees ±19° about where the head
# points: the sweep covers ±73° (the head's ±0.95 rad), the arcs the rest of the room.
SEEK_PERIOD_S = 9.0
SEEK_SWEEP = ((0.95, 1.1), (-0.95, 2.4), (0.0, 3.1))
# … between the sweeps it turns round in a tight arc (its innate side; ~150° by the next sweep)
# three times, then walks on with its own rhythm for one period to look from somewhere else
# (live 2026-09-28, the apartment: started facing away from its dock with no memory of it, the
# rhythm's loose search arcs took it down the corridor and it never saw the tag)
SEEK_ARC = (0.37, 0.9)
SEEK_CYCLES = 4
SEEK_ARC_ROOM_M = 0.5  # a search arc sweeps ~0.45 m: it needs this much clear on its side
# (sill-circling 2026-10-28) the search ends: a home with no nest (live `--no-nest`, learned dock null) made going home an
# endless round of these tight arcs (25 of 65 live loop captures; 53 % of its walking home was in them). After one whole round
# of looking (SEEK_GIVE_UP_S of searching, its own clock, a fall or a stop in between does not start it again) without the
# nest in view it lets going home go for NEST_REST_S[0] — the sleepy pull home and the low pack's pull are not felt as a
# walk home then, so it naps, rests or wanders where it is — and each search after that finds nothing doubles the rest, up
# to NEST_REST_S[1]; the nest seen (its tag) ends the rest at once and the next search starts afresh
SEEK_GIVE_UP_S = SEEK_PERIOD_S * SEEK_CYCLES
NEST_REST_S = (120.0, 1800.0)
# (2026-10-24, owner: "B") a command's search for a person turns on the spot with robotd's quick turn when the body has it
# (``can_pivot``): a step of this much, then the head sweeps and the eyes look this long, then the next step
SEARCH_PIVOT_RAD = 1.0
# the spin trick (neuro.tricks): the body pivots on the spot until its own odometry has turned this far
# (a little under a full turn: the quick turn's hand-back carries it the rest of the way)
SPIN_RAD = 5.9
SEARCH_LOOK_MS = 2000
DOCK_EYES_M = 3.2  # (dock:build) walking home, it looks at where it believes its dock is this near
# (dock-home 2026-10-17) Walking home by a dock frame carried farther than the program trusts to steer by (remembered_max_m), the walk
# and the eyes still go by that frame (the odometry's error stays ~0.3 m, p90 ~1 m, over 100 m of wandering, headless): to the lane's
# far point, where the tag is read square on (it is read from within 60° of its face: from the side, along the wall, it cannot be).
# Within NEST_AT_LANE_M of that point with no nest in view it stops to look for its nest (the head sweeps and the search arcs, SEEK_*)
# instead of standing there. A tag read (the program's own `remember_m` rule) gives the frame back; the legs are still taken only by a
# tag read or a frame it trusts
NEST_AT_LANE_M = 0.6
NEST_LEAVE_M = 1.2  # … and goes on looking until the search has taken it this far from that point (the arcs turn it round about it)
TELEMETRY_SLOW_S = 1.0
HOME_EVERY_S = 10.0  # (MAP-EXPLORE) what it knows of its home is worked out again this often (a scan of its whole map)
NAME_RESIST = 0.5  # its own name resists habituation (a conditioned cue)
INVESTIGATE_USE = 1.5  # an investigation is used up as the thing's novelty is (× the drop)
ABSENT_S = 4.0  # looking where something should be and not seeing it: gone in about this long
# … standing still: while the legs step the frames are smeared and the eye misses a still thing most
# of the time (neuro_senses: P_SHARP_STEPPING), so not seeing it then says nothing about it
# … but a thing nearer than this is not expected to be seen: the camera sees no floor nearer than
# ~0.2 m (the last stretch of a walk up to a thing is blind, by its own odometry)
BLIND_NEAR_M = 0.3
# Going for a thing (2026-10-05, legs; LEGS-ACCEPTANCE-2026-10-04): the thing is far enough that the walk
# is a walk, and it is more or less ahead — it goes for it: up toward the top of the gait's band (a
# bolder duck further) and with only short stops. Nearer than this it goes at the slow steps of the
# last stretch as before.
PURSUE_FAR_M = 0.7
# … the last stretch (nearer than that, still on its way to the beak's reach and more or less ahead): the
# slow steps, but not a stop every few of them
PURSUE_NEAR_ALIGN_RAD = 1.3
PURSUE_NEAR = 0.45
TRAVEL_FREE_M = 1.0
TRAVEL_HALF_M = 1.0
# The thing in view that is the one it is going to: within this of where it believes it is (the eye's
# own error: a hand's breadth, a bit more at a distance)
FOCUS_MATCH_M = 0.3
FOCUS_FAR_PER_M = 0.12
PURSUE_ALIGN_RAD = 0.5
# A walk to a thing is not done until it has got there (the selector's ``walking_to``) — unless it
# makes no headway: this many seconds without having come 5 cm nearer, or the bout has lasted this long.
APPROACH_STALL_S = 8.0
APPROACH_PROGRESS_M = 0.05
APPROACH_NEAR_M = 0.35  # within this the last stretch has less patience:
APPROACH_NEAR_STALL_S = 4.0
APPROACH_MAX_S = 45.0  # … the longest (the boldest and liveliest; a fifth to a third of it for the laziest and shyest)
LOOKING_AT_RAD = 0.35  # the camera looks that way (its head's yaw within this)
# how long each leaky trace of the brain's own lasts (s): being called keeps it looking for the
# one who called for a while (it has to find them first: a mono microphone gives no direction)
TRACE_TAU_S = {"just_woke": 25.0, "name": 12.0, "switching": 3.0, "walked": 8.0, "rested": 12.0}
SWITCH_AFTER_S = 8.0  # an act this long, given up for another kind: an intention head shake
RESTED_AFTER_S = 15.0  # a rest this long: a stretch when it gets up
WALKED_AFTER_S = 15.0  # this much recent measured walking (s): a shake
WALK_PROGRESS_MIN_MPS = 0.02  # ignore odometry jitter / stationary commanded steps
WALK_PROGRESS_MAX_MPS = 0.65  # reject odometry resets/teleports, not a speed estimate
WALK_RECENT_TAU_S = 60.0
ANSWER_TAU_S = 15.0


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


# (dock:build) the pack running low: 0 at 46 %, 1 at 34 % — walking stops below 30 %
# (content.LOCOMOTION["min_battery"]), ~2.5 min of walking after 38 %
BATTERY_LOW_AT = 0.46
BATTERY_LOW_SPAN = 0.12


def _battery_low(frame: SensorFrame) -> float:
    return _clip((BATTERY_LOW_AT - frame.power.battery) / BATTERY_LOW_SPAN)


def _homing(frame: SensorFrame) -> float:
    """The low pack's pull home — off its contacts only (on them it is home)."""
    on_contacts = frame.power.charging or frame.power.on_dock
    return 0.0 if on_contacts else _battery_low(frame)


def _at_charge_place(frame: SensorFrame, dock: D.DockRun) -> bool:
    return (
        frame.power.on_dock
        or frame.power.charging
        or dock.mode == "home"
        or (dock.mode == "docking" and dock.backed)
    )


def _needs_charge(frame: SensorFrame, dock: D.DockRun, leave_battery: float, active: bool) -> bool:
    """Latch a hungry return until charged; a full arrival need not top up first."""
    hungry_return = _at_charge_place(frame, dock) and frame.power.battery < HUNGER_BATTERY_AT
    return (active or hungry_return) and frame.power.battery < leave_battery


# the soft-follow settings of ``lastmap="soft"``: the route goes on through a brush (no reverse arc for geometry alone, a stall is no
# ground made or turned, a shut line must stay shut longer), steers at a short look-ahead, and is given up only after three real stalls
SOFT_FOLLOW: dict[str, Any] = {
    "soft_follow": True,
    "look_m": 0.35,
    "block_m": 0.10,
    "block_ms": 1500,
    "stall_ms": 3000,
    "stall_held": 1.1,
    "stall_yaw": 0.6,
    "jam_held": 0.6,
    "max_tries": 3,
}


def map_configs(
    mode: str,
    clutter: bool | None = None,
    lcfg: dict[str, Any] | None = None,
    ccfg: dict[str, Any] | None = None,
) -> tuple[LastMapConfig, ClutterConfig]:
    """The map's and the clutter escape's settings for a ``lastmap`` mode (the brain's, and the headless rig's: one place). ``lcfg`` /
    ``ccfg``: overrides (development; also ``DUCKMIND_LASTMAP_LCFG`` / ``_CCFG`` in the environment)."""
    env_l, env_c = overrides_from_env()
    lkw: dict[str, Any] = {
        "enabled": mode != "off",
        "soft": False,  # (a continuous soft margin is ``lcfg soft=True``; the ring ladder measured better)
        **env_l,
        **(lcfg or {}),
    }
    if mode in ("act", "actgoal", "wp", "soft"):
        ckw: dict[str, Any] = {
            "enabled": True,
            "action": "waypoint" if mode == "wp" else "route",
            "use_map": True,
            # (live: a route that has stalled twice is no better than the older layers + the wall pivot)
            "max_tries": 2,
            # (measured: a route for a walk to a place behind something helps in dense clutter and hurts in the real home;
            # a route out of a knot is what helped in both: "actgoal" is the one with both)
            "map_goal": mode in ("actgoal", "wp"),
        }
        if mode == "soft":
            ckw.update(SOFT_FOLLOW)
        ckw.update(env_c)
        ckw.update(ccfg or {})
    else:
        ckw = {"enabled": CLUTTER_ENABLED if clutter is None else bool(clutter)}
    return LastMapConfig(**lkw), ClutterConfig(**ckw)


class NeuroBrain:
    """One duck's nervous system. See the module doc."""

    def __init__(
        self,
        genome: Genome,
        seed: int = 0,
        *,
        dock: D.DockConfig | None = None,
        detour: bool | None = None,
        clutter: bool | None = None,
        lastmap: str | bool | None = None,
    ) -> None:
        self.genome = genome
        self.seed = int(seed)
        # (dock:build) its own way into its charging dock (neuro.dock): the turning side is how
        # the family set the dock up (:meth:`set_dock`); the rest is phase 3's measured numbers
        self.dock = D.DockRun(dock)
        self._dock_out = D.DockOut()
        self._pass_ms: int | None = None  # the camera pass the docking program read last
        self._odom_hist: list[tuple[int, tuple[float, float, float]]] = []
        self._head_hist: list[tuple[int, tuple[float, float, float, float]]] = []
        self._seek_nest = False  # it wants its nest and neither sees nor remembers it: it looks
        self._seek_t0: float | None = None  # when it began looking for it (s)
        self._seek_side = 1.0
        self._seek_spent = 0.0  # how long this search has looked so far (s; SEEK_GIVE_UP_S)
        self._nest_rest_until = -1e9  # it has let going home go until then (s; NEST_REST_S)
        self._nest_rests = 0  # the searches in a row that ended without the nest
        p = genome.p
        self.drives = Drives()
        self.drives.restless = 0.35 + 0.3 * genome.axes["activity"]
        self.hab = Habituation(p)
        self.learn = Learner(p)
        self.perceiver = Perceiver()
        self.appraiser = Appraiser(genome, self.learn)
        self.bound = Boundary()  # where the walls are: seen, remembered, steered around
        self._dbg_goal: tuple[float | None, float | None, float | None] = (None, None, None)
        self.detour = Detour(
            detour
        )  # going home round a wall it has seen (a setting: neuro.detour)
        # going round in one place among things: it picks one way out and keeps to it (neuro.clutter; a setting)
        # … and a lasting map of the place (hours, not seconds) built from the same senses, which that route is planned over
        lm_mode = mode_of(lastmap)
        lm_cfg, cl_cfg = map_configs(lm_mode, clutter)
        self.lastmap = LastMap(lm_cfg)
        self._clutter_t = 0  # when the clutter escape was last stepped (a watchdog: an escape nobody steps is ended)
        self._lm_fix: Any = None  # the dock tracker's last fix the map has been given
        self.clutter = Clutter(cl_cfg)
        self._back_prev = (
            False  # (a back arc was on last tick: the clutter memory counts its onsets)
        )
        self._steer_free = 9.9  # the way clear ahead (m), from the boundary memory
        self._reflex_p = R.params_from_genome(genome)
        self._reflex = R.initial(self.seed)
        self._loco_p = M.profile_from_genome(genome).loco
        self._loco = M.LocoState(rng=N.seed_from("neuro.brain.loco", self.seed))
        self.sel = S.Selector(genome, self.seed)
        stream = N.Stream(self.seed, "brain")
        self._stream = stream
        self._fix_b = 0.0  # where the gaze is held now (trunk frame, rad)
        self._fix_e = 0.05
        self._gaze_urge = 0.0
        self._gaze_theta = 1.0
        self._gaze_speed = 0.4
        self._fixations: list[tuple[float, float]] = []  # (bearing, t_s): not looked at again soon
        self._dream = N.OU(stream, 0.0, 1.0, 20.0)
        self._hand = N.OU(stream, 0.0, 0.6, 30.0)  # which way it turns: its side, drifting
        self._tag_ou = {tag: N.OU(stream, 0.0, 0.5, 30.0) for tag in genome.vocal_weights}
        self._vocal_urge = 0.0
        self._vocal_seq = 0
        self._vocal_pending: str | None = None
        self._phrase_left = 0
        self._help_urge = 0.0
        self._contact_urge = 0.0
        self._answer = 0.0  # the urge to answer a call (its name, a hello)
        self._answer_tag = "inquire"
        self._righting_failed = False
        self.behaviour = self.sel.names[self.sel.current]
        self.bout = 0
        self._bout_t_ms = 0
        self._bout: dict[str, Any] = {}
        self._walking_to = False  # the bout is a walk up to a thing it has not got to yet
        self._walk_called = False  # … and a call or a threat has just come in the middle of it
        self.errand = Errand()  # somebody pointed at a place (neuro.errand): the walk there, and how it ended
        self.cmd = S.CommandChannel()  # one owner of the selector's command channel (a command, an errand)
        self._look_for: tuple[int, float] | None = None  # (until ms, side) a command's search for a person
        self._spun = 0.0  # how far (rad, by its own odometry) the spin act has turned the body in this bout
        # the body runs robotd's quick turn (the driver sets it when robotd lists the skill; the rig and a bare body: no)
        self.can_pivot = False
        # the body can roll (robotd lists the official roulade and the driver says so): the flip trick is asked only then
        self.can_roll = False
        self._search_step: tuple[float, int] | None = None  # the search's pivot step: (odom yaw at its start, looking until ms)
        self._posture_want = "sit"
        self._recharging = False
        self._charge_escape = False
        self._dock_lift: LiftWitness | None = None
        self._dock_carried = False
        self._t0: int | None = None
        self._t_prev: int | None = None
        self._yaw_prev: float | None = None
        self._inertia = 0.0
        self._was_asleep = False
        self._traces: dict[str, float] = {}
        self._legs = 0.0
        self._alone = 0.3
        self._person: tuple[float, float, float] | None = None  # believed x, y, t_s (odometry)
        self.counts: dict[str, int] = {}
        self._slow_t = -1e9
        self._slow: dict[str, Any] = {}
        self._home_t = (
            -1e9
        )  # (MAP-EXPLORE) when what it knows of its home was last worked out (HOME_EVERY_S) …
        self._home: dict[str, Any] | None = (
            None  # … and that (``LastMap.home_view``; None with no map)
        )
        self._reason = msg("neuro.brain.boot_reason")
        self._feat = S.empty_features()
        self._walk_s = 0.0
        self._walk_recent = 0.0  # recent measured walking (a leaky integral, s)
        self._odom_prev: tuple[float, float] | None = None
        self._odom_now: tuple[float, float, float] = (0.0, 0.0, 0.0)  # its own latest odometry pose (errands)
        self._odom_t_prev: int | None = None
        self._stepping = False
        self._last_step_s = -1e9  # when it last asked for a walk (s)
        self._touch_away = (
            0  # +1 / −1: the turn that leads away from the thing it last touched (0: none) …
        )
        self._touch_away_until = -1e9  # … until this time (s)
        self._turned = 0.0
        self._turn_hint = 0.0
        self._turn_hint_t = -1e9
        self._turn_flipped = False
        self._turn_sign = float(genome.side)
        self._reason_bout = ""
        self._thing_key: int | None = None  # the thing it saw last (the learner's own key)
        self._thing_last: tuple[float, float, float] | None = None  # its bearing, range, t_s
        self._attend_track = 0  # eye track held through one object-directed bout
        self._attend_seen_s = -1e9  # its last actual eye reading; brief loss uses learned position
        self._attention_align_until = -1  # short initial body turn toward a newly noticed thing
        self._back: tuple[int, float] | None = None  # a back_arc going on: (until ms, turn)
        # … an escape's back arc ends on how far it has really gone back (odometry), not on a
        # clock the body link's holds can eat: (odometry x, y where it began, metres to go back)
        self._back_goal: tuple[float, float, float] | None = None
        self._backs = 0  # back arcs since it last moved on
        self._back_at: tuple[float, float] = (0.0, 0.0)  # where the last one began (odometry)
        self._unstick = Unstick()  # its sense of getting nowhere while it wants to walk
        self._escape: tuple[Escape, int] | None = None  # a back-off due, and when it was due
        self._low_back_t = (
            -1_000_000
        )  # when the last back-off from a low block was asked for (t_ms)
        self._low_backs: list[
            tuple[int, float, float]
        ] = []  # (t_ms, x, y) of those, for the count here
        self._esc_side = 0  # the side an escape chose to turn away to (+1 left, −1 right) …
        self._esc_corner = (
            False  # … and it is the third at that place: a corner (the on-the-spot turn)
        )
        self._esc_until = -1  # … until this t_ms
        self._startled_s: float | None = None  # when a startle last held the body, s
        self._strong_scare_s = -1e9  # when a strong scare (STRONG_SCARE) last startled it, s
        self._touch_s = -1e9  # when something last touched it (a touch onset), s
        self._touch_back_s = (
            -1e9
        )  # … and when it last touched its back / tail / body (the startle's back-steps)
        self._knocked = False  # it fell just after being touched (PUSH_FALL_S): knocked over
        self._req_vx = 0.0  # the forward speed it asked for last tick
        self._give_up = False  # an escape found it at the same place again: give the bout up
        self._corner_backs = 0  # back arcs out of a corner since it last moved on
        self._corner_at: tuple[float, float] = (0.0, 0.0)
        self._turn_to: tuple[float, int] | None = None  # turning its body toward: (heading, until)
        self._pivot = 0.0  # the side to pivot to this tick while turning toward something (0: none)
        self._looking_since: int | None = None  # when the far-side interest began (t_ms)
        self._person_b: float | None = None  # the bearing of the person it attends to, if any
        self._look_stop: tuple[int, int] | None = None  # stopped to look: (since, look then)
        self._look_done = -1_000_000
        self._by_nest = -1e9  # when it last came home beside its nest (s)
        self._by_nest_walk = 0.0  # its walking total then
        self._perform_until_ms = 0  # an emotion plays until then (neuro.emotions): it stands still
        self._still_until_ms = 0  # a told act waits for a still body until then (keep_still): no step, no turn
        self._over: str | None = None  # a told act whose command is over (act_over): its bout is used up
        self._stuck = 0.0  # its body's stalls, one after another at one place (a leaky count)
        self._stuck_at: tuple[float, float] = (0.0, 0.0)
        self.moment: str | None = None  # the last moment it played (MOMENTS)
        self._moment_seq = 0
        self._moment_busy = 0.0  # a leaky trace: a moment is playing
        self._needs_before: dict[str, float] = {}
        self._posture_tired = 0.0
        # (live voice, LV2) what the cloud brain asked the duck to attend to, as ("person"|"thing",
        # eye track): a track the duck itself sees, set by the live bridge and cleared by it; None
        # (the default, and always without the bridge) leaves attention exactly as it was
        self._live_attend: tuple[str, int] | None = None

    def _adopt_params(self, fresh: NeuroBrain) -> None:
        """(Growth G3, a regrow.) Take the genome-derived holders of ``fresh`` (a brain of the
        same seed built from the genome the duck has *now*, never stepped) in place of this
        brain's own, and nothing else: everything this brain has lived (the learner, the
        habituation, the drives, the reflex and the gait in progress, the bout, the odometry, the
        clocks) stays. Call it through :meth:`IntegratedBrain.regrow`, which only does so asleep."""
        genome = fresh.genome
        self.genome = genome
        self.hab._p = genome.p
        self.learn._p = genome.p
        self.appraiser._g = genome
        self._reflex_p = fresh._reflex_p
        self._loco_p = fresh._loco_p
        fresh.sel.adopt_state(self.sel)
        self.sel = fresh.sel
        self.behaviour = self.sel.names[self.sel.current]

    def set_live_attend(self, target: tuple[str, int] | None) -> None:
        """The person or thing (an eye track in view) the duck is asked to pay attention to; None
        to stop. It only chooses *which* of the things and people it sees is the one it attends
        to: whether it attends at all is the selector's competition, and the reflexes still go
        first."""
        self._live_attend = target

    def go_to(self, bearing: float, range_m: float, who: str | None = None) -> tuple[int, float, float]:
        """Somebody pointed at a place: ``bearing`` rad from the trunk (+ left) and ``range_m`` m from it, as a person
        pointing says it (the driver works that out from the page's floor point; the brain never sees the world's
        frame). The brain puts it in its OWN odometry frame, by its own odometry pose now, and keeps an errand
        (:mod:`duckmind.neuro.errand`) that it answers from its own senses. Returns ``(errand id, x, y)``."""
        x, y = to_odom(self._odom_now, float(bearing), float(range_m))
        return self.errand.start(x, y, self._clock_s(), self._odom_now, who), x, y

    def stop_errand(self, reason: str = "stopped") -> bool:
        return self.errand.cancel(reason, self._clock_s())

    def look_for(self, t_ms: int, seconds: float) -> None:
        """A command's search for the person it heard (:mod:`duckmind.neuro.command_brain`): while the
        ``look_around`` act has the body it sweeps its head (the act's own program) and turns, on its
        innate side: on the spot in quick-turn steps when the body has the quick turn (``can_pivot``),
        else in the seek's own tight arc (:data:`SEEK_ARC`) where there is room for it; until this
        long has gone or :meth:`look_for_stop`."""
        until = int(t_ms) + int(round(float(seconds) * 1000.0))
        self._look_for = (until, 1.0 if self.genome.side >= 0.0 else -1.0)
        self._seek_side = self._look_for[1]
        self._search_step = None

    def look_for_stop(self) -> None:
        self._look_for = None

    def spin_made(self) -> None:
        """A told spin is done (the command's own count of the turn): the spin act is used up, whatever bout it is in — a
        bout that began again mid-turn (a one-tick blip at robotd's quick-turn hand-back) would turn the duck round once more."""
        self._spun = max(self._spun, SPIN_RAD)

    def _clock_s(self) -> float:
        return 0.0 if self._t0 is None or self._t_prev is None else (self._t_prev - self._t0) / 1000.0

    def perform(self, until_ms: int) -> None:
        """An emotion plays (the body belongs to it until ``until_ms``): no walking, no head moments."""
        self._perform_until_ms = max(self._perform_until_ms, int(until_ms))

    def keep_still(self, until_ms: int) -> None:
        """A told act waits for a still body (the flip: the roll begins only from a steady stand): until ``until_ms``
        the legs make no step, turn or arc of the duck's own — a reflex's quick step away from a poke apart."""
        self._still_until_ms = max(self._still_until_ms, int(until_ms))

    def act_over(self, behaviour: str) -> None:
        """A told act's command is over (done, given up or cut short): the bout of ``behaviour`` is used up at once. A
        channel only a command gives salience (the flip) has no want of its own to finish its bout by, so a bout left
        unfinished could be taken only by urgent business — a fear act of noise size, and the flight that goes with it."""
        self._over = behaviour

    def fire_moment(self, name: str) -> bool:
        """Play the head act ``name`` (one of the genome's moments) now, as if its own competition had
        fired it: a person's command that the duck cannot do but has heard (a tilt of the head). Never
        while a moment is playing, it sleeps, a reflex or an emotion has the body; False then."""
        if (
            name not in self.sel.m_names
            or self._moment_busy > 0.3
            or self.behaviour in S.SLEEP
            or self._reflex.active is not None
        ):
            return False
        self.moment = name
        self._moment_seq += 1
        self._moment_busy = 1.0
        self.counts[name] = self.counts.get(name, 0) + 1
        return True

    def take_vocal_seq(self) -> int:
        """The next call sequence number. A call the brain did not decide itself (live voice, LV2:
        a call the cloud proposed that won, or the stop of one) takes one, so the motor lane and
        the page read exactly one call per number."""
        self._vocal_seq += 1
        return self._vocal_seq

    def answered(self) -> None:
        """Something else voiced the answer to the call it was waiting to answer (live voice: the
        cloud's reply that won): the urge to answer is spent. Left alone it called again the tick
        after (live: two calls 20 ms apart, the reply and its own greeting)."""
        self._answer = 0.0

    def _measured_walk_seconds(
        self, frame: SensorFrame, dt: float, requested_walk: bool, lifted: bool
    ) -> float:
        """Elapsed tick time supported by plausible contact-odometry translation.

        ``BodySense.vx`` is the applied request, not measured speed. Credit only displacement
        between consecutive valid odometry samples while the prior command requested a walk and
        posture/contact state says the duck is grounded. Unknown leg load is allowed, matching
        body-link semantics; a known unloaded sample is not. The maximum plausible displacement
        rejects resets, though small odometry relocalizations cannot be distinguished here.
        """
        body = frame.body
        previous, previous_t = self._odom_prev, self._odom_t_prev
        sample_dt = None if previous_t is None else (frame.t_ms - previous_t) / 1000.0
        valid = (
            dt > 0.0
            and (
                sample_dt is None
                or (0.0 < sample_dt <= 0.25 and abs(sample_dt - dt) <= max(0.03, dt * 0.5))
            )
            and math.isfinite(body.odom[0])
            and math.isfinite(body.odom[1])
            and body.posture == "walk"
            and not body.fallen
            and not body.limp
            and not lifted
            and (body.leg_load is None or body.leg_load >= 0.45)
            and (body.step_phase is None or 0.0 <= body.step_phase <= 1.0)
            and not any(
                event in frame.events for event in ("fell", "lifted", "set_down", "righted")
            )
        )
        if not valid:
            # Do not bridge held/fall/reset/invalid intervals with the next healthy sample.
            self._odom_prev = None
            self._odom_t_prev = None
            return 0.0
        current = (body.odom[0], body.odom[1])
        self._odom_prev, self._odom_t_prev = current, frame.t_ms
        if previous is None or previous_t is None or sample_dt is None or not requested_walk:
            return 0.0
        distance = math.hypot(current[0] - previous[0], current[1] - previous[1])
        speed = distance / sample_dt
        if speed > WALK_PROGRESS_MAX_MPS:
            # A teleport/reset must not seed credit on the following sample.
            self._odom_prev = None
            self._odom_t_prev = None
            return 0.0
        if speed < WALK_PROGRESS_MIN_MPS:
            return 0.0
        return min(dt, sample_dt)

    # ── the step ─────────────────────────────────────────────────────────────

    def step(self, t_ms: int, frame: SensorFrame) -> tuple[MotorCommand, BrainTelemetry]:
        g = self.genome
        p = g.p
        if self._t0 is None:
            self._t0 = t_ms
            self._bout_t_ms = t_ms
        dt = 0.05 if self._t_prev is None else _clip((t_ms - self._t_prev) / 1000.0, 0.0, 2.0)
        self._t_prev = t_ms
        t_s = (t_ms - self._t0) / 1000.0
        clock_offset = t_ms - frame.t_ms  # the frame's clock → the brain's (a look's time)
        if frame.t_ms != t_ms:
            frame = replace(frame, t_ms=t_ms)  # the arcs run on the brain's own clock
        d = self.drives
        asleep = self.behaviour in S.SLEEP
        odom = frame.body.odom
        self._odom_now = (float(odom[0]), float(odom[1]), float(odom[2]))
        posture = frame.body.posture
        sitting = posture == "sit"
        standing = posture in ("stand", "walk")
        walking_now = self._stepping and standing
        yaw = odom[2]
        self._turned = 0.0 if self._yaw_prev is None else abs(wrap_angle(yaw - self._yaw_prev))
        self._yaw_prev = yaw
        if self.behaviour == "spin":
            self._spun += self._turned
        # (dock:build) what the docking program reads: the raw body events, and the dock's tag
        # from a camera pass that just arrived (its bearing is from the trunk at the capture)
        raw_events = frame.events
        tag_read, dock_cue = self._dock_reading(t_ms, frame)
        if self.dock.expects_contact() and DOCK_CONTACT_EVENTS.intersection(frame.events):
            frame = replace(
                frame, events=tuple(e for e in frame.events if e not in DOCK_CONTACT_EVENTS)
            )

        pct = self.perceiver.perceive(frame, t_s, asleep)
        live_pin = False
        if self._live_attend is not None and not asleep:
            want_kind, want_track = self._live_attend
            for sight in frame.camera.sightings:
                if sight.track != want_track or not want_track:
                    continue
                if want_kind == "person" and sight.kind == "person":
                    pct.person = sight
                elif want_kind == "thing" and sight.kind in ("thing", "toy"):
                    pct.thing = sight
                    live_pin = True
        # the walls and furniture it has seen (the camera's floor line), carried by odometry
        self.bound.update(t_ms, frame, clock_offset=clock_offset)
        if self.clutter.mode == "escape" and t_ms - self._clutter_t > 500:
            self.clutter.reset(
                t_ms, cooldown=True
            )  # (the step was skipped for half a second: whoever had the legs did not give them back)
        self.lastmap.update(
            t_ms,
            frame,
            clock_offset=clock_offset,
            blocked_now="blocked" in frame.events or bool(frame.tof.touch),
        )

        # ── reflex arcs, with this brain's neuromodulators and plasticity ─────
        # (which way to turn when the senses do not say: its innate side, drifting slowly; the
        # way it has SEEN open; whether the way ahead is seen clear — review 2026-09-28: no
        # coins and no fixed turn times)
        seen = self.bound.last
        free_now = self.bound.ahead(odom) if standing else 9.9
        # which way to turn away: ONE side for every part of the brain that bends the walk. An
        # escape's chosen side first; else the way the boundary memory says is best (its veer goes
        # that way: the reflex arc turned the other, the two summed to a straight walk into the
        # wall, 2026-10-04); else the side it saw freer
        committed = self._esc_side if t_ms < self._esc_until else 0
        turn_side = committed or (
            (1 if seen.best > 0.0 else -1)
            if seen.urgency > 0.0 and abs(seen.best) >= 0.2
            else seen.side
        )
        # (fid:fix) its act holds the head still (freeze, startle, still, a 定格 moment): an
        # orienting onset is then covert — heard and seen, nothing moves, no row
        still_act = (
            self.behaviour in R.HOLDING_STILL
            or (self.moment == "still" and self._moment_busy > 0.45)
            # (dock:build) looking at its dock's tag, turning round, backing in: its head is on
            # the job — an orienting onset is covert (heard, seen, no turn of the head)
            or self.dock.head_busy()
        )
        # the fright is a flight (see STARTLE_FLIGHT_S): the freeze after a startle is only a beat
        flight = (
            self._startled_s is not None
            and standing
            and not asleep
            and frame.power.battery >= 0.3
            and self._fright_persists(t_s)
        )
        self._reflex, rx = R.step(
            self._reflex_p,
            self._reflex,
            frame,
            arousal=d.arousal,
            asleep=asleep,
            expect_touch=pct.expect_touch,
            fear=d.fear,
            comfort=d.comfort,
            walking=walking_now and self._req_vx >= 0.0,  # (backing out is not walking on)
            hab=self.hab.level_view(t_s),
            sens=self.hab.sens_view(t_s),
            lean=self._hand.x,
            free_side=turn_side,
            way_clear=free_now >= NEAR_M,
            turn_need=abs(seen.best) if seen.urgency > 0.0 and abs(seen.best) > 0.2 else None,
            still_act=still_act,
            engaged=self.behaviour in R.ENGAGED_ACTS,
            expect_hand=pct.expect_hand,
            commit_side=committed,
            flight=flight,
        )
        for onset in rx.presented:
            if onset.stimulus.startswith("touch:"):
                self._touch_s = t_s
                if onset.stimulus.split(":", 1)[1] in ("back", "tail", "body"):
                    self._touch_back_s = t_s
            if (
                onset.reflex == "startle"
                and onset.depress
                and not asleep
                and (
                    onset.floor  # a sudden loud sound it was not used to (R.STARTLE_FLOOR_RAW)
                    or onset.raw * self.hab.level(onset.stimulus, t_s) >= STRONG_SCARE
                )
            ):
                self._strong_scare_s = t_s  # (its habituation level now: before it is presented)
            if onset.depress:
                # a cue that has come to mean something habituates more slowly (its own name
                # above all: review 2026-09-28)
                resist = max(
                    NAME_RESIST if onset.stimulus == "sound:name" else 0.0,
                    self.learn.value_of(onset.stimulus),
                )
                self.hab.present(onset.stimulus, onset.raw, t_s, onset.reflex, resist)
            if onset.scare > 0.0:
                self.hab.sensitise(onset.stimulus, onset.scare, t_s)
            if onset.stimulus in ("sound:name", "sound:hello") and not asleep:
                # someone calls or greets it: the urge to answer (a leaky integrator; a social
                # duck answers the first call, an aloof one the second — review 2026-09-28)
                soc = g.axes["sociability"]
                kick = 0.6 + 0.6 * soc + (0.25 if onset.stimulus == "sound:name" else 0.0)
                kick += 0.3 * max(0.0, self.learn.value_of(onset.stimulus))
                self._answer = min(2.0, self._answer + kick)
                self._answer_tag = "greet" if onset.stimulus == "sound:hello" else "inquire"
        ap = self.appraiser.step(
            t_s,
            dt,
            frame,
            rx.presented,
            rx.events,
            asleep=asleep,
            walking=walking_now,
            lifted=rx.lifted,
            blocked_ahead=standing and rx.vx_cap is not None and rx.vx_cap < ARC_VX_MIN,
            moving=walking_now or frame.body.posture == "walk",
            looming_bearing=pct.looming_bearing,
            looming_track=pct.looming_track,
        )
        hold = rx.hold if rx.hold in HOLD_BEHAVIOUR else None
        if hold is not None and self.moment == "still":
            # (fid:fix) a reflex took the body: the 定格 moment is over (the motor drops it too)
            self._moment_busy = min(self._moment_busy, 0.44)
        measured_walk_s = self._measured_walk_seconds(frame, dt, self._stepping, rx.lifted)
        measured_walking = measured_walk_s / dt if dt > 0.0 else 0.0
        self._walk_s += measured_walk_s
        self._walk_recent += measured_walk_s
        for event in rx.events:
            if event.stimulus == "body:righting_failed":
                self._righting_failed = True
            elif event.stimulus in ("body:fall", "body:righted", "body:righted_help"):
                self._righting_failed = False
                self._help_urge = 0.0

        # its own body stalling again and again at one spot: it is stuck (its own sense — the
        # page may then offer 「抱到地墊上」; the place is learned as a bad one)
        self._stuck *= math.exp(-dt / STUCK_TAU_S)
        if "blocked" in frame.events:
            if math.hypot(odom[0] - self._stuck_at[0], odom[1] - self._stuck_at[1]) > 0.5:
                self._stuck = 0.0
                self._stuck_at = (odom[0], odom[1])
            self._stuck += 1.0
        elif math.hypot(odom[0] - self._stuck_at[0], odom[1] - self._stuck_at[1]) > 0.8:
            self._stuck = 0.0  # it got away from there

        # ── learning and the world as it seems ───────────────────────────────
        place_novelty = self.learn.visit(odom, dt)
        self.learn.fade(dt)
        novelty_seen = 0.0
        thing_near = 0.0
        thing_bearing: float | None = None
        new_visual_track = False
        attended_thing = pct.thing
        locked_attention = False
        if self.behaviour in THING_ACTS and self._attend_track:
            held = next((s for s in frame.camera.sightings if s.track == self._attend_track), None)
            if held is not None:
                attended_thing = held
                self._attend_seen_s = t_s
                locked_attention = True
            elif t_s - self._attend_seen_s <= 1.0:
                key = self.learn._tracks.get(self._attend_track)
                remembered_locked = None if key is None else self.learn.things.get(key)
                if remembered_locked is not None:
                    bearing, distance = bearing_to(odom, remembered_locked.x, remembered_locked.y)
                    attended_thing = Sighting(
                        kind=remembered_locked.kind,
                        bearing=bearing,
                        range_m=distance,
                        size=pct.thing.size if pct.thing is not None else 0.05,
                        track=self._attend_track,
                        source="memory",
                    )
                    locked_attention = True
        # The thing a walk is toward is the one thing it goes to (2026-10-05, legs): when the eye drops
        # its track (a still thing is lost while the legs step) and finds it again under another number,
        # or sees other things on the way, the walk does not turn to those — a thing in view where the
        # one it is going to is believed to be (or nearer to it than the eye's error) is that thing;
        # none there: it goes on by its own odometry (``remembered`` below).
        cap_odom = self._at_capture(t_ms, frame, odom)
        focus_key = self._bout.get("thing_key") if self.behaviour == "investigate" else None
        focus = None if focus_key is None else self.learn.things.get(focus_key)
        if focus is not None and not locked_attention:
            match: tuple[float, Sighting] | None = None
            for sight in pct.things:
                if sight.range_m is None or sight.kind not in ("thing", "toy"):
                    continue
                sx, sy = sighting_odom(
                    cap_odom, lens_point(frame), sight.bearing, float(sight.range_m)
                )
                gap = math.hypot(sx - focus.x, sy - focus.y)
                if gap <= FOCUS_MATCH_M + FOCUS_FAR_PER_M * float(sight.range_m) and (
                    match is None or gap < match[0]
                ):
                    match = (gap, sight)
            attended_thing = None if match is None else match[1]
            locked_attention = True
            if match is not None and match[1].track:
                self.learn._tracks[match[1].track] = focus_key  # the same thing under a new number
                self._attend_track = match[1].track
                self._attend_seen_s = t_s
        # Perceiver keeps every visible thing. Give a newly arrived numeric eye track a chance
        # to be attended before falling back to the largest silhouette; otherwise a familiar
        # nearby book can permanently hide a smaller new bottle from curiosity.
        new_things = [s for s in pct.things if s.track and s.track not in self.learn._tracks]
        if new_things and not locked_attention and not live_pin:
            attended_thing = max(new_things, key=lambda s: s.size)
        # The selected attention target must also feed the existing goal compiler below.
        # Investigate already recomputes its target every tick; this lets an active bout
        # hand from a familiar large object to a newly arrived track without a second selector
        # or a direct motor command.
        if attended_thing is not None:
            attended_thing = self._fresh(attended_thing, cap_odom, odom, frame)
        pct.thing = attended_thing
        if attended_thing is not None:
            rng = attended_thing.range_m if attended_thing.range_m is not None else 1.0
            if attended_thing.source == "memory":
                x, y = to_odom(odom, attended_thing.bearing, rng)  # its own belief: trunk's frame
            else:  # seen: bearing and range start at the lens, not at the trunk's centre
                x, y = sighting_odom(odom, lens_point(frame), attended_thing.bearing, rng)
            handling = 1.0 if self.behaviour in ("nibble", "peck") and rng < 0.35 else 0.0
            new_visual_track = bool(
                attended_thing.track and attended_thing.track not in self.learn._tracks
            )
            # known again by where it is (odometry) and what sort it is, not by the eye's track
            thing_key, novelty_seen = self.learn.see_thing(
                attended_thing.track, attended_thing.kind, odom, x, y, t_s, dt, handling
            )
            self._thing_key = thing_key
            if self.behaviour == "investigate":
                self._bout["thing_key"] = thing_key
            if self.behaviour == "investigate" and any(
                e.stimulus in ("body:bump", "body:brushed") for e in rx.events
            ):
                # it walked into the very thing it went to look at: that thing is less inviting
                self.learn.thing_value(thing_key, -0.3)
            known = self.learn.things.get(thing_key)
            if known is not None and known.value < 0.0:
                novelty_seen *= _clip(1.0 + known.value)  # a thing that went wrong is no draw
            thing_near = _clip((0.45 - rng) / 0.3)
            thing_bearing = attended_thing.bearing
            self._thing_last = (attended_thing.bearing, rng, t_s)
        remembered = None
        if pct.thing is None:
            remembered = self.learn.remembered_thing(odom, t_s)
            held_key = self._bout.get("thing_key") if self.behaviour == "investigate" else None
            if held_key is not None:
                # the walk is toward one thing: it stays in mind for the whole walk, by its own
                # odometry, however short a time ago the eye lost it (the eye loses a still thing
                # while the legs step, and the floor nearer than ~0.2 m is out of its sight)
                held = self.learn.thing_belief(held_key, odom, t_s)
                if held is not None:
                    remembered = held
            if (
                remembered is not None
                and self.behaviour == "investigate"
                and any(e.stimulus in ("body:bump", "body:brushed") for e in rx.events)
            ):
                # walking toward where it remembered a thing, it hit something: less inviting
                self.learn.thing_value(remembered[0], -0.3)
        if pct.person is not None:
            rng = pct.person.range_m if pct.person.range_m is not None else 1.5
            px, py = to_odom(odom, pct.person.bearing, rng)
            self._person = (px, py, t_s)
        # where the person it attends to is now (seen, or believed a moment after it left the
        # picture, carried by its own odometry): its eyes pursue them and its body follows
        # (fid:motor 2026-09-28)
        self._person_b = None
        if pct.person is not None:
            self._person_b = float(pct.person.bearing)
        elif self._person is not None and t_s - self._person[2] < PURSUE_BELIEF_S:
            self._person_b = bearing_to(odom, self._person[0], self._person[1])[0]
        touch_gentle = pct.touch_gentle
        presence = max(
            touch_gentle,
            _clip(1.6 - (pct.person.range_m or 3.0)) if pct.person is not None else 0.0,
            0.6 * pct.voice,
        )
        contact = 0.0
        sc = p.social_contact
        for onset in rx.presented:
            cls = onset.stimulus
            if cls == "sound:name":
                contact += sc["name"]
                if (
                    self._person is not None
                    and t_s - self._person[2] < PERSON_KNOWN_S
                    and self.genome.axes["sociability"] >= COMES_WHEN_CALLED_SOCIABILITY
                ):
                    # called by name by a person it knows is near, and it is the sociable sort: it
                    # goes to them (``follow`` steers by where it believes them)
                    self.sel.commit_next("follow")
            elif cls == "sound:hello":
                contact += sc["hello"]
            elif cls == "sound:voice":
                contact += sc["voice"]
            elif cls.startswith("touch:") and onset.reflex == "lean_in":
                contact += sc["touch"]
                self.learn.mark(odom, 0.6, family=1.0)
            elif cls == "visual:person":
                self.learn.mark(odom, 0.2, family=0.5)
        if "set_down" in frame.events:
            contact += sc["pickup"]
        contact = max(contact, ap.contact)
        if frame.power.charging or frame.power.on_dock:
            self.learn.at_dock(odom)
        if ap.positive > 0.0:
            self.learn.mark(odom, ap.positive, family=0.5)

        # ── drives ───────────────────────────────────────────────────────────
        beh = self.behaviour
        investigating = (
            1.0
            if beh in S.INVESTIGATIVE
            and (pct.thing is not None or beh in ("look_around", "survey"))
            else 0.0
        )
        stimulation = _clip(
            pct.voice
            + pct.music
            + pct.motion
            + touch_gentle
            + 0.5 * pct.tv
            + (0.3 if pct.person else 0.0)
        )
        novel_now = 1.0 if pct.thing is not None and novelty_seen > 0.5 else 0.0
        self._needs_before = {
            "groom": d.groom,
            "peckish": d.peckish,
            "vigil": d.vigil,
            "restless": d.restless,
        }
        d.update(
            DriveInputs(
                dt_s=dt,
                day_minute=frame.day_minute,
                asleep=asleep,
                sitting=sitting,
                walking=measured_walking,
                exertion=EXERTION.get(beh, 0.1),
                novelty=max(
                    ap.novelty,
                    1.2 * novelty_seen * novel_now * dt,
                    0.2 * place_novelty * measured_walking * dt,
                ),
                investigating=investigating,
                stimulation=stimulation,
                presence=presence,
                contact=contact,
                touch_gentle=touch_gentle,
                threat=ap.threat,
                pain=ap.pain,
                positive=ap.positive,
                playing=1.0 if beh in S.PLAY else 0.0,
                stretching=1.0 if beh in ("stretch_yawn", "wake_ritual") else 0.0,
                grooming=GROOMING.get(beh, 0.0),
                pecking=PECKING.get(beh, 0.0),
                # (the looks in a walk's pauses are looking around too: the walk's own stops
                # satisfy the urge to look, not only a separate act — fewer act switches)
                scanning=SCANNING.get(beh, 0.0)
                if beh not in S.LOCOMOTES or self._stepping
                else PAUSE_SCANNING,
                battery=frame.power.battery,
                charging=frame.power.charging,
            ),
            p,
            stiff_gain=1.6 if g.habit == "stretch_after_nap" else 1.0,
        )

        # ── traces (leaky memories of what just happened) ─────────────────────
        tr = self._traces
        for key in list(tr):
            tr[key] *= math.exp(-dt / TRACE_TAU_S.get(key, 5.0))
        if pct.voice > 0.0:
            tr["voice"] = max(tr.get("voice", 0.0), pct.voice)
        if pct.name > 0.0:
            tr["name"] = max(tr.get("name", 0.0), pct.name)
        if pct.hello > 0.0:
            tr["name"] = max(tr.get("name", 0.0), 0.7 * pct.hello)  # a hello calls it too
        if pct.person_new > 0.0:
            tr["person_new"] = max(tr.get("person_new", 0.0), pct.person_new)
        if pct.motion > 0.0:
            tr["motion"] = max(tr.get("motion", 0.0), pct.motion)
        if novelty_seen > 0.0:
            tr["novel"] = max(tr.get("novel", 0.0), novelty_seen * (1.0 if pct.thing else 0.0))
        # sleep inertia builds while asleep and becomes the groggy "just woke" trace on waking
        if asleep:
            self._inertia += (1.0 - self._inertia) * (1.0 - math.exp(-dt / 90.0))
        elif self._was_asleep:
            tr["just_woke"] = max(tr.get("just_woke", 0.0), self._inertia)
            self._inertia = 0.0
        self._was_asleep = asleep
        self._legs = _clip(
            self._legs
            + (dt / (3.0 * self._stand_s()) if standing and not self._stepping else 0.0)
            - (dt / 40.0 if sitting else 0.0)
            - (dt / 90.0 if self._stepping else 0.0)
        )
        self._alone += (
            (dt / 600.0) * (1.0 - self._alone)
            if presence < 0.15 and contact == 0.0
            else -self._alone * min(1.0, dt / 20.0 + contact * 3)
        )

        # ── features: the competition's inputs ───────────────────────────────
        f = self._feat
        F = S.F
        inp = ap.inputs
        f[F["curiosity"]] = d.curiosity
        f[F["boredom"]] = d.boredom
        f[F["restless"]] = d.restless * (0.5 + 0.5 * d.energy)
        f[F["energy"]] = d.energy
        f[F["fatigue"]] = d.fatigue
        f[F["sleepy"]] = d.sleepiness
        f[F["fear"]] = d.fear
        f[F["social"]] = d.social
        f[F["comfort"]] = d.comfort
        f[F["play"]] = d.play
        f[F["hunger"]] = d.hunger
        f[F["arousal"]] = d.arousal
        f[F["calm"]] = _clip(1.0 - 1.4 * d.arousal)
        f[F["happy"]] = _clip(d.valence)
        f[F["heard"]] = inp.get("heard", 0.0)
        f[F["sound_nodir"]] = inp.get("sound_nodir", 0.0)
        f[F["voice"]] = tr.get("voice", 0.0)
        f[F["name"]] = tr.get("name", 0.0)
        f[F["music"]] = pct.music
        f[F["tv"]] = pct.tv
        f[F["startled"]] = inp.get("startled", 0.0)
        f[F["recovering"]] = inp.get("recovering", 0.0)
        f[F["rhythm_slow"]] = inp.get("rhythm_slow", 0.0)
        f[F["rhythm_fast"]] = inp.get("rhythm_fast", 0.0)
        f[F["alert"]] = inp.get("alert", 0.0)
        f[F["petted"]] = inp.get("petted", 0.0)
        f[F["touch_now"]] = touch_gentle
        f[F["touched_sharp"]] = inp.get("touched_sharp", 0.0)
        f[F["person"]] = 1.0 if pct.person is not None else 0.0
        near = pct.person is not None and (pct.person.range_m or 9.0) < 1.0
        f[F["person_near"]] = max(1.0 if near else 0.0, touch_gentle)
        f[F["person_moving"]] = _clip(pct.person.motion * 2.0) if pct.person is not None else 0.0
        f[F["person_new"]] = tr.get("person_new", 0.0)
        f[F["greet_up"]] = tr.get("person_new", 0.0) * p.greet_reflex
        f[F["settle_in"]] = touch_gentle * p.settle
        # a far thing draws less than a near one: the walk there is long (0.15 m/s measured, 1-3 m in
        # 7-20 s) and the duck goes for what it can get to — a thing farther than TRAVEL_FREE_M draws
        # half as much at TRAVEL_FREE_M + TRAVEL_HALF_M, a fifth at one more (it is still there, and
        # draws as it nears)
        thing_m = (
            float(pct.thing.range_m)
            if pct.thing is not None and pct.thing.range_m is not None
            else (float(remembered[2]) if remembered is not None else None)
        )
        travel = (
            1.0
            if thing_m is None
            else 1.0 / (1.0 + (max(0.0, thing_m - TRAVEL_FREE_M) / TRAVEL_HALF_M) ** 2)
        )
        f[F["thing"]] = travel if pct.thing is not None else 0.0
        f[F["thing_near"]] = thing_near
        f[F["novel"]] = tr.get("novel", 0.0) * travel
        f[F["remembered_thing"]] = (
            _clip(remembered[3]) * 0.8 * travel if remembered is not None else 0.0
        )
        f[F["motion"]] = tr.get("motion", 0.0)
        f[F["looming"]] = inp.get("loom_scare", 0.0)
        f[F["dark"]] = pct.dark
        f[F["light_change"]] = inp.get("light_change", 0.0)
        f[F["light_on"]] = inp.get("light_on", 0.0)
        f[F["nest"]] = 1.0 if pct.nest is not None else 0.0
        f[F["on_dock"]] = 1.0 if frame.power.on_dock else 0.0
        f[F["just_woke"]] = tr.get("just_woke", 0.0)
        f[F["stiff"]] = d.stiffness
        f[F["after_fall"]] = inp.get("after_fall", 0.0)
        f[F["righted"]] = inp.get("righted", 0.0)
        f[F["bump"]] = inp.get("bump", 0.0)
        # the way ahead as it has SEEN it (the camera's floor line, remembered): a wall closing
        # in while it walks makes turning away pressing — before any bump (boundary.py)
        wall = 0.0
        if standing:
            self._steer_free = self.bound.ahead(odom)
            wall = _clip((SLOW_M - self._steer_free) / (SLOW_M - STOP_M))
        else:
            self._steer_free = 9.9
        # (a wall it sees coming is bent around inside the walk itself — boundary.steer — and a
        # wall right ahead makes the reflex stop and arc away; turn_around is its own act, not
        # the label of every bend: review 2026-09-28, wander ↔ turn_around every few seconds)
        f[F["turn_away"]] = inp.get("turn_away", 0.0) + inp.get("avoid_spot", 0.0) * 0.5
        f[F["obstacle"]] = max(inp.get("obstacle", 0.0), wall * (0.5 if self._stepping else 0.2))
        f[F["stumble"]] = inp.get("stumble", 0.0)
        f[F["set_down"]] = inp.get("set_down", 0.0)
        f[F["alone"]] = _clip((self._alone - 0.5) * 2.0) * d.social
        f[F["good_cue"]] = inp.get("good_cue", 0.0)
        f[F["bad_cue"]] = inp.get("bad_cue", 0.0)
        # a thing it knows came back into view: "there it is again" (a double take)
        f[F["double_take"]] = max(inp.get("double_take", 0.0), self.learn.reappeared)
        f[F["switching"]] = tr.get("switching", 0.0)
        f[F["walked"]] = tr.get("walked", 0.0)
        f[F["rested"]] = tr.get("rested", 0.0)
        f[F["reached"]] = tr.get("reached", 0.0)
        f[F["place_new"]] = place_novelty if not self._stepping and standing else 0.0
        # no arc it could make has room (the walls it saw): it stops, sits, calls its family
        cornered = standing and self.bound.last.cornered and self._back is None
        f[F["cornered"]] = 1.0 if cornered else 0.0
        if cornered:
            self._stuck += dt / CORNERED_STUCK_S  # (the page may offer 「抱到地墊上」)
        liked = self.learn.liked(odom, p.leash_m) if d.social > 0.5 else None
        f[F["liked_place"]] = _clip(liked[2]) * d.social if liked is not None else 0.0
        f[F["night"]] = 1.0 - d.alertness
        f[F["legs_tired"]] = self._legs
        f[F["dream"]] = _clip(self._dream.step(dt) - 1.6) if asleep else 0.0
        f[F["preen_next"]] = tr.get("preen_next", 0.0)
        # at its nest (charging on it, or resting beside it after coming home)
        by_nest = frame.power.on_dock or (
            t_s - self._by_nest < BY_NEST_S and self._walk_s - self._by_nest_walk < 2.0
        )
        # (dock:build: in its dock by its own belief counts too — standing up in it opens the
        # contacts, but it is still at its nest wanting to charge: it stays)
        at_nest = by_nest or self.dock.mode == "home"
        f[F["charging"]] = d.hunger if at_nest else 0.0
        f[F["on_contacts"]] = d.hunger if frame.power.on_dock else 0.0
        # (dock:build) a docking under way is seen through (the loop's self-excitation on its
        # channel); sleepy away from its nest, it wants to go home to sleep
        f[F["docking"]] = 1.0 if self.dock.mode == "docking" else 0.0
        if pct.nest is not None or at_nest or self.dock.mode != "idle":
            # its nest in view (or it is in it): a search that gave up is over, the next starts afresh (SEEK_GIVE_UP_S)
            self._nest_rest_until, self._nest_rests, self._seek_spent = -1e9, 0, 0.0
        nest_rest = t_s < self._nest_rest_until  # (it let going home go for now: NEST_REST_S)
        f[F["home_pull"]] = 0.0 if at_nest or nest_rest else d.sleepiness
        # (dock:build) the pack running low: it damps the walks away (on the dock too — it stays
        # to charge) and, off its contacts, pulls it home whatever its temperament
        f[F["battery_low"]] = _battery_low(frame)
        f[F["homing"]] = 0.0 if nest_rest else _homing(frame)
        # the itch pulls hard only once it is a real itch (squared) — and once it preens, the
        # act's own loop keeps it at it until the itch is scratched (linear): a duck preens
        # seldom and then for a while (pet:etho: bouts of 10 s–5 min), not a scratch when idle
        grooming = self.behaviour in ("preen", "body_shake")
        f[F["groom"]] = d.groom if grooming else d.groom * d.groom
        f[F["peckish"]] = d.peckish
        f[F["vigil"]] = d.vigil
        # called, and nobody in sight for a moment: look for them (turn round, look about)
        seen_person = self._person is not None and t_s - self._person[2] < PERSON_KNOWN_S
        f[F["called_unseen"]] = 0.0 if seen_person else tr.get("name", 0.0)
        f[F["called_seen"]] = tr.get("name", 0.0) if seen_person else 0.0

        # ── selection ────────────────────────────────────────────────────────
        battery_ok = frame.power.battery >= 0.3
        can_walk = (
            battery_ok
            and not rx.fallen
            and not rx.lifted
            and not (frame.camera.usable and frame.camera.light == "black")
        )
        wakefulness = _clip(
            1.3 - 1.7 * d.sleepiness + 1.2 * max(0.0, d.arousal - 0.35) + inp.get("woken", 0.0)
        )
        self._posture_tired *= math.exp(-dt / POSTURE_TIRED_TAU_S)
        self._recharging = _needs_charge(frame, self.dock, p.leave_battery, self._recharging)
        self._charge_escape = self._fright_persists(t_s) or rx.escape is not None
        ctx = S.Context(
            sitting=sitting,
            standing=standing,
            asleep=asleep,
            can_walk=can_walk,
            wakefulness=wakefulness,
            posture_tired=self._posture_tired,
            recharging=self._recharging and _at_charge_place(frame, self.dock),
            charge_escape=self._charge_escape,
            nest_rest=nest_rest,
        )
        self._touch_away_update(t_s, frame)
        veer = rx.vyaw_bias + ap.veer
        if veer:
            self._turn_hint = veer
            self._turn_hint_t = t_s
        elif t_s - self._turn_hint_t > 3.0:
            self._turn_hint = 0.0  # a veer is a hint for a turn-around only for a moment
        consumed = self._consumed(dt, pct, thing_near, frame, novelty_seen, remembered, odom, t_s)
        if self._give_up:
            consumed += GIVE_UP_USE  # (it keeps ending up here: it lets that go for now)
            self._give_up = False
        reflex_interrupted = any(event.reflex in R.INTERRUPTS for event in rx.events)
        # A newly tracked thing is an attention interrupt for an ordinary wander: it releases
        # the current wander's commitment so the existing selector can compare investigate
        # immediately; it does not apply to a familiar or untracked thing. With the finishing rule
        # (M5) every bout that is not done holds on, so something new that appears is let in to any
        # ordinary bout, not only a wander (``_novelty_interrupts``): a duck that preens, forages or
        # looks around does not ignore a new thing for the rest of the bout (v7 L2 novel_object
        # attended 0.67 -> 0.20). For a bout that is not a wander the curiosity act claims the body
        # for the tick (below).
        novelty_onset = new_visual_track and novelty_seen > 0.5 and self._novelty_interrupts(f)
        attention_interrupt = novelty_onset and self.behaviour == "wander"
        if attention_interrupt:
            # A new eye track is one discrete attention event. Release most of the old
            # wander accumulator before the normal selector competition; the selector still
            # decides whether investigate actually wins, and the release is not repeated while
            # the same track remains visible.
            self.sel.act[self.sel.current] *= 0.1
            investigate_i = self.sel.index("investigate")
            # One-shot attention pulse: make the new object's existing investigate channel
            # competitive with the just-interrupted wander. The Selector still applies all
            # gates, weights, safety state and the final winner comparison on this tick.
            self.sel.act[investigate_i] = max(
                self.sel.act[investigate_i], self.sel.act[self.sel.current] + 0.5
            )
            self._attention_align_until = t_ms + 1000
        elif novelty_onset:
            # Any other ordinary bout: the act is not killed and left to whoever is next (a release
            # of its act let sit_down take the body, 23 times in 64, for a second before investigate
            # did: M5's natural endings 93 -> 85 %): the curiosity act claims the body for this tick,
            # on the terms of a person's answer (Selector.claim_next), and the bout it starts is held on
            # to by the finishing rule like any other.
            self.sel.claim_next("investigate")
            self._attention_align_until = t_ms + 1000
        interrupted = reflex_interrupted
        # a startle whose fright has not gone is a flight, at once (owner 2026-10-04): when the
        # reflexes (the startle, then the freeze it goes on into) have let go of the body and the
        # fright is still up, it runs — it does not settle into a calm walk
        if any(e.stimulus == "body:fall" for e in rx.events) and t_s - self._touch_s <= PUSH_FALL_S:
            self._knocked = True
        if self._knocked:
            if frame.body.fallen or rx.hold == "righting":
                self._startled_s = self._strong_scare_s = t_s  # (the flight starts when it is up)
            else:
                self._knocked = False
        if rx.hold in ("startle", "freeze") or rx.active in ("startle", "freeze"):
            self._startled_s = t_s
        elif self._startled_s is not None and rx.hold is None:
            in_window = t_s - self._startled_s <= STARTLE_FLIGHT_S
            if in_window and standing and can_walk and not asleep and self._fright_persists(t_s):
                self.sel.commit_next("flee")
            # (2026-10-09) the body is not standing yet because a trained skill still has it (the held crouch and
            # startle_release's rise carry labels that are no posture): the flight waits for the stand, within
            # STARTLE_FLIGHT_S of the hold's end, instead of being forgotten at this tick (live fresh boots with the
            # startle floor fired: 4 of 12 never fled on a707a50, 0 of 13 on eb08fe8)
            waiting = in_window and not asleep and posture in ("unknown", "transition")
            if not waiting:
                self._startled_s = None
        # an errand (somebody pointed at a place): a sense input answered from its own senses. It never takes the body
        # from sleep, a fright, a walk home or a docking (the selector's claim rules); it holds the walk on the
        # selector's persistent command channel (``Selector.set_command``, shared with a command) until it arrives
        # or gives up, and steers the walk it gets (below)
        errand_out = self.errand.step(
            t_s,
            dt,
            odom,
            behaviour=self.behaviour,
            asleep=asleep,
            afraid=d.fear >= S.CLAIM_FEAR_MAX,
            homing=f[F["homing"]] >= S.HOMING_RELEASE or self.behaviour == "return_dock",
            docking=self.dock.mode in ("docking", "home"),
            lifted=bool(rx.lifted) or "lifted" in frame.events,
            can_walk=can_walk,
        )
        if errand_out.walking or errand_out.claim == WALK:
            self.cmd.hold(WALK, ERRAND_CLAIM)  # the walk is the errand's: it holds it, tick after tick
        else:
            self.cmd.drop_hold()  # (distracted: the body is somebody else's until it takes the walk again)
        if errand_out.claim is not None and errand_out.claim != WALK:
            self.sel.claim_next(errand_out.claim)  # (the look about on arriving: a one-shot)
        homing_i = self.sel.current if self.behaviour == "return_dock" else None
        sat_before = self.sel.sat[homing_i] if homing_i is not None else 0.0
        if self._recharging and f[F["homing"]] >= S.HOMING_RELEASE:
            # A previous arrival satisfies the trip, not the still-empty battery.
            home_i = self.sel.index("return_dock")
            if home_i is not None:
                self.sel.sat[home_i] = 0.0
        self.cmd.apply(self.sel)  # (one owner of the channel: a command's ask and an errand's hold, composed)
        chosen, new_bout = self.sel.step(
            dt,
            f,
            ctx,
            interrupted,
            consumed,
            release_commitment=attention_interrupt or self._walk_called,
            walking_to=self._walking_to,
        )
        if homing_i is not None and self.dock.mode == "docking" and consumed <= 0.0:
            # (dock:build, live check 2026-09-28) while its docking program has the legs, going
            # home does not tire of itself on the clock: it ends on its contacts, or when a try
            # fails (the program's own ``failed``), or when something else draws it more
            self.sel.sat[homing_i] = min(self.sel.sat[homing_i], sat_before)
        # (its tilt_before_preen habit is in the preen program itself: a tilt, then the preen)
        if new_bout and self.behaviour in THING_ACTS and chosen not in THING_ACTS:
            # (done with a thing and turning to something else: from one handling of it to the next
            # — the peck it came for — it stays)
            self._maybe_back_off(t_ms, t_s, odom, standing and can_walk)
        if (
            "blocked" in frame.events
            and self.behaviour in THING_ACTS | {"forage"}
            and standing
            and can_walk
        ):
            self._maybe_back_off(t_ms, t_s, odom, True)  # pushing into what it came to see
        if new_bout:
            # the after-effects the ending bout leaves (the comfort grammar's causes)
            ended = self.behaviour
            lasted = t_s - float(self._bout.get("t_s", t_s))
            if lasted >= SWITCH_AFTER_S and BEHAVIOURS[ended].family != BEHAVIOURS[chosen].family:
                tr["switching"] = 0.8  # a long act gives way to another kind: a head shake
            if ended in ("rest_sit", "doze", "nap", "melt") and lasted >= RESTED_AFTER_S:
                tr["rested"] = 1.0  # up from a good rest: a stretch
            need = NEED_OF.get(ended)
            if need is not None and lasted >= 5.0:
                used = float(self._bout.get("need0", 0.0)) - float(getattr(self.drives, need))
                if used > 0.2:
                    # a need well satisfied (a good preen, a good peck) feels good
                    self.drives.valence = min(1.0, self.drives.valence + 0.25 * used)
            self.behaviour = chosen
            if chosen in THING_ACTS and attended_thing is not None and attended_thing.track:
                self._attend_track = attended_thing.track
                self._attend_seen_s = t_s
            elif ended in THING_ACTS:
                self._attend_track = 0
                self._attend_seen_s = -1e9
            self.bout += 1
            self._bout_t_ms = t_ms
            self.counts[chosen] = self.counts.get(chosen, 0) + 1
            self._start_bout(chosen, f, pct, thing_bearing, remembered, odom, t_s)
        beh = self.behaviour
        spec = BEHAVIOURS[beh]

        # ── the docking program (dock:build): its own way into the dock, and out ──
        dout = self._dock_step(
            t_ms, frame, raw_events, tag_read, dock_cue, rx, hold, beh, spec, standing, can_walk
        )
        dock_drive = dout.drive
        # the dock is the one landmark the duck can place itself by: a tag read (or sitting on its contacts) says where it is in the
        # dock's frame, which pulls the lasting map's correction of the odometry
        dock_fix = self.dock.tracker.fix
        if dock_fix is not None and dock_fix is not self._lm_fix:
            self._lm_fix = dock_fix
            here = self.dock.pose((odom[0] * 1000.0, odom[1] * 1000.0, odom[2]))
            if self.lastmap.cfg.enabled and here is not None:
                self.lastmap.anchor_dock(odom, (here.x / 1000.0, here.y / 1000.0, here.h))

        # ── moments: short head acts played inside the current behaviour ─────
        self._moment_busy *= math.exp(-dt / MOMENT_BUSY_TAU_S)
        busy = (
            self._moment_busy > 0.3
            or hold is not None
            or asleep
            or not (sitting or standing)
            or new_bout
            # (fid:fix: the acts that hold the head still — 「定住盯著看」 with 「小動作：甩頭」 at the
            # same moment, the review: 20 of 29 holds had a moment playing inside)
            or beh in R.HOLDING_STILL
            # (dock:build) looking at the dock's tag, turning round, backing in: the head is busy
            or dout.quiet
            or t_ms < self._perform_until_ms  # an emotion plays
        )
        fired = self.sel.moment_step(
            dt, f, ctx, busy=busy, stepping=self._stepping, exclude=self._no_room(frame, beh)
        )
        if fired is not None:
            self.moment = fired
            self._moment_seq += 1
            self._moment_busy = 1.0
            self.counts[fired] = self.counts.get(fired, 0) + 1
        moment_now = self.moment if self._moment_busy > 0.3 else None
        still_now = (
            moment_now == "still" and self._moment_busy > 0.45
        ) or t_ms < self._perform_until_ms

        # ── posture ──────────────────────────────────────────────────────────
        want_before = self._posture_want
        if beh == "stand_up" or spec.posture == "stand":
            self._posture_want = "stand"
        elif beh == "sit_down" or spec.posture == "sit":
            self._posture_want = "sit"
        elif new_bout and (sitting or standing):
            # a behaviour that works either way keeps the posture it finds (no stray rises)
            self._posture_want = "sit" if sitting else "stand"
        if hold == "righting":
            self._posture_want = "stand"  # velstand at zero command rights the body
        arrived = False
        if dock_drive and dout.posture in ("sit", "stand"):
            self._posture_want = dout.posture  # the docking program's own stand and sit
        elif beh == "return_dock" and (self.dock.mode == "home" or frame.power.on_dock):
            self._posture_want = "sit"  # home, on its contacts: it sits and charges
        elif beh == "return_dock" and self.dock.mode == "idle":
            # (dock:build) no fix to dock by (the tag unread — a dark room, a side view — and no
            # remembered frame): it comes home to rest BESIDE its nest, as before, and tries to
            # dock another time
            dock_range = pct.nest.range_m if pct.nest is not None else None
            if dock_range is None and self.learn.dock is not None:
                dock_range = bearing_to(odom, self.learn.dock[0], self.learn.dock[1])[1]
            arrived = (
                dock_range is not None and dock_range < DOCK_REST_M and not self.dock.trusted()
            )
            if arrived:
                self._posture_want = "sit"  # settle beside the nest
                self._by_nest = t_s
                self._by_nest_walk = self._walk_s  # (until it walks off again)
        if self._posture_want != want_before:
            self._posture_tired = 1.0

        # ── locomotion: the duck's own rhythm, steered by the behaviour ────────
        goal, goal_range = self._goal(beh, pct, odom, t_s, liked, thing_bearing, remembered)
        errand_walk = errand_out.walking and beh == "wander"
        if errand_walk:
            goal, goal_range = errand_out.goal  # the place it was pointed to, in its own odometry frame
        if errand_out.arrived:
            arrived = True  # it is there: the walk stops this tick, then it looks about
        self.clutter.quiet = not (
            beh in ROUTE_ACTS
            or (
                beh in ("investigate", "follow")
                and (goal_range is None or goal_range > ROUTE_NEAR_THING_M)
            )
        )
        map_way = (
            self.clutter.waypoint
            if self.clutter.following and beh in WAYPOINT_ACTS and not dock_drive
            else None
        )
        if self.clutter.loosening(t_ms, odom) and beh in LOOSEN_ACTS and not dock_drive:
            # it keeps ending up in the same tangle: the pull of the thing it wants is let go for a moment — a walk home is then a walk
            # that looks for its nest (the seeking arcs and head sweeps), a walk to a thing the duck's own wandering
            goal, goal_range = None, None
        if (
            beh == "return_dock"
            and goal is not None
            and pct.nest is None
            and not dock_drive
            and self.dock.mode == "idle"
            and not self.dock.trusted()
        ):
            # (dock-home) at the lane point of a frame it carried too far to steer by, with no nest in view: the nest is not where the
            # walk could read it from, so it looks for it (the head sweeps and the search arcs, as with no frame) instead of standing
            # there; it keeps looking while it stays within NEST_LEAVE_M of that point (the arcs turn it round about it)
            lane = self.dock.rendezvous(
                (odom[0] * 1000.0, odom[1] * 1000.0, odom[2]), stale_ok=True
            )
            if lane is not None and lane[1] < (
                NEST_LEAVE_M if self._seek_t0 is not None else NEST_AT_LANE_M
            ):
                goal, goal_range = None, None
        # (dock:build) looking for its nest: a stop to sweep its head now and then (SEEK_*)
        sweep: float | None = None
        seek_arc = False
        if beh == "return_dock" and goal is None and not dock_drive and standing:
            if self._seek_t0 is None:
                self._seek_t0 = t_s
                self._seek_side = 1.0 if g.side >= 0 else -1.0
            into = (t_s - self._seek_t0) % SEEK_PERIOD_S
            sweep = next((b for b, until in SEEK_SWEEP if into < until), None)
            cycle = int((t_s - self._seek_t0) // SEEK_PERIOD_S) % SEEK_CYCLES
            seek_arc = sweep is None and cycle != SEEK_CYCLES - 1 and can_walk
        else:
            self._seek_t0 = None
        if beh == "return_dock" and goal is None and not dock_drive and pct.nest is None:
            self._seek_spent += dt
            if self._seek_spent >= SEEK_GIVE_UP_S:
                # a whole round of looking and no nest (SEEK_GIVE_UP_S): it lets going home go for a while
                rest = min(NEST_REST_S[1], NEST_REST_S[0] * 2.0**self._nest_rests)
                self._nest_rest_until = t_s + rest
                self._nest_rests += 1
                self._seek_spent = 0.0
        reach = REACH_M.get(beh)
        if beh == "follow":
            shy = (1.0 - g.axes["boldness"]) * (1.0 - 0.5 * g.axes["sociability"])
            reach = FOLLOW_REACH_M[0] + FOLLOW_REACH_M[1] * shy
        within = False  # a thing is as near as the beak needs: it stops walking at once …
        if beh == "investigate" and reach is not None:
            reach_range = self._reach_range(frame, pct.thing, remembered)
            within = reach_range is not None and reach_range < reach
            close_s = float(self._bout.get("near_s", 0.0)) + dt if within else 0.0
            self._bout["near_s"] = close_s
            arrived = within and close_s >= ARRIVE_HOLD_S  # … and says 到了 once that has held
        elif reach is not None and goal_range is not None and goal_range < reach:
            arrived = True  # near the person: it stops and looks
        # a walk up to a thing it has in mind is not done until it has got there (the selector's
        # ``walking_to``), unless it gets nowhere (APPROACH_STALL_S) or goes on far too long
        self._walking_to = False
        self._walk_called = False
        if beh == "investigate" and goal is not None and goal_range is not None and not arrived:
            bout = self._bout
            best_r = bout.get("best_r")
            if best_r is None or goal_range < float(best_r) - APPROACH_PROGRESS_M:
                bout["best_r"] = goal_range
                bout["best_t"] = t_s
            stall_s = APPROACH_STALL_S if goal_range > APPROACH_NEAR_M else APPROACH_NEAR_STALL_S
            stalled = t_s - float(bout.get("best_t", t_s)) > stall_s
            # a person who has just called or greeted it, or something that threatens it, comes
            # first (the same yield as a new thing's, NOVELTY_YIELDS_TO)
            called = max(S._feature(f, x) for x in NOVELTY_YIELDS_TO) >= NOVELTY_YIELD_AT
            # (the walk lets go of the body, not only of its hold on it: a call is answered)
            self._walk_called = called and not stalled
            budget_s = APPROACH_MAX_S * (0.1 + 0.9 * self._go_trait())
            self._walking_to = (
                not stalled
                and not called
                and t_s - float(bout.get("t_s", t_s)) < budget_s
                and not within
            )
        if errand_walk and goal is not None and goal_range is not None and not arrived:
            # a walk to a place somebody pointed at is a walk to a thing it has in mind: not done until it has got
            # there (the errand's own stall, budget and give-up end it), unless a call or a threat comes in
            called = max(S._feature(f, x) for x in NOVELTY_YIELDS_TO) >= NOVELTY_YIELD_AT
            self._walk_called = called
            self._walking_to = not called
        style = (
            LOCO.get(beh)
            if standing and spec.locomotes and not arrived and not within and can_walk
            else None
        )
        # a flight in panic: a threat to get away from, the legs in working order
        panic = (
            style is not None
            and beh in M.PANIC_ACTS
            and not rx.fallen
            and not rx.lifted
            and not rx.halt
        )
        go = 0.0
        top = NORMAL_VX_MAX
        wander = 1.0
        turn = 0.0
        pursue = 0.0
        hand = self._hand.step(dt)
        steer = None
        walk_goal = (
            goal  # the bearing the walk steers toward (a waypoint round a wall on the way home)
        )
        if map_way is not None:
            walk_goal = map_way[
                0
            ]  # (a route over the lasting map: head for its look-ahead point, not for the thing itself)
        detour_stepped = False
        if style is not None:
            if (
                beh == "return_dock"
                and goal is not None
                and goal_range is not None
                and not dock_drive
                and map_way is None
            ):
                walk_goal = self.detour.step(
                    t_ms,
                    self.bound,
                    odom,
                    frame,
                    goal,
                    goal_range,
                    escapes=self._unstick.fired,
                    free_side=self.bound.last.side,
                ).bearing
                detour_stepped = True
            self._dbg_goal = (goal, goal_range, walk_goal)
            # the walls and furniture it has seen bend the walk away before it touches them
            # (a goal it walks to — the thing, the person, the nest — is not a wall)
            steer = self.bound.steer(
                t_ms,
                odom,
                goal_bearing=walk_goal if goal_range is not None or map_way is not None else None,
                # (a bearing turned aside has no nest at its end: no point is "the nest")
                goal_range=None
                if (detour_stepped and self.detour.following) or map_way is not None
                else goal_range,
                goal_clear=(
                    GOAL_CLEAR_THING_M
                    if beh == "investigate" and goal is not None
                    else GOAL_CLEAR_M
                ),
            )
            if beh == "return_dock" and self.detour.following:
                # going round a wall it keeps the memory's bend away from what is ahead (the
                # follower's bearing only turns the nest's aside) but not the stop-and-look
                steer = replace(
                    steer,
                    cap=steer.cap if steer.urgency > 0.0 else FOLLOW_CAP,
                    look_ahead=False,
                )
            top, offset, wander = style
            pursue = self._pursuit(beh, goal, goal_range)
            top = self._pace(beh, top, goal_range, pursue)
            if panic:
                top, wander = NORMAL_VX_MAX, 0.0  # no wandering: it runs
            # A named thing is a goal-directed approach, not a continuation of the wander's
            # meander. Keep boundary/reflex steering below, but remove the exploratory curve so
            # the body first closes target-bearing error before it advances.
            if (beh == "investigate" and goal is not None) or map_way is not None or errand_walk:
                wander = 0.0
            drive = _clip(self.sel.sal[self.sel.current])
            # (a chosen walk walks: even a weakly wanted one moves about half its time —
            # pet:etho D1, pauses 30–60 % of a bout; the owner: 走太少)
            go = _clip(0.25 + 0.6 * drive + offset)
            go *= 0.4 + 0.6 * _clip(d.energy * 1.4)
            if beh == "return_dock" and self.detour.following:
                go = max(go, FOLLOW_GO)  # along a wall it keeps walking (fewer pauses)
            if errand_walk:
                go = max(go, ERRAND_GO)  # a walk to a place somebody pointed at walks on (fewer pauses)
            if panic:
                go = 1.0  # (a flight has no pauses and no tiring)
            if beh == "turn_around":
                if new_bout:
                    # toward the way it has SEEN open (the walls it saw), else away from what
                    # just stopped it, else its innate side (drifting, never a coin)
                    lean = (
                        steer.best
                        if steer.urgency > 0.15 and abs(steer.best) > 0.2
                        else self._turn_hint
                        if self._turn_hint
                        else float(steer.side)
                        if steer.side
                        else 0.45 * g.side + hand
                    )
                    self._turn_sign = 1.0 if lean >= 0.0 else -1.0
                    self._turn_flipped = False
                elif (
                    not self._turn_flipped
                    and steer.urgency >= 0.6
                    and abs(steer.best) > 0.3
                    and steer.best * self._turn_sign < 0.0
                ):
                    # the arc is carrying it toward something it now sees: the other way (once
                    # a bout — no dithering)
                    self._turn_sign = -self._turn_sign
                    self._turn_flipped = True
                turn = self._turn_sign * (0.75 + 0.2 * _clip(abs(hand)))
                goal = None
            elif beh == "follow" and goal is None:
                # called and nobody in sight: it goes round looking for them — an arc to its own
                # side (the head scanning wide), until it sees someone to walk to
                turn = 0.55 * (1.0 if 0.45 * g.side + hand >= 0.0 else -1.0)
            elif beh == "return_dock" and goal is None:
                # (dock:build) it wants its nest and neither sees nor remembers it: it goes round
                # looking for it (an arc to its own side, the head scanning) until the tag shows
                turn = 0.55 * (1.0 if 0.45 * g.side + hand >= 0.0 else -1.0)
            elif panic:
                # the tightest arc away from the threat until it faces away from it, then it
                # straightens and runs (the walls' bend and the reflexes' veer are added below)
                turn = 0.0
                if goal is not None and abs(goal) > PANIC_FACING_RAD:
                    turn = _clip(PANIC_TURN_GAIN * goal, -1.0, 1.0)
            else:
                turn = 0.25 * wander * self.learn.unvisited_bias(odom) * d.curiosity
            # Before closing on a thing well off the trunk axis, use the same legal walking
            # arc to bring it back toward centre. This keeps the object in view and prevents a
            # broad exploratory curve from reaching the boundary steering first; boundary and
            # reflex veers below remain authoritative.
            if (
                beh == "investigate"
                and goal is not None
                and abs(goal) > 0.3
                and t_ms <= self._attention_align_until
            ):
                turn = 0.9 if goal > 0.0 else -0.9
            if steer.veer and not (beh == "turn_around" and steer.veer * self._turn_sign < 0):
                # (2026-10-04, headless: a wall 0.1 m ahead, the reflex arc turned right at 0.9
                # rad/s and the wall memory's veer turned left at 0.9: they summed to ~0, a straight
                # walk the body link refuses within 30 cm of a wall — the duck stood at the wall
                # for a minute.) The reflexes now take the side the wall memory says (turn_side);
                # an escape's chosen side is the one authority, a bend against it is dropped.
                against = bool(committed and steer.veer * committed < 0.0)
                if not against:
                    veer = max(-1.0, min(1.0, veer + steer.veer))
            if committed and veer * committed < R.TURN_AWAY_VYAW:
                veer = committed * R.TURN_AWAY_VYAW  # an escape goes on turning away: an arc
        if not detour_stepped:
            self.detour.idle(t_ms)
        # owner A3: 0.40 m/s when coming up to a specific thing — the last stretch only (review
        # 2026-09-28: capped for the whole approach, 45 % of the explorer's walking was 0.40)
        approach = (
            beh == "investigate"
            and goal is not None
            and goal_range is not None
            and goal_range < 0.6
        )
        if beh == "return_dock" and goal_range is not None and goal_range < 0.6:
            approach = True
        cap = top if rx.vx_cap is None else min(top, rx.vx_cap)
        if steer is not None and steer.cap is not None:
            cap = min(cap, steer.cap)
        if beh == "turn_around" and not rx.halt and cap < ARC_VX_MIN:
            # a wall right ahead: the way out is an arc toward the freer side (turns are arcs,
            # never on the spot), at the slowest arc the gait allows
            cap = ARC_VX_MIN
            if rx.vyaw_bias:
                self._turn_sign = 1.0 if rx.vyaw_bias > 0.0 else -1.0
                turn = self._turn_sign * max(abs(turn), 0.8)
        # ── getting unstuck (owner 2026-10-04): its own request against its own progress ──
        self._unstick_step(
            t_ms,
            frame,
            rx,
            odom,
            active=standing
            and can_walk
            and not asleep
            and not dock_drive
            and not rx.lifted
            and not rx.fallen
            and self.dock.mode not in ("docking", "home"),
        )
        cornered_now = f[F["cornered"]] > 0.5 and beh != "return_dock" and not dock_drive
        if cornered_now and style is not None and steer is not None and not rx.halt:
            # it wants to walk and no forward arc has room: a small back arc out (owner
            # 2026-09-28), a few times here before it gives up and stops (fid:motor)
            if self._back_out_of_corner(t_ms, odom, steer):
                cornered_now = False
        # look before it goes on (fix 2026-09-28, headless: the camera gets no sharp frame
        # while the legs step and the body turns — it walked 4 s blind into the slot beside the
        # sofa): the way ahead unseen and light enough to see by, it stops a moment and looks
        looking = self._look_before(
            t_ms,
            frame,
            style is not None and steer is not None and steer.look_ahead and not dock_drive,
        )
        loco_in = M.LocoInput(
            go=go,
            bearing=None
            if panic
            else (walk_goal if goal is not None or map_way is not None else None),
            range_m=map_way[1] if map_way is not None else goal_range,
            approach=approach and not panic,
            panic=panic,
            # (dock:build: while the docking program has the legs the rhythm rests)
            halt=rx.halt
            or style is None
            or still_now
            or cornered_now
            or looking
            or dock_drive
            or sweep is not None,
            vx_cap=cap,
            vyaw_bias=veer
            + (
                self._touch_away * TOUCH_AVOID_BIAS
                if t_s < self._touch_away_until and (goal is None or panic) and not dock_drive
                else 0.0
            ),
            arousal=d.arousal,
            energy=d.energy,
            yaw_rate=frame.body.gyro[2],
            turn=turn,
            wander_scale=wander,
            pursue=pursue,
        )
        self._loco, lo = M.loco_step(self._loco_p, self._loco, loco_in, t_ms)
        vx, vyaw = lo.vx, lo.vyaw
        search_pivot = 0.0
        if seek_arc and not rx.halt:
            # (dock:build) looking for its nest: it turns round in a tight arc between the sweeps —
            # on a side that has room for it (2026-10-04: the blind arc ignored the ToF, the wall
            # memory and every reflex veer, and swept into the wall it was looking past: its
            # steering below stays on whenever a wall or a stop is near)
            way = self._seek_arc_way(odom, steer, rx)
            if way:
                vx, vyaw = legal_walk(SEEK_ARC[0], SEEK_ARC[1] * way)
        if self._look_for is not None and beh == "look_around" and not rx.halt:
            # a command's search for the person it heard: the head sweeps (the act's own program) and
            # the body turns in place in the seek's own tight arc, on a side that has room for it
            until, _side = self._look_for
            if t_ms >= until:
                self._look_for = None
            elif standing and can_walk and not asleep:
                if self.can_pivot:
                    # on the spot (the link runs the quick turn while ``pivot`` is asked): no room ahead needed
                    search_pivot = self._search_pivot(t_ms, odom, _side)
                    if search_pivot:
                        vx, vyaw = legal_walk(SEEK_ARC[0], SEEK_ARC[1] * search_pivot)
                else:
                    way = self._seek_arc_way(odom, steer, rx)
                    if way:
                        vx, vyaw = legal_walk(SEEK_ARC[0], SEEK_ARC[1] * way)
        if beh == "spin" and standing and can_walk and not asleep and not rx.halt:
            # the spin trick: the body turns on the spot (the link runs the quick turn while ``pivot`` is
            # asked; without it the seek's tight arc) until its own odometry has gone once round
            if self._spun < SPIN_RAD:
                way = 1.0 if int(self._bout.get("side", 1)) >= 0 else -1.0
                if self.can_pivot:
                    search_pivot = way
                vx, vyaw = legal_walk(SEEK_ARC[0], SEEK_ARC[1] * way)
        docking = False
        if dock_drive:
            # (dock:build) the docking program has the legs: its own twist (the J-hook, the
            # U-turn, the reverse in, the way out along the axis) through the same body path
            vx, vyaw = dout.vx, dout.vyaw
            docking = True
        backing = False
        if dock_drive:
            self._back = None  # (no curiosity back arc inside a docking)
        self._start_escape(t_ms, frame, rx, odom, standing and can_walk and not dock_drive)
        if self._back is not None and self._back_goal is not None:
            gx, gy, gm = self._back_goal
            if math.hypot(odom[0] - gx, odom[1] - gy) >= gm:
                self._back = None  # it has gone back as far as it meant to
        if self._back is None:
            self._back_goal = None
        # (an escape's back arc IS the answer to the bump: the bump reflex's own stop does not
        # cancel it — a startle, a fall or a lift still does)
        escaping = self._back is not None and self._back_goal is not None
        # ── going round in one place among things: one way out, kept to (neuro.clutter) ──
        if self._back is not None and not self._back_prev:
            self.clutter.note(t_ms, odom[0], odom[1], "back")
        self._back_prev = self._back is not None
        cdrive = None
        # (always stepped: a step that is skipped leaves an escape "on" with nobody to end it, and the older layers' escapes are held off)
        self._clutter_t = t_ms
        cdrive = self.clutter.step(
            t_ms,
            odom,
            active=standing
            and not dock_drive
            and can_walk
            and not asleep
            and not rx.lifted
            and not rx.fallen
            and not panic
            and not flight
            and not (rx.halt and rx.hold not in (None, "bump_stop"))
            and self.dock.mode not in ("docking", "home")
            and t_ms >= self._perform_until_ms,
            want=style is not None,
            bound=self.bound,
            lmap=self.lastmap,
            goal=(goal, goal_range) if goal is not None else None,
            held=rx.halt
            or (
                frame.tof.present
                and frame.tof.age_ms <= 500
                and (
                    bool(frame.tof.touch)
                    or (frame.tof.ahead_mm is not None and frame.tof.ahead_mm <= CLUTTER_NEAR_MM)
                )
            ),
        )
        if cdrive is not None:
            # it has picked its way out: the legs are the escape's (the older layers' own back arc and turn are dropped)
            self._back = None
            self._back_goal = None
            self._escape = None
            escaping = cdrive.back
        halted = rx.halt and not (escaping and rx.hold == "bump_stop")
        if cdrive is not None and not halted:
            vx, vyaw = legal_walk(cdrive.vx, cdrive.vyaw, back_arc=cdrive.back)
            backing = cdrive.back and vx < 0.0
        if self._back is not None:
            if t_ms >= self._back[0] or not standing or halted:
                self._back = None
                self._back_goal = None
            else:
                vx, vyaw = legal_walk(C_REVERSE_VX, self._back[1], back_arc=True)
                backing = vx < 0.0
        turn_to = None
        self._pivot = 0.0 if cdrive is None or halted else cdrive.pivot
        if search_pivot and cdrive is None and not halted:
            self._pivot = search_pivot
        # (a command's search for a person: a thing that catches its eye is not turned to; the search turns)
        searching_for = (self._look_for is not None and beh == "look_around") or beh == "spin"
        if not backing and not docking and cdrive is None and not searching_for:
            attend = self._person_b if beh in TURN_TOWARD_ACTS else None
            turn_to = self._turn_toward(
                t_ms, rx, beh, attend, standing and can_walk and not asleep, odom, vx != 0.0
            )
        if turn_to is not None:
            # the head has turned to it and the interest holds: the body follows, in an arc
            vx, vyaw = legal_walk(TURN_TOWARD_VX, turn_to, approach=True)
        if rx.escape is not None and standing:
            # a quick step away from a poke — from a touch in front a small back ARC (the hand
            # is where it would walk: reversing a small arc when cornered, owner 2026-09-28)
            back = rx.escape[0] < 0.0
            vx, vyaw = legal_walk(rx.escape[0], rx.escape[1], back_arc=back)
            backing = back and vx < 0.0
            docking = False  # (the reflex has the legs this moment)
        elif halted or t_ms < self._still_until_ms:
            vx, vyaw = 0.0, 0.0
            backing = False
            docking = False
        stepped_before = self._stepping
        self._stepping = vx != 0.0
        if self._stepping:
            self._last_step_s = t_s
        self._req_vx = vx
        self._walk_recent *= math.exp(-dt / WALK_RECENT_TAU_S)
        if stepped_before and not self._stepping and self._walk_recent >= WALKED_AFTER_S:
            tr["walked"] = 1.0  # a good walk ends: a shake
            self._walk_recent = 0.0
        if arrived and not self._bout.get("arrived") and beh in ("investigate", "follow"):
            self._bout["arrived"] = True
            if beh == "investigate":
                # at the thing: what it came for — it is satisfied (a walk is finished by arriving: the
                # natural end of the bout) and goes on to a peck or a nibble at it (the next tick)
                cur = self.sel.current
                self.sel.sat[cur] = max(self.sel.sat[cur], S.FINISH_SAT + S.FINISH_FADE)
                self.sel.commit_next("nibble" if int(self._bout.get("variant", 0)) & 1 else "peck")
            tr["reached"] = 1.0  # it got where it wanted: a proud little pose
            self.drives.valence = min(1.0, self.drives.valence + 0.08)

        # ── gaze, head, calls ────────────────────────────────────────────────
        body_k = REFLEX_HEAD_BY_POSTURE.get(posture, 0.0)
        self._seek_nest = beh == "return_dock" and goal is None and not dock_drive
        # (dock:build: walking the J-hook it looks about as on any walk — not locked on the goal)
        gaze_goal = None if dock_drive else goal
        homing_eyes = False
        if beh == "return_dock" and gaze_goal is not None:
            # (live check 2026-09-28) going home it keeps its eyes on its dock — where it sees it,
            # else where it believes it is (walking to the lane's far point, the dock is often
            # off to one side): a reading of the tag comes the first moment one can
            homing_eyes = True
            if pct.nest is None:
                believed = self.dock.dock_bearing(
                    (odom[0] * 1000.0, odom[1] * 1000.0, odom[2]), stale_ok=True
                )
                if believed is not None and abs(believed[0]) <= 1.0 and believed[1] <= DOCK_EYES_M:
                    gaze_goal = believed[0]
        gaze = self._gaze(dt, t_s, beh, gaze_goal, self._stepping, d, asleep)
        if homing_eyes and gaze_goal is not None and gaze.gain >= 0.8:
            gaze = Gaze(gaze_goal, self.dock.cfg.look_elevation, gaze.gain, gaze.speed)
        pursue = self._person_b if not asleep else None
        if rx.gaze is not None and not asleep:
            gaze = rx.gaze
            if pursue is not None and rx.active == "orient" and rx.stimulus == "visual:person":
                # it looks at a person who moves: the eyes pursue them (the onset's bearing
                # would stay where they were first seen)
                gaze = Gaze(pursue, gaze.elevation, gaze.gain, gaze.speed)
            if body_k == 0.0:  # getting up or sitting down: no saccade, a slow look at most
                gaze = Gaze(gaze.bearing, gaze.elevation, 0.4 * gaze.gain, min(0.3, gaze.speed))
        elif pursue is not None and beh in PURSUE_ACTS and body_k > 0.0:
            gaze = Gaze(pursue, 0.2, 0.85, 0.5)  # its eyes stay on the person it is about
        if sweep is not None and rx.gaze is None:
            gaze = Gaze(sweep, -0.1, 1.0, 0.6)  # looking for its nest: the sweep
        if dout.look is not None and rx.gaze is None:
            gaze = Gaze(dout.look[0], dout.look[1], 1.0, 0.6)  # a fix: it looks at the dock's tag
        elif dout.head_still and rx.gaze is None:
            # turning round, backing in: the head at HOME (the camera's own elevation there)
            gaze = Gaze(0.0, R.CAM_ELEVATION_HOME, 1.0, 0.4)
        if dout.event == "docked":
            self._vocal_pending = "coo"  # home: a soft coo
            self.drives.valence = min(1.0, self.drives.valence + 0.1)
        vocal = self._vocal(dt, beh, d, ap.alarm, down=rx.fallen)
        out_behaviour = HOLD_BEHAVIOUR.get(hold or "") or beh
        young = p.hatchling_wobble
        tempo = _clip(
            p.motor_tempo * (0.75 + 0.55 * d.arousal) * self._bout.get("tempo_k", 1.0), 0.5, 1.6
        )
        amplitude = _clip(
            p.amplitude_cap
            * (
                0.55
                + 0.3 * self._bout.get("intensity", 0.5)
                + 0.25 * (d.arousal - 0.4)
                + 0.1 * d.valence
            ),
            0.15,
            p.amplitude_cap,
        )
        alive = (
            0.0
            if asleep
            else _clip(
                p.head_alive * (0.55 + 0.8 * d.arousal) * (0.25 if hold == "freeze" else 1.0)
            )
        )
        head_busy = dout.head_still or dout.look is not None  # (dock:build)
        if head_busy:
            alive *= 0.3
        speed_level = _clip((vx - 0.35) / (NORMAL_VX_MAX - 0.35)) if vx > 0.0 else 0.0
        look = None
        if dout.look is not None:
            look = "fixate"
        elif beh == "return_dock" and pct.nest is None:
            look = "seek_nest"
        elif f[F["sound_nodir"]] > 0.4:
            look = "scan"
        elif self._stepping and place_novelty > 0.7:
            look = "glance_down"
        cmd = MotorCommand(
            t_ms=t_ms,
            behaviour=out_behaviour,
            bout=self.bout,
            variant=int(self._bout.get("variant", 0)),
            intensity=float(self._bout.get("intensity", 0.5)),
            tempo=tempo,
            amplitude=amplitude,
            side=int(self._bout.get("side", g.side)),
            target=self._bout_target(beh, goal, goal_range, pct, remembered),
            phase=self.sel.sat[self.sel.current],
            vx=vx,
            vyaw=vyaw,
            approach=approach and self._stepping,
            panic=bool(panic) and self._stepping,
            pivot=self._pivot if self._stepping and not backing else 0.0,
            docking=docking,
            posture=self._posture_want,
            gaze=gaze,
            head=rx.head,
            # (reflex.py already scales its pose by the posture: not a second time here)
            head_blend=rx.head_blend,
            stabilise=max(rx.stabilise, 0.85 if self._stepping and not head_busy else 0.0),
            # ducks waddle; they do not bob like pigeons (Necker 2007): a slow (≤ 1 Hz) head
            # sway locked to the waddle, bigger when it hurries and in the young (motor.py)
            bob=_clip((0.35 + 0.35 * speed_level) * (1.0 + 0.4 * young))
            if self._stepping and not head_busy
            else 0.0,
            alive=alive,
            vocal=vocal,
            vocal_seq=self._vocal_seq,
            vocal_level=_clip(
                0.3 + 0.5 * d.arousal + 0.2 * float(self._bout.get("intensity", 0.5))
            ),
            look=look,
            reflex=rx.active,
            reflex_track=rx.track,
            asleep=asleep,
            moment=self.moment,
            moment_seq=self._moment_seq,
            back_arc=backing,
        )
        self._reason = self._reason_now(out_behaviour, hold, f, rx.stimulus)
        if hold is None and (dock_drive or self.dock.mode == "home") and beh == "return_dock":
            words = dout.zh or D.PHASE_ZH.get(self.dock.phase, "")
            if words:
                self._reason = msg("neuro.reason.with_dock", reason=self._reason, dock_words=words)
        tel = self._telemetry(t_ms, t_s, out_behaviour, rx, lo, odom, moment_now)
        dock_view = self.dock.view(
            (odom[0] * 1000.0, odom[1] * 1000.0, odom[2]),
            t_ms,
            frame.power.on_dock,
            frame.power.charging,
        )
        locomotion = dict(tel.locomotion)
        locomotion["docking"] = 1.0 if dock_drive else 0.0
        return cmd, replace(tel, dock=dock_view, locomotion=locomotion)

    # ── pieces ───────────────────────────────────────────────────────────────

    def _no_room(self, frame: SensorFrame, beh: str) -> frozenset[str]:
        """(fid:fix 2026-09-28) The moments the head has no room for from where it IS (its own
        proprioception): a look down with the head already low (the review: 低頭看地上 seen in
        140/232 — missed mostly while foraging, pecking or preening), a look up with the head
        already raised. They wait until the head is somewhere they can be seen from."""
        up_room, down_room = R.nose_room(frame)
        out: set[str] = set()
        if down_room < MOMENT_ROOM_RAD or beh in HEAD_DOWN_ACTS:
            out.add("peek_down")
        if up_room < MOMENT_ROOM_RAD:
            out.update(("proud_puff", "perk_tilt"))
        return frozenset(out)

    # ── the dock (dock:build 2026-09-28) ─────────────────────────────────────

    def set_dock(self, side: str) -> None:
        """Which side of its dock the J-hook turns round on — how the family set the dock up
        (the home's layout: ``right``, or ``left`` where a wall stands on the dock's right, as
        in the apartment). Its own pose belief and memory are kept."""
        if side not in D.SIDES:
            raise ValueError(f"dock side {side!r} not in {D.SIDES}")
        self.dock.cfg = replace(self.dock.cfg, turn_side=side)

    def _at_capture(
        self, t_ms: int, frame: SensorFrame, odom: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """Its odometry at the capture of the newest camera pass (the pass reached it ``age_ms``
        ago, and was captured about DOCK_CAPTURE_MS before that): a bearing in a sighting is from the
        trunk as it was then — a duck that turns at half a radian a second turns 0.2–0.3 rad
        between the capture and the reading."""
        cam = frame.camera
        if not cam.usable:
            return odom
        capture = t_ms - int(cam.age_ms) - DOCK_CAPTURE_MS
        for when, where in reversed(self._odom_hist):
            if when <= capture:
                return where
        return odom

    def _fresh(
        self,
        sight: Sighting,
        at: tuple[float, float, float],
        odom: tuple[float, float, float],
        frame: SensorFrame,
    ) -> Sighting:
        """The sighting as seen from where the duck is now (bearing and range from its lens now):
        what the eye saw from where it was at the capture, carried by its own odometry."""
        if sight.source == "memory" or sight.range_m is None or at == odom:
            return sight
        lens = lens_point(frame)
        x, y = sighting_odom(at, lens, sight.bearing, float(sight.range_m))
        c, s = math.cos(odom[2]), math.sin(odom[2])
        dx, dy = x - odom[0], y - odom[1]
        px, py = c * dx + s * dy, -s * dx + c * dy
        return replace(
            sight,
            bearing=wrap_angle(math.atan2(py - lens[1], px - lens[0])),
            range_m=math.hypot(px - lens[0], py - lens[1]),
        )

    def _dock_reading(self, t_ms: int, frame: SensorFrame) -> tuple[D.TagRead | None, float | None]:
        """The dock as a camera pass that has just arrived saw it: a tag reading (range, bearing
        and facing from the trunk at the capture, with the odometry then) or a direction only
        (the dock's light, its panel). Each pass is read once."""
        odom = frame.body.odom
        hist = self._odom_hist
        hist.append((t_ms, odom))
        if len(hist) > 60:
            del hist[: len(hist) - 60]
        cam = frame.camera
        if not cam.usable:
            return None, None
        pass_ms = t_ms - int(cam.age_ms)
        if self._pass_ms is not None and abs(pass_ms - self._pass_ms) <= 5:
            return None, None  # the same pass as before
        self._pass_ms = pass_ms
        capture = pass_ms - DOCK_CAPTURE_MS
        at = odom
        for when, where in reversed(hist):
            if when <= capture:
                at = where
                break
        read: D.TagRead | None = None
        cue: float | None = None
        for sight in cam.sightings:
            if sight.kind != "nest":
                continue
            tag = D.sighting_read(sight, pass_ms, at)
            if tag is not None:
                read = tag
            else:
                cue = float(sight.bearing)
        return read, cue

    def _head_settled(self, t_ms: int, frame: SensorFrame) -> bool:
        """Its measured head has settled — still (each joint within 0.03 rad) for 0.3 s, looking
        ahead (yaw within 0.2 rad) — before it turns round or backs in (phase 3 did both with the
        head at HOME; the neural duck keeps its own carriage, which may hang a little low)."""
        hist = self._head_hist
        hist.append((t_ms, frame.body.head))
        while hist and t_ms - hist[0][0] > 400:
            hist.pop(0)
        if not hist or t_ms - hist[0][0] < 300 or abs(frame.body.head[2]) > 0.2:
            return False
        return all(
            max(h[1][i] for h in hist) - min(h[1][i] for h in hist) <= 0.03 for i in range(4)
        )

    def _dock_step(
        self,
        t_ms: int,
        frame: SensorFrame,
        raw_events: tuple[str, ...],
        tag_read: D.TagRead | None,
        cue: float | None,
        rx: Any,
        hold: str | None,
        beh: str,
        spec: Any,
        standing: bool,
        can_walk: bool,
    ) -> D.DockOut:
        """The docking program this tick: it takes the legs when ``return_dock`` has a fix to
        dock by, gives them back when the brain chooses something else, re-seats when the
        contact opened, and walks out along the axis when the duck wants to walk off."""
        upset = False
        if "lifted" in raw_events and "lay_down" not in raw_events:
            if frame.body.leg_load is None:
                upset = self._dock_carried = True
            elif self._dock_lift is None and not self._dock_carried:
                self._dock_lift = LiftWitness(t_ms)
        if self._dock_lift is not None and frame.body.leg_load is None:
            # the legs' load went unknown in the middle of the window: nothing to check the claim by any more - the same as a claim made
            # with no load (above): the frame is cleared, not kept on the strength of a half-read window (growth keeps its own witness)
            self._dock_lift = None
            upset = self._dock_carried = True
        if self._dock_lift is not None:
            witness = self._dock_lift
            verdict = witness.step(
                t_ms,
                frame.body.leg_load,
                ended=frame.body.fallen
                or "lay_down" in raw_events
                or any(e in raw_events for e in LIFT_DROP_EVENTS),
            )
            if verdict is not None:
                self._dock_lift = None
                if verdict == "yes":
                    upset = self._dock_carried = True
                elif "set_down" in raw_events and witness.light_max_ms >= LIFT_SHORT_MS:
                    # a short carry: the legs hung light for a moment (not the half second that tells a hold), then the floor again.
                    # A hand did lift it and put it down - the odometry frame is gone, whatever the half second said.
                    upset = True
        if "set_down" in raw_events and self._dock_carried:
            upset = True
            self._dock_carried = False
        odom = frame.body.odom
        din = D.DockIn(
            t_ms=t_ms,
            odom=(odom[0] * 1000.0, odom[1] * 1000.0, odom[2]),
            posture=frame.body.posture,
            pg=bool(frame.power.on_dock),
            applied=(frame.body.vx, frame.body.vyaw),
            blocked="blocked" in raw_events,
            tag=tag_read,
            cue=cue,
            halted=bool(rx.halt or rx.escape is not None or hold is not None),
            head_home=self._head_settled(t_ms, frame),
            upset=upset,
            fell="fell" in raw_events,
        )
        dock = self.dock
        recharging = (
            self._recharging
            and _at_charge_place(frame, dock)
            and not (beh in ("flee", "retreat") and self._charge_escape)
        )
        if (beh == "return_dock" or recharging) and can_walk and not rx.fallen and not rx.lifted:
            if dock.mode == "idle" and standing:
                why = dock.can_engage(din)
                if why is not None:
                    dock.engage(din, why)
            elif dock.mode == "home" and dock.contact_lost(t_ms):
                dock.reseat(din)
        elif dock.mode == "docking":
            dock.log.append(f"to:{beh}")
            dock.abort("switched")
        if (
            dock.mode == "home"
            and spec.locomotes
            and can_walk
            and beh != "return_dock"
            and not recharging
        ):
            dock.leave(din)  # it wants to walk off: up, and straight out along the axis
        out = dock.step(din)
        if out.event == "failed":
            self.drives.valence = max(-1.0, self.drives.valence - 0.05)
        self._dock_out = out
        return out

    def escape_command(self, cmd: MotorCommand) -> MotorCommand:
        """An escape outranks what a skill (carrying a toy home, walking into the sun, a ball) asked
        of the legs this tick: the skills merge their own twist after this brain has stepped, and
        pressed against the very thing they walk toward they would go on pushing (or, held by the
        body link a hand's breadth from a wall, ask for a straight walk it refuses). The back arc
        itself, and then — while it keeps turning away — an arc to the escape's side."""
        if self._back is not None and self._back_goal is not None and cmd.vx >= 0.0:
            vx, vyaw = legal_walk(C_REVERSE_VX, self._back[1], back_arc=True)
            return replace(cmd, vx=vx, vyaw=vyaw, approach=False, docking=False, back_arc=vx < 0.0)
        side = self._esc_side
        if (
            side
            and cmd.t_ms < self._esc_until
            and cmd.vx > 0.0
            and not cmd.docking
            and cmd.vyaw * side < R.TURN_AWAY_VYAW
        ):
            vx, vyaw = legal_walk(max(cmd.vx, ARC_VX_MIN), side * R.TURN_AWAY_VYAW)
            return replace(cmd, vx=vx, vyaw=vyaw, approach=False)
        return cmd

    def _search_pivot(self, t_ms: int, odom: tuple[float, float, float], side: float) -> float:
        """The side to pivot to this tick in a command's search (0: it stands and looks): a quick turn of
        :data:`SEARCH_PIVOT_RAD` by its own odometry, then :data:`SEARCH_LOOK_MS` of looking, and again."""
        step = self._search_step
        if step is None or (step[1] and t_ms >= step[1]):
            self._search_step = (odom[2], 0)
            return side
        yaw0, looking_until = step
        if looking_until:
            return 0.0
        if abs(wrap_angle(odom[2] - yaw0)) >= SEARCH_PIVOT_RAD:
            self._search_step = (odom[2], t_ms + SEARCH_LOOK_MS)
            return 0.0
        return side

    def _seek_arc_way(self, odom: tuple[float, float, float], steer: Any, rx: Any) -> float:
        """The side (+1 left, −1 right) a search arc may turn to, or 0 when none has room — a stop,
        a veer or a wall close ahead means the walk steers itself (the wall memory, the reflexes)."""
        if rx.vx_cap is not None and rx.vx_cap < ARC_VX_MIN:
            return 0.0
        if rx.vyaw_bias or (
            steer is not None and (steer.free_ahead < SEEK_ARC_ROOM_M or steer.cornered)
        ):
            return 0.0
        for way in (self._seek_side, -self._seek_side):
            if (
                min(self.bound.clearance(odom, way * h) for h in (0.0, 0.45, 0.9))
                >= SEEK_ARC_ROOM_M
            ):
                return way
        return 0.0

    def _novelty_interrupts(self, feat: list[float]) -> bool:
        """Whether a thing that has just come into view may break into the bout in hand: any
        ordinary bout — not sleep or a drowsy duck, not a reflex's act, not what is urgent business (a
        fear act), not while a person has just called or greeted it or something threatens it, not
        going home or a docking, not a skill's act in the middle of something it would spoil by
        stopping or one the head has a job in (the instrument, the dance), not the handling or the
        search of things on the floor (investigate, nibble, peck, forage), not a bout that began
        less than NOVELTY_MIN_BOUT_S ago, and not a thing farther off than NOVELTY_RANGE_M."""
        beh = self.behaviour
        if beh == "wander":
            return True
        spec = BEHAVIOURS[beh]
        if spec.sleep or spec.family == "reflex" or beh in NOVELTY_KEEPS:
            return False
        if beh in R.ENGAGED_ACTS:
            # (the instrument and the dance: the head has a job there, and a thing newly seen is noted,
            # not turned to — the sound skills' own rule, reflex.ENGAGED_ACTS)
            return False
        if feat[S.F["homing"]] >= S.HOMING_RELEASE or feat[S.F["sleepy"]] >= NOVELTY_SLEEPY_MAX:
            return False
        if self.sel.bout_age < NOVELTY_MIN_BOUT_S:
            return False
        if self._thing_last is not None and self._thing_last[1] > NOVELTY_RANGE_M:
            return False
        # a person who has just called or greeted it comes first (its turn toward them is the answer:
        # M5, a toy coming into view while it turned broke off 6 of the bold explorer's 21 displaced
        # response bouts, at 0.7-4.6 s), and so does anything that threatens it (a startle,
        # something looming, a touch that hurt)
        if max(S._feature(feat, x) for x in NOVELTY_YIELDS_TO) >= NOVELTY_YIELD_AT:
            return False
        if any(S._feature(feat, b) > 0.5 for b in S.BUSY_FEATURES):
            return False
        return not self.sel.urgent_now(feat)

    def _unstick_step(
        self,
        t_ms: int,
        frame: SensorFrame,
        rx: Any,
        odom: tuple[float, float, float],
        *,
        active: bool,
    ) -> None:
        """Feed unstick.py this tick's own senses; an escape that falls due waits for the reflex
        stop to end (:meth:`_start_escape`). A place it was blocked at goes into the wall memory
        and the learner at once."""
        tof = frame.tof
        tof_near = (
            tof.present
            and tof.age_ms <= 500
            and (
                tof.touch or (tof.ahead_mm if tof.ahead_mm is not None else 9999) <= UNSTICK_TOF_MM
            )
        )
        contact = any(e.stimulus in ("body:bump", "body:brushed", "tof:touch") for e in rx.events)
        cue = (
            contact
            or frame.touch.site is not None
            or tof_near
            or "blocked" in frame.events
            or "brushed" in frame.events
            or self.bound.last.free_ahead < UNSTICK_FREE_M
        )
        if contact:
            self.clutter.note(t_ms, odom[0], odom[1], "bump")
        esc = self._unstick.step(
            t_ms,
            odom,
            active=active and not self.clutter.active,
            # what it asks of its legs (a skill's own walk too: the integrated brain sets this to
            # the final command) or what the body is being given
            want_vx=max(self._req_vx, frame.body.vx),
            cue=cue,
            contact=contact,
            held=frame.body.posture == "stand" and abs(frame.body.vx) < 0.1,
            stationary=BEHAVIOURS[self.behaviour].stationary,
        )
        if esc is None:
            esc = self._low_block_escape(t_ms, frame, odom, active)
        if esc is not None:
            self.clutter.note(t_ms, odom[0], odom[1], "escape")
            self._escape = (esc, t_ms)
            self._give_up = esc.count >= GIVE_UP_AT
            if esc.at_obstacle:
                self.bound.note_block(t_ms, odom)
                self.lastmap.note_block(t_ms, odom)
            self.learn.bumped(odom)  # a bad place (a little): the learner remembers where

    def _low_block_escape(
        self, t_ms: int, frame: SensorFrame, odom: tuple[float, float, float], active: bool
    ) -> Escape | None:
        """The back-off from a low edge close ahead (see LOW_BLOCK_MM), or None."""
        tof = frame.tof
        if (
            not active
            or not tof.present
            or tof.age_ms > 500
            or tof.ahead_mm is None
            or tof.ahead_mm > LOW_BACK_MM
            or not LOW_BLOCK_MM[0] <= tof.low_edge_mm <= LOW_BLOCK_MM[1]
            or not LOW_BLOCK_MM[0] <= tof.ahead_top_mm <= LOW_BLOCK_MM[1]
            or t_ms - self._low_back_t < LOW_BACK_GAP_MS
            or self._escape is not None
            or self._back is not None
        ):
            return None
        if max(self._req_vx, frame.body.vx) < 0.1:
            return None  # it is not walking at it
        self._low_back_t = t_ms
        self._low_backs = [
            b
            for b in self._low_backs
            if t_ms - b[0] <= 45_000 and math.hypot(odom[0] - b[1], odom[1] - b[2]) <= 0.6
        ]
        self._low_backs.append((t_ms, odom[0], odom[1]))
        return Escape(
            back_ms=LOW_BACK_MS[0], turn_ms=LOW_BACK_MS[1], count=len(self._low_backs), why="low"
        )

    def _escape_side(self, frame: SensorFrame, count: int) -> int:
        """Which way an escape turns (+1 left, −1 right): away from what it feels and sees — the
        side it was touched on, else the side the ToF sees more room on (a wall at an angle: the
        far side is the way out), else toward the heading the wall memory says is best, else the
        side the camera saw freer, else — a second time at one place — the other way than last
        time, else its own side."""
        seen = self.bound.last
        site = frame.touch.site
        if site in ("left", "right"):
            return -1 if site == "left" else 1
        tof = frame.tof
        if tof.present and tof.age_ms <= 500:
            left = tof.left_mm if tof.left_mm is not None else 9999
            right = tof.right_mm if tof.right_mm is not None else 9999
            if abs(left - right) > 100:
                return 1 if left > right else -1
        if seen.urgency > 0.0 and abs(seen.best) >= 0.2:
            return 1 if seen.best > 0.0 else -1
        if seen.side:
            return int(seen.side)
        if count >= 2 and self._esc_side:
            return -self._esc_side
        return int(self.genome.side) or 1

    def _touch_away_update(self, t_s: float, frame: SensorFrame) -> None:
        """Remember which way to turn after a touch it felt while it walked (see :data:`TOUCH_AVOID_S`); its own sensing
        only: the touch site, the yaw rate it was turning at, no world truth."""
        site = frame.touch.site
        if site is None or t_s - self._last_step_s > 1.0:
            return
        if site in ("left", "right"):
            side = 1 if site == "left" else -1  # the thing is on that side
        elif abs(frame.body.vyaw) >= TOUCH_TURN_MIN:
            side = 1 if frame.body.vyaw > 0.0 else -1  # the thing is where it was turning to
        else:
            return
        self._touch_away = -side
        self._touch_away_until = t_s + TOUCH_AVOID_S

    def _fright_persists(self, t_s: float) -> bool:
        """The fright after a startle has not gone (its fear drive, or the thing still looming),
        for this duck: the bolder, the more fear it takes — or the startle was a strong scare
        (:data:`STRONG_SCARE`), which no temperament stands its ground to."""
        return (
            t_s - self._strong_scare_s <= STARTLE_FLIGHT_S
            or self.drives.fear >= STARTLE_FLIGHT_FEAR + 0.1 * self.genome.axes["boldness"]
            or self._feat[S.F["looming"]] >= STARTLE_FLIGHT_LOOM
        )

    def _go_trait(self) -> float:
        """0..1: how much this duck goes for what it wants (the bold and the lively do; the lazy and
        the shy go at it in their own time, as in ``_pace``)."""
        g = self.genome
        return _clip(0.5 * g.axes["boldness"] + 0.5 * g.axes["activity"])

    def _pursuit(self, beh: str, goal: float | None, goal_range: float | None) -> float:
        """0..1: how much it goes for a thing far ahead (see PURSUE_FAR_M): by boldness."""
        if beh != "investigate" or goal is None or goal_range is None:
            return 0.0
        trait = self._go_trait()
        if goal_range < PURSUE_FAR_M:
            # the last stretch: little stop-and-go while it is lined up on the thing and not there yet
            return PURSUE_NEAR * trait if abs(goal) <= PURSUE_NEAR_ALIGN_RAD else 0.0
        if abs(goal) > PURSUE_ALIGN_RAD:
            return 0.0
        return trait * trait

    def _pace(self, beh: str, top: float, goal_range: float | None, pursue: float = 0.0) -> float:
        """The top speed of an ordinary walk (see PACE_SLOW): by temperament, slowest beside the
        thing it investigates, up to the top of the band for a person it is falling behind."""
        if top >= PACE_FAST:
            g = self.genome
            trait = _clip(0.5 * g.axes["boldness"] + 0.5 * g.axes["activity"])
            top = PACE_SLOW + (PACE_FAST - PACE_SLOW) * trait
        if beh == "investigate" and goal_range is not None and goal_range < NEAR_THING_M:
            top = ARC_VX_MIN
        elif pursue > 0.0 and goal_range is not None and goal_range >= PURSUE_FAR_M:
            top += (NORMAL_VX_MAX - top) * pursue
        elif beh == "follow" and goal_range is not None:
            lo, hi = FOLLOW_CATCH_UP_M
            top += (NORMAL_VX_MAX - top) * _clip((goal_range - lo) / (hi - lo))
        return top

    def _start_escape(
        self,
        t_ms: int,
        frame: SensorFrame,
        rx: Any,
        odom: tuple[float, float, float],
        able: bool,
    ) -> None:
        """A due escape begins once the reflex stop is over: a small reverse ARC (never a straight
        reverse; owner: reversing allowed as curiosity's back_arc and when cornered) turning toward
        the open side — only where the walls it remembers leave room behind — and then it keeps
        turning away to that side, the reflexes and the wall memory all going the same way."""
        if self._escape is None:
            return
        esc, due = self._escape
        if t_ms - due > ESCAPE_WAIT_MS or not able:
            self._escape = None
            return
        if (rx.halt and rx.hold != "bump_stop") or self._back is not None:
            return
        self._escape = None
        side = self._escape_side(frame, esc.count)
        self._esc_side = side
        self._esc_corner = esc.count >= CORNER_AT
        self._esc_until = t_ms + esc.back_ms + esc.turn_ms
        self.bound.commit_side(side, t_ms)
        # (it cannot see behind it: only what it remembers — the way straight back.) The tail is
        # ~7 cm behind the trunk's origin, and a hand's breadth over: it backs as far as that
        # leaves, up to what the arc was meant to go (a shorter reverse beats none: stuck beside a
        # desk it had 17-20 cm and never backed at all, measured)
        room = min(self.bound.clearance(odom, h) for h in (math.pi - 0.15, math.pi, 0.15 - math.pi))
        goal = min(BACK_M_PER_S * esc.back_ms / 1000.0, room - ESCAPE_BACK_TAIL_M)
        if goal >= ESCAPE_BACK_MIN_M:
            # (calibration: 1.0 s of the reverse arc ≈ 8–12 cm back): the arc goes on until it has
            # gone back that far, at most the body link's hold (≈0.5 s) and a second more
            self._back = (t_ms + esc.back_ms + ESCAPE_BACK_SLACK_MS, float(side))
            self._back_goal = (odom[0], odom[1], goal)

    def _maybe_back_off(
        self, t_ms: int, t_s: float, odom: tuple[float, float, float], able: bool
    ) -> None:
        """Curiosity's back_arc when the thing it was busy with is right in front of it (seen a
        moment ago, close, nearly straight ahead): a small reverse arc turning its nose toward
        the side it saw open (else away from the thing, else its own side). At most
        BACKS_PER_PLACE before it has moved on."""
        if not able or self._back is not None or self._thing_last is None:
            return
        if math.hypot(odom[0] - self._back_at[0], odom[1] - self._back_at[1]) > BACK_RESET_M:
            self._backs = 0
        bearing, rng, seen_t = self._thing_last
        if t_s - seen_t > 3.0 or rng > BACK_NEAR_M or abs(bearing) > BACK_AHEAD_RAD:
            return
        if self._backs >= BACKS_PER_PLACE:
            return
        side = self.bound.last.side
        if side == 0:
            side = (-1 if bearing > 0.0 else 1) if abs(bearing) > 0.1 else self.genome.side
        self._back = (t_ms + BACK_ARC_MS, float(side))
        self._backs += 1
        self._back_at = (odom[0], odom[1])

    def _turn_toward(
        self,
        t_ms: int,
        rx: Any,
        beh: str,
        goal: float | None,
        able: bool,
        odom: tuple[float, float, float],
        walking: bool,
    ) -> float | None:
        """The body following the head (see TURN_TOWARD_*): the yaw rate of the arc toward it
        (+ = left) while it turns, else None. Only from standing (a walk steers itself), only
        for something seen (a mono microphone gives no side) or the target of an attentive act,
        never into a wall it has seen."""
        if self._turn_to is not None:
            heading, until = self._turn_to
            left = wrap_angle(heading - odom[2])
            if t_ms >= until or abs(left) < TURN_TOWARD_DONE_RAD or not able or rx.halt:
                self._turn_to = None
                return None
            if abs(left) >= PIVOT_MIN_RAD:
                self._pivot = 1.0 if left > 0.0 else -1.0
            return TURN_TOWARD_VYAW if left > 0.0 else -TURN_TOWARD_VYAW
        bearing = None
        if (
            rx.active == "orient"
            and rx.gaze is not None
            and str(rx.stimulus or "").startswith("visual:")
        ):
            bearing = rx.gaze.bearing
            if rx.stimulus == "visual:person" and self._person_b is not None:
                bearing = self._person_b  # where that person is now (they may be walking)
        elif beh in TURN_TOWARD_ACTS and goal is not None:
            bearing = goal
        if (
            bearing is None
            or abs(bearing) < TURN_TOWARD_RAD
            or not able
            or walking
            or rx.hold is not None
            or beh in TURN_TOWARD_NEVER
        ):
            self._looking_since = None
            return None
        if self._looking_since is None:
            self._looking_since = t_ms
            return None
        if t_ms - self._looking_since < TURN_TOWARD_AFTER_MS:
            return None
        way = 1.0 if bearing > 0.0 else -1.0
        room = min(self.bound.clearance(odom, way * h) for h in (0.0, 0.45, 0.9))
        if room < TURN_TOWARD_ROOM_M:
            return None  # a wall that way: the head alone keeps it in view
        self._turn_to = (wrap_angle(odom[2] + bearing), t_ms + TURN_TOWARD_MAX_MS)
        self._looking_since = None
        if abs(bearing) >= PIVOT_MIN_RAD:
            self._pivot = way
        return way * TURN_TOWARD_VYAW

    def _back_out_of_corner(self, t_ms: int, odom: tuple[float, float, float], steer: Any) -> bool:
        """A back arc out of a corner (owner 2026-09-28: reversing a small step/arc when
        cornered), turning toward the heading it saw clearest (else the freer side, else its own
        side) — at most CORNER_BACKS at one place. True while one is going (or just began)."""
        if self._back is not None:
            return True
        if math.hypot(odom[0] - self._corner_at[0], odom[1] - self._corner_at[1]) > BACK_RESET_M:
            self._corner_backs = 0
        if self._corner_backs >= CORNER_BACKS:
            return False
        # never back into a wall it has seen behind it (it cannot see behind: only memory)
        behind = min(self.bound.clearance(odom, h) for h in (math.pi - 0.4, math.pi, 0.4 - math.pi))
        if behind < CORNER_BACK_ROOM_M:
            return False
        best = float(getattr(steer, "best", 0.0))
        side = getattr(steer, "side", 0)
        turn = (1 if best > 0.0 else -1) if abs(best) > 0.2 else (side or self.genome.side or 1)
        self._back = (t_ms + CORNER_BACK_MS, float(turn))
        self._corner_backs += 1
        self._corner_at = (odom[0], odom[1])
        return True

    def _stand_s(self) -> float:
        return float(min(30, content.MOTION_STAGE[self.genome.stage]["stand_s"])) * (
            0.7 + 0.6 * self.genome.axes["activity"]
        )

    def _look_before(self, t_ms: int, frame: SensorFrame, unseen: bool) -> bool:
        """Whether it holds still to look ahead now: from the moment the way ahead is found
        unseen (the light good enough for a floor line) until a new floor line has come in
        (and at least LOOK_STOP_MIN_MS) — or LOOK_STOP_MAX_MS if none comes."""
        light_ok = frame.camera.usable and frame.camera.light in ("bright", "dim")
        seen_t = frame.camera.floor_t_ms if frame.camera.floor_t_ms is not None else -1
        if self._look_stop is None:
            if not (unseen and light_ok and self._stepping):
                return False
            self._look_stop = (t_ms, seen_t)
        since, before = self._look_stop
        looked = seen_t > before and t_ms - since >= LOOK_STOP_MIN_MS
        if looked or t_ms - since >= LOOK_STOP_MAX_MS or not light_ok:
            self._look_stop = None
            self._look_done = t_ms
            return False
        return True

    def _looking_at(self, frame: SensorFrame, bearing: float) -> bool:
        """Whether its camera looks that way now (the head's own yaw, measured)."""
        head_yaw = float(frame.body.head[2]) if len(frame.body.head) > 2 else 0.0
        return frame.camera.usable and abs(wrap_angle(bearing - head_yaw)) < LOOKING_AT_RAD

    def _consumed(
        self,
        dt: float,
        pct: Any,
        thing_near: float,
        frame: SensorFrame,
        novelty: float = 0.0,
        remembered: tuple[int, float, float, float] | None = None,
        odom: tuple[float, float, float] = (0.0, 0.0, 0.0),
        t_s_now: float = 0.0,
    ) -> float:
        """Satiation the current bout gets from its consequences this tick (the need it
        satisfies used up, the angle turned, the nest reached, the thing's novelty used up, a
        look where the thing or the person should be that finds nothing…) — never a clock
        (review 2026-09-28: fixed-rate give-ups)."""
        beh = self.behaviour
        extra = 0.0
        need = NEED_OF.get(beh)
        if need is not None:
            before = self._needs_before.get(need)
            now = float(getattr(self.drives, need))
            if before is not None and before > now:
                need0 = max(NEED_FLOOR, float(self._bout.get("need0", 1.0)))
                extra += NEED_SATIATES * (before - now) / need0
        if beh == "turn_around":
            extra += self._turned / math.pi
        elif beh == "spin" and self._spun >= SPIN_RAD:
            extra += 1.0  # the turn is made: the act is used up
        elif beh == self._over:
            extra += 1.0  # its command is over: the act is used up
        elif beh == "investigate" and thing_near > 0.5 and pct.thing is not None:
            # at the thing: satisfied as its novelty is used up (looking, nibbling at it)
            prev = self._bout.get("novelty_prev")
            if prev is not None and novelty < float(prev):
                novelty0 = max(0.2, float(self._bout.get("novelty0", 1.0)))
                extra += INVESTIGATE_USE * (float(prev) - novelty) / novelty0
            self._bout["novelty_prev"] = novelty
        elif beh == "investigate" and pct.thing is None:
            # it looks where it believed the thing was and does not see it: it is gone
            if (
                remembered is not None
                and remembered[2] > BLIND_NEAR_M
                and not self._stepping
                and self._looking_at(frame, remembered[1])
            ):
                extra += dt / ABSENT_S
        elif beh == "return_dock" and (frame.power.on_dock or t_s_now - self._by_nest < 1.0):
            extra += 1.0  # home: on its contacts (or resting beside its nest)
        elif beh == "return_dock" and self._dock_out.event == "failed":
            # (dock:build) the try did not work: it gives up for now — unless its pack runs low:
            # then it tries again (live check 2026-09-28: two failed tries and it gave up,
            # stranded below 30 %)
            extra += 0.35 * (1.0 - _homing(frame))
        elif beh == "follow" and pct.person is None:
            # it looks where the person was and nobody is there: they have gone
            if self._person is not None:
                where = bearing_to(odom, self._person[0], self._person[1])[0]
                if self._looking_at(frame, where):
                    extra += dt / ABSENT_S
            elif self._traces.get("name", 0.0) < 0.1:
                extra += dt / ABSENT_S  # nobody seen, nobody calling: nothing to follow
        elif beh == "nibble" and pct.thing is None:
            extra += dt / 2.0  # nothing at its beak to nibble
        return extra

    def _start_bout(
        self,
        beh: str,
        f: list[float],
        pct: Any,
        thing_bearing: float | None,
        remembered: tuple[int, float, float, float] | None,
        odom: tuple[float, float, float],
        t_s: float,
    ) -> None:
        g = self.genome
        sal = self.sel.sal[self.sel.current]
        intensity = _clip(0.25 + 0.75 * sal)
        side = g.side
        bearing = None
        if beh in ("investigate", "nibble", "peck"):
            bearing = (
                thing_bearing
                if thing_bearing is not None
                else (remembered[1] if remembered else None)
            )
        elif TARGETED.get(beh) == "person" and pct.person is not None:
            bearing = pct.person.bearing
        elif beh in ("retreat", "flee"):
            bearing = self.appraiser.threat_bearing(odom)
        if bearing is not None and abs(bearing) > 0.15:
            side = 1 if bearing > 0 else -1
            if beh in ("glance_away", "retreat", "flee", "tuck_away"):
                side = -side
        need = NEED_OF.get(beh)
        novel_now = self._traces.get("novel", 0.0)
        thing_key = None
        if beh == "investigate":
            thing_key = (
                self._thing_key
                if pct.thing is not None
                else (remembered[0] if remembered is not None else None)
            )
        self._bout = {
            "thing_key": thing_key,
            "novelty0": novel_now,
            "intensity": intensity,
            "side": side,
            "variant": self._stream.u64() & 0x7FFFFFFF,
            "tempo_k": 0.9 + 0.2 * intensity,
            "t_s": t_s,
            "need0": float(getattr(self.drives, need)) if need is not None else 1.0,
        }
        self._spun = 0.0
        self._over = None
        if beh in ("greet", "call_out", "wake_ritual", "little_song"):
            self._vocal_urge = 1.0  # these open with a call
        self._phrase_left = self.genome.p.phrase_len if beh in ("little_song", "self_duet") else 0
        self._reason_bout = self.sel.why(beh, f)

    def _reach_range(
        self,
        frame: SensorFrame,
        thing: Sighting | None,
        remembered: tuple[int, float, float, float] | None,
    ) -> float | None:
        """How far the thing's near edge is from the trunk's centre, m — what the beak's reach is
        measured against — from the duck's own senses: what it sees now (the eye's range starts at
        the lens: carried over the lens's offset; a belief the brain made from its odometry is
        already the trunk's own), else where it believes the thing is. None: no range to go by."""
        if thing is not None:
            if thing.range_m is None:
                return None
            if thing.source == "memory":
                return float(thing.range_m)
            return range_from_trunk(lens_point(frame), thing.bearing, float(thing.range_m))
        if remembered is not None:
            return float(remembered[2])
        return None

    def _goal(
        self,
        beh: str,
        pct: Any,
        odom: tuple[float, float, float],
        t_s: float,
        liked: tuple[float, float, float] | None,
        thing_bearing: float | None,
        remembered: tuple[int, float, float, float] | None,
    ) -> tuple[float | None, float | None]:
        """Where the current behaviour wants to go — (bearing from the trunk, range m) — from the
        senses or from where it believed something was (odometry)."""
        if beh == "investigate":
            if thing_bearing is not None and pct.thing is not None:
                return thing_bearing, pct.thing.range_m
            if remembered is not None:
                return remembered[1], remembered[2]
            return None, None
        if beh == "follow":
            if pct.person is not None:
                return float(pct.person.bearing), pct.person.range_m
            if self._person is not None and t_s - self._person[2] < 20.0:
                return bearing_to(odom, self._person[0], self._person[1])
            return None, None
        if beh in ("retreat", "flee"):
            threat = self.appraiser.threat_bearing(odom)
            if threat is not None:
                return wrap_angle(threat + math.pi), None
            return None, None
        if beh == "return_dock":
            if self.dock.mode == "home":
                # (dock:build) in its dock by its own belief it is home: nowhere to walk to (a
                # walk from between the guides toward the lane swept it over one, headless)
                return None, None
            if pct.nest is not None:
                return float(pct.nest.bearing), pct.nest.range_m
            # (dock:build) the dock's frame it remembers: to the lane's far point in front of
            # the dock, where the tag can be read square on. (dock-home) A frame carried past
            # remembered_max_m still points the way there: the old walk went straight at the
            # nest's place from wherever it was, and from beside the dock the tag cannot be read
            # (a tag is read from within 60° of its face; headless the real 1001's 60 samples within
            # 2.8 m were all at 62-88°). The legs are taken only by a tag read, not by that frame.
            rv = self.dock.rendezvous((odom[0] * 1000.0, odom[1] * 1000.0, odom[2]), stale_ok=True)
            if rv is not None:
                return rv
            if self.learn.dock is not None:
                return bearing_to(odom, self.learn.dock[0], self.learn.dock[1])
            return None, None
        if beh == "wander" and liked is not None and self.drives.social > 0.6:
            return liked[0], liked[1]
        return None, None

    def _bout_target(
        self,
        beh: str,
        goal: float | None,
        goal_range: float | None,
        pct: Any,
        remembered: tuple[int, float, float, float] | None = None,
    ) -> Target | None:
        kind = TARGETED.get(beh)
        if kind is None:
            return None
        if kind == "thing" and pct.thing is not None:
            return Target("thing", pct.thing.bearing, pct.thing.range_m, pct.thing.track)
        if kind == "person" and pct.person is not None:
            return Target("person", pct.person.bearing, pct.person.range_m, pct.person.track)
        if kind == "nest" and pct.nest is not None:
            return Target("nest", pct.nest.bearing, pct.nest.range_m, pct.nest.track)
        if kind == "motion" and pct.thing is not None:
            return Target("thing", pct.thing.bearing, pct.thing.range_m, pct.thing.track)
        if goal is not None:
            if kind == "threat":
                # what it flees from, as the eye gave it (0: it did not see it)
                return Target("threat", goal, goal_range, self.appraiser.threat_track)
            # a thing it remembers but does not see: the place it believes it is, named by the
            # eye's last track for it (0 for any other place)
            track = (
                self.learn.track_of(remembered[0])
                if kind == "thing" and beh == "investigate" and remembered is not None
                else 0
            )
            return Target("place", goal, goal_range, track)
        return None

    def _gaze(
        self,
        dt: float,
        t_s: float,
        beh: str,
        goal: float | None,
        stepping: bool,
        d: Drives,
        asleep: bool,
    ) -> Gaze:
        """Where attention points when no reflex holds it: the behaviour's target when it has
        one; else a bird's shift-and-hold gaze cycle (ethology A1): an integrate-and-fire urge
        with a log-normal threshold gives log-normal intervals (median ~1.5 s calm, ~0.9 s
        alert, 3–4 s drowsy; CV ~0.6 — never a metronome), mostly small shifts with a heavy
        tail, never straight back to where it just looked."""
        if asleep:
            return Gaze(0.0, -0.3, 0.0, 0.1)
        if beh in R.HOLDING_STILL or (self.moment == "still" and self._moment_busy > 0.45):
            # (fid:fix) held still — 「定住」, 「定格」: no gaze shifts inside it (the review: a
            # freeze of 40 s with 8 visible head moves)
            return Gaze(self._fix_b, self._fix_e, 0.0, self._gaze_speed)
        alert = _clip(0.5 + 1.2 * (d.arousal - 0.4))
        median = GAZE_MEDIAN_S * math.exp(-1.1 * (alert - 0.5)) * (1.0 + 1.6 * d.sleepiness)
        median /= self.genome.p.motor_tempo * (0.85 + 0.3 * d.curiosity)
        if self._seek_nest:
            median /= 1.5  # (dock:build) looking for its nest: it looks about more often
        if beh in ("rest_sit", "doze") or (not stepping and self._posture_want == "sit"):
            median *= 1.25
        self._gaze_urge += dt / median
        if self._gaze_urge >= self._gaze_theta:
            self._gaze_urge = 0.0
            self._gaze_theta = math.exp(0.6 * self._stream.normal())
            self._saccade(t_s, stepping, d)
        # a fixation is HELD still until the next shift (pet:etho A1: shift-and-hold)
        b = self._fix_b
        e = self._fix_e
        elevation = GAZE_ELEVATION.get(beh)
        if goal is not None and beh in TARGETED:
            return Gaze(goal, elevation if elevation is not None else 0.15, 0.8, 0.6)
        if beh in HEAD_ACTS:
            return Gaze(b, e, 0.0, self._gaze_speed)  # the act's own program has the head
        if (stepping or self._look_stop is not None) and self.bound.last.look_ahead:
            # the way ahead is not seen clear: it looks where it is going (the floor ahead)
            return Gaze(0.2 * b, -0.12, 0.8, 0.6)
        if elevation is not None:
            return Gaze(0.4 * b, elevation + 0.3 * (e - 0.05), 0.6, 0.5)
        if stepping:
            return Gaze(b, e, 0.45, self._gaze_speed)
        return Gaze(b, e, 0.35, self._gaze_speed)

    def _saccade(self, t_s: float, stepping: bool, d: Drives) -> None:
        """The next fixation: a log-normal shift (median ~13°, ~5 % beyond 45°) to whichever
        side is inside the envelope and not looked at in the last 4 s (inhibition of return)."""
        span = 0.5 if stepping else 1.0  # rad: walking keeps the looks narrower
        seeking = self._traces.get("name", 0.0) > 0.2 and self._person is None
        if seeking or self._seek_nest:
            span = 1.0  # called and it has not seen who: it looks about widely
        size = min(1.3, 0.22 * math.exp(0.7 * self._stream.normal()))
        if self._seek_nest:
            # (dock:build) looking for its nest: big shifts across the whole range
            size = min(1.3, 0.5 * math.exp(0.4 * self._stream.normal()))
        best_cost = math.inf
        target = self._fix_b
        for sign in (1.0, -1.0):
            candidate = self._fix_b + sign * size
            cost = 3.0 * max(0.0, abs(candidate) - span) + 0.4 * self._stream.uniform()
            if not stepping and abs(candidate) < ONE_EYE_RAD:
                cost += 0.8  # at rest: one eye to the side, not straight ahead
            for bearing, seen in self._fixations:
                if t_s - seen < 4.0 and abs(candidate - bearing) < 0.15:
                    cost += 1.0
            if cost < best_cost:
                best_cost, target = cost, candidate
        self._fix_b = _clip(target, -span, span)
        self._fix_e = _clip(0.06 + 0.07 * self._stream.normal() - 0.12 * d.sleepiness, -0.25, 0.3)
        recent = [(b, t) for b, t in self._fixations if t_s - t < 5.0][-3:]
        self._fixations = [*recent, (self._fix_b, t_s)]
        self._gaze_speed = _clip(0.35 + 0.6 * (d.arousal - 0.3), 0.25, 0.9)

    def _vocal(
        self, dt: float, beh: str, d: Drives, alarm: bool, *, down: bool = False
    ) -> str | None:
        """Calls come from an integrate-and-fire urge set by the behaviour and the mood."""
        g = self.genome
        self._answer *= math.exp(-dt / ANSWER_TAU_S)
        if alarm:
            self._vocal_seq += 1
            return "alarm"
        if down:
            # still down: the urge to call for help builds (faster once the body's own tries
            # have failed, and when afraid) and fires now and then
            self._help_urge += dt * (0.06 + 0.1 * d.fear + (0.25 if self._righting_failed else 0.0))
            if self._help_urge >= 1.0:
                self._help_urge = 0.0
                self._vocal_seq += 1
                return "alarm"
            return None
        spec = BEHAVIOURS[beh]
        tags = [t for t in spec.sounds if t and t != "alarm"]
        for ou in self._tag_ou.values():
            ou.step(dt)
        if spec.sleep and beh != "sleep_talk":
            return None
        if self._answer >= 1.0 and beh not in BEAK_BUSY:
            # it answers the one who called it (a greeting for a hello, a questioning peep for
            # its name) — the first call for a social duck, a second one for an aloof one
            self._answer = 0.0
            self._vocal_seq += 1
            return self._answer_tag
        if self._feat[S.F["rhythm_slow"]] > 0.3:
            return None  # a slow knocking train: a duckling freezes and falls silent (Miller)
        if not tags:
            return self._contact_call(dt, beh, d)
        silent = "" in spec.sounds
        rate = {
            "call_out": 0.15 + 0.25 * d.social,
            "little_song": 1.2,
            "self_duet": 1.0,
            "call_response": 0.5 * (self._traces.get("voice", 0.0) + 0.2),
            "sleep_talk": 0.4,
        }.get(
            beh,
            (0.03 if silent else 0.08)
            * g.p.vocal_rate
            * (0.5 + d.arousal)
            * (1.0 + max(0.0, d.valence)),
        )
        if self._vocal_pending is None:
            self._vocal_urge += dt * rate
        if self._vocal_urge < 1.0 and self._vocal_pending is None:
            return None
        self._vocal_urge = 0.0
        if self._vocal_pending is not None:
            tag = self._vocal_pending
            self._vocal_pending = None
        else:
            tag = self._tag_by_context(tags, beh, d)
            if beh == "self_duet":
                tag = "inquire" if self._vocal_seq % 2 else "chirp"
            if (
                g.habit == "extra_chirp"
                and d.valence > 0.3
                and beh in S.PLAY | {"greet", "happy_sway"}
            ):
                self._vocal_pending = "chirp"
                self._vocal_urge = 0.0
        if self._phrase_left > 0:
            self._phrase_left -= 1
            self._vocal_urge = 0.7 if self._phrase_left > 0 else 0.0
        self._vocal_seq += 1
        return tag

    def _contact_call(self, dt: float, beh: str, d: Drives) -> str | None:
        """Between its acts a duck keeps up soft contact calls (ethology C1: 0.5–3 a minute
        alone, more when it misses its family) — unless its beak is busy or it keeps still."""
        if beh in BEAK_BUSY or self._feat[S.F["startled"]] > 0.2:
            return None
        rate = self.genome.p.vocal_rate * (0.004 + 0.02 * d.social + 0.012 * self._alone)
        self._contact_urge += dt * rate * (0.6 + d.arousal)
        if self._contact_urge < 1.0:
            return None
        self._contact_urge = 0.0
        soft: list[str] = [t for t in ("chirp", "coo", "inquire") if t in self.genome.vocal_weights]
        tag = self._tag_by_context(soft, beh, d)
        self._vocal_seq += 1
        return tag

    def _tag_by_context(self, tags: list[str], beh: str, d: Drives) -> str:
        """Which call: what is going on sets it (a greeting for an arrival, a questioning peep
        at something new, a soft coo when comfortable or touched, a chirp when happy or missing
        its family, a wheee at play), its favourite sound only leans it a little — sampled, never
        the same argmax every time (review 2026-09-28: 76–81 % of the explorer's calls were its
        favourite "inquire")."""
        tr = self._traces
        f = self._feat
        context = {
            "greet": 1.0 + 3.0 * tr.get("person_new", 0.0) + 2.0 * tr.get("name", 0.0),
            "inquire": 1.0 + 2.0 * tr.get("novel", 0.0) + 1.0 * d.curiosity,
            "chirp": 1.0 + 1.5 * max(0.0, d.valence) + 1.0 * d.social,
            "coo": 1.0 + 2.0 * d.comfort * f[S.F["touch_now"]] + (1.5 if beh in COO_ACTS else 0.0),
            "wheee": 0.2 + (3.0 * d.play if beh in S.PLAY else 0.0),
        }
        weights = self.genome.vocal_weights
        total = 0.0
        scored: list[tuple[str, float]] = []
        for tag in tags:
            lean = math.sqrt(weights.get(tag, 1.0))  # the favourite: a mild lean
            ou = math.exp(0.5 * self._tag_ou[tag].x) if tag in self._tag_ou else 1.0
            w = lean * ou * context.get(tag, 1.0)
            scored.append((tag, w))
            total += w
        if not scored:
            return "chirp"
        x = self._stream.uniform() * total
        for tag, w in scored:
            x -= w
            if x <= 0.0:
                return tag
        return scored[-1][0]

    def _reason_now(
        self, beh: str, hold: str | None, f: list[float], stimulus: str | None = None
    ) -> str:
        zh = behaviour_zh(beh)
        if hold is not None:
            what = stimulus_zh(stimulus)
            cause = msg("neuro.reason.cause_prefix", cause=what) if what else ""
            return msg(
                "neuro.reason.reflex", act=zh, cause=cause, reflex_name=REFLEXES.get(hold, hold)
            )
        return msg("neuro.reason.chosen", act=zh, why=self.sel.why(beh, f))

    # ── telemetry ────────────────────────────────────────────────────────────

    def home_known(self) -> dict[str, Any] | None:
        """(MAP-EXPLORE) What its own map has seen of its home, as last worked out (``LastMap.home_view``), or None (no map yet)."""
        return None if self._home is None else dict(self._home)

    def _telemetry(
        self,
        t_ms: int,
        t_s: float,
        beh: str,
        rx: R.ReflexOutput,
        lo: M.LocoOut,
        odom: tuple[float, float, float],
        moment: str | None = None,
    ) -> BrainTelemetry:
        if t_s - self._slow_t >= TELEMETRY_SLOW_S:
            self._slow_t = t_s
            learned = self.learn.view(odom, self.genome.p.leash_m)
            learned["after_effects"] = self.appraiser.traces()
            if self.lastmap.cfg.enabled:
                # (MAP-EXPLORE) how much of its home its own map has seen: the growth input and the 成長 view (never the true layout)
                if self._home is None or t_s - self._home_t >= HOME_EVERY_S:
                    self._home_t = t_s
                    self._home = self.lastmap.home_view()
                learned["home"] = dict(self._home)
            self._slow = {
                "habituation": self.hab.levels(t_s),
                "sensitisation": self.hab.sensitisations(t_s),
                "learned": learned,
            }
        d = self.drives
        drives = d.as_dict()
        drives["sleepiness"] = d.sleepiness
        drives["stiffness"] = d.stiffness
        ranking = self.sel.ranking(6)
        competition = tuple(
            Candidate(name, round(sal, 3), round(act, 3), self.sel.why(name, self._feat, 1))
            for name, sal, act in ranking
        )
        locomotion = {k: float(v) for k, v in lo.telemetry().items()}
        locomotion["stepping"] = 1.0 if self._stepping else 0.0
        locomotion["walk_s"] = round(self._walk_s, 1)
        seen = self.bound.last  # the walls it has seen, as the last steering update had them
        locomotion["wall_free_m"] = round(min(self._steer_free, 9.9), 2)
        locomotion["wall_urgency"] = float(seen.urgency)
        locomotion["wall_veer"] = float(seen.veer)
        locomotion["wall_points"] = float(len(self.bound.points))
        locomotion["floor_looks"] = float(self.bound.looks)
        locomotion["stuck"] = round(min(1.0, self._stuck / STUCK_AT), 2)
        locomotion["back_arc"] = 1.0 if self._back is not None else 0.0
        locomotion["escapes"] = float(self._unstick.fired)
        locomotion["errand"] = 1.0 if self.errand.active else 0.0  # somebody pointed at a place and it is on the way
        # (the clutter answer: 1 = a route out has the legs, 2 = the pull of what it wants is let go; and how many it has begun)
        locomotion["clutter_escape"] = (
            1.0
            if self.clutter.active
            else 2.0
            if self.clutter.loosening(t_ms)
            else 3.0
            if self.clutter.following
            else 0.0
        )
        locomotion["clutter_escapes"] = float(self.clutter.started + self.clutter.loosened)
        if self.lastmap.cfg.enabled:
            # (the lasting map: how many places it knows, how many docks fixed its odometry, and the route's own count; telemetry only)
            locomotion["lastmap_cells"] = float(len(self.lastmap.cells))
            locomotion["lastmap_anchors"] = float(self.lastmap.anchors)
            locomotion["lastmap_routes"] = float(self.clutter.replans)
        # (2026-10-05, the trained skills' inputs; telemetry only: the motor commands are untouched) the room
        # its memory of the walls leaves straight behind it, and the side to turn to on the spot when an
        # escape has failed at the same place three times and the back arc is over (a corner)
        locomotion["room_behind_m"] = round(
            min(self.bound.clearance(odom, h) for h in (math.pi - 0.15, math.pi, 0.15 - math.pi)),
            2,
        )
        # (2026-10-10, the startle's back-steps; telemetry only) where the last scare came from, from the trunk now (rad, + = left),
        # and how long ago it was heard / seen / felt: absent when the duck never knew
        threat = self.appraiser.threat_bearing(odom)
        if threat is not None:
            locomotion["threat_bearing"] = round(threat, 3)
            locomotion["threat_age_s"] = round(max(0.0, t_s - self.appraiser.threat_t), 2)
        if self._touch_back_s > -1e8:
            locomotion["touch_behind_age_s"] = round(max(0.0, t_s - self._touch_back_s), 2)
        locomotion["corner_turn"] = float(
            self._esc_side
            if self._esc_corner and self._back is None and t_ms < self._esc_until
            else 0
        )
        return BrainTelemetry(
            t_ms=t_ms,
            behaviour=beh,
            behaviour_zh=behaviour_zh(beh),
            reason_zh=self._reason,
            bout=self.bout,
            bout_ms=t_ms - self._bout_t_ms,
            satiation=round(self.sel.sat[self.sel.current], 3),
            drives={k: round(v, 3) for k, v in drives.items()},
            arousal=round(d.arousal, 3),
            valence=round(d.valence, 3),
            reflex=rx.active,
            reflexes=tuple(rx.recent[-8:]),
            competition=competition,
            habituation=self._slow["habituation"],
            sensitisation=self._slow["sensitisation"],
            learned=self._slow["learned"],
            locomotion=locomotion,
            stage=self.genome.stage,
            counts=dict(self.counts),
            moment=moment,
        )
