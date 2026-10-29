"""Duck Core engine: the deterministic "self" that the simulator (and later duck-self) runs.

It consumes perceived stimuli, decides expressions and posture, keeps bounded learning and
growth, and writes every fact to the L1 event log. It never sends motor commands: an
expression is an *intent* for duck-body's validated clip player, and standing up, sitting
down or a short walk is an intent that duck-body checks against its validated bounds.

Owner decisions of 2026-09-25 are the defaults here: the duck hatches on arrival (with a
conservative mode until the bench gates pass), duck-ear listens, and the whole family
chooses the song. Later that day the owner let it stand up on its own and, in floor mode,
walk a few slow steps: autonomous behaviour from its drives, personality, growth and
memory, never something that needs a person present.
"""

from __future__ import annotations

import functools
import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Concatenate, ParamSpec, TypeVar

from duckmind import worldport
from duckmind.duckcore import content, curiosity, daycycle, excursion, moments, nest, rng
from duckmind.duckcore.acts import POSTURE, RESULTS, TIMEOUT_KIND, WALK, Act, Acts
from duckmind.duckcore.fixed import (
    SCALE,
    approach,
    atan2_mrad,
    clamp,
    cos_mrad,
    isqrt,
    signed_mrad,
    sin_mrad,
    wrap_mrad,
)
from duckmind.duckcore.memory import Distilled, EventLog, distill, mapping, num, seq
from duckmind.provenance import Event, LearningStatus, LearningUpdate
from duckmind.worldport import (
    VIA_INPUT,
    VIA_PULL,
    CoreTape,
    WorldError,
    WorldPort,
    due_second,
)

TPE = timezone(timedelta(hours=8), "Asia/Taipei")
DEFAULT_START = datetime(2026, 12, 20, 19, 0, tzinfo=TPE)
# The engine revision (Config.engine_rev; phase-4 design A6). v4_0 is the frozen engine in
# duckmind.engines.v4_0; this one takes the world's outcomes in through its world port.
ENGINE_REV = "v4_1"

SOURCES: dict[str, tuple[str, str, float]] = {
    "ear": ("duck-ear", "mic-pet-classifier", 0.85),
    "joint": ("duck-body", "head-joint-load", 0.9),
    "imu": ("duck-body", "body-imu", 0.9),
    "tof": ("duck-body", "tof", 0.8),
    "pad": ("pad", "dpad", 1.0),
    "owner": ("owner-phone", "owner", 1.0),
    "self": ("duck-self", "duckcore", 1.0),
    "clock": ("sim-clock", "ntp", 1.0),
    "body": ("robotd", "robot.state", 1.0),
    "sim": ("simulator", "operator", 1.0),
    # Where a percept came from (its ``src``, percept-v1): the record of an input.
    "robot": ("duck-body", "body.sock", 1.0),
    "live": ("duck-body", "live-sim", 1.0),
    "surrogate": ("home-world", "surrogate", 1.0),
    "legacy": ("legacy-world", "v4-outcomes", 1.0),
    "replay": ("replay", "tape", 1.0),
}
VERIFY_PORT_PERCEPTS = 1000  # see DuckCore.verify_left
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_SECOND = timedelta(seconds=1)
_P = ParamSpec("_P")
_R = TypeVar("_R")


# The inputs that only let time pass: on the tape they are ``advance_to(the second reached)``,
# one entry per stretch of time between two other calls (CoreTape.passed).
TIME_INPUTS = frozenset({"advance", "advance_seconds", "advance_to"})


def _input(
    method: Callable[Concatenate[DuckCore, _P], _R],
) -> Callable[Concatenate[DuckCore, _P], _R]:
    """A public input method: its outermost calls go on the replay tape (design A8), with
    their arguments at Duck Core's clock. Calls made from inside another input are repeated
    by the outer one, so they are not entries.

    Time passing (:data:`TIME_INPUTS`) is recorded by where it got to, and back-to-back steps
    are one entry: a page lives its duck a second at a time all day, and living on is the
    same whatever the step (tests/test_step_invariance.py). A call that raises stays on the
    tape marked with its exception (it may have changed the duck before it did): a replay
    expects the same exception there, so a refused call never spoils the replay."""
    name = method.__name__
    timed = name in TIME_INPUTS

    @functools.wraps(method)
    def recorded(self: DuckCore, /, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        tape = self.tape
        if tape is None or self._depth:
            return method(self, *args, **kwargs)
        t_ms = self.now_ms()
        entry = None if timed else tape.call(t_ms, name, args, kwargs)
        self._depth += 1
        try:
            result = method(self, *args, **kwargs)
        except Exception as error:
            if entry is None:
                entry = tape.call(t_ms, name, args, kwargs)
            tape.raised(entry, error)
            raise
        finally:
            self._depth -= 1
        if timed:
            tape.passed(t_ms, self.now_ms())
        return result

    return recorded


BATTERY_DRAIN_PER_MIN = 160  # about 62 awake minutes per pack
BEDTIME_BATTERY = 1500
EMPTY_BATTERY = 300
CHARGE_PER_MIN = 100  # on its charging cable the pack fills about 1% a minute
# Naps. A tired duck sleeps until it is rested again (75–85% energy, never longer than
# NAP_MAX_MIN), so one real nap replaces a string of short dozes that left it tired all
# day. Asleep it still stirs now and then (every STIR_GAP_S), and after someone wakes it
# up it stays awake for NAP_WAKE_BLOCK_MIN.
NAP_REFILL_PER_MIN = 300
NAP_RESTED = (7500, 1000)  # target energy: 75% plus up to 10%
NAP_MAX_MIN = 45
STIR_GAP_S = (300, 900)  # asleep for the night
NAP_STIR_GAP_S = (90, 240)  # a daytime doze is lighter: it stirs more often
NAP_WAKE_BLOCK_MIN = 10
# Waking chances when something reaches a dozing duck (a daytime nap; at night it only
# stirs and sleeps on).
NAP_WAKE_CHANCE = {
    "scratch": 5000,
    "normal": 3500,
    "light": 2000,
    "noise": 6000,
    "name": 5000,
    "hello": 5000,  # the owner's phone: 你好, 去走走
    "praise": 2000,  # 乖
}
STRUGGLE_PER_MIN = 1500  # lying stuck after a fall, it tries again every few minutes
STIR_KINDS: dict[str, tuple[str, str]] = {
    # kind: (clip, words)
    "shift": ("sleep_shift", "睡夢中調整坐姿"),
    "nuzzle": ("sleep_nuzzle", "頭往下蹭一蹭"),
    "shiver": ("sleep_shiver", "睡夢中抖一下"),
    "mumble": ("sleep_mumble", "說夢話"),
    "peek": ("sleep_peek", "睜眼看一下，又睡回去"),
}
STIR_WEIGHTS = {"shift": 3, "nuzzle": 3, "shiver": 2, "mumble": 2}
# The clip player's idle layer: glances between decisions, drawn from the seed and the
# clock (never from the decision counters, so they never change the event chain). One in
# every FIDGET_SLOT_S slot of the clock, somewhere in its first FIDGET_JITTER_S + 1 s, so
# two glances are never more than FIDGET_SLOT_S + FIDGET_JITTER_S apart: a duck that may
# move is never seen frozen for a whole ten seconds. None plays while something the duck
# really did is showing (FIDGET_CLEAR_S either side of it).
FIDGET_CLIPS = {
    "sit": ("glance", "glance_away", "peek_down", "look_up", "tilt", "scan"),
    "stand": ("glance", "glance_away", "peek_down", "look_up", "tilt", "scan"),
}
FIDGET_SLOT_S = 6
FIDGET_JITTER_S = 2
FIDGET_PER_MIN = (60 // FIDGET_SLOT_S, 60 // FIDGET_SLOT_S)  # ten glances a minute
FIDGET_SHARE = (8500, 1500)  # each glance is 85–100% of what the clip may do right now
FIDGET_CLEAR_S = 1
PRAISE_WINDOW = timedelta(seconds=3)
NOT_THAT_WINDOW = timedelta(seconds=60)
COO_CHANCE = (10_000, 7_000, 5_000, 3_500, 2_500)
PREF_STEPS = (60, 30, 15)  # diminishing returns per item per growth day (≤ 0.0105/GD)
OWNER_BOND_STEP = 100
OWNER_BOND_CAP_PER_GD = 3
BIDS_PER_DAY = 10
SELF_PHRASES_PER_HOUR = 10
CARE_CAP_PER_GD = 5
PROMOTION_EVERY_GD = 7  # at most one new family phrase a week
PREF_RELAX_KEEP = 9700  # preferences drift 3% back toward neutral each growth day
BRANCH_MIN_EPISODES = 30
# The growth rules a duck hatches with, recorded in sim.hatched so a replay knows which
# ones made it. Phase P1 (spec v4 part E/F) by default; an old duck's branch, adult name and
# first word can still be decided by the earlier rules (``DuckCore(rules=RULES_V1)``).
RULES = {
    "branch": "branch-v2",
    "adult": "adult-v2",
    "first_word": "first-word-v2",
    "growth": "pace-v1",
    "care": "fair-care-v1",
    "moments": "moment-v1",
}
RULES_V1 = {**RULES, "branch": "branch-v1", "adult": "adult-v1", "first_word": "first-word-v1"}
CARE_DO_ZH = {"talk": "跟我說話", "play": "陪我玩", "touch": "摸摸我", "calm": "安靜地陪著我"}
CODAS = ("回頭再看一次", "多叫一聲", "小小點頭")
LATE_CODAS = ("輕輕晃兩下", "慢慢轉一圈頭")
FAVOURED_FITS = {
    "songbird": {"happy", "social"},
    "explorer": {"curious"},
    "snuggler": {"calm"},
    "dreamer": {"tired", "calm"},
}
LEANING_ZH = {"songbird": "愛唱歌", "explorer": "好奇", "snuggler": "愛依偎", "dreamer": "悠哉"}
EXPERIENCE_MILESTONES = {
    "likes_pickup": ("喜歡被抱", "我喜歡被抱起來了！"),
    "tamed_hand": ("不怕手了", "以前手靠近我會縮一下，現在我會靠過去。"),
    "favorite_found": ("找到最喜歡的", "我找到最喜歡的了：{favorite}。"),
    "learned_name": ("認得自己的名字", "聽了好幾天，我知道「{name}」就是在叫我了！"),
    "first_steps": ("第一次自己走路", "我第一次自己走路了！搖搖晃晃地往前走了幾步。"),
}
NAME_LEARN_DAYS = 3  # heard its name (not from the TV) on three growth days → it knows it

# Whole-body motion. Bounds come from content.LOCOMOTION (what the A100 sweep validates);
# these integer forms keep every decision in fixed point.
_LOCOMOTION: dict[str, Any] = content.LOCOMOTION
WALK_VX_MM = (
    round(_LOCOMOTION["walk_vx"][0] * 1000),
    round(_LOCOMOTION["walk_vx"][1] * 1000),
)
TURN_VYAW_MRAD = round(_LOCOMOTION["turn_vyaw"] * 1000)
WALK_S_MAX = int(_LOCOMOTION["walk_s_max"])
MIN_WALK_S = 2  # the shortest leg toward a target (the last few centimetres home)
HOME_PURPOSES = ("home", "dock", "circle")  # the errand back to the nest
# A walk of its own lasts at least this long: the policy steps for 0.7–1.3 s before the
# body really moves (official physics, 2026-09-26), so a 2 s walk looked like marching on
# the spot and then two steps.
MIN_WANDER_S = 3
WANDER_CURVE_MRAD = 500  # a walk may curve gently; a turn walks the full turn_vyaw arc
# Facing further off than this (about 57°) it turns first, walking a tight arc; closer, it
# curves on the way (a 6 s leg at the gentlest curve turns ~0.9 rad).
TURN_IN_PLACE_MRAD = 1000
# The official walk barely turns on the spot (1.0 rad/s commanded → 5–11°/s, while the legs
# keep stepping), but on the move the same command turns 22–40°/s (official physics,
# 2026-09-26). So it turns by walking a tight arc at the slowest walk; turning in place is
# only the last resort when every arc is blocked.
ARC_VX_MM = WALK_VX_MM[0]
# Company and discovery lift the mood of a duck that lives the family's day (spec 6.5):
# someone right here settles its valence toward PRESENT_VALENCE; a place it had not seen for
# a while gives a small lift (EXPLORE_*).
PRESENT_VALENCE = 3000
EXPLORE_VALENCE = 200
EXPLORE_AROUSAL = 300
NEST_IDLE_SHARE = 7500  # a nest duck's idle head moves, of the stage's share (off the nest)
NESTED_IDLE_SHARE = 5000  # …and resting on the nest
# What the official velstand network actually does with a command, measured in the official
# mjlab physics (2026-09-25): 0.30 m/s commanded → ~0.11 m/s, 0.40 → ~0.16 m/s (linear in
# between); a turn at 1.0 rad/s on the move → ~0.3 rad/s or more (0.22–0.56 × command, with
# a steady drift to the right). Distances use these, not commands.
ACTUAL_SPEED_MM = ((300, 110), (400, 160))
ACTUAL_YAW_RATIO = 3000  # fixed point: actual yaw rate ≈ 0.3 × command (conservative)
# The tight arc it turns on, in this model: 0.11 m/s at ~0.3 rad/s is a circle of ~0.37 m.
# Turning round toward home from a pose that faces out carries it up to this far further
# out first, so near the edge of its radius it has to face along the edge, not out of it.
ARC_RADIUS_MM = ACTUAL_SPEED_MM[0][1] * 1000 // (1000 * ACTUAL_YAW_RATIO // SCALE)
# Facing home within TURN_IN_PLACE_MRAD it walks home curving gently (no turn needed).
FACING_HOME_COS = cos_mrad(1000)
CONSERVATIVE_SWEPT = 8500  # conservative cap for moves the physics sweep cleared
CONSERVATIVE_WALKS = 7000  # conservative mode: walks are shorter (half) and a bit rarer
CONSERVATIVE_OUTINGS = 8000
# Released (a hand let go, set down on the floor, woken, morning): the chance that it gets
# up to look about right away, before personality and tiredness.
PERK_UP_STAND = 9000
# A rested floor duck gets up to go and look about on its own every few minutes
# ("小鴨應該時常自主走動"): each awake, seated minute on the floor it sets off with this
# chance, times personality, growth stage, mood, what the family taught and how tired it
# is. That is a memoryless draw, so the gaps are irregular, never a metronome.
OUTING_PER_MIN = 7000
# …and every seated minute that passes without it getting up adds that same chance again
# (1×, 2×, 3×… the first minute's), so the wait is still a random draw but never a long
# one: a rested adult is up again within a few minutes, a tired or young one a little later.
OUTING_RESTLESS = SCALE
WANDER_NEED = 12000  # a standing duck's urge to walk on rather than stand and look
STAND_S_MAX = int(_LOCOMOTION["stand_s_max"])
MIN_MOTION_BATTERY = int(_LOCOMOTION["min_battery"])
STANDING = ("stand", "walk")
STAND_DRAIN = 13000  # standing costs about 1.3× the seated battery drain; walking about 2×
SIT_BATTERY = MIN_MOTION_BATTERY  # below 30% battery a standing duck sits down
# Energy per awake minute and per expression: a rested duck stays lively for three to four
# hours with naps between (it tired within two, and was "a bit tired" most of the time).
AWAKE_DRAIN = 14
AWAKE_DRAIN_STANDING = 22
EXPRESSION_DRAIN = 10
SIT_ENERGY = 2500  # below 25% energy it is too tired to stand: it sits (and soon naps)
TIRED_FULL_ENERGY = 5000  # from here down to SIT_ENERGY tiredness slows it (never stops it)
TIRED_FULL_BATTERY = 6000
# A short walk tires it a little (so it naps now and then, not constantly). Spec v4: 5, so
# a day of small trips from the nest (half an hour of walking) does not wear it out.
WALK_ENERGY_PER_S = 5
STAND_UP_ENERGY = 25
# The floor: a 6 m × 6 m room centred on where the family put the duck down (its mat), as
# drawn by duck3d.js (ROOM_HALF, WALL_MARGIN). Millimetres and milliradians, x/y as in
# MuJoCo (heading 0 = +x, counter-clockwise).
ROOM_HALF_MM = 3000
WALL_MARGIN_MM = 250
ROOM_LIMIT_MM = ROOM_HALF_MM - WALL_MARGIN_MM
PATH_STEPS_PER_S = 10  # a walk's path is integrated in 0.1 s steps
PROBE_MM = 800  # after turning away from a wall or a bad spot, this much must be clear
TURN_ROOM_MM = 400  # a walk ends with this much clear ahead (about a tight arc's radius)
# Falls. After a fall the body tries to roll back up by itself, up to RIGHTING_TRIES tries
# (when, and whether each works, is the body's: the world sends one body.righting per try,
# phase-4 design A12; LegacyWorld keeps v4's 2 s / 4 s timing and its righting_ppm). Up on
# the first try is an easy recovery; more tries, or none that worked (then it lies there
# until picked up), is a hard one. One fall, or falls it got up from easily, teach nothing
# but a gentle memory. HARD_FALLS_TO_LEARN hard ones within AVOID_RADIUS_MM of each other
# teach it that the spot is not for walking.
RIGHTING_TRIES = 3
HARD_FALLS_TO_LEARN = 3
AVOID_RADIUS_MM = 500
WALK_SAT_WANDER = 1500  # a walk satisfies the urge to explore for a little while…
WALK_SAT_TURN = 800
WALK_SAT_RELAX = 1200  # …and the urge comes back (12% of the gap per minute)
# An outing is a few short legs in a row with a look around between them, not one lone
# step: after setting off it may walk on up to EXTRA_LEGS more times (CONSERVATIVE_LEGS in
# conservative mode, where every leg is short), each with LEG_CHANCE × its personality's
# liking for walks, pausing LEG_PAUSE_S to look before each.
EXTRA_LEGS = 4
CONSERVATIVE_LEGS = 3
LEG_CHANCE = 6500
LEG_PAUSE_S = (2, 5)


def pose_along(
    plan: dict[str, Any],
    steps: int,
    blocked: Callable[[tuple[int, int], tuple[int, int]], str | None] | None = None,
) -> tuple[tuple[int, int, int], str | None]:
    """Where a walk has taken the duck after ``steps`` tenths of a second (integer math).

    ``plan`` holds ``from`` (x mm, y mm, heading mrad), ``speed_mm`` (mm/s) and
    ``yaw_mrad`` (mrad/s). With ``blocked``, stops at the first point it rejects and
    returns that reason too.
    """
    x0, y0, h0 = plan["from"]
    speed, yaw_rate = int(plan["speed_mm"]), int(plan["yaw_mrad"])
    xu, yu = x0 * 1000, y0 * 1000  # micrometres while integrating
    for k in range(1, steps + 1):
        heading = h0 + yaw_rate * k // PATH_STEPS_PER_S
        xu += speed * (1000 // PATH_STEPS_PER_S) * cos_mrad(heading) // SCALE
        yu += speed * (1000 // PATH_STEPS_PER_S) * sin_mrad(heading) // SCALE
        if blocked is not None and speed:
            why = blocked((x0, y0), (xu // 1000, yu // 1000))
            if why is not None:
                return (xu // 1000, yu // 1000, wrap_mrad(heading)), why
    heading = wrap_mrad(h0 + yaw_rate * steps // PATH_STEPS_PER_S)
    return (xu // 1000, yu // 1000, heading), None


def poses_by_second(
    plan: dict[str, Any],
    seconds: int,
    blocked: Callable[[tuple[int, int], tuple[int, int]], str | None] | None = None,
) -> tuple[list[tuple[int, int, int]], str | None]:
    """The pose after each whole second of a walk (1…``seconds``), integrated exactly as
    ``pose_along`` does, up to the last whole second before its path is blocked (and why
    it was, else None). One pass gives every length of the same arc."""
    x0, y0, h0 = plan["from"]
    speed, yaw_rate = int(plan["speed_mm"]), int(plan["yaw_mrad"])
    xu, yu = x0 * 1000, y0 * 1000
    out: list[tuple[int, int, int]] = []
    for k in range(1, seconds * PATH_STEPS_PER_S + 1):
        heading = h0 + yaw_rate * k // PATH_STEPS_PER_S
        xu += speed * (1000 // PATH_STEPS_PER_S) * cos_mrad(heading) // SCALE
        yu += speed * (1000 // PATH_STEPS_PER_S) * sin_mrad(heading) // SCALE
        if blocked is not None and speed:
            why = blocked((x0, y0), (xu // 1000, yu // 1000))
            if why is not None:
                return out, why
        if k % PATH_STEPS_PER_S == 0:
            out.append((xu // 1000, yu // 1000, wrap_mrad(heading)))
    return out, None


@dataclass(frozen=True, slots=True)
class HardwareConfig:
    tof: bool = False  # most Microducks have no head module
    head_imu: bool = False
    pickup_detect: bool = True  # assumed bench-validated in the simulator
    ear: bool = True  # duck-ear listens (owner decision 2026-09-25)
    # Recognizes only its own name, on-device (owner decision 2026-09-25). Off until the
    # trained keyword model passes its false-accept evaluation.
    name_ear: bool = False
    # Standing up, walking and turning on its own (owner decisions 2026-09-25). Off on the
    # real robot until the A100 physics sweep shows no falls and negligible drift for the
    # posture transitions and slow walks; sitting down (the rest pose) is always allowed.
    motion_validated: bool = False
    # The charging nest (spec v4 part B): a flat pad with belly contacts and a marker it
    # docks on by itself. Simulator only for now (owner decision 2026-09-26): off on the
    # real robot until the nest hardware and its bench gates exist.
    nest: bool = False
    nest_charge_per_min: int = 125  # net charge on the nest (a hardware requirement ≥ 125)
    nest_awake_drain: int = 20  # awake and seated on the nest (legs relaxed, power path)
    camera: bool = True  # the head camera can tell that someone is in the room (≤ 1 fps)
    # Phase 4: how often a walk falls, a righting try or a stand-up works, a dock try
    # touches, and whether the camera sees the nest's marker are the world's (the world
    # header's ``rates`` and ``marker``, LegacyWorld), no longer Duck Core's (rule R1).


@dataclass(slots=True)
class CoreState:
    name: str
    seed: int
    hatched_at: datetime
    now: datetime
    day_date: date
    trace: str = "life"
    boot_seq: int = 0
    awake: bool = False
    boot_at: datetime | None = None
    last_power_off: datetime | None = None
    lived_s: int = 0
    day_awake_s: int = 0
    gd2: int = 0
    absence_half_used: int = 0
    ntp_ok: bool = True
    docked: bool = False
    hot: bool = False
    conservative: bool = True
    battery: int = SCALE
    bedtime_done: bool = False
    # Simulator: the family swaps in a fresh pack when the battery gets too low to walk.
    battery_help: bool = False
    off_reason: str | None = None  # why it was last powered off
    floor_when_off: bool = False  # it was on the floor when it powered off
    # Affect and drives (fixed point).
    valence: int = 1000
    arousal: int = 3000
    energy: int = 8000
    social: int = 3000
    overload: int = 0
    bond_owner: int = 2000
    interest: dict[str, int] = field(default_factory=dict)
    # Timers.
    napping_until: datetime | None = None
    nap_ended_at: datetime | None = None
    nap_reason: str | None = None  # tired | bedtime | quiet_hours
    stir_at: datetime | None = None  # the next time it stirs in its sleep
    stir_next: datetime | None = None  # a stir due just after this hour of night sleep
    nap_block_until: datetime | None = None  # woken up: no dozing off again before this
    quiet_until: datetime | None = None
    stop_until: datetime | None = None
    stop_moments: list[datetime] = field(default_factory=list)
    lockout_until: datetime | None = None
    guest_until: datetime | None = None
    yielding: bool = False
    tv_on: bool = False
    presence_at: datetime | None = None
    last_owner_hello: datetime | None = None
    owner_seen_today: bool = False
    # Expression bookkeeping.
    last_verb: str | None = None
    last_flag: dict[str, Any] | None = None
    self_phrase_times: list[datetime] = field(default_factory=list)
    coo_step: int = 0
    coo_series_at: datetime | None = None
    last_touch_at: datetime | None = None
    last_pickup_at: datetime | None = None
    last_voice_reaction: datetime | None = None
    # Learning.
    prefs: dict[str, int] = field(default_factory=dict)
    pref_gd_count: dict[str, int] = field(default_factory=dict)
    owner_bond_gd: int = 0
    not_that_gd: set[str] = field(default_factory=set)
    favorite_found: bool = False
    # Counters.
    touches: int = 0
    pickups: int = 0
    slow_hands: int = 0
    name_calls: int = 0
    name_gds: list[int] = field(default_factory=list)
    play_meters: int = 0
    keepsakes: list[int] = field(default_factory=list)
    day_voice: int = 0
    day_noise: int = 0
    day_tv_min: int = 0
    noise_habit: int = 0
    favorite_uses: int = 0
    active_min_since_babble: int = 0
    # Growth.
    unlocked: list[str] = field(default_factory=list)
    milestones: dict[str, dict[str, Any]] = field(default_factory=dict)
    first_word: str | None = None
    quirks: dict[str, str] = field(default_factory=dict)
    revealed: list[str] = field(default_factory=list)
    nature: dict[str, int] = field(default_factory=dict)
    care_today: dict[str, int] = field(default_factory=dict)
    care_window: dict[str, int] = field(default_factory=dict)
    care_adult: dict[str, int] = field(default_factory=dict)
    branch: str | None = None
    runner_up: str | None = None
    branch_scores: dict[str, int] = field(default_factory=dict)
    adult_name: str | None = None
    # Voice lineage.
    phrases: dict[str, dict[str, Any]] = field(default_factory=dict)
    promotions: list[str] = field(default_factory=list)
    promotion_gd: int | None = None
    song: dict[str, Any] | None = None
    molt: dict[str, Any] | None = None
    molt_decided: bool = False
    # Attention bids.
    bid_times: list[datetime] = field(default_factory=list)
    bids_today: int = 0
    bid_pending_since: datetime | None = None
    bid_backoff: int = SCALE
    unanswered: int = 0
    bids_stopped: bool = False
    come_play_this_visit: bool = False
    play_accepted_gd: int | None = None
    # Rare events.
    rare_n: dict[str, int] = field(default_factory=dict)
    rare_last: dict[str, datetime] = field(default_factory=dict)
    rare_today: dict[str, int] = field(default_factory=dict)
    album: list[str] = field(default_factory=list)
    active_min_since_rare: int = 0
    # Posture and floor mode.
    posture: str = "sit"  # sit | stand | walk | fallen
    floor: bool = False  # the family marked it as being on the floor
    stood_at: datetime | None = None  # when this standing spell began
    stand_idle_from: datetime | None = None  # standing still since (stand-up or last walk)
    walk_start: datetime | None = None
    walk_until: datetime | None = None
    walk_plan: dict[str, Any] | None = None
    walk_sat: int = 0
    outing_legs: int = 0  # legs still to walk on this outing
    # Seated minutes on the floor, free to get up, since it last stood up: the longer it has
    # sat, the more it wants to go and look about (see OUTING_RESTLESS).
    restless_min: int = 0
    leg_pause_s: int = 0  # the look around before the next leg
    walked_cm: int = 0
    walks: int = 0
    turns: int = 0
    stands: int = 0
    falls: int = 0
    # Where Duck Core says the duck is on the floor (mm, mrad; see ROOM_HALF_MM).
    pos_x: int = 0
    pos_y: int = 0
    heading: int = 0
    righting: dict[str, Any] | None = None  # self-righting tries in progress after a fall
    fall_spots: list[dict[str, Any]] = field(default_factory=list)
    avoid_spots: dict[str, dict[str, Any]] = field(default_factory=dict)  # learned, forgettable
    walk_keepsakes: list[int] = field(default_factory=list)
    # Visit.
    visit_events: list[str] = field(default_factory=list)
    visit_summary: dict[str, Any] = field(default_factory=dict)
    last_visit: dict[str, Any] = field(default_factory=dict)
    forget_log: list[dict[str, Any]] = field(default_factory=list)
    # The clock, to the second (see DuckCore.advance_seconds): seconds lived into the
    # current minute, its beat (when the minute's own life runs; −1 = not drawn yet),
    # whether that beat has run, whether it is a minute asleep on the charger at night,
    # and the last 07:00 the household woke up to.
    tick_s: int = 0
    beat_s: int = -1
    beat_done: bool = False
    sleep_minute: bool = False
    last_morning: date | None = None
    # A day at home (spec v4 part A, daycycle.py). Cues stay in RAM until they are logged
    # as env.cue; every "how long since" is counted from logged moments only.
    phase: str = "family"  # night | dawn | solo | family | alone | wind_down
    phase_since: datetime | None = None
    family: str = "awake"  # what it believes: awake | asleep | away | unknown
    family_since: datetime | None = None
    cue_buffer: dict[str, int] = field(default_factory=dict)
    cue_room: bool = False
    cue_logged_at: datetime | None = None  # the last logged cue of any kind
    room_logged_at: datetime | None = None  # the last logged sign of someone right here
    door_logged_at: datetime | None = None
    voice_logged_at: datetime | None = None
    door_hold_until: datetime | None = None
    phones_home: list[str] = field(default_factory=list)
    return_model: dict[str, list[int]] = field(default_factory=dict)
    depart_model: dict[str, list[int]] = field(default_factory=dict)
    away_since: datetime | None = None
    wake_plan: datetime | None = None
    woke_at: datetime | None = None
    dawn_until: datetime | None = None
    anticipate_at: datetime | None = None
    anticipating: dict[str, Any] | None = None
    rest_spell_until: datetime | None = None
    reunion_until: datetime | None = None
    goodnight_at: datetime | None = None
    bedtime_nest: bool = False  # tonight's goodnight in the nest is done
    solo_exc_s: int = 0
    hush: bool = False
    temperament: dict[str, int] = field(default_factory=dict)
    day_aware: bool = False  # lives the family's day even without a nest (see DuckCore.aware)
    # The nest (part B, nest.py).
    nested: bool = False  # sitting on the nest's contacts (never walks while it is)
    homing: dict[str, Any] | None = None  # the errand back to the nest, leg by leg
    stranded: dict[str, Any] | None = None
    pose_err_mm: int = 0
    unplaced: bool = False  # lifted up: not walking until it is put back somewhere
    nest_trouble: bool = False
    nest_trouble_falls: int = 0
    stand_fail_times: list[datetime] = field(default_factory=list)
    stand_block_until: datetime | None = None
    docks: int = 0
    roam_radius_mm: int = 1500
    errand_step: int = 0  # counts every step an errand or outing takes (see _spend)
    hand_placed: bool = False  # set down by a hand and has not walked since
    need: dict[str, Any] | None = None  # a call for help it is waiting on
    # Outings (part C, excursion.py).
    excursion: dict[str, Any] | None = None
    last_excursion_end: datetime | None = None
    cells: dict[int, list[int]] = field(default_factory=dict)  # cell → [last seen min, good]
    excursions: int = 0
    last_play_at: datetime | None = None
    # One line of numbers a day (duck.day_stats), for the family page and the harness.
    day_awake_min: int = 0
    day_lonely_min: int = 0
    day_alone_awake_min: int = 0
    day_offnest_min: int = 0
    day_walk_s: int = 0
    day_turn_s: int = 0
    day_drain: int = 0
    day_docks: int = 0
    day_excursions: int = 0
    day_social_sum: int = 0
    day_phase_min: dict[str, int] = field(default_factory=dict)
    # Personality (spec v4 part E, phase P1): its temperament, how fast it grows, its own
    # experience before it becomes a juvenile, and which rules it hatched with.
    stage_days: dict[str, int] = field(default_factory=dict)  # stage → growth day it starts
    self_window: dict[str, int] = field(default_factory=dict)
    self_today: dict[str, int] = field(default_factory=dict)
    rules: dict[str, str] = field(default_factory=dict)
    branch_detail: dict[str, Any] = field(default_factory=dict)
    epithet: str | None = None
    word_answers: dict[str, int] = field(default_factory=dict)  # sounds answered, GD 3–8
    care_at: dict[str, datetime] = field(default_factory=dict)  # the last counted, by kind
    company_min: int = 0  # quiet company: minutes resting with someone here, no contact
    play_cell: list[Any] | None = None  # [cell, date] where the family last played with it
    last_outing: dict[str, Any] | None = None  # the outing that just ended (its way out)
    last_outing_day: str | None = None
    room_cue_event: str | None = None  # the last logged sign of someone right here
    last_sounds: dict[str, Any] | None = None  # the sounds it last made by itself
    last_hand_at: datetime | None = None
    # Clever and goofy moments (part F, moments.py).
    form_day: str | None = None
    form: str | None = None
    moments_today: dict[str, int] = field(default_factory=dict)
    moment_last: dict[str, datetime] = field(default_factory=dict)
    moment_album: list[str] = field(default_factory=list)
    moment_counts: dict[str, int] = field(default_factory=dict)  # before it grew up
    moment_attempts: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_moment: dict[str, Any] | None = None
    gag_days: dict[str, list[int]] = field(default_factory=dict)
    gags: list[str] = field(default_factory=list)
    gag_gd: int | None = None
    # Curiosity (curiosity spec §7.1): sockets only until curiosity P2 (phase-4 design §9).
    things: dict[str, dict[str, Any]] = field(default_factory=dict)
    curio: dict[str, Any] | None = None
    curio_seq: int = 0
    curio_today: dict[str, int] = field(default_factory=dict)
    zones: list[dict[str, Any]] = field(default_factory=list)
    asks: dict[str, dict[str, Any]] = field(default_factory=dict)
    keepsake: list[str] = field(default_factory=list)
    # Attached to the neural duck (docs/design/GROWTH-G2-2026-10-10.md; ``DuckCore.bridge_on``): this
    # Duck Core simulates no body and no experiences of its own, it is given what the neural duck
    # lives (:mod:`duckmind.duckcore.bridge`). False: the duck as it always was.
    bridged: bool = False
    bridge_company_event: str | None = None  # the last logged sign of company (care evidence)

    @property
    def gd(self) -> int:
        return self.gd2 // 2


class DuckCore:
    """One duck's self. All randomness is counter-based; all decisions are logged."""

    def __init__(
        self,
        *,
        seed: int = 7,
        name: str = "小鴨",
        start: datetime | None = None,
        hardware: HardwareConfig | None = None,
        conservative: bool = True,
        expression_caps: dict[str, int] | None = None,
        battery_help: bool = False,
        day_aware: bool = False,
        rules: dict[str, str] | None = None,
        port: WorldPort | None = None,
        tape: CoreTape | None = None,
    ) -> None:
        begin = start or DEFAULT_START
        if begin.utcoffset() is None:
            raise ValueError("start must be timezone-aware")
        self.seed = int(seed)
        self.table = content.table_hash()
        self.hw = hardware or HardwareConfig()
        # The world (phase 4): Duck Core pulls its percepts inside its own time steps and
        # sends its commands through ``acts`` (duckmind.worldport, duckcore/acts.py).
        self.acts = Acts(self, port)
        # The replay tape (design A8), and how deep in an input call it is (only outermost
        # calls are entries; hatching is the Config's, not an input).
        self.tape = tape
        self._depth = 1
        # Port percepts still to check against percept-v1 before they are taken in (None:
        # every one). An in-process world is trusted after its first thousand (design A13:
        # 7 µs a percept would eat half the phase's CPU budget); a percept handed in from
        # outside (on_percept via "input": api.sense, the family proxy) is always checked.
        self.verify_left: int | None = VERIFY_PORT_PERCEPTS
        # Per-clip amplitude caps validated by the expression sweep (fixed point). A clip
        # missing from a validated set was never swept and stays at the conservative cap.
        self.caps = dict(expression_caps) if expression_caps is not None else None
        self.log = EventLog()
        self.journal: list[dict[str, Any]] = []  # simulator ground truth: never the duck's memory
        self.counters: dict[str, int] = {}
        self.s = CoreState(
            name=name, seed=self.seed, hatched_at=begin, now=begin, day_date=begin.date()
        )
        self.s.conservative = conservative
        self.s.battery_help = battery_help
        self.s.day_aware = day_aware
        self.s.rules = dict(rules if rules is not None else RULES)
        self._memory: Distilled | None = None
        self._memory_len = -1
        self._stage_cache: tuple[int, content.Stage] | None = None
        self._hatch()
        self._depth = 0

    @classmethod
    def carried_over(
        cls,
        state: CoreState,
        log: EventLog,
        counters: Mapping[str, int],
        *,
        hardware: HardwareConfig,
        expression_caps: dict[str, int] | None = None,
        journal: list[dict[str, Any]] | None = None,
        port: WorldPort | None = None,
        tape: CoreTape | None = None,
        acts_seq: int = 0,
    ) -> DuckCore:
        """A duck carried over from another engine (the migration seam, phase-4 design WP13):
        its state, its chain and its own draw counters exactly as they were, and no new hatch.
        The world's outcome counters go to its world (LegacyWorld), not here; ``acts_seq`` is
        how many acts it has issued (0 from v4, which issued none). Only at a quiet moment:
        never in the middle of a walk or of the tries to get up after a fall (those schedules
        were the old engine's own)."""
        if state.walk_plan is not None or state.righting is not None:
            raise ValueError("a duck is carried over between walks, never while it gets up")
        core = cls.__new__(cls)
        core.seed = int(state.seed)
        core.table = content.table_hash()
        core.hw = hardware
        core.acts = Acts(core, port)
        core.acts.seq = int(acts_seq)
        core.tape = tape
        core._depth = 0
        core.verify_left = VERIFY_PORT_PERCEPTS
        core.caps = dict(expression_caps) if expression_caps is not None else None
        core.log = log
        core.journal = list(journal or [])
        core.counters = dict(counters)
        core.s = state
        core._memory = None
        core._memory_len = -1
        core._stage_cache = None
        return core

    # ------------------------------------------------------------------ the world port

    @property
    def port(self) -> WorldPort | None:
        return self.acts.port

    def attach_port(self, port: WorldPort | None) -> None:
        """Connect (or replace) the world this duck lives in."""
        self.acts.port = port

    def now_ms(self) -> int:
        """Duck Core's clock on the port's axis (epoch milliseconds, a whole second)."""
        return ((self.s.now - _EPOCH) // _SECOND) * 1000

    @property
    def outbox(self) -> list[str]:
        """Command events (ids) issued since the last :meth:`drain_outbox`."""
        return self.acts.outbox

    def drain_outbox(self) -> list[str]:
        return self.acts.drain_outbox()

    def emit_command(self, kind: str, payload: dict[str, Any], parents: tuple[str, ...]) -> Event:
        """(acts) Write a command event: Duck Core's own output, signed self."""
        return self._emit(kind, payload, parents=parents)

    def deliver(self, percepts: list[dict[str, Any]], via: str) -> None:
        """(acts and the time step only) Take in what the port delivered, in order."""
        tape = self.tape
        for event in percepts:
            if tape is not None:
                tape.percept(self.now_ms(), via, self.acts.n_cmd, self.acts.n_poll, event)
            self.on_percept(event, via=via)

    def _pull(self, target: datetime) -> bool:
        """Take in, in time order, every percept the port has due by ``target`` and close
        the acts whose deadline second has come (a result of that second first: the world
        comes first at a second). Design A1. Returns whether anything came or closed.

        A port that says a percept is due but hands none over is not asked again in this
        pull (never spin on it); what it holds comes when it gives it."""
        acts = self.acts
        port = acts.port
        target_s: int | None = None
        took = False
        silent = False
        while True:
            t_world = port.next_at_ms() if port is not None and not silent else None
            t_dead = acts.next_deadline_ms() if acts.open else None
            if t_world is None and t_dead is None:
                return took
            if target_s is None:
                target_s = (target - _EPOCH) // _SECOND
            world_s = due_second(t_world) if t_world is not None else None
            dead_s = due_second(t_dead) if t_dead is not None else None
            if (
                world_s is not None
                and world_s <= target_s
                and (dead_s is None or world_s <= dead_s)
            ):
                assert port is not None
                self._step_to(world_s)
                came = port.due(world_s * 1000)
                if came:
                    took = True
                    self.deliver(came, VIA_PULL)
                else:
                    silent = True
            elif dead_s is not None and dead_s <= target_s:
                self._step_to(dead_s)
                for act in acts.expire(dead_s * 1000):
                    took = True
                    self._act_timeout(act)
            else:
                return took

    def _world_next_ms(self) -> int | None:
        """The earliest of the port's next percept and the next act deadline (None: none)."""
        port = self.acts.port
        times = [
            t
            for t in (
                port.next_at_ms() if port is not None else None,
                self.acts.next_deadline_ms() if self.acts.open else None,
            )
            if t is not None
        ]
        return min(times) if times else None

    def _step_to(self, second: int) -> None:
        """Move the clock forward to epoch second ``second`` (never back), in its own zone."""
        s = self.s
        ahead = second - (s.now - _EPOCH) // _SECOND
        if ahead > 0:
            s.now = s.now + timedelta(seconds=ahead)

    def _act_timeout(self, act: Act) -> None:
        """An act whose result never came by its deadline: Duck Core's own decision that it
        is over (a result that comes later is stale)."""
        s = self.s
        timeout = self._emit(
            TIMEOUT_KIND,
            {"act_id": act.act_id, "cmd": act.cmd, "deadline_ms": act.deadline_ms},
            parents=(act.event_id,),
        )
        if act.body and self.acts.port is not None:
            # The body is told to stop what Duck Core gave up on: a walk it never ended must
            # not go on (nor end in a fall) under a duck that now believes it stands still.
            self.acts.stop(act.act_id, "timeout", parents=(timeout.event_id,))
        if act.cmd == WALK and s.walk_plan is not None and s.walk_plan.get("act_id") == act.act_id:
            # The body never said how the walk ended: it stands where it last knew it was.
            self._cut_walk()
            s.posture = "stand"
            s.stand_idle_from = s.now
            self._walk_keepsakes(timeout.event_id)
        elif act.cmd == "duck.curio":
            curiosity.on_timeout(self, act.act_id)

    # ------------------------------------------------------------------ plumbing

    @property
    def aware(self) -> bool:
        """It lives the family's day (spec v4): a duck with a nest, or one whose family is a
        household-v2 family. The old visit routines and the plain page duck do not."""
        return self.hw.nest or self.s.day_aware

    def _rand(self, purpose: str) -> int:
        counter = self.counters.get(purpose, 0) + 1
        self.counters[purpose] = counter
        return rng.draw(self.seed, self.table, purpose, counter)

    def _emit(
        self,
        kind: str,
        payload: dict[str, Any],
        source: str = "self",
        parents: tuple[str, ...] | list[str] = (),
    ) -> Event:
        event = self.log.append(
            kind, self.s.now, SOURCES[source], payload, self.s.trace, tuple(parents)
        )
        if self.s.awake:
            self.s.visit_events.append(event.event_id)
        return event

    def _journal(
        self, persona: str | None, action: str, detail: str, perceived: bool, events: list[Event]
    ) -> None:
        self.journal.append(
            {
                "at": self.s.now.isoformat(),
                "persona": persona,
                "persona_zh": content.PERSONAS.get(persona or "", "環境"),
                "action": action,
                "detail": detail,
                "perceived": perceived,
                "event_ids": [event.event_id for event in events],
            }
        )

    def _set_now(self, target: datetime) -> None:
        s = self.s
        # Everything due by ``target``, in time order (design A1): what the world sensed
        # meanwhile (a walk's end or its fall, a righting try) and the deadlines of its open
        # acts, and Duck Core's own clock events: asleep it stirs now and then, and a night's
        # sleep crosses midnight. The world's percepts come first at a second: they are pulled
        # up to and including the second of the next clock event before it runs, so a percept
        # just after midnight counts in the new day and a stir falls where it falls, the same
        # whatever the step size. (With LegacyWorld nothing of the world's is ever due past a
        # clock event within one step: walks never run between 22:00 and 07:00, and a napping
        # duck neither walks nor lies waiting to get up. So this is v4's order.)
        while True:
            midnight = self._next_midnight()
            stir = s.stir_at
            clock: datetime | None
            if stir is not None and stir <= target and stir < midnight:
                clock = stir
            elif target >= midnight:
                clock, stir = midnight, None
            else:
                clock = stir = None
            if (self.acts.port is not None or self.acts.open) and self._pull(
                target if clock is None else clock
            ):
                continue  # (what came may change what comes next: it woke up, it was switched off)
            if stir is not None:
                s.now = max(s.now, stir)
                self._sleep_stir("dream")
            elif clock is not None:
                self._finalize_day()
            else:
                break
        # A morning jumped over (switched off, or no chance to stop there) is simply past.
        if self._morning_due() < target:
            latest = datetime.combine(target.date(), time(7), TPE)
            if latest >= target:
                latest -= timedelta(days=1)
            s.last_morning = latest.date()
        s.now = target

    def _next_midnight(self) -> datetime:
        return datetime.combine(self.s.day_date + timedelta(days=1), time(0), TPE)

    def _morning_due(self) -> datetime:
        """The next 07:00 on the duck's clock that the household has not woken up to."""
        s = self.s
        moment = datetime.combine(s.now.date(), time(7), TPE)
        if s.now > moment or s.last_morning == moment.date():
            moment += timedelta(days=1)
        return moment

    def _spend(self, seconds: int) -> None:
        """Let ``seconds`` pass, stopping wherever a standing duck has to walk on or sit,
        and at 07:00, when the household's day begins.

        Standing still is bounded (``stand_s_max``, shorter when young), and that moment can
        fall inside a minute or inside a long touch. The choice made there is the same
        logged utility decision as any other, restricted to what a standing duck may do.
        """
        s = self.s
        if s.bridged:
            return  # (time is the calendar's: an input takes none of its own)
        target = s.now + timedelta(seconds=seconds)
        while s.awake:
            due = self._stand_due()
            morning = self._morning_due()
            point = due if due is not None and due <= morning else morning
            if point > target:
                break
            if point > s.now:
                self._pass(int((point - s.now).total_seconds()))
            if point is morning:
                self._morning()
                continue
            if self._stand_due() != due:
                continue  # something happened on the way there (a fall, a hand): look again
            before = (s.posture, s.stand_idle_from, s.walk_until, s.errand_step)
            if s.homing is not None:
                nest.next_leg(self)
            elif s.excursion is not None:
                excursion.next_leg(self)
            elif self.hw.nest and s.posture == "stand":
                self._nest_stood()
            else:
                self._decide_and_express("walk_on" if s.outing_legs > 0 else "stand_limit")
            if (s.posture, s.stand_idle_from, s.walk_until, s.errand_step) == before:
                break  # nothing it could do (never expected): let time pass rather than spin
        if target > s.now:
            self._pass(int((target - s.now).total_seconds()))

    def _pass(self, seconds: int) -> None:
        """Let ``seconds`` pass. Awake time before midnight counts for the day it was in,
        so a long step and many short ones grow the same duck."""
        s = self.s
        target = s.now + timedelta(seconds=seconds)
        awake = s.awake
        before = 0
        if awake:
            to_midnight = int((self._next_midnight() - s.now).total_seconds())
            before = max(0, min(seconds, to_midnight))
            s.day_awake_s += before
            s.lived_s += seconds
        self._set_now(target)
        if awake and seconds > before:
            since_midnight = int(
                (target - datetime.combine(s.day_date, time(0), TPE)).total_seconds()
            )
            s.day_awake_s += min(seconds - before, since_midnight)

    def _until(self, moment: datetime | None) -> bool:
        return moment is not None and self.s.now < moment

    def _present(self, minutes: int) -> bool:
        s = self.s
        if s.presence_at is not None and s.now - s.presence_at <= timedelta(minutes=minutes):
            return True
        # With a nest the duck also knows someone is here from what it saw and heard.
        return (
            self.aware
            and s.family == "awake"
            and s.room_logged_at is not None
            and s.now - s.room_logged_at <= timedelta(minutes=minutes)
        )

    def _presence(self, by: str = "", evidence: Event | None = None) -> None:
        """Someone paid it attention: they are here (a cue for the day's rhythm), a bid it
        made is answered, and any "I stopped asking" is over. ``by`` names a reaction a
        moment can get (touch, praise, voice, name): it may have been for the moment."""
        s = self.s
        self._answered_sounds()
        s.presence_at = s.now
        if self.aware:
            daycycle.hear(self, daycycle.PERSON_CONTACT, room=True)  # (known from the input)
            if s.family in ("away", "unknown", "asleep"):
                # Someone is right here (an interaction is certain): the belief changes
                # now, not at the next beat, so nothing it does next assumes an empty house.
                daycycle.update_family(self, daycycle.flush(self))
                daycycle.refresh_phase(self)
        if s.bid_pending_since is not None:
            after = int((s.now - s.bid_pending_since).total_seconds())
            s.bid_pending_since = None
            self._emit("duck.bid_answered", {"after_s": after})
            self._contact("bid_answered")
        s.unanswered = 0
        s.bid_backoff = SCALE
        s.bids_stopped = False
        moments.presence(self, by, evidence)

    def _answered_sounds(self) -> None:
        """Someone answered what it just said by itself (within 3 s): early on (growth
        days 3–8, before its first word) that sound counts toward its first word; a babble
        answered is its own songbird experience."""
        s = self.s
        flag = self._flag_within(PRAISE_WINDOW)
        if flag is not None and flag["kind"] == "phrase" and s.presence_at != s.now:
            self._self_note("songbird", str(flag["event"]), "babble_answered")
        said = s.last_sounds
        if said is None or s.now - said["at"] > PRAISE_WINDOW or s.first_word is not None:
            return
        low, high = content.FIRST_WORD_DAYS
        if not int(low) <= s.gd <= int(high):
            return
        s.last_sounds = None  # one answer per thing it said
        for tag in sorted(set(said["tags"])):
            s.word_answers[tag] = s.word_answers.get(tag, 0) + 1

    def _contact(self, kind: str) -> None:
        """Any contact lowers the wish for company a little (never below the floor)."""
        social: dict[str, Any] = content.SOCIAL
        amount = int(social["contact"][kind])
        self.s.social = max(int(social["floor"]), self.s.social - amount)

    def stage(self) -> content.Stage:
        gd = self.s.gd
        cached = self._stage_cache
        if cached is None or cached[0] != gd:
            cached = self._stage_cache = (gd, self._stage_at(gd))
        return cached[1]

    def memory(self) -> Distilled:
        if self._memory is None or self._memory_len != len(self.log.events):
            self._memory = distill(self.log.events, self.s.now, self.s.name)
            self._memory_len = len(self.log.events)
        return self._memory

    # ------------------------------------------------------------------ hatching

    def _temperament(self) -> None:
        """Its temperament (spec 7.2): seven axes and a primary temperament, drawn once from
        the seed (never from the decision counters). Its nature (the four branches' weights)
        leans on the primary one: all four are equally likely to lead."""
        s = self.s
        t: dict[str, Any] = content.TEMPERAMENT
        low, high = (int(value) for value in t["range"])
        for axis in t["axes"]:
            raw = rng.draw(self.seed, self.table, f"temperament.{axis}", 1)
            s.temperament[axis] = low + rng.unit(raw) * (high - low) // SCALE
        branches = [branch.branch_id for branch in content.BRANCHES]
        primary = branches[
            rng.pick_index(rng.draw(self.seed, self.table, "temperament.primary", 1), 4)
        ]
        base, span, extra = (int(value) for value in t["nature"])
        weights = {
            branch_id: base
            + rng.unit(rng.draw(self.seed, self.table, f"temperament.{branch_id}", 1))
            * span
            // SCALE
            + (extra if branch_id == primary else 0)
            for branch_id in branches
        }
        total = sum(weights.values())
        s.nature = {key: value * SCALE // total for key, value in sorted(weights.items())}
        s.branch_detail = {"primary": primary}
        # Its own pace of growing up (spec 7.6): 0.87–1.15×, from its tempo.
        pace = self._pace()
        s.stage_days = {
            stage.stage_id: (stage.gd_min * pace + SCALE // 2) // SCALE for stage in content.STAGES
        }

    def _pace(self) -> int:
        low, high = (int(value) for value in content.TEMPERAMENT["pace"])
        return low + (int(self.s.temperament.get("tempo", 5000)) - 2000) * (high - low) // 6000

    def _axis(self, axis: str, key: str) -> int:
        """A temperament axis as a multiplier (fixed point, TEMPERAMENT[key]'s range)."""
        low, high = (int(value) for value in content.TEMPERAMENT[key])
        value = int(self.s.temperament.get(axis, 5000))
        return low + (value - 2000) * (high - low) // 6000

    def _stage_day(self, stage_id: str) -> int:
        """The growth day this duck reaches a stage (its own pace)."""
        days = self.s.stage_days
        if stage_id in days:
            return int(days[stage_id])
        return next(stage.gd_min for stage in content.STAGES if stage.stage_id == stage_id)

    def _stage_at(self, gd: int) -> content.Stage:
        current = content.STAGES[0]
        for stage in content.STAGES:
            if gd >= self._stage_day(stage.stage_id):
                current = stage
        return current

    def _calendar_gd(self, index: int, milestone: content.Milestone) -> int:
        """When a milestone unlocks for this duck: its base day at its pace, the usual
        ±2 jitter, in the calendar's order (the stages' own days exactly)."""
        s = self.s
        fixed = {
            "hatched": 0,
            "duckling": self._stage_day("duckling"),
            "branch": self._stage_day("juvenile"),
            "adult": self._stage_day("adult"),
        }
        if milestone.milestone_id in fixed:
            return fixed[milestone.milestone_id]
        jitter = 0
        if milestone.jitter:
            raw = rng.draw(self.seed, self.table, f"jitter.{milestone.milestone_id}", 1)
            jitter = rng.pick_index(raw, 5) - 2
        gd = (milestone.base_gd * self._pace() + SCALE // 2) // SCALE + jitter
        before = [int(s.milestones[m.milestone_id]["gd"]) for m in content.CALENDAR[:index]] or [0]
        after = [
            fixed[m.milestone_id] for m in content.CALENDAR[index + 1 :] if m.milestone_id in fixed
        ]
        high = after[0] - 1 if after else gd
        return max(max(before) + 1, min(gd, high))

    def _hatch(self) -> None:
        s = self.s
        self._temperament()
        side = "左" if rng.below(rng.draw(self.seed, self.table, "quirk.side", 1), 5000) else "右"
        favourite = content.TAGS[rng.pick_index(rng.draw(self.seed, self.table, "quirk.tag", 1), 4)]
        habit_id, habit_text = content.QUIRK_HABITS[
            rng.pick_index(
                rng.draw(self.seed, self.table, "quirk.habit", 1), len(content.QUIRK_HABITS)
            )
        ]
        tempo = 8500 + rng.unit(rng.draw(self.seed, self.table, "quirk.tempo", 1)) * 3000 // SCALE
        s.quirks = {
            "tilt_side": side,
            "favorite_tag": favourite,
            "habit": habit_id,
            "habit_text": habit_text,
            "tempo": str(tempo),
        }
        s.unlocked = [
            verb.verb_id
            for verb in content.VERBS
            if verb.unlock_gd == 0 and verb.kind not in ("signature", "pause", "moment")
        ]
        for index, milestone in enumerate(content.CALENDAR):
            s.milestones[milestone.milestone_id] = {
                "gd": self._calendar_gd(index, milestone),
                "status": "locked",
                "title": milestone.title,
                "order": index,
            }
        for key in content.CARE_TYPES:
            s.care_today[key] = 0
            s.care_window[key] = 0
            s.care_adult[key] = 0
        hatched = self._emit(
            "sim.hatched",
            {
                "name": s.name,
                "seed": self.seed,
                "table_hash": self.table,
                "engine_rev": ENGINE_REV,
                "hardware": {
                    "tof": self.hw.tof,
                    "head_imu": self.hw.head_imu,
                    "pickup_detect": self.hw.pickup_detect,
                    "ear": self.hw.ear,
                    "name_ear": self.hw.name_ear,
                    "motion_validated": self.hw.motion_validated,
                    **(
                        {
                            "nest": True,
                            "nest_charge_per_min": self.hw.nest_charge_per_min,
                            "nest_awake_drain": self.hw.nest_awake_drain,
                            "camera": self.hw.camera,
                        }
                        if self.hw.nest
                        else {}
                    ),
                },
                "conservative": s.conservative,
                "battery_help": s.battery_help,
                **({"day_aware": True} if s.day_aware else {}),
                "nature": dict(s.nature),
                "temperament": dict(s.temperament),
                "primary": s.branch_detail.get("primary"),
                "stage_days": dict(s.stage_days),
                "rules": dict(s.rules),
                "expression_caps": (
                    hashlib.sha256(json.dumps(self.caps, sort_keys=True).encode()).hexdigest()[:16]
                    if self.caps is not None
                    else None
                ),
            },
            source="sim",
        )
        self._journal(
            "owner", "hatch", f"開箱就孵化（{s.name}，種子 {self.seed}）", True, [hatched]
        )
        if self.hw.nest:
            # Its nest is its first home: it hatches in it, on the floor.
            s.nested = True
            s.floor = True
        if self.aware:
            daycycle.hatch(self)
        self.power_on("owner")
        s.milestones["hatched"]["status"] = "pending"
        self._debut("hatched")
        self._reveal_quirk("tilt_side", f"我歪頭總是往{side}邊歪。")

    # ------------------------------------------------------------------ time

    def _finalize_day(self) -> None:
        s = self.s
        old_gd = s.gd
        awake_min = s.day_awake_s // 60
        if s.ntp_ok:
            if awake_min >= 15:
                s.gd2 += 2
                s.absence_half_used = 0
                credit = "full"
            elif s.absence_half_used < 10:
                s.gd2 += 1
                s.absence_half_used += 1
                credit = "half"
            else:
                credit = "none"
        else:
            credit = "paused"
        saved_now = s.now
        s.now = datetime.combine(s.day_date + timedelta(days=1), datetime.min.time(), TPE)
        self._emit(
            "growth.gd",
            {
                "date": s.day_date.isoformat(),
                "awake_min": awake_min,
                "credit": credit,
                "gd": s.gd,
                "gd2": s.gd2,
            },
            source="clock",
        )
        if s.day_voice or s.day_noise or s.day_tv_min:
            self._emit(
                "env.daily_sound",
                {
                    "date": s.day_date.isoformat(),
                    "voice": s.day_voice,
                    "noise": s.day_noise,
                    "tv_minutes": s.day_tv_min,
                },
                source="self",  # its own count of what it heard (phase 4: not the ear's)
            )
        if not s.owner_seen_today:
            s.bond_owner = approach(
                s.bond_owner, 2000, 228
            )  # about a 30-day half-life toward neutral
        self._day_stats()
        s.day_date = s.day_date + timedelta(days=1)
        s.day_awake_s = 0
        s.day_voice = s.day_noise = s.day_tv_min = 0
        s.owner_seen_today = False
        s.bids_today = 0
        s.rare_today = {}
        moments.new_day(self)
        if self.aware:
            daycycle.new_day(self)
        if s.gd != old_gd:
            self._on_new_gd(old_gd)
        s.now = max(saved_now, s.now)

    def _day_stats(self) -> None:
        """One line a day of what the day was like (duck.day_stats), then start afresh."""
        s = self.s
        if s.day_awake_min or s.day_offnest_min or self.aware:
            self._emit(
                "duck.day_stats",
                {
                    "date": s.day_date.isoformat(),
                    "awake_min": s.day_awake_min,
                    "lonely_min": s.day_lonely_min,
                    "alone_awake_min": s.day_alone_awake_min,
                    "offnest_min": s.day_offnest_min,
                    "walk_s": s.day_walk_s,
                    "turn_s": s.day_turn_s,
                    "docks": s.day_docks,
                    "excursions": s.day_excursions,
                    "drain_units": s.day_drain,
                    "social_mean": s.day_social_sum // s.day_awake_min if s.day_awake_min else None,
                    "phase_min": dict(sorted(s.day_phase_min.items())),
                    "battery": s.battery,
                },
                source="self",
            )
        s.day_awake_min = s.day_lonely_min = s.day_alone_awake_min = 0
        s.day_offnest_min = s.day_walk_s = s.day_turn_s = s.day_drain = 0
        s.day_docks = s.day_excursions = s.day_social_sum = 0
        s.day_phase_min = {}

    def _on_new_gd(self, old_gd: int) -> None:
        s = self.s
        gd = s.gd
        s.pref_gd_count = {}
        if self.aware:
            daycycle.relax_models(self)
        if self.hw.nest:
            excursion.relax_good(self)
        relaxed = 0
        for key in sorted(s.prefs):
            value = s.prefs[key]
            new_value = 5000 + (value - 5000) * PREF_RELAX_KEEP // SCALE
            if new_value != value:
                s.prefs[key] = new_value
                relaxed += 1
        if relaxed:
            self._emit(
                "learning.relaxed",
                {"count": relaxed, "keep": PREF_RELAX_KEEP, "rule": "pref-relax-v1"},
                source="clock",
            )
        s.owner_bond_gd = 0
        s.not_that_gd = set()
        s.care_today = {key: 0 for key in content.CARE_TYPES}
        s.self_today = {}
        if gd >= content.FIRST_STEPS_GD and "wander" not in s.unlocked:
            s.unlocked.append("wander")  # the body is ready to try; the first try is a milestone
        old_stage = self._stage_at(old_gd)
        new_stage = self._stage_at(gd)
        if new_stage.stage_id != old_stage.stage_id:
            self._emit("growth.stage", {"stage": new_stage.stage_id, "zh": new_stage.zh, "gd": gd})
        juvenile, adult = self._stage_day("juvenile"), self._stage_day("adult")
        if old_gd < juvenile <= gd and s.branch is None:
            self._choose_branch()
        if old_gd < adult <= gd and s.adult_name is None:
            self._choose_adult()
        for milestone_id, info in sorted(s.milestones.items(), key=lambda item: item[1]["order"]):
            if info["status"] == "locked" and info["gd"] <= gd:
                info["status"] = "pending"
                info["unlocked_gd"] = gd
                self._emit(
                    "growth.unlocked", {"milestone": milestone_id, "title": info["title"], "gd": gd}
                )
        if gd >= 62 and not s.favorite_found and "favorite_found" not in s.milestones:
            self._experience_milestone("favorite_found", gentle=True)
        if gd >= 120 and s.molt is not None and s.molt.get("status") == "open":
            s.molt["status"] = "kept"
            self._emit("voice.molt_decision", {"choice": "keep", "by": "default"})
        if (
            gd >= 120
            and gd % 30 == 0
            and s.song is not None
            and (s.molt is None or s.molt.get("status") != "open")
        ):
            self._offer_molt()
        if self.s.awake and self._present(15):
            self._debut_pending()

    @_input
    def advance(self, minutes: int) -> dict[str, Any]:
        """Let whole minutes pass (see ``advance_seconds``)."""
        result = self.advance_seconds(max(0, int(minutes)) * 60)
        result["minutes"] = minutes
        return result

    @_input
    def advance_seconds(self, seconds: int) -> dict[str, Any]:
        """Let time pass, to the second. Awake minutes run the full behavior loop; off time jumps.

        Every awake minute has a beat, drawn when the minute starts: the moment its own life
        runs (battery and mood, dozing off, the ordinary decision, babbling), so what the duck
        does falls anywhere in the minute rather than always at its end. Standing ducks also
        decide the moment they have stood long enough, and 07:00 starts the household's day.
        Everything happens at its own second, so the result is the same whatever the step:
        a day at once, minute by minute, or second by second in real time.
        """
        s = self.s
        start_len = len(self.log.events)
        start_gd = s.gd
        remaining = max(0, int(seconds))
        while remaining > 0:
            if s.awake:
                lived = (
                    self._bridged_seconds(remaining) if s.bridged else self._live_seconds(remaining)
                )
            else:
                lived = self._off_seconds(remaining)
            remaining -= lived
        return {
            "seconds": seconds,
            "minutes": seconds // 60,
            "events": len(self.log.events) - start_len,
            "gd_from": start_gd,
            "gd_to": s.gd,
        }

    @_input
    def advance_to(self, t_ms: int, *, stop_on_command: bool = False) -> dict[str, Any]:
        """Live until epoch second ``due_second(t_ms)`` (the port's time axis). With
        ``stop_on_command`` it returns right after the second in which it issued a command
        (the driver then hands the outbox to a body that answers later): each step goes no
        further than :meth:`next_due`, so a command is never issued inside a step that goes
        on past it. Any cut of the same span lives the same duck (step-size invariance)."""
        s = self.s
        seconds = due_second(t_ms) - (s.now - _EPOCH) // _SECOND
        if seconds <= 0:
            return {"seconds": 0, "minutes": 0, "events": 0, "gd_from": s.gd, "gd_to": s.gd}
        if not stop_on_command:
            return self.advance_seconds(seconds)
        start_len = len(self.log.events)
        start_gd = s.gd
        sent = self.acts.n_cmd
        remaining = seconds
        while remaining > 0:
            budget = min(remaining, max(1, self.next_due() // 1000 - (s.now - _EPOCH) // _SECOND))
            remaining -= self._live_seconds(budget) if s.awake else self._off_seconds(budget)
            if self.acts.n_cmd != sent:
                break
        return {
            "seconds": seconds - remaining,
            "minutes": (seconds - remaining) // 60,
            "events": len(self.log.events) - start_len,
            "gd_from": start_gd,
            "gd_to": s.gd,
            "commanded": self.acts.n_cmd != sent,
        }

    def next_due(self) -> int:
        """The next moment (epoch ms, after now) Duck Core has something to do by itself: its
        minute's beat (a new minute's beat is not drawn yet: one second ahead), a standing
        duck's next move, 07:00, midnight, a stir, a nap's end, the day's next alarm, an act's
        deadline, the port's next percept. An upper bound on how long a driver may sleep;
        advancing further is still exact."""
        s = self.s
        now = s.now
        second = timedelta(seconds=1)
        found: list[datetime] = [self._next_midnight()]
        if s.awake:
            if s.tick_s == 0 and s.beat_s < 0:
                found.append(now + second)
            elif not s.beat_done:
                found.append(now + timedelta(seconds=max(1, s.beat_s - s.tick_s)))
            else:
                found.append(now + timedelta(seconds=max(1, 60 - s.tick_s)))
            due = self._stand_due()
            if due is not None:
                found.append(due)
            found.append(self._morning_due())
            if s.napping_until is not None:
                found.append(s.napping_until)
            if self.aware:
                alarm = daycycle.next_alarm(self)
                if alarm is not None:
                    found.append(alarm)
            wake = curiosity.next_due(self)
            if wake is not None:
                found.append(wake)
        if s.stir_at is not None:
            found.append(s.stir_at)
        now_ms = self.now_ms()
        best = min(
            (((moment - _EPOCH) // _SECOND) * 1000 for moment in found if moment > now),
            default=now_ms + 1000,
        )
        port = self.acts.port
        for t_ms in (port.next_at_ms() if port is not None else None, self.acts.next_deadline_ms()):
            if t_ms is not None:
                best = min(best, due_second(t_ms) * 1000)
        return max(best, now_ms + 1000)

    def _minute_done(self, seconds: int) -> bool:
        """Count ``seconds`` into the current minute; True when that minute is over."""
        s = self.s
        s.tick_s += seconds
        if s.tick_s < 60:
            return False
        s.tick_s = 0
        s.beat_s = -1
        s.beat_done = False
        s.sleep_minute = False
        return True

    def _off_seconds(self, budget: int) -> int:
        """Switched off: time jumps (on its cable it still charges, a minute at a time)."""
        s = self.s
        s.beat_s, s.beat_done, s.sleep_minute = -1, False, False
        if s.tick_s == 0 and budget >= 60:
            minutes = budget // 60
            self._set_now(s.now + timedelta(minutes=minutes))
            s.battery = min(SCALE, s.battery + self._charge_per_min(asleep=True) * minutes)
            return minutes * 60
        step = min(budget, 60 - s.tick_s)
        self._set_now(s.now + timedelta(seconds=step))
        if self._minute_done(step):
            s.battery = min(SCALE, s.battery + self._charge_per_min(asleep=True))
        return step

    def _charge_per_min(self, *, asleep: bool) -> int:
        """What its charger gives a minute: the cable, the nest (net of an awake duck's
        use), or nothing."""
        s = self.s
        if s.nested:
            return self.hw.nest_charge_per_min - (0 if asleep else self.hw.nest_awake_drain)
        return CHARGE_PER_MIN if s.docked else 0

    def _live_seconds(self, budget: int) -> int:
        """Awake: up to the next point in this minute (its beat, or its end). Returns the
        seconds lived (0 when the beat itself just ran)."""
        s = self.s
        if s.tick_s == 0 and s.beat_s < 0:
            # A new minute. Asleep on its charger at night nothing is decided: the minute
            # only refills energy and battery (and the duck stirs now and then).
            s.sleep_minute = self._can_bulk_sleep()
            s.beat_done = s.sleep_minute
            s.beat_s = 60 if s.sleep_minute else rng.pick_index(self._rand("beat"), 60)
        if s.sleep_minute:
            if s.tick_s == 0 and budget >= 60:
                # Whole sleeping minutes at once, only as far as a minute-by-minute run
                # would sleep through in one go (the nap's end, and never past 07:00).
                morning = self._morning_due()
                end = min(
                    s.napping_until or s.now,
                    morning - timedelta(seconds=1),
                    s.now + timedelta(minutes=budget // 60),
                )
                if self.aware:
                    alarm = daycycle.next_alarm(self)
                    if alarm is not None:
                        end = min(end, alarm - timedelta(seconds=1))
                    # The day's numbers close at midnight: never sleep across it in one go.
                    end = min(end, self._next_midnight() - timedelta(seconds=1))
                # Nor across what its world has on the way (a sender's touch, an act's
                # deadline): that minute is lived as a minute, where a percept that wakes it
                # takes the minute's rest (LegacyWorld has nothing due while it sleeps).
                due = self._world_next_ms()
                if due is not None:
                    end = min(end, worldport.at_ms(due, TPE) - timedelta(seconds=1))
                minutes = int((end - s.now).total_seconds() // 60)
                if minutes >= 1:
                    self._spend(minutes * 60)
                    self._slept(minutes)
                    s.beat_s = -1  # the next minute decides afresh
                    return minutes * 60
            step = min(budget, 60 - s.tick_s)
            self._spend(step)
            refill = s.sleep_minute  # woken during the minute: that minute refills nothing
            if self._minute_done(step) and refill:
                self._slept(1)
            return step
        if not s.beat_done:
            wait = s.beat_s - s.tick_s
            if wait <= 0:
                s.beat_done = True
                self._minute()
                return 0
            step = min(budget, wait)
        else:
            step = min(budget, 60 - s.tick_s)
        self._spend(step)
        self._minute_done(step)
        return step

    def _bridged_seconds(self, budget: int) -> int:
        """Awake and attached to the neural duck: time passes (the day's awake seconds count, the
        midnights close the days), and each minute only keeps the growth's own books
        (:meth:`_bridged_minute`). No beat, no decision, no nap, no walk of its own."""
        s = self.s
        step = min(budget, 60 - s.tick_s)
        self._pass(step)
        if self._minute_done(step):
            self._bridged_minute()
        return step

    def _bridged_minute(self) -> None:
        """One awake minute attached: the mood settles as it always does, the day counts the
        minute, and a milestone waiting for someone to be there debuts when someone is."""
        s = self.s
        s.valence = approach(s.valence, 0, 300)
        s.arousal = approach(s.arousal, 2000, 500)
        s.overload = approach(s.overload, 0, 1500)
        s.day_awake_min += 1
        if self._present(15):
            self._debut_pending()

    @_input
    def bridge_on(self, api: str) -> dict[str, Any]:
        """Attach this Duck Core to the neural duck (the input that marks the seam in a saved
        duck). From here on it simulates no body and no experiences of its own: the walk, the
        nap, the outing, the babble, the bids it was living are ended where they stand, and what
        it learns from now on is what :meth:`experience` is given. Refused for a duck with a nest
        or a family routine (the physical nest and family are the neural duck's)."""
        from duckmind.duckcore import bridge, bridge_v2

        s = self.s
        if api not in bridge_v2.APIS:
            raise bridge.BridgeError(bridge.Code.UNKNOWN_API, api=api, takes=list(bridge_v2.APIS))
        if self.hw.nest or s.day_aware:
            raise bridge.BridgeError(bridge.Code.NOT_FOR_A_NEST)
        if s.bridged:
            return {"note": "already attached"}
        if s.walk_plan is not None:
            self._cut_walk("attached")
        if s.righting is not None:
            self._righting_cut("attached")
        if self._until(s.napping_until):
            self._end_nap("attached")
        if s.excursion is not None:
            excursion.end(self, "attached", decide=False)
        s.bridged = True
        s.posture = "sit"
        s.outing_legs = 0
        s.stood_at = s.stand_idle_from = None
        s.napping_until = s.nap_reason = s.stir_at = s.stir_next = s.nap_block_until = None
        s.sleep_minute = False
        s.need = None
        s.homing = s.stranded = None
        s.bid_pending_since = None
        s.last_flag = None
        s.tick_s, s.beat_s, s.beat_done = 0, -1, False
        self._emit("sim.bridged", {"api": api, "gd": s.gd}, source="sim")
        return {"note": "attached to the neural duck", "api": api}

    @_input
    def experience(self, event: Mapping[str, Any]) -> dict[str, Any]:
        """One thing the neural duck lived (``bridge-v2``, :mod:`duckmind.duckcore.bridge_v2`: the
        frozen ``bridge-v1`` and the trick lessons) — an input handler like any other: it lives on to
        the event's second, writes the L1 record of the input it has just received (the only place a
        sense-like record of the neural duck is written), and counts it by Duck Core's own growth
        rules."""
        from duckmind.duckcore import bridge_v2

        ev, awake = bridge_v2.arrive(self, event)
        if not awake:
            return {"taken": False, "kind": ev["k"], "note": "off"}
        spec = bridge_v2.record_of(self, ev)
        record = None if spec is None else self._emit(spec.kind, spec.payload, source=spec.source)
        note = bridge_v2.HANDLERS[ev["k"]](self, ev, record)
        return {"taken": True, "kind": ev["k"], "note": note}

    def minutes_until(self, clock: str) -> int:
        """Whole minutes from now until the next ``HH:MM`` on the duck's clock (≥ 1)."""
        hour, minute = (int(part) for part in clock.split(":", 1))
        if not (0 <= hour < 24 and 0 <= minute < 60):
            raise ValueError(f"not a time of day: {clock}")
        now = self.s.now
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return max(1, -(-int((target - now).total_seconds()) // 60))

    def _slept(self, minutes: int) -> None:
        """Minutes asleep on its charger: energy and battery refill, the wish for company
        settles, and the day's numbers count them."""
        s = self.s
        s.energy = min(SCALE, s.energy + NAP_REFILL_PER_MIN * minutes)
        s.battery = min(SCALE, s.battery + self._charge_per_min(asleep=True) * minutes)
        social: dict[str, Any] = content.SOCIAL
        target, step = int(social["asleep_target"]), int(social["asleep_step"])
        for _ in range(minutes):  # minute by minute, so a long step and short ones agree
            if s.social == target:
                break
            s.social = approach(s.social, target, step)
        if self.aware:
            s.day_phase_min[s.phase] = s.day_phase_min.get(s.phase, 0) + minutes
            if not s.nested:
                s.day_offnest_min += minutes

    def _can_bulk_sleep(self) -> bool:
        s = self.s
        if self.aware:
            # Asleep in the nest (or on its cable) and nobody right here: minutes pass
            # without a decision until the nap ends or the day's rhythm has to look again.
            if not (s.nested or s.docked) or not self._until(s.napping_until):
                return False
            if s.cue_buffer or self._present(10):
                return False
            alarm = daycycle.next_alarm(self)
            return alarm is None or alarm > s.now + timedelta(seconds=59)
        if not s.docked or self._present(10) or not self._until(s.napping_until):
            return False
        return s.now.hour >= 22 or s.now.hour < 7

    # ------------------------------------------------------------------ the awake minute

    def _minute(self) -> None:
        """The minute's own life, at its beat: battery and mood, dozing off, the ordinary
        decision, babbling and bids."""
        s = self.s
        if not s.awake:
            return
        if self.hw.nest:
            nest.battery_minute(self)
        elif s.docked:
            s.battery = min(SCALE, s.battery + CHARGE_PER_MIN)
        else:
            drain = BATTERY_DRAIN_PER_MIN
            if s.posture in STANDING:
                drain = drain * STAND_DRAIN // SCALE
            s.battery = max(0, s.battery - drain)
            s.day_drain += drain
            if s.battery_help and s.battery < MIN_MOTION_BATTERY:
                self._swap_battery("family")
            if s.battery <= BEDTIME_BATTERY and not s.bedtime_done:
                self._bedtime()
            if s.battery <= EMPTY_BATTERY:
                self._power_down("battery_empty")
                return
        if not s.awake:
            return
        if self.aware:
            daycycle.minute(self)
        self._relax_minute()
        if s.tv_on:
            self._tv_minute()
        if self._until(s.napping_until):
            self._nap_minute()
            self._social_minute(asleep=True)
            self._day_minute(asleep=True)
            return
        if s.napping_until is not None:
            self._end_nap("rested")
            if not self.aware:
                self._perk_up("rested")
        self._social_minute(asleep=False)
        self._day_minute(asleep=False)
        self._check_bid_answer()
        moments.minute(self)
        if self.aware:
            self._company_minute()
        if self._present(15):
            self._debut_pending()
        if self._until(s.stop_until) or self._until(s.lockout_until) or s.yielding or s.hot:
            return
        if s.posture == "fallen":
            # Lying on its side until someone sets it upright: now and then it struggles
            # again (not another try that counts, just visible effort).
            # (An effort it makes, an output of its own: duck.struggle, not a sensation.)
            if s.righting is None and rng.below(self._rand("struggle"), STRUGGLE_PER_MIN):
                self._emit("duck.struggle", {"at": self._pose_list()})
            return
        if self.aware:
            nest.retry_stranded(self)
            if (
                nest.maybe_rest(self)
                or nest.errand_minute(self)
                or curiosity.minute(self)
                or excursion.minute(self)
            ):
                self._maybe_bid()
                return
            started = nest.maybe_leave(self) or excursion.maybe_start(self)
            share = 0 if started else self.stage().active_share
            # A duck that goes out dozens of times a day shows itself mostly by walking:
            # fewer idle head moves between trips, and calmer still resting in its nest.
            share = share * (NESTED_IDLE_SHARE if s.nested else NEST_IDLE_SHARE) // SCALE
            if s.stranded is not None:
                share = share * int(content.NEST["stranded_share"]) // SCALE
        else:
            quiet_hours = s.docked and (s.now.hour >= 22 or s.now.hour < 7)
            sleepy = (s.energy < 3000 and not self._present(3)) or (
                quiet_hours and not self._present(10)
            )
            if sleepy and not self._until(s.nap_block_until):
                self._start_nap(quiet_hours)
                return
            if self._maybe_set_off():
                share = 0  # it got up to walk about this minute
            else:
                share = self.stage().active_share
                if s.boot_at is not None and s.now - s.boot_at <= timedelta(minutes=15):
                    share = max(share, 6000)
        if share and rng.below(self._rand("activity"), share):
            self._decide_and_express()
        moments.idle(self)
        self._maybe_babble()
        self._maybe_bid()
        self._maybe_rare()

    def _social_minute(self, *, asleep: bool) -> None:
        """How much it misses the family, minute by minute (spec v4 part D).

        Asleep it settles. Someone right here: calmer (never below the floor: company
        never makes it not need the family). The family home but elsewhere, or itself safe
        on its nest while they are out: it misses them only a little. Awake, off the nest,
        nobody for a while: more, and more so the younger (and the more sociable) it is.
        """
        s = self.s
        social: dict[str, Any] = content.SOCIAL
        if asleep:
            s.social = approach(s.social, int(social["asleep_target"]), int(social["asleep_step"]))
            return
        if self._present(int(content.RHYTHM["room_timeout_min"])):
            s.social = max(int(social["floor"]), s.social - int(social["calm_present"]))
            if self.aware:
                # The family right here is a safe base: a little happier for it (spec 6.5).
                s.valence = approach(s.valence, PRESENT_VALENCE, 600)
            return
        rise = int(social["rise_home"])
        alone = not self._present(int(social["alone_after_min"]))
        if self.aware:
            home = s.family in ("awake", "asleep")
            if alone and not home and not s.nested and not s.docked:
                rise = int(social["rise_alone"][self.stage().stage_id])
        elif alone:
            rise = int(social["rise_alone"][self.stage().stage_id])
        rise = rise * self._style_social() // SCALE
        rise = rise * self._axis("sociability", "social") // SCALE
        s.social = min(8000, s.social + rise)

    def _style_social(self) -> int:
        """Personality and company: a cuddly duck misses you more, an explorer less
        (always more than nothing)."""
        s = self.s
        table = {"snuggler": 12000, "songbird": 10000, "dreamer": 9000, "explorer": 8000}
        if s.branch is not None:
            return table[s.branch]
        return sum(s.nature[branch] * table[branch] for branch in sorted(s.nature)) // SCALE

    def _day_minute(self, *, asleep: bool) -> None:
        s = self.s
        if not asleep:
            s.day_awake_min += 1
            s.day_social_sum += s.social
            if s.social >= 6000:
                s.day_lonely_min += 1
            if self.aware and s.family in ("away", "unknown"):
                s.day_alone_awake_min += 1
        if self.aware:
            s.day_phase_min[s.phase] = s.day_phase_min.get(s.phase, 0) + 1
            if not s.nested:
                s.day_offnest_min += 1

    def _relax_minute(self) -> None:
        s = self.s
        s.valence = approach(s.valence, 0, 300)
        s.arousal = approach(s.arousal, 2000, 500)
        s.overload = approach(s.overload, 0, 1500)
        s.energy = max(
            0, s.energy - (AWAKE_DRAIN_STANDING if s.posture in STANDING else AWAKE_DRAIN)
        )
        s.walk_sat = approach(s.walk_sat, 0, WALK_SAT_RELAX)
        for verb_id in list(s.interest):
            s.interest[verb_id] = min(SCALE, s.interest[verb_id] + 700)
        if s.coo_series_at is not None and s.now - s.coo_series_at >= timedelta(hours=2):
            s.coo_step = 0
            s.coo_series_at = None
        cutoff = s.now - timedelta(hours=1)
        s.self_phrase_times = [moment for moment in s.self_phrase_times if moment > cutoff]
        s.bid_times = [moment for moment in s.bid_times if moment > cutoff]

    def _tv_minute(self) -> None:
        s = self.s
        s.day_tv_min += 1
        s.day_noise += 1
        s.noise_habit = min(SCALE, s.noise_habit + 50)
        if self.hw.ear:
            # Its ear may mistake the TV for a scratch on the head: the world says (a
            # sense.touch marked legacy_phantom, taken in by on_percept; design A10).
            self.acts.poll(worldport.POLL_TV_MINUTE)

    def _nap_minute(self) -> None:
        s = self.s
        s.energy = min(SCALE, s.energy + 300)
        if (
            s.branch == "dreamer"
            and "sleep_talk" in s.unlocked
            and rng.below(self._rand("sleep_talk"), 400)
        ):
            self._express("sleep_talk", reason="nap")

    def _start_nap(
        self,
        quiet_hours: bool,
        *,
        bedtime: bool = False,
        minutes: int | None = None,
        reason: str | None = None,
    ) -> None:
        s = self.s
        if s.yielding or s.posture == "fallen":
            return
        self._sit_down_now("nap")  # it never dozes off standing
        if minutes is not None and reason is not None:
            self._begin_nap(minutes, reason, quiet_hours)
            return
        minutes = 4 + rng.pick_index(self._rand("nap.len"), 9)
        reason = "tired"
        if quiet_hours:
            minutes, reason = 60, "quiet_hours"
        elif bedtime:
            reason = "bedtime"
        else:
            # Tired: sleep until rested again, not a doze that leaves it tired all day. The
            # nap refills at each minute's beat and this minute's beat has run already, so
            # one minute more than the refills it needs (else it woke a refill or two short).
            rested = NAP_RESTED[0] + rng.pick_index(self._rand("nap.rested"), NAP_RESTED[1] + 1)
            need = -(-(rested - s.energy) // NAP_REFILL_PER_MIN) + 1
            minutes = min(NAP_MAX_MIN, max(minutes, need))
        self._begin_nap(minutes, reason, quiet_hours)

    def _begin_nap(self, minutes: int, reason: str, quiet_hours: bool) -> None:
        s = self.s
        s.napping_until = s.now + timedelta(minutes=minutes)
        s.nap_reason = reason
        self._emit(
            "duck.nap_start",
            {
                "minutes": minutes,
                "energy": s.energy,
                "quiet_hours": quiet_hours,
                "reason": reason,
            },
        )
        nap = self._express("nap", reason=reason)
        if minutes >= 20 and reason not in ("quiet_hours", "doze"):
            self._self_note("dreamer", nap, "long_nap")
        self._schedule_stir(carry=reason == "quiet_hours")

    def _end_nap(self, by: str, parents: tuple[str, ...] = ()) -> None:
        """The nap is over (rested, or someone woke it): stretch and carry on."""
        s = self.s
        s.napping_until = None
        s.nap_reason = None
        s.stir_at = None
        if by != "rested":
            s.stir_next = None
        s.sleep_minute = False
        s.nap_ended_at = s.now
        if by != "rested":  # woken up: it stays awake for a while
            s.nap_block_until = s.now + timedelta(minutes=NAP_WAKE_BLOCK_MIN)
        ended = self._emit("duck.nap_end", {"by": by}, parents=parents)
        self._express("stretch_yawn", reason="wake_from_nap", parents=(ended.event_id,))
        if s.quirks.get("habit") == "stretch_after_nap":
            self._reveal_quirk("habit", s.quirks["habit_text"])

    def _asleep_notices(self, trigger: str, parents: tuple[str, ...]) -> bool:
        """Something reaches a dozing duck (the owner's 你好, 乖 or 去走走): in a daytime nap it
        may wake up (drawn); otherwise it opens its eyes for a moment and sleeps on. True
        when it is still asleep (then it does nothing else about it)."""
        if not self._until(self.s.napping_until):
            return False
        if self._nap_wakes(trigger):
            self._end_nap(trigger, parents=parents)
            return False
        self._sleep_stir(trigger, "peek", parents=parents)
        return True

    def _nap_wakes(self, trigger: str) -> bool:
        """Whether this wakes a dozing duck (drawn). At night on its charger it sleeps on."""
        if self.s.nap_reason == "quiet_hours":
            return False
        return rng.below(self._rand("nap.wake"), NAP_WAKE_CHANCE.get(trigger, 0))

    def _schedule_stir(self, *, carry: bool = False) -> None:
        """The next stir in its sleep. A night's sleep is renewed every hour; the stir that
        was due just after the renewal carries over (``carry``), so the night's rhythm of
        stirs never skips a beat at the hour."""
        s = self.s
        s.stir_at = None
        if s.napping_until is None:
            return
        pending = s.stir_next
        s.stir_next = None
        if carry and pending is not None and s.now < pending < s.napping_until:
            s.stir_at = pending
            return
        low, high = STIR_GAP_S if s.nap_reason in ("quiet_hours", "away_rest") else NAP_STIR_GAP_S
        at = s.now + timedelta(seconds=low + rng.pick_index(self._rand("stir.gap"), high - low + 1))
        if at < s.napping_until:
            s.stir_at = at
        elif s.nap_reason == "quiet_hours":
            s.stir_next = at  # after this hour's nap: the next one picks it up

    def _sleep_stir(
        self, reason: str, kind: str | None = None, parents: tuple[str, ...] = ()
    ) -> Event:
        """Asleep, it still moves a little: shifts, nuzzles down, shivers, mumbles, or
        opens its eyes for a moment. A drawn, logged event; it stays asleep."""
        s = self.s
        if reason == "dream":
            s.stir_at = None
        if kind is None:
            weights = dict(STIR_WEIGHTS)
            if self._until(s.quiet_until) or s.hot:
                weights.pop("mumble")
            kind = rng.pick_weighted(self._rand("stir.kind"), weights)
        clip, words = STIR_KINDS[kind]
        sounds = ["coo"] if kind == "mumble" else []
        event = self._emit(
            "duck.sleep_stir",
            {
                "kind": kind,
                "clip": clip,
                "zh": words,
                "reason": reason,
                "sounds": sounds,
                "amplitude": self._amplitude_cap(clip),
            },
            parents=parents,
        )
        if reason == "dream":
            self._schedule_stir()
        return event

    def _swap_battery(self, by: str, parents: tuple[str, ...] = ()) -> Event:
        s = self.s
        before = s.battery
        s.battery = SCALE
        s.bedtime_done = False
        return self._emit(
            "power.battery_swap",
            {"by": by, "before": before * 100 // SCALE},
            source="sim" if by == "family" else "owner",
            parents=parents,
        )

    # ------------------------------------------------------------------ decisions

    def _candidates(self) -> list[str]:
        s = self.s
        if s.posture == "stand" and self._must_sit() is not None:
            return ["sit_down"]
        result = []
        for verb_id in s.unlocked:
            verb = content.VERB_BY_ID[verb_id]
            if verb.kind not in content.IDLE_KINDS:
                continue
            if verb.kind == "motion":
                # Walking on after a walk is natural, so motion verbs may repeat; the posture
                # and safety gates decide what is possible right now.
                if self._motion_block(verb_id) is not None:
                    continue
                if self.hw.nest and verb_id != "sit_down":
                    continue  # with a nest: getting up and walking are its outings
            elif verb_id == s.last_verb:
                continue
            if verb.needs_tof and not self.hw.tof:
                continue
            if verb.branch is not None and verb.branch != s.branch:
                continue
            result.append(verb_id)
        return sorted(result)

    def _standing_options(self, trigger: str) -> list[str]:
        """What a standing duck may do when it has to move on: walk, turn, or sit down."""
        s = self.s
        if s.posture != "stand":
            return []
        if trigger != "stand_limit" or self._must_sit() is not None or self.hw.nest:
            return ["sit_down"]
        return sorted(
            verb_id
            for verb_id in ("sit_down", *content.WALK_VERBS)
            if verb_id in s.unlocked and self._motion_block(verb_id) is None
        )

    # ------------------------------------------------------------------ posture and motion gates

    def _night(self) -> bool:
        return self.s.now.hour >= 22 or self.s.now.hour < 7

    def _motion_block(self, verb_id: str) -> str | None:
        """Why this whole-body verb is impossible right now (None when it is allowed).

        Hard gates, never learned away: posture, floor mode for walking, the physics
        validation, battery, night, and anything that holds the body still.
        """
        s = self.s
        if verb_id == "stand_up":
            if s.posture != "sit":
                return "posture"  # standing up from a fall is never autonomous
        elif s.posture != "stand":
            return "posture"
        if verb_id == "sit_down":
            return None  # the rest pose: always allowed once standing
        if verb_id in content.WALK_VERBS:
            if s.unplaced:
                return "unplaced"  # lifted up: not until it is put back somewhere
            if not s.floor:
                return "desk"
            if s.docked:
                return "tethered"  # on its charging cable
            if s.nested:
                return "charging"  # on the nest's contacts: stand up (off the nest) first
            if not self.hw.pickup_detect:
                return "no_pickup_sensor"  # could not tell that it was lifted off the floor
            if s.stranded is not None:
                return "stranded"
            if self.hw.nest and s.family in ("away", "unknown") and s.homing is None:
                return "alone"  # nobody home: it rests in the nest, no walks
            if self.hw.nest and s.homing is None and nest.low_for_a_walk(self):
                # Its battery is down to what it keeps for the way home (a reflex turn off
                # the nest would take it further than that reserve allows).
                return "going_home"
        elif verb_id == "stand_up" and self._until(s.stand_block_until):
            return "stand_backoff"
        if not self.hw.motion_validated:
            return "unvalidated"
        if not s.docked and s.battery < MIN_MOTION_BATTERY:
            return "battery"
        if self._night():
            return "night"
        if (
            s.yielding
            or s.hot
            or self._until(s.stop_until)
            or self._until(s.lockout_until)
            or self._until(s.napping_until)
            or self._until(s.quiet_until)
            or self.acts.busy()  # a body act still open (one at a time, body.sock)
        ):
            return "busy"
        if curiosity.holds_body(self):
            return "curio"
        return None

    def _must_sit(self) -> str | None:
        """Why a standing duck has to sit down now (None when it may keep standing)."""
        s = self.s
        if s.hot:
            return "hot"
        if self._until(s.napping_until):
            return "nap"
        if not s.docked and (s.battery < SIT_BATTERY or s.bedtime_done):
            return "battery"
        if s.energy < SIT_ENERGY:
            return "tired"
        if self._night():
            return "night"
        return None

    def _stand_limit_s(self) -> int:
        return min(STAND_S_MAX, content.MOTION_STAGE[self.stage().stage_id]["stand_s"])

    def _stand_due(self) -> datetime | None:
        """When a standing duck has stood still long enough that it must walk on or sit
        (on an outing: when it has looked around and walks its next leg)."""
        s = self.s
        if s.posture == "walk" and s.walk_until is not None:
            start = s.walk_until
        elif s.posture == "stand" and s.stand_idle_from is not None:
            start = s.stand_idle_from
        else:
            return None
        if s.yielding:
            return None  # the gamepad has the body
        if s.excursion is not None and s.excursion.get("pause"):
            wait = 0  # a leg just ended: look around now (then pause before the next)
        elif s.homing is not None or s.excursion is not None:
            wait = s.leg_pause_s
        else:
            wait = s.leg_pause_s if s.outing_legs > 0 else self._stand_limit_s()
        due = start + timedelta(seconds=wait)
        if s.stop_until is not None and s.stop_until > s.now:
            due = max(due, s.stop_until)  # a hand is holding it still: never an abrupt sit
        return due

    def _style(self, key: str, leaning: str | None = None) -> int:
        """Personality multiplier: the branch's row, or the nature-weighted blend before it."""
        s = self.s
        if s.branch is not None:
            return content.MOTION_STYLE[s.branch][key]
        blended = (
            sum(s.nature[branch] * content.MOTION_STYLE[branch][key] for branch in sorted(s.nature))
            // SCALE
        )
        if leaning is not None:
            blended = (blended + content.MOTION_STYLE[leaning][key]) // 2
        return blended

    def _tiredness(self) -> int:
        """How much tiredness and a low battery slow its motion (1.0 rested … 0 must rest).

        Tired means fewer and calmer outings, never none, until it is too tired to stand
        (SIT_ENERGY) or the battery reaches the motion floor.
        """
        s = self.s
        if s.energy < SIT_ENERGY:
            return 0
        factor = (
            3000
            + 7000
            * clamp((s.energy - SIT_ENERGY) * SCALE // (TIRED_FULL_ENERGY - SIT_ENERGY))
            // SCALE
        )
        if not s.docked:
            if s.battery < SIT_BATTERY:
                return 0
            battery = (
                5000
                + 5000
                * clamp((s.battery - SIT_BATTERY) * SCALE // (TIRED_FULL_BATTERY - SIT_BATTERY))
                // SCALE
            )
            factor = factor * battery // SCALE
        return factor

    def _outing_chance(self, leaning: str | None) -> int:
        """Per-minute chance that a seated floor duck gets up to walk about (fixed point):
        the first minute's chance, once more for every minute it has sat since."""
        base = self._outing_base(leaning)
        return clamp(base * (SCALE + OUTING_RESTLESS * self.s.restless_min) // SCALE)

    def _outing_base(self, leaning: str | None) -> int:
        """The chance of getting up in its first seated minute (personality, age, mood, what
        the family taught, how recently it walked, how tired it is)."""
        s = self.s
        stage_id = self.stage().stage_id
        chance = OUTING_PER_MIN * self._style("outing", leaning) // SCALE
        chance = chance * content.MOTION_STAGE[stage_id]["outing"] // SCALE
        # Mood: lively and happy a bit more often, flat and low a bit less (0.8–1.25×).
        chance = chance * (8000 + s.arousal * 3 // 10 + s.valence // 5) // SCALE
        pref = s.prefs.get("household.verb:wander", 5000)
        chance = chance * (7500 + pref // 2) // SCALE  # what the family taught: 0.75–1.25×
        chance = chance * (SCALE - s.walk_sat // 2) // SCALE  # just been out: a bit less
        chance = chance * self._tiredness() // SCALE
        if s.conservative:
            chance = chance * CONSERVATIVE_OUTINGS // SCALE
        return clamp(chance)

    def outing_eta(self) -> int | None:
        """Minutes (at most) until a seated floor duck gets up again, as things stand now:
        the minute its restlessness makes getting up certain (None: it cannot, or will not
        before something changes)."""
        if not self._may_set_off():
            return None
        s = self.s
        base = self._outing_base(self._leaning())
        if base <= 0:
            return None
        need = -(-SCALE * SCALE // base)  # SCALE + OUTING_RESTLESS × minutes must reach this
        certain = max(0, -(-(need - SCALE) // OUTING_RESTLESS))
        return max(1, certain - s.restless_min + 1)

    def _may_set_off(self) -> bool:
        """Seated on the floor and free to get up by itself (the hard gates, quiet time)."""
        s = self.s
        return (
            s.posture == "sit"
            and s.floor
            and "stand_up" in s.unlocked
            and self._motion_block("stand_up") is None
            and not self._until(s.quiet_until)
        )

    def _maybe_set_off(self) -> bool:
        """A seated floor duck gets up to go and look about, every few minutes on average.

        Only the hard gates and real rest (a nap, quiet time, too tired) stop it; the
        chance itself carries personality, age, mood and what it learned. The draw is kept
        in the decision it produces.
        """
        s = self.s
        if not self._may_set_off():
            return False
        leaning = self._leaning()
        chance = self._outing_chance(leaning)
        if chance <= 0:
            return False
        raw = self._rand("outing")
        if not rng.below(raw, chance):
            s.restless_min += 1
            return False
        decision = self._emit(
            "duck.decision",
            {
                "verb": "stand_up",
                "rule": "outing-v1",
                "trigger": "restless",
                "table": self.table,
                "counter": self.counters["outing"],
                "draw": rng.unit(raw),
                "inputs": {**self._decision_inputs(leaning), "restless_min": s.restless_min},
                "utilities": {"stand_up": chance, "stay": SCALE - chance},
                "probabilities": {"stand_up": chance, "stay": SCALE - chance},
            },
        )
        self._perform("stand_up", reason="restless", parents=(decision.event_id,))
        self._set_off(leaning)
        return True

    def _set_off(self, leaning: str | None) -> None:
        """Just stood up to go and look about: walk a few legs where it may walk; a duck
        too young to walk (or on the desk) stands and looks around, then sits again."""
        s = self.s
        if not any(
            verb_id in s.unlocked and self._motion_block(verb_id) is None
            for verb_id in content.WALK_VERBS
        ):
            self._decide_and_express("stood")
            return
        chance = LEG_CHANCE * self._style("wander", leaning) // SCALE
        chance = clamp(chance * content.MOTION_STAGE[self.stage().stage_id]["outing"] // SCALE)
        s.outing_legs = sum(
            rng.below(self._rand("outing.legs"), chance)
            for _ in range(CONSERVATIVE_LEGS if s.conservative else EXTRA_LEGS)
        )
        self._decide_and_express("set_off")

    def _perk_up(self, trigger: str, parents: tuple[str, ...] = ()) -> bool:
        """Whatever kept it still is over (a hand let go, it was set down on the floor, it
        was woken, morning came…): it gets up to go and look about. A drawn, logged decision
        like any other: a curious, rested duck nearly always does, a tired one less often,
        and the hard gates still decide whether it may stand at all."""
        s = self.s
        if (
            not s.awake
            or s.yielding
            or s.posture != "sit"
            or self._until(s.napping_until)
            or self._until(s.quiet_until)
            or "stand_up" not in s.unlocked
            or self._motion_block("stand_up") is not None
        ):
            return False
        if self.hw.nest:
            # With a nest, getting up to look about is an outing, with all its rules.
            if excursion.can_start(self) is not None:
                return False
            chance = PERK_UP_STAND * self._style("outing", self._leaning()) // SCALE
            chance = clamp(chance * self._tiredness() // SCALE)
            raw = self._rand("reflex")
            go = rng.below(raw, chance)
            decision = self._emit(
                "duck.decision",
                {
                    "verb": "stand_up" if go else "stay",
                    "rule": "reflex-v1",
                    "trigger": trigger,
                    "table": self.table,
                    "counter": self.counters["reflex"],
                    "draw": rng.unit(raw),
                    "inputs": self._decision_inputs(self._leaning()),
                    "utilities": {"stand_up": chance, "stay": SCALE - chance},
                    "probabilities": {"stand_up": chance, "stay": SCALE - chance},
                },
                parents=parents,
            )
            if not go:
                return False
            reason = "play" if trigger == "invited" else excursion.pick_reason(self)
            return excursion.set_out(self, reason, parents=(decision.event_id,))
        leaning = self._leaning()
        chance = PERK_UP_STAND * self._style("outing", leaning) // SCALE
        chance = chance * self._tiredness() // SCALE
        if self._reflex("stand_up", trigger, chance, parents) is None:
            return False
        self._set_off(leaning)
        return True

    def _morning(self) -> None:
        """07:00 on the duck's clock: the quiet hours are over. A duck asleep on its
        charger for the night wakes up, and one that may stand gets up to start the day."""
        s = self.s
        s.last_morning = s.now.date()
        if not s.awake or self.aware:
            return  # a duck with a nest wakes by itself (daycycle: 07:00–07:04)
        if self._until(s.napping_until) and s.nap_reason == "quiet_hours":
            self._end_nap("morning")
        self._perk_up("morning")

    def _motion_utility(self, verb_id: str, leaning: str | None, trigger: str) -> int:
        s = self.s
        positive = max(0, s.valence)
        if verb_id == "stand_up":
            need = 2500 + s.arousal // 2 + positive // 4
            if self._present(3):
                need += s.social // 4  # someone is here to look at (never required)
        elif verb_id == "wander":
            need = WANDER_NEED + s.arousal // 2 + positive // 5
            need = need * (SCALE - s.walk_sat) // SCALE
        elif verb_id == "turn_around":
            need = 2500 + s.arousal * 4 // 10 + positive // 5
            need = need * (SCALE - s.walk_sat // 2) // SCALE
        else:  # sit_down
            standing_min = 0
            if s.stood_at is not None:
                standing_min = int((s.now - s.stood_at).total_seconds()) // 60
            need = 1500 + (SCALE - s.energy) // 2 + s.walk_sat // 3 + min(4000, standing_min * 800)
            if trigger == "stand_limit":
                need += 1500  # stood still long enough: walk on or sit down
            if self._must_sit() is not None:
                need += 6000
        if verb_id != "sit_down":
            need = need * self._tiredness() // SCALE  # tired: slower to go, never stopped
        pref = s.prefs.get(f"household.verb:{verb_id}", 5000)
        value = need * (5000 + pref // 2) // SCALE
        value = value * s.interest.get(verb_id, SCALE) // SCALE
        value = value * self._style(verb_id, leaning) // SCALE
        value = value * content.MOTION_STAGE[self.stage().stage_id][verb_id] // SCALE
        if verb_id in content.WALK_VERBS and s.conservative:
            value = value * CONSERVATIVE_WALKS // SCALE
        return max(1, value) if verb_id == "sit_down" else value

    def _utility(self, verb_id: str, leaning: str | None, trigger: str = "idle") -> int:
        s = self.s
        verb = content.VERB_BY_ID[verb_id]
        if verb.kind == "motion":
            return self._motion_utility(verb_id, leaning, trigger)
        positive_valence = max(0, s.valence)
        need = {
            "curious": 4000 + s.arousal // 2,
            "calm": 3000 + (SCALE - s.arousal) // 2 + positive_valence // 4,
            "happy": 2000 + positive_valence * 2 // 3 + s.arousal // 4,
            "tired": 1000 + (SCALE - s.energy) * 2 // 3,
            "social": 2500 + s.social // 2,
        }.get(verb.fit, 0)
        pref = s.prefs.get(f"household.verb:{verb_id}", 5000)
        value = need * (5000 + pref // 2) // SCALE
        value = value * s.interest.get(verb_id, SCALE) // SCALE
        if s.branch is not None and (
            verb.fit in FAVOURED_FITS[s.branch] or verb.branch == s.branch
        ):
            value = value * 12000 // SCALE
        elif leaning is not None and verb.fit in FAVOURED_FITS[leaning]:
            value = value * 11500 // SCALE
        if s.posture in STANDING and verb.fit == "happy" and (s.branch or leaning) == "songbird":
            value = value * 11500 // SCALE  # a songbird stands up straight to sing
        return value

    def _leaning(self) -> str | None:
        """Which way it is growing before its branch (from about growth day 28 at its own
        pace): the branch-v2 score's leader, without the lottery."""
        s = self.s
        if s.branch is not None or s.gd < (28 * self._pace() + SCALE // 2) // SCALE:
            return None
        if not sum(s.care_window.values()) and not s.self_window:
            return None
        scores, _, _ = self._branch_scores(s.care_window, content.CARE_PRIOR)
        return max(sorted(scores), key=lambda key: scores[key])

    def _leaning_text(self) -> str:
        """The leaning milestone names both sources: its nature and the family's care."""
        s = self.s
        leaning = self._leaning() or max(sorted(s.nature), key=lambda key: s.nature[key])
        born = max(sorted(s.nature), key=lambda key: s.nature[key])
        growing = f"我好像越來越{LEANING_ZH[leaning]}了。"
        if not sum(s.care_window.values()):
            return f"我天生比較{LEANING_ZH[born]}，{growing}"
        lift = self._lift(s.care_window, content.CARE_PRIOR)
        top = max(sorted(lift), key=lambda key: lift[key])
        return f"我天生比較{LEANING_ZH[born]}，你們又常{CARE_DO_ZH[top]}：{growing}"

    def _first_word(self) -> tuple[str, dict[str, int]]:
        """first-word-v2 (spec 7.5): any of the five words; more likely the one with the
        sound it likes best, and the ones whose sound the family answered early on."""
        s = self.s
        if s.rules.get("first_word") == "first-word-v1":
            return ("hi", "yay")[rng.pick_index(self._rand("first_word"), 2)], {}
        rule: dict[str, int] = content.FIRST_WORD_WEIGHTS
        favourite = s.quirks.get("favorite_tag")
        weights = {}
        for word in content.FIRST_WORDS:
            answered = min(
                rule["answered_cap"], s.word_answers.get(content.FIRST_WORD_SOUND[word], 0)
            )
            weights[word] = (
                rule["base"]
                + (rule["favourite"] if favourite in content.WORDS[word] else 0)
                + rule["answered"] * answered
            )
        return rng.pick_weighted(self._rand("first_word"), weights), weights

    def _decision_inputs(self, leaning: str | None) -> dict[str, Any]:
        s = self.s
        return {
            "energy": s.energy,
            "social": s.social,
            "valence": s.valence,
            "arousal": s.arousal,
            "overload": s.overload,
            "leaning": leaning,
            "posture": s.posture,
            "floor": s.floor,
            "battery": s.battery,
            "walk_sat": s.walk_sat,
            "pose": self._pose_list(),
            "avoid_spots": len(self.active_spots()),
        }

    def _decide_and_express(self, trigger: str = "idle") -> None:
        """One logged utility decision over everything the duck may do right now.

        ``idle`` is the ordinary per-minute choice (head expressions, standing up, sitting
        down, walking). ``stood`` follows a stand-up it chose itself: what it stood up
        for (set off, turn, look around), never straight back down. ``set_off`` follows
        getting up to walk about (``_maybe_set_off``): walk or turn. ``stand_limit`` comes
        when a standing duck has stood still long enough: walk on, turn, or sit. Any other
        trigger (``nap``, ``hot``…) names why a standing duck has to sit down now.
        """
        s = self.s
        if trigger == "idle":
            candidates = self._candidates()
        elif trigger == "stood":
            candidates = [verb_id for verb_id in self._candidates() if verb_id != "sit_down"]
        elif trigger in ("set_off", "walk_on"):
            # It got up to walk about (set off, or turn to look first), or it is on an
            # outing and walks its next leg; if it cannot walk on, it may sit.
            candidates = [
                verb_id
                for verb_id in content.WALK_VERBS
                if verb_id in s.unlocked and self._motion_block(verb_id) is None
            ]
            if trigger == "walk_on":
                s.outing_legs -= 1
                if not candidates or self._must_sit() is not None:
                    s.outing_legs = 0
                    candidates = self._standing_options("stand_limit")
        else:
            candidates = self._standing_options(trigger)
        if not candidates:
            return
        leaning = self._leaning()
        utilities = {verb_id: self._utility(verb_id, leaning, trigger) for verb_id in candidates}
        top = max(utilities.values())
        if top <= 0:
            return
        # A pet is not a best-choice machine: anything at least a quarter as appealing as
        # the favourite stays in the draw, weighted in plain proportion (the family asked
        # for livelier, less predictable behaviour than the old top-half, squared draw).
        kept = {verb_id: value for verb_id, value in utilities.items() if value * 4 >= top}
        exponent = 1
        weights = {
            verb_id: max(1, value**exponent // SCALE ** (exponent - 1))
            for verb_id, value in kept.items()
        }
        raw = self._rand("decide")
        choice = rng.pick_weighted(raw, weights)
        total = sum(weights.values())
        decision = self._emit(
            "duck.decision",
            {
                "verb": choice,
                "rule": "decide-v1",
                "trigger": trigger,
                "table": self.table,
                "counter": self.counters["decide"],
                "draw": rng.unit(raw),
                "exponent": exponent,
                "inputs": self._decision_inputs(leaning),
                "utilities": utilities,
                "probabilities": {
                    verb_id: weight * SCALE // total for verb_id, weight in weights.items()
                },
            },
        )
        self._perform(
            choice, reason="idle" if trigger == "idle" else trigger, parents=(decision.event_id,)
        )
        if choice == "stand_up" and trigger == "idle":
            # It stood up for something: set off (on the floor) or look around, right away.
            self._decide_and_express("stood")

    # ------------------------------------------------------------------ whole-body motion

    def _perform(
        self,
        verb_id: str,
        *,
        reason: str,
        parents: tuple[str, ...] = (),
        force_sound: tuple[str, ...] | None = None,
    ) -> Event:
        """Express a verb; for the whole-body verbs also change posture or walk."""
        s = self.s
        if verb_id not in content.MOTION_VERBS:
            return self._express(verb_id, reason=reason, parents=parents, force_sound=force_sound)
        before = s.posture
        if verb_id in content.WALK_VERBS:
            # Plan first: a walk that would leave the room or head into a spot it learned to
            # avoid turns away instead, and the expression shows what it really does.
            plan = self._plan_walk(verb_id, short=reason == "name_turn")
            event = self._express(
                str(plan["verb"]), reason=reason, parents=parents, force_sound=force_sound
            )
            self._walk(plan, event)
            return event
        stood: Event | None = None
        if verb_id == "stand_up":
            stood, failed = self._stand_try(parents)
            if failed:
                return stood
            if s.nested:
                nest.undock(self, "stand_up", parents)
            s.posture = "stand"
            s.stood_at = s.now
            s.stand_idle_from = s.now
            s.stands += 1
            s.restless_min = 0
            s.energy = max(0, s.energy - STAND_UP_ENERGY)
        else:  # sit_down
            s.posture = "sit"
            s.outing_legs = 0
            s.stood_at = None
            s.stand_idle_from = None
            if s.excursion is not None:
                excursion.end(self, reason, decide=False)
        event = self._express(verb_id, reason=reason, parents=parents, force_sound=force_sound)
        if verb_id == "sit_down" and self.hw.nest and reason != "dock":
            nest.reseat(self)
        if stood is None:
            # (Standing up needs no record of its own: the body's body.posture is it.)
            self._emit(
                "duck.posture",
                {"posture": s.posture, "from": before, "reason": reason, "by": "self"},
                parents=(event.event_id,),
            )
        return event

    def _stand_try(self, parents: tuple[str, ...]) -> tuple[Event, bool]:
        """Ask the body to stand up (``duck.act {cmd: duck.posture, want: stand}``) and take
        its answer: the body.posture record, and whether it failed. Getting up can fail (the
        official body server manages ~2 in 3): then it stays seated, tries again in two
        minutes, and after two misses in ten minutes it rests half an hour first
        (stand-backoff)."""
        s = self.s
        act = self.acts.start(POSTURE, {"want": "stand"}, parents=parents)
        record = self._act_record(act)
        assert act.result is not None
        if act.result.get("got") == "stand":
            return record, False
        s.stand_fail_times = [
            moment for moment in s.stand_fail_times if s.now - moment <= timedelta(minutes=10)
        ] + [s.now]
        backoff = len(s.stand_fail_times) >= 2
        s.stand_block_until = s.now + timedelta(minutes=30 if backoff else 2)
        return record, True

    def _act_record(self, act: Act) -> Event:
        """The input record of an act's result, for an act the world answered at once.

        Phase 4 (design A2): LegacyWorld answers a stand-up, a dock try and the nest look in
        the same call, and the caller goes on in place exactly as v4 did. A world that answers
        later needs Duck Core's waiting logic (phase 5c) first."""
        if not act.done or act.record_id is None:
            raise WorldError(
                f"{act.cmd} {act.act_id} was not answered at once: this Duck Core (phase 4) "
                "needs a world that answers stand-ups, dock tries and looks in the same call"
            )
        record = self.log.get(act.record_id)
        assert record is not None
        return record

    @staticmethod
    def _actual_speed_mm(vx_mm: int) -> int:
        """Expected ground speed for a forward command (measured; see ACTUAL_SPEED_MM)."""
        if vx_mm <= 0:
            return 0
        (x0, y0), (x1, y1) = ACTUAL_SPEED_MM
        return y0 + (vx_mm - x0) * (y1 - y0) // (x1 - x0)

    # ------------------------------------------------------------------ where it is on the floor

    def _pose(self) -> tuple[int, int, int]:
        return self.s.pos_x, self.s.pos_y, self.s.heading

    def _pose_list(self, pose: tuple[int, int, int] | None = None) -> list[float]:
        """[x m, y m, heading rad] for events and the page (presentation only)."""
        x, y, heading = pose if pose is not None else self._pose()
        return [x / 1000, y / 1000, heading / 1000]

    def pose_now(self) -> list[float]:
        """Where the duck is right now, mid-walk included."""
        s = self.s
        plan, started = s.walk_plan, s.walk_start
        if plan is None or started is None:
            return self._pose_list()
        elapsed = min(int(plan["seconds"]), int((s.now - started).total_seconds()))
        return self._pose_list(self._along(plan, elapsed * PATH_STEPS_PER_S)[0])

    def _place_on_mat(self) -> None:
        """Put down at its mat, facing +x (the family always sets it down there). With a
        nest the mat is the nest: set down "on the floor" it stands just in front of it."""
        s = self.s
        s.pos_x = s.pos_y = s.heading = 0
        if self.hw.nest and not s.nested:
            s.pos_x = int(content.NEST["front_mm"])
            s.pose_err_mm = max(s.pose_err_mm, 150)

    def _place_on_nest(self, *, walked_in: bool = False) -> None:
        """In the nest, at its centre. It sits the way it walked in (the ring contacts
        charge at any heading: it never turns round on the spot to face out); a hand puts
        it down facing the room (+x, the opening)."""
        self.s.pos_x = self.s.pos_y = 0
        if not walked_in:
            self.s.heading = 0

    def _nest_stood(self) -> None:
        """Standing still with a nest (a reflex stood it up, an errand was cut short): off
        on an outing if it may, otherwise it sits down (back onto the nest if it is there)."""
        s = self.s
        if excursion.can_start(self) is None and not self._until(s.napping_until):
            reason = "social" if self._present(3) else "restless"
            decision = self._emit(
                "duck.decision",
                {
                    "verb": "set_off",
                    "rule": "outing-v2",
                    "trigger": reason,
                    "from": "standing",
                    "table": self.table,
                    "inputs": self._decision_inputs(None),
                    "utilities": {"set_off": SCALE},
                    "probabilities": {"set_off": SCALE},
                },
            )
            if excursion.set_out(self, reason, parents=(decision.event_id,)):
                return
        self._decide_and_express("stand_limit")

    def active_spots(self) -> list[dict[str, Any]]:
        return [spot for _, spot in sorted(self.s.avoid_spots.items()) if spot.get("active", True)]

    def _blocked(self, start: tuple[int, int], point: tuple[int, int]) -> str | None:
        """Why a path may not pass ``point`` (None when it may).

        Leaving the room, or getting closer than AVOID_RADIUS_MM to a spot it learned to
        avoid. A path that starts too close may still move away.
        """
        return self._blocker()(start, point)

    def _blocker(self) -> Callable[[tuple[int, int], tuple[int, int]], str | None]:
        """``_blocked`` as it stands right now, worked out once for a whole path (a path is
        checked ten times a second of walking): the room's walls, the leash, the spots."""
        leash = self._path_leash
        leash2 = leash * leash if leash is not None else None
        homing = self.s.homing is not None
        corridor = int(content.NEST["corridor_mm"])
        spots = []
        for spot in self.active_spots():
            sx, sy, radius = int(spot["x"]), int(spot["y"]), int(spot["r"])
            if homing and sx * sx + sy * sy <= (corridor + radius) ** 2:
                continue  # the way into its own nest is never closed off
            spots.append((sx, sy, radius))
        limit = ROOM_LIMIT_MM
        # The family's zones (curiosity P2; none before it, so no call at all then).
        zones = self.s.zones
        zone_block = curiosity.zone_block

        def blocked(start: tuple[int, int], point: tuple[int, int]) -> str | None:
            (x0, y0), (x, y) = start, point
            if (abs(x) > limit and abs(x) > abs(x0)) or (abs(y) > limit and abs(y) > abs(y0)):
                return "wall"
            if leash2 is not None:
                d2 = x * x + y * y
                if d2 > leash2 and d2 > x0 * x0 + y0 * y0:
                    return "leash"
            for sx, sy, radius in spots:
                d2 = (x - sx) ** 2 + (y - sy) ** 2
                if d2 < radius * radius and d2 < (x0 - sx) ** 2 + (y0 - sy) ** 2:
                    return "spot"
            if zones:
                return zone_block(self, point)
            return None

        return blocked

    @property
    def _path_leash(self) -> int | None:
        """How far from the nest a path may lead (None: no nest). Exploring: its leash (see
        excursion.leash_mm). Going home: the family's radius itself (1.5 m while the family
        sleeps), so not even the arc round toward home carries it outside."""
        s = self.s
        if not self.hw.nest:
            return None
        if s.homing is not None:
            return self._home_radius()
        return excursion.leash_mm(self)

    def _home_radius(self) -> int | None:
        """The family's roam radius around the nest (1.5 m at most while the family is
        asleep): no path of any kind leads outside it. None without a nest.

        Read through here (and ``_turn_home_fits``): it is the duck's belief of where it is
        against the family's setting, which the camera-first design will supply."""
        s = self.s
        if not self.hw.nest:
            return None
        radius = int(s.roam_radius_mm)
        if s.phase == "solo":
            radius = min(radius, int(content.RHYTHM["solo_radius_mm"]))
        return radius

    def _turn_home_fits(self, x: int, y: int, ux: int, uy: int) -> bool:
        """Whether at (x, y) mm, facing the unit vector (ux, uy) × SCALE, it could turn
        round toward the nest on its tight arc without leaving the home radius (the turning
        circle on the nest's side stays inside it). Facing roughly home already, it needs
        no turn at all."""
        radius = self._home_radius()
        if radius is None:
            return True
        dist = isqrt(x * x + y * y)
        if dist == 0 or -(x * ux + y * uy) // dist >= FACING_HOME_COS:
            return True
        # Home to its left (counter-clockwise): the circle's centre is on its left side.
        side = 1 if uy * x - ux * y >= 0 else -1
        cx = x - side * uy * ARC_RADIUS_MM // SCALE
        cy = y + side * ux * ARC_RADIUS_MM // SCALE
        return isqrt(cx * cx + cy * cy) + ARC_RADIUS_MM <= radius

    def _pose_turns_home(self, pose: tuple[int, int, int]) -> bool:
        x, y, heading = pose
        return self._turn_home_fits(x, y, cos_mrad(heading), sin_mrad(heading))

    def _along(
        self, plan: dict[str, Any], steps: int, *, check: bool = False
    ) -> tuple[tuple[int, int, int], str | None]:
        """Integrate a walk's path; with ``check``, stop at the first reason it is blocked."""
        return pose_along(plan, steps, self._blocker() if check else None)

    def _turn_away(self, longest: int, shortest: int = MIN_WALK_S) -> tuple[int, int, int]:
        """The turn (vx mm, vyaw mrad, seconds) that points it back into open floor.

        Deterministic: the turn after which a straight PROBE_MM ahead is clear, else one
        that at least leaves room to go on (not nose to the wall); a tight arc (ARC_VX_MM)
        whose own path is clear before turning on the spot; facing the room's middle as
        much as possible; the smaller turn wins a tie. A turn of its own lasts at least
        ``shortest`` (MIN_WANDER_S; the way home may take MIN_WALK_S).
        """
        x, y, heading = self._pose()
        yaw = TURN_VYAW_MRAD * ACTUAL_YAW_RATIO // SCALE
        arc_speed = self._actual_speed_mm(ARC_VX_MM)
        shortest = min(shortest, longest)
        blocker = self._blocker()
        # (clear ahead, room to go on, walks, inward, -seconds, side)
        best: tuple[int, int, int, int, int, int] | None = None
        for walks in (1, 0):
            for side in (1, -1):
                arc = {"from": (x, y, heading), "speed_mm": arc_speed, "yaw_mrad": side * yaw}
                poses = poses_by_second(arc, longest, blocker)[0] if walks else []
                for seconds in range(shortest, longest + 1):
                    if walks:
                        if seconds > len(poses):
                            break  # a longer arc runs into the same thing
                        ex, ey, facing = poses[seconds - 1]
                    else:
                        ex, ey, facing = x, y, wrap_mrad(heading + side * yaw * seconds)
                    probe = {"from": (ex, ey, facing), "speed_mm": PROBE_MM, "yaw_mrad": 0}
                    clear = self._along(probe, PATH_STEPS_PER_S, check=True)[1] is None
                    room = clear or self._room_to_turn((ex, ey, facing))
                    inward = -(ex * cos_mrad(facing) + ey * sin_mrad(facing)) // SCALE
                    key = (int(clear), int(room), walks, inward, -seconds, side)
                    if best is None or key > best:
                        best = key
        assert best is not None
        return (ARC_VX_MM if best[2] else 0), best[5] * TURN_VYAW_MRAD, -best[4]

    def _room_to_turn(self, pose: tuple[int, int, int]) -> bool:
        """Whether a walk ending at ``pose`` can go on from there: a clear way straight
        ahead or a clear tight arc to either side (else only turning on the spot is left),
        and, with a nest, room to turn round toward home inside the family's radius."""
        x, y, heading = pose
        if self.hw.nest and self.s.homing is None and not self._pose_turns_home(pose):
            return False
        ahead = {"from": pose, "speed_mm": TURN_ROOM_MM, "yaw_mrad": 0}
        if self._along(ahead, PATH_STEPS_PER_S, check=True)[1] is None:
            return True
        speed = self._actual_speed_mm(ARC_VX_MM)
        yaw = TURN_VYAW_MRAD * ACTUAL_YAW_RATIO // SCALE
        for side in (1, -1):
            arc = {"from": (x, y, heading), "speed_mm": speed, "yaw_mrad": side * yaw}
            if self._along(arc, MIN_WANDER_S * PATH_STEPS_PER_S, check=True)[1] is None:
                return True
        return False

    def _arc_turn(
        self,
        err: int,
        top: int,
        target: tuple[int, int] | None = None,
        *,
        shortest: int = MIN_WALK_S,
        side: int | None = None,
    ) -> tuple[int, int] | None:
        """The tight arc (vyaw mrad, seconds) that leaves it facing its goal best: toward
        ``target`` (mm) when given, else ``err`` (mrad) round from where it faces now. The
        turning side first, then the long way round (``side``: this side whenever it has a
        clear arc at all, e.g. the right arc out of the nest); only a clear path that ends
        facing closer than now (None when there is none), at least ``shortest`` seconds."""
        x, y, heading = self._pose()
        goal = wrap_mrad(heading + err)
        speed = self._actual_speed_mm(ARC_VX_MM)
        yaw = TURN_VYAW_MRAD * ACTUAL_YAW_RATIO // SCALE
        first = side if side is not None else (1 if err >= 0 else -1)
        shortest = min(shortest, top)
        best: tuple[tuple[int, int, int], int, int] | None = None  # (key, side, seconds)
        blocker = self._blocker()
        for rank, turn in enumerate((first, -first)):
            found: tuple[tuple[int, int, int], int, int] | None = None
            arc = {"from": (x, y, heading), "speed_mm": speed, "yaw_mrad": turn * yaw}
            poses = poses_by_second(arc, top, blocker)[0]
            for seconds in range(shortest, len(poses) + 1):
                ex, ey, eh = poses[seconds - 1]
                aim = goal
                if target is not None and (target[0] != ex or target[1] != ey):
                    aim = atan2_mrad(target[1] - ey, target[0] - ex)
                key = (abs(signed_mrad(aim - eh)), rank, seconds)
                if found is None or key < found[0]:
                    found = (key, turn, seconds)
            if found is None or found[0][0] >= abs(err):
                continue  # this side ends no closer to facing its goal
            if side is not None:
                return found[1] * TURN_VYAW_MRAD, found[2]  # the preferred side works
            if best is None or found[0] < best[0]:
                best = found
        if best is None:
            return None
        return best[1] * TURN_VYAW_MRAD, best[2]

    def _plan_walk(self, verb_id: str, *, short: bool = False) -> dict[str, Any]:
        """A short walk (wander) or a turn (a tight arc), within the validated bounds.

        The walking policy does not move at all below ~0.3 m/s, so a gentler walk is never a
        slower command: younger ducks and conservative mode walk *shorter* (and, through
        their utilities, rarer), never shorter than MIN_WANDER_S. A walk whose path would
        leave the room or pass a spot it learned to avoid tries its other gentle curves,
        then turns away instead.
        """
        s = self.s
        stage_id = self.stage().stage_id
        scale = content.MOTION_STAGE[stage_id]["scale"] * (5000 if s.conservative else SCALE)
        longest = max(MIN_WANDER_S, WALK_S_MAX * scale // SCALE**2)
        if short:
            longest = max(MIN_WANDER_S, longest // 2)
        shortest = max(MIN_WANDER_S, longest // 3)
        span = longest - shortest + 1
        seconds = shortest + rng.pick_index(self._rand("walk.seconds"), span)
        if stage_id in ("hatchling", "duckling"):  # short toddles
            seconds = min(seconds, shortest + rng.pick_index(self._rand("walk.seconds"), span))
        elif stage_id == "adult":  # confident, longer walks
            seconds = max(seconds, shortest + rng.pick_index(self._rand("walk.seconds"), span))
        low, high = WALK_VX_MM
        avoided = None
        if verb_id == "wander":
            vx = low + rng.pick_index(self._rand("walk.vx"), high - low + 1)
            curve = min(WANDER_CURVE_MRAD, TURN_VYAW_MRAD)
            vyaw = rng.pick_index(self._rand("walk.vyaw"), 2 * curve + 1) - curve
            speed = self._actual_speed_mm(vx)
            tries = [vyaw] + [
                option for option in (curve, -curve, curve // 2, -curve // 2, 0) if option != vyaw
            ]
            chosen = None
            for option in tries:
                probe = {
                    "from": self._pose(),
                    "speed_mm": speed,
                    "yaw_mrad": option * ACTUAL_YAW_RATIO // SCALE,
                }
                end, why = self._along(probe, seconds * PATH_STEPS_PER_S, check=True)
                if why is None and not self._room_to_turn(end):
                    # It would stop nose to the wall, with no room left but to turn on the
                    # spot: another curve, or turn away now while there is still room.
                    why = "wall"
                if why is None:
                    chosen = option
                    break
                avoided = avoided or why
            if chosen is not None:
                vyaw = chosen
            else:
                verb_id = "turn_around"  # nowhere to go that way: turn away instead
                vx, vyaw, seconds = self._turn_away(longest, MIN_WANDER_S)
        else:
            # A turn walks the tight arc at the full turn command; it usually turns toward
            # its own side, the other way round if that arc is blocked, and on the spot only
            # when both are.
            side = 1 if s.quirks.get("tilt_side") == "左" else -1
            if rng.below(self._rand("turn.side"), 3000):
                side = -side
            vx, vyaw = 0, side * TURN_VYAW_MRAD
            arc_speed = self._actual_speed_mm(ARC_VX_MM)
            yaw = TURN_VYAW_MRAD * ACTUAL_YAW_RATIO // SCALE
            cramped = None
            for option in (side, -side):
                probe = {"from": self._pose(), "speed_mm": arc_speed, "yaw_mrad": option * yaw}
                end, why = self._along(probe, seconds * PATH_STEPS_PER_S, check=True)
                if why is not None:
                    avoided = avoided or why
                elif self._room_to_turn(end):
                    vx, vyaw, cramped = ARC_VX_MM, option * TURN_VYAW_MRAD, None
                    break
                elif cramped is None:
                    cramped = option  # it would end nose to the wall: the other way first
            if cramped is not None:
                vx, vyaw = ARC_VX_MM, cramped * TURN_VYAW_MRAD
        plan: dict[str, Any] = {
            "verb": verb_id,
            "vx": vx,
            "vyaw": vyaw,
            "seconds": seconds,
            "speed_mm": self._actual_speed_mm(vx),
            "yaw_mrad": vyaw * ACTUAL_YAW_RATIO // SCALE,
            "from": self._pose(),
            "avoided": avoided,
        }
        plan["to"] = self._along(plan, seconds * PATH_STEPS_PER_S)[0]
        return plan

    def _leg_longest(self) -> int:
        stage_id = self.stage().stage_id
        scale = content.MOTION_STAGE[stage_id]["scale"] * (5000 if self.s.conservative else SCALE)
        return int(max(MIN_WANDER_S, WALK_S_MAX * scale // SCALE**2))

    def _plan_turn(
        self,
        err: int,
        *,
        purpose: str,
        longest: int | None = None,
        target: tuple[int, int] | None = None,
        side: int | None = None,
    ) -> dict[str, Any]:
        """Turn toward ``err`` (mrad) — or to face ``target`` — at the full turn command:
        walking a tight arc (the same validated command at any age or mode: whole turns),
        in place only when every arc is blocked. ``side``: the side it turns to whenever
        that side has a clear arc (the nest is left by a right arc)."""
        top = longest if longest is not None else max(MIN_WALK_S, WALK_S_MAX)
        shortest = MIN_WALK_S if purpose in HOME_PURPOSES else MIN_WANDER_S
        arc = self._arc_turn(err, top, target, shortest=shortest, side=side)
        if arc is not None:
            return self._make_plan("turn_around", ARC_VX_MM, arc[0], arc[1], purpose)
        actual = TURN_VYAW_MRAD * ACTUAL_YAW_RATIO // SCALE
        seconds = clamp(-(-abs(err) // actual), min(shortest, top), top)
        vyaw = TURN_VYAW_MRAD if err > 0 else -TURN_VYAW_MRAD
        return self._make_plan("turn_around", 0, vyaw, seconds, purpose)

    def _make_plan(
        self,
        verb_id: str,
        vx: int,
        vyaw: int,
        seconds: int,
        purpose: str,
        avoided: str | None = None,
    ) -> dict[str, Any]:
        plan: dict[str, Any] = {
            "verb": verb_id,
            "vx": vx,
            "vyaw": vyaw,
            "seconds": seconds,
            "speed_mm": self._actual_speed_mm(vx),
            "yaw_mrad": vyaw * ACTUAL_YAW_RATIO // SCALE,
            "from": self._pose(),
            "avoided": avoided,
            "purpose": purpose,
        }
        plan["to"] = self._along(plan, seconds * PATH_STEPS_PER_S)[0]
        return plan

    def _plan_leg_toward(
        self,
        target: tuple[int, int],
        purpose: str,
        *,
        vx: int | None = None,
        longest: int | None = None,
        side: int | None = None,
    ) -> dict[str, Any]:
        """One leg toward ``target`` (mm): turn on a tight arc when it faces well away, else
        walk forward curving gently so it arrives facing the target. Only forward walking at
        the validated commands, ≤ 6 s a leg (a leg of its own ≥ MIN_WANDER_S; the way home
        may end with a MIN_WALK_S one). A path that would leave the room, cross a spot it
        learned to avoid or (on an outing) go past the leash tries its other gentle curves,
        then turns away instead. An outing's leg that would end facing out near the edge of
        the family's radius stops earlier, where it can still turn round toward home."""
        x, y, heading = self._pose()
        dx, dy = int(target[0]) - x, int(target[1]) - y
        dist = isqrt(dx * dx + dy * dy)
        err = signed_mrad(atan2_mrad(dy, dx) - heading) if dist else 0
        top = longest if longest is not None else self._leg_longest()
        homing = purpose in HOME_PURPOSES
        shortest = MIN_WALK_S if homing else MIN_WANDER_S
        if abs(err) > TURN_IN_PLACE_MRAD:
            return self._plan_turn(
                err, purpose=purpose, target=(int(target[0]), int(target[1])), side=side
            )
        vx = vx if vx is not None else WALK_VX_MM[0]
        speed = self._actual_speed_mm(vx)
        seconds = clamp(-(-dist // max(1, speed)), shortest, max(top, shortest))
        curve = min(WANDER_CURVE_MRAD, TURN_VYAW_MRAD)
        vyaw = clamp(err * SCALE // (ACTUAL_YAW_RATIO * seconds), -curve, curve)
        avoided = None
        edge = self.hw.nest and not homing
        for option in [vyaw] + [
            o for o in (curve, -curve, curve // 2, -curve // 2, 0) if o != vyaw
        ]:
            probe = {
                "from": (x, y, heading),
                "speed_mm": speed,
                "yaw_mrad": option * ACTUAL_YAW_RATIO // SCALE,
            }
            end, why = self._along(probe, seconds * PATH_STEPS_PER_S, check=True)
            if why is None and edge and not self._pose_turns_home(end):
                # It would end facing out near the edge: stop where it can still turn home.
                for shorter in range(seconds - 1, shortest - 1, -1):
                    if self._pose_turns_home(self._along(probe, shorter * PATH_STEPS_PER_S)[0]):
                        return self._make_plan("wander", vx, option, shorter, purpose, avoided)
                why = "leash"
            if why is None:
                return self._make_plan("wander", vx, option, seconds, purpose, avoided)
            avoided = avoided or why
        vx, vyaw, seconds = self._turn_away(max(top, shortest), shortest)
        return self._make_plan("turn_around", vx, vyaw, seconds, purpose, avoided)

    def _walk_leg(
        self, plan: dict[str, Any], *, reason: str, outing_id: str | None = None
    ) -> Event:
        """Walk a planned leg of an errand or outing: only ``duck.walk`` (the outing's first
        decision, and the looks between legs, are its other events)."""
        s = self.s
        parent = self.log.events[-1].event_id if self.log.events else None
        walk = self._walk(plan, None, reason=reason, outing_id=outing_id, parent=parent)
        # Something it chose to do by itself: praise can attach to the walk.
        s.last_flag = {"kind": "verb", "id": str(plan["verb"]), "at": s.now, "event": walk.event_id}
        return walk

    def _walk(
        self,
        plan: dict[str, Any],
        expression: Event | None,
        *,
        reason: str = "",
        outing_id: str | None = None,
        parent: str | None = None,
    ) -> Event:
        """Start a planned walk: the ``duck.walk`` command (an act, design A7). How it ends
        (on time, or in a fall) is the world's: its ``walk.done`` or ``body.fell`` arrives as
        time passes."""
        s = self.s
        verb_id, seconds = str(plan["verb"]), int(plan["seconds"])
        vx, vyaw = int(plan["vx"]), int(plan["vyaw"])
        speed, yaw_rate = int(plan["speed_mm"]), int(plan["yaw_mrad"])
        cm = speed * seconds // 10
        start, end = plan["from"], plan["to"]
        payload: dict[str, Any] = {
            "verb": verb_id,
            "vx": vx / 1000,
            "vyaw": vyaw / 1000,
            "seconds": seconds,
            "meters": cm / 100,
            "cm": cm,
            "speed": speed / 1000,
            "yaw_rate": yaw_rate / 1000,
            # Where it believes it starts and expects to end (predictions, not odometry).
            "bel_from": {"x_mm": int(start[0]), "y_mm": int(start[1]), "yaw_mrad": int(start[2])},
            "expect": {"x_mm": int(end[0]), "y_mm": int(end[1]), "yaw_mrad": int(end[2])},
            "stage": self.stage().stage_id,
            "conservative": s.conservative,
            "battery": s.battery,
            "floor": s.floor,
            # The body.sock command (duckmind.duckbody.protocol.WalkCmd).
            "vx_mm": vx,
            "vyaw_mrad": vyaw,
            "max_ms": seconds * 1000,
        }
        if plan["avoided"] is not None:
            payload["avoided"] = plan["avoided"]  # "wall" or "spot": it steered or turned away
        if plan.get("moment"):
            payload["moment"] = plan["moment"]  # one of its moments (a tail chase, a wrong turn)
        if self.hw.nest:
            # Enough to check every safety rule from the log alone.
            fx, fy = int(plan["from"][0]), int(plan["from"][1])
            payload.update(
                {
                    "purpose": plan.get("purpose") or reason or "idle",
                    "excursion": outing_id,
                    "family": s.family,
                    "phase": s.phase,
                    "nested": s.nested,
                    "docked": s.docked,
                    "unplaced": s.unplaced,
                    "dist_from_nest_mm": isqrt(fx * fx + fy * fy),
                    "dist_to_mm": isqrt(int(plan["to"][0]) ** 2 + int(plan["to"][1]) ** 2),
                    "leash_mm": excursion_leash if (excursion_leash := self._path_leash) else None,
                    "homing": s.homing is not None,
                }
            )
        parents = (
            (expression.event_id,) if expression is not None else ((parent,) if parent else ())
        )
        act = self.acts.start(WALK, payload, parents=parents)
        walk = self.log.get(act.event_id)
        assert walk is not None
        s.posture = "walk"
        s.walk_start = s.now
        s.walk_until = s.now + timedelta(seconds=seconds)  # when it expects the walk to end
        s.walk_plan = {**plan, "event": walk.event_id, "cm": cm, "act_id": act.act_id}
        s.walked_cm += cm
        s.hand_placed = False
        errand = s.excursion is not None or s.homing is not None
        if verb_id == "wander":
            s.walks += 1
            s.day_walk_s += seconds
            if not errand:
                s.walk_sat = min(SCALE, s.walk_sat + WALK_SAT_WANDER)
            # Odometry drifts: about 3% of the way walked (the camera and the nest reset it;
            # the A100 gate asks for negligible drift).
            s.pose_err_mm += speed * seconds * 3 // 1000 if self.hw.nest else 0
        else:
            s.turns += 1
            s.day_turn_s += seconds
            if not errand:
                s.walk_sat = min(SCALE, s.walk_sat + WALK_SAT_TURN)
            if self.hw.nest and plan.get("purpose") != "circle":
                s.pose_err_mm += 10  # turning blurs the heading a little
                # …and an arc also drifts like any walk (about 3% of the way)
                s.pose_err_mm += speed * seconds * 3 // 1000
        s.energy = max(0, s.energy - WALK_ENERGY_PER_S * seconds)
        if not s.docked:
            used = BATTERY_DRAIN_PER_MIN * seconds // 60
            s.battery = max(0, s.battery - used)
            s.day_drain += used
        if s.outing_legs > 0:
            low, high = LEG_PAUSE_S
            s.leg_pause_s = low + rng.pick_index(self._rand("outing.pause"), high - low + 1)
        if s.excursion is not None and outing_id is not None:
            s.excursion["walk_s"] = int(s.excursion["walk_s"]) + seconds
            s.excursion["cm"] = int(s.excursion["cm"]) + cm
            tx, ty = int(plan["to"][0]), int(plan["to"][1])
            s.excursion["max_mm"] = max(int(s.excursion["max_mm"]), isqrt(tx * tx + ty * ty))
        if s.homing is not None:
            s.homing["cm"] = int(s.homing.get("cm", 0)) + cm
        if verb_id == "wander" and "first_steps" not in s.milestones:
            self._experience_milestone("first_steps")
        if plan["avoided"] == "spot":
            moments.detour(self, walk)  # steering round the spot where it fell
        return walk

    def _cut_walk(self, why: str | None = None) -> dict[str, Any] | None:
        """Stop the walk in progress; only the distance actually walked counts. With ``why``
        Duck Core also tells the body to stop (``duck.stop``; its walk.done answers the act):
        every cut but a fall, which the body already reported."""
        s = self.s
        plan, started = s.walk_plan, s.walk_start
        s.walk_plan = s.walk_start = s.walk_until = None
        if plan is None or started is None:
            return None
        act_id = plan.get("act_id")
        if why is not None and act_id in self.acts.open:
            self.acts.stop(str(act_id), why, parents=(str(plan["event"]),))
        elapsed = min(int(plan["seconds"]), int((s.now - started).total_seconds()))
        walked = int(plan["cm"]) * elapsed // int(plan["seconds"])
        short = int(plan["cm"]) - walked
        s.walked_cm -= short
        s.pos_x, s.pos_y, s.heading = self._along(plan, elapsed * PATH_STEPS_PER_S)[0]
        return {
            "walk": plan["event"],
            "after_s": elapsed,
            "cm_short": short,
            "pose": self._pose_list(),
        }

    def _finish_walk(self) -> None:
        s = self.s
        plan = s.walk_plan
        s.posture = "stand"
        s.stand_idle_from = s.walk_until
        s.walk_plan = s.walk_start = s.walk_until = None
        if plan is not None:
            s.pos_x, s.pos_y, s.heading = plan["to"]
            self._walk_keepsakes(str(plan["event"]))
            moments.leg_done(self, str(plan["event"]), False)

    # ------------------------------------------------------------------ falls

    def _fall(self, fell: Event) -> None:
        """The body fell (``fell`` is the input record of its body.fell): a walk ends there,
        it lies on its side, and the body soon tries to roll back up by itself."""
        s = self.s
        plan = s.walk_plan
        before = s.posture
        cut = self._cut_walk()  # (no stop: the body's walk.done {fell} closes the act)
        s.posture = "fallen"
        s.outing_legs = 0
        if s.excursion is not None:
            excursion.end(self, "fall", decide=False)
        s.stood_at = s.stand_idle_from = None
        s.falls += 1
        fallen = self._emit(
            "duck.posture",
            {
                "posture": "fallen",
                "from": before,
                "reason": "fell",
                "by": "self",
                "during": plan["verb"] if plan else None,
                "after_s": cut["after_s"] if cut else 0,
                "falls": s.falls,
                "walk_cut": cut,
                "at": self._pose_list(),
            },
            parents=(fell.event_id,),
        )
        s.righting = {"fell": fell.event_id, "tries": 0, "x": s.pos_x, "y": s.pos_y}
        self._walk_keepsakes(fallen.event_id)
        if plan is not None:
            moments.leg_done(self, str(plan["event"]), True)

    def _righting_try(self, attempt: Event, ok: bool, told: int | None = None) -> None:
        """One try of the body's to roll back up after a fall (``attempt`` is the input
        record of its body.righting): up (easy on the first try, hard after more), or, after
        the last try, stuck until someone helps. ``told``: the percept's own count of tries,
        when the body counted more than Duck Core has seen (phase 5's duck-body reports only
        the last try; LegacyWorld reports each one, so the counts agree)."""
        s = self.s
        state = s.righting
        assert state is not None
        state["tries"] = max(int(state["tries"]) + 1, told or 0)
        tries = int(state["tries"])
        outcome = None
        if ok:
            outcome = "easy" if tries == 1 else "hard"
        elif tries >= RIGHTING_TRIES:
            outcome = "stuck"  # it lies there until someone picks it up
        if outcome is None:
            return  # (the body tries again by itself)
        s.righting = None
        if ok:
            # Rolled back onto its belly: the seated rest pose, then life goes on.
            s.posture = "sit"
            self._emit(
                "duck.posture",
                {"posture": "sit", "from": "fallen", "reason": "righted", "by": "self"},
                parents=(attempt.event_id,),
            )
        self._fall_outcome(state, outcome, attempt)
        if ok:
            moments.righted(self, attempt.event_id)
        if outcome == "stuck":
            # It cannot get up by itself: it asks for a hand (a need, never an attention
            # bid), and a duck with a nest tells the family's phone too.
            if self.hw.nest:
                nest.call_for_help(self, "help", (attempt.event_id,))
            else:
                s.need = {"kind": "help", "since": s.now}
                self._emit(
                    "duck.bid",
                    {
                        "kind": "help",
                        "need": True,
                        "sounds": ["inquire"],
                        "text": content.phrase_text(("inquire",)),
                    },
                    parents=(attempt.event_id,),
                )

    def _righting_cut(self, reason: str) -> None:
        """Someone set it upright while it was still trying (a hand, or powering off): the
        body is told to stop trying, and Duck Core notes how the fall ended."""
        s = self.s
        state = s.righting
        if state is None:
            return
        s.righting = None
        self.acts.stop(None, reason, parents=(str(state["fell"]),))
        tries = int(state["tries"])
        outcome = "hard" if tries > 0 else "helped"
        attempt = self._emit(
            "duck.righting_end",
            {
                "tries": tries,
                "of": RIGHTING_TRIES,
                "helped": reason,
                "outcome": outcome,
                "at": self._pose_list(),
            },
            parents=(str(state["fell"]),),
        )
        self._fall_outcome(state, outcome, attempt)

    def _fall_outcome(self, state: dict[str, Any], outcome: str, evidence: Event) -> None:
        """Remember where it fell and how it got up; hard ones may teach it about the spot."""
        s = self.s
        spot = {
            "fell": str(state["fell"]),
            "outcome": evidence.event_id,
            "x": int(state["x"]),
            "y": int(state["y"]),
            "hard": outcome in ("hard", "stuck"),
            "zone": None,
        }
        s.fall_spots.append(spot)
        curiosity.on_fall(self, spot)
        if spot["hard"]:
            self._learn_spot(spot)

    @staticmethod
    def _near(a: dict[str, Any], b: dict[str, Any], radius: int) -> bool:
        return (int(a["x"]) - int(b["x"])) ** 2 + (int(a["y"]) - int(b["y"])) ** 2 <= radius**2

    def _learn_spot(self, latest: dict[str, Any]) -> None:
        """Three or more hard falls near one spot: that spot is not for walking.

        One fall, or falls it got up from easily, teach nothing. The spot is a learned,
        forgettable memory (a ``learning.committed`` whose evidence is the falls).
        """
        s = self.s
        for zone_id, zone in sorted(s.avoid_spots.items()):
            if zone.get("active", True) and self._near(latest, zone, int(zone["r"])):
                latest["zone"] = zone_id  # a known spot: nothing new to learn
                return
        near = [
            spot
            for spot in s.fall_spots
            if spot["hard"] and spot["zone"] is None and self._near(spot, latest, AVOID_RADIUS_MM)
        ]
        if len(near) < HARD_FALLS_TO_LEARN:
            return
        corridor = int(content.NEST["corridor_mm"])
        cx = sum(int(spot["x"]) for spot in near) // len(near)
        cy = sum(int(spot["y"]) for spot in near) // len(near)
        if self.hw.nest and (
            self._near(latest, {"x": 0, "y": 0}, corridor)
            or self._near({"x": cx, "y": cy}, {"x": 0, "y": 0}, corridor)
        ):
            # Near its own nest it never learns to be afraid (it would fear its home): the
            # family is asked to check the floor instead, and until they say so it goes in
            # and out only while someone is there.
            evidence = [str(spot[key]) for spot in near for key in ("fell", "outcome")]
            for spot in near:
                spot["zone"] = "nest"
            s.nest_trouble = True
            s.nest_trouble_falls += len(near)
            trouble = self._emit(
                "nest.trouble",
                {"falls": len(near), "evidence": evidence, "corridor_mm": corridor},
                parents=(str(latest["outcome"]),),
            )
            self._emit(
                "owner.notice",
                {"kind": "nest_trouble", "text": "窩附近跌倒好幾次，請檢查地面或換個位置"},
                parents=(trouble.event_id,),
            )
            return
        zone_id = f"S{len(s.avoid_spots) + 1}"
        cx = sum(int(spot["x"]) for spot in near) // len(near)
        cy = sum(int(spot["y"]) for spot in near) // len(near)
        evidence = [str(spot[key]) for spot in near for key in ("fell", "outcome")]
        update = LearningUpdate(
            update_id=f"U-{latest['fell']}-avoid",
            episode_id=s.trace,
            parameter=f"self.avoid_spot:{zone_id}",
            before=0.0,
            after=1.0,
            max_abs_delta=1.0,
            rule_version="avoid-spot-v1",
            outcome_ids=tuple(evidence),
            status=LearningStatus.COMMITTED,
        )
        learned = self._emit(
            "learning.committed",
            {
                "update_id": update.update_id,
                "parameter": update.parameter,
                "before": update.before,
                "after": update.after,
                "max_abs_delta": update.max_abs_delta,
                "rule_version": update.rule_version,
                "evidence_event_ids": evidence,
                "spot": {"x": cx / 1000, "y": cy / 1000, "r": AVOID_RADIUS_MM / 1000},
                "falls": len(near),
            },
            parents=(str(latest["outcome"]),),
        )
        s.avoid_spots[zone_id] = {
            "x": cx,
            "y": cy,
            "r": AVOID_RADIUS_MM,
            "active": True,
            "event": learned.event_id,
        }
        for spot in near:
            spot["zone"] = zone_id

    def _set_posture(
        self, posture: str, *, reason: str, by: str, parents: tuple[str, ...] = ()
    ) -> Event | None:
        """A posture change the duck did not choose (a hand, the gamepad, powering off).
        Duck Core's own record of it (signed self), after the input that caused it."""
        s = self.s
        before = s.posture
        if before == posture or s.bridged:
            return None
        if before == "fallen":
            self._righting_cut(reason)  # set upright while it was still trying by itself
        cut = self._cut_walk(why=reason) if before == "walk" else None
        s.posture = posture
        s.outing_legs = 0  # whatever set the posture ends the outing
        if s.excursion is not None:
            excursion.end(self, reason, decide=False)
        if posture == "stand":
            s.stand_idle_from = s.now
            s.restless_min = 0
            if before in ("sit", "fallen"):
                s.stood_at = s.now
        else:
            s.stand_idle_from = None
            s.stood_at = None
        payload: dict[str, Any] = {"posture": posture, "from": before, "reason": reason, "by": by}
        if cut is not None:
            payload["walk_cut"] = cut
        event = self._emit("duck.posture", payload, parents=parents)
        if cut is not None:
            self._walk_keepsakes(event.event_id)
        return event

    def _set_floor(self, on: bool, why: str, *, parents: tuple[str, ...] = ()) -> Event:
        # Put on the floor, picked up or back on the desk: it is (again) at its mat.
        s = self.s
        if self.hw.nest and why == "family":
            if s.nested and not on:
                nest.undock(self, "family", parents)
            if on and not s.nested:
                s.unplaced = False
                s.stranded = None
                s.hand_placed = True
            elif not on:
                s.unplaced = False
        self.s.floor = on
        if not (self.hw.nest and s.nested and on):
            self._place_on_mat()
        payload: dict[str, Any] = {"floor": on, "why": why, "pose": self._pose_list()}
        if on:
            # Whether it can walk about from here, or what still stops it (for the words).
            blocks = self.walk_blocks()
            payload["walk"] = blocks[0] if blocks else None
        if why == "family":
            # The family's own word on where it is (the page's floor switch): this record is
            # that input, signed by the simulator's operator as in v4.
            return self._emit("env.floor", payload, source="sim", parents=parents)
        return self._emit("env.floor", payload, parents=parents)  # its belief: signed self

    def _walk_keepsakes(self, evidence_id: str) -> None:
        s = self.s
        for mark in content.WALK_KEEPSAKES_M:
            if s.walked_cm >= mark * 100 and mark not in s.walk_keepsakes:
                s.walk_keepsakes.append(mark)
                self._emit(
                    "keepsake",
                    {
                        "meters": mark,
                        "kind": "self_walk",
                        "text": f"紀念：我自己走了 {mark} 公尺！",
                    },
                    parents=(evidence_id,),
                )

    def _reflex(
        self, verb_id: str, trigger: str, chance: int, parents: tuple[str, ...]
    ) -> Event | None:
        """A quick whole-body reaction, drawn and logged like any decision.

        The same hard gates as autonomous motion apply; ``chance`` already carries
        personality and growth stage.
        """
        s = self.s
        if s.bridged:
            return None  # (no body of its own to stand up or sit down)
        if verb_id not in s.unlocked or self._motion_block(verb_id) is not None:
            return None
        chance = clamp(chance)
        raw = self._rand("reflex")
        go = rng.below(raw, chance)
        decision = self._emit(
            "duck.decision",
            {
                "verb": verb_id if go else "stay",
                "rule": "reflex-v1",
                "trigger": trigger,
                "table": self.table,
                "counter": self.counters["reflex"],
                "draw": rng.unit(raw),
                "inputs": self._decision_inputs(self._leaning()),
                "utilities": {verb_id: chance, "stay": SCALE - chance},
                "probabilities": {verb_id: chance, "stay": SCALE - chance},
            },
            parents=parents,
        )
        if not go:
            return None
        return self._perform(verb_id, reason=trigger, parents=(decision.event_id,), force_sound=())

    def _stand_reflex_chance(self, base: int, key: str = "reflex") -> int:
        chance = base * content.MOTION_STAGE[self.stage().stage_id]["reflex"] // SCALE
        return chance * self._style(key, self._leaning()) // SCALE

    def _sit_down_now(self, trigger: str) -> None:
        """Sit down because it has to (tired, bedtime, a nap): still a logged decision."""
        if self.s.posture == "walk":
            self._set_posture("stand", reason=trigger, by="self")
        if self.s.posture == "stand":
            self._decide_and_express(trigger)

    def _amplitude_cap(self, clip: str) -> int:
        """How big a head move may be right now (fixed point of the seated envelope)."""
        s = self.s
        # Conservative mode held every move to 60% before any sweep evidence; a move the
        # physics sweep has cleared may go to 85% (unswept moves keep the 60% default).
        conservative_cap = CONSERVATIVE_SWEPT if self.caps is not None else 6000
        cap = min(self.stage().amplitude_cap, conservative_cap if s.conservative else SCALE)
        if self.caps is not None:
            # Swept limits per posture when known (a head move that is safe seated can
            # load the neck too much while balancing), else the stricter of the two.
            posture = "stand" if s.posture in ("stand", "walk") else "sit"
            cap = min(cap, self.caps.get(f"{clip}@{posture}", self.caps.get(clip, 6000)))
        return cap

    def _tempo(self) -> int:
        s = self.s
        tempo = int(s.quirks.get("tempo", "10000")) * (8000 + s.arousal * 4 // 10) // SCALE
        return tempo * 8000 // SCALE if s.conservative else tempo

    def _express(
        self,
        verb_id: str,
        *,
        reason: str,
        parents: tuple[str, ...] = (),
        force_sound: tuple[str, ...] | None = None,
        tempo: int | None = None,
    ) -> Event:
        s = self.s
        if s.bridged:
            # (the neural duck moves and sounds for itself: what it did arrives as an experience)
            return self.log.events[-1]
        verb = content.VERB_BY_ID[verb_id]
        amplitude = self._amplitude_cap(verb_id) * (8000 + max(0, s.valence) // 5) // SCALE
        moment = reason == "moment"  # a beat of one of its moments (see moments.beats)
        tempo = self._tempo() if tempo is None else tempo
        quiet = self._until(s.quiet_until) or s.hot
        if force_sound is not None:
            sounds = list(force_sound)
        else:
            options = [tag for tag in verb.sounds if tag]
            sounds = []
            if options and not quiet and len(s.self_phrase_times) < SELF_PHRASES_PER_HOUR:
                pick = options[rng.pick_index(self._rand("sound.pick"), len(options))]
                if s.quirks.get("favorite_tag") in options and rng.below(
                    self._rand("sound.favorite"), 3000
                ):
                    pick = s.quirks["favorite_tag"]
                if "" in verb.sounds and rng.below(self._rand("sound.silent"), 5000):
                    pick = ""
                if pick:
                    sounds = [pick]
                    if s.quirks.get("habit") == "extra_chirp" and verb.fit == "happy":
                        sounds.append("chirp")
                        self._reveal_quirk("habit", s.quirks["habit_text"])
        if quiet:
            sounds = []
        if s.hush:
            sounds = [tag for tag in sounds if tag in ("chirp", "coo")][:1]
        if sounds and force_sound is None:
            s.self_phrase_times.append(s.now)
            s.last_sounds = {"tags": list(sounds), "at": s.now}
        coda = None
        if verb.kind in content.IDLE_KINDS and not moment and self._prd("coda"):
            pool = CODAS + (LATE_CODAS if s.milestones["new_coda"]["status"] == "debuted" else ())
            coda = pool[rng.pick_index(self._rand("coda.pick"), len(pool))]
            if coda == "回頭再看一次" and s.quirks.get("habit") == "double_look":
                self._reveal_quirk("habit", s.quirks["habit_text"])
        event = self._emit(
            "duck.expression",
            {
                "verb": verb_id,
                "zh": verb.zh,
                "reason": reason,
                "sounds": sounds,
                "sound_text": content.phrase_text(tuple(sounds)) if sounds else "",
                "amplitude": amplitude,
                "tempo": tempo,
                "coda": coda,
                "posture": s.posture,
                "validated": verb.validated,
            },
            parents=parents,
        )
        s.energy = max(0, s.energy - EXPRESSION_DRAIN)
        if moment:
            return event  # a moment's beat: not something its idle choices tire of
        # Doing something makes it less interesting for a while. Walking has its own,
        # slower satiation (walk_sat), so the whole-body verbs habituate less here.
        drop = 1000 if verb.kind == "motion" else 2500
        s.interest[verb_id] = max(1000, s.interest.get(verb_id, SCALE) - drop)
        s.last_verb = verb_id
        if verb_id == "snuggle" and reason == "idle":
            self._self_note("snuggler", event, "snuggle")
        if verb.kind in content.IDLE_KINDS and reason in (
            "idle",
            "stood",
            "stand_limit",
            "restless",
            "set_off",
            "walk_on",
            "pause",
        ):
            # Something it chose to do by itself: praise (or "not that") can attach to it.
            s.last_flag = {"kind": "verb", "id": verb_id, "at": s.now, "event": event.event_id}
        if s.quirks.get("favorite_tag") in sounds:
            s.favorite_uses += 1
            if s.favorite_uses >= 20:
                self._reveal_quirk(
                    "favorite_tag",
                    f"我發現我最常說「{content.TAG_ZH[s.quirks['favorite_tag']]}」。",
                )
        return event

    # ------------------------------------------------------------------ voice lineage

    def _maybe_babble(self) -> None:
        s = self.s
        if s.gd < self._stage_day("duckling") or self._until(s.quiet_until) or s.hot:
            return
        s.active_min_since_babble += 1
        if s.active_min_since_babble < 10 or not rng.below(self._rand("babble"), 5000):
            return
        s.active_min_since_babble = 0
        stage = self.stage()
        length = stage.phrase_len
        if stage.stage_id == "adult":
            length = 4 + rng.pick_index(self._rand("babble.len"), 3)
        allowed = ["chirp", "inquire", "coo", "greet"] + (
            ["wheee"] if s.gd >= self._stage_day("juvenile") else []
        )
        parent_code: str | None = None
        promoted = [code for code in s.promotions if code in s.phrases]
        liked = [
            code
            for code, item in sorted(s.phrases.items())
            if item["liked_gds"] and not item["promoted"]
        ]
        rehearse = bool(liked) and rng.below(self._rand("babble.rehearse"), 3000)
        if rehearse:
            code = liked[rng.pick_index(self._rand("babble.rehearse_pick"), len(liked))]
            event = self._emit(
                "voice.babble",
                {
                    "code": code,
                    "text": s.phrases[code]["text"],
                    "tags": list(s.phrases[code]["tags"]),
                    "gen": s.phrases[code]["gen"],
                    "parents": list(s.phrases[code]["parents"]),
                    "trying": True,
                    "rehearsal": True,
                },
            )
            s.self_phrase_times.append(s.now)
            s.last_flag = {"kind": "phrase", "id": code, "at": s.now, "event": event.event_id}
            return
        if promoted and rng.below(self._rand("babble.mutate"), 5000):
            parent_code = promoted[rng.pick_index(self._rand("babble.parent"), len(promoted))]
            tags = list(s.phrases[parent_code]["tags"])
            if len(tags) < length:
                tags.append(allowed[rng.pick_index(self._rand("babble.extend"), len(allowed))])
            else:
                slot = rng.pick_index(self._rand("babble.slot"), len(tags))
                tags[slot] = allowed[rng.pick_index(self._rand("babble.swap"), len(allowed))]
            tags = tags[:length]
        else:
            weights = {tag: 2 for tag in allowed}
            weights[s.quirks.get("favorite_tag", "chirp")] = 6
            if s.branch is not None:
                for tag in content.BRANCH_BY_ID[s.branch].favorite_tags:
                    if tag in weights:
                        weights[tag] += 3
            tags = [rng.pick_weighted(self._rand("babble.tag"), weights) for _ in range(length)]
        code = content.phrase_code(tuple(tags))
        generation = 0 if parent_code is None else int(s.phrases[parent_code]["gen"]) + 1
        if code not in s.phrases:
            s.phrases[code] = {
                "code": code,
                "tags": list(tags),
                "text": content.phrase_text(tuple(tags)),
                "gen": generation,
                "parents": [parent_code] if parent_code else [],
                "born_gd": s.gd,
                "promoted": False,
                "praise": 0,
                "liked_gds": [],
            }
        event = self._emit(
            "voice.babble",
            {
                "code": code,
                "text": s.phrases[code]["text"],
                "tags": list(tags),
                "gen": s.phrases[code]["gen"],
                "parents": list(s.phrases[code]["parents"]),
                "trying": True,
            },
        )
        s.self_phrase_times.append(s.now)
        s.last_flag = {"kind": "phrase", "id": code, "at": s.now, "event": event.event_id}

    def _promote(self, code: str, by: str, evidence: Event) -> str:
        """Praise a phrase. It joins the family's phrases once liked on two different growth days."""

        s = self.s
        phrase = s.phrases.get(code)
        if phrase is None or self._until(s.guest_until):
            return "none"
        if s.gd not in phrase["liked_gds"]:
            phrase["liked_gds"].append(s.gd)
        phrase["praise"] = int(phrase["praise"]) + 1
        ready = (
            len(phrase["liked_gds"]) >= 2
            and not phrase["promoted"]
            and s.milestones["phrase_vote"]["status"] == "debuted"
            and (s.promotion_gd is None or s.gd - s.promotion_gd >= PROMOTION_EVERY_GD)
        )
        if not ready:
            self._emit(
                "voice.liked",
                {"code": code, "text": phrase["text"], "days": len(phrase["liked_gds"]), "by": by},
                parents=(evidence.event_id,),
            )
            self._learn_pref(f"phrase:{code}", +1, evidence)
            return "liked"
        phrase["promoted"] = True
        phrase["promoted_gd"] = s.gd
        if code not in s.promotions:
            s.promotions.append(code)
        s.promotion_gd = s.gd
        self._emit(
            "voice.promoted",
            {"code": code, "text": phrase["text"], "gen": phrase["gen"], "by": by, "gd": s.gd},
            parents=(evidence.event_id,),
        )
        self._learn_pref(f"phrase:{code}", +1, evidence)
        return "promoted"

    def _build_song(self, version: int) -> None:
        s = self.s
        ranked = sorted(
            s.promotions,
            key=lambda code: (-int(s.phrases[code]["praise"]), s.promotions.index(code)),
        )
        chosen = ranked[:3]
        if not chosen:
            fallback = sorted(s.phrases, key=lambda code: (-int(s.phrases[code]["gen"]), code))[:2]
            chosen = fallback or [content.phrase_code(content.WORDS[s.first_word or "hi"])]
            for code in chosen:
                if code not in s.phrases:
                    tags = content.WORDS[s.first_word or "hi"]
                    s.phrases[code] = {
                        "code": code,
                        "tags": list(tags),
                        "text": content.phrase_text(tags),
                        "gen": 0,
                        "parents": [],
                        "born_gd": s.gd,
                        "promoted": False,
                        "praise": 0,
                        "liked_gds": [],
                    }
        s.song = {"version": version, "phrases": chosen, "gd": s.gd}
        text = " ／ ".join(s.phrases[code]["text"] for code in chosen)
        self._emit("voice.song", {"version": version, "phrases": chosen, "text": text, "gd": s.gd})

    def _offer_molt(self) -> None:
        s = self.s
        if s.song is None or (s.molt is not None and s.molt.get("status") == "open"):
            return
        base = list(s.song["phrases"])
        target = base[rng.pick_index(self._rand("molt.slot"), len(base))]
        tags = list(s.phrases[target]["tags"])
        slot = rng.pick_index(self._rand("molt.tag"), len(tags))
        options = [
            tag for tag in ("chirp", "inquire", "coo", "greet", "wheee") if tag != tags[slot]
        ]
        tags[slot] = options[rng.pick_index(self._rand("molt.new"), len(options))]
        code = content.phrase_code(tuple(tags))
        if code not in s.phrases:
            s.phrases[code] = {
                "code": code,
                "tags": tags,
                "text": content.phrase_text(tuple(tags)),
                "gen": int(s.phrases[target]["gen"]) + 1,
                "parents": [target],
                "born_gd": s.gd,
                "promoted": False,
                "praise": 0,
                "liked_gds": [],
            }
        s.molt = {
            "status": "open",
            "gd": s.gd,
            "replace": target,
            "with": code,
            "text": s.phrases[code]["text"],
        }
        self._emit(
            "voice.molt_offer",
            {
                "replace": target,
                "with": code,
                "text": s.phrases[code]["text"],
                "default": "keep",
                "gd": s.gd,
            },
        )

    # ------------------------------------------------------------------ learning

    def _learn_pref(self, item: str, direction: int, evidence: Event) -> None:
        s = self.s
        if self._until(s.guest_until):
            return
        # (Curiosity's things and programmes keep their own keys: thing:<id>, curio:<id>.)
        key = item if item.startswith(curiosity.PREF_PREFIXES) else f"household.{item}"
        used = s.pref_gd_count.get(key, 0)
        if used >= len(PREF_STEPS):
            return
        delta = PREF_STEPS[used] * direction
        before = s.prefs.get(key, 5000)
        after = clamp(before + delta)
        if after == before:
            return
        update = LearningUpdate(
            update_id=f"U-{evidence.event_id}-{item}",
            episode_id=s.trace,
            parameter=key,
            before=before / SCALE,
            after=after / SCALE,
            max_abs_delta=abs(delta) / SCALE,
            rule_version="household-pref-v1",
            outcome_ids=(evidence.event_id,),
            status=LearningStatus.COMMITTED,
        )
        self._emit(
            "learning.committed",
            {
                "update_id": update.update_id,
                "parameter": update.parameter,
                "before": update.before,
                "after": update.after,
                "max_abs_delta": update.max_abs_delta,
                "rule_version": update.rule_version,
                "evidence_event_ids": [evidence.event_id],
            },
            parents=(evidence.event_id,),
        )
        s.prefs[key] = after
        s.pref_gd_count[key] = used + 1
        if after >= 6500 and not s.favorite_found and item.startswith("verb:"):
            self._experience_milestone("favorite_found")

    def _learn_owner_bond(self, evidence: Event) -> None:
        s = self.s
        if s.owner_bond_gd >= OWNER_BOND_CAP_PER_GD:
            return
        before = s.bond_owner
        step = max(20, OWNER_BOND_STEP * (SCALE - before) // SCALE)  # diminishing near the top
        after = clamp(before + step)
        if after == before:
            return
        update = LearningUpdate(
            update_id=f"U-{evidence.event_id}-bond",
            episode_id=s.trace,
            parameter="owner.bond",
            before=before / SCALE,
            after=after / SCALE,
            max_abs_delta=OWNER_BOND_STEP / SCALE,
            rule_version="owner-bond-v1",
            outcome_ids=(evidence.event_id,),
            status=LearningStatus.COMMITTED,
        )
        self._emit(
            "learning.committed",
            {
                "update_id": update.update_id,
                "parameter": "owner.bond",
                "before": update.before,
                "after": update.after,
                "max_abs_delta": update.max_abs_delta,
                "rule_version": update.rule_version,
                "evidence_event_ids": [evidence.event_id],
            },
            parents=(evidence.event_id,),
        )
        s.bond_owner = after
        s.owner_bond_gd += 1

    def _care(self, care_type: str, evidence: Event | str, why: str | None = None) -> None:
        """One episode of care (at most CARE_CAP_PER_GD of each kind count a growth day):
        from becoming a duckling to becoming a juvenile they shape its branch, from then to
        adulthood its adult name."""
        s = self.s
        parent = evidence if isinstance(evidence, str) else evidence.event_id
        extra = {"why": why} if why else {}
        if self._until(s.guest_until):
            self._emit(
                "care.episode",
                {"type": care_type, "gd": s.gd, "counted": False, "why": "guest_mode"},
                parents=(parent,),
            )
            return
        counted = s.care_today.get(care_type, 0) < CARE_CAP_PER_GD
        s.care_today[care_type] = s.care_today.get(care_type, 0) + 1
        duckling, juvenile = self._stage_day("duckling"), self._stage_day("juvenile")
        if counted and duckling <= s.gd < juvenile:
            s.care_window[care_type] += 1
        if counted and juvenile <= s.gd < self._stage_day("adult"):
            s.care_adult[care_type] += 1
        s.care_at[care_type] = s.now
        self._emit(
            "care.episode",
            {"type": care_type, "gd": s.gd, "counted": counted, **extra},
            parents=(parent,),
        )

    def _self_note(self, branch_id: str, evidence: Event | str, why: str) -> None:
        """Its own experience (spec 7.3), from becoming a duckling to becoming a juvenile,
        at most SELF_CAP_PER_GD of each kind a growth day: exploring somewhere new
        (explorer), a babble answered (songbird), sitting down to enjoy a touch or
        snuggling (snuggler), a long nap (dreamer)."""
        s = self.s
        if not self._stage_day("duckling") <= s.gd < self._stage_day("juvenile"):
            return
        if s.self_today.get(branch_id, 0) >= content.SELF_CAP_PER_GD:
            return
        s.self_today[branch_id] = s.self_today.get(branch_id, 0) + 1
        s.self_window[branch_id] = s.self_window.get(branch_id, 0) + 1
        parent = evidence if isinstance(evidence, str) else evidence.event_id
        self._emit("care.self", {"branch": branch_id, "why": why, "gd": s.gd}, parents=(parent,))

    def _care_limited(self, care_type: str, key: str, minutes: int, evidence: str) -> None:
        """Care that counts at most once every ``minutes`` (fair care: company, play)."""
        s = self.s
        last = s.care_at.get(key)
        if last is not None and s.now - last < timedelta(minutes=minutes):
            return
        s.care_at[key] = s.now
        self._care(care_type, evidence, why=key)

    def _company_minute(self) -> None:
        """Fair care from simply being there (spec 7.3): the family awake and right here
        while it is out exploring counts as play (at most every 10 minutes); here, nobody
        having paid it attention for 20 minutes, while it rests 20 minutes more in a row,
        as calm company (at most every 20 minutes)."""
        s = self.s
        limits: dict[str, Any] = content.CARE_LIMITS
        evidence = s.room_cue_event
        if evidence is None or not daycycle.someone_here(self):
            s.company_min = 0
            return
        if s.excursion is not None:
            s.company_min = 0
            self._care_limited("play", "company_play", int(limits["company_play_min"]), evidence)
            return
        quiet = s.presence_at is None or s.now - s.presence_at >= timedelta(
            minutes=int(limits["company_calm_min"])
        )
        if s.posture != "sit" or not quiet:
            s.company_min = 0
            return
        s.company_min += 1
        if s.company_min >= int(limits["company_calm_min"]):
            s.company_min = 0
            self._care_limited("calm", "company_calm", int(limits["company_calm_min"]), evidence)

    def _flag_within(self, window: timedelta) -> dict[str, Any] | None:
        flag = self.s.last_flag
        if flag is None or self.s.now - flag["at"] > window:
            return None
        return flag

    def _praise(self, channel: str, persona: str | None, evidence: Event) -> str:
        s = self.s
        count_care = channel != "touch"
        flag = self._flag_within(PRAISE_WINDOW)
        s.valence = min(SCALE, s.valence + 1000)
        self._contact("praise")
        excursion.good_here(self, 800)
        note = "開心"
        # Fair care (spec 7.3): praise counts as the care of what was praised.
        care_type = "talk"
        if flag is not None:
            if flag["kind"] == "phrase":
                outcome = self._promote(
                    str(flag["id"]), "owner" if channel == "owner" else "household", evidence
                )
                text = s.phrases[flag["id"]]["text"]
                if outcome == "promoted":
                    note = f"「{text}」被誇了兩天，正式成為我們家的片語！"
                elif outcome == "liked":
                    note = f"記住你們喜歡「{text}」；牠會再試著說，改天再被誇就會留下來"
                else:
                    note = "開心（訪客模式中，不學新東西）"
            elif flag["kind"] == "moment":
                moment = content.MOMENT_BY_ID[str(flag["id"])]
                self._learn_pref(f"moment:{moment.moment_id}", +1, evidence)
                care_type = moment.care
                note = f"記住你們喜歡牠這樣：{moment.zh}"
            else:
                self._learn_pref(f"verb:{flag['id']}", +1, evidence)
                verb = content.VERB_BY_ID[str(flag["id"])]
                note = f"記住你們喜歡「{verb.zh}」"
                care_type = content.VERB_CARE.get(
                    verb.verb_id, content.FIT_CARE.get(verb.fit, "talk")
                )
        if count_care:
            self._care(care_type, evidence, why="praise")
        quiet = self._until(s.quiet_until)
        self._express(
            "happy_sway",
            reason="praised",
            parents=(evidence.event_id,),
            force_sound=() if quiet else ("chirp",),
        )
        s.last_flag = None
        return note

    # ------------------------------------------------------------------ growth

    def _debut_pending(self) -> None:
        for milestone_id, info in sorted(
            self.s.milestones.items(), key=lambda item: item[1]["order"]
        ):
            if info["status"] == "pending":
                self._debut(milestone_id)

    def _debut(self, milestone_id: str) -> None:
        s = self.s
        info = s.milestones[milestone_id]
        if info["status"] == "debuted":
            return
        milestone = content.MILESTONE_BY_ID.get(milestone_id)
        title = info["title"]
        text = milestone.diary if milestone is not None else str(info.get("text", ""))
        unlocks = list(milestone.unlocks) if milestone is not None else []
        extra: dict[str, Any] = {}
        if milestone_id == "first_word":
            if s.first_word is None:
                s.first_word, weights = self._first_word()
                extra = {
                    "word": s.first_word,
                    "weights": weights,
                    "rule": s.rules.get("first_word"),
                }
            text = text.format(word=content.phrase_text(content.WORDS[s.first_word]))
        elif milestone_id == "hatched":
            text = text.format(name=s.name)
        elif milestone_id == "leaning":
            text = self._leaning_text()
        elif milestone_id == "branch":
            if s.branch is None:
                self._choose_branch()
            branch = content.BRANCH_BY_ID[s.branch or "dreamer"]
            text = text.format(branch=branch.zh)
            unlocks.append(branch.signatures[0])
        elif milestone_id == "song_draft":
            self._build_song(1)
            if s.branch is not None:
                unlocks.append(content.BRANCH_BY_ID[s.branch].signatures[1])
        elif milestone_id == "song_draft_2":
            self._build_song(2)
        elif milestone_id == "quarter":
            self._offer_molt()
        elif milestone_id == "adult":
            if s.adult_name is None:
                self._choose_adult()
            self._build_song(3)
            text = f"我長大了，是「{moments.title(self) or s.adult_name}」！我的歌也定型了。"
        elif milestone_id == "first_steps":
            unlocks.append("turn_around")  # after the first steps it can turn to look behind
        for verb_id in unlocks:
            if verb_id not in s.unlocked:
                s.unlocked.append(verb_id)
        info["status"] = "debuted"
        info["debuted_gd"] = s.gd
        info["text"] = text
        event = self._emit(
            "growth.milestone",
            {
                "milestone": milestone_id,
                "title": title,
                "text": text,
                "unlocks": unlocks,
                "gd": s.gd,
                **extra,
            },
        )
        if milestone_id == "branch":
            self._express(
                "perk_tilt",
                reason="proud_pose",
                parents=(event.event_id,),
                force_sound=("greet", "chirp"),
            )

    def _experience_milestone(self, milestone_id: str, gentle: bool = False) -> None:
        s = self.s
        if milestone_id in s.milestones:
            return
        title, text = EXPERIENCE_MILESTONES[milestone_id]
        if milestone_id == "favorite_found":
            s.favorite_found = True
            verb_prefs = {
                key: value for key, value in s.prefs.items() if key.startswith("household.verb:")
            }
            if verb_prefs:
                best = max(sorted(verb_prefs), key=lambda key: verb_prefs[key]).split(":", 1)[1]
                favorite = content.VERB_BY_ID[best].zh
            else:
                favorite = content.VERB_BY_ID["happy_sway"].zh
            text = (
                text.format(favorite=favorite) if not gentle else f"我好像比較喜歡「{favorite}」。"
            )
        elif milestone_id == "learned_name":
            text = text.format(name=s.name)
        s.milestones[milestone_id] = {
            "gd": s.gd,
            "status": "pending",
            "title": title,
            "order": 100 + len(s.milestones),
            "text": text,
        }
        if s.awake and self._present(15):
            self._debut(milestone_id)

    @staticmethod
    def _lift(counts: dict[str, int], prior: dict[str, int]) -> dict[str, int]:
        """How much more of each kind than an average duck gets (share ÷ prior, at most
        BRANCH_RULE cap), normalised to sum 1.0 (fixed point). Nothing yet: all even."""
        keys = sorted(prior)
        total = sum(max(0, int(counts.get(key, 0))) for key in keys)
        if total <= 0:
            return {key: SCALE // len(keys) for key in keys}
        cap = int(content.BRANCH_RULE["cap"])
        raw = {
            key: min(
                cap, max(0, int(counts.get(key, 0))) * SCALE // total * SCALE // max(1, prior[key])
            )
            for key in keys
        }
        norm = sum(raw.values()) or 1
        return {key: value * SCALE // norm for key, value in raw.items()}

    def _branch_scores(
        self, care: dict[str, int], prior: dict[str, int], exclude: str | None = None
    ) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
        """branch-v2's scores (spec 7.4): 0.4 nature + 0.4 care lift + 0.2 its own
        experience (0.7 nature + 0.3 experience with too little care to go by)."""
        s = self.s
        rule: dict[str, int] = content.BRANCH_RULE
        few: dict[str, int] = content.BRANCH_NATURE_ONLY
        lift = self._lift(care, prior)
        own = self._lift(s.self_window, content.SELF_PRIOR)
        enough = sum(care.values()) >= BRANCH_MIN_EPISODES
        scores = {}
        for branch in content.BRANCHES:
            b = branch.branch_id
            if b == exclude:
                continue
            if enough:
                score = (
                    s.nature[b] * rule["nature"]
                    + lift[branch.care_type] * rule["care"]
                    + own[b] * rule["self"]
                ) // SCALE
            else:
                score = (s.nature[b] * few["nature"] + own[b] * few["self"]) // SCALE
            scores[b] = score
        return scores, lift, own

    def _draw_branch(
        self, purpose: str, scores: dict[str, int]
    ) -> tuple[str, dict[str, int], int, int]:
        """The lottery: weight score^k, drawn (choice, probabilities, draw, counter)."""
        k = int(content.BRANCH_RULE["k"])
        weights = {b: max(1, value**k // SCALE ** (k - 1)) for b, value in scores.items()}
        raw = self._rand(purpose)
        choice = rng.pick_weighted(raw, weights)
        total = sum(weights.values())
        probabilities = {b: weight * SCALE // total for b, weight in sorted(weights.items())}
        return choice, probabilities, rng.unit(raw), self.counters[purpose]

    def _choose_branch(self) -> None:
        """branch-v2 (spec 7.4): its nature, how much more of each care its family gave than
        most families do, and its own experience; then a lottery, so care clearly raises
        the odds and the same family can still raise different ducks."""
        s = self.s
        if s.rules.get("branch") == "branch-v1":
            self._choose_branch_v1()
            return
        scores, lift, own = self._branch_scores(s.care_window, content.CARE_PRIOR)
        choice, probabilities, draw, counter = self._draw_branch("branch.pick", scores)
        s.branch = choice
        others = sorted(
            (b for b in probabilities if b != choice), key=lambda b: (-probabilities[b], b)
        )
        s.runner_up = others[0] if others else None
        s.branch_scores = scores
        total = sum(s.care_window.values())
        s.branch_detail.update(
            {"lift": lift, "self": own, "probabilities": probabilities, "episodes": total}
        )
        self._emit(
            "growth.branch",
            {
                "branch": s.branch,
                "zh": content.BRANCH_BY_ID[s.branch].zh,
                "runner_up": s.runner_up,
                "scores": scores,
                "care": dict(s.care_window),
                "episodes": total,
                "nature_only": total < BRANCH_MIN_EPISODES,
                "nature": dict(s.nature),
                "primary": s.branch_detail.get("primary"),
                "lift": lift,
                "self": own,
                "self_window": dict(s.self_window),
                "probabilities": probabilities,
                "draw": draw,
                "counter": counter,
                "rule": "branch-v2",
            },
        )

    def _choose_branch_v1(self) -> None:
        s = self.s
        total = sum(s.care_window.values())
        scores: dict[str, int] = {}
        for branch in content.BRANCHES:
            share = s.care_window[branch.care_type] * SCALE // total if total else 0
            noise = rng.unit(self._rand(f"branch.noise.{branch.branch_id}")) * 1000 // SCALE - 500
            if total < BRANCH_MIN_EPISODES:
                scores[branch.branch_id] = s.nature[branch.branch_id] + noise
            else:
                scores[branch.branch_id] = s.nature[branch.branch_id] // 2 + share // 2 + noise
        ranked = sorted(scores, key=lambda key: (-scores[key], key))
        s.branch, s.runner_up = ranked[0], ranked[1]
        s.branch_scores = scores
        self._emit(
            "growth.branch",
            {
                "branch": s.branch,
                "zh": content.BRANCH_BY_ID[s.branch].zh,
                "runner_up": s.runner_up,
                "scores": scores,
                "care": dict(s.care_window),
                "episodes": total,
                "nature_only": total < BRANCH_MIN_EPISODES,
                "nature": dict(s.nature),
                "rule": "branch-v1",
            },
        )

    def _choose_adult(self) -> None:
        """adult-v2 (spec 7.5): the second branch by the same lottery over the care it got
        as a juvenile (never its own branch), so all twelve names happen; and its epithet,
        from its running gag or the moment it had most often."""
        s = self.s
        if s.rules.get("adult") == "adult-v1":
            self._choose_adult_v1()
            return
        primary = s.branch or "dreamer"
        scores, lift, _ = self._branch_scores(s.care_adult, content.CARE_PRIOR_ADULT, primary)
        secondary, probabilities, draw, counter = self._draw_branch("adult.pick", scores)
        s.adult_name = content.ADULT_NAMES[(primary, secondary)]
        s.epithet = moments.epithet(self)
        self._emit(
            "growth.adult",
            {
                "name": s.adult_name,
                "primary": primary,
                "secondary": secondary,
                "scores": scores,
                "lift": lift,
                "probabilities": probabilities,
                "draw": draw,
                "counter": counter,
                "epithet": s.epithet,
                "epithet_zh": content.EPITHETS.get(s.epithet or ""),
                "title": moments.title(self),
                "rule": "adult-v2",
            },
        )

    def _choose_adult_v1(self) -> None:
        s = self.s
        primary = s.branch or "dreamer"
        others = {
            branch.branch_id: s.care_adult[branch.care_type]
            for branch in content.BRANCHES
            if branch.branch_id != primary
        }
        secondary = (
            max(sorted(others), key=lambda key: others[key])
            if any(others.values())
            else (s.runner_up or "songbird")
        )
        if secondary == primary:
            secondary = "songbird" if primary != "songbird" else "dreamer"
        s.adult_name = content.ADULT_NAMES[(primary, secondary)]
        self._emit(
            "growth.adult",
            {"name": s.adult_name, "primary": primary, "secondary": secondary, "rule": "adult-v1"},
        )

    def _reveal_quirk(self, key: str, text: str) -> None:
        s = self.s
        if key in s.revealed:
            return
        s.revealed.append(key)
        self._emit("growth.quirk", {"quirk": key, "text": text})

    # ------------------------------------------------------------------ bids and rare events

    def _maybe_bid(self) -> None:
        s = self.s
        if s.bids_stopped or self._until(s.quiet_until) or s.boot_at is None:
            return
        if s.now - s.boot_at < timedelta(minutes=5) or s.bid_pending_since is not None:
            return
        if s.docked and (s.now.hour >= 22 or s.now.hour < 7):
            return
        social: dict[str, Any] = content.SOCIAL
        if self.aware:
            # Only while the family is up and someone was right here a moment ago, and
            # nobody has paid it attention for a while: then a call is usually answered.
            if s.phase != "family" or s.family != "awake" or self._until(s.napping_until):
                return
            if not daycycle.recent(self, s.room_logged_at, int(social["bid_room_min"])):
                return
            if s.presence_at is not None and s.now - s.presence_at < timedelta(
                minutes=int(social["bid_quiet_min"])
            ):
                return
        elif not self._present(int(content.RHYTHM["room_timeout_min"])):
            return  # nobody around to hear it
        if s.bids_today >= BIDS_PER_DAY or len(s.bid_times) >= self.stage().bids_per_hour:
            return
        if s.bid_times and s.now - s.bid_times[-1] < timedelta(minutes=10):
            return
        # A constant chance, never growing with time since the last interaction.
        chance = 800 * s.bid_backoff // SCALE
        if not rng.below(self._rand("bid"), chance):
            return
        kind = "greet"
        if (
            not s.come_play_this_visit
            and s.play_accepted_gd is not None
            and s.gd - s.play_accepted_gd <= 7
        ):
            kind = "come_play"
            s.come_play_this_visit = True
        comes_over = (
            kind == "greet"
            and self.hw.nest
            and s.posture == "sit"
            and excursion.can_start(self) is None
            and rng.below(self._rand("bid.kind"), 5000)
        )
        if comes_over:
            kind = "comes_over"  # it walks over to the family (an outing, too)
        s.bid_times.append(s.now)
        s.bids_today += 1
        s.bid_pending_since = s.now
        sounds: tuple[str, ...] = (
            ("greet",) if kind in ("greet", "comes_over") else ("inquire", "chirp")
        )
        if s.hush:
            sounds = ()
        bid = self._emit(
            "duck.bid",
            {
                "kind": kind,
                "sounds": list(sounds),
                "text": content.phrase_text(sounds) if sounds else "",
                "room_min": daycycle.minutes_since(self, s.room_logged_at),
            },
        )
        if comes_over:
            excursion.set_out(self, "social", parents=(bid.event_id,))

    def _check_bid_answer(self) -> None:
        s = self.s
        answer_s = int(content.SOCIAL["bid_answer_s"])
        if s.bid_pending_since is None or s.now - s.bid_pending_since < timedelta(seconds=answer_s):
            return
        s.bid_pending_since = None
        s.unanswered += 1
        s.bid_backoff = s.bid_backoff // 2
        s.bid_times.append(s.now + timedelta(minutes=20))
        if s.unanswered >= 2:
            s.bids_stopped = True
        self._emit("duck.bid_unanswered", {"unanswered": s.unanswered, "stopped": s.bids_stopped})

    def _prd(self, tier: str) -> bool:
        s = self.s
        spec = content.RARE_TIERS[tier]
        if s.gd < int(spec["gd_min"]):
            return False
        last = s.rare_last.get(tier)
        if last is not None and s.now - last < timedelta(minutes=int(spec["min_gap_min"])):
            return False
        if s.rare_today.get(tier, 0) >= int(spec["per_day"]):
            return False
        if spec["presence"]:
            fresh_session = s.boot_at is not None and s.now - s.boot_at < timedelta(minutes=15)
            if not (self._present(15) or fresh_session):
                return False
        n = s.rare_n.get(tier, 0) + 1
        s.rare_n[tier] = n
        chance = min(content.PPM, int(spec["c_ppm"]) * n)
        raw = self._rand(f"prd.{tier}")
        if (raw * content.PPM) >> 64 >= chance:
            return False
        s.rare_n[tier] = 0
        s.rare_last[tier] = s.now
        s.rare_today[tier] = s.rare_today.get(tier, 0) + 1
        return True

    def _maybe_rare(self) -> None:
        s = self.s
        s.active_min_since_rare += 1
        if s.active_min_since_rare < 5:
            return
        s.active_min_since_rare = 0
        for tier in ("very_rare", "rare", "uncommon"):
            if not self._prd(tier):
                continue
            pool = []
            for rare_id, zh in content.RARE_POOLS[tier]:
                if rare_id == "sneeze" and s.gd < 60:
                    continue
                if rare_id == "hiccup" and (
                    s.nap_ended_at is None or s.now - s.nap_ended_at > timedelta(minutes=10)
                ):
                    continue
                if rare_id == "sleep_mutter":
                    continue  # only while napping
                if rare_id == "full_song" and s.song is None:
                    continue
                if rare_id == "other_signature" and s.branch is None:
                    continue
                pool.append((rare_id, zh))
            if not pool:
                continue
            rare_id, zh = pool[rng.pick_index(self._rand(f"rare.pick.{tier}"), len(pool))]
            new = rare_id not in s.album
            if new:
                s.album.append(rare_id)
            self._emit("rare.event", {"tier": tier, "id": rare_id, "zh": zh, "new": new})
            return

    # ------------------------------------------------------------------ stimuli (perceived)

    def _needs_awake(self, persona: str | None, action: str) -> bool:
        if self.s.awake:
            return True
        self._journal(persona, action, "小鴨現在關機（換電池或充電中），沒有感覺到", False, [])
        return False

    def _arouse(self, amount: int) -> None:
        """Something happened: a little more alert for a while (relaxes 5% a minute)."""
        self.s.arousal = min(SCALE, self.s.arousal + amount)

    # ------------------------------------------------------------------ percepts

    @_input
    def on_percept(self, percept: Mapping[str, Any], *, via: str = VIA_INPUT) -> dict[str, Any]:
        """The input handler for every percept (percept-v1; design §7): what the world sensed
        (its port delivered it, ``via`` pull / reply / poll) or what an outside sender handed
        in (``via`` input: api.sense, the family proxy after the world's detection).

        First the record of the input: kind = the percept's kind, source = its ``src``, the
        payload every other field (``t_ms`` and ``pv`` kept), parents = the command it
        answers. A result naming an act no one is waiting for (late, or for an act already
        closed) is recorded ``stale`` and not used; one naming no act is unasked (phase 5's
        duck-eye seeing the nest by itself, spec §7.2) and taken as such. A ``body.fell`` is
        always taken: the body is the authority on falling (a walk Duck Core gave up on was
        stopped, boundary §4). Then v4's logic for that kind. Returns ``{"taken", "event",
        "note", "act_id"}``."""
        s = self.s
        kind = str(percept["kind"])
        if via == VIA_INPUT:
            worldport.check(percept)
        elif self.verify_left is None or self.verify_left > 0:
            worldport.check(percept)
            if self.verify_left is not None:
                self.verify_left -= 1
        acts = self.acts
        act = acts.settle(percept)  # the act it answers (now closed), if it answers one
        stale = (act is None and kind in RESULTS and percept.get("act_id") is not None) or (
            kind == "body.righting" and s.righting is None
        )
        payload = {key: value for key, value in percept.items() if key != "kind" and key != "src"}
        parents: tuple[str, ...] = ()
        if act is not None:
            parents = (act.event_id,)
        elif kind == "body.righting" and s.righting is not None:
            parents = (str(s.righting["fell"]),)
        elif kind == "body.fell":
            ctx = percept.get("ctx")
            ctx_act = ctx.get("act_id") if isinstance(ctx, Mapping) else None
            walking = acts.open.get(ctx_act) if isinstance(ctx_act, str) else None
            if walking is not None:
                parents = (walking.event_id,)
        if stale:
            payload["stale"] = True
        record = self._emit(kind, payload, source=str(percept["src"]), parents=parents)
        if act is not None:
            act.record_id = record.event_id
        out: dict[str, Any] = {
            "taken": not stale,
            "event": record.event_id,
            "note": None,
            "act_id": act.act_id if act is not None else None,
        }
        if not stale:
            handler = _PERCEPT_HANDLERS.get(kind)
            if handler is not None:
                out["note"] = handler(self, percept, record, act)
        return out

    def _on_walk_done(self, p: Mapping[str, Any], record: Event, act: Act | None) -> str | None:
        s = self.s
        plan = s.walk_plan
        if act is None or plan is None or plan.get("act_id") != act.act_id:
            return None  # a walk it already cut (a hand, a fall): the act is closed, that's all
        if p.get("reason") == "time":
            self._finish_walk()
            return None
        # Ended early by the body itself (phase 5: a reflex, a stall …): where it stopped.
        self._cut_walk()
        s.posture = "stand"
        s.stand_idle_from = s.now
        self._walk_keepsakes(record.event_id)
        return None

    def _on_fell(self, p: Mapping[str, Any], record: Event, act: Act | None) -> str | None:
        if self.s.posture == "fallen" or not self.s.awake:
            return None
        self._fall(record)
        return None

    def _on_righting(self, p: Mapping[str, Any], record: Event, act: Act | None) -> str | None:
        tries = p.get("tries")
        self._righting_try(
            record, bool(p.get("ok")), tries if type(tries) is int and tries > 0 else None
        )
        return None

    def _on_touch(self, p: Mapping[str, Any], record: Event, act: Act | None) -> str | None:
        s = self.s
        if not s.awake:
            return None
        if p.get("legacy_phantom"):
            # The TV mistaken for a scratch on the head (LegacyWorld's false alarm; it
            # reacts as v4 did to a phantom: no praise, no liked place, no settling).
            self._journal(
                None, "false_alarm", "電視聲被誤判成搔頭（每清醒小時約 0.2 次）", False, [record]
            )
            return self._touch_reaction(record, calm=False, phantom=True)
        return self._touch_reaction(record, calm=self._until(s.napping_until))

    def _on_hold_head(self, p: Mapping[str, Any], record: Event, act: Act | None) -> str | None:
        """A hand holds its head (detected): it stops, stays still a minute, and after three
        stops in ten minutes it keeps quiet for half an hour."""
        s = self.s
        if not s.awake:
            return None
        moving = not self._until(s.napping_until)
        self._presence()
        walking = s.posture == "walk"
        if walking:
            # Stop at once and stay standing: sitting down under a hand would be abrupt.
            self._set_posture("stand", reason="held", by="hand", parents=(record.event_id,))
        s.stop_until = s.now + timedelta(seconds=60)
        s.outing_legs = 0
        self._emit(
            "duck.stop",
            {
                "until": s.stop_until.isoformat(),
                "rest_pose": True,
                "posture": s.posture,
                "sounds": ["chirp"],
            },
            parents=(record.event_id,),
        )
        if moving and (not s.stop_moments or s.now - s.stop_moments[-1] >= timedelta(seconds=30)):
            s.stop_moments.append(s.now)
        s.stop_moments = [
            moment for moment in s.stop_moments if s.now - moment <= timedelta(minutes=10)
        ]
        note = "停下來，送休息姿勢，啾一聲確認；60 秒沒碰就恢復"
        if walking:
            note = "立刻停下腳步、站著不動（不會突然坐下），啾一聲確認；60 秒沒碰就恢復"
        if len(s.stop_moments) >= 3:
            s.lockout_until = s.now + timedelta(minutes=30)
            s.stop_moments = []
            self._emit(
                "duck.lockout", {"until": s.lockout_until.isoformat()}, parents=(record.event_id,)
            )
            note = "10 分鐘內被叫停 3 次 → 先安靜 30 分鐘（不是懲罰）"
        return note

    def _on_thing(self, p: Mapping[str, Any], record: Event, act: Act | None) -> str | None:
        curiosity.on_sense(self, p, record)
        return None

    def _on_curio_step(self, p: Mapping[str, Any], record: Event, act: Act | None) -> str | None:
        curiosity.on_result(self, p, record)
        return None

    def _on_guard(self, p: Mapping[str, Any], record: Event, act: Act | None) -> str | None:
        curiosity.on_guard(self, p, record)
        return None

    @_input
    def spend_input(self, seconds: int) -> None:
        """An input's own seconds (a 6-second touch, a 3-second hold): time passes as in
        v4's ``_spend`` (world percepts due meanwhile are taken in at their second), but the
        minute's own clock does not count them."""
        self._spend(max(0, int(seconds)))

    def needs_awake(self, persona: str | None, action: str) -> bool:
        """For the family proxy: whether it can sense anything right now (journals why not)."""
        return self._needs_awake(persona, action)

    def journal_entry(
        self,
        persona: str | None,
        action: str,
        detail: str,
        perceived: bool,
        event_ids: tuple[str, ...] | list[str] = (),
    ) -> None:
        """For the family proxy: a line of the simulator's ground-truth journal (never the
        duck's memory)."""
        events = [event for event_id in event_ids if (event := self.log.get(event_id)) is not None]
        self._journal(persona, action, detail, perceived, events)

    def _touch_reaction(self, event: Event, calm: bool, phantom: bool = False) -> str:
        s = self.s
        self._presence("" if phantom else "touch", event)
        flag = self._flag_within(PRAISE_WINDOW)
        if flag is not None and not phantom:
            self._praise("touch", None, event)
        burst = s.last_touch_at is not None and s.now - s.last_touch_at < timedelta(minutes=2)
        s.last_touch_at = s.now
        s.touches += 1
        s.overload = min(SCALE, s.overload + 700 + (700 if burst else 0))
        if s.overload >= 7000:
            s.overload = 4000
            self._emit(
                "duck.reaction",
                {"to": "overload", "verb": "tuck_away", "text": "太多了，縮一下"},
                parents=(event.event_id,),
            )
            self._express("tuck_away", reason="overload", parents=(event.event_id,), force_sound=())
            return "太多了，縮一下（這個反應永遠不會被習慣掉）"
        if self._until(s.stop_until) or self._until(s.lockout_until):
            return "停止中，只輕輕動一下嘴"
        if calm:
            self._care("calm", event)
            s.energy = min(SCALE, s.energy + 200)
            if self._nap_wakes(str(event.payload.get("firmness", "normal"))):
                self._end_nap("touch", parents=(event.event_id,))
                return "被摸醒了，伸個懶腰"
            self._sleep_stir("touch", "nuzzle", parents=(event.event_id,))
            return "在你手下蹭一蹭，繼續打瞌睡"
        chance = COO_CHANCE[min(s.coo_step, len(COO_CHANCE) - 1)]
        if s.coo_series_at is None:
            s.coo_series_at = s.now
        s.coo_step += 1
        coo = rng.below(self._rand("touch.coo"), chance) and not self._until(s.quiet_until)
        s.valence = min(SCALE, s.valence + 600)
        self._contact("touch")
        if not phantom:
            excursion.good_here(self)
        self._arouse(300)
        self._care("touch", event)
        settled = None
        if s.posture == "stand" and not phantom:
            # A cuddly duck may sit down to enjoy it; an explorer mostly stays up.
            settled = self._reflex(
                "sit_down",
                "touch_settle",
                self._style("settle", self._leaning()),
                (event.event_id,),
            )
        if settled is not None:
            self._self_note("snuggler", event, "settled_for_touch")
        sounds = ("coo",) if coo else ()
        reaction = "melt" if s.branch == "snuggler" and "melt" in s.unlocked else "perk_tilt"
        self._emit(
            "duck.reaction",
            {
                "to": "touch",
                "mouth": "open_first",
                "sounds": list(sounds),
                "verb": reaction,
                "coo_chance": chance,
            },
            parents=(event.event_id,),
        )
        self._express(reaction, reason="touched", parents=(event.event_id,), force_sound=sounds)
        return (
            ("坐下來享受，" if settled is not None else "")
            + "先張嘴，再"
            + ("咕～一聲" if coo else "安靜地歪頭（這次沒出聲，每次反應都有點不同）")
        )

    @_input
    def pickup(self, persona: str) -> dict[str, Any]:
        s = self.s
        if not self._needs_awake(persona, "pickup"):
            return {"perceived": False, "note": "關機中"}
        if not self.hw.pickup_detect:
            self._journal(
                persona, "pickup", "抱起來 → 這個硬體設定偵測不到（上游已棄用抱起偵測）", False, []
            )
            self._spend(5)
            return {"perceived": False, "note": "偵測不到抱起"}
        event = self._emit("sense.pickup", {"path": "imu"}, source="imu")
        self._journal(persona, "pickup", "抱起來", True, [event])
        self._presence()
        self._contact("pickup")
        # Held in the air, then set down seated: a fall ends here, and so does floor mode.
        self._set_posture("sit", reason="picked_up", by="hand", parents=(event.event_id,))
        if self.hw.nest:
            # Lifted off (the nest, or the floor): it does not know where it will be put.
            nest.undock(self, "pickup", (event.event_id,))
            moments.homing_off(self)
            s.homing = None
            s.stranded = None
            s.unplaced = True
        if s.floor:
            self._set_floor(False, "pickup", parents=(event.event_id,))
        self._arouse(800)
        quick = s.last_pickup_at is not None and s.now - s.last_pickup_at < timedelta(minutes=2)
        s.last_pickup_at = s.now
        s.pickups += 1
        s.overload = min(SCALE, s.overload + (1200 if quick else 400))
        if s.overload >= 7000:
            s.overload = 4000
            self._emit(
                "duck.reaction",
                {"to": "overload", "verb": "tuck_away", "text": "太多了，縮一下"},
                parents=(event.event_id,),
            )
            self._express("tuck_away", reason="overload", parents=(event.event_id,), force_sound=())
            self._spend(5)
            return {"perceived": True, "note": "被抱太多次，縮一下"}
        if s.pickups <= 5:
            self._emit(
                "duck.reaction",
                {"to": "pickup", "verb": "freeze", "sounds": ["inquire"], "text": "僵住，嗯？"},
                parents=(event.event_id,),
            )
            note = "僵住，發出「嗯？」（前 5 次）"
        else:
            self._emit(
                "duck.reaction",
                {"to": "pickup", "verb": "snuggle", "sounds": ["coo"], "text": "咕～"},
                parents=(event.event_id,),
            )
            note = "舒服地咕～"
        self._care("touch", event)
        if s.need is not None and s.need.get("kind") == "help":
            s.need = None
        if s.pickups == 5:
            self._experience_milestone("likes_pickup")
        self._spend(5)
        return {"perceived": True, "note": note}

    @_input
    def hand_near(self, persona: str, *, speed: str = "slow") -> dict[str, Any]:
        s = self.s
        if not self._needs_awake(persona, "hand_near"):
            return {"perceived": False, "note": "關機中"}
        if not self.hw.tof:
            self._journal(persona, "hand_near", "手伸過來 → 這隻鴨沒有 ToF，看不到手", False, [])
            self._spend(2)
            return {"perceived": False, "note": "沒有 ToF"}
        event = self._emit("sense.hand_near", {"speed": speed}, source="tof")
        self._journal(
            persona, "hand_near", f"手{'慢慢' if speed == 'slow' else '很快地'}靠近", True, [event]
        )
        self._presence()
        since = None
        if s.last_hand_at is not None:
            since = int((s.now - s.last_hand_at).total_seconds())
        s.last_hand_at = s.now
        moments.hand(self, event, since)
        self._arouse(1500 if speed == "fast" else 300)
        if speed == "fast":
            self._express(
                "perk_tilt", reason="startle", parents=(event.event_id,), force_sound=("inquire",)
            )
            note = "嚇一跳（快速靠近的驚嚇只會減弱，不會完全消失）"
        else:
            s.slow_hands += 1
            if s.slow_hands < 8:
                self._express("tuck_away", reason="shy", parents=(event.event_id,), force_sound=())
                note = f"還有點怕，縮一下（馴服進度 {s.slow_hands}/8）"
            else:
                self._express(
                    "snuggle" if "snuggle" in s.unlocked else "perk_tilt",
                    reason="lean_in",
                    parents=(event.event_id,),
                    force_sound=("coo",),
                )
                note = "靠過來了！"
                if s.slow_hands == 8:
                    self._experience_milestone("tamed_hand")
        self._spend(2)
        return {"perceived": True, "note": note}

    @_input
    def call_name(self, persona: str) -> dict[str, Any]:
        """Someone says the duck's name. Only the fact that it was heard is kept, never audio."""

        s = self.s
        if not self._needs_awake(persona, "call_name"):
            return {"perceived": False, "note": "關機中"}
        if not self.hw.name_ear:
            self._journal(persona, "call_name", "叫牠的名字 → 這隻鴨沒有認名字的耳朵", False, [])
            self._spend(2)
            return {"perceived": False, "note": "沒有認名字的耳朵"}
        tv = s.tv_on
        event = self._emit("sense.name_heard", {"tv_on": tv}, source="ear")
        self._journal(persona, "call_name", f"叫「{s.name}」", True, [event])
        self._presence("name", event)
        self._contact("name")
        self._arouse(1000)
        s.name_calls += 1
        # With the TV on it could have been the TV, so it does not count toward learning.
        if not tv and s.gd not in s.name_gds:
            s.name_gds.append(s.gd)
        knows = "learned_name" in s.milestones
        if not knows and len(s.name_gds) >= NAME_LEARN_DAYS:
            self._experience_milestone("learned_name")
            knows = True
        self._care("talk", event)
        if self._until(s.napping_until):
            # Its name reaches it even asleep: in a daytime nap it may wake up; otherwise it
            # opens its eyes for a moment and sleeps on.
            if self._nap_wakes("name"):
                self._end_nap("name", parents=(event.event_id,))
                note = "聽到名字醒來了，伸個懶腰"
            else:
                self._sleep_stir("name", "peek", parents=(event.event_id,))
                note = "睡著了，聽到名字睜眼看一下，又睡回去"
        elif s.yielding or self._until(s.stop_until) or self._until(s.lockout_until) or s.hot:
            note = "聽到了，但身體正忙"
        elif knows and not tv and moments.name_called(self, event.event_id):
            note = "知道是在叫牠：站起來，走去你們常待的地方找你"
        else:
            quiet = self._until(s.quiet_until)
            called = knows and not tv
            stood = None
            if called:
                stood = self._reflex(
                    "stand_up", "name_stand", self._stand_reflex_chance(7000), (event.event_id,)
                )
                verb, sounds, note = "perk_tilt", ("greet",), "知道是在叫牠，抬頭回應"
                if stood is not None:
                    note = "知道是在叫牠，站起來抬頭回應"
            else:
                verb, sounds, note = "look_around", ("inquire",), "好像有人叫我？張望一下"
            self._express(
                verb, reason="name", parents=(event.event_id,), force_sound=() if quiet else sounds
            )
            if called and s.floor and s.posture == "stand":
                turned = self._reflex(
                    "turn_around",
                    "name_turn",
                    3500 * self._style("turn_around", self._leaning()) // SCALE,
                    (event.event_id,),
                )
                if turned is not None:
                    note += "，還轉身找聲音"
        self._spend(2)
        return {"perceived": True, "note": note}

    @_input
    def say(self, persona: str | None, *, kind: str = "voice") -> dict[str, Any]:
        """Voice and noise stay in RAM; only daily counts ever reach the log."""

        s = self.s
        if not self._needs_awake(persona, "say"):
            return {"perceived": False, "note": "關機中"}
        if not self.hw.ear:
            self._journal(persona, "say", "說話 → 這隻鴨沒有開耳朵，聽不到", False, [])
            self._spend(3)
            return {"perceived": False, "note": "沒有開耳朵，聽不到"}
        if kind == "voice":
            s.day_voice += 1
            self._arouse(300)
            # Someone talking to it is someone here (and an answer to a bid it made).
            self._presence("voice")
            self._contact("voice")
        else:
            s.day_noise += 1
            self._arouse(1500)
        self._journal(
            persona,
            "say",
            "說話（只留在記憶體，日誌只記每天的次數）" if kind == "voice" else "很大的聲音",
            True,
            [],
        )
        note = "聽到了"
        cooled = s.last_voice_reaction is None or s.now - s.last_voice_reaction >= timedelta(
            seconds=4
        )
        if cooled and self._until(s.napping_until):
            s.last_voice_reaction = s.now
            if kind == "noise" and self._nap_wakes("noise"):
                self._end_nap("noise")
                note = "被聲音吵醒了，伸個懶腰"
            else:
                self._sleep_stir("sound", "peek" if kind == "noise" else "shiver")
                note = "睡夢中動了一下，又睡著了"
        elif cooled and not s.yielding:
            s.last_voice_reaction = s.now
            startle_chance = max(1000, 6000 - s.noise_habit // 2)
            if kind == "noise" and rng.below(self._rand("noise.startle"), startle_chance):
                tuck = s.overload > 5000
                alert = None
                if not tuck and s.posture == "sit":
                    # A curious duck stands up to see what it was; a shy one tucks instead.
                    alert = self._reflex(
                        "stand_up", "noise_alert", self._stand_reflex_chance(6000, "alert"), ()
                    )
                self._express(
                    "tuck_away" if tuck else "perk_tilt",
                    reason="ambient_sound",
                    force_sound=(),
                )
                note = "被聲音嚇一下（會慢慢習慣）"
                if alert is not None:
                    note = "被聲音嚇一下，站起來警覺地看看（會慢慢習慣）"
            else:
                reply: tuple[str, ...] = ()
                if "call_response" in s.unlocked and not self._until(s.quiet_until):
                    reply = ("inquire",) if rng.below(self._rand("reply"), 5000) else ("chirp",)
                self._express("perk_tilt", reason="ambient_sound", force_sound=reply)
                note = "抬頭看聲音" + ("，回你一聲" if reply else "")
        self._spend(3)
        return {"perceived": True, "note": note}

    @_input
    def tv(self, on: bool) -> dict[str, Any]:
        self.s.tv_on = on
        event = self._emit("env.settings", {"tv_on": on}, source="sim")
        self._journal(None, "tv", "開電視" if on else "關電視", True, [event])
        return {
            "note": "電視開著時，小鴨偶爾會把電視聲誤判成搔頭（約每清醒小時 0.2 次）"
            if on
            else "電視關了"
        }

    @_input
    def pad_good(self, persona: str) -> dict[str, Any]:
        if not self._needs_awake(persona, "pad_good"):
            return {"perceived": False, "note": "關機中"}
        event = self._emit("pad.good", {"button": "dpad_left"}, source="pad")
        self._journal(persona, "pad_good", "按手把十字鍵左（乖）", True, [event])
        self._presence("praise", event)
        if self._asleep_notices("praise", (event.event_id,)):
            self._spend(2)
            return {"perceived": True, "note": "睡著了：睜眼看一下，又睡回去"}
        note = self._praise("pad", persona, event)
        self._spend(2)
        return {"perceived": True, "note": note}

    @_input
    def pad_guest(self, persona: str) -> dict[str, Any]:
        s = self.s
        s.guest_until = s.now + timedelta(hours=4)
        event = self._emit("pad.guest_mode", {"until": s.guest_until.isoformat()}, source="pad")
        self._journal(persona, "pad_guest", "長按十字鍵左 3 秒：訪客模式 4 小時", True, [event])
        return {"note": "訪客模式：4 小時內家庭計數與學習暫停"}

    @_input
    def pad_play(self, persona: str, *, minutes: int = 10) -> dict[str, Any]:
        s = self.s
        if not self._needs_awake(persona, "pad_play"):
            return {"perceived": False, "note": "關機中"}
        s.napping_until = s.nap_reason = s.stir_at = None
        s.sleep_minute = False
        start = self._emit("pad.play_start", {}, source="pad")
        self._journal(
            persona,
            "pad_play",
            f"用手把帶牠玩 {minutes} 分鐘（Duck Core 讓位，只在旁邊看）",
            True,
            [start],
        )
        # An invitation to play: the gamepad stands it up; afterwards it decides for itself
        # when to sit down again.
        if self.hw.nest:
            nest.undock(self, "gamepad", (start.event_id,))
            s.homing = None
        self._set_posture("stand", reason="gamepad", by="gamepad", parents=(start.event_id,))
        s.yielding = True
        meters = 0
        skills: list[str] = []
        for minute in range(max(1, minutes)):
            self.advance_seconds(60)
            if not s.awake:
                break
            self._presence()
            self._contact("play_min")
            # How far the gamepad walked it this minute and the skill it played are the
            # body's to say (a body.play percept, taken in as the poll's answer).
            answer = self.acts.poll(worldport.POLL_PLAY_MINUTE, minute=minute)
            play = next((p for p in answer if p.get("kind") == "body.play"), None)
            walked = int(play["meters"]) if play is not None else 0
            meters += walked
            s.energy = max(0, s.energy - 300)
            s.valence = min(SCALE, s.valence + 150)
            if play is not None and play.get("skill") is not None:
                skills.append(str(play["skill"]))
        s.yielding = False
        self._arouse(1000)
        if s.posture == "stand":
            s.stand_idle_from = s.now  # the gamepad let go: it stands still from here
        end = self._emit(
            "pad.play_end",
            {"minutes": minutes, "meters": meters, "skills": skills},
            source="pad",
            parents=(start.event_id,),
        )
        s.play_meters += meters
        s.play_accepted_gd = s.gd
        s.last_play_at = s.now
        excursion.good_here(self, 1000)
        if self.hw.nest:
            s.play_cell = [excursion.cell_of(s.pos_x, s.pos_y), s.now.date().isoformat()]
        # Fair care: play counts once for every few minutes of it (spec 7.3).
        per = int(content.CARE_LIMITS["pad_play_min"])
        for _ in range(max(1, minutes // per)):
            self._care("play", end, why="pad_play")
        if s.awake:
            self._express(
                "stretch_yawn", reason="happy_tired", parents=(end.event_id,), force_sound=("coo",)
            )
        for mark in (100, 500, 1000, 5000):
            if s.play_meters >= mark and mark not in s.keepsakes:
                s.keepsakes.append(mark)
                self._emit(
                    "keepsake",
                    {"meters": mark, "text": f"紀念：我們一起走了 {mark} 公尺！"},
                    parents=(end.event_id,),
                )
        return {"perceived": True, "note": f"玩了 {minutes} 分鐘、走了約 {meters} 公尺，開心的累"}

    # Owner channel (phone / CLI): the only owner-attributed inputs.

    @_input
    def owner_hello(self) -> dict[str, Any]:
        s = self.s
        if not self._needs_awake("owner", "owner_hello"):
            return {"perceived": False, "note": "關機中"}
        absence_h = None
        if s.last_owner_hello is not None and s.ntp_ok:
            absence_h = int((s.now - s.last_owner_hello).total_seconds() // 3600)
        event = self._emit("owner.hello", {"absence_h": absence_h}, source="owner")
        self._journal("owner", "owner_hello", "主人手機按「你好」", True, [event])
        self._presence("voice", event)
        if self._asleep_notices("hello", (event.event_id,)):
            self._spend(3)
            return {
                "perceived": True,
                "note": "睡著了：聽到你好睜眼看一下，又睡回去（醒著時再打一次招呼）",
            }
        first_today = not s.owner_seen_today
        s.last_owner_hello = s.now
        s.owner_seen_today = True
        self._contact("hello")
        tier, sounds = self._greeting_sounds(absence_h)
        self._arouse(1000 + 500 * tier)
        word = content.WORDS.get(s.first_word or "", ())
        favourite = s.phrases[s.promotions[0]]["tags"] if s.promotions else None
        owner_sounds = tuple(favourite) if favourite else (word or sounds)
        text = f"主人專屬招呼（{content.phrase_text(tuple(owner_sounds))}），熱情等級 {tier}"
        greeting = self._emit(
            "duck.greeting",
            {
                "kind": "owner",
                "tier": tier,
                "sounds": list(owner_sounds),
                "text": text,
                "warmth": s.bond_owner,
            },
            parents=(event.event_id,),
        )
        stood = None
        if tier >= 1 or first_today:
            # After a while apart it stands up to greet (a hatchling mostly just perks up).
            stood = self._reflex(
                "stand_up", "greet_stand", self._stand_reflex_chance(12000), (greeting.event_id,)
            )
        self._express(
            "happy_sway" if tier >= 2 else "perk_tilt",
            reason="owner_greeting",
            parents=(greeting.event_id,),
            force_sound=tuple(owner_sounds),
        )
        self._learn_owner_bond(event)
        self._care("talk", event)
        self._spend(3)
        return {"perceived": True, "note": ("站起來迎接，" if stood is not None else "") + text}

    @_input
    def owner_good(self) -> dict[str, Any]:
        if not self._needs_awake("owner", "owner_good"):
            return {"perceived": False, "note": "關機中"}
        event = self._emit("owner.good", {}, source="owner")
        self._journal("owner", "owner_good", "主人手機按「乖」", True, [event])
        self._presence("praise", event)
        if self._asleep_notices("praise", (event.event_id,)):
            self._spend(2)
            return {"perceived": True, "note": "睡著了：睜眼看一下，又睡回去"}
        self.s.owner_seen_today = True
        note = self._praise("owner", "owner", event)
        self._learn_owner_bond(event)
        self._spend(2)
        return {"perceived": True, "note": note}

    @_input
    def owner_walk(self) -> dict[str, Any]:
        """Owner phone 「去走走」: an invitation to get up and walk about now. The duck decides
        in a drawn, logged reflex like any other (a rested, curious duck nearly always goes,
        a tired one may stay put), and every hard gate still holds."""
        s = self.s
        if not self._needs_awake("owner", "owner_walk"):
            return {"perceived": False, "note": "關機中"}
        event = self._emit("owner.walk", {}, source="owner")
        self._journal("owner", "owner_walk", "主人手機按「去走走」", True, [event])
        self._presence()
        self._arouse(800)
        parents = (event.event_id,)
        if self._asleep_notices("hello", parents):
            self._spend(2)
            return {"perceived": True, "note": "睡著了：睜眼看一下，又睡回去"}
        blocks = self.walk_blocks()
        stuck = {"righting", "fallen", "hot", "yield", "lockout", "stop", "nap", "quiet"}
        if s.posture == "walk":
            note = "牠正在走了"
        elif s.posture in ("sit", "stand") and not stuck & set(blocks):
            leaning = self._leaning()
            standing = s.posture == "stand"
            if standing and self.hw.nest:
                # Already up: an outing, with every rule of the nest (or it stays put).
                went = excursion.can_start(self) is None and excursion.set_out(
                    self, "play", parents=parents
                )
            elif standing:
                self._set_off(leaning)  # already up: off it goes (or it looks about)
                went = True
            else:
                went = self._perk_up("invited", parents)
            up = "" if standing else "站起來"
            if went and not blocks:
                note = f"{up}去走走了"
            elif went:
                note = f"{up}看看四周；" + self.walk_reason()
            elif (
                s.posture == "sit"
                and "stand_up" in s.unlocked
                and not self._motion_block("stand_up")
            ):
                self._express("perk_tilt", reason="invited", parents=parents, force_sound=())
                note = "抬頭看你一眼，還想坐著休息一下（有點累；等一下會自己起來走走）"
                if self.hw.nest and excursion.can_start(self) not in (None, "tired"):
                    note = "抬頭看你一眼；" + self.walk_reason()
            else:
                note = self.walk_reason()
        else:
            note = self.walk_reason()
        self._spend(2)
        return {"perceived": True, "note": note}

    @_input
    def owner_enough(self) -> dict[str, Any]:
        s = self.s
        if not self._needs_awake("owner", "owner_enough"):
            return {"perceived": False, "note": "關機中"}
        event = self._emit("owner.enough", {}, source="owner")
        self._journal("owner", "owner_enough", "主人手機按「夠了」", True, [event])
        s.quiet_until = s.now + timedelta(minutes=15)
        self._emit(
            "duck.quiet",
            {"until": s.quiet_until.isoformat(), "learns": False},
            parents=(event.event_id,),
        )
        if "nod" in s.unlocked and not self._until(s.napping_until):
            self._express(
                "nod", reason="acknowledge_quiet", parents=(event.event_id,), force_sound=()
            )
        self._spend(2)
        return {"perceived": True, "note": "安靜 15 分鐘；不會因此學到「不喜歡」"}

    @_input
    def owner_not_that(self) -> dict[str, Any]:
        s = self.s
        if not self._needs_awake("owner", "owner_not_that"):
            return {"perceived": False, "note": "關機中"}
        event = self._emit("owner.not_that", {}, source="owner")
        self._journal("owner", "owner_not_that", "主人手機按「不是那個」", True, [event])
        flag = self._flag_within(NOT_THAT_WINDOW)
        note = "中性地低一下頭（不發難過的聲音）"
        if flag is not None:
            item = f"{flag['kind']}:{flag['id']}"
            if item not in s.not_that_gd:
                s.not_that_gd.add(item)
                self._learn_pref(item, -1, event)
                if flag["kind"] == "verb":
                    name = content.VERB_BY_ID[str(flag["id"])].zh
                elif flag["kind"] == "moment":
                    name = content.MOMENT_BY_ID[str(flag["id"])].zh.rstrip("。")
                else:
                    name = s.phrases[flag["id"]]["text"]
                note += f"；以後少做一點「{name}」"
        self._spend(2)
        return {"perceived": True, "note": note}

    @_input
    def end_quiet(self) -> dict[str, Any]:
        """Owner phone 「好了，可以動了」: ends quiet time, a lockout and a hand's stop."""
        s = self.s
        if not self._needs_awake("owner", "end_quiet"):
            return {"perceived": False, "note": "關機中"}
        ended = [
            name
            for name, until in (
                ("quiet", s.quiet_until),
                ("lockout", s.lockout_until),
                ("stop", s.stop_until),
            )
            if self._until(until)
        ]
        if s.yielding:
            ended.append("gamepad")
        event = self._emit("owner.release", {"ended": ended}, source="owner")
        self._journal("owner", "end_quiet", "主人手機按「好了，可以動了」", True, [event])
        s.quiet_until = s.lockout_until = s.stop_until = None
        s.stop_moments = []
        s.yielding = False
        self._presence()
        if self._until(s.napping_until):
            self._sleep_stir("released", "peek", parents=(event.event_id,))
        else:
            self._express(
                "perk_tilt", reason="released", parents=(event.event_id,), force_sound=("chirp",)
            )
        self._spend(2)
        if not ended:
            return {"perceived": True, "note": "牠本來就可以動，抬頭啾一聲"}
        self._perk_up("released", (event.event_id,))
        return {"perceived": True, "note": "好了，可以動了：抬頭啾一聲，照常生活"}

    # The day at home (spec v4): what the house sounds like, the family app, the nest.

    @_input
    def hear_home(
        self, persona: str | None = None, *, kind: str = "voice", journal: bool = True
    ) -> dict[str, Any]:
        """A sound or sight of the family in the house (duck-ear, the head camera, a
        consenting phone). Only the fact is kept, as counts; never audio or images."""
        s = self.s
        if kind not in daycycle.CUE_KINDS:
            raise ValueError(f"unknown cue {kind}")
        if not s.awake:
            if journal:
                self._journal(persona, "hear_home", "小鴨關機中，沒有聽到", False, [])
            return {"perceived": False, "note": "關機中"}
        if kind in ("voice", "steps", "door", "tv_voice") and not self.hw.ear:
            return {"perceived": False, "note": "沒有開耳朵"}
        if kind == "person" and not self.hw.camera:
            return {"perceived": False, "note": "沒有相機"}
        if self.aware:
            daycycle.hear(self, kind, room=kind == "person", persona=persona)
        if journal:
            self._journal(
                persona,
                "hear_home",
                f"家裡的聲音：{daycycle.CUE_ZH[kind]}（只記次數）",
                True,
                [],
            )
        return {"perceived": True, "note": f"聽到{daycycle.CUE_ZH[kind]}（只記次數）"}

    @_input
    def family_leave(self) -> dict[str, Any]:
        """The family app's 「出門」: everyone is going out (authoritative)."""
        if not self._needs_awake("owner", "family_leave"):
            return {"perceived": False, "note": "關機中"}
        event = self._emit("owner.leave", {}, source="owner")
        self._journal("owner", "family_leave", "主人手機按「出門」", True, [event])
        if self.aware:
            daycycle.owner_button(self, "away", (event.event_id,))
        return {"perceived": True, "note": "知道大家出門了：牠會回窩休息，等你們回來"}

    @_input
    def family_return(self) -> dict[str, Any]:
        """The family app's 「到家」: someone is home again (authoritative)."""
        if not self._needs_awake("owner", "family_return"):
            return {"perceived": False, "note": "關機中"}
        event = self._emit("owner.return", {}, source="owner")
        self._journal("owner", "family_return", "主人手機按「到家」", True, [event])
        self._presence()
        if self.aware:
            daycycle.owner_button(self, "awake", (event.event_id,))
        return {"perceived": True, "note": "知道你們回來了！"}

    @_input
    def goodnight(self, persona: str | None = None) -> dict[str, Any]:
        """The family's 「晚安」 (from 21:00 it starts the wind-down early)."""
        s = self.s
        if not self._needs_awake(persona, "goodnight"):
            return {"perceived": False, "note": "關機中"}
        event = self._emit("owner.goodnight", {}, source="owner")
        self._journal(persona, "goodnight", "跟牠說晚安", True, [event])
        self._presence()
        early = daycycle.minute_of_day(s.now) >= int(content.RHYTHM["goodnight_after_min"])
        if self.aware and early:
            s.goodnight_at = s.now
            return {"perceived": True, "note": "晚安！牠會自己走回窩、坐好睡覺"}
        return {"perceived": True, "note": "晚安（晚上 9 點以後說，牠才會早點回窩睡覺）"}

    @_input
    def place_on_nest(self, persona: str | None = None) -> dict[str, Any]:
        """A hand puts the duck into its nest, onto the contacts (it charges at once)."""
        s = self.s
        persona = persona or "owner"
        if not self.hw.nest:
            self._journal(persona, "place_on_nest", "放回窩裡 → 這隻鴨沒有窩", False, [])
            return {"perceived": False, "note": "這隻鴨沒有窩（插電或換電池）"}
        event = self._emit("sense.placed", {"on": "nest", "path": "contacts"}, source="body")
        self._journal(persona, "place_on_nest", "把小鴨放回窩裡", True, [event])
        if s.awake:
            self._presence()
            self._set_posture("sit", reason="placed", by="hand", parents=(event.event_id,))
        elif s.posture != "sit":
            s.posture = "sit"
        s.homing = None
        s.need = None
        s.floor = True
        nest.docked(self, "hand")
        note = "放回窩裡了：坐在接觸片上充電"
        if not s.awake:
            note += "（關機中也會充電）"
        return {"perceived": True, "note": note}

    @_input
    def ack_nest_trouble(self) -> dict[str, Any]:
        """The owner checked the floor by the nest (or moved it): back to normal."""
        s = self.s
        event = self._emit("owner.ack_nest_trouble", {"was": s.nest_trouble}, source="owner")
        self._journal("owner", "ack_nest_trouble", "主人確認窩附近的地面沒問題了", True, [event])
        s.nest_trouble = False
        return {"note": "好，牠會照常自己進出窩"}

    @_input
    def set_roam_radius(self, mm: int = 1500) -> dict[str, Any]:
        """The family app's roam radius (1.0–2.75 m; 1.5 m by default)."""
        s = self.s
        s.roam_radius_mm = clamp(int(mm), 1000, 2750)
        event = self._emit("owner.roam", {"radius_mm": s.roam_radius_mm}, source="owner")
        self._journal(
            "owner",
            "set_roam_radius",
            f"活動範圍 {s.roam_radius_mm / 1000:.2f} 公尺",
            True,
            [event],
        )
        return {"note": f"牠會在窩旁 {s.roam_radius_mm / 1000:.2f} 公尺內走走"}

    @_input
    def nest_moved(self) -> dict[str, Any]:
        """The family moved the nest (a new home): the map starts afresh, the trouble too."""
        s = self.s
        event = self._emit("owner.nest_moved", {"cells": len(s.cells)}, source="owner")
        self._journal("owner", "nest_moved", "把窩搬到新的地方", True, [event])
        s.cells = {}
        s.nest_trouble = False
        s.pose_err_mm = 0 if s.nested else max(s.pose_err_mm, 300)
        return {"note": "窩換了位置：牠會重新認識周圍"}

    # Simulator hands: waking it, setting it upright, a fresh battery.

    @_input
    def wake(self, persona: str) -> dict[str, Any]:
        """Someone gently wakes the dozing duck; it stays awake for a while."""
        s = self.s
        if not self._needs_awake(persona, "wake"):
            return {"perceived": False, "note": "關機中（要先開機）"}
        if not self._until(s.napping_until):
            self._journal(persona, "wake", "叫醒牠 → 牠本來就醒著", True, [])
            return {"perceived": True, "note": "牠本來就醒著"}
        event = self._emit("sense.nudge", {"path": "joint"}, source="joint")
        self._journal(persona, "wake", "輕輕叫醒牠", True, [event])
        self._presence()
        night = s.nap_reason == "quiet_hours"
        self._end_nap("family", parents=(event.event_id,))
        self._spend(2)
        self._perk_up("woken", (event.event_id,))
        note = f"醒了，伸個懶腰；{NAP_WAKE_BLOCK_MIN} 分鐘內不會再睡"
        if night and s.docked:
            note += "（晚上插著電，沒人陪時會再睡回去）"
        elif self._night():
            note += "（晚上只會坐著動頭，早上 7 點以後才會站起來走走）"
        return {"perceived": True, "note": note}

    @_input
    def nap_now(self, persona: str) -> dict[str, Any]:
        """Someone settles the tired duck down for a nap now (instead of waiting until
        nobody is around); it sleeps until rested, as a tired nap always does."""
        s = self.s
        if not self._needs_awake(persona, "nap_now"):
            return {"perceived": False, "note": "關機中"}
        if self._until(s.napping_until):
            return {"perceived": True, "note": "牠已經在睡了"}
        if s.yielding or s.posture == "fallen":
            return {"perceived": True, "note": "現在沒辦法睡（先扶起來或結束手把）"}
        event = self._emit("sense.nudge", {"path": "joint", "why": "nap"}, source="joint")
        self._journal(persona, "nap_now", "輕輕拍拍牠，讓牠小睡一下", True, [event])
        self._start_nap(False)
        return {
            "perceived": True,
            "note": f"閉上眼睛小睡，大約 {self.nap_left_min()} 分鐘後睡飽醒來",
        }

    @_input
    def set_upright(self, persona: str) -> dict[str, Any]:
        """A hand sets a fallen duck back on its belly, where it lay (floor mode stays)."""
        s = self.s
        if not self._needs_awake(persona, "set_upright"):
            return {"perceived": False, "note": "關機中"}
        if s.posture != "fallen":
            self._journal(persona, "set_upright", "扶起來 → 牠沒有跌倒", True, [])
            return {"perceived": True, "note": "牠沒有跌倒"}
        event = self._emit("sense.set_upright", {"path": "imu"}, source="imu")
        self._journal(persona, "set_upright", "把跌倒的小鴨扶起來坐好（還在地上）", True, [event])
        self._presence()
        s.hand_placed = True  # (the battery it lay there with was not its own walking)
        if s.need is not None and s.need.get("kind") == "help":
            s.need = None
        self._set_posture("sit", reason="set_upright", by="hand", parents=(event.event_id,))
        self._spend(2)
        return {"perceived": True, "note": "扶起來坐好了，還在地上，等一下會再走走"}

    @_input
    def swap_battery(self, persona: str) -> dict[str, Any]:
        """A fresh pack (the simulator skips the power cycle a real swap needs)."""
        s = self.s
        if s.docked:
            self._journal(persona, "swap_battery", "換電池 → 插著電線，不用換", True, [])
            return {"note": "插著電線，不用換電池"}
        if not s.awake:
            if self.hw.nest:
                # (Without a nest switching on puts in a fresh pack anyway; with one the
                # pack carries on, so the fresh one goes in first.)
                event = self._swap_battery("hand")
                self._journal(persona, "swap_battery", "換上新電池（100%）", True, [event])
            self.power_on(persona)
            return {"note": "換上新電池、開機"}
        event = self._swap_battery("hand")
        self._journal(persona, "swap_battery", "換上新電池（100%）", True, [event])
        if self._until(s.napping_until) and s.nap_reason == "bedtime":
            self._end_nap("battery", parents=(event.event_id,))
        self._perk_up("fresh_battery", (event.event_id,))
        return {"note": "換上新電池（100%）"}

    # ------------------------------------------------------------------ power and visits

    def _greeting_sounds(self, absence_h: int | None) -> tuple[int, tuple[str, ...]]:
        """Frozen reunion table: joy only, capped, and never learnable."""

        if absence_h is None or absence_h < 2:
            return 0, ("chirp",)
        if absence_h < 24:
            return 1, ("greet", "chirp")
        if absence_h < 24 * 7:
            return 2, ("greet", "chirp", "wheee")
        return 3, ("greet", "chirp", "chirp", "wheee")

    @_input
    def power_on(self, persona: str | None) -> dict[str, Any]:
        s = self.s
        if s.awake:
            return {"note": "已經開機了"}
        if (
            self.hw.nest
            and not (s.nested or s.docked)
            and s.battery <= EMPTY_BATTERY + BATTERY_DRAIN_PER_MIN
        ):
            # With a nest the pack is its own and only the nest (or the cable) charges it:
            # switched on with less than a minute left it would only switch off again at
            # once, over and over. It stays off until someone puts it back in its nest.
            self._journal(persona, "power_on", "開機 → 電池沒電，開不了機", False, [])
            return {
                "perceived": False,
                "note": "電池沒電，開不了機：先放回窩裡充電（關機中也會充電）",
            }
        s.boot_seq += 1
        s.awake = True
        s.boot_at = s.now
        s.trace = f"visit-{s.boot_seq:04d}"
        s.visit_events = []
        if not self.hw.nest:
            s.battery = SCALE  # a fresh pack; with a nest the battery simply carries on
        s.bedtime_done = False
        s.energy = max(s.energy, 7500)
        s.come_play_this_visit = False
        s.bids_stopped = False
        s.unanswered = 0
        s.bid_backoff = SCALE
        absence_h = None
        if s.last_power_off is not None and s.ntp_ok:
            absence_h = int((s.now - s.last_power_off).total_seconds() // 3600)
        event = self._emit(
            "power.on",
            {"boot_seq": s.boot_seq, "docked": s.docked, "absence_h": absence_h},
            source="body",
        )
        self._journal(
            persona, "power_on", "放電池、開機（robotd 的開機招呼已在設定時關閉）", True, [event]
        )
        self._presence()
        self._emit(
            "duck.reaction",
            {"to": "boot", "sounds": ["chirp"], "text": "身體還軟軟的，迷糊地啾一聲"},
            parents=(event.event_id,),
        )
        if s.boot_seq > 1:
            tier, sounds = self._greeting_sounds(absence_h)
            self._emit(
                "duck.greeting",
                {
                    "kind": "reunion",
                    "tier": tier,
                    "sounds": list(sounds),
                    "text": f"重逢招呼（{content.phrase_text(sounds)}）",
                },
                parents=(event.event_id,),
            )
            if s.milestones["morning_habit"]["status"] == "debuted":
                self._express(
                    "wake_ritual",
                    reason="wake",
                    parents=(event.event_id,),
                    force_sound=("coo", "chirp"),
                )
            if s.last_visit.get("favorite"):
                self._emit(
                    "duck.reaction",
                    {"to": "callback", "text": f"想起上次：{s.last_visit['favorite']}"},
                    parents=(event.event_id,),
                )
            self._debut_pending()
        return {"note": "開機", "absence_h": absence_h}

    def _bedtime(self) -> None:
        s = self.s
        s.bedtime_done = True
        self._emit(
            "duck.bedtime",
            {
                "battery": s.battery,
                "sounds": ["coo", "coo"],
                "text": "電量約 15%：節奏變慢、晚安咕～咕～",
            },
        )
        self._start_nap(False, bedtime=True)

    @_input
    def power_off(self, persona: str | None, *, reason: str = "owner") -> dict[str, Any]:
        """Someone switches it off (the owner, the family): an input."""
        if not self.s.awake:
            return {"note": "已經關機了"}
        awake_min, visit_ids, summary = self._shutdown_begin(reason)
        off = self._emit(
            "power.off",
            {"reason": reason, "awake_min": awake_min, "battery": self.s.battery},
            source="body",
        )
        return self._shutdown_end(off, persona, reason, visit_ids, summary)

    def _power_down(self, reason: str = "battery_empty") -> dict[str, Any]:
        """Its battery is flat: it sits down and switches itself off (its own decision, not
        an input: signed self)."""
        if not self.s.awake:
            return {"note": "已經關機了"}
        awake_min, visit_ids, summary = self._shutdown_begin(reason)
        off = self._emit(
            "power.off", {"reason": reason, "awake_min": awake_min, "battery": self.s.battery}
        )
        return self._shutdown_end(off, None, reason, visit_ids, summary)

    def _shutdown_begin(self, reason: str) -> tuple[int, list[str], dict[str, Any]]:
        s = self.s
        if not s.bedtime_done and reason == "owner":
            self._emit(
                "duck.bedtime",
                {"battery": s.battery, "sounds": ["coo", "coo"], "text": "晚安咕～咕～"},
            )
        moments.forget_guesses(self, "power_off")
        # robotd sits down to power off (a fallen duck gets set upright to change the battery),
        # and floor mode always ends: it wakes up again in the safe desk mode.
        self._set_posture("sit", reason="power_off", by="body")
        s.floor_when_off = s.floor
        s.homing = None
        if self.hw.nest and s.nested:
            pass  # asleep in its nest: it stays there (and charges)
        elif s.floor:
            self._set_floor(False, "power_off")
            if self.hw.nest:
                s.unplaced = True
        awake_min = int((s.now - s.boot_at).total_seconds() // 60) if s.boot_at else 0
        visit_ids = list(s.visit_events)
        return awake_min, visit_ids, self._visit_summary(visit_ids)

    def _shutdown_end(
        self,
        off: Event,
        persona: str | None,
        reason: str,
        visit_ids: list[str],
        summary: dict[str, Any],
    ) -> dict[str, Any]:
        s = self.s
        parents = [visit_ids[0]] if visit_ids else []
        parents += [
            event_id
            for event_id in visit_ids
            if (self.log.get(event_id) or off).kind
            in {"sense.touch", "owner.hello", "pad.play_end", "growth.milestone", "voice.promoted"}
        ][:40]
        self._emit(
            "memory.consolidated",
            {
                "visit": s.trace,
                "summary": summary,
                "rule": "consolidate-v1",
                "sources": len(visit_ids),
            },
            parents=tuple(parents) or (off.event_id,),
        )
        self._journal(
            persona,
            "power_off",
            "關機"
            if reason == "owner"
            else "電池用完，robotd 坐下關機（最後的 peck 是上游的告別）",
            True,
            [off],
        )
        s.last_visit = summary
        s.awake = False
        s.last_power_off = s.now
        s.off_reason = reason
        s.napping_until = s.nap_reason = s.stir_at = s.nap_block_until = None
        s.quiet_until = None
        s.stop_until = None
        s.lockout_until = None
        s.yielding = False
        s.bid_pending_since = None
        s.last_flag = None
        s.trace = "life"
        return {"note": "關機", "summary": summary}

    def _visit_summary(self, ids: list[str]) -> dict[str, Any]:
        counts: dict[str, int] = {}
        verbs: dict[str, int] = {}
        walks = 0
        for event_id in ids:
            event = self.log.get(event_id)
            if event is None:
                continue
            counts[event.kind] = counts.get(event.kind, 0) + 1
            if event.kind == "duck.expression":
                verb = str(event.payload.get("verb"))
                verbs[verb] = verbs.get(verb, 0) + 1
            elif event.kind == "duck.walk" and event.payload.get("verb") == "wander":
                walks += 1
        favorite = None
        if counts.get("pad.play_end"):
            favorite = "一起用手把玩"
        elif counts.get("sense.touch", 0) >= 2:
            favorite = "被搔頭頂"
        elif walks >= 3:
            favorite = "自己到處走走"
        elif verbs:
            top = max(sorted(verbs), key=lambda key: verbs[key])
            favorite = content.VERB_BY_ID[top].zh
        return {
            "touches": counts.get("sense.touch", 0),
            "praise": counts.get("pad.good", 0) + counts.get("owner.good", 0),
            "play": counts.get("pad.play_end", 0),
            "milestones": counts.get("growth.milestone", 0),
            "walks": walks,
            "favorite": favorite,
        }

    # ------------------------------------------------------------------ settings and choices

    @_input
    def set_setting(self, key: str, value: bool) -> dict[str, Any]:
        s = self.s
        if key == "floor":
            return self._floor_setting(bool(value))
        if key in {"pickup_detect", "motion_validated"}:
            return self._hardware_setting(key, bool(value))
        if key not in {"docked", "ntp_ok", "hot", "conservative", "battery_help"}:
            raise ValueError(f"unknown setting {key}")
        setattr(s, key, bool(value))
        event = self._emit("env.settings", {key: bool(value)}, source="sim")
        self._journal(None, "setting", f"{key} = {value}", True, [event])
        if key == "hot" and value and s.awake:
            self._sit_down_now("hot")  # overheating: rest pose right away
        if (key, bool(value)) in {("docked", False), ("hot", False)}:
            self._perk_up("unplugged" if key == "docked" else "cooled", (event.event_id,))
        notes = {
            "docked": "插電常駐：不耗電、慢慢充電，22:00–07:00 安靜時段"
            if value
            else f"改用電池（現在 {s.battery * 100 // SCALE}%，一顆約 1 小時）",
            "battery_help": "家人會在電量低於 30% 時幫牠換電池（模擬器）"
            if value
            else "沒人幫忙換電池：沒電就會關機",
            "ntp_ok": "時間校正正常，成長日會前進"
            if value
            else "沒有校正時間：成長日暫停（照常生活、不懲罰）",
            "hot": "太熱：休息姿勢、只發很少的聲音" if value else "溫度恢復",
            "conservative": "保守模式：幅度 ≤ 60%、節奏放慢、走路短一點也少一點（工作台門檻通過前）"
            if value
            else "工作台門檻通過：解除保守模式",
        }
        return {"note": notes[key]}

    def _hardware_setting(self, key: str, value: bool) -> dict[str, Any]:
        """Simulator only: the simulated robot gains (or loses) its pickup sensor, or passes
        (or not) the physics validation for standing up and walking. A recorded input like
        any setting, so a replay changes the hardware at the same moment."""
        self.hw = replace(self.hw, **{key: value})
        event = self._emit("env.hardware", {key: value}, source="sim")
        self._journal(None, "setting", f"模擬硬體 {key} = {value}", True, [event])
        if value:
            self._perk_up("hardware", (event.event_id,))
        notes = {
            "pickup_detect": "模擬：裝上抱起偵測，在地上可以自己走了"
            if value
            else "模擬：沒有抱起偵測，分不出是不是被抱起來，所以不會自己走",
            "motion_validated": "模擬：當作站立和走路已通過物理驗證，會自己站起來走走"
            if value
            else "模擬：站立和走路還沒通過驗證，不會自己站起來",
        }
        return {"note": notes[key]}

    def _floor_setting(self, on: bool) -> dict[str, Any]:
        """The family says where the duck is: on the floor (it may walk) or on the desk."""
        s = self.s
        event = self._set_floor(on, "family")
        self._journal(None, "floor", "放在地上（地板模式）" if on else "放回桌上", True, [event])
        if on:
            self._perk_up("set_down", (event.event_id,))
        if not on and s.posture == "walk":
            self._set_posture("stand", reason="floor_off", by="family", parents=(event.event_id,))
            self._place_on_mat()  # lifted mid-walk and set on the desk
        if not on:
            return {"note": "放回桌上：還是會自己站起來、坐下，但不會走"}
        blocks = self.walk_blocks()
        if blocks:
            # Whatever still stops it right now (night, asleep, its cable, the battery…).
            note = "放到地上了（地板模式），但現在還不會走：" + self.walk_reason()
            if "night" in blocks and "晚上" not in note:
                note += "而且晚上 10 點到早上 7 點不走動，早上 7 點以後才會自己起來走走。"
            return {"note": note}
        return {
            "note": "地板模式：牠會自己站起來、慢慢走幾步、轉身看看（白天、電量 30% 以上；"
            "不用有人在）。抱起來或關機就回到桌上模式"
        }

    @_input
    def choose(self, kind: str, option: str) -> dict[str, Any]:
        s = self.s
        if kind == "first_word":
            if s.milestones["first_word"]["status"] == "debuted":
                return {"note": "第一個字已經說出來了，不能再換"}
            if option not in content.FIRST_WORDS:
                raise ValueError("first word must be one of " + "/".join(content.FIRST_WORDS))
            s.first_word = option
            event = self._emit(
                "growth.choice", {"kind": "first_word", "option": option}, source="owner"
            )
            self._journal(
                "owner",
                "choose",
                f"選第一個字：{content.phrase_text(content.WORDS[option])}",
                True,
                [event],
            )
            return {"note": f"第一個字會是 {content.phrase_text(content.WORDS[option])}"}
        if kind == "branch":
            if s.branch is None:
                return {"note": "還沒到分支（約第 45 個成長日）"}
            if s.gd >= self._stage_day("adult") or s.molt_decided:
                return {"note": "已經成年或換羽過，分支不能再改"}
            target = option if option in content.BRANCH_BY_ID else s.runner_up
            if target is None or target == s.branch:
                return {"note": "已經是這一型了"}
            old = s.branch
            s.branch, s.runner_up = target, old
            event = self._emit(
                "growth.branch_changed",
                {"from": old, "to": target, "zh": content.BRANCH_BY_ID[target].zh},
                source="owner",
            )
            self._journal(
                "owner", "choose", f"改選分支：{content.BRANCH_BY_ID[target].zh}", True, [event]
            )
            signature = content.BRANCH_BY_ID[target].signatures[0]
            if signature not in s.unlocked:
                s.unlocked.append(signature)
            return {
                "note": f"改成{content.BRANCH_BY_ID[target].zh}（沒有期限壓力；成年或換羽前都能改）"
            }
        if kind == "molt":
            if s.molt is None or s.molt.get("status") != "open":
                return {"note": "現在沒有換羽提議"}
            if option == "change":
                replace, new = s.molt["replace"], s.molt["with"]
                if s.song is not None:
                    s.song["phrases"] = [
                        new if code == replace else code for code in s.song["phrases"]
                    ]
                    s.song["version"] = int(s.song["version"]) + 1
                s.molt["status"] = "changed"
                s.molt_decided = True
                event = self._emit(
                    "voice.molt_decision", {"choice": "change", "by": "family"}, source="pad"
                )
                text = " ／ ".join(
                    s.phrases[code]["text"] for code in (s.song or {}).get("phrases", [])
                )
                self._emit(
                    "voice.song",
                    {
                        "version": (s.song or {}).get("version", 0),
                        "phrases": (s.song or {}).get("phrases", []),
                        "text": text,
                        "gd": s.gd,
                    },
                )
            else:
                s.molt["status"] = "kept"
                event = self._emit(
                    "voice.molt_decision", {"choice": "keep", "by": "family"}, source="pad"
                )
            self._journal(
                "owner",
                "choose",
                f"換羽：{'換' if option == 'change' else '維持現狀'}",
                True,
                [event],
            )
            return {"note": "換羽了！" if option == "change" else "維持現狀"}
        raise ValueError(f"unknown choice {kind}")

    @_input
    def undo_promotion(self) -> dict[str, Any]:
        s = self.s
        if not s.promotions or s.promotion_gd != s.gd:
            return {"note": "今天沒有可以復原的片語"}
        code = s.promotions.pop()
        s.phrases[code]["promoted"] = False
        s.promotion_gd = None
        event = self._emit(
            "voice.promotion_undone",
            {"code": code, "text": s.phrases[code]["text"]},
            source="owner",
        )
        self._journal("owner", "undo", f"復原片語：{s.phrases[code]['text']}", True, [event])
        return {"note": f"已復原「{s.phrases[code]['text']}」"}

    # ------------------------------------------------------------------ forgetting

    @_input
    def forget(self, fact_id: str) -> dict[str, Any]:
        memory = self.memory()
        fact = memory.facts.get(fact_id)
        if fact is None:
            return {"ok": False, "note": "找不到這段記憶"}
        if fact.category in {"trait", "milestone", "quirk"}:
            return {"ok": False, "note": "成長與個性是牠的一部分，不能單獨忘記（可以重置整隻鴨）"}
        evidence = set(fact.evidence)
        # What this memory taught: updates that are the evidence, or were drawn from it.
        related = [
            event
            for event in self.log.events
            if event.kind == "learning.committed"
            and (
                event.event_id in evidence
                or {str(item) for item in seq(event.payload.get("evidence_event_ids"))} & evidence
            )
        ]
        forget = self._emit(
            "memory.forget",
            {
                "target": fact_id,
                "text": fact.text,
                "event_ids": sorted(evidence | {event.event_id for event in related}),
            },
            source="owner",
        )
        taught: dict[str, int] = {}
        for event in related:
            parameter = str(event.payload.get("parameter"))
            before = int(round(num(event.payload.get("before"), 0.5) * SCALE))
            after = int(round(num(event.payload.get("after"), 0.5) * SCALE))
            taught[parameter] = taught.get(parameter, 0) + after - before
        removed: dict[str, int] = {}
        for parameter, delta in sorted(taught.items()):
            current = self._learned_value(parameter)
            # Take back what was taught, but never past neutral: daily relaxation may
            # already have returned part of it.
            neutral = 0 if parameter.startswith("self.avoid_spot:") else 5000
            neutral = {"owner.bond": 2000}.get(parameter, neutral)
            if delta > 0:
                target = max(current - delta, min(current, neutral))
            else:
                target = min(current - delta, max(current, neutral))
            self._set_learned_value(parameter, target)
            removed[parameter] = current - target
        rolled = [
            self._emit(
                "learning.rolled_back",
                {
                    "update_id": event.payload.get("update_id"),
                    "parameter": event.payload.get("parameter"),
                    "reason": "forget",
                },
                parents=(forget.event_id, event.event_id),
            ).event_id
            for event in related
        ]
        self.s.forget_log.append(
            {"forget": forget.event_id, "at": self.s.now, "rolled": rolled, "removed": removed}
        )
        self._journal("owner", "forget", f"忘記：{fact.text}", True, [forget])
        rebuilt = distill(self.log.events, self.s.now, self.s.name)
        again = rebuilt.facts.get(fact_id)
        return {
            "ok": True,
            "receipt": forget.event_id,
            "removed_events": len(evidence),
            "rolled_back": len(rolled),
            "still_there": again is not None
            and again.value == fact.value
            and again.text == fact.text,
            "note": "已忘記；衍生的記憶都重建過，學習也退回了（24 小時內可以復原）",
        }

    def _learned_value(self, parameter: str) -> int:
        if parameter == "owner.bond":
            return self.s.bond_owner
        if parameter.startswith("self.avoid_spot:"):
            spot = self.s.avoid_spots.get(parameter.split(":", 1)[1])
            return SCALE if spot is not None and spot.get("active", True) else 0
        return self.s.prefs.get(parameter, 5000)

    def _set_learned_value(self, parameter: str, value: int) -> None:
        if parameter == "owner.bond":
            self.s.bond_owner = clamp(value)
        elif parameter.startswith("self.avoid_spot:"):
            # Forgetting the falls takes the spot back (and undoing that restores it); the
            # forgotten falls stay counted, so they never teach the same spot again.
            spot = self.s.avoid_spots.get(parameter.split(":", 1)[1])
            if spot is not None:
                spot["active"] = value >= SCALE // 2
        else:
            self.s.prefs[parameter] = clamp(value)

    @_input
    def undo_forget(self) -> dict[str, Any]:
        s = self.s
        if not s.forget_log:
            return {"note": "沒有可以復原的忘記"}
        last = s.forget_log[-1]
        if s.now - last["at"] > timedelta(hours=24):
            return {"note": "超過 24 小時，不能復原了"}
        s.forget_log.pop()
        for parameter, amount in sorted(last["removed"].items()):
            self._set_learned_value(parameter, self._learned_value(parameter) + amount)
        event = self._emit(
            "memory.forget_undone", {"forget_event_id": last["forget"]}, source="owner"
        )
        self._journal("owner", "undo_forget", "復原上一次的忘記", True, [event])
        return {"note": "已復原"}

    # ------------------------------------------------------------------ the clip player's idle layer

    def can_fidget(self) -> bool:
        """Awake, not asleep and not held still: small idle glances may play."""
        s = self.s
        return (
            s.awake
            and s.posture in ("sit", "stand", "walk")
            and not self._until(s.napping_until)
            and not s.yielding
            and not s.hot
            and not self._until(s.stop_until)
            and not self._until(s.lockout_until)
        )

    def fidgets_between(
        self, start: datetime, end: datetime, events: list[Event]
    ) -> list[dict[str, Any]]:
        """The idle layer's glances after ``start`` up to ``end`` (``can_fidget`` first).

        Between its decisions a real duck keeps glancing about. These are the clip player's
        idle layer, like breathing: one in every FIDGET_SLOT_S seconds of the clock, drawn
        from the seed and the slot (never from the decision counters), so they are the same
        every time, whatever the step, and never change the event chain. None plays while
        a move the duck really did is showing (``events``: what it did up to ``end``).
        """
        s = self.s
        busy = [span for event in events if (span := self.move_span(event)) is not None]
        options = FIDGET_CLIPS["sit" if s.posture == "sit" else "stand"]
        tempo = self._tempo()
        out: list[dict[str, Any]] = []

        def draw(purpose: str, slot: int) -> int:
            return rng.draw(self.seed, self.table, f"fidget.{purpose}", slot)

        def clip_of(slot: int) -> int:
            # Never the same glance twice in a row (the same look again reads as a twitch):
            # even slots draw from one half of the glances, odd slots from the other.
            half = len(options) // 2
            index = rng.pick_index(draw("clip", slot), half)
            return index * 2 + slot % 2

        first = int(start.timestamp()) // FIDGET_SLOT_S
        last = int(end.timestamp()) // FIDGET_SLOT_S
        for slot in range(first, last + 1):
            offset = rng.pick_index(draw("at", slot), FIDGET_JITTER_S + 1)
            moment = datetime.fromtimestamp(slot * FIDGET_SLOT_S + offset, tz=s.now.tzinfo)
            if not start < moment <= end:
                continue
            if any(begin <= moment <= until for begin, until in busy):
                continue
            clip = options[clip_of(slot)]
            share = FIDGET_SHARE[0] + rng.pick_index(draw("size", slot), FIDGET_SHARE[1] + 1)
            out.append(
                {
                    "at": moment,
                    "clip": clip,
                    "amplitude": self._amplitude_cap(clip) * share // SCALE,
                    "tempo": tempo,
                }
            )
        return out

    @staticmethod
    def move_span(event: Event) -> tuple[datetime, datetime] | None:
        """While a move the duck really did is showing (with FIDGET_CLEAR_S either side),
        or None for an event nobody sees move."""
        p = event.payload
        if event.kind == "duck.walk":
            seconds = int(num(p.get("seconds"), 0))
        elif event.kind in {"duck.expression", "duck.sleep_stir", "duck.reaction"}:
            frames = content.CLIPS.get(str(p.get("verb") or p.get("clip")))
            if frames is None:
                return None
            tempo = max(5000, int(num(p.get("tempo"), SCALE)) or SCALE)
            seconds = -(-frames[-1][0] * SCALE // tempo // 1000)
        elif event.kind in {"duck.posture", "body.righting", "duck.struggle", "duck.righting_end"}:
            seconds = 2
        elif event.kind == "body.posture" and p.get("got") == "stand":
            seconds = 2  # (it stood up: the body's answer is the record of the move)
        else:
            return None
        clear = timedelta(seconds=FIDGET_CLEAR_S)
        return event.occurred_at - clear, event.occurred_at + timedelta(seconds=seconds) + clear

    # ------------------------------------------------------------------ explanations

    def night(self) -> bool:
        """22:00–07:00 on the duck's clock: no standing up or walking."""
        return self._night()

    def nap_left_min(self) -> int:
        s = self.s
        if not self._until(s.napping_until):
            return 0
        assert s.napping_until is not None
        return max(1, -(-int((s.napping_until - s.now).total_seconds()) // 60))

    def left_min(self, until: datetime | None) -> int:
        """Whole minutes (rounded up) until ``until``, 0 when it has passed."""
        if not self._until(until):
            return 0
        assert until is not None
        return max(1, -(-int((until - self.s.now).total_seconds()) // 60))

    def walk_blocks(self) -> list[str]:
        """Everything that keeps it from walking right now, most pressing first ([] = it
        may walk). The same hard gates as the decisions (``_motion_block``), named."""
        s = self.s
        if not s.awake:
            return ["off"]
        if s.posture == "fallen":
            return ["righting" if s.righting is not None else "fallen"]
        blocks = [
            name
            for name, on in (
                ("hot", s.hot),
                ("yield", s.yielding),
                ("lockout", self._until(s.lockout_until)),
                ("stop", self._until(s.stop_until)),
                ("nap", self._until(s.napping_until)),
                ("quiet", self._until(s.quiet_until)),
                ("night", self._night()),
                ("battery", not s.docked and s.battery < MIN_MOTION_BATTERY),
                ("desk", not s.floor),
                ("tethered", s.floor and s.docked),
                ("no_pickup_sensor", not self.hw.pickup_detect),
                ("unvalidated", not self.hw.motion_validated),
                ("too_young", "wander" not in s.unlocked),
            )
            if on
        ]
        if self.hw.nest:
            if s.unplaced and "desk" in blocks:
                blocks[blocks.index("desk")] = "unplaced"
            blocks += [
                name
                for name, on in (
                    ("stranded", s.stranded is not None),
                    ("stand_backoff", self._until(s.stand_block_until)),
                    ("dawn", s.phase == "dawn"),
                    ("bedtime", s.phase == "wind_down"),
                    ("alone", s.phase == "alone"),
                    ("solo_done", excursion.can_start(self) == "solo_done"),
                    ("family_quiet", s.phase == "family" and excursion.can_start(self) == "quiet"),
                    ("door", excursion.can_start(self) == "door"),
                    ("nest_trouble", excursion.can_start(self) == "nest_trouble"),
                    (
                        "charging",
                        s.nested and s.battery <= int(content.NEST["leave_min"]),
                    ),
                    ("going_home", not s.nested and s.battery <= nest.threshold(self)),
                )
                if on and name not in blocks
            ]
        if not blocks and self._tiredness() == 0:
            blocks.append("tired")
        return blocks

    def still_reason(self) -> tuple[str, str]:
        s = self.s
        if not s.awake:
            if s.off_reason == "battery_empty":
                return "off", "沒電了，坐下關機。換電池或插電開機後牠會醒來找你。"
            return "off", "關機中（家人讓牠休息了）。開機後牠會醒來找你。"
        if s.posture == "fallen" and s.righting is not None:
            return "fallen", "走路時跌了一跤，正在試著自己翻回來（最多試 3 次）。"
        if s.posture == "fallen":
            return (
                "fallen",
                "走路時跌了一跤，試了幾次翻不回來，側躺著等人扶（抱起來，或關機再開機）。牠沒事。",
            )
        if s.hot:
            return "hot", "太熱了，休息姿勢、只發很少的聲音。溫度降下來就恢復。"
        if s.yielding:
            return "yield", "手把正在操作，Duck Core 讓位，只在旁邊看。"
        if self._until(s.lockout_until):
            return "lockout", "短時間內被叫停好幾次，先安靜一下（最多 30 分鐘）。"
        if self._until(s.stop_until):
            return "stop", "有人扶住牠的頭，牠停下來了；60 秒沒碰就恢復。"
        if self._until(s.napping_until):
            assert s.napping_until is not None
            if s.nap_reason == "quiet_hours" and self.hw.nest:
                where = "在窩裡" if s.nested else "在窩外（來不及回窩）"
                return (
                    "nap",
                    f"晚上{where}睡覺，早上 7 點會自己醒來；睡著時偶爾會動一動、說夢話。",
                )
            if s.nap_reason == "away_rest":
                return (
                    "nap",
                    "家裡沒人，在窩裡睡一覺（一邊充電），偶爾醒來伸懶腰、張望一下，等你們回來。",
                )
            if s.nap_reason == "quiet_hours":
                return (
                    "nap",
                    "晚上插著電在睡覺（晚上 10 點到早上 7 點），每小時醒來一下，"
                    f"下次約 {s.napping_until.strftime('%H:%M')}；睡著時偶爾會動一動、說夢話。",
                )
            if s.nap_reason == "bedtime":
                return "nap", "電量低，打起瞌睡了；換新電池就會醒來。"
            return (
                "nap",
                f"累了，小睡中（精力 {s.energy * 100 // SCALE}%），大約再 {self.nap_left_min()} "
                "分鐘醒。摸摸頭、叫名字或拍手可能會吵醒牠。",
            )
        if self._until(s.quiet_until):
            return "quiet", "有人說「夠了」，安靜 15 分鐘（不會因此學到不喜歡）。"
        if s.battery <= BEDTIME_BATTERY and not s.docked:
            return "bedtime", "電量低，準備睡覺了。"
        if self.hw.nest:
            if s.stranded is not None:
                return "stranded", "走不動了，坐著等人抱牠回窩。"
            if s.unplaced:
                return "unplaced", "剛被抱起來：放回窩裡，或放到地上。"
            if s.homing is not None:
                return "homing", "正在自己走回窩。"
            if s.excursion is not None:
                return "awake", "出門走一小趟：" + excursion.REASON_ZH.get(
                    str(s.excursion.get("reason")), "走走"
                ) + "。"
        verb = content.VERB_BY_ID.get(s.last_verb or "")
        where = {"sit": "坐著", "stand": "站著", "walk": "正在走"}.get(s.posture, "")
        floor = "、在地板上" if s.floor else ""
        if self.hw.nest and s.nested:
            floor = "、在窩裡充電"
        return "awake", f"醒著（{where}{floor}），剛剛在「{verb.zh if verb else '發呆'}」。"

    def walk_reason(self) -> str:
        """Plain-language answer to "why isn't it walking?" (the hard gates, then the mood)."""
        blocks = self.walk_blocks()
        texts = {
            "off": "關機中，不會走。",
            "righting": "牠跌倒了，正在試著自己翻回來；翻回來坐好之後就會再走走。",
            "fallen": "牠跌倒了，自己翻不回來，要等人扶起來（或抱起來、關機再開機）。",
            "desk": "牠在桌上（沒有開地板模式）：只會自己站起來、坐下，不會走。",
            "too_young": f"牠還太小，大約第 {content.FIRST_STEPS_GD} 個成長日身體才準備好走第一步。",
            "tethered": "插著電線，為了安全不會走。",
            "no_pickup_sensor": "沒有抱起偵測，分不出是不是被抱起來，所以不會走。",
            "unvalidated": "站起來和走路還沒通過物理驗證（真機要等 A100 驗證）。",
            "battery": "電量低於 30%，先不站起來、不走。",
            "night": "晚上 10 點到早上 7 點不走動。",
            "hot": "太熱了，先休息，不走。",
            "yield": "手把在操作，先不自己走。",
            "lockout": "被叫停好幾次，先安靜一下，不走。",
            "stop": "有人扶住牠，先不走。",
            "nap": "正在睡覺，醒來才會走。",
            "quiet": "有人說「夠了」，安靜中，先不走。",
            "tired": "累了（精力太低），先坐著休息，恢復了就會再起來走走。",
            "unplaced": "剛被抱起來，還不知道自己在哪裡：放回窩裡，或按「放到地上」就會再走。",
            "stranded": "電量太低或進不去窩，走不動了：請抱牠回窩。",
            "stand_backoff": "剛剛站不起來：休息一下再試。",
            "dawn": "剛睡醒，在窩上賴一下，等一下就會出來走走。",
            "bedtime": "晚上 9 點半了：回窩準備睡覺，明天早上 7 點再出來。",
            "alone": "家裡沒人：牠在窩裡休息，不會自己出去走，等你們回來。",
            "solo_done": "家人還在睡：牠在窩旁逛夠了，回窩等你們起床。",
            "family_quiet": "最近沒聽到家人的聲音：牠在窩旁休息，聽到有人就會出來走走。",
            "door": "剛聽到開關門：先停下來聽聽看是不是有人出門了。",
            "nest_trouble": "窩附近跌倒好幾次：有人在旁邊時才會進出窩（請檢查地面）。",
            "charging": "在窩裡充電：電量過一半就會自己出來走走。",
            "going_home": "電量快到回窩的線了：只會走回窩充電。",
            "charging_nest": "在窩裡充電。",
        }
        order = (
            "off",
            "righting",
            "fallen",
            "unplaced",
            "stranded",
            "desk",
            "too_young",
            "tethered",
            "no_pickup_sensor",
            "unvalidated",
            "battery",
            "night",
            "hot",
            "yield",
            "lockout",
            "stop",
            "nap",
            "quiet",
            "stand_backoff",
            "dawn",
            "bedtime",
            "alone",
            "solo_done",
            "family_quiet",
            "door",
            "nest_trouble",
            "charging",
            "going_home",
            "tired",
        )
        for code in order:
            if code in blocks:
                return texts[code]
        if self.hw.nest:
            s = self.s
            where = "在窩裡" if s.nested else "在窩外"
            solo = (
                f"（家人還在睡：只在窩旁 {excursion.leash_mm(self) / 1000:.1f} 公尺內、小小聲）"
                if s.phase == "solo"
                else ""
            )
            return (
                f"可以走（{where}、白天、電量夠）{solo}：牠會自己出門走一小趟，走幾段、停下來看看，"
                "累了或電少了就自己回窩充電，充好電再出來。"
            )
        spots = len(self.active_spots())
        avoid = f"；有 {spots} 個地方牠跌倒好幾次、很難翻回來，會繞開那裡" if spots else ""
        return (
            "可以走（地板模式、白天、電量夠）：醒著的鴨子每隔幾分鐘就會自己起來走走。"
            f"多常走、走多遠看好奇心、個性、年紀、心情、你們喜不喜歡、剛剛走過沒{avoid}。"
        )

    def why_verb(self, verb_id: str) -> dict[str, Any]:
        for event in reversed(self.log.events):
            if event.kind == "duck.decision" and event.payload.get("verb") == verb_id:
                payload = event.payload
                inputs = mapping(payload.get("inputs"))
                probabilities = mapping(payload.get("probabilities"))
                verb = content.VERB_BY_ID[verb_id]
                share = int(num(probabilities.get(verb_id)))
                return {
                    "found": True,
                    "decision_event": event.event_id,
                    "text": (
                        f"我在 {event.occurred_at.strftime('%m/%d %H:%M')} 選了「{verb.zh}」："
                        f"那時精力 {int(num(inputs.get('energy'))) * 100 // SCALE}%、心情 {int(num(inputs.get('valence'))) * 100 // SCALE}、"
                        f"興奮 {int(num(inputs.get('arousal'))) * 100 // SCALE}%；"
                        f"它的機率是 {share * 100 // SCALE}%，抽到的數是 {int(num(payload.get('draw'))) * 100 // SCALE}%（第 {payload.get('counter')} 次抽）。"
                    ),
                    "inputs": dict(inputs),
                    "probabilities": dict(probabilities),
                }
        return {"found": False, "text": "我最近沒有做過這個動作。"}


PerceptHandler = Callable[[DuckCore, Mapping[str, Any], Event, Act | None], str | None]
# What Duck Core does with each percept after recording it (on_percept). An answer to an act
# the world gave at once (a stand-up's body.posture, a dock try's body.dock, the nest look's
# sense.sighting / sense.scan) and a minute of gamepad play (body.play) is used where it was
# asked for, in place (design A2): their handlers only take what nobody asked for.
_PERCEPT_HANDLERS: dict[str, PerceptHandler] = {
    "walk.done": DuckCore._on_walk_done,
    "body.fell": DuckCore._on_fell,
    "body.righting": DuckCore._on_righting,
    "sense.touch": DuckCore._on_touch,
    "sense.hold_head": DuckCore._on_hold_head,
    "sense.sighting": nest.on_sighting,
    "sense.scan": nest.on_sighting,
    "body.dock": nest.on_dock,
    "sense.thing": DuckCore._on_thing,
    "body.curio_step": DuckCore._on_curio_step,
    "body.guard": DuckCore._on_guard,
}
