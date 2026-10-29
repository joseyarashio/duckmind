"""The duck's 14 emotions in the simulator (2026-10-08): Pollen Robotics' ``microduck_emotions`` played by the
neural duck's body link, chosen by an emotion layer from its own state.

NOTICE: the motions (``emotions_data.json``: the keyframes of every emotion, 0.1 s steps) and the way they are played
(head targets as deltas from HOME on ``robot.head``, body pitch on ``robot.pose``, the beak on ``robot.mouth``, the
sound at ``sound_at_s``) are from https://github.com/pollen-robotics/microduck_emotions (Apache-2.0, commit 431c73d;
``third_party/microduck_emotions``). The sounds are made again in each duck's own voice (:mod:`duckmind.simweb.emotion_voice`).

**Playing** (:class:`Player`, pure and clock-injected: the live driver calls :meth:`Player.frame` once per robot.state
tick): the keyframes are interpolated linearly at the control rate; the frame gives the head offsets, the beak, the body
pitch, whether the duck sits, and the tick the sound starts at. The body link (:mod:`duckmind.neuro.bodylink`) sends them
under its own rules (the head and the pose go home while it rights itself, rises or sits down), the driver zeroes the twist
and holds the brain still. A reflex, a fall, a lift or the sleep ends an emotion at once; nothing resumes.

**Choosing** (:class:`Picker`, the brain's emotion layer): reads what the brain already knows (its drives, behaviour,
bouts, stalls, reflexes, temperament) and the world's cues the page injects (a voice, a person arriving, a ball), and
picks an emotion with a probability and the temperament's weights, per-emotion cooldowns and never while it flees, is
startled, rights itself, sleeps, docks or runs one of its programs. Pure: the same seed and the same cues give the same
choices. The page's 「動作試玩」 group 「情緒」 presses the same emotions by hand.
"""

from __future__ import annotations

import json
import random
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

DATA = Path(__file__).with_name("emotions_data.json")
ID_PREFIX = "emo_"  # the page's ids (POST /api/neuro/motion {"id": "emo_sad"})
SOUND_PREFIX = "emo_"  # the voice tags (simweb/emotion_voice.TAG_PREFIX): emo_<name>
Offsets = tuple[float, float, float, float]
# the page's order and its plain description keys live in static/i18n.*.page.js (page.emotion.*)
ORDER: tuple[str, ...] = (
    "curious", "mmh", "yes", "yes_fast", "no", "excited", "laugh", "mock",
    "impatient", "defiant", "angry", "sad", "devastated", "play_dead",
)  # fmt: skip
SIT = frozenset(
    {"devastated", "play_dead"}
)  # the duck sits down in these (and stands up again after)
# the verdict of each from the sim check (docs/design/neural-duck/EMOTIONS-2026-10-08.md and BUGS-2026-10-09.md): pass | limits | fail
VERDICT: dict[str, str] = {name: "pass" for name in ORDER}
VERDICT.update(
    play_dead="limits"
)  # (sitting only, with the official robotd: see lying_verdict(); defiant and angry now pass: BEAK_CAPPED)
# Defiant and angry opened the beak wide while the head was turned to the side, and the lower beak then touched the neck
# housing (37 and 61 ticks of 5 presses, 2026-10-08). With the head yawed the housing is in the jaw's way, so the opening
# is capped by the yaw (:func:`mouth_cap`; measured on the composed model with the jaw, tests/test_emotions_beak.py).
# Only these two are reshaped: the other twelve showed no touch in the same check.
BEAK_CAPPED = frozenset({"defiant", "angry"})
MOUTH_CAP_BASE = (
    1.02  # the largest opening with the head straight, and how fast it shrinks with the yaw:
)
MOUTH_CAP_PER_RAD = (
    0.62  # cap = 1.02 - 0.62 |yaw| (the model's free opening is 0.05 to 0.08 above it)
)


# ── play dead lies down (2026-10-11, docs/design/neural-duck/PLAY-DEAD-2026-10-11.md) ─────────────────────────────
# The workshop's play dead (microduck_emotions, workshop patch 17-22; emotions.json ``pose_joints``): the duck sits and
# sounds the alarm at t = 0, its head whips to the side and back (the keyframes); at DEAD1_AT the head servos hang free and
# the legs are ramped to straight (hip yaw / roll / pitch, knee, ankle all 0) at servo gain 160, so the seated duck rolls flat
# onto its back; at DEAD2_AT the legs come up (hip pitch -1.0 / +1.0, knee +1.5 / -1.5, mirrored); the death quack plays; the
# pose is held (the beak stays live). Where the workshop waited for a person's Start, the duck here gets up by itself after
# WAKE_HOLD_S seconds with its legs up (robot.init: torque back on, the two-second ramp home; robot.enable: the walking policy).
# It needs a robotd that answers robot.poseJoints (the simulator-only build; the official 0.15.0 has none: play dead sits).
DEAD_JOINTS = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll", "mouth",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)  # fmt: skip
DEAD_OFF = ("neck_pitch", "head_pitch", "head_yaw")  # the servos that hang free
DEAD_GAIN = 160  # servo P gain of the legs' ramps (the standing value)
DEAD1_AT_S, DEAD1_RAMP_S = (
    1.8,
    1.5,
)  # legs straight: a late seat topples the duck forward (browser finding)
DEAD2_AT_S, DEAD2_RAMP_S = 4.0, 1.0  # legs up
DEAD_SEAT_HOLD_S = 0.6  # the legs go straight only when robotd's seat has held this long (the workshop's 1.8 s counts on it)
DEAD_SEAT_GIVE_UP_S = 6.0  # no seat by then: the play stays the sitting one
DEAD_LEG_UP_LEFT = (0.0, 0.0, -1.0, 1.5, 0.0)  # hip yaw, hip roll, hip pitch, knee, ankle
DEAD_LEG_UP_RIGHT = (0.0, 0.0, 1.0, -1.5, 0.0)
WAKE_HOLD_S = (
    5.0,
    8.0,
)  # it lies with its legs up this long (a draw per play), then gets up by itself
WAKE_HOMING_S = 2.15  # robotd's init ramp home takes 2 s: the walking policy is enabled after it
WAKE_UP_TIMEOUT_S = 12.0  # not up this long after the policy is back (a rise takes about 4 s): the ordinary fall and righting take over
WAKE_UP_HOLD_S = (
    1.0  # up = tilt < 25 degrees, robotd not fallen, walk / stand, this long (proprio's own rule)
)
WAKE_UP_TILT_DEG = 25.0


def dead_pose(stage: int) -> list[float | None]:
    """robot.poseJoints ``targets`` (15 values in robotd's joint order, None = hold where it is) of stage 1 (legs
    straight) or 2 (legs up)."""
    left = (0.0,) * 5 if stage == 1 else DEAD_LEG_UP_LEFT
    right = (0.0,) * 5 if stage == 1 else DEAD_LEG_UP_RIGHT
    return [*left, None, None, None, None, None, *right]


def wake_hold_s(seed: object) -> float:
    """How long this play lies with its legs up before it gets up (seeded: the same play gets the same time)."""
    low, high = WAKE_HOLD_S
    return round(random.Random(str(seed)).uniform(low, high), 2)


def lying_verdict(can_lie: bool) -> str:
    """The page's verdict of play dead: it passes when this simulator's robotd can lie the duck down."""
    return "pass" if can_lie else "limits"


def mouth_cap(yaw: float) -> float:
    """The largest beak opening (0..1) that keeps the lower beak off the neck housing at this head yaw (rad)."""
    return max(0.0, min(1.0, MOUTH_CAP_BASE - MOUTH_CAP_PER_RAD * abs(yaw)))


# the head may go further than the walking envelope in an emotion (the workshop's targets: neck down to -1.5 rad, the
# head 1.0 rad; yaw 1.0): robotd's own limits clamp what the real servos can do
EMOTION_HEAD = ((-1.60, 0.30), (-1.10, 1.10), (-1.10, 1.10), (-0.60, 0.60))
BOW_MAX = 0.26  # robot.pose pitch: the trained range (duck-ipc-proto PoseParams: z -0.025..0.010 m, pitch +-0.26 rad)
SETTLE_TWIST = 0.05  # a body slower than this (m/s and rad/s) is still enough to start
SIT_HOLD_S = 1.2  # a sit emotion stays seated this long after its last keyframe, then stands up


@lru_cache(maxsize=1)
def load() -> dict[str, Any]:
    """The workshop's keyframes (``emotions_data.json``)."""
    return json.loads(DATA.read_text())


@dataclass(frozen=True)
class Emotion:
    name: str
    duration_s: float
    sound_at_s: float
    sound_tag: (
        str  # the voice tag (emo_<name>); the sound is a random variant of it, as the robot does
    )
    sit_from_s: float | None
    rows: tuple[
        tuple[float, ...], ...
    ]  # t, neck, head_pitch, head_yaw, head_roll, body_pitch, mouth

    @property
    def id(self) -> str:
        return ID_PREFIX + self.name

    @property
    def sits(self) -> bool:
        return self.sit_from_s is not None

    @property
    def bows(self) -> bool:
        return any(row[5] > 1e-6 for row in self.rows)


@lru_cache(maxsize=1)
def catalog() -> dict[str, Emotion]:
    data = load()
    out: dict[str, Emotion] = {}
    for name in ORDER:
        e = data["emotions"][name]
        rows = tuple(tuple(float(v) for v in row) for row in e["keyframes"])
        sit = e["sit_from_s"]
        lead = float(e["sound_at_s"])
        if lead > 0.0:
            # devastated: the file starts 0.3 s before the press (the sit and the wav start at 0.3 s): the press is t = 0
            rows = tuple((row[0] - lead, *row[1:]) for row in rows if row[0] - lead >= -1e-9)
            sit = None if sit is None else max(0.0, float(sit) - lead)
        if (
            name in BEAK_CAPPED
        ):  # (the cap is linear in |yaw|, so it holds between the keyframes too)
            rows = tuple((*row[:6], min(row[6], mouth_cap(row[3]))) for row in rows)
        # the sound starts when the beak's keyframes say (the workshop's mouth lags its wav by ~0.16 s: tools/build)
        out[name] = Emotion(
            name,
            float(e["duration_s"]),
            float(e.get("sound_delay_s", 0.0)),
            SOUND_PREFIX + name,
            sit,
            rows,
        )
    return out


BY_ID = {ID_PREFIX + name: name for name in ORDER}


def status(loaded: bool = True, can_lie: bool | None = None) -> list[dict[str, Any]]:
    """The page's list: ids and facts, never words (``/api/neuro/status`` ``emotions``). ``can_lie``: this
    simulator's robotd can lie the duck down for play dead (it answers ``robot.poseJoints``)."""
    cat = catalog()
    out = []
    for name in ORDER:
        e = cat[name]
        out.append(
            {
                "id": e.id,
                "group": "emotion",
                "verdict": lying_verdict(can_lie is not False)
                if name == "play_dead"
                else VERDICT.get(name, "pass"),
                "postures": ["stand"],  # (a sitting duck is not asked: it stands up first)
                "brain": name in BRAIN_USE,
                "held_s": None,
                "duration_s": e.duration_s,
                "sits": e.sits,
                "lies": can_lie
                if name == "play_dead"
                else False,  # play dead lies down on its back: needs the robotd with poseJoints (None: not known yet)
                "needs": [],
                "installed": loaded,
            }
        )
    return out


# ── playing ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Frame:
    """What an emotion wants of the body at one moment."""

    t_s: float
    head: Offsets
    mouth: float
    pitch: float | None  # robot.pose pitch (rad; > 0 bows), None: the pose is not driven
    sit: bool  # the keyframes say 'sit' (or it holds the seat before standing)
    voice: str | None  # the voice tag, once, at the tick the sound starts
    done: bool


def _lerp_rows(rows: Sequence[Sequence[float]], t: float) -> Sequence[float]:
    if t <= rows[0][0]:
        return rows[0]
    if t >= rows[-1][0]:
        return rows[-1]
    lo, hi = 0, len(rows) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if rows[mid][0] <= t:
            lo = mid
        else:
            hi = mid
    a, b = rows[lo], rows[hi]
    u = (t - a[0]) / (b[0] - a[0]) if b[0] > a[0] else 0.0
    return [x + (y - x) * u for x, y in zip(a, b, strict=True)]


class Player:
    """One emotion playing: :meth:`frame` at each tick's time (seconds on any steady clock)."""

    def __init__(self, emotion: Emotion, t0_s: float) -> None:
        self.emotion = emotion
        self.t0_s = float(t0_s)
        self.voiced = False
        self.ended_s: float | None = None
        self.aborted: str | None = None
        self._bowed = emotion.bows

    @property
    def length_s(self) -> float:
        e = self.emotion
        return e.duration_s + (SIT_HOLD_S if e.sits else 0.0)

    def abort(self, why: str) -> None:
        if self.aborted is None:
            self.aborted = why

    def frame(self, now_s: float) -> Frame:
        e = self.emotion
        t = max(0.0, now_s - self.t0_s)
        row = _lerp_rows(e.rows, t)
        done = self.aborted is not None or t >= self.length_s
        if done and self.ended_s is None:
            self.ended_s = t
        voice = None
        if not self.voiced and t >= e.sound_at_s and not done:
            self.voiced = True
            voice = e.sound_tag
        sit = (
            e.sit_from_s is not None and t >= e.sit_from_s and t <= e.duration_s + SIT_HOLD_S - 0.6
        )
        return Frame(
            t_s=t,
            head=(float(row[1]), float(row[2]), float(row[3]), float(row[4])),
            mouth=max(0.0, min(1.0, float(row[6]))),
            pitch=min(BOW_MAX, max(0.0, float(row[5]))) if self._bowed else None,
            sit=sit,
            voice=voice,
            done=done,
        )


# ── choosing ──────────────────────────────────────────────────────────────────

# how the brain uses each (the page says it in plain words: page.emotion.brain.<name>)
BRAIN_USE: dict[str, str] = {
    "curious": "novelty", "mmh": "speech", "excited": "arrival", "laugh": "arrival", "mock": "tease",
    "impatient": "blocked", "defiant": "blocked", "angry": "blocked", "sad": "lonely",
    "devastated": "lonely", "play_dead": "scare",
}  # fmt: skip
# per-emotion cooldown (s, on the duck's clock) after one is played: the same feeling does not come back at once
COOLDOWN_S: dict[str, float] = {
    "curious": 45.0, "mmh": 30.0, "yes": 20.0, "yes_fast": 20.0, "no": 20.0, "excited": 60.0,
    "laugh": 75.0, "mock": 150.0, "impatient": 70.0, "defiant": 110.0, "angry": 300.0, "sad": 200.0,
    "devastated": 900.0, "play_dead": 600.0,
}  # fmt: skip
GAP_S = 14.0  # no emotion starts this soon after another ended
SPEECH_DELAY_S = 1.2  # it listens first, then says 'what?'
# behaviours that are not a quiet moment: it flees, is startled, rights itself, sleeps, is held, docks …
BUSY_BEHAVIOURS = frozenset(
    {
        "flee", "retreat", "startle", "withdraw", "righting", "held", "freeze", "return_dock", "zoomies",
        "nap", "doze", "sleep_talk", "melt", "stand_up", "sit_down", "peekaboo", "peekaboo_master",
    }
)  # fmt: skip
PROGRAM_BEHAVIOURS = frozenset(
    {"follow_mom", "hide", "circle_settle", "theremin", "wheee_ride", "dance", "play_ball",
     "carry_toy", "sunbathe"}
)  # fmt: skip
STILL_V = 0.05  # m/s: it asks for no walking


@dataclass(frozen=True)
class Temperament:
    """Weights from the genome: bold / shy / songbird / lazy, each 0..1."""

    bold: float = 0.5
    shy: float = 0.5
    songbird: float = 0.25
    lazy: float = 0.5

    @classmethod
    def of(cls, genome: Any) -> Temperament:
        if genome is None:
            return cls()
        axes = getattr(genome, "axes", {}) or {}
        nature = getattr(genome, "nature", {}) or {}
        bold = float(axes.get("boldness", 0.5))
        song = float(nature.get("songbird", 0.25))
        if getattr(genome, "branch", None):
            song = 1.0 if genome.branch == "songbird" else 0.0
        return cls(bold=bold, shy=1.0 - bold, songbird=min(1.0, song * 2.0),
                   lazy=1.0 - float(axes.get("activity", 0.5)))  # fmt: skip


def weights(t: Temperament) -> dict[str, float]:
    """The temperament's weight of each emotion (1 = neutral): bold more defiant / angry / mock, shy more sad /
    play_dead / mmh, a songbird more laugh / excited / mock, a lazy duck fewer of everything."""
    base = 1.0 - 0.45 * t.lazy
    w = {
        "curious": 1.0 + 0.4 * t.bold,
        "mmh": 0.6 + 0.9 * t.shy,
        "excited": 0.6 + 0.5 * t.songbird + 0.3 * t.bold,
        "laugh": 0.3 + 1.4 * t.songbird,
        "mock": 0.1 + 0.9 * t.bold * (0.4 + 0.6 * t.songbird) + 0.3 * t.songbird,
        "impatient": 0.9 + 0.3 * t.bold,
        "defiant": 0.15 + 1.3 * t.bold,
        "angry": 0.05 + 1.0 * t.bold * t.bold,
        "sad": 0.4 + 1.3 * t.shy,
        "devastated": 0.2 + 1.2 * t.shy,
        "play_dead": 0.03 + 1.3 * t.shy * t.shy,
    }
    return {k: v * base for k, v in w.items()}


@dataclass
class Context:
    """What the picker sees each tick (built by the driver from the brain's command and telemetry)."""

    t_s: float
    behaviour: str
    new_bout: bool
    reflex: str | None
    asleep: bool
    wants_walk: bool  # the brain asks for a walk
    moving: bool  # the body is walking / turning now
    standing: bool
    quiet: bool  # nothing holds the body (no fall, lift, righting, battery gate, dock, trained skill running)
    drives: Mapping[str, float]
    stuck: float  # 0..1 (the brain's own stalls at one place)
    blocked: bool  # a stall / bump happened this tick
    strong_scare: float  # the largest reflex strength in the last few seconds (0..1)
    cornered: bool
    person_in_view: bool
    bout_ms: int = 0
    play_over: bool = False  # a social / play program ended this tick (a game, a dance, a ride)
    greeted: bool = False  # a person called it / came to it this tick
    cues: tuple[str, ...] = ()  # world cues this tick (speech, arrival, ball, …)


@dataclass
class Choice:
    name: str
    why: str


class Picker:
    """The emotion layer: :meth:`step` once per tick when no emotion plays; returns a Choice or None."""

    def __init__(self, temperament: Temperament, seed: int = 0) -> None:
        self.temperament = temperament
        self.w = weights(temperament)
        self.rng = random.Random(int(seed) * 7919 + 4107)
        self.cool_until: dict[str, float] = {}
        self.last_end_s = -1e9
        self.pending: deque[tuple[float, str, float, str]] = (
            deque()
        )  # (due_s, emotion, probability, why)
        self.lonely_since: float | None = None
        self.very_lonely_since: float | None = None
        self.prev_behaviour = ""
        self.next_slow_s = 0.0
        self.counts: dict[str, int] = {}
        self.refused: dict[str, int] = {}

    # the driver says when one ended (for the gap)
    def ended(self, t_s: float) -> None:
        self.last_end_s = t_s

    def cooling(self, name: str, t_s: float) -> bool:
        return t_s < self.cool_until.get(name, -1e9)

    def _roll(self, name: str, p: float, t_s: float) -> bool:
        """A trigger fired for ``name``: play it with probability ``p`` × its temperament weight."""
        if self.cooling(name, t_s):
            return False
        return self.rng.random() < min(0.95, p * self.w.get(name, 1.0))

    def played(self, name: str, t_s: float) -> None:
        self.cool_until[name] = t_s + COOLDOWN_S.get(name, 60.0) * (0.8 + 0.4 * self.rng.random())
        self.counts[name] = self.counts.get(name, 0) + 1

    def _later(self, ctx: Context, delay_s: float, name: str, p: float, why: str) -> None:
        self.pending.append((ctx.t_s + delay_s, name, p, why))
        while len(self.pending) > 6:
            self.pending.popleft()

    def step(self, ctx: Context) -> Choice | None:
        t = ctx.t_s
        d = ctx.drives
        beh, new_bout = ctx.behaviour, ctx.new_bout
        # ── feelings that build up: a social need unmet for long ──
        lonely = float(d.get("social", 0.0)) >= 0.80 and not ctx.person_in_view
        self.lonely_since = (
            (self.lonely_since if self.lonely_since is not None else t) if lonely else None
        )
        very = float(d.get("social", 0.0)) >= 0.92 and not ctx.person_in_view
        self.very_lonely_since = (
            (self.very_lonely_since if self.very_lonely_since is not None else t) if very else None
        )
        calm = ctx.quiet and ctx.standing and not ctx.asleep and ctx.reflex is None
        busy = (
            not calm
            or beh in BUSY_BEHAVIOURS
            or beh in PROGRAM_BEHAVIOURS
            or ctx.moving
            or t - self.last_end_s < GAP_S
        )
        # ── cues to remember for a moment (they fire later, when it is quiet) ──
        for cue in ctx.cues:
            if cue == "speech":
                self._later(ctx, SPEECH_DELAY_S, "mmh", 0.5, "speech")
            elif cue == "arrival":
                self._later(ctx, 1.5, "excited", 0.5, "arrival")
                self._later(ctx, 2.0, "laugh", 0.3, "arrival")
            elif cue == "greeted":
                self._later(ctx, 1.0, "excited", 0.4, "greeted")
                self._later(ctx, 1.2, "laugh", 0.3, "greeted")
        if new_bout and beh == "investigate":
            self._later(
                ctx, 0.0, "curious", 0.4 * (0.5 + float(d.get("curiosity", 0.45))), "novelty"
            )
        if ctx.play_over:
            self._later(ctx, 0.5, "laugh", 0.45, "play")
            self._later(ctx, 0.6, "excited", 0.35, "play")
        if ctx.greeted:
            self._later(ctx, 1.0, "excited", 0.4, "greeted")
        if new_bout and beh in (
            "happy_sway",
            "little_dance",
            "little_song",
            "call_response",
            "nod",
        ):
            self._later(ctx, 0.3, "mock", 0.18 * (0.5 + float(d.get("play", 0.0))), "tease")
        # blocked / stuck again and again
        if ctx.blocked and ctx.stuck >= 0.60:
            self._later(ctx, 0.8, "impatient", 0.55, "blocked")
            self._later(ctx, 1.0, "defiant", 0.35, "blocked")
            if ctx.stuck >= 0.95:
                self._later(ctx, 1.2, "angry", 0.30, "blocked")
        if beh == "return_dock" and ctx.bout_ms > 45_000 and float(d.get("hunger", 0.0)) > 0.5:
            self._later(ctx, 0.0, "impatient", 0.25, "waiting")
        # a very strong scare while cornered: play dead instead of flight (shy, rare)
        if ctx.strong_scare >= 0.85 and ctx.cornered and ctx.reflex is None:
            self._later(ctx, 1.5, "play_dead", 0.45, "scare")
        # loneliness: once a second it may well up (about 1 in 50 s once it has lasted 70 s; devastated, much longer)
        if t >= self.next_slow_s:
            self.next_slow_s = t + 1.0
            if self.lonely_since is not None and t - self.lonely_since > 70.0:
                self._later(ctx, 0.0, "sad", 0.02, "lonely")
            if self.very_lonely_since is not None and t - self.very_lonely_since > 200.0:
                self._later(ctx, 0.0, "devastated", 0.004, "lonely")
        # ── the one that is due and quiet enough ──
        out: Choice | None = None
        keep: deque[tuple[float, str, float, str]] = deque()
        while self.pending:
            due, name, p, why = self.pending.popleft()
            if due > t:
                keep.append((due, name, p, why))
                continue
            if out is not None or busy or (name in SIT and not calm):
                if t - due < 4.0:
                    keep.append((due, name, p, why))  # (it waits a few seconds for a quiet moment)
                continue
            if self._roll(name, p, t):
                out = Choice(name, why)
        self.pending = keep
        self.prev_behaviour = beh
        return out


def cue_of_stimulus(name: str) -> str | None:
    """The cue a page stimulus is for the picker: speech heard, a person arriving / greeting."""
    if name in ("voice",):
        return "speech"
    if name in ("walk_by", "sit_near"):
        return "arrival"
    if name in ("call_name", "hand_pat", "pet", "stroke"):
        return "greeted"
    return None


@dataclass
class EmotionLog:
    """What was played, for the page's note and the measurements: one dict per emotion."""

    items: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=400))
    seq: int = 0

    def add(self, item: dict[str, Any]) -> dict[str, Any]:
        self.seq += 1
        item = {"seq": self.seq, **item}
        self.items.append(item)
        return item

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for it in self.items:
            if it.get("state") == "started":
                out[it["name"]] = out.get(it["name"], 0) + 1
        return out


def clamp_head(head: Offsets) -> Offsets:
    return tuple(  # type: ignore[return-value]
        min(hi, max(lo, v)) for v, (lo, hi) in zip(head, EMOTION_HEAD, strict=True)
    )


__all__ = [
    "BRAIN_USE",
    "BY_ID",
    "Choice",
    "Context",
    "Emotion",
    "EmotionLog",
    "Frame",
    "ORDER",
    "Picker",
    "Player",
    "Temperament",
    "VERDICT",
    "catalog",
    "clamp_head",
    "cue_of_stimulus",
    "status",
    "weights",
]
