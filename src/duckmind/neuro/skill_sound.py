"""The sound skills in the neural brain — hand theremin, wheee ride, dancing (a:sound 2026-09-28).

Owner, 2026-09-28 (「A B 全數」): behaviours (2) 手的特雷門琴, (3) 「咻～」兜風, (4) 聽音樂跳舞 must
EMERGE from the neural brain (drives, senses, learning, temperament), be physically performed
and visible in the 3D view, and be decided only from the duck's own senses. The three skills'
perception and appraisal are their own modules (:mod:`skill_theremin`, :mod:`skill_ride`,
:mod:`skill_dance`); this module is how they enter the brain:

* **Channels of the one competition.** ``theremin`` (玩特雷門琴) and ``dance`` (聽音樂跳舞) are new
  behaviours of :data:`~duckmind.neuro.contract.BEHAVIOURS`, i.e. new channels of the basal-
  ganglia selection (:mod:`duckmind.neuro.select`) with their own features (``hand_hover``,
  ``beat``), weights (:data:`WEIGHTS`), satiation and temperament gains (:func:`gains_for`) —
  they win the body only by out-competing everything else the duck wants, and end by
  satiation or a stronger pull, never a timer. ``wheee_ride`` (咻～兜風) is the thrilled form of
  being held (the lifted reflex's hold): what the body does in the air is the ride appraisal's.
  After a ride, ``after_ride`` pulls a head shake and a body shake.
* **Predictive hearing.** An onset the beat oscillator predicted (``BeatTick.expected``) is
  anticipated, not a surprise: it is taken out of the frame the reflex arcs see (no orienting to
  every kick drum), while the beat keeps it (a sound it cannot predict still startles it). So is
  an off-beat onset where the song had one one, two or four bars before
  (``DanceTick.recurring``): its syncopation surprises it once, not every time.
* **Engaged.** While the dance or the instrument is the act, a thing or an unnamed movement newly
  seen is noted but does not take the head (``reflex.ENGAGED_ACTS``, a covert look): the nod and
  the beam on the hand are the head's job. People, hands, sounds, touches and anything defensive
  still do.
* **What the body link and robotd get** (:class:`SoundOut`): the theremin up or down (robotd's
  ``robot.theremin``), the joy ride's ``wheee`` hold (robotd's ``robot.sound {tag: wheee, hold}``,
  re-notified each tick while it rides), the dance's head nod (:func:`skill_dance.bob_pose`, a
  motor overlay) and its beat-timed chirps (the brain's own calls).

:class:`SkillBrain` wraps a :class:`~duckmind.neuro.brain.NeuroBrain` so the prototype can run
this before the integration writes it into ``brain.py`` / ``select.py`` / ``contract.py`` /
``repertoire.py`` / ``motor.py``: :func:`install` adds the channels, features and programs to
those modules' tables AT RUNTIME (before any genome or brain is made — only the a:sound demo and
its headless harness call it); the wrapper feeds the features in and applies :class:`SoundOut`.
a:sound's report (its integration note) lists the permanent edits, one by one.

Standard library only (the brain side): no world, no sockets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from duckmind.i18n import msg
from duckmind.neuro import contract as C
from duckmind.neuro import skill_dance as SD
from duckmind.neuro import skill_ride as SR
from duckmind.neuro import skill_theremin as ST
from duckmind.neuro.contract import BehaviourSpec, Gaze, HeadPose, MotorCommand, SensorFrame

SOUND_SKILLS_VERSION = "neuro-sound-skills-1"

# ── the new behaviours, features and wiring (contract / select additions) ──────────────


def _spec(
    zh: str, family: str, posture: str, body_zh: str, sounds: tuple[str, ...]
) -> BehaviourSpec:
    return BehaviourSpec(zh, family, 0, posture, False, sounds, body_zh)


BEHAVIOURS: dict[str, BehaviourSpec] = {
    "theremin": _spec(
        msg("neuro.act.theremin.name"),
        "ethogram",
        "any",
        msg("neuro.act.theremin.body"),
        ("",),
    ),
    "dance": _spec(
        msg("neuro.act.dance.name"),
        "ethogram",
        "any",
        msg("neuro.act.dance.body"),
        ("",),
    ),
    "wheee_ride": _spec(
        msg("neuro.act.wheee_ride.name"),
        "reflex",
        "any",
        msg("neuro.act.wheee_ride.body"),
        ("wheee",),
    ),
}
FEATURES: tuple[str, ...] = ("hand_hover", "beat", "after_ride", "ride_joy", "ride_fear")
FEATURE_ZH: dict[str, str] = {
    "hand_hover": msg("neuro.why.hand_hover"),
    "beat": msg("neuro.why.beat"),
    "after_ride": msg("neuro.why.after_ride"),
    "ride_joy": msg("neuro.why.ride_joy"),
    "ride_fear": msg("neuro.why.ride_fear"),
}
# behaviour → {feature: weight} for the new channels. The hand and the beat are RELEASERS: the
# drives that make the duck want to play (play, curiosity, company, a good mood; for the dance
# the music, play, mood and arousal) scale their pull (:data:`PULL`, folded into the feature by
# :meth:`SoundSkills.pre`) but can never select the act by themselves — measured 2026-09-28:
# with the drives as terms of their own, a playful ToF duck took up the theremin with no hand
# anywhere (18 bouts in 18 min of free roaming, 12 × 90 s).
WEIGHTS: dict[str, dict[str, float]] = {
    "theremin": {
        "hand_hover": 1.25,
        "fear": -0.9,
        "sleepy": -0.6,
        "startled": -0.5,
        "touched_sharp": -0.6,
    },
    "dance": {
        "beat": 1.3,
        "fear": -0.8,
        "sleepy": -0.7,
        "startled": -0.4,
    },
}
# releaser feature → (its weight, {drive feature: weight}): the pull is weight × releaser ×
# (1 + Σ drive weights × drives / weight) — with the releaser at 1, exactly the old sum
PULL: dict[str, tuple[float, dict[str, float]]] = {
    "hand_hover": (1.25, {"play": 0.35, "curiosity": 0.2, "social": 0.25, "happy": 0.15}),
    "beat": (1.3, {"music": 0.25, "play": 0.35, "happy": 0.3, "arousal": 0.1}),
}


def pulled(releaser: str, value: float, drives: dict[str, float]) -> float:
    """A releaser's feature with the drives' pull folded in (0 without the releaser)."""
    weight, pulls = PULL[releaser]
    if value <= 0.0:
        return 0.0
    extra = sum(w * max(0.0, float(drives.get(name, 0.0))) for name, w in pulls.items())
    return value * (1.0 + extra / weight)


# … and the new features' pull on existing channels (a shake after a ride; a beat to dance to
# inhibits the self-paced sway and 小舞步, which follow no music's beat)
EXTRA_WEIGHTS: dict[str, dict[str, float]] = {
    "head_shake": {"after_ride": 0.9},
    "body_shake": {"after_ride": 1.3},
    "stand_up": {"after_ride": 0.7},
    "preen": {"after_ride": 0.25},
    "little_dance": {"beat": -0.6},
    "happy_sway": {"beat": -0.5},
    # A hand hovering at the face is the nearest thing there is, and a thing far away that is new
    # to the eye must not win the duck from it on a hair: the hand pulls the theremin by up to
    # 1.25, a new thing pulls investigate by up to 0.9 + 0.3, and in the headless test room (a
    # dreamer, 2026-10-04) the two were level — the thing won by 0.01 and the duck walked off to
    # it while the hand played. A fifth of the hand's pull settles it; a duck with no hand in its
    # beam (hand_hover 0) loses nothing.
    "investigate": {"hand_hover": -0.25},
}
SATIATE_S: dict[str, float] = {"theremin": 45.0, "dance": 40.0, "wheee_ride": 30.0}
PLAY_ACTS = frozenset({"theremin", "dance"})  # playing satisfies the play drive
HEAD_ACTS = frozenset({"theremin", "dance"})  # their programs own the head (no idle gaze cycle)
# … and keep it against looks at props: a thing newly seen is noted, not turned to (reflex.py)
ENGAGED_ACTS = frozenset({"theremin", "dance"})
BEAK_BUSY = frozenset({"theremin"})  # robotd has the beak (and the voice) while it plays
EXERTION = {"dance": 0.35}


def gains_for(genome: Any) -> dict[str, float]:
    """The new channels' temperament gains (how likely, never whether — genome._gains' rule):
    the sociable and the bold play the theremin (a hand near the face), the lively and the
    sociable dance, a songbird most of all (its branch sings along)."""
    a = genome.axes
    songbird = _songbird(genome)
    return {
        "theremin": (0.55 + 0.45 * a["sociability"] + 0.4 * a["boldness"]),
        "dance": (0.7 + 0.3 * (a["activity"] + a["sociability"])) * (1.0 + 0.5 * songbird),
        "wheee_ride": 1.0,
    }


def _songbird(genome: Any) -> float:
    """How much of a songbird it is: 1 for the grown branch, its nature's share before."""
    branch = getattr(genome, "branch", None)
    if branch is not None:
        return 1.0 if branch == "songbird" else 0.0
    return float(getattr(genome, "nature", {}).get("songbird", 0.0))


# ── the programs (repertoire additions) ────────────────────────────────────────────────


def build_theremin(b: Any, ctx: Any) -> Any:
    """Playing the theremin: it looks up at the hand and holds its head still there (the brain's
    gaze has the yaw); now and then a slow listening tilt. The beak is robotd's (the note's).
    Looking up matters to the instrument too: at HOME a standing duck's lowest ToF zones see the
    floor ~45 cm ahead (robotd's tracker plays the raw beam); with the nose up ~12–18° the floor
    moves past 58–69 cm and the hand plays the whole band (a:sound, headless ToF model)."""
    side = ctx.cmd.side if ctx.cmd.side in (-1, 1) else ctx.style.side
    b.to(b.u(300, 450), hp=b.u(0.65, 0.9), np=b.u(-0.15, 0.0), hr=0.0, m=0.0)
    b.hold(b.u(1800, 2600), drift=0.01)
    if b.p(0.35):
        b.rel(b.u(350, 500), dhr=0.35 * side, ease="soft")
        b.hold(b.u(600, 1000), drift=0.0)
        b.rel(b.u(350, 500), dhr=-0.35 * side, ease="soft")
    b.hold(b.u(600, 1200), drift=0.01)
    return b.score(label="聽自己的聲音", alive=0.15, gaze=1.0)


def build_dance(b: Any, ctx: Any) -> Any:
    """Dancing: a lifted, open carriage held (the beat's nod and sway are the motor's overlay,
    phase-locked to what it hears)."""
    b.to(b.u(250, 400), hp=b.u(0.15, 0.3), np=b.u(-0.15, 0.0), hy=0.0, hr=0.0, m=0.0)
    b.hold(b.u(2500, 4000), drift=0.0)
    return b.score(label="跟著拍子點頭", alive=0.1, gaze=0.2)


def build_wheee_ride(b: Any, ctx: Any) -> Any:
    """In the air, thrilled: the head up into the wind, the beak a little open."""
    pose = SR.JOY_POSE
    b.to(b.u(250, 400), np=pose.np, hp=pose.hp, hy=0.0, hr=0.0, m=pose.mouth)
    b.hold(b.u(1500, 2500), drift=0.03)
    return b.score(label=msg("neuro.vocal.wheee"), alive=0.2, gaze=0.2)


BUILDERS = {"theremin": build_theremin, "dance": build_dance, "wheee_ride": build_wheee_ride}

# ── install: the tables at runtime (the a:sound demo and headless harness only) ───────

_INSTALLED = False
_SAVED: dict[str, Any] = {}


def install() -> None:
    """Add the new channels, features, weights and programs to the brain's tables — at runtime,
    before any genome or brain is made. Idempotent; :func:`uninstall` puts the tables back.
    (The integration makes it permanent in the files instead; see the module docstring.)"""
    global _INSTALLED
    if _INSTALLED:
        return
    from duckmind.neuro import brain as B
    from duckmind.neuro import genome as G
    from duckmind.neuro import reflex as R
    from duckmind.neuro import repertoire as P
    from duckmind.neuro import select as S

    _SAVED.update(
        engaged_acts=R.ENGAGED_ACTS,
        features=S.FEATURES,
        f=S.F,
        feature_zh=dict(S.FEATURE_ZH),
        w={name: dict(S.W[name]) for name in EXTRA_WEIGHTS if name in S.W},
        reflex_only=S.REFLEX_ONLY,
        play=S.PLAY,
        head_acts=B.HEAD_ACTS,
        beak_busy=B.BEAK_BUSY,
        exertion=dict(B.EXERTION),
        gains=G._gains,
    )
    for name, spec in BEHAVIOURS.items():
        C.BEHAVIOURS.setdefault(name, spec)
    new = tuple(f for f in FEATURES if f not in S.FEATURES)
    if new:
        S.FEATURES = S.FEATURES + new
        # Keep the mapping object stable: selectors/facades may already hold its reference.
        S.F.clear()
        S.F.update({name: index for index, name in enumerate(S.FEATURES)})
    S.FEATURE_ZH.update(FEATURE_ZH)
    for name, weights in WEIGHTS.items():
        S.W.setdefault(name, dict(weights))
    for name, weights in EXTRA_WEIGHTS.items():
        if name in S.W:
            S.W[name] = {**S.W[name], **weights}
    for name, seconds in SATIATE_S.items():
        S.SATIATE_S.setdefault(name, seconds)
    S.REFLEX_ONLY = S.REFLEX_ONLY | {"wheee_ride"}
    R.ENGAGED_ACTS = R.ENGAGED_ACTS | ENGAGED_ACTS
    S.PLAY = S.PLAY | PLAY_ACTS
    B.HEAD_ACTS = B.HEAD_ACTS | HEAD_ACTS
    B.BEAK_BUSY = B.BEAK_BUSY | BEAK_BUSY
    B.EXERTION.update(EXERTION)
    for name, builder in BUILDERS.items():
        P.BUILDERS.setdefault(name, builder)
    base_gains = G._gains

    def gains(*args: Any, **kwargs: Any) -> dict[str, float]:
        return {**base_gains(*args, **kwargs), **{name: 1.0 for name in BEHAVIOURS}}

    G._gains = gains  # (the temperament gains are applied per genome: SkillBrain)
    _INSTALLED = True


def uninstall() -> None:
    """Put the tables back as they were before :func:`install` (tests)."""
    global _INSTALLED
    if not _INSTALLED:
        return
    from duckmind.neuro import brain as B
    from duckmind.neuro import genome as G
    from duckmind.neuro import reflex as R
    from duckmind.neuro import repertoire as P
    from duckmind.neuro import select as S

    R.ENGAGED_ACTS = _SAVED["engaged_acts"]
    for name in BEHAVIOURS:
        C.BEHAVIOURS.pop(name, None)
        S.W.pop(name, None)
        S.SATIATE_S.pop(name, None)
        P.BUILDERS.pop(name, None)
    S.FEATURES = _SAVED["features"]
    S.F = _SAVED["f"]
    S.FEATURE_ZH.clear()
    S.FEATURE_ZH.update(_SAVED["feature_zh"])
    for name, weights in _SAVED["w"].items():
        S.W[name] = weights
    S.REFLEX_ONLY = _SAVED["reflex_only"]
    S.PLAY = _SAVED["play"]
    B.HEAD_ACTS = _SAVED["head_acts"]
    B.BEAK_BUSY = _SAVED["beak_busy"]
    B.EXERTION.clear()
    B.EXERTION.update(_SAVED["exertion"])
    G._gains = _SAVED["gains"]
    _SAVED.clear()
    _INSTALLED = False


def installed() -> bool:
    return _INSTALLED


# ── the skills together ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SoundOut:
    """What the sound skills add to one tick (for the body link, robotd and the page)."""

    t_ms: int
    theremin_up: bool  # robot.theremin: the instrument up
    wheee_hold: bool  # robot.sound {tag: wheee, hold: true} this tick (re-notified)
    dancing: bool
    bob: HeadPose  # the dance's head overlay now (clip units, added to the program's pose)
    chirp: str | None  # a beat-timed call started this tick
    alarm: bool  # a frightened ride called
    masked_onset: bool  # an expected beat onset was taken out of the reflexes' frame
    beat: SD.BeatTick
    theremin: ST.ThereminTick
    ride: SR.RideTick
    music: bool
    liking: float = 0.0  # its learned liking for dancing (skill_dance)
    timing_s: float = 0.0  # the nod timing it learned by feel (skill_dance.NodTimer.adjust)
    felt_nods: int = 0

    def view(self) -> dict[str, Any]:
        """Plain numbers and words for the page (no truth: all the duck's own)."""
        beat, th, ride = self.beat, self.theremin, self.ride
        return {
            "t_ms": self.t_ms,
            "dance": {
                "on": self.dancing,
                "beat_found": beat.active,
                "bpm": round(beat.bpm, 1) if beat.active else None,
                "lock": round(beat.lock, 2),
                "beats": beat.beats,
                "music": self.music,
                "per_nod": SD.beats_per_bob(beat.hz, "stand") if beat.active else None,
                "chirp": self.chirp,
                "liking": round(self.liking, 2),
                "timing_ms": round(self.timing_s * 1000.0),
                "felt_nods": self.felt_nods,
            },
            "theremin": {
                "up": self.theremin_up,
                "hand": th.view.present,
                "hand_mm": None if th.view.range_m is None else int(th.view.range_m * 1000),
                "moving": round(th.view.moving, 2),
                "feature": round(th.feature, 2),
                "value": round(th.value, 2),
                "playing_s": round(th.playing_s, 1),
            },
            "ride": {
                "lifted": ride.view.lifted,
                "riding": ride.view.riding,
                "outcome": "fear" if ride.frightened else "joy" if ride.thrilled else "none",
                "swing": round(ride.view.energy, 2),
                "thrilled": ride.thrilled,
                "frightened": ride.frightened,
                "wheee": self.wheee_hold,
                "after": round(ride.after_ride, 2),
                "value": round(ride.value, 2),
                "rides": ride.rides,
            },
        }


class SoundSkills:
    """The three skills for one duck (its genome): :meth:`pre` before the brain steps (what it
    hears and feels: the features, the masked onset), :meth:`post` after (the command's changes
    and :class:`SoundOut`)."""

    def __init__(self, genome: Any, seed: int = 0) -> None:
        a = genome.axes
        young = {"hatchling": 1.0, "duckling": 0.6, "juvenile": 0.25, "adult": 0.0}.get(
            str(getattr(genome, "stage", "juvenile")), 0.25
        )
        self.genome = genome
        self.dance = SD.DanceSkill(
            songbird=_songbird(genome),
            activity=float(a["activity"]),
            sociability=float(a["sociability"]),
            seed=int(seed),
        )
        self.theremin = ST.ThereminSkill()
        self.ride = SR.RideSkill(
            boldness=float(a["boldness"]), activity=float(a["activity"]), young=young
        )
        self.gains = gains_for(genome)
        self._beat: SD.BeatTick | None = None
        self._th: ST.ThereminTick | None = None
        self._ride: SR.RideTick | None = None
        self._dance_bout = -1
        self._dance_since: int | None = None  # when the dance bout now going on began (ms)
        self._music = False
        self._chirp: str | None = None
        self._masked = False
        self._posture = "stand"  # the body's posture now (the nod rate the head can follow)
        self.last: SoundOut | None = None

    # ── before the brain's step ──

    def pre(
        self,
        t_ms: int,
        frame: SensorFrame,
        *,
        behaviour: str,
        play: float,
        fear: float,
        arousal: float,
        asleep: bool,
        fright: bool,
        light: bool = False,
        drives: dict[str, float] | None = None,
    ) -> tuple[SensorFrame, dict[str, float]]:
        """What it hears and feels this tick. Returns the frame the reflex arcs should see (an
        expected beat onset taken out) and the new features for the selection. ``drives``: the
        brain's play / curiosity / social / happy / music / arousal features (the last tick's),
        which scale the hand's and the beat's pull (:data:`PULL`)."""
        dancing = behaviour == "dance"
        self._posture = "sit" if frame.body.posture == "sit" else "stand"
        d = self.dance.step(
            t_ms,
            frame.sound.onset,
            frame.sound.kind,
            dancing=dancing,
            asleep=asleep,
            arousal=arousal,
        )
        self._beat = d.beat
        self._music = d.music
        self._chirp = d.chirp
        # its own nods, felt (the neck's joint sense): the timing it gives them is learned
        head = frame.body.head
        self.dance.timer.step(
            t_ms / 1000.0,
            float(head[1]) if head and len(head) > 1 else None,
            d.beat,
            self._posture,
            dancing=dancing,
        )
        th = self.theremin.step(t_ms, frame, playing=behaviour == "theremin", fright=fright)
        self._th = th
        ride = self.ride.step(t_ms, frame, play=play, fear=fear, arousal=arousal, light=light)
        self._ride = ride
        # Only the quiet, unclassified beat-like transient is predictable. A classified event
        # (for example a clap) or a loud unclassified onset stays visible to the reflexes even
        # when it lands on a predicted beat.
        beat_like = frame.sound.kind in (None, "music") and frame.sound.level_db < 75.0
        masked = (d.beat.expected or d.recurring) and frame.sound.onset > 0.0 and beat_like
        self._masked = masked
        seen = frame
        if masked:
            seen = replace(frame, sound=replace(frame.sound, onset=0.0))
        drives = drives or {}
        features = {
            "hand_hover": pulled("hand_hover", th.feature, drives),
            "beat": pulled("beat", d.feature, drives),
            "after_ride": ride.after_ride,
            "ride_joy": ride.ride_joy,
            "ride_fear": ride.ride_fear,
        }
        return seen, features

    # ── after it ──

    def post(
        self, cmd: MotorCommand, *, bout: int, vocal_seq: int
    ) -> tuple[MotorCommand, SoundOut]:
        """The command as the skills change it, and what they add (:class:`SoundOut`).
        ``vocal_seq``: the brain's own call counter (a call of the skills takes the next)."""
        beat, th, ride = self._beat, self._th, self._ride
        assert beat is not None and th is not None and ride is not None
        behaviour = cmd.behaviour
        # Any reflex currently owns the head/body/voice output. ``lifted`` is the one deliberate
        # exception: the ride skill is the lifted reflex's own program.
        reflex_owned = cmd.reflex is not None and cmd.reflex != "lifted"
        dancing = behaviour == "dance" and not reflex_owned
        chirp: str | None = None
        alarm = False
        bob = HeadPose()
        changes: dict[str, Any] = {}
        if not dancing and self._dance_since is not None:
            self.dance.learn((cmd.t_ms - self._dance_since) / 1000.0)  # a dance ended
            self._dance_since = None
        if dancing:
            if bout != self._dance_bout:
                self._dance_bout = bout
                self._dance_since = cmd.t_ms
                self.dance.start_bout(bout, beat.hz)
            amount = max(0.0, min(1.0, 0.45 + 0.55 * cmd.amplitude))
            bob = SD.bob_pose(beat, self._posture, amount, adjust=self.dance.timer.adjust)
            self.dance.timer.sent_nod(-bob.hp)  # (its efference copy: + = nose down)
            chirp = self._chirp
            if cmd.vocal is not None and cmd.vocal_seq != vocal_seq and chirp is None:
                # (its own calls in a dance are only the beat's: an off-beat call is held back)
                changes.update(vocal=None, vocal_seq=vocal_seq)
            if chirp is not None:
                changes.update(vocal=chirp, vocal_seq=max(cmd.vocal_seq, vocal_seq) + 1)
        if behaviour == "theremin" and not reflex_owned:
            # the head holds still facing the hand (the ToF beam stays on it)
            changes.update(gaze=Gaze(th.gaze_bearing, 0.1, 1.0, 0.3), stabilise=0.0)
            if cmd.vocal is not None and cmd.vocal_seq != vocal_seq:
                changes.update(vocal=None, vocal_seq=vocal_seq)  # (robotd has the voice)
        if behaviour == "carry_toy":
            # The toy occupies the beak; no sound skill may reserve it this tick.
            chirp = None
            changes.update(vocal=None, vocal_seq=vocal_seq)
        ride_owns_output = cmd.reflex == "lifted"
        if ride.view.riding and ride_owns_output:
            # in the air and swung: the ride is the act (the lifted reflex's hold, thrilled or
            # frightened) — nothing walks, the head is the ride's
            changes.update(behaviour="wheee_ride" if ride.thrilled else "held", vx=0.0, vyaw=0.0)
            if ride.pose is not None:
                changes.update(head=ride.pose, head_blend=0.85)
            if ride.alarm:
                alarm = True
                changes.update(vocal="alarm", vocal_seq=max(cmd.vocal_seq, vocal_seq) + 1)
            elif ride.thrilled and cmd.vocal is not None and cmd.vocal_seq != vocal_seq:
                changes.update(vocal=None, vocal_seq=vocal_seq)  # (the ride has the voice)
        out_cmd = replace(cmd, **changes) if changes else cmd
        out = SoundOut(
            t_ms=cmd.t_ms,
            theremin_up=th.up and behaviour == "theremin" and not reflex_owned,
            wheee_hold=ride.wheee_hold and ride_owns_output,
            dancing=dancing,
            bob=bob,
            chirp=chirp,
            alarm=alarm,
            masked_onset=self._masked,
            beat=beat,
            theremin=th,
            ride=ride,
            music=self._music,
            liking=self.dance.liking,
            timing_s=self.dance.timer.adjust,
            felt_nods=self.dance.timer.felt,
        )
        self.last = out
        return out_cmd, out


class SkillBrain:
    """A :class:`~duckmind.neuro.brain.NeuroBrain` with the sound skills (the prototype's runtime
    composition; :func:`install` must have run before the brain was made). Everything else is the
    wrapped brain's (attributes are passed through)."""

    def __init__(self, base: Any) -> None:
        if not _INSTALLED:
            raise RuntimeError("skill_sound.install() must run before the brain is made")
        self.base = base
        # duck-body's unsure pickup states (the legs light / swinging free), set by the body
        # side each tick before :meth:`step` (the contract has no field for them yet)
        self.light = False
        self.skills = SoundSkills(base.genome, base.seed)
        # the temperament gains of the new channels (select.Selector keeps its own copy)
        sel = base.sel
        for index, name in enumerate(sel.names):
            if name in self.skills.gains:
                sel._gain[index] = self.skills.gains[name]
        self.last: SoundOut | None = None
        self.felt_head: tuple[float, ...] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def step(self, t_ms: int, frame: SensorFrame) -> tuple[MotorCommand, C.BrainTelemetry]:
        base = self.base
        d = base.drives
        self.felt_head = frame.body.head  # (its joint sense this tick: the evaluators' record)
        rx = getattr(base, "_reflex", None)
        active = getattr(rx, "active", None)
        fright = active is not None and getattr(active, "reflex", "") in ("startle", "withdraw")
        from duckmind.neuro import select as S

        feat = base._feat
        drives: dict[str, float] = {
            name: float(feat[S.F[name]])
            for name in ("play", "curiosity", "social", "happy", "music", "arousal")
            if name in S.F and S.F[name] < len(feat)
        }
        seen, features = self.skills.pre(
            t_ms,
            frame,
            behaviour=str(base.behaviour),
            play=float(d.play),
            fear=float(d.fear),
            arousal=float(d.arousal),
            asleep=str(base.behaviour) in _sleep_acts(),
            fright=fright,
            light=self.light,
            drives=drives,
        )
        if len(feat) < len(S.FEATURES):  # (a brain made before install: grow its vector)
            feat.extend([0.0] * (len(S.FEATURES) - len(feat)))
        for name, value in features.items():
            feat[S.F[name]] = float(value)
        seq_before = int(base._vocal_seq)
        cmd, tel = base.step(t_ms, seen)
        # the skills' own calls count on the brain's counter (every call a new one)
        out_cmd, out = self.skills.post(cmd, bout=int(base.bout), vocal_seq=seq_before)
        if out_cmd.vocal_seq > base._vocal_seq:
            base._vocal_seq = out_cmd.vocal_seq
        ride = out.ride
        if ride.started:
            # the ride's start, felt by the slower systems: a fright is a threat (the fear
            # drive, once — not a spiral), a thrill lifts the mood and the wish to play
            if ride.frightened:
                d.fear = min(1.0, d.fear + 0.15)
            else:
                d.valence = min(1.0, d.valence + 0.1)
                d.play = min(1.0, d.play + 0.1)
        self.last = out
        if out_cmd.behaviour != cmd.behaviour:
            tel = replace(
                tel,
                behaviour=out_cmd.behaviour,
                behaviour_zh=C.behaviour_zh(out_cmd.behaviour),
                reason_zh=msg("neuro.incoming.incoming_skill_sound.SkillBrain.step"),
            )
        return out_cmd, tel


def _sleep_acts() -> frozenset[str]:
    return frozenset(b for b, spec in C.BEHAVIOURS.items() if spec.sleep)


def phase_of(out: SoundOut | None) -> float:
    """The dance's nod phase now (0 = the head at its lowest), for evaluators."""
    if out is None or not out.beat.active:
        return math.nan
    n = SD.beats_per_bob(out.beat.hz, "stand")
    return ((out.beat.beats % n) + out.beat.phase) / n


__all__ = [
    "BEHAVIOURS",
    "BUILDERS",
    "EXTRA_WEIGHTS",
    "PULL",
    "FEATURES",
    "FEATURE_ZH",
    "SATIATE_S",
    "SOUND_SKILLS_VERSION",
    "SkillBrain",
    "SoundOut",
    "SoundSkills",
    "WEIGHTS",
    "gains_for",
    "install",
    "pulled",
    "installed",
]
