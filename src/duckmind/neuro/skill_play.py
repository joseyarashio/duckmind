"""a:play — ball play, carrying a toy home and sunbathing on the neural brain (2026-09-28).

The owner chose 「A B 全數」 (2026-09-28 ~19:50): among them 踢球推球, 叼玩具回窩, 曬太陽 — each to
*emerge* from the neural brain (drives, senses, learning, temperament), be performed by the
physical body and seen in the 3D view, decided only from the duck's own senses, with falls as
real outcomes. This module brings the three acts to the brain:

* :class:`PlayMind` — what they share: the duck's own memory of things (:class:`~duckmind.neuro.
  skill_common.ThingBook`: what rolled, what stayed in its beak), its light sense
  (:class:`~duckmind.neuro.skill_sun.SunSense`), two new urges (``nest_urge``, ``bask``), the
  competition's new inputs (:data:`PLAY_FEATURES`) and the three acts' own step machines
  (:mod:`~duckmind.neuro.skill_ball`, :mod:`~duckmind.neuro.skill_carry`,
  :mod:`~duckmind.neuro.skill_sun`).
* :class:`PlayBrain` — :class:`~duckmind.neuro.brain.NeuroBrain` with the three acts wired in
  through its own hooks (``_goal``, ``_consumed``, ``_start_bout``, ``_look_before``): the
  features go into the basal-ganglia competition like every other input; while a play act is
  chosen its step machine says where the legs, the head, the beak and robotd's one-shots go
  (:attr:`PlayBrain.play_out`). A holding reflex still takes the body first; orient/lean-in may keep compatible leg intent.
* :func:`play_installed` — the shared tables' new entries (behaviours, the competition's wiring,
  the walking styles, the motor programs, the temperament gains), added for as long as the
  context lasts and removed after (the build phase writes only new files: the integration puts
  exactly these entries into ``contract.py``, ``select.py``, ``brain.py``, ``repertoire.py``,
  ``genome.py`` — see :data:`INTEGRATION`).

Pure: the brain's own modules only (never :mod:`duckmind.world`).
"""

from __future__ import annotations

import contextlib
import math
from collections import deque
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

from duckmind.i18n import msg
from duckmind.neuro import brain as B
from duckmind.neuro import contract as C
from duckmind.neuro import genome as G
from duckmind.neuro import repertoire as P
from duckmind.neuro import select as S
from duckmind.neuro.skill_ball import BallPlay, BallStyle, ball_features
from duckmind.neuro.skill_carry import CarryHome, carry_features
from duckmind.neuro.skill_common import IDLE, OdomHistory, PlayOut, ThingBook, bearing_range
from duckmind.neuro.skill_sun import SunBask, SunSense, sun_features

PLAY_VERSION = "neuro-play-1"
PLAY_ACTS = ("play_ball", "carry_toy", "sunbathe")
# These reflexes own the head but do not halt the legs; only they may coexist with a
# play act's locomotor side channel. Veer/stabilise and every holding reflex keep ownership.
PLAY_REFLEX_COEXIST = frozenset({"orient", "lean_in"})

# ── the shared tables' new entries (the integration adds exactly these) ─────────

PLAY_BEHAVIOURS: dict[str, C.BehaviourSpec] = {
    "play_ball": C.BehaviourSpec(
        msg("neuro.act.play_ball.name"),
        "ethogram",
        5,
        "stand",
        True,
        ("", "chirp", "wheee"),
        msg("neuro.act.play_ball.body"),
    ),
    "carry_toy": C.BehaviourSpec(
        msg("neuro.act.carry_toy.name"),
        "ethogram",
        14,
        "stand",
        True,
        ("",),
        msg("neuro.act.carry_toy.body"),
        stationary=True,  # (it goes round the toy and lines up on it, on the spot)
    ),
    "sunbathe": C.BehaviourSpec(
        msg("neuro.act.sunbathe.name"),
        "rest",
        0,
        "any",
        True,
        ("", "coo"),
        msg("neuro.act.sunbathe.body"),
        stationary=True,  # (into the light, a look down, a little further in: all inside the patch)
    ),
}
PLAY_FEATURES: tuple[str, ...] = (
    "ball",
    "ball_rolling",
    "ball_fun",
    "toy",
    "carrying",
    "nest_urge",
    "sunny",
    "in_sun",
    "bask",
    "ball_busy",
    "toy_busy",
)
PLAY_FEATURE_ZH = {
    "ball": msg("neuro.why.ball"),
    "ball_rolling": msg("neuro.why.ball_rolling"),
    "ball_fun": msg("neuro.why.ball_fun"),
    "toy": msg("neuro.why.toy"),
    "carrying": msg("neuro.why.carrying"),
    "nest_urge": msg("neuro.why.nest_urge"),
    "sunny": msg("neuro.why.sunny"),
    "in_sun": msg("neuro.why.in_sun"),
    "bask": msg("neuro.why.bask"),
    "ball_busy": msg("neuro.why.ball_busy"),
    "toy_busy": msg("neuro.why.toy_busy"),
}
# The phases in which an act is in the middle of something it would spoil by stopping (lined up
# on the ball, the foot about to kick; round the toy, the beak on it, carrying it home): the act's
# own "I am in the middle of it" input — like the docking program's ``docking`` input — so a
# passing whim does not break off a kick or a carry halfway. Its weight is the act's alone.
BALL_BUSY_PHASES = frozenset({"line_up", "kick", "beak", "chest"})
CARRY_BUSY_PHASES = frozenset(
    {"survey", "around", "line_up", "grab", "backoff", "carry", "search", "drop", "regrip"}
)
PLAY_W: dict[str, dict[str, float]] = {
    # the stimulus features already carry their urge (PlayMind.sense: a ball × the play drive, a
    # toy × the nesting urge, the sun × the basking urge), so an act never wins on its urge alone
    "play_ball": {
        "ball": 2.0,  # a ball it knows, near: the undirected walk gives way (a play releaser)
        "ball_rolling": 1.6,  # a ball rolling by is a strong releaser of play (and of a chase)
        "ball_fun": 0.3,
        "ball_busy": 0.8,
        "play": 0.08,
        "fear": -0.6,
        "sleepy": -0.5,
        "carrying": -1.0,
        "cornered": -0.5,
        "dark": -0.5,
    },
    "carry_toy": {
        "toy": 1.8,  # (× the nesting urge already: a snuggly duck ~1.0, a dreamy one ~0.65)
        "carrying": 1.4,
        "toy_busy": 1.0,
        "nest_urge": 0.2,
        "fear": -0.4,
        "sleepy": -0.3,
        "dark": -0.5,
        "cornered": -0.4,
    },
    "sunbathe": {
        "sunny": 0.9,
        "in_sun": 1.0,
        "bask": 0.15,
        "calm": 0.12,
        "sleepy": 0.12,
        "fear": -0.5,
        "restless": -0.25,
        "night": -0.8,
        "carrying": -1.0,
        "ball_rolling": -0.3,
    },
}
# existing acts that the new inputs also reach (added to their wiring)
PLAY_CROSS_W: dict[str, dict[str, float]] = {
    "doze": {"in_sun": 0.3},  # a warm spot makes it drowsy
    "rest_sit": {"in_sun": 0.2},
    "watch": {"ball_rolling": 0.35},  # a timid duck watches the ball roll
    "investigate": {"toy": 0.1},
}
PLAY_SATIATE_S = {"play_ball": 45.0, "carry_toy": 150.0, "sunbathe": 600.0}
PLAY_LOCO = {
    "play_ball": (C.NORMAL_VX_MAX, 0.35, 0.4),  # a chase: eager, fairly straight
    "carry_toy": (C.APPROACH_VX, 0.25, 0.3),
    "sunbathe": (C.APPROACH_VX, 0.1, 0.6),  # an unhurried walk to the sun
}
PLAY_EXERTION = {"play_ball": 0.5, "carry_toy": 0.35, "sunbathe": 0.0}
PLAY_FIT = {"play_ball": "curious", "carry_toy": "calm", "sunbathe": "tired"}


def play_gain(behaviour: str, axes: dict[str, float], stage: str, lean: Any) -> float:
    """The temperament's lean on each act (how likely, never whether): bold, lively ducks play
    ball more (hatchlings chase more); sociable, calm, snuggly ducks bring things home; calm,
    dreamy or snuggly ducks bask."""
    bold, act, soc = axes["boldness"], axes["activity"], axes["sociability"]
    young = {"hatchling": 1.0, "duckling": 0.6, "juvenile": 0.25, "adult": 0.0}.get(stage, 0.25)
    if behaviour == "play_ball":
        return (
            (0.55 + 0.9 * bold)
            * (0.85 + 0.3 * act)
            * (1.0 + 0.3 * young)
            * (1.0 + 0.25 * lean("explorer"))
        )
    if behaviour == "carry_toy":
        clumsy = 0.6 if young >= 1.0 else 1.0
        return (0.6 + 0.5 * soc + 0.3 * (1.0 - act)) * (1.0 + 0.5 * lean("snuggler")) * clumsy
    if behaviour == "sunbathe":
        return (0.6 + 0.6 * (1.0 - act)) * (1.0 + 0.4 * lean("dreamer") + 0.3 * lean("snuggler"))
    return 1.0


# ── the motor programs (repertoire builders) ──────────────────────────────────


def build_play_ball(b: Any, ctx: Any) -> Any:
    if ctx.mode == "walk":
        b.to(b.u(250, 400), np=b.u(0.15, 0.3), hp=b.u(-0.35, -0.2))  # eyes on the ball ahead
        b.hold(b.u(500, 1100), drift=0.0)
        return b.score(label="追球", gaze=1.0, alive=0.5)
    b.to(b.u(250, 400), np=b.u(0.3, 0.5), hp=b.u(-0.6, -0.4))  # a good look down at it
    if b.p(0.4):
        b.rel(b.u(250, 380), dhr=b.sign() * b.u(0.3, 0.5))  # a little tilt: which way?
    b.hold(b.u(600, 1300), drift=0.0)
    return b.score(label="盯著球", gaze=0.9, alive=0.4)


def build_carry_toy(b: Any, ctx: Any) -> Any:
    if ctx.mode == "walk":
        b.to(b.u(300, 450), np=0.0, hp=b.u(0.0, 0.1), hy=0.0, hr=0.0)
        b.hold(b.u(1500, 2500), drift=0.0)
        return b.score(label="叼著走", gaze=0.1, alive=0.15)
    b.to(b.u(300, 450), np=b.u(0.2, 0.35), hp=b.u(-0.5, -0.3))
    b.hold(b.u(700, 1400), drift=0.0)
    return b.score(label="低頭看小玩具", gaze=0.8, alive=0.3)


def build_sunbathe(b: Any, ctx: Any) -> Any:
    if ctx.mode == "walk":
        b.to(b.u(350, 500), np=b.u(0.05, 0.15), hp=b.u(-0.2, -0.1))
        b.hold(b.u(900, 1500))
        return b.score(label="走向太陽", gaze=0.6, alive=0.5)
    # seated in the sun: the head sinks low and relaxed, the face a little up toward the light,
    # held long; now and then a slow drowsy dip (no eyelids on this body: the carriage says it)
    b.to(b.u(900, 1400), np=b.u(0.3, 0.45), hp=b.u(0.1, 0.25), hr=b.sign() * b.u(0.0, 0.15))
    b.hold(b.u(3500, 7000), sigma=0.35, drift=0.02)
    if b.p(0.6):
        b.rel(b.u(1100, 1500), dhp=-b.u(0.18, 0.28), ease="soft")
        b.hold(b.u(600, 1400), drift=0.0)
        b.rel(b.u(1300, 1800), dhp=b.u(0.15, 0.25), ease="soft")
    b.hold(b.u(2500, 5000), sigma=0.35, drift=0.02)
    if b.p(0.15):
        b.vocal(b.tag(("", "coo"), silent=2.0), 0.3)
    return b.score(label="瞇著眼曬太陽", gaze=0.15, alive=0.2)


PLAY_BUILDERS = {
    "play_ball": build_play_ball,
    "carry_toy": build_carry_toy,
    "sunbathe": build_sunbathe,
}

# ── installing the entries (reversible; tests and the headless rig) ───────────


def _installed_now() -> bool:
    return all(b in C.BEHAVIOURS for b in PLAY_ACTS) and all(f in S.F for f in PLAY_FEATURES)


_PERMANENT_INSTALLED = False
_LEGACY_CONTEXTS = 0


def legacy_context_active() -> bool:
    """Whether a reversible play fixture currently owns the shared tables."""
    return _LEGACY_CONTEXTS > 0


def install_permanent() -> None:
    """Install the play contribution for a process-wide integrated registry.

    Unlike :func:`play_installed`, this is deliberately one-way: production brains may already
    hold selector state built from these tables, so the integration registry owns this lifetime.
    """
    global _PERMANENT_INSTALLED
    if _PERMANENT_INSTALLED:
        return
    _apply_play_tables()
    _PERMANENT_INSTALLED = True


def _apply_play_tables() -> None:
    """Apply play entries without taking a reversible snapshot (integration path)."""
    C.BEHAVIOURS.update(PLAY_BEHAVIOURS)
    if not all(f in S.F for f in PLAY_FEATURES):
        S.FEATURES = tuple(S.FEATURES) + tuple(f for f in PLAY_FEATURES if f not in S.F)
        S.F.clear()
        S.F.update({name: index for index, name in enumerate(S.FEATURES)})
    S.FEATURE_ZH.update(PLAY_FEATURE_ZH)
    for act, weights in PLAY_W.items():
        S.W[act] = dict(weights)
    for act, weights in PLAY_CROSS_W.items():
        if act in S.W:
            S.W[act] = {**S.W[act], **weights}
    S.SATIATE_S.update(PLAY_SATIATE_S)
    S.LOCOMOTES = S.LOCOMOTES | frozenset(PLAY_ACTS)
    B.LOCO.update(PLAY_LOCO)
    B.EXERTION.update(PLAY_EXERTION)
    B.BEAK_BUSY = B.BEAK_BUSY | frozenset({"carry_toy"})
    B.HEAD_ACTS = B.HEAD_ACTS | frozenset({"carry_toy", "sunbathe"})
    B.TURN_TOWARD_NEVER = B.TURN_TOWARD_NEVER | frozenset({"carry_toy", "sunbathe"})
    P.BUILDERS.update(PLAY_BUILDERS)
    P.REACTIVE = P.REACTIVE | frozenset(PLAY_ACTS)
    G.ETHOGRAM_FIT.update(PLAY_FIT)
    if not getattr(install_permanent, "_gains_installed", False):
        gains_before = G._gains

        def gains(a: Any, stage: str, style: Any, branch: Any, primary: str) -> dict[str, float]:
            out = gains_before(a, stage, style, branch, primary)
            for act in PLAY_ACTS:
                if act in out:
                    out[act] *= play_gain(act, dict(a), stage, style.lean)
            return out

        G._gains = gains
        install_permanent._gains_installed = True  # type: ignore[attr-defined]


@contextlib.contextmanager
def play_installed() -> Iterator[None]:
    """The three acts in the shared tables for as long as the context lasts (see the module doc).
    A genome and a brain made inside it have them; everything is restored on exit. When the
    entries are already there (after the integration) it does nothing."""
    global _LEGACY_CONTEXTS
    if _installed_now():
        yield
        return
    saved: list[tuple[Any, str, Any]] = []

    def setattr_saved(module: Any, name: str, value: Any) -> None:
        saved.append((module, name, getattr(module, name)))
        setattr(module, name, value)

    behaviours_before = dict(C.BEHAVIOURS)
    w_before = {k: dict(v) for k, v in S.W.items()}
    satiate_before = dict(S.SATIATE_S)
    zh_before = dict(S.FEATURE_ZH)
    f_before = dict(S.F)
    loco_before = dict(B.LOCO)
    exertion_before = dict(B.EXERTION)
    builders_before = dict(P.BUILDERS)
    fit_before = dict(G.ETHOGRAM_FIT)
    _LEGACY_CONTEXTS += 1
    try:
        C.BEHAVIOURS.update(PLAY_BEHAVIOURS)
        setattr_saved(S, "FEATURES", S.FEATURES + PLAY_FEATURES)
        S.F.clear()
        S.F.update({name: index for index, name in enumerate(S.FEATURES)})
        S.FEATURE_ZH.update(PLAY_FEATURE_ZH)
        for act, weights in PLAY_W.items():
            S.W[act] = dict(weights)
        for act, weights in PLAY_CROSS_W.items():
            if act in S.W:
                S.W[act] = {**S.W[act], **weights}
        S.SATIATE_S.update(PLAY_SATIATE_S)
        setattr_saved(S, "LOCOMOTES", S.LOCOMOTES | frozenset(PLAY_ACTS))
        B.LOCO.update(PLAY_LOCO)
        B.EXERTION.update(PLAY_EXERTION)
        setattr_saved(B, "BEAK_BUSY", B.BEAK_BUSY | frozenset({"carry_toy"}))
        setattr_saved(B, "HEAD_ACTS", B.HEAD_ACTS | frozenset({"carry_toy", "sunbathe"}))
        setattr_saved(
            B, "TURN_TOWARD_NEVER", B.TURN_TOWARD_NEVER | frozenset({"carry_toy", "sunbathe"})
        )
        P.BUILDERS.update(PLAY_BUILDERS)
        setattr_saved(P, "REACTIVE", P.REACTIVE | frozenset(PLAY_ACTS))
        G.ETHOGRAM_FIT.update(PLAY_FIT)
        gains_before = G._gains

        def gains(a: Any, stage: str, style: Any, branch: Any, primary: str) -> dict[str, float]:
            out = gains_before(a, stage, style, branch, primary)
            for act in PLAY_ACTS:
                if act in out:
                    out[act] *= play_gain(act, dict(a), stage, style.lean)
            return out

        setattr_saved(G, "_gains", gains)
        yield
    finally:
        _LEGACY_CONTEXTS -= 1
        for module, name, value in reversed(saved):
            setattr(module, name, value)
        C.BEHAVIOURS.clear()
        C.BEHAVIOURS.update(behaviours_before)
        S.W.clear()
        S.W.update(w_before)
        S.SATIATE_S.clear()
        S.SATIATE_S.update(satiate_before)
        S.FEATURE_ZH.clear()
        S.FEATURE_ZH.update(zh_before)
        S.F.clear()
        S.F.update(f_before)
        B.LOCO.clear()
        B.LOCO.update(loco_before)
        B.EXERTION.clear()
        B.EXERTION.update(exertion_before)
        P.BUILDERS.clear()
        P.BUILDERS.update(builders_before)
        G.ETHOGRAM_FIT.clear()
        G.ETHOGRAM_FIT.update(fit_before)


# ── the play mind ─────────────────────────────────────────────────────────────

NEST_TAU_S = 1200.0
PLAY_TIME_S = 240.0  # four minutes of ball play use the play drive up by time alone
ROLLING_TAU_S = 4.0
BASK_TAU_S = 900.0
BASK_USE_S = 240.0  # four minutes in the sun use up a full urge
NEST_USE = 0.7  # a toy brought home
# A trace for expected act sounds. SensorFrame has no causal event identity, so it never permits
# the brain to hide an onset from its reflexes.
OWN_SOUND_MS = 1200
OWN_SOUND_EVENTS = frozenset(
    {"slipped", "dropped_home", "grab", "grabbed", "kicked", "pushed", "regrip"}
)
OWN_SOUND_PHASES = frozenset({"grab", "drop", "kick", "beak", "chest", "regrip"})


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


class PlayMind:
    """The three acts' shared senses, memories, urges and step machines (one per duck)."""

    def __init__(self, genome: G.Genome, seed: int = 0, *, turn_side: str = "right") -> None:
        axes = dict(genome.axes)
        young = genome.p.hatchling_wobble
        self.genome = genome
        self.hist = OdomHistory()
        self.book = ThingBook()
        self.sun = SunSense()
        self.ball = BallPlay(style=BallStyle.of(axes, young), seed=seed)
        self.carry = CarryHome(seed=seed, turn_side=turn_side)
        self.bask_ctl = SunBask()
        soc = axes.get("sociability", 0.5)
        act = axes.get("activity", 0.5)
        lean = genome.nature.get
        snug = 1.0 if genome.branch == "snuggler" else lean("snuggler", 0.25)
        dream = 1.0 if genome.branch == "dreamer" else lean("dreamer", 0.25)
        self.nest_gain = 0.3 + 0.4 * soc + 0.3 * snug
        self.bask_gain = 0.3 + 0.35 * (1.0 - act) + 0.35 * max(snug, dream)
        self.nest_urge = 0.2
        self.bask = 0.2
        self.out: PlayOut = IDLE
        self.act: str | None = None
        self.mouth_deg: float | None = None
        self.home_spot: tuple[float, float] | None = None
        self.consumed = 0.0
        self.events: deque[dict[str, Any]] = deque(maxlen=64)
        self.counts: dict[str, int] = {}
        self.legs = False  # the act holds the legs (its own twist): the brain's rhythm rests
        self._first = True
        self.rolling = 0.0  # a leaky trace of a ball seen rolling
        self.own_until = -1  # its own act's sounds are expected until then (brain ms)

    # senses and urges

    def sense(self, t_ms: int, dt: float, frame: C.SensorFrame, brain: Any) -> dict[str, float]:
        odom = frame.body.odom
        self.hist.add(t_ms, odom)
        # (a one-shot under way — its skill sent this act, or robotd still running it — the body
        # dips: the camera's passes then are not used for where things lie)
        dipped = self.out.skill is not None or frame.body.posture not in ("stand", "walk", "sit")
        self.book.see(t_ms, frame, self.hist, dipped=dipped)
        if self.ball.settle_pending(t_ms, self.book):
            # a push it could not watch end, seen now to have gone (whatever act runs now)
            self.counts["rolled"] = self.counts.get("rolled", 0) + 1
            self.events.append({"t_ms": t_ms, "act": "play_ball", "event": "rolled"})
            self.consumed += 0.08 if brain.behaviour == "play_ball" else 0.0
        self.sun.step(t_ms, dt, frame, self.hist)
        joint = getattr(frame.body, "mouth", None)
        self.mouth_deg = None if joint is None else math.degrees(float(joint) + math.radians(5.0))
        self.home_spot = self._home(odom, brain)
        d = brain.drives
        asleep = brain.behaviour in S.SLEEP
        if self._first:
            # at boot the urges start where the day puts them (awake since morning)
            self._first = False
            day = _clip((d.alertness - 0.45) / 0.3)
            self.nest_urge = 0.7 * self.nest_gain
            self.bask = self.bask_gain * day * 0.8
        if not asleep:
            target = self.nest_gain * (0.4 + 0.6 * d.energy)
            self.nest_urge += (target - self.nest_urge) * (1.0 - math.exp(-dt / NEST_TAU_S))
            day = _clip((d.alertness - 0.45) / 0.3)
            calm = _clip(1.0 - 1.4 * d.arousal)
            target = self.bask_gain * day * (0.5 + 0.5 * calm)
            self.bask += (target - self.bask) * (1.0 - math.exp(-dt / BASK_TAU_S))
        if brain.behaviour == "sunbathe" and self.sun.in_sun() and frame.body.posture == "sit":
            self.bask = max(0.0, self.bask - dt / BASK_USE_S)
        feats: dict[str, float] = {}
        feats.update(ball_features(self.book, odom, t_ms))
        feats.update(carry_features(self.book, odom, t_ms, self.home_spot, self.carry.holding))
        feats.update(sun_features(self.sun, t_ms, odom, frame.day_minute))
        # a ball seen rolling holds its attention a few seconds after it left the picture
        self.rolling = max(self.rolling * math.exp(-dt / ROLLING_TAU_S), feats["ball_rolling"])
        feats["ball_rolling"] = self.rolling
        # each stimulus carries its urge (a product, as a gate: no ball, no ball play)
        play = _clip(d.play)
        # (chasing a ball is walking too: the urge to walk adds to the pull of a ball near)
        feats["ball"] *= 0.4 + 0.4 * play + 0.3 * _clip(d.restless)
        feats["ball_rolling"] *= 0.5 + 0.5 * play
        feats["ball_fun"] *= min(1.0, feats["ball"] * 2.0)
        feats["toy"] *= 0.3 + 0.7 * _clip(self.nest_urge)
        feats["sunny"] *= 0.3 + 0.7 * _clip(self.bask)
        feats["in_sun"] *= 0.5 + 0.5 * _clip(self.bask)
        feats["nest_urge"] = _clip(self.nest_urge)
        feats["bask"] = _clip(self.bask)
        running = brain.behaviour
        feats["ball_busy"] = (
            1.0 if running == "play_ball" and self.ball.phase in BALL_BUSY_PHASES else 0.0
        )
        feats["toy_busy"] = (
            1.0 if running == "carry_toy" and self.carry.phase in CARRY_BUSY_PHASES else 0.0
        )
        if self.home_spot is None:
            feats["toy"] = 0.0  # no home known: nothing to carry it to
        return feats

    def _home(self, odom: tuple[float, float, float], brain: Any) -> tuple[float, float] | None:
        """The spot beside its dock (its odometry), from the dock's frame it remembers."""
        dock = getattr(brain, "dock", None)
        pose = None
        if dock is not None:
            try:
                p = dock.pose(tuple(v * (1000.0 if i < 2 else 1.0) for i, v in enumerate(odom)))
            except Exception:  # (the dock lane's API may move under us: no home then)
                p = None
            if p is not None:
                pose = (float(p.x), float(p.y), float(p.h))
        if pose is not None:
            self.carry.turn_side = getattr(getattr(dock, "cfg", None), "turn_side", "right")
        return self.carry.drop_spot(pose, odom)

    # the act

    def start(self, act: str, t_ms: int) -> None:
        self.act = act
        if act == "play_ball":
            self.ball.start(t_ms)
        elif act == "carry_toy":
            self.carry.start(t_ms)
        elif act == "sunbathe":
            self.bask_ctl.start(t_ms)

    def goal(self, act: str, odom: tuple[float, float, float]) -> tuple[float, float] | None:
        """Where the brain's own walk should go for this act now (bearing, range), or None."""
        out = self.out
        if act != self.act or not out.far or out.target is None:
            return None
        return float(out.target.bearing), float(out.target.range_m or 1.0)

    def control(
        self, act: str, t_ms: int, dt: float, frame: C.SensorFrame, cmd: C.MotorCommand
    ) -> PlayOut:
        odom = frame.body.odom
        body = frame.body
        halted = (
            (cmd.reflex is not None and cmd.reflex not in PLAY_REFLEX_COEXIST)
            or cmd.behaviour != act
            or body.fallen
            or body.posture in ("transition", "unknown")
            and self.out.skill is None
        )
        standing = body.posture in ("stand", "walk")
        if act == "play_ball":
            out = self.ball.step(t_ms, frame, self.book, odom, [], halted=halted, standing=standing)
            event = self.ball.last_event or out.event
        elif act == "carry_toy":
            out = self.carry.step(
                t_ms,
                frame,
                self.book,
                odom,
                self.home_spot,
                self.mouth_deg,
                halted=halted,
                standing=standing,
                distracted=cmd.reflex is not None,
            )
            event = self.carry.last_event or out.event
        elif act == "sunbathe":
            out = self.bask_ctl.step(t_ms, dt, frame, self.sun, odom, halted=halted)
            event = self.bask_ctl.last_event or out.event
        else:
            out, event = IDLE, None
        if event:
            self.counts[event] = self.counts.get(event, 0) + 1
            self.events.append({"t_ms": t_ms, "act": act, "event": event})
            self._consume(act, event, frame)
        if (event in OWN_SOUND_EVENTS) or out.phase in OWN_SOUND_PHASES:
            self.own_until = t_ms + OWN_SOUND_MS
        if (
            act == "sunbathe"
            and not self.sun.in_sun()
            and self.sun.best(t_ms, odom, frame.day_minute) is None
        ):
            self.consumed += dt / 5.0  # no sun anywhere it knows: the urge has nowhere to go
        self.out = out
        self.legs = out.drive and not out.far
        return out

    def _consume(self, act: str, event: str, frame: C.SensorFrame) -> None:
        if event == "dropped_home":
            self.nest_urge = max(0.0, self.nest_urge - NEST_USE)
            self.consumed += 1.0
        elif event.startswith("gave_up") or event == "no_home":
            self.consumed += 0.5
        elif event == "lost":
            self.consumed += 0.4
        elif event == "rolled":
            self.consumed += 0.08

    def play_used(self, act: str, dt: float, event: str | None) -> float:
        """How much of the play drive this tick used up: ball play is used up by what happens —
        the ball rolling away after its push (the fun of it), each push or kick a little — and
        only slowly by the time spent (the brain's other play acts use it up by time, 45 s)."""
        if act != "play_ball":
            return 0.0
        used = dt / PLAY_TIME_S
        if event == "rolled":
            used += 0.08
        elif event in ("kicked", "pushed"):
            used += 0.03
        return used

    def take_consumed(self) -> float:
        value, self.consumed = self.consumed, 0.0
        return value

    def view(self) -> dict[str, Any]:
        return {
            "version": PLAY_VERSION,
            "act": self.act,
            "phase": self.out.phase,
            "zh": self.out.zh,
            "nest_urge": round(self.nest_urge, 3),
            "bask": round(self.bask, 3),
            "mouth_deg": None if self.mouth_deg is None else round(self.mouth_deg, 1),
            "home_mm": None
            if self.home_spot is None
            else [round(self.home_spot[0] * 1000), round(self.home_spot[1] * 1000)],
            "things": self.book.view(),
            "sun": self.sun.view(),
            "ball": self.ball.view(),
            "carry": self.carry.view(),
            "bask_act": self.bask_ctl.view(),
            "events": dict(self.counts),
        }


# ── the brain with the play acts ──────────────────────────────────────────────


class PlayBrain(B.NeuroBrain):
    """:class:`~duckmind.neuro.brain.NeuroBrain` with ball play, carrying a toy home and
    sunbathing (see the module doc). Make it (and its genome) inside :func:`play_installed`
    until the integration has put the entries into the shared tables."""

    def __init__(self, genome: G.Genome, seed: int = 0, **kw: Any) -> None:
        if not _installed_now():
            raise RuntimeError("PlayBrain needs the play entries: build it inside play_installed()")
        super().__init__(genome, seed, **kw)
        turn = getattr(getattr(getattr(self, "dock", None), "cfg", None), "turn_side", "right")
        self.play = PlayMind(genome, seed, turn_side=turn)
        self.play_out: PlayOut = IDLE
        self._play_t: int | None = None
        self._t_now = 0

    def _adopt_params(self, fresh: B.NeuroBrain) -> None:
        """(Growth G3, a regrow; see ``NeuroBrain._adopt_params``.) The play acts' genome, the
        style of its ball play and how it leans to the nest and the sun follow the genome; what it
        saw, carried and pushed stays."""
        super()._adopt_params(fresh)
        assert isinstance(fresh, PlayBrain)
        self.play.genome = fresh.play.genome
        self.play.ball.style = fresh.play.ball.style
        self.play.nest_gain = fresh.play.nest_gain
        self.play.bask_gain = fresh.play.bask_gain

    # ── the step ─────────────────────────────────────────────────────────────

    def step(self, t_ms: int, frame: C.SensorFrame) -> tuple[C.MotorCommand, C.BrainTelemetry]:
        dt = 0.05 if self._play_t is None else _clip((t_ms - self._play_t) / 1000.0, 0.0, 2.0)
        self._play_t = t_ms
        feats = self.play.sense(t_ms, dt, frame, self)
        self._t_now = t_ms
        frame = self._reafference(frame)
        F = S.F
        for name, value in feats.items():
            self._feat[F[name]] = value
        cmd, tel = super().step(t_ms, frame)
        act = self.behaviour
        if act not in PLAY_ACTS:
            self.play_out = IDLE
            self.play.legs = False
            self.play.act = (
                None if self.play.act in PLAY_ACTS and act != self.play.act else self.play.act
            )
            return cmd, self._tel(tel, None)
        out = self.play.control(act, t_ms, dt, frame, cmd)
        # The runtime consumes this side channel for fine steps, mouth and one-shots. Keep
        # only the leg/target portion through the two head-only reflexes; clear everything else.
        self.play_out = self._reflex_safe_out(out, cmd, act)
        used = self.play.play_used(act, dt, self.play.ball.last_event or out.event)
        if used:
            self.drives.play = max(0.0, self.drives.play - used)
        return self._merge(cmd, out, act), self._tel(tel, out)

    @staticmethod
    def _reflex_safe_out(out: PlayOut, cmd: C.MotorCommand, act: str) -> PlayOut:
        """Keep compatible leg intent through a head-only reflex, with reflex ownership intact.

        One-shots are never exposed while any reflex is active. Head, gaze, mouth, posture,
        vocal and moment control remain with the reflex/program; only locomotion and its target
        may coexist with orienting or leaning into a pet.
        """
        if cmd.behaviour != act:
            return IDLE
        if cmd.reflex is None:
            return out
        if cmd.reflex not in PLAY_REFLEX_COEXIST:
            return IDLE
        return replace(
            out,
            head=None,
            head_blend=0.0,
            gaze=None,
            mouth=None,
            skill=None,
            skill_seq=0,
            posture=None,
            vocal=None,
            quiet=False,
        )

    def _reafference(self, frame: C.SensorFrame) -> C.SensorFrame:
        """Keep every sensed event visible to the reflexes.

        The contract has no causal identifier linking a touch or sound to the duck's play act;
        suppressing expected contact could also hide an external poke on the same tick.
        """
        # The current SensorFrame has no causal identity linking a touch/sound to this duck's
        # play act. Preserve the complete sensory frame: masking a ball impact can also hide a
        # hand poke or a clap arriving on the same tick. OWN_SOUND_MS remains a play-state trace
        # for evaluators, never permission to suppress reflex input.
        return frame

    def _merge(self, cmd: C.MotorCommand, out: PlayOut, act: str) -> C.MotorCommand:
        reflex_coexists = cmd.reflex is None or cmd.reflex in PLAY_REFLEX_COEXIST
        held = cmd.behaviour != act or not reflex_coexists
        if held:
            return cmd  # reflex / selected act owns the whole motor output this tick
        changes: dict[str, Any] = {}
        if out.drive and not out.far:
            if out.reverse:
                vx, vyaw = C.legal_walk(out.vx, out.vyaw, back_arc=True)
            else:
                vx, vyaw = out.vx, out.vyaw
            changes.update(vx=vx, vyaw=vyaw, approach=vx > 0.0, back_arc=vx < 0.0)
        elif out.far and out.target is not None and (out.target.range_m or 9.0) < 0.6:
            changes["approach"] = cmd.vx > 0.0
        if cmd.reflex is None:
            if out.head is not None:
                changes.update(head=out.head, head_blend=out.head_blend)
            if out.gaze is not None:
                changes["gaze"] = out.gaze
            if out.posture in ("sit", "stand"):
                changes["posture"] = out.posture
        if out.target is not None:
            changes["target"] = out.target
        if out.vocal and cmd.reflex is None:
            self._vocal_seq += 1
            changes.update(vocal=out.vocal, vocal_seq=self._vocal_seq)
        if cmd.reflex is None and out.quiet and cmd.moment is not None:
            changes["moment"] = None  # (the motor records the seq and starts nothing)
        if act == "carry_toy" and self.play.carry.holding and cmd.reflex is None:
            changes["vocal"] = None  # a beak full of toy makes no call
        return replace(cmd, **changes) if changes else cmd

    def _tel(self, tel: C.BrainTelemetry, out: PlayOut | None) -> C.BrainTelemetry:
        learned = dict(tel.learned)
        learned["play"] = self.play.view()
        reason = tel.reason_zh
        if out is not None and out.zh and tel.behaviour in PLAY_ACTS:
            reason = msg(
                "neuro.incoming.incoming_skill_play.PlayBrain._tel", reason=reason, zh=out.zh
            )
        return replace(tel, learned=learned, reason_zh=reason)

    # ── the brain's own hooks ────────────────────────────────────────────────

    def _goal(self, beh: str, pct: Any, odom: tuple[float, float, float], *args: Any) -> Any:
        if beh in PLAY_ACTS:
            got = self.play.goal(beh, odom)
            return got if got is not None else (None, None)
        return super()._goal(beh, pct, odom, *args)

    def _consumed(self, *args: Any, **kwargs: Any) -> float:
        extra = super()._consumed(*args, **kwargs)
        if self.behaviour in PLAY_ACTS:
            extra += self.play.take_consumed()
        return extra

    def _start_bout(self, beh: str, *args: Any, **kwargs: Any) -> None:
        super()._start_bout(beh, *args, **kwargs)
        if beh in PLAY_ACTS:
            self.play.start(beh, int(self._t_prev or 0))

    def _look_before(self, t_ms: int, frame: C.SensorFrame, unseen: bool) -> bool:
        if self.behaviour in PLAY_ACTS and self.play.legs:
            return True  # the act has the legs (its own twist): the brain's rhythm rests
        return bool(super()._look_before(t_ms, frame, unseen))


def home_bearing(mind: PlayMind, odom: tuple[float, float, float]) -> tuple[float, float] | None:
    """(bearing, range) to the spot beside the dock (for tools)."""
    if mind.home_spot is None:
        return None
    return bearing_range(odom, *mind.home_spot)


INTEGRATION = """See the a:play report (2026-09-28): contract / select / brain / repertoire / genome /
drives / motor / bodylink / neuro_senses / neuro_world / neuro_live / neuro_api / page entries."""

__all__ = [
    "PLAY_ACTS",
    "PLAY_REFLEX_COEXIST",
    "PLAY_BEHAVIOURS",
    "PLAY_BUILDERS",
    "PLAY_FEATURES",
    "PLAY_VERSION",
    "PLAY_W",
    "PlayBrain",
    "PlayMind",
    "play_gain",
    "play_installed",
]
