"""a:social's three acts inside the neural brain — the seam until they are folded into brain.py.

The owner's A list (2026-09-28 「A B 全數」), behaviours (1) 跟著媽媽走, (8) 害怕時躲起來 and
(9) 睡前轉一圈. The skills themselves are pure modules (:mod:`duckmind.neuro.skill_imprint`,
:mod:`~duckmind.neuro.skill_shelter`, :mod:`~duckmind.neuro.skill_circle`); their motor programs
are :mod:`~duckmind.neuro.skill_social_programs`. This module puts them into the brain's own
machinery so that they EMERGE from it — the basal-ganglia competition (:mod:`duckmind.neuro.
select`) with the brain's drives, traces, temperament gains and gates, its locomotor rhythm
(:func:`duckmind.neuro.motor.loco_step`), its reflexes and its body path — and never replace it:

* :func:`registered` / :func:`register` add the acts to the shared tables (contract
  ``BEHAVIOURS`` / ``MOMENTS``, select ``FEATURES`` / ``W`` / ``SATIATE_S``, brain ``LOCO`` /
  ``TARGETED`` / ``REACH_M`` / act sets, repertoire ``BUILDERS`` / ``REACTIVE``) — exactly the
  additions the integration makes in those files (see each ``SOCIAL_*`` table); ``registered`` is
  reversible (tests), ``register`` is for a process of its own (a measurement script, a live
  launcher).
* :class:`SocialBrain` is :class:`~duckmind.neuro.brain.NeuroBrain` with the skills' state, their
  features filled into the competition (:class:`SocialSelector`), their goals, satiation and
  targets (the brain's own ``_goal`` / ``_consumed`` / ``_start_bout`` / ``_bout_target`` hooks),
  and — like the docking program — the loop's own twist, the arcs it walks looking for a mother it
  has lost, her look-stops and the hide run's peeks on the command.

Honest senses: everything comes from the :class:`~duckmind.neuro.contract.SensorFrame` plus three
readings the contract does not carry yet — a person sighting's *look*, a voice's *speaker
signature* and the camera's *roof* reading (all MODEL, :mod:`duckmind.world.neuro_social_senses`),
given through :meth:`SocialBrain.sense_social` (:class:`SocialInput`) until the contract has
``Sighting.look``, ``SoundSense.voice`` and ``CameraSense.roof`` (then :func:`social_of` reads
them from the frame and the side channel goes).
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any

from duckmind.i18n import msg
from duckmind.neuro import brain as B
from duckmind.neuro import contract as C
from duckmind.neuro import repertoire as P
from duckmind.neuro import select as S
from duckmind.neuro import skill_social_programs as SP
from duckmind.neuro.contract import (
    NORMAL_VX_MAX,
    BehaviourSpec,
    Gaze,
    MotorCommand,
    SensorFrame,
    Target,
    legal_walk,
)
from duckmind.neuro.skill_circle import ArcModel, CircleRun
from duckmind.neuro.skill_imprint import CareCues, Imprint, persons_from
from duckmind.neuro.skill_shelter import (
    SHELTER_MIN,
    HideRun,
    ShelterMap,
    hide_urge,
    shelter_words,
)

SOCIAL_VERSION = "neuro-social-2"

# ── the acts (contract.BEHAVIOURS) ─────────────────────────────────────────────


def _spec(
    zh: str,
    family: str,
    unlock_gd: int,
    posture: str,
    body_zh: str,
    sounds: tuple[str, ...] = ("",),
    *,
    locomotes: bool = False,
    stationary: bool = False,
) -> BehaviourSpec:
    return BehaviourSpec(
        zh, family, unlock_gd, posture, locomotes, sounds, body_zh, stationary=stationary
    )


SOCIAL_BEHAVIOURS: dict[str, BehaviourSpec] = {
    "follow_mom": _spec(
        msg("neuro.act.follow_mom.name"),
        "ethogram",
        0,
        "stand",
        msg("neuro.act.follow_mom.body"),
        ("", "chirp", "inquire"),
        locomotes=True,
    ),
    "settle_feet": _spec(
        msg("neuro.act.settle_feet.name"),
        "ethogram",
        0,
        "sit",
        msg("neuro.act.settle_feet.body"),
        ("", "coo"),
    ),
    "hide": _spec(
        msg("neuro.act.hide.name"),
        "ethogram",
        5,
        "stand",
        msg("neuro.act.hide.body"),
        ("", "alarm"),
        locomotes=True,
    ),
    "peek_out": _spec(
        msg("neuro.act.peek_out.name"), "moment", 0, "any", msg("neuro.act.peek_out.body")
    ),
    "check_mom": _spec(
        msg("neuro.act.check_mom.name"), "moment", 0, "any", msg("neuro.act.check_mom.body")
    ),
    "circle_settle": _spec(
        msg("neuro.act.circle_settle.name"),
        "locomotion",
        5,
        "stand",
        msg("neuro.act.circle_settle.body"),
        ("", "coo"),
        locomotes=True,
        stationary=True,  # (it turns round on its spot before it lies down)
    ),
}
# fired by the acts themselves (no inputs of their own in the moments' competition): the hide
# run's peeks, the imprinted duck's glances at where it believes its mother is
SOCIAL_MOMENTS = frozenset({"peek_out", "check_mom"})

# ── the competition's inputs (select.FEATURES / FEATURE_ZH) ─────────────────────

SOCIAL_FEATURES: dict[str, str] = {
    "mom_far": msg("neuro.why.mom_far"),
    "mom_moving": msg("neuro.why.mom_moving"),
    "mom_gone": msg("neuro.why.mom_gone"),
    "mom_near_still": msg("neuro.why.mom_near_still"),
    "mom_fear": msg("neuro.why.mom_fear"),
    "mom_bond": msg("neuro.why.mom_bond"),
    "mom_safe": msg("neuro.why.mom_safe"),
    "hide_urge": msg("neuro.why.hide_urge"),
    "shelter_known": msg("neuro.why.shelter_known"),
    "hiding": msg("neuro.why.hiding"),
    "circling": msg("neuro.why.circling"),
    "circled": msg("neuro.why.circled"),
}

# ── the wiring (select.W): the new channels, and what the old ones gain ─────────

SOCIAL_W: dict[str, dict[str, float]] = {
    # imprinting-follow: the further its mother, the more it walks after her; afraid, to her
    "follow_mom": {
        "mom_far": 1.3,
        "mom_moving": 0.75,
        "mom_gone": 0.85,
        "mom_fear": 1.1,
        "sleepy": -0.3,
        "mom_near_still": -0.8,
    },
    "settle_feet": {
        "mom_near_still": 1.35,
        "comfort": 0.15,
        "sleepy": 0.15,
        "calm": 0.1,
        "mom_far": -1.0,
        "mom_moving": -0.8,
        "mom_gone": -0.6,
    },
    # hiding: a fright past its own threshold with a shelter known; not with its mother right
    # there (then it runs to her); once in, it keeps at it (the loop's self-excitation).
    # (dock-home 2026-10-17) A hungry duck, and one whose pack runs low, hides less: the pack is
    # what ends the stay. Hunger is 0 until 58 % and battery_low 0 until 46 %, so a rested duck hides
    # as before. Headless selector, a duck already in its shelter: the pack level at which going home
    # wins over hiding was 43 % (mild leftover fear) / 37 % (strong fright) and is now 44.5 % / 40.5 %
    # when going home is fresh; with it half used up (an earlier try) 39 % / never became 43.5 % /
    # 37.5 %. Seed 1004 sat hiding at 39-41 % with nothing in these weights that counted the pack
    "hide": {
        "hide_urge": 1.7,
        "hiding": 0.9,
        "looming": 0.3,
        "startled": 0.2,
        "mom_safe": -1.2,
        "hunger": -0.9,
        "battery_low": -1.0,
    },
    "peek_out": {},
    "check_mom": {},
    # the pre-sleep circle: the same pull as falling asleep here (so where the brain would sleep
    # here, it circles first; where it would go home to its dock, it goes home) — gated to a
    # standing duck that has not just circled (SocialSelector._gate)
    "circle_settle": {
        "sleepy": 1.3,
        "comfort": 0.12,
        "night": 0.3,
        "fear": -0.6,
        "restless": -0.2,
        "home_pull": -0.5,
        "recovering": -0.5,
        "circling": 1.2,  # once begun, the loop is walked through (one motor pattern)
    },
}
# added to the existing channels' wiring (merged into select.W)
SOCIAL_W_EXTRA: dict[str, dict[str, float]] = {
    # the loop walked, it lies down here to sleep (not off somewhere else)
    "nap": {"circled": 0.9},
    "doze": {"circled": 0.8},
    "sit_down": {"circled": 0.7},
    "return_dock": {"circled": -0.5},
    "wander": {"circled": -0.4},
    "forage": {"circled": -0.3},
    "flee": {"hide_urge": -0.6},  # it would rather hide than just get away, when it can
    "follow": {"mom_bond": -0.6},  # imprinted, it follows its mother, not whoever walks by
    "call_out": {"mom_gone": 0.5},  # it peeps for her …
    "turn_around": {"mom_gone": 0.8},  # … turns round to look for her …
    "look_around": {"mom_gone": 0.4},  # … and looks about
}
SOCIAL_SATIATE_S = {
    "follow_mom": 60.0,
    "settle_feet": 60.0,
    "hide": 120.0,
    "peek_out": 2.0,
    "check_mom": 1.5,
    "circle_settle": 60.0,
}
# brain.LOCO: (top speed, go offset, meander) — after its mother keen and straight (a:social's
# physics runs at offset 0.35 / meander 0.35: the rhythm's pauses and the look-stops made it close
# on her at 0.06–0.1 m/s); the flight at its top
SOCIAL_LOCO = {"follow_mom": (NORMAL_VX_MAX, 0.6, 0.2), "hide": (NORMAL_VX_MAX, 0.6, 0.2)}
SOCIAL_TARGETED = {"follow_mom": "person", "settle_feet": "person", "hide": "place"}
REACH_FOLLOW_M = 0.62  # brain.REACH_M for follow_mom (the duck's own stop distance: goal shift)
REACH_HIDE_M = 0.18
SOCIAL_EXERTION = {"hide": 0.8, "follow_mom": 0.3, "circle_settle": 0.2}
SOCIAL_HEAD_ACTS = frozenset({"settle_feet", "hide", "circle_settle"})
SOCIAL_TURN_NEVER = frozenset({"settle_feet", "hide"})
SOCIAL_BEAK_BUSY = frozenset({"hide"})
SLEEP_ACTS = frozenset({"nap", "doze", "sit_down"})  # the acts the pre-sleep circle precedes
CIRCLED_TAU_S = 300.0  # "just circled" lasts about this (it does not circle again soon)
CIRCLE_COUNTS_RAD = 1.5  # a loop stopped after this much turning still counts as its circle
CIRCLE_RETRY_TRACE = 0.33  # … one stopped sooner closes the channel for ~30 s (0.33 → 0.3)
SLEEPY_MIN = 0.3  # the circle channel is open above this sleepiness
MOM_SAFE_S = 5.0
MOM_SAFE_M = 2.5
LOOK_STOP_AFTER_S = 3.5  # following, out of sight this long …
LOOK_STOP_STEPS_S = 2.0  # … after walking this long: it stops to look for her …
LOOK_STOP_S = 2.0  # … this long (the camera sees a still person only from a still head) …
LOOK_STOP_GAP_S = 4.0  # … and walks at least this long before the next such stop (it trusts
# where it believes she is for a few seconds: 2.0 / 3.0 stopped it every 3–5 s of its approach)
# She should be right beside it (believed within FACE_NEAR_M) but beyond what the head can turn
# to (FACE_RAD), unseen FACE_AFTER_S: it turns its body toward her in an arc (brain.TURN_TOWARD_*:
# 0.37 m/s, 0.9 rad/s; at most FACE_MAX_S, done when she is seen or within FACE_DONE_RAD)
FACE_AFTER_S = 2.0
FACE_NEAR_M = 2.0
FACE_RAD = 1.0
FACE_DONE_RAD = 0.5
FACE_MAX_S = 3.5
FACE_GAP_S = 3.0
FACE_VX = 0.37
FACE_VYAW = 0.9
# Lost: it has been to where it last saw her and she is not there — it goes round looking, an arc
# (this body cannot turn on the spot) with the look-stops above between (a still head sees a still
# person), toward where she was heading when it last saw her walk, else its own side; a bump turns
# the arc the other way. (a:social's physics runs: it walked straight on from the empty spot,
# away from her, for up to 40 s — she had left behind it, where its narrow camera never looked.)
SCAN_VX = 0.37
SCAN_VYAW = 0.9
SCAN_HEADINGS = (0.0, 0.5, 1.0)  # rad toward the arc's side: the way the arc sweeps next …
SCAN_CLEAR_M = 0.4  # … must be this clear of what it has seen (the arc's radius and a body)
# Imprinted, it keeps an eye on its mother whatever it is doing (not in an act that has the head:
# preening, pecking, a look around …): its eyes rest on her when she is in view, and when she has
# been out of view a while it glances where it believes she is — so it sees her walk off.
WATCH_BOND = 0.35
WATCH_RAD = 1.1  # as far as its head turns (±1.0 rad) and a little
CHECK_AFTER_S = 4.0  # out of view this long: a glance where it believes she is …
CHECK_EVERY_S = (4.0, 6.0)  # … every a + b × (1 − bond) s: the more attached, the more often


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


# ── the registry ────────────────────────────────────────────────────────────────

_ACTIVE: list[dict[str, Any]] = []
_LEGACY_CONTEXTS = 0


def legacy_context_active() -> bool:
    """Whether a reversible social fixture currently owns the shared tables."""
    return _LEGACY_CONTEXTS > 0


def _apply() -> dict[str, Any]:
    """Put the acts into the shared tables; returns what to restore."""
    saved: dict[str, Any] = {
        "moments": C.MOMENTS,
        "s_moments": S.MOMENTS,
        "features": S.FEATURES,
        "locomotes": S.LOCOMOTES,
        "head_acts": B.HEAD_ACTS,
        "turn_never": B.TURN_TOWARD_NEVER,
        "beak_busy": B.BEAK_BUSY,
        "reactive": P.REACTIVE,
        "moment_programs": P.MOMENT_PROGRAMS,
        "w_old": {k: dict(S.W[k]) for k in SOCIAL_W_EXTRA if k in S.W},
    }
    for key, spec in SOCIAL_BEHAVIOURS.items():
        C.BEHAVIOURS[key] = spec
    C.MOMENTS = frozenset(C.MOMENTS | SOCIAL_MOMENTS)
    S.MOMENTS = frozenset(S.MOMENTS | SOCIAL_MOMENTS)
    new = tuple(f for f in SOCIAL_FEATURES if f not in S.F)
    S.FEATURES = tuple(S.FEATURES) + new
    for name in new:
        S.F[name] = len(S.F)
    S.FEATURE_ZH.update(SOCIAL_FEATURES)
    for key, weights in SOCIAL_W.items():
        S.W[key] = dict(weights)
    for key, extra in SOCIAL_W_EXTRA.items():
        if key in S.W:
            S.W[key] = {**S.W[key], **extra}
    S.SATIATE_S.update(SOCIAL_SATIATE_S)
    S.LOCOMOTES = frozenset(S.LOCOMOTES | {k for k, v in SOCIAL_BEHAVIOURS.items() if v.locomotes})
    B.LOCO.update(SOCIAL_LOCO)
    B.TARGETED.update(SOCIAL_TARGETED)
    B.REACH_M.update({"follow_mom": REACH_FOLLOW_M, "hide": REACH_HIDE_M})
    B.EXERTION.update(SOCIAL_EXERTION)
    B.HEAD_ACTS = frozenset(B.HEAD_ACTS | SOCIAL_HEAD_ACTS)
    B.TURN_TOWARD_NEVER = frozenset(B.TURN_TOWARD_NEVER | SOCIAL_TURN_NEVER)
    B.BEAK_BUSY = frozenset(B.BEAK_BUSY | SOCIAL_BEAK_BUSY)
    P.BUILDERS.update(SP.BUILDERS)
    P.REACTIVE = frozenset(P.REACTIVE | SP.REACTIVE)
    P.MOMENT_PROGRAMS = frozenset(P.MOMENT_PROGRAMS | SP.MOMENT_PROGRAMS)
    return saved


def _restore(saved: dict[str, Any]) -> None:
    for key in SOCIAL_BEHAVIOURS:
        C.BEHAVIOURS.pop(key, None)
    C.MOMENTS = saved["moments"]
    S.MOMENTS = saved["s_moments"]
    for name in SOCIAL_FEATURES:
        if name not in saved["features"]:
            S.F.pop(name, None)
            S.FEATURE_ZH.pop(name, None)
    S.FEATURES = saved["features"]
    for key in SOCIAL_W:
        S.W.pop(key, None)
    for key, weights in saved["w_old"].items():
        S.W[key] = weights
    for key in SOCIAL_SATIATE_S:
        S.SATIATE_S.pop(key, None)
    S.LOCOMOTES = saved["locomotes"]
    for key in SOCIAL_LOCO:
        B.LOCO.pop(key, None)
    for key in SOCIAL_TARGETED:
        B.TARGETED.pop(key, None)
    B.REACH_M.pop("follow_mom", None)
    B.REACH_M.pop("hide", None)
    for key in SOCIAL_EXERTION:
        B.EXERTION.pop(key, None)
    B.HEAD_ACTS = saved["head_acts"]
    B.TURN_TOWARD_NEVER = saved["turn_never"]
    B.BEAK_BUSY = saved["beak_busy"]
    for key in SP.BUILDERS:
        P.BUILDERS.pop(key, None)
    P.REACTIVE = saved["reactive"]
    P.MOMENT_PROGRAMS = saved["moment_programs"]


def is_registered() -> bool:
    return bool(_ACTIVE) or all(k in C.BEHAVIOURS for k in SOCIAL_BEHAVIOURS)


def register() -> None:
    """Add the acts to the shared tables for the rest of this process (idempotent)."""
    if not is_registered():
        _ACTIVE.append(_apply())


@contextmanager
def registered() -> Iterator[None]:
    """The acts in the shared tables for the duration (restored exactly afterwards)."""
    global _LEGACY_CONTEXTS
    if is_registered():
        yield
        return
    saved = _apply()
    _ACTIVE.append(saved)
    _LEGACY_CONTEXTS += 1
    try:
        yield
    finally:
        _LEGACY_CONTEXTS -= 1
        _ACTIVE.remove(saved)
        _restore(saved)


# ── the genome's share ──────────────────────────────────────────────────────────


def social_gains(genome: Any) -> dict[str, float]:
    """Salience multipliers for the acts (temperament, stage): sociable ducks follow and settle
    most, the young follow their mother everywhere; the timid hide (like flee/freeze in
    genome._gains); the circle is everyone's."""
    a = genome.axes
    young = {"hatchling": 1.0, "duckling": 0.6, "juvenile": 0.25, "adult": 0.0}.get(
        str(genome.stage), 0.25
    )
    soc, bold = float(a["sociability"]), float(a["boldness"])
    return {
        "follow_mom": (0.6 + 0.8 * soc) * (1.0 + 0.6 * young),
        "settle_feet": 0.6 + 0.8 * soc,
        "hide": 1.45 - 0.9 * bold,
        "peek_out": 1.0,
        "check_mom": 1.0,
        "circle_settle": 1.0,
    }


def social_unlocked(genome: Any) -> frozenset[str]:
    """The acts this duck can do at its growth day (genome._unlocked's rule: walking acts from
    its first steps at its own pace)."""
    from duckmind.neuro.genome import FIRST_STEPS_GD

    out: set[str] = set()
    pace = float(genome.pace)
    for key, spec in SOCIAL_BEHAVIOURS.items():
        day = spec.unlock_gd
        if spec.locomotes:
            first = int(round(FIRST_STEPS_GD * pace))
            day = first if day <= FIRST_STEPS_GD else int(round(day * pace))
        elif day > 0:
            day = int(round(day * pace))
        if int(genome.gd) >= day:
            out.add(key)
    return frozenset(out)


def with_social(genome: Any) -> Any:
    """The genome with the acts unlocked (by its growth day) and their gains."""
    added = social_gains(genome)
    gains = {**genome.gains, **added}
    if getattr(genome, "care_applied", False):
        # (growth G3: the care is already in the genome's gains; the acts rewritten here get it again)
        from duckmind.neuro.genome import care_gain_factor

        for key in added:
            gains[key] *= care_gain_factor(genome.care, key)
    return replace(
        genome, unlocked=frozenset(genome.unlocked | social_unlocked(genome)), gains=gains
    )


# ── the three MODEL readings until the contract carries them ───────────────────


@dataclass(frozen=True, slots=True)
class SocialInput:
    """``looks``: the eye track id → a person's look (skill_imprint.LOOK_DIMS; MODEL);
    ``voice``: the speaker signature of the speech heard now (MODEL); ``roof``: m above the lens
    of a surface covering the top of the picture (skill_shelter.ROOF_OPEN = seen open above,
    None = cannot tell; MODEL)."""

    looks: Mapping[int, tuple[float, ...]] = field(default_factory=dict)
    voice: tuple[float, float] | None = None
    roof: float | None = None


def social_of(frame: SensorFrame, side: SocialInput | None) -> SocialInput:
    """The readings from the frame once the contract carries them (``Sighting.look``,
    ``SoundSense.voice``, ``CameraSense.roof``), else from the side channel."""
    looks: dict[int, tuple[float, ...]] = dict(side.looks) if side is not None else {}
    for sight in frame.camera.sightings:
        look = getattr(sight, "look", None)
        if look is not None:
            looks[int(sight.track)] = tuple(float(v) for v in look)
    voice = getattr(frame.sound, "voice", None)
    if voice is None and side is not None:
        voice = side.voice
    roof = getattr(frame.camera, "roof", None)
    if roof is None and side is not None:
        roof = side.roof
    return SocialInput(looks, voice, roof)


# ── the selector: the acts' features and gates in the same competition ─────────


class SocialSelector(S.Selector):
    """:class:`~duckmind.neuro.select.Selector` that fills the acts' features into the brain's
    feature vector right before the competition (after the drives have been updated this tick)
    and adds their gates (the pre-sleep circle: a standing duck that has not just circled; the
    hide run keeps its channel while it sits in the shelter)."""

    def __init__(self, genome: Any, seed: int, owner: SocialBrain) -> None:
        super().__init__(genome, seed)
        self.owner = owner

    def step(
        self,
        dt_s: float,
        feat: list[float],
        ctx: S.Context,
        interrupted: bool,
        consumed: float,
        *,
        release_commitment: bool = False,
        walking_to: bool = False,
    ) -> tuple[str, bool]:
        self.owner._fill_social(feat, ctx)
        return super().step(
            dt_s,
            feat,
            ctx,
            interrupted,
            consumed,
            release_commitment=release_commitment,
            walking_to=walking_to,
        )

    def _gate(self, behaviour: str, ctx: S.Context) -> float:
        owner = self.owner
        if behaviour == "circle_settle":
            if not ctx.standing or ctx.asleep or not owner._circle_open:
                return 0.0
            return super()._gate(behaviour, ctx)
        if behaviour == "hide" and owner.hide_run.phase == "in":
            return 1.0  # sitting low in its shelter is the act itself (no posture cost)
        g = super()._gate(behaviour, ctx)
        if behaviour in SLEEP_ACTS and ctx.standing and owner._circle_open:
            g *= 0.3  # standing and about to sleep: the circle comes first
        return g


# ── the brain ───────────────────────────────────────────────────────────────────


class SocialBrain(B.NeuroBrain):
    """The neural brain with a:social's acts (see the module doc). ``social``: the side channel's
    first readings (usually given every tick through :meth:`sense_social`)."""

    def __init__(
        self,
        genome: Any,
        seed: int = 0,
        *,
        dock: Any = None,
        social: SocialInput | None = None,
    ) -> None:
        if not is_registered():
            raise RuntimeError(
                "a:social's acts are not in the shared tables: use skill_social_brain.register() "
                "or registered() first"
            )
        genome = with_social(genome)
        super().__init__(genome, seed, dock=dock)
        self.sel = SocialSelector(genome, self.seed, self)
        self.behaviour = self.sel.names[self.sel.current]
        self.imprint = Imprint.from_genome(genome)
        self.shelters = ShelterMap()
        self.hide_run = HideRun(boldness=float(genome.axes["boldness"]), seed=self.seed)
        self.arcs = ArcModel()
        self.circle = CircleRun(
            innate_side=int(genome.side),
            tempo=_clip(0.3 * float(genome.axes["activity"]) + 0.7 * float(genome.axes["tempo"])),
        )
        self._social_in = social or SocialInput()
        self._sfeat: dict[str, float] = {}
        self._circled = 0.0
        self._circle_open = False
        self._s_t: float | None = None
        self._s_odom: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._s_frame: SensorFrame | None = None
        self._pass_s_ms: int | None = None
        self._speech_prev: str | None = None
        self._last_twist = (0.0, 0.0)
        self._roof: float | None = None
        self._reflex_seen = 0
        self._turned_prev = 0.0
        self._step_s = 0.0  # following: how long it has walked since it last stopped
        self._post_t = 0.0
        self._look_until = -1e9  # a stop to look for its mother lasts until then
        self._check_next = 0.0  # the next glance to check on its mother (not before)
        self._face_until = -1e9  # a turn to face its mother (she is beside it, unseen) until then
        self._face_sign = 1.0
        self._scan_sign = 0.0  # lost: the side it goes round to look for her (0 = not scanning)
        self._scan_flip_t = -1e9
        self.social_log: list[tuple[float, str]] = []  # (t_s, what) for evaluations and the page

    # ── the side channel ─────────────────────────────────────────────────────

    def _adopt_params(self, fresh: B.NeuroBrain) -> None:
        """(Growth G3, a regrow; see ``NeuroBrain._adopt_params``.) Its age shapes how it holds to
        its family: the imprint's stage and plasticity follow the genome; the bonds it has formed
        stay."""
        super()._adopt_params(fresh)
        assert isinstance(fresh, SocialBrain)
        self.sel.owner = self  # type: ignore[attr-defined]
        self.imprint.stage = fresh.imprint.stage
        self.imprint.young = fresh.imprint.young
        self.imprint.plast = fresh.imprint.plast

    def sense_social(self, social: SocialInput) -> None:
        """This tick's MODEL readings (looks, voice, roof) until the contract carries them."""
        self._social_in = social

    # ── the step ─────────────────────────────────────────────────────────────

    def step(self, t_ms: int, frame: SensorFrame) -> tuple[MotorCommand, C.BrainTelemetry]:
        social = social_of(frame, self._social_in)
        self._pre(t_ms, frame, social)
        cmd, tel = super().step(t_ms, frame)
        cmd, reason = self._post(t_ms, frame, cmd)
        self._last_twist = (cmd.vx, cmd.vyaw)
        learned = dict(tel.learned)
        learned["social"] = self.social_view()
        tel = replace(tel, learned=learned, reason_zh=reason or tel.reason_zh)
        return cmd, tel

    def _log(self, t_s: float, what: str) -> None:
        self.social_log.append((round(t_s, 2), what))
        del self.social_log[:-200]

    def _pre(self, t_ms: int, frame: SensorFrame, social: SocialInput) -> None:
        """The skills' perception from this tick's frame (before the brain decides)."""
        t_s = t_ms / 1000.0
        dt = 0.05 if self._s_t is None else _clip(t_s - self._s_t, 0.0, 2.0)
        self._s_t = t_s
        odom = frame.body.odom
        self._s_odom = odom
        self._s_frame = frame
        clock_offset = t_ms - frame.t_ms
        # care it can feel (perceive's gentle/sharp rule on the frame's touch)
        touch = frame.touch
        gentle = sharp = 0.0
        if touch.site is not None:
            if touch.stroke > 0.2 or touch.pressure < 0.45:
                gentle = _clip(0.4 + 0.4 * touch.pressure + 0.4 * touch.stroke)
            else:
                sharp = _clip(touch.pressure)
        kind = frame.sound.kind if frame.sound.kind in ("voice", "name", "hello") else None
        speech = kind if kind is not None and kind != self._speech_prev else None
        self._speech_prev = kind
        care = CareCues(
            touch_gentle=gentle,
            touch_sharp=sharp,
            set_down="set_down" in frame.events,
            speech=speech,
            voice=social.voice if speech is not None else None,
        )
        # a new camera pass: its person sightings (with their looks); between passes the last
        cam = frame.camera
        persons = None
        if cam.usable:
            pass_ms = t_ms - int(cam.age_ms)
            if self._pass_s_ms is None or abs(pass_ms - self._pass_s_ms) > 5:
                self._pass_s_ms = pass_ms
                persons = persons_from(cam.sightings, social.looks)
        self.imprint.observe(
            t_s,
            dt,
            odom,
            persons if persons is not None else (),
            care,
            battery=frame.power.battery,
            fresh=persons is not None,
        )
        head_yaw = float(frame.body.head[2]) if len(frame.body.head) > 2 else 0.0
        stepping = frame.body.posture == "walk" or abs(frame.body.vx) > 0.2
        # (a stepping duck's camera smears: it cannot tell a still person is not there)
        can_see = cam.usable and cam.light in ("bright", "dim") and not stepping
        self.imprint.check_absent(dt, odom, head_yaw, can_see)
        # shelters: the floor line and the roof reading into the map
        self._roof = social.roof
        self.shelters.observe(
            t_ms,
            odom,
            cam.floor,
            cam.floor_t_ms,
            social.roof,
            clock_offset=clock_offset,
        )
        # the arcs this body really makes (the loop's size)
        walking = frame.body.posture == "walk" or abs(frame.body.vx) > 0.2
        self.arcs.observe(
            t_s, odom, self._last_twist[0], self._last_twist[1], frame.body.gyro[2], walking
        )
        self._circled *= math.exp(-dt / CIRCLED_TAU_S)

    def _fill_social(self, feat: list[float], ctx: S.Context) -> None:
        """The acts' inputs into the competition (called by the selector, drives fresh)."""
        F = S.F
        d = self.drives
        odom = self._s_odom
        t_s = self._s_t or 0.0
        out = self.imprint.features(odom, d.fear)
        view = self.imprint.mom_view(odom)
        safe = 0.0
        if view is not None and view.since_s <= MOM_SAFE_S and view.range_m <= MOM_SAFE_M:
            safe = view.bond
        out["mom_safe"] = safe
        known = self.shelters.known(odom, t_s)
        startled = feat[F["startled"]] if "startled" in F else 0.0
        urge = hide_urge(d.fear, float(self.genome.axes["boldness"]), startled)
        out["hide_urge"] = urge if known >= SHELTER_MIN else 0.0
        out["shelter_known"] = known
        out["hiding"] = (
            1.0 if self.hide_run.phase in ("go", "in") and self.behaviour == "hide" else 0.0
        )
        out["circled"] = self._circled
        out["circling"] = (
            1.0
            if self.circle.phase in ("turning", "seeking") and self.behaviour == "circle_settle"
            else 0.0
        )
        docking = self.dock.mode in ("docking", "home")
        on_dock = bool(self._s_frame is not None and self._s_frame.power.on_dock)
        self._circle_open = (
            d.sleepiness >= SLEEPY_MIN
            and self._circled < 0.3
            and not docking
            and not on_dock
            and "circle_settle" in self.sel.names
        )
        for name, value in out.items():
            index = F.get(name)
            if index is not None and index < len(feat):
                feat[index] = value
        self._sfeat = out

    # ── the brain's hooks ────────────────────────────────────────────────────

    def _goal(self, beh: str, pct: Any, odom: Any, t_s: float, *args: Any) -> Any:
        if beh == "follow_mom":
            goal = self.imprint.goal(odom)
            if goal is None:
                return None, None
            bearing, rng = goal
            if self.imprint.searching(odom):
                return bearing, rng  # to where it last saw her (it stops within REACH_M)
            # the brain stops at REACH_FOLLOW_M; this duck stops at its own distance
            return bearing, max(0.0, rng - (self.imprint.stop_m() - REACH_FOLLOW_M))
        if beh == "settle_feet":
            goal = self.imprint.goal(odom)
            return (None, None) if goal is None else goal
        if beh == "hide":
            if self.hide_run.phase == "go":
                goal = self.hide_run.goal(odom)
                return (None, None) if goal is None else goal
            return 0.0, 0.0  # in its shelter (or done): arrived, no walking
        if beh == "circle_settle":
            return None, None
        return super()._goal(beh, pct, odom, t_s, *args)

    def _consumed(self, dt: float, pct: Any, *args: Any, **kwargs: Any) -> float:
        extra = super()._consumed(dt, pct, *args, **kwargs)
        beh = self.behaviour
        f = self._sfeat
        if beh == "follow_mom":
            if f.get("mom_near_still", 0.0) > 0.0:
                extra += dt / 2.0  # caught up and she has stopped: settle instead
            elif f.get("mom_bond", 0.0) <= 0.0 or (
                f.get("mom_far", 0.0) <= 0.0
                and f.get("mom_gone", 0.0) <= 0.0
                and f.get("mom_moving", 0.0) <= 0.0
            ):
                extra += dt / 3.0  # nothing pulls it after her now
        elif beh == "settle_feet":
            if f.get("mom_moving", 0.0) > 0.0 or f.get("mom_far", 0.0) > 0.2:
                extra += dt / 1.5
            elif f.get("mom_gone", 0.0) > 0.0:
                extra += dt / 4.0
        elif beh == "hide":
            if self.hide_run.phase in ("out", "failed", "idle"):
                extra += 1.0 if self.hide_run.phase == "out" else 0.6
        elif beh == "circle_settle":
            # one motor pattern: used up when the loop is walked (or cannot be), not by degrees
            if self.circle.phase in ("done", "stopped", "idle"):
                extra += 1.0
        return extra

    def _start_bout(self, beh: str, *args: Any, **kwargs: Any) -> None:
        super()._start_bout(beh, *args, **kwargs)
        t_s = self._s_t or 0.0
        odom = self._s_odom
        if beh != "circle_settle" and self.circle.phase in ("turning", "seeking"):
            self.circle.phase = "stopped"
        if beh == "hide":
            ok = self.hide_run.start(
                t_s, odom, self.shelters, self.appraiser.threat_heading, self.drives.fear
            )
            self._log(t_s, f"hide {'go' if ok else 'none'}")
            if ok and self.hide_run.refuge is not None:
                bearing, _ = self.hide_run.refuge.bearing_range(odom)
                self._bout["side"] = 1 if bearing >= 0.0 else -1
        elif beh == "circle_settle":
            clearance, known = self._room_views(odom)
            ok = self.circle.start(t_s, odom, self.arcs, clearance=clearance, known=known)
            self._turned_prev = 0.0
            self._bout["side"] = self.circle.side
            self._log(t_s, f"circle {'start' if ok else 'no room'} {self.circle.side:+d}")
        elif beh in ("follow_mom", "settle_feet"):
            view = self.imprint.mom_view(odom)
            if view is not None and abs(view.bearing) > 0.15:
                self._bout["side"] = 1 if view.bearing > 0 else -1
            self._log(t_s, beh)

    def _room_views(self, odom: tuple[float, float, float]) -> tuple[Any, Any]:
        """How far the way is clear along a heading (the walls and furniture it has seen — a loose
        toy in the way is only nudged) and how far it has SEEN the floor clear, from here now."""
        try:
            rel = self.bound._relative(odom, None, walls_only=True)
        except TypeError:  # (a boundary without the walls-only view)
            rel = None

        def clearance(h: float) -> float:
            return float(self.bound.clearance(odom, h, rel=rel))

        def known(h: float) -> float:
            return float(self.bound.known_free(odom, h))

        return clearance, known

    def _bout_target(
        self, beh: str, goal: Any, goal_range: Any, pct: Any, remembered: Any = None
    ) -> Any:
        if beh in ("follow_mom", "settle_feet"):
            view = self.imprint.mom_view(self._s_odom)
            if view is None or view.since_s > 3.0:
                return None  # she is not in sight: the program looks for her
            return Target("person", view.bearing, view.range_m, view.track)
        if beh == "hide":
            goal_now = self.hide_run.goal(self._s_odom)
            if goal_now is None:
                return None
            return Target("place", goal_now[0], goal_now[1])
        return super()._bout_target(beh, goal, goal_range, pct, remembered)

    def _no_room(self, frame: SensorFrame, beh: str) -> frozenset[str]:
        base = super()._no_room(frame, beh)
        if beh == "hide" and self.hide_run.phase == "in":
            return frozenset(C.MOMENTS)  # hidden and still: no glances of its own
        return base

    # ── after the brain: the loop's twist, the flight's end, the peeks ───────

    def _post(self, t_ms: int, frame: SensorFrame, cmd: MotorCommand) -> tuple[MotorCommand, str]:
        t_s = t_ms / 1000.0
        odom = frame.body.odom
        beh = self.behaviour
        held = cmd.behaviour != beh  # a reflex holds the body (its own program)
        standing = frame.body.posture in ("stand", "walk")
        reason = ""
        # the reflexes' frights this tick (a new startle keeps it hidden longer)
        recent = [e for e in self._reflex_events() if e.reflex in ("startle", "withdraw")]
        if recent:
            self.hide_run.scared(t_s)
        # ── the pre-sleep circle ──
        if beh != "circle_settle" and self.circle.phase in ("turning", "seeking"):
            self.circle.phase = "stopped"
        if beh == "circle_settle":
            clearance, known = self._room_views(odom)
            fright = cmd.reflex in ("startle", "freeze")
            touched = cmd.reflex in ("withdraw", "bump_stop") or "blocked" in frame.events
            if fright or (not standing and self.circle.turned > 0.3):
                # a fright (or it sat down): the loop ends where it is
                phase = self.circle.step(
                    t_s, odom, stopped=True, arcs=self.arcs, clearance=clearance, known=known
                )
            elif touched and self.circle.phase == "turning":
                # its body touched something as it looped: early on, it finds room and tries again
                phase = self.circle.bumped(t_s, odom, clearance)
                self._log(t_s, f"circle bumped: {phase}")
            elif held:
                # a reflex's moment has the legs: the loop waits with it
                self.circle.pause(_clip(t_s - self._post_t, 0.0, 0.5))
                phase = self.circle.phase
            else:
                # (a seek pressed against something looks for room again from there)
                stopped = self.circle.phase == "seeking" and "blocked" in frame.events
                phase = self.circle.step(
                    t_s, odom, stopped=stopped, arcs=self.arcs, clearance=clearance, known=known
                )
            if phase in ("turning", "seeking") and not held and standing and not cmd.back_arc:
                vx, vyaw = self.circle.command(odom)
                vx, vyaw = legal_walk(vx, vyaw, approach=True)
                cmd = replace(cmd, vx=vx, vyaw=vyaw, side=self.circle.side, approach=False)
                self._stepping = vx != 0.0
            if phase in ("done", "stopped") and self._circled < 0.9:
                # a loop walked (or most of one) counts; one stopped at the start only rests the
                # channel a moment (CIRCLE_RETRY_S) and it may try again
                walked = phase == "done" or self.circle.turned >= CIRCLE_COUNTS_RAD
                self._circled = 1.0 if walked else max(self._circled, CIRCLE_RETRY_TRACE)
                self._log(t_s, f"circle {phase} {math.degrees(self.circle.turned):.0f}")
            turned = math.degrees(self.circle.turned)
            if phase == "seeking":
                reason = msg("neuro.incoming.incoming_skill_social_brain.SocialBrain._post.cramped")
            elif phase == "turning":
                reason = msg(
                    "neuro.incoming.incoming_skill_social_brain.SocialBrain._post.805ddd",
                    turned=turned,
                )
            elif phase == "done":
                reason = msg(
                    "neuro.incoming.incoming_skill_social_brain.SocialBrain._post.6f53d5",
                    turned=turned,
                )
        # ── the flight and the hiding ──
        if beh == "hide" or self.hide_run.phase in ("go", "in"):
            encl, _out = self.shelters.enclosure(odom[0], odom[1])
            before = self.hide_run.phase
            phase = self.hide_run.step(
                t_s,
                odom,
                fear=self.drives.fear,
                blocked="blocked" in frame.events,
                roof=self._roof,
                enclosure_here=encl,
                shelters=self.shelters,
                free=not held and standing and cmd.reflex is None,
            )
            if before == "go" and phase == "in":
                self._log(t_s, "hide in")
            if before == "in" and phase == "out":
                self._log(t_s, "hide out calm")  # calm again: it comes out
            if beh != "hide" and phase in ("go", "in"):
                # something else took over: out of hiding (the fear has ebbed below what else
                # it wants) or the flight given up
                self._log(t_s, f"hide {'out' if phase == 'in' else 'stop'} {beh}")
                self.hide_run.phase = "idle"
            elif phase == "in":
                if cmd.vx != 0.0 and not held:
                    cmd = replace(cmd, vx=0.0, vyaw=0.0)
                    self._stepping = False
                if not held and self.hide_run.peek_due(t_s, self.drives.fear):
                    bearing = self.hide_run.peek_bearing(odom)
                    self.moment = "peek_out"
                    self._moment_seq += 1
                    self._moment_busy = 1.0
                    self.counts["peek_out"] = self.counts.get("peek_out", 0) + 1
                    cmd = replace(
                        cmd,
                        moment="peek_out",
                        moment_seq=self._moment_seq,
                        side=1 if bearing >= 0.0 else -1,
                        intensity=_clip(abs(bearing)),
                    )
                    self._log(t_s, f"peek {self.hide_run.peeks}")
                where = shelter_words(self.hide_run.refuge)
                reason = msg(
                    "neuro.incoming.incoming_skill_social_brain.SocialBrain._post.060332",
                    shelter=where,
                    peeks=self.hide_run.peeks,
                )
            elif phase == "go" and beh == "hide":
                where = shelter_words(self.hide_run.refuge)
                reason = msg(
                    "neuro.incoming.incoming_skill_social_brain.SocialBrain._post.c343c1",
                    shelter=where,
                )
        # ── keeping an eye on its mother (a duckling watches the hen: it sees her go) ──
        view_m = self.imprint.mom_view(odom)
        watching = (
            view_m is not None
            and view_m.bond >= WATCH_BOND
            and not held
            and cmd.reflex is None
            and not cmd.asleep
            and beh not in B.HEAD_ACTS
            and beh not in ("follow_mom", "settle_feet")
            and self._moment_busy <= 0.3  # (not while a moment plays: cmd.moment is the last one)
        )
        if watching and view_m is not None and view_m.seen and abs(view_m.bearing) <= WATCH_RAD:
            # she is in view: its eyes stay on her between its own looks
            gain = 0.5 + 0.4 * view_m.bond
            cmd = replace(cmd, gaze=Gaze(view_m.bearing, 0.1, gain, 0.5))
        checking = (
            view_m is not None
            and view_m.bond >= WATCH_BOND
            and view_m.believed
            and not view_m.seen
            and view_m.since_s >= CHECK_AFTER_S
            and abs(view_m.bearing) <= WATCH_RAD
            and not held
            and cmd.reflex is None
            and not cmd.asleep
            and self._moment_busy <= 0.3
            and beh != "hide"
            and t_s >= self._check_next
        )
        if checking and view_m is not None:
            # a glance where it believes she is (a moment inside whatever it is doing: a
            # duckling keeps checking on the hen) — so it sees her, or finds she has gone
            self._check_next = t_s + CHECK_EVERY_S[0] + CHECK_EVERY_S[1] * (1.0 - view_m.bond)
            self.moment = "check_mom"
            self._moment_seq += 1
            self._moment_busy = 1.0
            self.counts["check_mom"] = self.counts.get("check_mom", 0) + 1
            cmd = replace(
                cmd,
                moment="check_mom",
                moment_seq=self._moment_seq,
                side=1 if view_m.bearing >= 0.0 else -1,
                intensity=_clip(abs(view_m.bearing)),
            )
        # ── following its mother ──
        if beh in ("follow_mom", "settle_feet") and not held:
            view = self.imprint.mom_view(odom)
            if view is not None and view.since_s <= 3.0 and cmd.reflex is None:
                cmd = replace(
                    cmd, gaze=Gaze(view.bearing, 0.2 if beh == "settle_feet" else 0.1, 0.9, 0.5)
                )
        # following and it has not seen her for a moment: it stops and looks where it thinks she
        # is (a stepping duck's camera smears: a still person is seen only from a still head)
        if beh == "follow_mom" and not held and cmd.reflex is None:
            view = self.imprint.mom_view(odom)
            # lost right where it last saw her: it goes round looking (between the look-stops)
            lost_here = view is not None and not view.believed and self.imprint.goal(odom) is None
            if not lost_here:
                self._scan_sign = 0.0
            elif standing and t_s >= self._look_until and not cmd.back_arc:
                if self._scan_sign == 0.0:
                    self._scan_sign = self._scan_side(odom)
                    self._scan_flip_t = t_s
                    self._log(t_s, f"look round for mom {self._scan_sign:+.0f}")
                elif "blocked" in frame.events and t_s >= self._scan_flip_t + 3.0:
                    self._scan_sign = -self._scan_sign  # something in the way: round the other way
                    self._scan_flip_t = t_s
                # the arc goes only where it has seen the floor clear (its boundary memory: walls,
                # furniture, things); something close that way: round the other way, or it stays
                # and looks (the look-stops) — it does not walk into what it has seen
                rel = self.bound._relative(odom, None)
                side = self._scan_sign
                way = min(self.bound.clearance(odom, h * side, rel=rel) for h in SCAN_HEADINGS)
                if way < SCAN_CLEAR_M and t_s >= self._scan_flip_t + 1.0:
                    other = min(
                        self.bound.clearance(odom, -h * side, rel=rel) for h in SCAN_HEADINGS
                    )
                    if other >= SCAN_CLEAR_M:
                        self._scan_sign = -side
                        self._scan_flip_t = t_s
                        way = other
                if way >= SCAN_CLEAR_M:
                    vx, vyaw = legal_walk(SCAN_VX, SCAN_VYAW * self._scan_sign, approach=True)
                    cmd = replace(cmd, vx=vx, vyaw=vyaw)
                else:
                    cmd = replace(cmd, vx=0.0, vyaw=0.0)
                self._stepping = cmd.vx != 0.0
                reason = msg(
                    "neuro.incoming.incoming_skill_social_brain.SocialBrain._post.mom_lost"
                )
            stepping = cmd.vx != 0.0
            self._step_s = self._step_s + (t_s - self._post_t) if stepping else 0.0
            if view is not None and view.seen:
                self._look_until = min(self._look_until, t_s)
            elif (
                view is not None
                and view.since_s >= LOOK_STOP_AFTER_S
                and self._step_s >= LOOK_STOP_STEPS_S
                and t_s >= self._look_until + LOOK_STOP_GAP_S
            ):
                self._look_until = t_s + LOOK_STOP_S
                self._log(t_s, "look for mom")
            if t_s < self._look_until and view is not None:
                if lost_here:  # (its head sweeps round on its own: the program's looking about)
                    cmd = replace(cmd, vx=0.0, vyaw=0.0)
                    reason = msg(
                        "neuro.incoming.incoming_skill_social_brain.SocialBrain._post.mom_look_around"
                    )
                else:
                    bearing = max(-1.0, min(1.0, view.bearing))
                    cmd = replace(cmd, vx=0.0, vyaw=0.0, gaze=Gaze(bearing, 0.15, 1.0, 0.6))
                    reason = msg(
                        "neuro.incoming.incoming_skill_social_brain.SocialBrain._post.mom_look"
                    )
                self._stepping = False
                self._step_s = 0.0
            # she should be right beside it but out of the head's reach (the head turns ±1.0
            # rad): its body turns to face her — an arc (this body cannot turn on the spot)
            if (
                view is not None
                and view.believed
                and not view.seen
                and view.since_s >= FACE_AFTER_S
                and view.range_m <= FACE_NEAR_M
                and abs(view.bearing) > FACE_RAD
                and standing
                and t_s >= self._face_until + FACE_GAP_S
            ):
                self._face_until = t_s + FACE_MAX_S
                self._face_sign = 1.0 if view.bearing > 0.0 else -1.0
                self._log(t_s, "turn to face mom")
            if t_s < self._face_until and view is not None:
                if view.seen or abs(view.bearing) < FACE_DONE_RAD or not standing:
                    self._face_until = t_s
                else:
                    vx, vyaw = legal_walk(FACE_VX, FACE_VYAW * self._face_sign, approach=True)
                    gaze = Gaze(max(-1.0, min(1.0, view.bearing)), 0.15, 1.0, 0.6)
                    cmd = replace(cmd, vx=vx, vyaw=vyaw, gaze=gaze)
                    self._stepping = vx != 0.0
                    reason = msg(
                        "neuro.incoming.incoming_skill_social_brain.SocialBrain._post.mom_turn"
                    )
        self._post_t = t_s
        return cmd, reason

    def _scan_side(self, odom: tuple[float, float, float]) -> float:
        """Which way it goes round looking for its mother: toward where she would be now had she
        walked on as it last saw her walk (its own sighting of her, carried by its odometry), else
        its own side (drifting, never a coin)."""
        mom = self.imprint.mom
        person = self.imprint.people.people.get(mom) if mom is not None else None
        if person is not None and math.hypot(person.vx, person.vy) > 0.15:
            px = person.x + person.vx * 3.0 - odom[0]
            py = person.y + person.vy * 3.0 - odom[1]
            rel = math.atan2(py, px) - odom[2]
            rel = math.atan2(math.sin(rel), math.cos(rel))
            if abs(rel) > 0.2:
                return 1.0 if rel > 0.0 else -1.0
        return 1.0 if float(self.genome.side) >= 0.0 else -1.0

    def _reflex_events(self) -> list[Any]:
        """The reflex events that came since the last tick (the brain's own recent list)."""
        recent = list(getattr(self._reflex, "events", ()) or ())
        if not recent:
            return []
        last_t = recent[-1].t_ms
        new = [e for e in recent if e.t_ms > self._reflex_seen]
        self._reflex_seen = max(self._reflex_seen, last_t)
        return new

    # ── telemetry ────────────────────────────────────────────────────────────

    def social_view(self) -> dict[str, Any]:
        odom = self._s_odom
        return {
            "version": SOCIAL_VERSION,
            "imprint": self.imprint.view(odom),
            "shelters": self.shelters.view(),
            "hide": self.hide_run.view(),
            "circle": self.circle.view(),
            "arcs": self.arcs.view(),
            "features": {k: round(v, 3) for k, v in self._sfeat.items()},
        }


def mom_goal_for_page(brain: SocialBrain) -> dict[str, Any] | None:
    """(page) its mother as it believes her to be: bearing, range, whether in sight."""
    view = brain.imprint.mom_view(brain._s_odom)
    if view is None:
        return None
    return {
        "bearing": round(view.bearing, 3),
        "range_m": round(view.range_m, 2),
        "seen": view.seen,
        "bond": round(view.bond, 2),
    }


def looks_by_track(pairs: Sequence[tuple[int, tuple[float, ...]]]) -> dict[int, tuple[float, ...]]:
    """(harness) a SocialInput's looks from (track, look) pairs."""
    return {int(track): tuple(look) for track, look in pairs}


__all__ = [
    "SOCIAL_BEHAVIOURS",
    "SOCIAL_FEATURES",
    "SOCIAL_VERSION",
    "SOCIAL_W",
    "SOCIAL_W_EXTRA",
    "SocialBrain",
    "SocialInput",
    "SocialSelector",
    "is_registered",
    "register",
    "registered",
    "social_gains",
    "social_of",
    "social_unlocked",
    "with_social",
]
