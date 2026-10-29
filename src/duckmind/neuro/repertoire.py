"""The duck's discrete repertoire as parametric motor programs (neuro:motor).

Every behaviour in :data:`duckmind.neuro.contract.BEHAVIOURS` has a *program*: a builder that,
at the start of each unit (one phrase of the behaviour), draws that unit's movement from the
bout's seed (``MotorCommand.variant``), the duck's style (:class:`Style`: its temperament,
branch, growth stage, innate side and quirk habit — Duck Core's diversity design) and the
command's intensity, tempo, amplitude, side and target. So a preen is never the same preen:
the side, the number of nibbles, their depth and rhythm, the hesitation before, all differ —
but a snuggler's preen stays low and slow and an explorer's looks are wide.

A bout lasts as long as the brain keeps the behaviour (it ends by satiation, not by a timer):
when a unit ends and the bout goes on, the next unit is drawn — the next glance, the next peck,
the next verse. Reactive programs (walking, foraging, investigating, following, sleeping,
listening…) rebuild their unit when their situation changes (walking ↔ stopped, awake ↔
asleep, a reply heard), so what the head does always fits what the body is doing.

The programs write head poses in Duck Core's clip units (:class:`~duckmind.neuro.contract.
HeadPose`: fractions of ``content.SEATED_ENVELOPE``, ``hy``/``hr`` + = the duck's left),
mouth opening, vocal cues, posture cues and — for the few behaviours that shape the walk
themselves (a body shake's shuffle, zoomies' circles) — a twist. :mod:`duckmind.neuro.motor`
overlays gaze, reflexes, liveliness, head stabilisation and bob, clamps everything to the
measured envelopes and robotd's walking rules, and turns it into robotd intents.

Pure: :func:`start` / :func:`step` take and return frozen state; randomness is the bout's own
seeded generator (:mod:`duckmind.neuro.signals`). Floats in the prototype; phase 5 converts.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from duckmind.i18n import msg
from duckmind.neuro import signals as S
from duckmind.neuro.contract import BEHAVIOURS, HeadPose, MotorCommand, SensorFrame

REPERTOIRE_VERSION = "neuro-repertoire-1"

# ── the duck's style ──────────────────────────────────────────────────────────

STAGE_AMP = {"hatchling": 0.60, "duckling": 0.75, "juvenile": 0.90, "adult": 1.00}  # content.STAGES
STAGE_PHRASE = {"hatchling": 1, "duckling": 2, "juvenile": 3, "adult": 5}  # content.STAGES
BRANCHES = ("songbird", "explorer", "snuggler", "dreamer")
FAVOURITE_TAGS = {  # content.BRANCHES favorite_tags
    "songbird": ("chirp", "greet"),
    "explorer": ("inquire",),
    "snuggler": ("coo",),
    "dreamer": ("coo",),
}


@dataclass(frozen=True, slots=True)
class Style:
    """How this duck moves: its temperament axes (0..1), its branch weights (the nature blend
    before the branch at GD 45, one branch after), its growth stage, innate side, favourite
    sound and quirk habit. Built by :func:`style_from`; the same duck always gets the same."""

    stage: str = "juvenile"
    side: int = 1  # the innate side: +1 its left, −1 its right (Duck Core quirk tilt_side)
    boldness: float = 0.5
    activity: float = 0.5
    sociability: float = 0.5
    exploration: float = 0.5
    wit: float = 0.5
    tempo: float = 0.5
    branch: tuple[tuple[str, float], ...] = (
        ("dreamer", 0.25),
        ("explorer", 0.25),
        ("snuggler", 0.25),
        ("songbird", 0.25),
    )
    favourite_tag: str = "chirp"
    habit: str = ""  # content.QUIRK_HABITS id

    def w(self, branch: str) -> float:
        return dict(self.branch).get(branch, 0.0)

    @property
    def amp_cap(self) -> float:
        return STAGE_AMP.get(self.stage, 0.9)

    @property
    def phrase(self) -> int:
        return STAGE_PHRASE.get(self.stage, 3)

    @property
    def young(self) -> float:
        """1 for a hatchling … 0 for an adult: clumsier, shorter attention, more wobble."""
        return {"hatchling": 1.0, "duckling": 0.6, "juvenile": 0.25, "adult": 0.0}.get(
            self.stage, 0.25
        )

    @property
    def yaw_scale(self) -> float:
        """Explorers turn their heads wide, snugglers stay small (content.BRANCHES style)."""
        return (
            1.0
            + 0.25 * self.w("explorer")
            - 0.15 * self.w("snuggler")
            + 0.3 * (self.exploration - 0.5)
        )

    @property
    def hold_scale(self) -> float:
        """How long it dwells: dreamers and snugglers linger, the young flit."""
        return (
            1.0
            + 0.45 * self.w("dreamer")
            + 0.3 * self.w("snuggler")
            - 0.1 * self.w("explorer")
            - 0.3 * self.young
        )

    @property
    def carriage(self) -> HeadPose:
        """Its resting head carriage (content.BRANCHES style): songbirds upright, snugglers
        low, dreamers gaze a little up and far."""
        np_ = -0.12 * self.w("songbird") + 0.18 * self.w("snuggler") + 0.04 * self.young
        hp = 0.10 * self.w("songbird") - 0.08 * self.w("snuggler") + 0.12 * self.w("dreamer")
        return HeadPose(np=np_, hp=hp)


def style_from(
    temperament: Mapping[str, float | int] | None = None,
    *,
    stage: str = "juvenile",
    side: int = 1,
    nature: Mapping[str, float | int] | None = None,
    branch: str | None = None,
    favourite_tag: str = "chirp",
    habit: str = "",
) -> Style:
    """A :class:`Style` from temperament axes 0..1 (``genome.Genome.axes``), the nature weights
    (before the branch) or the branch, the stage, the innate side (+1 left / −1 right), the
    favourite tag and the quirk habit. A real duck's comes from :func:`style_from_genome`."""
    axes = dict(temperament or {})
    if branch in BRANCHES:
        weights = {b: (0.85 if b == branch else 0.05) for b in BRANCHES}
    elif nature:
        raw = {b: max(0.0, float(nature.get(b, 0.0))) for b in BRANCHES}
        total = sum(raw.values()) or 1.0
        weights = {b: raw[b] / total for b in BRANCHES}
    else:
        weights = {b: 0.25 for b in BRANCHES}
    return Style(
        stage=stage if stage in STAGE_AMP else "juvenile",
        side=1 if side >= 0 else -1,
        boldness=S.norm_axis(axes.get("boldness")),
        activity=S.norm_axis(axes.get("activity")),
        sociability=S.norm_axis(axes.get("sociability")),
        exploration=S.norm_axis(axes.get("exploration")),
        wit=S.norm_axis(axes.get("wit")),
        tempo=S.norm_axis(axes.get("tempo")),
        branch=tuple(sorted(weights.items())),
        favourite_tag=favourite_tag,
        habit=habit,
    )


def style_from_genome(genome: Any) -> Style:
    """The :class:`Style` of a brain :class:`~duckmind.neuro.genome.Genome` — Duck Core's own
    draws for the seed (``genome.py`` is the prototype's ONE diversity mapping): its axes 0..1,
    its stage at its own pace, its branch (or the nature blend before it), innate side,
    favourite sound and quirk habit."""
    return style_from(
        dict(genome.axes),
        stage=str(genome.stage),
        side=int(genome.side),
        nature=dict(genome.nature),
        branch=genome.branch,
        favourite_tag=str(genome.favorite_tag),
        habit=str(genome.habit),
    )


# ── scores: keys, oscillations, cues ──────────────────────────────────────────

JOINTS = ("np", "hp", "hy", "hr", "m")


@dataclass(frozen=True, slots=True)
class Key:
    """Reach this pose ``t`` ms after the unit starts, along ``ease`` (signals.ease)."""

    t: float
    pose: tuple[float, float, float, float, float]  # np, hp, hy, hr, m
    ease: str = "mj"


@dataclass(frozen=True, slots=True)
class Osc:
    """A rhythmic overlay on one joint (index into :data:`JOINTS`) between ``t0`` and ``t1``."""

    joint: int
    amp: float
    hz: float
    t0: float
    t1: float
    phase: float = 0.0
    decay: float = 0.0  # 1/s


@dataclass(frozen=True, slots=True)
class Cue:
    """A timed event: ``vocal`` (tag, level ``a``), ``posture`` (tag sit/stand), ``twist``
    (vx ``a``, vyaw ``b`` for ``c`` ms), ``mark`` (tag = what it is doing, for the page)."""

    t: float
    kind: str
    tag: str = ""
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0


MOUTH_CLOSE_MS = 150.0  # a beak left open by the unit before closes this fast (review 2026-09-28)


@dataclass(frozen=True, slots=True)
class Score:
    keys: tuple[Key, ...]
    oscs: tuple[Osc, ...] = ()
    cues: tuple[Cue, ...] = ()
    length: float = 1000.0  # ms
    alive: float = 1.0  # × the idle micro-motion while it plays
    gaze: float = 0.3  # how much the brain's (non-reflex) gaze may take over the head
    label: str = ""
    mouth0: float = 0.0  # the beak's opening when the unit began (it closes within 150 ms)

    def pose_at(self, t: float) -> tuple[float, float, float, float, float]:
        keys = self.keys
        out = list(keys[-1].pose)
        if t <= keys[0].t:
            out = list(keys[0].pose)
        else:
            for k0, k1 in zip(keys, keys[1:], strict=False):
                if t <= k1.t:
                    u = (t - k0.t) / max(1.0, k1.t - k0.t)
                    e = S.ease(u, k1.ease)
                    out = [a + (b - a) * e for a, b in zip(k0.pose, k1.pose, strict=True)]
                    break
        for o in self.oscs:
            if o.t0 <= t <= o.t1:
                local = t - o.t0
                fade = min(1.0, local / 120.0, (o.t1 - t) / 120.0)
                out[o.joint] += (
                    o.amp
                    * math.exp(-o.decay * local / 1000.0)
                    * math.sin(2.0 * math.pi * o.hz * local / 1000.0 + o.phase)
                    * fade
                )
        if self.mouth0 > out[4] and t < MOUTH_CLOSE_MS:
            # a beak the last unit left open closes first (unless this unit opens it itself)
            u = S.min_jerk(t / MOUTH_CLOSE_MS)
            out[4] = self.mouth0 + (out[4] - self.mouth0) * u
        return (out[0], out[1], out[2], out[3], S.clamp(out[4], 0.0, 1.0))


class _B:
    """A score builder for one unit (mutable while building; the result is frozen)."""

    def __init__(
        self,
        rng: int,
        start: tuple[float, float, float, float, float],
        *,
        amp: float,
        dur: float,
        style: Style,
        cmd: MotorCommand,
    ) -> None:
        self.rng = rng
        self.amp = amp
        self.dur = dur  # ms multiplier (1 / tempo)
        self.style = style
        self.cmd = cmd
        self.t = 0.0
        # every unit starts with the beak shut (a unit that wants it open opens it itself): a
        # beak left open by a peck, a nibble or a preen that was cut short closes at once
        self.mouth0 = S.clamp(start[4], 0.0, 1.0)
        start = (start[0], start[1], start[2], start[3], 0.0)
        self.cur = start
        self.keys: list[Key] = [Key(0.0, start, "lin")]
        self.oscs: list[Osc] = []
        self.cues: list[Cue] = []

    # draws
    def u(self, low: float = 0.0, high: float = 1.0) -> float:
        self.rng, value = S.between(self.rng, low, high)
        return value

    def g(self) -> float:
        self.rng, value = S.gauss(self.rng)
        return value

    def ln(self, median: float, sigma: float = 0.35) -> float:
        self.rng, value = S.lognormal(self.rng, median, sigma)
        return value

    def p(self, probability: float) -> bool:
        self.rng, hit = S.chance(self.rng, probability)
        return hit

    def sign(self, p_left: float = 0.5) -> int:
        return 1 if self.p(p_left) else -1

    def ms(self, nominal: float, jitter: float = 0.12) -> float:
        """A duration: nominal × tempo × a little structured jitter (never below 60 ms)."""
        return max(60.0, nominal * self.dur * (1.0 + jitter * self.g()))

    # motion
    def to(
        self,
        ms: float,
        *,
        np: float | None = None,
        hp: float | None = None,
        hy: float | None = None,
        hr: float | None = None,
        m: float | None = None,
        ease: str = "mj",
        raw_ms: bool = False,
    ) -> _B:
        """Move to a pose (clip units before the amplitude; ``None`` keeps a joint)."""
        a = self.amp
        cur = self.cur
        target = (
            cur[0] if np is None else np * a,
            cur[1] if hp is None else hp * a,
            cur[2] if hy is None else hy * a,
            cur[3] if hr is None else hr * a,
            cur[4] if m is None else S.clamp(m, 0.0, 1.0),
        )
        if self.style.young > 0.5 and ease == "mj" and self.p(0.35 * self.style.young):
            ease = "lin"  # a hatchling's movement is less smooth
        self.t += ms if raw_ms else self.ms(ms)
        self.keys.append(Key(self.t, target, ease))
        self.cur = target
        return self

    def rel(
        self,
        ms: float,
        *,
        dnp: float = 0.0,
        dhp: float = 0.0,
        dhy: float = 0.0,
        dhr: float = 0.0,
        ease: str = "mj",
    ) -> _B:
        a = self.amp
        cur = self.cur
        target = (cur[0] + dnp * a, cur[1] + dhp * a, cur[2] + dhy * a, cur[3] + dhr * a, cur[4])
        self.t += self.ms(ms)
        self.keys.append(Key(self.t, target, ease))
        self.cur = target
        return self

    def hold(self, ms: float, sigma: float = 0.3, drift: float = 0.04) -> _B:
        """Dwell (log-normal around ``ms`` × the style's dwell), drifting a little."""
        length = max(80.0, self.ln(ms * self.dur * self.style.hold_scale, sigma))
        cur = self.cur
        if drift > 0.0:
            d = drift * self.amp
            cur = (
                cur[0] + d * 0.4 * self.g(),
                cur[1] + d * 0.5 * self.g(),
                cur[2] + d * self.g(),
                cur[3] + d * 0.4 * self.g(),
                cur[4],
            )
        self.t += length
        self.keys.append(Key(self.t, cur, "soft"))
        self.cur = cur
        return self

    def home(self, ms: float = 500.0, ease: str = "mj") -> _B:
        return self.to(ms, np=0.0, hp=0.0, hy=0.0, hr=0.0, m=0.0, ease=ease)

    def settle(self, ms: float = 500.0) -> _B:
        """End a unit where it is, relaxed — not back at HOME (the review, 2026-09-28: acts that
        all recentred the head swung it out and back 2–3 times a minute, like a machine): a tilt
        lets go, the neck and head pitch ease halfway back, the head keeps looking about where
        it looked (the next gaze shift moves it on), the beak shuts."""
        cur = self.cur
        target = (0.5 * cur[0], 0.5 * cur[1], 0.8 * cur[2], 0.0, 0.0)
        self.t += self.ms(ms)
        self.keys.append(Key(self.t, target, "mj"))
        self.cur = target
        return self

    def osc(
        self,
        joint: str,
        amp: float,
        hz: float,
        ms: float,
        *,
        decay: float = 0.0,
        phase: float = 0.0,
        advance: bool = True,
        scale: bool = True,
    ) -> _B:
        length = self.ms(ms)
        index = JOINTS.index(joint)
        self.oscs.append(
            Osc(
                index,
                amp * (self.amp if scale and joint != "m" else 1.0),
                hz / self.dur,
                self.t,
                self.t + length,
                phase,
                decay,
            )
        )
        if advance:
            self.t += length
            self.keys.append(Key(self.t, self.cur, "soft"))
        return self

    def mouth_pulses(self, n: int, depth: float, open_ms: float, close_ms: float) -> _B:
        for _ in range(max(0, n)):
            self.to(open_ms, m=depth * (0.8 + 0.25 * self.u()), ease="snap")
            self.to(close_ms, m=0.0, ease="soft")
        return self

    def breathe(self, amp: float, hz: float) -> _B:
        """Slow breathing on the neck over the whole unit so far (not scaled: it is tiny)."""
        self.oscs.append(Osc(0, amp, hz, 0.0, max(1.0, self.t)))
        return self

    # cues
    def vocal(self, tag: str, level: float = 0.6, dt: float = 0.0) -> _B:
        if tag:
            self.cues.append(Cue(self.t + dt, "vocal", tag, S.clamp(level, 0.05, 1.0)))
        return self

    def posture(self, which: str) -> _B:
        self.cues.append(Cue(self.t, "posture", which))
        return self

    def twist(self, vx: float, vyaw: float, ms: float) -> _B:
        self.cues.append(Cue(self.t, "twist", "", vx, vyaw, ms))
        return self

    def mark(self, label: str) -> _B:
        self.cues.append(Cue(self.t, "mark", label))
        return self

    def wait(self, ms: float) -> _B:
        self.t += ms
        self.keys.append(Key(self.t, self.cur, "soft"))
        return self

    def tag(self, options: tuple[str, ...], silent: float = 1.0) -> str:
        """A sound from ``options`` ("" = silent), leaning to its favourite and its branch's."""
        weights: dict[str, float] = {}
        for option in options:
            if option == "":
                weights["~"] = weights.get("~", 0.0) + silent
                continue
            w = 1.0
            if option == self.style.favourite_tag:
                w *= 2.5
            for branch, favourites in FAVOURITE_TAGS.items():
                if option in favourites:
                    w *= 1.0 + 0.6 * self.style.w(branch)
            weights[option] = weights.get(option, 0.0) + w
        if not weights:
            return ""
        self.rng, key = S.pick_weighted(self.rng, weights)
        return "" if key == "~" else key

    def score(self, *, alive: float = 1.0, gaze: float = 0.3, label: str = "") -> Score:
        length = max(self.t, self.keys[-1].t, max((o.t1 for o in self.oscs), default=0.0))
        return Score(
            tuple(self.keys),
            tuple(self.oscs),
            tuple(sorted(self.cues, key=lambda c: c.t)),
            length + 1.0,
            alive,
            gaze,
            label,
            self.mouth0,
        )


# ── program context and builders ──────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Ctx:
    style: Style
    cmd: MotorCommand
    frame: SensorFrame
    mode: str
    unit: int


Builder = Callable[[_B, Ctx], Score]

SLEEP_POSE = (0.8, -0.7)  # content.NAP_POSE: np, hp


def _side(b: _B, ctx: Ctx, p_innate: float = 0.7) -> int:
    """The command's side (the stimulus's or the innate one), sometimes the other way."""
    base = ctx.cmd.side if ctx.cmd.side in (-1, 1) else ctx.style.side
    return base if b.p(p_innate) else -base


# A shake has to be slow enough and big enough for the head to perform it in the posture it is
# done in (fid:motor 2026-09-28; fid:audit: a 3–4 Hz head shake standing, 15° commanded → 1.6°
# performed). The standing head is a low-pass filter (A100 yaw sines at head_alpha 1.0: gain 0.77
# at 1 Hz, 0.63 at 1.5 Hz, 0.50 at 2 Hz, 0.32 at 3 Hz; pet:body at 0.2: 0.30 at 2 Hz); seated it
# follows up to ~3 Hz (0.91 at 3 Hz) but ≥ 2 Hz seated sines strain the head servos (A100: head
# torque 0.26–0.47 N·m, feet shuffling), so a seated shake stays under ~2.8 Hz with a moderate
# swing (its peak rate under the motor's 5 rad/s seated limit). Rates are AFTER the tempo.
# While WALKING a shake stays inside what was tested with the legs going (pet:probe: a 1.2-Hz head
# motion and ±1.0-rad snaps while walking, 0/1,080 falls; the A100 scans went to 1 Hz).
SHAKE_HZ = {"stand": (1.5, 2.0), "walk": (1.0, 1.2), "sit": (2.2, 2.8)}
SHAKE_AMP = {"stand": 1.0, "walk": 0.8, "sit": 0.55}  # × the nominal swing: the gain made up for


def _shake(b: _B, ctx: Ctx) -> tuple[float, float]:
    """(the rate to pass to :meth:`_B.osc` — before the tempo — and the swing factor) for a shake
    in the body's posture now (see SHAKE_HZ)."""
    posture = ctx.frame.body.posture
    key = posture if posture in ("stand", "walk") else "sit"
    low, high = SHAKE_HZ[key]
    return b.u(low, high) * b.dur, SHAKE_AMP[key]


def _target_yaw(ctx: Ctx) -> float | None:
    """The target's bearing as head yaw in clip units (±1 = ±0.6 rad), or None."""
    target = ctx.cmd.target
    if target is None:
        return None
    return S.clamp(target.bearing / 0.6, -1.0, 1.0)


def _look(
    b: _B,
    hy: float,
    hp: float = 0.0,
    *,
    sacc: float = 260.0,
    fix: float = 800.0,
    np: float | None = None,
    hr: float | None = None,
) -> None:
    """A saccade and a fixation, with the small corrective second saccade animals make."""
    ys = b.style.yaw_scale
    over = 1.0 + 0.12 * b.g() * (0.5 + b.style.young)
    b.to(sacc, hy=S.clamp(hy * ys * over, -1.0, 1.0), hp=hp, np=np, hr=hr, ease="snap")
    if abs(over - 1.0) > 0.08:
        b.to(90.0, hy=S.clamp(hy * ys, -1.0, 1.0), ease="soft")
    b.hold(fix)


def build_look_around(b: _B, ctx: Ctx) -> Score:
    st = ctx.style
    n = 2 + int(b.u(0, 1.2 + 1.5 * st.exploration + 0.4 * st.phrase / 5 + ctx.cmd.intensity))
    side = _side(b, ctx, 0.6)
    previous = 0.0
    for i in range(n):
        side = -side if b.p(0.8) else side
        yaw = side * b.u(0.35, 0.9)
        if abs(yaw - previous) < 0.3:
            yaw = -yaw
        pitch = b.u(-0.15, 0.35) + 0.15 * st.w("dreamer")
        _look(
            b,
            yaw,
            pitch,
            sacc=b.u(170, 320),
            fix=b.u(450, 1200),
            hr=0.35 * side * b.u(0, 1) * st.w("explorer") if b.p(0.3) else 0.0,
        )
        if st.habit == "double_look" and i == n - 1 and b.p(0.7):
            _look(b, -yaw * 0.6, pitch, sacc=200, fix=300)
            _look(b, yaw, pitch, sacc=220, fix=600)
        previous = yaw
    tag = b.tag(("", "chirp", "inquire"), silent=2.0)
    b.vocal(tag, 0.4 + 0.3 * ctx.cmd.intensity)
    b.settle(b.u(350, 600))  # it stays looking about where it looked last
    return b.score(label="左看右看")


def _tilt_side(b: _B, side: int) -> int:
    """(fid:fix) A tilt goes to its side unless the head already leans that way (then there is
    no room for it to be seen: the other side)."""
    return -side if b.cur[3] * side > 0.35 else side


def build_perk_tilt(b: _B, ctx: Ctx) -> Score:
    side = ctx.style.side if b.p(0.85) else -ctx.style.side  # its innate side, almost always
    side = _tilt_side(b, side)
    # (fid:fix: up from where the head IS — the brain asks for it only with room above; the
    # review: 抬頭歪頭 seen 61/95, an absolute pose from a raised head barely moved it)
    b.rel(b.u(300, 480), dhp=b.u(0.4, 0.8), dnp=-b.u(0.0, 0.3), ease="snap")
    b.to(b.u(300, 500), hr=side * b.u(0.55, 0.95), hy=side * b.u(0.05, 0.3))
    if b.p(0.6):
        b.vocal(b.tag(("inquire",), silent=0.6), 0.5)
    b.hold(b.u(900, 1800))
    if b.p(0.35):  # a second, smaller re-tilt: "hm?"
        b.rel(260, dhr=-side * 0.25).rel(300, dhr=side * 0.3)
        b.hold(600)
    b.settle(b.u(450, 700))
    return b.score(label="歪頭")


def build_stretch_yawn(b: _B, ctx: Ctx) -> Score:
    standing = ctx.frame.body.posture in ("stand", "walk")
    cap = 0.8 if standing else 1.0  # expression_caps: stretch_yawn stands at 0.8
    b.amp *= cap
    b.to(b.u(700, 1000), np=b.u(-0.6, -0.3), hp=b.u(0.6, 1.0), m=b.u(0.6, 1.0))
    b.hold(b.u(700, 1400), drift=0.02)
    b.to(b.u(500, 750), np=b.u(0.1, 0.3), hp=b.u(-0.3, -0.1), m=0.0)
    b.hold(b.u(300, 600))
    b.vocal(b.tag(("coo",), silent=0.8), 0.4)
    if b.p(0.3):
        hz, swing = _shake(b, ctx)
        b.osc("hr", swing * 0.7, hz, 800, decay=1.2)  # a little shake-off
    b.settle(b.u(600, 900))
    return b.score(label=msg("neuro.act.stretch_yawn.name"))


def build_preen(b: _B, ctx: Ctx) -> Score:
    st = ctx.style
    if st.habit == "tilt_before_preen" and b.p(0.8):
        b.to(350, hr=st.side * 0.7, hp=0.3).hold(500)
    side = _side(b, ctx, 0.7)
    sets = 1 + (1 if b.p(0.3 + 0.1 * st.phrase / 5) else 0)
    for s in range(sets):
        if s:
            side = -side
        b.to(
            b.u(550, 800),
            hy=side * b.u(0.7, 1.0),
            hp=b.u(-0.9, -0.5),
            np=b.u(0.4, 0.75),
            hr=side * b.u(0.0, 0.4),
        )
        # nibbling preening (pet:etho F1: 2–10 s per site, the beak chattering 3–6 Hz): the
        # beak is the raw actuator, fast enough to chatter; the head only shifts slowly on
        # the site (its bandwidth is ~1 Hz), and now and then it looks up (every 5–20 s)
        for _ in range(1 + int(b.u(0, 1.5 + ctx.cmd.intensity))):
            chatter = b.u(2000, 6000)
            start = b.t
            b.to(200, m=0.25, raw_ms=True)
            b.oscs.append(Osc(4, b.u(0.15, 0.25), b.u(3.0, 6.0), start, start + chatter, 0.0))
            steps = max(1, int(chatter / b.u(700, 1400)))
            for _k in range(steps):
                b.rel(chatter / steps, dhy=0.12 * b.g(), dhp=0.08 * b.g(), ease="soft")
            b.to(150, m=0.0, raw_ms=True)
            if b.p(0.3):
                b.to(b.u(300, 450), hp=b.u(0.1, 0.3), np=0.0, hy=side * 0.2).hold(b.u(500, 1200))
                b.to(
                    b.u(400, 600),
                    hy=side * b.u(0.7, 1.0),
                    hp=b.u(-0.9, -0.5),
                    np=b.u(0.4, 0.75),
                )
    b.settle(b.u(500, 800))
    return b.score(label=msg("neuro.act.preen.name"), alive=0.5)


def build_happy_sway(b: _B, ctx: Ctx) -> Score:
    hz = b.u(0.9, 1.4)
    cycles = int(b.u(2, 4 + ctx.cmd.intensity))
    b.to(b.u(250, 400), hp=b.u(0.2, 0.4), m=0.2)
    start = b.t
    b.osc("hr", b.u(0.5, 0.9), hz, cycles / hz * 1000.0, phase=0.0)
    for i in range(cycles * 2):
        if i % 2 == 0:
            b.cues.append(Cue(start + i * 500.0 / hz * b.dur, "mark", "晃"))
    b.oscs.append(Osc(4, 0.25, hz * 2 / b.dur, start, b.t, 0.0))  # the beak keeps the beat
    tag = b.tag(("chirp",), silent=0.5)
    if tag:
        b.cues.append(Cue(start + 200.0, "vocal", tag, 0.55))
        if ctx.style.habit == "extra_chirp":
            b.cues.append(Cue(start + 900.0 * b.dur, "vocal", "chirp", 0.5))
    b.settle(b.u(350, 550))
    return b.score(label=msg("neuro.act.happy_sway.name"))


def build_tuck_away(b: _B, ctx: Ctx) -> Score:
    away = -(ctx.cmd.side if ctx.cmd.side in (-1, 1) else ctx.style.side)
    b.to(b.u(450, 700), hy=away * b.u(0.6, 0.95), np=b.u(0.5, 0.8), hp=b.u(-0.6, -0.3))
    b.hold(b.u(2000, 3500), drift=0.02)
    if b.p(0.4 + 0.3 * ctx.style.sociability):  # a shy peek back
        b.rel(350, dhy=-away * 0.35, dhp=0.2).hold(500).rel(350, dhy=away * 0.35, dhp=-0.2)
        b.hold(700)
    b.vocal(b.tag(("", "coo"), silent=2.0), 0.3)
    b.settle(b.u(700, 1100))
    return b.score(label=msg("neuro.act.tuck_away.name"), alive=0.4)


def _sleep_stir(b: _B, ctx: Ctx) -> None:
    np0, hp0 = SLEEP_POSE
    b.rng, kind = S.pick_weighted(
        b.rng, {"shift": 3.0, "nuzzle": 2.0, "shiver": 1.0, "mumble": 1.5, "peek": 0.7}
    )
    if kind == "shift":
        b.to(600, np=np0 - 0.25, hp=hp0 + 0.25, hr=0.25 * b.sign())
        b.to(800, np=np0 - 0.2, hp=hp0 + 0.2, hr=-0.2 * b.sign())
    elif kind == "nuzzle":
        b.to(700, np=min(1.0, np0 + 0.15), hp=hp0 - 0.1, hr=0.5 * b.sign())
        b.to(800, hr=-0.3 * b.sign())
    elif kind == "shiver":
        b.osc("hy", 0.2, 6.0, 600, decay=3.0)
    elif kind == "mumble":
        b.to(400, m=0.3).to(400, m=0.0).to(400, m=0.25).to(400, m=0.0)
        if ctx.cmd.behaviour == "sleep_talk" or b.p(0.3):
            b.vocal("coo", 0.15, dt=-800)
    else:  # a sleepy peek
        b.to(800, np=0.2, hp=-0.05).to(800, hy=0.5 * b.sign()).to(800, hy=-0.4 * b.sign())
    b.to(900, np=np0, hp=hp0, hr=0.0, hy=0.0, m=0.0)


def build_nap(b: _B, ctx: Ctx) -> Score:
    np0, hp0 = SLEEP_POSE
    b.posture("sit")
    if ctx.unit == 0:  # falls asleep in two or three steps, with a catch on the way
        b.to(b.u(900, 1500), np=0.3, hp=-0.3, hy=0.0, hr=0.0)
        b.hold(b.u(1200, 2500), drift=0.01)
        if b.p(0.6):
            b.to(220, np=0.1, hp=-0.1, ease="snap").hold(500)  # nodding off, caught
        b.to(b.u(1000, 1600), np=0.6, hp=-0.5)
        b.hold(b.u(1000, 2500), drift=0.01)
        b.to(b.u(1200, 1800), np=np0, hp=hp0)
    else:
        b.hold(b.ln(45_000.0 / b.dur, 0.6), drift=0.0)  # pet:etho G3: peeks every ~1–2 min
        _sleep_stir(b, ctx)
    b.breathe(0.03, 0.22)
    return b.score(label=msg("neuro.loco.1de4"), alive=0.15, gaze=0.0)


def build_sleep_talk(b: _B, ctx: Ctx) -> Score:
    np0, hp0 = SLEEP_POSE
    b.posture("sit")
    b.to(b.u(800, 1200), np=np0, hp=hp0)
    b.hold(b.u(1500, 4000), drift=0.0)
    b.to(300, m=b.u(0.2, 0.35)).vocal("coo", 0.15).to(350, m=0.0)
    if b.p(0.5):
        b.hold(600).to(250, m=0.25).to(300, m=0.0)
    b.hold(b.u(2000, 5000), drift=0.0)
    return b.score(label=msg("neuro.act.sleep_talk.name"), alive=0.15, gaze=0.0)


def build_wake_ritual(b: _B, ctx: Ctx) -> Score:
    longer = 1.4 if ctx.style.habit == "stretch_after_nap" else 1.0
    b.to(b.u(800, 1100) * longer, np=-0.5, hp=0.9, m=b.u(0.7, 1.0))
    b.hold(b.u(700, 1200) * longer, drift=0.02)
    b.to(500, np=0.0, hp=0.0, m=0.0)
    b.vocal(b.tag(("coo",), silent=0.3), 0.35)
    _look(b, 0.7 * b.sign(), 0.1, fix=600)
    _look(b, -0.6 * b.sign(), 0.1, fix=600)
    b.to(400, hp=0.3, m=0.5).vocal(b.tag(("chirp",), silent=0.1), 0.6).to(300, m=0.0)
    b.settle(500)
    return b.score(label="醒來了")


def build_nod(b: _B, ctx: Ctx) -> Score:
    n = 1 + int(b.u(0, 2.2 + ctx.cmd.intensity))
    for i in range(n):
        depth = b.u(0.4, 0.8) * (1.0 if i == 0 else b.u(0.6, 0.95))
        b.to(b.u(300, 450), hp=-depth, np=b.u(0.0, 0.2))
        b.to(b.u(300, 450), hp=0.05 * b.g(), np=0.0)
        if b.p(0.3):
            b.hold(250)
    if b.p(0.3):
        b.to(350, hr=ctx.style.side * 0.4).hold(400)
    b.settle(400)
    return b.score(label=msg("neuro.act.nod.name"))


def build_peekaboo(b: _B, ctx: Ctx) -> Score:
    rounds = 1 + int(b.u(0, 1.5 + ctx.cmd.intensity))
    side = _side(b, ctx, 0.6)
    target = _target_yaw(ctx) or 0.0
    for _ in range(rounds):
        b.to(b.u(350, 550), hy=side * b.u(0.8, 0.95), np=b.u(0.4, 0.6), hp=b.u(-0.5, -0.3))
        b.hold(b.u(800, 1800), drift=0.01)
        b.to(b.u(180, 280), hy=target, hp=b.u(0.4, 0.6), np=0.0, m=0.45, ease="snap")
        b.vocal(b.tag(("inquire", "chirp"), silent=0.3), 0.55)
        b.to(200, m=0.0).hold(b.u(500, 900))
        side = -side if b.p(0.5) else side
    b.settle(450)
    return b.score(label=msg("neuro.act.peekaboo.name"), gaze=0.1)


def build_peekaboo_master(b: _B, ctx: Ctx) -> Score:
    side = _side(b, ctx, 0.5)
    for _ in range(2 + int(b.u(0, 2))):
        b.to(b.u(250, 380), hy=side * 0.9, np=0.5, hp=-0.4)
        b.hold(b.u(500, 1100))
        if b.p(0.5):  # a feint: starts out one side, pops out of the other
            b.rel(160, dhy=-side * 0.3).rel(160, dhy=side * 0.3)
        b.to(b.u(160, 240), hy=-side * 0.2, hp=0.5, np=0.0, m=0.4, ease="snap")
        b.vocal(b.tag(("inquire", "chirp"), silent=0.2), 0.6)
        b.to(200, m=0.0).hold(b.u(350, 700))
        side = -side
    b.settle(450)
    return b.score(label=msg("neuro.act.peekaboo_master.name"), gaze=0.1)


def build_call_response(b: _B, ctx: Ctx) -> Score:
    side = _side(b, ctx, 0.5)
    if ctx.mode == "reply":  # a sound came while it listened: answer at once
        b.to(b.u(150, 250), hp=0.5, hr=side * 0.4, m=0.8, ease="snap")
        b.vocal(b.tag(("chirp", "inquire"), silent=0.0), 0.6)
        b.to(250, m=0.0).hold(b.u(400, 700))
        return b.score(label="回答")
    b.to(b.u(250, 400), hp=0.5, hr=side * 0.4, m=0.8)
    b.vocal(b.tag(("inquire", "chirp"), silent=0.0), 0.6)
    b.to(250, m=0.0)
    b.mark("聽")
    b.to(400, hr=-side * 0.5, hp=0.35)
    b.hold(b.u(1400, 2600), drift=0.01)
    b.settle(400)
    return b.score(label=msg("neuro.act.call_response.name"), alive=0.4)


def build_snuggle(b: _B, ctx: Ctx) -> Score:
    side = _side(b, ctx, 0.7)
    # (fid:motor 2026-09-28: standing, a 2–5° tilt and rub vanished in the gearboxes — the head
    # now leans low into the tilt and rubs from side to side above the deadband)
    b.to(b.u(700, 1100), np=b.u(0.5, 0.75), hp=b.u(-0.6, -0.35), hr=side * b.u(0.6, 0.9))
    b.osc("hy", b.u(0.2, 0.3), b.u(0.4, 0.7), b.u(2000, 4500) * (0.8 + ctx.style.sociability))
    b.vocal(b.tag(("coo",), silent=0.5), 0.35)
    b.hold(b.u(600, 1400))
    b.settle(b.u(700, 1000))
    return b.score(label=msg("neuro.act.snuggle.name"), alive=0.5, gaze=0.5)


def build_little_dance(b: _B, ctx: Ctx) -> Score:
    hz = b.u(1.2, 2.0)
    beats = int(b.u(4, 7 + 2 * ctx.cmd.intensity))
    side = _side(b, ctx, 0.5)
    for i in range(beats):
        s = side if i % 2 == 0 else -side
        b.to(
            500.0 / hz, hr=s * b.u(0.45, 0.7), hy=s * b.u(0.15, 0.4), hp=b.u(0.1, 0.35), ease="soft"
        )
        if b.p(0.3):
            b.vocal(b.tag(("chirp", "wheee"), silent=1.0), 0.5)
    if b.p(0.4):  # a flourish
        b.to(400, np=-0.4, hp=0.6, hr=0.0, hy=0.0).hold(600)
    b.settle(400)
    return b.score(label=msg("neuro.act.little_dance.name"))


def build_not_now(b: _B, ctx: Ctx) -> Score:
    hz = b.u(1.2, 1.8)
    standing = ctx.frame.body.posture in ("stand", "walk")
    b.to(250, np=b.u(0.05, 0.2), hp=b.u(-0.2, 0.0))
    # (fid:motor: standing the head follows 1.2–1.8 Hz at ~0.6–0.7: a wider "no")
    swing = b.u(0.6, 0.85) if standing else b.u(0.3, 0.55)
    b.osc("hy", swing, hz, b.u(2.0, 3.0) / hz * 1000.0, decay=0.4)
    away = -(ctx.cmd.side if ctx.cmd.side in (-1, 1) else ctx.style.side)
    b.to(400, hy=away * 0.35, np=0.2).hold(500)
    b.settle(500)
    return b.score(label=msg("neuro.act.not_now.name"), alive=0.6)


def build_little_song(b: _B, ctx: Ctx) -> Score:
    notes = 2 + int(b.u(0, ctx.style.phrase + 1.5 * ctx.style.w("songbird") + 1))
    b.to(b.u(400, 550), np=-0.3, hp=b.u(0.6, 0.8))
    for i in range(notes):
        opening = b.u(0.5, 0.9)
        b.to(b.u(150, 260), m=opening, ease="snap")
        b.vocal(b.tag(("chirp", "greet"), silent=0.0), 0.55 + 0.1 * math.sin(i))
        b.to(b.u(160, 300), m=0.0)
        if b.p(0.35):
            b.rel(250, dhr=0.25 * b.sign())
        b.wait(b.ln(160, 0.4))
    b.settle(b.u(500, 700))
    return b.score(label="唱歌", alive=0.6)


def build_self_duet(b: _B, ctx: Ctx) -> Score:
    side = _side(b, ctx, 0.5)
    for i in range(2 + int(b.u(0, 2.5))):
        s = side if i % 2 == 0 else -side
        b.to(b.u(300, 450), hy=s * b.u(0.5, 0.7), hp=0.2 + 0.2 * (i % 2), m=0.6)
        b.vocal("chirp" if i % 2 == 0 else "inquire", 0.55)
        b.to(250, m=0.0).hold(b.u(250, 500))
    b.settle(450)
    return b.score(label=msg("neuro.act.self_duet.name"))


def build_survey(b: _B, ctx: Ctx) -> Score:
    side = _side(b, ctx, 0.5)
    b.to(b.u(900, 1400), hy=side * b.u(0.85, 1.0), hp=b.u(0.1, 0.3))
    b.hold(b.u(500, 900))
    stops = 1 + int(b.u(0, 2.5))
    for k in range(stops):
        frac = (k + 1) / (stops + 1)
        b.to(b.u(600, 900), hy=side * (1.0 - 2.0 * frac) * 0.95, hp=b.u(0.05, 0.35))
        b.hold(b.u(400, 800))
    b.to(b.u(900, 1300), hy=-side * b.u(0.85, 1.0), hp=b.u(0.1, 0.3))
    b.hold(b.u(500, 900))
    b.vocal(b.tag(("inquire",), silent=1.0), 0.5)
    b.settle(b.u(600, 900))
    return b.score(label=msg("neuro.act.survey.name"), gaze=0.2)


def build_melt(b: _B, ctx: Ctx) -> Score:
    side = _side(b, ctx, 0.7)
    b.to(b.u(1500, 2500), np=b.u(0.6, 0.9), hp=b.u(-0.8, -0.5), hr=side * b.u(0.4, 0.7))
    b.vocal(b.tag(("coo",), silent=0.3), 0.3)
    b.hold(b.u(2500, 5000), drift=0.01)
    return b.score(label=msg("neuro.act.melt.name"), alive=0.25, gaze=0.0)


def build_nuzzle_tilt(b: _B, ctx: Ctx) -> Score:
    side = _side(b, ctx, 0.7)
    b.to(b.u(550, 800), np=b.u(0.5, 0.65), hp=b.u(-0.35, -0.2))
    b.osc("hy", b.u(0.25, 0.35), b.u(0.8, 1.2), b.u(1600, 2600), decay=0.2)
    b.to(b.u(450, 650), hp=0.3, np=0.0, hr=side * b.u(0.6, 0.9), hy=0.0)
    b.vocal(b.tag(("coo",), silent=0.4), 0.35)
    b.hold(b.u(600, 1100))
    b.settle(600)
    return b.score(label=msg("neuro.act.nuzzle_tilt.name"))


def build_moon_gaze(b: _B, ctx: Ctx) -> Score:
    b.to(b.u(1500, 2400), np=b.u(-0.6, -0.4), hp=b.u(0.8, 1.0), hy=0.2 * b.g())
    b.hold(b.u(3000, 7000), sigma=0.4, drift=0.015)
    if b.p(0.3):
        b.rel(1500, dhy=0.2 * b.sign()).hold(1500, drift=0.01)
    b.settle(b.u(1200, 1800))
    return b.score(label=msg("neuro.act.moon_gaze.name"), alive=0.2, gaze=0.1)


def build_stand_up(b: _B, ctx: Ctx) -> Score:
    b.to(b.u(300, 500), np=b.u(0.2, 0.35), hp=b.u(-0.25, -0.1))
    b.posture("stand")
    b.to(b.u(600, 800), np=-0.3, hp=b.u(0.35, 0.55))
    b.to(b.u(500, 700), np=0.0, hp=0.3)
    if b.p(0.5):
        _look(b, 0.5 * b.sign(), 0.2, fix=400)
    b.vocal(b.tag(("", "chirp"), silent=2.0), 0.45)
    b.settle(500)
    b.wait(800)
    return b.score(label=msg("neuro.act.stand_up.name"), alive=0.6)


def build_sit_down(b: _B, ctx: Ctx) -> Score:
    b.to(b.u(300, 450), np=b.u(0.2, 0.35), hp=b.u(0.1, 0.25))
    b.posture("sit")
    b.to(b.u(700, 900), np=0.2, hp=-0.2)
    b.vocal(b.tag(("", "coo"), silent=2.0), 0.35)
    b.to(b.u(600, 900), np=b.u(0.05, 0.2), hp=b.u(-0.15, 0.0))
    b.wait(1200)
    return b.score(label=msg("neuro.act.sit_down.name"), alive=0.6)


def _walk_head(b: _B, ctx: Ctx, *, low: float = 0.0, look_p: float = 0.35) -> None:
    """A walking unit (~1.5–3 s): eyes ahead with a slow wander, sometimes a glance at the side
    (邊走邊看). The motor adds the lead into turns, stabilisation and bob."""
    length = b.u(1400, 2800)
    if b.p(look_p * (0.6 + 0.8 * ctx.style.exploration)):
        s = b.sign()
        b.to(
            b.u(200, 320),
            hy=s * b.u(0.35, 0.75),
            hp=b.u(-0.1, 0.25) - low,
            np=0.1 * low,
            ease="snap",
        )
        b.hold(b.u(400, 900))
        b.to(b.u(300, 450), hy=0.25 * b.g(), hp=-low, np=0.3 * low)
    else:
        b.to(b.u(500, 900), hy=0.25 * b.g(), hp=0.08 * b.g() - low, np=0.3 * low, hr=0.0)
        b.wait(max(200.0, length - b.t))


def _pause_head(b: _B, ctx: Ctx) -> str:
    """What a stop between two legs looks like (a pause verb, chosen by its style)."""
    st = ctx.style
    weights = {
        "look": 1.0 + st.exploration + 0.5 * st.w("explorer"),
        "peek_down": 0.7 + 0.4 * st.activity,
        "glance": 0.9,
        "tilt": 0.5 + 0.8 * st.w("explorer"),
        "look_up": 0.3 + 0.9 * st.w("dreamer"),
        "chirp": 0.2 + 0.8 * st.w("songbird"),
    }
    b.rng, kind = S.pick_weighted(b.rng, weights)
    if kind == "look":
        side = b.sign()
        _look(b, side * b.u(0.4, 0.85), b.u(0.0, 0.3), fix=b.u(400, 900))
        if b.p(0.6):
            _look(b, -side * b.u(0.4, 0.85), b.u(0.0, 0.3), fix=b.u(400, 900))
    elif kind == "peek_down":
        b.to(b.u(350, 500), np=b.u(0.5, 0.8), hp=b.u(-0.9, -0.6)).hold(b.u(700, 1300))
        if b.p(0.4):
            b.rel(400, dhy=0.4 * b.sign()).hold(600)
    elif kind == "glance":
        _look(b, b.sign() * b.u(0.6, 0.95), 0.1 * b.g(), fix=b.u(500, 1500))
    elif kind == "tilt":
        s = b.sign()
        b.to(b.u(300, 450), hp=b.u(0.2, 0.5), hr=s * b.u(0.6, 0.95), hy=s * b.u(0.3, 0.6))
        b.hold(b.u(700, 1600))
    elif kind == "look_up":
        b.to(b.u(500, 800), np=-0.4, hp=b.u(0.7, 1.0)).hold(b.u(900, 2000))
    else:
        b.to(250, hp=0.4, m=0.6).vocal(b.tag(("chirp", "inquire"), silent=0.2), 0.5)
        b.to(250, m=0.0).hold(400)
    b.settle(b.u(300, 500))
    return kind


def build_wander(b: _B, ctx: Ctx) -> Score:
    if ctx.mode == "walk":
        _walk_head(b, ctx)
        if b.p(0.06 + 0.1 * ctx.style.w("songbird")):
            b.vocal(b.tag(("", "inquire", "chirp"), silent=0.5), 0.4)
        return b.score(label=msg("neuro.act.wander.name"), gaze=0.6)
    kind = _pause_head(b, ctx)
    return b.score(label=f"停一下（{kind}）", gaze=0.5)


def build_turn_around(b: _B, ctx: Ctx) -> Score:
    turning = ctx.cmd.vyaw if abs(ctx.cmd.vyaw) > 0.05 else ctx.cmd.side * 0.8
    s = 1 if turning > 0 else -1
    if ctx.mode == "walk":  # looks over its shoulder into the turn, then settles ahead
        b.to(b.u(250, 400), hy=s * b.u(0.6, 0.9), hp=b.u(0.0, 0.2), ease="snap")
        b.hold(b.u(700, 1300))
        b.to(b.u(500, 800), hy=s * 0.25)
        if b.p(0.3):
            b.vocal(b.tag(("", "inquire"), silent=1.5), 0.4)
        return b.score(label="轉身", gaze=0.4)
    _look(b, -s * b.u(0.5, 0.85), 0.15, fix=b.u(600, 1200))  # a look back where it came from
    b.settle(400)
    return b.score(label="回頭看看", gaze=0.4)


def build_peek_down(b: _B, ctx: Ctx) -> Score:
    # (fid:fix: down from where the head IS — the brain asks for it only with room below; the
    # review: 低頭看地上 reached ≥ 10° in 140/232, an absolute pose from a head already low (a
    # forage, a peck) barely moved it)
    b.rel(b.u(400, 550), dnp=b.u(0.25, 0.45), dhp=-b.u(0.6, 0.9))
    b.hold(b.u(800, 1400))
    if b.p(0.6):
        b.rel(b.u(400, 600), dhy=b.sign() * b.u(0.3, 0.6)).hold(b.u(700, 1200))
    b.settle(b.u(450, 650))
    return b.score(label="低頭看")


def build_glance(b: _B, ctx: Ctx) -> Score:
    target = _target_yaw(ctx)
    side = _side(b, ctx, 0.5)
    if b.cur[2] * side > 0.3:
        side = -side  # (fid:fix) the head already looks that way: a glance the other way shows
    yaw = target if target is not None else side * b.u(0.6, 0.95)
    _look(b, yaw, 0.1 * b.g(), fix=b.u(600, 2200))
    b.settle(b.u(350, 500))
    return b.score(label=msg("neuro.act.glance.name"), gaze=0.3)


def build_tilt(b: _B, ctx: Ctx) -> Score:
    s = _tilt_side(b, _side(b, ctx, 0.75))  # (fid:fix: 歪一下頭 seen 21/44 — often already tilted)
    b.to(b.u(350, 500), hp=b.u(0.2, 0.5), hr=s * b.u(0.6, 0.95), hy=s * b.u(0.3, 0.6))
    b.hold(b.u(1000, 2400))
    b.settle(b.u(400, 550))
    return b.score(label="歪頭看")


def build_dabble(b: _B, ctx: Ctx) -> Score:
    b.to(b.u(400, 550), np=b.u(0.6, 0.8), hp=b.u(-0.7, -0.5))
    for _ in range(1 + int(b.u(0, 2.5))):
        b.mouth_pulses(int(b.u(3, 6)), b.u(0.45, 0.7), b.u(70, 110), b.u(80, 130))
        b.rel(b.u(150, 250), dhy=b.u(0.2, 0.35) * b.sign())
    b.settle(b.u(450, 650))
    return b.score(label=msg("neuro.act.dabble.name"), alive=0.4)


def build_double_take(b: _B, ctx: Ctx) -> Score:
    s = _side(b, ctx, 0.5)
    b.to(b.u(250, 350), hy=s * b.u(0.5, 0.7), ease="snap").hold(b.u(250, 400), drift=0.0)
    b.to(b.u(250, 350), hy=0.0).hold(b.u(200, 450), drift=0.0)
    b.to(b.u(130, 190), hy=s * b.u(0.7, 0.95), hp=0.3, ease="snap", raw_ms=True)
    b.hold(b.u(800, 1500), drift=0.0)
    if b.p(0.4):
        b.vocal(b.tag(("inquire",), silent=0.5), 0.45)
    b.settle(450)
    return b.score(label=msg("neuro.act.double_take.name"), alive=0.3)


def build_proud_puff(b: _B, ctx: Ctx) -> Score:
    # (fid:fix: up from where the head IS; the brain asks for it only with room above)
    b.rel(b.u(450, 650), dnp=-b.u(0.4, 0.6), dhp=b.u(0.5, 0.7))
    b.hold(b.u(900, 1700))
    if b.p(0.3):
        hz, swing = _shake(b, ctx)
        b.osc("hr", swing * 0.6, hz, 800, decay=1.2)
    b.settle(500)
    return b.score(label=msg("neuro.act.proud_puff.name"))


def build_still(b: _B, ctx: Ctx) -> Score:
    # (fid:integrate 2026-09-28: caught walking, the legs need ~0.5 s and 2–3 steps to stop and a
    # stop reads as a stop only when held ≥ 0.8 s — fid:audit: a 0.5–1.5 s 定格 while walking was
    # seen as a stop 2/12 — so a walking 定格 lasts 1.6–2.4 s; standing it stays short)
    body = ctx.frame.body
    if body.posture == "walk" or abs(body.vx) > 0.2:
        b.wait(b.u(1600, 2400))
    else:
        b.wait(b.u(600, 1400) * b.dur)
    return b.score(label=msg("neuro.act.still.name"), alive=0.0, gaze=0.0)


def build_freeze(b: _B, ctx: Ctx) -> Score:
    # (fid:motor 2026-09-28: the head raised HIGH and held still — standing it was ~4° of head
    # pitch, unseen; now 12–17°: the head's validated look-up range, the neck too when seated)
    b.to(b.u(220, 320), hp=b.u(0.7, 0.95), np=b.u(-0.5, -0.25), ease="snap")
    b.hold(b.u(1300, 3000), drift=0.0)
    b.settle(b.u(450, 700))
    return b.score(label="盯著不動", alive=0.0, gaze=1.0)


def build_glance_away(b: _B, ctx: Ctx) -> Score:
    s = -(ctx.cmd.side if ctx.cmd.side in (-1, 1) else ctx.style.side)
    _look(b, s * b.u(0.8, 0.95), 0.1 * b.g(), fix=b.u(800, 2200))
    b.settle(450)
    return b.score(label=msg("neuro.act.glance_away.name"))


def _pecks(b: _B, n: int, *, vigour: float) -> None:
    """Pecks at this body's pace (pet:etho E1: fixate 0.15–0.5 s, strike 0.2–0.35 s with the
    neck forward and the head pitched down, the beak open at contact; 0.7–1.5 pecks a second —
    the head's bandwidth), each one a different depth and place."""
    # (fid:integrate 2026-09-28: each strike is mostly HEAD PITCH, 12–17° before the amplitude —
    # fid:audit: strikes of 2–5° on a neck-heavy pose were not seen, the pecks' head-pitch swings
    # 2 commanded → 1–1.5 performed — from a shallow head-down pose that leaves the strike room
    # inside the validated pitch range: _peck_base)
    for _ in range(n):
        b.hold(b.u(150, 500) / (0.8 + 0.4 * vigour), sigma=0.2, drift=0.0)  # fixate: still
        depth = b.u(0.6, 0.85)
        b.rel(b.u(200, 350) / (0.8 + 0.4 * vigour), dnp=0.1 * depth, dhp=-depth, ease="snap")
        b.to(b.u(60, 120), m=b.u(0.3, 0.6), raw_ms=True)
        b.rel(b.u(250, 400), dnp=-0.1 * depth, dhp=depth, dhy=0.15 * b.g())
        b.to(60, m=0.0, raw_ms=True)


def _peck_base(b: _B) -> tuple[float, float]:
    """The head-down pose pecks strike from (neck a little forward, head a little down)."""
    return b.u(0.2, 0.35), b.u(-0.4, -0.25)


def build_peck(b: _B, ctx: Ctx) -> Score:
    target = _target_yaw(ctx)
    ms = b.u(400, 600)
    np_, hp = _peck_base(b)
    b.to(ms, np=np_, hp=hp, hy=0.0 if target is None else target * 0.8)
    _pecks(b, int(b.u(2, 6 + ctx.cmd.intensity)), vigour=ctx.cmd.intensity)
    if b.p(0.4):
        _look(b, b.sign() * b.u(0.3, 0.6), 0.1, fix=500)
    b.settle(b.u(450, 650))
    return b.score(label=msg("neuro.act.peck.name"), gaze=0.3)


def build_forage(b: _B, ctx: Ctx) -> Score:
    if ctx.mode == "walk":
        _walk_head(b, ctx, low=0.4, look_p=0.2)  # eyes on the floor ahead
        return b.score(label=msg("neuro.act.forage.name"), gaze=0.3)
    ms = b.u(350, 500)
    np_, hp = _peck_base(b)
    b.to(ms, np=np_, hp=hp, hy=0.3 * b.g())
    _pecks(b, int(b.u(2, 5)), vigour=ctx.cmd.intensity)
    if b.p(0.25):
        b.vocal(b.tag(("", "chirp"), silent=2.0), 0.35)
    b.to(b.u(300, 450), np=0.2, hp=-0.3)
    return b.score(label="啄一啄", gaze=0.3)


def build_nibble(b: _B, ctx: Ctx) -> Score:
    target = _target_yaw(ctx) or 0.0
    reach = 0.35 + 0.3 * ctx.style.boldness
    # C1 (root 批 2026-10-01)：真低頭——俯角加深到 -0.8~-0.6，直渲特寫可見。
    # 配合 arrived 門（0.3m 內才咬），不再對空氣咬。
    b.to(b.u(400, 600), np=b.u(reach, reach + 0.2), hp=b.u(-0.8, -0.6), hy=target * 0.9)
    for _ in range(int(b.u(3, 6 + ctx.cmd.intensity))):
        b.to(b.ln(160, 0.3), m=b.u(0.35, 0.7), ease="snap")
        b.rel(b.ln(140, 0.3), dnp=b.u(0.05, 0.12))
        b.to(b.ln(150, 0.3), m=0.0)
        b.rel(b.ln(140, 0.3), dnp=-b.u(0.05, 0.12))
    if b.p(0.3):
        b.vocal(b.tag(("", "inquire"), silent=1.5), 0.4)
    b.to(b.u(350, 500), np=0.1, hp=0.0, hy=target * 0.5)
    return b.score(label=msg("neuro.act.nibble.name"), gaze=0.8)


def build_head_shake(b: _B, ctx: Ctx) -> Score:
    hz, swing = _shake(b, ctx)  # (fid:motor: slow and wide enough to be performed standing)
    cycles = b.u(2.0, 3.0)
    b.osc("hy", swing * b.u(1.0, 1.3), hz, cycles / (hz / b.dur) * 1000.0 / b.dur, decay=0.8)
    b.oscs.append(Osc(3, swing * b.u(0.5, 0.8) * b.amp, hz / b.dur, 0.0, b.t, math.pi / 3, 0.8))
    b.settle(300)
    return b.score(label=msg("neuro.act.head_shake.name"), alive=0.3)


def build_body_shake(b: _B, ctx: Ctx) -> Score:
    standing = ctx.frame.body.posture in ("stand", "walk")
    if standing and b.p(0.8):
        # a short shuffle: a few steps with the yaw rate flipping (the trunk shakes under it)
        flips = int(b.u(2, 4))
        for k in range(flips):
            b.cues.append(
                Cue(
                    b.t + k * b.u(260, 340),
                    "twist",
                    "",
                    b.u(0.35, 0.40),
                    (1 if k % 2 == 0 else -1) * b.u(0.7, 1.0),
                    320.0,
                )
            )
    # (fid:motor: the head's part of the shake at a rate the head can follow in this posture)
    hz, swing = _shake(b, ctx)
    b.osc("hr", swing * b.u(1.0, 1.3), hz, b.u(900, 1300), decay=0.8, advance=False)
    b.osc("hy", swing * b.u(0.4, 0.6), hz, b.u(900, 1300), decay=0.8, phase=math.pi / 2)
    b.hold(b.u(300, 500), drift=0.0)
    if b.p(0.3):
        b.osc("hy", swing * 0.8, hz, 700, decay=1.2)
    b.settle(400)
    return b.score(label=msg("neuro.act.body_shake.name"), alive=0.3)


# -- the tricks a person can teach (phase 2, neuro.tricks): the head's part; the legs' part is the
# body's (the bow and lie-low skills run from guard.BRAIN_SCRIPTS, the turn from the brain's own pivot) ---


def build_bow(b: _B, ctx: Ctx) -> Score:
    b.to(b.u(260, 360), np=b.u(0.4, 0.55), hp=b.u(-0.55, -0.4))
    b.hold(b.u(800, 1200), drift=0.02)
    b.vocal(b.tag(("greet", "chirp"), silent=0.5), 0.55)
    b.to(b.u(300, 420), np=0.0, hp=b.u(0.05, 0.2))
    b.settle(400)
    return b.score(label=msg("page.cmd.verb.bow"), alive=0.2, gaze=0.4)


def build_lie_low(b: _B, ctx: Ctx) -> Score:
    b.to(b.u(320, 420), np=b.u(0.5, 0.65), hp=b.u(-0.6, -0.45))
    b.hold(b.u(1400, 1900), drift=0.01)
    b.vocal(b.tag(("", "coo"), silent=0.6), 0.45)
    b.to(b.u(400, 520), np=0.0, hp=0.0)
    b.settle(400)
    return b.score(label=msg("page.motion.name.lie_low"), alive=0.1, gaze=0.3)


def build_spin(b: _B, ctx: Ctx) -> Score:
    # the turn is the body's (the brain's pivot); the head looks ahead of it and the beak calls
    b.to(b.u(240, 340), np=0.1, hp=b.u(0.1, 0.3), hy=ctx.style.side * b.u(0.3, 0.5))
    b.vocal(b.tag(("", "wheee"), silent=0.5), 0.55)
    b.hold(b.u(1800, 2400), drift=0.03)
    b.settle(400)
    return b.score(label=msg("page.cmd.verb.spin"), alive=0.5, gaze=0.2)


def build_flip(b: _B, ctx: Ctx) -> Score:
    # (the roll is the body's, official roulade: the brain does not ask for it yet — only the get-ready
    # crouch of the head exists, so a channel that is chosen has a program)
    b.to(b.u(300, 400), np=b.u(0.4, 0.5), hp=b.u(-0.5, -0.4))
    b.hold(b.u(700, 1000), drift=0.01)
    b.settle(400)
    return b.score(label=msg("page.cmd.verb.flip"), alive=0.1, gaze=0.2)


def build_investigate(b: _B, ctx: Ctx) -> Score:
    rng = ctx.cmd.target.range_m if ctx.cmd.target is not None else None
    near = rng is not None and rng < 0.6
    reach = (0.2 + 0.25 * ctx.style.boldness) * (1.5 if near else 1.0)
    if ctx.mode == "walk":
        b.to(b.u(300, 500), np=S.clamp(reach, 0.0, 0.5), hp=-0.25 if near else -0.05)
        b.hold(b.u(700, 1500))
        return b.score(label=msg("neuro.act.investigate.name"), gaze=1.0)
    b.to(b.u(350, 550), np=S.clamp(reach + 0.2, 0.0, 0.8), hp=b.u(-0.4, -0.1))  # neck stretched
    target = _target_yaw(ctx) or 0.0
    # looking with one eye and then the other (pet:etho A2: a new thing is viewed off-axis on
    # one side with a little roll, the sides swapped 2–6 times, 0.5–2 s each)
    eye = _side(b, ctx, 0.5)
    for _ in range(2 + int(b.u(0, 2.5 * (0.5 + ctx.style.exploration)))):
        b.to(
            b.u(250, 400),
            hy=S.clamp(target + eye * b.u(0.35, 0.55), -1.0, 1.0),
            hr=eye * b.u(0.3, 0.6),
        )
        b.hold(b.u(500, 1600))
        eye = -eye
    if b.p(0.3):
        b.vocal(b.tag(("", "inquire"), silent=1.0), 0.45)
    b.to(b.u(300, 450), np=0.1, hr=0.0, hy=target)
    return b.score(label="湊近看", gaze=0.6)


def build_follow(b: _B, ctx: Ctx) -> Score:
    if ctx.mode == "walk":
        b.to(b.u(300, 500), hp=b.u(0.2, 0.45), np=b.u(-0.1, 0.1))
        b.hold(b.u(800, 1600))
        return b.score(label=msg("neuro.act.follow.name"), gaze=1.0)
    b.to(b.u(250, 400), hp=b.u(0.4, 0.7), np=-0.2)
    if b.p(0.45):
        b.to(200, m=0.6).vocal(b.tag(("", "chirp", "greet"), silent=0.8), 0.55).to(250, m=0.0)
    b.hold(b.u(700, 1400))
    return b.score(label="抬頭看家人", gaze=1.0)


def build_greet(b: _B, ctx: Ctx) -> Score:
    b.to(b.u(300, 450), np=b.u(-0.5, -0.3), hp=b.u(0.5, 0.8))
    for _ in range(1 + int(b.u(0, 1.5 + ctx.style.sociability))):
        b.to(b.u(120, 200), m=b.u(0.8, 1.0), ease="snap")
        b.vocal(b.tag(("greet", "chirp"), silent=0.0), 0.6 + 0.3 * ctx.cmd.intensity)
        b.to(b.u(200, 300), m=0.0)
        if b.p(0.4):
            b.rel(250, dhp=-0.3).rel(250, dhp=0.3)
    b.hold(b.u(400, 800))
    b.settle(500)
    return b.score(label=msg("neuro.act.greet.name"), gaze=0.9)


def build_call_out(b: _B, ctx: Ctx) -> Score:
    s = _side(b, ctx, 0.5)
    for i in range(1 + int(b.u(0, 2))):
        s = -s if i else s
        b.to(b.u(350, 500), hp=b.u(0.5, 0.8), np=-0.2, hy=s * b.u(0.2, 0.6))
        b.to(150, m=0.8, ease="snap").vocal(b.tag(("chirp", "greet", "inquire"), silent=0.0), 0.7)
        b.to(250, m=0.0)
        b.mark("聽")
        b.to(350, hr=s * 0.5).hold(b.u(1400, 2800))
        b.to(300, hr=0.0)
    b.settle(500)
    return b.score(label=msg("neuro.act.call_out.name"), alive=0.5)


def build_watch(b: _B, ctx: Ctx) -> Score:
    target = _target_yaw(ctx) or 0.0
    b.to(b.u(300, 450), np=0.1, hp=b.u(0.0, 0.2), hy=target)
    b.hold(b.u(900, 2000), drift=0.02)
    if b.p(0.3 + 0.4 * ctx.style.w("explorer")):  # an eye at a time (pet:etho A2)
        eye = _side(b, ctx, 0.6)
        b.to(300, hr=eye * b.u(0.3, 0.6), hy=S.clamp(target + eye * b.u(0.3, 0.5), -1.0, 1.0))
        b.hold(b.u(500, 1200)).to(300, hr=0.0, hy=target)
    return b.score(label=msg("neuro.act.watch.name"), gaze=0.7, alive=0.5)


def build_retreat(b: _B, ctx: Ctx) -> Score:
    threat = ctx.cmd.side if ctx.cmd.side in (-1, 1) else ctx.style.side
    b.to(b.u(300, 450), np=0.2, hp=-0.1, hy=-threat * 0.2)
    b.hold(b.u(500, 1000))
    if b.p(0.6):  # a look back over the shoulder at what it is getting away from
        b.to(b.u(250, 350), hy=threat * b.u(0.6, 0.85), ease="snap").hold(b.u(400, 700))
        b.to(300, hy=0.0)
    return b.score(label=msg("neuro.act.retreat.name"), gaze=0.2)


def build_flee(b: _B, ctx: Ctx) -> Score:
    if ctx.unit == 0 and b.p(0.6):
        b.vocal("alarm", 0.7)
    b.to(b.u(200, 300), np=0.3, hp=0.1, hy=0.0, ease="snap")
    b.hold(b.u(800, 1500), drift=0.01)
    body = ctx.frame.body
    if not (ctx.cmd.vx > 0.0 or body.posture == "walk" or abs(body.vx) > 0.2):
        # (fid:integrate: stopped, it looks back over its shoulder at what it got away from —
        # fid:audit: a flee's pause held the head still, 1.5° of motion, unseen)
        threat = ctx.cmd.side if ctx.cmd.side in (-1, 1) else ctx.style.side
        b.to(b.u(250, 350), hy=-threat * b.u(0.6, 0.85), hp=0.2, ease="snap").hold(b.u(400, 800))
        b.to(b.u(300, 400), hy=0.0, hp=0.1)
    return b.score(label=msg("neuro.act.flee.name"), alive=0.3, gaze=0.1)


def build_doze(b: _B, ctx: Ctx) -> Score:
    b.posture("sit")
    depth = min(0.9, 0.35 + 0.15 * ctx.unit + 0.1 * b.g())
    b.to(b.u(2500, 5000), np=depth, hp=-0.8 * depth)
    b.hold(b.u(600, 1800), drift=0.01)
    if b.p(0.75):  # "點一下又抬起來": the head drops and is caught
        b.to(200, np=min(1.0, depth + 0.15), ease="snap", raw_ms=True)
        b.to(260, np=0.1, hp=-0.1, ease="snap", raw_ms=True)
        b.hold(b.u(800, 1600))
    b.breathe(0.025, 0.25)
    return b.score(label=msg("neuro.act.doze.name"), alive=0.25, gaze=0.0)


def build_rest_sit(b: _B, ctx: Ctx) -> Score:
    b.posture("sit")
    b.to(b.u(900, 1500), np=b.u(0.2, 0.35), hp=b.u(-0.25, -0.1), hy=0.1 * b.g())
    b.hold(b.u(3000, 7000), drift=0.02)
    if b.p(0.5):
        _look(b, b.sign() * b.u(0.3, 0.6), 0.0, fix=b.u(700, 1500), np=0.25)
    if b.p(0.15):
        b.vocal(b.tag(("", "coo"), silent=1.0), 0.3)
    return b.score(label=msg("neuro.act.rest_sit.name"), alive=0.6, gaze=0.4)


def build_return_dock(b: _B, ctx: Ctx) -> Score:
    nest_seen = any(s.kind == "nest" for s in ctx.frame.camera.sightings)
    if nest_seen:
        b.to(b.u(300, 450), hp=b.u(-0.15, 0.05), np=0.1).hold(b.u(800, 1500))
        return b.score(label=msg("neuro.why.nest"), gaze=1.0)
    if ctx.mode == "walk":
        s = b.sign()
        b.to(b.u(600, 900), hy=s * b.u(0.4, 0.7), hp=0.0).hold(b.u(300, 600))
        b.to(b.u(700, 1000), hy=-s * b.u(0.3, 0.6)).hold(b.u(300, 600))
        return b.score(label="找窩", gaze=0.8)
    _look(b, b.sign() * b.u(0.5, 0.9), 0.05, fix=b.u(500, 900))
    _look(b, b.sign() * b.u(0.5, 0.9), 0.05, fix=b.u(500, 900))
    if b.p(0.2):
        b.vocal(b.tag(("", "coo"), silent=1.0), 0.3)
    return b.score(label="找窩", gaze=0.8)


def build_zoomies(b: _B, ctx: Ctx) -> Score:
    """A straight dash at its fastest walk (0.55) and then a tight circle (an arc: ≤ 0.40),
    repeated. (fid:integrate 2026-09-28: fid:audit measured the old 1.2–2 s dash at 1.1 × its
    ordinary walk and the 1–1.8 s circle at 12° of heading — the body's speed and turn rate lag
    the command (~0.4 s) and it turns ~0.4 × the commanded rate — so the dash is 2–3 s and the
    circle 2.4–3.4 s, long enough to be seen; nothing faster than the owner's 0.55, so its words
    no longer say it is fast. fid:fix: the review measured its "circle" at −74…+20° of heading
    (n = 12, never a circle): 「興奮地走走轉轉」.)"""
    if ctx.unit == 0:
        b.to(200, np=-0.1, hp=0.35, m=0.7, ease="snap").vocal("wheee", 0.7).to(300, m=0.0)
    s = _side(b, ctx, 0.5)
    dash = b.u(2000, 3000)
    b.twist(0.55, 0.12 * b.g(), dash)
    b.to(b.u(300, 450), hy=0.0, hp=0.3, np=-0.1)
    b.wait(max(200.0, dash - 450.0))
    arc = b.u(2400, 3400)
    b.twist(0.40, s * b.u(0.8, 1.0), arc)
    b.to(b.u(250, 400), hy=s * 0.5, hp=0.25)
    b.wait(max(200.0, arc - 400.0))
    if b.p(0.35):
        b.vocal(b.tag(("wheee", "chirp"), silent=0.5), 0.6)
    return b.score(label=msg("neuro.act.zoomies.name"), gaze=0.1, alive=0.4)


def build_startle(b: _B, ctx: Ctx) -> Score:
    # The snap itself is the reflex overlay (cmd.head); this is what follows it: an alert,
    # raised head that stays on the stimulus, sometimes a shake-off, then it lets go.
    # (fid:motor: raised as high as the freeze's alert pose — it was ~8° standing)
    # (fid:integrate: seated the neck pulls back ~0.5 of its command — fid:audit: 5.5° asked →
    # 1.9°, then 9° → 3–6° — so seated the neck is asked for its whole validated range whatever
    # the amplitude (the motor's envelope clamps it: −14°); standing the head pitch does it,
    # motor.NECK_TO_PITCH)
    seated = ctx.frame.body.posture not in ("stand", "walk")
    neck = b.u(-1.3, -1.1) / max(0.2, b.amp) if seated else b.u(-0.5, -0.3)
    b.to(b.u(200, 300), np=neck, hp=b.u(0.6, 0.85), ease="snap")
    b.hold(b.u(900, 2200), drift=0.0)
    if b.p(0.35):
        hz, swing = _shake(b, ctx)
        b.osc("hy", swing * 0.9, hz, 900, decay=1.0)
    b.settle(b.u(600, 900))
    return b.score(label=msg("neuro.act.startle.name"), alive=0.0, gaze=1.0)


def build_withdraw(b: _B, ctx: Ctx) -> Score:
    site = ctx.cmd.target.site if ctx.cmd.target is not None else None
    s = ctx.cmd.side if ctx.cmd.side in (-1, 1) else ctx.style.side
    b.hold(b.u(500, 900), drift=0.0)
    if b.p(0.6):  # peeks at where it was touched
        yaw = 0.9 * s if site in ("left", "right", "tail", "back") else 0.3 * s
        b.to(b.u(300, 450), hy=yaw, hp=0.1).hold(b.u(500, 900))
    b.settle(b.u(500, 800))
    return b.score(label=msg("neuro.act.withdraw.name"), alive=0.3, gaze=0.3)


def build_righting(b: _B, ctx: Ctx) -> Score:
    if ctx.mode == "down":  # fallen: the head stays still at HOME; velstand gets it up
        b.to(300, np=0.0, hp=0.0, hy=0.0, hr=0.0, m=0.0).wait(500)
        return b.score(label=msg("neuro.body.event.fell"), alive=0.0, gaze=0.0)
    if b.p(0.75):  # up again: a shake, then a look at its feet (spec §8.9 E16)
        hz, swing = _shake(b, ctx)
        b.osc("hy", swing * b.u(0.9, 1.2), hz, b.u(900, 1300), decay=0.8)
    b.to(b.u(400, 550), np=b.u(0.5, 0.7), hp=b.u(-0.8, -0.6)).hold(b.u(600, 1100))
    b.settle(500)
    return b.score(label="爬起來了", alive=0.5)


def build_held(b: _B, ctx: Ctx) -> Score:
    s = b.sign()
    b.to(b.u(1500, 2500), hy=s * b.u(0.3, 0.55), hp=b.u(0.1, 0.3), np=0.0)
    b.hold(b.u(600, 1400))
    b.to(b.u(1800, 2800), hy=-s * b.u(0.3, 0.55))
    b.hold(b.u(600, 1400))
    if b.p(0.2):
        b.vocal(b.tag(("", "inquire"), silent=1.0), 0.35)
    return b.score(label=msg("neuro.act.held.name"), alive=0.3, gaze=0.3)


BUILDERS: dict[str, Builder] = {
    "look_around": build_look_around,
    "perk_tilt": build_perk_tilt,
    "stretch_yawn": build_stretch_yawn,
    "preen": build_preen,
    "happy_sway": build_happy_sway,
    "tuck_away": build_tuck_away,
    "nap": build_nap,
    "wake_ritual": build_wake_ritual,
    "nod": build_nod,
    "peekaboo": build_peekaboo,
    "call_response": build_call_response,
    "snuggle": build_snuggle,
    "little_dance": build_little_dance,
    "not_now": build_not_now,
    "little_song": build_little_song,
    "self_duet": build_self_duet,
    "survey": build_survey,
    "peekaboo_master": build_peekaboo_master,
    "melt": build_melt,
    "nuzzle_tilt": build_nuzzle_tilt,
    "moon_gaze": build_moon_gaze,
    "sleep_talk": build_sleep_talk,
    "stand_up": build_stand_up,
    "sit_down": build_sit_down,
    "wander": build_wander,
    "turn_around": build_turn_around,
    "peek_down": build_peek_down,
    "glance": build_glance,
    "tilt": build_tilt,
    "dabble": build_dabble,
    "double_take": build_double_take,
    "proud_puff": build_proud_puff,
    "still": build_still,
    "freeze": build_freeze,
    "glance_away": build_glance_away,
    "peck": build_peck,
    "forage": build_forage,
    "nibble": build_nibble,
    "head_shake": build_head_shake,
    "body_shake": build_body_shake,
    "bow": build_bow,
    "lie_low": build_lie_low,
    "spin": build_spin,
    "flip": build_flip,
    "investigate": build_investigate,
    "follow": build_follow,
    "greet": build_greet,
    "call_out": build_call_out,
    "watch": build_watch,
    "retreat": build_retreat,
    "flee": build_flee,
    "doze": build_doze,
    "rest_sit": build_rest_sit,
    "return_dock": build_return_dock,
    "zoomies": build_zoomies,
    "startle": build_startle,
    "withdraw": build_withdraw,
    "righting": build_righting,
    "held": build_held,
}

# Programs whose unit depends on the situation (rebuilt when the situation changes).
REACTIVE = frozenset(
    {
        "wander",
        "turn_around",
        "forage",
        "investigate",
        "follow",
        "return_dock",
        "call_response",
        "righting",
    }
)
# Programs the motor may shape the walk for (their twist cues are honoured while standing).
SHAPES_WALK = frozenset({"zoomies", "body_shake"})
MIN_UNIT_MS = 300.0  # a reactive unit runs at least this long before it may switch (hysteresis)


def mode_of(behaviour: str, cmd: MotorCommand, frame: SensorFrame, previous: str) -> str:
    """The situation a reactive program adapts to (from the command and the body's own sense)."""
    if behaviour == "righting":
        down = frame.body.fallen or "fell" in frame.events
        return "down" if down else "up"
    if behaviour == "call_response":
        heard = frame.sound.onset > 0.25 and frame.sound.kind in ("voice", "name", "hello", "clap")
        if previous == "listen" and heard:
            return "reply"
        return "listen"
    moving = cmd.vx > 0.0 or frame.body.posture == "walk" or abs(frame.body.vx) > 0.2
    return "walk" if moving else "stop"


# ── the program state machine ─────────────────────────────────────────────────

Pose5 = tuple[float, float, float, float, float]
REST: Pose5 = (0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class ProgramState:
    behaviour: str
    bout: int
    variant: int
    unit: int = 0
    unit_t0: int = 0
    score: Score = field(default_factory=lambda: Score((Key(0.0, REST),)))
    rng: int = 0
    cue_i: int = 0
    last: Pose5 = REST
    mode: str = ""
    label: str = ""
    twist: tuple[float, float, int] | None = None  # the program's own (vx, vyaw, until_ms)


@dataclass(frozen=True, slots=True)
class ProgramOut:
    pose: HeadPose  # clip units, absolute side
    gaze: float  # how much the brain's (non-reflex) gaze may take over
    alive: float  # × the idle micro-motion
    posture: str | None  # a posture the program asks for now (sit/stand) or None
    vocals: tuple[tuple[str, float], ...]  # (tag, level) to emit now
    twist: tuple[float, float] | None  # the program's own walk shaping (vx, vyaw) or None
    label: str  # what it is doing (plain Chinese, for the page)
    unit: int
    unit_done: bool


def _amp(style: Style, cmd: MotorCommand) -> float:
    """The command's amplitude (a fraction of the A100-validated seated envelope). The brain
    already folds the stage's amplitude cap in (``brain.py``: amplitude ≤ ``STAGES`` cap); the
    stage cap stays a ceiling here, so a command without it can never exceed its stage. As the
    bout's motivation is used up (``phase`` → 1, satiation) its units grow smaller: a bout
    winds down instead of stopping dead."""
    winding = 1.0 - 0.25 * S.clamp(cmd.phase, 0.0, 1.0)
    return min(S.clamp(cmd.amplitude, 0.05, 1.0), style.amp_cap) * winding


def _build(
    behaviour: str,
    unit: int,
    variant: int,
    start: Pose5,
    style: Style,
    cmd: MotorCommand,
    frame: SensorFrame,
    mode: str,
) -> tuple[Score, int]:
    rng = S.seed_from("neuro.program", behaviour, variant, unit)
    builder = BUILDERS.get(behaviour, build_look_around)
    slowing = 1.0 - 0.15 * S.clamp(cmd.phase, 0.0, 1.0)  # … and a little slower
    b = _B(
        rng,
        start,
        amp=_amp(style, cmd),
        dur=1.0 / (S.clamp(cmd.tempo, 0.5, 1.6) * slowing),
        style=style,
        cmd=cmd,
    )
    score = builder(b, Ctx(style, cmd, frame, mode, unit))
    return _visible(score, behaviour), b.rng


# ── every gesture big enough to be SEEN (fid:integrate 2026-09-28) ─────────────
# The owner judges only what the body visibly does. fid:audit's measure: a head gesture is seen
# when the beak moves 1.2 cm or the head turns 10° (V ≥ 1), and the body performs 0.6–0.9 of a
# small command (the gearboxes, the standing balance). With the brain's amplitude (a calm duck,
# a hatchling's stage cap 0.6, a bout winding down) 34 act cells were commanded at V 0.6–1.25 and
# came out weak. So a unit's gesture is commanded at ≥ VISIBLE_DEG (16°: after the gain ≥ 10°,
# the 3D view's 17 px on a desktop): a smaller one grows (about its start pose, the whole unit
# alike, ×≤ FLOOR_GAIN_MAX); a bigger one is left as it is (a bold adult's glance stays wider
# than a hatchling's). A unit that moves under HOLD_DEG is a hold (a
# fixation, a rest), not a gesture: left alone. The resting programs' small moves (「頭低一點」)
# only need to be seen as small: QUIET_DEG.
CLIP_RAD = (0.25, 0.35, 0.6, 0.25)  # rad per clip unit: content.SEATED_ENVELOPE (= motor.ENVELOPE)
VISIBLE_DEG = 16.0
QUIET_DEG = 8.0
HOLD_DEG = 3.0
FLOOR_GAIN_MAX = 4.0
QUIET_PROGRAMS = frozenset({"rest_sit", "melt", "sleep_talk"})
NO_FLOOR = frozenset({"still"})
# (fid:fix 2026-09-28) a moment is NAMED on the page (「小動作：低頭看地上」), and shown only once
# the head was seen to do it: its gesture is commanded at ≥ MOMENT_DEG (a look down standing makes
# ~0.65 of its command — pet:body — so 16° came out at 10–11°: 低頭看地上 seen 8/14 headless)
MOMENT_DEG = 22.0
MOMENT_PROGRAMS = frozenset(
    {
        "glance",
        "tilt",
        "perk_tilt",
        "double_take",
        "peek_down",
        "head_shake",
        "glance_away",
        "proud_puff",
    }
)


def gesture_deg(score: Score, step_ms: float = 20.0) -> float:
    """How far a unit moves the head from where it starts: the largest head rotation, over the
    unit, of its pose (keys and oscillations together) from its first pose — the nose's pitch
    (neck and head pitch together), yaw and roll, small angles — in degrees."""
    ref = score.keys[0].pose
    best = 0.0
    steps = max(1, int(score.length / step_ms))
    for n in range(steps + 1):
        pose = score.pose_at(n * step_ms)
        d = [pose[i] - ref[i] for i in range(4)]
        pitch = CLIP_RAD[1] * d[1] - CLIP_RAD[0] * d[0]
        yaw = CLIP_RAD[2] * d[2]
        roll = CLIP_RAD[3] * d[3]
        best = max(best, pitch * pitch + yaw * yaw + roll * roll)
    return math.degrees(math.sqrt(best))


def _visible(score: Score, behaviour: str) -> Score:
    """The unit, its gesture grown to at least VISIBLE_DEG (see above)."""
    if behaviour in NO_FLOOR or len(score.keys) < 2:
        return score
    size = gesture_deg(score)
    want = (
        QUIET_DEG
        if behaviour in QUIET_PROGRAMS
        else MOMENT_DEG
        if behaviour in MOMENT_PROGRAMS
        else VISIBLE_DEG
    )
    if size < HOLD_DEG or size >= want:
        return score
    k = min(FLOOR_GAIN_MAX, want / size)
    ref = score.keys[0].pose
    # … never past the clip envelope (±1: the motor's validated range; a pose already beyond it
    # is not grown further)
    for key in score.keys[1:]:
        for i in range(4):
            d = key.pose[i] - ref[i]
            if abs(d) > 1e-9:
                room = ((1.0 if d > 0 else -1.0) - ref[i]) / d
                k = min(k, max(1.0, room))
    if k <= 1.0:
        return score
    keys = tuple(
        Key(
            key.t,
            (
                ref[0] + k * (key.pose[0] - ref[0]),
                ref[1] + k * (key.pose[1] - ref[1]),
                ref[2] + k * (key.pose[2] - ref[2]),
                ref[3] + k * (key.pose[3] - ref[3]),
                key.pose[4],
            ),
            key.ease,
        )
        for key in score.keys
    )
    oscs = tuple(replace(o, amp=o.amp * k) if o.joint < 4 else o for o in score.oscs)
    return replace(score, keys=keys, oscs=oscs)


def start(
    cmd: MotorCommand, style: Style, frame: SensorFrame, t_ms: int, last: Pose5 = REST
) -> ProgramState:
    """A new bout of ``cmd.behaviour`` (unknown behaviours play a look around)."""
    behaviour = cmd.behaviour if cmd.behaviour in BUILDERS else "look_around"
    mode = mode_of(behaviour, cmd, frame, "") if behaviour in REACTIVE else ""
    score, rng = _build(behaviour, 0, cmd.variant, last, style, cmd, frame, mode)
    return ProgramState(
        behaviour=behaviour,
        bout=cmd.bout,
        variant=cmd.variant,
        unit=0,
        unit_t0=t_ms,
        score=score,
        rng=rng,
        last=last,
        mode=mode,
        label=score.label,
    )


def step(
    state: ProgramState | None, cmd: MotorCommand, style: Style, frame: SensorFrame, t_ms: int
) -> tuple[ProgramState, ProgramOut]:
    """Advance the program to ``t_ms`` (starting a new one when the bout changed)."""
    if state is None or state.bout != cmd.bout or state.behaviour != cmd.behaviour:
        state = start(cmd, style, frame, t_ms, state.last if state is not None else REST)
    unit_done = False
    elapsed = float(t_ms - state.unit_t0)
    if state.behaviour in REACTIVE:
        mode = mode_of(state.behaviour, cmd, frame, state.mode)
        if mode != state.mode and elapsed >= MIN_UNIT_MS:
            score, rng = _build(
                state.behaviour, state.unit + 1, state.variant, state.last, style, cmd, frame, mode
            )
            state = replace(
                state,
                unit=state.unit + 1,
                unit_t0=t_ms,
                score=score,
                rng=rng,
                cue_i=0,
                mode=mode,
                label=score.label,
            )
            elapsed = 0.0
            unit_done = True
    if elapsed >= state.score.length:
        mode = (
            mode_of(state.behaviour, cmd, frame, state.mode) if state.behaviour in REACTIVE else ""
        )
        score, rng = _build(
            state.behaviour, state.unit + 1, state.variant, state.last, style, cmd, frame, mode
        )
        state = replace(
            state,
            unit=state.unit + 1,
            unit_t0=t_ms,
            score=score,
            rng=rng,
            cue_i=0,
            mode=mode,
            label=score.label,
        )
        elapsed = 0.0
        unit_done = True
    score = state.score
    pose = score.pose_at(elapsed)
    vocals: list[tuple[str, float]] = []
    posture: str | None = None
    twist = state.twist
    label = state.label
    cue_i = state.cue_i
    while cue_i < len(score.cues) and score.cues[cue_i].t <= elapsed:
        cue = score.cues[cue_i]
        if cue.kind == "vocal":
            vocals.append((cue.tag, cue.a))
        elif cue.kind == "posture":
            posture = cue.tag
        elif cue.kind == "twist":
            twist = (cue.a, cue.b, t_ms + int(cue.c))
        elif cue.kind == "mark":
            label = cue.tag
        cue_i += 1
    spec = BEHAVIOURS.get(state.behaviour)
    if posture is None and spec is not None and spec.posture in ("sit", "stand"):
        posture = spec.posture
    out_twist: tuple[float, float] | None = None
    if twist is not None and t_ms < twist[2]:
        out_twist = (twist[0], twist[1])
    elif twist is not None:
        twist = None
    state = replace(state, cue_i=cue_i, last=pose, label=label, twist=twist)
    return state, ProgramOut(
        pose=HeadPose(np=pose[0], hp=pose[1], hy=pose[2], hr=pose[3], mouth=pose[4]),
        gaze=score.gaze,
        alive=score.alive,
        posture=posture,
        vocals=tuple(vocals),
        twist=out_twist,
        label=label or state.behaviour,
        unit=state.unit,
        unit_done=unit_done,
    )


__all__ = [
    "BUILDERS",
    "Cue",
    "Key",
    "Osc",
    "ProgramOut",
    "ProgramState",
    "REACTIVE",
    "REPERTOIRE_VERSION",
    "SHAPES_WALK",
    "Score",
    "Style",
    "mode_of",
    "start",
    "step",
    "style_from",
    "style_from_genome",
]
