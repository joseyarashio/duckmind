"""跟著媽媽走 — a young duck imprints on one family member and follows them (a:social, 2026-09-28).

The owner's A list, behaviour (1): *a young duck imprints on one family member and follows them,
settles at their feet when they stop*; it must EMERGE from the brain (drives, senses, learning,
temperament) and be decided only from the duck's own senses.

**Who is who — from its own senses.** The duck never gets a world id. Each person sighting may
carry a *look*: the camera's person re-identification signature (:data:`LOOK_DIMS`: the colour of
the legs — what a 20-cm-tall duck sees of a person —, how long the legs look at the range the eye
measures, and from further off the colour of the shirt; MODEL: there is no validated detector,
spec §8.8, :mod:`duckmind.world.neuro_social_senses`), and speech may carry a *voice* (a speaker
signature from the mono microphone; MODEL). :class:`Individuals` clusters the looks it gets into
the people it knows (its own ids 1, 2, 3 …), keeping an eye track on the same person while it
stays tracked; :class:`Voices` does the same for voices and learns which voice goes with which
person from hearing one while seeing the other close by. A dim room makes the looks noisier and
the duck mixes people up more — nothing tells it the truth.

**Imprinting** (Lorenz 1935; Hess 1959: a sensitive period in the first days after hatching, the
bond strongest in the youngest). Care it can feel and put on someone: a gentle touch with that
person right there (the hand's owner is the one it saw close a moment ago), being carried and set
down, being talked to (their voice), a charged battery swapped in, and time spent close to them.
Each bout of care grows that person's bond by the duck's plasticity — its growth stage
(:data:`STAGE_PLASTICITY`: a hatchling bonds within a few pats, an adult hardly in an hour) × its
sociability (sociable ducks bond fastest and strongest) — and bonding is winner-take-most: once one
bond is strong, care from the others grows theirs only a little (:data:`EXCLUSIVE`). Its mother
(媽媽, :attr:`Imprint.mom`) is the person whose bond is at least :data:`MOM_MIN` and clearly ahead.

**The need to stay near** (:meth:`Imprint.features`, for the competition in
:mod:`duckmind.neuro.select`): the further its mother is beyond its own comfort radius (timid and
young ducks keep a short leash, bold ones a long one; once she has stopped the leash shortens to
its stop distance — it comes to her feet), the stronger ``mom_far``; she walks off:
``mom_moving``; out of sight: ``mom_gone`` (it goes looking and peeps); afraid: ``mom_fear`` (a
frightened duckling runs to its mother); she has stopped and it is close: ``mom_near_still`` (it
settles at her feet). :meth:`Imprint.goal` gives where she is — seen, or believed where she was
last seen, carried by its own odometry (it drifts) for :data:`BELIEF_S`.

Pure and clock-injected (``t_s`` / odometry passed in); deterministic; standard library only.
Phase 5 (fixed point): looks in ppm, distances in mm, bonds in ppm; nothing needs more than
+, ×, a square root and atan2.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

IMPRINT_VERSION = "neuro-imprint-2"

# ── looks and voices ──────────────────────────────────────────────────────────

# The look's components (MODEL: the camera's person re-identification signature). NaN = not seen
# this time (the shirt shows only from further off; the legs' length needs a range).
LOOK_DIMS = ("legs_r", "legs_g", "legs_luma", "legs_m", "shirt_r", "shirt_g")
# How far apart two looks may be, per component, before they are one standard step apart: about
# twice the eye's own scatter in good light. (a:social measured the model's looks, n = 1,500 per
# condition: the same person's look lies 0.53 / 0.78 from its mean in bright / dim light (median;
# 95 % within 1.0 / 1.5), the nearest other person's 1.2–1.3 up close — where only the legs show,
# the adults' dark trousers overlap: 5 % of looks lie within 0.6 of another adult — and 2.4–2.6
# from 1.6 m, where the shirt shows. So up close it can mix the adults up, as a real eye would.)
LOOK_SCALE = (0.025, 0.025, 0.06, 0.1, 0.03, 0.03)
JOIN_D = 1.6  # a look within this (scaled RMS distance) of someone it knows is them …
MERGE_D = 0.7  # … and two clusters that came this close are one person
TRACK_SWITCH = 3  # looks in a row that fit another person better move an eye track over
MEAN_RATE_MIN = 1.0 / 40.0  # a known person's look keeps drifting in slowly (clothes, light)
MAX_PEOPLE = 8
VOICE_SCALE = (0.06, 0.06)  # the speaker signature's two components
VOICE_JOIN_D = 1.8
MAX_VOICES = 8
VOICE_SEEN_S = 2.5  # a voice heard while exactly one person was seen this recently …
VOICE_NEAR_M = 2.5  # … this close: the voice may be theirs (co-occurrence)

# ── care and bonds ────────────────────────────────────────────────────────────

# The sensitive period: how much a bout of care bonds, by growth stage (Duck Core's stages; Hess
# 1959: imprinting peaks in the first day and is nearly closed after a few — a juvenile or an
# adult grows fond of people slowly, it does not imprint).
STAGE_PLASTICITY = {"hatchling": 1.0, "duckling": 0.55, "juvenile": 0.2, "adult": 0.07}
SOCIAL_PLASTICITY = (0.4, 1.2)  # × (a + b × sociability): sociable ducks bond fastest
EXCLUSIVE = 0.95  # once a bond is strong, care from others grows theirs only (1 − 0.95 × it)
MOM_MIN = 0.35  # a bond at least this strong …
MOM_MARGIN = 0.1  # … and this much ahead of the next makes that person its mother
# Care, per unit (a gentle touch per second of touching; the others per event).
CARE_TOUCH_PER_S = 0.12
CARE_SET_DOWN = 0.25
CARE_BATTERY = 0.2
CARE_SPEECH = {"voice": 0.04, "name": 0.08, "hello": 0.06}
CARE_NEAR_PER_S = 0.004  # time spent close (≤ CARE_NEAR_M)
CARE_NEAR_M = 1.0
SHARP_TOUCH = -0.15  # a poke by someone: a little less fond of them
# Whose hand, whose voice: the person it believes is right beside it now — seen within
# HAND_OWNER_S, believed (where it last saw them, carried by its own odometry) within
# HAND_OWNER_M of where it is now (object permanence: she came over and is still there, even
# when its narrow camera looks the other way); the one seen last when two are there.
HAND_OWNER_S = 60.0
HAND_OWNER_M = 1.6
VOICE_OWNER_M = 2.5  # a voice with no known owner: the one person it believes this close
SET_DOWN_OWNER_S = 3.0  # set down: the one believed close within this after it

# ── following ─────────────────────────────────────────────────────────────────

PASS_KEEP_S = 1.0  # a person seen on the last camera pass stays "in view" this long
BELIEF_S = 120.0  # it believes its mother is where it last saw her for this long …
PREDICT_S = 3.0  # … carried on along her walk for up to this long if she was walking away
ABSENT_S = 3.0  # … unless it looks right there this long and she is not there (object permanence;
# only while it stands still enough to see a still person: a stepping duck's camera smears)
LOOKING_RAD = 0.3  # its camera looks that way: the head's yaw within this of her believed bearing
LOOK_RANGE_M = 4.0  # … and she would be close enough to be seen
NEAR_ABSENT_M = 0.9  # standing where it believed she was, facing about that way …
NEAR_LOOK_RAD = 0.7  # … (the camera's half width and a little): she is not here
SEARCH_FROM_M = 1.0  # lost: it goes to where it last saw her, until it is this close
GONE_AFTER_S = 2.5  # out of sight this long: it misses her (mom_gone rises over GONE_RISE_S)
GONE_RISE_S = 10.0
GONE_GIVE_UP_S = 150.0  # … and after this long the urge to look for her fades to a third
MOVING_TAU_S = 1.0
# her believed position moved MOVED_M + MOVED_PER_M × range within MOVED_WINDOW_S: she is walking
# (more than the eye's range and bearing scatter make of a person standing still — the range
# scatters ~8 % (neuro_senses.RANGE_SIGMA): a:social's physics runs, 0.4 m flat made a mother
# standing 1.6 m off "walk" every few seconds, and the duck never settled)
MOVED_M = 0.3
MOVED_PER_M = 0.25
MOVED_WINDOW_S = 2.0
STILL_FOR_S = 2.5  # she has stood still this long: she has stopped
NEAR_SEEN_S = 6.0  # … and it saw her there this recently (it glances at her every few seconds)
FAR_RISE_M = 0.9  # mom_far grows from 0 at the comfort radius to 1 this much beyond it
# She has stopped: the brood gathers round her — it comes on to its stop distance (+ this), the
# pull full FAR_RISE_STOPPED_M beyond (a:social's physics runs: with the comfort radius and the
# long rise it stalled 1.5–1.8 m from her, the pull too weak to win there, and never settled)
LEASH_STOPPED_M = 0.15
FAR_RISE_STOPPED_M = 0.45
DEFAULT_RANGE_M = 1.3  # a sighting without a range: about this far (a person fills the frame)
# the comfort radius (m, to her feet): a + b × boldness × (1 − c × young) + d × (1 − bond) — the
# shy and the young stay close (ethology: ducklings keep within ~1 m of the hen), the bold roam
REACH_OK = (0.85, 0.8, 0.4, 0.3)
# where it stops when it has caught up (m, to her feet — what its eye's ground-plane range reads):
# a + b × boldness. Not closer: the family's bodies keep ~0.86 m between their hips and a duck
# (world.people PLAN_KEEP_M) and do not walk off while a duck stands inside that — a:social's
# physics runs: stopped 0.64 m from her hips, the carer could not leave (waited 80 s).
REACH_STOP = (0.8, 0.2)
SETTLE_EXTRA_M = 0.3  # it settles at her feet within its leash (Imprint.leash_m) + this


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def look_distance(a: Sequence[float], b: Sequence[float]) -> float | None:
    """The scaled RMS distance between two looks over the components both have (None: nothing
    in common — e.g. one saw only the shirt, the other only the legs)."""
    total = 0.0
    n = 0
    for i, scale in enumerate(LOOK_SCALE):
        if i >= len(a) or i >= len(b):
            break
        x, y = a[i], b[i]
        if math.isnan(x) or math.isnan(y):
            continue
        total += ((x - y) / scale) ** 2
        n += 1
    if n == 0:
        return None
    # the legs' colour must be among them: the shirt or the length alone cannot tell people apart
    if math.isnan(a[0]) or math.isnan(b[0]):
        return None
    return math.sqrt(total / n)


@dataclass(frozen=True, slots=True)
class PersonSeen:
    """One person sighting as the brain gets it (the contract's ``Sighting`` of kind "person",
    plus its look when the eye gave one)."""

    track: int
    bearing: float  # rad from the trunk, + = left
    range_m: float | None
    motion: float = 0.0
    look: tuple[float, ...] | None = None


@dataclass(slots=True)
class Person:
    """Someone it knows: its own id, what they look like (a running mean), when and where it
    last saw them (its own odometry frame)."""

    pid: int
    mean: list[float]
    counts: list[int]
    seen_n: int = 0
    first_s: float = 0.0
    seen_s: float = -1e9
    x: float = 0.0
    y: float = 0.0
    range_m: float = DEFAULT_RANGE_M
    bearing: float = 0.0
    motion: float = 0.0  # its image motion, a leaky mean
    moved_s: float = -1e9  # the last time it was seen moving (or its place moved)
    hist: list[tuple[float, float, float]] = field(default_factory=list)  # (t, x, y) recent
    vx: float = 0.0  # how it was walking when last seen (m/s, odometry frame)
    vy: float = 0.0
    bond: float = 0.0
    care: float = 0.0  # all the care put on them (for the page)
    track: int = 0  # the eye track it has now (0: none)
    lost: bool = False  # it looked where it believed them to be and they were not there
    looked_s: float = 0.0  # how long it has looked at where they should be without seeing them

    def absorb(self, look: Sequence[float]) -> None:
        for i, value in enumerate(look[: len(self.mean)]):
            if math.isnan(value):
                continue
            if math.isnan(self.mean[i]):
                self.mean[i] = value
                self.counts[i] = 1
                continue
            self.counts[i] += 1
            rate = max(MEAN_RATE_MIN, 1.0 / self.counts[i])
            self.mean[i] += rate * (value - self.mean[i])


class Individuals:
    """The people it knows, by their looks (see the module doc)."""

    __slots__ = ("people", "_next", "_track_of", "_vote")

    def __init__(self) -> None:
        self.people: dict[int, Person] = {}
        self._next = 1
        self._track_of: dict[int, int] = {}  # eye track → person id
        self._vote: dict[int, tuple[int, int]] = {}  # eye track → (other pid, looks in a row)

    def _nearest(self, look: Sequence[float]) -> tuple[int | None, float]:
        best: tuple[int | None, float] = (None, math.inf)
        for pid, person in self.people.items():
            d = look_distance(look, person.mean)
            if d is not None and d < best[1]:
                best = (pid, d)
        return best

    def assign(self, t_s: float, seen: PersonSeen) -> int | None:
        """Which person this sighting is (their id), or None when it cannot tell (no look and no
        track it knows). A new look far from everyone is someone new."""
        track = int(seen.track)
        known = self._track_of.get(track) if track else None
        look = seen.look
        if look is None or math.isnan(look[0]):
            return known  # the same eye track: the same person; otherwise it cannot tell
        pid, d = self._nearest(look)
        if known is not None and known in self.people:
            if pid is not None and pid != known and d < JOIN_D:
                own = look_distance(look, self.people[known].mean)
                if own is None or own > d + 0.3:
                    other, n = self._vote.get(track, (pid, 0))
                    n = n + 1 if other == pid else 1
                    self._vote[track] = (pid, n)
                    if n >= TRACK_SWITCH:  # the tracker jumped to someone else
                        self._track_of[track] = pid
                        self._vote.pop(track, None)
                        self.people[pid].absorb(look)
                        return pid
                    return known
            self._vote.pop(track, None)
            self.people[known].absorb(look)
            return known
        if pid is None or d >= JOIN_D:
            pid = self._new(t_s, look)
        else:
            self.people[pid].absorb(look)
        if track:
            self._track_of[track] = pid
            if len(self._track_of) > 64:
                for old in list(self._track_of)[:16]:
                    del self._track_of[old]
        self._merge()
        return pid

    def _new(self, t_s: float, look: Sequence[float]) -> int:
        if len(self.people) >= MAX_PEOPLE:
            # forget the one it knows least (never someone it is bonded with)
            spare = [p for p in self.people.values() if p.bond < 0.05]
            if spare:
                drop = min(spare, key=lambda p: (p.seen_n, p.seen_s))
                del self.people[drop.pid]
        pid = self._next
        self._next += 1
        mean = [float(v) for v in look] + [math.nan] * (len(LOOK_DIMS) - len(look))
        counts = [0 if math.isnan(v) else 1 for v in mean]
        self.people[pid] = Person(pid, mean, counts, first_s=t_s)
        return pid

    def _merge(self) -> None:
        """Two clusters that turned out to be one person (early noise split them) become one:
        the older keeps its id, their bonds and care add up."""
        ids = sorted(self.people)
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                pa, pb = self.people.get(a), self.people.get(b)
                if pa is None or pb is None:
                    continue
                d = look_distance(pa.mean, pb.mean)
                if d is not None and d < MERGE_D:
                    pa.bond = _clip(pa.bond + pb.bond)
                    pa.care += pb.care
                    pa.seen_n += pb.seen_n
                    if pb.seen_s > pa.seen_s:
                        pa.seen_s, pa.x, pa.y = pb.seen_s, pb.x, pb.y
                        pa.range_m, pa.bearing = pb.range_m, pb.bearing
                    for k in range(len(pa.mean)):
                        if math.isnan(pa.mean[k]) and not math.isnan(pb.mean[k]):
                            pa.mean[k], pa.counts[k] = pb.mean[k], pb.counts[k]
                    for track, pid in list(self._track_of.items()):
                        if pid == b:
                            self._track_of[track] = a
                    del self.people[b]


@dataclass(slots=True)
class VoiceKnown:
    vid: int
    mean: list[float]
    n: int = 1
    who: dict[int, float] = field(default_factory=dict)  # person id → co-occurrence weight
    care: float = 0.0  # care this voice gave before it knew whose voice it is (handed on later)


class Voices:
    """The voices it knows (speaker signatures, MODEL) and whose they are (co-occurrence)."""

    __slots__ = ("voices", "_next")

    def __init__(self) -> None:
        self.voices: dict[int, VoiceKnown] = {}
        self._next = 1

    def assign(self, voice: Sequence[float]) -> int:
        best: tuple[int | None, float] = (None, math.inf)
        for key, known in self.voices.items():
            d = math.sqrt(
                sum(((voice[i] - known.mean[i]) / VOICE_SCALE[i]) ** 2 for i in range(2)) / 2.0
            )
            if d < best[1]:
                best = (key, d)
        found = best[0]
        if found is None or best[1] >= VOICE_JOIN_D:
            if len(self.voices) >= MAX_VOICES:
                del self.voices[min(self.voices.values(), key=lambda v: v.n).vid]
            vid = self._next
            self._next += 1
            self.voices[vid] = VoiceKnown(vid, [float(voice[0]), float(voice[1])])
            return vid
        vid = found
        known = self.voices[vid]
        known.n += 1
        rate = max(1.0 / 30.0, 1.0 / known.n)
        for i in range(2):
            known.mean[i] += rate * (voice[i] - known.mean[i])
        return vid

    def owner(self, vid: int) -> tuple[int | None, float]:
        """Whose voice this is (person id, confidence 0..1) from co-occurrence."""
        known = self.voices.get(vid)
        if known is None or not known.who:
            return None, 0.0
        pid, w = max(known.who.items(), key=lambda item: item[1])
        total = sum(known.who.values())
        return pid, _clip((w / total) * _clip(w / 3.0))


@dataclass(frozen=True, slots=True)
class CareCues:
    """What happened to the duck this tick that someone may have done for it (from its own
    senses: the frame's touch, its body events, the microphone, the power sense)."""

    touch_gentle: float = 0.0  # 0..1 a gentle touch or stroke going on (perceive's rule)
    touch_sharp: float = 0.0  # 0..1 a poke / a firm touch going on
    set_down: bool = False  # put back on the floor (proprio's set_down)
    speech: str | None = None  # "voice" | "name" | "hello" heard now (its onset tick)
    voice: tuple[float, float] | None = None  # the speaker signature of that speech (MODEL)
    battery_swap: bool = False  # the battery jumped up (someone swapped a charged pack in)


@dataclass(frozen=True, slots=True)
class MomView:
    """Where it believes its mother is now, and how the following is going."""

    pid: int
    bond: float
    seen: bool  # in view this tick
    believed: bool  # it believes it knows where she is (seen, or where it last saw her)
    since_s: float  # since it last saw her
    bearing: float  # rad from the trunk now (seen, or believed from odometry)
    range_m: float
    moving: bool
    still_s: float  # how long she has stood still (0 while walking)
    track: int


def plasticity(stage: str, sociability: float) -> float:
    """How much a unit of care bonds this duck (the sensitive period × sociability)."""
    a, b = SOCIAL_PLASTICITY
    return STAGE_PLASTICITY.get(stage, 0.2) * (a + b * _clip(sociability))


class Imprint:
    """One duck's family bonds and the need to stay near its mother (see the module doc)."""

    __slots__ = (
        "stage",
        "boldness",
        "sociability",
        "young",
        "plast",
        "people",
        "voices",
        "mom",
        "_seen_now",
        "_last_care",
        "_battery",
        "_t_s",
        "_pass_s",
        "events",
    )

    def __init__(
        self, *, stage: str, boldness: float, sociability: float, young: float = 0.0
    ) -> None:
        self.stage = stage
        self.boldness = _clip(boldness)
        self.sociability = _clip(sociability)
        self.young = _clip(young)
        self.plast = plasticity(stage, sociability)
        self.people = Individuals()
        self.voices = Voices()
        self.mom: int | None = None
        self._seen_now: set[int] = set()
        self._last_care: dict[str, float] = {}
        self._battery: float | None = None
        self._t_s = 0.0
        self._pass_s = -1e9
        self.events: list[str] = []  # plain-word events for the page (newest last, ≤ 12)

    @classmethod
    def from_genome(cls, genome: Any) -> Imprint:
        young = {"hatchling": 1.0, "duckling": 0.6, "juvenile": 0.25, "adult": 0.0}.get(
            str(genome.stage), 0.25
        )
        return cls(
            stage=str(genome.stage),
            boldness=float(genome.axes["boldness"]),
            sociability=float(genome.axes["sociability"]),
            young=young,
        )

    # ── the radii ────────────────────────────────────────────────────────────

    def comfort_m(self, bond: float) -> float:
        a, b, c, d = REACH_OK
        return a + b * self.boldness * (1.0 - c * self.young) + d * (1.0 - _clip(bond))

    def stop_m(self) -> float:
        a, b = REACH_STOP
        return a + b * self.boldness

    def leash_m(self, bond: float, stopped: bool) -> tuple[float, float]:
        """How far from her it is content (m, to her feet) and over how much further the pull
        grows to full: its comfort radius while she walks; once she has stopped, its stop
        distance (+ a little) — it comes to her feet."""
        comfort = self.comfort_m(bond)
        if stopped:
            return min(comfort, self.stop_m() + LEASH_STOPPED_M), FAR_RISE_STOPPED_M
        return comfort, FAR_RISE_M

    # ── the step ─────────────────────────────────────────────────────────────

    def observe(
        self,
        t_s: float,
        dt: float,
        odom: tuple[float, float, float],
        persons: Iterable[PersonSeen],
        care: CareCues,
        *,
        battery: float | None = None,
        fresh: bool = True,
    ) -> None:
        """This tick's person sightings and care, into who-is-who and the bonds. ``fresh``: the
        sightings are a new camera pass (False: no new pass this tick — who it saw on the last
        one stays "in view" for :data:`PASS_KEEP_S`)."""
        self._t_s = t_s
        dt_pass = dt  # the time this camera pass stands for (proximity care)
        if not fresh:
            if t_s - self._pass_s > PASS_KEEP_S:
                self._seen_now = set()
            persons = ()
        else:
            dt_pass = _clip(t_s - self._pass_s, 0.0, 1.0) if self._pass_s > -1e8 else dt
            self._pass_s = t_s
        seen_now: set[int] = set()
        for sight in persons:
            pid = self.people.assign(t_s, sight)
            if pid is None:
                continue
            person = self.people.people[pid]
            rng = sight.range_m if sight.range_m is not None else DEFAULT_RANGE_M
            heading = odom[2] + sight.bearing
            x = odom[0] + rng * math.cos(heading)
            y = odom[1] + rng * math.sin(heading)
            person.seen_n += 1
            person.seen_s = t_s
            person.lost = False
            person.looked_s = 0.0
            person.track = int(sight.track)
            person.range_m = rng
            person.bearing = sight.bearing
            person.x, person.y = x, y
            k = 1.0 - math.exp(-max(dt_pass, 0.02) / MOVING_TAU_S)
            person.motion += k * (float(sight.motion) - person.motion)
            person.hist.append((t_s, x, y))
            person.hist = [h for h in person.hist if t_s - h[0] <= MOVED_WINDOW_S]
            first = person.hist[0]
            moved = math.hypot(x - first[1], y - first[2])
            if person.motion > 0.15 or moved > MOVED_M + MOVED_PER_M * rng:
                person.moved_s = t_s
            span = t_s - first[0]
            if span >= 0.5:
                person.vx, person.vy = (x - first[1]) / span, (y - first[2]) / span
            if t_s - person.moved_s > 1.0:
                person.vx = person.vy = 0.0
            seen_now.add(pid)
            if rng <= CARE_NEAR_M:
                self._care(pid, CARE_NEAR_PER_S * dt_pass, quiet=True)
        if fresh:
            self._seen_now = seen_now
        # a voice, and whose it is
        if care.speech is not None:
            amount = CARE_SPEECH.get(care.speech, 0.04)
            vid = self.voices.assign(care.voice) if care.voice is not None else None
            owner: int | None = None
            if vid is not None:
                recent = [
                    p
                    for p in self.people.people.values()
                    if t_s - p.seen_s <= VOICE_SEEN_S and p.range_m <= VOICE_NEAR_M
                ]
                known = self.voices.voices[vid]
                if len(recent) == 1:  # one person in view as it hears the voice: theirs?
                    pid = recent[0].pid
                    known.who[pid] = known.who.get(pid, 0.0) + 1.0
                owner, conf = self.voices.owner(vid)
                if conf < 0.4:
                    owner = None
                if owner is not None and known.care > 0.0:
                    # now it knows whose voice that was: what that voice did counts for them
                    self._care(owner, known.care, why="voice_known")
                    known.care = 0.0
            if owner is None:
                owner = self._hand_owner(t_s, odom, reach=VOICE_OWNER_M, single=True)
            if owner is not None:
                self._care(owner, amount, why=care.speech)
            elif vid is not None:
                self.voices.voices[vid].care += amount
        # a touch, a lift, a battery: whose hand
        if care.touch_gentle > 0.3:
            pid = self._hand_owner(t_s, odom)
            if pid is not None:
                self._care(pid, CARE_TOUCH_PER_S * dt * care.touch_gentle, why="touch")
        if care.touch_sharp > 0.5 and t_s - self._last_care.get("sharp", -1e9) > 1.0:
            pid = self._hand_owner(t_s, odom)
            if pid is not None:
                self._last_care["sharp"] = t_s
                person = self.people.people[pid]
                person.bond = _clip(person.bond + SHARP_TOUCH * self.plast * person.bond)
        if care.set_down:
            self._last_care["set_down"] = t_s
        pending = self._last_care.get("set_down")
        if pending is not None and t_s - pending <= SET_DOWN_OWNER_S:
            pid = self._hand_owner(t_s, odom, window=SET_DOWN_OWNER_S + HAND_OWNER_S)
            if pid is not None:
                self._care(pid, CARE_SET_DOWN, why="carried")
                self._last_care.pop("set_down", None)
        swapped = care.battery_swap
        if battery is not None and self._battery is not None and battery - self._battery > 0.2:
            swapped = True
        if battery is not None:
            self._battery = battery
        if swapped:
            pid = self._hand_owner(t_s, odom, reach=2.0)
            if pid is not None:
                self._care(pid, CARE_BATTERY, why="battery")
        self._choose_mom()

    def _hand_owner(
        self,
        t_s: float,
        odom: tuple[float, float, float],
        window: float = HAND_OWNER_S,
        reach: float = HAND_OWNER_M,
        *,
        single: bool = False,
    ) -> int | None:
        """Whose hand (or voice) it was: the person it believes is beside it now (seen within
        the window, where it last saw them within ``reach`` of where it is now, not found
        missing since); the nearest (then the one seen last) when two are there (``single``:
        only when one is)."""
        near: list[tuple[float, int]] = []
        for person in self.people.people.values():
            if person.lost or t_s - person.seen_s > window:
                continue
            dist = math.hypot(person.x - odom[0], person.y - odom[1])
            if dist <= reach:
                # the nearest, and the one seen lately (a passer-by seen a moment ago further
                # off is not the one whose hand is on its head)
                near.append((dist + 0.01 * (t_s - person.seen_s), person.pid))
        if not near or (single and len(near) > 1):
            return None
        return min(near)[1]

    def _care(self, pid: int, amount: float, *, why: str = "", quiet: bool = False) -> None:
        person = self.people.people.get(pid)
        if person is None or amount <= 0.0:
            return
        others = max((p.bond for p in self.people.people.values() if p.pid != pid), default=0.0)
        gain = self.plast * amount * (1.0 - person.bond) * (1.0 - EXCLUSIVE * others)
        person.bond = _clip(person.bond + max(0.0, gain))
        person.care += amount
        if not quiet and why:
            self.events.append(f"{why}:{pid}:{person.bond:.2f}")
            del self.events[:-12]

    def _choose_mom(self) -> None:
        ranked = sorted(self.people.people.values(), key=lambda p: -p.bond)
        if not ranked:
            self.mom = None
            return
        top = ranked[0]
        second = ranked[1].bond if len(ranked) > 1 else 0.0
        if top.bond >= MOM_MIN and top.bond - second >= MOM_MARGIN:
            if self.mom != top.pid:
                self.events.append(f"mom:{top.pid}:{top.bond:.2f}")
                del self.events[:-12]
            self.mom = top.pid
        elif self.mom is not None and self.mom in self.people.people:
            pass  # a mother stays a mother until another is clearly ahead
        else:
            self.mom = None

    # ── the mother, and the need to be near her ─────────────────────────────

    def check_absent(
        self, dt: float, odom: tuple[float, float, float], head_yaw: float, can_see: bool
    ) -> None:
        """Object permanence: its camera looks right where it believes its mother is (the head's
        measured yaw), close enough to see her, and she is not there — for ABSENT_S: she has gone
        from there (the belief is dropped; it will look for her)."""
        if self.mom is None:
            return
        person = self.people.people.get(self.mom)
        if person is None or person.lost or self.mom in self._seen_now:
            return
        ahead = min(self._t_s - person.seen_s, PREDICT_S)
        dx = person.x + person.vx * ahead - odom[0]
        dy = person.y + person.vy * ahead - odom[1]
        bearing = _wrap(math.atan2(dy, dx) - odom[2])
        dist = math.hypot(dx, dy)
        off = abs(_wrap(bearing - head_yaw))
        looking = can_see and off < LOOKING_RAD
        nearly = can_see and off < NEAR_LOOK_RAD and dist <= NEAR_ABSENT_M
        if (looking and dist <= LOOK_RANGE_M) or nearly:
            # (right where she should be and facing about that way, it would see her legs even
            # when the narrow camera looks a little aside: a slower "not here". Facing away it
            # learns nothing — it turns round to look: mom_gone → turn_around / follow_mom)
            person.looked_s += dt if looking else 0.5 * dt
            if person.looked_s >= ABSENT_S:
                person.lost = True
                self.events.append(f"not_there:{person.pid}")
                del self.events[:-12]
        else:
            person.looked_s = max(0.0, person.looked_s - 0.5 * dt)

    def mom_view(self, odom: tuple[float, float, float]) -> MomView | None:
        if self.mom is None:
            return None
        person = self.people.people.get(self.mom)
        if person is None:
            return None
        t_s = self._t_s
        since = t_s - person.seen_s
        seen = self.mom in self._seen_now
        if seen:
            bearing, rng = person.bearing, person.range_m
        else:
            # out of sight: where she was going (her walk carried on a little, then she stopped)
            ahead = min(since, PREDICT_S) if not person.lost else 0.0
            px = person.x + person.vx * ahead
            py = person.y + person.vy * ahead
            dx, dy = px - odom[0], py - odom[1]
            bearing = _wrap(math.atan2(dy, dx) - odom[2])
            rng = math.hypot(dx, dy)
        moving = t_s - person.moved_s <= 1.0 and since <= MOVED_WINDOW_S
        still = 0.0 if moving else max(0.0, min(t_s - person.moved_s, t_s - person.first_s))
        return MomView(
            pid=person.pid,
            bond=person.bond,
            seen=seen,
            believed=not person.lost and since <= BELIEF_S,
            since_s=since,
            bearing=bearing,
            range_m=rng,
            moving=moving,
            still_s=still,
            track=person.track,
        )

    def features(self, odom: tuple[float, float, float], fear: float) -> dict[str, float]:
        """The competition's inputs (0..1): ``mom_far``, ``mom_moving``, ``mom_gone``,
        ``mom_near_still``, ``mom_fear`` and ``mom_bond`` (how strongly it is imprinted)."""
        out = {
            "mom_far": 0.0,
            "mom_moving": 0.0,
            "mom_gone": 0.0,
            "mom_near_still": 0.0,
            "mom_fear": 0.0,
            "mom_bond": 0.0,
        }
        view = self.mom_view(odom)
        if view is None:
            return out
        bond = view.bond
        out["mom_bond"] = bond
        if view.believed:
            stopped = view.still_s >= STILL_FOR_S
            leash, rise = self.leash_m(bond, stopped)
            out["mom_far"] = bond * _clip((view.range_m - leash) / rise)
            if view.moving:
                out["mom_moving"] = bond
            if stopped and view.range_m <= leash + SETTLE_EXTRA_M and view.since_s <= NEAR_SEEN_S:
                out["mom_near_still"] = bond
        if view.since_s > GONE_AFTER_S:
            rise = _clip((view.since_s - GONE_AFTER_S) / GONE_RISE_S)
            fade = 1.0 if view.since_s <= GONE_GIVE_UP_S else 0.35
            out["mom_gone"] = bond * rise * fade
        out["mom_fear"] = bond * _clip(fear)
        return out

    def goal(self, odom: tuple[float, float, float]) -> tuple[float, float] | None:
        """Where to walk to be with her: (bearing rad, range m to her feet), seen or believed;
        None when she has been out of sight too long (it goes looking instead)."""
        view = self.mom_view(odom)
        if view is None:
            return None
        if view.believed:
            return view.bearing, view.range_m
        if view.since_s <= BELIEF_S and view.range_m > SEARCH_FROM_M:
            # she was not where it thought: it goes to where it last saw her and looks from there
            return view.bearing, view.range_m
        return None  # right where it last saw her and she is not here: it looks about

    def searching(self, odom: tuple[float, float, float]) -> bool:
        """Its mother is not where it thought (it is going to where it last saw her, or looking
        about there)."""
        view = self.mom_view(odom)
        return view is not None and not view.believed

    # ── telemetry ────────────────────────────────────────────────────────────

    def view(self, odom: tuple[float, float, float]) -> dict[str, Any]:
        mom = self.mom_view(odom)
        people = sorted(self.people.people.values(), key=lambda p: -p.bond)[:6]
        return {
            "version": IMPRINT_VERSION,
            "plasticity": round(self.plast, 3),
            "mom": None
            if mom is None
            else {
                "id": mom.pid,
                "bond": round(mom.bond, 3),
                "seen": mom.seen,
                "believed": mom.believed,
                "since_s": round(mom.since_s, 1),
                "range_m": round(mom.range_m, 2),
                "bearing": round(mom.bearing, 3),
                "moving": mom.moving,
                "still_s": round(mom.still_s, 1),
            },
            "people": [
                {
                    "id": p.pid,
                    "bond": round(p.bond, 3),
                    "care": round(p.care, 3),
                    "seen_n": p.seen_n,
                }
                for p in people
            ],
            "voices": len(self.voices.voices),
            "comfort_m": round(self.comfort_m(mom.bond if mom else 0.0), 2),
            "events": list(self.events[-6:]),
        }


def persons_from(
    sightings: Iterable[Any], looks: Mapping[int, Sequence[float]] | None = None
) -> list[PersonSeen]:
    """The frame's person sightings as :class:`PersonSeen`, each with its look: the sighting's
    own ``look`` field once the contract carries it, else the side channel's (keyed by the eye's
    track id; see :mod:`duckmind.neuro.skill_social_brain`)."""
    out: list[PersonSeen] = []
    for s in sightings:
        if getattr(s, "kind", "") != "person":
            continue
        look = getattr(s, "look", None)
        if look is None and looks is not None:
            look = looks.get(int(getattr(s, "track", 0) or 0))
        out.append(
            PersonSeen(
                track=int(getattr(s, "track", 0) or 0),
                bearing=float(s.bearing),
                range_m=None if s.range_m is None else float(s.range_m),
                motion=float(getattr(s, "motion", 0.0) or 0.0),
                look=None if look is None else tuple(float(v) for v in look),
            )
        )
    return out


__all__ = [
    "CareCues",
    "IMPRINT_VERSION",
    "Imprint",
    "Individuals",
    "LOOK_DIMS",
    "MomView",
    "PersonSeen",
    "Voices",
    "look_distance",
    "persons_from",
    "plasticity",
]
