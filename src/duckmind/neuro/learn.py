"""Simple associative learning, all in the duck's own frames (odometry from boot; drifts).

* **Places** (0.5 m cells of odometry): a value that good things raise (a gentle touch, a voice
  close by, play, charging) and bad things lower (falls, bumps, scares); visits (how known a
  place is); how often the family was met there (liked spots near family and touch).
* **Owner rule (2026-09-25)**: a spot where it fell and had a *hard* recovery (up by itself only
  after > 4 s, or needing a second try or a helping hand) three or more times is avoided when
  walking. One fall never makes it timid.
* **Things** it has seen, known again by *where* they are (its own odometry) and what sort they
  are — not by the eye's track id, which a real tracker re-issues whenever a thing leaves the
  picture for a moment (the review, 2026-09-28: the same toy was "never seen" 36 times in 12 min
  and the duck kept going back to it). Novelty wears off with exposure (faster for an explorer)
  and comes back over an hour; a value from what happened with it (a thing it bumped into is
  less inviting, and stays so); where it believed the thing was (object permanence).
* **Cues** (Rescorla–Wagner): a sound or sight that is often followed by a gentle touch or a
  hello gains value — its name called → "someone nice is coming" (comes when called); a hand
  that pokes teaches the opposite.
* **The nest**: where it was last charging (odometry), for the way home when the tag is not in view.
* **People** (live voice, LV2): each person it has seen is a record of its own (its own id, what
  they look like — the look of :mod:`duckmind.neuro.skill_imprint`, an eye track while it stays
  tracked), with the name the family gave them (data: written as the family said it, never
  translated, never guessed), the names that were proposed but are not sure yet, and notes. A name
  is bound only to a person the duck itself sees; nobody is made up from a name. Each record also
  holds how close the duck feels to that person (``bond``: 親近, live voice LV4, kept by
  :mod:`duckmind.live.drives`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from duckmind.neuro.genome import NeuralParams
from duckmind.neuro.perceive import bearing_to
from duckmind.neuro.skill_imprint import JOIN_D, LOOK_DIMS, MEAN_RATE_MIN, look_distance
from duckmind.neuro.tricks import TrickBook

CELL_M = 0.5
HARD_RIGHTING_S = 4.0  # spec §5.4 step 6 / owner D3: "辛苦的翻身"
AVOID_AFTER = 3  # owner: three or more hard falls at the same spot
CUE_WINDOW_S = 8.0
THING_FORGET_S = 3600.0
# Knowing a thing again: the same eye track while it stays tracked; else the nearest remembered
# thing of the same sort within this radius (m) — looser for a thing seen far away (a bearing
# error carries it further: THING_FAR_PER_M × its distance) and with time (odometry drifts)
THING_SAME_M = 0.35
THING_FAR_PER_M = 0.3
THING_DRIFT_M_PER_MIN = 0.02
THING_DRIFT_MAX_M = 0.25
TRACK_KEEP_S = 3.0
# how fast looking at a thing uses up its novelty (× the duck's thing_hab_rate, per second): a
# curious duck has had a good look after ~10–25 s of looking (half its novelty gone)
NOVELTY_PER_S = 0.5
REAPPEAR_S = (2.0, 60.0)  # out of view this long and seen again: "there it is again"
KINDS_ALIKE = {"thing": "thing", "toy": "thing"}
PEOPLE_MAX = 12
PERSON_GONE_S = 20.0  # a person out of view this long: a track it sees now may be anyone
NOTES_MAX = 50
CUES = (
    "sound:name",
    "sound:voice",
    "sound:hello",
    "sound:door",
    "sound:footsteps",
    "visual:person",
    "visual:hand",
)


@dataclass(slots=True)
class Place:
    value: float = 0.0
    visits: float = 0.0  # seconds spent here
    family: float = 0.0  # social contact met here
    falls: int = 0
    hard_falls: int = 0
    bumps: int = 0


@dataclass(slots=True)
class Thing:
    exposure: float = 0.0  # seconds looked at / handled
    value: float = 0.0
    x: float = 0.0  # believed position (odometry frame)
    y: float = 0.0
    seen_t: float = 0.0
    kind: str = "thing"
    first_t: float = 0.0
    sightings: int = 0  # times it came into view (a new eye track that was this thing)
    track: int = 0  # the eye's latest track for it (telemetry: a target that names it carries it)


@dataclass(slots=True)
class PersonRecord:
    """Someone the duck has seen: its own id, a look, and what the family told it."""

    pid: int
    mean: list[float] = field(default_factory=lambda: [math.nan] * len(LOOK_DIMS))
    counts: list[int] = field(default_factory=lambda: [0] * len(LOOK_DIMS))
    name: str | None = None  # the family's name for them, as the duck keeps it (data)
    name_raw: str | None = None  # … as it was last proposed, before it was normalised
    pending: dict[str, int] = field(default_factory=dict)  # proposed, not sure: name → evidence
    bound_t: float | None = None  # when the name was bound (the learner's clock)
    first_t: float = 0.0
    seen_t: float = -1e9
    sightings: int = 0  # times it came into view (a new eye track that was them)
    track: int = 0  # the eye track they have now (0: none)
    greeted: int = 0  # replies the duck has made while they were in view
    # (live voice, LV4) 親近: how close the duck feels to them, 0..1. It rises with being called
    # warmly by name, being talked to and being petted by them, falls when they startle it and
    # decays slowly (duckmind.live.drives); it makes the duck trust them (answer them, go to them).
    bond: float = 0.0
    exchanges: int = 0  # to-and-fro exchanges with them (a reply of the duck and their speech)

    def absorb(self, look: Sequence[float]) -> None:
        for i, value in enumerate(look[: len(self.mean)]):
            if math.isnan(value):
                continue
            if math.isnan(self.mean[i]):
                self.mean[i] = float(value)
                self.counts[i] = 1
                continue
            self.counts[i] += 1
            self.mean[i] += max(MEAN_RATE_MIN, 1.0 / self.counts[i]) * (value - self.mean[i])

    @property
    def has_look(self) -> bool:
        return not math.isnan(self.mean[0])

    def view(self) -> dict[str, object]:
        return {
            "pid": self.pid,
            "name": self.name,
            "pending": dict(self.pending),
            "sightings": self.sightings,
            "greeted": self.greeted,
            "bond": round(self.bond, 3),
            "exchanges": self.exchanges,
        }


class Learner:
    __slots__ = (
        "_p",
        "places",
        "things",
        "cues",
        "_pending",
        "dock",
        "_fall_t",
        "_fall_cell",
        "_fall_tries",
        "_tracks",
        "_next_thing",
        "reappeared",
        "people",
        "_ptracks",
        "_next_person",
        "notes",
        "tricks",
    )

    def __init__(self, p: NeuralParams) -> None:
        self._p = p
        self.places: dict[tuple[int, int], Place] = {}
        self.things: dict[int, Thing] = {}  # its own key (not the eye's track id) → the thing
        self._tracks: dict[int, int] = {}  # the eye's track id → the thing it was
        self._next_thing = 1
        self.people: dict[int, PersonRecord] = {}  # its own person id → the record
        self._ptracks: dict[int, int] = {}  # the eye's track id → the person it was
        self._next_person = 1
        self.notes: list[tuple[float, int | None, str]] = []  # (t, person, note) kept locally
        self.reappeared = 0.0  # 0..1: a thing it knows just came back into view (a double take)
        self.cues: dict[str, float] = {}
        self._pending: dict[str, float] = {}  # cue → when it was last met (awaiting outcome)
        self.dock: tuple[float, float] | None = None
        self._fall_t: float | None = None
        self._fall_cell: tuple[int, int] | None = None
        self._fall_tries = 0
        self.tricks = TrickBook()  # the proficiency of each trick it was taught (neuro.tricks)

    # ── places ───────────────────────────────────────────────────────────────

    @staticmethod
    def cell(odom: tuple[float, float, float]) -> tuple[int, int]:
        return (math.floor(odom[0] / CELL_M), math.floor(odom[1] / CELL_M))

    def place(self, odom: tuple[float, float, float]) -> Place:
        key = self.cell(odom)
        spot = self.places.get(key)
        if spot is None:
            spot = self.places[key] = Place()
        return spot

    def visit(self, odom: tuple[float, float, float], dt_s: float) -> float:
        """Time passes here. Returns how new this place still is (1 new … 0 known)."""
        spot = self.place(odom)
        spot.visits += dt_s
        spot.value *= math.exp(-dt_s / 7200.0)  # slow forgetting
        return 1.0 / (1.0 + spot.visits / 45.0)

    def mark(self, odom: tuple[float, float, float], amount: float, family: float = 0.0) -> None:
        """A good (+) or bad (−) thing happened here."""
        spot = self.place(odom)
        spot.value = max(-1.0, min(1.0, spot.value + self._p.place_gain * amount))
        spot.family += family

    # ── falls (owner rule) ───────────────────────────────────────────────────

    def fell(self, odom: tuple[float, float, float], t_s: float) -> None:
        if self._fall_t is None:
            self._fall_t = t_s
            self._fall_cell = self.cell(odom)
            self._fall_tries = 1
            spot = self.place(odom)
            spot.falls += 1
            self.mark(odom, -0.8)
        else:
            self._fall_tries += 1

    def righted(self, t_s: float, helped: bool) -> bool:
        """Up again. Returns whether it was a hard recovery (counts toward avoiding the spot)."""
        if self._fall_t is None or self._fall_cell is None:
            return False
        hard = helped or self._fall_tries > 1 or t_s - self._fall_t > HARD_RIGHTING_S
        if hard:
            spot = self.places.setdefault(self._fall_cell, Place())
            spot.hard_falls += 1
        self._fall_t = None
        self._fall_cell = None
        self._fall_tries = 0
        return hard

    def bumped(self, odom: tuple[float, float, float]) -> None:
        self.place(odom).bumps += 1
        self.mark(odom, -0.25)

    def avoided_near(
        self, odom: tuple[float, float, float], within_m: float = 1.4
    ) -> list[tuple[float, float]]:
        """(bearing, range) of every avoided spot near it."""
        out = []
        for (i, j), spot in self.places.items():
            if spot.hard_falls >= AVOID_AFTER:
                bearing, dist = bearing_to(odom, (i + 0.5) * CELL_M, (j + 0.5) * CELL_M)
                if dist <= within_m:
                    out.append((bearing, dist))
        return out

    def liked(
        self, odom: tuple[float, float, float], leash_m: float
    ) -> tuple[float, float, float] | None:
        """The best-liked known spot (bearing, range, value): near family and touch."""
        best: tuple[float, float, float] | None = None
        for (i, j), spot in self.places.items():
            score = spot.value + 0.15 * min(3.0, spot.family)
            if score < 0.25 or spot.hard_falls >= AVOID_AFTER:
                continue
            bearing, dist = bearing_to(odom, (i + 0.5) * CELL_M, (j + 0.5) * CELL_M)
            if dist < 0.35 or dist > leash_m * 1.5:
                continue
            if best is None or score > best[2]:
                best = (bearing, dist, score)
        return best

    def unvisited_bias(self, odom: tuple[float, float, float]) -> float:
        """−1..1: which side (left +) has the less-known ground a metre ahead."""
        x, y, yaw = odom
        scores = []
        for turn in (0.7, -0.7):
            px = x + math.cos(yaw + turn)
            py = y + math.sin(yaw + turn)
            spot = self.places.get((math.floor(px / CELL_M), math.floor(py / CELL_M)))
            scores.append(1.0 / (1.0 + (spot.visits if spot else 0.0) / 45.0))
        return max(-1.0, min(1.0, (scores[0] - scores[1]) * 2.0))

    # ── things ───────────────────────────────────────────────────────────────

    def which_thing(
        self, track: int, kind: str, x: float, y: float, t_s: float, dist: float = 1.0
    ) -> int | None:
        """The remembered thing this sighting is (its key), or None for a new one: the same eye
        track while it stays tracked, else the nearest known thing of the same sort where it
        was believed to be (``dist``: how far away it is seen from, m)."""
        sort = KINDS_ALIKE.get(kind, kind)
        far = THING_FAR_PER_M * max(0.0, dist)
        key = self._tracks.get(track) if track else None
        if key is not None:
            known = self.things.get(key)
            if known is not None and t_s - known.seen_t <= TRACK_KEEP_S:
                return key
        # A track the eye has never issued before may be matched by place, but only to a thing
        # whose own sighting is no longer live. Two cases pull in opposite directions:
        #   - D2: a NEW bottle first seen near remembered books. The books were seen within
        #     TRACK_KEEP_S, i.e. another track is looking at them right now, so this new track is a
        #     different object and gets its own key.
        #   - review fix: a toy whose track ended (it left the picture) and that comes back under a
        #     NEW track id at the same place. Its last sighting is older than TRACK_KEEP_S, so
        #     nothing is looking at it any more and the place match knows it again.
        unregistered = bool(track) and track not in self._tracks
        best: tuple[float, int] | None = None
        for k, thing in self.things.items():
            if KINDS_ALIKE.get(thing.kind, thing.kind) != sort:
                continue
            if unregistered and t_s - thing.seen_t <= TRACK_KEEP_S:
                continue
            age_min = max(0.0, t_s - thing.seen_t) / 60.0
            radius = THING_SAME_M + far + min(THING_DRIFT_MAX_M, THING_DRIFT_M_PER_MIN * age_min)
            d = math.hypot(thing.x - x, thing.y - y)
            if d <= radius and (best is None or d < best[0]):
                best = (d, k)
        return None if best is None else best[1]

    def see_thing(
        self,
        track: int,
        kind: str,
        odom: tuple[float, float, float],
        x: float,
        y: float,
        t_s: float,
        dt_s: float,
        handling: float,
    ) -> tuple[int, float]:
        """A thing is in view (or being handled). Returns (its key, its novelty 0..1 before this
        look). A thing it knows that comes back into view after a while sets ``reappeared``."""
        dist = math.hypot(x - odom[0], y - odom[1])
        key = self.which_thing(track, kind, x, y, t_s, dist)
        if key is None:
            key = self._next_thing
            self._next_thing += 1
            thing = self.things[key] = Thing(kind=kind, seen_t=t_s, x=x, y=y, first_t=t_s)
            thing.sightings = 1
        else:
            thing = self.things[key]
            away = max(0.0, t_s - thing.seen_t)
            thing.exposure *= math.exp(-away / THING_FORGET_S)
            if track and self._tracks.get(track) != key:
                thing.sightings += 1  # it came into view again
                if REAPPEAR_S[0] <= away <= REAPPEAR_S[1]:
                    self.reappeared = 1.0
            # where it is believed to be: a newer and nearer look counts more (a noisy eye, a
            # bearing error that grows with the distance)
            gain = 0.35 / (1.0 + dist)
            if dist < 1.0:
                gain *= 1.0 + 2.5 * (1.0 - dist)  # a near look is the accurate one
            thing.x += gain * (x - thing.x)
            thing.y += gain * (y - thing.y)
        if track:
            self._tracks[track] = key
            thing.track = track
            if len(self._tracks) > 64:
                for old in list(self._tracks)[:16]:
                    del self._tracks[old]
        novelty = 1.0 / (1.0 + thing.exposure * self._p.thing_hab_rate * NOVELTY_PER_S)
        thing.exposure += dt_s * (1.0 + 2.0 * handling)
        thing.seen_t = t_s
        return key, novelty

    def fade(self, dt_s: float) -> None:
        """The "there it is again" impulse fades (a leaky trace, ~1.5 s)."""
        self.reappeared *= math.exp(-dt_s / 1.5)

    def thing_value(self, key: int, amount: float) -> None:
        thing = self.things.get(key)
        if thing is not None:
            thing.value = max(-1.0, min(1.0, thing.value + self._p.learn_rate * amount))

    def track_of(self, key: int) -> int:
        """The eye's latest track for a remembered thing (its key), 0 when it never had one."""
        thing = self.things.get(key)
        return 0 if thing is None else thing.track

    def remembered_thing(
        self, odom: tuple[float, float, float], t_s: float
    ) -> tuple[int, float, float, float] | None:
        """The most interesting thing it remembers but does not see: (its key, bearing, range,
        interest)."""
        best = None
        for track, thing in self.things.items():
            age = t_s - thing.seen_t
            if age < 2.0 or age > 900.0:
                continue
            exposure = thing.exposure * math.exp(-age / THING_FORGET_S)
            novelty = 1.0 / (1.0 + exposure * self._p.thing_hab_rate * NOVELTY_PER_S)
            interest = novelty + 0.5 * thing.value
            bearing, dist = bearing_to(odom, thing.x, thing.y)
            if dist > 3.0:
                continue
            if best is None or interest > best[3]:
                best = (track, bearing, dist, interest)
        return best

    def thing_belief(
        self, key: int, odom: tuple[float, float, float], t_s: float
    ) -> tuple[int, float, float, float] | None:
        """Where it believes one particular thing is, by its own odometry: (its key, bearing, range,
        interest) like :meth:`remembered_thing`, but for the thing named, however short a time ago
        it was last seen. The thing a walk is toward is kept in mind for the whole walk — the eye
        loses it while the legs step, and when it is nearer than the camera can see the floor."""
        thing = self.things.get(key)
        if thing is None:
            return None
        age = max(0.0, t_s - thing.seen_t)
        exposure = thing.exposure * math.exp(-age / THING_FORGET_S)
        novelty = 1.0 / (1.0 + exposure * self._p.thing_hab_rate * NOVELTY_PER_S)
        bearing, dist = bearing_to(odom, thing.x, thing.y)
        return (key, bearing, dist, novelty + 0.5 * thing.value)

    # ── people (live voice, LV2) ─────────────────────────────────────────────

    def see_person(self, track: int, look: Sequence[float] | None, t_s: float) -> int:
        """A person is in view: which of the people it knows (its own id). The same eye track
        while it stays tracked; else the nearest known look within :data:`JOIN_D` (the scatter of
        the eye's own looks); else the one person it saw a moment ago when no look can tell (the
        eye re-issues a track whenever someone leaves the picture: an id is no identity); else
        someone new. Never a name: a name is bound by :meth:`bind_name`."""
        record = self._person_for(track, look, t_s)
        if look is not None:
            record.absorb(look)
        if track and self._ptracks.get(track) != record.pid:
            record.sightings += 1
            self._ptracks[track] = record.pid
            if len(self._ptracks) > 64:
                for old in list(self._ptracks)[:16]:
                    del self._ptracks[old]
        record.track = track
        record.seen_t = t_s
        return record.pid

    def _person_for(self, track: int, look: Sequence[float] | None, t_s: float) -> PersonRecord:
        key = self._ptracks.get(track) if track else None
        if key is not None:
            known = self.people.get(key)
            if known is not None and t_s - known.seen_t <= TRACK_KEEP_S:
                return known
        if look is not None and not math.isnan(look[0]):
            best: tuple[float, PersonRecord] | None = None
            for person in self.people.values():
                if not person.has_look:
                    continue
                d = look_distance(look, person.mean)
                if d is not None and d < JOIN_D and (best is None or d < best[0]):
                    best = (d, person)
            if best is not None:
                return best[1]
        else:
            recent = [
                p
                for p in self.people.values()
                if t_s - p.seen_t <= PERSON_GONE_S
                and not (p.track and p.track != track and t_s - p.seen_t <= TRACK_KEEP_S)
            ]
            if len(recent) == 1:
                return recent[0]
        return self._new_person(t_s)

    def _new_person(self, t_s: float) -> PersonRecord:
        if len(self.people) >= PEOPLE_MAX:
            spare = [p for p in self.people.values() if p.name is None]
            drop = min(spare or list(self.people.values()), key=lambda p: (p.sightings, p.seen_t))
            del self.people[drop.pid]
            for track, pid in list(self._ptracks.items()):
                if pid == drop.pid:
                    del self._ptracks[track]
        record = PersonRecord(pid=self._next_person, first_t=t_s)
        self._next_person += 1
        self.people[record.pid] = record
        return record

    def person_names(self) -> dict[int, str]:
        """The names it has been given: person id → name."""
        return {pid: p.name for pid, p in self.people.items() if p.name is not None}

    def person_named(self, name: str) -> PersonRecord | None:
        for person in self.people.values():
            if person.name == name:
                return person
        return None

    def bind_name(self, pid: int, name: str, raw: str, t_s: float) -> bool:
        """The family's name for a person it sees. False if it has no such person."""
        person = self.people.get(pid)
        if person is None:
            return False
        # one name is one person: someone else already carrying exactly this name is the same
        # person seen again under another id (the look drifted, the eye re-issued a track): their
        # record is folded into this one, not left beside it as a second 伴侶
        for other in [p for p in self.people.values() if p.pid != pid and p.name == name]:
            self._merge_person(person, other)
        person.name, person.name_raw, person.bound_t = name, raw, t_s
        person.pending.pop(name, None)
        return True

    def _merge_person(self, keep: PersonRecord, drop: PersonRecord) -> None:
        """``drop`` was ``keep`` under another id: what the duck holds about them goes to ``keep``
        (its look averaged by how often each was seen, its history added), ``drop`` is gone and
        its eye tracks and notes point at ``keep``."""
        for i in range(min(len(keep.mean), len(drop.mean))):
            if math.isnan(drop.mean[i]):
                continue
            if math.isnan(keep.mean[i]):
                keep.mean[i], keep.counts[i] = drop.mean[i], drop.counts[i]
                continue
            total = max(1, keep.counts[i] + drop.counts[i])
            keep.mean[i] = (keep.mean[i] * keep.counts[i] + drop.mean[i] * drop.counts[i]) / total
            keep.counts[i] = total
        keep.sightings += drop.sightings
        keep.greeted += drop.greeted
        keep.exchanges += drop.exchanges
        keep.bond = max(keep.bond, drop.bond)
        keep.first_t = min(keep.first_t, drop.first_t)
        keep.seen_t = max(keep.seen_t, drop.seen_t)
        for name, evidence in drop.pending.items():
            keep.pending[name] = keep.pending.get(name, 0) + evidence
        del self.people[drop.pid]
        for track, pid in list(self._ptracks.items()):
            if pid == drop.pid:
                self._ptracks[track] = keep.pid
        self.notes = [
            (t, keep.pid if who == drop.pid else who, note) for t, who, note in self.notes
        ]

    def pend_name(self, pid: int, name: str) -> int:
        """A name proposed for a person that is not sure: the evidence for it so far."""
        person = self.people.get(pid)
        if person is None:
            return 0
        person.pending[name] = person.pending.get(name, 0) + 1
        if len(person.pending) > 6:
            weakest = min(person.pending, key=lambda n: person.pending[n])
            del person.pending[weakest]
        return person.pending[name]

    def keep_note(self, t_s: float, person: int | None, note: str) -> None:
        """Something worth remembering, kept here only (local, never sent anywhere)."""
        self.notes.append((t_s, person, note))
        del self.notes[:-NOTES_MAX]

    def people_view(self) -> list[dict[str, object]]:
        return [self.people[pid].view() for pid in sorted(self.people)]

    # ── cues (Rescorla–Wagner) ───────────────────────────────────────────────

    def cue(self, cls: str, t_s: float) -> float:
        """A cue was met: remember to learn from what follows. Returns its learned value."""
        if cls not in CUES:
            return 0.0
        self._settle(t_s)
        self._pending[cls] = t_s
        return self.cues.get(cls, 0.0)

    def outcome(self, value: float, t_s: float) -> None:
        """Something good (+1) or bad (−1) happened: every recent cue learns it."""
        alpha = self._p.learn_rate
        for cls, when in list(self._pending.items()):
            if t_s - when <= CUE_WINDOW_S:
                trace = math.exp(-(t_s - when) / CUE_WINDOW_S)
                v = self.cues.get(cls, 0.0)
                self.cues[cls] = v + alpha * trace * (value - v)
                del self._pending[cls]

    def _settle(self, t_s: float) -> None:
        """Cues whose window passed with nothing following: a little extinction."""
        alpha = self._p.learn_rate
        for cls, when in list(self._pending.items()):
            if t_s - when > CUE_WINDOW_S:
                v = self.cues.get(cls, 0.0)
                self.cues[cls] = v + 0.3 * alpha * (0.0 - v)
                del self._pending[cls]

    def value_of(self, cls: str) -> float:
        return self.cues.get(cls, 0.0)

    # ── the nest ─────────────────────────────────────────────────────────────

    def at_dock(self, odom: tuple[float, float, float]) -> None:
        self.dock = (odom[0], odom[1])

    # ── the slow state, kept across a restart (duckmind.life.snapshot) ───────────

    def to_state(self, t_s: float) -> dict[str, Any]:
        """What this learner has learnt, as plain JSON values: the places, the things, the cues,
        the nest, the people (with the names the family gave them and their looks) and the notes.
        Times are kept as ages at ``t_s`` (the brain's clock now), so they mean the same after a
        restart. The eye's track ids, the cues waiting for an outcome and a fall in progress are
        moments, not memories, and are not kept."""

        def age(t: float) -> float | None:
            return None if t <= -1e8 else round(max(0.0, t_s - t), 3)

        return {
            "places": [
                [
                    i,
                    j,
                    spot.value,
                    spot.visits,
                    spot.family,
                    spot.falls,
                    spot.hard_falls,
                    spot.bumps,
                ]
                for (i, j), spot in sorted(self.places.items())
            ],
            "things": [
                {
                    "key": key,
                    "kind": thing.kind,
                    "exposure": thing.exposure,
                    "value": thing.value,
                    "x": thing.x,
                    "y": thing.y,
                    "seen_age": age(thing.seen_t),
                    "first_age": age(thing.first_t),
                    "sightings": thing.sightings,
                }
                for key, thing in sorted(self.things.items())
            ],
            "next_thing": self._next_thing,
            "cues": dict(sorted(self.cues.items())),
            "dock": None if self.dock is None else [self.dock[0], self.dock[1]],
            "people": [
                {
                    "pid": p.pid,
                    "mean": [None if math.isnan(v) else v for v in p.mean],
                    "counts": list(p.counts),
                    "name": p.name,
                    "name_raw": p.name_raw,
                    "pending": dict(p.pending),
                    "bound_age": None if p.bound_t is None else age(p.bound_t),
                    "first_age": age(p.first_t),
                    "seen_age": age(p.seen_t),
                    "sightings": p.sightings,
                    "greeted": p.greeted,
                    "bond": p.bond,
                    "exchanges": p.exchanges,
                }
                for p in (self.people[pid] for pid in sorted(self.people))
            ],
            "next_person": self._next_person,
            "notes": [{"age": age(t), "person": who, "note": note} for t, who, note in self.notes],
            "tricks": self.tricks.to_state(),
        }

    def load_state(
        self, state: Mapping[str, Any], t_now: float, down_s: float = 0.0, bond_tau_s: float = 0.0
    ) -> None:
        """Take a saved ``to_state`` back in. The brain's clock is ``t_now`` now and ``down_s``
        seconds went by while nothing ran, so every saved age grows by that much (a thing not seen
        for a day is a thing not seen for a day). ``bond_tau_s``: the time constant that closeness
        to a person decays with (``duckmind.live.drives.BOND_TAU_S``); 0 leaves bonds as they were."""
        down = max(0.0, float(down_s))

        def at(age: float | None, default: float = -1e9) -> float:
            return default if age is None else t_now - float(age) - down

        self.places = {}
        for row in state.get("places", ()):
            i, j, value, visits, family, falls, hard, bumps = row
            self.places[(int(i), int(j))] = Place(
                float(value), float(visits), float(family), int(falls), int(hard), int(bumps)
            )
        self.things = {}
        for item in state.get("things", ()):
            thing = Thing(
                exposure=float(item["exposure"]),
                value=float(item["value"]),
                x=float(item["x"]),
                y=float(item["y"]),
                seen_t=at(item.get("seen_age")),
                kind=str(item.get("kind", "thing")),
                first_t=at(item.get("first_age")),
                sightings=int(item.get("sightings", 0)),
            )
            self.things[int(item["key"])] = thing
        self._tracks = {}
        self._next_thing = max(int(state.get("next_thing", 1)), 1 + max(self.things, default=0))
        self.cues = {str(k): float(v) for k, v in dict(state.get("cues", {})).items()}
        self._pending = {}
        dock = state.get("dock")
        self.dock = None if dock is None else (float(dock[0]), float(dock[1]))
        self.people = {}
        keep = math.exp(-down / bond_tau_s) if bond_tau_s > 0.0 else 1.0
        for item in state.get("people", ()):
            person = PersonRecord(pid=int(item["pid"]))
            person.mean = [math.nan if v is None else float(v) for v in item["mean"]]
            person.counts = [int(c) for c in item["counts"]]
            person.name = item.get("name")
            person.name_raw = item.get("name_raw")
            person.pending = {str(k): int(v) for k, v in dict(item.get("pending", {})).items()}
            person.bound_t = None if item.get("bound_age") is None else at(item["bound_age"])
            person.first_t = at(item.get("first_age"), 0.0)
            person.seen_t = at(item.get("seen_age"))
            person.sightings = int(item.get("sightings", 0))
            person.greeted = int(item.get("greeted", 0))
            person.bond = float(item.get("bond", 0.0)) * keep
            person.exchanges = int(item.get("exchanges", 0))
            self.people[person.pid] = person
        self._ptracks = {}
        self._next_person = max(int(state.get("next_person", 1)), 1 + max(self.people, default=0))
        self.notes = [
            (at(item.get("age"), t_now), item.get("person"), str(item["note"]))
            for item in state.get("notes", ())
        ][-NOTES_MAX:]
        self.reappeared = 0.0
        self._fall_t = None
        self._fall_cell = None
        self._fall_tries = 0
        self.tricks = TrickBook.from_state(state.get("tricks"))

    # ── telemetry ────────────────────────────────────────────────────────────

    def view(self, odom: tuple[float, float, float], leash_m: float) -> dict[str, object]:
        near = []
        cx, cy = self.cell(odom)
        for (i, j), spot in sorted(self.places.items()):
            if (
                abs(i - cx) <= 6
                and abs(j - cy) <= 6
                and (spot.visits > 5 or spot.value or spot.falls)
            ):
                near.append(
                    {
                        "cell": [i, j],
                        "value": round(spot.value, 3),
                        "visits_s": round(spot.visits, 1),
                        "family": round(spot.family, 2),
                        "falls": spot.falls,
                        "hard_falls": spot.hard_falls,
                        "avoid": spot.hard_falls >= AVOID_AFTER,
                    }
                )
        liked = self.liked(odom, leash_m)
        return {
            "cell_m": CELL_M,
            "here": [cx, cy],
            "places": near[:60],
            "avoided": [c for c, s in sorted(self.places.items()) if s.hard_falls >= AVOID_AFTER],
            "liked": None
            if liked is None
            else {
                "bearing": round(liked[0], 3),
                "range_m": round(liked[1], 2),
                "score": round(liked[2], 3),
            },
            "things": [
                {
                    "track": k,
                    "kind": t.kind,
                    "value": round(t.value, 3),
                    "seen_s": round(t.exposure, 1),
                    "sightings": t.sightings,
                }
                for k, t in sorted(self.things.items())
            ][:20],
            "cues": {k: round(v, 3) for k, v in sorted(self.cues.items())},
            "dock": None if self.dock is None else [round(self.dock[0], 2), round(self.dock[1], 2)],
        }
