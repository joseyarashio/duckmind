"""What the duck itself perceives, as the cloud brain's proposals are checked against it (LV2).

The cloud model says "attend to the red ball", "this is 媽媽". Whether there is a red ball, or a
person to put a name on, is not for the model to say and not for the world to tell the duck: it is
what the duck's own senses (the :class:`~duckmind.neuro.contract.SensorFrame` the brain gets: its
camera's sightings, its microphone's sound kinds) hold, and what its learner remembers. This module
keeps that picture, tick by tick, and answers two questions against it:

* :meth:`Perception.person_in_view` — the person the duck sees *now* (the nearest), with its own
  person id (:meth:`~duckmind.neuro.learn.Learner.see_person`), for a name to be bound to;
* :meth:`Perception.resolve_attend` — the thing, person or sound a hint points at, among what the
  duck perceives; **nothing matches → a refusal**, never a target taken from anywhere else.

Hints are the model's words (繁體中文 mostly; a few other languages are accepted for the words of
kinds). They are matched by **kind** (a ball, a toy, a door, music, somebody) and, for people, by
**name** (:func:`duckmind.live.names.match_known`). Colour, size and the like cannot be checked
against what the duck's sightings carry (a sighting has a kind, a bearing, a range, a size, and a
look for a person): such words do not match anything and do not block a match; they are reported
as ``unverified`` so the metrics (LV6) can see how much of a hint was checked. Nothing here reads
the world's truth.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from duckmind.live import names as N

IN_VIEW_GRACE_S = 1.5  # a person seen this recently still counts as "in view" for a name
SOUND_RECENT_S = 8.0  # a sound heard this recently can still be attended to
SOUND_KEEP = 24
MIN_SOUND_DB = 36.0  # above the quiet room (30): something

# words of kinds in a hint, per kind it can match (not translated page words: the model's hint)
THING_WORDS = (
    "東西",
    "玩具",
    "球",
    "積木",
    "方塊",
    "物",
    "toy",
    "ball",
    "thing",
    "object",
    "おもちゃ",
    "ボール",
    "juguete",
    "pelota",
    "장난감",
    "공",
)
TOY_WORDS = ("玩具", "球", "積木", "方塊", "toy", "ball", "おもちゃ", "ボール", "juguete", "pelota")
SOUND_WORDS: dict[str, tuple[str, ...]] = {
    "voice": ("說話", "講話", "聲音", "人聲", "voice", "speech", "talk"),
    "door": ("門", "door"),
    "clap": ("拍手", "掌", "clap"),
    "bang": ("碰", "砰", "bang", "slam"),
    "footsteps": ("腳步", "footstep"),
    "music": ("音樂", "歌", "music", "song"),
    "tv": ("電視", "tv"),
}
UNVERIFIABLE = (
    "紅",
    "橘",
    "橙",
    "黃",
    "綠",
    "藍",
    "紫",
    "粉",
    "黑",
    "白",
    "灰",
    "咖啡",
    "大",
    "小",
    "遠",
    "近",
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "black",
    "white",
    "big",
    "small",
)


@dataclass(frozen=True, slots=True)
class SeenPerson:
    pid: int  # the duck's own person id
    track: int
    bearing: float
    range_m: float | None
    size: float
    name: str | None  # what the family has called them (the learner's)


@dataclass(frozen=True, slots=True)
class SeenThing:
    track: int
    kind: str  # "thing" | "toy"
    bearing: float
    range_m: float | None
    size: float


@dataclass(frozen=True, slots=True)
class HeardSound:
    t: float
    kind: str
    level_db: float


@dataclass(frozen=True, slots=True)
class AttendTarget:
    """What an ``attend`` proposal resolved to (everything in it is the duck's own)."""

    kind: str  # "person" | "thing" | "sound"
    track: int = 0  # the eye track (person, thing)
    pid: int | None = None  # the person's own id
    sound_kind: str | None = None
    matched: tuple[str, ...] = ()  # what of the hint was checked: "kind", "name", "generic"
    unverified: tuple[str, ...] = ()  # words of the hint that cannot be checked (colour, size…)

    def view(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "track": self.track,
            "pid": self.pid,
            "sound_kind": self.sound_kind,
            "matched": list(self.matched),
            "unverified": list(self.unverified),
        }


@dataclass(frozen=True, slots=True)
class AttendMiss:
    """Nothing the duck perceives matches: the reason is the refusal's."""

    reason: str  # "nothing_matches" | "not_in_view" | "unknown_person"


@dataclass
class Perception:
    """The duck's own picture of what is around it right now. ``update`` once per brain tick."""

    people: list[SeenPerson] = field(default_factory=list)
    things: list[SeenThing] = field(default_factory=list)
    sounds: deque[HeardSound] = field(default_factory=lambda: deque(maxlen=SOUND_KEEP))
    _last_person: SeenPerson | None = None
    _last_person_t: float = -1e9

    def update(self, frame: Any, learner: Any, now: float, t_s: float, *, asleep: bool) -> None:
        """Read one :class:`~duckmind.neuro.contract.SensorFrame` (``now``: the bridge's clock,
        ``t_s``: the brain's, which is the learner's)."""
        self.people = []
        self.things = []
        camera = frame.camera
        if camera.usable and not asleep:
            names = learner.person_names()
            for sight in camera.sightings:
                if sight.kind == "person":
                    pid = learner.see_person(sight.track, sight.look, t_s)
                    self.people.append(
                        SeenPerson(
                            pid,
                            sight.track,
                            sight.bearing,
                            sight.range_m,
                            sight.size,
                            names.get(pid),
                        )
                    )
                elif sight.kind in ("thing", "toy"):
                    self.things.append(
                        SeenThing(sight.track, sight.kind, sight.bearing, sight.range_m, sight.size)
                    )
        sound = frame.sound
        if sound.kind and sound.level_db >= MIN_SOUND_DB:
            last = self.sounds[-1] if self.sounds else None
            if last is None or last.kind != sound.kind or now - last.t > 1.0:
                self.sounds.append(HeardSound(now, str(sound.kind), float(sound.level_db)))
            else:
                self.sounds[-1] = HeardSound(last.t, last.kind, float(sound.level_db))
        if self.people:
            near = self._nearest(self.people)
            self._last_person, self._last_person_t = near, now

    @staticmethod
    def _nearest(people: Sequence[SeenPerson]) -> SeenPerson:
        return min(people, key=lambda p: (p.range_m if p.range_m is not None else 9.0, -p.size))

    # ── the person a name may be bound to ─────────────────────────────────────

    def person_in_view(self, now: float) -> tuple[SeenPerson | None, int]:
        """The nearest person in view now (or, a moment after they left it, the last one seen),
        and how many people are in view now. ``(None, 0)``: nobody to put a name on."""
        if self.people:
            return self._nearest(self.people), len(self.people)
        if self._last_person is not None and now - self._last_person_t <= IN_VIEW_GRACE_S:
            return self._last_person, 0
        return None, 0

    def person_recent(self, now: float, grace_s: float) -> SeenPerson | None:
        """The person a startle, a touch or a speech is put on: the nearest in view now, else the last
        one seen within ``grace_s`` (a startle turns the head away: they have just left the picture)."""
        if self.people:
            return self._nearest(self.people)
        if self._last_person is not None and now - self._last_person_t <= grace_s:
            return self._last_person
        return None

    # ── an attend hint against what the duck perceives ────────────────────────

    def resolve_attend(
        self, kind: str, hint: str | None, learner: Any, now: float
    ) -> AttendTarget | AttendMiss:
        text = (hint or "").strip()
        unverified = tuple(w for w in UNVERIFIABLE if w in text.casefold())
        if kind == "person":
            return self._attend_person(text, learner)
        if kind == "thing":
            return self._attend_thing(text, unverified)
        return self._attend_sound(text, now, unverified)

    def _attend_person(self, hint: str, learner: Any) -> AttendTarget | AttendMiss:
        if not hint or N.is_generic(hint) or any(g in hint for g in ("說話", "講話", "speaking")):
            if not self.people:
                return AttendMiss("nothing_matches")
            near = self._nearest(self.people)
            return AttendTarget("person", near.track, near.pid, matched=("generic",))
        pid = N.match_known(hint, learner.person_names())
        if pid is None:
            # a word of kind ("人") with something more: still only people the duck sees
            return AttendMiss("unknown_person")
        for person in sorted(
            self.people, key=lambda p: p.range_m if p.range_m is not None else 9.0
        ):
            if person.pid == pid:
                return AttendTarget("person", person.track, pid, matched=("name",))
        return AttendMiss("not_in_view")

    def _attend_thing(self, hint: str, unverified: tuple[str, ...]) -> AttendTarget | AttendMiss:
        if not self.things:
            return AttendMiss("nothing_matches")
        low = hint.casefold()
        want_toy = any(w in low for w in TOY_WORDS)
        pool = [t for t in self.things if t.kind == "toy"] if want_toy else []
        pool = pool or list(self.things)
        pick = min(pool, key=lambda t: (t.range_m if t.range_m is not None else 9.0, -t.size))
        matched = ("kind",) if any(w in low for w in THING_WORDS) else ("generic",)
        return AttendTarget("thing", pick.track, matched=matched, unverified=unverified)

    def _attend_sound(
        self, hint: str, now: float, unverified: tuple[str, ...]
    ) -> AttendTarget | AttendMiss:
        recent = [s for s in self.sounds if now - s.t <= SOUND_RECENT_S]
        if not recent:
            return AttendMiss("nothing_matches")
        low = hint.casefold()
        asked = [
            kind
            for kind, words in SOUND_WORDS.items()
            if kind != "voice" and any(w in low for w in words)
        ]
        if asked:  # a kind of sound it was told about: one it has not heard is no match
            pool = [s for s in recent if s.kind in asked]
            if not pool:
                return AttendMiss("nothing_matches")
            return AttendTarget(
                "sound", sound_kind=pool[-1].kind, matched=("kind",), unverified=unverified
            )
        voices = [s for s in recent if s.kind in ("voice", "name", "hello")]
        if voices and any(w in low for w in SOUND_WORDS["voice"]):
            return AttendTarget(
                "sound", sound_kind=voices[-1].kind, matched=("kind",), unverified=unverified
            )
        return AttendTarget(
            "sound", sound_kind=recent[-1].kind, matched=("generic",), unverified=unverified
        )

    def view(self) -> dict[str, Any]:
        return {
            "people": [p.pid for p in self.people],
            "things": [t.track for t in self.things],
            "sounds": [s.kind for s in list(self.sounds)[-4:]],
        }


__all__ = [
    "IN_VIEW_GRACE_S",
    "AttendMiss",
    "AttendTarget",
    "HeardSound",
    "Perception",
    "SeenPerson",
    "SeenThing",
]
