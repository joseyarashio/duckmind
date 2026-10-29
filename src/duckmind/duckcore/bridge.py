"""``bridge-v1``: what the neural duck lives, as Duck Core's inputs (growth integration G2,
docs/design/GROWTH-G2-2026-10-10.md).

The neural duck is the living interface (it feels, decides and moves on the physical body); Duck
Core is who it is becoming (growth days, care, preferences, bond, the places it learnt to avoid, the
voice, the memory and the diary). Attached to the neural duck (``DuckCore.bridge_on``), Duck Core
simulates no body and no experiences of its own, and is *given* the neural duck's real experiences
as events of this vocabulary, one ``experience`` input each (``api.dispatch(session, "experience",
{"events": […]})``), which it counts by its **own growth rules**:

* every handler below calls the same primitives the family's hands call (``_presence``,
  ``_contact``, ``_arouse``, ``_care``, ``_self_note``, ``_learn_owner_bond``, ``_praise``,
  ``_experience_milestone``, ``_fall_outcome``): the caps (five of a kind of care a growth day), the
  diminishing steps of a preference, the bond's three a day, the milestones (the fifth pickup, the
  eighth slow hand, the third day of its name), the three hard falls near one spot — all of them
  are Duck Core's, none is new. ``tests/test_duckcore_bridge.py`` runs the family's own acts on an
  unattached twin and the same experiences on an attached duck and compares what each one learnt;
* what the neural duck does with its body (stands up, tucks, coos, walks, sleeps) is not decided
  here: ``_express``, ``_reflex`` and ``_spend`` do nothing for an attached duck, a reaction that
  Duck Core would have simulated is the neural duck's reflex arc now;
* an input that has no Duck Core counterpart is said so in its own docstring (``poke``).

**Events** are flat dicts of integers and a few enum words (no floats, no free text): ``k`` the
kind, ``t`` the second they happened (epoch seconds on the one clock, the duck's calendar; a second
ahead of Duck Core's clock lives it on to that second, one behind is taken at once, one more than
:data:`AHEAD_S` ahead is refused with its batch) and the kind's own fields (:data:`KINDS`). The
vocabulary is frozen with its name: a change is ``bridge-v2`` next to this one, never an edit.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, NamedTuple

from duckmind.duckcore import content
from duckmind.duckcore.fixed import SCALE

if TYPE_CHECKING:
    from duckmind.duckcore.engine import DuckCore
    from duckmind.provenance import Event

BRIDGE_API = "bridge-v1"
AHEAD_S = 60  # as api.SENSE_AHEAD_S: a sender on another clock must not live the duck on for hours
WHERE = ("head", "beak", "back", "body")
HAND_SPEEDS = ("slow", "fast")
NOISES = ("bang", "clap", "door", "other")
FALL_OUTCOMES = ("easy", "hard", "stuck", "helped")
# the verbs a duck chooses by itself, which a touch within three seconds praises (the engine's
# own rule: ``_express`` flags only these)
DOING = tuple(verb.verb_id for verb in content.VERBS if verb.kind in content.IDLE_KINDS)
LONG_NAP_MIN = 20  # engine._begin_nap: a nap this long (not the night's) is a dreamer's experience
MM = 100_000  # |x|, |y| of a fall, millimetres: a room, not an overflow

_INT_MAX = 10**9


def _say(name: str, value: Any) -> str:
    return " " + name + "=" + repr(value)


class Code(str, Enum):
    """Which rule of the vocabulary a :class:`BridgeError` says was broken."""

    NOT_A_DICT = "not_a_dict"
    UNKNOWN_KIND = "unknown_kind"
    BAD_TIME = "bad_time"
    UNKNOWN_FIELD = "unknown_field"
    MISSING_FIELD = "missing_field"
    BAD_VALUE = "bad_value"
    EVENTS_NOT_A_LIST = "events_not_a_list"
    NOT_ATTACHED = "not_attached"
    AHEAD = "ahead"
    UNKNOWN_API = "unknown_api"
    NOT_FOR_A_NEST = "not_for_a_nest_or_a_family_routine"


class BridgeError(ValueError):
    """An event (or a use of the bridge) that is not ``bridge-v1``. ``code`` names the rule it
    broke and ``detail`` the values, for the log and the caller: it says nothing in words, so it
    is no text for the page to translate."""

    def __init__(self, code: Code, **detail: Any) -> None:
        super().__init__(
            code.value + "".join(_say(name, value) for name, value in sorted(detail.items()))
        )
        self.code = code
        self.detail = detail


def _flag(value: Any) -> bool:
    return type(value) is int and value in (0, 1)


def _int(lo: int, hi: int) -> Callable[[Any], bool]:
    def check(value: Any) -> bool:
        return type(value) is int and lo <= value <= hi

    return check


def _one_of(options: tuple[str, ...]) -> Callable[[Any], bool]:
    def check(value: Any) -> bool:
        return isinstance(value, str) and value in options

    return check


def _tags(value: Any) -> bool:
    return (
        isinstance(value, list)
        and 1 <= len(value) <= 8
        and all(isinstance(tag, str) and tag in content.TAGS for tag in value)
    )


# kind → {field: (check, default)}; a field with default None must be given
_Field = tuple[Callable[[Any], bool], Any]
KINDS: dict[str, dict[str, _Field]] = {
    # a hand on it, stroking or resting (the duck's touch sense told a caress from a poke)
    "touch": {
        "where": (_one_of(WHERE), "head"),
        "ms": (_int(0, 86_400_000), 0),
        "asleep": (_flag, 0),
        "settled": (_flag, 0),  # it leaned into the hand / sat down for it (its reflex said so)
        "doing": (_one_of(("", *DOING)), ""),  # what it was doing by itself just before
    },
    "poke": {"ms": (_int(0, 60_000), 0)},  # a sharp, short, unwelcome contact
    "lift": {},  # picked up
    "hand": {"speed": (_one_of(HAND_SPEEDS), None)},  # a hand coming at the face
    "name": {"tv": (_flag, 0), "asleep": (_flag, 0)},  # someone says its name
    "voice": {"asleep": (_flag, 0)},  # someone talks
    "hello": {"asleep": (_flag, 0)},  # a greeting word
    "noise": {"what": (_one_of(NOISES), "other")},  # a bang, a clap, a door
    "company": {
        "moving": (_flag, 0)
    },  # a minute with a person right here, calm (0) or out and about (1)
    "play": {"minutes": (_int(1, 600), None)},  # played (with the ball) for that long
    "fall": {
        "x_mm": (_int(-MM, MM), None),
        "y_mm": (_int(-MM, MM), None),
        "outcome": (_one_of(FALL_OUTCOMES), None),
    },
    "sleep": {"minutes": (_int(1, 1440), None), "night": (_flag, 0)},  # it slept, and woke
    "dock": {"minutes": (_int(0, 1440), 0)},  # it sat on its charger that long
    "feed": {},  # a person put a charged pack in
    "visit": {},  # a person newly here (the family arrived)
    "voiced": {"tags": (_tags, None)},  # what it said (the tags of one call)
    "explore": {},  # somewhere or something new
}


def check(
    event: Mapping[str, Any], kinds: Mapping[str, Mapping[str, _Field]] = KINDS
) -> dict[str, Any]:
    """The event as a plain dict with its defaults, or :class:`BridgeError`. ``kinds``: the
    vocabulary it is checked against (this one; ``bridge_v2.KINDS`` is a superset of it)."""
    if not isinstance(event, Mapping):
        raise BridgeError(Code.NOT_A_DICT, got=type(event).__name__)
    kind = event.get("k")
    if not isinstance(kind, str) or kind not in kinds:
        raise BridgeError(Code.UNKNOWN_KIND, kind=kind, known=sorted(kinds))
    t = event.get("t")
    if type(t) is not int or not 0 <= t <= _INT_MAX * 4:
        raise BridgeError(Code.BAD_TIME, kind=kind, t=t)
    fields = kinds[kind]
    unknown = sorted(set(event) - set(fields) - {"k", "t"})
    if unknown:
        raise BridgeError(Code.UNKNOWN_FIELD, kind=kind, fields=unknown, has=sorted(fields))
    out: dict[str, Any] = {"k": kind, "t": t}
    for name, (checker, default) in fields.items():
        value = event.get(name, default)
        if value is None and name not in event:
            raise BridgeError(Code.MISSING_FIELD, kind=kind, field=name)
        if not checker(value):
            raise BridgeError(Code.BAD_VALUE, kind=kind, field=name, value=value)
        out[name] = value
    return out


def check_all(
    events: Any, kinds: Mapping[str, Mapping[str, _Field]] = KINDS
) -> list[dict[str, Any]]:
    if not isinstance(events, (list, tuple)):
        raise BridgeError(Code.EVENTS_NOT_A_LIST)
    return [check(event, kinds) for event in events]


# ── taking one in ────────────────────────────────────────────────────────────────────────────

RECORD_KINDS = frozenset(
    {
        "sense.touch",
        "sense.poke",
        "sense.pickup",
        "sense.hand_near",
        "sense.name_heard",
        "owner.hello",
        "sense.company",
        "sense.play",
        "body.fell",
        "sense.dock",
        "power.battery_swap",
        "owner.return",
        "sense.explore",
    }
)


class Record(NamedTuple):
    """The L1 record of an experience: what ``DuckCore.experience`` (the input handler) writes of
    the input it has just received, before the handler counts it (rule R2: a sense enters the log
    only as an input handler's own record of the input)."""

    kind: str
    payload: dict[str, Any]
    source: str


def arrive(
    core: DuckCore, event: Mapping[str, Any], kinds: Mapping[str, Mapping[str, _Field]] = KINDS
) -> tuple[dict[str, Any], bool]:
    """Check the event and live on to its second. The second answer is whether the duck is awake
    to live it (switched off, it did not: the rules of a duck off are the engine's own)."""
    ev = check(event, kinds)
    s = core.s
    if not s.bridged:
        raise BridgeError(Code.NOT_ATTACHED)
    due_ms = ev["t"] * 1000
    if due_ms - core.now_ms() > AHEAD_S * 1000:
        raise BridgeError(
            Code.AHEAD, kind=ev["k"], t=ev["t"], seconds=(due_ms - core.now_ms()) // 1000
        )
    if due_ms > core.now_ms():
        core.advance_to(due_ms)
    return ev, bool(s.awake)


def record_of(core: DuckCore, ev: dict[str, Any]) -> Record | None:
    """What Duck Core writes of this input, or None (talk and noise are kept in RAM, the day's
    count only, as for the family's voice; a sensor it does not have senses nothing; the sign of
    company is logged once in half an hour)."""
    s = core.s
    kind = ev["k"]
    if kind == "touch":
        where = "mouth" if ev["where"] == "beak" else "crown"
        payload: dict[str, Any] = {
            "where": where,
            "site": ev["where"],
            "seconds": ev["ms"] // 1000,
            "firmness": "normal",
            "path": "joint",
            "latency_ms": 0,
        }
        return Record("sense.touch", payload, "live")
    if kind == "poke":
        return Record("sense.poke", {"seconds": ev["ms"] // 1000}, "live")
    if kind == "lift":
        return Record("sense.pickup", {"path": "imu"}, "imu")
    if kind == "hand":
        return Record("sense.hand_near", {"speed": ev["speed"]}, "tof") if core.hw.tof else None
    if kind == "name":
        if not core.hw.name_ear:
            return None
        return Record("sense.name_heard", {"tv_on": bool(ev["tv"])}, "ear")
    if kind == "hello":
        absence_h = None
        if s.last_owner_hello is not None and s.ntp_ok:
            absence_h = int((s.now - s.last_owner_hello).total_seconds() // 3600)
        return Record("owner.hello", {"absence_h": absence_h}, "owner")
    if kind == "company":
        sign = core.log.get(s.bridge_company_event) if s.bridge_company_event else None
        if sign is None or s.now - sign.occurred_at >= timedelta(minutes=30):
            return Record("sense.company", {}, "live")
        return None
    if kind == "play":
        return Record("sense.play", {"minutes": ev["minutes"], "with": "ball"}, "live")
    if kind == "fall":
        payload = {"x_mm": ev["x_mm"], "y_mm": ev["y_mm"], "outcome": ev["outcome"]}
        return Record("body.fell", payload, "live")
    if kind == "dock":
        return Record("sense.dock", {"minutes": ev["minutes"]}, "live")
    if kind == "feed":
        return Record("power.battery_swap", {"by": "family", "before": 0}, "live")
    if kind == "visit":
        return Record("owner.return", {}, "owner")
    if kind == "explore":
        return Record("sense.explore", {}, "live")
    return None  # voice, noise, sleep, voiced: no record of the input itself


def _doing(core: DuckCore, ev: Mapping[str, Any], evidence: Event) -> None:
    """What it chose to do just before (touch, voice, name…): the engine's flag that a touch
    within PRAISE_WINDOW praises (``_express`` sets it for an idle verb it chose itself)."""
    verb = str(ev.get("doing") or "")
    from duckmind.duckcore import engine as E

    live = core._flag_within(E.PRAISE_WINDOW)
    if verb and not (live is not None and live["kind"] == "phrase"):  # (a babble just now wins)
        core.s.last_flag = {
            "kind": "verb",
            "id": verb,
            "at": core.s.now,
            "event": evidence.event_id,
        }


# ── the handlers: each one names the engine rule it counts by ───────────────────────────────
# (``record`` is the input's own L1 record that ``DuckCore.experience`` just wrote; it is the
# evidence of what is learnt; None for the inputs the engine keeps in RAM)


def _touch(core: DuckCore, ev: dict[str, Any], record: Event | None) -> str:
    """engine._touch_reaction (what a hand on the head does to its growth): the person is here
    (and a touch within three seconds of what it chose to do praises that), too many touches too
    fast are *overload* (never care), a caress while it sleeps is calm company, a caress while it
    is up is the family's touch care and a happier, more alert duck; a duck that leaned into the
    hand (its own reflex) had an experience of its own: the snuggler's."""
    from duckmind.duckcore import engine as E

    assert record is not None
    s = core.s
    _doing(core, ev, record)
    core._presence("touch", record)
    flag = core._flag_within(E.PRAISE_WINDOW)
    if flag is not None:
        core._praise("touch", None, record)
    burst = s.last_touch_at is not None and s.now - s.last_touch_at < timedelta(minutes=2)
    s.last_touch_at = s.now
    s.touches += 1
    s.overload = min(SCALE, s.overload + 700 + (700 if burst else 0))
    if s.overload >= 7000:
        s.overload = 4000
        return "too much at once"
    if ev["asleep"]:
        core._care("calm", record)
        return "calm company"
    s.valence = min(SCALE, s.valence + 600)
    core._contact("touch")
    core._arouse(300)
    core._care("touch", record)
    if ev["settled"]:
        core._self_note("snuggler", record, "settled_for_touch")
    return "touched"


def _poke(core: DuckCore, ev: dict[str, Any], record: Event | None) -> str:
    """NO Duck Core counterpart: Duck Core's touch is a caress on the head and its only "too
    much" is *overload*. A poke (sharp, short, unwelcome) is counted the way the engine counts a
    quick second pickup — someone is here, a jolt of arousal, an overload step of 1200 — and never
    as care and never as a happy touch. (Proposed, not built: a poke within 60 s of something it
    did is a gentle "not that".)"""
    s = core.s
    core._presence("", record)
    core._arouse(1500)
    s.overload = min(SCALE, s.overload + 1200)
    if s.overload >= 7000:
        s.overload = 4000
    return "poked"


def _lift(core: DuckCore, ev: dict[str, Any], record: Event | None) -> str:
    """engine.pickup: the person is here, a contact, a jolt of arousal; picked up again within
    two minutes is overload (and then it counts no care); the fifth pickup is the "likes being
    held" milestone."""
    assert record is not None
    s = core.s
    core._presence()
    core._contact("pickup")
    core._arouse(800)
    quick = s.last_pickup_at is not None and s.now - s.last_pickup_at < timedelta(minutes=2)
    s.last_pickup_at = s.now
    s.pickups += 1
    s.overload = min(SCALE, s.overload + (1200 if quick else 400))
    if s.overload >= 7000:
        s.overload = 4000
        return "held too much"
    core._care("touch", record)
    if s.need is not None and s.need.get("kind") == "help":
        s.need = None
    if s.pickups == 5:
        core._experience_milestone("likes_pickup")
    return "picked up"


def _hand(core: DuckCore, ev: dict[str, Any], record: Event | None) -> str:
    """engine.hand_near (a hand in the ToF's view): a ToF duck sees it. The person is here; a
    fast hand startles (only that, and it only weakens); a slow one is shy for seven times and
    leans in from the eighth, which is the "not afraid of hands" milestone."""
    s = core.s
    if record is None:
        return "no ToF"
    core._presence()
    s.last_hand_at = s.now
    core._arouse(1500 if ev["speed"] == "fast" else 300)
    if ev["speed"] == "slow":
        s.slow_hands += 1
        if s.slow_hands == 8:
            core._experience_milestone("tamed_hand")
    return ev["speed"]


def _name(core: DuckCore, ev: dict[str, Any], record: Event | None) -> str:
    """engine.call_name: it heard its name (with the TV on it could have been the TV, which
    does not count toward learning it); the third growth day of hearing it is the "knows its
    name" milestone; being called is care (talk)."""
    from duckmind.duckcore import engine as E

    s = core.s
    if record is None:
        return "no name ear"
    core._presence("name", record)
    core._contact("name")
    core._arouse(1000)
    s.name_calls += 1
    if not ev["tv"] and s.gd not in s.name_gds:
        s.name_gds.append(s.gd)
    if "learned_name" not in s.milestones and len(s.name_gds) >= E.NAME_LEARN_DAYS:
        core._experience_milestone("learned_name")
    core._care("talk", record)
    return "name"


def _voice(core: DuckCore, ev: dict[str, Any], record: Event | None) -> str:
    """engine.say(kind="voice"): talk is kept in RAM, only the day's count reaches the log; the
    person is here, a contact, a small jolt (it answers a sound it made by itself, which counts
    toward its first word)."""
    s = core.s
    if not core.hw.ear:
        return "no ear"
    s.day_voice += 1
    core._arouse(300)
    core._presence("voice")
    core._contact("voice")
    return "voice"


def _hello(core: DuckCore, ev: dict[str, Any], record: Event | None) -> str:
    """engine.owner_hello: a greeting is the person's voice here; awake, it is the bond's step
    (diminishing, three a day, and the day counts as one the owner was seen) and care (talk).
    Asleep it only opens its eyes (the engine's: say hello again when it is awake)."""
    assert record is not None
    s = core.s
    core._presence("voice", record)
    if ev["asleep"]:
        return "asleep"
    s.last_owner_hello = s.now
    s.owner_seen_today = True
    core._contact("hello")
    tier, _ = core._greeting_sounds(record.payload["absence_h"])
    core._arouse(1000 + 500 * tier)
    core._learn_owner_bond(record)
    core._care("talk", record)
    return "hello"


def _noise(core: DuckCore, ev: dict[str, Any], record: Event | None) -> str:
    """engine.say(kind="noise"): a loud sound; only the day's count is kept, and a big jolt."""
    if not core.hw.ear:
        return "no ear"
    core.s.day_noise += 1
    core._arouse(1500)
    return ev["what"]


def _company(core: DuckCore, ev: dict[str, Any], record: Event | None) -> str:
    """engine._company_minute (fair care from simply being there; a duck with a nest or a family
    routine counts it by its own cues, this one by the minute the neural duck lived with a person
    right here): out and about, a person here counts as play at most every ten minutes; resting,
    nobody having paid it attention for twenty minutes, twenty minutes of it in a row count as
    calm company at most every twenty."""
    s = core.s
    limits: dict[str, Any] = content.CARE_LIMITS
    if record is not None:  # (the evidence of the care: the sign that someone is here)
        s.bridge_company_event = record.event_id
    assert s.bridge_company_event is not None
    evidence = s.bridge_company_event
    if ev["moving"]:
        s.company_min = 0
        core._care_limited("play", "company_play", int(limits["company_play_min"]), evidence)
        return "play company"
    quiet = s.presence_at is None or s.now - s.presence_at >= timedelta(
        minutes=int(limits["company_calm_min"])
    )
    if not quiet:
        s.company_min = 0
        return "attended"
    s.company_min += 1
    if s.company_min >= int(limits["company_calm_min"]):
        s.company_min = 0
        core._care_limited("calm", "company_calm", int(limits["company_calm_min"]), evidence)
        return "calm company"
    return "company"


def _play(core: DuckCore, ev: dict[str, Any], record: Event | None) -> str:
    """engine.pad_play (play with the family): play counts as care once for every
    ``pad_play_min`` minutes of it, the person is here, a contact each minute; it ends happily
    tired. The family's hands or the ball's — the engine does not tell them apart."""
    assert record is not None
    s = core.s
    minutes = ev["minutes"]
    core._presence()
    for _ in range(min(minutes, 10)):
        core._contact("play_min")
    s.valence = min(SCALE, s.valence + 150 * min(minutes, 10))
    core._arouse(1000)
    s.play_accepted_gd = s.gd
    s.last_play_at = s.now
    per = int(content.CARE_LIMITS["pad_play_min"])
    for _ in range(max(1, minutes // per)):
        core._care("play", record, why="ball_play")
    return "played"


def _fall(core: DuckCore, ev: dict[str, Any], record: Event | None) -> str:
    """engine._fall_outcome (remember where it fell and how it got up): the spot is in the neural
    duck's own odometry frame (millimetres); one fall, or falls it got up from easily, teach
    nothing; three hard ones within half a metre teach it that the spot is not for walking."""
    assert record is not None
    s = core.s
    s.falls += 1
    attempt = core._emit(
        "duck.righting_end",
        {
            "tries": 0 if ev["outcome"] == "helped" else 1,
            "of": 3,
            "helped": "bridge",
            "outcome": ev["outcome"],
            "at": [ev["x_mm"], ev["y_mm"]],
        },
        parents=(record.event_id,),
    )
    state = {"fell": record.event_id, "x": ev["x_mm"], "y": ev["y_mm"]}
    core._fall_outcome(state, ev["outcome"], attempt)
    if ev["outcome"] == "helped":
        core._presence()  # (a hand set it upright)
    return ev["outcome"]


def _sleep(core: DuckCore, ev: dict[str, Any], record: Event | None) -> str:
    """engine._begin_nap / _end_nap: it slept ``minutes`` and woke. A nap of twenty minutes or
    more that is not the night's is the dreamer's own experience. (Duck Core's own note of it, the
    way it writes its naps; ``neural`` says the neural duck slept, not a shadow of its own.)"""
    s = core.s
    minutes = ev["minutes"]
    reason = "quiet_hours" if ev["night"] else "tired"
    start = core._emit(
        "duck.nap_start",
        {
            "minutes": minutes,
            "energy": s.energy,
            "quiet_hours": bool(ev["night"]),
            "reason": reason,
            "neural": True,
        },
    )
    if minutes >= LONG_NAP_MIN and not ev["night"]:
        core._self_note("dreamer", start, "long_nap")
    core._emit("duck.nap_end", {"by": "rested", "neural": True}, parents=(start.event_id,))
    s.nap_ended_at = s.now
    return reason


def _dock(core: DuckCore, ev: dict[str, Any], record: Event | None) -> str:
    """The charger (a nest-less duck's cable/dock): one more stay at it today."""
    s = core.s
    s.docks += 1
    s.day_docks += 1
    return "docked"


def _feed(core: DuckCore, ev: dict[str, Any], record: Event | None) -> str:
    """engine._swap_battery: a person fed it a charged pack — someone is here, and the record
    says who (the family)."""
    core._presence()
    return "fed"


def _visit(core: DuckCore, ev: dict[str, Any], record: Event | None) -> str:
    """engine.family_return: a person newly here is someone home again (and a long wait for them
    ends: the reunion's contact)."""
    core._presence("", record)
    core._contact("reunion")
    return "visit"


def _voiced(core: DuckCore, ev: dict[str, Any], record: Event | None) -> str:
    """engine._express / _maybe_babble (what it said): the sound it made by itself, which a
    person who answers within three seconds turns into a word it is learning (first word, days
    3–8) and, as a phrase of two sounds or more, a babble the family can praise (voice lineage)."""
    s = core.s
    tags = list(ev["tags"])
    if len(tags) == 1:
        s.last_sounds = {"tags": tags, "at": s.now}
        return "sound"
    code = content.phrase_code(tuple(tags))
    if code not in s.phrases:
        s.phrases[code] = {
            "code": code,
            "tags": tags,
            "text": content.phrase_text(tuple(tags)),
            "gen": 0,
            "parents": [],
            "born_gd": s.gd,
            "promoted": False,
            "praise": 0,
            "liked_gds": [],
        }
    event = core._emit(
        "voice.babble",
        {
            "code": code,
            "text": s.phrases[code]["text"],
            "tags": tags,
            "gen": s.phrases[code]["gen"],
            "parents": list(s.phrases[code]["parents"]),
            "trying": True,
            "neural": True,
        },
    )
    s.last_flag = {"kind": "phrase", "id": code, "at": s.now, "event": event.event_id}
    return "babble"


def _explore(core: DuckCore, ev: dict[str, Any], record: Event | None) -> str:
    """engine._self_note("explorer"): somewhere new, or something new, found by itself."""
    assert record is not None
    core._self_note("explorer", record, "new_place")
    return "explored"


HANDLERS: dict[str, Callable[[DuckCore, dict[str, Any], Event | None], str]] = {
    "touch": _touch,
    "poke": _poke,
    "lift": _lift,
    "hand": _hand,
    "name": _name,
    "voice": _voice,
    "hello": _hello,
    "noise": _noise,
    "company": _company,
    "play": _play,
    "fall": _fall,
    "sleep": _sleep,
    "dock": _dock,
    "feed": _feed,
    "visit": _visit,
    "voiced": _voiced,
    "explore": _explore,
}
assert set(HANDLERS) == set(KINDS)
