"""The two new drives of the live duck (live voice, LV4): 想說話 (talk) and 親近 (bond).

They replace LV2's placeholder ``TalkDrive``. Both are the duck's own state, built from what its own
senses and its own learner hold (a person in view, speech it hears, a touch it feels, a startle it
had, what the cloud brain proposed): nothing here reads the world, and nothing here moves a servo.
They only change how much the competition in the selector wants a reply, an approach or a call.

**想說話 (talk)**, one number 0..1 for the duck (:class:`TalkDrive`):

* rises with time alone (nobody in view and nobody heard for ``talk_alone_s``) and with family present
  but not engaging (a person in view who has neither spoken to it nor touched it, nor just arrived,
  for ``talk_idle_s``): toward 1,
  at ``talk_rise`` a second, faster in a vocal duck (the genome's ``vocal_rate``) and in a sociable one
  (``talk_soc_k``);
* **satiates with exchanges**: a reply the duck voices and a person's speech within
  ``exchange_window_s`` of it (before or after) is an exchange, and takes ``talk_sat_exchange`` of the
  wish away; every voiced tag takes a little (``talk_sat_tag``);
* is held at rest (``talk_rest``, higher in a vocal or sociable duck) while a person is engaging it or
  while it sleeps;
* is **suppressed** by fear and by sleepiness: the wish that reaches the competition is
  ``value × (1 − 0.8·fear) × (1 − 0.7·sleepiness)`` and nothing is voiced while it is frightened
  (fear ≥ ``FEAR_MUTE``), asleep or in a reflex: rules, not parameters (the tests pin them);
* is still pulled a little by the cloud's ``feel(talk)`` as in LV2.

**親近 (bond)**, one number 0..1 per person, kept in the learner's ``PersonRecord.bond``:

* **rises** with being *called warmly by name* (the duck's name is heard from a person it sees, and the
  cloud brain's own reading of the tone confirms it: a ``feel`` with a good valence, or a friendly reply
  proposal, within ``WARM_WAIT_S``), with being *talked to* (their speech while they are in view), with
  being *petted* (a gentle touch, per second, while they are the person at hand) and with being
  *introduced* (the family gave them a name); each is a fraction of what is left
  (``bond += gain × (1 − bond)``);
* **falls** with being *startled by them* (a startle or withdraw reflex while they are the person it
  saw: ``bond −= bond_l_startle × bond``);
* **decays slowly** (a time constant of hours: :data:`BOND_TAU_S`);
* **raises the reply weight toward that person** and the wish to approach them: the trust the reply
  weight is multiplied by is ``floor + (1 − floor) · min(1, bond / bond_full)``, the floor being what a
  stranger gets (a bold duck's is higher: ``stranger_bold_k``); the approach is an extra input on the
  ``follow`` channel while a bonded person who has just spoken to the duck or touched it is in view.

Every parameter that is fitted is in :class:`DriveParams` and the box the search looks in is
:data:`SEARCH_SPACE`. What is a rule and not a parameter (the suppression by fear and sleepiness, the
mute, the slow decay, the windows) is a module constant, with a test.

Pure Python, deterministic, clock-injected (``now`` is passed in); no network, no random numbers.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from typing import Any

# what the duck's microphone calls a person talking
SPEECH_KINDS = frozenset({"voice", "name", "hello"})
SPEECH_MIN_DB = 40.0  # a speech sound above the quiet room that counts as heard
UTTERANCE_GAP_S = 0.8  # speech that stops this long and comes again is a new utterance
ATTRIB_GRACE_S = (
    4.0  # the person a startle, a touch or a speech is put on: in view, or this recently
)
TALKER_MEMORY_S = 20.0  # the last person who talked to it, for a voice it hears with nobody in view
WARM_WAIT_S = 6.0  # how long a name call waits for the cloud's reading of its tone
WARM_VALENCE = 0.1  # a ``feel`` at least this good is a warm tone, at most minus this a cold one
FRIENDLY_TAGS = frozenset({"greet", "coo", "chirp"})  # a reply proposal that is a kind word
ENGAGED_S = 20.0  # a person who spoke to it or touched it this recently has its attention
FEAR_MUTE = 0.5  # nothing is voiced above this fear (a rule)
FEAR_SUPPRESS = 0.8  # the wish and the reply weight lose this × fear (the LV2 weights)
SLEEP_SUPPRESS = 0.7  # … and this × sleepiness
BOND_TAU_S = 4.0 * 3600.0  # bond decays slowly


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass(frozen=True)
class DriveParams:
    """Everything the new drives take from a fit. The defaults are the trained values (LV4: CMA-ES on
    the scripted conversation day, training seeds 1–12, ``docs/design/live-voice/lv4/``); use
    :meth:`neutral` for LV2's placeholder behaviour (the weights of LV2 and no new effect)."""

    # ── 想說話 ──
    talk_rest: float = 0.56625  # the wish when a person is engaging the duck (× its temperament)
    talk_rise: float = (
        0.02214  # per second toward 1, alone or with the family idle (a neutral duck)
    )
    talk_soc_k: float = -0.9608  # exp(k · (sociability − 0.5)) on that rise
    talk_sat_exchange: float = 0.28322  # of the wish, taken by one exchange
    talk_sat_tag: float = 0.05  # of the wish, taken by one voiced tag
    # ── 親近 ──
    bond_g_name: float = 0.32674  # a name called warmly
    bond_g_talk: float = 0.0493  # an utterance said to it
    bond_g_pet: float = 0.02649  # per second of a gentle touch
    bond_g_intro: float = 0.15458  # the family named them
    bond_l_startle: float = 0.15531  # a startle they caused takes this of the bond
    bond_full: float = 0.36501  # the bond at which a person is fully trusted
    stranger_floor: float = 0.62077  # the reply weight a stranger gets (a person with no bond), × …
    stranger_bold_k: float = 3.5047  # … exp(k · (boldness − 0.5))
    # ── the reply weight (toward whom, how much, how soon, how long) ──
    reply_scale: float = 0.72056
    reply_soc_k: float = 0.27917  # exp(k · (sociability − 0.5)) on the scale
    reply_act_k: float = 1.33022  # exp(k · (activity − 0.5)) on the scale
    reply_wish_gain: float = 1.2662  # the wish above its rest adds this to the reply's drive
    hesitate_s: float = 1.74024  # seconds before a reply's first tag is due, for a middling duck
    hesitate_tempo_k: float = 1.7568  # × exp(k · (0.5 − tempo)): a slow duck waits longer
    # a reply goes on to its next tag only while the wish is at least this
    continue_wish: float = 0.52053
    # ── starting a conversation ──
    init_gain: float = 1.10471  # the strength of a self-started call_out, at the top of the wish
    init_threshold: float = 0.26063  # the wish above which it starts to want to
    approach_gain: float = 0.15  # the wish to go to a bonded person who has just spoken to it

    @classmethod
    def neutral(cls) -> DriveParams:
        """LV2's behaviour: the LV2 reply weight, no stranger reserve beyond its 0.85, no hesitation,
        every tag voiced, nothing self-started, no approach."""
        return cls(
            talk_rise=0.004,
            talk_soc_k=0.0,
            talk_sat_exchange=0.15,
            talk_sat_tag=0.05,
            stranger_floor=0.85,
            stranger_bold_k=0.0,
            reply_scale=1.0,
            reply_soc_k=0.0,
            reply_act_k=0.0,
            reply_wish_gain=0.30,
            hesitate_s=0.0,
            hesitate_tempo_k=0.0,
            continue_wish=0.0,
            init_gain=0.0,
            approach_gain=0.0,
        )

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, values: Mapping[str, float]) -> DriveParams:
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown drive parameters: {sorted(unknown)}")
        return cls(**{k: float(v) for k, v in values.items()})

    def with_values(self, **values: float) -> DriveParams:
        return replace(self, **values)


@dataclass(frozen=True, slots=True)
class ParamSpec:
    """One searched parameter: its box, and whether it is searched on a log scale."""

    name: str
    low: float
    high: float
    log: bool = False

    def value(self, u: float) -> float:
        """The parameter at ``u`` in 0..1 of its box."""
        u = _clip(u)
        if self.log:
            return math.exp(math.log(self.low) + u * (math.log(self.high) - math.log(self.low)))
        return self.low + u * (self.high - self.low)

    def unit(self, value: float) -> float:
        """Where in 0..1 of its box a value is (clipped)."""
        if self.log:
            span = math.log(self.high) - math.log(self.low)
            return _clip((math.log(max(value, self.low)) - math.log(self.low)) / span)
        return _clip((value - self.low) / (self.high - self.low))


# the box of the search: only the new drives' parameters (never the brain's, the gate's, the selector's)
SEARCH_SPACE: tuple[ParamSpec, ...] = (
    ParamSpec("talk_rest", 0.1, 0.6),
    ParamSpec("talk_rise", 0.001, 0.1, log=True),
    ParamSpec("talk_soc_k", -1.0, 4.0),
    ParamSpec("talk_sat_exchange", 0.05, 0.7),
    ParamSpec("bond_g_name", 0.02, 0.5),
    ParamSpec("bond_g_talk", 0.005, 0.2),
    ParamSpec("bond_g_pet", 0.002, 0.1),
    ParamSpec("bond_g_intro", 0.0, 0.5),
    ParamSpec("bond_l_startle", 0.1, 0.8),
    ParamSpec("bond_full", 0.15, 1.0),
    ParamSpec("stranger_floor", 0.0, 1.0),
    ParamSpec("stranger_bold_k", -2.0, 4.0),
    ParamSpec("reply_scale", 0.3, 4.0, log=True),
    ParamSpec("reply_soc_k", -2.0, 4.0),
    ParamSpec("reply_act_k", -4.0, 4.0),
    ParamSpec("reply_wish_gain", 0.0, 2.0),
    ParamSpec("hesitate_s", 0.0, 3.0),
    ParamSpec("hesitate_tempo_k", 0.0, 6.0),
    ParamSpec("continue_wish", 0.0, 0.8),
    ParamSpec("init_gain", 0.0, 1.5),
    ParamSpec("init_threshold", 0.2, 0.9),
)


def params_from_vector(u: Sequence[float], base: DriveParams | None = None) -> DriveParams:
    """The parameters at a point of the unit cube (one coordinate per :data:`SEARCH_SPACE` entry)."""
    if len(u) != len(SEARCH_SPACE):
        raise ValueError(f"expected {len(SEARCH_SPACE)} coordinates, got {len(u)}")
    values = {spec.name: spec.value(x) for spec, x in zip(SEARCH_SPACE, u, strict=True)}
    return (base or DriveParams()).with_values(**values)


def vector_from_params(params: DriveParams) -> list[float]:
    """The point of the unit cube a set of parameters is at."""
    return [spec.unit(getattr(params, spec.name)) for spec in SEARCH_SPACE]


@dataclass(slots=True)
class TalkDrive:
    """想說話: the duck's wish to talk, 0..1. ``rest`` is where it sits when undisturbed."""

    value: float = 0.3
    rest: float = 0.3

    def relax(self, dt_s: float, tau_s: float) -> None:
        if dt_s > 0.0:
            keep = math.exp(-dt_s / max(1e-3, tau_s))
            self.value = self.rest + (self.value - self.rest) * keep

    def rise(self, dt_s: float, rate: float) -> None:
        """Toward 1 at ``rate`` a second (exact for any tick)."""
        if dt_s > 0.0 and rate > 0.0:
            self.value = _clip(1.0 - (1.0 - self.value) * math.exp(-rate * dt_s))

    def pull(self, target: float, rate: float) -> None:
        self.value = _clip(self.value + rate * (target - self.value))

    def satisfy(self, amount: float = 0.15) -> None:
        self.value = _clip(self.value - amount)

    def satiate(self, fraction: float) -> None:
        """A fraction of the wish is gone (an exchange, a voiced tag)."""
        self.value = _clip(self.value * (1.0 - _clip(fraction)))


@dataclass(frozen=True, slots=True)
class Observation:
    """What the duck's own senses and bridge say this tick (everything the drives read)."""

    now: float
    dt: float
    in_view_pid: int | None  # the nearest person it sees now
    person_pid: int | None  # … or the one it saw a moment ago (who a startle or a touch is put on)
    speech: bool = (
        False  # a person is speaking (the microphone's sound kind, or the voice detection)
    )
    name: bool = False  # the duck's own name is being called
    gentle_touch: bool = False
    asleep: bool = False
    fear: float = 0.0
    sleepiness: float = 0.0


@dataclass(frozen=True, slots=True)
class DriveEvent:
    """One thing that moved a drive: for the logs, the tests and the verification."""

    t: float
    kind: str  # talked_to | called_warmly | petted | startled | introduced | exchange
    pid: int | None
    detail: float = 0.0  # the bond after (or the wish after, for an exchange)


class LiveDrives:
    """The two drives of one duck. ``tick`` once per brain step; the events (a reply voiced, a startle,
    a name given, the cloud's tone) as they happen. ``learner`` holds the people (bond is theirs)."""

    def __init__(
        self,
        params: DriveParams,
        *,
        axes: Mapping[str, float],
        vocal_rate: float = 1.0,
        talk: TalkDrive | None = None,
        talk_tau_s: float = 30.0,
        talk_pull: float = 0.35,
        talk_alone_s: float = 10.0,
        talk_idle_s: float = 8.0,
        exchange_window_s: float = 4.0,
    ) -> None:
        self.params = params
        self.axes = {k: float(v) for k, v in axes.items()}
        self.vocal_rate = float(vocal_rate)
        self.talk = talk or TalkDrive(value=params.talk_rest, rest=params.talk_rest)
        was_at_rest = abs(self.talk.value - self.talk.rest) < 1e-9
        self.talk.rest = _clip(params.talk_rest * self.temperament_mult(), 0.0, 0.9)
        if was_at_rest:
            self.talk.value = self.talk.rest
        self.talk_tau_s = talk_tau_s
        self.talk_pull = talk_pull
        self.talk_alone_s = talk_alone_s
        self.talk_idle_s = talk_idle_s
        self.exchange_window_s = exchange_window_s
        self.learner: Any = None
        self.events: deque[DriveEvent] = deque(maxlen=400)
        self.counts: Counter[str] = Counter()
        self.last_person_t = -1e9  # a person was last in view
        self.arrived_t = -1e9  # … and the person in view now came into it (after being away)
        self.last_contact_t = -1e9  # … last spoke to it or touched it
        self.last_speech_t = -1e9  # … last tick with speech in it
        self.last_talker: tuple[int, float] | None = (
            None  # (pid, when) of the last person it was put on
        )
        self._speech_on = False
        self._speech_off_t = -1e9
        self._name_on = False
        self._touch_on = False
        self._pending_names: list[tuple[int, float]] = []  # (pid, when) waiting for a tone
        self._pending_exchange: dict[int, tuple[float, int | None]] = {}  # intent → (voiced t, pid)
        self.alone = False
        self.idle = False

    @property
    def speech_now(self) -> bool:
        """A person is speaking this tick (the sound kinds of the senses or the voice detection)."""
        return self._speech_on

    # ── temperament ───────────────────────────────────────────────────────────

    def _axis(self, name: str) -> float:
        return float(self.axes.get(name, 0.5))

    def temperament_mult(self) -> float:
        """How much more (or less) this duck wants to talk than a middling one: the genome's
        ``vocal_rate`` and its sociability (``talk_soc_k``). It scales the rise and the resting level."""
        return self.vocal_rate * math.exp(
            self.params.talk_soc_k * (self._axis("sociability") - 0.5)
        )

    @property
    def rise_rate(self) -> float:
        """Per second, at which the wish rises when it may (this duck's)."""
        return self.params.talk_rise * self.temperament_mult()

    def hesitation_s(self) -> float:
        p = self.params
        return max(0.0, p.hesitate_s * math.exp(p.hesitate_tempo_k * (0.5 - self._axis("tempo"))))

    def reply_scale(self) -> float:
        p = self.params
        return p.reply_scale * math.exp(
            p.reply_soc_k * (self._axis("sociability") - 0.5)
            + p.reply_act_k * (self._axis("activity") - 0.5)
        )

    def stranger_floor(self) -> float:
        p = self.params
        return _clip(
            p.stranger_floor * math.exp(p.stranger_bold_k * (self._axis("boldness") - 0.5))
        )

    # ── reads ─────────────────────────────────────────────────────────────────

    def bond(self, pid: int | None) -> float:
        person = self._person(pid)
        return 0.0 if person is None else float(person.bond)

    def trust(self, pid: int | None) -> float:
        """What a person's bond makes of the reply weight: 1 for a voice with nobody behind it (the
        duck cannot tell it is a stranger), a stranger's floor for a person it has no bond with,
        1 for one it is close to."""
        if pid is None or self._person(pid) is None:
            return 1.0
        floor = self.stranger_floor()
        frac = _clip(self.bond(pid) / max(1e-3, self.params.bond_full))
        return floor + (1.0 - floor) * frac

    def wish(self, fear: float, sleepiness: float) -> float:
        """The wish to talk that reaches the competition: suppressed by fear and by sleepiness."""
        return (
            _clip(self.talk.value)
            * (1.0 - FEAR_SUPPRESS * _clip(fear))
            * (1.0 - SLEEP_SUPPRESS * _clip(sleepiness))
        )

    def reply_drive(self, social: float, valence: float) -> float:
        """The reply's drive term (LV2's 0.35 + 0.45·social + 0.2·mood, and the wish)."""
        return (
            0.35
            + 0.45 * social
            + 0.2 * max(0.0, valence)
            + self.params.reply_wish_gain * (self.talk.value - self.talk.rest)
        )

    def continues(self, fear: float, sleepiness: float) -> bool:
        """Whether a reply that has voiced a tag goes on to its next one."""
        return self.wish(fear, sleepiness) >= self.params.continue_wish

    def initiation(self, pid: int | None, fear: float, sleepiness: float) -> float:
        """The strength, in the units of a feature × its weight, of the duck's own wish to start a
        conversation with the person it sees (0 when it has no such wish)."""
        p = self.params
        if p.init_gain <= 0.0 or pid is None:
            return 0.0
        wish = self.wish(fear, sleepiness)
        span = max(1e-3, 1.0 - p.init_threshold)
        want = _clip((wish - p.init_threshold) / span)
        return p.init_gain * want * self.trust(pid)

    def approach(self, pid: int | None, now: float) -> float:
        """The wish to go to a person it is close to who has just spoken to it or touched it."""
        if self.params.approach_gain <= 0.0 or pid is None:
            return 0.0
        if now - self.last_contact_t > ENGAGED_S:
            return 0.0
        return self.params.approach_gain * _clip(self.bond(pid) / max(1e-3, self.params.bond_full))

    # ── time ──────────────────────────────────────────────────────────────────

    def tick(self, learner: Any, obs: Observation) -> None:
        self.learner = learner
        now, dt = obs.now, max(0.0, obs.dt)
        if obs.in_view_pid is not None:
            if now - self.last_person_t >= self.talk_alone_s:
                self.arrived_t = now
            self.last_person_t = now
        self._speech(obs)
        self._touch(obs, dt)
        self._expire(now)
        self._decay_bonds(dt)
        self._talk(obs)

    def _touch(self, obs: Observation, dt: float) -> None:
        """A gentle touch with a person right there: bond per second, one event per touch."""
        now, pid = obs.now, obs.person_pid
        on = obs.gentle_touch and pid is not None and not obs.asleep
        if on:
            self.last_contact_t = now
            assert pid is not None
            self._note_talker(pid, now)
            if not self._touch_on:
                self.counts["petted"] += 1
                self.events.append(DriveEvent(now, "petted", pid, round(self.bond(pid), 4)))
            self._gain(pid, self.params.bond_g_pet * dt, "petted", now, event=False)
        self._touch_on = on

    def _speech(self, obs: Observation) -> None:
        now = obs.now
        if obs.speech:
            self.last_speech_t = now
            if not self._speech_on and now - self._speech_off_t >= UTTERANCE_GAP_S:
                self._utterance(obs)
            self._speech_on = True
        elif self._speech_on:
            self._speech_on, self._speech_off_t = False, now
        if obs.name and not self._name_on and obs.person_pid is not None:
            self._pending_names.append((obs.person_pid, now))
            self.counts["name_heard"] += 1
        self._name_on = obs.name

    def _utterance(self, obs: Observation) -> None:
        """A person began to speak."""
        now, pid = obs.now, obs.person_pid
        self.last_contact_t = now
        self.counts["utterance"] += 1
        for intent, (when, who) in list(self._pending_exchange.items()):
            if now - when <= self.exchange_window_s:
                del self._pending_exchange[intent]
                self._exchange(now, who if who is not None else pid)
        if pid is not None and not obs.asleep:
            self._note_talker(pid, now)
            self._gain(pid, self.params.bond_g_talk, "talked_to", now)

    def _note_talker(self, pid: int, now: float) -> None:
        self.last_talker = (pid, now)

    def talker(self, now: float) -> int | None:
        """The last person it was put on, if that is recent."""
        if self.last_talker is not None and now - self.last_talker[1] <= TALKER_MEMORY_S:
            return self.last_talker[0]
        return None

    def _expire(self, now: float) -> None:
        self._pending_names = [(p, t) for p, t in self._pending_names if now - t <= WARM_WAIT_S]
        for intent, (when, _who) in list(self._pending_exchange.items()):
            if now - when > self.exchange_window_s:
                del self._pending_exchange[intent]

    def _decay_bonds(self, dt: float) -> None:
        if dt <= 0.0 or self.learner is None:
            return
        keep = math.exp(-dt / BOND_TAU_S)
        for person in self.learner.people.values():
            if person.bond > 0.0:
                person.bond *= keep

    def _talk(self, obs: Observation) -> None:
        now, dt = obs.now, obs.dt
        present = obs.in_view_pid is not None
        quiet_for = now - max(self.last_person_t, self.last_contact_t)
        self.alone = (not present) and quiet_for >= self.talk_alone_s
        self.idle = present and (now - max(self.last_contact_t, self.arrived_t)) >= self.talk_idle_s
        if obs.asleep:
            self.talk.relax(dt, self.talk_tau_s)
        elif self.alone or self.idle:
            self.talk.rise(dt, self.rise_rate)
        else:
            self.talk.relax(dt, self.talk_tau_s)

    # ── events ────────────────────────────────────────────────────────────────

    def _person(self, pid: int | None) -> Any:
        if pid is None or self.learner is None:
            return None
        return self.learner.people.get(pid)

    def _gain(
        self, pid: int | None, amount: float, kind: str, now: float, *, event: bool = True
    ) -> None:
        person = self._person(pid)
        if person is None or amount <= 0.0:
            return
        person.bond = _clip(person.bond + amount * (1.0 - person.bond))
        if event:
            self.counts[kind] += 1
            self.events.append(DriveEvent(now, kind, pid, round(float(person.bond), 4)))

    def introduced(self, now: float, pid: int | None) -> None:
        """The family gave a person its name."""
        self._gain(pid, self.params.bond_g_intro, "introduced", now)

    def startled(self, now: float, pid: int | None) -> None:
        """A startle or a withdrawal while ``pid`` is who it saw: it is a little wary of them now."""
        person = self._person(pid)
        if person is None:
            return
        person.bond = _clip(person.bond - self.params.bond_l_startle * person.bond)
        self.counts["startled"] += 1
        self.events.append(DriveEvent(now, "startled", pid, round(float(person.bond), 4)))

    def on_feel(self, now: float, valence: float) -> None:
        """The cloud brain's reading of the family's tone (a ``feel`` the gate accepted)."""
        if valence >= WARM_VALENCE:
            self._warm(now)
        elif valence <= -WARM_VALENCE:
            self._pending_names.clear()  # a cold tone: the name call was not a warm one

    def on_proposal(self, now: float, lead_tag: str) -> None:
        """The cloud brain proposed a reply (a ``vocalize`` the gate accepted): a kind word means it
        took the call for a friendly one."""
        if lead_tag in FRIENDLY_TAGS:
            self._warm(now)

    def _warm(self, now: float) -> None:
        for pid, when in self._pending_names:
            if now - when <= WARM_WAIT_S:
                self._gain(pid, self.params.bond_g_name, "called_warmly", now)
        self._pending_names.clear()

    def reply_voiced(self, now: float, intent: int, pid: int | None, *, first_tag: bool) -> None:
        """The duck voiced a tag of a reply. The first one may be an exchange (a person's speech within
        the window); every tag takes a little of the wish."""
        self.talk.satiate(self.params.talk_sat_tag)
        if not first_tag:
            return
        if now - self.last_speech_t <= self.exchange_window_s:
            self._exchange(now, pid)
        else:
            self._pending_exchange[intent] = (now, pid)

    def own_call(self) -> None:
        """The brain voiced a call of its own (a call_out): it satisfies the wish a little."""
        self.talk.satiate(self.params.talk_sat_tag)

    def _exchange(self, now: float, pid: int | None) -> None:
        self.talk.satiate(self.params.talk_sat_exchange)
        person = self._person(pid)
        if person is not None:
            person.exchanges += 1
        self.counts["exchange"] += 1
        self.events.append(DriveEvent(now, "exchange", pid, round(self.talk.value, 4)))

    def feel_talk(self, target: float) -> None:
        """The cloud's ``feel(talk)``: the wish moves toward it a little (as in LV2)."""
        self.talk.pull(target, self.talk_pull)

    # ── what the page, the cloud and the tests read ───────────────────────────

    def view(self, fear: float = 0.0, sleepiness: float = 0.0) -> dict[str, Any]:
        people = []
        if self.learner is not None:
            people = [
                {"pid": pid, "bond": round(float(p.bond), 3), "exchanges": int(p.exchanges)}
                for pid, p in sorted(self.learner.people.items())
            ]
        return {
            "talk": round(float(self.talk.value), 3),
            "wish": round(self.wish(fear, sleepiness), 3),
            "alone": self.alone,
            "idle": self.idle,
            "counts": dict(sorted(self.counts.items())),
            "people": people,
        }


__all__ = [
    "ATTRIB_GRACE_S",
    "BOND_TAU_S",
    "ENGAGED_S",
    "FEAR_MUTE",
    "FEAR_SUPPRESS",
    "FRIENDLY_TAGS",
    "SEARCH_SPACE",
    "SLEEP_SUPPRESS",
    "SPEECH_KINDS",
    "SPEECH_MIN_DB",
    "DriveEvent",
    "DriveParams",
    "LiveDrives",
    "Observation",
    "ParamSpec",
    "TalkDrive",
    "params_from_vector",
    "vector_from_params",
]
