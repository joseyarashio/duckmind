"""The live bridge (live voice, LV2): the cloud brain's proposals become the duck's behaviour, safely.

The path of one tool call, from the cloud to the body::

    LiveSession.take_calls ─ QueuedCall ─▶ GroundingGate.judge ─ Verdict ─▶ the bridge ─┐
        (parsed, marked, queued)             (rules, caps, echo)           (what is it?)  │
                                                                                          ▼
    brain.pre ─▶ selector.set_extra  (a candidate with weights from the duck's drives)  ──▶ the competition
    brain.post ◀─ a voiced call (the existing vocal path: bank tag + beak, ``vocal_seq``) ◀─ if it won

The LLM only **proposes** and never drives a servo; reflexes never wait for the cloud:

* an accepted ``vocalize`` becomes a *reply* candidate (the social acts ``call_response`` /
  ``greet`` — the first one this growth stage has), an accepted ``attend`` an *attention*
  candidate (``watch`` for a person, ``investigate`` for a thing, ``look_around`` for a sound),
  each as extra input on its channel (:meth:`~duckmind.neuro.select.Selector.set_extra`), weighted
  by the duck's own drives (social need, mood, fear, sleepiness, its wish to talk) and passed
  through its temperament gain, the body's gates and its satiation like any other reason. A
  candidate can lose (a frightened, sleepy or busy duck; a reflex, which runs before and without
  the bridge) and then lapses (``ttl``): nothing is voiced.
* the call is voiced only when its channel *is* the duck's act this tick, no reflex has the body, a
  person is not speaking, and the brain has not just voiced something itself: the bridge sets
  ``vocal`` / ``vocal_level`` on the tick's :class:`~duckmind.neuro.contract.MotorCommand` and takes
  the next ``vocal_seq`` (one call per sequence number, as the motor lane and the page read it).
* ``attend`` is resolved against what the duck itself perceives (:mod:`duckmind.live.perception`):
  a hint with nothing behind it in its own senses is refused; the track it resolves to is the one
  the brain attends to while the candidate holds (``set_live_attend``).
* ``recognize(name)`` binds a name (:mod:`duckmind.live.names`) to the person the duck sees now, in
  the learner's record of that person; when it is not sure it keeps a *pending* name on that person
  and invents nobody. ``remember`` keeps a note locally. ``feel`` nudges the mood a little (bounded
  per call and per minute, logged) and pulls the wish to talk a little.
* The two new drives of LV4 (:mod:`duckmind.live.drives`) are the bridge's own state: 想說話 (talk, one
  for the duck) and 親近 (bond, one per person, in the learner's person records). They are fed from the
  duck's own senses (the person in view, speech heard, a touch, a startle) and from what the cloud
  proposed (the tone of a name call), and they weight the reply candidate (toward whom, how much, how
  soon, how long), start a conversation of the duck's own (an extra input on ``call_out``) and make it
  go to a person it is close to (``follow``).
* **Barge-in** (:meth:`hear`): the local voice detection (:mod:`duckmind.live.vad`, the duck's own
  echo taken out) hears a person begin to speak over the duck's call; the next tick's command stops
  the call (``vocal`` None and a bumped ``vocal_seq``: the motor closes the beak, the sound lane cuts
  the call), the replies not yet voiced are cancelled and the duck stays quiet for a moment.

* **The duck's own ear** (:meth:`LiveBridge._ear`): the cloud model's transcript of the duck's name is
  unreliable (蛋黃 comes back as 郑华 / 大华 / 段华…), so a name call rarely becomes a proposal. The duck's
  hearing model classifies a name call on the device (``SoundSense.kind == "name"`` in the sensor frame
  the brain gets, never the world's truth); when it reports its own name the bridge has the session send
  one short text (:data:`~duckmind.live.soul.EAR_NAME_HINT`, a context update the server does not
  answer), once per name event, not while the cloud is off or the line is down, and not twice within
  ``ear_hint_gap_s``. The hint only tells the model what the duck's ear heard; the model still proposes
  and the duck's drives still decide.

Every decision is logged with its reason (:attr:`LiveBridge.log`, counted in :attr:`LiveBridge.counts`
and :attr:`LiveBridge.reasons`). Nothing here reads the world: the brain's own sensor frame, its
learner and its drives. LV2 produces codes, not page words (the talk panel and its words are LV5).
"""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import cache
from typing import Any

from duckmind.live import names as N
from duckmind.live.drives import (
    ATTRIB_GRACE_S,
    FEAR_MUTE,
    FEAR_SUPPRESS,
    SLEEP_SUPPRESS,
    SPEECH_KINDS,
    SPEECH_MIN_DB,
    DriveParams,
    LiveDrives,
    Observation,
    TalkDrive,
)
from duckmind.live.gate import GateConfig, GroundingGate, Verdict
from duckmind.live.perception import AttendMiss, AttendTarget, Perception
from duckmind.live.session import OFF, QueuedCall
from duckmind.live.soul import EAR_NAME_HINT, ContextSnapshot
from duckmind.live.tools import ATTEND, FEEL, RECOGNIZE, REMEMBER, VOCALIZE
from duckmind.live.vad import SpeechEvent, VoiceDetector
from duckmind.neuro.perceive import is_stroking

# the acts a proposal is a candidate for (the first the duck's growth stage has)
REPLY_FOR_TAG: dict[str, tuple[str, ...]] = {
    "greet": ("greet", "call_response", "call_out"),
}
REPLY_DEFAULT = ("call_response", "greet", "call_out")
ATTEND_CHANNELS: dict[str, tuple[str, ...]] = {
    "person": ("watch",),
    "thing": ("investigate",),
    "sound": ("look_around",),
}
# how long a bank call lasts when the bridge has to guess (neuro_mic.CALL_S, the official bank)
CALL_LEN_S: dict[str, float] = {
    "alarm": 0.5,
    "greet": 0.6,
    "inquire": 0.5,
    "chirp": 0.3,
    "coo": 0.8,
}
LOG_MAX = 800

# bridge-level refusals and notes (codes)
NO_REPLY_CHANNEL = "no_reply_channel"
NO_PERSON = "no_person_in_view"
SEVERAL_IN_VIEW = "several_in_view"
CLOUD_OFF = "cloud_off"
BARGE_IN = "barge_in"
LOST = "lost_competition"
STARTLE_REFLEXES = frozenset({"startle", "withdraw"})
FAMILY_HINT_S = 15.0  # how long a family speaker's role is held for the body that stands for them
STRANGER_RESERVE = 0.6  # LV2: the push of a reply meant for a person the family has not named

# why a name event of the duck's own ear did not go up as a hint (codes, for the status and the tests)
EAR_RATE = "rate"
EAR_NO_LINE = "no_line"  # no session, or the line is not up
EAR_OFF = "switched_off"  # the hint is switched off in the config


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass(frozen=True)
class BridgeConfig:
    duck_name: str = "蛋黃"
    gate: GateConfig = field(default_factory=GateConfig)
    reply_ttl_s: float = 5.0  # a reply that has not won the competition this long after its last
    # tag was due lapses
    attend_ttl_s: float = 6.0
    attend_dwell_s: float = 1.0  # attention that has been the duck's act this long has been paid
    # the strength of a proposal as input to the competition (the units of a feature × its weight),
    # chosen so a social duck answers at once and a busy or lazy one often does not
    # (tools/live_voice: 18 trials per temperament, LV2 notes)
    reply_weight: float = 2.0
    # a person who has spoken to the duck and waits for its answer makes the urge to answer grow
    # (the brain's own answer to a hello or its name is an integrate-and-fire urge too: ``_answer``):
    # the proposal's input is ``reply_urge`` times more after ``reply_urge_s`` than when it came.
    # Live (LV5, real physics, 10 tries a duck, cloud on): a calm duck's acts hold the body at
    # 0.5-2.3 and a fixed input of 0.6-0.8 never got over the commitment of the act it was in (the
    # reply lapsed 5 s later: dreamer and explorer 0 of 13 proposals voiced, hatchling and snuggler
    # 10 of 12); with the urge growing five-fold in 2 s the snuggler voiced 6 of 6, the dreamer 4 of 8
    # and the bold explorer 1 of 6 of the proposals made (the lazy and the bold may still decline).
    reply_urge: float = 6.0
    reply_urge_s: float = 2.0
    # LV7: how much of that urge a person the family has not named gets. 0.0 (the default): none, so that
    # LV4's reserve toward a stranger (its floor, 0.85 and hesitation) is not overrun by waiting: with the
    # full urge a shy duck answered a stranger 100 % of the time (LV4 target 2: at most 30 %), with none
    # 8 %; 1.0: all of it, as the playable branch had it. A person it knows, and a voice with nobody
    # it can tell behind it, always get all of it (docs/design/live-voice/LV7-merge.md)
    reply_urge_stranger: float = 0.0
    # a person's answer is an event: its input claims the body from a bout that is not done
    # (``select.CLAIM_MIN``), whoever it is for. False: the plain competition, what the bridge did
    # before 2026-10-04 (the reply competes like any other reason and a busy duck can lose it)
    reply_claims: bool = True
    # fear takes the input away: full while the duck is under ``fear_calm``, none from ``fear_gone``
    # (a frightened duck does not answer, whatever the cloud proposed; a duck a little on the
    # alert, 0.1-0.3, still does)
    fear_calm: float = 0.3
    fear_gone: float = 0.7
    # … and so does sleepiness (a drowsy duck on its way to bed does not answer either)
    drowsy_from: float = 0.5
    drowsy_gone: float = 0.9
    attend_weight: float = 1.6
    min_gap_s: float = 0.45  # between two calls the bridge voices
    speech_hold_s: float = 0.5  # quiet after a person stops speaking
    barge_hold_s: float = 1.5  # quiet after a person spoke over the duck's call
    valence_step: float = 0.05  # the most one ``feel`` moves the mood
    valence_per_min: float = 0.15  # … and all of them together in a minute
    # the duck's own ear reported its name: what is sent to the cloud (None: nothing ever is) and the
    # least time between two such hints. A name call is heard for about a second (the keyword is
    # known when it has been said, held 0.4 s), so 3 s keeps one call from counting as two and
    # still lets a call repeated a few seconds later through.
    ear_name_hint: str | None = EAR_NAME_HINT
    ear_hint_gap_s: float = 3.0
    talk_pull: float = 0.35  # how far one ``feel`` moves the wish to talk toward what it says
    talk_tau_s: float = 30.0  # the wish goes back to rest this fast while a person engages it
    drives: DriveParams = field(default_factory=DriveParams)  # the new drives (LV4)
    attribution_s: float = (
        ATTRIB_GRACE_S  # a startle or a touch is put on a person seen this recently
    )
    # tell the cloud the duck's state (and the two drives) this often; None: the host does it
    context_every_s: float | None = None
    # what the duck is doing, as the cloud is told it (the behaviour's id in, words out): the page
    # tells it in the family's words; None: the id as it is
    behaviour_text: Callable[[str], str] | None = None


@dataclass
class PendingTag:
    tag: str
    level: float
    due: float  # on the bridge's clock
    emitted_at: float | None = None


@dataclass
class Intent:
    """One accepted proposal competing in the selector."""

    id: int
    kind: str  # "reply" | "attend"
    seq: int  # the call it came from
    born: float
    ttl_s: float
    channels: tuple[str, ...]
    level: float = 0.5
    tags: list[PendingTag] = field(default_factory=list)
    target: AttendTarget | None = None
    addressee: int | None = None  # the person (its own id) a reply is for, if it sees one
    known: bool = False  # … and whether it has their name
    state: str = "competing"  # competing | won | done | expired | cancelled
    won_since: float | None = None
    why: str | None = None

    @property
    def active(self) -> bool:
        return self.state in ("competing", "won")

    def deadline(self) -> float:
        last = max((t.due for t in self.tags), default=self.born)
        return max(self.born, last) + self.ttl_s

    def view(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "seq": self.seq,
            "state": self.state,
            "channels": list(self.channels),
            "level": round(self.level, 3),
            "why": self.why,
        }
        if self.tags:
            out["tags"] = [
                {"tag": t.tag, "level": t.level, "done": t.emitted_at is not None}
                for t in self.tags
            ]
        if self.target is not None:
            out["target"] = self.target.view()
        return out


@dataclass(slots=True)
class StopRequest:
    onset_t: float  # the detector's clock: the person began to speak
    decided_t: float  # … the detector was sure
    heard_now: float  # the bridge's clock when it was told


@dataclass(frozen=True, slots=True)
class BargeRecord:
    """One time a person spoke over the duck's call and what the bridge did about it."""

    onset_t: float
    decided_t: float
    detect_s: float  # decided − onset (the detector's debounce)
    apply_s: float  # the bridge's wait for the next tick's command
    stopped: bool  # False: the call had already ended
    cancelled: int  # replies not yet voiced that were cancelled

    @property
    def total_s(self) -> float:
        return self.detect_s + self.apply_s

    def view(self) -> dict[str, Any]:
        return {
            "onset_t": round(self.onset_t, 3),
            "detect_s": round(self.detect_s, 3),
            "apply_s": round(self.apply_s, 3),
            "total_s": round(self.total_s, 3),
            "stopped": self.stopped,
            "cancelled": self.cancelled,
        }


class LiveBridge:
    """See the module docstring. One per duck. The host feeds it the microphone's frames
    (:meth:`hear`) and the brain calls :meth:`pre` / :meth:`post` around each step (set
    ``brain.live = bridge``)."""

    def __init__(
        self,
        session: Any = None,
        *,
        config: BridgeConfig | None = None,
        gate: GroundingGate | None = None,
        vad: VoiceDetector | None = None,
        clock: Callable[[], float] | None = None,
        on_stop: Callable[[StopRequest], None] | None = None,
    ) -> None:
        self.session = session
        self._lock = threading.RLock()  # (the tick thread works, the page's thread reads the view)
        self.config = config or BridgeConfig()
        self.clock: Callable[[], float] = clock or (
            session.clock if session is not None else time.monotonic
        )
        self.vad = vad
        self.on_stop = on_stop
        self.gate = gate or GroundingGate(
            self.config.gate, cloud_on=self._cloud_on, human_evidence=self._human_between
        )
        self.perception = Perception()
        p = self.config.drives
        self.talk = TalkDrive(rest=p.talk_rest, value=p.talk_rest)
        self.drives: LiveDrives | None = (
            None  # (made on the first step: it needs the duck's genome)
        )
        # the role a family member speaks as (the page's chooser): (its name, when), see family_speaks
        self._family: tuple[str, float] | None = None
        self._attrib: int | None = None  # the person a startle or a touch is put on, this tick
        self._reflex_prev: str | None = None
        self._own_seq = 0  # the vocal_seq of the last call the bridge itself voiced
        self._last_context = -1e9
        self.intents: list[Intent] = []
        self.entered: list[dict[str, Any]] = []  # every candidate that was given to the selector
        # every call the bridge took from the queue (the gate's verdicts are in ``gate.log``)
        self.seen: deque[QueuedCall] = deque(maxlen=400)
        self.extras_trace: list[tuple[float, dict[str, float]]] = []  # each change of the inputs
        self.finished: deque[Intent] = deque(maxlen=200)
        self.log: deque[dict[str, Any]] = deque(maxlen=LOG_MAX)
        self.counts: Counter[str] = Counter()
        self.reasons: Counter[str] = Counter()
        self.nudges: list[dict[str, Any]] = []
        self.barges: list[BargeRecord] = []
        self.voiced: list[dict[str, Any]] = []  # the calls the bridge voiced
        self.faults = 0
        self.last_fault: str | None = None
        self.last_extras: dict[str, float] = {}
        self.last_wishes: dict[str, float] = {}  # … and the part of it that is the duck's own wish
        self._inbox: deque[QueuedCall] = deque()
        self._next_intent = 1
        self._last_emit = -1e9
        self._last_t: float | None = None
        self._calling_until = -1e9
        self._seen_vocal_seq = 0
        self._quiet_until = -1e9
        self._speaking = False
        self._spans: deque[list[float | None]] = deque(maxlen=60)  # [start, end|None], bridge clock
        self._stop: StopRequest | None = None
        self._valence_moves: deque[tuple[float, float]] = deque()
        self._pin: tuple[str, int] | None = None
        # the duck's own ear (see :meth:`_ear`)
        self._ear_named = False  # the last frame's sound kind was "name"
        self._ear_last = -1e9  # the bridge's clock at the last name event that was let through
        self.ear_counts: Counter[str] = Counter()
        self.ear_log: deque[dict[str, Any]] = deque(maxlen=60)

    # ── the host's side ───────────────────────────────────────────────────────

    def attach(self, brain: Any) -> LiveBridge:
        """Make ``brain`` (an :class:`~duckmind.neuro.integration.IntegratedBrain`) step through
        this bridge."""
        brain.live = self
        return self

    def submit(self, call: QueuedCall) -> None:
        """Hand the bridge a call directly (the session's queue is read in :meth:`pre`)."""
        with self._lock:
            self._inbox.append(call)

    def fault(self, where: str, error: BaseException) -> None:
        self.faults += 1
        self.last_fault = f"{where}: {type(error).__name__}: {error}"[:300]

    def _cloud_on(self) -> bool:
        s = self.session
        return s is None or bool(s.enabled and s.state != OFF and s.state != "capped")

    # ── the microphone: barge-in ──────────────────────────────────────────────

    def hear(self, frame: Any, windows: Any = ()) -> list[SpeechEvent]:
        """One 20 ms microphone frame (``pcm``, ``sim_s``, ``echo``) and the duck's own vocal
        windows. Feeds the local voice detection; a person beginning to speak while the duck calls
        asks for a stop (applied by the next :meth:`post`)."""
        if self.vad is None:
            return []
        try:
            with self._lock:
                return self._hear(frame, windows)
        except Exception as error:  # (a fault in the detector never stops the duck)
            self.fault("hear", error)
            return []

    def _hear(self, frame: Any, windows: Any) -> list[SpeechEvent]:
        assert self.vad is not None
        events = self.vad.process(frame.pcm, frame.sim_s, windows, echo=bool(frame.echo))
        now = self.clock()
        for w in windows:  # (the duck's own voicings say how long its call still has to run)
            if getattr(w, "kind", "call") == "call" and w.t0 <= frame.sim_s:
                left = (w.t1 if w.t1 is not None else frame.sim_s + 0.5) - frame.sim_s
                if left > 0.0:
                    self._calling_until = max(self._calling_until, now + left)
        for event in events:
            if event.kind == "speech_start":
                self._speaking = True
                self._spans.append([now, None])
                if event.over_duck or now < self._calling_until:
                    self._stop = StopRequest(event.t, event.decided_t, now)
            else:
                self._speaking = False
                if self._spans and self._spans[-1][1] is None:
                    self._spans[-1][1] = now
                self._quiet_until = now + self.config.speech_hold_s
        return events

    def _human_between(self, t0: float, t1: float) -> bool:
        """The duck's own voice detection heard a person speak between two moments (a gate rescue)."""
        for start, end in self._spans:
            if start is not None and start <= t1 and (end is None or end >= t0):
                return True
        return False

    @property
    def human_speaking(self) -> bool:
        return self._speaking

    # ── before the brain's step ───────────────────────────────────────────────

    def pre(self, brain: Any, t_ms: int, frame: Any) -> None:
        with self._lock:
            self._pre(brain, t_ms, frame)

    def _pre(self, brain: Any, t_ms: int, frame: Any) -> None:
        now = self.clock()
        t_s = t_ms / 1000.0
        asleep = str(getattr(brain, "behaviour", "")) in _sleep_acts()
        self.perception.update(frame, brain.learn, now, t_s, asleep=asleep)
        try:
            self._ear(frame, now)
        except Exception as error:  # (the rest of the tick goes on without it)
            self.fault("ear", error)
        dt = 0.0 if self._last_t is None else max(0.0, now - self._last_t)
        self._last_t = now
        self._observe(brain, frame, now, dt, asleep)
        self._guard_cloud(now)
        for call in self._drain():
            try:
                self._handle(call, now, brain, t_s)
            except Exception as error:
                self.fault("handle", error)
        self._expire(now)
        self._introduce_family(now, brain, t_s)
        self._inputs(now, brain)
        self._push_context(brain, now)

    # ── the family speaks through the page ────────────────────────────────────

    def family_speaks(self, name: str) -> None:
        """The page speaks as a member of the family (主人 or a role of the chooser): the body that
        stands for them is a known person to the duck, introduced by the role's name (names are data:
        ``name`` is whatever the page calls the role). A guest is never given here: a stranger is
        a person the duck has no name for."""
        with self._lock:
            self._family = (name, self.clock())

    def _introduce_family(self, now: float, brain: Any, t_s: float) -> None:
        held = self._family
        if held is None:
            return
        name, at = held
        if now - at > FAMILY_HINT_S:
            self._family = None
            return
        person, _many = self.perception.person_in_view(now)
        if person is None or person.name is not None:
            return
        if brain.learn.bind_name(person.pid, name, name, t_s):
            self._ensure_drives(brain).introduced(now, person.pid)
            self._family = None
            self.counts["family_introduced"] += 1

    # ── the two new drives: what the duck's own senses say this tick ──────────

    def _ensure_drives(self, brain: Any) -> LiveDrives:
        if self.drives is None:
            genome = brain.genome
            self.drives = LiveDrives(
                self.config.drives,
                axes=genome.axes,
                vocal_rate=float(genome.p.vocal_rate),
                talk=self.talk,
                talk_tau_s=self.config.talk_tau_s,
                talk_pull=self.config.talk_pull,
            )
        return self.drives

    def _observe(self, brain: Any, frame: Any, now: float, dt: float, asleep: bool) -> None:
        dr = self._ensure_drives(brain)
        sound, touch = frame.sound, frame.touch
        speech = self._speaking or (
            sound.kind in SPEECH_KINDS and float(sound.level_db) >= SPEECH_MIN_DB and not asleep
        )
        name = sound.kind == "name" and float(sound.level_db) >= SPEECH_MIN_DB and not asleep
        gentle = touch.site is not None and (is_stroking(touch.stroke) or touch.pressure < 0.45)
        near = self.perception.people
        in_view = (
            min(near, key=lambda q: q.range_m if q.range_m is not None else 9.0).pid
            if near
            else None
        )
        recent = self.perception.person_recent(now, self.config.attribution_s)
        self._attrib = None if recent is None else recent.pid
        d = brain.drives
        dr.tick(
            brain.learn,
            Observation(
                now=now,
                dt=dt,
                in_view_pid=in_view,
                person_pid=self._attrib if self._attrib is not None else dr.talker(now),
                speech=bool(speech),
                name=bool(name),
                gentle_touch=bool(gentle),
                asleep=asleep,
                fear=float(d.fear),
                sleepiness=float(d.sleepiness),
            ),
        )

    def snapshot(self, brain: Any) -> ContextSnapshot:
        """What the duck tells the cloud brain about itself every few seconds: what it is doing, its
        mood, its drives (the wish to talk, and its closeness to the person it sees), who it knows
        (with how close it is to each) and what its eyes see."""
        dr = self._ensure_drives(brain)
        d = brain.drives
        drives = dict(d.as_dict())
        drives["talk"] = dr.wish(float(d.fear), float(d.sleepiness))
        names = brain.learn.person_names()
        sees = list(self.perception.people)
        person = None
        if sees:
            person = min(sees, key=lambda q: q.range_m if q.range_m is not None else 9.0)
            drives["bond"] = dr.bond(person.pid)
        seen: list[str] = []
        for q in sees:
            seen.append(names.get(q.pid) or "不認識的人")
        seen += ["玩具" if t.kind == "toy" else "東西" for t in self.perception.things]
        behaviour = str(getattr(brain, "behaviour", "") or "")
        if behaviour and self.config.behaviour_text is not None:
            try:
                behaviour = str(self.config.behaviour_text(behaviour))
            except Exception:  # (the id is still true)
                pass
        return ContextSnapshot(
            behaviour=behaviour,
            valence=float(d.valence),
            drives=drives,
            known_people=tuple(sorted(names.values())),
            in_view=tuple(seen),
            bonds={n: dr.bond(pid) for pid, n in names.items()},
        )

    def _push_context(self, brain: Any, now: float) -> None:
        every = self.config.context_every_s
        session = self.session
        if every is None or session is None or now - self._last_context < every:
            return
        setter = getattr(session, "set_context", None)
        if setter is None:
            return
        self._last_context = now
        setter(self.snapshot(brain))

    # ── the duck's own ear ────────────────────────────────────────────────────

    def _ear(self, frame: Any, now: float) -> None:
        """The duck's own hearing (the sound kind in the sensor frame the brain gets: the hearing
        model's on-device classification, never the world's truth) reports that its name was called:
        the cloud is told once, in one short text (``ear_name_hint``). A name event begins when the
        kind turns to ``name`` from anything else; it is not told while the cloud is off, nor when
        the line is not up (nothing is kept for later), nor within ``ear_hint_gap_s`` of the last
        hint that went up."""
        sound = getattr(frame, "sound", None)
        named = getattr(sound, "kind", None) == "name"
        began = named and not self._ear_named
        self._ear_named = named
        if not began:
            return
        cfg = self.config
        self.ear_counts["name_events"] += 1
        if cfg.ear_name_hint is None:
            self._ear_skip(now, EAR_OFF, sound)
            return
        if not self._cloud_on():
            self._ear_skip(now, CLOUD_OFF, sound)
            return
        if now - self._ear_last < cfg.ear_hint_gap_s:
            self._ear_skip(now, EAR_RATE, sound)
            return
        send = getattr(self.session, "send_note", None)
        sent = False
        if callable(send):
            sent = bool(send(cfg.ear_name_hint, about="ear_name"))
        if not sent:
            self._ear_skip(now, EAR_NO_LINE, sound)
            return
        self._ear_last = now
        self.ear_counts["hints_sent"] += 1
        self.ear_log.append({"t": round(now, 3), "sent": True, "why": None, **_ear_level(sound)})

    def _ear_skip(self, now: float, why: str, sound: Any) -> None:
        self.ear_counts[f"skipped:{why}"] += 1
        self.ear_log.append({"t": round(now, 3), "sent": False, "why": why, **_ear_level(sound)})

    def _drain(self) -> list[QueuedCall]:
        calls: list[QueuedCall] = []
        if self.session is not None:
            calls += self.session.take_calls()
        while self._inbox:
            calls.append(self._inbox.popleft())
        self.seen.extend(calls)
        return calls

    def _guard_cloud(self, now: float) -> None:
        if not self._cloud_on():  # (the duck acts on nothing the cloud said once it is off)
            for intent in self.intents:
                if intent.active:
                    self._end(intent, "cancelled", CLOUD_OFF, now)

    # ── one call ──────────────────────────────────────────────────────────────

    def _record(
        self,
        call: QueuedCall,
        decision: str,
        reason: str | None,
        now: float,
        stage: str,
        **more: Any,
    ) -> None:
        self.counts[f"{call.name}:{decision}"] += 1
        if reason and stage != "gate":  # (the gate's reasons are counted from its own refusals)
            self.reasons[reason] += 1
        self.log.append(
            {
                "t": round(now, 3),
                "seq": call.seq,
                "tool": call.name,
                "decision": decision,
                "stage": stage,
                "reason": reason,
                **more,
            }
        )

    def _handle(self, call: QueuedCall, now: float, brain: Any, t_s: float) -> None:
        verdict = self.gate.judge(call, now)
        for refusal in verdict.refused:
            self.reasons[refusal.reason] += 1
        if not verdict.accepted:
            self._record(call, "refused", verdict.reason, now, "gate")
            return
        if call.name == VOCALIZE:
            self._vocalize(call, verdict, now, brain)
        elif call.name == ATTEND:
            self._attend(call, verdict, now, brain)
        elif call.name == RECOGNIZE:
            self._recognize(call, verdict, now, brain, t_s)
        elif call.name == REMEMBER:
            self._remember(call, verdict, now, brain, t_s)
        elif call.name == FEEL:
            self._feel(call, verdict, now, brain)

    # ── vocalize → a reply candidate ──────────────────────────────────────────

    def _reply_channels(self, brain: Any, lead: str) -> tuple[str, ...]:
        names = set(brain.sel.names)
        for channel in REPLY_FOR_TAG.get(lead, REPLY_DEFAULT):
            if channel in names:
                return (channel,)
        return ()

    def _vocalize(self, call: QueuedCall, verdict: Verdict, now: float, brain: Any) -> None:
        lead = verdict.tags[0].tag
        channels = self._reply_channels(brain, lead)
        if not channels:
            self._record(call, "refused", NO_REPLY_CHANNEL, now, "bridge")
            return
        person, _many = self.perception.person_in_view(now)
        names = brain.learn.person_names()
        dr = self._ensure_drives(brain)
        wait = dr.hesitation_s()  # (a slow duck takes its time)
        intent = Intent(
            id=self._next_intent,
            kind="reply",
            seq=call.seq,
            born=now,
            ttl_s=self.config.reply_ttl_s,
            channels=channels,
            level=sum(t.level for t in verdict.tags) / len(verdict.tags),
            tags=[PendingTag(t.tag, t.level, now + wait + t.at_ms / 1000.0) for t in verdict.tags],
            addressee=None if person is None else person.pid,
            known=person is not None and person.pid in names,
        )
        dr.on_proposal(now, lead)
        self._enter(
            intent,
            call,
            now,
            tags=[(t.tag, t.level, t.at_ms) for t in verdict.tags],
            addressee=intent.addressee,
            known=intent.known,
        )
        self._record(
            call, "accepted", None, now, "bridge", intent=intent.id, rescued=verdict.rescued
        )

    def _enter(self, intent: Intent, call: QueuedCall, now: float, **more: Any) -> None:
        self._next_intent += 1
        self.intents.append(intent)
        del self.entered[: -LOG_MAX // 2]
        self.entered.append(
            {
                "intent": intent.id,
                "seq": call.seq,
                "tool": call.name,
                "kind": intent.kind,
                "channels": list(intent.channels),
                "t": round(now, 3),
                **more,
            }
        )

    # ── attend → an attention candidate ───────────────────────────────────────

    def _attend(self, call: QueuedCall, verdict: Verdict, now: float, brain: Any) -> None:
        kind = str(verdict.args["target_kind"])
        found = self.perception.resolve_attend(kind, verdict.args.get("hint"), brain.learn, now)
        if isinstance(found, AttendMiss):
            self._record(call, "refused", found.reason, now, "bridge", kind=kind)
            return
        channels = tuple(c for c in ATTEND_CHANNELS[kind] if c in brain.sel.names)
        if not channels:
            self._record(call, "refused", NO_REPLY_CHANNEL, now, "bridge", kind=kind)
            return
        for old in self.intents:  # (one thing at a time: a new proposal replaces the old one)
            if old.kind == "attend" and old.active:
                self._end(old, "cancelled", "replaced", now)
        intent = Intent(
            id=self._next_intent,
            kind="attend",
            seq=call.seq,
            born=now,
            ttl_s=self.config.attend_ttl_s,
            channels=channels,
            level=0.6,
            target=found,
        )
        self._enter(intent, call, now, target=found.view())
        self._record(call, "accepted", None, now, "bridge", intent=intent.id, target=found.view())

    # ── recognize → a name on a person it sees ────────────────────────────────

    def _recognize(
        self, call: QueuedCall, verdict: Verdict, now: float, brain: Any, t_s: float
    ) -> None:
        learner = brain.learn
        raw = str(verdict.args["name"])
        person, many = self.perception.person_in_view(now)
        names = learner.person_names()
        speech_age = self._speech_age(call, now)
        decision = N.decide_name(
            raw,
            in_view_pid=None if person is None else person.pid,
            in_view_name=None if person is None else names.get(person.pid),
            known=names,
            duck_name=self.config.duck_name,
            speech_age_s=speech_age,
        )
        action, reason = decision.action, decision.reason
        if action == N.BIND and reason == "new_name" and many > 1:
            action, reason = N.PENDING, SEVERAL_IN_VIEW  # (it cannot tell whom they mean)
        detail = {"name": decision.name, "score": round(decision.score, 2), "near": decision.near}
        if action == N.REFUSE:
            self._record(call, "refused", reason, now, "bridge", **detail)
            return
        assert decision.pid is not None
        pid = decision.pid
        if action == N.BIND:
            if reason != "confirmed":
                learner.bind_name(pid, decision.name, raw, t_s)
                self._ensure_drives(brain).introduced(now, pid)
            self._record(call, "accepted", None, now, "bridge", how=reason, pid=pid, **detail)
            return
        evidence = learner.pend_name(pid, decision.name)
        if N.promotable(
            reason, evidence
        ):  # (two separate pieces of evidence: a rename, a late name)
            learner.bind_name(pid, decision.name, raw, t_s)
            self._ensure_drives(brain).introduced(now, pid)
            self._record(call, "accepted", None, now, "bridge", how="promoted", pid=pid, **detail)
            return
        self._record(call, "pending", reason, now, "bridge", pid=pid, evidence=evidence, **detail)

    def _speech_age(self, call: QueuedCall, now: float) -> float | None:
        """Seconds since a person last spoke, by the echo gate's count and by the duck's own voice
        detection (whichever is the more recent); None if neither has heard anyone."""
        ages: list[float] = []
        if call.mark.since_human_s is not None:
            ages.append(call.mark.since_human_s)
        if self._speaking:
            ages.append(0.0)
        elif self._spans and self._spans[-1][1] is not None:
            ages.append(max(0.0, now - float(self._spans[-1][1] or 0.0)))
        return min(ages) if ages else None

    # ── remember ──────────────────────────────────────────────────────────────

    def _remember(
        self, call: QueuedCall, verdict: Verdict, now: float, brain: Any, t_s: float
    ) -> None:
        person, _many = self.perception.person_in_view(now)
        brain.learn.keep_note(
            t_s, None if person is None else person.pid, str(verdict.args["note"])
        )
        self._record(call, "accepted", None, now, "bridge")

    # ── feel → small bounded nudges ───────────────────────────────────────────

    def _feel(self, call: QueuedCall, verdict: Verdict, now: float, brain: Any) -> None:
        cfg = self.config
        drives = brain.drives
        while self._valence_moves and now - self._valence_moves[0][0] > 60.0:
            self._valence_moves.popleft()
        spent = sum(abs(d) for _t, d in self._valence_moves)
        want = float(verdict.args["valence"])
        delta = _clip(0.25 * (want - drives.valence), -cfg.valence_step, cfg.valence_step)
        room = max(0.0, cfg.valence_per_min - spent)
        delta = _clip(delta, -room, room)
        before = drives.valence
        drives.valence = _clip(before + delta, -1.0, 1.0)
        if delta:
            self._valence_moves.append((now, delta))
        talk_in = float(verdict.args["talk"])
        talk_before = self.talk.value
        dr = self._ensure_drives(brain)
        dr.feel_talk(talk_in)
        dr.on_feel(now, want)
        entry = {
            "t": round(now, 3),
            "seq": call.seq,
            "valence_in": want,
            "valence_before": round(before, 3),
            "valence_after": round(drives.valence, 3),
            "talk_in": talk_in,
            "talk_before": round(talk_before, 3),
            "talk_after": round(self.talk.value, 3),
        }
        self.nudges.append(entry)
        del self.nudges[:-100]
        self._record(call, "accepted", None, now, "bridge", delta=round(delta, 4))

    # ── lifetimes ─────────────────────────────────────────────────────────────

    def _end(self, intent: Intent, state: str, why: str, now: float) -> None:
        intent.state, intent.why = state, why
        self.finished.append(intent)
        self.log.append(
            {
                "t": round(now, 3),
                "seq": intent.seq,
                "tool": VOCALIZE if intent.kind == "reply" else ATTEND,
                "decision": state,
                "stage": "selector",
                "reason": why,
                "intent": intent.id,
            }
        )
        self.counts[f"intent:{state}"] += 1
        self.reasons[why] += 1

    def _expire(self, now: float) -> None:
        for intent in self.intents:
            if intent.active and now > intent.deadline():
                self._end(intent, "expired", LOST, now)
        self.intents = [i for i in self.intents if i.active]

    # ── the competition's input ───────────────────────────────────────────────

    def _inputs(self, now: float, brain: Any) -> None:
        extras: dict[str, float] = {}
        claims: dict[str, float] = {}  # … the part of it that is a person's answer (a reply)
        pin: tuple[str, int] | None = None
        dr = self._ensure_drives(brain)
        who = self._attrib if self._attrib is not None else dr.talker(now)
        for intent in self.intents:
            if not intent.active:
                continue
            if (
                intent.kind == "reply" and who is not None
            ):  # (whom the reply is for: who is there now)
                intent.addressee = who
                intent.known = who in brain.learn.person_names()
            weight = self._weight(intent, brain, now)
            for channel in intent.channels:
                extras[channel] = extras.get(channel, 0.0) + weight
                # (only a known person's answer claims the body; a stranger's goes through the plain
                # drive-weighted competition, LV4's shy-stranger reserve; no addressee: as before)
                if intent.kind == "reply" and (intent.addressee is None or intent.known):
                    claims[channel] = claims.get(channel, 0.0) + weight
            target = intent.target
            if target is not None and target.track and target.kind in ("person", "thing"):
                pin = (target.kind, target.track)
        wishes = self._wishes(now, brain, dr)
        for channel, value in wishes.items():
            extras[channel] = extras.get(channel, 0.0) + value
        self.last_wishes = wishes  # (set before the selector is told: a tap on it can split them)
        # (a person's answer is an event, not a drift: it may claim the body from a bout that is not
        # done — select.CLAIM_MIN. Its own wishes and the attention of an ``attend`` are not claims)
        brain.sel.set_extra(extras, claims=claims if self.config.reply_claims else None)
        if extras != self.last_extras:
            self.extras_trace.append((now, dict(extras)))
            del self.extras_trace[:-400]
        self.last_extras = extras
        if pin != self._pin:
            brain.set_live_attend(pin)
            self._pin = pin

    def _wishes(self, now: float, brain: Any, dr: LiveDrives) -> dict[str, float]:
        """The duck's own wishes toward the person it sees, as input to the competition (not the
        cloud's: nothing here comes from a call): to start a conversation (``call_out``: the wish to
        talk, toward a person it trusts) and to go to a person it is close to who has just spoken to it
        or touched it (``follow``). Nothing when nobody is in view, when it sleeps, when a person is
        speaking or when a reply is already on its way."""
        d = brain.drives
        sees = self.perception.people
        wishes: dict[str, float] = {}
        if not sees or self._speaking or dr.speech_now:
            return wishes
        if str(getattr(brain, "behaviour", "")) in _sleep_acts():
            return wishes
        pid = min(sees, key=lambda q: q.range_m if q.range_m is not None else 9.0).pid
        names = brain.sel.names
        busy = any(i.kind == "reply" and i.active for i in self.intents)
        start = 0.0 if busy else dr.initiation(pid, float(d.fear), float(d.sleepiness))
        if start > 0.0 and "call_out" in names:
            wishes["call_out"] = start
        go = dr.approach(pid, now) * (1.0 - FEAR_SUPPRESS * _clip(d.fear))
        if go > 0.0 and "follow" in names:
            wishes["follow"] = go
        return wishes

    def _weight(self, intent: Intent, brain: Any, now: float | None = None) -> float:
        """The candidate's input, from the duck's own drives (the units of a feature × weight);
        ``now``: for a reply, the urge to answer has grown the longer the person has waited."""
        d = brain.drives
        cfg = self.config
        fear = _clip(d.fear)
        rested = 1.0 - SLEEP_SUPPRESS * _clip(d.sleepiness)
        if intent.kind == "reply":
            dr = self._ensure_drives(brain)
            drive = dr.reply_drive(d.social, d.valence)
            # a person who waits for the answer makes the urge to answer grow (LV6, Fix B) ...
            waited = (
                0.0 if now is None else _clip((now - intent.born) / max(1e-3, cfg.reply_urge_s))
            )
            stranger = intent.addressee is not None and not intent.known
            share = cfg.reply_urge_stranger if stranger else 1.0
            urge = 1.0 + (cfg.reply_urge - 1.0) * waited * share
            # ... and a frightened or drowsy duck loses the input altogether (LV6, Fix B)
            calm = _clip((cfg.fear_gone - fear) / max(1e-3, cfg.fear_gone - cfg.fear_calm))
            sleepy = _clip(d.sleepiness)
            calm *= _clip((cfg.drowsy_gone - sleepy) / max(1e-3, cfg.drowsy_gone - cfg.drowsy_from))
            return (
                cfg.reply_weight
                * dr.reply_scale()
                * (0.4 + 0.6 * intent.level)
                * _clip(drive, 0.1, 1.3)
                * (1.0 - FEAR_SUPPRESS * fear)
                * calm
                * urge
                * rested
                * dr.trust(intent.addressee)
                * (1.0 if intent.addressee is None or intent.known else STRANGER_RESERVE)
            )
        interest = 0.5 + 0.5 * _clip(d.curiosity)
        return (
            cfg.attend_weight * (0.4 + 0.6 * intent.level) * interest * (1.0 - 0.8 * fear) * rested
        )

    # ── after the brain's step ────────────────────────────────────────────────

    def post(self, brain: Any, cmd: Any, t_ms: int) -> Any:
        with self._lock:
            return self._post(brain, cmd)

    def _post(self, brain: Any, cmd: Any) -> Any:
        now = self.clock()
        self._startle(brain, cmd, now)
        cmd = self._barge_stop(brain, cmd, now)
        cmd = self._emit(brain, cmd, now)
        self._note_vocal(brain, cmd, now)
        self._attention(brain, cmd, now)
        return cmd

    def _startle(self, brain: Any, cmd: Any, now: float) -> None:
        """A startle or a withdrawal begins: the person it saw is who it is wary of now."""
        reflex = None if cmd.reflex is None else str(cmd.reflex)
        began = reflex in STARTLE_REFLEXES and self._reflex_prev not in STARTLE_REFLEXES
        self._reflex_prev = reflex
        if began:
            self._ensure_drives(brain).startled(now, self._attrib)

    def _note_vocal(self, brain: Any, cmd: Any, now: float) -> None:
        if cmd.vocal_seq != self._seen_vocal_seq:
            if cmd.vocal:
                self._calling_until = now + CALL_LEN_S.get(str(cmd.vocal), 0.5)
                self._last_emit = max(self._last_emit, now)  # (a call of its own counts: no second
                # one a tick later)
                if cmd.vocal_seq != self._own_seq and cmd.behaviour == "call_out":
                    self._ensure_drives(brain).own_call()  # (its own call_out: said a little)
            self._seen_vocal_seq = cmd.vocal_seq

    def _barge_stop(self, brain: Any, cmd: Any, now: float) -> Any:
        request = self._stop
        if request is None:
            return cmd
        self._stop = None
        calling = now < self._calling_until or cmd.vocal is not None
        cancelled = 0
        for intent in self.intents:
            if intent.kind == "reply" and intent.active:
                self._end(intent, "cancelled", BARGE_IN, now)
                cancelled += 1
        self.intents = [i for i in self.intents if i.active]
        if calling:
            cmd = replace(cmd, vocal=None, vocal_seq=brain.take_vocal_seq())
            self._calling_until = now
        self._quiet_until = max(self._quiet_until, now + self.config.barge_hold_s)
        record = BargeRecord(
            request.onset_t,
            request.decided_t,
            max(0.0, request.decided_t - request.onset_t),
            max(0.0, now - request.heard_now),
            calling,
            cancelled,
        )
        self.barges.append(record)
        del self.barges[:-200]
        if calling and self.on_stop is not None:
            try:
                self.on_stop(request)
            except Exception as error:
                self.fault("on_stop", error)
        return cmd

    def _emit(self, brain: Any, cmd: Any, now: float) -> Any:
        """Voice the next due tag of a reply whose act the duck is doing right now."""
        if cmd.reflex is not None or cmd.asleep or cmd.vocal is not None:
            return cmd
        if float(brain.drives.fear) >= FEAR_MUTE:  # (frightened: nothing is voiced)
            return cmd
        if (
            self._speaking
            or now < self._quiet_until
            or now - self._last_emit < self.config.min_gap_s
        ):
            return cmd
        for intent in self.intents:
            if intent.kind != "reply" or not intent.active:
                continue
            if cmd.behaviour not in intent.channels:
                continue
            if intent.state == "competing":
                intent.state, intent.won_since = "won", now
            pending = [t for t in intent.tags if t.emitted_at is None]
            ready = next((t for t in pending if t.due <= now), None)
            if ready is None:
                continue
            said = len(intent.tags) - len(pending)
            dr = self._ensure_drives(brain)
            if said and not dr.continues(float(brain.drives.fear), float(brain.drives.sleepiness)):
                self._end(intent, "done", "brief", now)  # (it has said its piece: no more wish)
                continue
            ready.emitted_at = now
            own = float(cmd.vocal_level)
            cmd = replace(
                cmd,
                vocal=ready.tag,
                vocal_seq=brain.take_vocal_seq(),
                vocal_level=_clip(0.7 * ready.level + 0.3 * own),
            )
            self._last_emit = now
            brain.answered()  # (the reply is its answer: its own urge to answer is spent)
            self._own_seq = cmd.vocal_seq
            dr.reply_voiced(now, intent.id, intent.addressee, first_tag=not said)
            if intent.addressee is not None:
                person = brain.learn.people.get(intent.addressee)
                if person is not None:
                    person.greeted += 1
            self.voiced.append(
                {
                    "t": round(now, 3),
                    "seq": intent.seq,
                    "intent": intent.id,
                    "tag": ready.tag,
                    "first": not said,
                    "addressee": intent.addressee,
                }
            )
            del self.voiced[: -LOG_MAX // 2]
            if all(t.emitted_at is not None for t in intent.tags):
                self._end(intent, "done", "voiced", now)
            return cmd
        self.intents = [i for i in self.intents if i.active]
        return cmd

    def _attention(self, brain: Any, cmd: Any, now: float) -> None:
        """Attention paid: it has been the duck's own act for a while."""
        for intent in self.intents:
            if intent.kind != "attend" or not intent.active:
                continue
            if cmd.behaviour in intent.channels:
                if intent.won_since is None:
                    intent.state, intent.won_since = "won", now
                elif now - intent.won_since >= self.config.attend_dwell_s:
                    self._end(intent, "done", "attended", now)
            elif intent.won_since is not None:
                intent.state, intent.won_since = "competing", None
        self.intents = [i for i in self.intents if i.active]

    # ── what the page and the tests read ──────────────────────────────────────

    def barge_stats(self) -> dict[str, Any]:
        with self._lock:
            return self._barge_stats()

    def _barge_stats(self) -> dict[str, Any]:
        stopped = [b for b in self.barges if b.stopped]
        totals = sorted(b.total_s for b in stopped)
        return {
            "events": len(self.barges),
            "stopped": len(stopped),
            "total_s_max": round(totals[-1], 3) if totals else None,
            "total_s_mean": round(sum(totals) / len(totals), 3) if totals else None,
            "detect_s_max": round(max((b.detect_s for b in stopped), default=0.0), 3),
            "apply_s_max": round(max((b.apply_s for b in stopped), default=0.0), 3),
        }

    def view(self) -> dict[str, Any]:
        with self._lock:
            return self._view()

    def _view(self) -> dict[str, Any]:
        return {
            "gate": self.gate.status(),
            "counts": dict(sorted(self.counts.items())),
            "reasons": dict(sorted(self.reasons.items())),
            "intents": [i.view() for i in self.intents],
            "extras": {k: round(v, 3) for k, v in self.last_extras.items()},
            "wishes": {k: round(v, 3) for k, v in self.last_wishes.items()},
            "talk": round(self.talk.value, 3),
            "drives": None if self.drives is None else self.drives.view(),
            "speaking": self._speaking,
            "voiced": len(self.voiced),
            "barge": self._barge_stats(),
            "faults": self.faults,
            "perception": self.perception.view(),
            "vad": None if self.vad is None else self.vad.status(),
            "ear": self._ear_view(),
        }

    def _ear_view(self) -> dict[str, Any]:
        """What the duck's own ear told the cloud: name events heard, hints sent, why the others
        were not, and the last few."""
        skipped = {
            key.split(":", 1)[1]: n for key, n in sorted(self.ear_counts.items()) if ":" in key
        }
        return {
            "name_events": self.ear_counts["name_events"],
            "hints_sent": self.ear_counts["hints_sent"],
            "skipped": skipped,
            "last": list(self.ear_log)[-6:],
        }


def _ear_level(sound: Any) -> dict[str, Any]:
    level = getattr(sound, "level_db", None)
    return {} if not isinstance(level, (int, float)) else {"level_db": round(float(level), 1)}


@cache
def _sleep_acts() -> frozenset[str]:
    from duckmind.neuro import select as S

    return frozenset(S.SLEEP)


__all__ = [
    "ATTEND_CHANNELS",
    "BARGE_IN",
    "EAR_NO_LINE",
    "EAR_OFF",
    "EAR_RATE",
    "LOST",
    "NO_PERSON",
    "NO_REPLY_CHANNEL",
    "SEVERAL_IN_VIEW",
    "BargeRecord",
    "BridgeConfig",
    "Intent",
    "LiveBridge",
    "PendingTag",
    "StopRequest",
]
