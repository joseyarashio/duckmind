"""The brain's side of a command: what the duck does with what it heard.

A command (:mod:`duckmind.neuro.commands`) is a sense input, never a motor command. :class:`CommandBrain`
sits beside one :class:`~duckmind.neuro.integration.IntegratedBrain` (``brain.commands``). The web
thread only hands over what was heard (:meth:`CommandBrain.hear`); the brain's own tick takes it in
(:meth:`pre`, before the step), looks at its own state, and either

* **obeys**: the verb becomes a claim on a behaviour channel of the selector
  (:meth:`~duckmind.neuro.select.Selector.set_command`: the same units and the same rules as a person's
  live answer, on a channel of its own so the live bridge's ``set_extra`` never erases it). The
  selector still decides: a claim never takes the body from urgent business, a duck that is afraid or
  asleep, going home or docking, and a reflex never waits for it;
* **refuses**, with a reason code (``neuro.command.reason.<code>``): asleep, afraid, held, fallen,
  running away, body busy (a reflex or an emotion has it), docking, charging, battery too low, too dark,
  too young for the act, nobody seen to come to, no home known;
* or, for a trick (the five tricks of :mod:`duckmind.neuro.tricks`), **draws**: a seeded draw against
  how well it was taught (:func:`~duckmind.neuro.tricks.obey_probability`: its proficiency first, a
  scared or sleepy duck less, closeness and temperament only nudge). A pass is carried out like a
  built-in verb; a miss **reacts** so the person can see it heard: a tilt of the head (a look while
  walking) and an ``inquire`` chirp;

Built-in verbs are obeyed when the duck is awake, not afraid, not held, not fallen or in flight, and not
docking. Personality only changes how long it takes (a dreamer a half beat slower, a slow tempo
slower) and whether it chirps first (the snuggler greets, the songbird chirps). ``come`` and
``look_at_me`` use the duck's OWN sighting of the speaker (or where it believed them a moment ago),
never the world's position.

The outcome of the last command (done / refused / gave up / interrupted, why, and how long until the
body moved) is kept for the page, in catalog words. Pure and clock-injected (``clock`` returns seconds).
"""

from __future__ import annotations

import math
import random
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping

from duckmind.i18n import msg
from duckmind.neuro import contract as C
from duckmind.neuro import reflex as R
from duckmind.neuro import select as S
from duckmind.neuro import tricks as T
from duckmind.neuro.commands import Command, CommandError, is_builtin, is_trick
from duckmind.neuro.perceive import bearing_to
from duckmind.skilltrain import guard as SG

# verb -> the behaviour channel the claim goes to
BEHAVIOUR_OF: dict[str, str] = {
    "sit": "sit_down",
    "stand": "stand_up",
    "come": "follow",
    "look_at_me": "watch",
    "stop": "freeze",
    "follow": "follow",
    "go_home": "return_dock",
    # the tricks (neuro.tricks): ``shake`` is the shake-off the duck also does by itself
    "lie_down": "lie_low",
    "spin": "spin",
    "bow": "bow",
    "shake": "body_shake",
    "flip": "flip",
}
CLAIM_WEIGHT = 1.2  # the claim's input (units of a feature x its weight): above any wish of its own
LOCOMOTION_VERBS = frozenset({"come", "follow", "go_home"})
MOVING_TRICKS = frozenset({"spin", "flip"})  # tricks that turn or roll the body: walking rules apply
# what it does by itself that is a trick's act (the owner's praise within three seconds of it is shaping)
SPONTANEOUS: dict[str, str] = {
    "greet": "bow",  # the greeting bow
    "peekaboo": "lie_down",  # the lie-low of the peekaboo game
    "body_shake": "shake",  # the shake-off
    # the told channels themselves: the last moments of a trick the plan has just finished
    "bow": "bow",
    "lie_low": "lie_down",
    "spin": "spin",
    "flip": "flip",
}
ACT_GAP_S = 1.0  # an act that stops this long and starts again is a new performance
FEEDBACK_KINDS = ("praise", "treat", "scold")
# seconds: how long each plan may take before the duck gives it up (``follow`` is a duration: it is done after it)
DEADLINE_S: dict[str, float] = {
    "sit": 8.0,
    "stand": 8.0,
    "come": 15.0,
    "look_at_me": 6.0,
    "stop": 5.0,
    "follow": 20.0,
    "go_home": 120.0,
    "lie_down": 12.0,
    "spin": 10.0,
    "bow": 8.0,
    "shake": 8.0,
    "flip": 25.0,  # (the roll about 1 s, then the hold until it is upright; a slow getting-up took 11-14 s)
}
# a trick is done once it has been performed this long (s) since the body began it; ``spin`` is done
# when the duck's own odometry says it has gone once round (``brain.spun``)
PERFORM_S: dict[str, float] = {"lie_down": 1.4, "bow": 1.0, "shake": 1.0}
# the flip is the official forward roll (ROULADE-2026-10-22.md §5): asked only from what the duck itself senses
FLIP_CALM_S = 1.5  # standing, still and upright this long (the guard's settle time), no fall or getting-up in it …
FLIP_TILT_RAD = math.radians(8.0)  # … trunk within 8° of upright (its IMU) …
FLIP_GYRO_RAD_S = 0.35  # … the trunk quiet (the guard's still gyro) …
FLIP_BATTERY = 0.5  # the roll draws the servos' peak: half a pack at least
# room to roll (the real-simulator margin run, LEARN-TRICKS §4.4: by the true distance at the ask, 10/10 rolled over and up
# by itself with the first solid 0.6-0.7 m ahead and 11/11 at 0.7-0.8 m, 14/14 with the nearest solid 0.3-0.4 m away and
# 11/11 at 0.4-0.5 m; closer ahead some rolls stopped against it part way)
FLIP_AHEAD_M = 0.6  # nothing solid in the run it rolls along this far (it travels 0.44-0.73 m) …
# … in the straight path it rolls along, this far either side of its heading (the margin run measured the room along that path:
# its truth was rays on the axis and 0.12 m either side; a walking cone ±20° wide also took walls 25-35° beside the path at
# 0.4-0.6 m for "in the run" — LEARN-TRICKS §4.5) …
FLIP_PATH_HALF_M = 0.12
FLIP_AROUND_M = 0.3  # … nor this close all round (headings every 45°)
# its ToF's left / right (the nearest hit beside its ±0.10 m body corridor, up to 0.7 m aside) this near: inside the run such a
# hit can be 0.10-0.12 m beside the path (a sofa leg 0.42 m ahead read 434 mm left); counted out to 0.8 m (a longer run, tried
# and not kept) they mostly took things well beside it (5 of 37 clear open asks refused at 0.57-0.8 m) — LEARN-TRICKS §4.5
FLIP_TOF_SIDE_MM = 600
FLIP_ROLLED_TILT_RAD = 1.8  # … it has gone over once the trunk tilt passed ~100° …
FLIP_UP_TILT_RAD = math.radians(25.0)  # … and is up again at this tilt (proprio's own "up")
FLIP_UP_S = 1.0  # … and is done once it has stood upright by itself this long
FLIP_LOAD = 0.6  # its feet carry it (leg load against quiet standing = 1): lighter is a hand or the air
FLIP_START_S = 4.0  # the roll must have begun this long after the claim (or, never claimed, after the beat), else it gave up
FLIP_STILL_S = 0.25  # from the yes until the roll begins its legs are held still, re-armed every tick for this long
FLIP_SOLID_KINDS = frozenset({"thing", "toy", "person", "hand", "duck", "nest"})  # what it sees that is in the way
FLIP_SIGHT_RAD = 0.5  # … in the run ahead: this far either side of straight on
FLIP_UNSETTLING = frozenset(  # body events that restart the calm (a fall, a lift, a trip, a bump, a getting-up)
    {"fell", "righted", "righted_help", "righting_failed", "lifted", "set_down", "stumbled", "blocked", "brushed"}
)
FLIP_AROUND = tuple(math.radians(a) for a in range(0, 360, 45))
FLIP_BEGUN_RAD = 0.6  # the roll has begun once the trunk tilt passed this (rad) …
SPIN_BEGUN_RAD = 0.12  # the spin has begun once the heading (its odometry) has moved this far
SPIN_DONE_RAD = 5.4  # … and is done once it has turned this far in all (a little under 310°)
ARRIVED_M = 0.7  # ``come`` is done this close to the person (the follow act's own REACH_M is 0.5)
LOOK_HOLD_S = 1.5  # ``look_at_me`` is done after the watch has held this long
PERSON_LOST_S = 8.0  # the person not SEEN for this long (it walks to where it believes them) ends come / follow
# a person-verb heard with nobody seen: it looks about and turns this long at most (the live body
# turns ~0.33 rad/s in the search arc, measured 2026-10-23: a full turn takes ~19 s) …
LOOK_FOR_S = 20.0
LOOK_FOR_TURN_RAD = 2.0 * math.pi  # … and stops sooner once its own odometry says it turned all round
LOOK_FOR_LOST_S = 4.0  # … and this long when come / follow loses the person in the middle of the walk
MOVING_MPS = 0.05  # robotd's applied forward speed above this: the legs are going
DECISION_WAIT_S = 1.0  # how long the API waits for the brain to take a command in
# Holds that end by themselves in a beat (a stop at a wall or an edge, a flinch, a freeze): a command
# heard then is carried out after them, unless one lasts this long (live :8796, 2026-10-23: 'sit' and
# 'stop' refused ``body_busy`` at a bump stop or a short moment). Righting, being held and a startle
# keep the body.
BRIEF_HOLDS = frozenset({"bump_stop", "drop_stop", "withdraw", "freeze"})
BRIEF_HOLD_MAX_S = 3.0
LATENCY_KEEP = 200
# The personality's beat before it acts (spec §4: a command to the start of motion ≤ 1.5 s median).
# The body's own change — the head settling at HOME, robotd's rise or sit — takes ~0.9 s live
# (:8783, 2026-10-24: heard → decided ~0.1 s, decided → the rise's policy ~0.8 s), so the beat may
# take no more than the rest of that budget: 0.10 + 0.25 + 0.10 + 0.15 = 0.60 s at most.
BEAT_BASE_S = 0.10  # every duck takes this long to gather itself
BEAT_TEMPO_S = 0.25  # … a slow tempo adds up to this
BEAT_RAND_S = 0.10  # … and a little of its own
BEAT_DREAMER_S = 0.15  # … a dreamer a little more

STATES = ("heard", "looking", "doing", "done", "refused", "gave_up", "interrupted")

# the catalog words (literal keys: the i18n checks find them)
STATE_MSG = {
    "heard": msg("neuro.command.state.heard"),
    "looking": msg("neuro.command.state.looking"),
    "doing": msg("neuro.command.state.doing"),
    "done": msg("neuro.command.state.done"),
    "refused": msg("neuro.command.state.refused"),
    "gave_up": msg("neuro.command.state.gave_up"),
    "interrupted": msg("neuro.command.state.interrupted"),
}
REASON_MSG = {
    "asleep": msg("neuro.command.reason.asleep"),
    "afraid": msg("neuro.command.reason.afraid"),
    "held": msg("neuro.command.reason.held"),
    "fallen": msg("neuro.command.reason.fallen"),
    "fleeing": msg("neuro.command.reason.fleeing"),
    "body_busy": msg("neuro.command.reason.body_busy"),
    "docking": msg("neuro.command.reason.docking"),
    "charging": msg("neuro.command.reason.charging"),
    "low_battery": msg("neuro.command.reason.low_battery"),
    "too_dark": msg("neuro.command.reason.too_dark"),
    "too_young": msg("neuro.command.reason.too_young"),
    "no_one_seen": msg("neuro.command.reason.no_one_seen"),
    "no_home": msg("neuro.command.reason.no_home"),
    "not_learned": msg("neuro.command.reason.not_learned"),
    "not_running": msg("neuro.command.reason.not_running"),
    "not_yet": msg("neuro.command.reason.not_yet"),
    # the flip (the official roll): what its own senses say against it
    "not_standing": msg("neuro.command.reason.not_standing"),
    "not_steady": msg("neuro.command.reason.not_steady"),
    "battery_half": msg("neuro.command.reason.battery_half"),
    "no_room": msg("neuro.command.reason.no_room"),
    "not_floor": msg("neuro.command.reason.not_floor"),
    "rest_first": msg("neuro.command.reason.rest_first"),
    "busy": msg("neuro.command.reason.busy"),
    "timeout": msg("neuro.command.reason.timeout"),
}
ERROR_MSG = {
    "empty": msg("neuro.command.error.empty"),
    "negated": msg("neuro.command.error.negated"),
    "not_understood": msg("neuro.command.error.not_understood"),
    "unknown_verb": msg("neuro.command.error.unknown_verb"),
    "bad_source": msg("neuro.command.error.bad_source"),
    "bad_lang": msg("neuro.command.error.bad_lang"),
    "bad_confidence": msg("neuro.command.error.bad_confidence"),
    "no_target": msg("neuro.command.error.no_target"),
    "target_only_for_go_there": msg("neuro.command.error.target_only_for_go_there"),
}


def error_words(error: CommandError) -> Any:
    """The catalog words of a command that cannot be made (its reason code)."""
    return ERROR_MSG.get(error.reason, str(error))


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def _cls(posture: str) -> str:
    """``up`` (on its feet), ``down`` (sitting) or ``mid`` (changing posture, unknown)."""
    if posture in ("stand", "walk"):
        return "up"
    if posture == "sit":
        return "down"
    return "mid"


def _moving(frame: C.SensorFrame) -> bool:
    return frame.body.posture == "walk" or abs(frame.body.vx) > MOVING_MPS


@dataclass(slots=True)
class _Act:
    """One performance of a trick lately: when it last was the act in hand, whether it was told (else
    the duck's own: shaping), and whether the owner's word has been taken for it."""

    t: float
    taught: bool
    credited: bool = False
    scolded: bool = False


def _window(act: _Act) -> float:
    """How long after the act the owner's word still counts for it."""
    return T.TEACH_WINDOW_S if act.taught else T.SHAPE_WINDOW_S


def _no_feedback(trick: str | None, proficiency: float | None, reason: str) -> dict[str, Any]:
    return {
        "trick": trick, "proficiency": proficiency, "step": 0.0, "capped": False,
        "shaped": False, "counted": False, "reason": reason,
    }  # fmt: skip


@dataclass(slots=True)
class Ticket:
    """What the web thread waits on: set once the brain has decided."""

    command: Command
    t_heard: float
    event: threading.Event = field(default_factory=threading.Event)
    answer: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Plan:
    """A command the duck is carrying out."""

    ticket: Ticket
    behaviour: str
    t_heard: float
    act_at: float  # the claim starts here (the personality's beat)
    deadline: float
    start_cls: str
    start_odom: tuple[float, float]
    lead_in: str | None
    decided: float | None = None
    motion: float | None = None
    held_since: float | None = None
    seen_at: float = 0.0  # last time a person was seen or believed
    claiming: bool = False
    claimed_at: float | None = None  # when it first claimed the body for the act (s)
    searching: bool = False  # nobody seen yet: it is looking about for the person it heard
    search_until: float | None = None  # … until this (s)
    lost_until: float | None = None  # come / follow lost the person: it looks about until this (s)
    search_s: float = 0.0  # how long it has looked about for them (s)
    # the headings its body has faced while looking (rad from where it began, its odometry): the
    # search has looked all round when hi − lo spans a full turn (turning back and forth is not more)
    search_net: float = 0.0
    search_lo: float = 0.0
    search_hi: float = 0.0
    search_yaw: float | None = None  # the odometry yaw at the last look (rad)
    held_by_since: float | None = None  # a brief hold has had the body since (s)
    last_drive: float = 0.0  # the last tick this plan was driven (s)
    # a trick: where its own body began (the yaw and the measured head), and how far round it has turned
    # while the trick was the act in hand (rad, its odometry)
    start_yaw: float = 0.0
    yaw_prev: float = 0.0
    spun: float = 0.0
    performed: float = 0.0  # seconds the trick has been the act in hand, on its feet
    start_head: tuple[float, ...] = ()
    max_tilt: float = 0.0  # the flip: the most the trunk has tilted since the claim (rad, its IMU)
    up_s: float = 0.0  # … and how long it has stood upright since it went over (s)


class CommandBrain:
    """The command intake of one brain (see the module doc)."""

    def __init__(self, brain: Any, clock: Callable[[], float] = time.monotonic) -> None:
        self._brain = brain
        self.clock = clock
        self._lock = threading.RLock()
        self._pending: Ticket | None = None
        self._plan: Plan | None = None
        self._claiming = False
        self._reaction: tuple[str | None, str | None] | None = None  # (moment, vocal) to play on the next post
        self.seq = 0
        self.last: dict[str, Any] | None = None
        self._last_ticket = 0  # id of the ticket ``last`` is about
        # commands taken in, and how each ended (a command that is still being carried out is in neither)
        self.counts: dict[str, int] = {"total": 0, "done": 0, "refused": 0, "gave_up": 0, "interrupted": 0}
        self._latency: deque[float] = deque(maxlen=LATENCY_KEEP)
        self._decision: deque[float] = deque(maxlen=LATENCY_KEEP)
        self.faults = 0
        # what it did of each trick lately, taught or by itself (the owner's word is for the last one)
        self._acts: dict[str, _Act] = {}
        self.last_fault = ""
        # since when the body has stood calm (the flip is asked only from a steady stand)
        self._calm_since: float | None = None
        self._rolls: deque[float] = deque(maxlen=16)  # when it rolled lately (its own wear record)

    # -- the web thread ----------------------------------------------------------------------

    def hear(self, command: Command) -> Ticket:
        """Hand a command to the brain. It is taken in at the brain's next tick; ``ticket.event`` is set
        when it has decided (obeys, refuses or reacts) and ``ticket.answer`` says which."""
        ticket = Ticket(command, self.clock())
        with self._lock:
            older = self._pending
            self._pending = ticket
        if older is not None:
            older.answer = {"state": "interrupted", "queued": False, "reason": None}
            older.event.set()
        return ticket

    def view(self) -> dict[str, Any]:
        """What the page shows: the last command and its outcome, what is being done, the numbers."""
        with self._lock:
            plan = self._plan
            now = self.clock()
            latency = list(self._latency)
            decision = list(self._decision)
            return {
                "last": dict(self.last) if self.last is not None else None,
                "active": None
                if plan is None
                else {
                    "verb": plan.ticket.command.verb,
                    "since_s": round(max(0.0, now - plan.t_heard), 2),
                },
                "pending": self._pending is not None,
                "counts": dict(self.counts),
                "latency_ms": _stats(latency),
                "decision_ms": _stats(decision),
                "faults": self.faults,
            }

    def fault(self, where: str, error: BaseException) -> None:
        """A fault in the intake is counted and never stops the brain."""
        self.faults += 1
        self.last_fault = f"{where}: {type(error).__name__}: {error}"
        with self._lock:
            self._plan = None
            self._pending = None
        self._release()

    # -- the brain's tick --------------------------------------------------------------------

    def pre(self, brain: Any, t_ms: int, frame: C.SensorFrame) -> None:
        """Before the step: take in what was heard, watch the plan in hand, set the claim."""
        with self._lock:
            now = self.clock()
            self._watch_calm(frame, now)
            ticket, self._pending = self._pending, None
            if ticket is None and self._plan is None:
                return
            if ticket is not None:
                self._take(ticket, brain, t_ms, frame, now)
            plan = self._plan
            if plan is not None:
                self._drive(plan, brain, t_ms, frame, now)

    def post(self, brain: Any, cmd: C.MotorCommand, t_ms: int) -> C.MotorCommand:
        """After the step: the visible reaction of a trick it has no word for, and the lead-in chirp."""
        with self._lock:
            self._note_act(brain, cmd)
            reaction, self._reaction = self._reaction, None
            plan = self._plan
            vocal = None
            if reaction is not None:
                vocal = reaction[1]
            elif plan is not None and plan.lead_in is not None and self.clock() >= plan.act_at:
                vocal, plan.lead_in = plan.lead_in, None
        if vocal is not None and cmd.reflex is None and not cmd.asleep and cmd.vocal is None:
            cmd = replace(cmd, vocal=vocal, vocal_seq=brain.take_vocal_seq(), vocal_level=0.55)
        return cmd

    # -- the owner's word on a trick ---------------------------------------------------------

    def _note_act(self, brain: Any, cmd: C.MotorCommand) -> None:
        """Note what it does this tick when that is (the body of) a trick: the one it was told, or the
        same act by itself (a shake-off, a greeting bow, a peekaboo lie-low, the pre-sleep turn). The
        owner's praise within a few seconds is for it (:meth:`trick_feedback`)."""
        if cmd.asleep or cmd.reflex is not None:
            return
        now = self.clock()
        behaviour = cmd.behaviour
        plan = self._plan
        told = plan is not None and is_trick(plan.ticket.command.verb) and plan.behaviour == behaviour
        trick = plan.ticket.command.verb if told else SPONTANEOUS.get(behaviour)
        if trick is None:
            return
        act = self._acts.get(trick)
        if act is None or now - act.t > ACT_GAP_S:
            self._acts[trick] = _Act(t=now, taught=bool(told))  # a new performance
            return
        act.t = now
        act.taught = act.taught or bool(told)

    def trick_feedback(self, brain: Any, kind: str, trick: str | None = None) -> dict[str, Any]:
        """The owner's 「乖」 / a treat (``praise`` and ``treat`` are one thing for the trick) or 「不對」
        (``scold``) for a trick: the one named, else the one it did last — a trick it was told within
        :data:`~duckmind.neuro.tricks.TEACH_WINDOW_S` that has not had its word yet comes first, then the
        newest act of its own within :data:`~duckmind.neuro.tricks.SHAPE_WINDOW_S`: praise for that is
        *shaping*, half a step; a scold has the same windows. The proficiency moves in
        ``brain.learn.tricks``.

        Returns ``{trick, proficiency, step, capped, shaped, counted, reason}``: ``counted`` is whether
        the word was taken as practice; ``reason`` says why not (``no_trick``: it did none lately;
        ``not_just_done``: the named one it did not do lately; ``already_taken``: this performance
        has had its word) — a trick that was named still answers with where it stands."""
        if kind not in FEEDBACK_KINDS:
            raise ValueError(f"feedback kind {kind!r}: one of {FEEDBACK_KINDS}")
        if trick is not None and trick not in T.TRICKS:
            raise ValueError(f"trick {trick!r}: one of {T.TRICKS}")
        with self._lock:
            now = self.clock()
            scold = kind == "scold"
            chosen = trick
            if chosen is None:
                # a trick it was told and the owner has not spoken to yet comes before anything it did by
                # itself (live: a shake-off right after a slow told bow took the praise as shaping)
                pending = [
                    (act.t, name)
                    for name, act in self._acts.items()
                    if act.taught
                    and now - act.t <= _window(act)
                    and not (act.scolded if scold else act.credited)
                ]
                recent = [
                    (act.t, name)
                    for name, act in self._acts.items()
                    if now - act.t <= _window(act)
                ]
                pool = pending or recent
                if not pool:
                    return _no_feedback(None, None, "no_trick")
                chosen = max(pool)[1]
            book = brain.learn.tricks
            gd = int(brain.genome.gd)
            book.settle(gd)
            act = self._acts.get(chosen)
            if act is None or now - act.t > _window(act):
                return _no_feedback(chosen, book.proficiency(chosen), "not_just_done")
            if scold:
                if act.scolded:
                    return _no_feedback(chosen, book.proficiency(chosen), "already_taken")
                act.scolded = True
                fb = book.scold(chosen, gd)
                return {
                    "trick": chosen, "proficiency": fb.proficiency, "step": round(fb.delta, 4),
                    "capped": False, "shaped": False, "counted": False, "reason": None,
                }  # fmt: skip
            if act.credited:
                return _no_feedback(chosen, book.proficiency(chosen), "already_taken")
            act.credited = True
            fb = book.praise(chosen, gd, shaped=not act.taught)
            return {
                "trick": chosen, "proficiency": fb.proficiency, "step": round(fb.delta, 4),
                "capped": fb.capped, "shaped": fb.shaped, "counted": True, "reason": None,
            }  # fmt: skip

    # -- taking a command in -----------------------------------------------------------------

    def _take(
        self, ticket: Ticket, brain: Any, t_ms: int, frame: C.SensorFrame, now: float
    ) -> None:
        command = ticket.command
        if self._plan is not None:  # a new command ends the old one
            self._finish(self._plan, "interrupted", now, brain=brain, reason=None)
        self._release()
        self.seq += 1
        refusal = self._refusal(command, brain, t_ms, frame)
        if refusal is not None:
            self._record(ticket, "refused", now, reason=refusal)
            self._answer(ticket, "refused", refusal)
            return
        if is_trick(command.verb):
            self._take_trick(ticket, brain, t_ms, frame, now)
            return
        if not is_builtin(command.verb) or command.verb == "go_there":
            self._record(ticket, "refused", now, reason="not_yet")
            self._answer(ticket, "refused", "not_yet")
            return
        verb = command.verb
        behaviour = BEHAVIOUR_OF[verb]
        # already as asked: nothing to do (and no latency to measure)
        if self._already(verb, brain, frame):
            self._record(ticket, "done", now, decided=0, motion=0, done=0, already=True)
            self._answer(ticket, "done", None, already=True)
            return
        person = self._person(brain, t_ms, frame)
        beat, lead_in = self._personality(brain)
        # a person-verb heard with nobody seen or believed: it looks about for them first (the head
        # sweeps, the body turns in place), and goes on with the verb as soon as its own eyes call one
        searching = person is None and verb in ("come", "follow", "look_at_me")
        # (a seated duck cannot turn in place: it stands first, then sweeps and turns)
        search_act = "look_around" if _cls(frame.body.posture) == "up" else "stand_up"
        plan = Plan(
            ticket=ticket,
            behaviour=search_act if searching else behaviour,
            t_heard=ticket.t_heard,
            act_at=now + beat,
            deadline=now + beat + DEADLINE_S[verb] + (LOOK_FOR_S if searching else 0.0),
            start_cls=_cls(frame.body.posture),
            start_odom=(float(frame.body.odom[0]), float(frame.body.odom[1])),
            lead_in=lead_in,
            seen_at=now,
            searching=searching,
            search_until=now + LOOK_FOR_S if searching else None,
            last_drive=now,
        )
        self._plan = plan
        if searching:
            brain.look_for(t_ms, LOOK_FOR_S)
        self._record(ticket, "looking" if searching else "heard", now)
        self._answer(ticket, "heard", None, queued=True, beat_s=round(beat, 2))

    def _take_trick(
        self, ticket: Ticket, brain: Any, t_ms: int, frame: C.SensorFrame, now: float
    ) -> None:
        """A trick: the seeded draw against how well it was taught (module doc). A miss is the reaction
        a trick it has no word for always had (a tilt of the head and an ``inquire`` chirp); a pass is
        carried out like a built-in verb."""
        verb = ticket.command.verb
        chance, proficiency = self._obey_chance(verb, brain, t_ms)
        draw = random.Random(f"trick:{int(brain.seed)}:{self.seq}").random()
        if draw >= chance:
            self._react_unknown(brain)
            self._record(
                ticket, "refused", now, reason="not_learned", reaction="tilt",
                p_obey=round(chance, 3), proficiency=round(proficiency, 3),
            )  # fmt: skip
            self._answer(ticket, "refused", "not_learned", reaction="tilt")
            return
        beat, lead_in = self._personality(brain)
        odom = frame.body.odom
        plan = Plan(
            ticket=ticket,
            behaviour=BEHAVIOUR_OF[verb],
            t_heard=ticket.t_heard,
            act_at=now + beat,
            deadline=now + beat + DEADLINE_S[verb],
            start_cls=_cls(frame.body.posture),
            start_odom=(float(odom[0]), float(odom[1])),
            lead_in=lead_in,
            seen_at=now,
            last_drive=now,
            start_yaw=float(odom[2]),
            yaw_prev=float(odom[2]),
            start_head=tuple(float(v) for v in frame.body.head),
        )
        self._plan = plan
        self._acts.pop(verb, None)  # (a new performance, told: the owner's word is for this one)
        self._record(
            ticket, "heard", now, p_obey=round(chance, 3), proficiency=round(proficiency, 3)
        )
        self._answer(ticket, "heard", None, queued=True, beat_s=round(beat, 2))

    def _obey_chance(self, verb: str, brain: Any, t_ms: int) -> tuple[float, float]:
        """(the chance it does the trick now, its proficiency): from this duck's own state only."""
        book = brain.learn.tricks
        book.settle(int(brain.genome.gd))
        proficiency = book.proficiency(verb)
        chance = T.obey_probability(
            proficiency,
            fear=float(brain.drives.fear),
            sleepiness=float(brain.drives.sleepiness),
            bond=self._bond(brain),
            primary=str(getattr(brain.genome, "primary", "")),
        )
        return chance, proficiency

    def _bond(self, brain: Any) -> float:
        """How close it feels to the person it saw most recently (0 when nobody lately): its own
        record of them (:attr:`~duckmind.neuro.learn.PersonRecord.bond`), never the world's."""
        now_s = brain._clock_s()
        seen = [
            p for p in brain.learn.people.values() if now_s - float(p.seen_t) <= _person_known_s()
        ]
        return max((float(p.bond) for p in seen), default=0.0)

    def _trick_refusal(self, verb: str, brain: Any, frame: C.SensorFrame) -> str | None:
        """What stops a taught trick before the draw (its own state): not charging; a turn or a roll not on
        a low pack or in the dark; and the flip's own list (:meth:`_flip_refusal`)."""
        if frame.power.charging or (brain._recharging and _at_charge_place(brain, frame)):
            return "charging"
        if verb in MOVING_TRICKS:
            if frame.power.battery < 0.3:
                return "low_battery"
            if frame.camera.usable and frame.camera.light == "black":
                return "too_dark"
        if verb == "flip":
            return self._flip_refusal(brain, frame, self.clock())
        return None

    def _watch_calm(self, frame: C.SensorFrame, now: float) -> None:
        """Since when the body has stood still and upright (its IMU and odometry; a fall, a getting-up, a
        lift, a trip or a bump starts it over): the flip is asked only from a steady stand."""
        body = frame.body
        gyro = math.sqrt(sum(g * g for g in body.gyro))
        steady = (
            body.posture == "stand"
            and not body.fallen
            and body.tilt < FLIP_TILT_RAD
            and gyro < FLIP_GYRO_RAD_S
            and abs(body.vx) <= MOVING_MPS
            and abs(body.vyaw) <= 0.1
            and not any(e in frame.events for e in FLIP_UNSETTLING)
        )
        if not steady:
            self._calm_since = None
        elif self._calm_since is None:
            self._calm_since = now

    def _flip_refusal(self, brain: Any, frame: C.SensorFrame, now: float) -> str | None:
        """Why it will not roll now, from its own senses only (ROULADE-2026-10-22.md §5; the body's skill
        guard repeats the posture, tilt, settle and wear rules): the body is wired for it, standing on the
        floor and steady (calm for a moment, upright, its feet carrying it, no fall or getting-up lately),
        a pack that is at least half full, room to roll along by its own map, its ToF and its eyes, and not
        rolled too often. None: it will."""
        if not getattr(brain, "can_roll", False):
            return "not_yet"
        body, power = frame.body, frame.power
        cls = _cls(body.posture)
        if cls == "down":
            return "not_standing"
        if power.on_dock:
            return "not_floor"
        reflex = getattr(brain._reflex, "active", None)
        calm = self._calm_since
        if (
            cls != "up"
            or body.posture != "stand"
            or reflex is not None
            or calm is None
            or now - calm < FLIP_CALM_S
            or (body.leg_load is not None and body.leg_load < FLIP_LOAD)
        ):
            return "not_steady"
        if frame.tof.present and frame.tof.drop_zones >= 4:
            return "not_floor"
        if power.battery < FLIP_BATTERY:
            return "battery_half"
        if not self._room_to_roll(brain, frame):
            return "no_room"
        rule = SG.DEFAULT_RULES[SG.ROLL_ENTRY]
        hour = [t for t in self._rolls if now - t < SG.WINDOW_S]
        if hour and now - hour[-1] < rule.min_gap_s:
            return "rest_first"
        if rule.per_hour is not None and len(hour) >= rule.per_hour:
            return "rest_first"
        return None

    def _room_to_roll(self, brain: Any, frame: C.SensorFrame) -> bool:
        """Nothing solid in the FLIP_AHEAD_M it rolls along (its straight path, FLIP_PATH_HALF_M either side) and none within
        FLIP_AROUND_M all round, by the walls and things it has seen or bumped into (its boundary memory — not the side brushes'
        guesses — and its lasting map, in its own odometry), its ToF and what its eye sees now."""
        odom = frame.body.odom
        bound = brain.bound
        if bound.clearances(odom, [0.0], half=FLIP_PATH_HALF_M, brushes=False)[0] < FLIP_AHEAD_M:
            return False
        if min(bound.clearances(odom, list(FLIP_AROUND), brushes=False)) < FLIP_AROUND_M:
            return False
        lastmap = brain.lastmap
        if lastmap.cfg.enabled:
            if lastmap.clearance(odom, 0.0, max_m=FLIP_AHEAD_M, half_m=FLIP_PATH_HALF_M) < FLIP_AHEAD_M:
                return False
            if min(lastmap.clearance(odom, h, max_m=FLIP_AROUND_M) for h in FLIP_AROUND) < FLIP_AROUND_M:
                return False
        tof = frame.tof
        if tof.present:
            if tof.touch:
                return False
            if tof.ahead_mm is not None and tof.ahead_mm <= FLIP_AHEAD_M * 1000.0:
                return False
            for side in (tof.left_mm, tof.right_mm):
                if side is not None and side <= FLIP_TOF_SIDE_MM:
                    return False
        for sight in frame.camera.sightings:
            if sight.kind not in FLIP_SOLID_KINDS or sight.range_m is None:
                continue
            if sight.range_m <= FLIP_AROUND_M:
                return False
            if sight.range_m <= FLIP_AHEAD_M and abs(sight.bearing) <= FLIP_SIGHT_RAD:
                return False
        return True

    def _already(self, verb: str, brain: Any, frame: C.SensorFrame) -> bool:
        cls = _cls(frame.body.posture)
        if verb == "sit":
            return cls == "down"
        if verb == "stand":
            return cls == "up"
        if verb == "stop":
            return not _moving(frame) and brain.behaviour not in ("wander", "follow", "return_dock")
        if verb == "go_home":
            return bool(frame.power.on_dock or frame.power.charging or brain.dock.mode == "home")
        return False

    def _refusal(self, command: Command, brain: Any, t_ms: int, frame: C.SensorFrame) -> str | None:
        """Why the duck will not do this now, from its own state (None: it will)."""
        verb = command.verb
        reflex = brain._reflex
        active = getattr(reflex, "active", None)
        if brain.behaviour in S.SLEEP:
            return "asleep"
        if getattr(reflex, "lifted", False) or "lifted" in frame.events:
            return "held"
        if frame.body.fallen or getattr(reflex, "fallen", False):
            return "fallen"
        if float(brain.drives.fear) >= S.CLAIM_FEAR_MAX:
            return "afraid"
        if brain.behaviour in ("flee", "retreat") or brain._startled_s is not None:
            return "fleeing"
        if active is not None and active.reflex in R.HOLDS and active.reflex not in BRIEF_HOLDS:
            return "body_busy"
        mode = brain.dock.mode
        if verb == "go_home":
            if mode == "leaving":
                return "docking"
        elif mode in ("docking", "leaving"):
            return "docking"
        if (
            verb != "go_home"
            and brain.behaviour == "return_dock"
            and brain._feat[S.F["homing"]] >= S.HOMING_RELEASE
        ):
            return "low_battery"  # walking home to charge: the claim never takes the body from that
        if verb == "go_there" or not (is_trick(verb) or is_builtin(verb)):
            return None
        behaviour = BEHAVIOUR_OF[verb]
        if behaviour not in brain.sel.names:
            return "too_young"
        if is_trick(verb):
            return self._trick_refusal(verb, brain, frame)
        if verb in LOCOMOTION_VERBS or verb == "stand":
            if frame.power.battery < 0.3 and verb != "stand":
                return "low_battery"
            if verb != "go_home" and brain._recharging and _at_charge_place(brain, frame):
                return "charging"
        if verb in LOCOMOTION_VERBS and frame.camera.usable and frame.camera.light == "black":
            return "too_dark"
        if verb in ("come", "follow", "look_at_me") and self._person(brain, t_ms, frame) is None:
            # nobody seen or believed: it looks about for them first (``LOOK_FOR_S``) — unless its own
            # eyes cannot give it a person at all, when looking would be for nothing
            if not frame.camera.usable or frame.camera.light == "black":
                return "no_one_seen"
        if verb == "go_home" and not self._home_known(brain, frame):
            return "no_home"
        return None

    def _home_known(self, brain: Any, frame: C.SensorFrame) -> bool:
        if frame.power.on_dock or frame.power.charging or brain.dock.mode == "home":
            return True
        if any(s.kind == "nest" for s in frame.camera.sightings):
            return True
        odom = frame.body.odom
        here = (odom[0] * 1000.0, odom[1] * 1000.0, odom[2])
        return brain.learn.dock is not None or brain.dock.dock_bearing(here, stale_ok=True) is not None

    def _person(
        self, brain: Any, t_ms: int, frame: C.SensorFrame
    ) -> tuple[float, float, bool] | None:
        """(bearing rad, range m, seen now) of the person the duck sees, or believes it saw a moment
        ago (its own odometry); None when nobody. The duck's own sense, never the world's."""
        best = None
        for sight in frame.camera.sightings:
            if sight.kind == "person" and (
                best is None or (sight.range_m or 9.0) < (best.range_m or 9.0)
            ):
                best = sight
        if best is not None:
            return float(best.bearing), float(best.range_m if best.range_m is not None else 1.5), True
        believed = brain._person
        t0 = brain._t0 if brain._t0 is not None else t_ms
        if believed is not None and (t_ms - t0) / 1000.0 - believed[2] < _person_known_s():
            bearing, rng = bearing_to(frame.body.odom, believed[0], believed[1])
            return float(bearing), float(rng), False
        return None

    def _personality(self, brain: Any) -> tuple[float, str | None]:
        """A beat before it acts and what it says first. A slow tempo and a dreamer take longer; the
        snuggler greets and the songbird chirps first. Nothing else about the duck changes whether it
        obeys a built-in verb. The beat stays inside the spec's latency budget (:data:`BEAT_BASE_S`):
        the personality may change how fast it is, never past the 1.5 s the spec asks."""
        genome = brain.genome
        tempo = _clip(float(genome.axes.get("tempo", 0.5)))
        rng = random.Random(int(brain.seed) * 7919 + self.seq)
        beat = BEAT_BASE_S + BEAT_TEMPO_S * (1.0 - tempo) + BEAT_RAND_S * rng.random()
        primary = str(getattr(genome, "primary", ""))
        if primary == "dreamer":
            beat += BEAT_DREAMER_S
        lead_in = {"snuggler": "greet", "songbird": "chirp"}.get(primary)
        return beat, lead_in

    # -- carrying it out ---------------------------------------------------------------------

    def _drive(self, plan: Plan, brain: Any, t_ms: int, frame: C.SensorFrame, now: float) -> None:
        command = plan.ticket.command
        verb = command.verb
        dt = max(0.0, now - plan.last_drive)
        plan.last_drive = now
        person = self._person(brain, t_ms, frame)
        if person is not None and person[2]:
            plan.seen_at = now  # (seen now, not merely believed)
        # an interruption: it is not the duck's to carry on
        stop = self._interrupted(plan, brain, frame, now)
        if stop is not None:
            self._finish(plan, "interrupted", now, brain=brain, reason=stop)
            return
        if plan.searching:
            # nobody seen yet: it looks about for them (the head sweeps, the body turns in place)
            plan.search_s += dt
            yaw = float(frame.body.odom[2])
            if plan.search_yaw is not None:
                plan.search_net += C.wrap_angle(yaw - plan.search_yaw)
                plan.search_lo = min(plan.search_lo, plan.search_net)
                plan.search_hi = max(plan.search_hi, plan.search_net)
            plan.search_yaw = yaw
            self._mark_looking(plan, now)
            if person is not None:
                # its own eyes have called one: the verb goes on
                plan.searching = False
                plan.behaviour = BEHAVIOUR_OF[verb]
                plan.decided = None
                plan.held_since = None
                brain.look_for_stop()
                self._mark_doing(plan, now)
            elif plan.behaviour == "stand_up" and _cls(frame.body.posture) == "up":
                plan.behaviour = "look_around"  # (up: now it sweeps its head and turns in place)
            elif now >= plan.search_until or plan.search_hi - plan.search_lo >= LOOK_FOR_TURN_RAD:
                # not found in the bound: it says so, and tilts its head and chirps that it heard
                self._finish(plan, "refused", now, brain=brain, reason="no_one_seen")
                self._react_unknown(brain)
                return
        elif verb in ("come", "follow"):
            # the person it walks to is not seen: it looks about for them a moment before giving up
            if person is not None and person[2]:
                if plan.lost_until is not None:
                    plan.lost_until = None
                    plan.behaviour = BEHAVIOUR_OF[verb]
                    brain.look_for_stop()
                    self._mark_doing(plan, now)
            elif plan.lost_until is None and now - plan.seen_at > PERSON_LOST_S:
                plan.lost_until = now + LOOK_FOR_LOST_S
                plan.behaviour = "look_around"
                brain.look_for(t_ms, LOOK_FOR_LOST_S)
                self._mark_looking(plan, now)
            elif plan.lost_until is not None:
                plan.search_s += dt
                self._mark_looking(plan, now)
                if now >= plan.lost_until:
                    self._finish(plan, "gave_up", now, brain=brain, reason="no_one_seen")
                    return
        # what it did: chosen (the behaviour is the one asked for) and moving
        if not plan.searching and plan.lost_until is None:
            if plan.decided is None and brain.behaviour == plan.behaviour:
                plan.decided = now
            if is_trick(verb):
                self._track_trick(plan, brain, frame, dt)
            if plan.motion is None and plan.decided is not None:
                if self._moved(verb, plan, frame):
                    plan.motion = now
                    if verb == "flip":
                        self._rolls.append(now)  # (its own wear record: the rest it asks of itself)
        done = self._done(plan, brain, frame, person, now)
        if done:
            if verb == "spin" and callable(getattr(brain, "spin_made", None)):
                brain.spin_made()  # (the turn is made: the act stops here, not after a bout that began again mid-turn)
            self._finish(plan, "done", now, brain=brain)
            return
        if verb == "flip" and plan.motion is None:
            # from the yes until the roll begins it stands still (no step, no turn, no arc: a walk or a turn would
            # unsettle it, and the body's guard starts the roll only from a settled stand)
            if callable(getattr(brain, "keep_still", None)):
                brain.keep_still(t_ms + int(FLIP_STILL_S * 1000))
            since = plan.claimed_at if plan.claimed_at is not None else plan.act_at
            if now >= since + FLIP_START_S:
                # not steady by its own senses all that time, or the body did not take the roll up (its guard said
                # wait, or a refusal of its own): it says it is not steady, and the claim is let go
                self._finish(plan, "gave_up", now, brain=brain, reason="not_steady")
                return
        if now >= plan.deadline:
            reason = "timeout" if plan.decided is not None else "busy"
            self._finish(plan, "gave_up", now, brain=brain, reason=reason)
            return
        if now >= plan.act_at:
            if verb == "follow" and person is not None and person[1] <= ARRIVED_M:
                self._release()  # close to them: it rests there and follows again when they walk off
            elif verb == "flip" and not plan.claiming and (self._calm_since is None or now - self._calm_since < FLIP_CALM_S):
                return  # the roll is asked only from a steady stand by its own senses: it waits, standing still
            else:
                self._claim(brain, plan)
            if not plan.searching and plan.lost_until is None:
                self._mark_doing(plan, now)

    def _claim(self, brain: Any, plan: Plan) -> None:
        brain.cmd.ask(plan.behaviour, CLAIM_WEIGHT)  # (the channel is written once per tick, in the step)
        self._claiming = True
        plan.claiming = True
        if plan.claimed_at is None:
            plan.claimed_at = self.clock()

    def _release(self) -> None:
        if self._claiming:
            try:
                self._brain.cmd.drop_ask()
            except Exception:  # (a brain that is being replaced)
                pass
            self._claiming = False

    def _track_trick(self, plan: Plan, brain: Any, frame: C.SensorFrame, dt: float) -> None:
        """A trick in hand: how long it has been performed on its feet, and how far round the body has
        turned (signed, by its own odometry) while the trick was the act."""
        yaw = float(frame.body.odom[2])
        turned = C.wrap_angle(yaw - plan.yaw_prev)
        plan.yaw_prev = yaw
        if plan.ticket.command.verb == "flip":
            # the roll: how far the trunk has tipped (its IMU), and how long it has stood upright again since
            # it went over — by itself: it is done when it is up, not when it was asked
            tilt = float(frame.body.tilt)
            plan.max_tilt = max(plan.max_tilt, tilt)
            up = (
                plan.max_tilt >= FLIP_ROLLED_TILT_RAD
                and frame.body.posture == "stand"
                and not frame.body.fallen
                and tilt < FLIP_UP_TILT_RAD
            )
            plan.up_s = plan.up_s + dt if up else 0.0
            return
        if brain.behaviour == plan.behaviour and _cls(frame.body.posture) == "up":
            plan.performed += dt
            plan.spun += turned

    def _moved(self, verb: str, plan: Plan, frame: C.SensorFrame) -> bool:
        """The body has begun what was asked (its own measure, not the brain's choice).

        A sit or a stand begins when the body says it is between postures (``transition``: robotd's
        rise or sit machine runs, the legs are folding), not when the class has changed — that is the
        change's END (live :8783, 2026-10-24: a sit-down's class changed 1.9 s after the command while
        the body had begun it a second before). The class change stays as the fallback for a body that
        never says ``transition``. A turn trick begins when the odometry's heading has moved; the head
        tricks (a bow, a lie-low, a shake) begin when the act is chosen, as a look does."""
        posture = frame.body.posture
        cls = _cls(posture)
        if verb in ("sit", "stand"):
            return posture == "transition" or cls != plan.start_cls
        if verb == "spin":
            return abs(C.wrap_angle(float(frame.body.odom[2]) - plan.start_yaw)) >= SPIN_BEGUN_RAD
        if verb == "flip":
            return plan.max_tilt >= FLIP_BEGUN_RAD  # (the trunk has begun to go over)
        if is_trick(verb):
            return cls != "down"  # (on its feet or rising to them: a seated duck gets up first)
        if verb in LOCOMOTION_VERBS:
            odom = frame.body.odom
            gone = math.hypot(odom[0] - plan.start_odom[0], odom[1] - plan.start_odom[1])
            return _moving(frame) or gone > 0.05
        if verb == "stop":
            return not _moving(frame)
        return True  # (a look: the head turns the moment the act is chosen)

    def _interrupted(self, plan: Plan, brain: Any, frame: C.SensorFrame, now: float) -> str | None:
        reflex = brain._reflex
        active = getattr(reflex, "active", None)
        if brain.behaviour in S.SLEEP:
            return "asleep"
        if getattr(reflex, "lifted", False) or "lifted" in frame.events:
            return "held"
        if frame.body.fallen or getattr(reflex, "fallen", False):
            return "fallen"
        if float(brain.drives.fear) >= S.CLAIM_FEAR_MAX or brain._startled_s is not None:
            return "afraid"
        if active is None or active.reflex not in R.HOLDS or active.reflex == "freeze":
            plan.held_by_since = None
            return None
        if active.reflex not in BRIEF_HOLDS:
            return "body_busy"
        if plan.held_by_since is None:
            plan.held_by_since = now
        return "body_busy" if now - plan.held_by_since >= BRIEF_HOLD_MAX_S else None

    def _done(
        self,
        plan: Plan,
        brain: Any,
        frame: C.SensorFrame,
        person: tuple[float, float, bool] | None,
        now: float,
    ) -> bool:
        verb = plan.ticket.command.verb
        cls = _cls(frame.body.posture)
        if verb == "sit":
            return cls == "down"
        if verb == "stand":
            return cls == "up"
        if verb == "come":
            return person is not None and person[1] <= ARRIVED_M
        if verb == "follow":
            return now >= plan.act_at + DEADLINE_S["follow"] - 0.001
        if verb == "look_at_me":
            if plan.decided is not None and brain.behaviour == "watch":
                if plan.held_since is None:
                    plan.held_since = now
                return now - plan.held_since >= LOOK_HOLD_S
            plan.held_since = None
            return False
        if verb == "stop":
            return brain.behaviour == "freeze" and not _moving(frame)
        if verb == "go_home":
            return bool(frame.power.on_dock or frame.power.charging or brain.dock.mode == "home")
        if is_trick(verb):
            if verb == "spin":
                return abs(plan.spun) >= SPIN_DONE_RAD
            if verb == "flip":
                return plan.motion is not None and plan.up_s >= FLIP_UP_S
            return plan.motion is not None and plan.performed >= PERFORM_S.get(verb, 1.0)
        return False

    # -- telemetry ---------------------------------------------------------------------------

    def _mark_doing(self, plan: Plan, now: float) -> None:
        last = self.last
        if last is not None and last.get("seq") == self.seq and last["state"] in ("heard", "looking"):
            self._set_state(last, "doing")

    def _mark_looking(self, plan: Plan, now: float) -> None:
        """It is looking about for the person it heard: the page says so, with how long it has looked."""
        last = self.last
        if last is not None and last.get("seq") == self.seq:
            self._set_state(last, "looking")
            last["search_ms"] = round(plan.search_s * 1000.0)


    def _finish(
        self,
        plan: Plan,
        state: str,
        now: float,
        *,
        brain: Any,
        reason: str | None = None,
    ) -> None:
        ticket = plan.ticket
        self._plan = None
        self._release()
        brain.look_for_stop()
        if plan.ticket.command.verb == "flip" and callable(getattr(brain, "act_over", None)):
            # the roll's bout is used up with its command (done, given up or cut short): what the duck does next is
            # chosen by its own wants, not left to a fright of noise size (the real simulator: a 1 s panic run after it)
            brain.act_over("flip")
        last = self.last
        if last is None or self._last_ticket != id(ticket):
            return
        heard = plan.t_heard
        last["decided_ms"] = None if plan.decided is None else round((plan.decided - heard) * 1000.0)
        last["motion_ms"] = None if plan.motion is None else round((plan.motion - heard) * 1000.0)
        last["done_ms"] = round((now - heard) * 1000.0)
        if plan.search_s > 0.0:
            last["search_ms"] = round(plan.search_s * 1000.0)
        self._set_state(last, state, reason)
        if plan.motion is not None and state in ("done", "doing"):
            self._latency.append((plan.motion - heard) * 1000.0)
        if plan.decided is not None:
            self._decision.append((plan.decided - heard) * 1000.0)

    def _record(
        self, ticket: Ticket, state: str, now: float, *, reason: str | None = None, **extra: Any
    ) -> None:
        command = ticket.command
        record: dict[str, Any] = {
            "seq": self.seq,
            "verb": command.verb,
            "source": command.source,
            "lang": command.lang,
            "who": command.who,
            "text": command.text,
            "t_heard": round(ticket.t_heard, 3),
            "decided_ms": extra.pop("decided", None),
            "motion_ms": extra.pop("motion", None),
            "done_ms": extra.pop("done", None),
            "state": state,
            "state_zh": STATE_MSG[state],
            "reason": reason,
            "reason_zh": REASON_MSG.get(reason) if reason else None,
        }
        record.update(extra)
        self.last = record
        self._last_ticket = id(ticket)
        self.counts["total"] += 1
        if state in ("done", "refused"):
            self.counts[state] += 1

    def _set_state(self, last: dict[str, Any], state: str, reason: str | None = None) -> None:
        if last["state"] == state and reason is None:
            return
        previous = last["state"]
        last["state"] = state
        last["state_zh"] = STATE_MSG[state]
        last["reason"] = reason
        last["reason_zh"] = REASON_MSG.get(reason) if reason else None
        if state in ("gave_up", "interrupted", "done") and previous not in ("gave_up", "interrupted", "done"):
            self.counts[state] += 1

    def _answer(
        self, ticket: Ticket, state: str, reason: str | None, **extra: Any
    ) -> None:
        answer: dict[str, Any] = {"state": state, "state_zh": STATE_MSG[state], "reason": reason}
        if reason:
            answer["reason_zh"] = REASON_MSG.get(reason)
        answer.update(extra)
        ticket.answer = answer
        ticket.event.set()

    def _react_unknown(self, brain: Any) -> None:
        """A trick it has not been taught: it heard, it does not know it. A tilt of the head (a glance
        while it walks: a tilt is not a walking moment) and an ``inquire`` chirp, from the next tick."""
        moment = "glance" if brain._stepping else "tilt"
        if not brain.fire_moment(moment):
            moment = None
        self._reaction = (moment, "inquire")


def _at_charge_place(brain: Any, frame: C.SensorFrame) -> bool:
    from duckmind.neuro.brain import _at_charge_place as at

    return bool(at(frame, brain.dock))


def _person_known_s() -> float:
    from duckmind.neuro.brain import PERSON_KNOWN_S

    return float(PERSON_KNOWN_S)


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "median": None, "p90": None}
    ordered = sorted(values)
    p90 = ordered[min(len(ordered) - 1, int(math.ceil(0.9 * len(ordered))) - 1)]
    return {"n": len(values), "median": round(statistics.median(values)), "p90": round(p90)}


__all__ = [
    "BEHAVIOUR_OF",
    "CommandBrain",
    "CommandError",
    "ERROR_MSG",
    "REASON_MSG",
    "STATES",
    "STATE_MSG",
    "Ticket",
    "error_words",
]
