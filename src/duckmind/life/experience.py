"""What the neural duck lived, as Duck Core's inputs (growth integration G2).

:class:`Experiences` watches the neural duck the way its own senses do — the sensor frame the brain
was given, the command it answered with and its telemetry, once per brain tick — and turns what
really happened to it into ``bridge-v1`` events (:mod:`duckmind.duckcore.bridge`): integers and a
few enum words, stamped with the second they happened on the duck's one clock. It decides nothing
about growth: that is Duck Core's, by its own rules. It reads nothing the duck's own senses could
not tell (a touch it felt, a sound it heard, a person its camera saw, a fall its body reported, a
sleep it chose), never the page's button that caused it, and never a clock but the one the caller
passes in.

Pure and deterministic: the same frames, commands and seconds give the same events. The server
(:mod:`duckmind.simweb.life_host`) and the headless harness
(``tools/growth/care_harness.py``) use this one class.

What each detector reads, and the rule it keeps:

* **touch / poke** — ``frame.touch``: a contact episode by a hand (flickers under 0.4 s are one
  contact, a contact under 0.1 s is none). A contact by a *thing* (a slipper, a shoe or a toy
  knocking against it) or by a person's body (a leg walking past) is not a hand: ``touch.by`` says
  who, and only the kit hand (a pat, a poke) — or an unlabelled contact, which the synthetic world
  and old recordings give — is ever told to Duck Core. The duck's reflexes still feel the object;
  Duck Core has no event for "bumped into a thing", so its growth never hears of it. A hand's
  contact that is sharp (pressure 0.6+), short (1.5 s or less) and not a stroke — or on
  its side (left, right: a caress never lands there) — is a *poke*; anything else is a *touch*, with
  where, how long, whether it was asleep, whether it leaned into the hand (its own ``lean_in``
  reflex) and what it was doing by itself in the three seconds before. Emitted when the contact ends.
* **lift** — the body's ``lifted`` event (a contact within 1.5 s of a lift or a set-down is the
  hand that did it, not a caress), *once the legs bear it out*: a duck in the air carries nothing
  on its legs, and keeps carrying nothing while it is held (seated in the hand ≈ 0.4 of a quiet
  stand's load, seated on a floor ≈ 0.7 or more), so the claim is told when the legs have been
  light for :data:`~duckmind.neuro.liftcheck.LIFT_LIGHT_MS` within :data:`~duckmind.neuro.liftcheck.LIFT_CONFIRM_MS` of it and no set-down, fall or
  righting came first. A claim the legs do not bear out — a duck standing still on the floor whose
  legs went light, or one standing on a sill or a rug with its legs working hard and its trunk
  quiet — is counted in ``ignored`` as ``lift_false`` and not told. The body's own reflex (it stops
  and sits on the claim) is not touched; only the growth input is filtered. A frame with no leg
  load (the synthetic world, old recordings) has nothing to check the claim by: it is told at once.
  A duck that lies down *on purpose* (the play-dead emotion: it sits, rolls onto its back and lifts its legs, then gets up
  by itself) is no fall and no lift: the body link says ``lay_down`` instead of ``fell`` / ``lifted`` for it, nothing is
  told, and ``ignored['lie_down_self']`` counts the lie-downs (docs/design/neural-duck/PLAY-DEAD-2026-10-11.md). If it
  then does NOT get up by itself, the ordinary fall (and its righting) begins from there and is told as any fall.
  **fall** — ``fell`` and how it ended (``righted`` within
  six seconds is easy, later hard; ``righting_failed`` is stuck; ``righted_help`` is helped, or hard
  after six seconds), with the spot in the duck's own odometry (millimetres).
* **hand** — a ``hand`` sighting (the camera's class), slow or fast by its looming; or a fast
  ``looming`` one, with a person here or something close at its face for the ToF, *while the duck
  itself is quiet* (:data:`QUIET_MS` without walking, turning, a stumble, a fall or a righting, and a
  still trunk): a thing coming at the face of a duck that is itself walking, turning or tumbling is its
  own motion, not a hand (``ignored['hand_false']``). Only a ``hand`` (or ``person``) sighting means
  someone is here; a ToF-near looming one does not (``ignored['company_false']`` counts the hand events
  that would have made it so). **name / voice / hello / noise** — the
  sound classes (``name`` with the TV playing in the last five seconds says so). A *noise* the
  duck made itself — it bumped a thing, a thing it had pushed fell, a thing hit it — is not told: it
  is counted in ``ignored`` as ``noise_self``. The sound sense says what made it (``sound.by``, the
  world's label in the simulator); with no label (a real microphone, old recordings) the rule a
  real duck can apply is used: a loud sound within :data:`SELF_NOISE_S` of its own bump (a thing or
  a person's body against it, or ``blocked`` / ``brushed`` / ``stumbled`` / ``fell``) is its own.
  Its reflexes (startle, orient) hear it as before; only the growth input is filtered.
* **company** — a person in view or heard within the last ten seconds, once for each calendar minute,
  moving or resting; **visit** — a person newly here after ten minutes without one.
* **sleep** — ``cmd.asleep`` ending after a minute or more (the night's when it began between 22:00
  and 07:00). **play** — a bout of the ball (``play_ball``, ``carry_toy``) of a minute or more.
* **dock** — a stay on the charger; **feed** — a charged pack put in (the battery jumping up while it
  is not charging). **voiced** — the calls it made (the tags of one call or song).
  **explore** — a place or a thing it had not met (the learner's counts rising), or, with the lasting map on, a new 1 m
  square of its home its own map has seen clear (``learned['home']['places']`` rising; MAP-EXPLORE-2026-10-12). A place
  found within ``HELD_MS`` of a lift, a set-down, a lie-down on purpose or a fall, or while one lasts, is not told
  (``ignored['explore_held']``): the map writes nothing then, and nothing then is a real find.
* **teach_trick** (bridge-v2, :mod:`duckmind.duckcore.bridge_v2`) — a trick it was told to do (what its own command
  intake heard, ``learned['command']['last']``) ended ``done``, ``refused`` or ``gave_up``, once per command.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from duckmind.duckcore import bridge, bridge_v2, content
from duckmind.neuro import contract as C
from duckmind.neuro.liftcheck import LIFT_DROP_EVENTS, LiftWitness

TPE = timezone(timedelta(hours=8))

TOUCH_MERGE_MS = 400  # a flicker shorter than this is one contact (contract.py, 2026-09-27)
TOUCH_MIN_MS = 100
POKE_PRESSURE = 0.6
POKE_STROKE = 0.25
POKE_MAX_MS = 1500
# a hard touch on its sides is a poke whatever it slides like (measured live, :8996)
POKE_SIDES = ("left", "right")
# who may be a "touch" or a "poke" to growth: the kit hand ("hand") and a contact whose maker the
# sense does not tell ("": the synthetic world, recordings made before TouchSense had ``by``)
HAND_BY = ("", "hand")
# a contact this close to a lift or a set-down is the hand that did it, not a caress
HANDLING_S = 1.5
NEAR_FACE_MM = 300  # something this close in front of the ToF is at its face
DOING_WINDOW_MS = 3000  # engine.PRAISE_WINDOW
HAND_FAST_LOOM = 1.2  # a hand coming at the face faster than this (1/s) is a fast hand
HAND_MEASURE_MS = 500
HAND_NEW_GAP_MS = 1500
SOUND_GAP_MS = 1500  # a sound of the same class this soon after is the same sound
# an unlabelled loud sound this soon after the duck's own bump (a contact that is not a hand, or a
# body event of walking into something) is the duck's own noise, not a noise in the house
SELF_NOISE_S = 1.0
OWN_BUMP_EVENTS = ("blocked", "brushed", "stumbled", "fell")
LAY_DOWN_GAP_MS = 3000  # a 'lay_down' every tick of a lie-down is one lie-down in ``ignored``
# a looming sighting is a hand only while the duck itself has been quiet this long: no walking,
# turning, stumble, bump, fall or righting, a trunk that is still
QUIET_MS = 1500
QUIET_GYRO = 0.5  # rad/s: the trunk is still below this
QUIET_VX = 0.02
QUIET_VYAW = 0.1
BUSY_EVENTS = (
    "blocked",
    "brushed",
    "stumbled",
    "fell",
    "righted",
    "righted_help",
    "righting_failed",
    "lifted",
    "set_down",
)
TV_MEMORY_MS = 5000
RIGHTED_EASY_MS = 6000
FALL_GIVE_UP_S = 120
GAP_S = 30  # the calendar jumped this much between two ticks: it was not watched meanwhile
PERSON_HERE_S = 10
VISIT_AFTER_S = 600
MIN_SLEEP_S = 60
MIN_PLAY_S = 60
PLAY_GAP_S = 20
VOICE_GROUP_MS = 1500
EXPLORE_EVERY_S = 300
HELD_MS = 10_000  # (map places) nothing it finds this soon after a lift, a set-down, a lie-down or a fall is told
FEED_JUMP = 0.25
PLAY_BEHAVIOURS = ("play_ball", "carry_toy")
TOUCH_WHERE = {
    "head": "head",
    "beak": "beak",
    "back": "back",
    "tail": "back",
    "left": "body",
    "right": "body",
    "body": "body",
}


@dataclass
class _Touch:
    start_ms: int
    site: str
    asleep: bool
    doing: str
    last_ms: int
    by: str = ""
    max_pressure: float = 0.0
    stroke_sum: float = 0.0
    n: int = 0
    settled: bool = False


@dataclass
class _Lift:
    """A lift the body claimed and the legs have not yet borne out."""

    t_s: int
    witness: LiftWitness


@dataclass
class _Fall:
    t_ms: int
    t_s: int
    x_mm: int
    y_mm: int
    resolved: bool = False


@dataclass
class Experiences:
    """One neural duck's experiences, a tick at a time (see the module doc)."""

    tz: timezone = TPE
    _touch: _Touch | None = None
    _object_touching: bool = False
    _verbs: deque[tuple[int, str]] = field(default_factory=lambda: deque(maxlen=64))
    _lifted_ms: int = -(10**9)
    _lay_down_ms: int = -(10**9)  # the last 'lay_down' event (a play dead: lying on purpose)
    _handled_ms: deque[int] = field(default_factory=lambda: deque(maxlen=16))  # lifts, set-downs
    _fall: _Fall | None = None
    _lift: _Lift | None = None
    _hand_track: int = -1
    _hand_seen_ms: int = -(10**9)
    _hand_start_ms: int = -(10**9)
    _hand_max: float = 0.0
    _hand_sent: bool = True
    _hand_real: bool = False  # the episode had a camera 'hand' sighting (not just a looming one)
    _hand_blocked_ms: int = -(10**9)
    _busy_ms: int = -(10**9)  # the last time the duck itself was walking, turning, tumbling
    _sound_at: dict[str, int] = field(default_factory=dict)
    _own_bump_ms: int = -(10**9)  # the last time something not a hand was against it, or it bumped
    _tv_ms: int = -(10**9)
    _person_ms: int = -(10**9)
    _person_s: int | None = None  # the last calendar second a person was here
    _started_s: int | None = None
    _last_minute: int | None = None
    _minute_moving: int = 0
    _minute_ticks: int = 0
    _minute_person: bool = False
    _asleep_since: int | None = None
    _play_since: int | None = None
    _play_last: int | None = None
    _dock_since: int | None = None
    _battery: float | None = None
    _vocal_seq: int = -1
    _vocal_tags: list[str] = field(default_factory=list)
    _vocal_ms: int = -(10**9)
    _vocal_company: bool = False  # someone was here when it called (else nobody could answer)
    _explored: int | None = None
    _explore_s: int = -(10**9)
    _home_places: int | None = (
        None  # the 1 m squares of its home its own map had seen at the last look (lasting map on)
    )
    places_found: int = 0  # map places told this watch (the 成長 view)
    _command: tuple[Any, Any] | None = None  # the last command whose end was looked at (seq, heard)
    _command_primed: bool = False
    _last_s: int | None = None
    sent: dict[str, int] = field(default_factory=dict)
    # contacts the duck felt that were not a hand (a thing, a person's leg): counted, never told
    ignored: dict[str, int] = field(default_factory=dict)
    # the last contacts it felt, as numbers (site, ms, strongest press, mean stroke, what it was
    # told as): for the status page and for calibrating the poke against the touch sense
    contacts: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=24))

    # ── the tick ──────────────────────────────────────────────────────────────────────────

    def observe(
        self,
        t_s: int,
        frame: C.SensorFrame,
        cmd: C.MotorCommand | None = None,
        tel: C.BrainTelemetry | None = None,
    ) -> list[dict[str, Any]]:
        """The events of this tick (usually none). ``t_s``: the duck's calendar second."""
        out: list[dict[str, Any]] = []
        t_ms = frame.t_ms
        if self._last_s is not None and t_s - self._last_s > GAP_S:
            self._unwatched(self._last_s, out)  # (a computer that slept, a clock that jumped)
        self._last_s = t_s
        asleep = bool(cmd.asleep) if cmd is not None else False
        self._behaviour(t_ms, cmd)
        self._motion(frame, cmd)
        self._touches(t_s, frame, cmd, tel, asleep, out)
        self._body(t_s, frame, out)
        self._sounds(t_s, frame, asleep, out)
        self._company(t_s, frame, cmd, out)
        self._hand(t_s, frame, out)
        self._sleep(t_s, asleep, out)
        self._play(t_s, cmd, out)
        self._power(t_s, frame, out)
        self._voice(t_s, cmd, out)
        self._explore(t_s, t_ms, tel, out)
        self._tricks(t_s, tel, out)
        for event in out:
            self.sent[event["k"]] = self.sent.get(event["k"], 0) + 1
        return out

    def flush(self, t_s: int) -> list[dict[str, Any]]:
        """The end of the watch (a clean stop, or the end of a window): what is still open, as of
        ``t_s``. The watch carries on after it as a new one (nothing seen before counts again)."""
        out: list[dict[str, Any]] = []
        self._unwatched(t_s, out)
        return out

    def _unwatched(self, t_s: int, out: list[dict[str, Any]]) -> None:
        """Time passed that nobody watched (the duck was off, or the clock jumped): what was open
        ends where the watch did, and the next tick starts every detector afresh. A sleep or a
        stay at the charger counts only up to there."""
        self._flush_voice(t_s, out)
        if self._touch is not None:
            self._close_touch(t_s, out)
        self._object_touching = False
        self._own_bump_ms = -(10**9)
        if self._asleep_since is not None:
            began, self._asleep_since = self._asleep_since, None
            self._sleep_ended(began, t_s, out)
        if self._play_since is not None and self._play_last is not None:
            seconds = self._play_last - self._play_since
            if seconds >= MIN_PLAY_S:
                out.append({"k": "play", "t": t_s, "minutes": max(1, min(600, seconds // 60))})
        self._play_since = self._play_last = None
        if self._dock_since is not None:
            minutes = (t_s - self._dock_since) // 60
            out.append({"k": "dock", "t": t_s, "minutes": int(min(1440, minutes))})
            self._dock_since = None
        self._fall = None
        self._lift = None  # (a claim not yet borne out when the watch ended: not told)
        self._hand_sent = True
        self._person_ms = -(10**9)
        self._last_minute = None
        self._minute_moving = self._minute_ticks = 0
        self._minute_person = False
        self._battery = None
        self._vocal_seq = -1

    # ── behaviour ─────────────────────────────────────────────────────────────────────────

    def _behaviour(self, t_ms: int, cmd: C.MotorCommand | None) -> None:
        if cmd is None:
            return
        spec = C.BEHAVIOURS.get(cmd.behaviour)
        verb = None if spec is None else spec.verb
        if verb in bridge.DOING:
            self._verbs.append((t_ms, str(verb)))

    def _doing(self, t_ms: int) -> str:
        """What it chose to do by itself in the last three seconds (the newest such act)."""
        if self._verbs and t_ms - self._verbs[-1][0] <= DOING_WINDOW_MS:
            return self._verbs[-1][1]
        return ""

    # ── touch, poke ───────────────────────────────────────────────────────────────────────

    def _touches(
        self,
        t_s: int,
        frame: C.SensorFrame,
        cmd: C.MotorCommand | None,
        tel: C.BrainTelemetry | None,
        asleep: bool,
        out: list[dict[str, Any]],
    ) -> None:
        t_ms = frame.t_ms
        touch = frame.touch
        episode = self._touch
        if touch.site is not None and touch.by not in HAND_BY:
            # a thing (or a person's leg) against it: not a hand, so no touch and no poke (it also
            # ends a hand's episode that was open: the frame no longer shows the hand)
            if not self._object_touching:
                self.ignored[touch.by] = self.ignored.get(touch.by, 0) + 1
            self._object_touching = True
            self._own_bump_ms = frame.t_ms
            touch = C.TouchSense()
        else:
            self._object_touching = False
        if touch.site is not None:
            if episode is None:
                episode = self._touch = _Touch(
                    start_ms=t_ms,
                    site=touch.site,
                    by=touch.by,
                    asleep=asleep,
                    doing=self._doing(t_ms),
                    last_ms=t_ms,
                )
            episode.last_ms = t_ms
            episode.max_pressure = max(episode.max_pressure, float(touch.pressure))
            episode.stroke_sum += float(touch.stroke)
            episode.n += 1
            if tel is not None and (
                tel.reflex == "lean_in"
                or any(
                    r.reflex == "lean_in" and r.t_ms >= episode.start_ms - 200 for r in tel.reflexes
                )
            ):
                episode.settled = True
        elif episode is not None and t_ms - episode.last_ms > TOUCH_MERGE_MS:
            self._close_touch(t_s, out)

    def _close_touch(self, t_s: int, out: list[dict[str, Any]]) -> None:
        episode, self._touch = self._touch, None
        if episode is None:
            return
        duration = episode.last_ms - episode.start_ms
        if duration < TOUCH_MIN_MS or episode.n == 0:
            return
        stroke = episode.stroke_sum / episode.n
        handled = any(
            abs(episode.start_ms - mark) <= HANDLING_S * 1000
            or abs(episode.last_ms - mark) <= HANDLING_S * 1000
            for mark in self._handled_ms
        )
        sharp = (
            episode.max_pressure >= POKE_PRESSURE
            and duration <= POKE_MAX_MS
            and (stroke < POKE_STROKE or episode.site in POKE_SIDES)
        )
        told = "held" if handled else "poke" if sharp else "touch"
        self.contacts.append(
            {
                "site": episode.site,
                "by": episode.by,
                "ms": duration,
                "press": round(episode.max_pressure, 2),
                "stroke": round(stroke, 2),
                "told": told,
            }
        )
        if handled:
            return  # (the hand that lifted it or put it down: the lift is the event)
        if sharp:
            out.append({"k": "poke", "t": t_s, "ms": duration})
            return
        out.append(
            {
                "k": "touch",
                "t": t_s,
                "where": TOUCH_WHERE.get(episode.site, "body"),
                "ms": duration,
                "asleep": int(episode.asleep),
                "settled": int(episode.settled),
                "doing": episode.doing,
            }
        )

    # ── the body: lift, fall ──────────────────────────────────────────────────────────────

    def _body(self, t_s: int, frame: C.SensorFrame, out: list[dict[str, Any]]) -> None:
        t_ms = frame.t_ms
        events = frame.events
        if "lay_down" in events and t_ms - self._lay_down_ms > LAY_DOWN_GAP_MS:
            # it lay down on purpose (play dead): the body link already holds back the fall and lift claims of that;
            # counted here so the page can show that they were seen and not told
            self.ignored["lie_down_self"] = self.ignored.get("lie_down_self", 0) + 1
        if "lay_down" in events:
            self._lay_down_ms = t_ms
            self._lift = None  # (a lift claim made while it lay down on purpose is not borne out)
            if self._fall is None:
                events = tuple(e for e in events if e not in ("fell", "lifted"))
        if any(e in events for e in OWN_BUMP_EVENTS):
            self._own_bump_ms = t_ms
        if "lifted" in events and t_ms - self._lifted_ms > 2000:
            self._lifted_ms = t_ms
            if frame.body.leg_load is None:
                out.append({"k": "lift", "t": t_s})  # (nothing to check the claim by)
                self._handled_ms.append(t_ms)
            else:
                self._lift = _Lift(t_s=t_s, witness=LiftWitness(t_ms))
        if "set_down" in events:
            self._handled_ms.append(t_ms)
        self._lift_borne_out(frame, out)
        fall = self._fall
        if "fell" in events and fall is None:
            ox, oy, _ = frame.body.odom
            self._fall = fall = _Fall(
                t_ms=t_ms,
                t_s=t_s,
                x_mm=int(round(ox * 1000)),
                y_mm=int(round(oy * 1000)),
            )
        if fall is None:
            return
        elapsed = t_ms - fall.t_ms
        outcome: str | None = None
        if "righting_failed" in events and not fall.resolved:
            fall.resolved = True
            outcome = "stuck"
        elif "righted" in events:
            outcome = None if fall.resolved else ("easy" if elapsed <= RIGHTED_EASY_MS else "hard")
            fall.resolved = True
            self._fall = None
        elif "righted_help" in events:
            outcome = (
                None if fall.resolved else ("helped" if elapsed <= RIGHTED_EASY_MS else "hard")
            )
            fall.resolved = True
            self._fall = None
        elif not fall.resolved and t_s - fall.t_s >= FALL_GIVE_UP_S:
            fall.resolved = True
            outcome = "stuck"
        if outcome is not None:
            out.append(
                {"k": "fall", "t": t_s, "x_mm": fall.x_mm, "y_mm": fall.y_mm, "outcome": outcome}
            )

    def _lift_borne_out(self, frame: C.SensorFrame, out: list[dict[str, Any]]) -> None:
        """The body claimed a lift: tell it once the legs bear it out, drop it if the floor or a
        fall ends it first or they never do (``ignored['lift_false']``)."""
        lift = self._lift
        if lift is None:
            return
        # (a duck that is falling, lying or righting is not being carried; the floor ended the claim)
        ended = (
            frame.body.fallen
            or (self._fall is not None and not self._fall.resolved)
            or any(e in frame.events for e in LIFT_DROP_EVENTS)
        )
        verdict = lift.witness.step(frame.t_ms, frame.body.leg_load, ended=ended)
        if verdict == "no":
            self._lift = None
            self.ignored["lift_false"] = self.ignored.get("lift_false", 0) + 1
        elif verdict == "yes":
            self._lift = None
            self._handled_ms.append(lift.witness.t_ms)  # the hand that lifted it
            out.append({"k": "lift", "t": lift.t_s})

    # ── sounds ────────────────────────────────────────────────────────────────────────────

    def _sounds(
        self, t_s: int, frame: C.SensorFrame, asleep: bool, out: list[dict[str, Any]]
    ) -> None:
        t_ms = frame.t_ms
        kind = frame.sound.kind
        if kind is None or frame.sound.level_db <= 40.0:
            return
        if kind == "tv":
            self._tv_ms = t_ms
            return
        if kind in ("bang", "clap", "door", "noise") and self._own_noise(t_ms, frame.sound.by):
            # the duck's own impact: its reflexes heard it, Duck Core is not told (once per sound)
            mine = f"{kind}#self"
            gone = self._sound_at.get(mine)
            self._sound_at[mine] = t_ms
            if gone is None or t_ms - gone > SOUND_GAP_MS:
                self.ignored["noise_self"] = self.ignored.get("noise_self", 0) + 1
            return
        last = self._sound_at.get(kind)
        self._sound_at[kind] = t_ms
        if last is not None and t_ms - last <= SOUND_GAP_MS:
            return  # (the same sound still going on)
        if kind == "name":
            out.append(
                {
                    "k": "name",
                    "t": t_s,
                    "tv": int(t_ms - self._tv_ms <= TV_MEMORY_MS),
                    "asleep": int(asleep),
                }
            )
        elif kind == "voice":
            self._person_ms = t_ms
            out.append({"k": "voice", "t": t_s, "asleep": int(asleep)})
        elif kind == "hello":
            self._person_ms = t_ms
            out.append({"k": "hello", "t": t_s, "asleep": int(asleep)})
        elif kind in ("bang", "clap", "door", "noise"):
            out.append({"k": "noise", "t": t_s, "what": kind if kind != "noise" else "other"})

    def _own_noise(self, t_ms: int, by: str) -> bool:
        """Did the duck make this sound itself? The world's label when it gives one ("self"); else
        the rule a real robot can use: it bumped something a moment before it (or is still)."""
        if by:
            return by == "self"
        return t_ms - self._own_bump_ms <= SELF_NOISE_S * 1000

    # ── a hand ────────────────────────────────────────────────────────────────────────────

    def _motion(self, frame: C.SensorFrame, cmd: C.MotorCommand | None) -> None:
        """When the duck itself last moved: its own senses say (the applied walk, the trunk's rate,
        the posture changing, a bump, a fall) and what it chose to do."""
        body = frame.body
        spec = None if cmd is None else C.BEHAVIOURS.get(cmd.behaviour)
        gyro = math.sqrt(sum(g * g for g in body.gyro))
        if (
            abs(body.vx) > QUIET_VX
            or abs(body.vyaw) > QUIET_VYAW
            or gyro > QUIET_GYRO
            or body.fallen
            or body.posture in ("walk", "unknown", "transition")
            or (spec is not None and spec.locomotes)
            or any(e in frame.events for e in BUSY_EVENTS)
        ):
            self._busy_ms = frame.t_ms

    def _hand(self, t_s: int, frame: C.SensorFrame, out: list[dict[str, Any]]) -> None:
        t_ms = frame.t_ms
        # A hand: the camera's hand near its face (slow or fast by its looming) — and, with a person
        # here or something at its face for the ToF, anything that comes at its face fast (the eye
        # cannot tell a hand from a ball by looming alone) — but only while the duck itself is
        # quiet: walking up to a thing makes it loom, and a duck that turns or tumbles sees the
        # world loom. Only the camera's own ``hand`` means someone is here.
        someone = t_ms - self._person_ms <= PERSON_HERE_S * 1000
        ahead = frame.tof.ahead_mm
        at_face = frame.tof.present and ahead is not None and ahead <= NEAR_FACE_MM
        quiet = t_ms - self._busy_ms >= QUIET_MS
        real = [s for s in frame.camera.sightings if s.kind == "hand"]
        loomed = [
            s
            for s in frame.camera.sightings
            if s.kind == "looming" and (someone or at_face) and s.looming >= HAND_FAST_LOOM
        ]
        if loomed and not quiet:
            if t_ms - self._hand_blocked_ms > HAND_NEW_GAP_MS:
                self.ignored["hand_false"] = self.ignored.get("hand_false", 0) + 1
            self._hand_blocked_ms = t_ms
            loomed = []
        hands = real + loomed
        if hands:
            best = max(hands, key=lambda s: s.looming)
            if t_ms - self._hand_seen_ms > HAND_NEW_GAP_MS or best.track != self._hand_track:
                self._hand_track = best.track
                self._hand_start_ms = t_ms
                self._hand_max = 0.0
                self._hand_sent = False
                self._hand_real = False
            self._hand_seen_ms = t_ms
            self._hand_max = max(self._hand_max, float(best.looming))
            self._hand_real = self._hand_real or bool(real)
        if not self._hand_sent and (
            (hands and t_ms - self._hand_start_ms >= HAND_MEASURE_MS)
            or (not hands and t_ms - self._hand_seen_ms > TOUCH_MERGE_MS)
        ):
            self._hand_sent = True
            if self._hand_real:
                self._person_ms = t_ms
            else:
                self.ignored["company_false"] = self.ignored.get("company_false", 0) + 1
            speed = "fast" if self._hand_max >= HAND_FAST_LOOM else "slow"
            out.append({"k": "hand", "t": t_s, "speed": speed})

    # ── company, a visit ──────────────────────────────────────────────────────────────────

    def _company(
        self,
        t_s: int,
        frame: C.SensorFrame,
        cmd: C.MotorCommand | None,
        out: list[dict[str, Any]],
    ) -> None:
        t_ms = frame.t_ms
        if self._started_s is None:
            self._started_s = t_s
        if any(s.kind == "person" for s in frame.camera.sightings):
            since = t_s - max(self._started_s, self._person_s or self._started_s)
            if since > VISIT_AFTER_S:
                out.append({"k": "visit", "t": t_s})
            self._person_ms = t_ms
        here = t_ms - self._person_ms <= PERSON_HERE_S * 1000  # (seen, or heard talking)
        if here:
            self._person_s = t_s
        minute = t_s // 60
        spec = None if cmd is None else C.BEHAVIOURS.get(cmd.behaviour)
        moving = (spec is not None and spec.locomotes) or frame.body.posture == "walk"
        if self._last_minute is None:
            self._last_minute = minute
        if minute != self._last_minute:
            if self._minute_person and self._minute_ticks:
                out.append(
                    {
                        "k": "company",
                        "t": (self._last_minute + 1) * 60,
                        "moving": int(self._minute_moving * 2 >= self._minute_ticks),
                    }
                )
            self._last_minute = minute
            self._minute_moving = self._minute_ticks = 0
            self._minute_person = False
        self._minute_ticks += 1
        self._minute_moving += int(moving)
        self._minute_person = self._minute_person or here

    # ── sleep, play ───────────────────────────────────────────────────────────────────────

    def _sleep(self, t_s: int, asleep: bool, out: list[dict[str, Any]]) -> None:
        if asleep and self._asleep_since is None:
            self._asleep_since = t_s
        elif not asleep and self._asleep_since is not None:
            began, self._asleep_since = self._asleep_since, None
            self._sleep_ended(began, t_s, out)

    def _sleep_ended(self, began: int, t_s: int, out: list[dict[str, Any]]) -> None:
        seconds = t_s - began
        if seconds >= MIN_SLEEP_S:
            hour = datetime.fromtimestamp(began, self.tz).hour
            out.append(
                {
                    "k": "sleep",
                    "t": t_s,
                    "minutes": max(1, min(1440, seconds // 60)),
                    "night": int(hour >= 22 or hour < 7),
                }
            )

    def _play(self, t_s: int, cmd: C.MotorCommand | None, out: list[dict[str, Any]]) -> None:
        playing = cmd is not None and cmd.behaviour in PLAY_BEHAVIOURS
        if playing:
            if self._play_since is None:
                self._play_since = t_s
            self._play_last = t_s
        elif self._play_since is not None and self._play_last is not None:
            if t_s - self._play_last >= PLAY_GAP_S:
                seconds = self._play_last - self._play_since
                self._play_since = self._play_last = None
                if seconds >= MIN_PLAY_S:
                    out.append({"k": "play", "t": t_s, "minutes": max(1, min(600, seconds // 60))})

    # ── power ─────────────────────────────────────────────────────────────────────────────

    def _power(self, t_s: int, frame: C.SensorFrame, out: list[dict[str, Any]]) -> None:
        power = frame.power
        if power.on_dock and self._dock_since is None:
            self._dock_since = t_s
        elif not power.on_dock and self._dock_since is not None:
            minutes = (t_s - self._dock_since) // 60
            self._dock_since = None
            out.append({"k": "dock", "t": t_s, "minutes": int(min(1440, minutes))})
        battery = float(power.battery)
        if (
            self._battery is not None
            and battery - self._battery >= FEED_JUMP
            and not power.charging
        ):
            out.append({"k": "feed", "t": t_s})
        self._battery = battery

    # ── what it said ──────────────────────────────────────────────────────────────────────

    def _voice(self, t_s: int, cmd: C.MotorCommand | None, out: list[dict[str, Any]]) -> None:
        if cmd is None:
            return
        t_ms = cmd.t_ms
        if cmd.vocal_seq != self._vocal_seq:
            if self._vocal_seq >= 0 and cmd.vocal in content.TAGS:
                if self._vocal_tags and t_ms - self._vocal_ms > VOICE_GROUP_MS:
                    self._flush_voice(t_s, out)
                self._vocal_tags.append(str(cmd.vocal))
                self._vocal_ms = t_ms
                self._vocal_company = self._vocal_company or (
                    t_ms - self._person_ms <= PERSON_HERE_S * 1000
                )
            self._vocal_seq = cmd.vocal_seq
        if self._vocal_tags and t_ms - self._vocal_ms > VOICE_GROUP_MS:
            self._flush_voice(t_s, out)

    def _flush_voice(self, t_s: int, out: list[dict[str, Any]]) -> None:
        """What it said, once it has finished: Duck Core is told only what was said in company (it
        learns a word or a phrase from someone answering or praising it; a call to an empty room
        teaches it nothing and would fill its record)."""
        if self._vocal_tags and self._vocal_company:
            out.append({"k": "voiced", "t": t_s, "tags": self._vocal_tags[:8]})
        self._vocal_tags = []
        self._vocal_company = False

    # ── somewhere new ─────────────────────────────────────────────────────────────────────

    def _held(self, t_ms: int) -> bool:
        """A lift, a set-down, a lie-down on purpose or a fall lasts or was a moment ago."""
        if self._fall is not None or self._lift is not None:
            return True
        recent = max(self._lifted_ms, self._lay_down_ms, max(self._handled_ms, default=-(10**9)))
        return t_ms - recent < HELD_MS

    def _explore(
        self, t_s: int, t_ms: int, tel: C.BrainTelemetry | None, out: list[dict[str, Any]]
    ) -> None:
        if tel is None:
            return
        learned = tel.learned
        known = len(learned.get("places") or ()) + len(learned.get("things") or ())
        home = learned.get("home")
        places = home.get("places") if isinstance(home, dict) else None
        found = 0
        if isinstance(places, int):
            if self._home_places is None:
                self._home_places = places  # (what it knew when the watch began is not a find)
            elif places > self._home_places:
                if self._held(t_ms):
                    self.ignored["explore_held"] = self.ignored.get("explore_held", 0) + 1
                else:
                    found = places - self._home_places
                self._home_places = places
        if self._explored is None:
            self._explored = known
            return
        if (known > self._explored or found) and t_s - self._explore_s >= EXPLORE_EVERY_S:
            self._explore_s = t_s
            self.places_found += found
            out.append({"k": "explore", "t": t_s})
        self._explored = max(self._explored, known)

    # ── tricks ────────────────────────────────────────────────────────────────────────────

    def _tricks(self, t_s: int, tel: C.BrainTelemetry | None, out: list[dict[str, Any]]) -> None:
        """A trick it was told to do, ended (bridge-v2 ``teach_trick``): the brain's own command
        telemetry (``learned['command']['last']``, what its command intake heard and how that
        ended) says ``done``, ``refused`` or ``gave_up`` for one of the five tricks. Each command is
        told once (by its number and the moment it was heard); one interrupted by another is not a
        lesson. The first look of a watch only notes where it stands (a command that ended before
        the watch began was told, or not, by the watch it ended in)."""
        learned = getattr(tel, "learned", None)
        command = learned.get("command") if isinstance(learned, Mapping) else None
        last = command.get("last") if isinstance(command, Mapping) else None
        if not isinstance(last, Mapping):
            self._command_primed = True
            return
        ident = (last.get("seq"), last.get("t_heard"))
        state = last.get("state")
        ended = state in bridge_v2.TEACH_OUTCOMES
        if not self._command_primed:
            self._command_primed = True
            if ended:
                self._command = ident
            return
        if not ended or ident == self._command:
            return
        self._command = ident
        verb = last.get("verb")
        if verb not in bridge_v2.TRICKS:
            return
        done_ms = last.get("done_ms")
        ms = int(done_ms) if isinstance(done_ms, (int, float)) and done_ms > 0 else 0
        out.append(
            {
                "k": "teach_trick",
                "t": t_s,
                "trick": str(verb),
                "ms": min(ms, bridge_v2.MS_MAX),
                "outcome": str(state),
            }
        )


def stats(experiences: Experiences) -> dict[str, int]:
    """How many events of each kind were given (the harness's and the server's count)."""
    return dict(sorted(experiences.sent.items()))


__all__ = ["Experiences", "TPE", "stats"]
