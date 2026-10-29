"""What the reflexes' events mean to the rest of the brain (neuro:brain).

The reflex arcs themselves live in :mod:`duckmind.neuro.reflex` (neuro:motor's module: onset
detection, latency, the response's size and shape, the head overlay, halting and veering, the
fall episode, being held). A reflex is fast and local; its *consequences* reach the slower
systems through this module:

* **drives** — a startle is a threat impulse (fear, arousal), a fall is pain, a stroke leaned
  into is a good thing and contact (comfort, the social need);
* **selection** — short-lived after-effects (leaky traces) that feed the competition: just
  startled (→ freeze, flee, tuck_away), a loud sound (→ stand up alert, look around), petted
  (→ snuggle, melt), just righted (→ head shake, proud puff), put down (→ look around), a sound
  it could not place (→ look around, perk and tilt), a cue it has learned means something good;
* **learning** — places get worse where it was scared, bumped or fell; the owner's rule: a spot
  where it had three or more *hard* falls is avoided (:mod:`duckmind.neuro.learn`); cues
  (Rescorla–Wagner) learn what usually follows them;
* **steering** — learned danger spots bend the walk away.

Every after-effect is a leaky trace (exact for any tick); nothing here decides a behaviour or a
duration. Pure: only the SensorFrame, the reflex output and the brain's own memory; never
:mod:`duckmind.world`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from duckmind.neuro.contract import SensorFrame, wrap_angle
from duckmind.neuro.genome import Genome
from duckmind.neuro.learn import Learner
from duckmind.neuro.perceive import TOUCH_BEARING

# Trace time constants (s): how long an after-effect keeps feeding the competition.
TRACE_TAU_S = {
    "after_fall": 12.0,
    "woken": 4.0,
    "double_take": 2.5,
    "heard": 1.5,
    "sound_nodir": 2.0,
    "good_cue": 4.0,
    "bad_cue": 4.0,
    "turn_away": 1.5,
    "obstacle": 0.8,
    "avoid_spot": 1.0,
    "recovering": 45.0,
    "rhythm_slow": 4.0,
    "rhythm_fast": 4.0,
}
TRACE_TAU_DEFAULT_S = 6.0
TRACE_FLOOR = 0.02
# The ToF distances (mm) at which the path starts bending / a wall counts as in the way.
TOF_NEAR_MM = 650
TOF_STOP_MM = 300  # reflex.py stops straight walking here
# A sound kind whose onset is sharp enough to put a duck on alert (stand up and look).
SHARP_SOUNDS = frozenset({"clap", "bang", "door", "noise"})
BUMP_REFRACTORY_S = 1.5
# A real fright gets the alarm call (fid:motor 2026-09-28: a startle is the head's snap, the
# beak's gape, the legs stopping — and a call); a mild startle stays silent. Once per scare.
STARTLE_ALARM = 0.5
ALARM_REFRACTORY_S = 4.0
# Miller's rhythm rule (Pekin ducklings; Miller 1983): a train of sharp sounds at 0.8–1.8 per
# second makes a duckling freeze and fall silent, a fast train (≥ 2.6–2.8/s, a hen's exodus call
# ~4/s) makes it call and come. Strongest just after hatching, weak in adults.
PULSE_KINDS = frozenset({"sound:clap", "sound:door", "sound:bang", "sound:noise"})
PULSE_GAP_S = 2.0  # a longer gap ends the train
PULSE_MIN_GAP_S = 0.2  # closer onsets are one pulse (a clap and its echo)
PULSE_MIN = 4  # pulses before it counts as a train
PULSE_MAX_CV = 0.35  # a train is regular; knocks at random are not a rhythm
SLOW_TRAIN_HZ = (0.7, 1.9)
FAST_TRAIN_HZ = 2.4
# While it walks, what it hears is mostly its own doing (feet, things it knocks): higher
# systems take sounds at a fraction then (movement-related suppression of hearing: mice,
# Schneider et al. 2014), and no train is counted.
MOVING_HEARING = 0.35


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


class PresentedLike(Protocol):
    """One onset the reflex arcs processed (``reflex.Presented``)."""

    @property
    def stimulus(self) -> str: ...
    @property
    def raw(self) -> float: ...
    @property
    def response(self) -> float: ...
    @property
    def reflex(self) -> str | None: ...
    @property
    def bearing(self) -> float | None: ...
    @property
    def track(self) -> int: ...


@dataclass(frozen=True, slots=True)
class Onset:
    """A plain :class:`PresentedLike` (for tests and synthetic drivers)."""

    stimulus: str
    raw: float
    response: float
    scare: float = 0.0
    reflex: str | None = None
    bearing: float | None = None
    depress: bool = True
    track: int = 0


@dataclass(slots=True)
class Appraisal:
    """This tick's consequences of the reflexes, for the drives, selection and steering."""

    inputs: dict[str, float] = field(default_factory=dict)  # selection inputs (0..1)
    threat: float = 0.0  # drive impulses (once per event)
    pain: float = 0.0
    positive: float = 0.0
    contact: float = 0.0
    novelty: float = 0.0
    alarm: bool = False  # a vocal alarm now (a fall, a real fright)
    veer: float = 0.0  # extra yaw-rate bias (rad/s, + = left): learned danger spots


class Appraiser:
    """Keeps the after-effect traces and where the last scare came from."""

    __slots__ = (
        "_g",
        "_learn",
        "_traces",
        "_lifted",
        "_bump_t",
        "_alarm_t",
        "threat_heading",
        "threat_t",
        "threat_track",
        "_pulses",
    )

    def __init__(self, genome: Genome, learner: Learner) -> None:
        self._g = genome
        self._learn = learner
        self._traces: dict[str, float] = {}
        self._lifted = False
        self._bump_t = -1e9
        self._alarm_t = -1e9  # its last startle alarm call (s)
        # where the last scare came from, as a heading in the duck's own odometry frame (so it
        # stays put while the duck turns away from it); None = it never knew
        self.threat_heading: float | None = None
        # … and when it was set (s): telemetry only — the startle's back-steps are not made toward a scare that came from behind
        self.threat_t = -1e9
        # … and the eye's track of the sighting that scared it (0: it did not see what scared it —
        # a sound, a touch — or the eye gave no track). Telemetry: nothing in the brain reads it
        # to decide; it goes out on the ``threat`` target so a verifier can tell what it fled from
        self.threat_track = 0
        self._pulses: list[float] = []  # recent sharp-sound onset times (the rhythm rule)

    # ── the step ─────────────────────────────────────────────────────────────

    def step(
        self,
        t_s: float,
        dt_s: float,
        frame: SensorFrame,
        presented: Iterable[PresentedLike],
        events: Iterable[Any],
        *,
        asleep: bool,
        walking: bool,
        lifted: bool = False,
        blocked_ahead: bool = False,
        moving: bool = False,
        looming_bearing: float | None = None,
        looming_track: int = 0,
    ) -> Appraisal:
        """``presented``: every onset the reflex arcs processed this tick; ``events``: the
        reflex events (:class:`~duckmind.neuro.contract.ReflexEvent`: fired reflexes and the
        fall episode's "body:fall" / "body:righted" / … marks); ``lifted``: held off the floor;
        ``blocked_ahead``: the veer reflex will not let it walk straight on (a wall close ahead)."""
        out = Appraisal()
        tr = self._traces
        for key in list(tr):
            value = tr[key] * math.exp(-dt_s / TRACE_TAU_S.get(key, TRACE_TAU_DEFAULT_S))
            if value < TRACE_FLOOR:
                del tr[key]
            else:
                tr[key] = value
        odom = frame.body.odom
        bearings: dict[str, float | None] = {}
        tracks: dict[str, int] = {}
        for onset in presented:
            bearings[onset.stimulus] = onset.bearing
            tracks[onset.stimulus] = onset.track
            self._onset(onset, t_s, out, asleep, moving)
        for event in events:
            self._event(
                event, t_s, odom, out, asleep, bearings, looming_bearing, tracks, looming_track
            )
        if self._lifted and not lifted and "set_down" in frame.events:
            self._bump("set_down", 1.0)
        elif self._lifted and not lifted:
            self._bump("set_down", 0.6)
        self._lifted = lifted
        if "stumbled" in frame.events:
            self._bump("stumble", 1.0)  # look at its feet; no pain (proprio's stumble is noisy)
        self._tof(frame, walking or frame.body.posture in ("stand", "walk"))
        if blocked_ahead:
            self._bump("obstacle", 1.0)
            self._bump("turn_away", 1.0)
        if walking:
            for bearing, dist in self._learn.avoided_near(odom):
                if abs(bearing) < 1.0 and dist < 1.4:
                    out.veer += (-1.0 if bearing > 0 else 1.0) * 0.6 * _clip(1.4 - dist)
                    self._bump("avoid_spot", 1.0)
        for key, value in tr.items():
            out.inputs[key] = value
        return out

    # ── onsets: what each one means whether or not a reflex fired ─────────────

    def _onset(
        self, onset: PresentedLike, t_s: float, out: Appraisal, asleep: bool, moving: bool
    ) -> None:
        channel, _, kind = onset.stimulus.partition(":")
        response = _clip(onset.response)
        if channel == "sound" and moving:
            response *= MOVING_HEARING
            if onset.reflex != "startle":
                return  # its own footsteps and knocks: not a cue, not a rhythm, not news
        value = self._learn.cue(onset.stimulus, t_s)  # a cue: learn from what follows
        if value > 0.05:
            self._bump("good_cue", value)
        elif value < -0.05:
            self._bump("bad_cue", -value)
        if channel in ("visual", "sound"):
            out.novelty = max(out.novelty, 0.4 * response)
        if channel == "sound":
            self._bump("heard", response)
            if onset.stimulus in PULSE_KINDS and not moving:
                self._pulse(t_s)
            if onset.bearing is None and response > 0.1:
                # the real duck's mono microphone: it heard it but cannot tell where
                self._bump("sound_nodir", response)
            if asleep and response > 0.35:
                self._bump("woken", response)
            if (kind in SHARP_SOUNDS or onset.reflex == "startle") and response > 0.2:
                self._bump("alert", response * self._g.p.alert_stand)
        elif onset.stimulus == "light:off":
            self._bump("light_change", response)
        elif onset.stimulus == "light:on":
            self._bump("light_change", response)
            self._bump("light_on", response)

    # ── reflex events: their consequences ─────────────────────────────────────

    def _event(
        self,
        event: Any,
        t_s: float,
        odom: tuple[float, float, float],
        out: Appraisal,
        asleep: bool,
        bearings: dict[str, float | None],
        looming_bearing: float | None,
        tracks: dict[str, int],
        looming_track: int,
    ) -> None:
        reflex = str(event.reflex)
        stimulus = str(event.stimulus)
        level = _clip(float(event.strength))
        channel = stimulus.split(":", 1)[0]
        if stimulus == "body:fall":  # a fall episode opens (robotd's verdict: REAL)
            out.pain = max(out.pain, 0.8)
            out.alarm = True
            self._learn.fell(odom, t_s)
            self._learn.outcome(-1.0, t_s)
        elif stimulus == "body:righting_failed":
            self._learn.fell(odom, t_s)  # another try: a hard recovery
        elif stimulus in ("body:righted", "body:righted_help"):
            self._learn.righted(t_s, stimulus == "body:righted_help")
            self._bump("after_fall", 1.0)
            self._bump("righted", 1.0)
        elif stimulus == "body:lifted":
            out.threat = max(out.threat, 0.25 * (1.2 - self._g.axes["boldness"]))
        elif stimulus == "body:bump":
            if t_s - self._bump_t > BUMP_REFRACTORY_S:
                self._bump_t = t_s
                self._learn.bumped(odom)
            self._bump("bump", 1.0)  # "oops": never a scare, never timid (spec §8.9 E4)
        elif stimulus == "habit:double_look":
            self._bump("double_take", 0.5)  # its quirk: it looks again
        elif reflex == "startle":
            out.threat = max(out.threat, 0.8 * level)
            if level >= STARTLE_ALARM and not asleep and t_s - self._alarm_t >= ALARM_REFRACTORY_S:
                out.alarm = True  # a real fright: the alarm call (its beak opens with it)
                self._alarm_t = t_s
            self._bump("startled", level)
            self._bump("recovering", level)  # then minutes of alert scanning, a shake, a preen
            if channel == "visual":
                self._bump("loom_scare", level)
            self._learn.outcome(-level, t_s)
            self._learn.mark(odom, -0.3 * level)
            if asleep:
                self._bump("woken", 1.0)
            bearing = bearings.get(stimulus)
            track = tracks.get(stimulus, 0)
            if bearing is None and channel == "visual":
                bearing = looming_bearing
                track = looming_track
            if bearing is not None:
                self.threat_heading = wrap_angle(odom[2] + bearing)
                self.threat_t = t_s
                self.threat_track = track
        elif reflex == "withdraw":
            out.threat = max(out.threat, 0.45 * level)
            self._bump("touched_sharp", level)
            self._bump("recovering", 0.5 * level)
            self._learn.outcome(-0.6 * level, t_s)
            bearing = bearings.get(stimulus)
            if bearing is None and channel == "touch":
                bearing = TOUCH_BEARING.get(stimulus.split(":", 1)[1])
            if bearing is not None:
                self.threat_heading = wrap_angle(odom[2] + bearing)
                self.threat_t = t_s
                self.threat_track = tracks.get(stimulus, 0)
        elif reflex == "lean_in":
            out.positive = max(out.positive, 0.5 * level)
            out.contact = max(out.contact, 0.15)
            self._bump("petted", level)
            self._learn.outcome(0.8, t_s)
        elif reflex == "freeze" and channel == "light":
            out.threat = max(out.threat, 0.2 * level)
            self._bump("recovering", 0.5 * level)
        elif stimulus == "tof:drop":
            self._bump("turn_away", 1.0)
            self._learn.mark(odom, -0.4)

    def _tof(self, frame: SensorFrame, upright: bool) -> None:
        """A wall or an edge in front of a standing duck: turning away becomes urgent (the
        reflex has already slowed it; only an arc takes it clear — no turning on the spot)."""
        tof = frame.tof
        if not tof.present:
            return
        if tof.drop_zones >= 4:
            self._bump("turn_away", 1.0)
        if not upright:
            return
        left = tof.left_mm if tof.left_mm is not None else 4000
        right = tof.right_mm if tof.right_mm is not None else 4000
        ahead = tof.ahead_mm if tof.ahead_mm is not None else 4000
        near = min(left, right, ahead)
        if near < TOF_NEAR_MM:
            self._bump("obstacle", (TOF_NEAR_MM - near) / TOF_NEAR_MM)
        if ahead <= TOF_STOP_MM or tof.touch:
            self._bump("obstacle", 1.0)
            self._bump("turn_away", 1.0)

    def _pulse(self, t_s: float) -> None:
        """One sharp sound in a possible train: estimate the train's rate from its last
        intervals (the median, so one missed pulse does not flip it)."""
        pulses = [t for t in self._pulses if t_s - t <= 4 * PULSE_GAP_S]
        if pulses and t_s - pulses[-1] > PULSE_GAP_S:
            pulses = []  # the old train ended
        if pulses and t_s - pulses[-1] < PULSE_MIN_GAP_S:
            return  # the same pulse (an echo, a second class label)
        pulses.append(t_s)
        self._pulses = pulses[-6:]
        if len(self._pulses) < PULSE_MIN:
            return
        gaps = [b - a for a, b in zip(self._pulses, self._pulses[1:], strict=False)]
        mean = sum(gaps) / len(gaps)
        spread = math.sqrt(sum((g - mean) ** 2 for g in gaps) / len(gaps))
        if mean <= 0.0 or spread / mean > PULSE_MAX_CV:
            return  # knocks at random, not a rhythm
        rate = 1.0 / sorted(gaps)[len(gaps) // 2]
        young = self._g.p.hatchling_wobble
        strength = 0.35 + 0.65 * young  # a duckling's reflex; an adult barely shows it
        if SLOW_TRAIN_HZ[0] <= rate <= SLOW_TRAIN_HZ[1]:
            self._bump("rhythm_slow", strength)
        elif rate >= FAST_TRAIN_HZ:
            self._bump("rhythm_fast", strength)

    # ── helpers ──────────────────────────────────────────────────────────────

    def threat_bearing(self, odom: tuple[float, float, float]) -> float | None:
        """Where the last scare came from, from the trunk now (rad, + = left)."""
        if self.threat_heading is None:
            return None
        return wrap_angle(self.threat_heading - odom[2])

    def _bump(self, key: str, value: float) -> None:
        self._traces[key] = max(self._traces.get(key, 0.0), _clip(value))

    def traces(self) -> dict[str, float]:
        return {k: round(v, 3) for k, v in sorted(self._traces.items())}


__all__ = ["Appraisal", "Appraiser", "Onset", "PresentedLike"]
