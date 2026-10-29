"""「咻～」兜風 — the wheee ride: lifted and swung, then a shake (a:sound 2026-09-28).

Owner, 2026-09-28 (「A B 全數」, behaviour 3): lifted and swung, the duck streams the official
segmented wheee (start / loop / end) until it is set down, then shakes itself — emerging from
the neural brain, decided only from its own senses.

**What it senses.** Its own body only: duck-body's proprio — ``lifted`` / ``set_down`` (the
body link's confirmed pickup), and its unsure states: the legs ``unloaded`` or swinging free
(``airborne``) — and the trunk IMU's gyro
(``BodySense.gyro``, 50 Hz). A ride is being carried through the air *and swung*: the trunk
turning and tilting (the gyro's magnitude, low-passed: :data:`SWING_ON` rad/s) — not merely held
(:class:`RideSense`). The robot's IMU gives the gyro and the on-chip orientation, not the
accelerometer (pet:body), so a ride that only translates, never turns, is not felt — neither
here nor on the robot.

**What it feels** (:class:`RideSkill`): a ride is thrilling or frightening, and which is not a
dice roll — it is the duck: bold, playful, lively ducks feel the thrill, timid ones the fear,
fear already in the body tips it to fear, and it learns: a ride that ends with a gentle set-down
and nothing bad after is a good ride (Rescorla–Wagner on its own ``value``), so a timid duck
that is swung gently again and again comes to enjoy it; a fall right after makes the next one
scary. Thrill → it holds the joy-ride voice (robotd's ``wheee`` hold: start, then the loop for
as long as the ride lasts, then — when it stops holding — the end segment) with its head up
into the wind and the beak a little open; fear → an ``alarm`` call and the head tucked low.
The ride ends when it is set down (or held still for :data:`STILL_END_S`). Being set down after
a ride leaves ``after_ride`` (the feathers ruffled): the selection's pull toward a shake (a
head shake at once, a body shake once it stands).

Standard library only; floats in the prototype.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from duckmind.neuro.contract import HeadPose, SensorFrame

RIDE_VERSION = "neuro-ride-1"

ENERGY_TAU_S = 0.35  # the gyro's magnitude, low-passed
SWING_ON = 0.9  # rad/s: being swung (a hand carrying it along an arc, turning it)
SWING_OFF = 0.45  # … and still again below this
STILL_END_S = 6.0  # held still this long in the air: the ride is over (else at the set-down)
LIGHT_HOLD_S = 0.6  # the legs' "light" state bridges its own flicker this long
JOY_AT = 0.45  # the appraisal at or above this is a thrill, below it fear
AFTER_TAU_S = 10.0  # the ruffled feeling after a ride fades over this
LEARN_RATE = 0.35  # per ride (Rescorla–Wagner toward its outcome)
FALL_WINDOW_S = 3.0  # a fall this soon after being set down: the ride ended badly
ALARM_EVERY_S = 2.5  # a frightened ride calls again this often while it lasts
JOY_POSE = HeadPose(np=-0.25, hp=0.55, hy=0.0, hr=0.0, mouth=0.28)  # head up into the wind
FEAR_POSE = HeadPose(np=0.35, hp=-0.45, hy=0.0, hr=0.0, mouth=0.0)  # tucked low


@dataclass(frozen=True, slots=True)
class RideView:
    lifted: bool  # its own sense: off the floor (proprio)
    energy: float  # the gyro's magnitude, low-passed (rad/s)
    swinging: bool
    riding: bool  # a ride is on (lifted and swung, not yet set down or long still)
    ride_s: float  # how long this ride has lasted
    still_s: float  # how long it has been still in the air


@dataclass(slots=True)
class RideSense:
    """Lifted and swung, from its own body (see the module docstring)."""

    lifted: bool = False
    energy: float = 0.0
    swinging: bool = False
    riding: bool = False
    ride_s: float = 0.0
    still_s: float = 0.0
    last_t: float | None = None
    light_until: float = -1e9  # the legs were light (unsure) until this moment (+ a hold)

    def step(
        self, t_s: float, frame: SensorFrame, *, light: bool = False
    ) -> tuple[RideView, str | None]:
        """The view now and the ride's edge this tick: ``"start"``, ``"end"`` or None.
        ``light``: duck-body's proprio says the legs carry nothing but it is not sure it is lifted
        (``unloaded``, or the legs swinging free: ``airborne``) — while being swung its lift test
        flickers between the two, so either counts as off the floor."""
        dt = 0.0 if self.last_t is None else min(0.25, max(0.0, t_s - self.last_t))
        self.last_t = t_s
        events = frame.events
        if "lifted" in events:
            self.lifted = True
        if "set_down" in events or frame.body.fallen:
            self.lifted = False
            self.light_until = -1e9
        if light:
            self.light_until = t_s + LIGHT_HOLD_S
        off_floor = self.lifted or t_s < self.light_until
        g = frame.body.gyro
        mag = math.sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2])
        if dt > 0.0:
            self.energy += (mag - self.energy) * (1.0 - math.exp(-dt / ENERGY_TAU_S))
        if self.swinging:
            self.swinging = self.energy >= SWING_OFF
        else:
            self.swinging = self.energy >= SWING_ON
        edge: str | None = None
        if self.riding:
            self.ride_s += dt
            self.still_s = 0.0 if self.swinging else self.still_s + dt
            if not off_floor or self.still_s >= STILL_END_S:
                self.riding = False
                edge = "end"
        elif off_floor and self.swinging:
            self.riding = True
            self.ride_s = 0.0
            self.still_s = 0.0
            edge = "start"
        view = RideView(
            lifted=off_floor,
            energy=self.energy,
            swinging=self.swinging,
            riding=self.riding,
            ride_s=self.ride_s,
            still_s=self.still_s,
        )
        return view, edge


@dataclass(frozen=True, slots=True)
class RideTick:
    view: RideView
    joy: float  # the appraisal of this ride (0..1; ≥ JOY_AT: a thrill)
    thrilled: bool  # a ride on, and it is a thrill
    frightened: bool  # a ride on, and it is fear
    wheee_hold: bool  # hold the joy-ride voice this tick (robotd's wheee hold)
    alarm: bool  # call ``alarm`` now (a frightened ride)
    pose: HeadPose | None  # the ride's head (joy up into the wind, fear tucked), or None
    after_ride: float  # the ruffled feeling after a ride (0..1): the pull to shake
    ride_joy: float  # the selection feature while riding (thrill)
    ride_fear: float  # … (fear)
    value: float  # the learned value of being swung (−1..1)
    rides: int
    started: bool = False  # a ride began this tick
    ended: bool = False  # … ended this tick


@dataclass(slots=True)
class RideSkill:
    """How a duck takes a ride. ``boldness`` / ``activity`` (0..1) are its temperament axes;
    ``young`` 1 for a hatchling … 0 for an adult (a hatchling is more easily frightened)."""

    boldness: float = 0.5
    activity: float = 0.5
    young: float = 0.25
    sense: RideSense | None = None
    value: float = 0.0
    joy: float = 0.0
    after: float = 0.0
    rides: int = 0
    last_alarm: float = -1e9
    ended_at: float | None = None  # when the last ride ended (a fall soon after: a bad one)
    ended_joy: float = 0.0
    last_t: float | None = None

    def __post_init__(self) -> None:
        if self.sense is None:
            self.sense = RideSense()

    def appraise(self, *, play: float, fear: float, arousal: float) -> float:
        """This duck, now: a thrill (≥ JOY_AT) or fear. Its temperament, its play and fear drives
        and what it learned of rides; never a coin."""
        joy = (
            0.12
            + 0.55 * self.boldness
            + 0.2 * self.activity
            + 0.3 * play
            + 0.1 * min(1.0, arousal)
            + 0.45 * self.value
            - 0.5 * fear
            - 0.1 * self.young
        )
        return max(0.0, min(1.0, joy))

    def step(
        self,
        t_ms: int,
        frame: SensorFrame,
        *,
        play: float,
        fear: float,
        arousal: float,
        light: bool = False,
    ) -> RideTick:
        t_s = t_ms / 1000.0
        dt = 0.0 if self.last_t is None else min(0.25, max(0.0, t_s - self.last_t))
        self.last_t = t_s
        assert self.sense is not None
        view, edge = self.sense.step(t_s, frame, light=light)
        alarm = False
        if edge == "start":
            self.rides += 1
            self.joy = self.appraise(play=play, fear=fear, arousal=arousal)
            self.last_alarm = -1e9
        if view.riding:
            # the feeling follows the ride: fear already in the body spoils it as it goes
            now = self.appraise(play=play, fear=fear, arousal=arousal)
            self.joy += (now - self.joy) * (1.0 - math.exp(-dt / 2.0))
        thrilled = view.riding and self.joy >= JOY_AT
        frightened = view.riding and not thrilled
        if frightened and t_s - self.last_alarm >= ALARM_EVERY_S and view.swinging:
            alarm = True
            self.last_alarm = t_s
        if edge == "end":
            # the feathers ruffled: the pull to shake, the longer the ride the stronger
            self.after = max(self.after, min(1.0, 0.55 + 0.15 * view.ride_s))
            self.ended_at = t_s
            self.ended_joy = self.joy
        # learning: a ride that ended with a set-down and no fall soon after was a good ride
        if self.ended_at is not None:
            if frame.body.fallen or "fell" in frame.events:
                self.value += LEARN_RATE * (-1.0 - self.value)
                self.ended_at = None
            elif t_s - self.ended_at >= FALL_WINDOW_S:
                target = 1.0 if self.ended_joy >= JOY_AT else 0.5  # (a scary one that went well)
                self.value += LEARN_RATE * (target - self.value)
                self.ended_at = None
        self.value = max(-1.0, min(1.0, self.value))
        self.after *= math.exp(-dt / AFTER_TAU_S) if dt else 1.0
        pose = JOY_POSE if thrilled else FEAR_POSE if frightened else None
        return RideTick(
            view=view,
            joy=self.joy,
            thrilled=thrilled,
            frightened=frightened,
            wheee_hold=thrilled,
            alarm=alarm,
            pose=pose,
            after_ride=self.after,
            ride_joy=self.joy if thrilled else 0.0,
            ride_fear=(1.0 - self.joy) if frightened else 0.0,
            value=self.value,
            rides=self.rides,
            started=edge == "start",
            ended=edge == "end",
        )


__all__ = [
    "FEAR_POSE",
    "JOY_AT",
    "JOY_POSE",
    "RIDE_VERSION",
    "RideSense",
    "RideSkill",
    "RideTick",
    "RideView",
    "STILL_END_S",
    "SWING_ON",
]
