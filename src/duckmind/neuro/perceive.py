"""From a SensorFrame to what the brain notices as ongoing levels.

Onsets — the sudden things that orient, startle, habituate — are detected in ONE place, the
reflex arcs (:mod:`duckmind.neuro.reflex`, which report every onset they processed as
``ReflexOutput.presented``). This module keeps the slower picture: who is talking, what music
is on, the nearest person and whether someone just arrived, the most salient thing, the nest,
how dark it is, a gentle or a sharp touch going on, how much something is looming, and whether a
touch right now would come as no surprise (a hand was seen, a voice heard, a touch just felt).

Everything is computed from the frame alone plus the brain's own short memory of earlier frames.
Bearings stay egocentric; a remembered thing or person is kept in the duck's own odometry frame
(where it believed it was), never in world coordinates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from duckmind.neuro.contract import SensorFrame, Sighting, wrap_angle
from duckmind.neuro.touch import is_stroking

TOUCH_BEARING = {"head": 0.0, "beak": 0.0, "left": math.pi / 2, "right": -math.pi / 2}
TOUCH_BEARING.update({"back": math.pi, "tail": math.pi, "body": math.pi})
PERSON_GONE_S = 20.0  # nobody seen this long: seeing someone again is a new arrival (the eye
# re-issues a track id whenever a person leaves the picture for a moment: an id is no identity)
HAND_SEEN_S = 2.5  # a hand seen this recently announces a touch
VOICE_HEARD_S = 6.0  # … and a voice heard this recently, a little
# A touch just felt makes the next one expected: after a pleasant one for a good while (the hand
# is still there, patting again), after a sharp one only briefly (s).
TOUCH_AFTER_TAU_S = {True: 6.0, False: 2.0}
# … but not a touch it caused itself (owner 2026-10-04: a box it pushed into made the next touch
# of the box "expected", and the duck leant into it): while it walks, or soon after its body
# felt a stall, a gentle touch with no hand in view and nothing sliding on it is a thing it met
OWN_TOUCH_S = 4.0


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass(slots=True)
class Percept:
    """What the brain notices this tick (levels; the onsets are the reflex arcs')."""

    voice: float = 0.0  # ongoing speech level 0..1
    name: float = 0.0  # its name is being called (0..1)
    hello: float = 0.0
    music: float = 0.0
    tv: float = 0.0
    touch_site: str | None = None
    touch_gentle: float = 0.0
    touch_sharp: float = 0.0
    person: Sighting | None = None  # the nearest person in view
    person_new: float = 0.0  # impulse: someone appeared (a track unseen for a while)
    thing: Sighting | None = None  # the most salient thing in view
    things: tuple[Sighting, ...] = ()  # all visible thing/toy tracks; brain chooses attention
    nest: Sighting | None = None
    motion: float = 0.0  # visual motion level in view
    looming: float = 0.0
    looming_bearing: float | None = None
    looming_track: int = 0  # the eye's track of the sighting that is looming most (0 = none)
    dark: float = 0.0  # 0 bright … 1 black
    expect_touch: float = 0.0  # 0..1 a touch now would be expected
    expect_hand: float = 0.0  # … of which what a hand seen and a voice heard make of it
    events: tuple[str, ...] = ()


class Perceiver:
    """Keeps the few earlier readings needed to tell an arrival from someone who stayed."""

    __slots__ = ("_person_t", "_hand_t", "_voice_t", "_touch_t", "_touch_pleasant", "_bump_t")

    def __init__(self) -> None:
        self._person_t = -1e9  # the last time it saw a person (any), s
        self._hand_t = -1e9  # last time a hand/person was seen close, s
        self._voice_t = -1e9  # last time a voice was heard, s
        self._touch_t = -1e9  # last time a touch was felt, s
        self._touch_pleasant = False  # … and whether it was a gentle one
        self._bump_t = -1e9  # last time its body felt a stall or a brush, s

    def perceive(self, frame: SensorFrame, t_s: float, asleep: bool) -> Percept:
        out = Percept(events=frame.events)
        self._sound(frame, t_s, out)
        self._vision(frame, t_s, out, asleep)
        self._touch(frame, t_s, out)
        return out

    # ── channels ─────────────────────────────────────────────────────────────

    def _sound(self, frame: SensorFrame, t_s: float, out: Percept) -> None:
        s = frame.sound
        level = _clip((s.level_db - 40.0) / 45.0)
        kind = s.kind
        if kind in ("voice", "name", "hello"):
            out.voice = max(0.25, level)
            self._voice_t = t_s
            if kind == "name":
                out.name = max(0.5, level)
            elif kind == "hello":
                out.hello = max(0.5, level)
        elif kind == "music":
            out.music = max(0.2, level)
        elif kind == "tv":
            out.tv = max(0.2, level)

    def _vision(self, frame: SensorFrame, t_s: float, out: Percept, asleep: bool) -> None:
        cam = frame.camera
        out.dark = {"bright": 0.0, "dim": 0.35, "dark": 0.75, "black": 1.0}.get(cam.light, 0.0)
        if asleep or not cam.usable:
            return
        out.motion = cam.motion
        best_thing: Sighting | None = None
        things: list[Sighting] = []
        for sight in cam.sightings:
            if sight.kind in ("person", "hand"):
                near = sight.range_m is not None and sight.range_m < 0.8
                if near or sight.kind == "hand":
                    self._hand_t = t_s
                if sight.kind == "person":
                    if out.person is None or (sight.range_m or 9.0) < (out.person.range_m or 9.0):
                        out.person = sight
                    if t_s - self._person_t > PERSON_GONE_S:
                        out.person_new = max(out.person_new, 0.45 + (0.35 if near else 0.0))
                    self._person_t = t_s
            elif sight.kind in ("thing", "toy"):
                things.append(sight)
                if best_thing is None or sight.size > best_thing.size:
                    best_thing = sight
            elif sight.kind == "nest":
                out.nest = sight
            if sight.looming > 0.6 and sight.size > 0.06:
                strength = _clip(sight.looming / 3.0) * _clip(0.4 + sight.size / 0.4)
                if strength > out.looming:
                    out.looming = strength
                    out.looming_bearing = sight.bearing
                    out.looming_track = sight.track
        out.thing = best_thing
        out.things = tuple(things)

    def _touch(self, frame: SensorFrame, t_s: float, out: Percept) -> None:
        touch = frame.touch
        site = touch.site
        if "blocked" in frame.events or "brushed" in frame.events:
            self._bump_t = t_s
        # a hand seen close, a voice just heard, a touch just felt: a touch now is no surprise
        pleasant = self._touch_pleasant
        hand = _clip(
            (0.7 if t_s - self._hand_t < HAND_SEEN_S else 0.0)
            + (0.35 if t_s - self._voice_t < VOICE_HEARD_S else 0.0)
        )
        out.expect_hand = hand
        out.expect_touch = _clip(
            hand
            + (0.85 if pleasant else 0.5)
            * math.exp(-max(0.0, t_s - self._touch_t) / TOUCH_AFTER_TAU_S[pleasant])
        )
        if site is None:
            return
        out.touch_site = site
        stroking = is_stroking(touch.stroke)
        gentle = stroking or touch.pressure < 0.45
        own = (
            (frame.body.posture == "walk" or t_s - self._bump_t <= OWN_TOUCH_S)
            and gentle
            and not stroking
            and hand < 0.5
        )
        if own:
            # a thing it walked into (a box): no one is touching it — not a gentle touch that
            # means company, and not one that makes the next touch expected
            return
        if gentle:
            out.touch_gentle = _clip(0.4 + touch.pressure * 0.4 + touch.stroke * 0.4)
        else:
            out.touch_sharp = _clip(touch.pressure)
        self._touch_t = t_s
        self._touch_pleasant = gentle


def bearing_to(odom: tuple[float, float, float], x: float, y: float) -> tuple[float, float]:
    """(bearing, range) from the duck's current odometry pose to an odometry-frame point."""
    ox, oy, yaw = odom
    dx, dy = x - ox, y - oy
    return wrap_angle(math.atan2(dy, dx) - yaw), math.hypot(dx, dy)


def to_odom(
    odom: tuple[float, float, float], bearing: float, range_m: float
) -> tuple[float, float]:
    """Where an egocentric sighting is in the duck's odometry frame (a belief; odometry drifts)."""
    ox, oy, yaw = odom
    return ox + range_m * math.cos(yaw + bearing), oy + range_m * math.sin(yaw + bearing)


# ── the lens is not the trunk ─────────────────────────────────────────────────
# A thing's bearing and ground range come from the head camera's lens (the eye model's
# convention), which sits ahead of the trunk's centre: 0.081 m at HOME, (0.074, ±0.031) at ±0.6
# rad of head yaw (this rig, standing; measured on the official MJCF, 2026-10-02). The beak is
# about where the lens is, and "how far is it to me" is a question about the trunk's centre.
LENS_PIVOT_M = 0.028  # the head's yaw axis ahead of the trunk origin (this rig, standing)
LENS_ARM_M = 0.053  # … and the lens ahead of that axis: lens ≈ pivot + arm·(cos yaw, sin yaw)
LENS_SEATED_M = -0.013  # seated, the lens sits this much further back


def lens_point(frame: SensorFrame) -> tuple[float, float]:
    """Where the head camera's lens is in the trunk's level frame (ahead, left), from the head's
    own measured yaw (robotd's joints; this lane's rig: 0.081 m ahead at HOME, (0.074, ±0.031)
    at ±0.6 rad of yaw) — the eye's ranges and bearings start there, not at the trunk origin."""
    head = frame.body.head
    yaw = float(head[2]) if len(head) > 2 else 0.0
    ahead = LENS_PIVOT_M + LENS_ARM_M * math.cos(yaw)
    if frame.body.posture == "sit":
        ahead += LENS_SEATED_M
    return ahead, LENS_ARM_M * math.sin(yaw)


def sighting_odom(
    at: tuple[float, float, float], lens: tuple[float, float], bearing: float, range_m: float
) -> tuple[float, float]:
    """A sighting (bearing and ground range from the lens, the trunk's heading) in the odometry
    frame of the capture."""
    ax = lens[0] + range_m * math.cos(bearing)
    ay = lens[1] + range_m * math.sin(bearing)
    ox, oy, yaw = at
    c, s = math.cos(yaw), math.sin(yaw)
    return ox + c * ax - s * ay, oy + s * ax + c * ay


def range_from_trunk(lens: tuple[float, float], bearing: float, range_m: float) -> float:
    """How far the point a sighting names is from the trunk's centre (the lens's own ground range
    and bearing, carried over the lens's offset)."""
    return math.hypot(lens[0] + range_m * math.cos(bearing), lens[1] + range_m * math.sin(bearing))
