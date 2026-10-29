"""曬太陽 — sunbathing in the brightest spot (a:play, 2026-09-28; owner item 7).

On a sunny morning the east window of ``default_home`` lays a long strip of sunlight across the
floor (the world's truth: :mod:`duckmind.world.neuro_sun`); by ten o'clock it has shrunk toward
the wall, by eleven it is gone. The duck knows none of that. It has two senses for it:

* its camera's light meter (``CameraSense.lux``: the auto-exposure reading, 2 Hz) — standing in
  the sun the floor in front of it reads 5,000–30,000 lux where the room reads 100–300; the duck
  keeps its own sense of how bright things usually are (a slow average of the log light level, away
  from the sun) and feels "in the sun" when the light is ≥ 4× that and ≥ 1,500 lux;
* a bright floor region in its picture (a ``light`` sighting: bearing, the ground-plane range to
  its near edge, its size and brightness) — where the sun lies, seen from afar.

It remembers where it saw the sun (its own odometry) and at what time of its day (its clock), so
the next morning it may go straight there. The act (``sunbathe``) wins from a basking urge
(``bask``: it builds while the duck is awake, calm and it is day; a dreamy or snuggly, calm duck's
faster; basking uses it up), a sunny spot seen or remembered (``sunny``) and being in the sun
(``in_sun``), against fear, restlessness and the night.

What the body does: it walks there (the brain's own walk), into the bright part — stops once the
light under its eyes says sun, walks a little further in, faces into the light and sits down;
then basks — seated, the head low and relaxed, the face a little up toward the light, nearly
still, now and then a slow drowsy dip of the head. This body has no eyelids: "eyes half closed" is
shown with the head's carriage and stillness, and the page says so. When the sun moves off it
(the light meter drops for a while) it gets up and follows the patch; when the urge is used up,
or the sun is gone, the act ends.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from duckmind.i18n import msg
from duckmind.neuro.contract import Gaze, HeadPose, SensorFrame, Target
from duckmind.neuro.skill_common import (
    CAPTURE_LAG_MS,
    TAIL_MS,
    OdomHistory,
    PlayOut,
    bearing_range,
    legs_still,
    lens_point,
    sighting_odom,
)

SUN_VERSION = "neuro-sun-1"
SUN_MIN_LUX = 1500.0  # the light meter this bright …
SUN_CONTRAST = 10.0  # … and this many times its usual light: in the sun (most of the floor it
# sees is sunlit — one sunny strip at the picture's edge gives ~5×)
SHADE_DROP = 0.1  # basking, the light fallen to a tenth of what it settled in: the sun moved
TYPICAL_TAU_S = 600.0  # its sense of the usual light level adapts this slowly
INTO_M = 0.45  # it walks this far past the patch's near edge (into the light)
PATCH_FORGET_S = 5400.0  # a sunny spot seen longer ago than this is not trusted (the sun moves)
PATCH_SAME_M = 0.6
DEEPER_WALK_MS = 2200  # … and once the floor at its chest is sunny, ≈ 0.3 m more (its body in)
SHADE_MS = 12_000  # out of the sun this long while basking: the sun moved, follow it
SETTLE_WALK_MS = 1500  # into the light a little more before it sits (≈ 0.2 m)
# Walking in (``into`` / ``deeper``) it steers toward the light — the brightest floor it saw in
# the last BRIGHT_FRESH_MS, else the spot it remembers — never just straight on: coming up along
# a patch's edge, straight on kept it walking beside the sun (final evaluation, snuggler trial 0:
# 30 walks in, the meter at 5,000–40,000 lux looking ahead, its own footprint in sun 0 of 9 points)
BRIGHT_FRESH_MS = 1500
INTO_GAIN = 1.2  # yaw rate per radian off (clipped to the arc walk's ±1 rad/s)
# The middle of the bright region: the sighting's bearing and elevation are the region's middle in
# the picture — with the lens this high over a level floor the elevation gives its ground range
# (the eye's own geometry, as ``head_look``). It steers for that middle, not for the near edge: a
# walk aimed past the near edge along the line of sight ended at the patch's rim when it came up
# at a slant (after the steering fix, final2: seated with 2–4 of 9 footprint points in sun in 6 of
# 11 settles).
CAM_H_M = 0.16
CENTRE_MAX_M = 1.2  # the middle at most this far past the near edge (else: the edge + INTO_M)
NEAR_LOOK_M = 0.30  # a look whose bright floor starts nearer than this sees only the bit in front
# of it: it does not move a spot it already remembers
AT_SPOT_M = 0.15
PHASE_ZH = {
    "idle": "",
    "go": msg("neuro.incoming.incoming_skill_sun.PHASE_ZH.1a9940"),
    "into": msg("neuro.incoming.incoming_skill_sun.PHASE_ZH.d10f9f"),
    "settle": msg("neuro.incoming.incoming_skill_sun.PHASE_ZH.d5d910"),
    "bask": msg("neuro.incoming.incoming_skill_sun.PHASE_ZH.2ff603"),
    "shift": msg("neuro.incoming.incoming_skill_sun.PHASE_ZH.d4d43c"),
    "search": msg("neuro.incoming.incoming_skill_sun.PHASE_ZH.34a56e"),
}


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass(slots=True)
class SunSpot:
    """A sunny spot it saw (its odometry frame): where the light's near edge was, the direction
    it looked, how bright, when (its brain time and its day's minute)."""

    x: float
    y: float
    dir: float  # the heading it looked along (odometry yaw of the bearing)
    lux: float
    seen_ms: int
    day_minute: float
    n: int = 1
    cx: float | None = None  # the middle of the bright region it saw (odometry), if it could tell
    cy: float | None = None

    def goal(self) -> tuple[float, float]:
        """The middle of the bright region it saw; else a point INTO_M past the near edge, along
        the direction it saw it."""
        if self.cx is not None and self.cy is not None:
            return self.cx, self.cy
        return self.x + INTO_M * math.cos(self.dir), self.y + INTO_M * math.sin(self.dir)


class SunSense:
    """Its light meter against its own usual light, and the sunny spots it has seen."""

    def __init__(self) -> None:
        self.log_typ: float | None = None  # log10 of its usual light (away from the sun)
        self.lux: float | None = None
        self.readings: deque[float] = deque(maxlen=3)  # its last meter readings (~1.5 s)
        self.meter_ms: int | None = None
        self.spots: list[SunSpot] = []
        self.pass_ms: int | None = None
        self.best_bearing: float | None = None  # the brightest light in view now (rad)
        self.best_ms = -1_000_000

    def step(self, t_ms: int, dt: float, frame: SensorFrame, hist: OdomHistory) -> None:
        cam = frame.camera
        lux = getattr(cam, "lux", None)
        if cam.usable and lux is not None:
            self.lux = float(lux)
            pass_ms = t_ms - int(cam.age_ms)
            if self.meter_ms is None or pass_ms - self.meter_ms >= 400:
                self.meter_ms = pass_ms
                self.readings.append(self.lux)
            log = math.log10(max(self.lux, 0.01))
            if self.log_typ is None:
                self.log_typ = min(log, math.log10(300.0))
            elif self.lux < SUN_MIN_LUX:
                k = 1.0 - math.exp(-dt / TYPICAL_TAU_S)
                self.log_typ += k * (log - self.log_typ)
        if not cam.usable:
            return
        pass_ms = t_ms - int(cam.age_ms)
        if self.pass_ms is not None and abs(pass_ms - self.pass_ms) <= 5:
            return
        self.pass_ms = pass_ms
        at = hist.at(pass_ms - CAPTURE_LAG_MS)
        lens = lens_point(frame)
        for s in cam.sightings:
            if s.kind != "light" or s.range_m is None:
                continue
            slux = float(getattr(s, "lux", None) or 0.0)
            if slux < SUN_MIN_LUX:
                continue
            near = float(s.range_m)
            x, y = sighting_odom(at, lens, s.bearing, near)
            centre = None
            elevation = getattr(s, "elevation", None)
            if elevation is not None and elevation < -0.05:
                ground = CAM_H_M / math.tan(-float(elevation))
                if near + 0.05 <= ground <= near + CENTRE_MAX_M:
                    centre = sighting_odom(at, lens, s.bearing, ground)
            heading = at[2] + s.bearing
            if near >= NEAR_LOOK_M or not self.spots:
                self._remember(x, y, heading, slux, pass_ms, frame.day_minute, centre)
            if self.best_ms < pass_ms or s.bearing is not None:
                self.best_bearing, self.best_ms = s.bearing, pass_ms

    def _remember(
        self,
        x: float,
        y: float,
        heading: float,
        lux: float,
        t_ms: int,
        minute: float,
        centre: tuple[float, float] | None = None,
    ) -> None:
        for spot in self.spots:
            if math.hypot(spot.x - x, spot.y - y) < PATCH_SAME_M:
                w = 1.0 / (1.0 + min(spot.n, 4))
                spot.x += w * (x - spot.x)
                spot.y += w * (y - spot.y)
                if centre is not None:
                    if spot.cx is None or spot.cy is None:
                        spot.cx, spot.cy = centre
                    else:
                        spot.cx += w * (centre[0] - spot.cx)
                        spot.cy += w * (centre[1] - spot.cy)
                spot.dir = math.atan2(
                    math.sin(spot.dir) * (1 - w) + math.sin(heading) * w,
                    math.cos(spot.dir) * (1 - w) + math.cos(heading) * w,
                )
                spot.lux, spot.seen_ms, spot.day_minute = lux, t_ms, minute
                spot.n += 1
                return
        cx, cy = centre if centre is not None else (None, None)
        self.spots.append(SunSpot(x, y, heading, lux, t_ms, minute, cx=cx, cy=cy))
        if len(self.spots) > 12:
            self.spots.sort(key=lambda s: -s.seen_ms)
            del self.spots[12:]

    def level(self) -> float | None:
        """The middle of its last meter readings (lux), None before any."""
        if not self.readings:
            return None
        return sorted(self.readings)[len(self.readings) // 2]

    def in_sun(self) -> bool:
        """In bright sun: the middle of its last three meter readings (~1.5 s: one reading at a
        patch's edge or a head turn does not count) is ≥ 1,500 lux and ≥ 10× its usual light."""
        if not self.readings or self.log_typ is None:
            return False
        mid = sorted(self.readings)[len(self.readings) // 2]
        return mid >= SUN_MIN_LUX and math.log10(mid) - self.log_typ >= math.log10(SUN_CONTRAST)

    def best(self, t_ms: int, odom: tuple[float, float, float], minute: float) -> SunSpot | None:
        """The sunny spot to go to: seen recently (or at about this time of its day before),
        bright, near."""
        best, best_score = None, 0.0
        for spot in self.spots:
            age_s = (t_ms - spot.seen_ms) / 1000.0
            same_time = abs(((minute - spot.day_minute + 720.0) % 1440.0) - 720.0) < 45.0
            if age_s > PATCH_FORGET_S and not same_time:
                continue
            _, rng = bearing_range(odom, *spot.goal())
            fresh = _clip(1.0 - age_s / PATCH_FORGET_S) if age_s <= PATCH_FORGET_S else 0.4
            score = fresh * math.log10(max(spot.lux, 10.0)) / (0.6 + rng)
            if score > best_score:
                best, best_score = spot, score
        return best

    def view(self) -> dict[str, object]:
        return {
            "lux": None if self.lux is None else round(self.lux),
            "usual_lux": None if self.log_typ is None else round(10.0**self.log_typ),
            "in_sun": self.in_sun(),
            "spots": [
                {
                    "x_mm": round(s.x * 1000),
                    "y_mm": round(s.y * 1000),
                    "lux": round(s.lux),
                    "minute": round(s.day_minute),
                    "n": s.n,
                }
                for s in self.spots[:4]
            ],
        }


@dataclass
class SunBask:
    """The sunbathing act's state machine (one per duck)."""

    phase: str = "idle"
    phase_ms: int = 0
    shade_since: int | None = None
    basked_s: float = 0.0
    settles: int = 0
    shifts: int = 0
    tries: int = 0
    deeper: bool = False
    settle_lux: float | None = None
    last_event: str | None = None
    history: list[str] = field(default_factory=list)

    def start(self, t_ms: int) -> None:
        self.phase, self.phase_ms = "idle", t_ms
        self.shade_since = None

    def _go(self, phase: str, t_ms: int) -> None:
        self.phase, self.phase_ms = phase, t_ms
        self.history.append(phase)
        if len(self.history) > 40:
            del self.history[:10]

    def step(
        self,
        t_ms: int,
        dt: float,
        frame: SensorFrame,
        sense: SunSense,
        odom: tuple[float, float, float],
        *,
        halted: bool,
    ) -> PlayOut:
        self.last_event = None
        posture = frame.body.posture
        sitting = posture == "sit"
        standing = posture in ("stand", "walk")
        sunny = sense.in_sun()
        if halted:
            return PlayOut(phase=self.phase, zh=PHASE_ZH.get(self.phase, ""))
        if self.phase == "idle":
            self._go("bask" if sunny and sitting else "go", t_ms)
        if self.phase in ("go", "shift"):
            if sunny and standing:
                self.tries = 0
                self._go("into", t_ms)
            else:
                spot = sense.best(t_ms, odom, frame.day_minute)
                if spot is None:
                    return PlayOut(phase="search", zh=PHASE_ZH["search"], posture="stand")
                gx, gy = spot.goal()
                bearing, rng = bearing_range(odom, gx, gy)
                if not standing:
                    return PlayOut(phase=self.phase, zh=PHASE_ZH[self.phase], posture="stand")
                return PlayOut(
                    far=True,
                    posture="stand",
                    phase=self.phase,
                    zh=PHASE_ZH[self.phase],
                    gaze=Gaze(bearing, -0.25, 0.7, 0.4),
                    target=Target("place", bearing, rng),
                )
        if self.phase == "into":
            since = t_ms - self.phase_ms
            if since < SETTLE_WALK_MS:
                return PlayOut(
                    drive=True,
                    vx=0.40,
                    vyaw=self._toward(t_ms, sense, odom, frame),
                    posture="stand",
                    phase="into",
                    zh=PHASE_ZH["into"],
                    quiet=True,
                )
            self._go("check", t_ms)
        if self.phase == "deeper":
            since = t_ms - self.phase_ms
            if since < DEEPER_WALK_MS:
                return PlayOut(
                    drive=True,
                    vx=0.40,
                    vyaw=self._toward(t_ms, sense, odom, frame),
                    posture="stand",
                    phase="deeper",
                    zh=PHASE_ZH["into"],
                    quiet=True,
                )
            self._go("check", t_ms)
        if self.phase == "check":
            # it looks down at the floor right in front of its chest: sun here too?
            since = t_ms - self.phase_ms
            look = PlayOut(
                drive=True,
                posture="stand",
                phase="check",
                zh=msg("neuro.incoming.incoming_skill_sun.SunBask.step"),
                head=HeadPose(np=1.0, hp=-1.0),
                head_blend=1.0,
                quiet=True,
                gaze=Gaze(0.0, -0.9, 1.0, 0.5),
            )
            if not legs_still(frame) or since < TAIL_MS + 1100:
                return look
            if sunny and self.deeper:
                self.tries = 0
                self.deeper = False
                self._go("settle", t_ms)
            elif sunny:
                # the floor right in front of its chest is sunny: its body is still ~0.2–0.4 m
                # short of it (the eye sits ahead of the body) — in a little further, look again
                self.deeper = True
                self._go("deeper", t_ms)
                return PlayOut(drive=True, posture="stand", phase="deeper", zh=PHASE_ZH["into"])
            else:
                self.deeper = False
                self.tries += 1
                if self.tries >= 3:
                    self.tries = 0
                    self._go("go", t_ms)
                    return PlayOut(phase="go", zh=PHASE_ZH["go"], posture="stand")
                self._go("into", t_ms)  # a little further in
                return PlayOut(drive=True, posture="stand", phase="into", zh=PHASE_ZH["into"])
        if self.phase == "settle":
            if sitting:
                self.settles += 1
                self.last_event = "settled"
                self.settle_lux = sense.level()
                self._go("bask", t_ms)
            else:
                return PlayOut(
                    drive=True,
                    posture="sit",
                    phase="settle",
                    zh=PHASE_ZH["settle"],
                    head=HeadPose(),
                    head_blend=0.6,
                    quiet=True,
                )
        if self.phase == "bask":
            if not sitting:
                self._go("settle", t_ms)
                return PlayOut(drive=True, posture="sit", phase="settle", zh=PHASE_ZH["settle"])
            level = sense.level()
            ref = self.settle_lux or level or 1.0
            shaded = level is not None and level < SHADE_DROP * ref and not sunny
            if not shaded:
                self.shade_since = None
                if sunny or (level is not None and level >= 0.3 * ref):
                    self.basked_s += dt
            elif self.shade_since is None:
                self.shade_since = t_ms
            elif t_ms - self.shade_since > SHADE_MS:
                self.shifts += 1
                self.last_event = "shade"
                self.shade_since = None
                self._go("shift", t_ms)
                return PlayOut(phase="shift", zh=PHASE_ZH["shift"], posture="stand", event="shade")
            toward = sense.best_bearing if t_ms - sense.best_ms < 8000 else None
            face = 0.0 if toward is None else max(-0.5, min(0.5, toward))
            return PlayOut(
                drive=True,
                posture="sit",
                phase="bask",
                zh=PHASE_ZH["bask"],
                quiet=True,
                gaze=Gaze(face, 0.1, 0.25, 0.15),
                event=self.last_event,
            )
        return PlayOut(phase=self.phase, zh=PHASE_ZH.get(self.phase, ""))

    @staticmethod
    def _toward(
        t_ms: int, sense: SunSense, odom: tuple[float, float, float], frame: SensorFrame
    ) -> float:
        """The yaw rate that bends the walk in toward the light (its own sightings only): toward
        the middle of the bright region it remembers while that is still ahead of it, else toward
        the brightest floor it saw last (a look down at its feet sees only the strip in front)."""
        aim = 0.0
        spot = sense.best(t_ms, odom, frame.day_minute)
        goal = None if spot is None else bearing_range(odom, *spot.goal())
        if goal is not None and goal[1] > AT_SPOT_M:
            aim = goal[0]
        elif sense.best_bearing is not None and t_ms - sense.best_ms < BRIGHT_FRESH_MS:
            aim = float(sense.best_bearing)
        return _clip(INTO_GAIN * aim, -1.0, 1.0)

    def view(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "basked_s": round(self.basked_s, 1),
            "settles": self.settles,
            "shifts": self.shifts,
        }


def sun_features(
    sense: SunSense, t_ms: int, odom: tuple[float, float, float], minute: float
) -> dict[str, float]:
    """``sunny`` (a sunny spot seen or remembered for this time of its day, not where it is),
    ``in_sun`` (in bright sunlight now), 0..1."""
    in_sun = 1.0 if sense.in_sun() else 0.0
    spot = sense.best(t_ms, odom, minute)
    sunny = 0.0
    if spot is not None and not in_sun:
        age_s = (t_ms - spot.seen_ms) / 1000.0
        _, rng = bearing_range(odom, *spot.goal())
        fresh = _clip(1.0 - age_s / PATCH_FORGET_S) if age_s <= PATCH_FORGET_S else 0.4
        sunny = fresh * _clip(1.3 - 0.2 * rng)
    return {"sunny": sunny, "in_sun": in_sun}


__all__ = ["SUN_VERSION", "SunBask", "SunSense", "SunSpot", "sun_features"]
