"""A synthetic room for a:social's brain-level measurements: the family, the furniture, a scare.

NOT the simulator and NOT physics — like :mod:`duckmind.neuro.synth` (which it extends): a fast
stand-in so the neural brain's social acts can be run over many ducks and seeds in seconds. The
headless physics runs (measurement records, not in the repo) are the body's truth.

It adds to the synthetic world what a:social's acts need, as the real senses would give it:

* **the family** (:data:`duckmind.world.people.PERSONAS`: 主人, 伴侶, 小孩, 長輩, 訪客) as walkers
  with a script (:meth:`SocialSynth.come`, :meth:`walk`, :meth:`leave`, :meth:`care` — come close,
  pat its head, talk to it — :meth:`call`); the camera reports each as a ``person`` sighting in its
  field of view and light (range to the feet, image motion, an eye track re-issued after 0.8 s out
  of view) with its *look* (:func:`duckmind.world.neuro_social_senses.observe_look`); speech
  carries its speaker signature; a pat is a touch on the head;
* **the furniture** of the default home, moved into this room's frame (0…6 m): the coffee table
  (walkable underneath, its top 0.40 m up: a roof), the sofa against the north wall (solid; the
  nook between its east arm and the east wall), the bookshelf; the floor line ends at them
  ("wall") and the body stops against them (a stall → proprio's ``blocked``);
* **the roof reading** under the table (MODEL, :func:`~duckmind.world.neuro_social_senses.
  read_roof`'s limits: light, a sharp frame);
* **the social side channel** (:meth:`SocialSynth.social`) the brain gets until the contract
  carries ``look`` / ``voice`` / ``roof``.

Deterministic for a seed; truth (who is who, where the furniture is) is for the evaluation only.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from typing import Any

from duckmind.neuro.contract import (
    FLOOR_RANGE_M,
    CameraSense,
    FreeSector,
    MotorCommand,
    Sighting,
    wrap_angle,
)
from duckmind.neuro.skill_social_brain import SocialInput
from duckmind.neuro.synth import CAM_HALF, ROOM, SoundEvent, SynthWorld
from duckmind.world.neuro_social_senses import (
    ROOF_OPEN,
    observe_look,
    observe_voice,
    read_roof,
)
from duckmind.world.people import PERSONAS

DOOR = (5.5, 0.4)
FEET_M = 0.12  # a person's feet are this much nearer than their centre
PERSON_R = 0.2  # a person's footprint (the duck stops short of their legs)
TRACK_DROP_MS = 800
DUCK_R = 0.09
CAMERA_M = 0.2  # the lens above the floor


@dataclass(frozen=True)
class Box:
    name: str
    x0: float
    y0: float
    x1: float
    y1: float
    solid: bool = True  # the body cannot go in (a table's top is not solid: only its legs)
    top_m: float | None = None  # a roof at this height (the table)

    def inside(self, x: float, y: float, pad: float = 0.0) -> bool:
        return self.x0 - pad <= x <= self.x1 + pad and self.y0 - pad <= y <= self.y1 + pad


# The default home's furniture in this room's frame (world + 3 m): default_home.json.
TABLE = Box("table", 3.55, 4.0, 4.85, 4.6, solid=False, top_m=0.40)
TABLE_LEGS = tuple(
    Box(f"table_leg_{i}", x - 0.025, y - 0.025, x + 0.025, y + 0.025)
    for i, (x, y) in enumerate(((3.55, 4.6), (4.85, 4.6), (3.55, 4.0), (4.85, 4.0)))
)
SOFA = Box("sofa", 2.69, 5.02, 5.11, 5.9)
BOOKSHELF = Box("bookshelf", 0.075, 3.7, 0.425, 4.7)
FURNITURE = (TABLE, SOFA, BOOKSHELF, *TABLE_LEGS)
SOLID = tuple(b for b in FURNITURE if b.solid)
# the places a:social's evaluation calls shelters (truth, for scoring only)
SHELTER_TRUTH = {
    "under_table": (TABLE.x0 + 0.1, TABLE.y0 + 0.1, TABLE.x1 - 0.1, TABLE.y1 - 0.1),
    "sofa_nook_east": (5.11, 5.02, 6.0, 6.0),
    "sofa_nook_west": (2.0, 5.02, 2.69, 6.0),
    "shelf_north": (0.0, 4.7, 0.7, 5.3),
    "shelf_south": (0.0, 3.1, 0.7, 3.7),
    "corner_sw": (0.0, 0.0, 0.6, 0.6),
    "corner_se": (5.4, 0.0, 6.0, 0.6),
    "corner_nw": (0.0, 5.4, 0.6, 6.0),
}


def shelter_at(x: float, y: float) -> str | None:
    for name, (x0, y0, x1, y1) in SHELTER_TRUTH.items():
        if x0 <= x <= x1 and y0 <= y <= y1:
            return name
    return None


@dataclass
class Walker:
    who: str
    speed: float
    x: float = DOOR[0]
    y: float = DOOR[1]
    here: bool = False
    goal: tuple[float, float] | None = None
    leaving: bool = False
    vx: float = 0.0
    vy: float = 0.0
    track: int = 0
    seen_ms: int = -10_000
    next_step_ms: int = 0


@dataclass
class SocialSynth(SynthWorld):
    """The synthetic world with the family, the furniture and the roof (see the module doc).
    ``family`` (the base world's own scripted person) is off: this world's people follow
    :attr:`script` only."""

    family: bool = False
    x: float = 3.0
    y: float = 2.0
    yaw: float = 1.57
    posture: str = "stand"
    script: list[tuple[int, str, dict[str, Any]]] = field(default_factory=list)
    lux_band: str = "bright"

    def __post_init__(self) -> None:
        super().__post_init__()
        self._dock = (3.0, 0.25)
        self._rng = random.Random(f"social-synth:{self.seed}")
        self.walkers = {p.id: Walker(p.id, p.speed_m_s) for p in PERSONAS}
        self._script_i = 0
        self._cares: list[tuple[int, int, str]] = []  # (t0, t1, who)
        self._next_track = 500
        self._speech: tuple[int, str, str] | None = None  # (t_end, kind, who)
        self._speech_sig: tuple[float, float] | None = None
        self._looks: dict[int, tuple[float, ...]] = {}
        self._roof: float | None = None
        self._roof_t = -10_000
        self.truth_log: list[dict[str, Any]] = []
        self.script.sort(key=lambda item: item[0])
        self.toys = []  # the social runs keep the floor clear of the base world's toys

    # ── the script ─────────────────────────────────────────────────────────

    def come(self, t_ms: int, who: str, to: tuple[float, float] | str = "duck") -> None:
        self.script.append((t_ms, "come", {"who": who, "to": to}))
        self.script.sort(key=lambda item: item[0])

    def walk(self, t_ms: int, who: str, to: tuple[float, float]) -> None:
        self.script.append((t_ms, "walk", {"who": who, "to": to}))
        self.script.sort(key=lambda item: item[0])

    def leave(self, t_ms: int, who: str) -> None:
        self.script.append((t_ms, "leave", {"who": who}))
        self.script.sort(key=lambda item: item[0])

    def care(self, t_ms: int, who: str, seconds: float = 60.0) -> None:
        """``who`` comes to the duck and spends ``seconds`` with it: pats and talk."""
        self.script.append((t_ms, "care", {"who": who, "s": seconds}))
        self.script.sort(key=lambda item: item[0])

    def call(self, t_ms: int, who: str) -> None:
        self.script.append((t_ms, "call", {"who": who}))
        self.script.sort(key=lambda item: item[0])

    def scare(self, t_ms: int) -> None:
        """A big fright: a bang near it and a hand looming at its face."""
        self.bang(t_ms)
        self.loom(t_ms + 150)

    # ── the family ─────────────────────────────────────────────────────────

    def _family(self, t_ms: int, dt: float) -> None:
        while self._script_i < len(self.script) and self.script[self._script_i][0] <= t_ms:
            _, what, p = self.script[self._script_i]
            self._script_i += 1
            self._do(t_ms, what, p)
        for w in self.walkers.values():
            if not w.here:
                continue
            goal = w.goal
            if w.leaving and goal is None:
                goal = DOOR
            if goal is None:
                w.vx = w.vy = 0.0
                continue
            dx, dy = goal[0] - w.x, goal[1] - w.y
            dist = math.hypot(dx, dy)
            if dist < 0.05:
                w.vx = w.vy = 0.0
                w.goal = None
                if w.leaving:
                    w.here = False
                continue
            step = min(dist, w.speed * dt)
            w.vx, w.vy = dx / dist * w.speed, dy / dist * w.speed
            w.x += dx / dist * step
            w.y += dy / dist * step
            if t_ms >= w.next_step_ms:  # their footsteps
                w.next_step_ms = t_ms + 450
                self.sounds.append(SoundEvent(t_ms, "footsteps", 52.0, 150, w.x, w.y))
        for t0, t1, who in self._cares:
            w = self.walkers[who]
            if not (t0 <= t_ms < t1) or not w.here or w.goal is not None:
                continue
            if math.hypot(w.x - self.x, w.y - self.y) > 1.1:
                continue
            k = (t_ms - t0) // 1000
            if k % 9 == 2 and (t_ms - t0) % 1000 < 100:  # a pat every 9 s, ~3.5 s long
                self.touches.append((t_ms, t_ms + 3500, "head", 0.3, 0.6))
            if k % 7 == 5 and (t_ms - t0) % 1000 < 100:  # a few words every 7 s
                self._say(t_ms, who, "voice" if k % 21 else "name")

    def _do(self, t_ms: int, what: str, p: dict[str, Any]) -> None:
        who = str(p.get("who"))
        w = self.walkers.get(who)
        if w is None:
            return
        if what in ("come", "care"):
            if not w.here:
                w.x, w.y = DOOR
                w.here = True
            w.leaving = False
            w.goal = self._near_duck() if p.get("to", "duck") == "duck" else tuple(p["to"])
            if what == "care":
                self._cares.append((t_ms, t_ms + int(1000 * float(p.get("s", 60.0))), who))
        elif what == "walk":
            if not w.here:
                w.x, w.y = DOOR
                w.here = True
            w.leaving = False
            w.goal = (float(p["to"][0]), float(p["to"][1]))
        elif what == "leave":
            w.leaving = True
            w.goal = DOOR
        elif what == "call":
            self._say(t_ms, who, "name")

    def _near_duck(self) -> tuple[float, float]:
        """Where a person comes to stand by the duck: ~0.75 m in front of it, inside the room."""
        for turn in (0.0, 0.6, -0.6, 1.2, -1.2, math.pi):
            gx = self.x + 0.75 * math.cos(self.yaw + turn)
            gy = self.y + 0.75 * math.sin(self.yaw + turn)
            if 0.35 < gx < ROOM - 0.35 and 0.35 < gy < ROOM - 0.35:
                if not any(b.inside(gx, gy, 0.25) for b in SOLID):
                    return gx, gy
        return self.x, self.y

    def _say(self, t_ms: int, who: str, kind: str) -> None:
        w = self.walkers[who]
        dur = 1500 if kind == "voice" else 700
        self.sounds.append(SoundEvent(t_ms, kind, 64.0, dur, w.x, w.y))
        dist = max(0.3, math.hypot(w.x - self.x, w.y - self.y))
        snr = 64.0 - 20.0 * math.log10(dist) - 30.0
        self._speech = (t_ms + dur, kind, who)
        self._speech_sig = observe_voice(who, snr, self._rng)

    # ── the senses ─────────────────────────────────────────────────────────

    def frame(self, t_ms: int) -> Any:
        """The base world's frame, with the head's yaw where the camera looks (the synthetic
        head: the gaze it was sent) so the brain's own proprioception matches its view."""
        frame = super().frame(t_ms)
        head = (0.349, 0.349, float(self._head_yaw), 0.0)
        return replace(frame, body=replace(frame.body, head=head))

    def _band(self) -> str:
        b = self._brightness
        return "bright" if b > 0.5 else "dim" if b > 0.25 else "dark" if b > 0.08 else "black"

    def _camera(self, t_ms: int, light: str) -> CameraSense:
        base = super()._camera(t_ms, light)  # nest, looms, the floor line (walls only)
        seen = [s for s in base.sightings if s.kind != "person"]
        looks: dict[int, tuple[float, ...]] = {}
        if light not in ("black",):
            view = self.yaw + self._head_yaw
            for w in self.walkers.values():
                if not w.here:
                    continue
                dx, dy = w.x - self.x, w.y - self.y
                dist = math.hypot(dx, dy)
                if dist < 0.1 or dist > 5.5:
                    continue
                bearing = wrap_angle(math.atan2(dy, dx) - self.yaw)
                if abs(wrap_angle(bearing - (view - self.yaw))) > CAM_HALF + 0.25 / dist:
                    continue
                if self._blocked_view(w.x, w.y):
                    continue
                if w.track == 0 or t_ms - w.seen_ms > TRACK_DROP_MS:
                    self._next_track += 1
                    w.track = self._next_track
                w.seen_ms = t_ms
                rng_m = max(0.1, dist - FEET_M) * (1.0 + self._rng.gauss(0.0, 0.08))
                speed = math.hypot(w.vx, w.vy)
                seen.append(
                    Sighting(
                        "person",
                        bearing + self._rng.gauss(0.0, 0.02),
                        0.3,
                        round(rng_m, 3),
                        0.25,
                        min(1.0, speed * 2.0),
                        0.0,
                        w.track,
                        "model",
                    )
                )
                look = observe_look(w.who, light, rng_m, self._rng)
                if look is not None:
                    looks[w.track] = look
        self._looks = looks
        floor = base.floor
        if base.floor_t_ms == t_ms and floor:
            floor = self._floor_line()
            self._floor = floor
        return CameraSense(
            base.usable,
            base.light,
            base.brightness,
            max((s.motion for s in seen), default=0.0),
            tuple(seen),
            base.age_ms,
            base.source,
            floor,
            base.floor_t_ms,
        )

    def _blocked_view(self, px: float, py: float) -> bool:
        """The sofa or the bookshelf between the lens and a person's legs hides them."""
        for b in (SOFA, BOOKSHELF):
            for k in range(1, 12):
                u = k / 12.0
                if b.inside(self.x + u * (px - self.x), self.y + u * (py - self.y)):
                    return True
        return False

    def _floor_line(self) -> tuple[FreeSector, ...]:
        view = self.yaw + self._head_yaw
        light = self._band()
        sectors = []
        for k in range(5):
            rel = CAM_HALF * (0.8 - 0.4 * k)
            heading = view + rel
            free = min(self._ray(heading), FLOOR_RANGE_M)
            end = "wall" if free < FLOOR_RANGE_M - 1e-6 else "unknown"
            for w in self.walkers.values():
                if not w.here:
                    continue
                dx, dy = w.x - self.x, w.y - self.y
                along = dx * math.cos(heading) + dy * math.sin(heading)
                across = -dx * math.sin(heading) + dy * math.cos(heading)
                if 0.0 < along - PERSON_R < free and abs(across) < PERSON_R:
                    free, end = along - PERSON_R, "thing"
            free *= 1.0 + self._rng.gauss(0.0, 0.02)
            if light == "dim":
                free *= 0.8
            sectors.append(
                FreeSector(wrap_angle(heading - self.yaw), 0.13, round(max(0.05, free), 3), end)
            )
        return tuple(sectors)

    def _ray(self, angle: float) -> float:
        best = super()._ray(angle)
        c, s = math.cos(angle), math.sin(angle)
        for b in SOLID:
            # slab test against the box
            tmin, tmax = -math.inf, math.inf
            for p0, d, lo, hi in ((self.x, c, b.x0, b.x1), (self.y, s, b.y0, b.y1)):
                if abs(d) < 1e-9:
                    if not lo <= p0 <= hi:
                        tmin, tmax = math.inf, -math.inf
                    continue
                t0, t1 = (lo - p0) / d, (hi - p0) / d
                if t0 > t1:
                    t0, t1 = t1, t0
                tmin, tmax = max(tmin, t0), min(tmax, t1)
            if tmax >= max(tmin, 0.0) and tmin > 0.0:
                best = min(best, tmin)
        return best

    def social(self, t_ms: int) -> SocialInput:
        """The side channel for this tick (looks for the person tracks in view, the speaker's
        signature while speech sounds, the roof reading)."""
        voice = None
        if self._speech is not None and t_ms <= self._speech[0]:
            voice = self._speech_sig
        roof = None
        if t_ms - self._roof_t >= 250:
            self._roof_t = t_ms
            under = TABLE.inside(self.x, self.y, -0.05)
            truth = (TABLE.top_m - CAMERA_M) if under and TABLE.top_m is not None else None
            stepping = self.posture == "walk"
            roof = read_roof(truth, self._band(), stepping, True, self._rng)
        return SocialInput(looks=dict(self._looks), voice=voice, roof=roof)

    # ── the body ───────────────────────────────────────────────────────────

    def apply(self, cmd: MotorCommand, dt: float, head_yaw: float | None = None) -> None:
        x0, y0 = self.x, self.y
        super().apply(cmd, dt, head_yaw)
        if any(b.inside(self.x, self.y, DUCK_R) for b in SOLID):
            self.x, self.y = x0, y0  # it walked into the furniture: it stays
            if cmd.vx != 0.0:
                self._stall_s += dt
                if self._stall_s >= 0.6:
                    self._events.append("blocked")
                    self._stall_s = -0.8
        for w in self.walkers.values():
            if w.here and math.hypot(w.x - self.x, w.y - self.y) < PERSON_R + DUCK_R:
                self.x, self.y = x0, y0  # a person's legs
        self.truth_log.append(
            {
                "t": self._t_ms,
                "x": round(self.x, 3),
                "y": round(self.y, 3),
                "yaw": round(self.yaw, 3),
                "posture": self.posture,
                "behaviour": cmd.behaviour,
            }
        )
        if len(self.truth_log) > 40_000:
            del self.truth_log[:10_000]


def run_social(
    brain: Any, world: SocialSynth, seconds: float, hz: float = 10.0, each: Any = None
) -> None:
    """Step a :class:`~duckmind.neuro.skill_social_brain.SocialBrain` in the world (the side
    channel given every tick). ``each(t_ms, cmd, tel)`` is called after every tick."""
    dt_ms = int(round(1000.0 / hz))
    t = 0
    end = int(seconds * 1000)
    while t < end:
        frame = world.frame(t)
        brain.sense_social(world.social(t))
        cmd, tel = brain.step(t, frame)
        world.apply(cmd, dt_ms / 1000.0)
        if each is not None:
            each(t, cmd, tel)
        t += dt_ms


__all__ = [
    "BOOKSHELF",
    "FURNITURE",
    "ROOF_OPEN",
    "SHELTER_TRUTH",
    "SOFA",
    "SocialSynth",
    "TABLE",
    "run_social",
    "shelter_at",
]
