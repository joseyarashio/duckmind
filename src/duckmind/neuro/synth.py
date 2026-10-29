"""A small synthetic sensor world, for tests and fast measurements of the brain only.

NOT the simulator and NOT physics: the live prototype runs the brain on the real body path
(robotd on MuJoCo) with neuro:senses' frames. This toy exists so the brain's dynamics can be
measured over simulated hours in seconds: a 6 × 6 m room with walls, a charging nest, a few
toys, one family member who comes and goes, talks, calls the duck's name, sits near it and
strokes it, plus scheduled sounds. It produces frames the way the real senses would — only
what the head camera's field of view and the light allow, sounds without direction (a mono
microphone), touch only when the hand is on the body, odometry that drifts — so the brain
under test gets no more than on the robot. Kinematics are rough stand-ins for the measured
gait (commanded 0.30–0.55 m/s moves ~0.1–0.17 m/s; arcs turn 20–45°/s; straight walks drift).

Deterministic for a seed. Imports only the contract (never :mod:`duckmind.world`).
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from duckmind.neuro.contract import (
    FLOOR_RANGE_M,
    BodySense,
    CameraSense,
    FreeSector,
    MotorCommand,
    PowerSense,
    SensorFrame,
    Sighting,
    SoundSense,
    TofSense,
    TouchSense,
    wrap_angle,
)
from duckmind.neuro.signals import Stream

ROOM = 6.0
MARGIN = 0.15
CAM_HALF = math.radians(37.4) / 2
HEAD_YAW_MAX = 0.6


@dataclass
class Toy:
    x: float
    y: float
    track: int
    kind: str = "thing"


@dataclass
class SoundEvent:
    t_ms: int
    kind: str
    db: float  # at 1 m
    dur_ms: int
    x: float | None = None  # source position (None: from elsewhere in the home)
    y: float | None = None
    dist: float | None = None  # or: always this far from the duck (a scripted test stimulus)


@dataclass
class SynthWorld:
    seed: int = 1
    tof: bool = True
    camera: bool = True
    family: bool = True
    day_minute0: float = 540.0
    x: float = 0.6  # it starts on its nest (hatched on arrival, sitting on the dock)
    y: float = 0.6
    yaw: float = 0.8
    posture: str = "sit"
    battery: float = 0.85
    toys: list[Toy] = field(default_factory=list)
    sounds: list[SoundEvent] = field(default_factory=list)
    touches: list[tuple[int, int, str, float, float]] = field(
        default_factory=list
    )  # t0, t1, site, pressure, stroke
    looms: list[tuple[int, int]] = field(default_factory=list)
    lights: list[tuple[int, float]] = field(default_factory=list)  # (t_ms, brightness)
    falls: list[int] = field(default_factory=list)  # t_ms of a push that makes it fall
    # like the eye's tracker (neuro_senses.TRACK_DROP_MS): a thing out of view for 0.8 s gets a
    # NEW track id when it is seen again — the brain must know it again by where it is
    reissue_tracks: bool = True

    def __post_init__(self) -> None:
        self._s = Stream(self.seed, "synth-world")
        self._odom_err = [0.0, 0.0, 0.0]
        self._posture_t = 0
        self._want = "sit"
        self._head_yaw = 0.0
        self._events: list[str] = []
        self._fallen_until = -1
        self._person = [4.5, 4.5, 0.0, 0.0]  # x, y, vx, vy
        self._home = True
        self._plan_t = 0
        self._goal = (4.5, 4.5)
        self._pet_until = -1
        self._next_pet = 90_000
        self._next_talk = 20_000
        self._step_t = 0
        self._brightness = 0.75
        self._t_ms = 0
        self._last_frame_t = 0
        self._dock = (0.6, 0.6)
        self._on_dock = False
        self._stall_s = 0.0
        self._floor_t = -10_000
        self._floor: tuple[FreeSector, ...] = ()
        self._floor_look_ms: int | None = None
        if not self.toys:
            self.toys = [Toy(3.2, 1.4, 11), Toy(1.6, 4.1, 12), Toy(4.8, 2.9, 13, "toy")]
        self._track_last: dict[int, int] = {}  # toy index → when it was last in view (ms)
        self._track_now: dict[int, int] = {}  # toy index → its current track id
        self._next_track = 100

    # ── scenario helpers (tests) ───────────────────────────────────────────

    def clap(self, t_ms: int, db: float = 78.0, dist: float | None = 1.0) -> None:
        """A clap ``dist`` metres from wherever the duck is then (the same stimulus each time)."""
        self.sounds.append(SoundEvent(t_ms, "clap", db, 120, self.x + 1.0, self.y, dist))

    def bang(self, t_ms: int) -> None:
        self.sounds.append(SoundEvent(t_ms, "bang", 92.0, 200, self.x + 1.5, self.y + 0.5))

    def call_name(self, t_ms: int) -> None:
        self.sounds.append(SoundEvent(t_ms, "name", 66.0, 700, self._person[0], self._person[1]))

    def pet(self, t_ms: int, dur_ms: int = 6000, site: str = "head") -> None:
        self.touches.append((t_ms, t_ms + dur_ms, site, 0.35, 0.6))

    def poke(self, t_ms: int, site: str = "left") -> None:
        self.touches.append((t_ms, t_ms + 250, site, 0.8, 0.0))

    def loom(self, t_ms: int) -> None:
        self.looms.append((t_ms, t_ms + 600))

    # ── the frame ──────────────────────────────────────────────────────────

    def frame(self, t_ms: int) -> SensorFrame:
        dt = max(0.0, (t_ms - self._t_ms) / 1000.0)
        self._t_ms = t_ms
        self._family(t_ms, dt)
        events = tuple(self._events)
        self._events = []
        fallen = t_ms < self._fallen_until
        odom = (
            self.x + self._odom_err[0],
            self.y + self._odom_err[1],
            wrap_angle(self.yaw + self._odom_err[2]),
        )
        body = BodySense(
            posture="unknown" if fallen else self.posture,
            fallen=fallen,
            tilt=1.3 if fallen else 0.02,
            gyro=(0.0, 0.0, 0.0),
            vx=0.0,
            odom=odom,
        )
        for t_light, value in self.lights:
            if t_light <= t_ms:
                self._brightness = value
        light = (
            "bright"
            if self._brightness > 0.5
            else "dim"
            if self._brightness > 0.25
            else "dark"
            if self._brightness > 0.08
            else "black"
        )
        cam = CameraSense(usable=False, source="none")
        if self.camera:
            cam = self._camera(t_ms, light)
        tof = TofSense()
        if self.tof:
            tof = self._tof()
        return SensorFrame(
            t_ms=t_ms,
            body=body,
            events=events,
            tof=tof,
            camera=cam,
            sound=self._sound(t_ms),
            touch=self._touch(t_ms),
            power=PowerSense(battery=self.battery, on_dock=self._on_dock, charging=self._on_dock),
            day_minute=(self.day_minute0 + t_ms / 60000.0) % 1440.0,
            sources={
                "body": "model",
                "proprio": "model",
                "tof": "model" if self.tof else "none",
                "camera": "model" if self.camera else "none",
                "sound": "model",
                "touch": "model",
                "power": "model",
                "clock": "model",
            },
        )

    def _camera(self, t_ms: int, light: str) -> CameraSense:
        seen: list[Sighting] = []
        if light != "black":
            view = self.yaw + self._head_yaw
            px, py, pvx, pvy = self._person
            if self._home:
                sight = self._sight(px, py, view, "person", 1, 0.35)
                if sight is not None:
                    speed = math.hypot(pvx, pvy)
                    seen.append(
                        Sighting(
                            "person",
                            sight[0],
                            0.3,
                            sight[2],
                            0.25,
                            _clip(speed * 2.0),
                            0.0,
                            1,
                            "model",
                        )
                    )
            for index, toy in enumerate(self.toys):
                sight = self._sight(toy.x, toy.y, view, toy.kind, toy.track, 0.08)
                if sight is not None and sight[2] < 3.0 and (light == "bright" or sight[2] < 1.5):
                    seen.append(
                        Sighting(
                            toy.kind,
                            sight[0],
                            -0.2,
                            sight[2],
                            0.08 / max(0.2, sight[2]),
                            0.0,
                            0.0,
                            self._track_of(index, toy, t_ms),
                            "model",
                        )
                    )
            sight = self._sight(self._dock[0], self._dock[1], view, "nest", 99, 0.1)
            if sight is not None and sight[2] < 2.5:
                seen.append(Sighting("nest", sight[0], -0.2, sight[2], 0.1, 0.0, 0.0, 99, "model"))
            for t0, t1 in self.looms:
                if t0 <= t_ms < t1:
                    u = (t_ms - t0) / (t1 - t0)
                    seen.append(
                        Sighting(
                            "hand",
                            self._head_yaw,
                            0.1,
                            0.5 - 0.35 * u,
                            0.15 + 0.5 * u,
                            0.8,
                            2.5,
                            7,
                            "model",
                        )
                    )
        motion = max((s.motion for s in seen), default=0.0)
        if light in ("bright", "dim") and t_ms - self._floor_t >= 330:
            # the floor line (duck-eye FL-v1's shape): five sectors across the picture, the
            # clear floor to the walls and toys, dim light ×0.8
            self._floor_t = t_ms
            view = self.yaw + self._head_yaw
            sectors = []
            for k in range(5):
                rel = CAM_HALF * (0.8 - 0.4 * k)  # left to right
                heading = view + rel
                free, end = min(self._ray(heading), FLOOR_RANGE_M), "wall"
                for toy in self.toys:
                    dx, dy = toy.x - self.x, toy.y - self.y
                    along = dx * math.cos(heading) + dy * math.sin(heading)
                    across = -dx * math.sin(heading) + dy * math.cos(heading)
                    if 0.0 < along < free and abs(across) < 0.09:
                        free, end = along - 0.04, "thing"
                if free >= FLOOR_RANGE_M - 1e-6:
                    end = "unknown"
                if light == "dim":
                    free *= 0.8
                sectors.append(
                    FreeSector(wrap_angle(heading - self.yaw), 0.13, round(max(0.05, free), 3), end)
                )
            self._floor = tuple(sectors)
            self._floor_look_ms = t_ms
        return CameraSense(
            True,
            light,
            self._brightness,
            motion,
            tuple(seen),
            0,
            "model",
            self._floor,
            self._floor_look_ms,
        )

    def _track_of(self, index: int, toy: Toy, t_ms: int) -> int:
        """The track id the eye gives this toy now: its own while it stays in view, a new one
        after it was out of view for 0.8 s (neuro_senses.TRACK_DROP_MS)."""
        if not self.reissue_tracks:
            return toy.track
        last = self._track_last.get(index)
        if last is None or t_ms - last > 800:
            self._next_track += 1
            self._track_now[index] = self._next_track
        self._track_last[index] = t_ms
        return self._track_now[index]

    def _sight(
        self, x: float, y: float, view: float, kind: str, track: int, size: float
    ) -> tuple[float, float, float] | None:
        dx, dy = x - self.x, y - self.y
        dist = math.hypot(dx, dy)
        if dist < 0.05:
            return None
        bearing = wrap_angle(math.atan2(dy, dx) - self.yaw)
        if abs(wrap_angle(bearing - (view - self.yaw))) > CAM_HALF + size / max(0.3, dist):
            return None
        return bearing, 0.0, dist

    def _tof(self) -> TofSense:
        out = []
        for off in (0.26, 0.0, -0.26):
            out.append(int(1000 * self._ray(self.yaw + self._head_yaw * 0.5 + off)))
        left, ahead, right = (d if d < 4000 else None for d in out)
        touch = out[1] < 60
        return TofSense(True, left, ahead, right, 0, touch, 0)

    def _ray(self, angle: float) -> float:
        c, s = math.cos(angle), math.sin(angle)
        best = 9.9
        for bound, comp, pos in (
            (0.0, c, self.x),
            (ROOM, c, self.x),
            (0.0, s, self.y),
            (ROOM, s, self.y),
        ):
            if abs(comp) > 1e-6:
                t = (bound - pos) / comp
                if t > 0:
                    best = min(best, t)
        return best

    def _sound(self, t_ms: int) -> SoundSense:
        level = 32.0
        kind = None
        onset = 0.0
        for ev in self.sounds:
            if ev.t_ms <= t_ms < ev.t_ms + ev.dur_ms:
                if ev.dist is not None:
                    dist = ev.dist
                elif ev.x is not None and ev.y is not None:
                    dist = max(0.3, math.hypot(ev.x - self.x, ev.y - self.y))
                else:
                    dist = 4.0
                db = ev.db - 20.0 * math.log10(dist)
                if db > level:
                    level, kind = db, ev.kind
                    if t_ms - ev.t_ms < max(60, self._tick_ms()):
                        onset = _clip((db - 35.0) / 45.0 + 0.3)
        return SoundSense(level, onset, kind if level > 40.0 else None)

    def _tick_ms(self) -> int:
        return 100

    def _touch(self, t_ms: int) -> TouchSense:
        for t0, t1, site, pressure, stroke in self.touches:
            if t0 <= t_ms < t1:
                return TouchSense(site, pressure, stroke)
        return TouchSense()

    # ── the family member (a scripted person: the environment, not the duck) ──

    def _family(self, t_ms: int, dt: float) -> None:
        if not self.family:
            self._home = False
            return
        s = self._s
        px, py, _, _ = self._person
        if t_ms >= self._plan_t:
            roll = s.uniform()
            if roll < 0.12:
                self._home = not self._home
                if self._home:
                    self.sounds.append(SoundEvent(t_ms, "door", 70.0, 600, None, None))
            gx = MARGIN + 0.5 + (ROOM - 1.3) * s.uniform()
            gy = MARGIN + 0.5 + (ROOM - 1.3) * s.uniform()
            if s.uniform() < 0.35:  # come and sit near the duck
                gx = _clip(self.x + 0.5 * (s.uniform() - 0.5), 0.3, ROOM - 0.3)
                gy = _clip(self.y + 0.7, 0.3, ROOM - 0.3)
            self._goal = (gx, gy)
            self._plan_t = t_ms + int(40_000 + 140_000 * s.uniform())
        if not self._home:
            self._person = [px, py, 0.0, 0.0]
            return
        gx, gy = self._goal
        dx, dy = gx - px, gy - py
        dist = math.hypot(dx, dy)
        speed = 0.6 if dist > 0.1 else 0.0
        vx, vy = (dx / dist * speed, dy / dist * speed) if dist > 0.1 else (0.0, 0.0)
        self._person = [px + vx * dt, py + vy * dt, vx, vy]
        if speed > 0.0 and t_ms >= self._step_t:
            # their footsteps (as the live world's walking people make them)
            self._step_t = t_ms + 450
            self.sounds.append(SoundEvent(t_ms, "footsteps", 52.0, 150, px, py))
        near = math.hypot(self._person[0] - self.x, self._person[1] - self.y) < 0.9
        if t_ms >= self._next_talk:
            kind = "name" if s.uniform() < 0.25 else "voice"
            self.sounds.append(
                SoundEvent(
                    t_ms,
                    kind,
                    64.0,
                    1500 if kind == "voice" else 700,
                    self._person[0],
                    self._person[1],
                )
            )
            self._next_talk = t_ms + int(25_000 + 120_000 * s.uniform())
        if near and speed == 0.0 and t_ms >= self._next_pet:
            site = "head" if s.uniform() < 0.6 else "back"
            dur = int(3000 + 9000 * s.uniform())
            self.touches.append((t_ms, t_ms + dur, site, 0.3 + 0.2 * s.uniform(), 0.5))
            self._next_pet = t_ms + int(60_000 + 200_000 * s.uniform())

    # ── the duck's body in this toy world ─────────────────────────────────────

    def apply(self, cmd: MotorCommand, dt: float, head_yaw: float | None = None) -> None:
        """The body follows the command for ``dt`` s. ``head_yaw`` (rad from HOME): where the
        motor really has the head (the camera looks there); None: the gaze command's yaw."""
        t_ms = self._t_ms
        for t_fall in self.falls:
            if t_fall <= t_ms < t_fall + int(dt * 1000) + 1 and self.posture != "sit":
                self._fallen_until = t_ms + 3000
                self._events.append("fell")
        if t_ms < self._fallen_until:
            return
        if self._fallen_until > 0 and t_ms >= self._fallen_until:
            self._events.append("righted")
            self._fallen_until = -1
            self.posture = "stand"
        if head_yaw is None:
            head_yaw = cmd.gaze.bearing * cmd.gaze.gain
        head = max(-HEAD_YAW_MAX, min(HEAD_YAW_MAX, head_yaw))
        self._head_yaw += (head - self._head_yaw) * min(1.0, dt * 2.0 * math.pi * 1.78)
        want = "stand" if cmd.posture == "stand" else "sit"
        if want != self._want:
            self._want = want
            self._posture_t = t_ms + 1200
            self.posture = "transition"
        if self.posture == "transition" and t_ms >= self._posture_t:
            self.posture = self._want
        # (fid:motor 2026-09-28: a reverse arc — the nest, curiosity's back_arc, backing out of a
        # corner, a withdrawal — moves it backwards too: phase-3 calibration, −0.40 with ±1.0
        # rad/s for 1 s from standing ≈ 8–12 cm back and ~35°)
        moving = self.posture in ("stand", "walk") and cmd.vx != 0.0
        if moving:
            speed = (0.31 if cmd.vx > 0.0 else 0.25) * cmd.vx
            rate = 0.6 * cmd.vyaw - (0.05 if cmd.vx > 0.0 else 0.0)
            self.yaw = wrap_angle(self.yaw + rate * dt)
            nx = self.x + speed * math.cos(self.yaw) * dt
            ny = self.y + speed * math.sin(self.yaw) * dt
            # against a wall the body slides along it; proprio says "blocked" when the
            # progress along the command stays under 30 % for 0.6 s (spec §5.2)
            cx = min(ROOM - MARGIN, max(MARGIN, nx))
            cy = min(ROOM - MARGIN, max(MARGIN, ny))
            progress = math.hypot(cx - self.x, cy - self.y) / max(1e-6, abs(speed) * dt)
            self.x, self.y = cx, cy
            self._odom_err[2] += 0.004 * dt * (1 if cmd.vyaw > 0 else -1)
            self._stall_s = self._stall_s + dt if progress < 0.3 else 0.0
            if self._stall_s >= 0.6:
                self._events.append("blocked")
                self._stall_s = -0.8  # proprio re-arms after a moment
            self.posture = "walk"
            self.battery = max(0.0, self.battery - dt / 5400.0)
        elif self.posture == "walk":
            self.posture = "stand"
        self.battery = max(0.0, self.battery - dt / 20000.0)
        self._on_dock = (
            math.hypot(self.x - self._dock[0], self.y - self._dock[1]) < 0.2
            and self.posture == "sit"
        )
        if self._on_dock:
            self.battery = min(1.0, self.battery + dt / 2400.0)


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def iterate(
    brain: Any, world: SynthWorld, seconds: float, hz: float = 10.0, start_ms: int = 0
) -> Iterator[tuple[MotorCommand, Any]]:
    """Step ``brain`` in ``world`` for ``seconds``, yielding every (command, telemetry)."""
    dt_ms = int(round(1000.0 / hz))
    t = start_ms
    end = start_ms + int(seconds * 1000)
    while t < end:
        frame = world.frame(t - start_ms)
        cmd, tel = brain.step(t, frame)
        world.apply(cmd, dt_ms / 1000.0)
        yield cmd, tel
        t += dt_ms


def run(
    brain: Any, world: SynthWorld, seconds: float, hz: float = 10.0, start_ms: int = 0
) -> list[tuple[MotorCommand, Any]]:
    """Step ``brain`` in ``world`` for ``seconds``; returns every (command, telemetry)."""
    return list(iterate(brain, world, seconds, hz, start_ms))
