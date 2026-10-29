"""Scaffolding for the live seam (neuro:live) until the senses, brain and motor lanes land.

NOT the neural duck. Three stand-ins with the contract's shapes (:mod:`duckmind.neuro.contract`),
so the live driver (:mod:`duckmind.simweb.neuro_live`) can drive the physical body end to end
and the seam can be proven in live physics before the real lanes exist:

* :func:`body_frame` — a :class:`~duckmind.neuro.contract.SensorFrame` from the real doors only
  (robotd ``robot.state`` + duck-body's proprio + the power reading): no camera, sound or touch.
* :class:`StubBrain` — a seeded wanderer (sit, stand, walk in bursts with varying speed and arcs,
  pause and look around, chirp now and then). It implements the contract's ``Brain`` protocol
  (``step(t_ms, frame) -> (MotorCommand, BrainTelemetry)``) and says ``stub`` everywhere.
* :class:`StubMotor` — MotorCommand → :class:`~duckmind.neuro.bodylink.BodyRequest` (the twist
  as given, the head from the reflex pose and the gaze, a beak pulse for each call).

Pure and deterministic for a seed and an input stream; standard library + duck-body/duckcore
content only (never :mod:`duckmind.world`).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from duckmind.duckbody.protocol import ProprioView
from duckmind.duckbody.robotd_client import RobotState
from duckmind.i18n import msg
from duckmind.neuro import contract as C
from duckmind.neuro.bodylink import BodyRequest, Offsets, head_offsets_of, mouth_open_of

PANIC_LABELS = frozenset(
    {"panic_run", "panic_arc_left", "panic_arc_right", "quick_turn_left", "quick_turn_right"}
)  # skilltrain.guard.PANIC_ENTRIES + QUICK_TURN_ENTRIES (both run upright and moving)
STUB_NOTE_ZH = msg("neuro.lane.stub_brain")
QUIET_LEG_LOAD = 35.0  # duck-body's floor band for a duck standing still (sim leg-force proxy)


def _posture(view: ProprioView | None, state: RobotState) -> str:
    if state.fallen or (view is not None and view.fallen):
        return "unknown"
    base = (state.policy or "").split(":", 1)[0]
    if base in PANIC_LABELS:
        return "walk"  # the panic network runs the flight (a robotd skill): the duck is walking
    if base == "skill_abort":
        return "stand"  # its one tick of v5 under a zero command, ending a skill
    if state.policy == "walk" and abs(state.applied[0]) > 0.05:
        return "walk"
    if state.policy in ("walk", "stand"):
        return "stand"
    if state.policy == "sit":
        return "sit"
    if state.policy in ("rise", "held", "sitting", "homing"):
        return "transition"  # (robotd's sit/stand machine runs: the same set the world's own sense maps)
    return "unknown"


def body_frame(
    t_ms: int,
    state: RobotState,
    view: ProprioView | None = None,
    *,
    events: tuple[str, ...] = (),
    battery: float | None = None,
    on_dock: bool | None = None,
    charging: bool | None = None,
    day_minute: float = 600.0,
) -> C.SensorFrame:
    """A SensorFrame from the real doors only (stand-in for neuro:senses). ``battery``/
    ``on_dock``/``charging`` are the world's modelled power readings when known."""
    down = state.down or (0.0, 0.0, -1.0)
    tilt = state.tilt_rad or 0.0
    roll = math.atan2(-down[1], -down[2]) if down[2] < 0 else 0.0
    pitch = math.atan2(down[0], -down[2]) if down[2] < 0 else 0.0
    load = state.leg_load
    sources = dict(C.CHANNEL_SOURCE)
    sources.update(tof="none", camera="none", sound="none", touch="none")
    if battery is None:
        sources["power"] = "none"
    return C.SensorFrame(
        t_ms=int(t_ms),
        body=C.BodySense(
            posture=_posture(view, state),
            fallen=bool(state.fallen),
            limp=bool(state.limp),
            tilt=float(tilt),
            roll=float(roll),
            pitch=float(pitch),
            gyro=state.gyro or (0.0, 0.0, 0.0),
            head=state.head,
            vx=float(state.applied[0]),
            vyaw=float(state.applied[2]),
            odom=(state.odom_pos[0], state.odom_pos[1], state.odom_yaw),
            leg_load=None if load is None else load / QUIET_LEG_LOAD,
        ),
        events=events,
        power=C.PowerSense(
            battery=0.8 if battery is None else float(battery),
            on_dock=bool(on_dock),
            charging=bool(charging),
            source="model" if battery is not None else "none",
        ),
        camera=C.CameraSense(usable=False, source="none"),
        day_minute=float(day_minute),
        sources=sources,
    )


class _Ou:
    """An Ornstein–Uhlenbeck process (mean-reverting noise), stepped by dt seconds."""

    def __init__(self, rng: random.Random, tau_s: float, sigma: float, mean: float = 0.0) -> None:
        self.rng, self.tau, self.sigma, self.mean = rng, tau_s, sigma, mean
        self.x = mean

    def step(self, dt: float) -> float:
        a = math.exp(-dt / self.tau)
        self.x = (
            self.mean
            + (self.x - self.mean) * a
            + self.sigma * math.sqrt(1 - a * a) * self.rng.gauss(0.0, 1.0)
        )
        return self.x


@dataclass
class StubBrain:
    """A seeded wanderer (see the module doc). Not the neural duck: only proves the seam."""

    seed: int = 7
    name: str = "stub"
    behaviour: str = "rest_sit"
    bout: int = 0
    bout_start_ms: int = 0
    bout_len_ms: int = 4000
    counts: dict[str, int] = field(default_factory=dict)
    vocal_seq: int = 0
    last_ms: int | None = None

    def __post_init__(self) -> None:
        self.rng = random.Random(f"neuro-stub:{self.seed}")
        self.speed = _Ou(self.rng, 2.5, 0.08, 0.45)
        self.curve = _Ou(self.rng, 1.8, 0.45, 0.0)
        self.gaze_yaw = _Ou(self.rng, 1.2, 0.35, 0.0)
        self.gaze_pitch = _Ou(self.rng, 1.5, 0.12, 0.0)
        self.next_call_ms = 6000

    def _start(self, t_ms: int, behaviour: str, lo_s: float, hi_s: float) -> None:
        self.behaviour = behaviour
        self.bout += 1
        self.bout_start_ms = t_ms
        self.bout_len_ms = int(1000 * self.rng.uniform(lo_s, hi_s))
        self.counts[behaviour] = self.counts.get(behaviour, 0) + 1

    def _next(self, t_ms: int, frame: C.SensorFrame) -> None:
        posture = frame.body.posture
        roll = self.rng.random()
        if self.behaviour in ("wander", "turn_around"):
            self._start(t_ms, "look_around" if roll < 0.7 else "sit_down", 2.0, 5.0)
        elif self.behaviour == "sit_down":
            self._start(t_ms, "rest_sit", 4.0, 12.0)
        elif self.behaviour == "rest_sit":
            self._start(t_ms, "stand_up", 2.5, 4.0)
        elif self.behaviour == "stand_up" or posture in ("stand", "walk"):
            self._start(t_ms, "wander" if roll < 0.8 else "turn_around", 3.0, 9.0)
        else:
            self._start(t_ms, "stand_up", 2.5, 4.0)

    def step(self, t_ms: int, frame: C.SensorFrame) -> tuple[C.MotorCommand, C.BrainTelemetry]:
        dt = 0.02 if self.last_ms is None else max(0.0, (t_ms - self.last_ms) / 1000.0)
        self.last_ms = t_ms
        if frame.body.fallen or "fell" in frame.events:
            self.behaviour = "righting"
        elif self.behaviour == "righting":
            self._start(t_ms, "look_around", 2.0, 4.0)
        elif t_ms - self.bout_start_ms >= self.bout_len_ms:
            self._next(t_ms, frame)
        speed = self.speed.step(dt)
        curve = self.curve.step(dt)
        gy = self.gaze_yaw.step(dt)
        gp = self.gaze_pitch.step(dt)
        vx = vyaw = 0.0
        posture = "sit" if self.behaviour in ("rest_sit", "sit_down") else "stand"
        if self.behaviour in ("wander", "turn_around"):
            # bursts and pauses: a slow rhythm gates the walking
            burst = math.sin(2 * math.pi * (t_ms - self.bout_start_ms) / 4200.0) > -0.35
            if burst:
                vx = min(0.55, max(0.36, speed))
                vyaw = max(
                    -1.0, min(1.0, curve + (0.8 if self.behaviour == "turn_around" else 0.0))
                )
            vx, vyaw = C.legal_walk(vx, vyaw)
        vocal = None
        if t_ms >= self.next_call_ms and self.behaviour not in ("righting",):
            vocal = "chirp" if self.rng.random() < 0.7 else "inquire"
            self.vocal_seq += 1
            self.next_call_ms = t_ms + int(1000 * self.rng.uniform(8.0, 25.0))
        cmd = C.MotorCommand(
            t_ms=t_ms,
            behaviour=self.behaviour,
            bout=self.bout,
            variant=self.rng.randrange(1 << 30) if vocal else self.bout,
            vx=vx,
            vyaw=vyaw,
            posture=posture,
            gaze=C.Gaze(bearing=0.5 * gy, elevation=0.25 * gp, gain=0.6, speed=0.4),
            alive=0.6,
            vocal=vocal if vocal else None,
            vocal_seq=self.vocal_seq,
        )
        telemetry = C.BrainTelemetry(
            t_ms=t_ms,
            behaviour=self.behaviour,
            behaviour_zh=C.behaviour_zh(self.behaviour),
            reason_zh=STUB_NOTE_ZH,
            bout=self.bout,
            bout_ms=t_ms - self.bout_start_ms,
            satiation=min(1.0, (t_ms - self.bout_start_ms) / max(1, self.bout_len_ms)),
            drives={},
            arousal=0.0,
            valence=0.0,
            reflex=None,
            reflexes=(),
            competition=(),
            habituation={},
            sensitisation={},
            learned={},
            locomotion={"speed": vx, "curvature": vyaw},
            stage="stub",
            counts=dict(self.counts),
        )
        return cmd, telemetry


@dataclass
class StubMotor:
    """MotorCommand → BodyRequest (stand-in for neuro:motor): the twist as given, the head from
    the reflex pose blended over the gaze plus a little life, and a short beak pulse per call."""

    call_open_s: float = 0.18
    vocal_seq: int = 0
    mouth_until_ms: int = 0
    phase: float = 0.0

    def step(
        self, t_ms: int, cmd: C.MotorCommand, frame: C.SensorFrame | None = None
    ) -> tuple[BodyRequest, list[dict[str, Any]]]:
        vocals: list[dict[str, Any]] = []
        if cmd.vocal and cmd.vocal_seq != self.vocal_seq:
            self.vocal_seq = cmd.vocal_seq
            self.mouth_until_ms = t_ms + int(self.call_open_s * 1000)
            vocals.append({"tag": cmd.vocal, "level": cmd.vocal_level, "behaviour": cmd.behaviour})
        self.phase += 0.02
        life = 0.08 * cmd.alive * math.sin(2 * math.pi * self.phase / 3.1)
        yaw = max(-1.0, min(1.0, cmd.gaze.bearing / 0.6)) * cmd.gaze.gain
        pitch = max(-1.0, min(1.0, cmd.gaze.elevation / 0.35)) * cmd.gaze.gain
        pose = {"np": 0.0, "hp": pitch + life, "hy": yaw, "hr": 0.0}
        if cmd.head is not None and cmd.head_blend > 0.0:
            w = max(0.0, min(1.0, cmd.head_blend))
            reflex = {"np": cmd.head.np, "hp": cmd.head.hp, "hy": cmd.head.hy, "hr": cmd.head.hr}
            pose = {k: (1 - w) * pose[k] + w * reflex[k] for k in pose}
        head: Offsets = head_offsets_of(pose, side=1)
        mouth = mouth_open_of(cmd.head.mouth if cmd.head is not None else 0.0)
        if t_ms < self.mouth_until_ms:
            mouth = max(mouth, mouth_open_of(0.8))
        request = BodyRequest(
            vx=cmd.vx,
            vyaw=cmd.vyaw,
            head=head,
            mouth=mouth,
            posture=cmd.posture,
            approach=cmd.approach,
            docking=cmd.behaviour == "return_dock",
        )
        return request, vocals


__all__ = ["STUB_NOTE_ZH", "StubBrain", "StubMotor", "body_frame"]
