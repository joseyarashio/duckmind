"""A small kinematic stand-in world for the Act Runner's tests (not a physics claim).

It answers the runner the way the body loop and duck-eye would: robot.state (policy, the mouth
joint), the pose tap (odometry, stillness), eye.result for each look (the ``things`` task, in the
trunk frame, with seeded noise, only when the thing is in view for that head pose), and a few
effects: forward steps with a lag and a tail, ground_pick (2.85 s; a thing under the beak is
nudged, a fin in the grab window is held when the beak closes on it), kicks (a ball in the kick
window rolls along the kick direction), bumps (the duck's front pushes the thing).

The numbers loosely follow the phase-3 live measurements (creep ~42 mm, window positions); the
tests check the runner's logic and contracts, never these numbers.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from duckmind.curiosity import calib as C
from duckmind.curiosity.act_runner import CurioStep
from duckmind.duckbody.protocol import ActContext, ActOutput, CurioCmd, PoseSample, ProprioView
from duckmind.duckbody.robotd_client import RobotState
from duckmind.eye.protocol import EyeResult

TICK_MS = 20
MOUTH_CLOSED = -0.0873
HELD_MOUTH = 0.17  # ≈ +10° (the live jaw on the fin)


@dataclass
class Thing:
    label: str
    x: float  # mm, world (= odometry) frame
    y: float
    cls: str = "ball_slow"
    w: int = 70
    h: int = 70
    d: int = 70
    held: bool = False
    fixed: bool = False


@dataclass
class FakeWorld:
    thing: Thing
    seed: int = 0
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    light: str = "bright"
    lifted_at_ms: int | None = None
    fallen_at_ms: int | None = None
    see: bool = True
    noise_mm: float = 3.0
    t_ms: int = 1000
    v: float = 0.0  # m/s forward
    w: float = 0.0  # rad/s
    policy: str = "stand"
    skill: str | None = None
    skill_t: int = 0
    head: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    mouth_cmd: float = 0.0
    mouth: float = MOUTH_CLOSED
    pending: list[tuple[int, dict[str, Any]]] = field(default_factory=list)
    outputs: list[ActOutput] = field(default_factory=list)
    still_since: int = 0
    seq: int = 0
    rng: random.Random = field(default_factory=random.Random)
    kicked: bool = False
    pending_grip: bool = False
    stall_head_down: bool = False  # phase 1: 15 % of ducks barely walked head down
    kick_mm: float = 700.0
    look_log: list[bool] = field(default_factory=list)  # was the thing in view at each look

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    # ── the runner's view ──

    def ctx(self) -> ActContext:
        eye = []
        keep = []
        for due, result in self.pending:
            if due <= self.t_ms:
                eye.append(result)
            else:
                keep.append((due, result))
        self.pending = keep
        self.seq += 1
        joints = [0.0] * 15
        joints[5:9] = [0.3491 + self.head[0], 0.3491 + self.head[1], self.head[2], self.head[3]]
        joints[9] = self.mouth
        state = RobotState.from_wire(
            {
                "t": self.t_ms / 1000.0,
                "t_ns": self.t_ms * 1_000_000,
                "policy": self.policy,
                "move": {
                    "requested": [self.v, 0, self.w],
                    "applied": [self.v, 0, self.w],
                    "limited_by": [],
                },
                "safety": {"fallen": self.fallen, "limp": False},
                "joints": joints,
                "odom": {"position": [self.x / 1000.0, self.y / 1000.0, 0.0], "yaw": self.yaw},
            }
        )
        still = abs(self.v) < 0.005 and abs(self.w) < 0.01 and self.t_ms - self.still_since >= 150
        pose = PoseSample(
            seq=self.seq,
            t_ns=self.t_ms * 1_000_000,
            trunk_height_mm=120,
            camera_pos=None,
            camera_quat=None,
            tof_pos=None,
            tof_quat=None,
            imu_quat=None,
            gyro=None,
            head=self.head,
            head_vel=0.0,
            odom=(self.x / 1000.0, self.y / 1000.0, self.yaw),
            epoch=0,
            still=still,
            policy=self.policy,
        )
        proprio = ProprioView(
            posture="stand", policy=self.policy, lifted=self.lifted, fallen=self.fallen
        )
        return ActContext(
            t_ns=self.t_ms * 1_000_000,
            state=state,
            pose=pose,
            depth=None,
            proprio=proprio,
            light=self.light,
            eye=tuple(eye),
        )

    @property
    def lifted(self) -> bool:
        return self.lifted_at_ms is not None and self.t_ms >= self.lifted_at_ms

    @property
    def fallen(self) -> bool:
        return self.fallen_at_ms is not None and self.t_ms >= self.fallen_at_ms

    # ── geometry ──

    def to_trunk(self, x: float, y: float) -> tuple[float, float]:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        dx, dy = x - self.x, y - self.y
        return dx * c + dy * s, -dx * s + dy * c

    def to_world(self, a: float, side: float) -> tuple[float, float]:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return self.x + a * c - side * s, self.y + a * s + side * c

    def in_view(self, a: float, side: float) -> bool:
        pitch_near = 380.0
        if self.head[1] >= 0.3:
            pitch_near = 80.0
        elif self.head[1] >= 0.15:
            pitch_near = 180.0
        far = 900.0 if self.head[1] >= 0.3 else 1600.0 if self.head[1] >= 0.15 else 4000.0
        r = math.hypot(a, side)
        if r < pitch_near or r > far:
            return False
        return abs(math.atan2(side, a) - self.head[2]) <= 0.40

    # ── one tick ──

    def apply(self, out: ActOutput, runner: CurioStep) -> None:
        self.outputs.append(out)
        self.head = out.head or (0.0, 0.0, 0.0, 0.0)
        if out.mouth is not None:
            self.mouth_cmd = out.mouth
        for look in out.looks:
            self.pending.append((self.t_ms + 120, self._result(look.look_id)))
        if out.do is not None and self.policy in ("walk", "stand"):
            self.skill, self.skill_t, self.policy = out.do, self.t_ms, out.do
        self._step_body(out)
        self.t_ms += TICK_MS

    def _step_body(self, out: ActOutput) -> None:
        # the mouth servo: toward its target; stopped by a held fin
        target = MOUTH_CLOSED + max(0.0, min(1.0, self.mouth_cmd)) * (0.5236 - MOUTH_CLOSED)
        if self.thing.held and target < HELD_MOUTH:
            target = HELD_MOUTH
        self.mouth += (target - self.mouth) * 0.3
        if self.thing.held and self.mouth_cmd > 0.5:
            self.thing.held = False  # let go: it drops under the beak
        # skills
        if self.skill is not None:
            elapsed = self.t_ms - self.skill_t
            if self.skill == "ground_pick":
                if 1100 <= elapsed < 1120:
                    self._peck_bottom()
                if elapsed >= 2850:
                    self.skill, self.policy = None, "stand"
            else:
                if 150 <= elapsed < 170:
                    self._kick(self.skill)
                if elapsed >= 520:
                    self.skill, self.policy = None, "stand"
            return
        # walking: lagged speed, a tail after the stop
        vx, wz = out.twist or (0, 0)
        if self.stall_head_down and self.head[1] > 0.1:
            vx = 0  # this duck steps on the spot with its head down
        if vx != 0 and abs(vx) >= 300:
            # fitted to the live step table: 0.5 s → 42 mm, 1.0 s → ~121 mm (tails included)
            goal_v = 0.68 * vx / 1000.0 * (1.0 if vx > 0 else 0.28)
            goal_w = 0.8 * wz / 1000.0
            self.policy = "walk"
        else:
            goal_v, goal_w = 0.0, 0.0
        rise = 0.02 if goal_v != 0.0 else 0.167  # spin-up τ ≈ 1 s, tail τ ≈ 0.12 s
        self.v += (goal_v - self.v) * rise
        self.w += (goal_w - self.w) * rise
        if abs(self.v) < 0.002 and goal_v == 0.0:
            self.v = 0.0
        if abs(self.w) < 0.005 and goal_w == 0.0:
            self.w = 0.0
        if self.v == 0.0 and self.w == 0.0:
            if self.policy == "walk":
                self.policy = "stand"
                self.still_since = self.t_ms
        else:
            self.still_since = self.t_ms
        dt = TICK_MS / 1000.0
        self.yaw += self.w * dt
        self.x += 1000.0 * self.v * dt * math.cos(self.yaw)
        self.y += 1000.0 * self.v * dt * math.sin(self.yaw)
        if self.thing.held:
            self.thing.x, self.thing.y = self.to_world(90.0, 0.0)
        else:
            self._push()

    def _push(self) -> None:
        a, side = self.to_trunk(self.thing.x, self.thing.y)
        front = 40.0 + self.thing.d / 2.0
        if abs(side) < 40 and a < front and self.v > 0 and not self.thing.fixed:
            gap = front - a
            self.thing.x, self.thing.y = self.to_world(a + gap + 1.0, side)

    def _peck_bottom(self) -> None:
        a, side = self.to_trunk(self.thing.x, self.thing.y)
        grab = C.CALIB.grab
        if (
            self.thing.cls == "handle_toy"
            and grab.contains(int(a), int(side))
            and self.mouth_cmd > 0.5
        ):
            self.pending_grip = True
            return
        if C.CALIB.peck.contains(int(a), int(side)) and not self.thing.fixed:
            self.thing.x, self.thing.y = self.to_world(a + 35.0, side)

    def close_grip(self) -> None:
        if self.pending_grip and self.mouth_cmd < 0.5:
            self.thing.held = True
            self.pending_grip = False

    def _kick(self, skill: str) -> None:
        a, side = self.to_trunk(self.thing.x, self.thing.y)
        foot = 1 if skill == "kick_left" else -1
        if C.CALIB.kick_window.contains(int(a), int(side * foot)) and not self.thing.fixed:
            d = self.yaw + C.CALIB.kick_dir_mrad["left" if foot > 0 else "right"] / 1000.0
            self.thing.x += self.kick_mm * math.cos(d)
            self.thing.y += self.kick_mm * math.sin(d)
            self.kicked = True

    def _result(self, look_id: str) -> dict[str, Any]:
        things: list[dict[str, Any]] = []
        a, side = self.to_trunk(self.thing.x, self.thing.y)
        self.look_log.append(self.in_view(a, side))
        # duck-eye's things task names nothing in dark or black frames (the camera stays honest)
        lit = self.light in ("bright", "dim")
        if self.see and lit and not self.thing.held and self.in_view(a, side):
            n = self.noise_mm
            things.append(
                {
                    "label": self.thing.label,
                    "cls": self.thing.cls,
                    "x_mm": int(round(a + self.rng.gauss(0.0, n))),
                    "y_mm": int(round(side + self.rng.gauss(0.0, n))),
                    "w_mm": self.thing.w,
                    "h_mm": self.thing.h,
                    "d_mm": self.thing.d,
                    "conf_ppm": 900_000,
                }
            )
        result = EyeResult(
            look_id=look_id,
            attempt=0,
            kind="fixate",
            status="ok",
            pv="p:0123456789abcdef",
            captured_at_ns=self.t_ms * 1_000_000,
            results={
                "light": {"level": self.light, "luma": 90, "noise_q": 2},
                "things": {"things": things},
            },
        )
        return result.to_json()


def run(
    world: FakeWorld, command: CurioCmd, *, limit_s: float = 90.0, **kwargs: Any
) -> tuple[CurioStep, ActOutput, list[ActOutput]]:
    runner = CurioStep(command, **kwargs)
    ticks = int(limit_s * 1000 / TICK_MS)
    for _ in range(ticks):
        ctx = world.ctx()
        out = runner.step(ctx)
        world.apply(out, runner)
        world.close_grip()
        if out.done:
            return runner, out, world.outputs
    raise AssertionError(f"the step did not end in {limit_s} s (phase {runner.phase})")


def command(verb: str, thing: str = "ball", *, step: int = 1, **params: Any) -> CurioCmd:
    return CurioCmd(
        act_id=f"C{step}",
        deadline_ms=10**9,
        phase="step",
        thing=thing,
        step=step,
        params={"verb": verb, **params},
    )


def step_event(out: ActOutput) -> Mapping[str, Any]:
    events = [p for p in out.percepts if p.get("kind") == "body.curio_step"]
    assert len(events) == 1, out.percepts
    return events[0]
