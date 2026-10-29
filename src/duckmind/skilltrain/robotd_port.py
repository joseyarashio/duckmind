"""robotd's controller and tick, in Python, with the one-shot skill path (Stage 1 gates).

A line-by-line port of ``robotd/src/control.rs`` (``Controller``: ``start_skill``,
``sit_toggle``, ``begin_boot_rise``, ``step``) and of the parts of ``robotd/src/main.rs`` around it
that decide what a skill sees (read-only upstream, microduck at the pinned build):

* **Per tick, in main.rs order:** the pending requests are taken once (the sit toggle, then the
  configured skills in index order — a *mask*, so a second request of the same skill in one
  tick is nothing extra); the client's twist is gated by the deadman (500 ms) and smoothed
  (``cmd_alpha`` 0.2), the head smoothed (``head_alpha``, robotd default 0.2; the neural duck's
  simulator runs 1.0); then ``Controller.step``; then the ±π actuator clamp
  (``duck-control`` ``safety.rs``). The EMAs keep integrating the client's command while a
  skill runs, so walking resumes from them when it hands back.
* **Controller.step:** windows expire first — a Holding skill with ``unwind_s > 0`` goes to
  Unwinding, otherwise a chaining skill whose chain counter is still positive restarts (a new
  episode: recurrent memory reset), otherwise it ends; a finished rise becomes Up. Then the
  network: skill > (ground pick, not modelled) > Sitting (sitstand, twist [1,0,0], head and body
  slots live) > Rising (sitstand, twist 0) > Up (stand when the stand slot exists and
  |twist| ≤ 0.05, else walk). A skill's command block is its constant ``command`` (``unwind``
  while unwinding) with head and body **zero**. Scale and gain: a skill's own
  ``params`` else walking tuning (0.9, gain 200) — standing tuning only with a stand network;
  sitstand 1.0; stand 1.0 × gain 0.8. ``targets = HOME + scale · scale_mult · action``, then
  the low-pass (head 0.5, legs 0.7) against the previous targets. The previous raw action is
  shared across networks (a skill's first observation carries the walk network's last
  output). Windows advance after the tick that used them.
* **start_skill:** the same chaining skill → refresh its chain counter (0.15 s); a
  non-chaining request while any skill runs → refused; a chaining one pre-empts whatever runs.
  Not gated on posture: a seated duck keeps ``Sitting`` and is re-seated when the skill ends.

Recurrent (``model_api`` 2) policies follow ``docs/recurrent-policies.md``: memory is zero on
activation, on every network switch and on a chained restart. Not modelled: the ground pick
(not a Stage-1 concern), limp-fall (off in the v5 set and blocked by any running skill), the
chorale head sway, pose mode (duck-body forbids ``robot.pose`` in Stage 1).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from duckmind.skilltrain.table import CHAIN_WINDOW_S, STANDING_THRESHOLD, SkillDef
from duckmind.world.headless_duck import HEAD14, HOME14

CONTROL_HZ = 50
DT = 1.0 / CONTROL_HZ
CMD_ALPHA = 0.2
HEAD_ALPHA_DEFAULT = 0.2
DEADMAN_S = 0.5
ACTUATOR_MIN, ACTUATOR_MAX = -math.pi, math.pi
NOMINAL_V = 7.4
VIN_CLAMP = (6.0, 9.5)
SITSTAND_RISE_S = 1.0
OBS_LEN, ACTION_LEN = 61, 14
HOME = np.array(HOME14, dtype=float)
HEAD_MASK = np.zeros(14, dtype=bool)
HEAD_MASK[list(HEAD14)] = True

WALK, STAND, SITSTAND = "walk", "stand", "sitstand"


@dataclass(frozen=True)
class Tuning:
    """``control.rs`` ``Tuning::default()`` (the prototype's alpha configuration)."""

    action_scale: float = 0.9
    standing_action_scale: float = 1.0
    standing_gain_ratio: float = 0.8
    gain: int = 200
    head_lowpass: float | None = 0.5
    legs_lowpass: float | None = 0.7


# ── networks ─────────────────────────────────────────────────────────────────────────────────

_SESSIONS: dict[str, Any] = {}


def _session(path: Path) -> Any:
    key = str(Path(path).resolve())
    if key not in _SESSIONS:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        _SESSIONS[key] = ort.InferenceSession(
            key, sess_options=opts, providers=["CPUExecutionProvider"]
        )
    return _SESSIONS[key]


class OnnxNet:
    """One policy file, per duck: feed-forward (``obs [1,61]`` → one ``[1,14]`` output) or the
    explicit-state LSTM export (``obs``, ``h_in``, ``c_in`` → ``actions``, ``h_out``,
    ``c_out``), inputs and outputs matched by name as duck-control's loader does."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.session = _session(self.path)
        inputs = {i.name: i for i in self.session.get_inputs()}
        outputs = [o.name for o in self.session.get_outputs()]
        self.recurrent = len(inputs) == 3 and {"obs", "h_in", "c_in"} <= set(inputs)
        if self.recurrent:
            if not {"actions", "h_out", "c_out"} <= set(outputs):
                raise ValueError(f"{self.path}: an LSTM export needs actions/h_out/c_out outputs")
            self.obs_name = "obs"
            self.out_names = ["actions", "h_out", "c_out"]
            shape = [int(d) if isinstance(d, int) else 1 for d in inputs["h_in"].shape]
            self._state_shape = tuple(shape)
        else:
            if len(inputs) != 1 or len(outputs) != 1:
                raise ValueError(f"{self.path}: expected obs → actions or obs/h_in/c_in (LSTM)")
            self.obs_name = next(iter(inputs))
            self.out_names = outputs
            self._state_shape = ()
        self.h: np.ndarray | None = None
        self.c: np.ndarray | None = None
        self.reset()

    def reset(self) -> None:
        if self.recurrent:
            self.h = np.zeros(self._state_shape, dtype=np.float32)
            self.c = np.zeros(self._state_shape, dtype=np.float32)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32).reshape(1, OBS_LEN)
        if self.recurrent:
            action, h, c = self.session.run(
                self.out_names, {"obs": x, "h_in": self.h, "c_in": self.c}
            )
            out = np.asarray(action, dtype=np.float32).reshape(ACTION_LEN)
            if not np.all(np.isfinite(out)):
                self.reset()  # failed inference clears memory (recurrent-policies.md)
                raise FloatingPointError(f"{self.path.name}: non-finite action")
            self.h, self.c = np.asarray(h, np.float32), np.asarray(c, np.float32)
            return out
        (action,) = self.session.run(self.out_names, {self.obs_name: x})
        out = np.asarray(action, dtype=np.float32).reshape(ACTION_LEN)
        if not np.all(np.isfinite(out)):
            raise FloatingPointError(f"{self.path.name}: non-finite action")
        return out


class PolicySet:
    """``duck_control::policy::Policy``: the slots and the skill networks, with the recurrent
    memory rules. v5 runs ``stand = "none"`` (no standing network)."""

    def __init__(
        self,
        walk: str | Path,
        sitstand: str | Path | None,
        skills: Sequence[SkillDef],
        stand: str | Path | None = None,
    ) -> None:
        self.walk = OnnxNet(walk)
        self.stand = OnnxNet(stand) if stand is not None else None
        self.sitstand = OnnxNet(sitstand) if sitstand is not None else None
        self.skills: list[OnnxNet] = []
        for d in skills:
            if d.path is None:
                raise ValueError(f"skill {d.name!r} has no path")
            self.skills.append(OnnxNet(d.path))
        self.active: tuple[str, int] | None = None

    def will_stand(self, twist_magnitude: float) -> bool:
        return self.stand is not None and twist_magnitude <= STANDING_THRESHOLD

    def has_standing(self) -> bool:
        return self.stand is not None

    def net(self, key: tuple[str, int]) -> OnnxNet:
        kind, index = key
        if kind == "skill":
            return self.skills[index]
        found = {WALK: self.walk, STAND: self.stand, SITSTAND: self.sitstand}[kind]
        if found is None:
            raise RuntimeError(f"no {kind} network loaded")
        return found

    def infer(self, obs: np.ndarray, key: tuple[str, int]) -> np.ndarray:
        net = self.net(key)
        if key != self.active:
            net.reset()  # memory starts at zero on activation and on every switch
            self.active = key
        return net(obs)

    def reset(self) -> None:
        for net in (self.walk, self.stand, self.sitstand, *self.skills):
            if net is not None:
                net.reset()


# ── the controller (control.rs) ──────────────────────────────────────────────────────────────


@dataclass
class ActiveSkill:
    index: int
    phase: str  # "holding" | "unwinding"
    remaining: float
    chain: float = 0.0


@dataclass(frozen=True)
class Sensors:
    """What robotd reads each tick, in the 14 policy joints' order (the mouth left out)."""

    gyro: np.ndarray  # trunk frame, rad/s
    gravity: np.ndarray  # projected gravity, trunk frame
    q: np.ndarray  # 14 joint positions
    qd: np.ndarray  # 14 joint velocities


@dataclass(frozen=True)
class Command:
    twist: tuple[float, float, float] = (0.0, 0.0, 0.0)
    head: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    body: tuple[float, float, float] = (0.0, 0.0, 0.0)  # z, roll, pitch

    def twist_magnitude(self) -> float:
        return math.sqrt(sum(v * v for v in self.twist))


@dataclass(frozen=True)
class Step:
    targets: np.ndarray  # 14, before the ±π clamp
    label: str  # walk | stand | sit | rise | <skill> | <skill>:unwind
    gain: int
    busy: bool
    net: tuple[str, int]
    scale: float
    command: Command  # the command block the network saw


class Controller:
    def __init__(
        self,
        policy: PolicySet,
        skills: Sequence[SkillDef],
        tuning: Tuning = Tuning(),
        sitstand_rise_s: float = SITSTAND_RISE_S,
    ) -> None:
        if len(skills) != len(policy.skills):
            raise ValueError("one network per skill entry")
        self.policy = policy
        self.skills = list(skills)
        self.tuning = tuning
        self.sitstand_rise_s = float(sitstand_rise_s)
        self.last_action = np.zeros(ACTION_LEN, dtype=np.float32)
        self.previous: np.ndarray | None = None
        self.active: ActiveSkill | None = None
        self.sit = "up"  # up | sitting | rising
        self.rise_remaining = 0.0
        self.last_net: tuple[str, int] | None = None
        self.events: list[dict[str, Any]] = []  # skill starts, refreshes, refusals, phases

    # ── requests ──
    def skill_names(self) -> list[str]:
        return [d.name for d in self.skills]

    def start_skill(self, index: int) -> tuple[bool | None, str]:
        """``Ok(True)`` = started, ``Ok(False)`` = a running chaining skill refreshed,
        ``(None, reason)`` = refused."""
        if not 0 <= index < len(self.skills):
            return None, "no such skill on this robot"
        d = self.skills[index]
        if self.active is not None:
            if self.active.index == index and d.chain:
                self.active.chain = CHAIN_WINDOW_S
                return False, "chain refreshed"
            if not d.chain:
                return None, "a scripted move is already running"
        self.active = ActiveSkill(index=index, phase="holding", remaining=d.duration)
        return True, "started"

    def sit_toggle(self) -> tuple[bool, str]:
        if self.sit == "up":
            if self.policy.sitstand is None:
                return False, "no sitstand policy loaded"
            self.sit = "sitting"
            return True, "sit"
        if self.sit == "sitting":
            self.sit, self.rise_remaining = "rising", self.sitstand_rise_s
            return True, "stand"
        return False, "already standing up"

    def begin_boot_rise(self) -> None:
        self.sit, self.rise_remaining = "rising", self.sitstand_rise_s

    def busy(self) -> bool:
        return self.active is not None or self.sit == "rising"

    def reset(self) -> None:
        self.policy.reset()
        self.last_action = np.zeros(ACTION_LEN, dtype=np.float32)
        self.previous = None

    # ── one tick ──
    def step(
        self,
        sensors: Sensors,
        command: Command,
        body_active: bool = False,
        dt: float = DT,
        scale_mult: float = 1.0,
        now: float | None = None,
    ) -> Step:
        # 1. windows expire first
        a = self.active
        if a is not None and a.remaining <= 0.0:
            d = self.skills[a.index]
            if a.phase == "holding" and d.unwind_s > 0.0:
                self.active = ActiveSkill(a.index, "unwinding", d.unwind_s, a.chain)
                self._log(now, "unwind", d.name)
            else:
                chains = a.phase == "holding" and d.chain and a.chain > 0.0
                if chains:
                    self.policy.reset()  # a chained replay is a new episode
                    self.active = ActiveSkill(a.index, "holding", d.duration, 0.0)
                    self._log(now, "chain_restart", d.name)
                else:
                    self.active = None
                    self._log(now, "ended", d.name)
        if self.sit == "rising" and self.rise_remaining <= 0.0:
            self.sit = "up"

        # 2. the network and its command block
        net: tuple[str, int]
        if self.active is not None:
            d = self.skills[self.active.index]
            holding = self.active.phase == "holding"
            effective = Command(twist=d.command if holding else d.unwind)
            net = ("skill", self.active.index)
            label = d.name if holding else f"{d.name}:unwind"
        elif self.sit == "sitting":
            effective = Command((1.0, 0.0, 0.0), command.head, command.body)
            net, label = (SITSTAND, 0), "sit"
        elif self.sit == "rising":
            effective = Command((0.0, 0.0, 0.0), command.head, command.body)
            net, label = (SITSTAND, 0), "rise"
        else:
            effective = command
            if body_active:
                effective = Command((0.0, 0.0, 0.0), command.head, command.body)
            standing = self.policy.will_stand(effective.twist_magnitude()) or (
                body_active and self.policy.has_standing()
            )
            net, label = ((STAND, 0), "stand") if standing else ((WALK, 0), "walk")
        self.last_net = net

        # 3. inference
        obs = np.concatenate(
            [
                np.asarray(sensors.gyro, dtype=float),
                np.asarray(sensors.gravity, dtype=float),
                np.asarray(sensors.q, dtype=float) - HOME,
                np.asarray(sensors.qd, dtype=float),
                self.last_action.astype(float),
                np.asarray(effective.twist, dtype=float),
                np.asarray(effective.head, dtype=float),
                np.array([0.0, 0.0, *effective.body, 0.0]),
            ]
        ).astype(np.float32)
        action = self.policy.infer(obs, net)
        self.last_action = action

        # 4. scale and gain from the active state
        will_stand = self.policy.will_stand(effective.twist_magnitude())
        kind = net[0]
        standing_tuned = kind == STAND or (kind in ("skill", SITSTAND) and will_stand)
        t = self.tuning
        if kind == "skill":
            d = self.skills[net[1]]
            scale = (
                d.action_scale
                if d.action_scale is not None
                else (t.standing_action_scale if standing_tuned else t.action_scale)
            )
            ratio = (
                d.gain_ratio
                if d.gain_ratio is not None
                else (t.standing_gain_ratio if standing_tuned else 1.0)
            )
            gain = int(round(t.gain * ratio))
        elif kind == SITSTAND:
            scale = 1.0
            gain = int(round(t.gain * t.standing_gain_ratio)) if standing_tuned else t.gain
        elif standing_tuned:
            scale, gain = t.standing_action_scale, int(round(t.gain * t.standing_gain_ratio))
        else:
            scale, gain = t.action_scale, t.gain
        scale *= scale_mult

        targets = HOME + scale * action.astype(float)
        if self.previous is not None:
            if t.head_lowpass is not None:
                al = t.head_lowpass
                targets[HEAD_MASK] = al * targets[HEAD_MASK] + (1 - al) * self.previous[HEAD_MASK]
            if t.legs_lowpass is not None:
                al = t.legs_lowpass
                legs = ~HEAD_MASK
                targets[legs] = al * targets[legs] + (1 - al) * self.previous[legs]
        self.previous = targets.copy()

        # 5. windows advance after the tick that used them
        if self.active is not None:
            self.active.remaining -= dt
            self.active.chain = max(self.active.chain - dt, 0.0)
        if self.sit == "rising":
            self.rise_remaining -= dt
        return Step(targets, label, gain, self.busy(), net, scale, effective)

    def _log(self, now: float | None, what: str, name: str) -> None:
        self.events.append(
            {"t": None if now is None else round(now, 4), "kind": what, "skill": name}
        )


# ── the tick around it (main.rs) ─────────────────────────────────────────────────────────────


@dataclass
class Intents:
    twist: tuple[float, float, float] = (0.0, 0.0, 0.0)
    twist_at: float = -math.inf
    head: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    pose: tuple[float, float, float] = (0.0, 0.0, 0.0)  # robot.pose z, roll, pitch
    pose_active: bool = False
    sit_toggle: bool = False
    skills: set[int] = field(default_factory=set)


class Robotd:
    """The daemon's tick for one simulated duck: intents in, one :class:`Step` out."""

    def __init__(
        self,
        policy: PolicySet,
        skills: Sequence[SkillDef],
        *,
        head_alpha: float = HEAD_ALPHA_DEFAULT,
        cmd_alpha: float = CMD_ALPHA,
        tuning: Tuning = Tuning(),
        voltage_adapt: bool = True,
        nominal_v: float = NOMINAL_V,
        pose_zeroes_twist: bool = True,
    ) -> None:
        if not 0.0 < head_alpha <= 1.0:
            raise ValueError("head_alpha must be in (0, 1]")
        self.controller = Controller(policy, skills, tuning)
        self.head_alpha = float(head_alpha)
        self.cmd_alpha = float(cmd_alpha)
        self.voltage_adapt = voltage_adapt
        self.nominal_v = float(nominal_v)
        self.intents = Intents()
        self.twist_ema = np.zeros(3)
        self.head_ema = np.zeros(4)
        self.body_ema = np.zeros(3)
        self.pose_zeroes_twist = bool(
            pose_zeroes_twist
        )  # robotd does; a gate may ask for a walk with a pose
        self.ready = False
        self.last: Step | None = None

    # ── the client's side ──
    def move(self, vx: float, vy: float, vyaw: float, now: float) -> None:
        self.intents.twist = (float(vx), float(vy), float(vyaw))
        self.intents.twist_at = float(now)

    def head(self, offsets: Sequence[float]) -> None:
        values = [float(v) for v in offsets]
        if len(values) != 4:
            raise ValueError("robot.head takes 4 values")
        self.intents.head = (values[0], values[1], values[2], values[3])

    def pose(self, z: float, roll: float, pitch: float, active: bool = True) -> None:
        """``robot.pose``: the standing body pose (robotd smooths it at cmd_alpha while held, snaps it to 0 on release)."""
        self.intents.pose = (float(z), float(roll), float(pitch))
        self.intents.pose_active = bool(active)

    def request(self, name: str) -> tuple[bool, str]:
        """``robot.do`` as a request: refused for what the dispatcher can know (not driving,
        an unknown name); otherwise queued for the next tick, where the controller arbitrates."""
        if not self.ready:
            return False, "the policy is not driving"
        if name == "sit_toggle":
            self.intents.sit_toggle = True
            return True, "queued"
        names = self.controller.skill_names()
        if name not in names:
            return False, f"no skill named {name!r}; this robot has {', '.join(names)}"
        self.intents.skills.add(names.index(name))
        return True, "queued"

    def boot(self, seated: bool) -> None:
        """Torque on: a seated boot rises through sitstand; otherwise ready at once (the 2 s
        home ramp is the caller's)."""
        self.ready = True
        if seated and self.controller.policy.sitstand is not None:
            self.controller.begin_boot_rise()

    # ── one tick ──
    def tick(self, sensors: Sensors, now: float, battery_v: float = NOMINAL_V) -> Step:
        c = self.controller
        if self.ready:
            if self.intents.sit_toggle:
                ok, what = c.sit_toggle()
                c.events.append({"t": round(now, 4), "kind": "sit_toggle", "ok": ok, "what": what})
            for index in sorted(self.intents.skills):
                started, why = c.start_skill(index)
                if started is not False:  # a refresh is quiet, as robotd's journal is
                    c.events.append(
                        {
                            "t": round(now, 4),
                            "kind": "started" if started else "refused",
                            "skill": c.skills[index].name,
                            "why": why,
                        }
                    )
                else:
                    c.events.append(
                        {"t": round(now, 4), "kind": "refreshed", "skill": c.skills[index].name}
                    )
        self.intents.sit_toggle = False
        self.intents.skills = set()
        stale = now - self.intents.twist_at > DEADMAN_S + 1e-9
        target = np.zeros(3) if stale else np.array(self.intents.twist, dtype=float)
        self.twist_ema += self.cmd_alpha * (target - self.twist_ema)
        self.head_ema += self.head_alpha * (np.array(self.intents.head, float) - self.head_ema)
        if self.intents.pose_active:
            self.body_ema += self.cmd_alpha * (np.array(self.intents.pose, float) - self.body_ema)
        else:
            self.body_ema = np.zeros(3)
        command = Command(
            body=(float(self.body_ema[0]), float(self.body_ema[1]), float(self.body_ema[2])),
            twist=(float(self.twist_ema[0]), float(self.twist_ema[1]), float(self.twist_ema[2])),
            head=(
                float(self.head_ema[0]),
                float(self.head_ema[1]),
                float(self.head_ema[2]),
                float(self.head_ema[3]),
            ),
        )
        mult = 1.0
        if self.voltage_adapt and battery_v > 0.0:
            mult = self.nominal_v / min(max(battery_v, VIN_CLAMP[0]), VIN_CLAMP[1])
        step = c.step(
            sensors,
            command,
            body_active=self.intents.pose_active and self.pose_zeroes_twist,
            dt=DT,
            scale_mult=mult,
            now=now,
        )
        clamped = np.clip(step.targets, ACTUATOR_MIN, ACTUATOR_MAX)
        self.last = Step(
            clamped, step.label, step.gain, step.busy, step.net, step.scale, step.command
        )
        return self.last


__all__ = [
    "ActiveSkill",
    "Command",
    "Controller",
    "DEADMAN_S",
    "OnnxNet",
    "PolicySet",
    "Robotd",
    "Sensors",
    "Step",
    "Tuning",
]
