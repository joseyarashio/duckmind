"""duck-body's contracts: body.sock, the act executors, robot intents and the posture tap.

**body.sock** (spec §6, §7.1, §7.3) is a unix socket of JSON lines, one object per line, each
with a ``type``:

* server → client, first line: ``hello`` = ``{type, api: "body-v1", percept: "percept-v1",
  pv, src, t_ms, unix_ms, caps: {tof, eye, pg}}``. ``t_ms`` is duck-body's clock (its
  ``CLOCK_MONOTONIC`` in ms, the axis of every percept ``t_ms``, of ``deadline_ms`` and of
  ``robot.state.t_ns / 1e6``); ``unix_ms`` is the wall clock read at the same moment, so a
  client can map one onto the other.
* client → server: ``cmd`` = ``{type: "cmd", cmd: "duck.walk" | …, act_id, deadline_ms, …}``;
  the fields per command are the dataclasses below (:func:`parse_command`).
* server → client: ``ack`` = ``{type, act_id, ok, reason, t_ms}`` for every command, at once;
  ``percept`` = ``{type, event}`` with one percept-v1 event (:mod:`duckmind.percept`) sent at
  the moment it is sensed; ``done`` = ``{type, act_id, kind, t_ms}`` after an act's result
  percept (the harness's cue); ``error`` = ``{type, error}`` for a line it could not read.

**Acts.** One act at a time. An act command while another runs is refused (``busy``) unless it
says ``preempt: true``; ``duck.stop`` always preempts. Every act ends with exactly one result
percept (:data:`RESULT_KINDS`) carrying its ``act_id`` — also when it was preempted
(``reason: "preempt"``), overran its ``deadline_ms`` (``"deadline"``) or was cut by a reflex.

**Executors** (walkexec, dockexec, lookplan's looks, curiosity steps) implement
:class:`ActExecutor`: each tick the body loop hands them an :class:`ActContext` and sends what
their :class:`ActOutput` asks for through the reflexes and the forbidden-command guard
(:mod:`duckmind.duckbody.reflex`). They never talk to robotd themselves.

**Units.** Wire commands and percepts are integers: mm, mm/s, mrad, mrad/s, ms. Robot intents
are what robotd takes: SI floats (m/s, rad/s, rad). The posture tap carries robotd's own floats
(metres, radians) plus the integer ``trunk_height_mm``.

Pure: standard library and :mod:`duckmind.percept` only.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from duckmind.percept import ACT_ID_PATTERN, PERCEPT_API, SOURCES

if TYPE_CHECKING:  # the full robot.state record lives with its client
    from duckmind.duckbody.robotd_client import RobotState

BODY_API = "body-v1"
BOARD_SOCKET = "/run/duckmind/body.sock"
MAX_LINE_BYTES = 256_000

COMMANDS = (
    "duck.walk",
    "duck.look",
    "duck.posture",
    "duck.dock",
    "duck.curio",
    "duck.stop",
    "duck.express",
    "duck.sound",
)
ACT_COMMANDS = ("duck.walk", "duck.look", "duck.posture", "duck.dock", "duck.curio")
RESULT_KINDS: dict[str, tuple[str, ...]] = {
    "duck.walk": ("walk.done",),
    "duck.look": ("sense.scan", "sense.sighting"),
    "duck.posture": ("body.posture",),
    "duck.dock": ("body.dock",),
    "duck.curio": ("body.curio_step",),
}
PRE_LOOKS = ("glance", "auto", "none")
# Where the duck's attention is on this walk (owner 2026-09-28 「保留鴨子偶爾不注意撞到東西的呆萌感」):
# ``on`` — watching its way (the default: every rule of lookplan.need_glance, and a slow short
# step into floor it has not seen); ``low`` — excited or playing, sleepy, a hatchling, of low wit
# or too bold to check (it glances only when leaving the nest, just put down or just bumped);
# ``elsewhere`` — its gaze is on something else (a toy, a person, a sound): no glance before the
# walk and no look at its end. Neither ``low`` nor ``elsewhere`` slows into floor it has not seen
# (it does not know it has not seen it); what a recent look saw still stops it. The caller (Duck
# Core, the neural duck) sets it from its own state; duck-body never draws it by chance.
ATTENTION = ("on", "low", "elsewhere")
LOOK_KINDS = ("scan", "look_up", "seek_nest", "fixate", "idle", "glance_down")
LOOK_WANTS = ("nest", "thing", "duck", "person", "none")
HEAD_MODES = ("home", "down")
CURIO_PHASES = ("start", "step", "timeout", "end")
DOCK_STYLES = ("backin", "forward")
DOCK_SIDES = ("right", "left")  # backin's J-hook: which side of the dock it turns round on
REFUSALS = (
    "busy",
    "past_deadline",
    "invalid",
    "forbidden",
    "lifted",
    "fallen",
    "righting",
    "unavailable",
)

# Speed rules (spec §5.3, gait.md:42, owner A3). ``NORMAL_VX_MAX_MM``: owner A3 asked for the
# fastest of 0.45/0.50/0.55 m/s with 0/30 falls; body-motion measured 0 falls at every one
# (docs/validation/evidence/phase3/a3_walk_speed.json: 0/150 per speed and servo model) and
# walkexec walks normally at 0.55 — arcs and approaches stay at 0.40. Caveat from that
# measurement: velstand was trained within ±0.4 m/s.
MIN_STEP_VX_MM = 300  # below 0.30 m/s velstand does not step (0 < |vx| < 300 is refused)
NORMAL_VX_MAX_MM = 550
APPROACH_VX_MM = 400  # owner A3: slow to 0.40 when approaching a specific thing
VYAW_MAX_MRAD = 1000
STOP_AHEAD_MM = 300
MAX_WALK_MS = 60_000


class ProtocolError(ValueError):
    """A body.sock line that breaks the contract; ``reason`` is the ack refusal word."""

    def __init__(self, message: str, reason: str = "invalid") -> None:
        super().__init__(message)
        self.reason = reason


# ── field readers ──────────────────────────────────────────────────────────────


def _int(
    obj: Mapping[str, Any],
    name: str,
    default: int | None = None,
    *,
    lo: int | None = None,
    hi: int | None = None,
) -> int:
    value = obj.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"{name}: expected an integer")
    if lo is not None and value < lo:
        raise ProtocolError(f"{name}: {value} < {lo}")
    if hi is not None and value > hi:
        raise ProtocolError(f"{name}: {value} > {hi}")
    return value


def _opt_int(obj: Mapping[str, Any], name: str) -> int | None:
    return None if obj.get(name) is None else _int(obj, name)


def _choice(
    obj: Mapping[str, Any], name: str, options: Sequence[str], default: str | None = None
) -> str:
    value = obj.get(name, default)
    if value not in options:
        raise ProtocolError(f"{name}: {value!r} not in {tuple(options)}")
    return str(value)


def _text(obj: Mapping[str, Any], name: str, default: str | None = None) -> str:
    value = obj.get(name, default)
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"{name}: expected text")
    return value


def _act_id(obj: Mapping[str, Any]) -> str:
    value = obj.get("act_id")
    if not isinstance(value, str) or not ACT_ID_PATTERN.match(value):
        raise ProtocolError("act_id: expected 1-64 characters of [A-Za-z0-9_.:-]")
    return value


def _mapping(obj: Mapping[str, Any], name: str) -> dict[str, Any] | None:
    value = obj.get(name)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ProtocolError(f"{name}: expected an object")
    return dict(value)


# ── the commands (spec §7.1) ──────────────────────────────────────────────────


@dataclass(frozen=True)
class Toward:
    """The just-seen target a walk approaches (owner A1/A2/A3): the look that saw it, where it
    was (trunk frame at that look) and when. Only a walk carrying a fresh ``toward`` inside a
    curiosity act may relax the ToF stop (A1) or walk head down (A2)."""

    look_id: str
    b_mrad: int
    r_mm: int
    seen_t_ms: int
    thing: str | None = None

    @classmethod
    def from_wire(cls, obj: Mapping[str, Any]) -> Toward:
        thing = obj.get("thing")
        if thing is not None and not isinstance(thing, str):
            raise ProtocolError("toward.thing: expected text")
        return cls(
            look_id=_text(obj, "look_id"),
            b_mrad=_int(obj, "b_mrad", lo=-3142, hi=3142),
            r_mm=_int(obj, "r_mm", lo=0, hi=20_000),
            seen_t_ms=_int(obj, "seen_t_ms", lo=0),
            thing=thing,
        )

    def to_wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {
            "look_id": self.look_id,
            "b_mrad": self.b_mrad,
            "r_mm": self.r_mm,
            "seen_t_ms": self.seen_t_ms,
        }
        if self.thing is not None:
            wire["thing"] = self.thing
        return wire


@dataclass(frozen=True)
class WalkCmd:
    """``duck.walk`` — one segment. ``vx_mm`` mm/s forward, ``vyaw_mrad`` mrad/s (left +).
    ``expect`` and ``bel_from`` are Duck Core's *predictions*, carried for the record only.
    A negative ``vx_mm`` asks for a small back step (a back arc with ``vyaw_mrad``): walkexec
    takes it only when duck-body's own senses say the duck is cornered (owner 「2. 可以」),
    at −0.30…−0.40 m/s and at most 0.2 m; otherwise it ends ``veto_look`` without moving.
    ``attention`` (:data:`ATTENTION`, default ``on``) is where the caller's attention is: with
    ``pre_look: auto`` it decides whether the walk glances first, ``elsewhere`` also skips the
    look at the walk's end, and a walk whose attention is not ``on`` goes as asked into floor it
    has not seen (no slow unseen step) — so it may bump into what is there."""

    act_id: str
    deadline_ms: int
    vx_mm: int
    vyaw_mrad: int = 0
    max_ms: int = 3000
    pre_look: str = "auto"
    stop_ahead_mm: int = STOP_AHEAD_MM
    trim_mrad: int = 0
    purpose: str = "wander"
    excursion: str | None = None
    bel_from: dict[str, Any] | None = None
    expect: dict[str, Any] | None = None
    toward: Toward | None = None
    head: str = "home"
    curio_id: str | None = None
    preempt: bool = False
    attention: str = "on"
    cmd: str = field(default="duck.walk", init=False)

    @property
    def approach(self) -> bool:
        """Approaching a specific thing (owner A3: at most 0.40 m/s)."""
        return self.toward is not None


@dataclass(frozen=True)
class LookCmd:
    act_id: str
    deadline_ms: int
    kind: str
    want: str = "none"
    bearing_mrad: int | None = None
    target: str | None = None
    preempt: bool = False
    cmd: str = field(default="duck.look", init=False)


@dataclass(frozen=True)
class PostureCmd:
    act_id: str
    deadline_ms: int
    want: str
    preempt: bool = False
    cmd: str = field(default="duck.posture", init=False)


@dataclass(frozen=True)
class DockCmd:
    """``duck.dock``. ``style``: how this dock is entered (``backin`` | ``forward``; else the
    body's configured style). ``side``: backin's turning side — ``right`` (the J-hook's lane on
    the dock's right, a right U-turn) or ``left`` (mirrored, where the room needs it: the
    apartment's dock has the corridor wall on its right); else the configured side."""

    act_id: str
    deadline_ms: int
    style: str | None = None
    preempt: bool = False
    side: str | None = None
    cmd: str = field(default="duck.dock", init=False)


@dataclass(frozen=True)
class CurioCmd:
    """``duck.curio`` (curiosity spec §11): ``act_id`` is the curiosity step id ``C<n>``."""

    act_id: str
    deadline_ms: int
    phase: str
    thing: str | None = None
    step: int | None = None
    params: dict[str, Any] = field(default_factory=dict)
    expect_s: int | None = None
    preempt: bool = False
    cmd: str = field(default="duck.curio", init=False)


@dataclass(frozen=True)
class StopCmd:
    why: str = "stop"
    act_id: str | None = None
    cmd: str = field(default="duck.stop", init=False)


@dataclass(frozen=True)
class ExpressCmd:
    """``duck.express`` / ``duck.sound``: unchanged from today, passed to the expression layer."""

    cmd: str
    name: str
    act_id: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


Command = WalkCmd | LookCmd | PostureCmd | DockCmd | CurioCmd | StopCmd | ExpressCmd


def parse_command(obj: Mapping[str, Any]) -> Command:
    """One ``cmd`` line as a typed command, or :class:`ProtocolError`."""
    if not isinstance(obj, Mapping):
        raise ProtocolError("a command must be an object")
    name = obj.get("cmd")
    if name not in COMMANDS:
        raise ProtocolError(f"cmd: {name!r} not in {COMMANDS}")
    preempt = obj.get("preempt", False)
    if not isinstance(preempt, bool):
        raise ProtocolError("preempt: expected true/false")
    if name == "duck.stop":
        act = obj.get("act_id")
        return StopCmd(why=_text(obj, "why", "stop"), act_id=act if isinstance(act, str) else None)
    if name in ("duck.express", "duck.sound"):
        act = obj.get("act_id")
        return ExpressCmd(
            cmd=str(name),
            name=_text(obj, "name"),
            act_id=act if isinstance(act, str) else None,
            params=_mapping(obj, "params") or {},
        )
    act_id = _act_id(obj)
    deadline_ms = _int(obj, "deadline_ms", lo=0)
    if name == "duck.walk":
        toward_obj = _mapping(obj, "toward")
        excursion = obj.get("excursion")
        curio_id = obj.get("curio_id")
        return WalkCmd(
            act_id=act_id,
            deadline_ms=deadline_ms,
            vx_mm=_int(obj, "vx_mm", lo=-1000, hi=1000),
            vyaw_mrad=_int(obj, "vyaw_mrad", 0, lo=-3000, hi=3000),
            max_ms=_int(obj, "max_ms", 3000, lo=1, hi=MAX_WALK_MS),
            pre_look=_choice(obj, "pre_look", PRE_LOOKS, "auto"),
            stop_ahead_mm=_int(obj, "stop_ahead_mm", STOP_AHEAD_MM, lo=0, hi=2000),
            trim_mrad=_int(obj, "trim_mrad", 0, lo=-500, hi=500),
            purpose=_text(obj, "purpose", "wander"),
            excursion=None if excursion is None else str(excursion),
            bel_from=_mapping(obj, "bel_from"),
            expect=_mapping(obj, "expect"),
            toward=None if toward_obj is None else Toward.from_wire(toward_obj),
            head=_choice(obj, "head", HEAD_MODES, "home"),
            curio_id=None if curio_id is None else str(curio_id),
            preempt=preempt,
            attention=_choice(obj, "attention", ATTENTION, "on"),
        )
    if name == "duck.look":
        target = obj.get("target")
        return LookCmd(
            act_id=act_id,
            deadline_ms=deadline_ms,
            kind=_choice(obj, "kind", LOOK_KINDS),
            want=_choice(obj, "want", LOOK_WANTS, "none"),
            bearing_mrad=_opt_int(obj, "bearing_mrad"),
            target=None if target is None else str(target),
            preempt=preempt,
        )
    if name == "duck.posture":
        return PostureCmd(act_id, deadline_ms, _choice(obj, "want", ("stand", "sit")), preempt)
    if name == "duck.dock":
        style = obj.get("style")
        if style is not None and style not in DOCK_STYLES:
            raise ProtocolError(f"style: {style!r} not in {DOCK_STYLES}")
        side = obj.get("side")
        if side is not None and side not in DOCK_SIDES:
            raise ProtocolError(f"side: {side!r} not in {DOCK_SIDES}")
        return DockCmd(act_id, deadline_ms, style, preempt, side)
    thing = obj.get("thing")
    return CurioCmd(
        act_id=act_id,
        deadline_ms=deadline_ms,
        phase=_choice(obj, "phase", CURIO_PHASES),
        thing=None if thing is None else str(thing),
        step=_opt_int(obj, "step"),
        params=_mapping(obj, "params") or {},
        expect_s=_opt_int(obj, "expect_s"),
        preempt=preempt,
    )


def command_to_wire(command: Command) -> dict[str, Any]:
    """The ``cmd`` line for a typed command (what a harness or Duck Core sends)."""
    wire: dict[str, Any] = {"type": "cmd", "cmd": command.cmd}
    for name, value in command.__dict__.items():
        if name == "cmd" or value is None:
            continue
        if isinstance(value, Toward):
            value = value.to_wire()
        if name == "preempt" and value is False:
            continue
        if name == "attention" and value == "on":
            continue  # the default: a v4 caller's walk line stays as it was
        wire[name] = value
    return wire


# ── server → client lines ─────────────────────────────────────────────────────


def hello(
    *, pv: str, src: str, t_ms: int, unix_ms: int, caps: Mapping[str, bool]
) -> dict[str, Any]:
    if src not in SOURCES:
        raise ProtocolError(f"src: {src!r} not in {SOURCES}")
    return {
        "type": "hello",
        "api": BODY_API,
        "percept": PERCEPT_API,
        "pv": pv,
        "src": src,
        "t_ms": int(t_ms),
        "unix_ms": int(unix_ms),
        "caps": {name: bool(value) for name, value in caps.items()},
    }


def ack(act_id: str | None, ok: bool, t_ms: int, reason: str | None = None) -> dict[str, Any]:
    if not ok and reason not in REFUSALS:
        raise ProtocolError(f"a refusal needs a reason in {REFUSALS}")
    return {"type": "ack", "act_id": act_id, "ok": ok, "reason": reason, "t_ms": int(t_ms)}


def percept_line(event: Mapping[str, Any]) -> dict[str, Any]:
    return {"type": "percept", "event": dict(event)}


def done(act_id: str, kind: str, t_ms: int) -> dict[str, Any]:
    return {"type": "done", "act_id": act_id, "kind": kind, "t_ms": int(t_ms)}


def error(message: str) -> dict[str, Any]:
    return {"type": "error", "error": message[:500]}


def encode_line(message: Mapping[str, Any]) -> bytes:
    data = json.dumps(message, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    encoded = data.encode("utf-8") + b"\n"
    if len(encoded) > MAX_LINE_BYTES:
        raise ProtocolError("line too long")
    return encoded


def decode_line(line: bytes | str) -> dict[str, Any]:
    if isinstance(line, bytes) and len(line) > MAX_LINE_BYTES:
        raise ProtocolError("line too long")
    try:
        value = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as problem:
        raise ProtocolError(f"not JSON: {problem}") from problem
    if not isinstance(value, dict):
        raise ProtocolError("a line must be a JSON object")
    return value


# ── robot intents (what the body loop sends robotd, after the guard) ──────────


@dataclass(frozen=True)
class Intent:
    """One robotd request. ``robot.move``/``robot.head``/``robot.mouth``/``robot.pose`` are continuous
    notifications; ``robot.do``/``robot.stop``/``robot.look`` are calls."""

    method: str
    params: Mapping[str, Any] = field(default_factory=dict)

    @staticmethod
    def move(vx: float, vyaw: float, vy: float = 0.0) -> Intent:
        return Intent("robot.move", {"vx": float(vx), "vy": float(vy), "vyaw": float(vyaw)})

    @staticmethod
    def head(offsets: Sequence[float]) -> Intent:
        """Head offsets from the v5 HOME (neck_pitch, head_pitch, head_yaw, head_roll), rad."""
        names = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
        return Intent("robot.head", {n: float(v) for n, v in zip(names, offsets, strict=True)})

    @staticmethod
    def pose(pitch: float, active: bool = True) -> Intent:
        """The standing body pose: a bow of ``pitch`` rad (> 0 bows), the height and roll at nominal. ``active=False``
        snaps the pose back to nominal (robotd's own exit)."""
        return Intent(
            "robot.pose", {"z": 0.0, "roll": 0.0, "pitch": float(pitch), "active": bool(active)}
        )

    @staticmethod
    def do(skill: str) -> Intent:
        return Intent("robot.do", {"skill": skill})

    @staticmethod
    def stop() -> Intent:
        return Intent("robot.stop", {})

    @property
    def twist(self) -> tuple[float, float]:
        """(vx m/s, vyaw rad/s) of a ``robot.move``; (0, 0) for anything else."""
        if self.method != "robot.move":
            return (0.0, 0.0)
        return (float(self.params.get("vx", 0.0)), float(self.params.get("vyaw", 0.0)))


# ── the posture tap (spec §4.3 step 5) ────────────────────────────────────────


@dataclass(frozen=True)
class PoseSample:
    """One posture-tap sample. duck-body keeps them at 50 Hz and streams every other one (25 Hz)
    to duck-eye, which interpolates to the frame's capture time on the monotonic axis.

    * ``seq`` — the tap's counter; ``eye.look.pose_tap_ref`` is the ``seq`` of the newest sample
      the body had sent when it asked for the look;
    * ``t_ns`` — ``robot.state.t_ns`` (``CLOCK_MONOTONIC``);
    * ``trunk_height_mm`` — the trunk origin above the lower sole, from leg FK and gravity;
    * ``camera_pos``/``camera_quat`` — the camera in the trunk frame (OpenCV camera axes; metres,
      scalar-first quaternion), robotd's ``frames.camera`` or head FK; ``tof_pos``/``tof_quat``
      likewise (``None`` when unknown). On the wire (:meth:`to_wire`, the ``eye.pose`` line) they
      are ``camera_in_trunk``/``tof_in_trunk`` = ``{pos, quat}``, robotd's own PoseState shape;
    * ``imu_quat`` — trunk → world (the IMU's gravity-aligned frame), ``[w, x, y, z]``;
    * ``gyro`` — rad/s, trunk frame; ``head`` — the measured head joints (neck_pitch,
      head_pitch, head_yaw, head_roll), rad; ``head_vel`` — their largest |velocity|, rad/s
      (``None`` when robotd does not report velocities);
    * ``odom`` — (x m, y m, yaw rad) in the odometry frame; ``epoch`` — bumped whenever that
      frame breaks (lifted, robotd restarted), so odom-frame memories reset;
    * ``still`` — spec §4.2 rule 2: head joints < 0.2 rad/s and gyro < 0.3 rad/s for 150 ms;
    * ``policy`` — robotd's policy this tick (``walk``/``stand``/``held``/…).
    """

    seq: int
    t_ns: int
    trunk_height_mm: int | None
    camera_pos: tuple[float, float, float] | None
    camera_quat: tuple[float, float, float, float] | None
    tof_pos: tuple[float, float, float] | None
    tof_quat: tuple[float, float, float, float] | None
    imu_quat: tuple[float, float, float, float] | None
    gyro: tuple[float, float, float] | None
    head: tuple[float, float, float, float]
    head_vel: float | None
    odom: tuple[float, float, float]
    epoch: int
    still: bool
    policy: str

    def to_wire(self) -> dict[str, Any]:
        """The ``eye.pose`` line (duck-eye's :class:`duckmind.eye.protocol.PoseSample` reads it:
        ``camera_in_trunk``/``tof_in_trunk`` as robotd's ``{pos, quat}``)."""
        wire: dict[str, Any] = {"type": "eye.pose", "seq": self.seq, "t_ns": self.t_ns}
        wire["trunk_height_mm"] = self.trunk_height_mm
        if self.camera_pos is not None and self.camera_quat is not None:
            wire["camera_in_trunk"] = {"pos": list(self.camera_pos), "quat": list(self.camera_quat)}
            # robotd's frames.camera / HeadFk's cv2 frame: upright (duck-eye turns it by the
            # frame's rotate to its sensor frame)
            wire["cam_frame"] = "cv2_upright"
        if self.tof_pos is not None and self.tof_quat is not None:
            wire["tof_in_trunk"] = {"pos": list(self.tof_pos), "quat": list(self.tof_quat)}
        for name in ("imu_quat", "gyro", "head", "odom"):
            value = getattr(self, name)
            wire[name] = None if value is None else list(value)
        wire.update(head_vel=self.head_vel, epoch=self.epoch, still=self.still, policy=self.policy)
        return wire

    @classmethod
    def from_wire(cls, obj: Mapping[str, Any]) -> PoseSample:
        def vec(value: Any, name: str, n: int) -> Any:
            if value is None:
                return None
            if (
                not isinstance(value, list)
                or len(value) != n
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value)
                or not all(math.isfinite(v) for v in value)
            ):
                raise ProtocolError(f"pose.{name}: expected {n} finite numbers")
            return tuple(float(v) for v in value)

        def pose(name: str) -> tuple[Any, Any]:
            value = obj.get(name)
            if value is None:
                return None, None
            if not isinstance(value, Mapping):
                raise ProtocolError(f"pose.{name}: expected {{pos, quat}}")
            return vec(value.get("pos"), f"{name}.pos", 3), vec(
                value.get("quat"), f"{name}.quat", 4
            )

        head = vec(obj.get("head"), "head", 4)
        odom = vec(obj.get("odom"), "odom", 3)
        if head is None or odom is None:
            raise ProtocolError("pose: head and odom are required")
        camera_pos, camera_quat = pose("camera_in_trunk")
        tof_pos, tof_quat = pose("tof_in_trunk")
        height = obj.get("trunk_height_mm")
        head_vel = obj.get("head_vel")
        policy = obj.get("policy", "")
        if not isinstance(policy, str):
            raise ProtocolError("pose.policy: expected text")
        return cls(
            seq=_int(obj, "seq", lo=0),
            t_ns=_int(obj, "t_ns", lo=0),
            trunk_height_mm=None if height is None else _int(obj, "trunk_height_mm"),
            camera_pos=camera_pos,
            camera_quat=camera_quat,
            tof_pos=tof_pos,
            tof_quat=tof_quat,
            imu_quat=vec(obj.get("imu_quat"), "imu_quat", 4),
            gyro=vec(obj.get("gyro"), "gyro", 3),
            head=head,
            head_vel=None if head_vel is None else float(head_vel),
            odom=odom,
            epoch=_int(obj, "epoch", lo=0),
            still=bool(obj.get("still", False)),
            policy=policy,
        )


# ── what an act executor sees and returns each tick ───────────────────────────


@dataclass(frozen=True)
class DepthView:
    """The integer ToF summary (DS-v1, spec §5.1) the reflexes read; ``duckbody.depth`` makes
    it from ``tof.frame``. ``None`` ranges mean no usable return in that corridor. ``target_*``
    is the nearest ahead return's bearing, for the A1 "is this the target?" check."""

    t_ns: int
    ahead_mm: int | None
    left_mm: int | None
    right_mm: int | None
    drop_zones: int
    touch: bool
    head_pitch_mrad: int
    ahead_b_mrad: int = 0
    ahead_top_mm: int = 0  # the height above the floor of what the ahead return is on (0: unknown)
    virtual: bool = False  # the ahead / side return is a remembered low thing, under the beams now
    low_edge_mm: int = 0  # the height estimate (mm) of a wide low EDGE in view, else 0


@dataclass(frozen=True)
class ProprioView:
    """What :mod:`duckmind.duckbody.proprio` says about the body right now (levels, not edges;
    the edges become percepts)."""

    posture: str = "unknown"  # stand / sit / unknown
    policy: str = ""
    lifted: bool = False
    unloaded: bool = False  # legs light but not a claimed lift (owner B3: stop, do not claim)
    fallen: bool = False  # a fall episode is open (trigger A or B)
    tilt_mrad: int = 0
    blocked: bool = False
    brushed: bool = False
    stumbled: bool = False
    steps: int = 0
    odom_epoch: int = 0
    battery_pct: int | None = None


@dataclass(frozen=True)
class LookRequest:
    """Ask duck-eye for one look (eye.look minus ``pose_tap_ref``, which the body fills)."""

    look_id: str
    tasks: tuple[str, ...]
    not_before_ns: int = 0


@dataclass(frozen=True)
class ActContext:
    t_ns: int
    state: RobotState | None
    pose: PoseSample | None
    depth: DepthView | None
    proprio: ProprioView
    light: str = "none"
    eye: tuple[Mapping[str, Any], ...] = ()
    halted_by: str | None = None  # the reflex that zeroed the last move, if any
    # how long the guard has counted the head at HOME (its rule 3 wants 300 ms before a walk
    # starts); None when the caller does not know (executors then judge the head themselves)
    head_home_ms: int | None = None
    pg: bool | None = None  # the dock's pad contact (the sim-only ``pg`` op), when the loop has it
    # duck-body's own senses say every way forward is shut here — ahead, left and right
    # (:mod:`duckmind.duckbody.cornered`): a duck.walk back step may go (owner 「2. 可以」)
    cornered: bool = False

    @property
    def t_ms(self) -> int:
        return self.t_ns // 1_000_000


@dataclass(frozen=True)
class ActOutput:
    """``twist`` = (vx mm/s, vyaw mrad/s) to stream (``None`` = hold still); ``head`` = offsets
    from HOME in rad (``None`` = HOME); ``do`` = one ``robot.do`` skill; ``done`` = the act has
    ended and ``percepts`` holds its result."""

    twist: tuple[int, int] | None = None
    head: tuple[float, float, float, float] | None = None
    do: str | None = None
    mouth: float | None = None
    looks: tuple[LookRequest, ...] = ()
    percepts: tuple[Mapping[str, Any], ...] = ()
    done: bool = False


class ActExecutor(Protocol):
    act_id: str
    command: Command

    def step(self, ctx: ActContext) -> ActOutput: ...

    def preempt(self, why: str, ctx: ActContext) -> ActOutput:
        """End now (``why`` is a walk.done reason such as ``preempt``/``lifted``/``fell``/
        ``deadline``) and return the result percept."""
        ...


__all__ = [
    "ACT_COMMANDS",
    "APPROACH_VX_MM",
    "ATTENTION",
    "BODY_API",
    "COMMANDS",
    "ActContext",
    "ActExecutor",
    "ActOutput",
    "Command",
    "CurioCmd",
    "DepthView",
    "DockCmd",
    "ExpressCmd",
    "Intent",
    "LookCmd",
    "LookRequest",
    "MIN_STEP_VX_MM",
    "NORMAL_VX_MAX_MM",
    "PoseSample",
    "PostureCmd",
    "ProprioView",
    "ProtocolError",
    "RESULT_KINDS",
    "StopCmd",
    "Toward",
    "VYAW_MAX_MRAD",
    "WalkCmd",
    "ack",
    "command_to_wire",
    "decode_line",
    "done",
    "encode_line",
    "error",
    "hello",
    "parse_command",
    "percept_line",
]
