"""Duck Mind's only robotd client (spec §6: duck-body is the one process on ``robotd.sock``).

Three connections, as the official design wants them:

* **state** — ``robot.subscribe`` with no ``hz``: every 50 Hz tick. :class:`RobotState` keeps the
  whole record (joints, velocities, currents, IMU, frames, odometry, the safety verdict and its
  gravity) — the old :class:`duckmind.robotd.RobotdBodyPort` drops most of it. An empty
  ``velocities``/``currents_ma`` means *not reported*, never zero (upstream ``lib.rs`` 3746,
  3765), so those become ``None``.
* **command** — ``robot.move``/``robot.head``/``robot.mouth`` notifications and the
  ``robot.do``/``robot.model`` calls. Moves are streamed at 50 Hz by :class:`MoveStreamer`,
  because robotd's deadman zeroes a twist that is not refreshed within 500 ms.
* **safety** — ``robot.stop`` and ``robot.health`` (1 Hz, :class:`Cadence`) on their own
  socket, so a blocked command stream can never delay a stop.

Sockets live in :mod:`duckmind.duckbody.server`; this module talks to a :class:`Wire` (send one
JSON object, receive one or time out), so the headless rig and the tests plug in fakes and the
same code runs everywhere. Every outgoing intent passes the forbidden-command guard
(:func:`duckmind.duckbody.reflex.check_intent`) in the body loop before it reaches
:meth:`RobotdClient.send`; the client refuses anything outside :data:`ALLOWED_METHODS` itself.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from duckmind.duckbody.protocol import Intent

API_VERSION = 37
JOINT_NAMES = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "mouth",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)
LEG_JOINTS = (0, 1, 2, 3, 4, 10, 11, 12, 13, 14)
HEAD_JOINTS = (5, 6, 7, 8)
# robotd's v5 HOME for the head (duckmind.robotd.V5_HOME_HEAD): robot.head takes offsets from it
V5_HOME_HEAD = (0.3491, 0.3491, 0.0, 0.0)
# robot.pose (the standing body pose: a bow) is sent by the neural duck's emotions only (2026-10-08: Pollen's
# microduck_emotions play body pitch on it); the body link sends it only while an emotion plays (bodylink.BodyRequest.pose)
NOTIFY_METHODS = frozenset({"robot.move", "robot.head", "robot.mouth", "robot.pose"})
CALL_METHODS = frozenset({"robot.do", "robot.stop", "robot.model", "robot.health", "robot.look"})
ALLOWED_METHODS = NOTIFY_METHODS | CALL_METHODS
# Never from duck-body: servos, gains, torque, policy loads, reboots, enable (operator only).
FORBIDDEN_METHODS = frozenset(
    {
        "robot.enable",
        "robot.gains",
        "robot.torque",
        "robot.policy",
        "robot.load_policy",
        "robot.reboot_motors",
        "robot.servo",
        "robot.calibrate",
    }
)
DEADMAN_NS = 500_000_000
MOVE_PERIOD_NS = 20_000_000  # 50 Hz
HEALTH_PERIOD_NS = 1_000_000_000  # 1 Hz


class RobotdError(RuntimeError):
    """robotd answered with an error, broke the protocol, or went away."""


# ── the robot.state record ─────────────────────────────────────────────────────

Vec3 = tuple[float, float, float]
Quat4 = tuple[float, float, float, float]


def _num(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _vec(value: object, n: int) -> tuple[float, ...] | None:
    if not isinstance(value, (list, tuple)) or len(value) != n:
        return None
    numbers = [_num(v) for v in value]
    if any(v is None for v in numbers):
        return None
    return tuple(float(v) for v in numbers if v is not None)


def _floats(value: object) -> tuple[float, ...] | None:
    """A per-joint array, or ``None`` when absent/empty (*not reported*, never zero)."""
    if not isinstance(value, (list, tuple)) or not value:
        return None
    numbers = [_num(v) for v in value]
    if any(v is None for v in numbers):
        return None
    return tuple(float(v) for v in numbers if v is not None)


def _mapping(value: object) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _pose(value: object) -> tuple[Vec3, Quat4] | None:
    if not isinstance(value, Mapping):
        return None
    pos = _vec(value.get("pos"), 3)
    quat = _vec(value.get("quat"), 4)
    if pos is None or quat is None:
        return None
    return ((pos[0], pos[1], pos[2]), (quat[0], quat[1], quat[2], quat[3]))


@dataclass(frozen=True)
class RobotState:
    """One ``robot.state`` tick, as robotd sent it (SI units)."""

    t: float
    t_ns: int
    policy: str
    requested: Vec3
    applied: Vec3
    limited_by: tuple[str, ...]
    head_cmd: Quat4
    fallen: bool
    limp: bool
    gravity: Vec3 | None
    gain: int | None
    loop_hz: float
    loop_missed: int
    joints: tuple[float, ...]
    targets: tuple[float, ...] | None
    velocities: tuple[float, ...] | None
    currents_ma: tuple[float, ...] | None
    odom_pos: Vec3
    odom_yaw: float
    gyro: Vec3 | None
    imu_quat: Quat4 | None
    camera: tuple[Vec3, Quat4] | None
    tof: tuple[Vec3, Quat4] | None

    @classmethod
    def from_wire(cls, params: Mapping[str, Any]) -> RobotState:
        """Parse the ``params`` of a ``robot.state`` notification (or a recorded frame)."""
        move = params.get("move")
        safety = params.get("safety")
        if not isinstance(move, Mapping) or not isinstance(safety, Mapping):
            raise RobotdError("robot.state lacks move or safety")
        policy = params.get("policy")
        if not isinstance(policy, str):
            raise RobotdError("robot.state lacks policy")
        fallen, limp = safety.get("fallen"), safety.get("limp")
        if not isinstance(fallen, bool) or not isinstance(limp, bool):
            raise RobotdError("robot.state safety flags are not booleans")
        joints = _floats(params.get("joints"))
        if joints is None or len(joints) != len(JOINT_NAMES):
            raise RobotdError("robot.state lacks the 15 measured joints")
        requested = _vec(move.get("requested"), 3) or (0.0, 0.0, 0.0)
        applied = _vec(move.get("applied"), 3) or (0.0, 0.0, 0.0)
        limited = move.get("limited_by") or []
        if not isinstance(limited, list) or not all(isinstance(x, str) for x in limited):
            raise RobotdError("robot.state limited_by is malformed")
        head = _vec(params.get("head"), 4) or (0.0, 0.0, 0.0, 0.0)
        gravity = _vec(safety.get("gravity"), 3)
        if gravity is not None and gravity == (0.0, 0.0, 0.0):
            gravity = None  # a daemon predating the field sends zeros
        gain = safety.get("gain")
        loop: Mapping[str, Any] = _mapping(params.get("loop")) or {}
        odom: Mapping[str, Any] = _mapping(params.get("odom")) or {}
        imu = _mapping(params.get("imu"))
        frames = _mapping(params.get("frames"))
        t_ns = params.get("t_ns", 0)
        odom_pos = _vec(odom.get("position"), 3) or (0.0, 0.0, 0.0)
        velocities = _floats(params.get("velocities"))
        currents = _floats(params.get("currents_ma"))
        gyro = None if imu is None else _vec(imu.get("gyro"), 3)
        quat = None if imu is None else _vec(imu.get("quat"), 4)
        return cls(
            t=_num(params.get("t")) or 0.0,
            t_ns=int(t_ns) if isinstance(t_ns, int) and not isinstance(t_ns, bool) else 0,
            policy=policy,
            requested=(requested[0], requested[1], requested[2]),
            applied=(applied[0], applied[1], applied[2]),
            limited_by=tuple(limited),
            head_cmd=(head[0], head[1], head[2], head[3]),
            fallen=fallen,
            limp=limp,
            gravity=None if gravity is None else (gravity[0], gravity[1], gravity[2]),
            gain=gain if isinstance(gain, int) and not isinstance(gain, bool) else None,
            loop_hz=_num(loop.get("hz")) or 0.0,
            loop_missed=int(loop.get("missed", 0) or 0),
            joints=joints,
            targets=_floats(params.get("targets")),
            velocities=velocities if velocities and len(velocities) == len(JOINT_NAMES) else None,
            currents_ma=currents if currents and len(currents) == len(JOINT_NAMES) else None,
            odom_pos=(odom_pos[0], odom_pos[1], odom_pos[2]),
            odom_yaw=_num(odom.get("yaw")) or 0.0,
            gyro=None if gyro is None else (gyro[0], gyro[1], gyro[2]),
            imu_quat=None if quat is None else (quat[0], quat[1], quat[2], quat[3]),
            camera=None if frames is None else _pose(frames.get("camera")),
            tof=None if frames is None else _pose(frames.get("tof")),
        )

    # ── derived readings every sensor module shares ──

    @property
    def head(self) -> Quat4:
        """Measured head joints (neck_pitch, head_pitch, head_yaw, head_roll), rad."""
        j = self.joints
        return (j[5], j[6], j[7], j[8])

    @property
    def head_offsets(self) -> Quat4:
        """Measured head joints as offsets from the v5 HOME (what ``robot.head`` takes)."""
        h = self.head
        return (
            h[0] - V5_HOME_HEAD[0],
            h[1] - V5_HOME_HEAD[1],
            h[2] - V5_HOME_HEAD[2],
            h[3] - V5_HOME_HEAD[3],
        )

    @property
    def head_speed(self) -> float | None:
        """Largest |velocity| of the four head joints, rad/s (``None`` if not reported)."""
        if self.velocities is None:
            return None
        return max(abs(self.velocities[i]) for i in HEAD_JOINTS)

    @property
    def leg_load(self) -> float | None:
        """Sum of the ten leg currents (mA on the board; the twin's |force|×100 stand-in in the
        sim, uncalibrated — owner B3 sets thresholds on it now and recalibrates on the board)."""
        if self.currents_ma is None:
            return None
        return sum(self.currents_ma[i] for i in LEG_JOINTS)

    @property
    def down(self) -> Vec3 | None:
        """Unit gravity direction in the trunk frame (upright ≈ (0, 0, −1)): robotd's projected
        gravity, else the IMU quaternion."""
        if self.gravity is not None:
            g = self.gravity
        elif self.imu_quat is not None:
            w, x, y, z = self.imu_quat
            # world −z expressed in the trunk frame: R^T · (0, 0, −1)
            g = (-2.0 * (x * z - w * y), -2.0 * (y * z + w * x), -(1.0 - 2.0 * (x * x + y * y)))
        else:
            return None
        n = math.sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2])
        if n < 1e-9:
            return None
        return (g[0] / n, g[1] / n, g[2] / n)

    @property
    def tilt_rad(self) -> float | None:
        """Angle between the trunk's up axis and the world's up, rad (robotd's own reading)."""
        d = self.down
        if d is None:
            return None
        return math.acos(max(-1.0, min(1.0, -d[2])))


# ── transport ──────────────────────────────────────────────────────────────────


class Wire(Protocol):
    """One JSON-lines connection: the server's sockets, or a fake in tests and the rig."""

    def send(self, obj: Mapping[str, Any]) -> None: ...

    def recv(self, timeout_s: float) -> dict[str, Any] | None: ...

    def close(self) -> None: ...


class _Rpc:
    def __init__(self, wire: Wire, timeout_s: float) -> None:
        self.wire = wire
        self.timeout_s = timeout_s
        self._next_id = 1
        self.pending: list[dict[str, Any]] = []  # notifications read while waiting

    def call(self, method: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        request_id = self._next_id
        self._next_id += 1
        self.wire.send(
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": dict(params or {})}
        )
        while True:
            frame = self.wire.recv(self.timeout_s)
            if frame is None:
                raise RobotdError(f"{method}: no answer within {self.timeout_s} s")
            if "id" not in frame:
                self.pending.append(frame)
                continue
            if frame.get("id") != request_id:
                raise RobotdError(f"{method}: answer for id {frame.get('id')}, wanted {request_id}")
            if "error" in frame:
                raise RobotdError(f"{method}: {frame['error']}")
            result = frame.get("result")
            if not isinstance(result, dict):
                raise RobotdError(f"{method}: no result object")
            return result

    def notify(self, method: str, params: Mapping[str, Any]) -> None:
        self.wire.send({"jsonrpc": "2.0", "method": method, "params": dict(params)})


class RobotdClient:
    """The three robotd connections behind one object (see the module doc)."""

    def __init__(self, command: Wire, state: Wire, safety: Wire, *, timeout_s: float = 2.0) -> None:
        self._command = _Rpc(command, timeout_s)
        self._state = _Rpc(state, timeout_s)
        self._safety = _Rpc(safety, timeout_s)
        self.hello: dict[str, Any] = {}
        self.capabilities: dict[str, Any] = {}
        self.model: dict[str, Any] = {}

    def start(self, *, read_model: bool = True) -> None:
        """hello on every connection, subscribe to every tick, read ``robot.model`` once."""
        self.hello = self._command.call("hello", {"api_version": API_VERSION})
        self._state.call("hello", {"api_version": API_VERSION})
        self._safety.call("hello", {"api_version": API_VERSION})
        self.capabilities = self._state.call("robot.subscribe", {})
        if self.capabilities.get("accepted") is False:
            raise RobotdError("robot.subscribe was refused")
        if read_model:
            try:
                self.model = self._command.call("robot.model")
            except RobotdError:
                self.model = {}  # an older daemon: head FK from the MJCF instead

    @property
    def skills(self) -> tuple[str, ...]:
        """The ``robot.do`` names this robot answers to (from the subscribe acknowledgement)."""
        skills = self.capabilities.get("skills")
        if isinstance(skills, list):
            return tuple(str(s) for s in skills)
        return ("sit_toggle", "ground_pick", "kick_left", "kick_right", "roulade")

    def read_state(self, timeout_s: float) -> RobotState | None:
        """The next tick (``None`` on a quiet socket)."""
        frame = (
            self._state.pending.pop(0) if self._state.pending else self._state.wire.recv(timeout_s)
        )
        if frame is None:
            return None
        if frame.get("method") != "robot.state":
            return None
        params = frame.get("params")
        if not isinstance(params, Mapping):
            raise RobotdError("robot.state without params")
        return RobotState.from_wire(params)

    def send(self, intent: Intent) -> dict[str, Any] | None:
        """One intent: notifications return ``None``, calls return robotd's result."""
        if intent.method not in ALLOWED_METHODS or intent.method in FORBIDDEN_METHODS:
            raise RobotdError(f"duck-body never sends {intent.method}")
        if intent.method == "robot.stop":
            return self.stop()
        if intent.method == "robot.health":
            return self.health()
        if intent.method in NOTIFY_METHODS:
            self._command.notify(intent.method, intent.params)
            return None
        return self._command.call(intent.method, intent.params)

    def stop(self) -> dict[str, Any]:
        return self._safety.call("robot.stop")

    def health(self) -> dict[str, Any]:
        result = self._safety.call("robot.health")
        if not isinstance(result.get("healthy"), bool):
            raise RobotdError("robot.health lacks the healthy verdict")
        return result


# ── pure cadences (clock-injected) ───────────────────────────────────────────


class MoveStreamer:
    """Streams the wanted twist as ``robot.move`` at 50 Hz while it is non-zero (robotd's deadman
    zeroes a twist after 500 ms without a refresh). A change goes out at once; so does the first
    zero after motion, which then stops the stream (nothing else writes the twist)."""

    def __init__(self, period_ns: int = MOVE_PERIOD_NS) -> None:
        self.period_ns = period_ns
        self.last: tuple[float, float] = (0.0, 0.0)
        self.last_ns: int | None = None
        self.sent = 0

    def step(self, now_ns: int, twist: tuple[float, float] | None) -> Intent | None:
        vx, vyaw = twist if twist is not None else (0.0, 0.0)
        want = (float(vx), float(vyaw))
        moving = want != (0.0, 0.0)
        changed = want != self.last
        due = self.last_ns is None or now_ns - self.last_ns >= self.period_ns
        if changed or (moving and due):
            self.last, self.last_ns = want, now_ns
            self.sent += 1
            return Intent.move(want[0], want[1])
        return None

    def zero_now(self, now_ns: int) -> Intent:
        """A zero twist regardless of cadence (reflexes, falls, lifts)."""
        self.last, self.last_ns = (0.0, 0.0), now_ns
        self.sent += 1
        return Intent.move(0.0, 0.0)


class Cadence:
    """True once per period (``robot.health`` at 1 Hz, sense.light every 10 min …)."""

    def __init__(self, period_ns: int = HEALTH_PERIOD_NS) -> None:
        self.period_ns = period_ns
        self.next_ns: int | None = None

    def due(self, now_ns: int) -> bool:
        if self.next_ns is None or now_ns >= self.next_ns:
            self.next_ns = now_ns + self.period_ns
            return True
        return False


def tilt_deg(state: RobotState) -> float | None:
    tilt = state.tilt_rad
    return None if tilt is None else math.degrees(tilt)


def leg_sum(currents: Sequence[float]) -> float:
    return sum(currents[i] for i in LEG_JOINTS)


__all__ = [
    "ALLOWED_METHODS",
    "Cadence",
    "FORBIDDEN_METHODS",
    "HEAD_JOINTS",
    "JOINT_NAMES",
    "LEG_JOINTS",
    "MoveStreamer",
    "RobotState",
    "RobotdClient",
    "RobotdError",
    "V5_HOME_HEAD",
    "Wire",
    "leg_sum",
    "tilt_deg",
]
