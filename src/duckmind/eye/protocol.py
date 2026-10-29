"""duck-eye's socket protocol, ``eye-v1`` (spec §4.3, §6): what duck-body and duck-eye say.

duck-eye is its own process (owner decision D7): it holds no robot socket and never opens the
camera itself. For each look duck-body opens a fresh connection to ``media.sock``, says
``hello`` on it (JSON-RPC id 1), and hands the connected fd to duck-eye over ``eye.sock`` with
SCM_RIGHTS, riding on the ``eye.look`` line. duck-eye asks that connection for the *next* frame
(``media.frame``, id :data:`MEDIA_ID`), pairs the frame with the head pose at the capture
instant, runs the look's tasks, answers ``eye.result`` with integer features only and drops the
frame bytes (spec §4.8: never stored).

**Wire.** A unix stream socket (``eye.sock``; :data:`EYE_SOCKET` on the board, ``<state>/eye.sock``
beside the simulator), duck-body is the only client. One compact JSON object per line (UTF-8,
at most :data:`MAX_LINE_BYTES` including the newline), each with a ``type``. The only file
descriptor ever sent rides on an ``eye.look`` line: exactly one fd, in the same ``sendmsg`` as
the line's bytes (``socket.send_fds``). No other line carries one.

**Clocks.** Every ``*_ns`` field is ``CLOCK_MONOTONIC`` in nanoseconds — robotd's
``RobotState.t_ns`` clock (``duck_ipc_proto::clock::monotonic_ns``). In Python read it with
``time.clock_gettime_ns(time.CLOCK_MONOTONIC)``: on macOS ``time.monotonic_ns()`` is a
*different* clock (``mach_absolute_time``, which stops in sleep), about 10 s apart on this Mac.
The frame header's ``captured_at_unix_us`` is the wall clock; duck-eye bridges it
(:mod:`duckmind.eye.frames`).

Messages, duck-body → duck-eye:

* ``eye.hello`` ``{api, role}`` — first line; answered ``eye.hello {api, perception_hash, tasks,
  max_retakes}`` or ``eye.error`` (``api``) and the connection closes.
* ``eye.look`` (:class:`EyeLook`, + one media fd) — answered by exactly one ``eye.result``.
* ``eye.pose`` (:class:`PoseSample`) — the pose tap, 25 Hz, at least from a look's
  ``pose_tap_ref`` until its result (duck-body may stream it all the time: ~200 bytes a sample).
* ``eye.health`` (:class:`EyeHealth`) — ``robot.health``'s thermal part, 1 Hz; the governor's input.
* ``eye.learn`` (:class:`EyeLearn`) — how a walk through an earlier look's view went (FL-v1 learns
  the floor from ground that was walked without contact, owner decision D1).

duck-eye → duck-body:

* ``eye.result`` (:class:`EyeResult`) — ``status``:

  - ``ok``: a frame, paired with the head pose at its capture; ``results`` holds every task's
    integers (``light`` always).
  - ``retake``: the head turned more than :data:`HEAD_MOVED_MDEG` between the two pose samples
    around the capture (or the pose is missing); ask again with a new fd and ``attempt + 1``
    after ``retake_in_ms``. Given only while ``attempt < MAX_RETAKES``.
  - ``moved``: the same on the last allowed attempt: only the pose-free results (``light``).
  - ``none``: no frame (``why``: ``timeout``, ``deadline``, ``frame_error``, ``no_fd``,
    ``governor``, ``busy``, ``bad_request``). duck-body treats the camera as unusable for this
    look (spec §6: the eye being down is "black").

* ``eye.error`` ``{code, message, look_id?}`` — a line that could not be taken.

Nothing in here needs numpy, sockets or clocks: every class is plain data with a strict
``from_json`` (a :class:`ProtocolError` names the field) and a ``to_json`` whose output
``from_json`` reads back unchanged. :func:`integers_only` is the "integer features only" check
every ``eye.result`` passes before it is sent (P2: the same frame bytes give the same integers).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

API = "eye-v1"
EYE_SOCKET = "/run/duckmind/eye.sock"
MAX_LINE_BYTES = 65536

HELLO = "eye.hello"
LOOK = "eye.look"
POSE = "eye.pose"
HEALTH = "eye.health"
LEARN = "eye.learn"
RESULT = "eye.result"
ERROR = "eye.error"
TO_EYE = (HELLO, LOOK, POSE, HEALTH, LEARN)
FROM_EYE = (HELLO, RESULT, ERROR)

MEDIA_ID = 2  # duck-body said hello on the media connection with id 1
MAX_RETAKES = 2  # spec §4.3 step 5: at most two retakes
HEAD_MOVED_MDEG = 1000  # more than 1° between the samples around the capture: retake
RETAKE_IN_MS = 150  # the stillness window again before the retake

# What a look can ask for (spec §4.4). "light" runs on every look whatever was asked.
TASKS = ("light", "floor", "nest", "things", "duck")
# The look catalogue (spec §4.2) plus the light meter's lone frame (asleep, black: §4.6).
LOOK_KINDS = (
    "end_look",
    "glance_down",
    "scan",
    "look_up",
    "fixate",
    "seek_nest",
    "idle_glance",
    "meter",
)
SAFE_LOOKS = ("end_look", "glance_down")  # never skipped, whatever the heat (spec §4.6)
NEST_LOOKS = ("seek_nest", "fixate")  # "and going home" at 78–85 °C
# The duck's state for the look budget (spec §4.6's rows).
MODES = ("outing", "docking", "awake", "curio", "asleep", "dark")
CAM_FRAMES = ("sensor", "cv2_upright")  # eye.pose's camera frame (PoseSample.cam_frame)
STATUSES = ("ok", "retake", "moved", "none")
WHY = (
    "head_moved",
    "no_pose",
    "timeout",
    "deadline",
    "frame_error",
    "no_fd",
    "governor",
    "busy",
    "bad_request",
    "clock",
    "no_eye",  # duck-body's own answer while duck-eye is down (never sent by the eye)
)
OUTCOMES = ("clear", "contact", "stumble", "fell", "lifted", "stopped")
LIGHT_LEVELS = ("bright", "dim", "dark", "black")

ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:\-]{1,64}$")  # percept-v1's act and look ids
PV_PATTERN = re.compile(r"^[pcl]:[0-9a-f]{8,64}$")  # percept-v1's pv
U64_MAX = 2**64 - 1


class ProtocolError(ValueError):
    """A message that is not eye-v1; ``code`` is what ``eye.error`` says."""

    def __init__(self, message: str, code: str = "bad_request") -> None:
        super().__init__(message)
        self.code = code


# ── field readers: strict, and they name the field ─────────────────────────────


def _int(value: Any, name: str, lo: int | None = 0, hi: int | None = U64_MAX) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"{name} must be an integer")
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        raise ProtocolError(f"{name} {value} is out of range")
    return value


def _opt_int(value: Any, name: str, lo: int | None = 0, hi: int | None = U64_MAX) -> int | None:
    return None if value is None else _int(value, name, lo, hi)


def _num(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{name} must be a number")
    out = float(value)
    if not math.isfinite(out):
        raise ProtocolError(f"{name} must be finite")
    return out


def _nums(value: Any, name: str, size: int) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ProtocolError(f"{name} must be {size} numbers")
    return tuple(_num(v, f"{name}[{i}]") for i, v in enumerate(value))


def _id(value: Any, name: str) -> str:
    if not isinstance(value, str) or not ID_PATTERN.match(value):
        raise ProtocolError(f"{name} must be 1–64 of [A-Za-z0-9_.:-]")
    return value


def _choice(value: Any, name: str, options: Sequence[str]) -> str:
    if not isinstance(value, str) or value not in options:
        raise ProtocolError(f"{name} must be one of {', '.join(options)}")
    return value


def _unit_quat(value: Any, name: str) -> tuple[float, float, float, float]:
    """Scalar-first [w, x, y, z] (robotd's order), normalised; a zero quat is refused."""
    w, x, y, z = _nums(value, name, 4)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-6:
        raise ProtocolError(f"{name} is not a rotation")
    return (w / norm, x / norm, y / norm, z / norm)


def integers_only(value: Any, where: str = "results") -> list[str]:
    """Where a value holds anything but integers, strings, booleans, None, lists and objects
    with string keys: the "integer features only" rule (floats are refused)."""
    problems: list[str] = []
    if value is None or isinstance(value, (bool, str)):
        return problems
    if isinstance(value, int):
        if not -(2**63) <= value <= U64_MAX:
            problems.append(f"{where}: {value} does not fit 64 bits")
        return problems
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            problems += integers_only(item, f"{where}[{i}]")
        return problems
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                problems.append(f"{where}: key {key!r} is not a string")
            problems += integers_only(item, f"{where}.{key}")
        return problems
    problems.append(f"{where}: {type(value).__name__} is not an integer feature")
    return problems


# ── encoding ──────────────────────────────────────────────────────────────


def encode(message: Mapping[str, Any]) -> bytes:
    """One line: compact JSON, ASCII, newline-terminated, at most MAX_LINE_BYTES."""
    line = json.dumps(message, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    data = line.encode("ascii") + b"\n"
    if len(data) > MAX_LINE_BYTES:
        raise ProtocolError(f"a {message.get('type')} line of {len(data)} bytes is too long")
    return data


def decode(line: bytes | str) -> dict[str, Any]:
    """One line back to a message dict with a known ``type`` (not yet validated per type)."""
    if isinstance(line, bytes):
        if len(line) > MAX_LINE_BYTES:
            raise ProtocolError("line too long", "too_long")
        try:
            line = line.decode("utf-8")
        except UnicodeDecodeError:
            raise ProtocolError("line is not UTF-8", "parse") from None
    try:
        message = json.loads(line, parse_constant=_no_constant)
    except ValueError as error:
        raise ProtocolError(f"line is not JSON: {error}", "parse") from None
    if not isinstance(message, dict):
        raise ProtocolError("a message is a JSON object", "parse")
    kind = message.get("type")
    if kind not in TO_EYE and kind not in FROM_EYE:
        raise ProtocolError(f"unknown message type {kind!r}", "unknown_type")
    return message


def _no_constant(name: str) -> Any:
    raise ValueError(f"{name} is not JSON")


# ── duck-body → duck-eye ─────────────────────────────────────────────────


@dataclass(frozen=True)
class Hello:
    api: str = API
    role: str = "duck-body"

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> Hello:
        api = value.get("api")
        if api != API:
            raise ProtocolError(f"this eye speaks {API}, not {api!r}", "api")
        role = value.get("role", "duck-body")
        if not isinstance(role, str) or not role:
            raise ProtocolError("role must be a string")
        return cls(api=api, role=role)

    def to_json(self) -> dict[str, Any]:
        return {"type": HELLO, "api": self.api, "role": self.role}


@dataclass(frozen=True)
class EyeHello:
    """The eye's answer to ``eye.hello``."""

    perception_hash: str
    tasks: tuple[str, ...]
    api: str = API
    max_retakes: int = MAX_RETAKES

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> EyeHello:
        if value.get("api") != API:
            raise ProtocolError(f"the eye speaks {value.get('api')!r}, not {API}", "api")
        pv = value.get("perception_hash")
        if not isinstance(pv, str) or not PV_PATTERN.match(pv):
            raise ProtocolError("perception_hash must be p:<hex>")
        tasks = value.get("tasks")
        if not isinstance(tasks, list):
            raise ProtocolError("tasks must be a list")
        return cls(
            perception_hash=pv,
            tasks=tuple(_choice(t, "tasks[]", TASKS) for t in tasks),
            max_retakes=_int(value.get("max_retakes", MAX_RETAKES), "max_retakes", 0, 9),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "type": HELLO,
            "api": self.api,
            "perception_hash": self.perception_hash,
            "tasks": list(self.tasks),
            "max_retakes": self.max_retakes,
        }


def _hints(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ProtocolError("hints must be an object")
    problems = integers_only(value, "hints")
    if problems:
        raise ProtocolError("; ".join(problems[:3]))
    return dict(value)


@dataclass(frozen=True)
class EyeLook:
    """One look (spec §4.3 step 3): which frame, which tasks, which pose.

    * ``pose_tap_ref`` — the ``seq`` of the pose-tap sample at which duck-body found the head
      still (joints < 0.2 rad/s, gyro < 0.3 rad/s for 150 ms, spec §4.2 rule 2).
    * ``not_before_ns`` — do not ask for the frame before this (CLOCK_MONOTONIC).
    * ``deadline_ns`` — answer by then (status ``none``, why ``deadline``); ``None``: the media
      timeouts alone bound it.
    * ``hints`` — integers (or short strings) a task may use: ``nest_b_mrad`` (where duck-body
      believes the nest is, trunk frame), ``target_b_mrad``, ``target_id`` …
    """

    look_id: str
    tasks: tuple[str, ...]
    pose_tap_ref: int
    not_before_ns: int
    kind: str = "end_look"
    attempt: int = 0
    deadline_ns: int | None = None
    media_id: int = MEDIA_ID
    mode: str = "awake"
    hints: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _id(self.look_id, "look_id")
        if not self.tasks:
            raise ProtocolError("tasks is empty")
        for task in self.tasks:
            _choice(task, "tasks[]", TASKS)
        if len(set(self.tasks)) != len(self.tasks):
            raise ProtocolError("tasks repeats a task")
        _choice(self.kind, "kind", LOOK_KINDS)
        _choice(self.mode, "mode", MODES)
        _int(self.attempt, "attempt", 0, MAX_RETAKES)
        _int(self.pose_tap_ref, "pose_tap_ref")
        _int(self.not_before_ns, "not_before_ns")
        _opt_int(self.deadline_ns, "deadline_ns")
        _int(self.media_id, "media_id", 0)
        _hints(self.hints)

    @property
    def runs(self) -> tuple[str, ...]:
        """The tasks that run, in :data:`TASKS` order whatever order they were asked in:
        ``light`` first, always, then ``floor`` before ``things`` (which reuses its line)."""
        return ("light", *(t for t in TASKS if t in self.tasks and t != "light"))

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> EyeLook:
        tasks = value.get("tasks")
        if not isinstance(tasks, list):
            raise ProtocolError("tasks must be a list")
        return cls(
            look_id=_id(value.get("look_id"), "look_id"),
            tasks=tuple(_choice(t, "tasks[]", TASKS) for t in tasks),
            pose_tap_ref=_int(value.get("pose_tap_ref"), "pose_tap_ref"),
            not_before_ns=_int(value.get("not_before_ns"), "not_before_ns"),
            kind=_choice(value.get("kind", "end_look"), "kind", LOOK_KINDS),
            attempt=_int(value.get("attempt", 0), "attempt", 0, MAX_RETAKES),
            deadline_ns=_opt_int(value.get("deadline_ns"), "deadline_ns"),
            media_id=_int(value.get("media_id", MEDIA_ID), "media_id", 0),
            mode=_choice(value.get("mode", "awake"), "mode", MODES),
            hints=_hints(value.get("hints")),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "type": LOOK,
            "look_id": self.look_id,
            "attempt": self.attempt,
            "kind": self.kind,
            "tasks": list(self.tasks),
            "pose_tap_ref": self.pose_tap_ref,
            "not_before_ns": self.not_before_ns,
            "deadline_ns": self.deadline_ns,
            "media_id": self.media_id,
            "mode": self.mode,
            "hints": dict(self.hints),
        }

    def retake(self) -> EyeLook:
        """The same look, one attempt later (duck-body sends it with a new fd)."""
        return replace(self, attempt=self.attempt + 1)


IDENTITY_Q = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class PoseSample:
    """One pose-tap sample (duck-body's tap, 25 Hz; spec §4.3 step 5).

    * ``t_ns`` — CLOCK_MONOTONIC of the robotd tick it comes from (``RobotState.t_ns``).
    * ``cam_pos_m``, ``cam_quat`` — ``RobotState.frames.camera``: the OpenCV camera frame of the
      sensor as mounted (+x right, +y down, +z along the axis, *unrotated*: the frame K is
      solved in) in the trunk frame (x forward, y left, z up); position in metres, orientation
      scalar-first ``[w, x, y, z]`` taking camera vectors to trunk vectors.
    * ``imu_quat`` — ``RobotState.imu.quat``, trunk → world (gravity-aligned, arbitrary yaw).
    * ``trunk_height_mm`` — the trunk origin above the floor, from leg FK (duck-body).
    * ``head`` — measured head joints, radians: neck_pitch, head_pitch, head_yaw, head_roll.

    ``from_json`` accepts the spellings duck-body may use — ``camera`` / ``camera_in_trunk`` as
    ``{pos, quat}`` (robotd's PoseState) or ``cam_pos_mm`` + ``cam_quat_e6`` integers;
    ``imu_quat`` / ``imu: {quat}`` / ``imu_quat_e6``; ``head`` / ``head_joints`` /
    ``head_mrad`` — and the sample either flat in the ``eye.pose`` line or under ``sample``.
    """

    seq: int
    t_ns: int
    trunk_height_mm: int
    cam_pos_m: tuple[float, float, float]
    cam_quat: tuple[float, float, float, float]
    imu_quat: tuple[float, float, float, float] = IDENTITY_Q
    head: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    # Which camera frame ``cam_quat`` is in: "sensor" (the unrotated sensor frame above) or
    # "cv2_upright" — robotd's own ``frames.camera`` / ``HeadFk::camera_in_trunk_cv2``, the
    # upright cv2 frame (+x the duck's right), which the look turns to the sensor frame with its
    # frame's ``rotate`` (:meth:`duckmind.eye.frames.PairedPose.camera_pose`). duck-body sends
    # "cv2_upright" (measured 2026-09-27: the two differ by the mount's 90°).
    cam_frame: str = "sensor"

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> PoseSample:
        if isinstance(value.get("sample"), Mapping):
            value = value["sample"]
        seq = _int(value.get("seq", 0), "seq")
        t_ns = _int(value.get("t_ns"), "t_ns")
        camera_given = value.get("camera", value.get("camera_in_trunk"))
        if value.get("trunk_height_mm", value.get("trunk_height_m")) is None or (
            camera_given is None and value.get("cam_pos_mm") is None
        ):
            # duck-body sends null while leg FK or the head FK has no answer: not an error, but
            # no geometry either (the eye skips it; a look without a pose gives light only)
            raise ProtocolError(
                "the sample has no trunk height or no camera pose", "incomplete_pose"
            )
        if "trunk_height_mm" in value:
            height = _int(
                round(_num(value["trunk_height_mm"], "trunk_height_mm")), "trunk_height_mm"
            )
        elif "trunk_height_m" in value:
            height = int(round(_num(value["trunk_height_m"], "trunk_height_m") * 1000))
        else:
            raise ProtocolError("trunk_height_mm is missing")
        camera = value.get("camera", value.get("camera_in_trunk"))
        if isinstance(camera, Mapping):
            pos = _nums(camera.get("pos"), "camera.pos", 3)
            quat = _unit_quat(camera.get("quat"), "camera.quat")
        elif "cam_pos_mm" in value:
            pos = tuple(v / 1000.0 for v in _nums(value.get("cam_pos_mm"), "cam_pos_mm", 3))
            quat = _unit_quat(value.get("cam_quat_e6"), "cam_quat_e6")
        else:
            raise ProtocolError("the camera pose is missing")
        imu: Any = value.get("imu_quat")
        if imu is None and isinstance(value.get("imu"), Mapping):
            imu = value["imu"].get("quat")
        if imu is None:
            imu = value.get("imu_quat_e6")
        imu_q = _unit_quat(imu, "imu_quat") if imu is not None else IDENTITY_Q
        head: Any = value.get("head", value.get("head_joints"))
        if head is not None:
            joints = _nums(head, "head", 4)
        elif "head_mrad" in value:
            joints = tuple(v / 1000.0 for v in _nums(value["head_mrad"], "head_mrad", 4))
        else:
            joints = (0.0, 0.0, 0.0, 0.0)
        return cls(
            seq=seq,
            t_ns=t_ns,
            trunk_height_mm=height,
            cam_pos_m=(pos[0], pos[1], pos[2]),
            cam_quat=quat,
            imu_quat=imu_q,
            head=(joints[0], joints[1], joints[2], joints[3]),
            cam_frame=_choice(value.get("cam_frame", "sensor"), "cam_frame", CAM_FRAMES),
        )

    def to_json(self) -> dict[str, Any]:
        out = {
            "type": POSE,
            "seq": self.seq,
            "t_ns": self.t_ns,
            "trunk_height_mm": self.trunk_height_mm,
            "camera": {"pos": list(self.cam_pos_m), "quat": list(self.cam_quat)},
            "imu_quat": list(self.imu_quat),
            "head": list(self.head),
        }
        if self.cam_frame != "sensor":
            out["cam_frame"] = self.cam_frame
        return out


@dataclass(frozen=True)
class CpuThrottle:
    """``duck_ipc_proto::CpuThrottle``: the cooling level and the clock ceiling."""

    level: int = 0
    max_level: int = 0
    khz: int = 0
    max_khz: int = 0

    @property
    def throttled(self) -> bool:
        """``CpuThrottle::throttled``: the governor wound the level up, or the ceiling is below
        the hardware maximum (either reading counts)."""
        return self.level > 0 or (self.max_khz > 0 and self.khz < self.max_khz)

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> CpuThrottle:
        return cls(
            **{
                name: _int(value.get(name, 0), f"cpu_throttle.{name}", 0, 2**32 - 1)
                for name in ("level", "max_level", "khz", "max_khz")
            }
        )

    def to_json(self) -> dict[str, int]:
        return {
            "level": self.level,
            "max_level": self.max_level,
            "khz": self.khz,
            "max_khz": self.max_khz,
        }


@dataclass(frozen=True)
class EyeHealth:
    """``robot.health``'s thermal part (``cpu_temp_c``, ``cpu_throttle``; both absent off
    Linux, which the governor reads as the normal tier)."""

    t_ns: int
    cpu_temp_c: float | None = None
    cpu_throttle: CpuThrottle | None = None

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> EyeHealth:
        temp = value.get("cpu_temp_c")
        throttle = value.get("cpu_throttle")
        if throttle is not None and not isinstance(throttle, Mapping):
            raise ProtocolError("cpu_throttle must be an object")
        return cls(
            t_ns=_int(value.get("t_ns", 0), "t_ns"),
            cpu_temp_c=None if temp is None else _num(temp, "cpu_temp_c"),
            cpu_throttle=CpuThrottle.from_json(throttle) if throttle is not None else None,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "type": HEALTH,
            "t_ns": self.t_ns,
            "cpu_temp_c": self.cpu_temp_c,
            "cpu_throttle": self.cpu_throttle.to_json() if self.cpu_throttle else None,
        }


@dataclass(frozen=True)
class EyeLearn:
    """How a walk through an earlier look's view went (owner decision D1).

    * ``look_id`` — the look (usually the pre-departure one) whose view the walk crossed;
    * ``outcome`` — :data:`OUTCOMES`; only ``clear`` ground becomes floor samples;
    * ``floor`` — ground points the duck walked over, ``[x_mm, y_mm]`` in the level trunk frame
      *at that look's capture* (x forward, y left, on the floor), in walking order;
    * ``width_mm`` — the corridor the body swept around those points;
    * ``contact`` — where a contact happened (same frame), if one did.
    """

    walk_id: str
    look_id: str
    outcome: str
    floor: tuple[tuple[int, int], ...]
    width_mm: int = 120
    contact: tuple[int, int] | None = None

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> EyeLearn:
        points = value.get("floor", [])
        if not isinstance(points, list) or len(points) > 512:
            raise ProtocolError("floor must be a list of at most 512 [x_mm, y_mm]")
        floor = []
        for i, point in enumerate(points):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ProtocolError(f"floor[{i}] must be [x_mm, y_mm]")
            floor.append(
                (
                    _int(point[0], f"floor[{i}][0]", -100_000, 100_000),
                    _int(point[1], f"floor[{i}][1]", -100_000, 100_000),
                )
            )
        contact = value.get("contact")
        if contact is not None:
            if not isinstance(contact, (list, tuple)) or len(contact) != 2:
                raise ProtocolError("contact must be [x_mm, y_mm]")
            contact = (
                _int(contact[0], "contact[0]", -100_000, 100_000),
                _int(contact[1], "contact[1]", -100_000, 100_000),
            )
        return cls(
            walk_id=_id(value.get("walk_id"), "walk_id"),
            look_id=_id(value.get("look_id"), "look_id"),
            outcome=_choice(value.get("outcome"), "outcome", OUTCOMES),
            floor=tuple(floor),
            width_mm=_int(value.get("width_mm", 120), "width_mm", 0, 2000),
            contact=contact,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "type": LEARN,
            "walk_id": self.walk_id,
            "look_id": self.look_id,
            "outcome": self.outcome,
            "floor": [list(p) for p in self.floor],
            "width_mm": self.width_mm,
            "contact": list(self.contact) if self.contact is not None else None,
        }


# ── duck-eye → duck-body ─────────────────────────────────────────────────


@dataclass(frozen=True)
class EyeResult:
    """The answer to one ``eye.look`` (spec §4.3 steps 8–9). Integers only.

    * ``captured_at_ns`` — the capture on CLOCK_MONOTONIC (bridged from the header's wall clock);
    * ``pose`` — the head pose paired at the capture (:meth:`duckmind.eye.frames.PairedPose.to_json`):
      ``t_ns``, ``trunk_height_mm``, ``cam_pos_mm``, ``cam_quat_e6``, ``imu_quat_e6``,
      ``head_mrad``, ``moved_mdeg`` (how far the camera turned between the two samples),
      ``interp`` (``lerp`` or ``hold``) — ``None`` when there was none;
    * ``results`` — ``{task: {…}}``; ``light`` is ``{level, luma, noise_q, raw, glow,
      settle_ms}`` (:mod:`duckmind.eye.light`); the other tasks' shapes are their modules';
    * ``frame`` — ``{w, h, rotate, sha16}`` (the first 16 hex digits of the bytes' SHA-256, for
      audit only; the bytes are gone);
    * ``cost_us`` — microseconds per stage (``wait``, ``fetch``, ``sample`` and each task);
    * ``gov`` — the governor's tier and permissions at this look (:mod:`duckmind.eye.governor`).
    """

    look_id: str
    attempt: int
    kind: str
    status: str
    pv: str
    why: str | None = None
    retake_in_ms: int | None = None
    captured_at_ns: int | None = None
    captured_at_unix_us: int | None = None
    pose: dict[str, Any] | None = None
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    frame: dict[str, Any] | None = None
    cost_us: dict[str, int] = field(default_factory=dict)
    gov: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _id(self.look_id, "look_id")
        _choice(self.status, "status", STATUSES)
        if self.why is not None:
            _choice(self.why, "why", WHY)
        if self.status != "ok" and self.why is None:
            raise ProtocolError(f"status {self.status} needs a why")
        if not PV_PATTERN.match(self.pv):
            raise ProtocolError("pv must be p:<hex>")

    def problems(self) -> list[str]:
        out = integers_only(self.results, "results")
        out += integers_only(self.pose, "pose")
        out += integers_only(self.frame, "frame")
        out += integers_only(self.cost_us, "cost_us")
        out += integers_only(self.gov, "gov")
        return out

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> EyeResult:
        results = value.get("results") or {}
        if not isinstance(results, Mapping) or not all(
            isinstance(v, Mapping) for v in results.values()
        ):
            raise ProtocolError("results must be an object of objects")
        for task in results:
            _choice(task, "results key", TASKS)
        pose = value.get("pose")
        frame = value.get("frame")
        cost = value.get("cost_us") or {}
        gov = value.get("gov") or {}
        out = cls(
            look_id=_id(value.get("look_id"), "look_id"),
            attempt=_int(value.get("attempt", 0), "attempt", 0, MAX_RETAKES),
            kind=_choice(value.get("kind", "end_look"), "kind", LOOK_KINDS),
            status=_choice(value.get("status"), "status", STATUSES),
            pv=str(value.get("pv", "")),
            why=None if value.get("why") is None else _choice(value.get("why"), "why", WHY),
            retake_in_ms=_opt_int(value.get("retake_in_ms"), "retake_in_ms"),
            captured_at_ns=_opt_int(value.get("captured_at_ns"), "captured_at_ns"),
            captured_at_unix_us=_opt_int(value.get("captured_at_unix_us"), "captured_at_unix_us"),
            pose=dict(pose) if isinstance(pose, Mapping) else None,
            results={k: dict(v) for k, v in results.items()},
            frame=dict(frame) if isinstance(frame, Mapping) else None,
            cost_us=dict(cost) if isinstance(cost, Mapping) else {},
            gov=dict(gov) if isinstance(gov, Mapping) else {},
        )
        problems = out.problems()
        if problems:
            raise ProtocolError("; ".join(problems[:3]))
        return out

    def to_json(self) -> dict[str, Any]:
        return {
            "type": RESULT,
            "look_id": self.look_id,
            "attempt": self.attempt,
            "kind": self.kind,
            "status": self.status,
            "why": self.why,
            "retake_in_ms": self.retake_in_ms,
            "captured_at_ns": self.captured_at_ns,
            "captured_at_unix_us": self.captured_at_unix_us,
            "pose": self.pose,
            "results": self.results,
            "frame": self.frame,
            "cost_us": self.cost_us,
            "gov": self.gov,
            "pv": self.pv,
        }

    @property
    def light(self) -> dict[str, Any] | None:
        return self.results.get("light")


def error(code: str, message: str, look_id: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"type": ERROR, "code": code, "message": message}
    if look_id is not None:
        out["look_id"] = look_id
    return out


def parse_to_eye(message: Mapping[str, Any]) -> Hello | EyeLook | PoseSample | EyeHealth | EyeLearn:
    """A decoded duck-body line as its class (:class:`ProtocolError` otherwise)."""
    kind = message.get("type")
    if kind == HELLO:
        return Hello.from_json(message)
    if kind == LOOK:
        return EyeLook.from_json(message)
    if kind == POSE:
        return PoseSample.from_json(message)
    if kind == HEALTH:
        return EyeHealth.from_json(message)
    if kind == LEARN:
        return EyeLearn.from_json(message)
    raise ProtocolError(f"{kind!r} is not something duck-body sends", "unknown_type")


def task_names(values: Iterable[str]) -> tuple[str, ...]:
    """Known task names, in :data:`TASKS` order, each once."""
    wanted = set(values)
    return tuple(t for t in TASKS if t in wanted)
