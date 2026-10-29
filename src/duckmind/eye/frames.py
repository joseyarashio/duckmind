"""duck-eye's frames, v1: one camera frame per look, paired with the head pose (spec §4.3).

The same client for the board and the simulator (spec R3): on a duck the socket is mediad's
``/run/mediad/media.sock``; beside the simulator it is the world's media-shim
(:mod:`duckmind.world.media_shim`, whose path the world server reports in ``x_sensors``). It is
the official client's shape (``robotctl/src/frame.rs``): connect, ``hello``, one ``media.frame``
request, a header line of at most 4096 bytes that must answer that request's id with JSON-RPC
2.0 and describe a valid UYVY frame (``MediaFrameHeader::valid_uyvy``), then exactly ``bytes``
raw bytes. One connection per frame, as mediad serves them; 3 s socket timeouts, as robotctl.

Nothing here knows the simulator: the header carries no exposure and no physics time, and this
module imports neither :mod:`duckmind.world` nor MuJoCo. It opens no socket of its own for a
look and reads no clock: :mod:`duckmind.eye.server` owns those and passes them in.

* :func:`fetch` — one frame on a connection of its own (the page's eye window, tools).
* :func:`fetch_on` — one frame on a connection duck-body opened, said ``hello`` on and handed
  over (SCM_RIGHTS): the request id carries on from duck-body's hello (``media_id``, 2).
* :func:`send_with_fd` / :class:`FdLineReader` — the hand-over itself: one fd riding on one
  ``eye.look`` line, and the eye's reader that gives each look line the fd that came with it.
* :class:`ClockPair` / :class:`ClockBridge` — the frame's wall-clock stamp onto CLOCK_MONOTONIC
  (robotd's clock), from REALTIME/MONOTONIC pairs read before and after the look.
* :class:`PoseTap` / :func:`pair_pose` — the pose-tap samples around the capture, interpolated
  to it; more than 1° of camera turn between them means retake (spec §4.3 step 5).
* :func:`sample_quarter` — the one-pass upright sampler: quarter-resolution luma (180×320 at
  720p) and chroma (90×160) straight from the UYVY bytes, shared by every task; :func:`roi_luma`
  cuts a full-resolution upright box (tags); :func:`upright_luma`, :func:`upright_rgb` sample
  the whole frame every ``step``-th pixel.
* :class:`LookView` — what a task gets: the look, the frame (for this look only), the camera
  model, the planes, the paired pose, the light reading.
* :class:`EyeWindow` — the simulator page's eye window (owner decision F4): it fetches a frame
  only while the window is open (the page touched it recently), and nothing at all otherwise.

**The one exemption from the phase-3 rule** "sockets, threads and clocks only in ``server.py``":
the socket helpers (:func:`fetch`, :func:`fetch_on`, :func:`send_with_fd`, :class:`FdLineReader`)
and :class:`EyeWindow`'s thread and clock serve the *page* (phase 0) and the server's I/O; no
look's result depends on them — the pure path (:func:`sample_quarter` → the tasks) takes bytes,
poses and times as arguments, and P2 compares every result field but ``cost_us`` (the timings).
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import socket
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from duckmind.eye.camera import CameraPose, EyeCamera, q_angle, q_mul, q_normalized, q_slerp
from duckmind.eye.protocol import EyeLook, PoseSample
from duckmind.i18n import msg

API_VERSION = 37  # duck-ipc-proto API_VERSION
MEDIA_SOCKET = "/run/mediad/media.sock"  # duck-ipc-proto socket::MEDIA
MEDIA_FRAME = "media.frame"
HELLO = "hello"
JSONRPC = "2.0"
MAX_HEADER_BYTES = 4096  # robotctl reads at most this, newline included
TIMEOUT_S = 3.0  # robotctl's read and write timeouts
MAX_FRAME_BYTES = 16 * 1024 * 1024
FORMAT = "UYVY"
# BT.601 full range, the inverse of the camera's packing (official sim/camera.py)
_R_V, _G_U, _G_V, _B_U = 1.402, 0.344136, 0.714136, 1.772


class FrameError(RuntimeError):
    """No frame: the endpoint said why (``code``), or the reply was not a valid frame."""

    def __init__(self, message: str, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class MediaFrameHeader:
    """``duck-ipc-proto`` ``MediaFrameHeader``: the metadata before the binary tail."""

    width: int
    height: int
    format: str
    bytes: int
    captured_at_unix_us: int
    rotate: int

    @classmethod
    def from_json(cls, value: Any) -> MediaFrameHeader:
        """As serde reads it: every field present with its type (u32, String, usize, u128)."""
        if not isinstance(value, dict):
            raise FrameError("missing snapshot result")
        out: dict[str, Any] = {}
        for name in ("width", "height", "bytes", "captured_at_unix_us", "rotate"):
            item = value.get(name)
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise FrameError(f"snapshot header field {name!r} is not an unsigned integer")
            out[name] = item
        if not isinstance(value.get("format"), str):
            raise FrameError("snapshot header field 'format' is not a string")
        return cls(format=value["format"], **out)

    def valid_uyvy(self) -> bool:
        """Bound allocation and reject malformed geometry before decoding pixels."""
        return (
            self.width > 0
            and self.height > 0
            and self.width % 2 == 0
            and self.format == FORMAT
            and self.rotate in (0, 90, 180, 270)
            and self.bytes <= MAX_FRAME_BYTES
            and self.width * self.height * 2 == self.bytes
        )

    @property
    def upright_size(self) -> tuple[int, int]:
        """(width, height) once turned upright: a quarter turn swaps them."""
        if self.rotate in (90, 270):
            return self.height, self.width
        return self.width, self.height

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "format": self.format,
            "bytes": self.bytes,
            "captured_at_unix_us": self.captured_at_unix_us,
            "rotate": self.rotate,
        }


def _request(request_id: int, method: str, params: Any = None) -> bytes:
    message: dict[str, Any] = {"jsonrpc": JSONRPC, "id": request_id, "method": method}
    if params is not None:
        message["params"] = params
    return json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\n"


class _Reader:
    """A buffered reader over a socket: bounded lines, then exact byte counts."""

    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.buffer = bytearray()

    def line(self, limit: int) -> bytes:
        """Up to and including the first newline, reading at most ``limit`` bytes of it."""
        while True:
            end = self.buffer.find(b"\n", 0, limit)
            if end >= 0:
                out = bytes(self.buffer[: end + 1])
                del self.buffer[: end + 1]
                return out
            if len(self.buffer) >= limit:
                out = bytes(self.buffer[:limit])
                del self.buffer[:limit]
                return out
            chunk = self.sock.recv(65536)
            if not chunk:
                out = bytes(self.buffer)
                self.buffer.clear()
                return out
            self.buffer += chunk

    def exact(self, size: int) -> bytes:
        out = bytearray(self.buffer[:size])
        del self.buffer[:size]
        view = memoryview(bytearray(size - len(out)))
        got = 0
        while got < len(view):
            n = self.sock.recv_into(view[got:], len(view) - got)
            if n == 0:
                raise FrameError(f"the frame ended after {len(out) + got} of {size} bytes")
            got += n
        out += view
        return bytes(out)


def _response(reader: _Reader, request_id: int, what: str) -> dict[str, Any]:
    line = reader.line(MAX_HEADER_BYTES + 1)
    if len(line) > MAX_HEADER_BYTES or not line.endswith(b"\n"):
        raise FrameError(f"invalid {what} length")
    try:
        response = json.loads(line)
    except (ValueError, UnicodeDecodeError) as error:
        raise FrameError(f"{what} is not JSON: {error}") from None
    if not isinstance(response, dict):
        raise FrameError(f"{what} is not a JSON-RPC response")
    if response.get("id") != request_id or response.get("jsonrpc") != JSONRPC:
        raise FrameError(f"{what} does not match the request")
    refusal = response.get("error")
    if refusal is not None:
        code = refusal.get("code") if isinstance(refusal, dict) else None
        message = refusal.get("message") if isinstance(refusal, dict) else str(refusal)
        raise FrameError(f"{code}: {message}", code if isinstance(code, int) else None)
    return response


def fetch(
    socket_path: Path | str = MEDIA_SOCKET,
    *,
    timeout_s: float = TIMEOUT_S,
    hello: bool = True,
) -> tuple[MediaFrameHeader, bytes]:
    """One frame off ``media.frame``: connect, (``hello``,) ask for the next frame, read the
    header and exactly its bytes. Raises :class:`FrameError` (or ``OSError`` when the socket
    cannot be reached)."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout_s)
        sock.connect(str(socket_path))
        reader = _Reader(sock)
        request_id = 1
        if hello:
            say_hello(sock, reader, request_id)
            request_id += 1
        return _frame(sock, reader, request_id)


def say_hello(sock: socket.socket, reader: _Reader | None = None, request_id: int = 1) -> None:
    """``hello`` on a fresh media connection (what duck-body does before handing it over)."""
    reader = reader if reader is not None else _Reader(sock)
    sock.sendall(_request(request_id, HELLO, {"api_version": API_VERSION}))
    answer = _response(reader, request_id, "hello answer")
    result = answer.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("api_version"), int):
        raise FrameError("daemon answered hello in an unexpected shape")
    if reader.buffer:
        raise FrameError("the daemon said more than its hello answer")


def _frame(sock: socket.socket, reader: _Reader, request_id: int) -> tuple[MediaFrameHeader, bytes]:
    sock.sendall(_request(request_id, MEDIA_FRAME))
    response = _response(reader, request_id, "snapshot header")
    head = MediaFrameHeader.from_json(response.get("result"))
    if not head.valid_uyvy():
        raise FrameError("invalid UYVY geometry or size")
    return head, reader.exact(head.bytes)


def fetch_on(
    sock: socket.socket, *, media_id: int = 2, timeout_s: float = TIMEOUT_S
) -> tuple[MediaFrameHeader, bytes]:
    """One frame on a media connection someone else opened and said ``hello`` on (duck-body,
    handing it over with SCM_RIGHTS): ``media.frame`` with id ``media_id``, then the header and
    exactly its bytes. The caller closes the socket (one frame per connection)."""
    sock.settimeout(timeout_s)
    return _frame(sock, _Reader(sock), media_id)


# ── the fd hand-over (SCM_RIGHTS) ────────────────────────────────────────────────

MAX_FDS_PER_READ = 8


def send_with_fd(sock: socket.socket, line: bytes, fd: int) -> None:
    """Send one line with one fd riding on it (duck-body's side of the hand-over): the fd goes
    in the same ``sendmsg`` as the line's first bytes; the rest follows if the kernel took only
    part of it."""
    sent = socket.send_fds(sock, [line], [fd])
    if sent < len(line):
        sock.sendall(line[sent:])


class FdLineReader:
    """Lines from a unix stream socket, and the fds that arrived with them.

    An fd sent with a line's bytes arrives with the ``recvmsg`` that holds that line's first
    byte — on Linux possibly after earlier lines queued before it (the kernel glues data without
    fds in front of the chunk that carries them, and stops right after it). So each received fd
    is kept with the stream range [start, end) of the read that brought it; :meth:`take_fd`
    gives a look line starting at offset S the oldest waiting fd whose read covers S, and
    :meth:`drop_stale` closes fds whose read ended before the line being handled started (no
    later line can claim them). :meth:`close` closes every fd nobody took: they must never leak
    into the eye.
    """

    def __init__(self, sock: socket.socket, *, max_line: int = 65536) -> None:
        self.sock = sock
        self.max_line = max_line
        self.buffer = bytearray()
        self.offset = 0  # stream offset of buffer[0]
        self.fds: list[tuple[int, int, int]] = []  # (start, end) of the read that brought it, fd
        self.eof = False

    def _recv(self) -> tuple[bytes, list[int]]:
        """One read: the bytes and the fds that came with them."""
        data, fds, _flags, _addr = socket.recv_fds(self.sock, 65536, MAX_FDS_PER_READ)
        return data, list(fds)

    def _fill(self) -> None:
        data, fds = self._recv()
        start = self.offset + len(self.buffer)
        end = start + max(1, len(data))
        self.fds += [(start, end, fd) for fd in fds]
        if not data:
            self.eof = True
            return
        self.buffer += data

    def read_line(self) -> tuple[bytes, int] | None:
        """The next line (newline included) and its stream offset; ``None`` at end of stream.
        Raises ``ValueError`` for a line longer than ``max_line``."""
        while True:
            end = self.buffer.find(b"\n")
            if end >= 0:
                if end + 1 > self.max_line:
                    raise ValueError("line too long")
                line = bytes(self.buffer[: end + 1])
                start = self.offset
                del self.buffer[: end + 1]
                self.offset += end + 1
                return line, start
            if len(self.buffer) > self.max_line:
                raise ValueError("line too long")
            if self.eof:
                return None
            self._fill()

    def take_fd(self, line_offset: int) -> int | None:
        """The fd that came with the line starting at ``line_offset`` (``None``: none came)."""
        for i, (start, end, fd) in enumerate(self.fds):
            if start <= line_offset < end:
                del self.fds[i]
                return fd
        return None

    def drop_stale(self, line_offset: int) -> int:
        """Close the fds no line from ``line_offset`` on can claim (their read ended before it)."""
        import os

        kept, closed = [], 0
        for start, end, fd in self.fds:
            if end <= line_offset:
                os.close(fd)
                closed += 1
            else:
                kept.append((start, end, fd))
        self.fds = kept
        return closed

    def close(self) -> None:
        import os

        for _start, _end, fd in self.fds:
            try:
                os.close(fd)
            except OSError:
                pass
        self.fds = []


# ── the clock bridge (spec §4.3 step 5) ──────────────────────────────────────────

CLOCK_STEP_TOLERANCE_NS = 2_000_000  # the wall clock moved against CLOCK_MONOTONIC in a look
CAPTURE_SLACK_NS = 50_000_000  # a capture before the request (one 30 fps frame and a bit)


@dataclass(frozen=True)
class ClockPair:
    """``CLOCK_REALTIME`` and ``CLOCK_MONOTONIC`` read together (ns)."""

    realtime_ns: int
    monotonic_ns: int

    @property
    def offset_ns(self) -> int:
        return self.realtime_ns - self.monotonic_ns


@dataclass(frozen=True)
class ClockBridge:
    """Two pairs, before the frame request and after its bytes: the wall clock's offset from
    CLOCK_MONOTONIC over the look, and whether it held still (NTP may step or slew it)."""

    before: ClockPair
    after: ClockPair

    @property
    def offset_ns(self) -> int:
        return (self.before.offset_ns + self.after.offset_ns) // 2

    @property
    def drift_ns(self) -> int:
        return abs(self.after.offset_ns - self.before.offset_ns)

    @property
    def steady(self) -> bool:
        return self.drift_ns <= CLOCK_STEP_TOLERANCE_NS

    def to_monotonic_ns(self, unix_us: int) -> int:
        """A wall-clock stamp (µs since the epoch) on CLOCK_MONOTONIC."""
        return unix_us * 1000 - self.offset_ns

    def capture_ns(self, header: MediaFrameHeader) -> int | None:
        """The frame's capture on CLOCK_MONOTONIC, or ``None`` when it cannot be trusted: the
        wall clock moved during the look, or the stamp falls outside the request … answer
        window (the next frame cannot be captured before the request nor after its bytes)."""
        if not self.steady:
            return None
        at = self.to_monotonic_ns(header.captured_at_unix_us)
        if at < self.before.monotonic_ns - CAPTURE_SLACK_NS or at > self.after.monotonic_ns:
            return None
        return at


def wait_ns(not_before_ns: int, now_ns: int) -> int:
    """How long to wait before asking for the frame (``not_before_ns``)."""
    return max(0, not_before_ns - now_ns)


# ── pose pairing (spec §4.3 step 5) ─────────────────────────────────────────────

TAP_PERIOD_NS = 40_000_000  # the pose tap's 25 Hz
MAX_GAP_NS = 100_000_000  # two samples further apart than this do not bracket a capture
HOLD_NS = 40_000_000  # one sample this close stands for the capture when no later one comes
POSE_WAIT_NS = 120_000_000  # how long after the capture to wait for the sample after it
TAP_KEEP = 64  # samples kept: 2.5 s at 25 Hz


class PoseTap:
    """The latest pose-tap samples, in time order (a ring of :data:`TAP_KEEP`)."""

    def __init__(self, keep: int = TAP_KEEP) -> None:
        self.keep = keep
        self.samples: list[PoseSample] = []

    def add(self, sample: PoseSample) -> None:
        if self.samples and sample.t_ns <= self.samples[-1].t_ns:
            if sample.t_ns == self.samples[-1].t_ns:
                return
            # out of order or a restarted tap: keep time order
            self.samples = [s for s in self.samples if s.t_ns < sample.t_ns]
        self.samples.append(sample)
        if len(self.samples) > self.keep:
            del self.samples[: len(self.samples) - self.keep]

    def latest_seq(self) -> int | None:
        return self.samples[-1].seq if self.samples else None


@dataclass(frozen=True)
class PairedPose:
    """The head pose at the capture instant."""

    t_ns: int
    trunk_height_mm: int
    cam_pos_m: tuple[float, float, float]
    cam_quat: tuple[float, float, float, float]
    imu_quat: tuple[float, float, float, float]
    head: tuple[float, float, float, float]
    moved_mdeg: int  # how far the camera turned (in the world) between the samples used
    interp: str  # "lerp" between the samples around the capture, "hold" one sample
    seqs: tuple[int, int]
    cam_frame: str = "sensor"  # the samples' camera frame (protocol.PoseSample.cam_frame)

    def camera_pose(self, rotate: int = 0) -> CameraPose:
        """The camera pose in the eye's sensor frame. robotd's upright cv2 frame
        (``cam_frame`` "cv2_upright") is turned by the frame's ``rotate`` about the optical
        axis — the sensor is mounted that far from upright."""
        quat = self.cam_quat
        if self.cam_frame == "cv2_upright" and rotate % 360:
            half = math.radians(rotate % 360) / 2.0
            quat = q_normalized(q_mul(quat, (math.cos(half), 0.0, 0.0, math.sin(half))))
        return CameraPose(
            cam_pos_m=self.cam_pos_m,
            cam_quat=quat,
            imu_quat=self.imu_quat,
            trunk_height_m=self.trunk_height_mm / 1000.0,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "t_ns": self.t_ns,
            "trunk_height_mm": self.trunk_height_mm,
            "cam_pos_mm": [round(v * 1000) for v in self.cam_pos_m],
            "cam_quat_e6": [round(v * 1_000_000) for v in self.cam_quat],
            "imu_quat_e6": [round(v * 1_000_000) for v in self.imu_quat],
            "head_mrad": [round(v * 1000) for v in self.head],
            "moved_mdeg": self.moved_mdeg,
            "interp": self.interp,
            "seq": list(self.seqs),
        }


@dataclass(frozen=True)
class Pairing:
    """What :func:`pair_pose` found: ``ok`` (``pose`` set), ``moved`` (the camera turned more
    than the limit; ``pose`` still set, for the record), ``pending`` (the sample after the
    capture may still come), ``no_pose``."""

    status: str
    pose: PairedPose | None = None


def _world_quat(sample: PoseSample) -> tuple[float, float, float, float]:
    return q_normalized(q_mul(sample.imu_quat, sample.cam_quat))


def _moved_mdeg(a: PoseSample, b: PoseSample) -> int:
    return round(math.degrees(q_angle(_world_quat(a), _world_quat(b))) * 1000)


def _lerp(a: float, b: float, f: float) -> float:
    return a + (b - a) * f


def pair_pose(
    samples: Sequence[PoseSample],
    capture_ns: int,
    now_ns: int,
    *,
    limit_mdeg: int = 1000,
    max_gap_ns: int = MAX_GAP_NS,
    hold_ns: int = HOLD_NS,
    wait_ns_after: int = POSE_WAIT_NS,
) -> Pairing:
    """The pose at ``capture_ns`` from time-ordered pose-tap samples.

    The two samples around the capture (at most ``max_gap_ns`` apart) are interpolated to it —
    positions and joints linearly, orientations by slerp — unless the camera turned more than
    ``limit_mdeg`` between them (``moved``: retake). With no sample after the capture yet the
    answer is ``pending`` until ``wait_ns_after`` has passed; then one sample within ``hold_ns``
    before the capture stands for it (``hold``), checked for movement against the sample before
    it. A capture before the first sample is held by that sample the same way.
    """
    before = [s for s in samples if s.t_ns <= capture_ns]
    after = [s for s in samples if s.t_ns > capture_ns]
    if before and after and after[0].t_ns - before[-1].t_ns <= max_gap_ns:
        s0, s1 = before[-1], after[0]
        moved = _moved_mdeg(s0, s1)
        f = (capture_ns - s0.t_ns) / (s1.t_ns - s0.t_ns)
        pose = PairedPose(
            t_ns=capture_ns,
            trunk_height_mm=round(_lerp(s0.trunk_height_mm, s1.trunk_height_mm, f)),
            cam_pos_m=(
                _lerp(s0.cam_pos_m[0], s1.cam_pos_m[0], f),
                _lerp(s0.cam_pos_m[1], s1.cam_pos_m[1], f),
                _lerp(s0.cam_pos_m[2], s1.cam_pos_m[2], f),
            ),
            cam_quat=q_slerp(s0.cam_quat, s1.cam_quat, f),
            imu_quat=q_slerp(s0.imu_quat, s1.imu_quat, f),
            head=(
                _lerp(s0.head[0], s1.head[0], f),
                _lerp(s0.head[1], s1.head[1], f),
                _lerp(s0.head[2], s1.head[2], f),
                _lerp(s0.head[3], s1.head[3], f),
            ),
            moved_mdeg=moved,
            interp="lerp",
            seqs=(s0.seq, s1.seq),
            cam_frame=s0.cam_frame,
        )
        return Pairing("moved" if moved > limit_mdeg else "ok", pose)
    if not after and now_ns < capture_ns + wait_ns_after:
        return Pairing("pending")
    held: PoseSample | None = None
    neighbour: PoseSample | None = None
    if before and capture_ns - before[-1].t_ns <= hold_ns:
        held = before[-1]
        neighbour = before[-2] if len(before) > 1 else None
    elif after and after[0].t_ns - capture_ns <= hold_ns:
        held = after[0]
        neighbour = after[1] if len(after) > 1 else None
    if held is None:
        return Pairing("no_pose")
    moved = (
        _moved_mdeg(neighbour, held)
        if neighbour is not None and abs(neighbour.t_ns - held.t_ns) <= max_gap_ns
        else 0
    )
    pose = PairedPose(
        t_ns=capture_ns,
        trunk_height_mm=held.trunk_height_mm,
        cam_pos_m=held.cam_pos_m,
        cam_quat=held.cam_quat,
        imu_quat=held.imu_quat,
        head=held.head,
        moved_mdeg=moved,
        interp="hold",
        seqs=(held.seq, held.seq),
        cam_frame=held.cam_frame,
    )
    return Pairing("moved" if moved > limit_mdeg else "ok", pose)


# ── sampling the UYVY bytes upright (uyvy::Turn) ──────────────────────────────────


def _planes(data: bytes, width: int, height: int) -> tuple[Any, Any, Any]:
    """Y (h, w), U and V (h, w/2) as uint8 views of packed UYVY."""
    import numpy as np

    packed = np.frombuffer(data, dtype=np.uint8)
    if packed.size != width * height * 2:
        raise FrameError(f"{packed.size} bytes is not a {width}x{height} UYVY frame")
    quads = packed.reshape(height, width // 2, 4)
    y = packed[1::2].reshape(height, width)
    return y, quads[:, :, 0], quads[:, :, 2]


def _turn(plane: Any, rotate: int) -> Any:
    """The plane upright: ``rotate`` degrees clockwise (uyvy::Turn)."""
    import numpy as np

    turns = {0: 0, 90: -1, 180: 2, 270: 1}
    if rotate not in turns:
        raise FrameError(f"rotate {rotate} is not a quarter turn")
    return np.rot90(plane, k=turns[rotate])


def upright_luma(data: bytes, width: int, height: int, rotate: int = 90, *, step: int = 1) -> Any:
    """The luma plane upright, every ``step``-th pixel (step 4 at 720p: 180×320)."""
    import numpy as np

    y, _, _ = _planes(data, width, height)
    return np.ascontiguousarray(_turn(y, rotate)[::step, ::step])


def upright_rgb(data: bytes, width: int, height: int, rotate: int = 90, *, step: int = 1) -> Any:
    """RGB (uint8) upright, every ``step``-th pixel, chroma from the pixel's pair (BT.601 full
    range, the inverse of the camera's packing)."""
    import numpy as np

    y, u, v = _planes(data, width, height)
    u_full = np.repeat(u, 2, axis=1)
    v_full = np.repeat(v, 2, axis=1)
    yy = _turn(y, rotate)[::step, ::step].astype(np.float32)
    uu = _turn(u_full, rotate)[::step, ::step].astype(np.float32) - 128.0
    vv = _turn(v_full, rotate)[::step, ::step].astype(np.float32) - 128.0
    rgb = np.stack([yy + _R_V * vv, yy - _G_U * uu - _G_V * vv, yy + _B_U * uu], axis=-1)
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


# ── the one-pass sampler (spec §4.3 step 6) ────────────────────────────────────

LUMA_STEP = 4  # quarter resolution: 180×320 upright at 720p
CHROMA_STEP = 8  # chroma at half of that: 90×160


def _source_xy(
    u: Any, v: Any, width: int, height: int, rotate: int
) -> tuple[Any, Any]:  # numpy arrays in, numpy arrays out
    """Upright pixel (u, v) → the delivered frame's (x, y): the inverse of ``uyvy::Turn``."""
    if rotate == 0:
        return u, v
    if rotate == 90:
        return v, height - 1 - u
    if rotate == 180:
        return width - 1 - u, height - 1 - v
    if rotate == 270:
        return width - 1 - v, u
    raise FrameError(f"rotate {rotate} is not a quarter turn")


@functools.lru_cache(maxsize=16)
def _gather(width: int, height: int, rotate: int, step: int, byte: int) -> Any:
    """Flat byte offsets of every ``step``-th upright pixel's Y (``byte`` 1), U (0) or V (2)."""
    import numpy as np

    uw, uh = (height, width) if rotate in (90, 270) else (width, height)
    vv, uu = np.mgrid[0:uh:step, 0:uw:step]
    x, y = _source_xy(uu, vv, width, height, rotate)
    if byte == 1:
        index = (y * width + x) * 2 + 1
    else:
        index = (y * width + (x & ~1)) * 2 + byte
    out = np.ascontiguousarray(index.astype(np.intp))
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class QuarterPlanes:
    """The shared upright planes of one frame: luma every :data:`LUMA_STEP` pixels (``y``,
    rows × cols = 320 × 180 at 720p) and chroma every :data:`CHROMA_STEP` (``u``, ``v``: 160 ×
    90), uint8. Chroma sample (i, j) sits on luma sample (2i, 2j)."""

    y: Any
    u: Any
    v: Any
    luma_step: int = LUMA_STEP
    chroma_step: int = CHROMA_STEP

    def rgb_at_chroma(self) -> Any:
        """RGB (uint8) at the chroma sites (160 × 90 at 720p), BT.601 full range."""
        import numpy as np

        ratio = self.chroma_step // self.luma_step
        yy = self.y[::ratio, ::ratio][: self.u.shape[0], : self.u.shape[1]].astype(np.float32)
        uu = self.u.astype(np.float32) - 128.0
        vv = self.v.astype(np.float32) - 128.0
        rgb = np.stack([yy + _R_V * vv, yy - _G_U * uu - _G_V * vv, yy + _B_U * uu], axis=-1)
        return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def sample_quarter(
    data: bytes, width: int, height: int, rotate: int = 90, *, luma_step: int = LUMA_STEP
) -> QuarterPlanes:
    """The upright luma and chroma planes straight from the UYVY bytes in one gather each (no
    full-frame colour conversion: upstream measured 345 of 407 ms spent there, ``detect.rs``).
    The luma plane equals ``upright_luma(..., step=luma_step)``."""
    import numpy as np

    packed = np.frombuffer(data, dtype=np.uint8)
    if packed.size != width * height * 2 or width % 2:
        raise FrameError(f"{packed.size} bytes is not a {width}x{height} UYVY frame")
    chroma_step = 2 * luma_step
    return QuarterPlanes(
        y=packed[_gather(width, height, rotate, luma_step, 1)],
        u=packed[_gather(width, height, rotate, chroma_step, 0)],
        v=packed[_gather(width, height, rotate, chroma_step, 2)],
        luma_step=luma_step,
        chroma_step=chroma_step,
    )


def roi_luma(
    data: bytes,
    width: int,
    height: int,
    rotate: int,
    box: tuple[int, int, int, int],
    *,
    step: int = 1,
) -> Any:
    """Full-resolution upright luma inside ``box`` = (u0, v0, u1, v1) (upright pixels, u1 and v1
    exclusive, clipped to the picture), every ``step``-th pixel: the cut a tag reader wants."""
    import numpy as np

    uw, uh = (height, width) if rotate in (90, 270) else (width, height)
    u0, v0, u1, v1 = box
    u0, v0 = max(0, int(u0)), max(0, int(v0))
    u1, v1 = min(uw, int(u1)), min(uh, int(v1))
    if u1 <= u0 or v1 <= v0:
        return np.zeros((0, 0), dtype=np.uint8)
    packed = np.frombuffer(data, dtype=np.uint8)
    if packed.size != width * height * 2:
        raise FrameError(f"{packed.size} bytes is not a {width}x{height} UYVY frame")
    vv, uu = np.mgrid[v0:v1:step, u0:u1:step]
    x, y = _source_xy(uu, vv, width, height, rotate)
    return packed[(y * width + x) * 2 + 1]


def frame_sha16(data: bytes) -> str:
    """The first 16 hex digits of the bytes' SHA-256 (audit only; spec §4.3 step 9)."""
    return hashlib.sha256(data).hexdigest()[:16]


# ── what a task sees ─────────────────────────────────────────────────────────────


@dataclass
class LookView:
    """One look as the tasks see it. ``data`` (the frame's bytes) lives only as long as this
    look: never store it, never log it (spec §4.8). ``light`` is this look's light result
    (``{level, luma, noise_q, …}``); ``pose`` is ``None`` when the look is ``moved``/pose-less,
    and a task that needs the geometry then reports nothing."""

    look: EyeLook
    header: MediaFrameHeader
    data: bytes
    camera: EyeCamera
    planes: QuarterPlanes
    captured_at_ns: int
    pose: PairedPose | None = None
    light: dict[str, Any] = field(default_factory=dict)

    @property
    def camera_pose(self) -> CameraPose | None:
        return self.pose.camera_pose(self.header.rotate) if self.pose is not None else None

    def roi_luma(self, box: tuple[int, int, int, int], *, step: int = 1) -> Any:
        h = self.header
        return roi_luma(self.data, h.width, h.height, h.rotate, box, step=step)

    def luma(self, step: int = 2) -> Any:
        """The whole upright luma every ``step``-th pixel (half resolution by default)."""
        h = self.header
        return upright_luma(self.data, h.width, h.height, h.rotate, step=step)


# ── the page's eye window (F4: frames only while it is open) ─────────────────────


def window_light(data: bytes, head: MediaFrameHeader) -> dict[str, Any]:
    """The duck's own light reading of one frame (the board path, :mod:`duckmind.eye.light`),
    in the shape the page reads (``level``, ``level_zh``, ``adapting`` …)."""
    from duckmind.eye import light as eye_light

    planes = sample_quarter(data, head.width, head.height, head.rotate)
    reading = eye_light.read_quarter(planes.y)
    level = eye_light.level_q(reading)
    return {
        "level": level,
        "level_zh": eye_light.LEVEL_ZH[level],
        "method": "frame",
        "luma": reading.luma,
        "noise": round(reading.sigma_milli / 1000.0, 3),
        "noise_q": reading.noise_q,
        "glow": reading.glow_px,
        "level_frame": level,
        "level_exposure": None,
        "lux_estimate": None,
        "exposure": None,
        "adapting": eye_light.settling(reading),
        "agree": True,
        "thresholds": eye_light.THRESHOLDS.source,
    }


NOT_OPEN = msg("eye.frames.NOT_OPEN._")
NO_SOCKET = msg("eye.frames.NO_SOCKET._")


class EyeWindow:
    """The simulator page's duck's-eye window as a ``media.frame`` client.

    The page calls :meth:`touch` whenever it asks for the picture (the window is open and
    visible); while the last touch is younger than ``idle_s`` a frame is fetched every
    ``period_s`` on this object's own thread, and none at all otherwise (a closed window costs
    the world nothing: no render, no copy). :meth:`close_window` stops at once.
    :meth:`latest` holds the upright picture (RGB, every ``step``-th pixel), its header, the
    light reading of the full-resolution luma (:func:`duckmind.eye.light.measure`, frame-based:
    the header carries no exposure), the fetch latency and the last error. ``on_frame`` is
    called with that dict after each frame (the page encodes its JPEG there).
    """

    def __init__(
        self,
        socket_path: Path | str | Callable[[], Path | str | None],
        *,
        period_s: float = 2.0,
        idle_s: float = 6.0,
        step: int = 2,
        timeout_s: float = TIMEOUT_S,
        on_frame: Callable[[dict[str, Any]], None] | None = None,
        meter: bool = True,
    ) -> None:
        self._socket = socket_path
        self.period_s = float(period_s)
        self.idle_s = float(idle_s)
        self.step = max(1, int(step))
        self.timeout_s = float(timeout_s)
        self.on_frame = on_frame
        self.meter = meter
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._touched = -float("inf")
        self._latest: dict[str, Any] | None = None
        self._state: dict[str, Any] = {"open": False, "error": None, "note": NOT_OPEN}
        self.fetched = 0
        self.failed = 0

    # ── what the page does ──────────────────────────────────────────

    def touch(self) -> None:
        """The window is open and someone is looking: keep fetching for ``idle_s``."""
        with self._lock:
            opening = not self._open_locked()
            self._touched = time.monotonic()
        if opening:
            self._wake.set()

    def close_window(self) -> None:
        with self._lock:
            self._touched = -float("inf")
        self._wake.set()

    @property
    def open(self) -> bool:
        with self._lock:
            return self._open_locked()

    def _open_locked(self) -> bool:
        return time.monotonic() - self._touched < self.idle_s

    def latest(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._latest) if self._latest is not None else None

    def status(self) -> dict[str, Any]:
        with self._lock:
            out = dict(self._state)
            out["open"] = self._open_locked()
            out["fetched"] = self.fetched
            out["failed"] = self.failed
            if self._latest is not None:
                out.update({k: v for k, v in self._latest.items() if k not in ("rgb", "luma")})
        return out

    # ── lifecycle ───────────────────────────────────────────────────

    def start(self) -> EyeWindow:
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="eye-window", daemon=True)
            self._thread.start()
        return self

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            self._thread = None

    # ── the thread ──────────────────────────────────────────────────

    def _path(self) -> Path | None:
        where = self._socket() if callable(self._socket) else self._socket
        return Path(where) if where else None

    def _run(self) -> None:
        next_at = 0.0
        while not self._stop.is_set():
            if not self.open:
                with self._lock:
                    self._state.update(note=NOT_OPEN)
                self._wake.clear()
                self._wake.wait(timeout=1.0)
                next_at = 0.0
                continue
            now = time.monotonic()
            if now < next_at:
                self._wake.clear()
                self._wake.wait(timeout=next_at - now)
                continue
            next_at = now + self.period_s
            self.fetch_once()

    def fetch_once(self) -> dict[str, Any] | None:
        """One frame now (the thread calls this; tests may too)."""
        path = self._path()
        if path is None:
            with self._lock:
                self._state.update(error=NO_SOCKET, note=NO_SOCKET)
            return None
        t0 = time.monotonic()
        try:
            head, data = fetch(path, timeout_s=self.timeout_s)
        except (OSError, FrameError) as error:
            with self._lock:
                self.failed += 1
                self._state.update(
                    error=msg("eye.frames.EyeWindow.fetch_once.4f1ad1b6"),
                    error_detail=f"{type(error).__name__}: {error}",
                )
            return None
        latency = (time.monotonic() - t0) * 1000.0
        result: dict[str, Any] = {
            "header": head.to_dict(),
            "latency_ms": round(latency, 2),
            "fetched_at": time.time(),
            "age_ms": round(time.time() * 1000.0 - head.captured_at_unix_us / 1000.0, 2),
        }
        try:
            result["rgb"] = upright_rgb(data, head.width, head.height, head.rotate, step=self.step)
            if self.meter:
                result["light"] = window_light(data, head)
        except ImportError as error:  # no numpy here: the header alone
            result["error"] = f"{type(error).__name__}: {error}"
        with self._lock:
            self.fetched += 1
            self._latest = result
            self._state.update(error=None, error_detail=None, note="")
        if self.on_frame is not None:
            try:
                self.on_frame(result)
            except Exception as error:  # the page's encoder failing must not stop the window
                with self._lock:
                    self._state.update(error_detail=f"on_frame: {type(error).__name__}: {error}")
        return result
