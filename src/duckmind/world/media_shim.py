"""media-shim: mediad's ``media.frame`` endpoint, for the simulated head camera (spec §11.1, §11.2).

The real duck's ``mediad`` serves one raw camera frame per connection on a local unix socket
(``mediad/src/frame.rs``); duck-eye (phase 3) and the simulator page's eye window read it. This
module is the same endpoint with the world's rendered camera behind it, so a client cannot tell
the two apart (P1: ``tests/test_media_frame_protocol.py`` checks the numbers and the behaviour
against the upstream sources, read by path at test time). Pure standard library: nothing here
renders; :mod:`duckmind.world.eye_render` is the camera that answers.

What it copies from ``frame.rs`` and ``pipeline.rs`` (upstream 0.15.0, API 37):

* **The socket.** ``bind``: the parent folder is made; ``<socket>.lock`` is opened (created,
  never truncated) and ``flock``-ed exclusively without waiting — "another mediad owns the frame
  socket" when that fails; the socket is bound, and when its path is taken, only a *socket* whose
  listener refuses a connection (a stale one) is removed and rebound — a regular file or a live
  listener is an error and is left alone. Mode 0660, handed to group ``robot`` (a missing group is
  a warning, as on a laptop). mediad does not remove its socket when it stops; neither does this.
* **The connection.** At most :data:`SLOTS` (16) connections at once: the 17th is accepted and
  closed without a word. Each connection lives at most :data:`CONNECTION_S` (5 s) from accept,
  whatever it is doing (reading, waiting for a frame, writing pixels): then it is closed.
* **The requests.** JSON-RPC 2.0, one per line, read with a bound *before* buffering: a line
  longer than :data:`MAX_REQUEST_BYTES` (4096, the newline included) is answered ``-32602
  "request is too large"`` with a null id and the connection closes. A line that is not a
  ``Request`` (serde's rules: ``jsonrpc`` and ``method`` strings, an optional ``id`` that is an
  unsigned 64-bit integer or a string, anything as ``params``, unknown members ignored, a
  duplicated known member refused, no NaN, nesting below 128) is answered ``-32700`` with a null
  id and the connection closes. ``hello`` is answered ``{api_version: 37, daemon_version:
  "0.15.0", revision: null}`` (a build that did not come from CI) and the connection stays open;
  any other method but ``media.frame`` gets ``-32601 "<method> is not served by mediad"`` and the
  connection stays open.
* **The frame.** ``media.frame`` asks for the *next* capture (:class:`Frames`, the rendezvous of
  ``pipeline.rs``: the generation is read before the demand is raised, one capture answers every
  reader waiting at that moment, a reader waits at most :data:`FRAME_TIMEOUT_S`, 500 ms). No frame
  in time: ``-32603 "no frame arrived within the capture timeout"`` and the connection closes.
  Otherwise one header line ``{"width", "height", "format": "UYVY", "bytes",
  "captured_at_unix_us", "rotate": 90}`` — nothing else in it, never the exposure or the physics
  time — then exactly ``bytes`` raw bytes, then the connection closes: one frame per connection.

Responses are serde's compact JSON: ``{"jsonrpc":"2.0","id":…,"result":…}`` or ``…"error":
{"code":…,"message":…}}``, UTF-8, one line. Parse-error *messages* are Python's wording, not
serde's (only the codes are protocol).

The simulator's capture branch is :mod:`duckmind.world.eye_render`: it waits for
:attr:`Frames.demand`, takes the request on the camera's own clock (``take_request``), renders
and :meth:`Frames.deliver` s. ``captured_at_unix_us`` is the host's wall clock at the world
snapshot the frame was rendered from, as mediad stamps a buffer with ``SystemTime::now()``.
"""

from __future__ import annotations

import errno
import fcntl
import grp
import json
import os
import socket
import stat
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

# mediad/src/frame.rs
SOCKET_MODE = 0o660
GROUP = "robot"
MAX_REQUEST_BYTES = 4096
SLOTS = 16  # Semaphore::new(16)
CONNECTION_S = 5.0  # Duration::from_secs(5)
# mediad/src/pipeline.rs
FRAME_TIMEOUT_S = 0.5  # FRAME_TIMEOUT
# mediad/src/main.rs: --rotate defaults to 90
ROTATE = 90
CAPTURE_FORMAT = "UYVY"
# duck-ipc-proto/src/lib.rs
API_VERSION = 37
DAEMON_VERSION = "0.15.0"  # the workspace's CARGO_PKG_VERSION
REVISION: str | None = None  # build_info!().revision: None for a build that did not come from CI
JSONRPC = "2.0"
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
HELLO = "hello"
MEDIA_FRAME = "media.frame"
MAX_FRAME_BYTES = 16 * 1024 * 1024  # MediaFrameHeader::valid_uyvy
TOO_LARGE = "request is too large"
NO_FRAME = "no frame arrived within the capture timeout"
# serde_json refuses a document nested 128 deep (its recursion limit)
SERDE_DEPTH_LIMIT = 128
U64_MAX = 2**64 - 1
SUN_PATH_MAX = 104  # macOS sockaddr_un.sun_path (Linux: 108)
REQUEST_FIELDS = ("jsonrpc", "id", "method", "params")

Log = Callable[[str], None]


def _stderr(line: str) -> None:
    sys.stderr.write(line + "\n")
    sys.stderr.flush()


class ShimError(RuntimeError):
    """The socket could not be claimed (another shim holds it, a file is in the way)."""


# ── the capture rendezvous (pipeline.rs `Frames`) ───────────────────────────────


@dataclass(frozen=True)
class Frame:
    """One captured frame, as the capture branch hands it over."""

    width: int
    height: int
    data: bytes
    captured_at_unix_us: int
    format: str = CAPTURE_FORMAT


class Frames:
    """A frame handed over on request (``pipeline.rs`` ``Frames``): asked for, not published.

    A reader raises the demand and waits for the *next* capture; the capture branch answers the
    demand once and one capture answers every reader waiting at that moment. The generation is
    read before the demand is raised, so a capture landing between the two counts as the one the
    reader asked for (as upstream). A reader that gets nothing in time gets ``None``, never the
    frame a stopped camera stopped on.
    """

    def __init__(self, timeout_s: float = FRAME_TIMEOUT_S) -> None:
        self.timeout_s = float(timeout_s)
        self._cond = threading.Condition()
        self._frame: Frame | None = None
        self._generation = 0
        self._wanted = False
        # Set whenever a reader is waiting: what the capture branch sleeps on (the real branch
        # looks at the flag once per buffer, thirty times a second).
        self.demand = threading.Event()

    @property
    def generation(self) -> int:
        with self._cond:
            return self._generation

    def next_frame(self, timeout_s: float | None = None) -> Frame | None:
        """Ask for the next frame and wait for it (at most :attr:`timeout_s`, or less)."""
        wait = self.timeout_s if timeout_s is None else min(float(timeout_s), self.timeout_s)
        with self._cond:
            seen = self._generation
            self._wanted = True
            self.demand.set()
            self._cond.wait_for(lambda: self._generation != seen, timeout=max(0.0, wait))
            if self._generation == seen:
                return None
            return self._frame

    def take_request(self) -> bool:
        """Whether a reader is waiting, taking the request if one is."""
        with self._cond:
            wanted, self._wanted = self._wanted, False
            self.demand.clear()
            return wanted

    def deliver(self, frame: Frame) -> None:
        """Hand a captured frame to every reader waiting for one."""
        with self._cond:
            self._frame = frame
            self._generation += 1
            self._cond.notify_all()


# ── requests and responses (duck-ipc-proto `Request` / `Response`) ──────────────


@dataclass(frozen=True)
class Request:
    jsonrpc: str
    id: int | str | None
    method: str
    params: Any = None


class RequestError(ValueError):
    """A line that is not a ``Request`` (serde would refuse it): answered -32700."""


class _Pairs(list[tuple[str, Any]]):
    """An object's members in order, before duplicates are resolved."""


def _pairs(items: list[tuple[str, Any]]) -> _Pairs:
    return _Pairs(items)


def _constant(name: str) -> Any:
    raise RequestError(f"{name} is not JSON")


def _integer(text: str) -> Any:
    # serde_json reads "-0" as the float -0.0 (a negative zero is not an integer to it).
    return -0.0 if text.lstrip() == "-0" else int(text)


def _finish(value: Any, depth: int) -> tuple[Any, int]:
    """Objects to dicts (a repeated member: the last one, as serde_json's ``Value``), the
    document's nesting depth, and serde's refusal of unpaired surrogates."""
    if isinstance(value, _Pairs):
        out: dict[str, Any] = {}
        deepest = depth
        for key, item in value:
            _check_text(key)
            out[key], got = _finish(item, depth + 1)
            deepest = max(deepest, got)
        return out, deepest
    if isinstance(value, list):
        items = []
        deepest = depth
        for item in value:
            done, got = _finish(item, depth + 1)
            items.append(done)
            deepest = max(deepest, got)
        return items, deepest
    if isinstance(value, str):
        _check_text(value)
    return value, depth - 1


def _check_text(text: str) -> None:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        raise RequestError("lone surrogate in a string") from None


def _request_id(value: Any) -> int | str | None:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= U64_MAX:
        return value
    raise RequestError("data did not match any variant of untagged enum Id")


def parse_request(line: bytes) -> Request:
    """One request line as ``serde_json::from_slice::<Request>`` reads it, or
    :class:`RequestError`."""
    try:
        text = line.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RequestError(f"invalid UTF-8 ({error.reason})") from None
    try:
        raw = json.loads(
            text,
            object_pairs_hook=_pairs,
            parse_constant=_constant,
            parse_int=_integer,
        )
    except RecursionError:
        raise RequestError("recursion limit exceeded") from None
    except json.JSONDecodeError as error:
        raise RequestError(f"{error.msg} at line {error.lineno} column {error.colno}") from None
    top = raw
    if isinstance(raw, _Pairs):
        seen: set[str] = set()
        for key, _ in raw:
            if key in REQUEST_FIELDS:
                if key in seen:
                    raise RequestError(f"duplicate field `{key}`")
                seen.add(key)
    value, depth = _finish(top, 1)
    if depth >= SERDE_DEPTH_LIMIT:
        raise RequestError("recursion limit exceeded")
    if isinstance(value, dict):
        fields = value
    elif isinstance(value, list):
        # serde derives a struct from a sequence too: [jsonrpc, id?, method, params?] by place
        if not 3 <= len(value) <= 4:
            raise RequestError(f"invalid length {len(value)}, expected struct Request")
        fields = dict(zip(REQUEST_FIELDS, value, strict=False))
    else:
        raise RequestError("invalid type, expected struct Request")
    for name in ("jsonrpc", "method"):
        if name not in fields:
            raise RequestError(f"missing field `{name}`")
        if not isinstance(fields[name], str):
            raise RequestError(f"invalid type for `{name}`, expected a string")
    return Request(
        jsonrpc=fields["jsonrpc"],
        id=_request_id(fields.get("id")),
        method=fields["method"],
        params=fields.get("params"),
    )


def encode(message: dict[str, Any]) -> bytes:
    """serde_json's compact form, UTF-8, plus the newline."""
    return json.dumps(message, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n"


def response_ok(request_id: int | str | None, result: Any) -> bytes:
    return encode({"jsonrpc": JSONRPC, "id": request_id, "result": result})


def response_error(request_id: int | str | None, code: int, message: str) -> bytes:
    return encode(
        {"jsonrpc": JSONRPC, "id": request_id, "error": {"code": code, "message": message}}
    )


def hello_result() -> dict[str, Any]:
    return {"api_version": API_VERSION, "daemon_version": DAEMON_VERSION, "revision": REVISION}


def header(frame: Frame, rotate: int = ROTATE) -> dict[str, Any]:
    """``MediaFrameHeader``, in its field order and nothing more."""
    return {
        "width": int(frame.width),
        "height": int(frame.height),
        "format": frame.format,
        "bytes": len(frame.data),
        "captured_at_unix_us": int(frame.captured_at_unix_us),
        "rotate": int(rotate),
    }


def valid_uyvy(head: dict[str, Any]) -> bool:
    """``MediaFrameHeader::valid_uyvy``: geometry and size a reader may allocate for."""
    try:
        width, height, size = int(head["width"]), int(head["height"]), int(head["bytes"])
        rotate, fmt = int(head["rotate"]), head["format"]
    except (KeyError, TypeError, ValueError):
        return False
    return (
        width > 0
        and height > 0
        and width % 2 == 0
        and fmt == CAPTURE_FORMAT
        and rotate in (0, 90, 180, 270)
        and size <= MAX_FRAME_BYTES
        and width * height * 2 == size
    )


# ── the socket (frame.rs `bind`) ────────────────────────────────────────────────


def give_to_group(path: Path, group: str = GROUP) -> None:
    """Hand the socket to ``group`` (OSError when there is no such group or chown fails)."""
    try:
        gid = grp.getgrnam(group).gr_gid
    except KeyError:
        raise OSError(f"no {group} group on this system") from None
    os.chown(path, -1, gid)


def bind(
    path: Path | str, *, group: str = GROUP, backlog: int = 128, log: Log = _stderr
) -> tuple[IO[bytes], socket.socket]:
    """Claim the socket without replacing a live listener or a non-socket file: the lock file
    (kept open for the listener's lifetime) and the listening socket."""
    path = Path(path)
    if len(os.fsencode(str(path))) >= SUN_PATH_MAX:
        raise ShimError(f"{path}: a unix socket path must be shorter than {SUN_PATH_MAX} bytes")
    if str(path.parent) not in ("", "."):
        path.parent.mkdir(parents=True, exist_ok=True)
    lock = open(str(path) + ".lock", "a+b")  # created, never truncated, read and write
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.close()
        raise ShimError("another mediad owns the frame socket") from None
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        try:
            listener.bind(str(path))
        except OSError as error:
            if error.errno != errno.EADDRINUSE:
                raise
            if not stat.S_ISSOCK(os.lstat(path).st_mode):
                raise
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            probe.settimeout(1.0)
            try:
                probe.connect(str(path))
            except ConnectionRefusedError:
                os.remove(path)
                listener.bind(str(path))
            except OSError:
                raise error from None
            else:
                raise error  # a live listener: not ours to replace
            finally:
                probe.close()
        os.chmod(path, SOCKET_MODE)
        try:
            give_to_group(path, group)
        except OSError as error:
            log(
                f"media-shim: WARN {error} ({path}): media.frame stays private to mediad — "
                "nothing else can ask for a picture"
            )
        listener.listen(backlog)
    except BaseException:
        listener.close()
        lock.close()
        raise
    log(f"media-shim: serving media.frame locally at {path} (mode {SOCKET_MODE:o})")
    return lock, listener


# ── the server (frame.rs `serve` and `handle`) ──────────────────────────────────


class _Expired(Exception):
    """The connection's five seconds are up."""


class MediaShim:
    """The ``media.frame`` endpoint on ``path``, answered from ``frames``.

    ``start()`` claims the socket and serves on a thread of its own (one thread per connection,
    at most :data:`SLOTS`); ``close()`` stops accepting. ``stats()`` counts connections, frames
    served, timeouts and refusals (for evaluators; nothing of it reaches a client).
    """

    def __init__(
        self,
        path: Path | str,
        frames: Frames,
        *,
        rotate: int = ROTATE,
        slots: int = SLOTS,
        lifetime_s: float = CONNECTION_S,
        group: str = GROUP,
        log: Log = _stderr,
    ) -> None:
        self.path = Path(path)
        self.frames = frames
        self.rotate = int(rotate)
        self.slots = int(slots)
        self.lifetime_s = float(lifetime_s)
        self.group = group
        self.log = log
        self._lock_file: IO[bytes] | None = None
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._count = threading.Lock()
        self._active = 0
        self._stats = {
            "accepted": 0,
            "dropped": 0,  # over the slot limit: accepted and closed at once
            "frames": 0,
            "timeouts": 0,  # -32603 sent
            "expired": 0,  # closed at the five-second limit
            "errors": 0,  # parse errors and oversize lines
            "hello": 0,
            "unknown": 0,
        }
        # (request received, header sent, last byte sent) per served frame, monotonic seconds
        self.served: list[tuple[float, float, float]] = []

    # ── lifecycle ─────────────────────────────────────────────────────

    def start(self) -> MediaShim:
        self._lock_file, self._listener = bind(self.path, group=self.group, log=self.log)
        self._listener.settimeout(0.25)
        self._stop.clear()
        self._thread = threading.Thread(target=self._accept, name="media-shim", daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._listener is not None:
            self._listener.close()
            self._listener = None
        if self._lock_file is not None:
            self._lock_file.close()  # the flock goes with it
            self._lock_file = None

    def __enter__(self) -> MediaShim:
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @property
    def active(self) -> int:
        with self._count:
            return self._active

    def stats(self) -> dict[str, Any]:
        with self._count:
            out: dict[str, Any] = dict(self._stats)
            out["active"] = self._active
        return out

    def _bump(self, key: str) -> None:
        with self._count:
            self._stats[key] += 1

    # ── accepting ────────────────────────────────────────────────────

    def _accept(self) -> None:
        listener = self._listener
        assert listener is not None
        while not self._stop.is_set():
            try:
                conn, _ = listener.accept()
            except TimeoutError:
                continue
            except OSError as error:
                if self._stop.is_set():
                    return
                self.log(f"media-shim: WARN media.frame accept failed: {error}")
                time.sleep(0.05)
                continue
            with self._count:
                full = self._active >= self.slots
                if full:
                    self._stats["dropped"] += 1
                else:
                    self._active += 1
                    self._stats["accepted"] += 1
            if full:
                conn.close()  # the 17th: accepted and dropped without a reply
                continue
            threading.Thread(
                target=self._serve, args=(conn,), name="media-shim-conn", daemon=True
            ).start()

    def _serve(self, conn: socket.socket) -> None:
        deadline = time.monotonic() + self.lifetime_s
        try:
            self._handle(conn, deadline)
        except _Expired:
            self._bump("expired")
        except OSError:
            pass  # the client went away
        finally:
            try:
                conn.close()
            finally:
                with self._count:
                    self._active -= 1

    # ── one connection ──────────────────────────────────────────────

    @staticmethod
    def _remaining(deadline: float) -> float:
        left = deadline - time.monotonic()
        if left <= 0:
            raise _Expired
        return left

    def _send(self, conn: socket.socket, data: bytes, deadline: float) -> None:
        conn.settimeout(self._remaining(deadline))
        try:
            conn.sendall(data)
        except TimeoutError:
            raise _Expired from None

    def _read_line(self, conn: socket.socket, buffer: bytearray, deadline: float) -> bytes:
        """``take(MAX + 1).read_until(b'\\n')``: a line (newline included), at most
        MAX + 1 bytes of one, what is left at end of file, or b"" at end of file."""
        limit = MAX_REQUEST_BYTES + 1
        while True:
            end = buffer.find(b"\n", 0, limit)
            if end >= 0:
                line = bytes(buffer[: end + 1])
                del buffer[: end + 1]
                return line
            if len(buffer) >= limit:
                line = bytes(buffer[:limit])
                del buffer[:limit]
                return line
            conn.settimeout(self._remaining(deadline))
            try:
                chunk = conn.recv(65536)
            except TimeoutError:
                raise _Expired from None
            if not chunk:
                line = bytes(buffer)
                buffer.clear()
                return line
            buffer += chunk

    def _handle(self, conn: socket.socket, deadline: float) -> None:
        buffer = bytearray()
        while True:
            line = self._read_line(conn, buffer, deadline)
            if not line:
                return
            if len(line) > MAX_REQUEST_BYTES:
                self._bump("errors")
                self._send(conn, response_error(None, INVALID_PARAMS, TOO_LARGE), deadline)
                return
            try:
                request = parse_request(line)
            except RequestError as error:
                self._bump("errors")
                self._send(conn, response_error(None, PARSE_ERROR, str(error)), deadline)
                return
            if request.method == HELLO:
                self._bump("hello")
                self._send(conn, response_ok(request.id, hello_result()), deadline)
                continue
            if request.method != MEDIA_FRAME:
                self._bump("unknown")
                message = f"{request.method} is not served by mediad"
                self._send(conn, response_error(request.id, METHOD_NOT_FOUND, message), deadline)
                continue
            asked = time.monotonic()
            wait = min(self.frames.timeout_s, self._remaining(deadline))
            frame = self.frames.next_frame(wait)
            if frame is None:
                self._remaining(deadline)  # the five seconds ran out first: closed, no reply
                self._bump("timeouts")
                self._send(conn, response_error(request.id, INTERNAL_ERROR, NO_FRAME), deadline)
                return
            self._send(conn, response_ok(request.id, header(frame, self.rotate)), deadline)
            sent_header = time.monotonic()
            self._send(conn, frame.data, deadline)
            done = time.monotonic()
            with self._count:
                self._stats["frames"] += 1
                self.served.append((asked, sent_header, done))
                if len(self.served) > 4096:
                    del self.served[:2048]
            return
