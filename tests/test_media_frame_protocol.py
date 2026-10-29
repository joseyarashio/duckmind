"""P1 for the camera's transport: the world's media-shim speaks mediad's ``media.frame``.

Two halves (spec §11.7 P1: "fields, types, timeouts and limits all the same"):

* **the numbers and shapes**, read at test time from the official sources by path
  (``mediad/src/frame.rs``, ``pipeline.rs``, ``main.rs``, ``duck-ipc-proto/src/lib.rs``, the
  workspace ``Cargo.toml``) and compared with :mod:`duckmind.world.media_shim` and the client
  :mod:`duckmind.eye.frames` — plus digests of the code the shim copies, so an upstream change
  fails here with the file named until someone reads it and re-pins;
* **the behaviour**: mediad's own unit tests (every ``#[tokio::test]`` in ``frame.rs``)
  re-expressed against the shim over a real unix socket, and the rendezvous, the slot limit, the
  client and the page's eye window on top.

Pure Python; the upstream half skips without the official checkout
(``DUCKMIND_MICRODUCK_MJCF``). Sockets live under a short ``/tmp`` folder: macOS limits a unix
socket path to 104 bytes, which pytest's own temporary folders exceed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from duckmind.eye import frames as eye_frames
from duckmind.world import media_shim as ms


def _upstream() -> Path | None:
    raw = os.environ.get("DUCKMIND_UPSTREAM")
    if raw:
        return Path(raw)
    mjcf = os.environ.get("DUCKMIND_MICRODUCK_MJCF")
    if mjcf:
        parents = Path(mjcf).expanduser().resolve().parents
        if len(parents) > 4 and (parents[4] / "microduck").is_dir():
            return parents[4]
    return None


UP = _upstream()
MICRODUCK = UP / "microduck" if UP else None
needs_upstream = pytest.mark.skipif(
    MICRODUCK is None or not (MICRODUCK / "mediad" / "src" / "frame.rs").is_file(),
    reason="needs the official checkout (DUCKMIND_MICRODUCK_MJCF)",
)

# The upstream code the shim copies, as reviewed on 2026-09-27 (microduck 0.15.0, a9ec4b2).
# A change fails the pin test with the file named: read it, adjust media_shim, then re-pin.
PINNED = {
    # whole file
    "mediad/src/frame.rs": "b2440af0a98ba8f95c457410b99c935cb672ab0941def49bbcefcdc6043d9647",
    # `pub struct Frames` … `pub struct Channel`
    "mediad/src/pipeline.rs#Frames": (
        "e8a0567f478268b838e7fe6db9d02511218fa5393c8b8c38466419eadd96215f"
    ),
    # the header and valid_uyvy
    "duck-ipc-proto/src/lib.rs#MediaFrameHeader": (
        "a5187a760df56f6d3e007579a66de8e18b5cc4f482515345f3f2738f87831b7f"
    ),
}


def _read(relative: str) -> str:
    assert MICRODUCK is not None
    return (MICRODUCK / relative).read_text()


def _pinned_text(key: str) -> str:
    path, _, part = key.partition("#")
    text = _read(path)
    if part == "Frames":
        start, end = text.index("pub struct Frames("), text.index("pub struct Channel")
        return text[start:end]
    if part == "MediaFrameHeader":
        start = text.index("pub struct MediaFrameHeader")
        end = text.index("pub enum Phase")
        return text[start:end]
    return text


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── the numbers, read from upstream ─────────────────────────────────────────────


@needs_upstream
def test_the_endpoint_constants_are_mediad_s() -> None:
    frame = _read("mediad/src/frame.rs")
    assert re.search(r"const SOCKET_MODE: u32 = 0o(\d+);", frame).group(1) == "660"  # type: ignore[union-attr]
    assert ms.SOCKET_MODE == 0o660
    assert re.search(r'const GROUP: &str = "(\w+)";', frame).group(1) == ms.GROUP  # type: ignore[union-attr]
    found = re.search(r"const MAX_REQUEST_BYTES: usize = (\d+);", frame)
    assert found and int(found.group(1)) == ms.MAX_REQUEST_BYTES
    found = re.search(r"Semaphore::new\((\d+)\)", frame)
    assert found and int(found.group(1)) == ms.SLOTS
    found = re.search(r"tokio::time::timeout\(\s*Duration::from_secs\((\d+)\),\s*handle", frame)
    assert found and float(found.group(1)) == ms.CONNECTION_S
    assert f'"{ms.TOO_LARGE}"' in frame and f'"{ms.NO_FRAME}"' in frame
    assert '"{} is not served by mediad"' in frame
    assert "another mediad owns the frame socket" in frame and 'lock_path.push(".lock")' in frame
    pipeline = _read("mediad/src/pipeline.rs")
    found = re.search(
        r"pub const FRAME_TIMEOUT: std::time::Duration = .*from_millis\((\d+)\)", pipeline
    )
    assert found and int(found.group(1)) / 1000 == ms.FRAME_TIMEOUT_S
    main = _read("mediad/src/main.rs")
    found = re.search(r"args\.rotate\.unwrap_or\((\d+)\)", main)
    assert found and int(found.group(1)) == ms.ROTATE


@needs_upstream
def test_the_protocol_constants_are_duck_ipc_proto_s() -> None:
    proto = _read("duck-ipc-proto/src/lib.rs")
    found = re.search(r"pub const API_VERSION: u32 = (\d+);", proto)
    assert found and int(found.group(1)) == ms.API_VERSION == eye_frames.API_VERSION
    codes = dict(re.findall(r"pub const ([A-Z_]+): i32 = (-?\d+);", proto))
    assert int(codes["PARSE_ERROR"]) == ms.PARSE_ERROR
    assert int(codes["METHOD_NOT_FOUND"]) == ms.METHOD_NOT_FOUND
    assert int(codes["INVALID_PARAMS"]) == ms.INVALID_PARAMS
    assert int(codes["INTERNAL_ERROR"]) == ms.INTERNAL_ERROR
    assert f'pub const HELLO: &str = "{ms.HELLO}";' in proto
    assert f'pub const MEDIA_FRAME: &str = "{ms.MEDIA_FRAME}";' in proto
    assert f'pub const JSONRPC_VERSION: &str = "{ms.JSONRPC}";' in proto
    found = re.search(r'pub const MEDIA: &str = "([^"]+)";', proto)
    assert found and found.group(1) == eye_frames.MEDIA_SOCKET
    cargo = _read("Cargo.toml")
    found = re.search(r'\[workspace\.package\]\s*version = "([^"]+)"', cargo)
    assert found and found.group(1) == ms.DAEMON_VERSION


@needs_upstream
def test_the_header_and_the_envelopes_have_upstream_s_fields_types_and_order() -> None:
    proto = _read("duck-ipc-proto/src/lib.rs")

    def fields(struct: str) -> list[tuple[str, str]]:
        body = proto[proto.index(f"pub struct {struct} {{") :]
        body = body[: body.index("\n}")]
        return re.findall(r"^\s*pub (\w+): ([^,]+),", body, flags=re.M)

    header = fields("MediaFrameHeader")
    assert header == [
        ("width", "u32"),
        ("height", "u32"),
        ("format", "String"),
        ("bytes", "usize"),
        ("captured_at_unix_us", "u128"),
        ("rotate", "u32"),
    ]
    frame = ms.Frame(2, 1, bytes(4), 7)
    assert list(ms.header(frame)) == [name for name, _ in header]
    hello = fields("HelloResult")
    assert [name for name, _ in hello] == list(ms.hello_result())
    response = fields("Response")
    assert [name for name, _ in response] == ["jsonrpc", "id", "result", "error"]
    request = fields("Request")
    assert [name for name, _ in request] == list(ms.REQUEST_FIELDS)
    error = fields("Error")
    assert [name for name, _ in error][:2] == ["code", "message"]
    valid = proto[proto.index("pub fn valid_uyvy") :]
    valid = valid[: valid.index("\n    }")]
    for rule in (
        "self.width > 0",
        "self.height > 0",
        "self.width.is_multiple_of(2)",
        'self.format == "UYVY"',
        "matches!(self.rotate, 0 | 90 | 180 | 270)",
        "self.bytes <= 16 * 1024 * 1024",
    ):
        assert rule in valid, rule
    assert ms.MAX_FRAME_BYTES == 16 * 1024 * 1024


@needs_upstream
def test_the_upstream_code_the_shim_copies_is_as_reviewed() -> None:
    changed = [key for key, want in PINNED.items() if _digest(_pinned_text(key)) != want]
    assert not changed, (
        f"upstream changed since media_shim was checked against it: {changed}; read the change, "
        "adjust src/duckmind/world/media_shim.py, then update PINNED here"
    )


@needs_upstream
def test_every_mediad_endpoint_test_is_replayed_here() -> None:
    frame = _read("mediad/src/frame.rs")
    upstream = set(re.findall(r"#\[tokio::test\]\s*async fn (\w+)\(", frame))
    here = {
        name.removeprefix("test_mediad_") for name in globals() if name.startswith("test_mediad_")
    }
    assert upstream and upstream <= here, f"not replayed: {sorted(upstream - here)}"


# ── fixtures: a short folder, a shim, a stand-in camera ─────────────────────────


@pytest.fixture
def folder() -> Iterator[Path]:
    path = Path(tempfile.mkdtemp(prefix="dmshim-", dir="/tmp"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class Camera:
    """Stands in for the capture branch: answers each demand with a numbered frame."""

    def __init__(self, frames: ms.Frames, *, delay_s: float = 0.0, width: int = 2) -> None:
        self.frames = frames
        self.delay_s = delay_s
        self.width = width
        self.taken = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self.frames.demand.wait(0.05):
                continue
            if self._stop.is_set():
                return
            time.sleep(self.delay_s)
            if self.frames.take_request():
                self.taken += 1
                data = bytes([128, self.taken % 256] * self.width)
                self.frames.deliver(ms.Frame(self.width, 1, data, 1_000_000 * self.taken))

    def __enter__(self) -> Camera:
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        self.frames.demand.set()
        self._thread.join(timeout=2)


def _answer_once(frames: ms.Frames, frame: ms.Frame) -> threading.Thread:
    """mediad's answer_once: wait for the demand, then deliver once."""

    def run() -> None:
        deadline = time.monotonic() + 5.0
        while not frames.take_request():
            assert time.monotonic() < deadline, "the endpoint never asked for a frame"
            time.sleep(0.001)
        frames.deliver(frame)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


@pytest.fixture
def shim(folder: Path) -> Iterator[ms.MediaShim]:
    server = ms.MediaShim(folder / "media.sock", ms.Frames(), log=lambda line: None)
    server.start()
    try:
        yield server
    finally:
        server.close()


def _connect(path: Path) -> socket.socket:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(10.0)
    sock.connect(str(path))
    return sock


def _line(sock: socket.socket) -> dict[str, Any]:
    data = bytearray()
    while not data.endswith(b"\n"):
        chunk = sock.recv(1)
        if not chunk:
            break
        data += chunk
    return json.loads(data)


def _rest(sock: socket.socket) -> bytes:
    out = bytearray()
    while chunk := sock.recv(65536):
        out += chunk
    return bytes(out)


def _reply(shim: ms.MediaShim, request: bytes) -> dict[str, Any]:
    """mediad's `reply`: write the request, shut down writing, read one line."""
    with _connect(shim.path) as sock:
        sock.sendall(request)
        sock.shutdown(socket.SHUT_WR)
        return _line(sock)


# ── mediad's own tests (frame.rs), against the shim ─────────────────────────────


def test_mediad_hello_and_unknown_method_keep_the_connection_for_a_frame(shim: ms.MediaShim):
    producer = _answer_once(shim.frames, ms.Frame(2, 1, bytes([128, 16, 128, 235]), 0))
    with _connect(shim.path) as sock:
        sock.sendall(
            b'{"jsonrpc":"2.0","id":1,"method":"hello"}\n'
            b'{"jsonrpc":"2.0","id":2,"method":"unknown"}\n'
            b'{"jsonrpc":"2.0","id":3,"method":"media.frame"}\n'
        )
        for request_id in (1, 2, 3):
            response = _line(sock)
            assert response["id"] == request_id
            if request_id == 1:
                assert response["result"]["api_version"] == ms.API_VERSION
            if request_id == 2:
                assert response["error"]["code"] == ms.METHOD_NOT_FOUND
        assert _rest(sock) == bytes([128, 16, 128, 235])
    producer.join(timeout=2)


def test_mediad_invalid_utf8_gets_a_parse_error(shim: ms.MediaShim):
    response = _reply(shim, bytes([255]) + b"\n")
    assert response["error"]["code"] == ms.PARSE_ERROR and response["id"] is None


def test_mediad_socket_claim_preserves_files_and_live_listeners_but_recovers_stale_sockets(
    folder: Path,
):
    quiet = lambda line: None  # noqa: E731
    regular = folder / "regular"
    regular.write_bytes(b"keep")
    with pytest.raises(OSError):
        ms.bind(regular, log=quiet)
    assert regular.read_bytes() == b"keep"
    live = folder / "live"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(live))
    listener.listen(16)
    with pytest.raises(OSError):
        ms.bind(live, log=quiet)
    _connect(live).close()
    listener.close()  # the file stays: a stale socket
    lock, claimed = ms.bind(live, log=quiet)
    with pytest.raises(ms.ShimError):
        ms.bind(live, log=quiet)  # the lock is held
    claimed.close()
    lock.close()
    lock, claimed = ms.bind(live, log=quiet)
    claimed.close()
    lock.close()
    assert oct(live.stat().st_mode & 0o777) == oct(ms.SOCKET_MODE)


def test_mediad_idle_clients_are_bounded_and_expire(shim: ms.MediaShim):
    clients = []
    for _ in range(ms.SLOTS):
        sock = _connect(shim.path)
        sock.sendall(b'{"jsonrpc":"2.0","id":1,"method":"hello"}\n')
        assert _line(sock)["result"]
        clients.append(sock)
    excess = _connect(shim.path)
    excess.settimeout(1.0)
    assert excess.recv(1) == b""  # accepted and dropped without a word
    clients[0].settimeout(6.0)
    started = time.monotonic()
    assert clients[0].recv(1) == b""  # the five seconds ran out
    assert time.monotonic() - started < 6.0
    for sock in [excess, *clients]:
        sock.close()
    assert shim.stats()["dropped"] >= 1 and shim.stats()["expired"] >= 1


def test_mediad_a_capture_that_never_comes_is_an_explicit_error(shim: ms.MediaShim):
    started = time.monotonic()
    response = _reply(shim, b'{"jsonrpc":"2.0","id":7,"method":"media.frame","params":{}}\n')
    assert response["id"] == 7 and response["error"]["code"] == ms.INTERNAL_ERROR
    assert response["error"]["message"] == ms.NO_FRAME
    assert time.monotonic() - started >= ms.FRAME_TIMEOUT_S - 0.01


def test_mediad_an_unknown_method_is_refused_without_asking_for_a_frame(shim: ms.MediaShim):
    response = _reply(shim, b'{"jsonrpc":"2.0","id":"request","method":"media.other"}\n')
    assert response["id"] == "request"
    assert response["error"]["code"] == ms.METHOD_NOT_FOUND
    assert not shim.frames.take_request(), "a refused method must not leave demand behind"


def test_mediad_an_oversized_request_is_rejected_before_it_is_parsed(shim: ms.MediaShim):
    response = _reply(shim, b"x" * (ms.MAX_REQUEST_BYTES + 1) + b"\n")
    assert response["id"] is None and response["error"]["code"] == ms.INVALID_PARAMS


def test_mediad_a_request_without_a_newline_is_still_bounded(shim: ms.MediaShim):
    with _connect(shim.path) as sock:
        try:
            sock.sendall(b"y" * (ms.MAX_REQUEST_BYTES * 4))
        except OSError:
            # the bound working: the server answered and hung up mid-write (EPIPE; on macOS
            # also ENOTCONN, about 1 write in 150 — the reply is still there to read)
            pass
        try:
            sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        response = _line(sock)
    assert response["error"]["code"] == ms.INVALID_PARAMS


def test_mediad_a_frame_reply_names_and_follows_with_exactly_its_pixels(shim: ms.MediaShim):
    producer = _answer_once(shim.frames, ms.Frame(2, 1, bytes([128, 32, 128, 64]), 1_000_000))
    with _connect(shim.path) as sock:
        sock.sendall(b'{"jsonrpc":"2.0","id":1,"method":"media.frame"}\n')
        result = _line(sock)["result"]
        assert result == {
            "width": 2,
            "height": 1,
            "format": "UYVY",
            "bytes": 4,
            "captured_at_unix_us": 1_000_000,
            "rotate": 90,
        }
        assert _rest(sock) == bytes([128, 32, 128, 64])
    producer.join(timeout=2)


# ── more of the same contract ──────────────────────────────────────────────────


def test_the_wire_is_serde_s_compact_json(shim: ms.MediaShim):
    producer = _answer_once(shim.frames, ms.Frame(2, 1, bytes(4), 5))
    with _connect(shim.path) as sock:
        sock.sendall(b'{"jsonrpc":"2.0","id":9,"method":"hello"}\n')
        hello = bytearray()
        while not hello.endswith(b"\n"):
            hello += sock.recv(1)
        assert bytes(hello) == (
            b'{"jsonrpc":"2.0","id":9,"result":{"api_version":37,'
            b'"daemon_version":"0.15.0","revision":null}}\n'
        )
        sock.sendall(b'{"jsonrpc":"2.0","id":10,"method":"media.frame"}\n')
        head = bytearray()
        while not head.endswith(b"\n"):
            head += sock.recv(1)
        assert bytes(head) == (
            b'{"jsonrpc":"2.0","id":10,"result":{"width":2,"height":1,"format":"UYVY",'
            b'"bytes":4,"captured_at_unix_us":5,"rotate":90}}\n'
        )
    producer.join(timeout=2)


@pytest.mark.parametrize(
    "line",
    [
        b"\n",
        b"{}\n",
        b'{"jsonrpc":"2.0"}\n',
        b'{"jsonrpc":2,"method":"hello"}\n',
        b'{"jsonrpc":"2.0","id":-1,"method":"hello"}\n',
        b'{"jsonrpc":"2.0","id":1.5,"method":"hello"}\n',
        b'{"jsonrpc":"2.0","id":true,"method":"hello"}\n',
        b'{"jsonrpc":"2.0","id":18446744073709551616,"method":"hello"}\n',
        b'{"jsonrpc":"2.0","method":"hello","method":"hello"}\n',
        b'{"jsonrpc":"2.0","method":"hello","params":NaN}\n',
        b'{"jsonrpc":"2.0","method":"hello"} x\n',
        b'"hello"\n',
    ],
)
def test_what_serde_would_not_read_is_a_parse_error_and_closes(shim: ms.MediaShim, line: bytes):
    with _connect(shim.path) as sock:
        sock.sendall(line + b'{"jsonrpc":"2.0","id":2,"method":"hello"}\n')
        response = _line(sock)
        assert response["id"] is None and response["error"]["code"] == ms.PARSE_ERROR
        assert _rest(sock) == b""  # closed: the hello after it is never answered


@pytest.mark.parametrize(
    ("line", "request_id"),
    [
        (b'{"jsonrpc":"1.0","id":3,"method":"hello"}\n', 3),  # jsonrpc is any string
        (b'{"method":"hello","jsonrpc":"2.0","id":"x","extra":[1,{"a":1,"a":2}]}\n', "x"),
        (b'["2.0",4,"hello"]\n', 4),  # a struct from a sequence, as serde reads it
        (b'  {"jsonrpc":"2.0","id":null,"method":"hello"}\r\n', None),
        (b'{"jsonrpc":"2.0","id":18446744073709551615,"method":"hello"}\n', 2**64 - 1),
    ],
)
def test_what_serde_reads_is_answered(shim: ms.MediaShim, line: bytes, request_id: Any):
    with _connect(shim.path) as sock:
        sock.sendall(line)
        response = _line(sock)
    assert response["id"] == request_id and response["result"]["api_version"] == 37


def test_a_request_line_of_exactly_4096_bytes_is_read(shim: ms.MediaShim):
    head = b'{"jsonrpc":"2.0","id":1,"method":"hello","params":"'
    tail = b'"}\n'
    line = head + b"p" * (ms.MAX_REQUEST_BYTES - len(head) - len(tail)) + tail
    assert len(line) == ms.MAX_REQUEST_BYTES
    assert _reply(shim, line)["result"]["api_version"] == 37
    longer = head + b"p" * (ms.MAX_REQUEST_BYTES + 1 - len(head) - len(tail)) + tail
    assert _reply(shim, longer)["error"]["code"] == ms.INVALID_PARAMS


def test_one_capture_answers_every_reader_waiting_for_it(shim: ms.MediaShim):
    results: list[dict[str, Any]] = []

    def ask() -> None:
        with _connect(shim.path) as sock:
            sock.sendall(b'{"jsonrpc":"2.0","id":1,"method":"media.frame"}\n')
            results.append(_line(sock)["result"])

    readers = [threading.Thread(target=ask) for _ in range(3)]
    for reader in readers:
        reader.start()
    deadline = time.monotonic() + 2
    while shim.active < 3 and time.monotonic() < deadline:
        time.sleep(0.005)
    time.sleep(0.05)
    assert shim.frames.take_request()
    shim.frames.deliver(ms.Frame(2, 1, bytes(4), 42))
    for reader in readers:
        reader.join(timeout=2)
    assert [r["captured_at_unix_us"] for r in results] == [42, 42, 42]


def test_a_reader_gets_the_next_frame_not_the_last_one(shim: ms.MediaShim):
    shim.frames.deliver(ms.Frame(2, 1, bytes(4), 1))  # an old frame nobody asked for
    response = _reply(shim, b'{"jsonrpc":"2.0","id":1,"method":"media.frame"}\n')
    assert response["error"]["code"] == ms.INTERNAL_ERROR


def test_seventeen_waiting_readers_the_seventeenth_is_dropped(shim: ms.MediaShim):
    socks = []
    for _ in range(ms.SLOTS):
        sock = _connect(shim.path)
        sock.sendall(b'{"jsonrpc":"2.0","id":1,"method":"hello"}\n')
        _line(sock)
        socks.append(sock)
    extra = _connect(shim.path)
    extra.settimeout(1.0)
    assert extra.recv(1) == b""
    for sock in [extra, *socks]:
        sock.close()
    deadline = time.monotonic() + 2
    while shim.active and time.monotonic() < deadline:
        time.sleep(0.01)
    assert shim.active == 0
    assert _reply(shim, b'{"jsonrpc":"2.0","id":1,"method":"hello"}\n')["result"]


def test_a_missing_group_is_a_warning_not_a_failure(folder: Path):
    said: list[str] = []
    lock, listener = ms.bind(folder / "g.sock", group="no-such-group-here", log=said.append)
    listener.close()
    lock.close()
    assert any("stays private" in line for line in said)


def test_a_socket_path_too_long_for_macos_is_named(folder: Path):
    with pytest.raises(ms.ShimError):
        ms.bind(folder / ("x" * 120), log=lambda line: None)


def test_header_validity_is_valid_uyvy():
    good = {"width": 1280, "height": 720, "format": "UYVY", "bytes": 1280 * 720 * 2, "rotate": 90}
    assert ms.valid_uyvy(good)
    assert eye_frames.MediaFrameHeader(captured_at_unix_us=0, **good).valid_uyvy()
    for bad in (
        {"width": 3},
        {"format": "YUYV"},
        {"rotate": 45},
        {"bytes": 17},
        {"width": 4096, "height": 4096, "bytes": 4096 * 4096 * 2},
    ):
        head = {**good, **bad}
        assert not ms.valid_uyvy(head), bad
        assert not eye_frames.MediaFrameHeader(captured_at_unix_us=0, **head).valid_uyvy()


# ── the client (duck-eye's frames.py) and the page's eye window ────────────────


def test_the_client_fetches_one_frame_the_way_robotctl_does(shim: ms.MediaShim):
    with Camera(shim.frames, width=4):
        head, data = eye_frames.fetch(shim.path, timeout_s=3.0)
    assert head.valid_uyvy() and (head.width, head.height, head.rotate) == (4, 1, 90)
    assert head.upright_size == (1, 4) and len(data) == head.bytes == 8
    assert shim.stats()["hello"] == 1 and shim.stats()["frames"] == 1


def test_the_client_reports_the_camera_timeout_with_its_code(shim: ms.MediaShim):
    with pytest.raises(eye_frames.FrameError) as caught:
        eye_frames.fetch(shim.path)
    assert caught.value.code == ms.INTERNAL_ERROR


def test_the_client_refuses_a_reply_that_is_not_its_frame(folder: Path):
    path = folder / "fake.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(1)

    def serve() -> None:
        conn, _ = server.accept()
        with conn:
            conn.recv(4096)
            conn.sendall(
                b'{"jsonrpc":"2.0","id":1,"result":{"width":3,"height":1,"format":"UYVY",'
                b'"bytes":6,"captured_at_unix_us":1,"rotate":90}}\n'
            )

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    with pytest.raises(eye_frames.FrameError, match="invalid UYVY"):
        eye_frames.fetch(path, hello=False)
    thread.join(timeout=2)
    server.close()


def test_upright_sampling_turns_the_frame_a_quarter_clockwise():
    np = pytest.importorskip("numpy")
    width, height = 4, 2
    y = np.arange(width * height, dtype=np.uint8).reshape(height, width) * 10 + 5
    packed = np.zeros((height, width // 2, 4), dtype=np.uint8)
    packed[:, :, 0] = 128
    packed[:, :, 2] = 128
    packed[:, :, 1] = y[:, 0::2]
    packed[:, :, 3] = y[:, 1::2]
    data = packed.tobytes()
    upright = eye_frames.upright_luma(data, width, height, 90)
    assert upright.shape == (width, height)
    for uy in range(width):  # uyvy::Turn::Right: upright (ux, uy) from source (uy, h-1-ux)
        for ux in range(height):
            assert upright[uy, ux] == y[height - 1 - ux, uy]
    rgb = eye_frames.upright_rgb(data, width, height, 90)
    assert rgb.shape == (width, height, 3) and (rgb[..., 0] == upright).all()


def test_the_eye_window_fetches_only_while_it_is_open(shim: ms.MediaShim):
    pytest.importorskip("numpy")
    got: list[dict[str, Any]] = []
    with Camera(shim.frames, width=64) as camera:
        window = eye_frames.EyeWindow(shim.path, period_s=0.1, idle_s=0.5, on_frame=got.append)
        window.start()
        try:
            time.sleep(0.4)
            assert camera.taken == 0 and not window.open  # closed: nothing is asked for
            window.touch()
            time.sleep(0.45)
            assert window.open and camera.taken >= 3
            latest = window.latest()
            assert latest is not None and latest["rgb"].shape == (32, 1, 3)  # every 2nd pixel
            assert latest["header"]["rotate"] == 90 and "light" in latest
            window.close_window()
            time.sleep(0.15)
            settled = camera.taken
            time.sleep(0.4)
            assert camera.taken == settled  # closed again: no more frames
        finally:
            window.stop()
    assert got and window.status()["fetched"] == len(got)


def test_the_eye_s_frame_client_knows_nothing_of_the_simulator():
    """duck-eye's frames.py runs the same on the board: no world, no page, no physics."""
    import ast

    source = Path(eye_frames.__file__).read_text()
    forbidden = ("duckmind.world", "duckmind.simweb", "duckmind.duckcore", "mujoco", "mjlab")
    for node in ast.walk(ast.parse(source)):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            assert not name.startswith(forbidden), f"eye/frames.py imports {name}"
