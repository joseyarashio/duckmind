"""The brain's end of the remote-body link: the body's daemons as local sockets.

:class:`RemoteLink` dials the body's bridge (``remote://host:port``), proves it holds the shared
token, checks that the body holds it too, and then offers each channel the body has as a local
endpoint in a private folder — ``robotd.sock``, ``tof.sock``, ``media.sock`` (unix sockets) and,
for a simulated body, a loopback TCP port for the world server. The brain's existing clients
(:class:`~duckmind.robotd.RobotdBodyPort`, the neural driver's ``RobotdClient``, the world's
``HostClient``) connect to those exactly as they connect to a local body.

When the link goes down — the body stops answering pings for :data:`~duckmind.remotebody.link.LOST_S`
or the connection breaks — every local connection is closed (the brain's clients see robotd go
away, as they would locally, and reconnect), and the link dials again with a short backoff. While
it is down the local endpoints stay in place but close each new connection at once.
"""

from __future__ import annotations

import os
import secrets
import shutil
import socket
import statistics
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from duckmind.remotebody import link as L
from duckmind.remotebody.pipe import LocalPipe
from duckmind.remotebody.quality import LinkQuality
from duckmind.tmpdir import short_tmp

RECONNECT_S = (0.1, 0.25, 0.5, 1.0)
UNIX_CHANNELS = {"robotd": "robotd.sock", "tofd": "tof.sock", "media": "media.sock"}
RTT_KEEP = 200


class RemoteLink:
    def __init__(
        self,
        url: str,
        token: bytes,
        *,
        lost_s: float = L.LOST_S,
        heartbeat_s: float = L.HEARTBEAT_S,
        connect_timeout_s: float = 2.0,
        log: Callable[..., None] | None = None,
    ) -> None:
        self.url = L.parse_url(url)
        self._token = token
        self.lost_s = lost_s
        self.heartbeat_s = heartbeat_s
        self.connect_timeout_s = connect_timeout_s
        self.log = log or (lambda event, **fields: None)
        self.dir = Path(tempfile.mkdtemp(prefix="dm-rb-", dir=short_tmp()))
        os.chmod(self.dir, 0o700)
        self.channels: tuple[str, ...] = ()
        self.info: dict[str, Any] = {}
        self.deadman_ms: int | None = None
        self.world_port: int | None = None
        self.error: str | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._up = threading.Event()
        self._sock: socket.socket | None = None
        self._codec: L.FrameCodec | None = None
        self._generation = 0
        self._streams: dict[int, LocalPipe] = {}
        self._next_sid = 1
        self._listeners: list[socket.socket] = []
        self._last_rx = 0.0
        self._rtt_ms: deque[float] = deque(maxlen=RTT_KEEP)
        self.offset_ns: int | None = None  # body monotonic − brain monotonic (pong midpoints)
        self.quality = LinkQuality()  # every pong: round trips, silences at the duck, wall clocks
        self.max_gap_ms = 0.0  # the longest silence from the body while the link was up
        self.connects = 0
        self.drops = 0
        self.up_since: float | None = None
        self.down_since: float | None = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="remote-link", daemon=True)

    # ── lifecycle ──

    def start(self, timeout_s: float = 10.0) -> None:
        """Dial and wait for the first link (raises when it never comes up, with the reason)."""
        self._thread.start()
        if not self._up.wait(timeout_s):
            reason = self.error or "no answer"
            self.close()
            raise ConnectionError(f"remote body {self.url}: {reason}")

    def close(self) -> None:
        self._stop.set()
        self._drop("closing")
        for listener in self._listeners:
            L.close_listener(listener)
        shutil.rmtree(self.dir, ignore_errors=True)

    @property
    def up(self) -> bool:
        return self._up.is_set()

    def wait_up(self, timeout_s: float) -> bool:
        return self._up.wait(timeout_s)

    def endpoint(self, channel: str) -> Path | None:
        name = UNIX_CHANNELS.get(channel)
        return None if name is None or channel not in self.channels else self.dir / name

    @property
    def robotd_socket(self) -> Path:
        return self.dir / UNIX_CHANNELS["robotd"]

    def stats(self) -> dict[str, Any]:
        rtts = list(self._rtt_ms)
        now = time.monotonic()
        return {
            "url": str(self.url),
            "up": self.up,
            "rtt_ms_p50": round(statistics.median(rtts), 1) if rtts else None,
            "rtt_ms_p95": round(sorted(rtts)[int(0.95 * (len(rtts) - 1))], 1) if rtts else None,
            "last_heard_ms": round((now - self._last_rx) * 1000) if self.up else None,
            "connects": self.connects,
            "drops": self.drops,
            "max_gap_ms": round(self.max_gap_ms),
            "deadman_ms": self.deadman_ms,
            "error": self.error,
        }

    # ── dialling ──

    def _run(self) -> None:
        tries = 0
        while not self._stop.is_set():
            try:
                sock, codec, welcome = self._dial()
            except (OSError, L.LinkError) as error:
                self.error = str(error)
                wait = RECONNECT_S[min(tries, len(RECONNECT_S) - 1)]
                tries += 1
                self._stop.wait(wait)
                continue
            tries = 0
            self._adopt(welcome)
            self.quality.new_link()
            with self._lock:
                self._sock, self._codec = sock, codec
                self._generation += 1
                generation = self._generation
                self._last_rx = time.monotonic()
            self.connects += 1
            self.error = None
            self.up_since, self.down_since = time.monotonic(), None
            self._up.set()
            self.log("link_up", url=str(self.url), connects=self.connects)
            pinger = threading.Thread(
                target=self._ping, args=(generation,), name="remote-link-ping", daemon=True
            )
            pinger.start()
            reader = L.FrameReader(sock)
            try:
                while not self._stop.is_set():
                    kind, sid, payload = codec.recv(reader)
                    now = time.monotonic()
                    self.max_gap_ms = max(self.max_gap_ms, (now - self._last_rx) * 1000.0)
                    self._last_rx = now
                    self._frame(kind, sid, payload)
            except (OSError, L.LinkError) as error:
                self._drop(f"link: {error}", generation)

    def _dial(self) -> tuple[socket.socket, L.FrameCodec, dict[str, Any]]:
        sock = socket.create_connection(
            (self.url.host, self.url.port), timeout=self.connect_timeout_s
        )
        try:
            if not L.lan_address(str(sock.getpeername()[0])):
                raise L.LinkError("the body is outside the home LAN")
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            sock.settimeout(L.HANDSHAKE_TIMEOUT_S)
            hello = L.read_json_line(sock)
            body_nonce = hello.get("nonce")
            if hello.get("protocol") != L.PROTOCOL or not isinstance(body_nonce, str):
                raise L.LinkError(f"not a duckmind remote-body bridge: {hello.get('protocol')!r}")
            brain_nonce = secrets.token_hex(16)
            L.send_json_line(
                sock,
                {
                    "protocol": L.PROTOCOL,
                    "nonce": brain_nonce,
                    "proof": L.proof(self._token, b"brain", body_nonce, brain_nonce),
                },
            )
            welcome = L.read_json_line(sock)
            if welcome.get("ok") is not True:
                raise L.AuthError("the body refused our token")
            expected = L.proof(self._token, b"body", body_nonce, brain_nonce)
            given = welcome.get("proof")
            if not L.same_proof(expected, given):
                raise L.AuthError("the body does not hold our token")
            sock.settimeout(None)
            key = L.session_key(self._token, body_nonce, brain_nonce)
            return sock, L.FrameCodec(key, outgoing=b"b", incoming=b"B"), welcome
        except BaseException:
            sock.close()
            raise

    def _adopt(self, welcome: dict[str, Any]) -> None:
        channels = tuple(str(c) for c in welcome.get("channels") or () if c in L.CHANNELS)
        self.info = dict(welcome.get("body") or {})
        deadman = welcome.get("deadman_ms")
        self.deadman_ms = int(deadman) if isinstance(deadman, (int, float)) else None
        if self.channels:
            return  # the endpoints are made once; a channel gone later refuses its connections
        self.channels = channels
        for channel in channels:
            if channel == "world":
                server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server.bind(("127.0.0.1", 0))
                self.world_port = int(server.getsockname()[1])
            else:
                server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                path = self.dir / UNIX_CHANNELS[channel]
                server.bind(str(path))
                os.chmod(path, 0o600)
            server.listen(16)
            self._listeners.append(server)
            threading.Thread(
                target=self._accept,
                args=(server, channel),
                name=f"remote-link-{channel}",
                daemon=True,
            ).start()

    # ── local connections ──

    def _accept(self, server: socket.socket, channel: str) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = server.accept()
            except OSError:
                return
            if channel == "world":
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self._lock:
                codec, sock, generation = self._codec, self._sock, self._generation
                if codec is None or sock is None or not self._up.is_set():
                    conn.close()  # the link is down: robotd "went away"
                    continue
                sid = self._next_sid
                self._next_sid += 1

            def to_body(data: bytes, sid: int = sid, generation: int = generation) -> None:
                self._send(b"D", sid, data, generation)

            def closed(reason: str, sid: int = sid, generation: int = generation) -> None:
                with self._lock:
                    self._streams.pop(sid, None)
                try:
                    self._send(b"C", sid, reason.encode()[:200], generation)
                except (OSError, L.LinkError):
                    pass

            pipe = LocalPipe(conn, on_data=to_body, on_close=closed, name=f"rb-{channel}-{sid}")
            with self._lock:
                self._streams[sid] = pipe
            try:
                self._send(b"O", sid, channel.encode(), generation)
            except (OSError, L.LinkError):
                pipe.close("link down", notify=False)
                with self._lock:
                    self._streams.pop(sid, None)
                continue
            pipe.start()

    def _send(self, kind: bytes, sid: int, payload: bytes, generation: int) -> None:
        with self._lock:
            codec, sock = self._codec, self._sock
            if codec is None or sock is None or generation != self._generation:
                raise L.LinkError("the link is down")
        codec.send(sock, kind, sid, payload)

    def _frame(self, kind: bytes, sid: int, payload: bytes) -> None:
        if kind == b"D":
            pipe = self._streams.get(sid)
            if pipe is not None:
                pipe.put(payload)
        elif kind == b"C":
            with self._lock:
                pipe = self._streams.pop(sid, None)
            if pipe is not None:  # (what came before the close is still written)
                pipe.finish(payload.decode("utf-8", "replace"))
        elif kind == b"Q":
            parts = payload.split(b" ")
            if len(parts) in (2, 3):
                sent, body_ns = int(parts[0]), int(parts[1])
                now = time.monotonic_ns()
                wall = time.time_ns() - (now - sent) // 2 if len(parts) == 3 else None
                self._rtt_ms.append((now - sent) / 1e6)
                self.offset_ns = body_ns - (sent + now) // 2
                self.quality.pong(
                    now - sent, body_ns, None if wall is None else int(parts[2]) - wall
                )
        else:
            raise L.LinkError(f"unexpected frame {kind!r} from the body")

    def _ping(self, generation: int) -> None:
        while not self._stop.is_set():
            with self._lock:
                if generation != self._generation or self._sock is None:
                    return
            silent = time.monotonic() - self._last_rx
            if silent > self.lost_s:
                self._drop(f"body silent {silent * 1000:.0f} ms", generation)
                return
            try:
                self._send(b"P", 0, b"%d" % time.monotonic_ns(), generation)
            except (OSError, L.LinkError) as error:
                self._drop(f"ping: {error}", generation)
                return
            self._stop.wait(self.heartbeat_s)

    def _drop(self, reason: str, generation: int | None = None) -> None:
        with self._lock:
            if generation is not None and generation != self._generation:
                return
            sock, self._sock, self._codec = self._sock, None, None
            self._generation += 1
            streams, self._streams = list(self._streams.values()), {}
            was_up = self._up.is_set()
            self._up.clear()
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()
        for pipe in streams:
            pipe.close("remote body link lost", notify=False)
        if was_up:
            self.drops += 1
            self.up_since, self.down_since = None, time.monotonic()
            self.error = reason
            self.log("link_lost", url=str(self.url), reason=reason)
