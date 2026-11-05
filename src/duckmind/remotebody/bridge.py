"""The body-side bridge: robotd stays on the body; the brain reaches it through this, over the LAN.

It listens on one TCP port, lets in one brain that proves it holds the shared token (see
:mod:`duckmind.remotebody.link`), and forwards that brain's streams to the body's own daemons over
their unix sockets — robotd, and optionally tofd, mediad's frame socket and (a simulated body only)
the world server. It forwards only what a brain needs: requests outside :data:`ROBOTD_METHODS` (and
the other channels' lists) are refused with an error, never passed on.

**The deadman.** The brain pings every :data:`~duckmind.remotebody.link.HEARTBEAT_S`. When nothing
has come from it for ``deadman_s`` (:data:`~duckmind.remotebody.link.DEADMAN_S`), or the
connection breaks, or a frame fails its signature, the bridge drops the link — every stream closes,
so nothing the brain sent before can arrive late — and :class:`SafetyKeeper` sends ``robot.stop``
at once and then sits the duck down (``robot.do sit_toggle`` while it stands), retrying until it
sits or a brain is back. robotd's own deadman has already zeroed a walk 0.5 s after its last
``robot.move``; the bridge adds the explicit stop, the sit and the guarantee that a stalled link's
backlog is never applied.
"""

from __future__ import annotations

import argparse
import json
import secrets
import socket
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from duckmind.body import BodyIntent
from duckmind.remotebody import link as L
from duckmind.remotebody.pipe import LocalPipe

ROBOTD_METHODS = frozenset(
    {
        "hello",
        "robot.subscribe",
        "robot.move",
        "robot.head",
        "robot.mouth",
        "robot.pose",
        "robot.look",
        "robot.do",
        "robot.stop",
        "robot.health",
        "robot.model",
        "robot.modelApi",
        "robot.policies",
        "robot.safeToRestart",
        # the brain's operator: the walking policy on/off, the ramp home, play dead's pose
        "robot.enable",
        "robot.init",
        "robot.poseJoints",
    }
)
CHANNEL_METHODS: dict[str, frozenset[str]] = {
    "robotd": ROBOTD_METHODS,
    "tofd": frozenset({"hello", "tof.stream", "tof.frame"}),
    "media": frozenset({"hello", "media.frame"}),
}
# The simulated body's port also takes robotd's servo writes; only robotd may send those.
WORLD_REFUSED_OPS = frozenset({"write", "gain", "torque"})
MAX_REQUEST_LINE = 256 * 1024
MAX_PENDING_BYTES = 1024 * 1024  # requests held while a daemon's socket connects
SIT_RETRY_S = 3.0  # a sit_toggle robotd accepted gets this long before it is sent again
SIT_REFUSED_RETRY_S = 0.5
AUTH_FAIL_DELAY_S = 0.5
MAX_HANDSHAKES = 4
WATCH_PERIOD_S = 0.02


@dataclass(frozen=True)
class Channel:
    name: str
    unix: str | None = None
    tcp: tuple[str, int] | None = None

    def connect(self, timeout_s: float = 1.0) -> socket.socket:
        if self.unix is not None:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(timeout_s)
            try:
                sock.connect(self.unix)
            except OSError:
                sock.close()
                raise
        else:
            assert self.tcp is not None
            sock = socket.create_connection(self.tcp, timeout=timeout_s)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(None)
        return sock


def check_request(channel: str, line: bytes) -> tuple[bool, bytes | None]:
    """(forward it, the refusal to answer the brain with). A refused notification gets no answer,
    exactly like robotd's own notifications."""
    try:
        request = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False, None
    if not isinstance(request, dict):
        return False, None
    if channel == "world":
        op = request.get("op")
        if isinstance(op, str) and op not in WORLD_REFUSED_OPS:
            return True, None
        answer = {"error": f"refused by the remote-body bridge: op {op!r}"}
        return False, json.dumps(answer).encode() + b"\n"
    method = request.get("method")
    if isinstance(method, str) and method in CHANNEL_METHODS.get(channel, frozenset()):
        return True, None
    if "id" not in request:
        return False, None
    answer = {
        "jsonrpc": "2.0",
        "id": request.get("id"),
        "error": {"code": -32601, "message": f"refused by the remote-body bridge: {method!r}"},
    }
    return False, json.dumps(answer).encode() + b"\n"


class SafetyKeeper:
    """The body's own robotd connection, used only when the link is lost: robot.stop, then sit.

    ``port_factory`` opens a :class:`~duckmind.robotd.RobotdBodyPort` (or anything with
    ``observe()``, ``stop()``, ``dispatch_intent()`` and ``close()``)."""

    def __init__(
        self,
        port_factory: Callable[[], Any],
        *,
        sit: bool = True,
        log: Callable[..., None] = lambda event, **fields: None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._factory = port_factory
        self._sit = sit
        self._log = log
        self._clock = clock
        self._port: Any = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.engaged = False
        self.engaged_at = 0.0
        self.policy: str | None = None
        self.fallen = False
        self._next_sit = 0.0
        self._sat_logged = False
        self._thread = threading.Thread(target=self._run, name="safety-keeper", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            port, self._port = self._port, None
        if port is not None:
            try:
                port.close()
            except Exception:
                pass

    def _ensure(self) -> Any:
        with self._lock:
            if self._port is None:
                self._port = self._factory()
            return self._port

    def _lose(self, port: Any) -> None:
        with self._lock:
            if self._port is port:
                self._port = None
        try:
            port.close()
        except Exception:
            pass

    def engage(self, reason: str) -> None:
        """The link is lost: stop now (in the caller's thread), then sit (the keeper's thread)."""
        with self._lock:
            self.engaged = True
            self.engaged_at = self._clock()
            self._next_sit = 0.0
            self._sat_logged = False
        for attempt in (1, 2):
            port: Any = None
            try:
                port = self._ensure()
                ack = port.stop()
                self._log("stop_sent", reason=reason, accepted=bool(getattr(ack, "accepted", True)))
                return
            except Exception as error:
                if port is not None:
                    self._lose(port)
                if attempt == 2:
                    self._log("stop_failed", reason=reason, error=str(error))

    def release(self) -> None:
        """A brain is back in charge."""
        with self._lock:
            was = self.engaged
            self.engaged = False
        if was:
            self._log("released")

    def _run(self) -> None:
        while not self._stop.is_set():
            port: Any = None
            try:
                port = self._ensure()
                observed = port.observe()
            except Exception:
                if port is not None:
                    self._lose(port)
                self._stop.wait(0.5)
                continue
            self.policy = observed.policy
            self.fallen = bool(observed.safety.fallen)
            self._act(port)

    def _act(self, port: Any) -> None:
        with self._lock:
            if not self.engaged or not self._sit:
                return
            now = self._clock()
            if self.policy == "sit":
                if not self._sat_logged:
                    self._sat_logged = True
                    self._log("sat", after_s=round(now - self.engaged_at, 3))
                return
            if self.fallen or self.policy not in ("walk", "stand") or now < self._next_sit:
                return
            try:
                ack = port.dispatch_intent(BodyIntent("robot.do", {"skill": "sit_toggle"}))
            except Exception as error:
                self._next_sit = now + SIT_REFUSED_RETRY_S
                self._log("sit_failed", error=str(error))
                return
            accepted = bool(getattr(ack, "accepted", False))
            self._next_sit = now + (SIT_RETRY_S if accepted else SIT_REFUSED_RETRY_S)
            self._log("sit_sent", accepted=accepted, after_s=round(now - self.engaged_at, 3))


class _Stream:
    def __init__(self, channel: str) -> None:
        self.channel = channel
        self.pipe: LocalPipe | None = None  # None while the daemon's socket is connecting
        self.pending: list[bytes] = []
        self.pending_bytes = 0
        self.lines = L.LineSplitter(MAX_REQUEST_LINE)

    def forward(self, line: bytes) -> None:
        """To the daemon now, or once its socket is connected (bounded meanwhile)."""
        pipe = self.pipe
        if pipe is not None:
            pipe.put(line)
            return
        self.pending_bytes += len(line)
        if self.pending_bytes > MAX_PENDING_BYTES:
            raise L.LinkError("the daemon is too slow to accept")
        self.pending.append(line)


class _Link:
    def __init__(self, sock: socket.socket, codec: L.FrameCodec, peer: str) -> None:
        self.sock = sock
        self.codec = codec
        self.peer = peer
        self.streams: dict[int, _Stream] = {}
        self.last_rx = time.monotonic()
        self.max_gap = 0.0  # the longest silence from the brain (s)
        self.busy: tuple[float, bytes] | None = None  # (since, kind): the reader handles a frame
        self.dead = False
        self.lock = threading.Lock()

    def send(self, kind: bytes, stream: int, payload: bytes = b"") -> None:
        if self.dead:
            raise L.LinkError("link is down")
        self.codec.send(self.sock, kind, stream, payload)


class Bridge:
    def __init__(
        self,
        *,
        listen: tuple[str, int],
        token: bytes,
        channels: dict[str, Channel],
        keeper: SafetyKeeper | None,
        body_info: dict[str, Any] | None = None,
        deadman_s: float = L.DEADMAN_S,
        log: Callable[..., None] | None = None,
    ) -> None:
        if not (L.DEADMAN_MIN_S <= deadman_s <= L.DEADMAN_MAX_S):
            raise ValueError(f"deadman must be {L.DEADMAN_MIN_S}..{L.DEADMAN_MAX_S} s")
        host = listen[0]
        if host not in ("0.0.0.0", "::") and not L.lan_address(host):
            raise ValueError(f"the bridge listens on the home LAN only, not {host!r}")
        unknown = set(channels) - set(L.CHANNELS)
        if unknown or "robotd" not in channels:
            raise ValueError("channels: robotd is required; known: " + ", ".join(L.CHANNELS))
        self.token = token
        self.channels = channels
        self.keeper = keeper
        self.body_info = dict(body_info or {})
        self.deadman_s = deadman_s
        self.log = log or (lambda event, **fields: None)
        family = socket.AF_INET6 if ":" in host else socket.AF_INET
        self._server = socket.socket(family, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(listen)
        self._server.listen(8)
        self.port = int(self._server.getsockname()[1])
        self._lock = threading.Lock()
        self._link: _Link | None = None
        self._handshakes = threading.BoundedSemaphore(MAX_HANDSHAKES)
        self._stop = threading.Event()
        self.links = 0
        self.drops: list[str] = []

    # ── lifecycle ──

    def start(self) -> None:
        if self.keeper is not None:
            self.keeper.start()
        threading.Thread(target=self._accept, name="bridge-accept", daemon=True).start()
        threading.Thread(target=self._watch, name="bridge-deadman", daemon=True).start()

    def close(self) -> None:
        self._stop.set()
        L.close_listener(self._server)
        link = self._link
        if link is not None:
            self._drop(link, "bridge closing", engage=True)
        if self.keeper is not None:
            self.keeper.close()

    @property
    def link_up(self) -> bool:
        link = self._link
        return link is not None and not link.dead

    # ── accepting a brain ──

    def _accept(self) -> None:
        while not self._stop.is_set():
            try:
                conn, addr = self._server.accept()
            except OSError:
                return
            peer = str(addr[0])
            if not L.lan_address(peer):
                self.log("refused_peer", peer=peer)
                conn.close()
                continue
            if not self._handshakes.acquire(blocking=False):
                conn.close()
                continue
            threading.Thread(
                target=self._serve, args=(conn, peer), name="bridge-link", daemon=True
            ).start()

    def _handshake(self, conn: socket.socket) -> L.FrameCodec:
        conn.settimeout(L.HANDSHAKE_TIMEOUT_S)
        body_nonce = secrets.token_hex(16)
        L.send_json_line(conn, {"protocol": L.PROTOCOL, "nonce": body_nonce})
        hello = L.read_json_line(conn)
        brain_nonce = hello.get("nonce")
        if hello.get("protocol") != L.PROTOCOL or not isinstance(brain_nonce, str):
            raise L.AuthError("not a duckmind remote-body brain")
        if not (16 <= len(brain_nonce) <= 128) or brain_nonce == body_nonce:
            raise L.AuthError("bad nonce")
        expected = L.proof(self.token, b"brain", body_nonce, brain_nonce)
        given = hello.get("proof")
        if not L.same_proof(expected, given):
            raise L.AuthError("wrong token")
        L.send_json_line(
            conn,
            {
                "ok": True,
                "proof": L.proof(self.token, b"body", body_nonce, brain_nonce),
                "channels": sorted(self.channels),
                "deadman_ms": round(self.deadman_s * 1000),
                "body": self.body_info,
            },
        )
        conn.settimeout(None)
        key = L.session_key(self.token, body_nonce, brain_nonce)
        return L.FrameCodec(key, outgoing=b"B", incoming=b"b")

    def _serve(self, conn: socket.socket, peer: str) -> None:
        try:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            try:
                codec = self._handshake(conn)
            except (L.LinkError, OSError) as error:
                self.log("auth_failed", peer=peer, error=str(error))
                time.sleep(AUTH_FAIL_DELAY_S)
                try:
                    L.send_json_line(conn, {"ok": False, "error": "refused"})
                except OSError:
                    pass
                conn.close()
                return
        finally:
            self._handshakes.release()
        link = _Link(conn, codec, peer)
        with self._lock:
            old, self._link = self._link, link
            self.links += 1
        if self.keeper is not None:
            self.keeper.release()
        if old is not None:
            self._drop(old, "replaced by a new brain", engage=False)
        self.log("link_up", peer=peer)
        reader = L.FrameReader(conn)
        try:
            while not link.dead:
                kind, sid, payload = codec.recv(reader)
                now = time.monotonic()
                link.max_gap = max(link.max_gap, now - link.last_rx)
                link.last_rx = now
                link.busy = (now, kind)
                self._frame(link, kind, sid, payload)
                link.busy = None
        except (L.LinkError, OSError) as error:
            self._drop(link, f"link: {error}", engage=True)

    # ── frames from the brain ──

    def _frame(self, link: _Link, kind: bytes, sid: int, payload: bytes) -> None:
        if kind == b"P":
            link.send(b"Q", 0, payload + b" %d" % time.monotonic_ns())
        elif kind == b"O":
            self._open(link, sid, payload.decode("ascii", "replace"))
        elif kind == b"D":
            stream = link.streams.get(sid)
            if stream is None:
                return
            try:
                lines = stream.lines.feed(payload)
                for line in lines:
                    ok, answer = check_request(stream.channel, line)
                    if ok:
                        with link.lock:
                            stream.forward(line)
                    elif answer is not None:
                        link.send(b"D", sid, answer)
            except L.LinkError as error:
                self._close_stream(link, sid, str(error), tell=True)
        elif kind == b"C":
            self._close_stream(link, sid, "brain closed it", tell=False)
        else:
            raise L.LinkError(f"unexpected frame {kind!r} from the brain")

    def _open(self, link: _Link, sid: int, name: str) -> None:
        """Register the stream at once and connect to the daemon in a thread of its own: a daemon
        slow to accept must never stall the link's reader (and so its heartbeat)."""
        channel = self.channels.get(name)
        if channel is None or sid in link.streams:
            link.send(b"C", sid, b"no such channel")
            return
        stream = _Stream(name)
        with link.lock:
            link.streams[sid] = stream
        threading.Thread(
            target=self._connect_stream,
            args=(link, sid, stream, channel),
            name=f"bridge-open-{name}-{sid}",
            daemon=True,
        ).start()

    def _connect_stream(self, link: _Link, sid: int, stream: _Stream, channel: Channel) -> None:
        try:
            sock = channel.connect()
        except OSError as error:
            self._close_stream(link, sid, f"{channel.name}: {error}", tell=True)
            return

        def to_brain(data: bytes) -> None:
            link.send(b"D", sid, data)

        def closed(reason: str) -> None:
            with link.lock:
                if link.streams.get(sid) is stream:
                    del link.streams[sid]
            try:
                link.send(b"C", sid, reason.encode()[:200])
            except (L.LinkError, OSError):
                pass

        pipe = LocalPipe(
            sock, on_data=to_brain, on_close=closed, name=f"bridge-{channel.name}-{sid}"
        )
        with link.lock:
            alive = not link.dead and link.streams.get(sid) is stream
            if alive:
                stream.pipe = pipe
                for line in stream.pending:  # (what the brain sent while it connected)
                    pipe.put(line)
                stream.pending.clear()
        if not alive:
            pipe.close("closed while connecting", notify=False)
            return
        pipe.start()

    def _close_stream(self, link: _Link, sid: int, reason: str, *, tell: bool) -> None:
        with link.lock:
            stream = link.streams.pop(sid, None)
        if stream is not None and stream.pipe is not None:
            stream.pipe.close(reason, notify=False)
        if tell:
            try:
                link.send(b"C", sid, reason.encode()[:200])
            except (L.LinkError, OSError):
                pass

    # ── losing the brain ──

    def _watch(self) -> None:
        while not self._stop.wait(WATCH_PERIOD_S):
            link = self._link
            if link is not None and not link.dead:
                silent = time.monotonic() - link.last_rx
                if silent > self.deadman_s:
                    busy = link.busy
                    why = f"deadman: brain silent {silent * 1000:.0f} ms"
                    if busy is not None:  # (diagnosis: the reader itself was held up)
                        why += f" (reader busy {busy[1]!r} for {(time.monotonic() - busy[0]) * 1000:.0f} ms)"
                    self._drop(link, why, engage=True)

    def _drop(self, link: _Link, reason: str, *, engage: bool) -> None:
        with self._lock:
            if link.dead:
                return
            link.dead = True
            current = self._link is link
            if current:
                self._link = None
        try:
            link.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        link.sock.close()
        with link.lock:
            streams, link.streams = list(link.streams.values()), {}
        for stream in streams:
            if stream.pipe is not None:
                stream.pipe.close("link lost", notify=False)
        self.drops.append(reason)
        self.log("link_lost", peer=link.peer, reason=reason, max_gap_ms=round(link.max_gap * 1000))
        if engage and current and self.keeper is not None:
            self.keeper.engage(reason)


# ── command line ─────────────────────────────────────────────────────────


def event_logger(path: Path | None) -> Callable[..., None]:
    """Events as JSON lines (wall time ``t``, so two processes on one machine line up)."""
    handle = path.open("a", encoding="utf-8") if path is not None else None
    lock = threading.Lock()

    def log(event: str, **fields: Any) -> None:
        line = json.dumps({"t": time.time(), "event": event, **fields}, ensure_ascii=False)
        with lock:
            print(f"remote-body: {line}", flush=True)
            if handle is not None:
                handle.write(line + "\n")
                handle.flush()

    return log


def host_port(text: str, default_port: int = L.DEFAULT_PORT) -> tuple[str, int]:
    host, sep, port = text.rpartition(":")
    if not sep:
        return text, default_port
    return host.strip("[]"), int(port)


def keeper_for(robotd: str, backend: str, *, sit: bool, log: Callable[..., None]) -> SafetyKeeper:
    from duckmind.robotd import RobotdBodyPort

    return SafetyKeeper(
        lambda: RobotdBodyPort(robotd, backend=backend, timeout_s=1.0, hz=10), sit=sit, log=log
    )


def add_bridge_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--listen",
        default=f"127.0.0.1:{L.DEFAULT_PORT}",
        help="host:port to listen on (a LAN address, or 0.0.0.0; default loopback only)",
    )
    parser.add_argument("--token-file", type=Path, required=True, help="the shared token (0600)")
    parser.add_argument(
        "--deadman-ms",
        type=float,
        default=L.DEADMAN_S * 1000,
        help="brain silence before stop + sit",
    )
    parser.add_argument("--no-sit", action="store_true", help="on link loss stop only, never sit")
    parser.add_argument("--events", type=Path, help="append link/safety events here (JSON lines)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m duckmind.remotebody bridge",
        description="Body-side bridge: robotd (and tofd/mediad) for one brain on the home LAN.",
    )
    add_bridge_arguments(parser)
    parser.add_argument("--robotd", default="/run/robotd.sock")
    parser.add_argument("--tofd", help="tofd's socket (e.g. /run/tofd/tof.sock)")
    parser.add_argument("--media", help="mediad's frame socket (e.g. /run/mediad/media.sock)")
    parser.add_argument("--backend", choices=("real", "simulator"), default="real")
    args = parser.parse_args(argv)
    log = event_logger(args.events)
    channels = {"robotd": Channel("robotd", unix=args.robotd)}
    if args.tofd:
        channels["tofd"] = Channel("tofd", unix=args.tofd)
    if args.media:
        channels["media"] = Channel("media", unix=args.media)
    bridge = Bridge(
        listen=host_port(args.listen),
        token=L.read_token(args.token_file),
        channels=channels,
        keeper=keeper_for(args.robotd, args.backend, sit=not args.no_sit, log=log),
        body_info={"backend": args.backend},
        deadman_s=args.deadman_ms / 1000.0,
        log=log,
    )
    bridge.start()
    log("listening", port=bridge.port, channels=sorted(channels))
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
