"""The body-side bridge: robotd stays on the body; the brain reaches it through this, over the LAN.

It listens on one TCP port, lets in one brain that proves it holds the shared token (see
:mod:`duckmind.remotebody.link`), and forwards that brain's streams to the body's own daemons over
their unix sockets — robotd, and optionally tofd, mediad's frame socket and (a simulated body only)
the world server. It forwards only what a brain needs: requests outside :data:`ROBOTD_METHODS` (and
the other channels' lists) are refused with an error, never passed on.

**duckmind never turns a real duck on** (SAFETY.md). The brain's operator methods
(:data:`OPERATOR_METHODS`: ``robot.enable``, ``robot.init``, ``robot.poseJoints``) pass only when
the bridge's body is a simulated one (``body_info`` says ``backend: simulator``). For every other
body — a real duck, ``sim-body --as-real``, a bridge that does not say — they are refused with a
JSON-RPC error that says why, never reach robotd, and the bridge logs ``refused_operator``. A person
turns a real duck on with the duck's own control.

**The deadman.** The brain pings every :data:`~duckmind.remotebody.link.HEARTBEAT_S`. When nothing
has come from it for ``deadman_s`` (:data:`~duckmind.remotebody.link.DEADMAN_S`), or the
connection breaks, or a frame fails its signature, the bridge drops the link — every stream closes,
so nothing the brain sent before can arrive late — and :class:`SafetyKeeper` sends ``robot.stop``
at once and then sits the duck down (``robot.do sit_toggle`` while it stands; asked again only by
what robotd's label says, :class:`SafetyKeeper`) until it sits or a brain is back. robotd's own
deadman has already zeroed a walk 0.5 s after its last
``robot.move``; the bridge adds the explicit stop, the sit and the guarantee that a stalled link's
backlog is never applied.

**Pollen's usage data.** On a real duck the bridge asks updaterd for its telemetry switch when it
starts and turns it off once (the owner's choice, 2026-11-08; ``--keep-pollen-telemetry`` leaves it
alone), logs a ``telemetry`` event and reports the state to the brain's ``check``
(:mod:`duckmind.remotebody.telemetry`).

**Pollen's other doors.** mediad (TCP 8443, 8080) and netpadd (UDP 4210) let a LAN peer drive the
duck around this bridge and its deadman. A real duck's bridge knocks on them when it starts and
does not serve while one is open (exit 2, the fix on stderr) unless ``--allow-open-mediad`` is
given (:mod:`duckmind.remotebody.doors`, ``tools/duck/close-mediad.sh``).

**Wrong tokens.** An address that offers a wrong token :data:`AUTH_LOCK_AFTER` times within
:data:`AUTH_LOCK_WINDOW_S` is locked out for :data:`AUTH_LOCK_S` (``peer_locked`` in the log).
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
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
from duckmind.remotebody import doors, telemetry
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
    }
)
# The brain's operator: the walking policy on/off, torque on and the ramp home, play dead's
# scripted pose. Passed on for a simulated body only: a real duck is turned on, homed and posed
# by a person, never through this bridge (SAFETY.md).
OPERATOR_METHODS = frozenset({"robot.enable", "robot.init", "robot.poseJoints"})
OPERATOR_REFUSAL = (
    "refused by the remote-body bridge: {method} is never sent to a real duck; "
    "a person turns the duck on with its own control"
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
# The seat (robotd 0.18.5, ``Controller::sit_toggle``): ``robot.do sit_toggle`` is answered
# ``accepted`` when it is queued; the control loop then sits a standing duck, refuses a toggle for
# the seat's ramp (robotd-params ``DEFAULT_SITSTAND_RAMP_S``, said only in its own log) and
# STANDS a seated duck on a toggle after it. So a sit robotd accepted is asked again only when
# its label has read walk/stand without a break for the ramp and a margin, on robotd's own clock.
SIT_RAMP_S = 2.0
SIT_MARGIN_S = 2.0
SIT_REFUSED_RETRY_S = 0.5
AUTH_FAIL_DELAY_S = 0.5
# An address that offers a wrong token AUTH_LOCK_AFTER times within AUTH_LOCK_WINDOW_S is not
# listened to for AUTH_LOCK_S (no handshake is begun; it is told how long). A right token
# forgets the wrong ones before it.
AUTH_LOCK_AFTER = 5
AUTH_LOCK_WINDOW_S = 60.0
AUTH_LOCK_S = 60.0
AUTH_LOCK_PEERS = 256  # addresses remembered at most (the oldest go first)
MAX_HANDSHAKES = 4
WATCH_PERIOD_S = 0.02


@dataclass(frozen=True)
class Channel:
    name: str
    unix: str | None = None
    tcp: tuple[str, int] | None = None
    # (a simulated body only) each whole line the daemon sends, rewritten before the brain gets
    # it: the sim-body's robot.health carries the world's pack (remotebody.battery)
    answers: Callable[[bytes], bytes] | None = None

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


def check_request(
    channel: str,
    line: bytes,
    *,
    operator: bool = False,
    refused_operator: Callable[[str], None] | None = None,
) -> tuple[bool, bytes | None]:
    """(forward it, the refusal to answer the brain with). A refused notification gets no answer,
    exactly like robotd's own notifications. ``operator``: the body is a simulated one, so the
    brain's :data:`OPERATOR_METHODS` pass; otherwise they are refused (``refused_operator`` is
    told which, for the bridge's log)."""
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
    refusal = "refused by the remote-body bridge: {method!r}"
    if isinstance(method, str):
        if method in CHANNEL_METHODS.get(channel, frozenset()):
            return True, None
        if channel == "robotd" and method in OPERATOR_METHODS:
            if operator:
                return True, None
            if refused_operator is not None:
                refused_operator(method)
            refusal = OPERATOR_REFUSAL
    if "id" not in request:
        return False, None
    error = {"code": -32601, "message": refusal.format(method=method)}
    answer = {"jsonrpc": "2.0", "id": request.get("id"), "error": error}
    return False, json.dumps(answer).encode() + b"\n"


class SafetyKeeper:
    """The body's own robotd connection, used only when the link is lost: robot.stop, then sit.

    ``port_factory`` opens a :class:`~duckmind.robotd.RobotdBodyPort` (or anything with
    ``observe()``, ``stop()``, ``dispatch_intent()`` and ``close()``).

    The sit is a toggle, so it is sent by what robotd's label says, never by time alone: at once
    while the label reads walk/stand; never while it reads anything else (sitting, rising, a
    move, held, fallen); and after a toggle robotd accepted, again only when the label has read
    walk/stand in every frame since for :data:`SIT_RAMP_S` + :data:`SIT_MARGIN_S` of robotd's
    own clock (``robot.state``'s ``t``) — a second toggle after the seat's ramp would stand a
    duck that did sit and whose label was slow to say so."""

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
        self._robot_t = 0.0  # robotd's clock in the last frame
        self._upright_since: float | None = None  # … at the first of the unbroken walk/stand frames
        self._asked: float | None = None  # … when it accepted a sit_toggle its label has not shown
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
            if not self.step():
                self._stop.wait(0.5)

    def step(self) -> bool:
        """One ``robot.state`` frame: what the duck is doing and, while the link is lost, the
        sit. False: robotd did not answer (its connection is opened again at the next step)."""
        port: Any = None
        try:
            port = self._ensure()
            observed = port.observe()
        except Exception:
            if port is not None:
                self._lose(port)
            return False
        self.policy = observed.policy
        self.fallen = bool(observed.safety.fallen)
        robot_t = getattr(observed, "robot_time_s", None)
        self._act(port, self._clock() if robot_t is None else float(robot_t))
        return True

    def _act(self, port: Any, robot_t: float) -> None:
        with self._lock:
            upright = self.policy in ("walk", "stand") and not self.fallen
            if robot_t < self._robot_t:  # robotd started again: its clock begins anew
                self._upright_since = None
                self._asked = None if self._asked is None else robot_t
            self._robot_t = robot_t
            if not upright:
                self._upright_since = None
            elif self._upright_since is None:
                self._upright_since = robot_t
            if self.policy == "sit":
                self._asked = None
            if not self.engaged or not self._sit:
                return
            now = self._clock()
            if self.policy == "sit":
                if not self._sat_logged:
                    self._sat_logged = True
                    self._log("sat", after_s=round(now - self.engaged_at, 3))
                return
            if not upright or self._upright_since is None or now < self._next_sit:
                return
            wait = SIT_RAMP_S + SIT_MARGIN_S
            if self._asked is not None and robot_t - max(self._asked, self._upright_since) < wait:
                return
            try:
                ack = port.dispatch_intent(BodyIntent("robot.do", {"skill": "sit_toggle"}))
            except Exception as error:
                # (no answer is not a refusal: robotd may have queued it)
                self._asked, self._next_sit = robot_t, now + SIT_REFUSED_RETRY_S
                self._log("sit_failed", error=str(error))
                return
            accepted = bool(getattr(ack, "accepted", False))
            if accepted:
                self._asked = robot_t
            self._next_sit = now + (wait if accepted else SIT_REFUSED_RETRY_S)
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
        self._auth: dict[str, list[float]] = {}  # address → [wrong tokens, since, locked until]

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

    @property
    def simulated(self) -> bool:
        """The body is a simulated one (its ``body_info`` says ``backend: simulator``): only then
        do the brain's :data:`OPERATOR_METHODS` reach robotd. A real duck, ``sim-body --as-real``
        and a bridge that does not say are never enabled, homed or posed through the link."""
        return self.body_info.get("backend") == "simulator"

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
            locked_s = self._locked_for(peer)
            if locked_s > 0:
                try:
                    L.send_json_line(conn, {"protocol": L.PROTOCOL, "locked_s": locked_s})
                except OSError:
                    pass
                conn.close()
                continue
            if not self._handshakes.acquire(blocking=False):
                conn.close()
                continue
            threading.Thread(
                target=self._serve, args=(conn, peer), name="bridge-link", daemon=True
            ).start()

    def _locked_for(self, peer: str) -> int:
        """Whole seconds this address is still locked out for (0: it is listened to)."""
        entry = self._auth.get(peer)
        left = 0.0 if entry is None else entry[2] - time.monotonic()
        return max(0, math.ceil(left))

    def _wrong_token(self, peer: str) -> None:
        now = time.monotonic()
        with self._lock:
            entry = self._auth.get(peer)
            if entry is None or now - entry[1] > AUTH_LOCK_WINDOW_S:
                entry = self._auth[peer] = [0.0, now, 0.0]
                while len(self._auth) > AUTH_LOCK_PEERS:
                    del self._auth[next(iter(self._auth))]
            entry[0] += 1
            lock = entry[0] >= AUTH_LOCK_AFTER and entry[2] <= now
            if lock:
                entry[:] = [0.0, now, now + AUTH_LOCK_S]
        if lock:
            self.log("peer_locked", peer=peer, seconds=AUTH_LOCK_S, wrong_tokens=AUTH_LOCK_AFTER)

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
                if isinstance(error, L.AuthError):
                    self._wrong_token(peer)
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
            self._auth.pop(peer, None)  # (a right token forgets the wrong ones before it)
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
            link.send(b"Q", 0, payload + b" %d %d" % (time.monotonic_ns(), time.time_ns()))
        elif kind == b"O":
            self._open(link, sid, payload.decode("ascii", "replace"))
        elif kind == b"D":
            stream = link.streams.get(sid)
            if stream is None:
                return

            def refused(method: str) -> None:
                self.log("refused_operator", method=method, peer=link.peer)

            try:
                lines = stream.lines.feed(payload)
                operator = self.simulated
                for line in lines:
                    ok, answer = check_request(
                        stream.channel, line, operator=operator, refused_operator=refused
                    )
                    if ok:
                        with link.lock:
                            stream.forward(line)
                    elif answer is not None:
                        link.send(b"D", sid, answer)
            except L.LinkError as error:
                self._close_stream(link, sid, str(error), tell=True)
        elif kind == b"C":
            # (closed at once: requests still queued for a stalled daemon are dropped, never applied
            # late; only the brain's end drains a closed stream, LocalPipe.finish)
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

        answers = channel.answers
        if answers is None:

            def to_brain(data: bytes) -> None:
                link.send(b"D", sid, data)

        else:
            lines = bytearray()

            def to_brain(data: bytes) -> None:
                lines.extend(data)
                end = lines.rfind(b"\n")
                if end < 0:
                    if len(lines) > MAX_REQUEST_LINE:  # (no line this long to rewrite: pass it on)
                        link.send(b"D", sid, bytes(lines))
                        lines.clear()
                    return
                whole = bytes(lines[: end + 1])
                del lines[: end + 1]
                link.send(b"D", sid, b"".join(answers(line) for line in whole.splitlines(True)))

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


def real_body_info(
    walk: Path | None = None, base: Path | None = None, user: Path | None = None
) -> dict[str, Any]:
    """What the bridge on a real duck tells the brain about it at the handshake (read only, from
    the duck's own files): the official walk and sit-stand networks' sha256 and set version
    (:func:`duckmind.policy_approval.official_policy_hashes`; the brain's measured walk and seat
    tables are keyed by them) and the board's merged ``robotd.toml`` (board, idle, pickup,
    limp-fall: :mod:`duckmind.robotd_config`; None when this user cannot read it)."""
    from duckmind import policy_approval, robotd_config

    hashes = policy_approval.official_policy_hashes(
        walk or Path(policy_approval.OFFICIAL_WALK_PATH)
    )
    config = robotd_config.board_config(
        base or robotd_config.BASE_PATH, user or robotd_config.USER_PATH
    )
    return {
        "backend": "real",
        "policies": hashes,
        "robotd_config": None if config is None else config.to_json(),
    }


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
    parser.add_argument(
        "--has-dock",
        action="store_true",
        help="a real duck only: it has a charging dock it can back into (default: none exists, "
        "and the brain never looks for one; the family swaps or charges the pack)",
    )


def bridge_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m duckmind.remotebody bridge",
        description="Body-side bridge: robotd (and tofd/mediad) for one brain on the home LAN.",
    )
    add_bridge_arguments(parser)
    parser.add_argument("--robotd", default="/run/robotd.sock")
    parser.add_argument("--tofd", help="tofd's socket (e.g. /run/tofd/tof.sock)")
    parser.add_argument("--media", help="mediad's frame socket (e.g. /run/mediad/media.sock)")
    parser.add_argument(
        "--backend",
        choices=("real", "simulator"),
        default="real",
        help="what robotd drives. real (default): robot.enable, robot.init and robot.poseJoints "
        "are refused, a person turns the duck on. simulator: a simulated robotd only, never a "
        "real duck (the brain's operator may then enable it)",
    )
    parser.add_argument(
        "--updaterd",
        default=telemetry.UPDATERD_SOCKET,
        help="updaterd's socket, which answers Pollen's telemetry switch (a real duck only)",
    )
    parser.add_argument(
        "--keep-pollen-telemetry",
        action="store_true",
        help="leave Pollen's usage data as it is (default: a real duck's bridge turns it off once "
        "at start, the owner's choice of 2026-11-08)",
    )
    parser.add_argument(
        "--allow-open-mediad",
        action="store_true",
        help="serve even while mediad's or netpadd's LAN doors are open (default: a real duck's "
        "bridge refuses to start; close them with tools/duck/close-mediad.sh)",
    )
    parser.add_argument(
        "--lan-address",
        help="the duck's own LAN address to knock on those doors at (default: the --listen "
        "address, or the address of the duck's default route)",
    )
    parser.add_argument(
        "--mediad-ports",
        default=",".join(str(p) for p in doors.MEDIAD_PORTS),
        help="mediad's LAN ports (TCP)",
    )
    parser.add_argument(
        "--netpadd-port", type=int, default=doors.NETPADD_PORT, help="netpadd's LAN port (UDP)"
    )
    return parser


def doors_for(args: argparse.Namespace, log: Callable[..., None]) -> tuple[dict[str, Any], bool]:
    """A real duck's bridge, before it serves: (what it found at mediad's and netpadd's LAN
    doors, whether it may serve). While a door is open it says what, why and the fix on stderr,
    logs ``doors_open`` and may serve only with ``--allow-open-mediad``
    (:mod:`duckmind.remotebody.doors`)."""
    host = args.lan_address
    if not host:
        listen = host_port(args.listen)[0]
        try:
            address = ipaddress.ip_address(listen)
            specific = not (address.is_loopback or address.is_unspecified)
        except ValueError:
            specific = False
        host = listen if specific else doors.own_lan_address()
    ports = [int(p) for p in str(args.mediad_ports).split(",") if p.strip()]
    found = doors.look(host, ports, args.netpadd_port)
    allowed = bool(args.allow_open_mediad)
    report = {**found.to_json(), "allowed": allowed}
    if not found.open:
        log("doors_closed", **report)
        return report, True
    log("doors_open", **report)
    for line in doors.refusal(found, allowed=allowed):
        print(f"remote-body: {line}", file=sys.stderr, flush=True)
    return report, allowed


def body_info_for(
    args: argparse.Namespace, log: Callable[..., None], doors_found: dict[str, Any] | None = None
) -> dict[str, Any]:
    """What the bridge tells the brain about its body. On a real duck this is also where Pollen's
    usage data is settled (:func:`duckmind.remotebody.telemetry.settle`: off unless
    ``--keep-pollen-telemetry``), once per start, and logged; a simulated body has no updaterd.
    ``doors_found``: what :func:`doors_for` found (``check`` on the computer reads it)."""
    if args.backend != "real":
        return {"backend": args.backend}
    info = real_body_info()
    info["telemetry"] = telemetry.settle(
        args.updaterd, turn_off=not args.keep_pollen_telemetry, log=log
    )
    info["doors"] = doors_found
    info["has_dock"] = bool(args.has_dock)
    return info


def main(argv: list[str] | None = None) -> int:
    args = bridge_parser().parse_args(argv)
    log = event_logger(args.events)
    doors_found = None
    if args.backend == "real":
        doors_found, serve = doors_for(args, log)
        if not serve:
            return 2
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
        body_info=body_info_for(args, log, doors_found),
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
