"""The remote-body link's wire: one TCP connection between the brain's computer and the body.

Everything the brain says to the body's daemons (robotd, tofd, mediad and, for a simulated body
only, the world server) travels as *streams* multiplexed over this one connection, so the link has
one life: it is up or it is down, and when it goes down every stream goes with it.

**Handshake** (JSON lines, before anything else)::

    body  -> brain  {"protocol": PROTOCOL, "nonce": <hex>}
    brain -> body   {"protocol": PROTOCOL, "nonce": <hex>, "proof": HMAC(token, "brain"|nb|nr)}
    body  -> brain  {"ok": true, "proof": HMAC(token, "body"|nb|nr), "channels": [...], "body": {...}}

Both sides prove they hold the shared token without sending it; neither nonce is reused, so a
recorded handshake cannot be replayed. The session key ``HMAC(token, "session"|nb|nr)`` then signs
every frame.

**Frames** (after the handshake): an ASCII header line and a binary payload::

    <kind> <stream> <seq> <length> <mac>\\n<payload: length bytes>

``seq`` counts the frames in one direction from 1 and must arrive in order (TCP keeps order, so a
gap, a repeat or a reordering is tampering); ``mac`` is the first 16 bytes (hex) of
``HMAC(session, direction|kind|stream|seq|length|payload)``. A frame that fails either check ends
the link. Kinds:

* ``O`` open a stream (brain → body; payload: the channel name); ``C`` close a stream (either way);
* ``D`` data (either way; raw bytes of the daemon's own protocol);
* ``P`` ping (brain → body, every :data:`HEARTBEAT_S`; payload: the brain's monotonic ns);
* ``Q`` pong (body → brain; payload: ``<brain ns> <body monotonic ns> <body wall ns>``) — round
  trip, clock offset and the silences at the body (:mod:`duckmind.remotebody.quality`); a brain
  also takes the older two-field pong.

The link signs frames but does not encrypt them: it is meant for the home LAN, never the internet
(both ends refuse addresses outside loopback, private and link-local ranges).
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import os
import secrets
import socket
import stat
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

PROTOCOL = "duckmind-remote-body/1"
DEFAULT_PORT = 8797
URL_SCHEME = "remote"
# The brain pings this often; the body answers every ping.
HEARTBEAT_S = 0.1
# Silence from the brain this long and the body declares the link lost: robot.stop at once, then
# sit. robotd's own deadman has already zeroed a stale walk 0.5 s after the last robot.move; this
# one is longer because TCP over Wi-Fi stalls for a few hundred ms now and then (a lost packet
# waits a 200 ms retransmission, two in a row 600 ms: at 1 % loss the rehearsal saw 629 ms gaps in
# 90 s), and every false alarm sits the duck down.
DEADMAN_S = 1.0
DEADMAN_MIN_S, DEADMAN_MAX_S = 0.2, 2.0
# Silence from the body this long and the brain drops the link and dials again.
LOST_S = 1.0
HANDSHAKE_TIMEOUT_S = 3.0
TOKEN_MIN_HEX = 32  # 128 bits
MAX_HEADER = 128
MAX_PAYLOAD = 4 * 1024 * 1024
CHUNK = 65536
MAC_BYTES = 16
CHANNELS = ("robotd", "tofd", "media", "world")
KINDS = frozenset(b"OCDPQ")


class LinkError(RuntimeError):
    """The link broke or was tampered with: drop it."""


class AuthError(LinkError):
    """The other side does not hold the token (or refused ours)."""


# ── addresses ────────────────────────────────────────────────────────────


def lan_address(host: str) -> bool:
    """Loopback, private (RFC 1918 / ULA) or link-local: the home LAN, never the internet."""
    try:
        address = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return bool(address.is_loopback or address.is_private or address.is_link_local) and not (
        address.is_unspecified or address.is_multicast
    )


@dataclass(frozen=True)
class BodyUrl:
    host: str
    port: int

    def __str__(self) -> str:
        return f"{URL_SCHEME}://{self.host}:{self.port}"


def parse_url(url: str) -> BodyUrl:
    """``remote://host[:port]`` → the body's address. A name is resolved here and the address it
    resolves to must be on the LAN (:func:`lan_address`)."""
    parts = urlsplit(url)
    if parts.scheme != URL_SCHEME or not parts.hostname:
        raise ValueError(f"a remote body is remote://host:port, not {url!r}")
    if parts.path not in ("", "/") or parts.query or parts.username or parts.password:
        raise ValueError(
            "remote://host:port takes no path, query or credentials (the token is a file)"
        )
    port = parts.port or DEFAULT_PORT
    host = parts.hostname
    try:
        resolved = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as error:
        raise ValueError(f"cannot resolve {host!r}: {error}") from error
    addresses = {info[4][0] for info in resolved}
    outside = sorted(a for a in addresses if not lan_address(str(a)))
    if outside:
        raise ValueError(f"{host!r} resolves outside the home LAN ({', '.join(outside)}); refused")
    return BodyUrl(host, port)


# ── the token ────────────────────────────────────────────────────────────


def new_token(path: Path) -> str:
    """Write a fresh 256-bit token to ``path`` (mode 0600, refusing to overwrite)."""
    token = secrets.token_hex(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(token + "\n")
    return token


def read_token(path: Path) -> bytes:
    """The shared token from ``path``: at least :data:`TOKEN_MIN_HEX` hex characters, and the file
    readable by its owner only (a token others can read authenticates nobody)."""
    info = path.stat()
    if info.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise PermissionError(f"{path} is readable by others; chmod 600 it")
    text = path.read_text(encoding="ascii").strip()
    if len(text) < TOKEN_MIN_HEX or any(c not in "0123456789abcdefABCDEF" for c in text):
        raise ValueError(f"{path} must hold at least {TOKEN_MIN_HEX} hex characters")
    return text.lower().encode("ascii")


def same_proof(expected: str, given: object) -> bool:
    """A proof check that takes the same time however much of it matches."""
    return isinstance(given, str) and hmac.compare_digest(expected.encode(), given.encode())


def proof(token: bytes, role: bytes, body_nonce: str, brain_nonce: str) -> str:
    return hmac.new(
        token, role + b"|" + body_nonce.encode() + b"|" + brain_nonce.encode(), hashlib.sha256
    ).hexdigest()


def session_key(token: bytes, body_nonce: str, brain_nonce: str) -> bytes:
    return hmac.new(
        token, b"session|" + body_nonce.encode() + b"|" + brain_nonce.encode(), hashlib.sha256
    ).digest()


# ── framing ──────────────────────────────────────────────────────────────


def read_json_line(sock: socket.socket, limit: int = 4096) -> dict[str, Any]:
    """One handshake line (byte by byte: nothing after it may be consumed here)."""
    data = bytearray()
    while True:
        byte = sock.recv(1)
        if not byte:
            raise LinkError("closed during the handshake")
        if byte == b"\n":
            break
        data += byte
        if len(data) > limit:
            raise LinkError("handshake line too long")
    try:
        value = json.loads(bytes(data))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LinkError("handshake line is not JSON") from error
    if not isinstance(value, dict):
        raise LinkError("handshake line is not an object")
    return value


def send_json_line(sock: socket.socket, value: dict[str, Any]) -> None:
    sock.sendall(json.dumps(value, separators=(",", ":")).encode() + b"\n")


def close_listener(server: socket.socket) -> None:
    """Close a listening socket so that a thread blocked in its ``accept()`` returns: Linux (the duck's
    own OS) does not wake that thread on ``close()`` alone, only on ``shutdown()`` first; macOS wakes it
    either way and refuses the shutdown of a socket that is not connected, which is fine."""
    try:
        server.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    try:
        server.close()
    except OSError:
        pass


class FrameCodec:
    """Signs and checks frames for one side of one session. ``outgoing``/``incoming``: the
    direction bytes (b"b" brain → body, b"B" body → brain)."""

    def __init__(self, key: bytes, outgoing: bytes, incoming: bytes) -> None:
        self._key = key
        self._out = outgoing
        self._in = incoming
        self._send_seq = 0
        self._recv_seq = 0
        self._send_lock = threading.Lock()

    def _mac(self, direction: bytes, kind: int, stream: int, seq: int, payload: bytes) -> str:
        head = b"%c%c|%d|%d|%d|" % (direction[0], kind, stream, seq, len(payload))
        return hmac.new(self._key, head + payload, hashlib.sha256).digest()[:MAC_BYTES].hex()

    def send(self, sock: socket.socket, kind: bytes, stream: int, payload: bytes = b"") -> None:
        if len(payload) > MAX_PAYLOAD:
            raise LinkError("frame payload too large")
        code = kind[0]
        with self._send_lock:  # (seq order on the wire is the order they were signed in)
            self._send_seq += 1
            seq = self._send_seq
            mac = self._mac(self._out, code, stream, seq, payload)
            header = b"%c %d %d %d %s\n" % (code, stream, seq, len(payload), mac.encode())
            sock.sendall(header + payload)

    def recv(self, reader: FrameReader) -> tuple[bytes, int, bytes]:
        header = reader.line(MAX_HEADER)
        fields = header.split(b" ")
        if len(fields) != 5 or len(fields[0]) != 1 or fields[0][0] not in KINDS:
            raise LinkError("malformed frame header")
        try:
            stream, seq, length = int(fields[1]), int(fields[2]), int(fields[3])
        except ValueError as error:
            raise LinkError("malformed frame header") from error
        if stream < 0 or length < 0 or length > MAX_PAYLOAD:
            raise LinkError("frame out of bounds")
        payload = reader.exact(length)
        if seq != self._recv_seq + 1:
            raise LinkError(f"frame out of sequence ({seq} after {self._recv_seq})")
        expected = self._mac(self._in, fields[0][0], stream, seq, payload)
        if not hmac.compare_digest(expected.encode(), fields[4]):
            raise LinkError("frame signature does not match")
        self._recv_seq = seq
        return fields[0], stream, payload


class FrameReader:
    """Buffered reads of header lines and exact payloads from one socket."""

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._buffer = bytearray()

    def _fill(self) -> None:
        data = self._sock.recv(CHUNK)
        if not data:
            raise LinkError("the link closed")
        self._buffer += data

    def line(self, limit: int) -> bytes:
        while True:
            end = self._buffer.find(b"\n")
            if end >= 0:
                line = bytes(self._buffer[:end])
                del self._buffer[: end + 1]
                return line
            if len(self._buffer) > limit:
                raise LinkError("frame header too long")
            self._fill()

    def exact(self, size: int) -> bytes:
        while len(self._buffer) < size:
            self._fill()
        out = bytes(self._buffer[:size])
        del self._buffer[:size]
        return out


class LineSplitter:
    """Whole lines out of a byte stream (the brain's requests are JSON lines)."""

    def __init__(self, limit: int) -> None:
        self._buffer = bytearray()
        self._limit = limit

    def feed(self, data: bytes) -> list[bytes]:
        self._buffer += data
        lines: list[bytes] = []
        while True:
            end = self._buffer.find(b"\n")
            if end < 0:
                break
            lines.append(bytes(self._buffer[: end + 1]))
            del self._buffer[: end + 1]
        if len(self._buffer) > self._limit:
            raise LinkError("request line too long")
        return lines
