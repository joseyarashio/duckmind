"""A minimal RFC 6455 WebSocket server side for the remote twist (``/api/neuro/remote/ws``).

Why a WebSocket and not one ``POST`` per twist: the simulator's HTTP server is thread-per-connection
HTTP/1.0 (``http.server``), so every POST pays a fresh TCP connection (one more round trip on a LAN
or a tailnet) and a new thread. A socket opened once carries 20 twists a second at about half a round
trip each, in order, and its close is the person's disconnect (the brain takes the legs back at once).
EventSource is server → page only, so it cannot carry the twist.

Only what the page needs: text frames in (masked, as every browser client sends them), text frames
out, ping/pong, close. No extensions, no fragmentation of our small messages (a fragmented message is
reassembled, a message over :data:`MAX_MESSAGE` closes the socket with 1009).
"""

from __future__ import annotations

import base64
import hashlib
import select
import socket
import struct
import threading
from typing import Any

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
MAX_MESSAGE = 4096
OP_CONT, OP_TEXT, OP_BIN, OP_CLOSE, OP_PING, OP_PONG = 0x0, 0x1, 0x2, 0x8, 0x9, 0xA


class Closed(Exception):
    """The peer closed the socket (or it broke)."""


def accept_key(key: str) -> str:
    return base64.b64encode(hashlib.sha1((key + GUID).encode("ascii")).digest()).decode("ascii")


class Conn:
    """One accepted WebSocket. ``recv`` blocks up to ``timeout`` s for the next text message (None on
    timeout); sends are serialised by a lock (the page's reads and the server's acks never interleave)."""

    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self._buf = bytearray()
        self._send_lock = threading.Lock()
        self.closed = False
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        except OSError:
            pass

    # ── frames out ──

    def _send(self, opcode: int, payload: bytes) -> None:
        size = len(payload)
        if size < 126:
            head = struct.pack("!BB", 0x80 | opcode, size)
        elif size < 65536:
            head = struct.pack("!BBH", 0x80 | opcode, 126, size)
        else:
            head = struct.pack("!BBQ", 0x80 | opcode, 127, size)
        with self._send_lock:
            try:
                self.sock.sendall(head + payload)
            except OSError as error:
                self.closed = True
                raise Closed(str(error)) from error

    def send_text(self, text: str) -> None:
        self._send(OP_TEXT, text.encode("utf-8"))

    def close(self, code: int = 1000) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            with self._send_lock:
                self.sock.sendall(struct.pack("!BBH", 0x80 | OP_CLOSE, 2, code))
        except OSError:
            pass

    # ── frames in ──

    def _fill(self, need: int, timeout: float | None) -> bool:
        """Read until ``need`` bytes are buffered. False on timeout with nothing consumed."""
        while len(self._buf) < need:
            if timeout is not None:
                ready, _, _ = select.select([self.sock], [], [], timeout)
                if not ready:
                    return False
            try:
                chunk = self.sock.recv(4096)
            except (OSError, ValueError) as error:
                self.closed = True
                raise Closed(str(error)) from error
            if not chunk:
                self.closed = True
                raise Closed("eof")
            self._buf += chunk
        return True

    def _frame(self, timeout: float | None) -> tuple[int, bool, bytes] | None:
        if not self._fill(2, timeout):
            return None
        b0, b1 = self._buf[0], self._buf[1]
        fin, opcode = bool(b0 & 0x80), b0 & 0x0F
        masked, length, offset = bool(b1 & 0x80), b1 & 0x7F, 2
        if length == 126:
            self._fill(4, None)
            length, offset = struct.unpack("!H", bytes(self._buf[2:4]))[0], 4
        elif length == 127:
            self._fill(10, None)
            length, offset = struct.unpack("!Q", bytes(self._buf[2:10]))[0], 10
        if length > MAX_MESSAGE:
            self.close(1009)
            raise Closed("message too big")
        if not masked:  # a client must mask (RFC 6455 5.1)
            self.close(1002)
            raise Closed("unmasked frame")
        self._fill(offset + 4 + length, None)
        mask = bytes(self._buf[offset : offset + 4])
        data = bytearray(self._buf[offset + 4 : offset + 4 + length])
        for i in range(length):
            data[i] ^= mask[i & 3]
        del self._buf[: offset + 4 + length]
        return opcode, fin, bytes(data)

    def recv(self, timeout: float | None = 1.0) -> str | None:
        """The next text message; None when none arrived within ``timeout``. Pings are answered,
        a close raises :class:`Closed`."""
        parts: list[bytes] = []
        while True:
            got = self._frame(timeout if not parts else None)
            if got is None:
                return None
            opcode, fin, data = got
            if opcode == OP_PING:
                self._send(OP_PONG, data)
                continue
            if opcode == OP_PONG:
                continue
            if opcode == OP_CLOSE:
                self.close(1000)
                raise Closed("close")
            if opcode in (OP_TEXT, OP_CONT, OP_BIN):
                parts.append(data)
                if fin:
                    return b"".join(parts).decode("utf-8", "replace")
                continue


def handshake(handler: Any) -> Conn | None:
    """Upgrade ``handler``'s request to a WebSocket (101), or answer 400 and return None. The
    handler's connection is the socket from then on; the HTTP loop must not touch it again."""
    headers = handler.headers
    key = headers.get("Sec-WebSocket-Key", "")
    upgrade = headers.get("Upgrade", "").lower()
    if "websocket" not in upgrade or not key or headers.get("Sec-WebSocket-Version", "") != "13":
        body = b'{"ok":false,"error":"a WebSocket upgrade is needed here"}'
        handler.send_response(400)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)
        return None
    handler.send_response(101, "Switching Protocols")
    handler.send_header("Upgrade", "websocket")
    handler.send_header("Connection", "Upgrade")
    handler.send_header("Sec-WebSocket-Accept", accept_key(key))
    handler.end_headers()
    handler.wfile.flush()
    handler.close_connection = True
    return Conn(handler.connection)


__all__ = ["Closed", "Conn", "accept_key", "handshake"]
