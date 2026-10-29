"""Non-blocking UnixWire polls must drain queued JSON frames without waiting."""

from __future__ import annotations

import socket
import threading

from duckmind.duckbody.server import UnixWire


def _wire_pair() -> tuple[UnixWire, socket.socket]:
    client, peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    wire = UnixWire.__new__(UnixWire)
    wire.sock = client
    wire.buffer = bytearray()
    wire.lock = threading.Lock()
    return wire, peer


def test_zero_timeout_drains_multiple_frames_then_returns_when_quiet() -> None:
    wire, peer = _wire_pair()
    try:
        peer.sendall(b'{"policy":"kick_left"}\n{"policy":"walk"}\n')
        assert wire.recv(0.0) == {"policy": "kick_left"}
        assert wire.recv(0.0) == {"policy": "walk"}
        assert wire.recv(0.0) is None
    finally:
        wire.close()
        peer.close()


def test_zero_timeout_preserves_incomplete_frame_until_new_bytes_arrive() -> None:
    wire, peer = _wire_pair()
    try:
        peer.sendall(b'{"policy":')
        assert wire.recv(0.0) is None
        assert wire.buffer == b'{"policy":'
        peer.sendall(b'"stand"}\n')
        assert wire.recv(0.0) == {"policy": "stand"}
    finally:
        wire.close()
        peer.close()
