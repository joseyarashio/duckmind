"""A TCP shim that makes one machine look like two on a home Wi-Fi: delay, jitter, loss, cuts.

It sits between the brain's link and the body's bridge (``brain → shim → bridge``) and, per
direction, holds every chunk for ``delay_ms`` plus a jitter sample before passing it on, in order
(TCP never reorders). A *lost* packet is what TCP turns it into: the chunk and everything behind it
wait one retransmission timeout (``rto_ms``; again, doubled, with the same probability). A **cut**
holds everything (both ways, new connections too) like a Wi-Fi drop — TCP keeps it and delivers it
late if the drop ends before anyone gives up; :meth:`NetShim.restore` lets it flow again.
:meth:`NetShim.reset` breaks every connection instead.

``python -m duckmind.remotebody shim --listen 127.0.0.1:0 --to 127.0.0.1:8797 --delay-ms 50``
"""

from __future__ import annotations

import argparse
import heapq
import random
import socket
import sys
import threading
import time
from dataclasses import dataclass

from duckmind.remotebody.link import close_listener

CHUNK = 65536


@dataclass
class Impairment:
    delay_ms: float = 0.0
    jitter_ms: float = 0.0
    loss: float = 0.0
    rto_ms: float = 200.0

    def hold_s(self, rng: random.Random) -> float:
        hold = self.delay_ms + (
            rng.uniform(-self.jitter_ms, self.jitter_ms) if self.jitter_ms else 0
        )
        rto = self.rto_ms
        while self.loss and rng.random() < self.loss:
            hold += rto
            rto *= 2
        return max(0.0, hold) / 1000.0


class _Direction:
    """One direction of one connection: chunks in, each released at its time, in order."""

    def __init__(self, shim: NetShim, src: socket.socket, dst: socket.socket, name: str) -> None:
        self.shim = shim
        self.src = src
        self.dst = dst
        self.name = name
        self._queue: list[tuple[float, int, bytes]] = []
        self._order = 0
        self._last_due = 0.0
        self._wake = threading.Condition()
        self._done = False

    def start(self) -> None:
        threading.Thread(target=self._read, name=f"shim-{self.name}-in", daemon=True).start()
        threading.Thread(target=self._write, name=f"shim-{self.name}-out", daemon=True).start()

    def _read(self) -> None:
        while True:
            try:
                data = self.src.recv(CHUNK)
            except OSError as error:
                self.shim.ended.append(f"{self.name} read: {error!r}")
                data = b""
            if not data:
                self.shim.ended.append(f"{self.name} end of stream")
            now = time.monotonic()
            with self._wake:
                due = max(self._last_due, now + self.shim.impairment.hold_s(self.shim.rng))
                self._last_due = due
                self._order += 1
                heapq.heappush(self._queue, (due, self._order, data))
                self._wake.notify()
            if not data:
                return

    def _write(self) -> None:
        while True:
            with self._wake:
                while True:
                    if self._done:
                        return
                    now = time.monotonic()
                    if self._queue and not self.shim.cut_now and self._queue[0][0] <= now:
                        _, _, data = heapq.heappop(self._queue)
                        break
                    # a cut holds even the chunks whose time has come: sleep on them (a wait of 0 here
                    # looped at full speed and starved every other thread of the process)
                    due = self._queue[0][0] - now if self._queue and not self.shim.cut_now else 0.05
                    self._wake.wait(min(max(0.0, due), 0.05))
            if not data:
                try:
                    self.dst.shutdown(socket.SHUT_WR)
                except OSError:
                    pass
                return
            try:
                self.dst.sendall(data)
            except OSError as error:
                self.shim.ended.append(f"{self.name} write: {error!r}")
                self.stop()
                return

    def stop(self) -> None:
        with self._wake:
            self._done = True
            self._wake.notify_all()


class NetShim:
    def __init__(
        self,
        target: tuple[str, int],
        *,
        listen: tuple[str, int] = ("127.0.0.1", 0),
        impairment: Impairment | None = None,
        seed: int | None = None,
    ) -> None:
        self.target = target
        self.impairment = impairment or Impairment()
        self.rng = random.Random(seed)
        self.cut_now = False
        self.cut_at: float | None = None
        self.ended: list[str] = []  # why each direction of each connection ended
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(listen)
        self._server.listen(16)
        self.port = int(self._server.getsockname()[1])
        self._lock = threading.Lock()
        self._pairs: list[tuple[socket.socket, socket.socket, _Direction, _Direction]] = []
        self._closed = False

    def start(self) -> NetShim:
        threading.Thread(target=self._accept, name="shim-accept", daemon=True).start()
        return self

    def cut(self) -> float:
        """Hold everything, both ways, from now on; returns the monotonic time of the cut."""
        self.cut_now = True
        self.cut_at = time.monotonic()
        return self.cut_at

    def restore(self) -> float:
        self.cut_now = False
        return time.monotonic()

    def reset(self) -> None:
        """Break every connection (an RST, not a silent drop)."""
        with self._lock:
            pairs, self._pairs = self._pairs, []
        for a, b, d1, d2 in pairs:
            d1.stop()
            d2.stop()
            for sock in (a, b):
                try:
                    sock.setsockopt(
                        socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00"
                    )
                except OSError:
                    pass
                try:  # (wakes this side's reader on Linux; sends nothing)
                    sock.shutdown(socket.SHUT_RD)
                except OSError:
                    pass
                sock.close()

    def close(self) -> None:
        self._closed = True
        close_listener(self._server)
        self.reset()

    def _accept(self) -> None:
        while not self._closed:
            try:
                inner, _ = self._server.accept()
            except OSError:
                return
            try:
                outer = socket.create_connection(self.target, timeout=2.0)
            except OSError:
                inner.close()
                continue
            outer.settimeout(None)
            for sock in (inner, outer):
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            up = _Direction(self, inner, outer, "up")
            down = _Direction(self, outer, inner, "down")
            with self._lock:
                self._pairs.append((inner, outer, up, down))
            up.start()
            down.start()


def main(argv: list[str] | None = None) -> int:
    from duckmind.remotebody.bridge import host_port

    parser = argparse.ArgumentParser(prog="python -m duckmind.remotebody shim")
    parser.add_argument("--listen", default="127.0.0.1:0")
    parser.add_argument("--to", required=True, help="the bridge's host:port")
    parser.add_argument("--delay-ms", type=float, default=50.0)
    parser.add_argument("--jitter-ms", type=float, default=10.0)
    parser.add_argument("--loss", type=float, default=0.01)
    parser.add_argument("--rto-ms", type=float, default=200.0)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    shim = NetShim(
        host_port(args.to),
        listen=host_port(args.listen, 0),
        impairment=Impairment(args.delay_ms, args.jitter_ms, args.loss, args.rto_ms),
        seed=args.seed,
    ).start()
    print(f"shim listening on 127.0.0.1:{shim.port} → {args.to}", flush=True)
    print("type: cut | restore | reset | quit", flush=True)
    for line in sys.stdin:
        word = line.strip()
        if word == "cut":
            shim.cut()
        elif word == "restore":
            shim.restore()
        elif word == "reset":
            shim.reset()
        elif word == "quit":
            break
        print(f"{word}: ok {time.time():.3f}", flush=True)
    shim.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
