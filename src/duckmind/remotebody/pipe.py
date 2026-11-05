"""One local socket at either end of a link stream: a reader thread and a bounded writer queue.

A daemon that stops reading (or a brain thread that stalls) must never block the link itself: what
goes to the local socket waits in a queue served by its own thread, and a queue that overflows
closes that one stream instead of holding up the heartbeat.
"""

from __future__ import annotations

import socket
import threading
from collections import deque
from collections.abc import Callable

from duckmind.remotebody.link import CHUNK

QUEUE_MAX_BYTES = 8 * 1024 * 1024


class LocalPipe:
    def __init__(
        self,
        sock: socket.socket,
        *,
        on_data: Callable[[bytes], None],
        on_close: Callable[[str], None],
        name: str,
    ) -> None:
        self.sock = sock
        self._on_data = on_data
        self._on_close = on_close
        self._queue: deque[bytes] = deque()
        self._queued = 0
        self._wake = threading.Condition()
        self._closed = False
        self._name = name

    def start(self) -> None:
        threading.Thread(target=self._read, name=f"{self._name}-read", daemon=True).start()
        threading.Thread(target=self._write, name=f"{self._name}-write", daemon=True).start()

    @property
    def closed(self) -> bool:
        return self._closed

    def put(self, data: bytes) -> None:
        with self._wake:
            if self._closed:
                return
            if self._queued + len(data) > QUEUE_MAX_BYTES:
                overflow = True
            else:
                overflow = False
                self._queue.append(data)
                self._queued += len(data)
                self._wake.notify()
        if overflow:
            self.close("local reader too slow (queue full)")

    def close(self, reason: str = "closed", *, notify: bool = True) -> None:
        with self._wake:
            if self._closed:
                return
            self._closed = True
            self._queue.clear()
            self._wake.notify_all()
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()
        if notify:
            self._on_close(reason)

    def _read(self) -> None:
        while not self._closed:
            try:
                data = self.sock.recv(CHUNK)
            except OSError as error:
                self.close(f"local read: {error}")
                return
            if not data:
                self.close("local end closed")
                return
            try:
                self._on_data(data)
            except Exception as error:  # the link is gone or refused the data
                self.close(f"link: {error}")
                return

    def _write(self) -> None:
        while True:
            with self._wake:
                while not self._queue and not self._closed:
                    self._wake.wait()
                if self._closed:
                    return
                data = self._queue.popleft()
                self._queued -= len(data)
            try:
                self.sock.sendall(data)
            except OSError as error:
                self.close(f"local write: {error}")
                return
