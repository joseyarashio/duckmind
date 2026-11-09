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
FINISH_TIMEOUT_S = 5.0


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
        self._finishing: str | None = None  # the far end closed the stream: drain, then close
        self._name = name

    def start(self) -> None:
        threading.Thread(target=self._read, name=f"{self._name}-read", daemon=True).start()
        threading.Thread(target=self._write, name=f"{self._name}-write", daemon=True).start()

    @property
    def closed(self) -> bool:
        return self._closed

    def put(self, data: bytes) -> None:
        with self._wake:
            if self._closed or self._finishing is not None:
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

    def finish(self, reason: str = "closed") -> None:
        """The far end closed this stream after its last data (a ``C`` frame): write what is still
        queued — mediad closes a frame's connection right after the frame, so the frame's tail is
        usually still here — then close, without telling the far end (it closed first). The
        brain's end only: on the duck a closed stream's queued requests are dropped (:meth:`close`),
        as is a lost link's backlog, so nothing reaches robotd late. A local end that stops reading
        meanwhile is closed after :data:`FINISH_TIMEOUT_S`."""
        with self._wake:
            if self._closed or self._finishing is not None:
                return
            self._finishing = reason
            self._wake.notify_all()
        try:
            self.sock.settimeout(FINISH_TIMEOUT_S)
        except OSError:
            pass

    def close(self, reason: str = "closed", *, notify: bool = True) -> None:
        with self._wake:
            if self._closed:
                return
            self._closed = True
            notify = notify and self._finishing is None
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
                while not self._queue and not self._closed and self._finishing is None:
                    self._wake.wait()
                if self._closed:
                    return
                if not self._queue:  # (finishing, and all of it written)
                    finished = self._finishing or "closed"
                    break
                data = self._queue.popleft()
                self._queued -= len(data)
            try:
                self.sock.sendall(data)
            except OSError as error:
                self.close(f"local write: {error}")
                return
        self.close(finished, notify=False)
