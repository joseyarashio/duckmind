"""The duck on the remote-body link, as the bench recording needs it (:class:`duckmind.bringup.record.Body`).

It dials the bridge with the shared token like a brain does, reads ``robot.state`` (and tofd's ``tof.stream`` when the
bridge offers it) and can send what the body client may send — of which the recording uses a ``sit_toggle``, a forward
``robot.move`` and ``robot.stop``. The body client's own allow-list holds here too: ``robot.enable``, ``robot.init``,
gains, torque and servo commands are never sent (SAFETY.md), and a real duck's bridge refuses them anyway.
"""

from __future__ import annotations

import threading
from collections import deque
from pathlib import Path
from typing import Any

from duckmind.duckbody import protocol as P
from duckmind.duckbody.robotd_client import API_VERSION, RobotdClient
from duckmind.duckbody.server import UnixWire
from duckmind.remotebody.client import RemoteLink


class _Tee:
    """A wire that keeps every ``robot.state`` notification's params as they came."""

    def __init__(self, wire: UnixWire) -> None:
        self.wire = wire
        self.states: deque[dict[str, Any]] = deque(maxlen=500)

    def send(self, obj: Any) -> None:
        self.wire.send(obj)

    def recv(self, timeout_s: float) -> dict[str, Any] | None:
        frame = self.wire.recv(timeout_s)
        if frame is not None and frame.get("method") == "robot.state":
            params = frame.get("params")
            if isinstance(params, dict):
                self.states.append(params)
        return frame

    def close(self) -> None:
        self.wire.close()


class LinkBody:
    def __init__(self, url: str, token: bytes, *, timeout_s: float = 10.0) -> None:
        self.link = RemoteLink(url, token)
        self.link.start(timeout_s=timeout_s)
        self.info: dict[str, Any] = dict(self.link.info)
        sock = self.link.robotd_socket
        self._tee = _Tee(UnixWire(sock))
        self._wires = [UnixWire(sock), self._tee, UnixWire(sock)]
        self.client = RobotdClient(self._wires[0], self._tee, self._wires[2])
        self.client.start(read_model=False)
        self._tof: dict[str, Any] | None = None
        self._tof_seen: dict[str, Any] | None = None
        self._stop = threading.Event()
        self._tof_thread: threading.Thread | None = None
        tofd = self.link.endpoint("tofd")
        if tofd is not None:
            self._tof_thread = threading.Thread(
                target=self._read_tof, args=(tofd,), name="bringup-tof", daemon=True
            )
            self._tof_thread.start()

    @property
    def has_tof(self) -> bool:
        return self._tof_thread is not None

    def read_state(self, timeout_s: float) -> dict[str, Any] | None:
        if not self._tee.states:
            self.client.read_state(timeout_s)
        return self._tee.states.popleft() if self._tee.states else None

    def read_tof(self, timeout_s: float) -> dict[str, Any] | None:
        frame = self._tof
        if frame is None or frame is self._tof_seen:
            return None
        self._tof_seen = frame
        return frame

    def sit_toggle(self) -> None:
        self.client.send(P.Intent.do("sit_toggle"))

    def move(self, vx: float) -> None:
        self.client.send(P.Intent.move(float(vx), 0.0))

    def stop(self) -> None:
        self.client.stop()

    def health(self) -> dict[str, Any]:
        return self.client.health()

    def close(self) -> None:
        self._stop.set()
        for wire in self._wires:
            try:
                wire.close()
            except OSError:
                pass
        self.link.close()

    def _read_tof(self, path: Path) -> None:
        wire: UnixWire | None = None
        try:
            wire = UnixWire(path)
            wire.send(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "hello",
                    "params": {"api_version": API_VERSION},
                }
            )
            wire.recv(2.0)
            wire.send({"jsonrpc": "2.0", "id": 2, "method": "tof.stream", "params": {}})
            while not self._stop.is_set():
                line = wire.recv(0.5)
                if line is not None and line.get("method") == "tof.frame":
                    params = line.get("params")
                    if isinstance(params, dict):
                        self._tof = params
        except (OSError, ConnectionError, ValueError):
            self._tof = None
        finally:
            if wire is not None:
                wire.close()


__all__ = ["LinkBody"]
