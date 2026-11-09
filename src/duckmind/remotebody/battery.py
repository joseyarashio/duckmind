"""A remote body's battery as its robotd reports it: ``robot.health`` over the link.

On a body that runs on another machine the brain's battery comes from the duck, not from a
simulated world. robotd answers ``robot.health`` with ``battery: {"volts", "percent"}``
(duck-ipc-proto ``lib.rs`` ``HealthResult.battery`` / ``Battery``): the servos' supply voltage,
smoothed (robotd ``main.rs`` ``BATTERY_EMA_ALPHA``), and robotd's own percentage of it (6.6 V is
0 %, 8.2 V under load is 100 %: duck-control ``model.rs`` ``battery_percent``). There is no fuel
gauge: it sags while the duck walks and recovers at rest. An absent battery means *not known
yet* (the first second after robotd starts), never an empty one.

**Charging.** robotd 0.18.5 has no charging flag (Pollen #470 is still open), so a real duck's
reading says ``charging: None`` (unknown). The simulated body behind the bridge
(:mod:`duckmind.remotebody.simbody`) rewrites its robotd's answer with the world's pack, which
does know (``charging``), so the rehearsal can test the dock; see :func:`world_health_battery`.

:class:`HealthBattery` asks every :data:`POLL_S` on one robotd connection of its own (reconnecting
when the link drops) and keeps the last answer; :meth:`HealthBattery.reading` is that answer while
it is younger than :data:`STALE_S`.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

POLL_S = 2.0
STALE_S = 10.0
HEALTH_TIMEOUT_S = 2.0


def parse(health: Mapping[str, Any]) -> dict[str, Any] | None:
    """The battery in one ``robot.health`` answer: ``{"percent", "volts", "charging", "source"}``,
    or None when robotd has no reading (or a malformed one)."""
    battery = health.get("battery")
    if not isinstance(battery, Mapping):
        return None
    percent = _number(battery.get("percent"))
    if percent is None:
        return None
    charging = battery.get("charging")
    return {
        "percent": max(0.0, min(100.0, percent)),
        "volts": _number(battery.get("volts")),
        "charging": charging if isinstance(charging, bool) else None,
        "source": "robotd",
    }


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _client(path: str | Path) -> Any:
    from duckmind.robotd import RobotdSafetyClient

    return RobotdSafetyClient(path, timeout_s=HEALTH_TIMEOUT_S)


class HealthBattery:
    """``robot.health``'s battery every ``period_s`` from the robotd socket ``socket()`` names."""

    def __init__(
        self,
        socket: Callable[[], str | Path | None],
        *,
        period_s: float = POLL_S,
        stale_s: float = STALE_S,
        clock: Callable[[], float] = time.monotonic,
        connect: Callable[[str | Path], Any] = _client,
    ) -> None:
        self._socket = socket
        self.period_s = period_s
        self.stale_s = stale_s
        self._clock = clock
        self._connect = connect
        self._client: Any = None
        self._last: dict[str, Any] | None = None
        self._at: float | None = None
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="health-battery", daemon=True)

    def start(self) -> HealthBattery:
        self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()
        self._drop()

    def poll_once(self) -> dict[str, Any] | None:
        """Ask robotd once (connecting first when needed); the battery it gave, or None."""
        path = self._socket()
        if path is None:
            return None
        try:
            if self._client is None:
                self._client = self._connect(path)
            health = self._client.health()
        except Exception as error:  # robotd went away (the link dropped): connect again next time
            self.error = f"{type(error).__name__}: {error}"
            self._drop()
            return None
        self.error = None
        battery = parse(health)
        self._last, self._at = battery, self._clock()
        return battery

    def reading(self) -> dict[str, Any] | None:
        """The last battery robotd gave, with its age (``age_s``); None before the first answer,
        when robotd had none, or when it is older than ``stale_s``."""
        last, at = self._last, self._at
        if last is None or at is None:
            return None
        age = self._clock() - at
        if age > self.stale_s:
            return None
        return {**last, "age_s": round(age, 3)}

    def _drop(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    def _run(self) -> None:
        while not self._stop.is_set():
            self.poll_once()
            self._stop.wait(self.period_s)


# ── the simulated body: the world's pack in robotd's answer ─────────────────────────────────


def world_battery(world: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """What a fed robotd would say about the world's pack (:mod:`duckmind.world.battery`): its
    loaded voltage and robotd's percentage of it (``robotd_percent_if_fed``, so it sags as a real
    one does), plus ``charging``, which only the simulated world knows. None: no pack."""
    pack = (world or {}).get("battery")
    if not isinstance(pack, Mapping):
        return None
    volts_mv = _number(pack.get("volts_mv"))
    percent = _number(pack.get("robotd_percent_if_fed"))
    if volts_mv is None or percent is None:
        return None
    return {
        "volts": round(volts_mv / 1000.0, 3),
        "percent": percent,
        "charging": bool(pack.get("charging")),
    }


def world_health_battery(
    world: Callable[[], Mapping[str, Any] | None],
) -> Callable[[bytes], bytes]:
    """For the simulated body's bridge: a rewrite of robotd's answers (one JSON line at a time)
    that puts the world's pack into every ``robot.health`` result. robotd --sim itself reports
    the nominal 7.4 V (the world does not feed its battery to robotd, owner decision E1:
    :mod:`duckmind.world.battery`). Every other line passes unchanged; so does a health answer
    while the world has no pack."""

    def rewrite(line: bytes) -> bytes:
        if b'"healthy"' not in line:  # (cheap: robot.state frames never parse here)
            return line
        try:
            message = json.loads(line)
        except ValueError:
            return line
        result = message.get("result") if isinstance(message, dict) else None
        if not isinstance(result, dict) or "healthy" not in result:
            return line
        try:
            pack = world_battery(world())
        except Exception:
            pack = None
        if pack is None:
            return line
        result["battery"] = pack
        return json.dumps(message).encode() + b"\n"

    return rewrite
