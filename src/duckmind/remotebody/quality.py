"""How good the home Wi-Fi is for the remote body: what ``check`` measures from the link's pings.

The brain pings every :data:`~duckmind.remotebody.link.HEARTBEAT_S` and the bridge answers each
ping at once with its own clocks (``Q <brain ns> <body monotonic ns> <body wall ns>``). From the
answers, for every link while it is up:

* the **round trip** (brain → body → brain) of every ping;
* the **longest silence at the duck**: the time between two pings arriving at the bridge, read off
  the bridge's own clock in successive answers. During ``check`` the pings are the only thing the
  brain sends, so this is exactly the silence the bridge's deadman judges (it drops the link, stops
  and sits the duck when it passes the deadman);
* the **clock offset**: the duck's wall clock minus this computer's at the ping's midpoint (± half
  the round trip), to line the bridge's events file up with the computer's and a video. The brain
  itself needs no clock sync: it uses robotd's own ``t_ns``.

A link that drops while the brain is alive and pinging is a **false deadman alarm** (the duck sits
down for nothing): ``check --wifi-minutes 10`` counts them per hour, with the near misses (silences
at the duck over half the deadman).
"""

from __future__ import annotations

import statistics
import threading
from collections import deque
from typing import Any

NEAR_MISS = 0.5  # a silence at the duck over this share of the deadman is a near miss
# The pongs kept (a brain's link lives for days): over an hour at 10 pings a second, so the
# 10-minute Wi-Fi check sees every one; a longer window sees its last KEEP.
KEEP = 40_000


class LinkQuality:
    """The last :data:`KEEP` pongs' round trips and the silences between pings at the duck, across
    reconnects."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.rtt_ms: deque[float] = deque(maxlen=KEEP)
        self.body_gaps_ms: deque[float] = deque(maxlen=KEEP)
        self._counts = [0, 0]  # how many of each were ever added
        self._last_body_ns: int | None = None
        self.wall_offset_ms: float | None = None
        self.wall_offset_err_ms: float | None = None

    def new_link(self) -> None:
        """A fresh link: the silence across a reconnect is the drop itself, not a gap."""
        with self._lock:
            self._last_body_ns = None

    def pong(self, rtt_ns: int, body_ns: int, wall_offset_ns: int | None = None) -> None:
        with self._lock:
            self.rtt_ms.append(rtt_ns / 1e6)
            self._counts[0] += 1
            if self._last_body_ns is not None:
                self.body_gaps_ms.append((body_ns - self._last_body_ns) / 1e6)
                self._counts[1] += 1
            self._last_body_ns = body_ns
            if wall_offset_ns is not None:
                self.wall_offset_ms = wall_offset_ns / 1e6
                self.wall_offset_err_ms = rtt_ns / 2e6

    def mark(self) -> tuple[int, int]:
        """Where a window starts (for :meth:`summary` of one minute)."""
        with self._lock:
            return self._counts[0], self._counts[1]

    def summary(
        self,
        *,
        seconds: float,
        drops: int,
        deadman_ms: float | None,
        since: tuple[int, int] = (0, 0),
    ) -> dict[str, Any]:
        with self._lock:
            rtts = sorted(_since(self.rtt_ms, self._counts[0] - since[0]))
            gaps = _since(self.body_gaps_ms, self._counts[1] - since[1])
        hours = seconds / 3600.0
        near = None if deadman_ms is None else sum(g > NEAR_MISS * deadman_ms for g in gaps)
        return {
            "seconds": round(seconds, 1),
            "pings": len(rtts),
            "rtt_ms_p50": round(statistics.median(rtts), 1) if rtts else None,
            "rtt_ms_p95": round(rtts[int(0.95 * (len(rtts) - 1))], 1) if rtts else None,
            "rtt_ms_max": round(rtts[-1], 1) if rtts else None,
            "duck_silence_ms_max": round(max(gaps)) if gaps else None,
            "drops": drops,
            "false_alarms_per_hour": round(drops / hours, 1) if hours > 0 else None,
            "near_misses": near,
            "near_misses_per_hour": None if near is None or hours <= 0 else round(near / hours, 1),
        }


def _since(kept: deque[float], newest: int) -> list[float]:
    """The ``newest`` items of ``kept`` (all it still holds when it holds fewer)."""
    if newest <= 0:
        return []
    items = list(kept)
    return items[-newest:]
