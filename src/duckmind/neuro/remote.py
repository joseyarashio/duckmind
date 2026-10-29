"""Remote mode: the family's hand on the legs (docs/design/neural-duck/REMOTE-GOTHERE-2026-10-23.md).

An explicit hand-over, never a drift of the brain's own wishes: while it is on, the twist the legs get
is the person's (arrow keys, WASD, a gamepad stick), and nothing else decides it. Everything below
the twist stays the body link's: a fall and its righting, a pick-up, a low battery, the ToF stop and
the stall stop, the walking head envelope, the guard of every ``robot.move``. The brain keeps stepping
(its head, its voice, its senses and its memory carry on); only what it asks of the legs is dropped.

* **Safe ranges.** :func:`clamp_twist` is the walk policy's own: forward only (0 or 0.35–0.55 m/s: the
  policy does not step below 0.30 and the body link lifts a slower walk to 0.35), no sideways steps,
  no reversing (a reverse is the docking's and the cornered back-arc's, never a drive), turning only
  while stepping — a turn with no forward speed is the slowest arc (0.35 m/s, as the body link makes
  it), an arc is never faster than 0.40 m/s, ``|vyaw| ≤ 1.0``.
* **Deadman.** A twist that has not been renewed for :data:`DEADMAN_S` (300 ms) is zero. The page
  renews it at 20 Hz while a key is down and sends a zero the moment the last key goes up.
* **Lease.** Remote mode itself ends after :data:`LEASE_S` without any message (a closed page, a lost
  connection): the brain takes the legs back, so a duck is never left under a hand that has gone
  (it could not go home to charge). A closed WebSocket ends it at once.
* **Latency.** Each rising edge (the twist going from zero to a walk) is timed: key press → server
  (same-host clocks: the page stamps the key event), server → the tick that sends ``robot.move``,
  and → robotd's ``move.applied``. The last :data:`EDGES_KEEP` are kept for the telemetry.

Pure and clock-injected (``now`` is a monotonic time in seconds); thread-safe (the tick thread and the
web threads share one :class:`RemoteHand`).
"""

from __future__ import annotations

import math
import statistics
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

from duckmind.neuro import contract as C
from duckmind.neuro.bodylink import DEFAULT as LINK
from duckmind.neuro.bodylink import shape_twist

DEADMAN_S = 0.30  # a twist not renewed for this long is zero
LEASE_S = 5.0  # remote mode with no message for this long ends: the brain takes the legs back
EDGES_KEEP = 64
ARC_VYAW = 0.3  # |vyaw| above this is an arc: never faster than ARC_VX_MAX (the motor's own rule)
ARC_VX_MAX = C.APPROACH_VX  # 0.40
APPLIED_MIN = 0.05  # robotd's move.applied forward above this: the legs have taken the twist
HANDBACK_S = 0.6  # after remote mode ends the legs hold zero this long before the brain has them (no fall)

# why remote mode ended (``RemoteHand.ended``)
ENDED = ("off", "lease", "disconnect", "stopped")


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def clamp_twist(vx: Any, vyaw: Any = 0.0, vy: Any = 0.0) -> tuple[float, float]:
    """The nearest twist the walk policy accepts for what a person asked: ``(vx m/s, vyaw rad/s)``.

    ``vy`` is accepted and dropped (the legs do not step sideways); a negative ``vx`` is a stop (no
    reversing); a turn with no forward speed becomes the slowest arc; an arc is at most 0.40 m/s. A message
    that is not a number (NaN, text) is a stop: nothing of it is trusted."""
    del vy
    try:
        if not (math.isfinite(float(vx)) and math.isfinite(float(vyaw))):
            return 0.0, 0.0
    except (TypeError, ValueError):
        return 0.0, 0.0
    vx_in = max(0.0, _finite(vx))
    vyaw_in = max(-LINK.vyaw_max, min(LINK.vyaw_max, _finite(vyaw)))
    out_vx, out_vyaw, _notes = shape_twist(vx_in, vyaw_in, cfg=LINK)
    if out_vx == 0.0:
        return 0.0, 0.0
    if abs(out_vyaw) > ARC_VYAW:
        out_vx = min(out_vx, ARC_VX_MAX)
    return round(out_vx, 4), round(out_vyaw, 4)


@dataclass
class _Edge:
    t_recv: float  # monotonic, when the twist reached the server
    key_to_recv_ms: float | None  # the page's key event → the server (same-host clocks), None unknown
    sent_ms: float | None = None  # → the tick that sent robot.move
    applied_ms: float | None = None  # → robotd's move.applied


class RemoteHand:
    """The remote state (module doc)."""

    def __init__(self, deadman_s: float = DEADMAN_S, lease_s: float = LEASE_S) -> None:
        self.deadman_s = float(deadman_s)
        self.lease_s = float(lease_s)
        self._lock = threading.Lock()
        self.on = False
        self.since: float | None = None
        self.ended: str | None = None  # why it last ended (ENDED); None while on or never on
        self._alive = 0.0  # the last message
        self._handback_until = 0.0  # the legs are held at zero until then (remote mode has just ended)
        self._input_at = -1e9  # the last twist
        self._vx = 0.0
        self._vyaw = 0.0
        self.deadman_stops = 0
        self.lease_ends = 0
        self.turned_on = 0
        self.messages = 0
        self._edge_open: _Edge | None = None  # a rising edge waiting for its sent / applied times
        self._edges: deque[_Edge] = deque(maxlen=EDGES_KEEP)
        self._live_twist: tuple[float, float] = (0.0, 0.0)  # what the last poll handed the legs
        self._owner: object | None = None

    # ── from the web threads ──

    def set_on(self, on: bool, now: float, *, why: str = "off", owner: object | None = None) -> bool:
        """Turn remote mode on or off. Returns whether the state changed. ``owner`` (a connection)
        is remembered when turning on; ``why`` says what ended it."""
        with self._lock:
            if on:
                self._alive = now
                self._input_at = -1e9
                self._vx = self._vyaw = 0.0
                self._owner = owner
                if self.on:
                    return False
                self.on, self.since, self.ended = True, now, None
                self.turned_on += 1
                self._edge_open = None
                return True
            if not self.on:
                return False
            self._end(why if why in ENDED else "off", now)
            return True

    def owned_by(self, owner: object) -> bool:
        with self._lock:
            return self.on and self._owner is owner

    def alive(self, now: float) -> None:
        """Any message from the person's side keeps the lease."""
        with self._lock:
            if self.on:
                self._alive = now
                self.messages += 1

    def push(
        self,
        vx: Any,
        vyaw: Any,
        now: float,
        *,
        vy: Any = 0.0,
        key_to_recv_ms: float | None = None,
    ) -> tuple[float, float]:
        """A twist from the person (``on`` must be true, else nothing is kept). Returns what was
        kept: the clamped twist."""
        twist = clamp_twist(vx, vyaw, vy)
        with self._lock:
            if not self.on:
                return 0.0, 0.0
            self._alive = now
            self.messages += 1
            was_zero = (self._vx, self._vyaw) == (0.0, 0.0) or now - self._input_at > self.deadman_s
            self._vx, self._vyaw, self._input_at = twist[0], twist[1], now
            if twist != (0.0, 0.0) and was_zero and self._edge_open is None:
                self._edge_open = _Edge(now, key_to_recv_ms)
            return twist

    # ── from the tick thread ──

    def poll(self, now: float) -> tuple[float, float] | None:
        """The twist the legs get this tick: None when the brain has the legs; else the person's twist,
        zero once the deadman has run out, and zero for :data:`HANDBACK_S` after remote mode ended (the
        body stops before the brain walks again). Ends remote mode itself when the lease has."""
        with self._lock:
            if not self.on:
                self._live_twist = (0.0, 0.0)
                return (0.0, 0.0) if now < self._handback_until else None
            if now - self._alive > self.lease_s:
                self.lease_ends += 1
                self._end("lease", now)
                self._live_twist = (0.0, 0.0)
                return (0.0, 0.0)
            twist = (self._vx, self._vyaw)
            if twist != (0.0, 0.0) and now - self._input_at > self.deadman_s:
                self._vx = self._vyaw = 0.0
                self.deadman_stops += 1
                twist = (0.0, 0.0)
            self._live_twist = twist
            return twist

    def note_tick(self, now: float, sent_twist: tuple[float, float], applied_vx: float) -> None:
        """What the tick sent and what robotd reports applied, for the edge latencies."""
        with self._lock:
            edge = self._edge_open
            if edge is None:
                return
            if edge.sent_ms is None and sent_twist[0] > 0.0:
                edge.sent_ms = (now - edge.t_recv) * 1000.0
            if edge.sent_ms is not None and applied_vx > APPLIED_MIN:
                edge.applied_ms = (now - edge.t_recv) * 1000.0
                self._edges.append(edge)
                self._edge_open = None
            elif edge.sent_ms is None and (self._vx, self._vyaw) == (0.0, 0.0):
                self._edge_open = None  # released before it was ever sent (a gate held the legs)

    def _end(self, why: str, now: float) -> None:
        self.on, self.ended, self.since = False, why, None
        self._vx = self._vyaw = 0.0
        self._owner = None
        self._edge_open = None
        self._handback_until = now + HANDBACK_S

    # ── for the telemetry ──

    def view(self, now: float | None = None) -> dict[str, Any]:
        with self._lock:
            edges = list(self._edges)
            out: dict[str, Any] = {
                "on": self.on,
                "ended": self.ended,
                "vx": self._live_twist[0],
                "vyaw": self._live_twist[1],
                "deadman_s": self.deadman_s,
                "lease_s": self.lease_s,
                "deadman_stops": self.deadman_stops,
                "lease_ends": self.lease_ends,
                "turned_on": self.turned_on,
            }
            if now is not None and self.on:
                out["idle_ms"] = int(max(0.0, now - self._input_at) * 1000)
        out["latency_ms"] = latency_view(edges)
        return out


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def latency_view(edges: list[_Edge]) -> dict[str, Any]:
    """Medians, 95th percentiles and maxima (ms) of the finished edges."""
    out: dict[str, Any] = {"n": len(edges)}
    for name, values in (
        ("key_to_recv", [e.key_to_recv_ms for e in edges if e.key_to_recv_ms is not None]),
        ("sent", [e.sent_ms for e in edges if e.sent_ms is not None]),
        ("applied", [e.applied_ms for e in edges if e.applied_ms is not None]),
        (
            "key_to_applied",
            [
                e.key_to_recv_ms + e.applied_ms
                for e in edges
                if e.key_to_recv_ms is not None and e.applied_ms is not None
            ],
        ),
    ):
        if values:
            out[name] = {
                "p50": round(statistics.median(values), 1),
                "p95": round(_pct(values, 0.95), 1),
                "max": round(max(values), 1),
            }
    return out


__all__ = [
    "ARC_VX_MAX",
    "DEADMAN_S",
    "LEASE_S",
    "RemoteHand",
    "clamp_twist",
    "latency_view",
]
