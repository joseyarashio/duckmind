"""The duck's calendar clock: one time for the whole duck.

Real time by default (``speed`` 1.0). ``speed`` N makes the duck's days N times as fast as the wall
clock, which tests use to cross midnights in seconds; the page does not offer it yet.

The clock keeps two readings:

* :meth:`calendar` is the duck's date and time. Growth days count on it: it never jumps.
* :meth:`local` is what the duck lives by this minute: the calendar plus the demo offset that the
  page's time-of-day buttons set (:meth:`set`). With no offset (the live duck, normally) the two
  are the same. The neural duck's circadian drive, the room's lights, the world camera's light and
  the duck's-eye light all read :meth:`local`, so they can never disagree.

The time zone is Duck Core's (Asia/Taipei): its local midnights credit growth days.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

TPE = timezone(timedelta(hours=8), "Asia/Taipei")  # (= duckmind.duckcore.engine.TPE)
DAY_S = 86_400.0


class DuckClock:
    """``start``: the calendar moment at the first reading (default: the real time now, in TPE).
    ``speed``: how many calendar seconds pass per wall second. ``wall``: the wall clock (epoch
    seconds; tests inject their own)."""

    def __init__(
        self,
        start: datetime | None = None,
        *,
        speed: float = 1.0,
        wall: Callable[[], float] = time.time,
    ) -> None:
        if not speed > 0.0:
            raise ValueError("the clock's speed must be above 0")
        self._wall = wall
        self._lock = threading.Lock()
        self._speed = float(speed)
        wall0 = float(wall())
        if start is None:
            start = datetime.fromtimestamp(wall0, TPE)
        elif start.utcoffset() is None:
            raise ValueError("the clock's start must be timezone-aware")
        self._anchor = (wall0, start.astimezone(TPE))
        self._offset_s = 0.0  # the page's time-of-day choice (demo): local = calendar + offset

    # ── reading ──────────────────────────────────────────────────────────────

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def rate(self) -> float:
        """The pace of the duck's time against the wall clock (the page's ``day.rate``)."""
        return self._speed

    def calendar(self) -> datetime:
        """The duck's date and time (TPE, aware). Growth days count on this."""
        with self._lock:
            wall0, cal0 = self._anchor
            speed = self._speed
        return cal0 + timedelta(seconds=(float(self._wall()) - wall0) * speed)

    def local(self) -> datetime:
        """The time the duck lives by right now: the calendar plus the page's demo offset."""
        return self.calendar() + timedelta(seconds=self._offset_s)

    def minute(self) -> float:
        """Minute of the day (0 … 1440) on :meth:`local`."""
        now = self.local()
        return (now.hour * 60 + now.minute + now.second / 60.0 + now.microsecond / 6e7) % 1440.0

    @property
    def offset_s(self) -> float:
        return self._offset_s

    # ── setting ──────────────────────────────────────────────────────────────

    def set(self, minute: float) -> float:
        """The page's time-of-day choice: the duck's day jumps to ``minute`` and runs on from
        there (at the clock's pace). Only the demo offset moves; the calendar (and with it the
        growth days) stays what it is."""
        cal = self.calendar()
        have = cal.hour * 3600 + cal.minute * 60 + cal.second + cal.microsecond / 1e6
        want = (float(minute) % 1440.0) * 60.0
        shift = (want - have) % DAY_S
        if shift > DAY_S / 2:
            shift -= DAY_S
        with self._lock:
            self._offset_s = shift
        return self.minute()

    def clear_offset(self) -> None:
        with self._lock:
            self._offset_s = 0.0

    def set_speed(self, speed: float) -> None:
        """A new pace from now on (the calendar does not jump)."""
        if not speed > 0.0:
            raise ValueError("the clock's speed must be above 0")
        now = self.calendar()
        with self._lock:
            self._anchor = (float(self._wall()), now)
            self._speed = float(speed)

    def jump_to(self, moment: datetime) -> None:
        """Tests: the calendar is ``moment`` from now on."""
        if moment.utcoffset() is None:
            raise ValueError("the clock's moment must be timezone-aware")
        with self._lock:
            self._anchor = (float(self._wall()), moment.astimezone(TPE))

    # ── views ────────────────────────────────────────────────────────────────

    def view(self, words: Callable[[float], Any] | None = None) -> dict[str, Any]:
        minute = self.minute()
        out: dict[str, Any] = {"minute": round(minute, 2), "rate": self._speed}
        if words is not None:
            out["zh"] = words(minute)
        return out
