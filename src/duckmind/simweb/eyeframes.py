"""The duck-eye picture as a source of frames for the live connection (live voice, LV0).

Later, the duck's eye goes into the same Gemini Live connection as its microphone: robot-bridge
pushes a JPEG about once a second (``LiveSession.send_frame`` with ``reserve_frame``). This is the
simulator's side of that, with **no network in it**: it takes the pictures the simulator already
makes (:class:`~duckmind.simweb.eyecam.EyeCam`'s render thread, or the world camera's
:class:`~duckmind.simweb.live.WindowEye`; both answer ``latest()`` with the newest JPEG and say
when it was made) and hands them out **at most once a second, each picture once**.

* :meth:`EyeFrames.poll` — the next frame if one is due and new, else None;
* :meth:`EyeFrames.frames` — the same as a blocking generator that sleeps between frames;
* :meth:`EyeFrames.from_eye` / :meth:`EyeFrames.from_bridge` — the sources: an eye with
  ``latest()`` (a window-eye is "touched" first: the world's camera only renders while somebody
  asks, owner decision F4), or ``LiveBridge.eye_image`` (JPEG, tag, why-not).

The picture is the duck's own sensor reading (computed, not a camera recording: spec §4.8).
Nothing is written to disk.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

MAX_HZ = 1.0  # the most frames a second the live connection takes
MAX_JPEG_BYTES = 4_000_000  # a sanity bound (the page's picture is 20–60 kB)
SOI, EOI = b"\xff\xd8", b"\xff\xd9"
MIME = "image/jpeg"

Fetch = Callable[[], "tuple[bytes | None, str, str]"]


@dataclass(frozen=True)
class EyeFrame:
    """One picture for the connection."""

    seq: int  # 1, 2, 3 … as handed out
    jpeg: bytes
    mime: str
    tag: str  # which picture it is (the source's own tag, else a hash)
    wall: float  # when the simulator made it (epoch s), or when it was fetched
    width: int | None = None
    height: int | None = None


class EyeFrames:
    """Frames of the duck's eye, at most :data:`MAX_HZ` a second, each picture once.

    ``fetch`` returns ``(jpeg or None, tag, why)`` (the page's own :meth:`LiveBridge.eye_image`
    shape); ``clock`` is a monotonic clock in seconds (tests move it). A picture is *new* when its
    tag differs from the last one handed out (a source without tags: when its bytes do)."""

    def __init__(
        self,
        fetch: Fetch,
        *,
        max_hz: float = MAX_HZ,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
    ) -> None:
        self._fetch = fetch
        self.period_s = 1.0 / min(float(max_hz), MAX_HZ)
        self._clock = clock
        self._wall = wall
        self._lock = threading.Lock()
        self._next_at = -1e9
        self._last_tag: str | None = None
        self.sent = 0
        self.same = 0  # polls that found nothing new
        self.bad = 0  # pictures that were not a JPEG (or too big) and were dropped
        self.last_why = ""  # why there was no picture, in the source's words
        self.meta: dict[str, Any] = {}  # what the source says of the latest picture (its size)

    # ── sources ───────────────────────────────────────────────────────────────

    @classmethod
    def from_eye(cls, eye: Any, **kwargs: Any) -> EyeFrames:
        """An eye with ``latest()`` (EyeCam, WindowEye): the dict holds ``jpeg``, ``wall``, and
        ``frames`` (a count), ``width``/``height`` when it knows them."""
        meta: dict[str, Any] = {}

        def fetch() -> tuple[bytes | None, str, str]:
            touch = getattr(eye, "touch", None)
            if callable(touch):
                touch()  # (a window-eye renders only while somebody keeps asking)
            latest = eye.latest()
            if not latest or not latest.get("jpeg"):
                status = getattr(eye, "status", None)
                why = ""
                if callable(status):
                    info = status() or {}
                    why = str(info.get("error") or info.get("note") or "")
                return None, "", why
            meta.clear()
            meta.update(width=latest.get("width"), height=latest.get("height"))
            tag = f"{int(float(latest.get('wall') or 0) * 1000)}-{int(latest.get('frames') or 0)}"
            return bytes(latest["jpeg"]), tag, ""

        out = cls(fetch, **kwargs)
        out.meta = meta
        return out

    @classmethod
    def from_bridge(cls, bridge: Any, **kwargs: Any) -> EyeFrames:
        """A live bridge (``eye_image() -> (jpeg or None, tag, why)``)."""
        return cls(bridge.eye_image, **kwargs)

    # ── frames ────────────────────────────────────────────────────────────────

    def poll(self) -> EyeFrame | None:
        """The next frame, or None: not due yet, no picture, nothing new, or not a JPEG."""
        with self._lock:
            now = self._clock()
            if now < self._next_at:
                return None
            jpeg, tag, why = self._fetch()
            self.last_why = why or ""
            if not jpeg:
                return None
            if not (jpeg[:2] == SOI and jpeg[-2:] == EOI) or len(jpeg) > MAX_JPEG_BYTES:
                self.bad += 1
                return None
            tag = tag or hashlib.sha1(jpeg).hexdigest()[:16]
            if tag == self._last_tag:
                self.same += 1
                return None
            self._last_tag = tag
            self._next_at = now + self.period_s
            self.sent += 1
            meta = self.meta
            return EyeFrame(
                seq=self.sent,
                jpeg=bytes(jpeg),
                mime=MIME,
                tag=tag,
                wall=self._wall(),
                width=meta.get("width"),
                height=meta.get("height"),
            )

    def frames(
        self,
        stop: threading.Event | None = None,
        *,
        sleep: Callable[[float], Any] | None = None,
        poll_s: float = 0.1,
    ) -> Iterator[EyeFrame]:
        """Frames as they come, until ``stop`` is set (blocking: a thread's loop)."""
        stop = stop or threading.Event()
        wait = sleep or stop.wait
        while not stop.is_set():
            frame = self.poll()
            if frame is not None:
                yield frame
            else:
                wait(poll_s)

    def status(self) -> dict[str, Any]:
        return {
            "max_hz": 1.0 / self.period_s,
            "sent": self.sent,
            "same": self.same,
            "bad": self.bad,
            "why": self.last_why,
        }


__all__ = ["MAX_HZ", "MIME", "EyeFrame", "EyeFrames"]
