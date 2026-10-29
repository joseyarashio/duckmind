"""Helpers of the live-voice tests (not a test file): a tone of a given level and a session on a
virtual clock over a fake transport."""

from __future__ import annotations

import math
from array import array
from typing import Any

from duckmind.live import session as S
from duckmind.live.fake import FakeGemini
from duckmind.live.scenario import ManualClock


def tone(dbfs: float, n: int = 320, phase: int = 0) -> bytes:
    """A 20 ms sine at ``dbfs`` (RMS re full scale): quiet room at -62, speech at -30."""
    amp = 32768.0 * 10.0 ** (dbfs / 20.0) * math.sqrt(2.0)
    return array("h", [int(amp * math.sin(0.2 * (i + phase))) for i in range(n)]).tobytes()


LOUD, QUIET = tone(-30.0), tone(-62.0)


class Rig:
    """A session on a virtual clock over a fake: ``run`` feeds frames for some seconds."""

    def __init__(self, fake: Any = None, **kwargs: Any) -> None:
        self.fake = fake if fake is not None else FakeGemini()
        self.clock = ManualClock()
        self.session = S.LiveSession(self.fake, "SOUL", clock=self.clock, **kwargs)

    def run(
        self, seconds: float, *, pcm: bytes = QUIET, echo: bool = False, until: float | None = None
    ) -> None:
        """Frames up to ``clock + seconds`` (or to ``until``)."""
        end = until if until is not None else self.clock.t + seconds
        while self.clock.t < end - 1e-9:
            self.clock.set(round(self.clock.t + 0.02, 6))
            self.session.feed_pcm(pcm, echo=echo)
            self.session.pump()

    def jump(self, seconds: float) -> None:
        """Advance the clock with only pumps (no audio: the host is not feeding)."""
        end = self.clock.t + seconds
        while self.clock.t < end - 1e-9:
            self.clock.set(round(self.clock.t + 0.1, 6))
            self.session.pump()

    def kinds(self) -> list[str]:
        return [e["kind"] for e in self.session.log]
