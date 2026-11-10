"""A rehearsal's hands: the person at the bench, played for ``sim-body`` through its world's visible hand.

``python -m duckmind.remotebody sim-body`` prints its world port in the ``remote-body: ready`` line. With
``--sim-person PORT`` the recording's prompts are answered by the simulated world's hand (``x_hand``): it holds the
duck in the air for the ``air_*`` segments, sets it on the family's mat for the floor segments and the walks, and lifts
and sets it down at the cues. This is the test rig's side (the world's truth, as a person's eyes are): nothing of it
reaches the recording but what the duck's own robotd reports.

The ToF segments are not played (there is no board to hold in the simulated world); a rehearsal keeps the simulated
sensor's own band.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from duckmind.bringup.record import Segment, SegmentFailed

HAND_WAIT_S = 30.0


class SimPerson:
    def __init__(
        self,
        port: int,
        say: Callable[[str], None] = print,
        *,
        pause: Callable[[float], None] = time.sleep,
    ) -> None:
        from duckmind.bodyhost import HostClient

        self.host = HostClient(int(port), timeout_s=5.0)
        self.say = say
        self.pause = pause

    def _hand(self) -> dict[str, Any]:
        hand = self.host.call("x_world").get("hand")
        return hand if isinstance(hand, dict) else {}

    def _holding(self) -> bool:
        hand = self._hand()
        return bool(hand.get("holding") or hand.get("held"))

    def _wait(self, held: bool, what: str) -> None:
        deadline = time.monotonic() + HAND_WAIT_S
        while time.monotonic() < deadline:
            hand = self._hand()
            idle = hand.get("state") in (None, "idle", "parked", "hold", "holding", "held")
            if bool(hand.get("held")) == held and (held or not hand.get("holding")) and idle:
                return
            self.pause(0.1)
        raise SegmentFailed(
            f"the simulated hand did not {what} within {HAND_WAIT_S:.0f} s: {self._hand()}"
        )

    def _pick_up(self) -> None:
        if not self._holding():
            self.host.call("x_hand", do="pickup")
        self._wait(True, "pick the duck up")

    def _put_down(self, to: str) -> None:
        if self._holding() or to != "here":
            self.host.call("x_hand", do="place", to=to)
        self._wait(False, "set the duck down")

    def ready(self, segment: Segment, rep: int) -> None:
        self.say(f"[stage {segment.stage}] {segment.name} #{rep + 1}")
        if segment.tof_mm is not None:
            raise SegmentFailed(
                "the simulated person holds no board: record a rehearsal with --no-tof"
            )
        if segment.name.startswith("air_"):
            self._pick_up()
        else:
            self._put_down("mat")

    def cue(self, segment: Segment, rep: int, name: str) -> None:
        self.say(f"  >>> {name}")
        if name == "lift":
            self.host.call("x_hand", do="pickup")
        elif name == "set_down":
            self.host.call("x_hand", do="place", to="here")

    def note(self, text: str) -> None:
        self.say(f"  {text}")

    def close(self) -> None:
        try:
            if self._holding():
                self._put_down("here")
        except (OSError, RuntimeError, SegmentFailed):
            pass


__all__ = ["SimPerson"]
