"""Bug 1 (2026-10-09): the cloud's errors (1011 "internal error", 1007, 1006, a broken pipe) must never
leak as "Task exception was never retrieved", never take the simulator down, be logged once per
burst, be told on the talk panel in plain words, and be retried with backoff while the duck lives on.

The old :8791 log had 54 ``received 1011 (internal error)`` from Google and 45 "Task exception was
never retrieved" from ``GenaiTransport._sender`` / ``_receiver``: when the service closes the line both
tasks fail in the same breath and ``_main`` read only the first one's outcome."""

from __future__ import annotations

import asyncio
import gc
import logging
import threading
import time
from types import SimpleNamespace as NS
from typing import Any

import pytest

from duckmind.live import events as E
from duckmind.live import genai_transport as G
from duckmind.live import session as S
from duckmind.live import tools as T
from duckmind.live.scenario import ManualClock
from duckmind.live.transport import SentLog, Setup

pytestmark = pytest.mark.usefixtures("no_network")


class Closed(Exception):
    """A websockets ConnectionClosed: ``rcvd.code``."""

    def __init__(self, code: int, reason: str) -> None:
        super().__init__(f"received {code} (internal error) {reason}; then sent {code}")
        self.rcvd = NS(code=code, reason=reason)


def setup() -> Setup:
    return Setup(system_instruction="SOUL", tools=T.tools_config())


class DyingSession:
    """A session whose line is closed by the service: every send and the receive raise."""

    def __init__(self, error: BaseException) -> None:
        self.error = error

    async def send_realtime_input(self, **kw: Any) -> None:
        raise self.error

    async def send_client_content(self, **kw: Any) -> None:
        raise self.error

    async def send_tool_response(self, **kw: Any) -> None:
        raise self.error

    async def receive(self) -> Any:
        raise self.error
        yield  # pragma: no cover  (makes this an async generator)


class DyingClient:
    """``client.aio.live.connect(...)``: opens, then the service closes with ``code`` at once."""

    def __init__(self, code: int = 1011, reason: str = "Internal error encountered.") -> None:
        self.code, self.reason = code, reason
        self.opens = 0
        self.aio = NS(live=NS(connect=self._connect))

    def _connect(self, *, model: str, config: dict[str, Any]) -> Any:
        client = self

        class Manager:
            async def __aenter__(self_inner) -> DyingSession:
                client.opens += 1
                return DyingSession(Closed(client.code, client.reason))

            async def __aexit__(self_inner, *exc: Any) -> None:
                return None

        return Manager()


def drive(transport: G.GenaiTransport, until: Any, timeout: float = 5.0) -> list[E.LiveEvent]:
    events: list[E.LiveEvent] = []
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        events += transport.poll(0.0)
        if until(events):
            return events
        time.sleep(0.005)
    raise AssertionError(f"timed out: {[e.kind for e in events]}")


def test_sender_and_receiver_failing_together_leave_no_task_exception_never_retrieved(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Reproduction of the 45 leaked tasks: audio waits in the queue, the service closes the line,
    the sender and the receiver both raise 1011. One CLOSED event, nothing printed by asyncio."""
    caplog.set_level(logging.DEBUG)
    client = DyingClient()
    transport = G.GenaiTransport(client_factory=lambda key: client, key_loader=lambda: "k" * 8)
    for _ in range(5):
        transport.open(setup(), 0.0)
        transport.send_audio(b"\x00" * 640, 0.0)  # (waits for the loop: both tasks fail at once)
        events = drive(transport, lambda ev: any(e.kind == E.CLOSED for e in ev))
        closed = [e for e in events if e.kind == E.CLOSED]
        assert closed[0].data["kind"] == E.SERVER and closed[0].data["code"] == 1011
        transport.close()
    for _ in range(3):
        gc.collect()
    leaked = [r for r in caplog.records if "never retrieved" in r.getMessage()]
    assert leaked == [], [r.getMessage()[:120] for r in leaked]
    assert transport.loop_errors == 0


def test_a_task_nobody_awaits_is_counted_and_written_once_not_printed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The loop's own error handler is ours: a stray failing task cannot print a traceback."""
    transport = G.GenaiTransport(key_loader=lambda: "k")
    seen: list[str] = []
    loop_ctx = {"message": "boom", "exception": ValueError("stray")}
    with caplog.at_level(logging.WARNING, logger="duckmind.live.genai"):
        for _ in range(3):
            transport._loop_error(asyncio.new_event_loop(), loop_ctx)
    seen = [r.getMessage() for r in caplog.records if r.name == "duckmind.live.genai"]
    assert transport.loop_errors == 3 and len(seen) == 1 and "stray" in seen[0]


class StormTransport:
    """A transport on the session's virtual clock: ``open`` succeeds, then the service closes the
    line (code 1011) ``lives`` seconds later; ``break_calls`` makes the send methods and ``poll``
    raise as well. ``healthy_after`` virtual seconds: from then on a connection simply stays up."""

    name = "storm"

    def __init__(
        self,
        clock: ManualClock,
        *,
        lives: float = 0.4,
        healthy_after: float = 1e9,
        break_calls: bool,
    ) -> None:
        self.clock = clock
        self.lives, self.healthy_after, self.break_calls = lives, healthy_after, break_calls
        self.sent = SentLog()
        self.opens = 0
        self._queue: list[E.LiveEvent] = []
        self._closes_at: float | None = None

    def open(self, setup: Setup, now: float) -> None:
        self.opens += 1
        self._queue.append(E.LiveEvent(E.OPENED, now, {"resumed": False}))
        self._closes_at = None if now >= self.healthy_after else now + self.lives

    def _maybe_raise(self) -> None:
        if self.break_calls and self.clock.t < self.healthy_after:
            raise Closed(1011, "Internal error encountered.")

    def send_audio(self, pcm: bytes, now: float) -> None:
        self._maybe_raise()

    def send_audio_end(self, now: float) -> None:
        self._maybe_raise()

    def send_image(self, jpeg: bytes, now: float, mime: str = "image/jpeg") -> None:
        self._maybe_raise()

    def send_text(self, text: str, now: float, *, turn_complete: bool = False) -> None:
        self._maybe_raise()

    def send_tool_response(
        self, call_id: str, name: str, response: dict[str, Any], scheduling: str, now: float
    ) -> None:
        self._maybe_raise()

    def poll(self, now: float) -> list[E.LiveEvent]:
        if self.break_calls and self.clock.t < self.healthy_after and now % 3 < 0.5:
            raise Closed(1011, "Internal error encountered.")
        out, self._queue = self._queue, []
        if self._closes_at is not None and now >= self._closes_at:
            self._closes_at = None
            out.append(
                E.LiveEvent(
                    E.CLOSED,
                    now,
                    {"kind": E.SERVER, "code": 1011, "reason": "received 1011 (internal error)"},
                )
            )
        return out

    def close(self) -> None:
        self._closes_at = None


def pump_for(session: S.LiveSession, clock: ManualClock, seconds: float, pcm: bytes) -> float:
    """Frames and pumps for ``seconds`` of virtual time; the slowest single frame (real seconds)."""
    worst = 0.0
    end = clock.t + seconds
    while clock.t < end - 1e-9:
        clock.set(round(clock.t + 0.02, 6))
        t0 = time.perf_counter()
        session.feed_pcm(pcm)
        session.pump()
        worst = max(worst, time.perf_counter() - t0)
    return worst


@pytest.mark.parametrize("break_calls", [False, True])
def test_a_storm_of_1011_is_retried_with_backoff_logged_once_and_told_in_words(
    break_calls: bool, caplog: pytest.LogCaptureFixture
) -> None:
    """Ten virtual minutes of the service closing every connection with 1011 (and, in one case, every
    send and poll raising too): the session never raises, goes down but keeps trying with backoff,
    writes one line for the whole burst, and says what is wrong in the catalog's words; when the
    service recovers it comes back online and a later burst is logged again."""
    from tests.live_rig import QUIET  # noqa: PLC0415

    clock = ManualClock()
    transport = StormTransport(clock, break_calls=break_calls, healthy_after=600.0)
    session = S.LiveSession(transport, "SOUL", clock=clock)
    session.set_enabled(True)
    with caplog.at_level(logging.WARNING, logger="duckmind.live.session"):
        worst = pump_for(session, clock, 590.0, QUIET)
        status = session.status()
        burst_lines = [r for r in caplog.records if r.name == "duckmind.live.session"]
        assert status["state"] in (S.RECONNECTING, S.DOWN)
        assert status["state"] == S.DOWN  # (3+ failures in a row)
        assert status["failures_in_a_row"] >= 10
        # backoff: 590 s at a 1.5/3/4.5/6/7.5/8 s ladder tops out near 8 s a try
        assert 40 <= transport.opens <= 120, transport.opens
        assert len(burst_lines) == 1, [r.getMessage() for r in burst_lines]
        if not break_calls:
            assert "1011" in burst_lines[0].getMessage()
        words = str(status["error_words"])
        assert words and "{" not in words
        assert str(status["failures_in_a_row"]) in words
        assert ("內部錯誤" in words) is (not break_calls)  # (a raising poll is told as a lost line)
        assert worst < 0.05  # (a frame is never held up by the storm)
        # the service gets well: the next connection stays up
        pump_for(session, clock, 60.0, QUIET)
        assert session.status()["state"] == S.ONLINE
        assert session.status()["error_words"] is None
        # ...and 30 s later it drops again: a new burst, a second line
        transport.healthy_after = 1e9
        transport._closes_at = clock.t + 0.1
        pump_for(session, clock, 40.0, QUIET)
        assert len([r for r in caplog.records if r.name == "duckmind.live.session"]) == 2


def test_a_connection_that_opens_and_drops_at_once_keeps_its_backoff() -> None:
    """The first version reset the count on every OPENED: a service that accepts and then closes
    with 1011 was retried every 1.5 s for ever and never reported as down."""
    clock = ManualClock()
    transport = StormTransport(clock, lives=0.2, break_calls=False)
    session = S.LiveSession(transport, "SOUL", clock=clock)
    session.set_enabled(True)
    pump_for(session, clock, 120.0, b"\x00" * 640)
    assert session.status()["state"] == S.DOWN
    assert transport.opens < 20  # (not one every 1.5 s = 80)


def test_the_panel_says_what_is_wrong_in_all_six_languages() -> None:
    from duckmind.i18n import msg  # noqa: PLC0415
    from duckmind.i18n.catalog import read_catalog  # noqa: PLC0415

    for kind in ("server", "network", "bad_request", "unknown"):
        key = f"neuro.live.err.{kind}"
        assert "{" not in str(msg(key, n=3, s=5))
        for lang in ("zh-Hant", "en", "ja", "es", "zh-Hans", "ko"):
            text = (read_catalog(lang, "neuro") or {})[key]
            assert text and "{n" in text and "{s" in text, (lang, key)


def test_a_threading_sanity_the_session_pump_is_safe_from_two_threads() -> None:
    """The page's status reads while the microphone thread pumps: no deadlock under a storm."""
    clock = ManualClock()
    transport = StormTransport(clock, break_calls=True)
    session = S.LiveSession(transport, "SOUL", clock=clock)
    session.set_enabled(True)
    stop = threading.Event()
    reads: list[dict[str, Any]] = []

    def reader() -> None:
        while not stop.is_set():
            reads.append(session.status())
            time.sleep(0.001)

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        pump_for(session, clock, 60.0, b"\x00" * 640)
    finally:
        stop.set()
        thread.join(5)
    assert reads and not thread.is_alive()
