"""Live voice, LV1: a tiny smoke test of the real connection. **Skipped unless
``DUCKMIND_LIVE_NET=1``**: it reaches Google (the key comes from the Keychain or
``$GEMINI_API_KEY``, read by the connection itself) and uses a few seconds of the free tier.

It also keeps the two API facts the design rests on from going stale: ``gemini-3.8-live`` takes
``AUDIO`` as its only response modality (``TEXT`` is refused) and answers a short run of silence
without an error."""

from __future__ import annotations

import os
import time

import pytest

from duckmind.live import events as E
from duckmind.live import keys as K
from duckmind.live.genai_transport import GenaiTransport
from duckmind.live.tools import tools_config
from duckmind.live.transport import Setup

pytestmark = pytest.mark.skipif(
    os.environ.get("DUCKMIND_LIVE_NET") != "1",
    reason="the real-network smoke test is opt-in: set DUCKMIND_LIVE_NET=1",
)


def wait_for(
    transport: GenaiTransport, kinds: set[str], timeout: float = 20.0
) -> list[E.LiveEvent]:
    seen: list[E.LiveEvent] = []
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        seen += transport.poll(0.0)
        if any(e.kind in kinds for e in seen):
            return seen
        time.sleep(0.02)
    raise AssertionError(f"no {sorted(kinds)} in {timeout} s; got {[e.kind for e in seen]}")


@pytest.fixture
def key_present() -> None:
    if K.load_api_key() is None:
        pytest.skip("no Gemini API key (Keychain 'duckmind-gemini-api-key' or $GEMINI_API_KEY)")


def test_the_real_service_opens_a_session_takes_silence_and_a_status_text(
    key_present: None,
) -> None:
    transport = GenaiTransport()
    transport.open(
        Setup(system_instruction="你是一隻小鴨的腦。不要說話。", tools=tools_config()), 0.0
    )
    try:
        events = wait_for(transport, {E.OPENED, E.CLOSED})
        assert events[0].kind == E.OPENED, events[0].data
        transport.send_text(
            "【小鴨的狀態通知，不需要回應】正在做：坐著看四周", 0.0, turn_complete=False
        )
        for _ in range(50):  # one second of silence, in real time
            transport.send_audio(b"\x00\x00" * 320, 0.0)
            time.sleep(0.02)
        events += transport.poll(0.0)
        assert all(e.kind != E.CLOSED for e in events), [
            e.data for e in events if e.kind == E.CLOSED
        ]
        if not any(e.kind == E.RESUMPTION for e in events):
            events += wait_for(transport, {E.RESUMPTION})
        assert any(e.kind == E.RESUMPTION and e.data["resumable"] for e in events)
    finally:
        transport.close()


def test_the_real_service_still_refuses_text_as_the_only_response_modality(
    key_present: None,
) -> None:
    transport = GenaiTransport()
    transport.open(
        Setup(system_instruction="x", tools=[], response_modalities=("TEXT",)),
        0.0,
    )
    try:
        # (the session opens; the refusal comes with the first turn, as a close 1007)
        wait_for(transport, {E.OPENED, E.CLOSED})
        transport.send_text("你好嗎？", 0.0, turn_complete=True)
        events = wait_for(transport, {E.CLOSED}, timeout=20.0)
        closed = next(e for e in events if e.kind == E.CLOSED)
        assert closed.data["kind"] == E.BAD_REQUEST, closed.data
        assert not K.find_keys(str(closed.data))
    finally:
        transport.close()


def collect(transport: GenaiTransport, until_kind: str, timeout: float = 30.0) -> list[E.LiveEvent]:
    """Events up to and including the first of ``until_kind``."""
    seen: list[E.LiveEvent] = []
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        seen += transport.poll(0.0)
        if any(e.kind == until_kind for e in seen):
            return seen
        time.sleep(0.02)
    raise AssertionError(f"no {until_kind} in {timeout} s; got {[e.kind for e in seen]}")


def test_a_session_resumes_from_its_handle_with_what_it_was_told(key_present: None) -> None:
    """The reconnect the duck relies on: a new connection given the resumption handle carries on
    the old session (and its compression config), not a fresh one."""
    soul = "你是一隻小鴨的腦。家人告訴你的暗號要記住。被問到時，用一句話回答。"
    first = GenaiTransport()
    first.open(Setup(system_instruction=soul, tools=[]), 0.0)
    handle: str | None = None
    try:
        wait_for(first, {E.OPENED})
        first.send_text("暗號是「紫色企鵝」，請記住。", 0.0, turn_complete=True)
        events = collect(first, E.TURN_COMPLETE)
        handles = [
            e.data["handle"] for e in events if e.kind == E.RESUMPTION and e.data.get("handle")
        ]
        deadline = time.monotonic() + 10
        while not handles and time.monotonic() < deadline:
            handles = [
                e.data["handle"]
                for e in first.poll(0.0)
                if e.kind == E.RESUMPTION and e.data.get("handle")
            ]
            time.sleep(0.05)
        assert handles, "the server gave no resumption handle"
        handle = handles[-1]
    finally:
        first.close()
    assert handle
    second = GenaiTransport()
    second.open(Setup(system_instruction=soul, tools=[], resume_handle=handle), 0.0)
    try:
        events = wait_for(second, {E.OPENED, E.CLOSED})
        assert events[0].kind == E.OPENED and events[0].data["resumed"] is True, events[0].data
        second.send_text("暗號是什麼？", 0.0, turn_complete=True)
        answer = "".join(
            e.data["text"] for e in collect(second, E.TURN_COMPLETE) if e.kind == E.OUTPUT_TEXT
        )
        assert "紫色企鵝" in answer or "企鵝" in answer, answer
    finally:
        second.close()
