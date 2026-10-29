"""Live voice, LV1: the real connection (``duckmind.live.genai_transport``) driven with a fake
client: no network, no key, no SDK needed (a few tests use the SDK's own types when it is
installed). What is checked: the config the SDK is given, what each ``send_*`` becomes, what each
server message becomes (shapes taken from the real service), how failures are classified and that
no message that leaves the module holds a key."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from types import SimpleNamespace as NS
from typing import Any

import pytest

from duckmind.live import events as E
from duckmind.live import genai_transport as G
from duckmind.live import keys as K
from duckmind.live import tools as T
from duckmind.live.transport import Setup

pytestmark = pytest.mark.usefixtures("no_network")

FAKE_KEY = "AQ." + "Zq7_-" * 9


def setup(**kw: Any) -> Setup:
    return Setup(system_instruction="SOUL", tools=T.tools_config(), **kw)


class FakeSession:
    def __init__(self, delay: float = 0.0) -> None:
        self.sent: list[tuple[str, dict[str, Any]]] = []
        self.inbox: asyncio.Queue[Any] | None = None
        self.delay = delay

    async def send_realtime_input(self, **kw: Any) -> None:
        if self.delay:
            await asyncio.sleep(self.delay)
        self.sent.append(("realtime", kw))

    async def send_client_content(self, **kw: Any) -> None:
        self.sent.append(("content", kw))

    async def send_tool_response(self, **kw: Any) -> None:
        self.sent.append(("tool", kw))

    async def receive(self) -> Any:
        assert self.inbox is not None
        while True:
            item = await self.inbox.get()
            if isinstance(item, BaseException):
                raise item
            if item == "END":  # (a turn is over: the SDK's receive() returns)
                return
            yield item


class FakeClient:
    """``client.aio.live.connect(model=, config=)`` → an async context manager."""

    def __init__(self, *, connect_error: BaseException | None = None, delay: float = 0.0) -> None:
        self.connect_error = connect_error
        self.sessions: list[FakeSession] = []
        self.calls: list[dict[str, Any]] = []
        self.delay = delay
        self.aio = NS(live=NS(connect=self._connect))

    def _connect(self, *, model: str, config: dict[str, Any]) -> Any:
        self.calls.append({"model": model, "config": config})
        client = self

        class Manager:
            async def __aenter__(self_inner) -> FakeSession:
                if client.connect_error is not None:
                    raise client.connect_error
                session = FakeSession(client.delay)
                session.inbox = asyncio.Queue()
                client.sessions.append(session)
                return session

            async def __aexit__(self_inner, *exc: Any) -> None:
                return None

        return Manager()

    def push(self, item: Any) -> None:
        """Hand a message to the (latest) session's receive loop, from any thread."""
        session = self.sessions[-1]
        assert session.inbox is not None
        loop = self.loop
        loop.call_soon_threadsafe(session.inbox.put_nowait, item)

    loop: asyncio.AbstractEventLoop


class Rig:
    def __init__(self, client: FakeClient, key: str | None = FAKE_KEY, **kw: Any) -> None:
        self.client = client
        self.keys_asked = 0
        self.factory_keys: list[str] = []

        def loader() -> str | None:
            self.keys_asked += 1
            return key

        def factory(k: str) -> FakeClient:
            self.factory_keys.append(k)
            return client

        self.transport = G.GenaiTransport(client_factory=factory, key_loader=loader, **kw)
        self.events: list[E.LiveEvent] = []

    def open(self, **kw: Any) -> None:
        self.transport.open(setup(**kw), 0.0)

    def wait(self, pred: Callable[[], bool], timeout: float = 5.0) -> None:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.events += self.transport.poll(0.0)
            if pred():
                return
            time.sleep(0.005)
        raise AssertionError(f"timed out; events so far: {[e.kind for e in self.events]}")

    def kinds(self) -> list[str]:
        return [e.kind for e in self.events]

    def opened(self) -> FakeSession:
        self.wait(lambda: E.OPENED in self.kinds())
        # the fake's loop is the connection thread's: remember it for ``push``
        conn = self.transport._conn
        assert conn is not None
        self.wait(lambda: conn.loop is not None)
        self.client.loop = conn.loop  # type: ignore[assignment]
        return self.client.sessions[-1]


def msg(**kw: Any) -> NS:
    return NS(**kw)


# ── the config the SDK is given ────────────────────────────────────────────────────


def test_the_config_asks_for_audio_transcribes_both_ways_and_keeps_the_session_long() -> None:
    cfg = G.build_config(setup(resume_handle="h1", media_resolution="MEDIA_RESOLUTION_LOW"))
    assert cfg["response_modalities"] == ["AUDIO"]  # (TEXT is refused by gemini-3.8-live)
    assert cfg["system_instruction"] == "SOUL" and cfg["tools"] == T.tools_config()
    langs = {"language_codes": ["zh-TW", "en-US"]}  # (the household's language first)
    assert cfg["input_audio_transcription"] == langs
    assert cfg["output_audio_transcription"] == langs
    assert cfg["context_window_compression"] == {"sliding_window": {}}
    assert cfg["session_resumption"] == {"handle": "h1"}
    assert cfg["media_resolution"] == "MEDIA_RESOLUTION_LOW"
    vad = cfg["realtime_input_config"]["automatic_activity_detection"]
    assert vad["start_of_speech_sensitivity"] == "START_SENSITIVITY_HIGH"
    assert vad["end_of_speech_sensitivity"] == "END_SENSITIVITY_LOW"
    assert vad["prefix_padding_ms"] == 300 and vad["silence_duration_ms"] == 500


def test_the_transcriber_is_told_the_language_so_it_does_not_guess_one_per_sentence() -> None:
    """Live 2026-10-04: without a hint 「過來這裡」 came back as 过来这里, 「蛋黃」 as 赵华 and a
    short sentence as Spanish. The hint is a setting; empty is the service's own guessing."""
    cfg = G.build_config(setup())
    assert cfg["input_audio_transcription"]["language_codes"] == ["zh-TW", "en-US"]
    assert cfg["output_audio_transcription"]["language_codes"] == ["zh-TW", "en-US"]
    other = G.build_config(setup(transcription_languages=("ja-JP", "zh-TW", "")))
    assert other["input_audio_transcription"] == {"language_codes": ["ja-JP", "zh-TW"]}
    off = G.build_config(setup(transcription_languages=()))
    assert off["input_audio_transcription"] == {} and off["output_audio_transcription"] == {}
    none = G.build_config(setup(input_transcription=False, output_transcription=False))
    assert "input_audio_transcription" not in none and "output_audio_transcription" not in none


def test_the_family_speech_is_transcribed_with_a_bias_toward_the_names_and_the_duck_own_is_not() -> (
    None
):
    """The SDK's ``custom_vocabulary``: the duck's name and the family's names, for the input
    transcription only (the duck's own words are not the family's names to listen for)."""
    cfg = G.build_config(setup(custom_vocabulary=("蛋黃", "媽媽", "")))
    assert cfg["input_audio_transcription"] == {
        "language_codes": ["zh-TW", "en-US"],
        "custom_vocabulary": ["蛋黃", "媽媽"],
    }
    assert cfg["output_audio_transcription"] == {"language_codes": ["zh-TW", "en-US"]}
    assert "custom_vocabulary" not in G.build_config(setup())["input_audio_transcription"]
    alone = G.build_config(setup(transcription_languages=(), custom_vocabulary=("蛋黃",)))
    assert alone["input_audio_transcription"] == {"custom_vocabulary": ["蛋黃"]}
    types = pytest.importorskip("google.genai.types")
    model = types.LiveConnectConfig.model_validate(cfg)
    assert model.input_audio_transcription.custom_vocabulary == ["蛋黃", "媽媽"]


def test_a_first_connection_has_resumption_on_without_a_handle() -> None:
    cfg = G.build_config(setup())
    assert cfg["session_resumption"] == {} and "media_resolution" not in cfg


def test_the_sdk_accepts_the_whole_config() -> None:
    types = pytest.importorskip("google.genai.types")
    cfg = types.LiveConnectConfig.model_validate(G.build_config(setup(resume_handle="h")))
    assert cfg.session_resumption.handle == "h"
    assert cfg.context_window_compression.sliding_window is not None
    assert cfg.input_audio_transcription is not None and cfg.output_audio_transcription is not None
    assert cfg.input_audio_transcription.language_codes == ["zh-TW", "en-US"]
    assert cfg.output_audio_transcription.language_codes == ["zh-TW", "en-US"]
    assert cfg.realtime_input_config.automatic_activity_detection.silence_duration_ms == 500


# ── messages ───────────────────────────────────────────────────────────────────────


def test_messages_become_events_shapes_from_the_real_service() -> None:

    def kinds(message: Any) -> list[tuple[str, dict[str, Any]]]:
        return [(e.kind, e.data) for e in G.map_message(message, 1.5)]

    assert kinds(msg(session_resumption_update=msg(new_handle="abc", resumable=True))) == [
        (E.RESUMPTION, {"resumable": True, "handle": "abc"})
    ]
    assert kinds(msg(server_content=msg(input_transcription=msg(text="蛋黃，過來這裡。")))) == [
        (E.INPUT_TEXT, {"text": "蛋黃，過來這裡。"})
    ]
    assert kinds(msg(server_content=msg(output_transcription=msg(text="呱！")))) == [
        (E.OUTPUT_TEXT, {"text": "呱！"})
    ]
    assert kinds(
        msg(
            voice_activity=msg(voice_activity_type=NS(name="ACTIVITY_START"), audio_offset="1.160s")
        )
    ) == [(E.VAD, {"type": "START", "offset_s": 1.16})]
    call = msg(
        function_calls=[
            msg(
                id="call_1",
                name="vocalize",
                args={"tags": [{"tag": "greet", "level": 1, "at_ms": 0}]},
            )
        ]
    )
    assert kinds(msg(tool_call=call)) == [
        (
            E.TOOL_CALL,
            {
                "calls": [
                    {
                        "id": "call_1",
                        "name": "vocalize",
                        "args": {"tags": [{"tag": "greet", "level": 1, "at_ms": 0}]},
                    }
                ]
            },
        )
    ]
    assert kinds(msg(tool_call_cancellation=msg(ids=["a", "b"]))) == [
        (E.TOOL_CANCEL, {"ids": ["a", "b"]})
    ]
    assert kinds(msg(go_away=msg(time_left="10s"))) == [(E.GO_AWAY, {"time_left_s": 10.0})]
    assert kinds(msg(server_content=msg(interrupted=True))) == [(E.INTERRUPTED, {})]
    assert kinds(msg(server_content=msg(generation_complete=True))) == [(E.GENERATION_COMPLETE, {})]
    assert kinds(msg(server_content=msg(turn_complete=True))) == [(E.TURN_COMPLETE, {})]


def test_the_models_voice_is_counted_in_bytes_and_nothing_else() -> None:
    parts = [
        msg(inline_data=msg(data=b"\x00" * 4800)),
        msg(inline_data=msg(data=b"\x00" * 1000)),
        msg(text=None),
    ]
    out = G.map_message(msg(server_content=msg(model_turn=msg(parts=parts))), 2.0)
    assert [(e.kind, e.data) for e in out] == [(E.AUDIO_OUT, {"bytes": 5800})]


def test_usage_has_the_totals_and_the_input_by_modality() -> None:
    usage = msg(
        prompt_token_count=953,
        response_token_count=25,
        thoughts_token_count=113,
        total_token_count=978,
        prompt_tokens_details=[
            msg(modality=NS(name="TEXT"), token_count=529),
            msg(modality=NS(name="AUDIO"), token_count=277),
            msg(modality="VIDEO", token_count=3),
        ],
    )
    [event] = G.map_message(msg(usage_metadata=usage), 3.0)
    assert event.kind == E.USAGE and event.data == {
        "total": 978,
        "prompt": 953,
        "response": 25,
        "thoughts": 113,
        "by_modality": {"TEXT": 529, "AUDIO": 277, "VIDEO": 3},
    }
    assert G.map_message(msg(usage_metadata=msg(total_token_count=None)), 3.0) == []


def test_one_message_can_say_several_things() -> None:
    both = msg(
        server_content=msg(turn_complete=True, input_transcription=msg(text="嗨")),
        usage_metadata=msg(total_token_count=5, prompt_token_count=3, response_token_count=2),
    )
    assert [e.kind for e in G.map_message(both, 0.0)] == [E.INPUT_TEXT, E.TURN_COMPLETE, E.USAGE]
    assert G.map_message(msg(), 0.0) == [] and G.map_message(msg(server_content=msg()), 0.0) == []


def test_the_real_sdk_message_types_map_too() -> None:
    types = pytest.importorskip("google.genai.types")
    real = types.LiveServerMessage.model_validate(
        {
            "server_content": {"input_transcription": {"text": "这是妈妈。"}},
            "usage_metadata": {
                "prompt_token_count": 936,
                "response_token_count": 11,
                "thoughts_token_count": 77,
                "total_token_count": 947,
                "prompt_tokens_details": [
                    {"modality": "TEXT", "token_count": 525},
                    {"modality": "AUDIO", "token_count": 264},
                ],
            },
        }
    )
    kinds = [(e.kind, e.data) for e in G.map_message(real, 0.0)]
    assert kinds[0] == (E.INPUT_TEXT, {"text": "这是妈妈。"})
    assert kinds[1][1]["by_modality"] == {"TEXT": 525, "AUDIO": 264}
    vad = types.LiveServerMessage.model_validate(
        {"voice_activity": {"voice_activity_type": "ACTIVITY_END", "audio_offset": "3.040s"}}
    )
    assert [(e.kind, e.data) for e in G.map_message(vad, 0.0)] == [
        (E.VAD, {"type": "END", "offset_s": 3.04})
    ]
    call = types.LiveServerMessage.model_validate(
        {
            "tool_call": {
                "function_calls": [
                    {
                        "id": "call_53863",
                        "args": {"tags": [{"at_ms": 0, "tag": "greet", "level": 1}]},
                        "name": "vocalize",
                    }
                ]
            }
        }
    )
    [event] = G.map_message(call, 0.0)
    assert (
        event.data["calls"][0]["name"] == "vocalize"
        and event.data["calls"][0]["args"]["tags"][0]["tag"] == "greet"
    )
    away = types.LiveServerMessage.model_validate({"go_away": {"time_left": "2s"}})
    assert G.map_message(away, 0.0)[0].data["time_left_s"] == 2.0


@pytest.mark.parametrize(
    "value, seconds",
    [("1.160s", 1.16), ("10s", 10.0), (3, 3.0), (2.5, 2.5), (None, None), ("soon", None)],
)
def test_durations(value: Any, seconds: float | None) -> None:
    assert G.seconds_of(value) == seconds


# ── a connection, end to end through a fake client ─────────────────────────────────


def test_open_connects_with_the_model_and_config_and_reads_the_key_once() -> None:
    client = FakeClient()
    rig = Rig(client)
    rig.open(resume_handle="h9")
    rig.opened()
    assert rig.keys_asked == 1 and rig.factory_keys == [FAKE_KEY]
    [call] = client.calls
    assert call["model"] == "gemini-3.8-live" and call["config"]["session_resumption"] == {
        "handle": "h9"
    }
    opened = [e for e in rig.events if e.kind == E.OPENED][0]
    assert opened.data == {"resumed": True}
    rig.transport.close()


def test_what_the_session_sends_reaches_the_server_in_the_sdks_shapes() -> None:
    client = FakeClient()
    rig = Rig(client)
    rig.open()
    session = rig.opened()
    rig.transport.send_audio(b"\x01\x00" * 320, 0.0)
    rig.transport.send_image(b"\xff\xd8x\xff\xd9", 0.0)
    rig.transport.send_text("狀態", 0.0, turn_complete=False)
    rig.transport.send_tool_response("c1", "vocalize", {"status": "queued"}, "SILENT", 0.0)
    rig.transport.send_audio_end(0.0)
    end = time.monotonic() + 5
    while len(session.sent) < 5 and time.monotonic() < end:
        time.sleep(0.005)
    assert session.sent == [
        ("realtime", {"audio": {"data": b"\x01\x00" * 320, "mime_type": "audio/pcm;rate=16000"}}),
        ("realtime", {"video": {"data": b"\xff\xd8x\xff\xd9", "mime_type": "image/jpeg"}}),
        (
            "content",
            {"turns": [{"role": "user", "parts": [{"text": "狀態"}]}], "turn_complete": False},
        ),
        (
            "tool",
            {
                "function_responses": [
                    {
                        "id": "c1",
                        "name": "vocalize",
                        "response": {"status": "queued"},
                        "scheduling": "SILENT",
                    }
                ]
            },
        ),
        ("realtime", {"audio_stream_end": True}),
    ]
    rig.transport.close()


def test_what_the_server_says_comes_out_of_poll_in_order() -> None:
    client = FakeClient()
    rig = Rig(client)
    rig.open()
    rig.opened()
    client.push(msg(session_resumption_update=msg(new_handle="h1", resumable=True)))
    client.push(
        msg(voice_activity=msg(voice_activity_type="ACTIVITY_START", audio_offset="1.160s"))
    )
    client.push(msg(server_content=msg(input_transcription=msg(text="蛋黃"))))
    client.push(
        msg(
            tool_call=msg(
                function_calls=[msg(id="c", name="feel", args={"valence": 0.4, "talk": 0.7})]
            )
        )
    )
    client.push(msg(server_content=msg(turn_complete=True)))
    client.push("END")
    client.push(msg(server_content=msg(input_transcription=msg(text="（下一輪）"))))
    rig.wait(lambda: rig.kinds().count(E.INPUT_TEXT) == 2)
    assert rig.kinds() == [
        E.OPENED,
        E.RESUMPTION,
        E.VAD,
        E.INPUT_TEXT,
        E.TOOL_CALL,
        E.TURN_COMPLETE,
        E.INPUT_TEXT,
    ]
    assert rig.events[1].data["handle"] == "h1"
    assert all(a.t <= b.t for a, b in zip(rig.events, rig.events[1:], strict=False))
    rig.transport.close()


# ── failures ───────────────────────────────────────────────────────────────────────


class ApiError(Exception):
    def __init__(self, code: int, status: str | None, message: str) -> None:
        super().__init__(f"{code} {status}. {message}")
        self.code, self.status = code, status


class Closed(Exception):
    """A websockets ConnectionClosed: ``rcvd.code``."""

    def __init__(self, code: int, reason: str) -> None:
        super().__init__(f"received {code} (internal error) {reason}; then sent {code}")
        self.rcvd = NS(code=code, reason=reason)


@pytest.mark.parametrize(
    "error, kind",
    [
        (
            ApiError(
                1007,
                None,
                "The requested combination of response modalities (TEXT) is not supported by the model.",
            ),
            E.BAD_REQUEST,
        ),
        (ApiError(429, "RESOURCE_EXHAUSTED", "Quota exceeded for this project"), E.QUOTA),
        (
            ApiError(403, "PERMISSION_DENIED", "API key not valid. Please pass a valid API key."),
            E.AUTH,
        ),
        (Closed(1011, "Resource has been exhausted (e.g. check quota)."), E.QUOTA),
        (Closed(1011, "Internal error occurred."), E.SERVER),
        (ApiError(503, "UNAVAILABLE", "The service is currently unavailable."), E.SERVER),
        (ConnectionResetError("connection reset by peer"), E.NETWORK),
        (asyncio.TimeoutError("timed out"), E.NETWORK),
        (RuntimeError("something odd"), E.UNKNOWN),
    ],
)
def test_a_failure_to_connect_is_a_closed_event_of_the_right_kind(
    error: BaseException, kind: str
) -> None:
    rig = Rig(FakeClient(connect_error=error))
    rig.open()
    rig.wait(lambda: E.CLOSED in rig.kinds())
    [closed] = [e for e in rig.events if e.kind == E.CLOSED]
    assert closed.data["kind"] == kind and E.OPENED not in rig.kinds()


def test_a_close_in_the_middle_of_a_session_is_a_closed_event_too() -> None:
    client = FakeClient()
    rig = Rig(client)
    rig.open()
    rig.opened()
    client.push(Closed(1011, "Resource has been exhausted (e.g. check quota)."))
    rig.wait(lambda: E.CLOSED in rig.kinds())
    closed = [e for e in rig.events if e.kind == E.CLOSED][0]
    assert closed.data["kind"] == E.QUOTA and closed.data["code"] == 1011


def test_no_key_is_a_closed_event_and_the_sdk_is_never_asked() -> None:
    client = FakeClient()
    rig = Rig(client, key=None)
    rig.open()
    [event] = rig.transport.poll(0.0)
    assert event.kind == E.CLOSED and event.data["kind"] == E.NO_KEY
    assert rig.factory_keys == [] and client.calls == []


def test_no_message_that_leaves_holds_the_key() -> None:
    boom = ApiError(
        400, None, f"bad request to wss://example.test/ws?key={FAKE_KEY}&x=1 with {FAKE_KEY}"
    )
    rig = Rig(FakeClient(connect_error=boom))
    rig.open()
    rig.wait(lambda: E.CLOSED in rig.kinds())
    text = str([(e.kind, e.data) for e in rig.events])
    assert FAKE_KEY not in text and "key=" not in text.replace("key=<redacted>", "")
    assert K.MASK in text
    assert not K.find_keys(text)


def test_describe_error_has_a_kind_a_code_and_no_key() -> None:
    kind, code, reason = G.describe_error(
        ApiError(429, "RESOURCE_EXHAUSTED", f"quota {FAKE_KEY}"), FAKE_KEY
    )
    assert (kind, code) == (E.QUOTA, 429) and FAKE_KEY not in reason
    kind, code, _ = G.describe_error(Closed(1006, ""), None)
    assert (kind, code) == (E.NETWORK, 1006)
    kind, code, _ = G.describe_error(ValueError("x"), None)
    assert (kind, code) == (E.UNKNOWN, None)


# ── lifecycle ──────────────────────────────────────────────────────────────────────


def test_close_ends_the_thread_and_silences_the_connection() -> None:
    client = FakeClient()
    rig = Rig(client)
    rig.open()
    rig.opened()
    conn = rig.transport._conn
    assert conn is not None and conn.thread is not None
    rig.transport.close()
    conn.thread.join(timeout=5)
    assert not conn.thread.is_alive()
    client.loop = conn.loop  # type: ignore[assignment]
    assert rig.transport.poll(0.0) == [] or all(e.kind != E.CLOSED for e in rig.transport.poll(0.0))
    rig.transport.send_audio(b"x", 0.0)  # (no connection: a no-op, no error)


def test_a_new_open_replaces_the_old_connection_and_its_events_are_dropped() -> None:
    client = FakeClient()
    rig = Rig(client)
    rig.open()
    rig.opened()
    first = rig.transport._conn
    rig.events.clear()
    rig.open()
    rig.wait(lambda: E.OPENED in rig.kinds())
    assert rig.transport._conn is not first and len(client.sessions) == 2
    assert rig.kinds().count(E.OPENED) == 1
    rig.transport.close()


def test_audio_over_the_backlog_is_dropped_and_counted_not_queued_forever() -> None:
    client = FakeClient(delay=0.05)  # (a stalled line: each send takes 50 ms)
    rig = Rig(client, max_backlog_s=1.0)
    rig.open()
    rig.opened()
    for _ in range(300):
        rig.transport.send_audio(b"\x00\x00" * 320, 0.0)
    assert rig.transport.audio_dropped >= 200
    rig.transport.close()


def test_things_sent_before_the_connection_is_up_are_kept_and_go_first() -> None:
    client = FakeClient()
    rig = Rig(client)
    rig.open()
    rig.transport.send_text("early", 0.0)  # (the connection is still opening)
    session = rig.opened()
    end = time.monotonic() + 5
    while not session.sent and time.monotonic() < end:
        time.sleep(0.005)
    assert session.sent[0][0] == "content"
    rig.transport.close()


def test_a_missing_sdk_is_a_closed_event_not_an_exception() -> None:
    def no_sdk(key: str) -> Any:
        raise ModuleNotFoundError("No module named 'google'")

    transport = G.GenaiTransport(client_factory=no_sdk, key_loader=lambda: FAKE_KEY)
    transport.open(setup(), 0.0)
    end = time.monotonic() + 5
    events: list[E.LiveEvent] = []
    while not events and time.monotonic() < end:
        events = transport.poll(0.0)
        time.sleep(0.005)
    [closed] = events
    assert closed.kind == E.CLOSED and "No module named" in closed.data["reason"]
    assert closed.data["kind"] in (E.UNKNOWN, E.NETWORK, E.SERVER)
    assert FAKE_KEY not in str(closed.data)


def test_a_session_for_gemini_shares_its_clock_with_the_connection() -> None:
    from duckmind.live.session import LiveSession

    ticks = iter(range(1000))

    def clock() -> float:
        return float(next(ticks))

    session = LiveSession.for_gemini("SOUL", clock=clock, enabled=False)
    assert isinstance(session.transport, G.GenaiTransport)
    assert session.transport._clock is clock and session.clock is clock
    assert session.status()["state"] == "off"
