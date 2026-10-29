"""The real connection: Gemini Live through the ``google-genai`` SDK (a required dependency).

One connection = one background thread with its own asyncio loop (the duck's brain loop is
synchronous and never waits for the network). The session hands things over through
:class:`GenaiTransport`'s ``send_*`` methods (queued for the loop thread; audio over a backlog is
dropped and counted) and takes what the server sent from :meth:`GenaiTransport.poll`, already
turned into :class:`~duckmind.live.events.LiveEvent` values by :func:`map_message`.

Facts this module is built on (checked against the live service, 2026-10-02, SDK 2.27.0):

* ``gemini-3.8-live`` refuses ``response_modalities=["TEXT"]`` (close 1007: "requested combination
  of response modalities (TEXT) is not supported by the model"): the duck asks for AUDIO, throws
  the audio away (counting bytes) and reads **output transcription** as the model's text, and
  **input transcription** as what it heard;
* the server sends ``voice_activity`` (ACTIVITY_START / ACTIVITY_END with the audio offset) and one
  ``usage_metadata`` per turn; transcripts come as chunks, tool calls as one message per batch;
* a status text goes up as ``send_client_content(turn_complete=False)`` (the server waits for
  more and does not answer it); pictures and audio as ``send_realtime_input``;
* tools are ``NON_BLOCKING``; their responses go back with ``scheduling=SILENT``.

The SDK is reached through two seams (``client_factory`` and the plain-dict arguments), so the
tests drive this class with a fake client and no network. The API key is read when a connection
opens (:func:`duckmind.live.keys.load_api_key`), lives in one local variable, and every message that
leaves this module passes through :func:`~duckmind.live.keys.redact`.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from duckmind.live import events as E
from duckmind.live.keys import load_api_key, redact
from duckmind.live.transport import Setup, classify_error

IN_MIME = "audio/pcm;rate=16000"
MAX_BACKLOG_S = 3.0  # audio waiting to go up beyond this is dropped (a stalled line)
FRAME_S = 0.02
RECV_IDLE_S = 0.01

LOG = logging.getLogger("duckmind.live.genai")

Clock = Callable[[], float]
ClientFactory = Callable[[str], Any]


def default_client(key: str) -> Any:
    """``google.genai.Client`` for ``key`` (imported here: the SDK is only needed by the real
    connection)."""
    from google import genai

    return genai.Client(api_key=key)


# ── config and messages ────────────────────────────────────────────────────────────


def build_config(setup: Setup) -> dict[str, Any]:
    """The ``LiveConnectConfig`` of a setup, as the plain dictionary the SDK takes."""
    config: dict[str, Any] = {
        "response_modalities": list(setup.response_modalities),
        "system_instruction": setup.system_instruction,
        "tools": setup.tools,
        "realtime_input_config": {
            "automatic_activity_detection": {
                "start_of_speech_sensitivity": setup.vad_start,
                "end_of_speech_sensitivity": setup.vad_end,
                "prefix_padding_ms": setup.vad_prefix_ms,
                "silence_duration_ms": setup.vad_silence_ms,
            }
        },
        "session_resumption": {"handle": setup.resume_handle} if setup.resume_handle else {},
    }
    if setup.input_transcription:
        config["input_audio_transcription"] = transcription_config(setup, vocabulary=True)
    if setup.output_transcription:
        config["output_audio_transcription"] = transcription_config(setup)
    if setup.compression:
        config["context_window_compression"] = {"sliding_window": {}}
    if setup.media_resolution:
        config["media_resolution"] = setup.media_resolution
    return config


def transcription_config(setup: Setup, *, vocabulary: bool = False) -> dict[str, Any]:
    """``AudioTranscriptionConfig`` as a plain dictionary: the language hint, and (for the input
    transcription, the family's speech: ``vocabulary=True``) the SDK's ``custom_vocabulary``, a bias
    towards the duck's name and the family's names. Without the language hint the service guesses a
    language for each sentence (live 2026-10-04: 过来这里 in simplified characters, a short sentence as
    Spanish or Vietnamese words). The vocabulary was first measured on the duck's name alone and left
    out (16 calls of 蛋黃: 1/16 right without it, 2/16 with it, the same with zh-TW alone); the
    planner turned it on 2026-10-04 on the overall accuracy it showed (0.82 to 0.90), as a setting
    (:attr:`~duckmind.live.session.SessionConfig.custom_vocabulary`)."""
    languages = [code for code in setup.transcription_languages if code]
    config: dict[str, Any] = {"language_codes": languages} if languages else {}
    phrases = [p for p in setup.custom_vocabulary if p]
    if vocabulary and phrases:
        config["custom_vocabulary"] = phrases
    return config


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _name(value: Any) -> str:
    """An enum's name (or a string as is)."""
    if value is None:
        return ""
    return str(getattr(value, "name", None) or getattr(value, "value", None) or value)


_SECONDS = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*s?\s*$")


def seconds_of(value: Any) -> float | None:
    """A duration the SDK gives as ``"1.160s"``, a number or a timedelta."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    total = getattr(value, "total_seconds", None)
    if callable(total):
        return float(total())
    match = _SECONDS.match(str(value))
    return float(match.group(1)) if match else None


def map_message(message: Any, t: float) -> list[E.LiveEvent]:
    """What one server message says, as events (it can say several things at once)."""
    out: list[E.LiveEvent] = []
    update = _get(message, "session_resumption_update")
    if update is not None:
        out.append(
            E.LiveEvent(
                E.RESUMPTION,
                t,
                {
                    "resumable": bool(_get(update, "resumable")),
                    "handle": _get(update, "new_handle"),
                },
            )
        )
    call = _get(message, "tool_call")
    if call is not None:
        calls = [
            {
                "id": str(_get(fc, "id") or ""),
                "name": str(_get(fc, "name") or ""),
                "args": dict(_get(fc, "args") or {}),
            }
            for fc in _get(call, "function_calls") or []
        ]
        if calls:
            out.append(E.LiveEvent(E.TOOL_CALL, t, {"calls": calls}))
    cancel = _get(message, "tool_call_cancellation")
    if cancel is not None:
        out.append(
            E.LiveEvent(E.TOOL_CANCEL, t, {"ids": [str(i) for i in _get(cancel, "ids") or []]})
        )
    away = _get(message, "go_away")
    if away is not None:
        out.append(E.LiveEvent(E.GO_AWAY, t, {"time_left_s": seconds_of(_get(away, "time_left"))}))
    vad = _get(message, "voice_activity")
    if vad is not None:
        kind = _name(_get(vad, "voice_activity_type")).replace("ACTIVITY_", "")
        out.append(
            E.LiveEvent(E.VAD, t, {"type": kind, "offset_s": seconds_of(_get(vad, "audio_offset"))})
        )
    content = _get(message, "server_content")
    if content is not None:
        heard = _get(_get(content, "input_transcription"), "text")
        if heard:
            out.append(E.LiveEvent(E.INPUT_TEXT, t, {"text": str(heard)}))
        said = _get(_get(content, "output_transcription"), "text")
        if said:
            out.append(E.LiveEvent(E.OUTPUT_TEXT, t, {"text": str(said)}))
        audio_bytes = 0
        for part in _get(_get(content, "model_turn"), "parts") or []:
            blob = _get(part, "inline_data")
            data = _get(blob, "data")
            if data:
                audio_bytes += len(data)
        if audio_bytes:
            out.append(E.LiveEvent(E.AUDIO_OUT, t, {"bytes": audio_bytes}))
        if _get(content, "interrupted"):
            out.append(E.LiveEvent(E.INTERRUPTED, t, {}))
        if _get(content, "generation_complete"):
            out.append(E.LiveEvent(E.GENERATION_COMPLETE, t, {}))
        if _get(content, "turn_complete"):
            out.append(E.LiveEvent(E.TURN_COMPLETE, t, {}))
    usage = _get(message, "usage_metadata")
    if usage is not None and _get(usage, "total_token_count") is not None:
        by: dict[str, int] = {}
        for item in _get(usage, "prompt_tokens_details") or []:
            by[_name(_get(item, "modality"))] = by.get(_name(_get(item, "modality")), 0) + int(
                _get(item, "token_count") or 0
            )
        out.append(
            E.LiveEvent(
                E.USAGE,
                t,
                {
                    "total": int(_get(usage, "total_token_count") or 0),
                    "prompt": int(_get(usage, "prompt_token_count") or 0),
                    "response": int(_get(usage, "response_token_count") or 0),
                    "thoughts": int(_get(usage, "thoughts_token_count") or 0),
                    "by_modality": by,
                },
            )
        )
    return out


def describe_error(error: BaseException, key: str | None = None) -> tuple[str, int | None, str]:
    """(kind, code, reason) of an exception from the SDK or the socket, with no key in it."""
    code = _get(error, "code")
    if not isinstance(code, int):
        code = _get(_get(error, "rcvd"), "code")
    code = code if isinstance(code, int) and not isinstance(code, bool) else None
    status = _get(error, "status")
    reason = redact(f"{type(error).__name__}: {error}", [key])
    return (
        classify_error(code, status if isinstance(status, str) else None, reason),
        code,
        reason[:300],
    )


# ── the transport ──────────────────────────────────────────────────────────────────


@dataclass
class _Conn:
    gen: int
    stop: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    loop: asyncio.AbstractEventLoop | None = None
    cmds: asyncio.Queue[tuple[str, Any]] | None = None
    pre: list[tuple[str, Any]] = field(default_factory=list)
    audio_pending: int = 0
    thread: threading.Thread | None = None


class GenaiTransport:
    name = "genai"

    def __init__(
        self,
        *,
        clock: Clock = time.monotonic,
        client_factory: ClientFactory = default_client,
        key_loader: Callable[[], str | None] = load_api_key,
        max_backlog_s: float = MAX_BACKLOG_S,
    ) -> None:
        self._clock = clock
        self._client_factory = client_factory
        self._key_loader = key_loader
        self._max_backlog = max(1, int(max_backlog_s / FRAME_S))
        self._events: queue.SimpleQueue[E.LiveEvent] = queue.SimpleQueue()
        self._gen = 0
        self._conn: _Conn | None = None
        self.audio_dropped = 0
        self.sent_audio = 0
        self.sent_images = 0
        self.sent_texts = 0
        self.loop_errors = 0  # what the connection loops would have printed (see _loop_error)

    # ── Transport ─────────────────────────────────────────────────────────────

    def open(self, setup: Setup, now: float) -> None:
        self.close()
        key = self._key_loader()
        self._gen += 1
        conn = _Conn(self._gen)
        self._conn = conn
        if not key:
            self._emit(conn, E.CLOSED, {"kind": E.NO_KEY, "reason": "no API key"})
            return
        thread = threading.Thread(
            target=self._run, args=(conn, setup, key), name=f"live-conn-{conn.gen}", daemon=True
        )
        conn.thread = thread
        thread.start()

    def send_audio(self, pcm: bytes, now: float) -> None:
        conn = self._conn
        if conn is None:
            return
        if conn.audio_pending >= self._max_backlog:
            self.audio_dropped += 1
            return
        conn.audio_pending += 1
        self._submit(conn, ("audio", bytes(pcm)))

    def send_audio_end(self, now: float) -> None:
        if self._conn is not None:
            self._submit(self._conn, ("audio_end", None))

    def send_image(self, jpeg: bytes, now: float, mime: str = "image/jpeg") -> None:
        if self._conn is not None:
            self._submit(self._conn, ("image", (bytes(jpeg), mime)))

    def send_text(self, text: str, now: float, *, turn_complete: bool = False) -> None:
        if self._conn is not None:
            self._submit(self._conn, ("text", (text, turn_complete)))

    def send_tool_response(
        self, call_id: str, name: str, response: dict[str, Any], scheduling: str, now: float
    ) -> None:
        if self._conn is not None:
            item = {"id": call_id, "name": name, "response": response, "scheduling": scheduling}
            self._submit(self._conn, ("tool", item))

    def poll(self, now: float) -> list[E.LiveEvent]:
        out: list[E.LiveEvent] = []
        while True:
            try:
                out.append(self._events.get_nowait())
            except queue.Empty:
                return out

    def close(self) -> None:
        conn, self._conn = self._conn, None
        if conn is None:
            return
        conn.stop.set()
        self._submit(conn, ("stop", None))

    # ── plumbing ──────────────────────────────────────────────────────────────

    def _emit(self, conn: _Conn, kind: str, data: dict[str, Any]) -> None:
        if conn.gen != self._gen or conn.stop.is_set():
            return  # (an old connection says nothing any more)
        self._events.put(E.LiveEvent(kind, self._clock(), data))

    def _submit(self, conn: _Conn, item: tuple[str, Any]) -> None:
        with conn.lock:
            loop, cmds = conn.loop, conn.cmds
            if cmds is None or loop is None:
                conn.pre.append(item)
                return
        try:
            loop.call_soon_threadsafe(cmds.put_nowait, item)
        except RuntimeError:  # (the loop has ended: the connection is over)
            pass

    def _run(self, conn: _Conn, setup: Setup, key: str) -> None:
        try:
            asyncio.run(self._main(conn, setup, key))
        except BaseException as error:  # (nothing may kill the thread without a word)
            kind, code, reason = describe_error(error, key)
            self._emit(conn, E.CLOSED, {"kind": kind, "code": code, "reason": reason})

    def _loop_error(self, loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        """Whatever the connection's loop would print on its own (a task nobody awaited, a
        callback that raised): counted, and the first one of a connection written once, redacted.
        Nothing here may raise."""
        self.loop_errors += 1
        if self.loop_errors == 1 or self.loop_errors % 100 == 0:
            error = context.get("exception")
            what = f"{type(error).__name__}: {error}" if error else str(context.get("message"))
            LOG.warning(
                "cloud connection loop (%d so far): %s", self.loop_errors, redact(what)[:300]
            )

    async def _main(self, conn: _Conn, setup: Setup, key: str) -> None:
        asyncio.get_running_loop().set_exception_handler(self._loop_error)
        try:
            client = self._client_factory(key)
            manager = client.aio.live.connect(model=setup.model, config=build_config(setup))
            async with manager as session:
                loop = asyncio.get_running_loop()
                cmds: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
                with conn.lock:
                    conn.loop, conn.cmds = loop, cmds
                    for item in conn.pre:
                        cmds.put_nowait(item)
                    conn.pre = []
                self._emit(conn, E.OPENED, {"resumed": setup.resume_handle is not None})
                sender = asyncio.create_task(self._sender(conn, session, cmds))
                receiver = asyncio.create_task(self._receiver(conn, session))
                await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
                for task in (sender, receiver):
                    if not task.done():
                        task.cancel()
                # every task's outcome is read here (both can fail in the same breath when the
                # service closes the line: the second one would be "never retrieved" otherwise)
                outcomes = await asyncio.gather(sender, receiver, return_exceptions=True)
                for outcome in outcomes:
                    if isinstance(outcome, BaseException) and not isinstance(
                        outcome, asyncio.CancelledError
                    ):
                        raise outcome
            if not conn.stop.is_set():
                self._emit(conn, E.CLOSED, {"kind": E.NETWORK, "code": None, "reason": "ended"})
        except asyncio.CancelledError:
            raise
        except Exception as error:
            kind, code, reason = describe_error(error, key)
            self._emit(conn, E.CLOSED, {"kind": kind, "code": code, "reason": reason})

    async def _sender(
        self, conn: _Conn, session: Any, cmds: asyncio.Queue[tuple[str, Any]]
    ) -> None:
        while True:
            kind, payload = await cmds.get()
            if kind == "stop":
                return
            if kind == "audio":
                conn.audio_pending = max(0, conn.audio_pending - 1)
                await session.send_realtime_input(audio={"data": payload, "mime_type": IN_MIME})
                self.sent_audio += 1
            elif kind == "audio_end":
                await session.send_realtime_input(audio_stream_end=True)
            elif kind == "image":
                data, mime = payload
                await session.send_realtime_input(video={"data": data, "mime_type": mime})
                self.sent_images += 1
            elif kind == "text":
                text, complete = payload
                await session.send_client_content(
                    turns=[{"role": "user", "parts": [{"text": text}]}], turn_complete=complete
                )
                self.sent_texts += 1
            elif kind == "tool":
                await session.send_tool_response(function_responses=[payload])

    async def _receiver(self, conn: _Conn, session: Any) -> None:
        while not conn.stop.is_set():
            got = False
            async for message in session.receive():
                got = True
                for event in map_message(message, self._clock()):
                    self._emit(conn, event.kind, event.data)
            if not got:
                await asyncio.sleep(RECV_IDLE_S)


__all__ = [
    "GenaiTransport",
    "build_config",
    "default_client",
    "describe_error",
    "map_message",
    "seconds_of",
    "transcription_config",
]
