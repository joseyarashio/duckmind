"""One persistent Gemini Live session for the duck (live voice, LV1).

The duck's microphone (20 ms PCM frames, :class:`~duckmind.simweb.neuro_mic.MicStream`) and eye
(JPEG pictures, at most one a second, :class:`~duckmind.simweb.eyeframes.EyeFrames`) go up one
connection; a text status of the duck (mood, drives, what it does, who it knows, what it sees) goes
up every few seconds; what comes back is **text and tool calls only** — the duck's mouth is the
official voice bank, so the model's own voice (gemini-3.8-live answers in audio only) is thrown
away. The tools (:mod:`duckmind.live.tools`) are queued with timestamps and marks; acting on them
is LV2.

Shape. The session is a small synchronous state machine over a :class:`~duckmind.live.transport.Transport`
(the real connection, the fake Gemini or a cassette): the host feeds it (``feed_audio``,
``feed_image``, ``set_context``), calls ``pump`` regularly (``attach_mic`` does both for the
microphone's ticks) and reads ``take_calls`` and ``status``. Nothing here raises into the host's
loop: a fault is counted, kept (redacted) and shown in the status.

* **Cloud switch**: ``set_enabled(False)`` stops everything that leaves the machine at once and
  closes the connection; ``status()['sending']`` says what is leaving.
* **Down, off or out of quota**: the duck lives on (the brain never waits for the cloud); the
  status says which, in words from the catalogs (``neuro.live.status.*``), and the connection is
  retried with a back-off (short for a dropped line, long for a quota refusal).
* **Reconnect**: the server's resumption handle is kept (in memory); a dropped or ``go_away``
  connection is replaced by a resumed one, the audio of the gap (the last :attr:`SessionConfig.hold_audio_s`)
  waiting to go first; two failures in a row forget the handle (a stale one fails the same way each time).
* **Echo gate** (:mod:`duckmind.live.echo`): every transcript chunk and every tool call is marked
  ``echo_suspect`` / ``echo_only`` when it can only be the duck hearing itself.
* **Usage meter** (:mod:`duckmind.live.meter`): per-minute counts, quota errors, optional caps.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from array import array
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from duckmind.i18n import msg
from duckmind.live import events as E
from duckmind.live.echo import CallMark, EchoGate
from duckmind.live.keys import redact
from duckmind.live.meter import UsageMeter
from duckmind.live.soul import ContextSnapshot
from duckmind.live.tools import ParsedCall, parse_call, tools_config
from duckmind.live.transport import DEFAULT_MODEL, TRANSCRIPTION_LANGUAGES, Setup, Transport

LOG = logging.getLogger("duckmind.live.session")

_ERROR_WORD_KINDS = (E.SERVER, E.NETWORK, E.BAD_REQUEST, E.UNKNOWN)

# connection / cloud states (the status words are ``neuro.live.status.<state>``)
OFF = "off"
CONNECTING = "connecting"
ONLINE = "online"
RECONNECTING = "reconnecting"
DOWN = "down"
QUOTA = "quota"
CAPPED = "capped"
NO_KEY = "no_key"
STATES = (OFF, CONNECTING, ONLINE, RECONNECTING, DOWN, QUOTA, CAPPED, NO_KEY)

SILENT = "SILENT"  # a tool response the model files away without a new turn
FRAME_S = 0.02
FLOOR_DBFS = -120.0

Clock = Callable[[], float]
Soul = str | Callable[[], str]
Responder = Callable[["QueuedCall"], Mapping[str, Any]]


@dataclass(frozen=True)
class SessionConfig:
    model: str = DEFAULT_MODEL
    context_period_s: float = (
        5.0  # the status text goes up at most this often (and only if changed)
    )
    image_period_s: float = 1.0  # at most one picture a second
    hold_audio_s: float = 2.0  # audio kept while a connection is replaced
    backoff_base_s: float = 1.5
    backoff_max_s: float = 8.0
    down_after_fails: int = 3  # this many failures in a row: the status says "down"
    drop_handle_after_fails: int = 2
    quota_backoff_s: float = 60.0  # RESOURCE_EXHAUSTED: wait this long, then double, up to …
    quota_backoff_max_s: float = 900.0
    auth_retry_s: float = 60.0
    stable_after_s: float = 20.0  # a connection that stayed up this long ends a burst of failures
    call_queue_max: int = 200
    log_max: int = 600
    media_resolution: str | None = None
    # what language the transcriber is told the family speaks (the household's own first; ``()``:
    # the service guesses a language for each sentence, which is how simplified characters and
    # Spanish words got into a Taiwanese family's transcripts)
    transcription_languages: tuple[str, ...] = TRANSCRIPTION_LANGUAGES
    # tell the transcriber the duck's name and the family's names (``LiveSession.set_vocabulary``;
    # the SDK's ``custom_vocabulary``): on by default (planner 2026-10-04: overall accuracy 0.82 to
    # 0.90); False sends none
    custom_vocabulary: bool = True


@dataclass
class QueuedCall:
    """A tool call the model made, waiting for LV2 to act on (or refuse) it."""

    seq: int
    id: str
    name: str
    args: dict[str, Any]
    raw: dict[str, Any]
    problems: tuple[str, ...]
    off_bank: tuple[str, ...]
    t: float  # session clock when it arrived
    wall: float  # epoch seconds when it arrived (live only: the clock may be virtual)
    connection: int
    mark: CallMark
    echo_suspect: bool  # = mark.echo_only: the duck heard itself; LV2 refuses these

    def view(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "id": self.id,
            "name": self.name,
            "args": self.args,
            "raw": self.raw,
            "problems": list(self.problems),
            "off_bank": list(self.off_bank),
            "t": round(self.t, 3),
            "connection": self.connection,
            "echo_suspect": self.echo_suspect,
            "echo": self.mark.view(),
        }


@dataclass
class _Held:
    t: float
    pcm: bytes


def rms_dbfs(pcm: bytes) -> float:
    """The level of 16-bit little-endian mono PCM in dB re full scale (-120 for silence)."""
    n = len(pcm) // 2
    if n == 0:
        return FLOOR_DBFS
    samples = array("h")
    samples.frombytes(pcm[: n * 2])
    if samples.itemsize != 2:  # pragma: no cover (every platform we run on)
        return FLOOR_DBFS
    if _BIG_ENDIAN:  # pragma: no cover
        samples.byteswap()
    power = sum(s * s for s in samples) / n
    if power <= 0:
        return FLOOR_DBFS
    return max(FLOOR_DBFS, 10.0 * math.log10(power / (32768.0 * 32768.0)))


_BIG_ENDIAN = array("h", [1]).tobytes()[0] == 0


class LiveSession:
    def __init__(
        self,
        transport: Transport,
        soul: Soul,
        *,
        clock: Clock = time.monotonic,
        config: SessionConfig | None = None,
        meter: UsageMeter | None = None,
        gate: EchoGate | None = None,
        enabled: bool = True,
        responder: Responder | None = None,
        wall: Clock = time.time,
    ) -> None:
        self.transport = transport
        self._soul = soul
        self.clock = clock
        self._wall = wall
        self.config = config or SessionConfig()
        self.meter = meter or UsageMeter()
        self.gate = gate or EchoGate()
        self._responder = responder
        self._lock = threading.RLock()
        self.enabled = bool(enabled)
        self.state = OFF if not enabled else CONNECTING
        self._state_since = clock()
        self._conn = "closed"  # closed | opening | open
        self._connect_n = 0  # connections opened so far (a number for each call and transcript)
        self._resumes = 0
        self._fails = 0
        self._quota_fails = 0
        self._opened_at: float | None = None
        self._burst_errors = 0  # failures since the last good connection
        self._burst_logged = False  # the burst's one log line has been written
        self._retry_at = 0.0
        self._handle: str | None = None
        self._soul_text: str | None = None
        self._reconnect_now = False
        self._hold: deque[_Held] = deque(maxlen=max(1, int(self.config.hold_audio_s / FRAME_S)))
        self._last_image = -1e9
        self._pending_image: tuple[bytes, str] | None = None
        self._context: ContextSnapshot | str | None = None
        self._vocabulary: tuple[str, ...] = ()
        self._context_sent: str | None = None
        self._context_sent_at = -1e9
        self._queue: deque[QueuedCall] = deque()
        self._seq = 0
        self.log: deque[dict[str, Any]] = deque(maxlen=self.config.log_max)
        self.counts: dict[str, int] = {}
        self.calls_total = 0
        self.calls_dropped = 0
        self.faults = 0
        self.last_fault: str | None = None
        self.last_error: dict[str, Any] | None = None
        self.audio_out_bytes = 0
        self.audio_sent_s = 0.0
        self.last_vad: dict[str, Any] | None = None
        self.last_text_at: float | None = None
        self.notes_sent = 0  # one-off texts that went up (``send_note``), apart from the status

    @classmethod
    def for_gemini(cls, soul: Soul, *, clock: Clock = time.monotonic, **kwargs: Any) -> LiveSession:
        """A session over the real Gemini Live connection, the connection stamping its events on
        the same clock as the session (the two must agree: a call's time is its event's time)."""
        from duckmind.live.genai_transport import GenaiTransport

        return cls(GenaiTransport(clock=clock), soul, clock=clock, **kwargs)

    def set_human_evidence(self, evidence: Callable[[float, float], bool] | None) -> None:
        """Hand the echo gate the duck's own voice detection (``evidence(t0, t1)``: a person spoke
        between two moments of the session's clock): a transcript that overlaps the duck's own call
        is not marked as its echo when the duck itself heard a person speak in it."""
        self.gate.set_human_evidence(evidence)

    # ── switches ──────────────────────────────────────────────────────────────

    def set_enabled(self, on: bool) -> None:
        """The cloud switch. Off: nothing is sent any more and the connection is closed."""
        with self._lock:
            try:
                on = bool(on)
                if on == self.enabled:
                    return
                self.enabled = on
                now = self.clock()
                if not on:
                    self._shutdown_connection(now, "switched off")
                    self._hold.clear()
                    self._set_state(OFF, now)
                else:
                    self._retry_at = 0.0
                    self._fails = 0
                    self._quota_fails = 0
                    self._set_state(CONNECTING if self._connect_n == 0 else RECONNECTING, now)
                self._note("switch", on=on)
            except Exception as error:
                self._fault("set_enabled", error)

    def close(self) -> None:
        """For good: the connection ends and nothing more is sent."""
        with self._lock:
            try:
                self.enabled = False
                self._shutdown_connection(self.clock(), "closed")
                self._hold.clear()
                self._set_state(OFF, self.clock())
            except Exception as error:
                self._fault("close", error)

    # ── what goes up ──────────────────────────────────────────────────────────

    def attach_mic(self, mic: Any) -> None:
        """Feed every frame the microphone makes (``MicStream.listen``) and pump on it."""
        mic.listen(self._on_frame)

    def _on_frame(self, frame: Any) -> None:
        self.feed_audio(frame)
        self.pump()

    def feed_audio(self, frame: Any) -> bool:
        """One microphone frame (``pcm``, ``echo``, ``level_dbfs`` as of
        :class:`~duckmind.simweb.neuro_mic.MicFrame`). Returns True when it went up now. A frame
        with a browser microphone's person in it carries the sound the cloud is to hear
        (``cloud_pcm``: that person at their recording's own level, not the world's distance); it
        goes up, and the echo gate takes its level, instead of ``pcm``."""
        cloud = getattr(frame, "cloud_pcm", None)
        pcm = bytes(cloud if cloud is not None else frame.pcm)
        echo = bool(getattr(frame, "echo", False))
        level = getattr(frame, "cloud_level_dbfs", None) if cloud is not None else None
        if level is None:
            level = getattr(frame, "level_dbfs", None)
        return self.feed_pcm(pcm, echo=echo, level_dbfs=None if level is None else float(level))

    def feed_pcm(self, pcm: bytes, *, echo: bool = False, level_dbfs: float | None = None) -> bool:
        with self._lock:
            try:
                now = self.clock()
                if not self.enabled or self.state in (CAPPED, NO_KEY):
                    return False
                if level_dbfs is None:
                    level_dbfs = rms_dbfs(pcm)
                self.gate.on_frame(now, echo, level_dbfs)
                if self._conn == "open":
                    try:
                        self.transport.send_audio(pcm, now)
                    except Exception as error:
                        self._fault("send_audio", error)
                        self._lost(now, "send_audio failed")
                        return False
                    self.audio_sent_s += len(pcm) / 32000.0
                    self.meter.add(now, "audio_s", len(pcm) / 32000.0)
                    return True
                if len(self._hold) == self._hold.maxlen:
                    self.meter.add(now, "audio_dropped_s", FRAME_S)
                self._hold.append(_Held(now, pcm))
                return False
            except Exception as error:
                self._fault("feed_audio", error)
                return False

    def feed_image(self, jpeg: bytes, mime: str = "image/jpeg") -> bool:
        """One picture of the duck's eye. At most one goes up per :attr:`SessionConfig.image_period_s`
        (the service takes one frame a second): a picture that comes a little early (a source
        ticking once a second jitters) waits and goes at the next :meth:`pump`; a newer one
        replaces it. Returns True when this one went up now."""
        with self._lock:
            try:
                now = self.clock()
                if not self.enabled:
                    return False
                if self._conn != "open":
                    self.meter.add(now, "frames_skipped")
                    return False
                if now - self._last_image < self.config.image_period_s - 1e-9:
                    if self._pending_image is not None:
                        self.meter.add(now, "frames_skipped")  # (the older one is never sent)
                    self._pending_image = (bytes(jpeg), mime)
                    return False
                self._pending_image = None
                return self._send_image(bytes(jpeg), mime, now)
            except Exception as error:
                self._fault("feed_image", error)
                return False

    def _send_image(self, jpeg: bytes, mime: str, now: float) -> bool:
        try:
            self.transport.send_image(jpeg, now, mime)
        except Exception as error:
            self._fault("send_image", error)
            self._lost(now, "send_image failed")
            return False
        self._last_image = now
        self.meter.add(now, "frames")
        return True

    def _flush_image(self, now: float) -> None:
        pending = self._pending_image
        if pending is None:
            return
        if self._conn != "open" or not self.enabled:
            self._pending_image = None
            return
        if now - self._last_image >= self.config.image_period_s - 1e-9:
            self._pending_image = None
            self._send_image(pending[0], pending[1], now)

    def set_context(self, context: ContextSnapshot | str) -> None:
        """The duck's latest state, for the model's next status update."""
        with self._lock:
            self._context = context

    def set_vocabulary(self, phrases: Iterable[str]) -> None:
        """The words (the duck's name, the family's names) the transcriber is biased toward from the
        next connection on (``SessionConfig.custom_vocabulary`` on: the SDK's ``custom_vocabulary`` of
        the input transcription). A connection that is open keeps what it was opened with."""
        with self._lock:
            self._vocabulary = tuple(str(p) for p in phrases if str(p).strip())

    def send_note(self, text: str, *, about: str = "note") -> bool:
        """One short text to the model **now**, apart from the status (which waits for its period):
        the same ``send_client_content`` with ``turn_complete`` False, so the server takes it as
        context and does not answer it. For what is only worth saying at the moment it happens (the
        duck's own ear heard its name). It goes only while the cloud is on and the line is up: a
        note that cannot go now is dropped, never kept for later (a stale one would mislead).
        Returns True when it went up."""
        with self._lock:
            try:
                now = self.clock()
                if not self.enabled or self.state != ONLINE or self._conn != "open":
                    return False
                try:
                    self.transport.send_text(text, now, turn_complete=False)
                except Exception as error:
                    self._fault("send_text", error)
                    self._lost(now, "send_text failed")
                    return False
                self.notes_sent += 1
                self.meter.add(now, "text_updates")
                self._note("note", about=about, text=text)
                return True
            except Exception as error:
                self._fault("send_note", error)
                return False

    # ── the service loop ──────────────────────────────────────────────────────

    def pump(self, now: float | None = None) -> None:
        """Read what arrived, keep the connection, send the status text. Call it often (the
        microphone's ticks do, through :meth:`attach_mic`)."""
        with self._lock:
            try:
                now = self.clock() if now is None else float(now)
                if not self.enabled:
                    return
                if self._conn != "closed":
                    try:
                        arrived = self.transport.poll(now)
                    except Exception as error:
                        self._fault("poll", error)
                        self._lost(now, "poll failed")
                        arrived = []
                    for event in arrived:
                        self._handle_event(event, now)
                over = self.meter.over_cap()
                if over is not None and self.state != CAPPED:
                    self._shutdown_connection(now, f"cap {over}")
                    self._hold.clear()
                    self._set_state(CAPPED, now)
                    self._note("capped", cap=over)
                if self.state == CAPPED:
                    return
                if self._reconnect_now and self._conn == "open":
                    self._reconnect_now = False
                    self._shutdown_connection(now, "go_away")
                    self._retry_at = now
                    self.meter.add(now, "reconnects")
                    self._set_state(RECONNECTING, now)
                if self._conn == "closed" and now >= self._retry_at:
                    self._open(now)
                self._flush_image(now)
                self._send_context(now)
            except Exception as error:
                self._fault("pump", error)

    # ── what comes back ───────────────────────────────────────────────────────

    def take_calls(self, limit: int | None = None) -> list[QueuedCall]:
        """The queued tool calls, oldest first, removed from the queue."""
        with self._lock:
            out: list[QueuedCall] = []
            while self._queue and (limit is None or len(out) < limit):
                out.append(self._queue.popleft())
            return out

    def peek_calls(self) -> list[QueuedCall]:
        with self._lock:
            return list(self._queue)

    @property
    def sending(self) -> dict[str, bool]:
        """What is leaving the machine right now."""
        with self._lock:
            live = self.enabled and self._conn == "open" and self.state == ONLINE
            return {"audio": live, "video": live, "text": live}

    # ── events ────────────────────────────────────────────────────────────────

    def _handle_event(self, event: E.LiveEvent, now: float) -> None:
        kind = event.kind
        self.counts[kind] = self.counts.get(kind, 0) + 1
        data = event.data
        if kind == E.OPENED:
            self._on_opened(event, now)
        elif kind == E.INPUT_TEXT:
            mark = self.gate.mark_text(event.t)
            self.last_text_at = event.t
            self._note("heard", t=event.t, text=str(data.get("text", "")), **mark.view())
        elif kind == E.OUTPUT_TEXT:
            self._note("said", t=event.t, text=str(data.get("text", "")))
        elif kind == E.AUDIO_OUT:
            self.audio_out_bytes += int(data.get("bytes") or 0)  # (counted, never played)
        elif kind == E.TOOL_CALL:
            self._on_tool_call(event, now)
        elif kind == E.TOOL_CANCEL:
            self._note("tool_cancel", t=event.t, ids=list(data.get("ids") or []))
        elif kind == E.VAD:
            self.last_vad = {
                "type": data.get("type"),
                "offset_s": data.get("offset_s"),
                "t": event.t,
            }
            self._note("vad", t=event.t, type=data.get("type"), offset_s=data.get("offset_s"))
        elif kind == E.INTERRUPTED:
            self._note("interrupted", t=event.t)
        elif kind in (E.GENERATION_COMPLETE, E.TURN_COMPLETE):
            pass  # (counted above)
        elif kind == E.USAGE:
            self.meter.usage(now, data)
        elif kind == E.RESUMPTION:
            handle = data.get("handle")
            if data.get("resumable") and isinstance(handle, str) and handle:
                self._handle = handle
        elif kind == E.GO_AWAY:
            self._reconnect_now = True
            self._note("go_away", t=event.t, time_left_s=data.get("time_left_s"))
        elif kind == E.CLOSED:
            self._on_closed(event, now)
        elif kind == E.ERROR:
            self._on_error(event, now)

    def _on_opened(self, event: E.LiveEvent, now: float) -> None:
        self._conn = "open"
        self._opened_at = now  # (a connection that drops at once is not a recovery: see _on_closed)
        self._quota_fails = 0
        self._set_state(ONLINE, now)
        resumed = bool(event.data.get("resumed"))
        if resumed:
            self._resumes += 1
        self._note("opened", t=event.t, connection=self._connect_n, resumed=resumed)
        self._context_sent = None  # (the first status goes up at once, resumed or not)
        self._context_sent_at = -1e9
        held, self._hold = list(self._hold), deque(maxlen=self._hold.maxlen)
        for item in held:  # (what the gap kept, oldest first, before anything new)
            try:
                self.transport.send_audio(item.pcm, now)
            except Exception as error:
                self._fault("send_audio", error)
                self._lost(now, "send_audio failed")
                return
            self.audio_sent_s += len(item.pcm) / 32000.0
            self.meter.add(now, "audio_s", len(item.pcm) / 32000.0)

    def _on_tool_call(self, event: E.LiveEvent, now: float) -> None:
        mark = self.gate.mark_call(event.t)
        for item in event.data.get("calls") or []:
            parsed: ParsedCall = parse_call(str(item.get("name") or ""), item.get("args"))
            self._seq += 1
            call = QueuedCall(
                seq=self._seq,
                id=str(item.get("id") or f"call-{self._seq}"),
                name=parsed.name,
                args=parsed.args,
                raw=parsed.raw,
                problems=parsed.problems,
                off_bank=parsed.off_bank,
                t=event.t,
                wall=self._wall(),
                connection=self._connect_n,
                mark=mark,
                echo_suspect=mark.echo_only,
            )
            self.calls_total += 1
            self.meter.add(now, "tool_calls")
            if len(self._queue) >= self.config.call_queue_max:
                self._queue.popleft()
                self.calls_dropped += 1
            self._queue.append(call)
            self._note("call", **call.view())
            response: Mapping[str, Any] = {"status": "queued"}
            if self._responder is not None:
                try:
                    response = self._responder(call)
                except Exception as error:
                    self._fault("responder", error)
            try:
                self.transport.send_tool_response(call.id, call.name, dict(response), SILENT, now)
            except Exception as error:
                self._fault("send_tool_response", error)

    def _on_closed(self, event: E.LiveEvent, now: float) -> None:
        kind = str(event.data.get("kind") or E.NETWORK)
        reason = redact(event.data.get("reason") or "")
        self._conn = "closed"
        try:
            self.transport.close()
        except Exception as error:
            self._fault("transport.close", error)
        cfg = self.config
        if self._opened_at is not None and now - self._opened_at >= cfg.stable_after_s:
            # it stayed up long enough: the old failures are over, this one starts a new burst
            self._fails = 0
            self._burst_errors = 0
            self._burst_logged = False
        self._opened_at = None
        self.last_error = {"kind": kind, "t": round(event.t, 3), "reason": reason[:200]}
        if kind == E.QUOTA:
            self._quota_fails += 1
            delay = min(cfg.quota_backoff_max_s, cfg.quota_backoff_s * 2 ** (self._quota_fails - 1))
            self.meter.add(now, "quota_errors")
            self._set_state(QUOTA, now)
        elif kind in (E.AUTH, E.NO_KEY):
            delay = cfg.auth_retry_s
            self.meter.add(now, "errors")
            self._set_state(NO_KEY if kind == E.NO_KEY else DOWN, now)
        else:
            self._fails += 1
            delay = min(cfg.backoff_max_s, cfg.backoff_base_s * self._fails)
            if self._fails >= cfg.drop_handle_after_fails and self._handle is not None:
                self._handle = None  # (a stale handle fails the same way every time)
            self.meter.add(now, "errors")
            self.meter.add(now, "reconnects")
            self._set_state(RECONNECTING if self._fails < cfg.down_after_fails else DOWN, now)
        self._retry_at = now + delay
        self._note("closed", t=event.t, error=kind, reason=reason[:200], retry_in_s=round(delay, 2))
        self._burst_errors += 1
        if not self._burst_logged:  # (one line per burst of failures, not one per failure)
            self._burst_logged = True
            LOG.warning(
                "cloud connection closed (%s): %s; the duck carries on without it, retrying in %.1f s",
                kind,
                reason[:200],
                delay,
            )

    def _on_error(self, event: E.LiveEvent, now: float) -> None:
        kind = str(event.data.get("kind") or E.UNKNOWN)
        message = redact(event.data.get("message") or "")
        self.last_error = {"kind": kind, "t": round(event.t, 3), "reason": message[:200]}
        self.meter.add(now, "quota_errors" if kind == E.QUOTA else "errors")
        self._note("error", t=event.t, error=kind, message=message[:200])

    # ── connecting ────────────────────────────────────────────────────────────

    def _build_soul(self) -> str:
        soul = self._soul() if callable(self._soul) else self._soul
        return str(soul)

    def _open(self, now: float) -> None:
        resuming = self._handle is not None and self._soul_text is not None
        if not resuming:
            self._soul_text = self._build_soul()
            self._handle = None
        setup = Setup(
            system_instruction=self._soul_text or "",
            tools=tools_config(),
            model=self.config.model,
            resume_handle=self._handle,
            media_resolution=self.config.media_resolution,
            transcription_languages=tuple(self.config.transcription_languages),
            custom_vocabulary=self._vocabulary if self.config.custom_vocabulary else (),
        )
        self._connect_n += 1
        self._conn = "opening"
        if self.state not in (RECONNECTING, DOWN, QUOTA, NO_KEY):
            self._set_state(CONNECTING if self._connect_n == 1 else RECONNECTING, now)
        self._note("open", connection=self._connect_n, resume=resuming)
        try:
            self.transport.open(setup, now)
        except Exception as error:
            self._fault("open", error)
            self._lost(now, "open failed")

    def _lost(self, now: float, why: str) -> None:
        """The transport failed us (an exception, not an event): treat the line as dropped."""
        if self._conn != "closed" and self.enabled:
            self._on_closed(E.LiveEvent(E.CLOSED, now, {"kind": E.NETWORK, "reason": why}), now)

    def _shutdown_connection(self, now: float, why: str) -> None:
        if self._conn == "open":
            try:
                self.transport.send_audio_end(now)
            except Exception as error:
                self._fault("send_audio_end", error)
        if self._conn != "closed":
            try:
                self.transport.close()
            except Exception as error:
                self._fault("transport.close", error)
        self._conn = "closed"
        self._note("shutdown", why=why)

    # ── the status text going up ──────────────────────────────────────────────

    def _send_context(self, now: float) -> None:
        if self._conn != "open" or self._context is None:
            return
        context = self._context
        text = context if isinstance(context, str) else context.text()
        if text == self._context_sent:
            return
        if now - self._context_sent_at < self.config.context_period_s - 1e-9:
            return
        try:
            self.transport.send_text(text, now, turn_complete=False)
        except Exception as error:
            self._fault("send_text", error)
            self._lost(now, "send_text failed")
            return
        self._context_sent = text
        self._context_sent_at = now
        self.meter.add(now, "text_updates")

    # ── status, log ───────────────────────────────────────────────────────────

    def _set_state(self, state: str, now: float) -> None:
        if state != self.state:
            self.state = state
            self._state_since = now
            self._note("state", state=state)

    def _note(self, what: str, **fields: Any) -> None:
        """One line of the log (``kind`` is what happened; a field never overrides it)."""
        entry: dict[str, Any] = dict(fields)
        entry["kind"] = what
        entry.setdefault("t", round(self.clock(), 3))
        self.log.append(entry)

    def _fault(self, where: str, error: BaseException) -> None:
        self.faults += 1
        self.last_fault = redact(f"{where}: {type(error).__name__}: {error}")[:300]
        try:
            self.log.append({"kind": "fault", "t": round(self.clock(), 3), "where": where})
        except Exception:  # pragma: no cover
            pass

    @staticmethod
    def _words(state: str) -> str:
        try:
            return msg(f"neuro.live.status.{state}")
        except Exception:  # (a catalog problem must not break the status)
            return state

    def _error_words(self, retry_in: float | None) -> Any:
        """Plain words for what went wrong while the cloud is reconnecting or down (``None``
        otherwise): which kind of failure, how many times in a row, when the next try is."""
        if self.state not in (RECONNECTING, DOWN) or self.last_error is None:
            return None
        kind = str(self.last_error.get("kind") or E.UNKNOWN)
        if kind not in _ERROR_WORD_KINDS:
            kind = E.UNKNOWN
        try:
            return msg(
                f"neuro.live.err.{kind}",
                n=max(1, self._burst_errors),
                s=int(round(retry_in or 0)),
            )
        except Exception:  # (a catalog problem must not break the status)
            return None

    def status(self) -> dict[str, Any]:
        """What the page's status line shows: on or off, what state, what is leaving, the
        words for it (a catalog message), the numbers."""
        with self._lock:
            now = self.clock()
            sending = self.sending
            retry_in = None
            if self.enabled and self._conn == "closed" and self.state != CAPPED:
                retry_in = max(0.0, round(self._retry_at - now, 2))
            return {
                "enabled": self.enabled,
                "state": self.state,
                "words": self._words(self.state),
                "understands_speech": self.state == ONLINE,
                "sending": sending,
                "model": self.config.model,
                "since_s": round(now - self._state_since, 2),
                "connections": self._connect_n,
                "resumes": self._resumes,
                "retry_in_s": retry_in,
                "failures_in_a_row": self._burst_errors,
                "error_words": self._error_words(retry_in),
                "last_error": None if self.last_error is None else dict(self.last_error),
                "calls": {
                    "queued": len(self._queue),
                    "total": self.calls_total,
                    "dropped": self.calls_dropped,
                },
                "echo": self.gate.status(),
                "usage": self.meter.status(now),
                "audio_sent_s": round(self.audio_sent_s, 2),
                "notes_sent": self.notes_sent,
                "audio_out_discarded_bytes": self.audio_out_bytes,
                "events": dict(sorted(self.counts.items())),
                "faults": self.faults,
                "last_fault": self.last_fault,
            }


__all__ = [
    "CAPPED",
    "CONNECTING",
    "DOWN",
    "NO_KEY",
    "OFF",
    "ONLINE",
    "QUOTA",
    "RECONNECTING",
    "SILENT",
    "STATES",
    "LiveSession",
    "QueuedCall",
    "SessionConfig",
    "rms_dbfs",
]
