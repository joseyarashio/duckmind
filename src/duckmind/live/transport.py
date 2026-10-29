"""The line between the session and whatever is on the other end.

:class:`Transport` is what the session talks to; it has three implementations that look the same
from here: the real Gemini Live connection (:mod:`duckmind.live.genai_transport`), the rule-based
fake (:mod:`duckmind.live.fake`) and a cassette replay (:mod:`duckmind.live.cassette`). It is
**synchronous and non-blocking**: the session sends things (they are queued, or counted and
dropped, never waited for) and *polls* for the events that arrived. A real transport keeps its
own network thread; the others answer from memory on the session's clock, which is what makes
every test deterministic.

One transport object serves one connection at a time: ``open`` starts one, ``close`` ends it, the
session decides when to open another (and passes the resumption handle it holds).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from duckmind.live import events as E

DEFAULT_MODEL = "gemini-3.8-live"
SDK_PIN_NOTE = "google-genai"  # the dependency the real transport needs (pyproject.toml)

# What the server was asked to do about turns (see VOICE-PLAN-v3 and LV1-live-session.md):
# the server's own voice activity detection ends the family's turn; the duck's brain answers
# with tools only.
# Which languages the transcriber is told the family speaks (BCP-47, the household's own first).
# Without the hint the service picks a language per sentence: 「過來這裡」 came back in simplified
# characters, 「小鴨」 as Spanish or Vietnamese words (live, 2026-10-04). robot-bridge's finding.
TRANSCRIPTION_LANGUAGES = ("zh-TW", "en-US")
VAD_START = "START_SENSITIVITY_HIGH"
VAD_END = "END_SENSITIVITY_LOW"
VAD_PREFIX_MS = 300
VAD_SILENCE_MS = 500


@dataclass(frozen=True)
class Setup:
    """Everything a connection is opened with."""

    system_instruction: str
    tools: list[dict[str, Any]]
    model: str = DEFAULT_MODEL
    resume_handle: str | None = None  # in memory only
    # gemini-3.8-live takes AUDIO only (TEXT is refused: 1007 "requested combination of response
    # modalities (TEXT) is not supported"): the duck asks for AUDIO, throws the audio away and
    # reads the transcription of what the model said
    response_modalities: tuple[str, ...] = ("AUDIO",)
    input_transcription: bool = True
    output_transcription: bool = True
    # language hint for both transcriptions (empty: the service guesses per sentence)
    transcription_languages: tuple[str, ...] = TRANSCRIPTION_LANGUAGES
    # words the family's speech is transcribed with a bias toward (the duck's name, the family's
    # names: the SDK's ``custom_vocabulary`` of the input transcription); empty: none
    custom_vocabulary: tuple[str, ...] = ()
    compression: bool = True  # sliding-window context compression (audio+video sessions would
    # end after 2 minutes without it)
    vad_start: str = VAD_START
    vad_end: str = VAD_END
    vad_prefix_ms: int = VAD_PREFIX_MS
    vad_silence_ms: int = VAD_SILENCE_MS
    media_resolution: str | None = None  # e.g. "MEDIA_RESOLUTION_LOW" for the duck's pictures


class Transport(Protocol):
    """See the module docstring. None of these may raise into the session's caller (the session
    catches anyway, but a transport that raises is a bug it reports as a fault)."""

    name: str

    def open(self, setup: Setup, now: float) -> None: ...

    def send_audio(self, pcm: bytes, now: float) -> None: ...

    def send_audio_end(self, now: float) -> None: ...

    def send_image(self, jpeg: bytes, now: float, mime: str = "image/jpeg") -> None: ...

    def send_text(self, text: str, now: float, *, turn_complete: bool = False) -> None: ...

    def send_tool_response(
        self, call_id: str, name: str, response: dict[str, Any], scheduling: str, now: float
    ) -> None: ...

    def poll(self, now: float) -> list[E.LiveEvent]: ...

    def close(self) -> None: ...


@dataclass
class SentLog:
    """What a transport that does not talk to anyone (fake, replay) was sent: for the tests."""

    audio_frames: int = 0
    audio_ends: int = 0
    images: int = 0
    texts: list[tuple[float, str, bool]] = field(default_factory=list)
    tool_responses: list[dict[str, Any]] = field(default_factory=list)
    opens: list[dict[str, Any]] = field(default_factory=list)
    closes: int = 0

    def view(self) -> dict[str, Any]:
        return {
            "audio_frames": self.audio_frames,
            "audio_ends": self.audio_ends,
            "images": self.images,
            "texts": len(self.texts),
            "tool_responses": len(self.tool_responses),
            "opens": len(self.opens),
            "closes": self.closes,
        }


def classify_error(code: object = None, status: object = None, text: object = "") -> str:
    """Which kind of failure this is (:data:`duckmind.live.events.ERROR_KINDS`), from whatever the
    SDK or the socket gave: an HTTP-like ``code``, a gRPC-like ``status`` word, the message."""
    words = f"{status or ''} {text or ''}".lower()
    number = code if isinstance(code, int) and not isinstance(code, bool) else None
    if (
        number == 429
        or "resource_exhausted" in words
        or "resource has been exhausted" in words
        or "quota" in words
        or "rate limit" in words
        or "too many requests" in words
    ):
        return E.QUOTA
    if (
        number in (401, 403)
        or "api key not valid" in words
        or "api_key_invalid" in words
        or "permission_denied" in words
        or "unauthenticated" in words
        or "invalid api key" in words
    ):
        return E.AUTH
    if "response modalities" in words or "not supported by the model" in words or number == 400:
        return E.BAD_REQUEST
    if number == 1007:
        return E.BAD_REQUEST
    if number is not None and (500 <= number < 600 or number in (1011, 1012, 1013)):
        return E.SERVER
    if number in (1001, 1002, 1005, 1006, 1015):
        return E.NETWORK
    if "unavailable" in words or "internal" in words or "deadline" in words:
        return E.SERVER
    if any(
        w in words for w in ("connect", "timed out", "timeout", "network", "reset", "eof", "dns")
    ):
        return E.NETWORK
    return E.UNKNOWN


__all__ = [
    "DEFAULT_MODEL",
    "TRANSCRIPTION_LANGUAGES",
    "SentLog",
    "Setup",
    "Transport",
    "classify_error",
]
