"""The messages a Live connection produces, in one shape that does not depend on the SDK.

A transport (the real one, the fake Gemini, a cassette replay) turns what the server sends into
:class:`LiveEvent` values; the session and the cassette only ever see these. ``t`` is seconds on
the clock of whoever made the event (the session's clock: monotonic seconds live, the scenario's
seconds in a recording or a replay).

Kinds and their ``data``:

========================  ==========================================================
``opened``                a connection is up (``resumed``: it continued a session)
``input_text``            a piece of what the microphone's speech was heard as (``text``)
``output_text``           a piece of what the model said, as text (``text``): never spoken
``audio_out``             the model's own voice, counted and thrown away (``bytes``)
``tool_call``             ``calls``: [{``id``, ``name``, ``args``}] — one message, one batch
``tool_cancel``           ``ids`` of calls the server dropped (an interruption)
``vad``                   the server's voice activity (``type``: START | END, ``offset_s``)
``interrupted``           the model's turn was cut by new speech
``generation_complete``   the model finished generating
``turn_complete``         the model's turn is over
``usage``                 token counts of a turn (``total``, ``prompt``, ``response``,
                          ``thoughts``, ``by_modality``: {AUDIO: n, …})
``resumption``            ``resumable`` and, in memory only, ``handle`` (never stored)
``go_away``               the connection ends soon (``time_left_s``)
``closed``                the connection ended (``kind``, ``code``, ``reason``)
``error``                 a failure (``kind``, ``message``, redacted)
========================  ==========================================================

Error kinds: ``quota`` (RESOURCE_EXHAUSTED, 429), ``auth`` (a key the server refuses),
``no_key`` (none to send), ``network``, ``server`` (5xx, 1011), ``bad_request`` (the server does
not take what was sent), ``unknown``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

OPENED = "opened"
INPUT_TEXT = "input_text"
OUTPUT_TEXT = "output_text"
AUDIO_OUT = "audio_out"
TOOL_CALL = "tool_call"
TOOL_CANCEL = "tool_cancel"
VAD = "vad"
INTERRUPTED = "interrupted"
GENERATION_COMPLETE = "generation_complete"
TURN_COMPLETE = "turn_complete"
USAGE = "usage"
RESUMPTION = "resumption"
GO_AWAY = "go_away"
CLOSED = "closed"
ERROR = "error"
KINDS = (
    OPENED,
    INPUT_TEXT,
    OUTPUT_TEXT,
    AUDIO_OUT,
    TOOL_CALL,
    TOOL_CANCEL,
    VAD,
    INTERRUPTED,
    GENERATION_COMPLETE,
    TURN_COMPLETE,
    USAGE,
    RESUMPTION,
    GO_AWAY,
    CLOSED,
    ERROR,
)

QUOTA = "quota"
AUTH = "auth"
NO_KEY = "no_key"
NETWORK = "network"
SERVER = "server"
BAD_REQUEST = "bad_request"
UNKNOWN = "unknown"
ERROR_KINDS = (QUOTA, AUTH, NO_KEY, NETWORK, SERVER, BAD_REQUEST, UNKNOWN)

# what is never written into a cassette (a session's resumption token is a credential too)
NEVER_STORED = frozenset({"handle"})


@dataclass(frozen=True, slots=True)
class LiveEvent:
    kind: str
    t: float
    data: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"t": round(self.t, 3), "kind": self.kind}
        data = {k: v for k, v in self.data.items() if k not in NEVER_STORED}
        if data:
            out["data"] = data
        return out

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> LiveEvent:
        kind = str(raw["kind"])
        if kind not in KINDS:
            raise ValueError(f"unknown event kind {kind!r}")
        data = raw.get("data") or {}
        if not isinstance(data, dict):
            raise ValueError("event data must be an object")
        return cls(kind=kind, t=float(raw["t"]), data=dict(data))


__all__ = [
    "AUDIO_OUT",
    "AUTH",
    "BAD_REQUEST",
    "CLOSED",
    "ERROR",
    "ERROR_KINDS",
    "GENERATION_COMPLETE",
    "GO_AWAY",
    "INPUT_TEXT",
    "INTERRUPTED",
    "KINDS",
    "NETWORK",
    "NEVER_STORED",
    "NO_KEY",
    "OPENED",
    "OUTPUT_TEXT",
    "QUOTA",
    "RESUMPTION",
    "SERVER",
    "TOOL_CALL",
    "TOOL_CANCEL",
    "TURN_COMPLETE",
    "UNKNOWN",
    "USAGE",
    "VAD",
    "LiveEvent",
]
