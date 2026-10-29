"""Cassettes: a real Live session recorded once, replayed any number of times.

A cassette holds what the server *sent back* in one scenario run (tool calls, transcripts, the
server's voice activity, token counts, closings: :class:`~duckmind.live.events.LiveEvent`) with
the time of each on the scenario's clock, and **by reference** what was sent up — fixture ids
(the scripted lines of ``tests/fixtures/voice``, the pictures of ``tests/fixtures/live/eye``)
with their hashes, positions, seeds, the model, the tool schema, the soul — never raw audio or
video. The key is the **scenario hash**: the SHA-256 of those inputs in a canonical form
(:func:`scenario_hash`). Change a fixture, a position, a seed, the model, a tool declaration or the
soul and the hash changes; a replay then refuses (:class:`CassetteMismatch`) instead of quietly
playing answers to a different question.

* :class:`RecordingTransport` wraps any transport (the real one) and records what it returns;
* :class:`ReplayTransport` is a transport that plays a cassette back on the same timeline: an
  event comes out when the session's clock has passed the moment it came in, counted from the
  session's first ``open``. It sends nothing anywhere.

Never stored: the API key (the cassette is scanned for anything key-shaped before it is
written), the server's resumption handles, the model's own voice (only how many bytes), and any
message text that was not redacted.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from duckmind.live import events as E
from duckmind.live.keys import find_keys, redact
from duckmind.live.tools import schema_hash
from duckmind.live.transport import SentLog, Setup, Transport

CASSETTE_FORMAT = 1


class CassetteMismatch(Exception):
    """The scenario asked for is not the one the cassette recorded."""


class KeyInCassette(Exception):
    """Something key-shaped was about to be written: nothing is."""


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def scenario_hash(inputs: Mapping[str, Any]) -> str:
    """The cassette's key: SHA-256 of the scenario inputs in canonical form."""
    return hashlib.sha256(canonical(dict(inputs)).encode("utf-8")).hexdigest()


def diff_inputs(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[str]:
    """Which top-level parts of two input sets differ (for the message of a mismatch)."""
    keys = sorted(set(a) | set(b))
    return [k for k in keys if canonical(a.get(k)) != canonical(b.get(k))]


@dataclass
class Cassette:
    scenario_id: str
    inputs: dict[str, Any]
    events: list[E.LiveEvent]
    scenario_hash: str = ""
    complete: bool = True
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.scenario_hash:
            self.scenario_hash = scenario_hash(self.inputs)

    # ── files ─────────────────────────────────────────────────────────────────

    def to_json(self) -> dict[str, Any]:
        return {
            "format": CASSETTE_FORMAT,
            "scenario_id": self.scenario_id,
            "scenario_hash": self.scenario_hash,
            "complete": self.complete,
            "inputs": self.inputs,
            "meta": self.meta,
            "events": [e.to_json() for e in self.events],
        }

    def dumps(self) -> str:
        text = json.dumps(self.to_json(), ensure_ascii=False, indent=1) + "\n"
        if find_keys(text):
            raise KeyInCassette("something key-shaped is in the cassette; not written")
        return text

    def save(self, path: Path | str) -> int:
        """Write the cassette (never with a key in it); returns its size in bytes."""
        text = self.dumps()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(target)
        return len(text.encode("utf-8"))

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> Cassette:
        if int(raw.get("format", 0)) != CASSETTE_FORMAT:
            raise CassetteMismatch(
                f"cassette format {raw.get('format')!r} is not {CASSETTE_FORMAT}"
            )
        out = cls(
            scenario_id=str(raw["scenario_id"]),
            inputs=dict(raw["inputs"]),
            events=[E.LiveEvent.from_json(e) for e in raw.get("events", [])],
            scenario_hash=str(raw["scenario_hash"]),
            complete=bool(raw.get("complete", True)),
            meta=dict(raw.get("meta") or {}),
        )
        if scenario_hash(out.inputs) != out.scenario_hash:
            raise CassetteMismatch(
                f"cassette {out.scenario_id!r}: its inputs do not hash to its scenario hash "
                "(the file was edited)"
            )
        return out

    @classmethod
    def load(cls, path: Path | str) -> Cassette:
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))

    # ── reading ───────────────────────────────────────────────────────────────

    def of_kind(self, kind: str) -> list[E.LiveEvent]:
        return [e for e in self.events if e.kind == kind]


# ── recording ──────────────────────────────────────────────────────────────────────


class RecordingTransport:
    """Wraps a transport and records the events it returns, timed from its first ``open``."""

    def __init__(self, inner: Transport, scenario_id: str, inputs: Mapping[str, Any]) -> None:
        self.inner = inner
        self.name = f"recording:{getattr(inner, 'name', '?')}"
        self.scenario_id = scenario_id
        self.inputs = dict(inputs)
        self._t0: float | None = None
        self.events: list[E.LiveEvent] = []
        self.audio_out_bytes = 0
        self.audio_out_span: list[float] = []
        self.opens: list[dict[str, Any]] = []

    # sends go straight through
    def open(self, setup: Setup, now: float) -> None:
        if self._t0 is None:
            self._t0 = now
        self.opens.append(
            {"t": round(now - self._t0, 3), "resume": setup.resume_handle is not None}
        )
        self.inner.open(setup, now)

    def send_audio(self, pcm: bytes, now: float) -> None:
        self.inner.send_audio(pcm, now)

    def send_audio_end(self, now: float) -> None:
        self.inner.send_audio_end(now)

    def send_image(self, jpeg: bytes, now: float, mime: str = "image/jpeg") -> None:
        self.inner.send_image(jpeg, now, mime)

    def send_text(self, text: str, now: float, *, turn_complete: bool = False) -> None:
        self.inner.send_text(text, now, turn_complete=turn_complete)

    def send_tool_response(
        self, call_id: str, name: str, response: dict[str, Any], scheduling: str, now: float
    ) -> None:
        self.inner.send_tool_response(call_id, name, response, scheduling, now)

    def close(self) -> None:
        self.inner.close()

    def poll(self, now: float) -> list[E.LiveEvent]:
        out = self.inner.poll(now)
        t0 = self._t0 if self._t0 is not None else now
        for event in out:
            if event.kind == E.AUDIO_OUT:  # (the model's own voice: counted, not kept)
                self.audio_out_bytes += int(event.data.get("bytes") or 0)
                self.audio_out_span = [
                    self.audio_out_span[0] if self.audio_out_span else event.t - t0,
                    event.t - t0,
                ]
                continue
            data = {k: _clean(v) for k, v in event.data.items() if k not in E.NEVER_STORED}
            self.events.append(E.LiveEvent(event.kind, round(event.t - t0, 3), data))
        return out

    def cassette(self, *, complete: bool, meta: Mapping[str, Any] | None = None) -> Cassette:
        info: dict[str, Any] = {
            "opens": self.opens,
            "audio_out_bytes": self.audio_out_bytes,
            "audio_out_span_s": [round(x, 3) for x in self.audio_out_span],
        }
        info.update(meta or {})
        return Cassette(
            scenario_id=self.scenario_id,
            inputs=self.inputs,
            events=list(self.events),
            complete=complete,
            meta=info,
        )


def _clean(value: Any) -> Any:
    """Text that came off the wire is redacted again on its way into a cassette."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


# ── replay ─────────────────────────────────────────────────────────────────────────


class ReplayTransport:
    """Plays a cassette on the session's clock. Fails loudly (:class:`CassetteMismatch`) at once
    if ``expected_inputs`` do not hash to the cassette's key, and again at ``open`` if the model,
    the tool schema or the soul the session sends are not the recorded ones."""

    def __init__(
        self,
        cassette: Cassette,
        expected_inputs: Mapping[str, Any] | None = None,
        *,
        check_setup: bool = True,
    ) -> None:
        self.cassette = cassette
        self.name = "replay"
        self.failure: CassetteMismatch | None = None
        self.sent = SentLog()
        self._check_setup = check_setup
        if expected_inputs is not None:
            expected = scenario_hash(expected_inputs)
            if expected != cassette.scenario_hash:
                parts = ", ".join(diff_inputs(expected_inputs, cassette.inputs)) or "?"
                raise CassetteMismatch(
                    f"cassette {cassette.scenario_id!r} was recorded for other inputs "
                    f"(differs in: {parts}); record it again"
                )
        self._t0: float | None = None
        self._next = 0
        self._events = sorted(cassette.events, key=lambda e: e.t)
        self._open = False

    def open(self, setup: Setup, now: float) -> None:
        if self._check_setup:
            self._verify(setup)
        if self._t0 is None:
            self._t0 = now
        self._open = True
        self.sent.opens.append({"t": now, "resume": setup.resume_handle is not None})

    def _verify(self, setup: Setup) -> None:
        inputs = self.cassette.inputs
        problems: list[str] = []
        if inputs.get("model") not in (None, setup.model):
            problems.append("model")
        tool_schema = inputs.get("tool_schema")
        if isinstance(tool_schema, Mapping) and tool_schema.get("hash") != schema_hash():
            problems.append("tool_schema")
        soul = inputs.get("soul")
        if isinstance(soul, Mapping):
            digest = hashlib.sha256(setup.system_instruction.encode("utf-8")).hexdigest()
            if soul.get("sha256") != digest:
                problems.append("soul")
        if problems:
            self.failure = CassetteMismatch(
                f"cassette {self.cassette.scenario_id!r}: the session opened with another "
                f"{', '.join(problems)} than the one recorded"
            )
            raise self.failure

    def send_audio(self, pcm: bytes, now: float) -> None:
        self.sent.audio_frames += 1

    def send_audio_end(self, now: float) -> None:
        self.sent.audio_ends += 1

    def send_image(self, jpeg: bytes, now: float, mime: str = "image/jpeg") -> None:
        self.sent.images += 1

    def send_text(self, text: str, now: float, *, turn_complete: bool = False) -> None:
        self.sent.texts.append((now, text, turn_complete))

    def send_tool_response(
        self, call_id: str, name: str, response: dict[str, Any], scheduling: str, now: float
    ) -> None:
        self.sent.tool_responses.append(
            {"id": call_id, "name": name, "response": response, "scheduling": scheduling, "t": now}
        )

    def poll(self, now: float) -> list[E.LiveEvent]:
        if self._t0 is None or not self._open:
            return []
        out: list[E.LiveEvent] = []
        while self._next < len(self._events):
            event = self._events[self._next]
            if self._t0 + event.t > now:
                break
            self._next += 1
            out.append(E.LiveEvent(event.kind, self._t0 + event.t, dict(event.data)))
            if event.kind == E.CLOSED:
                self._open = False
                break
        return out

    def close(self) -> None:
        self.sent.closes += 1
        self._open = False

    @property
    def finished(self) -> bool:
        return self._next >= len(self._events)


__all__ = [
    "CASSETTE_FORMAT",
    "Cassette",
    "CassetteMismatch",
    "KeyInCassette",
    "RecordingTransport",
    "ReplayTransport",
    "canonical",
    "diff_inputs",
    "scenario_hash",
]
