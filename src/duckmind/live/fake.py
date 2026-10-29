"""A rule-based fake Gemini, for the unit tests: the same interface as the real connection.

It is deterministic and needs no network and no audio understanding: it is told what is *said*
in the audio it will be sent (:class:`SpokenLine`: when, what words), and it "hears" a line only
if enough loud audio was really sent during it. Then it behaves like the model the duck's soul
asks for, by a few plain rules:

* the duck's name (or 小鴨) called → ``vocalize`` a ``greet`` (and ``feel`` happy, wants to talk);
* 你好 / 嗨 → ``vocalize`` a gentler ``greet`` and ``attend`` the speaker;
* 「這是X」 → ``recognize`` X (and ``remember`` it);
* 你看到什麼 → ``attend`` a thing the last picture shows (``image_labels``: picture hash → what
  it shows), or a doubtful ``feel`` when it shows nothing;
* 不要叫了 / 安靜 → ``feel`` with no wish to talk;
* anything else (another name, chat) → nothing.

``tag_counts`` makes the replies chattier on purpose (for the drive measurements of LV4): the
probabilities that a ``vocalize`` has 1, 2 or 3 tags (the extra ones a ``chirp`` and a ``coo``, spaced
as the gate wants), drawn from ``chat_seed`` and the call's number so that the same run is the same.
The 68 real recordings of LV1 have 30 ``vocalize`` calls and each has one tag: the default is that.

Everything arrives with the delays a real one has (the server's voice activity, the transcript a
moment after the speech, the tools a moment after that); ``hallucinate=True`` makes it also
"hear" loud audio that no line explains (the duck's own call coming back) and answer it — the
very mistake the echo gate exists for. Faults for the reconnect tests: ``outcomes`` (what each
successive open does: ``ok``, ``quota``, ``network``, …), ``drops`` and ``go_aways`` (times),
and resumption handles it only accepts back when it issued them.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from duckmind.live import events as E
from duckmind.live.session import rms_dbfs
from duckmind.live.transport import SentLog, Setup

FRAME_MS = 20
VAD_START_DELAY_S = 0.16
VAD_END_DELAY_S = 0.10
TEXT_DELAY_S = 0.30
REPLY_DELAY_S = 0.60
MIN_HEARD = 0.35  # of a line's length, the loud audio that must have been sent (speech has pauses)
HEAR_FLOOR_DBFS = -50.0
HALLUCINATE_AFTER_MS = 600

_INTRO = re.compile(r"這是(?!什麼|甚麼|誰)([^。！？!?，,\s]{1,12})")
_STOP = re.compile(r"不要叫|別叫|安靜|小聲")
_SEE = re.compile(r"你看到什麼|你看見什麼")
_HELLO = re.compile(r"你好|嗨|哈囉")


@dataclass(frozen=True)
class SpokenLine:
    """What is said in the audio the fake will be sent: ``t0``–``t1`` on the session clock."""

    t0: float
    t1: float
    text: str
    who: str = ""

    @property
    def duration(self) -> float:
        return max(0.02, self.t1 - self.t0)


@dataclass
class _Due:
    t: float
    event: E.LiveEvent


@dataclass
class _Line:
    spec: SpokenLine
    loud_ms: float = 0.0
    started: bool = False
    done: bool = False


@dataclass
class FakeGemini:
    duck_name: str = "蛋黃"
    lines: Sequence[SpokenLine] = ()
    image_labels: Mapping[str, Sequence[str]] = field(
        default_factory=dict
    )  # sha256 → what it shows
    hallucinate: bool = False
    tag_counts: Sequence[float] = (1.0,)  # P(1 tag), P(2 tags), P(3 tags) of a vocalize
    chat_seed: int = 0
    outcomes: Sequence[str] = ()  # what each open does, in order (then "ok")
    drops: Sequence[float] = ()
    go_aways: Sequence[float] = ()
    resume_every_s: float = 5.0
    name: str = "fake"
    sent: SentLog = field(default_factory=SentLog)
    seen: list[str] = field(default_factory=list)  # what the latest picture shows

    def __post_init__(self) -> None:
        self._lines = [_Line(spec) for spec in sorted(self.lines, key=lambda s: s.t0)]
        self._due: list[_Due] = []
        self._open = False
        self._opens = 0
        self._handles: list[str] = []
        self._last_resume = -1e9
        self._drops = sorted(self.drops)
        self._aways = sorted(self.go_aways)
        self._call_n = 0
        self._noise_ms = 0.0
        self._quiet_ms = 0.0
        self._noise_done = False
        self._turns = 0

    # ── Transport ─────────────────────────────────────────────────────────────

    def open(self, setup: Setup, now: float) -> None:
        self._opens += 1
        self.sent.opens.append(
            {
                "t": now,
                "resume": setup.resume_handle is not None,
                "model": setup.model,
                "tools": [
                    d["name"] for t in setup.tools for d in t.get("function_declarations", [])
                ],
                "modalities": list(setup.response_modalities),
                "instruction_chars": len(setup.system_instruction),
                "vocabulary": list(getattr(setup, "custom_vocabulary", ())),
            }
        )
        outcome = self.outcomes[self._opens - 1] if self._opens <= len(self.outcomes) else "ok"
        if setup.resume_handle is not None and setup.resume_handle not in self._handles:
            outcome = "network" if outcome == "ok" else outcome
        if outcome != "ok":
            self._due.append(
                _Due(
                    now + 0.05,
                    E.LiveEvent(
                        E.CLOSED, now + 0.05, {"kind": outcome, "reason": f"fake {outcome}"}
                    ),
                )
            )
            self._open = False
            return
        self._open = True
        resumed = setup.resume_handle is not None
        self._due.append(_Due(now + 0.05, E.LiveEvent(E.OPENED, now + 0.05, {"resumed": resumed})))
        self._issue_handle(now + 0.06)

    def send_audio(self, pcm: bytes, now: float) -> None:
        if not self._open:
            return
        self.sent.audio_frames += 1
        if rms_dbfs(pcm) <= HEAR_FLOOR_DBFS:
            self._quiet_ms += FRAME_MS
            if self._quiet_ms >= 400:
                self._noise_ms = 0.0
                self._noise_done = False
            return
        self._quiet_ms = 0.0
        spoken = False
        for line in self._lines:
            if line.spec.t0 - 0.05 <= now <= line.spec.t1 + 0.05:
                spoken = True
                line.loud_ms += FRAME_MS
                if not line.started:
                    line.started = True
                    self._schedule(
                        now + VAD_START_DELAY_S,
                        E.VAD,
                        {"type": "ACTIVITY_START", "offset_s": round(now, 3)},
                    )
                    self._schedule(
                        line.spec.t1 + VAD_END_DELAY_S,
                        E.VAD,
                        {"type": "ACTIVITY_END", "offset_s": round(line.spec.t1, 3)},
                    )
        if not spoken and self.hallucinate and not self._noise_done:
            self._noise_ms += FRAME_MS
            if self._noise_ms >= HALLUCINATE_AFTER_MS:
                self._noise_done = True
                self._say(
                    "（鴨叫聲）",
                    now + 0.2,
                    calls=[
                        self._call(
                            "vocalize", {"tags": [{"tag": "greet", "level": 0.7, "at_ms": 0}]}
                        )
                    ],
                )

    def send_audio_end(self, now: float) -> None:
        self.sent.audio_ends += 1

    def send_image(self, jpeg: bytes, now: float, mime: str = "image/jpeg") -> None:
        if not self._open:
            return
        self.sent.images += 1
        self.seen = list(self.image_labels.get(hashlib.sha256(jpeg).hexdigest(), ()))

    def send_text(self, text: str, now: float, *, turn_complete: bool = False) -> None:
        if self._open:
            self.sent.texts.append((now, text, turn_complete))

    def send_tool_response(
        self, call_id: str, name: str, response: dict[str, Any], scheduling: str, now: float
    ) -> None:
        self.sent.tool_responses.append(
            {"id": call_id, "name": name, "response": response, "scheduling": scheduling, "t": now}
        )

    def poll(self, now: float) -> list[E.LiveEvent]:
        self._advance(now)
        due = sorted((d for d in self._due if d.t <= now), key=lambda d: d.t)
        self._due = [d for d in self._due if d.t > now]
        out: list[E.LiveEvent] = []
        for item in due:
            if item.event.kind == E.CLOSED:
                self._open = False
            out.append(item.event)
        return out

    def close(self) -> None:
        self.sent.closes += 1
        self._open = False
        self._due = [d for d in self._due if d.event.kind not in (E.OPENED, E.RESUMPTION)]

    # ── the model ─────────────────────────────────────────────────────────────

    def _advance(self, now: float) -> None:
        if not self._open:
            return
        if now - self._last_resume >= self.resume_every_s:
            self._issue_handle(now)
        while self._drops and self._drops[0] <= now:
            self._drops.pop(0)
            self._schedule(now, E.CLOSED, {"kind": E.NETWORK, "reason": "fake drop"})
        while self._aways and self._aways[0] <= now:
            self._aways.pop(0)
            self._schedule(now, E.GO_AWAY, {"time_left_s": 3.0})
        for line in self._lines:
            if line.done or now < line.spec.t1 + VAD_END_DELAY_S:
                continue
            line.done = True
            if line.loud_ms < MIN_HEARD * line.spec.duration * 1000.0:
                continue  # (not enough of it was sent: it was not heard)
            self._say(
                line.spec.text, line.spec.t1 + TEXT_DELAY_S, calls=self._react(line.spec.text)
            )

    def react(self, text: str) -> list[dict[str, Any]]:
        """The tool calls the model makes to a line it heard (the rules of the module doc)."""
        return self._react(text)

    def _tags(self, lead: float) -> list[dict[str, Any]]:
        """The tags of one ``vocalize``: a greet, and sometimes a chirp and a coo after it."""
        tags: list[dict[str, Any]] = [{"tag": "greet", "level": lead, "at_ms": 0}]
        weights = list(self.tag_counts)
        if len(weights) > 1:
            pick = random.Random(f"{self.chat_seed}:{self._call_n}").random() * sum(weights)
            count = 1
            for k, w in enumerate(weights, start=1):
                if pick < w:
                    count = k
                    break
                pick -= w
            extra = [("chirp", 0.75, 700), ("coo", 0.6, 1500)]
            for tag, factor, at_ms in extra[: max(0, min(2, count - 1))]:
                tags.append({"tag": tag, "level": round(lead * factor, 3), "at_ms": at_ms})
        return tags

    def _react(self, text: str) -> list[dict[str, Any]]:
        intro = _INTRO.search(text)
        if intro:
            who = intro.group(1)
            return [
                self._call("recognize", {"name": who}),
                self._call("remember", {"note": f"家人介紹了{who}"}),
            ]
        if _SEE.search(text):
            if self.seen:
                return [self._call("attend", {"target_kind": "thing", "hint": self.seen[0]})]
            return [self._call("feel", {"valence": 0.0, "talk": 0.1})]
        if _STOP.search(text):
            return [self._call("feel", {"valence": -0.3, "talk": 0.0})]
        if self.duck_name and self.duck_name in text or "小鴨" in text:
            return [
                self._call("vocalize", {"tags": self._tags(0.8)}),
                self._call("feel", {"valence": 0.4, "talk": 0.7}),
            ]
        if _HELLO.search(text):
            return [
                self._call("vocalize", {"tags": self._tags(0.6)}),
                self._call("attend", {"target_kind": "person", "hint": "說話的人"}),
            ]
        return []

    def _say(self, text: str, t: float, *, calls: list[dict[str, Any]]) -> None:
        half = max(1, len(text) // 2)
        pieces = [text[:half], text[half:]] if len(text) > 3 else [text]
        for k, piece in enumerate(pieces):
            self._schedule(t + 0.15 * k, E.INPUT_TEXT, {"text": piece})
        t_reply = t + REPLY_DELAY_S
        if calls:
            self._schedule(t_reply, E.TOOL_CALL, {"calls": calls})
        self._turns += 1
        self._schedule(t_reply + 0.03, E.GENERATION_COMPLETE, {})
        self._schedule(
            t_reply + 0.05,
            E.USAGE,
            {
                "total": 900 + 40 * self._turns,
                "prompt": 800 + 40 * self._turns,
                "response": 25,
                "thoughts": 100,
                "by_modality": {"TEXT": 500, "AUDIO": 250},
            },
        )
        self._schedule(t_reply + 0.05, E.TURN_COMPLETE, {})

    def _call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        self._call_n += 1
        return {"id": f"fake-{self._call_n}", "name": name, "args": args}

    def _schedule(self, t: float, kind: str, data: dict[str, Any]) -> None:
        self._due.append(_Due(t, E.LiveEvent(kind, t, data)))

    def _issue_handle(self, t: float) -> None:
        self._last_resume = t
        handle = f"fake-handle-{len(self._handles) + 1}"
        self._handles.append(handle)
        self._schedule(t, E.RESUMPTION, {"resumable": True, "handle": handle})


__all__ = ["FakeGemini", "SpokenLine"]
