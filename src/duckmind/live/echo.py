"""The echo gate: telling the duck's own voice from a person's, as far as the cloud link can.

The duck's speaker is a few centimetres from its microphone, so what it calls comes back into
the audio it sends up (LV0: :class:`~duckmind.simweb.neuro_mic.MicStream` flags every frame that
holds its own voice, ``MicFrame.echo``, and lists its vocal windows). The cloud model cannot be
told which is which, so it may transcribe the duck's own call as speech, or answer it. This gate
watches the audio that is sent and marks what comes back:

* **a transcript chunk** (the model's ``input_text``) is attributed to the *burst* of loud audio
  it most likely transcribes (loud frames with gaps under :data:`BURST_GAP_S` are one burst; the
  latest one, unless it is younger than :data:`FRESH_BURST_S` and an older one is in reach); it is
  ``echo_suspect`` when that burst holds any of the duck's own voice (``echo_ms`` > 0) **and the
  duck's own voice detection did not hear a person in that burst** (``rescued``: the same rescue
  the grounding gate gives a call, see :meth:`EchoGate.set_human_evidence`: a person who speaks
  while the duck calls is inside the flagged frames too, and the model then transcribes *them*).
  The numbers are kept next to the mark (``echo_ms``, ``clean_ms``, ``echo_dominant``: more echo
  than clean audio) so a later stage can be stricter or kinder; a chunk with no loud audio behind
  it at all is ``no_audio`` (made up);
* **a tool call** is ``echo_only`` when, since the last human-like (loud, not flagged) audio, the
  only loud audio sent was the duck's own, and that speech is not recent enough to explain the
  call (:data:`ANSWER_S`), or was already answered by an earlier call.

What the gate can not do by itself: a person who speaks *over* the duck's call is inside a flagged
frame too (the flag is the duck's, the mix is not separable), so a barge-in is told apart by the
local voice detection of LV2, which the host hands in as ``human_evidence``. Marks are logged;
refusing marked calls is LV2.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

SPEECH_FLOOR_DBFS = -50.0  # louder than the quiet room (-62) and a distant voice: "something"
FRAME_MS = 20.0
BURST_GAP_S = 0.4  # loud frames this close are one burst
LOOKBACK_S = 2.5  # a transcript is attributed to a burst no older than this
ANSWER_S = 2.5  # a call this soon after human speech is that speech's answer
ECHO_ONLY_MIN_MS = 100.0  # the duck's own voice that counts: at least this much
CAUSE_S = 4.0  # a call is "caused by" the latest transcript chunk when that came this recently
KEEP_BURSTS = 40
HUMAN_PAD_S = 0.1  # the local detector's speech runs this long past the burst's last loud frame
FRESH_BURST_S = 0.3  # a burst this young (or still going) is not yet what a transcript says: the
# service needs a moment after the audio, and the duck's own answer (a call) starts before the
# transcript of the words it answers arrives (live: 晚安 spoken at 332.6-334.0, the duck's call at
# 334.7, the transcript at 335.0: it was attributed to the call and labelled its echo)


@dataclass
class _Burst:
    t0: float
    last: float
    echo_ms: float = 0.0
    clean_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class TextMark:
    """The gate's verdict on one transcript chunk."""

    echo_suspect: bool
    echo_ms: float
    clean_ms: float
    echo_dominant: bool
    no_audio: bool
    # the burst holds the duck's own voice, but the duck's own detector heard a person in it:
    # the chunk is taken as the person's (``echo_suspect`` is False)
    rescued: bool = False

    def view(self) -> dict[str, Any]:
        return {
            "echo_suspect": self.echo_suspect,
            "echo_ms": round(self.echo_ms),
            "clean_ms": round(self.clean_ms),
            "echo_dominant": self.echo_dominant,
            "no_audio": self.no_audio,
            "rescued": self.rescued,
        }


@dataclass(frozen=True, slots=True)
class CallMark:
    """The gate's verdict on one tool-call message."""

    echo_only: bool
    since_human_s: float | None  # seconds since the last human-like audio (None: none yet)
    echo_ms_since_human: float
    answered: bool  # an earlier call message already followed that audio
    no_audio: bool  # nothing loud at all in the last LOOKBACK_S
    # the latest transcript chunk before the call (within CAUSE_S): what the call most likely
    # answers, and whether that chunk overlapped the duck's own voice (LV2 refuses such calls)
    cause_age_s: float | None = None
    cause_echo: bool = False

    def view(self) -> dict[str, Any]:
        return {
            "echo_only": self.echo_only,
            "since_human_s": None if self.since_human_s is None else round(self.since_human_s, 2),
            "echo_ms_since_human": round(self.echo_ms_since_human),
            "answered": self.answered,
            "no_audio": self.no_audio,
            "cause_age_s": None if self.cause_age_s is None else round(self.cause_age_s, 2),
            "cause_echo": self.cause_echo,
        }


class EchoGate:
    def __init__(
        self,
        *,
        floor_dbfs: float = SPEECH_FLOOR_DBFS,
        burst_gap_s: float = BURST_GAP_S,
        lookback_s: float = LOOKBACK_S,
        answer_s: float = ANSWER_S,
        human_evidence: Callable[[float, float], bool] | None = None,
    ) -> None:
        self._human = human_evidence
        self.floor = floor_dbfs
        self.gap = burst_gap_s
        self.lookback = lookback_s
        self.answer = answer_s
        self._bursts: deque[_Burst] = deque(maxlen=KEEP_BURSTS)
        self._last_human: float | None = None
        self._echo_ms_since_human = 0.0
        self._calls_since_human = 0
        self._last_loud: float | None = None
        self._last_text: tuple[float, TextMark] | None = None
        self.frames = 0
        self.echo_frames = 0
        self.texts_suspect = 0
        self.texts_rescued = 0
        self.calls_echo_only = 0

    def set_human_evidence(self, evidence: Callable[[float, float], bool] | None) -> None:
        """``evidence(t0, t1)``: did the duck's own voice detection hear a person speak between two
        moments (the session's clock)? What the grounding gate uses to take a call that looks like
        an echo after all."""
        self._human = evidence

    def _person_heard(self, t0: float, t1: float) -> bool:
        if self._human is None:
            return False
        try:
            return bool(self._human(t0, t1 + HUMAN_PAD_S))
        except Exception:  # (a fault in the detector never takes a mark away: it stays suspect)
            return False

    def on_frame(self, t: float, echo: bool, level_dbfs: float) -> None:
        """One audio frame that was sent (session time ``t``)."""
        self.frames += 1
        if echo:
            self.echo_frames += 1
        if level_dbfs <= self.floor:
            return
        burst = self._bursts[-1] if self._bursts else None
        if burst is None or t - burst.last > self.gap:
            burst = _Burst(t0=t, last=t)
            self._bursts.append(burst)
        burst.last = t
        self._last_loud = t
        if echo:
            burst.echo_ms += FRAME_MS
            self._echo_ms_since_human += FRAME_MS
        else:
            burst.clean_ms += FRAME_MS
            self._last_human = t
            self._echo_ms_since_human = 0.0
            self._calls_since_human = 0

    def _burst_for(self, now: float) -> _Burst | None:
        """The burst a transcript that arrives at ``now`` most likely says: the latest within the
        lookback that is older than :data:`FRESH_BURST_S`; when only a fresh one is there, that."""
        reach = [b for b in reversed(self._bursts) if now - b.last <= self.lookback]
        for burst in reach:
            if now - burst.last >= FRESH_BURST_S:
                return burst
        return reach[0] if reach else None

    def mark_text(self, now: float) -> TextMark:
        """A transcript chunk arrived at ``now``."""
        burst = self._burst_for(now)
        if burst is None:
            mark = TextMark(False, 0.0, 0.0, False, True)
            self._last_text = (now, mark)
            return mark
        suspect = burst.echo_ms > 0
        rescued = suspect and self._person_heard(burst.t0, burst.last)
        if rescued:
            suspect = False
            self.texts_rescued += 1
        if suspect:
            self.texts_suspect += 1
        mark = TextMark(
            echo_suspect=suspect,
            echo_ms=burst.echo_ms,
            clean_ms=burst.clean_ms,
            echo_dominant=burst.echo_ms > burst.clean_ms,
            no_audio=False,
            rescued=rescued,
        )
        self._last_text = (now, mark)
        return mark

    def mark_call(self, now: float) -> CallMark:
        """A tool-call message arrived at ``now``."""
        since = None if self._last_human is None else now - self._last_human
        answered = self._calls_since_human > 0
        no_audio = self._last_loud is None or now - self._last_loud > self.lookback
        echo_ms = self._echo_ms_since_human
        stale = since is None or since > self.answer or answered
        echo_only = echo_ms >= ECHO_ONLY_MIN_MS and stale
        self._calls_since_human += 1
        if echo_only:
            self.calls_echo_only += 1
        cause_age: float | None = None
        cause_echo = False
        if self._last_text is not None and now - self._last_text[0] <= CAUSE_S:
            cause_age = now - self._last_text[0]
            cause_echo = self._last_text[1].echo_suspect
        return CallMark(echo_only, since, echo_ms, answered, no_audio, cause_age, cause_echo)

    def status(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "echo_frames": self.echo_frames,
            "texts_suspect": self.texts_suspect,
            "texts_rescued": self.texts_rescued,
            "calls_echo_only": self.calls_echo_only,
        }


__all__ = ["CallMark", "EchoGate", "TextMark"]
