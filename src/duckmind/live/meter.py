"""The usage meter: what the cloud connection is costing, per minute.

Everything that leaves the machine is counted as it goes — seconds of microphone audio, pictures,
text updates — and everything the server reports back: tokens (from the server's own usage
messages, one per turn), errors, quota refusals and reconnects. Minutes are buckets of the
session's clock; :meth:`UsageMeter.status` is what the status dict shows (this minute, the last
full minute, the totals), and a **cap** (tokens, seconds of audio, or pictures) turns the cloud
off by itself (:meth:`UsageMeter.over_cap`).

The server reports tokens only when a model turn ends, and not at all for audio that was
streamed in silence, so the meter also keeps the audio seconds (25 tokens a second: the
published rate) as an estimate of what the input costs.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

AUDIO_TOKENS_PER_S = 25.0  # Gemini's published input rate for audio
KEEP_MINUTES = 120
COUNTERS = (
    "audio_s",
    "frames",  # pictures sent
    "text_updates",
    "audio_dropped_s",
    "frames_skipped",
    "tokens",  # reported by the server
    "prompt_tokens",
    "response_tokens",
    "turns",
    "tool_calls",
    "quota_errors",
    "errors",
    "reconnects",
)


@dataclass
class _Minute:
    minute: int
    counts: dict[str, float] = field(default_factory=lambda: dict.fromkeys(COUNTERS, 0.0))

    def view(self) -> dict[str, Any]:
        out: dict[str, Any] = {"minute": self.minute}
        for key, value in self.counts.items():
            out[key] = round(value, 2) if key.endswith("_s") else int(value)
        out["audio_tokens_est"] = int(self.counts["audio_s"] * AUDIO_TOKENS_PER_S)
        return out


@dataclass(frozen=True)
class UsageCaps:
    """A limit of the cloud's use (None: no limit); passing one stops the cloud (status ``capped``)."""

    tokens: int | None = None
    audio_s: float | None = None
    frames: int | None = None


class UsageMeter:
    def __init__(self, caps: UsageCaps | None = None) -> None:
        self.caps = caps or UsageCaps()
        self._minutes: deque[_Minute] = deque(maxlen=KEEP_MINUTES)
        self.totals: dict[str, float] = dict.fromkeys(COUNTERS, 0.0)
        self.by_modality: dict[str, int] = {}
        self.last_prompt_tokens = 0  # the context the last turn carried

    # ── counting ──────────────────────────────────────────────────────────────

    def add(self, now: float, key: str, amount: float = 1.0) -> None:
        if key not in self.totals:
            raise KeyError(key)
        bucket = self._bucket(now)
        bucket.counts[key] += amount
        self.totals[key] += amount

    def usage(self, now: float, data: dict[str, Any]) -> None:
        """A server ``usage`` event: one model turn's token counts."""
        self.add(now, "turns")
        self.add(now, "tokens", float(data.get("total") or 0))
        self.add(now, "prompt_tokens", float(data.get("prompt") or 0))
        self.add(now, "response_tokens", float(data.get("response") or 0))
        self.last_prompt_tokens = int(data.get("prompt") or 0)
        for modality, count in dict(data.get("by_modality") or {}).items():
            self.by_modality[str(modality)] = self.by_modality.get(str(modality), 0) + int(count)

    def _bucket(self, now: float) -> _Minute:
        minute = int(now // 60)
        if self._minutes and self._minutes[-1].minute == minute:
            return self._minutes[-1]
        bucket = _Minute(minute)
        self._minutes.append(bucket)
        return bucket

    # ── reading ───────────────────────────────────────────────────────────────

    def per_minute(self, now: float, count: int = 5) -> list[dict[str, Any]]:
        """The last ``count`` minutes up to now (empty minutes included), oldest first."""
        by = {m.minute: m for m in self._minutes}
        current = int(now // 60)
        return [by.get(m, _Minute(m)).view() for m in range(current - count + 1, current + 1)]

    def over_cap(self) -> str | None:
        """The name of the cap that has been passed, or None."""
        caps = self.caps
        if caps.tokens is not None and self.tokens_estimated() > caps.tokens:
            return "tokens"
        if caps.audio_s is not None and self.totals["audio_s"] > caps.audio_s:
            return "audio_s"
        if caps.frames is not None and self.totals["frames"] > caps.frames:
            return "frames"
        return None

    def tokens_estimated(self) -> int:
        """What the server reported plus the audio it has not reported yet (25 tokens a second)."""
        return int(self.totals["tokens"] + self.totals["audio_s"] * AUDIO_TOKENS_PER_S)

    def status(self, now: float) -> dict[str, Any]:
        minutes = self.per_minute(now, 2)
        totals = {k: (round(v, 2) if k.endswith("_s") else int(v)) for k, v in self.totals.items()}
        return {
            "this_minute": minutes[-1],
            "last_minute": minutes[0],
            "totals": totals,
            "tokens_estimated": self.tokens_estimated(),
            "context_tokens": self.last_prompt_tokens,
            "by_modality": dict(sorted(self.by_modality.items())),
            "caps": {
                "tokens": self.caps.tokens,
                "audio_s": self.caps.audio_s,
                "frames": self.caps.frames,
            },
            "over_cap": self.over_cap(),
        }


__all__ = ["AUDIO_TOKENS_PER_S", "UsageCaps", "UsageMeter"]
