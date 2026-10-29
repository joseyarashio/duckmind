"""The grounding gate (live voice, LV2): what the cloud model proposes, and what the duck lets through.

The cloud brain only **proposes** (:mod:`duckmind.live.tools`); nothing it says moves a servo or
makes a sound by itself. The session queues what it calls (:class:`~duckmind.live.session.QueuedCall`,
parsed but not judged); this gate judges each call, for the grounding rules below, and returns a
:class:`Verdict`: an accepted act (with what was clamped) or a refusal **with its reason**. Every
decision is logged (:attr:`GroundingGate.log`, counted per tool and per reason in
:attr:`GroundingGate.counts`).

Rules, in the order they are checked (the first that fails is the call's reason):

1. ``cloud_off`` — the cloud switch is off (or the cloud is capped): the duck acts on nothing the
   cloud says, including what was already queued;
2. ``unknown_tool`` — a name that is not one of the five tools;
3. ``stale`` — the call is older than :attr:`GateConfig.stale_s` (a late reply to a moment that has
   passed is worse than none);
4. ``echo_only`` / ``echo_cause`` — the call can only be the duck hearing itself (LV1's echo gate:
   since the last human-like sound the only loud audio was the duck's own call, or the transcript
   chunk it answers overlapped the duck's own voice). The one exception is **local evidence** that
   a person really spoke (the duck's own voice detection, :mod:`duckmind.live.vad`, with the duck's
   echo taken out): then the call answers that person;
5. the tool's own rules — ``vocalize``: every tag one of the voice bank's calls the duck's mouth can
   voice (``tag_unknown`` off the bank; ``tag_not_voiceable`` in the bank but not in the brain's
   vocal path: ``peck``; ``tag_reserved``: ``alarm`` is the duck's own fright and ``wheee`` the
   ride's stream, neither is the cloud's to ask for), level clamped to 0..1 (``level_silent`` below
   :attr:`GateConfig.level_floor`), ``at_ms`` clamped to :attr:`GateConfig.at_ms_max` with the tags
   of one call spaced :attr:`GateConfig.tag_gap_ms` apart, at most :attr:`GateConfig.max_tags` tags;
   ``recognize`` / ``remember`` / ``attend`` / ``feel``: their fields present, valid and bounded;
6. caps — ``rate_cap``: at most :attr:`GateConfig.rate_max_tags` accepted call-tags per
   :attr:`GateConfig.rate_window_s`; ``utterance_cap``: at most :attr:`GateConfig.utterance_max_tags`
   tags (and :attr:`GateConfig.utterance_max_calls` calls) for one thing a person said; and a
   per-tool cap for the others (``call_cap``).

A ``vocalize`` call is judged tag by tag (an off-bank tag does not spoil the bank tag beside it);
it is accepted when at least one tag is. The gate decides nothing about *when* or *whether the duck
feels like it*: an accepted act is only a candidate for the selector (:mod:`duckmind.live.bridge`),
and the body can still say no.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from duckmind.live.session import QueuedCall
from duckmind.live.tools import (
    ATTEND,
    BANK_TAGS,
    FEEL,
    RECOGNIZE,
    REMEMBER,
    TARGET_KINDS,
    TOOL_NAMES,
    VOCALIZE,
)
from duckmind.neuro.contract import VOCAL_TAGS

# the refusal reasons (codes: the words for a page come with the page, LV5)
CLOUD_OFF = "cloud_off"
UNKNOWN_TOOL = "unknown_tool"
STALE = "stale"
ECHO_ONLY = "echo_only"
ECHO_CAUSE = "echo_cause"
MALFORMED = "malformed"
TAG_UNKNOWN = "tag_unknown"
TAG_NOT_VOICEABLE = "tag_not_voiceable"
TAG_RESERVED = "tag_reserved"
LEVEL_SILENT = "level_silent"
AT_TOO_LATE = "at_ms_too_late"
TOO_MANY_TAGS = "too_many_tags"
RATE_CAP = "rate_cap"
UTTERANCE_CAP = "utterance_cap"
CALL_CAP = "call_cap"
REASONS: tuple[str, ...] = (
    CLOUD_OFF,
    UNKNOWN_TOOL,
    STALE,
    ECHO_ONLY,
    ECHO_CAUSE,
    MALFORMED,
    TAG_UNKNOWN,
    TAG_NOT_VOICEABLE,
    TAG_RESERVED,
    LEVEL_SILENT,
    AT_TOO_LATE,
    TOO_MANY_TAGS,
    RATE_CAP,
    UTTERANCE_CAP,
    CALL_CAP,
)

# the calls the cloud may ask for: the social ones; ``alarm`` is the duck's own fright (its
# reflexes and its pain), ``wheee`` is the ride's stream — neither is a proposal
DEFAULT_ALLOWED: tuple[str, ...] = ("chirp", "inquire", "coo", "greet")

LOG_MAX = 800


@dataclass(frozen=True)
class GateConfig:
    allowed_tags: tuple[str, ...] = DEFAULT_ALLOWED
    level_floor: float = 0.05  # below this a call is silence, not a call
    at_ms_max: int = 3000  # a reply that comes later than this is not a reply
    tag_gap_ms: int = 350  # the tags of one call are at least this far apart (a bank call lasts
    # 0.15–0.8 s: two at once would be a screech)
    max_tags: int = 3  # tags of one call (the schema allows 4)
    rate_window_s: float = 10.0
    rate_max_tags: int = 4  # accepted call-tags in any such window
    utterance_max_tags: int = 3
    utterance_max_calls: int = 2
    utterance_join_s: float = 1.5  # calls whose last human sound is this close: one utterance
    stale_s: float = 6.0
    # the other tools: (calls, per seconds)
    call_caps: Mapping[str, tuple[int, float]] = field(
        default_factory=lambda: {
            ATTEND: (3, 10.0),
            RECOGNIZE: (3, 30.0),
            REMEMBER: (4, 60.0),
            FEEL: (4, 10.0),
        }
    )
    note_max: int = 200


@dataclass(frozen=True, slots=True)
class VocalTag:
    tag: str
    level: float  # 0..1
    at_ms: int  # from the call's arrival


@dataclass(frozen=True, slots=True)
class Refusal:
    seq: int
    call_id: str
    tool: str
    reason: str
    t: float
    detail: Mapping[str, Any] = field(default_factory=dict)

    def view(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "tool": self.tool,
            "reason": self.reason,
            "t": round(self.t, 3),
            **{k: v for k, v in self.detail.items()},
        }


@dataclass(frozen=True, slots=True)
class Verdict:
    """What the gate made of one call. ``accepted``: at least something got through."""

    call: QueuedCall
    accepted: bool
    reason: str | None  # why it was refused (the first rule that failed); None if accepted
    tags: tuple[VocalTag, ...] = ()  # vocalize: the tags that got through
    args: Mapping[str, Any] = field(default_factory=dict)  # the other tools: validated, bounded
    refused: tuple[Refusal, ...] = ()  # every refusal of this call (a tag, or the whole call)
    notes: tuple[str, ...] = ()  # what was clamped or moved
    rescued: bool = False  # echo-marked, taken because the duck itself heard a person speak

    def view(self) -> dict[str, Any]:
        return {
            "seq": self.call.seq,
            "tool": self.call.name,
            "accepted": self.accepted,
            "reason": self.reason,
            "tags": [{"tag": t.tag, "level": t.level, "at_ms": t.at_ms} for t in self.tags],
            "args": dict(self.args),
            "refused": [r.view() for r in self.refused],
            "notes": list(self.notes),
            "rescued": self.rescued,
        }


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clean_text(value: Any, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    return text[:limit] if text else None


HumanEvidence = Callable[[float, float], bool]


class GroundingGate:
    """See the module docstring. Not thread-safe by itself: the bridge calls it from one place."""

    def __init__(
        self,
        config: GateConfig | None = None,
        *,
        cloud_on: Callable[[], bool] = lambda: True,
        human_evidence: HumanEvidence | None = None,
    ) -> None:
        self.config = config or GateConfig()
        self._cloud_on = cloud_on
        self._human = human_evidence
        self.log: deque[dict[str, Any]] = deque(maxlen=LOG_MAX)
        self.counts: Counter[str] = Counter()
        self.refusals: Counter[str] = Counter()
        self._tag_times: deque[float] = deque()  # when each accepted tag was accepted
        self._call_times: dict[str, deque[float]] = {name: deque() for name in TOOL_NAMES}
        self._utt: dict[str, Any] = {"human_t": None, "t": -1e9, "tags": 0, "calls": 0, "n": 0}

    # ── the verdict ───────────────────────────────────────────────────────────

    def judge(self, call: QueuedCall, now: float) -> Verdict:
        verdict = self._judge(call, now)
        self.counts[f"{call.name}:{'accepted' if verdict.accepted else 'refused'}"] += 1
        for refusal in verdict.refused:
            self.refusals[refusal.reason] += 1
        self.log.append(verdict.view() | {"now": round(now, 3)})
        return verdict

    def _refuse(
        self,
        call: QueuedCall,
        reason: str,
        now: float,
        **detail: Any,
    ) -> Verdict:
        refusal = Refusal(call.seq, call.id, call.name, reason, now, detail)
        return Verdict(call, False, reason, refused=(refusal,))

    def _judge(self, call: QueuedCall, now: float) -> Verdict:
        cfg = self.config
        if not self._cloud_on():
            return self._refuse(call, CLOUD_OFF, now)
        if call.name not in TOOL_NAMES:
            return self._refuse(call, UNKNOWN_TOOL, now, tool=call.name)
        if now - call.t > cfg.stale_s:
            return self._refuse(call, STALE, now, age_s=round(now - call.t, 2))
        rescued = False
        echo = self._echo_reason(call)
        if echo is not None:
            if self._heard_a_person(call):
                rescued = True
            else:
                return self._refuse(call, echo, now, mark=call.mark.view())
        if call.name == VOCALIZE:
            verdict = self._vocalize(call, now)
        else:
            verdict = self._other(call, now)
        if rescued and verdict.accepted:
            verdict = Verdict(
                verdict.call,
                True,
                None,
                verdict.tags,
                verdict.args,
                verdict.refused,
                verdict.notes,
                rescued=True,
            )
        return verdict

    # ── echo ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _echo_reason(call: QueuedCall) -> str | None:
        if call.echo_suspect or call.mark.echo_only:
            return ECHO_ONLY
        if call.mark.cause_echo:
            return ECHO_CAUSE
        return None

    def _heard_a_person(self, call: QueuedCall) -> bool:
        """The duck's own voice detection heard a person speak just before this call."""
        if self._human is None:
            return False
        try:
            return bool(self._human(call.t - 4.0, call.t + 0.05))
        except Exception:  # (a fault in the detector never lets a call through)
            return False

    # ── vocalize ──────────────────────────────────────────────────────────────

    def _utterance(self, call: QueuedCall) -> dict[str, Any]:
        """The state of the utterance a call belongs to: calls whose last human sound is within
        :attr:`GateConfig.utterance_join_s` of the previous call's, and soon after it, are one."""
        cfg = self.config
        since = call.mark.since_human_s
        human_t = None if since is None else call.t - since
        u = self._utt
        same = (
            call.t - u["t"] <= 8.0
            and (u["human_t"] is None) == (human_t is None)
            and (human_t is None or abs(human_t - u["human_t"]) <= cfg.utterance_join_s)
        )
        if not same:
            u = self._utt = {
                "human_t": human_t,
                "t": call.t,
                "tags": 0,
                "calls": 0,
                "n": u["n"] + 1,
            }
        u["t"] = call.t
        if human_t is not None:
            u["human_t"] = human_t
        return u

    def _vocalize(self, call: QueuedCall, now: float) -> Verdict:
        cfg = self.config
        raw = call.raw.get("tags")
        if not isinstance(raw, Sequence) or isinstance(raw, str) or not raw:
            return self._refuse(call, MALFORMED, now, why="tags_missing")
        utt = self._utterance(call)
        self._trim(self._tag_times, now, cfg.rate_window_s)
        accepted: list[VocalTag] = []
        refused: list[Refusal] = []
        notes: list[str] = []
        last_at = -cfg.tag_gap_ms
        if len(raw) > cfg.max_tags:
            for extra in list(raw)[cfg.max_tags :]:
                refused.append(self._tag_refusal(call, TOO_MANY_TAGS, now, extra))
            notes.append("tags_cut")
        if utt["calls"] >= cfg.utterance_max_calls:
            return self._refuse(call, UTTERANCE_CAP, now, utterance=utt["n"], calls=utt["calls"])
        for item in list(raw)[: cfg.max_tags]:
            if not isinstance(item, Mapping):
                refused.append(self._tag_refusal(call, MALFORMED, now, item))
                continue
            tag = _clean_text(item.get("tag"), 40)
            level = _number(item.get("level"))
            at_ms = _number(item.get("at_ms", 0))
            if tag is None or level is None or at_ms is None:
                refused.append(self._tag_refusal(call, MALFORMED, now, item))
                continue
            reason = self._tag_reason(tag)
            if reason is not None:
                refused.append(self._tag_refusal(call, reason, now, item, tag=tag))
                continue
            clamped = min(1.0, max(0.0, level))
            if clamped != level:
                notes.append("level_clamped")
            if clamped < cfg.level_floor:
                refused.append(self._tag_refusal(call, LEVEL_SILENT, now, item, tag=tag))
                continue
            at = int(min(float(cfg.at_ms_max), max(0.0, at_ms)))
            if at != int(at_ms):
                notes.append("at_ms_clamped")
            if at < last_at + cfg.tag_gap_ms:
                at = last_at + cfg.tag_gap_ms
                notes.append("tags_spaced")
            if at > cfg.at_ms_max:
                refused.append(self._tag_refusal(call, AT_TOO_LATE, now, item, tag=tag))
                continue
            if len(self._tag_times) >= cfg.rate_max_tags:
                refused.append(self._tag_refusal(call, RATE_CAP, now, item, tag=tag))
                continue
            if utt["tags"] >= cfg.utterance_max_tags:
                refused.append(self._tag_refusal(call, UTTERANCE_CAP, now, item, tag=tag))
                continue
            accepted.append(VocalTag(tag, round(clamped, 3), at))
            last_at = at
            self._tag_times.append(now)
            utt["tags"] += 1
        if not accepted:
            first = refused[0].reason if refused else MALFORMED
            return Verdict(call, False, first, refused=tuple(refused), notes=tuple(notes))
        utt["calls"] += 1
        return Verdict(
            call,
            True,
            None,
            tuple(accepted),
            refused=tuple(refused),
            notes=tuple(dict.fromkeys(notes)),
        )

    def _tag_reason(self, tag: str) -> str | None:
        if tag not in BANK_TAGS:
            return TAG_UNKNOWN
        if tag not in VOCAL_TAGS:
            return TAG_NOT_VOICEABLE
        if tag not in self.config.allowed_tags:
            return TAG_RESERVED
        return None

    @staticmethod
    def _tag_refusal(
        call: QueuedCall, reason: str, now: float, item: Any, *, tag: str | None = None
    ) -> Refusal:
        detail: dict[str, Any] = {}
        if tag is not None:
            detail["tag"] = tag
        elif isinstance(item, Mapping) and isinstance(item.get("tag"), str):
            detail["tag"] = str(item["tag"])[:40]
        return Refusal(call.seq, call.id, call.name, reason, now, detail)

    @staticmethod
    def _trim(times: deque[float], now: float, window_s: float) -> None:
        while times and now - times[0] > window_s:
            times.popleft()

    # ── the other tools ───────────────────────────────────────────────────────

    def _other(self, call: QueuedCall, now: float) -> Verdict:
        cfg = self.config
        raw = call.raw
        args: dict[str, Any] = {}
        notes: list[str] = []
        if call.name == ATTEND:
            kind = _clean_text(raw.get("target_kind"), 20)
            if kind not in TARGET_KINDS:
                return self._refuse(call, MALFORMED, now, why="target_kind")
            args["target_kind"] = kind
            hint = _clean_text(raw.get("hint"), cfg.note_max)
            if hint is not None:
                args["hint"] = hint
        elif call.name == RECOGNIZE:
            name = _clean_text(raw.get("name"), cfg.note_max)
            if name is None:
                return self._refuse(call, MALFORMED, now, why="name")
            args["name"] = name
        elif call.name == REMEMBER:
            note = _clean_text(raw.get("note"), cfg.note_max)
            if note is None:
                return self._refuse(call, MALFORMED, now, why="note")
            args["note"] = note
        elif call.name == FEEL:
            valence = _number(raw.get("valence"))
            talk = _number(raw.get("talk"))
            if valence is None or talk is None:
                return self._refuse(call, MALFORMED, now, why="valence_talk")
            clamped_v, clamped_t = min(1.0, max(-1.0, valence)), min(1.0, max(0.0, talk))
            if (clamped_v, clamped_t) != (valence, talk):
                notes.append("clamped")
            args["valence"] = round(clamped_v, 3)
            args["talk"] = round(clamped_t, 3)
        limit, window = cfg.call_caps.get(call.name, (6, 10.0))
        times = self._call_times[call.name]
        self._trim(times, now, window)
        if len(times) >= limit:
            return self._refuse(call, CALL_CAP, now, tool=call.name, limit=limit, window_s=window)
        times.append(now)
        return Verdict(call, True, None, args=args, notes=tuple(notes))

    # ── status ────────────────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        counts, refusals = dict(self.counts), dict(self.refusals)  # (copies: another thread works)
        return {
            "accepted": sum(v for k, v in counts.items() if k.endswith(":accepted")),
            "refused_calls": sum(v for k, v in counts.items() if k.endswith(":refused")),
            "counts": dict(sorted(counts.items())),
            "refusals": dict(sorted(refusals.items())),
            "allowed_tags": list(self.config.allowed_tags),
            "caps": {
                "rate_window_s": self.config.rate_window_s,
                "rate_max_tags": self.config.rate_max_tags,
                "utterance_max_tags": self.config.utterance_max_tags,
                "utterance_max_calls": self.config.utterance_max_calls,
                "at_ms_max": self.config.at_ms_max,
                "max_tags": self.config.max_tags,
            },
        }


__all__ = [
    "AT_TOO_LATE",
    "CALL_CAP",
    "CLOUD_OFF",
    "DEFAULT_ALLOWED",
    "ECHO_CAUSE",
    "ECHO_ONLY",
    "LEVEL_SILENT",
    "MALFORMED",
    "RATE_CAP",
    "REASONS",
    "STALE",
    "TAG_NOT_VOICEABLE",
    "TAG_RESERVED",
    "TAG_UNKNOWN",
    "TOO_MANY_TAGS",
    "UNKNOWN_TOOL",
    "UTTERANCE_CAP",
    "GateConfig",
    "GroundingGate",
    "HumanEvidence",
    "Refusal",
    "Verdict",
    "VocalTag",
]
