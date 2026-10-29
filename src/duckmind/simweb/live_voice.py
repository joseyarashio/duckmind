"""The talk panel's server side (live voice, LV5): the live conversation inside the simulator.

``--live-voice`` (needs ``--brain neuro --live``) plugs LV2's :func:`~duckmind.live.wiring.attach_live`
into the running neural duck: its microphone stream goes up one :class:`~duckmind.live.session.LiveSession`
(Gemini Live), the duck-eye pictures (at most one a second, the same ``eye_image`` the page's eye window
reads) go up beside it, the tool calls that come back go through the grounding gate and the bridge into
the duck's own competition of behaviours, and a person who speaks over the duck's call stops it. The
**cloud switch is off at start**; the page turns it on. With the cloud off, down or out of quota the duck
lives on as always and the status says why (words from the catalogs, ``neuro.live.*``).

Routes (the microphone's own chunk endpoint stays ``POST /api/neuro/mic/chunk``)::

    GET  /api/live/status    everything the talk panel shows (see :meth:`LiveVoice.status`)
    GET  /api/live/eye.jpg   the last picture of the duck's eye that was sent to the cloud
    POST /api/live/cloud     {"on": true|false}: the cloud switch
    POST /api/live/say       {"utterance": "<scripted line id>", "who": "<persona>", "position": "<id>"}:
                             a family member says one of LV0's scripted lines (a way to try the panel
                             with no microphone; the world's own ``say`` stimulus)

Everything the page needs to know about words comes as catalog messages (``_i18n`` maps, like the rest of
the neural duck's API); what the family said and the names the family gave are data and are never
translated. Nothing is stored: transcripts, the last picture and the call list live in the process's
memory only, and the API key never touches this module (it is read at the moment a connection opens).

Threads: the physics tick thread feeds the session (the microphone's frames, the duck-eye pictures, the
status text of the duck) and reads the brain; the HTTP threads read what the session, the bridge and the
brain hold, copying the deques first, and flip the cloud switch (the session's own lock).
"""

from __future__ import annotations

import math
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from duckmind.i18n import Msg, lst, msg
from duckmind.live.bridge import BridgeConfig, LiveBridge
from duckmind.live.meter import UsageCaps, UsageMeter
from duckmind.live.names import to_traditional_text, vocabulary
from duckmind.live.session import LiveSession, SessionConfig
from duckmind.live.soul import SoulInputs, build_soul
from duckmind.live.tools import ATTEND, FEEL, RECOGNIZE, REMEMBER, VOCALIZE
from duckmind.live.wiring import CONTEXT_EVERY_S as BRIDGE_CONTEXT_EVERY_S
from duckmind.live.wiring import attach_live
from duckmind.simweb.eyeframes import EyeFrames
from duckmind.simweb.live_voice_api import ROUTE
from duckmind.world.neuro_utterances import MAX_PLACE_MM, UtteranceLibrary
from duckmind.world.people import PERSONAS

VOCABULARY_EVERY_S = (
    2.0  # the transcriber's words (the duck's name, the family's) are re-read this often
)
HEARD_ROWS = 8
CALL_ROWS = 8
HEARD_JOIN_S = 1.5  # transcript chunks this close are one thing a person said
NEAR_M = 1.0  # the speaker's chooser: this far from the duck …
FAR_M = 3.0
MIN_DB_1M, MAX_DB_1M = (
    40.0,
    90.0,
)  # how loud a scripted line may be said, dB at 1 m (a whisper … a shout)
PLACES = (
    "auto",
    "front",
    "left",
    "right",
    "behind",
    "far",
)  # the chooser's ids ("auto": the world's)

# what a call came to (the words are ``neuro.live.outcome.<code>``)
VOICED, WAITING, LAPSED, CANCELLED = "voiced", "waiting", "lapsed", "cancelled"
ATTENDED, BOUND, NOTED, FELT, PENDING, REFUSED = (
    "attended",
    "bound",
    "noted",
    "felt",
    "pending",
    "refused",
)
OUTCOMES = (VOICED, WAITING, LAPSED, CANCELLED, ATTENDED, BOUND, NOTED, FELT, PENDING, REFUSED)
# every reason the gate, the bridge, the duck's perception and the names can give (``neuro.live.reason.<code>``)
REASONS = (
    "cloud_off",
    "unknown_tool",
    "stale",
    "echo_only",
    "echo_cause",
    "malformed",
    "tag_unknown",
    "tag_not_voiceable",
    "tag_reserved",
    "level_silent",
    "at_ms_too_late",
    "too_many_tags",
    "rate_cap",
    "utterance_cap",
    "call_cap",
    "no_reply_channel",
    "no_person_in_view",
    "several_in_view",
    "barge_in",
    "lost_competition",
    "replaced",
    "nothing_matches",
    "not_in_view",
    "unknown_person",
    "name_empty",
    "name_too_long",
    "not_a_name",
    "duck_name",
    "sounds_like_duck_name",
    "name_taken",
    "conflict",
    "no_recent_speech",
)
TAGS = ("greet", "inquire", "chirp", "coo", "alarm", "wheee", "peck")
TOOLS = (VOCALIZE, ATTEND, RECOGNIZE, REMEMBER, FEEL)
ATTEND_KINDS = ("person", "thing", "sound")


@dataclass(frozen=True)
class SentFrame:
    """One picture of the duck's eye that was handed to the cloud connection."""

    seq: int
    jpeg: bytes
    tag: str
    wall: float
    width: int | None
    height: int | None


def _copy(items: Any) -> list[Any]:
    """A list copy of a deque another thread appends to (retried: a copy can race an append)."""
    for _ in range(8):
        try:
            return list(items)
        except RuntimeError:
            continue
    return []


def _word(prefix: str, code: str, allowed: Sequence[str]) -> Msg | None:
    """``prefix.code`` as words when the code is one we have words for (never a made-up key)."""
    return msg(f"{prefix}.{code}") if code in allowed else None


class LiveVoice:
    """The live conversation of one neural duck (see the module doc). ``driver`` is the live
    :class:`~duckmind.simweb.neuro_live.NeuroDriver` (its ``lanes``, ``inject``, ``_truth``);
    ``eye_source`` the physics bridge (``eye_image() -> (jpeg, tag, why)``) or None;
    ``name_of()`` the duck's name now (``name`` overrides it: what the family calls it);
    ``session_factory(soul, meter=…, enabled=…)`` makes the session (the real Gemini one by
    default; a fake in the tests)."""

    def __init__(
        self,
        driver: Any,
        *,
        eye_source: Any = None,
        name_of: Callable[[], str] = lambda: "小鴨",
        name: str | None = None,
        caps: UsageCaps | None = None,
        session_factory: Callable[..., LiveSession] | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
        library: UtteranceLibrary | None = None,
        transcription_languages: Sequence[str] | None = None,
    ) -> None:
        self.driver = driver
        self.eye_source = eye_source
        self._name_of = name_of
        self._name_fixed = name
        self._factory = session_factory
        self._clock = clock
        self._wall = wall
        self._library_given = library
        # the languages the transcriber is told the family speaks (None: the session's default,
        # zh-TW then en-US; empty: no hint, the service guesses per sentence)
        self._languages = (
            None if transcription_languages is None else tuple(transcription_languages)
        )
        self._library_loaded = False
        self._library_cache: UtteranceLibrary | None = None
        self.meter = UsageMeter(caps)
        self._lock = threading.RLock()
        self._want = False  # the cloud switch as the family left it: off at start, always
        self.session: LiveSession | None = None
        self.bridge: LiveBridge | None = None
        self.eyes: EyeFrames | None = None
        self.name = ""
        self.error: Msg | None = None  # why there is no live session (words)
        self.faults = 0
        self.last_fault: str | None = None
        self._lanes: Any = None
        self._gen = 0
        self._vocab_next = 0.0
        self._fed: SentFrame | None = None
        self._sent: SentFrame | None = None
        self._fed_seq = 0
        self._frames_seen = 0
        self.attach()

    # ── wiring to the duck ────────────────────────────────────────────────────

    def duck_name(self) -> str:
        if self._name_fixed:
            return self._name_fixed
        try:
            return str(self._name_of() or "小鴨")
        except Exception:
            return "小鴨"

    def attach(self) -> None:
        """(Re)connect to the duck's current lanes: a new session and bridge (a switch of the duck is
        a new brain, a new name and a new soul), the cloud switch kept as the family left it and the
        usage meter carried over."""
        with self._lock:
            self._detach()
            lanes = self.driver.lanes
            self.name = self.duck_name()
            self._gen += 1
            gen = self._gen
            session: LiveSession | None = None
            try:
                session = self._make_session()
                # the duck's state goes to the cloud brain from the bridge itself, every 5 s: LV4's
                # snapshot (想說話, 親近 to the person it sees, whom it knows and how close, what its
                # eyes see); the page no longer builds a second snapshot of its own over it
                bridge = attach_live(
                    lanes,
                    session,
                    config=BridgeConfig(
                        duck_name=self.name,
                        context_every_s=BRIDGE_CONTEXT_EVERY_S,
                        behaviour_text=self._behaviour_word,
                    ),
                )
            except Exception as error:
                if session is not None:
                    session.close()
                self.error = (
                    msg("neuro.live.web.no_mic")
                    if isinstance(error, ValueError)
                    else msg("neuro.live.web.not_ready")
                )
                self.last_fault = f"attach: {type(error).__name__}: {error}"[:300]
                self.faults += 1
                return
            self.error = None
            self._lanes = lanes
            self.session, self.bridge = session, bridge
            self.eyes = (
                EyeFrames.from_bridge(self.eye_source, clock=self._clock, wall=self._wall)
                if self.eye_source is not None and hasattr(self.eye_source, "eye_image")
                else None
            )
            self._vocab_next = 0.0
            self._fed = self._sent = None
            self._frames_seen = int(self.meter.totals["frames"])
            self._tell_vocabulary(session)
            lanes.sound.mic.listen(lambda frame: self._on_frame(gen, frame))

    def _make_session(self) -> LiveSession:
        if self._factory is not None:
            return self._factory(self._soul, meter=self.meter, enabled=self._want)
        config = (
            SessionConfig()
            if self._languages is None
            else SessionConfig(transcription_languages=self._languages)
        )
        return LiveSession.for_gemini(
            self._soul, clock=self._clock, meter=self.meter, enabled=self._want, config=config
        )

    def _detach(self) -> None:
        old, self.session, self.bridge, self.eyes = self.session, None, None, None
        self._gen += 1  # (a frame listener of the old microphone stops doing anything)
        if old is not None:
            old.close()

    def cloud_on(self) -> bool:
        """Whether the family left the cloud switch on and a session is there (the emotion layer: a voice the cloud
        answers is understood; with the cloud off the duck only hears that someone talks)."""
        return bool(self._want and self.session is not None)

    def calls_seen(self) -> int:
        """How many calls the cloud has made on this duck so far (the newest call's number): a voice that changes
        nothing in a few seconds was not understood."""
        bridge = self.bridge
        seen = None if bridge is None else _copy(bridge.seen)
        return int(seen[-1].seq) if seen else 0

    def on_switch(self, *_args: Any) -> None:
        """The family picked another duck: its brain is new, so the connection is new too."""
        self.attach()

    def close(self) -> None:
        with self._lock:
            self._detach()

    # ── the soul and the status text (read in the tick thread, where the brain lives) ──

    def _brain(self) -> Any:
        return getattr(self._lanes, "brain", None)

    def _soul(self) -> str:
        brain = self._brain()
        bridge = self.bridge
        genome: dict[str, Any] = {}
        try:
            for key, value in dict(brain.genome.summary()).items():
                genome[key] = (
                    value if isinstance(value, (int, float, list, dict, type(None))) else str(value)
                )
        except Exception:
            genome = {}
        return build_soul(
            SoulInputs(
                name=self.name,
                genome=genome,
                learner={},
                drives=self._drives(brain, bridge),
                known_people=self._known_people(brain),
                behaviour=self._behaviour(brain),
            )
        )

    @staticmethod
    def _behaviour_word(behaviour: str) -> str:
        """What the duck is doing, in the family's words (the contract's name for the act)."""
        from duckmind.neuro import contract as C

        return str(C.behaviour_zh(str(behaviour)))

    @classmethod
    def _behaviour(cls, brain: Any) -> str:
        try:
            return cls._behaviour_word(str(brain.behaviour))
        except Exception:
            return ""

    @staticmethod
    def _drives(brain: Any, bridge: LiveBridge | None) -> dict[str, float]:
        try:
            out = {k: float(v) for k, v in brain.drives.as_dict().items()}
        except Exception:
            return {}
        if bridge is not None:
            out["talk"] = float(bridge.talk.value)
        return out

    @staticmethod
    def _known_people(brain: Any) -> tuple[str, ...]:
        try:
            return tuple(str(n) for n in brain.learn.person_names().values())
        except Exception:
            return ()

    def _tell_vocabulary(self, session: LiveSession) -> None:
        """The transcriber is told the duck's name and the names the family has given (the next
        connection opens with them: the session's ``custom_vocabulary`` setting)."""
        try:
            session.set_vocabulary(vocabulary(self.name, self._known_people(self._brain())))
        except Exception as error:  # (the words are a help: a fault here never stops the duck)
            self.faults += 1
            self.last_fault = f"vocabulary: {type(error).__name__}: {error}"[:300]

    # ── the tick thread: the pictures and the transcriber's words ──────────────

    def _on_frame(self, gen: int, frame: Any) -> None:
        if gen != self._gen:
            return
        session = self.session
        if session is None:
            return
        try:
            now = self._clock()
            if now >= self._vocab_next:
                self._vocab_next = now + VOCABULARY_EVERY_S
                self._tell_vocabulary(session)
            eyes = self.eyes
            if eyes is not None and session.sending["video"]:
                picture = eyes.poll()
                if picture is not None:
                    self._fed_seq += 1
                    self._fed = SentFrame(
                        self._fed_seq,
                        picture.jpeg,
                        picture.tag,
                        picture.wall,
                        picture.width,
                        picture.height,
                    )
                    session.feed_image(picture.jpeg, picture.mime)
            total = int(session.meter.totals["frames"])
            if total != self._frames_seen:  # (a picture went up: it is the newest one fed)
                self._frames_seen = total
                self._sent = self._fed
        except Exception as error:  # (a fault here never stops the duck)
            self.faults += 1
            self.last_fault = f"frame: {type(error).__name__}: {error}"[:300]

    # ── the switch ────────────────────────────────────────────────────────────

    def set_cloud(self, on: bool) -> dict[str, Any]:
        """The cloud switch. On: the connection opens (the key is read then) and the sound and the
        duck-eye pictures start to go; off: they stop at once and the connection closes."""
        with self._lock:
            session = self.session
            if session is None:
                raise ValueError(self.error or msg("neuro.live.web.not_ready"))
            self._want = bool(on)
            session.set_enabled(self._want)
        return {"ok": True, "cloud": self._cloud(session.status())}

    # ── the family says a scripted line ───────────────────────────────────────

    def library(self) -> UtteranceLibrary | None:
        if self._library_given is not None:
            return self._library_given
        if not self._library_loaded:
            self._library_loaded = True
            try:
                self._library_cache = UtteranceLibrary.default()
            except (OSError, ValueError, KeyError):
                self._library_cache = None
        return self._library_cache

    def places(self) -> list[dict[str, Any]]:
        """The speaker's places the chooser offers: ``auto`` (the world puts the person where that
        family member stands) and a few spots round the duck (home millimetres, from where the duck
        stands now; none without the duck's pose)."""
        out: list[dict[str, Any]] = [{"id": "auto"}]
        from duckmind.simweb.neuro_live import _truth_pose

        pose = _truth_pose(getattr(self.driver, "_truth", None))
        if pose is None:
            return out
        x, y, yaw = pose
        for ident, metres, turn in (
            ("front", NEAR_M, 0.0),
            ("left", NEAR_M, math.pi / 2),
            ("right", NEAR_M, -math.pi / 2),
            ("behind", NEAR_M, math.pi),
            ("far", FAR_M, 0.0),
        ):
            px = x + metres * math.cos(yaw + turn)
            py = y + metres * math.sin(yaw + turn)
            if max(abs(px), abs(py)) * 1000.0 <= MAX_PLACE_MM:
                out.append({"id": ident, "x_mm": round(px * 1000.0), "y_mm": round(py * 1000.0)})
        return out

    def say(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """A family member says one of the scripted lines from a place (the world's ``say``
        stimulus: the duck's ears and the microphone hear it as they hear anyone). Refusals are
        ValueErrors with their words."""
        library = self.library()
        if library is None:
            raise ValueError(msg("neuro.world_refusal.say_no_lines"))
        asked = str(params.get("utterance") or "")
        if asked not in library.lines:
            raise ValueError(
                msg(
                    "neuro.world_refusal.say_unknown",
                    given_id=asked,
                    known_ids=lst(library.ids()),
                )
            )
        line = library.get(asked)
        who = str(params.get("who") or line.persona or "")
        if who and who not in {p.id for p in PERSONAS}:
            raise ValueError(msg("neuro.live.web.bad_who"))
        position = str(params.get("position") or "auto")
        if position not in PLACES:
            raise ValueError(msg("neuro.live.web.bad_position"))
        request: dict[str, Any] = {"utterance": asked, "duck_name": self.name}
        if (
            params.get("db1m") is not None
        ):  # how loud they speak, at 1 m (the kind's own level if left out)
            try:
                loud = float(params["db1m"])
            except (TypeError, ValueError):
                loud = math.nan
            if not (MIN_DB_1M <= loud <= MAX_DB_1M):
                raise ValueError(msg("neuro.live.web.bad_loudness", low=MIN_DB_1M, high=MAX_DB_1M))
            request["db1m"] = loud
        if who:
            request["who"] = who
        if position != "auto":
            spot = next((p for p in self.places() if p["id"] == position), None)
            if spot is None:
                raise ValueError(msg("neuro.live.web.no_position"))
            request["where_mm"] = [spot["x_mm"], spot["y_mm"]]
        inject = getattr(self.driver, "inject", None)
        if not callable(inject):
            raise ValueError(msg("neuro.live.web.not_ready"))
        answer = dict(inject("say", request) or {})
        self.family_speaks(who)
        out: dict[str, Any] = {"ok": True, "line": line.view(), "who": who}
        if answer.get("where_mm") is not None:
            out["where_mm"] = answer["where_mm"]
        if answer.get("note_zh"):
            out["note_zh"] = answer["note_zh"]
        return out

    def family_speaks(self, who: str) -> None:
        """The page speaks as ``who``: a family role (主人, 伴侶, 小孩, 長輩) is a known person to the
        duck (the body that stands for them is introduced by the role's name); a guest, or no one,
        is a stranger."""
        role = next((p for p in PERSONAS if p.id == who and p.id != "guest"), None)
        bridge = self.bridge
        if role is not None and bridge is not None:
            bridge.family_speaks(str(role.zh))

    # ── the last picture sent ─────────────────────────────────────────────────

    def eye_frame(self) -> SentFrame | None:
        return self._sent

    # ── status: everything the panel shows ────────────────────────────────────

    @staticmethod
    def _cloud(s: Mapping[str, Any]) -> dict[str, Any]:
        error = s.get("last_error") or {}
        return {
            "on": bool(s["enabled"]),
            "state": s["state"],
            "connected": s["state"] == "online",
            "words": s["words"],
            "understands_speech": bool(s["understands_speech"]),
            "sending": dict(s["sending"]),
            "model": s["model"],
            "since_s": s["since_s"],
            "retry_in_s": s["retry_in_s"],
            "error_kind": error.get("kind"),
            "error_words": s.get("error_words"),
            "failures_in_a_row": s.get("failures_in_a_row", 0),
            "connections": s["connections"],
        }

    @staticmethod
    def _usage(s: Mapping[str, Any]) -> dict[str, Any]:
        u = s["usage"]
        last, now_, totals = u["last_minute"], u["this_minute"], u["totals"]
        return {
            "last_minute": {
                "audio_s": last["audio_s"],
                "frames": last["frames"],
                "text_updates": last["text_updates"],
                "tokens": last["tokens"] + last["audio_tokens_est"],
                "tool_calls": last["tool_calls"],
            },
            "this_minute": {
                "audio_s": now_["audio_s"],
                "frames": now_["frames"],
                "text_updates": now_["text_updates"],
                "tokens": now_["tokens"] + now_["audio_tokens_est"],
                "tool_calls": now_["tool_calls"],
            },
            "totals": {
                "audio_s": totals["audio_s"],
                "frames": totals["frames"],
                "text_updates": totals["text_updates"],
                "tokens": u["tokens_estimated"],
                "turns": totals["turns"],
                "tool_calls": totals["tool_calls"],
                "errors": totals["errors"],
                "quota_errors": totals["quota_errors"],
            },
            "caps": dict(u["caps"]),
            "over_cap": u["over_cap"],
        }

    def _heard(self, session: LiveSession, now: float) -> list[dict[str, Any]]:
        """What the cloud heard the family say (its own transcript: data, not translated), chunks
        that came close together joined, newest first. Shown in traditional characters (a sentence
        that came back simplified is converted here; the session's log keeps the raw text)."""
        chunks = [e for e in _copy(session.log) if e.get("kind") == "heard"]
        lines: list[dict[str, Any]] = []
        for e in chunks:
            text = str(e.get("text") or "")
            t = float(e.get("t") or 0.0)
            suspect = bool(e.get("echo_suspect"))
            if lines and t - lines[-1]["_t"] <= HEARD_JOIN_S:
                row = lines[-1]
                row["text"] += text
                row["_n"] += 1
                row["_suspect"] += 1 if suspect else 0
                row["_t"] = t
            else:
                lines.append({"text": text, "_t": t, "_n": 1, "_suspect": 1 if suspect else 0})
        out = [
            {
                "age_s": round(max(0.0, now - row["_t"]), 1),
                "text": to_traditional_text(row["text"].strip()),
                "echo": row["_suspect"] * 2 > row["_n"],
            }
            for row in lines
            if row["text"].strip()
        ]
        return list(reversed(out))[:HEARD_ROWS]

    def _calls(self, bridge: LiveBridge, now: float) -> list[dict[str, Any]]:
        """The cloud's proposals and what became of each, newest first: its call (the tool, the
        calls it asked for), and the outcome with its reason when the duck did not do it."""
        seen = _copy(bridge.seen)
        log = _copy(bridge.log)
        voiced = _copy(bridge.voiced)
        by_seq: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in log:
            if record.get("seq") is not None:
                by_seq[int(record["seq"])].append(record)
        tags_voiced: dict[int, list[str]] = defaultdict(list)
        for item in voiced:
            tags_voiced[int(item["seq"])].append(str(item["tag"]))
        out: list[dict[str, Any]] = []
        for call in reversed(seen[-CALL_ROWS:]):
            records = by_seq.get(int(call.seq), [])
            first = next((r for r in records if r.get("stage") in ("gate", "bridge")), None)
            final = next((r for r in reversed(records) if r.get("stage") == "selector"), None)
            outcome, reason = self._outcome(call.name, first, final, tags_voiced.get(int(call.seq)))
            out.append(self._call_row(call, outcome, reason, tags_voiced.get(int(call.seq)), now))
        return out

    @staticmethod
    def _outcome(
        tool: str,
        first: Mapping[str, Any] | None,
        final: Mapping[str, Any] | None,
        voiced: Sequence[str] | None,
    ) -> tuple[str, str | None]:
        if first is None:
            return WAITING, None
        if first.get("decision") == "refused":
            return REFUSED, first.get("reason")
        if first.get("decision") == "pending":
            return PENDING, first.get("reason")
        if tool == VOCALIZE:
            if voiced:
                return VOICED, None
            if final is None:
                return WAITING, None
            return (LAPSED if final.get("decision") == "expired" else CANCELLED), final.get(
                "reason"
            )
        if tool == ATTEND:
            if final is None:
                return WAITING, None
            if final.get("decision") == "done":
                return ATTENDED, None
            return (LAPSED if final.get("decision") == "expired" else CANCELLED), final.get(
                "reason"
            )
        return {RECOGNIZE: BOUND, REMEMBER: NOTED, FEEL: FELT}.get(tool, FELT), None

    @staticmethod
    def _call_row(
        call: Any, outcome: str, reason: str | None, voiced: Sequence[str] | None, now: float
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "seq": int(call.seq),
            "age_s": round(max(0.0, now - float(call.t)), 1),
            "tool": call.name,
            "tool_words": _word("neuro.live.tool", call.name, TOOLS),
            "outcome": outcome,
            "outcome_words": msg(f"neuro.live.outcome.{outcome}"),
            "reason": reason,
            "reason_words": _word("neuro.live.reason", reason, REASONS) if reason else None,
            "echo_suspect": bool(call.echo_suspect),
        }
        args = call.args if isinstance(call.args, Mapping) else {}
        if call.name == VOCALIZE:
            tags = []
            for item in list(args.get("tags") or []) or list((call.raw or {}).get("tags") or []):
                if not isinstance(item, Mapping):
                    continue
                tag = str(item.get("tag") or "")
                level = item.get("level")
                tags.append(
                    {
                        "tag": tag,
                        "level": round(float(level), 2)
                        if isinstance(level, (int, float))
                        else None,
                        "meaning": _word("neuro.live.tag", tag, TAGS),
                        "voiced": tag in (voiced or ()),
                    }
                )
            row["tags"] = tags
        elif call.name == ATTEND:
            kind = str(args.get("target_kind") or "")
            row["kind_words"] = _word("neuro.live.attend", kind, ATTEND_KINDS)
            row["hint"] = args.get("hint")  # (the model's words: data)
        elif call.name == RECOGNIZE:
            row["name"] = args.get("name")  # (the family's name for someone: data)
        elif call.name == REMEMBER:
            row["note"] = args.get("note")
        elif call.name == FEEL:
            row["valence"] = args.get("valence")
            row["talk"] = args.get("talk")
        return row

    def _people(self) -> dict[str, Any]:
        brain = self._brain()
        named: list[dict[str, Any]] = []
        nameless = 0
        try:
            people = list(brain.learn.people.values())
        except Exception:
            people = []
        for person in people:
            if person.name:
                named.append(
                    {
                        "pid": int(person.pid),
                        "name": str(person.name),
                        "greeted": int(person.greeted),
                        "sightings": int(person.sightings),
                        "pending": [str(n) for n in dict(person.pending)],
                    }
                )
            else:
                nameless += 1
        return {"named": named, "nameless": nameless}

    def _look(
        self, bridge: LiveBridge, calls: Sequence[Mapping[str, Any]], now: float
    ) -> dict[str, Any]:
        sent = self._sent
        attended = next((c for c in calls if c["tool"] == ATTEND), None)
        view = bridge.perception.view()
        return {
            "frame": None
            if sent is None
            else {
                "seq": sent.seq,
                "age_s": round(max(0.0, self._wall() - sent.wall), 1),
                "bytes": len(sent.jpeg),
                "width": sent.width,
                "height": sent.height,
                "url": f"{ROUTE}/eye.jpg?v={sent.seq}",
            },
            "frames_sent": int(self.meter.totals["frames"]),
            "attended": attended,
            "in_view": {
                "people": len(view["people"]),
                "things": len(view["things"]),
                "sounds": len(view["sounds"]),
            },
        }

    def status(self) -> dict[str, Any]:
        """Everything the talk panel shows. ``cloud`` says whether the switch is on, the state of the
        connection (words) and what is leaving the machine right now (``sending``: the sound, the
        duck's eye, the text)."""
        session, bridge = self.session, self.bridge
        base: dict[str, Any] = {
            "ok": True,
            "enabled": True,
            "ready": session is not None and bridge is not None,
            "duck": {"name": self.name},
            "personas": [p.id for p in PERSONAS],
            "places": self.places(),
        }
        library = self.library()
        base["lines"] = [] if library is None else [library.get(i).view() for i in library.ids()]
        if session is None or bridge is None:
            base["error"] = self.error
            base["cloud"] = {
                "on": False,
                "state": "off",
                "words": msg("neuro.live.status.off"),
                "understands_speech": False,
                "sending": {"audio": False, "video": False, "text": False},
            }
            return base
        now = self._clock()
        report = session.status()
        calls = self._calls(bridge, now)
        mic = getattr(getattr(getattr(self._lanes, "sound", None), "mic", None), "view", None)
        base.update(
            cloud=self._cloud(report),
            usage=self._usage(report),
            heard=self._heard(session, now),
            calls=calls,
            look=self._look(bridge, calls, now),
            people=self._people(),
            mic=mic() if callable(mic) else None,
            gate=bridge.gate.status(),
            decision=self._decision(bridge),
            faults=self.faults + session.faults + bridge.faults,
        )
        return base

    def _decision(self, bridge: LiveBridge) -> dict[str, Any]:
        """What the duck is weighing right now (for whoever looks into why it did not answer): its
        act, its fear and sleepiness, the candidates the cloud's proposals made, how much each pulls."""
        brain = self._brain()
        view = bridge.view()
        out: dict[str, Any] = {
            "intents": view["intents"],
            "extras": view["extras"],
            "talk": view["talk"],
            "speaking": view["speaking"],
            "voiced": view["voiced"],
            "barge": view["barge"],
            "reasons": view["reasons"],
            "vad": view["vad"],
            "barges": [b.view() for b in list(bridge.barges)[-6:]],
            "ear": view["ear"],  # (what the duck's own ear told the cloud)
        }
        try:
            out["behaviour"] = str(brain.behaviour)
            out["fear"] = round(float(brain.drives.fear), 3)
            out["sleepiness"] = round(float(brain.drives.sleepiness), 3)
            out["social"] = round(float(brain.drives.social), 3)
        except Exception:
            pass
        return out


def caps_from_minutes(minutes: float | None) -> UsageCaps | None:
    """``--live-voice-cap-minutes``: stop the cloud after this many minutes of microphone audio were
    sent in this run (None: no cap)."""
    if minutes is None:
        return None
    return UsageCaps(audio_s=float(minutes) * 60.0)


__all__ = [
    "OUTCOMES",
    "PLACES",
    "REASONS",
    "LiveVoice",
    "SentFrame",
    "caps_from_minutes",
]
