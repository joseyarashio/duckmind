"""Voice commands: the page's 「按住說話」 clip → local speech recognition → a command or the owner's word.

``POST /api/neuro/voice/command?lang=<page language>`` with a body of raw **16 kHz mono 16-bit
little-endian PCM** (``application/octet-stream``, at most :data:`duckmind.speech.asr.MAX_S` seconds:
what the page recorded while the button was held). The clip is recognised on this computer
(:mod:`duckmind.speech.asr`: no cloud, nothing written to disk) and the words are read by
:func:`duckmind.neuro.commands.understand`:

* a command (「坐下」) goes to the duck exactly as typed words do (``NeuroHub.command`` with
  ``source: "voice"``): the duck hears it and decides from its own state;
* the owner's word (「乖」「好棒」 → ``praise``, 「不對」 → ``scold``) goes to ``NeuroHub.feedback``, as the
  feedback buttons do;
* anything else does nothing.

The answer always says what was heard and what it was understood as, so the page can show both::

    {"ok": true, "heard": {"text", "lang", "audio_s", "took_s", "empty"},
     "understood": {"kind": "command" | "feedback" | "none", "verb"?, "feedback"?, "reason"?},
     "answer": <the command's or the feedback's own answer> | null, "took_s": <server seconds>}

A command or feedback the duck cannot take is still a 200 here (the hearing worked); its own answer
carries ``ok: false`` and the ``reason``. A clip that cannot be heard at all is a 400 (``reason``:
``bad_lang``, ``odd_bytes``, ``too_long``, ``empty_body``), a 413 when the body is too large, a 429 when
a recognition is already running, a 503 when there is no recogniser (``reason``: ``no_recognizer``,
with ``problem``) or the duck's brain is not on (``not_running``).

``GET /api/neuro/voice/status``: ``{"enabled", "ready", "engine", "model", "problem", "max_s",
"langs"}`` — the page enables the button from it.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from typing import Any
from urllib.parse import parse_qs

from duckmind.i18n import msg
from duckmind.neuro import command_brain as CB
from duckmind.neuro import command_words as W
from duckmind.neuro.commands import CommandError
from duckmind.simweb.neuro_api import FeedbackError, _send_json
from duckmind.speech import asr
from duckmind.speech.listen import read_words

ROUTE = "/api/neuro/voice"
COMMAND_ROUTE = ROUTE + "/command"
STATUS_ROUTE = ROUTE + "/status"
MAX_BODY = int(asr.MAX_S * asr.SR * 2) + 1024
BODY_TIMEOUT_S = 10.0


class VoiceRefusal(Exception):
    def __init__(self, status: int, reason: str, **extra: Any) -> None:
        super().__init__(reason)
        self.status = int(status)
        self.reason = reason
        self.extra = extra


class VoiceHub:
    """``/api/neuro/voice/…``. ``hub`` is the page's :class:`~duckmind.simweb.neuro_api.NeuroHub` (its
    ``command``, ``feedback``, ``enabled`` and the duck's name); ``recognizer`` the local recogniser, or
    None with ``problem`` saying why there is none."""

    def __init__(
        self,
        hub: Any,
        recognizer: asr.Recognizer | None,
        *,
        problem: str | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.hub = hub
        self.recognizer = recognizer
        self.problem = problem
        self.clock = clock
        self._busy = threading.Lock()

    @staticmethod
    def handles(path: str) -> bool:
        return path == ROUTE or path.startswith(ROUTE + "/")

    def status(self) -> dict[str, Any]:
        rec = self.recognizer
        out: dict[str, Any] = {
            "ok": True,
            "enabled": rec is not None and bool(getattr(self.hub, "enabled", False)),
            "running": bool(getattr(self.hub, "enabled", False)),
            "ready": bool(rec and rec.ready),
            "engine": rec.engine.name if rec else None,
            "model": rec.engine.model if rec else None,
            "problem": (rec.problem if rec else self.problem),
            "max_s": asr.MAX_S,
            "rate": asr.SR,
            "langs": list(W.LANGS),
        }
        return out

    def _duck_name(self) -> str | None:
        choice = getattr(self.hub, "choice", None)
        name = getattr(choice, "name", None)
        return str(name) if name else None

    def command(self, pcm: bytes, lang: str) -> dict[str, Any]:
        """One clip: heard, understood, and handed to the duck. Raises :class:`VoiceRefusal`."""
        begun = self.clock()
        rec = self.recognizer
        if rec is None:
            raise VoiceRefusal(HTTPStatus.SERVICE_UNAVAILABLE, "no_recognizer", problem=self.problem)
        if not getattr(self.hub, "enabled", False):
            raise VoiceRefusal(HTTPStatus.SERVICE_UNAVAILABLE, "not_running")
        if lang not in W.LANGS:
            raise VoiceRefusal(HTTPStatus.BAD_REQUEST, "bad_lang")
        if not pcm:
            raise VoiceRefusal(HTTPStatus.BAD_REQUEST, "empty_body")
        if not self._busy.acquire(blocking=False):
            raise VoiceRefusal(HTTPStatus.TOO_MANY_REQUESTS, "busy")
        try:
            name = self._duck_name()
            try:
                heard = rec.hear(pcm, lang, name)
            except asr.Refused as error:
                raise VoiceRefusal(HTTPStatus.BAD_REQUEST, error.reason) from None
            except Exception as error:  # (the model could not load or run: said on the page)
                rec.problem = f"{type(error).__name__}: {error}"
                raise VoiceRefusal(HTTPStatus.SERVICE_UNAVAILABLE, "recognizer_failed",
                                   problem=rec.problem) from None  # fmt: skip
        finally:
            self._busy.release()
        heard_out = {"text": heard.text, "lang": heard.lang, "audio_s": heard.audio_s,
                     "took_s": heard.took_s, "empty": heard.empty}  # fmt: skip
        spoken = read_words(heard, lang, name)
        got = spoken.understood
        understood: dict[str, Any] = {"kind": "none", "reason": got.reason}
        answer: dict[str, Any] | None = None
        words = spoken.by_sound or heard.text  # (what the duck is told: the table phrase when found by sound)
        if got.verb is not None:
            understood = {"kind": "command", "verb": got.verb, "lang": got.lang,
                          "confidence": got.confidence}  # fmt: skip
            answer = self._command(words, lang)
        elif got.feedback is not None:
            understood = {"kind": "feedback", "feedback": got.feedback, "lang": got.lang,
                          "confidence": got.confidence}  # fmt: skip
            answer = self._feedback(got.feedback)
        if spoken.by_sound and understood["kind"] != "none":
            understood["by_sound"] = spoken.by_sound
        return {"ok": True, "heard": heard_out, "understood": understood, "answer": answer,
                "took_s": round(self.clock() - begun, 3)}  # fmt: skip

    def _command(self, text: str, lang: str) -> dict[str, Any]:
        try:
            return dict(self.hub.command({"text": text, "lang": lang, "source": "voice"}))
        except CommandError as error:
            return {"ok": False, "reason": error.reason, "error": CB.error_words(error)}
        except ValueError as error:  # (the brain stopped meanwhile)
            return {"ok": False, "reason": "not_running", "error": str(error)}

    def _feedback(self, kind: str) -> dict[str, Any]:
        try:
            return dict(self.hub.feedback({"kind": kind}))
        except FeedbackError as error:
            return {"ok": False, "reason": error.reason,
                    "error": msg(f"neuro.feedback.error.{error.reason}")}  # fmt: skip

    # ── HTTP ──

    def get(self, handler: BaseHTTPRequestHandler, path: str, query: str) -> None:
        if path == STATUS_ROUTE:
            _send_json(handler, HTTPStatus.OK, self.status())
        else:
            _send_json(handler, HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})

    def post(self, handler: BaseHTTPRequestHandler, path: str, query: str) -> None:
        if path != COMMAND_ROUTE:
            _send_json(handler, HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
            return
        try:
            length = int(handler.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if length < 0 or length > MAX_BODY:
            _send_json(handler, HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                       {"ok": False, "reason": "too_long"})  # fmt: skip
            return
        lang = (parse_qs(query).get("lang") or [""])[0]
        try:
            handler.connection.settimeout(BODY_TIMEOUT_S)
            pcm = handler.rfile.read(length) if length else b""
            _send_json(handler, HTTPStatus.OK, self.command(pcm, lang))
        except VoiceRefusal as refusal:
            _send_json(handler, refusal.status,
                       {"ok": False, "reason": refusal.reason, **refusal.extra})  # fmt: skip
        except TimeoutError:
            _send_json(handler, HTTPStatus.REQUEST_TIMEOUT, {"ok": False, "reason": "slow_body"})
