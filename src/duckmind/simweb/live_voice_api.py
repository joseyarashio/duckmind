"""The talk panel's HTTP routes (live voice, LV5): ``/api/live/…``.

Light on purpose (the server imports it whether or not ``--live-voice`` is on): the routes answer
``{"ok": true, "enabled": false}`` when no :class:`~duckmind.simweb.live_voice.LiveVoice` was attached,
so the page can tell by asking. The words are catalog messages (``neuro.live.web.*``), sent with their
``_i18n`` maps like the rest of the neural duck's API.

    GET  /api/live/status    what the talk panel shows
    GET  /api/live/eye.jpg   the last picture of the duck's eye that was sent to the cloud (204: none yet)
    POST /api/live/cloud     {"on": true|false}
    POST /api/live/say       {"utterance": id, "who": persona, "position": id}

A refusal in plain words (the world said no, the line does not exist) is a normal answer
(``{"ok": false, "error": <words>}`` with HTTP 200), not an HTTP error; a malformed request is a 400.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from typing import Any, Protocol

from duckmind.i18n import boundary, msg

ROUTE = "/api/live"
MAX_BODY = 4096


class Voice(Protocol):
    """What the routes need of the live conversation (:class:`~duckmind.simweb.live_voice.LiveVoice`)."""

    def status(self) -> dict[str, Any]: ...

    def set_cloud(self, on: bool) -> dict[str, Any]: ...

    def say(self, params: Mapping[str, Any]) -> dict[str, Any]: ...

    def eye_frame(self) -> Any: ...

    def close(self) -> None: ...


# ── the routes ─────────────────────────────────────────────────────────────


class LiveHub:
    """``/api/live/…``: answers whether or not the server runs live voice (``enabled``), so the page
    can tell by asking. The server hands it the :class:`LiveVoice` once the duck is up."""

    def __init__(self) -> None:
        self.voice: Voice | None = None

    def attach(self, voice: Voice | None) -> None:
        self.voice = voice

    def speaker(self, who: str) -> None:
        """The page's microphone speaks as ``who`` (a role of the chooser)."""
        speak = getattr(self.voice, "family_speaks", None)
        if callable(speak):
            speak(who)

    @staticmethod
    def handles(path: str) -> bool:
        return path == ROUTE or path.startswith(ROUTE + "/")

    def status(self) -> dict[str, Any]:
        voice = self.voice
        return {"ok": True, "enabled": False} if voice is None else voice.status()

    def get(self, handler: BaseHTTPRequestHandler, path: str, query: str = "") -> None:
        if path in (ROUTE, ROUTE + "/status"):
            _send(handler, HTTPStatus.OK, self.status())
            return
        if path == ROUTE + "/eye.jpg":
            frame = None if self.voice is None else self.voice.eye_frame()
            if frame is None:
                handler.send_response(HTTPStatus.NO_CONTENT)
                handler.send_header("Cache-Control", "no-store")
                handler.end_headers()
                return
            handler.send_response(HTTPStatus.OK)
            handler.send_header("Content-Type", "image/jpeg")
            handler.send_header("Content-Length", str(len(frame.jpeg)))
            handler.send_header("Cache-Control", "no-store")
            handler.send_header("X-Content-Type-Options", "nosniff")
            handler.end_headers()
            handler.wfile.write(frame.jpeg)
            return
        _send(handler, HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})

    def post(self, handler: BaseHTTPRequestHandler, path: str) -> None:
        raw = handler.headers.get("Content-Length", "")
        if not raw.strip().isdigit():
            _send(handler, HTTPStatus.LENGTH_REQUIRED, _refusal("neuro.live.web.no_length"))
            return
        length = int(raw)
        if length > MAX_BODY:
            _send(handler, HTTPStatus.REQUEST_ENTITY_TOO_LARGE, _refusal("neuro.live.web.too_big"))
            return
        try:
            try:
                body = json.loads(handler.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                body = None
            if not isinstance(body, dict):
                _send(handler, HTTPStatus.BAD_REQUEST, _refusal("neuro.live.web.bad_request"))
                return
            voice = self.voice
            if voice is None:
                _send(handler, HTTPStatus.OK, _refusal("neuro.live.web.not_enabled"))
            elif path == ROUTE + "/cloud":
                if not isinstance(body.get("on"), bool):
                    _send(handler, HTTPStatus.BAD_REQUEST, _refusal("neuro.live.web.bad_request"))
                    return
                _send(handler, HTTPStatus.OK, voice.set_cloud(body["on"]))
            elif path == ROUTE + "/say":
                _send(handler, HTTPStatus.OK, voice.say(body))
            else:
                _send(handler, HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
        except ValueError as error:
            # (a refusal in plain words, not an HTTP error: the page says it, nothing broke)
            words = error.args[0] if error.args else str(error)
            _send(handler, HTTPStatus.OK, {"ok": False, "error": words})
        except OSError:
            return  # (the page went away mid-body)
        except Exception as error:
            text = str(error)
            if "'error':" in text:
                text = text.split("'error':", 1)[1].strip(" }'\"")
            failed = msg("neuro.api.post.b592", error_type=text or type(error).__name__)
            _send(handler, HTTPStatus.SERVICE_UNAVAILABLE, {"ok": False, "error": failed})


def _refusal(key: str) -> dict[str, Any]:
    return {"ok": False, "error": msg(key)}


def _send(handler: BaseHTTPRequestHandler, status: int, payload: Mapping[str, Any]) -> None:
    body = json.dumps(boundary.attach(dict(payload)), ensure_ascii=False).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("X-Content-Type-Options", "nosniff")
    handler.end_headers()
    handler.wfile.write(body)


__all__ = ["MAX_BODY", "ROUTE", "LiveHub", "Voice"]
