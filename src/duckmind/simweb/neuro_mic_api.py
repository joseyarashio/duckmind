"""A browser microphone as a person in the home (live voice, phase LV0: the server side only).

``POST /api/neuro/mic/chunk?who=<key>&x_mm=<mm>&y_mm=<mm>`` with a body of raw **16 kHz mono
16-bit little-endian PCM** (``application/octet-stream``; the page's microphone sends a piece
every 100–250 ms) makes the person at that place speak: the audio is queued into the duck's
microphone stream (:meth:`duckmind.simweb.neuro_mic.MicStream.feed_live`) and, while there is
speech in it, the world is told that a voice sounds there (``x_neuro_stim`` ``say`` with
``live: true``: a one-second sound event at the place, renewed while the person keeps speaking).
So the loudness the duck's microphone hears and the onset its hearing model feels are the
world's: they follow the distance from where the person stands (and a wall between). The place is
optional (``x_mm``/``y_mm``, home millimetres; left out, the world puts the person where ``who``
stands, else somewhere in the room). There is no page for it yet.

Limits, because anyone who can reach the page can post here: a piece is at most
:data:`MAX_CHUNK_BYTES` (1 s of audio); a speaker may send at most :data:`RATE_X` seconds of audio
per second (with a burst of :data:`BURST_S` seconds), so the stream cannot be flooded or run
faster than real time; at most :data:`MAX_SPEAKERS` people speak at once; ``who`` is a short key of
lower-case letters, digits and ``_``; the place is within 50 m of the origin. Nothing is written to
disk: the audio lives in a queue of the newest second and the stream's ring of the last 10 s.

``GET /api/neuro/mic/status`` says what the stream and the limits are.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from collections.abc import Callable, Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from typing import Any
from urllib.parse import parse_qs

from duckmind.i18n import boundary, msg
from duckmind.world import neuro_audio as A
from duckmind.world.neuro_utterances import MAX_PLACE_MM

ROUTE = "/api/neuro/mic"
CHUNK_ROUTE = ROUTE + "/chunk"
STATUS_ROUTE = ROUTE + "/status"
MAX_CHUNK_BYTES = 2 * A.SR  # 1 s
RATE_X = 1.5  # audio seconds accepted per wall second, per speaker
BURST_S = 2.0
MAX_SPEAKERS = 3
SPEAKER_IDLE_S = 5.0  # a speaker nobody has heard from this long is not counted
SEGMENT_S = 1.0  # the world's sound for a person speaking lasts this long, renewed
RENEW_BEFORE_S = 0.35  # renewed when it has less than this left
WORLD_EVERY_S = 0.25  # at most this often a call into the world
VOICED_DBFS = -45.0  # a piece louder than this has speech in it
WHO = re.compile(r"^[a-z][a-z0-9_]{0,15}$")
BODY_TIMEOUT_S = 5.0


class MicRefusal(Exception):
    """A request the hub does not take: an HTTP status and the reason in plain words."""

    def __init__(self, status: int, words: Any) -> None:
        super().__init__(words)
        self.status = int(status)
        self.words = words


class _Bucket:
    """A token bucket in seconds of audio."""

    def __init__(self, rate: float, burst: float, now: float) -> None:
        self.rate, self.burst = rate, burst
        self.tokens = burst
        self.at = now

    def take(self, seconds: float, now: float) -> bool:
        self.tokens = min(self.burst, self.tokens + (now - self.at) * self.rate)
        self.at = now
        if seconds > self.tokens:
            return False
        self.tokens -= seconds
        return True


class _Speaker:
    def __init__(self, now: float) -> None:
        self.bucket = _Bucket(RATE_X, BURST_S, now)
        self.last = now
        self.voice_until = 0.0  # when the world's sound for this speaker ends (monotonic)
        self.place: tuple[float, float] | None = None
        self.world_at = -1e9
        self.accepted_ms = 0
        self.refused = 0


class MicHub:
    """``/api/neuro/mic/…``: the microphone's pieces in (POST ``chunk``), its state out (GET
    ``status``). ``driver_of`` finds the live neural-duck driver (its ``lanes.sound.mic`` and the
    body's ``world_call``); ``clock`` is the monotonic clock (tests move it)."""

    def __init__(
        self,
        driver_of: Callable[[], Any],
        *,
        clock: Callable[[], float] = time.monotonic,
        on_speaker: Callable[[str], None] | None = None,
    ) -> None:
        self.driver_of = driver_of
        self.on_speaker = on_speaker  # told who speaks (the live voice's family roles)
        self.clock = clock
        self._lock = threading.Lock()
        self._speakers: dict[str, _Speaker] = {}
        self.accepted_ms = 0
        self.refused = 0

    @staticmethod
    def handles(path: str) -> bool:
        return path == ROUTE or path.startswith(ROUTE + "/")

    # ── the parts the hub needs from the driver ──

    def _mic(self) -> Any:
        driver = self.driver_of()
        sound = getattr(getattr(driver, "lanes", None), "sound", None)
        return getattr(sound, "mic", None)

    def _world_call(self) -> Callable[..., Any] | None:
        driver = self.driver_of()
        call = getattr(getattr(driver, "_body", None), "world_call", None)
        return call if callable(call) else None

    # ── the core (what the tests drive) ──

    def accept(self, who: str, place: tuple[float, float] | None, pcm: bytes) -> dict[str, Any]:
        """Take one piece of audio from ``who`` (at ``place`` in home millimetres). Raises
        :class:`MicRefusal` with the reason; otherwise says what was taken."""
        if not WHO.match(who):
            raise MicRefusal(HTTPStatus.BAD_REQUEST, msg("neuro.web.neuro_mic_api.bad_who"))
        if place is not None and not (
            all(math.isfinite(v) for v in place) and max(abs(v) for v in place) <= MAX_PLACE_MM
        ):
            raise MicRefusal(HTTPStatus.BAD_REQUEST, msg("neuro.world_refusal.say_where"))
        if not pcm or len(pcm) % 2:
            raise MicRefusal(HTTPStatus.BAD_REQUEST, msg("neuro.web.neuro_mic_api.bad_audio"))
        if len(pcm) > MAX_CHUNK_BYTES:
            raise MicRefusal(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                msg(
                    "neuro.web.neuro_mic_api.too_big",  # gitleaks:allow
                    seconds=MAX_CHUNK_BYTES // (2 * A.SR),
                ),
            )
        mic = self._mic()
        if mic is None:
            raise MicRefusal(HTTPStatus.SERVICE_UNAVAILABLE, msg("neuro.web.neuro_mic_api.no_mic"))
        call = self._world_call()
        if call is None:
            raise MicRefusal(
                HTTPStatus.SERVICE_UNAVAILABLE, msg("neuro.web.neuro_mic_api.no_world")
            )
        seconds = len(pcm) / 2 / A.SR
        now = self.clock()
        voiced = A.rms_dbfs(A.to_float(pcm)) > VOICED_DBFS
        with self._lock:
            for key in [k for k, v in self._speakers.items() if now - v.last > SPEAKER_IDLE_S]:
                del self._speakers[key]
            speaker = self._speakers.get(who)
            if speaker is None:
                if len(self._speakers) >= MAX_SPEAKERS:
                    self.refused += 1
                    raise MicRefusal(
                        HTTPStatus.TOO_MANY_REQUESTS,
                        msg("neuro.web.neuro_mic_api.too_many", count=MAX_SPEAKERS),
                    )
                speaker = self._speakers[who] = _Speaker(now)
            if not speaker.bucket.take(seconds, now):
                speaker.refused += 1
                self.refused += 1
                raise MicRefusal(
                    HTTPStatus.TOO_MANY_REQUESTS, msg("neuro.web.neuro_mic_api.too_fast")
                )
            speaker.last = now
            moved = place is not None and speaker.place != place
            running_out = now >= speaker.voice_until - RENEW_BEFORE_S
            renew = voiced and (running_out or moved) and now - speaker.world_at >= WORLD_EVERY_S
            if renew:
                speaker.world_at = now  # (claimed: no other piece asks while this one is out)
        event: dict[str, Any] | None = None
        if renew:
            request: dict[str, Any] = {
                "stim": "say",
                "live": True,
                "who": who,
                "seconds": SEGMENT_S,
            }
            if place is not None:
                request["where_mm"] = [place[0], place[1]]
            try:
                event = dict(call("x_neuro_stim", **request))
            except (ValueError, RuntimeError, ConnectionError) as error:
                with self._lock:
                    speaker.world_at = -1e9
                # (the world said no, in its own words: an answer, not an HTTP error)
                raise MicRefusal(
                    HTTPStatus.OK, error.args[0] if error.args else str(error)
                ) from error
            with self._lock:
                speaker.voice_until = now + SEGMENT_S
                if place is not None:
                    speaker.place = place
            self._wake()
        taken = mic.feed_live(who, pcm)
        if voiced and self.on_speaker is not None:
            self.on_speaker(who)
        with self._lock:
            speaker.accepted_ms += taken["accepted_ms"]
            self.accepted_ms += taken["accepted_ms"]
        out: dict[str, Any] = {"ok": True, "who": who, "voiced": voiced, **taken}
        if event is not None:
            out["sound_id"] = (event.get("sound") or {}).get("id")
        return out

    def _wake(self) -> None:
        wake = getattr(self.driver_of(), "_aux_wake", None)
        if wake is not None:
            wake.set()  # (the world's truth is read again at once)

    def status(self) -> dict[str, Any]:
        mic = self._mic()
        with self._lock:
            speakers = {
                who: {"accepted_ms": s.accepted_ms, "refused": s.refused}
                for who, s in self._speakers.items()
            }
            out: dict[str, Any] = {
                "ok": True,
                "running": mic is not None,
                "accepted_ms": self.accepted_ms,
                "refused": self.refused,
                "speakers": speakers,
                "limits": {
                    "format": "pcm_s16le;rate=16000;channels=1",
                    "max_chunk_bytes": MAX_CHUNK_BYTES,
                    "rate_x": RATE_X,
                    "burst_s": BURST_S,
                    "max_speakers": MAX_SPEAKERS,
                    "max_place_mm": MAX_PLACE_MM,
                },
            }
        if mic is not None:
            out["mic"] = mic.view()
        return out

    # ── HTTP ──

    def get(self, handler: BaseHTTPRequestHandler, path: str, query: str) -> None:
        if path in (ROUTE, STATUS_ROUTE):
            _send(handler, HTTPStatus.OK, self.status())
        else:
            _send(handler, HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})

    def post(self, handler: BaseHTTPRequestHandler, path: str, query: str = "") -> None:
        if path != CHUNK_ROUTE:
            _send(handler, HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
            return
        try:
            who, place = _params(query)
            raw = handler.headers.get("Content-Length")
            if raw is None or not raw.strip().isdigit():
                raise MicRefusal(
                    HTTPStatus.LENGTH_REQUIRED, msg("neuro.web.neuro_mic_api.no_length")
                )
            length = int(raw)
            if length > MAX_CHUNK_BYTES:  # (refused before a byte of it is read)
                raise MicRefusal(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    msg(
                        "neuro.web.neuro_mic_api.too_big",  # gitleaks:allow
                        seconds=MAX_CHUNK_BYTES // (2 * A.SR),
                    ),
                )
            connection = getattr(handler, "connection", None)
            if connection is not None and hasattr(connection, "settimeout"):
                connection.settimeout(BODY_TIMEOUT_S)
            body = handler.rfile.read(length)
            if len(body) != length:
                raise MicRefusal(HTTPStatus.BAD_REQUEST, msg("neuro.web.neuro_mic_api.bad_audio"))
            _send(handler, HTTPStatus.OK, self.accept(who, place, body))
        except MicRefusal as refusal:
            _send(handler, refusal.status, {"ok": False, "error": refusal.words})
        except OSError:
            return  # (the page went away mid-body)


def _params(query: str) -> tuple[str, tuple[float, float] | None]:
    """``who`` and the optional place from the query string."""
    params = parse_qs(query, keep_blank_values=False)
    who = (params.get("who") or ["guest"])[0]
    xs, ys = params.get("x_mm"), params.get("y_mm")
    if xs is None and ys is None:
        return who, None
    try:
        if xs is None or ys is None:
            raise ValueError("one of the two")
        place = (float(xs[0]), float(ys[0]))
    except ValueError:
        raise MicRefusal(HTTPStatus.BAD_REQUEST, msg("neuro.world_refusal.say_where")) from None
    return who, place


def _send(handler: BaseHTTPRequestHandler, status: int, payload: Mapping[str, Any]) -> None:
    body = json.dumps(boundary.attach(dict(payload)), ensure_ascii=False).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("X-Content-Type-Options", "nosniff")
    handler.end_headers()
    handler.wfile.write(body)


__all__ = ["BURST_S", "MAX_CHUNK_BYTES", "MAX_SPEAKERS", "RATE_X", "MicHub", "MicRefusal"]
