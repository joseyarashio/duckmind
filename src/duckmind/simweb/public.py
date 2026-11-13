"""Public play (``python -m duckmind.simweb --public``): one visitor, one fresh duck, a set time.

The server runs inside one container per visitor (docs/design and the deploy's contract): a new duck hatched
from ``DUCKMIND_PUBLIC_SEED`` in a temporary folder, already :data:`PUBLIC_GD` growth days old, named from its
seed (:func:`name_for`). Only the endpoints the player's page needs exist (:func:`allows_get`,
:func:`allows_post`); anything else answers 404 with one plain reason. After ``DUCKMIND_PUBLIC_MINUTES`` the
duck sits and rests, every write answers 410 with a plain reason, the duck file can still be downloaded for
:data:`EXPORT_GRACE_S`, and then the process ends (exit 0). No answer carries a file path or the host's name
(:class:`Scrubbing`), and the process opens no connection off this machine (:func:`install_network_guard`).
"""

from __future__ import annotations

import io
import ipaddress
import json
import os
import re
import socket
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from pathlib import Path
from typing import Any

from duckmind.i18n import boundary, msg

SEED_ENV = "DUCKMIND_PUBLIC_SEED"
MINUTES_ENV = "DUCKMIND_PUBLIC_MINUTES"
DEFAULT_MINUTES = 5.0  # (owner 2026-11-14: one visitor at a time, 5 minutes each)
MAX_MINUTES = 120.0
# After the time is up, the duck file can still be downloaded this long; then the process ends (owner budget
# decision 2026-11-10: 3 minutes).
EXPORT_GRACE_S = 180.0
# The age a visitor's duck hatches at (backdated, lived at once at start-up). Growth G1: no duck walks
# before GD 5 × its pace, and the pace of every seed is 0.87–1.15, so every duck walks from GD 6 on; at
# GD 32 every duck also has its nod and peekaboo (14 × pace ≤ 16), its call-and-answer (18 × pace ≤ 21),
# snuggle and its little dance (28 × pace ≤ 32), and none is a juvenile yet (45 × pace ≥ 39): a young
# duck that walks and plays, with its branch (and 蛋黃's explorer look) still ahead of it.
PUBLIC_GD = 32
# The duck's names: data (never translated), short Latin letters every language's page can show.
NAMES = (
    "Pip", "Mochi", "Nori", "Kiki", "Tofu", "Bao", "Lulu", "Momo",
    "Pico", "Coco", "Miso", "Yuzu", "Taro", "Nana", "Kiwi", "Popo",
)  # fmt: skip
SESSION_ROUTE = "/api/public/session"
REPO_URL = "https://github.com/joseyarashio/duckmind"
# What the page greys out (public.js's gates), each with the one plain reason.
REFUSES = (
    "body_link",
    "clock",
    "duck_picker",
    "duckfile_import",
    "lab",
    "layout",
    "live_voice",
    "remote",
    "voice_commands",
)

# ── the endpoints that exist ─────────────────────────────────────────────────

GET_EXACT = frozenset(
    {
        "/",
        "/index.html",
        "/api/ping",
        SESSION_ROUTE,
        "/api/neuro",
        "/api/robot",
        "/api/body",
        "/api/home",
        "/api/eye.jpg",
        "/api/clips",
        "/api/life",
        "/api/life/growth",
        "/api/life/duckfile",
        "/api/life/duckfile/export",
        "/api/neuro/status",
        "/api/neuro/telemetry",
        "/api/neuro/stream",
        "/api/neuro/knows",
        "/api/neuro/sound/status",
        "/api/neuro/sound/pcm",
        # (features that are off or out of reach: their status reads say so, the page greys them out;
        # the fast-forward test's status is read by the growth view on every page load)
        "/api/life/fastforward",
        "/api/live/status",
        "/api/neuro/voice/status",
    }
)
# The page's own files: its fonts and libraries, the duck's model, the duck's own voice.
GET_PREFIXES = ("/fonts/", "/vendor/", "/mesh/", "/api/voice/")
# A top-level page file (static/): its name and one of the server's static types.
_STATIC_FILE = re.compile(r"^/[A-Za-z0-9_][A-Za-z0-9_.-]*\.(?:html|css|js|svg|woff2)$")
# What a visitor does with the duck (each a write: 410 once the time is up).
POST_PLAY = frozenset(
    {
        "/api/neuro/stimulus",
        "/api/neuro/help",
        "/api/neuro/command",
        "/api/neuro/feedback",
        "/api/neuro/gothere",
        "/api/neuro/motion",
        "/api/neuro/sound/world",
    }
)
# Duck Core's page calls (POST /api/call): only the ones that read.
CALL_ROUTE = "/api/call"
CALL_READS = frozenset({"state", "diary", "events", "event", "why", "graph", "fact"})


def allows_get(path: str) -> bool:
    """A GET (path without its query) that exists in public mode."""
    if path in GET_EXACT or _STATIC_FILE.match(path):
        return True
    return path.startswith(GET_PREFIXES) and ".." not in path


def allows_post(path: str) -> bool:
    """A POST that exists in public mode (``/api/call`` only for :data:`CALL_READS`: :func:`call_reads`)."""
    return path in POST_PLAY or path == CALL_ROUTE


def call_reads(body: bytes) -> bool:
    """An ``/api/call`` body whose method only reads."""
    try:
        request = json.loads(body or b"{}")
    except ValueError:
        return False
    return isinstance(request, dict) and str(request.get("method", "state")) in CALL_READS


# ── the duck ────────────────────────────────────────────────────────────────


class PublicConfigError(ValueError):
    """The container's settings are missing or wrong (said on the console; the server does not start)."""


def seed_from_env(environ: Mapping[str, str]) -> int:
    raw = str(environ.get(SEED_ENV, "")).strip()
    if not re.fullmatch(r"\d{1,10}", raw) or int(raw) > 2**31 - 1:
        raise PublicConfigError(f"--public needs {SEED_ENV}: a whole number 0..{2**31 - 1}")
    return int(raw)


def minutes_from_env(environ: Mapping[str, str]) -> float:
    raw = str(environ.get(MINUTES_ENV, "")).strip()
    if not raw:
        return DEFAULT_MINUTES
    try:
        minutes = float(raw)
    except ValueError:
        minutes = -1.0
    if not 0.0 < minutes <= MAX_MINUTES:
        raise PublicConfigError(
            f"{MINUTES_ENV} must be a number of minutes in (0, {MAX_MINUTES:g}]"
        )
    return minutes


def name_for(seed: int) -> str:
    """The duck's name: one of :data:`NAMES`, always the same for a seed."""
    return NAMES[int(seed) % len(NAMES)]


# What ``--public`` picks itself: given on the command line too, the server does not start.
PICKED = (
    "seed",
    "name",
    "life",
    "life_backdate_days",
    "life_birth",
    "life_eye_from",
    "live_voice",
    "live_voice_name",
    "live_voice_cap_minutes",
    "body",
    "body_token_file",
    "neuro_stub",
    "neuro_duck",
    "neuro_time",
    "neuro_record",
    "clock_speed",
    "layouts_dir",
)


@dataclass(frozen=True)
class PublicPlan:
    seed: int
    name: str
    minutes: float
    folder: Path  # (a fresh temporary folder: the duck's life and its layouts; removed at the end)


def apply_args(
    args: Any, default_of: Callable[[str], Any], environ: Mapping[str, str]
) -> PublicPlan:
    """Fill the server's arguments for public play (``default_of``: the parser's defaults): the duck's seed
    from the environment, its name, its age, a new life in a temporary folder, the neural duck in its live
    body without a nest, no voice commands. PublicConfigError: a flag it picks itself was given, or the
    environment is wrong."""
    given = [
        f"--{dest.replace('_', '-')}"
        for dest in PICKED
        if getattr(args, dest, None) != default_of(dest)
    ]
    if given:
        raise PublicConfigError("--public picks these itself: " + ", ".join(given))
    if str(args.voice_commands) not in ("auto", "off"):
        raise PublicConfigError("--public has no voice commands (leave out --voice-commands)")
    if not args.live:
        raise PublicConfigError("--public runs the duck's live body: add --live and its paths")
    seed = seed_from_env(environ)
    minutes = minutes_from_env(environ)
    folder = Path(tempfile.mkdtemp(prefix="duckmind-public-"))
    args.brain = "neuro"
    args.no_nest = True
    args.voice_commands = "off"
    args.home = args.home or "real_home"
    args.seed = seed
    args.name = name_for(seed)
    args.life = folder / "duck"
    args.life_backdate_days = PUBLIC_GD
    args.layouts_dir = folder / "layouts"
    return PublicPlan(seed, args.name, minutes, folder)


# ── the visit's time ─────────────────────────────────────────────────────────


class PublicSession:
    """One visit: ``minutes`` of play from the moment it is made, then :data:`EXPORT_GRACE_S` in which only
    the duck file can be taken, then the end. ``wall``: the clock (epoch seconds; tests use their own).
    ``on_end``: called once, in order, when the time is up (the duck sits); ``on_exit``: when the grace is
    over (the server stops)."""

    def __init__(
        self,
        seed: int,
        name: str,
        minutes: float = DEFAULT_MINUTES,
        *,
        wall: Callable[[], float] = time.time,
        grace_s: float = EXPORT_GRACE_S,
    ) -> None:
        self.seed = int(seed)
        self.name = str(name)
        self.wall = wall
        self.started = float(wall())
        self.ends = self.started + float(minutes) * 60.0
        self.exits = self.ends + float(grace_s)
        self.grace_minutes = max(1, round(float(grace_s) / 60.0))
        self.on_end: list[Callable[[], None]] = []
        self.on_exit: list[Callable[[], None]] = []
        self._lock = threading.Lock()
        self._ended_done = False
        self._exit_done = False

    def left_s(self) -> float:
        return max(0.0, self.ends - float(self.wall()))

    def ended(self) -> bool:
        return float(self.wall()) >= self.ends

    def view(self) -> dict[str, Any]:
        """GET /api/public/session (the contract's fields, then what the page needs besides)."""
        now = float(self.wall())
        return {
            "ok": True,
            "seed": self.seed,
            "name": self.name,
            "started_at": _iso(self.started),
            "ends_at": _iso(self.ends),
            "left_s": round(max(0.0, self.ends - now), 1),
            "ended": now >= self.ends,
            "export_until": _iso(self.exits),
            "export_left_s": round(max(0.0, self.exits - now), 1),
            "export_minutes": self.grace_minutes,
            "refuses": list(REFUSES),
            "repo": REPO_URL,
        }

    def check(self) -> str:
        """Where the visit is (``play``, ``ended``, ``over``); runs ``on_end`` once when the time is up and
        ``on_exit`` once when the grace is over. A hook's fault is printed, never stops the rest."""
        now = float(self.wall())
        hooks: list[Callable[[], None]] = []
        with self._lock:
            if now >= self.ends and not self._ended_done:
                self._ended_done = True
                hooks += self.on_end
            if now >= self.exits and not self._exit_done:
                self._exit_done = True
                hooks += self.on_exit
        for hook in hooks:
            try:
                hook()
            except Exception as error:
                print(f"public: {type(error).__name__}: {error}", file=sys.stderr)
        return "over" if now >= self.exits else "ended" if now >= self.ends else "play"

    def run(self, stop: threading.Event, every_s: float = 1.0) -> None:
        """The visit's own thread: :meth:`check` every ``every_s`` until it is over or ``stop`` is set."""
        while not stop.is_set():
            if self.check() == "over":
                return
            stop.wait(every_s)


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, UTC).isoformat(timespec="seconds")


def refused() -> dict[str, Any]:
    """A feature the visitor does not have (404): the one plain reason."""
    return {"ok": False, "public": "refused", "error": msg("web.public.refused")}


def ended(minutes: int) -> dict[str, Any]:
    """A write after the time is up (410); ``minutes``: how long the duck file can still be downloaded."""
    return {"ok": False, "public": "ended", "error": msg("web.public.ended_grace", minutes=minutes)}


def send_json(handler: Any, status: int, payload: Mapping[str, Any]) -> None:
    body = json.dumps(boundary.attach(dict(payload)), ensure_ascii=False).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("X-Content-Type-Options", "nosniff")
    handler.end_headers()
    handler.wfile.write(body)


def gate(handler: Any, session: PublicSession, method: str) -> bool:
    """Answer the request here when public mode does not pass it on (True: answered). A POST to
    ``/api/call`` is read here (its method decides) and put back for the server to read again."""
    path = handler.path.partition("?")[0]
    if method == "GET":
        if path == SESSION_ROUTE:
            send_json(handler, HTTPStatus.OK, session.view())
            return True
        if not allows_get(path):
            send_json(handler, HTTPStatus.NOT_FOUND, refused())
            return True
        return False
    if not allows_post(path):
        send_json(handler, HTTPStatus.NOT_FOUND, refused())
        return True
    if path == CALL_ROUTE:
        raw = handler.headers.get("Content-Length", "0") or "0"
        length = int(raw) if raw.isdigit() else -1
        if not 0 <= length <= 256 * 1024:
            send_json(handler, HTTPStatus.BAD_REQUEST, refused())
            return True
        body = handler.rfile.read(length)
        if not call_reads(body):
            send_json(handler, HTTPStatus.GONE if session.ended() else HTTPStatus.NOT_FOUND,
                      ended(session.grace_minutes) if session.ended() else refused())  # fmt: skip
            return True
        handler.rfile = io.BytesIO(body)
        return False
    if session.ended():
        send_json(handler, HTTPStatus.GONE, ended(session.grace_minutes))
        return True
    return False


# ── no file paths or host details in any answer ──────────────────────────────

# An absolute path of this machine (two or more parts, or under ~): not a page address of this server.
_PATH = re.compile(r"(?<![\w.:/~-])~?/(?:[^\s\"'\\/<>|:;,()\[\]{}=]+/)+[^\s\"'\\/<>|:;,()\[\]{}=]*")
PAGE_URLS = ("/api/", "/mesh/", "/fonts/", "/vendor/")
_SCRUBBED_TYPES = ("application/json", "text/plain", "text/event-stream")
HIDDEN = "…"


def secrets() -> list[str]:
    """Words of this machine that never leave it: its host name, the home folder, the temp folder."""
    out = []
    host = socket.gethostname()
    if len(host) >= 4:
        out.append(host)
    for folder in (os.path.expanduser("~"), os.environ.get("TMPDIR", ""), "/tmp"):
        folder = folder.rstrip("/")
        if len(folder) >= 4:
            out.append(folder)
    return sorted(set(out), key=len, reverse=True)


def scrub(text: str, hidden: Sequence[str] = ()) -> str:
    """``text`` without this machine's paths and ``hidden`` words (each becomes :data:`HIDDEN`)."""
    for word in hidden:
        if word:
            text = text.replace(word, HIDDEN)
    return _PATH.sub(lambda m: m.group(0) if m.group(0).startswith(PAGE_URLS) else HIDDEN, text)


class _ScrubWriter:
    """The handler's output while a scrubbed answer is written: holds a sized body until it is whole, then
    sends its new length and the body; a stream (no size) is scrubbed write by write."""

    def __init__(self, handler: Scrubbing, raw: Any) -> None:
        self.handler = handler
        self.raw = raw
        self.buffer = bytearray()

    def write(self, data: bytes) -> int:
        handler = self.handler
        if handler._dm_mode == "sized":
            self.buffer += data
            if len(self.buffer) >= (handler._dm_length or 0):
                body = scrub(
                    bytes(self.buffer).decode("utf-8", "replace"), handler._dm_hidden
                ).encode()
                self.buffer.clear()
                handler._dm_mode = None
                handler._dm_send_header("Content-Length", str(len(body)))
                handler._dm_end_headers()
                self.raw.write(body)
            return len(data)
        if handler._dm_mode == "stream":
            self.raw.write(scrub(data.decode("utf-8", "replace"), handler._dm_hidden).encode())
            return len(data)
        return int(self.raw.write(data) or 0)

    def flush(self) -> None:
        self.raw.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.raw, name)


class Scrubbing:
    """A request handler mixin (before the handler class): every text answer (JSON, plain text, a stream)
    leaves without this machine's paths and :func:`secrets`, whichever part of the server wrote it."""

    _dm_hidden: Sequence[str] = ()

    def setup(self) -> None:
        super().setup()  # type: ignore[misc]
        self.wfile = _ScrubWriter(self, self.wfile)
        self._dm_reset()

    def _dm_reset(self) -> None:
        self._dm_type = ""
        self._dm_length: int | None = None
        self._dm_mode: str | None = None

    def send_response(self, code: int, message: str | None = None) -> None:
        self._dm_reset()
        super().send_response(code, message)  # type: ignore[misc]

    def send_header(self, keyword: str, value: str) -> None:
        key = keyword.lower()
        if key == "content-length":
            self._dm_length = int(value)
            return  # (sent at the end of the headers, or after the body is scrubbed)
        if key == "content-type":
            self._dm_type = str(value).split(";", 1)[0].strip().lower()
        super().send_header(keyword, value)  # type: ignore[misc]

    def _dm_send_header(self, keyword: str, value: str) -> None:
        super().send_header(keyword, value)  # type: ignore[misc]

    def _dm_end_headers(self) -> None:
        super().end_headers()  # type: ignore[misc]

    def end_headers(self) -> None:
        text = self._dm_type in _SCRUBBED_TYPES
        if text and self._dm_length:
            self._dm_mode = "sized"  # (the headers go out with the scrubbed body's length)
            return
        if self._dm_length is not None:
            self._dm_send_header("Content-Length", str(self._dm_length))
        self._dm_mode = None
        super().end_headers()  # type: ignore[misc]
        if text:
            self._dm_mode = "stream"


# ── no connection off this machine ───────────────────────────────────────────


def _local_host(host: Any) -> bool:
    if host is None or host in ("", "localhost", b"localhost"):
        return True
    text = host.decode() if isinstance(host, bytes) else str(host)
    try:
        return ipaddress.ip_address(text.split("%", 1)[0]).is_loopback or text in ("0.0.0.0", "::")
    except ValueError:
        return False


def outbound(event: str, args: tuple[Any, ...]) -> bool:
    """Whether a socket audit event would reach off this machine (a unix socket or loopback never does)."""
    if event == "socket.getaddrinfo":
        return not _local_host(args[0])
    if event in ("socket.connect", "socket.sendto", "socket.sendmsg"):
        address = args[1] if len(args) > 1 else None
        if not isinstance(address, tuple):
            return False  # (a unix socket's path, or none)
        return not _local_host(address[0])
    return False


_guard_installed = False


def install_network_guard() -> None:
    """From now on this process refuses (OSError) every lookup and connection off this machine. An audit
    hook (``sys.addaudithook``) cannot be taken back: only for a ``--public`` process."""
    global _guard_installed
    if _guard_installed:
        return

    def hook(event: str, args: tuple[Any, ...]) -> None:
        if event.startswith("socket.") and outbound(event, args):
            raise PermissionError("public mode: no network off this machine")

    sys.addaudithook(hook)
    _guard_installed = True


__all__ = [
    "EXPORT_GRACE_S",
    "MINUTES_ENV",
    "NAMES",
    "PUBLIC_GD",
    "REFUSES",
    "SEED_ENV",
    "SESSION_ROUTE",
    "PublicConfigError",
    "PublicSession",
    "Scrubbing",
    "allows_get",
    "allows_post",
    "gate",
    "install_network_guard",
    "minutes_from_env",
    "name_for",
    "scrub",
    "secrets",
    "seed_from_env",
]
