"""The duck file's HTTP routes (docs/duck-file.md): ``/api/life/duckfile/…``.

    GET  /api/life/duckfile          {"ok": true, "life": bool}: whether this server lives a duck
                                     (only then is there one to export)
    GET  /api/life/duckfile/export   the duck file of the duck this server lives (a download). Its
                                     mind is saved first, then Duck Core's save, so the file is now.
    POST /api/life/duckfile/inspect  body: the file's bytes. What is in it and whether this
                                     computer has the same duck already (nothing is written)
    POST /api/life/duckfile/import   body: the file's bytes; ``?replace=1`` replaces the same duck
                                     (its folder kept as a backup). Puts the duck in a new life
                                     folder; the server goes on living its own duck (start a server
                                     with ``--life <folder>`` to live the imported one)

The file holds people's names and looks, so these routes answer only a browser on this computer
(a loopback client), and a POST must carry the ``X-Duckmind-Duckfile: 1`` header (a page of another
site cannot send it without asking first, and this server never says yes). A refusal in plain words
is ``{"ok": false, "error": <words>}`` with HTTP 200 (the page shows it); a malformed request is 4xx.
"""

from __future__ import annotations

import ipaddress
import json
import traceback
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote

from duckmind.i18n import boundary, msg
from duckmind.life import duckfile as df

ROUTE = "/api/life/duckfile"
HEADER = "X-Duckmind-Duckfile"
MEDIA_TYPE = "application/vnd.duckmind.duck+zip"


class DuckFileHub:
    """``/api/life/duckfile/…`` for a server (``state``: its :class:`~duckmind.simweb.server.SimState`;
    ``state.life_host`` is the duck it lives, or None). ``any_client``: public play — the export answers the
    visitor wherever they are (their duck has no family's names; the import is not routed there)."""

    def __init__(self, state: Any, *, any_client: bool = False) -> None:
        self.state = state
        self.any_client = any_client

    @staticmethod
    def handles(path: str) -> bool:
        return path == ROUTE or path.startswith(ROUTE + "/")

    # ── what the server lives ──────────────────────────────────────────────

    def _host(self) -> Any:
        return getattr(self.state, "life_host", None)

    def _running(self) -> Path | None:
        host = self._host()
        return None if host is None else Path(host.record.folder)

    def _voices(self) -> Path:
        """Where this server keeps the voice libraries (its lineage voice's root)."""
        lineage = getattr(self._host(), "voice_lineage", None)
        root = getattr(lineage, "root", None)
        return Path(root) if root is not None else df.lineage_root()

    def _library(self) -> Path | bool:
        """The library the duck sounds like now on this server (its lineage voice's active one), else
        True: found from the lineage as the command line does."""
        lineage = getattr(self._host(), "voice_lineage", None)
        active = getattr(lineage, "active", None)
        folder = getattr(active, "folder", None)
        return Path(folder) if folder is not None else True

    # ── the routes ─────────────────────────────────────────────────────────

    def get(self, handler: BaseHTTPRequestHandler, path: str) -> None:
        if path == ROUTE:
            _send(handler, HTTPStatus.OK, {"ok": True, "life": self._host() is not None})
            return
        if path != ROUTE + "/export":
            _send(handler, HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
            return
        if not (self.any_client or _local(handler)):
            _send(handler, HTTPStatus.FORBIDDEN, _refusal("web.duckfile.not_local"))
            return
        host = self._host()
        if host is None:
            _send(handler, HTTPStatus.CONFLICT, _refusal("web.duckfile.no_life"))
            return
        try:
            # the mind now (when it is on screen; else the one saved when the page left it) and
            # Duck Core's save now, so the file is the duck as it is
            host.snapshot_now("export")
            with self.state.lock:
                self.state.save()
            data, manifest = df.pack(Path(host.record.folder), library=self._library())
        except df.DuckFileError as error:
            _send(handler, HTTPStatus.CONFLICT, {"ok": False, "error": error.args[0]})
            return
        name = df.file_name(manifest)
        handler.send_response(HTTPStatus.OK)
        handler.send_header("Content-Type", MEDIA_TYPE)
        handler.send_header("Content-Length", str(len(data)))
        handler.send_header(
            "Content-Disposition",
            f"attachment; filename=\"duck{df.SUFFIX}\"; filename*=UTF-8''{quote(name)}",
        )
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("X-Content-Type-Options", "nosniff")
        handler.end_headers()
        handler.wfile.write(data)

    def post(self, handler: BaseHTTPRequestHandler, path: str, query: str = "") -> None:
        if path not in (ROUTE + "/inspect", ROUTE + "/import"):
            _send(handler, HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
            return
        if not _local(handler):
            _send(handler, HTTPStatus.FORBIDDEN, _refusal("web.duckfile.not_local"))
            return
        if handler.headers.get(HEADER) != "1":
            _send(handler, HTTPStatus.FORBIDDEN, _refusal("web.duckfile.bad_request"))
            return
        raw = handler.headers.get("Content-Length", "")
        if not raw.strip().isdigit():
            _send(handler, HTTPStatus.LENGTH_REQUIRED, _refusal("web.duckfile.bad_request"))
            return
        length = int(raw)
        if length > df.MAX_FILE_BYTES:
            _send(handler, HTTPStatus.REQUEST_ENTITY_TOO_LARGE, _refusal("web.duckfile.too_big"))
            return
        try:
            body = handler.rfile.read(length)
        except OSError:
            return  # (the page went away mid-body)
        try:
            checked = df.read(body)
            if path == ROUTE + "/inspect":
                answer = {"ok": True, **df.summary(checked, running=self._running())}
            else:
                replace = parse_qs(query).get("replace") == ["1"]
                answer = df.install(
                    checked, replace=replace, running=self._running(), voices=self._voices()
                )
        except df.DuckFileError as error:
            _send(handler, HTTPStatus.OK, {"ok": False, "error": error.args[0]})
            return
        except Exception:
            traceback.print_exc()
            _send(handler, HTTPStatus.INTERNAL_SERVER_ERROR, _refusal("web.duckfile.failed"))
            return
        _send(handler, HTTPStatus.OK, answer)


def _local(handler: BaseHTTPRequestHandler) -> bool:
    """A client on this computer (the file holds people's names and looks)."""
    try:
        return ipaddress.ip_address(str(handler.client_address[0])).is_loopback
    except (ValueError, IndexError, TypeError):
        return False


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


__all__ = ["HEADER", "MEDIA_TYPE", "ROUTE", "DuckFileHub"]
