"""The simulator page loads nothing from another site (owner decision Q7, 2026-10-29).

Fonts and the JavaScript libraries are served by the simulator itself from static/fonts and
static/vendor; the page names only its own paths.
"""

from __future__ import annotations

import json
import re
import threading
import urllib.error
import urllib.request
from html.parser import HTMLParser
from importlib import resources

import pytest

from duckmind.duckcore.api import Config
from duckmind.simweb.server import SimServer, SimState, make_handler

STATIC = resources.files("duckmind.simweb").joinpath("static")


@pytest.fixture()
def base():
    httpd = SimServer(("127.0.0.1", 0), make_handler(SimState(Config(seed=7, name="測試"))))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _get(url: str) -> tuple[int, str, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read()
    except urllib.error.HTTPError as error:
        return error.code, "", b""


class _Refs(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.refs: list[str] = []
        self.importmap = ""
        self._in_map = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag == "link" and a.get("href"):
            self.refs.append(a["href"] or "")
        if tag == "script":
            if a.get("src"):
                self.refs.append(a["src"] or "")
            self._in_map = a.get("type") == "importmap"

    def handle_data(self, data: str) -> None:
        if self._in_map:
            self.importmap += data

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self._in_map = False


def _page_refs() -> list[str]:
    page = _Refs()
    page.feed(STATIC.joinpath("index.html").read_text(encoding="utf-8"))
    imports = json.loads(page.importmap)["imports"]
    return [r for r in page.refs if not r.startswith("data:")] + list(imports.values())


def test_the_page_names_only_its_own_files() -> None:
    refs = _page_refs()
    assert "/fonts/fonts.css" in refs and "/vendor/d3/d3.min.js" in refs
    assert [r for r in refs if not r.startswith("/")] == []
    scripts = "".join(
        p.read_text(encoding="utf-8")
        for p in STATIC.iterdir()
        if p.name.endswith((".js", ".css", ".html")) and not re.match(r"i18n\..+\..+\.js$", p.name)
    )
    assert re.findall(r"https?://(?:fonts\.g|cdn\.|cdnjs|unpkg)[^\s\"')]*", scripts) == []


def test_the_server_serves_the_fonts_and_libraries(base: str) -> None:
    for ref in _page_refs():
        if ref.endswith("/"):
            ref += "controls/OrbitControls.js"
        status, kind, body = _get(base + ref)
        assert status == 200 and body, ref
    status, kind, body = _get(base + "/vendor/three/addons/loaders/STLLoader.js")
    assert status == 200 and kind.startswith("text/javascript")
    css = _get(base + "/fonts/fonts.css")[2].decode("utf-8")
    first = re.search(r"url\(\./([^)]+)\)", css)
    assert first
    status, kind, body = _get(base + "/fonts/" + first.group(1))
    assert (status, kind) == (200, "font/woff2") and body[:4] == b"wOF2"
    status, kind, _ = _get(base + "/fonts/noto-sans-jp.css")
    assert status == 200 and kind.startswith("text/css")


@pytest.mark.parametrize(
    "path",
    [
        "/fonts/../server.py",
        "/vendor/./d3/d3.min.js",
        "/fonts/.hidden.css",
        "/i18n/context.json",
        "/elsewhere/x.js",
        "/vendor//d3/d3.min.js",
        "/vendor/three/LICENSE",
    ],
)
def test_the_server_serves_nothing_else(base: str, path: str) -> None:
    assert _get(base + path)[0] == 404
