"""The simulator page loads nothing from another site (owner decision Q7, 2026-10-29).

Fonts and the JavaScript libraries are served by the simulator itself from static/fonts and
static/vendor; the page names only its own paths, and so do its stylesheets and its modules.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from importlib import resources
from pathlib import Path

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
    """What the page asks the browser to load: every link and script (``refs``), of those the
    stylesheets (``sheets``) and the module scripts (``modules``), and the import map."""

    def __init__(self) -> None:
        super().__init__()
        self.refs: list[str] = []
        self.sheets: list[str] = []
        self.modules: list[str] = []
        self.importmap = ""
        self._in_map = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag == "link" and a.get("href"):
            self.refs.append(a["href"] or "")
            if a.get("rel") == "stylesheet":
                self.sheets.append(a["href"] or "")
        if tag == "script":
            if a.get("src"):
                self.refs.append(a["src"] or "")
                if a.get("type") == "module":
                    self.modules.append(a["src"] or "")
            self._in_map = a.get("type") == "importmap"

    def handle_data(self, data: str) -> None:
        if self._in_map:
            self.importmap += data

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self._in_map = False


def _page() -> _Refs:
    page = _Refs()
    page.feed(STATIC.joinpath("index.html").read_text(encoding="utf-8"))
    return page


def _page_refs() -> list[str]:
    page = _page()
    imports = json.loads(page.importmap)["imports"]
    return [r for r in page.refs if not r.startswith("data:")] + list(imports.values())


def test_the_page_names_only_its_own_files() -> None:
    refs = _page_refs()
    assert "/fonts/fonts.css" in refs and "/vendor/d3/d3.min.js" in refs
    assert [r for r in refs if not r.startswith("/")] == []


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


def test_what_the_stylesheets_name_is_served_by_this_server_too(base: str) -> None:
    """A font or a picture a stylesheet asks for (``url(…)``, ``@import``) is fetched by the
    browser like the page's own files: each one is this server's and is there."""
    sheets = _page().sheets
    assert "/fonts/fonts.css" in sheets and len(sheets) > 3
    fetched = 0
    for sheet in sheets:
        status, kind, body = _get(base + sheet)
        assert status == 200 and kind.startswith("text/css"), sheet
        css = body.decode("utf-8")
        # (a quoted url is taken whole: an inline picture's own markup may hold a "url(" too)
        urls = re.findall(r"url\(\s*(?:\"([^\"]*)\"|'([^']*)'|([^\"')\s]+))\s*\)", css)
        named = {a or b or c for a, b, c in urls} | set(
            re.findall(r"@import\s+[\"']([^\"']+)", css)
        )
        for ref in sorted(ref for ref in named if not ref.startswith("data:")):
            url = urllib.parse.urljoin(base + sheet, ref)
            assert url.startswith(base + "/"), (sheet, ref)
            assert _get(url)[0] == 200, (sheet, ref)
            fetched += 1
    assert fetched > 5  # (the fonts)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_every_module_the_page_imports_comes_from_this_server(base: str) -> None:
    """The page's module scripts, followed import by import as a browser loads them (Node reads
    each module's imports from the source this server sends, with the page's import map): a
    module from another site, or one this server does not have, would leave the 3D view blank."""
    page = _page()
    assert page.modules, "the page has module scripts"
    done = subprocess.run(
        [
            str(shutil.which("node")),
            "--experimental-vm-modules",
            "--no-warnings",
            str(Path(__file__).with_name("module_graph_node_harness.mjs")),
            base,
            json.dumps({"entries": page.modules, "imports": json.loads(page.importmap)["imports"]}),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    graph = json.loads(done.stdout)
    assert graph["problems"] == []
    assert set(page.modules) <= set(graph["modules"])
    imported = {name for names in graph["modules"].values() for name in names}
    assert imported - set(page.modules), "the modules import others (three, the page's own)"
    assert imported <= set(graph["modules"])  # (each one was fetched and read in turn)


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
