"""我的小鴨 → 真的小鴨 (item 6, docs/design/neural-duck/UI-REDESIGN-2026-11-05.md): which body this duck's
brain drives, said plainly, from /api/body (``remote``: the link's own numbers when the body is on
another machine), and how to connect a real Microduck. The card's logic runs in Node; the server's
part is the one field the /api/body answer carries."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

STATIC = Path(__file__).resolve().parents[1] / "src" / "duckmind" / "simweb" / "static"
PAGE = (STATIC / "index.html").read_text(encoding="utf-8")
JS = (STATIC / "bodylink.js").read_text(encoding="utf-8")
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="needs Node")

LINK = {
    "url": "remote://duck.local:8797",
    "up": True,
    "rtt_ms_p50": 64.6,
    "rtt_ms_p95": 93.0,
    "last_heard_ms": 40,
    "connects": 2,
    "drops": 1,
    "max_gap_ms": 812.4,
    "deadman_ms": 1000,
    "error": None,
}
ANSWERS: dict[str, Any] = {
    "offline": None,
    "off": {"available": False, "enabled": False},
    "starting": {
        "enabled": True,
        "available": False,
        "starting": True,
        "error": None,
        "remote": None,
    },
    "error": {"enabled": True, "available": False, "starting": False, "error": "robotd exited"},
    "local": {"enabled": True, "available": True, "starting": False, "error": None, "remote": None},
    "remote_up": {"enabled": True, "available": True, "remote": LINK},
    "remote_new": {"enabled": True, "available": True, "remote": {**LINK, "rtt_ms_p50": None}},
    "remote_down": {"enabled": True, "available": True, "remote": {**LINK, "up": False}},
    # (a link that went down while the body was starting: the link's own state is what the card says)
    "remote_down_starting": {"enabled": True, "starting": True, "remote": {**LINK, "up": False}},
}


@pytest.fixture(scope="module")
def said() -> dict[str, Any]:
    if NODE is None:
        pytest.skip("needs Node")
    script = (
        f"const api = require({json.dumps(str(STATIC / 'bodylink.js'))});"
        f"const answers = {json.dumps(ANSWERS)};"
        "const out = {};"
        "for (const [name, answer] of Object.entries(answers)) out[name] = api.linkState(answer);"
        "console.log(JSON.stringify(out));"
    )
    run = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


@needs_node
def test_the_card_says_which_body_the_brain_drives(said: dict[str, Any]) -> None:
    assert said["offline"]["key"] == "page.body.state.offline"
    assert said["off"]["key"] == "page.body.state.off"
    assert said["starting"]["key"] == "page.body.state.starting"
    assert said["error"]["key"] == "page.body.state.error"
    assert said["local"]["key"] == "page.body.state.local" and said["local"]["stats"] is None
    up = said["remote_up"]
    assert up["state"] == "remote_up" and up["key"] == "page.body.state.remote_up"
    assert up["params"] == {"url": "remote://duck.local:8797", "ms": 65}
    assert said["remote_new"]["key"] == "page.body.state.remote_up_new"
    assert said["remote_down"]["state"] == "remote_down"
    assert said["remote_down_starting"]["state"] == "remote_down"


@needs_node
def test_a_remote_body_shows_the_links_own_numbers(said: dict[str, Any]) -> None:
    stats = said["remote_down"]["stats"]
    assert stats == {
        "key": "page.body.stats",
        "params": {"connects": 2, "drops": 1, "gap": 812, "deadman": 1000},
    }
    for name in ("offline", "off", "starting", "error", "local"):
        assert said[name]["stats"] is None, name


def test_the_card_is_in_my_duck_with_the_steps_and_its_words_in_six_languages() -> None:
    tab = PAGE[PAGE.index('id="ctl-duck"') : PAGE.index('id="ctl-status"')]
    for element in ("bodyLink", "bodyLinkState", "bodyLinkStats", "bodyLinkHow"):
        assert f'id="{element}"' in tab, element
    assert '<script src="/bodylink.js"></script>' in PAGE
    # the steps name the same commands docs/real-microduck.md does
    doc = (STATIC.parents[3] / "docs" / "real-microduck.md").read_text(encoding="utf-8")
    for command in re.findall(r"<code>(python -m duckmind\.remotebody \w+)", tab):
        assert command in doc, command
    keys = set(re.findall(r'"(page\.body\.[\w.]+)"', JS)) | set(
        re.findall(r'data-i18n="(page\.body\.[\w.]+)"', tab)
    )
    assert len(keys) >= 15
    from duckmind.i18n import CODES
    from duckmind.i18n.catalog import read_catalog

    for code in CODES:
        catalog = read_catalog(code, "page") or {}
        assert not keys - set(catalog), (code, sorted(keys - set(catalog)))
    assert "innerHTML" not in JS  # (the address is data: text only)


# ── the server: /api/body carries the link's numbers ─────────────────────────────────────────


class _Body:
    error = None
    host_info = {"ops": []}
    home = "default_home"
    policy = "sit"
    passive = False

    def __init__(self, remote: dict[str, Any] | None) -> None:
        self.remote = remote

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def frame(self) -> dict[str, Any]:
        return {"t": 1.0, "joints": {}, "trunk": [0.0, 0.0, 0.06], "quat": [1, 0, 0, 0]}

    def world(self) -> dict[str, Any]:
        return {"hand": {"held": False}, "people": [], "mocap": []}

    def status(self) -> dict[str, Any]:
        status = {"realtime": 1.0, "policy": "sit", "fallen": False, "monitor": {}, "held": False}
        if self.remote is not None:
            status["remote"] = self.remote
        return status


@pytest.mark.parametrize("remote", [None, LINK])
def test_the_body_answer_carries_the_remote_links_numbers(tmp_path: Path, remote: Any) -> None:
    from duckmind.livebody import LivePaths
    from duckmind.simweb.live import LiveBridge

    paths = LivePaths(tmp_path, tmp_path, tmp_path, tmp_path / "robotd", tmp_path / "ort")
    bridge = LiveBridge(paths, eye=False)  # never started: no processes
    bridge.body = _Body(remote)  # type: ignore[assignment]
    assert bridge.snapshot()["remote"] == remote
