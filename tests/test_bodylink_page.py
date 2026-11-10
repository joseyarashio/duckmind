"""我的小鴨 → 真的小鴨 (item 6, docs/design/neural-duck/UI-REDESIGN-2026-11-05.md): which body this duck's
brain drives, said plainly, from /api/body (``remote``: the link's own numbers when the body is on
another machine), and how to connect a real Microduck. The card's logic runs in Node, and the card
itself on the page's own markup (tests/page_tree_node_harness.py); the server's part is the one
field the /api/body answer carries."""

from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from page_tree_node_harness import element_texts, run

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


def test_the_cards_words_are_in_six_languages() -> None:
    tab = PAGE[PAGE.index('id="ctl-duck"') : PAGE.index('id="ctl-status"')]
    keys = set(re.findall(r'"(page\.body\.[\w.]+)"', JS)) | set(
        re.findall(r'data-i18n="(page\.body\.[\w.]+)"', tab)
    )
    assert len(keys) >= 15
    from duckmind.i18n import CODES
    from duckmind.i18n.catalog import read_catalog

    for code in CODES:
        catalog = read_catalog(code, "page") or {}
        assert not keys - set(catalog), (code, sorted(keys - set(catalog)))


def test_the_steps_name_commands_the_tool_really_takes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """「怎麼接真的小鴨」: the card's commands, run as they are written (in a home folder of this
    test's own; nothing answers at the steps' example address, and nothing is dialled)."""
    from duckmind.remotebody import __main__ as cli
    from duckmind.remotebody import client
    from duckmind.remotebody.bridge import bridge_parser
    from duckmind.remotebody.link import read_token

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("DUCKMIND_HOME", raising=False)
    dialled: list[tuple[str, bytes]] = []

    class NoDuck:
        def __init__(self, url: str, token: bytes) -> None:
            dialled.append((url, token))

        def start(self, timeout_s: float) -> None:
            raise OSError("nothing answers")

        def close(self) -> None:
            return None

    monkeypatch.setattr(client, "RemoteLink", NoDuck)
    lines = [shlex.split(line) for line in element_texts("code", inside="bodyLinkHow")]
    commands = [line[3:] for line in lines if line[:3] == ["python", "-m", "duckmind.remotebody"]]
    assert [command[0] for command in commands] == ["token", "bridge", "check"]
    token, bridge, check = commands
    assert cli.main(token) == 0  # a fresh token, where the step says
    written = home / ".duckmind" / "remote-body.token"
    assert written.is_file()
    args = bridge_parser().parse_args(bridge[1:])  # (the duck's side: its options exist)
    assert args.backend == "real" and args.token_file.expanduser() == written
    assert cli.main(check) == 1  # (no duck at the example address: it says so and stops)
    assert dialled == [(check[1], read_token(written))]  # with the token the first step made


@needs_node
def test_the_card_in_my_duck_says_the_bodys_link_while_its_tab_is_open(tmp_path: Path) -> None:
    link = {**LINK, "url": "remote://<b>duck</b>.local:8797"}  # (an address is data, not markup)
    answer = {"enabled": True, **_bridge(tmp_path, link).snapshot()}
    card = run("bodylink", {"shell.js", "bodylink.js"}, {"body": answer})
    asked = card["asked"]
    assert asked["elsewhere"] == 0  # another tab is open: the body is not asked about
    assert asked["opened"] == 1  # 我的小鴨 opens: at once
    assert 3 <= asked["half_minute_open"] - asked["opened"] <= 8  # then every few seconds
    assert asked["left"] == asked["half_minute_open"] and card["sent"] == 0  # (it only reads)
    said = json.dumps({"url": link["url"], "ms": 65}, separators=(",", ":"))
    assert card["line"] == {
        "state": "remote_up",
        "text": f"page.body.state.remote_up|{said}",
        "in_tab": True,
    }
    numbers = json.dumps({"connects": 2, "drops": 1, "gap": 812, "deadman": 1000})
    assert card["stats"] == {
        "hidden": False,
        "text": "page.body.stats|" + numbers.replace(" ", ""),
    }
    assert card["markup"] == 0


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


def _bridge(tmp_path: Path, remote: dict[str, Any] | None) -> Any:
    from duckmind.livebody import LivePaths
    from duckmind.simweb.live import LiveBridge

    paths = LivePaths(tmp_path, tmp_path, tmp_path, tmp_path / "robotd", tmp_path / "ort")
    bridge = LiveBridge(paths, eye=False)  # never started: no processes
    bridge.body = _Body(remote)  # type: ignore[assignment]
    return bridge


@pytest.mark.parametrize("remote", [None, LINK])
def test_the_body_answer_carries_the_remote_links_numbers(tmp_path: Path, remote: Any) -> None:
    assert _bridge(tmp_path, remote).snapshot()["remote"] == remote
