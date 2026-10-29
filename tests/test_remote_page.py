"""Remote mode on the simulator page (docs/design/neural-duck/REMOTE-GOTHERE-2026-10-23.md).

Static checks on the markup, the scripts, the catalogs and the style; and the page logic itself in Node over a
fake page (a fake clock, a fake WebSocket, a fake ``fetch``): ``drive.js`` (what keys, buttons and a stick mean as
a twist) and ``remote.js`` (the toggle, the socket and its POST fallback, every way out). How it looks and feels
in a browser, and the real latency, are checked live by hand.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from duckmind.i18n import CODES

STATIC = Path(__file__).resolve().parents[1] / "src" / "duckmind" / "simweb" / "static"
PAGE = (STATIC / "index.html").read_text(encoding="utf-8")
CSS = (STATIC / "style.css").read_text(encoding="utf-8")
REMOTE = (STATIC / "remote.js").read_text(encoding="utf-8")
NEURO = (STATIC / "neuro.js").read_text(encoding="utf-8")
NODE = shutil.which("node")
HARNESS = Path(__file__).with_name("remote_page_node_harness.mjs")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


def _run(scenario: str) -> Any:
    done = subprocess.run(
        [str(NODE), str(HARNESS), str(STATIC), scenario],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def _catalog(lang: str) -> str:
    return (STATIC / f"i18n.{lang}.page.js").read_text(encoding="utf-8")


# ── the markup, the scripts, the style ────────────────────────────────────────────────────────


def test_the_slot_holds_the_toggle_the_pad_and_the_status_line_with_their_hooks() -> None:
    slot = PAGE[PAGE.index('id="cmdRemoteSlot"') : PAGE.index('id="cmdKeys"')]
    for hook in ('id="remoteToggle"', 'id="remotePad"', 'id="remoteState"'):
        assert hook in slot, hook
    toggle = re.search(r'<button[^>]*id="remoteToggle"[^>]*>', slot)
    assert toggle and 'aria-pressed="false"' in toggle.group(0) and 'class="toggle"' in toggle.group(0)
    pad = re.search(r'<div[^>]*id="remotePad"[^>]*>', slot)
    assert pad and " hidden" in pad.group(0)  # (the pad shows only while remote is on)
    assert re.findall(r'data-remote-key="(\w+)"', slot) == ["left", "forward", "right", "stop"]
    # the strip stays a scenario container, and the remote buttons are not locked with the command keys
    strip = re.search(r'<form[^>]*id="cmdStrip"[^>]*>', PAGE)
    assert strip and "data-scenarios" in strip.group(0)
    assert ":not([data-remote])" in NEURO


def test_the_script_loads_after_the_view_and_the_telemetry_reaches_it() -> None:
    assert PAGE.index('src="/duck3d.js"') < PAGE.index('src="/remote.js"')
    assert 'type="module" src="/remote.js"' in PAGE
    assert '"duck:telemetry"' in NEURO and '"duck:telemetry"' in REMOTE
    for name in ("drive.js", "remote.js"):
        assert (STATIC / name).is_file()
    assert 'from "./drive.js"' in REMOTE


def test_the_page_talks_to_the_backends_the_notes_describe() -> None:
    assert "/api/neuro/remote/ws" in REMOTE
    assert '"/api/neuro/remote/twist"' in REMOTE and '"/api/neuro/remote"' in REMOTE
    # the wire: {"on"}, {"vx","vyaw","t"}; the key-event time is the epoch ms
    assert "epochOf(" in REMOTE and "message.t = " in REMOTE


def test_the_remote_styles_use_only_the_design_tokens() -> None:
    block = CSS[CSS.index("/* 遙控與點地板") :]
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", block)
    assert not re.search(r"\b(rgba?|hsla?)\(", block)
    assert "border-radius" not in block and "blur(" not in block and "gradient" not in block
    for token in set(re.findall(r"var\((--[\w-]+)\)", block)):
        assert f"{token}:" in CSS, token
    assert "width: 44px; height: 44px" in block  # the pad's buttons: a thumb's size


def test_every_word_remote_adds_is_in_all_six_languages() -> None:
    keys = set(re.findall(r'"(page\.remote\.[\w.]+)"', REMOTE))
    keys |= set(re.findall(r'data-i18n(?:-attr="[\w-]+:|="|-html=")(page\.remote\.[\w.]+)', PAGE))
    for pair in re.findall(r'data-i18n-attr="([^"]+)"', PAGE):
        keys |= {p.split(":", 1)[1] for p in pair.split(";") if p.split(":", 1)[1].startswith("page.remote.")}
    assert len(keys) >= 12, keys
    # the words of the notices the page gives for why remote ended by itself
    assert {"page.remote.notice.lease", "page.remote.notice.lost", "page.remote.notice.stopped", "page.remote.notice.failed"} <= keys
    for lang in CODES:
        text = _catalog(lang)
        for key in keys:
            assert f'"{key}":' in text, (lang, key)


def test_the_hint_is_one_line_in_the_longest_languages() -> None:
    # the strip's lines are one line at 390 px in 繁體中文; the catalogs' room says how long the others may be
    for key in ("page.remote.state.on_latency", "page.remote.state.off", "page.remote.toggle"):
        for lang in CODES:
            line = next(x for x in _catalog(lang).splitlines() if x.startswith(f'"{key}":'))
            assert len(line) < 220, (lang, key)


# ── drive.js, pointing.js, remote.js in node ──────────────────────────────────────────────────


@needs_node
def test_keys_a_stick_and_the_pad_mean_twists_in_the_walk_policys_ranges() -> None:
    d = _run("drive")
    assert d["fwd"] == {"vx": 0.45, "vyaw": 0}
    assert d["wasd"] == {"vx": 0.4, "vyaw": 0.8}  # forward + left: an arc, never faster than 0.40 m/s
    assert d["turnAlone"] == {"vx": 0.35, "vyaw": 0.8}  # the legs cannot turn on the spot: the slowest arc
    assert d["cancel"] == d["brake"] == d["none"] == {"vx": 0, "vyaw": 0}
    assert d["stickFull"]["vx"] == 0.55 and d["stickFull"]["vyaw"] == 0
    assert d["stickDead"] == d["stickBack"] == d["stickNaN"] == {"vx": 0, "vyaw": 0}
    assert d["stickRight"] == {"vx": 0.4, "vyaw": -1}  # right is a negative yaw rate
    assert d["pad"] == {"vx": 0.4, "vyaw": 0.8}
    assert d["merged"]["vx"] == 0.55
    assert d["typing"] == [True, False, True, False]
    assert d["epoch"] == [1500, 1600]  # the event's own clock when it agrees with the wall clock, else now
    assert d["range"] == {"vx": [0.35, 0.55, 0.4], "vyaw": 1}


@needs_node
def test_remote_mode_over_a_websocket_from_the_toggle_to_escape() -> None:
    r = _run("websocketFlow")
    assert r["start"] == {"mode": "idle", "pressed": "false", "padHidden": True, "state": "page.remote.state.off"}
    assert r["connecting"]["mode"] == "connecting" and r["connecting"]["url"] == "ws://duck.local:8790/api/neuro/remote/ws"
    assert r["afterOpen"] == [{"on": True}]
    assert r["on"] == {"mode": "on", "pressed": "true", "padHidden": False, "state": "page.remote.state.on"}
    assert r["keyPrevented"] is True
    held = r["held"]
    assert held[0] == {"vx": 0.45, "vyaw": 0, "t": 1000010}  # the first twist carries the key event's time
    assert all(m == {"vx": 0.45, "vyaw": 0} for m in held[1:-1]) and len(held) >= 4  # 20 Hz while held
    assert held[-1] == {"vx": 0, "vyaw": 0}  # let go: a zero at once
    assert r["afterRelease"] == [{"vx": 0, "vyaw": 0}]  # then one now and then, to keep the lease
    esc = r["escape"]
    assert esc["stopped"] is True and esc["mode"] == "idle" and esc["padHidden"] is True
    assert esc["sentOff"] == {"on": False} and esc["closed"] is True and esc["pressed"] == "false"
    assert esc["state"] == "page.remote.state.off"


@needs_node
def test_typing_is_never_a_drive_key_and_turns_follow_the_walk_policy() -> None:
    r = _run("keysAreNotTypingAndTurnsAreArcs")
    assert r["typedPrevented"] is False and r["typedSent"] == 0
    assert r["twists"] == [[0.35, 0.8], [0, 0], [0.35, 0.8], [0, 0], [0.45, 0], [0.4, -0.8], [0, 0]]


@needs_node
def test_the_on_screen_buttons_and_a_gamepad_stick_drive_and_stop() -> None:
    r = _run("padButtonsAndGamepad")
    assert r["held"] == "true" and r["released"] is None
    assert r["twists"][0] == [0.45, 0] and r["twists"][-1] == [0, 0]
    assert r["stick"][0] == [0.55, 0] and r["stick"][-1] == [0, 0]


@needs_node
def test_every_way_out_tells_the_server_and_closes_the_socket() -> None:
    r = _run("everyWayOutTellsTheServer")
    for how in ("hidden", "pagehide", "toggle"):
        assert r[how] == {"mode": "idle", "last": {"on": False}, "closed": True}, how
    # losing focus lets every key go (a key-up is never reported to a page that cannot see it) but is not leaving
    assert r["blur"]["mode"] == "on" and r["blur"]["last"] == {"vx": 0, "vyaw": 0}


@needs_node
def test_the_server_ending_remote_mode_is_said_for_a_few_seconds_and_a_stale_snapshot_is_not_believed() -> None:
    r = _run("serverEndsItAndLeaseNotice")
    assert r["early"] == "on"  # (a snapshot from before our own on)
    assert r["shown"] == 'page.remote.state.on_latency|{"ms":120}'  # the median key → legs delay from the telemetry
    assert r["mode"] == "idle" and r["notice"] == "page.remote.notice.lease"
    assert r["closed"] is True and r["toldServer"] == 0  # (the server ended it: nothing to tell it)
    assert r["afterNotice"] == "page.remote.state.off"
    drop = _run("socketDropEndsIt")
    assert drop == {"mode": "idle", "notice": "page.remote.notice.lost"}


@needs_node
def test_without_a_websocket_the_post_routes_do_the_same() -> None:
    r = _run("noSocketFallsBackToPost")
    for how in ("refuse", "throw", "silent"):
        got = r[how]
        assert got["mode"] == "on", how
        assert got["posts"] == [["/api/neuro/remote", {"on": True}]], how
        assert got["twistPosts"] == [[0.45, 0], [0, 0]], how
        assert got["offPost"] == {"url": "/api/neuro/remote", "body": {"on": False}} and got["after"] == "idle", how


@needs_node
def test_a_refusal_is_shown_in_the_servers_words_and_nothing_stays_on() -> None:
    r = _run("refusedIsAnAnswer")
    assert r["mode"] == "idle" and r["pressed"] == "false" and r["closed"] is True
    assert r["state"] == "field:身體還沒準備好，現在不能遙控"


@needs_node
def test_the_latency_shown_is_the_telemetrys_median() -> None:
    assert _run("latency") == [120, 81, None, None]
