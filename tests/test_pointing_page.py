"""Click on the floor on the simulator page (docs/design/neural-duck/REMOTE-GOTHERE-2026-10-23.md §2).

Static checks on the markup, the script, the catalogs and the style; and ``pointing.js`` in Node (click or drag, the
floor point in the world's frame, what a click hit, how an errand is followed from the click to its end). How the marker looks and the click feels in a browser is checked live.
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
DUCK3D = (STATIC / "duck3d.js").read_text(encoding="utf-8")
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


# ── the markup, the script, the style ─────────────────────────────────────────────────────────


def test_the_hint_line_is_under_the_last_command_and_the_view_uses_pointing_js() -> None:
    assert re.search(r'<p[^>]*id="cmdPoint"[^>]*data-i18n="page\.point\.hint_beta"', PAGE)
    assert PAGE.index('id="cmdOut"') < PAGE.index('id="cmdPoint"') < PAGE.index("</form>", PAGE.index('id="cmdPoint"'))
    assert (STATIC / "pointing.js").is_file() and 'from "./pointing.js"' in DUCK3D
    assert '"duck:telemetry"' in DUCK3D and '"duck:telemetry"' in NEURO
    assert '"/api/neuro/gothere"' in DUCK3D
    # the click is ignored while remote mode has the legs
    assert "remoteHasLegs()" in DUCK3D and "window.duckRemote" in DUCK3D
    # the floor of the fallback room and of the home are tagged for the pick
    assert 'floor.userData.kind = "floor"' in DUCK3D and "home.floors" in DUCK3D


def test_the_pointing_styles_use_only_the_design_tokens() -> None:
    block = CSS[CSS.index(".cmd-point {") :]
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", block)
    assert not re.search(r"\b(rgba?|hsla?)\(", block)
    assert "border-radius" not in block and "blur(" not in block and "gradient" not in block
    for token in set(re.findall(r"var\((--[\w-]+)\)", block)):
        assert f"{token}:" in CSS, token
    # the marker on the floor takes its colours from the tokens too
    assert 'token("--orange"' in DUCK3D and 'token("--magenta"' in DUCK3D


def test_every_word_pointing_adds_is_in_all_six_languages() -> None:
    keys = set(re.findall(r'"(page\.point\.[\w.]+)"', DUCK3D))
    keys |= set(re.findall(r'data-i18n="(page\.point\.[\w.]+)"', PAGE))
    assert len(keys) >= 15, keys
    for lang in CODES:
        text = _catalog(lang)
        for key in keys:
            assert f'"{key}":' in text, (lang, key)
    # every reason the duck can give for not getting there has its words
    # (`replaced` is never said: a newer errand under another id takes the line, pointing.js errandNews)
    from duckmind.neuro.errand import REASONS

    for reason in (r for r in REASONS if r != "replaced"):
        assert f'"page.point.why.{reason}"' in DUCK3D, reason
        for lang in CODES:
            assert f'"page.point.why.{reason}":' in _catalog(lang), (lang, reason)
    # the line that says the server could not be reached is the page's existing one
    assert '"page.neuro.why"' in DUCK3D


def test_the_hint_is_short_enough_in_every_language() -> None:
    for key in ("page.point.hint_beta", "page.point.sent", "page.point.walking", "page.point.notfloor"):
        for lang in CODES:
            line = next(x for x in _catalog(lang).splitlines() if x.startswith(f'"{key}":'))
            assert len(line) < 200, (lang, key)


# ── pointing.js in node ───────────────────────────────────────────────────────────────────────


@needs_node
def test_a_click_is_told_from_a_drag_and_the_floor_point_is_in_the_worlds_frame() -> None:
    p = _run("pointing")
    # the ray from scene (0.5, 1, 1) toward scene (0.5, 0, −1): the floor point is world (0.5, +1) (scene z = −world y)
    assert p["floor"] == {"x_m": 0.5, "y_m": 1}
    assert p["sky"] == [None] * 5
    assert p["sceneOfWorld"] == [[1.5, -2], [0, 3]]
    assert p["clicks"] == [{"x": 102, "y": 101}, None, None, None, None, None, {"x": 28, "y": 20}]
    pick = p["pick"]
    assert pick["open"]["kind"] == "floor" and pick["open"]["point"] == {"x_m": 0, "y_m": 0.333}
    assert pick["noHomeYet"]["kind"] == "floor"  # (before the home is drawn the floor plane is the floor)
    assert pick["outsideTheHome"] == {"kind": "sky"}
    assert pick["blocked"] == {"kind": "blocked"} and pick["duckBehind"] == {"kind": "blocked"}
    assert pick["duck"] == {"kind": "duck"}
    assert pick["rug"]["kind"] == "floor" and pick["step"]["kind"] == "floor"


@needs_node
def test_the_errand_is_followed_from_the_click_to_its_end_and_a_refused_click_leaves_it_alone() -> None:
    p = _run("pointing")
    assert p["news"] == [
        {"kind": "none"},  # (nothing yet)
        {"kind": "none"},  # (the telemetry still shows an older errand)
        {"kind": "walking"},
        {"kind": "gave_up", "reason": "stuck"},
        {"kind": "replaced"},  # (another click or page took its place)
        {"kind": "none"},
    ]
    # [time, marker shown, line kind]
    assert p["track"] == [
        [0, False, None],
        [1001, True, "sent"],
        [1501, True, "sent"],
        [2001, True, "walking"],
        [3001, True, "refused"],  # a refused second click: said for a while, the marker stays
        [11001, True, "walking"],
        [12001, False, "gave_up"],  # the marker goes when the errand ends
        [20001, False, None],  # and the line is the hint again
    ]
    assert p["marker30s"] == [True, False, None]  # 30 s at most
    assert p["replaced"] == [False, None]
