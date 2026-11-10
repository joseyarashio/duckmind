"""Click on the floor on the simulator page (docs/design/neural-duck/REMOTE-GOTHERE-2026-10-23.md §2).

The catalogs' words; and ``pointing.js`` in Node (click or drag, the floor point in the world's frame, what a click
hit, how an errand is followed from the click to its end). How the marker looks and the click feels in a browser is checked live.
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
DUCK3D = (STATIC / "duck3d.js").read_text(encoding="utf-8")
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


# ── the catalogs ──────────────────────────────────────────────────────────────────────────────


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


def test_the_hint_is_short_enough_in_every_language() -> None:
    for key in (
        "page.point.hint",
        "page.point.sent",
        "page.point.walking",
        "page.point.notfloor",
    ):
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
    assert (
        pick["noHomeYet"]["kind"] == "floor"
    )  # (before the home is drawn the floor plane is the floor)
    assert pick["outsideTheHome"] == {"kind": "sky"}
    assert pick["blocked"] == {"kind": "blocked"} and pick["duckBehind"] == {"kind": "blocked"}
    assert pick["duck"] == {"kind": "duck"}
    assert pick["rug"]["kind"] == "floor" and pick["step"]["kind"] == "floor"


@needs_node
def test_the_errand_is_followed_from_the_click_to_its_end_and_a_refused_click_leaves_it_alone() -> (
    None
):
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
