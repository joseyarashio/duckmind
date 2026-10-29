"""學會聽話 on the simulator page (phase 2; docs/design/neural-duck/LEARN-TRICKS-2026-10-25.md).

The three feedback buttons in the command strip and the 訓練 tab's trick book (``static/tricks.js``), run in Node
over a fake page (a fake clock, a fake ``fetch``, a tiny fake DOM): what each answer of ``POST /api/neuro/feedback``
is said as, how a book the server really draws (``growth_view``) becomes rows, when the book is read, and that a
row's button goes through the command strip's own path. The words: every key in all six catalogs; the style:
tokens only. How it looks in a browser is checked by hand (screenshots in the measurement records, not in the repo).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from duckmind.i18n import CODES
from duckmind.life import growth_view as gv
from duckmind.neuro.command_words import TRICKS
from duckmind.neuro.tricks import TrickBook

STATIC = Path(__file__).resolve().parents[1] / "src" / "duckmind" / "simweb" / "static"
PAGE = (STATIC / "index.html").read_text(encoding="utf-8")
CSS = (STATIC / "style.css").read_text(encoding="utf-8")
JS = (STATIC / "tricks.js").read_text(encoding="utf-8")
NODE = shutil.which("node")
HARNESS = Path(__file__).with_name("tricks_page_node_harness.mjs")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


def _run(scenario: str, data: Any = None) -> Any:
    args = [str(NODE), str(HARNESS), str(STATIC), scenario]
    if data is not None:
        args.append(json.dumps(data, ensure_ascii=False))
    done = subprocess.run(args, capture_output=True, text=True, timeout=60, check=False)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def _catalog(lang: str) -> str:
    return (STATIC / f"i18n.{lang}.page.js").read_text(encoding="utf-8")


# ── words and style ───────────────────────────────────────────────────────────────────────────


def test_every_word_the_trick_page_names_is_in_all_six_languages() -> None:
    keys = set(re.findall(r'"(page\.trick\.[\w.]+)"', JS))
    keys |= set(re.findall(r'data-i18n="(page\.trick\.[\w.]+)"', PAGE))
    for pair in re.findall(r'data-i18n-attr="([^"]+)"', PAGE):
        keys |= {p.split(":", 1)[1] for p in pair.split(";") if p.split(":", 1)[1].startswith("page.trick.")}
    keys |= {f"page.cmd.verb.{trick}" for trick in TRICKS}  # (a row's name and the answers' trick)
    assert len(keys) >= 35, keys
    for lang in CODES:
        text = _catalog(lang)
        for key in keys:
            assert f'"{key}":' in text, (lang, key)


def test_the_trick_styles_use_only_the_design_tokens() -> None:
    block = CSS[CSS.index("/* 學會聽話") :]
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", block)
    assert not re.search(r"\b(rgba?|hsla?)\(", block)
    assert "border-radius" not in block and "blur(" not in block and "gradient" not in block
    for token in set(re.findall(r"var\((--[\w-]+)\)", block)):
        assert f"{token}:" in CSS, token


# ── tricks.js in node ─────────────────────────────────────────────────────────────────────────


@needs_node
def test_every_feedback_answer_is_said_in_plain_words() -> None:
    a = _run("answers")
    # [key, the button's word, the trick, its proficiency %, the line's tone]
    assert a["praised"] == ["page.trick.fb.praised", "page.trick.fb.praise", "page.cmd.verb.bow", 12, "up"]
    assert a["treat"] == ["page.trick.fb.praised", "page.trick.fb.treat", "page.cmd.verb.bow", 12, "up"]
    assert a["shaped"][0] == "page.trick.fb.shaped"
    assert a["capped"][0] == "page.trick.fb.capped" and a["capped"][3] == 12
    assert a["scolded"] == ["page.trick.fb.scolded", "page.trick.fb.scold", "page.cmd.verb.bow", 9, "down"]
    assert a["already"][0] == "page.trick.fb.already_taken" and a["already"][2] == "page.cmd.verb.bow"
    assert a["notJustDone"][0] == "page.trick.fb.not_just_done"
    assert a["noTrick"] == ["page.trick.fb.no_trick", "page.trick.fb.praise", None, None, "error"]
    assert a["notRunning"][0] == "page.trick.fb.not_running" and a["notRunning"][1] == "page.trick.fb.scold"
    for case in ("badKind", "nothing", "server503"):
        assert a[case][0] == "page.trick.fb.failed", case


@needs_node
def test_the_book_the_server_draws_becomes_one_row_per_trick() -> None:
    book = TrickBook()
    for _ in range(3):  # 0.06 + 0.04 + 0.03 > the day's 0.12: the third praise is capped
        book.praise("bow", 3)
    book.praise("spin", 3)
    core = SimpleNamespace(s=SimpleNamespace(gd=3), _stage_day=lambda stage: 10)  # (flip opens at the juvenile stage)
    rows = {r["id"]: r for r in _run("real", gv._tricks(core, SimpleNamespace(tricks=book)))}
    assert list(rows) == list(TRICKS)
    assert rows["bow"] == {"id": "bow", "pct": 12, "state": "half", "today": 3, "total": 3, "capped": True, "suggest": False, "open": True}
    assert rows["spin"]["pct"] == 6 and rows["spin"]["today"] == 1 and not rows["spin"]["capped"]
    assert rows["lie_down"]["state"] == "none" and rows["lie_down"]["suggest"]  # the least practised open one, first in order
    assert rows["flip"]["state"] == "locked" and not rows["flip"]["open"] and not rows["flip"]["suggest"]


@needs_node
def test_rows_take_only_known_tricks_and_sane_numbers() -> None:
    r = _run("rows")
    assert r["none"] is None
    assert r["odd"] == [{"id": "bow", "pct": 100, "state": "none", "today": 0, "total": 0, "capped": False, "suggest": False, "open": True}]
    # a praise's answer on its row until the book is read again; another trick's answer leaves the row alone
    assert r["patched"]["pct"] == 6 and r["patched"]["today"] == 1 and r["patched"]["state"] == "half"
    assert r["patchedScold"]["pct"] == 39 and r["patchedScold"]["today"] == 2  # a scold is no practice
    assert r["patchedOther"]["pct"] == 42


@needs_node
def test_from_opening_the_tab_to_a_praise_and_the_book_read_again() -> None:
    r = _run("flow")
    assert r["closedGets"] == 0 and r["closedAfterPoll"] == 0  # shut: the book is never read
    assert r["openGets"] == 1 and r["statusHidden"] is True
    drawn = {row["id"]: row for row in r["drawn"]}
    assert drawn["spin"]["bar"] == {
        "now": "42", "min": "0", "max": "100", "text": 'page.trick.pct{"pct":42}',
        "label": 'page.trick.bar.aria{"trick":"page.cmd.verb.spin"}', "width": "42%",
    }  # fmt: skip
    assert drawn["spin"]["chip"] == "page.trick.state.learning" and drawn["spin"]["today"] == 'page.trick.today{"today":2,"total":9}'
    assert drawn["lie_down"]["marker"] == "page.trick.suggest" and drawn["lie_down"]["suggest"]
    assert [row["marker"] for row in r["drawn"]].count("page.trick.suggest") == 1
    assert drawn["shake"]["capped"] and drawn["shake"]["chip"] == "page.trick.state.learned"
    assert drawn["flip"]["locked"] and drawn["flip"]["button"]["disabled"]
    assert r["commands"] == ["bow"]  # a row's button: the command path; the locked one sends nothing
    assert r["post"] == {"kind": "praise", "disabledWhileSent": [True] * 6}  # the strip's three and the card's three
    said = 'page.trick.fb.praised{"kind":"page.trick.fb.praise","trick":"page.cmd.verb.bow","pct":12}'
    assert r["out"] == {"text": said, "state": "up", "keyGone": True}
    assert r["note"] == {"text": said, "hidden": False}
    assert r["bowNow"]["bar"]["now"] == "12" and r["bowNow"]["today"] == 'page.trick.today{"today":2,"total":2}'
    assert r["buttonsAfter"] == [False] * 6
    # the server keeps its picture up to 8 s: read again once, after that, and the server's numbers win
    assert r["getsRightAfter"] == 1 and r["readsWaiting"] == 1 and r["getsAfterStale"] == 2
    assert r["bowRead"]["today"] == 'page.trick.today{"today":3,"total":7}'
    assert r["noTrick"]["text"].startswith("page.trick.fb.no_trick") and r["noTrick"]["state"] == "error"
    assert r["noTrick"]["post"] == "scold"
    assert r["readsWaitingAfterRefusal"] == 0  # nothing moved: no extra read
    assert r["pollsOpen"] == 2 and r["pollsShut"] == 0


@needs_node
def test_a_read_older_than_the_answer_never_takes_the_answer_back() -> None:
    r = _run("staleRead")
    assert r["patched"] == ["12", 'page.trick.today{"today":2,"total":2}']
    assert r["olderRead"] == r["patched"]  # the server's picture had one try fewer: older, so the row stays
    assert r["newerRead"] == ["12", 'page.trick.today{"today":2,"total":2}', True]  # caught up: the server's row
    assert r["spinPatched"] == ["46", 'page.trick.today{"today":3,"total":10}']
    assert r["spinHeld"] == r["spinPatched"]
    # a new book (another duck) is believed once the hold is over
    assert r["spinAfterHold"] == ["0", 'page.trick.today{"today":0,"total":0}', "none"]


@needs_node
def test_no_duck_no_book_and_no_server_are_each_said() -> None:
    r = _run("states")
    assert r["off"] == {"text": "page.trick.off", "hidden": False, "rows": 0}
    assert r["noBook"] == {"text": "page.trick.no_book", "rows": 0}
    assert r["offline"] == {"text": "page.trick.offline", "phase": "offline"}
