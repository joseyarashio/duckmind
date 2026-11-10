"""The page shell (UI rebuild 2026-10-10, docs/design/UI-REBUILD-2026-10-10.md; item 6,
docs/design/neural-duck/UI-REDESIGN-2026-11-05.md): the view and its control panel are one element,
the panel's tabs are the players' (玩, 訓練, 成長, 我的小鴨), everything for development is on the
「實驗室」 view of the page (/lab). ``shell.js`` runs in Node over the page's own markup
(tests/page_tree_node_harness.py); the layout itself is checked in a browser (the screenshots in
the measurement records, not in the repo)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from page_tree_node_harness import NODE, run

STATIC = Path(__file__).resolve().parents[1] / "src" / "duckmind" / "simweb" / "static"
PAGE = (STATIC / "index.html").read_text(encoding="utf-8")
needs_node = pytest.mark.skipif(NODE is None, reason="needs Node")


@pytest.fixture(scope="module")
def shell() -> dict[str, Any]:
    if NODE is None:
        pytest.skip("needs Node")
    return run("shell", {"shell.js"})


@needs_node
def test_every_tab_shows_its_own_panel_and_only_that_one(shell: dict[str, Any]) -> None:
    first = shell["first"]
    assert len(first["selected"]) == 1 and first["shown"] == first["selected"]
    assert len(shell["tabs"]) >= 7  # the players' four and Duck Core's own three
    for tab, pressed in shell["tabs"].items():
        assert pressed["selected"] == [tab], tab
        assert pressed["shown"] == [tab], tab
        # (the panel its aria-controls names)
        assert pressed["panel_ids"] == [pressed["controls"]], tab
    assert shell["remembered"] == list(shell["tabs"])[-1]  # the viewer's last choice is kept


@needs_node
def test_the_lab_is_the_same_page_at_its_own_address_and_back_goes_back(
    shell: dict[str, Any],
) -> None:
    assert shell["first"]["path"] == "/" and not shell["first"]["lab"]
    assert shell["first"]["lab_hidden"] and not shell["first"]["sim_hidden"]
    lab = shell["lab"]
    assert lab["path"] == "/lab" and lab["lab"] and lab["sim_hidden"] and not lab["lab_hidden"]
    assert lab["current"] == ["data"]  # the header marks the view that is open
    back = shell["back"]
    assert not back["lab"] and back["lab_hidden"] and not back["sim_hidden"]
    assert back["current"] == ["sim"]
    assert shell["views"][:3] == ["sim", "data", "sim"]  # (the other scripts are told each time)
    # the lab's old address still opens it, and becomes /lab
    old = run("shell", {"shell.js"}, {"start": "/data"})["first"]
    assert old["path"] == "/lab" and old["lab"] and not old["lab_hidden"]


@needs_node
def test_a_link_into_a_part_of_the_other_view_opens_that_view_and_shows_the_part(
    shell: dict[str, Any],
) -> None:
    links = {link["from"]: link for link in shell["links"]}
    assert "btnBrain" in links and links["btnBrain"]["to"] == "neuroSide"  # 「腦」 on the HUD
    for link in shell["links"]:
        assert link["found"], link["to"]
        assert link["lab"] and link["path"] == f"/lab#{link['to']}", link["to"]
        assert link["scrolled"] == [link["to"]], link["to"]
    # every part the lab's links name is on the page; one that sits in the panel takes the
    # viewer back to the simulator, to its tab, with the fold-out it is in opened
    assert len(shell["parts"]) >= 8
    in_panel = [part for part in shell["parts"] if part["panel"]]
    for part in shell["parts"]:
        assert part["found"], part["to"]
        if not part["panel"]:
            assert part["in_lab"], part["to"]
            continue
        assert part["handled"] and not part["lab"] and part["path"] == "/", part["to"]
        assert part["selected"] == [part["panel"]] == part["shown"], part["to"]
        assert all(part["folds"]) and part["scrolled"] == [part["to"]], part["to"]
    assert any(part["folds"] for part in in_panel)  # (the emotions, in 玩's fold-out)


@needs_node
def test_the_enlarged_view_keeps_its_panel_and_the_label_rides_on_the_hud(
    shell: dict[str, Any],
) -> None:
    assert shell["big"] == "open"  # enlarged: the panel floats open over the view
    assert shell["small"] == "closed" and shell["handle"] == "false"
    assert shell["opened"] == ["open", "true"]  # the handle opens it again
    # the label that says which duck is drawn is not shown over the view: the HUD's tooltip has it
    assert shell["hud_title"] == "page.truth.TRUTH_TITLE\nits line"


def test_every_word_the_shell_and_the_lab_add_is_in_all_six_languages() -> None:
    keys = set(
        re.findall(
            r'data-i18n(?:-attr="[\w-]+:|="|-html=")(page\.(?:shell|nav|lab|play|growtab|body)\.[\w.]+)',
            PAGE,
        )
    )
    for script in ("shell.js", "growth.js", "bodylink.js"):
        keys |= set(
            re.findall(
                r'"(page\.(?:shell|lab|growtab|body)\.[\w.]+)"',
                (STATIC / script).read_text(encoding="utf-8"),
            )
        )
    assert len(keys) >= 40
    for lang in ("zh-Hant", "en", "ja", "es", "zh-Hans", "ko"):
        text = (STATIC / f"i18n.{lang}.page.js").read_text(encoding="utf-8")
        for key in keys:
            assert f'"{key}":' in text, (lang, key)
