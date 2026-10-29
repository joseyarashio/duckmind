"""The minimap on the simulator view (owner 2026-10-11): static checks on the markup, the scripts and the
style. How it looks and moves is checked in a browser (screenshots in the measurement records, not in the repo)."""

from __future__ import annotations

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "src" / "duckmind" / "simweb" / "static"
PAGE = (STATIC / "index.html").read_text(encoding="utf-8")
CSS = (STATIC / "style.css").read_text(encoding="utf-8")
JS = (STATIC / "minimap.js").read_text(encoding="utf-8")
PLAN = (STATIC / "plan2d.js").read_text(encoding="utf-8")
EDITOR = (STATIC / "layout_editor.js").read_text(encoding="utf-8")
ZH = (STATIC / "i18n.zh-Hant.page.js").read_text(encoding="utf-8")


def _minimap_css() -> str:
    return CSS[CSS.index("---- the minimap on the view") :]


def test_markup_keeps_its_hooks_and_starts_hidden() -> None:
    for hook in ('id="miniMap"', 'id="miniMapFold"', 'id="miniMapBody"', 'id="miniMapSvg"'):
        assert hook in PAGE, hook
    tag = re.search(r'<section class="minimap"[^>]*>', PAGE)
    assert tag and " hidden" in tag.group(0)
    # it lives in the stage area, so the 2D duck and the enlarged view have it too
    area = PAGE.index('class="stage-area"')
    assert area < PAGE.index('id="miniMap"') < PAGE.index('id="duck3d"')
    assert (
        'aria-expanded="true"'
        in PAGE[PAGE.index('id="miniMapFold"') : PAGE.index('id="miniMapBody"')]
    )


def test_scripts_load_in_order_and_the_editor_shares_the_drawing() -> None:
    plan = PAGE.index('src="/plan2d.js"')
    assert plan < PAGE.index('src="/layout_editor.js"') < PAGE.index('src="/minimap.js"')
    assert "window.DuckPlan.drawFixed" in EDITOR
    assert "boxPoints" in PLAN and "drawFixed" in PLAN
    # one drawing of the fixed home: the editor no longer has its own polygon code for it
    assert "le-k-thing" not in EDITOR


def test_minimap_uses_data_the_page_already_polls_and_never_the_brain() -> None:
    assert "duck:body" in JS and "duck:home" in JS
    assert "setInterval" not in JS
    # (MAP-EXPLORE 2026-10-12) the one exception: the duck's own map for its faint layer, read only (a GET, nothing sent)
    assert set(re.findall(r"/api/neuro[\w/]*", JS)) == {"/api/neuro/knows"}
    assert "method:" not in JS and "POST" not in JS
    assert "frame.truth" in JS  # the viewer's ground truth, for the page only
    # fold choice per viewer, inside try/catch
    assert "localStorage" in JS and JS.count("try {") >= 2


def test_words_come_from_the_catalogs_in_the_page_language() -> None:
    keys = {
        "page.minimap.title": "家的平面圖",
        "page.minimap.aria": "家的平面圖，橘色圓點是小鴨現在的位置",
        "page.minimap.collapse": "收起平面圖",
        "page.minimap.expand": "展開平面圖",
        "page.minimap.enlarge": "放大平面圖",
        "page.minimap.shrink": "縮小平面圖",
    }
    for key, words in keys.items():
        assert f'"{key}": "{words}"' in ZH, key
    for key in ("title", "aria", "collapse", "enlarge"):
        assert f"page.minimap.{key}" in PAGE, key


def test_style_uses_tokens_only() -> None:
    css = _minimap_css()
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", css), "colours come from the tokens"
    assert "border-radius" not in css
    assert not re.search(r"box-shadow:[^;]*\d+px\s+\d+px\s+[1-9]\d*px\s", css), "hard shadows only"
    assert "blur(" not in css


def test_the_map_sits_on_the_left_and_the_physics_label_is_not_drawn() -> None:
    css = _minimap_css()
    rule = re.search(r"\.minimap \{[^}]*\}", css).group(0)
    assert "left: 10px" in rule and "right:" not in rule
    assert "max-width: calc(50% - 14px)" in rule  # never past the middle of the view
    minimap_rules = re.findall(r"[^{}]*\.minimap[^{}]*\{[^}]*\}", css)
    assert minimap_rules and not any("right: calc(min(430px" in r for r in minimap_rules)  # nor beside the floating panel
    # the 物理真相 label stays in the DOM for the scripts but is not shown (owner 2026-10-11)
    assert 'id="stageMode"' in PAGE
    assert re.search(r"\.stage-mode[^{]*\{ display: none !important; \}", CSS)
    shell = (STATIC / "shell.js").read_text(encoding="utf-8")
    assert '$("#stageMode")' in shell and "hudBox.title" in shell
