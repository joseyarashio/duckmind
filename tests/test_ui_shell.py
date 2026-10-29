"""The page shell (UI rebuild 2026-10-10, docs/design/UI-REBUILD-2026-10-10.md): the view and its
control panel are one element, the panel's tabs hold the controls, 「資料」 is a view of the page.
Static checks on the markup and the scripts; the layout itself is checked in a browser (the
screenshots in the measurement records, not in the repo)."""

from __future__ import annotations

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "src" / "duckmind" / "simweb" / "static"
PAGE = (STATIC / "index.html").read_text(encoding="utf-8")
CSS = (STATIC / "style.css").read_text(encoding="utf-8")


def _between(start: str, end: str) -> str:
    at = PAGE.index(start)
    return PAGE[at : PAGE.index(end, at)]


def test_every_tab_has_its_panel_and_the_controls_sit_in_the_right_one() -> None:
    tabs = re.findall(r'data-ctl-tab="(\w+)"', PAGE)
    panels = re.findall(r'data-ctl-panel="(\w+)"', PAGE)
    assert (
        tabs
        == panels
        == [
            "stim",
            "motion",
            "emotion",
            "trick",
            "voice",
            "duck",
            "brain",
            "status",
            "play",
            "tools",
            "phys",
        ]
    )
    where = {
        "stim": ['id="neuroStim"', 'id="neuroSoundMount"'],
        "motion": ['id="neuroMotions"', 'id="motionGroups"'],
        "emotion": ['id="neuroEmotions"', 'id="emotionGroups"'],
        "trick": ['id="trickBook"', 'id="trickList"', 'id="trickFeedback"'],
        "voice": ['id="liveVoice"'],
        "duck": ['id="neuroPicker"'],
        "brain": ['id="neuroBrain"', 'id="neuroReflexes"', 'id="neuroRhythm"'],
        "phys": ['id="duck3dReadout"', 'id="liveToggle"', 'id="worldBattery"'],
        "status": ['id="duckBanner"'],
        "play": ['id="scenarios"', 'id="cmdForm"'],
        "tools": ['id="labTools"'],
    }
    for panel, hooks in where.items():
        start = PAGE.index(f'id="ctl-{panel}"')
        nxt = [m.start() for m in re.finditer(r'class="ctl-panel"', PAGE) if m.start() > start]
        body = PAGE[start : nxt[0] if nxt else PAGE.index("</aside>", start)]
        for hook in hooks:
            assert hook in body, (panel, hook)


def test_the_view_and_the_panel_are_one_element_so_enlarging_keeps_the_panel() -> None:
    view = _between('id="duckView"', "</aside>")
    assert 'id="ctlPanel"' in view and 'class="stage-frame"' in view
    app = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "requestFullscreen" in app and 'duckView.classList.toggle("is-big", on)' in app
    assert ".duck-view.is-big" in CSS and "max-height: min(54%, 520px)" in CSS


def test_the_data_page_holds_the_evidence_and_the_skills_and_the_view_does_not() -> None:
    data = PAGE[PAGE.index('id="viewData"') : PAGE.index("</main>")]
    sim = PAGE[PAGE.index('id="viewSim"') : PAGE.index('id="viewData"')]
    for hook in ('id="capabilities"', 'id="neuroSkills"', 'id="capSkills"', 'id="neuroSkillCards"'):
        assert hook in data and hook not in sim, hook
    assert 'href="/data"' in PAGE and 'data-view-link="data"' in PAGE
    shell = (STATIC / "shell.js").read_text(encoding="utf-8")
    assert '"/data"' in shell and "pushState" in shell and "popstate" in shell
    caps = (STATIC / "capabilities.js").read_text(encoding="utf-8")
    assert '"duck:view"' in caps  # (the evidence and its clips load when 「資料」 opens)


def test_the_brain_is_not_on_the_main_view_by_default_and_the_hud_opens_it() -> None:
    assert 'data-ctl-open="brain"' in PAGE and 'id="hudNow"' in PAGE
    brain = _between('id="ctl-brain"', 'id="ctl-phys"')
    assert "hidden" in brain.split(">", 1)[0]  # (a tab panel: shut until it is chosen)
    js = (STATIC / "neuro.js").read_text(encoding="utf-8")
    assert "function renderHud" in js and "renderHud(t, newest)" in js
    assert 'for (const card of [$("#neuroMotions"), $("#neuroEmotions")])' in js


def test_the_shell_styles_use_only_the_design_tokens() -> None:
    block = CSS[CSS.index("/* ---------- app shell (UI rebuild 2026-10-10") :]
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", block)
    assert not re.search(r"\b(rgba?|hsla?)\(", block)
    assert (
        "border-radius" not in block
        and "blur(" not in block
        and "gradient" not in block.replace("var(--dots-ink)", "")
    )
    for token in set(re.findall(r"var\((--[\w-]+)\)", block)):
        assert f"{token}:" in CSS, token


def test_every_word_the_shell_adds_is_in_all_six_languages() -> None:
    keys = set(
        re.findall(
            r'data-i18n(?:-attr="[\w-]+:|="|-html=")(page\.(?:shell|nav|data)\.[\w.]+)', PAGE
        )
    )
    keys |= set(
        re.findall(r'"(page\.shell\.[\w.]+)"', (STATIC / "shell.js").read_text(encoding="utf-8"))
    )
    assert len(keys) >= 20
    for lang in ("zh-Hant", "en", "ja", "es", "zh-Hans", "ko"):
        text = (STATIC / f"i18n.{lang}.page.js").read_text(encoding="utf-8")
        for key in keys:
            assert f'"{key}":' in text, (lang, key)
