"""The page shell (UI rebuild 2026-10-10, docs/design/UI-REBUILD-2026-10-10.md; item 6,
docs/design/neural-duck/UI-REDESIGN-2026-11-05.md): the view and its control panel are one element,
the panel's tabs are the players' (玩, 訓練, 成長, 我的小鴨), everything for development is on the
「實驗室」 view of the page (/lab). Static checks on the markup and the scripts; the layout itself is
checked in a browser (the screenshots in the measurement records, not in the repo)."""

from __future__ import annotations

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "src" / "duckmind" / "simweb" / "static"
PAGE = (STATIC / "index.html").read_text(encoding="utf-8")
CSS = (STATIC / "style.css").read_text(encoding="utf-8")
SIM = PAGE[PAGE.index('id="viewSim"') : PAGE.index('id="viewData"')]
LAB = PAGE[PAGE.index('id="viewData"') : PAGE.index("</main>")]


def _between(start: str, end: str) -> str:
    at = PAGE.index(start)
    return PAGE[at : PAGE.index(end, at)]


def test_every_tab_has_its_panel_and_the_controls_sit_in_the_right_one() -> None:
    tabs = re.findall(r'data-ctl-tab="(\w+)"', PAGE)
    panels = re.findall(r'data-ctl-panel="(\w+)"', PAGE)
    assert tabs == panels == ["stim", "trick", "grow", "duck", "status", "play", "tools"]
    where = {
        # 玩 (owner 2026-11-05): the play card, the talk card (shown with --live-voice), the fourteen
        # emotions in a fold-out, the sound scenes
        "stim": [
            'id="neuroStim"',
            'id="neuroStimButtons"',
            'id="liveVoice"',
            'id="emotionFold"',
            'id="neuroEmotions"',
            'id="emotionGroups"',
            'id="neuroSoundMount"',
        ],
        "trick": ['id="trickBook"', 'id="trickList"', 'id="trickFeedback"'],
        "grow": ['id="growthTab"', 'id="growthTabStatus"', 'id="growthTabBody"'],
        "duck": ['id="duckFile"', 'id="duckFileSwitch"', 'id="bodyLink"', 'id="neuroPicker"'],
        "status": ['id="duckBanner"'],
        "play": ['id="scenarios"', 'id="cmdForm"'],
        "tools": ['id="labTools"'],
    }
    fold = _between('id="emotionFold"', "</details>")
    assert 'id="neuroEmotions"' in fold and "<summary" in fold
    for panel, hooks in where.items():
        start = PAGE.index(f'id="ctl-{panel}"')
        nxt = [m.start() for m in re.finditer(r'class="ctl-panel"', PAGE) if m.start() > start]
        body = PAGE[start : nxt[0] if nxt else PAGE.index("</aside>", start)]
        for hook in hooks:
            assert hook in body, (panel, hook)


def test_the_players_tabs_are_big_and_say_what_they_are_with_an_icon() -> None:
    for tab, key in (
        ("stim", "page.shell.tab.play"),
        ("trick", "page.trick.tab"),
        ("grow", "page.shell.tab.grow"),
        ("duck", "page.shell.tab.myduck"),
    ):
        tag = _between(f'id="ctlTab-{tab}"', "</button>")
        assert 'class="ctl-tab-icon" aria-hidden="true"' in tag and f'data-i18n="{key}"' in tag
    assert ".ctl-tab:has(.ctl-tab-icon)" in CSS and "min-height: 56px" in CSS
    # 玩: the five things a player does first — touch, hold, ball, a new thing, clap — as big buttons
    play = _between('id="neuroStimButtons"', "</div>")
    stims = re.findall(r'data-stim="(\w+)"', play)
    assert stims[:6] == ["pat_head", "pick_up", "set_down", "roll_ball", "novel_object", "clap"]
    assert ".neuro-stim .row.play-buttons .btn { min-height: 56px" in CSS


def test_the_view_and_the_panel_are_one_element_so_enlarging_keeps_the_panel() -> None:
    view = _between('id="duckView"', "</aside>")
    assert 'id="ctlPanel"' in view and 'class="stage-frame"' in view
    app = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "requestFullscreen" in app and 'duckView.classList.toggle("is-big", on)' in app
    assert ".duck-view.is-big" in CSS and "max-height: min(54%, 520px)" in CSS


def test_the_stage_and_the_command_strip_keep_the_players_controls() -> None:
    stage = _between('class="stage-frame"', 'id="duck3dNote"')
    for hook in (
        'id="duck3d"',
        'id="eyeWindow"',
        'id="cmdStrip"',
        'id="cmdText"',
        'id="cmdMic"',
        'id="cmdKeys"',
        'id="cmdFeedback"',
        'id="cmdPoint"',
        'id="remoteToggle"',
    ):
        assert hook in stage, hook
    # (owner 2026-11-05) the 玩 tab is the one place for the stimuli: no quick strip under the stage
    assert 'id="neuroQuick"' not in PAGE and "data-stim=" not in stage


def test_the_lab_holds_everything_for_development_and_the_players_view_does_not() -> None:
    for hook in (
        'id="neuroSide"',
        'id="neuroBrain"',
        'id="neuroReflexes"',
        'id="neuroRhythm"',
        'id="physCard"',
        'id="duck3dReadout"',
        'id="liveToggle"',
        'id="worldBattery"',
        'id="neuroMotions"',
        'id="motionGroups"',
        'id="hudBody"',
        'id="hudReflex"',
        'id="growthView"',
        'id="gvFf"',
        'id="mind"',
        'id="homeEditor"',
        'id="capabilities"',
        'id="neuroSkills"',
    ):
        assert hook in LAB and hook not in SIM, hook
    # every part of the lab is one click from its top
    jump = _between('class="lab-jump"', "</nav>")
    for target in re.findall(r'href="#(\w+)"', jump):
        # (the emotions' one copy is in 玩: the lab's button goes there, the shell opens its fold-out)
        assert f'id="{target}"' in (SIM if target == "neuroEmotions" else LAB), target
    assert "fold.open = true" in (STATIC / "shell.js").read_text(encoding="utf-8")
    assert 'href="/lab"' in PAGE and 'data-view-link="data"' in PAGE
    shell = (STATIC / "shell.js").read_text(encoding="utf-8")
    assert '"/lab"' in shell and "pushState" in shell and "popstate" in shell
    caps = (STATIC / "capabilities.js").read_text(encoding="utf-8")
    assert '"duck:view"' in caps  # (the evidence and its clips load when the lab opens)


def test_the_hud_opens_the_brain_in_the_lab() -> None:
    hud = _between('id="hud"', 'class="stage-tools"')
    assert (
        'id="btnBrain" href="/lab#neuroSide" data-view-link="data"' in hud and 'id="hudNow"' in hud
    )
    # (owner 2026-11-05) only 「現在在做」 on the stage; the body and reflex lines are in the lab's brain
    assert 'id="hudBody"' not in hud and 'id="hudReflex"' not in hud
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
