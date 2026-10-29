"""Growth G4: the 「成長」 section of the 資料 page (docs/design/GROWTH-G4-2026-10-11.md).

Static checks on the markup, the scripts, the catalogs and the style; and the component itself in
Node over a tiny fake page, drawing the views the server really makes. How it looks and behaves in a
browser is checked by hand (screenshots in the measurement records, not in the repo).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import duckmind.world.legacy  # noqa: F401  (registers LegacyWorld)
from duckmind.duckcore import bridge, bridge_v2
from duckmind.duckcore.api import Config, Session, dispatch
from duckmind.i18n import CODES
from duckmind.life import growth_view as gv
from duckmind.simweb import i18nwire

STATIC = Path(__file__).resolve().parents[1] / "src" / "duckmind" / "simweb" / "static"
PAGE = (STATIC / "index.html").read_text(encoding="utf-8")
CSS = (STATIC / "growth.css").read_text(encoding="utf-8")
JS = (STATIC / "growth.js").read_text(encoding="utf-8")
NODE = shutil.which("node")
HARNESS = Path(__file__).with_name("growth_node_harness.js")
TPE = timezone(timedelta(hours=8))

IDS = (
    "growthView gvStatus gvLive gvFf gvFfDays gvFfStyleA gvFfStyleB gvFfSeed gvFfStart gvFfCancel "
    "gvFfProgress gvFfNote gvFfProblem gvResults"
).split()


def _catalog(lang: str) -> str:
    return (STATIC / f"i18n.{lang}.page.js").read_text(encoding="utf-8")


def _keys(text: str) -> set[str]:
    return set(re.findall(r"page\.growth\.[a-z0-9_.]+[a-z0-9_]", text))


# ── the markup, the scripts, the style ────────────────────────────────────────────────────────


def test_the_section_is_on_the_data_page_with_its_hooks() -> None:
    data = PAGE[PAGE.index('id="viewData"') :]
    for element in IDS:
        assert f'id="{element}"' in data, element
    assert data.index('id="growthView"') < data.index(
        'id="homeEditor"'
    )  # (before the layout editor)
    assert 'id="growthView"' not in PAGE[: PAGE.index('id="viewData"')]
    # existing hooks of the page stay
    for element in ("neuroSkills", "leMap", "capSkills", "tab-growth", "growthOut"):
        assert f'id="{element}"' in PAGE, element
    assert '<link rel="stylesheet" href="/growth.css">' in PAGE
    assert (
        PAGE.index('src="/layout_editor.js"')
        < PAGE.index('src="/growth.js"')
        < PAGE.index('src="/minimap.js"')
    )
    assert PAGE.index('src="/i18n.js"') < PAGE.index('src="/growth.js"')


def test_the_two_care_style_selects_offer_every_style_the_server_does() -> None:
    from duckmind.life.fastforward import STYLES

    for select, none in (("gvFfStyleA", False), ("gvFfStyleB", True)):
        block = PAGE[PAGE.index(f'id="{select}"') :]
        block = block[: block.index("</select>")]
        values = re.findall(r'<option value="([^"]*)"', block)
        assert [v for v in values if v] == list(STYLES), select
        assert ("" in values) is none


def test_every_word_the_component_names_is_in_all_six_catalogs() -> None:
    named = _keys(JS) | _keys(PAGE)
    assert len(named) > 150
    for lang in CODES:
        have = _keys(_catalog(lang))
        assert not (named - have), (lang, sorted(named - have))
    # nothing in the component's own text: no 繁體中文 outside comments
    code = re.sub(r"/\*.*?\*/", "", JS, flags=re.S)
    code = re.sub(r"(^|[^:])//.*", r"\1", code)
    assert not re.search(r"[㐀-鿿]", code), "page words come from the catalogs"


def test_the_style_uses_tokens_only() -> None:
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", CSS), "colours come from the tokens"
    assert "border-radius" not in CSS
    assert "blur(" not in CSS and "gradient" not in CSS
    assert not re.search(r"box-shadow:[^;]*\d+px\s+\d+px\s+[1-9]\d*px\s", CSS), "hard shadows only"


def test_the_component_never_writes_markup_from_data() -> None:
    assert "innerHTML" not in JS and "insertAdjacentHTML" not in JS and "document.write" not in JS


def test_polling_is_light() -> None:
    assert "POLL_MS = 15000" in JS and 15000 <= 30000
    assert "visibilitychange" in JS and "view-data" in JS  # (only while the data page is shown)


# ── the component, in Node, over what the server makes ────────────────────────────────────────


def _session(days: int = 3) -> Session:
    start = datetime(2026, 10, 1, 8, 0, tzinfo=TPE).isoformat()
    session = Session(Config(seed=4107, name="蛋黃", tof=True, nest=False, start=start))
    dispatch(session, "bridge", {"api": bridge.BRIDGE_API})
    dispatch(session, "advance", {"seconds": days * 86400})
    session.core.s.first_word = "hm"  # (a first word, so that there is a call to play)
    t = int(session.core.s.now.timestamp())
    events = [
        {
            "k": "touch",
            "t": t + 1,
            "where": "head",
            "ms": 4000,
            "asleep": 0,
            "settled": 0,
            "doing": "",
        },
        {"k": "fall", "t": t + 50, "x_mm": 100, "y_mm": 200, "outcome": "hard"},
        {"k": "play", "t": t + 100, "minutes": 7},
    ]
    dispatch(session, "experience", {"events": events})
    return session


def _view(session: Session, **kw: Any) -> dict[str, Any]:
    tally = gv.Tally()
    tally.update(session.inputs)
    base: dict[str, Any] = {"name": "蛋黃", "live": True, "branch_id": "explorer", "pinned": True}
    return i18nwire.growth_answer(
        {"ok": True, "growth": gv.build(session.core, tally=tally, **{**base, **kw})}
    )["growth"]


def _live_view() -> dict[str, Any]:
    session = _session(3)
    learner = SimpleNamespace(
        people={
            1: SimpleNamespace(pid=1, name="小明", bond=0.4, greeted=2, exchanges=3, sightings=5),
            2: SimpleNamespace(pid=2, name=None, bond=0.1, greeted=0, exchanges=0, sightings=1),
        },
        places={},
        things={},
    )
    change = {
        "at": "2026-10-04T03:10:00+08:00", "why": "night", "waited_s": 41.5, "gd": [2, 3],
        "stage": ["hatchling", "hatchling"], "branch": ["explorer", "dreamer"], "unlocked": ["wander"],
        "dropped": [], "params_moved": 4, "gains_moved": 9, "largest_params": [], "largest_gains": [],
        "care": {"bond": [2000, 2080], "prefs": 2},
    }  # fmt: skip
    regrow = {
        "count": 2, "growth_day": 3, "applied_gd": 2, "due": True, "waiting_for": "awake",
        "problem": None, "last": change, "history": [change, change],
        "genome": {"gd": 2, "stage": "hatchling", "branch": None, "bond": 2080, "prefs": 2, "unlocked": 40,
                   "leash_m": 1.1, "social_tau_s": 3000.0, "greet_reflex": 0.3, "hatchling_wobble": 0.9},
    }  # fmt: skip
    voice = {
        "state": "ready",
        "active": {
            "key8": "abc12345",
            "calls_learned": [{"clip": "ph_I", "ms": 400}],
            "song": None,
        },
    }
    return _view(
        session,
        learner=learner,
        regrow=regrow,
        voice=voice,
        ignored={
            "thing": 2,
            "person": 1,
            "noise_self": 5,
            "lift_false": 1,
            "hand_false": 2,
            "company_false": 1,
            "a_kind_nobody_named": 3,
        },
        delivered={"refused": 0},
    )


def _test_view() -> dict[str, Any]:
    view = _view(_session(2), name="", live=False, branch_id=None, pinned=False)
    view["style"] = "rough"
    return view


needs_node = pytest.mark.skipif(NODE is None, reason="needs Node")


@pytest.fixture(scope="module")
def drawn(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    if NODE is None:
        pytest.skip("needs Node")
    live = _live_view()
    # a call the voice player can play, to put its button on the card
    live["voice"]["calls"][0]["playable"] = True
    path = tmp_path_factory.mktemp("growth") / "views.json"
    path.write_text(json.dumps({"live": live, "test": _test_view()}, ensure_ascii=False))
    out = subprocess.run(
        [NODE, str(HARNESS), str(STATIC), str(path)], capture_output=True, text=True, timeout=60
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


@needs_node
def test_the_live_duck_card_shows_who_it_is_with_names_as_data(drawn: dict[str, Any]) -> None:
    text = drawn["live"]["text"]
    assert "蛋黃" in text and "小明" in text  # (the duck's and a person's name, shown as they are)
    assert "page.growth.id.branch_family" in text  # the family's choice
    assert re.search(r"page\.growth\.id\.next_stage\|.*\"gd\":11", text)
    assert "page.growth.exp.k.touch" in text and "page.growth.exp.k.fall" in text
    assert (
        "page.growth.exp.fall.hard" in text and "page.growth.exp.minutes" in text
    )  # recent, with detail
    assert "page.growth.exp.ignored_list" in text  # what was left out as not real, in words
    for kind in ("thing", "person", "noise_self", "lift_false", "hand_false", "company_false"):
        assert f"page.growth.exp.ignored.{kind}" in text
    assert "page.growth.exp.ignored.other" in text  # (a kind with no words of its own)
    assert "a_kind_nobody_named" not in text  # (never a raw key on the page)
    assert "page.growth.bond.unnamed" in text  # a person it has not learnt the name of
    assert "page.growth.regrow.wait.awake" in text  # what the regrow is waiting for
    assert "page.growth.regrow.change" in text and "page.growth.regrow.branch_changed" in text
    assert "page.growth.voice.first_word" in text
    assert "page.growth.diary.title" in text
    for gone in ("innerHTML", "undefined", "NaN", "[object"):
        assert gone not in text


@needs_node
def test_a_learned_call_is_played_by_the_pages_voice_player_or_the_page_says_why(
    drawn: dict[str, Any],
) -> None:
    assert drawn["live"]["buttons"] == ["page.growth.voice.play"]
    assert drawn["played"] == [["ph_I"]]
    assert drawn["refused"] == [["ph_I"], "refresh"] and drawn["note"] is True


@needs_node
def test_a_test_duck_is_shown_by_the_same_component_without_a_name_or_sound(
    drawn: dict[str, Any],
) -> None:
    text = drawn["test"]["text"]
    assert "page.growth.ff.result_title" in text and "page.growth.ff.style.rough" in text
    assert "gv-name" not in drawn["test"]["classes"] and "gv-name" in drawn["live"]["classes"]
    assert drawn["test"]["buttons"] == 0
    assert "page.growth.voice.test_silent" in text


@needs_node
def test_two_ducks_are_compared_row_by_row(drawn: dict[str, Any]) -> None:
    rows = {r["key"]: r for r in drawn["rows"]}
    assert set(rows) == {f"page.growth.cmp.{k}" for k in (
        "growth_day stage branch bond likes avoid falls touches calls experiences regrows".split()
    )}  # fmt: skip
    assert (
        rows["page.growth.cmp.growth_day"]["cells"] == ["3", "2"]
        and rows["page.growth.cmp.growth_day"]["differs"]
    )
    assert rows["page.growth.cmp.bond"]["cells"][0].endswith("%")
    assert not any(drawn["rows_same"])  # (the same duck twice: nothing differs)


@needs_node
def test_the_time_estimate_weighs_the_cheap_early_days(drawn: dict[str, Any]) -> None:
    after_five, nothing_yet, finished, middle = drawn["eta"]
    assert nothing_yet is None and finished == 0
    assert after_five == round(7 / 5 * 25 * 3.2)  # (five cheap days in 7 s: 25 dearer ones to come)
    assert middle == round(
        50 / (5 + 10 * 3.2) * 15 * 3.2
    )  # (half the days in 50 s: the dearer half to come)


@needs_node
def test_every_kind_wait_style_and_outcome_the_server_can_send_has_a_word(
    drawn: dict[str, Any],
) -> None:
    from duckmind.life.fastforward import STYLES
    from duckmind.life.regrow import calm  # noqa: F401  (the reasons are its words)

    assert set(drawn["keys"]["kinds"]) == set(gv.KIND_ORDER) == set(bridge_v2.KINDS)
    assert set(drawn["keys"]["styles"]) == set(STYLES)
    assert set(drawn["keys"]["falls"]) == set(bridge.FALL_OUTCOMES)
    assert set(drawn["keys"]["how"]) >= {"liked", "promoted", "song", "first_word"}
    # the reasons a regrow can be waiting (regrow.calm's, the host's driver check, "settling")
    source = (
        Path(__file__).resolve().parents[1] / "src" / "duckmind" / "life" / "regrow.py"
    ).read_text()
    reasons = set(
        re.findall(
            r'return "([a-z]+)"', source[source.index("def calm") : source.index("class Report")]
        )
    )
    host = (
        Path(__file__).resolve().parents[1] / "src" / "duckmind" / "simweb" / "life_host.py"
    ).read_text()
    reasons |= set(
        re.findall(
            r'return "([a-z]+)"',
            host[host.index("def _driver_busy") : host.index("def regrow_tick")],
        )
    )
    reasons.add("settling")
    assert reasons <= set(drawn["keys"]["waits"]), reasons - set(drawn["keys"]["waits"])


IGNORED_KINDS = (
    "thing",
    "person",
    "noise_self",
    "lift_false",
    "hand_false",
    "company_false",
    "other",
)


def test_every_kind_growth_leaves_out_has_words_in_all_six_languages() -> None:
    """The growth view's 'left out as not real' line names each kind in plain words, not a key."""
    from duckmind.i18n import icu, locales
    from duckmind.i18n.catalog import read_catalog

    js = (STATIC / "growth.js").read_text(encoding="utf-8")
    for kind in IGNORED_KINDS:
        assert f"page.growth.exp.ignored.{kind}" in js
    for code in locales.CODES:
        catalog = read_catalog(code, "page") or {}
        for kind in IGNORED_KINDS:
            key = f"page.growth.exp.ignored.{kind}"
            words = catalog[key]
            assert "{n}" in words and key not in words, (code, key)
            assert icu.check(words, code) == []
        assert "{by}" in catalog["page.growth.exp.ignored_list"]
    zh = read_catalog("zh-Hant", "page") or {}
    en = read_catalog("en", "page") or {}
    for kind in IGNORED_KINDS:
        key = f"page.growth.exp.ignored.{kind}"
        assert zh[key] != en[key]  # (translated, not copied)


def test_every_ignored_kind_the_experiences_count_has_a_line() -> None:
    """The kinds Experiences puts in ``ignored`` are the ones the page has words for."""
    import inspect

    from duckmind.life import experience

    source = inspect.getsource(experience)
    for kind in ("noise_self", "lift_false", "hand_false", "company_false"):
        assert f'"{kind}"' in source
        assert f"page.growth.exp.ignored.{kind}" in (STATIC / "growth.js").read_text(
            encoding="utf-8"
        )
