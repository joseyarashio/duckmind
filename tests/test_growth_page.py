"""Growth G4: the 「成長」 section of the 資料 page (docs/design/GROWTH-G4-2026-10-11.md).

The catalogs' words; the component itself in Node over a tiny fake page, drawing the views the
server really makes; and the component on the page's own markup (tests/page_tree_node_harness.py):
when it reads, what its form sends. How it looks and behaves in a browser is checked by hand
(screenshots in the measurement records, not in the repo).
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
from page_tree_node_harness import run

import duckmind.world.legacy  # noqa: F401  (registers LegacyWorld)
from duckmind.duckcore import bridge, bridge_v2
from duckmind.duckcore.api import Config, Session, dispatch
from duckmind.i18n import CODES
from duckmind.life import growth_view as gv
from duckmind.simweb import i18nwire

STATIC = Path(__file__).resolve().parents[1] / "src" / "duckmind" / "simweb" / "static"
PAGE = (STATIC / "index.html").read_text(encoding="utf-8")
JS = (STATIC / "growth.js").read_text(encoding="utf-8")
NODE = shutil.which("node")
HARNESS = Path(__file__).with_name("growth_node_harness.js")
TPE = timezone(timedelta(hours=8))

# names with markup in them: data, shown as they are
DUCK_NAME = "<img src=x onerror=alert(1)>"
PERSON_NAME = "<b>小明</b>"


def _catalog(lang: str) -> str:
    return (STATIC / f"i18n.{lang}.page.js").read_text(encoding="utf-8")


def _keys(text: str) -> set[str]:
    return set(re.findall(r"page\.growth\.[a-z0-9_.]+[a-z0-9_]", text))


# ── the catalogs ──────────────────────────────────────────────────────────────────────────────


def test_every_word_the_component_names_is_in_all_six_catalogs() -> None:
    named = _keys(JS) | _keys(PAGE)
    assert len(named) > 150
    for lang in CODES:
        have = _keys(_catalog(lang))
        assert not (named - have), (lang, sorted(named - have))


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

    assert set(drawn["keys"]["kinds"]) == set(gv.KIND_ORDER) == set(bridge_v2.KINDS)
    assert set(drawn["keys"]["styles"]) == set(STYLES)
    assert set(drawn["keys"]["falls"]) == set(bridge.FALL_OUTCOMES)
    assert set(drawn["keys"]["how"]) >= {"liked", "promoted", "song", "first_word"}


@needs_node
def test_every_reason_a_regrow_waits_for_has_a_word(drawn: dict[str, Any]) -> None:
    """A regrow waits for a calm sleeping duck: ``regrow.calm`` and the host's driver check say what
    is in the way, the page says it in words."""
    from duckmind.life import regrow
    from duckmind.neuro import select
    from duckmind.neuro.contract import BEHAVIOURS
    from duckmind.simweb.life_host import LifeHost

    asleep = sorted(select.SLEEP)[0]
    awake = sorted(set(BEHAVIOURS) - select.SLEEP)[0]

    def brain(**state: Any) -> SimpleNamespace:
        return SimpleNamespace(behaviour=asleep, **state)

    def frame(*events: str, **body: Any) -> SimpleNamespace:
        return SimpleNamespace(body=SimpleNamespace(**body), events=list(events))

    still = frame()
    assert regrow.calm(brain(), still, 1000) is None  # asleep, nothing going on: no wait
    in_the_way = [
        regrow.calm(SimpleNamespace(behaviour=awake), still, 1000),
        regrow.calm(brain(), frame(fallen=True), 1000),
        regrow.calm(brain(), frame(posture="stand"), 1000),
        regrow.calm(brain(), frame(active_skill="kick_left"), 1000),
        regrow.calm(brain(), frame(sorted(regrow.BUSY_EVENTS)[0]), 1000),
        regrow.calm(brain(), still, 1000, SimpleNamespace(reflex="withdraw")),
        regrow.calm(brain(_escape=object()), still, 1000),
        regrow.calm(brain(_perform_until_ms=5000), still, 1000),
        regrow.calm(brain(_t0=0, _touch_s=0.5), still, 1000),
        regrow.calm(brain(play_out=SimpleNamespace(skill="kick_left")), still, 1000),
    ]
    sitting = SimpleNamespace(policy="sit")
    for driver, state in (
        (SimpleNamespace(_motion_run=object()), sitting),
        (SimpleNamespace(emotions=SimpleNamespace(run=object())), sitting),
        (SimpleNamespace(), SimpleNamespace(policy="get_up")),
        (SimpleNamespace(_link=SimpleNamespace(flight_t0_ns=1)), sitting),
    ):
        in_the_way.append(LifeHost._driver_busy(None, driver, state))  # type: ignore[arg-type]
    assert LifeHost._driver_busy(None, SimpleNamespace(), sitting) is None  # type: ignore[arg-type]
    assert None not in in_the_way
    reasons = {str(reason) for reason in in_the_way}
    assert len(reasons) == 12  # (a skill and an emotion are said the same by both)
    reasons.add("settling")  # (the regrower's own: calm, but not for long enough yet)
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


# ── the players' 成長 tab (item 6): the short view of the same reading ──────────────────────────


@needs_node
def test_the_growth_tab_says_day_stage_direction_closeness_and_real_recent_events(
    drawn: dict[str, Any],
) -> None:
    short, live = drawn["short"]["text"], drawn["live"]["text"]
    assert "page.growth.id.growth_day" in short and "gv-gd-num" in drawn["short"]["classes"]
    assert "page.growtab.direction" in short and "page.growth.id.next_stage" in short
    assert "小明" in short and "page.growth.bond.family" in short  # (names as data)
    # the events and the diary lines are the duck's own record, the same the lab shows
    for line in re.findall(r"page\.growth\.exp\.recent_line\|[^\n]*", short):
        assert line in live
    assert "page.growth.diary.day" in short
    assert drawn["short"]["buttons"] == 0  # (nothing to press but the link under it)
    for gone in ("innerHTML", "undefined", "NaN", "[object"):
        assert gone not in short


@needs_node
def test_the_growth_tab_keeps_to_the_newest_few_and_says_what_is_missing(
    drawn: dict[str, Any],
) -> None:
    busy = drawn["short_busy"]
    assert busy.count("page.growth.exp.recent_line") == 5
    assert "02:00:00" not in busy and "T00:00" not in busy  # (no raw times on the page)
    assert "day0-line0" in busy and "day1-line3" in busy and "day0-line4" not in busy
    assert "day2-line0" not in busy  # (two days only)
    assert (
        busy.count('page.growth.diary.more|{"n":5}') == 2
    )  # (3 lines left out + 2 the server held back)
    bare = drawn["short_bare"]
    assert "page.growth.bond.none" in bare and "page.growth.exp.recent_none" in bare
    assert "page.growth.diary.none" in bare


# ── the component on the page's own markup ────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def mounted() -> dict[str, Any]:
    """shell.js and growth.js over the page's markup: the tabs are pressed, a minute passes here
    and there, the lab is opened, the fast-forward form is sent once for each style it offers."""
    if NODE is None:
        pytest.skip("needs Node")
    view = _live_view()
    view["identity"]["name"] = DUCK_NAME
    view["bonds"]["people"][0]["name"] = PERSON_NAME
    return run("growth", {"shell.js", "growth.js"}, {"view": view})


@needs_node
def test_the_record_is_read_only_while_it_can_be_seen_and_not_often(
    mounted: dict[str, Any],
) -> None:
    reads = mounted["reads"]
    assert reads["elsewhere"] == 0  # another tab of the panel is open: nothing is asked
    assert reads["opened"] == 1  # the 成長 tab opens: read at once…
    assert "page.growth.id.growth_day" in mounted["short"] and PERSON_NAME in mounted["short"]
    assert 2 <= reads["minute_open"] - reads["opened"] <= 6  # …then now and then
    assert reads["in_the_back"] == reads["minute_open"]  # the browser tab in the back: nothing
    assert reads["to_the_front"] == reads["in_the_back"] + 1  # in front again: at once
    assert reads["left"] == reads["to_the_front"]  # away to another tab: nothing more
    assert reads["lab"] == reads["left"] + 1  # the lab opens: its whole record, at once
    assert "page.growth.id.title" in mounted["long"]


@needs_node
def test_names_with_markup_in_them_are_shown_as_they_are(mounted: dict[str, Any]) -> None:
    assert DUCK_NAME in mounted["long"]  # (a text of its own, letter for letter)
    assert PERSON_NAME in mounted["long"] and PERSON_NAME in mounted["short"]
    assert "img" not in mounted["made"] and mounted["markup"] == 0  # nothing was parsed as markup


@needs_node
def test_every_care_style_the_form_offers_is_one_the_server_runs(mounted: dict[str, Any]) -> None:
    from duckmind.life.fastforward import STYLES, check_request

    assert mounted["offered"]["gvFfStyleA"] == list(STYLES)
    assert mounted["offered"]["gvFfStyleB"] == ["", *STYLES]  # (B may be left out)
    sent = [post["body"] for post in mounted["sent"] if post["url"] == "/api/life/fastforward"]
    assert len(sent) == 2 * len(STYLES) + 1  # the form was sent once for each choice
    asked: set[str] = set()
    for body in sent:
        days, styles, seed = check_request(body["days"], body["styles"], body["seed"])
        assert (days, seed) == (30, 4107) and 1 <= len(styles) <= 2
        asked.update(styles)
    assert asked == set(STYLES)
