"""i18n: the page runtime (static/i18n.js) renders exactly what the Python layer renders.

Runs i18n.js in Node with a tiny fake page (tests/i18n_node_harness.js) over a catalog of its
own, and compares every message, number, plural, list and server field with duckmind.i18n's
rendering in the same language; also the language choice (?lang=, the viewer's choice,
navigator.languages), the pseudo-locale, switching, and a language without catalogs.
Skips when Node is not installed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from duckmind.i18n import Catalogs, locales, render, use_catalogs
from duckmind.i18n.catalog import STATIC, write_catalog, write_manifest

NODE = shutil.which("node")
HARNESS = Path(__file__).with_name("i18n_node_harness.js")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node is not installed")

SOURCE = {
    "fmt.sep.enum": "、",
    "fmt.sep.babble": "·",
    "page.i18n.picker.label": "語言",
    "page.i18n.picker.title": "頁面語言",
    "page.i18n.load_failed": "這個語言的文字沒有載入成功，先用繁體中文顯示",
    "page.t.days": "快轉 {days} 天中……",
    "page.t.secs": "驚嚇：愣住 {s, number, ::.0} 秒",
    "page.t.speed": "腦想走 {v, number, ::.00} 公尺/秒",
    "page.t.count": "共 {n, number, ::group-auto} 件",
    "page.t.pad": "{h, number, ::integer-width/*00}:{m, number, ::integer-width/*00}",
    "page.t.tv": "電視{on, select, yes {開著} other {關著}}",
    "page.t.nth": "第 {n} 號",
    "page.t.html": "按 <b>{name}</b> 開始",
    "neuro.rx.lean": "{site}{felt}，靠過去",
    "neuro.site.head": "頭頂",
    "neuro.felt.stroked": "被摸",
    "neuro.body.line": "身體：{parts}",
    "core.tag.chirp": "啾",
    "core.tag.coo": "咕～",
}
EN = {
    "fmt.sep.enum": ", ",
    "fmt.sep.babble": " · ",
    "page.i18n.picker.label": "Language",
    "page.i18n.picker.title": "Page language",
    "page.i18n.load_failed": "This language did not load; showing 繁體中文 for now",
    "page.t.days": "Fast-forwarding {days, plural, one {# day} other {# days}}…",
    "page.t.secs": "Startle: froze for {s, number, ::.0} {s, plural, one {second} other {seconds}}",
    "page.t.speed": "Brain wants to walk {v, number, ::.00} m/s",
    "page.t.count": "{n, number, ::group-auto} {n, plural, one {item} other {items}} in all",
    "page.t.pad": "{h, number, ::integer-width/*00}:{m, number, ::integer-width/*00}",
    "page.t.tv": "TV {on, select, yes {on} other {off}}",
    "page.t.nth": "{n, selectordinal, one {#st} two {#nd} few {#rd} other {#th}}",
    "page.t.html": "Press <b>{name}</b> to start",
    "neuro.rx.lean": "{felt} on the {site}, leans in",
    "neuro.site.head": "head",
    "neuro.felt.stroked": "stroked",
    "neuro.body.line": "Body: {parts}",
    "core.tag.chirp": "peep",
    "core.tag.coo": "coo~",
}
ES = {
    **EN,
    "fmt.sep.enum": ", ",
    "page.t.days": "Avanzando {days, plural, one {# día} other {# días}}…",
    "page.t.secs": "Susto: se quedó quieto {s, number, ::.0} s",
    "page.t.speed": "Quiere caminar a {v, number, ::.00} m/s",
    "page.t.count": "{n, number, ::group-auto} en total",
}


@pytest.fixture(scope="module")
def static(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    root = tmp_path_factory.mktemp("static")
    shutil.copy(STATIC / "i18n.js", root / "i18n.js")
    for lang, messages in (("zh-Hant", SOURCE), ("en", EN), ("es", ES)):
        by_domain: dict[str, dict[str, str]] = {}
        for key, text in messages.items():
            domain = {"fmt": "page", "page": "page", "neuro": "neuro", "core": "core"}[
                key.split(".")[0]
            ]
            by_domain.setdefault(domain, {})[key] = text
        for domain, items in by_domain.items():
            write_catalog(lang, domain, items, root)
    write_manifest(root)
    use_catalogs(Catalogs(root))
    yield root
    use_catalogs(None)


def run(static: Path, cases: list[dict[str, Any]], **spec: Any) -> dict[str, Any]:
    file = static / "cases.json"
    file.write_text(json.dumps({**spec, "cases": cases}, ensure_ascii=False), encoding="utf-8")
    done = subprocess.run(
        [str(NODE), str(HARNESS), str(static), str(file)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout)
    for index, result in enumerate(out["results"]):
        assert result["ok"], (cases[index], result)
    return out


WIRES: list[Any] = [
    {"k": "page.t.days", "p": {"days": 1}},
    {"k": "page.t.days", "p": {"days": 3}},
    {"k": "page.t.secs", "p": {"s": 1.26}},
    {"k": "page.t.secs", "p": {"s": 1}},
    {"k": "page.t.secs", "p": {"s": -0.04}},
    {"k": "page.t.speed", "p": {"v": 0.4}},
    {"k": "page.t.count", "p": {"n": 12345}},
    {"k": "page.t.count", "p": {"n": 1234}},
    {"k": "page.t.count", "p": {"n": 1}},
    {"k": "page.t.pad", "p": {"h": 7, "m": 5}},
    {"k": "page.t.tv", "p": {"on": "yes"}},
    {"k": "page.t.tv", "p": {"on": "no"}},
    {"k": "page.t.nth", "p": {"n": 2}},
    {"k": "page.t.nth", "p": {"n": 13}},
    {
        "k": "neuro.rx.lean",
        "p": {"site": {"k": "neuro.site.head"}, "felt": {"k": "neuro.felt.stroked"}},
    },
    {"k": "neuro.rx.lean", "p": {"site": "蛋黃", "felt": {"k": "neuro.felt.stroked"}}},
    {
        "k": "neuro.body.line",
        "p": {"parts": {"list": [{"k": "neuro.site.head"}, "蛋黃", 0.5], "sep": "fmt.sep.enum"}},
    },
    {"list": [{"k": "core.tag.chirp"}, {"k": "core.tag.coo"}], "sep": "fmt.sep.babble"},
    {"join": [{"k": "neuro.site.head"}, "！"]},
    {"cut": {"k": "neuro.body.line", "p": {"parts": "x"}}, "text": "身體"},
    "純資料",
    2.5,
]


def _keys(wire: Any) -> set[str]:
    if isinstance(wire, dict):
        out = {wire["k"]} if "k" in wire else set()
        for value in wire.values():
            out |= _keys(value)
        return out
    if isinstance(wire, list):
        return set().union(*(_keys(v) for v in wire)) if wire else set()
    return set()


@pytest.mark.parametrize("lang", ["zh-Hant", "en", "es", "pseudo"])
def test_page_renders_what_python_renders(static: Path, lang: str) -> None:
    # the source language loads only the page's own words (server words arrive in 繁體中文)
    wires = [
        w
        for w in WIRES
        if lang != "zh-Hant" or all(k.startswith(("page.", "fmt.")) for k in _keys(w))
    ]
    cases = [{"op": "wire", "lang": lang, "wire": wire} for wire in wires]
    out = run(static, cases)
    for wire, result in zip(wires, out["results"], strict=True):
        assert result["out"] == render(wire, lang, strict=True), wire


def test_t_html_escapes_values_but_keeps_markup(static: Path) -> None:
    out = run(
        static,
        [{"op": "tHtml", "lang": "en", "key": "page.t.html", "params": {"name": "<i>x</i>"}}],
    )
    assert out["results"][0]["out"] == "Press <b>&lt;i&gt;x&lt;/i&gt;</b> to start"


def test_language_tags_are_read_like_python(static: Path) -> None:
    tags = [
        "zh-TW",
        "zh-HK",
        "zh",
        "zh-CN",
        "zh-Hans-SG",
        "en-GB",
        "es-419",
        "ja",
        "ko-KR",
        "fr",
        "",
        "pseudo",
    ]
    out = run(
        static,
        [{"op": "normalize", "tag": tag} for tag in tags]
        + [{"op": "negotiate", "tags": ["fr", "de-AT", "es-MX"]}],
    )
    got = [r["out"] for r in out["results"]]
    assert got[:-1] == [locales.normalize(tag) for tag in tags]
    assert got[-1] == "es"


def test_choice_order_url_then_stored_then_browser(static: Path) -> None:
    assert run(static, [{"op": "lang"}], search="?lang=en", stored="es")["results"][0]["out"] == {
        "lang": "en",
        "htmlLang": "en",
        "stored": "es",
    }
    assert (
        run(static, [{"op": "lang"}], stored="es", languages=["en-US"])["results"][0]["out"]["lang"]
        == "es"
    )
    assert (
        run(static, [{"op": "lang"}], languages=["fr-FR", "en-US"])["results"][0]["out"]["lang"]
        == "en"
    )
    assert (
        run(static, [{"op": "lang"}], languages=["fr-FR"])["results"][0]["out"]["lang"] == "zh-Hant"
    )
    # a language whose catalogs do not exist yet falls back to the source, quietly
    assert (
        run(static, [{"op": "lang"}], search="?lang=ko")["results"][0]["out"]["lang"] == "zh-Hant"
    )
    pseudo = run(
        static,
        [{"op": "lang"}, {"op": "t", "key": "page.t.days", "params": {"days": 2}}],
        search="?lang=pseudo",
    )
    assert (
        pseudo["results"][0]["out"]["lang"] == "pseudo"
        and pseudo["results"][0]["out"]["htmlLang"] == "zh-Hant"
    )
    assert pseudo["results"][1]["out"] == "⟦快轉 ⟧2⟦ 天中……⟧"


def test_a_choice_from_the_picker_is_remembered(static: Path) -> None:
    out = run(static, [{"op": "lang", "lang": "es", "remember": True}])
    assert out["results"][0]["out"] == {"lang": "es", "htmlLang": "es", "stored": "es"}


def test_switching_rerenders_marked_elements(static: Path) -> None:
    out = run(
        static,
        [
            {"op": "apply", "key": "page.t.days", "params": {"days": 2}, "to": "en"},
            {"op": "events"},
        ],
    )
    assert out["results"][0]["out"] == {
        "before": "快轉 2 天中……",
        "after": "Fast-forwarding 2 days…",
    }
    assert "i18n:change" in out["results"][1]["out"]


def test_switching_rerenders_server_fields_and_keeps_values(static: Path) -> None:
    obj = {"note": "頭頂", "_i18n": {"note": {"k": "neuro.site.head"}}}
    cases = [
        {"op": "show", "obj": obj, "name": "note", "to": "en"},
        {
            "op": "textAndTitle",
            "key": "page.t.days",
            "params": {"days": 5},
            "titleKey": "page.i18n.picker.title",
            "to": "zh-Hant",
        },
    ]
    got = [r["out"] for r in run(static, cases)["results"]]
    assert got[0] == {"before": "頭頂", "after": "head"}
    assert got[1] == {"text": "快轉 5 天中……", "title": "頁面語言"}


def test_server_fields(static: Path) -> None:
    obj = {
        "note": "頭頂被摸，靠過去",
        "tags": ["頭頂", "沒有訊息"],
        "_i18n": {
            "note": {
                "k": "neuro.rx.lean",
                "p": {"site": {"k": "neuro.site.head"}, "felt": {"k": "neuro.felt.stroked"}},
            },
            "tags": [{"k": "neuro.site.head"}, None],
        },
    }
    broken = {"note": "原文", "_i18n": {"note": {"k": "page.nowhere"}}}
    cases = [
        {"op": "field", "lang": "zh-Hant", "obj": obj, "name": "note"},
        {"op": "field", "lang": "en", "obj": obj, "name": "note"},
        {"op": "items", "lang": "en", "obj": obj, "name": "tags"},
        {"op": "field", "lang": "en", "obj": broken, "name": "note"},
        {"op": "field", "lang": "en", "obj": {"note": "只有字"}, "name": "note"},
    ]
    got = [r["out"] for r in run(static, cases)["results"]]
    assert got == [
        "頭頂被摸，靠過去",
        "stroked on the head, leans in",
        ["head", "沒有訊息"],
        "原文",
        "只有字",
    ]


def test_weekday_in_the_source_is_todays_character(static: Path) -> None:
    out = run(
        static, [{"op": "weekday", "lang": "zh-Hant"}, {"op": "number", "value": 1.25, "digits": 1}]
    )
    assert "".join(out["results"][0]["out"]) == "一二三四五六日"
    assert out["results"][1]["out"] == "1.3"  # the source keeps toFixed (what the page printed)
