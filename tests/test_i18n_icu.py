"""i18n: the ICU message subset, plural rules, number formats and language choice (Python side).

The page runtime (static/i18n.js) implements the same rules; tests/test_i18n_runtime_js.py
checks that both sides render the same words.
"""

from __future__ import annotations

import pytest

from duckmind.i18n import icu, locales, numbers, plural


def fmt(message: str, params: dict, lang: str = "zh-Hant") -> str:
    return icu.format_nodes(icu.parse(message), params, lang, lambda value, _lang: str(value))


def test_plain_arguments_and_escapes() -> None:
    assert fmt("快轉 {days} 天中……", {"days": 3}) == "快轉 3 天中……"
    assert fmt("it''s '{'x'}' {x}", {"x": "y"}, "en") == "it's {x} y"
    assert fmt("don't stop {x}", {"x": 1}, "en") == "don't stop 1"
    for text in ["{{x}}", "a{b}c", "'{", "{'", "x''y", "}{"]:
        assert fmt(icu.escape(text), {}, "en") == text


def test_numbers_follow_python_in_the_source() -> None:
    # the source language prints exactly what the f-string printed
    for value, spec in [
        (1.25, ".1f"),
        (0.4, ".2f"),
        (3, ".1f"),
        (-0.04, ".1f"),
        (2.5, ".0f"),
        (7, "02d"),
    ]:
        skeleton = numbers.style_for_spec(spec)
        assert skeleton is not None
        assert fmt("{x, number, " + skeleton + "}", {"x": value}) == format(value, spec)
    assert fmt("{x}", {"x": 1.0}) == "1.0"  # str(): what f"{x}" printed
    assert fmt("{x}", {"x": 1.0}, "en") == "1"  # other languages: as the page (3.0 === 3)
    assert fmt("{x, number, ::.0}", {"x": 1.25}, "es") == "1,2"
    assert fmt("{x}", {"x": 0.5}, "es") == "0,5"
    assert fmt("{x, number, ::.0 group-auto}", {"x": 12345.5}, "es") == "12.345,5"
    assert (
        fmt("{x, number, ::.0 group-auto}", {"x": 1234.5}, "es") == "1234,5"
    )  # es groups from 5 digits
    assert fmt("{x, number, ::.0 group-auto}", {"x": 1234.5}, "en") == "1,234.5"
    assert fmt("{x, number}", {"x": 1234.5}, "en") == "1234.5"  # no grouping unless asked
    assert fmt("{x, number, integer}", {"x": 3.6}, "en") == "4"


def test_plural_and_select() -> None:
    days = "{n, plural, one {# day} other {# days}}"
    assert fmt(days, {"n": 1}, "en") == "1 day"
    assert fmt(days, {"n": 3}, "en") == "3 days"
    assert fmt(days, {"n": 1.5}, "en") == "1.5 days"
    es = "{n, plural, one {# segundo} many {# de segundos} other {# segundos}}"
    assert fmt(es, {"n": 1}, "es") == "1 segundo"
    assert fmt(es, {"n": 2}, "es") == "2 segundos"
    assert fmt(es, {"n": 1000000}, "es") == "1000000 de segundos"
    assert fmt("{n, plural, =0 {nobody} one {one} other {#}}", {"n": 0}, "en") == "nobody"
    assert (
        fmt(
            "{n, plural, offset:1 =0 {nobody} one {you and one other} other {you and # others}}",
            {"n": 3},
            "en",
        )
        == "you and 2 others"
    )
    assert (
        fmt("{n, selectordinal, one {#st} two {#nd} few {#rd} other {#th}}", {"n": 22}, "en")
        == "22nd"
    )
    assert (
        fmt("{n, selectordinal, one {#st} two {#nd} few {#rd} other {#th}}", {"n": 13}, "en")
        == "13th"
    )
    assert (
        fmt("{side, select, left {左邊} right {右邊} other {正前方}}", {"side": "right"}) == "右邊"
    )
    assert (
        fmt("{side, select, left {左邊} right {右邊} other {正前方}}", {"side": "up"}) == "正前方"
    )
    assert fmt("{n, plural, other {# 秒}}", {"n": 2}, "ja") == "2 秒"
    nested = "{n, plural, one {{n, number, ::.0} second} other {{n, number, ::.0} seconds}}"
    assert fmt(nested, {"n": 1.25}, "en") == "1.2 seconds"


def test_plural_rules_match_cldr() -> None:
    assert [plural.cardinal("en", n) for n in (0, 1, 2, 1.5, 1.0)] == [
        "other",
        "one",
        "other",
        "other",
        "one",
    ]
    assert [plural.cardinal("es", n) for n in (0, 1, 2, 1000000, 2000000, 1.5)] == [
        "other",
        "one",
        "other",
        "many",
        "many",
        "other",
    ]
    for lang in ("zh-Hant", "zh-Hans", "ja", "ko", "pseudo"):
        assert {plural.cardinal(lang, n) for n in (0, 1, 2, 1.5)} == {"other"}
    assert [plural.ordinal("en", n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 101)] == [
        "one",
        "two",
        "few",
        "other",
        "other",
        "other",
        "other",
        "one",
        "two",
        "few",
        "one",
    ]
    assert plural.categories("es") == ("one", "many", "other")


def test_check_reports_forms_a_language_does_not_have() -> None:
    assert icu.check("{n, plural, one {x} other {y}}", "en") == []
    assert icu.check("{n, plural, one {x} few {y} other {z}}", "es") == [
        "n: es has no 'few' plural form (it has one, many, other)"
    ]
    assert icu.check("{n, plural, one {x} other {y}}", "ja") == [
        "n: ja has no 'one' plural form (it has other)"
    ]
    assert icu.check("{n, plural, =1 {x} other {y}}", "ja") == []
    assert icu.check("{n, plural, one {x}}", "en")  # no other: an error
    assert icu.check("{a", "en")
    assert icu.check("x}", "en")
    assert icu.check("{x, number, ::.0 grouping-weird}", "en")


def test_arguments_and_plain_text() -> None:
    nodes = icu.parse("{a} {b, number, ::.0} {c, plural, one {{d}} other {# 天}}")
    assert icu.arguments(nodes) == {"a": "plain", "b": "number", "c": "plural", "d": "plain"}
    assert icu.plain_text(icu.parse("身體：{parts}（{s, number, ::.0} 秒）")) == "身體：（ 秒）"


def test_pseudo_wraps_every_catalog_text() -> None:
    out = fmt("驚嚇：愣住 {s, number, ::.0} 秒", {"s": 1.25}, "pseudo")
    assert out == "⟦驚嚇：愣住 ⟧1.2⟦ 秒⟧"


def test_missing_values() -> None:
    with pytest.raises(icu.MessageError):
        fmt("{x}", {})
    assert (
        icu.format_nodes(icu.parse("a{x}b"), {}, "en", lambda v, _l: str(v), strict=False)
        == "a{x}b"
    )


@pytest.mark.parametrize(
    ("tag", "code"),
    [
        ("zh-TW", "zh-Hant"),
        ("zh-Hant-HK", "zh-Hant"),
        ("zh", "zh-Hant"),
        ("zh_hk", "zh-Hant"),
        ("zh-CN", "zh-Hans"),
        ("zh-Hans", "zh-Hans"),
        ("zh-SG", "zh-Hans"),
        ("en-US", "en"),
        ("EN", "en"),
        ("es-419", "es"),
        ("ja-JP", "ja"),
        ("ko-KR", "ko"),
        ("fr-FR", None),
        ("", None),
        (None, None),
        ("pseudo", "pseudo"),
    ],
)
def test_normalize(tag: str | None, code: str | None) -> None:
    assert locales.normalize(tag) == code


def test_negotiate_takes_the_first_supported_language() -> None:
    assert locales.negotiate(["fr-FR", "de", "ja-JP", "en"]) == "ja"
    assert locales.negotiate(["fr", "de"]) == "zh-Hant"
    assert locales.negotiate([]) == "zh-Hant"
    assert locales.negotiate(["pseudo", "ko"]) == "ko"  # never chosen from the browser


def test_the_six_languages() -> None:
    assert locales.CODES == ("zh-Hant", "en", "ja", "es", "zh-Hans", "ko")
    assert [lang.native for lang in locales.LANGS] == [
        "繁體中文",
        "English",
        "日本語",
        "Español",
        "简体中文",
        "한국어",
    ]
    assert locales.SOURCE == "zh-Hant"
