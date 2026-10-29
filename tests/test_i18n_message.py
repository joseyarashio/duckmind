"""i18n: the message layer — Msg is the 繁體中文 str it replaces, and carries its key and values.

Uses a small catalog of its own (tmp_path), so it does not depend on the real catalog's words.
"""

from __future__ import annotations

import copy
import json
import pickle
from collections.abc import Iterator
from pathlib import Path

import pytest

from duckmind.i18n import (
    Catalogs,
    MissingKey,
    Msg,
    join,
    lst,
    msg,
    render,
    render_key,
    to_wire,
    use_catalogs,
)
from duckmind.i18n.catalog import dump_file, parse_file, write_catalog

SOURCE = {
    "fmt.sep.enum": "、",
    "fmt.sep.babble": "·",
    "neuro.rx.lean_in.felt_at": "{site}{felt}，靠過去",
    "neuro.touch_site.head": "頭頂",
    "neuro.felt.stroked": "被摸",
    "neuro.stub.clap": "驚嚇：愣住 {s, number, ::.0} 秒",
    "page.app.toast.fast_forward": "快轉 {days} 天中……",
    "neuro.body.line": "身體：{parts}",
    "page.app.tv": "電視{on, select, yes {開著} other {關著}}",
}
EN = {
    "fmt.sep.enum": ", ",
    "neuro.rx.lean_in.felt_at": "{felt} on the {site}, leans in",
    "neuro.touch_site.head": "head",
    "neuro.felt.stroked": "stroked",
    "page.app.toast.fast_forward": "Fast-forwarding {days, plural, one {# day} other {# days}}…",
    "neuro.body.line": "Body: {parts}",
}


@pytest.fixture()
def cats(tmp_path: Path) -> Iterator[Catalogs]:
    for lang, messages in (("zh-Hant", SOURCE), ("en", EN)):
        by_domain: dict[str, dict[str, str]] = {}
        for key, text in messages.items():
            domain = {"fmt": "page", "page": "page", "neuro": "neuro"}[key.split(".")[0]]
            by_domain.setdefault(domain, {})[key] = text
        for domain, items in by_domain.items():
            write_catalog(lang, domain, items, tmp_path)
    catalogs = Catalogs(tmp_path)
    use_catalogs(catalogs)
    try:
        yield catalogs
    finally:
        use_catalogs(None)


def test_msg_is_the_source_str(cats: Catalogs) -> None:
    m = msg(
        "neuro.rx.lean_in.felt_at",
        site=msg("neuro.touch_site.head"),
        felt=msg("neuro.felt.stroked"),
    )
    assert isinstance(m, str)
    assert m == "頭頂被摸，靠過去"
    assert str(m) == "頭頂被摸，靠過去" and type(str(m)) is str
    assert "腦：" + m == "腦：頭頂被摸，靠過去"
    assert f"{m}!" == "頭頂被摸，靠過去!"
    assert "靠過去" in m and m.endswith("靠過去")
    assert json.dumps({"zh": m}, ensure_ascii=False) == '{"zh": "頭頂被摸，靠過去"}'
    assert repr(m) == repr("頭頂被摸，靠過去")
    assert hash(m) == hash("頭頂被摸，靠過去")


def test_numbers_render_like_the_fstring(cats: Catalogs) -> None:
    for s in (1.25, 0.04, 3, 12.0):
        assert msg("neuro.stub.clap", s=s) == f"驚嚇：愣住 {s:.1f} 秒"


def test_wire_and_other_languages(cats: Catalogs) -> None:
    m = msg(
        "neuro.rx.lean_in.felt_at",
        site=msg("neuro.touch_site.head"),
        felt=msg("neuro.felt.stroked"),
    )
    wire = m.wire()
    assert wire == {
        "k": "neuro.rx.lean_in.felt_at",
        "p": {"site": {"k": "neuro.touch_site.head"}, "felt": {"k": "neuro.felt.stroked"}},
    }
    assert json.loads(json.dumps(wire)) == wire
    assert render(m, "en") == "stroked on the head, leans in"
    assert render(wire, "en") == "stroked on the head, leans in"
    assert render(wire, "zh-Hant") == "頭頂被摸，靠過去"
    assert render_key("page.app.toast.fast_forward", {"days": 1}, "en") == "Fast-forwarding 1 day…"
    assert render_key("page.app.toast.fast_forward", {"days": 3}, "en") == "Fast-forwarding 3 days…"


def test_lists_joins_and_data(cats: Catalogs) -> None:
    parts = lst([msg("neuro.touch_site.head"), "蛋黃"])
    assert parts == "頭頂、蛋黃"
    line = msg("neuro.body.line", parts=parts)
    assert line == "身體：頭頂、蛋黃"
    assert line.wire() == {
        "k": "neuro.body.line",
        "p": {"parts": {"list": [{"k": "neuro.touch_site.head"}, "蛋黃"], "sep": "fmt.sep.enum"}},
    }
    assert render(line, "en") == "Body: head, 蛋黃"  # the name is data: never translated
    glued = join(msg("neuro.touch_site.head"), "！")
    assert glued == "頭頂！" and to_wire(glued) == {"join": [{"k": "neuro.touch_site.head"}, "！"]}
    assert render(glued, "en") == "head！"


def test_select_and_pseudo(cats: Catalogs) -> None:
    assert msg("page.app.tv", on="yes") == "電視開著"
    assert msg("page.app.tv", on="no") == "電視關著"
    m = msg("neuro.rx.lean_in.felt_at", site=msg("neuro.touch_site.head"), felt="被摸")
    assert render(m, "pseudo") == "⟦頭頂⟧被摸⟦，靠過去⟧"  # data stays outside the brackets


def test_missing_translation_falls_back_to_the_source(cats: Catalogs) -> None:
    m = msg("page.app.tv", on="yes")
    assert render(m, "en") == "電視開著"  # no English yet: the source words
    with pytest.raises(MissingKey):
        render(m, "en", strict=True)
    ghost = msg("page.nowhere.key")
    assert ghost == "page.nowhere.key"  # a missing key never crashes the duck: its text is the key
    with pytest.raises(MissingKey):
        render(ghost.wire(), "zh-Hant", strict=True)


def test_msg_survives_copy_and_pickle(cats: Catalogs) -> None:
    m = msg("neuro.stub.clap", s=1.5)
    for other in (copy.copy(m), copy.deepcopy(m), pickle.loads(pickle.dumps(m))):
        assert other == m and isinstance(other, Msg)
        assert other.key == "neuro.stub.clap" and other.params == {"s": 1.5}


def test_catalog_file_round_trip(tmp_path: Path) -> None:
    messages = {"page.a.b": "一{x}二", "fmt.sep.enum": "、", "page.q.quote": '他說「"好"」\n'}
    text = dump_file("zh-Hant", "page", messages)
    assert parse_file(text) == ("zh-Hant", "page", messages)
    with pytest.raises(ValueError):
        write_catalog("zh-Hant", "core", messages, tmp_path)  # page keys in the core file
