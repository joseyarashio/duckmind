"""i18n: no page words outside the catalogs.

Every text a viewer can see in static/* (HTML text and attributes, script strings, CSS
content) must come from a catalog key; the only exceptions are data and names
(src/duckmind/i18n/allowlist.json). Static HTML keeps its 繁體中文 words for the first paint,
marked with data-i18n / data-i18n-attr, and those words must equal the catalog's.

Until a tree has the i18n system (index.html loads i18n.js), the check reports how many texts
are still hard-coded and skips; from then on it is strict (CLAUDE.md: every page change ships
all six languages).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from duckmind.i18n import checks
from duckmind.i18n.catalog import write_catalog


def test_the_page_has_no_hard_coded_words() -> None:
    found = checks.scan_static()
    if not checks.page_uses_i18n():
        pytest.skip(
            f"i18n not integrated in this tree yet: {len(found)} page texts still hard-coded"
        )
    assert [str(f) for f in found[:40]] == []


def test_scanner_finds_words_in_scripts_but_not_in_comments(tmp_path: Path) -> None:
    (tmp_path / "a.js").write_text(
        "// 註解不算\n"
        "/* 這也不算\n 多行 */\n"
        'const a = "按鈕";\n'
        "const b = `快轉 ${days} 天中……`;\n"
        "const c = x / 2; const d = /沒電/.test(t);\n"
        'const e = `外 ${ok ? `內層` : ""} 尾`;\n'
        "const f = '蛋黃';\n"
        'I18N.t("page.x.y");\n',
        encoding="utf-8",
    )
    found = checks.scan_static(tmp_path, source={})
    got = {(f.line, f.kind, f.text) for f in found}
    assert got == {
        (4, "string", "按鈕"),
        (5, "template", "快轉 "),
        (5, "template", " 天中……"),
        (6, "regex", "沒電"),
        (7, "template", "內層"),
        (7, "template", "外 "),
        (7, "template", " 尾"),
    }


def test_scanner_accepts_marked_html_with_the_catalogs_words(tmp_path: Path) -> None:
    write_catalog("zh-Hant", "page", {"page.t.hello": "你好", "page.t.tip": "說明"}, tmp_path)
    (tmp_path / "index.html").write_text(
        '<!doctype html><html><head><title data-i18n="page.t.hello">你好</title></head><body>\n'
        "<!-- 註解 -->\n"
        '<p data-i18n="page.t.hello">你好</p>\n'
        '<button title="說明" data-i18n-attr="title:page.t.tip">OK</button>\n'
        '<p data-i18n="page.t.hello">不一樣</p>\n'
        "<p>沒有標記</p>\n"
        '<img alt="沒有標記的圖">\n'
        "<p>蛋黃</p>\n"
        "<script>const x = 1;</script>\n"
        "</body></html>\n",
        encoding="utf-8",
    )
    found = checks.scan_static(tmp_path)
    got = sorted((f.line, f.kind) for f in found)
    assert got == [(5, "html-text"), (6, "html-text"), (7, "html-attr")]


def test_scanner_reads_css_content(tmp_path: Path) -> None:
    (tmp_path / "s.css").write_text(
        '/* 註解 */\n.x::after { content: "（示範）"; }\n.y { font-family: "Noto Sans TC"; }\n',
        encoding="utf-8",
    )
    found = checks.scan_static(tmp_path, source={})
    assert [(f.line, f.kind, f.text) for f in found] == [(2, "css", "（示範）")]


def test_catalog_files_themselves_are_not_scanned(tmp_path: Path) -> None:
    write_catalog("zh-Hant", "page", {"page.t.hello": "你好"}, tmp_path)
    (tmp_path / "i18n.manifest.js").write_text("DuckI18n.manifest({});\n", encoding="utf-8")
    assert checks.scan_static(tmp_path, source={}) == []


def test_integration_is_detected_by_the_script_tag(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text('<script src="/app.js"></script>', encoding="utf-8")
    assert not checks.page_uses_i18n(tmp_path)
    (tmp_path / "index.html").write_text(
        '<script src="/i18n.js"></script><script src="/app.js"></script>', encoding="utf-8"
    )
    assert checks.page_uses_i18n(tmp_path)
