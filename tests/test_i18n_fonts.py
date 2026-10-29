"""The page's fonts come from static/fonts, never from a CDN (owner decision Q7, 2026-10-29).

Noto Sans TC (the source language) and the Latin fonts are Google's files, unmodified. Noto Sans
JP/SC/KR are subsets holding only the glyphs their catalogs use: a new word in one of those
languages needs `uv run --with fonttools --with brotli python -m duckmind.i18n.fonts subset`, or
the page would draw it in another font.
"""

from __future__ import annotations

import json
import re

from duckmind.i18n import fonts
from duckmind.i18n.catalog import STATIC
from duckmind.i18n.locales import LANGS, SOURCE


def _urls(css: str) -> list[str]:
    return re.findall(r"url\(([^)]+)\)", css)


def test_every_font_stylesheet_points_at_files_in_the_fonts_folder() -> None:
    sheets = ["fonts.css"] + [fonts.css_file(lang.font) for lang in LANGS if lang.font]
    for sheet in dict.fromkeys(sheets):
        if sheet == fonts.css_file("Noto Sans TC"):
            continue  # the source language's font is in fonts.css with the others
        css = (fonts.FONTS / sheet).read_text(encoding="utf-8")
        urls = _urls(css)
        assert urls, sheet
        for url in urls:
            assert url.startswith("./") and url.endswith(".woff2"), (sheet, url)
            assert (fonts.FONTS / url[2:]).is_file(), (sheet, url)
    assert "font-family: 'Noto Sans TC'" in (fonts.FONTS / "fonts.css").read_text(encoding="utf-8")


def test_each_subset_holds_every_character_its_catalogs_use() -> None:
    coverage = json.loads(fonts.COVERAGE.read_text(encoding="utf-8"))
    for lang in LANGS:
        if not lang.font or lang.code == SOURCE:
            continue
        held = set(coverage[lang.code]["chars"])
        missing = sorted(set(fonts.wanted_chars(lang.code, STATIC)) - held)
        assert missing == [], (
            f"{lang.code}: run `python -m duckmind.i18n.fonts subset` ({''.join(missing[:40])})"
        )
        assert (fonts.FONTS / coverage[lang.code]["file"]).is_file()


def test_every_font_has_its_licence() -> None:
    for licence in fonts.LICENCES:
        text = (fonts.FONTS / licence).read_text(encoding="utf-8")
        assert "SIL Open Font License, Version 1.1" in text, licence
    assert "Reserved Font Name 'Source'" in (fonts.FONTS / "OFL-NotoSansCJK.txt").read_text(
        encoding="utf-8"
    )
