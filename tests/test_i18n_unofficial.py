"""The page never claims to be official (owner decision Q5, 2026-10-29).

duckmind is not a Pollen Robotics product. Page words name Pollen's things as 「Pollen 的」 /
"Pollen's" (the model, the calls) and everything else from the original projects as 「上游」 /
"upstream"; the word 「官方」 / "official" (and its translations) appears only in the footer line
that says duckmind is unofficial, in all six languages.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

from duckmind.i18n.catalog import DOMAINS, STATIC, read_catalog
from duckmind.i18n.locales import CODES

DISCLAIMER = "page.html.footer.unofficial"

# "official" in each language; ko 공식으로 and ja 計算式 are "formula", not "official".
OFFICIAL = {
    "zh-Hant": re.compile("官方"),
    "zh-Hans": re.compile("官方"),
    "en": re.compile(r"\bofficial\b", re.I),
    "ja": re.compile("公式"),
    "ko": re.compile("공식(?!으로)"),
    "es": re.compile(r"\boficial(es)?\b", re.I),
}


def _messages(lang: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for domain in DOMAINS:
        found.update(read_catalog(lang, domain, STATIC) or {})
    return found


def test_no_page_words_say_official() -> None:
    claims = [
        f"{lang} {key}: {text}"
        for lang in CODES
        for key, text in _messages(lang).items()
        if key != DISCLAIMER and OFFICIAL[lang].search(text)
    ]
    assert claims == []


def test_the_footer_says_unofficial_in_every_language() -> None:
    for lang in CODES:
        text = _messages(lang).get(DISCLAIMER, "")
        assert "Pollen Robotics" in text, lang
        assert OFFICIAL[lang].search(text) or "unofficial" in text, lang

    class Footer(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.keys: list[str] = []
            self.inside = False

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            if tag == "footer":
                self.inside = True
            elif self.inside:
                self.keys.append(dict(attrs).get("data-i18n") or "")

        def handle_endtag(self, tag: str) -> None:
            if tag == "footer":
                self.inside = False

    footer = Footer()
    footer.feed((STATIC / "index.html").read_text(encoding="utf-8"))
    assert DISCLAIMER in footer.keys
