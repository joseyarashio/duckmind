"""The six page languages (owner, 2026-09-28) and how a viewer's language is chosen.

繁體中文 is the default and the source; English, 日本語, Español, 简体中文 and 한국어 are
translations. ``static/i18n.js`` holds the same table and the same negotiation rule (a test
compares them), so the page and the server always agree on what a language tag means.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

SOURCE = "zh-Hant"
# A test-only language: every catalog text wrapped in ⟦ ⟧, so any words the page shows
# outside the brackets did not come from a catalog (a hard-coded leftover).
PSEUDO = "pseudo"
PSEUDO_OPEN = "⟦"
PSEUDO_CLOSE = "⟧"


@dataclass(frozen=True)
class Lang:
    code: str  # BCP 47 tag: catalog file names, <html lang>, ?lang=
    native: str  # its own name (the language picker shows these)
    english: str
    script: str  # ISO 15924
    font: str | None  # the Noto family for this script's display and body text (None: Latin)
    decimal: str  # decimal mark
    group: str  # grouping mark (numbers never group unless a message asks for it)
    min_grouping: int  # CLDR minimumGroupingDigits (es: 2 → 1234 stays, 12.345 groups)


LANGS: tuple[Lang, ...] = (
    Lang("zh-Hant", "繁體中文", "Traditional Chinese", "Hant", "Noto Sans TC", ".", ",", 1),
    Lang("en", "English", "English", "Latn", None, ".", ",", 1),
    Lang("ja", "日本語", "Japanese", "Jpan", "Noto Sans JP", ".", ",", 1),
    Lang("es", "Español", "Spanish", "Latn", None, ",", ".", 2),
    Lang("zh-Hans", "简体中文", "Simplified Chinese", "Hans", "Noto Sans SC", ".", ",", 1),
    Lang("ko", "한국어", "Korean", "Kore", "Noto Sans KR", ".", ",", 1),
)
CODES: tuple[str, ...] = tuple(lang.code for lang in LANGS)
BY_CODE: dict[str, Lang] = {lang.code: lang for lang in LANGS}

# Tags people's browsers send that name one of the six by another spelling.
_HANT_REGIONS = {"tw", "hk", "mo"}
_HANS_REGIONS = {"cn", "sg", "my"}


def lang_info(code: str) -> Lang:
    """The table row of a language (the pseudo-locale formats like the source)."""
    return BY_CODE[SOURCE if code == PSEUDO else code]


def normalize(tag: str | None) -> str | None:
    """One of the six codes for a language tag, or None.

    ``zh-TW`` / ``zh-HK`` / ``zh-Hant-*`` → zh-Hant; ``zh-CN`` / ``zh-SG`` / ``zh-Hans-*`` →
    zh-Hans; a bare ``zh`` → zh-Hant (the source: the family's own language); ``en-US`` → en,
    ``es-419`` → es, ``ja-JP`` → ja, ``ko-KR`` → ko. Case and ``_`` vs ``-`` do not matter.
    """
    if not tag:
        return None
    parts = [p for p in re.split(r"[-_]", str(tag).strip().lower()) if p]
    if not parts:
        return None
    if parts[0] == PSEUDO:
        return PSEUDO
    base = parts[0]
    rest = parts[1:]
    if base == "zh":
        if "hant" in rest or any(p in _HANT_REGIONS for p in rest):
            return "zh-Hant"
        if "hans" in rest or any(p in _HANS_REGIONS for p in rest):
            return "zh-Hans"
        return "zh-Hant"
    for code in CODES:
        if code == base:
            return code
    return None


def negotiate(preferred: Iterable[str | None]) -> str:
    """The best of the six for a browser's ``navigator.languages`` (first match wins), else
    the source language. The pseudo-locale is never chosen this way (only by ``?lang=``)."""
    for tag in preferred:
        code = normalize(tag)
        if code is not None and code != PSEUDO:
            return code
    return SOURCE


def is_cjk_lang(code: str) -> bool:
    return lang_info(code).script in {"Hant", "Hans", "Jpan", "Kore"}
