"""The command phrase a misheard sentence sounds most like.

A recogniser that hears a two-syllable command alone, with no sentence around it, often writes a word that
sounds the same or almost the same: 「按我」 for 「看我」, 「好怪」 for 「好乖」, 「行李」 for 「行禮」, ``Tómbate`` for
``túmbate``, 「すぐに戻って」 for 「巣に戻って」. When the words it wrote are not a command (and not a negated one),
:func:`nearest` compares how they SOUND with every phrase of the language's command and praise/scold tables
(:mod:`duckmind.neuro.command_words`) and gives the closest phrase when it is close enough and clearly closer than
any phrase of another meaning.

How a text sounds (:func:`sounds`), letters only, so an edit distance measures how far two sayings are:

* 中文: Hanyu Pinyin without tones (`pypinyin <https://github.com/mozillazg/python-pinyin>`_, MIT; optional, part of
  the voice packages: without it Chinese is compared by its characters);
* 한국어: each syllable split into its letters (jamo);
* 日本語: kana as rōmaji (katakana and hiragana alike); a kanji stays itself (no dictionary here: the tables also
  carry the kana spellings a recogniser may write);
* English, Español: the folded letters (accents dropped), spaces left out.

A phrase is compared with the whole sentence (the duck's name and the filler words left out); the distance is the
edit distance divided by the longer of the two. The thresholds (:data:`MAX_DISTANCE`, :data:`MARGIN`) were set on the
development set (``tools/voice_commands/tune.py``: every table phrase spoken once and sentences that are not for the
duck, never the test sentences). Matching a phrase inside a longer sentence was tried there too and dropped: it took
「我們晚上吃什麼」 as 跟上 and 「車子停在樓下」 as 停止 (up to half of a language's not-for-the-duck sentences).
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from functools import cache

from duckmind.neuro import command_words as W
from duckmind.neuro.commands import fold, prepare

MAX_DISTANCE = 0.52  # at most this share of the sounds may differ (dev set, 2026-11-05)
MARGIN = 0.1  # the closest phrase must beat every phrase of another meaning by this much

_KANA = {
    "あ": "a", "い": "i", "う": "u", "え": "e", "お": "o", "か": "ka", "き": "ki", "く": "ku", "け": "ke",
    "こ": "ko", "さ": "sa", "し": "shi", "す": "su", "せ": "se", "そ": "so", "た": "ta", "ち": "chi", "つ": "tsu",
    "て": "te", "と": "to", "な": "na", "に": "ni", "ぬ": "nu", "ね": "ne", "の": "no", "は": "ha", "ひ": "hi",
    "ふ": "fu", "へ": "he", "ほ": "ho", "ま": "ma", "み": "mi", "む": "mu", "め": "me", "も": "mo", "や": "ya",
    "ゆ": "yu", "よ": "yo", "ら": "ra", "り": "ri", "る": "ru", "れ": "re", "ろ": "ro", "わ": "wa", "を": "o",
    "ん": "n", "が": "ga", "ぎ": "gi", "ぐ": "gu", "げ": "ge", "ご": "go", "ざ": "za", "じ": "ji", "ず": "zu",
    "ぜ": "ze", "ぞ": "zo", "だ": "da", "ぢ": "ji", "づ": "zu", "で": "de", "ど": "do", "ば": "ba", "び": "bi",
    "ぶ": "bu", "べ": "be", "ぼ": "bo", "ぱ": "pa", "ぴ": "pi", "ぷ": "pu", "ぺ": "pe", "ぽ": "po", "ぁ": "a",
    "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o", "ゔ": "vu",
}  # fmt: skip
_SMALL_Y = {"ゃ": "ya", "ゅ": "yu", "ょ": "yo"}
_JAMO_L = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_JAMO_V = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_JAMO_T = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ",
           "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]  # fmt: skip


@cache
def _pinyin():  # type: ignore[no-untyped-def]
    try:
        from pypinyin import Style, lazy_pinyin
    except ImportError:
        return None
    return lambda text: lazy_pinyin(text, style=Style.NORMAL, errors="default")


def _romaji(text: str) -> str:
    out: list[str] = []
    double = False
    for ch in text:
        if "\u30a1" <= ch <= "\u30f6":
            ch = chr(ord(ch) - 0x60)  # katakana → hiragana
        if ch == "っ":
            double = True
            continue
        if ch == "ー":
            if out and out[-1]:
                out.append(out[-1][-1])
            continue
        if ch in _SMALL_Y and out:
            prev = out.pop()  # (き + ゃ = kya; し + ゃ = sha)
            stem = prev[:-1] if prev.endswith("i") else prev
            vowel = _SMALL_Y[ch]
            out.append(stem + (vowel[1:] if prev in ("shi", "chi", "ji") else vowel))
            continue
        syl = _KANA.get(ch)
        if syl is None:
            syl = ch if ch.isalnum() else ""
        if double and syl:
            syl = syl[0] + syl
        double = False
        out.append(syl)
    return "".join(out)


def _jamo(text: str) -> str:
    out: list[str] = []
    for ch in text:
        code = ord(ch) - 0xAC00
        if 0 <= code < 11172:
            out.append(_JAMO_L[code // 588] + _JAMO_V[(code % 588) // 28] + _JAMO_T[code % 28])
        elif ch.isalnum():
            out.append(ch)
    return "".join(out)


def sounds(text: str, lang: str) -> str:
    """How ``text`` sounds in ``lang``, as letters (see the module's notes)."""
    folded = fold(text)
    if lang in ("zh-Hant", "zh-Hans"):
        pinyin = _pinyin()
        if pinyin is not None:
            return "".join(p for p in pinyin(folded) if p.isalnum())
        return "".join(ch for ch in folded if ch.isalnum())
    if lang == "ko":
        return _jamo(folded)
    if lang == "ja":
        return _romaji(unicodedata.normalize("NFKC", folded))
    return "".join(ch for ch in folded if ch.isalnum())


def _edit(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@cache
def _phrases(lang: str) -> tuple[tuple[str, str, str], ...]:
    """(meaning, phrase, how it sounds) for every table phrase of ``lang``."""
    rows = []
    for table in (W.PHRASES, W.FEEDBACK_PHRASES):
        for meaning, langs in table.items():
            for raw in langs.get(lang, ()):
                phrase = raw.lstrip("=")
                key = sounds(phrase, lang)
                if key:
                    rows.append((meaning, phrase, key))
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class Near:
    """The closest table phrase: its meaning (a verb, ``praise`` or ``scold``), the phrase, and how far it is
    (0: sounds the same; the share of its sounds that differ)."""

    meaning: str
    phrase: str
    distance: float


def _negated(text: str, lang: str) -> bool:
    folded = fold(text)
    if lang in ("zh-Hant", "zh-Hans"):
        return any(n in folded for n in W.NEGATIONS_BEFORE["zh"])
    kind = {"en": "en", "es": "es"}.get(lang)
    if kind is None:
        return False
    words = f" {folded} "
    return any(f" {fold(n)} " in words for n in W.NEGATIONS_BEFORE[kind])


def nearest(text: str, lang: str, duck_name: str | None = None) -> Near | None:
    """The table phrase ``text`` sounds most like in ``lang``, or None when none is close enough, when two
    meanings are about as close, or when the sentence carries a "no" (a command it sounds like would be the
    opposite of what was said)."""
    if lang not in W.LANGS:
        return None
    _, bare, _ = prepare(text, duck_name)
    whole = sounds(bare, lang)
    if not whole:
        return None
    best: dict[str, tuple[float, str]] = {}
    for meaning, phrase, key in _phrases(lang):
        distance = _edit(key, whole) / max(len(key), len(whole))
        if meaning not in best or distance < best[meaning][0]:
            best[meaning] = (distance, phrase)
    ranked = sorted((d, meaning, phrase) for meaning, (d, phrase) in best.items())
    distance, meaning, phrase = ranked[0]
    if distance > MAX_DISTANCE:
        return None
    if len(ranked) > 1 and ranked[1][0] - distance < MARGIN:
        return None
    if meaning not in ("praise", "scold") and _negated(text, lang):
        return None
    return Near(meaning, phrase, round(distance, 3))
