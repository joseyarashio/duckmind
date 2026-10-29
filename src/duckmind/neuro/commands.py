"""Commands: what a person says or presses, as one event the duck's brain may hear.

One schema for every way in (the text box, the page's buttons, a click on the floor, the voice
recogniser of a later phase)::

    {"verb", "who", "source", "lang", "confidence", "target"?}

``verb`` is one of :data:`VERBS`; ``who`` a family role (or None); ``source`` ``text`` | ``button`` |
``voice``; ``lang`` the language the words were in (``zh-Hant`` for a button); ``confidence`` 0..1 (a
button is 1.0); ``target`` only for ``go_there``: a place in the DUCK'S OWN odometry frame, never the
world's (see :func:`duckmind.simweb.neuro_live.NeuroDriver.go_there` for the conversion).

A command is a sense input: what the duck heard. It is never a motor command. The brain
(:mod:`duckmind.neuro.command_brain`) decides from its own senses whether and how to answer it.

The parser reads the one phrase table in :mod:`duckmind.neuro.command_words`.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import cache
from typing import Any, Mapping

from duckmind.neuro import command_words as W

VERBS: tuple[str, ...] = (*W.BUILTIN, *W.TRICKS, *W.POINTED)
SOURCES: tuple[str, ...] = ("text", "button", "voice")
DEFAULT_LANG = "zh-Hant"
MAX_TEXT = 200


class CommandError(ValueError):
    """A command that cannot be made. ``reason`` is the catalog key suffix (``neuro.command.error.<reason>``)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class Command:
    verb: str
    who: str | None = None
    source: str = "button"
    lang: str = DEFAULT_LANG
    confidence: float = 1.0
    target: Mapping[str, float] | None = None
    # (data: what the person typed, shown as it is; and whether the duck's own name was said)
    text: str = ""
    named: bool = False

    def __post_init__(self) -> None:
        if self.verb not in VERBS:
            raise CommandError("unknown_verb")
        if self.source not in SOURCES:
            raise CommandError("bad_source")
        if self.lang not in W.LANGS:
            raise CommandError("bad_lang")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise CommandError("bad_confidence")
        if self.verb == "go_there":
            t = self.target
            if not t or not all(_finite(t.get(k)) for k in ("x_m", "y_m")):
                raise CommandError("no_target")
        elif self.target is not None:
            raise CommandError("target_only_for_go_there")

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "verb": self.verb,
            "who": self.who,
            "source": self.source,
            "lang": self.lang,
            "confidence": round(float(self.confidence), 3),
        }
        if self.target is not None:
            out["target"] = {k: round(float(v), 3) for k, v in self.target.items()}
        if self.text:
            out["text"] = self.text
        if self.named:
            out["named"] = True
        return out


def _finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and abs(number) != float("inf")


# ── reading the words ─────────────────────────────────────────────────────────

_PUNCT = re.compile(r"[\s\u3000.,;:!?¡¿、。，．！？；：…~～\-—()\[\]{}\"“”「」『』《》〈〉]+")


def fold(text: str) -> str:
    """The text the parser compares: NFKC, lower case, Latin accents dropped, punctuation and runs of
    space made one space (an apostrophe stays: "don't"). Accents go from Latin letters only, never
    from Hangul or kana."""
    out: list[str] = []
    for ch in unicodedata.normalize("NFKC", text).lower():
        if ch in "\u2018\u2019":
            ch = "'"
        elif ord(ch) < 0x250 and ch.isalpha():
            ch = "".join(c for c in unicodedata.normalize("NFD", ch) if not unicodedata.combining(c))
        out.append(ch)
    return _PUNCT.sub(" ", "".join(out)).strip()


def _latin(phrase: str) -> bool:
    return all(c.isascii() and (c.isalnum() or c in " '") for c in phrase)


@cache
def _table() -> tuple[tuple[str, str, str, bool], ...]:
    """(verb, language, folded phrase, whole-utterance only) for every phrase, longest first."""
    rows: list[tuple[str, str, str, bool]] = []
    for verb, langs in W.PHRASES.items():
        for lang, phrases in langs.items():
            for raw in phrases:
                whole = raw.startswith("=")
                phrase = fold(raw[1:] if whole else raw)
                if phrase:
                    rows.append((verb, lang, phrase, whole))
    rows.sort(key=lambda row: -len(row[2]))
    return tuple(rows)


@cache
def _fillers() -> tuple[str, ...]:
    return tuple(sorted({fold(f) for f in W.FILLERS if fold(f)}, key=len, reverse=True))


@cache
def _borders(phrase: str) -> re.Pattern[str]:
    return re.compile(r"(?<![a-z0-9'])" + re.escape(phrase) + r"(?![a-z0-9'])")


@cache
def _negation(kind: str) -> tuple[str, ...]:
    return tuple(fold(n) for n in W.NEGATIONS_BEFORE[kind])


def _negated(text: str, start: int, lang: str) -> bool:
    """A "no" just before the phrase: the sentence is not that command."""
    before = text[:start].rstrip()
    if not before:
        return False
    if lang.startswith("zh"):
        return any(before.endswith(n) for n in _negation("zh"))
    kind = {"en": "en", "es": "es"}.get(lang)
    if kind is None:
        return False  # (Japanese and Korean put the "no" after the verb: the phrase is not found)
    words = before.split()
    tail = " ".join(words[-2:])
    return any(
        tail == n or tail.endswith(" " + n) or (words and words[-1] == n)
        for n in _negation(kind)
    )


@dataclass(frozen=True, slots=True)
class Heard:
    """What the parser made of a sentence: a verb (None: no command in it), its language, how sure."""

    verb: str | None
    lang: str
    confidence: float
    named: bool = False
    negated: bool = False


def _remove(text: str, word: str) -> str:
    return re.sub(r"\s+", " ", text.replace(word, " ")).strip() if word else text


def parse(text: str, *, lang: str | None = None, duck_name: str | None = None) -> Heard:
    """The verb in ``text`` (any of the six languages, with or without the duck's name, with
    politeness words around it), or none. ``lang``: the viewer's own language, used only to decide a
    phrase two languages share (坐下 is both zh-Hant and zh-Hans). ``duck_name`` is data."""
    hint = lang if lang in W.LANGS else None
    body = fold(str(text)[:MAX_TEXT])
    name = fold(duck_name) if duck_name else ""
    named = bool(name) and name in body
    if named:
        body = _remove(body, name)
    if not body:
        return Heard(None, hint or DEFAULT_LANG, 0.0, named)
    bare = f" {body} "
    for filler in _fillers():
        bare = bare.replace(f" {filler} ", " ") if _latin(filler) else bare.replace(filler, " ")
        bare = re.sub(r"\s+", " ", bare)
    bare = bare.strip()
    best: tuple[int, int, int, str, str, str] | None = None  # (length, -position, hint, verb, lang, phrase)
    rival = False
    negated = False
    for verb, plang, phrase, whole in _table():
        if whole:
            if bare != phrase:
                continue
            start = 0
        else:
            if _latin(phrase):
                found = _borders(phrase).search(body)
                start = found.start() if found else -1
            else:
                start = body.find(phrase)
            if start < 0:
                continue
        if _negated(body, start, plang):
            negated = True
            continue
        key = (len(phrase), -start, 1 if plang == hint else 0)
        if best is None:
            best = (*key, verb, plang, phrase)
            continue
        if key > best[:3]:
            if verb != best[3]:
                rival = True
            best = (*key, verb, plang, phrase)
        elif verb != best[3] and key[0] >= best[0] - 1 and plang != best[4]:
            rival = rival or key[0] == best[0]
    if best is None:
        return Heard(None, hint or DEFAULT_LANG, 0.0, named, negated)
    length, _, _, verb, plang, phrase = best
    spoken = max(1, len(bare))
    coverage = min(1.0, length / spoken)
    confidence = 0.55 + 0.45 * coverage
    if rival:
        confidence = min(confidence, 0.6)
    if plang == "zh-Hans" and hint != "zh-Hans" and _both_chinese(verb, phrase):
        plang = hint if hint in ("zh-Hant", "zh-Hans") else "zh-Hant"
    return Heard(verb, plang, round(confidence, 3), named)


def _both_chinese(verb: str, phrase: str) -> bool:
    """The matched phrase is spelled the same in zh-Hant and zh-Hans (坐下)."""
    langs = W.PHRASES[verb]
    return any(fold(p.lstrip("=")) == phrase for p in langs["zh-Hant"])


def from_request(
    params: Mapping[str, Any], *, duck_name: str | None = None, default_who: str | None = None
) -> Command:
    """The command a request names: ``{"verb": …}`` (a button; ``source`` defaults to ``button``) or
    ``{"text": …}`` (parsed; ``source`` ``text``), with ``who``, ``lang``, ``target`` as in the
    schema. Raises :class:`CommandError` with the reason the duck cannot be told."""
    who = params.get("who") or default_who
    who = str(who) if who else None
    lang = str(params.get("lang") or DEFAULT_LANG)
    if lang not in W.LANGS:
        raise CommandError("bad_lang")
    text = params.get("text")
    verb = params.get("verb")
    if verb is not None:
        source = str(params.get("source") or "button")
        target = params.get("target")
        if target is not None and not isinstance(target, Mapping):
            raise CommandError("no_target")
        try:
            confidence = float(params.get("confidence", 1.0))
        except (TypeError, ValueError):
            raise CommandError("bad_confidence") from None
        return Command(
            str(verb), who, source, lang, confidence, target, str(text or "")[:MAX_TEXT]
        )
    if not isinstance(text, str) or not text.strip():
        raise CommandError("empty")
    heard = parse(text, lang=lang, duck_name=duck_name)
    if heard.verb is None:
        raise CommandError("negated" if heard.negated else "not_understood")
    return Command(
        heard.verb,
        who,
        str(params.get("source") or "text"),
        heard.lang,
        heard.confidence,
        None,
        text.strip()[:MAX_TEXT],
        heard.named,
    )


def is_builtin(verb: str) -> bool:
    return verb in W.BUILTIN or verb == "go_there"


def is_trick(verb: str) -> bool:
    return verb in W.TRICKS
