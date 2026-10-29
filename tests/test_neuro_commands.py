"""Commands v1: the event schema and the six-language text parser (``duckmind.neuro.commands``).

Every phrase of the one table (``command_words.PHRASES``) is heard as its verb, alone, with the
duck's name, and with polite words around it; the other things a person says (a negation, a
sentence with no command, a word that only looks like one) are not commands."""

from __future__ import annotations

import pytest

from duckmind.neuro import command_words as W
from duckmind.neuro import commands as K

NAMES = {
    "zh-Hant": "蛋黃",
    "en": "Danhuang",
    "ja": "蛋黃",
    "es": "Danhuang",
    "zh-Hans": "蛋黄",
    "ko": "단황",
}
POLITE_BEFORE = {
    "zh-Hant": "請",
    "en": "please ",
    "ja": "",
    "es": "por favor ",
    "zh-Hans": "请",
    "ko": "",
}
POLITE_AFTER = {
    "zh-Hant": "啦",
    "en": " please",
    "ja": "ください",
    "es": " por favor",
    "zh-Hans": "啦",
    "ko": " 좀",
}


def _cases() -> list[tuple[str, str, str]]:
    out = []
    for verb, langs in W.PHRASES.items():
        for lang, phrases in langs.items():
            for phrase in phrases:
                out.append((verb, lang, phrase.lstrip("=")))
    return out


CASES = _cases()


def test_the_table_has_every_verb_in_every_language() -> None:
    assert set(W.PHRASES) == set(K.VERBS) - set(W.POINTED)
    for verb, langs in W.PHRASES.items():
        assert set(langs) == set(W.LANGS), verb
        for lang, phrases in langs.items():
            plain = [p for p in phrases if not p.startswith("=")]
            assert len(plain) >= 3, (verb, lang)  # (a few ways to say it, not one)
            assert len(set(phrases)) == len(phrases), (verb, lang)


@pytest.mark.parametrize("lang", W.LANGS)
def test_each_phrase_alone_is_its_verb(lang: str) -> None:
    for verb, plang, phrase in CASES:
        if plang != lang:
            continue
        heard = K.parse(phrase, lang=lang)
        assert heard.verb == verb, (lang, verb, phrase, heard)
        assert heard.confidence >= 0.9, (lang, verb, phrase, heard)


@pytest.mark.parametrize("lang", W.LANGS)
def test_each_phrase_with_the_ducks_name(lang: str) -> None:
    name = NAMES[lang]
    for verb, plang, phrase in CASES:
        if plang != lang:
            continue
        sep = " " if lang in ("en", "es", "ko") else ""
        for said in (f"{name}{sep}{phrase}", f"{name}, {phrase}", f"{phrase}, {name}!"):
            heard = K.parse(said, lang=lang, duck_name=name)
            assert heard.verb == verb, (lang, verb, said, heard)
            assert heard.named, (lang, said)


@pytest.mark.parametrize("lang", W.LANGS)
def test_each_phrase_politely(lang: str) -> None:
    before, after = POLITE_BEFORE[lang], POLITE_AFTER[lang]
    for verb, plang, phrase in CASES:
        if plang != lang:
            continue
        for said in (f"{before}{phrase}", f"{phrase}{after}", f"{before}{phrase}{after}"):
            heard = K.parse(said, lang=lang)
            assert heard.verb == verb, (lang, verb, said, heard)


@pytest.mark.parametrize(
    "said,lang,verb",
    [
        ("蛋黃,請你坐下", "zh-Hant", "sit"),
        ("蛋黃快過來這邊", "zh-Hant", "come"),
        ("蛋黃去睡午覺", "zh-Hant", "lie_down"),
        ("可以翻個跟斗嗎", "zh-Hant", "flip"),
        ("Danhuang, can you please sit down for me?", "en", "sit"),
        ("hey Danhuang come here!", "en", "come"),
        ("Danhuang, take a bow", "en", "bow"),
        ("ねえ蛋黃、こっちを見て", "ja", "look_at_me"),
        ("蛋黃、お家に帰ってね", "ja", "go_home"),
        ("ついておいで", "ja", "follow"),
        ("Danhuang, ¿puedes sentarte? sientate por favor", "es", "sit"),
        ("Danhuang, ven aquí", "es", "come"),
        ("¡Vuelta de campana, Danhuang!", "es", "flip"),
        ("蛋黄,请你站起来", "zh-Hans", "stand"),
        ("蛋黄,跟着我走", "zh-Hans", "follow"),
        ("단황아 이리 와", "ko", "come"),
        ("단황, 집에 가자", "ko", "go_home"),
        ("단황 앞구르기 해봐", "ko", "flip"),
    ],
)
def test_sentences(said: str, lang: str, verb: str) -> None:
    assert K.parse(said, lang=lang, duck_name=NAMES[lang]).verb == verb


@pytest.mark.parametrize(
    "said,lang",
    [
        ("不要坐下", "zh-Hant"),
        ("別過來", "zh-Hant"),
        ("不要翻跟斗", "zh-Hant"),
        ("别坐下", "zh-Hans"),
        ("don't sit", "en"),
        ("do not come here", "en"),
        ("never stop", "en"),
        ("no te sientes", "es"),
        ("nunca ven aqui", "es"),
        ("no ven aqui", "es"),
    ],
)
def test_a_negation_is_not_the_command(said: str, lang: str) -> None:
    assert K.parse(said, lang=lang).verb is None


@pytest.mark.parametrize(
    "said,lang",
    [
        ("hello there", "en"),
        ("what a nice day", "en"),
        ("vengo mañana", "es"),  # ('ven' only counts on word borders)
        ("el paraguas", "es"),  # ('para' only counts alone)
        ("sitting duck", "en"),  # ('sit' on a word border)
        ("今天天氣很好", "zh-Hant"),
        ("おはよう", "ja"),
        ("안녕하세요", "ko"),
        ("蛋黃", "zh-Hant"),
        ("", "en"),
        ("   !!!  ", "en"),
    ],
)
def test_not_a_command(said: str, lang: str) -> None:
    assert K.parse(said, lang=lang, duck_name="蛋黃").verb is None


def test_longest_phrase_wins() -> None:
    assert K.parse("eyes on me, come on over").verb == "come"  # (the longer phrase)
    assert K.parse("看過來").verb == "look_at_me"  # (not 過來 = come)
    assert K.parse("跟過來").verb == "follow"
    assert K.parse("ven conmigo").verb == "follow"  # (not ven = come)
    assert K.parse("ついておいで").verb == "follow"  # (not おいで = come)
    assert K.parse("stand still").verb == "stop"  # (not stand)
    assert K.parse("나를 따라와").verb == "follow"


def test_a_phrase_two_chinese_languages_share_takes_the_viewers_language() -> None:
    assert K.parse("坐下", lang="zh-Hans").lang == "zh-Hans"
    assert K.parse("坐下", lang="zh-Hant").lang == "zh-Hant"
    assert K.parse("坐下").lang == "zh-Hant"  # (the source language when nothing says)
    assert K.parse("坐下", lang="en").lang == "zh-Hant"
    assert K.parse("请坐", lang="zh-Hant").lang == "zh-Hans"  # (spelled only in simplified)


def test_confidence_is_lower_for_a_sentence_with_two_commands_than_for_one() -> None:
    one = K.parse("sit down")
    two = K.parse("sit down, then maybe later come here and look at me")
    assert one.confidence == 1.0
    assert two.confidence < one.confidence


# ── the event ────────────────────────────────────────────────────────────────


def test_a_button_command_is_the_schema() -> None:
    c = K.from_request({"verb": "sit", "who": "owner"})
    assert c.to_json() == {
        "verb": "sit",
        "who": "owner",
        "source": "button",
        "lang": "zh-Hant",
        "confidence": 1.0,
    }


def test_a_text_command_carries_what_was_said_and_its_language() -> None:
    c = K.from_request({"text": "Danhuang, come here!", "who": "child"}, duck_name="Danhuang")
    assert (c.verb, c.who, c.source, c.lang, c.named) == ("come", "child", "text", "en", True)
    assert c.text == "Danhuang, come here!"
    assert c.to_json()["named"] is True


def test_go_there_needs_a_place_and_only_it_takes_one() -> None:
    c = K.from_request({"verb": "go_there", "target": {"x_m": 1.0, "y_m": -0.5}})
    assert c.target == {"x_m": 1.0, "y_m": -0.5}
    assert c.to_json()["target"] == {"x_m": 1.0, "y_m": -0.5}
    for bad in (
        {"verb": "go_there"},
        {"verb": "go_there", "target": {"x_m": 1.0}},
        {"verb": "go_there", "target": {"x_m": float("nan"), "y_m": 0.0}},
        {"verb": "sit", "target": {"x_m": 1.0, "y_m": 0.0}},
    ):
        with pytest.raises(K.CommandError):
            K.from_request(bad)


@pytest.mark.parametrize(
    "params,reason",
    [
        ({}, "empty"),
        ({"text": "   "}, "empty"),
        ({"text": "blah blah"}, "not_understood"),
        ({"text": "don't sit"}, "negated"),
        ({"verb": "dance"}, "unknown_verb"),
        ({"verb": "sit", "source": "carrier pigeon"}, "bad_source"),
        ({"verb": "sit", "lang": "xx"}, "bad_lang"),
        ({"verb": "sit", "confidence": 2.0}, "bad_confidence"),
        ({"verb": "sit", "confidence": "high"}, "bad_confidence"),
    ],
)
def test_requests_that_cannot_be_made(params: dict, reason: str) -> None:
    with pytest.raises(K.CommandError) as error:
        K.from_request(params)
    assert error.value.reason == reason


def test_the_verb_lists() -> None:
    assert W.BUILTIN == ("sit", "stand", "come", "look_at_me", "stop", "follow", "go_home")
    assert W.TRICKS == ("lie_down", "spin", "bow", "shake", "flip")
    assert all(K.is_builtin(v) for v in W.BUILTIN) and K.is_builtin("go_there")
    assert not any(K.is_builtin(v) for v in W.TRICKS)
    assert all(K.is_trick(v) for v in W.TRICKS)
