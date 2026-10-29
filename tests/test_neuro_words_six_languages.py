"""The neural duck's cards and day row speak every language the page speaks.

Every word the hub sends for a duck (its stage, nature, branch, side, habit, temperament, its one
plain line, the picker's titles) and for its own clock (the day choices, the time of day) is a
catalog message: rendered in English, Spanish and Korean no Chinese character is left but the duck's name, and the
繁體中文 text itself is the same words as before.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from duckmind.i18n import boundary, render
from duckmind.simweb import neuro_api as N
from duckmind.simweb import neuro_live as L

CJK = re.compile(r"[　-〿㐀-鿿＀-￯]")
LANGS = ("en", "es", "ko")  # (ja and zh-Hans are written in Han characters themselves)
DATA_FIELDS = {"name"}  # a duck's own name is data, never translated


def _words(obj: Any, lang: str, path: str = "") -> list[tuple[str, str]]:
    """Every shown word of an answer in ``lang``: (path, text). A field with a message is
    rendered from it; one without is shown as it is (data)."""
    out: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        said = obj.get("_i18n", {})
        for key, value in obj.items():
            if key == "_i18n":
                continue
            where = f"{path}.{key}"
            if key in said and said[key] is not None:
                shown = said[key]
                if isinstance(shown, list):
                    out += [(where, render(item, lang)) for item in shown if item is not None]
                else:
                    out.append((where, render(shown, lang)))
            elif isinstance(value, str):
                out.append((where, value))
            else:
                out += _words(value, lang, where)
    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            out += _words(item, lang, f"{path}[{index}]")
    return out


def _left_over(answer: Any, lang: str) -> list[tuple[str, str]]:
    shown = _words(boundary.attach(answer), lang)
    return [(p, t) for p, t in shown if CJK.search(t) and p.rsplit(".", 1)[-1] not in DATA_FIELDS]


CARDS = [N.duck_card(item) for item in N.PRESETS] + [
    N.card_of(N.genome_summary(seed, gd)) for seed, gd in ((7, 60), (123456, 0), (99, 15), (5, 400))
]


@pytest.mark.parametrize("lang", LANGS)
def test_a_ducks_card_has_no_繁體中文_left_but_its_name(lang: str) -> None:
    for card in CARDS:
        assert _left_over(card, lang) == [], (card["name"], lang)


@pytest.mark.parametrize("lang", LANGS)
def test_the_day_row_and_the_clock_have_no_繁體中文_left(lang: str) -> None:
    row = [{"zh": zh, "minute": minute} for zh, minute in N.DAY_CHOICES]
    clock = [{"minute": m, "zh": L.minute_zh(m), "rate": 1.0} for m in range(0, 1440, 35)]
    assert _left_over({"day_choices": row, "day": clock}, lang) == []


def test_the_whole_status_has_no_繁體中文_left_but_names() -> None:
    hub = N.NeuroHub(enabled=True, source=N.StubSource(N.duck_card(N.PRESETS[0])))
    status = hub.status()
    for lang in ("en", "es"):
        left = [(p, t) for p, t in _left_over(status, lang)]
        assert left == [], lang


def test_the_words_in_繁體中文_are_the_ones_they_always_were() -> None:
    card = N.duck_card(N.PRESET_BY_ID["snuggler"])
    assert (card["stage_zh"], card["nature_zh"], card["branch_zh"]) == ("幼鴨", "依偎型", "")
    assert (card["side_zh"], card["habit_zh"]) == ("右", "理毛前會先歪一下頭")
    assert card["line_zh"] == (
        "很膽小、不太愛探險、很黏人、不太愛動的幼鴨（天性：依偎型）；歪頭習慣往右邊"
    )
    brave = N.duck_card(N.PRESET_BY_ID["explorer"])
    assert brave["branch_zh"] == "好奇型" and brave["stage_zh"] == "少年鴨"
    assert brave["line_zh"] == "好奇心很強、膽子大、不太黏人、很好動的好奇型少年鴨；歪頭習慣往左邊"
    assert L.minute_zh(600) == "上午 10:00" and L.minute_zh(21 * 60 + 30) == "晚上 9:30"
    assert [zh for zh, _ in N.DAY_CHOICES] == ["早上", "上午", "下午", "傍晚", "晚上", "半夜"]


def test_the_other_languages_say_something_sensible() -> None:
    card = N.duck_card(N.PRESET_BY_ID["explorer"])
    wire = boundary.attach(card)["_i18n"]
    assert render(wire["stage_zh"], "en") == "Juvenile"
    assert render(wire["branch_zh"], "en") == "Explorer"
    assert render(wire["line_zh"], "en").endswith("tilts its head to the left")
    assert (
        render(boundary.attach({"zh": L.minute_zh(600)})["_i18n"]["zh"], "en")
        == "Late morning 10:00"
    )


# ── the words only a live body makes (the real brain, the world, the body's witness) ──────────


def test_the_real_brains_page_speaks_every_language() -> None:
    """What the page shows from the driver's telemetry (the behaviour and why, the candidates,
    the reflex rows, the calls, the drives, the places, the senses) has no 繁體中文 left."""
    from test_neuro_web import _Driver

    driver = _Driver(N.PRESET_BY_ID["snuggler"])
    source = N.DriverSource(lambda: driver, clock=iter(range(0, 10_000)).__next__)
    driver.run(0, 3000)
    driver.run(3000, 9000, clap_at_ms=3500)
    snapshot = source.snapshot(0)
    assert snapshot["reflexes"] and snapshot["behaviour"]["why_zh"]
    for lang in LANGS:
        assert _left_over(snapshot, lang) == [], lang


def test_words_that_crossed_the_worlds_socket_keep_their_message() -> None:
    from duckmind.i18n import carried_words, from_wire, msg, to_wire

    said = msg("neuro.world.pet_head.e252", strokes=2)
    crossed = {"note_zh": str(said), "_i18n": {"note_zh": to_wire(said)}}
    again = carried_words(crossed, "note_zh")
    assert again == said and render(again, "en").startswith("A hand pets its head 2 times")
    row = msg("neuro.row.world", stimulus=again)
    assert render(row, "es").startswith("Mundo: ") and not CJK.search(render(row, "es"))
    # a message that no longer says these words is not trusted
    assert (
        carried_words({"note_zh": "別的話", "_i18n": {"note_zh": to_wire(said)}}, "note_zh")
        == "別的話"
    )
    assert carried_words({"note_zh": "沒有訊息"}, "note_zh") == "沒有訊息"
    assert from_wire(to_wire(said)) == said


def test_the_bodys_motion_names_are_messages_and_the_same_words() -> None:
    from duckmind.simweb import neuro_witness as W

    for dof in ("pitch", "yaw", "roll"):
        for value in (18.0, -18.0):
            for neck in (0.0, 5.0):
                name = W._name(dof, value, neck)
                assert str(name) == str(W.head_words(dof, value, neck)).split(" ")[0]
                for lang in ("en", "es", "ko"):
                    assert not CJK.search(render(name, lang))
                    assert not CJK.search(render(W.head_words(dof, value, neck), lang))
