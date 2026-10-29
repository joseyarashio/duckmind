"""Live voice, LV1: the duck's soul for the cloud brain (``duckmind.live.soul``): the system
instruction and the status text, built from the genome summary, the learner's view and the drives."""

from __future__ import annotations

import pytest

from duckmind.live import keys as K
from duckmind.live import soul as S
from duckmind.live import tools as T
from duckmind.neuro.drives import Drives
from duckmind.neuro.genome import PRESETS, Genome

pytestmark = pytest.mark.usefixtures("no_network")


def _inputs(preset: str = "explorer", **extra: object) -> S.SoulInputs:
    seed, gd = PRESETS[preset]
    summary = Genome.from_seed(seed, gd=gd).summary()
    return S.SoulInputs(
        name="蛋黃",
        genome={
            k: v if isinstance(v, (int, float, list, dict, type(None))) else str(v)
            for k, v in summary.items()
        },
        drives=Drives().as_dict(),
        **extra,  # type: ignore[arg-type]
    )


def test_the_soul_tells_the_model_what_it_is_and_what_it_may_not_do() -> None:
    text = S.build_soul(_inputs())
    assert "蛋黃" in text and "腦" in text
    # the duck never speaks human words; its mouth is the voice bank
    assert "不會說人話" in text and "鴨叫聲庫" in text
    for tag in T.BANK_TAGS:
        assert tag in text
    # it only proposes: every tool is named, the body is not its to move
    for tool in T.TOOL_NAMES:
        assert tool in text
    assert "不能直接控制" in text
    # its own calls come back through the microphone
    assert "回音" in text
    # no inventing
    assert "不要編造" in text


def test_the_ears_note_means_the_name_was_called_whatever_the_transcript_shows() -> None:
    """soul-2: the duck's own ear is reliable for its own name, the transcript of a name is not."""
    for preset in PRESETS:
        text = S.build_soul(_inputs(preset))
        # the exact words the bridge sends are in the instruction (they cannot drift apart)
        assert S.EAR_NAME_HINT in text
        lines = [line for line in text.splitlines() if S.EAR_NAME_HINT in line]
        assert len(lines) == 3  # the rule that answers it, the rule that has no note, the tail
        answer = next(line for line in lines if "有人在叫蛋黃" in line)
        # it is the duck's name being called, whatever name the transcript shows; answer as a pet
        # would, with a short call, and only by the tools
        assert "不管你聽到的文字顯示的是什麼名字" in answer and "vocalize" in answer
        assert "greet" in answer and "chirp" in answer and "inquire" in answer
        assert "早到或晚到" in answer  # (the note may come before or after the transcript)
        # the transcript of a name is known to be wrong (the model is told so)
        assert "常被聽成別的名字" in text and "比你聽到的文字可靠" in text
        assert "蛋黃" in answer


def test_without_the_ears_note_a_clearly_different_name_still_gets_no_greeting() -> None:
    text = S.build_soul(_inputs())
    other = next(line for line in text.splitlines() if line.startswith("- 沒有收到"))
    assert S.EAR_NAME_HINT in other and "另一個名字" in other
    assert "不是在叫蛋黃" in other and "不要 vocalize" in other and "最多 attend（sound）" in other
    # no note, no change to the rest: the rules that keep the duck from speaking human words, from
    # answering its own echo and from inventing stay in the text as they were
    for kept in ("不會說人話", "回音", "不要編造", "只能呼叫工具", "只能提議"):
        assert kept in text
    # the tail keeps the status text a notice that needs no answer, with the one exception
    assert "那只是通知，不需要回應" in text and text.rstrip().endswith("要照那裡說的處理。")


def test_the_soul_is_text_in_traditional_chinese_and_holds_no_key() -> None:
    text = S.build_soul(_inputs())
    assert "麼" in text or "個" in text  # (traditional forms)
    assert "这" not in text and "们" not in text  # (no simplified forms)
    assert not K.find_keys(text)


def test_equal_inputs_equal_text_and_other_ducks_other_text() -> None:
    a, b = S.build_soul(_inputs()), S.build_soul(_inputs())
    assert a == b
    texts = {preset: S.build_soul(_inputs(preset)) for preset in PRESETS}
    assert len(set(texts.values())) == len(PRESETS)


@pytest.mark.parametrize(
    "preset, expect",
    [
        ("explorer", "好奇型"),
        ("snuggler", "依偎型"),
        ("dreamer", "悠哉型"),
        ("hatchling", "愛唱歌型"),
    ],
)
def test_each_temperament_is_in_the_words(preset: str, expect: str) -> None:
    text = S.build_soul(_inputs(preset))
    assert expect in text
    assert "天性是" in text and "最常用的叫法" in text


def test_the_shy_duck_is_said_to_be_shy_and_the_bold_one_bold() -> None:
    shy = S.build_soul(_inputs("snuggler"))
    bold = S.build_soul(_inputs("explorer"))
    assert "膽子小" in shy and "膽子大" in bold


def test_drives_are_plain_words_and_zero_to_ten_in_a_fixed_order() -> None:
    line = S.drives_line({"fear": 0.0, "social": 0.64, "curiosity": 0.45, "nonsense": 9.0})
    assert line == "想有人陪 6、好奇 4、害怕 0"
    assert S.drives_line({}) == ""
    assert S.drives_line({"social": 7.0}) == "想有人陪 10"  # (clamped)


def test_the_learners_view_and_known_people_are_summarised() -> None:
    view = {
        "places": [{"value": 0.5}, {"value": 0.0}, {"value": 0.3}],
        "things": [{"kind": "ball"}, {"kind": "ball"}, {"kind": "cup"}],
    }
    text = S.build_soul(_inputs(learner=view, known_people=("媽媽", "爺爺"), behaviour="睡覺"))
    assert "3 個地方，其中 2 個牠喜歡" in text
    assert "ball、cup" in text
    assert "牠已經認得的人：媽媽、爺爺" in text
    assert "剛開始正在：睡覺" in text
    assert "還沒有認得的人" in S.build_soul(_inputs())


def test_the_status_text_is_a_notice_and_changes_only_with_the_state() -> None:
    quiet = S.ContextSnapshot(behaviour="坐著看四周", valence=0.15, drives={"social": 0.4})
    assert (
        quiet.text()
        == S.ContextSnapshot(behaviour="坐著看四周", valence=0.15, drives={"social": 0.4}).text()
    )
    text = quiet.text()
    assert "不需要回應" in text
    assert "正在做：坐著看四周" in text and "心情：心情不錯" in text and "想有人陪 4" in text
    assert "認得的人：還沒有" in text and "眼睛現在看到：沒有特別的東西" in text
    busy = S.ContextSnapshot(
        behaviour="走向窗邊",
        known_people=("媽媽",),
        in_view=("玩具", "一個人"),
        notes=("電池 60%",),
    )
    assert "認得的人：媽媽" in busy.text() and "眼睛現在看到：玩具、一個人" in busy.text()
    assert "另外：電池 60%" in busy.text()
    assert busy.text() != text


@pytest.mark.parametrize(
    "valence, word",
    [
        (0.9, "很開心"),
        (0.2, "心情不錯"),
        (0.0, "心情普通"),
        (-0.3, "有點不舒服"),
        (-0.9, "很不舒服"),
    ],
)
def test_the_mood_words(valence: float, word: str) -> None:
    assert word in S.ContextSnapshot(valence=valence).text()


def test_the_version_is_a_label_the_cassette_key_carries() -> None:
    assert S.SOUL_VERSION.startswith("soul-")
    assert S.SOUL_VERSION == "soul-2"  # (the instruction changed with the ear note: soul-1 is gone)
