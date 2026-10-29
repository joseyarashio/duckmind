"""Live voice, LV1: the rule-based fake Gemini (``duckmind.live.fake``): the same interface as
the real connection, deterministic, and only as clever as a few plain rules."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest
from live_rig import LOUD, QUIET, Rig

from duckmind.live import events as E
from duckmind.live.fake import FakeGemini, SpokenLine

pytestmark = pytest.mark.usefixtures("no_network")


def say(
    text: str,
    *,
    labels: dict[str, list[str]] | None = None,
    image: bytes | None = None,
    name: str = "蛋黃",
) -> list[tuple[str, dict[str, Any]]]:
    """The calls the fake makes for ``text`` said from 1.0 to 2.0 s: (tool, args)."""
    fake = FakeGemini(duck_name=name, lines=[SpokenLine(1.0, 2.0, text)], image_labels=labels or {})
    rig = Rig(fake)
    rig.run(0, until=1.0)
    if image is not None:
        rig.session.feed_image(image)
    rig.run(0, pcm=LOUD, until=2.0)
    rig.run(0, until=4.0)
    return [(c.name, c.args) for c in rig.session.take_calls()]


def test_the_ducks_name_called_is_greeted() -> None:
    assert say("蛋黃，過來這裡。") == [
        ("vocalize", {"tags": [{"tag": "greet", "level": 0.8, "at_ms": 0}]}),
        ("feel", {"valence": 0.4, "talk": 0.7}),
    ]
    assert say("小鴨在嗎") and say("阿勇！", name="阿勇")[0][0] == "vocalize"


def test_another_name_is_not_a_call() -> None:
    assert say("小美，吃飯了！") == []


def test_hello_is_a_gentler_greeting_and_a_look() -> None:
    assert say("你好！") == [
        ("vocalize", {"tags": [{"tag": "greet", "level": 0.6, "at_ms": 0}]}),
        ("attend", {"target_kind": "person", "hint": "說話的人"}),
    ]


def test_this_is_x_is_an_introduction() -> None:
    assert say("這是媽媽。") == [
        ("recognize", {"name": "媽媽"}),
        ("remember", {"note": "家人介紹了媽媽"}),
    ]
    assert say("這是什麼東西？") == []  # (a question, not an introduction)


def test_what_do_you_see_points_at_what_the_picture_shows_or_doubts() -> None:
    toy = b"\xff\xd8toy\xff\xd9"
    labels = {hashlib.sha256(toy).hexdigest(): ["玩具"]}
    assert say("你看到什麼？", labels=labels, image=toy) == [
        ("attend", {"target_kind": "thing", "hint": "玩具"})
    ]
    assert say("你看到什麼？") == [("feel", {"valence": 0.0, "talk": 0.1})]


def test_be_quiet_is_a_feeling_not_a_call() -> None:
    assert say("不要叫了。") == [("feel", {"valence": -0.3, "talk": 0.0})]


def test_a_line_that_was_not_sent_is_not_heard() -> None:
    fake = FakeGemini(lines=[SpokenLine(1.0, 2.0, "蛋黃，過來這裡。")])
    rig = Rig(fake)
    rig.run(0, until=4.0)  # (the room was quiet: nothing was said as far as the audio goes)
    assert rig.session.take_calls() == [] and rig.session.calls_total == 0


def test_a_line_only_half_sent_is_not_heard_but_a_third_is_enough() -> None:
    def calls(loud_from: float, loud_to: float) -> int:
        rig = Rig(FakeGemini(lines=[SpokenLine(1.0, 2.0, "你好")]))
        rig.run(0, until=loud_from)
        rig.run(0, pcm=LOUD, until=loud_to)
        rig.run(0, until=4.0)
        return rig.session.calls_total

    assert calls(1.0, 1.1) == 0
    assert calls(1.0, 1.5) == 2


def test_the_cloud_off_during_the_line_means_it_is_not_heard() -> None:
    rig = Rig(FakeGemini(lines=[SpokenLine(1.0, 2.0, "蛋黃，過來這裡。")]))
    rig.run(0, until=0.9)
    rig.session.set_enabled(False)
    rig.run(0, pcm=LOUD, until=2.0)
    rig.session.set_enabled(True)
    rig.run(0, until=5.0)
    assert rig.session.calls_total == 0


def test_the_events_come_with_a_servers_delays_and_in_a_servers_order() -> None:
    seen: list[E.LiveEvent] = []

    class Tap(FakeGemini):
        def poll(self, now: float) -> list[E.LiveEvent]:
            out = super().poll(now)
            seen.extend(out)
            return out

    rig = Rig(Tap(lines=[SpokenLine(1.0, 2.0, "蛋黃，過來這裡。")]))
    rig.run(0, until=1.0)
    rig.run(0, pcm=LOUD, until=2.0)
    rig.run(0, until=4.0)
    kinds = [e.kind for e in seen]
    assert kinds[0] == E.OPENED and E.RESUMPTION in kinds
    order = [
        k
        for k in kinds
        if k in (E.VAD, E.INPUT_TEXT, E.TOOL_CALL, E.GENERATION_COMPLETE, E.TURN_COMPLETE)
    ]
    assert order == [
        E.VAD,
        E.VAD,
        E.INPUT_TEXT,
        E.INPUT_TEXT,
        E.TOOL_CALL,
        E.GENERATION_COMPLETE,
        E.TURN_COMPLETE,
    ]
    vad = [e for e in seen if e.kind == E.VAD]
    assert [v.data["type"] for v in vad] == ["ACTIVITY_START", "ACTIVITY_END"]
    call = next(e for e in seen if e.kind == E.TOOL_CALL)
    assert 2.0 < call.t < 3.2  # (the tools come a moment after the speech ends)
    usage = next(e for e in seen if e.kind == E.USAGE)
    assert usage.data["total"] > 0 and set(usage.data["by_modality"]) == {"TEXT", "AUDIO"}


def test_it_is_deterministic() -> None:
    def run() -> list[tuple[str, dict[str, Any]]]:
        fake = FakeGemini(lines=[SpokenLine(1.0, 2.0, "這是媽媽。"), SpokenLine(5.0, 6.0, "你好")])
        rig = Rig(fake)
        rig.run(0, until=1.0)
        rig.run(0, pcm=LOUD, until=2.0)
        rig.run(0, until=5.0)
        rig.run(0, pcm=LOUD, until=6.0)
        rig.run(0, until=9.0)
        return [(e["kind"], {k: v for k, v in e.items() if k != "wall"}) for e in rig.session.log]

    assert run() == run()


def test_the_echo_of_its_own_voice_is_answered_only_when_asked_to_hallucinate() -> None:
    for hallucinate, expect in ((False, 0), (True, 1)):
        rig = Rig(FakeGemini(hallucinate=hallucinate))
        rig.run(0, until=1.0)
        rig.run(0, pcm=LOUD, echo=True, until=2.0)
        rig.run(0, pcm=QUIET, until=5.0)
        assert rig.session.calls_total == expect


def test_outcomes_drops_and_go_aways_are_scripted_faults() -> None:
    fake = FakeGemini(outcomes=["quota"])
    rig = Rig(fake)
    rig.run(0, until=1.0)
    assert rig.session.state == "quota"
    stale = FakeGemini()
    stale.open(
        type(
            "S",
            (),
            {
                "resume_handle": "not-issued",
                "model": "m",
                "tools": [],
                "response_modalities": ("AUDIO",),
                "system_instruction": "",
            },
        )(),
        0.0,
    )  # type: ignore[arg-type]
    [closed] = stale.poll(1.0)
    assert (
        closed.kind == E.CLOSED and closed.data["kind"] == "network"
    )  # (a handle it never issued)


def test_it_never_makes_a_tool_the_session_does_not_declare() -> None:
    from duckmind.live import tools as T

    for text in ("蛋黃", "你好", "這是爺爺", "你看到什麼", "不要叫了", "天氣真好", "小美"):
        for name, args in say(text):
            parsed = T.parse_call(name, args)
            assert name in T.TOOL_NAMES and parsed.valid and not parsed.off_bank
