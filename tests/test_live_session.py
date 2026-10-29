"""Live voice, LV1: the persistent session (``duckmind.live.session``) against the fake Gemini.

Nothing here touches the network: the transport is a rule-based fake on a virtual clock. What is
checked: what goes up (audio only when a connection is open, pictures at most once a second, the
status text), what comes back (tool calls queued with timestamps and marks, answered SILENT),
the cloud switch, down/quota/cap states that never reach the caller, reconnecting with the
resumption handle, and the echo marks."""

from __future__ import annotations

from types import SimpleNamespace as NS
from typing import Any

import pytest
from live_rig import LOUD, QUIET, Rig, tone

from duckmind.i18n import CODES, render
from duckmind.live import events as E
from duckmind.live import session as S
from duckmind.live.fake import FakeGemini, SpokenLine
from duckmind.live.meter import UsageCaps, UsageMeter
from duckmind.live.soul import ContextSnapshot
from duckmind.live.transport import Setup

pytestmark = pytest.mark.usefixtures("no_network")

FAKE_KEY = "AQ." + "Ab3_-" * 9


# ── connecting and what goes up ────────────────────────────────────────────────────


def test_the_session_opens_one_connection_with_the_soul_the_tools_and_the_model() -> None:
    rig = Rig()
    rig.run(1.0)
    opened = rig.fake.sent.opens
    assert len(opened) == 1
    assert opened[0]["model"] == "gemini-3.8-live"
    assert opened[0]["tools"] == ["vocalize", "attend", "recognize", "remember", "feel"]
    assert opened[0]["modalities"] == [
        "AUDIO"
    ]  # (the model takes AUDIO only; the audio is thrown away)
    assert opened[0]["resume"] is False
    assert rig.session.state == S.ONLINE and rig.session.sending == {
        "audio": True,
        "video": True,
        "text": True,
    }


def test_the_connection_opens_with_the_language_hint_of_the_config() -> None:
    """Live 2026-10-04: no hint, no script: the service guessed a language per sentence."""
    seen: list[Setup] = []

    class Spy(FakeGemini):
        def open(self, setup: Setup, now: float) -> None:
            seen.append(setup)
            super().open(setup, now)

    Rig(Spy()).run(1.0)
    assert seen[0].transcription_languages == ("zh-TW", "en-US")
    Rig(Spy(), config=S.SessionConfig(transcription_languages=("en-US",))).run(1.0)
    assert seen[1].transcription_languages == ("en-US",)
    Rig(Spy(), config=S.SessionConfig(transcription_languages=())).run(1.0)
    assert seen[2].transcription_languages == ()


def test_the_connection_opens_with_the_vocabulary_the_host_set_unless_the_config_says_no() -> None:
    """The duck's name and the family's names go to the transcriber (planner 2026-10-04); the
    words are re-read at every connection, so a name given later is in the next one."""
    seen: list[Setup] = []

    class Spy(FakeGemini):
        def open(self, setup: Setup, now: float) -> None:
            seen.append(setup)
            super().open(setup, now)

    rig = Rig(Spy())
    rig.session.set_vocabulary(["蛋黃", "媽媽", " "])
    rig.run(1.0)
    assert seen[0].custom_vocabulary == ("蛋黃", "媽媽")
    assert rig.fake.sent.opens[0]["vocabulary"] == ["蛋黃", "媽媽"]
    off = Rig(Spy(), config=S.SessionConfig(custom_vocabulary=False))
    off.session.set_vocabulary(["蛋黃"])
    off.run(1.0)
    assert seen[1].custom_vocabulary == ()
    plain = Rig(Spy())
    plain.run(1.0)
    assert seen[2].custom_vocabulary == ()  # (nothing set: nothing sent)


def test_a_frame_with_a_cloud_copy_sends_the_copy_and_the_gate_takes_its_level() -> None:
    """A browser microphone's person reaches the cloud at their recording's own level, not at the
    world's distance (the duck's own ear keeps ``pcm``): the copy goes up, and its level is what the
    echo gate's bursts are made of."""
    sent: list[bytes] = []

    class Spy(FakeGemini):
        def send_audio(self, pcm: bytes, now: float) -> None:
            sent.append(pcm)
            super().send_audio(pcm, now)

    rig = Rig(Spy())
    rig.run(1.0)
    sent.clear()
    far = NS(pcm=QUIET, cloud_pcm=LOUD, echo=False, level_dbfs=-62.0, cloud_level_dbfs=-30.0)
    rig.clock.set(round(rig.clock.t + 0.02, 6))
    assert rig.session.feed_audio(far) is True
    assert sent == [LOUD]
    assert rig.session.gate.mark_text(rig.clock.t + 0.1).no_audio is False  # (a loud burst)
    plain = NS(pcm=QUIET, echo=False, level_dbfs=-62.0)  # (no copy: the frame as it is)
    rig.clock.set(round(rig.clock.t + 0.02, 6))
    rig.session.feed_audio(plain)
    assert sent[-1] == QUIET


def test_audio_waits_for_the_connection_and_goes_first_in_order() -> None:
    sent: list[bytes] = []

    class Spy(FakeGemini):
        def send_audio(self, pcm: bytes, now: float) -> None:
            sent.append(pcm)
            super().send_audio(pcm, now)

    rig = Rig(Spy())
    for k in range(8):  # (the connection takes 50 ms: the first frames wait)
        rig.clock.set(round(0.02 * (k + 1), 3))
        rig.session.feed_pcm(tone(-40.0, phase=k * 7), echo=False)
        rig.session.pump()
    assert sent, "nothing was sent after the connection opened"
    assert sent[: len(sent)] == [tone(-40.0, phase=k * 7) for k in range(len(sent))]
    assert len(sent) == 8


def test_audio_held_for_longer_than_the_hold_is_dropped_and_counted() -> None:
    rig = Rig(FakeGemini(outcomes=["network"] * 1))
    rig.session.config  # (default hold 2 s)
    rig.run(5.0)  # (the first connection fails; the retry comes after 1.5 s)
    totals = rig.session.status()["usage"]["totals"]
    assert totals["errors"] >= 1
    assert totals["audio_s"] > 0  # (it went up once the line was back)


def test_pictures_go_up_at_most_once_a_second() -> None:
    rig = Rig()
    rig.run(0.5)
    for _ in range(100):  # ten seconds of 10 pictures a second
        rig.run(0.1)
        rig.session.feed_image(b"\xff\xd8jpeg\xff\xd9")
    # (the pictures that came early wait for their slot, the newest replacing the older)
    assert 9 <= rig.fake.sent.images <= 11
    usage = rig.session.status()["usage"]["totals"]
    assert usage["frames"] == rig.fake.sent.images
    assert usage["frames"] + usage["frames_skipped"] >= 98


def test_a_picture_a_little_early_is_not_lost_it_waits_for_its_slot() -> None:
    """A source that ticks once a second jitters: the session must not drop every picture that
    comes 5 ms early (it used to: a third of a one-a-second source), and must still never send two
    less than a second apart."""
    rig = Rig()
    rig.run(1.0)
    times: list[float] = []

    original = rig.fake.send_image

    def spy(jpeg: bytes, now: float, mime: str = "image/jpeg") -> None:
        times.append(now)
        original(jpeg, now, mime)

    rig.fake.send_image = spy  # type: ignore[method-assign]
    schedule = [1.5 + k * 1.0 - (0.012 if k % 3 == 1 else 0.0) for k in range(10)]  # (jitter)
    for due in schedule:
        rig.run(0, until=round(due, 3))
        rig.session.feed_image(b"p")
    rig.run(0, until=schedule[-1] + 1.0)
    assert len(times) == 10, times  # (none lost)
    assert all(b - a >= 1.0 - 1e-6 for a, b in zip(times, times[1:], strict=False)), times
    assert rig.session.status()["usage"]["totals"]["frames_skipped"] == 0


def test_a_waiting_picture_is_forgotten_when_the_line_drops_or_the_cloud_goes_off() -> None:
    rig = Rig()
    rig.run(1.0)
    assert rig.session.feed_image(b"a") is True
    assert rig.session.feed_image(b"b") is False  # (too soon: waiting)
    rig.session.set_enabled(False)
    rig.run(2.0)
    rig.session.set_enabled(True)
    rig.run(3.0)
    assert rig.fake.sent.images == 1  # (the waiting one did not go out after the switch)


def test_no_pictures_while_the_line_is_not_up_or_the_cloud_is_off() -> None:
    rig = Rig()
    assert rig.session.feed_image(b"x") is False  # (not connected yet)
    rig.run(1.0)
    rig.session.set_enabled(False)
    assert rig.session.feed_image(b"x") is False and rig.fake.sent.images == 0


def test_the_status_text_goes_up_at_once_then_only_when_it_changed_and_not_too_often() -> None:
    rig = Rig()
    rig.session.set_context(ContextSnapshot(behaviour="坐著看四周"))
    rig.run(1.0)
    assert len(rig.fake.sent.texts) == 1
    _, text, complete = rig.fake.sent.texts[0]
    assert "坐著看四周" in text and complete is False  # (the server waits: it does not answer it)
    rig.run(2.0)
    assert len(rig.fake.sent.texts) == 1  # (nothing changed)
    rig.session.set_context(ContextSnapshot(behaviour="走向窗邊"))
    rig.session.set_context(ContextSnapshot(behaviour="走向窗邊", valence=0.3))
    rig.run(1.0)  # (not five seconds since the last one yet)
    assert len(rig.fake.sent.texts) == 1
    rig.run(1.5)
    assert len(rig.fake.sent.texts) == 2 and "心情不錯" in rig.fake.sent.texts[1][1]
    assert rig.session.status()["usage"]["totals"]["text_updates"] == 2


def test_a_note_goes_up_at_once_as_context_the_server_does_not_answer_and_keeps_the_status_rhythm() -> (
    None
):
    rig = Rig()
    rig.session.set_context(ContextSnapshot(behaviour="坐著看四周"))
    rig.run(1.0)
    assert len(rig.fake.sent.texts) == 1  # (the status, at once)
    assert rig.session.send_note("（我自己的耳朵：剛剛有人叫我的名字）", about="ear_name") is True
    now, text, complete = rig.fake.sent.texts[1]
    assert (text, complete) == ("（我自己的耳朵：剛剛有人叫我的名字）", False)
    assert now == pytest.approx(rig.clock.t)
    status = rig.session.status()
    assert status["notes_sent"] == 1 and status["usage"]["totals"]["text_updates"] == 2
    assert [e for e in rig.session.log if e["kind"] == "note"][0]["about"] == "ear_name"
    # the note is not the status: the status still goes up five seconds after the last status,
    # not five seconds after the note
    rig.session.set_context(ContextSnapshot(behaviour="走向窗邊"))
    rig.run(4.1)
    texts = [t for _, t, _ in rig.fake.sent.texts]
    assert len(texts) == 3 and "走向窗邊" in texts[2]


def test_a_note_that_cannot_go_now_is_dropped_and_never_kept_for_later() -> None:
    rig = Rig()
    assert rig.session.send_note("a") is False  # (the line is not up yet)
    rig.run(1.0)
    rig.session.set_enabled(False)
    assert rig.session.send_note("b") is False  # (the cloud is off)
    rig.run(1.0)
    rig.session.set_enabled(True)
    rig.run(3.0)
    assert rig.session.state == S.ONLINE
    assert [t for _, t, _ in rig.fake.sent.texts] == []  # (neither came later)
    assert rig.session.status()["notes_sent"] == 0
    rig.session.close()
    assert rig.session.send_note("c") is False


def test_a_note_the_transport_cannot_send_is_a_dropped_line_not_an_error() -> None:
    class NoText(FakeGemini):
        def send_text(self, text: str, now: float, *, turn_complete: bool = False) -> None:
            raise ConnectionError("socket closed")

    rig = Rig(NoText())
    rig.run(1.0)
    assert rig.session.send_note("x") is False
    assert rig.session.faults == 1 and rig.session.status()["notes_sent"] == 0
    rig.run(8.0)
    assert rig.session.state == S.ONLINE and len(rig.fake.sent.opens) == 2


def test_a_plain_string_is_a_status_text_too() -> None:
    rig = Rig()
    rig.session.set_context("今天很安靜")
    rig.run(0.5)
    assert rig.fake.sent.texts[0][1] == "今天很安靜"


# ── what comes back ────────────────────────────────────────────────────────────────


def run_speech(rig: Rig, start: float, end: float) -> None:
    rig.run(0, until=start)
    rig.run(0, pcm=LOUD, until=end)


def test_a_name_call_makes_the_model_propose_a_greeting_and_the_call_is_queued_with_its_time() -> (
    None
):
    fake = FakeGemini(lines=[SpokenLine(1.0, 2.0, "蛋黃，過來這裡。")])
    rig = Rig(fake)
    run_speech(rig, 1.0, 2.0)
    rig.run(0, until=4.0)
    calls = rig.session.take_calls()
    assert [c.name for c in calls] == ["vocalize", "feel"]
    first = calls[0]
    assert first.args == {"tags": [{"tag": "greet", "level": 0.8, "at_ms": 0}]}
    assert first.t > 2.0 and first.wall > 0 and first.connection == 1 and first.problems == ()
    assert calls[0].t == calls[1].t and not first.echo_suspect
    assert rig.session.take_calls() == []  # (taken: gone)
    assert rig.session.status()["calls"] == {"queued": 0, "total": 2, "dropped": 0}


def test_every_call_is_answered_to_the_model_silently() -> None:
    fake = FakeGemini(lines=[SpokenLine(1.0, 2.0, "蛋黃，過來這裡。")])
    rig = Rig(fake)
    run_speech(rig, 1.0, 2.0)
    rig.run(0, until=4.0)
    answers = fake.sent.tool_responses
    assert [a["name"] for a in answers] == ["vocalize", "feel"]
    assert all(
        a["scheduling"] == "SILENT" and a["response"] == {"status": "queued"} for a in answers
    )
    assert [a["id"] for a in answers] == [c.id for c in rig.session.peek_calls()]


def test_lv2_can_refuse_through_the_responder_and_a_broken_one_cannot_hurt() -> None:
    fake = FakeGemini(lines=[SpokenLine(1.0, 2.0, "蛋黃，過來這裡。")])
    rig = Rig(fake, responder=lambda call: {"status": "refused", "why": "rate"})
    run_speech(rig, 1.0, 2.0)
    rig.run(0, until=4.0)
    assert all(a["response"]["status"] == "refused" for a in fake.sent.tool_responses)

    def broken(call: Any) -> dict[str, Any]:
        raise RuntimeError("responder broke")

    rig = Rig(FakeGemini(lines=[SpokenLine(1.0, 2.0, "蛋黃，過來這裡。")]), responder=broken)
    run_speech(rig, 1.0, 2.0)
    rig.run(0, until=4.0)
    assert rig.session.faults == 2 and len(rig.session.peek_calls()) == 2
    assert rig.session.state == S.ONLINE


def test_transcripts_the_models_text_and_vad_are_logged_and_its_voice_is_discarded() -> None:
    fake = FakeGemini(lines=[SpokenLine(1.0, 2.0, "蛋黃，過來這裡。")])
    rig = Rig(fake)
    run_speech(rig, 1.0, 2.0)
    rig.run(0, until=4.0)
    heard = [e for e in rig.session.log if e["kind"] == "heard"]
    assert "".join(e["text"] for e in heard) == "蛋黃，過來這裡。"
    assert rig.session.last_vad is not None and rig.session.last_vad["type"] == "ACTIVITY_END"
    usage = rig.session.status()["usage"]
    assert usage["totals"]["turns"] == 1 and usage["totals"]["tokens"] > 0
    rig.session._handle_event(E.LiveEvent(E.AUDIO_OUT, 5.0, {"bytes": 4800}), 5.0)
    assert rig.session.status()["audio_out_discarded_bytes"] == 4800


def test_the_queue_is_bounded_and_says_what_it_dropped() -> None:
    lines = [SpokenLine(1.0 + 3 * k, 2.0 + 3 * k, "蛋黃，過來這裡。") for k in range(4)]
    rig = Rig(FakeGemini(lines=lines), config=S.SessionConfig(call_queue_max=3))
    for k in range(4):
        run_speech(rig, 1.0 + 3 * k, 2.0 + 3 * k)
    rig.run(0, until=14.0)
    assert len(rig.session.peek_calls()) == 3
    assert rig.session.calls_total == 8 and rig.session.calls_dropped == 5
    assert len(rig.session.take_calls(limit=2)) == 2 and len(rig.session.peek_calls()) == 1


def test_a_bad_call_is_queued_with_its_problems() -> None:
    rig = Rig()
    rig.run(0.5)
    rig.session._handle_event(
        E.LiveEvent(
            E.TOOL_CALL,
            1.0,
            {
                "calls": [
                    {"id": "x1", "name": "vocalize", "args": {"tags": [{"tag": "hi", "level": 9}]}}
                ]
            },
        ),
        1.0,
    )
    [call] = rig.session.peek_calls()
    assert call.off_bank == ("hi",) and "level_out_of_range" in call.problems
    assert call.args["tags"][0]["level"] == 1.0 and call.raw == {
        "tags": [{"tag": "hi", "level": 9}]
    }


# ── the echo marks ─────────────────────────────────────────────────────────────────


def test_the_ducks_own_call_heard_as_speech_is_marked_and_so_are_the_calls_it_causes() -> None:
    rig = Rig(FakeGemini(hallucinate=True))
    rig.run(1.0)
    rig.run(0, pcm=LOUD, echo=True, until=2.5)  # the duck calls; nobody speaks
    rig.run(0, until=5.0)
    heard = [e for e in rig.session.log if e["kind"] == "heard"]
    assert heard and all(e["echo_suspect"] for e in heard)
    calls = rig.session.take_calls()
    assert calls and all(c.echo_suspect and c.mark.echo_only for c in calls)
    echo = rig.session.status()["echo"]
    assert echo["echo_frames"] > 0 and echo["texts_suspect"] >= 1 and echo["calls_echo_only"] >= 1


def test_a_persons_words_over_the_ducks_call_are_not_marked_when_the_duck_itself_heard_them() -> (
    None
):
    """Live 2026-10-04: a person speaking while the duck calls is inside flagged frames; the duck's
    own voice detection (handed in by the wiring) says a person spoke: the transcript is theirs."""

    def run(evidence: Any) -> list[dict[str, Any]]:
        rig = Rig(FakeGemini(lines=[SpokenLine(1.0, 2.0, "你好")]))
        if evidence is not None:
            rig.session.set_human_evidence(evidence)
        rig.run(0, until=1.0)
        rig.run(0, pcm=LOUD, echo=True, until=2.0)  # the call and the person: one flagged burst
        rig.run(0, until=4.0)
        return [e for e in rig.session.log if e["kind"] == "heard"]

    for heard in (run(None), run(lambda t0, t1: False)):
        assert heard and all(e["echo_suspect"] and not e["rescued"] for e in heard)
    heard = run(lambda t0, t1: True)
    assert heard and all(not e["echo_suspect"] and e["rescued"] for e in heard)
    assert all(e["echo_ms"] > 0 for e in heard)  # (the audio did hold the duck's own voice)


def test_a_persons_speech_is_not_marked() -> None:
    fake = FakeGemini(lines=[SpokenLine(1.0, 2.0, "你好")])
    rig = Rig(fake)
    run_speech(rig, 1.0, 2.0)
    rig.run(0, until=4.0)
    assert all(not e["echo_suspect"] for e in rig.session.log if e["kind"] == "heard")
    assert all(not c.echo_suspect for c in rig.session.take_calls())


# ── the cloud switch ───────────────────────────────────────────────────────────────


def test_off_stops_everything_leaving_and_says_so() -> None:
    rig = Rig()
    rig.run(1.0)
    sent = rig.fake.sent.audio_frames
    rig.session.set_enabled(False)
    assert rig.fake.sent.audio_ends == 1 and rig.fake.sent.closes >= 1
    rig.run(2.0)
    assert rig.fake.sent.audio_frames == sent and rig.fake.sent.images == 0
    status = rig.session.status()
    assert status["enabled"] is False and status["state"] == "off"
    assert status["sending"] == {"audio": False, "video": False, "text": False}
    assert status["understands_speech"] is False and "聽不懂人話" in status["words"]


def test_off_while_already_off_and_on_again_resumes_the_session() -> None:
    rig = Rig()
    rig.run(2.0)
    rig.session.set_enabled(False)
    rig.session.set_enabled(False)  # (nothing happens twice)
    rig.run(1.0)
    rig.session.set_enabled(True)
    rig.run(1.0)
    assert rig.session.state == S.ONLINE
    assert [o["resume"] for o in rig.fake.sent.opens] == [False, True]
    assert rig.session.status()["resumes"] == 1


def test_a_session_started_off_never_connects() -> None:
    rig = Rig(enabled=False)
    rig.run(3.0)
    assert rig.fake.sent.opens == [] and rig.session.state == S.OFF
    assert rig.session.feed_pcm(LOUD) is False and rig.fake.sent.audio_frames == 0


def test_close_is_for_good() -> None:
    rig = Rig()
    rig.run(1.0)
    rig.session.close()
    rig.run(1.0)
    assert rig.session.state == S.OFF and len(rig.fake.sent.opens) == 1


# ── down, quota, cap: the duck lives on and the status says so ─────────────────────


def test_a_refused_line_is_retried_with_a_backoff_and_the_status_says_it_is_reconnecting() -> None:
    rig = Rig(FakeGemini(outcomes=["network", "network", "ok"]))
    rig.run(0.5)
    assert rig.session.state == S.RECONNECTING
    assert rig.session.status()["retry_in_s"] is not None
    rig.run(10.0)
    assert rig.session.state == S.ONLINE
    assert len(rig.fake.sent.opens) == 3
    assert rig.session.status()["usage"]["totals"]["errors"] == 2


def test_three_failures_in_a_row_say_down() -> None:
    rig = Rig(FakeGemini(outcomes=["network"] * 6))
    seen: set[str] = set()
    for _ in range(300):
        rig.run(0.1)
        seen.add(rig.session.state)
    assert S.DOWN in seen
    status = rig.session.status()
    assert "照常生活" in status["words"] and status["understands_speech"] is False


def test_quota_backs_off_long_and_doubles_and_is_counted() -> None:
    config = S.SessionConfig(quota_backoff_s=10.0, quota_backoff_max_s=40.0)
    rig = Rig(FakeGemini(outcomes=["quota", "quota", "quota", "quota", "ok"]), config=config)
    waits: list[float] = []
    seen = 0
    for _ in range(1500):
        rig.run(0.1)
        status = rig.session.status()
        errors = status["usage"]["totals"]["quota_errors"]
        if errors > seen:
            seen = errors
            waits.append(status["retry_in_s"])
            assert status["state"] == "quota" and status["understands_speech"] is False
        if status["state"] == "online":
            break
    assert [round(w, -1) for w in waits] == [10.0, 20.0, 40.0, 40.0]  # (doubling, then the cap)
    assert rig.session.state == S.ONLINE
    assert rig.session.status()["usage"]["totals"]["quota_errors"] == 4
    assert rig.session.last_error is not None and rig.session.last_error["kind"] == "quota"


def test_no_key_waits_a_minute_and_says_so() -> None:
    rig = Rig(FakeGemini(outcomes=["no_key", "ok"]))
    rig.run(2.0)
    assert rig.session.state == S.NO_KEY and "金鑰" in rig.session.status()["words"]
    rig.run(61.0)
    assert rig.session.state == S.ONLINE


def test_a_dropped_line_resumes_with_the_handle() -> None:
    rig = Rig(FakeGemini(drops=[6.0]))
    rig.run(10.0)
    assert [o["resume"] for o in rig.fake.sent.opens] == [False, True]
    assert rig.session.state == S.ONLINE
    assert rig.session.status()["resumes"] == 1 and rig.session.status()["connections"] == 2


def test_two_failures_forget_a_stale_handle_and_start_clean() -> None:
    rig = Rig(FakeGemini(outcomes=["ok", "network", "network", "ok"], drops=[3.0]))
    rig.run(30.0)
    assert [o["resume"] for o in rig.fake.sent.opens] == [False, True, False, False]
    assert rig.session.state == S.ONLINE


def test_go_away_moves_to_a_new_connection_without_losing_audio() -> None:
    sent: list[bytes] = []

    class Spy(FakeGemini):
        def send_audio(self, pcm: bytes, now: float) -> None:
            if self._open:
                sent.append(pcm)
            super().send_audio(pcm, now)

    rig = Rig(Spy(go_aways=[3.0]))
    fed = 0
    while rig.clock.t < 6.0:
        rig.clock.set(round(rig.clock.t + 0.02, 6))
        rig.session.feed_pcm(tone(-40.0, phase=fed), echo=False)
        rig.session.pump()
        fed += 1
    assert len(rig.fake.sent.opens) == 2 and rig.fake.sent.opens[1]["resume"] is True
    assert len(sent) == fed  # (every frame went up, the gap's ones waiting for the new line)


def test_a_cap_stops_the_cloud_by_itself() -> None:
    rig = Rig(meter=UsageMeter(UsageCaps(audio_s=2.0)))
    rig.run(1.5)
    assert rig.session.state == S.ONLINE
    rig.run(2.0)
    status = rig.session.status()
    assert status["state"] == "capped" and status["usage"]["over_cap"] == "audio_s"
    assert "自動關掉" in status["words"] and status["sending"]["audio"] is False
    sent = rig.fake.sent.audio_frames
    rig.run(1.0)
    assert rig.fake.sent.audio_frames == sent and rig.fake.sent.closes >= 1


# ── never raises into the brain loop ───────────────────────────────────────────────


class Broken:
    """A transport that fails at everything (and puts a key in every message)."""

    name = "broken"

    def _boom(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(f"boom with {FAKE_KEY} in it")

    open = send_audio = send_audio_end = send_image = send_text = _boom
    send_tool_response = poll = close = _boom


def test_a_broken_transport_never_raises_and_never_leaks_the_key() -> None:
    rig = Rig(Broken())
    for _ in range(100):
        rig.run(0.1)
        rig.session.feed_image(b"x")
        rig.session.set_context("c")
    rig.session.set_enabled(False)
    rig.session.set_enabled(True)
    rig.session.close()
    status = rig.session.status()
    assert status["faults"] > 0 and status["last_fault"] is not None
    assert FAKE_KEY not in str(status) and FAKE_KEY not in str(list(rig.session.log))
    assert "<redacted>" in status["last_fault"]


def test_a_transport_that_dies_mid_session_is_a_dropped_line() -> None:
    class Dies(FakeGemini):
        def send_audio(self, pcm: bytes, now: float) -> None:
            if now > 2.0 and not getattr(self, "died", False):
                self.died = True  # type: ignore[attr-defined]
                raise ConnectionError("socket closed")
            super().send_audio(pcm, now)

    rig = Rig(Dies())
    rig.run(8.0)
    assert rig.session.faults == 1 and rig.session.state == S.ONLINE
    assert len(rig.fake.sent.opens) == 2


def test_the_status_is_json_ready_and_complete() -> None:
    import json

    rig = Rig()
    rig.run(1.0)
    status = rig.session.status()
    assert {
        "enabled",
        "state",
        "words",
        "sending",
        "model",
        "usage",
        "calls",
        "echo",
        "faults",
    } <= set(status)
    json.dumps(status, ensure_ascii=False)
    assert status["model"] == "gemini-3.8-live" and status["connections"] == 1


@pytest.mark.parametrize("state", S.STATES)
def test_every_state_has_words_in_all_six_languages(state: str) -> None:
    from duckmind.i18n import msg

    words = {code: render(msg(f"neuro.live.status.{state}"), code) for code in CODES}
    assert len(words) == 6 and all(words.values())
    assert len(set(words.values())) == 6  # (six different texts: nothing left in one language)


def test_the_microphone_stream_feeds_the_session() -> None:
    from duckmind.simweb.neuro_mic import MicStream

    rig = Rig()
    mic = MicStream(seed=3, library=None, calls=None, wall=lambda: 0.0)
    rig.session.attach_mic(mic)
    for i in range(100):
        rig.clock.set(i * 0.02)
        mic.tick(i * 0.02, [], wall=i * 0.02)
    assert rig.session.state == S.ONLINE and rig.fake.sent.audio_frames >= 90
    assert rig.session.gate.frames == 100
