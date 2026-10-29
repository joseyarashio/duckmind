"""Live voice, LV1: the echo gate (``duckmind.live.echo``). The duck's own call comes back
through its microphone; a transcript chunk or a tool call that can only be that is marked."""

from __future__ import annotations

import pytest

from duckmind.live.echo import ANSWER_S, HUMAN_PAD_S, EchoGate

pytestmark = pytest.mark.usefixtures("no_network")

LOUD, QUIET = -30.0, -62.0


def feed(gate: EchoGate, t0: float, t1: float, *, echo: bool, level: float = LOUD) -> None:
    """Frames (20 ms) from ``t0`` to ``t1``."""
    n = int(round((t1 - t0) / 0.02))
    for i in range(n):
        gate.on_frame(t0 + i * 0.02, echo, level)


def test_a_transcript_of_the_ducks_own_call_is_suspect() -> None:
    gate = EchoGate()
    feed(gate, 0.0, 1.0, echo=False, level=QUIET)
    feed(gate, 1.0, 1.6, echo=True)  # the duck calls
    mark = gate.mark_text(2.0)
    assert mark.echo_suspect and mark.echo_ms == 600 and mark.clean_ms == 0
    assert mark.echo_dominant and not mark.no_audio
    assert gate.texts_suspect == 1


def test_a_transcript_of_a_person_is_not() -> None:
    gate = EchoGate()
    feed(gate, 1.0, 2.0, echo=False)
    mark = gate.mark_text(2.3)
    assert not mark.echo_suspect and mark.clean_ms == 1000 and mark.echo_ms == 0
    assert not mark.echo_dominant


def test_a_person_after_a_call_with_a_pause_between_is_not_suspect() -> None:
    gate = EchoGate()
    feed(gate, 1.0, 1.6, echo=True)
    feed(gate, 1.6, 2.4, echo=False, level=QUIET)  # (0.8 s of quiet room: a new burst follows)
    feed(gate, 2.4, 3.4, echo=False)
    mark = gate.mark_text(3.8)
    assert not mark.echo_suspect and mark.clean_ms == 1000


def test_speech_right_after_the_call_without_a_pause_is_one_burst_and_suspect_but_not_dominant() -> (
    None
):
    gate = EchoGate()
    feed(gate, 1.0, 1.4, echo=True)
    feed(gate, 1.4, 2.6, echo=False)
    mark = gate.mark_text(3.0)
    assert mark.echo_suspect  # the literal rule: it overlaps the duck's voice
    assert not mark.echo_dominant and mark.echo_ms == 400 and mark.clean_ms == 1200


def test_a_barge_in_is_inside_flagged_frames_and_so_suspect() -> None:
    """The flag is the duck's, the mix is not separable: telling a person from the echo while
    the duck calls is the local voice detection's job (LV2), not the gate's."""
    gate = EchoGate()
    feed(gate, 1.0, 1.9, echo=True)  # the call, with a person speaking over its end
    mark = gate.mark_text(2.3)
    assert mark.echo_suspect and mark.echo_dominant


def test_a_transcript_over_the_ducks_call_is_the_persons_when_the_duck_heard_them() -> None:
    """Live 2026-10-04: the owner's real words were labelled "probably the duck's own call coming
    back" because the duck had called at the same time. The duck's own voice detection heard a
    person in that burst: the transcript is theirs (the numbers stay, the mark goes)."""
    asked: list[tuple[float, float]] = []

    def heard(t0: float, t1: float) -> bool:
        asked.append((t0, t1))
        return True

    gate = EchoGate(human_evidence=heard)
    feed(gate, 1.0, 1.9, echo=True)  # the call, with a person speaking over it
    mark = gate.mark_text(2.3)
    assert mark.rescued and not mark.echo_suspect
    assert mark.echo_ms == 900 and mark.echo_dominant  # (what the audio held is still said)
    assert len(asked) == 1 and asked[0][0] == pytest.approx(1.0)  # (the burst's own window)
    assert asked[0][1] == pytest.approx(1.9 + HUMAN_PAD_S, abs=0.03)
    assert gate.texts_rescued == 1 and gate.texts_suspect == 0
    assert mark.view()["rescued"] is True and gate.status()["texts_rescued"] == 1


def test_the_rescue_looks_only_at_the_window_of_the_burst_the_text_belongs_to() -> None:
    spans = [(5.0, 6.0)]  # a person spoke later: not in this burst

    def heard(t0: float, t1: float) -> bool:
        return any(a <= t1 and b >= t0 for a, b in spans)

    gate = EchoGate(human_evidence=heard)
    feed(gate, 1.0, 1.6, echo=True)  # the duck's own call, nobody speaking
    assert gate.mark_text(2.0).echo_suspect
    spans.append((1.2, 1.5))  # now a person is heard inside the burst
    assert not gate.mark_text(2.0).echo_suspect and gate.texts_rescued == 1


def test_without_evidence_or_with_a_broken_detector_the_mark_stays() -> None:
    plain = EchoGate()
    feed(plain, 1.0, 1.6, echo=True)
    assert plain.mark_text(2.0).echo_suspect and not plain.mark_text(2.0).rescued

    def broken(t0: float, t1: float) -> bool:
        raise RuntimeError("detector fault")

    gate = EchoGate(human_evidence=broken)
    feed(gate, 1.0, 1.6, echo=True)
    assert gate.mark_text(2.0).echo_suspect  # (a fault never takes a mark away)
    gate.set_human_evidence(None)
    assert gate.mark_text(2.0).echo_suspect


def test_a_person_in_no_burst_with_the_ducks_voice_needs_no_rescue_and_asks_nobody() -> None:
    asked: list[float] = []
    gate = EchoGate(human_evidence=lambda t0, t1: bool(asked.append(t0)) or True)
    feed(gate, 1.0, 2.0, echo=False)
    mark = gate.mark_text(2.3)
    assert not mark.echo_suspect and not mark.rescued and asked == []


def test_a_rescued_transcript_does_not_make_the_calls_it_causes_echo_causes() -> None:
    gate = EchoGate(human_evidence=lambda t0, t1: True)
    feed(gate, 1.0, 1.9, echo=True)
    gate.mark_text(2.3)
    call = gate.mark_call(2.8)
    assert call.cause_age_s is not None and call.cause_echo is False


def test_a_transcript_is_not_attributed_to_the_ducks_answer_that_began_before_it_arrived() -> None:
    """Live 2026-10-04: a person's words (332.6-334.0), the duck's call answering them (334.7), the
    transcript of the words (335.0): the call is the latest burst but it cannot be what the service
    is transcribing yet. The person's burst is: the chunk is not an echo."""
    gate = EchoGate()
    feed(gate, 332.6, 334.0, echo=False)  # the person
    feed(gate, 334.7, 335.2, echo=True)  # the duck's own answer, still going
    mark = gate.mark_text(335.04)
    assert not mark.echo_suspect and mark.clean_ms == 1400 and mark.echo_ms == 0
    # the same chunk a moment later, the call over for 0.3 s: now it is the call's
    assert gate.mark_text(335.55).echo_suspect


def test_a_fresh_burst_with_nothing_older_in_reach_is_what_the_transcript_says() -> None:
    gate = EchoGate()
    feed(gate, 1.0, 1.4, echo=True)
    assert gate.mark_text(1.5).echo_suspect  # (the only audio there is)
    feed(gate, 1.0 + 6.0, 1.4 + 6.0, echo=True)  # (the older burst is out of the lookback)
    assert gate.mark_text(7.5).echo_suspect


def test_a_transcript_with_no_loud_audio_behind_it_is_made_up() -> None:
    gate = EchoGate()
    feed(gate, 0.0, 5.0, echo=False, level=QUIET)
    mark = gate.mark_text(5.0)
    assert mark.no_audio and not mark.echo_suspect
    # and one long after the last loud burst
    feed(gate, 5.0, 6.0, echo=False)
    assert gate.mark_text(6.5).no_audio is False
    assert gate.mark_text(9.5).no_audio is True


def test_a_call_while_only_the_ducks_voice_was_sent_is_echo_only() -> None:
    gate = EchoGate()
    feed(gate, 0.0, 2.0, echo=False, level=QUIET)
    feed(gate, 2.0, 2.6, echo=True)
    mark = gate.mark_call(3.4)
    assert mark.echo_only and mark.since_human_s is None and mark.echo_ms_since_human == 600
    assert gate.calls_echo_only == 1


def test_a_call_that_answers_a_person_is_not_even_with_the_duck_calling_meanwhile() -> None:
    gate = EchoGate()
    feed(gate, 1.0, 2.0, echo=False)  # a person speaks
    feed(gate, 2.1, 2.7, echo=True)  # the duck starts calling at once
    mark = gate.mark_call(2.9)  # the model's answer to the person
    assert not mark.echo_only and mark.since_human_s is not None and mark.since_human_s < ANSWER_S


def test_a_second_call_after_the_ducks_own_call_with_no_new_speech_is_echo_only() -> None:
    gate = EchoGate()
    feed(gate, 1.0, 2.0, echo=False)
    feed(gate, 2.1, 2.7, echo=True)
    assert not gate.mark_call(2.9).echo_only  # the answer
    feed(gate, 2.7, 3.4, echo=False, level=QUIET)
    feed(gate, 3.4, 4.0, echo=True)  # the duck calls again
    second = gate.mark_call(4.4)  # the model reacts to what it heard: its own voice
    assert second.echo_only and second.answered


def test_a_call_long_after_speech_with_echo_since_is_echo_only() -> None:
    gate = EchoGate()
    feed(gate, 1.0, 2.0, echo=False)
    feed(gate, 5.0, 5.6, echo=True)
    mark = gate.mark_call(8.0)
    assert mark.echo_only and mark.since_human_s is not None and mark.since_human_s > ANSWER_S


def test_new_speech_clears_the_echo_count() -> None:
    gate = EchoGate()
    feed(gate, 0.0, 0.6, echo=True)
    feed(gate, 1.0, 1.4, echo=False)
    mark = gate.mark_call(2.0)
    assert not mark.echo_only and mark.echo_ms_since_human == 0


def test_a_call_from_a_silent_room_is_flagged_no_audio_not_echo() -> None:
    gate = EchoGate()
    feed(gate, 0.0, 6.0, echo=False, level=QUIET)
    mark = gate.mark_call(6.0)
    assert mark.no_audio and not mark.echo_only


def test_only_a_little_echo_does_not_make_a_call_echo_only() -> None:
    gate = EchoGate()
    feed(gate, 5.0, 5.06, echo=True)  # three frames
    assert not gate.mark_call(9.0).echo_only


def test_the_status_counts_and_the_marks_have_views() -> None:
    gate = EchoGate()
    feed(gate, 0.0, 0.4, echo=True)
    text = gate.mark_text(0.6)
    call = gate.mark_call(0.8)
    assert text.view()["echo_suspect"] is True and call.view()["echo_only"] is True
    assert gate.status() == {
        "frames": 20,
        "echo_frames": 20,
        "texts_suspect": 1,
        "texts_rescued": 0,
        "calls_echo_only": 1,
    }
