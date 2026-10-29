"""Live voice, LV1: cassettes (``duckmind.live.cassette``): a session recorded once and replayed
on the same timeline, keyed by a hash of the scenario inputs, failing loudly when they differ."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from live_rig import LOUD, Rig

from duckmind.live import cassette as C
from duckmind.live import events as E
from duckmind.live import keys as K
from duckmind.live import session as S
from duckmind.live.fake import FakeGemini, SpokenLine
from duckmind.live.scenario import ManualClock
from duckmind.live.soul import ContextSnapshot
from duckmind.live.tools import TOOL_SCHEMA_VERSION, schema_hash

pytestmark = pytest.mark.usefixtures("no_network")

FAKE_KEY = "AQ." + "Kp2_-" * 9
SOUL = "SOUL"


def inputs(**over: Any) -> dict[str, Any]:
    import hashlib

    base: dict[str, Any] = {
        "scenario": "unit",
        "seed": 7,
        "model": "gemini-3.8-live",
        "tool_schema": {"version": TOOL_SCHEMA_VERSION, "hash": schema_hash()},
        "soul": {"version": "soul-1", "sha256": hashlib.sha256(SOUL.encode()).hexdigest()},
        "say": [{"t": 1.0, "utterance": "call_name", "sha256": "ab" * 32, "at_m": [2.0, 0.5]}],
    }
    base.update(over)
    return base


def record(fake: FakeGemini, in_: dict[str, Any], seconds: float = 5.0) -> tuple[C.Cassette, Rig]:
    recorder = C.RecordingTransport(fake, "unit", in_)
    rig = Rig(recorder)
    rig.session.set_context(ContextSnapshot(behaviour="坐著看四周"))
    rig.run(0, until=1.0)
    rig.run(0, pcm=LOUD, until=2.0)
    rig.run(0, until=seconds)
    return recorder.cassette(complete=True), rig


def talker() -> FakeGemini:
    return FakeGemini(lines=[SpokenLine(1.0, 2.0, "蛋黃，過來這裡。")])


# ── the key ────────────────────────────────────────────────────────────────────────


def test_the_hash_is_a_fingerprint_of_the_inputs_in_canonical_form() -> None:
    a = C.scenario_hash(inputs())
    assert a == C.scenario_hash(dict(reversed(list(inputs().items()))))  # (order does not matter)
    assert len(a) == 64
    for change in (
        {"seed": 8},
        {"model": "gemini-3.8-live-extended-thinking"},
        {"say": [{"t": 1.5, "utterance": "call_name", "sha256": "ab" * 32, "at_m": [2.0, 0.5]}]},
        {
            "say": [{"t": 1.0, "utterance": "call_name", "sha256": "cd" * 32, "at_m": [2.0, 0.5]}]
        },  # a fixture was regenerated
        {
            "say": [{"t": 1.0, "utterance": "call_name", "sha256": "ab" * 32, "at_m": [2.0, 0.6]}]
        },  # a position moved
        {"tool_schema": {"version": "tools-2", "hash": "x"}},
        {"soul": {"version": "soul-2", "sha256": "y"}},
    ):
        assert C.scenario_hash(inputs(**change)) != a, change


def test_diff_inputs_names_what_differs() -> None:
    assert C.diff_inputs(inputs(), inputs(seed=9, model="x")) == ["model", "seed"]
    assert C.diff_inputs(inputs(), inputs()) == []


# ── recording and files ────────────────────────────────────────────────────────────


def test_a_recording_keeps_what_came_back_on_the_scenarios_timeline() -> None:
    cassette, rig = record(talker(), inputs())
    kinds = [e.kind for e in cassette.events]
    assert (
        kinds[0] == E.OPENED and E.INPUT_TEXT in kinds and E.TOOL_CALL in kinds and E.USAGE in kinds
    )
    assert all(a.t <= b.t for a, b in zip(cassette.events, cassette.events[1:], strict=False))
    call = next(e for e in cassette.events if e.kind == E.TOOL_CALL)
    assert call.data["calls"][0]["name"] == "vocalize" and 2.0 < call.t < 3.5
    assert cassette.scenario_hash == C.scenario_hash(inputs()) and cassette.complete


def open_setup(**over: Any) -> Any:
    """What the session hands ``open``: the recorded soul, model and tools."""
    from duckmind.live.tools import tools_config
    from duckmind.live.transport import Setup

    return Setup(
        system_instruction=over.pop("system_instruction", SOUL), tools=tools_config(), **over
    )


def test_resumption_handles_and_the_models_voice_are_never_stored() -> None:
    cassette, _ = record(talker(), inputs())
    resumption = [e for e in cassette.events if e.kind == E.RESUMPTION]
    assert resumption and all("handle" not in e.data for e in resumption)
    assert "fake-handle" not in json.dumps(cassette.to_json())
    # (the voice: counted, not kept)
    fake = FakeGemini()
    rec = C.RecordingTransport(fake, "u", inputs())
    rec.open(open_setup(), 0.0)
    fake._schedule(0.1, E.AUDIO_OUT, {"bytes": 4800})
    fake._schedule(0.2, E.AUDIO_OUT, {"bytes": 1000})
    rec.poll(1.0)
    assert [e.kind for e in rec.events] == [E.OPENED, E.RESUMPTION]
    assert (
        rec.audio_out_bytes == 5800 and rec.cassette(complete=True).meta["audio_out_bytes"] == 5800
    )


def test_a_cassette_is_a_small_readable_json_file_that_round_trips(tmp_path: Path) -> None:
    cassette, _ = record(talker(), inputs())
    path = tmp_path / "unit.json"
    size = cassette.save(path)
    assert size == path.stat().st_size and size < 20_000
    loaded = C.Cassette.load(path)
    assert loaded.scenario_hash == cassette.scenario_hash and loaded.inputs == cassette.inputs
    assert [(e.kind, e.t, e.data) for e in loaded.events] == [
        (e.kind, e.t, e.data) for e in cassette.events
    ]
    assert "蛋黃" in path.read_text(encoding="utf-8")  # (utf-8 text as it is, not escapes)
    assert not list(tmp_path.glob("*.tmp"))


def test_an_edited_cassette_is_detected() -> None:
    cassette, _ = record(talker(), inputs())
    raw = cassette.to_json()
    raw["inputs"]["seed"] = 99
    with pytest.raises(C.CassetteMismatch, match="edited"):
        C.Cassette.from_json(raw)
    raw = cassette.to_json()
    raw["format"] = 99
    with pytest.raises(C.CassetteMismatch, match="format"):
        C.Cassette.from_json(raw)


def test_nothing_key_shaped_is_ever_written(tmp_path: Path) -> None:
    cassette = C.Cassette(
        "k", inputs(), [E.LiveEvent(E.ERROR, 1.0, {"message": f"oops {FAKE_KEY}"})]
    )
    with pytest.raises(C.KeyInCassette):
        cassette.save(tmp_path / "k.json")
    assert not (tmp_path / "k.json").exists()


def test_text_off_the_wire_is_redacted_on_its_way_into_a_cassette() -> None:
    fake = FakeGemini()
    recorder = C.RecordingTransport(fake, "u", inputs())
    rig = Rig(recorder)
    rig.run(0, until=0.5)
    fake._schedule(0.6, E.ERROR, {"kind": "server", "message": f"x wss://h/?key={FAKE_KEY}"})
    rig.run(0, until=1.0)
    text = json.dumps(recorder.cassette(complete=True).to_json())
    assert FAKE_KEY not in text and K.MASK in text


# ── replay ─────────────────────────────────────────────────────────────────────────


def summary(rig: Rig) -> list[tuple[Any, ...]]:
    return [(c.name, c.args, round(c.t, 3), c.echo_suspect) for c in rig.session.peek_calls()]


def replay_through(cassette: C.Cassette, in_: dict[str, Any]) -> Rig:
    rig = Rig(C.ReplayTransport(cassette, in_))
    rig.session.set_context(ContextSnapshot(behaviour="坐著看四周"))
    rig.run(0, until=1.0)
    rig.run(0, pcm=LOUD, until=2.0)
    rig.run(0, until=5.0)
    return rig


def test_a_replay_gives_the_session_the_same_events_at_the_same_times() -> None:
    cassette, recorded = record(talker(), inputs())
    replayed = replay_through(cassette, inputs())
    assert summary(replayed) == summary(recorded) and summary(replayed)

    def heard(rig: Rig) -> list[tuple[Any, ...]]:
        return [
            (e["text"], round(e["t"], 3), e["echo_suspect"])
            for e in rig.session.log
            if e["kind"] == "heard"
        ]

    assert heard(replayed) == heard(recorded) and heard(replayed)
    assert (
        replayed.session.status()["usage"]["totals"]["tokens"]
        == recorded.session.status()["usage"]["totals"]["tokens"]
    )
    assert replayed.fake.finished


def test_a_replay_is_deterministic_and_sends_nothing_anywhere() -> None:
    cassette, _ = record(talker(), inputs())
    a, b = replay_through(cassette, inputs()), replay_through(cassette, inputs())
    assert summary(a) == summary(b)
    assert a.fake.sent.audio_frames > 0 and a.fake.name == "replay"


def test_events_come_out_only_when_the_clock_has_passed_them() -> None:
    cassette, _ = record(talker(), inputs())
    call_t = next(e.t for e in cassette.events if e.kind == E.TOOL_CALL)
    replay = C.ReplayTransport(cassette, inputs())
    replay.open(open_setup(), 10.0)  # (the session's clock was at 10 s when it opened)
    early = replay.poll(10.0 + call_t - 0.001)
    assert early and E.TOOL_CALL not in [e.kind for e in early]
    out = replay.poll(10.0 + call_t + 0.001)
    assert [e.kind for e in out if e.kind == E.TOOL_CALL] == [E.TOOL_CALL]
    assert abs(next(e.t for e in out if e.kind == E.TOOL_CALL) - (10.0 + call_t)) < 1e-6


def test_a_replay_of_other_inputs_fails_loudly_and_says_what_differs() -> None:
    cassette, _ = record(talker(), inputs())
    for change, part in (
        ({"seed": 8}, "seed"),
        ({"say": []}, "say"),
        ({"model": "other"}, "model"),
    ):
        with pytest.raises(C.CassetteMismatch, match=part):
            C.ReplayTransport(cassette, inputs(**change))


def test_a_session_that_opens_with_another_soul_model_or_tools_fails_loudly() -> None:
    cassette, _ = record(talker(), inputs())
    for kwargs in ({"model": "gemini-3.8-live-extended-thinking"}, {}):
        replay = C.ReplayTransport(cassette, inputs())
        rig = Rig(replay)
        rig.session.config = S.SessionConfig(**kwargs)
        rig.session._soul = "ANOTHER SOUL" if not kwargs else SOUL
        rig.run(0, until=1.0)
        assert replay.failure is not None and isinstance(replay.failure, C.CassetteMismatch)
        assert (
            rig.session.faults >= 1
        )  # (the session survived it; the harness re-raises the failure)


def test_a_tool_declared_differently_is_a_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    cassette, _ = record(talker(), inputs())
    monkeypatch.setattr("duckmind.live.cassette.schema_hash", lambda: "changed")
    replay = C.ReplayTransport(cassette, inputs())
    rig = Rig(replay)
    rig.run(0, until=0.5)
    assert replay.failure is not None and "tool_schema" in str(replay.failure)


def test_a_recorded_close_ends_the_replay_until_the_session_opens_again() -> None:
    fake = FakeGemini(drops=[3.0])
    recorder = C.RecordingTransport(fake, "u", inputs())
    rig = Rig(recorder)
    rig.run(0, until=8.0)
    cassette = recorder.cassette(complete=True)
    assert [e.kind for e in cassette.events].count(E.OPENED) == 2 and E.CLOSED in [
        e.kind for e in cassette.events
    ]
    replayed = Rig(C.ReplayTransport(cassette, inputs()))
    replayed.run(0, until=8.0)
    assert replayed.session.state == S.ONLINE and replayed.session.status()["connections"] == 2


def test_a_cassette_not_from_this_clock_still_replays_from_its_own_zero() -> None:
    cassette, recorded = record(talker(), inputs())
    rig = Rig(C.ReplayTransport(cassette, inputs()))
    rig.clock = ManualClock()
    rig.clock.set(100.0)
    rig.session.clock = rig.clock
    rig.run(0, until=100.0 + 1.0)
    rig.run(0, pcm=LOUD, until=100.0 + 2.0)
    rig.run(0, until=100.0 + 5.0)
    assert [c.name for c in rig.session.peek_calls()] == [c[0] for c in summary(recorded)]
