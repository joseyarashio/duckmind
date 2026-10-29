from datetime import datetime, timedelta, timezone

import pytest

from duckmind.core import DuckSelf, reduce_event, replay_events
from duckmind.provenance import Event, SourceRef

START = datetime(2026, 9, 24, tzinfo=timezone.utc)


def test_self_snapshot_cannot_be_mutated_through_live_state() -> None:
    state = DuckSelf.from_snapshot(
        {
            **DuckSelf.new(identity="duck", born_at=START).snapshot(),
            "preferences": {"preferences.alex.look_game": 0.6},
        }
    )
    with pytest.raises(TypeError):
        state.preferences["preferences.alex.look_game"] = 1.0
    snapshot = state.snapshot()
    snapshot["preferences"]["preferences.alex.look_game"] = 0.0
    assert state.preferences["preferences.alex.look_game"] == 0.6


def test_invalid_snapshot_is_rejected_before_replay() -> None:
    snapshot = DuckSelf.new(identity="duck", born_at=START).snapshot()
    snapshot["personality"]["sociability"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        DuckSelf.from_snapshot(snapshot)


def test_signed_valence_snapshot_roundtrips_without_weakening_fraction_checks() -> None:
    snapshot = DuckSelf.new(identity="duck", born_at=START).snapshot()
    snapshot["affect"]["valence"] = -0.4
    assert DuckSelf.from_snapshot(snapshot).affect.valence == -0.4
    snapshot["drives"]["rest_need"] = -0.1
    with pytest.raises(ValueError, match="rest_need"):
        DuckSelf.from_snapshot(snapshot)


def evt(index, kind, payload, *, source="touch_sensor", reliability=1.0):
    at = START + timedelta(seconds=index)
    return Event(
        event_id=f"event-{index}",
        kind=kind,
        occurred_at=at,
        observed_at=at,
        source=SourceRef(kind=source, source_id="test", reliability=reliability),
        trace_id="trace-core",
        payload=payload,
    )


def test_repeated_touch_changes_transient_affect_but_not_personality() -> None:
    initial = DuckSelf.new(identity="duck-a", born_at=START)
    events = [evt(i, "touch", {"actor": "alex"}) for i in range(100)]
    grown = replay_events(initial, events)

    assert grown.affect.valence <= 1.0
    assert grown.affect.arousal <= 1.0
    assert grown.personality == initial.personality
    assert grown.relationships == initial.relationships


def test_clock_decay_is_explicit_and_absence_does_not_punish_family() -> None:
    initial = DuckSelf.new(identity="duck-a", born_at=START)
    excited = reduce_event(initial, evt(1, "touch", {"actor": "alex"}))
    decayed = reduce_event(
        excited,
        evt(
            2,
            "time_passed",
            {"elapsed_s": 3600, "asleep": False},
            source="runtime_clock",
        ),
    )

    assert decayed.affect.valence < excited.affect.valence
    assert decayed.drives.social_need <= 0.65
    assert decayed.relationships == initial.relationships


def test_owner_stop_and_fallen_body_latch_play_off() -> None:
    initial = DuckSelf.new(identity="duck-a", born_at=START)
    enabled = reduce_event(initial, evt(1, "owner_play_enabled", {}, source="owner_control"))
    stopped = reduce_event(enabled, evt(2, "owner_stop", {}, source="owner_control"))
    assert enabled.play_enabled
    assert not stopped.play_enabled

    enabled = reduce_event(stopped, evt(3, "owner_play_enabled", {}, source="owner_control"))
    fallen = reduce_event(
        enabled,
        evt(
            4,
            "robot_state",
            {"battery_fraction": 0.8, "fallen": True, "limp": False},
            source="robotd_state",
        ),
    )
    assert fallen.fallen
    assert not fallen.play_enabled


def test_untrusted_actor_cannot_enable_play_or_commit_learning() -> None:
    initial = DuckSelf.new(identity="duck-a", born_at=START)
    with pytest.raises(ValueError, match="trusted source"):
        reduce_event(initial, evt(1, "owner_play_enabled", {}, source="speech_guess"))
    with pytest.raises(ValueError, match="trusted source"):
        reduce_event(
            initial,
            evt(
                2,
                "learning_committed",
                {
                    "update_id": "u-1",
                    "parameter": "preferences.alex.play",
                    "before": 0.5,
                    "after": 0.501,
                },
                source="llm_reflection",
            ),
        )


def test_snapshot_roundtrip_replays_to_the_same_state() -> None:
    initial = DuckSelf.new(identity="duck-a", born_at=START)
    events = [
        evt(1, "touch", {"actor": "alex"}),
        evt(
            2,
            "time_passed",
            {"elapsed_s": 120, "asleep": False},
            source="runtime_clock",
        ),
    ]
    grown = replay_events(initial, events)

    assert DuckSelf.from_snapshot(grown.snapshot()) == grown
    assert replay_events(initial, events) == grown
