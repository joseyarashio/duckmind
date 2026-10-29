from datetime import datetime, timedelta, timezone

from duckmind.behavior import BehaviorChoice, choose_invitation
from duckmind.core import DuckSelf, reduce_event
from duckmind.provenance import Event, SourceRef

NOW = datetime(2026, 9, 24, tzinfo=timezone.utc)


def event(index, kind, payload, source):
    at = NOW + timedelta(seconds=index)
    return Event(
        event_id=f"event-{index}",
        kind=kind,
        occurred_at=at,
        observed_at=at,
        source=SourceRef(kind=source, source_id="test", reliability=1.0),
        trace_id="trace-behavior",
        payload=payload,
    )


def invitation(index=2):
    return event(
        index,
        "invitation",
        {"actor": "alex", "activity": "look_game"},
        "owner_request",
    )


def enabled_state():
    return reduce_event(
        DuckSelf.new(identity="duck-a", born_at=NOW),
        event(1, "owner_play_enabled", {}, "owner_control"),
    )


def test_owner_enable_and_safety_gate_precede_preferences() -> None:
    disabled = DuckSelf.new(identity="duck-a", born_at=NOW)
    assert choose_invitation(disabled, invitation()).choice is BehaviorChoice.DECLINE

    enabled = enabled_state()
    low_battery = reduce_event(
        enabled,
        event(
            2,
            "robot_state",
            {"battery_fraction": 0.19, "fallen": False, "limp": False},
            "robotd_state",
        ),
    )
    assert choose_invitation(low_battery, invitation(3)).choice is BehaviorChoice.DECLINE


def test_learned_preference_changes_choice_without_changing_safety() -> None:
    initial = enabled_state()
    low_preference = DuckSelf.from_snapshot(
        {**initial.snapshot(), "preferences": {"preferences.alex.look_game": 0.0}}
    )
    high_preference = DuckSelf.from_snapshot(
        {**initial.snapshot(), "preferences": {"preferences.alex.look_game": 1.0}}
    )

    low = choose_invitation(low_preference, invitation())
    high = choose_invitation(high_preference, invitation())

    assert low.choice is BehaviorChoice.OBSERVE
    assert high.choice is BehaviorChoice.ENGAGE
    assert "preferences.alex.look_game" in high.influences


def test_cooldown_blocks_rapid_action_thrash() -> None:
    state = enabled_state()
    chosen = choose_invitation(state, invitation(10))
    decided = reduce_event(
        state,
        event(
            10,
            "behavior_decision",
            {"goal": "play:alex:look_game", "invitation_event_id": "event-10"},
            "system1",
        ),
    )
    rapid = choose_invitation(decided, invitation(12))

    assert chosen.choice in {BehaviorChoice.ENGAGE, BehaviorChoice.OBSERVE}
    assert rapid.choice is BehaviorChoice.CONTINUE


def test_unsupported_invitation_source_never_moves_the_body() -> None:
    untrusted = event(3, "invitation", {"actor": "alex", "activity": "look_game"}, "llm_guess")
    result = choose_invitation(enabled_state(), untrusted)

    assert result.choice is BehaviorChoice.DECLINE
    assert result.reason == "untrusted_invitation"


def test_repeated_cooldown_decisions_preserve_engagement_and_original_deadline() -> None:
    initial = enabled_state()
    state = DuckSelf.from_snapshot(
        {**initial.snapshot(), "preferences": {"preferences.alex.look_game": 1.0}}
    )
    for index, actor, expected in (
        (10, "alex", BehaviorChoice.ENGAGE),
        (11, "alex", BehaviorChoice.CONTINUE),
        (12, "alex", BehaviorChoice.CONTINUE),
        (13, "someone_else", BehaviorChoice.OBSERVE),
        (14, "alex", BehaviorChoice.CONTINUE),
    ):
        requested = event(
            index,
            "invitation",
            {"actor": actor, "activity": "look_game"},
            "owner_request",
        )
        chosen = choose_invitation(state, requested)
        assert chosen.choice is expected
        state = reduce_event(
            state,
            event(
                index,
                "behavior_decision",
                {
                    "goal": f"play:{actor}:look_game"
                    if chosen.choice is BehaviorChoice.ENGAGE
                    else None,
                    "invitation_event_id": requested.event_id,
                    "choice": chosen.choice.value,
                    "reason": chosen.reason,
                },
                "system1",
            ),
        )
        assert state.current_goal == "play:alex:look_game"
        assert state.last_decision_at == NOW + timedelta(seconds=10)
    assert choose_invitation(state, invitation(15)).choice is BehaviorChoice.ENGAGE
