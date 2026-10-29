from dataclasses import replace

import pytest

from duckmind.body import BodyAck, BodyIntent, RobotObservation, SafetyObservation
from duckmind.skills import (
    DispatchAck,
    EmbodiedSkillLayer,
    OneShotSkillIntent,
    SkillPhase,
    SkillTracker,
)


def observation(at: float, policy: str, *, fallen: bool = False) -> RobotObservation:
    return RobotObservation(
        monotonic_s=at,
        policy=policy,
        movement_requested=(0.0, 0.0, 0.0),
        movement_applied=(0.0, 0.0, 0.0),
        limited_by=(),
        safety=SafetyObservation(fallen=fallen, limp=False),
    )


def test_rpc_acceptance_is_not_skill_completion() -> None:
    tracker = SkillTracker(
        execution_id="exec-1",
        expected_policy="polite-bow",
        created_at=0.0,
        start_timeout_s=1.0,
        completion_timeout_s=5.0,
    )

    accepted = tracker.on_dispatch(DispatchAck(accepted=True), at=0.01)

    assert accepted.phase is SkillPhase.ACCEPTED
    assert not accepted.terminal


def test_skill_completes_only_after_observed_start_and_exit() -> None:
    tracker = SkillTracker(
        execution_id="exec-1",
        expected_policy="polite-bow",
        created_at=0.0,
        start_timeout_s=1.0,
        completion_timeout_s=5.0,
    )
    tracker.on_dispatch(DispatchAck(accepted=True), at=0.01)

    started = tracker.on_observation(observation(0.2, "polite-bow"))
    completed = tracker.on_observation(observation(3.9, "stand"))

    assert started.phase is SkillPhase.STARTED
    assert completed.phase is SkillPhase.COMPLETED
    assert completed.terminal


def test_accepted_skill_times_out_when_robotd_never_reports_start() -> None:
    tracker = SkillTracker(
        execution_id="exec-1",
        expected_policy="polite-bow",
        created_at=0.0,
        start_timeout_s=0.5,
        completion_timeout_s=5.0,
    )
    tracker.on_dispatch(DispatchAck(accepted=True), at=0.01)

    timed_out = tracker.on_tick(at=0.6)

    assert timed_out.phase is SkillPhase.TIMED_OUT
    assert timed_out.reason == "start_timeout"


def test_refused_dispatch_is_terminal_and_preserves_reason() -> None:
    tracker = SkillTracker(
        execution_id="exec-1",
        expected_policy="polite-bow",
        created_at=0.0,
        start_timeout_s=1.0,
        completion_timeout_s=5.0,
    )

    refused = tracker.on_dispatch(
        DispatchAck(accepted=False, reason="a scripted move is already running"),
        at=0.01,
    )

    assert refused.phase is SkillPhase.REFUSED
    assert refused.terminal
    assert refused.reason == "a scripted move is already running"


def test_skill_exit_to_held_is_not_completion() -> None:
    tracker = SkillTracker(
        execution_id="exec-2",
        expected_policy="kick_left",
        created_at=0.0,
        start_timeout_s=1.0,
        completion_timeout_s=5.0,
    )
    tracker.on_dispatch(DispatchAck(accepted=True), at=0.01)
    tracker.on_observation(observation(0.1, "kick_left"))

    status = tracker.on_observation(observation(0.2, "held"))

    assert status.phase is SkillPhase.INTERRUPTED
    assert status.reason == "unexpected_policy:held"


def test_fall_interrupts_skill_even_if_policy_still_matches() -> None:
    tracker = SkillTracker(
        execution_id="exec-3",
        expected_policy="kick_left",
        created_at=0.0,
        start_timeout_s=1.0,
        completion_timeout_s=5.0,
    )
    tracker.on_dispatch(DispatchAck(accepted=True), at=0.01)
    tracker.on_observation(observation(0.1, "kick_left"))

    status = tracker.on_observation(observation(0.2, "kick_left", fallen=True))

    assert status.phase is SkillPhase.INTERRUPTED
    assert status.reason == "robot_safety_state"


def test_embodied_skill_layer_is_the_only_translator_of_named_skill() -> None:
    class FakeBody:
        def __init__(self) -> None:
            self.intents: list[BodyIntent] = []
            self.states = [observation(0.2, "kick_left"), observation(0.8, "walk")]

        def dispatch_intent(self, intent: BodyIntent) -> BodyAck:
            self.intents.append(intent)
            return BodyAck(True)

        def observe(self) -> RobotObservation:
            return self.states.pop(0)

        def stop(self) -> BodyAck:
            return BodyAck(True)

    fake = FakeBody()
    now = [0.0]
    layer = EmbodiedSkillLayer(fake, clock=lambda: now[0])
    execution = layer.start_one_shot(OneShotSkillIntent("intent-1", "kick_left"))
    with pytest.raises(RuntimeError, match="still active"):
        layer.start_one_shot(OneShotSkillIntent("intent-overlap", "kick_right"))
    now[0] = 0.2
    _, first = execution.poll()
    now[0] = 0.8
    _, second = execution.poll()

    assert fake.intents == [BodyIntent("robot.do", {"skill": "kick_left"})]
    assert first.phase is SkillPhase.STARTED
    assert second.phase is SkillPhase.COMPLETED


def test_sit_toggle_requires_its_own_multiphase_skill_adapter() -> None:
    class NoCallBody:
        def dispatch_intent(self, intent):
            pytest.fail("sit_toggle must be refused before dispatch")

    with pytest.raises(ValueError, match="multi-phase"):
        EmbodiedSkillLayer(NoCallBody()).start_one_shot(
            OneShotSkillIntent("intent-2", "sit_toggle")
        )


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("other_policy", "unexpected_policy:kick_left"),
        ("requested_motion", "motion_command_active"),
        ("applied_motion", "motion_command_active"),
        ("nan_motion", "motion_command_active"),
    ],
)
def test_front_gaze_cannot_complete_after_policy_or_motion_changes(fault, reason) -> None:
    now = [0.0]
    first = replace(observation(0.1, "walk"), head_tracking_error_rad=0.03)
    second = replace(observation(0.2, "walk"), head_tracking_error_rad=0.02)
    if fault == "other_policy":
        second = replace(second, policy="kick_left")
    elif fault == "requested_motion":
        second = replace(second, movement_requested=(0.2, 0.0, 0.0))
    elif fault == "nan_motion":
        second = replace(second, movement_requested=(float("nan"), 0.0, 0.0))
    else:
        second = replace(second, movement_applied=(0.2, 0.0, 0.0))

    class ChangingBody:
        def __init__(self):
            self.frames = iter((first, second))

        def dispatch_intent(self, intent):
            assert intent.method == "robot.head"
            return BodyAck(True, "sent_unconfirmed", confirmed=False)

        def observe(self):
            frame = next(self.frames)
            now[0] = frame.monotonic_s
            return frame

    status = EmbodiedSkillLayer(ChangingBody(), clock=lambda: now[0]).perform_front_gaze()
    assert status.phase is SkillPhase.INTERRUPTED
    assert status.reason == reason


@pytest.mark.parametrize("late_start", [True, False])
def test_late_policy_transition_cannot_override_execution_deadline(late_start) -> None:
    now = [0.0]
    policy = ["kick_left"]

    class DelayedBody:
        def dispatch_intent(self, intent):
            return BodyAck(True)

        def observe(self):
            return observation(now[0], policy[0])

        def stop(self):
            return BodyAck(True)

    execution = EmbodiedSkillLayer(DelayedBody(), clock=lambda: now[0]).start_one_shot(
        OneShotSkillIntent("intent-late", "kick_left"),
        start_timeout_s=1.0,
        completion_timeout_s=5.0,
    )
    if not late_start:
        now[0] = 0.2
        assert execution.poll()[1].phase is SkillPhase.STARTED
        policy[0] = "walk"
    now[0] = 10.0
    status = execution.poll()[1]
    assert status.phase is SkillPhase.TIMED_OUT
    assert status.reason == ("start_timeout" if late_start else "completion_timeout")
