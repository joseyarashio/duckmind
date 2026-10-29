from duckmind.body import RobotObservation, SafetyObservation
from duckmind.replay import ReplayExpectation, ReplayValidator


def obs(at: float, policy: str, applied: float = 0.0) -> RobotObservation:
    return RobotObservation(
        monotonic_s=at,
        policy=policy,
        movement_requested=(applied, 0.0, 0.0),
        movement_applied=(applied, 0.0, 0.0),
        limited_by=(),
        safety=SafetyObservation(fallen=False, limp=False),
        backend="replay",
    )


def test_replay_reports_first_divergence_with_evidence() -> None:
    expected = ReplayExpectation(
        policies=("stand", "walk", "stand"),
        max_abs_vx_error=0.02,
    )
    actual = (obs(0.0, "stand"), obs(0.1, "polite-bow"), obs(0.2, "stand"))

    report = ReplayValidator().validate(expected, actual)

    assert not report.passed
    assert report.first_divergence_index == 1
    assert report.reasons == ("policy:expected=walk actual=polite-bow",)


def test_counterfactual_guard_rejects_unrelated_behavior_flip() -> None:
    validator = ReplayValidator()

    report = validator.compare_counterfactual(
        baseline_decision="approach_alex",
        counterfactual_decision="fear_alex",
        changed_fields={"battery": (0.72, 0.71)},
        allowed_influences={"fear_alex": frozenset({"threat", "relationship.alex.trust"})},
    )

    assert not report.passed
    assert report.reasons == ("decision_changed_without_allowed_cause:battery",)
