import inspect

from duckmind.body import BodyPort, RobotObservation, SafetyObservation
from duckmind.cognition import CognitiveProposal, GroundedFact, route_proposal


def test_body_port_exposes_no_servo_or_joint_command() -> None:
    public = {
        name
        for name, _ in inspect.getmembers(BodyPort, inspect.isfunction)
        if not name.startswith("_")
    }

    assert public == {"dispatch_intent", "observe", "stop"}


def test_system2_cannot_ground_a_proposal_in_an_unknown_fact() -> None:
    proposal = CognitiveProposal(
        intent="approach",
        confidence=0.8,
        fact_ids=("memory-invented",),
        planner="system2",
    )
    facts = (GroundedFact(fact_id="memory-real", statement="Mum is near the door"),)

    decision = route_proposal(proposal, facts=facts, system2_threshold=0.6)

    assert not decision.accepted
    assert decision.reason == "ungrounded_fact:memory-invented"


def test_observation_is_high_level_and_backend_neutral() -> None:
    state = RobotObservation(
        monotonic_s=1.0,
        policy="walk",
        movement_requested=(0.2, 0.0, 0.0),
        movement_applied=(0.1, 0.0, 0.0),
        limited_by=("deadman",),
        safety=SafetyObservation(fallen=False, limp=False),
        backend="simulator",
    )

    assert state.backend == "simulator"
    assert state.limited_by == ("deadman",)
