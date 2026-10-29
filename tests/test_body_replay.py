import json
from dataclasses import replace
from pathlib import Path

import pytest

from duckmind.body import BodyAck, BodyIntent, RobotObservation, SafetyObservation
from duckmind.body_replay import (
    RecordedBodyPort,
    ReplayDispatch,
    ReplayFrame,
    ReplayMismatch,
    ReplayStop,
    replay_named_skill_trace,
)
from tests.checkout_parts import needs_evidence

TRACE = Path(__file__).resolve().parents[1] / "docs/validation/evidence/official-v5-kick-left.jsonl"


def _frame(at: float) -> RobotObservation:
    return RobotObservation(
        monotonic_s=at,
        policy="walk",
        movement_requested=(0.0, 0.0, 0.0),
        movement_applied=(0.0, 0.0, 0.0),
        limited_by=(),
        safety=SafetyObservation(fallen=False, limp=False),
        backend="replay",
    )


def test_recorded_body_port_requires_exact_command_order_and_consumption() -> None:
    intent = BodyIntent("robot.head", {"head_pitch": 0.15})
    body = RecordedBodyPort(
        (
            ReplayDispatch(intent, BodyAck(True, "sent_unconfirmed", confirmed=False)),
            ReplayFrame(_frame(1.0)),
            ReplayStop(BodyAck(True)),
        ),
        initial_time=0.0,
    )

    assert body.dispatch_intent(intent).confirmed is False
    assert body.observe().monotonic_s == body.clock()
    assert body.stop().accepted
    body.assert_consumed()
    with pytest.raises(ReplayMismatch, match="exhausted"):
        body.observe()

    bad_order = RecordedBodyPort((ReplayFrame(_frame(1.0)),), initial_time=0.0)
    with pytest.raises(ReplayMismatch, match="expected observation"):
        bad_order.stop()
    with pytest.raises(ReplayMismatch, match="earlier mismatch"):
        bad_order.assert_consumed()


def test_recorded_body_port_rejects_rewritten_intent() -> None:
    body = RecordedBodyPort(
        (ReplayDispatch(BodyIntent("robot.do", {"skill": "kick_left"}), BodyAck(True)),),
        initial_time=0.0,
    )
    with pytest.raises(ReplayMismatch, match="intent mismatch"):
        body.dispatch_intent(BodyIntent("robot.do", {"skill": "kick_right"}))
    with pytest.raises(ReplayMismatch, match="earlier mismatch"):
        body.assert_consumed()


def test_recorded_body_port_rejects_missing_frames() -> None:
    body = RecordedBodyPort((ReplayFrame(_frame(1.0)),), initial_time=0.0)
    with pytest.raises(ReplayMismatch, match="unconsumed"):
        body.assert_consumed()


def test_recorded_body_port_rejects_nonmonotonic_observations() -> None:
    body = RecordedBodyPort((ReplayFrame(_frame(1.0)),), initial_time=1.0)
    with pytest.raises(ReplayMismatch, match="not increasing"):
        body.observe()
    with pytest.raises(ReplayMismatch, match="earlier mismatch"):
        body.assert_consumed()


def test_replay_frame_does_not_expose_simulator_oracle_to_skills() -> None:
    with pytest.raises(ValueError, match="simulator-only metadata"):
        ReplayFrame(replace(_frame(1.0), metadata={"oracle": {"trunk_z": 0.1}}))


@needs_evidence
def test_official_trace_replays_through_real_skill_layer_and_body_contract() -> None:
    report = replay_named_skill_trace(TRACE)
    assert report.passed, report.reasons


@needs_evidence
def test_named_skill_replay_rejects_trace_with_wrong_recorded_phase(tmp_path) -> None:
    rows = [json.loads(line) for line in TRACE.read_text().splitlines()]
    next(row for row in rows if row.get("type") == "sample")["phase"] = "completed"
    path = tmp_path / "tampered.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    report = replay_named_skill_trace(path)

    assert not report.passed
    assert "phase_diverged" in report.reasons
