import json
from dataclasses import replace

import pytest

from duckmind.body import BodyAck, BodyIntent, RobotObservation, SafetyObservation
from duckmind.pet_body_trace import BodyTraceRecorder, replay_front_gaze_trace
from duckmind.skills import EmbodiedSkillLayer, SkillPhase


def _observation(at: float, error: float) -> RobotObservation:
    return RobotObservation(
        monotonic_s=at,
        robot_time_s=at,
        policy="walk",
        movement_requested=(0.0, 0.0, 0.0),
        movement_applied=(0.0, 0.0, 0.0),
        limited_by=(),
        safety=SafetyObservation(fallen=False, limp=False),
        backend="simulator",
        head_tracking_error_rad=error,
    )


def _record_completed_gaze(path):
    now = [1.0]

    class FakeBody:
        def __init__(self):
            self.frames = iter(
                (_observation(1.1, 0.20), _observation(1.2, 0.10), _observation(1.3, 0.08))
            )

        def dispatch_intent(self, intent: BodyIntent) -> BodyAck:
            assert intent.method == "robot.head"
            return BodyAck(True, "sent_unconfirmed", confirmed=False)

        def observe(self) -> RobotObservation:
            frame = next(self.frames)
            now[0] = frame.monotonic_s
            return frame

        def stop(self) -> BodyAck:
            return BodyAck(True)

    recorder = BodyTraceRecorder(
        FakeBody(),
        interaction_id="sim-interaction",
        source_sha256="a" * 64,
        clock=lambda: now[0],
    )
    status = EmbodiedSkillLayer(recorder, clock=lambda: now[0]).perform_front_gaze()
    recorder.save(path, status=status)
    return status


def test_recorded_front_gaze_replays_through_actual_skill_layer(tmp_path) -> None:
    path = tmp_path / "gaze.body.json"
    status = _record_completed_gaze(path)

    assert status.phase is SkillPhase.COMPLETED
    report = replay_front_gaze_trace(path)
    assert report.passed, report.reasons
    record = json.loads(path.read_text())
    assert [step["type"] for step in record["steps"]] == [
        "dispatch",
        "observation",
        "observation",
        "observation",
    ]
    assert all("oracle" not in step and "joints" not in step for step in record["steps"])


@pytest.mark.parametrize(
    "attack",
    ["head_target", "missing_frame", "oracle", "fake_result", "policy_switch", "motion"],
)
def test_front_gaze_replay_rejects_tampered_or_incomplete_trace(tmp_path, attack) -> None:
    path = tmp_path / "gaze.body.json"
    _record_completed_gaze(path)
    record = json.loads(path.read_text())
    if attack == "head_target":
        record["steps"][0]["params"]["head_pitch"] = 0.9
    elif attack == "missing_frame":
        record["steps"].pop()
    elif attack == "oracle":
        record["steps"][1]["oracle"] = {"trunk_z": 0.1}
    elif attack == "policy_switch":
        record["steps"][3]["policy"] = "kick_left"
    elif attack == "motion":
        record["steps"][3]["move"]["requested"][0] = 0.2
    else:
        record["result"]["phase"] = "refused"
    path.write_text(json.dumps(record))

    report = replay_front_gaze_trace(path)

    assert not report.passed


def test_recording_rejects_simulator_oracle_in_body_observation(tmp_path) -> None:
    class FakeBody:
        def observe(self):
            return replace(_observation(1.1, 0.1), metadata={"oracle": {"trunk_z": 0.1}})

    recorder = BodyTraceRecorder(
        FakeBody(),
        interaction_id="sim-interaction",
        source_sha256="a" * 64,
        clock=lambda: 1.0,
    )
    with pytest.raises(ValueError, match="simulator-only metadata"):
        recorder.observe()
