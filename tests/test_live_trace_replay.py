import json
from pathlib import Path

import pytest

from duckmind.replay import ReplayValidator
from tests.checkout_parts import needs_evidence

pytestmark = needs_evidence

TRACE = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "validation"
    / "evidence"
    / "official-v5-kick-left.jsonl"
)


def records():
    return [json.loads(line) for line in TRACE.read_text().splitlines()]


def test_official_robotd_mujoco_trace_replays() -> None:
    assert ReplayValidator().validate_skill_trace(TRACE).passed
    outcome = next(record for record in records() if record["type"] == "outcome")
    assert outcome["execution_gate"] == "pass"
    assert outcome["verdict"] == "unknown"  # no ball/contact goal in this scene


def test_simulator_oracle_cannot_leak_into_skill_observation() -> None:
    sample = next(record for record in records() if record["type"] == "sample")
    projected = ReplayValidator.project_robot_observation(sample)

    assert "oracle" in sample
    assert projected.metadata is None
    assert not hasattr(projected, "trunk_z")


def test_malformed_robot_state_cannot_be_projected_from_trace() -> None:
    sample = next(record for record in records() if record["type"] == "sample")
    sample["move"]["applied"] = [True, 0.0, 0.0]
    with pytest.raises(ValueError, match="applied movement"):
        ReplayValidator.project_robot_observation(sample)


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda rows: rows[-2].update(verdict="success"), "outcome_verdict_mismatch"),
        (
            lambda rows: next(row for row in rows if row["type"] == "sample").update(policy="held"),
            "phase_diverged",
        ),
        (
            lambda rows: next(row for row in rows if row["type"] == "sample")["oracle"].update(
                trunk_z=0.0
            ),
            "trunk_height_metric_mismatch",
        ),
        (
            lambda rows: next(row for row in rows if row["type"] == "sample").update(
                event_id="invented"
            ),
            "event_order_or_id_mismatch",
        ),
    ],
)
def test_adversarial_trace_tampering_is_detected(tmp_path, mutate, expected) -> None:
    rows = records()
    mutate(rows)
    path = tmp_path / "tampered.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    report = ReplayValidator().validate_skill_trace(path)

    assert not report.passed
    assert expected in report.reasons
