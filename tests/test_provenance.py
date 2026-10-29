import math
from datetime import UTC, datetime

import pytest

from duckmind.provenance import (
    Event,
    LearningStatus,
    LearningUpdate,
    Outcome,
    OutcomeVerdict,
    SourceRef,
)

NOW = datetime(2026, 9, 23, tzinfo=UTC)


def test_learning_update_requires_outcome_evidence() -> None:
    with pytest.raises(ValueError, match="outcome evidence"):
        LearningUpdate(
            update_id="learn-1",
            episode_id="episode-1",
            parameter="relationship.alex.trust",
            before=0.5,
            after=0.501,
            max_abs_delta=0.002,
            rule_version="trust-v1",
            outcome_ids=(),
            status=LearningStatus.PROPOSED,
        )


def test_learning_update_enforces_slow_plasticity_bound() -> None:
    with pytest.raises(ValueError, match="plasticity"):
        LearningUpdate(
            update_id="learn-1",
            episode_id="episode-1",
            parameter="personality.sociability",
            before=0.4,
            after=0.5,
            max_abs_delta=0.002,
            rule_version="trait-v1",
            outcome_ids=("outcome-1",),
            status=LearningStatus.PROPOSED,
        )


def test_event_and_outcome_keep_traceable_evidence() -> None:
    event = Event(
        event_id="event-1",
        kind="robot.state",
        occurred_at=NOW,
        observed_at=NOW,
        source=SourceRef(kind="robotd", source_id="duck-a", reliability=1.0),
        trace_id="trace-1",
        payload={"policy": "polite-bow"},
    )
    outcome = Outcome(
        outcome_id="outcome-1",
        episode_id="episode-1",
        verdict=OutcomeVerdict.SUCCESS,
        evaluator_version="bow-v1",
        evidence_event_ids=(event.event_id,),
        metrics={"duration_s": 3.8},
    )

    assert outcome.evidence_event_ids == ("event-1",)
    assert event.source.source_id == "duck-a"


def test_nonfinite_learning_cannot_bypass_delta_bound() -> None:
    with pytest.raises(ValueError, match="finite"):
        LearningUpdate(
            update_id="learn-nan",
            episode_id="episode-1",
            parameter="personality.sociability",
            before=0.4,
            after=math.nan,
            max_abs_delta=0.005,
            rule_version="trait-v1",
            outcome_ids=("outcome-1",),
            status=LearningStatus.COMMITTED,
        )


def test_duplicate_evidence_cannot_count_twice() -> None:
    with pytest.raises(ValueError, match="unique"):
        Outcome(
            outcome_id="outcome-dup",
            episode_id="episode-1",
            verdict=OutcomeVerdict.SUCCESS,
            evaluator_version="test-v1",
            evidence_event_ids=("event-1", "event-1"),
        )


def test_event_payload_is_defensively_frozen() -> None:
    original = {"actor": "alex", "context": {"tags": ["play"]}}
    event = Event(
        event_id="event-frozen",
        kind="touch",
        occurred_at=NOW,
        observed_at=NOW,
        source=SourceRef("touch_sensor", "head", 1.0),
        trace_id="trace-frozen",
        payload=original,
    )
    original["context"]["tags"].append("forged")
    assert event.payload["context"]["tags"] == ("play",)
    with pytest.raises(TypeError):
        event.payload["actor"] = "someone_else"
    with pytest.raises(TypeError):
        event.payload["context"]["tags"] = ("forged",)
