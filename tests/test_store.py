from datetime import datetime, timedelta, timezone

import pytest

from duckmind.provenance import (
    Episode,
    Event,
    LearningStatus,
    LearningUpdate,
    Outcome,
    OutcomeVerdict,
    SourceRef,
)
from duckmind.store import SQLiteEventStore

NOW = datetime(2026, 9, 24, tzinfo=timezone.utc)


def event(event_id: str = "event-1", payload=None) -> Event:
    return Event(
        event_id=event_id,
        kind="touch",
        occurred_at=NOW,
        observed_at=NOW,
        source=SourceRef(kind="touch_sensor", source_id="head", reliability=0.9),
        trace_id="trace-1",
        payload=payload or {"actor": "alex"},
    )


def closed_episode() -> Episode:
    return Episode(
        episode_id="episode-1",
        trace_id="trace-1",
        started_at=NOW,
        ended_at=NOW + timedelta(seconds=5),
        event_ids=("event-1",),
        intent_ids=("intent-1",),
        outcome_ids=("outcome-1",),
        runtime_versions={"duckmind": "0.1.0"},
    )


def outcome(verdict=OutcomeVerdict.SUCCESS) -> Outcome:
    return Outcome(
        outcome_id="outcome-1",
        episode_id="episode-1",
        verdict=verdict,
        evaluator_version="test-v1",
        evidence_event_ids=("event-1",),
        metrics={"feedback": 1.0},
    )


def learning_update() -> LearningUpdate:
    return LearningUpdate(
        update_id="update-1",
        episode_id="episode-1",
        parameter="preferences.alex.play",
        before=0.5,
        after=0.503,
        max_abs_delta=0.005,
        rule_version="bounded-v1",
        outcome_ids=("outcome-1",),
        status=LearningStatus.COMMITTED,
    )


def learning_event() -> Event:
    return Event(
        event_id="learning:update-1",
        kind="learning_committed",
        occurred_at=NOW + timedelta(seconds=5),
        observed_at=NOW + timedelta(seconds=5),
        source=SourceRef(kind="learning_engine", source_id="bounded-v1", reliability=1.0),
        trace_id="trace-1",
        payload={
            "update_id": "update-1",
            "parameter": "preferences.alex.play",
            "before": 0.5,
            "after": 0.503,
            "evidence_event_ids": ["event-1"],
        },
        parent_event_ids=("event-1",),
    )


def test_event_log_survives_reopen_and_rejects_id_conflicts(tmp_path) -> None:
    path = tmp_path / "duckmind.sqlite3"
    with SQLiteEventStore(path) as store:
        assert store.append_event(event())
        assert not store.append_event(event())
        with pytest.raises(ValueError, match="conflicting event_id"):
            store.append_event(event(payload={"actor": "someone_else"}))

    with SQLiteEventStore(path) as store:
        entries = store.list_events()
        assert len(entries) == 1
        assert entries[0].event == event()
        assert entries[0].seq == 1


def test_nested_payload_round_trips_without_caller_mutation(tmp_path) -> None:
    original = {"actor": "alex", "context": {"tags": ["play"]}}
    captured = event(payload=original)
    original["context"]["tags"].append("later")
    with SQLiteEventStore(tmp_path / "pet.sqlite3") as store:
        store.append_event(captured)
        assert store.event_by_id(captured.event_id) == captured
        assert store.event_by_id(captured.event_id).payload["context"]["tags"] == ("play",)


def test_closed_episode_requires_real_event_and_atomic_outcome(tmp_path) -> None:
    with SQLiteEventStore(tmp_path / "duckmind.sqlite3") as store:
        with pytest.raises(ValueError, match="missing event"):
            store.commit_episode(closed_episode(), outcome())
        assert store.count("episodes") == 0
        assert store.count("outcomes") == 0

        store.append_event(event())
        assert store.commit_episode(closed_episode(), outcome())
        assert not store.commit_episode(closed_episode(), outcome())
        assert store.count("episodes") == 1
        assert store.count("outcomes") == 1


def test_episode_rejects_event_outside_time_bounds(tmp_path) -> None:
    with SQLiteEventStore(tmp_path / "duckmind.sqlite3") as store:
        late = Event(
            event_id="event-1",
            kind="touch",
            occurred_at=NOW + timedelta(seconds=10),
            observed_at=NOW + timedelta(seconds=10),
            source=SourceRef(kind="touch_sensor", source_id="head", reliability=0.9),
            trace_id="trace-1",
            payload={"actor": "alex"},
        )
        store.append_event(late)
        with pytest.raises(ValueError, match="chronological bounds"):
            store.commit_episode(closed_episode(), outcome())


def test_learning_needs_supported_outcome_and_cannot_duplicate(tmp_path) -> None:
    with SQLiteEventStore(tmp_path / "duckmind.sqlite3") as store:
        store.append_event(event())
        store.commit_episode(closed_episode(), outcome())
        with pytest.raises(ValueError, match="atomic"):
            store.append_learning_update(learning_update())
        assert store.commit_learning_with_event(learning_update(), learning_event())
        assert not store.commit_learning_with_event(learning_update(), learning_event())
        duplicate = LearningUpdate(
            update_id="update-2",
            episode_id="episode-1",
            parameter="preferences.alex.play",
            before=0.503,
            after=0.504,
            max_abs_delta=0.005,
            rule_version="bounded-v1",
            outcome_ids=("outcome-1",),
            status=LearningStatus.COMMITTED,
        )
        with pytest.raises(ValueError, match="already updated"):
            store.commit_learning_with_event(
                duplicate,
                Event(
                    event_id="learning:update-2",
                    kind="learning_committed",
                    occurred_at=NOW + timedelta(seconds=5),
                    observed_at=NOW + timedelta(seconds=5),
                    source=SourceRef(
                        kind="learning_engine", source_id="bounded-v1", reliability=1.0
                    ),
                    trace_id="trace-1",
                    payload={
                        "update_id": "update-2",
                        "parameter": "preferences.alex.play",
                        "before": 0.503,
                        "after": 0.504,
                        "evidence_event_ids": ["event-1"],
                    },
                    parent_event_ids=("event-1",),
                ),
            )

    with SQLiteEventStore(tmp_path / "unknown.sqlite3") as store:
        store.append_event(event())
        store.commit_episode(closed_episode(), outcome(OutcomeVerdict.UNKNOWN))
        with pytest.raises(ValueError, match="supported outcome"):
            store.commit_learning_with_event(learning_update(), learning_event())


def test_outbox_is_atomic_with_event_and_survives_restart(tmp_path) -> None:
    path = tmp_path / "duckmind.sqlite3"
    with SQLiteEventStore(path) as store:
        store.append_event(
            event(),
            memory_job={
                "item_id": "job-1",
                "payload": {
                    "kind": "autobiographical",
                    "text": "Owner greeted duck.",
                    "source_event_ids": ["event-1"],
                    "reliability": 0.8,
                    "approved_for_external_memory": True,
                },
            },
        )
        assert store.count("outbox") == 1
    with SQLiteEventStore(path) as store:
        pending = store.due_memory_jobs(at=NOW + timedelta(seconds=1))
        assert len(pending) == 1
        assert pending[0].source_event_id == "event-1"
        assert pending[0].payload == {
            "kind": "autobiographical",
            "text": "Owner greeted duck.",
            "source_event_ids": ["event-1"],
            "reliability": 0.8,
            "approved_for_external_memory": True,
        }
        store.fail_memory_job("job-1", "offline", next_at=NOW + timedelta(minutes=1))
        assert store.due_memory_jobs(at=NOW + timedelta(seconds=1)) == []
        assert len(store.due_memory_jobs(at=NOW + timedelta(minutes=2))) == 1
        store.complete_memory_job("job-1")
        assert store.due_memory_jobs(at=NOW + timedelta(minutes=2)) == []


def test_snapshot_compare_and_swap_guards_recovery(tmp_path) -> None:
    with SQLiteEventStore(tmp_path / "duckmind.sqlite3") as store:
        store.append_event(event())
        first = store.save_snapshot("self", {"valence": 0.1}, expected_revision=0, last_event_seq=1)
        assert first.revision == 1
        assert store.load_snapshot("self").state == {"valence": 0.1}
        with pytest.raises(ValueError, match="snapshot revision"):
            store.save_snapshot("self", {"valence": 0.2}, expected_revision=0, last_event_seq=1)
