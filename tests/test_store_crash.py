import sqlite3
import subprocess
import sys
import textwrap
from datetime import UTC, datetime

import pytest

from duckmind.provenance import Event, SourceRef
from duckmind.runtime import DuckRuntime
from duckmind.store import SCHEMA_VERSION, SQLiteEventStore

CRASH_DURING_SCHEMA = textwrap.dedent(
    """
    import os
    import sqlite3
    import sys

    from duckmind.store import SQLiteEventStore

    connect = sqlite3.connect

    class CrashConnection(sqlite3.Connection):
        def executescript(self, sql):
            def trace(statement):
                if statement.strip().startswith(sys.argv[2]):
                    os._exit(71)
            self.set_trace_callback(trace)
            return super().executescript(sql)

    sqlite3.connect = lambda *args, **kwargs: connect(
        *args, **kwargs, factory=CrashConnection
    )
    SQLiteEventStore(sys.argv[1])
    """
)


# SQLite's WAL commit record is the visibility boundary. Kill the writer just
# after individual SQL statements or the outer COMMIT, never close it cleanly.
# https://www.sqlite.org/wal.html#how_wal_works
CRASH_DURING_LEARNING = textwrap.dedent(
    """
    import os
    import sqlite3
    import sys
    from datetime import UTC, datetime, timedelta

    from duckmind.provenance import (
        Episode, Event, LearningStatus, LearningUpdate, Outcome, OutcomeVerdict, SourceRef,
    )
    from duckmind.store import SQLiteEventStore

    path, mode = sys.argv[1:3]
    cut = sys.argv[3] if len(sys.argv) > 3 else ""
    at = datetime(2026, 9, 24, tzinfo=UTC)
    evidence = Event(
        event_id="evidence-1", kind="touch", occurred_at=at, observed_at=at,
        source=SourceRef("touch_sensor", "head", 0.9), trace_id="trace-1",
        payload={"actor": "alex"},
    )
    episode = Episode(
        episode_id="episode-1", trace_id="trace-1", started_at=at,
        ended_at=at + timedelta(seconds=5), event_ids=("evidence-1",),
        intent_ids=("intent-1",), outcome_ids=("outcome-1",),
        runtime_versions={"duckmind": "0.1.0"},
    )
    outcome = Outcome(
        outcome_id="outcome-1", episode_id="episode-1",
        verdict=OutcomeVerdict.SUCCESS, evaluator_version="test-v1",
        evidence_event_ids=("evidence-1",), metrics={"feedback": 1.0},
    )
    update = LearningUpdate(
        update_id="update-1", episode_id="episode-1",
        parameter="preferences.alex.play", before=0.5, after=0.503,
        max_abs_delta=0.005, rule_version="bounded-v1",
        outcome_ids=("outcome-1",), status=LearningStatus.COMMITTED,
    )
    learned = Event(
        event_id="learning:update-1", kind="learning_committed",
        occurred_at=at + timedelta(seconds=5),
        observed_at=at + timedelta(seconds=5),
        source=SourceRef("learning_engine", "bounded-v1", 1.0),
        trace_id="trace-1",
        payload={
            "update_id": "update-1", "parameter": "preferences.alex.play",
            "before": 0.5, "after": 0.503,
            "evidence_event_ids": ["evidence-1"],
        },
        parent_event_ids=("evidence-1",),
    )

    if mode == "learn":
        connect = sqlite3.connect

        class CrashConnection(sqlite3.Connection):
            def execute(self, sql, parameters=()):
                result = super().execute(sql, parameters)
                statement = sql.strip().lower()
                if cut == "update" and statement.startswith("insert into learning_updates"):
                    os._exit(71)
                if (cut == "event" and statement.startswith("insert into events(")
                    and parameters[0] == "learning:update-1"):
                    os._exit(71)
                if cut == "commit" and statement == "commit":
                    os._exit(71)
                return result

        sqlite3.connect = lambda *args, **kwargs: connect(
            *args, **kwargs, factory=CrashConnection
        )

    with SQLiteEventStore(path) as store:
        if mode == "seed":
            store.append_event(evidence)
            store.commit_episode(episode, outcome)
        else:
            print(store.commit_learning_with_event(update, learned), flush=True)
    """
)


CRASH_AFTER_EVENT_COMMIT = textwrap.dedent(
    """
    import os
    import sys
    from datetime import UTC, datetime

    from duckmind.provenance import Event, SourceRef
    from duckmind.runtime import DuckRuntime

    at = datetime(2026, 9, 24, tzinfo=UTC)
    with DuckRuntime(sys.argv[1], identity="duck", born_at=at) as runtime:
        original = runtime.store.append_event

        def persist_then_exit(*args, **kwargs):
            original(*args, **kwargs)
            os._exit(73)

        runtime.store.append_event = persist_then_exit
        runtime.ingest(Event(
            event_id="committed-before-snapshot", kind="touch",
            occurred_at=at, observed_at=at,
            source=SourceRef("touch_sensor", "head", 0.9),
            trace_id="committed-before-snapshot", payload={"actor": "alex"},
        ))
    """
)


@pytest.mark.parametrize(
    "crash_before", ["CREATE TABLE outcomes", "PRAGMA user_version=1", "COMMIT"]
)
def test_abrupt_exit_during_schema_leaves_restartable_empty_database(
    tmp_path, crash_before
) -> None:
    path = tmp_path / "state.sqlite3"
    crashed = subprocess.run(
        [sys.executable, "-c", CRASH_DURING_SCHEMA, str(path), crash_before],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert crashed.returncode == 71, crashed.stderr
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
        assert (
            connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == []
        )

    at = datetime(2026, 9, 24, tzinfo=UTC)
    event = Event(
        event_id="after-crash",
        kind="touch",
        occurred_at=at,
        observed_at=at,
        source=SourceRef("touch_sensor", "head", 1.0),
        trace_id="after-crash",
        payload={"actor": "alex"},
    )
    with SQLiteEventStore(path) as recovered:
        assert recovered.integrity_check() == "ok"
        assert recovered.append_event(event)
        for table in ("episodes", "outcomes", "learning_updates", "outbox"):
            assert recovered.count(table) == 0
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    with SQLiteEventStore(path) as restarted:
        assert restarted.event_by_id(event.event_id) == event


def test_schema_statement_failure_rolls_back_and_closes_connection(tmp_path, monkeypatch) -> None:
    path = tmp_path / "state.sqlite3"
    connect = sqlite3.connect
    opened = []
    cleanup = []

    class RejectSchemaConnection(sqlite3.Connection):
        def executescript(self, sql):
            def authorize(action, name, *_):
                if action == sqlite3.SQLITE_CREATE_TABLE and name == "outcomes":
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK

            self.set_authorizer(authorize)
            return super().executescript(sql)

        def rollback(self):
            cleanup.append("rollback")
            return super().rollback()

        def close(self):
            cleanup.append("close")
            return super().close()

    def open_connection(*args, **kwargs):
        connection = connect(*args, **kwargs, factory=RejectSchemaConnection)
        opened.append(connection)
        return connection

    with monkeypatch.context() as patch:
        patch.setattr("duckmind.store.sqlite3.connect", open_connection)
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            SQLiteEventStore(path)
    assert cleanup == ["rollback", "close"]
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        opened[0].execute("SELECT 1")
    with connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
        assert (
            connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == []
        )
    with SQLiteEventStore(path) as recovered:
        assert recovered.count("events") == 0
        assert recovered.integrity_check() == "ok"


@pytest.mark.parametrize("cut", ["update", "event", "commit"])
def test_abrupt_exit_during_learning_is_all_or_nothing_and_retryable(tmp_path, cut) -> None:
    path = tmp_path / "state.sqlite3"
    seeded = subprocess.run(
        [sys.executable, "-c", CRASH_DURING_LEARNING, str(path), "seed"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert seeded.returncode == 0, seeded.stderr
    crashed = subprocess.run(
        [sys.executable, "-c", CRASH_DURING_LEARNING, str(path), "learn", cut],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert crashed.returncode == 71, crashed.stderr
    with SQLiteEventStore(path) as recovered:
        assert recovered.integrity_check() == "ok"
        assert recovered.count("episodes") == 1
        assert recovered.count("outcomes") == 1
        assert recovered.count("learning_updates") == (1 if cut == "commit" else 0)
        assert recovered.count("events") == (2 if cut == "commit" else 1)
    retried = subprocess.run(
        [sys.executable, "-c", CRASH_DURING_LEARNING, str(path), "retry"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert retried.returncode == 0, retried.stderr
    assert retried.stdout.strip() == ("False" if cut == "commit" else "True")
    with DuckRuntime(path, identity="duck", born_at=datetime(2026, 9, 24, tzinfo=UTC)) as duck:
        assert duck.health()["database_integrity"] == "ok"
        assert duck.store.count("learning_updates") == 1
        assert duck.store.count("events") == 2
        assert duck.state.preferences["preferences.alex.play"] == 0.503
        assert duck.state.applied_update_ids == ("update-1",)


def test_committed_event_replays_when_process_dies_before_snapshot(tmp_path) -> None:
    path = tmp_path / "state.sqlite3"
    born_at = datetime(2026, 9, 24, tzinfo=UTC)
    with DuckRuntime(path, identity="duck", born_at=born_at) as duck:
        assert duck.store.count("events") == 0
    crashed = subprocess.run(
        [sys.executable, "-c", CRASH_AFTER_EVENT_COMMIT, str(path)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert crashed.returncode == 73, crashed.stderr
    with DuckRuntime(path, identity="duck", born_at=born_at) as recovered:
        assert recovered.store.count("events") == 1
        assert recovered.health()["last_event_seq"] == 1
        assert recovered.state.affect.valence > 0
    with DuckRuntime(path, identity="duck", born_at=born_at) as restarted:
        assert restarted.health()["last_event_seq"] == 1
        assert restarted.store.count("events") == 1
