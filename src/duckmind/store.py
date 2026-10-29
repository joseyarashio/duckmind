"""Crash-safe, append-only local authority for Duck Mind's lived experience.

Nowledge Mem is an optional downstream index; this database and its evidence
links remain authoritative even when memory sync is offline.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterator, Mapping

from duckmind.provenance import (
    Episode,
    Event,
    LearningStatus,
    LearningUpdate,
    Outcome,
    OutcomeVerdict,
    SourceRef,
)

SCHEMA_VERSION = 1
IMMEDIATE = datetime(1970, 1, 1, tzinfo=timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class StoredEvent:
    seq: int
    event: Event


@dataclass(frozen=True, slots=True)
class Snapshot:
    key: str
    revision: int
    last_event_seq: int
    state: dict[str, object]
    digest: str


@dataclass(frozen=True, slots=True)
class MemoryJob:
    item_id: str
    source_event_id: str
    payload: object
    attempts: int
    next_attempt_at: datetime


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(item) for item in value]
    return value


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat()


def _event_record(event: Event) -> dict[str, object]:
    return {
        "event_id": event.event_id,
        "kind": event.kind,
        "occurred_at": _utc(event.occurred_at),
        "observed_at": _utc(event.observed_at),
        "source": asdict(event.source),
        "trace_id": event.trace_id,
        "payload": _thaw_json(event.payload),
        "parent_event_ids": list(event.parent_event_ids),
    }


def _read_event(record_json: str) -> Event:
    data = json.loads(record_json)
    return Event(
        event_id=data["event_id"],
        kind=data["kind"],
        occurred_at=datetime.fromisoformat(data["occurred_at"]),
        observed_at=datetime.fromisoformat(data["observed_at"]),
        source=SourceRef(**data["source"]),
        trace_id=data["trace_id"],
        payload=data["payload"],
        parent_event_ids=tuple(data["parent_event_ids"]),
    )


def _episode_record(episode: Episode) -> dict[str, object]:
    return {
        "episode_id": episode.episode_id,
        "trace_id": episode.trace_id,
        "started_at": _utc(episode.started_at),
        "ended_at": _utc(episode.ended_at) if episode.ended_at else None,
        "event_ids": list(episode.event_ids),
        "intent_ids": list(episode.intent_ids),
        "outcome_ids": list(episode.outcome_ids),
        "state_before_hash": episode.state_before_hash,
        "state_after_hash": episode.state_after_hash,
        "runtime_versions": dict(episode.runtime_versions),
    }


class SQLiteEventStore:
    """Single-process SQLite store with atomic evidence and outbox writes."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        existed = self.path.exists()
        self._db = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        try:
            self._initialize(existed=existed)
        except BaseException:
            try:
                self._db.rollback()
            finally:
                self._db.close()
            raise

    def _initialize(self, *, existed: bool) -> None:
        if not existed:
            os.chmod(self.path, 0o600)
        self._db.execute("PRAGMA busy_timeout=5000")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        version = self._db.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise RuntimeError(f"database schema {version} is newer than this runtime")
        if version == 0:
            # executescript does not supply an implicit transaction. Keep the
            # DDL and its version marker in one commit, including first boot.
            self._db.executescript(
                """
                BEGIN IMMEDIATE;
                CREATE TABLE events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    trace_id TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    digest TEXT NOT NULL
                );
                CREATE INDEX events_trace ON events(trace_id, seq);
                CREATE TABLE episodes (
                    episode_id TEXT PRIMARY KEY,
                    trace_id TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    digest TEXT NOT NULL
                );
                CREATE TABLE outcomes (
                    outcome_id TEXT PRIMARY KEY,
                    episode_id TEXT NOT NULL REFERENCES episodes(episode_id),
                    record_json TEXT NOT NULL,
                    digest TEXT NOT NULL
                );
                CREATE TABLE learning_updates (
                    update_id TEXT PRIMARY KEY,
                    episode_id TEXT NOT NULL REFERENCES episodes(episode_id),
                    record_json TEXT NOT NULL,
                    digest TEXT NOT NULL
                );
                CREATE TABLE snapshots (
                    key TEXT PRIMARY KEY,
                    revision INTEGER NOT NULL,
                    last_event_seq INTEGER NOT NULL,
                    state_json TEXT NOT NULL,
                    digest TEXT NOT NULL
                );
                CREATE TABLE outbox (
                    item_id TEXT PRIMARY KEY,
                    source_event_id TEXT NOT NULL REFERENCES events(event_id),
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('pending', 'done')),
                    attempts INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at TEXT NOT NULL,
                    last_error TEXT
                );
                CREATE INDEX outbox_due ON outbox(status, next_attempt_at);
                PRAGMA user_version=1;
                COMMIT;
                """
            )

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> SQLiteEventStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @contextmanager
    def _write(self) -> Iterator[None]:
        if self._db.in_transaction:
            self._db.execute("SAVEPOINT duckmind_nested")
            try:
                yield
                self._db.execute("RELEASE SAVEPOINT duckmind_nested")
            except BaseException:
                self._db.execute("ROLLBACK TO SAVEPOINT duckmind_nested")
                self._db.execute("RELEASE SAVEPOINT duckmind_nested")
                raise
            return
        self._db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise

    def append_event(
        self,
        event: Event,
        *,
        memory_job: Mapping[str, object] | None = None,
        _from_learning_commit: bool = False,
    ) -> bool:
        if event.kind == "learning_committed" and not _from_learning_commit:
            raise ValueError("learning_committed requires atomic update/event commit")
        record = _json(_event_record(event))
        digest = _digest(record)
        with self._write():
            existing = self._db.execute(
                "SELECT digest FROM events WHERE event_id=?", (event.event_id,)
            ).fetchone()
            if existing:
                if existing[0] != digest:
                    raise ValueError(f"conflicting event_id: {event.event_id}")
                return False
            self._db.execute(
                "INSERT INTO events(event_id, trace_id, record_json, digest) VALUES(?,?,?,?)",
                (event.event_id, event.trace_id, record, digest),
            )
            if memory_job is not None:
                item_id = memory_job.get("item_id")
                payload = memory_job.get("payload")
                if not isinstance(item_id, str) or not item_id or not isinstance(payload, dict):
                    raise ValueError("memory job needs item_id and object payload")
                if payload.get("source_event_ids") != [event.event_id]:
                    raise ValueError("memory job must link the source event exactly")
                self._db.execute(
                    """INSERT INTO outbox(item_id, source_event_id, payload_json, status,
                       next_attempt_at) VALUES(?,?,?,'pending',?)""",
                    (item_id, event.event_id, _json(payload), IMMEDIATE),
                )
        return True

    def list_events(self, *, after_seq: int = 0) -> list[StoredEvent]:
        rows = self._db.execute(
            "SELECT seq, record_json, digest FROM events WHERE seq>? ORDER BY seq",
            (after_seq,),
        ).fetchall()
        entries = []
        for seq, record, digest in rows:
            if _digest(record) != digest:
                raise RuntimeError(f"event digest mismatch at sequence {seq}")
            entries.append(StoredEvent(seq, _read_event(record)))
        return entries

    def events_for_trace(self, trace_id: str) -> list[Event]:
        rows = self._db.execute(
            "SELECT record_json, digest FROM events WHERE trace_id=? ORDER BY seq",
            (trace_id,),
        ).fetchall()
        events = []
        for record, digest in rows:
            if _digest(record) != digest:
                raise RuntimeError(f"event digest mismatch for trace {trace_id}")
            events.append(_read_event(record))
        return events

    def event_by_id(self, event_id: str) -> Event | None:
        row = self._db.execute(
            "SELECT record_json, digest FROM events WHERE event_id=?", (event_id,)
        ).fetchone()
        if row is None:
            return None
        record, digest = row
        if _digest(record) != digest:
            raise RuntimeError(f"event digest mismatch: {event_id}")
        return _read_event(record)

    def has_episode(self, episode_id: str) -> bool:
        return (
            self._db.execute("SELECT 1 FROM episodes WHERE episode_id=?", (episode_id,)).fetchone()
            is not None
        )

    def outcome_by_id(self, outcome_id: str) -> Outcome | None:
        row = self._db.execute(
            "SELECT record_json, digest FROM outcomes WHERE outcome_id=?", (outcome_id,)
        ).fetchone()
        if row is None:
            return None
        record, digest = row
        if _digest(record) != digest:
            raise RuntimeError(f"outcome digest mismatch: {outcome_id}")
        data = json.loads(record)
        return Outcome(
            outcome_id=data["outcome_id"],
            episode_id=data["episode_id"],
            verdict=OutcomeVerdict(data["verdict"]),
            evaluator_version=data["evaluator_version"],
            evidence_event_ids=tuple(data["evidence_event_ids"]),
            metrics=data["metrics"],
            reason=data["reason"],
        )

    def commit_episode(self, episode: Episode, outcome: Outcome) -> bool:
        if episode.ended_at is None:
            raise ValueError("episode must be closed before committing an outcome")
        if (
            outcome.episode_id != episode.episode_id
            or outcome.outcome_id not in episode.outcome_ids
        ):
            raise ValueError("outcome does not belong to episode")
        if episode.outcome_ids != (outcome.outcome_id,):
            raise ValueError("first-slice store requires exactly one outcome per closed episode")
        if not set(outcome.evidence_event_ids).issubset(episode.event_ids):
            raise ValueError("outcome evidence must belong to episode")
        episode_json = _json(_episode_record(episode))
        outcome_json = _json(
            {
                "outcome_id": outcome.outcome_id,
                "episode_id": outcome.episode_id,
                "verdict": outcome.verdict.value,
                "evaluator_version": outcome.evaluator_version,
                "evidence_event_ids": list(outcome.evidence_event_ids),
                "metrics": dict(outcome.metrics),
                "reason": outcome.reason,
            }
        )
        episode_digest, outcome_digest = _digest(episode_json), _digest(outcome_json)
        with self._write():
            existing_episode = self._db.execute(
                "SELECT digest FROM episodes WHERE episode_id=?", (episode.episode_id,)
            ).fetchone()
            if existing_episode:
                existing_outcome = self._db.execute(
                    "SELECT digest FROM outcomes WHERE outcome_id=?",
                    (outcome.outcome_id,),
                ).fetchone()
                if (
                    existing_episode[0] != episode_digest
                    or not existing_outcome
                    or existing_outcome[0] != outcome_digest
                ):
                    raise ValueError("conflicting episode or outcome ID")
                return False
            last_occurred_at = episode.started_at
            for event_id in episode.event_ids:
                row = self._db.execute(
                    "SELECT trace_id, record_json, digest FROM events WHERE event_id=?", (event_id,)
                ).fetchone()
                if row is None:
                    raise ValueError(f"missing event: {event_id}")
                if row[0] != episode.trace_id:
                    raise ValueError(f"event trace does not match episode: {event_id}")
                if _digest(row[1]) != row[2]:
                    raise RuntimeError(f"event digest mismatch: {event_id}")
                occurred_at = _read_event(row[1]).occurred_at
                if occurred_at < last_occurred_at or occurred_at > episode.ended_at:
                    raise ValueError(f"episode event is out of chronological bounds: {event_id}")
                last_occurred_at = occurred_at
            self._db.execute(
                "INSERT INTO episodes VALUES(?,?,?,?)",
                (episode.episode_id, episode.trace_id, episode_json, episode_digest),
            )
            self._db.execute(
                "INSERT INTO outcomes VALUES(?,?,?,?)",
                (outcome.outcome_id, episode.episode_id, outcome_json, outcome_digest),
            )
        return True

    def append_learning_update(
        self, update: LearningUpdate, *, _from_learning_commit: bool = False
    ) -> bool:
        if update.status is LearningStatus.COMMITTED and not _from_learning_commit:
            raise ValueError("committed learning requires atomic update/event commit")
        record_json = _json(asdict(update))
        digest = _digest(record_json)
        with self._write():
            existing = self._db.execute(
                "SELECT digest FROM learning_updates WHERE update_id=?",
                (update.update_id,),
            ).fetchone()
            if existing:
                if existing[0] != digest:
                    raise ValueError("conflicting learning update ID")
                return False
            prior = self._db.execute(
                "SELECT record_json FROM learning_updates WHERE episode_id=?",
                (update.episode_id,),
            ).fetchall()
            if any(json.loads(row[0])["parameter"] == update.parameter for row in prior):
                raise ValueError("episode already updated this parameter")
            for outcome_id in update.outcome_ids:
                row = self._db.execute(
                    "SELECT episode_id, record_json FROM outcomes WHERE outcome_id=?",
                    (outcome_id,),
                ).fetchone()
                if row is None or row[0] != update.episode_id:
                    raise ValueError("learning update references an unknown outcome")
                verdict = json.loads(row[1])["verdict"]
                if update.status is LearningStatus.COMMITTED and verdict not in {
                    "success",
                    "failure",
                }:
                    raise ValueError("committed learning requires a supported outcome")
            self._db.execute(
                "INSERT INTO learning_updates VALUES(?,?,?,?)",
                (update.update_id, update.episode_id, record_json, digest),
            )
        return True

    def has_learning_for_episode(self, episode_id: str, parameter: str) -> bool:
        rows = self._db.execute(
            "SELECT record_json FROM learning_updates WHERE episode_id=?", (episode_id,)
        ).fetchall()
        return any(json.loads(row[0])["parameter"] == parameter for row in rows)

    def daily_learning_magnitude(self, parameter: str, day: date) -> float:
        rows = self._db.execute(
            """SELECT updates.record_json, episodes.record_json
               FROM learning_updates AS updates JOIN episodes
               ON updates.episode_id=episodes.episode_id"""
        ).fetchall()
        total = 0.0
        for update_json, episode_json in rows:
            update = json.loads(update_json)
            if update["parameter"] != parameter or update["status"] != "committed":
                continue
            episode = json.loads(episode_json)
            if datetime.fromisoformat(episode["ended_at"]).date() == day:
                total += abs(update["after"] - update["before"])
        return total

    def commit_learning_with_event(self, update: LearningUpdate, event: Event) -> bool:
        if update.status is not LearningStatus.COMMITTED:
            raise ValueError("atomic learning commit needs COMMITTED status")
        if event.kind != "learning_committed" or event.source.kind != "learning_engine":
            raise ValueError("learning commit event has wrong kind or source")
        if (
            event.payload.get("update_id") != update.update_id
            or event.payload.get("parameter") != update.parameter
        ):
            raise ValueError("learning commit event does not match update")
        if (
            event.payload.get("before") != update.before
            or event.payload.get("after") != update.after
        ):
            raise ValueError("learning commit values do not match update")
        evidence_ids = event.payload.get("evidence_event_ids")
        if (
            not event.parent_event_ids
            or not isinstance(evidence_ids, (list, tuple))
            or any(not isinstance(item, str) for item in evidence_ids)
            or tuple(evidence_ids) != event.parent_event_ids
        ):
            raise ValueError("learning commit requires exact parent evidence")
        with self._write():
            day_row = self._db.execute(
                "SELECT record_json FROM episodes WHERE episode_id=?",
                (update.episode_id,),
            ).fetchone()
            if day_row is None:
                raise ValueError("learning episode does not exist")
            episode_record = json.loads(day_row[0])
            if event.trace_id != episode_record["trace_id"]:
                raise ValueError("learning event trace differs from episode")
            if update.outcome_ids != tuple(episode_record["outcome_ids"]):
                raise ValueError("learning outcomes differ from episode")
            outcome_row = self._db.execute(
                "SELECT record_json FROM outcomes WHERE outcome_id=?",
                (update.outcome_ids[0],),
            ).fetchone()
            if (
                outcome_row is None
                or tuple(json.loads(outcome_row[0])["evidence_event_ids"]) != event.parent_event_ids
            ):
                raise ValueError("learning event parents differ from outcome evidence")
            existing = self._db.execute(
                "SELECT digest FROM learning_updates WHERE update_id=?",
                (update.update_id,),
            ).fetchone()
            if existing is None:
                day = datetime.fromisoformat(episode_record["ended_at"]).date()
                if (
                    self.daily_learning_magnitude(update.parameter, day)
                    + abs(update.after - update.before)
                    > 0.005 + 1e-12
                ):
                    raise ValueError("daily plasticity cap exceeded")
            inserted_update = self.append_learning_update(update, _from_learning_commit=True)
            inserted_event = self.append_event(event, _from_learning_commit=True)
            if inserted_update != inserted_event:
                raise ValueError("learning update/event idempotency mismatch")
            return inserted_update

    def load_snapshot(self, key: str) -> Snapshot | None:
        row = self._db.execute(
            "SELECT revision, last_event_seq, state_json, digest FROM snapshots WHERE key=?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        revision, last_seq, state_json, digest = row
        if _digest(state_json) != digest:
            raise RuntimeError(f"snapshot digest mismatch: {key}")
        return Snapshot(key, revision, last_seq, json.loads(state_json), digest)

    def save_snapshot(
        self,
        key: str,
        state: Mapping[str, object],
        *,
        expected_revision: int,
        last_event_seq: int,
    ) -> Snapshot:
        state_json = _json(dict(state))
        digest = _digest(state_json)
        with self._write():
            row = self._db.execute(
                "SELECT revision, last_event_seq FROM snapshots WHERE key=?", (key,)
            ).fetchone()
            revision = row[0] if row else 0
            if revision != expected_revision:
                raise ValueError("snapshot revision changed")
            if row and last_event_seq < row[1]:
                raise ValueError("snapshot cannot move behind prior event sequence")
            largest_seq = self._db.execute("SELECT COALESCE(MAX(seq),0) FROM events").fetchone()[0]
            if last_event_seq > largest_seq:
                raise ValueError("snapshot references future events")
            new_revision = revision + 1
            self._db.execute(
                """INSERT INTO snapshots VALUES(?,?,?,?,?)
                   ON CONFLICT(key) DO UPDATE SET revision=excluded.revision,
                   last_event_seq=excluded.last_event_seq, state_json=excluded.state_json,
                   digest=excluded.digest""",
                (key, new_revision, last_event_seq, state_json, digest),
            )
        return Snapshot(key, new_revision, last_event_seq, json.loads(state_json), digest)

    def due_memory_jobs(self, *, at: datetime, limit: int = 100) -> list[MemoryJob]:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("memory job limit must be between 1 and 100")
        rows = self._db.execute(
            """SELECT item_id, source_event_id, payload_json, attempts, next_attempt_at FROM outbox
               WHERE status='pending' AND next_attempt_at<=?
               ORDER BY next_attempt_at, item_id LIMIT ?""",
            (_utc(at), limit),
        ).fetchall()
        jobs: list[MemoryJob] = []
        for item_id, source_event_id, payload, attempts, next_at in rows:
            try:
                parsed = json.loads(payload)
            except (TypeError, ValueError):
                # An outbox row is not the authoritative event. Let the worker
                # defer this one as invalid without starving later valid jobs.
                parsed = None
            jobs.append(
                MemoryJob(
                    item_id, source_event_id, parsed, attempts, datetime.fromisoformat(next_at)
                )
            )
        return jobs

    def fail_memory_job(self, item_id: str, error: str, *, next_at: datetime) -> None:
        with self._write():
            cursor = self._db.execute(
                """UPDATE outbox SET attempts=attempts+1, last_error=?, next_attempt_at=?
                   WHERE item_id=? AND status='pending'""",
                (error[:500], _utc(next_at), item_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("memory job is missing or completed")

    def complete_memory_job(self, item_id: str) -> None:
        with self._write():
            cursor = self._db.execute(
                "UPDATE outbox SET status='done' WHERE item_id=? AND status='pending'",
                (item_id,),
            )
            if cursor.rowcount != 1:
                raise ValueError("memory job is missing or completed")

    def count(self, table: str) -> int:
        if table not in {
            "events",
            "episodes",
            "outcomes",
            "learning_updates",
            "outbox",
        }:
            raise ValueError("unknown table")
        return int(self._db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

    def latest_event_seq(self) -> int:
        return int(self._db.execute("SELECT COALESCE(MAX(seq),0) FROM events").fetchone()[0])

    def integrity_check(self) -> str:
        return str(self._db.execute("PRAGMA quick_check").fetchone()[0])
