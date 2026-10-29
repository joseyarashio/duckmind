"""A Mem outage must not erase local evidence or fabricate delivery success."""

import asyncio
import json
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from duckmind.memory import MemoryRecord
from duckmind.memory_delivery import MemoryOutboxWorker
from duckmind.provenance import Event, SourceRef
from duckmind.store import SQLiteEventStore

NOW = datetime(2026, 9, 24, tzinfo=UTC)


def queue(store: SQLiteEventStore, suffix: str, *, reliability: float = 0.8) -> None:
    event_id = f"event-{suffix}"
    store.append_event(
        Event(
            event_id=event_id,
            kind="interaction_observed",
            occurred_at=NOW,
            observed_at=NOW,
            source=SourceRef("local_sensor", "test", 0.8),
            trace_id=event_id,
            payload={"actor": "owner"},
        ),
        memory_job={
            "item_id": f"memory-{suffix}",
            "payload": {
                "kind": "autobiographical",
                "text": f"Owner greeted duck {suffix}.",
                "source_event_ids": [event_id],
                "reliability": reliability,
                "approved_for_external_memory": True,
            },
        },
    )


class RecordingSink:
    def __init__(self, *, failures: int = 0) -> None:
        self.failures = failures
        self.calls: list[tuple[MemoryRecord, str]] = []
        self.accepted: set[str] = set()

    async def put(self, record: MemoryRecord, *, idempotency_key: str) -> None:
        self.calls.append((record, idempotency_key))
        if self.failures:
            self.failures -= 1
            raise ConnectionError("secret-token-must-not-enter-the-database")
        self.accepted.add(idempotency_key)


def test_delivers_only_grounded_opted_in_record_once_across_restart(tmp_path) -> None:
    path = tmp_path / "state.sqlite3"
    with SQLiteEventStore(path) as store:
        queue(store, "one")
        sink = RecordingSink()
        batch = asyncio.run(MemoryOutboxWorker(store, sink).run_once(at=NOW))
        assert batch.delivered == ("memory-one",)
        assert batch.deferred == ()
        assert sink.calls == [
            (
                MemoryRecord("memory-one", "Owner greeted duck one.", ("event-one",), 0.8),
                "memory-one",
            )
        ]
        assert store.due_memory_jobs(at=NOW + timedelta(days=1)) == []
    with SQLiteEventStore(path) as restarted:
        assert restarted.count("events") == 1
        assert restarted.count("outbox") == 1
        assert asyncio.run(MemoryOutboxWorker(restarted, sink).run_once(at=NOW)).delivered == ()
        assert len(sink.calls) == 1


def test_outage_retries_with_bounded_backoff_without_leaking_error_text(tmp_path) -> None:
    with SQLiteEventStore(tmp_path / "state.sqlite3") as store:
        queue(store, "one")
        sink = RecordingSink(failures=1)
        worker = MemoryOutboxWorker(store, sink)
        assert asyncio.run(worker.run_once(at=NOW)).deferred == ("memory-one",)
        assert store.due_memory_jobs(at=NOW + timedelta(seconds=4)) == []
        pending = store.due_memory_jobs(at=NOW + timedelta(seconds=5))
        assert len(pending) == 1 and pending[0].attempts == 1
        assert pending[0].source_event_id == "event-one"
        error = store._db.execute("SELECT last_error FROM outbox").fetchone()[0]
        assert error == "delivery_failed"
        assert asyncio.run(worker.run_once(at=NOW + timedelta(seconds=5))).delivered == (
            "memory-one",
        )
        assert [key for _, key in sink.calls] == ["memory-one", "memory-one"]


def test_poisoned_record_is_deferred_without_blocking_next_job(tmp_path) -> None:
    with SQLiteEventStore(tmp_path / "state.sqlite3") as store:
        queue(store, "bad")
        queue(store, "good")
        payload = {
            "kind": "autobiographical",
            "text": "untrusted text",
            "source_event_ids": ["event-good"],
            "reliability": 1.0,
            "approved_for_external_memory": False,
        }
        store._db.execute(
            "UPDATE outbox SET payload_json=? WHERE item_id='memory-bad'",
            (json.dumps(payload),),
        )
        sink = RecordingSink()
        batch = asyncio.run(MemoryOutboxWorker(store, sink).run_once(at=NOW))
        assert batch.deferred == ("memory-bad",)
        assert batch.delivered == ("memory-good",)
        assert [key for _, key in sink.calls] == ["memory-good"]
        assert store.due_memory_jobs(at=NOW + timedelta(seconds=5))[0].item_id == "memory-bad"


def test_invalid_json_job_is_deferred_without_blocking_next_job(tmp_path) -> None:
    with SQLiteEventStore(tmp_path / "state.sqlite3") as store:
        queue(store, "bad")
        queue(store, "good")
        store._db.execute(
            "UPDATE outbox SET payload_json=? WHERE item_id='memory-bad'", ("{not-json",)
        )
        sink = RecordingSink()
        batch = asyncio.run(MemoryOutboxWorker(store, sink).run_once(at=NOW))
        assert batch.deferred == ("memory-bad",)
        assert batch.delivered == ("memory-good",)
        assert [key for _, key in sink.calls] == ["memory-good"]


def test_corrupt_source_event_stops_delivery(tmp_path) -> None:
    with SQLiteEventStore(tmp_path / "state.sqlite3") as store:
        queue(store, "one")
        store._db.execute("UPDATE events SET record_json=? WHERE event_id='event-one'", ("{}",))
        sink = RecordingSink()
        with pytest.raises(RuntimeError, match="digest mismatch"):
            asyncio.run(MemoryOutboxWorker(store, sink).run_once(at=NOW))
        assert sink.calls == []
        assert store.due_memory_jobs(at=NOW)[0].attempts == 0


def test_remote_success_then_local_ack_failure_retries_same_key(tmp_path, monkeypatch) -> None:
    with SQLiteEventStore(tmp_path / "state.sqlite3") as store:
        queue(store, "one")
        sink = RecordingSink()
        worker = MemoryOutboxWorker(store, sink)
        original = store.complete_memory_job
        once = True

        def fail_ack(item_id: str) -> None:
            nonlocal once
            if once:
                once = False
                raise sqlite3.OperationalError("local commit failed")
            original(item_id)

        monkeypatch.setattr(store, "complete_memory_job", fail_ack)
        with pytest.raises(sqlite3.OperationalError):
            asyncio.run(worker.run_once(at=NOW))
        assert store.due_memory_jobs(at=NOW)[0].item_id == "memory-one"
        assert asyncio.run(worker.run_once(at=NOW)).delivered == ("memory-one",)
        assert [key for _, key in sink.calls] == ["memory-one", "memory-one"]
        assert sink.accepted == {"memory-one"}


def test_cancelled_delivery_stays_pending_for_retry(tmp_path) -> None:
    class BlockedSink:
        def __init__(self) -> None:
            self.started = asyncio.Event()

        async def put(self, _record: MemoryRecord, *, idempotency_key: str) -> None:
            self.started.set()
            await asyncio.Event().wait()

    async def cancel(store: SQLiteEventStore) -> None:
        sink = BlockedSink()
        task = asyncio.create_task(MemoryOutboxWorker(store, sink).run_once(at=NOW))
        await sink.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    with SQLiteEventStore(tmp_path / "state.sqlite3") as store:
        queue(store, "one")
        asyncio.run(cancel(store))
        assert store.due_memory_jobs(at=NOW)[0].attempts == 0
        assert asyncio.run(
            MemoryOutboxWorker(store, RecordingSink()).run_once(at=NOW)
        ).delivered == ("memory-one",)


def test_timeout_defers_without_holding_up_later_jobs(tmp_path) -> None:
    class BlockingFirstSink(RecordingSink):
        async def put(self, record: MemoryRecord, *, idempotency_key: str) -> None:
            if idempotency_key == "memory-first":
                await asyncio.Event().wait()
            await super().put(record, idempotency_key=idempotency_key)

    with SQLiteEventStore(tmp_path / "state.sqlite3") as store:
        queue(store, "first")
        queue(store, "second")
        sink = BlockingFirstSink()
        batch = asyncio.run(MemoryOutboxWorker(store, sink, timeout_s=0.01).run_once(at=NOW))
        assert batch.deferred == ("memory-first",)
        assert batch.delivered == ("memory-second",)
        assert (
            store._db.execute(
                "SELECT last_error FROM outbox WHERE item_id='memory-first'"
            ).fetchone()[0]
            == "delivery_timeout"
        )


def test_worker_and_store_bound_batch_inputs(tmp_path) -> None:
    with SQLiteEventStore(tmp_path / "state.sqlite3") as store:
        sink = RecordingSink()
        for bad in (0, 101, True):
            with pytest.raises(ValueError, match="limit"):
                store.due_memory_jobs(at=NOW, limit=bad)
            with pytest.raises(ValueError, match="batch size"):
                MemoryOutboxWorker(store, sink, batch_size=bad)
        for bad in (0.0, float("nan"), float("inf"), True, "5"):
            with pytest.raises(ValueError, match="timeout"):
                MemoryOutboxWorker(store, sink, timeout_s=bad)
