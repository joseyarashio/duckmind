"""Optional, bounded at-least-once delivery of distilled memory records.

This module has no network client. A caller must supply an async sink that
deduplicates the stable item ID; no sink is configured by the pet service.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from duckmind.memory import MemoryRecord
from duckmind.store import MemoryJob, SQLiteEventStore


class MemorySink(Protocol):
    async def put(self, record: MemoryRecord, *, idempotency_key: str) -> None: ...


@dataclass(frozen=True, slots=True)
class DeliveryBatch:
    delivered: tuple[str, ...]
    deferred: tuple[str, ...]


def _bounded_record(store: SQLiteEventStore, job: MemoryJob) -> MemoryRecord:
    payload = job.payload
    required = {
        "kind",
        "text",
        "source_event_ids",
        "reliability",
        "approved_for_external_memory",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("unsupported memory envelope")
    if (
        payload["kind"] != "autobiographical"
        or payload["approved_for_external_memory"] is not True
        or payload["source_event_ids"] != [job.source_event_id]
    ):
        raise ValueError("unapproved or ungrounded memory envelope")
    text = payload["text"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("memory text is empty")
    try:
        size = len(text.encode("utf-8"))
    except UnicodeError as error:
        raise ValueError("memory text is not UTF-8") from error
    if size > 512:
        raise ValueError("memory text exceeds the distilled-record limit")
    reliability = payload["reliability"]
    source = store.event_by_id(job.source_event_id)
    if source is None or source.event_id != job.source_event_id:
        raise RuntimeError("memory source event is missing")
    if (
        isinstance(reliability, bool)
        or not isinstance(reliability, (int, float))
        or not math.isfinite(reliability)
        or not 0.0 <= reliability <= source.source.reliability
    ):
        raise ValueError("memory reliability exceeds its source")
    return MemoryRecord(job.item_id, text, (job.source_event_id,), float(reliability))


class MemoryOutboxWorker:
    """One event-loop worker; retry keys are stable across crashes and restarts."""

    def __init__(
        self,
        store: SQLiteEventStore,
        sink: MemorySink,
        *,
        batch_size: int = 10,
        timeout_s: float = 5.0,
    ) -> None:
        if type(batch_size) is not int or not 1 <= batch_size <= 100:
            raise ValueError("batch size must be between 1 and 100")
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ValueError("delivery timeout must be positive and finite")
        self._store = store
        self._sink = sink
        self._batch_size = batch_size
        self._timeout_s = timeout_s
        self._lock = asyncio.Lock()

    def _defer(self, job: MemoryJob, *, at: datetime, reason: str) -> None:
        delay_s = min(3600, 5 * 2 ** min(max(job.attempts, 0), 10))
        self._store.fail_memory_job(job.item_id, reason, next_at=at + timedelta(seconds=delay_s))

    async def run_once(self, *, at: datetime | None = None) -> DeliveryBatch:
        now = datetime.now(UTC) if at is None else at
        if now.utcoffset() is None:
            raise ValueError("delivery time must be timezone-aware")
        async with self._lock:
            jobs = self._store.due_memory_jobs(at=now, limit=self._batch_size)
            delivered: list[str] = []
            deferred: list[str] = []
            for job in jobs:
                try:
                    record = _bounded_record(self._store, job)
                except ValueError:
                    self._defer(job, at=now, reason="invalid_memory_envelope")
                    deferred.append(job.item_id)
                    continue
                try:
                    await asyncio.wait_for(
                        self._sink.put(record, idempotency_key=job.item_id),
                        timeout=self._timeout_s,
                    )
                except TimeoutError:
                    self._defer(job, at=now, reason="delivery_timeout")
                    deferred.append(job.item_id)
                    continue
                except Exception:
                    # Never persist exception text or a remote-defined class name.
                    self._defer(job, at=now, reason="delivery_failed")
                    deferred.append(job.item_id)
                    continue
                # If this local commit fails after remote success, retry the same
                # item ID. The sink must deduplicate; exactly-once is not claimed.
                self._store.complete_memory_job(job.item_id)
                delivered.append(job.item_id)
            return DeliveryBatch(tuple(delivered), tuple(deferred))
