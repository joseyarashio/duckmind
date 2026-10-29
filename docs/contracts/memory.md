# Memory Integration Boundary

## Authority

The append-only event store is the source of truth. Vector indexes, summaries,
and Nowledge Mem are recall aids derived from events; they are not authorities
for physical facts.

## Reads

`MemoryPort.recall()` returns records with `memory_id`, source event IDs, and
reliability. A context builder applies recency, reliability, provenance,
relevance, and learned utility before facts reach cognition.

## Writes

- Sensor/adapter events enter through trusted event adapters.
- LLM reflection produces candidates only.
- Consolidation links every memory to source events.
- Learning changes require an evidence-backed `LearningUpdate` and explicit commit.
- Contradictions are stored as separate claims with sources and times.

The implemented SQLite outbox is an **opt-in delivery boundary**, not a memory
consolidator or permission system. A trusted local component may atomically
append an event and a distilled job with this exact envelope:

```json
{
  "item_id": "stable-record-id",
  "payload": {
    "kind": "autobiographical",
    "text": "A short, owner-approved sentence.",
    "source_event_ids": ["source-event-id"],
    "reliability": 0.8,
    "approved_for_external_memory": true
  }
}
```

`source_event_ids` must contain exactly the event appended in the same
transaction. Before delivery, the worker rereads its digest-checked source and
rejects an unapproved, ungrounded, oversized (>512 UTF-8 bytes), or
overconfident record. The boolean is **not** proof of consent on its own: a
future owner-consent/consolidation component must set it after a real approval
decision. The current pet service never creates these jobs.

`MemoryOutboxWorker.run_once()` processes at most 100 due jobs through an
injected async `MemorySink`; no network sink is built in. Each call has a
finite timeout, retry delay grows from 5 seconds to at most one hour, and
stored failure codes do not contain remote exception text. Invalid envelopes
remain pending but are deferred for operator diagnosis; malformed outbox JSON
does not starve later jobs. Source-event corruption stops the batch rather
than sending an ungrounded record. A sink must honor task cancellation for the
timeout to be effective; Python cannot forcibly stop a client that suppresses
cancellation. One runtime/worker process is assumed.

Delivery is **at least once**. If a remote write succeeds but the local
acknowledgement fails, the same `item_id` is retried. Any future sink must
provide durable deduplication on that stable `idempotency_key`; otherwise
duplicate records are possible. The fake sink tests only this local contract,
not a real Nowledge Mem API or its idempotency behavior.

## Availability

Memory unavailability must not block robotd safety, body stop, System 1
fallbacks, or event capture. Pending consolidation may retry later; cognition
must label recalled context stale or unavailable rather than inventing it.

Current implementation: event capture, a durable local outbox, and the
bounded, injected-sink delivery worker exist. No production Nowledge Mem sink,
owner-consent/consolidation flow, producer, scheduler, or retrieval adapter is
wired into the pet service. Mem loss therefore cannot block the local first
game; semantic recall and external memory sync are not yet features of it.

## Nowledge Mem

If used, Nowledge Mem stores distilled autobiographical/semantic records and
handoff summaries. It does not receive source repositories or full raw traces,
and it does not replace Git for code/version provenance.
