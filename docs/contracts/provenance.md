# Event, Episode, Outcome, and Learning Provenance

The executable schemas live in `src/duckmind/provenance.py`.

## Event

Required fields:

- `event_id`, `kind`, `trace_id`
- timezone-aware `occurred_at` and `observed_at`
- `source = {kind, source_id, reliability, version?}`
- immutable `payload`
- optional `parent_event_ids`

An LLM statement is not an event until a trusted adapter records its source.
Conflicting events are retained rather than overwritten.

## Episode

An episode is a replayable envelope:

- `episode_id`, `trace_id`, start/end
- ordered `event_ids`, `intent_ids`, `outcome_ids`
- optional `state_before_hash`, `state_after_hash`
- runtime, policy, cognition, schema, and evaluator versions

Large raw payloads may live in blob storage; the episode keeps their hashes and IDs.
The current first-pet service records event/outcome IDs and runtime versions but
does not yet populate per-episode before/after state hashes; SQLite snapshot
and event-record digests are the implemented replay integrity check.

## Outcome

An outcome contains:

- verdict: `success | failure | interrupted | unknown`
- evaluator version
- evidence event IDs
- numeric metrics and a reason

`unknown` is mandatory when the sensor needed to prove success was absent. Do
not convert missing load/velocity or a dropped state frame to zero.

## LearningUpdate

The schema permits this append-only lifecycle:

```text
PROPOSED -> COMMITTED
         -> REJECTED
COMMITTED -> ROLLED_BACK (new compensating record)
```

Each update contains before/after values, parameter path, rule version,
episode ID, outcome evidence IDs, and a maximum allowed absolute delta. The
schema rejects evidence-free updates and updates beyond the slow-plasticity
bound.
The current writer implements PROPOSED in memory and atomic COMMITTED
preference updates. Explicit rejection and compensating rollback writers are
future work; the enum alone is not a rollback mechanism.

## Causal audit

To explain a trait change, select committed updates for the parameter, traverse
their outcome IDs to episodes/events, then replay or ablate the relevant event
set. Provenance is considered valid only when the replayed update mechanism
reproduces the stored delta within tolerance.
