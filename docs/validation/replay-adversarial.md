# Replay and Adversarial Validation Plan

## Replay Lab record

Each record contains ordered events, body acknowledgements and observations,
behavior proposals, arbitration decisions, outcome evidence, and learning
updates. It pins every relevant version and a deterministic clock/seed.

## Validation levels

### L0 — schema and invariants

- reject joint/servo fields at the cognition/body boundary;
- reject evidence-free outcomes and learning updates;
- enforce trait/relationship delta bounds;
- reject unknown System 2 fact IDs.

### L1 — pure replay

- same inputs + versions + seed produce the same proposals and updates;
- first divergence reports index, state hash, decision, and evidence;
- snapshot replay supports event ablation for trait attribution.
- the implemented `RecordedBodyPort` checks exact BodyPort command order and
  observations; the official named-skill trace is replayed through the real
  `EmbodiedSkillLayer`. The first pet-service gaze now has a paired simulator
  body trace; full preflight/safety/feedback/storage call recording remains
  pending and is checked through separate gates.

### L2 — robotd + MuJoCo

- intent -> skill -> robotd -> RemoteIo -> MuJoCo -> robot.state -> outcome;
- ACK/start/completion are distinct;
- start/completion timeout, interrupt, reconnect, and deadman cases;
- simulator oracle checks trunk height/contact without leaking into cognition;
- randomized latency/noise/dropout remains seeded and replayable.

### L3 — real robot

- servo-bus dropout and hardware errors;
- battery/thermal behavior;
- BLE/radio asymmetry;
- camera ISP, NPU, encoder/RGA path;
- emergency stop and forbidden-zone enforcement.

## Adversarial suites

1. Memory poisoning: conflicting, duplicated, stale, and low-reliability facts.
2. Personality pumping: repeated reward/rejection cannot exceed bounded plasticity.
3. Emotion flooding: decay, saturation, cooldown, and refractory periods.
4. LLM hallucination: unknown fact IDs cannot become events or memories.
5. RAG collision: similar text with wrong provenance loses to reliable evidence.
6. System 1 ambiguity: low confidence escalates; timeout falls back safely.
7. Action thrashing: alternating approach/stop uses hysteresis and cooldown.
8. Safety bypass: no cognition/memory/personality state can expose servo control.
9. Command admission ambiguity: accepted-but-never-started becomes timeout, not success.
10. Link loss: continuous motion stops by explicit stop when possible and robotd deadman otherwise.
11. Memory delivery: unapproved, altered, oversized and malformed records
    never reach the sink; timeout/outage defers them without blocking local
    play; remote-success/local-ack failure retries a stable key.

## Counterfactual checks

Clone one snapshot, change exactly one variable, and compare decisions. A
decision flip is valid only when the changed field is listed in that decision's
declared influence set. Example: `rest_need 0.2 -> 0.9` may change play to rest;
`battery 0.72 -> 0.71` must not change attachment or fear of a person.

## Growth causality

For any durable trait change:

1. list its committed `LearningUpdate` records;
2. traverse outcomes to episode events;
3. replay from the prior snapshot;
4. ablate the attributed event group;
5. verify the trait difference shrinks as predicted;
6. fail the audit when stored attribution and update mechanics disagree.

## Initial acceptance gates

- all Duck Mind unit tests pass;
- official Microduck contract suites pass at the pinned commit;
- headless body protocol smoke passes;
- the minimal official `robotd` + `duck-body` trace closes one named skill
  episode and replays to the same execution gate and unknown task outcome;
- the same trace schema accepts a real-robot run without a simulator-only field.

The final real-robot trace gate is still pending hardware. The current
simulator trace is official-v5-kick-left.jsonl.
Run `python -m duckmind.body_replay` on that trace to check both its existing
wire/oracle evidence and the command path through the skill adapter.
