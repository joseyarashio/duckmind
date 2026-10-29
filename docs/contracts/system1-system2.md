# System 1 / System 2 Boundary

## System 1

The implemented System 1 slice makes a deterministic owner-invitation choice
for `look_game`, with rest/safety gates, a short goal-switch cooldown and seeded
exploration. Affect decay is a separate deterministic reducer. Future System 1
skills may orient to a known voice, stop on stale targets, or arbitrate
approach/retreat; these are not yet board-play features.

## System 2

System 2 handles language understanding, multi-step planning, explanation,
reflection, and ambiguous situations. It returns a proposal containing:

- intent and confidence;
- fact IDs used;
- assumptions and uncertainty;
- optional plan steps, never motor commands.

Only the `CognitiveProposal` and grounding/routing gate are implemented; no
live LLM/System 2 runtime is connected to the pet service.

## Routing

```text
System 1 high confidence -> arbiter
System 1 low confidence  -> System 2 proposal -> arbiter
System 2 timeout/error   -> safe fallback or no-op
```

The implemented grounding gate rejects unknown fact IDs. The pictured
escalation and timeout fallback are the target architecture, not a currently
running System 2 path.

## Outside both systems

The following are deterministic services and never delegated to an LLM:

- motor and physical safety;
- emotion/drive numeric update rules;
- relationship/personality learning bounds;
- event timestamps and provenance;
- memory commit and conflict rules;
- execution timeout/cancellation;
- policy capability discovery.
