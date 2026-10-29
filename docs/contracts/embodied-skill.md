# Embodied Skill Layer API Contract

## Inputs

Duck cognition emits semantic behavior intent, for example:

```yaml
intent: approach
target: alex
style: excited
urgency: 0.65
constraints:
  desired_distance_m: 0.8
  max_speed_mps: 0.25
```

No joint or servo field is legal at this boundary.

## BodyPort

`duckmind.body.BodyPort` exposes exactly:

- `dispatch_intent(BodyIntent) -> BodyAck`
- `observe() -> RobotObservation`
- `stop() -> BodyAck`

The implementation allowlist is exactly `robot.move`, `robot.head`,
`robot.look`, `robot.do`, and `robot.stop`, with method-specific parameter
validation and finite numeric values. `robot.enable`, policy changes, servo
reboot, gains and direct joint commands are not in this port. The separate
`RobotdOperatorClient` is used only by commissioning/simulation setup.

The first executable skill API is `OneShotSkillIntent(intent_id, skill)` ->
`EmbodiedSkillLayer.start_one_shot()` -> `OneShotExecution.poll()`. The layer
serializes its own in-flight one-shot requests because official state frames
have no execution ID. `sit_toggle` is excluded until a posture-phase adapter
exists. Other local clients can still issue a same-named skill; hardware
deployment needs owner/lease attribution for multi-client operation.

The supervised first pet activity is `look_game`, implemented as a fixed
`EmbodiedSkillLayer.perform_front_gaze()` head command. The v5 policy's head
observation is delta-from-HOME; the adapter computes a scalar measured
tracking error internally from `robot.state.joints` and exposes no joint
indices to cognition. Admission and each measured sample require `policy=walk`
and both requested and applied robotd velocity intents within 0.01 of zero;
this is a command-space check, **not proof of zero physical movement**. Two
fresh samples within 0.12 rad then mean execution completed. Missing
measurement, clamp/profile mismatch, policy preemption, commanded motion,
fall/limp, timeout, stop, or transport loss is not completion. Any failed
execution except an already-latched owner stop immediately revokes local play
and requests `robot.stop`; it cannot torque off or freeze a prior head command.
`robot.head` is a last-writer-wins notification, so socket write is explicitly
unconfirmed. Without an upstream exclusive lease, another client can still
race between an admission sample and dispatch; physical supervision remains
mandatory.

The upstream `robot.look` absolute-IK/v5-delta mismatch is recorded in
[validation](../upstream-validation.md#head-command-mismatch-discovered-by-the-pet-service-gate).
Until corrected upstream and revalidated, it is not a first-board play path.
Real `look_game` is disabled by default even with API 37 and `velstand.onnx`;
the operator must approve its exact board/policy/head mechanism after bench
tests. Each real-body preflight also reads `robot.policies`: `walk` must be
enabled and homed, with one unoverridden, error-free `official` walk slot at
`/opt/robot/policies/current/velstand.onnx`. A same-basename local/community
override is refused; the watchdog revokes play if selection later changes.
Because the upstream [origin label is not a security boundary](https://github.com/pollen-robotics/microduck/blob/a9ec4b2079ef8ee7904014089c885bb07d57d63c/robotd/src/main.rs),
each real preflight additionally hashes the on-disk walk ONNX and requires the
reviewed v5 SHA-256 recorded in the local
policy probe. A missing,
oversized, changed or unreadable file refuses play; the watchdog latches play
off if the file changes after enable. This checks **current disk bytes**, not
the ONNX graph already loaded in robotd memory, its binary, servo firmware or
mechanics. A newer official policy needs a newly reviewed Duck Mind release,
not a disabled hash guard. No LLM can set the approval flag.

## Execution lifecycle

The layer creates a unique Duck Mind `execution_id`. Since official
`robot.state` carries no matching request ID, correlation is conservative:

1. ACK accepted: request admitted, not started.
2. Expected `policy` observed: started.
3. Expected policy exits: execution window completed.
4. Outcome evaluator checks goal-specific physical evidence.
5. Missing evidence, timeout, connection loss, safety state, or preemption is
   recorded explicitly.

`SkillTracker` supports named episodic policy transitions, `:unwind`,
start/completion timeouts, refusal, safety interruption and unexpected-policy
interruption. It uses local receipt time; `robot.state.t` is retained only for
cross-sample correlation.

## Continuous skills (future)

Skills such as approach/follow are closed-loop controllers, not a single RPC.
They periodically refresh expiring velocity intents, observe applied movement
and `limited_by`, and send `robot.stop` on completion, interruption, timeout,
or owner cancellation.

## Cancellation and failure

- User stop is delivered on a separate robotd connection and never waits for
  cognition or a blocked state subscription. It calls `robot.stop`, which
  zeros locomotion intent; it is **not** a torque-off emergency stop and does
  not guarantee that a previously issued head command freezes instantly.
- On service exit, systemd runs an independent best-effort `robot.stop`.
- Transport loss ends the Duck Mind execution as interrupted/unknown; robotd's
  own deadman remains the physical fallback.
- A retry creates a new execution ID. It does not rewrite the failed episode.
