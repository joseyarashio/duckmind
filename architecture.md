# Duck Mind v0.3 Embodied Architecture

Status: first simulated pet-service slice and board release candidate,
2026-09-24 Asia/Taipei. Real-hardware acceptance remains open.

The [pre-implementation source review](docs/architecture-review.md) records
the official seam, v5 policy correction, and limitations found before coding.

## 1. Non-negotiable invariants

1. Cognition never sends servo IDs, joint positions, gains, torque, or ONNX actions.
2. All physical behavior goes through the Embodied Skill Layer and `robotd`.
3. Simulator and real robot share the same Duck Mind `BodyPort` and `robot.*` contract.
4. An accepted command is not evidence that motion started or completed.
5. Facts, outcomes, and learning updates retain source and version provenance.
6. LLM output may propose intent or interpretation; it cannot create memories,
   modify traits directly, bypass the arbiter, or weaken safety.
7. Safety remains below cognition and cannot be learned around.

These invariants are represented in `duckmind.body`, `duckmind.skills`, and
the contract tests.

## 2. Runtime layers

```text
Perception / A2A / time
          |
          v
World + Self + Affect + Drives + Relationships
          |
          +-----------------------------+
          |                             |
          v                             v
System 1 reactive policy       System 2 LLM/planner/reflection
          |                             |
          +---------- proposals --------+
                         |
                         v
              Grounding + Intent Arbiter
                         |
                  BehaviorIntent
                         |
                         v
               Embodied Skill Layer
       plan / dispatch / observe / timeout / interrupt
                         |
                    BodyPort
                         |
          robot.* JSON-RPC + robot.state
                         |
                      robotd
              50 Hz policy + safety
                 /               \
        DynamixelIo             RemoteIo
            |                       |
       real Microduck       TCP body protocol
                                    |
                             MuJoCo duck-body
```

The simulated and real paths diverge at official `RobotIo`, below `robotd`.
Duck Mind must not add a separate simulator command vocabulary above that seam.

## 3. Core data flow

```text
Event
  -> state update
  -> grounded context bundle
  -> System 1 decision or System 2 proposal
  -> arbiter
  -> behavior intent
  -> embodied skill execution
  -> robot.state observations
  -> outcome evaluation
  -> episode close
  -> evidence-gated learning proposal
  -> bounded commit
```

Every item uses stable IDs and a shared `trace_id`. An `Episode` groups the
events, intents, outcomes, runtime/model versions, and before/after state hashes.

## 4. Embodied Skill Layer

The layer translates semantic intent into a monitored sequence of allowed
robotd operations. A future `approach(target=alex)` would be a closed-loop
sequence (not yet implemented):

```text
look_at(alex)
  -> rotate toward target
  -> move with bounded vx/vyaw
  -> observe odometry / limits / safety
  -> slow down
  -> stop
  -> evaluate final distance
```

For named episodic official skills, lifecycle evidence is:

```text
PENDING -> ACCEPTED -> STARTED -> COMPLETED
              |          |           |
              |          |           +-- policy exits expected skill
              |          +-------------- robot.state.policy == skill
              +------------------------- RPC queued/accepted only
```

`REFUSED`, `TIMED_OUT`, and `INTERRUPTED` are terminal execution states. A
separate `Outcome` classifies the observed result as success, failure,
interrupted, or unknown. This avoids equating a finished time window with a
successful physical action.

The current live kick scenario reports **execution gate pass** but **task
outcome unknown**: the floor scene has no ball/contact goal. `sit_toggle` has
a separate multi-phase posture contract and is intentionally not accepted by
the generic named-skill adapter.

The implemented first play primitive is `look_game`: a fixed, small head
command, two measured tracking samples, safety/timeout/cancel checks, then
optional explicit owner feedback. It has no arbitrary target coordinates or
joint indices at the cognition boundary. Simulated completion proves tracked
motion, not that a person enjoyed the game.

## 5. Official policy/runtime contract

Current official alpha policies share `obs[1,61] -> actions[1,14]`:

```text
gyro                  3
projected gravity     3
joint position       14
joint velocity       14
previous action      14
command              13
-----------------------
total                 61
```

The command block is twist 3 + head 4 + body 6. Body x, y, and yaw are zero in
the current runtime. Duck Mind does not assemble this vector; `robotd` does.

The pinned v5 walk policy observes head commands as **deltas from HOME**, while
`robot.look` currently returns absolute IK joint angles. In the live MuJoCo
run, `robot.look` moved the head but missed the requested pose by about 0.3
rad; it is therefore not accepted as a completed first-board skill. The
adapter uses a fixed `robot.head` v5 delta profile and checks measured head
tracking instead. Real use requires API 37, an online `robot.policies` report
showing the unoverridden, homed official v5 walk slot at its reviewed path,
and an additional supervised board approval. This checks loaded policy
selection, not the bytes of a running binary, firmware, or the head mechanism;
it is a compatibility workaround, not a claim about future upstream releases.

## 6. State ownership

| State | Owner | Update authority |
|---|---|---|
| servo/joint/safety/policy session | `robotd` | official runtime only |
| body execution lifecycle | Embodied Skill Layer | observed robotd evidence |
| world/self/affect/drives | Duck Core deterministic reducers | events + time |
| semantic/autobiographical memory | Memory adapter | consolidation pipeline |
| preference | Learning engine | bounded, explicit-feedback-backed updates |
| personality/relationships | planned learning rules | no live mutation in this slice |
| narrative/reflection text | System 2 | proposal only; no direct state write |

## 7. System 1 / System 2

- System 1: deterministic or small-model reactive behavior, bounded latency,
  direct access only to the grounded current-state bundle.
- System 2: LLM reasoning, planning, explanation, and reflection. It returns
  proposals with fact IDs; unknown IDs are rejected.
- Arbiter target: confidence, freshness, cooldown, hysteresis, capabilities,
  relationship policy, and safety-independent constraints. The current code
  implements grounding/confidence routing plus a narrow invitation decision,
  not the full general-purpose arbiter.
- Safety: remains inside/below `robotd`; neither system can override it.

## 8. Memory boundary

The event log is the source of truth for lived experience. Retrieval systems
are indexes, not authority. Nowledge Mem or another semantic store can hold
distilled memories and recall context, but it cannot:

- issue body commands;
- mutate Duck state directly;
- convert an LLM claim into an event;
- replace source event IDs;
- block System 1 or safety when offline.

Learning is a two-phase operation: propose a `LearningUpdate`, validate its
outcome evidence and plasticity bound, then commit it append-only.

## 9. Validation topology

Three harness levels are required:

1. Pure replay: frozen event stream, versions, random seed, no simulator.
2. robotd + MuJoCo: same upper contract as hardware, plus a read-only simulator
   oracle for ground-truth assertions.
3. Real robot: driver, actuator, battery, thermal, camera, NPU, and radio gates
   that the twin explicitly cannot prove.

Simulator-only truth such as `trunk_z` must be visible only to evaluators. If it
enters cognition, the simulated Duck would learn from sensors the real Duck does
not possess.

## 10. First implementation slice

Implemented:

- immutable event/episode/outcome/learning schemas;
- slow-plasticity and evidence validation;
- backend-neutral `BodyPort` with a narrow robotd method allowlist;
- typed Unix JSON-RPC adapter shared by simulated and real `robotd`;
- separate operator-enable authority, not exposed to cognition or skills;
- semantic `OneShotSkillIntent` and `EmbodiedSkillLayer`;
- ACK/start/exit/timeout/interruption lifecycle tracker;
- System 1/System 2 grounding gate;
- memory, simulator oracle, and fault-injector protocols;
- replay divergence and counterfactual-cause checks;
- live headless `robotd` + v5 ONNX + MuJoCo scenario, JSONL episode trace,
  wire/oracle replay validator, command-level `RecordedBodyPort` replay through
  the actual Embodied Skill Layer, and adversarial trace-tampering tests.
- simulator-only first-pet `look_game` BodyPort recording and offline replay,
  paired by interaction ID and hash with HOME/SIT service reports; safety-stop,
  preflight, owner feedback and storage evidence remain separate gates.
- single-writer SQLite event log, snapshots, identity, fail-closed reducer,
  explicit source/outcome-linked preference updates, and a Mem outbox boundary;
- optional bounded outbox consumer with source revalidation, stable idempotency
  keys, finite timeout and retry; no external sink or automatic service wiring;
- deterministic 30-day virtual twins with feedback ablation;
- local-only pet service and CLI, fixed measured head game, owner stop,
  body/battery watchdog, incremental unfinished-episode reconciliation,
  no-feedback UNKNOWN outcome, graceful SIGTERM/SIGINT stop, and restart recovery;
- official live pet-service simulator gate including stop and persistence;
- bounded off-board official PPO/checkpoint/export/physics-parity job manager,
  with explicit CPU or Linux/CUDA dependency preflight and no cloud submission
  or policy deployment; four short official BAM trials reject observed falls
  but never qualify a candidate for hardware;
- actual Mac CPU official Microduck/BAM PPO smoke run, saved checkpoint,
  official ONNX export, checkpoint/ONNX parity and failed physical-trial evidence;
- trusted local checkpoint resume across separate CPU jobs, including a
  1000-iteration continuation; all tested candidates remain rejected;
- isolated, source-bound candidate-versus-v5 `robotd --sim` walking comparison;
  it rejects the present candidate without changing the installed policy set,
  but cannot qualify a new gait from four short scenarios;
- independent native-PD PPO balance experiment with paired seeded physics
  evaluation and saved-checkpoint/ONNX numerical parity (not BAM equivalence);
- standalone feed-forward ONNX compatibility inspector, deterministic CPU probes
  and content identities (not a physical-quality or normalizer-parity proof);
- versioned board wheel/service installer and rollback candidate;
- release-scoped, supervised real head-profile approval that fails closed if a
  retained systemd drop-in targets a different installed release, or the live
  `robot.policies` walk slot is disabled, unhomed, overridden or unexpected;
- read-only systemd pre-start guard that refuses to boot a split application
  and installed-unit pair after an interrupted deployment.
- process-kill fault injection across learning update/event COMMIT and the
  event-to-snapshot gap, with replay and idempotent retry on restart.

Not yet implemented or proven: real camera/ToF/touch intake, navigation and
sit-toggle controllers, full System 2 runtime, production Mem sink/consent,
outbox scheduling and semantic retrieval, remote
owner identity/lease, a usable new gait proven across command/scenario coverage,
broader robotd candidate/baseline physical simulation acceptance,
long-running training-campaign scheduling, and physical hardware acceptance.
The first playable slice is deliberately local and owner-supervised.
See [official RL workflow and evidence limits](docs/training/official-rl.md).
Local reproducible commands: [Mac CPU training](docs/training/local-cpu.md).
