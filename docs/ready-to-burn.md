# Ready-to-Burn Definition and Gate Status

Status: **software release candidate; NOT CLEARED FOR UNSUPERVISED REAL PLAY**
(2026-09-24 Asia/Taipei). The first pet service and official simulator gates
work. The author's real robot has not arrived yet; the exact
board and physical acceptance gates remain pending delivery.

Reference target: official Radxa Zero 3 / Armbian 26.2.1 Minimal with official
Microduck daemons provisioned first. This is an assumption from upstream
documentation, not confirmation of any actual board revision or wiring.

## Product promise for the first burn

After official base provisioning and Duck Mind installation, the duck should:

1. start Duck Mind passive, issuing no play commands until an owner explicitly
   enables it; official robotd and its human controls remain independent;
2. expose a local health/status and immediate owner stop path;
3. accept a local owner invitation to one bounded, measured head game without
   an LLM or Nowledge Mem connection;
4. persist identity, events, preferences and episode/outcome evidence across
   process restarts and power loss;
5. learn only from supported, non-duplicated feedback, with bounded changes
   and replayable attribution;
6. record enough data for offline simulator/regression and off-board policy
   training, but never retrain or replace the motor policy on the live robot;
7. fail closed on body disconnect, bad/unknown real battery, fall/limp, policy
   mismatch or service restart; unavailable external memory must not invent
   facts, block local evidence capture or weaken body safety; and
8. support health-gated application and matching-unit rollback without
   disturbing official `robotd`.

"Training" has two meanings that must not be conflated: on-device preference
learning in this slice is a bounded state update (relationship learning is
future work); gait/skill RL training
runs **off-device** in `microduck_rl`, then crosses official policy validation
and a human-controlled deployment gate. An LLM never changes either motor
weights or safety limits by itself.

## Acceptance matrix

| Gate | Required evidence | Current |
|---|---|---|
| Source/architecture | pinned contracts; no cognition-to-servo path | reviewed |
| Body simulator | real `robotd` + v5 ONNX + MuJoCo; replayable trace | latest execution gate and replay pass; six configured ONNX file hashes now match across pet/skill scenarios; task outcome unknown; loaded process bytes and physical stability unproven |
| Durable pet runtime | SQLite WAL/FULL, single writer, digests, snapshots/replay, passive restart | implemented; watchdog replays unfinished traces once at start, then reads only new events; simulated restart and process-kill cuts around learning COMMIT/snapshot pass; board power-loss unproven |
| Interaction and play | local owner-only socket, stop, watchdog, fixed measured head game | official HOME-positive, SIT-positive and HOME-negative scenarios pass end-to-end with paired gaze BodyPort command replay; policy and commanded-motion guard rejects competing body activity and immediately revokes play after failed gaze; real preflight additionally checks the loaded official walk slot via `robot.policies`; child-process SIGTERM/SIGINT shutdown pass; real profile locked |
| Learning | completed execution + explicit feedback + closed outcome, daily 0.005 cap, ablation | positive and negative simulator feedback changed preference by at most 0.005 in the expected direction and survived passive restart; unit/30-day/365-day synthetic twins pass; real owner feedback untested |
| Mem boundary | optional async outbox, outage and poisoning tests | local bounded worker and fake-sink outage/poison/timeout tests pass; production sink, consent/producer, scheduler and retrieval not wired |
| Simulation training | scripted twins, ablation, replay, official body | official Mac CPU PPO checkpoint/export/parity ran; the 2000-iteration candidate fell 4/4 in formal BAM trials and 1/4 in a diagnostic nonplay profile, then failed paired `robotd --sim` idle/forward-stop. Two short trusted continuations also fell 4/4. A further 1000-iteration continuation fell 4/4 formally despite 0/4 in a non-qualifying nonplay diagnostic, and failed paired `robotd --sim`. A paired `play-zero-push` ablation preserved initial actor state and event timing but fell 0/4, narrowing the failure to push sensitivity under four short seeds; it is explicitly non-qualifying. A subsequent 500-iteration continuation with 1–2 s training pushes made new checkpoint/ONNX bytes but still fell 4/4 formally and failed the independent robotd walk pair. The unchanged official v5 baseline passed the short comparisons. Earlier 1000-iteration and independent native-PD candidates also fell. Usable new gait, broad robotd acceptance and hardware behavior remain unproven |
| Board release | source-matched wheel, hashes/SBOM, installer, systemd, health, rollback | isolated installer/rollback and SIGKILL cut-point tests pass; boot guard and install/rollback preflight reject a split unit/application, and head approval is release-bound; real Linux/systemd and board install pending |
| Security/QA | lint, type check, tests, fault injection, self-review | local regression and deployment fault-injection checks; exact results belong to the selected release evidence; real adversarial tests pending |
| Hardware | exact board/servo revision and supervised bench safety tests | **pending robot delivery** |

The full upstream `scripts/duck-sim` launcher, camera/ToF/BLE/audio paths, a
charging dock, and physical-contact kick success are **not** proven by the
current execution trace. Each needs its own gate before enabling dependent
behaviors. No automatic locomotion should be enabled on first boot merely
because MuJoCo ran once.

The earlier wheel smoke check installed the package into a clean Python 3.12
virtual environment and ran both `duckmindd --help` and `duckmindctl --help`.
That result proves package loading for that artifact. Each new bundle must
carry its own exact wheel hash, source identity, checks, and SBOM; use the
release-candidate evidence and
the selected bundle's manifest rather than reusing a previous wheel hash.
No release has been transferred to or accepted on a real robot.

Deployment fault injection runs copied scripts against isolated filesystem
paths and command doubles. It checks handled install/rollback failures,
database backup, passive health, and restoration of configuration/state.
It does not prove actual Linux systemd sandbox behavior or board permissions.
Those remain required after the hardware arrives.

The first-board activity is owner-invited `look_game`, not autonomous
camera-based tracking yet. The official v5 `robot.look`/policy head-command
unit mismatch made an accepted look fail the measured gate, so only a fixed
small v5 head delta passed in simulation. That workaround is locked on real
hardware until a supervised bench check. The new online slot/path check can
reject same-basename overrides; the new on-disk SHA-256 check also rejects
changed or missing walk files. Neither check nor API 37 proves robotd's
already-loaded policy bytes,
the board's running binary revision (`hello.revision` was null locally), or
mechanical equivalence. `robot.stop` is not a torque-off emergency stop.
The service now catches normal SIGTERM/SIGINT and runs its own shutdown stop
path before exit; systemd `ExecStopPost` remains an independent best-effort
backup. SIGKILL, power loss, an unresponsive Python thread, and robotd failure
still require robotd's own safety behavior and a physical kill path.

## Unblock the real-board gate

After delivery, record the exact Microduck board, servo/head revision,
`robotd` binary and policy manifest hashes, power setup, camera/ToF/voice
availability, and the official base-provisioning health report. Then perform the
[supervised board checklist](deploy/board.md) with a physical kill path, small
head command, measured target/actual tracking, stop latency, fall/low-battery
refusals, restart/rollback, and a few real owner-feedback episodes. The
approval drop-in may be installed temporarily for the controlled trial with a
physical kill path; retain it for later supervised play only after acceptance.

## Release/rollback rule

Ship Duck Mind as a separate service and versioned application directory,
never overwrite the official Microduck daemon, policy set, or safety config.
Start with play disabled/passive, verify health and evidence storage, then enable a
small owner-supervised activity set. Rollback stops Duck Mind, restores the
selected application's matching archived systemd unit and application
pointer, and preserves the append-only state database. It refuses a release
without its archived unit. A handled install failure restores the prior unit,
application pointer, and enabled/active state. A failed rollback restores the
previously selected unit/pointer and prior active state without changing
enablement. Recovery must report failure if it cannot stop the service,
restore configuration, or verify the required passive health.
Do not roll back a schema by deleting data; use forward-compatible migration
or keep the previous binary able to read the new schema.

Systemd drop-ins, including head-profile approval, remain operator-managed;
neither install nor rollback versions or automatically restores them. Review
their compatibility with the selected release before a supervised trial.

Atomic symlink replacement does not make the entire deployment power-loss
safe. Handled errors and SIGTERM run recovery; SIGKILL and power cuts do not.
The read-only boot guard rejects a mismatched installed unit and selected
application before the service starts. Install and rollback also refuse to
mutate an already split installation. These paths were tested with isolated
SIGKILL injection, not with real systemd or power interruption. Automatic
recovery after reboot is not implemented. An interrupted deployment
requires inspection of the application pointer, unit/drop-ins, recovery
files, database, and journal while play stays disabled. SQLite durability
does not by itself prove install/rollback crash recovery.
The [learning crash-recovery tests](validation/learning-crash-recovery.md)
exercise abrupt process exit around the SQLite commit boundary. They do not
replace a supervised board power-cut/boot inspection or review of the board's
Python-linked SQLite version.

The final go/no-go decision requires both a green simulator/regression report
and supervised real-device evidence. A release that has only the former is a
candidate, not "ready to burn".

The [off-board RL workflow](training/official-rl.md) records preflight,
checkpoint/export, ONNX compatibility and short official physics/parity gates.
The [paired robotd diagnostic](validation/robotd-candidate.md) rejects the
tested candidates without modifying the installed policy set.
An observed fall rejects a model; four clean trials would still be inconclusive.
No model can pass hardware or policy-quality gates by implication. Earlier
release candidates remain immutable historical artifacts. Each newer bundle
must carry its own source-matched wheel, fresh simulator/learning evidence and
complete release checks; inspect that bundle's manifest rather than applying
another candidate's test claims to it.
