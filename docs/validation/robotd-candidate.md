# Isolated robotd candidate comparison (local Mac)

Status: diagnostic rejection gate, **not** gait certification or hardware
approval. The author's duck has not arrived yet. This uses the reviewed official
`microduck` revision `a9ec4b2079ef8ee7904014089c885bb07d57d63c` and
`microduck_rl` revision `cb70b792312d559a4da09064d92009079671815f`.

## What is actually shared with the real path

The official `robotd --sim` runs its control loop, policy runtime, arbitration,
JSON-RPC and safety path. The [official simulation path](https://github.com/pollen-robotics/microduck/blob/a9ec4b2079ef8ee7904014089c885bb07d57d63c/docs/robot/simulation.md)
replaces `RobotIo` with a MuJoCo bridge below `robotd`. Duck Mind sends the
same semantic `robot.move` and `robot.stop` through `RobotdBodyPort` in both
cases; cognition never writes joint or servo targets. The validation-only
`FixedForwardProbeSkill` owns a fixed forward speed and stop sequence, and
checks command ACKs. The [body server](https://github.com/pollen-robotics/microduck_rl/blob/cb70b792312d559a4da09064d92009079671815f/src/mjlab_microduck/sim/body_server.py)
also provides `trunk_z` and `sim_time` as an **oracle for the validator only**;
those fields must not enter cognition or a real-robot contract.

The harness loads a candidate into a new, temporary simulator TOML walking
slot. It does not alter the installed v5 policy set, deploy to `/opt/robot`,
or make the candidate callable by the pet service. The official
[policy manifest contract](https://github.com/pollen-robotics/microduck/blob/a9ec4b2079ef8ee7904014089c885bb07d57d63c/docs/policy-manifest.md)
and `robot.subscribe` walking basename are checked. A basename is not
enough to prove resolved model bytes, so the report also binds the entire
installed set, candidate copy, effective config, runtime binary/library, scene,
body server, source revisions and Python source hashes.

## Reproduce

Use the clean upstream checkouts, local v5 policies, built `robotd` and a
managed candidate job. Substitute their absolute local paths:

```bash
uv run --locked --python 3.12 --with mujoco==3.10.0 --with numpy \
  --with onnxruntime==1.27.0 python -m duckmind.robotd_compare \
  --candidate-job /abs/duckmind/runs/rl/mac-managed-gate-002 \
  --microduck /abs/microduck --microduck-rl /abs/microduck_rl \
  --policies /abs/official-v5/current --robotd-bin /abs/robotd \
  --report /abs/duckmind/runs/validation/robotd-pair-007.json
```

The report path must be new: the harness refuses to overwrite evidence. A
rejection exits nonzero. Compare the report's `assessment` and source/model
hashes, not just its process exit code. The four runs start from fresh MuJoCo
HOME keyframes: baseline and candidate each get idle and forward-then-stop.
The initial pose hash must match for every pair. If the baseline fails to
reach stable walk, command is not applied, stop is not observed, or simulation
is slower than 0.95× real time, the pair is **invalid**, not a candidate
failure. The short pair can reject an observed failure; even four clean runs
would remain **inconclusive**.

The readiness phase is intentionally light on extra MuJoCo reads: frequent
oracle polling during homing perturbed the baseline. `ready_fallen` is retained
as diagnostic data, while the walking-phase fall/limp flags determine active
stability. This does not hide a candidate that never reaches healthy walk;
that is a separate rejection reason. This timing sensitivity is itself a
reason not to use this short harness as certification.

## Recorded local result

The compact, recomputable summary
binds raw `runs/validation/robotd-pair-007.json` by SHA-256. The baseline
entered `walk` in both scenarios, remained above 0.10 m in the measured active
phase, applied the forward command, then converged after stop. The candidate
entered neither healthy walking phase and ended around 0.035 m trunk height in
both scenarios. Both pairs had matching HOME hashes and ~1.0× simulation
time. Assessment: **rejected**. The candidate's managed training job had
already failed 4/4 BAM trials; this independent `robotd` path corroborates
rejection, not deployment readiness. `physical_hardware_gate` and
`policy_deployment` remain `not_run`.

This is not a whole-body acceptance campaign: no terrain/perturbation breadth,
long-duration gait, full skill arbitration, ball/contact, real actuators,
power/thermal limits, or board deployment is tested here. A later candidate
needs its own source-bound, longer validation and supervised bench gate.

## 1000-iteration candidate follow-up

The same isolated harness compared `mac-official-1000-001` with the unchanged
official v5 policy set, arm64 `robotd` binary and HOME-keyframe scenes. The
compact evidence binds the new
raw `runs/validation/robotd-pair-008.json` by SHA-256. The baseline entered
walk in both scenarios, applied the forward request and confirmed stop;
minimum active trunk height stayed above 0.11 m. The new candidate entered
neither healthy walking phase and reached about 0.044 m in both. Initial HOME
pose hashes matched and simulation ran around real time. The verdict remains
**rejected**, with no policy deployment or hardware gate claimed.

## 2000-iteration candidate follow-up

The new seed-11 `mac-official-2000-001` candidate first failed all four
held-out official BAM physics trials despite successful checkpoint/ONNX parity.
The isolated `robotd --sim` pair then used matching HOME initial pose hashes
and ran at approximately real time. The v5 baseline entered walk in both
scenarios, applied forward and confirmed stop. The candidate entered neither
healthy walk and reached about 0.035 m trunk height in both. The
hash-bound summary references
the ignored raw `runs/validation/robotd-pair-009.json`. This corroborates
rejection only; no physical or policy-deployment approval is claimed.

## 3000-checkpoint continuation follow-up

The 1000-iteration continuation `mac-official-resume-3004-003` passed
checkpoint export and ONNX parity but fell in all four formal `play=True`
held-out BAM trials. Its 0/4-fall `play=False` run is a diagnostic using a
different push profile, **not** a replacement acceptance gate. The independent
`robotd --sim` pair used the same installed v5 baseline and matching HOME
initial pose hashes. The baseline reached walk in idle and forward/stop,
applied the forward request, and confirmed stop; the candidate reached neither
healthy walk phase and fell to about 0.044 m trunk height in both. The
hash-bound continuation summary
binds the ignored raw `runs/validation/robotd-pair-010.json` and training
reports. Assessment: **rejected**; no policy deployment or physical approval.

## Training push-schedule continuation follow-up

The 500-iteration continuation `mac-official-push-resume-3500-005` used an
official training-time `push_robot` interval override of 1–2 s. It produced a
new checkpoint/ONNX but still fell in all four formal `play=True` BAM trials.
The isolated `robotd --sim` comparison then used matching HOME initial-pose
hashes for the v5 baseline and new candidate. The baseline entered walk in
both scenarios, applied forward and confirmed stop; the candidate entered
neither healthy walk and dropped to about 0.044 m trunk height. The
hash-bound push-continuation summary
binds `runs/validation/robotd-pair-011.json`. Assessment remains **rejected**;
the override is a training experiment, not a relaxed qualification gate.

The full readiness trace sharpens the diagnosis. `walk_reached=false` means
**healthy walking was not reached**, not that the walk policy was never selected.
In the two current HOME-start scenarios, both policies first reported `walk`
while `fallen=true`. The v5 baseline cleared `fallen` at 2.31 s (idle) and
2.26 s (forward/stop); the new candidate entered `walk` at 2.14 s in both,
but was still `fallen=true` at the last readiness observations around 12.1 s.
The earlier 3000-checkpoint candidate showed the same pattern in its separate
hash-bound comparison.
The current summary
records the exact first-policy, first-healthy and terminal readiness times;
its regression test recomputes those fields from the raw local report when
present.

## One-shot first-walk simulator oracle (same ONNX Runtime as prior pair)

The new hash-bound summary
recomputes from ignored raw `runs/validation/robotd-pair-013.json` when that
file is present. This run used the **same 1.27.0 ONNX Runtime library bytes**
as pair 011, plus the same robotd binary, v5 files, scene, body server and
candidate ONNX. An initial exploratory pair 012 accidentally used ONNX Runtime
1.24.4; it also rejected the candidate, but its differing runtime hash makes
it unsuitable for a controlled cross-run comparison. Pair 013 is the recorded
comparison here. All four runs began with the same 0.125 m HOME pose hash.

The harness took exactly one MuJoCo oracle snapshot immediately after the
first **observed** `walk` status in each run. At that sample, the v5 baseline
trunk measured 0.043–0.049 m and the candidate 0.043 m; all four robot status
observations reported `fallen=true` and
body-frame projected gravity was dominated by its x component. The v5 VelStand
baseline then reached healthy walk, rising to about 0.116 m and proving the
forward/stop command path; the velocity-only candidate remained around 0.044 m
and never reached healthy walk within the short readiness deadline. Both
baseline and candidate ran at about 1.0× real time. Assessment: **rejected**;
hardware and policy deployment remain `not_run`.

This locates a **state near first observation of `walk`**, not the exact instant
the policy became active or when the body fell. The robot status and subsequent
oracle sample are not atomic. The one extra oracle read also
has possible timing effects. The official [task list](https://github.com/pollen-robotics/microduck_rl/blob/cb70b792312d559a4da09064d92009079671815f/README.md)
distinguishes Velocity walking from VelStand walking **plus fall recovery**;
its listed Velocity walk model strips trunk/head contacts while VelStand uses
ground-contact geometry. The pinned
[VelStand configuration](https://github.com/pollen-robotics/microduck_rl/blob/cb70b792312d559a4da09064d92009079671815f/src/mjlab_microduck/tasks/microduck_velstand_env_cfg.py)
explicitly trains protective falls and recovery. Those official distinctions
fit the observed result, but do not prove that task choice is the sole cause:
reset distribution, control timing and actuator semantics also differ. The
next *local* gait experiment should be recovery-capable and include the same
HOME→walk probe plus unchanged formal play evaluation. It must not silently
fetch teacher checkpoints or promote a failed weight. Cognition still sees
only semantic body observations and sends motion through the skill layer and
`robotd`; MuJoCo oracle fields stay inside validation.

The official body server [holds HOME until robotd takes control](https://github.com/pollen-robotics/microduck_rl/blob/cb70b792312d559a4da09064d92009079671815f/src/mjlab_microduck/sim/body_server.py),
and `robotd` passes through homing before walk. By contrast, the pinned
[PPO task config](https://github.com/pollen-robotics/microduck_rl/blob/cb70b792312d559a4da09064d92009079671815f/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py)
resets the base around 0.12–0.13 m and leaves base-orientation reset
randomization off. A startup/handoff distribution gap is therefore a concrete
*hypothesis* for the robotd failure, not an established cause: observations,
actuation and control timing also differ between the two paths. The next gait
experiment should target and measure HOME→walk recovery explicitly, then rerun
unchanged formal play and paired robotd gates. No failed candidate is installed.
