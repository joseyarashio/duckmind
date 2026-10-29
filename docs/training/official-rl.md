# Official off-board RL training and candidate verification

This workflow trains motor policies in the **official simulator**, never on a
live robot. Cognition does not gain servo access. On-device preference learning
(`train_sim` cognitive twins) and PPO motor-policy training are different jobs.

Status: the job orchestration and candidate inspector are implemented and
tested locally. The official v5 `velstand.onnx` passed a real CPU ONNX probe.
Mac CPU physics and actual official PPO training have now been exercised with
mjlab 1.3.0 / Warp 1.12.0 / MuJoCo 3.10.0 and the pinned BAM task. The first
4-environment / 5-iteration run produced a checkpoint and official ONNX export,
and passed compatibility inspection. A later managed 4-environment run also
executed the official physics/parity probe automatically: parity passed, but
all four trials fell and the candidate was rejected. This is a training-pipeline
result, **not a usable gait**. See [local CPU commands and experiment limits](local-cpu.md).
The author's robot has not arrived yet.

A subsequent 32-environment / 100-iteration run also completed. Official
checkpoint/ONNX parity passed on 235 actual simulator observations, but the
candidate fell in all four physical trials. It remains quarantined; the exact
[local results](local-cpu.md#official-checkpoint-parity-and-physical-evaluation)
separate numerical correctness from policy quality.

A 32-environment / 1000-iteration CPU follow-up also produced and exported a
checkpoint. Both the formal `play=True` probe and a diagnostic `play=False`
probe recorded **4/4 falls**; a separate paired `robotd --sim` run rejected the
same candidate in idle and forward/stop. The diagnostic profile cannot qualify
a policy. See the hash-bound result.

A fresh 32-environment / 2000-iteration CPU run (seed 11) again passed
checkpoint export, ONNX compatibility and 339-sample checkpoint/ONNX parity,
but **4/4 held-out `play=True` trials fell**. The isolated `robotd --sim`
pair also rejected it in idle and forward/stop. See the
source- and artifact-bound summary.
Neither failed candidate replaces official v5.

The same 2000-iteration checkpoint later fell in 1/4 held-out trials under a
diagnostic `play=False` profile, with 802-sample ONNX parity. This suggests an
evaluation-profile sensitivity but is not proof of a specific cause or an
alternate qualification path. See the hash-bound nonplay diagnostic.

Two further **two-iteration continuations** of that trusted checkpoint were
executed locally. The official runner loaded the staged checkpoint, advanced
its step counter from 48000 to 48048 to 48096, and produced new checkpoints
and ONNX exports. Both passed checkpoint/ONNX parity but fell in **4/4 formal
`play=True` trials**, so both remain rejected. This verifies the resume
mechanism, not gait quality. See the hash-bound resume chain.

A further 1000-iteration continuation (same seed and 32 environments) advanced
the official runner to 72096 common steps. The new `model_3000.pt` exported
and matched ONNX on 403 formal simulator observations, but **all four
`play=True` trials still fell**. A separate `play=False` diagnostic ran to
250 steps with 0/4 falls, while the paired `robotd --sim` candidate entered
neither healthy walk scenario. The easier profile is not an alternate pass;
the model remains quarantined. See the hash-bound 3000-iteration result.

A same-checkpoint push ablation then preserved the official play event timer
and RNG sampling but zeroed its x/y push velocity. Initial actor observations,
actions, commands and compiled model hashes matched formal play for all four
seeds. Formal play fell 4/4; zero-push play fell 0/4 over 250 steps each, with
1000-sample checkpoint/ONNX parity. This narrows the observed failure to
disturbance sensitivity under the tested seeds; it is not a train/deploy pass.
The ablation is marked diagnostic-only, and qualification rejects it. The
paired `robotd --sim` rejection remains in force. See the same
hash-bound summary.
A subsequent local training experiment changed only the official task's **training**
push interval through mjlab's CLI, leaving the pinned upstream source and
formal `play=True` evaluation untouched. Its job report retains the
exact interval, parent checkpoint hash and command; any resulting checkpoint
still needs export, parity, held-out play and independent robotd validation.
The first 500-iteration 1–2 s push-schedule continuation did produce a new
checkpoint and passed 395-sample ONNX parity, but it still fell in 4/4 formal
play trials and failed both independent robotd walk scenarios. It is
quarantined. See the hash-bound continuation result.
The same new checkpoint then passed a diagnostic-only, paired `play-zero-push`
run: 0/4 falls over four 250-step trials and 1000/1000 checkpoint/ONNX parity
samples. Initial actor state, first action/command and compiled model hashes
matched formal play for every seed; actions diverged at steps 48/33/43/39.
This reinforces observed push sensitivity under these four seeds, but the
unchanged formal and [HOME→walk robotd result](../validation/robotd-candidate.md#training-push-schedule-continuation-follow-up)
still reject the model. The robotd trace shows policy `walk` began while
`fallen=true` and never became healthy by the ~12 s readiness deadline;
homing-to-policy transfer is a separate unqualified condition, not proof that
the push schedule alone is the cause.

## Reviewed recipe and platform boundary

Use a clean `microduck_rl` checkout at
`cb70b792312d559a4da09064d92009079671815f`. Its Python requirement is
`>=3.12,<3.13`; the reviewed stack includes mjlab 1.3.0, Warp 1.12.0,
Torch 2.9.1 and rsl-rl 5.0.1. Prepare that separate environment using the
official lockfile before running Duck Mind's job manager. It does not install
dependencies or submit cloud jobs. Linux aarch64 must retain the upstream
cu129 Torch source; do not substitute a CPU-only PyPI wheel.

The first managed profile is `Mjlab-Velocity-Flat-MicroDuck`, with tensorboard
logging, upload disabled and no video. A fresh job has resume disabled;
an explicit `--resume-job` can continue a trusted completed local job. CUDA
remains the default;
the explicit `--device cpu` profile selects mjlab's `--gpu-ids None` path and
exports on CPU. The CPU profile is experimental and is not a throughput claim.
Default 64 environments / 5 iterations is an infrastructure smoke run, **not a
usable gait**. Increase the explicit iteration/time budget on a suitable host
for a real experiment; do not infer quality from checkpoint existence.

The upstream README lists CUDA as its training prerequisite. However mjlab has
an actual CPU path and the pinned dependencies have compatible Mac wheels.
An earlier Duck Mind preflight required Linux/CUDA too broadly. We retain the
CUDA profile's requirements but admit CPU explicitly, testing the complete
Microduck/BAM task before claiming training support. Never infer CPU failure
merely from CUDA being unavailable.

VelStand cannot silently replace the first profile: its default expert BC
configuration resolves two remote W&B checkpoints independently of the logger.
W&B disabled/offline alone does not make that training recipe offline. A future
VelStand profile must explicitly pin and supply both local expert/anchor
checkpoints and record their hashes. No `--hf-jobs`, Hub publication or robot
installation is performed by this manager.

## Plan, preflight, execute

From the Duck Mind checkout, substitute your clean upstream checkout and its
prepared Python environment. The `--python` path is the trainer interpreter,
not necessarily the interpreter running Duck Mind.

```bash
uv run --locked --python 3.12 python -m duckmind.training plan \
  --upstream /abs/microduck_rl --python /abs/microduck_rl/.venv/bin/python \
  --job /abs/duckmind/runs/rl/walk-smoke-001

uv run --locked --python 3.12 python -m duckmind.training preflight \
  --upstream /abs/microduck_rl --python /abs/microduck_rl/.venv/bin/python \
  --job /abs/duckmind/runs/rl/walk-smoke-001

uv run --locked --python 3.12 python -m duckmind.training run --execute \
  --upstream /abs/microduck_rl --python /abs/microduck_rl/.venv/bin/python \
  --job /abs/duckmind/runs/rl/walk-smoke-001 \
  --num-envs 64 --max-iterations 5 --seed 0 --timeout-s 600
```

`plan` does not create files or launch a trainer. `preflight` checks the reviewed
Git revision, clean source, Python/dependency versions, task entry point,
imported checkout and selected compute profile (visible CUDA for CUDA only).
`run` requires explicit execution,
a new directory and successful preflight; an existing job is never overwritten
or silently resumed. A blocked preflight is written as such, with training
`not_run`. A new full-training job can use e.g. `--max-iterations 30000` with an
explicit suitable time limit (maximum 86400 seconds); it is not automatically
restarted after timeout. To add iterations to a completed, validated local job:

```bash
uv run --locked --python 3.12 python -m duckmind.training run --execute \
  --upstream /abs/microduck_rl --python /abs/duckmind/.venv-official-cpu/bin/python \
  --resume-job /abs/duckmind/runs/rl/previous-job \
  --job /abs/duckmind/runs/rl/next-job \
  --device cpu --num-envs 32 --seed 11 --max-iterations 1000 --timeout-s 86400
```

For a resumed job, `--max-iterations` is the number of **additional** PPO
iterations, not the final iteration index. Use a new output directory, the
same device/environment count/seed and identical reviewed upstream source and
lockfile. The manager checks the completed parent's status, simulator verdict,
source identity and checkpoint digest; copies that checkpoint into the new
job; then verifies its bytes and parent report across training/export/probe.
The copied input cannot count as a newly produced checkpoint. The parent may
have been simulator-rejected: continuing training does not promote it or
weaken any later quality gate. The CLI exits nonzero for a rejected candidate
even when all pipeline stages completed successfully.

Only point `--resume-job` at a **trusted local** Duck Mind job. Pinned
PyTorch/rsl-rl checkpoint loading uses pickle-backed deserialization, so a
crafted `.pt` file is executable code; digests and job metadata are integrity
checks, not authentication against a hostile local writer. Interrupted,
timed-out or partially completed jobs are not automatically resumable. The
resume execution path has been exercised on this Mac CPU profile; CUDA resume
has not yet been exercised.

Outputs are routed through an absolute `--agent.experiment-name` into the job
directory, not the upstream checkout. mjlab adds its timestamp/run-name
subdirectory. The manager selects an actually produced `model_N.pt`, invokes
the official `scripts/export.py --checkpoint-file ... --onnx-file ...` and
then executes the standalone candidate inspector and official CPU physics/parity
probe with the trainer interpreter. The probe runs inside the same bounded job
deadline and writes a source/model-bound report; CPU evaluation after CUDA
training is not a claim of CUDA equivalence.
Export remains on the training host because the official exporter instantiates
the environment and runner. It is not just a CPU file converter.

The child environment disables W&B/Hub telemetry and removes API-key variables;
it is not a general network security sandbox. Only the reviewed, local-only
profile is exposed. Third-party code and checkpoints must be trusted; do not
use this command to execute arbitrary downloaded `.pt` files.

## Evidence and gates

```text
source/environment preflight
  -> bounded official PPO process
  -> actual checkpoint + digest
  -> official export (normalizer included by the official path)
  -> CPU ONNX compatibility probe
  -> official Microduck/BAM CPU physics + checkpoint/ONNX parity (4 seeds)
  -> candidate_rejected OR candidate_unqualified
  -> [still required] wider same-body skill/robotd candidate comparison
  -> [still required] human-reviewed physical bench acceptance
```

`job.json` records actual commands, exit/timeout states, source/lockfile
identities, the Duck Mind source fingerprint and artifact digests. Logs and
checkpoints stay under the new job directory. `training_gate=checkpoint_produced`
is intentionally weaker than "trained a good gait". Training exit zero does
not substitute for export success: upstream's automatic export can warn and
continue, so Duck Mind invokes and verifies the dedicated official exporter.

`normalizer_provenance=official_checkpoint_export` says which export route
ran. It is not a numerical parity proof. The isolated ONNX inspector reports
normalizer provenance as unverified, because tensor shape and inference alone
cannot prove normalization was embedded correctly. The managed official probe
compares the checkpoint inference actor (including its fitted normalizer) with
ONNX on actual simulator observations, then records fall/termination evidence.
It binds the report to checkpoint, ONNX, script, upstream commit and lockfile
hashes. The previous manual probe for `mac-official-002` remains historical;
its failing physical trials already prohibit promotion of that candidate.

The final local status is `candidate_rejected` after an observed fall or
`candidate_unqualified` when four short trials reveal no rejection condition.
The latter is **not a pass**: the manager has no policy-quality approval state.
The `simulator_gate` is `failed` or `inconclusive`; `physical_hardware_gate`
remains `not_run`. CLI `run` exits nonzero for either outcome so automation
cannot silently treat a checkpoint or parity result as a deployable policy.
Missing checkpoint, failed export, invalid graph, source drift, timeout and
interruption do not promote a policy. SIGKILL/power loss cannot run Python
cleanup; an interrupted job must be inspected rather than automatically resumed.
Cancellation/timeout cleanup owns only its process group and adds at most
3 seconds TERM grace plus 3 seconds leader wait. A process deliberately escaping
with `setsid` is outside this guard; this is not an OS security sandbox.

## CPU compatibility inspector

The runtime package has no mandatory ML dependency. For an isolated CPU check:

```bash
uv run --locked --python 3.12 --with numpy --with onnxruntime==1.27.0 \
  python src/duckmind/policy_check.py \
  --onnx /abs/candidate.onnx --report /abs/candidate-compatibility.json \
  --steps 50 --seed 0 --command-profile locomotion --timeout-s 30
```

The current inspector supports feed-forward/API 1 policies only. It requires
one `obs` float32 `[1|dynamic,61]` input and one float32 `[1|dynamic,14]` output;
the output name is unrestricted. It rejects recurrent/API 2 rather than
pretending to simulate its state. Controlled probes follow the actual 61-slot
observation layout, normalize gravity, and feed back raw previous actions.
NaN/Inf, wrong tensor contracts and temporally constant output fail. This is
only compatibility, not reward, stability, action safety or useful behavior.

The real official-model result is
official-v5-policy-check.json.
That artifact is a baseline model check, not a newly trained Duck Mind model.
The real ONNX backend also rejects three generated negative graphs (wrong input
name, NaN output, and constant-but-different-per-joint output): see
negative evidence.
Reproduce with `uv run --no-project --with onnx==1.20.1 --with onnxruntime==1.27.0
--with numpy python tests/probe_onnx_backend.py --report /abs/negative-report.json`.

## Skill/robotd promotion remains separate

The managed short probe uses the official RL task and candidate policy, but does
not exercise robotd's policy set, skill arbitration, command timing or safety
layer. The existing named-skill trace is lifecycle replay, not re-running new
weights. A separate [local `robotd --sim` pair](../validation/robotd-candidate.md)
checks the walking slot with real robotd against the official v5 baseline:
both tested new candidates failed to reach healthy walk and were rejected. This is
only an observed-failure diagnostic; before *promoting any future candidate*,
a broader harness must:

- copy an immutable policy set into an isolated workspace, hashing every ONNX,
  effective TOML, scene and runtime library;
- configure actual skill timing/command/scale/unwind values explicitly: the
  pinned robotd set-manifest lookup uses `/opt/robot/policies/current/manifest.json`,
  so a temporary directory's manifest is not automatically authoritative;
- avoid legacy `kick_left`/other top-level policy fields overriding a candidate
  named skill; verify the daemon's resolved paths, not only a basename;
- run baseline and candidate over broader paired scenarios beyond the current
  HOME-keyframe idle/forward/stop probe, including terrain and perturbations;
- retain warnings/falls/joint limits as acceptance evidence, and distinguish
  execution from task success; a kick without ball/contact evidence stays unknown.

No candidate is copied into `/opt/robot`, published to a Hub, hot-loaded by a
real daemon or made callable by cognition through this workflow.

## Pinned primary sources

- [microduck_rl recipe and CUDA requirement](https://github.com/pollen-robotics/microduck_rl/blob/cb70b792312d559a4da09064d92009079671815f/README.md)
- [Training CLI and output directory](https://github.com/mujocolab/mjlab/blob/v1.3.0/src/mjlab/scripts/train.py#L166-L187)
- [Logger/upload configuration](https://github.com/mujocolab/mjlab/blob/v1.3.0/src/mjlab/rl/config.py)
- [Official local checkpoint exporter](https://github.com/pollen-robotics/microduck_rl/blob/cb70b792312d559a4da09064d92009079671815f/src/mjlab_microduck/export.py)
- [VelStand expert dependency](https://github.com/pollen-robotics/microduck_rl/blob/cb70b792312d559a4da09064d92009079671815f/src/mjlab_microduck/tasks/microduck_velstand_env_cfg.py)
- [Embedded normalizer](https://github.com/leggedrobotics/rsl_rl/blob/v5.0.1/rsl_rl/models/mlp_model.py#L203)
- [Rust model contract](https://github.com/pollen-robotics/microduck/blob/a9ec4b2079ef8ee7904014089c885bb07d57d63c/duck-control/src/policy.rs)
