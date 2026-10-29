# Local Mac training and validation

There are two deliberately separate experiments. Neither writes to a robot,
publishes weights, nor changes the cognition/skill/robotd safety boundary.

| Path | Physics and learner | Meaning |
|---|---|---|
| `duckmind.training --device cpu` | Official pinned Microduck task, BAM, mjlab/Warp, rsl-rl PPO | Preferred actual official training route on this Mac; small CPU runs are infrastructure/learning experiments, not usable gait acceptance |
| `duckmind.local_train` | Official native `World`/`Body` scene + simplified PD, custom balance reward, SB3 PPO | Independent CPU experiment, **not** an equivalent implementation of the official task/BAM physics |

The native profile uses 61 raw observation values, a single 14-action output,
50 Hz policy steps, and a small bounded action range. It trains only zero-command
balance. Walking, commanded head tracking, recovery and sim-to-real quality are
not established. Its model is quarantined even when a reward increases.
This is off-board policy training, not a new way for cognition to command joints.

## Executed on this Mac — 2026-09-24

Apple Silicon / 48 GiB RAM / Python 3.12.13. The official upstream checkout
remained clean. CUDA was unavailable and was not required by these CPU runs.

| Run | Actual work | Result |
|---|---|---|
| `mac-official-001` | Official BAM task, 4 environments × 5 PPO iterations | Checkpoint, dedicated official export and ONNX compatibility passed |
| `mac-official-002` | Official BAM task, 32 environments × 100 PPO iterations | Training 136.12 s, export 8.73 s, compatibility passed; no deployment |
| `mac-managed-gate-001` | First automatic official physics/parity gate, 4 environments × 5 PPO iterations | 188 parity samples passed; **4/4 trials fell, candidate rejected** |
| `mac-managed-gate-002` | Same bounded run after stricter evidence-binding checks | 188 parity samples passed; **4/4 trials fell, candidate rejected**; source fingerprint matches its historical report |
| `mac-official-1000-001` | Official BAM task, 32 environments × 1000 PPO iterations; paired `play`/`nonplay` and robotd diagnostics | ONNX parity passed; **4/4 fell in both BAM profiles**; candidate failed to enter healthy walking in two robotd scenarios and remains rejected |
| `mac-official-2000-001` | New official BAM task run, 32 environments × 2000 PPO iterations, seed 11; automatic formal physics probe and isolated robotd pair | Checkpoint/export/parity passed; **4/4 held-out trials fell**; candidate entered neither healthy robotd walk scenario and remains rejected |
| `mac-official-resume-2002-001` | Resumed the seed-11 2000-iteration checkpoint for 2 additional PPO iterations in a new job | Official checkpoint loaded; new checkpoint/export/parity passed; **4/4 formal trials fell**, candidate rejected |
| `mac-official-resume-2004-002` | Resumed the preceding job for 2 more PPO iterations | Step counter continued and a new checkpoint/export/parity passed; **4/4 formal trials fell**, candidate rejected |
| `mac-official-resume-3004-003` | Resumed the preceding job for 1000 more PPO iterations, 32 environments, seed 11 | New checkpoint/export/parity passed; **4/4 formal trials fell**; a nonplay diagnostic fell 0/4 but the paired robotd candidate entered neither healthy walk scenario, so rejected |
| `mac-official-push-resume-3500-005` | Resumed for 500 PPO iterations with a training-only 1–2 s push interval | New checkpoint/export/parity passed; **4/4 formal play trials fell**. A paired zero-velocity push diagnostic survived 4/4 short trials, but robotd HOME-start reached no healthy walk; candidate rejected |
| `mac-native-001` | Independent native-PD/SB3 experiment, 8,192 transitions | Weights changed; checkpoint reload and ONNX parity passed; all 8 held-out trials still fell |

The native-PD experiment's mean time to episode termination increased from
0.7900 s to 1.4275 s. This is **not stable standing**. PyTorch/ONNX maximum
action difference was `5.96e-8` on 94 recorded simulator observations. This
numerical result applies to that native model, not automatically to the
official BAM model.

Preserved raw reports (in the unpublished research archive):
official first run,
official 100-iteration run,
first managed quality-gate run,
final-source managed run,
native experiment.
Longer official CPU run and paired diagnostics
record the new checkpoint, raw-report hashes and rejection evidence.
Checkpoint/ONNX binaries and full logs remain in the corresponding local
`runs/rl/` directories (git-ignored, not included in the board release).

### Official checkpoint parity and physical evaluation

The 100-iteration candidate was subsequently evaluated in the actual official
Microduck/BAM CPU environment (`play=True`, `auto_reset=False`, one environment,
four seeds 10001–10004, 250-step maximum). The official checkpoint inference
actor includes its fitted normalizer; no manual normalization was substituted.

- Checkpoint/PyTorch versus ONNX: **235 samples passed**, maximum absolute
  action difference `4.1723e-7` (`rtol=1e-4`, `atol=1e-5`).
- New candidate: **4/4 trials fell**, 235 total control steps.
- Official v5 VelStand reference: **1/4 fell**, 898 total control steps. Initial
  observation/config hashes matched for paired seeds. This is a different-task
  reference, not a quality oracle, and it was not universally stable either.
- Source, BAM parameters, compiled models, model artifacts, command/action
  traces and terminal states are preserved in
  official physics/parity evidence.

**Decision: do not promote this candidate.** A functioning training/export
pipeline and numerical parity do not establish a usable gait. More training,
command/scenario coverage and runtime/bench validation are still required.

The then-current-source managed run repeated the short official CPU workflow with a fresh
checkpoint. The automatic gate checked 188 real simulator observations against
the checkpoint actor, with maximum absolute ONNX action error `1.7881e-7`, then
recorded 4/4 falls (188 total control steps). Its `job.json` status is
`candidate_rejected`, `simulator_gate=failed`, and
`physical_hardware_gate=not_run`. The compact physics evidence
binds the model/script/lockfile hashes; the complete 279 KB trace remains in
the local git-ignored run directory. The four-seed gate can reject an observed
failure, but even zero falls would be `candidate_unqualified`, not approved.

The first attempt of the evaluation script incorrectly compared a tuple of
observation group names to a list and stopped before inference. The predicate
was corrected without changing or retraining the model; the
failed attempt is retained
separately from the successful v2 probe. No failed attempt is hidden as a pass.

The source used by `mac-managed-gate-002` passed **256 tests** in the
dependency-light release environment (2 optional tests skipped). A separate
full suite with pinned MuJoCo/Gymnasium/PyTorch/ONNX dependencies and the
reviewed upstream checkout passed **268 tests with no skips**, including real
native MuJoCo and ONNX backend checks. Ruff lint/format and mypy (28 source
modules) passed. Its package fingerprint
`7343e84b55c05f150fc3cef2426643bdd65bcd3f78fabfeabe3771d7a99918a7`
matches that historical run report, not the current package source.
The native exporter deliberately uses PyTorch's legacy TorchScript exporter
for this pinned simple graph; its two deprecation warnings are retained. It
needs migration/testing before a future PyTorch upgrade, not warning suppression.

That source was also rechecked through the shared skill/robotd path:
pet-service execution and learning gates passed, the 30-day preference ablation
remained `0.11`, and the official v5 kick trace replay passed. The kick's actual
task outcome remains `unknown` (no ball/task-goal evidence), with low-trunk and
joint-range warnings; minimum observed trunk height was about `0.0473 m`.
These warnings are retained, not treated as proof of a stable physical maneuver.

### 1000-iteration official CPU follow-up

The pinned official task's default recipe is 50,000 PPO iterations; this
32-environment, 1000-iteration Mac CPU run is a bounded experiment, not full
training or CUDA-equivalent policy quality. It took 1215 seconds to produce
`model_999.pt`; official export and ONNX contract checks passed. Across 169
real `play=True` observations, checkpoint/ONNX maximum action error was
`9.54e-7`, but all four held-out seeds fell after 29, 50, 42 and 48 steps.
The formal quality verdict is **rejected**.

To diagnose the harder play profile, the same immutable checkpoint and ONNX
were also evaluated with `--evaluation-profile nonplay`. The official task
uses push intervals of 0.5–1 s in play and 3–6 s in nonplay; this profile
comparison is **diagnostic only** and cannot pass the qualification function.
The nonplay run still fell in all four seeds after 29, 47, 42 and 49 steps.
The initial actor observations and compiled model hashes matched per seed;
the first seed's entire 29-step observation stream matched. The observed
failure is therefore not explained merely by the shorter play push interval.
This does not isolate every other cause of instability.

An isolated `robotd --sim` comparison used the same candidate in a temporary
walking slot. The official v5 baseline entered walk in both idle and
forward/stop; the candidate entered neither and reached approximately
0.044 m trunk height. Both pairs shared the HOME pose and ran around real
time. This is corroborating **rejection**, not a gait or hardware approval.
See the hash-bound compact evidence
and the [robotd comparison method](../validation/robotd-candidate.md).

The training reward changed weights at curriculum stages 500 and 750, so
raw reward values across those stages are not directly comparable. Longer
training alone is not a reason to promote this candidate. The official v5
policy remains the simulator/runtime baseline; no new motor weights are
shipped to a board.

### 2000-iteration local CPU follow-up

A fresh seed-11 run completed 32 environments × 2000 iterations in 2446 s,
then produced the official checkpoint export and passed ONNX compatibility.
The checkpoint actor and ONNX agreed on all 339 real `play=True` observations
(maximum absolute action difference `1.073e-6`). Held-out seeds 10001–10004
survived 113, 103, 73 and 50 control steps, but **all four fell**. This is
longer survival than the earlier seed-7 1000-iteration run on the same held-out
seeds, not a stable gait or a causal training-progress claim across seeds.

An isolated paired `robotd --sim` check used identical HOME initial poses.
The unchanged official v5 baseline entered walk in both idle and forward/stop,
applied the forward command and confirmed stop. The new model entered neither
healthy walk; minimum observed trunk height was about 0.035 m in both cases.
The hash-bound summary
points to the raw ignored job, physics report and robotd comparison. Both
simulator paths reject this candidate. Its source fingerprint records the
pre-release-verifier Duck Mind code used for training; the later application
release evidence is regenerated separately. No policy was published or
installed, and physical hardware remains `not_run`.

A follow-up on the **same** checkpoint used the official `play=False`
diagnostic profile with seeds 10001–10004. Checkpoint/ONNX parity passed on
802 observations (maximum absolute action error `8.345e-7`); trial lengths
were 250, 250, 250 and 52 steps, with one fall. The same-seed initial actor
and compiled-model hashes match the formal `play=True` trial. The pinned task
uses 0.5–1.0 s push intervals in play and 3–6 s in nonplay, consistent with
frequent pushes contributing to the formal failures, but push-event causation
was not measured. One nonplay fall and the failed paired `robotd` check keep
the candidate rejected. The diagnostic summary
binds the raw 1.16 MB report hash and does **not** replace the formal gate.

### Trusted local checkpoint continuation

Two bounded `--resume-job` runs added two PPO iterations each to the official
seed-11 checkpoint, without modifying either parent job. The official trainer
logged the staged checkpoint load; its common step counter advanced from
48000 to 48048 to 48096. The resulting `model_2000.pt` and `model_2001.pt`
were new checkpoint bytes, exported to new ONNX files and passed parity on
289 and 235 actual BAM observations respectively. Each failed the formal
`play=True` gate with **4/4 falls**; trial lengths were 78/73/70/68 and
64/61/59/51 steps. Thus the continuation mechanism is verified, but the
additional iterations did not establish a usable gait. The
hash-bound resume summary
points to raw ignored local jobs and logs. No candidate was installed or
published, and no physical test ran. A rejected job can be a training parent,
never an approved runtime policy.

The subsequent 1000-iteration continuation took 1201 s to train and produced
`model_3000.pt` with a restored common step counter of 72096. Its official
checkpoint/ONNX parity passed on 403 `play=True` observations (maximum action
error `7.15e-7`), but all four held-out trials fell after 160, 101, 77 and
65 steps. The same immutable model survived all four 250-step `play=False`
diagnostic trials with 1000/1000 parity samples; this profile is explicitly
non-qualifying. In a separate official `robotd --sim` idle/forward-stop pair,
the v5 baseline reached walk in both scenarios while the candidate reached
neither and descended to about 0.044 m. The
hash-bound 3000-iteration summary
ties together the raw ignored training, diagnostic and robotd reports. The
candidate stays rejected and no motor weight is deployed.

Two further CPU-only ablations used **that same checkpoint and ONNX**, four
held-out seeds and the official `play=True` configuration. Removing the
`push_robot` event entirely yielded 0/4 falls, but changed the initial actor
observations/actions, so it is not a tightly paired causal comparison. A
stronger `--evaluation-profile play-zero-push` retained the event's 0.5–1.0 s
schedule and sampling, changing only its x/y velocity ranges to zero. This
also yielded 0/4 falls and 1000/1000 checkpoint/ONNX parity samples. All four
initial actor observations, first actions, commands and compiled physics
model hashes matched the formal play trials; the first action divergences
were at steps 48, 33, 43 and 39 (0.96, 0.66, 0.86 and 0.78 s at 0.02 s/step).
This is strong **four-seed evidence of push sensitivity**, not a general gait
quality or hardware-safety result. Both ablations are explicitly
`diagnostic_only` and are rejected by the qualification gate; the independent
`robotd --sim` failure is unchanged. The
hash-bound summary
records both raw reports and logs. Future training should target disturbance
recovery under the official push distribution, followed by fresh formal play
and `robotd` checks; do not promote the no-push gait.

## Official CPU environment

Use the reviewed upstream checkout and **its own lockfile**, including BAM's git
revision and dependency overrides. Do not install the whole stack into the
dependency-free robot service environment.

```bash
UV_PROJECT_ENVIRONMENT=/abs/duckmind/.venv-official-cpu \
  uv sync --locked --project /abs/microduck_rl

uv run --locked --python 3.12 python -m duckmind.training run --execute \
  --upstream /abs/microduck_rl \
  --python /abs/duckmind/.venv-official-cpu/bin/python \
  --job /abs/duckmind/runs/rl/mac-official-001 \
  --device cpu --num-envs 4 --max-iterations 5 --seed 7 --timeout-s 900
```

The selected task is `Mjlab-Velocity-Flat-MicroDuck`, not VelStand (whose default
BC experts require W&B downloads). No cloud job, telemetry credentials or model
publication is requested. `job.json` distinguishes checkpoint, export,
compatibility, simulator and physical acceptance gates.

The subsequent one-shot first-walk oracle comparison
used the same ONNX Runtime 1.27.0 bytes as the preceding robotd pair. Both the
official baseline and velocity-only candidate reported fallen at the first
observed `walk` status, with low trunk height on the immediately following
simulator sample, but only official v5 VelStand
recovered. The official [task inventory](https://github.com/pollen-robotics/microduck_rl/blob/cb70b792312d559a4da09064d92009079671815f/README.md)
defines Velocity as command tracking and VelStand as walking plus recovery;
its model inventory also distinguishes the ground-contact geometry. This is
a reason to **stop treating more Velocity-only PPO iterations as the default
next step**, not a proof that the task alone caused failure. A local
recovery-capable experiment needs pinned, trusted walk and stand teacher
checkpoints or an explicitly different from-scratch recipe. The pinned
[VelStand config](https://github.com/pollen-robotics/microduck_rl/blob/cb70b792312d559a4da09064d92009079671815f/src/mjlab_microduck/tasks/microduck_velstand_env_cfg.py)
and [distillation config](https://github.com/pollen-robotics/microduck_rl/blob/cb70b792312d559a4da09064d92009079671815f/src/mjlab_microduck/tasks/distill.py)
default to remote W&B teacher checkpoints unless local checkpoint paths are
supplied. No teacher download or untrusted checkpoint loading was performed.
Any new weight remains quarantined until unchanged formal play, paired robotd
HOME→walk and later supervised physical gates pass.

### Experimental push-schedule continuation

The reviewed official task uses `push_robot` every 3–6 s during training but
every 0.5–1 s in `play=True` evaluation. To test disturbance recovery without
editing the pinned upstream checkout, the Duck Mind job manager accepts an
explicit bounded training-only override:

```bash
uv run --locked --python 3.12 python -m duckmind.training run --execute \
  --upstream /abs/microduck_rl \
  --python /abs/duckmind/.venv-official-cpu/bin/python \
  --job /abs/duckmind/runs/rl/push-continuation-new-id \
  --resume-job /abs/duckmind/runs/rl/trusted-parent \
  --device cpu --num-envs 32 --max-iterations 500 --seed 11 \
  --push-interval-s 1.0 2.0 --timeout-s 1800
```

The interval is recorded in `job.json` and passed through the official mjlab
CLI as one tuple. This profile is experimental, does not modify the formal
`play=True` 0.5–1 s evaluation, and does not weaken ONNX parity or policy
qualification. A short improvement in training reward is not a usable gait;
compare formal held-out falls and the independent `robotd --sim` pair before
considering a candidate. Rejected parents may be resumed for research but
never installed as motor policies.

The first 500-iteration continuation from trusted `model_3000.pt` used this
1–2 s schedule; the official trainer's saved `params/env.yaml` confirms it.
Training advanced the common step counter from 72096 to 84096 and produced
new `model_3499.pt`/ONNX bytes. Checkpoint/ONNX parity passed on 395 formal
observations, but the held-out `play=True` trials still fell **4/4** after
161, 99, 70 and 65 steps (the parent fell after 160, 101, 77 and 65). The
independent `robotd --sim` pair again passed the official v5 baseline but the
candidate reached neither healthy walk scenario. The
hash-bound push-continuation summary
records the job, training log/config, physics probe and robotd report. This
single intervention did not solve the failure; further CPU iterations should
not be treated as a promotion strategy without a new hypothesis and formal
evidence.

The same new checkpoint then ran a paired `play-zero-push` diagnostic. Zeroing
only scheduled push velocities survived all four 250-step trials and passed
1000-sample checkpoint/ONNX parity. The formal and diagnostic runs shared model
bytes, upstream, per-seed initial actor, first action/command and compiled
physics model hashes; actions first diverged at steps 48/33/43/39. This is
short-horizon push-sensitivity evidence, not a usable gait. Separately,
`robotd --sim` selected candidate policy `walk` around 2.14 s after HOME but
still reported `fallen=true` at ~12 s; v5 recovered by ~2.3 s. The
hash-bound summary
and [runtime analysis](../validation/robotd-candidate.md) preserve both
contexts. A startup/handoff mismatch is plausible, not causally proven; it
needs a targeted experiment rather than unqualified extra PPO iterations.

### Repeat the official physics/parity probe with a wall limit

Use a **trusted local checkpoint produced by this workflow**. The official
runner uses pickle-backed checkpoint loading; never substitute an untrusted
download. Adjust paths and choose fresh report/log filenames:

```bash
uv run --locked --python 3.12 python - <<'PY'
from pathlib import Path
from duckmind.training import child_environment, run_phase

root = Path('/abs/duckmind')
upstream = Path('/abs/microduck_rl')
job = root / 'runs/rl/mac-official-002'
command = [
    str(root / '.venv-official-cpu/bin/python'),
    str(root / 'src/duckmind/offline_eval.py'),
    '--upstream', str(upstream),
    '--checkpoint', str(job / 'checkpoints/ACTUAL_TIMESTAMP_duckmind/model_99.pt'),
    '--onnx', str(job / 'artifacts/policy.onnx'),
    '--reference-onnx', '/abs/official-v5/velstand.onnx',
    '--report', str(job / 'physics-new.json'),
]
result = run_phase(command, cwd=root,
                   env=child_environment(upstream, device='cpu'),
                   log=job / 'physics-new.log', timeout_s=180)
print(result)
if result.returncode or result.timed_out:
    raise SystemExit(1)
PY
```

Exit zero means the probe completed with numerical parity, **not** that its
physical trials passed. Read `candidate_summary.falls`, the per-seed traces,
and `paired_initial_observations_match`. The reference argument is optional.
New managed `duckmind.training run` jobs invoke this probe automatically without
the reference model. The manual command remains useful for a paired diagnostic
run or for historical candidates; it never overrides a rejection. An optional
`--evaluation-profile play-zero-push` is an ablation only: it retains the
official play push timer and samples zero x/y velocity. It cannot qualify a
policy, even with zero observed falls.

## Independent native-PD PPO experiment

```bash
uv run --locked --python 3.12 \
  --with stable-baselines3==2.7.1 --with torch==2.9.1 \
  --with gymnasium==1.2.3 --with mujoco==3.10.0 \
  --with onnx==1.20.1 --with onnxruntime==1.27.0 \
  python -m duckmind.local_train --upstream /abs/microduck_rl \
  --job /abs/duckmind/runs/rl/mac-native-001 \
  --timesteps 32768 --seed 7 --threads 2 --timeout-s 900
```

Each run needs a new directory. Total timesteps must be a multiple of 256 so
SB3 does not silently round the budget upward. One environment and bounded CPU
threads are used; the owner process applies a wall timeout and cleans its own
worker process group. There is no resume or deployment option.

Artifacts include initial/trained checkpoints, PPO CSV metrics, ONNX and JSON
evidence. Evaluation uses eight fixed, paired seeds (10001–10008) distinct from
the permitted training seeds. It compares zero-action, initial, trained and
exported policies, recording falls, survival, height, tilt and reward separately.
An increase in reward alone cannot pass the non-regression diagnostic.

The saved checkpoint is reloaded before evaluation/export. The export includes
the exact action clip (the SB3 documentation's basic example omits it). PyTorch
and ONNX actions are compared on actual simulated observations, and ONNX is run
through the native physics evaluation. Raw observations use identity
normalization; there is no hidden running normalizer. Result hashes identify the
exact artifacts and upstream source, not merely a model filename.

Passing these checks is still **not** permission to replace robotd's policy.
Native PD differs from BAM, only short zero-command trials are tested, and the
physical robot is unavailable. Full command coverage, longer/OOD trials,
official-runtime comparison and supervised bench acceptance remain necessary.

## Sources

- [SB3 2.7.1 PPO CPU and rollout contracts](https://stable-baselines3.readthedocs.io/en/v2.7.1/modules/ppo.html)
- [SB3 export and missing action postprocessing warning](https://stable-baselines3.readthedocs.io/en/v2.7.1/guide/export.html)
- [Gymnasium reset/step and termination versus truncation](https://gymnasium.farama.org/api/env/)
- [Pinned official native body](https://github.com/pollen-robotics/microduck_rl/blob/cb70b792312d559a4da09064d92009079671815f/src/mjlab_microduck/sim/body_server.py)
- [Pinned Rust raw observation contract](https://github.com/pollen-robotics/microduck/blob/a9ec4b2079ef8ee7904014089c885bb07d57d63c/duck-control/src/obs.rs)
