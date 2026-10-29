# Reproduce the official first-pet simulator gate

This runs the *same local Duck Mind service* that the board unit starts, but
points it at official `robotd --sim` and the official MuJoCo body. It uses a
synthetic owner invitation/feedback and only fresh `/private/tmp` children;
it never touches `/run/robotd.sock` or a physical robot.

Use the pinned `microduck` and `microduck_rl` commits, v5 policy directory,
arm64 `robotd`, and Python 3.12 dependencies from
[run-live-sim.md](run-live-sim.md). Then run from the Duck Mind repository:

```bash
uv run --python 3.12 \
  --with mujoco==3.10.0 --with numpy --with onnxruntime==1.27.0 \
  python -m duckmind.sim_pet \
  --microduck /path/to/microduck \
  --microduck-rl /path/to/microduck_rl \
  --policies /private/tmp/your-policy-root/current \
  --robotd-bin /path/to/microduck/target/aarch64-apple-darwin/debug/robotd \
  --keyframe HOME \
  --report /private/tmp/duckmind-pet-home.json
```

Repeat with `--keyframe SIT --report /private/tmp/duckmind-pet-sit.json`.
For the negative-feedback path, repeat HOME with `--feedback-score 0` and a
third report name. The default `--feedback-score 1` exercises positive feedback.
The [pinned official body server](https://github.com/pollen-robotics/microduck_rl/blob/cb70b792312d559a4da09064d92009079671815f/src/mjlab_microduck/sim/body_server.py)
defaults to `SIT`, where the folded duck must rise under the standing policy;
`HOME` begins upright. Duck Mind names and records the initial pose in each
report. The release gate requires HOME-positive, SIT-positive, and
HOME-negative on the same official `robotd` binary and policy manifest. The
server has no physics-seed CLI, so these are two starting poses and two
feedback directions, **not** a multi-seed robustness claim.

Each run also writes a sibling `*.body.json` file, for example
`duckmind-pet-sit.body.json`. Only simulator mode accepts the recorder option.
It captures the service's actual `robot.head` intent and robot-observable
frames during one `look_game`, without raw joints or simulator oracle fields.
The harness replays that trace through the fixed Embodied Skill Layer before
marking `body_replay_gate=pass`, and hashes it into the main report. The trace
does not include separate preflight, owner stop, feedback or SQLite operations.

The gate requires: operator enables the official simulator policy; Duck Mind
preflight observes `walk` with no requested/applied motion intent; deterministic
System 1 engages; every gaze sample keeps the same policy/stationary guard;
the bounded head command has measured tracking; synthetic feedback creates
exactly one bounded update; owner stop disables play; service restart retains
one bounded preference update in the correct direction but remains passive.
It records exact upstream commits plus SHA-256 for the robotd binary, ONNX
Runtime library, scene, body-server source, policy manifest and six selected
ONNX files. On failure,
bounded process log tails are retained in the report.

The report also records a read-only `robot.policies` snapshot from the running
official daemon. This simulator's temporary TOML deliberately overrides the
walk path with its isolated policy set, so `origin=local` and
`overridden=true` are expected **in the simulator**. The real-board preflight
instead requires the unoverridden official v5 path, enabled and homed. Neither
report proves ONNX bytes inside a running process or physical head safety; the
file hashes bind the configured model artifacts at report creation.

The recorded HOME-positive,
SIT-positive, and
HOME-negative examples, with their
HOME body trace and
SIT body trace and
negative body trace (kept in the unpublished research archive), have
`execution_gate=pass`, `learning_gate=pass`, and
`physical_hardware_gate=not_run`. The positive `success` and negative
`failure` outcomes describe **synthetic owner feedback**, not verified real
human enjoyment/displeasure or a physical-board
head-safety pass. The earlier `robot.look` failure and corrected fixed v5
delta are documented in [upstream validation](../upstream-validation.md).
