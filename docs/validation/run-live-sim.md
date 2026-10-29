# Reproduce the Official Live Simulator Gate

This uses official `microduck` and `microduck_rl` components, but deliberately
starts only `duck-body` and `robotd`. It does **not** run the broader official
`scripts/duck-sim` launcher (which currently selects an older gait).

## Inputs to pin

- `microduck`: `a9ec4b2079ef8ee7904014089c885bb07d57d63c`
- `microduck_rl`: `cb70b792312d559a4da09064d92009079671815f`
- `microduck-policies`: official `v5` manifest and all ONNX files
- Python 3.12; MuJoCo 3.10.0; ONNX Runtime 1.27.0
- On the validated Apple Silicon Mac, arm64 `robotd` (the installed Rust host
  toolchain is x86_64 under Rosetta; an x86 binary cannot load an arm64 ORT).

The trace records exact upstream commits and SHA-256 for the policy manifest,
six configured ONNX policy files, `robotd` binary, ONNX Runtime library,
scene and body-server source.
These file hashes do not attest bytes already loaded into a running process.
Keep the full policy directory
beside its `.source` record; do not put model binaries in Duck Mind Git.

## Prepare once

Check out the two upstream repositories at the commits above. On Apple Silicon:

```bash
cd /path/to/microduck
rustup target add aarch64-apple-darwin
cargo build -p robotd --target aarch64-apple-darwin
```

Create a **new, empty, short** policy directory under `/private/tmp` with
`mktemp -d /private/tmp/duckmind-policies.XXXXXX`. Use the printed directory
as the explicit argument to the official `scripts/seed-policies.sh`; inspect
the resulting `current/manifest.json`, `.source`, and ten `.onnx` files.
Do not point that script at an existing production policy root: it manages
its own release symlink and staging contents.

## Run

From this Duck Mind repository, replace the three `/path/to/...` values and
the policy directory with your actual checked-out paths:

```bash
uv run --python 3.12 \
  --with mujoco==3.10.0 --with numpy --with onnxruntime==1.27.0 \
  python -m duckmind.sim_live \
  --microduck /path/to/microduck \
  --microduck-rl /path/to/microduck_rl \
  --policies /private/tmp/your-new-policy-root/current \
  --robotd-bin /path/to/microduck/target/aarch64-apple-darwin/debug/robotd \
  --trace /private/tmp/duckmind-kick-trace.jsonl
```

The runner starts child processes in a fresh short temporary directory,
shuts down **only** those child handles, then writes the JSONL trace. It never
touches `/run/robotd.sock` or an already-running real robot. For a native
Linux build, use that platform's `robotd` binary and matching ONNX Runtime.

Replay without MuJoCo:

```bash
uv run --python 3.12 python -m duckmind.replay /private/tmp/duckmind-kick-trace.jsonl
uv run --python 3.12 python -m duckmind.body_replay /private/tmp/duckmind-kick-trace.jsonl
```

The second command additionally verifies that the semantic named skill emits
the same `robot.do` intent and consumes the recorded observations in order
through the real Embodied Skill Layer. It does not re-run MuJoCo or replay the
separate first-pet head-game service.

## What passing means

`execution_gate=pass` requires an accepted request, observed named policy
start and return to walk/stand, no safety interruption, and nontrivial joint
motion measured by the simulator oracle. The `Outcome.verdict` is still
`unknown`: the floor has no ball/contact goal, and motor motion alone cannot
prove a useful kick. Runs can warn about low trunk and `joint_range`
clipping. Real-time MuJoCo variation means a single passing execution does
not prove gait stability; inspect each
fresh trace before training a new skill or calling this gait healthy.

The recorded example trace (kept in the unpublished research archive) is a
regression fixture, not a hardware result. Tests corrupt policy transitions,
event IDs, oracle metrics and verdicts to ensure replay rejects them.
