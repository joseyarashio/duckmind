# Pre-Implementation Architecture and Source Review

Reviewed before the live adapter/harness implementation, 2026-09-23 UTC.
Pinned source: `microduck` `a9ec4b2079ef8ee7904014089c885bb07d57d63c`,
`microduck_rl` `cb70b792312d559a4da09064d92009079671815f`, official policy set `v5`.

## Decision: the simulation seam stays below `robotd`

| Official component | Reviewed contract | Duck Mind decision |
|---|---|---|
| `robotd` | Unix JSON-RPC, `robot.state`, 50 Hz control loop | one `RobotdBodyPort` for simulator and hardware |
| `duck_control::RobotIo` | real Dynamixel or `RemoteIo` | no simulator-only motor API above the seam |
| `duck-body` | TCP protocol 1, MuJoCo body with 15 joints | validation-only oracle on a separate connection |
| policy runtime | `obs[1,61] -> actions[1,14]` | Duck Mind never assembles observations or actions |
| `robot.do` | ACK admits request to a queue; loop can later refuse | separate ACK, start, execution exit, task outcome |
| `robot.state` | process-relative `t`, no request ID | local receipt clock for timeouts; one local active skill |

Sources: [RobotIo](https://github.com/pollen-robotics/microduck/blob/a9ec4b2079ef8ee7904014089c885bb07d57d63c/duck-control/src/io.rs),
[robotd simulation](https://github.com/pollen-robotics/microduck/blob/a9ec4b2079ef8ee7904014089c885bb07d57d63c/robotd/src/main.rs#L233-L261),
[IPC state](https://github.com/pollen-robotics/microduck/blob/a9ec4b2079ef8ee7904014089c885bb07d57d63c/duck-ipc-proto/src/lib.rs#L3700-L3740),
[MuJoCo body](https://github.com/pollen-robotics/microduck_rl/blob/cb70b792312d559a4da09064d92009079671815f/src/mjlab_microduck/sim/body_server.py).

## Corrections made during source review

1. **Policy version drift in the convenience script.** The v5 manifest contains
   `velstand.onnx` and the current `robotd-params` defaults to it with no
   standing network. `scripts/duck-sim` still explicitly selects
   `alpha_walking.onnx` plus `alpha_stand.onnx`. Both are present in v5, but
   that script tests a different gait. The Duck Mind harness explicitly loads
   `velstand.onnx` and `stand = "none"`, and pins the manifest hash in the trace.
   See [v5 manifest](https://huggingface.co/pollen-robotics/microduck-policies/blob/v5/manifest.json),
   [current default](https://github.com/pollen-robotics/microduck/blob/a9ec4b2079ef8ee7904014089c885bb07d57d63c/robotd-params/src/lib.rs#L1570-L1591),
   [script config](https://github.com/pollen-robotics/microduck/blob/a9ec4b2079ef8ee7904014089c885bb07d57d63c/scripts/duck-sim#L230-L263).
2. **Host architecture must match ONNX Runtime.** The installed Rust host
   toolchain was x86_64 under Rosetta while Python/ONNX Runtime was arm64.
   The first run correctly reported policy-unavailable. We installed the
   `aarch64-apple-darwin` Rust target and rebuilt `robotd` for arm64; the
   original failed trace was not accepted as a validation result.
3. **Daemon time is not client time.** `robot.state.t` restarts at zero with
   `robotd`. The adapter keeps it as `robot_time_s` for correlation but uses
   local monotonic receipt time for skill timeouts.
4. **No global request correlation.** Serializing Duck Mind's own one-shot
   executions reduces ambiguity, but another robotd client could still launch
   a same-named skill. A real deployment needs an owner/lease or a stronger
   upstream execution ID before claiming unambiguous multi-client attribution.
5. **A kick movement is not a proven kick outcome.** The empty floor has no ball
   or contact target. The live gate may pass for command-to-body execution,
   while the task `Outcome.verdict` remains `unknown`. The trace retains low
   trunk height and joint-range clipping as per-run warnings, not silently as
   success. The latest regenerated trace has both warnings; an earlier run
   had neither. That variation remains a stability-validation concern.

## Accepted first-slice boundary

The live harness covers one named episodic skill (`kick_left`) from semantic
intent through the Embodied Skill Layer, official IPC, ONNX policy, MuJoCo,
state stream, oracle metrics, versioned episode trace, and offline replay.
It does **not** test the full `scripts/duck-sim` multiprocess launcher, walking
navigation, sit-toggle posture phases, cognition/LLM, Mem, camera, ToF, BLE,
real motor bus, battery, or thermal behaviour. Those remain separate gates.

Safety authority remains in `robotd`; cognition gets neither the operator's
`robot.enable` capability nor simulator-only joint/trunk truth.
