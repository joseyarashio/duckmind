# Official Microduck Validation Baseline

Validated on 2026-09-23 against:

- `pollen-robotics/microduck` commit `a9ec4b2079ef8ee7904014089c885bb07d57d63c`
- `pollen-robotics/microduck_rl` commit `cb70b792312d559a4da09064d92009079671815f`
- host: Darwin arm64, Rust 1.96.0, Python 3.14.6

## Findings

| Question | Result | Evidence level |
|---|---|---|
| Does simulation run the real `robotd` loop and IPC? | Yes. Only `RobotIo` changes to `RemoteIo`. | source + official tests |
| Can real and sim share Duck Mind's upper contract? | Yes. Both expose `robot.*` / `robot.state`. | source confirmed |
| Policy observation/action shape | 61 -> 14; mouth excluded. | source + unit tests |
| Body simulator protocol | TCP newline JSON, protocol 1, 200 ms daemon timeout. | source + local smoke |
| One-shot skill ACK means complete | No. ACK can precede later loop arbitration. | source confirmed |
| Can completion be correlated by request ID? | No direct ID in `robot.state`. Infer from policy transitions + timeout. | source confirmed |
| Is simulator an exact hardware twin? | No. Drivers, BLE, ISP, NPU, encoder/RGA, thermal, and battery are absent/modelled. | official design boundary |
| Full official `scripts/duck-sim` launcher | Not run; it starts extra services and explicitly chooses the old gait. | explicitly pending |
| Minimal official `robotd` + v5 ONNX + `duck-body` end-to-end | Ran on this Mac with arm64 `robotd`, MuJoCo 3.10.0 and the official v5 bundle. | local live trace + replay |
| Does `robot.look` reach its returned joint pose under v5 `velstand`? | No in the pinned live run: absolute IK output enters a policy trained for deltas from HOME; measured head missed by about 0.3 rad. | source + live failure |
| Can a bounded v5 head-delta play action be observed? | Yes in the pinned live run: fixed `robot.head` delta, measured tracking, service outcome, synthetic feedback, restart. | local live report; hardware pending |

## Source-confirmed contracts

### Simulation seam

Official `RobotIo` owns read/write/gain/torque/slow sensors. The `RemoteIo`
backend keeps the control loop, policy, safety, odometry, kinematics, and IPC
unchanged. The body protocol reconnects on the next tick after failure and has
a 200 ms per-call timeout.

Sources:

- [RobotIo seam](https://github.com/pollen-robotics/microduck/blob/a9ec4b2079ef8ee7904014089c885bb07d57d63c/duck-control/src/io.rs#L1-L18)
- [RemoteIo protocol and timeout](https://github.com/pollen-robotics/microduck/blob/a9ec4b2079ef8ee7904014089c885bb07d57d63c/duck-control/src/sim.rs#L1-L84)
- [Official simulation design](https://github.com/pollen-robotics/microduck/blob/a9ec4b2079ef8ee7904014089c885bb07d57d63c/docs/design/simulation.md#L76-L125)
- [MuJoCo body-server counterpart](https://github.com/pollen-robotics/microduck_rl/blob/cb70b792312d559a4da09064d92009079671815f/src/mjlab_microduck/sim/body_server.py#L1-L40)

### State and command contract

Continuous `robot.move` and `robot.head` are last-writer-wins notifications;
discrete calls return `IntentResult`. `robot.state` publishes requested vs
applied movement, limit reasons, active policy, safety, loop health, joints,
targets, odometry, IMU, and optional velocity/load.

Sources:

- [robot intent vocabulary](https://github.com/pollen-robotics/microduck/blob/a9ec4b2079ef8ee7904014089c885bb07d57d63c/duck-ipc-proto/src/lib.rs#L580-L648)
- [IntentResult and RobotState](https://github.com/pollen-robotics/microduck/blob/a9ec4b2079ef8ee7904014089c885bb07d57d63c/duck-ipc-proto/src/lib.rs#L3659-L3765)

### Policy and skill runtime

The runtime assembles one fixed 61-D observation and receives 14 policy actions.
Skill priority is active skill, ground-pick, sit/rise, stand, then walk. The
active skill name is published as the driving policy, including `:unwind`.

Sources:

- [61-D observation layout](https://github.com/pollen-robotics/microduck/blob/a9ec4b2079ef8ee7904014089c885bb07d57d63c/duck-control/src/obs.rs#L1-L69)
- [driving policy name](https://github.com/pollen-robotics/microduck/blob/a9ec4b2079ef8ee7904014089c885bb07d57d63c/robotd/src/control.rs#L250-L265)
- [skill expiry and arbitration](https://github.com/pollen-robotics/microduck/blob/a9ec4b2079ef8ee7904014089c885bb07d57d63c/robotd/src/control.rs#L420-L503)

### ACK is not completion

`robot.do` is queued first. The 50 Hz loop later consumes pending skills and
may still refuse one because another scripted move or ground-pick is active.
The JSON-RPC reply therefore proves admission to the queue, not physical start
or successful outcome.

Sources:

- [robot.do queues an intent](https://github.com/pollen-robotics/microduck/blob/a9ec4b2079ef8ee7904014089c885bb07d57d63c/robotd/src/main.rs#L3747-L3784)
- [later control-loop arbitration](https://github.com/pollen-robotics/microduck/blob/a9ec4b2079ef8ee7904014089c885bb07d57d63c/robotd/src/main.rs#L2170-L2210)

## Locally executed verification

### Official Rust suites

Command:

```bash
cargo test -p duck-control -p duck-ipc-proto -p robotd
```

Result: **316 passed, 0 failed, 9 ignored**. The ignored cases require ONNX
Runtime >= 1.23 and cover recurrent-policy execution. Passing suites include
simulation protocol mismatch/reconnect/refusal behavior, 61-D observation
offsets, safety/deadman handling, state serialization, skill queuing, and
robotd integration gates.

### MuJoCo body protocol smoke

A minimal environment ran the official `duck-body` module headlessly and
exercised `hello`, `read`, `torque`, `gain`, `write`, and `slow` against the
real MuJoCo scene. Observed:

```text
protocol          1
joint_count       15
sim_advanced_s    0.28
neck_motion_rad   0.08469
trunk_z           0.11453
```

This proves the body server and protocol execute on the current Mac. It does
not by itself prove `robotd` + ONNX policy + MuJoCo end to end. The separate
live scenario below does.

### Live official control chain

The [headless integration runner](../src/duckmind/sim_live.py) launched the
official `duck-body`, official `robotd` built for arm64, v5 `velstand` plus
`ball_kick_left`, and ONNX Runtime 1.27.0. It issued a semantic
`OneShotSkillIntent`, then observed `accepted -> started -> completed` from
`robot.state`. A separate read-only simulator oracle measured joint motion and
trunk height. The recorded JSONL trace
is validated offline by `python -m duckmind.replay` and adversarial tests.

This is an **execution gate pass**, not a claim that a ball was kicked. The
scene has no ball/contact target, so task outcome is `unknown`. The latest
replayable trace passed the lifecycle gate but recorded both
`joint_range_limit_observed` and `low_trunk_height_observed` (minimum about
0.048 m). Other local runs differed, including one above the 0.06 m warning
threshold. Fresh runs must be evaluated individually; execution completion
and a passing replay do not clear physical stability.
See [reproduction](validation/run-live-sim.md).

The convenience `scripts/duck-sim` currently names `alpha_walking.onnx` and
`alpha_stand.onnx`; current v5 defaults instead use `velstand.onnx` and no
standing network. We chose the latter explicitly and pinned the manifest in
the trace; these two simulation configurations must not be conflated.

### Head command mismatch discovered by the pet-service gate

The official `robot.look` reply describes absolute IK joint angles, then
`robotd` passes those numbers into `command.head` unchanged. The pinned
`velstand` training environment defines its head command as deltas from the
HOME joint pose, and `duck-control` copies `command.head` straight into the
policy observation. In the failing live check, `robot.look` returned roughly
`neck=0.35, pitch=0.50`, while the measured head settled around
`neck=0.50, pitch=0.91`; the minimum error stayed above 0.3 rad. This is not
an admissible completion even though robotd accepted the command.

The first play slice therefore uses one fixed, small `robot.head` delta
(`head_pitch=+0.15 rad`, other deltas zero), not arbitrary `robot.look` targets.
The adapter checks two measured head samples within 0.12 rad of HOME+delta.
The pet-service live report
passed execution, synthetic-feedback learning, stop and passive restart. Its
`robotd` hello reported API 37 and daemon version 0.15.0 but **no build
revision**; API/policy names alone cannot prove the exact board binary. Real
motion is therefore locked behind explicit supervised approval in addition
to the online version/profile check.

The original pet run used `HOME`, an already-upright placement. The pinned
[official body server](https://github.com/pollen-robotics/microduck_rl/blob/cb70b792312d559a4da09064d92009079671815f/src/mjlab_microduck/sim/body_server.py)
defaults to `SIT`, a folded pose from which the standing policy rises. The
SIT pet report now
checks that path too. Both reports must name the same upstream checkout,
`robotd` binary and policy manifest. This adds a startup-state check, not
randomized physics or proof of real motor safety; the official body server
exposes no physics-seed CLI in this pinned version.

Pinned sources: [look reply contract](https://github.com/pollen-robotics/microduck/blob/a9ec4b2079ef8ee7904014089c885bb07d57d63c/duck-ipc-proto/src/lib.rs#L2133-L2160),
[robotd look dispatch](https://github.com/pollen-robotics/microduck/blob/a9ec4b2079ef8ee7904014089c885bb07d57d63c/robotd/src/main.rs#L4314-L4335),
[observation copy](https://github.com/pollen-robotics/microduck/blob/a9ec4b2079ef8ee7904014089c885bb07d57d63c/duck-control/src/obs.rs#L205-L225),
[v5 training head-delta contract](https://github.com/pollen-robotics/microduck_rl/blob/cb70b792312d559a4da09064d92009079671815f/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py#L653-L674).

## Architecture adjustments caused by limits

1. Add a Duck Mind `SkillTracker`; never map ACK directly to outcome.
2. Infer start/exit from `robot.state.policy`, with explicit start/completion timeouts and one local active skill.
3. Keep success evaluation separate from lifecycle completion.
4. Treat missing velocity/load as unknown, not zero.
5. Keep simulator ground truth (`trunk_z`, room position, sim time) in a
   validation-only oracle; real cognition cannot observe it.
6. Require real-hardware gates for drivers, radio, ISP/NPU/video path, thermal,
   battery behavior, and servo-bus faults.
7. Use local monotonic receipt time for timeouts; `robot.state.t` is daemon-relative.
8. Keep an execution gate separate from task outcome. An empty-floor kick
   scene cannot prove ball contact or social success.
9. Treat `robot.look` as an unverified first-board command on this v5 pair;
   never train from its ACK. A fixed, measured head-delta profile is the
   temporary compatible slice, guarded by a real-hardware approval.
