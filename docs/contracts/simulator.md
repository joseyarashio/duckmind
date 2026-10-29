# Simulator Adapter Interface

## Production-facing path

Duck Mind uses `BodyPort` for both backends:

```text
BodyPort -> robotd Unix JSON-RPC -> RobotIo
                                  |- DynamixelIo -> robot
                                  `- RemoteIo -> duck-body -> MuJoCo
```

There is intentionally no `SimulatedBodyPort` with richer commands. Doing so
would let behavior code diverge before deployment.

## Validation-only path

`SimulatorOracle` may expose truth unavailable on hardware, such as trunk
height, absolute room position, contacts, and simulation time. It is available
only to test evaluators, never cognition, Self, affect, or learning inputs.

`FaultInjector` is likewise a harness control plane for disconnects, latency,
noise, sensor dropout, or perturbation. It cannot be present in production
dependency injection.

## Replay adapter

`RecordedBodyPort` implements the same `BodyPort` using an exact ordered script
of `dispatch_intent`, `observe`, and `stop` steps. It rejects a changed method,
parameter, call order, non-monotonic frame time, unconsumed step, or reused port
after a mismatch. Replay frames allow only the production-observable `odom`
metadata; simulator oracle values stay with the evaluator. The release gate
also replays the recorded official named-skill trace through the actual
`EmbodiedSkillLayer`, in addition to re-deriving its wire/oracle verdict.

The simulator-only pet service now records the first `look_game` gaze's actual
`robot.head` BodyPort call and measured observations. Its paired HOME/SIT
reports hash the trace, and the release gate replays it through the same fixed
head-game skill. This is **not a recorder for every service call**: preflight,
operator enable, the independent safety-stop socket, owner feedback and
storage transitions are checked separately by the pet report and service
tests, not represented as BodyPort replay steps. The named-skill trace contains
one `robot.do` dispatch and state frames. A future full-system replay manifest
should pin:

- Duck snapshot and event-log hash;
- cognition/policy/evaluator versions;
- random seed and clock schedule;
- upstream Microduck commit and simulator protocol version.

## Known official limitations

- The twin models actuator response, IMU, ToF, RSSI, camera, and physics but is
  not proof of Dynamixel, BLE, ISP/3A, NPU, encoder/RGA, battery, or thermals.
- `robot.do` has no end-to-end request correlation ID.
- Simulator protocol 1 is synchronous request/answer and uses a 200 ms daemon timeout.
- `duck-body` real-time stepping is required because robotd health expects the 50 Hz loop.
- `microduck_rl` pins Python >=3.12,<3.13 for its full environment.
