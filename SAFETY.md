# Safety

**duckmind has not been validated on real hardware.** Everything it has been tested
with is the simulator. The board release is a candidate, not a go decision for a
real robot (see [docs/ready-to-burn.md](docs/ready-to-burn.md)).

If you run it on a real robot anyway, you do so at your own risk. Keep a person next
to the robot, keep it on a clear floor away from edges, stairs, water, children and
pets, and keep the robot's own physical way to cut power within reach.

## `robot.stop` is not an emergency stop

`duckmindctl stop` (and every stop duckmind sends) calls robotd's `robot.stop`.
That **zeros the walking command and latches play off. It does not cut servo power
or torque**, and it does not instantly freeze a head movement that was already sent.

Your emergency stop is the robot's own power switch, its battery, or the physical
method the robot maker gives. duckmind cannot replace it.

When duckmind is killed hard (SIGKILL, a crash, power loss) it cannot send anything.
Then only robotd's own deadman (the walking command expires after 500 ms without a
refresh) and the physical stop are left.

## What duckmind does to stay safe

These layers are there to reduce risk. None of them is a certified safety function.

- **duckmind never turns the robot on.** Its body link refuses to send
  `robot.enable`, `robot.torque`, `robot.policy`, `robot.gains`, `robot.servo`,
  `robot.calibrate` and `robot.reboot_motors`. A person enables the robot's walking
  policy through the robot's own controls. (Only the simulator and commissioning
  tools use a separate operator client that can enable a simulated robot.)
- **Play starts locked.** After every boot the body is marked lost until checks
  pass. On a real board, play stays locked until a supervised bench trial of the
  head profile has been recorded and an operator installs an approval tied to that
  exact release.
- **Checks before and during play.** Before play, and every 2 seconds during it,
  duckmind checks the connection to robotd, the robotd API version, that the walk
  policy is the expected one (by sha256), the battery (at least 30 %), and that the
  robot is not fallen or limp. Any failure stops play and calls `robot.stop`.
- **A stop always gets through.** Stops use their own connection to robotd, so a
  busy command stream cannot delay them. A zero move and a stop are always allowed.
- **Shutting down stops.** When the service is stopped normally, it cancels what it
  is doing, sends a stop, and the system service runs a separate stop step after
  it exits.
- **Head movements stay in range.** Head commands are small changes from a fixed
  home pose, and only for the robotd version and walk policy they were checked with.

## Policies and models

duckmind uses Pollen Robotics' v5 policy set and can train new skills. A policy that
behaves well in the simulator can still fall, hit things or overheat servos on a real
robot. New or retrained policies must go through the gates in
[docs/training/official-rl.md](docs/training/official-rl.md) and a supervised bench
trial before they come near a real robot.

## Reporting a safety problem

If you find a way around these layers, please report it privately as described in
[SECURITY.md](SECURITY.md).
