# First-board deployment and supervised acceptance

This is a **candidate procedure**, not evidence that any real duck has passed.
Duck Mind is a separate unprivileged service; the official Microduck image,
`robotd`, policies, updater, firmware and physical stop path remain authoritative.
The author's real robot has not arrived yet. The hardware gate
is pending delivery; installation and simulator evidence do not clear the
robot for unsupervised play.

## 1. Confirm the actual board first

The reference is the official [Radxa Zero 3 / Armbian 26.2.1 Minimal dev-board
guide](https://github.com/pollen-robotics/microduck/blob/a9ec4b2079ef8ee7904014089c885bb07d57d63c/docs/robot/install-dev.md).
Do not assume your board matches this reference. Record:

- board/OS/architecture, Python `>=3.12`, actual motor/head/servo revision,
  head travel stops, battery rail, physical kill method, and the Python-linked
  SQLite version/patch status for WAL safety;
- official `robotctl health` result, real battery percentage, `robotd` hello
  API/version/revision, `robot.subscribe` walk policy and policy manifest;
- whether a gamepad/operator can safely start and stop the official policy;
- actual robotd binary and v5 policy hashes in the bench report.

The service checks API 37 and the loaded `robot.policies` report: enabled,
homed, unoverridden official walk at
`/opt/robot/policies/current/velstand.onnx`, with no slot/change error. It
also requires that path's current ONNX file to match the reviewed v5 SHA-256
`1c659be55da94bc5753b707de5c6a3e7c49931e05ca3b6991615cef1a8ba9a45`.
It rejects absent real battery, unhealthy robotd, fall/limp and commanded
motion. A same-basename override or disk-byte drift is refused. The digest is
bound to the local v5 model probe,
not provided by the upstream manifest. These checks are necessary but **not
sufficient**: the locally built official `robotd` reported `revision: null`,
and the online report plus disk hash do not attest bytes already loaded in
robotd memory, its binary, servo firmware or mechanics. Record and review
their identities separately. The upstream seeder accepts newer official sets;
if this board has one, do not bypass the guard—review that set and cut a new
Duck Mind release before play.

## 2. Build and transfer a wheel

On the development Mac in this repository, generate one source-matched
candidate rather than using a loose wheel:

```bash
uv run --locked --python 3.12 --extra dev python -m duckmind.release prepare \
  --repo . --release-id 0.1.0-rc29 --output dist/duckmind-0.1.0-rc29
uv run --locked --python 3.12 --extra dev python -m duckmind.release verify \
  dist/duckmind-0.1.0-rc29
shasum -a 256 dist/duckmind-0.1.0-rc29/manifest.json
```

Use the [offline release bundle](release.md), not an ad hoc wheel from a
different checkout. Transfer the whole verified bundle by a trusted channel,
compare its `manifest.json` hash with the separately retained value, then run
`sha256sum -c SHA256SUMS` from inside the transferred bundle on Linux. The
bundle is unsigned, so the checksum file alone is not authentication. After
official Microduck provisioning, install from that bundle directory (substitute
the `wheel.sha256` in its manifest and its exact `release_id`):

```bash
sudo sh deploy/install-board.sh dist/duckmind-0.1.0-py3-none-any.whl <WHEEL_SHA256> <RELEASE_ID>
sudo -u duckmind /opt/duckmind/current/venv/bin/duckmindctl status
```

The installer checks the wheel hash before and after copying, Python version
and the official `robot` group. Before creating accounts or staging a release,
it runs the read-only boot guard on an existing installation and refuses a
split application/unit pair. An installed unit without a current release is
also refused; inspect and repair such a state under operator control before
retrying. It writes a root-owned `RELEASE_ID` record in
the selected release for real-play approval binding. It creates a dedicated
`duckmind` user and a root-owned, traversable release under
`/opt/duckmind/releases/`, archives the release's matching `duckmind.service`
and executable boot guard, and checks both CLI entry points as
the service user before stopping the current service. It then preserves the
previous unit and records whether the service was active and enabled. After
stopping Duck Mind, an upgrade makes a SQLite backup under
`/var/lib/duckmind/backups/` before publishing the new unit and switching the
application symlink. The health gate requires an active service, a successful
local status command, SQLite integrity `ok`, and `play_enabled: false`.

On a handled install failure, recovery restores both the previous application
pointer and unit and restores the previous enabled/active state. It retains
release and recovery files for diagnosis; it never automatically restores or
deletes the state database. If stopping the service or restoring its
configuration fails, the installer reports incomplete recovery; inspect the
unit and journal before starting play. No official Microduck file is
overwritten. If a future schema migration makes the old binary unable to read
new state, stop and recover from the backup under operator control—do not
silently replace newer state.

First boot is passive and real play is locked. The unit uses no TCP listener
or internet and stores state in `/var/lib/duckmind` mode 0700; the command
socket is `/run/duckmind/duckmind.sock` mode 0600. The service user is in
the official `robot` supplementary group only to reach `/run/robotd.sock`.
Before `duckmindd` starts, a read-only guard checks that `current` points to a
local release, its `RELEASE_ID` matches, and the installed unit is byte-for-byte
the archived unit of that release, and that the selected boot guard is
executable. A mismatch blocks startup and subsequent install/rollback rather
than running a split application/unit pair. The guard runs as the unprivileged
service user; [systemd's `ExecStartPre` contract](https://github.com/systemd/systemd/blob/main/man/systemd.service.xml)
stops `ExecStart` when an unprefixed pre-start command fails.

The install/rollback transaction tests use isolated filesystem paths and
command doubles. They cover failures, passive health checks, database backup,
and configuration recovery, but do **not** validate Linux systemd execution,
its sandbox, or the board's permissions. Isolated SIGKILL injection verifies
that a cut between unit and pointer publication fails the boot guard; it is
not an actual power-cut test. Those checks remain part of the
first-board acceptance after delivery.

The daemon's local SIGTERM/SIGINT handler cancels its server task, latches play
off and asks robotd to stop before exit; a real child-process signal test
verifies this path on the development Mac. The unit's `ExecStopPost` performs
another stop request if the daemon exits unexpectedly. Neither is a torque-off
emergency stop or a substitute for a board-level signal/physical kill test.

## 3. Supervised head-profile bench trial

The pinned official v5 `robot.look` path did **not** track its returned
absolute head pose in MuJoCo; the walk policy expects head deltas from HOME.
The first game uses a fixed `+0.15 rad` head-pitch delta, verified against
measured `robot.state` head joints in simulation. Do not treat the simulator
as proof that a physical head mechanism has the same sign or limits.

With the robot physically supported, clear of people/objects, battery healthy,
official operator present and a physical power/torque kill path in reach:

1. Verify the official policy stands normally and that `duckmindctl status`
   reports `play_enabled: false`.
2. For the *controlled trial only*, install
   `deploy/duckmind-play-approved.conf` as
   `/etc/systemd/system/duckmind.service.d/play-approved.conf`, then run
   `systemctl daemon-reload && systemctl restart duckmind.service`. Before
   installation, replace `REPLACE_WITH_CURRENT_RELEASE_ID` in the drop-in
   with the exact `RELEASE_ID` from the selected release. Verify
   `duckmindctl status` reports `real_head_profile_approval: approved`.
   This reports the release-scoped operator flag, **not** a fresh loaded-policy
   attestation. It only permits a later local `enable`; it does not move the
   robot.
3. Start the official policy with its official human control. Confirm
   `robotctl health`, actual battery, and that the head is not already at a
   mechanical limit. Verify `robot.policies` reports the loaded official walk
   slot without override/error and that it has homed. Run
   `sudo -u duckmind sha256sum /opt/robot/policies/current/velstand.onnx`
   and compare with the pinned digest above; Duck Mind will refuse real enable
   if either report or file differs. Use
   `sudo -u duckmind /opt/duckmind/current/venv/bin/duckmindctl enable`.
4. In one terminal run `duckmindctl invite --actor alex --activity look_game`
   through `sudo -u duckmind`; keep another terminal ready for `duckmindctl stop`.
   Observe direction, clearance, noise/load, measured tracking, fall/limp and
   stop response. **Any abnormal motion: use the physical stop, not Python.**
5. Only after a genuinely completed and safe game, supply explicit owner
   feedback: `duckmindctl feedback <interaction_id> 1` (or `0`). Check one
   bounded `learning_update_id`, not a personality or motor-policy change.
6. Restart Duck Mind and confirm it is passive, the preference survived, and
   no old invitation resumes. Test app rollback using an installed previous
   version, then verify the DB still passes integrity and play is passive.
7. Remove the approval drop-in if any criterion fails; retain it only after
   the signed/dated board acceptance report is reviewed.

The approval drop-in and any other systemd drop-ins are operator-managed.
[systemd's official unit documentation](https://github.com/systemd/systemd/blob/main/man/systemd.unit.xml)
defines drop-in precedence, and its
[service documentation](https://github.com/systemd/systemd/blob/main/man/systemd.service.xml)
defines the empty `ExecStart=` reset used by this override.
Install and rollback preserve them, but the service rereads the selected
release's root-owned `RELEASE_ID` on every real-body preflight and watchdog
cycle. A stale drop-in ID therefore leaves play locked after upgrade/rollback;
an old drop-in with only the former boolean flag also stays locked. Review
all drop-ins with `systemctl cat duckmind.service` and repeat supervised
acceptance before changing the approved ID. Release binding does **not** detect
all independent robotd, policy, firmware or mechanical changes. The live
policy-slot check catches a changed path/override or loss of homing, but cannot
prove identical model or daemon bytes. Revoke approval and repeat the bench
trial after any such change.

`duckmindctl stop` asks `robotd` to zero locomotion and latches play off. It
does **not** cut servo torque or instantly freeze an already-issued head
motion; the official/physical emergency method is still mandatory. Do not
create a low-battery test by deliberately draining a pack or induce a hard
fall just to exercise software gates.

## 4. Rollback and evidence

```bash
sudo sh deploy/rollback-board.sh <PREVIOUS_RELEASE_ID>
sudo -u duckmind /opt/duckmind/current/venv/bin/duckmindctl status
journalctl -u duckmind.service -b --no-pager
```

Rollback stops Duck Mind, selects the previous release's application pointer
and archived matching unit, reloads systemd, starts the selected service, and
requires the passive health gate. It refuses a release without an archived
unit, executable boot guard **or matching `RELEASE_ID` record**, and refuses a
symlinked release directory. Legacy releases without those files require a
manual, supervised recovery; do not force an old application into
the approved unit. Rollback leaves the SQLite DB and all official daemon/policy files untouched.
A failed rollback attempts to restore the application and unit that were
selected before the attempt; it restarts that service only if it had been
active. Rollback does not change the service's enabled state. As with install,
an incomplete recovery is an error requiring inspection, not a deploy pass.

The symlink replacement is atomic, but the combined unit/pointer/service
transaction is not a power-loss transaction. SIGTERM and ordinary command
failures invoke recovery; SIGKILL or a power cut cannot run the shell trap.
The boot guard refuses to start a mismatched pair on reboot. If both unit and
pointer were published before interruption, the guard may allow a **passive**
start, but that is not proof the interrupted installer finished its health
gate. There is currently no automatic recovery journal that resumes on reboot.
The installer and rollback command now refuse to proceed from a split pair;
they do not reconstruct the previous configuration automatically. After such
an interruption, keep play disabled and inspect the current
pointer, installed unit and drop-ins, retained recovery files, database
integrity, and service journal before choosing a release and restarting.
Database durability and deployment recovery are separate acceptance items.

Keep the wheel hash, `uv.lock`, simulator reports, board health, observer
notes and result of each physical gate in a versioned acceptance record.
The hardware is awaiting delivery; no real-board installation, rollback,
power-loss, or motion acceptance result has been recorded.
