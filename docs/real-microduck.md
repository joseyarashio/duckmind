# Connecting a real Microduck (the remote body)

[繁體中文](real-microduck.zh-Hant.md)

Duck Mind's brain is meant to live on your home computer while the body is the little robot duck.
The robot's own daemon, `robotd`, stays on the duck; a small **bridge** on the duck lets one
brain on your home network reach it. This page says what is ready, how to connect a duck, and what
still has to be checked on a real robot.

> **Status (2026-11-08).** The link, the bridge and the safety stop are built and rehearsed with a
> *simulated* duck on a second process behind a network shim (delay, jitter, loss, cuts). No real
> Microduck has been connected yet. Every step below marked **verify** is a guess from the
> official software until someone runs it on a robot. The robots that ship run Pollen's
> production ("beta") main board with the V1.0 face board and the factory Yocto image: robotd
> 0.18.x (API 49), policy set v8 (the next release moves to v9, which is v7's walk network), a
> two-layer `robotd.toml` and an upright GC2093 camera. The simulator runs robotd 0.18.5 + v8.

## How it fits together

```text
home computer                                   the duck (Pollen's board)
─────────────                                   ───────────────────────
brain (simulator page / neural duck)            robotd   /run/robotd.sock
  │ robotd.sock, tof.sock, media.sock (local)   tofd     /run/tofd/tof.sock
  ▼                                             mediad   /run/mediad/media.sock
RemoteLink ══ one TCP connection on the LAN ══► bridge ──► the unix sockets above
            token handshake, signed frames,       │
            ping every 100 ms                     └─ deadman: no brain for 1.0 s →
                                                     robot.stop, then sit
```

- The official `robotd` speaks JSON-RPC over a **unix socket only** (no TCP, no authentication;
  the socket's `0660 root:robot` mode is the whole gate). The bridge does not change robotd: it
  forwards a brain's streams to robotd's socket, as `ssh -L` would, plus authentication and a
  deadman.
- On the computer, `RemoteLink` offers the duck's daemons as local sockets in a private folder, so
  the brain's existing robotd clients connect to them as they connect to a simulator's robotd.

## Safety design

| | |
|---|---|
| Where it listens | loopback by default; you give the duck's LAN address (or `0.0.0.0`). Both ends refuse addresses outside loopback, private and link-local ranges. |
| Who may drive | the one brain that proves it holds the shared token (HMAC challenge both ways; the token never crosses the network). A new brain with the token replaces the old one. |
| Tampering | every frame after the handshake carries a sequence number and a signature (HMAC-SHA256 with a per-session key); a bad or out-of-order frame ends the link and stops the duck. The link is **not encrypted** — LAN only. |
| What a brain may send | robotd: `hello`, `robot.subscribe`, `move`, `head`, `mouth`, `pose`, `look`, `do`, `stop`, `health`, `model`, `modelApi`, `policies`, `safeToRestart`, and the operator's `enable`, `init`, `poseJoints`. Everything else (`robot.relax`, `loadPolicy`, `shutdown`, `setMode`, `rebootMotors`, `system.*`, `net.*`, …) is refused with an error and never reaches robotd. |
| Link lost | the brain pings every 100 ms. After **1.0 s** without a frame (or a broken connection, or a bad frame) the bridge closes every stream — nothing the brain sent earlier can arrive late — sends `robot.stop`, then `robot.do sit_toggle` while the duck stands (retried until it sits; never while it sits or lies fallen). robotd's own deadman has already zeroed a walk 0.5 s after its last `robot.move`. |
| Coming back | the brain dials again (0.1 → 1 s backoff) and takes the duck back where it is (seated). |

Why 1.0 s and not less: TCP over Wi-Fi stalls now and then (a lost packet waits a 200 ms
retransmission, three in a row 1.4 s). In the rehearsal a 0.6 s deadman tripped on ordinary 1 %
loss; every false alarm sits the duck down.

## Rehearsal results (simulated duck, one Mac, two processes)

The same brain-side body code drove a local body and a remote one behind the shim (50 ± 15 ms each
way, 1 % loss as TCP retransmissions). Each run: 20 command round trips, stand, walk 3 s, turn,
kick, sit. Produced by `tools/measure_remote_body.py`.

| body | link, each way | command round trip p50 / p95 | walk | turn | kick | falls | link drops |
|---|---|---|---|---|---|---|---|
| local | – | 19 / 38 ms | 10/10 | 10/10 | 10/10 | 0 | – |
| remote | 20 ± 5 ms, 0.2 % loss | 65 / 93 ms | 10/10 | 10/10 | 10/10 | 0 | 0 |
| remote | 50 ± 15 ms, 1 % loss | 131 / 285 ms | 10/10 | 9/10 | 10/10 | 0 | 1 |
| remote | 80 ± 20 ms, 3 % loss | 207 / 408 ms | 10/10 | 8/10 | 9/10 | 0 | 4 |

*Command round trip:* `robot.head` sent by the brain → robotd's state frame showing it, back at
the brain. *Link drop:* the shim held the brain's packets over 1.0 s (three TCP retransmissions in a
row); each time the duck stopped and sat and the brain took it back. Every failed turn or kick was
in a run with a drop.

**Link cut mid-walk** (16 cuts: 10 at 50 ms, 3 at 20 ms, 3 at 80 ms; the shim holds everything, as
a Wi-Fi drop does):

| from the cut to … | median | worst |
|---|---|---|
| the duck stopped (true speed < 3 cm/s for 0.3 s; robotd's own 0.5 s deadman) | 0.94 s | 1.04 s |
| the bridge declares the link lost and sends `robot.stop` | 1.00 s | 1.02 s |
| the bridge sends sit | 1.05 s | 1.13 s |
| the duck seated (true trunk height below 8.5 cm) | 1.81 s | 2.00 s |
| the brain notices | 1.17 s | 1.25 s |

After the link was restored it was back in 0.06–0.21 s and the brain's body had robotd's state
again in 0.22–0.95 s; the duck walked again after all 16 cuts; no falls. Holding the link seated
for 10 minutes at 1 % loss (and 5 minutes each at 0.2 % and 3 %) dropped it 0 times.

**The neural duck over the link** (the simulator page with `--brain neuro --body remote://…`, the
page's remote control walking at 0.3 m/s, 10 cuts of 4 s, 50 ± 15 ms / 1 % loss): it walked before
every cut (0.23–0.50 m in 5 s, 10/10); the bridge stopped the duck 0.97 s (worst 1.01 s) after
each cut and robotd's policy was sitting 1.16 s (worst 1.22 s) after; the brain was ticking again
1.58 s (worst 2.17 s) after the link came back, and the duck walked again every time (0.33–0.64 m
in 6 s, 10/10). The simulated body must match the brain's duck: a body with a ToF module under a
brain whose duck has none (or the reverse) behaves differently (`sim-body --tof on|off`).

## Connecting a duck — the steps known today

1. **Set the duck up with Pollen's own tools** first: Wi-Fi (`duckctl wifi connect` from your
   computer, or `sudo robotctl net connect <ssid> --psk <passphrase>` on the duck), then
   `duckctl ip` for its address and `robotctl health` on the duck. A shipped duck comes with
   Pollen's image already flashed; a hand-built Radxa Zero 3 follows the official dev-board guide.
2. **Install Duck Mind on the duck** as in [docs/deploy/board.md](deploy/board.md) (a wheel in a
   venv; Python ≥ 3.12). The bridge needs only the standard library and Duck Mind itself.
3. **Make the shared token** on the computer and copy it to the duck (keep both copies mode 600):

   ```sh
   python -m duckmind.remotebody token ~/.duckmind/remote-body.token
   duckctl scp ~/.duckmind/remote-body.token :.duckmind/remote-body.token
   ```

   (`duckctl scp` finds the duck itself; a leading `:` is a path on the duck, in the home of
   `--user`, else `DUCK_BOARD_USER`, else `microduck`. Make the folder first with `duckctl ssh`.)

4. **Start the bridge on the duck**, as a user in the `robot` group (that is what opens
   `/run/robotd.sock`):

   ```sh
   python -m duckmind.remotebody bridge --listen 0.0.0.0:8797 \
       --token-file ~/.duckmind/remote-body.token --backend real \
       --robotd /run/robotd.sock --tofd /run/tofd/tof.sock --media /run/mediad/media.sock \
       --events ~/remote-body-events.jsonl
   ```

   On a real duck the bridge also turns **Pollen's anonymous usage data** off once when it starts
   (the owner's choice; it is on by default since daemon 0.18.3). `updaterd` holds that switch
   and answers `telemetry.set {"enabled": false}` on `/run/updaterd.sock` (`0660 root:robot`, so
   the bridge's user may call it); off survives reboots and updates and deletes the random robot
   id. The bridge logs a `telemetry` event with what it found. `--keep-pollen-telemetry` leaves
   the switch alone. The simulator runs no `updaterd`, so nothing is sent from the computer.

5. **Check the link from the computer** — it reads robotd's health and policies and moves nothing:

   ```sh
   python -m duckmind.remotebody check remote://<duck-address>:8797
   ```

   It prints the channels, the deadman, robotd's health and the battery as the brain gets it (see
   below), the link's round trip and longest silence, the **clock offset** (the duck's wall clock
   minus the computer's, ± half a round trip: add it to the computer's times to line them up with
   the bridge's events file and a video), the **camera's frame rate over the link** (3 s of
   `media.frame` back to back; `--camera-seconds`), and whether **mediad's LAN doors** (8443,
   8080) answer from the computer. From a real duck's bridge it also prints the walk and sit-stand networks the
   duck runs (their sha256 and set version) and whether the brain was measured on them in the
   simulator (v5 and v8 are; an unmeasured walk is driven with v5's numbers), the board and what
   its `robotd.toml` (`/etc/robot/robotd.toml` with the owner's `/var/lib/robot/robotd.toml`
   over it) turns on, what to change on the duck, and which camera lens duck-eye needs.
   Turn robotd's own idle look-around off: `sudo robotctl configure`, `[idle]` `enabled`
   off, then restart robotd (it takes the head and a yawn opens the beak).
   It also prints Pollen's usage data as the bridge left it: `usage data (Pollen telemetry): off
   (enabled False, active False; …)`. `ON` or `UNKNOWN` means the duck still sends, or the bridge
   could not ask `updaterd`: switch it off in the console's drawer (*share anonymous usage
   data*), or put `MICRODUCK_TELEMETRY=0` in `updaterd`'s environment.

   **Home Wi-Fi quality:** `check remote://<duck-address>:8797 --wifi-minutes 10` then holds the
   link ten minutes and prints each minute and the total: round trip p50 / p95 / max, the longest
   silence *at the duck* (the gap between two pings arriving at the bridge, read off the duck's
   own clock: exactly what its deadman judges), drops, and **false deadman alarms per hour** (every
   drop while the computer kept pinging sat the duck down for nothing) with the near misses
   (silences over half the deadman).
6. **Rehearse without the robot** at any time: `python -m duckmind.remotebody sim-body` runs a
   simulated duck behind the bridge, and `python -m duckmind.simweb --live --body
   remote://127.0.0.1:8797` drives it (add `python -m duckmind.remotebody shim` in between to add
   delay, loss and cuts: type `cut` / `restore`).

### Opening the page on a real duck

With the bridge running on the duck (step 4) and `check` content (step 5):

```sh
python -m duckmind.simweb --live --brain neuro --life ~/.duckmind/lives/<your-duck> --no-nest \
    --microduck <microduck checkout> --microduck-rl <microduck_rl checkout> \
    --policies <policy set> --robotd-bin <robotd built for this computer> \
    --body remote://<duck-address>:8797 --body-token-file ~/.duckmind/remote-body.token
```

The four paths are the same ones a simulated body on this computer uses (the `python` needs
`onnxruntime` installed, as there). The page reads the robot's model and the policy set from
them; with `--body remote://…` it starts no robotd and no physics on this computer. Without them
it stops at once with `--live needs --microduck, --microduck-rl, --policies and --robotd-bin`.
`--no-nest` matters for a duck that hatches now (a new `--life` folder): the neural brain does
not take a duck with a charging nest, and the page would stop while starting. Verified on
2026-11-09 with exactly this command (a new life, `--port 8786`) against
`sim-body --as-real`: the page came up in real-duck mode (*rehearsal*), with the duck's camera
and robotd's battery.

The page sees that the bridge's backend is real and switches to **real-duck mode**. The brain
drives the duck with only what the duck itself measures and sends over the link. Nothing comes
from a simulated world: there is none.

| | In real-duck mode |
|---|---|
| 3D view | The duck's own 15 joints from `robot.state`, placed by robotd's contact odometry (x, y and height) and tilted by its IMU, turned to the odometry's heading. It stands on an empty floor grid: no rooms, no furniture, no minimap. Odometry drifts, so the place on the grid is where the duck *thinks* it is. The view's label says "the duck's own joints and odometry", never physics truth. |
| Eye window | The duck's camera through the link (mediad's `media.frame` on the bridge's `media` channel). If the bridge offers no camera, the window says so. |
| Status | Ticker chips for *real duck* (or *rehearsal*), the link (round trip, drops, longest silence; up or down) and the battery from robotd's `robot.health` (a placeholder until robotd answers; charging is unknown). The *real duck* card in 我的小鴨 shows the same. |
| Works | The remote control (legs), the command bar (words, praise, treat, no), the motions and tricks the duck's robotd runs, emotions, the time of day, the talk panel. The brain's own senses that come from robotd work too: posture, falls, being lifted (robotd's `safety.picked_up`), the ToF when the bridge offers `tofd`. |
| Refused, and the page says why | Everything the simulated world did: the play stimuli (sounds, toys, people, lights), the world's hand (help up, carry to the mat), family visits and battery swaps, the lab's world buttons (place, roll, tidy, show), the demo walk, *upright* and *reset the body*, applying a layout to the home, and pointing at the floor for go-there. Each such button is greyed out with the reason 「這是真的小鴨：沒有模擬的世界，這件事做不到」 (in the page's language) and a note under it; the server refuses the same requests with the same words. |
| Not from the robot yet | The microphone and touch (below). |

**Rehearse it without hardware.** `python -m duckmind.remotebody sim-body --as-real …` runs the
simulated duck behind its bridge but presents it as a real duck: the bridge says `backend: real`
and offers no world channel (only robotd, tofd and the camera). The page then runs in real-duck
mode on this computer and its chip says *rehearsal*. The simulated world still runs on the body's
side (it makes the camera's pictures and the ToF), but the brain and the page never see it.

The **battery** comes from the body on every remote body: the brain reads robotd's `robot.health` battery over the link
every 2 s (`LiveBody.battery()`, `duckmind/remotebody/battery.py`), never the simulated world's
pack. robotd reports volts (the servos' supply, smoothed) and its own percentage of them (6.6 V is
0 %, 8.2 V under load is 100 %), so it sags while the duck walks and recovers at rest; it has **no
charging flag** (Pollen #470), so on a real duck "charging" is unknown and the brain sees it as
not charging. The simulated body behind the bridge (`sim-body`) puts the world's pack into its
robotd's `robot.health` (the loaded volts, robotd's percentage of them, and `charging`), so the
rehearsal goes through the same path. On a *simulated* remote body (plain `sim-body`) the page
still draws and senses through that body's world, as it does for a body on this computer. The
page's 我的小鴨 (my duck) tab has a *The real duck* card: it says which body the brain drives now
(the simulated one on this computer, a rehearsal, or a real duck, with the link's round trip,
drops and longest silence) and lists these steps.

### Microphone and touch on the production duck

**No daemon serves the microphone to a client** in robotd 0.18.5 (checked in Pollen's source,
`microduck` at #526):

- `mediad` is camera, WebRTC and the remote gateway; the microphone is only in its description
  (`mediad/src/lib.rs:1-18` lists what exists; the pipeline is video only). The audio track in
  `docs/design/architecture.md:454` is a design, not code.
- The only reader of the microphone is **robotd's own mic worker**: an `arecord` child at 16 kHz,
  mono, 16-bit (`pet-detect/src/worker.rs:366-374`) feeding the petting classifier and an
  ambient-sound watcher. The capture device is single-client (`worker.rs:5`). On the production
  board it is ALSA's `default`, which the image routes to the face board's microphone
  (`robotd-params/src/lib.rs:1449-1458`). The worker runs only with `[audio] pet_detect = true`
  (off by default, `lib.rs:1426-1433`).
- What it hears stays inside robotd: petting makes the duck coo (`robotd/src/main.rs:2795-2809`);
  ambient-sound events are only logged at debug, "no consumer until the autonomous brain arrives"
  (`main.rs:2812-2816`). No IPC method or notification carries either
  (`duck-ipc-proto/src/lib.rs` has none).

What would carry it: the bridge could run its own `arecord` on the duck (16 kHz mono 16-bit is
256 kbit/s, small next to the camera) while robotd's mic worker is off, and serve it as a new link
channel to the brain's hearing; or Pollen exposes the mic (or its petting and sound events) over
IPC. Neither exists today, so the brain's hearing on a remote body is still the simulated world's.

**Touch:** the production duck has **no touch sensor** (no touch or capacitive part anywhere in
Pollen's source). What it can sense that a person handles it: (1) being picked up — robotd's
classifier, `safety.picked_up` in `robot.state`, which the brain already reads; (2) a head
scratch — the petting classifier on the microphone above ("practically a contact mic for head
scratches", `pet-detect/src/worker.rs:52`), heard only inside robotd; (3) a push or a held leg —
the joints' own positions, targets and currents in `robot.state`, which reach the brain over the
link today but which the brain does not yet read as touch. Carrying (2) needs the same new path as
the microphone.

## Checklist for the first real duck (not verified yet)

- [ ] **Socket and permissions:** `/run/robotd.sock` exists, mode `0660 root:robot`; the bridge's
  user is in `robot`; `hello` answers (API 49 on daemon 0.18.x; the brain asks with 37 and robotd
  refuses no client version).
- [ ] **Policies:** `robot.subscribe` reports `walk: velstand.onnx`; `check` prints the walk and
  sit-stand sha256 and set version (record them as in `docs/deploy/board.md`) and whether the
  brain was measured on them (v5, v8: yes; v9 = v7's walk: no).
- [ ] **robotd.toml:** `check` prints the board and the merged `[idle]` / `[pickup]` /
  `limp_fall`; `[idle]` turned off in the owner layer (`sudo robotctl configure`). With
  `[pickup]` on, lift the duck: robotd pauses (`safety.picked_up`) and the brain says *lifted*
  at once (robotd does not pause a seated duck: the brain's own sense finds that lift). Watch one
  get-up after a fall: in the simulator robotd's classifier paused get-up tries on the side.
- [ ] **Usage data off:** `check` prints `usage data (Pollen telemetry): off` (`enabled False,
  active False`); restart the duck and check again (it stays off).
- [ ] **`robot.stop` on hardware** zeroes the twist without going limp (upstream: "stop is not
  limp").
- [ ] **`sit_toggle` on hardware** from walk and stand: time to seated, on the table surface you
  use; never sent while seated or fallen.
- [ ] **robotd's own 500 ms deadman** on hardware (stop streaming `robot.move` mid-walk).
- [ ] **Pull the plug test:** walk, then switch the home Wi-Fi off (or walk the computer out of
  range): time from the cut to stopped and to seated, from the bridge's events file and a video
  (`check` prints the duck's clock offset to line the events file up with the computer's clock).
- [ ] **Home Wi-Fi quality:** `check --wifi-minutes 10`: round trip, longest silence at the duck,
  drops and false alarms per hour at the 1.0 s deadman.
- [ ] **Reconnect:** Wi-Fi back → the link returns and the brain takes the duck back.
- [ ] **Head profile:** the head deltas from HOME (`robot.head`) on the real head (the board
  service sends them only with `--approve-real-v5-head-profile`; the flag keeps its v5 name).
- [ ] **Battery:** the code is ready (robotd's `robot.health` battery reaches the brain over the
  link; rehearsed with the sim-body); on the duck: `check` prints `battery (what the brain gets)`,
  and watch the percent sag while it walks.
- [ ] **Camera over the link:** `media.frame` through the bridge — a 720p UYVY frame is 1.8 MB;
  `check` prints the frames per second it got over the link (3 s sample; the sim-body behind the
  shim at 20 ± 5 ms and 0.2 % loss gave 5.2 frames/s of 720p, 77 Mbit/s), or ask for a smaller
  quality. The production camera is
  the GC2093, upright, about 85° × 55°: start duck-eye with `--board beta` (or `face-board-v1`).
- [ ] **Microphone, touch:** no daemon serves the microphone and the duck has no touch sensor (see
  *Microphone and touch* above); today the brain's hearing and touch are the simulated world's.
- [ ] **The page in real-duck mode:** the 3D view follows the real joints, odometry and tilt (the
  code is ready and rehearsed with `sim-body --as-real`); check on the robot that the drawn pose
  matches the duck (seated, standing, walking, lifted), that the eye window shows its camera, and
  how far the odometry drifts in a few minutes of walking.
- [ ] **Other doors on the duck:** the official `mediad` listens on `0.0.0.0:8443`/`8080` with no
  authentication and lets a LAN peer drive the robot; Bluetooth pairing (`btd`) is off by default.
  `check` says whether 8443/8080 answer from the computer; `docs/deploy/board.md` §5 has the
  nftables table that closes them while the bridge is in use (in memory only; nothing is changed
  on the duck by default).
- [ ] **CPU on the duck's board:** measured on the computer (Apple M5 Pro, % of one core;
  `tools/measure_bridge_cpu.py`: the bridge as its own process, fakes sending robotd 0.18.5-shaped
  `robot.state` lines of 5.4 kB, tofd's 8×8 frames and mediad's 720p UYVY frames): the link alone
  0.4 %, + `robot.state` at 50 Hz 0.6 %, + ToF at 15 Hz 0.6 %, + the camera at 2 frames/s 1.2 %, at
  5 frames/s 2.2 %; flat out it relayed 205 frames/s (3 Gbit/s) at 60 %, about 3 ms of CPU per
  1.8 MB frame (0.6 ms of it the frame's signature). **Estimate** for the board's RK3566 (four
  Cortex-A55 at 1.8 GHz), not measured: 15–30× slower per core for this Python work (published
  single-core benchmark ratios of an in-order A55 against an Apple performance core): state + ToF
  about 10–20 % of one core (3–5 % of the board), each camera frame per second another 5–9 % of a
  core. The camera's real limit is the Wi-Fi: raw 720p UYVY is 29.5 Mbit/s at 2 frames/s and
  74 Mbit/s at 5. On the duck: run `tools/measure_bridge_cpu.py` there (it needs only Duck Mind),
  and watch the bridge in `top` while the brain drives.
- [ ] **Clock:** the brain uses robotd's own `t_ns` (no clock sync needed); `check` prints the
  duck's wall-clock offset from the computer (± half a round trip).
