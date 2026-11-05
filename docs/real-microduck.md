# Connecting a real Microduck (the remote body)

[繁體中文](real-microduck.zh-Hant.md)

Duck Mind's brain is meant to live on your home computer while the body is the little robot duck.
The robot's own daemon, `robotd`, stays on the duck; a small **bridge** on the duck lets one
brain on your home network reach it. This page says what is ready, how to connect a duck, and what
still has to be checked on a real robot.

> **Status (2026-11-04).** The link, the bridge and the safety stop are built and rehearsed with a
> *simulated* duck on a second process behind a network shim (delay, jitter, loss, cuts). No real
> Microduck has been connected yet. Every step below marked **verify** is a guess from the
> official software until someone runs it on a robot.

## How it fits together

```text
home computer                                   the duck (Radxa Zero 3)
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
   `duckctl ip` for its address and `robotctl health` on the duck. Follow the official dev-board
   guide for the board itself.
2. **Install Duck Mind on the duck** as in [docs/deploy/board.md](deploy/board.md) (a wheel in a
   venv; Python ≥ 3.12). The bridge needs only the standard library and Duck Mind itself.
3. **Make the shared token** on the computer and copy it to the duck (keep both copies mode 600):

   ```sh
   python -m duckmind.remotebody token ~/.duckmind/remote-body.token
   duckctl scp ~/.duckmind/remote-body.token :.duckmind/remote-body.token
   ```

   (`duckctl scp` finds the duck itself; a leading `:` is a path on the duck, in the home of
   `--user`, else `DUCK_BOARD_USER`, else `radxa`. Make the folder first with `duckctl ssh`.)

4. **Start the bridge on the duck**, as a user in the `robot` group (that is what opens
   `/run/robotd.sock`):

   ```sh
   python -m duckmind.remotebody bridge --listen 0.0.0.0:8797 \
       --token-file ~/.duckmind/remote-body.token --backend real \
       --robotd /run/robotd.sock --tofd /run/tofd/tof.sock --media /run/mediad/media.sock \
       --events ~/remote-body-events.jsonl
   ```

5. **Check the link from the computer** — it reads robotd's health and policies and moves nothing:

   ```sh
   python -m duckmind.remotebody check remote://<duck-address>:8797
   ```

   It prints the channels, the deadman, robotd's health (battery) and the link's round trip and
   longest silence.
6. **Rehearse without the robot** at any time: `python -m duckmind.remotebody sim-body` runs a
   simulated duck behind the bridge, and `python -m duckmind.simweb --live --body
   remote://127.0.0.1:8797` drives it (add `python -m duckmind.remotebody shim` in between to add
   delay, loss and cuts: type `cut` / `restore`).

The simulator page itself drives **simulated** remote bodies only for now: its senses (camera,
hearing, touch, battery) and its 3D view still come from the simulated world. A real duck needs
those senses from the robot first (see the checklist). The page's 我的小鴨 (my duck) tab has a
*The real duck* card: it says which body the brain drives now (the simulated one on this computer,
or the remote one with its round trip, drops and longest silence) and lists these steps.

## Checklist for the first real duck (not verified yet)

- [ ] **Socket and permissions:** `/run/robotd.sock` exists, mode `0660 root:robot`; the bridge's
  user is in `robot`; `hello` answers API 37.
- [ ] **Policies:** `robot.subscribe` reports `walk: velstand.onnx`; record the policy hashes as in
  `docs/deploy/board.md`.
- [ ] **`robot.stop` on hardware** zeroes the twist without going limp (upstream: "stop is not
  limp").
- [ ] **`sit_toggle` on hardware** from walk and stand: time to seated, on the table surface you
  use; never sent while seated or fallen.
- [ ] **robotd's own 500 ms deadman** on hardware (stop streaming `robot.move` mid-walk).
- [ ] **Pull the plug test:** walk, then switch the home Wi-Fi off (or walk the computer out of
  range): time from the cut to stopped and to seated, from the bridge's events file and a video.
- [ ] **Home Wi-Fi quality:** `check` round trip, longest silence and drops over 10 minutes;
  false alarms per hour at the 1.0 s deadman.
- [ ] **Reconnect:** Wi-Fi back → the link returns and the brain takes the duck back.
- [ ] **Head profile:** the v5 head deltas (`robot.head`) on the real head (the board service
  sends them only with `--approve-real-v5-head-profile`).
- [ ] **Battery:** `robot.health.battery` reaches the brain (today the brain's battery is the
  simulated world's).
- [ ] **Camera over the link:** `media.frame` through the bridge — a 720p UYVY frame is 1.8 MB;
  measure frames per second on Wi-Fi, or ask for a smaller quality.
- [ ] **Microphone, touch:** no path from the robot yet (today: the simulated world's models).
- [ ] **The page's 3D view** from `robot.state` joints instead of the simulator's truth.
- [ ] **Other doors on the duck:** the official `mediad` listens on `0.0.0.0:8443`/`8080` with no
  authentication and lets a LAN peer drive the robot; Bluetooth pairing (`btd`) is off by default.
  Decide whether to firewall them while the bridge is in use.
- [ ] **CPU on the Radxa Zero 3:** the bridge's load at 50 Hz state plus ToF.
- [ ] **Clock:** the brain uses robotd's own `t_ns` (no clock sync needed); `check` reports the
  round trip, the link reports a clock offset.
