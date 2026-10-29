# Run the Duck Core simulator page

A local web page that runs the real `duckmind.duckcore` in Python and only renders
what it decides: growth days, memory (L1 events → distilled facts → diary), the voice
lineage, and the head/mouth clips drawn on the official Microduck model. Nothing on
the page re-implements Duck Core logic, and nothing here talks to `robotd` or a robot.

```bash
uv run --python 3.12 python -m duckmind.simweb --port 8765 --seed 7 --name Mochi \
  --microduck-mjcf /path/to/microduck_rl/src/mjlab_microduck/robot/microduck
# open http://127.0.0.1:8765/
```

Options: `--seed`, `--name`, `--host` (default `127.0.0.1`), and
`--microduck-mjcf DIR` (or `DUCKMIND_MICRODUCK_MJCF`) naming the official folder
with `robot_walk.xml`, `scene.xml` and `assets/`. The meshes are CC BY-NC-SA and
are served from that checkout, never copied into this repository. Without it the
page falls back to a 2D duck.

A banner over the duck always says what it is doing and why, in plain words, with
one-click ways forward that are ordinary recorded Duck Core inputs: time paused
(「▶ 讓時間流動」; when real time would land in the night on your clock the page says
so first and offers 「☀ 從早上 7 點開始」), switched off (「開機並跳到早上 7 點」,
「換電池開機」, 「讓家人幫忙換電池，開機」, 「插電開機」), asleep or awake at night
(「跳到早上 7 點（看牠起床走走）」, 「叫醒牠一下」), on the desk, on its cable, a
hatchling, fallen (「扶起來」), quiet, locked out or held (「好了，可以動了」),
overheated (「降溫」), too tired to stand (「讓牠睡飽再起來走走」), hatched without a pickup
sensor or motion validation (「模擬：…」), and free to walk: then it says what the duck
is doing and, while it sits, within how many minutes it will get up again
(`DuckCore.outing_eta`), with 「叫牠現在去走走」 (the owner phone's 「去走走」, a drawn
reflex that every hard gate still holds). When a gate opens this way (a hand lets go,
set down on the floor, unplugged, cooled, woken, a fresh battery, 07:00) the duck gets
up to look about at once — a drawn, logged decision. Only the ▶/⏸ button turns the
server's real time on or off; 「對齊我的時鐘」 says beforehand how long it will
fast-forward and whether that lands in the night. The banner's buttons stay put while
real time refreshes the words every second (they are rebuilt only when the list of
actions changes), and in real time the words change when the 3D duck shows the move,
2.5 s after it happened, never before.

If the server process is older than the page (it was started before the code
changed), the page says so in red and asks for a restart instead of misbehaving: the
server reports `api` and whether its code changed on disk (`code_changed`).

If the server stops answering (stopped, restarting, the computer slept), the page says
so within a few seconds instead of showing the last status as if it were now: every
answer counts as a sign of life and a cheap `GET /api/ping` asks when there has been
none for a moment; two misses (or one after 4 s of silence) turn the banner into
「連不上模擬器伺服器」 (since when, and what it last said, marked as the last word, with
「馬上重新連線」), the time line magenta, the 3D clock stopped with 「⚠ 連不上伺服器」, the
physics line and readout saying so, and the page's own clocks stopped. The server names
its process (`boot`); a different one answering means it was restarted, and the page
follows it afresh (its lists and 3D queue cleared) and says whether the duck came back
from its save or started over. Error messages are in Chinese (never 「Failed to fetch」).

Leaving a fast-forward (paused, or back to real time) applies whatever the 3D view still
had queued from the skipped minutes at once (walks to where they ended, postures as
they are) and clears the words about them, instead of playing them on as if they were
happening now; a fast-forward step that answers after the switch is applied, not
played. ▶ and ⏸ say on the 3D view what they did (and that a switched-off duck will not
move until it is switched on); pressing ▶ in real time shows 「● 真實時間：開始中…」
until the server confirms it, never a fast-forward in between. Every reply is also a
toast, readable 5–15 s by its length, so a move's words replacing the caption never
hide it; the TV button says what was pressed (開電視／關電視), not what it became.

Real time runs Duck Core second by second at the wall clock's pace: every awake
minute has a beat at a random second (its per-minute life: battery, mood, the
ordinary decision), so nothing is decided ahead of time and a pickup, a hand or
switching off always comes before whatever follows it. The page (polling every
second) and the live physics body show each event 2.5 s after it happened, at the
same moment; something someone does plays at once, after (or, when it interrupts,
instead of) anything still waiting. Back-to-back seconds are stored as one input, and
time passes the same whatever the step (tested down to one-second steps).

A battery lasts about an hour of simulated time. By default the simulated family
swaps in a fresh pack when it drops below 30% (「家人會幫牠換電池」, a setting and a
new-duck option); on its cable the pack charges about 1% a minute. Asleep at night it
stirs every 5–15 minutes (a daytime doze every 1½–4 minutes) and wakes at 07:00; a
greeting or praise reaches a sleeping duck as a peek (a daytime nap may wake up). In
real time the clip player adds a glance in every 6-second slot of the clock (never more
than 8 s apart), each turning or tipping the head by at least 8° and holding that gaze
about three seconds until the next one takes over; none plays while a move the duck
really did is showing (drawn from the seed and the slot, never part of the event
chain). A seated floor duck's wish to get up grows with every minute it sits (the
first minute's chance, once more for each minute since), so it is up again within a
few minutes — about 4–8 at most, a hatchling or a tired duck a little later. A rested
duck on the floor walks about 40–60 short legs an hour (a few in a row per outing); a
hatchling stands up to look about twenty-odd times an hour.

## Reach it over the tailnet

Keep the server on loopback and let Tailscale terminate TLS for tailnet devices only:

```bash
tailscale serve --bg --https=8765 http://127.0.0.1:8765
# then open the https address that tailscale serve prints
tailscale serve --https=8765 off   # remove only this mapping
```

The page has no login; do not use `tailscale funnel` or bind `--host 0.0.0.0`.
The server raises its listen backlog because the proxy opens every mesh request
at once (the stdlib default of 5 turned that burst into 502s), gzips meshes
(~22 MB of STL → ~7 MB), and skips the parts upstream's own bake hides inside the
shells.

## A duck with a nest (Duck Core v4, phase P0)

New ducks hatched on the page (and a fresh server with no saved duck) have a charging
nest (「有充電窩」 in the new-duck dialog; owner decision 2026-09-26: simulator only, the
real robot's `HardwareConfig.nest` stays off until the nest hardware exists). A saved
duck from before comes back exactly as it was, without one. With a nest the duck lives
the whole day by itself:

- it wakes by itself at 07:00–07:04 in its nest (with verified time), and tells from
  the sounds of the house whether the family is up, asleep or out (only counts, logged
  as `env.cue` at most every five minutes; every "how long since" counts from those
  logged moments, so the log alone decides what it believed);
- it goes out on small trips of a few legs (each still ≤ 6 s at the validated walking
  commands) toward places it has not seen for a while or toward the family, within the
  roam radius (1.5 m by default; 1.5 m from the nest while the family sleeps, quietly);
- it keeps the battery for the way home (`nest.threshold`: the straight way, the walked
  arc round toward home, whole-second legs, the leg in and a forward loop after a miss,
  ×1.5), finds its way back, walks in and sits down the way it came onto the ring
  contacts (`nest.docked`; the open nest, owner decision D1: it never turns round on the
  spot, and leaves by standing up and walking a right arc forward); a miss means a
  forward loop and another try (never backing out), three misses mean sitting beside the
  nest and asking for a hand (`duck.bid` with `need`, and an `owner.notice`), and it
  tries again by itself a few minutes later; no path of any kind — not even the arc
  round toward home — leads outside the family's roam radius (1.5 m while the family
  sleeps), so an outing's leg ends where it can still turn home inside it; an outing
  heads only for places it may walk to (never into, or straight through, a spot it
  learned to avoid), and when its leash and those spots leave it nowhere to go it stays
  where it is instead of turning on the spot;
- while everyone is out it rests in the nest (45–90 minute sleeps, a few minutes awake
  between) instead of waiting awake; once it has learned when the family usually comes
  home it wakes a little before and watches the door; at 21:30 (or after 「晚安」 from
  21:00) it walks home and says goodnight; it never walks between 22:00 and 07:00;
- a fall near its own nest never teaches it to fear the nest (the family is asked to
  check the floor instead); lifted up, it waits to be put back (「放回窩裡」 or
  「放到地上」, which sets it down in front of the nest).

The banner names these states too (在窩裡充電、正在走回窩、家裡沒人在休息、需要幫忙：
抱牠回窩、剛被抱起來、準備睡覺…) with one-click ways forward (「家人回來了」、
「跟牠說話」、「抱牠回窩」…), and the line under the 3D view says where it is relative to
its nest and what the family is doing as far as it can tell. The 「生活快轉」 list has
whole-day families (雙薪、有學齡孩子、每週在家上班幾天、長輩整天在家、單身工時長):
a household-v2 family (`duckmind.duckcore.routines.HouseholdV2`) that gets up, goes
out, comes back, talks, touches, plays, answers the duck's calls and praises what it
likes, as one recorded input. The owner phone group has 「出門」「到家」「晚安」
「放回窩裡」. The old evening routines work with a nest duck too: the family puts it back
in its nest (never switches it off or plugs it in), where it charges and lives its own
day between the visits; a birthday party sets it back in its nest to charge when it runs
low. A nest duck with an empty battery off its nest cannot be switched on (it would only
switch off again): the switched-off banner's way forward is 「放回窩裡充電、開機」, and it
says when it was switched off (not the clock now).

The 3D view shows the charging place from the home the physics runs: since 2026-09-27 a
dock against the wall (see "The charging dock" below; phase 2's open pad with ring contacts,
a low rim and a backboard is the `open_nest` variant), the leash (a faint ring, Duck Core's own `leash_m`) while the family is still asleep, and
a lifted duck held up until it is put down. The clock's third line is the day's phase
(家人還在睡、家裡沒人…). Above the words of the moves, the caption's lead line says in Duck
Core's words what it is up to without a move of its own — setting off, the way home, the
marker in sight, each try at the nest, a guess and whether it came true, the family's
comings and goings as it believes them — and a clever or goofy moment (「✨ 機智時刻：…」,
magenta), each lead readable for at least 1.6 s and never ahead of the moves on screen;
the banner names a moment for three minutes (「剛剛的機智時刻：…」). A tired duck's one click
is 「讓牠睡飽再起來走走」: a nap until it is rested (`advance {"until": "nap_end"}`), after
which it gets up by itself; waking it early says it will still be too tired.

## Personality and moments (Duck Core v4, phase P1)

Every duck hatches with a temperament (seven axes and a primary temperament, the four
equally likely, drawn once from its seed) and grows up at its own pace (0.87–1.15×: a
juvenile around growth day 39–52, an adult around 157–207; the growth panel says so).
Its branch at the juvenile day is `branch-v2`: its nature, how much more of each kind of
care the family gave than most families do (praise counts as the care of what was
praised; playing with it, keeping it company quietly and being there while it explores
count too), and its own experience, then a lottery — care clearly raises the odds, the
same family can raise different ducks. Its adult name (all twelve happen) gets an
epithet from its running gag or its most frequent moment (「愛撈水的月光歌手」), and any of
the five words can be its first.

It has clever and goofy moments (`duckmind.duckcore.moments`, `duck.moment`): clever ones
are skills it really learned and always start with a guess (`duck.predict`) that must
come true (`duck.predict_outcome`) — it remembers when you come home, docks in one go
from far away, walks to where you usually sit when you call it; goofy ones are real bird
behaviour (preening mid-errand, dabbling on a dry floor, standing up and sitting straight
back down) and happen only when it is fed, rested and not in a hurry, always with the
comic beat (a freeze, a double take or 嗯？, a sheepish preen). A wrong guess becomes the
goofy one, and a moment the family reacts to on three growth days becomes a running gag.
The growth panel shows today's moments, its wits and the day's form. The content table
changed (`duckcore-content-3`): a saved duck is re-lived from its inputs under the new
rules, so its life differs from before.

For experiments without touching the owner's page, run your own server on another port
with `--state-file ''` (it then keeps nothing between runs).

## What the 3D view shows

Head and mouth follow Duck Core's clip frames exactly as a clip player would send
them: `HOME + JOINT_SIGN · SEATED_ENVELOPE · delta`
(`duckmind.duckcore.content.head_targets`), low-pass filtered like robotd's head
smoothing (~1.78 Hz). The sign table was derived by forward kinematics on the
official MJCF (the face points +x): `neck_pitch` + leans back, `head_pitch` −
looks up (robotd's chorale assumes the same), `head_yaw`/`head_roll` + go to the
duck's left. `tests/test_simweb.py` re-derives it from the MJCF when the checkout is
present. The MJCF jaw is a fixed geom while the real mouth is a servo (−5°..+30°,
duck-control's `MOUTH_CLOSED`/`MOUTH_OPEN`), so the hinge pin is read off the jaw
mesh's rear lobe. Legs, pickup lift and gamepad replays are illustrative only:
Duck Core never drives the legs.

Add `?debug3d` to expose `window.__duck3d` for calibration scripts.

## The duck's voice (叫聲, owner 2026-09-28)

「模擬器鴨子叫時要真的播放出聲音，希望能跟官方的叫聲一致」: every call Duck Core makes is played
in the duck's own official Microduck voice. The voice is upstream's `sounds` crate (Apache-2.0), a
seedable synth: `--microduck` (or `DUCKMIND_MICRODUCK`, live physics or not) names the checkout,
and its built `target/debug/sounds` renders each duck's bank once, on demand (about a second), into
`<data folder>/voices/v5/seed<voice seed>/` (`duckmind.paths.data_dir()`) — upstream's layout and `.seed` marker, never copied
into this repository (`src/duckmind/simweb/voice.py`). Without the binary the page says so in
plain words under the stage and the duck has only its speech bubbles.

- **Which voice**: a simulated duck's hardware identity is its Duck Core seed, so its voice seed is
  upstream's own `seed_from_id("duckmind-duck:<seed>")` (sha256 → u32, as a robot hashes its SoC
  serial). Each duck keeps one voice; 蛋黃 (seed 7) has voice seed 2,838,558,865.
  `python -m duckmind.simweb.voice 7 --render` prints it and renders the bank ahead of time.
- **Which events**: those whose payload records the sounds the duck made (`sounds`: expressions,
  reactions, greetings, bids, bedtime, a stop, a sleepy mumble; `voice.babble`'s `tags`). The
  server puts them on each feed event as `calls`; words-only events make no sound.
- **Like robotd**: a random variant of a tag each time it plays; one voice at a time (a later call
  cuts one still sounding; a fast-forward drops it instead); a wheee plays start → loop (repeated
  while the move carrying it lasts, decided per pass ~250 ms ahead, as robotd's writer does) → end,
  and keeps the speaker meanwhile. The calls of one moment are one utterance, said in order.
- **The beak** opens with the sound's own loudness (read on the audio clock), in 3D and on the
  2D duck — silently too when this viewer muted it or has not clicked yet.
- **Routes**: `GET /api/voice/manifest` (the current duck's tags, variants and URLs, or why there
  is no voice yet) and `GET /api/voice/v5/<voice seed>/<tag>/<file>.wav` (only the files upstream
  renders, only for the current duck).
- **The controls** under the stage: 「🔊 叫聲 開／🔇 叫聲 關」 and the volume, this viewer's own
  (localStorage). Browsers allow sound only after a click or a key; the note says so until then.

## Live physics (`--live`)

With `--live` (and `--microduck`, `--microduck-rl`, `--policies`, `--robotd-bin`) every
motion is also played on the official `robotd --sim` and MuJoCo body
(`duckmind.livebody`). The body server is started through `duckmind.bodyhost`, which
imports the official `mjlab_microduck.sim.body_server` and edits no upstream file (owner
decisions 2026-09-26, phase 0):

- the full-body collision model: the official `scene_allcollisions.xml`
  (`robot_allcollisions.xml`: 70 colliding robot geoms instead of 11, so thighs, knees and
  head meet the floor), the same scene as the official `scene.xml` in everything else;
- lockstep stepping: each joint-target write from robotd advances the world by one 20 ms
  control period and nothing else does, so a busy Mac slows the simulation down instead
  of letting the physics run on without commands; if robotd stops, the world stops;
- a stall monitor, in every `/api/body` frame as `frame.monitor`: `max_stall_ms_10s` and
  `max_stall_ms_60s` (the longest wall-clock gap between physics advances), `robotd_hz`,
  `realtime` (over 10 s) and `trust`, which is `"suspect"`, with plain reasons in
  `why_zh`, when robotd stopped writing, sent fewer than 45 commands in any second of the
  last 10 s, the simulation ran below 0.97 of real time over them, or the physics stalled:
  60 ms or more with real-time stepping (the world then catches up with no command in
  between: phase 1's only cause of failed rises), 200 ms or more in lockstep. In lockstep a
  shorter stall is a late robotd tick the world waits for — robotd's next read is still
  20 ms of physics after its write (phase 1: lockstep rises 30/30 under 10 % × 100 ms
  stalls and 30/30 with robotd slowed to 17–20 Hz) — so it shows in the numbers but is not a
  reason; 200 ms is robotd's own simulator timeout, past which it drops the link.

The host refuses to start, naming what changed in the live body's error, when an official
function it patches or relies on differs from the reviewed source
(`bodyhost.UPSTREAM_SHA256`) or the two official scenes differ in more than the robot they
include. Everything else that decides the physics — the whole `body_server.py`, the scene
and robot files of both collision models and the robots' meshes — is pinned whole
(`bodyhost.REVIEWED_SHA256`): if any of it changes the host still starts, but the monitor
stays `suspect`, naming the file (「官方的 … 和核對過的版本不一樣：還沒有人核對，物理可能變了」),
until someone has reviewed the change and updated the pin (`x_info` lists it as
`unreviewed`). For comparisons, `DUCKMIND_LIVE_COLLISION=groundcontact` and
`DUCKMIND_LIVE_LOCKSTEP=0` in the server's environment bring back the old collision model
and real-time stepping. `tests/test_bodyhost.py` has an opt-in live smoke run
(`DUCKMIND_LIVE_SMOKE=1` with the `DUCKMIND_MICRODUCK`, `DUCKMIND_MICRODUCK_RL`,
`DUCKMIND_POLICIES` and `DUCKMIND_ROBOTD` paths).

### What the page shows with live physics (phase 0, 「看見真相」)

(Phase 2 replaced this section's anchors with world operations and a real hand, and the flat
floor with the physical home: see the next section. What follows about the two ducks, the
trail, 「物理不可信」 and the eye still holds.)

The 3D view draws the duck from the physics truth only (spec §13, G15/W05). Every
`/api/body` frame carries `truth` — `x_mm`, `y_mm`, `yaw_mrad` in Duck Core's room frame
(the nest or mat at 0, 0), `anchored_at` (epoch ms) and `anchor_why` — computed by
`simweb.live.LiveBridge` from the physics trunk pose and an *anchor*: the physics world (the
body starts at its own origin) is seated in the room where Duck Core's pose is when the body
starts or restarts (`start`: a new duck, 「重設物理身體」) and after time the physics did not
live — a fast-forward step, a catch-up, a summary (`sync`). Each anchor is taken where the
body comes to rest (within 3 s) — but only a change of posture is absorbed that way (sitting
down the trunk slides back ~7.5 cm and twists up to ~0.28 rad and back; absorbed up to 10 cm
and 0.5 rad from where the anchor found it): a body that walks,
turns or falls is going somewhere, so the world stays seated where it last held still and
the truth follows every centimetre of it. The heading that seats the world is the twist of
the trunk about the vertical, which stays right for a body lying on its face or back. These
are world operations: the physics body is never moved and Duck Core is never told. A hand
setting the duck down is the third (`hand`, since 2026-09-27): the physics world has no hands
yet (W03, phase 2) and its floor is empty, so 「放到地上」「放回桌上」「放回窩裡」, a pickup
(back at its mat) and the family putting it into its nest seat the world where the hand put
it, with the caption 「手把牠放到這裡…」. Between anchors the truth is whatever the physics did
and nothing else moves it (W05): walking "into" the nest moves only Duck Core's idea of where
it is (the shadow, with a caption saying so), and after 「示範走一段」 the body stays where it
walked; 「重設物理身體」 brings the two together again. Phase 2's hands will carry the body
instead. The trunk is the reference point, so
sitting down or standing up moves the truth a few centimetres.

- The solid duck: the truth's position and heading, the physics trunk height and lean, and
  the physics joints (「頭部也用物理」 is now ticked by default; untick it to draw Duck Core's
  head command on it instead; the mouth is always the command: the model has no mouth).
- The shadow duck: translucent blue, at Duck Core's plan (its pose and moves), with a ring
  of its own position error (`pose_err_mm`, which the server adds to `state.body`) and a
  blue line on the floor from the duck to it; it fades out where it would lie on the solid
  duck. A fall Duck Core rolled on its dice falls only the shadow.
- The last 10 minutes of the truth: a green line on the floor (smoothed over one gait
  period), broken at every new anchor.
- Everything the physics world does not have — walls, furniture, the nest — is drawn
  translucent while the truth is drawn, and a label under the clock says which duck is drawn:
  「物理真相」 (with 「藍色影子＝牠以為的位置；淡的家具：只有畫面，物理裡沒有」), or
  「這是小鴨的計畫，不是物理」 in a fast-forward, without live physics, while the body starts
  or with 「真實物理」 unticked. On a phone the label keeps to its title and its words go
  into the readout under the view, which also says where the body is and where the duck
  thinks it is (「離窩 0.83 m（物理）・牠以為 0.00 m・差 83 公分」).
- 「物理不可信」 (3D and 2D alike) whenever the stall monitor says `suspect`, listing its
  reasons (`/api/body` now carries `monitor` at the top level too), and whenever the body
  itself broke — robotd or the body server stopped or hangs, or the host refused to start:
  `/api/body` then has `error` (the program's words, shown as a tooltip) and `error_zh`
  (the same in plain words, first in the list). The label then says
  「物理身體出錯了，停住不動（原因在右上）」, never 「物理身體啟動中」; an older server whose
  frames carry no truth is named as such.
- The 2D duck (no WebGL, or 3D failed to load) plays only Duck Core's plan: with live
  physics the label says 「這是小鴨的計畫，不是物理」 there too.
- 「牠的鏡頭看到的（每 2 秒一張）」 (3D and 2D alike): the duck's-eye picture
  (`simweb.eyecam.EyeCam`, started and stopped with the live body, posed from its frames, lit
  by `world.lights` at Duck Core's time), with the room's light and the level its light meter
  read. `GET /api/eye.jpg` serves the latest picture (`ETag`; 304 when unchanged; 404 with the
  reason in words when there is none) and `/api/body` its status (`eye`: `t`, `age_s`,
  `level`, `label`, `render_ms`, `error`…). When the physics stands still (the frame's
  physics time stops for 1 s: robotd or the body server stopped) no new picture is made —
  `eye.frozen`, and the page says 「物理停住了：這是 N 秒前的畫面」; when the body broke,
  `eye.error` says the picture is the last one before it did. Right after the light changes
  the exposure is still the old room's; while the frame is far off mediad's setpoint the
  meter judges from the frame alone (`eye.adapting`, 「（鏡頭還在調整）」). The page fetches it every 2 s only while the
  window is open, on screen and the tab visible, and not during a fast-forward (the physics
  body only lives real time); on a phone it starts closed.
- The room's light is the server's: every `/api/call` answer carries `server.lights` (the
  light at the duck's time from `world.lights`); the page's own rules remain only for an
  older server. Nights are now moonlit rather than dusky (the page's old twilight line never
  faded).

### The physical home (phase 2, 「真的家與真的感測器」)

With `--live` the body now runs in the world server (`duckmind.world.simhome`) in the page's
6×6 room made physical, `default_home` (owner decision A1): its four walls, the sofa, the
coffee table, the bookshelf, the plant pot, the toys and the charging dock are in the physics
(`src/duckmind/world/homes/default_home.json`; the dock: see "The charging dock" below). A duck without a nest — 蛋黃's old save —
gets its `no_nest` variant: the mat, and "the desk" is the mat (A2, A3). The household
(`duckmind.world.worldkit`: the nest's contact, a visible hand, the family's bodies, the world
battery, the things) and the sensors (`duckmind.world.sensors`: the camera behind
`media.frame`, the ToF behind the official `tofd`, when it is built next to robotd) are loaded
into it. `DUCKMIND_LIVE_HOME` picks another home (`flat`, a layout name such as `apartment`, a
layout file; a duck without a nest gets the `no_nest` variant when the home has one);
`DUCKMIND_LIVE_HOME=bodyhost` brings back phase 0's body host on the flat floor (no world
ops; the drawn room is then translucent: only on the page). `DUCKMIND_SIMHOME_EXT` replaces
the extension list.

- **Where the duck is.** Every frame's `truth` is the physics body in the home's world
  coordinates (`x_mm`, `y_mm`, `yaw_mrad`, `frame: "world"`; in `default_home` the same as
  Duck Core's room) and `placed_at` / `placed_why`: when a world operation last moved the body.
  Duck Core's plan is moved into the world by the home's frame (its nest, else its mat;
  identity in `default_home`).
- **World operations, labelled.** When the body starts it is put where Duck Core is
  (`x_place`, 「物理世界把牠放到小鴨現在的位置」); after a fast-forward, once the steps stop
  (3 s without one, at every page speed: 1, 5 and 15 min/s), where the plan then is
  (「…物理世界把身體放到小鴨計畫的位置」). A spot taken by furniture or a wall moves to the
  nearest free one within 80 cm, and says so in plain words (沙發、牆…). The stand-in spot is
  always inside the house: `x_place` refuses spots outside the layout's rooms (the official
  floor plane runs on past the walls), and one that stands in for a taken spot must not be
  over a drop or behind a wall, railing or gate from it (`"from_mm"`; `"force": true` skips
  both, for evaluators).
- **The hand** (D1). Picked up, put on the floor, on the desk (the mat), into the nest, set
  upright: a visible hand carries the physics body there in 2–4 s — the shadow (the plan)
  jumps at once, the solid body arrives after. Held in the air the body sits with its legs
  tucked (walks wait). A duck under the coffee table (40 cm clear) is first slid out along the
  floor, and a spot under a table top is never a set-down spot: the hand never pushes the body
  through furniture. 「扶起物理身體」 (under the view, and in card 09) is the everyday
  recovery after a physics fall (D2): the hand rights it where it lies; 「重設物理身體」 stays as
  the fallback (a new body). A fall only Duck Core's plan had (a dice roll in its family
  routine) brings no hand: the physics body did not fall. During a 示範 the hand waits.
- **The family** (D3, D4). A touch pressed as a family member (摸頭, 抱起來, 放回窩裡, 扶起來,
  換新電池, the power button…) first brings that person's body over (they keep 40 cm or more
  from the duck); their hand does it, and only then does Duck Core hear about it (a few
  seconds after the button: 「小孩先走到小鴨旁邊…」). Voice, the gamepad, the other buttons and
  the banner's suggested steps (they must stay in their order) go to Duck Core at once. People appear only for these buttons, linger a few seconds and
  leave.
- **Drawn from the same layout.** The room is `static/home3d.js`, from `/api/home` (or
  `/api/body?home=<fingerprint>`'s `home_json`): nothing is marked 「只有畫面」 any more, walls
  between the camera and the duck fade (B4), the nest's AprilTags are the physics' own bits,
  its charging light and backlight follow the world. The line under the physics controls says
  which home runs, whether the nest's contact is closed, what the hand is doing and who came
  over; the world battery sits under Duck Core's (「物理電池」: display only, never given to
  robotd until phase 5, E1).
- **The eye window** is a `media.frame` client (`duckmind.eye.frames.EyeWindow`): the world's
  camera renders a frame only while the window asks (F4). The camera sees the floor the page
  draws: the layout's `"texture": "wood"` gives the physics' copy the page's planks (sRGB, as
  three.js decodes them), not the official blue-grey checker; the page's light is the light of
  the room the duck is in, from the same layout and model as the camera's (world.lights). A
  world whose camera did not start shows no picture and says why.
- **The open nest's backboard** (the `open_nest` variants) stands 480 mm behind the pad
  centre (the right-arc exit of a duck sitting toward the board touched it at 410 mm; phase-2
  gate evidence); the apartment's open nest keeps 410 mm (480 mm reaches its south wall).
- **Card 09 物理世界** (模擬工具): 示範 撞撞看／啄一下／踢一下 on a chosen toy (the body walks
  over, the world puts the toy where the official skill meets it, the skill runs; Duck Core is
  not told), 滾球給牠, 收玩具, 扶起物理身體. Without `--live` its buttons are greyed out.

`API_VERSION` is 8. The page tests are `tests/test_page_truth_source.py` (static checks and
`static/truth.js` run in node) and `tests/test_simweb.py`.

### The charging dock (owner decision 2026-09-27: 「窩 做成充電座 樣式，公寓的窩 放置在牆邊」)

The nest is now a robot-vacuum-style dock standing against a wall (`duckmind.world.dock`,
built from our own primitives through MjSpec): a housing whose back is on the wall, a cyan
face panel with the tag36h11 tag 17 (C3), a 4 mm floor plate ramped on every side (no step:
the feet lift ~1.1 cm), V side guides (a ±130 mm channel, then a flare to a 460 mm mouth,
sized from the duck's footprint and its sit), tag 18 flat at the plate's front edge, and a
status light strip above the panel (C4: on only in the dark, the charging colour while the
duck charges; the camera's emission pass picks it). Two docking styles, a layout variant each:

- `backin` (the homes' default since 2026-09-28, `world.dock.DEFAULT_STYLE`): it turns round in
  front of the dock and reverses in (blind: the camera cannot see behind), stops about 0.3 m out
  (the walk coasts 60–75 mm, the sit moves the trunk ~72 mm back), and sits; a probe on its lower
  rear presses the spring plates on the face. It leaves by walking straight out. The layout's
  `nest.side` says which side of the dock it turns round on — `right` (`default_home`,
  `plain_home`) or the mirrored `left` (the apartment: the corridor wall is on the dock's right) —
  and the page's 「用眼睛做」 passes it to duck-body (`--dock-side`).
- `forward` (the `forward` variant; the default until 2026-09-28): the duck walks in facing the
  tag until its chest meets the soft bumper, and sits; its thighs land on two floor pads (+ and −)
  and a probe under each thigh closes the contact. It leaves by reversing straight out (phase 1:
  reversing is stable).

Both report power-good through the nest's contact path (`pg`, the world battery,
`x_world` → `nest` with `kind`, `style`, `seat_mm`, every probe and whether the bumper is
touched). The probes are massless and in group 5: they push nothing. `open_nest`,
`edge_3mm` and `ramp` bring back phase 2's open nest for comparison.

`default_home` stands the dock in the middle of the south wall. Duck Core's frame stays the
room (`"frame": "room"`), so its imagined walls are the walls; a nest duck still believes its
nest is at its origin (0, 0), the middle of the room, until phase 5 teaches it the dock. The
page shows both honestly: a blue ring on the floor at (0, 0) (「小鴨以為的窩」), the world line
「充電座在牆邊；小鴨還以為窩在房間正中間…」, the readout 「離充電座 … m（物理）・牠以為離窩 … m」;
when the body starts (or after a fast-forward) with the duck in its nest, the world puts it in
the physical dock, seated as its style docks, and says so; 「放回窩裡」 is carried into the dock
by the hand. 蛋黃 (`default_home,no_nest`) is unchanged: its mat at the origin.

The apartment's dock stands on the living room's south wall at x = −1.70 m, facing +y,
between the TV stand and the plant: the one wall spot of the living room where the dock, its
0.9 m lane and its turning room are clear of furniture (the south-west corner's shelf blocked
phase 2's nest exit; the west wall between the shelf and the sofa puts the sofa's corner in the
lane and has the camera face the window). The toy corner, the tissue box and the books moved
out of its way; the official dock and the blue landmark stay hidden (B1). The layout checks
keep a dock's back on a wall (`dock_not_on_wall`), its lane and turning room clear
(`nest_blocked`) and a toy corner off it (`zone_on_nest`). Headless sanity runs and the API for
measurements: `docs/validation/evidence/phase2/dock_sanity.json`; tests:
`tests/test_world_dock.py` (in the render check).

## Phase 3 (「小鴨的眼睛與身體」): duck-eye, duck-body and the curiosity step

Phase 3 builds the duck's own sensing and body processes (physical-design spec §4–§7, §15, §17
「期3」; owner decisions 2026-09-27 A1–E4). Each is a pure, clock-injected step function; the
sockets, threads and clocks live only in its `server.py`, so the headless rig and the tests drive
the same code. Every `*_ns` / `t_ms` is `CLOCK_MONOTONIC` (`time.clock_gettime_ns`; on macOS
`time.monotonic_ns` is another clock, ~10 s away from robotd's). The board side — `duckmind.eye`,
`duckmind.duckbody`, `duckmind.curiosity`, `duckmind.percept` — imports nothing of the world, the
page, MuJoCo or Duck Core (`tests/test_imports.py`; duck-body's integer trig is its own pinned copy,
`duckbody/fixedmath.py`).

- **duck-eye** (`python -m duckmind.eye.server --socket <dir>/eye.sock [--floor-model F]`): eye-v1
  JSON lines (`eye/protocol.py`). duck-body says `eye.hello`, then sends `eye.look` with one
  `media.sock` fd over SCM_RIGHTS (its `hello` already answered), `eye.pose` (the 25 Hz pose tap),
  `eye.health` and `eye.learn`; the eye answers `eye.result` (integers only). Tasks: `light` (the
  board-path meter, C3 refit 77 / 3.1 / 5.5, sim-provisional; C1: the dock light in the dark reads
  「很暗」), `floor` (FL-v1: the floor learnt from walked-safe ground, D1 — `--floor-model` keeps
  it between runs), `nest` (the dock's AprilTags through pupil-apriltags, D3, OpenCV fallback;
  the light and the cyan panel give direction only, C5), `things` (colour, histogram, registered
  toys; only once the floor is learnt), `duck` (duck-detect framework, no weights, D4). The pose
  tap's camera is robotd's `frames.camera` — the *upright* cv2 frame — marked
  `cam_frame: "cv2_upright"`; duck-eye turns it by the frame's `rotate` into its sensor frame.
- **duck-body** (`python -m duckmind.duckbody --robotd <state>/robotd.sock --body <state>/body.sock
  --mjcf <microduck>/robot_allcollisions.xml --anchors <upstream>/microduck/odometry/src/anchors.rs
  [--tof <tof.sock>] [--eye <eye.sock> --media <media.sock>]`): body-v1 on `body.sock`
  (`duckbody/protocol.py`): `hello`, then `cmd` lines (`duck.walk`, `duck.look`, `duck.posture`,
  `duck.dock`, `duck.curio`, `duck.stop`) with `act_id` and `deadline_ms`; answers `ack`,
  `percept` (percept-v1, `duckmind/percept.py`) and `done`. One act at a time; every robotd
  intent passes the forbidden-command guard (`duckbody/reflex.py`). Normal walks up to 0.55 m/s
  (A3), approaches 0.40; B1 fall trigger (tilt > 40° for 1 s); B2 sit-then-stand second try on;
  B3/B4 pickup from the legs' load (it sits when *it* senses the lift). Each finished walk's
  `eye.learn` goes to duck-eye. Its stderr says when duck-eye is missing, a look had no frame or
  the guard refused an intent (`DUCKMIND_BODY_LOG_LOOKS=1`: every look). duck-body must be
  robotd's only writer: the page's live body is then passive (`LiveBody.passive`). `duck.dock`
  docks `backin` by default (owner 2026-09-28 「6. 可以」: dock:measure's choreography from
  `docs/validation/evidence/phase2/dock_styles.json` — the J-hook to 1.5 m, the far U-turn at
  1.15 m, the straight reverse start, the alignment gate, the face stop with a 0.7 s walk-out,
  "stand, sit again"); `--dock-style forward|backin` follows the home's dock (the homes' docks
  are `backin` since 2026-09-28; their `forward` variant builds the other one) and
  `--dock-side left` (or the command's `side`; the page passes the layout's `nest.side`) mirrors
  the J-hook where the room needs it (the apartment). After a docking a
  `duck.walk` with purpose `leave_nest` steers out along the dock's axis. Reversing: curiosity's
  back_arc, inside `duck.dock` (owner 「1. 可以」) and a small back step — `duck.walk` with
  `vx_mm < 0`, taken only when duck-body's own senses find ahead, left and right shut
  (`duckbody/cornered.py`; owner 「2. 可以」), −0.30…−0.40 m/s, ≤ 0.2 m. The look veto (owner
  「3. 一律放寬」): floor no look has seen free only shortens or slows a walk (a 2 s step at 0.40,
  then a glance), a wall or thing a recent look saw still stops it (owner ~12:30 「1. 照建議」).
  **The dark refuses nothing** (owner 2026-09-28 ~12:30 「2. 放寬」, C2 reversed): in a dark or
  black room (or with the eye down) a walk goes on what was seen free while the eye could see
  (10 min bright, 5 min dim, whatever the light is now) and steps slowly into the rest; the eye
  simply sees less there (dim −20 %, dark and black no floor), and physics decides whether it
  bumps or falls. Since the phase-4 landing (「一律放寬」) the dark no longer caps a segment at
  3 s either (it was pacing only): a dark walk is as long as the same walk in the light. C1
  stays a light-level fact (the dock's light alone in the dark reads 「很暗」). **Attention** (owner ~12:30 「保留鴨子偶爾不注意撞到東西的呆萌感」): a
  `duck.walk` may say `attention` — `on` (the default, Duck Core v4's walks: every glance rule),
  `low` (sleepy, playing, a hatchling, too bold to check: it glances only when leaving the nest,
  just put down or just bumped) or `elsewhere` (its gaze is on a toy, a person, a sound: no
  glance before, no look after). An inattentive duck does not know which floor it has not looked
  at, so it walks there as asked (not the attentive duck's slow unseen step) and may bump into
  what is there; a wall or thing it did see recently still stops it. duck-body never skips a look
  by chance (the brain decides, from its state). `walk.done` carries `attention` when it was not
  `on`. Measured headless (default_home, bumps per 100 walks, 0 falls unless said): attentive 5.0
  (mostly things its eye misses: the terracotta plant pot, thin table legs), distracted 6.3,
  `low` 9.3 (1 fall in 30 runs); a box it never looked at: attentive 0/30 hit, distracted 4/30
  (`evidence/phase3/attention_bumps_owner0928b.json`). The dark: 56/60 pitch-dark departures went,
  0 falls; walking a room never seen lit, 1 fall in 60 runs (`dark_relaxed_owner0928b.json`).
- **curiosity** (`duckmind.curiosity`): the Act Runner, one `duck.curio` step (look → approach →
  confirm → act → watch → report `body.curio_step`); A1 relaxes the ToF 30 cm stop only toward
  the just-seen thing, A2 the final approach may go head down (with a no-progress head-up
  fallback), A5 the knock's sound is `unknown`. `label: "*"` means the thing at the believed place
  whatever the eye calls it. In the dark (since the phase-4 landing, 「一律放寬」) an act starts
  when the duck's own perception still has the thing — a sighting its tracker still keeps (10 s,
  3 s if it moved; duck-body hands the next step on the same thing the last step's memory, in the
  same odometry frame) or the thing in its beak (`carry`, `put_down`) — and ends `dark`
  (「太暗了，看不到它」) otherwise: duck-eye names no things in dark frames, so it cannot see it
  there. Started on a memory, its looks see nothing in the dark, so it walks toward where it
  remembers the thing and ends `lost` if it cannot confirm it; in the light nothing changed.
  `duckmind.world.jaw` (A4, provisional pivot) is an opt-in world extension, not in the live
  body's default: `DUCKMIND_SIMHOME_EXT=duckmind.world.jaw,duckmind.world.worldkit,
  duckmind.world.sensors` (the jaw before the sensors).

**「用眼睛做」 on the page (A6, card 09).** With `--live`, card 09 has 「用眼睛做：撞撞看 / 啄一下 /
踢一下」 under the 示範 buttons. The world only puts the chosen thing on the floor roughly half a
metre in front of the duck (the duck is told only "in front of you"). Then the page's server
starts `duckmind.eye.server` and `duckmind.duckbody` against the live body's own sockets, the live
body goes passive, duck-body stands the duck (`duck.posture`), looks once (`duck.look`: the light)
and runs one `duck.curio` step with duck-eye's `things` task, and the page shows the duck's own
report (`body.curio_step`) in plain words. The processes stop after the step and the live body
takes over again; Duck Core is not told. The duck's learned floor is kept next to its saved state
(`<state>.eye/<home>.floor.json`); a duck in `default_home` starts with the floor learnt there in
headless renders (`simweb/floors/default_home.json`), elsewhere its eye starts unlearned and names
no things until it has walked (a duck cannot yet walk to learn its first floor: see the phase-3
evidence, `integration.json`). The server's Python needs the phase-3 dependencies (numpy,
pupil-apriltags, onnxruntime: required since D5, so `uv run` in the repository installs them). The
two processes' logs are in the live body's run folder (`<dm-live-…>/phase3/`), removed with it
unless `DUCKMIND_KEEP_LIVE_LOGS=1`.

**The gate harness** (headless, no live slot): `duckmind.world.headless_duck` (robotd's control
path in-process), `duckmind.world.rig` (the lockstep rig: physics 50 Hz, the multi-ray ToF at
tofd's cadence, media.frame from the world camera, the `pg` op, the light ladder, C1/C3 truth),
`duckmind.world.rig_scenes` (phase 1's obstacle classes and the apartment stairs) and
`homes/plain_home.json` (D1's plain-floor comparison; made from `default_home` by
a measurement script that is not in this repository, for tests only). The gate scripts and their run
lines are part of the author's measurement records, not in this repository.

**Checks.** The standard command (above) now runs duck-eye's numpy tests too (numpy is a project
dependency). The render check covers the phase-3 render and numpy tests and needs the phase-3
packages:

```bash
DUCKMIND_REQUIRE_RENDER=1 DUCKMIND_RENDER_ONLY=1 \
DUCKMIND_MICRODUCK_MJCF=/path/to/microduck_rl/src/mjlab_microduck/robot/microduck \
  uv run --python 3.12 --with pytest --with numpy --with mujoco==3.10.0 --with pillow \
  --with pupil-apriltags --with onnxruntime --with onnx --with opencv-python-headless \
  python -m pytest -q -p no:warnings tests
```

`DUCKMIND_RENDER_ONLY=1` runs exactly the files in `tests/conftest.py`'s `RENDER_FILES` (phase 0–3).

Evidence: `docs/validation/evidence/phase3/` (`MANIFEST.json` lists every file with its sha256
and where its raw data lives in the author's measurement records, not in this repository).

## Phase 4 (「邊界」): Duck Core learns its world only from its inputs

Phase 4 puts a boundary between Duck Core and the world (`docs/design/phase4-boundary.md`; design
decisions A1–A14 in the author's design notes, not in this repository). Duck Core `v4_1` pulls percept-v1 events
from a **world port** (`duckmind/worldport.py`: the port protocol, the registry, the replay tapes),
sends its commands as events with act ids (`duckcore/acts.py`), and no longer rolls v4's outcome
dice: the ten dice, the fall and righting schedules and the touch detectors live in **LegacyWorld
v1** (`duckmind/world/legacy.py`, which registers itself on import), with the duck's own key, so a
`v4_1` duck with LegacyWorld behaves exactly as v4 did (the gates in
`docs/validation/evidence/phase4/`). The three kinds the world hands in for the family's hands and
the gamepad (`sense.touch`, `sense.hold_head`, `body.play`) are percept-v1's own since the landing
(WP9, `duckmind/percept.py`).

- **Which engine a duck runs.** A new page duck (「重新孵化」) is `engine_rev: v4_1`: the server
  opens it through `api.open_session` and saves `{"config", "world", "inputs"}` (the world header
  carries the outcome rates; the dialog's 「測試用：走路很容易跌倒」 sets its
  `rates.fall_ppm_per_s` to 150,000). **An old save (no `engine_rev`, 蛋黃's) opens on the frozen
  `v4_0` engine** (`duckmind/engines/v4_0/`, generated from git by `tools/freeze_engine.py`,
  never edited) with the identical chain head, and is saved back with `engine_rev: v4_0` and no
  world header (an older build ignores the field). `POST /api/call {"method": "state"}` answers
  `state.engine_rev`; `replay_check` works for both, `percept_replay_check` and
  `world_replay_check` only for `v4_1` ducks (a `v4_0` duck has no world port). Moving a `v4_0`
  duck to `v4_1` with its memories (`api.migrate`, the WP13 seam; owner P4-1 = a) is phase 5's.
- **The page.** Unchanged controls. A `v4_1` fall is Duck Core's own `duck.posture {posture:
  fallen, at, walk_cut}` right after the world's `body.fell`: the 3D view cuts the walk where it
  fell and lays the body down at `at`, as it did with v4's `body.fell`.
- **Pinned.** `tests/golden_v4_1/` (five saves, `tools/record_golden_v4_1.py --check`) and
  `tests/golden/` (six of the 50-session v4 corpus; the full corpus in
  the author's records, not in this repository) replay on every run; `tests/test_engine_frozen.py`
  checks the frozen engine's bytes. Any change to Duck Core's or LegacyWorld's behaviour or
  records moves these heads: then freeze `v4_1` first (boundary §11), never re-record.

Checks (besides the standard command):

```bash
uv run --python 3.12 python tools/freeze_engine.py --check        # "v4_0: 23 files … identical to git"
uv run --python 3.12 python tools/record_golden_v4_1.py --check   # the five v4_1 saves: "same"
```

The gates (G1 behaviour projection per duck, G2 the population numbers, G3 cost, G4 the golden
corpus and 蛋黃's read-only copy through `api.open_session` / `api.restore`, G5 the three replays,
WP13 seams) and their raw rows: `docs/validation/evidence/phase4/` (`MANIFEST.json`) and
the author's measurement records (with the landing's own re-runs), not in this repository.

## What is verified

```bash
DUCKMIND_MICRODUCK_MJCF=/path/to/microduck_rl/src/mjlab_microduck/robot/microduck \
  uv run --python 3.12 python -m pytest tests/test_duckcore_*.py tests/test_simweb.py
```

Without `DUCKMIND_MICRODUCK_MJCF` the two official-model tests are skipped.

**The render check (phase 0; must pass too).** The standard check runs without numpy and
MuJoCo, so the camera model (P5: checkerboard and principal-point sign), the light meter,
the duck's-eye camera and its JPEG encoder, and the MuJoCo comparison of the two collision
models skip there — the run ends by saying how many. Run them with:

```bash
DUCKMIND_REQUIRE_RENDER=1 \
DUCKMIND_MICRODUCK_MJCF=/path/to/microduck_rl/src/mjlab_microduck/robot/microduck \
  uv run --python 3.12 --with pytest --with numpy --with mujoco==3.10.0 --with pillow \
  python -m pytest -q -p no:warnings tests/test_camera_model.py tests/test_eyecam.py \
  tests/test_eye_light.py tests/test_bodyhost.py tests/test_ae.py tests/test_lights.py \
  tests/test_homelayout.py
```

`tests/test_homelayout.py` (phase 2) composes every home and variant from the official files
by path: the duck spawns clear, things lie still for 5 s (< 1 mm), the page JSON is the physics
and the stair gate stops a ray; its layout checks and page JSON also run in the standard check.

`DUCKMIND_REQUIRE_RENDER=1` makes any of those tests that would skip (no numpy, no MuJoCo, no
GL, no checkout) fail instead (`tests/conftest.py`); only the opt-in live smoke run stays
skipped. The numpy JPEG encoder the live simulator uses (it has no Pillow) is also checked by a
small decoder in `tests/test_eyecam.py`, without Pillow.

- same seed + same inputs → the same SHA-256 event chain; the page's
  「重播驗證」 rebuilds the duck from its inputs and compares chain heads;
- growth days: full credit at ≥15 awake minutes, half credit per day off capped at
  five growth days per absence, paused without verified time;
- learning caps: diminishing preference steps per growth day, at most three
  owner-bond steps per day and only from the owner's phone, 「夠了」 teaches nothing,
  guest mode learns nothing, at most one promoted phrase per seven growth days;
- attention bids never become more likely after an absence;
- forgetting removes the fact, rolls back what it taught (never past neutral) and
  can be undone within 24 hours; growth, traits and quirks cannot be forgotten
  piecemeal; every fact cites real L1 events; the diary never uses guilt words;
- the page's JSON, static paths, mesh paths and request limits;
- with a nest (`tests/test_duckcore_daycycle.py`, `…_nest.py`, `…_excursion.py`,
  `…_social.py`, `tests/test_routines_v2.py`): the 07:00 self-wake and none without
  verified time, a sleeping family never taken for gone out, a voice alone never
  bringing the family home, the belief explained by logged cues only, resting while
  nobody is home, learning the return time and waiting for it, winding down and never
  walking at night, charging only on the nest and never walking on it, starting home
  with the reserve, three misses then a call for help, getting up failing and backing
  off, no fear of its own nest, legs inside the validated bounds and the leash, outings
  never heading into or through a spot it avoids (and staying put with nowhere to go), and
  whole months with a family replaying identically whatever the step size.

Not claimed: that the seated envelope is safe on hardware (it is a target until the
seated validation protocol runs), that the clips look right on a real duck, or any
statement about gait or balance.

## The world's sensors (phase 2)

`duckmind.world.sensors` is a world-server extension that gives the simulated duck the real
duck's sensors through the real doors (spec §11.1–§11.4). Load it with
`DUCKMIND_SIMHOME_EXT=duckmind.world.sensors` in the environment of a live run that uses the
world server (`DUCKMIND_LIVE_HOME=default_home,no_nest` for 蛋黃):

- **Head camera.** A render thread in the world server (its own CGL context) meters twice a
  second always and renders a frame only when someone asks, on the camera's 30 fps clock. It
  answers mediad's `media.frame` on `<dir>/mediad/media.sock`: one 1280×720 UYVY frame per
  connection, `rotate: 90`, 500 ms timeout, 16 connections, 5 s per connection.
  `DUCKMIND_MEDIA_QUALITY` picks another `[media] quality` rung.
- **ToF.** The body port's `tof` op is the multi-ray sensor (`duckmind.world.tof_model`). With
  `DUCKMIND_TOFD=<path>` (or `auto`: next to robotd) the official `tofd --sim … --hz 15` runs
  against it and serves `tof.stream` on `<dir>/tofd/tof.sock`. A duck without the ToF module:
  `DUCKMIND_SIM_TOF=0` — tofd still runs and says the sensor is unavailable (owner decision F2).
- **Truth for evaluators only** (body port): `x_sensors` (paths, counters, timings),
  `x_eye_log`, `x_tof_log`, `x_lights` (the lights' clock; also `x_world`'s `lights`).

`<dir>` is `DUCKMIND_SENSORS_DIR`, else `/private/tmp/dm-sensors-<body port>` (removed when the
world stops). Build tofd once (it writes only the checkout's `target/`):

```bash
cd /path/to/upstream/microduck && ~/.cargo/bin/cargo build --offline -p tof --bin tofd \
  --target aarch64-apple-darwin
```

The page's eye window client is `duckmind.eye.frames.EyeWindow` (it fetches only while the
window is open); `--live` loads the sensors and the household by default (see 「The physical
home」 above).

Checks (render check, plus the standard check for the pure parts):

```bash
DUCKMIND_REQUIRE_RENDER=1 \
DUCKMIND_MICRODUCK_MJCF=/path/to/microduck_rl/src/mjlab_microduck/robot/microduck \
  uv run --python 3.12 --with pytest --with numpy --with mujoco==3.10.0 --with pillow \
  python -m pytest -q -p no:warnings tests/test_media_frame_protocol.py \
  tests/test_tof_op_protocol.py tests/test_tof_model.py tests/test_eye_render.py \
  tests/test_sensors.py tests/test_reprojector_vectors.py
```

P1 compares the shim and the `tof` op with the upstream sources read by path (and pins the
code they copy); P4 runs the official Reprojector's six tests on the Python port
(`duckmind.duckbody.tof_reproject`). The tests that run the real tofd skip (opt-in) when it is
not built; `DUCKMIND_ROBOTD_SOCKET=<a running robotd's socket>` adds a bit-for-bit check of the
beams against `robot.model`.
