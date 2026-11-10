# Bringing up the first real duck: nest first, then hatch

繁體中文：[bring-up.zh-Hant.md](bring-up.zh-Hant.md)

**Status: not run on hardware.** No production Microduck has been connected yet. Every command here
has been run against the simulated duck (`sim-body --as-real`, the rehearsal) and against fakes in the
tests. Numbers marked *rehearsal* were measured there and say nothing yet about a real duck.

This is the ladder for the first production duck. It has five stages. A stage is entered only when
the one before it has passed, and each stage has numbers that were written down **before** any
data (registered 2026-11-10). If a number is missed, the stage is not passed: do not change the
number, go back one stage and find out why.

It does not replace:

- [SAFETY.md](../../SAFETY.md): what Duck Mind does and does not do to stay safe.
- [docs/real-microduck.md](../real-microduck.md): how the brain on a computer reaches the duck (the
  bridge, the token, `check`), and its **checklist for the first real duck**. This page says *when*
  each checklist item is done; the item itself is described there.
- [docs/deploy/board.md](board.md): confirming the board, the wheel, rollback (§4) and closing
  mediad's LAN doors (§5).

## Rules for every stage

1. **Two people.** One at the computer, one with a hand next to the duck and the duck's own power
   switch within reach. `robot.stop` is not an emergency stop ([SAFETY.md](../../SAFETY.md)): it
   zeroes walking and does not cut torque. The hand and the switch are the emergency stop.
2. **Duck Mind never turns the duck on.** A person does, with the duck's own control. The bridge on
   a real duck refuses `robot.enable`, `robot.init` and `robot.poseJoints`.
3. **One session folder** for the evidence (below: `SESSION`, any folder you choose, kept outside
   the repository). Every stage adds files to it. Start the bridge with `--events
   SESSION/bridge-events.jsonl` (on the duck; copy it back at the end of each stage).
4. **A stage log.** `SESSION/stage-N.md`: date, who was there, the software versions `check`
   printed, each pass number with the measured value, and the line "cleared by <name>" or "not
   cleared: <reason>". **Only the owner clears a stage.** The program and coding agents never do.
5. **Abort means: stop, power off by hand if needed, write down what happened, go back one stage.**
   The abort triggers of an earlier stage hold in every later stage.
6. **Rollback is one file.** The brain walks a real duck only with a valid body calibration for its
   board and its two networks (stage 3). Delete that file (`python -m duckmind.bringup show URL`
   prints its path) and the brain is head-only again at its next start. For the software on the
   board, [board.md §4](board.md).

Short names used below:

```sh
URL=remote://<duck-address>:8797          # the bridge on the duck (docs/real-microduck.md, step 4)
SESSION=<a folder for this bring-up>      # outside the repository
```

## The numbers, in one table (registered 2026-11-10)

| # | What | Pass | Where it is read |
|---|---|---|---|
| N1 | Link, 10 minutes idle | 0 drops; 0 false deadman alarms per hour; longest silence at the duck ≤ 500 ms | `check --wifi-minutes 10` |
| N2 | Leg currents are reported | `currents_reported: true` | `bringup watch` |
| N3 | Battery at the start of a stage | ≥ 50 % at rest | `check`, `bringup watch` |
| N4 | Battery under load | lowest reading of the stage ≥ 30 % (Duck Mind's own play floor); walking takes ≤ 20 points off the resting reading | `bringup watch`, the page's battery chip |
| N5 | Link cut while the legs move (deadman) | bridge sends stop ≤ 1.5 s after the last thing it heard; duck seated ≤ 4.0 s; legs stop stepping ≤ 1.0 s (video); no fall. 3 of 3 in the air (stage 1), 5 of 5 on the floor (stage 3) | `bridge-events.jsonl` (`link_lost.max_gap_ms`, `stop_sent`, `sat.after_s`), video |
| N6 | Head only, 10 minutes | the legs never move; 0 drops | by eye, `--neuro-record`, events |
| N7 | Sit–stand cycles on the floor | 5 of 5 without a fall | the bench report (`cycles`) |
| N8 | Standing: hand against floor by leg load | ratio ≥ 1.3 | the bench report, *Separations* |
| N9 | Pickups in the bench recording | standing 3 of 3 claimed, 0 claimed before the hand came, every set-down sensed; seated 3 of 3, **or** the report says the seated claim is off | the bench report, *Replay* |
| N10 | False claims in the bench recording | 0 pickups claimed on the floor or while walking; 0 walks stopped as "unloaded"; 0 falls | the bench report, *Replay* |
| N11 | ToF touch band | fitted, with a gap ≥ 16 zones between a face on the board (≤ 30 mm) and a clear board (≥ 150 mm) | the bench report, *ToF touch band* |
| N12 | Floor watch, 10 minutes, calibrated, nobody touching | 0 `lifted`, 0 `unloaded`, 0 `fell`; 0 drops | `bringup watch` (exit 0) |
| N13 | Supervised walk with the brain, 15 minutes in the square | 0 falls; 0 false pickups; 0 link drops; it stops at the square's edge 9 of 10 times without pushing for more than 2 s | the page, `--neuro-record`, video, the stage log |
| N14 | Picked up while walking, 5 times | legs stop stepping ≤ 2 s, 5 of 5; the page shows "picked up" ≤ 4 s, 5 of 5 | video, the page |
| N15 | Free roam, three sessions of 30 minutes | 0 falls it could not get up from, ≤ 1 fall in all; ≤ 1 false pickup per hour; ≤ 1 link drop per hour, each ending seated within N5; never pushing on a thing for more than 2 s more than twice a session; ends above 30 % or seated | the page, `--neuro-record`, events, video |

N5's limits come from the design (bridge deadman 1.0 s, robotd's own 0.5 s, a 2 s sit ramp) with
margin; in the *rehearsal* the same cuts gave stop 1.00 s (worst 1.02 s) and seated 1.81 s (worst
2.00 s) over 16 cuts ([real-microduck.md](../real-microduck.md), *Rehearsal results*).

## Stage 0 — read only, nothing moves

**Entry.** The duck is on a stand or lying seated on a table, powered, **not enabled**. The board is
confirmed ([board.md §1](board.md)). Wi-Fi, the token and the bridge are set up
([real-microduck.md](../real-microduck.md), steps 1–4). mediad's LAN doors are closed
([board.md §5](board.md)).

**Commands.**

```sh
# on the duck (Pollen's own tool; keep the output)
robotctl health

# on the computer
python -m duckmind.remotebody check $URL --wifi-minutes 10 | tee $SESSION/stage0-check.txt
python -m duckmind.bringup show $URL                       # expect: no valid calibration (none): head only
python -m duckmind.bringup watch $URL --out $SESSION --minutes 5
```

**Record.** `stage0-check.txt`, `watch-*.json`, `robotctl health`'s output, the events file.

**Pass.** N1, N2, N3, and in `check`'s lines:

- the walk and sit-stand sha256 are printed, and which policy set they are (v8 or v9: see *Open
  questions*);
- `robotd.toml`: `idle False` (else `check` says what to change on the duck);
- `usage data (Pollen telemetry): off`;
- mediad's doors: `check` says 8443 and 8080 do **not** answer from the computer;
- the events file has no `refused_operator` line (nothing asked robotd to enable anything).

Checklist items done here ([real-microduck.md](../real-microduck.md)): the bridge on the duck, the
API version, clock, home Wi-Fi quality, battery reading, camera over the link, other doors, CPU on
the board.

**Abort.** The duck moves at all (nothing was commanded). A door cannot be closed. `check` fails.
No currents are reported (then no leg-load sense is possible on this robotd: stop and ask Pollen).

**Cleared by** the owner. **Rollback:** stop the bridge; nothing was changed on the duck.

## Stage 1 — legs off the ground

The duck is held by its trunk or stands on a stand with the legs free. Nothing it does here can
make it fall.

### 1a. Head only (10 minutes)

**Entry.** Stage 0 cleared. A person turns the duck on with its own control (write down which
control: this is a checklist item).

**Commands.** Start the page as in [real-microduck.md](../real-microduck.md), *Opening the page on
a real duck*, and add `--neuro-record $SESSION/stage1-neuro.jsonl`. The *real duck* card must say
that the duck moves only its head because its body has not been measured (no calibration yet: this
is expected, and it is what keeps the legs still). Let it run 10 minutes, then stop the page
(Ctrl-C): the bridge stops and sits the duck.

**Pass.** N6; the head stays inside its range (no servo warning in `robotctl health` afterwards).
This is also where the checklist's **head profile** and **the page in real-duck mode** (the drawn
pose matches the duck; the eye window shows its camera) are checked.

**Abort.** A leg moves. The head hits a stop or a cable. A servo warning.

### 1b. Legs free in the air (the bench recording, stage 1)

**Entry.** 1a passed. The page is stopped (the bridge serves one computer program at a time).

**Commands.**

```sh
python -m duckmind.bringup record $URL --out $SESSION --stages 1 --drive --unit "<your label>"
```

The tool tells the person what to do before each recording and waits for Enter: hold it seated in
the air (8 s), standing in the air (10 s), then the legs **step in the air** for 4 s and stop. With
`--drive` the tool sends the sit/stand toggle and that one slow walk command itself, and nothing
else. Without `--drive` it only reads (and ends with a stop), and the person changes the posture
with the duck's own control.

Then the deadman, three times: start the same command again, and while the legs step in the air
cut the link (switch the computer's Wi-Fi off, or `kill -9` the tool). The bridge must stop and sit
the duck by itself.

**Record.** `SESSION/recording.jsonl.gz` (the tool appends to it at every stage), the events file,
a phone video of the three cuts.

**Pass.** All three recordings were taken (none voided); N4; N5 (3 of 3). Write down whether
robotd's own pickup pause came on (the policy reads `held`): that is Pollen's detector, and the
report says so later.

**Abort.** A joint hits a hard stop, a cable snags, a servo warning, the legs do not stop after a
cut.

**Cleared by** the owner. **Rollback:** stage 0.

## Stage 2 — on the floor: stand, sit, be picked up

**Entry.** Stage 1 cleared. A clear, level floor of at least 1 m × 1 m, no edges, a hand next to
the duck. N3.

**Commands.**

```sh
python -m duckmind.bringup record $URL --out $SESSION --stages 2 --drive
python -m duckmind.bringup fit $SESSION          # no --save yet
```

The recording: seated on the floor (8 s), standing (12 s), five sit–stand cycles, then three
pickups standing and three seated (the tool says LIFT and SET DOWN; lift about 20 cm, hold still,
put it back), then the ToF: a flat board (a book) against the duck's face, then 30, 60, 100, 150,
200 and 300 mm in front of the sensor (use a ruler; 3 s each).

`fit` writes `SESSION/report.md` and `SESSION/fit.json`. At this stage it must say **NOT YET** for
one reason only: *the segment walk was not recorded*.

**Record.** The recording, `report.md`, `fit.json`, video of the cycles and the pickups.

**Pass.** N7, N8, N9, N11, N4. Read the report's section *What this body's leg load cannot tell
apart*: each line names a claim that will not work on this duck and what covers it instead.

**Abort.** A fall in a sit–stand cycle (stand it up by hand; one repeat is allowed after looking for
the cause, a second fall ends the stage). The report says standing in the hand cannot be told from
standing on the floor (N8): then the brain must not walk this duck on leg load, and the stage ends
with that finding for the owner.

**Cleared by** the owner. **Rollback:** stage 1.

## Stage 3 — supervised walk in an empty 1.5 m square

A square of 1.5 m × 1.5 m bounded by boards at least 15 cm high (the ToF must see them), nothing
inside, a hand next to the duck the whole time. **The owner decides whether the duck may walk at
all in the first session** (see *Open questions*); without that decision the bring-up ends at
stage 2.

### 3a. The bench walk and the calibration

```sh
python -m duckmind.bringup record $URL --out $SESSION --stages 3 --drive
python -m duckmind.bringup fit $SESSION --save
```

Three walks: forward at the slowest walk for 4 s (about 0.6 m), stop, 3 s. `fit --save` installs the
calibration only when everything passes, and prints its path.

**Pass.** N10, and the report's verdict is **PASS**. `python -m duckmind.bringup show $URL` now
prints the calibration.

### 3b. Floor watch (10 minutes, no brain)

```sh
python -m duckmind.bringup watch $URL --out $SESSION --minutes 10
```

The duck stands and sits on the floor (change it with its own control a few times), nobody touches
it. **Pass.** N12.

### 3c. The deadman on the floor (5 cuts)

Start `record --stages 3 --drive` again into a **new** folder (`--out $SESSION/deadman`) and cut
the link during each walk, five times. **Pass.** N5 (5 of 5), no fall.

### 3d. The brain walks it (15 minutes)

Start the page with `--neuro-record $SESSION/stage3-neuro.jsonl`. The *real duck* card must say
"connected" without the head-only line. Let it roam in the square for 15 minutes. Then pick it up
while it walks, five times (N14).

**Record.** The neuro record, the events file, video of the whole 15 minutes, the stage log with
the counts (every time the page says "picked up" with no hand on it is a false pickup).

**Pass.** N13, N14, N4.

**Abort.** A fall. Two approaches to the square's edge that end in pushing. More than two false
pickups (it sits down each time). Battery under 30 %. A second link drop. Anything that needs the
hand.

**Cleared by** the owner. **Rollback:** delete the calibration file (rule 6): the brain is
head-only again. Then stage 2.

## Stage 4 — free roam in one cleared room

**Entry.** Stage 3 cleared on two different days. One room, door closed, floor cleared: no cables,
no stairs or edges, no water, no pets or small children, nothing lower than the ToF can see that
the duck could climb (a sill of 20 mm can tip it: a known, accepted limit). A person stays in the
room. Charging and the dock are off on a real duck (it has no charging signal); it ends a session
seated and asking, and a person swaps or charges the pack.

**Commands.** The page as in stage 3d, three sessions of 30 minutes, each with its own
`--neuro-record` file. Between sessions: `python -m duckmind.remotebody check $URL` (battery, link).

**Record.** As stage 3d, per session.

**Pass.** N15, N4.

**Abort.** As stage 3, and: a fall it does not get up from; the same thing pushed on twice.

**Cleared by** the owner. Leaving the duck alone in a room is the hatch decision. It is **not** part
of this ladder: it comes after two weeks of stage 4 without an abort (the owner's rule: nest first,
then hatch).

**Rollback:** stage 3 (the square), or rule 6.

## When to measure again

The brain checks the calibration against the duck at every start. It goes back to head only by
itself, and the card says why, when:

- the duck's walk or sit-stand network changed (an update: the file is for other networks);
- the board changed (another board version in `robotd.toml`);
- the file is missing or cannot be read.

Measure again as well after a servo or a foot was replaced, or when stage 4 shows false pickups
above N15: the loads have moved. A new measurement is stages 1b, 2 and 3a in a new session folder
(about 15 minutes).

## What the bench tool does, exactly

`python -m duckmind.bringup` (in `src/duckmind/bringup/`):

| Command | What it does | Sends to the duck |
|---|---|---|
| `show URL` | Which calibration this computer has for that duck, or why none | nothing |
| `watch URL --out DIR --minutes M` | Reads `robot.state` for M minutes; says what the body sense would claim with that duck's calibration, the battery every 30 s, the link's numbers | nothing |
| `record URL --out DIR --stages …` | The guided recording | a zero twist and `robot.stop` when it ends, always; with `--drive` also `sit_toggle` and one forward walk at 0.30 for 4 s per walking segment |
| `fit DIR [--save]` | Fits the bands, replays the recording through the body sense, writes the report; `--save` installs the calibration if it passes | (no link) |
| `calibrate URL --out DIR` | `record` for all stages, then `fit --save` (for the rehearsal; on a real duck go stage by stage) | as `record` |

The calibration is one JSON document (`duckmind.body-calibration/1`) in `<duckmind
home>/body-calibration/`, named by the board and the first digits of the two networks' sha256. It
holds the leg-load bands, the ToF touch band and where they came from. The simulator's own numbers
ship inside the program and are never used for a real duck.

**Rehearse it first** (no robot needed): start `python -m duckmind.remotebody sim-body --as-real
…`, note the `world_port` in its `remote-body: ready` line, and run

```sh
python -m duckmind.bringup calibrate remote://127.0.0.1:8797 --out $SESSION --drive --no-tof \
    --sim-person <world_port>
```

The simulated world's hand plays the person. A rehearsal is held head-only like a real duck until
this has been done.

## Open questions for the owner

1. **May the duck walk in the first session, or head only?** Without a yes the bring-up ends at
   stage 2 (everything up to there is measured without a single step on the floor).
2. **v8 or v9 on the real duck?** The factory image is moving to v9; the simulator's numbers are
   best on v8. The calibration is per network, so the choice must be made before stage 1b, and a
   later change means measuring again (15 minutes).
3. **Who is the second person?** The emergency stop is a hand.
4. **One duck or more?** The calibration is keyed by board version and networks, not by the single
   duck (the bridge does not report a serial number). With two ducks of the same board each needs
   its own computer account or its own duckmind home until that is added.

## Not covered here

- Go-there by pointing at the floor: refused on a real duck for now.
- The dock and charging on a real duck, and the bridge refusing to start while mediad's doors are
  open: separate work; this page will link them when they land.
- The microphone and touch: no daemon serves the microphone, and the duck has no touch sensor.
