# The bench: measuring go-there quickly, the same way every time

`python -m duckmind.bench gothere …` is the repository's one tool for measuring the walk to a place somebody pointed
at ("go there"). It replaces the scripts that used to be copied from folder to folder for every comparison. It answers
two kinds of question:

- **How good is this tree?** — a fixed, versioned set of clicks, answered by the real brain on the real physics without
  a server or a clock (*headless*), a few hundred clicks in minutes, the same rows every time.
- **Is tree B better than tree A?** — both answer the same clicks; the comparison is paired click by click, says how
  many clicks the wanted gain needs before anything runs, and stops at the first look that decides it.

A live mode drives a running test server with the same clicks in real time. It is for the last confirmation of a
decision the headless runner has already made; it is too slow to search with.

Nothing in the bench changes the duck. The truth planner (below) reads the world's truth and exists only in
`src/duckmind/bench/`.

## What it needs

The simulator's dependencies (`mujoco==3.10.0`, `numpy`, `onnxruntime`), the upstream checkout with the robot model
(`DUCKMIND_MICRODUCK_MJCF` / `DUCKMIND_UPSTREAM`) and the walk policy set to measure on (`DUCKMIND_POLICY_DIR` or
`--policies`). The body is the live simulated duck's by default: the collision model `allcollisions_backlash`
(`--collision`), the simulator's 20 ms servo command delay, the body link with that walk's measured map, the production
camera's senses (`DUCKMIND_BOARD`, default `beta`), the lasting map in `soft` mode.

```sh
uv run --no-project --python 3.12 --with mujoco==3.10.0 --with numpy --with onnxruntime==1.27.0 \
  python -m duckmind.bench gothere run --set real_home-v1 --blocks 0-3 --out rows/main.jsonl
```

## The click sets

A click is a start pose and a place, with labels worked out from the home's layout (truth, used only for scoring and
tables). The sets are in `src/duckmind/bench/data/`.

| set | clicks | what it is |
|---|---|---|
| `real_home-v1` | 800 (8 blocks of 100: 75 in the start's room, 25 in another) | made once by `gothere clicks` (seed 20261110). The start has 25 cm of room all round and is not on a sill or a cable; the place is 0.8–1.6 m away, has room for the body (12 cm) and is reachable by a body-wide route (10 cm from anything solid) of at most 4 m. Candidates that failed were dropped and counted in `real_home-v1.meta.json`. |
| `legacy-205` | 205 | the clicks of the earlier out-of-repo rig (live clicks replayed), kept so numbers stay comparable. Nothing is dropped; what the new set excludes is flagged: 122 starts in clutter, 11 places on something solid, 69 places with no room for the body, 5 with no body-wide route. |

Labels: `cls` same / other (room); `line` — the first solid thing on the body-wide straight line (clear, low, thing,
furniture, wall; *low* is something a body steps over: a sill, a cable, a sock); `reach`, `path_m`, `detour` (route
over straight line), `clear_start_m`, `clear_point_m`, `clear_path_m`; `flags`.

The generator is deterministic and a shorter set is a prefix of a longer one, so any whole number of blocks is a fair
sample. A changed layout changes the generator's output: make a new version (`--name real_home-v2`), never edit a set.

## The headless runner

Per click: a fresh duck (the brain's seed is `9201 + k mod 7`) is stood at the start pose at 10:00, given the
known-home lasting map (`data/known_home-real_home.json`: the map of the 40-minute roam every known-home measurement
since 2026-11-06 has used, laid where it truly is), sent to the place exactly as the page's click does (a bearing and a
range), and watched for 40 s. Workers run clicks in parallel (`--workers`, default 6); the run waits until the machine
has room (1-minute load ≤ 7 and fewer than two other simulators or batches) unless `--no-wait`. Rows already in the
output file are kept, so an interrupted run continues. Stop a run with Ctrl-C; a parent that is killed outright leaves
its workers to finish the click they are on.

Another home's known-home map is made with `gothere roam --home … --seed … --minutes 40 --out …` (a seeded headless
roam; the same command gives the same map).

### Arms

| arm | what steers the legs during the errand |
|---|---|
| `main` | the brain as it is |
| `truth` | the errand's own steering (face the place, walk at it, the wall memory's bend and cap) is replaced by the truth planner; every reflex, look-before-going, the cornered stop, the escapes, the selector's takeovers, the body link, the walk and the physics stay |
| `truth-notake` | `truth`, and an errand that loses the walk to another act takes it back at once |
| `truth-reflex` | the planner has the legs for the whole errand; only the reflex layer still acts on them |
| `truth-free` | the planner has the legs and nothing of the brain reaches them (the body link's own gates stay) |

The truth planner keeps a body-wide route over the layout's truth with every movable thing where it is now, re-made
when a thing has been pushed, and walks it by the errand's own law (turn on the spot when the next point is well off
the heading, else walk at it). It knows the body link's rules: where the link lets only an arc through (something
within 30 cm ahead) it asks for arcs. It measures what a perfect route would reach on this body — a ceiling, not a
feature.

### A row

One JSON object per click and arm (schema `bench-gothere-v1`):

| field | meaning |
|---|---|
| `set`, `k`, `arm`, `mode`, `tree`, `seed`, `pose`, `point` | which click, how it was run |
| `cls`, `line`, `flagged`, `detour` | the click's labels |
| `state`, `reason` | the errand's end: arrived, gave_up (and why), walking (out of time) |
| `arrived`, `arrived_s` | **the measure**: the brain said it arrived within 40 s (in `truth-reflex` / `truth-free`: the body was truly within 18 cm of the place for 0.3 s) |
| `truth_ok`, `truth_end`, `near`, `truth_in_s` | truth: arrived *and* ended within 25 cm; the distance at the end; the nearest it got; when it was first truly there |
| `falls`, `bumps`, `brushes`, `startles` | falls (sampled twice a second), bump-stop reflexes plus the link's blocked stops, the link's brushes, startles |
| `gave_up`, `takeovers`, `escapes`, `clutter_started`, `ph` | gave up; times another act took the walk; unstick and clutter escapes; seconds by what had the legs |
| `rt_min` | (live) the lowest realtime factor during the click |
| `plan` | (truth arms) route re-makes, back-offs, the share of ticks with no route |

## Reading a result

`gothere report rows.jsonl` prints one arm: arrivals with a 95 % Wilson interval for all clicks, the same room, another
room, the same room by what is on the line, and the same room without flagged clicks; then falls, bumps, brushes,
give-ups, clicks still walking at 40 s, takeovers. `gothere report a.jsonl b.jsonl` adds the paired table: both
arrived, only A, only B, neither, B − A with its interval and the exact McNemar p-value; and, click by click, on how
many clicks B fell, bumped, brushed or gave up more or less often (exact sign test).

The same-room row is the one every go-there decision has used. A difference is real when the paired interval excludes
zero; an unpaired comparison of two rates wastes most of the clicks.

## Comparing two trees

```sh
python -m duckmind.bench gothere compare --a main --b my-branch --out cmp/my-branch --blocks 0-7 --looks 4 --delta 5
```

`--a` / `--b` are git refs (their `src` is exported into the comparison's folder) or snapshot folders. Each tree
answers in its own interpreter with its own `src` first on `PYTHONPATH`; the bench that asks and scores is always the
one that was started, so only the duck differs — the tree may be older than the bench.

With `--looks 4` the blocks are answered in four equal parts. After each part the same-room difference gets a repeated
confidence interval (Pocock's boundary: the four intervals hold together with 95 % confidence), and the comparison
stops as soon as it is **better** (interval above 0), **worse** (below 0) or shows **no gain** of `--delta` points
(interval below `--delta`); at the last look it is otherwise **undecided**. `RESULT.md` in the folder has the tables.

First use (2026-11-10): `main` against the camera go-round that had been reverted after a live measurement of hours
could not decide (63 % against 57 %). The comparison stopped at its first look — 150 same-room pairs, about ten
minutes — with **no gain** (−2.7 points, repeated interval −7.3 to +2.0; only 5 % of the clicks ended differently).
The block answered from the exported `main` ref was row for row the one the working tree gives.

`gothere power` prints how many same-room clicks a paired comparison needs for +5, +9 and +15 points (two-sided 5 %,
power 80 %), by the share of clicks that end differently between the two arms. That share is printed by every paired
report; use the one measured on a comparison like the one planned.

## The live mode

```sh
python -m duckmind.bench gothere run --live http://127.0.0.1:8781 --set real_home-v1 --blocks 0 \
  --out rows/live.jsonl --rec server-record.jsonl
```

Start a test server with a copy of the known-home life first (never the live duck's). A live duck cannot be put at a
click's start pose — its lasting map would no longer know where it is — so before each click the duck's true pose is
read and the unused click with the nearest start is asked (within 35 cm, with 20 cm of room); when none is near, the
duck is first sent to the nearest unused start by an errand that is not scored. The row is labelled again from where
the duck really stood. The scene is checked before every click (daytime, battery, not afraid, not fallen, nobody
holding it), and a click during which the body's realtime factor fell under 0.95 goes to `<out>.discarded.jsonl` and
is asked again. Without `--rec` (the server's `--neuro-record` file) brushes and the link's blocked stops are not
visible and stay empty. A live click takes its real time: about 100 clicks an hour.

## How long it takes, and how small a difference it sees

Measured on the development Mac on 2026-11-10 with three workers at nice 10, beside the live simulated duck (whose
realtime factor stayed at 0.998 or above): 100 clicks of `main` take about 3.0 minutes, of `truth` about 2.6; the whole
800-click set about 23 minutes. Six workers are about twice as fast. A live click takes its 40 real seconds.

Same-room clicks a paired comparison needs (two-sided 5 %, power 80 %):

| clicks that end differently between the arms | +5 points | +9 points | +15 points |
|---|---|---|---|
| 10 % | 312 | 95 | 50 |
| 18 % (measured: `main` against `truth`) | 563 | 173 | 61 |
| 25 % | 783 | 240 | 85 |
| 35 % | 1097 | 337 | 120 |
| not paired, from 72 % (each arm) | 1192 | 348 | 113 |

The 600 same-room clicks of `real_home-v1` show +4.8 points when 18 % of the clicks differ. The live measurements this
tool replaces had about 110 unpaired same-room clicks an arm, took two to three hours and could show about +17 points.

## Validation (2026-11-10)

- **Continuity.** `main` on `legacy-205`: 112/205 arrived, the same room 79/123 = 64 % (55–72) — inside the earlier
  rig's interval on the same clicks (75/123 = 61 %, 52–69; another room 33/82 on both) and overlapping the three live
  measurements together (170/280 = 61 %, 55–66).
- **Repeatability.** The same command twice gives byte-identical row files (205 of 205 rows; 100 of 100 and 20 of 20
  again for chunks that had first been run on a machine at a load above 100 with a different number of workers).
- **A fair set reads higher.** On `real_home-v1` `main` arrives on 433/600 = 72 % (68–76) of the same-room clicks:
  much of the old 61 % was starts in clutter and places with no room for the body.
- **The ceiling.** `truth` 512/600 = 85 % (82–88) in the same room; `truth-reflex` 288/300 = 96 % (93–98);
  `truth-free` 290/300 = 97 % (94–98). The roam command gave byte-identical fixtures for the same seed.

## Tests

`tests/test_bench_clicks.py` (the generator gives the committed set; fair questions; the body-wide route),
`tests/test_bench_stats.py` (known values; the stopping rule keeps its error rate over four looks),
`tests/test_bench_rows.py` (pairing, tables, looks, trees, the live row and its realtime gate) run without a simulator.
`tests/test_bench_headless_smoke.py` runs one click on the real physics (heavy).
