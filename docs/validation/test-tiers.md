# Test tiers: what runs when, and how long it takes

The whole suite is 6,481 tests in 391 files (collected with the `sim` extra, 2026-11-10). It took
28.2, 29.2 and 30.8 minutes in three whole-suite runs on the owner's Mac (2026-11-09/10, at
6,433 to 6,477 tests). That is right before a merge and too long after every edit, so the tests
carry markers, and the tests without `heavy` are the **fast tier**:

```bash
uv run --python 3.12 --extra dev pytest -m "not heavy"
```

A plain `pytest` is still every test: the markers take nothing out of the suite, they only say
what each test is. `tests/tiers.py` holds the tables, `tests/conftest.py` puts the markers on
while pytest collects, and `tests/test_tiers.py` checks both.

## The markers

Counted by collecting the suite once (2026-11-10; parametrized cases count one each):

| Marker | Tests | Files | What it means |
|---|---:|---:|---|
| (all) | 6,481 | 391 | |
| `heavy` | 1,619 | 166 | `render`, `physics`, and the measured-slow tests |
| `physics` | 475 | 57 | steps or compiles MuJoCo; skips without the `sim` extra and the duck model |
| `render` | 752 | 37 | every test of conftest's `RENDER_FILES` (numpy, MuJoCo's offscreen render) |
| `node` | 104 | 20 | runs a page script in Node; skips without Node |
| `subprocess` | 306 | 49 | starts a child process (a child pytest, a CLI, a server) |
| `timing` | 115 | 29 | sleeps or polls on the wall clock: the first to fail on a busy machine |
| not `heavy` (the fast tier) | 4,862 | 338 | of them `node` 83, `subprocess` 167, `timing` 49 |

`physics` or `render`: 883 tests. Heavy only because they are measured slow:
736 tests.

A test is **measured slow** when its slowest case took 0.5 s or more, or its cases together took
1 s or more, in the median of the three whole-suite runs above; a test that shares a slow test's
cached run (a module-scoped fixture, an `lru_cache`) is slow with it, because whichever of them
runs first pays for the run. The child-process suite runners (`test_live_isolated.py`,
`test_life_isolated.py`, `test_live_page_isolated.py`, `test_machine_verify_v3_runner.py`) are
measured slow like any other test.

## The fast tier, measured

On the owner's Mac (Apple silicon), Python 3.12, the development environment without MuJoCo,
`python -m pytest -q -p no:cacheprovider -m "not heavy"`, wall time of the whole command:

| Run | Tests | Skipped | Failed | Wall time (s) | 1-min load at the start |
|---:|---:|---:|---:|---:|---|
| 1 | 4875 | 41 | 0 | 109 | 10.3 |
| 2 | 4875 | 41 | 0 | 111 | 9.0 |
| 3 | 4875 | 41 | 0 | 118 | 11.3 |

Three runs, 109 to 118 s; the limit the tier is held to is 180 s. The Mac was busy during all
three (other simulations were running: a 1-minute load of 9 to 11), and the tier was not timed
on a quiet machine. Two earlier runs of nearly the same set took 105 s (load 7) and 126 s (load
25). The count is the 4,862 tests of the tier and 13 test files that are skipped whole when
they are collected (they need MuJoCo, torch or another optional package to import; a file
skipped whole has no tests to leave out). Of the 28 tests that skip, 23 lack an optional
package of the development environment (scipy, torch, pypinyin, soundfile), three are the
opt-in network tests and two say why themselves; none of them is a test that needs MuJoCo.

When the fast tier ends it names its own tests that took 2 s or more on that run. If one is
that slow on a quiet machine, add it to `SLOW` in `tests/tiers.py`.

## The 30 slowest test files

Seconds per file, summed over its tests (setup and teardown included), median and range of the
three whole-suite runs. Together they are 1,322 s of the suite's 1,735 s of test time.

| # | File | Tests | Median (s) | Range (s) |
|---:|---|---:|---:|---|
| 1 | `test_neuro_brain.py` | 38 | 247 | 245–249 |
| 2 | `test_neuro_skill_shelter.py` | 9 | 196 | 187–206 |
| 3 | `test_live_isolated.py` | 1 | 94 | 93–96 |
| 4 | `test_life_isolated.py` | 1 | 78 | 77–80 |
| 5 | `test_neuro_righting_lift.py` | 19 | 63 | 62–64 |
| 6 | `test_training.py` | 75 | 47 | 43–51 |
| 7 | `test_deploy.py` | 43 | 46 | 37–77 |
| 8 | `test_neuro_skill_imprint.py` | 13 | 46 | 46–46 |
| 9 | `test_duckcore_motion.py` | 27 | 45 | 43–48 |
| 10 | `test_battery_help.py` | 33 | 45 | 38–48 |
| 11 | `test_duckcore_moments.py` | 11 | 42 | 41–58 |
| 12 | `test_golden_corpus.py` | 15 | 37 | 35–39 |
| 13 | `test_lv7_stranger.py` | 2 | 32 | 31–32 |
| 14 | `test_heap_settle.py` | 3 | 29 | 28–35 |
| 15 | `test_duckcore_excursion.py` | 10 | 25 | 24–37 |
| 16 | `test_simweb.py` | 34 | 23 | 23–23 |
| 17 | `test_neuro_unstuck.py` | 37 | 22 | 22–23 |
| 18 | `test_remotebody.py` | 24 | 21 | 17–21 |
| 19 | `test_neuro_dock_home.py` | 10 | 19 | 19–19 |
| 20 | `test_neuro_lift_lock.py` | 10 | 17 | 17–18 |
| 21 | `test_duckcore_personality.py` | 13 | 16 | 15–17 |
| 22 | `test_machine_verify_v5_runner.py` | 15 | 16 | 15–16 |
| 23 | `test_media_frame_protocol.py` | 45 | 16 | 16–16 |
| 24 | `test_machine_verify_v3_runner.py` | 22 | 16 | 15–16 |
| 25 | `test_step_invariance.py` | 32 | 16 | 15–16 |
| 26 | `test_life_false_lift.py` | 27 | 15 | 15–16 |
| 27 | `test_world_hand.py` | 35 | 14 | 14–14 |
| 28 | `test_neuro_boundary.py` | 18 | 13 | 13–13 |
| 29 | `test_duckcore_nest.py` | 21 | 12 | 12–12 |
| 30 | `test_remote_check.py` | 4 | 12 | 12–12 |

## What CI runs

- `lint` and `secrets`: always.
- `fast`: the fast tier, on every push and pull request in a public repository (a private copy
  runs only `lint` and `secrets`; its gate is the local suite). It checks out
  the whole history, because the frozen v4 engine is regenerated from the commit it was frozen
  at and compared byte for byte (`tests/test_engine_frozen.py`; a shallow clone skips that).
- `tests`: the whole suite in four shards, in a public repository only.

No CI job runs the `physics` and `render` tests (883 tests): the runners have no MuJoCo,
no duck model and no offscreen GL context, so those tests skip there. The owner's Mac runs
them before a merge. A self-hosted macOS runner for them would need: Apple silicon, `uv` and
Node 22; the upstream checkouts and policies `python -m duckmind.setup_cmd` fetches, kept on
the runner; the render check's packages (`run-duck-simulator.md`); a logged-in desktop session
(MuJoCo's offscreen context on macOS is CGL: a launchd agent of that user, not a daemon);
about 35 minutes per run and one run at a time; and, for a public repository, no runs for pull
requests from forks. None is set up.
