#!/usr/bin/env python3
"""LV4: the new drives change nothing about the duck's other behaviour when no one talks.

With the bridge (and so the two drives) attached to an integrated brain and no cloud proposal ever made:

* **exact**: in SynthWorld with no family (nobody is ever in view, nothing is ever said) the commands
  are, tick for tick, the commands of the same duck with no bridge at all;
* **the L1 lives time-budget profile with no family** (the world of ``tools/machine_verify/run_l1.py``:
  nobody talks, comes or touches; here two hours at 5 Hz and ten claps): the share of time in each
  behaviour with the bridge is the share without it, per temperament and seed, to the digit;
* **the same two hours with the world's own family** (``tests/test_neuro_brain.py``'s world: a person who
  comes and goes, talks, calls the name and strokes): here the duck *is* talked to, so the drives may
  act (it may start a conversation, go to a person it is close to). This is reported, not required to be
  zero: the distance between the shares with and without the bridge, against how far two seeds of the
  same duck are from each other (the noise), and the temperaments test's own claims (the bold explorer
  walks more than the lazy dreamer; between-temperament distances beat within; each one's signature act
  shows) with and without.

The temperaments test itself (``tests/test_neuro_brain.py``) builds a ``NeuroBrain`` with no bridge and is
run as it is.

Run:  PYTHONPATH=src python tools/live_voice/lv4_profile.py --params best.json --out profile.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from duckmind.live.bridge import BridgeConfig, LiveBridge
from duckmind.live.drives import DriveParams
from duckmind.neuro.integration import install_default_skills, make_brain
from duckmind.neuro.synth import SynthWorld, iterate
from duckmind.neuro.telemetry import Budget
from duckmind.verify.temperaments import PRESET_NAMES, genome

SIGNATURE = {"bold_explorer": "survey", "shy_snuggler": "melt", "lazy_dreamer": "moon_gaze"}


def _distance(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def exact(args: tuple[str, int, dict[str, float]]) -> dict[str, Any]:
    preset, seed, values = args
    install_default_skills()

    def trace(attach: bool) -> list[tuple[Any, ...]]:
        brain = make_brain(genome(preset, seed), seed)
        if attach:
            cfg = BridgeConfig(drives=DriveParams.from_dict(values))
            LiveBridge(None, config=cfg).attach(brain)
        world = SynthWorld(seed=5 + seed, family=False)
        world.clap(30_000)
        out = []
        for cmd, _ in iterate(brain, world, 300.0, hz=10.0):
            out.append((cmd.behaviour, cmd.bout, cmd.vx, cmd.vyaw, cmd.vocal, cmd.reflex))
        return out

    a, b = trace(False), trace(True)
    return {"preset": preset, "seed": seed, "ticks": len(a), "identical": a == b}


def budget(args: tuple[str, int, bool, dict[str, float], bool]) -> dict[str, Any]:
    preset, seed, attach, values, family = args
    install_default_skills()
    world = SynthWorld(seed=100 + seed, family=family)
    for k in range(10):
        world.clap(600_000 + k * 600_000)
    brain = make_brain(genome(preset, seed), seed)
    if attach:
        LiveBridge(None, config=BridgeConfig(drives=DriveParams.from_dict(values))).attach(brain)
    b = Budget()
    dt = 1.0 / 5.0
    for cmd, tel in iterate(brain, world, 7200.0, hz=5.0):
        b.add(cmd, tel, dt)
    extra: dict[str, Any] = {}
    if attach and brain.live is not None:
        extra = {"drives": brain.live.view().get("drives"), "voiced": brain.live.view()["voiced"]}
    return {
        "preset": preset,
        "seed": seed,
        "attach": attach,
        "family": family,
        "shares": b.shares(),
        "walk": b.walk_s / b.total_s,
        "time_s": dict(b.time_s),
        "calls": b.calls,
        **extra,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="1,2,3,4")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    values = json.loads(Path(args.params).read_text())["params"]
    seeds = [int(s) for s in args.seeds.split(",")]
    t0 = time.time()
    tasks = [
        (p, s, a, values, fam)
        for fam in (False, True)
        for p in PRESET_NAMES
        for s in seeds
        for a in (False, True)
    ]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        exacts = pool.map(exact, [(p, s, values) for p in PRESET_NAMES for s in seeds[:2]], 1)
        runs = pool.map(budget, tasks, 1)
    by = {(r["preset"], r["seed"], r["attach"], r["family"]): r for r in runs}

    def table(family: bool) -> list[dict[str, Any]]:
        rows = []
        for p in PRESET_NAMES:
            within = [
                _distance(by[(p, a, False, family)]["shares"], by[(p, b, False, family)]["shares"])
                for i, a in enumerate(seeds)
                for b in seeds[i + 1 :]
            ]
            moved = [
                _distance(by[(p, s, False, family)]["shares"], by[(p, s, True, family)]["shares"])
                for s in seeds
            ]
            rows.append(
                {
                    "preset": p,
                    "seed_to_seed_mean": round(statistics.fmean(within), 4),
                    "with_vs_without_mean": round(statistics.fmean(moved), 4),
                    "with_vs_without_max": round(max(moved), 4),
                    "mean_within_noise": statistics.fmean(moved) <= statistics.fmean(within),
                }
            )
        return rows

    def claims(family: bool) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for attach in (False, True):
            walk = {
                p: statistics.fmean(by[(p, s, attach, family)]["walk"] for s in seeds)
                for p in PRESET_NAMES
            }
            first, second = seeds[0], seeds[1]
            between, within_t = [], []
            for i, a in enumerate(PRESET_NAMES):
                sh = lambda p, s, at=attach: by[(p, s, at, family)]["shares"]  # noqa: E731
                within_t.append(_distance(sh(a, first), sh(a, second)))
                for b in PRESET_NAMES[i + 1 :]:
                    between.append(_distance(sh(a, first), sh(b, first)))
            out["with" if attach else "without"] = {
                "bold_over_lazy_walk_ratio": round(walk["bold_explorer"] / walk["lazy_dreamer"], 4),
                "min_walk_share": round(min(walk.values()), 4),
                "between_min_over_within_max": round(min(between) / max(within_t), 3),
                "signature_acts_all_seeds": {
                    p: all(verb in by[(p, s, attach, family)]["time_s"] for s in seeds)
                    for p, verb in SIGNATURE.items()
                },
                "walk": {p: round(v, 4) for p, v in walk.items()},
            }
        return out

    no_family = table(False)
    report = {
        "exact": exacts,
        "no_family": no_family,
        "no_family_claims": claims(False),
        "family": table(True),
        "family_claims": claims(True),
        "family_drives": {
            f"{p}/{s}": by[(p, s, True, True)].get("drives")
            for p in PRESET_NAMES
            for s in seeds[:1]
        },
        "wall_s": round(time.time() - t0, 1),
    }
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "family_drives"}, indent=1))
    ok = all(e["identical"] for e in exacts) and all(
        r["with_vs_without_max"] == 0.0 for r in no_family
    )
    print("NOTHING MOVES WHEN NO ONE TALKS" if ok else "SOMETHING MOVED WITH NO ONE TALKING")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
