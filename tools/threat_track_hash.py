#!/usr/bin/env python3
"""A hash of the integrated brain's motor commands, tick for tick, over SynthWorld runs.

Used to prove that carrying the source track on a ``threat`` / remembered ``place`` target
(fix/threat-track) is telemetry only: the same brain, the same world, the same seed give the
same motor commands before and after. The fields that are allowed to differ are the telemetry
the changes add: ``Target.track`` of a ``threat`` or ``place`` target (set to 0 for those two
kinds; ``thing`` / ``person`` / ``nest`` tracks are hashed), ``MotorCommand.reflex_track`` and
``MotorCommand.panic`` (the flag that lets the body link run a flight as the panic skill, 2026-10-09: it changes
no command) and ``MotorCommand.pivot`` (the same for the quick turn); all left out of the hash.

  PYTHONPATH=src python tools/threat_track_hash.py --out hashes.json [--minutes 30] [--workers 4]

4 temperaments (duckmind.verify.temperaments) x seeds 1,2 x 30 min at 10 Hz, in a world with a
family member, plus scheduled looms, claps, bangs, name calls and pokes so that startles, flees
and retreats happen. Each run is its own process (building a brain installs the skills and
leaves global state behind). The output also counts the threat and place targets a run made and
how many of them carry a track (all 0 before the fix).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any

PRESETS = ("bold_explorer", "shy_snuggler", "lazy_dreamer", "social_songbird")
SEEDS = (1, 2)
DT_MS = 100


def schedule(world: Any, seed: int, seconds: float) -> None:
    """The same stimuli every time for a seed: a few of each kind that can scare the duck."""
    off = 7 * seed
    t = 40.0 + off
    while t < seconds:
        world.loom(int(t * 1000))
        t += 110.0
    t = 55.0 + off
    while t < seconds:
        world.clap(int(t * 1000))
        t += 143.0
    t = 100.0 + off
    while t < seconds:
        world.bang(int(t * 1000))
        t += 227.0
    t = 130.0 + off
    while t < seconds:
        world.call_name(int(t * 1000))
        t += 199.0
    t = 85.0 + off
    flip = 0
    while t < seconds:
        world.poke(int(t * 1000), "left" if flip % 2 == 0 else "right")
        flip += 1
        t += 167.0


def cmd_key(cmd: Any) -> str:
    """The command as text, with the track of a threat or place target set to 0 and ``reflex_track`` left out."""
    d = dataclasses.asdict(cmd)
    d.pop("reflex_track", None)
    d.pop("panic", None)
    d.pop("pivot", None)
    tgt = d.get("target")
    if tgt is not None and tgt.get("kind") in ("threat", "place"):
        tgt["track"] = 0
    return repr(d)


def run_one(task: tuple[str, int, float]) -> dict[str, Any]:
    preset, seed, minutes = task
    from duckmind.neuro.integration import install_default_skills, make_brain
    from duckmind.neuro.synth import SynthWorld
    from duckmind.verify.temperaments import genome

    install_default_skills()
    seconds = minutes * 60.0
    world = SynthWorld(seed=seed, family=True)
    schedule(world, seed, seconds)
    brain = make_brain(genome(preset, seed), seed)
    h = hashlib.sha256()
    n = int(seconds * 1000 / DT_MS)
    count = {
        "threat": 0,
        "threat_track": 0,
        "place": 0,
        "place_track": 0,
        "thing": 0,
        "thing_track": 0,
        "flee_retreat": 0,
        "reflex": 0,
        "reflex_track": 0,
    }
    threat_tracks: dict[int, int] = {}
    place_tracks: dict[int, int] = {}
    reflex_tracks: dict[str, int] = {}  # "reflex:track" -> ticks
    t0 = time.time()
    for k in range(n):
        t_ms = k * DT_MS
        frame = world.frame(t_ms)
        cmd, _tel = brain.step(t_ms, frame)
        h.update(cmd_key(cmd).encode())
        tgt = cmd.target
        if tgt is not None and tgt.kind in count:
            count[tgt.kind] += 1
            if tgt.track:
                count[f"{tgt.kind}_track"] += 1
                if tgt.kind == "threat":
                    threat_tracks[tgt.track] = threat_tracks.get(tgt.track, 0) + 1
                elif tgt.kind == "place":
                    place_tracks[tgt.track] = place_tracks.get(tgt.track, 0) + 1
        if cmd.behaviour in ("flee", "retreat"):
            count["flee_retreat"] += 1
        if cmd.reflex is not None:
            count["reflex"] += 1
            track = getattr(cmd, "reflex_track", 0)
            if track:
                count["reflex_track"] += 1
                key = f"{cmd.reflex}:{track}"
                reflex_tracks[key] = reflex_tracks.get(key, 0) + 1
        world.apply(cmd, DT_MS / 1000.0)
    return {
        "preset": preset,
        "seed": seed,
        "ticks": n,
        "sha256": h.hexdigest(),
        "counts": count,
        "threat_tracks": {str(k): v for k, v in sorted(threat_tracks.items())},
        "place_tracks": {str(k): v for k, v in sorted(place_tracks.items())},
        "reflex_tracks": dict(sorted(reflex_tracks.items())),
        "wall_s": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--presets", default=",".join(PRESETS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    args = ap.parse_args()
    tasks = [
        (p, int(s), args.minutes) for p in args.presets.split(",") for s in args.seeds.split(",")
    ]
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(args.workers, len(tasks)), maxtasksperchild=1) as pool:
        runs = pool.map(run_one, tasks, chunksize=1)
    runs.sort(key=lambda r: (r["preset"], r["seed"]))
    overall = hashlib.sha256("".join(r["sha256"] for r in runs).encode()).hexdigest()
    out = {"minutes": args.minutes, "overall_sha256": overall, "runs": runs}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    for r in runs:
        print(r["preset"], r["seed"], r["sha256"][:16], r["counts"], f"{r['wall_s']} s")
    print("overall", overall)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    raise SystemExit(main())
