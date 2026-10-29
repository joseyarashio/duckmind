#!/usr/bin/env python3
"""LV4: fit the new drives' parameters against the frozen targets (docs/design/live-voice/LV4-TARGETS.md).

``search``: CMA-ES (duckmind.live.cmaes) over duckmind.live.drives.SEARCH_SPACE (the new drives'
parameters only), each candidate lived through the scripted conversation day (duckmind.live.convday) as
the four presets on the *training* seeds, scored by convday.search_loss (a sum of squared hinge
violations of the targets, each a margin inside its threshold). The held-out seeds are not touched here.

``evaluate``: one parameter set on a set of seeds (the held-out ones for the final check): every day and
every night run, the seven targets with PASS/FAIL, the numbers.

Run (no physics, no network, plain Python; one process per core):
  PYTHONPATH=src python tools/live_voice/lv4_search.py search --out DIR --seeds 1-6 --workers 10
  PYTHONPATH=src python tools/live_voice/lv4_search.py search --out DIR --resume
  PYTHONPATH=src python tools/live_voice/lv4_search.py evaluate --params DIR/best.json --seeds 13-28 \
      --out DIR/heldout.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from duckmind.live import convday as D
from duckmind.live.cmaes import CMAES
from duckmind.live.drives import (
    SEARCH_SPACE,
    DriveParams,
    params_from_vector,
    vector_from_params,
)

# the fixed part of every candidate: what is not searched (the satiation by a tag, the approach wish)
BASE = DriveParams()
# where the search starts: LV2's weights with a middling stranger reserve, hesitation, a short reply
# and a modest wish to start a conversation (the search is free to leave all of it)
START = DriveParams.neutral().with_values(
    approach_gain=BASE.approach_gain,
    talk_rise=0.01,
    talk_rest=0.3,
    stranger_floor=0.5,
    bond_full=0.5,
    hesitate_s=0.6,
    hesitate_tempo_k=2.0,
    continue_wish=0.2,
    init_gain=0.4,
    init_threshold=0.5,
    reply_act_k=0.5,
    reply_soc_k=0.5,
    bond_g_name=0.12,
    bond_g_talk=0.03,
    bond_g_pet=0.02,
    bond_g_intro=0.1,
    bond_l_startle=0.4,
)
OUT_OF_BOX = 5.0  # the penalty weight for how far a point lies outside the unit cube


def parse_seeds(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def _run(task: tuple[int, list[float], str, int, str]) -> tuple[int, dict[str, Any]]:
    idx, vec, preset, seed, kind = task
    params = params_from_vector(vec, BASE)
    return idx, D.run_day(preset, seed, params, kind=kind).to_dict()


def _clip(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def evaluate_batch(
    pool: Any, xs: np.ndarray, seeds: list[int], margin: float
) -> list[tuple[float, dict[str, float]]]:
    """The loss of each candidate (the rows of ``xs``) on ``seeds``, all presets, day runs."""
    tasks = [
        (i, [float(v) for v in _clip(x)], preset, seed, "day")
        for i, x in enumerate(xs)
        for preset in D.PRESETS
        for seed in seeds
    ]
    by: dict[int, list[dict[str, Any]]] = {}
    for idx, res in pool.imap_unordered(_run, tasks, chunksize=1):
        by.setdefault(idx, []).append(res)
    out = []
    for i, x in enumerate(xs):
        loss, terms = D.search_loss(by[i], margin=margin)
        loss += OUT_OF_BOX * float(np.sum((x - _clip(x)) ** 2))
        out.append((loss, terms))
    return out


def cmd_search(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    checkpoint = out / "cma.pkl"
    meta_path = out / "meta.json"
    if args.start:
        start = DriveParams.from_dict(json.loads(Path(args.start).read_text())["params"])
    else:
        start = START
    x0 = np.array(vector_from_params(start))
    cma = CMAES(x0, args.sigma, popsize=args.pop, seed=args.seed)
    if args.resume and checkpoint.exists():
        cma.restore(pickle.loads(checkpoint.read_bytes()))
        print(f"resumed at generation {cma.generation}", flush=True)
    meta_path.write_text(
        json.dumps(
            {
                "seeds": seeds,
                "pop": cma.lam,
                "sigma0": args.sigma,
                "margin": args.margin,
                "dim": len(SEARCH_SPACE),
                "space": [s.name for s in SEARCH_SPACE],
                "workers": args.workers,
                "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=1,
        )
    )
    best_loss = cma.best_f
    t_start = time.time()
    with (
        mp.get_context("spawn").Pool(args.workers) as pool,
        (out / "history.jsonl").open("a") as log,
    ):
        while cma.generation < args.gens:
            t0 = time.time()
            xs = cma.ask()
            scored = evaluate_batch(pool, xs, seeds, args.margin)
            losses = [s[0] for s in scored]
            cma.tell(losses)
            k = int(np.argmin(losses))
            row = {
                "gen": cma.generation,
                "wall_s": round(time.time() - t0, 1),
                "best": round(losses[k], 5),
                "median": round(float(statistics.median(losses)), 5),
                "sigma": round(cma.sigma, 4),
                "terms": {n: round(v, 4) for n, v in scored[k][1].items() if v > 1e-6},
            }
            log.write(json.dumps(row) + "\n")
            log.flush()
            if losses[k] < best_loss - 1e-12:
                best_loss = losses[k]
                vec = [float(v) for v in _clip(xs[k])]
                params = params_from_vector(vec, BASE)
                (out / "best.json").write_text(
                    json.dumps(
                        {
                            "loss": best_loss,
                            "gen": cma.generation,
                            "vector": vec,
                            "params": params.as_dict(),
                            "terms": scored[k][1],
                        },
                        indent=1,
                    )
                )
            checkpoint.write_bytes(pickle.dumps(cma.state()))
            print(json.dumps(row), flush=True)
            if best_loss < args.stop_loss:
                print("stop loss reached", flush=True)
                break
    print(f"done: {cma.generation} generations, {time.time() - t_start:.0f} s, best {best_loss}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    params = DriveParams.from_dict(json.loads(Path(args.params).read_text())["params"])
    seeds = parse_seeds(args.seeds)
    tasks = [
        (0, vector_from_params(params), preset, seed, kind)
        for preset in D.PRESETS
        for seed in seeds
        for kind in ("day", "night")
    ]
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        results = [
            r for _i, r in pool.imap_unordered(_run_exact, [(params, *t[2:]) for t in tasks], 1)
        ]
    report = D.targets_report(results)
    report["params"] = params.as_dict()
    report["seeds"] = seeds
    report["wall_s"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps({"report": report, "runs": results}, indent=1))
    for v in report["verdicts"]:
        print("PASS" if v["pass"] else "FAIL", v["name"])
    return 0


def _run_exact(task: tuple[DriveParams, str, int, str]) -> tuple[int, dict[str, Any]]:
    """One run with the parameters exactly as given (not through the unit cube's clipping)."""
    params, preset, seed, kind = task
    return 0, D.run_day(preset, seed, params, kind=kind).to_dict()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--out", required=True)
    s.add_argument("--seeds", default="1-6")
    s.add_argument("--workers", type=int, default=10)
    s.add_argument("--pop", type=int, default=16)
    s.add_argument("--gens", type=int, default=40)
    s.add_argument("--sigma", type=float, default=0.25)
    s.add_argument("--margin", type=float, default=0.05)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--start", default=None, help="a best.json to start from")
    s.add_argument("--resume", action="store_true")
    s.add_argument("--stop-loss", type=float, default=1e-6)
    e = sub.add_parser("evaluate")
    e.add_argument("--params", required=True)
    e.add_argument("--seeds", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--workers", type=int, default=10)
    args = ap.parse_args(argv)
    return cmd_search(args) if args.cmd == "search" else cmd_evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
