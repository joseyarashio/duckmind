#!/usr/bin/env python3
"""LV7: LV4's targets (docs/design/live-voice/LV4-TARGETS.md) checked again on the merged bridge.

The same scripted conversation day and the same seven targets as ``lv4_search.py evaluate``, with the
bridge's own settings given on the command line (``--set reply_urge=2.0``), so that the bridge of the
merge (LV4's drive-weighted reply with LV6's urge that grows while a person waits, fear and sleepiness
taking the input away, the ear hint) can be set against LV4's alone (``--set reply_urge=1.0`` and the
fear / drowsy ramps open) and against any adjustment. Nothing is searched; the parameters are LV4's,
exactly as recorded in docs/design/live-voice/lv4/drive-params.json.

  PYTHONPATH=src python tools/live_voice/lv7_check.py --seeds 13-28 --out /tmp/lv7.json
  PYTHONPATH=src python tools/live_voice/lv7_check.py --seeds 1-12 --set reply_urge=2.0 --out /tmp/x.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

from duckmind.live import convday as D
from duckmind.live.bridge import BridgeConfig
from duckmind.live.drives import DriveParams

PARAMS = Path(__file__).resolve().parents[2] / "docs/design/live-voice/lv4/drive-params.json"


def parse_seeds(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def parse_sets(items: list[str]) -> dict[str, float]:
    known = {f.name for f in fields(BridgeConfig)}
    out: dict[str, float] = {}
    for item in items:
        key, _, value = item.partition("=")
        if key not in known or key in ("gate", "drives"):
            raise SystemExit(f"--set {key}: not a number setting of BridgeConfig")
        out[key] = float(value)
    return out


def _run(task: tuple[dict[str, float], str, int, str]) -> dict[str, Any]:
    sets, preset, seed, kind = task
    params = DriveParams.from_dict(json.loads(PARAMS.read_text())["params"])
    cfg = BridgeConfig(duck_name=D.NAME, drives=params, **sets)
    return D.run_day(preset, seed, params, kind=kind, bridge_config=cfg).to_dict()


def summary(report: dict[str, Any]) -> list[str]:
    lines = []
    for v in report["verdicts"]:
        lines.append(f"{'PASS' if v['pass'] else 'FAIL'} {v['name']}")
        d = v.get("detail", {})
        if "rates" in d:
            rates = ", ".join(f"{k} {x:.3f}" for k, x in d["rates"].items())
            lines.append(f"     reply rate: {rates}; call_out {d['call_out_mean']}")
        elif "rate" in d:
            lines.append(f"     rate {d['rate']:.3f} (n {d['n']})")
        elif "latency_median" in d:
            lat = ", ".join(f"{k} {x:.2f}" for k, x in d["latency_median"].items())
            lines.append(f"     rate {d['rate_all']:.3f}; median latency: {lat}")
        elif "tags_per_reply" in d:
            lines.append(
                f"     rate {d['rate_all']:.3f}; tags {d['tags_per_reply']:.3f}; "
                f"{d['runs_with_a_reply']} runs; visitors {d['visitors_answered']}"
            )
        elif "violations_and_unavailable_proposals" in d:
            lines.append(
                f"     [violations, unavailable]: {d['violations_and_unavailable_proposals']}"
            )
        else:
            lines.append(f"     {json.dumps(d)[:400]}")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--set", action="append", default=[], help="a BridgeConfig number: name=value")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args(argv)
    sets = parse_sets(args.set)
    seeds = parse_seeds(args.seeds)
    tasks = [
        (sets, preset, seed, kind)
        for preset in D.PRESETS
        for seed in seeds
        for kind in ("day", "night")
    ]
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        results = list(pool.imap_unordered(_run, tasks, 1))
    report = D.targets_report(results)
    report["bridge_settings"] = sets
    report["seeds"] = seeds
    report["wall_s"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps({"report": report, "runs": results}))
    print(f"seeds {args.seeds}, bridge settings {sets or '(the defaults)'}")
    print("\n".join(summary(report)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
