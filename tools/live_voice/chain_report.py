"""What the duck does with the 68 real recordings (LV2): every cassette through the whole chain on a
fake clock, then the numbers: per scenario what the gate accepted and refused (and why), what got
as far as the competition and what was voiced, who got named, and how fast a person speaking over
the duck's call stops it (the recordings' own, and a closed loop with the duck's echo cut).

    PYTHONPATH=src python tools/live_voice/chain_report.py [--only stop_calling,echo_only] [--closed]

No network, no key, no physics; about 15 s (the closed loop another 20)."""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter, defaultdict
from typing import Any

from duckmind.live import chain as CH
from duckmind.live import scenario as S


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))] if ordered else float("nan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", default="", help="comma-separated scenario ids")
    parser.add_argument(
        "--closed", action="store_true", help="also run the closed-loop stop trials"
    )
    args = parser.parse_args(argv)
    only = {s for s in args.only.split(",") if s}
    file = S.load_scenario_file()
    per: dict[str, list[CH.ChainResult]] = defaultdict(list)
    for sid, tid, seed in file.matrix():
        if only and sid not in only:
            continue
        per[sid].append(CH.run_chain(file, sid, tid, seed))

    print("== per scenario: the gate and the bridge (accepted / refused: reason)")
    for sid, results in per.items():
        counts: Counter[str] = Counter()
        reasons: Counter[str] = Counter()
        voiced = entered = 0
        for r in results:
            for e in r.bridge.log:
                if e["stage"] in ("gate", "bridge"):
                    key = f"{e['tool']}:{e['decision']}"
                    counts[key] += 1
                    if e["reason"]:
                        reasons[f"{e['tool']}:{e['reason']}"] += 1
            voiced += len(r.bridge.voiced)
            entered += len(r.bridge.entered)
        faults = sum(r.bridge.faults for r in results)
        print(f"{sid} ({len(results)} recordings)")
        print("   calls:", dict(sorted(counts.items())) or "none")
        if reasons:
            print("   reasons:", dict(sorted(reasons.items())))
        print(f"   into the selector: {entered}; voiced: {voiced}; faults: {faults}")
    named = [
        (r.run_id, [(p["pid"], p["name"], p["pending"]) for p in r.brain.learn.people_view()])
        for sid in ("intro_then_hello_mom", "intro_then_hello_kid")
        for r in per.get(sid, [])
    ]
    if named:
        print("== people after the introductions (id, name, pending)")
        for rid, people in named:
            print(f"   {rid}: {people}")

    barges = [(r.run_id, b) for r in per.get("stop_calling", []) for b in r.bridge.barges]
    if barges:
        print("== a person speaks over the call (the recordings' own, open loop)")
        total = [b.total_s for _rid, b in barges]
        detect = [b.detect_s for _rid, b in barges]
        apply = [b.apply_s for _rid, b in barges]
        print(
            f"   n={len(barges)} stopped={sum(b.stopped for _r, b in barges)} "
            f"total max {max(total) * 1000:.0f} ms mean {statistics.mean(total) * 1000:.0f} ms "
            f"(detect {statistics.mean(detect) * 1000:.0f} ms, wait for the tick "
            f"{statistics.mean(apply) * 1000:.0f} ms, max {max(apply) * 1000:.0f} ms)"
        )

    if args.closed:
        print("== closed loop: the stop cuts the duck's own echo in the microphone")
        latencies: list[float] = []
        rows: list[Any] = []
        for tid in file.temperaments:
            for offset in (0.1, 0.2, 0.3, 0.4, 0.5):
                trial, _bridge, _mic = CH.closed_loop_barge(file, tid, speak_offset_s=offset)
                rows.append(trial)
                if trial.latency_s is not None:
                    latencies.append(trial.latency_s)
        stopped = [t for t in rows if t.stopped]
        print(
            f"   n={len(rows)} stopped={len(stopped)}; speech onset -> the stop in the body: "
            f"min {min(latencies) * 1000:.0f} ms, median {statistics.median(latencies) * 1000:.0f} ms, "
            f"p95 {pct(latencies, 0.95) * 1000:.0f} ms, max {max(latencies) * 1000:.0f} ms"
        )
        cut = [
            t.call_end_natural - t.echo_end for t in stopped if t.echo_end and t.call_end_natural
        ]
        print(
            f"   the call was cut {statistics.mean(cut) * 1000:.0f} ms before it would have ended (mean)"
        )
        controls = [
            CH.closed_loop_barge(file, tid, speak_offset_s=None) for tid in file.temperaments
        ]
        print(
            "   the duck's own call alone: stops "
            f"{sum(c[0].stopped for c in controls)} of {len(controls)}, voiced frames "
            f"{sum(c[1].vad.voiced_frames for c in controls if c[1].vad)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
