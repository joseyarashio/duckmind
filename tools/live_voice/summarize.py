"""Read the recorded cassettes (replayed through the session, no network) and say what the real
model did, scenario by scenario: which tools it called, how soon after the speech ended, what it
heard, which of it the echo gate marked, what it cost.

    PYTHONPATH=src python tools/live_voice/summarize.py [--only echo_only,call_name] [--calls]

A plain table for the person running it; the numbers are the recordings' own (nothing is
measured again, nothing is sent anywhere).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from duckmind.live import scenario as S
from duckmind.live.cassette import Cassette


def run_for(stem: str, file: S.ScenarioFile) -> S.ScenarioRun:
    sid, tid, seed = stem.rsplit("__", 2)
    return S.ScenarioRun(file, sid, tid, int(seed.lstrip("s")), require_bank=False)


def latency(result: S.ScenarioResult, run: S.ScenarioRun, call: S.QueuedCall) -> float | None:
    """Seconds from the end of the latest line that ended before the call to the call."""
    ends = [s.t + run.library().get(s.utterance).duration_s for s in run.scenario.say]
    before = [e for e in ends if e <= call.t]
    return None if not before else round(call.t - max(before), 2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", default=str(S.CASSETTE_DIR))
    parser.add_argument("--only", help="comma-separated scenario ids")
    parser.add_argument("--calls", action="store_true", help="list every call of every recording")
    args = parser.parse_args(argv)
    only = set(args.only.split(",")) if args.only else None
    file = S.load_scenario_file()
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(Path(args.folder).glob("*.json")):
        if path.name == "RECORDING.json":
            continue
        sid = path.stem.rsplit("__", 2)[0]
        if only and sid not in only:
            continue
        run = run_for(path.stem, file)
        result = S.replay(run, Cassette.load(path))
        calls = result.calls
        heard = result.log("heard")
        status = result.session.status()
        row = {
            "id": path.stem,
            "tools": [c.name for c in calls],
            "calls": calls,
            "first_s": None if not calls else latency(result, run, calls[0]),
            "echo_calls": sum(c.echo_suspect for c in calls),
            "problems": sum(bool(c.problems) for c in calls),
            "off_bank": sum(len(c.off_bank) for c in calls),
            "heard": [(e["text"], e["echo_suspect"]) for e in heard],
            "tokens": status["usage"]["totals"]["tokens"],
        }
        by_scenario[sid].append(row)
    for sid, rows in by_scenario.items():
        print(f"\n== {sid}  ({len(rows)} recordings)")
        counts = Counter(tuple(r["tools"]) for r in rows)
        for tools, n in counts.most_common():
            print(f"   {n:2d} x {', '.join(tools) if tools else '(nothing)'}")
        first = [r["first_s"] for r in rows if r["first_s"] is not None]
        if first:
            print(f"   first call after the speech ended: min {min(first)} s, max {max(first)} s")
        print(
            f"   calls marked echo-only: {sum(r['echo_calls'] for r in rows)}; with problems: "
            f"{sum(r['problems'] for r in rows)}; tags off the bank: {sum(r['off_bank'] for r in rows)}; "
            f"tokens: {sum(r['tokens'] for r in rows)}"
        )
        heard = Counter(h for r in rows for h in r["heard"])
        for (text, suspect), n in heard.most_common(6):
            print(f"   heard {n:2d} x {text!r}{'  [echo suspect]' if suspect else ''}")
        if args.calls:
            for r in rows:
                for c in r["calls"]:
                    print(
                        f"   {r['id']}: t={c.t:6.2f} {c.name} {json.dumps(c.args, ensure_ascii=False)}"
                    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
