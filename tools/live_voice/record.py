"""Record the live-voice cassettes: one real Gemini Live session per scenario × temperament × seed.

Run with the duckmind venv plus the ``google-genai`` SDK (a required dependency)::

    PYTHONPATH=src python tools/live_voice/record.py [--only call_name,hello] [--dry-run]

The key is read from the Keychain (or ``$GEMINI_API_KEY``) by the connection itself when it opens
and is never printed, logged or written (every line this tool prints or logs goes through
:func:`duckmind.live.keys.redact`; every cassette is scanned for a key before it is written).

For each (scenario, temperament, seed) of ``tools/live_voice/scenarios.json`` — the whole matrix, or
what ``--only`` / ``--temperaments`` / ``--seeds`` name — it feeds the scenario to a real session
**in real time** (the server's voice activity detection needs the audio at the speed it was spoken)
and keeps what the server sent back as ``tests/fixtures/live/<scenario>__<temperament>__s<seed>.json``
(:mod:`duckmind.live.cassette`). Free-tier limits are respected by spacing the sessions
(``--space-s``) and, when the service answers RESOURCE_EXHAUSTED, by waiting (``--quota-wait-s``,
doubling up to ``--quota-wait-max-s``) and trying that recording again later; what could not be
obtained, and why, is in ``tests/fixtures/live/RECORDING.json`` beside the recorded ones.

Scenarios that need the duck's eye (``needs_eye``) need their pictures in
``tests/fixtures/live/eye`` (from a physics run, ``tools/live_voice/make_eye_fixtures.py``); without
them they are listed as not obtained, not recorded with a stand-in.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from duckmind import paths
from duckmind.live import scenario as S
from duckmind.live.cassette import Cassette, CassetteMismatch, KeyInCassette, scenario_hash
from duckmind.live.genai_transport import GenaiTransport
from duckmind.live.keys import redact
from duckmind.live.meter import UsageCaps
from duckmind.live.transport import DEFAULT_MODEL

LOG_DEFAULT = paths.data_dir() / "live-voice" / "record-log.jsonl"
INDEX_NAME = "RECORDING.json"
CAPS = UsageCaps(tokens=120_000, audio_s=90.0, frames=60)  # (one recording never costs more)


def log(path: Path | None, **fields: Any) -> None:
    line = redact(
        json.dumps(
            {"at": datetime.now(UTC).isoformat(timespec="seconds"), **fields}, ensure_ascii=False
        )
    )
    print(line, flush=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def load_index(folder: Path) -> dict[str, Any]:
    path = folder / INDEX_NAME
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"recorded": {}, "not_obtained": {}}


def save_index(folder: Path, index: dict[str, Any]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    out = {
        "note": "Real Gemini Live recordings (tools/live_voice/record.py). 'not_obtained' lists what the service did not give and why; nothing is made up to fill a gap.",
        "model": DEFAULT_MODEL,
        "sdk": f"google-genai {metadata.version('google-genai')}",
        "recorded": dict(sorted(index["recorded"].items())),
        "not_obtained": dict(sorted(index["not_obtained"].items())),
    }
    (folder / INDEX_NAME).write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )


def git_state() -> str:
    """The commit this tree is at (and ``+dirty`` when tracked files differ from it), for the cassette's notes."""
    try:
        here = Path(__file__).resolve().parents[2]
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=here,
            capture_output=True,
            text=True,
            timeout=10,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no", "--", "src", "tools", "tests"],
            cwd=here,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return head.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def folder_size(folder: Path) -> int:
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())


def try_record(run: S.ScenarioRun, caps: UsageCaps) -> tuple[Cassette | None, str, dict[str, Any]]:
    """One attempt: (cassette or None, why not, facts)."""
    clock = S.RealtimeClock()
    transport = GenaiTransport(clock=clock)
    meta = {
        "commit": git_state(),
        "sdk": f"google-genai {metadata.version('google-genai')}",
        "python": platform.python_version(),
        "recorded_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    try:
        cassette, result = S.record_scenario(run, transport, clock, caps=caps, meta=meta)
    except (CassetteMismatch, KeyInCassette):
        raise
    except Exception as error:  # (whatever it was, the next recording goes on)
        return None, "error", {"error": redact(f"{type(error).__name__}: {error}")[:200]}
    status = result.session.status()
    facts = {
        "state": status["state"],
        "connections": status["connections"],
        "last_error": status["last_error"],
        "faults": status["faults"],
        "events": status["events"],
        "tokens": status["usage"]["totals"]["tokens"],
        "audio_dropped_s": status["usage"]["totals"]["audio_dropped_s"],
        "wall_s": round(result.wall_s, 1),
    }
    errors = [e for e in cassette.events if e.kind in ("closed", "error")]
    if errors:
        kind = str(errors[0].data.get("kind"))
        return None, ("quota" if kind == "quota" else kind), facts
    if not any(e.kind == "opened" for e in cassette.events):
        return None, "never_connected", facts
    if result.session.faults:
        return None, "session_fault", facts
    return cassette, "", facts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", help="comma-separated scenario ids")
    parser.add_argument("--temperaments", help="comma-separated temperament ids")
    parser.add_argument("--seeds", help="comma-separated microphone seeds")
    parser.add_argument("--out", default=str(S.CASSETTE_DIR))
    parser.add_argument("--log", default=str(LOG_DEFAULT))
    parser.add_argument("--space-s", type=float, default=8.0, help="rest between two sessions")
    parser.add_argument("--quota-wait-s", type=float, default=90.0)
    parser.add_argument("--quota-wait-max-s", type=float, default=900.0)
    parser.add_argument("--attempts", type=int, default=4, help="tries per recording")
    parser.add_argument(
        "--budget-s", type=float, default=3 * 3600.0, help="stop after this much wall time"
    )
    parser.add_argument(
        "--redo", action="store_true", help="record again what already has a good cassette"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="list what would be recorded and why not; no network"
    )
    args = parser.parse_args(argv)

    folder = Path(args.out)
    log_path = Path(args.log) if args.log != "-" else None
    file = S.load_scenario_file()
    only = set(args.only.split(",")) if args.only else None
    temps = set(args.temperaments.split(",")) if args.temperaments else None
    seeds = {int(s) for s in args.seeds.split(",")} if args.seeds else None
    todo = [
        (sid, tid, seed)
        for sid, tid, seed in file.matrix()
        if (only is None or sid in only)
        and (temps is None or tid in temps)
        and (seeds is None or seed in seeds)
    ]
    index = load_index(folder)
    started = time.monotonic()
    wait_s = args.quota_wait_s
    for number, (sid, tid, seed) in enumerate(todo, start=1):
        run = S.ScenarioRun(file, sid, tid, seed, eye_dir=folder / "eye")
        rid = run.run_id
        path = S.cassette_path(rid, folder)
        try:
            inputs = run.inputs()
        except S.FixtureMissing as error:
            index["not_obtained"][rid] = {"why": "fixture_missing", "detail": str(error)}
            log(
                log_path,
                run=rid,
                n=number,
                of=len(todo),
                result="not_obtained",
                why="fixture_missing",
                detail=str(error),
            )
            continue
        if path.is_file() and not args.redo:
            try:
                old = Cassette.load(path)
                if old.scenario_hash == scenario_hash(inputs) and old.complete:
                    index["recorded"].setdefault(rid, {"bytes": path.stat().st_size})
                    log(log_path, run=rid, n=number, of=len(todo), result="kept")
                    continue
            except (CassetteMismatch, ValueError, KeyError) as error:
                log(log_path, run=rid, note="old cassette not usable", detail=str(error)[:120])
        if args.dry_run:
            log(
                None,
                run=rid,
                n=number,
                of=len(todo),
                result="would_record",
                seconds=run.scenario.total_s,
            )
            continue
        if time.monotonic() - started > args.budget_s:
            index["not_obtained"][rid] = {
                "why": "budget",
                "detail": "the run's time budget ran out",
            }
            log(log_path, run=rid, result="not_obtained", why="budget")
            continue
        for attempt in range(1, args.attempts + 1):
            log(
                log_path,
                run=rid,
                n=number,
                of=len(todo),
                result="recording",
                attempt=attempt,
                seconds=run.scenario.total_s,
            )
            cassette, why, facts = try_record(run, CAPS)
            if cassette is not None:
                size = cassette.save(path)
                index["recorded"][rid] = {
                    "bytes": size,
                    "events": len(cassette.events),
                    "tool_calls": sum(
                        len(e.data["calls"]) for e in cassette.events if e.kind == "tool_call"
                    ),
                    "attempts": attempt,
                }
                index["not_obtained"].pop(rid, None)
                log(log_path, run=rid, result="recorded", bytes=size, **facts)
                wait_s = args.quota_wait_s
                break
            log(log_path, run=rid, result="failed", why=why, attempt=attempt, **facts)
            if attempt == args.attempts:
                index["not_obtained"][rid] = {"why": why, "attempts": attempt, "last": facts}
                break
            if why == "quota":
                log(log_path, run=rid, note=f"quota: waiting {wait_s:.0f} s")
                time.sleep(wait_s)
                wait_s = min(args.quota_wait_max_s, wait_s * 2)
            else:
                time.sleep(30.0)
        save_index(folder, index)
        time.sleep(args.space_s)
    if not args.dry_run:
        save_index(folder, index)
    obtained, missing = len(index["recorded"]), len(index["not_obtained"])
    log(
        log_path if not args.dry_run else None,
        summary=True,
        recorded=obtained,
        not_obtained=missing,
        folder_bytes=folder_size(folder) if folder.is_dir() else 0,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
