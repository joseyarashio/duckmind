#!/usr/bin/env python3
"""Record the v4_1 golden sessions (``tests/golden_v4_1/``; phase-4 review, finding 3).

A v4_1 duck is Duck Core *and* LegacyWorld v1: its saved history (``{"config", "world",
"inputs"}``) replays only while both stay as they are. These five small sessions pin them the
way ``tests/golden/`` pins the frozen v4 engine: each keeps a v4_1 save, the chain head and the
number of events it produced, a digest of every percept its LegacyWorld sent, and (events,
digest) after every input so a failure names the first input that went wrong.
``tests/test_golden_v4_1.py`` replays them on every run.

Re-recording is **not** how a failing golden test is fixed. A change that moves these heads
changes every saved v4_1 duck's history: first freeze v4_1 with LegacyWorld v1
(``tools/freeze_engine.py``), give the changed engine a new ``ENGINE_REV`` (and the changed
world a new header ``v``), and only then record the new revision's own golden sessions
(docs/design/phase4-boundary.md §11). This script is for that last step, and for the first
recording (2026-09-28, phase-4 fix pass).

Usage::

    python tools/record_golden_v4_1.py            # (re)write tests/golden_v4_1/
    python tools/record_golden_v4_1.py --check    # exit 1 if a recorded session replays otherwise
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "golden_v4_1"
FORMAT = "duckmind-golden-v4_1-v1"
MANIFEST_FORMAT = "duckmind-golden-v4_1-manifest-v1"
RULE = (
    "A v4_1 duck is Duck Core and LegacyWorld v1 together. If these heads move, every saved v4_1 "
    "duck's history moves: freeze v4_1 with LegacyWorld v1 first (tools/freeze_engine.py), give "
    "the changed engine a new ENGINE_REV and the changed world a new header v, and record that "
    "revision's own golden sessions (docs/design/phase4-boundary.md §11). Never re-record these."
)
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]


def _page(session: Any) -> None:
    """What a page user does (phase4_harness's busy day, after the page's own buttons)."""
    from phase4_harness import act, lively, send, walking_age

    walking_age(session)
    send(session, "set_setting", {"key": "conservative", "value": False})
    act(session, "tv", None, on=True)
    send(session, "advance", {"minutes": 60})
    act(session, "tv", None, on=False)
    act(session, "pad_play", "child", minutes=6)
    send(session, "advance", {"minutes": 30})
    act(session, "hold_head", "owner")
    act(session, "touch", "owner", where="crown", seconds=8, firmness="scratch")
    send(session, "command", {"text": "摸摸頭", "persona": "owner"})
    send(session, "party", {"guest_mode": False})
    act(session, "pickup", "owner")
    act(session, "place_on_nest", "owner")
    send(session, "set_setting", {"key": "floor", "value": False})
    send(session, "set_setting", {"key": "floor", "value": True})
    for _ in range(90):  # (a minute and a half in real time, a second at a time)
        send(session, "advance", {"seconds": 1})
    lively(session, 61, days=1)


def _stress(session: Any) -> None:
    from phase4_harness import lively, walking_age

    walking_age(session)
    lively(session, 43, days=1)


def _moved() -> Any:
    """The vendored v4 golden g07 (a clumsy duck) moved to v4_1 halfway, then on."""
    from duckmind.duckcore import api
    from duckmind.engines.v4_0.duckcore import api as v4

    folder = ROOT / "tests" / "golden"
    with gzip.open(folder / "g07_web_clumsy_dual7_s5.json.gz", "rb") as handle:
        saved = json.loads(handle.read())
    old = v4.Session(v4.Config(**saved["config"]))
    inputs = saved["inputs"]
    seam = len(inputs) // 2
    for item in inputs[:seam]:
        v4.dispatch(old, item["method"], item["params"])
    moved = api.migrate(old)
    for item in inputs[seam:]:
        api.dispatch(moved, item["method"], item["params"])
    return moved


def _sessions() -> list[tuple[str, str, str, Callable[[], Any]]]:
    from phase4_harness import (
        household_legacy,
        household_v2,
        live_legacy,
        live_v2,
        lively_config,
        open_v41,
    )

    def page() -> Any:
        session = open_v41(lively_config(61, True, False))
        _page(session)
        return session

    def stress() -> Any:
        session = open_v41(lively_config(43, True, True))
        _stress(session)
        return session

    def v2() -> Any:
        config, kind = household_v2(3)
        session = open_v41(config)
        live_v2(session, kind, 8)
        return session

    def legacy() -> Any:
        config, first, second = household_legacy(1)
        session = open_v41(config)
        live_legacy(session, first, second, 8)
        return session

    return [
        (
            "v41_page_nest_s61",
            "page",
            "a nest duck through the page's buttons: settings, the TV, the gamepad, a held head, "
            "the command box, a party, lifted onto its nest, the floor switch, real time, then a "
            "busy day of hands and walks",
            page,
        ),
        (
            "v41_stress_nest_s43",
            "stress",
            "a nest duck at 600× v4's falls, weak righting and failed stand-ups, through a busy day",
            stress,
        ),
        ("v41_v2_i3", "v2", "run A's household-v2 duck 3 for 8 days", v2),
        ("v41_legacy_i1", "legacy", "run A's legacy duck 1 for 8 days", legacy),
        (
            "v41_moved_g07",
            "moved",
            "the v4 golden g07 moved to v4_1 at its middle input (the migration seam), then on",
            _moved,
        ),
    ]


def world_digest(session: Any) -> tuple[int, str]:
    """How many percepts its world sent, and a sha256 over them (canonical JSON)."""
    from duckmind import worldport

    sent = list(getattr(session.port, "sent", []))
    return len(sent), hashlib.sha256(worldport.canonical(sent).encode("utf-8")).hexdigest()


def replay(save: dict[str, Any]) -> tuple[Any, list[list[Any]]]:
    """The saved duck rebuilt input by input (``api.restore``'s way), with (events, digest)
    after each input."""
    from duckmind.duckcore import api

    session: Any = api.open_session(dict(save["config"]), save.get("world"))
    marks: list[list[Any]] = []
    for item in save["inputs"]:
        method, params = str(item["method"]), dict(item["params"])
        if method == api.MIGRATE:
            session = api.migrate(session, str(params.get("to", api.ENGINE_REV)))
        else:
            api.dispatch(session, method, params)
        log = session.core.log
        marks.append([len(log.events), log.digests[-1] if log.digests else "genesis"])
    return session, marks


def record() -> dict[str, Any]:
    import duckmind.world.legacy as legacy
    from duckmind.duckcore import api, content

    OUT.mkdir(parents=True, exist_ok=True)
    entries = []
    for session_id, kind, about, make in _sessions():
        session = make()
        save = json.loads(json.dumps(api.save_json(session), ensure_ascii=False))
        again, marks = replay(save)
        if again.core.log.head != session.core.log.head:
            raise SystemExit(f"{session_id}: its own save does not rebuild it")
        count, digest = world_digest(again)
        if (count, digest) != world_digest(session):
            raise SystemExit(f"{session_id}: its world's percepts differ when rebuilt")
        data = {
            "format": FORMAT,
            "id": session_id,
            "kind": kind,
            "about": about,
            "save": save,
            "head": session.core.log.head,
            "events": len(session.core.log.events),
            "world_percepts": count,
            "world_digest": digest,
            "marks": marks,
        }
        path = OUT / f"{session_id}.json.gz"
        raw = json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8")
        path.write_bytes(gzip.compress(raw, mtime=0))
        entries.append(
            {
                "id": session_id,
                "kind": kind,
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "head": data["head"],
                "events": data["events"],
                "inputs": len(save["inputs"]),
                "world_percepts": count,
                "world_digest": digest,
            }
        )
        print(f"{session_id}: {data['events']} events, {count} world percepts, {len(marks)} inputs")
    manifest = {
        "format": MANIFEST_FORMAT,
        "about": (
            "Five small v4_1 sessions (Duck Core v4_1 living in LegacyWorld v1), recorded by "
            "tools/record_golden_v4_1.py and replayed by tests/test_golden_v4_1.py on every run."
        ),
        "rule": RULE,
        "engine": {
            "engine_rev": api.ENGINE_REV,
            "table_hash": content.table_hash(),
            "legacy_world_v": legacy.HEADER_V,
        },
        "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "sessions": entries,
    }
    (OUT / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return manifest


def check() -> int:
    manifest = json.loads((OUT / "MANIFEST.json").read_text(encoding="utf-8"))
    bad = 0
    for entry in manifest["sessions"]:
        with gzip.open(OUT / entry["file"], "rb") as handle:
            data = json.loads(handle.read())
        session, _marks = replay(data["save"])
        same = session.core.log.head == data["head"] and world_digest(session) == (
            data["world_percepts"],
            data["world_digest"],
        )
        print(f"{entry['id']}: {'same' if same else 'DIFFERENT'}")
        bad += not same
    return 1 if bad else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="replay; exit 1 on a difference")
    args = parser.parse_args()
    import duckmind.world.legacy  # noqa: F401  (registers LegacyWorld)

    if args.check:
        return check()
    record()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
