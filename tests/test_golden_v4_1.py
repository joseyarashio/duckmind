"""v4_1 ducks replay byte for byte (phase-4 review, finding 3; design A6, §17.7 「舊存檔永遠不變」).

A duck hatched on engine v4_1 lives in LegacyWorld v1: its saved history is Duck Core *and* that
world. ``tests/golden_v4_1/`` keeps five small v4_1 saves (``tools/record_golden_v4_1.py``): a page
duck through the page's buttons and real time, a nest duck at 600× v4's falls, a household-v2
duck, a legacy duck, and the v4 golden g07 moved to v4_1 at its middle input. Each must rebuild
through ``api.restore``'s path to the same chain head, with its world sending the same percepts.

If one moves, the change has rewritten every saved v4_1 duck's history. The fix is never to
re-record: freeze v4_1 with LegacyWorld v1 first, then give the change a new engine revision
(and the world a new header ``v``) — docs/design/phase4-boundary.md §11.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).with_name("golden_v4_1")
MANIFEST = json.loads((GOLDEN / "MANIFEST.json").read_text(encoding="utf-8"))
IDS = [entry["id"] for entry in MANIFEST["sessions"]]


def _tool() -> Any:
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import record_golden_v4_1 as tool
    finally:
        sys.path.remove(str(ROOT / "tools"))
    return tool


def _load(entry: dict[str, Any]) -> dict[str, Any]:
    path = GOLDEN / entry["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], f"{path} changed"
    with gzip.open(path, "rb") as handle:
        data: dict[str, Any] = json.loads(handle.read())
    return data


def test_the_manifest_names_this_engine_and_world() -> None:
    import duckmind.world.legacy as legacy
    from duckmind.duckcore import api, content

    tool = _tool()
    assert MANIFEST["format"] == tool.MANIFEST_FORMAT and MANIFEST["rule"] == tool.RULE
    assert MANIFEST["engine"] == {
        "engine_rev": api.ENGINE_REV,
        "table_hash": content.table_hash(),
        "legacy_world_v": legacy.HEADER_V,
    }, (
        "the v4_1 golden sessions belong to engine v4_1, this content table and LegacyWorld v1: "
        + MANIFEST["rule"]
    )
    assert len(IDS) == 5 and len(set(IDS)) == 5
    kinds = {entry["kind"] for entry in MANIFEST["sessions"]}
    assert kinds == {"page", "stress", "v2", "legacy", "moved"}


@pytest.mark.parametrize("session_id", IDS)
def test_a_v4_1_golden_session_replays_identically(session_id: str) -> None:
    entry = next(e for e in MANIFEST["sessions"] if e["id"] == session_id)
    data = _load(entry)
    assert data["format"] == _tool().FORMAT and data["id"] == session_id
    assert data["head"] == entry["head"] and data["events"] == entry["events"]
    session, marks = _tool().replay(data["save"])
    first_bad = next(
        (
            index
            for index, (got, want) in enumerate(zip(marks, data["marks"], strict=False))
            if got != want
        ),
        None,
    )
    log = session.core.log
    assert (log.head, len(log.events)) == (data["head"], data["events"]), (
        f"{session_id}: head {log.head[:16]} ({len(log.events)} events), recorded "
        f"{data['head'][:16]} ({data['events']}); first input that differs: {first_bad} "
        f"({data['save']['inputs'][first_bad] if first_bad is not None else '-'}). "
        + MANIFEST["rule"]
    )
    assert _tool().world_digest(session) == (data["world_percepts"], data["world_digest"]), (
        f"{session_id}: its LegacyWorld sent other percepts. " + MANIFEST["rule"]
    )
    assert log.verify()
    if data["kind"] == "moved":
        from duckmind.duckcore import api

        assert data["save"]["config"]["engine_rev"] == "v4_0"
        assert any(item["method"] == api.MIGRATE for item in data["save"]["inputs"])
