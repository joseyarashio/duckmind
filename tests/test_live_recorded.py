"""Live voice, LV1: the real recordings in ``tests/fixtures/live`` (``tools/live_voice/record.py``).

Every cassette is loaded, checked (its key, its time order, nothing key-shaped, no handle, no raw
media), and **replayed through a fresh session** with no network: the session must see exactly the
tool calls and transcripts that were recorded, on the recorded timeline, and fail loudly for a
scenario that is not the one recorded. The folder stays small. What the real model did is a record,
not a promise: these tests pin down the plumbing (nothing lost, nothing added, nothing leaked), not
the model's choices."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from duckmind.live import cassette as C
from duckmind.live import events as E
from duckmind.live import keys as K
from duckmind.live import scenario as S

pytestmark = pytest.mark.usefixtures("no_network")

FOLDER = S.CASSETTE_DIR
FILE = S.load_scenario_file()
MAX_BYTES = 5_000_000
CASSETTES = sorted(p for p in FOLDER.glob("*.json") if p.name != "RECORDING.json")


def run_for(path: Path) -> S.ScenarioRun:
    sid, tid, seed = path.stem.rsplit("__", 2)
    return S.ScenarioRun(FILE, sid, tid, int(seed.lstrip("s")), require_bank=False)


def test_the_folder_is_small_and_has_an_index() -> None:
    assert FOLDER.is_dir()
    size = sum(p.stat().st_size for p in FOLDER.rglob("*") if p.is_file())
    assert size <= MAX_BYTES, f"{size} bytes"
    index = json.loads((FOLDER / "RECORDING.json").read_text(encoding="utf-8"))
    assert index["model"] == "gemini-3.8-live" and index["sdk"].startswith("google-genai ")
    recorded = set(index["recorded"])
    assert recorded == {p.stem for p in CASSETTES}
    assert not recorded & set(index["not_obtained"])
    matrix = {S.ScenarioRun(FILE, *m).run_id for m in FILE.matrix()}
    assert recorded | set(index["not_obtained"]) == matrix, (
        "every run is either recorded or listed as not obtained"
    )
    for rid, why in index["not_obtained"].items():
        assert why["why"], rid
    for rid, info in index["recorded"].items():
        assert info["bytes"] == (FOLDER / f"{rid}.json").stat().st_size, rid


@pytest.mark.parametrize("path", CASSETTES, ids=[p.stem for p in CASSETTES])
def test_a_cassette_is_sound(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert not K.find_keys(text)
    cassette = C.Cassette.load(path)  # (its inputs hash to its key)
    assert cassette.complete and cassette.scenario_id == path.stem.rsplit("__", 2)[0]
    assert cassette.meta["run_id"] == path.stem and cassette.meta["sdk"].startswith("google-genai ")
    assert cassette.inputs["model"] == "gemini-3.8-live"
    times = [e.t for e in cassette.events]
    assert times == sorted(times) and cassette.events[0].kind == E.OPENED
    assert all("handle" not in e.data for e in cassette.events)
    assert E.AUDIO_OUT not in {e.kind for e in cassette.events}  # (the model's voice is not kept)
    assert not {E.CLOSED, E.ERROR} & {e.kind for e in cassette.events}
    for event in cassette.of_kind(E.TOOL_CALL):
        assert event.data["calls"] and all(c["id"] and c["name"] for c in event.data["calls"])


@pytest.mark.parametrize("path", CASSETTES, ids=[p.stem for p in CASSETTES])
def test_a_cassette_replays_through_a_session_to_what_was_recorded(path: Path) -> None:
    cassette = C.Cassette.load(path)
    run = run_for(path)
    result = S.replay(run, cassette)  # (raises CassetteMismatch if the scenario is not this one)
    status = result.session.status()
    assert status["faults"] == 0 and status["connections"] == 1
    recorded = [c for e in cassette.of_kind(E.TOOL_CALL) for c in e.data["calls"]]
    assert [(c.name, c.raw) for c in result.calls] == [(c["name"], c["args"]) for c in recorded]
    heard = "".join(e["text"] for e in result.log("heard"))
    assert heard == "".join(e.data["text"] for e in cassette.of_kind(E.INPUT_TEXT))
    assert status["usage"]["totals"]["turns"] == len(cassette.of_kind(E.USAGE))
    assert status["usage"]["totals"]["tokens"] == sum(
        e.data["total"] for e in cassette.of_kind(E.USAGE)
    )
    for call in result.calls:  # (queued with the time they arrived on the scenario's clock)
        assert 0 < call.t < run.scenario.total_s + 1.0


def test_a_recording_does_not_replay_for_another_temperament_or_seed() -> None:
    if not CASSETTES:
        pytest.skip("no recordings in this tree")
    path = CASSETTES[0]
    cassette = C.Cassette.load(path)
    sid, tid, seed = path.stem.rsplit("__", 2)
    other_t = next(t for t in FILE.temperaments if t != tid)
    with pytest.raises(C.CassetteMismatch):
        S.replay(
            S.ScenarioRun(FILE, sid, other_t, int(seed.lstrip("s")), require_bank=False), cassette
        )
    other_seed = 11 if int(seed.lstrip("s")) != 11 else 7
    with pytest.raises(C.CassetteMismatch):
        S.replay(S.ScenarioRun(FILE, sid, tid, other_seed, require_bank=False), cassette)
