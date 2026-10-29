"""The golden corpus (spec §7.6 「黃金語料」, §17 phase 4; phase-4 design G4, test T13): old ducks
replay byte for byte.

The corpus was recorded on 2026-09-27 from the unchanged v4 engine (Duck Core tree cdc8ead =
git 79cc25d): each session is a Config, every input the duck was given, and the event-chain head
v4 produced, with (events, digest) after every input and a digest every 1,000 events so a
failure names the first input that went wrong. Five small sessions are vendored in
``tests/golden/`` and replay on every run; the full 50 (``phase4/golden`` in the data folder, or
``DUCKMIND_GOLDEN_DIR``; set it to an empty string to leave them out) replay whenever they are on
this machine, on up to ``DUCKMIND_GOLDEN_WORKERS`` (default 4) processes, about a minute.

A replay builds the Config strictly (a field the engine no longer accepts fails, never dropped)
and must end on the same head with the same number of events, re-record the same inputs and
verify its chain. The vendored five go through the frozen engine ``duckmind.engines.v4_0``
directly and through ``api.open_session`` (a saved Config without ``engine_rev`` is a v4 duck:
the router must hand it to the frozen engine); the full fifty go through ``api.open_session``.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from duckmind import paths

GOLDEN = Path(__file__).with_name("golden")
VENDORED = json.loads((GOLDEN / "MANIFEST.json").read_text(encoding="utf-8"))
FULL = Path(os.environ.get("DUCKMIND_GOLDEN_DIR", str(paths.data_dir() / "phase4" / "golden")))
HAS_FULL = bool(os.environ.get("DUCKMIND_GOLDEN_DIR", "x")) and (FULL / "MANIFEST.json").is_file()
IDS = [entry["id"] for entry in VENDORED["sessions"]]


def _load(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rb") as handle:
        data: dict[str, Any] = json.loads(handle.read())
    return data


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(saved: dict[str, Any], folder: Path) -> dict[str, Any]:
    config = dict(saved["config"])
    caps = config.get("caps_file") or ""
    if caps and not os.path.isabs(caps):
        config["caps_file"] = str((folder / caps).resolve())
    return config


def _open(config: dict[str, Any], how: str) -> tuple[Any, Any]:
    """(session, dispatch) for an old duck's Config, the way ``how`` restores it."""
    if how == "frozen":
        from duckmind.engines.v4_0.duckcore import api as frozen

        unknown = sorted(set(config) - set(frozen.Config.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Config no longer accepts {unknown} (an old duck would lose them)")
        return frozen.Session(frozen.Config(**config)), frozen.dispatch
    from duckmind.duckcore import api

    return api.open_session(config), api.dispatch


def replay(saved: dict[str, Any], folder: Path, how: str) -> dict[str, Any]:
    """Replay one recorded session; what differs, with the first input that went wrong."""
    out: dict[str, Any] = {"id": saved["id"], "how": how}
    try:
        session, dispatch = _open(_config(saved, folder), how)
        log = session.core.log
        marks = saved.get("marks") or []
        first_bad_input = None
        for index, item in enumerate(saved["inputs"]):
            dispatch(session, item["method"], item["params"])
            if first_bad_input is None and index < len(marks):
                want_events, want_digest = marks[index]
                got_digest = log.digests[-1] if log.digests else "genesis"
                if len(log.events) != want_events or got_digest != want_digest:
                    first_bad_input = index
        first_bad_checkpoint = next(
            (
                at
                for at, digest in saved.get("checkpoints") or []
                if at >= len(log.digests) or log.digests[at] != digest
            ),
            None,
        )
        rerecorded = [{"method": method, "params": params} for method, params in session.inputs]
        out.update(
            head=log.head,
            events=len(log.events),
            same_head=log.head == saved["head"] and len(log.events) == saved["events"],
            same_inputs=rerecorded == saved["inputs"],
            verifies=log.verify(),
            first_bad_input=first_bad_input,
            first_bad_checkpoint=first_bad_checkpoint,
        )
    except Exception as error:  # a crash is a failure with its reason, not a stopped run
        out.update(same_head=False, error=f"{type(error).__name__}: {error}")
    out["ok"] = bool(out.get("same_head") and out.get("same_inputs") and out.get("verifies"))
    return out


def _explain(result: dict[str, Any], saved_head: str, saved_events: int) -> str:
    if "error" in result:
        return f"{result['id']} ({result['how']}): {result['error']}"
    return (
        f"{result['id']} ({result['how']}): head {str(result['head'])[:16]} want "
        f"{saved_head[:16]}, events {result['events']} want {saved_events}, "
        f"inputs re-recorded same {result['same_inputs']}, chain verifies {result['verifies']}, "
        f"first bad input {result['first_bad_input']}, "
        f"first bad 1,000-event checkpoint {result['first_bad_checkpoint']}"
    )


# ---------------------------------------------------------------------------- vendored five


def test_the_vendored_sessions_are_the_recorded_ones() -> None:
    assert VENDORED["engine"]["engine_rev"] == "v4_0"
    assert VENDORED["engine"]["duckcore_tree"] == "cdc8eadc7ad3c73791f06673a29b8b969605248e"
    assert len(IDS) == 5
    for entry in VENDORED["sessions"]:
        path = GOLDEN / entry["file"]
        assert _sha256(path) == entry["sha256"], f"{entry['file']} changed"
        saved = _load(path)
        assert saved["format"] == "duckmind-golden-v1"
        assert saved["id"] == entry["id"] and saved["head"] == entry["head"]
        assert saved["events"] == entry["events"] and len(saved["inputs"]) == entry["inputs"]
        assert "engine_rev" not in saved["config"], "a golden duck is a pre-phase-4 save"
    if HAS_FULL:  # the vendored files are byte-identical to the full corpus's
        full = {e["id"]: e for e in json.loads((FULL / "MANIFEST.json").read_text())["sessions"]}
        for entry in VENDORED["sessions"]:
            assert full[entry["id"]]["sha256"] == entry["sha256"], entry["id"]


@pytest.mark.parametrize("session_id", IDS)
def test_the_frozen_engine_replays_a_vendored_session(session_id: str) -> None:
    saved = _load(GOLDEN / f"{session_id}.json.gz")
    result = replay(saved, GOLDEN, "frozen")
    assert result["ok"], _explain(result, saved["head"], saved["events"])


@pytest.mark.parametrize("session_id", IDS)
def test_open_session_hands_an_old_duck_to_the_frozen_engine(session_id: str) -> None:
    from duckmind.duckcore import api

    saved = _load(GOLDEN / f"{session_id}.json.gz")
    session = api.open_session(_config(saved, GOLDEN))
    assert type(session.core).__module__ == "duckmind.engines.v4_0.duckcore.engine", (
        "a saved Config without engine_rev must be restored on the frozen v4 engine"
    )
    result = replay(saved, GOLDEN, "router")
    assert result["ok"], _explain(result, saved["head"], saved["events"])


# ---------------------------------------------------------------------------- a caps file

PREFIXES = VENDORED.get("prefixes", [])


def test_the_caps_prefix_is_g48s_first_inputs_with_its_caps_file() -> None:
    """The corpus has one session with an expression caps file (g48; 蛋黃 has one too): its
    first two inputs are vendored with the caps file, byte for byte, so the frozen engine's
    caps path replays on every run (a caps file it cannot find would quietly be another duck:
    ``read_caps`` ignores it, so the file must be there)."""
    assert [entry["id"] for entry in PREFIXES] == ["g48_cfg_caps_s410_first2"]
    caps = GOLDEN / VENDORED["caps_file"]
    assert caps.is_file() and _sha256(caps) == VENDORED["caps_sha256"]
    entry = PREFIXES[0]
    saved = _load(GOLDEN / entry["file"])
    assert _sha256(GOLDEN / entry["file"]) == entry["sha256"]
    assert saved["config"]["caps_file"] == VENDORED["caps_file"]
    assert (saved["head"], saved["events"], len(saved["inputs"])) == (
        entry["head"],
        entry["events"],
        entry["inputs"],
    )
    if HAS_FULL:  # the prefix is the corpus's own first inputs and marks
        manifest = json.loads((FULL / "MANIFEST.json").read_text())
        assert manifest["caps_sha256"] == VENDORED["caps_sha256"]
        full_entry = next(e for e in manifest["sessions"] if e["id"] == entry["prefix_of"])
        full = _load(FULL / full_entry["file"])
        count = entry["inputs"]
        assert saved["inputs"] == full["inputs"][:count] and saved["marks"] == full["marks"][:count]
        assert saved["prefix_of"]["sha256"] == full_entry["sha256"]


@pytest.mark.parametrize("how", ["frozen", "router"])
def test_the_caps_prefix_replays_with_its_caps_file(how: str) -> None:
    saved = _load(GOLDEN / PREFIXES[0]["file"])
    result = replay(saved, GOLDEN, how)
    assert result["ok"], _explain(result, saved["head"], saved["events"])


# ---------------------------------------------------------------------------- the full corpus


def _replay_file(args: tuple[str, str, str]) -> dict[str, Any]:
    folder, file, how = args
    return replay(_load(Path(folder) / file), Path(folder), how)


@pytest.mark.skipif(not HAS_FULL, reason=f"the full golden corpus is not at {FULL}")
def test_the_full_golden_corpus_replays_identically() -> None:
    manifest = json.loads((FULL / "MANIFEST.json").read_text(encoding="utf-8"))
    entries = manifest["sessions"]
    assert len(entries) == 50
    bad_files = [e["id"] for e in entries if _sha256(FULL / e["file"]) != e["sha256"]]
    caps = manifest.get("caps_file")
    if caps and _sha256(FULL / caps) != manifest["caps_sha256"]:
        bad_files.append(f"{caps} (caps file)")
    assert not bad_files, f"golden files changed on disk: {bad_files}"
    # Through phase 4's router, the way the simulator restores a saved duck (it must reach the
    # frozen engine: test_open_session_hands_an_old_duck_to_the_frozen_engine).
    how = "router"
    todo = [(str(FULL), e["file"], how) for e in sorted(entries, key=lambda e: -int(e["events"]))]
    workers = max(1, int(os.environ.get("DUCKMIND_GOLDEN_WORKERS", "4")))
    if workers == 1:
        results = [_replay_file(item) for item in todo]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            results = list(pool.map(_replay_file, todo))
    want = {e["id"]: e for e in entries}
    wrong = [
        _explain(r, want[r["id"]]["head"], want[r["id"]]["events"])
        for r in sorted(results, key=lambda r: str(r["id"]))
        if not r["ok"]
    ]
    assert not wrong, (
        f"{len(results) - len(wrong)}/{len(results)} golden sessions identical ({how}); "
        "these differ:\n" + "\n".join(wrong)
    )
