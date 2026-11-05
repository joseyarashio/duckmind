"""One duck, one file (docs/duck-file.md; PLAYABLE-PET §3.5): a duck exported on one computer and
imported on another is the same duck — name, nature, growth, the people it knows, its tricks, its
calls — while the home it knew stays behind. Import never overwrites a duck silently, and a damaged,
changed or too-new file is refused with nothing written."""

from __future__ import annotations

import io
import json
import threading
import urllib.error
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import duckmind.world.legacy  # noqa: F401  (registers LegacyWorld)
from duckmind.duckcore import api
from duckmind.duckcore.api import Config
from duckmind.life import duckfile as df
from duckmind.life import snapshot as snap
from duckmind.life import voice_lineage as VL
from duckmind.life.clock import TPE, DuckClock
from duckmind.life.record import LifeRecord
from duckmind.neuro.drives import Drives
from duckmind.neuro.genome import Genome
from duckmind.neuro.habituation import Habituation
from duckmind.neuro.learn import Learner
from duckmind.simweb import lineage_voice as LV
from duckmind.simweb import voice
from duckmind.simweb.life_host import LifeHost
from duckmind.simweb.server import SimServer, SimState, make_handler

SEED = 4107
BIRTH = datetime(2026, 9, 1, 8, 0, tzinfo=TPE)
LOOK = [0.30, 0.40, 0.50, 0.20, 0.10, 0.60]
HOME_CTX = {"home": "default_home,no_nest", "start": "mat"}


class FakeBrain:
    """The parts of a NeuroBrain the snapshot touches (as tests/test_life_snapshot.py)."""

    def __init__(self) -> None:
        p = Genome.from_seed(SEED, gd=0).p
        self.learn = Learner(p)
        self.hab = Habituation(p)
        self.drives = Drives()
        self._t0, self._t_prev = 0, 400_000


def taught() -> FakeBrain:
    """A duck that knows 小明, two tricks, a scare spot, a toy and its nest (the last three: the home)."""
    brain = FakeBrain()
    learn, t = brain.learn, 400.0
    pid = learn.see_person(5, LOOK, t - 30)
    learn.bind_name(pid, "小明", "小明", t - 20)
    learn.people[pid].bond = 0.6
    learn.keep_note(t - 10, pid, "喜歡抱抱")
    for _ in range(3):
        learn.tricks.praise("bow", 50)
    learn.tricks.praise("shake", 50)
    learn.tricks.scold("spin", 50)
    for k in range(3):
        learn.fell((1.2, 0.3, 0.0), t - 100 + k * 10)
        assert learn.righted(t - 100 + k * 10 + 6.0, helped=False)
    learn.see_thing(3, "toy", (0.0, 0.0, 0.0), 1.0, 0.5, t - 50, 5.0, 0.0)
    learn.at_dock((0.1, -0.2, 0.0))
    learn.cue("sound:name", t - 5)
    learn.outcome(1.0, t - 4)
    return brain


def computer(monkeypatch: pytest.MonkeyPatch, home: Path) -> Path:
    """Be one computer: its own DUCKMIND_HOME (lives, data, voice libraries)."""
    monkeypatch.setenv("DUCKMIND_HOME", str(home))
    for name in ("DUCKMIND_DATA_DIR", "DUCKMIND_PATHS_FILE", "DUCKMIND_LINEAGE_ROOT"):
        monkeypatch.delenv(name, raising=False)
    return home


def make_bank(vseed: int) -> Path:
    """A voice bank of tones in upstream's layout, in this computer's data folder."""
    folder = df.bank_root() / f"v{voice.BANK_VERSION}" / f"seed{vseed}"
    for t_index, tag in enumerate(voice.TAGS):
        (folder / tag).mkdir(parents=True, exist_ok=True)
        for index, name in enumerate(voice.file_names(tag)):
            n = int(voice.SAMPLE_RATE * (0.2 + 0.02 * index))
            f0 = 400.0 + 60.0 * t_index + 5.0 * index
            x = 0.3 * np.sin(2 * np.pi * f0 * np.arange(n) / voice.SAMPLE_RATE)
            LV.write_wav(folder / tag / name, x)
    (folder / voice.MARKER).write_text(f"{vseed}:{voice.BANK_VERSION}")
    return folder


def hatch(folder: Path) -> tuple[LifeRecord, SimState]:
    """A life lived a few days (Duck Core's save on disk), its mind saved, and its eye and layouts
    (the home) beside it."""
    record = LifeRecord.create(folder, name="蛋黃", seed=SEED, birth=BIRTH, nature="explorer")
    wall = [BIRTH.timestamp() + 5 * 86400 + 3600]
    clock = DuckClock(wall=lambda: wall[0])
    config = Config(seed=SEED, name="蛋黃", tof=True, nest=False, start=record.birth.isoformat())
    state = SimState(config, None, record.core_path, life=record, clock=clock, attach_neural=True)
    with state.lock:
        state.save()
    data = snap.capture(
        taught(),
        name="蛋黃",
        seed=SEED,
        gd=int(state.session.core.s.gd),
        calendar=clock.calendar(),
        context=HOME_CTX,
        why="test",
        wall=wall[0],
        talk=0.4,
    )
    data["map"] = {
        "v": 1,
        "cell": 0.1,
        "corr": [0, 0, 0],
        "dock": None,
        "cells": [[1, 2, 0.5, 9.0]],
    }
    snap.write(record.neural_path, data)
    (record.eye_dir).mkdir()
    (record.eye_dir / "default_home.floor.json").write_text("{}")
    (folder / "layouts").mkdir()
    (folder / "layouts" / "active.json").write_text("{}")
    return record, state


def voice_library(record: LifeRecord) -> Path:
    """The library its lineage sounds like now, built on this computer from its bank."""
    session = api.restore(json.loads(record.core_path.read_text(encoding="utf-8")))
    vseed = voice.voice_seed(SEED)
    bank = make_bank(vseed)
    plan = VL.plan_for(VL.view_of(session.core.s), vseed, LV.bank_letters(bank))
    return LV.build_library(plan, bank, df.lineage_root()).folder


@pytest.fixture()
def exported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Computer A: a duck with a voice library, exported to a file."""
    computer(monkeypatch, tmp_path / "A")
    record, _state = hatch(df.lives_dir() / "danhuang")
    library = voice_library(record)
    data, manifest = df.pack(record.folder)
    return {"record": record, "library": library, "data": data, "manifest": manifest}


def repack(data: bytes, change: dict[str, bytes | None], *, manifest: Any = None) -> bytes:
    """The same duck file with members changed (None: removed), the manifest kept as it was (or
    replaced by ``manifest`` — a dict, or a function of the old one)."""
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        members = {name: zf.read(name) for name in zf.namelist()}
    old = json.loads(members[df.MANIFEST])
    members[df.MANIFEST] = json.dumps(
        manifest(old) if callable(manifest) else manifest or old
    ).encode()
    for name, body in change.items():
        if body is None:
            members.pop(name, None)
        else:
            members[name] = body
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zf:
        for name, body in members.items():
            zf.writestr(name, body)
    return out.getvalue()


def resealed(data: bytes, change: dict[str, bytes]) -> bytes:
    """Members changed and the manifest's checksums made to match (a file another build wrote)."""

    def fix(old: dict[str, Any]) -> dict[str, Any]:
        for name, body in change.items():
            old["parts"][name] = {"sha256": df._sha(body), "bytes": len(body)}
        return old

    return repack(data, dict(change), manifest=fix)


# ── the same duck on the other computer ─────────────────────────────────────────────────────


def test_a_duck_moved_to_another_computer_is_the_same_duck(
    exported: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record_a: LifeRecord = exported["record"]
    core_a = record_a.core_path.read_bytes()
    library_hash = LV.small_hash(exported["library"])

    computer(monkeypatch, tmp_path / "B")  # another DUCKMIND_HOME: no duck, no bank, no library
    done = df.install(exported["data"])
    folder = Path(done["folder"])
    assert folder.parent == tmp_path / "B" / "lives" and not done["replaced"]

    # name, nature, birth (so its growth day is the calendar's), the same identity
    record_b = LifeRecord.load(folder)
    assert record_b.view() == record_a.view()
    assert df.record_id(record_b) == df.record_id(record_a) == done["id"]
    # its growth: Duck Core's save byte for byte, replayed to the same day, stage, branch and chain
    assert (folder / "core.json").read_bytes() == core_a
    a = api.restore(json.loads(core_a))
    b = api.restore(json.loads((folder / "core.json").read_text(encoding="utf-8")))
    assert (b.core.s.gd, b.core.stage().stage_id, b.core.s.branch, b.core.log.head) == (
        a.core.s.gd,
        a.core.stage().stage_id,
        a.core.s.branch,
        a.core.log.head,
    )
    # its mind, brought into a fresh brain on B as the server does: the people and the tricks
    original = FakeBrain()
    snap.apply(original, snap.read(record_a.neural_path), seed=SEED, context=HOME_CTX, wall=0.0)
    moved = FakeBrain()
    report = snap.apply(
        moved, snap.read(folder / "neural.json"), seed=SEED, context=HOME_CTX, wall=0.0
    )
    assert report["names"] == ["小明"] and report["people"] == len(original.learn.people) == 1
    who_a, who_b = original.learn.person_named("小明"), moved.learn.person_named("小明")
    assert who_b.bond == pytest.approx(who_a.bond) and who_b.mean == pytest.approx(who_a.mean)
    for trick in ("bow", "shake", "spin", "flip"):
        assert moved.learn.tricks.proficiency(trick) == original.learn.tricks.proficiency(trick)
    assert moved.learn.tricks.proficiency("bow") > moved.learn.tricks.proficiency("shake") > 0
    assert moved.learn.cues == original.learn.cues
    # its calls: the same library (key and every clip), where B's server looks for it
    assert done["voice"] == {"installed": True, "key": exported["library"].name, "same": True}
    library_b = df.lineage_root() / exported["library"].parent.name / exported["library"].name
    assert library_b.parent.parent == tmp_path / "B" / "data" / "voices" / "lineage"
    assert LV.small_hash(library_b) == library_hash
    assert (
        LV.build_library(  # (B's server takes it as it is: nothing rebuilt)
            VL.Plan.from_dict(json.loads((library_b / LV.PLAN_FILE).read_text())["plan"]),
            tmp_path / "no-bank",
            df.lineage_root(),
        ).folder
        == library_b
    )


def test_the_home_stays_behind(exported: dict[str, Any]) -> None:
    manifest = exported["manifest"]
    assert manifest["left_behind"] == {
        "places": 1,
        "things": 1,
        "nest_spot": True,
        "map_cells": 1,
        "eye": True,
        "layouts": True,
    }
    with zipfile.ZipFile(io.BytesIO(exported["data"])) as zf:
        names = zf.namelist()
        neural = json.loads(zf.read(df.NEURAL))
    assert not [n for n in names if n.startswith(("duck/eye", "duck/layouts", "eye", "layouts"))]
    assert "duck/neural.prev.json" not in names
    learner = neural["learner"]
    assert learner["places"] == [] and learner["things"] == [] and learner["dock"] is None
    assert "map" not in neural and neural["context"] == {}
    # (and a fresh brain in the very same home learns it again from nothing)
    fresh = FakeBrain()
    report = snap.apply(fresh, neural, seed=SEED, context=HOME_CTX, wall=0.0)
    assert (report["places"], report["things"], fresh.learn.dock) == (0, 0, None)
    assert report["people"] == 1


def test_the_manifest_says_who_and_checks_every_part(exported: dict[str, Any]) -> None:
    manifest = exported["manifest"]
    duck = manifest["duck"]
    assert (manifest["format"], manifest["version"]) == (df.FORMAT, df.VERSION)
    assert (duck["name"], duck["seed"], duck["nature"]) == ("蛋黃", SEED, "explorer")
    assert duck["id"] == df.duck_id(SEED, BIRTH) and duck["people"] == 1
    assert duck["tricks"]["bow"] > duck["tricks"]["shake"] > 0 and duck["growth_day"] > 0
    assert "小明" not in json.dumps(manifest, ensure_ascii=False)  # (counts only in the manifest)
    with zipfile.ZipFile(io.BytesIO(exported["data"])) as zf:
        members = [n for n in zf.namelist() if n != df.MANIFEST]
        assert sorted(manifest["parts"]) == sorted(members)
        for name in members:
            assert manifest["parts"][name]["sha256"] == df._sha(zf.read(name))
    assert manifest["voice"]["embedded"] and manifest["voice"]["key"] == exported["library"].name


# ── never overwritten silently ─────────────────────────────────────────────────────────────


def test_import_never_overwrites_a_duck_and_replaces_only_with_a_backup(
    exported: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    computer(monkeypatch, tmp_path / "B")
    first = Path(df.install(exported["data"])["folder"])
    (first / "neural.json").write_text("the duck lived on here")  # (B's own life since)
    with pytest.raises(df.DuckFileError) as refused:
        df.install(exported["data"])
    assert str(first) in str(refused.value)
    assert sorted(p.name for p in df.lives_dir().iterdir()) == [first.name]
    # asked: the one here moves aside first, whole, and the file's duck takes its place
    done = df.install(exported["data"], replace=True, stamp="20261104-120000")
    backup = Path(done["backup"])
    assert done["replaced"] and backup.name == f"{first.name}.replaced-20261104-120000"
    assert (backup / "neural.json").read_text() == "the duck lived on here"
    assert Path(done["folder"]) == first and snap.read(first / "neural.json")["format"]
    # the duck a server lives is never replaced, asked or not
    with pytest.raises(df.DuckFileError):
        df.install(exported["data"], replace=True, running=first)
    # a folder named for it must not exist
    with pytest.raises(df.DuckFileError):
        df.install(exported["data"], into=first)


def test_a_voice_library_that_cannot_be_put_in_does_not_lose_the_duck(
    exported: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    computer(monkeypatch, tmp_path / "B")
    blocked = tmp_path / "B" / "voices-here-is-a-file"
    blocked.parent.mkdir(parents=True)
    blocked.write_text("not a folder")
    done = df.install(exported["data"], voices=blocked)
    assert done["ok"] and done["voice"]["installed"] is False
    assert (
        LifeRecord.load(Path(done["folder"])).name == "蛋黃"
    )  # (its server makes the library again)


def test_another_duck_or_a_taken_folder_name_is_never_written_over(
    exported: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    computer(monkeypatch, tmp_path / "B")
    duck = exported["manifest"]["duck"]["id"]
    # a duck of the same name and seed, hatched another day, is another duck
    twin = LifeRecord.create(
        df.lives_dir() / "twin",
        name="蛋黃",
        seed=SEED,
        birth=datetime(2026, 9, 2, 8, 0, tzinfo=TPE),
    )
    assert df.record_id(twin) != duck and df.find_same(duck) == []
    # the folder the import would take holds something else: it gets a folder of its own
    taken = df.lives_dir() / f"蛋黃-{duck[len('duck-') :][:8]}"
    taken.mkdir()
    (taken / "notes.txt").write_text("mine")
    folder = Path(df.install(exported["data"])["folder"])
    assert folder == taken.with_name(taken.name + "-2")
    assert (taken / "notes.txt").read_text() == "mine" and LifeRecord.load(
        twin.folder
    ).name == "蛋黃"


# ── refused, with nothing written ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("spoil", "why"),
    [
        pytest.param(lambda d: b"not a zip at all", "not_duckfile", id="not-a-zip"),
        pytest.param(
            lambda d: repack(d, {df.CORE: b'{"config": {}}'}), "checksum", id="changed-part"
        ),
        pytest.param(lambda d: repack(d, {df.NEURAL: None}), "missing", id="missing-part"),
        pytest.param(
            lambda d: repack(d, {"voice/x/extra.wav": b"{}"}), "bad_member", id="unlisted-part"
        ),
        pytest.param(lambda d: repack(d, {"duck/eye.json": b"{}"}), "bad_member", id="odd-part"),
        pytest.param(
            lambda d: repack(d, {"../../escape.json": b"{}"}), "bad_member", id="path-outside"
        ),
        pytest.param(
            lambda d: repack(d, {}, manifest=lambda m: {**m, "version": df.VERSION + 1}),
            "newer",
            id="newer-file",
        ),
        pytest.param(
            lambda d: repack(d, {}, manifest=lambda m: {**m, "format": "other"}),
            "not_duckfile",
            id="other-format",
        ),
        pytest.param(
            lambda d: resealed(d, {df.CORE: _with_content(d, table_hash="0000000000000000")}),
            "engine",
            id="other-engine",
        ),
        pytest.param(
            lambda d: resealed(d, {df.NEURAL: _neural_with(d, version=snap.SNAPSHOT_VERSION + 1)}),
            "newer",
            id="newer-snapshot",
        ),
        pytest.param(
            lambda d: resealed(d, {df.NEURAL: _neural_with(d, seed=1)}),
            "mismatch",
            id="another-ducks-mind",
        ),
        pytest.param(
            lambda d: repack(
                d, {}, manifest=lambda m: {**m, "duck": {**m["duck"], "id": "duck-x"}}
            ),
            "mismatch",
            id="another-ducks-id",
        ),
    ],
)
def test_a_damaged_changed_or_too_new_file_is_refused_and_nothing_is_written(
    exported: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spoil: Any, why: str
) -> None:
    computer(monkeypatch, tmp_path / "B")
    with pytest.raises(df.DuckFileError) as refused:
        df.install(spoil(exported["data"]))
    assert refused.value.args[0].key == f"web.duckfile.{why}"  # (said in words, six languages)
    assert not (tmp_path / "B").exists()


def _with_content(data: bytes, **content: str) -> bytes:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        saved = json.loads(zf.read(df.CORE))
    saved["content"] = {**saved["content"], **content}
    return json.dumps(saved, ensure_ascii=False).encode()


def _neural_with(data: bytes, *, version: int | None = None, seed: int | None = None) -> bytes:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        neural = json.loads(zf.read(df.NEURAL))
    if version is not None:
        neural["version"] = version
    if seed is not None:
        neural["duck"]["seed"] = seed
    return json.dumps(neural, ensure_ascii=False).encode()


# ── older versions are read through their migrations ───────────────────────────────────────


def test_a_file_with_an_older_snapshot_is_migrated_on_import(
    exported: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with zipfile.ZipFile(io.BytesIO(exported["data"])) as zf:
        neural = json.loads(zf.read(df.NEURAL))
    neural["version"] = 1
    neural["learner"].pop("tricks")  # (version 1 had no trick book)
    old = resealed(exported["data"], {df.NEURAL: json.dumps(neural).encode()})
    computer(monkeypatch, tmp_path / "B")
    folder = Path(df.install(old)["folder"])
    moved = snap.read(folder / "neural.json")
    assert moved["version"] == snap.SNAPSHOT_VERSION and "tricks" in moved["learner"]
    assert [p["name"] for p in moved["learner"]["people"]] == ["小明"]


def test_an_older_duck_file_version_is_read_through_its_migration(
    exported: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    older = repack(exported["data"], {}, manifest=lambda m: {**m, "version": 0, "parts_v0": 1})
    with pytest.raises(df.DuckFileError):  # (no way to read version 0 without its step)
        df.read(older)

    def v0_to_v1(
        manifest: dict[str, Any], members: df.Members
    ) -> tuple[dict[str, Any], df.Members]:
        manifest = {k: v for k, v in manifest.items() if k != "parts_v0"}
        return {**manifest, "version": 1}, members

    monkeypatch.setitem(df.MIGRATIONS, 0, v0_to_v1)
    read = df.read(older)
    assert read.manifest["version"] == df.VERSION and "parts_v0" not in read.manifest


# ── the command line ───────────────────────────────────────────────────────────────────────


def test_the_command_line_exports_inspects_and_imports(
    exported: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    out = tmp_path / "moving" / "蛋黃.duck"
    assert df.main(["export", str(exported["record"].folder), "-o", str(out)]) == 0
    assert out.stat().st_mode & 0o777 == 0o600
    assert (
        df.main(["export", str(exported["record"].folder), "-o", str(out)]) == 1
    )  # (never over a file)
    capsys.readouterr()
    computer(monkeypatch, tmp_path / "B")
    assert df.main(["inspect", str(out)]) == 0
    seen = json.loads(capsys.readouterr().out)
    assert seen["duck"]["name"] == "蛋黃" and seen["same_here"] == []
    assert df.main(["import", str(out)]) == 0
    put = json.loads(capsys.readouterr().out)
    assert LifeRecord.load(Path(put["folder"])).name == "蛋黃"
    assert df.main(["import", str(out)]) == 1  # (it is here now)
    assert "蛋黃" in capsys.readouterr().err


# ── the page's routes ──────────────────────────────────────────────────────────────────────


@pytest.fixture()
def server(exported: dict[str, Any], tmp_path: Path):
    """Computer A's server living the duck (its mind on screen)."""
    record: LifeRecord = exported["record"]
    wall = [BIRTH.timestamp() + 5 * 86400 + 7200]
    clock = DuckClock(wall=lambda: wall[0])
    config = Config(seed=SEED, name="蛋黃", tof=True, nest=False, start=record.birth.isoformat())
    state = SimState(config, None, record.core_path, life=record, clock=clock, attach_neural=True)
    host = LifeHost(record, state, clock, every_s=600.0, wall=lambda: wall[0])
    state.life_host = host
    brain = taught()
    brain.learn.tricks.praise("bow", 50)  # (learnt since the file on disk)
    host.driver = SimpleNamespace(
        spec=SimpleNamespace(life=True),
        lanes=SimpleNamespace(brain=brain),
        _tick_lock=threading.Lock(),
        start_at="mat",
    )
    httpd = SimServer(("127.0.0.1", 0), make_handler(state))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/api/life/duckfile", brain
    httpd.shutdown()
    httpd.server_close()


def _call(url: str, body: bytes | None = None, headers: dict[str, str] | None = None) -> Any:
    request = urllib.request.Request(url, body, headers or {})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read()


def test_the_page_exports_the_duck_as_it_is_now_and_imports_it_as_a_new_life(
    server: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url, brain = server
    status, _, body = _call(url)
    assert status == 200 and json.loads(body) == {"ok": True, "life": True}
    status, headers, data = _call(url + "/export")
    assert status == 200 and headers["Content-Type"] == "application/vnd.duckmind.duck+zip"
    assert "attachment" in headers["Content-Disposition"]
    # the mind on screen now (one more praise of 「鞠躬」 than the file on disk had)
    moved = FakeBrain()
    snap.apply(moved, df.read(data).neural, seed=SEED, context=HOME_CTX, wall=0.0)
    assert moved.learn.tricks.proficiency("bow") == brain.learn.tricks.proficiency("bow")

    sent = {"Content-Type": "application/octet-stream", df_api_header(): "1"}
    # the same duck as the one this server lives: shown, and never put in or replaced
    status, _, body = _call(url + "/inspect", data, sent)
    seen = json.loads(body)
    assert status == 200 and seen["ok"] and seen["running_here"] and seen["duck"]["people"] == 1
    status, _, body = _call(url + "/import?replace=1", data, sent)
    refused = json.loads(body)
    assert not refused["ok"] and refused["_i18n"]["error"]["k"] == "web.duckfile.running"
    # (the life folders of this computer: only the duck it lives)
    assert [p.name for p in df.lives_dir().iterdir()] == ["danhuang"]

    # a POST without the page's header (another site's page cannot send it) is refused
    status, _, body = _call(url + "/import", data, {"Content-Type": "application/octet-stream"})
    assert status == 403 and json.loads(body)["_i18n"]["error"]["k"] == "web.duckfile.bad_request"
    # a damaged file: refused in words
    status, _, body = _call(url + "/inspect", data[:-40], sent)
    assert status == 200 and not json.loads(body)["ok"]

    # the other computer's server (no duck of its own): nothing to export, and the file's duck goes
    # into a new life folder there
    computer(monkeypatch, tmp_path / "B")
    other = SimState(Config(seed=7, name="x", tof=True, nest=False), None, None)
    httpd = SimServer(("127.0.0.1", 0), make_handler(other))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        url_b = f"http://127.0.0.1:{httpd.server_address[1]}/api/life/duckfile"
        assert json.loads(_call(url_b)[2]) == {"ok": True, "life": False}
        status, _, body = _call(url_b + "/export")
        assert status == 409 and json.loads(body)["_i18n"]["error"]["k"] == "web.duckfile.no_life"
        status, _, body = _call(url_b + "/import", data, sent)
        done = json.loads(body)
        assert status == 200 and done["ok"], done
        assert Path(done["folder"]).parent == tmp_path / "B" / "lives"
        assert df.find_same(seen["duck"]["id"]) == [Path(done["folder"])]
    finally:
        httpd.shutdown()
        httpd.server_close()


def df_api_header() -> str:
    from duckmind.simweb.duckfile_api import HEADER

    return HEADER
