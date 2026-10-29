"""Growth G4: the fast-forward of throwaway test ducks never touches a real duck's folder, runs one
job at a time in its own process, and gives a growth view in the same shape as the live duck's
(docs/design/GROWTH-G4-2026-10-11.md)."""

from __future__ import annotations

import json
import os
import stat
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

import duckmind.world.legacy  # noqa: F401  (registers LegacyWorld)
from duckmind.duckcore.api import Config
from duckmind.life import fastforward as ff
from duckmind.simweb import homeview
from duckmind.simweb.server import SimServer, SimState, make_handler
from tests.checkout_parts import needs_tool

needs_growth = needs_tool("growth")  # the fast-forward's worker harness


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A home of its own: ``~/.duckmind`` is a folder of this test, never the real one."""
    home = tmp_path / "home"
    (home / ".duckmind" / "lives" / "danhuang").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    return home


def _stub(tmp_path: Path, seconds: int = 60) -> str:
    """A 'python' that ignores its arguments and sleeps: a worker that is still running."""
    path = tmp_path / "stub-python.sh"
    path.write_text(f"#!/bin/sh\nsleep {seconds}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def _live_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "lives" / "danhuang"
    folder.mkdir(parents=True)
    (folder / "life.json").write_text("{}")
    (folder / "core.json").write_text("{}")
    return folder


# ── the folder rule ─────────────────────────────────────────────────────────────────────────


def test_a_real_ducks_folder_is_refused_whatever_way_it_is_named(
    tmp_path: Path, _home: Path
) -> None:
    live = _live_folder(tmp_path)
    protected = [live]
    for bad in (
        live,  # the folder itself
        live / "run-attentive",  # inside it
        live.parent,  # the folder that holds it
        tmp_path,  # a folder further up
        live / ".." / "danhuang",  # the same folder by another path
        _home / ".duckmind" / "lives" / "danhuang",  # the default home of the lives
        _home / ".duckmind",
    ):
        with pytest.raises(ff.Refused):
            ff.check_workdir(bad, protected)


def test_a_folder_that_looks_like_a_duck_is_refused_even_when_nobody_listed_it(
    tmp_path: Path,
) -> None:
    other = tmp_path / "somebodys-duck"
    other.mkdir()
    (other / "life.json").write_text("{}")
    with pytest.raises(ff.Refused):
        ff.check_workdir(other, [])
    with pytest.raises(ff.Refused):
        ff.check_workdir(other / "scratch", [])
    nested = tmp_path / "lives2"
    (nested / "one").mkdir(parents=True)
    (nested / "one" / "core.json").write_text("{}")
    with pytest.raises(ff.Refused):
        ff.check_workdir(nested, [])
    assert ff.check_workdir(tmp_path / "fresh", []) == (tmp_path / "fresh").resolve()


def test_a_symlink_to_a_real_ducks_folder_is_still_that_folder(tmp_path: Path) -> None:
    live = _live_folder(tmp_path)
    link = tmp_path / "innocent"
    link.symlink_to(live, target_is_directory=True)
    with pytest.raises(ff.Refused):
        ff.check_workdir(link, [live])


@needs_growth
def test_the_manager_refuses_the_live_life_folder_and_starts_nothing(tmp_path: Path) -> None:
    live = _live_folder(tmp_path)
    manager = ff.FastForward([live], python=_stub(tmp_path))
    before = sorted(p.name for p in live.iterdir())
    with pytest.raises(ff.Refused) as error:
        manager.start(days=3, styles=["attentive"], workdir=live)
    assert "danhuang" in str(error.value)  # (it says which folder)
    with pytest.raises(ff.Refused):
        manager.start(days=3, styles=["attentive"], workdir=live / "ff")
    assert manager.status()["state"] == "idle"
    assert sorted(p.name for p in live.iterdir()) == before  # not a file added or changed
    assert (live / "core.json").read_text() == "{}"


@needs_growth
def test_the_cli_refuses_the_live_life_folder_too(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    live = _live_folder(tmp_path)
    assert ff.main(["--days", "1", "--out", str(live), "--protect", str(live)]) == 3
    assert (
        ff.main(["--days", "1", "--out", str(live)]) == 3
    )  # (it looks like a duck's: no list needed)
    assert capsys.readouterr().err.count("danhuang") == 2  # (each refusal says which folder)
    assert (live / "core.json").read_text() == "{}"


def test_the_worker_checks_its_own_folder_again(tmp_path: Path) -> None:
    live = _live_folder(tmp_path)
    with pytest.raises(ff.Refused):
        ff.run_worker("attentive", 1, 4107, live / "run-attentive", [live])
    assert not (live / "run-attentive").exists()


def test_requests_outside_the_limits_are_invalid() -> None:
    assert ff.check_request(30, ["attentive", "rough"], 4107) == (30, ["attentive", "rough"], 4107)
    assert ff.check_request(1, "praising", 0)[1] == ["praising"]
    assert ff.check_request(5, ["rough", "rough"], 1)[1] == ["rough"]
    for days, styles, seed in (
        (0, ["rough"], 1),
        (ff.MAX_DAYS + 1, ["rough"], 1),
        ("30", ["rough"], 1),
        (True, ["rough"], 1),
        (3, [], 1),
        (3, ["rough", "attentive", "praising"], 1),
        (3, ["shouting"], 1),
        (3, ["rough"], -1),
        (3, ["rough"], ff.MAX_SEED + 1),
        (3, ["rough"], 1.5),
    ):
        with pytest.raises(ff.Invalid):
            ff.check_request(days, styles, seed)


# ── one at a time, in its own process, cancelled by its own pid ──────────────────────────────


@needs_growth
def test_one_job_at_a_time_and_cancel_stops_exactly_its_workers(tmp_path: Path) -> None:
    manager = ff.FastForward([], python=_stub(tmp_path), tmp_root=tmp_path / "tmp")
    status = manager.start(days=3, styles=["attentive", "rough"])
    assert status["state"] == "running" and [r["style"] for r in status["runs"]] == [
        "attentive",
        "rough",
    ]
    job = manager._job
    assert job is not None
    pids = [run.proc.pid for run in job.runs]
    assert all(pid != os.getpid() for pid in pids)
    with pytest.raises(ff.Busy):
        manager.start(days=3, styles=["attentive"])
    folder = job.workdir
    assert folder.parent == (tmp_path / "tmp").resolve() and folder.name.startswith("duckmind-ff-")
    assert manager.cancel() is True
    deadline = time.time() + 15
    while manager.status()["state"] == "running" and time.time() < deadline:
        time.sleep(0.1)
    done = manager.status()
    assert done["state"] == "cancelled" and {r["state"] for r in done["runs"]} == {"cancelled"}
    assert all(run.proc.poll() is not None for run in job.runs)  # (both workers are gone)
    assert not folder.exists()  # (its temporary folder is removed)
    assert manager.cancel() is False  # (nothing running now)
    again = manager.start(days=2, styles=["rough"])  # (a new job after the last one)
    assert again["state"] == "running"
    manager.shutdown()
    assert manager.status()["state"] == "cancelled"


@needs_growth
def test_a_worker_that_dies_is_a_failed_run_with_its_last_line(tmp_path: Path) -> None:
    script = tmp_path / "dies.sh"
    script.write_text("#!/bin/sh\necho 'it broke: no numpy'\nexit 7\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    manager = ff.FastForward([], python=str(script), tmp_root=tmp_path / "tmp")
    manager.start(days=2, styles=["rough"])
    deadline = time.time() + 15
    while manager.status()["state"] == "running" and time.time() < deadline:
        time.sleep(0.1)
    status = manager.status(results=True)
    assert status["state"] == "failed" and status["runs"][0]["state"] == "failed"
    assert "no numpy" in status["runs"][0]["error"] and "results" not in status
    manager.shutdown()


# ── a real, short run (the headless harness, in a worker process) ────────────────────────────


@needs_growth
def test_a_short_real_fast_forward_gives_a_growth_view_in_a_temporary_folder(
    tmp_path: Path,
) -> None:
    live = _live_folder(tmp_path)
    manager = ff.FastForward([live], python=sys.executable, tmp_root=tmp_path / "tmp")
    manager.start(days=3, styles=["attentive", "rough"], seed=4107)
    job = manager._job
    assert job is not None
    folder = job.workdir
    deadline = time.time() + 240
    while manager.status()["state"] == "running" and time.time() < deadline:
        time.sleep(0.5)
    status = manager.status(results=True)
    assert status["state"] == "done", status
    assert status["speed"]["days"] == 6 and status["speed"]["days_per_s"] > 0
    assert not folder.exists() and (live / "core.json").read_text() == "{}"
    results = status["results"]
    assert set(results) == {"attentive", "rough"}
    for style, result in results.items():
        view = result["view"]
        assert view["api"] == "growth-view-v1" and view["live"] is False and view["style"] == style
        assert view["identity"]["growth_day"] == 3 and view["identity"]["name"] == ""
        assert view["identity"]["stage"]["id"] == "hatchling"
        assert (
            view["identity"]["stage"]["_i18n"]["label"]["k"] == "core.content.STAGES.hatchling_zh"
        )
        assert view["regrow"]["count"] == result["regrows"] >= 1
        assert view["regrow"]["genome"]["gd"] in (2, 3)
        assert view["experiences"]["total"] > 0
        assert view["voice"]["calls"] == [] or all(
            c["playable"] is False for c in view["voice"]["calls"]
        )
        json.dumps(view)
    # the two care styles lived different lives
    a, b = (results[s]["view"] for s in ("attentive", "rough"))
    assert a["experiences"]["by_kind"] != b["experiences"]["by_kind"]
    manager.shutdown()


# ── the endpoint ────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def server(tmp_path: Path):
    homeview.set_home("real_home")
    state = SimState(Config(seed=7, name="測試"))
    state.fastforward = ff.FastForward([], python=_stub(tmp_path), tmp_root=tmp_path / "tmp")
    httpd = SimServer(("127.0.0.1", 0), make_handler(state))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", state
    state.fastforward.shutdown()
    httpd.shutdown()
    httpd.server_close()
    homeview.set_home(None)


def _call(
    url: str, body: dict[str, Any] | None = None, method: str | None = None
) -> tuple[int, dict]:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method=method
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


@needs_growth
def test_the_page_starts_watches_and_cancels_a_fast_forward(server) -> None:
    base, state = server
    code, idle = _call(f"{base}/api/life/fastforward")
    assert code == 200 and idle["fastforward"]["state"] == "idle"
    assert idle["fastforward"]["limits"]["styles"] == list(ff.STYLES)
    code, bad = _call(f"{base}/api/life/fastforward", {"days": 0, "styles": ["rough"], "seed": 1})
    assert (
        code == 400
        and bad["ok"] is False
        and bad["_i18n"]["error"]["k"] == "web.growth.ff.bad_days"
    )
    code, bad = _call(f"{base}/api/life/fastforward", {"days": 5, "styles": ["x"], "seed": 1})
    assert code == 400 and bad["_i18n"]["error"]["k"] == "web.growth.ff.bad_styles"
    code, started = _call(
        f"{base}/api/life/fastforward", {"days": 5, "styles": ["rough", "attentive"], "seed": 4107}
    )
    assert code == 200 and started["fastforward"]["state"] == "running"
    code, busy = _call(
        f"{base}/api/life/fastforward", {"days": 5, "styles": ["rough"], "seed": 4107}
    )
    assert code == 409 and busy["_i18n"]["error"]["k"] == "web.growth.ff.busy"
    code, running = _call(f"{base}/api/life/fastforward")
    assert running["fastforward"]["state"] == "running" and len(running["fastforward"]["runs"]) == 2
    code, cancelled = _call(f"{base}/api/life/fastforward/cancel", {}, "POST")
    assert code == 200
    deadline = time.time() + 15
    while (
        _call(f"{base}/api/life/fastforward")[1]["fastforward"]["state"] == "running"
        and time.time() < deadline
    ):
        time.sleep(0.1)
    assert _call(f"{base}/api/life/fastforward")[1]["fastforward"]["state"] == "cancelled"
    # the button names no folder: whatever else the body says, the workers live in the temporary one
    code, _ = _call(
        f"{base}/api/life/fastforward", {"days": 1, "styles": ["rough"], "seed": 1, "workdir": "/"}
    )
    assert code == 200
    job = state.fastforward._job
    assert job is not None and job.workdir.name.startswith("duckmind-ff-")


@needs_growth
def test_the_servers_manager_is_given_the_life_folder_to_keep_away_from(tmp_path: Path) -> None:
    from datetime import datetime, timedelta, timezone

    from duckmind.life.record import LifeRecord

    birth = datetime(2026, 10, 1, 8, 0, tzinfo=timezone(timedelta(hours=8)))
    record = LifeRecord.create(
        tmp_path / "life",
        name="蛋黃",
        seed=4107,
        birth=birth,
        nest=False,
        tof=True,
        backdate_days=0,
    )
    state = SimState(
        Config(seed=4107, name="蛋黃", tof=True, nest=False, start=birth.isoformat()),
        None,
        record.core_path,
        life=record,
    )
    httpd = SimServer(("127.0.0.1", 0), make_handler(state))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        code, _ = _call(f"{base}/api/life/fastforward")
        assert code == 200
        manager = state.fastforward
        assert manager is not None and record.folder in manager.protected
        with pytest.raises(ff.Refused):
            manager.start(days=1, styles=["rough"], workdir=record.folder)
        with pytest.raises(ff.Refused):
            manager.start(days=1, styles=["rough"], workdir=record.folder / "ff")
    finally:
        httpd.shutdown()
        httpd.server_close()
