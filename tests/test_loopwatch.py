"""Round 5 (2026-10-10): the live loop detector writes the last minute when `bump_stop` repeats at one place, once per 10 minutes at most, and changes nothing
about the body (it only reads what the driver tick already has)."""

from __future__ import annotations

import gzip
import json
import tempfile
from pathlib import Path

from test_neuro_live import drive, robot_state, stub_driver

from duckmind import paths
from duckmind.simweb import loopwatch as LW


def _watch(tmp_path: Path, **kw):
    lines: list[str] = []
    clock = {"t": 1000.0}
    watch = LW.LoopWatch(
        tmp_path / "loops",
        clock=lambda: clock["t"],
        stamp=lambda: "T",
        log=lines.append,
        sync=True,
        **kw,
    )
    return watch, lines, clock


def _tick(watch, t_s: float, reflex: str | None = None, x: float = 1.0, y: float = 2.0):
    return watch.feed(
        {"t_ms": int(t_s * 1000), "reflex": reflex, "behaviour": "turn_around", "pose": [x, y, 0.0]}
    )


def _bump(watch, t_s: float, **kw):
    """One bump_stop onset: the reflex on for 0.5 s, then off."""
    out = _tick(watch, t_s, LW.BUMP, **kw)
    _tick(watch, t_s + 0.5, None, **kw)
    return out


def test_three_bumps_in_ten_seconds_at_one_place_write_the_last_minute(tmp_path: Path) -> None:
    watch, lines, _ = _watch(tmp_path)
    for k in range(120):  # 70 s of life, then the loop
        _tick(watch, k * 0.5)
    assert _bump(watch, 70.0) is None and _bump(watch, 72.0) is None
    path = _bump(watch, 74.0)
    assert path == tmp_path / "loops" / "T.jsonl.gz" and path.exists()
    rows = [json.loads(line) for line in gzip.open(path, "rt", encoding="utf-8")]
    assert rows[0]["loop"]["bumps"] and len(rows[0]["loop"]["bumps"]) == 3
    body = rows[1:]
    assert (
        body[-1]["t_ms"] == 74_000 and body[0]["t_ms"] >= (74 - LW.KEEP_S) * 1000
    )  # the last 60 s, no more
    assert sum(1 for r in body if r["reflex"] == LW.BUMP) == 3
    assert len(lines) == 1 and "bump_stop" in lines[0] and str(path) in lines[0]


def test_bumps_far_apart_in_time_or_in_place_are_not_a_loop(tmp_path: Path) -> None:
    watch, lines, _ = _watch(tmp_path)
    for k, t in enumerate((10.0, 21.0, 32.0, 43.0)):  # 11 s apart
        assert _bump(watch, t) is None
    for k, t in enumerate((60.0, 62.0, 64.0)):  # 2 s apart but a metre from each other
        assert _bump(watch, t, x=1.0 + k * 1.0) is None
    assert watch.hits == 0 and not lines and not (tmp_path / "loops").exists()


def test_a_held_bump_stop_is_one_onset(tmp_path: Path) -> None:
    watch, _, _ = _watch(tmp_path)
    for k in range(40):  # the reflex stays on for 4 s: one onset, not forty
        assert _tick(watch, 5.0 + k * 0.1, LW.BUMP) is None
    assert len(watch.onsets) == 1


def test_at_most_one_file_per_ten_minutes(tmp_path: Path) -> None:
    watch, lines, clock = _watch(tmp_path)
    for t in (10.0, 12.0, 14.0):
        _bump(watch, t)
    assert len(watch.written) == 1
    for t in (30.0, 32.0, 34.0):  # a second loop 20 s later: counted, not written
        _bump(watch, t)
    assert watch.hits == 2 and len(watch.written) == 1 and len(lines) == 1
    clock["t"] += LW.MIN_GAP_S + 1
    watch.stamp = lambda: "T2"
    for t in (50.0, 52.0, 54.0):
        _bump(watch, t)
    assert len(watch.written) == 2 and watch.written[-1].name == "T2.jsonl.gz"


def test_a_folder_that_cannot_be_written_only_logs(tmp_path: Path) -> None:
    blocker = tmp_path / "loops"
    blocker.write_text("a file where the folder should be")
    watch, lines, _ = _watch(tmp_path)
    for t in (10.0, 12.0, 14.0):
        _bump(watch, t)  # must not raise
    assert len(lines) == 1 and "失敗" in lines[0]


def _loop_through(driver, at_s: float = 70.0) -> Path | None:
    """Three bump_stop onsets within 4 s at one place, fed to the driver's own loop watch (written in this thread)."""
    driver.loopwatch.sync = True
    driver.loopwatch.log = lambda line: None
    _bump(driver.loopwatch, at_s)
    _bump(driver.loopwatch, at_s + 2.0)
    return _bump(driver.loopwatch, at_s + 4.0)


def test_a_server_living_a_tmp_life_writes_its_captures_there_and_nothing_into_the_live_folder(
    tmp_path: Path, monkeypatch
) -> None:
    # (V8Seat 2026-11-06: a soak server started from a copy of the live duck's life wrote its captures into the live
    # duck's <data>/live/loops, the default folder then)
    from duckmind.duckcore.api import Config
    from duckmind.life.clock import DuckClock
    from duckmind.life.record import LifeRecord
    from duckmind.simweb.life_host import LifeHost
    from duckmind.simweb.server import SimState

    monkeypatch.setenv(paths.DATA_ENV, str(tmp_path / "data"))
    monkeypatch.delenv(paths.LOOP_ENV, raising=False)
    live = tmp_path / "data" / "live" / "loops"
    record = LifeRecord.create(tmp_path / "life", name="測試", seed=7)
    driver = stub_driver()
    LifeHost(record, SimState(Config(seed=7)), DuckClock()).attach(driver)
    path = _loop_through(driver)
    assert path is not None and path.parent == tmp_path / "life" / "loops" and path.exists()
    assert not live.exists()


def test_a_server_without_a_life_writes_into_a_folder_of_its_own_and_the_live_launch_names_the_live_one(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(paths.DATA_ENV, str(tmp_path / "data"))
    monkeypatch.delenv(paths.LOOP_ENV, raising=False)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    live = tmp_path / "data" / "live" / "loops"
    path = _loop_through(stub_driver())
    assert path is not None and path.exists() and not live.exists()
    assert path.parent == paths.loops_dir() and path.parent.name.startswith("duckmind-loops-")
    # the live launch names its folder (launch-8791.sh: DUCKMIND_LOOP_DIR=<data>/live/loops), life or not
    monkeypatch.setenv(paths.LOOP_ENV, str(live))
    path = _loop_through(stub_driver())
    assert path is not None and path.parent == live and path.exists()


def test_the_driver_feeds_the_detector_and_the_body_commands_are_the_same_with_it_broken() -> None:
    driver = stub_driver()
    sent = drive(driver, 2.0)
    assert len(driver.loopwatch.records) > 50
    rec = driver.loopwatch.records[-1]
    assert {"t_ms", "behaviour", "reflex", "policy", "request", "link", "applied"} <= set(rec)
    assert {"twist", "gate", "refused", "events", "do"} <= set(rec["link"])

    other = stub_driver()
    other.loopwatch.feed = lambda record: 1 / 0  # type: ignore[method-assign]
    assert drive(other, 2.0) == sent  # a detector that fails changes nothing the body gets
    other.tick(robot_state(10.0))


def test_the_file_is_written_by_a_background_thread_when_not_sync(tmp_path: Path) -> None:
    import time

    lines: list[str] = []
    watch = LW.LoopWatch(tmp_path / "loops", stamp=lambda: "B", log=lines.append)
    for t in (10.0, 12.0, 14.0):
        _bump(watch, t)
    deadline = time.time() + 5.0
    while not (tmp_path / "loops" / "B.jsonl.gz").exists() and time.time() < deadline:
        time.sleep(0.02)
    assert (tmp_path / "loops" / "B.jsonl.gz").exists()
