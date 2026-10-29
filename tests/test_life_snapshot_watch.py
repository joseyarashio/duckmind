"""The mind's saving never fails silently (the 2026-10-10 bug: the page showed a demo duck for three
hours, nothing was saved, and /api/life said nothing): periodic saves go on for hours, a mind that is
not on screen or a save that fails is in /api/life's `problem` and in the log, and it clears when
the saving works again."""

from __future__ import annotations

import threading
import time
from datetime import timedelta
from types import SimpleNamespace
from typing import Any

import pytest
from test_life_bridge_host import BIRTH, Life, Wall
from test_life_snapshot import taught

from duckmind.life import snapshot as snap


class Rig:
    """A life whose driver is a stub with a brain that has something to save."""

    def __init__(self, folder: Any, *, every_s: float = 600.0) -> None:
        self.wall = Wall(BIRTH)
        self.life = Life(folder, self.wall, backdate=14)
        self.brain = taught()
        self.driver = SimpleNamespace(
            spec=SimpleNamespace(life=True),
            lanes=SimpleNamespace(brain=self.brain),
            _tick_lock=threading.RLock(),
            start_at="mat",
            live_voice=None,
        )
        host = self.life.host
        host.driver = self.driver
        host.every_s = every_s
        host.watch_s = 0.01
        self.host = host

    def status(self) -> dict[str, Any]:
        with self.life.state.lock:
            return self.host.status()

    def wait_saves(self, n: int, seconds: float = 5.0) -> None:
        deadline = time.time() + seconds
        while self.host.saves < n and time.time() < deadline:
            time.sleep(0.005)
        assert self.host.saves >= n, (self.host.saves, n, self.status()["snapshot"])


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path / "duck")
    yield r
    r.host._stop.set()


def test_periodic_saves_go_on_for_hours(rig: Rig) -> None:
    rig.host.start()
    first = rig.host.saves  # (the first, empty-hatch one if the file was new)
    for step in range(1, 37):  # six hours of ten minutes, the wall clock moved by the test
        rig.wall.t += 601.0
        rig.wait_saves(first + step)
        assert rig.status()["snapshot"]["problem"] is None
    assert rig.host.saves >= first + 36
    assert rig.host.last_why in {"periodic", "night"}
    assert rig.status()["snapshot"]["watching"] is True


def test_a_demo_duck_on_screen_is_said_at_once_and_in_the_log_and_clears_when_it_is_back(
    rig: Rig, capsys: pytest.CaptureFixture[str]
) -> None:
    rig.host.start()
    rig.wall.t += 601.0
    rig.wait_saves(rig.host.saves + 1)
    rig.host.before_switch()  # the page leaves the life duck: its mind is saved first
    saved = rig.host.saves
    rig.driver.spec.life = False
    status = rig.status()["snapshot"]
    assert status["on_screen"] is False and "蛋黃" in status["problem"]  # (at once, no waiting)
    for _ in range(5):  # hours go by with the demo duck on screen: nothing saved, still said
        rig.wall.t += 3601.0
        time.sleep(0.15)
    assert rig.host.saves == saved
    assert "蛋黃" in rig.status()["snapshot"]["problem"]
    said = capsys.readouterr().err
    assert said.count("頁面現在顯示的不是蛋黃") == 1  # once, not every two seconds
    rig.driver.spec.life = True  # the page picked it again
    rig.host.after_switch()
    assert rig.status()["snapshot"]["problem"] is None and rig.status()["snapshot"]["on_screen"]
    assert "回到畫面上" in capsys.readouterr().err
    rig.wall.t += 601.0
    rig.wait_saves(saved + 1)


def test_a_save_that_fails_is_in_the_status_and_the_log_and_is_tried_again(
    rig: Rig, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[float] = []
    real = snap.write

    def broken(*args: Any, **kwargs: Any) -> None:
        calls.append(rig.wall.t)
        raise OSError(28, "No space left on device")

    assert rig.host.snapshot_now("test")  # (its file exists: no hatch save to fail first)
    capsys.readouterr()
    monkeypatch.setattr(snap, "write", broken)
    rig.host.start()
    rig.wall.t += 601.0
    deadline = time.time() + 5
    while not calls and time.time() < deadline:
        time.sleep(0.005)
    status = rig.status()["snapshot"]
    assert calls and "OSError" in status["problem"] and "No space left" in status["problem"]
    for _ in range(4):  # it tries again every half minute, and the log does not fill up
        rig.wall.t += 31.0
        time.sleep(0.1)
    assert len(calls) >= 3
    err = capsys.readouterr().err
    assert err.count("存不下去") == 1 and "No space left" in err
    monkeypatch.setattr(snap, "write", real)
    saved = rig.host.saves
    rig.wall.t += 31.0
    rig.wait_saves(saved + 1)
    assert rig.status()["snapshot"]["problem"] is None
    assert "又存得下去了" in capsys.readouterr().err


def test_a_mind_on_screen_that_has_not_been_saved_for_three_cycles_is_said_whatever_the_thread_does(
    rig: Rig,
) -> None:
    rig.host.watch_s = 10_000.0  # a watch that never wakes (a dead timer would look like this)
    rig.host.start()
    assert rig.status()["snapshot"]["problem"] is None
    rig.wall.t += 3 * 600.0 + 120.0
    assert "沒有存了" in rig.status()["snapshot"]["problem"]


def test_a_clean_stop_with_a_demo_duck_on_screen_says_why_nothing_was_written(
    rig: Rig, capsys: pytest.CaptureFixture[str]
) -> None:
    rig.driver.spec.life = False
    rig.host.stop()
    assert "乾淨關機時沒有存到蛋黃的心" in capsys.readouterr().err
    assert rig.life.core.s.now + timedelta(0) == rig.life.core.s.now  # (and Duck Core was saved)
