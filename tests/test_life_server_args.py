"""The server's life flags (``--life``, ``--no-nest``, ``--tof``, ``--clock-speed``): a new duck is hatched
from them, an existing one keeps what it hatched with, and a life lives by the calendar."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pytest

from duckmind.life.clock import TPE
from duckmind.life.record import LifeRecord
from duckmind.simweb.server import _make_clock, _open_life


def _args(folder: Path | None, **over) -> argparse.Namespace:
    base = dict(
        life=folder,
        seed=4107,
        name="蛋黃",
        name_given="蛋黃",
        neuro_duck="explorer",
        neuro_time=None,
        no_nest=True,
        tof=True,
        life_birth=None,
        life_backdate_days=0,
        life_eye_from=None,
        expression_caps="",
        clock_speed=1.0,
    )
    base.update(over)
    return argparse.Namespace(**base)


def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(prog="test")


def test_no_life_flag_is_the_simulator_as_it_was() -> None:
    assert _open_life(_args(None), _parser()) == (None, None)


def test_a_new_life_is_hatched_from_the_flags_and_a_known_one_keeps_what_it_hatched_with(
    tmp_path, capsys
) -> None:
    folder = tmp_path / "danhuang"
    birth = "2026-10-10T15:00:00+08:00"
    life, config = _open_life(_args(folder, life_birth=birth), _parser())
    assert life is not None and config is not None
    assert (life.name, life.seed, life.nest, life.tof, life.nature) == (
        "蛋黃",
        4107,
        False,
        True,
        "explorer",
    )
    assert life.birth == datetime(2026, 10, 10, 15, 0, tzinfo=TPE)
    assert (config.seed, config.nest, config.tof, config.start) == (4107, False, True, birth)
    assert "新的一生" in capsys.readouterr().err
    # started again with other flags: the hardware it hatched with stays (said, not obeyed)
    again, config2 = _open_life(_args(folder, no_nest=False, tof=False), _parser())
    assert again is not None and config2 is not None
    assert (again.nest, again.tof, config2.nest, config2.tof) == (False, True, False, True)
    assert "不改牠" in capsys.readouterr().err
    assert LifeRecord.load(folder).birth == life.birth


def test_a_life_refuses_a_test_start_time_and_the_clock_has_a_speed() -> None:
    with pytest.raises(SystemExit):
        _open_life(_args(Path("/nonexistent/x"), neuro_time="10:00"), _parser())
    clock = _make_clock(_args(None, clock_speed=3600.0), neuro=False)
    assert clock.speed == 3600.0
    with pytest.raises(SystemExit):
        _make_clock(_args(None, clock_speed=0.0), neuro=False)
    day = _make_clock(_args(None, neuro_time="07:30"), neuro=True)
    assert abs(day.minute() - 450.0) < 1.0  # the tests' fixed start, as it always was
    live = _make_clock(_args(None), neuro=True)
    assert abs((live.calendar() - datetime.now(TPE)).total_seconds()) < 2.0  # real time otherwise
