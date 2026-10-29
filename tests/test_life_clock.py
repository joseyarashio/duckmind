"""The duck's one clock (docs/design/GROWTH-G1-2026-10-10.md): real time by default, a speed factor
for tests; the page's time-of-day buttons move only the demo offset, never the calendar."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from duckmind.life.clock import TPE, DuckClock
from duckmind.simweb.neuro_live import DayClock


class Wall:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def test_the_calendar_is_the_real_time_and_runs_at_the_wall_clocks_pace() -> None:
    wall = Wall()
    clock = DuckClock(wall=wall)
    start = clock.calendar()
    assert start == datetime.fromtimestamp(wall.t, TPE) and start.utcoffset() == timedelta(hours=8)
    wall.t += 90.0
    assert clock.calendar() - start == timedelta(seconds=90)
    assert clock.rate == 1.0 and clock.local() == clock.calendar()


def test_a_speed_factor_makes_days_pass_faster_without_a_jump() -> None:
    wall = Wall()
    clock = DuckClock(datetime(2026, 10, 10, 23, 0, tzinfo=TPE), speed=3600.0, wall=wall)
    wall.t += 1.0
    assert clock.calendar() == datetime(2026, 10, 11, 0, 0, tzinfo=TPE)  # an hour a second
    clock.set_speed(60.0)  # a new pace from now on: the moment does not move
    assert clock.calendar() == datetime(2026, 10, 11, 0, 0, tzinfo=TPE)
    wall.t += 60.0
    assert clock.calendar() == datetime(2026, 10, 11, 1, 0, tzinfo=TPE)
    with pytest.raises(ValueError):
        DuckClock(speed=0.0)


def test_the_pages_time_of_day_choice_moves_the_lights_not_the_growth_calendar() -> None:
    wall = Wall()
    clock = DuckClock(datetime(2026, 10, 10, 14, 30, tzinfo=TPE), wall=wall)
    assert clock.minute() == 14 * 60 + 30
    clock.set(21 * 60 + 30)  # the page's 「晚上」
    assert abs(clock.minute() - (21 * 60 + 30)) < 1e-6
    assert clock.local().hour == 21
    assert clock.calendar() == datetime(2026, 10, 10, 14, 30, tzinfo=TPE)  # growth days read this
    wall.t += 120.0
    assert abs(clock.minute() - (21 * 60 + 32)) < 1e-6  # and it runs on from there
    clock.set(1 * 60)  # 「半夜」 from the afternoon: the nearer way round, a few hours ahead
    assert clock.local().hour == 1 and clock.calendar().hour == 14
    clock.clear_offset()
    assert clock.local() == clock.calendar()


def test_the_neural_day_clock_is_this_clock_and_keeps_its_old_face() -> None:
    wall = Wall(1_000_000.0)
    day = DayClock(wall=wall)  # the demo and the tests: today at 10:00
    assert isinstance(day, DuckClock) and day.minute() == 600.0
    assert day.view()["zh"].startswith("上午") and day.view()["rate"] == 1.0
    real = DayClock(None, wall=wall)  # the live duck: the real calendar
    assert real.calendar() == datetime.fromtimestamp(wall.t, TPE)


def test_a_test_can_put_the_calendar_anywhere() -> None:
    clock = DuckClock(wall=Wall())
    clock.jump_to(datetime(2027, 2, 28, 18, 37, tzinfo=TPE))
    assert clock.calendar() == datetime(2027, 2, 28, 18, 37, tzinfo=TPE)
    with pytest.raises(ValueError):
        clock.jump_to(datetime(2027, 2, 28, 18, 37))  # naive: refused
