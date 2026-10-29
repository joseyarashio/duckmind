"""Bug A (2026-10-09): the simulator's 20 ms servo command delay makes the walking network walk faster at the pace commands and
stop later than the brain's distance rules were tuned for (more contact with things: headless free lives, 24 runs, touch time
561 → 1149 samples). The body link asks for the command that gives the designed ground speed back
(:func:`duckmind.neuro.bodylink.delay_compensated_vx`)."""

from __future__ import annotations

import math
import os

import pytest

from duckmind.neuro import bodylink as BL


def test_the_pace_table_is_monotone_and_leaves_flights_and_slow_commands_alone() -> None:
    cmds = [c for c, _ in BL.DELAY_PACE_TABLE]
    outs = [t for _, t in BL.DELAY_PACE_TABLE]
    assert cmds == sorted(cmds) and outs == sorted(outs)
    assert all(
        t <= c + 1e-9 for c, t in BL.DELAY_PACE_TABLE
    )  # (the delayed walk is faster: it is asked for less)
    for v in (0.0, 0.2, 0.3, 0.349, 0.525, 0.55, 0.9):
        assert BL.delay_compensated_vx(v, 20.0) == v  # (outside the table: as it is)
    for v in (0.35, 0.4, 0.45, 0.5):
        assert BL.delay_compensated_vx(v, 0.0) == v  # (no delay: no change)
        full = BL.delay_compensated_vx(v, 20.0)
        half = BL.delay_compensated_vx(v, 10.0)
        assert full < half < v
        assert (
            BL.delay_compensated_vx(v, 40.0) == full
        )  # (a longer delay takes the whole table, no more)


def test_the_link_without_a_delay_sends_the_commands_it_always_did(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DUCKMIND_BODY_COMMAND_DELAY_MS", raising=False)
    assert BL.command_delay_ms() == 0.0 and BL.BodyLink().delay_ms == 0.0
    monkeypatch.setenv("DUCKMIND_BODY_COMMAND_DELAY_MS", "20")
    assert BL.BodyLink().delay_ms == 20.0
    monkeypatch.setenv("DUCKMIND_BODY_COMMAND_DELAY_MS", "junk")
    assert BL.command_delay_ms() == 0.0


def _cruise(delay_ms: float, vx: float) -> float:
    from duckmind.world.headless_duck import HeadlessDuck

    duck = HeadlessDuck(
        "default_home", keyframe="SIT", x=-2.0, y=-1.0, yaw=0.0, command_delay_ms=delay_ms
    )
    duck.stand()
    for _ in range(25):
        duck.tick()
    pos = []
    for _ in range(170):
        duck.move(vx, 0.0, 0.0)
        duck.tick()
        pos.append(duck.truth()["trunk"])
    return math.hypot(pos[-1][0] - pos[-71][0], pos[-1][1] - pos[-71][1]) / (70 * 0.02)


def test_the_delay_walks_faster_and_the_compensation_gives_the_designed_speed_back() -> None:
    pytest.importorskip("mujoco")
    folder = os.environ.get("DUCKMIND_MICRODUCK_MJCF")
    if not folder or not os.path.isdir(folder):
        pytest.skip("DUCKMIND_MICRODUCK_MJCF (the official MJCF folder) is not set")
    for vx in (0.35, 0.40, 0.50):
        designed = _cruise(0.0, vx)
        delayed = _cruise(20.0, vx)
        compensated = _cruise(20.0, BL.delay_compensated_vx(vx, 20.0))
        assert delayed > 1.08 * designed, (
            vx,
            designed,
            delayed,
        )  # (the problem: faster with the delay)
        assert abs(compensated - designed) <= 0.06 * designed, (vx, designed, compensated)
