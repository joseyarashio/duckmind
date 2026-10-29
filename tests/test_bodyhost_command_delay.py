"""The body host's opt-in servo command delay (tr:gate 2026-09-28): no MuJoCo, fakes only."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from duckmind import bodyhost


class FakeWorld:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.data = type("D", (), {"ctrl": np.zeros(3)})()
        self.model = type("M", (), {"opt": type("O", (), {"timestep": 0.005})()})()
        self.acted: list[float] = []  # the target the servo acted on at each physics step

    def step(self, times: int = 1) -> None:
        for _ in range(times):
            self.acted.append(float(self.data.ctrl[0]))


class FakeBody:
    def __init__(self) -> None:
        self.world = FakeWorld()
        self.actuator_slice = np.array([0, 1, 2])


def test_delay_steps_from_the_environment() -> None:
    assert bodyhost.command_delay_steps(0.005, {}) == 0
    assert bodyhost.command_delay_steps(0.005, {bodyhost.COMMAND_DELAY_ENV: "20"}) == 4
    assert bodyhost.command_delay_steps(0.005, {bodyhost.COMMAND_DELAY_ENV: "0"}) == 0
    with pytest.raises(ValueError):
        bodyhost.command_delay_steps(0.005, {bodyhost.COMMAND_DELAY_ENV: "500"})


def test_targets_act_four_steps_late(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(bodyhost.COMMAND_DELAY_ENV, "20")
    host = bodyhost.Host(collision="allcollisions", lockstep=True, scene=bodyhost.Path("x"))
    body = FakeBody()
    world = body.world
    assert host.delay_steps(world) == 4
    for tick, target in enumerate((1.0, 2.0, 3.0)):
        held = host.held_targets(body)
        world.data.ctrl[:] = target  # what the official write does
        host.step_delayed(body, world, 4, held)
        assert len(world.acted) == 4 * (tick + 1)
    # tick 0: 4 steps on the held target (0) ; then each target acts exactly 4 steps (20 ms) late
    assert world.acted == [0.0] * 4 + [1.0] * 4 + [2.0] * 4


def test_no_delay_is_todays_behaviour(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(bodyhost.COMMAND_DELAY_ENV, raising=False)
    host = bodyhost.Host(collision="allcollisions", lockstep=True, scene=bodyhost.Path("x"))
    assert host.delay_steps(FakeBody().world) == 0
    assert host.held_targets(FakeBody()) is None
