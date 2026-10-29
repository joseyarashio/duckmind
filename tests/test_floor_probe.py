"""The re-test of a lift claim that persists (duckmind.neuro.liftcheck.FloorProbe; DOCK-FALL-2026-10-11.md §7.6): pure, on a clock."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from duckmind.neuro.liftcheck import PROBE_RESPONSE, FloorProbe

Head = tuple[float, float, float, float]


def drive(
    load_at: Callable[[float], float | None],
    *,
    gyro: float = 0.02,
    reach: bool = True,
    ended_at: float | None = None,
) -> tuple[str | None, FloorProbe]:
    """The probe on a 50 Hz clock; the measured head follows the probe's own pose (or never leaves HOME); the load is ``load_at(up)``
    with ``up`` how far up the head is (0..1)."""
    probe = FloorProbe(0)
    head = [0.0, 0.0, 0.0, 0.0]
    for k in range(1500):
        t = k * 20
        goal = probe.head() if reach else (0.0, 0.0, 0.0, 0.0)
        head = [h + (g - h) * 0.1 for h, g in zip(head, goal, strict=True)]
        up = min(1.0, max(0.0, -head[1] / 0.35))
        verdict = probe.step(
            t,
            load_at(up),
            (head[0], head[1], head[2], head[3]),
            gyro,
            ended=ended_at is not None and t >= ended_at,
        )
        if verdict is not None:
            return verdict, probe
    return None, probe


def test_legs_that_carry_the_trunk_answer_the_raised_head() -> None:
    for base, response in ((0.44, 0.25), (0.40, 0.17), (0.49, 0.15)):
        verdict, probe = drive(lambda up, b=base, r=response: b * (1.0 + r * up))
        assert verdict == "supported", (base, response)
        assert probe.base_load == pytest.approx(base, rel=0.02)


def test_legs_that_carry_nothing_do_not_answer() -> None:
    assert drive(lambda up: 0.387 * (1.0 - 0.005 * up))[0] == "held"  # a hand
    assert drive(lambda up: 0.387)[0] == "held"
    assert drive(lambda up: 0.70 * (1.0 + 0.013 * up))[0] == "held"  # the floor's own ripple


def test_the_answer_is_judged_at_its_lowest_read_so_a_bump_is_no_answer() -> None:
    def bump(up: float) -> float:
        return 0.387 * (
            1.0 + (0.4 if 0.2 < up < 0.35 else 0.0)
        )  # one jolt while the head swings up

    assert drive(bump)[0] in ("held", "unclear")


def test_nothing_is_judged_when_it_cannot_be() -> None:
    assert drive(lambda up: None)[0] == "unclear"  # no load
    assert drive(lambda up: 0.44, reach=False)[0] == "unclear"  # the head never got there
    assert (
        drive(lambda up: 0.44 * (1.0 + 0.25 * up), ended_at=600)[0] == "unclear"
    )  # the claim ended
    assert (
        drive(lambda up: 0.44 * (1.0 + 0.25 * up), gyro=0.6)[0] == "unclear"
    )  # a trunk that is carried
    swing = 0
    flip = {"n": 0}

    def noisy(up: float) -> float:
        flip["n"] += 1
        return 0.44 + (0.12 if (flip["n"] // 7) % 2 else -0.12)

    assert drive(noisy)[0] == "unclear" and swing == 0  # a load that will not keep still


def test_the_response_threshold_sits_between_the_ledge_and_the_hand() -> None:
    assert (
        0.0 < PROBE_RESPONSE < 0.146
    )  # the lowest ledge read (+14.6 %) is well above it, a hand's -0.5 % well below
