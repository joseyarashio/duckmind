"""The numbers of tools/machine_verify/measure_unstuck.py (pushing seconds, stuck episodes, escape
times, contacts, falls) from 5 Hz samples — pure, no MuJoCo."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

if not Path("tools/machine_verify/measure_unstuck.py").is_file():
    pytest.skip("tools/machine_verify is not in this checkout", allow_module_level=True)
sys.path.insert(0, "tools/machine_verify")

import measure_unstuck as M  # noqa: E402


def _sample(
    i: int, x: float, *, cmd: float = 0.0, want: float = 0.0, gap: float = 1.0, fallen: bool = False
):
    return {
        "t": round(i * M.SAMPLE_S, 2),
        "x": x,
        "y": 0.0,
        "cmd_vx": cmd,
        "b_vx": want,
        "gap": gap,
        "near": "wall",
        "fallen": fallen,
    }


def test_walking_along_is_not_pushing() -> None:
    out = M.analyse([_sample(i, 0.04 * i, cmd=0.4, want=0.4, gap=0.05) for i in range(100)])
    assert out["push_s"] == 0.0 and out["n_episodes"] == 0


def test_a_commanded_walk_that_goes_nowhere_next_to_something_is_pushing_and_one_episode() -> None:
    samples = [_sample(i, 0.04 * i, cmd=0.4, want=0.4) for i in range(10)]
    samples += [_sample(10 + i, 0.4, cmd=0.4, want=0.4, gap=0.0) for i in range(40)]  # 8 s pressed
    samples += [_sample(50 + i, 0.4 + 0.05 * i, cmd=0.4, want=0.4) for i in range(30)]  # then away
    out = M.analyse(samples)
    assert 6.5 <= out["push_s"] <= 8.0
    assert out["n_episodes"] == 1
    episode = out["episodes"][0]
    assert episode["kind"] == "push" and episode["dur_s"] >= 6.0
    assert episode["escape_s"] is not None and 8.0 <= episode["escape_s"] <= 14.0


def test_a_request_the_body_link_holds_back_is_wedged_not_pushing() -> None:
    out = M.analyse([_sample(i, 0.0, cmd=0.0, want=0.4, gap=0.1) for i in range(60)])
    assert out["push_s"] == 0.0 and out["wedged_s"] > 8.0


def test_staying_inside_a_small_circle_for_fifteen_seconds_while_walking_is_stuck() -> None:
    samples = [_sample(i, 0.05 * ((i % 4) - 2), cmd=0.4, want=0.4) for i in range(100)]  # 20 s
    out = M.analyse(samples)
    assert out["n_episodes"] == 1 and out["episodes"][0]["kind"] == "confined"
    assert out["episodes"][0]["escape_s"] is None  # it never got away: censored


def test_falls_and_contacts_are_counted_once_each() -> None:
    samples = [_sample(i, 0.04 * i, cmd=0.4, want=0.4) for i in range(50)]
    for i in (10, 11, 12):
        samples[i]["fallen"] = True
    for i in (20, 21, 40):
        samples[i]["gap"] = 0.0
    out = M.analyse(samples)
    assert out["falls"] == 1
    assert out["contacts"] == 2  # 20–21 together, 40 apart (more than 1 s)


def test_a_duck_the_body_link_holds_at_a_wall_while_it_asks_to_walk_is_stuck_too() -> None:
    samples = [_sample(i, 0.0, cmd=0.0, want=0.4, gap=0.1) for i in range(100)]  # 20 s
    out = M.analyse(samples)
    assert out["n_episodes"] == 1 and out["episodes"][0]["kind"] == "confined"
    assert out["push_s"] == 0.0  # …nothing was pushed: the link let no walk through


def _basking(n: int, beh: str | None) -> list[dict]:
    samples = [_sample(i, 0.05 * ((i % 4) - 2), cmd=0.4, want=0.4) for i in range(n)]
    for s in samples:
        s["beh"] = beh
    return samples


def test_the_steps_of_an_act_that_stays_on_its_spot_on_purpose_are_not_confined() -> None:
    sunbathe = _basking(100, "sunbathe")  # 20 s inside a small circle, walk commanded all along
    assert M.analyse(sunbathe)["n_episodes"] == 0
    # … what the first definition (the stored runs before this change) counted
    old = M.analyse(sunbathe, stationary=frozenset())
    assert old["n_episodes"] == 1 and old["episodes"][0]["kind"] == "confined"
    # any other act in the same small circle is still stuck
    assert M.analyse(_basking(100, "wander"))["n_episodes"] == 1


def test_pushing_is_still_counted_in_an_act_that_stays_on_its_spot() -> None:
    samples = [_sample(i, 0.4, cmd=0.4, want=0.4, gap=0.0) for i in range(40)]  # 8 s pressed
    for s in samples:
        s["beh"] = "sunbathe"
    out = M.analyse(samples)
    assert out["push_s"] > 6.0 and out["episodes"][0]["kind"] == "push"
