"""A tiny run of the headless go-there runner (duckmind.bench.runner) on the real brain and the real physics: one click
on the brain's own steering and on the truth planner's, the row schema, and the same click twice giving the same row.

Needs numpy, MuJoCo 3.10, onnxruntime, the official MJCF and a walk policy set: a heavy physics test (about a minute).
"""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

pytestmark = [pytest.mark.physics, pytest.mark.heavy]

pytest.importorskip("numpy")
pytest.importorskip("mujoco")
pytest.importorskip("onnxruntime")

from duckmind.bench import clicks as C  # noqa: E402
from duckmind.bench import runner as RN  # noqa: E402
from duckmind.world import headless_duck as hd  # noqa: E402

MJCF_DIR = os.environ.get("DUCKMIND_MICRODUCK_MJCF")
if not MJCF_DIR or not (Path(MJCF_DIR) / "robot_allcollisions.xml").is_file():
    pytest.skip("needs DUCKMIND_MICRODUCK_MJCF (the official MJCF)", allow_module_level=True)
if not (hd.policy_dir() / "velstand.onnx").is_file():
    pytest.skip(f"needs the shipped policies in {hd.policy_dir()}", allow_module_level=True)

COLLISION = (
    RN.COLLISION if (Path(MJCF_DIR) / f"robot_{RN.COLLISION}.xml").is_file() else "allcollisions"
)
# the process's own settings before any click ran (a click must leave them as they were)
_DELAY0 = os.environ.get("DUCKMIND_BODY_COMMAND_DELAY_MS")
_LASTMAP0 = os.environ.get("DUCKMIND_LASTMAP")


def _job(click: dict, arm: str) -> dict:
    return {"click": click, "arm": arm, "known": None, "home": "real_home", "collision": COLLISION,
            "policies": str(hd.policy_dir()), "tree": "test", "trace": False}  # fmt: skip


def _run(job: dict) -> dict:
    """One click in a fresh process, as the bench's own workers run them: ``run_click`` sets the live duck's
    environment and body for its whole process (runner.prepare_env, _live_body), which must not reach the other
    tests of this run (2026-11-10: it gave them a 20 ms servo delay and the backlash body)."""
    with ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        return pool.submit(RN.run_click, job).result()


def test_a_bench_click_leaves_this_process_as_it_was(click: dict, main_row: dict) -> None:
    assert os.environ.get("DUCKMIND_BODY_COMMAND_DELAY_MS") == _DELAY0
    assert os.environ.get("DUCKMIND_LASTMAP") == _LASTMAP0
    assert getattr(hd, "_bench_body", None) is None


@pytest.fixture(scope="module")
def click() -> dict:
    """A short same-room click with nothing on the line."""
    clean = [
        c
        for c in C.load("real_home-v1")
        if c["cls"] == "same" and c["line"] == "clear" and c["range_m"] < 1.0
    ]
    return min(clean, key=lambda c: abs(c["bearing"]))


@pytest.fixture(scope="module")
def main_row(click: dict) -> dict:
    return _run(_job(click, "main"))


def test_a_click_gives_one_whole_row(click: dict, main_row: dict) -> None:
    row = main_row
    assert set(RN.ROW_FIELDS) <= set(row)
    assert (row["schema"], row["set"], row["k"], row["arm"], row["mode"]) == (
        RN.SCHEMA,
        click["set"],
        click["k"],
        "main",
        "headless",
    )
    assert row["state"] in ("arrived", "gave_up", "walking", "waiting")
    assert row["arrived"] == (row["state"] == "arrived") == (row["arrived_s"] is not None)
    assert 0.0 < row["secs"] <= RN.CLICK_S and (
        row["arrived_s"] is None or row["arrived_s"] == row["secs"]
    )
    assert (
        row["near"] <= row["range_m"] + 0.05
    )  # it never ends up farther than it began without having been nearer
    assert all(
        isinstance(row[key], int) and row[key] >= 0
        for key in ("falls", "bumps", "brushes", "startles", "takeovers")
    )
    assert row["plan"] is None and row["rt_min"] is None


def test_the_same_click_twice_is_the_same_row(click: dict, main_row: dict) -> None:
    assert _run(_job(click, "main")) == main_row


def test_the_truth_planner_steers_the_errand_and_the_brain_still_says_arrived(click: dict) -> None:
    row = _run(_job(click, "truth"))
    assert row["arm"] == "truth" and row["plan"]["ticks"] > 0 and row["plan"]["replans"] >= 1
    assert row["plan"]["no_route_share"] == 0.0  # a clean click has a body-wide route all the way
    if row["arrived"]:
        assert (
            row["truth_end"] <= 0.30
        )  # the place the brain believes in is the place that was clicked
