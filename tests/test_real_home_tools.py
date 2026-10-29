"""The obstacle course's and the home runs' scoring (tools/real_home): what a path did with an item."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "real_home"))
metrics = pytest.importorskip("metrics")

from duckmind.world import homelayout  # noqa: E402


def _samples(points: list[tuple[float, float]]) -> list[dict]:
    return [{"t": i * 0.2, "x": x, "y": y, "fallen": False} for i, (x, y) in enumerate(points)]


def _items(item: str, **extra) -> dict:
    raw = homelayout.load("default_home").to_json()
    raw["placements"] = [{"id": "o", "lib": item, "x_mm": 0, "y_mm": 0, "yaw_mrad": 1571, **extra}]
    raw.pop("placements_expanded", None)
    return metrics.outlines(homelayout.load(raw))


def test_walking_over_a_sill_is_a_crossing_and_walking_round_it_is_an_avoid() -> None:
    items = _items("threshold_10", length_mm=900)
    over = metrics.meets(_samples([(-0.6 + 0.1 * k, 0.0) for k in range(13)]), items)
    assert [(m["crossed"], m["avoided"], m["passability"]) for m in over] == [(True, False, "step")]
    round_it = metrics.meets(_samples([(-0.6 + 0.1 * k, 0.6) for k in range(13)]), items)
    assert [(m["crossed"], m["avoided"]) for m in round_it] == [(False, True)]
    far = metrics.meets(_samples([(-2.0 + 0.1 * k, 2.0) for k in range(10)]), items)
    assert far == []  # never within 25 cm: no meeting


def test_going_up_to_it_and_back_is_not_a_crossing() -> None:
    items = _items("threshold_10", length_mm=900)
    path = [(-0.8 + 0.1 * k, 0.0) for k in range(9)] + [(-0.1 - 0.1 * k, 0.0) for k in range(8)]
    (m,) = metrics.meets(_samples(path), items)
    assert m["over"] and not m["crossed"]


def test_a_loose_thing_is_listed_as_movable_with_its_class() -> None:
    items = _items("slipper_pair")
    (entry,) = items.values()
    assert entry["movable"] and entry["passability"] == "movable" and entry["thing"] == "o"


def test_the_course_table_groups_by_class() -> None:
    rows = [
        {
            "c": "step",
            "attempts": 2,
            "crosses": 1,
            "avoids": 0,
            "falls": 0,
            "stuck": 0,
            "t_s": 20.0,
            "outcome": "crossed",
        },
        {
            "c": "step",
            "attempts": 1,
            "crosses": 0,
            "avoids": 0,
            "falls": 1,
            "stuck": 0,
            "t_s": None,
            "outcome": "turned_back",
        },
    ]
    (row,) = metrics.table(rows, "c")
    assert (
        row["trials"],
        row["attempts"],
        row["crosses"],
        row["falls"],
        row["turned_back"],
        row["median_s"],
    ) == (2, 3, 1, 1, 1, 20.0)
