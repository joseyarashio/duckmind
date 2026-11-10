"""The go-there benchmark's rows and tables (duckmind.bench.report, .compare, .live, .cli): pairing by click, the paired
table's counts, the looks of a comparison, and the live mode's row, nearest-start choice and realtime gate."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from duckmind.bench import cli as CLI
from duckmind.bench import clicks as C
from duckmind.bench import compare as CP
from duckmind.bench import live as LV
from duckmind.bench import report as RP
from duckmind.bench import runner as RN


def _row(
    k: int, arrived: bool, cls: str = "same", line: str = "clear", **more: Any
) -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(RN.ROW_FIELDS)
    row.update(schema=RN.SCHEMA, set="t", k=k, arm="main", mode="headless", cls=cls, line=line, arrived=arrived,
               arrived_s=10.0 if arrived else None, truth_ok=arrived, state="arrived" if arrived else "walking",
               falls=0, bumps=0, brushes=0, startles=0, gave_up=False, takeovers=0, flagged=False)  # fmt: skip
    row.update(more)
    return row


def test_two_arms_are_paired_by_click_and_counted() -> None:
    a = [
        _row(0, True),
        _row(1, True),
        _row(2, False),
        _row(3, False),
        _row(4, True, cls="other"),
        _row(9, True),
    ]
    b = [
        _row(3, True, falls=1),
        _row(2, False),
        _row(1, False),
        _row(0, True),
        _row(4, True, cls="other"),
        _row(7, True),
    ]
    pairs = RP.pair(a, b)
    assert [x["k"] for x, _ in pairs] == [
        0,
        1,
        2,
        3,
        4,
    ]  # only the clicks both answered, in A's order
    assert all(x["k"] == y["k"] for x, y in pairs)
    assert RP.counts(pairs) == (2, 1, 1, 1)  # both, only A, only B, neither
    assert RP.discordant_share(a, b) == pytest.approx(2 / 4)  # same room: clicks 1 and 3 of 0–3
    assert CP.same_room_look(a, b) == (4, 1, 1)
    table = RP.paired(a, b, "A", "B")
    same = next(line for line in table.splitlines() if line.startswith("| same room |"))
    cells = [c.strip() for c in same.strip("|").split("|")]
    assert cells[1] == "4" and cells[4:8] == ["1", "1", "1", "1"] and cells[9] == "1.000"
    falls = next(line for line in table.splitlines() if line.startswith("| falls | same room |"))
    assert [c.strip() for c in falls.strip("|").split("|")][2:6] == ["0 (0)", "1 (1)", "1", "0"]


def test_summary_gives_each_class_its_rate_and_interval() -> None:
    rows = [_row(k, k < 6) for k in range(10)] + [
        _row(10 + k, k < 1, cls="other", line="wall") for k in range(4)
    ]
    text = RP.summary(rows, "main")
    lines = {
        line.split("|")[1].strip(): line for line in text.splitlines() if line.startswith("| ")
    }
    assert "6/10 = 60 % (31–83)" in lines["same room"]
    assert "1/4 = 25 %" in lines["other room"]
    assert "7/14 = 50 %" in lines["all"]
    assert (
        "same room, line clear" in lines and "same room, line thing" not in lines
    )  # only classes that have clicks


def test_every_row_field_is_named_once_and_the_power_table_has_the_three_gains() -> None:
    assert len(set(RN.ROW_FIELDS)) == len(RN.ROW_FIELDS)
    for key in (
        "k",
        "set",
        "arm",
        "arrived",
        "arrived_s",
        "falls",
        "bumps",
        "brushes",
        "gave_up",
        "cls",
        "line",
        "rt_min",
    ):
        assert key in RN.ROW_FIELDS
    table = RP.power_table([0.25], per_min=40.0)
    assert (
        "+5 points" in table
        and "+9 points" in table
        and "+15 points" in table
        and "min for both arms" in table
    )


def test_blocks_are_split_into_looks_and_ranges_are_read() -> None:
    assert CP.split_looks([0, 1, 2, 3, 4, 5, 6, 7], 4) == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert CP.split_looks([0, 1, 2], 2) == [[0, 1], [2]]
    assert CP.split_looks([0], 4) == [[0]]
    assert CLI._ints("0,2-4") == [0, 2, 3, 4] and CLI._ints(None) is None
    args = CLI.parser().parse_args(
        ["gothere", "compare", "--a", "main", "--b", "HEAD", "--out", "x", "--looks", "4"]
    )
    assert (args.cmd, args.looks, args.delta, args.set_name) == ("compare", 4, 5.0, "real_home-v1")


def test_a_tree_is_a_snapshot_folder_as_it_is(tmp_path: Path) -> None:
    (tmp_path / "snap" / "src" / "duckmind").mkdir(parents=True)
    src, label = CP.tree_of(str(tmp_path / "snap"), tmp_path / "work")
    assert src == (tmp_path / "snap" / "src").resolve() and label == "snap"
    assert not (tmp_path / "work").exists()  # nothing exported for a folder


@pytest.mark.subprocess
def test_a_tree_is_a_git_ref_exported_once(tmp_path: Path) -> None:
    if (
        subprocess.run(
            ["git", "-C", str(CP.REPO), "rev-parse", "HEAD"], capture_output=True
        ).returncode
        != 0
    ):
        pytest.skip("not a git checkout")
    src, label = CP.tree_of("HEAD", tmp_path)
    assert (src / "duckmind" / "neuro" / "errand.py").is_file() and label.startswith("HEAD@")
    assert src.parent.parent.parent == tmp_path  # <work>/trees/<sha>/src
    stamp = (src / "duckmind" / "__init__.py").stat().st_mtime_ns
    assert (
        CP.tree_of("HEAD", tmp_path)[0] == src
        and (src / "duckmind" / "__init__.py").stat().st_mtime_ns == stamp
    )


@pytest.mark.subprocess
def test_the_brain_never_loads_the_bench() -> None:
    """The truth planner is a measuring instrument: a process that runs the duck's brain has nothing of the bench in it."""
    code = "import sys, duckmind.neuro.brain, duckmind.neuro.errand; print(sorted(m for m in sys.modules if m.startswith('duckmind.bench')))"
    env = {**os.environ, "PYTHONPATH": str(CP.REPO / "src")}
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
    )
    assert out.stdout.strip() == "[]"


class _FakeServer:
    """A test server's answers: the duck stands at ``pose``; an errand walks two polls, then arrives."""

    def __init__(self, pose: tuple[float, float, float], realtime: list[float]) -> None:
        self.pose = pose
        self.realtime = realtime
        self.sent: list[tuple[float, float]] = []
        self.polls = 0

    def scene(self) -> dict[str, Any]:
        return {"battery": 0.9, "fear": 0.0, "behaviour": "wander", "posture": "stand", "fallen": False, "asleep": False,
                "holding": False, "visits": [], "day": 600.0, "realtime": 1.0, "truth": self.pose}  # fmt: skip

    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        assert path == "/api/neuro/gothere"
        self.sent.append((body["point"]["x_m"], body["point"]["y_m"]))
        self.polls = 0
        return {"ok": True, "range_m": 1.0, "bearing_rad": 0.1}

    def get(self, path: str) -> dict[str, Any]:
        if path.startswith("/api/neuro/telemetry"):
            self.polls += 1
            state = "walking" if self.polls < 4 else "arrived"
            return {
                "seq": self.polls,
                "errand": {"state": state},
                "behaviour": "wander",
                "events": [{"reflex": "bump_stop"}] if self.polls == 2 else [],
            }
        if path.startswith("/api/body"):
            x, y = self.sent[-1]
            return {
                "realtime": self.realtime[min(len(self.sent), len(self.realtime)) - 1],
                "frame": {"truth": {"x_mm": 1000 * x + 50, "y_mm": 1000 * y}},
            }
        return {"frame": {"body": {"fallen": False}}}


def test_live_asks_the_nearest_start_and_discards_a_click_the_machine_was_too_slow_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clicks = C.select(C.load("real_home-v1"), blocks=[0])
    target = next(c for c in clicks if c["cls"] == "same" and c["line"] == "clear")
    pose = (
        target["pose"][0] + 0.05,
        target["pose"][1],
        1.0,
    )  # the duck stands 5 cm from that click's start
    fake = _FakeServer(pose, realtime=[0.80, 0.99])  # the first click runs on a throttled machine
    monkeypatch.setattr(LV, "Server", lambda base: fake)
    out = tmp_path / "live.jsonl"
    rows = LV.run_live(
        "http://127.0.0.1:1", [target], out, sleep=lambda s: None, log=lambda s: None
    )
    assert fake.sent == [tuple(target["point"])] * 2  # asked, discarded, asked again
    assert len(rows) == 1 and rows[0]["k"] == target["k"] and rows[0]["mode"] == "live"
    assert (
        rows[0]["arrived"]
        and rows[0]["bumps"] == 1
        and rows[0]["rt_min"] == 0.99
        and rows[0]["start_off_m"] == 0.05
    )
    assert set(RN.ROW_FIELDS) <= set(rows[0])
    assert [json.loads(line)["k"] for line in out.read_text().splitlines()] == [target["k"]]
    slow = [
        json.loads(line) for line in out.with_suffix(".discarded.jsonl").read_text().splitlines()
    ]
    assert len(slow) == 1 and slow[0]["rt_min"] == 0.80
    # a second run asks nothing again
    assert (
        LV.run_live("http://127.0.0.1:1", [target], out, sleep=lambda s: None, log=lambda s: None)
        == rows
    )
    assert len(fake.sent) == 2


def test_live_goes_to_a_start_first_when_none_is_near() -> None:
    from duckmind.world import homelayout

    lab = C.Labeller(homelayout.load("real_home"))
    clicks = C.select(C.load("real_home-v1"), blocks=[0])
    far = max(clicks, key=lambda c: c["pose"][0] + c["pose"][1])
    click, info, go = LV.choose(lab, (-2.5, -2.5, 0.0), [far])
    assert click is None and info is None and go is far
    click, info, go = LV.choose(lab, tuple(far["pose"]), [far])
    assert click is far and info is not None and info["cls"] == far["cls"] and go is None


def test_only_simulators_and_batches_count_against_the_machine() -> None:
    live_duck = "/x/python -m duckmind.simweb --port 8791 --life /home/u/.duckmind/lives/danhuang --home real_home"
    assert not RN._is_batch(live_duck)
    assert not RN._is_batch(
        "/x/python -m duckmind.world.simhome --headless --port 5 --home /home/u/.duckmind/lives/danhuang/l.json"
    )
    assert RN._is_batch(
        "/x/python -m duckmind.world.simhome --headless --port 6 --home /tmp/life/real_home.json"
    )
    assert RN._is_batch("/x/python -m duckmind.bench gothere run --set real_home-v1")
    assert RN._is_batch("/x/python /w/src/duckmind/bench/_boot.py gothere run --arm main")
    assert not RN._is_batch(
        "/x/python -m duckmind.bench gothere compare --a main --b x"
    )  # (its two runs are counted)
    assert not RN._is_batch(
        "uv run -q python -m duckmind.bench gothere run"
    )  # (the launcher of one)
    assert not RN._is_batch('node -e require("/w/src/duckmind/simweb/static/realduck.js")')


def test_a_batch_waits_until_the_machine_has_room(monkeypatch: pytest.MonkeyPatch) -> None:
    loads = iter([9.5, 6.0, 6.0, 3.0])
    others = iter([0, 2, 1])
    said: list[str] = []
    naps: list[float] = []
    monkeypatch.setattr(RN.os, "getloadavg", lambda: (next(loads), 0.0, 0.0))
    monkeypatch.setattr(RN, "others_running", lambda: next(others))
    monkeypatch.setattr(RN.time, "sleep", naps.append)
    RN.gate(said.append)
    assert (
        len(naps) == 2 and len(said) == 2
    )  # too busy, then two other batches, then room (load 6, one other)
    assert "load 9.5" in said[0] and "2 other" in said[1]
