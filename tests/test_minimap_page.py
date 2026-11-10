"""The minimap on the simulator view (owner 2026-10-11): ``plan2d.js`` and ``minimap.js`` run in Node
over the page's own markup (tests/page_tree_node_harness.py), on the home the server sends and the
body answer the live bridge makes. How it looks and moves is checked in a browser (screenshots in
the measurement records, not in the repo)."""

from __future__ import annotations

import math
import re
from typing import Any

import pytest
from page_tree_node_harness import NODE, run

from duckmind.livebody import LivePaths
from duckmind.simweb.homeview import home_payload
from duckmind.simweb.live import LiveBridge

needs_node = pytest.mark.skipif(NODE is None, reason="needs Node")
YAW = 1.2


class _Body:
    """A physics body without processes, at one place in the home, with the world's truth."""

    error = None
    host_info: dict[str, Any] = {"ops": []}
    home = "default_home"
    policy = "sit"
    passive = False

    def frame(self) -> dict[str, Any]:
        quat = [math.cos(YAW / 2), 0.0, 0.0, math.sin(YAW / 2)]
        return {"t": 1.0, "joints": {}, "trunk": [1.2, -0.4, 0.06], "quat": quat}

    def world(self) -> dict[str, Any]:
        return {
            "people": [
                {"who": "owner", "state": "here", "pos_mm": [2000, 1000, 0]},
                {"who": "guest", "state": "away", "pos_mm": [0, 0, -5000]},
            ],
            "things": [{"id": "ball_felt", "pos_mm": [1500, 900, 35], "quat": [1, 0, 0, 0]}],
            "mocap": [],
        }

    def status(self) -> dict[str, Any]:
        return {"realtime": 1.0, "policy": "sit", "fallen": False, "monitor": {}, "held": False}


@pytest.fixture(scope="module")
def home() -> dict[str, Any]:
    return home_payload("default_home")


@pytest.fixture(scope="module")
def answer(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """What /api/body answers for that body: the live bridge's own snapshot."""
    folder = tmp_path_factory.mktemp("minimap")
    paths = LivePaths(folder, folder, folder, folder / "robotd", folder / "ort")
    bridge = LiveBridge(paths, eye=False)  # never started: no processes
    bridge.body = _Body()  # type: ignore[assignment]
    bridge.ready = True
    return {"enabled": True, **bridge.snapshot()}


@pytest.fixture(scope="module")
def drawn(home: dict[str, Any], answer: dict[str, Any]) -> dict[str, Any]:
    if NODE is None:
        pytest.skip("needs Node")
    return run("minimap", {"plan2d.js", "minimap.js"}, {"home": home, "body": answer})


@needs_node
def test_the_map_draws_the_home_the_server_sends_once_a_physics_body_runs(
    home: dict[str, Any], drawn: dict[str, Any]
) -> None:
    assert drawn["before"]["hidden"] and drawn["loaded"]["hidden"]  # no body yet: put away
    live = drawn["live"]
    assert not live["hidden"]
    layout = home["layout"]
    x0, y0, x1, y1 = layout["bounds_mm"]
    assert live["view_box"] == f"{x0} {-y1} {x1 - x0} {y1 - y0}"  # the whole home, north up
    shown = set(live["fixed"])
    standing = {e["id"] for e in layout["elements"] if e["kind"] in ("wall", "furniture")}
    assert len(standing) >= 10 and standing <= shown  # every wall and piece of furniture
    assert {thing["id"] for thing in layout["things"]} <= shown  # and every toy
    unseen = {e["id"] for e in layout["elements"] if not e["flags"]["page"]}
    assert unseen and not unseen & shown  # what only the camera has is not on the plan
    assert live["rooms"] == len(layout["rooms"])
    assert live["labels"] == [room["zh"] for room in layout["rooms"]]


@needs_node
def test_the_duck_is_drawn_where_the_bodys_truth_has_it(
    answer: dict[str, Any], drawn: dict[str, Any]
) -> None:
    truth = answer["frame"]["truth"]
    assert (truth["x_mm"], truth["y_mm"], truth["frame"]) == (1200, -400, "world")
    duck = drawn["live"]["duck"]
    assert duck["visibility"] == "visible"
    found = re.fullmatch(r"translate\((\S+) (\S+)\) rotate\((\S+)\)", duck["transform"])
    assert found is not None
    x, y, turn = (float(part) for part in found.groups())
    assert (x, y) == (1200, 400)  # (an SVG's y runs down)
    assert turn == pytest.approx(-math.degrees(truth["yaw_mrad"] / 1000))
    assert drawn["live"]["people"] == [[2000, -1000]]  # who is in the home; not who is away
    assert drawn["live"]["toys"]["ball_felt"] == "translate(700 -1500)"  # a toy that was moved
    # no truth, no duck: a body still starting, a real duck (no simulated home), no body at all
    assert not drawn["starting"]["hidden"] and drawn["starting"]["duck"]["visibility"] == "hidden"
    assert drawn["real"]["hidden"] and drawn["off"]["hidden"]


@needs_node
def test_the_map_listens_to_what_the_page_already_asks_and_sends_nothing(
    drawn: dict[str, Any],
) -> None:
    asked = [tuple(item) for item in drawn["asked"]]
    assert asked.count(("GET", "/api/home")) == 1  # the home, once
    # (MAP-EXPLORE 2026-10-12) the one thing it asks for again: the duck's own map, read only
    assert set(asked) == {("GET", "/api/home"), ("GET", "/api/neuro/knows")}
    assert {tuple(item) for item in drawn["asked_in_a_minute"]} == {("GET", "/api/neuro/knows")}


@needs_node
def test_the_fold_button_folds_the_map_and_this_viewer_keeps_the_choice(
    drawn: dict[str, Any],
) -> None:
    assert drawn["folded"] == {"state": "folded", "expanded": "false", "kept": "folded"}
