"""The go-there click sets (duckmind.bench.clicks, .floor): the generator is deterministic and gives the committed set,
places are reachable by a body-wide route and starts stand in the open, and the legacy set keeps every old click with
the unfair ones flagged."""

from __future__ import annotations

import math

import numpy as np
import pytest

from duckmind.bench import clicks as C
from duckmind.bench import floor as F
from duckmind.world import homelayout


@pytest.fixture(scope="module")
def first_block() -> list[dict]:
    return C.generate("real_home", blocks=1)[0]


def test_the_generator_is_deterministic_and_makes_the_committed_set(
    first_block: list[dict],
) -> None:
    again, meta = C.generate("real_home", blocks=1)
    assert again == first_block
    committed = C.load("real_home-v1")
    assert (
        committed[: len(first_block)] == first_block
    )  # a shorter set is a prefix of the committed one
    assert (
        meta["layout"]
        == C.meta_of("real_home-v1")["layout"]
        == str(homelayout.load("real_home").fingerprint)
    )


def test_every_committed_click_is_a_fair_question() -> None:
    clicks = C.load("real_home-v1")
    meta = C.meta_of("real_home-v1")
    assert len(clicks) == meta["clicks"] == 800
    for block in range(meta["blocks"]):
        part = C.select(clicks, blocks=[block])
        assert len(part) == C.BLOCK and sum(c["cls"] == "same" for c in part) == C.BLOCK_SAME
    for c in clicks:
        assert not c["flags"] and c["reach"]
        assert C.RANGE_M[0] - 0.01 <= c["range_m"] <= C.RANGE_M[1] + 0.01
        assert c["clear_start_m"] >= C.START_CLEAR_M - 1e-6  # not in clutter
        assert c["clear_point_m"] >= C.POINT_CLEAR_M - 1e-6  # room for the body at the place
        assert c["clear_path_m"] >= F.BODY_RADIUS_M - 1e-6  # the whole route is body-wide
        assert c["path_m"] <= C.PATH_MAX_M and c["detour"] >= 0.95
        assert (c["room"] == c["room_to"]) == (c["cls"] == "same")
        assert c["seed"] == C.SEED0 + c["k"] % C.SEEDS
    assert [c["k"] for c in clicks] == list(range(800))
    assert {c["line"] for c in clicks} >= {"clear", "thing", "furniture", "low"}


def test_labels_tell_a_reachable_place_from_one_on_a_wall_or_inside_furniture() -> None:
    lab = C.Labeller(homelayout.load("real_home"))
    start = (0.6, -0.3, 0.0)  # the open floor of the living room
    open_place = lab.label(start, (1.2, 0.2))
    assert open_place["reach"] and open_place["cls"] == "same" and not open_place["flags"]
    on_wall = lab.label(start, (-0.5, 0.6))  # on the wall between the living room and the bedroom
    assert C.FLAG_ON_SOLID in on_wall["flags"] and not on_wall["reach"]
    in_sofa = lab.label(start, (-0.9, -1.0))
    assert C.FLAG_ON_SOLID in in_sofa["flags"] and C.FLAG_UNREACHABLE in in_sofa["flags"]
    behind_wall = lab.label(
        (0.6, 0.2, 0.0), (-0.5, 1.2)
    )  # the bedroom, through the wall: round by the doorway
    assert behind_wall["cls"] == "other" and behind_wall["line"] == "wall" and behind_wall["reach"]
    assert behind_wall["detour"] > 1.5
    in_clutter = lab.label((0.25, -0.25, 0.0), (1.2, 0.2))  # a start beside the pile of books
    assert C.FLAG_CLUTTER in in_clutter["flags"]


def test_the_route_is_body_wide() -> None:
    """A gap narrower than the body is no way through; a wider one is; the cost of the way is about its length."""
    cell = F.CELL_M

    def reach(gap_cells: int) -> float:
        solid = np.zeros((60, 60), dtype=bool)
        solid[[0, -1], :] = True
        solid[:, [0, -1]] = True
        solid[:, 30] = True
        solid[30 - gap_cells // 2 : 30 - gap_cells // 2 + gap_cells, 30] = False
        clear = F.clearance(solid, cell)
        passable = ~solid & (clear >= F.BODY_RADIUS_M)
        field = F.cost_to_go(passable, np.ones(solid.shape), (30, 45), cell)
        return float(field[30, 15])

    assert math.isinf(reach(8))  # a 16 cm gap: the body (20 cm with its margin) does not fit
    through = reach(16)  # a 32 cm gap
    assert through == pytest.approx(30 * cell, rel=0.05)
    assert math.isinf(reach(0))


def test_clearance_is_the_distance_to_the_nearest_solid_cell() -> None:
    solid = np.zeros((40, 40), dtype=bool)
    solid[20, 20] = True
    clear = F.clearance(solid, 0.02)
    assert clear[20, 20] == 0.0
    assert clear[20, 25] == pytest.approx(0.10)
    assert clear[23, 24] == pytest.approx(0.10)  # 3-4-5
    assert clear[0, 0] == pytest.approx(0.02)  # the grid's edge is solid


def test_the_legacy_set_keeps_every_click_and_flags_the_unfair_ones() -> None:
    legacy = C.load("legacy-205")
    assert [c["k"] for c in legacy] == list(range(205))
    flags = C.meta_of("legacy-205")["flags"]
    assert flags[C.FLAG_ON_SOLID] >= 6  # the review's "6 of 205 sit on walls" (and those on things)
    assert flags[C.FLAG_CLUTTER] > 50 and flags[C.FLAG_UNREACHABLE] >= 1
    clean = C.select(legacy, clean=True)
    assert 0 < len(clean) < len(legacy) and all(not c["flags"] for c in clean)
    assert C.select(legacy, keys=[3, 7], limit=1)[0]["k"] == 3
