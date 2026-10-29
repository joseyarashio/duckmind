"""The eye's copy of the tag36h11 codes is the world's (spec §4.4 NB-v1; the eye may not import
the world, so it keeps its own copy, and this test keeps the two equal)."""

from __future__ import annotations

from duckmind.eye import tag36h11
from duckmind.world import apriltag


def test_the_eye_copy_equals_the_world_codes() -> None:
    assert tag36h11.CODES == apriltag.CODES
    assert tag36h11.FAMILY == apriltag.FAMILY
    assert (tag36h11.NBITS, tag36h11.BLACK_CELLS, tag36h11.TOTAL_CELLS) == (
        apriltag.NBITS,
        apriltag.BLACK_CELLS,
        apriltag.TOTAL_CELLS,
    )
    assert tag36h11.MIN_HAMMING == apriltag.MIN_HAMMING
    assert tag36h11.bit_cells() == apriltag.bit_cells()
    assert (tag36h11.BIT_X, tag36h11.BIT_Y) == (apriltag.BIT_X, apriltag.BIT_Y)


def test_the_dock_ids_are_the_worlds() -> None:
    assert tag36h11.NEST_TAG_ID == apriltag.NEST_TAG_ID == 17
    assert tag36h11.EDGE_TAG_ID == apriltag.EDGE_TAG_ID == 18
    assert tag36h11.DOCK_IDS == (17, 18)


def test_grids_and_decode_agree_with_the_world() -> None:
    for tag_id in range(len(tag36h11.CODES)):
        assert tag36h11.grid(tag_id) == apriltag.grid(tag_id)
        assert tag36h11.grid(tag_id, white_border=False) == apriltag.grid(
            tag_id, white_border=False
        )
    cells = apriltag.grid(tag36h11.NEST_TAG_ID, white_border=False)
    for turns in range(4):
        found = tag36h11.decode(cells)
        assert found is not None and found[0] == 17 and found[2] == 0
        assert found == apriltag.decode(cells)
        cells = apriltag.rotate(cells)


def test_decode_with_one_flipped_bit_and_restricted_ids() -> None:
    cells = [list(row) for row in tag36h11.grid(18, white_border=False)]
    x, y = tag36h11.BIT_X[5], tag36h11.BIT_Y[5]
    cells[y][x] ^= 1
    assert tag36h11.decode(cells, max_errors=1, ids=tag36h11.DOCK_IDS) == (18, 0, 1)
    assert tag36h11.decode(cells, max_errors=0) is None
    assert tag36h11.border_errors(cells) == 0
    cells[0][3] = 1  # a white cell in the black border
    assert tag36h11.border_errors(cells) == 1
