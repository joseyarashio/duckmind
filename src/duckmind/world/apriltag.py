"""AprilTag ``tag36h11`` tags for the nest, drawn from the tag family's own bits (spec §10.5,
§11.5; owner decision C3 keeps the tag on a cyan backboard).

The nest carries two tags: a 10 cm one on its backboard (seen from across the room) and a
small one flat on the pad's front edge (read looking down, within 0.5 m). The physics world
(:mod:`duckmind.world.nestpad`), the head camera and the simulator page all draw them from
:func:`grid` here, so the page's tag is the tag the camera sees — never a made-up pattern.

**The family.** tag36h11: 36 data bits in a 6×6 square inside a one-cell black border, inside
a one-cell white border (10×10 cells in all; the "tag size" is the black square, 8 cells).
Bit ``k`` of a code (most significant first) is the cell (:data:`BIT_X` [k], :data:`BIT_Y` [k])
of the black-bordered 8×8 square, x to the right and y down as the tag is printed; a 1 is
white. The layout is the AprilTag 3 one: nine cells in the top-left quarter, the same nine
turned a quarter turn for each other quarter (:func:`bit_cells`).

**The codes.** :data:`CODES` are the first 64 codes of the AprilTag 3 ``tag36h11`` family
(587 codes, minimum Hamming distance 11 under rotation). The AprilTag library is © 2013-2016
The Regents of The University of Michigan, BSD 2-Clause licence
(https://github.com/AprilRobotics/apriltag); only these numbers are used here, no code.
They were read on 2026-09-27 from OpenCV's ``DICT_APRILTAG_36h11`` marker images turned 180°
(OpenCV draws that dictionary upside down relative to AprilTag's own images) with this bit
layout, and checked against the library's first three codes (0xd7e00984b, 0xdda664ca7,
0xdc4a1c821) and against the family's distance: tests/test_world_nestpad.py checks every pair
of these 64 under all four rotations is at least 11 bits apart.

Nothing here imports MuJoCo or numpy.
"""

from __future__ import annotations

from collections.abc import Sequence

FAMILY = "tag36h11"
NBITS = 36
DATA_CELLS = 6
BLACK_CELLS = 8  # the tag's size: the black square
TOTAL_CELLS = 10  # with the white border
MIN_HAMMING = 11

# The nest's two tags (the spec's example percept reads "tag36h11:17").
NEST_TAG_ID = 17  # the backboard's tag
EDGE_TAG_ID = 18  # the small tag on the pad's front edge

_QUARTER = ((1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (2, 2), (3, 2), (4, 2), (3, 3))


def bit_cells() -> tuple[tuple[int, int], ...]:
    """(x, y) of each bit in the black-bordered 8×8 square, bit 0 (the most significant)
    first: the top-left quarter, then the same cells turned a quarter turn clockwise, three
    times."""
    cells: list[tuple[int, int]] = []
    quarter = list(_QUARTER)
    for _ in range(4):
        cells += quarter
        quarter = [(BLACK_CELLS - 1 - y, x) for (x, y) in quarter]
    return tuple(cells)


_CELLS = bit_cells()
BIT_X = tuple(x for x, _ in _CELLS)
BIT_Y = tuple(y for _, y in _CELLS)

CODES: tuple[int, ...] = (
    0xD7E00984B, 0xDDA664CA7, 0xDC4A1C821, 0xE17B470E9,
    0xEF91D01B1, 0xF429CDD73, 0x05DA29225, 0x1106CBA43,
    0x223BED79D, 0x21F51213C, 0x33EB19CA6, 0x3F76EB0F8,
    0x469A97414, 0x45DCFE0B0, 0x4A6465F72, 0x51801DB96,
    0x5EB946B4E, 0x68A7CC2EC, 0x6F0BA2652, 0x78765559D,
    0x87B83D129, 0x86CC4A5C5, 0x8B64DF90F, 0x9C577B611,
    0xA3810F2F5, 0xAF4D75B83, 0xB59A03FEF, 0xBB1096F85,
    0xD1B92FC76, 0xD0DD509D2, 0xE2CFDA160, 0x2FF497C63,
    0x47240671B, 0x5047A2E55, 0x635CA87C7, 0x691254166,
    0x68F43D94A, 0x6EF24BDB6, 0x8CDD8F886, 0x9DE96B718,
    0xAFF6E5A8A, 0xBAE46F029, 0xD225B6D59, 0xDF8BA8C01,
    0xE3744A22F, 0xFBB59375D, 0x18A916828, 0x22F29C1BA,
    0x286887D58, 0x41392322E, 0x75D18ECD1, 0x87C302743,
    0x8C6317BA9, 0x9E40F36D7, 0xC0E5A806A, 0xCC78CB87C,
    0x12D2F2D01, 0x379F36A21, 0x6973F59AC, 0x7789EA9F4,
    0x8F1C73E84, 0x8DD287A20, 0x94A4EEE4C, 0xA455379B5,
)  # fmt: skip


def code(tag_id: int) -> int:
    """The 36-bit code of a tag."""
    if not 0 <= tag_id < len(CODES):
        raise ValueError(f"{FAMILY} id {tag_id}: only 0-{len(CODES) - 1} are kept here")
    return CODES[tag_id]


def grid(tag_id: int, *, white_border: bool = True) -> tuple[tuple[int, ...], ...]:
    """The tag as printed, one value per cell (1 white, 0 black), rows top to bottom: 10×10
    with its white border, or the 8×8 black-bordered square."""
    value = code(tag_id)
    cells = [[0] * BLACK_CELLS for _ in range(BLACK_CELLS)]
    for k, (x, y) in enumerate(_CELLS):
        cells[y][x] = (value >> (NBITS - 1 - k)) & 1
    if not white_border:
        return tuple(tuple(row) for row in cells)
    top = (1,) * TOTAL_CELLS
    inner = tuple((1, *row, 1) for row in cells)
    return (top, *inner, top)


def rotate(cells: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    """A square grid turned a quarter turn clockwise (as seen)."""
    n = len(cells)
    return tuple(tuple(cells[n - 1 - c][r] for c in range(n)) for r in range(n))


def read_code(cells: Sequence[Sequence[int]]) -> int:
    """The 36 bits of an 8×8 black-bordered square (or a 10×10 one with its white border),
    in this family's bit order."""
    rows = [list(row) for row in cells]
    if len(rows) == TOTAL_CELLS:
        rows = [row[1:-1] for row in rows[1:-1]]
    if len(rows) != BLACK_CELLS or any(len(row) != BLACK_CELLS for row in rows):
        raise ValueError("a tag36h11 square is 8×8 cells (10×10 with its white border)")
    value = 0
    for x, y in _CELLS:
        value = (value << 1) | (1 if rows[y][x] else 0)
    return value


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def decode(cells: Sequence[Sequence[int]], *, max_errors: int = 2) -> tuple[int, int, int] | None:
    """(id, quarter turns clockwise that make it upright, bit errors) of a seen square, or
    None. Tries the four rotations against the kept codes; ``max_errors`` ≤ (11 − 1) / 2."""
    best: tuple[int, int, int] | None = None
    seen = tuple(tuple(1 if v else 0 for v in row) for row in cells)
    for turns in range(4):
        value = read_code(seen)
        for tag_id, want in enumerate(CODES):
            errors = hamming(value, want)
            if errors <= max_errors and (best is None or errors < best[2]):
                best = (tag_id, turns, errors)
        seen = rotate(seen)
    return best


def texture_rgb(
    tag_id: int, *, px_per_cell: int = 16, white_border: bool = False
) -> tuple[int, int, bytes]:
    """(width, height, RGB bytes) of the tag as an image (rows top to bottom), for a MuJoCo
    ``2d`` texture: the black-bordered square by default (the white border is the nest's
    mount panel), 8 cells of ``px_per_cell`` pixels."""
    if px_per_cell < 1:
        raise ValueError("px_per_cell must be at least 1")
    cells = grid(tag_id, white_border=white_border)
    n = len(cells)
    out = bytearray()
    for row in cells:
        line = bytearray()
        for value in row:
            line += (b"\xff\xff\xff" if value else b"\x00\x00\x00") * px_per_cell
        out += bytes(line) * px_per_cell
    side = n * px_per_cell
    return side, side, bytes(out)


def page_json(tag_id: int) -> dict[str, object]:
    """What the simulator page needs to draw the same tag: its cells, with the white border,
    rows top to bottom (1 white, 0 black)."""
    return {
        "family": FAMILY,
        "id": tag_id,
        "label": f"{FAMILY}:{tag_id}",
        "cells": [list(row) for row in grid(tag_id)],
        "black_cells": BLACK_CELLS,
        "total_cells": TOTAL_CELLS,
    }
