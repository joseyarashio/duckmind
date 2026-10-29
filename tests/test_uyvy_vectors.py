"""The upstream ``uyvy`` crate's test vectors (``microduck/uyvy/src/lib.rs`` tests), replayed twice:

1. against a line-by-line reference of the crate's two samplers (:func:`letterbox_from_uyvy`,
   :func:`rgb_from_uyvy` and ``Turn``) written here, so the numbers themselves are pinned;
2. against duck-eye's samplers (:mod:`duckmind.eye.frames`: ``upright_luma``, ``upright_rgb``,
   ``sample_quarter``, ``roi_luma``; :mod:`duckmind.eye.detect`: ``letterbox_rgb``), with the
   same frames and the same assertions where the eye does the same job — the turn while sampling,
   the bright row landing on the right side, the letterbox geometry, V high being red. Where the
   eye's contract differs on purpose it is said: a short frame is refused with a
   ``FrameError`` (the header's size check), where the crate pads it — neither crashes.

The crate converts BT.601 *limited* range (``298·(Y−16)``); the world camera packs and the eye
reads *full* range, so the channel values differ by design and only the crate's own thresholds
(> 200 bright, < 60 dark, 125–133 grey) are asserted on the eye.
"""

from __future__ import annotations

import random
import re

import pytest

from duckmind import paths

UPSTREAM = paths.microduck_dir() / "uyvy/src/lib.rs"
PAD = 114

# ── the reference (uyvy/src/lib.rs, integer maths as written there) ───────────

TURNS = {0: "none", 90: "right", 180: "half", 270: "left"}


def turn_from_degrees(degrees: int) -> str | None:
    return TURNS.get(degrees % 360)


def turn_upright(turn: str, width: int, height: int) -> tuple[int, int]:
    return (height, width) if turn in ("right", "left") else (width, height)


def turn_source(turn: str, ux: int, uy: int, width: int, height: int) -> tuple[int, int]:
    if turn == "none":
        return ux, uy
    if turn == "right":
        return uy, max(0, max(0, height - 1) - ux)
    if turn == "half":
        return max(0, max(0, width - 1) - ux), max(0, max(0, height - 1) - uy)
    return max(0, max(0, width - 1) - uy), ux


def _rgb(uyvy: bytes, width: int, sx: int, sy: int) -> tuple[int, int, int] | None:
    stride = width * 2
    row = sy * stride
    if row + stride > len(uyvy):
        return None
    pair = row + (sx // 2) * 4
    luma = uyvy[pair + 1 + 2 * (sx & 1)] - 16
    u = uyvy[pair] - 128
    v = uyvy[pair + 2] - 128
    r = (298 * luma + 409 * v + 128) >> 8
    g = (298 * luma - 100 * u - 208 * v + 128) >> 8
    b = (298 * luma + 516 * u + 128) >> 8
    return (min(255, max(0, r)), min(255, max(0, g)), min(255, max(0, b)))


def _round(x: float) -> int:
    """Rust's f32::round: half away from zero."""
    return int(x + 0.5) if x >= 0 else -int(-x + 0.5)


def letterbox_from_uyvy(
    uyvy: bytes, width: int, height: int, size: int, turn: str
) -> tuple[list[int], tuple[float, float, float]]:
    uw, uh = turn_upright(turn, width, height)
    scale = min(size / uw, size / uh)
    fw = min(max(_round(uw * scale), 1), size)
    fh = min(max(_round(uh * scale), 1), size)
    pad_x, pad_y = (size - fw) // 2, (size - fh) // 2
    out = [PAD] * (size * size * 3)
    for y in range(fh):
        uy = (y * uh) // fh
        for x in range(fw):
            ux = (x * uw) // fw
            sx, sy = turn_source(turn, ux, uy, width, height)
            px = _rgb(uyvy, width, sx, sy)
            if px is None:
                continue
            t = ((y + pad_y) * size + (x + pad_x)) * 3
            out[t : t + 3] = px
    return out, (scale, float(pad_x), float(pad_y))


def rgb_from_uyvy(
    uyvy: bytes, width: int, height: int, longest: int, turn: str
) -> tuple[list[int], int, int]:
    uw, uh = turn_upright(turn, width, height)
    scale = min(longest / max(uw, uh), 1.0)
    ow, oh = max(_round(uw * scale), 1), max(_round(uh * scale), 1)
    out = [0] * (ow * oh * 3)
    for y in range(oh):
        uy = (y * uh) // oh
        for x in range(ow):
            ux = (x * uw) // ow
            sx, sy = turn_source(turn, ux, uy, width, height)
            px = _rgb(uyvy, width, sx, sy)
            if px is None:
                continue
            t = (y * ow + x) * 3
            out[t : t + 3] = px
    return out, ow, oh


# ── the frames of the upstream tests ────────────────────────────────────────────


def grey_with_bright_top_left(width: int = 8, height: int = 4) -> bytes:
    """8x4 UYVY, all dark grey except the top-left pixel (Y0 of the first pair) white."""
    data = bytearray([128, 16, 128, 16] * (width * height // 2))
    data[1] = 235
    return bytes(data)


def mid_grey(width: int = 4, height: int = 8) -> bytes:
    return bytes([128, 126, 128, 126] * (width * height // 2))


def v_high_red() -> bytes:
    return bytes([64, 126, 200, 126] * 2)  # a 4x1 frame, V high


def dark_top_bright_bottom() -> bytes:
    dark, bright = [128, 16, 128, 16], [128, 235, 128, 235]
    return bytes(dark + dark + bright + bright)  # 4x2: row 0 dark, row 1 bright


# ── 1. the reference against the crate's assertions ──────────────────────────


def test_the_rgb_scaler_turns_the_picture_and_keeps_its_shape() -> None:
    uyvy = grey_with_bright_top_left()
    out, w, h = rgb_from_uyvy(uyvy, 8, 4, 8, "none")
    assert (w, h) == (8, 4)
    assert out[0] > 200
    out, w, h = rgb_from_uyvy(uyvy, 8, 4, 8, "right")
    assert (w, h) == (4, 8), "a quarter turn swaps the axes"
    top_right = (w - 1) * 3
    assert out[top_right] > 200, "the bright pixel moved to the top-right"
    assert out[0] < 200


def test_the_rgb_scaler_only_ever_shrinks() -> None:
    uyvy = bytes([128, 16, 128, 16] * (8 * 4 // 2))
    assert rgb_from_uyvy(uyvy, 8, 4, 64, "none")[1:] == (8, 4)
    assert rgb_from_uyvy(uyvy, 8, 4, 4, "none")[1:] == (4, 2)


def test_uyvy_letterboxes_in_one_pass() -> None:
    out, fit = letterbox_from_uyvy(mid_grey(), 4, 8, 8, "none")
    assert fit == (1.0, 2.0, 0.0)
    assert len(out) == 8 * 8 * 3
    assert out[0:3] == [PAD, PAD, PAD]
    assert all(125 <= v <= 133 for v in out[6:9]), "grey stayed grey"
    out, _ = letterbox_from_uyvy(v_high_red(), 4, 1, 4, "none")
    pixel = out[12:15]  # row 1 of the 4-wide square, first column
    assert pixel[0] > pixel[2], "V high is red, not blue"


def test_a_short_uyvy_frame_does_not_panic() -> None:
    out, _ = letterbox_from_uyvy(bytes([128] * (2 * 2 * 2)), 2, 8, 8, "none")
    assert len(out) == 8 * 8 * 3


def test_a_quarter_turn_happens_while_sampling() -> None:
    assert turn_from_degrees(90) == "right"
    assert turn_from_degrees(270) == "left"
    assert turn_from_degrees(45) is None
    assert turn_upright("right", 1280, 720) == (720, 1280)
    assert turn_upright("half", 1280, 720) == (1280, 720)
    w, h = 4, 2
    uw, _ = turn_upright("right", w, h)
    assert turn_source("right", uw - 1, 0, w, h) == (0, 0)
    assert turn_source("right", 0, 0, w, h) == (0, h - 1)
    assert turn_source("left", 0, 0, w, h) == (w - 1, 0)


def test_a_turned_frame_puts_the_bright_row_on_the_right_side() -> None:
    out, fit = letterbox_from_uyvy(dark_top_bright_bottom(), 4, 2, 4, "right")
    assert (fit[1], fit[2]) == (1.0, 0.0), "2x4 upright inside a 4x4 square"
    assert out[(0 * 4 + 1) * 3] > 200, "the source's bottom row is now the left column"
    assert out[(0 * 4 + 2) * 3] < 60, "and its top row is the right column"


# ── 2. duck-eye's samplers, same frames, same assertions ─────────────────────


def _eye() -> tuple:
    np = pytest.importorskip("numpy")
    frames = pytest.importorskip("duckmind.eye.frames")
    return np, frames


def test_eye_upright_turns_like_the_crate() -> None:
    np, frames = _eye()
    uyvy = grey_with_bright_top_left()
    flat = frames.upright_luma(uyvy, 8, 4, 0)
    assert flat.shape == (4, 8) and flat[0, 0] > 200
    turned = frames.upright_luma(uyvy, 8, 4, 90)
    assert turned.shape == (8, 4), "a quarter turn swaps the axes"
    assert turned[0, 3] > 200 and turned[0, 0] < 200, "top-left of the frame → top-right"
    rgb = frames.upright_rgb(uyvy, 8, 4, 90)
    assert rgb.shape == (8, 4, 3) and rgb[0, 3, 0] > 200 and rgb[0, 0, 0] < 200


def test_eye_bright_row_lands_on_the_left_column() -> None:
    np, frames = _eye()
    luma = frames.upright_luma(dark_top_bright_bottom(), 4, 2, 90)
    assert luma.shape == (4, 2)
    assert (luma[:, 0] > 200).all(), "the source's bottom row is now the left column"
    assert (luma[:, 1] < 60).all(), "and its top row is the right column"


@pytest.mark.parametrize("rotate", [0, 90, 180, 270])
def test_eye_every_turn_matches_the_crate_source_mapping(rotate: int) -> None:
    """Every upright pixel comes from the source pixel ``Turn::source`` names (a frame of unique
    lumas, all four turns)."""
    np, frames = _eye()
    w, h = 6, 4
    lumas = list(range(20, 20 + w * h))
    data = bytearray()
    for y in range(h):
        for x in range(0, w, 2):
            data += bytes([128, lumas[y * w + x], 128, lumas[y * w + x + 1]])
    turn = turn_from_degrees(rotate)
    assert turn is not None
    luma = frames.upright_luma(bytes(data), w, h, rotate)
    uw, uh = turn_upright(turn, w, h)
    assert luma.shape == (uh, uw)
    for uy in range(uh):
        for ux in range(uw):
            sx, sy = turn_source(turn, ux, uy, w, h)
            assert luma[uy, ux] == lumas[sy * w + sx], (rotate, ux, uy)


def test_eye_quarter_sampler_is_the_crate_nearest_sampling() -> None:
    """sample_quarter's luma = every 4th upright pixel of the crate's turn; its chroma sits on
    the pair the crate reads (U byte 0, V byte 2 of the source pair)."""
    np, frames = _eye()
    rng = random.Random(7)
    w, h = 64, 32
    data = bytes(rng.randrange(256) for _ in range(w * h * 2))
    planes = frames.sample_quarter(data, w, h, 90)
    uw, uh = turn_upright("right", w, h)
    assert planes.y.shape == (len(range(0, uh, 4)), len(range(0, uw, 4)))
    for i, uy in enumerate(range(0, uh, 4)):
        for j, ux in enumerate(range(0, uw, 4)):
            sx, sy = turn_source("right", ux, uy, w, h)
            assert planes.y[i, j] == data[(sy * w + sx) * 2 + 1]
    for i, uy in enumerate(range(0, uh, 8)):
        for j, ux in enumerate(range(0, uw, 8)):
            sx, sy = turn_source("right", ux, uy, w, h)
            pair = sy * w * 2 + (sx // 2) * 4
            assert planes.u[i, j] == data[pair] and planes.v[i, j] == data[pair + 2]


def test_eye_roi_cut_is_the_same_turn() -> None:
    np, frames = _eye()
    rng = random.Random(3)
    w, h = 16, 8
    data = bytes(rng.randrange(256) for _ in range(w * h * 2))
    full = frames.upright_luma(data, w, h, 90)
    roi = frames.roi_luma(data, w, h, 90, (2, 3, 6, 11))
    assert (roi == full[3:11, 2:6]).all()


def test_eye_letterbox_of_the_upright_picture() -> None:
    """The duck task's input: upright RGB then the crate's letterbox — the one-pass crate
    function's geometry and colours (grey stays grey, V high is red)."""
    np, frames = _eye()
    detect = pytest.importorskip("duckmind.eye.detect")
    rgb = frames.upright_rgb(mid_grey(), 4, 8, 0)
    out, fit = detect.letterbox_rgb(rgb, 8)
    assert (fit.scale, fit.pad_x, fit.pad_y) == (1.0, 2.0, 0.0)
    assert out[0, 0].tolist() == [PAD] * 3
    assert all(125 <= int(v) <= 133 for v in out[0, 2])
    red = frames.upright_rgb(v_high_red(), 4, 1, 0)
    boxed, _ = detect.letterbox_rgb(red, 4)
    pixel = boxed[1, 0]
    assert int(pixel[0]) > int(pixel[2]), "V high is red, not blue"
    turned = frames.upright_rgb(dark_top_bright_bottom(), 4, 2, 90)
    square, fit = detect.letterbox_rgb(turned, 4)
    assert (fit.pad_x, fit.pad_y) == (1.0, 0.0)
    assert square[0, 1, 0] > 200 and square[0, 2, 0] < 60


def test_eye_refuses_a_short_frame_instead_of_crashing() -> None:
    """The crate pads a short frame; the eye's header check refuses it (a controlled
    ``FrameError`` the look answers as ``none``/``frame_error``). Never an IndexError."""
    np, frames = _eye()
    short = bytes([128] * (2 * 2 * 2))
    for call in (
        lambda: frames.upright_luma(short, 2, 8, 0),
        lambda: frames.sample_quarter(short, 2, 8, 0),
    ):
        with pytest.raises(frames.FrameError):
            call()


# ── every upstream test is replayed ───────────────────────────────────────────


@pytest.mark.skipif(not UPSTREAM.is_file(), reason="needs the upstream checkout (read-only)")
def test_every_upstream_uyvy_test_is_replayed() -> None:
    text = UPSTREAM.read_text(encoding="utf-8")
    upstream = set(re.findall(r"#\[test\]\s*\n\s*fn (\w+)", text))
    here = {name.removeprefix("test_") for name in globals() if name.startswith("test_")}
    assert upstream, "found no upstream tests"
    assert upstream <= here, f"not replayed: {sorted(upstream - here)}"
