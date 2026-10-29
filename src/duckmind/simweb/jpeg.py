"""Baseline JPEG from a numpy RGB frame, for the duck's-eye picture on the page.

The live simulator runs with numpy and MuJoCo but not Pillow, so :func:`encode` has its own
encoder: JFIF baseline, 4:2:0, the standard (Annex K) quantisation and Huffman tables, the
entropy coding vectorised in numpy. When Pillow is installed it is used instead (faster); both
give ordinary JPEG files any browser shows.
"""

from __future__ import annotations

import io
import struct
from typing import Any

_ZIGZAG = (
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
)  # fmt: skip

_LUMA_Q = (
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
)  # fmt: skip

_CHROMA_Q = (
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
) + (99,) * 32  # fmt: skip

_DC_LUMA = ((0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0), tuple(range(12)))
_DC_CHROMA = ((0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0), tuple(range(12)))
_AC_LUMA = (
    (0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7D),
    (
        0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51,
        0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1,
        0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18,
        0x19, 0x1A, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
        0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57,
        0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75,
        0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x92,
        0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
        0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3,
        0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8,
        0xD9, 0xDA, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2,
        0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA,
    ),
)  # fmt: skip
_AC_CHROMA = (
    (0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77),
    (
        0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07,
        0x61, 0x71, 0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xA1, 0xB1, 0xC1, 0x09,
        0x23, 0x33, 0x52, 0xF0, 0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34, 0xE1, 0x25,
        0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38,
        0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56,
        0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74,
        0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
        0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5,
        0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA,
        0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6,
        0xD7, 0xD8, 0xD9, 0xDA, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF2,
        0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA,
    ),
)  # fmt: skip


def _codes(table: tuple[tuple[int, ...], tuple[int, ...]]) -> Any:
    """Canonical Huffman codes: (code, length) per symbol 0..255."""
    import numpy as np

    bits, values = table
    code_of = np.zeros(256, dtype=np.int64)
    size_of = np.zeros(256, dtype=np.int64)
    code = 0
    k = 0
    for length in range(1, 17):
        for _ in range(bits[length - 1]):
            code_of[values[k]] = code
            size_of[values[k]] = length
            code += 1
            k += 1
        code <<= 1
    return code_of, size_of


def _quant(base: tuple[int, ...], quality: int) -> Any:
    import numpy as np

    q = min(100, max(1, int(quality)))
    scale = 5000 / q if q < 50 else 200 - 2 * q
    table = (np.array(base, dtype=np.int64) * scale + 50) // 100
    return np.clip(table, 1, 255).astype(np.int64)


def _dct_matrix() -> Any:
    import numpy as np

    c = np.zeros((8, 8), dtype=np.float64)
    for u in range(8):
        a = np.sqrt(1 / 8) if u == 0 else np.sqrt(2 / 8)
        for x in range(8):
            c[u, x] = a * np.cos((2 * x + 1) * u * np.pi / 16)
    return c


def _blocks(plane: Any) -> Any:
    """(H, W) with H, W multiples of 8 → (H/8, W/8, 8, 8)."""
    h, w = plane.shape
    return plane.reshape(h // 8, 8, w // 8, 8).swapaxes(1, 2)


def _segment(marker: int, payload: bytes) -> bytes:
    return struct.pack(">HH", marker, len(payload) + 2) + payload


def _dht(table_class: int, table_id: int, table: tuple[tuple[int, ...], tuple[int, ...]]) -> bytes:
    bits, values = table
    return bytes([table_class << 4 | table_id]) + bytes(bits) + bytes(values)


_BIT_LENGTHS: Any = None


def _bit_lengths(values: Any) -> Any:
    """JPEG's magnitude category: bits needed for |v| (0 for 0), v within ±2047."""
    global _BIT_LENGTHS
    import numpy as np

    if _BIT_LENGTHS is None:
        _BIT_LENGTHS = np.array([v.bit_length() for v in range(2048)], dtype=np.int64)
    return _BIT_LENGTHS[np.abs(values).astype(np.int64)]


def _entropy(zz: Any, comp: Any) -> bytes:
    """Huffman-code quantised zigzag blocks (N, 64) in scan order; ``comp`` 0 = Y, 1/2 = chroma."""
    import numpy as np

    n = zz.shape[0]
    luma_dc, chroma_dc = _codes(_DC_LUMA), _codes(_DC_CHROMA)
    luma_ac, chroma_ac = _codes(_AC_LUMA), _codes(_AC_CHROMA)
    is_luma = comp == 0

    # DC: differences along each component's own sequence of blocks
    dc = zz[:, 0].astype(np.int64)
    diff = np.zeros(n, dtype=np.int64)
    for c in (0, 1, 2):
        idx = np.nonzero(comp == c)[0]
        if idx.size:
            seq = dc[idx]
            diff[idx] = np.diff(seq, prepend=0)
    diff = np.clip(diff, -2047, 2047)
    dsize = _bit_lengths(diff)
    dextra = np.where(diff >= 0, diff, diff + (1 << dsize) - 1)
    dcode = np.where(is_luma, luma_dc[0][dsize], chroma_dc[0][dsize])
    dlen = np.where(is_luma, luma_dc[1][dsize], chroma_dc[1][dsize])
    items_block = [np.arange(n)]
    items_key = [np.zeros(n, dtype=np.int64)]
    items_code = [(dcode << dsize) | dextra]
    items_len = [dlen + dsize]

    # AC: runs of zeros, ZRL for every 16 of them, then the (run, size) symbol and its bits
    ac = np.clip(zz[:, 1:].astype(np.int64), -1023, 1023)
    blk, pos0 = np.nonzero(ac)
    if blk.size:
        pos = pos0 + 1
        val = ac[blk, pos0]
        prev = np.empty_like(pos)
        prev[0] = 0
        prev[1:] = pos[:-1]
        first = np.ones(blk.size, dtype=bool)
        first[1:] = blk[1:] != blk[:-1]
        prev[first] = 0
        run = pos - prev - 1
        zrl = run // 16
        run = run % 16
        size = _bit_lengths(val)
        extra = np.where(val >= 0, val, val + (1 << size) - 1)
        sym = (run << 4) | size
        lum = is_luma[blk]
        code = np.where(lum, luma_ac[0][sym], chroma_ac[0][sym])
        clen = np.where(lum, luma_ac[1][sym], chroma_ac[1][sym])
        items_block.append(blk)
        items_key.append(pos * 4 + 3)
        items_code.append((code << size) | extra)
        items_len.append(clen + size)
        if zrl.any():
            reps = np.repeat(np.arange(blk.size), zrl)
            offset = np.arange(reps.size) - np.repeat(np.cumsum(zrl) - zrl, zrl)
            zl = is_luma[blk[reps]]
            items_block.append(blk[reps])
            items_key.append(pos[reps] * 4 + offset)
            items_code.append(np.where(zl, luma_ac[0][0xF0], chroma_ac[0][0xF0]))
            items_len.append(np.where(zl, luma_ac[1][0xF0], chroma_ac[1][0xF0]))
        last = np.zeros(n, dtype=np.int64)
        np.maximum.at(last, blk, pos)
    else:
        last = np.zeros(n, dtype=np.int64)
    eob = np.nonzero(last < 63)[0]
    items_block.append(eob)
    items_key.append(np.full(eob.size, 1000, dtype=np.int64))
    items_code.append(np.where(is_luma[eob], luma_ac[0][0x00], chroma_ac[0][0x00]))
    items_len.append(np.where(is_luma[eob], luma_ac[1][0x00], chroma_ac[1][0x00]))

    block = np.concatenate(items_block)
    key = np.concatenate(items_key)
    codes = np.concatenate(items_code)
    lengths = np.concatenate(items_len)
    order = np.lexsort((key, block))
    codes, lengths = codes[order], lengths[order]

    ends = np.cumsum(lengths)
    total = int(ends[-1]) if ends.size else 0
    padded = (total + 7) // 8 * 8
    bits = np.ones(padded, dtype=np.uint8)  # the tail pads with 1-bits
    starts = ends - lengths
    for j in range(int(lengths.max()) if lengths.size else 0):
        m = lengths > j
        bits[starts[m] + j] = (codes[m] >> (lengths[m] - 1 - j)) & 1
    data = np.packbits(bits).tobytes()
    return data.replace(b"\xff", b"\xff\x00")


def encode_numpy(rgb: Any, quality: int = 80) -> bytes:
    """Baseline JFIF, 4:2:0, from an (H, W, 3) uint8 array."""
    import numpy as np

    img = np.asarray(rgb)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("expected an (H, W, 3) RGB array")
    height, width = img.shape[:2]
    f = img.astype(np.float32)
    y = f @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    cb = f @ np.array([-0.168736, -0.331264, 0.5], dtype=np.float32) + 128
    cr = f @ np.array([0.5, -0.418688, -0.081312], dtype=np.float32) + 128
    ph, pw = -(-height // 16) * 16, -(-width // 16) * 16
    pad = ((0, ph - height), (0, pw - width))
    y, cb, cr = (np.pad(p, pad, mode="edge") for p in (y, cb, cr))
    cb = cb.reshape(ph // 2, 2, pw // 2, 2).mean(axis=(1, 3))
    cr = cr.reshape(ph // 2, 2, pw // 2, 2).mean(axis=(1, 3))

    c = _dct_matrix()
    zig = np.array(_ZIGZAG)
    qy, qc = _quant(_LUMA_Q, quality), _quant(_CHROMA_Q, quality)

    def coefficients(plane: Any, q: Any) -> Any:
        b = _blocks(plane.astype(np.float64) - 128.0)
        d = np.einsum("ux,ijxy,vy->ijuv", c, b, c, optimize=True)
        flat = d.reshape(d.shape[0], d.shape[1], 64)
        return np.round(flat / q).astype(np.int64)[..., zig]

    zy = coefficients(y, qy)  # (MH*2, MW*2, 64)
    zcb = coefficients(cb, qc)  # (MH, MW, 64)
    zcr = coefficients(cr, qc)
    mh, mw = zcb.shape[:2]
    # MCU order: Y00 Y01 Y10 Y11 Cb Cr
    ymcu = zy.reshape(mh, 2, mw, 2, 64).transpose(0, 2, 1, 3, 4).reshape(mh, mw, 4, 64)
    mcu = np.concatenate([ymcu, zcb[:, :, None], zcr[:, :, None]], axis=2).reshape(-1, 64)
    comp = np.tile(np.array([0, 0, 0, 0, 1, 2]), mh * mw)
    scan = _entropy(mcu, comp)

    out = io.BytesIO()
    out.write(b"\xff\xd8")
    out.write(_segment(0xFFE0, b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"))
    dqt = bytes([0]) + bytes(int(v) for v in qy[zig]) + bytes([1]) + bytes(int(v) for v in qc[zig])
    out.write(_segment(0xFFDB, dqt))
    sof = struct.pack(">BHHB", 8, height, width, 3) + bytes([1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1])
    out.write(_segment(0xFFC0, sof))
    dht = (
        _dht(0, 0, _DC_LUMA)
        + _dht(1, 0, _AC_LUMA)
        + _dht(0, 1, _DC_CHROMA)
        + _dht(1, 1, _AC_CHROMA)
    )
    out.write(_segment(0xFFC4, dht))
    out.write(_segment(0xFFDA, bytes([3, 1, 0x00, 2, 0x11, 3, 0x11, 0, 63, 0])))
    out.write(scan)
    out.write(b"\xff\xd9")
    return out.getvalue()


def pillow_available() -> bool:
    try:
        import PIL.Image  # noqa: F401
    except Exception:
        return False
    return True


def encode(rgb: Any, quality: int = 80, *, prefer_pillow: bool = True) -> tuple[bytes, str]:
    """JPEG bytes and which encoder made them (``"pillow"`` or ``"numpy"``)."""
    if prefer_pillow and pillow_available():
        import numpy as np
        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(rgb)).save(buffer, format="JPEG", quality=quality)
        return buffer.getvalue(), "pillow"
    return encode_numpy(rgb, quality), "numpy"
