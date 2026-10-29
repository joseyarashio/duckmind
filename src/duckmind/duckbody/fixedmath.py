"""duck-body's integer trigonometry (spec §14.2: the same integers on the Mac and the board).

A copy of the few helpers of :mod:`duckmind.duckcore.fixed` that duck-body's walk and look plans
use. It is a copy, not an import, because importing ``duckmind.duckcore.fixed`` runs the whole
``duckmind.duckcore`` package (its engine, day cycle …) on the board, and Duck Core is frozen in
phase 3. ``tests/test_duckbody_fixedmath.py`` pins every function equal to Duck Core's.
"""

from __future__ import annotations

SCALE = 10_000
TAU_MRAD = 6283  # one full turn, 2π rad, in whole milliradians
_PI_NRAD = 3_141_592_654  # π in nanoradians
_NRAD = 1_000_000_000
_QUARTER_MRAD = 1571  # π/2, rounded up to a whole milliradian
_HALF_MRAD = 3142
_MEMO_MAX = 200_000
_SIN_MEMO: dict[int, int] = {}
_COS_MEMO: dict[int, int] = {}


def wrap_mrad(angle: int) -> int:
    """A heading in [0, TAU_MRAD)."""
    return angle % TAU_MRAD


def _sin_nrad(x: int) -> int:
    x %= 2 * _PI_NRAD
    if x > _PI_NRAD:
        x -= 2 * _PI_NRAD  # now in (−π, π]
    if x > _PI_NRAD // 2:
        x = _PI_NRAD - x
    elif x < -_PI_NRAD // 2:
        x = -_PI_NRAD - x
    term = total = x
    n = 1
    while term:
        term = -term * x // _NRAD * x // _NRAD // ((2 * n) * (2 * n + 1))
        total += term
        n += 1
    return total * SCALE // _NRAD


def sin_mrad(angle: int) -> int:
    """sin(angle / 1000) scaled by ``SCALE`` (exact integer arithmetic)."""
    value = _SIN_MEMO.get(angle)
    if value is None:
        if len(_SIN_MEMO) > _MEMO_MAX:
            _SIN_MEMO.clear()
        value = _SIN_MEMO[angle] = _sin_nrad(angle * 1_000_000)
    return value


def cos_mrad(angle: int) -> int:
    """cos(angle / 1000) scaled by ``SCALE``."""
    value = _COS_MEMO.get(angle)
    if value is None:
        if len(_COS_MEMO) > _MEMO_MAX:
            _COS_MEMO.clear()
        value = _COS_MEMO[angle] = _sin_nrad(angle * 1_000_000 + _PI_NRAD // 2)
    return value


def signed_mrad(angle: int) -> int:
    """An angle difference folded into [−3141, 3141] milliradians."""
    return (angle + 3141) % TAU_MRAD - 3141


def isqrt(value: int) -> int:
    """Integer square root (exact; ⌊√value⌋, 0 for negatives)."""
    if value <= 0:
        return 0
    root = 1 << ((value.bit_length() + 1) // 2)
    while True:
        smaller = (root + value // root) // 2
        if smaller >= root:
            return root
        root = smaller


def atan2_mrad(dy: int, dx: int) -> int:
    """The heading (mrad, [0, TAU_MRAD)) of the vector (dx, dy), to one milliradian."""
    if dx == 0 and dy == 0:
        return 0
    ax, ay = abs(dx), abs(dy)
    low, high = 0, _QUARTER_MRAD  # the largest angle whose tangent is at most ay / ax
    while low < high:
        mid = (low + high + 1) // 2
        if ay * cos_mrad(mid) - ax * sin_mrad(mid) >= 0:
            low = mid
        else:
            high = mid - 1
    if dx >= 0 and dy >= 0:
        angle = low
    elif dx < 0 and dy >= 0:
        angle = _HALF_MRAD - low
    elif dx < 0:
        angle = _HALF_MRAD + low
    else:
        angle = TAU_MRAD - low
    return wrap_mrad(angle)


__all__ = [
    "SCALE",
    "TAU_MRAD",
    "atan2_mrad",
    "cos_mrad",
    "isqrt",
    "signed_mrad",
    "sin_mrad",
    "wrap_mrad",
]
