"""Fixed-point helpers so decisions replay identically on every platform.

Values are integers scaled by ``SCALE``: 10_000 means 1.0. Decision code must use these
helpers instead of floats or ``math`` so that a Mac and the robot's Cortex-A55 compute
the same utilities and draw the same choices.
"""

from __future__ import annotations

SCALE = 10_000


def fp(numerator: int, denominator: int = 1) -> int:
    """Exact rational constant as a fixed-point integer (rounded half up)."""

    return (numerator * SCALE * 2 + denominator) // (denominator * 2)


def clamp(value: int, low: int = 0, high: int = SCALE) -> int:
    return low if value < low else high if value > high else value


def mul(a: int, b: int) -> int:
    return a * b // SCALE


def approach(value: int, target: int, rate: int) -> int:
    """Move ``value`` toward ``target`` by ``rate`` (fixed-point fraction of the gap)."""

    return value + (target - value) * rate // SCALE


def power(base: int, exponent: int) -> int:
    """``base ** exponent`` in fixed point, by repeated squaring."""

    if exponent < 0:
        raise ValueError("negative exponent")
    result = SCALE
    factor = base
    while exponent:
        if exponent & 1:
            result = result * factor // SCALE
        factor = factor * factor // SCALE
        exponent >>= 1
    return result


def decay(value: int, target: int, keep_per_step: int, steps: int) -> int:
    """Relax toward ``target``, keeping ``keep_per_step`` of the gap each step."""

    if steps <= 0:
        return value
    return target + (value - target) * power(keep_per_step, steps) // SCALE


def to_unit(value: int) -> float:
    """Presentation only: never feed the result back into decisions."""

    return round(value / SCALE, 4)


# Angles on the floor are integer milliradians. Sine and cosine come from an integer Taylor
# series (no ``math``), so a walk's path is bit-identical on every platform.
TAU_MRAD = 6283  # one full turn, 2π rad, in whole milliradians
_PI_NRAD = 3_141_592_654  # π in nanoradians
_NRAD = 1_000_000_000


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


# The series is exact but slow, and walks integrate their path ten times a second: each
# result is kept (the same integer every time), so a year of walking stays fast.
_SIN_MEMO: dict[int, int] = {}
_COS_MEMO: dict[int, int] = {}
_MEMO_MAX = 200_000


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


_QUARTER_MRAD = 1571  # π/2, rounded up to a whole milliradian
_HALF_MRAD = 3142


def atan2_mrad(dy: int, dx: int) -> int:
    """The heading (mrad, [0, TAU_MRAD)) of the vector (dx, dy): integer bisection on the
    same sine and cosine the walks use, to one milliradian."""

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
