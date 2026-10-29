"""duck-body's copy of Duck Core's integer trigonometry answers the same integers."""

from __future__ import annotations

import random

from duckmind.duckbody import fixedmath as F
from duckmind.duckcore import fixed as D


def test_the_copy_equals_duck_core_s_fixed_point_maths() -> None:
    rng = random.Random(7)
    angles = [*range(-7000, 7000, 37), *(rng.randint(-(10**6), 10**6) for _ in range(400))]
    for angle in angles:
        assert F.sin_mrad(angle) == D.sin_mrad(angle)
        assert F.cos_mrad(angle) == D.cos_mrad(angle)
        assert F.signed_mrad(angle) == D.signed_mrad(angle)
        assert F.wrap_mrad(angle) == D.wrap_mrad(angle)
    for _ in range(600):
        dx, dy = rng.randint(-5000, 5000), rng.randint(-5000, 5000)
        assert F.atan2_mrad(dy, dx) == D.atan2_mrad(dy, dx)
        value = rng.randint(-10, 10**12)
        assert F.isqrt(value) == D.isqrt(value)
    assert F.atan2_mrad(0, 0) == D.atan2_mrad(0, 0) == 0
    assert (F.SCALE, F.TAU_MRAD) == (D.SCALE, D.TAU_MRAD)
