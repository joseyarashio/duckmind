"""b:feasibility (2026-09-28): the v5 probe's pure parts (cells, keys, statistics)."""

from __future__ import annotations

from duckmind.skilltrain import bsk_probe as P


def test_cells_cover_the_four_skills_with_unique_keys() -> None:
    cells = P.cells()
    keys = [c.key for c in cells]
    assert len(keys) == len(set(keys))
    fams = {c.family for c in cells}
    assert fams == {"turn", "march", "bow", "lie", "mat_up", "mat_dn"}
    assert any(c.soft for c in cells)


def test_seeds_are_stable_and_distinct() -> None:
    a = P.seed_of("turn/stand/vx0.00/yaw1.0", 0)
    assert a == P.seed_of("turn/stand/vx0.00/yaw1.0", 0)
    assert a != P.seed_of("turn/stand/vx0.00/yaw1.0", 1)


def test_rate_format() -> None:
    assert P.rate(0, 30) == "0/30 [0–11%]"
    assert P.rate(30, 30) == "30/30 [89–100%]"


def test_trained_cells_mirror_the_v5_cells() -> None:
    trained = P.trained_cells()
    keys = [c.key for c in trained]
    assert len(keys) == len(set(keys))
    assert all(c.skill for c in trained)
    assert {c.skill for c in trained} == {
        "turn_left",
        "turn_right",
        "march",
        "bow",
        "lie_low",
        "step_up",
        "step_down",
    }
    assert {c.family for c in trained} == {"turn", "march", "bow", "lie", "mat_up", "mat_dn"}
