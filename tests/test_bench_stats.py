"""The benchmark's statistics (duckmind.bench.stats): known values, and the stopping rule's error rate."""

from __future__ import annotations

import random

import pytest

from duckmind.bench import stats as S


def test_wilson_matches_the_published_same_room_interval() -> None:
    lo, hi = S.wilson(170, 280)  # the three live measurements of main together: 61 % (55–66)
    assert round(100 * lo) == 55 and round(100 * hi) == 66
    assert S.wilson(0, 0) == (0.0, 1.0)
    lo, hi = S.wilson(0, 20)
    assert lo == 0.0 and 0.10 < hi < 0.20


def test_mcnemar_exact_is_the_two_sided_binomial() -> None:
    assert S.mcnemar_exact(13, 1) == pytest.approx(30 / 2**14)  # GoThere-contact's 13 vs 1
    assert S.mcnemar_exact(1, 13) == S.mcnemar_exact(13, 1)
    assert S.mcnemar_exact(5, 4) == 1.0
    assert S.mcnemar_exact(0, 0) == 1.0
    assert S.mcnemar_exact(0, 6) == pytest.approx(2 / 64)


def test_paired_difference_and_its_interval() -> None:
    d, lo, hi = S.paired_diff(10, 30, 200)
    assert d == pytest.approx(0.10)
    assert 0.0 < lo < d < hi < 0.20
    d, lo, hi = S.paired_diff(20, 20, 200)
    assert d == 0.0 and lo < 0.0 < hi and lo == pytest.approx(-hi)


def test_clicks_needed_grow_as_the_gain_shrinks_and_pairing_saves_clicks() -> None:
    n5, n9, n15 = (S.paired_n(g, 0.25) for g in (0.05, 0.09, 0.15))
    assert n5 > n9 > n15 > 0
    assert (
        200 <= n9 <= 280
    )  # Connor's formula: about 240 pairs for +9 points when a quarter of the clicks differ
    assert (
        S.paired_n(0.09, 0.15) < n9 < S.paired_n(0.09, 0.35)
    )  # fewer differing clicks, fewer needed
    assert (
        S.unpaired_n(0.61, 0.09) > n9
    )  # two independent samples need more (the review's "about 436 a side")
    assert 400 <= S.unpaired_n(0.61, 0.09) <= 470
    assert S.paired_power(n9, 0.09, 0.25) == pytest.approx(0.80, abs=0.01)
    assert S.detectable(n9, 0.25) == pytest.approx(0.09, abs=0.003)


def test_sequential_stops_at_the_first_look_that_decides() -> None:
    better = S.sequential([(150, 5, 40)], looks=4, delta=0.05)
    assert [x.verdict for x in better] == ["better"]
    worse = S.sequential([(150, 40, 5)], looks=4)
    assert worse[-1].verdict == "worse"
    # no difference and a tight interval: B is not better by 5 points — stop, no more runs
    none = S.sequential(
        [(150, 20, 20), (300, 30, 30), (450, 35, 35), (600, 50, 50)], looks=4, delta=0.05
    )
    assert [x.verdict for x in none] == ["continue", "continue", "no_gain"]
    # too few pairs to say anything at the last planned look
    weak = S.sequential([(40, 8, 10), (80, 16, 20)], looks=2, delta=0.05)
    assert [x.verdict for x in weak] == ["continue", "undecided"]
    with pytest.raises(ValueError):
        S.sequential([(10, 1, 1)], looks=11)


def test_sequential_keeps_its_error_rate_over_four_looks() -> None:
    """Two arms that are the same: looking four times must not call a difference more than about 5 % of the time
    (looking four times at an unadjusted 5 % test would call one about 13 % of the time)."""
    rng = random.Random(7)
    wrong = naive = 0
    runs, per_look, psi = 2000, 150, 0.30
    for _ in range(runs):
        a = b = 0
        seen = []
        tripped = False
        for look in range(1, 5):
            for _ in range(per_look):
                u = rng.random()
                if u < psi / 2:
                    a += 1
                elif u < psi:
                    b += 1
            seen.append((look * per_look, a, b))
            tripped = tripped or S.mcnemar_exact(a, b) < 0.05
        verdict = S.sequential(seen, looks=4, delta=0.05)[-1].verdict
        wrong += verdict in ("better", "worse")
        naive += tripped
    assert wrong / runs <= 0.065
    assert naive / runs > 0.09
