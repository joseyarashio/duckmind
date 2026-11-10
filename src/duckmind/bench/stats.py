"""The statistics of a paired click comparison: nothing here knows about ducks.

Two arms answer the same clicks, so each click is a pair: both arrived, only A, only B, neither. The test is the exact
McNemar test on the two discordant counts; rates carry Wilson intervals; the difference B − A carries the Agresti–Min
interval for paired proportions. :func:`paired_n` says how many clicks a wanted gain needs before anything is run, and
:func:`sequential` is the stopping rule that lets a comparison end at the first look that already decides it, without
spending its error rate on the looks (Pocock's constant boundary, read as repeated confidence intervals).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

_N = NormalDist()
# Pocock's two-sided critical values for an overall 5 % error over K equally spaced looks (K = 1 … 10)
POCOCK_Z = (1.960, 2.178, 2.289, 2.361, 2.413, 2.453, 2.485, 2.512, 2.535, 2.555)


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """The Wilson score interval of ``k`` successes in ``n`` (95 % by default); (0, 1) for no data."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    d = 1.0 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return max(0.0, mid - half), min(1.0, mid + half)


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """The exact two-sided McNemar p-value of the discordant counts (a fair coin on ``only_a + only_b`` tosses)."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0**n
    return min(1.0, 2.0 * tail)


def paired_diff(
    only_a: int, only_b: int, n: int, z: float = 1.959964
) -> tuple[float, float, float]:
    """B − A as a share of the ``n`` pairs and its interval (Agresti–Min: half a click added to each discordant count)."""
    if n <= 0:
        return 0.0, -1.0, 1.0
    b, c, m = only_a + 0.5, only_b + 0.5, n + 2.0
    d = (c - b) / m
    se = math.sqrt(max(0.0, (b + c) - (c - b) ** 2 / m)) / m
    return (only_b - only_a) / n, max(-1.0, d - z * se), min(1.0, d + z * se)


def paired_n(delta: float, discordant: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Pairs needed for the exact-McNemar-like test to see a true gain of ``delta`` (a share, e.g. 0.09) when a share
    ``discordant`` of the clicks end differently between the arms (Connor 1987). ``discordant`` is at least ``|delta|``."""
    psi = max(float(discordant), abs(delta) + 1e-9)
    za, zb = _N.inv_cdf(1.0 - alpha / 2.0), _N.inv_cdf(power)
    return int(
        math.ceil(
            (za * math.sqrt(psi) + zb * math.sqrt(psi - delta * delta)) ** 2 / (delta * delta)
        )
    )


def unpaired_n(p0: float, delta: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Clicks per arm for two independent samples to see ``p0`` → ``p0 + delta`` (for contrast with :func:`paired_n`)."""
    p1 = p0 + delta
    pm = (p0 + p1) / 2.0
    za, zb = _N.inv_cdf(1.0 - alpha / 2.0), _N.inv_cdf(power)
    return int(
        math.ceil(
            (za * math.sqrt(2 * pm * (1 - pm)) + zb * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2
            / (delta * delta)
        )
    )


def paired_power(n: int, delta: float, discordant: float, alpha: float = 0.05) -> float:
    """The chance that ``n`` pairs show a true gain of ``delta`` (the inverse of :func:`paired_n`)."""
    psi = max(float(discordant), abs(delta) + 1e-9)
    za = _N.inv_cdf(1.0 - alpha / 2.0)
    return _N.cdf(
        (abs(delta) * math.sqrt(n) - za * math.sqrt(psi)) / math.sqrt(psi - delta * delta)
    )


def detectable(n: int, discordant: float, alpha: float = 0.05, power: float = 0.80) -> float:
    """The smallest gain ``n`` pairs can show with that power."""
    lo, hi = 1e-4, min(0.99, max(discordant, 1e-3))
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if paired_power(n, mid, discordant, alpha) >= power:
            hi = mid
        else:
            lo = mid
    return hi


@dataclass(frozen=True)
class Look:
    """One look of a sequential comparison."""

    n: int
    only_a: int
    only_b: int
    p: float
    diff: float
    lo: float
    hi: float
    verdict: str  # "better" | "worse" | "no_gain" | "continue" | "undecided"


def sequential(pairs: list[tuple[int, int, int]], looks: int, delta: float = 0.05) -> list[Look]:
    """The stopping rule. ``pairs``: per look so far (cumulative) ``(n, only_a, only_b)``; ``looks``: how many looks the
    plan has in all (1 … 10, equally spaced); ``delta``: the gain worth having (a share).

    At each look the difference B − A gets a repeated confidence interval at Pocock's level for ``looks`` looks (all of
    them hold together with 95 % confidence, however many are taken). The comparison stops at the first look where

    * the interval is wholly above 0 — ``better`` — or wholly below 0 — ``worse``;
    * the interval is wholly below ``delta`` — ``no_gain``: B is not better by the gain worth having.

    Otherwise ``continue``; at the last planned look, ``undecided`` (run no more: the plan was too small for this gain)."""
    if not 1 <= looks <= len(POCOCK_Z):
        raise ValueError(f"looks: 1 to {len(POCOCK_Z)}")
    z = POCOCK_Z[looks - 1]
    out: list[Look] = []
    for i, (n, a, b) in enumerate(pairs[:looks]):
        diff, lo, hi = paired_diff(a, b, n, z)
        if lo > 0.0:
            verdict = "better"
        elif hi < 0.0:
            verdict = "worse"
        elif hi < delta:
            verdict = "no_gain"
        elif i + 1 >= looks:
            verdict = "undecided"
        else:
            verdict = "continue"
        out.append(Look(n, a, b, mcnemar_exact(a, b), diff, lo, hi, verdict))
        if verdict != "continue":
            break
    return out


__all__ = [
    "POCOCK_Z",
    "Look",
    "detectable",
    "mcnemar_exact",
    "paired_diff",
    "paired_n",
    "paired_power",
    "sequential",
    "unpaired_n",
    "wilson",
]
