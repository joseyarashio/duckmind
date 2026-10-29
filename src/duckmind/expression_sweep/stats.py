"""Counting honestly: every rate with its n, its count and a Wilson 95% interval.

Pure Python so anyone can recompute a summary from the raw per-episode records without
the GPU stack.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

Z95 = 1.959963984540054  # two-sided 95%


def wilson(count: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for ``count`` successes in ``n`` trials; (0, 1) when n is 0."""
    if n <= 0:
        return (0.0, 1.0)
    if not 0 <= count <= n:
        raise ValueError(f"count {count} outside 0..{n}")
    p = count / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def proportion(count: int, n: int) -> dict[str, Any]:
    """``{n, count, rate, wilson95}``: the shape every rate in a result takes."""
    low, high = wilson(count, n)
    return {
        "n": n,
        "count": count,
        "rate": round(count / n, 5) if n else None,
        "wilson95": [round(low, 5), round(high, 5)],
    }


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (``q`` in 0..1); 0.0 for no values."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    k = (len(ordered) - 1) * q
    low, high = math.floor(k), math.ceil(k)
    return ordered[low] + (ordered[high] - ordered[low]) * (k - low)


def describe(values: Sequence[float], digits: int = 4) -> dict[str, Any]:
    """n, mean, sd, a normal-approximation 95% interval of the mean, and the spread."""
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    n = len(finite)
    if n == 0:
        return {"n": 0}
    mean = sum(finite) / n
    sd = math.sqrt(sum((value - mean) ** 2 for value in finite) / (n - 1)) if n > 1 else 0.0
    half = Z95 * sd / math.sqrt(n)
    return {
        "n": n,
        "mean": round(mean, digits),
        "sd": round(sd, digits),
        "ci95": [round(mean - half, digits), round(mean + half, digits)],
        "p5": round(percentile(finite, 0.05), digits),
        "p50": round(percentile(finite, 0.5), digits),
        "p95": round(percentile(finite, 0.95), digits),
        "min": round(min(finite), digits),
        "max": round(max(finite), digits),
    }


def difference(a: Sequence[float], b: Sequence[float], digits: int = 4) -> dict[str, Any]:
    """Mean of ``a`` minus mean of ``b`` with a normal-approximation 95% interval (unpaired)."""
    da, db = describe(a, digits=12), describe(b, digits=12)
    if da["n"] < 2 or db["n"] < 2:
        return {"n_a": da["n"], "n_b": db["n"]}
    diff = da["mean"] - db["mean"]
    se = math.sqrt(da["sd"] ** 2 / da["n"] + db["sd"] ** 2 / db["n"])
    ratio = da["mean"] / db["mean"] if db["mean"] else None
    return {
        "n_a": da["n"],
        "n_b": db["n"],
        "diff": round(diff, digits),
        "ci95": [round(diff - Z95 * se, digits), round(diff + Z95 * se, digits)],
        "ratio_of_means": round(ratio, digits) if ratio is not None else None,
    }
