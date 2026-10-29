"""M6 of MACHINE-VERIFY-v3.md: the same profile of a seed's life as v2 (``lives.py``, untouched), but
every number is first turned into a z score with the mean and standard deviation of the TRAINING seeds
only — never of the seed being guessed — and a number that does not move at all among the training
seeds (standard deviation 0) is dropped.

Pure functions. The guess is leave one seed out, nearest centroid in z-score space; a tie is a wrong
guess (v2's rule, unchanged).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from duckmind.verify import lives as LV
from duckmind.verify import spec_v3 as S

Key = LV.Key


@dataclass(frozen=True)
class ZFit:
    """What the training seeds say about each dimension: its mean, its (population) standard
    deviation, and which dimensions are kept (standard deviation above ``S.Z_SD_MIN``)."""

    mean: tuple[float, ...]
    sd: tuple[float, ...]
    keep: tuple[int, ...]


def fit_z(train: Sequence[Sequence[float]]) -> ZFit:
    """Mean and standard deviation of each dimension over ``train`` ONLY (population form, divide by
    N: the same expression as ``lives.standardise``)."""
    if not train:
        raise ValueError("no training seeds")
    dims = len(train[0])
    n = len(train)
    mean = tuple(sum(v[d] for v in train) / n for d in range(dims))
    sd = tuple(math.sqrt(sum((v[d] - mean[d]) ** 2 for v in train) / n) for d in range(dims))
    keep = tuple(d for d in range(dims) if sd[d] > S.Z_SD_MIN)
    return ZFit(mean, sd, keep)


def apply_z(vec: Sequence[float], fit: ZFit) -> list[float]:
    """z scores of the kept dimensions of ``vec``."""
    return [(vec[d] - fit.mean[d]) / fit.sd[d] for d in fit.keep]


def guess_z(
    profiles: Mapping[Key, Sequence[float]],
    presets: Sequence[str],
    *,
    columns: Sequence[int] | None = None,
) -> dict[Key, str | None]:
    """Leave-one-seed-out nearest centroid on z scores. Every (temperament, seed) in turn is taken
    out; the mean and standard deviation of every dimension come from the other seeds alone; the
    other seeds' z scores give each temperament its centroid; the one taken out is turned into z
    scores with the SAME training statistics and goes to the nearest centroid. A tie (distances equal
    to 1e-12) or a temperament with nothing left is ``None`` (counted as a wrong guess).
    ``columns``: use only these dimensions (the report-only profiles)."""

    def cut(v: Sequence[float]) -> list[float]:
        return list(v) if columns is None else [v[c] for c in columns]

    out: dict[Key, str | None] = {}
    for key, vec in profiles.items():
        pool = {k: cut(v) for k, v in profiles.items() if k != key}
        held = cut(vec)
        by_preset: dict[str, list[list[float]]] = {p: [] for p in presets}
        if not pool or any(not any(k[0] == p for k in pool) for p in presets):
            out[key] = None
            continue
        fit = fit_z(list(pool.values()))
        for (preset, _seed), v in pool.items():
            by_preset[preset].append(apply_z(v, fit))
        held_z = apply_z(held, fit)
        dists = {p: LV.euclid(held_z, LV.centroid(v)) for p, v in by_preset.items() if v}
        best = min(dists.values())
        winners = [p for p, d in dists.items() if d - best <= 1e-12]
        out[key] = winners[0] if len(winners) == 1 else None
    return out


accuracy = LV.accuracy
confusion = LV.confusion
