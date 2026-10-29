"""M6 of MACHINE-VERIFY-v2.md: a whole life per seed, a profile of it, and a guess of the temperament.

Pure functions over recorded lives (no simulator). A life record (schema ``mv-life-1``, one JSON
line) is one 2-hour run of one seed of one temperament::

    {"schema": "mv-life-1", "layer": "L1", "preset": str, "seed": int, "segment": "day" | "night",
     "dt_s": 0.2, "day_minute0": float,
     "cols": {"behaviour": [str], "bout": [int], "reflex": [str | None]},
     "stimuli": [{"kind": str, "idx": int}],   # the tick each stimulus was delivered
     "error": str | None}

A seed's life is its day run plus its night run (CHOICES §六). Its profile is a vector of plain
numbers in natural units (no standardising, no weights): the time share of every behaviour, then
for each of the nine stimuli the share of times it changed the behaviour, the share of times it
startled or froze the duck, and how fast it answered (seconds / 3).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from duckmind.verify import spec_v2 as S

LIFE_SCHEMA = "mv-life-1"
Life = dict[str, Any]
Key = tuple[str, int]  # (temperament, seed)


def validate_life(rec: Life) -> list[str]:
    problems: list[str] = []
    if rec.get("schema") != LIFE_SCHEMA:
        problems.append("schema")
    if rec.get("error"):
        problems.append(f"recorded error: {rec['error']}")
    cols = rec.get("cols") or {}
    lengths = {len(cols.get(name, ())) for name in ("behaviour", "bout", "reflex")}
    if len(lengths) != 1 or 0 in lengths:
        problems.append("columns missing or of different length")
        return problems
    n = lengths.pop()
    if n < int(S.LIFE_SECONDS * S.LIFE_HZ) - 1:
        problems.append(f"life shorter than planned ({n} ticks)")
    want = S.STIMULI_PER_RUN
    stimuli = rec.get("stimuli") or []
    if len(stimuli) != want:
        problems.append(f"{len(stimuli)} stimuli, planned {want}")
    for s in stimuli:
        idx = s.get("idx")
        if not isinstance(idx, int) or not 1 <= idx < n:
            problems.append("stimulus tick out of range")
            break
    return problems


# ── one stimulus: what the duck did in the next 3 seconds ──────────────────────────


def reaction(cols: Mapping[str, Sequence[Any]], idx: int, dt: float) -> tuple[bool, bool, float]:
    """(changed the behaviour, startled or froze, seconds to the first sign of an answer).

    ``changed``: the behaviour or the bout differs from the tick before the stimulus at some tick
    of the 3 s window. ``startled``: a startle or freeze reflex or behaviour in the window (M3's
    own test). The first sign: the first tick of the window where the behaviour or bout differs, or
    a reflex differs from the one just before and is not empty; none in the window = 3 s."""
    behaviour, bout, reflex = cols["behaviour"], cols["bout"], cols["reflex"]
    n = len(behaviour)
    base = idx - 1
    last = min(n - 1, idx + round(S.WINDOW_S / dt))
    changed = startled = False
    first: int | None = None
    for i in range(idx, last + 1):
        differs = behaviour[i] != behaviour[base] or bout[i] != bout[base]
        if differs:
            changed = True
        if reflex[i] in S.STARTLE_NAMES or behaviour[i] in S.STARTLE_NAMES:
            startled = True
        sign = differs or (reflex[i] is not None and reflex[i] != reflex[base])
        if sign and first is None:
            first = i
    latency = S.LATENCY_CAP_S if first is None else min(S.LATENCY_CAP_S, (first - idx) * dt)
    return changed, startled, latency


# ── a seed's life: the profile ─────────────────────────────────────────────────────


def behaviour_names(lives: Iterable[Life]) -> list[str]:
    """Every behaviour any life showed, sorted (the dimensions of the time-share part)."""
    names: set[str] = set()
    for rec in lives:
        names.update(set(rec["cols"]["behaviour"]))
    return sorted(names)


def profile(segments: Sequence[Life], names: Sequence[str]) -> list[float]:
    """The profile of one seed's life (its day and night runs together)."""
    counts: Counter[str] = Counter()
    total = 0
    by_kind: dict[str, list[tuple[bool, bool, float]]] = {k: [] for k in S.LIFE_STIM_ORDER}
    for seg in segments:
        counts.update(seg["cols"]["behaviour"])
        total += len(seg["cols"]["behaviour"])
        for stim in seg["stimuli"]:
            by_kind[stim["kind"]].append(reaction(seg["cols"], stim["idx"], seg["dt_s"]))
    vec = [counts.get(name, 0) / total if total else 0.0 for name in names]
    for kind in S.LIFE_STIM_ORDER:
        rows = by_kind[kind]
        if not rows:
            vec += [0.0, 0.0, 1.0]  # never delivered: no change, no startle, no answer
            continue
        vec += [
            sum(1 for r in rows if r[0]) / len(rows),
            sum(1 for r in rows if r[1]) / len(rows),
            sum(r[2] for r in rows) / len(rows) / S.LATENCY_CAP_S,
        ]
    return vec


def signature_seen(segments: Sequence[Life], acts: Sequence[str]) -> tuple[list[str], list[str]]:
    """(the acts that appeared in the life, the ones that never did)."""
    seen: set[str] = set()
    for seg in segments:
        seen.update(set(seg["cols"]["behaviour"]))
    return [a for a in acts if a in seen], [a for a in acts if a not in seen]


# ── guess the temperament: leave one seed out, nearest centroid ────────────────────


def euclid(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def centroid(vectors: Sequence[Sequence[float]]) -> list[float]:
    return [sum(col) / len(vectors) for col in zip(*vectors, strict=True)]


def standardise(
    train: Sequence[Sequence[float]], other: Sequence[float]
) -> tuple[list[list[float]], list[float]]:
    """(sensitivity only) z-scores with the training fold's mean and spread; a flat column = 0."""
    dims = len(other)
    mean = [sum(v[d] for v in train) / len(train) for d in range(dims)]
    sd = [math.sqrt(sum((v[d] - mean[d]) ** 2 for v in train) / len(train)) for d in range(dims)]

    def z(v: Sequence[float]) -> list[float]:
        return [(v[d] - mean[d]) / sd[d] if sd[d] > 1e-12 else 0.0 for d in range(dims)]

    return [z(v) for v in train], z(other)


def guess(
    profiles: Mapping[Key, Sequence[float]],
    presets: Sequence[str],
    *,
    columns: Sequence[int] | None = None,
    standardised: bool = False,
) -> dict[Key, str | None]:
    """Leave-one-seed-out nearest centroid. Every (temperament, seed) in turn is taken out; each
    temperament's mean profile is made of the other seeds; the one taken out goes to the nearest
    temperament. A tie (distances equal to 1e-12) or a temperament with nothing left is ``None``
    (counted as a wrong guess). ``columns``: use only these dimensions (sensitivity)."""

    def cut(v: Sequence[float]) -> list[float]:
        return list(v) if columns is None else [v[c] for c in columns]

    out: dict[Key, str | None] = {}
    for key, vec in profiles.items():
        pool = {k: cut(v) for k, v in profiles.items() if k != key}
        held = cut(vec)
        if standardised:
            train, held = standardise(list(pool.values()), held)
            pool = dict(zip(pool, train, strict=True))
        by_preset: dict[str, list[list[float]]] = {p: [] for p in presets}
        for (preset, _seed), v in pool.items():
            by_preset[preset].append(v)
        dists = {p: euclid(held, centroid(v)) for p, v in by_preset.items() if v}
        if len(dists) < len(presets):
            out[key] = None
            continue
        best = min(dists.values())
        winners = [p for p, d in dists.items() if d - best <= 1e-12]
        out[key] = winners[0] if len(winners) == 1 else None
    return out


def accuracy(guesses: Mapping[Key, str | None]) -> tuple[int, int]:
    """(right, total)."""
    return sum(1 for (preset, _), g in guesses.items() if g == preset), len(guesses)


def confusion(
    guesses: Mapping[Key, str | None], presets: Sequence[str]
) -> dict[str, dict[str, int]]:
    """truth -> guess -> count (``none`` for a tie)."""
    table = {p: {q: 0 for q in (*presets, "none")} for p in presets}
    for (preset, _), g in guesses.items():
        table[preset][g if g is not None else "none"] += 1
    return table
