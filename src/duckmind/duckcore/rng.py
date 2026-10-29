"""Counter-based randomness: every draw is addressable, logged and replayable.

A draw is ``BLAKE2b(seed ‖ table_hash ‖ purpose ‖ counter)``. There is no hidden
generator state, so adding a new expression cannot shift the draws of decisions, and
a verifier can recompute any logged draw from its key.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

from duckmind.duckcore.fixed import SCALE


def draw(seed: int, table_hash: str, purpose: str, counter: int) -> int:
    """A 64-bit unsigned integer for this exact key."""

    key = f"{seed}|{table_hash}|{purpose}|{counter}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")


def unit(raw: int) -> int:
    """Map a 64-bit draw onto ``[0, SCALE)``."""

    return (raw * SCALE) >> 64


def below(raw: int, probability: int) -> bool:
    """True with ``probability`` (fixed point) for this draw."""

    return unit(raw) < probability


def pick_index(raw: int, count: int) -> int:
    if count <= 0:
        raise ValueError("nothing to pick from")
    return (raw * count) >> 64


def pick_weighted(raw: int, weights: Mapping[str, int]) -> str:
    """Choose a key with probability proportional to its integer weight.

    Iteration order is the sorted key order, never dict or set order, so the same
    weights and draw give the same answer on every interpreter.
    """

    keys: Sequence[str] = sorted(key for key, weight in weights.items() if weight > 0)
    total = sum(weights[key] for key in keys)
    if total <= 0:
        raise ValueError("no positive weight")
    target = (raw * total) >> 64
    running = 0
    for key in keys:
        running += weights[key]
        if target < running:
            return key
    return keys[-1]
