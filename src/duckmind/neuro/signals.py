"""Seeded structured noise and smooth movement shapes for the neural duck's motor side.

Everything here is a pure function of explicit state: an integer generator state (splitmix64),
an Ornstein–Uhlenbeck value, an oscillator phase. The same seed and the same input stream give
the same output on every machine; nothing reads a clock or a global random generator.

Why structured noise and not dice: an animal's variability is correlated in time (a speed that
drifts, a head that wanders and returns) and spread over several time scales. An OU process is
a leaky integrator of noise (one time scale); :class:`Pink` sums three of them (0.3 s, 2 s,
12 s) for the long-memory drift real movement has. Movement profiles use minimum jerk (the
smoothest point-to-point reach, Flash & Hogan 1985), which is how biological reaches look.

Phase 5 (fixed point): splitmix64 is already integer; :func:`gauss` is a sum of four uniforms
(no log/sqrt); an OU step is a multiply-add once ``dt/τ`` and ``σ·√(2dt/τ)`` are precomputed per
tick rate (SCALE 10_000 constants); :func:`min_jerk` is a quintic polynomial; sines can use
``duckmind.duckbody.fixedmath``'s table.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

MASK64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15


def mix64(x: int) -> int:
    """splitmix64's finaliser: a well-mixed 64-bit value for a 64-bit input."""
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & MASK64
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB & MASK64
    return x ^ (x >> 31)


def seed_from(*parts: object) -> int:
    """A 64-bit seed for these parts (BLAKE2b of their repr, like Duck Core's counter draws)."""
    key = "|".join(repr(part) for part in parts).encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")


def next_u64(state: int) -> tuple[int, int]:
    """splitmix64: (new state, a 64-bit value)."""
    state = (state + GOLDEN) & MASK64
    return state, mix64(state)


def uniform(state: int) -> tuple[int, float]:
    """(new state, a float in [0, 1))."""
    state, value = next_u64(state)
    return state, (value >> 11) * (1.0 / (1 << 53))


def gauss(state: int) -> tuple[int, float]:
    """(new state, an approximately standard normal value): the sum of four uniforms, centred
    and scaled to unit variance (Irwin–Hall; bounded at ±3.46, which a motor system wants)."""
    total = 0.0
    for _ in range(4):
        state, u = uniform(state)
        total += u
    return state, (total - 2.0) * math.sqrt(3.0)


def between(state: int, low: float, high: float) -> tuple[int, float]:
    state, u = uniform(state)
    return state, low + (high - low) * u


def lognormal(state: int, median: float, sigma: float) -> tuple[int, float]:
    """(new state, a log-normal value): median ``median``, log-spread ``sigma``. Bout and pause
    lengths of animals are close to log-normal (heavy right tail, never negative)."""
    state, g = gauss(state)
    return state, median * math.exp(sigma * g)


def chance(state: int, probability: float) -> tuple[int, bool]:
    state, u = uniform(state)
    return state, u < probability


def pick(state: int, options: Sequence[object]) -> tuple[int, int]:
    """(new state, an index into ``options``)."""
    if not options:
        raise ValueError("nothing to pick from")
    state, u = uniform(state)
    return state, min(len(options) - 1, int(u * len(options)))


def pick_weighted(state: int, weights: Mapping[str, float]) -> tuple[int, str]:
    """(new state, a key with probability ∝ its weight). Keys are taken in sorted order."""
    keys = sorted(key for key, weight in weights.items() if weight > 0.0)
    if not keys:
        raise ValueError("no positive weight")
    total = sum(weights[key] for key in keys)
    state, u = uniform(state)
    target = u * total
    running = 0.0
    for key in keys:
        running += weights[key]
        if target < running:
            return state, key
    return state, keys[-1]


def ou_step(x: float, mu: float, sigma: float, tau: float, dt: float, g: float) -> float:
    """One Euler–Maruyama step of an OU process with stationary std ``sigma`` and time constant
    ``tau`` (seconds), driven by the standard normal ``g``."""
    if tau <= 0.0 or dt <= 0.0:
        return x
    a = min(1.0, dt / tau)
    return x + (mu - x) * a + sigma * math.sqrt(2.0 * a) * g


@dataclass(frozen=True, slots=True)
class Pink:
    """Three OU processes at 0.3 s, 2 s and 12 s: correlated drift over several time scales
    (roughly 1/f between ~0.01 and ~1 Hz). ``value`` is their weighted sum (unit-ish std)."""

    fast: float = 0.0
    mid: float = 0.0
    slow: float = 0.0

    @property
    def value(self) -> float:
        return 0.35 * self.fast + 0.5 * self.mid + 0.8 * self.slow

    def step(self, state: int, dt: float) -> tuple[int, Pink]:
        state, g1 = gauss(state)
        state, g2 = gauss(state)
        state, g3 = gauss(state)
        return state, Pink(
            ou_step(self.fast, 0.0, 1.0, 0.3, dt, g1),
            ou_step(self.mid, 0.0, 1.0, 2.0, dt, g2),
            ou_step(self.slow, 0.0, 1.0, 12.0, dt, g3),
        )


def clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def min_jerk(u: float) -> float:
    """Minimum-jerk position profile 0 → 1 for u in [0, 1]."""
    u = clamp(u, 0.0, 1.0)
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


def smoothstep(u: float) -> float:
    u = clamp(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def ease(u: float, kind: str) -> float:
    """``mj`` minimum jerk, ``lin`` linear, ``step`` jump, ``snap`` a fast start (a saccade:
    most of the way in the first third), ``soft`` smoothstep."""
    if kind == "lin":
        return clamp(u, 0.0, 1.0)
    if kind == "step":
        return 1.0 if u > 0.0 else 0.0
    if kind == "snap":
        u = clamp(u, 0.0, 1.0)
        return 1.0 - (1.0 - u) ** 3
    if kind == "soft":
        return smoothstep(u)
    return min_jerk(u)


def leaky(x: float, target: float, tau: float, dt: float) -> float:
    """A first-order leaky integrator toward ``target`` (the neuron's membrane equation)."""
    if tau <= 0.0:
        return target
    return x + (target - x) * min(1.0, dt / tau)


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


AXIS_RANGE = (2000, 8000)  # content.TEMPERAMENT["range"]: Duck Core's fixed-point axis range


def norm_axis(value: float | int | None, default: float = 0.5) -> float:
    """A temperament axis as 0..1, on the SAME scale as ``genome.Genome.axes`` (the one diversity
    mapping, ``genome._unit``): Duck Core's fixed point 2000–8000 (``content.TEMPERAMENT
    ['range']``) maps to 0–1; a value ≤ 1.5 is taken as already on that scale."""
    if value is None:
        return default
    v = float(value)
    if v > 1.5:
        low, high = AXIS_RANGE
        v = (v - low) / (high - low)
    return clamp(v, 0.0, 1.0)


# ── exact-for-any-dt processes and named streams (folded in from the brain's noise.py) ───────
#
# The brain holds mutable state, so it uses these object forms; the motor's frozen-state step
# functions use the functional forms above. Both draw from splitmix64. An OU step here is the
# exact transition for any ``dt`` (the brain's determinism test runs one duck at 5, 10 and
# 20 Hz, and the live tick jitters): Euler's ``min(1, dt/τ)`` would change the statistics with
# the tick. Phase 5: ``exp(-dt/τ)`` and ``√(1 − e^(−2dt/τ))`` become per-tick-rate constants.


def ou_exact(x: float, mu: float, sigma: float, tau: float, dt: float, g: float) -> float:
    """The exact OU transition over ``dt`` (stationary std ``sigma``, time constant ``tau`` s)
    driven by the standard normal ``g``: ``mu + (x − mu)·e^(−dt/τ) + σ·√(1 − e^(−2dt/τ))·g``."""
    if tau <= 0.0 or dt <= 0.0:
        return x
    decay = math.exp(-dt / tau)
    return mu + (x - mu) * decay + sigma * math.sqrt(max(0.0, 1.0 - decay * decay)) * g


def pink_exact(p: Pink, state: int, dt: float) -> tuple[int, Pink]:
    """:meth:`Pink.step` with the exact OU transition (same three time scales)."""
    state, g1 = gauss(state)
    state, g2 = gauss(state)
    state, g3 = gauss(state)
    return state, Pink(
        ou_exact(p.fast, 0.0, 1.0, 0.3, dt, g1),
        ou_exact(p.mid, 0.0, 1.0, 2.0, dt, g2),
        ou_exact(p.slow, 0.0, 1.0, 12.0, dt, g3),
    )


class Stream:
    """A NAMED stream (splitmix64): ``Stream(seed, "select")`` never shifts ``Stream(seed,
    "loco")``, so adding a draw in one part of the brain leaves every other part as it was."""

    __slots__ = ("state",)

    def __init__(self, seed: int, purpose: str) -> None:
        self.state = seed_from("neuro", purpose, int(seed))

    def u64(self) -> int:
        self.state, value = next_u64(self.state)
        return value

    def uniform(self) -> float:
        """[0, 1)."""
        self.state, value = uniform(self.state)
        return value

    def normal(self) -> float:
        """Irwin–Hall(4), unit variance, bounded (|z| < 3.47): :func:`gauss`."""
        self.state, value = gauss(self.state)
        return value


class OU:
    """A mutable Ornstein–Uhlenbeck process (mean ``mu``, stationary std ``sigma``, time
    constant ``tau_s``), exact for any ``dt`` (:func:`ou_exact`). ``mu`` may be moved."""

    __slots__ = ("_s", "mu", "sigma", "tau_s", "x")

    def __init__(self, stream: Stream, mu: float, sigma: float, tau_s: float) -> None:
        self._s = stream
        self.mu = mu
        self.sigma = sigma
        self.tau_s = max(1e-3, tau_s)
        self.x = mu + sigma * stream.normal()

    def step(self, dt_s: float) -> float:
        if dt_s > 0.0:
            self.x = ou_exact(self.x, self.mu, self.sigma, self.tau_s, dt_s, self._s.normal())
        return self.x


class Oscillator:
    """A phase oscillator whose rate wanders (an OU on the rate, around 1): a rhythm generator
    that never ticks like a clock."""

    __slots__ = ("_rate", "hz", "phase")

    def __init__(self, stream: Stream, hz: float, jitter: float, tau_s: float) -> None:
        self.hz = hz
        self.phase = stream.uniform() * 2.0 * math.pi
        self._rate = OU(stream, 1.0, jitter, tau_s)

    def step(self, dt_s: float, speed: float = 1.0) -> float:
        """Advance by ``dt_s`` at ``speed`` × the wandering rate; returns the phase (rad)."""
        rate = max(0.2, self._rate.step(dt_s))
        self.phase = (self.phase + 2.0 * math.pi * self.hz * rate * speed * dt_s) % (2.0 * math.pi)
        return self.phase

    def value(self) -> float:
        return math.sin(self.phase)


__all__ = [
    "OU",
    "Oscillator",
    "Pink",
    "Stream",
    "between",
    "chance",
    "clamp",
    "ease",
    "gauss",
    "leaky",
    "lognormal",
    "min_jerk",
    "mix64",
    "next_u64",
    "norm_axis",
    "ou_exact",
    "ou_step",
    "pick",
    "pick_weighted",
    "pink_exact",
    "seed_from",
    "smoothstep",
    "uniform",
    "wrap",
]
