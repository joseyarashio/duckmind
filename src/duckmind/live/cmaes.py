"""A small CMA-ES (covariance matrix adaptation evolution strategy), for fitting the new drives (LV4).

The standard (μ/μ_w, λ) strategy with rank-one and rank-μ covariance updates and cumulative step-size
adaptation, as in Hansen's tutorial ("The CMA Evolution Strategy: A Tutorial", 2016), minimising. It
works in whatever space the caller likes; the search of LV4 keeps its points in the unit cube (one
coordinate per fitted parameter) and clips them there when it evaluates them, so the strategy itself is
unconstrained and the caller adds a small penalty for how far a point lies outside the cube.

Deterministic for a seed (a ``numpy.random.Generator``); the whole state pickles, so a long search can
stop and resume. Nothing here knows about ducks.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class CMAState:
    """What is needed to carry on a search (all plain arrays and numbers)."""

    mean: np.ndarray
    sigma: float
    C: np.ndarray
    ps: np.ndarray
    pc: np.ndarray
    generation: int
    best_x: np.ndarray | None
    best_f: float
    rng_state: Mapping[str, Any]


class CMAES:
    def __init__(
        self,
        x0: np.ndarray | list[float],
        sigma0: float,
        *,
        popsize: int | None = None,
        seed: int = 0,
    ) -> None:
        self.mean = np.asarray(x0, dtype=float).copy()
        n = self.n = int(self.mean.size)
        self.sigma = float(sigma0)
        self.lam = int(popsize) if popsize else 4 + int(3 * math.log(n))
        self.mu = self.lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = w / w.sum()
        self.mueff = float(1.0 / np.sum(self.weights**2))
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(
            1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff)
        )
        self.damps = 1 + 2 * max(0.0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chi_n = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.C = np.eye(n)
        self.ps = np.zeros(n)
        self.pc = np.zeros(n)
        self.generation = 0
        self.best_x: np.ndarray | None = None
        self.best_f = math.inf
        self.rng = np.random.default_rng(seed)
        self._B = np.eye(n)
        self._D = np.ones(n)
        self._invsqrt = np.eye(n)
        self._last: tuple[np.ndarray, np.ndarray] | None = None

    # ── one generation ────────────────────────────────────────────────────────

    def ask(self) -> np.ndarray:
        """``lam`` candidate points (rows)."""
        z = self.rng.standard_normal((self.lam, self.n))
        y = z @ (self._B * self._D).T
        x = self.mean + self.sigma * y
        self._last = (x, y)
        return x

    def tell(self, fitness: list[float] | np.ndarray) -> None:
        """The values of the points ``ask`` gave (lower is better)."""
        if self._last is None:
            raise RuntimeError("tell before ask")
        x, y = self._last
        f = np.asarray(fitness, dtype=float)
        if f.shape != (self.lam,):
            raise ValueError(f"expected {self.lam} values, got {f.shape}")
        order = np.argsort(f, kind="stable")
        if f[order[0]] < self.best_f:
            self.best_f = float(f[order[0]])
            self.best_x = x[order[0]].copy()
        top = order[: self.mu]
        y_w = self.weights @ y[top]
        self.mean = self.mean + self.sigma * y_w
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * (
            self._invsqrt @ y_w
        )
        gen = self.generation + 1
        h_sigma = float(
            np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * gen)) / self.chi_n
            < 1.4 + 2 / (self.n + 1)
        )
        self.pc = (1 - self.cc) * self.pc + h_sigma * math.sqrt(
            self.cc * (2 - self.cc) * self.mueff
        ) * y_w
        rank_mu = (y[top].T * self.weights) @ y[top]
        self.C = (
            (1 - self.c1 - self.cmu + (1 - h_sigma) * self.c1 * self.cc * (2 - self.cc)) * self.C
            + self.c1 * np.outer(self.pc, self.pc)
            + self.cmu * rank_mu
        )
        self.sigma *= math.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chi_n - 1))
        self.generation = gen
        self._decompose()

    def _decompose(self) -> None:
        self.C = (self.C + self.C.T) / 2
        d2, b = np.linalg.eigh(self.C)
        d2 = np.maximum(d2, 1e-20)
        self._D = np.sqrt(d2)
        self._B = b
        self._invsqrt = (b / self._D) @ b.T

    # ── stop and resume ───────────────────────────────────────────────────────

    def state(self) -> CMAState:
        return CMAState(
            self.mean.copy(),
            self.sigma,
            self.C.copy(),
            self.ps.copy(),
            self.pc.copy(),
            self.generation,
            None if self.best_x is None else self.best_x.copy(),
            self.best_f,
            self.rng.bit_generator.state,
        )

    def restore(self, state: CMAState) -> None:
        self.mean, self.sigma = state.mean.copy(), state.sigma
        self.C, self.ps, self.pc = state.C.copy(), state.ps.copy(), state.pc.copy()
        self.generation = state.generation
        self.best_x = None if state.best_x is None else state.best_x.copy()
        self.best_f = state.best_f
        self.rng.bit_generator.state = state.rng_state
        self._decompose()


__all__ = ["CMAES", "CMAState"]
