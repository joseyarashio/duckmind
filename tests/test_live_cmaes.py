"""Live voice, LV4: the CMA-ES the drives are fitted with (``duckmind.live.cmaes``): it minimises, it is
deterministic for a seed, it can stop and resume, and it asks for what it is told."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from duckmind.live.cmaes import CMAES

pytestmark = pytest.mark.usefixtures("no_network")


def sphere(x: np.ndarray) -> float:
    return float(np.sum((x - 0.3) ** 2))


def rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def run(f, x0, sigma: float, gens: int, seed: int = 1, pop: int | None = None) -> CMAES:
    es = CMAES(x0, sigma, popsize=pop, seed=seed)
    for _ in range(gens):
        xs = es.ask()
        es.tell([f(x) for x in xs])
    return es


def test_it_minimises_a_sphere_in_twenty_dimensions() -> None:
    es = run(sphere, np.full(20, 0.8), 0.3, 250, pop=16)
    assert es.best_f < 1e-8
    assert np.allclose(es.best_x, 0.3, atol=1e-3)


def test_it_minimises_a_curved_valley() -> None:
    es = run(rosenbrock, np.zeros(5), 0.5, 400, pop=12)
    assert es.best_f < 1e-3


def test_the_same_seed_makes_the_same_search_and_another_seed_another() -> None:
    a = run(sphere, np.zeros(6), 0.3, 10, seed=4)
    b = run(sphere, np.zeros(6), 0.3, 10, seed=4)
    c = run(sphere, np.zeros(6), 0.3, 10, seed=5)
    assert np.array_equal(a.mean, b.mean) and a.best_f == b.best_f
    assert not np.array_equal(a.mean, c.mean)


def test_a_search_that_stops_and_resumes_goes_on_exactly_as_if_it_had_not_stopped() -> None:
    straight = run(sphere, np.zeros(6), 0.3, 12, seed=2)
    first = run(sphere, np.zeros(6), 0.3, 5, seed=2)
    state = pickle.loads(pickle.dumps(first.state()))
    second = CMAES(np.zeros(6), 0.3, seed=2)
    second.restore(state)
    for _ in range(7):
        xs = second.ask()
        second.tell([sphere(x) for x in xs])
    assert second.generation == 12
    assert np.allclose(second.mean, straight.mean, atol=1e-12) and second.sigma == pytest.approx(
        straight.sigma
    )
    assert second.best_f == straight.best_f


def test_it_asks_for_a_population_and_wants_one_value_for_each() -> None:
    es = CMAES(np.zeros(4), 0.2, popsize=10)
    xs = es.ask()
    assert xs.shape == (10, 4)
    with pytest.raises(ValueError):
        es.tell([0.0] * 9)
    with pytest.raises(RuntimeError):
        CMAES(np.zeros(4), 0.2).tell([0.0] * 8)


def test_it_keeps_the_best_it_has_seen_even_when_a_generation_is_worse() -> None:
    es = CMAES(np.zeros(3), 0.5, popsize=8, seed=3)
    es.ask()
    es.tell(list(range(8)))
    best = es.best_f
    es.ask()
    es.tell([10.0 + k for k in range(8)])
    assert es.best_f == best == 0.0


def test_the_step_size_falls_as_it_closes_in() -> None:
    es = run(sphere, np.full(8, 0.9), 0.4, 80, pop=12)
    assert es.sigma < 0.05
