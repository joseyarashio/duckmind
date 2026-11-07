"""A grown duck's heap is out of the garbage collector's full passes (docs/design/neural-duck/MAP-STALL-2026-11-07.md).

蛋黃's Duck Core replays ~350,000 objects into memory; every full collection walked them all (~110 ms every
~45 s, on the brain's tick). ``settle_heap`` sets what is alive aside, frees what an earlier settle set aside
once it is dropped, and changes nothing the duck decides."""

from __future__ import annotations

import gc
import json

import pytest

from duckmind.duckcore.api import save_json
from duckmind.neuro.brain import NeuroBrain
from duckmind.neuro.genome import preset
from duckmind.neuro.synth import SynthWorld, iterate
from duckmind.simweb.server import Config, SimState, dispatch, settle_heap

# What a full pass may still walk after a settle (the objects made since; the grown duck below adds ~80,000)
SCAN_BUDGET = 5_000


@pytest.fixture(autouse=True)
def _unfrozen():
    yield
    gc.unfreeze()  # (the rest of the test process is collected as usual)


def _scanned() -> int:
    """How many objects a full collection walks now (every generation; the set-aside ones are not)."""
    return sum(len(gc.get_objects(generation)) for generation in range(3))


def _grown_duck() -> SimState:
    state = SimState(Config(seed=4107, name="測試", tof=True, nest=False))
    dispatch(state.session, "advance", {"seconds": 15 * 86400})
    return state


def test_a_grown_ducks_life_is_not_walked_by_every_full_collection() -> None:
    gc.collect()
    before = _scanned()
    state = _grown_duck()
    grown = _scanned() - before
    assert len(state.session.core.log.events) > 20_000
    assert grown > 50_000  # (the duck's life is that many objects: each full pass walked them)
    settle_heap()
    assert _scanned() < SCAN_BUDGET
    assert gc.get_freeze_count() > grown


def test_a_dropped_duck_is_freed_at_the_next_settle() -> None:
    state = _grown_duck()
    settle_heap()
    held = gc.get_freeze_count()
    state.reset({"seed": 8, "name": "新"})  # (the grown Duck Core is dropped; it is cyclic garbage)
    settle_heap()
    assert gc.get_freeze_count() < held - 50_000


def test_settling_changes_nothing_the_duck_decides() -> None:
    def trace(settle_every: int | None) -> list[tuple]:
        world = SynthWorld(seed=5)
        world.clap(20_000)
        world.poke(45_000)
        out = []
        for k, (c, t) in enumerate(
            iterate(NeuroBrain(preset("explorer"), seed=11), world, 90.0, hz=20)
        ):
            if settle_every and k % settle_every == 0:
                settle_heap()
            out.append(
                (c.behaviour, c.bout, c.vx, c.vyaw, c.gaze.bearing, c.vocal, c.reflex, t.reason_zh)
            )
        return out

    assert trace(None) == trace(97)
    core_plain, core_settled = _grown_duck(), _grown_duck()
    settle_heap()
    dispatch(core_plain.session, "advance", {"seconds": 3600})
    dispatch(core_settled.session, "advance", {"seconds": 3600})
    assert json.dumps(save_json(core_plain.session), sort_keys=True) == json.dumps(
        save_json(core_settled.session), sort_keys=True
    )
