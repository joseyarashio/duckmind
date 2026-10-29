"""Curiosity's sockets are inert while its home has no things (phase-4 design §9; curiosity spec
§14 「沒有東西時事件鏈不變」, phase-4 version; test T8).

Phase 4 fits every hook curiosity P2 will use, so the engine changes only once, but the content
waits for P2 (it changes the content table hash, and with it every draw). Until then:

* no socket draws (no ``rng.draw`` while a socket runs), writes (no ``_emit``) or changes
  Duck Core's state (its pickled state is the same before and after every socket call);
* with the sockets pulled out (each replaced by its inert answer, never looking at the duck)
  the same duck writes the same chain, to the byte;
* every socket is actually reached by a lively life, so the check is not vacuous.
"""

from __future__ import annotations

import pickle
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from duckmind.duckcore import curiosity

# socket -> its inert answer (what P2 must still answer with no things at home)
INERT: dict[str, Any] = {
    "minute": False,
    "next_due": None,
    "holds_body": False,
    "at_pause": False,
    "attractions": curiosity.NO_PULL,
    "attraction": 0,
    "wants_toy": False,
    "zone_block": None,
    "on_fall": None,
    "on_sense": None,
    "on_result": None,
    "on_guard": None,
    "on_timeout": None,
    "on_home": None,
}
# sockets a lively v4_1 life with falls must reach (engine, excursion and percept hooks;
# ``zone_block`` is not even called while the family has marked no zone, ``on_result`` only
# for a curiosity step, ``on_home`` only for the App's thing inputs)
REACHED = (
    "minute",
    "holds_body",
    "at_pause",
    "attractions",
    "wants_toy",
    "on_fall",
    "next_due",
    "on_sense",
)


class _Untouchable:
    """A duck whose every draw, write or state read beyond ``s`` fails the test."""

    def __init__(self) -> None:
        self.s = SimpleNamespace(things={}, curio=None)

    def _rand(self, purpose: str) -> int:
        raise AssertionError(f"a socket drew {purpose}")

    def _emit(self, kind: str, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"a socket wrote {kind}")


def test_every_socket_answers_inertly_with_nothing_at_home() -> None:
    core: Any = _Untouchable()
    record: Any = SimpleNamespace(event_id="E000001", kind="sense.thing")
    calls: dict[str, Callable[[], Any]] = {
        "minute": lambda: curiosity.minute(core),
        "next_due": lambda: curiosity.next_due(core),
        "holds_body": lambda: curiosity.holds_body(core),
        "at_pause": lambda: curiosity.at_pause(core),
        "attractions": lambda: curiosity.attractions(core),
        "attraction": lambda: curiosity.attraction(core, 5),
        "wants_toy": lambda: curiosity.wants_toy(core),
        "zone_block": lambda: curiosity.zone_block(core, (100, -200)),
        "on_fall": lambda: curiosity.on_fall(core, {"x": 0, "y": 0}),
        "on_sense": lambda: curiosity.on_sense(core, {"things": []}, record),
        "on_result": lambda: curiosity.on_result(core, {"id": "C1", "step": 1}, record),
        "on_guard": lambda: curiosity.on_guard(core, {"kind": "cliff"}, record),
        "on_timeout": lambda: curiosity.on_timeout(core, "C1"),
        "on_home": lambda: curiosity.on_home(core, "home.thing", {}, record),
    }
    assert set(calls) == set(INERT)
    before = vars(core.s).copy()
    for name, call in calls.items():
        assert call() == INERT[name], name
    assert vars(core.s) == before
    assert not curiosity.has_things(core) and not curiosity.active(core)


@contextmanager
def _spied(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    """Count socket calls; fail any draw, write or state change made inside one."""
    from duckmind.duckcore import engine, rng

    seen: dict[str, Any] = {"calls": {}, "inside": 0, "problems": []}
    draw, emit = rng.draw, engine.DuckCore._emit

    def spy(name: str, socket: Callable[..., Any]) -> Callable[..., Any]:
        def call(*args: Any, **kwargs: Any) -> Any:
            core = args[0] if args else kwargs.get("core")
            state = pickle.dumps(core.s) if hasattr(core, "s") else None
            seen["calls"][name] = seen["calls"].get(name, 0) + 1
            seen["inside"] += 1
            try:
                result = socket(*args, **kwargs)
            finally:
                seen["inside"] -= 1
            if state is not None and pickle.dumps(core.s) != state:
                seen["problems"].append(f"{name} changed the duck's state")
            if result != INERT[name]:
                seen["problems"].append(f"{name} answered {result!r}")
            return result

        return call

    def watched_draw(*args: Any) -> int:
        if seen["inside"]:
            seen["problems"].append(f"a socket drew {args[2] if len(args) > 2 else args}")
        return int(draw(*args))

    def watched_emit(core: Any, kind: str, *args: Any, **kwargs: Any) -> Any:
        if seen["inside"]:
            seen["problems"].append(f"a socket wrote {kind}")
        return emit(core, kind, *args, **kwargs)

    for name in INERT:
        monkeypatch.setattr(curiosity, name, spy(name, getattr(curiosity, name)))
    monkeypatch.setattr(rng, "draw", watched_draw)
    monkeypatch.setattr(engine.DuckCore, "_emit", watched_emit)
    yield seen


def _pulled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The sockets taken out: every one answers inertly without looking at the duck."""
    for name, answer in INERT.items():
        monkeypatch.setattr(curiosity, name, lambda *args, _answer=answer, **kwargs: _answer)


def _life() -> Any:
    """A lively nest duck with falls, then a ``sense.thing`` (what only curiosity reads) through
    ``api.sense``. (``body.guard`` cannot be sent yet: percept-v1 names the guard's kind in a
    field called ``kind``, which the event's own ``kind`` stamp hides, so no guard validates.)"""
    from phase4_harness import lively, lively_config, open_v41, send, walking_age

    from duckmind import worldport

    session = open_v41(lively_config(7, nest=True, stress=True))
    walking_age(session)
    session.core.next_due()  # (awake at 07:00: it asks curiosity when it next has to look)
    lively(session, 7, days=1)
    thing = {
        "kind": "sense.thing",
        "src": "robot",
        "t_ms": session.core.now_ms() + 1000,
        "pv": "p:00112233aabbccdd",
        "things": [],
    }
    worldport.check(thing)
    send(session, "sense", {"events": [thing]})
    send(session, "advance", {"seconds": 2})
    session.core.next_due()
    return session


def test_the_sockets_draw_nothing_write_nothing_and_change_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _spied(monkeypatch) as seen:
        _life()
    assert not seen["problems"], "\n".join(seen["problems"][:20])
    never = [name for name in REACHED if not seen["calls"].get(name)]
    assert not never, f"sockets the life never reached: {never} (calls: {seen['calls']})"


def test_with_the_sockets_pulled_out_the_chain_is_the_same(monkeypatch: pytest.MonkeyPatch) -> None:
    fitted = _life()
    with monkeypatch.context() as patch:
        _pulled(patch)
        pulled = _life()
    assert pulled.core.log.head == fitted.core.log.head
    assert len(pulled.core.log.events) == len(fitted.core.log.events)
