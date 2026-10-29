"""Replaying a duck from what reached it (phase-4 design A8; tests T5 and T7).

``percept_replay_check``: a v4_1 duck's tape (every outside call on Duck Core with its
arguments at its clock, and every percept its port delivered, with how many commands and polls
Duck Core had made by then) is fed to a fresh Duck Core with **no world and no family** — only a
``TapePort`` — and must end on the same chain head. It proves Duck Core knows the world only
through its inputs (spec R1/R2): anything it read from elsewhere would not be on the tape.

``replay_check`` (the page's replay button: same Config, same recorded api inputs, world and
family rebuilt) must hold for v4_1 ducks too.

Twenty sessions in CI: busy scripted days (nest / no nest / stress), household-v2 families,
legacy routines and page-like sessions (settings, the gamepad, the command box). A tampered tape
must fail, so the check can.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any

import pytest
from phase4_harness import (
    act,
    household_legacy,
    household_v2,
    live_legacy,
    live_v2,
    lively,
    lively_config,
    open_v41,
    send,
    walking_age,
)

KINDS = (("nest", True, True), ("floor", False, True), ("calm", True, False))


def _lively(seed: int, index: int) -> Callable[[], Any]:
    name, nest, stress = KINDS[index % 3]

    def make() -> Any:
        session = open_v41(lively_config(seed, nest, stress))
        walking_age(session)
        lively(session, seed, days=1)
        return session

    make.__name__ = f"lively-{seed}-{name}"
    return make


def _v2(index: int) -> Callable[[], Any]:
    def make() -> Any:
        config, kind = household_v2(index)
        session = open_v41(config)
        live_v2(session, kind, 2)
        return session

    make.__name__ = f"v2-{index}"
    return make


def _legacy(index: int) -> Callable[[], Any]:
    def make() -> Any:
        config, first, second = household_legacy(index)
        session = open_v41(config)
        live_legacy(session, first, second, 3)
        return session

    make.__name__ = f"legacy-{index}"
    return make


def _page(seed: int) -> Callable[[], Any]:
    """What a page user does: settings, a head hold while walking, the gamepad, the TV for an
    hour, the command box, a party, lifting it back onto the nest."""

    def make() -> Any:
        session = open_v41(lively_config(seed, True, seed % 2 == 0))
        walking_age(session)
        send(session, "set_setting", {"key": "conservative", "value": False})
        act(session, "tv", None, on=True)
        send(session, "advance", {"minutes": 60})
        act(session, "tv", None, on=False)
        act(session, "pad_play", "child", minutes=6)
        send(session, "advance", {"minutes": 30})
        act(session, "hold_head", "owner")
        act(session, "touch", "owner", where="crown", seconds=8, firmness="scratch")
        send(session, "command", {"text": "摸摸頭", "persona": "owner"})
        send(session, "party", {"guest_mode": False})
        act(session, "pickup", "owner")
        act(session, "place_on_nest", "owner")
        lively(session, seed, days=1)
        return session

    make.__name__ = f"page-{seed}"
    return make


SESSIONS = [
    *(_lively(41 + n, n) for n in range(6)),
    *(_v2(n) for n in range(5)),
    *(_legacy(n) for n in range(4)),
    *(_page(60 + n) for n in range(5)),
]


@pytest.mark.parametrize("make", SESSIONS, ids=[make.__name__ for make in SESSIONS])
def test_the_tape_alone_replays_the_same_duck(make: Callable[[], Any]) -> None:
    from duckmind.duckcore import api

    session = make()
    tape = session.core.tape
    assert tape is not None, "a v4_1 session records its tape"
    json.dumps(tape.to_dict(), ensure_ascii=False, allow_nan=False)  # the tape is plain JSON
    assert tape.entries, "nothing was recorded"
    if make.__name__.startswith(("lively", "page")):  # (young family ducks may not walk yet)
        assert any(entry["op"] == "percept" for entry in tape.entries), "no percept came in"
    result = api.percept_replay_check(session)
    assert result["identical"], f"{make.__name__}: {result}"
    again = api.replay_check(session)  # T7: the page's replay, world and family rebuilt
    assert again["identical"], f"{make.__name__}: {again}"


def test_a_tampered_tape_is_caught() -> None:
    from duckmind.duckcore import api

    session = SESSIONS[0]()
    tape = session.core.tape
    saved = copy.deepcopy(tape.entries)
    victim = next(
        entry
        for entry in tape.entries
        if entry["op"] == "percept" and entry["p"]["kind"] in ("walk.done", "body.posture")
    )
    percept = victim["p"]
    if percept["kind"] == "walk.done":
        percept["reason"] = "fell" if percept["reason"] != "fell" else "time"
    else:
        percept["got"] = "sit" if percept["got"] == "stand" else "stand"
    try:
        assert not api.percept_replay_check(session)["identical"], "a changed percept went unseen"
    finally:
        tape.entries[:] = saved
    assert api.percept_replay_check(session)["identical"]
