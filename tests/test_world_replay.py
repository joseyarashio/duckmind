"""Replaying the world (phase-4 design §6 「世界帶」; test T6): LegacyWorld keeps every call that
changed it (Duck Core's commands, the ``due`` calls that delivered something, the polls, the
family's hands and settings) with its millisecond, and every percept it sent. A fresh world built
from the same header and fed that tape must send exactly the same percepts, one by one
(``world_replay_check``) — also after the tape went through JSON, as a saved session keeps it.
A different header or an edited tape must not.

First on a world driven by hand through every call it has (walks that fall, righting tries,
failed stand-ups, docks with and without the marker, nest looks, stops, the TV and gamepad
polls, touches and head holds); then on whole v4_1 ducks through the api once it has landed.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from duckmind import worldport

T0 = 1_790_000_000_000


def _world(**rates: Any) -> Any:
    from duckmind.world import legacy

    header = legacy.make_header(seed=77, table_hash="d56679edadd377a9", **rates)
    return worldport.make_world(header)


def _walk(act_id: str, seconds: int, x_mm: int = 0) -> dict[str, Any]:
    return {
        "cmd": "duck.walk",
        "act_id": act_id,
        "deadline_ms": 0,
        "verb": "wander",
        "vx": 0.3,
        "vyaw": 0.0,
        "seconds": seconds,
        "cm": 30 * seconds,
        "vx_mm": 300,
        "vyaw_mrad": 0,
        "max_ms": seconds * 1000,
        "purpose": "wander",
        "bel_from": {"x_mm": x_mm, "y_mm": 0, "yaw_mrad": 0},
        "expect": {"x_mm": x_mm + 300 * seconds, "y_mm": 0, "yaw_mrad": 0},
    }


def _drive(world: Any, steps: int = 400) -> list[dict[str, Any]]:
    """Every kind of call, many times over, at rising milliseconds; what came back."""
    got: list[dict[str, Any]] = []
    t = T0
    n = 0
    for step in range(steps):
        t += 1000 * (1 + step % 7)
        got += world.due(t)
        n += 1
        choice = step % 11
        if choice in (0, 1, 2):
            got += world.command([_walk(f"A{n}", 2 + step % 9, x_mm=step)], t)
        elif choice == 3:
            got += world.command([{"cmd": "duck.posture", "act_id": f"A{n}", "want": "stand"}], t)
        elif choice == 4:
            got += world.command([{"cmd": "duck.dock", "act_id": f"A{n}", "bel_err_mm": step}], t)
        elif choice == 5:
            look = {
                "cmd": "duck.look",
                "act_id": f"A{n}",
                "kind": "seek_nest",
                "want": "nest",
                "bel_pose": {"x_mm": 300 - step, "y_mm": step % 50, "yaw_mrad": step * 10},
            }
            got += world.command([look], t)
        elif choice == 6:
            got += world.command([{"cmd": "duck.stop", "why": "picked_up"}], t)
        elif choice == 7:
            got += world.poll(worldport.POLL_TV_MINUTE, t)
            got += world.poll(worldport.POLL_PLAY_MINUTE, t, minute=step)
        elif choice == 8:
            event = world.touch("child", seconds=6, where="crown", firmness="light", now_ms=t)
            got += [event] if event is not None else []
            event = world.hold_head("owner", moving=step % 2 == 0, now_ms=t)
            got += [event] if event is not None else []
        elif choice == 9:
            world.set_setting("conservative", step % 2 == 0, now_ms=t)
            world.tv(step % 4 == 1, now_ms=t)
        else:
            world.hand("pickup", persona="owner", now_ms=t)
            got += world.command([{"cmd": "duck.stop", "why": "set_upright"}], t)
    got += world.due(t + 60_000)
    return got


STRESS = {"fall_ppm_per_s": 200_000, "righting_ppm": 300_000, "stand_ppm": 600_000}


@pytest.mark.parametrize("marker", [True, False])
def test_a_fresh_world_fed_the_tape_sends_the_same_percepts(marker: bool) -> None:
    from duckmind.world import legacy

    world = _world(marker=marker, **STRESS)
    got = _drive(world)
    kinds = {p["kind"] for p in got}
    assert {"walk.done", "body.fell", "body.righting", "body.posture", "body.dock"} <= kinds
    # with the marker every nest look sees it; without, every look is an empty scan
    assert ("sense.sighting" in kinds, "sense.scan" in kinds) == (marker, not marker)
    assert {"body.play", "sense.hold_head"} <= kinds
    assert got == world.sent, "what the world sent and what it kept differ"
    for event in got:
        worldport.check(event)
        assert event["src"] == "legacy" and event["pv"] == "l:" + worldport.calib_hash(
            world.header()
        )
    tape = json.loads(json.dumps(world.tape_json(), ensure_ascii=False))
    result = legacy.world_replay_check(world.header(), tape, world.sent)
    assert result == {"identical": True, "percepts": len(got), "first_difference": None}


def test_another_header_or_an_edited_tape_sends_something_else() -> None:
    from duckmind.world import legacy

    world = _world(**STRESS)
    _drive(world, steps=200)
    tape, sent = world.tape_json(), world.sent
    other = {**world.header(), "seed": 78}
    assert not legacy.world_replay_check(other, tape, sent)["identical"]
    edited = json.loads(json.dumps(tape))
    walk = next(r for r in edited if r["call"] == "command" and r["args"]["cmd"] == "duck.walk")
    walk["args"]["seconds"] += 5
    result = legacy.world_replay_check(world.header(), edited, sent)
    assert not result["identical"] and result["first_difference"] is not None
    shorter = legacy.world_replay_check(world.header(), tape[:-3], sent)
    assert not shorter["identical"]


def test_a_world_that_keeps_no_tape_behaves_the_same() -> None:
    from duckmind.world import legacy

    kept = _world(**STRESS)
    bare = legacy.LegacyWorld(kept.header(), keep_tape=False)
    assert _drive(kept, steps=150) == _drive(bare, steps=150)
    assert bare.tape == [] and bare.sent == []
    assert kept.counters == bare.counters


# ---------------------------------------------------------------------------- whole ducks


@pytest.mark.parametrize(
    "seed,nest,stress", [(51, True, True), (52, False, True), (53, True, False)]
)
def test_a_v41_ducks_world_replays_from_its_tape(seed: int, nest: bool, stress: bool) -> None:
    from phase4_harness import lively_v41, world_of

    from duckmind.duckcore import api

    session = lively_v41(seed, nest, stress, days=1)
    world = world_of(session)
    assert world.sent, "the world sent nothing"
    result = api.world_replay_check(session)
    assert result["identical"], result
