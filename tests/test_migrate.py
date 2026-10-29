"""A duck moves to the newer engine with its memories (phase-4 WP13; owner decision P4-1 = a).

The seam (``duckmind.engines.migrate``, ``api.migrate``) takes a v4 duck on the frozen engine at
any moment between two inputs — sitting quietly, in the middle of a walk (also one whose fall v4
had already drawn), lying after a fall while its body still tries to get up — and carries its
state, its event chain, its draw counters and its family over to engine v4_1 in LegacyWorld.
From then on it must behave exactly as it would have on v4: the same behaviour projection
(decisions, expressions, walks, postures, positions, day stats, … and the draw counters, the
duck's and its world's together), seam by seam. The chain goes on from the same head; the move is
an input, so the duck restores and replays through it. The owner's own 蛋黃 is proven the same way
on a read-only copy of its save outside the repo (``docs/validation/evidence/phase4``).
"""

from __future__ import annotations

import copy
import dataclasses
import gzip
import json
import pickle
from pathlib import Path
from typing import Any

import phase4_harness as H
import pytest

from duckmind.duckcore import api
from duckmind.duckcore.engine import CoreState
from duckmind.engines import migrate as M
from duckmind.engines.v4_0.duckcore import api as v4
from duckmind.engines.v4_0.duckcore import engine as v4_engine

GOLDEN = Path(__file__).with_name("golden")
STRESS = {"seed": 77, "fall_ppm_per_s": 60_000, "righting_ppm": 250_000, "stand_ppm": 700_000}
# After the seam: time in small and large steps, hands, the TV, the gamepad, a lift, the
# owner's buttons and two days of the old visit routine.
LATER: tuple[tuple[str, dict[str, Any]], ...] = (
    *(("advance", {"seconds": 5}),) * 12,
    ("act", {"action": "touch", "params": {"seconds": 4}, "persona": "child"}),
    ("advance", {"minutes": 30}),
    ("act", {"action": "hold_head", "params": {}, "persona": "owner"}),
    ("tv", {"on": True}),
    ("act", {"action": "tv", "params": {"on": True}, "persona": None}),
    ("advance", {"minutes": 45}),
    ("act", {"action": "pad_play", "params": {"minutes": 3}, "persona": "child"}),
    ("act", {"action": "pickup", "params": {}, "persona": "owner"}),
    ("set_setting", {"key": "floor", "value": True}),
    ("set_setting", {"key": "conservative", "value": False}),
    ("advance", {"minutes": 240}),
    ("live", {"routine": "daily", "days": 2}),
)


class _Rows:
    """The state watch of a duck that moved: its old self's rows, then its new self's."""

    def __init__(self, *watches: H.Watch) -> None:
        self.rows = [row for watch in watches for row in watch.rows]


def _send(session: Any, method: str, params: dict[str, Any]) -> None:
    if method == "tv":  # (a page-less input: the family's TV)
        method, params = "act", {"action": "tv", "params": params, "persona": None}
    H.send(session, method, copy.deepcopy(params))


def _to_seam(kind: str) -> Any:
    """A clumsy v4 duck on the floor, lived on 7 s at a time until it is at a ``kind`` seam."""
    session = v4.Session(v4.Config(**STRESS))
    for method, params in (
        ("live", {"routine": "floor", "days": 5}),
        ("advance", {"until": "07:00"}),
        ("act", {"action": "power_on"}),
        ("set_setting", {"key": "floor", "value": True}),
    ):
        v4.dispatch(session, method, params)
    for _ in range(5000):
        if M.seam(session) == kind:
            return session
        v4.dispatch(session, "advance", {"seconds": 7})
    raise AssertionError(f"no {kind} seam in sight")


def _prefix(config: dict[str, Any], inputs: list[dict[str, Any]], k: int) -> tuple[Any, H.Watch]:
    session = v4.Session(v4.Config(**config))
    watch = H.Watch(session.core)
    for item in inputs[:k]:
        v4.dispatch(session, item["method"], copy.deepcopy(item["params"]))
    return session, watch


def _stay_and_move(
    config: dict[str, Any], inputs: list[dict[str, Any]], k: int
) -> tuple[H.Projection, H.Projection, Any, Any, int]:
    """The same duck twice: staying on v4 for every input, and moving to v4_1 after input
    ``k`` (both watched from their hatching); and how many events it had at the seam."""
    stay, stay_watch = _prefix(config, inputs, len(inputs))
    old, old_watch = _prefix(config, inputs, k)
    at_seam = len(old.core.log.events)
    moved = api.migrate(old, api.ENGINE_REV)
    new_watch = H.Watch(moved.core)
    for item in inputs[k:]:
        api.dispatch(moved, item["method"], copy.deepcopy(item["params"]))
    return (
        H.projection(stay, stay_watch),
        H.projection(moved, _Rows(old_watch, new_watch)),  # type: ignore[arg-type]
        stay,
        moved,
        at_seam,
    )


def _golden(name: str) -> dict[str, Any]:
    with gzip.open(GOLDEN / f"{name}.json.gz", "rb") as handle:
        data: dict[str, Any] = json.loads(handle.read())
    return data


def _seams(config: dict[str, Any], inputs: list[dict[str, Any]]) -> dict[int, str]:
    """Where to move: a quarter, half and three quarters of the way, and the first input after
    which the duck is walking, walking into a fall it has already drawn, or getting up."""
    n = len(inputs)
    seams = {max(1, n * q // 4): "" for q in (1, 2, 3)}
    session = v4.Session(v4.Config(**config))
    wanted = {"walk", "walk+fall", "righting"}
    for index, item in enumerate(inputs[:-1], 1):
        v4.dispatch(session, item["method"], copy.deepcopy(item["params"]))
        kind = M.seam(session)
        if kind in wanted:
            seams[index] = kind
            wanted.discard(kind)
    return {k: seams[k] for k in sorted(seams) if 0 < k < n}


# ---------------------------------------------------------------------------- behaviour


@pytest.mark.parametrize("kind", ["quiet", "walk", "walk+fall", "righting"])
def test_a_v4_duck_moves_at_any_moment_and_goes_on_exactly_as_on_v4(kind: str) -> None:
    stay, moved_old = _to_seam(kind), _to_seam(kind)
    assert stay.core.log.head == moved_old.core.log.head
    stay_watch = H.Watch(stay.core)
    moved = api.migrate(moved_old)
    moved_watch = H.Watch(moved.core)
    for method, params in LATER:
        _send(stay, method, params)
        _send(moved, method, params)
    a, b = H.projection(stay, stay_watch), H.projection(moved, moved_watch)
    assert H.first_difference(a, b, ("stayed on v4_0", "moved to v4_1")) == ""
    assert a.rows and a.kinds.get("duck.walk", 0) > 10
    # the fall v4 drew before the seam happened in the new world, at the same second
    if kind == "walk+fall":
        assert b.kinds.get("duck.walk") == a.kinds.get("duck.walk")
        assert any(
            e.kind == "body.fell" and e.source.kind == "legacy-world" for e in moved.core.log.events
        )
    if kind == "righting":
        assert any(
            e.kind == "body.righting" and e.source.kind == "legacy-world"
            for e in moved.core.log.events
        )


@pytest.mark.parametrize(
    "name",
    [
        "g04_web_young_s2026",
        "g07_web_clumsy_dual7_s5",
        "g20_legacy_party_s301",
        "g41_cfg_clumsy_hard_s403",
    ],
)
def test_recorded_page_ducks_move_at_their_seams_and_behave_the_same(name: str) -> None:
    saved = _golden(name)
    config, inputs = dict(saved["config"]), list(saved["inputs"])
    seams = _seams(config, inputs)
    assert len(seams) >= 3
    for k, kind in seams.items():
        a, b, stay, moved, at_seam = _stay_and_move(config, inputs, k)
        where = H.first_difference(a, b, ("stayed on v4_0", f"moved after input {k} {kind}"))
        assert where == "", f"{name} seam {k} ({kind}): {where}"
        assert stay.core.log.head == saved["head"]  # (the stay duck is the recorded one)
        # one chain: the moved duck's goes on from the old one's head at the seam
        assert moved.core.log.digests[:at_seam] == stay.core.log.digests[:at_seam]


# ---------------------------------------------------------------------------- what moves where


def test_state_chain_and_counters_move_and_the_old_duck_is_left_as_it_was() -> None:
    old = _to_seam("walk+fall")
    before = pickle.dumps((old.core.s, old.core.counters, old.core.log.digests, old.inputs))
    head, events = old.core.log.head, len(old.core.log.events)
    moved = api.migrate(old)
    assert pickle.dumps((old.core.s, old.core.counters, old.core.log.digests, old.inputs)) == before
    core, world = moved.core, moved.port
    # the chain: the same events and digests, verified in the new engine's classes
    assert core.log.head == head and len(core.log.events) == events and core.log.verify()
    assert [e.event_id for e in core.log.events] == [e.event_id for e in old.core.log.events]
    # the state: every v4 field as it was (the walk plan less the fall v4 drew; the new
    # engine's fields empty)
    for name in v4_engine.CoreState.__dataclass_fields__:
        mine, theirs = getattr(core.s, name), getattr(old.core.s, name)
        if name == "walk_plan":
            assert mine == {**{k: v for k, v in theirs.items() if k != "fall_at"}, "act_id": "A1"}
        else:
            assert mine == theirs, name
    new_fields = set(CoreState.__dataclass_fields__) - set(v4_engine.CoreState.__dataclass_fields__)
    assert new_fields == {
        "things",
        "curio",
        "curio_seq",
        "curio_today",
        "zones",
        "asks",
        "keepsake",
        "bridged",  # (growth G2: attached to the neural duck; a migrated duck starts unattached)
        "bridge_company_event",
    }
    assert all(not getattr(core.s, name) for name in new_fields)
    # the counters: the outcome dice are the world's now, the rest Duck Core's
    theirs = dict(old.core.counters)
    assert set(core.counters) | set(world.counters) == {k for k, v in theirs.items() if v}
    assert not set(core.counters) & set(M.WORLD_PURPOSES)
    assert world.counters == {k: v for k, v in theirs.items() if k in M.WORLD_PURPOSES and v}
    assert core.counters == {k: v for k, v in theirs.items() if k not in M.WORLD_PURPOSES}
    # the walk under way is Duck Core's open act, and the world has it (with its fall)
    assert core.acts.busy() and core.acts.open["A1"].event_id == old.core.s.walk_plan["event"]
    assert world.state()["walk"] == "A1" and world.next_at_ms() is not None
    # nothing of the frozen engine is left in the new duck, its family or its world
    assert M.leftovers(moved) == []
    assert type(moved.household).__module__ == "duckmind.duckcore.routines"
    # the seam is on record
    assert moved.origin == {**v4.config_json(old), "engine_rev": "v4_0"}
    assert moved.inputs[-1] == (api.MIGRATE, {"to": "v4_1"}) and moved.inputs[:-1] == old.inputs
    assert moved.core.tape is not None and moved.core.tape.header["carried_over"]["head"] == head


def test_a_moved_duck_saves_restores_and_replays_through_the_seam() -> None:
    old = _to_seam("righting")
    moved = api.migrate(old)
    for method, params in LATER[:16]:
        _send(moved, method, params)
    saved = json.loads(json.dumps(api.save_json(moved)))
    assert saved["config"].get("engine_rev") == "v4_0" and saved["world"]["world"] == "legacy"
    again = api.restore(saved)
    assert isinstance(again, api.Session) and again.origin is not None
    assert again.core.log.head == moved.core.log.head
    assert len(again.core.log.events) == len(moved.core.log.events)
    for check in (api.replay_check, api.percept_replay_check, api.world_replay_check):
        result = check(moved)
        assert result["identical"] is True, (check.__name__, result)
    replayed = api.percept_replay_check(moved)
    assert replayed["all_percepts_taken"] and replayed["percepts"] > 0
    # the moved duck keeps living on the page's calls
    for method in ("state", "diary"):
        api.dispatch(moved, method, {})
    assert isinstance(api.snapshot(moved), dict) and api.banner(moved)["code"]


def test_only_a_v4_duck_moves_and_only_to_v4_1() -> None:
    old = v4.Session(v4.Config(seed=3))
    with pytest.raises(ValueError):
        api.migrate(old, "v4_0")
    with pytest.raises(ValueError):
        api.migrate(old, "v5_0")
    with pytest.raises(M.MigrationError):
        M.migrate(api.open_session({"engine_rev": "v4_1", "seed": 3}))
    moved = api.migrate(old)
    with pytest.raises(ValueError):
        api.migrate(moved)  # (it already runs v4_1)
    with pytest.raises(ValueError):
        api.dispatch(moved, "migrate", {"to": "v4_1"})  # (a new session: api.migrate)
    tampered = v4.Session(v4.Config(seed=3))
    tampered.core.hw = dataclasses.replace(tampered.core.hw, righting_ppm=0)
    with pytest.raises(M.MigrationError):
        api.migrate(tampered)  # (its inputs could not rebuild it)
