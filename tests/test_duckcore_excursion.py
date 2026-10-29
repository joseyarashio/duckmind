"""Outings (Duck Core v4, part C): small trips of a few legs, often, within the leash.

Each leg stays inside the validated walking bounds; only the first leg of an outing is a
decision; it goes where it has not been for a while (or to the family); personality only
shapes the randomness, never whether it goes out.
"""

from __future__ import annotations

import collections
import math
from functools import cache

from duckmind.duckcore import content, excursion
from duckmind.duckcore.api import Config, Session, dispatch
from duckmind.duckcore.engine import WALK_S_MAX, WALK_VX_MM, DuckCore
from duckmind.duckcore.narrate import describe
from duckmind.provenance import Event

LEASH_SLACK_MM = 300  # spec 11.5: never beyond the leash + 300 mm


def _kinds(core: DuckCore, kind: str) -> list[Event]:
    return [event for event in core.log.events if event.kind == kind]


@cache
def _life(seed: int, kind: str = "elder_home", days: int = 21, **config: object) -> DuckCore:
    session = Session(
        Config(seed=seed, nest=True, conservative=False, household=kind, **config)  # type: ignore[arg-type]
    )
    dispatch(session, "live", {"family": kind, "days": days})
    return session.core


def test_an_outing_is_several_legs_and_only_the_first_is_a_decision():
    core = _life(1)
    starts = [e for e in _kinds(core, "duck.excursion") if e.payload["phase"] == "start"]
    ends = [e for e in _kinds(core, "duck.excursion") if e.payload["phase"] == "end"]
    assert len(starts) >= 150 and len(ends) >= len(starts) - 1
    legs = [e.payload["legs"] for e in ends if e.payload["ended_by"] == "legs"]
    assert collections.Counter(legs).most_common(1)[0][0] >= 2
    assert sum(legs) / len(legs) >= 3
    decisions = {
        e.event_id
        for e in _kinds(core, "duck.decision")
        if e.payload["rule"] in ("outing-v2", "nest-leave-v1")
    }
    bids = {e.event_id for e in _kinds(core, "duck.bid") if e.payload["kind"] == "comes_over"}
    reflexes = {
        e.event_id for e in _kinds(core, "duck.decision") if e.payload["rule"] == "reflex-v1"
    }
    guesses = {  # a clever moment's guess (drawn and logged like a decision): off to see
        e.event_id for e in _kinds(core, "duck.predict") if e.payload.get("rule") == "moment-v1"
    }
    for start in starts:  # every outing starts from one logged decision (or its call)
        assert set(start.parent_event_ids) & (decisions | bids | reflexes | guesses), start.payload
    by_outing = collections.Counter(
        e.payload["excursion"] for e in _kinds(core, "duck.walk") if e.payload.get("excursion")
    )
    assert max(by_outing.values()) >= 4  # many legs, one decision


def test_every_leg_stays_inside_the_validated_bounds_and_the_leash():
    for seed in (2, 3):
        core = _life(seed)
        walks = _kinds(core, "duck.walk")
        assert walks
        for walk in walks:
            p = walk.payload
            vx_mm, vyaw = round(p["vx"] * 1000), abs(p["vyaw"])
            assert vx_mm == 0 or WALK_VX_MM[0] <= vx_mm <= WALK_VX_MM[1]
            assert vyaw <= content.LOCOMOTION["turn_vyaw"] + 1e-9
            assert p["seconds"] <= WALK_S_MAX
            assert not p["nested"] and not p["docked"] and not p["unplaced"]
            if p["purpose"] in ("explore", "social", "zoomies") and p["leash_mm"]:
                assert p["dist_to_mm"] <= p["leash_mm"] + LEASH_SLACK_MM, p
            if p["phase"] == "solo":
                assert p["dist_to_mm"] <= content.RHYTHM["solo_radius_mm"] + LEASH_SLACK_MM
            at = walk.occurred_at
            assert 7 <= at.hour < 22


def test_no_outings_while_the_family_is_out_and_home_legs_only_just_after():
    core = _life(4, "dual_income")
    for walk in _kinds(core, "duck.walk"):
        p = walk.payload
        if p["family"] in ("away", "unknown"):
            assert p["homing"] and p["purpose"] in ("home", "dock", "circle"), p


def test_it_goes_to_places_it_has_not_seen_for_a_while():
    core = _life(5)
    seen_per_day = collections.defaultdict(set)
    for walk in _kinds(core, "duck.walk"):
        x, y = walk.payload["expect"]["x_mm"] / 1000, walk.payload["expect"]["y_mm"] / 1000
        seen_per_day[walk.occurred_at.date()].add((int((x + 3) // 0.5), int((y + 3) // 0.5)))
    grown = sorted(seen_per_day)[8:]  # once it walks well
    cells = sorted(len(seen_per_day[day]) for day in grown)
    assert cells[len(cells) // 2] >= 8  # a median day visits at least 8 half-metre cells
    assert len(core.s.cells) >= 12


def test_every_personality_goes_out_often():
    """Personality shapes how, never whether: each of the four goes out many times a day."""
    for branch in ("explorer", "songbird", "snuggler", "dreamer"):
        session = Session(Config(seed=21, nest=True, conservative=False, household="hybrid"))
        core = session.core
        core.s.nature = {key: (7000 if key == branch else 1000) for key in core.s.nature}
        dispatch(session, "live", {"family": "hybrid", "days": 14})
        stats = [e.payload for e in _kinds(core, "duck.day_stats")][7:]
        per_day = sorted(p["excursions"] for p in stats if p["awake_min"] > 0)
        assert per_day[len(per_day) // 2] >= 10, (branch, per_day)


def test_outings_are_logged_with_what_they_were_for():
    core = _life(1)
    reasons = collections.Counter(
        e.payload["reason"] for e in _kinds(core, "duck.excursion") if e.payload["phase"] == "start"
    )
    assert reasons["curious"] and reasons["restless"] and reasons["social"]
    ended = collections.Counter(
        e.payload["ended_by"] for e in _kinds(core, "duck.excursion") if e.payload["phase"] == "end"
    )
    assert ended["legs"] > sum(ended.values()) // 2
    homes = [e for e in _kinds(core, "duck.decision") if e.payload["rule"] == "home-base-v1"]
    assert homes and {e.payload["verb"] for e in homes} == {"go_home", "sit_down"}


# ---------------------------------------------------------------------------- where it heads
# (fixed 2026-09-26: with nothing new worth the walk it used to head for a fixed point in
# front of the nest even when a spot it learned to avoid lay there, and turned on the spot
# toward it again and again; in a small room with a few such spots that was most of its
# walking.)

SPOT_R = 500


def _floor_duck(seed: int, days: int = 8) -> DuckCore:
    """A nest duck old enough to walk, at 10:00 with the family up and around."""
    core = Session(Config(seed=seed, nest=True, conservative=False)).core
    core.advance(days * 1440)
    core.advance(core.minutes_until("10:00"))
    for minute in range(3):
        core.hear_home(None, kind="voice", journal=False)
        core.hear_home(None, kind="person", journal=False)
        core.advance(1)
    core.s.battery = core.s.energy = 10000
    core.s.homing = core.s.stranded = core.s.need = core.s.excursion = None
    return core


def _stand_at(core: DuckCore, x: int, y: int, heading: int) -> None:
    core.place_on_nest("owner")
    core.s.nested = False
    core.s.posture = "stand"
    core.s.stand_idle_from = core.s.stood_at = core.s.now
    core.s.pos_x, core.s.pos_y, core.s.heading = x, y, heading
    core.s.pose_err_mm = 0


def _spots(core: DuckCore, *centres: tuple[int, int]) -> list[dict]:
    core.s.avoid_spots = {
        f"S{n}": {"x": x, "y": y, "r": SPOT_R, "active": True, "event": "E1"}
        for n, (x, y) in enumerate(centres, 1)
    }
    return core.active_spots()


def test_an_outing_never_heads_into_or_through_a_spot_it_learned_to_avoid():
    core = _floor_duck(3, days=60)  # grown enough to range 1.5 m
    spots = _spots(core, (650, 0), (-700, 650))  # one right where the old fallback was
    found: collections.Counter = collections.Counter()
    for x, y, heading in ((0, 0, 0), (300, -600, 1571), (-500, -300, 3000), (200, 900, -1571)):
        _stand_at(core, x, y, heading)
        for stale in (False, True):
            if stale:  # every place seen just now: nothing new worth the walk
                now = excursion._minute_index(core)
                core.s.cells = {key: [now, 0] for key, *_ in excursion.CENTERS}
            for reason in ("curious", "restless", "patrol", "social", "play"):
                for _ in range(4):
                    target = excursion.pick_target(core, reason)
                    if target is None:
                        continue  # (nowhere from here: see the next test)
                    found[stale] += 1
                    assert target != excursion.FRONT_MM
                    for spot in spots:
                        gap = math.dist(target, (spot["x"], spot["y"]))
                        assert gap >= SPOT_R + 100, (x, y, reason, target)
                    assert excursion._clear_way(spots, (x, y), *target), (x, y, target)
    assert found[False] >= 60 and found[True] >= 40, found


def test_with_nowhere_left_to_go_it_stays_put_instead_of_turning_on_the_spot(monkeypatch):
    core = _floor_duck(4)
    core.s.roam_radius_mm = 1000
    _spots(core, (650, 0), (-650, 0), (0, 650), (0, -650))  # a small room full of them
    _stand_at(core, 0, 300, 1571)
    for reason in ("curious", "restless", "patrol", "social", "play"):
        assert excursion.pick_target(core, reason) is None
    assert not excursion.somewhere_to_go(core, "curious")
    # Its own urge to go out does not even get it up…
    monkeypatch.setattr(excursion.rng, "below", lambda raw, chance: True)
    core.s.posture = "sit"
    decisions = len(_kinds(core, "duck.decision"))
    assert excursion.can_start(core) is None
    assert not excursion.maybe_start(core)
    assert len(_kinds(core, "duck.decision")) == decisions
    # …and an outing something else starts (a call, a reflex) is over before a single leg.
    monkeypatch.undo()
    core.s.posture = "stand"
    walks = len(_kinds(core, "duck.walk"))
    if excursion.set_out(core, "curious"):
        end = [e for e in _kinds(core, "duck.excursion") if e.payload["phase"] == "end"][-1]
        assert end.payload["ended_by"] == "nowhere" and end.payload["legs"] == 0
        assert "沒有能去的地方" in describe(end)
    assert not [w for w in _kinds(core, "duck.walk")[walks:] if w.payload.get("excursion")]
    # The same duck with room to go does set off.
    core.s.avoid_spots = {}
    core.s.excursion = core.s.homing = core.s.last_excursion_end = None
    _stand_at(core, 0, 300, 1571)
    core.s.posture = "sit"
    monkeypatch.setattr(excursion.rng, "below", lambda raw, chance: True)
    assert excursion.somewhere_to_go(core, "curious")
    assert excursion.can_start(core) is None
    assert excursion.maybe_start(core)


# ---------------------------------------------------------------------------- curiosity sockets
# (phase 4, WP7: curiosity P2 plugs in here; until then the sockets are inert.)


def test_the_curiosity_sockets_do_nothing_while_its_home_has_no_things():
    """Every socket answers its inert value and leaves the duck exactly as it was: no draw
    (no counter moves), no event, no state change."""
    from duckmind.duckcore import curiosity

    core = _floor_duck(5)
    _stand_at(core, 0, 300, 1571)
    assert not curiosity.has_things(core) and not curiosity.active(core)
    counters, events, state = dict(core.counters), len(core.log.events), repr(core.s)
    record = core.log.events[-1]
    assert curiosity.minute(core) is False
    assert curiosity.at_pause(core) is False
    assert curiosity.wants_toy(core) is False
    assert curiosity.attractions(core) == {} and curiosity.attraction(core, 0) == 0
    assert curiosity.zone_block(core, (0, 0)) is None
    assert curiosity.holds_body(core) is False and curiosity.next_due(core) is None
    curiosity.on_fall(core, {"x": 0, "y": 0})
    curiosity.on_sense(core, {"kind": "sense.thing", "things": []}, record)
    curiosity.on_result(core, {"kind": "body.curio_step", "id": "C1", "step": 1}, record)
    curiosity.on_guard(core, {"kind": "body.guard"}, record)
    curiosity.on_timeout(core, "C1")
    assert curiosity.on_home(core, "home.thing", {"op": "register"}, record) is None
    assert core.counters == counters and len(core.log.events) == events and repr(core.s) == state
    # …and never a reason to set off for a toy.
    assert curiosity.TOY_REASON not in {excursion.pick_reason(core) for _ in range(40)}


def test_outings_reach_the_curiosity_sockets_and_nothing_changes(monkeypatch):
    """The sockets are really on the outing's path (the look between legs, the target's pull,
    the reason to set off), and a life with them is the life without them."""
    from duckmind.duckcore import curiosity

    calls: collections.Counter[str] = collections.Counter()

    def counted(name: str, value: object):  # type: ignore[no-untyped-def]
        def socket(*args: object) -> object:
            calls[name] += 1
            return value

        return socket

    def life() -> str:
        session = Session(Config(seed=31, nest=True, conservative=False, household="elder_home"))
        dispatch(session, "live", {"family": "elder_home", "days": 12})
        assert not [
            e
            for e in _kinds(session.core, "duck.excursion")
            if e.payload.get("reason") == curiosity.TOY_REASON
        ]
        return session.core.log.head

    with_sockets = life()
    monkeypatch.setattr(curiosity, "at_pause", counted("at_pause", False))
    monkeypatch.setattr(curiosity, "attractions", counted("attractions", {}))
    monkeypatch.setattr(curiosity, "wants_toy", counted("wants_toy", False))
    assert life() == with_sockets
    assert calls["at_pause"] >= 50 and calls["attractions"] >= 50 and calls["wants_toy"] >= 20, (
        calls
    )
