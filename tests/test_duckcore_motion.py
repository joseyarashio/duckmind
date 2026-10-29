"""Standing up, sitting down and short walks in floor mode (owner decisions 2026-09-25).

The duck's posture is autonomous: drives × personality × growth × learned preference ×
walk satiation, decided in logged draws and utility decisions, and never dependent on
someone being present. A rested floor duck in the daytime always stands up and walks
about ("鴨子一定會站，而且會走動"); personality changes how often, never whether. Duck
Core tracks where it is in the room; after a fall it tries to roll back up, and only
three hard falls near one spot teach it to keep away from there. Hard gates never move.

Phase 4: falls, each self-righting try and whether it gets up are the world's (LegacyWorld:
v4's own dice and schedule, moved out); a test turns the world's rates where v4's tests
turned the duck's hardware, and the family's hands on its head go through the world's
detectors (``LegacyActs``).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from datetime import datetime, time, timedelta
from typing import Any

import pytest

from duckmind import worldport
from duckmind.duckcore import commands, content, query
from duckmind.duckcore.api import (
    Config,
    LegacyActs,
    Session,
    _motion,
    dispatch,
    legacy_world,
    snapshot,
    v4_world_fields,
)
from duckmind.duckcore.engine import (
    AVOID_RADIUS_MM,
    MIN_WALK_S,
    MIN_WANDER_S,
    PATH_STEPS_PER_S,
    ROOM_LIMIT_MM,
    STAND_S_MAX,
    DuckCore,
    HardwareConfig,
    pose_along,
)
from duckmind.duckcore.memory import DIARY_BANNED
from duckmind.duckcore.routines import ROUTINES, Household
from duckmind.provenance import Event
from duckmind.simweb.server import SimState

LOW_VX, HIGH_VX = content.LOCOMOTION["walk_vx"]
TURN = content.LOCOMOTION["turn_vyaw"]
WALK_S_MAX = content.LOCOMOTION["walk_s_max"]
MOVED = HardwareConfig(motion_validated=True)
# The world's outcome rates (phase 4: the LegacyWorld header, no longer the duck's hardware).
V4_RATES = v4_world_fields()
ALWAYS_FALLS = {"fall_ppm_per_s": 1_000_000}
BRANCHES = ("explorer", "songbird", "snuggler", "dreamer")


def _life(seed: int, days: int, routine: str = "floor", **config: bool) -> Session:
    session = Session(Config(seed=seed, **config))  # type: ignore[arg-type]
    dispatch(session, "live", {"days": days, "routine": routine})
    return session


def _juvenile(seed: int, branch: str) -> Session:
    """Fifty days on the floor routine (a few more if it grows up slower: each duck has
    its own pace), then (if needed) the family picks the branch."""
    session = _life(seed, 50)
    while session.core.s.branch is None:
        dispatch(session, "live", {"days": 1, "routine": "floor"})
    dispatch(session, "choose", {"kind": "branch", "option": branch})
    assert session.core.s.branch == branch
    return session


def _kinds(core: DuckCore, kind: str) -> list[Event]:
    return [event for event in core.log.events if event.kind == kind]


def _core(seed: int, hardware: HardwareConfig = MOVED, **rates: int) -> DuckCore:
    """A bare Duck Core in its LegacyWorld (v4's physics, at ``rates``)."""
    port = worldport.make_world(legacy_world(Config(seed=seed), rates))
    return DuckCore(seed=seed, hardware=hardware, port=port)


def _rates(core: DuckCore, **rates: int) -> None:
    """Turn the world's outcome rates (as v4's tests turned ``core.hw``)."""
    for name, value in rates.items():
        assert name in V4_RATES
        setattr(core.acts.port, name, value)


def _hands(core: DuckCore) -> LegacyActs:
    """The family's hands on it (a touch or a hand on its head goes through the world)."""
    return LegacyActs(core, core.acts.port)


def _pose(value: Any) -> tuple[float, float, float]:
    """A walk's ``bel_from`` / ``expect`` (mm, mrad) as v4's [x m, y m, heading rad]."""
    return value["x_mm"] / 1000, value["y_mm"] / 1000, value["yaw_mrad"] / 1000


def _fall_place(core: DuckCore, fell: Event) -> Any:
    """Where it lay after a fall: its own duck.posture(fallen) record answers the body.fell."""
    return next(
        e.payload["at"]
        for e in _kinds(core, "duck.posture")
        if e.payload["posture"] == "fallen" and fell.event_id in e.parent_event_ids
    )


def _walks(core: DuckCore, verb: str | None = None) -> list[Event]:
    return [event for event in _kinds(core, "duck.walk") if verb in (None, event.payload["verb"])]


def _self_stands(core: DuckCore) -> list[Event]:
    """Every time it stood up by itself: its stand-up acts that the body answered "stand"
    (v4_1), and its own stops out of a walk."""
    return [
        event
        for event in core.log.events
        if (
            event.kind == "duck.posture"
            and event.payload["posture"] == "stand"
            and event.payload["by"] == "self"
        )
        or (event.kind == "body.posture" and event.payload["got"] == "stand")
    ]


def _daytime_on_floor(core: DuckCore, hour: int = 10) -> None:
    """The next day at ``hour``: a fresh battery, and the family puts it on the floor."""
    if core.s.awake:
        core.power_off(None)
    day = core.s.now.date() + timedelta(days=1)
    target = datetime.combine(day, time(hour), core.s.now.tzinfo)
    core.advance(int((target - core.s.now).total_seconds() // 60))
    core.power_on("owner")
    core.set_setting("floor", True)


def _until(core: DuckCore, done: Callable[[], bool], minutes: int = 900) -> bool:
    """Leave the duck to itself on the floor, a minute at a time, until ``done``."""
    _daytime_on_floor(core)
    for _ in range(minutes):
        if done():
            return True
        if not core.s.awake or core.s.now.hour >= 21:
            _daytime_on_floor(core)
        core.advance(1)
    return done()


# ---------------------------------------------------------------------------- determinism


def test_walking_ducks_replay_identically():
    session = _life(3, 70)
    assert _walks(session.core, "wander") and _self_stands(session.core)
    dispatch(session, "act", {"action": "pickup", "persona": "child"})
    assert dispatch(session, "replay_check", {})["identical"] is True
    again = _life(3, 70)
    dispatch(again, "act", {"action": "pickup", "persona": "child"})
    assert again.core.log.head == session.core.log.head
    assert session.core.log.verify()


# ---------------------------------------------------------------------------- hard gates


@pytest.fixture(scope="module")
def lives() -> list[Session]:
    return [_life(1, 120, "daily"), _life(2, 90, "floor"), _life(6, 60, "weekends")]


def test_walks_happen_only_in_floor_mode(lives):
    for session in lives:
        floor, walks = False, 0
        for event in session.core.log.events:
            if event.kind == "env.floor":
                floor = bool(event.payload["floor"])
            elif event.kind == "duck.walk":
                walks += 1
                assert floor and event.payload["floor"], event.event_id
        assert walks, "months with floor visits should include some walking"


def test_on_the_desk_it_stands_up_and_sits_down_but_never_walks(monkeypatch):
    monkeypatch.setitem(ROUTINES, "desk", dataclasses.replace(ROUTINES["daily"], floor_p=0))
    core = _life(2, 80, "desk").core
    assert _self_stands(core), "standing up is allowed anywhere"
    assert not _walks(core)
    assert not [event for event in _kinds(core, "env.floor") if event.payload["floor"]]


def test_never_at_night_never_below_the_battery_floor(lives, monkeypatch):
    late_visit = dataclasses.replace(ROUTINES["floor"], visits=((21, 20, 75),))
    monkeypatch.setitem(ROUTINES, "late", late_visit)
    late = _life(4, 60, "late")
    assert any(event.occurred_at.hour >= 22 for event in _kinds(late.core, "sense.touch"))
    for session in [*lives, late]:
        chosen = _walks(session.core) + _self_stands(session.core)
        assert chosen
        for event in chosen:
            assert 7 <= event.occurred_at.hour < 22, event.event_id
        for event in _walks(session.core):
            assert event.payload["battery"] >= content.LOCOMOTION["min_battery"]
        for event in _kinds(session.core, "duck.decision"):
            if event.payload["verb"] in {"stand_up", "wander", "turn_around"}:
                assert event.payload["inputs"]["battery"] >= content.LOCOMOTION["min_battery"]


def test_without_the_physics_validation_it_never_stands_up_or_walks_by_itself():
    session = _life(2, 90, "floor", motion_validated=False)
    core = session.core
    assert not _walks(core)
    assert not _self_stands(core)
    postures = _kinds(core, "duck.posture")
    # The gamepad may still stand it up; sitting down again (the rest pose) is always allowed.
    assert [event for event in postures if event.payload["reason"] == "gamepad"]
    assert [event for event in postures if event.payload["by"] == "self"]
    assert all(
        event.payload["posture"] == "sit" for event in postures if event.payload["by"] == "self"
    )
    assert snapshot(session)["body"]["motion_validated"] is False
    assert DuckCore(seed=1).hw.motion_validated is False  # the real robot waits for the A100


def test_floor_mode_ends_on_pickup_and_power_off():
    core = _core(5)
    core.set_setting("floor", True)
    assert core.s.floor
    core.pickup("child")
    assert not core.s.floor and core.s.posture == "sit"
    mat = (0.0, 0.0, 0.0)  # picked up (or put down) means back at its mat
    assert dict(_kinds(core, "env.floor")[-1].payload) == {
        "floor": False,
        "why": "pickup",
        "pose": mat,
    }
    core.set_setting("floor", True)
    core.power_off("owner")
    assert not core.s.floor
    assert dict(_kinds(core, "env.floor")[-1].payload) == {
        "floor": False,
        "why": "power_off",
        "pose": mat,
    }
    core.power_on("owner")
    assert not core.s.floor and core.s.posture == "sit"


def test_the_duck_never_stands_still_longer_than_it_may(lives):
    """Standing without walking is bounded (stand_s_max) unless a hand holds it still."""
    for session in lives:
        posture, still_since, held_until, driven = "sit", None, None, False
        for event in session.core.log.events:
            at, p = event.occurred_at, event.payload
            if event.kind == "pad.play_start":
                driven = True
            elif event.kind == "pad.play_end":
                driven, still_since = False, at
            elif event.kind == "duck.stop":
                held_until = at + timedelta(seconds=60)
            stood = event.kind == "body.posture" and p["got"] == "stand"
            moved = event.kind in {"duck.posture", "duck.walk"} or stood
            if moved and posture == "stand" and not driven and still_since is not None:
                start = max(still_since, held_until) if held_until else still_since
                assert (at - start).total_seconds() <= STAND_S_MAX, event.event_id
            if event.kind == "duck.posture" or stood:
                posture = "stand" if stood else p["posture"]
                still_since = at if posture == "stand" else None
            elif event.kind == "duck.walk":
                still_since = at + timedelta(seconds=p["seconds"])
            elif event.kind == "body.fell":
                posture = "fallen"


def test_every_walk_stays_inside_the_validated_bounds(lives):
    for session in lives:
        for event in _walks(session.core):
            p = event.payload
            assert MIN_WALK_S <= p["seconds"] <= WALK_S_MAX
            assert abs(p["vyaw"]) <= TURN
            if p["verb"] == "wander":
                # Never slower than the policy's start threshold: gentler means shorter.
                assert LOW_VX <= p["vx"] <= HIGH_VX
                assert abs(p["vyaw"]) <= 0.5
                assert 0.11 <= p["speed"] <= 0.16  # measured ground speed, not the command
                assert p["cm"] == round(p["speed"] * 1000) * p["seconds"] // 10
            elif p["vx"] > 0:
                # A turn walks a tight arc: the slowest walk at the full turn command.
                assert p["verb"] == "turn_around"
                assert p["vx"] == LOW_VX and abs(p["vyaw"]) == TURN
                assert p["cm"] == round(p["speed"] * 1000) * p["seconds"] // 10
            else:
                # On the spot only when every arc was blocked.
                assert abs(p["vyaw"]) == TURN and p["cm"] == 0


def test_it_turns_by_walking_an_arc_not_by_marching_on_the_spot(lives):
    """The official walk barely turns on the spot (5–11°/s while the legs keep stepping),
    so a turn is a tight arc; a walk of its own is long enough to get going."""
    turns = on_spot = 0
    for session in lives:
        for event in _walks(session.core):
            p = event.payload
            if p["verb"] == "turn_around":
                turns += 1
                on_spot += p["vx"] == 0
            elif p.get("purpose") is None:
                assert p["seconds"] >= MIN_WANDER_S
    assert turns and on_spot * 20 <= turns, (on_spot, turns)


def test_conservative_mode_walks_shorter_and_rarer():
    bold = _walks(_life(2, 90, "floor", conservative=False).core)
    careful = _walks(_life(2, 90, "floor", conservative=True).core)
    assert careful and bold
    assert all(event.payload["seconds"] <= WALK_S_MAX / 2 for event in careful)
    assert max(event.payload["seconds"] for event in bold) > WALK_S_MAX / 2
    assert all(
        LOW_VX <= e.payload["vx"] <= HIGH_VX for e in careful if e.payload["verb"] == "wander"
    )
    assert len(careful) < len(bold)


# ---------------------------------------------------------------------------- growth and personality


def test_no_steps_before_first_steps_and_grown_ducks_walk_more_than_hatchlings():
    core = _life(2, 70).core
    first_steps = next(
        event
        for event in _kinds(core, "growth.milestone")
        if event.payload["milestone"] == "first_steps"
    )
    assert first_steps.payload["title"] == "第一次自己走路"
    assert core.s.milestones["first_steps"]["gd"] >= content.FIRST_STEPS_GD
    wanders, turns = _walks(core, "wander"), _walks(core, "turn_around")
    assert wanders[0].event_id < first_steps.event_id  # that first walk *is* the milestone
    assert turns and turns[0].event_id > first_steps.event_id  # turning around comes after

    # Walks per hour on the floor: the hatchling (GD < 10) against the older duck.
    gd, floor, rates = 0, False, {"young": [0, 0.0], "older": [0, 0.0]}
    last = core.log.events[0].occurred_at
    for event in core.log.events:
        bucket = rates["young" if gd < 10 else "older"]
        if floor:
            bucket[1] += (event.occurred_at - last).total_seconds() / 3600
        last = event.occurred_at
        if event.kind == "growth.gd":
            gd = event.payload["gd"]
        elif event.kind == "env.floor":
            floor = event.payload["floor"]
        elif event.kind == "duck.walk":
            bucket[0] += 1
    young, older = rates["young"], rates["older"]
    assert young[1] > 2 and older[1] > 10
    assert young[0] / young[1] > 3  # even the hatchling toddles about on the floor
    assert older[0] / older[1] > 1.5 * young[0] / young[1]


def _moves(branch: str) -> tuple[int, int]:
    walks = stands = 0
    for seed in (1, 2, 3):
        session = _juvenile(seed, branch)
        walks_before, stands_before = len(_walks(session.core)), len(_self_stands(session.core))
        dispatch(session, "live", {"days": 30, "routine": "floor"})
        walks += len(_walks(session.core)) - walks_before
        stands += len(_self_stands(session.core)) - stands_before
    return walks, stands


def test_an_explorer_walks_more_than_a_snuggler_or_a_dreamer_but_they_all_walk():
    explorer, snuggler, dreamer = (_moves(branch) for branch in ("explorer", "snuggler", "dreamer"))
    # About 1.5×: with the family's praise reaching what it does mid-minute (the visits now
    # look up a second after its beat), liked walks lift every personality alike, so the
    # ratio sat at 1.49–1.51 across seeds; since every duck has its own temperament (P1,
    # 2026-09-26: its sociability and pace vary too) it sits at 1.44–1.48 over seed
    # triples 1–12. Personality shapes the rate, never whether.
    assert explorer[0] > 1.4 * snuggler[0]
    assert explorer[0] > 1.4 * dreamer[0]
    assert snuggler[0] > 0.3 * explorer[0] and dreamer[0] > 0.3 * explorer[0]
    assert min(snuggler[1], dreamer[1]) > 0.5 * explorer[1]  # they all get up, often


def _rested_hour(session: Session) -> tuple[int, int]:
    """Walks and turns, and times it stood up by itself, in its first hour on the floor."""
    core = session.core
    _daytime_on_floor(core)
    walks, stands = len(_walks(core)), len(_self_stands(core))
    core.advance(60)
    return len(_walks(core)) - walks, len(_self_stands(core)) - stands


def test_every_personality_stands_up_and_walks_about_many_times_an_hour():
    """ "鴨子一定會站，而且會走動，只差在隨機分佈性": personality changes the rate only.

    A fresh battery lasts about an hour and the duck sits for good below 30%, so this hour
    holds ~45 minutes of walking time (conservative mode, as on a new robot).
    """
    hours = {
        branch: [_rested_hour(_juvenile(seed, branch)) for seed in (1, 2)] for branch in BRANCHES
    }
    for branch, results in hours.items():
        for moves, stands in results:
            assert moves >= 6 and stands >= 3, (branch, results)
        assert sum(moves for moves, _ in results) / len(results) >= 8, (branch, results)
    mean = {branch: sum(moves for moves, _ in results) for branch, results in hours.items()}
    assert mean["explorer"] > mean["snuggler"] and mean["explorer"] > mean["dreamer"]
    for seed in (1, 2):  # a duckling (GD 20) is slower, but clearly on the move
        moves, stands = _rested_hour(_life(seed, 20))
        assert moves >= 5 and stands >= 3


def test_only_rest_night_and_the_hard_gates_stop_it():
    core = _juvenile(1, "dreamer").core
    _daytime_on_floor(core)
    core.owner_enough()  # "夠了": quiet for 15 minutes
    start = len(_walks(core))
    core.advance(14)
    assert len(_walks(core)) == start
    core.s.energy = 2000  # too tired to stand: it sits and naps rather than walk
    core.advance(10)
    assert len(_walks(core)) == start


# ---------------------------------------------------------------------------- reflexes


def _greet_after_a_day_apart(seed: int, days: int) -> tuple[list[Event], bool]:
    """The owner says hello after a day apart: the events it produced, and whether the
    duck was already up (a floor duck often is, and then it has nothing to stand for)."""
    core = _life(seed, days, "daily").core
    _daytime_on_floor(core)
    core.set_setting("floor", False)
    core.advance(2)
    core.s.last_owner_hello = core.s.now - timedelta(hours=30)
    start = len(core.log.events)
    already_up = core.s.posture != "sit"
    core.owner_hello()
    return core.log.events[start:], already_up


def _stood(events: list[Event]) -> bool:
    """It stood up for the greeting: the body answered "stand" to the stand-up act that the
    greet_stand reflex started."""
    by_id = {event.event_id: event for event in events}
    for event in events:
        if event.kind != "body.posture" or event.payload["got"] != "stand":
            continue
        command = by_id.get(event.parent_event_ids[0]) if event.parent_event_ids else None
        for parent in command.parent_event_ids if command is not None else ():
            if parent in by_id and by_id[parent].payload.get("trigger") == "greet_stand":
                return True
    return False


def test_greeting_after_time_apart_makes_a_grown_duck_stand_up():
    # 24 ducks each (12 made this a coin toss once every duck had its own temperament: a
    # hatchling's style follows its nature, so a few stand for a greeting and a few don't).
    grown = [_greet_after_a_day_apart(seed, 60) for seed in range(1, 25)]
    young = [_greet_after_a_day_apart(seed, 1) for seed in range(1, 25)]
    stood = sum(_stood(events) or up for events, up in grown)
    assert stood >= 14  # up for the greeting: stood for it, or was already standing
    assert sum(_stood(events) for events, _ in young) <= stood - 5  # a hatchling mostly perks up
    events = next(events for events, _ in grown if _stood(events))
    decision = next(event for event in events if event.kind == "duck.decision")
    assert decision.payload["rule"] == "reflex-v1"
    assert decision.payload["trigger"] == "greet_stand" and decision.payload["verb"] == "stand_up"
    kinds = [event.kind for event in events]
    happy = next(
        index
        for index, event in enumerate(events)
        if event.kind == "duck.expression" and event.payload["reason"] == "owner_greeting"
    )
    assert kinds.index("body.posture") < happy  # stand up first, then the happy greeting


def test_a_hand_stops_a_walking_duck_at_once_without_sitting_it_down():
    core = _juvenile(1, "explorer").core
    assert _until(core, lambda: core.s.posture == "walk")
    walks = len(_walks(core))
    _hands(core).hold_head("child")
    held_at = core.s.now
    stop = _kinds(core, "duck.posture")[-1]
    assert stop.payload["posture"] == "stand" and stop.payload["from"] == "walk"
    assert stop.payload["reason"] == "held" and stop.payload["walk_cut"]["cm_short"] >= 0
    assert _kinds(core, "duck.stop")[-1].payload["posture"] == "stand"
    core.advance(2)
    later = [
        event
        for event in core.log.events
        if event.kind in {"duck.posture", "duck.walk"} and event.occurred_at > stop.occurred_at
    ]
    # Nothing moves while the hand is there (60 s); afterwards it may sit or walk on.
    assert all(event.occurred_at >= held_at + timedelta(seconds=57) for event in later)
    assert len(_walks(core)) >= walks


def test_pickup_sits_it_down_and_a_snuggler_settles_under_a_hand_more_than_an_explorer():
    settle = {}
    for branch in ("snuggler", "explorer"):
        went = tried = 0
        for seed in (1, 2, 3, 4):
            session = _juvenile(seed, branch)
            start = len(session.core.log.events)
            dispatch(session, "live", {"days": 20, "routine": "floor"})
            for event in session.core.log.events[start:]:
                if event.kind == "duck.decision" and event.payload.get("trigger") == "touch_settle":
                    tried += 1
                    went += event.payload["verb"] == "sit_down"
        settle[branch] = (went, tried)
    snuggler, explorer = settle["snuggler"], settle["explorer"]
    assert snuggler[1] >= 5 and explorer[1] >= 5
    assert snuggler[0] / snuggler[1] > explorer[0] / explorer[1] + 0.3

    core = _core(2)
    core.pad_play("child", minutes=1)  # the gamepad stood it up
    assert core.s.posture == "stand"
    core.pickup("child")
    assert core.s.posture == "sit"


# ---------------------------------------------------------------------------- falls and memory


def _fell_once_more(core: DuckCore) -> bool:
    falls = len(_kinds(core, "body.fell"))
    return _until(core, lambda: len(_kinds(core, "body.fell")) > falls)


def test_after_a_fall_it_tries_to_roll_back_up_by_itself():
    easy = _juvenile(1, "explorer").core
    _rates(easy, **ALWAYS_FALLS, righting_ppm=1_000_000)
    assert _fell_once_more(easy)
    fell = _kinds(easy, "body.fell")[-1]  # (this fall: up again, it may walk and fall anew)
    assert fell.source.kind == "legacy-world" and fell.payload["during"] == "walk"
    easy.advance(1)
    first = next(e for e in _kinds(easy, "body.righting") if fell.event_id in e.parent_event_ids)
    assert first.payload["tries"] == 1 and first.payload["ok"]  # up at the first try: easy
    assert first.source.kind == "legacy-world"
    assert (first.occurred_at - fell.occurred_at).total_seconds() == 2
    up = next(e for e in _kinds(easy, "duck.posture") if first.event_id in e.parent_event_ids)
    assert dict(up.payload) == {
        "posture": "sit",
        "from": "fallen",
        "reason": "righted",
        "by": "self",
    }

    stuck = _juvenile(2, "explorer").core
    _rates(stuck, **ALWAYS_FALLS, righting_ppm=0)
    assert _fell_once_more(stuck)
    stuck.advance(5)
    last = _kinds(stuck, "body.fell")[-1]  # (its life so far may hold an earlier fall)
    tries = [e for e in _kinds(stuck, "body.righting") if last.event_id in e.parent_event_ids]
    assert [e.payload["tries"] for e in tries] == [1, 2, 3]  # one record per try (A12)
    assert [(b.occurred_at - a.occurred_at).total_seconds() for a, b in zip(tries, tries[1:])] == [
        4,
        4,
    ]
    assert not any(e.payload["ok"] for e in tries)  # three misses: stuck
    assert stuck.s.posture == "fallen" and "翻不回來" in stuck.walk_reason()
    stuck.pickup("owner")
    assert stuck.s.posture == "sit" and stuck.pose_now() == [0.0, 0.0, 0.0]


def test_one_fall_or_easy_recoveries_do_not_make_it_walk_less():
    walked = {"fell": 0, "steady": 0}
    for seed in range(1, 7):
        for label in ("fell", "steady"):
            session = _juvenile(seed, "explorer")
            core = session.core
            if label == "fell":
                _rates(core, **ALWAYS_FALLS)
                assert _fell_once_more(core)
                _rates(core, **V4_RATES)
                core.advance(1)
                if core.s.posture == "fallen":
                    core.pickup("owner")
            start = len(_walks(core))
            dispatch(session, "live", {"days": 3, "routine": "floor"})
            walked[label] += len(_walks(core)) - start
            learned = [e.payload["parameter"] for e in _kinds(core, "learning.committed")]
            assert not [p for p in learned if p.startswith("self.")]  # nothing about walking
    assert walked["fell"] > 0.85 * walked["steady"]

    # Many falls it got up from easily, all near its mat: still nothing to keep away from.
    core = _juvenile(3, "explorer").core
    _rates(core, **ALWAYS_FALLS, righting_ppm=1_000_000)
    for _ in range(5):
        assert _fell_once_more(core)
    assert not core.active_spots() and not core.s.avoid_spots


def _path_points(event: Event) -> list[tuple[int, int]]:
    """Duck Core's own path for a logged walk, re-integrated from its payload."""
    p = event.payload
    start = (p["bel_from"]["x_mm"], p["bel_from"]["y_mm"], p["bel_from"]["yaw_mrad"])
    plan = {
        "from": start,
        "speed_mm": round(p["speed"] * 1000),
        "yaw_mrad": round(p["yaw_rate"] * 1000),
    }
    steps = p["seconds"] * PATH_STEPS_PER_S
    return [pose_along(plan, k)[0][:2] for k in range(steps + 1)]


def _hard_falls_near_the_mat(seed: int) -> DuckCore:
    """Falls it can hardly get up from (never, here), each near its mat, until it learns."""
    core = _juvenile(seed, "explorer").core
    _rates(core, **ALWAYS_FALLS, righting_ppm=0)
    for _ in range(12):
        assert _fell_once_more(core)
        core.advance(1)
        assert core.s.posture == "fallen"
        if core.active_spots():
            break
        core.pickup("owner")  # back to the mat, and on the floor again at the next _until
    return core


def test_three_hard_falls_near_one_spot_teach_it_to_walk_around_that_spot():
    core = _hard_falls_near_the_mat(1)
    (spot,) = core.active_spots()
    learned = [
        e
        for e in _kinds(core, "learning.committed")
        if e.payload["parameter"] == "self.avoid_spot:S1"
    ]
    (lesson,) = learned
    evidence = set(lesson.payload["evidence_event_ids"])
    falls = [e for e in _kinds(core, "body.fell") if e.event_id in evidence]
    assert len(falls) >= 3 and lesson.payload["rule_version"] == "avoid-spot-v1"
    for fall in falls:  # all near that spot, and each a hard recovery
        x, y = (round(value * 1000) for value in _fall_place(core, fall)[:2])
        assert (x - spot["x"]) ** 2 + (y - spot["y"]) ** 2 <= (2 * AVOID_RADIUS_MM) ** 2
    fact = core.memory().facts["F-avoid-S1"]
    assert "不太敢走過去" in fact.text and lesson.event_id in fact.evidence
    diary = " ".join(line["text"] for day in query.diary(core, 60) for line in day["lines"])
    assert "不太敢走過去" in diary and "翻不回來" in diary
    assert not [word for word in DIARY_BANNED if word in diary + fact.text]
    assert "不太敢走過去" in query.search(core, "有沒有不敢去的地方？")["answer"]

    # From now on its walks keep away from the spot (from inside it, they only lead out).
    _rates(core, **V4_RATES)
    core.pickup("owner")
    start = len(core.log.events)
    for _ in range(3):
        _until(core, lambda: False, minutes=45)
    walks = [e for e in core.log.events[start:] if e.kind == "duck.walk"]
    assert len(walks) > 20
    radius2 = spot["r"] ** 2
    for walk in walks:
        points = _path_points(walk)
        start_d2 = (points[0][0] - spot["x"]) ** 2 + (points[0][1] - spot["y"]) ** 2
        for x, y in points:
            d2 = (x - spot["x"]) ** 2 + (y - spot["y"]) ** 2
            assert d2 >= radius2 or d2 >= start_d2, walk.event_id  # never further in
    assert any(walk.payload.get("avoided") == "spot" for walk in walks)

    # Forgetting the spot takes the lesson back, like any other learning.
    result = core.forget("F-avoid-S1")
    assert result["ok"] and not result["still_there"]
    assert not core.active_spots() and "F-avoid-S1" not in core.memory().facts
    rolled = _kinds(core, "learning.rolled_back")[-1]
    assert rolled.payload["parameter"] == "self.avoid_spot:S1"
    core.undo_forget()
    assert core.active_spots() == [spot]


def test_falls_and_walks_become_gentle_memories():
    core = _core(4, **ALWAYS_FALLS)
    Household(4).live(_hands(core), 30, "floor")  # type: ignore[arg-type]
    assert _kinds(core, "body.fell")
    fell = core.memory().facts["F-fell"]
    assert "跌倒" in fell.text and fell.category == "habit"
    diary = " ".join(line["text"] for day in query.diary(core, 60) for line in day["lines"])
    assert "跌了一跤" in diary and "翻回來" in diary
    assert not [word for word in DIARY_BANNED if word in diary + fell.text]
    result = core.forget("F-fell")
    assert result["ok"] and not result["still_there"]
    assert "F-fell" not in core.memory().facts
    assert not core.active_spots()  # a spot the forgotten falls taught goes with them


def test_walking_is_remembered_and_can_be_forgotten_but_first_steps_cannot():
    session = _juvenile(1, "explorer")
    dispatch(session, "live", {"days": 20, "routine": "floor"})
    core = session.core
    memory = core.memory()
    explore = memory.facts["F-explore"]
    assert "到處走走" in explore.text and "公尺" in explore.text
    assert all(core.log.get(event_id).kind == "duck.walk" for event_id in explore.evidence)
    diary = [line["text"] for day in query.diary(core, 90) for line in day["lines"]]
    assert any(text.startswith("今天我自己走了約") for text in diary)
    assert "walk" in {episode.kind for episode in memory.episodes}
    assert "我喜歡自己到處走走" in query.search(core, "你喜歡到處走嗎？")["answer"]
    assert "F-ms-first_steps" in memory.facts
    assert dispatch(session, "forget", {"fact_id": "F-ms-first_steps"})["ok"] is False
    forgotten = dispatch(session, "forget", {"fact_id": "F-explore"})
    assert forgotten["ok"] and not forgotten["still_there"]
    assert "F-explore" not in core.memory().facts


def test_praise_right_after_a_walk_teaches_that_the_family_likes_it():
    core = _juvenile(1, "explorer").core

    def just_walked() -> bool:
        flag = core.s.last_flag
        return flag is not None and flag["id"] == "wander" and flag["at"] == core.s.now

    assert _until(core, just_walked)
    before = core.s.prefs.get("household.verb:wander", 5000)
    assert "走走" in core.pad_good("child")["note"]
    assert core.s.prefs["household.verb:wander"] > before
    learned = [
        e
        for e in _kinds(core, "learning.committed")
        if e.payload["parameter"].startswith("household.")
    ]
    assert learned[-1].payload["parameter"] == "household.verb:wander"


# ---------------------------------------------------------------------------- simulator surface


def test_the_page_gets_posture_walks_falls_and_where_the_duck_is():
    session = _life(2, 60)
    core = session.core
    walk = _walks(core)[0]
    shape = _motion(walk)
    assert set(shape) == {"walk"} and {"vx", "vyaw", "seconds", "from", "to"} <= set(shape["walk"])
    assert tuple(shape["walk"]["from"]) == _pose(walk.payload["bel_from"])
    assert tuple(shape["walk"]["to"]) == _pose(walk.payload["expect"])
    assert shape["walk"]["id"] == walk.event_id
    posture = _kinds(core, "duck.posture")[0]
    assert _motion(posture) == {"posture": posture.payload["posture"]}
    floor = _kinds(core, "env.floor")[0]
    assert _motion(floor) == {"place": (0.0, 0.0, 0.0), "floor": floor.payload["floor"]}
    falls = _core(4, **ALWAYS_FALLS)
    Household(4).live(_hands(falls), 30, "floor")  # type: ignore[arg-type]
    fell = _kinds(falls, "body.fell")[0]
    assert _motion(fell) == {"fall": True}  # (still an interruption for the live bridge)
    # Where it fell and the walk it cut are on its own duck.posture(fallen), just after.
    fallen = next(e for e in _kinds(falls, "duck.posture") if fell.event_id in e.parent_event_ids)
    motion = _motion(fallen)
    assert motion is not None and motion["posture"] == "fallen" and motion["stop"] is True
    assert motion["cut"]["walk"] == fallen.payload["walk_cut"]["walk"]
    assert motion["at"] == fallen.payload["at"]  # the page lays it down where it fell
    righting = _kinds(falls, "body.righting")[0]
    assert _motion(righting) == {"righting": {"try": 1, "ok": righting.payload["ok"]}}
    body = snapshot(session)["body"]
    assert {"posture", "floor", "motion_validated", "position", "avoid_spots"} <= set(body)
    assert body["posture"] in {"sit", "stand", "walk", "fallen"}
    assert body["position"] == core.pose_now()


def test_the_duck_is_always_where_duck_core_says_and_inside_the_room(lives):
    """Every walk starts where the last one ended (or where a hand set it down), inside
    the 6 m × 6 m room less the wall margin; the page draws exactly these poses."""
    turned_from_walls = 0
    for session in lives:
        pose = (0.0, 0.0, 0.0)
        for event in session.core.log.events:
            p = event.payload
            if event.kind == "env.floor":
                pose = tuple(p["pose"])
            elif event.kind == "duck.walk":
                assert _pose(p["bel_from"]) == pose, event.event_id
                for x, y in _path_points(event):
                    assert abs(x) <= ROOM_LIMIT_MM and abs(y) <= ROOM_LIMIT_MM, event.event_id
                pose = _pose(p["expect"])
                turned_from_walls += p.get("avoided") == "wall"
            elif p.get("walk_cut"):
                pose = tuple(p["walk_cut"]["pose"])
        assert tuple(session.core.pose_now()) == pose
    assert turned_from_walls  # it met the walls now and then, and turned away


def test_the_family_can_put_it_on_the_floor_in_words():
    assert [(c.kind, c.params) for c in commands.parse("放在地上，然後放回桌上")] == [
        ("setting", {"key": "floor", "value": True}),
        ("setting", {"key": "floor", "value": False}),
    ]
    (live,) = commands.parse("每天放在地上過 30 天")
    assert live.params == {"days": 30, "routine": "floor"}
    session = Session(Config(seed=3))
    reply = dispatch(session, "command", {"text": "地板模式"})
    assert "地板模式" in reply["results"][0]["result"]["note"]
    assert session.core.s.floor
    why = query.why(session.core, "為什麼不走？")
    assert why["kind"] == "walk" and why["text"]


def test_the_simulator_defaults_to_a_validated_body_and_can_hatch_one_that_is_not():
    state = SimState(Config(seed=3))
    assert state.session.core.hw.motion_validated is True  # the simulator assumes the sweep passed
    rates = state.session.world["rates"]  # the world's rates (phase 4), not the duck's hardware
    assert rates["fall_ppm_per_s"] == 50 and rates["righting_ppm"] == 600_000
    state.reset({"seed": 3, "motion_validated": False, "fall_ppm_per_s": 150_000})
    core = state.session.core
    assert core.hw.motion_validated is False
    assert state.session.world["rates"]["fall_ppm_per_s"] == 150_000
    exported = dispatch(state.session, "export", {})
    assert exported["config"]["motion_validated"] is False
    assert "fall_ppm_per_s" not in exported["config"]
    assert exported["world"]["rates"]["fall_ppm_per_s"] == 150_000
    hatched = _kinds(core, "sim.hatched")[0].payload["hardware"]
    assert hatched["motion_validated"] is False
