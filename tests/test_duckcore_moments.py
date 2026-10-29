"""Clever and goofy moments (Duck Core v4, part F, phase P1).

Clever moments are real skills: it guesses first and the moment counts only when the
guess came true, with both events as evidence. Goofy moments are real bird behaviour and
happen only when it is relaxed, always with the comic beat. A wrong guess turns into the
matching goofy moment; a moment the family reacts to on three growth days becomes a
running gag; every walk a moment takes is an arc or a step forward, never a turn on the
spot, and goes through the same gates as any other walk.
"""

from __future__ import annotations

import collections
from datetime import timedelta
from functools import cache

from duckmind.duckcore import content, moments
from duckmind.duckcore.api import Config, Session, dispatch, snapshot
from duckmind.duckcore.engine import DuckCore
from duckmind.duckcore.fixed import SCALE
from duckmind.provenance import Event


def _kinds(core: DuckCore, kind: str) -> list[Event]:
    return [event for event in core.log.events if event.kind == kind]


@cache
def _life(seed: int, kind: str = "school_family", days: int = 45) -> Session:
    session = Session(Config(seed=seed, nest=True, conservative=False, household=kind))
    dispatch(session, "live", {"family": kind, "days": days})
    return session


def test_the_catalogue_is_what_the_spec_lists():
    kinds = collections.Counter(m.kind for m in content.MOMENTS)
    assert kinds == {"clever": 12, "goofy": 16}
    for moment in content.MOMENTS:
        assert moment.care in content.CARE_TYPES and moment.zh.endswith("。")
        if moment.fail_to is not None:
            assert moment.kind == "clever" and content.MOMENT_BY_ID[moment.fail_to].kind == "goofy"
    assert not content.MOMENT_BY_ID["sits_beside_nest"].promotable
    for clip in ("dabble", "double_take", "proud_puff", "still"):
        assert clip in content.CLIPS and clip in content.VERB_BY_ID
    # Every hardware setting keeps at least eight of each (spec 8.6).
    needs_hw = {"ear", "tof", "name"}
    bare = [m for m in content.MOMENTS if m.needs not in needs_hw]
    assert sum(m.kind == "clever" for m in bare) >= 8 and sum(m.kind == "goofy" for m in bare) >= 8


def test_a_family_life_has_both_kinds_of_moment_and_many_different_ones():
    core = _life(3).core
    found = _kinds(core, "duck.moment")
    by_kind = collections.Counter(e.payload["kind"] for e in found)
    assert by_kind["clever"] >= 20 and by_kind["goofy"] >= 20, by_kind
    assert len({e.payload["id"] for e in found}) >= 12
    per_day = collections.Counter((e.occurred_at.date(), e.payload["kind"]) for e in found)
    assert max(per_day.values()) <= 8
    last: dict[str, object] = {}
    for event in found:  # the same moment at most every 45 minutes
        previous = last.get(event.payload["id"])
        if previous is not None:
            assert event.occurred_at - previous >= timedelta(minutes=45)  # type: ignore[operator]
        last[event.payload["id"]] = event.occurred_at


def test_every_clever_moment_has_its_guess_and_the_outcome_that_proved_it():
    core = _life(3).core
    events = {e.event_id: e for e in core.log.events}
    clever = [e for e in _kinds(core, "duck.moment") if e.payload["kind"] == "clever"]
    assert clever
    for moment in clever:
        guess = events[moment.payload["predict_event"]]
        outcome = events[moment.payload["outcome_event"]]
        assert guess.kind == "duck.predict" and outcome.kind == "duck.predict_outcome"
        assert outcome.payload["hit"] is True
        assert guess.occurred_at <= outcome.occurred_at <= moment.occurred_at
        assert guess.event_id in outcome.parent_event_ids
        assert moment.payload["outcome"] == "hit"
    # Each guess is settled one way or the other, and never pretends.
    guesses = [e for e in _kinds(core, "duck.predict") if e.payload.get("rule") == "moment-v1"]
    settled = {
        parent
        for e in _kinds(core, "duck.predict_outcome")
        if e.payload.get("rule") == "moment-v1"
        for parent in e.parent_event_ids
    }
    open_now = {state["predict"] for state in core.s.moment_attempts.values()}
    assert all(g.event_id in settled | open_now for g in guesses)


def test_goofy_moments_happen_only_when_it_is_relaxed_and_play_the_comic_beat():
    core = _life(3).core
    goofy = [e for e in _kinds(core, "duck.moment") if e.payload["kind"] == "goofy"]
    assert goofy
    children: dict[str, list[Event]] = collections.defaultdict(list)
    for event in core.log.events:
        for parent in event.parent_event_ids:
            children[parent].append(event)
    for moment in goofy:
        field = moment.payload["field"]
        assert field["ok"] is True and field["phase"] in ("family", "solo", "dawn", "alone")
        assert field["battery"] >= field["battery_min"] >= 5000 and field["energy"] >= 4000
        assert not (moment.occurred_at.hour >= 22 or moment.occurred_at.hour < 7)
        beats = [
            e.payload["verb"]
            for e in children[moment.event_id]
            if e.kind == "duck.expression" and e.payload["reason"] == "moment"
        ]
        assert "still" in beats, (moment.payload["id"], beats)  # the freeze
        still = next(
            e
            for e in children[moment.event_id]
            if e.kind == "duck.expression" and e.payload["verb"] == "still"
        )
        ms = 1000 * SCALE // int(still.payload["tempo"])
        assert 600 <= ms <= 1400 or moment.payload["id"] in ("door_nap", "nest_circle")


def test_moments_walk_only_arcs_and_steps_forward_through_every_gate():
    walks = [
        walk
        for seed in (3, 5)
        for walk in _kinds(_life(seed).core, "duck.walk")
        if walk.payload.get("moment")
    ]
    assert {walk.payload["moment"] for walk in walks} <= {"chases_tail", "wrong_turn"}
    for walk in walks:
        assert walk.payload["vx"] > 0  # a tight arc or forward, never on the spot
        assert not walk.payload["nested"] and walk.payload["battery"] >= 3000
        assert not (walk.occurred_at.hour >= 22 or walk.occurred_at.hour < 7)
        assert walk.payload["dist_to_mm"] <= (walk.payload["leash_mm"] or 10**6) + 300


def _relaxed_duck(seed: int = 8) -> DuckCore:
    """A nest duck well past growth day 10, off its nest in the evening, family around."""
    core = _life(seed, "elder_home", 25).core
    core.advance(core.minutes_until("18:00"))
    for _ in range(3):
        core.hear_home(None, kind="person", journal=False)
        core.advance(1)
    core.s.battery = SCALE
    core.s.energy = SCALE
    core.s.homing = core.s.stranded = core.s.need = None  # (whatever this life was doing)
    core.s.overload = 0
    assert moments.relaxed(core, "idle")[0]
    return core


def test_a_wrong_guess_becomes_the_goofy_moment_when_it_is_relaxed():
    core = _relaxed_duck()
    if core.s.moment_attempts:
        moments.forget_guesses(core, "test")
    info = {"wits": 5000, "form": "ordinary", "draw": 0, "counter": 0, "audience": True}
    moment = content.MOMENT_BY_ID["comes_when_called"]
    moments.guess(core, moment, "name", info, {"expect": "family"})
    core.s.rare_today = {}  # (today's caps and the 45-minute gap are not the point here)
    core.s.moment_last.pop("wrong_way", None)
    assert moments.resolve(core, "comes_when_called", False) == "wrong_way"
    wrong = _kinds(core, "duck.moment")[-1].payload
    assert wrong["id"] == "wrong_way" and wrong["outcome"] == "missed_guess"
    assert wrong["from_guess"] == "comes_when_called" and wrong["field"]["ok"]
    # Not relaxed (the battery near its line): the wrong guess is just a wrong guess.
    core.s.battery = 4000
    moments.guess(core, moment, "name", info, {"expect": "family"})
    assert moments.resolve(core, "comes_when_called", False) is None
    assert _kinds(core, "duck.moment")[-1].payload["id"] == "wrong_way"  # nothing new


def test_the_family_s_reaction_teaches_it_and_three_days_make_a_running_gag():
    core = _relaxed_duck(9)
    core.s.gags, core.s.gag_gd, core.s.gag_days = [], None, {}  # (its own gags so far aside)
    moment = content.MOMENT_BY_ID["vacuum_dabble"]
    info = {"wits": 4000, "form": "ordinary", "field": {"ok": True}, "audience": True}
    before = core.s.prefs.get("household.moment:vacuum_dabble", 5000)
    for day in range(3):
        core.s.gd2 += 2
        moments.record(core, moment, "pause", "done", info)
        core.say("child", kind="voice")  # a laugh and a word, within ten seconds
    reactions = _kinds(core, "moment.reaction")
    assert len(reactions) >= 3 and reactions[-1].payload["id"] == "vacuum_dabble"
    assert core.s.prefs["household.moment:vacuum_dabble"] > before
    assert "vacuum_dabble" in core.s.gags
    assert _kinds(core, "moment.gag_promoted")[-1].payload["id"] == "vacuum_dabble"
    # A moment that frames a real miss never becomes a gag.
    miss = content.MOMENT_BY_ID["sits_beside_nest"]
    for day in range(4):
        core.s.gd2 += 16
        moments.record(core, miss, "dock_miss", "done", info)
        core.say("child", kind="voice")
    assert "sits_beside_nest" not in core.s.gags
    # A signature gag is rare: at most three of them.
    for moment_id in ("zoomies", "patrol", "chases_tail", "sings_to_wall"):
        for day in range(3):
            core.s.gd2 += 16
            moments.record(core, content.MOMENT_BY_ID[moment_id], "pause", "done", info)
            core.say("child", kind="voice")
    assert len(core.s.gags) == 3


def test_not_that_makes_a_moment_a_little_rarer():
    core = _relaxed_duck(10)
    before = core.s.prefs.get("household.moment:stare_at_nothing", 5000)
    moments.record(
        core, content.MOMENT_BY_ID["stare_at_nothing"], "idle", "done", {"field": {"ok": True}}
    )
    core.owner_not_that()
    assert core.s.prefs["household.moment:stare_at_nothing"] < before


def test_its_wits_stay_between_15_and_85_percent_and_the_day_has_one_form():
    core = _relaxed_duck(11)
    for arousal in (0, 2500, 5000, 7500, 10000):
        for energy in (1000, 9000):
            core.s.arousal, core.s.energy = arousal, energy
            assert 1500 <= moments.wits(core) <= 8500
    days = collections.Counter(e.payload["date"] for e in _kinds(core, "duck.day_form"))
    assert days and max(days.values()) == 1
    forms = collections.Counter(moments._form_of(core, f"2027-01-{d:02d}") for d in range(1, 29))
    assert forms["ordinary"] > forms["sharp"] and forms["ordinary"] > forms["muddled"]


def test_the_page_can_look_without_changing_anything():
    session = _life(4, "dual_income", 15)
    before = len(session.core.log.events)
    for _ in range(3):
        state = snapshot(session)
    assert len(session.core.log.events) == before
    assert state["moments"]["wits"] is None or 15 <= state["moments"]["wits"] <= 85


def test_a_life_with_moments_replays_exactly():
    session = Session(Config(seed=21, nest=True, conservative=False, household="elder_home"))
    dispatch(session, "live", {"family": "elder_home", "days": 30})
    assert _kinds(session.core, "duck.moment")
    assert dispatch(session, "replay_check", {})["identical"]
