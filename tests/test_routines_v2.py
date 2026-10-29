"""Household v2 (spec chapter 10): simulated families that live real days.

Each family is drawn from its seed (its own random stream), has its members' days, its
weekends and trips, and its own way of caring; the duck only perceives it through cues and
what they do. ``truth`` is the simulator's ground truth, never the duck's memory.
"""

from __future__ import annotations

import collections
from datetime import datetime

import pytest

import duckmind.world.legacy  # noqa: F401  (registers LegacyWorld: a v4_1 duck lives in it)
from duckmind.duckcore.api import Config, Session, dispatch, family_profile
from duckmind.duckcore.routines import CARE_STYLES, FAMILY_KINDS, make_profile


def test_families_are_drawn_from_the_mix_and_are_the_same_every_time():
    kinds = collections.Counter(make_profile(seed).kind for seed in range(600))
    assert set(kinds) == set(FAMILY_KINDS)
    for kind, (weight, _) in FAMILY_KINDS.items():
        assert abs(kinds[kind] / 600 - weight / 10000) < 0.06, kinds
    assert make_profile(42) == make_profile(42)
    school = make_profile(7, "school_family")
    assert {m.role for m in school.members} >= {"school_child"}
    child = next(m for m in school.members if m.persona == "child")
    assert child.care[2] >= 25  # a child plays
    low = make_profile(7, "low_attention")
    assert all(m.praise < 500 and m.pickups_per_day == 0 for m in low.members)
    cable = make_profile(3, "cable_only")
    assert cable.cable and not cable.nest
    assert sum(CARE_STYLES["elder"]) == 100


@pytest.mark.parametrize("kind", ["dual_income", "elder_home"])
def test_the_duck_believes_what_the_house_sounds_like(kind):
    session = Session(Config(seed=5, nest=True, household=kind))
    dispatch(session, "live", {"family": kind, "days": 10})
    core, family = session.core, session.family
    assert family is not None and family.truth
    # Compare, minute by minute from the truth's change points, where nobody was home
    # (for an hour or more) with what the duck believed 30 minutes into it.
    beliefs = [(e.occurred_at, e.payload["to"]) for e in core.log.events if e.kind == "env.family"]

    def believed(at: datetime) -> str:
        state = "awake"
        for moment, to in beliefs:
            if moment > at:
                break
            state = to
        return state

    rows = [(datetime.fromisoformat(at), home, asleep) for at, home, _, asleep in family.truth]
    empties = 0
    right = 0
    for (at, home, asleep), (nxt, _, _) in zip(rows, rows[1:], strict=False):
        if home == 0 and asleep == 0 and (nxt - at).total_seconds() >= 3600:
            empties += 1
            probe = at.replace(second=0) + (nxt - at) / 2
            right += believed(probe) in ("away", "unknown")
    if kind == "dual_income":
        assert empties >= 5
    if empties:
        assert right >= empties * 8 // 10, (right, empties)


def test_a_family_answers_praises_and_puts_the_duck_back():
    session = Session(Config(seed=6, nest=True, household="school_family"))
    dispatch(session, "live", {"family": "school_family", "days": 20})
    core, family = session.core, session.family
    assert family is not None
    kinds = collections.Counter(e.kind for e in core.log.events)
    assert kinds["pad.good"] + kinds["owner.good"] > 0
    assert kinds["sense.touch"] > 50 and kinds["sense.pickup"] > 0
    assert kinds["duck.bid_answered"] > 0
    assert kinds["sense.placed"] > 0  # back into its nest by hand now and then
    assert family.stats.get("answered", 0) > 0


def test_the_family_profile_is_part_of_the_session_and_replays():
    session = Session(Config(seed=9, nest=True, household="hybrid"))
    profile = family_profile(session, "hybrid")
    assert profile.kind == "hybrid" and profile.as_dict()["members"]
    dispatch(session, "live", {"family": "hybrid", "days": 5})
    dispatch(session, "live", {"family": "hybrid", "days": 2})
    assert dispatch(session, "replay_check", {})["identical"] is True
    with pytest.raises(ValueError):
        dispatch(session, "live", {"family": "martians", "days": 1})


# ---------------------------------------------------------------------------- the old routines with a nest


@pytest.mark.parametrize(
    "routine", ["daily", "three_per_week", "weekends", "floor", "docked", "neglect"]
)
def test_the_old_routines_keep_a_nest_duck_charged_and_growing(routine):
    """Page audit: the old routines switched a nest duck off at night off its nest and ran
    its battery flat (daily × 30 days: stuck at growth day 8, off at 0%). With a nest the
    family now puts it back in its nest instead: it charges there and lives its own day."""
    session = Session(Config(seed=7, nest=True))
    result = dispatch(session, "live", {"days": 14, "routine": routine})
    core = session.core
    assert result["gd_to"] >= 13, result
    flat = [e for e in core.log.events if e.kind == "power.off"]
    assert not flat, [(e.occurred_at, e.payload) for e in flat][:3]
    assert core.s.awake and not core.s.docked
    assert dispatch(session, "replay_check", {})["identical"] is True


def test_without_a_nest_the_old_routines_are_as_before():
    session = Session(Config(seed=7))
    result = dispatch(session, "live", {"days": 14, "routine": "daily"})
    offs = [e for e in session.core.log.events if e.kind == "power.off"]
    assert result["gd_to"] >= 13 and offs and not session.core.s.awake  # switched off at night


def test_a_birthday_party_never_switches_a_nest_duck_on_and_off():
    """Page audit: a 2-hour party ran a fresh nest duck flat, then power.off came six times
    in six minutes. The family now sets it back in its nest to charge when it runs low (or
    when nobody put it down), and a switched-off nest duck is only switched on there."""
    session = Session(Config(seed=7, nest=True))
    result = dispatch(session, "party", {"guest_mode": False})
    core = session.core
    assert not [e for e in core.log.events if e.kind == "power.off"]
    assert core.s.awake and not core.s.unplaced and result["touches"] > 0

    flat = Session(Config(seed=7, nest=True))
    dispatch(flat, "advance", {"until": "10:00"})
    dispatch(flat, "act", {"action": "pickup", "persona": "owner"})
    dispatch(flat, "advance", {"minutes": 90})
    assert not flat.core.s.awake
    before = len(flat.core.log.events)
    dispatch(flat, "party", {"guest_mode": True})
    during = flat.core.log.events[before:]
    assert not [e for e in during if e.kind == "power.off"]
    assert any(e.kind == "power.on" for e in during) and flat.core.s.awake
    assert dispatch(flat, "replay_check", {})["identical"] is True


# ---------------------------------------------------------------------------- phase 4: the family's hands


def test_a_familys_hands_reach_the_duck_through_the_world() -> None:
    """Phase 4 (design §10, G67/G68): the family still calls the duck's own method names, but
    whether a touch or a hand on its head was *detected* is the world's draw now (LegacyWorld,
    the duck's own key); Duck Core learns of a detected one only as a percept, and a missed one
    never reaches it. Every draw is one hand on an awake duck: detected + missed."""
    session = Session(Config(seed=6, nest=True, household="school_family"))
    dispatch(session, "live", {"family": "school_family", "days": 6})
    core = session.core
    counters = session.port.state()["counters"]  # type: ignore[attr-defined]
    assert not {"detect.touch", "detect.hold", "false_alarm.tv"} & set(core.counters)
    felt = [
        e
        for e in core.log.events
        if e.kind == "sense.touch" and not e.payload.get("legacy_phantom")
    ]
    missed = [j for j in core.journal if j["action"] == "touch" and "偵測器漏掉" in j["detail"]]
    assert felt and missed and counters["detect.touch"] == len(felt) + len(missed)
    assert all(e.payload["pv"].startswith("l:") and e.source.kind == "legacy-world" for e in felt)
    for check in ("replay_check", "percept_replay_check", "world_replay_check"):
        assert dispatch(session, check, {})["identical"] is True, check
