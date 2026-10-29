"""The neural duck's experiences as Duck Core's inputs (docs/design/GROWTH-G2-2026-10-10.md):
``bridge-v1``, the attached Duck Core, and that it counts what it is given by its own rules."""

from __future__ import annotations

import copy
import dataclasses
import json
from datetime import timedelta
from typing import Any

import pytest

import duckmind.world.legacy  # noqa: F401  (registers LegacyWorld)
from duckmind.duckcore import api, bridge
from duckmind.duckcore.api import Config, Session, dispatch

START = "2026-03-02T07:00:00+08:00"
SHADOW_KINDS = {
    "duck.expression",
    "duck.walk",
    "duck.decision",
    "duck.nap_start",
    "duck.nap_end",
    "duck.sleep_stir",
    "duck.bid",
    "duck.posture",
    "voice.babble",
    "duck.reaction",
    "duck.greeting",
    "moment.attempt",
    "moment.resolved",
    "rare.event",
    "duck.act",
    "walk.done",
}


def make(seed: int = 4107, *, attach: bool = True, tof: bool = True) -> Session:
    session = Session(Config(seed=seed, name="蛋黃", tof=tof, nest=False, start=START))
    if attach:
        dispatch(session, "bridge", {"api": bridge.BRIDGE_API})
    return session


def now_s(session: Session) -> int:
    return int(session.core.s.now.timestamp())


def give(session: Session, *events: dict[str, Any]) -> dict[str, Any]:
    return dispatch(session, "experience", {"events": list(events)})


def at(session: Session, seconds: int, kind: str, **fields: Any) -> dict[str, Any]:
    return {"k": kind, "t": now_s(session) + seconds, **fields}


def state_of(session: Session) -> dict[str, Any]:
    return dataclasses.asdict(session.core.s)


def later(session: Session, seconds: int, kind: str, **fields: Any) -> dict[str, Any]:
    """Live on ``seconds`` (the clock moves as it does on a server), then give the event."""
    dispatch(session, "advance", {"seconds": seconds})
    return give(session, {"k": kind, "t": now_s(session), **fields})


def kinds_since_attached(session: Session) -> set[str]:
    """The kinds of event written after the seam (the hatch's own first boot is before it)."""
    events = session.core.log.events
    seam = next(i for i, e in enumerate(events) if e.kind == "sim.bridged")
    return {e.kind for e in events[seam + 1 :]}


# ── the vocabulary ───────────────────────────────────────────────────────────────────────────


def test_an_event_is_integers_and_a_few_words_and_nothing_else() -> None:
    ok = bridge.check({"k": "touch", "t": 100, "where": "head", "ms": 3000})
    assert ok["asleep"] == 0 and ok["doing"] == "" and ok["settled"] == 0
    for bad in (
        {"k": "touch", "t": 100.5},
        {"k": "touch", "t": 100, "ms": 1.5},
        {"k": "touch", "t": 100, "ms": True},
        {"k": "touch", "t": 100, "where": "tail"},
        {"k": "touch", "t": 100, "why": "x"},
        {"k": "hand", "t": 100},
        {"k": "hand", "t": 100, "speed": "medium"},
        {"k": "fall", "t": 100, "x_mm": 1, "y_mm": 2},
        {"k": "voiced", "t": 100, "tags": []},
        {"k": "voiced", "t": 100, "tags": ["moo"]},
        {"k": "nap", "t": 100},
        {"k": "touch", "t": -1},
        {"k": "touch", "t": 100, "doing": "tuck_away"},  # (a reaction, not a thing it chose)
    ):
        with pytest.raises(bridge.BridgeError):
            bridge.check(bad)


def test_every_kind_has_a_handler_and_the_vocabulary_is_named_and_frozen() -> None:
    assert bridge.BRIDGE_API == "bridge-v1"
    assert sorted(bridge.KINDS) == sorted(bridge.HANDLERS)
    # the frozen v1 vocabulary: a change is bridge-v2 beside it (saves carry the name)
    assert sorted(bridge.KINDS) == [
        "company",
        "dock",
        "explore",
        "fall",
        "feed",
        "hand",
        "hello",
        "lift",
        "name",
        "noise",
        "play",
        "poke",
        "sleep",
        "touch",
        "visit",
        "voice",
        "voiced",
    ]


def test_an_unattached_duck_takes_nothing_and_a_nest_duck_cannot_be_attached() -> None:
    session = make(attach=False)
    with pytest.raises(bridge.BridgeError):
        give(session, {"k": "lift", "t": now_s(session)})
    nested = Session(Config(seed=3, nest=True, start=START))
    with pytest.raises(bridge.BridgeError, match="not_for_a_nest"):
        dispatch(nested, "bridge", {"api": bridge.BRIDGE_API})
    with pytest.raises(bridge.BridgeError, match="unknown_api"):
        dispatch(make(attach=False), "bridge", {"api": "bridge-v3"})


def test_the_frozen_v4_engine_has_no_bridge() -> None:
    frozen = api.open_session({"seed": 7, "name": "x"})
    with pytest.raises(ValueError, match="no world port"):
        dispatch(frozen, "bridge", {"api": bridge.BRIDGE_API})


def test_an_event_a_minute_ahead_is_refused_with_its_whole_batch() -> None:
    session = make()
    before = len(session.inputs)
    with pytest.raises(bridge.BridgeError, match="ahead"):
        give(session, at(session, 1, "lift"), at(session, 400, "lift"))
    assert len(session.inputs) == before and session.core.s.pickups == 0


# ── attached: no body, no experiences of its own ─────────────────────────────────────────────


def test_attached_for_weeks_it_simulates_no_life_of_its_own_and_still_grows() -> None:
    session = make()
    dispatch(session, "advance", {"seconds": 21 * 86400})
    core = session.core
    kinds = kinds_since_attached(session)
    assert not kinds & SHADOW_KINDS, sorted(kinds & SHADOW_KINDS)
    assert core.s.gd == 21 and core.s.posture == "sit" and core.s.walks == 0
    assert {method for method, _ in session.inputs} == {"bridge", "advance"}
    assert core.s.napping_until is None and core.s.walk_plan is None and core.s.bids_today == 0
    credits = [e.payload["credit"] for e in core.log.events if e.kind == "growth.gd"]
    assert credits == ["full"] * 21  # (an awake duck: the neural duck is on)


def test_the_unattached_twin_does_live_a_shadow_life() -> None:
    """The control for the test above: the same days, not attached."""
    session = make(attach=False)
    dispatch(session, "advance", {"seconds": 3 * 86400})
    kinds = {event.kind for event in session.core.log.events}
    assert {"duck.nap_start", "duck.expression"} <= kinds


def test_living_on_is_the_same_whatever_the_step() -> None:
    whole, parts = make(), make()
    dispatch(whole, "advance", {"seconds": 3 * 86400 + 1234})
    for seconds in [7, 60, 3599, 86400, 86400, 3600, 86400 - 3600 - 3599 - 60 - 7, 1234]:
        dispatch(parts, "advance", {"seconds": seconds})
    assert now_s(whole) == now_s(parts)
    assert state_of(whole) == state_of(parts)
    assert whole.core.log.head == parts.core.log.head


def test_attaching_ends_what_the_shadow_life_was_in_the_middle_of() -> None:
    session = make(attach=False)
    dispatch(session, "set_setting", {"key": "floor", "value": True})
    core = session.core
    for _ in range(600):  # until it is napping or walking
        dispatch(session, "advance", {"seconds": 60})
        if core.s.walk_plan is not None or core.s.napping_until is not None:
            break
    assert core.s.walk_plan is not None or core.s.napping_until is not None
    dispatch(session, "bridge", {"api": bridge.BRIDGE_API})
    s = core.s
    assert s.bridged and s.walk_plan is None and s.napping_until is None and s.posture == "sit"
    dispatch(session, "advance", {"seconds": 3600})
    assert api.replay_check(session)["identical"]


# ── it counts what it is given by its own rules ──────────────────────────────────────────────


def test_a_pat_is_touch_care_a_caress_asleep_is_calm_company() -> None:
    session = make()
    dispatch(
        session, "advance", {"seconds": 12 * 86400}
    )  # a duckling: care counts toward its branch
    core = session.core
    happy = core.s.valence
    give(session, at(session, 1, "touch", where="head", ms=4000))
    assert core.s.care_today["touch"] == 1 and core.s.touches == 1
    assert core.s.valence == happy + 600 and core.s.presence_at is not None
    give(session, at(session, 10, "touch", where="back", ms=4000, asleep=1))
    assert core.s.care_today["calm"] == 1 and core.s.care_today["touch"] == 1


def test_the_caps_are_duck_cores_own() -> None:
    session = make()
    dispatch(session, "advance", {"seconds": 12 * 86400})
    for _ in range(9):
        later(session, 600, "touch", where="head")
    counted = [
        e for e in session.core.log.events if e.kind == "care.episode" and e.payload["counted"]
    ]
    assert len(counted) == 5  # CARE_CAP_PER_GD, the engine's, not the bridge's
    assert session.core.s.care_window["touch"] == 5


def test_a_touch_within_three_seconds_of_what_it_chose_to_do_praises_that() -> None:
    session = make()
    give(session, at(session, 1, "touch", where="head", doing="look_around"))
    assert session.core.s.prefs.get("household.verb:look_around", 5000) > 5000
    other = make()
    give(other, at(other, 1, "touch", where="head"))
    assert not other.core.s.prefs


def test_praising_a_babble_by_touch_promotes_the_phrase_not_the_verb() -> None:
    session = make()
    dispatch(session, "advance", {"seconds": 12 * 86400})
    give(session, at(session, 1, "voiced", tags=["chirp", "coo"]))
    code = session.core.s.last_flag["id"]
    give(session, at(session, 2, "touch", where="head", doing="look_around"))
    assert session.core.s.phrases[code]["praise"] == 1 and session.core.s.phrases[code]["liked_gds"]
    assert "household.verb:look_around" not in session.core.s.prefs


def test_touches_too_fast_are_overload_and_then_no_care() -> None:
    session = make()
    events = [at(session, 1 + 5 * k, "touch", where="head") for k in range(8)]
    give(session, *events)
    care = [e for e in session.core.log.events if e.kind == "care.episode"]
    assert 0 < len(care) < 8  # the engine's burst rule: it tucks away, it does not count that


def test_the_fifth_pickup_is_the_milestone_and_quick_ones_are_overload() -> None:
    session = make()
    dispatch(session, "advance", {"seconds": 3600})
    for k in range(5):
        dispatch(session, "advance", {"seconds": 1800})
        give(session, {"k": "lift", "t": now_s(session)})
    assert session.core.s.pickups == 5 and "likes_pickup" in session.core.s.milestones
    quick = make()
    give(quick, *[at(quick, 1 + 20 * k, "lift") for k in range(6)])
    assert quick.core.s.overload >= 4000 or quick.core.s.pickups == 6


def test_eight_slow_hands_tame_it_and_a_fast_hand_only_startles() -> None:
    session = make()
    for k in range(8):
        dispatch(session, "advance", {"seconds": 600})
        give(session, at(session, 0, "hand", speed="slow"))
    assert session.core.s.slow_hands == 8 and "tamed_hand" in session.core.s.milestones
    fast = make()
    give(fast, at(fast, 1, "hand", speed="fast"))
    assert fast.core.s.slow_hands == 0 and fast.core.s.arousal > 3000


def test_its_name_on_three_days_is_the_milestone_and_the_tv_does_not_count() -> None:
    session = make()
    for day in range(3):
        give(session, at(session, 1, "name", tv=1))
        dispatch(session, "advance", {"seconds": 86400})
    assert not session.core.s.name_gds and "learned_name" not in session.core.s.milestones
    for day in range(3):
        give(session, at(session, 1, "name"))
        dispatch(session, "advance", {"seconds": 86400})
    assert len(session.core.s.name_gds) == 3 and "learned_name" in session.core.s.milestones
    assert session.core.s.name_calls == 6


def test_a_hello_is_the_bonds_step_three_a_day_and_talk_care() -> None:
    session = make()
    dispatch(session, "advance", {"seconds": 12 * 86400})
    bond = session.core.s.bond_owner
    give(session, *[at(session, 1 + 60 * k, "hello") for k in range(6)])
    assert session.core.s.bond_owner > bond
    assert session.core.s.owner_bond_gd == 3  # the engine's cap
    assert session.core.s.owner_seen_today
    asleep = make()
    give(asleep, at(asleep, 1, "hello", asleep=1))
    assert asleep.core.s.bond_owner == 2000 and not asleep.core.s.owner_seen_today


def test_a_bond_with_nobody_fades_toward_neutral_a_day_at_a_time() -> None:
    session = make()
    give(session, *[at(session, 1 + 60 * k, "hello") for k in range(3)])
    dispatch(session, "advance", {"seconds": 86400})
    high = session.core.s.bond_owner
    dispatch(session, "advance", {"seconds": 40 * 86400})
    assert 2000 <= session.core.s.bond_owner < high


def test_three_hard_falls_in_one_spot_are_a_spot_to_avoid_and_easy_ones_are_not() -> None:
    session = make()
    give(
        session,
        *[
            at(session, 10 * (k + 1), "fall", x_mm=1200 + 30 * k, y_mm=300, outcome="hard")
            for k in range(3)
        ],
    )
    assert len(session.core.active_spots()) == 1
    assert session.core.s.avoid_spots["S1"]["x"] // 100 == 12
    easy = make()
    give(
        easy,
        *[at(easy, 10 * (k + 1), "fall", x_mm=1200, y_mm=300, outcome="easy") for k in range(4)],
    )
    assert not easy.core.active_spots()
    far = make()
    give(
        far,
        *[at(far, 10 * (k + 1), "fall", x_mm=2000 * k, y_mm=0, outcome="stuck") for k in range(3)],
    )
    assert not far.core.active_spots()


def test_a_long_nap_by_day_is_the_dreamers_own_experience_the_night_is_not() -> None:
    session = make()
    dispatch(session, "advance", {"seconds": 12 * 86400})
    give(session, at(session, 1, "sleep", minutes=45, night=1))
    assert not session.core.s.self_window
    give(session, at(session, 2, "sleep", minutes=45))
    assert session.core.s.self_window == {"dreamer": 1}


def test_leaning_into_a_hand_and_somewhere_new_are_its_own_experience() -> None:
    session = make()
    dispatch(session, "advance", {"seconds": 12 * 86400})
    give(session, at(session, 1, "touch", where="head", settled=1))
    later(session, 700, "explore")
    assert session.core.s.self_window == {"snuggler": 1, "explorer": 1}


def test_play_is_care_once_for_every_few_minutes() -> None:
    session = make()
    dispatch(session, "advance", {"seconds": 12 * 86400})
    give(session, at(session, 1, "play", minutes=9))
    assert session.core.s.care_today["play"] == 3
    assert session.core.s.last_play_at is not None


def test_company_is_play_when_out_and_about_and_calm_after_twenty_quiet_minutes() -> None:
    session = make()
    dispatch(session, "advance", {"seconds": 12 * 86400})
    give(session, at(session, 1, "company", moving=1))
    assert session.core.s.care_today["play"] == 1
    session = make()
    dispatch(session, "advance", {"seconds": 12 * 86400})
    # a person here, resting, for twenty-one minutes of minutes
    give(session, *[at(session, 60 * (k + 1), "company") for k in range(21)])
    assert session.core.s.care_today["calm"] == 1
    # (attended in between: the count starts again)
    attended = make()
    dispatch(attended, "advance", {"seconds": 12 * 86400})
    events = [at(attended, 60 * (k + 1), "company") for k in range(15)]
    events.append(at(attended, 60 * 16, "touch", where="head"))
    events += [at(attended, 60 * (17 + k), "company") for k in range(15)]
    give(attended, *events)
    assert attended.core.s.care_today["calm"] == 0


def test_a_poke_has_no_counterpart_so_it_is_only_overload_never_care() -> None:
    session = make()
    dispatch(session, "advance", {"seconds": 12 * 86400})
    give(session, *[at(session, 1 + 5 * k, "poke") for k in range(3)])
    assert session.core.s.overload > 2000 and session.core.s.touches == 0
    assert not [e for e in session.core.log.events if e.kind == "care.episode"]


def test_a_call_it_made_that_is_answered_counts_toward_its_first_word() -> None:
    session = make()
    dispatch(session, "advance", {"seconds": 4 * 86400})  # growth day 4: words are being learnt
    give(session, at(session, 1, "voiced", tags=["chirp"]), at(session, 2, "voice"))
    assert session.core.s.word_answers == {"chirp": 1}
    later(session, 100, "voiced", tags=["chirp", "coo"])
    phrases = [p for p in session.core.s.phrases.values() if p["tags"] == ["chirp", "coo"]]
    assert phrases and session.core.s.last_flag["kind"] == "phrase"


def test_it_counts_nothing_while_it_is_switched_off() -> None:
    session = make()
    dispatch(session, "act", {"action": "power_off", "params": {}})
    taken = give(session, at(session, 1, "lift"))
    assert taken["taken"][0]["taken"] is False and session.core.s.pickups == 0


# ── parity with the family's own hands ───────────────────────────────────────────────────────

PERCEPT = {
    "kind": "sense.touch",
    "src": "live",
    "pv": "p:00000000",
    "where": "crown",
    "seconds": 4,
    "firmness": "normal",
    "path": "joint",
    "latency_ms": 0,
}
GROWTH = (
    "care_today",
    "care_window",
    "touches",
    "pickups",
    "slow_hands",
    "name_calls",
    "name_gds",
    "bond_owner",
    "owner_bond_gd",
    "day_voice",
    "day_noise",
    "owner_seen_today",
)


def growth_of(session: Session) -> dict[str, Any]:
    s = session.core.s
    out = {name: copy.deepcopy(getattr(s, name)) for name in GROWTH}
    out["milestones"] = sorted(s.milestones)
    return out


def test_what_the_family_hands_do_to_growth_is_what_the_bridge_does_to_growth() -> None:
    """The same sequence of a family's acts on an unattached duck (its own handlers) and of the
    experiences on an attached one: they learn the same."""
    hands = make(attach=False)
    fed = make()
    for session in (hands, fed):
        dispatch(session, "advance", {"seconds": 11 * 86400 + 3600})
    script = []
    for day in range(4):
        script += [
            ("touch", {}),
            ("hello", {}),
            ("name", {}),
            ("voice", {}),
            ("noise", {}),
            ("hand", {"speed": "slow"}),
            ("lift", {}),
            ("touch", {}),
        ]
    for index, (kind, fields) in enumerate(script):
        for session in (hands, fed):
            dispatch(session, "advance", {"seconds": 900})
        core = hands.core
        t_ms = core.now_ms()
        if kind in ("touch", "hello", "name"):
            # (the unattached twin has a shadow life and may be dozing: the neural duck's
            # sleeping is what the event says)
            fields = {**fields, "asleep": int(core._until(core.s.napping_until))}
        if kind == "touch":
            core.on_percept({**PERCEPT, "t_ms": t_ms})
        elif kind == "hello":
            core.owner_hello()
        elif kind == "name":
            core.call_name("owner")
        elif kind == "voice":
            core.say("owner", kind="voice")
        elif kind == "noise":
            core.say("owner", kind="noise")
        elif kind == "hand":
            core.hand_near("owner", speed=fields["speed"])
        elif kind == "lift":
            core.pickup("owner")
        give(fed, {"k": kind, "t": now_s(fed), **fields})
        if index % 8 == 7:
            for session in (hands, fed):
                dispatch(session, "advance", {"seconds": 86400 - 8 * 900})
    assert growth_of(fed) == growth_of(hands)
    assert growth_of(fed)["pickups"] == 4 and growth_of(fed)["slow_hands"] == 4


# ── replay, bit-identical ────────────────────────────────────────────────────────────────────


def busy_life() -> Session:
    session = make()
    dispatch(session, "advance", {"seconds": 12 * 86400})
    for day in range(6):
        events = [
            at(session, 5, "hello"),
            at(session, 9, "touch", where="head", ms=4000, doing="look_around"),
            at(session, 12, "name"),
            at(session, 20, "hand", speed="slow"),
            at(session, 40, "voiced", tags=["chirp", "coo"]),
            at(session, 41, "voice"),
            at(session, 60, "sleep", minutes=30),
            at(session, 80, "fall", x_mm=1000, y_mm=500, outcome="hard"),
            at(session, 90, "dock", minutes=12),
            at(session, 100, "explore"),
            at(session, 110, "lift"),
        ]
        give(session, *events)
        dispatch(session, "advance", {"seconds": 86400})
    return session


def test_a_bridged_save_replays_to_the_same_core_state_bit_identical() -> None:
    session = busy_life()
    assert api.replay_check(session)["identical"]
    assert api.percept_replay_check(session)["identical"]
    saved = json.loads(json.dumps(api.save_json(session)))
    assert saved["bridge"] == {"api": "bridge-v1"}
    again = api.restore(saved)
    assert state_of(again) == state_of(session)
    assert again.core.log.head == session.core.log.head
    assert again.core.counters == session.core.counters
    assert [e.event_id for e in again.core.log.events] == [
        e.event_id for e in session.core.log.events
    ]


def test_a_save_from_a_newer_vocabulary_is_refused_not_misread() -> None:
    saved = api.save_json(busy_life())
    saved["bridge"] = {"api": "bridge-v3"}
    with pytest.raises(api.SaveBridgeMismatch):
        api.restore(saved)


def test_an_older_save_replays_as_it_always_did_and_then_attaches_where_it_stopped() -> None:
    """The live 蛋黃's upgrade: its record (a shadow life, no bridge) replays exactly as before,
    one `bridge` input marks the seam, and new inputs append."""
    old = make(attach=False)
    dispatch(old, "advance", {"seconds": 5 * 86400})
    dispatch(old, "act", {"action": "owner_hello", "params": {}})
    dispatch(old, "advance", {"seconds": 3 * 86400})
    saved_old = json.loads(json.dumps(api.save_json(old)))
    assert "bridge" not in saved_old
    head_before = old.core.log.head
    replayed = api.restore(copy.deepcopy(saved_old))
    assert replayed.core.log.head == head_before and state_of(replayed) == state_of(old)

    upgraded = api.restore(copy.deepcopy(saved_old))
    dispatch(upgraded, "bridge", {"api": bridge.BRIDGE_API})
    give(upgraded, at(upgraded, 5, "touch", where="head"))
    dispatch(upgraded, "advance", {"seconds": 2 * 86400})
    saved_new = json.loads(json.dumps(api.save_json(upgraded)))
    # the old inputs are the new save's first inputs, untouched
    assert saved_new["inputs"][: len(saved_old["inputs"])] == saved_old["inputs"]
    assert [i["method"] for i in saved_new["inputs"][len(saved_old["inputs"]) :]] == [
        "bridge",
        "experience",
        "advance",
    ]
    again = api.restore(saved_new)
    assert state_of(again) == state_of(upgraded) and again.core.log.head == upgraded.core.log.head
    # the old events are the same events in the same chain
    old_ids = [e.event_id for e in old.core.log.events]
    assert [e.event_id for e in again.core.log.events][: len(old_ids)] == old_ids
    assert api.replay_check(again)["identical"]


def test_the_time_a_bridged_duck_is_switched_off_is_the_engines_own_rule() -> None:
    session = make()
    dispatch(session, "advance", {"seconds": 86400})
    dispatch(session, "act", {"action": "power_off", "params": {}})
    dispatch(session, "advance", {"seconds": 3 * 86400})
    dispatch(session, "act", {"action": "power_on", "params": {}})
    credits = [e.payload["credit"] for e in session.core.log.events if e.kind == "growth.gd"]
    assert credits == ["full", "full", "half", "half"]  # (day 2 was awake until 07:00)
    assert session.core.s.bridged and session.core.s.awake


def test_events_are_taken_at_their_second_whatever_the_batch() -> None:
    one, two = make(), make()
    base = now_s(one)
    events = [
        {"k": "hello", "t": base + 5},
        {"k": "name", "t": base + 9},
        {"k": "lift", "t": base + 30},
    ]
    give(one, *events)
    for event in events:
        give(two, event)
    assert state_of(one) == state_of(two)
    assert one.core.s.now - two.core.s.now == timedelta(0)


def test_any_mix_of_experiences_replays_to_the_same_duck_and_garbage_is_refused_cleanly() -> None:
    import random

    rng = random.Random(7)
    session = make()
    dispatch(session, "advance", {"seconds": 11 * 86400})
    kinds = sorted(bridge.KINDS)
    for step in range(400):
        kind = rng.choice(kinds)
        event: dict[str, Any] = {"k": kind, "t": now_s(session) + rng.randrange(0, 20)}
        for name, (_, default) in bridge.KINDS[kind].items():
            if kind == "touch" and name == "doing":
                event[name] = rng.choice(["", "look_around", "stretch_yawn", "preen"])
            elif name in ("where",):
                event[name] = rng.choice(bridge.WHERE)
            elif name in ("asleep", "settled", "tv", "moving", "night"):
                event[name] = rng.randrange(2)
            elif name == "speed":
                event[name] = rng.choice(bridge.HAND_SPEEDS)
            elif name == "what":
                event[name] = rng.choice(bridge.NOISES)
            elif name == "outcome":
                event[name] = rng.choice(bridge.FALL_OUTCOMES)
            elif name == "tags":
                event[name] = rng.sample(["chirp", "coo", "inquire", "greet"], rng.randrange(1, 4))
            elif name in ("x_mm", "y_mm"):
                event[name] = rng.randrange(1000, 1400)
            elif name == "minutes":
                event[name] = rng.randrange(1, 90)
            elif name == "ms":
                event[name] = rng.randrange(0, 5000)
        give(session, event)
        if step % 7 == 0:
            dispatch(session, "advance", {"seconds": rng.randrange(1, 40000)})
    assert api.replay_check(session)["identical"]
    assert api.percept_replay_check(session)["identical"]
    again = api.restore(json.loads(json.dumps(api.save_json(session))))
    assert state_of(again) == state_of(session)
    # garbage is a BridgeError, never anything else, and it leaves the duck as it was
    before = state_of(session)
    inputs = len(session.inputs)
    for bad in (
        None,
        5,
        "touch",
        [],
        {},
        {"k": 3},
        {"k": "touch", "t": "x"},
        {"k": "fall", "t": 1, "x_mm": 1.5, "y_mm": 2, "outcome": "easy"},
    ):
        with pytest.raises(bridge.BridgeError):
            dispatch(session, "experience", {"events": [bad]})
    with pytest.raises(bridge.BridgeError):
        dispatch(session, "experience", {"events": "touch"})
    assert state_of(session) == before and len(session.inputs) == inputs
