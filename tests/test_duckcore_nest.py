"""The charging nest (Duck Core v4, part B), in the simulator only (owner decision 2026-09-26).

It charges only on the nest's contacts and never walks while on them; it keeps the battery
for the way home; it finds its way back, walks in and sits down the way it came (the open
nest, owner decision D1: no turning round on the spot); a miss means a forward loop and
trying again, three misses mean asking for a hand; getting up can fail; a fall near the nest never
teaches it to fear its home; lifted up, it waits to be put back somewhere.

Phase 4: whether a try at the contacts touched them, and whether it saw the nest's tag, are
the world's answers (LegacyWorld: v4's own dice, moved out) to its ``duck.dock`` and
``duck.look`` acts; a flat battery is its own shutdown.
"""

from __future__ import annotations

import dataclasses
import math
from datetime import timedelta
from typing import Any

from duckmind.duckcore import content, nest
from duckmind.duckcore.api import Config, LegacyActs, Session, dispatch, legacy_world, snapshot
from duckmind.duckcore.engine import MIN_MOTION_BATTERY, DuckCore, HardwareConfig
from duckmind.duckcore.fixed import SCALE
from duckmind.provenance import Event


def _duck(
    seed: int = 3, days: int = 8, rates: dict[str, int] | None = None, **config: object
) -> DuckCore:
    """A nest duck old enough to walk, at 09:00 with the family up and around (``rates``:
    its world's outcome rates, where v4 had them in its Config)."""
    settings = Config(seed=seed, nest=True, conservative=False, **config)  # type: ignore[arg-type]
    core = Session(settings, world=legacy_world(settings, rates)).core
    core.advance(days * 1440)
    core.advance(core.minutes_until("09:00"))
    _family(core, 3)
    return core


def _family(core: DuckCore, minutes: int) -> None:
    for minute in range(minutes):
        if minute % 3 == 0:
            core.hear_home(None, kind="voice", journal=False)
            core.hear_home(None, kind="person", journal=False)
        core.advance(1)


def _kinds(core: DuckCore, kind: str) -> list[Event]:
    return [event for event in core.log.events if event.kind == kind]


def _world(core: DuckCore) -> Any:
    """The LegacyWorld the duck lives in (a test turns its rates, as v4's tests turned the
    duck's own hardware rates)."""
    return core.acts.port


def _hands(core: DuckCore) -> LegacyActs:
    """The family's hands on it (a touch or a hand on its head goes through the world's
    detectors first)."""
    return LegacyActs(core, core.acts.port)


def _parent(core: DuckCore, event: Event) -> Event:
    found = core.log.get(event.parent_event_ids[0])
    assert found is not None
    return found


def _pose(value: Any) -> tuple[float, float, float]:
    """A walk's ``bel_from`` / ``expect`` (mm, mrad) as metres and radians."""
    return value["x_mm"] / 1000, value["y_mm"] / 1000, value["yaw_mrad"] / 1000


def test_the_real_robot_has_no_nest_but_the_simulator_page_hatches_one():
    assert HardwareConfig().nest is False  # off until the nest hardware exists
    legacy = Session(Config(seed=1)).core
    assert not legacy.hw.nest and not legacy.aware and not legacy.s.nested
    duck = Session(Config(seed=1, nest=True)).core
    assert duck.s.nested and duck.s.floor and duck.aware
    hatched = _kinds(duck, "sim.hatched")[0].payload["hardware"]
    assert hatched["nest"] and hatched["nest_charge_per_min"] == 125


def test_on_the_nest_it_charges_and_never_walks():
    core = _duck(4)
    _family(core, 12)
    if not core.s.nested:
        core.place_on_nest("owner")
    core.s.battery = 5000
    core.s.energy = 2000  # tired: it naps in the nest (and charges faster asleep)
    _family(core, 1)
    assert core.s.nested and core.s.napping_until is not None
    before = core.s.battery
    core.advance(10)
    assert core.s.battery == min(SCALE, before + 10 * core.hw.nest_charge_per_min)
    _family(core, 6 * 60)
    for walk in _kinds(core, "duck.walk"):
        assert not walk.payload["nested"] and not walk.payload["docked"]
        assert walk.payload["battery"] >= MIN_MOTION_BATTERY
    undocked = _kinds(core, "nest.undocked")
    assert undocked and _kinds(core, "nest.docked")  # out and back again, many times
    assert len(_kinds(core, "nest.docked")) >= 5


def test_it_always_starts_home_with_the_battery_for_the_way_home():
    session = Session(Config(seed=5, nest=True, conservative=False, household="elder_home"))
    dispatch(session, "live", {"family": "elder_home", "days": 20})
    starts = _kinds(session.core, "nav.home_start")
    assert len(starts) >= 100
    for start in starts:
        assert start.payload["battery"] >= start.payload["threshold"], start.payload
    threshold = nest.threshold(session.core)
    assert MIN_MOTION_BATTERY + 500 <= threshold <= MIN_MOTION_BATTERY + 1500


def test_three_misses_at_the_nest_mean_sitting_beside_it_and_asking_for_a_hand():
    core = _duck(6)
    _world(core).dock_ppm = 0
    since = core.s.now  # tries before this could still dock
    for _ in range(6 * 60):
        _family(core, 1)
        if core.s.stranded is not None:
            break
    stranded = [e for e in _kinds(core, "nest.stranded") if e.payload["reason"] == "dock_failed"]
    assert stranded
    tries = [
        e
        for e in _kinds(core, "nest.dock_try")
        if since <= e.occurred_at <= stranded[0].occurred_at
    ]
    assert [e.payload["try"] for e in tries[-3:]] == [1, 2, 3]
    assert not any(e.payload["ok"] for e in tries)
    for attempt in tries:  # each try is the world's answer to its own duck.dock
        answer = _parent(core, attempt)
        command = _parent(core, answer)
        assert answer.kind == "body.dock" and answer.source.kind == "legacy-world"
        assert answer.payload["contact"] is False and answer.payload["tries"] == 1
        assert command.kind == "duck.act" and command.payload["cmd"] == "duck.dock"
        assert command.payload["bel_err_mm"] == attempt.payload["bel_err_mm"]
        assert answer.payload["act_id"] == command.payload["act_id"] == attempt.payload["act_id"]
    help_call = next(
        e for e in _kinds(core, "duck.bid") if e.occurred_at == stranded[0].occurred_at
    )
    assert help_call.payload["need"] and help_call.payload["kind"] == "nest_help"
    notice = _kinds(core, "owner.notice")[0]
    assert "窩" in notice.payload["text"]
    assert snapshot_banner(core)["code"] == "stranded"
    core.place_on_nest("owner")
    assert core.s.nested and core.s.stranded is None and core.s.need is None


def test_near_the_nest_it_looks_for_its_tag_and_each_try_is_the_worlds_answer():
    """Phase 4 (design A2, A11; G29, G30): the look for the nest's tag and every try at the
    contacts are acts; LegacyWorld answers each in the same second with v4's own draw, and
    Duck Core rolls no outcome die of its own."""
    core = _duck(11)
    _family(core, 6 * 60)
    commands = _kinds(core, "duck.act")
    looks = [e for e in commands if e.payload["cmd"] == "duck.look"]
    docks = [e for e in commands if e.payload["cmd"] == "duck.dock"]
    assert looks and docks
    answers = {
        e.parent_event_ids[0]: e
        for e in core.log.events
        if e.kind in ("sense.sighting", "sense.scan", "body.dock") and e.parent_event_ids
    }
    for look in looks:
        assert look.payload["kind"] == "seek_nest" and look.payload["want"] == "nest"
        seen = answers[look.event_id]  # its camera sees the tag: a sighting of the nest
        assert seen.kind == "sense.sighting" and seen.payload["what"] == "nest"
        assert seen.payload["act_id"] == look.payload["act_id"]
        assert seen.occurred_at == look.occurred_at and seen.source.kind == "legacy-world"
    for dock in docks:
        answer = answers[dock.event_id]
        assert answer.kind == "body.dock" and answer.occurred_at == dock.occurred_at
    assert len(_kinds(core, "nest.dock_try")) == len(docks)
    assert not _kinds(core, "sense.nest_seen")
    assert "dock.try" not in core.counters
    assert _world(core).counters["dock.try"] == len(docks)
    # What a look does: the tag in sight fixes its pose; no tag in sight changes nothing.
    while core.acts.busy():
        core.advance_seconds(1)
    core.s.pose_err_mm = 400
    assert nest.seek_nest(core) is True and core.s.pose_err_mm == 60
    _world(core).marker = False
    core.s.pose_err_mm = 400
    assert nest.seek_nest(core) is False and core.s.pose_err_mm == 400
    assert not _kinds(core, "sense.scan")[-1].payload["sightings"]


def test_without_a_camera_it_never_looks_for_the_tag_but_still_docks():
    session = Session(Config(seed=12, nest=True, camera=False, household="elder_home"))
    dispatch(session, "live", {"family": "elder_home", "days": 12})
    core = session.core
    commands = _kinds(core, "duck.act")
    assert not [e for e in commands if e.payload["cmd"] == "duck.look"]
    tries = _kinds(core, "nest.dock_try")
    assert tries and not any(e.payload["seen_tag"] for e in tries)


def snapshot_banner(core: DuckCore) -> dict:
    session = Session(Config(seed=1))
    session.core = core
    return snapshot(session)["body"]["banner"]


def test_getting_up_can_fail_and_then_it_waits_before_trying_again():
    core = _duck(7)
    _world(core).stand_ppm = 0
    _family(core, 40)
    failed = [
        e
        for e in _kinds(core, "body.posture")
        if e.payload["want"] == "stand" and e.payload["got"] == "sit"
    ]
    assert len(failed) >= 2 and not _kinds(core, "body.stand_failed")
    for answer in failed:  # the world's answer to its own try at standing up
        command = _parent(core, answer)
        assert command.kind == "duck.act" and command.payload["cmd"] == "duck.posture"
        assert answer.source.kind == "legacy-world"
    assert not _kinds(core, "duck.walk")[-1:] or all(
        e.occurred_at < failed[0].occurred_at for e in _kinds(core, "duck.walk")
    )
    first, second = failed[0], failed[1]
    # A miss: two minutes before the next try; a second within ten minutes: half an hour.
    assert timedelta(minutes=2) <= second.occurred_at - first.occurred_at <= timedelta(minutes=10)
    later = [e for e in failed if e.occurred_at > second.occurred_at]
    assert all(e.occurred_at - second.occurred_at >= timedelta(minutes=30) for e in later)
    assert "stand_backoff" in core.walk_blocks() or later


def test_falls_near_the_nest_never_make_it_afraid_of_its_nest():
    core = _duck(8)
    for k in range(3):
        spot = {"fell": f"F{k}", "outcome": f"O{k}", "x": 200 + 30 * k, "y": 100, "hard": True}
        spot["zone"] = None
        core.s.fall_spots.append(spot)
        fake = core._emit("body.righting", {"try": 2, "ok": True, "outcome": "hard"})
        spot["outcome"] = fake.event_id
        core._learn_spot(spot)
    assert not core.active_spots()  # no spot to avoid at its own nest…
    trouble = _kinds(core, "nest.trouble")
    assert trouble and trouble[0].payload["falls"] == 3  # …the family is asked to look instead
    assert _kinds(core, "owner.notice")[-1].payload["kind"] == "nest_trouble"
    assert core.s.nest_trouble
    core.ack_nest_trouble()
    assert not core.s.nest_trouble
    for k in range(3):  # far from the nest, three hard falls still teach it (unchanged)
        spot = {"fell": f"G{k}", "outcome": f"P{k}", "x": 1800, "y": -900 + 40 * k, "hard": True}
        spot["zone"] = None
        core.s.fall_spots.append(spot)
        fake = core._emit("body.righting", {"try": 3, "ok": True, "outcome": "hard"})
        spot["outcome"] = fake.event_id
        core._learn_spot(spot)
    assert len(core.active_spots()) == 1


def test_lifted_up_it_waits_to_be_put_back_somewhere():
    core = _duck(9)
    core.pickup("child")
    assert core.s.unplaced and not core.s.nested and not core.s.floor
    assert "unplaced" in core.walk_blocks()
    walks = len(_kinds(core, "duck.walk"))
    _family(core, 30)
    assert len(_kinds(core, "duck.walk")) == walks
    core.place_on_nest("child")
    assert core.s.nested and not core.s.unplaced and core.pose_now() == [0.0, 0.0, 0.0]
    core.pickup("child")
    core.set_setting("floor", True)  # 放地上: just in front of its nest
    assert not core.s.unplaced and core.s.floor and core.pose_now()[0] > 0.3


def test_stranded_off_the_nest_it_asks_to_be_carried_home_and_never_runs_flat_walking():
    session = Session(Config(seed=10, nest=True, conservative=False, household="dual_income"))
    dispatch(session, "live", {"family": "dual_income", "days": 30})
    core = session.core
    for off in _kinds(core, "power.off"):
        if off.payload["reason"] != "battery_empty":
            continue
        assert off.source.kind == "duck-self"  # its own shutdown, not a sensor's input
        # It never runs flat on its own trips: only when it could not get home (lifted
        # and left somewhere, stranded), and then it asked for a hand first.
        last_walk = max(
            w.occurred_at for w in _kinds(core, "duck.walk") if w.occurred_at < off.occurred_at
        )
        asked = [
            b
            for b in _kinds(core, "duck.bid")
            if b.payload.get("need") and last_walk <= b.occurred_at <= off.occurred_at
        ]
        unplaced = any(
            p.occurred_at >= last_walk and p.occurred_at <= off.occurred_at
            for p in _kinds(core, "sense.pickup")
        )
        assert asked or unplaced, off.occurred_at
    for walk in _kinds(core, "duck.walk"):
        assert walk.payload["battery"] >= MIN_MOTION_BATTERY
    stats = _kinds(core, "duck.day_stats")
    assert len(stats) == 30 and all(s.payload["docks"] >= 0 for s in stats)
    assert sum(s.payload["docks"] for s in stats[10:]) >= 15 * 20  # many trips a day


def test_the_nest_duck_and_its_events_are_explained_in_plain_words():
    from duckmind.duckcore.narrate import describe

    session = Session(Config(seed=11, nest=True, conservative=False, household="hybrid"))
    dispatch(session, "live", {"family": "hybrid", "days": 12})
    kinds = {
        "nav.home_start",
        "nest.docked",
        "nest.undocked",
        "nest.dock_try",
        "duck.excursion",
        "duck.heard",
        "env.family",
        "duck.phase",
        "duck.self_wake",
        "duck.day_stats",
    }
    seen = set()
    for event in session.core.log.events:
        if event.kind in kinds:
            text = describe(event)
            assert text != event.kind and not any(c.isascii() and c.isalpha() for c in text), (
                event.kind,
                text,
            )
            seen.add(event.kind)
    assert seen == kinds
    body = snapshot(session)["body"]
    from duckmind.duckcore.daycycle import PHASE_ZH

    assert body["nest"]["docks"] > 0 and body["day"]["phase"] in PHASE_ZH
    assert body["day"]["family_zh"] and body["banner"]["title"]


def test_the_family_can_say_it_in_words_too():
    from duckmind.duckcore import commands

    parsed = [commands.parse(text)[0] for text in ("把牠放回窩裡", "大家出門了", "我回來了")]
    assert [command.method for command in parsed] == [
        "place_on_nest",
        "family_leave",
        "family_return",
    ]
    session = Session(Config(seed=12, nest=True))
    dispatch(session, "advance", {"until": "10:00"})
    dispatch(session, "command", {"text": "大家出門了"})
    assert session.core.s.family == "away"
    dispatch(session, "command", {"text": "我回來了"})
    assert session.core.s.family == "awake"
    assert dispatch(session, "replay_check", {})["identical"] is True


def test_a_fall_on_the_way_home_is_never_walked_through():
    """Falls happen while time passes; an errand or outing never walks on while it lies
    there, and never docks with a leg still under way."""
    core = _duck(13, rates={"fall_ppm_per_s": 300_000, "righting_ppm": 500_000})
    _family(core, 4 * 60)
    down = False
    for event in core.log.events:
        if event.kind == "body.fell":
            down = True
        elif event.kind == "duck.posture" and event.payload["from"] == "fallen":
            down = False
        elif event.kind == "duck.walk":
            assert not down, event.occurred_at
            assert not event.payload["nested"]
    assert _kinds(core, "body.fell")


# ---------------------------------------------------------------------------- the way home
# (fixed after turns became walked arcs, 2026-09-26: the arc round toward home once swung it
# beyond the 1.5 m it keeps to while the family sleeps, and a reflex turn off the nest once
# left it below the battery it keeps for the way home.)


def _standing_out(core: DuckCore, x: int, y: int, heading: int) -> None:
    core.place_on_nest("owner")
    core.s.nested = False
    core.s.posture = "stand"
    core.s.stand_idle_from = core.s.stood_at = core.s.now
    core.s.pos_x, core.s.pos_y, core.s.heading = x, y, heading
    core.s.pose_err_mm = 0


def test_going_home_while_the_family_sleeps_never_leaves_the_1_5_m_it_keeps_to():
    for heading in (0, 400, 800, 1200, 1571, 5500):
        core = _duck(7)
        core.s.phase, core.s.family = "solo", "asleep"
        _standing_out(core, 1400, 0, heading)
        assert nest.start_homing(core, "solo_done")
        for _ in range(180):
            core.advance_seconds(5)
            if core.s.nested or core.s.homing is None:
                break
        walks = [w for w in _kinds(core, "duck.walk") if w.payload["homing"]]
        assert walks
        for walk in walks:
            assert walk.payload["dist_to_mm"] <= 1500, (heading, walk.payload)
            assert walk.payload["leash_mm"] == 1500


def test_an_outing_ends_where_it_can_still_turn_round_toward_home():
    session = Session(Config(seed=14, nest=True, conservative=False, household="dual_income"))
    dispatch(session, "live", {"family": "dual_income", "days": 30})
    core = session.core
    for start in _kinds(core, "nav.home_start"):
        own = not start.payload["hand_placed"] and start.payload["reason"] != "retry"
        assert not own or start.payload["battery"] >= start.payload["threshold"], start.payload
    turned = [w for w in _kinds(core, "duck.walk") if w.payload["verb"] == "turn_around"]
    on_spot = [w for w in turned if w.payload["vx"] == 0]
    assert len(on_spot) * 4 <= len(turned)  # the last resort, not the rule


def test_a_reflex_never_walks_it_off_the_nest_below_the_battery_it_keeps():
    core = _duck(5, days=12, name_ear=True)
    for day in range(3):  # it learns its name
        core.s.name_gds.append(core.s.gd - day - 1)
    core._experience_milestone("learned_name")
    if not core.s.nested:
        core.place_on_nest("owner")
    # a little above its line (and kept there: a nest that does not charge right now)
    core.hw = dataclasses.replace(core.hw, nest_charge_per_min=0, nest_awake_drain=0)
    core.s.battery = nest.threshold(core) + 300
    walks = len(_kinds(core, "duck.walk"))
    for _ in range(6):
        core.call_name("owner")
        core.advance(1)
    assert len(_kinds(core, "duck.walk")) == walks  # up to answer, maybe, but no walk off
    _standing_out(core, 0, 0, 0)
    core.s.battery = nest.threshold(core) + 300
    assert core._motion_block("turn_around") == "going_home"
    core.s.battery = nest.threshold(core) + nest.HOME_MARGIN + 100
    assert core._motion_block("turn_around") is None


def test_the_reserve_pays_for_the_arc_round_toward_home():
    core = _duck(3)
    core.s.pos_x = core.s.pos_y = 0
    near = nest.threshold(core)
    core.s.pos_x = 1500
    far = nest.threshold(core)
    assert MIN_MOTION_BATTERY + 500 <= near < far <= MIN_MOTION_BATTERY + 1500
    # The way home now includes the walked turn (up to half a turn at ~0.3 rad/s ≈ 11 s)
    # and a forward loop after a miss: more than the straight way alone.
    walk_only = 1500 // 110 * (160 + 208) // 60 * 15000 // 10000
    assert far - MIN_MOTION_BATTERY > walk_only * 3


def test_it_sits_down_the_way_it_walked_in_and_leaves_by_a_right_arc():
    core = _duck(11)
    for _ in range(12 * 60):
        _family(core, 1)
        if _kinds(core, "nest.docked") and _kinds(core, "nest.docked")[-1].payload["by"] == "self":
            break
    docked = [e for e in _kinds(core, "nest.docked") if e.payload["by"] == "self"]
    assert docked
    last_walk = [w for w in _kinds(core, "duck.walk") if w.occurred_at <= docked[-1].occurred_at][
        -1
    ]
    if core.s.nested:
        assert core.s.heading == last_walk.payload["expect"]["yaw_mrad"]  # no turn in the nest
    for walk in _kinds(core, "duck.walk"):
        if walk.payload.get("purpose") != "leave":
            continue
        x, y, heading = _pose(walk.payload["bel_from"])
        behind = abs(math.remainder(math.atan2(-y, 0.7 - x) - heading, math.tau)) >= 2.4
        assert walk.payload["vx"] > 0  # out forward, never backing out or turning on the spot
        if behind and walk.payload["verb"] == "turn_around":
            assert walk.payload["vyaw"] < 0  # the way out behind it: round on a right arc


# ---------------------------------------------------------------------------- polish: the page's nest states


def test_an_empty_nest_duck_is_never_switched_on_off_its_nest():
    """Page audit: switched on with an empty pack off its nest, a nest duck only switched
    off again (six power.off a minute apart). It now stays off until someone puts it back
    in its nest, and the switched-off banner's way forward goes through the nest."""
    session = Session(Config(seed=4, nest=True))
    core = session.core
    dispatch(session, "advance", {"until": "10:00"})
    dispatch(session, "act", {"action": "pickup", "persona": "owner"})  # lifted, never put down
    dispatch(session, "advance", {"minutes": 90})
    assert not core.s.awake and core.s.off_reason == "battery_empty"
    assert _kinds(core, "power.off")[-1].source.kind == "duck-self"
    offs = len(_kinds(core, "power.off"))
    ons = len(_kinds(core, "power.on"))
    reply = dispatch(session, "act", {"action": "power_on", "persona": "owner"})
    assert reply["perceived"] is False and "窩" in reply["note"]
    assert len(_kinds(core, "power.on")) == ons and not core.s.awake
    found = snapshot(session)["body"]["banner"]
    assert found["code"] == "off" and "關機" in found["text"]
    primary = next(action for action in found["actions"] if action["primary"])
    assert [call["params"].get("action") for call in primary["calls"]][-2:] == [
        "place_on_nest",
        "power_on",
    ]
    for call in primary["calls"]:
        dispatch(session, call["method"], call["params"])
    assert core.s.awake and core.s.nested
    charging = snapshot(session)["body"]["banner"]
    assert charging["title"] == "在窩裡充電" and "換新電池" not in str(charging["actions"])
    dispatch(session, "advance", {"minutes": 30})
    assert len(_kinds(core, "power.off")) == offs  # charging in the nest: no flapping
    assert dispatch(session, "replay_check", {})["identical"] is True


def test_walked_in_it_sits_in_the_nest_the_way_it_came():
    """Owner decision D1 on the 3D view: docking by itself keeps the heading it walked in
    with (no spin to face the room); a hand sets it down facing the room."""
    from duckmind.duckcore.api import _motion

    core = _duck(seed=6)
    for _ in range(600):
        if any(e.payload.get("by") == "self" for e in _kinds(core, "nest.docked")):
            break
        _family(core, 10)
    docked = [e for e in _kinds(core, "nest.docked") if e.payload.get("by") == "self"]
    assert docked, "it never walked home in the test window"
    assert _motion(docked[-1])["place"] == [0.0, 0.0]  # keeps the heading it came with
    core.pickup("owner")
    hand = core.place_on_nest("owner")
    assert hand["perceived"]
    assert _motion(_kinds(core, "nest.docked")[-1])["place"] == [0.0, 0.0, 0.0]


def test_praise_for_a_moment_is_offered_without_breaking_the_page():
    """Polish: for 3 s after every clever or goofy moment the page's state call failed
    (the praise prompt looked the moment up as a verb); now it names the moment."""
    session = Session(Config(seed=5, nest=True))
    core = session.core
    moment = content.MOMENTS[0]
    core.s.last_flag = {"kind": "moment", "id": moment.moment_id, "at": core.s.now, "event": ""}
    trying = snapshot(session)["body"]["trying"]
    assert trying == {"kind": "moment", "id": moment.moment_id, "text": moment.short}
