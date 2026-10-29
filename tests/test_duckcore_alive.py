"""A living duck, always explained (owner feedback 2026-09-25: "nobody likes a boring pet").

Whenever the duck can move it moves often; whenever it cannot (asleep, night, off, on the
desk, fallen, quiet…) it still shows life where that makes sense, and the page's banner
says why in plain words with one-click ways forward that are ordinary Duck Core inputs.
Everything stays deterministic: the same inputs replay the same chain, whatever the step
size time was advanced in.
"""

from __future__ import annotations

import itertools
import math
import re
from datetime import datetime, time, timedelta
from typing import Any

import pytest

from duckmind.duckcore import content
from duckmind.duckcore.api import (
    Config,
    LegacyActs,
    Session,
    _motion,
    banner,
    clips,
    dispatch,
    legacy_world,
    open_session,
    snapshot,
)
from duckmind.duckcore.engine import (
    FIDGET_JITTER_S,
    FIDGET_PER_MIN,
    FIDGET_SLOT_S,
    MIN_MOTION_BATTERY,
    NAP_MAX_MIN,
    NAP_WAKE_BLOCK_MIN,
    STIR_GAP_S,
    DuckCore,
)
from duckmind.duckcore.fixed import SCALE
from duckmind.provenance import Event
from duckmind.worldport import make_world


def _kinds(core: DuckCore | LegacyActs, kind: str) -> list[Event]:
    return [event for event in core.log.events if event.kind == kind]


def _bare(seed: int, **kwargs: Any) -> LegacyActs:
    """A duck built directly (no Session), living in v4's world (LegacyWorld), as the family's
    hands reach it (api.LegacyActs: a touch or a head hold is felt only when the world's
    detector senses it; everything else goes straight to Duck Core)."""
    port = make_world(legacy_world(Config(seed=seed)))
    return LegacyActs(DuckCore(seed=seed, port=port, **kwargs), port)


def _stood_up(event: Event) -> bool:
    """It came to stand: a stand-up is the world's answer ``body.posture {got: stand}`` (that
    record is the stand, phase 4); every other way to standing is Duck Core's duck.posture."""
    if event.kind == "body.posture":
        return event.payload.get("got") == "stand" and not event.payload.get("stale")
    return event.kind == "duck.posture" and event.payload["posture"] == "stand"


def _advance(session: Session, total: int, chunks: list[int]) -> None:
    left = total
    for step in itertools.cycle(chunks):
        if left <= 0:
            return
        size = min(step, left)
        session.core.advance(size)
        left -= size


def _at(core: DuckCore | LegacyActs, hour: int, minute: int = 0) -> None:
    core.advance(core.minutes_until(f"{hour:02d}:{minute:02d}"))


def _chain(session: Session) -> str:
    return session.core.log.head


# ---------------------------------------------------------------------------- determinism


@pytest.mark.parametrize("docked", [False, True])
def test_time_passes_the_same_whatever_the_step_size(docked):
    """A day in one step and minute by minute grow the same duck (growth days included:
    awake minutes before midnight count for the day they were in)."""

    def run(chunks: list[int]) -> Session:
        session = Session(Config(seed=11))
        if docked:
            dispatch(session, "set_setting", {"key": "docked", "value": True})
        _advance(session, 3 * 1440, chunks)
        return session

    whole = run([3 * 1440])
    assert _chain(run([1])) == _chain(whole)
    assert _chain(run([7, 60, 13, 240])) == _chain(whole)
    assert _kinds(whole.core, "growth.gd")


def test_new_inputs_are_recorded_and_replay_identically():
    session = Session(Config(seed=5))
    script = [
        ("set_setting", {"key": "docked", "value": True}),
        ("advance", {"until": "23:30"}),
        ("act", {"action": "wake", "persona": "child"}),
        ("advance", {"minutes": 90}),
        ("act", {"action": "owner_enough"}),
        ("act", {"action": "end_quiet"}),
        ("set_setting", {"key": "docked", "value": False}),
        ("act", {"action": "swap_battery"}),
        ("set_setting", {"key": "battery_help", "value": False}),
        ("advance", {"until": "07:00"}),
        ("command", {"text": "叫醒牠，然後跳到早上"}),
    ]
    for method, params in script:
        dispatch(session, method, params)
    assert dispatch(session, "replay_check", {})["identical"] is True


# ---------------------------------------------------------------------------- sleep shows life


def _night_on_the_charger(seed: int = 3) -> Session:
    session = Session(Config(seed=seed))
    dispatch(session, "set_setting", {"key": "docked", "value": True})
    dispatch(session, "advance", {"until": "07:00"})
    return session


def test_asleep_at_night_it_still_stirs_every_few_minutes():
    session = _night_on_the_charger()
    core = session.core
    stirs = [e for e in _kinds(core, "duck.sleep_stir") if e.payload["reason"] == "dream"]
    night = [e for e in stirs if e.occurred_at.hour >= 22 or e.occurred_at.hour < 7]
    assert len(night) >= 25  # nine hours of sleep: a stir every 5–15 minutes
    kinds = {e.payload["kind"] for e in night}
    assert {"shift", "nuzzle"} <= kinds and len(kinds) >= 3
    gaps = [
        (b.occurred_at - a.occurred_at).total_seconds()
        for a, b in zip(night, night[1:], strict=False)
        if b.occurred_at - a.occurred_at < timedelta(minutes=16)
    ]
    assert gaps and min(gaps) >= STIR_GAP_S[0] and max(gaps) <= STIR_GAP_S[1]
    # The log stays in time order across midnight, stirs and the day's roll-over included.
    times = [event.occurred_at for event in core.log.events]
    assert times == sorted(times)
    for stir in night:
        motion = _motion(stir)
        assert motion["stir"] is True and motion["clip"] in content.CLIPS


def test_the_owner_can_wake_it_and_it_stays_up_for_a_while():
    core = _night_on_the_charger().core
    _at(core, 2, 55)
    assert snapshot_mode(core) == "nap"
    note = core.wake("owner")["note"]
    assert "醒了" in note and not core.s.napping_until
    woke = _kinds(core, "duck.nap_end")[-1]
    assert woke.payload["by"] == "family"
    assert _kinds(core, "duck.expression")[-1].payload["verb"] == "stretch_yawn"
    core.advance(NAP_WAKE_BLOCK_MIN - 1)
    assert core.s.napping_until is None  # no dozing off again straight away
    core.advance(NAP_WAKE_BLOCK_MIN + 3)
    assert core.s.napping_until is not None  # alone at night on its charger, it sleeps on


def snapshot_mode(core: DuckCore) -> str:
    return core.still_reason()[0]


def test_a_tired_duck_naps_until_rested_and_a_pat_or_its_name_can_wake_it():
    session = Session(Config(seed=8))
    core = session.core
    _at(core, 10)
    core.s.energy = 2800
    core.advance(1)
    start = _kinds(core, "duck.nap_start")[-1]
    assert start.payload["reason"] == "tired"
    assert 10 <= start.payload["minutes"] <= NAP_MAX_MIN  # one real nap, not a doze
    core.advance(start.payload["minutes"] + 1)
    assert core.s.napping_until is None and core.s.energy >= 7000
    woken = 0
    for seed in range(1, 13):
        duck = _bare(seed, battery_help=True)
        duck.advance(duck.minutes_until("10:00"))
        duck.s.energy = 2500
        duck.advance(1)
        assert duck.s.napping_until is not None
        duck.touch("child", seconds=6, firmness="scratch")
        duck.call_name("owner") if duck.s.napping_until else None
        woken += duck.s.napping_until is None
    assert 3 <= woken <= 11  # sometimes, drawn by the duck


def test_a_pat_while_it_sleeps_is_a_visible_nuzzle_not_a_frozen_head():
    core = _night_on_the_charger().hands
    _at(core, 1, 30)
    assert core.s.napping_until is not None
    before = len(core.log.events)
    core.touch("owner", seconds=4, firmness="light")
    stirs = [e for e in core.log.events[before:] if e.kind == "duck.sleep_stir"]
    if stirs:  # the detector may miss a light touch
        assert stirs[0].payload["kind"] == "nuzzle" and core.s.napping_until is not None


# ---------------------------------------------------------------------------- battery and power


def test_the_family_swaps_the_battery_so_a_battery_duck_keeps_living():
    helped = Session(Config(seed=4))  # the simulator default
    alone = Session(Config(seed=4, battery_help=False))
    for session in (helped, alone):
        dispatch(session, "advance", {"minutes": 1440})
    assert helped.core.s.awake
    swaps = _kinds(helped.core, "power.battery_swap")
    assert len(swaps) >= 15 and all(e.payload["by"] == "family" for e in swaps)
    assert all(e.payload["before"] < MIN_MOTION_BATTERY * 100 // SCALE for e in swaps)
    assert not alone.core.s.awake  # without help it ran flat and stayed off
    assert alone.core.s.off_reason == "battery_empty"


def test_on_its_cable_it_charges_and_unplugging_gives_the_charged_battery():
    core = DuckCore(seed=2)
    core.advance(40)
    low = core.s.battery
    core.set_setting("docked", True)
    core.advance(30)
    assert core.s.battery == min(SCALE, low + 30 * 100)
    core.set_setting("docked", False)
    assert core.s.battery > low
    core.power_off("owner")
    core.set_setting("docked", True)
    core.advance(200)
    assert core.s.battery == SCALE  # switched off on the cable, it still charges


def test_a_fresh_battery_by_hand_wakes_a_bedtime_doze():
    core = DuckCore(seed=6)
    core.advance(56)  # about 15% left: bedtime, a doze
    assert core.s.bedtime_done and core.s.napping_until is not None
    assert core.swap_battery("owner")["note"].startswith("換上新電池")
    assert core.s.battery == SCALE and core.s.napping_until is None
    core.power_off("owner")
    assert core.swap_battery("owner")["note"] == "換上新電池、開機" and core.s.awake


# ---------------------------------------------------------------------------- hands and the owner


def _fallen_and_stuck(seed: int = 1) -> Session:
    # (the outcome rates are the world's: a v4 config's rate fields become the LegacyWorld
    # header when the duck is opened)
    session = open_session(
        {"engine_rev": "v4_1", "seed": seed, "fall_ppm_per_s": 1_000_000, "righting_ppm": 0}
    )
    assert isinstance(session, Session)
    dispatch(session, "live", {"days": 8, "routine": "floor"})
    core = session.core
    dispatch(session, "advance", {"until": "10:00"})
    dispatch(session, "act", {"action": "power_on"})
    dispatch(session, "set_setting", {"key": "floor", "value": True})
    for _ in range(240):
        if core.s.posture == "fallen" and core.s.righting is None:
            break
        core.advance(1)
    assert core.s.posture == "fallen" and core.s.righting is None
    return session


def test_set_upright_keeps_it_on_the_floor_and_lying_stuck_it_keeps_trying():
    session = _fallen_and_stuck()
    core = session.core
    where = core.pose_now()
    before = len(core.log.events)
    core.advance(60)
    assert [e for e in core.log.events[before:] if e.kind == "duck.struggle"]  # (its own)
    assert banner(session)["code"] == "fallen"
    dispatch(session, "act", {"action": "set_upright"})
    assert core.s.posture == "sit" and core.s.floor and core.pose_now() == where
    assert _kinds(core, "duck.posture")[-1].payload["reason"] == "set_upright"


def test_the_owner_can_end_quiet_time_a_lockout_and_a_stop():
    core = _bare(3)
    core.owner_enough()
    assert core.still_reason()[0] == "quiet"
    assert "可以動了" in core.end_quiet()["note"]
    assert core.s.quiet_until is None
    for _ in range(6):  # the joint detector may miss a hold now and then
        core.hold_head("owner")
        core.advance(1)
        if core.s.lockout_until is not None:
            break
    assert core.s.lockout_until is not None and core.still_reason()[0] == "lockout"
    core.end_quiet()
    assert core.s.lockout_until is None and core.still_reason()[0] != "lockout"
    assert list(_kinds(core, "owner.release")[-1].payload["ended"]) == ["lockout"]


def test_skipping_to_the_morning_lands_on_the_same_moment_in_a_replay():
    session = Session(Config(seed=9))
    dispatch(session, "advance", {"until": "07:00"})
    assert session.core.s.now.time() == time(7, 0)
    dispatch(session, "advance", {"until": "07:00"})
    assert session.core.s.now.time() == time(7, 0)  # the next one: a day later
    assert session.core.s.now.date() == datetime(2026, 12, 22).date()
    assert dispatch(session, "replay_check", {})["identical"] is True


# ---------------------------------------------------------------------------- the banner


def _run(session: Session, action: dict) -> None:
    for step in action["calls"]:
        dispatch(session, step["method"], step["params"])


def _primary(session: Session) -> dict:
    found = banner(session)
    return next(action for action in found["actions"] if action["primary"])


def _plain_chinese(text: str) -> bool:
    return not re.search(r"[A-Za-z_]", text)


def _states() -> dict[str, Session]:
    states: dict[str, Session] = {}
    states["desk"] = Session(Config(seed=1))
    off_night = Session(Config(seed=2))
    dispatch(off_night, "live", {"days": 7, "routine": "floor"})
    states["off_night"] = off_night
    states["night_nap"] = _night_on_the_charger(4)
    dispatch(states["night_nap"], "advance", {"until": "02:55"})
    hot = Session(Config(seed=3))
    dispatch(hot, "set_setting", {"key": "hot", "value": True})
    states["hot"] = hot
    quiet = Session(Config(seed=3))
    dispatch(quiet, "act", {"action": "owner_enough"})
    states["quiet"] = quiet
    tethered = Session(Config(seed=5))
    dispatch(tethered, "live", {"days": 8, "routine": "floor"})
    for method, params in (
        ("advance", {"until": "10:00"}),
        ("set_setting", {"key": "docked", "value": True}),
        ("act", {"action": "power_on"}),
        ("set_setting", {"key": "floor", "value": True}),
    ):
        dispatch(tethered, method, params)
    states["tethered"] = tethered
    low = Session(Config(seed=6, battery_help=False))
    dispatch(low, "advance", {"minutes": 46})
    states["battery"] = low
    return states


def test_every_state_is_explained_in_plain_words_with_ways_forward():
    for name, session in _states().items():
        found = banner(session)
        assert found["title"] and found["text"], name
        for text in (found["title"], found["text"], *[a["label"] for a in found["actions"]]):
            assert _plain_chinese(text), (name, text)
        if found["code"] != "ok":
            assert found["actions"], name
        for action in found["actions"]:  # every button is a list of ordinary inputs
            assert action["calls"] and all(
                step["method"] in {"act", "advance", "set_setting", "live"}
                for step in action["calls"]
            )
    assert snapshot(Session(Config(seed=1)))["body"]["banner"]["code"] == "desk"


def test_one_click_from_each_state_moves_the_duck_forward():
    states = _states()
    expectations = {
        "desk": lambda core: core.s.floor,
        "off_night": lambda core: (
            core.s.awake and core.s.now.hour == 7 and core.s.floor
        ),  # it was on the floor when the family switched it off
        "night_nap": lambda core: core.s.napping_until is None,
        "hot": lambda core: not core.s.hot,
        "quiet": lambda core: core.s.quiet_until is None,
        "tethered": lambda core: not core.s.docked and core.s.floor,
        "battery": lambda core: core.s.battery == SCALE,
    }
    for name, check in expectations.items():
        session = states[name]
        _run(session, _primary(session))
        assert check(session.core), (name, banner(session))
        assert dispatch(session, "replay_check", {})["identical"] is True, name


def test_a_hatchling_is_one_click_away_from_its_first_steps():
    session = Session(Config(seed=12))
    dispatch(session, "set_setting", {"key": "floor", "value": True})
    found = banner(session)
    assert found["code"] == "too_young"
    grow = next(action for action in found["actions"] if "成長日" in action["label"])
    _run(session, grow)
    core = session.core
    assert core.s.gd >= content.FIRST_STEPS_GD and "wander" in core.s.unlocked
    assert core.s.awake and core.s.floor and 8 <= core.s.now.hour < 9
    walks = len(_kinds(core, "duck.walk"))
    core.advance(60)
    assert len(_kinds(core, "duck.walk")) > walks  # and it walks, on its own


def test_a_walking_outing_has_a_few_legs_in_a_row():
    session = Session(Config(seed=2, conservative=False))
    dispatch(session, "live", {"days": 50, "routine": "floor"})
    core = session.core
    walk_on = [
        e
        for e in _kinds(core, "duck.decision")
        if e.payload.get("trigger") == "walk_on" and e.payload["verb"] in content.WALK_VERBS
    ]
    assert len(walk_on) >= 20


# ---------------------------------------------------------------------------- the idle layer


def test_idle_glances_are_drawn_from_the_seed_and_never_touch_the_chain():
    session = Session(Config(seed=7))
    core = session.core
    _at(core, 10)
    head = core.log.head
    start = core.s.now.replace(second=0)
    minute = start + timedelta(minutes=1)
    first = core.fidgets_between(start, minute, [])
    assert first == core.fidgets_between(start, minute, []) and core.log.head == head
    # One every ~6 s: 9–11 in a minute (the slot at the minute's first second falls on
    # the start itself, outside the window, when its jitter draws 0).
    assert FIDGET_PER_MIN[0] - 1 <= len(first) <= FIDGET_PER_MIN[1] + 1
    # Never frozen for long: two glances are at most a slot and its jitter apart.
    gaps = [(b["at"] - a["at"]).total_seconds() for a, b in itertools.pairwise(first)]
    assert (
        max(gaps) <= FIDGET_SLOT_S + FIDGET_JITTER_S
        and min(gaps) >= FIDGET_SLOT_S - FIDGET_JITTER_S
    )
    assert all(a["clip"] != b["clip"] for a, b in itertools.pairwise(first))  # never twice
    assert all(item["clip"] in content.CLIPS for item in first)
    assert all(start < item["at"] <= minute for item in first)
    # The same glances whether asked for second by second or for the whole minute.
    pieces = [
        item
        for k in range(60)
        for item in core.fidgets_between(
            start + timedelta(seconds=k), start + timedelta(seconds=k + 1), []
        )
    ]
    assert pieces == first
    later = [
        core.fidgets_between(start + timedelta(minutes=k), start + timedelta(minutes=k + 1), [])
        for k in range(1, 30)
    ]
    assert len({tuple(item["clip"] for item in items) for items in later}) > 10
    # Big enough to see across the room: every glance turns or tips the head by ≥ 8°.
    for items in later:
        for item in items:
            frames = content.CLIPS[item["clip"]]
            peak = max(
                abs(frame.get(key, 0.0)) * content.SEATED_ENVELOPE[joint]
                for _, frame in frames
                for key, joint in (
                    ("hy", "head_yaw"),
                    ("hp", "head_pitch"),
                    ("np", "neck_pitch"),
                    ("hr", "head_roll"),
                )
            )
            assert math.degrees(peak * item["amplitude"] / SCALE) >= 8
    before = len(core.log.events)
    core.advance(1)
    done = list(core.log.events[before:])
    for item in core.fidgets_between(start, core.s.now, done):
        # Never while a move it really did is showing (its whole clip, walk or posture).
        for event in done:
            span = core.move_span(event)
            if span is not None:
                assert not span[0] <= item["at"] <= span[1], (item, event.kind)
    core.s.energy = 2000
    core.advance(2)
    assert core.s.napping_until is not None and not core.can_fidget()


def test_the_page_can_name_every_clip_in_plain_words():
    payload = clips(Session(Config(seed=1)))
    assert set(content.CLIPS) <= set(payload["names"])
    assert all(_plain_chinese(name) for name in payload["names"].values())
    assert payload["nap_pose"] == content.NAP_POSE


# ---------------------------------------------------------------------------- real time, to the second


def _floor_morning(seed: int, days: int = 0, **config: bool) -> Session:
    """A rested duck (``days`` of the family's floor routine first), switched on at 09:00
    with a fresh battery and put on the floor."""
    session = Session(Config(seed=seed, **config))  # type: ignore[arg-type]
    if days:
        dispatch(session, "live", {"days": days, "routine": "floor"})
    for method, params in (
        ("act", {"action": "power_off"}),
        ("advance", {"until": "09:00"}),
        ("act", {"action": "power_on"}),
        ("set_setting", {"key": "floor", "value": True}),
    ):
        dispatch(session, method, params)
    return session


def _seconds(session: Session, total: int, chunks: list[int]) -> Session:
    left = total
    for step in itertools.cycle(chunks):
        if left <= 0:
            break
        size = min(step, left)
        dispatch(session, "advance", {"seconds": size})
        left -= size
    return session


def test_time_passes_the_same_to_the_second():
    """Real time lives the duck second by second; a replay does it in one step. The same
    duck either way, awake on the floor and asleep on its charger through the night."""
    whole = _seconds(_floor_morning(11), 3 * 3600, [3 * 3600])
    assert _chain(_seconds(_floor_morning(11), 3 * 3600, [1])) == _chain(whole)
    assert _chain(_seconds(_floor_morning(11), 3 * 3600, [7, 59, 61, 13])) == _chain(whole)
    # What it decides falls on every second of the minute, not always at its end.
    seconds = {e.occurred_at.second for e in _kinds(whole.core, "duck.decision")}
    assert len(seconds) >= 30

    def night(chunks: list[int]) -> Session:
        session = Session(Config(seed=4))
        dispatch(session, "set_setting", {"key": "docked", "value": True})
        dispatch(session, "advance", {"until": "21:50"})
        return _seconds(session, 10 * 3600, chunks)

    asleep = night([10 * 3600])
    assert _chain(night([1])) == _chain(asleep)
    assert _chain(night([17, 600, 3])) == _chain(asleep)
    assert _kinds(asleep.core, "duck.sleep_stir")


def test_back_to_back_seconds_are_one_input_and_replay_identically():
    session = _floor_morning(3)
    inputs = len(session.inputs)
    _seconds(session, 600, [1, 2, 3])
    assert len(session.inputs) == inputs + 1 and session.inputs[-1][1] == {"seconds": 600}
    dispatch(session, "act", {"action": "owner_hello"})
    _seconds(session, 90, [1])
    assert len(session.inputs) == inputs + 3
    assert dispatch(session, "replay_check", {})["identical"] is True


def test_at_seven_the_household_wakes_up_and_gets_up():
    woke = 0
    for seed in range(1, 9):
        session = Session(Config(seed=seed))
        for method, params in (
            ("set_setting", {"key": "docked", "value": True}),
            ("set_setting", {"key": "floor", "value": True}),
            ("advance", {"until": "06:59"}),
        ):
            dispatch(session, method, params)
        core = session.core
        assert core.s.nap_reason == "quiet_hours" and banner(session)["code"] == "nap"
        before = len(core.log.events)
        dispatch(session, "advance", {"until": "07:00"})  # the banner's "跳到早上 7 點"
        new = core.log.events[before:]
        assert any(e.kind == "duck.nap_end" and e.payload["by"] == "morning" for e in new)
        assert core.s.napping_until is None and banner(session)["code"] != "nap"
        woke += any(_stood_up(e) for e in new)
        assert dispatch(session, "replay_check", {})["identical"] is True
    assert woke >= 5  # most get up to start the day (a drawn decision, never forced)


def test_each_one_click_fix_gets_the_duck_up_right_away():
    """The verifier's case: after 好了，可以動了, 拔掉電線, 降溫 or 放到地上 the duck used to
    sit for minutes. Now it gets up to look about at once (a drawn, logged decision)."""
    fixes = {
        "hand": (("act", {"action": "hold_head"}),),
        "unplug": (("set_setting", {"key": "docked", "value": True}),),
        "hot": (("set_setting", {"key": "hot", "value": True}),),
        "desk": (("set_setting", {"key": "floor", "value": False}),),
    }
    up = dict.fromkeys(fixes, 0)
    for seed in range(1, 9):
        for name, setup in fixes.items():
            session = _floor_morning(seed)
            dispatch(session, "advance", {"minutes": 3})
            if session.core.s.posture != "sit":
                dispatch(session, "act", {"action": "pickup"})  # sitting, off the floor…
                dispatch(session, "set_setting", {"key": "floor", "value": True})
                dispatch(session, "advance", {"minutes": 2})
                if session.core.s.posture != "sit":
                    continue
            for method, params in setup:
                dispatch(session, method, params)
            found = banner(session)
            assert found["code"] != "ok" and found["actions"], (name, found)
            core = session.core
            before = len(core.log.events)
            _run(session, _primary(session))
            dispatch(session, "advance", {"seconds": 20})
            new = core.log.events[before:]
            up[name] += any(
                _stood_up(e)
                or (e.kind in {"duck.posture", "duck.walk"} and e.payload.get("posture") != "sit")
                for e in new
            )
            assert dispatch(session, "replay_check", {})["identical"] is True
    assert all(count >= 5 for count in up.values()), up


def test_a_hatchling_on_the_floor_stands_up_to_look_about_often():
    session = Session(Config(seed=12))
    dispatch(session, "advance", {"until": "09:00"})
    dispatch(session, "act", {"action": "power_on"})
    dispatch(session, "set_setting", {"key": "floor", "value": True})
    core = session.core
    before = len(core.log.events)
    dispatch(session, "advance", {"minutes": 120})
    new = core.log.events[before:]
    stands = [e for e in new if _stood_up(e)]
    assert len(stands) >= 8  # several times an hour, as the banner says
    assert not [e for e in new if e.kind == "duck.walk"]  # but no steps before GD 5
    assert "站起來" in banner(session)["text"] or banner(session)["code"] != "too_young"


def test_the_remaining_still_states_have_a_way_forward_too():
    tired = _floor_morning(5, days=8)
    core = tired.core
    core.s.energy = 2000
    dispatch(tired, "act", {"action": "owner_hello"})  # someone is here: it will not doze off
    dispatch(tired, "advance", {"seconds": 30})
    found = banner(tired)
    assert found["code"] == "tired" and found["actions"], found
    nap_start = len(core.log.events)
    _run(tired, _primary(tired))
    # One click forward: a nap until it is rested (not a nap you then wake it from, which
    # only brought it back to "tired"): it has slept its fill and is free to get up.
    ends = [e for e in core.log.events[nap_start:] if e.kind == "duck.nap_end"]
    assert ends and ends[-1].payload["by"] == "rested"
    assert banner(tired)["code"] not in {"tired", "nap"}, banner(tired)

    for missing in ("pickup_detect", "motion_validated"):
        session = _floor_morning(6, days=8, **{missing: False})
        found = banner(session)
        assert found["title"] == "這隻鴨設定成不會自己走" and found["actions"], found
        _run(session, _primary(session))
        if banner(session)["actions"]:
            _run(session, _primary(session))
        dispatch(session, "advance", {"minutes": 60})
        assert _kinds(session.core, "duck.walk"), missing
        assert dispatch(session, "replay_check", {})["identical"] is True

    flat = _floor_morning(7, days=8, battery_help=False)
    dispatch(flat, "advance", {"minutes": 120})
    assert not flat.core.s.awake and flat.core.s.off_reason == "battery_empty"
    first = _primary(flat)
    assert "家人幫忙換電池" in first["label"]
    _run(flat, first)
    assert flat.core.s.awake and flat.core.s.battery_help and flat.core.s.floor


def test_asleep_at_night_after_unplugging_it_does_not_claim_to_be_on_the_cable():
    session = Session(Config(seed=8))
    dispatch(session, "set_setting", {"key": "docked", "value": True})
    dispatch(session, "advance", {"until": "23:30"})
    dispatch(session, "set_setting", {"key": "docked", "value": False})
    found = banner(session)
    assert found["code"] == "nap" and "插著電" not in found["title"] + found["text"]
    note = dispatch(session, "act", {"action": "wake"})["note"]
    assert "插著電" not in note


# ---------------------------------------------------------------------------- round 3: never boring, always honest


def _able_sitting_stretches(
    session: Session, minutes: int
) -> tuple[list[int], list[tuple[int, int]]]:
    """Minute by minute: the stretches (in minutes) it sat able to get up without getting
    up, and for every able seated minute the banner's promise (eta) against the minutes it
    really took to get up."""
    core = session.core
    runs, promises, run = [], [], 0
    pending: list[tuple[int, int]] = []  # (minute the promise was made, eta)
    for minute in range(minutes):
        able = banner(session)["code"] == "ok" and core.s.posture == "sit"
        eta = core.outing_eta() if able else None
        if eta is not None:
            pending.append((minute, eta))
        before = len(core.log.events)
        dispatch(session, "advance", {"minutes": 1})
        new = core.log.events[before:]
        up = any(e.kind in {"duck.posture", "duck.walk"} for e in new)
        rest = any(e.kind in {"duck.nap_start", "power.off"} for e in new) or banner(session)[
            "code"
        ] not in {"ok"}
        if up or rest or not able:
            if run:
                runs.append(run)
            run = 0
            if up:
                promises += [(eta, minute - made + 1) for made, eta in pending]
            pending = []
        else:
            run += 1
    return runs, promises


@pytest.mark.parametrize("seed, days", [(7, 5), (11, 30), (42, 100)])
def test_a_seated_floor_duck_is_up_again_within_a_few_minutes(seed, days):
    """Round-3 verifier: the banner promised "every few minutes" while the duck sat 23–24
    minutes. Now every seated minute adds to its wish to get up, so the wait is still a
    random draw but never a long one, and the banner's "within N minutes" holds."""
    session = _floor_morning(seed, days=days)
    runs, promises = _able_sitting_stretches(session, 10 * 60)
    assert runs and max(runs) <= 8, sorted(runs)[-5:]
    assert sum(run >= 5 for run in runs) <= 12
    assert promises and all(took <= eta + 1 for eta, took in promises), [
        p for p in promises if p[1] > p[0] + 1
    ][:5]
    text = banner(session)["text"]
    napping = session.core.s.napping_until is not None
    assert (
        "分鐘內" in text
        or "隨時" in text
        or session.core.s.posture != "sit"
        or (napping and "分鐘醒" in text)
    )


def test_the_owner_can_invite_it_to_walk_and_every_gate_still_holds():
    went = 0
    for seed in range(1, 9):
        session = _floor_morning(seed, days=8)
        dispatch(session, "advance", {"minutes": 2})
        core = session.core
        if core.s.posture != "sit":
            dispatch(session, "advance", {"minutes": 2})
        found = banner(session)
        if found["code"] != "ok":
            continue
        button = next(a for a in found["actions"] if a["primary"])
        assert button["calls"] == [
            {"method": "act", "params": {"action": "owner_walk", "params": {}}}
        ]
        before = len(core.log.events)
        _run(session, button)
        dispatch(session, "advance", {"seconds": 15})
        new = core.log.events[before:]
        went += any(e.kind == "duck.walk" for e in new)
        assert any(e.kind == "owner.walk" for e in new)
        assert dispatch(session, "replay_check", {})["identical"] is True
    assert went >= 6  # a rested duck nearly always goes (a drawn reflex, never forced)

    night = _floor_morning(3, days=8)
    dispatch(night, "advance", {"until": "23:00"})
    core = night.core
    if core.s.napping_until is not None:
        dispatch(night, "act", {"action": "wake"})
    before = len(core.log.events)
    note = dispatch(night, "act", {"action": "owner_walk"})["note"]
    dispatch(night, "advance", {"seconds": 20})
    assert "晚上" in note
    assert not [e for e in core.log.events[before:] if e.kind in {"duck.walk", "duck.posture"}]


def test_greetings_reach_a_sleeping_duck_as_a_peek_not_as_awake_moves():
    session = Session(Config(seed=8))
    dispatch(session, "set_setting", {"key": "docked", "value": True})
    dispatch(session, "advance", {"until": "23:30"})
    core = session.core
    assert core.s.nap_reason == "quiet_hours"
    for action in ("owner_hello", "owner_good", "pad_good", "owner_walk"):
        before = len(core.log.events)
        result = dispatch(session, "act", {"action": action})
        new = core.log.events[before:]
        assert core.s.napping_until is not None, action  # at night it sleeps on
        assert not [e for e in new if e.kind in {"duck.expression", "duck.greeting"}], action
        assert [e for e in new if e.kind == "duck.sleep_stir" and e.payload["kind"] == "peek"]
        assert "睡" in result["note"], (action, result)
    assert dispatch(session, "replay_check", {})["identical"] is True


def test_put_down_at_night_it_says_it_will_not_walk_until_morning():
    session = Session(Config(seed=8))
    dispatch(session, "set_setting", {"key": "docked", "value": True})
    dispatch(session, "advance", {"until": "23:30"})
    dispatch(session, "set_setting", {"key": "docked", "value": False})
    note = dispatch(session, "set_setting", {"key": "floor", "value": True})["note"]
    assert "晚上" in note and "白天" not in note, note
    found = banner(session)
    assert found["actions"][0]["primary"] and found["actions"][0]["calls"] == [
        {"method": "advance", "params": {"until": "07:00"}}
    ]


def test_idle_glances_leave_no_ten_seconds_without_a_move():
    """Second by second, like real time: every 10 s while it may move holds a glance or a
    move it really did (the verifier's 10-s windows)."""
    session = _floor_morning(7, days=5)
    core = session.core
    start = core.s.now
    starts: list[datetime] = []
    for _ in range(20 * 60):
        began = core.s.now
        glance = core.can_fidget()
        before = len(core.log.events)
        dispatch(session, "advance", {"seconds": 1})
        for event in core.log.events[before:]:
            if core.move_span(event) is not None:
                starts.append(event.occurred_at)
        if glance and core.can_fidget():
            starts += [
                item["at"]
                for item in core.fidgets_between(began, core.s.now, core.log.events[-60:])
            ]
    starts.sort()
    gaps = [(b - a).total_seconds() for a, b in itertools.pairwise([start, *starts])]
    napped = core.s.napping_until is not None or _kinds(core, "duck.nap_start")
    assert napped or max(gaps) <= 10, sorted(gaps)[-5:]


# ---------------------------------------------------------------------------- polish: the leftover page issues


def _tired_with_company(seed: int = 7) -> Session:
    """A floor duck walked tired with the owner around (so it does not doze off by itself),
    from recorded inputs only: 你好 every other minute and 去走走 every minute."""
    session = _floor_morning(seed, days=8)
    for minute in range(180):
        if minute % 2 == 0:
            dispatch(session, "act", {"action": "owner_hello", "persona": "owner"})
        dispatch(session, "act", {"action": "owner_walk", "persona": "owner"})
        dispatch(session, "advance", {"minutes": 1})
        if banner(session)["code"] == "tired":
            return session
    raise AssertionError("never got tired")


def test_a_tired_duck_sleeps_its_fill_in_one_click_instead_of_looping():
    """Verifier: '累了' offered a nap, the nap offered '叫醒牠', and that led straight back to
    '累了'. Now the tired banner's one click is a nap until rested, the nap's own primary is
    to let it sleep its fill, and waking it early says it will still be too tired."""
    session = _tired_with_company()
    core = session.core
    found = banner(session)
    first = _primary(session)
    assert found["title"] == "累了，先坐著休息" and "睡飽" in first["label"]
    before = len(core.log.events)
    _run(session, first)
    ends = [e for e in core.log.events[before:] if e.kind == "duck.nap_end"]
    assert ends and ends[-1].payload["by"] == "rested"
    after = banner(session)
    assert after["code"] not in {"tired", "nap"} and core.s.energy >= SCALE // 2, after
    assert dispatch(session, "replay_check", {})["identical"] is True

    napping = _tired_with_company(11)
    dispatch(napping, "act", {"action": "nap_now", "persona": "owner"})
    nap = banner(napping)
    assert nap["title"] == "累了，小睡中"
    primary = _primary(napping)
    assert primary["calls"] == [{"method": "advance", "params": {"until": "nap_end"}}]
    wake = next(a for a in nap["actions"] if not a["primary"])
    assert "還很累" in wake["label"]
    _run(napping, primary)
    assert not napping.core._until(napping.core.s.napping_until)
    assert banner(napping)["code"] not in {"tired", "nap"}
    assert dispatch(napping, "replay_check", {})["identical"] is True


def test_the_switched_off_banner_says_when_it_was_switched_off():
    """Verifier: after a routine switched it off at 23:59 the banner said 關機（04:01）, the
    clock of the moment it was read."""
    session = _floor_morning(3)
    dispatch(session, "advance", {"until": "10:30"})
    dispatch(session, "act", {"action": "power_off"})
    dispatch(session, "advance", {"minutes": 95})
    text = banner(session)["text"]
    assert "10:30 關機" in text and "現在 12:05" in text, text
    dispatch(session, "advance", {"until": "09:00"})  # the next day: the date is said too
    text = banner(session)["text"]
    assert f"{session.core.s.last_power_off:%m/%d} 10:30 關機" in text, text
