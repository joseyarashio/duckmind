"""A day at home (Duck Core v4, part A): the duck's own day, from what it hears.

It wakes by itself at 07:00 in its nest, tells from logged cues whether the family is up,
asleep or out (never from anything it did not log), rests in the nest while everyone is
out, learns when they come home and wakes a little before, and winds down at 21:30.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta

from duckmind.duckcore import content, daycycle
from duckmind.duckcore.api import Config, LegacyActs, Session, dispatch
from duckmind.duckcore.engine import DuckCore
from duckmind.duckcore.narrate import describe
from duckmind.provenance import Event

R = content.RHYTHM


def _duck(seed: int = 3, **config: object) -> DuckCore:
    return Session(Config(seed=seed, nest=True, conservative=False, **config)).core  # type: ignore[arg-type]


def _kinds(core: DuckCore, kind: str) -> list[Event]:
    return [event for event in core.log.events if event.kind == kind]


def _to(core: DuckCore, hour: int, minute: int = 0) -> None:
    core.advance(core.minutes_until(f"{hour:02d}:{minute:02d}"))


def _cues(core: DuckCore, minutes: int, *, room: bool = True, every: int = 3) -> None:
    """Someone at home: a voice every few minutes, seen by the camera when ``room``."""
    for minute in range(minutes):
        if minute % every == 0:
            core.hear_home(None, kind="voice", journal=False)
            if room:
                core.hear_home(None, kind="person", journal=False)
        core.advance(1)


def _transitions(core: DuckCore) -> list[tuple[str, str]]:
    return [(e.payload["from"], e.payload["to"]) for e in _kinds(core, "env.family")]


def test_it_wakes_by_itself_between_seven_and_five_past_in_its_nest():
    woke = []
    for seed in range(1, 7):
        core = _duck(seed)
        _to(core, 6, 50)
        assert core.s.napping_until is not None and core.s.nested
        _to(core, 7, 30)
        wake = _kinds(core, "duck.self_wake")
        assert len(wake) == 1
        woke.append(wake[0].occurred_at.time())
        assert time(7, 0) <= wake[0].occurred_at.time() <= time(7, 5, 59)
        end = next(e for e in _kinds(core, "duck.nap_end") if e.occurred_at == wake[0].occurred_at)
        assert end.payload["by"] == "clock"
        assert core.s.trace == f"day-{wake[0].occurred_at:%Y%m%d}"
    assert len({moment.minute for moment in woke}) >= 2  # a few minutes' jitter, drawn


def test_without_verified_time_it_does_not_wake_itself_but_the_family_can():
    core = _duck(4)
    core.set_setting("ntp_ok", False)
    _to(core, 7, 40)
    assert not _kinds(core, "duck.self_wake")
    assert core.s.napping_until is not None and core.s.nap_reason == "quiet_hours"
    _cues(core, 15)
    assert core.s.family == "awake" and core.s.napping_until is None
    assert _kinds(core, "duck.nap_end")[-1].payload["by"] in ("cue", "rested")
    assert core.s.phase != "night" and not _kinds(core, "duck.self_wake")


def test_a_sleeping_family_is_never_taken_to_have_gone_out():
    """A weekend morning: everyone sleeps late. Quiet is not a door: never asleep → away."""
    core = _duck(5)
    _to(core, 12, 0)  # nobody made a sound all morning
    assert ("asleep", "away") not in _transitions(core)
    assert core.s.family == "unknown"  # 11:00 and nothing since last night: not sure
    assert core.s.phase == "alone"  # (not sure counts as nobody home: it rests)


def test_a_voice_alone_does_not_bring_the_family_home_but_a_door_and_a_voice_do():
    core = _duck(6)
    _to(core, 9, 0)
    _cues(core, 20)
    core.hear_home(None, kind="door", journal=False)
    core.advance(int(R["away_after_door_min"]) + 3)
    assert core.s.family == "away" and _transitions(core)[-1] == ("awake", "away")
    for _ in range(4):  # a radio left on: voices only
        core.hear_home(None, kind="voice", journal=False)
        core.advance(6)
    assert core.s.family == "away"
    core.hear_home(None, kind="door", journal=False)
    core.hear_home(None, kind="voice", journal=False)
    core.advance(2)
    assert core.s.family == "awake"
    greeting = _kinds(core, "duck.greeting")[-1]
    assert greeting.payload["kind"] == "reunion"


def test_the_belief_follows_only_what_was_logged():
    """Every change of mind is explained by duck.heard summaries already in the log and
    the clock: going out after a door needs 20 quiet logged minutes, coming home needs a
    logged sighting or a door with a voice."""
    session = Session(Config(seed=8, nest=True, conservative=False, household="dual_income"))
    dispatch(session, "live", {"family": "dual_income", "days": 10})
    events = session.core.log.events
    cues: list[Event] = []
    for event in events:
        if event.kind == "duck.heard" and event.payload["counted"]:
            cues.append(event)
        if event.kind != "env.family" or "app" in event.payload["by"]:
            continue
        before = [cue for cue in cues if cue.occurred_at <= event.occurred_at]
        by, to = set(event.payload["by"]), event.payload["to"]
        if to == "away" and "door" in by:
            last = before[-1]
            assert "door" in last.payload["counted"]
            assert event.occurred_at - last.occurred_at >= timedelta(
                minutes=int(R["away_after_door_min"])
            )
        if to == "away" and by == {"quiet"}:
            assert event.occurred_at - before[-1].occurred_at >= timedelta(
                minutes=int(R["away_after_quiet_min"])
            )
        if to == "awake" and event.payload["from"] in ("away", "unknown"):
            recent = [
                cue for cue in before if event.occurred_at - cue.occurred_at <= timedelta(minutes=5)
            ]
            seen = {kind for cue in recent for kind in cue.payload["counted"]}
            assert "person" in seen or {"door", "voice"} <= seen, (event.payload, seen)
    assert {"away", "awake"} <= {e.payload["to"] for e in events if e.kind == "env.family"}


def test_while_everyone_is_out_it_goes_home_and_rests():
    core = _duck(9)
    core.advance(8 * 1440)  # grown enough to walk (the family's cues are missing: it rests)
    _to(core, 8, 0)
    _cues(core, 40)
    core.family_leave()
    awake_before = core.s.day_alone_awake_min
    core.advance(8 * 60)  # a working day
    awake_alone = core.s.day_alone_awake_min - awake_before
    assert core.s.phase == "alone"
    assert core.s.nested
    naps = [e for e in _kinds(core, "duck.nap_start") if e.payload["reason"] == "away_rest"]
    assert naps and all(45 <= e.payload["minutes"] <= 90 for e in naps)
    assert awake_alone <= 8 * 60 * 12 // 100  # awake ≤ ~10% of the time nobody is home
    walks = [
        e
        for e in _kinds(core, "duck.walk")
        if e.payload.get("family") in ("away", "unknown") and not e.payload.get("homing")
    ]
    assert not walks  # no outings while nobody is home


def test_it_learns_when_the_family_comes_home_and_waits_for_them():
    core = _duck(10)
    for _ in range(9):
        _to(core, 8, 0)
        _cues(core, 20)
        core.family_leave()
        _to(core, 18, 25)
        core.family_return()
        _cues(core, 60)
    learned = [
        e
        for e in _kinds(core, "learning.committed")
        if str(e.payload["parameter"]).startswith("self.return_time:")
    ]
    assert len(learned) >= 5
    # (its clever moments guess too: only the guesses about the family coming home here)
    predicts = [e for e in _kinds(core, "duck.predict") if e.payload["model"] == "return"]
    assert predicts, "after five returns it should wake up before the usual time"
    for predict in predicts:
        lead = -int(predict.payload["window"][0])
        assert 5 <= lead <= 20 and predict.payload["window"][1] == 45
        expect = int(predict.payload["expect_min"])
        minute = daycycle.minute_of_day(predict.occurred_at)
        assert expect - 21 <= minute <= expect - 4
    outcomes = [e for e in _kinds(core, "duck.predict_outcome") if e.payload["model"] == "return"]
    assert outcomes and all(e.payload["hit"] for e in outcomes)


def test_it_winds_down_at_half_past_nine_and_never_walks_at_night():
    session = Session(Config(seed=11, nest=True, conservative=False, household="hybrid"))
    dispatch(session, "live", {"family": "hybrid", "days": 14})
    core = session.core
    for walk in _kinds(core, "duck.walk"):
        at = walk.occurred_at
        assert not (at.hour >= 22 or at.hour < 7), at
    bedtimes = [e for e in _kinds(core, "duck.bedtime") if e.payload.get("in_nest")]
    assert len(bedtimes) >= 8
    for event in bedtimes:
        assert time(21, 0) <= event.occurred_at.time() < time(22, 0)
    night = [e for e in _kinds(core, "duck.phase") if e.payload["to"] == "night"]
    assert night and all(
        e.occurred_at.time() >= time(21, 0) or e.occurred_at.time() < time(7) for e in night
    )


def test_out_at_bedtime_and_unable_to_get_home_it_sleeps_where_it_is():
    core = _duck(12)
    core.advance(8 * 1440)
    _to(core, 21, 24)
    _cues(core, 2)
    if core.s.napping_until is not None:
        core.wake("owner")
    core.pickup("owner")
    core.set_setting("floor", True)  # set down in front of the nest…
    hands = LegacyActs(core, core.acts.port)  # (a hand on its head: the world detects it)
    for _ in range(12):  # …and held still so often that it keeps still until too late
        hands.hold_head("owner")
        if core.s.lockout_until is not None:
            break
        core.advance_seconds(31)
    assert core.s.lockout_until is not None
    _to(core, 22, 30)
    stranded = _kinds(core, "nest.stranded")
    assert stranded and stranded[-1].payload["reason"] == "bedtime"
    assert time(21, 55) <= stranded[-1].occurred_at.time() < time(22, 0)
    slept = [
        e
        for e in _kinds(core, "duck.nap_start")
        if e.occurred_at >= stranded[-1].occurred_at and e.payload["reason"] == "quiet_hours"
    ]
    assert slept  # it sleeps where it is (off its charger, the battery will not last)
    assert not [e for e in _kinds(core, "duck.walk") if e.occurred_at >= stranded[-1].occurred_at]
    assert _kinds(core, "owner.notice")[-1].payload["kind"] == "bedtime"


def test_a_nest_day_is_the_same_whatever_the_step_size():
    def run(chunks: list[int]) -> str:
        core = _duck(13)
        left = 2 * 1440
        index = 0
        while left > 0:
            step = min(chunks[index % len(chunks)], left)
            core.advance(step)
            left -= step
            index += 1
        return core.log.head

    whole = run([2 * 1440])
    assert run([1]) == whole
    assert run([7, 61, 13]) == whole


def test_a_month_with_a_family_replays_identically():
    session = Session(Config(seed=14, nest=True, household="school_family"))
    dispatch(session, "live", {"family": "school_family", "days": 30})
    dispatch(session, "act", {"action": "hear_home", "params": {"kind": "door"}})
    dispatch(session, "act", {"action": "family_leave"})
    dispatch(session, "advance", {"minutes": 90})
    assert dispatch(session, "replay_check", {})["identical"] is True
    assert datetime.combine(session.core.s.day_date, time(0)) > datetime(2027, 1, 15)


def test_someone_right_here_changes_the_belief_at_once():
    """An interaction is certain: no waiting for the next minute's look at the cues."""
    core = _duck(15)
    core.advance(8 * 1440)
    _to(core, 10, 0)  # nobody made a sound all morning: not sure anyone is home
    assert core.s.family in ("unknown", "asleep") and core.s.phase in ("alone", "solo")
    note = core.owner_walk()["note"]
    assert core.s.family == "awake" and core.s.phase == "family"
    assert list(_kinds(core, "env.family")[-1].payload["by"]) == ["person"]
    assert "走" in note


def test_what_it_heard_is_its_own_summary_naming_the_sensor_of_each_cue():
    """G37 (phase 4): the cues came in as inputs (hear_home); the logged summary is the
    duck's own (signed self, never a sensor's name) and says which sensor each kind of cue
    came from: the ear for sounds, the head camera for someone seen, BLE for a phone."""
    assert set(daycycle.CUE_SENSOR) == set(daycycle.CUE_KINDS)
    core = _duck(16)
    _to(core, 9, 0)
    for kind in ("voice", "steps", "person"):
        core.hear_home(None, kind=kind, journal=False)
    core.hear_home("mom", kind="phone_in", journal=False)
    core.advance(1)
    heard = _kinds(core, "duck.heard")
    assert heard and not _kinds(core, "env.cue")
    last = heard[-1]
    assert last.source.kind == "duck-self"
    assert last.payload["counts"] == {"person": 1, "phone_in:mom": 1, "steps": 1, "voice": 1}
    assert last.payload["sources"] == {
        "person": "camera",
        "phone_in": "ble",
        "steps": "ear",
        "voice": "ear",
    }
    for event in heard:
        bases = {str(key).split(":", 1)[0] for key in event.payload["counts"]}
        assert set(event.payload["sources"]) == bases


def test_someone_known_from_a_touch_is_not_signed_as_the_cameras():
    """G37 (phase-4 review): an interaction tells the duck someone is here too, but that cue
    came from the input (a touch, a lift, a word), never from a camera, least of all on a duck
    without one: its ``duck.heard`` names it ``contact``."""
    from phase4_harness import open_v41, send

    for camera in (False, True):
        session = open_v41({"seed": 77, "nest": True, "camera": camera})
        send(session, "live", {"family": "elder_home", "days": 2})
        heard = [e for e in session.core.log.events if e.kind == "duck.heard"]
        people = [e.payload["sources"].get("person") for e in heard]
        assert heard and any(p is not None for p in people)
        if not camera:
            assert not [p for p in people if p is not None and "camera" in p], people
        assert any(p is not None and "contact" in p for p in people)
        for event in heard:
            keys = {str(key) for key in event.payload["counts"]}
            if "person:contact" in keys:
                assert "contact" in event.payload["sources"]["person"]
                assert "有人在身邊" in describe(event) and "有人跟牠互動" in describe(event)
            if "person" in keys:
                assert "camera" in event.payload["sources"]["person"]
                assert "看到人" in describe(event) and "鏡頭" in describe(event)
