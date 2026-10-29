"""Live voice, LV2: what the duck perceives, as an ``attend`` hint is checked against it
(``duckmind.live.perception``). A hint with nothing behind it in the duck's own senses is a miss, a
name is matched to the family it knows, what cannot be checked (a colour) is reported and does not
block a match. No network."""

from __future__ import annotations

from typing import Any

import pytest
from live_bridge_rig import DAD_LOOK, KID_LOOK, MOM_LOOK, frame, person_sight, thing_sight

from duckmind.live import perception as P
from duckmind.neuro.genome import Genome
from duckmind.neuro.learn import Learner

pytestmark = pytest.mark.usefixtures("no_network")


def learner() -> Learner:
    return Learner(Genome.from_seed(4107, gd=60).p)


def see(
    per: P.Perception,
    lr: Learner,
    *sights: Any,
    now: float = 1.0,
    sound: tuple[str, float] | None = None,
) -> None:
    kind, db = sound if sound else (None, 30.0)
    per.update(
        frame(int(now * 1000), *sights, sound_kind=kind, level_db=db), lr, now, now, asleep=False
    )


def test_the_people_and_things_in_view_are_the_ducks_own_with_its_own_person_ids() -> None:
    per, lr = P.Perception(), learner()
    see(per, lr, person_sight(1, look=MOM_LOOK), thing_sight(7))
    assert [p.pid for p in per.people] == [1] and [t.track for t in per.things] == [7]
    see(per, lr, person_sight(5, look=MOM_LOOK), now=2.0)  # (a new eye track, the same look)
    assert per.people[0].pid == 1 and len(lr.people) == 1
    see(per, lr, person_sight(6, look=DAD_LOOK), now=3.0)
    assert per.people[0].pid == 2


def test_nothing_is_seen_with_the_camera_unusable_or_the_duck_asleep() -> None:
    per, lr = P.Perception(), learner()
    per.update(frame(0, person_sight(1)), lr, 1.0, 1.0, asleep=True)
    assert per.people == [] and lr.people == {}


def test_the_person_in_view_is_the_nearest_and_lingers_a_moment_after_they_leave() -> None:
    per, lr = P.Perception(), learner()
    see(
        per,
        lr,
        person_sight(1, range_m=2.0, look=MOM_LOOK),
        person_sight(2, range_m=0.9, look=KID_LOOK),
    )
    near, many = per.person_in_view(1.0)
    assert near is not None and near.track == 2 and many == 2
    see(per, lr, now=2.0)  # (they both left)
    last, none_now = per.person_in_view(1.0 + 0.5 * P.IN_VIEW_GRACE_S)  # (seen last at 1.0)
    assert last is not None and last.track == 2 and none_now == 0
    assert per.person_in_view(1.0 + 2 * P.IN_VIEW_GRACE_S) == (None, 0)


def test_a_thing_hint_matches_by_kind_and_reports_what_it_could_not_check() -> None:
    per, lr = P.Perception(), learner()
    see(per, lr, thing_sight(7, kind="toy", range_m=0.6), thing_sight(8, kind="thing", range_m=0.3))
    hit = per.resolve_attend("thing", "紅色的球", lr, 1.0)
    assert isinstance(hit, P.AttendTarget)
    assert hit.track == 7 and hit.matched == ("kind",) and hit.unverified == ("紅",)
    generic = per.resolve_attend("thing", "那個", lr, 1.0)
    assert isinstance(generic, P.AttendTarget) and generic.track == 8  # (the nearest: no kind said)
    other_languages = [
        per.resolve_attend("thing", w, lr, 1.0) for w in ("the red ball", "おもちゃ", "juguete")
    ]
    assert all(isinstance(t, P.AttendTarget) and t.track == 7 for t in other_languages)


def test_a_thing_that_is_not_in_view_is_a_miss() -> None:
    per, lr = P.Perception(), learner()
    see(per, lr, person_sight(1))  # a person, no thing
    assert per.resolve_attend("thing", "球", lr, 1.0) == P.AttendMiss("nothing_matches")


def test_a_person_hint_is_a_name_of_someone_it_knows_who_is_in_view_or_somebody() -> None:
    per, lr = P.Perception(), learner()
    mom = lr.see_person(1, MOM_LOOK, 0.5)
    lr.bind_name(mom, "媽媽", "妈妈", 0.6)
    see(
        per,
        lr,
        person_sight(1, look=MOM_LOOK, range_m=2.0),
        person_sight(2, look=KID_LOOK, range_m=0.8),
    )
    named = per.resolve_attend("person", "我的妈妈", lr, 1.0)
    assert isinstance(named, P.AttendTarget) and named.pid == mom and named.matched == ("name",)
    for hint in (None, "", "人", "說話的人", "someone"):
        who = per.resolve_attend("person", hint, lr, 1.0)
        assert isinstance(who, P.AttendTarget) and who.track == 2 and who.matched == ("generic",)
    assert per.resolve_attend("person", "小美", lr, 1.0) == P.AttendMiss("unknown_person")
    see(per, lr, person_sight(2, look=KID_LOOK), now=2.0)  # (mom is not in view now)
    assert per.resolve_attend("person", "媽媽", lr, 2.0) == P.AttendMiss("not_in_view")
    see(per, lr, now=3.0)
    assert per.resolve_attend("person", "人", lr, 3.0) == P.AttendMiss("nothing_matches")


def test_a_sound_hint_needs_a_sound_the_duck_heard_recently() -> None:
    per, lr = P.Perception(), learner()
    assert per.resolve_attend("sound", "聲音", lr, 1.0) == P.AttendMiss("nothing_matches")
    see(per, lr, sound=("voice", 60.0), now=1.0)
    voice = per.resolve_attend("sound", "說話的聲音", lr, 2.0)
    assert (
        isinstance(voice, P.AttendTarget)
        and voice.sound_kind == "voice"
        and voice.matched == ("kind",)
    )
    generic = per.resolve_attend("sound", "吃飯的聲音", lr, 2.0)
    assert isinstance(generic, P.AttendTarget) and generic.sound_kind == "voice"
    assert per.resolve_attend("sound", "敲門聲", lr, 2.0) == P.AttendMiss("nothing_matches")
    assert per.resolve_attend("sound", "聲音", lr, 2.0 + P.SOUND_RECENT_S + 1) == P.AttendMiss(
        "nothing_matches"
    )
    see(per, lr, sound=("door", 70.0), now=3.0)
    door = per.resolve_attend("sound", "敲門聲", lr, 3.5)
    assert isinstance(door, P.AttendTarget) and door.sound_kind == "door"


def test_a_sound_in_the_quiet_room_is_not_something() -> None:
    per, lr = P.Perception(), learner()
    see(per, lr, sound=("voice", P.MIN_SOUND_DB - 5))
    assert list(per.sounds) == []
