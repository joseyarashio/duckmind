"""Live voice, LV4: the two new drives, on their own (no brain, no world, no network).

想說話 (talk) rises with time alone and with family present but not engaging, satiates with exchanges,
is held at rest while a person engages it, and is suppressed by fear and sleepiness. 親近 (bond, per
person, in the learner's person record) rises with being called warmly by name, being talked to,
being petted and being introduced, falls with being startled by that person, decays slowly, and raises
the trust a reply, a start of its own and an approach get. What the real brain does with them is in
``tests/live_isolated/case_drives.py``."""

from __future__ import annotations

import math
from typing import Any

import pytest

from duckmind.live import drives as DV
from duckmind.live.drives import DriveParams, LiveDrives, Observation, TalkDrive
from duckmind.neuro.learn import PersonRecord

pytestmark = pytest.mark.usefixtures("no_network")

AXES = {"boldness": 0.5, "activity": 0.5, "sociability": 0.5, "tempo": 0.5}


class People:
    """The part of the learner the drives read: the people it knows."""

    def __init__(self, *pids: int) -> None:
        self.people = {pid: PersonRecord(pid=pid) for pid in pids}


def make(params: DriveParams | None = None, *pids: int, **axes: float) -> tuple[LiveDrives, People]:
    learner = People(*(pids or (1,)))
    drives = LiveDrives(params or DriveParams.neutral(), axes={**AXES, **axes}, vocal_rate=1.0)
    drives.learner = learner
    return drives, learner


def run(
    drives: LiveDrives,
    learner: People,
    seconds: float,
    *,
    t0: float = 0.0,
    dt: float = 0.1,
    **obs: Any,
) -> float:
    """Tick ``drives`` for ``seconds`` with the same observation; returns the end time."""
    t = t0
    base = {"in_view_pid": None, "person_pid": None}
    base.update(obs)
    for _ in range(int(round(seconds / dt))):
        t += dt
        drives.tick(learner, Observation(now=t, dt=dt, **base))
    return t


# ── the rules that are not parameters ───────────────────────────────────────────────────


def test_the_suppression_and_the_mute_are_pinned() -> None:
    assert DV.FEAR_SUPPRESS == 0.8 and DV.SLEEP_SUPPRESS == 0.7  # (the LV2 weights)
    assert DV.FEAR_MUTE == 0.5
    assert DV.BOND_TAU_S >= 3600.0  # (decays slowly: hours, not minutes)


def test_the_wish_is_suppressed_by_fear_and_by_sleepiness() -> None:
    d, _ = make()
    d.talk.value = 0.8
    calm = d.wish(0.0, 0.0)
    assert calm == pytest.approx(0.8)
    assert d.wish(0.5, 0.0) == pytest.approx(0.8 * (1 - 0.8 * 0.5))
    assert d.wish(0.0, 1.0) == pytest.approx(0.8 * (1 - 0.7))
    assert d.wish(1.0, 1.0) < 0.2 * calm
    assert d.wish(9.0, -3.0) == pytest.approx(d.wish(1.0, 0.0))  # (out of range is clipped)


# ── 想說話 ──────────────────────────────────────────────────────────────────────────────


def test_the_talk_drive_steps_are_exact_for_any_tick_rate() -> None:
    a, b = TalkDrive(value=0.2, rest=0.3), TalkDrive(value=0.2, rest=0.3)
    for _ in range(100):
        a.rise(0.01, 0.05)
    b.rise(1.0, 0.05)
    assert a.value == pytest.approx(b.value, abs=1e-12)
    a.value = b.value = 0.9
    for _ in range(100):
        a.relax(0.01, 30.0)
    b.relax(1.0, 30.0)
    assert a.value == pytest.approx(b.value, abs=1e-12)
    assert a.value > a.rest  # (not yet there)


def test_the_wish_rises_with_time_alone_toward_one_and_never_past() -> None:
    d, learner = make(DriveParams.neutral().with_values(talk_rise=0.02))
    start = d.talk.value
    run(d, learner, 20.0)
    assert d.alone and d.talk.value > start + 0.05
    run(d, learner, 600.0, t0=20.0)
    assert 0.99 < d.talk.value <= 1.0


def test_it_does_not_count_as_alone_until_it_has_been_a_while() -> None:
    d, learner = make(DriveParams.neutral().with_values(talk_rise=0.02))
    t = run(d, learner, 5.0, in_view_pid=1, person_pid=1, speech=True)
    run(d, learner, 4.0, t0=t)  # nobody in view for 4 s only
    assert not d.alone
    run(d, learner, 8.0, t0=t + 4.0)
    assert d.alone


def test_family_present_but_not_engaging_makes_it_rise_after_a_while() -> None:
    d, learner = make(DriveParams.neutral().with_values(talk_rise=0.02))
    t = run(d, learner, 6.0, in_view_pid=1, person_pid=1)
    assert not d.idle and d.talk.value == pytest.approx(d.talk.rest, abs=0.02)  # (just arrived)
    run(d, learner, 20.0, t0=t, in_view_pid=1, person_pid=1)
    assert d.idle and d.talk.value > d.talk.rest + 0.1


def test_a_person_who_speaks_to_it_stops_the_rise_and_lets_it_settle_then_idle_makes_it_rise() -> (
    None
):
    d, learner = make(DriveParams.neutral().with_values(talk_rise=0.05))
    t = run(d, learner, 30.0)  # alone: high
    high = d.talk.value
    assert high > d.talk.rest + 0.3
    t = run(d, learner, 1.0, t0=t, in_view_pid=1, person_pid=1, speech=True)
    t = run(d, learner, 6.0, t0=t, in_view_pid=1, person_pid=1)  # present, they have spoken
    assert not d.idle and d.talk.value < high  # (settling toward rest, not rising)
    settled = d.talk.value
    run(d, learner, 20.0, t0=t, in_view_pid=1, person_pid=1)  # and then nothing more
    assert d.idle and d.talk.value > settled


def test_a_conversation_keeps_it_near_rest() -> None:
    d, learner = make(DriveParams.neutral().with_values(talk_rise=0.05))
    t = 0.0
    for _ in range(12):  # an utterance every 6 s for a minute
        t = run(d, learner, 1.5, t0=t, in_view_pid=1, person_pid=1, speech=True)
        t = run(d, learner, 4.5, t0=t, in_view_pid=1, person_pid=1)
    assert not d.idle and d.talk.value < d.talk.rest + 0.05


def test_sleep_lets_it_go_back_to_rest() -> None:
    d, learner = make(DriveParams.neutral().with_values(talk_rise=0.05))
    t = run(d, learner, 30.0)
    assert d.talk.value > 0.6
    run(d, learner, 120.0, t0=t, asleep=True)
    assert d.talk.value == pytest.approx(d.talk.rest, abs=0.02)


def test_a_vocal_and_a_sociable_duck_wants_to_talk_more() -> None:
    quiet = LiveDrives(DriveParams.neutral(), axes={**AXES, "sociability": 0.2}, vocal_rate=0.8)
    loud = LiveDrives(DriveParams.neutral(), axes={**AXES, "sociability": 0.9}, vocal_rate=1.3)
    assert loud.rise_rate > quiet.rise_rate and loud.talk.rest > quiet.talk.rest
    keen = LiveDrives(
        DriveParams.neutral().with_values(talk_soc_k=3.0), axes={**AXES, "sociability": 0.9}
    )
    flat = LiveDrives(DriveParams.neutral(), axes={**AXES, "sociability": 0.9})
    assert keen.rise_rate > flat.rise_rate


# ── exchanges satiate it ─────────────────────────────────────────────────────────────────


def test_a_reply_and_a_persons_speech_within_the_window_is_an_exchange() -> None:
    p = DriveParams.neutral().with_values(talk_sat_exchange=0.3, talk_sat_tag=0.0)
    d, learner = make(p)
    d.talk.value = 0.8
    t = run(d, learner, 1.0, in_view_pid=1, person_pid=1, speech=True)  # they speak
    t = run(d, learner, 1.5, t0=t, in_view_pid=1, person_pid=1)
    d.reply_voiced(t, 7, 1, first_tag=True)  # the duck answers 1.5 s later
    assert d.counts["exchange"] == 1 and learner.people[1].exchanges == 1
    assert d.talk.value < 0.8 * 0.75


def test_speech_before_the_window_is_no_exchange_and_speech_after_it_is() -> None:
    d, learner = make(DriveParams.neutral().with_values(talk_sat_exchange=0.3, talk_sat_tag=0.0))
    d.talk.value = 0.8
    t = run(d, learner, 1.0, in_view_pid=1, person_pid=1, speech=True)
    t = run(d, learner, 9.0, t0=t, in_view_pid=1, person_pid=1)  # 9 s of quiet
    d.reply_voiced(t, 1, 1, first_tag=True)
    assert d.counts["exchange"] == 0 and 1 in d._pending_exchange  # (waiting to see if they answer)
    run(d, learner, 1.0, t0=t + 1.0, in_view_pid=1, person_pid=1, speech=True)  # they answer back
    assert d.counts["exchange"] == 1 and not d._pending_exchange


def test_nobody_answers_the_duck_and_the_exchange_never_happens() -> None:
    d, learner = make()
    d.reply_voiced(1.0, 1, 1, first_tag=True)
    run(d, learner, 10.0, t0=1.0, in_view_pid=1, person_pid=1)
    assert d.counts["exchange"] == 0 and not d._pending_exchange


def test_only_the_first_tag_of_a_reply_can_be_an_exchange_but_every_tag_takes_a_little() -> None:
    p = DriveParams.neutral().with_values(talk_sat_exchange=0.0, talk_sat_tag=0.1)
    d, learner = make(p)
    d.talk.value = 0.8
    d.reply_voiced(5.0, 1, 1, first_tag=True)
    d.reply_voiced(5.8, 1, 1, first_tag=False)
    assert d.talk.value == pytest.approx(0.8 * 0.9 * 0.9)
    assert not d._pending_exchange.get(1) or d._pending_exchange[1][0] == 5.0


def test_its_own_call_satisfies_it_a_little() -> None:
    d, _ = make(DriveParams.neutral().with_values(talk_sat_tag=0.2))
    d.talk.value = 0.5
    d.own_call()
    assert d.talk.value == pytest.approx(0.4)


def test_the_clouds_feel_pulls_the_wish_as_in_lv2() -> None:
    d, _ = make()
    before = d.talk.value
    d.feel_talk(0.0)
    assert d.talk.value == pytest.approx(before * (1 - 0.35))
    d.feel_talk(1.0)
    assert d.talk.value > before * 0.65


# ── 親近 ────────────────────────────────────────────────────────────────────────────────


def test_a_name_called_warmly_raises_the_bond_when_the_cloud_hears_a_warm_tone() -> None:
    d, learner = make(DriveParams.neutral().with_values(bond_g_name=0.2, bond_g_talk=0.0))
    t = run(d, learner, 1.0, in_view_pid=1, person_pid=1, speech=True, name=True)
    assert d.counts["name_heard"] == 1 and learner.people[1].bond == 0.0  # (not yet: tone unknown)
    d.on_feel(t + 1.0, 0.4)
    assert learner.people[1].bond == pytest.approx(0.2) and d.counts["called_warmly"] == 1
    d.on_feel(t + 1.1, 0.4)
    assert learner.people[1].bond == pytest.approx(0.2)  # (once per call)


def test_a_friendly_reply_proposal_is_also_a_warm_reading_and_an_unfriendly_one_is_not() -> None:
    d, learner = make(DriveParams.neutral().with_values(bond_g_name=0.2, bond_g_talk=0.0))
    t = run(d, learner, 1.0, in_view_pid=1, person_pid=1, speech=True, name=True)
    d.on_proposal(t + 0.9, "inquire")
    assert learner.people[1].bond == 0.0
    d.on_proposal(t + 1.0, "greet")
    assert learner.people[1].bond == pytest.approx(0.2)


def test_a_cold_tone_or_no_reading_at_all_gives_no_warmth() -> None:
    d, learner = make(DriveParams.neutral().with_values(bond_g_name=0.2, bond_g_talk=0.0))
    t = run(d, learner, 1.0, in_view_pid=1, person_pid=1, speech=True, name=True)
    d.on_feel(t + 1.0, -0.4)  # a cold reading drops the pending name call
    d.on_feel(t + 1.2, 0.4)
    assert learner.people[1].bond == 0.0
    t = run(d, learner, 1.0, t0=t + 5.0, in_view_pid=1, person_pid=1, speech=True, name=True)
    run(d, learner, 7.0, t0=t, in_view_pid=1, person_pid=1)  # nobody ever reads it
    d.on_feel(t + 8.0, 0.4)
    assert learner.people[1].bond == 0.0


def test_the_name_has_to_come_from_a_person_it_can_put_it_on() -> None:
    d, learner = make(DriveParams.neutral().with_values(bond_g_name=0.2, bond_g_talk=0.0))
    t = run(d, learner, 1.0, speech=True, name=True)  # nobody in view
    d.on_feel(t + 1.0, 0.4)
    assert learner.people[1].bond == 0.0 and d.counts["called_warmly"] == 0


def test_being_talked_to_raises_the_bond_once_for_each_utterance() -> None:
    d, learner = make(DriveParams.neutral().with_values(bond_g_talk=0.1))
    t = run(d, learner, 3.0, in_view_pid=1, person_pid=1, speech=True)  # one long utterance
    assert learner.people[1].bond == pytest.approx(0.1, rel=2e-3)  # (a hair of the slow decay)
    t = run(d, learner, 0.4, t0=t, in_view_pid=1, person_pid=1)  # a breath, not a new one
    t = run(d, learner, 1.0, t0=t, in_view_pid=1, person_pid=1, speech=True)
    assert learner.people[1].bond == pytest.approx(0.1, rel=2e-3)
    t = run(d, learner, 2.0, t0=t, in_view_pid=1, person_pid=1)
    run(d, learner, 1.0, t0=t, in_view_pid=1, person_pid=1, speech=True)  # a new one
    assert learner.people[1].bond == pytest.approx(0.1 + 0.9 * 0.1, rel=2e-3)
    assert d.counts["talked_to"] == 2


def test_a_voice_with_nobody_to_put_it_on_raises_nobodys_bond() -> None:
    d, learner = make(DriveParams.neutral().with_values(bond_g_talk=0.2))
    run(d, learner, 2.0, speech=True)
    assert learner.people[1].bond == 0.0 and d.counts["utterance"] == 1


def test_petting_raises_the_bond_by_the_second_and_is_one_event_per_touch() -> None:
    d, learner = make(DriveParams.neutral().with_values(bond_g_pet=0.05))
    t = run(d, learner, 4.0, in_view_pid=1, person_pid=1, gentle_touch=True)
    expected = 1.0 - (1.0 - 0.05 * 0.1) ** 40
    assert learner.people[1].bond == pytest.approx(expected, rel=0.02)
    assert d.counts["petted"] == 1
    t = run(d, learner, 2.0, t0=t, in_view_pid=1, person_pid=1)
    run(d, learner, 2.0, t0=t, in_view_pid=1, person_pid=1, gentle_touch=True)
    assert d.counts["petted"] == 2
    run(d, learner, 4.0, t0=0.0, person_pid=1, gentle_touch=True, asleep=True)
    assert d.counts["petted"] == 2  # (asleep, it is not petted)


def test_being_introduced_gives_a_first_bond_and_a_startle_takes_a_part_of_it() -> None:
    d, learner = make(DriveParams.neutral().with_values(bond_g_intro=0.2, bond_l_startle=0.5))
    d.introduced(1.0, 1)
    assert learner.people[1].bond == pytest.approx(0.2)
    d.startled(2.0, 1)
    assert learner.people[1].bond == pytest.approx(0.1)
    d.startled(3.0, 1)
    assert learner.people[1].bond == pytest.approx(0.05)  # (it never goes below zero)
    assert d.counts["startled"] == 2 and d.events[-1].kind == "startled"


def test_a_startle_with_nobody_to_blame_changes_nothing() -> None:
    d, learner = make()
    learner.people[1].bond = 0.6
    d.startled(1.0, None)
    d.startled(1.0, 99)  # (a person it does not know)
    assert learner.people[1].bond == 0.6 and d.counts["startled"] == 0


def test_a_startle_by_one_person_leaves_the_others_bond_alone() -> None:
    d, learner = make(None, 1, 2)
    learner.people[1].bond = learner.people[2].bond = 0.6
    d.startled(1.0, 2)
    assert learner.people[1].bond == 0.6 and learner.people[2].bond < 0.6


def test_bond_gains_are_a_fraction_of_what_is_left_and_never_pass_one() -> None:
    d, learner = make(DriveParams.neutral().with_values(bond_g_intro=0.5))
    for k in range(40):
        d.introduced(float(k), 1)
    assert 0.999 < learner.people[1].bond <= 1.0


def test_the_bond_decays_slowly_over_hours_not_minutes() -> None:
    d, learner = make()
    learner.people[1].bond = 0.8
    run(d, learner, 600.0, dt=10.0)  # ten minutes
    assert learner.people[1].bond == pytest.approx(0.8 * math.exp(-600 / DV.BOND_TAU_S), rel=1e-3)
    assert learner.people[1].bond > 0.75
    run(d, learner, 4 * 3600.0, t0=600.0, dt=60.0)
    assert 0.2 < learner.people[1].bond < 0.3


# ── what the bond does ───────────────────────────────────────────────────────────────────


def test_trust_is_the_strangers_floor_at_no_bond_and_full_at_the_full_bond() -> None:
    p = DriveParams.neutral().with_values(stranger_floor=0.2, bond_full=0.5)
    d, learner = make(p)
    assert d.trust(1) == pytest.approx(0.2)
    learner.people[1].bond = 0.25
    assert d.trust(1) == pytest.approx(0.6)
    learner.people[1].bond = 0.9
    assert d.trust(1) == pytest.approx(1.0)
    assert d.trust(None) == 1.0 and d.trust(404) == 1.0  # (a voice with nobody behind it)


def test_a_bold_duck_gives_a_stranger_more_than_a_shy_one() -> None:
    p = DriveParams.neutral().with_values(stranger_floor=0.4, stranger_bold_k=2.0)
    bold, shy = make(p, boldness=0.9)[0], make(p, boldness=0.1)[0]
    assert bold.stranger_floor() > 0.4 > shy.stranger_floor()
    assert bold.trust(1) > shy.trust(1)
    flat = make(DriveParams.neutral().with_values(stranger_floor=0.4), boldness=0.9)[0]
    assert flat.stranger_floor() == pytest.approx(0.4)


def test_the_reply_scale_hesitation_and_drive_follow_the_temperament_and_the_wish() -> None:
    p = DriveParams.neutral().with_values(
        reply_scale=1.5,
        reply_soc_k=2.0,
        reply_act_k=-2.0,
        hesitate_s=1.0,
        hesitate_tempo_k=3.0,
        reply_wish_gain=0.5,
    )
    lively = make(p, sociability=0.9, activity=0.9, tempo=0.9)[0]
    dreamy = make(p, sociability=0.4, activity=0.1, tempo=0.1)[0]
    assert dreamy.reply_scale() > lively.reply_scale()  # (a busy duck answers less with this k)
    assert dreamy.hesitation_s() > lively.hesitation_s() > 0.0
    assert make(DriveParams.neutral())[0].hesitation_s() == 0.0
    d = make(p)[0]
    d.talk.value = d.talk.rest + 0.2
    high = d.reply_drive(0.5, 0.0)
    d.talk.value = d.talk.rest - 0.1
    assert d.reply_drive(0.5, 0.0) < high
    assert d.reply_drive(0.5, 0.0) == pytest.approx(0.35 + 0.45 * 0.5 - 0.5 * 0.1)


def test_a_reply_goes_on_to_its_next_tag_only_while_the_wish_lasts() -> None:
    d, _ = make(DriveParams.neutral().with_values(continue_wish=0.4))
    d.talk.value = 0.6
    assert d.continues(0.0, 0.0)
    assert not d.continues(0.9, 0.0)  # (fear takes the wish away)
    d.talk.value = 0.3
    assert not d.continues(0.0, 0.0)
    assert make(DriveParams.neutral())[0].continues(1.0, 1.0)  # (continue_wish 0: always)


def test_the_wish_to_start_a_conversation_is_zero_without_a_person_and_grows_with_wish_and_trust() -> (
    None
):
    p = DriveParams.neutral().with_values(
        init_gain=0.8, init_threshold=0.5, stranger_floor=0.3, bond_full=0.5
    )
    d, learner = make(p)
    d.talk.value = 0.9
    assert d.initiation(None, 0.0, 0.0) == 0.0
    stranger = d.initiation(1, 0.0, 0.0)
    learner.people[1].bond = 0.5
    friend = d.initiation(1, 0.0, 0.0)
    assert 0.0 < stranger < friend == pytest.approx(0.8 * 0.8)
    d.talk.value = 0.4
    assert d.initiation(1, 0.0, 0.0) == 0.0  # (below the threshold)
    d.talk.value = 0.9
    assert d.initiation(1, 0.9, 0.0) < friend  # (afraid: much less)
    assert make(DriveParams.neutral())[0].initiation(1, 0.0, 0.0) == 0.0  # (init_gain 0)


def test_the_wish_to_go_to_someone_needs_a_bond_and_a_recent_word_or_touch() -> None:
    p = DriveParams.neutral().with_values(approach_gain=0.3, bond_full=0.5)
    d, learner = make(p)
    learner.people[1].bond = 0.5
    assert d.approach(1, 10.0) == 0.0  # (they have not spoken to it)
    t = run(d, learner, 1.0, in_view_pid=1, person_pid=1, speech=True)
    assert d.approach(1, t + 5.0) == pytest.approx(0.3)
    assert d.approach(1, t + 40.0) == 0.0  # (a long time ago)
    learner.people[1].bond = 0.1
    assert d.approach(1, t + 5.0) == pytest.approx(0.06)
    assert d.approach(None, t) == 0.0
    assert make(DriveParams.neutral())[0].approach(1, t) == 0.0


# ── the parameters ───────────────────────────────────────────────────────────────────────


def test_every_searched_parameter_is_a_drive_parameter_with_a_box_around_a_value() -> None:
    names = {f for f in DriveParams.neutral().as_dict()}
    assert {s.name for s in DV.SEARCH_SPACE} <= names
    assert len({s.name for s in DV.SEARCH_SPACE}) == len(DV.SEARCH_SPACE)
    for spec in DV.SEARCH_SPACE:
        assert spec.low < spec.high
        assert spec.value(0.0) == pytest.approx(spec.low) and spec.value(1.0) == pytest.approx(
            spec.high
        )
        assert spec.value(-5.0) == pytest.approx(spec.low)  # (the box is a box)
        assert spec.unit(spec.value(0.37)) == pytest.approx(0.37, abs=1e-9)


def test_the_neutral_parameters_sit_inside_the_box_and_make_the_vector_a_round_trip() -> None:
    p = DriveParams.neutral()
    vector = DV.vector_from_params(p)
    assert len(vector) == len(DV.SEARCH_SPACE) and all(0.0 <= v <= 1.0 for v in vector)
    back = DV.params_from_vector(vector, p)
    for spec in DV.SEARCH_SPACE:
        assert getattr(back, spec.name) == pytest.approx(getattr(p, spec.name), rel=1e-9, abs=1e-9)
    with pytest.raises(ValueError):
        DV.params_from_vector([0.5] * 3)


def test_parameters_round_trip_through_a_dict_and_refuse_a_name_they_do_not_have() -> None:
    p = DriveParams().with_values(bond_g_name=0.31, init_gain=0.77)
    assert DriveParams.from_dict(p.as_dict()) == p
    with pytest.raises(ValueError):
        DriveParams.from_dict({"bond_g_nome": 1.0})


def test_neutral_is_lv2s_weights_and_changes_nothing_that_lv2_did_not_have() -> None:
    n = DriveParams.neutral()
    assert (n.reply_scale, n.reply_soc_k, n.reply_act_k, n.reply_wish_gain) == (1.0, 0.0, 0.0, 0.3)
    assert (n.hesitate_s, n.continue_wish, n.init_gain, n.approach_gain) == (0.0, 0.0, 0.0, 0.0)
    assert n.stranger_floor == 0.85 and n.stranger_bold_k == 0.0


def test_the_view_says_the_wish_and_every_persons_bond() -> None:
    d, learner = make(None, 1, 2)
    learner.people[2].bond = 0.4
    d.talk.value = 0.7
    view = d.view(fear=0.5, sleepiness=0.0)
    assert view["talk"] == 0.7 and view["wish"] == pytest.approx(0.7 * 0.6, abs=1e-3)
    assert [(p["pid"], p["bond"]) for p in view["people"]] == [(1, 0.0), (2, 0.4)]
