"""Live voice, LV2: the bridge (``duckmind.live.bridge``) on a real integrated brain. What the cloud
proposes becomes a candidate in the duck's own competition, weighted by its drives; it can lose;
reflexes still go first; a won call goes out through the existing vocal path with its own
``vocal_seq``; names bind only to a person the duck sees; a person who speaks over a call stops it.
No network, no physics."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from live_bridge_rig import (
    DAD_LOOK,
    MOM_LOOK,
    frame,
    make_brain,
    make_call,
    mark,
    person_sight,
    run_brain,
    tag,
    thing_sight,
    vocalize,
)

from duckmind.live import bridge as B
from duckmind.live import gate as G
from duckmind.live import vad as VD
from duckmind.live.drives import DriveParams
from duckmind.live.scenario import ManualClock
from duckmind.live.soul import EAR_NAME_HINT
from duckmind.neuro.brain import HOLD_BEHAVIOUR
from duckmind.neuro.integration import IntegratedBrain

pytestmark = pytest.mark.usefixtures("no_network")

EXPLORER = (4107, 60)
SNUGGLER = (1728, 20)
DREAMER = (4670, 200)
HATCHLING = (3341, 7)


def rig(
    who: tuple[int, int] = SNUGGLER, **config: Any
) -> tuple[IntegratedBrain, B.LiveBridge, Any]:
    brain, clock = make_brain(*who)
    bridge = B.LiveBridge(None, clock=clock, config=B.BridgeConfig(**config))
    bridge.attach(brain)
    return brain, bridge, clock


def at(clock: ManualClock) -> int:
    return int(round(clock.t * 1000))


class StubSession:
    """The part of a session the bridge reads: the cloud switch and the queue of calls."""

    def __init__(self, clock: ManualClock) -> None:
        self.clock = clock
        self.enabled = True
        self.state = "online"
        self.calls: list[Any] = []

    def take_calls(self, limit: int | None = None) -> list[Any]:
        out, self.calls = self.calls, []
        return out


# ── a proposal competes, and when it wins it is voiced through the vocal path ───────────


def test_a_social_duck_voices_a_proposed_greeting_through_the_existing_vocal_path() -> None:
    brain, bridge, clock = rig(SNUGGLER)
    run_brain(brain, clock, 20.0)
    bridge.submit(vocalize([tag("greet", 0.6, 0)], t=clock.t, seq=1))
    cmds = run_brain(brain, clock, 6.0)
    assert bridge.voiced and bridge.voiced[0]["tag"] == "greet"
    won = next(c for c in cmds if c.vocal == "greet" and c.reflex is None)
    assert won.behaviour in ("greet", "call_response")  # (the act that carried it)
    assert 0.2 <= won.vocal_level <= 1.0
    assert bridge.counts["intent:done"] == 1 and bridge.entered[0]["tool"] == "vocalize"
    # (it came through the gate and into the selector: both logged)
    assert [e["decision"] for e in bridge.log if e["tool"] == "vocalize"][:2] == [
        "accepted",
        "done",
    ]


def test_one_call_per_sequence_number() -> None:
    brain, bridge, clock = rig(HATCHLING)
    run_brain(brain, clock, 10.0)
    for k in range(3):
        bridge.submit(
            vocalize([tag("chirp", 0.5, 0), tag("coo", 0.4, 600)], seq=k + 1, t=clock.t + 40.0 * k)
        )
    cmds = run_brain(brain, clock, 120.0, hz=10.0)
    seqs = [c.vocal_seq for c in cmds]
    assert seqs == sorted(seqs)  # (never goes back)
    voiced = [(c.vocal_seq, c.vocal) for c in cmds if c.vocal]
    assert len({s for s, _ in voiced}) == len(voiced)  # (each call has a number of its own)
    assert brain._vocal_seq == max(seqs)
    assert len(bridge.voiced) >= 2


def test_a_lost_competition_voices_nothing_and_lapses() -> None:
    brain, bridge, clock = rig(SNUGGLER)

    def scared(t: float, cmd: Any) -> None:
        brain.drives.fear = 0.95

    run_brain(brain, clock, 20.0, on_tick=scared)
    bridge.submit(vocalize([tag("greet", 0.5, 0)], t=clock.t, seq=1))
    run_brain(brain, clock, 8.0, on_tick=scared)
    assert bridge.voiced == []
    assert bridge.counts["intent:expired"] == 1 and bridge.reasons[B.LOST] == 1
    assert bridge.intents == []


def test_the_same_proposal_wins_for_the_same_duck_when_it_is_not_afraid() -> None:
    brain, bridge, clock = rig(SNUGGLER)
    run_brain(brain, clock, 20.0)
    bridge.submit(vocalize([tag("greet", 0.5, 0)], t=clock.t, seq=1))
    run_brain(brain, clock, 8.0)
    assert bridge.voiced and bridge.counts["intent:expired"] == 0


def test_a_sleepy_duck_in_the_night_does_not_answer() -> None:
    brain, bridge, clock = rig(SNUGGLER)

    def night(t: float) -> Any:
        return frame(int(round(t * 1000)), day_minute=60.0)  # one in the morning

    def tired(t: float, cmd: Any) -> None:
        brain.drives.sleep_pressure = 1.0

    run_brain(brain, clock, 30.0, frames=night, on_tick=tired)
    bridge.submit(vocalize([tag("greet", 0.9, 0)], t=clock.t, seq=1))
    cmds = run_brain(brain, clock, 8.0, frames=night, on_tick=tired)
    assert bridge.voiced == [] and not any(
        c.vocal == "greet" and c.behaviour == "greet" for c in cmds
    )


def test_the_candidates_weight_follows_the_ducks_own_drives() -> None:
    brain, bridge, clock = rig(SNUGGLER)
    run_brain(brain, clock, 5.0)
    intent = B.Intent(1, "reply", 1, clock.t, 5.0, ("greet",), level=0.6)
    brain.drives.social, brain.drives.fear, brain.drives.sleep_pressure = 0.7, 0.0, 0.0
    social = bridge._weight(intent, brain)
    brain.drives.social = 0.1
    aloof = bridge._weight(intent, brain)
    brain.drives.social, brain.drives.fear = 0.7, 0.9
    afraid = bridge._weight(intent, brain)
    assert social > aloof > 0.0 and afraid < 0.35 * social
    loud = B.Intent(2, "reply", 1, clock.t, 5.0, ("greet",), level=0.95)
    soft = B.Intent(3, "reply", 1, clock.t, 5.0, ("greet",), level=0.1)
    brain.drives.fear = 0.0
    assert bridge._weight(loud, brain) > bridge._weight(soft, brain)


def test_the_urge_to_answer_grows_while_a_person_waits_and_stops_growing() -> None:
    brain, bridge, clock = rig(SNUGGLER)
    run_brain(brain, clock, 5.0)
    intent = B.Intent(1, "reply", 1, 100.0, 5.0, ("greet",), level=0.6)
    brain.drives.social, brain.drives.fear, brain.drives.sleep_pressure = 0.5, 0.0, 0.0
    cfg = bridge.config
    first = bridge._weight(intent, brain, 100.0)
    assert first == bridge._weight(intent, brain)  # (no clock given: as the proposal came)
    half = bridge._weight(intent, brain, 100.0 + cfg.reply_urge_s / 2)
    full = bridge._weight(intent, brain, 100.0 + cfg.reply_urge_s)
    later = bridge._weight(intent, brain, 100.0 + 4 * cfg.reply_urge_s)
    assert first < half < full == later
    assert full == pytest.approx(cfg.reply_urge * first)


def test_a_frightened_duck_does_not_answer_however_long_the_person_waits() -> None:
    brain, bridge, clock = rig(SNUGGLER)
    run_brain(brain, clock, 5.0)
    intent = B.Intent(1, "reply", 1, 100.0, 5.0, ("greet",), level=0.9)
    brain.drives.social, brain.drives.sleep_pressure = 0.9, 0.0
    cfg = bridge.config
    weights = []
    for fear in (0.0, cfg.fear_calm, 0.5, cfg.fear_gone, 0.95):
        brain.drives.fear = fear
        weights.append(bridge._weight(intent, brain, 100.0 + cfg.reply_urge_s))
    assert weights == sorted(weights, reverse=True)  # (the more afraid, the less it answers)
    assert weights[3] == 0.0 and weights[4] == 0.0
    assert weights[2] < 0.5 * weights[0]


def test_a_calm_duck_that_has_other_things_on_answers_after_a_short_wait() -> None:
    # (live, real physics: the dreamer and the explorer lost every reply the cloud proposed, the
    # commitment of the act they were in over a fixed input; the urge that grows turns that around)
    voiced = {"dreamer": 0, "hatchling": 0}
    for name, who in (("dreamer", DREAMER), ("hatchling", HATCHLING)):
        for warm in (8.0, 15.0, 25.0, 35.0, 45.0, 60.0):
            brain, bridge, clock = rig(who)
            run_brain(brain, clock, warm)
            bridge.submit(vocalize([tag("greet", 0.5, 0)], t=clock.t, seq=1))
            run_brain(brain, clock, 6.0)
            voiced[name] += bool(bridge.voiced)
    assert voiced == {"dreamer": 6, "hatchling": 6}


def test_personality_shows_in_who_answers() -> None:
    # (the input as it came, before the urge to answer has grown: the social ones always answer,
    # the explorer, often out exploring, can lose it.) This is the plain competition: since
    # 2026-10-04 a person's answer claims the body from a bout that is not done
    # (``reply_claims``, select.CLAIM_MIN), so the claim is switched off here, the test being about
    # what the competition does with an input of that size; the next test says what the claim does
    wins: dict[str, int] = {}
    for name, who in (("snuggler", SNUGGLER), ("hatchling", HATCHLING), ("explorer", EXPLORER)):
        wins[name] = 0
        for warm in (8.0, 25.0, 45.0):
            brain, bridge, clock = rig(who, reply_urge=1.0, reply_claims=False)
            run_brain(brain, clock, warm)
            bridge.submit(vocalize([tag("greet", 0.5, 0)], t=clock.t, seq=1))
            run_brain(brain, clock, 6.0)
            wins[name] += bool(bridge.voiced)
    assert wins["snuggler"] == wins["hatchling"] == 3  # (the social ones always answer)
    assert wins["explorer"] < 3  # (the explorer is often out exploring: a candidate can lose)


def test_a_persons_answer_claims_the_body_so_no_duck_loses_it_to_the_act_it_is_in() -> None:
    # (the same input, the urge off, the claim on: the explorer that lost it above answers every
    # time, and at once — the hesitation LV4 fitted still decides when the first tag is due)
    for name, who in (("snuggler", SNUGGLER), ("hatchling", HATCHLING), ("explorer", EXPLORER)):
        for warm in (8.0, 25.0, 45.0):
            brain, bridge, clock = rig(who, reply_urge=1.0)
            assert bridge.config.reply_claims
            run_brain(brain, clock, warm)
            bridge.submit(vocalize([tag("greet", 0.5, 0)], t=clock.t, seq=1))
            run_brain(brain, clock, 6.0)
            assert bridge.voiced, (name, warm)


def test_personality_shows_in_how_soon_it_answers() -> None:
    # (with the urge to answer growing, every calm duck answers in the end: the social ones at
    # once, the explorer and the dreamer when the urge has grown over what they are doing.
    # Measured with the neutral drives (LV2's weights, no fitted hesitation): this is the urge's
    # effect on the competition; what LV4's fitted hesitation adds is the next test)
    # (the plain competition again — ``reply_claims=False``: since 2026-10-04 a person's answer claims
    # the body, and then every calm duck answers at once whatever it is doing)
    delay: dict[str, list[float]] = {}
    for name, who in (
        ("snuggler", SNUGGLER),
        ("hatchling", HATCHLING),
        ("dreamer", DREAMER),
        ("explorer", EXPLORER),
    ):
        delay[name] = []
        for warm in (8.0, 25.0, 45.0):
            brain, bridge, clock = rig(who, drives=DriveParams.neutral(), reply_claims=False)
            run_brain(brain, clock, warm)
            asked = clock.t
            bridge.submit(vocalize([tag("greet", 0.5, 0)], t=clock.t, seq=1))
            run_brain(brain, clock, 6.0)
            assert bridge.voiced, (name, warm)
            delay[name].append(bridge.voiced[0]["t"] - asked)
    mean = {name: sum(v) / len(v) for name, v in delay.items()}
    social = max(mean["snuggler"], mean["hatchling"])
    assert social < 1.0
    assert mean["explorer"] > 1.3 * social
    assert mean["dreamer"] > 1.3 * social


def test_with_the_fitted_drives_every_duck_answers_after_its_own_hesitation() -> None:
    # (LV7: LV4's drives and the urge that grows together. The fitted hesitation sets when the first
    # tag is due, by the duck's tempo (its median is the longest for the dreamer: LV4 target 4);
    # the urge keeps the slow and the bold from losing the reply to the act they are in, so that
    # none of them lapses: LV4 alone left the dreamer and the explorer with 0 of 3 answered here)
    delay: dict[str, list[float]] = {}
    hesitation: dict[str, float] = {}
    for name, who in (
        ("snuggler", SNUGGLER),
        ("hatchling", HATCHLING),
        ("dreamer", DREAMER),
        ("explorer", EXPLORER),
    ):
        delay[name] = []
        for warm in (8.0, 25.0, 45.0):
            brain, bridge, clock = rig(who)
            run_brain(brain, clock, warm)
            asked = clock.t
            bridge.submit(vocalize([tag("greet", 0.5, 0)], t=clock.t, seq=1))
            run_brain(brain, clock, 6.0)
            assert bridge.voiced, (name, warm)
            assert bridge.drives is not None
            hesitation[name] = bridge.drives.hesitation_s()
            delay[name].append(bridge.voiced[0]["t"] - asked)
    for name, waits in delay.items():
        for wait in waits:
            assert hesitation[name] - 1e-9 <= wait <= hesitation[name] + 0.6, (name, wait)
    mean = {name: sum(v) / len(v) for name, v in delay.items()}
    assert all(mean["dreamer"] > mean[name] for name in ("snuggler", "hatchling", "explorer"))


def test_the_reply_and_the_ducks_own_answer_are_never_two_calls_a_moment_apart() -> None:
    # (live: a duck waiting to answer a hello answered with its own greeting the tick after the
    # reply won, 20 ms apart: two calls on top of each other in its microphone; the reply is its
    # answer, and a call of its own holds the next one back a moment)
    for who in (SNUGGLER, HATCHLING, EXPLORER):
        for warm in (8.0, 25.0, 45.0):
            brain, bridge, clock = rig(who)
            run_brain(brain, clock, warm)
            brain._answer = 1.8  # (it is waiting to answer a hello)
            t0 = clock.t
            bridge.submit(vocalize([tag("greet", 0.6, 0)], t=clock.t, seq=1))
            cmds = run_brain(brain, clock, 8.0)
            seen, starts = cmds[0].vocal_seq - 1, []
            for k, c in enumerate(cmds):
                if c.vocal_seq != seen:
                    seen = c.vocal_seq
                    if c.vocal:
                        starts.append(t0 + (k + 1) / 20.0)
            assert bridge.voiced, (who, warm)
            mine = bridge.voiced[0]["t"]
            near = [s for s in starts if abs(s - mine) > 1e-6 and abs(s - mine) < 0.4]
            assert near == [], (who, warm, mine, starts)


def test_a_drowsy_duck_does_not_answer_however_long_the_person_waits() -> None:
    brain, bridge, _clock = rig(SNUGGLER)
    intent = B.Intent(1, "reply", 1, 100.0, 5.0, ("greet",), level=0.9)
    cfg = bridge.config
    weights = []
    for sleepy in (0.0, cfg.drowsy_from, 0.7, cfg.drowsy_gone, 1.0):
        drives = SimpleNamespace(
            fear=0.0, social=0.9, valence=0.0, sleepiness=sleepy, curiosity=0.5
        )
        # (the drives are the brain's own state: the stand-in brain keeps the real one's genome, which
        # the bridge's LV4 drives are built from)
        stand_in = SimpleNamespace(drives=drives, genome=brain.genome)
        weights.append(bridge._weight(intent, stand_in, 100.0 + 10.0))
    assert weights == sorted(weights, reverse=True) and weights[3] == 0.0 == weights[4]
    assert weights[2] < 0.5 * weights[0]


def test_the_urge_to_answer_a_stranger_can_be_kept_out_so_the_strangers_floor_holds() -> None:
    # (LV7: the urge to answer makes every duck answer whoever waits, which overruns LV4's reserve
    # toward a person the family has not named; ``reply_urge_stranger`` is how much of the urge such
    # a person gets: none by default, all of it (1.0) is what the playable branch had)
    assert B.BridgeConfig().reply_urge_stranger == 0.0
    ratio: dict[tuple[float, bool], float] = {}
    for share in (1.0, 0.0):
        for known in (True, False):
            brain, bridge, _clock = rig(SNUGGLER, reply_urge_stranger=share)
            intent = B.Intent(
                1, "reply", 1, 100.0, 5.0, ("greet",), level=0.6, addressee=7, known=known
            )
            early = bridge._weight(intent, brain, 100.0)
            late = bridge._weight(intent, brain, 100.0 + bridge.config.reply_urge_s)
            ratio[(share, known)] = late / early
    urge = B.BridgeConfig().reply_urge
    assert ratio[(1.0, True)] == pytest.approx(urge)
    assert ratio[(1.0, False)] == pytest.approx(urge)  # (all of it: a stranger is urged too)
    assert ratio[(0.0, True)] == pytest.approx(urge)  # (a person it knows always is)
    assert ratio[(0.0, False)] == pytest.approx(1.0)  # (a stranger, kept out: the weight stays)


# ── reflexes first ─────────────────────────────────────────────────────────────────────


def clap_frames(at_s: float) -> Any:
    def make(t: float) -> Any:
        hit = abs(t - at_s) < 0.026
        return frame(
            int(round(t * 1000)), sound_kind="clap" if hit else None, level_db=88.0 if hit else 30.0
        )

    return make


def first_reflex(cmds: list[Any]) -> int | None:
    return next((i for i, c in enumerate(cmds) if c.reflex is not None), None)


def test_a_reflex_takes_the_body_on_the_same_tick_with_or_without_the_cloud_waiting() -> None:
    plain, clock_a = make_brain(*SNUGGLER)
    run_brain(plain, clock_a, 10.0)
    control = run_brain(plain, clock_a, 6.0, frames=clap_frames(clock_a.t + 2.0))

    brain, bridge, clock = rig(SNUGGLER, reply_weight=30.0)  # (a proposal that wants it very much)
    run_brain(brain, clock, 10.0)
    bridge.submit(vocalize([tag("greet", 1.0, 0)], t=clock.t, seq=1))
    live = run_brain(brain, clock, 6.0, frames=clap_frames(clock.t + 2.0))

    k = first_reflex(control)
    assert k is not None and first_reflex(live) == k  # (the reflex never waited for the cloud)
    assert live[k].behaviour in HOLD_BEHAVIOUR.values()
    held = [i for i, c in enumerate(live) if c.reflex is not None]
    # nothing the cloud proposed is voiced while a reflex has the body
    assert all(live[i].vocal is None or live[i].reflex is None for i in range(len(live)))
    assert held and bridge.faults == 0


# ── attend ─────────────────────────────────────────────────────────────────────────────


def scene(*sights: Any, sound: tuple[str, float] | None = None) -> Any:
    def make(t: float) -> Any:
        kind, db = sound if sound else (None, 30.0)
        return frame(int(round(t * 1000)), *sights, sound_kind=kind, level_db=db)

    return make


def test_attend_points_at_a_thing_the_duck_itself_sees_and_only_that() -> None:
    brain, bridge, clock = rig(HATCHLING)
    toy = thing_sight(7, range_m=0.6)
    far = thing_sight(8, kind="thing", range_m=1.4, bearing=0.3)
    seen = scene(toy, far)
    run_brain(brain, clock, 10.0, frames=seen)
    bridge.submit(make_call("attend", {"target_kind": "thing", "hint": "紅色的球"}, t=clock.t))
    run_brain(brain, clock, 0.3, frames=seen)
    assert brain._live_attend == ("thing", 7)  # (the nearest toy: a track in its own view)
    entered = bridge.entered[-1]
    assert entered["target"]["track"] == 7 and entered["target"]["unverified"] == ["紅"]
    assert "investigate" in bridge.last_extras
    run_brain(brain, clock, 8.0, frames=seen)
    assert brain._live_attend is None and bridge.intents == []  # (and it lets go)


def test_attend_with_nothing_there_is_refused_not_made_up() -> None:
    brain, bridge, clock = rig(HATCHLING)
    run_brain(brain, clock, 5.0)  # (an empty room)
    asked = (("thing", "紅色的球"), ("person", "人"), ("sound", "敲門聲"))
    for i, (kind, hint) in enumerate(asked):
        bridge.submit(
            make_call("attend", {"target_kind": kind, "hint": hint}, t=clock.t, seq=i + 1)
        )
    run_brain(brain, clock, 0.3)
    assert bridge.entered == [] and bridge.last_extras == {} and brain._live_attend is None
    refused = [e for e in bridge.log if e["decision"] == "refused"]
    assert len(refused) == 3 and {e["stage"] for e in refused} == {"bridge"}
    assert {e["reason"] for e in refused} == {"nothing_matches"}
    # a person nobody has named is not one it can point at either
    run_brain(brain, clock, 11.0)  # (the cap on how often it may be asked has passed)
    bridge.submit(make_call("attend", {"target_kind": "person", "hint": "媽媽"}, t=clock.t, seq=9))
    run_brain(brain, clock, 0.3)
    assert [e["reason"] for e in bridge.log if e["tool"] == "attend"][-1] == "unknown_person"


def test_attend_to_a_named_person_picks_that_person_among_the_ones_in_view() -> None:
    brain, bridge, clock = rig(SNUGGLER)
    mom = person_sight(1, range_m=1.8, bearing=0.2, look=MOM_LOOK)
    dad = person_sight(2, range_m=1.0, bearing=-0.3, look=DAD_LOOK)  # (nearer)
    seen = scene(mom, dad)
    run_brain(brain, clock, 6.0, frames=seen)
    bridge.submit(make_call("recognize", {"name": "媽媽"}, t=clock.t, seq=1))  # (introduced)
    run_brain(brain, clock, 0.2, frames=seen)
    names = brain.learn.person_names()
    # two people are in view: it cannot tell whom they mean, so the name waits
    assert names == {} and any(e["reason"] == B.SEVERAL_IN_VIEW for e in bridge.log)
    only_mom = scene(mom)
    run_brain(brain, clock, 25.0, frames=only_mom)  # (dad has left)
    bridge.submit(make_call("recognize", {"name": "妈妈"}, t=clock.t, seq=2))
    run_brain(brain, clock, 0.2, frames=only_mom)
    assert list(brain.learn.person_names().values()) == ["媽媽"]
    run_brain(brain, clock, 3.0, frames=seen)  # (they are both in view again)
    bridge.submit(make_call("attend", {"target_kind": "person", "hint": "媽媽"}, t=clock.t, seq=3))
    run_brain(brain, clock, 0.3, frames=seen)
    assert brain._live_attend == ("person", 1) and bridge.last_extras.get("watch", 0) > 0
    assert brain._person_b == pytest.approx(
        mom.bearing
    )  # (its eyes are on her, not the nearer dad)
    run_brain(brain, clock, 8.0, frames=seen)
    assert brain._live_attend is None  # (paid, and let go: the nearer person is its focus again)
    assert brain._person_b == pytest.approx(dad.bearing)


def test_attend_to_someone_who_is_not_in_view_is_refused() -> None:
    brain, bridge, clock = rig(SNUGGLER)
    mom = person_sight(1, look=MOM_LOOK)
    run_brain(brain, clock, 6.0, frames=scene(mom))
    bridge.submit(make_call("recognize", {"name": "媽媽"}, t=clock.t, seq=1))
    run_brain(brain, clock, 0.2, frames=scene(mom))
    run_brain(brain, clock, 4.0)  # (she left)
    bridge.submit(make_call("attend", {"target_kind": "person", "hint": "媽媽"}, t=clock.t, seq=2))
    run_brain(brain, clock, 0.2)
    assert [e["reason"] for e in bridge.log if e["tool"] == "attend"] == ["not_in_view"]


def test_attend_to_a_sound_it_just_heard() -> None:
    brain, bridge, clock = rig(HATCHLING)
    voice = scene(sound=("voice", 60.0))
    run_brain(brain, clock, 0.4, frames=voice)
    bridge.submit(make_call("attend", {"target_kind": "sound", "hint": "吃飯的聲音"}, t=clock.t))
    run_brain(brain, clock, 0.2)
    assert bridge.entered and bridge.entered[-1]["target"]["sound_kind"] == "voice"
    assert "look_around" in bridge.last_extras
    # a kind it has not heard is not there
    bridge.submit(make_call("attend", {"target_kind": "sound", "hint": "敲門聲"}, t=clock.t, seq=2))
    run_brain(brain, clock, 0.2)
    assert [e["reason"] for e in bridge.log if e["tool"] == "attend"][-1] == "nothing_matches"


def test_attention_is_a_candidate_too_and_a_frightened_duck_keeps_its_eyes_where_they_are() -> None:
    brain, bridge, clock = rig(SNUGGLER)
    toy = thing_sight(7, range_m=0.6)

    def scared(t: float, cmd: Any) -> None:
        brain.drives.fear = 0.95

    run_brain(brain, clock, 10.0, frames=scene(toy), on_tick=scared)
    bridge.submit(make_call("attend", {"target_kind": "thing", "hint": "玩具"}, t=clock.t))
    cmds = run_brain(brain, clock, 5.0, frames=scene(toy), on_tick=scared)
    assert not any(c.behaviour == "investigate" for c in cmds)


# ── names ──────────────────────────────────────────────────────────────────────────────


def introduce(
    brain: Any, bridge: B.LiveBridge, clock: Any, name: str, *, seq: int = 1, **mk: Any
) -> None:
    bridge.submit(make_call("recognize", {"name": name}, t=clock.t, seq=seq, call_mark=mark(**mk)))
    run_brain(brain, clock, 0.2, frames=scene(person_sight(1, look=MOM_LOOK)))


def test_a_name_binds_to_the_person_it_sees_in_the_learners_record() -> None:
    brain, bridge, clock = rig(SNUGGLER)
    seen = scene(person_sight(1, look=MOM_LOOK))
    run_brain(brain, clock, 5.0, frames=seen)
    introduce(brain, bridge, clock, "妈妈")
    ((pid, name),) = brain.learn.person_names().items()
    assert name == "媽媽" and brain.learn.people[pid].name_raw == "妈妈"
    assert len(brain.learn.people) == 1
    # said again another way: the same record, not a second person
    introduce(brain, bridge, clock, "媽媽", seq=2)
    introduce(brain, bridge, clock, "我的媽媽", seq=3)
    assert len(brain.learn.people) == 1 and brain.learn.person_names() == {pid: "媽媽"}
    # and the next time she is in view it is the same person (a new eye track, the same look)
    run_brain(brain, clock, 40.0)
    run_brain(brain, clock, 3.0, frames=scene(person_sight(9, look=MOM_LOOK)))
    assert len(brain.learn.people) == 1 and brain.learn.people[pid].sightings == 2


def test_a_name_with_nobody_in_view_makes_nobody() -> None:
    brain, bridge, clock = rig(SNUGGLER)
    run_brain(brain, clock, 5.0)
    bridge.submit(make_call("recognize", {"name": "媽媽"}, t=clock.t))
    run_brain(brain, clock, 0.2)
    assert brain.learn.people == {} and bridge.reasons[B.NO_PERSON] == 1


def test_a_misheard_name_is_not_stored_as_a_new_person() -> None:
    brain, bridge, clock = rig(SNUGGLER)
    seen = scene(person_sight(1, look=MOM_LOOK))
    run_brain(brain, clock, 5.0, frames=seen)
    before = len(brain.learn.people)
    # the family called 蛋黃; the model heard 郑华 and thought it an introduction
    introduce(brain, bridge, clock, "郑华")
    introduce(brain, bridge, clock, "蛋黄", seq=2)
    assert len(brain.learn.people) == before == 1 and brain.learn.person_names() == {}
    assert bridge.reasons["sounds_like_duck_name"] == 1 and bridge.reasons["duck_name"] == 1
    person = next(iter(brain.learn.people.values()))
    assert person.pending == {}  # (not even as a pending name: it is the duck's own)


def test_when_it_is_not_sure_the_name_waits_on_the_person_it_sees() -> None:
    brain, bridge, clock = rig(SNUGGLER)
    run_brain(brain, clock, 5.0, frames=scene(person_sight(1, look=MOM_LOOK)))
    introduce(brain, bridge, clock, "媽媽", since_human_s=None)  # (nobody has just spoken)
    person = next(iter(brain.learn.people.values()))
    assert person.name is None and person.pending == {"媽媽": 1}
    assert len(brain.learn.people) == 1
    introduce(brain, bridge, clock, "媽媽", seq=2, since_human_s=None)  # (a second, separate time)
    assert person.name == "媽媽" and person.pending == {}


def test_a_second_name_for_a_named_person_needs_two_pieces_of_evidence() -> None:
    brain, bridge, clock = rig(SNUGGLER)
    run_brain(brain, clock, 5.0, frames=scene(person_sight(1, look=MOM_LOOK)))
    introduce(brain, bridge, clock, "媽媽")
    person = next(iter(brain.learn.people.values()))
    introduce(brain, bridge, clock, "小美", seq=2)
    assert person.name == "媽媽" and person.pending == {"小美": 1}
    introduce(brain, bridge, clock, "小美", seq=3)
    assert person.name == "小美"


def test_what_is_kept_to_remember_stays_in_the_learner_and_an_echo_cannot_put_anything_there() -> (
    None
):
    brain, bridge, clock = rig(SNUGGLER)
    run_brain(brain, clock, 3.0)
    bridge.submit(make_call("remember", {"note": "這是媽媽。"}, t=clock.t, seq=1))
    bridge.submit(
        make_call("remember", {"note": "Mhm."}, t=clock.t, seq=2, call_mark=mark(echo_only=True))
    )
    bridge.submit(
        make_call(
            "remember",
            {"note": "Bye."},
            t=clock.t,
            seq=3,
            call_mark=mark(cause_echo=True, cause_age_s=0.5),
        )
    )
    run_brain(brain, clock, 0.2)
    assert [n[2] for n in brain.learn.notes] == ["這是媽媽。"]
    assert bridge.reasons[G.ECHO_ONLY] == 1 and bridge.reasons[G.ECHO_CAUSE] == 1


# ── feel ───────────────────────────────────────────────────────────────────────────────


def test_feel_nudges_the_mood_a_little_and_every_nudge_is_logged() -> None:
    brain, bridge, clock = rig(SNUGGLER, valence_per_min=0.12)
    run_brain(brain, clock, 3.0)
    moods = [brain.drives.valence]
    for k in range(6):
        bridge.submit(make_call("feel", {"valence": -1.0, "talk": 0.0}, t=clock.t, seq=k + 1))
        run_brain(brain, clock, 0.1)
        moods.append(brain.drives.valence)
    assert moods[-1] < moods[0]  # (it did turn it down)
    assert len(bridge.nudges) == 4  # (every one it took is logged: the gate allows 4 in 10 s)
    assert bridge.reasons[G.CALL_CAP] == 2
    moves = [abs(n["valence_after"] - n["valence_before"]) for n in bridge.nudges]
    assert all(m <= 0.05 + 1e-9 for m in moves)  # (a little at a time)
    assert sum(moves) <= 0.12 + 1e-9  # (and bounded for the minute)
    assert moves[-1] == 0.0  # (the minute's allowance was used up)
    assert bridge.nudges[0]["talk_after"] < bridge.nudges[0]["talk_before"]


def test_a_wish_not_to_talk_makes_the_duck_quieter() -> None:
    brain, bridge, clock = rig(SNUGGLER)
    run_brain(brain, clock, 5.0)
    intent = B.Intent(1, "reply", 1, clock.t, 5.0, ("greet",), level=0.6)
    before = bridge._weight(intent, brain)
    for k in range(4):
        bridge.submit(make_call("feel", {"valence": 0.2, "talk": 0.0}, t=clock.t, seq=k + 1))
        run_brain(brain, clock, 0.1)
    assert bridge.talk.value < bridge.talk.rest
    assert bridge._weight(intent, brain) < before


# ── the cloud switch ───────────────────────────────────────────────────────────────────


def test_switching_the_cloud_off_cancels_what_was_proposed_and_refuses_what_comes_after() -> None:
    brain, clock = make_brain(*SNUGGLER)
    session = StubSession(clock)
    bridge = B.LiveBridge(session, clock=clock).attach(brain)
    run_brain(brain, clock, 8.0)
    session.calls.append(vocalize([tag("greet", 0.5, 3000)], t=clock.t, seq=1))
    run_brain(brain, clock, 0.3)
    assert [i.active for i in bridge.intents] == [True]
    session.enabled = False
    session.calls.append(vocalize([tag("greet", 0.5, 0)], t=clock.t, seq=2))
    cmds = run_brain(brain, clock, 5.0)
    assert bridge.intents == [] and bridge.voiced == [] and bridge.last_extras == {}
    assert bridge.reasons[B.CLOUD_OFF] >= 2  # (the one in flight, and the new call at the gate)
    assert not any(c.vocal == "greet" for c in cmds)  # (nothing it was asked for sounded)
    session.enabled = True
    session.calls.append(vocalize([tag("greet", 0.5, 0)], t=clock.t, seq=3))
    run_brain(brain, clock, 6.0)
    assert bridge.voiced  # (and back on it works again)


# ── the duck's own ear tells the cloud its name was called ─────────────────────────────


class NoteSession(StubSession):
    """A session that also takes the one-off texts the bridge sends (``send_note``)."""

    def __init__(self, clock: ManualClock) -> None:
        super().__init__(clock)
        self.notes: list[tuple[float, str]] = []
        self.line_up = True

    def send_note(self, text: str, *, about: str = "note") -> bool:
        if not self.line_up:
            return False
        self.notes.append((self.clock.t, text))
        return True


def sound_frames(*spans: tuple[float, float, str | None]) -> Any:
    """``frames(t)`` for ``run_brain``: the hearing model reports ``kind`` from ``t0`` to ``t1``."""

    def make(t: float) -> Any:
        kind = next((k for t0, t1, k in spans if t0 <= t < t1), None)
        return frame(int(round(t * 1000)), sound_kind=kind, level_db=62.0 if kind else 30.0)

    return make


def ear_rig(**config: Any) -> tuple[IntegratedBrain, B.LiveBridge, NoteSession, ManualClock]:
    brain, clock = make_brain(*SNUGGLER)
    session = NoteSession(clock)
    bridge = B.LiveBridge(session, clock=clock, config=B.BridgeConfig(**config)).attach(brain)
    return brain, bridge, session, clock


def test_the_ducks_own_ear_hearing_its_name_tells_the_cloud_once_per_name_event() -> None:
    brain, bridge, session, clock = ear_rig()
    run_brain(brain, clock, 6.0, frames=sound_frames((2.0, 3.2, "name")))  # (a 1.2 s event)
    assert [text for _t, text in session.notes] == [EAR_NAME_HINT]
    assert 2.0 <= session.notes[0][0] <= 2.1  # (the moment the ear reports it)
    ear = bridge.view()["ear"]
    assert (ear["name_events"], ear["hints_sent"], ear["skipped"]) == (1, 1, {})
    assert ear["last"] == [{"t": session.notes[0][0], "sent": True, "why": None, "level_db": 62.0}]
    assert bridge.faults == 0


def test_a_sound_that_is_not_the_name_tells_nothing() -> None:
    brain, bridge, session, clock = ear_rig()
    kinds = ("voice", "hello", "clap", "bang", "door", "music", "tv", "footsteps", "noise")
    spans = tuple((2.0 + 2.0 * i, 3.0 + 2.0 * i, kind) for i, kind in enumerate(kinds))
    run_brain(brain, clock, 24.0, frames=sound_frames(*spans))
    assert session.notes == []
    assert bridge.view()["ear"] == {"name_events": 0, "hints_sent": 0, "skipped": {}, "last": []}


def test_the_hint_comes_from_the_ducks_ear_not_from_what_the_scene_holds() -> None:
    # a person speaks right in front of the duck (in its view) but its ear reports no name
    brain, bridge, session, clock = ear_rig()
    person = person_sight(1, range_m=1.0)

    def quiet(t: float) -> Any:
        return frame(int(round(t * 1000)), person, sound_kind="voice", level_db=65.0)

    run_brain(brain, clock, 5.0, frames=quiet)
    assert session.notes == [] and bridge.view()["ear"]["name_events"] == 0


def test_no_hint_while_the_cloud_is_off_and_none_for_the_event_that_came_before_it_was_on() -> None:
    brain, bridge, session, clock = ear_rig()
    session.enabled = False
    run_brain(brain, clock, 5.0, frames=sound_frames((1.0, 2.0, "name")))
    assert session.notes == []
    session.enabled = True  # (switched on while the name is still being heard: it is not told late)
    run_brain(brain, clock, 5.0)
    assert session.notes == []
    ear = bridge.view()["ear"]
    assert (ear["name_events"], ear["hints_sent"], ear["skipped"]) == (1, 0, {"cloud_off": 1})
    # off by the state words too (switched off / capped), and then on again: the next call is told
    for state in ("off", "capped"):
        session.state = state
        t0 = clock.t
        run_brain(brain, clock, 4.0, frames=sound_frames((t0 + 0.5, t0 + 1.5, "name")))
        assert session.notes == []
    session.state = "online"
    t0 = clock.t
    run_brain(brain, clock, 4.0, frames=sound_frames((t0 + 0.5, t0 + 1.5, "name")))
    assert len(session.notes) == 1


def test_no_session_no_line_or_the_hint_switched_off_in_the_config_tell_nothing_and_break_nothing():
    for kind in ("none", "line_down", "config"):
        brain, clock = make_brain(*SNUGGLER)
        session = NoteSession(clock)
        config = B.BridgeConfig(ear_name_hint=None) if kind == "config" else B.BridgeConfig()
        bridge = B.LiveBridge(None if kind == "none" else session, clock=clock, config=config)
        bridge.attach(brain)
        session.line_up = kind != "line_down"
        run_brain(brain, clock, 5.0, frames=sound_frames((1.0, 2.0, "name")))
        assert session.notes == [], kind
        ear = bridge.view()["ear"]
        assert ear["name_events"] == 1 and ear["hints_sent"] == 0, kind
        assert (
            ear["skipped"]
            == {
                "none": {"no_line": 1},
                "line_down": {"no_line": 1},
                "config": {"switched_off": 1},
            }[kind]
        )
        assert bridge.faults == 0


def test_the_hint_is_rate_limited_and_a_flickering_ear_is_one_event_not_many() -> None:
    brain, bridge, session, clock = ear_rig(ear_hint_gap_s=3.0)
    # the kind drops out for 0.2 s in the middle of one call (the ear's class hold ends and comes
    # back): two edges 0.5 s apart; and a second call 1.5 s after the first one ended
    spans = ((1.0, 1.4, "name"), (1.6, 2.0, "name"), (3.5, 4.0, "name"), (8.0, 8.5, "name"))
    run_brain(brain, clock, 12.0, frames=sound_frames(*spans))
    times = [t for t, _ in session.notes]
    assert len(times) == 2 and 1.0 <= times[0] <= 1.1 and 8.0 <= times[1] <= 8.1
    ear = bridge.view()["ear"]
    assert (ear["name_events"], ear["hints_sent"], ear["skipped"]) == (4, 2, {"rate": 2})
    assert [e["why"] for e in ear["last"]] == [None, "rate", "rate", None]
    # one event is one edge, however long the ear holds it
    brain2, bridge2, session2, clock2 = ear_rig()
    run_brain(brain2, clock2, 20.0, frames=sound_frames((1.0, 15.0, "name")))
    assert len(session2.notes) == 1


def test_a_name_event_that_follows_another_sound_without_a_gap_is_still_an_event() -> None:
    brain, bridge, session, clock = ear_rig()
    run_brain(brain, clock, 6.0, frames=sound_frames((1.0, 2.0, "voice"), (2.0, 3.0, "name")))
    assert len(session.notes) == 1 and 2.0 <= session.notes[0][0] <= 2.1


def test_the_hint_does_not_change_what_the_duck_does() -> None:
    """The same ear, the same frames: with the hint or without it the brain's commands are the same
    (the hint is only for the cloud)."""
    spans = ((2.0, 3.0, "name"), (9.0, 10.0, "name"))
    brain1, bridge1, session1, clock1 = ear_rig()
    cmds1 = run_brain(brain1, clock1, 14.0, frames=sound_frames(*spans))
    brain2, bridge2, session2, clock2 = ear_rig(ear_name_hint=None)
    cmds2 = run_brain(brain2, clock2, 14.0, frames=sound_frames(*spans))
    assert len(session1.notes) == 2 and session2.notes == []
    assert [(c.t_ms, c.behaviour, c.vocal, c.reflex) for c in cmds1] == [
        (c.t_ms, c.behaviour, c.vocal, c.reflex) for c in cmds2
    ]


def test_end_to_end_the_hint_goes_up_the_real_session_once_and_not_when_the_cloud_is_off() -> None:
    from live_rig import Rig

    rig = Rig()
    brain, _ = make_brain(*SNUGGLER, clock=rig.clock)
    bridge = B.LiveBridge(rig.session, clock=rig.clock).attach(brain)
    spans = ((3.0, 4.0, "name"), (9.0, 10.0, "name"))
    cloud_off_at = 7.0

    def tick(t: float, _cmd: Any) -> None:
        rig.session.feed_pcm(b"\x00\x00" * 320, echo=False)
        rig.session.pump()
        if t >= cloud_off_at and rig.session.enabled:
            rig.session.set_enabled(False)

    run_brain(brain, rig.clock, 12.0, frames=sound_frames(*spans), on_tick=tick)
    hints = [(t, c) for t, text, c in rig.fake.sent.texts if text == EAR_NAME_HINT]
    assert len(hints) == 1 and hints[0][1] is False and 3.0 <= hints[0][0] <= 3.1
    ear = bridge.view()["ear"]
    assert (ear["name_events"], ear["hints_sent"], ear["skipped"]) == (2, 1, {"cloud_off": 1})
    assert rig.session.status()["notes_sent"] == 1


# ── barge-in, closed loop ──────────────────────────────────────────────────────────────


def mic_and_vad(tid: str = "snuggler", seed: int = 7) -> tuple[Any, VD.VoiceDetector, int]:
    from duckmind.live import scenario as S

    run = S.ScenarioRun(S.load_scenario_file(), "echo_only", tid, seed, require_bank=True)
    return (
        run.make_mic(),
        VD.VoiceDetector(VD.BankReferences(run.bank(), run.voice_seed())),
        run.voice_seed(),
    )


def speech_at(t0: float) -> list[dict[str, Any]]:
    return [
        {
            "id": 1,
            "kind": "voice",
            "t0": t0,
            "t1": t0 + 0.72,
            "db": 66.0,
            "src": "person:mom",
            "utterance": "stop_calling",
        }
    ]


@pytest.mark.skipif(
    not __import__("duckmind.simweb.voice", fromlist=["x"])
    .VoiceBank(None)
    .ready(__import__("duckmind.simweb.voice", fromlist=["x"]).voice_seed(1728)),
    reason="the snuggler's voice bank is not rendered on this machine",
)
class TestBargeIn:
    def closed_loop(self, speak_at: float | None, *, call_tag: str = "coo") -> tuple[Any, ...]:
        """The duck calls (a reply of the cloud's, voiced through the bridge); the microphone hears
        its own voice; a person may speak over it; the local detection and the bridge stop the call."""
        brain, clock = make_brain(*SNUGGLER)
        mic, det, _vseed = mic_and_vad()
        stops: list[float] = []

        def cut(_request: Any) -> None:
            stops.append(clock.t)
            mic.cut_calls()

        bridge = B.LiveBridge(None, clock=clock, vad=det, on_stop=cut).attach(brain)
        run_brain(brain, clock, 12.0)
        bridge.submit(vocalize([tag(call_tag, 0.9, 0)], t=clock.t, seq=1))
        calls: list[dict[str, Any]] = []
        record: list[tuple[float, Any]] = []
        t = clock.t
        end = t + 8.0
        while t < end - 1e-9:
            t = round(t + 0.02, 6)
            clock.set(t)
            sounds = speech_at(speak_at) if speak_at is not None and t >= speak_at - 0.5 else []
            mic_frame = mic.tick(t, sounds, calls=calls, wall=t)
            calls = []
            bridge.hear(mic_frame, mic.vocal_windows())
            cmd, _ = brain.step(int(round(t * 1000)), frame(int(round(t * 1000))))
            record.append((t, cmd))
            if cmd.vocal:
                calls.append({"tag": cmd.vocal, "level": cmd.vocal_level})
        return brain, bridge, mic, record, stops

    def test_a_person_who_speaks_over_the_call_stops_it_within_the_bound(self) -> None:
        # find when the call begins in an unobstructed run, then speak 0.25 s into it
        _b, bridge0, mic0, rec0, _s = self.closed_loop(None)
        first = next(t for t, c in rec0 if c.vocal)
        window = mic0.vocal_windows()[0]
        assert window.t1 - window.t0 >= 0.7  # (a coo is long enough to be spoken over)
        speak = first + 0.25
        brain, bridge, mic, rec, stops = self.closed_loop(speak)
        assert len(bridge.barges) == 1 and bridge.barges[0].stopped
        barge = bridge.barges[0]
        stop_tick = next(
            t for t, c in rec if c.vocal is None and c.vocal_seq > 0 and t > first and t == stops[0]
        )
        assert stop_tick - speak <= 0.3  # (the target: within 300 ms of the onset)
        assert barge.total_s <= 0.3 and barge.detect_s <= 0.1
        # the call's echo ended there, not at the end of the call
        ended = mic.vocal_windows()[0].t1
        assert ended <= speak + 0.3 and ended < window.t1 - 0.1
        # the stop is a new sequence number with no call
        stop_cmd = next(c for t, c in rec if t == stops[0])
        assert stop_cmd.vocal is None and stop_cmd.vocal_seq == brain._vocal_seq

    def test_the_ducks_own_call_alone_never_stops_itself(self) -> None:
        brain, bridge, mic, rec, stops = self.closed_loop(None)
        assert bridge.voiced and stops == [] and bridge.barges == []
        assert bridge.vad is not None and bridge.vad.voiced_frames == 0
        window = mic.vocal_windows()[0]
        assert window.t1 - window.t0 >= 0.7  # (it called to the end)

    def test_what_was_not_yet_voiced_is_cancelled_and_the_duck_stays_quiet_a_moment(self) -> None:
        brain, clock = make_brain(*SNUGGLER)
        mic, det, _ = mic_and_vad()
        bridge = B.LiveBridge(None, clock=clock, vad=det, on_stop=lambda r: mic.cut_calls()).attach(
            brain
        )
        run_brain(brain, clock, 12.0)
        # two tags: the second is for later; a person speaks over the first
        bridge.submit(vocalize([tag("coo", 0.9, 0), tag("coo", 0.9, 1500)], t=clock.t, seq=1))
        calls: list[dict[str, Any]] = []
        t = clock.t
        first = None
        voiced_after: list[float] = []
        end = t + 8.0
        while t < end - 1e-9:
            t = round(t + 0.02, 6)
            clock.set(t)
            sounds = speech_at(first + 0.25) if first is not None and t >= first - 0.3 else []
            frame_ = mic.tick(t, sounds, calls=calls, wall=t)
            calls = []
            bridge.hear(frame_, mic.vocal_windows())
            cmd, _ = brain.step(int(round(t * 1000)), frame(int(round(t * 1000))))
            if cmd.vocal:
                calls.append({"tag": cmd.vocal, "level": cmd.vocal_level})
                if first is None:
                    first = t
                elif t > first + 0.3:
                    voiced_after.append(t)
        assert len(bridge.barges) == 1 and bridge.barges[0].cancelled >= 1
        assert voiced_after == []  # (the second tag never sounded)
        assert bridge.counts["intent:cancelled"] >= 1 and bridge.reasons[B.BARGE_IN] >= 1


def test_a_person_speaking_makes_the_duck_wait_before_it_answers() -> None:
    brain, clock = make_brain(*SNUGGLER)
    bridge = B.LiveBridge(None, clock=clock).attach(brain)
    run_brain(brain, clock, 12.0)
    bridge._speaking = True  # (the local detection hears a person right now)
    bridge.submit(vocalize([tag("greet", 0.9, 0)], t=clock.t, seq=1))
    run_brain(brain, clock, 2.0)
    assert bridge.voiced == []
    bridge._speaking = False
    bridge._quiet_until = clock.t + 0.3
    run_brain(brain, clock, 3.0)
    assert bridge.voiced  # (the proposal was still there when the person had finished)


def test_local_evidence_of_a_person_lets_a_call_the_echo_marks_doubt_through() -> None:
    brain, clock = make_brain(*SNUGGLER)
    bridge = B.LiveBridge(None, clock=clock).attach(brain)
    run_brain(brain, clock, 4.0)
    doubted = mark(echo_only=True, since_human_s=None)
    bridge.submit(vocalize([tag("greet")], t=clock.t, seq=1, call_mark=doubted))
    run_brain(brain, clock, 0.2)
    assert bridge.entered == [] and bridge.reasons[G.ECHO_ONLY] == 1
    bridge._spans.append([clock.t - 0.5, clock.t - 0.1])  # (the duck heard someone, locally)
    bridge.submit(vocalize([tag("greet")], t=clock.t, seq=2, call_mark=doubted))
    run_brain(brain, clock, 0.2)
    accepted = [e for e in bridge.log if e["decision"] == "accepted"]
    assert len(bridge.entered) == 1 and accepted[-1]["rescued"] is True


# ── the brain is not harmed ────────────────────────────────────────────────────────────


def test_without_a_bridge_nothing_changes_and_with_one_the_brain_shows_it() -> None:
    plain, clock_a = make_brain(*EXPLORER)
    cmds_a = run_brain(plain, clock_a, 10.0)
    other, clock_b = make_brain(*EXPLORER)
    bridge = B.LiveBridge(None, clock=clock_b).attach(other)
    cmds_b = run_brain(other, clock_b, 10.0)  # (a bridge with nothing to do)
    assert [c.to_json() for c in cmds_a] == [c.to_json() for c in cmds_b]
    _cmd, tel = other.step(
        int(round(clock_b.t * 1000)) + 50, frame(int(round(clock_b.t * 1000)) + 50)
    )
    assert "live" in tel.learned and tel.learned["live"]["faults"] == 0
    _cmd, tel0 = plain.step(
        int(round(clock_a.t * 1000)) + 50, frame(int(round(clock_a.t * 1000)) + 50)
    )
    assert "live" not in tel0.learned
    assert bridge.view()["gate"]["accepted"] == 0


def test_a_fault_in_the_bridge_is_counted_and_never_stops_the_brain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    brain, bridge, clock = rig(SNUGGLER)

    def boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("bridge fault")

    monkeypatch.setattr(bridge, "_inputs", boom)
    cmds = run_brain(brain, clock, 2.0)
    assert len(cmds) == 40 and bridge.faults == 40 and "bridge fault" in (bridge.last_fault or "")
    monkeypatch.undo()
    monkeypatch.setattr(bridge, "_emit", boom)
    assert len(run_brain(brain, clock, 1.0)) == 20


def test_the_selector_takes_outside_input_like_any_other_reason() -> None:
    brain, clock = make_brain(*SNUGGLER)
    sel = brain.sel
    ignored = sel.set_extra({"greet": 0.8, "no_such_act": 0.5, "rest_sit": 0.0})
    assert ignored == {"no_such_act": 0.5}
    assert sel._extra[sel.index("greet")] == 0.8 and sum(sel._extra) == 0.8
    sel.set_extra({})
    assert not any(sel._extra)
    # it goes through the temperament gain and the satiation: a tired-of-it channel gets less
    feats = list(brain._feat)
    ctx = type(
        "Ctx",
        (),
        {
            "sitting": True,
            "standing": False,
            "asleep": False,
            "can_walk": True,
            "wakefulness": 1.0,
            "posture_tired": 0.0,
            "recharging": False,
            "nest_rest": False,
        },
    )()
    sel.set_extra({"greet": 1.0})
    sel.step(0.05, feats, ctx, False, 0.0)  # type: ignore[arg-type]
    fresh = sel.sal[sel.index("greet")]
    sel.sat[sel.index("greet")] = 0.9
    sel.step(0.05, feats, ctx, False, 0.0)  # type: ignore[arg-type]
    assert sel.sal[sel.index("greet")] < 0.4 * fresh


def test_vocal_numbers_are_taken_not_invented() -> None:
    brain, _clock = make_brain(*SNUGGLER)
    a, b = brain.take_vocal_seq(), brain.take_vocal_seq()
    assert (a, b) == (1, 2) and brain._vocal_seq == 2
    assert np.isfinite(a)


def test_only_a_persons_answer_is_a_claim_on_the_body_not_attention_nor_the_ducks_own_wish() -> (
    None
):
    # (select.CLAIM_MIN: a live reply may take the body from a bout that is not done. The attention
    # an ``attend`` asks for and the wishes the bridge asks for by itself (call_out, follow) compete
    # like any other reason.)
    brain, bridge, clock = rig(HATCHLING)
    seen: list[tuple[dict[str, float], dict[str, float]]] = []
    original = brain.sel.set_extra

    def tap(inputs: dict[str, float], *, claims: dict[str, float] | None = None) -> Any:
        seen.append((dict(inputs), dict(claims or {})))
        return original(inputs, claims=claims)

    brain.sel.set_extra = tap  # type: ignore[method-assign]
    toy = thing_sight(7, range_m=0.6)
    frames = scene(toy)
    run_brain(brain, clock, 5.0, frames=frames)
    bridge.submit(make_call("attend", {"target_kind": "thing", "hint": "球"}, t=clock.t, seq=1))
    run_brain(brain, clock, 1.0, frames=frames)
    assert any("investigate" in i for i, _c in seen)
    assert all(not c for _i, c in seen)  # (attention: no claim, in all the ticks so far)
    bridge.submit(vocalize([tag("greet", 0.6, 0)], t=clock.t, seq=2))
    before = len(seen)
    run_brain(brain, clock, 0.3, frames=frames)
    replies = [(i, c) for i, c in seen[before:] if c]
    assert replies, "a reply's input went to the selector without a claim"
    inputs, claims = replies[0]
    assert "greet" in claims and set(claims) <= {"greet", "call_response", "call_out"}
    assert all(claims[k] > 0.0 and claims[k] <= inputs[k] + 1e-9 for k in claims)
    # …and nothing but replies is in it: the ``investigate`` of the attention is not
    assert "investigate" not in claims
