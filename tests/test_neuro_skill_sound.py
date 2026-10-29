"""a:sound (2026-09-28) — the sound skills in the brain's one competition (no physics: frames made
here): the new channels are real channels (installed at runtime until the integration), a music
beat pulls the dance, a hovering hand the theremin (ToF ducks only), a ride relabels being held;
an expected beat onset is taken out of what the reflexes hear; install() undoes itself."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterator
from dataclasses import replace

import pytest

from duckmind.neuro import contract as C
from duckmind.neuro import genome as G
from duckmind.neuro import select as S
from duckmind.neuro import skill_ride as SR
from duckmind.neuro import skill_sound as SS


@pytest.fixture()
def skills() -> Iterator[None]:
    SS.install()
    yield
    SS.uninstall()


def _brain(name: str):  # noqa: ANN202
    from duckmind.neuro.brain import NeuroBrain
    from duckmind.neuro.genome import preset

    genome = preset(name)
    return SS.SkillBrain(NeuroBrain(genome, seed=genome.seed))


def test_install_adds_channels_and_uninstall_takes_them_away() -> None:
    before = (len(S.FEATURES), set(C.BEHAVIOURS))
    SS.install()
    try:
        assert {"theremin", "dance", "wheee_ride"} <= set(C.BEHAVIOURS)
        assert "beat" in S.F and "hand_hover" in S.F
        brain = _brain("dreamer")
        assert "dance" in brain.sel.names and "theremin" in brain.sel.names
        assert "wheee_ride" not in brain.sel.names  # (a reflex's form, not a channel)
    finally:
        SS.uninstall()
    assert (len(S.FEATURES), set(C.BEHAVIOURS)) == before
    assert not SS.installed()


def _music(t_ms: int, bpm: float = 104.0) -> C.SensorFrame:
    t = t_ms / 1000.0
    period = 60.0 / bpm
    k = (t - 2.0 - 0.03) / period
    onset = 0.7 if t > 2.0 and abs(k - round(k)) * period < 0.0101 else 0.0
    sound = C.SoundSense(level_db=62.0, onset=onset, kind="music" if t > 3.0 else None)
    return C.SensorFrame(t_ms=t_ms, body=C.BodySense(posture="sit"), sound=sound)


def test_a_beat_in_music_pulls_the_dance_and_its_onsets_stop_surprising(skills: None) -> None:
    brain = _brain("dreamer")
    acts: Counter[str] = Counter()
    masked = 0
    for tick in range(60 * 50):
        cmd, _ = brain.step(tick * 20, _music(tick * 20))
        acts[cmd.behaviour] += 1
        masked += int(brain.last.masked_onset)
    assert acts["dance"] > 0.3 * sum(acts.values())
    assert masked > 40  # the kicks it expects no longer reach the orienting arcs
    view = brain.last.view()
    assert view["dance"]["beat_found"] and abs(view["dance"]["bpm"] - 104) < 3


def _hover(t_ms: int, *, tof: bool = True) -> C.SensorFrame:
    t = t_ms / 1000.0
    reading = C.TofSense(present=tof)
    if tof and 2.0 < t < 40.0:
        reading = C.TofSense(
            present=True, ahead_mm=int(300 + 120 * math.sin(2 * math.pi * 0.3 * t))
        )
    return C.SensorFrame(t_ms=t_ms, body=C.BodySense(posture="stand"), tof=reading)


def test_a_hovering_hand_pulls_the_theremin_for_a_tof_duck_only(skills: None) -> None:
    played = {}
    for tof in (True, False):
        brain = _brain("dreamer")
        ups = 0
        acts: Counter[str] = Counter()
        for tick in range(45 * 50):
            cmd, _ = brain.step(tick * 20, _hover(tick * 20, tof=tof))
            acts[cmd.behaviour] += 1
            ups += int(brain.last.theremin_up)
        played[tof] = (acts["theremin"], ups)
    assert played[True][0] > 50 * 5 and played[True][1] > 50 * 5
    assert played[False] == (0, 0)


def _with_things(frame: C.SensorFrame, *, hand: bool) -> C.SensorFrame:
    """``frame`` with, from the third second on, a thing the eye has not tracked before in the
    picture every second (a new track each time: a prop coming back into view as the head moves),
    and the camera's hand label when a hand hovers."""
    t = frame.t_ms / 1000.0
    sights: list[C.Sighting] = []
    if hand and 2.0 < t < 40.0:
        sights.append(C.Sighting(kind="hand", bearing=0.0, size=0.2, track=1, motion=0.4))
    if t > 3.0:
        sights.append(
            C.Sighting("thing", 0.3 + 0.3 * (int(t) % 2), range_m=1.6, size=0.06, track=10 + int(t))
        )
    return replace(frame, camera=C.CameraSense(usable=True, sightings=tuple(sights)))


def test_a_hovering_hand_outweighs_a_new_thing_far_away(skills: None) -> None:
    # (Without the hand's weight on investigate a dreamer was level between the two in this very
    # scene: 615 ticks of theremin to 720 of walking off to the new things; 2026-10-04.)
    brain = _brain("dreamer")
    acts: Counter[str] = Counter()
    for tick in range(30 * 50):
        cmd, _ = brain.step(tick * 20, _with_things(_hover(tick * 20), hand=True))
        acts[cmd.behaviour] += 1
    assert acts["theremin"] > 50 * 20
    assert acts["theremin"] > 3 * acts["investigate"]
    assert S.W["investigate"]["hand_hover"] < 0.0
    assert S.W["investigate"]["novel"] > 0.0  # (the other pulls on it are untouched)


def test_new_things_do_not_take_the_head_from_the_instrument_or_the_dance(skills: None) -> None:
    # A thing newly seen is noted, not turned to, while the head has a job (reflex.ENGAGED_ACTS):
    # in these scenes the look at each prop owned the duck's head for 72 % of the instrument's
    # ticks (its note is silent then) and 79 % of the dance's (no nod then); 2026-10-04.
    brain = _brain("dreamer")
    playing = up = 0
    for tick in range(30 * 50):
        cmd, _ = brain.step(tick * 20, _with_things(_hover(tick * 20), hand=True))
        if cmd.behaviour == "theremin":
            playing += 1
            up += int(brain.last.theremin_up)
    assert playing > 50 * 20 and up > 0.6 * playing
    brain = _brain("snuggler")
    danced = nodded = 0
    for tick in range(60 * 50):
        cmd, _ = brain.step(tick * 20, _with_things(_music(tick * 20), hand=False))
        if cmd.behaviour == "dance":
            danced += 1
            nodded += int(brain.last.dancing)
    assert danced > 50 * 20 and nodded > 0.8 * danced


def test_a_dancing_duck_still_turns_its_head_to_its_name(skills: None) -> None:
    brain = _brain("snuggler")
    call = 40_000
    turned = None
    for tick in range(46 * 50):
        t_ms = tick * 20
        frame = _with_things(_music(t_ms), hand=False)
        if call <= t_ms < call + 40:  # (a name called from the duck's left, heard with a bearing)
            frame = replace(
                frame,
                sound=C.SoundSense(
                    level_db=64.0, onset=1.0, kind="name", bearing=0.6, bearing_source="model"
                ),
            )
        cmd, _ = brain.step(t_ms, frame)
        if t_ms == call - 20:
            assert cmd.behaviour == "dance"
        if call <= t_ms < call + 1500 and cmd.reflex == "orient" and turned is None:
            turned = cmd.gaze.bearing
    assert turned == pytest.approx(0.6)


def test_a_ride_is_the_act_while_it_lasts(skills: None) -> None:
    brain = _brain("explorer")
    acts = []
    holds = 0
    for tick in range(14 * 50):
        t = tick / 50.0
        events = ("lifted",) if tick == 50 else ("set_down",) if tick == 11 * 50 else ()
        w = 2 * math.pi / 1.7
        gyro = (1.0 * math.cos(w * t), 0.2, 2.6 * math.cos(w * t)) if 2.0 < t < 8.0 else (0, 0, 0)
        frame = C.SensorFrame(
            t_ms=tick * 20, body=C.BodySense(posture="sit", gyro=gyro), events=events
        )
        cmd, tel = brain.step(tick * 20, frame)
        acts.append(cmd.behaviour)
        holds += int(brain.last.wheee_hold)
        if brain.last.ride.view.riding:
            assert cmd.vx == 0.0 and tel.behaviour == cmd.behaviour
    assert "wheee_ride" in acts
    assert holds > 50 * 5  # the joy ride's voice held for the ride
    assert acts[-1] != "wheee_ride"


def test_unexpected_loud_sound_on_a_predicted_beat_still_reaches_reflexes(skills: None) -> None:
    mind = SS.SoundSkills(G.preset("dreamer"), seed=17)
    last_seen = None
    for tick in range(210):
        t_ms = tick * 20
        frame = _music(t_ms)
        if t_ms > 2500 and frame.sound.onset > 0.0:
            frame = C.SensorFrame(
                t_ms=t_ms,
                body=frame.body,
                sound=C.SoundSense(level_db=90.0, onset=1.0, kind="clap"),
            )
            last_seen, _ = mind.pre(
                t_ms,
                frame,
                behaviour="dance",
                play=0.5,
                fear=0.0,
                arousal=0.2,
                asleep=False,
                fright=False,
            )
            break
        last_seen, _ = mind.pre(
            t_ms,
            frame,
            behaviour="dance",
            play=0.5,
            fear=0.0,
            arousal=0.2,
            asleep=False,
            fright=False,
        )
    assert last_seen is not None and last_seen.sound.kind == "clap"
    assert last_seen.sound.onset == 1.0


def test_sound_outputs_yield_to_reflex_and_carry_toy(skills: None) -> None:
    mind = SS.SoundSkills(G.preset("explorer"), seed=9)
    frame = C.SensorFrame(t_ms=1000, body=C.BodySense(posture="stand"))

    def prime(behaviour: str) -> None:
        mind.pre(
            1000,
            frame,
            behaviour=behaviour,
            play=0.7,
            fear=0.0,
            arousal=0.3,
            asleep=False,
            fright=False,
        )

    prime("dance")
    dancing, dance_out = mind.post(
        C.MotorCommand(t_ms=1000, behaviour="dance", bout=1, variant=1, reflex="startle"),
        bout=1,
        vocal_seq=0,
    )
    assert dancing.behaviour == "dance"
    assert not dance_out.dancing and dance_out.bob == C.HeadPose()
    assert not dance_out.wheee_hold

    prime("theremin")
    hand, hand_out = mind.post(
        C.MotorCommand(t_ms=1000, behaviour="theremin", bout=1, variant=1, reflex="withdraw"),
        bout=1,
        vocal_seq=0,
    )
    assert hand.gaze == C.Gaze()
    assert not hand_out.theremin_up

    prime("carry_toy")
    carry, carry_out = mind.post(
        C.MotorCommand(
            t_ms=1000,
            behaviour="carry_toy",
            bout=1,
            variant=1,
            vocal="chirp",
            vocal_seq=1,
        ),
        bout=1,
        vocal_seq=0,
    )
    assert carry.vocal is None
    assert not carry_out.dancing and not carry_out.theremin_up and not carry_out.wheee_hold

    ride = SR.RideTick(
        view=SR.RideView(True, 1.1, True, True, 1.0, 0.0),
        joy=0.8,
        thrilled=True,
        frightened=False,
        wheee_hold=True,
        alarm=False,
        pose=C.HeadPose(np=-0.25, hp=0.55),
        after_ride=0.0,
        ride_joy=0.8,
        ride_fear=0.0,
        value=0.0,
        rides=1,
    )
    mind._ride = ride
    prime("dance")
    protected, protected_out = mind.post(
        C.MotorCommand(t_ms=1000, behaviour="dance", bout=1, variant=1, reflex="startle"),
        bout=1,
        vocal_seq=0,
    )
    assert protected.behaviour == "dance" and not protected_out.wheee_hold

    mind._ride = ride
    lifted, lifted_out = mind.post(
        C.MotorCommand(t_ms=1000, behaviour="held", bout=1, variant=1, reflex="lifted"),
        bout=1,
        vocal_seq=0,
    )
    assert lifted.behaviour == "wheee_ride" and lifted_out.wheee_hold
