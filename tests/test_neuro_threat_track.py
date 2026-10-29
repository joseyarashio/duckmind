"""The brain's ``threat`` target names what the duck fled from (fix/threat-track).

Before: the ``threat`` target (flee, retreat) and the ``place`` target of a remembered thing carried track 0, so a
verifier could not tell that a duck startled by a rolled ball and running away was reacting to the ball. Now the target
carries the eye's track of the sighting that scared it (a looming thing in view), or of the remembered thing whose place
it walks to, and 0 when there is none (a sound, a touch, a place that is not a thing). Telemetry only: no decision reads
the track (tests/test_threat_track_hash.py proves the motor commands did not change).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from duckmind.neuro import contract as C
from duckmind.neuro import reflex as R
from duckmind.neuro.brain import NeuroBrain
from duckmind.neuro.genome import Genome
from duckmind.neuro.learn import Learner, Thing
from duckmind.neuro.perceive import Perceiver, Percept
from duckmind.neuro.synth import SynthWorld

ROOT = Path(__file__).resolve().parents[1]
PARAMS = R.params_from({"boldness": 0.5, "wit": 0.5}, stage="juvenile", side=1)
SHY = dict(
    boldness=0.1, activity=0.3, sociability=0.9, exploration=0.2, wit=0.4, chronotype=0.5, tempo=0.4
)


def _genome() -> Genome:
    return Genome.from_axes(SHY, primary="snuggler", branch="snuggler", gd=120)


def _frame(t: int, sightings: tuple[C.Sighting, ...] = (), **kw: object) -> C.SensorFrame:
    return C.SensorFrame(
        t_ms=t,
        body=C.BodySense(posture="stand"),
        camera=C.CameraSense(usable=True, sightings=sightings),
        power=C.PowerSense(battery=0.85),
        **kw,  # type: ignore[arg-type]
    )


def test_the_reflex_presents_a_looming_onset_with_the_eyes_track() -> None:
    sight = C.Sighting(kind="hand", bearing=0.1, looming=3.5, size=0.4, motion=0.9, track=2)
    state = R.initial(1)
    seen: list[R.Presented] = []
    for t in range(0, 1500, 20):
        sightings = (sight,) if t == 100 else ()
        state, out = R.step(PARAMS, state, _frame(t, sightings), arousal=0.5, walking=False)
        seen += list(out.presented)
    loom = [p for p in seen if p.stimulus == "visual:looming"]
    assert loom and loom[0].reflex == "startle" and loom[0].track == 2


def test_a_sound_is_presented_with_no_track() -> None:
    state = R.initial(1)
    seen: list[R.Presented] = []
    for t in range(0, 600, 20):
        sound = C.SoundSense(level_db=88.0, onset=1.0, kind="bang") if t == 100 else C.SoundSense()
        state, out = R.step(
            PARAMS,
            state,
            C.SensorFrame(t_ms=t, body=C.BodySense(posture="stand"), sound=sound),
            arousal=0.5,
            walking=False,
        )
        seen += list(out.presented)
    bang = [p for p in seen if p.stimulus == "sound:bang"]
    assert bang and all(p.track == 0 for p in bang)


def test_the_perceiver_names_the_track_of_what_is_looming_most() -> None:
    near = C.Sighting(kind="thing", bearing=-0.2, looming=1.0, size=0.1, track=5)
    ball = C.Sighting(kind="toy", bearing=0.3, looming=2.5, size=0.2, track=9)
    pct = Perceiver().perceive(_frame(0, (near, ball)), 0.0, False)
    assert pct.looming_track == 9 and pct.looming_bearing == pytest.approx(0.3)
    calm = Perceiver().perceive(_frame(0, ()), 0.0, False)
    assert calm.looming_track == 0


def _startled(brain: NeuroBrain, sight: C.Sighting, t0: int = 100) -> None:
    for t in range(0, t0 + 400, 20):
        brain.step(t, _frame(t, (sight,) if t == t0 else ()))


def test_the_brain_remembers_which_sighting_scared_it() -> None:
    brain = NeuroBrain(_genome(), seed=3)
    assert brain.appraiser.threat_track == 0
    ball = C.Sighting(
        kind="toy", bearing=0.2, range_m=0.6, looming=3.5, size=0.4, motion=0.9, track=42
    )
    _startled(brain, ball)
    assert brain.appraiser.threat_heading is not None
    assert brain.appraiser.threat_track == 42


def test_a_scare_with_a_direction_and_no_sighting_clears_the_track() -> None:
    brain = NeuroBrain(_genome(), seed=3)
    ball = C.Sighting(
        kind="toy", bearing=0.2, range_m=0.6, looming=3.5, size=0.4, motion=0.9, track=42
    )
    _startled(brain, ball)
    assert brain.appraiser.threat_track == 42
    # a loud sound the ear can place: where the new scare came from has no track
    for t in range(1000, 1600, 20):
        sound = (
            C.SoundSense(
                level_db=92.0, onset=1.0, kind="bang", bearing=-0.8, bearing_source="model"
            )
            if t == 1100
            else C.SoundSense()
        )
        brain.step(
            t,
            C.SensorFrame(
                t_ms=t,
                body=C.BodySense(posture="stand"),
                camera=C.CameraSense(usable=True),
                power=C.PowerSense(battery=0.85),
                sound=sound,
            ),
        )
    assert brain.appraiser.threat_track == 0


def test_the_threat_target_carries_the_track_and_a_place_carries_the_remembered_things() -> None:
    brain = NeuroBrain(_genome(), seed=3)
    brain.appraiser.threat_track = 42
    flee = brain._bout_target("flee", 2.5, None, Percept())
    assert flee is not None and (flee.kind, flee.track) == ("threat", 42)
    retreat = brain._bout_target("retreat", 2.5, None, Percept())
    assert retreat is not None and (retreat.kind, retreat.track) == ("threat", 42)
    brain.appraiser.threat_track = 0
    unseen = brain._bout_target("flee", 2.5, None, Percept())
    assert unseen is not None and (unseen.kind, unseen.track) == ("threat", 0)

    brain.learn.things[7] = Thing(kind="toy", track=77)
    brain.learn.things[8] = Thing(kind="thing")  # a thing the eye never gave a track for
    here = brain._bout_target("investigate", 0.3, 1.2, Percept(), (7, 0.3, 1.2, 0.5))
    assert here is not None and (here.kind, here.track) == ("place", 77)
    untracked = brain._bout_target("investigate", 0.3, 1.2, Percept(), (8, 0.3, 1.2, 0.5))
    assert untracked is not None and (untracked.kind, untracked.track) == ("place", 0)
    bare = brain._bout_target("investigate", 0.3, 1.2, Percept())  # no remembered thing: v4's call
    assert bare is not None and (bare.kind, bare.track) == ("place", 0)
    seen = C.Sighting(kind="thing", bearing=0.1, range_m=0.5, track=3)
    sighted = brain._bout_target("investigate", 0.1, 0.5, Percept(thing=seen), (7, 0.3, 1.2, 0.5))
    assert sighted is not None and (sighted.kind, sighted.track) == ("thing", 3)
    # the seen target stays as it was, and a person's place is nobody's thing
    assert brain._bout_target("wander", None, None, Percept()) is None


def test_the_learner_keeps_the_eyes_latest_track_of_a_thing() -> None:
    learn = Learner(_genome().p)
    odom = (0.0, 0.0, 0.0)
    key, _ = learn.see_thing(11, "toy", odom, 1.0, 0.0, 1.0, 0.1, 0.0)
    assert learn.track_of(key) == 11
    key2, _ = learn.see_thing(12, "toy", odom, 1.0, 0.0, 10.0, 0.1, 0.0)  # the eye re-issued it
    assert key2 == key and learn.track_of(key) == 12
    assert learn.track_of(999) == 0


def test_a_flee_after_a_looming_hand_names_the_hand_in_synth_world() -> None:
    """The scenario of tools/threat_track_hash.py in miniature: a shy duck, one loom at 20 s (SynthWorld's
    hand has track 7). Its flee or retreat carries track 7; before the loom no threat target has a track."""
    world = SynthWorld(seed=1, family=False)
    world.loom(20_000)
    brain = NeuroBrain(_genome(), seed=1)
    before: list[C.Target] = []
    after: list[C.Target] = []
    for k in range(900):
        t_ms = k * 100
        cmd, _ = brain.step(t_ms, world.frame(t_ms))
        if cmd.target is not None and cmd.target.kind == "threat":
            (before if t_ms < 20_000 else after).append(cmd.target)
        world.apply(cmd, 0.1)
    assert not [t for t in before if t.track]
    assert after and {t.track for t in after} == {7}


@pytest.mark.skipif(
    not (ROOT / "tools" / "machine_verify").is_dir(), reason="tools/machine_verify is not here"
)
def test_the_recorder_writes_the_track_of_a_threat_target() -> None:
    sys.path.insert(0, str(ROOT / "tools" / "machine_verify"))
    from common import Rec

    rec = Rec()
    base = dict(t=0.0, x=0.0, y=0.0, yaw=0.0, gaze=0.0, fallen=False, arrived=False)
    for kind, track in (("threat", 42), ("place", 77), ("thing", 3), ("threat", 0)):
        cmd = SimpleNamespace(
            behaviour="flee", bout=1, reflex=None, phase=0.0, vx=0.0,
            target=C.Target(kind, 0.1, 1.0, track),
        )  # fmt: skip
        rec.add(cmd=cmd, **base)
    rec.add(
        cmd=SimpleNamespace(
            behaviour="wander", bout=2, reflex=None, phase=0.0, vx=0.0, target=None
        ),
        **base,
    )
    assert rec.cols["tgt_kind"] == ["threat", "place", "thing", "threat", None]
    assert rec.cols["tgt_track"] == [42, 77, 3, 0, None]
