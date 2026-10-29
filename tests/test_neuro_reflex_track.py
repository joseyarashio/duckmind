"""The reflex arc says which sighting started a reflex (fix/threat-track, second step).

``Presented.track`` (the onset's source track) is now carried by the reflex that onset starts: ``Active.track`` ->
``ReflexOutput.track`` -> ``MotorCommand.reflex_track`` and ``ReflexEvent.track``; a freeze that follows a startle, and a double look,
carry the same track; a sound, a touch or a light change has none (0). Telemetry only: no decision reads it
(tests/test_threat_track_hash.py proves the motor commands did not change).
"""

from __future__ import annotations

from duckmind.neuro import contract as C
from duckmind.neuro import reflex as R
from duckmind.neuro.brain import NeuroBrain
from duckmind.neuro.genome import Genome
from duckmind.neuro.synth import SynthWorld

PARAMS = R.params_from({"boldness": 0.5, "wit": 0.5}, stage="juvenile", side=1)
SHY = dict(
    boldness=0.1, activity=0.3, sociability=0.9, exploration=0.2, wit=0.4, chronotype=0.5, tempo=0.4
)


def _frame(t: int, **kw: object) -> C.SensorFrame:
    return C.SensorFrame(t_ms=t, body=C.BodySense(posture="stand"), **kw)  # type: ignore[arg-type]


def _run(frames: list[C.SensorFrame]) -> list[R.ReflexOutput]:
    state = R.initial(1)
    outs = []
    for frame in frames:
        state, out = R.step(PARAMS, state, frame, arousal=0.5, walking=False)
        outs.append(out)
    return outs


def _camera(*sightings: C.Sighting) -> C.CameraSense:
    return C.CameraSense(usable=True, sightings=sightings)


def test_a_startle_and_the_freeze_after_it_carry_the_track_of_what_loomed() -> None:
    sight = C.Sighting(kind="hand", bearing=0.1, looming=3.5, size=0.4, motion=0.9, track=2)
    frames = [_frame(t) for t in range(0, 100, 20)]
    frames += [_frame(100, camera=_camera(sight))]
    frames += [_frame(t) for t in range(120, 5000, 20)]
    outs = _run(frames)
    startle = [o for o in outs if o.active == "startle"]
    freeze = [o for o in outs if o.active == "freeze"]
    assert startle and freeze
    assert {o.track for o in startle} == {2} and {o.track for o in freeze} == {2}
    events = [e for o in outs for e in o.events]
    assert [e.track for e in events if e.reflex == "startle"] == [2]
    assert all(o.track == 0 for o in outs if o.active is None)


def test_an_orient_carries_the_track_of_what_moved() -> None:
    sight = C.Sighting(kind="thing", bearing=-0.3, motion=0.6, size=0.15, track=4)
    frames = [_frame(t) for t in range(0, 100, 20)]
    frames += [_frame(t, camera=_camera(sight)) for t in range(100, 1500, 20)]
    outs = _run(frames)
    orient = [o for o in outs if o.active == "orient"]
    assert orient and {o.track for o in orient} == {4}
    assert [e.track for o in outs for e in o.events if e.reflex == "orient"] == [4]


def test_a_sound_a_touch_and_a_light_change_have_no_track() -> None:
    bang = [_frame(t) for t in range(0, 200, 20)]
    bang += [_frame(200, sound=C.SoundSense(level_db=88.0, onset=1.0, kind="bang"))]
    bang += [_frame(t) for t in range(220, 2000, 20)]
    outs = _run(bang)
    assert any(o.active == "startle" for o in outs)
    assert all(o.track == 0 for o in outs) and all(e.track == 0 for o in outs for e in o.events)
    poke = [_frame(t) for t in range(0, 200, 20)]
    poke += [
        _frame(t, touch=C.TouchSense(site="left", pressure=0.8, stroke=0.0))
        for t in range(200, 300, 20)
    ]
    poke += [_frame(t) for t in range(300, 1500, 20)]
    outs = _run(poke)
    assert any(o.active == "withdraw" for o in outs)
    assert all(o.track == 0 for o in outs)


def test_a_covert_look_or_a_visual_onset_without_a_track_is_zero() -> None:
    sight = C.Sighting(kind="hand", bearing=0.1, looming=3.5, size=0.4, motion=0.9, track=0)
    frames = [_frame(t) for t in range(0, 100, 20)]
    frames += [_frame(100, camera=_camera(sight))]
    frames += [_frame(t) for t in range(120, 1500, 20)]
    outs = _run(frames)
    assert any(o.active == "startle" for o in outs) and all(o.track == 0 for o in outs)


def test_the_motor_command_names_the_track_only_while_a_reflex_is_active_in_synth_world() -> None:
    """A shy duck, one loom at 20 s (SynthWorld's hand: track 7): the command's reflex_track is 7 on the startle and the freeze
    after it; a toy that comes into view (tracks 101 and up) gets an orient with its own track; a reflex_track is never set
    without a reflex. Whether a toy comes into view in 90 s depends on where this duck wanders, which the selector's finishing
    rule (fix/finish-bouts) changed for seed 1: so the toy orients are counted over four seeds (every run keeps the track rules)."""
    toy_orients = 0
    for seed in (1, 2, 3, 4):
        world = SynthWorld(seed=seed, family=False)
        world.loom(20_000)
        brain = NeuroBrain(
            Genome.from_axes(SHY, primary="snuggler", branch="snuggler", gd=120), seed=seed
        )
        seen: dict[tuple[str | None, int], int] = {}
        first_hand = None
        for k in range(900):
            t_ms = k * 100
            cmd, _ = brain.step(t_ms, world.frame(t_ms))
            seen[(cmd.reflex, cmd.reflex_track)] = seen.get((cmd.reflex, cmd.reflex_track), 0) + 1
            if cmd.reflex_track:
                assert cmd.reflex is not None
            if cmd.reflex_track == 7 and first_hand is None:
                first_hand = t_ms
            world.apply(cmd, 0.1)
        assert seen.get(("startle", 7), 0) > 0 and seen.get(("freeze", 7), 0) > 0
        assert first_hand is not None and first_hand >= 20_000
        assert all(
            track == 7 for (reflex, track) in seen if reflex in ("startle", "freeze") and track
        )
        others = {key for key in seen if key[1] not in (0, 7)}
        assert all(reflex == "orient" and track > 100 for reflex, track in others)
        toy_orients += len(others)
    assert toy_orients > 0


# ── a loom-watch report has no identity; its source track is only for the reflex's record ──


def _startle_run(sight: C.Sighting) -> list[R.ReflexOutput]:
    frames = [_frame(t) for t in range(0, 100, 20)]
    frames += [_frame(100, camera=_camera(sight))]
    frames += [_frame(t) for t in range(120, 5000, 20)]
    return _run(frames)


def test_a_startle_from_a_watch_report_carries_the_source_track_in_the_reflex_record_only() -> None:
    watch = C.Sighting(
        kind="looming", bearing=0.1, looming=3.5, size=0.4, motion=0.0, track=0, source_track=5
    )
    outs = _startle_run(watch)
    startle = [o for o in outs if o.active == "startle"]
    freeze = [o for o in outs if o.active == "freeze"]
    assert startle and freeze
    assert {o.track for o in startle} == {5} and {o.track for o in freeze} == {5}
    assert [e.track for o in outs for e in o.events if e.reflex == "startle"] == [5]
    # the onset the brain's appraisal reads is as it was: the watch's report has no track
    loom = [p for o in outs for p in o.presented if p.stimulus == "visual:looming"]
    assert loom and all(p.track == 0 for p in loom)


def test_a_sightings_own_track_wins_over_its_source_track_and_none_means_none() -> None:
    own = C.Sighting(
        kind="hand", bearing=0.1, looming=3.5, size=0.4, motion=0.9, track=2, source_track=5
    )
    assert {o.track for o in _startle_run(own) if o.active == "startle"} == {2}
    none = C.Sighting(kind="looming", bearing=0.1, looming=3.5, size=0.4, motion=0.0)
    assert {o.track for o in _startle_run(none) if o.active == "startle"} == {0}


def test_the_threat_target_is_not_given_the_watchs_source_track() -> None:
    """Only the reflex's record gets the fallback: the brain's threat target (what it flees from) keeps what the eye's sighting says."""
    brain = NeuroBrain(Genome.from_axes(SHY, primary="snuggler", branch="snuggler", gd=120), seed=3)
    watch = C.Sighting(
        kind="looming",
        bearing=0.2,
        range_m=None,
        looming=3.5,
        size=0.4,
        motion=0.0,
        track=0,
        source_track=9,
    )
    cmds = []
    for t in range(0, 600, 20):
        cmd, _ = brain.step(t, _frame(t, camera=_camera(watch) if t == 100 else _camera()))
        cmds.append(cmd)
    assert any(c.reflex == "startle" and c.reflex_track == 9 for c in cmds)
    assert brain.appraiser.threat_track == 0
    assert all(
        c.target is None or c.target.track == 0
        for c in cmds
        if c.target is not None and c.target.kind == "threat"
    )
