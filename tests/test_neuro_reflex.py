"""neuro:motor — reflex arcs straight from the senses.

A bang startles (head snaps back and up, the walk stops, then a freeze); the tenth clap gets a
weaker response than the first and the response recovers with rest; a scare makes the duck
jumpier; arousal quickens and strengthens responses; a bold duck startles less and freezes
shorter than a timid one; a sound without a bearing (the real mono mic) makes it listen, never
turn toward a direction it cannot know; a poke is withdrawn from, a stroke leaned into; the ToF
veers, stops at a drop and at a wall; a fall holds the head still for righting.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from duckmind.neuro import reflex as R
from duckmind.neuro.contract import (
    BodySense,
    CameraSense,
    SensorFrame,
    Sighting,
    SoundSense,
    TofSense,
    TouchSense,
)

PARAMS = R.params_from({"boldness": 0.5, "wit": 0.5}, stage="juvenile", side=1)


def _frame(t: int, **kw: object) -> SensorFrame:
    body = kw.pop("body", BodySense(posture="stand"))
    return SensorFrame(t_ms=t, body=body, **kw)  # type: ignore[arg-type]


def _run(
    params: R.ReflexParams,
    frames: list[SensorFrame],
    *,
    arousal: float = 0.5,
    state: R.ReflexState | None = None,
    walking: bool | None = None,
) -> tuple[R.ReflexState, list[R.ReflexOutput]]:
    state = state or R.initial(1)
    outs = []
    for frame in frames:
        state, out = R.step(params, state, frame, arousal=arousal, walking=walking)
        outs.append(out)
    return state, outs


def _quiet(t0: int, t1: int) -> list[SensorFrame]:
    return [_frame(t) for t in range(t0, t1, 20)]


def _bang(t: int, db: float = 88.0, kind: str = "bang", onset: float = 1.0) -> SensorFrame:
    return _frame(t, sound=SoundSense(level_db=db, onset=onset, kind=kind))


def test_a_bang_startles_then_freezes() -> None:
    frames = _quiet(0, 200) + [_bang(200)] + _quiet(220, 5000)
    _, outs = _run(PARAMS, frames)
    events = [e for o in outs for e in o.events]
    assert events and events[0].reflex == "startle" and events[0].stimulus == "sound:bang"
    first = next(i for i, o in enumerate(outs) if o.active == "startle")
    latency_ms = frames[first].t_ms - 200
    assert 20 <= latency_ms <= 160
    peak = max((o for o in outs if o.active == "startle"), key=lambda o: o.head_blend)
    assert peak.head is not None and peak.head.np < -0.3 and peak.head.hp > 0.2
    assert peak.halt and peak.interrupt and peak.vx_cap == 0.0
    assert any(o.active == "freeze" for o in outs[first:])
    assert outs[-1].active is None  # it lets go


def _responses(
    params: R.ReflexParams,
    n: int,
    gap_ms: int,
    kind: str = "clap",
    db: float = 75.0,
    state: R.ReflexState | None = None,
    start: int = 0,
) -> tuple[R.ReflexState, list[float]]:
    frames: list[SensorFrame] = []
    for i in range(n):
        t = start + i * gap_ms
        frames.append(_frame(t, sound=SoundSense(level_db=db, onset=0.8, kind=kind)))
        frames += _quiet(t + 20, t + gap_ms)
    state, outs = _run(params, frames, state=state)
    # (fid:fix: a look the head is already on, or one inside a hold, is covert — no row — but it
    # is still a response: every onset that set a reflex off, fired or covert)
    return state, [
        p.response
        for o in outs
        for p in o.presented
        if p.stimulus == f"sound:{kind}" and p.reflex is not None
    ]


def _mean_responses(
    params: R.ReflexParams, n: int, gap_ms: int, seeds: int = 16, **kw: object
) -> list[float]:
    """The responses averaged over ducks of the same kind (each response has its own trial-to-
    trial variation: reflex.RESP_SIGMA)."""
    rows = []
    for seed in range(seeds):
        _, strengths = _responses(params, n, gap_ms, state=R.initial(seed), **kw)  # type: ignore[arg-type]
        rows.append(strengths)
    assert all(len(r) == n for r in rows)
    return [sum(r[i] for r in rows) / seeds for i in range(n)]


def test_the_tenth_clap_gets_a_weaker_response_and_rest_restores_it() -> None:
    mean = _mean_responses(PARAMS, 10, 2000, seeds=40)
    assert mean[-1] < 0.5 * mean[0]
    assert mean[1] <= 1.15 * mean[0]  # (a small early sensitisation: the dual process)
    assert all(b <= a + 0.05 for a, b in zip(mean[1:], mean[2:], strict=False))
    firsts, agains = [], []
    for seed in range(20):
        state, strengths = _responses(PARAMS, 10, 2000, state=R.initial(seed))
        assert len(strengths) == 10
        assert dict(state.hab)["sound:clap"] < 0.5
        # ten quiet minutes later it answers almost like the first time (spontaneous recovery)
        rested = R.step(PARAMS, state, _frame(10 * 2000 + 600_000))[0]
        _, again = _responses(PARAMS, 1, 2000, state=rested, start=10 * 2000 + 600_020)
        firsts.append(strengths[0])
        agains.append(again[0])
    assert sum(agains) > 0.8 * sum(firsts)


def test_habituation_is_stimulus_specific() -> None:
    state, _ = _responses(PARAMS, 10, 1500)
    _, door = _responses(PARAMS, 1, 1500, kind="door", state=state, start=15_000)
    _, door_fresh = _responses(PARAMS, 1, 1500, kind="door", start=15_000)
    assert door[0] > 0.8 * door_fresh[0]  # the clap habituated, the door did not


def _door(state: R.ReflexState, t: int) -> list:
    frames = [_frame(t, sound=SoundSense(level_db=66.0, onset=0.7, kind="door"))]
    frames += _quiet(t + 20, t + 1500)
    return [e for o in _run(PARAMS, frames, state=state)[1] for e in o.events]


def test_a_scare_makes_it_jumpier_for_a_while() -> None:
    fresh, _ = _run(PARAMS, _quiet(0, 6000))
    scared, _ = _run(PARAMS, [_bang(0, 92.0)] + _quiet(20, 6000))
    calm = _door(fresh, 6000)
    jumpy = _door(scared, 6000)
    assert calm and jumpy
    assert calm[0].reflex == "orient" and jumpy[0].reflex == "startle"  # a door now startles
    assert jumpy[0].gain > 1.2 * calm[0].gain
    assert scared.sens > 0.1 and dict(scared.sens_by)["sound"] > 0.1
    later = R.step(PARAMS, scared, _frame(6000 + 600_000))[0]
    assert later.sens < 0.05 * scared.sens  # it decays
    assert _door(later, 606_020)[0].reflex == "orient"  # calm again


def test_arousal_quickens_and_strengthens() -> None:
    def first(arousal: float) -> tuple[int, float]:
        # a voice: the same reflex (orienting) at both arousal levels, never a startle
        frames = _quiet(0, 100) + [_bang(100, 60.0, "voice", 0.6)] + _quiet(120, 1500)
        _, outs = _run(PARAMS, frames, arousal=arousal)
        onset = next(frames[i].t_ms for i, o in enumerate(outs) if o.active is not None)
        strength = next(e.strength for o in outs for e in o.events)
        return onset, strength

    slow, weak = first(0.1)
    fast, strong = first(0.9)
    assert fast <= slow and strong > 1.3 * weak


def test_a_bold_duck_startles_less_and_freezes_shorter() -> None:
    bold = R.params_from({"boldness": 1.0}, stage="adult")
    timid = R.params_from({"boldness": 0.0}, stage="adult")
    frames = [_bang(0, 80.0, "clap", 0.9)] + _quiet(20, 12_000)

    def measure(params: R.ReflexParams) -> tuple[float, int]:
        _, outs = _run(params, frames)
        strength = outs[0].events[0].strength if outs[0].events else 0.0
        freeze = sum(1 for o in outs if o.active == "freeze")
        return strength, freeze

    s_bold, f_bold = measure(bold)
    s_timid, f_timid = measure(timid)
    assert s_timid > s_bold
    assert f_timid > f_bold


def test_a_sound_without_a_bearing_makes_it_listen_not_turn() -> None:
    frames = (
        _quiet(0, 100)
        + [_frame(100, sound=SoundSense(level_db=58.0, onset=0.5, kind="voice"))]
        + _quiet(120, 2500)
    )
    _, outs = _run(PARAMS, frames)
    active = [o for o in outs if o.active == "orient"]
    assert active
    assert all(o.gaze is None for o in active)  # no direction it could not know
    peak = max(active, key=lambda o: o.head_blend)
    assert peak.head is not None and peak.head.hp > 0.1 and abs(peak.head.hr) > 0.2  # cocked


def test_a_modelled_bearing_or_a_sighting_turns_the_gaze() -> None:
    frames = _quiet(0, 100) + [
        _frame(
            100,
            sound=SoundSense(
                level_db=60.0, onset=0.6, kind="voice", bearing=0.5, bearing_source="model"
            ),
        )
    ]
    frames += _quiet(120, 1500)
    _, outs = _run(PARAMS, frames)
    gazes = [o.gaze for o in outs if o.gaze is not None]
    assert gazes and all(g.bearing == pytest.approx(0.5) for g in gazes)
    sight = Sighting(kind="thing", bearing=-0.3, motion=0.6, size=0.15, track=4)
    frames = _quiet(0, 100) + [
        _frame(t, camera=CameraSense(usable=True, sightings=(sight,))) for t in range(100, 1500, 20)
    ]
    _, outs = _run(PARAMS, frames)
    gazes = [o.gaze for o in outs if o.gaze is not None]
    assert gazes and gazes[-1].bearing == pytest.approx(-0.3)
    assert sum(len(o.events) for o in outs) == 1  # one onset, not one per frame


def test_a_thing_that_left_the_picture_for_a_moment_is_not_new_when_found_again() -> None:
    """The eye gives a thing that leaves the picture for a moment (the head's own nod, a pass that did
    not re-find it) the same track id when it is found again: that is not a new thing, so no second
    orienting look at it (the track is remembered for ``SEEN_KEEP_MS``, not only for the last tick)."""

    def looks(gap_ms: int) -> int:
        there = Sighting(kind="thing", bearing=-0.3, motion=0.6, size=0.15, track=4)
        back = Sighting(kind="thing", bearing=0.6, motion=0.6, size=0.15, track=4)
        frames = _quiet(0, 100)
        frames += [
            _frame(t, camera=CameraSense(usable=True, sightings=(there,)))
            for t in range(100, 500, 20)
        ]
        frames += _quiet(500, 500 + gap_ms)
        frames += [
            _frame(t, camera=CameraSense(usable=True, sightings=(back,)))
            for t in range(500 + gap_ms, 1100 + gap_ms, 20)
        ]
        _, outs = _run(PARAMS, frames + _quiet(1100 + gap_ms, 4600 + gap_ms))
        return sum(1 for o in outs for e in o.events if e.reflex == "orient")

    assert looks(300) == looks(1500) == 1  # the same thing, back: one look in all
    assert looks(R.SEEN_KEEP_MS + 500) == 2  # gone long enough to be a new sighting


def test_looming_at_the_face_startles() -> None:
    sight = Sighting(kind="hand", bearing=0.1, looming=3.5, size=0.4, motion=0.9, track=2)
    frames = _quiet(0, 100) + [_frame(100, camera=CameraSense(usable=True, sightings=(sight,)))]
    frames += _quiet(120, 1500)
    _, outs = _run(PARAMS, frames)
    assert any(
        e.reflex == "startle" and e.stimulus == "visual:looming" for o in outs for e in o.events
    )


def _seen_once(sight: Sighting, *, engaged: bool, sound: SoundSense | None = None) -> list:
    """The reflex's outputs while ``sight`` comes into view (and, 0.4 s later, ``sound`` is heard),
    with the brain's act one of ENGAGED_ACTS or not."""
    state = R.initial(1)
    outs = []
    for t in range(0, 2500, 20):
        kw: dict[str, object] = {}
        if 100 <= t < 400:
            kw["camera"] = CameraSense(usable=True, sightings=(sight,))
        if sound is not None and 500 <= t < 540:
            kw["sound"] = sound
        state, out = R.step(PARAMS, state, _frame(t, **kw), arousal=0.5, engaged=engaged)
        outs.append(out)
    return outs


def test_a_dancing_or_playing_duck_notes_a_new_thing_but_does_not_turn_its_head_to_it() -> None:
    # (the dance's nod and the instrument's hold on the hand are the head's job: props coming in
    # and out of the picture took the head 45-72 % of a dance and silenced the instrument)
    thing = Sighting(kind="thing", bearing=0.5, size=0.1, track=4)
    free = _seen_once(thing, engaged=False)
    assert any(o.active == "orient" for o in free)
    engaged = _seen_once(thing, engaged=True)
    assert not any(o.active for o in engaged) and not any(o.events for o in engaged)
    # … it still counts as seen: the habituation hears it (a covert look, as under a freeze)
    assert any(p.stimulus == "visual:thing" for o in engaged for p in o.presented)
    # a movement it cannot name is the same
    blob = Sighting(kind="motion", bearing=0.5, size=0.1, track=5, motion=0.8)
    assert any(o.active == "orient" for o in _seen_once(blob, engaged=False))
    assert not any(o.active for o in _seen_once(blob, engaged=True))
    # a person or a hand coming into view, a sound, a loom at the face: still taken
    for kind in ("person", "hand"):
        who = Sighting(kind=kind, bearing=0.5, size=0.1, track=6)
        assert any(o.active == "orient" for o in _seen_once(who, engaged=True)), kind
    voice = SoundSense(level_db=62.0, onset=1.0, kind="voice", bearing=0.6, bearing_source="model")
    heard = _seen_once(thing, engaged=True, sound=voice)
    assert any(e.stimulus == "sound:voice" for o in heard for e in o.events)
    loom = Sighting(kind="hand", bearing=0.1, looming=3.5, size=0.4, motion=0.9, track=2)
    assert any(o.active == "startle" for o in _seen_once(loom, engaged=True))


def test_a_poke_is_withdrawn_from_and_a_stroke_leaned_into() -> None:
    poke = [_frame(t, touch=TouchSense(site="left", pressure=0.8)) for t in range(100, 300, 20)]
    _, outs = _run(PARAMS, _quiet(0, 100) + poke + _quiet(300, 2500))
    withdrawn = [o for o in outs if o.active == "withdraw"]
    assert withdrawn
    peak = max(withdrawn, key=lambda o: o.head_blend)
    assert peak.head is not None and peak.head.hy < -0.3  # away from the left side
    assert any(o.escape is not None and o.escape[1] < 0 for o in withdrawn)  # a step away
    stroke = [
        _frame(t, touch=TouchSense(site="head", pressure=0.4, stroke=0.8))
        for t in range(100, 3000, 20)
    ]
    _, outs = _run(PARAMS, _quiet(0, 100) + stroke)
    leaned = [o for o in outs if o.active == "lean_in"]
    assert leaned and max(o.head_blend for o in leaned) > 0.4
    assert leaned[-1].head is not None and leaned[-1].head.np > 0.2
    assert not any(o.halt for o in leaned)


@pytest.mark.parametrize("stroke", [0.21, 0.30, 0.39])
def test_a_sliding_touch_in_the_perceived_stroke_band_is_leaned_into(stroke: float) -> None:
    touch = TouchSense(site="head", pressure=0.6, stroke=stroke)
    _, outs = _run(PARAMS, _quiet(0, 100) + [_frame(t, touch=touch) for t in range(100, 1200, 20)])
    assert any(o.active == "lean_in" for o in outs)
    assert not any(o.active == "withdraw" for o in outs)


def test_stroke_boundary_and_poke_remain_distinct() -> None:
    from duckmind.neuro.touch import is_stroking

    assert not is_stroking(0.2)
    assert is_stroking(0.200001)
    poke = TouchSense(site="head", pressure=0.6, stroke=0.2)
    _, outs = _run(PARAMS, _quiet(0, 100) + [_frame(t, touch=poke) for t in range(100, 600, 20)])
    assert any(o.active == "withdraw" for o in outs)
    assert not any(o.active == "lean_in" for o in outs)


def test_unfamiliar_light_point_touch_can_still_withdraw() -> None:
    light_poke = TouchSense(site="head", pressure=0.3, stroke=0.0)
    _, outs = _run(
        PARAMS, _quiet(0, 100) + [_frame(t, touch=light_poke) for t in range(100, 600, 20)]
    )
    assert any(o.active == "withdraw" for o in outs)
    assert not any(o.active == "lean_in" for o in outs)


def test_fall_righting_keeps_priority_over_a_stroke() -> None:
    down = BodySense(posture="unknown", fallen=True, tilt=1.4)
    stroke = TouchSense(site="head", pressure=0.6, stroke=0.3)
    frames = [
        _frame(t, body=down, events=("fell",) if t == 0 else (), touch=stroke)
        for t in range(0, 1200, 20)
    ]
    _, outs = _run(PARAMS, frames)
    assert all(o.active == "righting" and o.halt for o in outs[100:])


def test_lifted_hold_keeps_priority_over_a_stroke() -> None:
    stroke = TouchSense(site="head", pressure=0.6, stroke=0.3)
    frames = [
        _frame(t, events=("lifted",) if t == 0 else (), touch=stroke) for t in range(0, 800, 20)
    ]
    _, outs = _run(PARAMS, frames)
    assert all(o.active == "lifted" and o.halt for o in outs[100:])


def test_an_expected_touch_is_leaned_into() -> None:
    state = R.initial(3)
    touch = TouchSense(site="left", pressure=0.8)
    state, out = R.step(PARAMS, state, _frame(0))
    state, out = R.step(PARAMS, state, _frame(20, touch=touch), expect_touch=0.8)
    for t in range(40, 800, 20):
        state, out = R.step(PARAMS, state, _frame(t, touch=touch), expect_touch=0.8)
    assert out.active == "lean_in"


def test_tof_veers_caps_and_stops() -> None:
    walking = BodySense(posture="walk", vx=0.45)
    tof = TofSense(present=True, ahead_mm=450, left_mm=None, right_mm=500, age_ms=40)
    _, outs = _run(PARAMS, [_frame(t, body=walking, tof=tof) for t in range(0, 400, 20)])
    assert outs[-1].vyaw_bias > 0.2 and outs[-1].vx_cap == pytest.approx(0.40)
    assert outs[-1].gaze is not None and outs[-1].gaze.bearing > 0
    near = replace(tof, ahead_mm=250)
    _, outs = _run(PARAMS, [_frame(t, body=walking, tof=near) for t in range(0, 200, 20)])
    assert outs[-1].vx_cap == 0.0
    side = TofSense(present=True, ahead_mm=None, left_mm=250, right_mm=None, age_ms=40)
    _, outs = _run(PARAMS, [_frame(t, body=walking, tof=side) for t in range(0, 200, 20)])
    assert outs[-1].vyaw_bias < 0  # something close on its left: turns right


def test_a_drop_stops_and_looks_down() -> None:
    walking = BodySense(posture="walk", vx=0.45)
    frames = [
        _frame(t, body=walking, tof=TofSense(present=True, drop_zones=6, age_ms=40))
        for t in range(0, 1500, 20)
    ]
    _, outs = _run(PARAMS, frames)
    stops = [o for o in outs if o.active == "drop_stop"]
    assert stops and all(o.halt for o in stops[:20])
    assert stops[10].head is not None and stops[10].head.np > 0.3  # looks down (× 0.6 walking)


def test_no_tof_means_no_tof_reflex() -> None:
    walking = BodySense(posture="walk", vx=0.45)
    frames = [
        _frame(t, body=walking, tof=TofSense(present=False, ahead_mm=100, drop_zones=9))
        for t in range(0, 400, 20)
    ]
    _, outs = _run(PARAMS, frames)
    assert all(o.vyaw_bias == 0.0 and o.active is None for o in outs)


def test_a_fall_holds_the_head_for_righting_until_up() -> None:
    down = BodySense(posture="unknown", fallen=True, tilt=1.4)
    frames = [_frame(t, body=down, events=("fell",) if t == 0 else ()) for t in range(0, 3000, 20)]
    frames += _quiet(3000, 6000)
    _, outs = _run(PARAMS, frames)
    during = outs[100]
    assert during.active == "righting" and during.halt and during.interrupt
    assert during.head is not None and during.head_blend == pytest.approx(1.0)
    assert outs[-1].active is None


def test_lifted_until_set_down() -> None:
    frames = [_frame(0, events=("lifted",))] + _quiet(20, 3000)
    frames += [_frame(3000, events=("set_down",))] + _quiet(3020, 5000)
    _, outs = _run(PARAMS, frames)
    assert outs[100].active == "lifted" and outs[100].halt
    assert outs[-1].active is None


def test_a_jolt_while_walking_braces() -> None:
    walking = BodySense(posture="walk", vx=0.45, tilt=0.45)
    _, outs = _run(PARAMS, [_frame(t, body=walking) for t in range(0, 600, 20)])
    assert any(o.active == "stabilise" and o.halt for o in outs)


def test_light_off_freezes_and_light_on_orients_up() -> None:
    lit = CameraSense(usable=True, light="bright")
    dark = CameraSense(usable=True, light="dark")
    frames = [_frame(t, camera=lit) for t in range(0, 200, 20)]
    frames += [_frame(t, camera=dark) for t in range(200, 2500, 20)]
    frames += [_frame(t, camera=lit) for t in range(2500, 4000, 20)]
    _, outs = _run(PARAMS, frames)
    stims = [e.stimulus for o in outs for e in o.events]
    assert "light:off" in stims
    # (fid:fix) the light comes back while the freeze still holds the head: a covert look (seen,
    # no row, nothing moves); once the freeze has let go it looks up
    assert any(p.stimulus == "light:on" for o in outs for p in o.presented)
    frames = [_frame(t, camera=lit) for t in range(0, 200, 20)]
    frames += [_frame(t, camera=dark) for t in range(200, 9000, 20)]
    frames += [_frame(t, camera=lit) for t in range(9000, 11000, 20)]
    _, outs = _run(PARAMS, frames)
    assert "light:on" in [e.stimulus for o in outs for e in o.events]


def test_deterministic_for_a_seed_and_input_stream() -> None:
    frames = _quiet(0, 100) + [_bang(100)] + _quiet(120, 3000)
    a = _run(PARAMS, frames, state=R.initial(9))[1]
    b = _run(PARAMS, frames, state=R.initial(9))[1]
    c = _run(PARAMS, frames, state=R.initial(10))[1]
    assert a == b
    assert a != c  # another duck's (seed's) startle is not the identical one


def test_asleep_needs_a_stronger_stimulus() -> None:
    state = R.initial(2)
    state, awake = R.step(PARAMS, state, _bang(0, 62.0, "door", 0.5))
    state2 = R.initial(2)
    state2, sleeping = R.step(PARAMS, state2, _bang(0, 62.0, "door", 0.5), asleep=True)
    assert awake.events and (
        not sleeping.events or sleeping.events[0].strength < awake.events[0].strength
    )


# ── the brain's side of the one reflex module (habituation, sensitisation, fear) ─────────────


def test_the_brains_habituation_and_sensitisation_drive_the_response() -> None:
    clap = _frame(0, sound=SoundSense(level_db=75.0, onset=0.8, kind="clap"))
    fresh = R.step(PARAMS, R.initial(4), clap, hab={}, sens={})[1]
    tired = R.step(PARAMS, R.initial(4), clap, hab={"sound:clap": 0.2}, sens={})[1]
    jumpy = R.step(PARAMS, R.initial(4), clap, hab={}, sens={"sound:clap": 0.8})[1]
    assert fresh.events and tired.events and jumpy.events
    assert tired.events[0].strength < 0.4 * fresh.events[0].strength
    assert jumpy.events[0].strength > 1.3 * fresh.events[0].strength
    # given the brain's, the module keeps none of its own and reports the onset once
    state, out = R.step(PARAMS, R.initial(4), clap, hab={}, sens={})
    assert state.hab == () and state.sens == 0.0 and state.sens_by == ()
    assert len(out.presented) == 1
    p = out.presented[0]
    assert p.stimulus == "sound:clap" and p.depress and p.reflex in ("startle", "orient")
    assert 0.0 < p.raw <= 1.0 and p.bearing is None


def test_every_processed_onset_is_presented_fired_or_not() -> None:
    faint = _frame(0, sound=SoundSense(level_db=36.0, onset=0.1, kind="voice"))
    out = R.step(PARAMS, R.initial(5), faint, hab={"sound:voice": 0.1}, sens={})[1]
    assert not out.events and len(out.presented) == 1 and out.presented[0].reflex is None
    fall = _frame(0, body=BodySense(posture="unknown", fallen=True, tilt=1.3), events=("fell",))
    out = R.step(PARAMS, R.initial(5), fall, hab={}, sens={})[1]
    assert out.presented and out.presented[0].stimulus == "body:fall"
    assert not out.presented[0].depress and out.presented[0].scare > 0.3  # never habituates


def test_fear_holds_the_freeze() -> None:
    def freeze_ms(fear: float) -> int:
        state = R.initial(6)
        frames = [_bang(0, 90.0)] + _quiet(20, 15_000)
        n = 0
        for frame in frames:
            state, out = R.step(PARAMS, state, frame, fear=fear, hab={}, sens={})
            n += out.active == "freeze"
        return n * 20

    calm, afraid = freeze_ms(0.0), freeze_ms(0.8)
    assert calm > 0
    assert afraid > 2 * calm and afraid <= R.FREEZE_MAX_MS + 1500


def test_a_startle_ends_in_a_freeze_when_afraid() -> None:
    mild = _frame(0, sound=SoundSense(level_db=66.0, onset=0.6, kind="door"))
    params = replace(PARAMS, startle_threshold=0.1)
    outs = [R.step(params, R.initial(7), mild, fear=0.0, hab={}, sens={})]
    assert outs[0][1].events and outs[0][1].events[0].reflex == "startle"
    for fear, want in ((0.0, False), (0.5, True)):
        state = R.initial(7)
        froze = False
        for frame in [mild] + _quiet(20, 4000):
            state, out = R.step(params, state, frame, fear=fear, hab={"sound:door": 0.3}, sens={})
            froze = froze or out.active == "freeze"
        assert froze == want


def test_the_freeze_of_a_flight_is_only_a_beat() -> None:
    """The brain has decided the fright is a flight (2026-10-04): the startle still goes on into its
    freeze, but only for the beat it needs to be seen — the fear does not hold it on — then the body
    is let go for the brain to run; without the flight the same fear holds the freeze for seconds."""

    def freeze_ms(flight: bool) -> int:
        state = R.initial(6)
        n = 0
        for frame in [_bang(0, 90.0)] + _quiet(20, 15_000):
            state, out = R.step(PARAMS, state, frame, fear=0.8, hab={}, sens={}, flight=flight)
            n += out.active == "freeze"
        return n * 20

    held, beat = freeze_ms(False), freeze_ms(True)
    assert 0 < beat <= R.FREEZE_MIN_MS + 1000 and held > 2 * beat  # (+ its rise and release)


def test_the_orienting_gaze_stays_on_the_source_while_the_body_turns() -> None:
    state = R.initial(8)
    onset = _frame(0, sound=SoundSense(level_db=62.0, onset=0.6, kind="voice", bearing=0.4))
    state, _ = R.step(PARAMS, state, onset)
    gaze = None
    for k in range(1, 40):  # the body turns left at 0.5 rad/s: the source drifts to the right
        body = BodySense(posture="walk", vx=0.4, odom=(0.0, 0.0, 0.5 * k * 0.02))
        state, out = R.step(PARAMS, state, _frame(k * 20, body=body))
        gaze = out.gaze or gaze
    assert gaze is not None and gaze.bearing == pytest.approx(0.4 - 0.5 * 39 * 0.02, abs=0.02)


def test_the_double_look_habit_looks_once_more() -> None:
    sight = Sighting(kind="thing", bearing=0.5, motion=0.8, size=0.2, track=3)
    frames = [_frame(0, camera=CameraSense(usable=True, sightings=(sight,)))] + _quiet(20, 9000)
    for habit, want in (("double_look", 1), ("", 0)):
        params = replace(PARAMS, habit=habit)
        _, outs = _run(params, frames)
        again = [e for o in outs for e in o.events if e.stimulus == "habit:double_look"]
        assert len(again) == want


def test_a_bump_is_oops_not_a_scare_and_a_brush_swerves_away() -> None:
    walking = BodySense(posture="walk", vx=0.45)
    state, out = R.step(PARAMS, R.initial(9), _frame(0, body=walking, events=("blocked",)))
    assert out.events[0].reflex == "bump_stop" and out.fear_kick == 0.0 and state.sens == 0.0
    kicked = BodySense(posture="walk", vx=0.45, gyro=(0.0, 0.0, 0.6))  # kicked to its left
    # proprio's "brushed" label alone is mostly noise under continuous steering (live
    # 2026-09-28: 67 of 72 away from any wall): it swerves only when something backs it
    _, alone = R.step(PARAMS, R.initial(9), _frame(0, body=kicked, events=("brushed",)))
    assert not [e for e in alone.events if e.reflex == "veer"] and alone.vyaw_bias == 0.0
    near = TofSense(True, 250, None, None)  # the ToF sees the thing close on its left
    state, out = R.step(PARAMS, R.initial(9), _frame(0, body=kicked, events=("brushed",), tof=near))
    assert out.events[0].reflex == "veer" and out.vyaw_bias < 0.0  # swerves right, away
    for t in range(20, 1000, 20):
        state, out = R.step(PARAMS, state, _frame(t, body=walking))
    assert out.vyaw_bias == 0.0  # a short swerve, not a new heading


def test_a_fall_episode_reports_up_again() -> None:
    down = BodySense(posture="unknown", fallen=True, tilt=1.3)
    frames = [_frame(t, body=down, events=("fell",) if t == 0 else ()) for t in range(0, 2000, 20)]
    frames += [_frame(2000, events=("righted",))] + _quiet(2020, 3000)
    _, outs = _run(PARAMS, frames)
    assert outs[50].fallen and outs[50].hold == "righting"
    ups = [e for o in outs for e in o.events if e.stimulus == "body:righted"]
    assert len(ups) == 1 and not outs[-1].fallen
    falls = [e for o in outs for e in o.events if e.stimulus == "body:fall"]
    assert len(falls) == 1  # one fall, one event (not one per tick)


def test_params_from_the_genome_are_duck_cores() -> None:
    from duckmind.neuro.genome import Genome

    bold = Genome.from_axes({"boldness": 1.0}, primary="explorer", gd=200)
    timid = Genome.from_axes({"boldness": 0.0}, primary="snuggler", gd=200, habit="double_look")
    pb, pt = R.params_from_genome(bold), R.params_from_genome(timid)
    assert pb.startle_gain < pt.startle_gain and pb.freeze_ms < pt.freeze_ms
    assert pt.habit == "double_look" and pb.boldness == 1.0
    assert pt.hab_rate == pytest.approx(timid.p.hab_rate)
    assert pt.sens_decay_s == pytest.approx(timid.p.sens_tau_s)
    seeded = Genome.from_seed(7, gd=30)
    assert R.params_from_genome(seeded).side == seeded.side


def test_reflex_head_poses_follow_the_bodys_posture() -> None:
    """Live physics 2026-09-27: a startle head snap during a rise made the duck fall — getting up
    or sitting down no snap at all (the head stays put). Seated and standing the WHOLE snap
    (fid:motor 2026-09-28: at half size standing the owner saw no startle; the A100 validated the
    head's range standing and walking, and the motor clamps every pose into it)."""

    def peak(posture: str) -> R.ReflexOutput:
        body = BodySense(posture=posture)
        frames = [_frame(t, body=body) for t in range(0, 100, 20)]
        frames += [_frame(100, body=body, sound=SoundSense(level_db=90.0, onset=1.0, kind="bang"))]
        frames += [_frame(t, body=body) for t in range(120, 800, 20)]
        _, outs = _run(PARAMS, frames)
        startles = [o for o in outs if o.active == "startle"]
        assert startles
        return max(startles, key=lambda o: o.head_blend)

    sit, stand, rising = peak("sit"), peak("stand"), peak("transition")
    assert sit.head is not None and stand.head is not None
    assert R.posture_scale("stand", "startle") == R.posture_scale("sit", "startle") == (1.0, 1.0)
    assert stand.head.hp >= 0.7 and sit.head.hp >= 0.7  # the head snaps UP (the look-up)
    assert stand.head_blend == pytest.approx(1.0)
    assert rising.head_blend == 0.0  # no snap at all while getting up
    assert rising.halt and rising.interrupt  # it still stops what it was doing
    assert R.posture_scale("transition", "withdraw") == (0.2, 0.4)
    assert R.posture_scale("unknown", "righting") == (1.0, 1.0)  # righting holds HOME


def _body_loop(
    params: R.ReflexParams,
    ticks: int,
    tof_at: object = None,
    first_events: tuple[str, ...] = (),
    **kw: object,
) -> list[tuple[R.ReflexOutput, float]]:
    """The reflexes with a crude body: it walks while it may, stops (after the legs' tail) when
    the reflexes cap it at 0, and turns while they arc it away (yaw rate 0.45 × the bias) —
    so a turn-away can end on its consequences. ``tof_at(yaw)``: the ToF for the heading."""
    state = R.initial(1)
    yaw = 0.0
    speed = 0.45
    out_rows = []
    for k in range(ticks):
        t = 20 * k
        walking = speed > 0.0
        body = BodySense(posture="walk" if walking else "stand", vx=speed, odom=(0.0, 0.0, yaw))
        tof = tof_at(yaw) if callable(tof_at) else TofSense()
        frame = _frame(t, body=body, tof=tof, events=first_events if k == 0 else ())
        state, out = R.step(params, state, frame, **kw)  # type: ignore[arg-type]
        out_rows.append((out, yaw))
        if out.vx_cap == 0.0:
            speed = 0.0
        elif out.turning_away:
            speed = 0.35
            yaw += 0.45 * out.vyaw_bias * 0.02
        else:
            speed = 0.45
    return out_rows


def test_a_wall_right_ahead_stops_then_turns_away_never_stuck() -> None:
    """Nose to a wall (ToF ≤ 300 mm): a stop until the legs have stopped, then an arc away
    (never a permanent stop: this body cannot turn in place, and the body link only passes arcs
    with |vyaw| ≥ 0.5) — which ends when it has turned enough and the way ahead is clear, not
    on a timer (review 2026-09-28)."""

    def tof(yaw: float) -> TofSense:
        if abs(yaw) < 0.7:  # the wall in front until it has turned ~40°
            return TofSense(present=True, ahead_mm=250, left_mm=900, right_mm=300, age_ms=40)
        return TofSense(present=True, ahead_mm=1200, left_mm=1200, right_mm=600, age_ms=40)

    rows = _body_loop(PARAMS, 400, tof, way_clear=False)
    outs = [o for o, _ in rows]
    assert outs[3].vx_cap == 0.0  # it stops first …
    arcs = [i for i, o in enumerate(outs) if o.turning_away]
    assert arcs and outs[arcs[0]].vx_cap == pytest.approx(0.40)
    assert all(outs[i].vyaw_bias >= R.TURN_AWAY_VYAW - 1e-9 for i in arcs)  # … away to its left
    assert all(abs(rows[i][1]) < 0.7 for i in arcs[:5])  # still facing the wall as it arcs
    assert not outs[-1].turning_away  # and once it has turned enough, it walks on
    assert abs(rows[-1][1]) >= R.TURN_NEED_RAD[0]


def test_a_stall_without_tof_turns_away_after_the_bump() -> None:
    rows = _body_loop(PARAMS, 400, first_events=("blocked",))
    outs = [o for o, _ in rows]
    assert outs[0].events[0].reflex == "bump_stop" and outs[10].vx_cap == 0.0
    arcs = [i for i, o in enumerate(outs) if o.turning_away]
    assert arcs and 0.4 <= arcs[0] * 0.02 <= 1.2  # the arc follows the stop
    assert abs(outs[arcs[0]].vyaw_bias) >= R.TURN_AWAY_VYAW - 1e-9
    turned = abs(rows[arcs[-1]][1])
    assert turned >= R.TURN_DEFAULT_RAD - 0.05  # it turned about a radian away …
    assert not outs[-1].turning_away  # … and then went back to its own path


def test_the_turn_away_side_comes_from_what_it_saw_not_a_coin() -> None:
    """A stall with no heading kick: it turns to the side its camera saw open (``free_side``),
    and with no view to its own (innate) side — the same every time, never a coin."""
    for free_side in (1, -1):
        rows = _body_loop(PARAMS, 120, first_events=("blocked",), free_side=free_side)
        arcs = [o for o, _ in rows if o.turning_away]
        assert arcs and all(o.vyaw_bias * free_side > 0 for o in arcs)
    sides = set()
    for seed in range(8):
        state = R.initial(seed)
        frame = _frame(0, body=BodySense(posture="walk", vx=0.45), events=("blocked",))
        state, _ = R.step(PARAMS, state, frame)
        sides.add(state.avoid[0])
    assert sides == {PARAMS.side}


def test_a_moderate_clap_startles_a_timid_duck_not_a_bold_one() -> None:
    """Diversity: boldness moves the startle threshold (the first brain's live finding: with one
    threshold every temperament startled at the first clap)."""
    from duckmind.neuro.genome import Genome

    clap = _frame(0, sound=SoundSense(level_db=72.0, onset=0.8, kind="clap"))
    bang = _frame(0, sound=SoundSense(level_db=92.0, onset=1.0, kind="bang"))

    def startled(boldness: float, frame: SensorFrame) -> int:
        genome = Genome.from_axes({"boldness": boldness}, primary="explorer", gd=200)
        params = R.params_from_genome(genome)
        hits = 0
        for seed in range(30):
            out = R.step(params, R.initial(seed), frame, hab={}, sens={})[1]
            hits += any(e.reflex == "startle" for e in out.events)
        return hits

    # (each response has its own variation: RESP_SIGMA)
    assert startled(0.1, clap) >= 27 and startled(0.9, clap) <= 3
    assert startled(0.9, bang) >= 24  # a real bang still startles the boldest (most times)


def test_a_sudden_loud_bang_startles_every_temperament_every_time() -> None:
    """The startle floor (owner 2026-10-05: 「嚇一跳也是」): a bang of raw size >= 0.9 the duck is not
    yet used to (habituation level >= 0.75) startles the boldest and the calmest every time, whatever
    its arousal and this onset's own variation; milder sounds, a bang it has got used to and the
    sounds that never startle (its own name, the family's voice) are decided as before."""
    from duckmind.neuro.genome import Genome

    bang = _frame(0, sound=SoundSense(level_db=94.0, onset=1.0, kind="bang"))
    unnamed = _frame(0, sound=SoundSense(level_db=90.0, onset=1.0, kind=None))

    def startled(boldness: float, frame: SensorFrame, *, arousal: float, level: float) -> int:
        genome = Genome.from_axes({"boldness": boldness}, primary="explorer", gd=200)
        params = R.params_from_genome(genome)
        hits = 0
        for seed in range(40):
            out = R.step(
                params,
                R.initial(seed),
                frame,
                arousal=arousal,
                hab={"sound:bang": level, "sound:noise": level},
                sens={},
            )[1]
            hits += any(e.reflex == "startle" for e in out.events)
        return hits

    for boldness in (0.0, 0.5, 0.95):
        for arousal in (0.0, 0.5, 1.0):
            assert startled(boldness, bang, arousal=arousal, level=0.8) == 40
            assert startled(boldness, unnamed, arousal=arousal, level=0.8) == 40
    # used to it (level under the floor): decided by the response as before, the bold shrug it off
    assert startled(0.95, bang, arousal=0.0, level=0.5) < 40
    # raw under the floor: a clap at 75 dB is not forced on the bold
    clap = _frame(0, sound=SoundSense(level_db=75.0, onset=0.8, kind="clap"))
    assert startled(0.95, clap, arousal=0.0, level=1.0) < 40
    # a sound that never startles never does, however loud
    name = _frame(0, sound=SoundSense(level_db=94.0, onset=1.0, kind="name"))
    out = R.step(PARAMS, R.initial(3), name, hab={}, sens={})[1]
    assert not any(e.reflex == "startle" for e in out.events)


def test_a_flickering_contact_is_one_touch_not_many() -> None:
    """Rubbing along a wall: physics contact flickers off/on every ~0.2 s. One withdrawal, then
    no more until the side has been free for a while (else the duck is held by its own reflex)."""
    frames = []
    for t in range(0, 6000, 20):
        on = (t // 100) % 2 == 0  # 0.1 s on, 0.1 s off
        touch = TouchSense(site="right", pressure=0.5) if on else TouchSense()
        frames.append(_frame(t, touch=touch))
    frames += _quiet(6000, 8000)
    frames += [
        _frame(t, touch=TouchSense(site="right", pressure=0.5)) for t in range(8000, 8200, 20)
    ]
    _, outs = _run(PARAMS, frames)
    onsets = [e for o in outs for e in o.events if e.stimulus == "touch:right"]
    assert len(onsets) == 2  # the rubbing once, and the new touch after 2 s free


def test_its_own_leg_bumping_something_is_a_brush_not_a_touch() -> None:
    """Reafference: a side contact while it walks, with its body feeling the brush, is its own
    doing — a swerve, never a withdrawal that holds the body (live :8891/:8892: leg contacts
    along walls held it for minutes)."""
    walking = BodySense(posture="walk", vx=0.45)
    frames = [_frame(0, body=walking, events=("brushed",))]
    frames += [
        _frame(t, body=walking, touch=TouchSense(site="left", pressure=0.6))
        for t in range(20, 400, 20)
    ]
    frames += [_frame(t, body=walking) for t in range(400, 1500, 20)]
    _, outs = _run(PARAMS, frames, walking=True)
    reflexes = [e.reflex for o in outs for e in o.events]
    assert "withdraw" not in reflexes and "veer" in reflexes
    assert not any(o.halt for o in outs)
    assert min(o.vyaw_bias for o in outs) < -0.3  # away from the left side


def test_a_well_known_touch_gets_a_look_not_a_flinch() -> None:
    touch = _frame(0, touch=TouchSense(site="right", pressure=0.7))
    fresh = R.step(PARAMS, R.initial(3), touch, hab={}, sens={})[1]
    known = R.step(
        PARAMS, R.initial(3), touch, hab={"touch:right": 0.2}, sens={"touch:right": 0.9}
    )[1]
    assert fresh.events and fresh.events[0].reflex == "withdraw"
    assert known.events and known.events[0].reflex == "orient"
    later = R.step(PARAMS, R.initial(3), touch, hab={"touch:right": 0.2}, sens={"touch:right": 0.9})
    state = later[0]
    outs = []
    for t in range(20, 1500, 20):
        state, out = R.step(PARAMS, state, _frame(t), hab={"touch:right": 0.2}, sens={})
        outs.append(out)
    assert not any(o.halt for o in outs)
    assert any(o.gaze is not None and o.gaze.bearing < 0 for o in outs)  # it looks to its right


def test_a_harmless_repeated_clap_does_not_feed_its_own_jumpiness() -> None:
    """The scare is judged without the sensitisation it caused: the 2nd and 3rd of a series of
    moderate claps are not stronger than the 1st (the first brain's merged-code measurement)."""
    mean = _mean_responses(PARAMS, 10, 8000, seeds=40, db=72.0)
    # (a small early sensitisation is the dual process's shape — pet:etho B1 — not a runaway:
    # 1.034× without the per-onset variation, ~1.07× with it, 80 ducks)
    assert max(mean[1:3]) <= 1.12 * mean[0]
    assert mean[-1] < 0.6 * mean[0]


def test_the_same_stimulus_never_gives_quite_the_same_response() -> None:
    """Review 2026-09-28: the response sizes repeated to two decimals. Each onset now draws its
    own variation (pet:etho: CV 0.1–0.3), the average still follows the stimulus."""
    firsts = [_responses(PARAMS, 1, 2000, state=R.initial(seed))[1][0] for seed in range(40)]
    mean = sum(firsts) / len(firsts)
    sd = (sum((x - mean) ** 2 for x in firsts) / len(firsts)) ** 0.5
    assert 0.1 <= sd / mean <= 0.35
    assert len({round(x, 2) for x in firsts}) >= 20


def test_a_poke_while_walking_is_still_a_touch() -> None:
    walking = BodySense(posture="walk", vx=0.45)
    frames = [_frame(0, body=walking)]
    frames += [
        _frame(t, body=walking, touch=TouchSense(site="left", pressure=0.8))
        for t in range(20, 200, 20)
    ]
    frames += [_frame(t, body=walking) for t in range(200, 1500, 20)]
    _, outs = _run(PARAMS, frames, walking=True)
    assert any(e.reflex == "withdraw" for o in outs for e in o.events)


def test_after_a_drop_stop_it_arcs_away_from_the_edge() -> None:
    def tof(yaw: float) -> TofSense:
        if abs(yaw) < 0.6:  # the edge ahead until it has turned away from it
            return TofSense(present=True, drop_zones=6, left_mm=None, right_mm=None, age_ms=40)
        return TofSense(present=True, ahead_mm=1200, left_mm=1200, right_mm=1200, age_ms=40)

    rows = _body_loop(PARAMS, 800, tof)  # (it stops again at the edge while it still sees it)
    outs = [o for o, _ in rows]
    stops = [i for i, o in enumerate(outs) if o.active == "drop_stop" and o.halt]
    arcs = [i for i, o in enumerate(outs) if o.turning_away and o.vx_cap != 0.0]
    assert stops and arcs and arcs[0] > stops[0]  # first the stop, then the arc away
    assert all(abs(outs[i].vyaw_bias) >= R.TURN_AWAY_VYAW - 1e-9 for i in arcs)
    assert not outs[-1].turning_away and abs(rows[-1][1]) >= 0.6  # away from the edge, done


def test_still_down_is_still_righting_whatever_the_labels_say() -> None:
    """Live 2026-09-28: after a fall the duck lay wedged at 45°; robotd's 60° verdict cleared and
    proprio said "righted". The IMU says it is not up: the righting hold stays until it is."""
    down = BodySense(posture="unknown", fallen=True, tilt=1.1)
    wedged = BodySense(posture="unknown", fallen=False, tilt=0.78)
    frames = [_frame(0, body=down, events=("fell",))]
    frames += [_frame(t, body=down) for t in range(20, 3000, 20)]
    frames += [_frame(3000, body=wedged, events=("righted",))]
    frames += [_frame(t, body=wedged) for t in range(3020, 12000, 20)]
    frames += [_frame(t) for t in range(12000, 15000, 20)]  # really up (standing, tilt 0)
    _, outs = _run(PARAMS, frames)
    assert all(o.fallen and o.hold == "righting" for o in outs[160:590])  # 3.2 s … 11.8 s
    ups = [e for o in outs for e in o.events if e.stimulus.startswith("body:righted")]
    assert len(ups) == 1 and ups[0].t_ms >= 12000
    assert not outs[-1].fallen and outs[-1].hold is None


def test_the_reflex_words_say_the_side_in_plain_chinese() -> None:
    """The owner reads the reflexes on the page: 「頭轉向右邊」, and a mono microphone gives no
    side (it lifts its head to listen); the page adds the durations when a hold ends."""
    assert R._zh("orient", "visual:motion", -0.5) == "頭轉向右邊"
    assert R._zh("orient", "visual:person", 0.6) == "頭轉向左邊"
    assert R._zh("orient", "sound:clap", None) == "側過頭聽（聽不出方向）"
    assert R._zh("withdraw", "touch:left") == "從左邊縮開"
    assert R._zh("lean_in", "touch:head") == "頭頂被摸，靠過去"
    assert "嚇一跳" in R._zh("startle", "sound:bang", None)
