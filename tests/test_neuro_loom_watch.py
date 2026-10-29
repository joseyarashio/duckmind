"""The loom watch (duckmind.world.neuro_senses.LoomWatch), pure: no MuJoCo.

Machine verification v3, finding ②: a hand comes at the face in ~0.45 s, the camera's look clock gives
a result every ~0.55 s, so the fast approach fell between two looks and the duck mostly did not startle.
The loom watch measures the expansion of the nearest large blob on every world frame (10 Hz), outside
the look governor, and reports nothing but ``looming``. Here: a 0.45-s loom sampled at the world rate is
found in a scene whose look clock misses it; a slow approach, a still hand, a hidden or far-too-small
blob, the dark and a saccade are not looms; a quiet scene is bit-for-bit what it was; the watch costs no
look; the loom's own look result replaces the watch's for the same frame."""

from __future__ import annotations

import math
from typing import Any

import pytest
from test_neuro_senses import Doors, World, robot_state, run, samples_every, thing, view, wilson

from duckmind.neuro import contract as C
from duckmind.world import neuro_integration as integrated
from duckmind.world import neuro_senses as ns

# the world's hand_loom, as the physics world samples it (diagnosis 2026-10-02: expansion
# 2.7 / 8.4 / 12.6 per second at +0.4 / +0.5 / +0.6 s, the hand 0.75 m away, then 9 cm from the beak)
HAND_LOOM = {
    0.0: (0.75, 0.0, 0.12),
    0.1: (0.75, 0.0, 0.12),
    0.2: (0.75, 0.0, 0.12),
    0.3: (0.68, 0.6, 0.13),
    0.4: (0.55, 1.5, 0.16),
    0.5: (0.35, 2.9, 0.27),
    0.6: (0.18, 2.3, 0.50),
    0.7: (0.09, 0.0, 0.57),
}


def hand(dist: float, vr: float, size: float, **kw: Any) -> dict[str, Any]:
    return thing(id="hand", kind="hand", dist=dist, vr=vr, size=size, rng=None, **kw)


def loom_samples(onset: float, end: float) -> list[dict[str, Any]]:
    """View samples every 0.1 s up to ``end``; the 0.45-s hand loom starts at ``onset``."""
    out = []
    for sample in samples_every(0.0, end, lambda t: []):
        k = round(sample["t"] - onset, 1)
        if k in HAND_LOOM:
            sample = {**sample, "seen": [hand(*HAND_LOOM[k])]}
        out.append(sample)
    return out


def quiet_look_times(seed: int, seconds: float = 12.0) -> list[float]:
    """When the camera's look clock looks (it does not depend on what the scene shows)."""
    hub, _ = run(
        World(samples=samples_every(0.0, seconds, lambda t: [])),
        seconds,
        hub=ns.SenseHub(seed=seed),
    )
    return [round(t, 1) for t in hub.look_times]


def onset_between_two_looks(seed: int) -> float:
    """An onset such that the loom's 0.4–0.6 s window falls between two looks, 0.6 s apart (looks at
    +0.3 s, where the hand only starts to move, and at +0.9 s, where the hand is gone)."""
    looks = quiet_look_times(seed)
    for a, b in zip(looks, looks[1:], strict=False):
        if b - a >= 0.59 and a >= 0.4:
            return round(a - 0.3, 1)
    raise AssertionError(f"seed {seed}: no 0.6-s gap between looks in {looks}")


def strong(frames: list[C.SensorFrame], lo_ms: int = 0, hi_ms: int = 10**9) -> list[C.SensorFrame]:
    """Frames in [lo_ms, hi_ms] whose camera gives a looming sighting at the reflex's level (1.2 /s)."""
    return [
        f
        for f in frames
        if lo_ms <= f.t_ms <= hi_ms and any(s.looming >= 1.2 for s in f.camera.sightings)
    ]


SEEDS = (1, 2, 3, 5, 6)  # seeds whose clocks have a 0.6-s gap to put the loom into


@pytest.mark.parametrize("seed", SEEDS)
def test_a_fast_loom_between_two_looks_reaches_the_brain(seed: int) -> None:
    """The scene the look clock misses: the loom is found at the world rate, a detector latency later."""
    onset = onset_between_two_looks(seed)
    world = World(samples=loom_samples(onset, onset + 3.0))
    _, frames = run(world, onset + 3.0, hub=ns.SenseHub(seed=seed))
    window = (int((onset + 0.4) * 1000), int((onset + 1.0) * 1000))
    hit = strong(frames, *window)
    assert hit, "no looming at the reflex's level reached the brain"
    first = hit[0].t_ms - int(onset * 1000)
    # the first expanding frame is +0.4 s (a miss, p 0.03, moves it a frame): plus the detector's
    # latency (70–140 ms) and the 20-ms brain tick
    assert 400 + ns.DETECT_LATENCY_MS[0] <= first <= 500 + ns.DETECT_LATENCY_MS[1] + 20
    sights = [s for s in hit[0].camera.sightings if s.looming >= 1.2]
    assert all(s.kind == "looming" and s.range_m is None and s.track == 0 for s in sights)
    # it is the hand's growth that is reported, to the noise of the sensor
    assert all(2.0 < s.looming < 14.0 and s.size > 0.1 for s in sights)


def test_the_scene_above_is_one_the_look_clock_alone_misses() -> None:
    """The control of the test above: with the watch off, the same scene gives only the weak 0.88 /s."""
    seed = 2
    onset = onset_between_two_looks(seed)
    world = World(samples=loom_samples(onset, onset + 3.0))
    _, frames = run(world, onset + 3.0, hub=ns.SenseHub(seed=seed, loom_watch_on=False))
    assert not strong(frames)
    assert any(s.kind == "looming" and s.looming < 1.2 for f in frames for s in f.camera.sightings)


def test_the_loom_is_found_when_the_world_is_copied_only_every_fifth_of_a_second() -> None:
    """The rig and the live driver hand the senses two new frames at a time."""
    seed = 2
    onset = onset_between_two_looks(seed)
    world = World(samples=loom_samples(onset, onset + 3.0))
    hub = ns.SenseHub(seed=seed)
    frames = []
    for k in range(int((onset + 3.0) * 50)):
        t = k * 0.02
        doors = Doors(
            t_ms=int(round(t * 1000)), world=world.section(math.floor(t / 0.2 + 1e-9) * 0.2)
        )
        frames.append(hub(doors))
    assert strong(frames, int((onset + 0.4) * 1000), int((onset + 1.2) * 1000))


@pytest.mark.parametrize("seed", SEEDS)
def test_a_slow_approach_and_a_still_hand_are_not_looming(seed: int) -> None:
    def slow(t: float) -> list[dict[str, Any]]:  # 0.1 m/s at 0.75 m: 0.13 /s, well under 0.6 /s
        return [hand(0.75, 0.1, 0.12)] if t >= 1.0 else []

    def still(t: float) -> list[dict[str, Any]]:
        return [hand(0.4, 0.0, 0.3)] if t >= 1.0 else []

    for scene in (slow, still):
        _, frames = run(
            World(samples=samples_every(0.0, 6.0, scene)), 6.0, hub=ns.SenseHub(seed=seed)
        )
        assert [s for f in frames for s in f.camera.sightings if s.kind == "looming"] == []


def test_a_quiet_scene_is_bit_for_bit_what_it_was() -> None:
    """No extra detections: with a moving blob, a person walking up slowly and things resting, the
    frames with the watch on are the frames with it off."""

    def scene(t: float) -> list[dict[str, Any]]:
        return [
            thing(om=0.5, rng=None),
            thing(id="thing:book", kind="thing", dist=1.1, vr=0.0, om=0.0, size=0.2),
            thing(id="person:owner", kind="person", dist=2.0 - 0.15 * t, vr=0.15, om=0.1, size=0.4),
            hand(0.6, 0.2, 0.15),  # a hand easing towards the duck at 0.33 /s
        ]

    world = World(samples=samples_every(0.0, 8.0, scene))
    on_hub, on = run(world, 8.0, hub=ns.SenseHub(seed=3))
    _, off = run(world, 8.0, hub=ns.SenseHub(seed=3, loom_watch_on=False))
    assert on == off
    assert on_hub.loom_watch.reports == 0 and on_hub.loom_watch.seen == 0
    assert on_hub.loom_watch.frames >= 70  # it did look at every frame


def test_the_watch_spends_no_look_and_changes_nothing_but_the_looming_sightings() -> None:
    seed = 2
    onset = onset_between_two_looks(seed)
    world = World(samples=loom_samples(onset, onset + 3.0))
    on_hub, on = run(world, onset + 3.0, hub=ns.SenseHub(seed=seed))
    off_hub, off = run(world, onset + 3.0, hub=ns.SenseHub(seed=seed, loom_watch_on=False))
    assert list(on_hub.look_times) == list(off_hub.look_times)
    assert (on_hub.looks, on_hub.looks_refused) == (off_hub.looks, off_hub.looks_refused)
    assert on_hub.looks_per_s() == off_hub.looks_per_s() <= 2.6
    for a, b in zip(on, off, strict=True):
        assert a.camera.age_ms == b.camera.age_ms  # the look's own clock, untouched
        assert (a.camera.light, a.camera.floor, a.camera.motion) == (
            b.camera.light,
            b.camera.floor,
            b.camera.motion,
        )
        assert a.body == b.body and a.sound == b.sound and a.touch == b.touch
        extra = [s for s in a.camera.sightings if s not in b.camera.sightings]
        assert all(s.kind == "looming" and s.track == 0 for s in extra)
        assert all(s in a.camera.sightings for s in b.camera.sightings)  # nothing taken away


def test_the_report_comes_after_the_detector_latency_and_is_held_two_frames() -> None:
    """One frame with a hand at 4 /s (no look on it: seed 1 looks at 0.0, 0.5, 1.1): the report is not
    there before the latency, is there at once after, and is gone when stale."""
    world = World(
        samples=samples_every(
            0.0, 3.0, lambda t: [hand(0.3, 1.2, 0.3)] if abs(t - 1.0) < 1e-6 else []
        )
    )
    _, frames = run(world, 3.0, hub=ns.SenseHub(seed=1), tick_ms=10)
    shown = [f.t_ms for f in frames if any(s.kind == "looming" for s in f.camera.sightings)]
    assert shown, "the single frame was missed (p 0.03)"
    assert 1000 + ns.DETECT_LATENCY_MS[0] <= shown[0] <= 1000 + ns.DETECT_LATENCY_MS[1] + 10
    assert shown[-1] - shown[0] <= ns.LOOM_WATCH_HOLD_MS
    assert shown[-1] - shown[0] >= ns.LOOM_WATCH_HOLD_MS - 20  # held the whole two frames


def test_the_looks_own_looming_replaces_the_watchs_for_the_same_frame() -> None:
    """seed 1 looks at 1.1 s: the frame is seen twice (watch and look); the brain sees it once."""
    world = World(
        samples=samples_every(
            0.0, 3.0, lambda t: [hand(0.3, 1.2, 0.3)] if abs(t - 1.1) < 1e-6 else []
        )
    )
    hub, frames = run(world, 3.0, hub=ns.SenseHub(seed=1), tick_ms=10)
    assert 1.1 in [round(t, 1) for t in hub.look_times]
    counts = [sum(s.kind == "looming" for s in f.camera.sightings) for f in frames]
    assert max(counts) == 1 and sum(counts) > 0
    # the look's labelled sighting (a track) is what is held once it has arrived
    late = [f for f in frames if 1100 + ns.DETECT_LATENCY_MS[1] + 20 <= f.t_ms <= 1100 + 600]
    assert late and all(
        [s.track > 0 for s in f.camera.sightings if s.kind == "looming"] == [True] for f in late
    )


def test_a_watch_report_carries_expansion_only_no_image_motion() -> None:
    """The reflex arc reads a sighting with image motion as a new moving thing (an orient); the
    watch reports growth only, so a weak looming (0.9 /s) never reaches the motion branch."""

    def scene(t: float) -> list[dict[str, Any]]:
        if t < 1.0 or t > 3.0:
            return []
        return [
            hand(0.4, 0.36, 0.2, om=0.5),
            thing(id="thing:b", kind="thing", dist=0.3, vr=1.5, size=0.2, om=0.6, rng=None),
        ]

    _, frames = run(World(samples=samples_every(0.0, 4.0, scene)), 4.0, hub=ns.SenseHub(seed=1))
    watch = [s for f in frames for s in f.camera.sightings if s.kind == "looming" and s.track == 0]
    assert watch and all(s.motion == 0.0 for s in watch)
    assert any(s.looming > 3.0 for s in watch)  # the 5 /s blob, not the 0.9 /s hand


def test_the_nearest_blob_is_the_one_with_the_highest_expansion_rate() -> None:
    def scene(t: float) -> list[dict[str, Any]]:
        if abs(t - 1.0) > 1e-6:
            return []
        return [
            thing(id="thing:slow", kind="thing", dist=0.9, vr=0.9, size=0.2, rng=None),  # 1.0 /s
            hand(0.3, 0.9, 0.3),  # 3.0 /s
            thing(id="thing:far", kind="thing", dist=2.0, vr=1.6, size=0.2, rng=None),  # 0.8 /s
        ]

    _, frames = run(World(samples=samples_every(0.0, 3.0, scene)), 3.0, hub=ns.SenseHub(seed=1))
    got = [s for f in frames for s in f.camera.sightings if s.kind == "looming" and s.track == 0]
    assert got and all(2.6 < s.looming < 3.4 for s in got)


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param(hand(0.3, 1.2, 0.3, vis=0), id="hidden"),
        pytest.param(hand(0.3, 1.2, 0.04), id="too_small"),
        pytest.param(hand(0.3, 1.2, 0.3, lux=0.1), id="black"),
        pytest.param(hand(0.3, 1.2, 0.3, lux=1.0), id="dark_wants_0.10_rad"),
        pytest.param(
            thing(id="nest", kind="nest", dist=0.3, vr=1.2, size=0.3, rng=None), id="the_dock"
        ),
        pytest.param(hand(0.3, -1.2, 0.3), id="going_away"),
        pytest.param(hand(0.3, 0.12, 0.3), id="0.4_per_s_is_no_loom"),
    ],
)
def test_things_that_are_no_loom_are_not_reported(entry: dict[str, Any]) -> None:
    entry = {**entry}
    if entry.get("lux") == 1.0:
        entry["size"] = 0.09  # the dark band wants 0.10 rad
    world = World(samples=samples_every(0.0, 5.0, lambda t: [entry]))
    _, frames = run(world, 5.0, hub=ns.SenseHub(seed=4))
    assert [s for f in frames for s in f.camera.sightings if s.kind == "looming"] == []


def test_a_saccade_blinds_the_watch_and_stepping_smears_most_of_its_frames() -> None:
    world = World(samples=samples_every(0.0, 20.0, lambda t: [hand(0.3, 1.2, 0.3)]))
    hub, frames = run(world, 20.0, state=robot_state(gyro=(0.0, 0.0, 2.0)), hub=ns.SenseHub(seed=5))
    assert hub.loom_watch.seen > 150 and hub.loom_watch.reports == 0
    assert not [s for f in frames for s in f.camera.sightings if s.kind == "looming"]
    still, _ = run(world, 20.0, hub=ns.SenseHub(seed=5))
    # (a hand 10 cm away closing at 1.2 m/s: 12 /s, far more than a walking duck gives a still thing)
    close = World(samples=samples_every(0.0, 20.0, lambda t: [hand(0.1, 1.2, 0.3)]))
    walking, _ = run(close, 20.0, state=robot_state(policy="walk", vx=0.4), hub=ns.SenseHub(seed=5))
    lo, hi = wilson(still.loom_watch.reports, still.loom_watch.seen)
    assert lo <= ns.P_SEE["bright"] <= hi  # a still camera finds the moving blob
    lo, hi = wilson(walking.loom_watch.reports, walking.loom_watch.seen)
    assert lo <= ns.P_SEE["bright"] * ns.P_SHARP_STEPPING <= hi  # the legs smear most frames


def test_the_live_hub_of_the_rig_has_the_watch_too() -> None:
    seed = 2
    onset = onset_between_two_looks(seed)
    world = World(samples=loom_samples(onset, onset + 3.0))
    hub = integrated.IntegratedSenseHub(seed=seed)
    frames = []
    for k in range(int((onset + 3.0) * 50)):
        t = k * 0.02
        frames.append(hub(Doors(t_ms=int(round(t * 1000)), world=world.section(t))))
    assert strong(frames, int((onset + 0.4) * 1000), int((onset + 1.0) * 1000))


def test_the_page_view_is_unchanged() -> None:
    """The watch adds no key to the page's view of the senses (the page is not touched)."""
    hub, _ = run(World(samples=[view(0.0, [])]), 0.2, hub=ns.SenseHub(seed=1))
    assert set(hub.view()) == {
        "version",
        "frames",
        "offset_ms",
        "light",
        "view",
        "background_db",
        "tracks",
        "looks",
        "looks_refused",
        "looks_per_s",
    }


# ── the report's source track (telemetry only) ──────────────────────────────────────────


def _strong_loom_sample(t: float = 0.0, key: str = "ball") -> dict[str, Any]:
    return view(
        t, [thing(id=key, kind="thing", dist=0.5, vr=3.0, size=0.3, rng=None, b=0.0, e=0.0)]
    )


def _first_report(watch: ns.LoomWatch, key: str = "ball", tracks: Any = None) -> C.Sighting:
    for k in range(200):  # (a report is given with the detector's own rate, not every frame)
        got = watch.step(
            k * 100, _strong_loom_sample(k * 0.1, key), stepping=False, saccade=False, tracks=tracks
        )
        if got is not None:
            return got
    raise AssertionError("the watch gave no report")


def test_a_watch_report_names_the_eyes_track_of_the_same_object_and_keeps_none_of_its_own() -> None:
    known = {"ball": ns._Track(tid=5, key="ball", kind="thing", first_ms=0, seen_ms=0)}
    report = _first_report(ns.LoomWatch(seed=3), tracks=known)
    assert report.kind == "looming" and report.track == 0 and report.source_track == 5
    assert report.motion == 0.0 and report.range_m is None  # the report is what it was


def test_no_source_track_when_the_eye_has_no_track_for_that_object() -> None:
    for tracks in (None, {}, {"another": ns._Track(7, "another", "thing", 0, 0)}):
        report = _first_report(ns.LoomWatch(seed=3), tracks=tracks)
        assert report.track == 0 and report.source_track == 0


def test_the_source_track_changes_nothing_else_in_the_report() -> None:
    known = {"ball": ns._Track(tid=5, key="ball", kind="thing", first_ms=0, seen_ms=0)}
    with_tracks = _first_report(ns.LoomWatch(seed=3), tracks=known)
    without = _first_report(ns.LoomWatch(seed=3))
    from dataclasses import replace

    assert replace(with_tracks, source_track=0) == without


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_through_the_hub_the_watch_report_carries_the_looks_track_of_the_object(seed: int) -> None:
    """A thing the looks track (a still thing in view, then it comes at the face): the watch's reports say its track, the
    sightings of the looks keep theirs, and the report's own ``track`` stays 0."""

    def scene(t: float) -> list[dict[str, Any]]:
        vr = 3.0 if t >= 4.0 else 0.0  # a still thing, then it expands at 3 m/s from 0.8 m
        return [thing(id="ball", kind="thing", dist=0.8, vr=vr, size=0.3, rng=0.78)]

    hub, frames = run(
        World(samples=samples_every(0.0, 6.0, scene)), 6.0, hub=ns.SenseHub(seed=seed)
    )
    tid = hub.vision.tracks["ball"].tid
    reports = [
        s for f in frames for s in f.camera.sightings if s.kind == "looming" and s.track == 0
    ]
    assert reports and all(s.source_track in (0, tid) for s in reports)
    assert any(s.source_track == tid for s in reports)
    # nothing in the picture but the loom watch's own: the looks' sightings of the ball have its track
    tracked = [s for f in frames for s in f.camera.sightings if s.track]
    assert (
        tracked
        and {s.track for s in tracked} == {tid}
        and all(s.source_track == 0 for s in tracked)
    )
