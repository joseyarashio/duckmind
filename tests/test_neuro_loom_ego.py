"""The loom watch does not take the duck's own walk for something coming at it (pure, no MuJoCo).

The reach fix makes the duck walk up to a thing until its beak can touch it (the thing's near edge
0.126 m from the trunk, about 0.05 m from the lens). Every still thing the duck closes on expands in
the picture at v·cos(a)/d, so the loom watch (a frame-rate expansion detector) saw its own approach
as a loom — and a non-bold duck could startle at the toy it chose to walk up to. The watch now
withholds an expansion that the duck's own forward speed alone can give a still thing of that
angular size and bearing, from what the duck knows (its applied speed, the blob's measured
bearing, elevation and size). A hand coming at a standing or slowly walking duck is still found.

Only the watch is discounted: the looks' own ``looming`` labels (the old path, a sighting with a
track) are unchanged here, so the toy tests give the reflex arc what the watch gives it."""

from __future__ import annotations

import math
from typing import Any

import pytest
from test_neuro_loom_watch import (
    HAND_LOOM,
    hand,
    loom_samples,
    onset_between_two_looks,
    strong,
)
from test_neuro_senses import Doors, World, robot_state, run, thing

from duckmind.neuro import contract as C
from duckmind.neuro import reflex as R
from duckmind.world import neuro_senses as ns

STARTLES = frozenset({"startle", "freeze"})
SHY = R.params_from({"boldness": 0.0, "sociability": 0.9, "wit": 0.5}, stage="juvenile", side=1)
LENS_TO_NEAR_EDGE_M = (
    0.126 - 0.081
)  # the arrival: near edge 0.126 m from the trunk, lens 0.081 ahead


def approach_frames(
    applied_vx: float, ratio: float, width: float, *, seed: int = 3, start: float = 0.9
) -> list[C.SensorFrame]:
    """The duck walks (applied forward speed ``applied_vx``) up to a still toy ``width`` m wide: the
    camera closes at ``ratio`` × that on it (the body is not as fast as the command, or the head
    swings), the walk stops at the arrival and the closing speed dies away over 0.25 s (the coast),
    then it stands in front of the toy for a second."""
    d_end = LENS_TO_NEAR_EDGE_M + 0.5 * width  # lens to the toy's centre at the arrival
    wide = max(width, 0.06)  # the spread of the visible points: a toy has some height
    rows: list[tuple[float, float, float, float]] = []  # (t, dist, closing, applied)
    d, t, stop_t = start, 0.0, None
    while t < 30.0:
        if stop_t is None:
            closing, applied = ratio * applied_vx, applied_vx
        else:
            closing, applied = ratio * applied_vx * max(0.0, 1.0 - (t - stop_t) / 0.25), 0.0
        rows.append((t, d, closing, applied))
        d = max(0.03, d - closing * 0.1)
        if stop_t is None and d <= d_end:
            stop_t = t
        t = round(t + 0.1, 4)
        if stop_t is not None and t - stop_t > 1.5:
            break
    samples: list[dict[str, Any]] = []
    applied_at: dict[int, float] = {}
    for t, d, closing, applied in rows:
        toy = thing(
            id="thing:toy",
            kind="thing",
            dist=round(d, 3),
            vr=round(closing, 3),
            size=round(2.0 * math.atan(wide / (2.0 * d)), 4),
            om=0.0,
            rng=None,
        )
        samples.append(
            {
                "t": round(t, 4),
                "cam": {"lux": 150.0, "fwd": [1.0, 0.0, 0.0], "backlight": False},
                "duck": {"yaw": 0.0},
                "seen": [toy],
            }
        )
        applied_at[round(t * 10)] = applied
    hub = ns.SenseHub(seed=seed)
    sim = World(samples=samples)
    frames = []
    for k in range(int(round((rows[-1][0] + 0.5) * 50))):
        now = k * 0.02
        vx = applied_at.get(int(now * 10 + 1e-9), 0.0)
        state = robot_state(policy="walk" if vx else "stand", vx=vx)
        frames.append(hub(Doors(t_ms=int(round(now * 1000)), state=state, world=sim.section(now))))
    return frames


def watch_only(frames: list[C.SensorFrame]) -> list[C.Sighting]:
    """What the loom watch gave the brain (its sightings are the untracked ``looming`` ones; the
    looks' own labels carry a track)."""
    return [s for f in frames for s in f.camera.sightings if s.kind == "looming" and s.track == 0]


def only_the_watch(frame: C.SensorFrame) -> C.SensorFrame:
    sights = tuple(s for s in frame.camera.sightings if s.track == 0)
    return C.SensorFrame(
        t_ms=frame.t_ms,
        body=frame.body,
        camera=C.CameraSense(usable=frame.camera.usable, sightings=sights),
        power=frame.power,
    )


@pytest.mark.parametrize("applied_vx", (0.1, 0.2, 0.3, 0.4))
@pytest.mark.parametrize("ratio", (0.6, 1.0, 1.3))
@pytest.mark.parametrize("width", (0.03, 0.07, 0.15))
def test_walking_up_to_a_toy_is_no_loom_for_the_watch(
    applied_vx: float, ratio: float, width: float
) -> None:
    """Down to the 0.126 m arrival and through the coast: the watch reports nothing, so a shy duck's
    reflex arc, given what the watch gives it, has nothing to startle at."""
    frames = approach_frames(applied_vx, ratio, width)
    assert watch_only(frames) == []
    state = R.initial(5)
    fired: set[str] = set()
    presented: list[str] = []
    for frame in frames:
        state, out = R.step(SHY, state, only_the_watch(frame), hab={}, sens={})
        fired |= {e.reflex for e in out.events}
        presented += [p.stimulus for p in out.presented if "looming" in p.stimulus]
    assert not (fired & STARTLES) and not presented


def test_the_same_approach_is_a_loom_to_a_watch_that_does_not_know_the_walk() -> None:
    """The control: the arrival frame of a 0.3 m/s approach, seen by a watch told the duck is
    standing, is a loom at the reflex's level (what the merged tree did before the discount)."""
    toy = thing(id="thing:toy", kind="thing", dist=0.08, vr=0.3, size=0.9, om=0.0, rng=None)
    blind = ns.LoomWatch(seed=3)
    heard = [
        blind.step(1000 + k * 100, {"seen": [toy]}, stepping=False, saccade=False, ego_v=0.0)
        for k in range(20)
    ]
    assert any(h is not None and h.looming >= 1.2 for h in heard)
    aware = ns.LoomWatch(seed=3)
    heard = [
        aware.step(1000 + k * 100, {"seen": [toy]}, stepping=False, saccade=False, ego_v=0.3)
        for k in range(20)
    ]
    assert all(h is None for h in heard) and aware.explained == 20


@pytest.mark.parametrize("ego_v", (0.0, 0.05, 0.1, 0.2, 0.4))
def test_a_hand_loom_is_reported_at_its_strong_frames_when_the_duck_stands_or_walks(
    ego_v: float,
) -> None:
    """The discount is the duck's own walk and nothing else: the table's hand frames at +0.5 and
    +0.6 s (8 and 13 /s) are reported at every walking speed up to the top of the walk, and the
    first strong frame (2.7 /s) too while it stands or walks slowly."""
    watch = ns.LoomWatch(seed=4)
    reported: dict[float, float] = {}
    for k, (dist, vr, size) in HAND_LOOM.items():
        sample = {"seen": [hand(dist, vr, size)]}
        got = watch.step(round(k * 1000), sample, stepping=False, saccade=False, ego_v=ego_v)
        if got is not None:
            reported[k] = got.looming
    assert 0.5 in reported and 0.6 in reported
    assert reported[0.5] == pytest.approx(8.4, rel=0.15)
    if ego_v <= 0.1:
        assert 0.4 in reported
    if ego_v == 0.0:
        assert watch.explained == 0


def test_a_standing_duck_finds_the_hand_loom_through_the_hub() -> None:
    for seed in (1, 2, 3, 5, 6):
        onset = onset_between_two_looks(seed)
        world = World(samples=loom_samples(onset, onset + 3.0))
        _, frames = run(
            world, onset + 3.0, state=robot_state(policy="stand"), hub=ns.SenseHub(seed=seed)
        )
        assert strong(frames, int((onset + 0.4) * 1000), int((onset + 1.0) * 1000)), seed


def test_the_speed_is_held_while_the_body_coasts_and_then_forgotten() -> None:
    hub = ns.SenseHub(seed=1)
    for k in range(60):  # 50 Hz: walking until 0.4 s, then stopped
        hub.own_speed.append((k * 20, 0.3 if k < 20 else 0.0))
    assert hub._own_speed(380) == pytest.approx(0.3)
    assert hub._own_speed(700) == pytest.approx(0.3)  # 0.3 s after the stop: still counted
    assert hub._own_speed(1000) == 0.0  # forgotten after the hold


@pytest.mark.parametrize(
    ("ego_v", "bearing", "elevation", "size", "looming", "explained"),
    [
        (0.0, 0.0, 0.0, 0.6, 2.0, False),  # standing: nothing is the duck's own walk
        (0.01, 0.0, 0.0, 0.6, 2.0, False),  # creeping: below the minimum
        (0.2, 0.0, 0.0, 0.6, 2.0, True),  # walking up to a thing right ahead
        (0.2, 0.0, 0.0, 0.6, 12.0, False),  # far more than the walk gives: something comes
        (0.2, math.pi / 2, 0.0, 0.6, 2.0, False),  # to the side: walking forward does not close
        (0.2, math.pi, 0.0, 0.6, 2.0, False),  # behind
        (0.2, 0.0, 0.0, 0.1, 2.0, False),  # a small patch cannot be a still thing that close
    ],
)
def test_walk_explains(
    ego_v: float, bearing: float, elevation: float, size: float, looming: float, explained: bool
) -> None:
    assert ns.LoomWatch.walk_explains(ego_v, bearing, elevation, size, looming) is explained


def test_a_hand_loom_and_a_toy_approach_in_one_frame_report_the_hand() -> None:
    """The toy is withheld, not the whole frame: the other blob still gets its chance."""
    watch = ns.LoomWatch(seed=1)
    toy = thing(id="thing:toy", kind="thing", dist=0.08, vr=0.2, size=0.9, om=0.0, rng=None)
    frame = {"seen": [toy, hand(0.3, 2.4, 0.4)]}  # 2.5 /s and 8 /s
    got = [
        h
        for k in range(30)
        if (h := watch.step(1000 + k * 100, frame, stepping=False, saccade=False, ego_v=0.25))
    ]
    assert got and all(7.0 < h.looming < 9.0 for h in got)
    assert watch.explained >= 30


# ── the looks' own looming label (the 2 Hz path) gets the same discount ───────────────────────


@pytest.mark.parametrize("applied_vx", (0.1, 0.2, 0.3, 0.4))
@pytest.mark.parametrize("ratio", (0.6, 1.0, 1.3))
@pytest.mark.parametrize("width", (0.03, 0.07, 0.15))
def test_walking_up_to_a_toy_is_no_loom_for_the_looks_either(
    applied_vx: float, ratio: float, width: float
) -> None:
    """With everything the camera gives (the watch and the looks), a walk up to a toy down to the
    arrival has no ``looming`` sighting at all, and a shy duck's reflex arc does not startle."""
    frames = approach_frames(applied_vx, ratio, width)
    assert [s for f in frames for s in f.camera.sightings if s.kind == "looming"] == []
    state = R.initial(5)
    fired: set[str] = set()
    presented: list[str] = []
    for frame in frames:
        state, out = R.step(SHY, state, frame, hab={}, sens={})
        fired |= {e.reflex for e in out.events}
        presented += [p.stimulus for p in out.presented if "looming" in p.stimulus]
    assert not (fired & STARTLES) and not presented


@pytest.mark.parametrize("dist", (0.07, 0.10, 0.14))
def test_a_look_at_a_toy_the_duck_is_walking_up_to_labels_no_loom_but_still_finds_it(
    dist: float,
) -> None:
    toy = thing(id="thing:toy", kind="thing", dist=dist, vr=0.2, size=0.8, om=0.0, rng=None)
    sample = {"seen": [toy], "cam": {"lux": 150.0}}
    still = ns.VisionModel(seed=2)
    walking = ns.VisionModel(seed=2)
    old = [
        s
        for k in range(20)
        for s in still.step(1000 + k * 100, sample, stepping=False, saccade=False, ego_v=0.0)
    ]
    new = [
        s
        for k in range(20)
        for s in walking.step(1000 + k * 100, sample, stepping=False, saccade=False, ego_v=0.25)
    ]
    assert any(s.kind == "looming" and s.looming >= 1.2 for s in old)  # what a look did before
    assert new and all(s.kind != "looming" and s.looming == 0.0 for s in new if s.track)
    assert any(s.track for s in new)  # …and it is still found and tracked


def test_a_look_still_labels_a_hand_loom_while_the_duck_stands_or_walks_slowly() -> None:
    """A look on the hand's strong frames (8 and 13 /s) labels it looming standing and at a slow
    walk; only the watch is switched off here, so this is the look path alone."""
    frames = {
        k: {"seen": [hand(d, vr, size)], "cam": {"lux": 150.0}}
        for k, (d, vr, size) in HAND_LOOM.items()
        if k >= 0.4
    }
    for ego_v in (0.0, 0.05, 0.1, 0.2):
        vision = ns.VisionModel(seed=6)
        labelled = {
            k: [
                s
                for s in vision.step(
                    round(k * 1000), sample, stepping=False, saccade=False, ego_v=ego_v
                )
                if s.kind == "looming"
            ]
            for k, sample in frames.items()
        }
        assert labelled[0.5] and labelled[0.6], ego_v
        assert labelled[0.5][0].looming == pytest.approx(8.4, rel=0.15)
    # …and through the hub, standing, with the watch off
    seed = 2
    onset = onset_between_two_looks(seed)
    world = World(samples=loom_samples(onset, onset + 3.0))
    hub = ns.SenseHub(seed=seed, loom_watch_on=False)
    _, hub_frames = run(world, onset + 3.0, state=robot_state(policy="stand"), hub=hub)
    kinds = [
        s.kind for f in hub_frames for s in f.camera.sightings if s.looming >= 0.6 and s.track > 0
    ]
    assert "looming" in kinds
