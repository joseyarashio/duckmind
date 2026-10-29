"""neuro:senses' sensor models (duckmind.world.neuro_senses), pure: no MuJoCo.

What the brain may get from the world's truth, and what it may not: a thing outside the camera's
frame, hidden behind something or in the dark gives no sighting; a sound gives a level, an onset
and (after the classifier's delay) a class, never a direction; a touch is felt only where a
contact really is, late and sometimes not at all. Latencies, noise and misses are measured here
with their counts (rates with Wilson 95 % intervals)."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from typing import Any

import pytest

from duckmind.duckbody.robotd_client import RobotState
from duckmind.neuro import contract as C
from duckmind.neuro.perceive import Perceiver
from duckmind.world import neuro_senses as ns


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def robot_state(t_s: float = 0.0, *, policy: str = "sit", vx: float = 0.0, gyro=(0.0, 0.0, 0.0)):
    joints = [0.0] * 15
    joints[5], joints[6] = 0.3491, 0.3491
    return RobotState.from_wire(
        {
            "t": t_s,
            "t_ns": int(t_s * 1e9),
            "policy": policy,
            "move": {"requested": [vx, 0.0, 0.0], "applied": [vx, 0.0, 0.0], "limited_by": []},
            "safety": {"fallen": False, "limp": False, "gravity": [0.0, 0.0, -1.0]},
            "joints": joints,
            "velocities": [0.0] * 15,
            "currents_ma": [3.5] * 15,
            "odom": {"position": [0.1, 0.2, 0.0], "yaw": 0.3},
            "imu": {"gyro": list(gyro), "quat": [1.0, 0.0, 0.0, 0.0]},
            "head": [0.0, 0.0, 0.0, 0.0],
        }
    )


@dataclass
class Doors:
    t_ms: int
    state: Any = None
    link: Any = None
    events: tuple[str, ...] = ()
    tof: Any = None
    depth: Any = None
    pg: bool | None = None
    day_minute: float = 600.0
    world: dict[str, Any] | None = None
    truth: Any = None


@dataclass
class World:
    """A hand-made ``neuro`` section: view samples every 100 ms, sounds, touches."""

    samples: list[dict[str, Any]] = field(default_factory=list)
    sounds: list[dict[str, Any]] = field(default_factory=list)
    touches: list[dict[str, Any]] = field(default_factory=list)
    battery: dict[str, Any] | None = None

    def section(self, t: float) -> dict[str, Any]:
        return {
            "neuro": {
                "v": 1,
                "sim_time": t,
                "samples": [s for s in self.samples if s["t"] <= t + 1e-9][-8:],
                "sounds": [e for e in self.sounds if e["t0"] <= t + 1e-9],
                "touches": [c for c in self.touches if c["t0"] <= t + 1e-9],
            },
            **({"battery": self.battery} if self.battery else {}),
        }


def view(t: float, seen: list[dict[str, Any]], lux: float = 150.0) -> dict[str, Any]:
    return {
        "t": round(t, 4),
        "cam": {"lux": lux, "fwd": [1.0, 0.0, 0.0], "backlight": False},
        "duck": {"yaw": 0.0},
        "seen": seen,
    }


def thing(**kw: Any) -> dict[str, Any]:
    base = {
        "id": "thing:ball_felt",
        "kind": "thing",
        "cls": "ball_slow",
        "n": 1,
        "fov": 1,
        "vis": 1,
        "b": 0.1,
        "e": -0.2,
        "size": 0.09,
        "dist": 0.8,
        "vr": 0.0,
        "om": 0.0,
        "lux": 150.0,
        "rng": 0.78,
    }
    base.update(kw)
    return base


def run(
    world: World,
    seconds: float,
    *,
    hub: ns.SenseHub | None = None,
    start: float = 0.0,
    state: Any = None,
    tick_ms: int = 20,
) -> tuple[ns.SenseHub, list[C.SensorFrame]]:
    hub = hub or ns.SenseHub(seed=7)
    frames = []
    for k in range(int(round(seconds * 1000 / tick_ms))):
        t = start + k * tick_ms / 1000.0
        doors = Doors(t_ms=int(round(t * 1000)), state=state, world=world.section(t))
        frames.append(hub(doors))
    return hub, frames


def samples_every(t0: float, t1: float, make: Any) -> list[dict[str, Any]]:
    out = []
    t = t0
    while t <= t1 + 1e-9:
        out.append(view(t, make(t)))
        t = round(t + 0.1, 4)
    return out


def kinds(frames: list[C.SensorFrame]) -> list[str]:
    return [s.kind for f in frames for s in f.camera.sightings]


# ── the light meter ─────────────────────────────────────────────────────────────


def test_light_bands_are_owner_c3():
    assert [ns.light_band(x) for x in (300, 30, 29.9, 2.5, 2.4, 0.5, 0.49)] == [
        "bright",
        "bright",
        "dim",
        "dim",
        "dark",
        "dark",
        "black",
    ]
    assert ns.brightness_of(0.03) == pytest.approx(0.0, abs=0.01)
    assert ns.brightness_of(300) > 0.95 and ns.brightness_of(5) < ns.brightness_of(120)


def test_the_meter_switches_after_two_readings_and_the_dock_light_makes_black_dark():
    meter = ns.LightMeter(seed=1)
    meter.read(0, 150.0)
    assert meter.level == "bright"
    meter.read(500, 0.05)
    assert meter.level == "bright"  # one reading is not enough (hysteresis)
    meter.read(1000, 0.05)
    assert meter.level == "black"
    meter.read(1500, 0.05, glow_in_view=True)
    meter.read(2000, 0.05, glow_in_view=True)
    assert meter.level == "dark"  # owner C1
    meter.read(2100, 150.0)  # not due yet (twice a second)
    assert meter.level == "dark"


# ── the camera: only what is inside the frame, not hidden, and lit ─────────────


def test_nothing_behind_the_duck_or_outside_the_frame_is_ever_sighted():
    # the world lists only candidates with a point inside the frame: behind = not listed
    world = World(samples=samples_every(0.0, 3.0, lambda t: []))
    _, frames = run(world, 3.0)
    assert kinds(frames) in ([], ["motion"])  # at most the detector's own rare noise
    assert all(s.track == 0 for f in frames for s in f.camera.sightings)


def test_a_hidden_thing_gives_no_sighting_and_a_visible_one_does():
    hidden = World(samples=samples_every(0.0, 4.0, lambda t: [thing(vis=0, fov=1)]))
    _, frames = run(hidden, 4.0)
    assert "thing" not in kinds(frames)
    seen = World(samples=samples_every(0.0, 4.0, lambda t: [thing()]))
    _, frames = run(seen, 4.0)
    sightings = [s for f in frames for s in f.camera.sightings if s.kind == "thing"]
    assert sightings, "a still thing on the floor in good light is found"
    assert all(abs(s.bearing - 0.1) < 0.12 for s in sightings)
    assert all(s.range_m is not None and 0.5 < s.range_m < 1.1 for s in sightings)
    assert all(s.source == "model" for s in sightings)


def test_in_the_dark_nothing_is_seen_but_the_glowing_dock():
    dark = World(
        samples=samples_every(
            0.0,
            4.0,
            lambda t: [
                thing(lux=0.2, om=0.5),
                {
                    **thing(id="nest", kind="nest", cls="", lux=0.2, glow=True, rng=None),
                    "face": 0.2,
                },
            ],
        )
    )
    for sample in dark.samples:
        sample["cam"]["lux"] = 0.2
    _, frames = run(dark, 4.0)
    assert set(kinds(frames)) <= {"nest", "motion"}
    assert "nest" in kinds(frames)
    moving_thing = [
        s for f in frames for s in f.camera.sightings if s.track != 0 and s.kind != "nest"
    ]
    assert moving_thing == []
    assert frames[-1].camera.light == "dark"  # C1: black with the dock light in view


def test_a_moving_thing_is_motion_and_one_coming_at_the_face_is_looming():
    moving = World(samples=samples_every(0.0, 2.0, lambda t: [thing(om=0.5, rng=None)]))
    _, frames = run(moving, 2.0)
    assert "motion" in kinds(frames)
    motion = [s for f in frames for s in f.camera.sightings if s.kind == "motion"]
    assert max(s.motion for s in motion) == pytest.approx(0.5, abs=0.01)
    loom = World(
        samples=samples_every(
            0.0,
            1.0,
            lambda t: [thing(id="hand", kind="hand", size=0.3, dist=0.3, vr=1.2, rng=None)],
        )
    )
    _, frames = run(loom, 1.0)
    looming = [s for f in frames for s in f.camera.sightings if s.kind == "looming"]
    assert looming and all(s.looming > 2.0 for s in looming)  # 1.2 m/s at 0.3 m ≈ 4 /s


def test_a_saccade_blinds_the_motion_pass_and_stepping_raises_its_threshold():
    slow = World(samples=samples_every(0.0, 3.0, lambda t: [thing(om=0.1, rng=None)]))
    _, frames = run(slow, 3.0, state=robot_state(policy="walk", vx=0.4))
    assert "motion" not in kinds(frames)  # 0.1 rad/s is below the stepping threshold
    _, frames = run(slow, 3.0, state=robot_state())
    assert "motion" in kinds(frames)
    _, frames = run(slow, 3.0, state=robot_state(gyro=(0.0, 0.0, 2.0)))
    assert [s for f in frames for s in f.camera.sightings if s.track != 0] == []


def test_a_ranged_static_thing_within_two_metres_can_be_acquired_while_walking():
    """A sharp walking frame may acquire a nearby floor-associated static thing."""
    world = World(samples=samples_every(0.0, 12.0, lambda _t: [thing(dist=1.8, rng=1.76, om=0.0)]))
    _, frames = run(world, 12.0, state=robot_state(policy="walk", vx=0.4), hub=ns.SenseHub(seed=7))
    things = [s for f in frames for s in f.camera.sightings if s.kind == "thing"]
    assert things
    assert all(s.range_m is not None and s.range_m <= 2.0 for s in things)


def test_a_static_thing_beyond_two_metres_is_not_acquired_while_walking():
    """The walking exception is deliberately limited to nearby range-confirmed things."""
    world = World(samples=samples_every(0.0, 12.0, lambda _t: [thing(dist=2.2, rng=2.18, om=0.0)]))
    _, frames = run(world, 12.0, state=robot_state(policy="walk", vx=0.4), hub=ns.SenseHub(seed=7))
    assert not [s for f in frames for s in f.camera.sightings if s.kind == "thing"]


def test_the_camera_reports_late_by_the_detector_latency():
    t_appear = 1.0
    world = World(
        samples=samples_every(
            0.0, 3.0, lambda t: [thing(om=0.6, rng=None)] if t >= t_appear - 1e-9 else []
        )
    )
    _, frames = run(world, 3.0, tick_ms=10)
    first = next(f.t_ms for f in frames if any(s.kind == "motion" for s in f.camera.sightings))
    lag = first - int(t_appear * 1000)
    # the next look (the look budget: one every ~0.5 s) plus the detector's own latency
    most = ns.LOOK_GAP_S * 1.15 * 1000 + ns.DETECT_LATENCY_MS[1] + 10
    assert ns.DETECT_LATENCY_MS[0] <= lag <= most


def test_person_and_hand_labels_are_modelled_and_rates_follow_the_light():
    def person(t: float) -> list[dict[str, Any]]:
        return [thing(id="person:owner", kind="person", om=0.4, size=0.5, dist=1.5, rng=1.4)]

    hits = 0
    n = 60
    for seed in range(n):
        world = World(samples=samples_every(0.0, 1.0, person))
        _, frames = run(world, 1.0, hub=ns.SenseHub(seed=seed))
        hits += "person" in kinds(frames)
    lo, hi = wilson(hits, n)
    assert lo <= ns.P_PERSON["bright"] <= hi or hits / n > 0.6


def person(**kw: Any) -> dict[str, Any]:
    """A family member in the view (kind person), by default standing still 0.84 m ahead."""
    base = {
        "id": "person:owner",
        "kind": "person",
        "cls": "",
        "n": 312,
        "fov": 80,
        "vis": 78,
        "b": 0.0,
        "e": -0.07,
        "size": 0.86,
        "dist": 0.84,
        "om": 0.0,
        "rng": 0.7,
    }
    base.update(kw)
    return thing(**base)


def first_person_ms(seed: int, seconds: float, make: Any, **kw: Any) -> int | None:
    _, frames = run(
        World(samples=samples_every(0.0, seconds, make)), seconds, hub=ns.SenseHub(seed=seed), **kw
    )
    return next((f.t_ms for f in frames if "person" in [s.kind for s in f.camera.sightings]), None)


def called_share(make: Any, seconds: float, seeds: int = 200, **kw: Any) -> float:
    return sum(first_person_ms(s, seconds, make, **kw) is not None for s in range(seeds)) / seeds


@pytest.mark.parametrize("rng", [0.7, None], ids=["floor_ranged", "feet_below_the_frame"])
def test_a_still_person_in_view_is_called_a_person_within_a_few_looks(rng):
    """Measured on the smoke server: a person standing 0.84 m from the duck was never reported
    (a still person was not labelled at all, and the person call was rolled once per track). A
    still person in good light is now called within ~6 looks (3 s) for 95 % of seeds, with or
    without a floor range (a person this near has their feet below the frame)."""
    assert called_share(lambda _t: [person(rng=rng)], 3.0) >= 0.95


def test_a_still_person_in_dim_light_is_called_later_but_called():
    assert called_share(lambda _t: [person(lux=10.0, rng=0.7)], 8.0) >= 0.95


def test_a_person_outside_the_frame_or_hidden_or_in_the_dark_is_never_called():
    """The world lists only what has a point inside the frame; a hidden person is listed with no
    visible point; in the dark nothing on the floor is found."""
    assert called_share(lambda _t: [], 4.0, seeds=40) == 0.0
    assert called_share(lambda _t: [person(vis=0)], 4.0, seeds=40) == 0.0
    dark = called_share(lambda _t: [person(lux=0.2)], 4.0, seeds=40)
    assert dark == 0.0


def test_a_person_called_stays_a_person_on_the_same_track_while_in_view():
    for seed in range(20):
        world = World(samples=samples_every(0.0, 6.0, lambda _t: [person()]))
        _, frames = run(world, 6.0, hub=ns.SenseHub(seed=seed))
        got = [s for f in frames for s in f.camera.sightings if s.track != 0]
        assert got and got[0].kind == "person"
        assert {s.kind for s in got} == {"person"}  # never back to a motion or a thing
        assert len({s.track for s in got}) == 1


def test_a_track_exists_only_for_something_reported():
    """An uncalled still person is not reported and leaves no track; every track is a sighting."""
    sample = view(0.0, [person()])
    silent = reported = 0
    for seed in range(300):
        vision = ns.VisionModel(seed=seed)
        out = vision.step(1000, sample, stepping=False, saccade=False)
        tids = {s.track for s in out if s.track != 0}
        assert {tr.tid for tr in vision.tracks.values()} == tids
        silent += not out
        reported += bool(out)
    assert silent and reported  # (both outcomes happen: the rolls are real)


def test_a_moving_person_is_called_as_often_on_the_first_look_and_surely_later():
    """Unchanged for a moving person: the first look calls them a person with P_PERSON (the track
    is no longer stuck as motion when the first roll fails)."""
    sample = view(0.0, [person(om=0.4, size=0.5, dist=1.5, rng=1.4)])
    firsts = []
    for seed in range(400):
        out = ns.VisionModel(seed=seed).step(1000, sample, stepping=False, saccade=False)
        out = [s for s in out if s.track != 0]
        if out:
            firsts.append(out[0].kind == "person")
    lo, hi = wilson(sum(firsts), len(firsts))
    assert lo <= ns.P_PERSON["bright"] <= hi
    assert called_share(lambda _t: [person(om=0.4, size=0.5, dist=1.5, rng=1.4)], 3.0) >= 0.99


def test_a_still_person_near_the_feet_is_found_while_walking_but_not_far_off():
    walking = robot_state(policy="walk", vx=0.4)
    near = [
        first_person_ms(s, 12.0, lambda _t: [person(dist=1.8, rng=1.7)], state=walking) is not None
        for s in range(20)
    ]
    assert sum(near) >= 19
    far = [
        first_person_ms(s, 12.0, lambda _t: [person(dist=2.5, rng=2.4)], state=walking) is not None
        for s in range(20)
    ]
    assert not any(far)


# ── the microphone: level, onset, class; never a direction ─────────────────────


def sound(sid: int, kind: str, t0: float, seconds: float, db: float, **kw: Any) -> dict[str, Any]:
    env = kw.pop("env", "impulse" if kind in ("clap", "bang", "door", "knock") else "speech")
    return {"id": sid, "kind": kind, "t0": t0, "t1": t0 + seconds, "db": db, "env": env, **kw}


def test_a_clap_gives_an_onset_within_the_onset_delay_and_no_direction():
    world = World(sounds=[sound(1, "clap", 1.0, 0.04, 75.0)])
    _, frames = run(world, 2.0, tick_ms=10)
    onsets = [f for f in frames if f.sound.onset > 0]
    assert len(onsets) == 1
    lag = onsets[0].t_ms - 1000
    assert ns.ONSET_LATENCY_MS[0] <= lag <= ns.ONSET_LATENCY_MS[1] + 10
    assert onsets[0].sound.onset > 0.9
    assert all(f.sound.bearing is None and f.sound.bearing_source == "none" for f in frames)
    assert max(f.sound.level_db for f in frames) > 60.0
    heard = [f.sound.kind for f in frames if f.sound.kind]
    assert heard and heard[0] in ("clap", "bang", "noise")
    assert all(f.sources["sound"] == "model" for f in frames)


def test_quiet_sounds_are_heard_as_a_level_but_not_classified():
    world = World(sounds=[sound(1, "name", 1.0, 0.7, 34.0, name=True)])
    _, frames = run(world, 3.0)
    assert all(f.sound.kind is None for f in frames)


def test_the_name_is_known_only_after_it_was_said_and_louder_names_are_heard_more():
    world = World(sounds=[sound(1, "name", 1.0, 0.7, 68.0, name=True)])
    _, frames = run(world, 3.0, tick_ms=10)
    first = next((f.t_ms for f in frames if f.sound.kind), None)
    assert first is not None and first >= 1700 + ns.CLASS_LATENCY_MS["speech"][0]

    def rate(db: float, n: int = 200) -> tuple[int, int]:
        hub = ns.SenseHub(seed=3)
        k = 0
        for i in range(n):
            ev = ns._Heard(i, "name", 0.0, 0.7, db, "speech", True)
            hub.hearing._decide(ev, noise_floor=32.0)
            k += ev.heard_as == "name"
        return k, n

    loud, quiet = rate(70.0), rate(41.0)
    lo_loud, _ = wilson(*loud)
    _, hi_quiet = wilson(*quiet)
    assert lo_loud > 0.8 and hi_quiet < 0.8 and loud[0] > quiet[0]


def test_the_level_noise_is_half_a_decibel_in_silence():
    _, frames = run(World(), 10.0)
    levels = [f.sound.level_db for f in frames[50:]]
    sd = statistics.pstdev(levels)
    assert 0.3 < sd < 0.8
    assert 29.0 < statistics.mean(levels) < 35.0


# ── touch: where the contact really is, late, sometimes missed ─────────────────


def test_perceiver_calls_the_shared_sliding_stroke_band_gentle():
    for stroke, gentle in ((0.2, False), (0.21, True), (0.39, True), (0.4, True)):
        frame = C.SensorFrame(
            t_ms=0,
            touch=C.TouchSense(site="head", pressure=0.6, stroke=stroke),
        )
        percept = Perceiver().perceive(frame, 0.0, asleep=False)
        assert bool(percept.touch_gentle) is gentle
        assert bool(percept.touch_sharp) is not gentle


def contact(t0: float, t1: float, site: str = "head", depth: float = 4.0, **kw: Any) -> dict:
    return {
        "key": f"hand@{site}",
        "by": "hand",
        "site": site,
        "t0": t0,
        "t1": t1,
        "active": kw.pop("active", True),
        "depth_mm": depth,
        "closing": kw.pop("closing", 0.05),
        "slide": kw.pop("slide", 0.05),
        "now_depth_mm": depth,
        "now_slide": kw.pop("now_slide", 0.05),
    }


def test_a_touch_is_felt_late_at_its_site_and_not_before_it_happens():
    world = World(touches=[contact(1.0, 2.0, "head")])
    felt = None
    for seed in range(20):
        _, frames = run(world, 2.5, hub=ns.SenseHub(seed=seed), tick_ms=10)
        assert all(f.touch.site is None for f in frames if f.t_ms < 1000)
        on = [f for f in frames if f.touch.site]
        if on:
            felt = on[0]
            lag = felt.t_ms - 1000
            assert ns.TOUCH_LATENCY_MS[0] <= lag <= ns.TOUCH_LATENCY_MS[1] + 10
            assert felt.touch.site in ("head", "beak")
            assert felt.sources["touch"] == "model"
    assert felt is not None


def test_light_touches_are_missed_about_a_quarter_of_the_time():
    n = 400
    felt = 0
    for i in range(n):
        model = ns.TouchModel(seed=i)
        model.update([contact(0.0, 0.5, "back", depth=0.5, closing=0.0, slide=0.0)])
        # (a light touch on the back is heard by pet-detect: up to a second late, unplaced)
        felt += any(model.step(t / 100.0).site == "body" for t in range(0, 200, 5))
    lo, hi = wilson(felt, n)
    assert lo <= ns.P_FEEL_LIGHT <= hi


# ── the whole frame ────────────────────────────────────────────────────────────


def test_the_frame_carries_no_world_coordinates_or_ids():
    world = World(
        samples=samples_every(0.0, 2.0, lambda t: [thing(om=0.5, rng=None)]),
        sounds=[sound(1, "clap", 0.5, 0.04, 75.0)],
        touches=[contact(1.0, 1.5)],
        battery={"percent": 64, "soc_ppm": 640000, "charging": False},
    )
    _, frames = run(world, 2.0, state=robot_state())
    text = json.dumps([f.to_json() for f in frames])
    for leak in ("ball_felt", "thing:", "person:", "pos_mm", "x_mm", "sim_time", "offscreen"):
        assert leak not in text
    last = frames[-1]
    assert last.sources["camera"] == "model" and last.sources["body"] == "real"
    assert last.power.battery == pytest.approx(0.64, abs=0.02)
    assert C.SensorFrame.from_json(last.to_json()) == last


def test_same_seed_same_frames_other_seed_other_noise():
    world = World(
        samples=samples_every(0.0, 3.0, lambda t: [thing(om=0.5, rng=None)]),
        sounds=[sound(1, "voice", 0.5, 2.0, 60.0)],
    )
    _, a = run(world, 3.0, hub=ns.SenseHub(seed=11), state=robot_state())
    _, b = run(world, 3.0, hub=ns.SenseHub(seed=11), state=robot_state())
    _, c = run(world, 3.0, hub=ns.SenseHub(seed=12), state=robot_state())
    assert [f.to_json() for f in a] == [f.to_json() for f in b]
    assert [f.to_json() for f in a] != [f.to_json() for f in c]


def test_without_the_world_every_modelled_channel_says_none():
    hub = ns.SenseHub(seed=1)
    frame = hub(Doors(t_ms=0, state=robot_state(), world=None))
    assert frame.sources["camera"] == "none" and frame.sources["sound"] == "none"
    assert frame.sound.source == "none" and frame.touch.site is None
    assert frame.body.posture == "sit" and frame.body.odom[0] == pytest.approx(0.1)


def test_the_injector_goes_to_the_world_op_only():
    calls: list[tuple[str, dict[str, Any]]] = []

    def world_call(op: str, **params: Any) -> dict[str, Any]:
        calls.append((op, params))
        return {"ok": True, "note_zh": "好"}

    inject = ns.make_injector(world_call)
    assert inject("clap", {"kind": "clap"})["ok"]
    assert calls == [("x_neuro_stim", {"stim": "clap"})]
    with pytest.raises(ValueError):
        inject("teleport", {})


def test_every_page_and_contract_stimulus_is_one_the_world_makes():
    from duckmind.world import neuro_world

    assert set(ns.SUPPORTED_STIMULI) == set(neuro_world.STIMULI)
    # (swap_battery is the world's own battery op, x_battery, made by the driver — integration;
    # carry_to_mat is the world's hand, x_hand place → mat, made by the driver too — neuro:fix)
    contract_keys = set(C.STIMULI) - {"swap_battery", "carry_to_mat"}
    missing = contract_keys - set(ns.SUPPORTED_STIMULI)
    assert missing == set(), missing


def test_the_look_budget_paces_the_camera():
    """Every salience pass and floor line is a camera look through duck-eye's own governor
    (review 2026-09-28: the model had looked 10 times a second): about 2 a second, faster for a
    moment after something happened, never beyond the governor's burst limit (4/s)."""
    world = World(samples=samples_every(0.0, 10.0, lambda t: [thing(om=0.5, rng=None)]))
    hub, frames = run(world, 10.0, hub=ns.SenseHub(seed=7))
    updates = sorted({f.t_ms - f.camera.age_ms for f in frames if f.camera.usable})
    assert 15 <= len(updates) <= 26  # ~2 looks a second over 10 s
    gaps = [b - a for a, b in zip(updates, updates[1:], strict=False)]
    assert min(gaps) >= 240  # never faster than the governor's 4/s
    assert hub.looks_per_s() <= 2.6
    _, frames = run(world, 3.0, hub=ns.SenseHub(seed=7, vision_hz=1.0))  # a slower knob
    assert len({f.t_ms - f.camera.age_ms for f in frames if f.camera.usable}) <= 4


def test_while_stepping_most_frames_are_too_smeared_to_find_anything():
    moving = World(samples=samples_every(0.0, 20.0, lambda t: [thing(om=0.5, rng=None)]))
    hub, frames = run(moving, 20.0, state=robot_state(policy="walk", vx=0.4))
    passes = hub.vision.passes
    tracks = {s.track for f in frames for s in f.camera.sightings if s.track}
    seen = sum(1 for f in frames if f.camera.sightings) / len(frames)
    assert 30 <= passes <= 55 and tracks  # found, but later and less often than when still
    _, still_frames = run(moving, 20.0)
    assert seen < 0.7 * sum(1 for f in still_frames if f.camera.sightings) / len(still_frames)


def test_a_thing_resting_against_the_duck_is_soon_not_felt_but_a_stroke_is():
    resting = contact(1.0, 6.0, "left", depth=0.0, closing=0.0, slide=0.0, now_slide=0.0)
    resting["now_closing"] = 0.0
    felt_resting = felt_stroke = 0
    n = 40
    for seed in range(n):
        _, frames = run(World(touches=[resting]), 6.0, hub=ns.SenseHub(seed=seed))
        late = [f for f in frames if f.t_ms >= 2500]
        felt_resting += any(f.touch.site for f in late)
        stroke = contact(1.0, 6.0, "back", depth=2.0, slide=0.1, now_slide=0.1)
        stroke["now_closing"] = 0.0
        _, frames = run(World(touches=[stroke]), 6.0, hub=ns.SenseHub(seed=seed))
        late = [f for f in frames if f.t_ms >= 2500]
        felt_stroke += (
            bool(late) and all(f.touch.site for f in late) and late[-1].touch.stroke > 0.9
        )
    assert felt_resting == 0  # adapted away (still in contact until 6 s)
    lo, _ = wilson(felt_stroke, n)
    assert lo > 0.7  # felt all along unless missed at its onset (P_FEEL_FIRM)


def test_a_clap_that_reaches_the_senses_late_is_heard_late_not_lost():
    """The live driver copies the world five times a second: a 40 ms clap may arrive after it
    ended. It is heard from its arrival (late), never lost."""
    world = World(sounds=[sound(1, "clap", 1.013, 0.04, 75.0)])
    hub = ns.SenseHub(seed=7)
    frames = []
    copy = None
    for k in range(300):
        t = k * 0.01
        if k % 20 == 0:  # the driver's 5 Hz world copy
            copy = world.section(t)
        frames.append(hub(Doors(t_ms=k * 10, world=copy)))
    onsets = [f for f in frames if f.sound.onset > 0]
    assert len(onsets) == 1 and onsets[0].sound.onset > 0.9
    lag = onsets[0].t_ms - 1013
    assert 0 < lag <= 200 + ns.ONSET_LATENCY_MS[1] + 10


def test_the_floor_line_is_seen_only_in_good_light_and_not_every_frame() -> None:
    """MODEL of duck-eye FL-v1: none in the dark, ×0.8 in dim light, at most FLOOR_HZ looks a
    second, blind in a saccade, while stepping only in the sharp frames."""
    truth = {
        "sectors": [{"b": 0.25 - 0.12 * k, "w": 0.12, "free": 0.6, "end": "wall"} for k in range(5)]
    }
    model = ns.FloorLineModel(seed=3)
    assert model.look(0.0, truth, "dark", stepping=False, saccade=False) is None
    assert model.look(0.0, truth, "bright", stepping=False, saccade=True) is None
    line = model.look(0.0, truth, "bright", stepping=False, saccade=False)
    assert line is not None and len(line) == 5 and line[0].bearing > line[-1].bearing
    assert all(s.end in ("wall", "unknown") for s in line)
    assert sum(abs(s.free_m - 0.6) < 0.15 for s in line) >= 4
    assert model.look(0.1, truth, "bright", stepping=False, saccade=False) is None  # rate
    dim = ns.FloorLineModel(seed=3).look(0.0, truth, "dim", stepping=False, saccade=False)
    assert dim is not None and sum(s.free_m for s in dim) / 5 < 0.56  # ×0.8
    walking = ns.FloorLineModel(seed=4)
    got = [walking.look(k * 0.4, truth, "bright", stepping=True, saccade=False) for k in range(200)]
    share = sum(g is not None for g in got) / len(got)
    assert 0.2 < share < 0.5  # only the sharp frames while the legs step (pet:probe)
