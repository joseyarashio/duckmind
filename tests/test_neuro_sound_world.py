"""a:sound (2026-09-28) — the world side, on the official MJCF headless: the song with a tempo is
real sound events the EXISTING microphone model turns into onsets at the beats (the brain gets no
beat times); the hovering hand is seen by the ToF model where it is (robotd's hand tracker ranges
it); the ride really carries the duck through the air and its own IMU feels the swing."""

from __future__ import annotations

import math
import os
import statistics
from typing import Any

import pytest

from duckmind import paths

pytest.importorskip("mujoco")
pytest.importorskip("numpy")

MJCF = str(paths.microduck_mjcf_dir())
if not os.path.isdir(MJCF):
    pytest.skip("the official MJCF folder is not here", allow_module_level=True)

from duckmind.duckbody.robotd_client import RobotState  # noqa: E402
from duckmind.neuro import skill_ride as SR  # noqa: E402
from duckmind.neuro.contract import SensorFrame  # noqa: E402
from duckmind.simweb.neuro_voice_stream import HandTracker  # noqa: E402
from duckmind.world import neuro_senses as ns  # noqa: E402
from duckmind.world import neuro_sound as nsd  # noqa: E402
from duckmind.world import neuro_world as nw  # noqa: E402
from duckmind.world import tof_model as T  # noqa: E402
from duckmind.world.headless_duck import HeadlessDuck  # noqa: E402
from duckmind.world.neuro_music import Song  # noqa: E402


def _duck(**kw: Any) -> tuple[HeadlessDuck, Any, Any]:
    d = HeadlessDuck(
        "default_home", mjcf_dir=MJCF, keyframe="STAND", x=0.0, y=-1.5, yaw=math.pi / 2, **kw
    )
    part = nw.attach(d, seed=7)
    sound = nsd.attach(d, part, seed=7)
    ok, _ = d.stand(timeout=10.0, hold=1.0)
    assert ok
    return d, part, sound


def test_the_song_is_heard_as_onsets_at_its_beats() -> None:
    d, part, sound = _duck()
    answer = sound.op(None, {"do": "music", "bpm": 104})
    assert answer["on"] and "104" in answer["note_zh"]
    hearing = ns.HearingModel(seed=5)
    onsets: list[float] = []
    kinds: dict[str | None, int] = {}
    for tick in range(20 * 50):
        d.tick()
        if tick % 10 == 0:  # the world's truth at 5 Hz, as the live driver reads it
            part.pump()
            with d.world.lock:
                section = part.snapshot(d.world)
            now = d.now()
            hearing.update(section["sounds"], now_s=now)
        snd = hearing.step(d.now(), 28.0)
        kinds[snd.kind] = kinds.get(snd.kind, 0) + 1
        if snd.onset > 0.0:
            onsets.append(d.now())
    music = sound.snapshot(d.world)["music"]
    t0, bpm = music["t0"], music["bpm"]
    beats = [t0 + t for t, _ in Song(bpm, 7).beats(0.0, d.now() - t0)]
    matched = [min((o - b for o in onsets if 0.0 <= o - b < 0.12), default=None) for b in beats]
    late = [m for m in matched if m is not None]
    assert len(late) >= 0.85 * len(beats)  # nearly every beat is heard …
    assert 0.02 <= statistics.median(late) <= 0.08  # … a little late (the microphone model)
    assert kinds.get("music", 0) > 0.5 * sum(kinds.values())  # and it is called music
    sound.op(None, {"do": "music_off"})
    for _ in range(3 * 50):
        d.tick()
    with d.world.lock:
        section = part.snapshot(d.world)
    later = [s for s in section["sounds"] if s["src"].startswith("music:") and s["t0"] > d.now()]
    assert later == [] and not sound.snapshot(d.world)["music"]["on"]


@pytest.mark.parametrize("head_pitch", [0.0, -0.25])
def test_the_hovering_hand_is_where_the_tof_ranges_it(head_pitch: float) -> None:
    d, part, sound = _duck()
    d.head([0.0, head_pitch, 0.0, 0.0])
    for _ in range(100):
        d.tick()
    tof = T.TofSensor(d.model, seed=7)
    tracker = HandTracker()
    reach = (120, 250, 380) if head_pitch == 0.0 else (120, 250, 400, 550)
    for mm in reach:
        tracker.reset()
        sound.op(None, {"do": "hover", "distance_mm": mm})
        for _ in range(75):
            d.tick()
        ranges = []
        for _ in range(6):
            for _ in range(3):
                d.tick()
            frame = tof.capture(d.world)
            hand = tracker.track(frame["distance_mm"], frame["status"], d.now())
            assert hand is not None
            ranges.append(hand.range_m * 1000.0)
        assert abs(statistics.median(ranges) - mm) < 30.0, (mm, ranges)
    sound.op(None, {"do": "hover_end"})
    for _ in range(120):
        d.tick()
    assert sound.snapshot(d.world)["hover"] is None


def test_the_ride_is_real_and_its_imu_feels_it() -> None:
    d, part, sound = _duck()
    answer = sound.op(None, {"do": "swing", "cycles": 3})
    assert answer["swing"]["cycles"] == 3
    ride = SR.RideSense()
    energies = []
    heights = []
    riding_at = None
    for tick in range(12 * 50):
        d.tick()
        state = RobotState.from_wire(d.state())
        body = ns.body_sense(state)
        events: tuple[str, ...] = ()
        if tick == 60:
            events = ("lifted",)  # (the body link's proprio says so in the whole duck)
        view, edge = ride.step(tick / 50.0, SensorFrame(t_ms=tick * 20, body=body, events=events))
        energies.append(view.energy)
        heights.append(d.truth()["trunk"][2])
        if edge == "start" and riding_at is None:
            riding_at = tick / 50.0
    assert max(heights) > 0.3  # in the air
    assert max(energies) > 1.5  # swung: its own gyro
    assert riding_at is not None and riding_at < 6.0
    snap = sound.snapshot(d.world)
    assert snap["swing"] is None and snap["swings"] == 1  # swung, now held
    assert part.hand.holding and not d.safety.fallen
