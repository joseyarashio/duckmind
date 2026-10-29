"""a:sound (2026-09-28) — robotd's streamed voice in the simulator: the hand tracker and the theremin
are robotd's (``kinematics::hand``, ``theremin.rs``: the upstream tests' cases), the ride is
``sound.rs``'s writer (start → loop while held → the end when the hold decays, a cut on release),
the samples are upstream's own (the helper's theremin stream is bit-identical to upstream's
``sounds theremin`` audition; the ride's segments are the bank's files), and the page's stream
frames round-trip."""

from __future__ import annotations

import math
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pytest

from duckmind import paths
from duckmind.simweb import neuro_sound_live as NSL
from duckmind.simweb import neuro_voice_stream as V

UPSTREAM = paths.microduck_dir()
SOUNDS_BIN = UPSTREAM / "target" / "debug" / "sounds"
needs_helper = pytest.mark.skipif(V.helper_path() is None, reason="duck-voice-stream not built")


def _frame(distance_m: float, code: int, zones: int) -> dict:
    """``theremin.rs`` tests' frame: ``zones`` zones at ``distance_m`` with ``code``, the rest 255."""
    distance = [0] * 64
    status = [255] * 64
    for zone in range(min(zones, 64)):
        distance[zone] = int(distance_m * 1000)
        status[zone] = code
    return {"distance_mm": distance, "status": status}


# ── kinematics::hand ───────────────────────────────────────────────────────────


def test_the_statuses_a_real_hand_arrives_with_are_believed() -> None:
    for code in V.HandConfig().statuses:
        tracker = V.HandTracker()
        f = _frame(0.40, code, 8)
        hand = tracker.track(f["distance_mm"], f["status"], 0.0)
        assert hand is not None and abs(hand.range_m - 0.40) < 0.01
    tracker = V.HandTracker()
    for code in (255, 1):
        f = _frame(0.40, code, 8)
        assert tracker.track(f["distance_mm"], f["status"], 0.0) is None


def test_closeness_is_one_near_and_zero_far() -> None:
    cfg = V.HandConfig()
    for distance, want in ((cfg.near_m, 1.0), (cfg.far_m, 0.0)):
        f = _frame(distance, 5, 8)
        hand = V.HandTracker().track(f["distance_mm"], f["status"], 0.0)
        assert hand is not None and hand.closeness == pytest.approx(want)
    f = _frame(cfg.far_m + 0.2, 5, 8)
    assert V.HandTracker().track(f["distance_mm"], f["status"], 0.0) is None


def test_a_dropout_is_held_and_a_withdrawn_hand_ends() -> None:
    tracker = V.HandTracker()
    good, empty = _frame(0.30, 5, 8), _frame(0.30, 255, 8)
    assert tracker.track(good["distance_mm"], good["status"], 0.0) is not None
    for k in (1, 2, 3):
        held = tracker.track(empty["distance_mm"], empty["status"], 0.066 * k)
        assert held is not None and held.held
    assert tracker.track(empty["distance_mm"], empty["status"], 0.251 + 0.001) is None


def test_one_stray_zone_is_not_a_hand_and_bad_frames_are_not_notes() -> None:
    f = _frame(0.25, 5, 1)
    assert V.HandTracker().track(f["distance_mm"], f["status"], 0.0) is None
    assert V.HandTracker().track([-3] * 64, [5] * 64, 0.0) is None
    assert V.HandTracker().track([], [], 0.0) is None
    assert V.HandTracker().track([300, 300, 300], [5, 5], 0.0) is not None


def test_the_describe_line_marks_the_believed_codes() -> None:
    line = V.describe(_frame(0.40, 4, 32)["status"], V.HandConfig())
    assert line.startswith("32 usable") and "4*:32" in line and "255:32" in line


# ── theremin.rs ────────────────────────────────────────────────────────────────


def test_a_theremin_that_is_down_says_nothing_and_plays_at_once_when_up() -> None:
    t = V.Theremin()
    assert t.tick(_frame(0.3, 5, 8), 0.0, 0.0) is None
    t.set_active(True)
    note = t.tick(_frame(0.30, 5, 8), 0.0, 0.0)
    assert note is not None and note.range_m == pytest.approx(0.30, abs=0.01)
    assert note.mouth > 0.0 and note.mouth == note.closeness


def test_the_mouth_opens_with_the_note() -> None:
    t = V.Theremin()
    t.set_active(True)
    far, mid, near = (t.tick(_frame(d, 5, 8), 0.0, 0.0).mouth for d in (0.65, 0.40, 0.12))
    assert far < mid < near and near > 0.8


def test_a_stale_frame_is_silence_and_the_instrument_stays_up() -> None:
    t = V.Theremin()
    t.set_active(True)
    note = t.tick(_frame(0.25, 5, 8), V.FRAME_STALE_S * 2, 0.0)
    assert note is not None and note.closeness is None and note.mouth == 0.0
    assert note.sensor == "no depth frames" and t.active
    assert t.tick(_frame(0.25, 5, 8), 0.0, 0.1).mouth > 0.0


# ── sound.rs: the ride's writer ─────────────────────────────────────────────────


def _voice(**kw) -> V.RobotdVoice:
    start, loop, end = b"\x01\x00" * 1500, b"\x02\x00" * 2000, b"\x03\x00" * 1000
    return V.RobotdVoice(
        7, helper=None, segments=lambda letter: (start, loop, end), letters=["a"], **kw
    )


def _values(pcm: bytes) -> list[int]:
    return list(struct.unpack(f"<{len(pcm) // 2}h", pcm))


def test_a_held_ride_streams_start_then_loops() -> None:
    voice = _voice()
    voice.wheee("held")
    assert voice.ride == "riding" and voice.owns_pcm()
    samples = []
    for _ in range(10):  # 200 ms per tick call below: 9600 samples
        samples += _values(voice.render(960))
    assert samples[:1500] == [1] * 1500
    assert samples[1500:3500] == [2] * 2000
    assert samples[3500:5500] == [2] * 2000  # the loop again: still held
    assert 3 not in samples


def test_a_decayed_hold_lands_after_the_loop_pass_and_plays_the_end() -> None:
    voice = _voice()
    voice.wheee("held")
    first = _values(voice.render(1500 + 500))  # the start and a quarter of the loop
    voice.wheee("decayed")
    assert voice.ride == "landing"
    rest: list[int] = []
    for _ in range(10):
        pcm = voice.render(960)
        if pcm is None:
            break
        rest += _values(pcm)
    assert first[-1] == 2
    assert rest[:1500] == [2] * 1500  # the loop pass it was in, to its end
    assert rest[1500:2500] == [3] * 1000  # then the end segment
    assert voice.ride == "off" and not voice.owns_pcm()


def test_an_explicit_release_cuts_the_ride() -> None:
    voice = _voice()
    voice.wheee("held")
    voice.render(960)
    voice.wheee("released")
    assert voice.ride == "off" and voice.render(960) is None


def test_a_press_during_a_landing_starts_a_new_ride() -> None:
    voice = _voice()
    voice.wheee("held")
    voice.render(960)
    voice.wheee("decayed")
    voice.wheee("held")
    assert voice.ride == "riding"
    assert _values(voice.render(960))[:10] == [1] * 10  # from the start segment again
    assert voice.rides == 2


# ── the engine, tick for tick (fake helper) ─────────────────────────────────────


class FakeHelper:
    """Stands in for duck-voice-stream: hz = 200 + 400·position; a block is the level."""

    def __init__(self) -> None:
        self.seed: int | None = None
        self.level = 0.0
        self.shown = 0.0
        self.voices = 0

    def voice(self, seed: int, kind: str = "wheee", variant: int = 0) -> None:
        self.seed = seed
        self.voices += 1
        self.shown = 0.0

    def hz_at(self, position: float) -> float:
        return 200.0 + 400.0 * position

    def set(self, hz: float, level: float, open_: float) -> None:
        self.level = level

    def silent(self) -> bool:
        return self.level <= 0.0 and self.shown < 1e-4

    def block(self, n: int) -> bytes:
        self.shown += (self.level - self.shown) * 0.5
        return struct.pack(f"<{n}h", *([int(1000 * self.shown)] * n))

    def segments(self, seed: int, variant: int) -> tuple[bytes, bytes, bytes]:
        return b"\x01\x00" * 4800, b"\x02\x00" * 9600, b"\x03\x00" * 2400


def _engine() -> V.VoiceEngine:
    engine = V.VoiceEngine(
        duck_seed=7, stream=V.SoundStream(), helper_bin=Path("/nonexistent"), rng_seed=1
    )
    fake = FakeHelper()
    engine.helper = fake  # type: ignore[assignment]
    engine.voice.helper = fake  # type: ignore[assignment]
    return engine


def test_real_engine_releases_pcm_and_keeps_reflex_alarm_on_takeover() -> None:
    stream = V.SoundStream()
    engine = V.VoiceEngine(duck_seed=7, stream=stream, helper_bin=Path("/nonexistent"), rng_seed=1)
    # Keep VoiceEngine, SoundStream, RobotdVoice rendering and after_tick real; only the WAV
    # segments are deterministic test samples so this regression needs no external voice assets.
    engine.voice = _voice()
    brain = type("Brain", (), {"felt_head": None})()
    lanes = type("Lanes", (), {})()
    lanes.brain = brain
    lanes.senses = type("Senses", (), {"sim_s": staticmethod(lambda t: t / 1000.0)})()
    lanes.sound = NSL.Sound(brain=brain, motor=None, engine=engine)
    driver = type("Driver", (), {})()
    driver.lanes = lanes
    driver._last_t_ms = 0
    driver._last_cmd = type("Command", (), {"behaviour": "wheee_ride", "reflex": None})()
    driver._tof_frame = None
    driver._world = {}
    driver._body = None

    def sound_out(hold: bool):
        return type(
            "SoundOut",
            (),
            {
                "wheee_hold": hold,
                "theremin_up": False,
                "dancing": False,
                "beat": type("Beat", (), {"active": False, "bpm": 0.0, "lock": 0.0})(),
                "chirp": None,
                "timing_s": 0.0,
                "felt_nods": 0,
            },
        )()

    brain.sound_out = sound_out(True)
    intents = NSL.after_tick(driver, type("State", (), {"t_ns": 0})(), [])
    first = engine.last
    assert first is not None and first.ride == "riding" and first.owns_pcm
    assert first.chunk is not None and first.chunk.duck
    assert lanes.sound.wheee_owned

    # SoundMotor must let the first reflex vocal through even though the prior PCM tick owned the
    # stream; after_tick then releases that stream in this same tick.
    motor = NSL.SoundMotor(
        lambda *_: (
            type("Request", (), {"head": (0.0, 0.0, 0.0, 0.0)})(),
            [{"tag": "alarm"}],
        ),
        brain,
        engine,
    )
    req, vocals = motor(
        50,
        type("Command", (), {"reflex": "startle"})(),
        None,
        None,
    )
    assert vocals == [{"tag": "alarm"}]
    assert req.head == (0.0, 0.0, 0.0, 0.0)
    brain.sound_out = sound_out(False)
    driver._last_t_ms = 50
    driver._last_cmd = type("Command", (), {"behaviour": "startle", "reflex": "startle"})()
    NSL.after_tick(driver, type("State", (), {"t_ns": 50_000_000})(), intents)
    stopped = engine.last
    assert stopped is not None and stopped.ride == "off" and not stopped.owns_pcm
    assert stopped.chunk is not None and stopped.chunk.duck is None
    assert not lanes.sound.wheee_owned


def test_the_theremin_plays_the_hand_and_lets_go_with_a_fade() -> None:
    engine = _engine()
    t = 0.0
    out = None
    for _ in range(10):
        out = engine.tick(
            t_s=t,
            sim_s=t,
            w=t,
            wheee_hold=False,
            theremin_up=True,
            tof=_frame(0.25, 5, 8),
            tof_age_s=0.02,
        )
        t += V.TICK_S
    assert out is not None and out.ride == "theremin" and out.owns_pcm
    closeness = (0.70 - 0.25) / 0.60
    assert out.note_hz == pytest.approx(200 + 400 * closeness, abs=1.0)
    assert out.mouth == pytest.approx(closeness, abs=0.01)
    assert out.chunk is not None and out.chunk.duck is not None
    # the hand goes: silence at the last pitch (level 0), the instrument still up
    out = engine.tick(
        t_s=t,
        sim_s=t,
        w=t,
        wheee_hold=False,
        theremin_up=True,
        tof=_frame(0.25, 255, 8),
        tof_age_s=0.02,
    )
    t += 1.0
    out = engine.tick(
        t_s=t,
        sim_s=t,
        w=t,
        wheee_hold=False,
        theremin_up=True,
        tof=_frame(0.25, 255, 8),
        tof_age_s=0.02,
    )
    assert out.closeness is None and out.ride == "theremin"
    # put down: it fades, then the PCM is free
    rides = []
    for _ in range(40):
        t += V.TICK_S
        rides.append(
            engine.tick(
                t_s=t, sim_s=t, w=t, wheee_hold=False, theremin_up=False, tof=None, tof_age_s=None
            ).ride
        )
    assert rides[0] == "landing" and rides[-1] == "off"


def test_the_ride_holds_while_re_notified_and_lands_300_ms_after() -> None:
    engine = _engine()
    t = 0.0
    for _ in range(50):
        out = engine.tick(
            t_s=t, sim_s=t, w=t, wheee_hold=True, theremin_up=False, tof=None, tof_age_s=None
        )
        t += V.TICK_S
    assert out.ride == "riding"
    states = []
    for _ in range(40):
        states.append(
            engine.tick(
                t_s=t, sim_s=t, w=t, wheee_hold=False, theremin_up=False, tof=None, tof_age_s=None
            ).ride
        )
        t += V.TICK_S
    assert states[:14] == ["riding"] * 14  # still fresh (< 300 ms)
    assert "landing" in states[14:17]


def test_a_ride_after_the_theremin_lands_through_its_end_segment() -> None:
    # (the dry run's order: the instrument played and put down, then a ride — its landing is
    # the ride's, held until the end segment is written, not a faded-out instrument's)
    engine = _engine()
    t = 0.0

    def tick(**kw: Any) -> V.TickOut:
        nonlocal t
        out = engine.tick(t_s=t, sim_s=t, w=t, theremin_up=kw.pop("up", False), **kw)
        t += V.TICK_S
        return out

    for _ in range(10):
        tick(wheee_hold=False, up=True, tof=_frame(0.25, 5, 8), tof_age_s=0.02)
    for _ in range(5):
        tick(wheee_hold=False, tof=None, tof_age_s=None)
    assert engine.last is not None and engine.last.ride == "off"
    for _ in range(50):
        assert tick(wheee_hold=True, tof=None, tof_age_s=None).ride == "riding"
    after = []
    for _ in range(200):
        out = tick(wheee_hold=False, tof=None, tof_age_s=None)
        after.append((out.ride, out.owns_pcm, engine.voice._part))
        if out.ride == "off":
            break
    landing = [a for a in after if a[0] == "landing"]
    assert landing and all(owns for _, owns, _ in landing)
    ends = [a for a in after if a[2] == "end"]
    assert ends and all(ride == "landing" for ride, _, _ in ends)  # the end plays while landing
    assert after[-1][0] == "off"


def test_the_theremin_keeps_the_pcm_from_a_ride() -> None:
    engine = _engine()
    engine.tick(
        t_s=0.0,
        sim_s=0.0,
        w=0.0,
        wheee_hold=False,
        theremin_up=True,
        tof=_frame(0.3, 5, 8),
        tof_age_s=0.0,
    )
    out = engine.tick(
        t_s=0.02,
        sim_s=0.02,
        w=0.02,
        wheee_hold=True,
        theremin_up=True,
        tof=_frame(0.3, 5, 8),
        tof_age_s=0.0,
    )
    assert out.ride == "theremin"


# ── the page's stream ──────────────────────────────────────────────────────────


def test_frames_round_trip() -> None:
    stream = V.SoundStream()
    a = stream.push(1.0, 2.0, 960, b"\x01\x00" * 960, None, [(10, 1)])
    b = stream.push(1.02, 2.02, 960, None, b"\x02\x00" * 960, [])
    c = stream.push(1.04, 2.04, 960, None, None, [(0, 2), (480, 1)])
    blob = a.encode() + b.encode() + c.encode()
    chunks, used = V.decode(blob + b"\x00" * 7)
    assert used == len(blob)
    assert [x.seq for x in chunks] == [1, 2, 3]
    assert chunks[0].duck == b"\x01\x00" * 960 and chunks[0].room is None
    assert chunks[1].room == b"\x02\x00" * 960 and chunks[1].marks == ()
    assert chunks[2].marks == ((0, 2), (480, 1)) and chunks[2].w == pytest.approx(1.04)
    assert [x.seq for x in stream.since(1)] == [2, 3]


def test_the_room_plays_the_world_s_song_continuously() -> None:
    pytest.importorskip("numpy")
    from duckmind.world.neuro_music import SR, Song

    room = V.RoomMusic()
    room.set({"on": True, "bpm": 104.0, "seed": 7, "t0": 10.0})
    out = b""
    marks = []
    sim = 9.98
    for k in range(60):
        pcm, m = room.render(sim + k * V.TICK_S + (0.0004 if k % 7 == 3 else 0.0), 960)
        out += pcm or b"\x00\x00" * 960
        marks += [(k * 960 + off, kind) for off, kind in m]
    song = Song(104.0, 7)
    ref = song.pcm16(0, 960 * 60 - 960)
    assert out[2 * 960 :] == ref  # (the first tick is before the song: silence)
    beat = int(round(60.0 / 104.0 * SR))
    assert [p for p, _ in marks][:2] == [960, 960 + beat]


def _room_run(step_s: float, jitter_s: float, ticks: int, jump_at: int | None = None) -> Any:
    """The room's song over ``ticks`` ticks whose physics time advances ``step_s`` a tick, read
    with a uniform jitter of ±``jitter_s`` (and a clock jump of +0.5 s at ``jump_at``)."""
    import random

    rng = random.Random(3)
    room = V.RoomMusic()
    room.set({"on": True, "bpm": 104.0, "seed": 7, "t0": 10.3})
    truth = 10.0
    offs = []
    for k in range(ticks):
        if jump_at is not None and k == jump_at:
            truth += 0.5
        reading = truth + rng.uniform(-jitter_s, jitter_s)
        start = room.next_sample
        room.render(reading, 960)
        if start is not None and truth > 11.0:
            offs.append((start - (truth - 10.3) * 48000) / 48.0)  # ms the stream is off
        truth += step_s
    return room, offs


def test_the_room_follows_a_jittery_physics_clock_without_jumps() -> None:
    pytest.importorskip("numpy")
    room, offs = _room_run(0.02, 0.016, 3000)  # 60 s, the live jitter
    assert room.jumps == 0 and room.slews <= 20
    assert max(abs(o) for o in offs) < 12.0  # ms: the page's music stays on the duck's song


@pytest.mark.parametrize("step", [0.01998, 0.02002])
def test_the_room_slews_after_a_slightly_fast_or_slow_physics_clock(step: float) -> None:
    pytest.importorskip("numpy")
    room, offs = _room_run(step, 0.016, 3000)  # 0.1 % off for 60 s: 60 ms of drift
    assert room.jumps == 0 and room.slews > 0
    assert max(abs(o) for o in offs) < 16.0


def test_the_room_jumps_only_when_the_physics_clock_jumps() -> None:
    pytest.importorskip("numpy")
    room, offs = _room_run(0.02, 0.016, 600, jump_at=300)
    assert room.jumps == 1
    assert abs(offs[-1]) < 16.0


def test_a_physics_stall_is_one_jump_not_half_a_minute_off() -> None:
    # (live, under load: the physics clock fell 50 ms behind the ticks once and stayed there)
    pytest.importorskip("numpy")
    import random

    rng = random.Random(4)
    room = V.RoomMusic()
    room.set({"on": True, "bpm": 104.0, "seed": 7, "t0": 10.3})
    truth = 10.0
    offs = []
    for k in range(1500):
        if k == 400:
            truth -= 0.05  # the physics stalled 50 ms
        start = room.next_sample
        room.render(truth + rng.uniform(-0.016, 0.016), 960)
        if start is not None and k > 400:
            offs.append((k, (start - (truth - 10.3) * 48000) / 48.0))
        truth += 0.02
    assert room.jumps == 1
    assert all(abs(o) < 16.0 for k, o in offs if k > 400 + 150)  # lined up within 3 s


# ── upstream's own samples ─────────────────────────────────────────────────────


@needs_helper
@pytest.mark.skipif(not SOUNDS_BIN.is_file(), reason="upstream sounds binary not built")
@pytest.mark.parametrize("seed", [7, 2838558865])
def test_the_helper_is_upstream_s_theremin_sample_for_sample(seed: int) -> None:
    """``sounds theremin`` auditions the live voice exactly as robotd drives it (15-Hz parameter
    updates of Stream::wheee(p, 0)); the helper, driven the same way, gives the same samples."""
    out = Path(tempfile.mkdtemp()) / "sweep.wav"
    subprocess.run(
        [str(SOUNDS_BIN), "theremin", "--out", str(out), "--seed", str(seed)],
        check=True,
        capture_output=True,
    )
    ref = V._wav_payload(out)
    helper = V.VoiceHelper()
    helper.voice(seed, "wheee", 0)
    legs = (
        (1.5, 0.05, True),
        (2.5, 0.95, True),
        (1.5, 0.95, True),
        (2.0, 0.55, True),
        (1.5, 0.05, True),
        (1.0, 0.0, False),
    )
    got = bytearray()
    start, wobble = 0.0, 0.0
    for seconds, to, present in legs:
        frames = int(math.floor(seconds * 15.0 + 0.5))  # Rust's round()
        for f in range(frames):
            closeness = start + (to - start) * ((f + 1.0) / frames)
            if abs(to - start) < 1e-9:
                wobble += 2 * math.pi * 0.8 / 15.0
                closeness += 0.04 * math.sin(wobble)
            helper.set(helper.hz_at(closeness), 1.0 if present else 0.0, closeness)
            got += helper.block(3200)
        start = to
    helper.close()
    assert bytes(got) == ref


@needs_helper
def test_the_ride_segments_are_the_banks_files_byte_for_byte() -> None:
    bank = paths.data_dir() / "voices" / "v5" / "seed7" / "wheee"
    if not bank.is_dir():
        pytest.skip("no bench bank (voices/v5/seed7 in the data folder)")
    helper = V.VoiceHelper()
    try:
        for variant, letter in enumerate("abcdef"):
            got = helper.segments(7, variant)
            files = tuple(
                V._wav_payload(bank / f"wheee_{p}_{letter}.wav") for p in ("start", "loop", "end")
            )
            assert got == files
    finally:
        helper.close()


@needs_helper
def test_the_voice_seed_is_the_voice_workflows() -> None:
    helper = V.VoiceHelper()
    try:
        assert V.voice_seed(7) == helper.seed_from_id("duckmind-duck:7") == 2_838_558_865
    finally:
        helper.close()
