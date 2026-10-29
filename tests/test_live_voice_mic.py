"""Live voice, LV0: the duck's microphone as a PCM stream (``duckmind.simweb.neuro_mic``): the format,
the level the world's own dB gives, family speech at a place, the world's other sounds, the duck's
own calls as an echo with its flag and windows, a browser microphone's audio, and the sound
adapter that ticks it. numpy only: no MuJoCo, no network."""

from __future__ import annotations

import dataclasses
import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from duckmind.simweb import neuro_mic as M
from duckmind.simweb import neuro_sound_live as NSL
from duckmind.simweb import neuro_voice_stream as V
from duckmind.world import neuro_audio as A
from duckmind.world import neuro_senses as ns
from duckmind.world import neuro_sound_synth as S
from duckmind.world.neuro_utterances import MANIFEST, UtteranceLibrary
from duckmind.world.neuro_world import level_at

FRAME = A.FRAME
NOISE_DBFS = ns.ROOM_DB - A.FULL_SCALE_DB  # the quiet room at the microphone: −62 dBFS


def _library(root: Path) -> UtteranceLibrary:
    """Three scripted lines of speech-like sound (so the tests need no recorded files)."""
    lines = {
        "hello": S.render("voice", seconds=1.0, seed=3, ident=1)[0],
        "call": S.render("voice", seconds=1.4, seed=3, ident=0)[0],
        "tone": (0.4 * np.sin(2 * math.pi * 500 * np.arange(16_000) / 16_000)).astype(np.float32),
    }
    entries = []
    for name, wave in lines.items():
        wave = A.normalise(wave)
        A.write_wav(root / f"{name}.wav", A.to_pcm16(wave), A.SR)
        entries.append(
            {
                "id": name,
                "text": name,
                "speaker": "mom",
                "persona": "partner",
                "intent": "call_name" if name == "call" else "greet",
                "name": "蛋黃" if name == "call" else None,
                "file": f"{name}.wav",
                "duration_s": len(wave) / A.SR,
                "samples": len(wave),
                "active_rms_dbfs": -26.0,
            }
        )
    body = {"version": 1, "sample_rate": A.SR, "speakers": {}, "utterances": entries}
    (root / MANIFEST).write_text(json.dumps(body), encoding="utf-8")
    return UtteranceLibrary(root)


def _event(
    ident: int,
    kind: str,
    t0: float,
    seconds: float,
    *,
    db1m: float = 62.0,
    metres: float = 1.0,
    wall: bool = False,
    utterance: str | None = None,
    src: str = "person:owner",
    env: str = "speech",
) -> dict[str, Any]:
    """A world sound as the section ``neuro`` lists it (``SoundEvent.view()``), its level at the
    head by the world's own :func:`level_at`."""
    item: dict[str, Any] = {
        "id": ident,
        "kind": kind,
        "t0": t0,
        "t1": t0 + seconds,
        "db1m": db1m,
        "db": level_at(db1m, metres, wall=wall),
        "env": env,
        "name": False,
        "wall": wall,
        "src": src,
        "pos_mm": [0, 0, 1450],
    }
    if utterance is not None:
        item["utterance"] = utterance
    return item


def _run(
    mic: M.MicStream,
    seconds: float,
    sounds: list[dict[str, Any]] | None,
    *,
    start: float = 0.0,
    **kwargs: Any,
) -> list[M.MicFrame]:
    frames = []
    for k in range(int(round(seconds / A.FRAME_S))):
        frames.append(mic.tick(start + k * A.FRAME_S, sounds, **kwargs))
    return frames


def _signal(frames: list[M.MicFrame]) -> np.ndarray:
    return np.concatenate([A.to_float(f.pcm) for f in frames])


def _active_db(x: np.ndarray) -> float:
    return A.active_rms_dbfs(x)


# ── the format ───────────────────────────────────────────────────────────────────────


def test_a_frame_is_320_samples_of_16bit_mono_in_order_on_the_physics_clock() -> None:
    mic = M.MicStream(seed=7)
    frames = _run(mic, 1.0, [], start=5.0)
    assert [f.seq for f in frames] == list(range(1, 51))
    assert all(len(f.pcm) == A.FRAME_BYTES == 640 for f in frames)
    assert frames[0].sim_s == 5.0
    assert [round(b.sim_s - a.sim_s, 6) for a, b in zip(frames, frames[1:], strict=False)] == [
        0.02
    ] * 49
    assert not any(f.echo for f in frames)
    # (little-endian int16: what Gemini Live's ``audio/pcm;rate=16000`` takes)
    assert np.frombuffer(frames[10].pcm, dtype="<i2").shape == (320,)


def test_a_quiet_room_is_the_hearing_models_30_db_and_the_ducks_motors_add_to_it() -> None:
    mic = M.MicStream(seed=7)
    quiet = _signal(_run(mic, 2.0, []))
    assert A.rms_dbfs(quiet) == pytest.approx(NOISE_DBFS, abs=0.6)
    assert NOISE_DBFS == pytest.approx(-62.0)
    loud_motors = _signal(_run(M.MicStream(seed=7), 2.0, [], motor_db=50.0))
    assert A.rms_dbfs(loud_motors) == pytest.approx(50.0 - A.FULL_SCALE_DB, abs=0.6)


def test_the_stream_is_deterministic_for_a_seed_and_its_inputs(tmp_path: Path) -> None:
    library = _library(tmp_path)
    sounds = [_event(1, "voice", 0.3, 1.0, utterance="hello")]

    def run(seed: int) -> bytes:
        return b"".join(f.pcm for f in _run(M.MicStream(seed=seed, library=library), 2.0, sounds))

    assert run(7) == run(7) and run(7) != run(8)


def test_the_ring_keeps_ten_seconds_and_readers_ask_since_a_frame() -> None:
    mic = M.MicStream(seed=7)
    _run(mic, 11.0, [])
    assert len(mic.frames) == M.KEEP_FRAMES == 500
    assert mic.frames[0].seq == mic.seq - 499
    got = mic.since(mic.seq - 3)
    assert [f.seq for f in got] == [mic.seq - 2, mic.seq - 1, mic.seq]
    assert mic.wait(mic.seq, 0.01) == []  # nothing new: it waits, then says so


def test_listeners_see_every_frame_and_a_faulty_one_never_stops_the_microphone() -> None:
    mic = M.MicStream(seed=7)
    seen: list[int] = []

    def broken(frame: M.MicFrame) -> None:
        raise RuntimeError("listener fault")

    mic.listen(broken)
    mic.listen(lambda frame: seen.append(frame.seq))
    _run(mic, 0.2, [])
    assert (
        seen == list(range(1, 11)) and len(mic.errors) == 10 and "listener fault" in mic.errors[0]
    )


# ── family speech, at the world's level ──────────────────────────────────────────────


def test_a_scripted_line_comes_out_at_the_level_the_world_gives_its_place(tmp_path: Path) -> None:
    library = _library(tmp_path)
    levels = {}
    for metres in (1.0, 2.0, 4.0):
        sounds = [_event(1, "voice", 0.2, 1.0, metres=metres, utterance="hello")]
        mic = M.MicStream(seed=7, library=library)
        x = _signal(_run(mic, 1.6, sounds))
        speech = x[int(0.25 * A.SR) : int(1.15 * A.SR)]
        levels[metres] = _active_db(speech)
        expected = level_at(62.0, metres) - A.FULL_SCALE_DB
        assert levels[metres] == pytest.approx(expected, abs=1.0), metres
    # farther is quieter, by what the world's own model says (6 dB a doubling, less the room's field)
    assert levels[1.0] - levels[2.0] == pytest.approx(
        level_at(62, 1.0) - level_at(62, 2.0), abs=1.0
    )
    assert levels[2.0] > levels[4.0]


def test_a_wall_between_takes_the_worlds_15_db_off(tmp_path: Path) -> None:
    library = _library(tmp_path)

    def level(wall: bool) -> float:
        sounds = [_event(1, "voice", 0.2, 1.0, metres=1.5, wall=wall, utterance="hello")]
        x = _signal(_run(M.MicStream(seed=7, library=library), 1.6, sounds))
        return _active_db(x[int(0.25 * A.SR) : int(1.15 * A.SR)])

    assert level(False) - level(True) == pytest.approx(
        level_at(62, 1.5) - level_at(62, 1.5, wall=True), abs=1.0
    )


def test_the_hearing_model_and_the_microphone_hear_the_same_event_start(tmp_path: Path) -> None:
    """The stimulus is one world event: the hearing model's onset and the stream's first loud frame
    come together (the hearing model's onset lags by its own 20–40 ms)."""
    library = _library(tmp_path)
    t0 = 0.5
    sounds = [_event(1, "name", t0, 1.0, db1m=70.0, metres=1.2, utterance="call")]
    sounds[0]["name"] = True
    mic = M.MicStream(seed=7, library=library)
    hearing = ns.HearingModel(seed=7)
    onset_at = loud_at = None
    for k in range(100):
        s = k * A.FRAME_S
        frame = mic.tick(s, sounds)
        hearing.update(sounds, now_s=s)
        sense = hearing.step(s, 28.0)
        if onset_at is None and sense.onset > 0:
            onset_at = s
        if loud_at is None and frame.level_dbfs > NOISE_DBFS + 12:
            loud_at = s
    assert onset_at is not None and loud_at is not None
    assert t0 <= loud_at <= t0 + 0.12  # the speech starts with a soft consonant at most
    assert t0 <= onset_at <= t0 + 0.12
    assert abs(onset_at - loud_at) <= 0.08


def test_a_line_first_seen_late_is_played_whole_from_its_start(tmp_path: Path) -> None:
    library = _library(tmp_path)
    sounds = [_event(1, "voice", 0.0, 1.0, utterance="tone")]
    mic = M.MicStream(seed=7, library=library)
    _run(mic, 0.4, [])  # nothing known yet: the world's list reaches the stream late
    frames = _run(mic, 1.5, sounds, start=0.4)
    x = _signal(frames)
    wave = A.to_float(library.pcm("tone"))
    gain = 10 ** ((sounds[0]["db"] - A.FULL_SCALE_DB - A.REF_DBFS) / 20)
    # from the first frame it saw it, sample 0 of the line, whole (as the hearing model's LATE_S)
    assert np.allclose(x[:3000], (wave * gain)[:3000] * np.linspace(1, 1, 3000), atol=0.004)
    assert np.allclose(x[3000:15000], (wave * gain)[3000:15000], atol=0.004)


def test_a_line_that_starts_later_waits_for_its_time(tmp_path: Path) -> None:
    library = _library(tmp_path)
    sounds = [_event(1, "voice", 0.5, 1.0, utterance="tone")]
    frames = _run(M.MicStream(seed=7, library=library), 1.2, sounds)
    first = next(f.sim_s for f in frames if f.level_dbfs > NOISE_DBFS + 10)
    assert first == pytest.approx(0.5, abs=0.021)


def test_no_news_from_the_world_keeps_a_sound_playing(tmp_path: Path) -> None:
    library = _library(tmp_path)
    sounds = [_event(1, "voice", 0.0, 1.0, utterance="tone")]
    mic = M.MicStream(seed=7, library=library)
    a = _run(mic, 0.3, sounds)
    b = _run(mic, 0.5, None, start=0.3)  # (the driver's world copy was not there this tick)
    assert all(f.level_dbfs > NOISE_DBFS + 15 for f in a + b)


def test_the_microphone_clock_follows_the_physics_clock_without_bending_the_audio(
    tmp_path: Path,
) -> None:
    library = _library(tmp_path)
    sounds = [_event(1, "voice", 0.0, 1.0, utterance="tone")]
    rng = random.Random(1)
    mic = M.MicStream(seed=7, library=library)
    jittery = [
        mic.tick(k * 0.02 + (rng.uniform(-0.004, 0.004) if k else 0.0), sounds) for k in range(60)
    ]
    steady = _run(M.MicStream(seed=7, library=library), 1.2, sounds)
    a, b = _signal(jittery), _signal(steady)
    assert np.allclose(a[:16_000], b[:16_000], atol=0.002), "a jittery clock moved no sample"
    assert mic.resyncs == 0
    mic.tick(60 * 0.02 + 5.0, sounds)  # the physics clock jumped
    assert mic.resyncs == 1


def test_an_unknown_line_is_said_as_unscripted_speech_and_counted(tmp_path: Path) -> None:
    library = _library(tmp_path)
    sounds = [_event(1, "voice", 0.1, 1.0, utterance="not_in_the_library")]
    mic = M.MicStream(seed=7, library=library)
    frames = _run(mic, 0.8, sounds)
    assert mic.missing_lines == 1 and max(f.level_dbfs for f in frames) > NOISE_DBFS + 15


# ── the world's other sounds ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kind,env,db1m",
    [
        ("clap", "impulse", 82.0),
        ("bang", "impulse", 90.0),
        ("door", "impulse", 74.0),
        ("knock", "impulse", 72.0),
        ("footsteps", "impulse", 52.0),
        ("voice", "speech", 62.0),
        ("tv", "steady", 60.0),
        ("noise", "steady", 60.0),
    ],
)
def test_every_world_sound_has_audio_at_its_level_and_ends(
    kind: str, env: str, db1m: float
) -> None:
    sounds = [_event(4, kind, 0.2, 1.0, db1m=db1m, metres=1.0, env=env, src="world")]
    mic = M.MicStream(seed=7)
    frames = _run(mic, 3.0, sounds)
    loud = [f.level_dbfs for f in frames[10:40]]
    assert max(loud) > NOISE_DBFS + 10, kind
    tail = _signal(frames[-40:])  # two seconds on: a steady sound ended with its event
    assert A.rms_dbfs(tail) < NOISE_DBFS + 3, kind
    assert not any(f.echo for f in frames), "a world sound is never the duck's echo"


def test_the_world_song_is_in_the_microphone_once_with_its_drum_hits_in_it() -> None:
    from duckmind.world.neuro_music import Song

    bpm, seed, t0 = 104.0, 7, 0.3
    bed = _event(1, "music", t0, 3600.0, db1m=66.0, metres=1.0, env="steady", src="music:bed")
    hits = [
        _event(2 + k, "music", t0 + 0.5 * k, 0.07, db1m=80.0, env="impulse", src="music:kick")
        for k in range(6)
    ]
    music = {"on": True, "bpm": bpm, "seed": seed, "t0": t0}
    with_hits = _run(M.MicStream(seed=7), 3.0, [bed, *hits], music=music)
    bed_only = _run(M.MicStream(seed=7), 3.0, [bed], music=music)
    assert _signal(with_hits).tolist() == _signal(bed_only).tolist(), "the hits are the song's"
    x = _signal(bed_only)
    assert _active_db(x[int(0.6 * A.SR) :]) == pytest.approx(bed["db"] - A.FULL_SCALE_DB, abs=1.5)
    # the very samples of the world's song (decimated 48 → 16 kHz), where the page plays them too
    song = np.asarray(Song(bpm=bpm, seed=seed).render(int(1.0 * 48_000), 48_000), dtype=np.float64)
    want = A.StreamDecimator(3).process(song)
    got = x[int((1.0 + t0) * A.SR) : int((2.0 + t0) * A.SR)]
    corr = float(np.corrcoef(want[2000:-2000], got[2000:-2000])[0, 1])
    assert corr > 0.97, corr


def test_music_off_ends_the_song_in_the_microphone() -> None:
    music = {"on": True, "bpm": 104.0, "seed": 7, "t0": 0.0}
    bed = _event(1, "music", 0.0, 3600.0, env="steady", src="music:bed")
    mic = M.MicStream(seed=7)
    _run(mic, 1.0, [bed], music=music)
    bed["t1"] = 1.0  # music_off: the world ends the bed now
    after = _run(mic, 1.0, [bed], start=1.0, music=music)
    assert max(f.level_dbfs for f in after[3:]) < NOISE_DBFS + 3


# ── the duck's own voice ─────────────────────────────────────────────────────────────


class _Bank:
    """A call bank: the same short tone for every tag but ``mute``."""

    def __init__(self, seconds: float = 0.3) -> None:
        self.wave = (
            0.5 * np.sin(2 * math.pi * 900 * np.arange(int(seconds * A.SR)) / A.SR)
        ).astype(np.float32)
        self.asked: list[str] = []

    def call(self, tag: str, rng: random.Random) -> tuple[str, np.ndarray] | None:
        self.asked.append(tag)
        return None if tag == "mute" else ("a", self.wave)


def test_the_ducks_call_comes_back_as_a_loud_flagged_echo_with_its_window() -> None:
    bank = _Bank(0.3)
    mic = M.MicStream(seed=7, calls=bank)
    before = _run(mic, 0.2, [])
    asked = [{"tag": "chirp", "level": 1.0}]
    frames = [mic.tick(0.2, [], calls=asked)] + _run(mic, 0.7, [], start=0.22)
    after = before + frames
    flagged = [f for f in after if f.echo]
    # the call (0.3 s) and its tail (0.12 s) are flagged: 21 frames, no more, no fewer
    assert len(flagged) == math.ceil((0.3 + M.ECHO_TAIL_S) / 0.02) + 0
    assert min(f.seq for f in flagged) == 11 and not any(f.echo for f in before)
    # loud: a speaker a few cm from the microphone (−9 dB of a −9 dBFS tone)
    expected = A.rms_dbfs(bank.wave) + M.SPEAKER_MIC_GAIN_DB
    assert frames[3].level_dbfs == pytest.approx(expected, abs=0.7)
    assert expected > NOISE_DBFS + 30
    (window,) = mic.vocal_windows()
    assert (window.tag, window.kind) == ("chirp", "call")
    assert window.t0 == pytest.approx(0.2) and window.t1 == pytest.approx(0.5)
    assert mic.in_echo(0.3) and mic.in_echo(0.55) and not mic.in_echo(0.7) and not mic.in_echo(0.1)
    assert mic.echo_frames == len(flagged)
    assert asked and bank.asked == ["chirp"]


def test_the_calls_window_ends_where_its_samples_do_when_the_clock_has_wandered() -> None:
    # (the stream's clock follows the physics clock slowly: after a stall it runs fast for a while,
    # so frame stamps and samples part; the window must still end with the call's last sample)
    mic = M.MicStream(seed=7, calls=_Bank(0.4))
    frames = []
    for k in range(70):
        phys = k * 0.02 + (0.2 if k >= 6 else 0.0)  # the physics clock stalls ahead by 0.2 s
        asked = [{"tag": "chirp", "level": 1.0}] if k == 5 else []
        frames.append(mic.tick(phys, [], calls=asked))
    loud = [f for f in frames if f.level_dbfs > NOISE_DBFS + 20]
    assert len(loud) == 20  # (0.4 s of the call: 20 frames of samples, however the clock reads)
    span = loud[-1].sim_s - loud[0].sim_s
    assert abs(span - 19 * 0.02) > 0.01, "the clock did wander against the samples"
    (window,) = mic.vocal_windows()
    assert window.t0 == pytest.approx(loud[0].sim_s)
    assert loud[-1].sim_s < window.t1 <= loud[-1].sim_s + 0.02 + 1e-9
    flagged = [f for f in frames if f.echo]
    assert flagged[-1].sim_s < window.t1 + M.ECHO_TAIL_S  # (its tail is flagged, no more)
    assert not any(f.echo for f in frames if f.sim_s >= window.t1 + M.ECHO_TAIL_S)


def test_two_calls_at_once_never_stop_the_microphone() -> None:
    # (live: the reply and the duck's own answer came a frame apart; when the shorter one ended,
    # removing it from the list compared two sample arrays of different lengths, the tick raised
    # and no frame was made until the longer one ended: audio lost, and the detector put off)
    class Bank:
        def call(self, tag: str, rng: random.Random) -> tuple[str, np.ndarray]:
            n = {"chirp": 4000, "greet": 9000}[tag]
            return ("a", (0.4 * np.sin(2 * math.pi * 700 * np.arange(n) / A.SR)).astype(np.float32))

    mic = M.MicStream(seed=7, calls=Bank())
    first = mic.tick(0.0, [], calls=[{"tag": "greet", "level": 0.8}])
    second = mic.tick(0.02, [], calls=[{"tag": "chirp", "level": 0.8}])
    frames = [first, second] + [mic.tick(0.04 + 0.02 * k, []) for k in range(60)]
    assert mic.errors == [] and [f.seq for f in frames] == list(range(1, 63))
    assert len(mic.vocal_windows()) == 2 and mic._echoes == []


def test_a_quiet_call_echoes_quieter() -> None:
    def level(strength: float) -> float:
        mic = M.MicStream(seed=7, calls=_Bank())
        frame = mic.tick(0.0, [], calls=[{"tag": "chirp", "level": strength}])
        return frame.level_dbfs

    assert level(1.0) - level(0.25) == pytest.approx(12.0, abs=1.0)


def test_a_call_with_no_sound_for_it_still_has_its_window_and_flag() -> None:
    mic = M.MicStream(seed=7, calls=_Bank())
    frames = [mic.tick(0.0, [], calls=[{"tag": "mute"}])] + _run(mic, 0.7, [], start=0.02)
    (window,) = mic.vocal_windows()
    assert window.t1 == pytest.approx(0.4)  # (the default length of a tag it has no wav for)
    flagged = [f for f in frames if f.echo]
    assert flagged and all(f.level_dbfs < NOISE_DBFS + 3 for f in frames), "no sound, only the flag"
    nobank = M.MicStream(seed=7)
    nobank.tick(0.0, [], calls=[{"tag": "alarm", "level": 0.9}])
    assert nobank.vocal_windows()[0].t1 == pytest.approx(M.CALL_S["alarm"])


def test_the_voice_engines_plane_is_an_echo_too_and_a_silent_plane_is_not() -> None:
    tone48 = (0.4 * np.sin(2 * math.pi * 700 * np.arange(960) / 48_000)).astype(np.float32)
    plane = A.to_pcm16(tone48)
    mic = M.MicStream(seed=7)
    quiet = [mic.tick(k * 0.02, [], duck_pcm48=bytes(1920)) for k in range(5)]  # theremin, no hand
    ride = [mic.tick(0.1 + k * 0.02, [], duck_pcm48=plane) for k in range(25)]
    gone = [mic.tick(0.6 + k * 0.02, []) for k in range(15)]
    assert not any(f.echo for f in quiet)
    assert all(f.echo for f in ride)
    assert ride[10].level_dbfs == pytest.approx(A.rms_dbfs(tone48) + M.SPEAKER_MIC_GAIN_DB, abs=0.8)
    flagged_after = [f for f in gone if f.echo]
    assert 1 <= len(flagged_after) <= math.ceil((0.04 + M.ECHO_TAIL_S) / 0.02) + 1
    (window,) = mic.vocal_windows()
    assert window.kind == "stream" and window.t0 == pytest.approx(0.1)
    assert window.t1 is not None and window.t1 == pytest.approx(0.6, abs=0.001)


def test_voice_while_people_speak_keeps_both_and_only_the_ducks_is_flagged(tmp_path: Path) -> None:
    library = _library(tmp_path)
    sounds = [_event(1, "voice", 0.0, 1.0, utterance="hello")]
    mic = M.MicStream(seed=7, library=library, calls=_Bank(0.3))
    frames = _run(mic, 0.4, sounds)
    frames += [mic.tick(0.4, sounds, calls=[{"tag": "chirp", "level": 0.5}])]
    frames += _run(mic, 0.3, sounds, start=0.42)
    assert [f.echo for f in frames[:20]] == [False] * 20
    assert all(f.echo for f in frames[20:35])
    truth = mic.truth_since(0)
    assert {p["kind"] for t in truth[21:30] for p in t["playing"]} >= {"voice", "own_call"}


def test_frames_carry_audio_and_the_ducks_own_flag_never_what_is_what(tmp_path: Path) -> None:
    library = _library(tmp_path)
    mic = M.MicStream(seed=7, library=library)
    frame = _run(mic, 0.5, [_event(1, "name", 0.0, 1.0, utterance="call")])[-1]
    assert {f.name for f in dataclasses.fields(frame)} == {
        "seq",
        "sim_s",
        "wall",
        "pcm",
        "echo",
        "level_dbfs",
        "cloud_pcm",  # (audio again: the same frame with a browser microphone's person at
        "cloud_level_dbfs",  # their recording's own level; None when nobody speaks into one)
    }
    assert frame.cloud_pcm is None and frame.cloud_level_dbfs is None
    text = repr(frame)
    assert "call" not in text and "name" not in text and "person" not in text
    # what truly played is kept apart, for the evaluators
    truth = mic.truth_since(0)[-1]["playing"]
    assert truth[0]["utterance"] == "call" and truth[0]["kind"] == "name"


# ── a person at a browser microphone ─────────────────────────────────────────────────


def _speech_pcm(seconds: float, amp: float) -> bytes:
    t = np.arange(int(seconds * A.SR)) / A.SR
    return A.to_pcm16(amp * np.sin(2 * math.pi * 400 * t))


def _live_event(ident: int, who: str, t0: float, seconds: float, metres: float) -> dict[str, Any]:
    return _event(
        ident, "voice", t0, seconds, metres=metres, utterance=f"live:{who}", src=f"person:{who}"
    )


def test_live_speech_waits_for_its_world_event_then_plays_at_the_distance_level() -> None:
    mic = M.MicStream(seed=7, wall=lambda: 0.0)
    taken = mic.feed_live("owner", _speech_pcm(0.8, 0.1))
    assert taken == {"accepted_ms": 800, "queued_ms": 800, "dropped_ms": 0}
    waiting = _run(mic, 0.3, [])  # nobody is there yet: the speech is held, not lost
    assert max(f.level_dbfs for f in waiting) < NOISE_DBFS + 3
    assert mic.view()["live"]["owner"]["queued_ms"] == 800
    event = _live_event(9, "owner", 0.3, 1.0, metres=2.0)
    frames = _run(mic, 0.8, [event], start=0.3)
    x = _signal(frames)
    assert _active_db(x[: int(0.7 * A.SR)]) == pytest.approx(event["db"] - A.FULL_SCALE_DB, abs=1.0)
    assert mic.view()["live"]["owner"]["queued_ms"] == 0
    after = _run(mic, 0.3, [event], start=1.1)
    assert max(f.level_dbfs for f in after) < NOISE_DBFS + 3  # the queue is empty: silence again


def test_a_loud_and_a_quiet_talker_land_at_the_same_level_at_the_same_place() -> None:
    def level(amp: float) -> float:
        mic = M.MicStream(seed=7, wall=lambda: 0.0)
        mic.feed_live("owner", _speech_pcm(0.6, amp))
        event = _live_event(9, "owner", 0.0, 1.0, metres=1.0)
        return _active_db(_signal(_run(mic, 0.5, [event])))

    assert level(0.02) == pytest.approx(level(0.4), abs=0.8)


def test_a_far_talker_is_quieter_than_a_near_one() -> None:
    def level(metres: float) -> float:
        mic = M.MicStream(seed=7, wall=lambda: 0.0)
        mic.feed_live("owner", _speech_pcm(0.6, 0.1))
        return _active_db(_signal(_run(mic, 0.5, [_live_event(9, "owner", 0.0, 1.0, metres)])))

    assert level(1.0) - level(3.0) == pytest.approx(level_at(62, 1.0) - level_at(62, 3.0), abs=1.0)


def _cloud_signal(frames: list[M.MicFrame]) -> np.ndarray:
    return np.concatenate([A.to_float(f.cloud_pcm or f.pcm) for f in frames])


def test_the_cloud_hears_a_browser_microphone_person_at_their_recordings_level_wherever_they_stand() -> (
    None
):
    """The duck's ear follows the distance (its reflexes, its local detector), but the cloud is sent
    the real recording at speech's normal level: a person 8 m away is not sent at -50 dBFS under the
    room's and the motors' noise (live 2026-10-04: transcripts of Spanish and Vietnamese words)."""

    def levels(metres: float) -> tuple[float, float]:
        mic = M.MicStream(seed=7, wall=lambda: 0.0)
        mic.feed_live("owner", _speech_pcm(0.6, 0.1))
        frames = _run(mic, 0.5, [_live_event(9, "owner", 0.0, 1.0, metres)])
        return _active_db(_signal(frames)), _active_db(_cloud_signal(frames))

    near_ear, near_cloud = levels(1.0)
    far_ear, far_cloud = levels(8.0)
    assert near_ear - far_ear > 10  # (the ear still follows the distance)
    assert near_cloud == pytest.approx(far_cloud, abs=0.5)
    assert near_cloud == pytest.approx(M.LIVE_CLOUD_DBFS, abs=1.5)
    assert far_cloud > far_ear + 10


def test_only_the_frames_with_a_browser_microphone_person_in_them_carry_a_cloud_copy() -> None:
    mic = M.MicStream(seed=7, wall=lambda: 0.0)
    quiet = _run(mic, 0.2, [])
    assert all(f.cloud_pcm is None and f.cloud_level_dbfs is None for f in quiet)
    mic.feed_live("owner", _speech_pcm(0.3, 0.1))
    frames = _run(mic, 0.6, [_live_event(9, "owner", 0.2, 1.0, 2.0)], start=0.2)
    with_copy = [f for f in frames if f.cloud_pcm is not None]
    assert 10 <= len(with_copy) <= 20  # (the 0.3 s that was spoken: about 15 frames)
    assert all(len(f.cloud_pcm or b"") == A.FRAME_BYTES for f in with_copy)
    assert all(f.cloud_level_dbfs is not None and f.cloud_level_dbfs > -45 for f in with_copy)
    rest = [f for f in frames if f.cloud_pcm is None]
    assert rest and all(f.level_dbfs < NOISE_DBFS + 3 for f in rest[-5:])


def test_the_cloud_copy_differs_from_the_ear_only_in_the_persons_voice() -> None:
    """The room's noise (and the world's other sounds, the duck's own echo) are the same in both:
    take the same frames without the person and what is left of each is the person's voice."""
    event = _live_event(9, "owner", 0.0, 1.0, 8.0)
    mic = M.MicStream(seed=7, wall=lambda: 0.0)
    mic.feed_live("owner", _speech_pcm(0.6, 0.1))
    frames = _run(mic, 0.4, [event])
    empty = _run(M.MicStream(seed=7, wall=lambda: 0.0), 0.4, [])  # (same seed: the same noise)
    ear = _signal(frames) - _signal(empty)
    cloud = _cloud_signal(frames) - _signal(empty)
    assert _active_db(ear) == pytest.approx(event["db"] - A.FULL_SCALE_DB, abs=1.0)
    assert _active_db(cloud) == pytest.approx(M.LIVE_CLOUD_DBFS, abs=1.0)


def test_scripted_lines_and_a_switched_off_cloud_copy_leave_the_frame_as_it_was(
    tmp_path: Path,
) -> None:
    library = _library(tmp_path)
    mic = M.MicStream(seed=7, library=library)
    said = _run(mic, 0.6, [_event(1, "name", 0.0, 1.0, utterance="call")])
    assert all(f.cloud_pcm is None for f in said)  # (a scripted line is the world's, as modelled)
    plain = M.MicStream(seed=7, wall=lambda: 0.0, cloud_live_dbfs=None)
    plain.feed_live("owner", _speech_pcm(0.4, 0.1))
    frames = _run(plain, 0.4, [_live_event(9, "owner", 0.0, 1.0, 2.0)])
    assert all(f.cloud_pcm is None for f in frames)  # (None: the cloud hears what the ear hears)


def test_the_live_queue_keeps_only_the_newest_second() -> None:
    mic = M.MicStream(seed=7, wall=lambda: 0.0)
    taken = [mic.feed_live("owner", _speech_pcm(0.5, 0.1)) for _ in range(6)]
    assert taken[-1]["queued_ms"] == 1000
    assert sum(t["dropped_ms"] for t in taken) == 2000
    assert mic.view()["live"]["owner"]["dropped"] == 2 * A.SR
    with pytest.raises(ValueError):
        mic.feed_live("owner", b"\x01")


def test_two_people_speak_together_and_an_idle_one_is_forgotten() -> None:
    clock = {"t": 0.0}
    mic = M.MicStream(seed=7, wall=lambda: clock["t"])
    mic.feed_live("owner", _speech_pcm(0.4, 0.1))
    mic.feed_live("child", _speech_pcm(0.4, 0.1))
    events = [_live_event(1, "owner", 0.0, 1.0, 1.0), _live_event(2, "child", 0.0, 1.0, 1.0)]
    both = _active_db(_signal(_run(mic, 0.3, events)))
    one_mic = M.MicStream(seed=7, wall=lambda: 0.0)
    one_mic.feed_live("owner", _speech_pcm(0.4, 0.1))
    one = _active_db(_signal(_run(one_mic, 0.3, events[:1])))
    assert both > one + 1.5  # two voices are louder than one
    clock["t"] = 10.0
    _run(mic, 0.1, [], start=1.0)
    assert mic.view()["live"] == {}


# ── the voice bank's calls (the wavs the page plays) ─────────────────────────────────


def _bank_dir(root: Path, vseed: int) -> Path:
    from duckmind.simweb import voice as VB

    folder = root / f"v{VB.BANK_VERSION}" / f"seed{vseed}"
    for tag, names in VB.WHITELIST.items():
        (folder / tag).mkdir(parents=True)
        for name in names:
            freq = 300.0 + 40.0 * (ord(name[-5]) - ord("a")) + 25.0 * len(tag)
            t = np.arange(int(0.2 * 48_000)) / 48_000
            A.write_wav(
                folder / tag / name, A.to_pcm16(0.5 * np.sin(2 * math.pi * freq * t)), 48_000
            )
    return folder


class _FakeVoiceBank:
    def __init__(self, folder: Path, ready: bool = True) -> None:
        self.folder, self._ready = folder, ready

    def ready(self, vseed: int) -> bool:
        return self._ready

    def bank_dir(self, vseed: int) -> Path:
        return self.folder


def test_calls_come_from_the_ducks_own_bank_at_16k_one_random_variant_each_time(
    tmp_path: Path,
) -> None:
    folder = _bank_dir(tmp_path, 4242)
    bank = M.VoiceBankCalls(_FakeVoiceBank(folder), 4242)
    rng = random.Random(1)
    letters = set()
    for _ in range(20):
        got = bank.call("chirp", rng)
        assert got is not None
        letter, wave = got
        letters.add(letter)
        assert len(wave) == pytest.approx(0.2 * A.SR, abs=2) and wave.dtype == np.float32
        assert A.rms_dbfs(wave) == pytest.approx(A.rms_dbfs(np.full(10, 0.5)) - 3.0, abs=0.5)
    assert len(letters) > 3  # (robotd plays a random variant of a tag each time)
    assert bank.call("wheee", rng) is None  # a ride is a stream, not a call
    assert bank.call("not_a_tag", rng) is None
    assert M.VoiceBankCalls(_FakeVoiceBank(folder, ready=False), 4242).call("chirp", rng) is None
    assert M.VoiceBankCalls(None, 4242).call("chirp", rng) is None


def test_a_microphone_made_with_a_voice_engine_uses_that_engines_bank(tmp_path: Path) -> None:
    folder = _bank_dir(tmp_path, 77)
    engine = SimpleNamespace(bank=_FakeVoiceBank(folder), voice_seed=77)
    mic = M.make_mic(seed=3, engine=engine, use_default_library=False)
    assert isinstance(mic.calls, M.VoiceBankCalls) and mic.library is None
    mic.tick(0.0, [], calls=[{"tag": "greet", "level": 1.0}])
    assert mic.vocal_windows()[0].t1 == pytest.approx(0.2, abs=0.001)  # the wav's own length
    bare = M.make_mic(seed=3, engine=None, use_default_library=False)
    assert bare.calls is None


# ── the sound adapter ticks the microphone ───────────────────────────────────────────


def _driver(sounds: list[dict[str, Any]] | None, vocals: list[dict[str, Any]]) -> Any:
    brain = SimpleNamespace(
        felt_head=None,
        sound_out=SimpleNamespace(
            wheee_hold=False,
            theremin_up=False,
            dancing=False,
            beat=SimpleNamespace(active=False, bpm=0.0, lock=0.0),
            chirp=None,
            timing_s=0.0,
            felt_nods=0,
        ),
    )
    lanes = SimpleNamespace(
        brain=brain, senses=SimpleNamespace(sim_s=lambda t_ms: t_ms / 1000.0), sound=None
    )
    world: dict[str, Any] = {}
    if sounds is not None:
        world["neuro"] = {"sounds": sounds}
    return SimpleNamespace(
        lanes=lanes,
        _last_t_ms=0,
        _last_cmd=SimpleNamespace(behaviour="wander", reflex=None),
        _tof_frame=None,
        _world=world,
        _body=None,
        _tick_vocals=vocals,
    )


def test_after_tick_hands_the_microphone_the_worlds_sounds_and_the_calls_of_the_tick(
    tmp_path: Path,
) -> None:
    library = _library(tmp_path)
    engine = V.VoiceEngine(
        duck_seed=7, stream=V.SoundStream(), helper_bin=Path("/nonexistent"), rng_seed=1
    )
    engine.helper = None
    driver = _driver([_event(1, "voice", 0.0, 1.0, utterance="hello")], [])
    mic = M.MicStream(seed=7, library=library, calls=_Bank(0.2))
    driver.lanes.sound = NSL.Sound(brain=driver.lanes.brain, motor=None, engine=engine, mic=mic)
    state = SimpleNamespace(t_ns=0, velocities=None)
    for k in range(25):
        driver._last_t_ms = k * 20
        if k == 12:
            driver._tick_vocals = [{"tag": "chirp", "level": 1.0}]
        elif k == 13:
            driver._tick_vocals = []
        NSL.after_tick(driver, state, [])
    frames = list(mic.frames)
    assert len(frames) == 25 and [f.sim_s for f in frames][:3] == [0.0, 0.02, 0.04]
    assert max(f.level_dbfs for f in frames[:10]) > NOISE_DBFS + 15  # the line, from the world
    assert [i for i, f in enumerate(frames) if f.echo][0] == 12  # the call of that very tick
    assert mic.errors == [] and mic.vocal_windows()[0].tag == "chirp"


def test_a_fault_in_the_microphone_never_stops_the_voice_engine(tmp_path: Path) -> None:
    engine = V.VoiceEngine(
        duck_seed=7, stream=V.SoundStream(), helper_bin=Path("/nonexistent"), rng_seed=1
    )
    engine.helper = None
    driver = _driver([], [])

    class Broken(M.MicStream):
        def tick(self, *args: Any, **kwargs: Any) -> M.MicFrame:
            raise RuntimeError("boom")

    mic = Broken(seed=7)
    driver.lanes.sound = NSL.Sound(brain=driver.lanes.brain, motor=None, engine=engine, mic=mic)
    NSL.after_tick(driver, SimpleNamespace(t_ns=0), [])
    assert engine.ticks == 1 and mic.errors and "boom" in mic.errors[0]


def test_attach_lanes_gives_a_microphone_unless_it_is_switched_off(monkeypatch) -> None:
    monkeypatch.setenv(V.HELPER_ENV, "/nonexistent")

    def lanes() -> Any:
        brain = SimpleNamespace(
            genome=SimpleNamespace(seed=7),
            sound_skills=SimpleNamespace(pre=lambda *a: None, post=lambda *a: None),
            sound_out=None,
        )
        return SimpleNamespace(brain=brain, motor=None, notes=[], sound=None)

    on = NSL.attach_lanes(lanes())
    assert isinstance(on.mic, M.MicStream)
    monkeypatch.setenv(NSL.MIC_ENV, "0")
    assert NSL.attach_lanes(lanes()).mic is None


def test_the_sound_status_says_what_the_microphone_is_doing(tmp_path: Path) -> None:
    mic = M.MicStream(seed=7)
    _run(mic, 0.1, [])
    view = mic.view()
    assert view["rate"] == 16_000 and view["frame_ms"] == 20 and view["frames"] == 5
    assert view["errors"] == [] and view["live"] == {} and view["playing"] == 0
    json.dumps(view)  # (it goes into /api/neuro/sound/status)


def test_a_frame_is_cheap_enough_for_the_robotd_tick(tmp_path: Path) -> None:
    library = _library(tmp_path)
    sounds = [_event(i, "voice", 0.0, 3.0, utterance="hello") for i in range(1, 4)]
    mic = M.MicStream(seed=7, library=library)
    started = time.perf_counter()
    _run(mic, 4.0, sounds)
    per_tick_ms = (time.perf_counter() - started) / 200 * 1000
    assert per_tick_ms < 10.0, per_tick_ms  # (a tick is 20 ms; measured ≈ 0.3 ms alone)
