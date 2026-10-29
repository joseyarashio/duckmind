"""Live voice, LV0: the audio primitives (resampling, levels, wav files), the waveforms of the
world's sounds, and the scripted-line library. numpy only: no MuJoCo, no network."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from duckmind.world import neuro_audio as A
from duckmind.world import neuro_sound_synth as S
from duckmind.world.neuro_utterances import MANIFEST, UtteranceLibrary


def _tone(freq: float, seconds: float, rate: int, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (amp * np.sin(2 * math.pi * freq * t)).astype(np.float32)


def _bin_db(x: np.ndarray, rate: int, freq: float) -> float:
    """The level of one frequency in ``x`` (a Hann-windowed DFT bin), dB."""
    w = np.hanning(len(x))
    spectrum = np.abs(np.fft.rfft(x * w)) / (w.sum() / 2)
    k = int(round(freq * len(x) / rate))
    return 20.0 * math.log10(max(spectrum[k - 1 : k + 2].max(), 1e-12))


# ── the format ───────────────────────────────────────────────────────────────────────


def test_the_stream_format_is_gemini_lives_16k_mono_20ms() -> None:
    assert A.SR == 16_000 and A.FRAME == 320 and A.FRAME_BYTES == 640
    assert A.FRAME_S == 0.02


def test_pcm_round_trip_and_clipping() -> None:
    x = np.array([0.0, 0.5, -0.5, 1.5, -1.5], dtype=np.float32)
    pcm = A.to_pcm16(x)
    assert len(pcm) == 10
    back = A.to_float(pcm)
    assert back[1] == pytest.approx(0.5, abs=1 / 32768) and back[2] == pytest.approx(-0.5, abs=1e-4)
    assert back[3] == pytest.approx(32767 / 32768) and back[4] == -1.0  # clipped, not wrapped


def test_wav_round_trip_and_the_wrong_kinds_are_refused(tmp_path: Path) -> None:
    pcm = (_tone(440, 0.1, 16_000) * 32767).astype("<i2")
    path = tmp_path / "a.wav"
    A.write_wav(path, pcm, 16_000)
    back, rate = A.read_wav(path)
    assert rate == 16_000 and np.array_equal(back, pcm)
    import wave

    stereo = tmp_path / "s.wav"
    with wave.open(str(stereo), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(bytes(400))
    with pytest.raises(ValueError):
        A.read_wav(stereo)
    assert A.wav_payload(path.read_bytes()) == (pcm.tobytes(), 16_000)
    assert A.wav_payload(b"not a wav at all") is None


# ── levels ───────────────────────────────────────────────────────────────────────────


def test_rms_and_the_active_level_ignore_pauses() -> None:
    tone = _tone(300, 0.5, 16_000, 0.1)
    assert A.rms_dbfs(tone) == pytest.approx(-23.0, abs=0.1)  # 0.1 / √2 → −23 dBFS
    gappy = np.concatenate([tone, np.zeros(16_000), tone])
    assert A.rms_dbfs(gappy) < -26.0  # the pause pulls the plain RMS down
    assert A.active_rms_dbfs(gappy) == pytest.approx(-23.0, abs=0.2)  # …not the active level
    assert (
        A.rms_dbfs(np.zeros(100)) == A.FLOOR_DBFS and A.active_rms_dbfs(np.zeros(5)) == A.FLOOR_DBFS
    )


def test_normalise_sets_the_active_level() -> None:
    x = np.concatenate([_tone(250, 0.4, 16_000, 0.02), np.zeros(8000)])
    y = A.normalise(x, -26.0)
    assert A.active_rms_dbfs(y) == pytest.approx(-26.0, abs=0.05)
    assert not A.normalise(np.zeros(100)).any()


# ── resampling ───────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("src", [48_000, 24_000, 8_000, 44_100])
def test_resampling_keeps_a_tone_and_its_level(src: int) -> None:
    x = _tone(1000.0, 1.0, src)
    y = A.resample(x, src, 16_000)
    assert len(y) == 16_000
    mid = y[2000:-2000]
    assert A.rms_dbfs(mid) == pytest.approx(A.rms_dbfs(x), abs=0.3)
    # the tone is where it was (a 1 kHz bin, nothing else big)
    assert _bin_db(mid, 16_000, 1000.0) > -10.0


def test_resampling_removes_what_the_lower_rate_cannot_hold() -> None:
    x = _tone(10_000.0, 1.0, 48_000)  # above the 8 kHz Nyquist of 16 kHz
    y = A.resample(x, 48_000, 16_000)
    assert A.rms_dbfs(y[2000:-2000]) < A.rms_dbfs(x) - 50.0


def test_the_stream_decimator_matches_the_whole_signal_whatever_the_chunks() -> None:
    x = _tone(700.0, 0.6, 48_000) + _tone(2300.0, 0.6, 48_000, 0.2)
    whole = A.StreamDecimator(3).process(x)
    for sizes in ((960,), (7, 1000, 313), (1, 2, 3, 959)):
        dec = A.StreamDecimator(3)
        parts, pos, k = [], 0, 0
        while pos < len(x):
            n = sizes[k % len(sizes)]
            parts.append(dec.process(x[pos : pos + n]))
            pos, k = pos + n, k + 1
        out = np.concatenate(parts)
        assert np.allclose(out, whole, atol=1e-6), sizes
    assert len(whole) == len(x) // 3
    assert A.rms_dbfs(whole[100:-100]) == pytest.approx(A.rms_dbfs(x), abs=0.4)


# ── the waveforms of the world's sounds ─────────────────────────────────────────────


@pytest.mark.parametrize("kind", sorted(S.IMPULSE_KINDS))
def test_an_impulse_is_a_burst_that_decays(kind: str) -> None:
    wave, loop = S.render(kind, seconds=0.1, seed=7, ident=3)
    assert not loop and 0.1 * A.SR < len(wave) < 1.0 * A.SR
    assert np.isfinite(wave).all()
    assert A.active_rms_dbfs(wave) == pytest.approx(A.REF_DBFS, abs=0.05)
    n = len(wave) // A.FRAME
    power = np.square(wave[: n * A.FRAME]).reshape(n, A.FRAME).mean(axis=1)
    assert power[0] > 20 * power[-1], "the burst dies away, it does not sustain"


def test_unscripted_speech_has_syllables_at_about_four_a_second() -> None:
    wave, loop = S.render("voice", seconds=3.0, seed=7, ident=5)
    assert not loop and len(wave) == 3 * A.SR
    envelope = np.abs(wave).reshape(-1, 160).max(axis=1)  # 10 ms
    envelope = envelope - envelope.mean()
    spectrum = np.abs(np.fft.rfft(envelope))
    freqs = np.fft.rfftfreq(len(envelope), 0.01)
    band = (freqs > 1.5) & (freqs < 8.0)
    peak = freqs[band][np.argmax(spectrum[band])]
    assert 3.0 <= peak <= 5.5, peak


def test_voices_differ_by_event_and_sounds_repeat_for_the_same_event() -> None:
    a, _ = S.render("voice", seconds=1.0, seed=7, ident=0)
    b, _ = S.render("voice", seconds=1.0, seed=7, ident=1)
    again, _ = S.render("voice", seconds=1.0, seed=7, ident=0)
    assert np.array_equal(a, again) and not np.array_equal(a, b)

    def f0(
        x: np.ndarray,
    ) -> float:  # the median pitch over the loud 64 ms windows (autocorrelation)
        found = []
        for start in range(0, len(x) - 1024, 1024):
            w = x[start : start + 1024].astype(np.float64)
            if np.mean(w * w) < 0.25 * np.mean(x.astype(np.float64) ** 2):
                continue
            ac = np.fft.irfft(np.abs(np.fft.rfft(w, 2048)) ** 2)[:600]
            lo, hi = int(A.SR / 400), int(A.SR / 70)
            found.append(A.SR / (lo + int(np.argmax(ac[lo:hi]))))
        return float(np.median(found))

    low_voice, _ = S.render("voice", seconds=1.5, seed=7, ident=0)
    high_voice, _ = S.render("voice", seconds=1.5, seed=7, ident=2)
    assert f0(low_voice) < 160 < f0(high_voice)


@pytest.mark.parametrize("kind", ["tv", "music", "noise", "something_unknown"])
def test_steady_sounds_are_loops_with_no_click_at_the_seam(kind: str) -> None:
    wave, loop = S.render(kind, seconds=1.0, seed=7, ident=9)
    assert loop and len(wave) == int(S.LOOP_S * A.SR)
    assert A.active_rms_dbfs(wave) == pytest.approx(A.REF_DBFS, abs=0.05)
    seam = abs(float(wave[0]) - float(wave[-1]))
    typical = float(np.abs(np.diff(wave)).mean()) * 8
    assert seam < max(typical, 0.05), (seam, typical)


# ── the scripted-line library ────────────────────────────────────────────────────────


def _make_library(root: Path, lines: dict[str, np.ndarray]) -> None:
    entries = []
    for name, pcm in lines.items():
        A.write_wav(root / f"{name}.wav", pcm, 16_000)
        entries.append(
            {
                "id": name,
                "text": f"line {name}",
                "speaker": "mom",
                "persona": "partner",
                "intent": "greet" if name == "hello" else "call_name",
                "name": "蛋黃" if name == "call" else None,
                "file": f"{name}.wav",
                "duration_s": len(pcm) / 16_000,
                "samples": len(pcm),
                "active_rms_dbfs": -26.0,
            }
        )
    manifest = {"version": 1, "sample_rate": 16_000, "speakers": {}, "utterances": entries}
    (root / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")


def test_the_library_reads_its_manifest_and_plays_the_wavs(tmp_path: Path) -> None:
    pcm = (_tone(300, 0.5, 16_000) * 20000).astype("<i2")
    _make_library(tmp_path, {"call": pcm, "hello": pcm[:4000]})
    library = UtteranceLibrary(tmp_path)
    assert library.ids() == ["call", "hello"]
    line = library.get("call")
    assert line.name == "蛋黃" and line.duration_s == pytest.approx(0.5)
    assert library.get("hello").name is None
    assert np.array_equal(library.pcm("call"), pcm)
    assert library.pcm("call") is library.pcm("call"), "read once"
    with pytest.raises(KeyError):
        library.get("nope")
    assert line.view()["text"] == "line call"


def test_a_library_of_the_wrong_rate_or_folder_is_refused(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / MANIFEST).write_text(
        json.dumps({"version": 1, "sample_rate": 24000, "utterances": []}), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        UtteranceLibrary(tmp_path)
    monkeypatch.setenv("DUCKMIND_VOICE_FIXTURES", str(tmp_path / "missing"))
    assert UtteranceLibrary.default() is None
