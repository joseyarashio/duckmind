"""What the world's sounds would look like in the duck's microphone: waveforms for each kind.

The world models its sounds as events (``neuro_world.SoundEvent``: a kind, a time, a level at
1 m, an envelope). The hearing model (``neuro_senses.HearingModel``) turns the events into
levels, onsets and classes; this module gives the same events a *waveform*, so the microphone
stream (:mod:`duckmind.simweb.neuro_mic`) carries them as audio too. Nothing here is
recorded: each sound is a small deterministic recipe (a seeded noise burst, a few damped sines,
a vowel-like babble), made once per event and normalised to
:data:`~duckmind.world.neuro_audio.REF_DBFS`; the stream sets its level from the world's own dB at
the head. They are stand-ins for the sound of a clap or a door, not measurements of one.

* **impulses** (``clap``, ``bang``, ``door``, ``knock``, ``footsteps``): a short burst that decays
  in 25–150 ms, with the room's decay behind it;
* **speech with no script** (``voice``, ``hello``, ``name``): vowel-like babble, a harmonic voice
  through three formants in syllables of about 250 ms (the 4 Hz the hearing model's own speech
  envelope uses), pitch by the event's id (a low, a middle or a high voice);
* **steady** (``tv``, ``music`` without a song, ``noise``): a loop of 8 s — the TV is babble over a
  pink bed, the music a slow chord pad, noise pink noise.

A scripted line (a person saying a stored sentence) never comes here: it is a wav file
(:mod:`duckmind.world.neuro_utterances`).
"""

from __future__ import annotations

import math
import zlib
from typing import Any

import numpy as np

from duckmind.world.neuro_audio import SR, normalise

IMPULSE_KINDS = frozenset({"clap", "bang", "door", "knock", "footsteps"})
SPEECH_KINDS = frozenset({"voice", "name", "hello"})
LOOP_S = 8.0  # a steady sound is a loop of this length
MAX_SPEECH_S = 30.0
SYLLABLE_RATE = 4.0  # syllables a second (the hearing model's speech envelope is 4 Hz too)
VOWELS = (  # F1, F2, F3 (Hz): a, i, u, e, o
    (730.0, 1090.0, 2440.0),
    (270.0, 2290.0, 3010.0),
    (300.0, 870.0, 2240.0),
    (530.0, 1840.0, 2480.0),
    (570.0, 840.0, 2410.0),
)
PITCHES_HZ = (112.0, 190.0, 262.0)  # a low, a middle and a high voice


def rng_for(*key: Any) -> np.random.Generator:
    """A generator that depends only on ``key`` (stable across processes, unlike ``hash``)."""
    return np.random.default_rng(zlib.crc32(repr(key).encode("utf-8")))


def _time(n: int) -> np.ndarray[Any, Any]:
    return np.arange(n, dtype=np.float64) / SR


def _decay(n: int, tau: float, attack: float = 0.0008) -> np.ndarray[Any, Any]:
    t = _time(n)
    env = np.exp(-t / tau)
    if attack > 0:
        env *= np.minimum(1.0, t / attack)
    return env


def _highpass(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.concatenate([[0.0], np.diff(x)])


def _lowpass(x: np.ndarray[Any, Any], width: int) -> np.ndarray[Any, Any]:
    kernel = np.ones(width) / width
    return np.convolve(x, kernel, mode="same")


def _add(out: np.ndarray[Any, Any], at: int, part: np.ndarray[Any, Any], gain: float = 1.0) -> None:
    end = min(len(out), at + len(part))
    if at < end:
        out[at:end] += gain * part[: end - at]


def _edges(x: np.ndarray[Any, Any], seconds: float = 0.02) -> np.ndarray[Any, Any]:
    """``x`` faded in and out at its ends (a loop of babble must not click where it wraps)."""
    k = min(int(seconds * SR), len(x) // 2)
    if k > 0:
        ramp = np.linspace(0.0, 1.0, k)
        x = x.copy()
        x[:k] *= ramp
        x[-k:] *= ramp[::-1]
    return x


def pink(n: int, rng: np.random.Generator) -> np.ndarray[Any, Any]:
    """Pink (1/f) noise of ``n`` samples; it is circular, so it loops without a seam."""
    spectrum = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(spectrum), dtype=np.float64)
    f[0] = 1.0
    return np.fft.irfft(spectrum / np.sqrt(f), n)


def _impulse(kind: str, rng: np.random.Generator) -> np.ndarray[Any, Any]:
    if kind == "clap":
        n = int(0.30 * SR)
        burst = _highpass(rng.standard_normal(n)) * _decay(n, 0.022)
        out = burst.copy()
        _add(out, int(0.009 * SR), burst, 0.6)  # the second hand a few ms later
        return out
    if kind == "bang":
        n = int(0.60 * SR)
        t = _time(n)
        thump = np.sin(2 * math.pi * 55.0 * t) * _decay(n, 0.12)
        crack = _lowpass(rng.standard_normal(n), 3) * _decay(n, 0.05, 0.0004)
        return thump + 0.8 * crack
    if kind == "door":
        n = int(0.70 * SR)
        t = _time(n)
        thud = np.sin(2 * math.pi * 90.0 * t) * _decay(n, 0.09) + 0.5 * np.sin(
            2 * math.pi * 170.0 * t
        ) * _decay(n, 0.06)
        thud += 0.6 * _lowpass(rng.standard_normal(n), 6) * _decay(n, 0.04)
        click = _highpass(rng.standard_normal(int(0.05 * SR))) * _decay(int(0.05 * SR), 0.008)
        _add(thud, int(0.30 * SR), click, 0.5)  # the latch
        return thud
    if kind == "knock":
        n = int(0.18 * SR)
        t = _time(n)
        wood = np.sin(2 * math.pi * 220.0 * t) * _decay(n, 0.03)
        return wood + 0.5 * _highpass(rng.standard_normal(n)) * _decay(n, 0.004)
    # footsteps: a soft low thump
    n = int(0.30 * SR)
    t = _time(n)
    return _lowpass(rng.standard_normal(n), 14) * _decay(n, 0.05) + np.sin(
        2 * math.pi * 80.0 * t
    ) * _decay(n, 0.06)


def _syllable(
    length: int, rng: np.random.Generator, f0: float, vowel: tuple[float, float, float]
) -> np.ndarray[Any, Any]:
    """One vowel-like syllable: harmonics of a gliding pitch weighted by three formants."""
    t = _time(length)
    glide = rng.uniform(-0.08, 0.08)
    f = f0 * (1.0 + glide * (t / max(t[-1], 1e-3) - 0.5) + 0.012 * np.sin(2 * math.pi * 5.0 * t))
    phase = 2.0 * math.pi * np.cumsum(f) / SR
    out = np.zeros(length)
    bandwidths = (90.0, 130.0, 170.0)
    strengths = (1.0, 0.55, 0.28)
    for h in range(1, int(3800.0 / f0) + 1):
        freq = h * f0
        gain = sum(
            s * math.exp(-0.5 * ((freq - fc) / bw) ** 2)
            for fc, bw, s in zip(vowel, bandwidths, strengths, strict=True)
        )
        gain += 0.04 / (1.0 + (freq / 600.0) ** 2)  # the glottal source's own slope
        out += gain * np.sin(h * phase)
    out += 0.02 * _highpass(rng.standard_normal(length))  # breath
    return out * np.hanning(length)


def babble(
    seconds: float, rng: np.random.Generator, f0: float, rate: float = SYLLABLE_RATE
) -> np.ndarray[Any, Any]:
    """Speech-like sound of ``seconds``: syllables at ``rate`` a second, a vowel each, a short
    word gap now and then."""
    n = max(int(seconds * SR), int(0.2 * SR))
    out = np.zeros(n + SR // 2)
    pos = 0
    while pos < n:
        stride = int(SR / rate * rng.uniform(0.85, 1.15))
        length = int(stride * 0.9)
        vowel = VOWELS[int(rng.integers(len(VOWELS)))]
        syllable = _syllable(length, rng, f0 * rng.uniform(0.9, 1.12), vowel)
        _add(out, pos, syllable, rng.uniform(0.6, 1.0))
        pos += stride
        if rng.random() < 0.18:
            pos += int(SR * rng.uniform(0.12, 0.30))  # a gap between words
    return out[:n]


def _pad(rng: np.random.Generator) -> np.ndarray[Any, Any]:
    """A slow chord pad (A minor, F, C, G; 2 s a chord) that loops over :data:`LOOP_S`."""
    chords = (
        (220.0, 261.63, 329.63),
        (174.61, 220.0, 261.63),
        (261.63, 329.63, 392.0),
        (196.0, 246.94, 293.66),
    )
    n = int(LOOP_S * SR)
    tail = SR // 4
    out = np.zeros(n + tail)
    each = n // len(chords)
    for index, chord in enumerate(chords):
        t = _time(each + tail)
        note = sum(np.sin(2 * math.pi * f * t + rng.uniform(0, 6.28)) for f in chord)
        fade = np.minimum(1.0, np.minimum(t / 0.25, (t[-1] - t) / 0.25))
        _add(out, index * each, note * fade)
    out[:tail] += out[n:]  # the last chord fades out over the first one's fade-in: no seam
    return out[:n]


def render(
    kind: str, *, seconds: float, seed: int = 7, ident: int = 0
) -> tuple[np.ndarray[Any, Any], bool]:
    """The waveform of one world sound and whether it loops (a steady sound). ``seconds``: how
    long a speech event lasts (its end is the event's own). Deterministic for the arguments."""
    rng = rng_for("synth", seed, kind, ident)
    if kind in IMPULSE_KINDS:
        return normalise(_impulse(kind, rng)), False
    if kind in SPEECH_KINDS:
        f0 = PITCHES_HZ[ident % len(PITCHES_HZ)] * float(rng.uniform(0.94, 1.06))
        wave = babble(min(max(seconds, 0.3), MAX_SPEECH_S), rng, f0)
        return normalise(wave), False
    if kind == "tv":
        voice = babble(LOOP_S, rng, PITCHES_HZ[1] * 0.9, rate=4.4)
        bed = pink(len(voice), rng)
        bed *= 0.25 * float(np.std(voice) / max(np.std(bed), 1e-9))
        return normalise(_edges(voice + bed)), True
    if kind == "music":
        return normalise(_pad(rng)), True
    return normalise(pink(int(LOOP_S * SR), rng)), True  # noise, and any kind it does not know


__all__ = ["IMPULSE_KINDS", "LOOP_S", "SPEECH_KINDS", "babble", "pink", "render", "rng_for"]
