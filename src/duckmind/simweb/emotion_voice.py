"""蛋黃's emotion voice: the 17 sounds of Pollen Robotics' ``microduck_emotions`` made again in a duck's OWN voice.

NOTICE: the sound designs, the timings and the voice synth below are from **microduck_emotions**
(https://github.com/pollen-robotics/microduck_emotions, commit 431c73d and its tag ``workshop-2026-10-07``),
Copyright Pollen Robotics, Apache-2.0 (the licence is in ``docs/design/neural-duck/EMOTIONS-2026-10-08.md``'s
data folder and ``third_party/microduck_emotions/LICENSE``). ``Rng`` / ``Personality`` / ``quack`` are a port of its
``quack.py`` (itself an exact port of the robot's voice synth, ``sounds/`` in pollen-robotics/microduck).

**Why.** The originals are made with the robot's voice (seed 4145077059, pitch centre 238 Hz). A simulated duck has
its own voice (:mod:`duckmind.simweb.voice`: ``seed_from_id("duckmind-duck:<seed>")``, 蛋黃 = voice seed 936833334,
pitch centre 415 Hz). The emotions would sound like another duck, so each is rebuilt with the SAME recipe the
workshop used (notes, glides, levels, times) on THIS duck's ``Personality(voice seed)``:

* every frequency written in the workshop's recipes (made for 238 Hz) is scaled by ``personality.pitch_center_hz / 238``;
* the sounds the workshop cut from the robot's own bank (the inquire shock of devastated, the chirps of curious, the
  alarm of play dead) are cut from THIS duck's bank (``voices/v5/seed<voice seed>/``), the same variant letters;
* the timing of every sound is the original's, so the beak keyframes of the emotions (``mouth``) still match it.

**Recipes found and ported** (workshop scripts ``sounds/make_*.py`` at the tag): yes (Y2), yes_fast (Y3), no (N2 / N4 /
N6), mmh (C3), excited (X1), mock (L1), laugh (L4, ``motion/laugh/laugh_v2.py``), angry (S1 / S3), sad (S6 coo voice
200 -> 140 Hz), devastated (D3v2 sobs gentler), play_dead (D1), curious (Q3, bank chirps a and e raised two semitones).
**Approximated** (the tag has the wav and a one-line description but no script): ``defiant`` (two curt 'wak's, 270 and
285 Hz, each falling a third) and ``impatient`` (four short nasal grumbles stepping down, then a rising and falling
huff): notes written from the description and the onsets / lengths measured in the original wav. Every sound is checked
against its original (onset times and lengths of its notes within 25 ms: ``tools/emotion_voice_check.py``).

Pure numpy (the simulator's launch has no scipy): the workshop's ``lfilter`` and Butterworth are done with FFTs.
Rendered on demand into ``voices/emotions/v<VERSION>/seed<voice seed>/<tag>/<tag>_<letter>.wav`` in the data folder (48 kHz
mono 16-bit, the bank's own layout and rate), served by :mod:`duckmind.simweb.voice` as extra tags ``emo_<name>``.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
import threading
import wave
import zlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from duckmind import paths

VERSION = 1  # bump when a recipe changes: the cache path carries it
SR = 48_000
TUNED_SR = 22_050.0
TAU = 2 * math.pi
MASK = (1 << 64) - 1
ST = 2 ** (1 / 12)
ROBOT_SEED = 4145077059  # the workshop's voice
ROBOT_PITCH_HZ = (
    237.9499  # Personality(ROBOT_SEED).pitch_center_hz: what every Hz below was written for
)
TAG_PREFIX = "emo_"

# ---------------------------------------------------------------------------------------- the voice (port)


class Rng:
    """xoshiro256++ seeded by splitmix64, as in ``sounds/src/rng.rs`` (quack.py)."""

    def __init__(self, seed: int):
        x = seed & 0xFFFFFFFF

        def nxt() -> int:
            nonlocal x
            x = (x + 0x9E3779B97F4A7C15) & MASK
            z = x
            z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
            return z ^ (z >> 31)

        self.s = [nxt() for _ in range(4)]

    @staticmethod
    def _rotl(v: int, k: int) -> int:
        return ((v << k) | (v >> (64 - k))) & MASK

    def next_u64(self) -> int:
        s = self.s
        result = (self._rotl((s[0] + s[3]) & MASK, 23) + s[0]) & MASK
        t = (s[1] << 17) & MASK
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = self._rotl(s[3], 45)
        return result

    def random(self) -> float:
        return (self.next_u64() >> 11) * (1.0 / (1 << 53))

    def uniform(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self.random()

    def integers(self, lo: int, hi: int) -> int:
        return lo + int(math.floor(self.random() * (hi - lo)))

    def choice(self, choices: list[float]) -> float:
        return choices[self.integers(0, len(choices))]

    def numpy(self) -> np.random.Generator:
        return np.random.default_rng(self.next_u64())


@dataclass
class Personality:
    """The traits a voice seed gives (quack.py ``Personality``, an exact port of the Rust derivation)."""

    seed: int
    pitch_center_hz: float = 0.0
    register: float = 0.0
    pitch_spread: float = 0.0
    glide_bias: float = 0.0
    brightness: float = 0.0
    tilt: float = 0.0
    nasal: float = 0.0
    harmonic_skew: float = 0.0
    formant_n: int = 1
    formant_gain: float = 0.0
    vibrato_rate_hz: float = 0.0
    vibrato_depth: float = 0.0
    jitter_depth: float = 0.0
    breath: float = 0.0
    quackiness: float = 0.0
    am_rate_hz: float = 0.0
    am_depth: float = 0.0
    warble_hz: float = 0.0
    warble_depth: float = 0.0
    attack_sharpness: float = 0.0
    speed: float = 1.0

    def __post_init__(self) -> None:
        if self.pitch_center_hz:
            return  # already filled (a ``replace`` copy)
        rng = Rng(self.seed)
        register = rng.choice([-1.0, 0.0, 0.0, 1.0]) + rng.uniform(-0.4, 0.4)
        base = rng.uniform(160.0, 380.0)
        self.pitch_center_hz = min(max(base * 2.0 ** (register * 0.45), 110.0), 620.0)
        self.register = register
        self.pitch_spread = rng.uniform(0.4, 1.2)
        self.glide_bias = rng.uniform(-1.0, 1.0)
        self.brightness = rng.uniform(0.05, 0.55)
        self.tilt = rng.uniform(1.4, 2.8)
        self.nasal = rng.uniform(0.1, 1.0)
        self.harmonic_skew = rng.uniform(-1.0, 1.0)
        self.formant_n = rng.integers(1, 6)
        self.formant_gain = rng.uniform(0.0, 1.4)
        self.vibrato_rate_hz = rng.uniform(3.5, 9.5)
        self.vibrato_depth = rng.uniform(0.0, 0.7)
        self.jitter_depth = rng.uniform(0.03, 0.35)
        self.breath = rng.uniform(0.0, 0.30)
        self.quackiness = rng.uniform(0.2, 1.0)
        self.am_rate_hz = rng.uniform(18.0, 55.0)
        self.am_depth = rng.uniform(0.15, 0.70)
        self.warble_hz = rng.uniform(7.0, 18.0)
        self.warble_depth = rng.uniform(0.0, 1.4)
        self.attack_sharpness = rng.uniform(0.0, 1.0)
        self.speed = rng.uniform(0.82, 1.22)

    def variant_rng(self, tag: str, variant: int) -> Rng:
        h = (
            ((self.seed * 1_000_003) & MASK)
            ^ zlib.crc32(tag.encode())
            ^ ((variant * 2_654_435_761) & MASK)
        )
        return Rng(h & 0xFFFFFFFF)

    def harmonics(self) -> np.ndarray:
        n_harm = 7
        w = []
        for n in range(1, n_harm + 1):
            base = 1.0 / n**self.tilt
            high_lift = self.brightness * (n / n_harm) ** 1.5
            nasal = self.nasal * (0.6 if n in (2, 3) else 0.0)
            if self.harmonic_skew >= 0.0:
                skew = self.harmonic_skew * (0.4 if n % 2 == 0 else -0.2)
            else:
                skew = -self.harmonic_skew * (-0.3 if n % 2 == 0 else 0.4)
            formant = self.formant_gain if n == self.formant_n else 0.0
            w.append(max(base + high_lift + nasal + skew + formant * base * 1.5, 0.0))
        w[0] = max(w[0], 0.7)
        return np.array(w)

    def with_(self, **mods: float) -> Personality:
        return replace(self, **mods)


# ---------------------------------------------------------------------------------------- DSP primitives


def t_axis(duration_s: float) -> np.ndarray:
    return np.arange(max(1, int(round(duration_s * SR))), dtype=np.float64) / SR


def lerp(t: np.ndarray, points: list[tuple[float, float]]) -> np.ndarray:
    return np.interp(t, [float(p[0]) for p in points], [float(p[1]) for p in points])


def expdecay(t: np.ndarray, attack_s: float, decay_s: float) -> np.ndarray:
    a = np.clip(t / max(attack_s, 1e-4), 0, 1)
    return a * np.exp(-np.maximum(t - attack_s, 0) / max(decay_s, 1e-4))


def bell(t: np.ndarray, attack_s: float, release_s: float) -> np.ndarray:
    total = t[-1] if len(t) else 0.0
    rel_start = max(total - release_s, 0.0)
    v = np.ones_like(t)
    v = np.where(t < attack_s, t / max(attack_s, 1e-4), v)
    v = np.where(t > rel_start, np.maximum(1.0 - (t - rel_start) / max(release_s, 1e-4), 0.0), v)
    return np.clip(v, 0, 1)


def _fft_filter(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    n = len(x) + len(kernel)
    size = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(kernel, size), size)[: len(x)]


def pink_noise(n: int, nrng: np.random.Generator) -> np.ndarray:
    """quack.py's leaky integrator ``lfilter([1], [1, -a], white)``, as an FFT convolution."""
    a = 0.985 ** (TUNED_SR / SR)
    white = nrng.standard_normal(n)
    pink = _fft_filter(white, a ** np.arange(4096, dtype=np.float64))
    return pink / (np.abs(pink).max() + 1e-9)


def lowpass(x: np.ndarray, fc: float) -> np.ndarray:
    """A second-order Butterworth low-pass (the 'muffled beak'), applied in the frequency domain."""
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1.0 / SR)
    return np.fft.irfft(spectrum / np.sqrt(1.0 + (freqs / fc) ** 4), len(x))


def normalise(x: np.ndarray, peak_dbfs: float = -3.0) -> np.ndarray:
    return x * (10 ** (peak_dbfs / 20) / (np.abs(x).max() + 1e-9))


def db(x: np.ndarray, d: float) -> np.ndarray:
    return x * 10 ** (d / 20)


def voice(p: Personality, t: np.ndarray, freq: np.ndarray, rng: Rng) -> np.ndarray:
    """Harmonic oscillator + vibrato + jitter + AM buzz (quackiness) + breath."""
    phase0 = rng.uniform(0.0, TAU)
    vib = (
        2.0 ** (p.vibrato_depth * np.sin(TAU * p.vibrato_rate_hz * t + phase0) / 12.0)
        if p.vibrato_rate_hz > 0 and p.vibrato_depth > 0
        else np.ones_like(t)
    )
    nrng = rng.numpy()
    if p.jitter_depth > 0:
        raw = nrng.standard_normal(len(t))
        k = max(1, round(64.0 * SR / TUNED_SR))
        jit = 2.0 ** (p.jitter_depth * np.convolve(raw, np.ones(k) / k, mode="same") / 12.0)
    else:
        jit = np.ones_like(t)
    f = np.asarray(freq, dtype=np.float64) * vib * jit
    phase = TAU * np.cumsum(f) / SR
    body = np.zeros_like(phase)
    for i, w in enumerate(p.harmonics()):
        if w:
            body += w * np.sin((i + 1) * phase)
    am_d = p.am_depth * p.quackiness
    if am_d > 0.01:
        body *= 1.0 - am_d * (0.5 + 0.5 * np.sin(TAU * p.am_rate_hz * t))
    if p.breath > 0:
        body += p.breath * pink_noise(len(t), nrng)
    return body


def _click(n: int, nrng: np.random.Generator, length: int) -> np.ndarray:
    out = np.zeros(n)
    m = min(length, n)
    fade = 1.0 - np.arange(m) / m
    out[:m] = nrng.uniform(-1, 1, m) * fade * fade
    return out


EnvSpec = tuple[str, float, float] | np.ndarray


def quack(
    p: Personality,
    dur: float,
    contour: Any,
    env: EnvSpec,
    *,
    mods: Mapping[str, float] | None = None,
    rng: Rng | None = None,
    click_gain: float = 0.0,
    crackle: float = 0.0,
) -> np.ndarray:
    """One vocalisation (quack.py ``quack``): ``contour`` (time_s, hz) points or a per-sample array;
    ``env`` ("expdecay", attack, decay) | ("bell", attack, release) | an array."""
    pp = p.with_(**mods) if mods else p
    rng = rng or pp.variant_rng("emotion", 0)
    t = t_axis(dur)
    freq = (
        np.asarray(contour, dtype=np.float64)
        if isinstance(contour, np.ndarray)
        else lerp(t, contour)
    )
    if isinstance(env, np.ndarray):
        e = env
    elif env[0] == "expdecay":
        e = expdecay(t, env[1], env[2])
    else:
        e = bell(t, env[1], env[2])
    sig = voice(pp, t, freq, rng) * e
    if click_gain > 0:
        sig += click_gain * _click(
            len(t), rng.numpy(), int((0.003 + 0.006 * pp.attack_sharpness) * SR)
        )
    if crackle > 0:
        sig += crackle * rng.numpy().standard_normal(len(t)) * e
    return sig


def varispeed(x: np.ndarray, rate_points: list[tuple[float, float]]) -> np.ndarray:
    n = len(x)
    u = np.arange(n) / max(n - 1, 1)
    rate = np.interp(u, [q[0] for q in rate_points], [q[1] for q in rate_points])
    t_out = np.cumsum(1.0 / rate)
    t_out -= t_out[0]
    out_idx = np.arange(0, t_out[-1])
    return np.interp(np.interp(out_idx, t_out, np.arange(n)), np.arange(n), x)


def fade(x: np.ndarray, a: float = 0.03, r: float = 0.12) -> np.ndarray:
    x = x.copy()
    na, nr = int(a * SR), int(r * SR)
    x[:na] *= np.linspace(0, 1, na)
    x[-nr:] *= np.linspace(1, 0, nr)
    return x


class Track:
    """Sounds laid at times on one timeline."""

    def __init__(self, total_s: float):
        self.buf = np.zeros(int(total_s * SR) + SR)

    def put(self, t: float, sig: np.ndarray) -> None:
        i = int(t * SR)
        n = min(len(sig), len(self.buf) - i)
        self.buf[i : i + n] += sig[:n]


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32768.0
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels()).mean(axis=1)
    return x


def write_wav(path: Path, sig: np.ndarray) -> None:
    pcm = np.clip(sig * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


# ---------------------------------------------------------------------------------------- the recipes


@dataclass
class Voice:
    """One duck's voice for the recipes: its personality, its pitch scale, its bank."""

    p: Personality
    bank_dir: Path | None = None

    @property
    def scale(self) -> float:
        return self.p.pitch_center_hz / ROBOT_PITCH_HZ

    def hz(self, workshop_hz: float) -> float:
        """A frequency the workshop wrote for the robot's voice, in this duck's register."""
        return workshop_hz * self.scale

    def bank(self, tag: str, letter: str) -> np.ndarray | None:
        """A sound of this duck's own bank (None: no bank here, the recipe then synthesises it)."""
        if self.bank_dir is None:
            return None
        path = self.bank_dir / tag / f"{tag}_{letter}.wav"
        try:
            return read_wav(path)
        except (OSError, EOFError, wave.Error):
            return None


# the workshop's voices (overrides of the personality for one sound)
YES = {"quackiness": 0.75, "breath": 0.03, "vibrato_depth": 0.12, "attack_sharpness": 0.6}
NO = {"quackiness": 0.7, "breath": 0.04, "vibrato_depth": 0.15, "attack_sharpness": 0.5}
NASAL = {"quackiness": 0.95, "breath": 0.02, "vibrato_depth": 0.2, "attack_sharpness": 0.3, "nasal": 1.0,
         "am_depth": 0.5, "tilt": 1.6}  # fmt: skip
MUFFLED = {"brightness": 0.05, "nasal": 1.0, "quackiness": 0.6, "breath": 0.02, "vibrato_depth": 0.25,
           "attack_sharpness": 0.2, "tilt": 2.6, "am_depth": 0.35}  # fmt: skip
HAPPY = {
    "quackiness": 0.75,
    "breath": 0.03,
    "vibrato_depth": 0.12,
    "attack_sharpness": 0.65,
    "brightness": 0.35,
}
HA = {"quackiness": 0.95, "breath": 0.02, "vibrato_depth": 0.05, "attack_sharpness": 0.9, "brightness": 0.4,
      "am_depth": 0.45}  # fmt: skip
ANG = {"tilt": 1.25, "brightness": 0.55, "vibrato_rate_hz": 0.0, "vibrato_depth": 0.0, "attack_sharpness": 1.0,
       "breath": 0.03, "quackiness": 1.0, "am_depth": 1.0, "am_rate_hz": 28.0, "jitter_depth": 0.12}  # fmt: skip
GENTLER = {"quackiness": 0.28, "breath": 0.10, "vibrato_rate_hz": 2.7, "vibrato_depth": 0.35,
           "attack_sharpness": 0.0, "tilt": 2.5, "am_depth": 0.12}  # fmt: skip
DEATH = {"quackiness": 0.5, "breath": 0.10, "vibrato_depth": 0.0, "am_depth": 0.18, "attack_sharpness": 0.0,
         "tilt": 2.2}  # fmt: skip
GRUMBLE = {"quackiness": 0.95, "nasal": 1.0, "breath": 0.02, "vibrato_depth": 0.3, "vibrato_rate_hz": 4.5,
           "attack_sharpness": 0.5, "tilt": 1.6, "am_depth": 0.55, "am_rate_hz": 19.0}  # fmt: skip
HUFF = {"quackiness": 0.35, "breath": 0.34, "vibrato_depth": 0.1, "attack_sharpness": 0.1, "tilt": 2.4,
        "am_depth": 0.2}  # fmt: skip


def _coo(p: Personality) -> dict[str, float]:
    """The robot's coo recipe as voices.rs softens the personality for ``coo``."""
    return {
        "breath": max(p.breath, 0.12) + 0.10,
        "quackiness": p.quackiness * 0.25,
        "am_rate_hz": p.am_rate_hz * 0.30,
        "vibrato_rate_hz": p.vibrato_rate_hz * 0.45,
        "vibrato_depth": p.vibrato_depth * 0.7,
    }


def r_yes(v: Voice) -> Track:
    # Y2: one quack, 250 Hz falling 3 semitones over 0.30 s, on the nod's low point (0.45 s)
    dur, f0 = 0.30, 250.0
    f_end = f0 * 2 ** (-3.0 / 12)
    sig = quack(v.p, dur, [(0.0, v.hz(f0 * 0.97)), (0.04, v.hz(f0 * 1.03)), (0.12, v.hz(f0)), (dur, v.hz(f_end))],
                ("expdecay", 0.012, dur * 0.55), mods=YES)  # fmt: skip
    t = Track(1.5)
    t.put(0.45, normalise(sig, -3.0))
    return t


def r_yes_fast(v: Voice) -> Track:
    # Y3: a very short snappy 'wak' (0.18 s) falling a fourth
    dur, f0 = 0.18, 255.0
    f_end = f0 * 2 ** (-5.0 / 12)
    mods = dict(YES, attack_sharpness=0.9, quackiness=0.9)
    sig = quack(v.p, dur, [(0.0, v.hz(f0 * 0.97)), (0.04, v.hz(f0 * 1.03)), (0.12, v.hz(f0)), (dur, v.hz(f_end))],
                ("expdecay", 0.012, dur * 0.4), mods=mods)  # fmt: skip
    t = Track(1.5)
    t.put(0.45, normalise(sig, -3.0))
    return t


def _syl(v: Voice, f0: float, f1: float, dur: float, mods: Mapping[str, float], decay: float,
         level: float = -3.0) -> np.ndarray:  # fmt: skip
    sig = quack(v.p, dur, [(0.0, v.hz(f0 * 0.97)), (0.03, v.hz(f0)), (dur, v.hz(f1))],
                ("expdecay", 0.012, dur * decay), mods=mods)  # fmt: skip
    return normalise(sig, level)


def _no_pair(v: Voice, f0: float, semis: float, d1: float, d2: float, mods: Mapping[str, float],
             tail: float) -> Track:  # fmt: skip
    # 'no' short and nearly flat on the first yaw extreme (0.6 s), 'ah' `semis` lower and falling on the second (1.1 s)
    f2 = f0 * 2 ** (semis / 12)
    t = Track(1.8)
    t.put(0.6, _syl(v, f0, f0 * 2 ** (-0.5 / 12), d1, mods, 0.5))
    t.put(1.1, _syl(v, f2, f2 * 2 ** (-tail / 12), d2, mods, 0.5, level=-4.0))
    return t


def r_no_a(v: Voice) -> Track:
    return _no_pair(v, 250.0, -5, 0.2, 0.32, NO, 3.0)  # N2


def r_no_b(v: Voice) -> Track:
    return _no_pair(v, 235.0, -5, 0.22, 0.36, NASAL, 4.0)  # N4 nasal


def r_no_c(v: Voice) -> Track:
    return _no_pair(v, 250.0, -5, 0.18, 0.45, NO, 5.0)  # N6 drawn-out


def r_mmh(v: Voice) -> Track:
    # C3: a muffled rising 'mmh?' (205 -> 300 Hz, low-passed at 1100 Hz), 0.5 s in
    dur = 0.45
    sig = quack(v.p, dur, [(0.0, v.hz(205)), (0.15, v.hz(215)), (dur, v.hz(300))], ("expdecay", 0.05, dur * 0.8),
                mods=MUFFLED)  # fmt: skip
    t = Track(2.0)
    t.put(0.5, normalise(lowpass(sig, 1100.0 * v.scale), -3.0))
    return t


def r_excited(v: Voice) -> Track:
    # X1: six rising quacks on the six head swings, each higher, shorter and rising more
    t = Track(3.4)
    for i, at in enumerate([0.40, 0.95, 1.45, 1.90, 2.30, 2.65]):
        f0, rise, dur = 225 * ST ** (1.5 * i), 5 + i, 0.2 - 0.012 * i
        sig = quack(v.p, dur, [(0.0, v.hz(f0 * 0.93)), (0.03, v.hz(f0)), (dur, v.hz(f0 * ST**rise))],
                    ("expdecay", 0.008, dur * 0.55), mods=HAPPY)  # fmt: skip
        t.put(at, normalise(sig, -3.0))
    return t


def _ha(
    v: Voice, f0: float, dur: float = 0.12, level: float = -3.0, fall: float = 1.5
) -> np.ndarray:
    sig = quack(v.p, dur, [(0.0, v.hz(f0 * 1.04)), (0.02, v.hz(f0)), (dur, v.hz(f0 / ST**fall))],
                ("expdecay", 0.006, dur * 0.45), mods=HA, click_gain=0.25)  # fmt: skip
    return normalise(sig, level)


def r_mock(v: Voice) -> Track:
    # L1: seven staccato 'ha's, each a little lower, the second phrase lower and softer
    t = Track(2.6)
    for i, at in enumerate([0.35, 0.57, 0.79, 1.01, 1.45, 1.67, 1.89]):
        t.put(at, _ha(v, 300 / ST ** (0.7 * i), level=-3.0 - 0.4 * i - (1.5 if i >= 4 else 0.0)))
    return t


def r_laugh(v: Voice) -> Track:
    # L4: one longer falling 'haaa', then seven short ones getting shorter, softer and further apart
    t = Track(3.0)
    f0, dur = 305.0, 0.38
    sig = quack(v.p, dur, [(0.0, v.hz(f0 * 1.04)), (0.03, v.hz(f0)), (0.2, v.hz(f0 / ST**0.8)), (dur, v.hz(f0 / ST**3.5))],
                ("expdecay", 0.006, dur * 0.55), mods=dict(HA, vibrato_depth=0.25, vibrato_rate_hz=7.0),
                click_gain=0.25)  # fmt: skip
    t.put(0.35, normalise(sig, -3.0))
    for i, at in enumerate([0.90, 1.10, 1.30, 1.50, 1.85, 2.05, 2.30]):
        t.put(
            at, _ha(v, 290 / ST ** (0.5 * i), dur=max(0.07, 0.12 - 0.008 * i), level=-3.5 - 0.9 * i)
        )
    return t


def _angry_rng(v: Voice, i: int) -> Rng:
    return v.p.variant_rng("angry3", i)


def _hit(
    v: Voice, hz: float, dur: float, i: int, *, am_rate: float = 28.0, decay: float | None = None
) -> np.ndarray:
    """One short hard bark: instant attack with a click, crackle, slight fall."""
    return quack(v.p, dur, [(0.0, v.hz(hz)), (0.03, v.hz(hz) * 1.04), (dur, v.hz(hz) * 0.90)],
                 ("expdecay", 0.004, decay or dur * 0.55), mods=dict(ANG, am_rate_hz=am_rate),
                 rng=_angry_rng(v, i), click_gain=0.5, crackle=0.10)  # fmt: skip


def _chopped(v: Voice, hz: float, dur: float, i: int) -> np.ndarray:
    t = t_axis(dur)
    return quack(v.p, dur, [(0.0, v.hz(hz) * 1.02), (0.03, v.hz(hz)), (dur, v.hz(hz) * 0.985)],
                 bell(t, 0.004, 0.02), mods=dict(ANG, am_rate_hz=32.0), rng=_angry_rng(v, i),
                 click_gain=0.5, crackle=0.12)  # fmt: skip


def _growl(v: Voice, t0: float, t1: float, hz0: float, hz1: float, i: int) -> np.ndarray:
    dur = t1 - t0
    t = t_axis(dur)
    contour = lerp(t, [(0.0, v.hz(hz0)), (0.6 * dur, v.hz(hz0) * 1.15), (dur, v.hz(hz1))])
    env = lerp(t, [(0, 0.0), (0.15 * dur, 0.45), (0.7 * dur, 0.7), (dur, 1.0)])
    return 0.5 * quack(v.p, dur, contour, env, mods=dict(ANG, am_rate_hz=24.0, tilt=1.15, jitter_depth=0.2),
                       rng=_angry_rng(v, i), crackle=0.18)  # fmt: skip


ANGRY_BARKS = [0.30, 0.90, 1.50, 2.00]


def r_angry_a(v: Voice) -> Track:
    # S1: four hard barks (300 Hz rising 1.5 semitones each), the last one held and chopped dead
    t = Track(2.6)
    for k, at in enumerate(ANGRY_BARKS):
        hz = 300 * 2 ** (1.5 * k / 12)
        if k == len(ANGRY_BARKS) - 1:
            t.put(at, normalise(_chopped(v, hz, 0.30, 10 + k), -3))
        else:
            t.put(at, normalise(_hit(v, hz, 0.17, 10 + k), -3.5 + 0.5 * k))
    return t


def r_angry_b(v: Voice) -> Track:
    # S3: a growl swelling into the first bark, then one loud 330 Hz bark per snap
    t = Track(2.6)
    t.put(0.0, normalise(_growl(v, 0.0, ANGRY_BARKS[0] - 0.02, 150, 260, 90), -6))
    for k, at in enumerate(ANGRY_BARKS):
        t.put(at, normalise(_hit(v, 330 * 2 ** (k / 12), 0.20, 30 + k, am_rate=30, decay=0.09), -3))
    return t


def r_sad(v: Voice) -> Track:
    # S6: the robot's coo voice gliding 200 -> 140 Hz along the half-cosine of the 2.5 s droop
    d0, d1 = 0.0, 2.5
    start = d0 + 0.05
    dur = (d1 - start) + 0.45
    t = t_axis(dur)
    down = 0.5 - 0.5 * np.cos(np.pi * np.clip((t + start - d0) / (d1 - d0), 0, 1))
    f_start, f_end, drift = 200.0, 140.0, 0.05
    semis_fall = 12 * math.log2(f_end / f_start)
    freq = (
        v.hz(f_start)
        * ST ** (semis_fall * down)
        * (1 + drift * np.sin(np.pi * np.clip(down, 0, 0.5) * 2) ** 2)
    )
    out = Track(3.2)
    out.put(start, quack(v.p, dur, freq, bell(t, 0.26 * dur, 0.30 * dur), mods=_coo(v.p)))
    return out


def r_devastated(v: Voice) -> Track:
    # D3v2 gentler: the inquire shock at the sit (0.3 s), three soft sobs on the head swings (3.3 / 4.3 / 5.3 s), the third
    # dying into the ending; the file starts 0.3 s before the press, so the wav is cut from there (see :func:`render_one`).
    t = Track(9.6)
    shock = v.bank("inquire", "b")
    if shock is None:
        shock = quack(
            v.p,
            0.5,
            [(0.0, v.hz(300)), (0.12, v.hz(520)), (0.5, v.hz(380))],
            ("expdecay", 0.01, 0.25),
        )
    t.put(0.3, normalise(shock, -3.0))
    ex, hold_end, f_hi, drop, step, attack = [3.3, 4.3, 5.3], 6.8, 210.0, 3.0, 1.5, 0.32
    period = ex[1] - ex[0]
    for i, te in enumerate(ex):
        hi = f_hi * ST ** (-step * i)
        lo = hi * ST ** (-drop)
        last = i + 1 == len(ex)
        dur = (hold_end - te) if last else period * 0.9
        contour = (
            [(0.0, v.hz(hi)), (dur, v.hz(lo))]
            if not last
            else [(0.0, v.hz(hi)), (period * 0.9, v.hz(lo)), (dur, v.hz(lo * ST**-3))]
        )
        sig = quack(
            v.p, dur, contour, ("bell", attack, dur * (0.6 if last else 0.45)), mods=GENTLER
        )
        t.put(te, db(normalise(sig, -3.0), -1.5 * i))
    return t


def r_play_dead(v: Voice) -> Track:
    # D1: the alarm scream at the press, silence while it keels over, then a long falling death quack with a dying wobble
    t = Track(7.5)
    alarm = v.bank("alarm", "a")
    if alarm is None:
        alarm = quack(
            v.p,
            0.35,
            [(0.0, v.hz(380)), (0.02, v.hz(430)), (0.35, v.hz(300))],
            ("expdecay", 0.004, 0.14),
        )
    t.put(0.0, normalise(alarm, -3.0))
    dur, f0, f1, wob_hz, wob_semis = 1.8, 260.0, 120.0, 5.0, 1.2
    tt = t_axis(dur)
    u = tt / dur
    base = v.hz(f0) * (f1 / f0) ** (u**0.8)
    decay = np.exp(-tt / (0.55 * dur))
    wob = wob_semis * np.sin(2 * np.pi * wob_hz * (tt - 0.6 * tt**2 / dur)) * decay
    env = bell(tt, 0.12, 0.55 * dur) * (1.0 - 0.35 * u * np.sin(2 * np.pi * 4.0 * tt) ** 2)
    t.put(5.0, normalise(quack(v.p, dur, base * ST**wob, env, mods=DEATH), -6.0))
    return t


def r_curious(v: Voice) -> Track:
    # Q3 up2: two of the bank's rising chirps (a, then e) raised two semitones, on the two head tilts (0.6 / 1.2 s)
    t = Track(3.0)
    for at, letter, level in ((0.6, "a", -4.0), (1.2, "e", -3.0)):
        chirp = v.bank("chirp", letter)
        if chirp is None:
            f = v.hz(300 if letter == "a" else 340)
            chirp = quack(v.p, 0.16, [(0.0, f), (0.16, f * 1.6)], ("expdecay", 0.01, 0.1))
        chirp = fade(varispeed(chirp, [(0.0, 2 ** (2 / 12)), (1.0, 2 ** (2 / 12))]), 0.004, 0.03)
        t.put(at, normalise(chirp, level))
    return t


def r_defiant(v: Voice) -> Track:
    # APPROXIMATED (no script in the workshop): 'says who?', two curt 'wak's, 270 and 285 Hz, each falling a third,
    # one per head turn (0.45 s and 1.15 s, 0.2 s long); the second a little louder (measured in the wav)
    t = Track(2.2)
    mods = dict(YES, attack_sharpness=0.9, quackiness=0.9)
    for at, f0, level in ((0.45, 270.0, -3.4), (1.15, 285.0, -3.0)):
        dur = 0.20
        sig = quack(v.p, dur, [(0.0, v.hz(f0 * 0.97)), (0.04, v.hz(f0 * 1.03)), (0.10, v.hz(f0)),
                               (dur, v.hz(f0 * 2 ** (-4 / 12)))], ("expdecay", 0.008, dur * 0.45), mods=mods)  # fmt: skip
        t.put(at, normalise(sig, level))
    return t


def r_impatient(v: Voice) -> Track:
    # APPROXIMATED (no script in the workshop): 'come on!', four short nasal grumbles stepping down (0.4 / 0.8 / 1.2 /
    # 1.6 s, 0.16 s each), then a rising and falling huff on the beak flick (1.95-2.25 s)
    t = Track(2.6)
    for i, (at, f0) in enumerate(((0.40, 245.0), (0.80, 228.0), (1.20, 212.0), (1.60, 196.0))):
        dur = 0.16
        sig = quack(v.p, dur, [(0.0, v.hz(f0 * 0.98)), (0.03, v.hz(f0)), (dur, v.hz(f0 * 2 ** (-1.5 / 12)))],
                    ("expdecay", 0.010, dur * 0.55), mods=GRUMBLE, click_gain=0.1)  # fmt: skip
        t.put(at, normalise(sig, -3.3 - 0.0 * i))
    dur = 0.30
    tt = t_axis(dur)
    contour = lerp(tt, [(0.0, v.hz(200)), (0.45 * dur, v.hz(300)), (dur, v.hz(205))])
    t.put(1.95, normalise(quack(v.p, dur, contour, bell(tt, 0.06, 0.12), mods=HUFF), -3.0))
    return t


@dataclass(frozen=True)
class Recipe:
    """A sound file: the tag it belongs to (``emo_<emotion>``), its variant letter, how it is made."""

    tag: str
    letter: str
    make: Callable[[Voice], Track]
    cut_from: float = (
        0.0  # seconds dropped from the start (devastated: the file starts 0.3 s before the press)
    )
    approximated: bool = False


RECIPES: tuple[Recipe, ...] = (
    Recipe("sad", "a", r_sad),
    Recipe("devastated", "a", r_devastated, cut_from=0.3),
    Recipe("curious", "a", r_curious),
    Recipe("mmh", "a", r_mmh),
    Recipe("yes", "a", r_yes),
    Recipe("yes_fast", "a", r_yes_fast),
    Recipe("no", "a", r_no_a),
    Recipe("no", "b", r_no_b),
    Recipe("no", "c", r_no_c),
    Recipe("defiant", "a", r_defiant, approximated=True),
    Recipe("angry", "a", r_angry_a),
    Recipe("angry", "b", r_angry_b),
    Recipe("mock", "a", r_mock),
    Recipe("excited", "a", r_excited),
    Recipe("impatient", "a", r_impatient, approximated=True),
    Recipe("laugh", "a", r_laugh),
    Recipe("play_dead", "a", r_play_dead),
)
EMOTIONS: tuple[str, ...] = tuple(dict.fromkeys(r.tag for r in RECIPES))
VARIANTS: dict[str, tuple[str, ...]] = {
    name: tuple(r.letter for r in RECIPES if r.tag == name) for name in EMOTIONS
}


def render_one(recipe: Recipe, v: Voice) -> np.ndarray:
    """The recipe's sound: cut at ``cut_from``, the trailing silence trimmed (50 ms kept), peak -3 dBFS (the bank's)."""
    sig = recipe.make(v).buf
    sig = sig[int(recipe.cut_from * SR) :]
    loud = np.flatnonzero(np.abs(sig) > 1e-4 * (np.abs(sig).max() + 1e-12))
    sig = sig[: (loud[-1] if len(loud) else 0) + int(0.05 * SR)]
    return normalise(sig, -3.0)


def sound_name(tag: str, letter: str) -> str:
    return f"{TAG_PREFIX}{tag}_{letter}.wav"


# ---------------------------------------------------------------------------------------- the cache


DEFAULT_ROOT = paths.data_dir() / "voices" / "emotions"
MARKER = ".seed"


def cache_dir(voice_seed: int, root: Path | None = DEFAULT_ROOT) -> Path | None:
    return None if root is None else root / f"v{VERSION}" / f"seed{int(voice_seed)}"


def files() -> tuple[tuple[str, str], ...]:
    """(tag, file name) of every sound: ``emo_<emotion>`` and ``emo_<emotion>_<letter>.wav``."""
    return tuple((TAG_PREFIX + r.tag, sound_name(r.tag, r.letter)) for r in RECIPES)


class EmotionBank:
    """The emotion sounds of every duck on this machine, rendered once per voice seed (in the background) and served."""

    def __init__(self, root: Path | None = DEFAULT_ROOT, bank_root: Path | None = None) -> None:
        self.root = root
        self.bank_root = (
            bank_root  # voices/v5 (the duck's own bank is read from it), None: synthesise
        )
        self._lock = threading.Lock()
        self._threads: dict[int, threading.Thread] = {}
        self._ready: set[int] = set()
        self.renders = 0

    def _complete(self, voice_seed: int) -> bool:
        folder = cache_dir(voice_seed, self.root)
        if folder is None:
            return False
        try:
            if (folder / MARKER).read_text().strip() != f"{int(voice_seed)}:{VERSION}":
                return False
            return all((folder / tag / name).stat().st_size > 44 for tag, name in files())
        except OSError:
            return False

    def ready(self, voice_seed: int) -> bool:
        with self._lock:
            if voice_seed in self._ready:
                return True
        if not self._complete(voice_seed):
            return False
        with self._lock:
            self._ready.add(voice_seed)
        return True

    def render(self, voice_seed: int, bank_dir: Path | None = None) -> None:
        """Render a voice's 17 sounds now (blocking) unless they are there; complete or absent, never half there."""
        folder = cache_dir(voice_seed, self.root)
        if folder is None or self.ready(voice_seed):
            return
        v = Voice(Personality(int(voice_seed)), bank_dir)
        folder.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".seed{voice_seed}.", suffix=".new", dir=folder.parent)
        )
        try:
            staging.chmod(0o755)
            for recipe in RECIPES:
                out = staging / (TAG_PREFIX + recipe.tag)
                out.mkdir(exist_ok=True)
                write_wav(out / sound_name(recipe.tag, recipe.letter), render_one(recipe, v))
            (staging / MARKER).write_text(f"{int(voice_seed)}:{VERSION}")
            if folder.exists():
                shutil.rmtree(folder, ignore_errors=True)
            staging.rename(folder)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        with self._lock:
            self._ready.add(voice_seed)
            self.renders += 1

    def prepare(self, voice_seed: int, bank_dir: Path | None) -> str:
        """Make sure the sounds are on their way: ``ready`` | ``rendering`` | ``unavailable``."""
        if self.ready(voice_seed):
            return "ready"
        if self.root is None:
            return "unavailable"
        with self._lock:
            thread = self._threads.get(voice_seed)
            if thread is not None and thread.is_alive():
                return "rendering"
            thread = threading.Thread(target=self._render_quietly, args=(voice_seed, bank_dir),
                                      name=f"emotion-voice-{voice_seed}", daemon=True)  # fmt: skip
            self._threads[voice_seed] = thread
            thread.start()
        return "rendering"

    def _render_quietly(self, voice_seed: int, bank_dir: Path | None) -> None:
        try:
            self.render(voice_seed, bank_dir)
        except Exception:  # (the page then has no emotion sounds; the beak and the head still play)
            return

    def tags(self, voice_seed: int, url_base: str) -> dict[str, Any]:
        """The manifest entries of the emotion tags: ``{"emo_yes": {"variants": {"a": url}}, …}``."""
        out: dict[str, Any] = {}
        for recipe in RECIPES:
            tag = TAG_PREFIX + recipe.tag
            entry = out.setdefault(tag, {"variants": {}})
            entry["variants"][recipe.letter] = (
                f"{url_base}/{tag}/{sound_name(recipe.tag, recipe.letter)}"
            )
        return out

    def read(self, voice_seed: int, tag: str, name: str) -> bytes | None:
        """One served file (only a name this module writes, only a ready seed)."""
        if (tag, name) not in files() or not self.ready(voice_seed):
            return None
        folder = cache_dir(voice_seed, self.root)
        if folder is None:
            return None
        try:
            return (folder / tag / name).read_bytes()
        except OSError:
            with self._lock:
                self._ready.discard(voice_seed)
            return None


def bank_dir_of(voices_root: Path, bank_version: int, voice_seed: int) -> Path:
    return voices_root / f"v{bank_version}" / f"seed{int(voice_seed)}"


def main(argv: list[str] | None = None) -> None:
    """``python -m duckmind.simweb.emotion_voice <voice seed> [--out DIR]``: render a voice's emotion sounds."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Render a duck's emotion sounds (microduck_emotions, in its own voice)."
    )
    parser.add_argument("voice_seed", type=int, help="the duck's VOICE seed (蛋黃: 936833334)")
    parser.add_argument(
        "--bank",
        type=Path,
        default=None,
        help="the duck's bank folder (voices/v5/seed<voice seed>)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write the wavs flat into this folder instead of the cache",
    )
    args = parser.parse_args(argv)
    v = Voice(Personality(args.voice_seed), args.bank)
    print(
        f"voice seed {args.voice_seed}: pitch centre {v.p.pitch_center_hz:.1f} Hz, scale {v.scale:.3f}, bank {args.bank}"
    )
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        for recipe in RECIPES:
            write_wav(args.out / f"{recipe.tag}_{recipe.letter}.wav", render_one(recipe, v))
        print("wrote", len(RECIPES), "files to", args.out)
        return
    bank = EmotionBank()
    bank.render(args.voice_seed, args.bank)
    print(
        "cache:",
        cache_dir(args.voice_seed, bank.root),
        "ready" if bank.ready(args.voice_seed) else "NOT ready",
    )


if __name__ == "__main__":
    os.umask(0o022)
    main()
