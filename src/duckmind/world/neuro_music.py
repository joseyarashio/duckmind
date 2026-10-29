"""The world's music: a song with a tempo, as sound in the room (a:sound 2026-09-28).

For behaviour (4) 聽音樂跳舞 the WORLD plays music with a tempo; the duck's microphone model
hears it; its brain finds the beat by itself. This module is the song:

* :class:`Song` — a small procedural pop loop, the same for the same ``(bpm, seed)``: a kick on
  beats 1 and 3, a snare on 2 and 4 (drum-forward, the way dance music is mixed), hi-hats on the
  eighths, a bass on the chord roots (I–V–vi–IV, four bars) and a pentatonic melody. It is
  synthesised sample by sample from absolute time (:meth:`Song.render`: any chunking gives the
  same samples), 48 kHz mono 16-bit — the room's sound the page plays, and what the microphone
  model hears.
* :meth:`Song.hits` — the loud moments (the kicks and snares) between two times, each with how
  far it rises over the song's steady bed (measured from the song's own audio,
  :meth:`Song.levels`). The world makes each one a short ``music`` sound event at the speaker
  (:mod:`duckmind.world.neuro_sound`): the existing microphone model
  (:class:`duckmind.world.neuro_senses.HearingModel`) turns such rises into onsets — late,
  sometimes missed, masked by what else plays — exactly as it does for a clap. The beat times
  themselves never reach the brain.

numpy (the world side); deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

SR = 48_000
BEATS_PER_BAR = 4
TEMPOS = {"slow": 84.0, "mid": 104.0, "fast": 124.0}  # the page's three (a minute)
BPM_RANGE = (60.0, 160.0)
CHORDS = (0, 7, 9, 5)  # I – V – vi – IV (semitones over the key's root), one bar each
PENTATONIC = (0, 2, 4, 7, 9, 12, 14, 16)
MASTER = 0.55  # the mix's gain before the soft clip (peaks ~ −4 dBFS)
LEVEL_WINDOW_S = 0.06  # a hit's loudness: the RMS over this after its start


def _midi_hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def _np() -> Any:
    import numpy as np

    return np


@dataclass(frozen=True)
class Event:
    kind: str  # kick | snare | hat | bass | lead
    t: float  # seconds after the song's start
    length: float
    pitch: float = 0.0  # Hz (bass, lead)
    level: float = 1.0
    index: int = 0  # its own number (its noise is seeded by it)


@dataclass
class Song:
    """A song at ``bpm`` (60–160 a minute) from ``seed``. Times are seconds from its start."""

    bpm: float = TEMPOS["mid"]
    seed: int = 7
    _events: dict[int, list[Event]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.bpm = float(min(BPM_RANGE[1], max(BPM_RANGE[0], self.bpm)))
        self.beat_s = 60.0 / self.bpm
        self.bar_s = BEATS_PER_BAR * self.beat_s
        rng = _Rng(self.seed * 7919 + 17)
        self.root = 55 + rng.below(7)  # G3 … C#4
        # the melody: one rhythm and contour per bar of the four-bar loop
        self.melody: list[list[tuple[int, int, float]]] = []
        for bar in range(4):
            notes: list[tuple[int, int, float]] = []
            step = 0
            while step < 8:
                if rng.chance(0.62):
                    length = 1 if rng.chance(0.65) else 2
                    degree = PENTATONIC[rng.below(len(PENTATONIC))]
                    notes.append((step, length, float(degree)))
                    step += length
                else:
                    step += 1
            self.melody.append(notes)

    # ── the score ──

    def bar_events(self, bar: int) -> list[Event]:
        """Every note of one bar (cached)."""
        if bar in self._events:
            return self._events[bar]
        b = self.beat_s
        t0 = bar * self.bar_s
        out: list[Event] = []
        base = bar * 64
        for beat in range(BEATS_PER_BAR):
            t = t0 + beat * b
            if beat in (0, 2):
                out.append(Event("kick", t, 0.35, level=1.0, index=base + beat))
            else:
                out.append(Event("snare", t, 0.25, level=1.0, index=base + beat))
        if bar % 2 == 1:  # a pickup kick before the next bar, every other bar
            out.append(Event("kick", t0 + 3.5 * b, 0.3, level=0.55, index=base + 9))
        for eighth in range(8):
            level = 0.55 if eighth % 2 else 0.35
            out.append(
                Event("hat", t0 + eighth * b / 2, 0.08, level=level, index=base + 16 + eighth)
            )
        chord = CHORDS[bar % 4]
        for eighth in range(8):
            midi = self.root - 24 + chord + (12 if eighth % 4 == 3 else 0)
            out.append(Event("bass", t0 + eighth * b / 2, b / 2 * 0.9, _midi_hz(midi), 0.8))
        for step, length, degree in self.melody[bar % 4]:
            note = float(self.root + chord) + degree
            while note > self.root + 14:
                note -= 12.0
            out.append(Event("lead", t0 + step * b / 2, length * b / 2 * 0.95, _midi_hz(note)))
        out.sort(key=lambda e: e.t)
        self._events[bar] = out
        if len(self._events) > 64:
            for key in sorted(self._events)[:16]:
                self._events.pop(key, None)
        return out

    def events_between(self, t_a: float, t_b: float, margin: float = 0.4) -> list[Event]:
        first = max(0, int(math.floor((t_a - margin) / self.bar_s)))
        last = max(first, int(math.floor(t_b / self.bar_s)))
        out: list[Event] = []
        for bar in range(first, last + 1):
            out.extend(e for e in self.bar_events(bar) if e.t < t_b and e.t + e.length > t_a)
        return out

    def beats(self, t_a: float, t_b: float) -> list[tuple[float, int]]:
        """The beats between two song times: (time, beat number from the start) — the truth,
        for evaluators and the page's check only."""
        k0 = max(0, int(math.ceil(t_a / self.beat_s - 1e-9)))
        out = []
        k = k0
        while k * self.beat_s < t_b:
            out.append((k * self.beat_s, k))
            k += 1
        return out

    # ── the sound ──

    def render(self, start: int, n: int) -> Any:
        """Samples ``start`` … ``start + n`` of the song (absolute sample numbers from its start)
        as float32 in [−1, 1]. The same samples whatever the chunking."""
        np = _np()
        out = np.zeros(n, dtype=np.float64)
        t_a = start / SR
        t_b = (start + n) / SR
        for event in self.events_between(t_a, t_b):
            first = int(math.ceil(event.t * SR - 1e-9))
            count = int(math.ceil(event.length * SR))
            lo = max(start, first)
            hi = min(start + n, first + count)
            if hi <= lo:
                continue
            u = (np.arange(lo, hi, dtype=np.float64) - first) / SR  # seconds into the note
            out[lo - start : hi - start] += self._voice(event, u, lo - first, hi - first)
        return np.tanh(MASTER * out).astype(np.float32)

    def _voice(self, e: Event, u: Any, i0: int, i1: int) -> Any:
        """One note's samples ``i0`` … ``i1`` from its start (``u``: their times, s)."""
        np = _np()
        count = int(math.ceil(e.length * SR))
        if e.kind == "kick":
            phase = 2.0 * np.pi * (48.0 * u + 95.0 * 0.028 * (1.0 - np.exp(-u / 0.028)))
            body = np.sin(phase) * np.exp(-u / 0.16)
            click = _noise(self.seed, "kick", e.index, count)[i0:i1] * np.exp(-u / 0.004) * 0.35
            return 1.05 * e.level * (body + click)
        if e.kind == "snare":
            noise = _noise(self.seed, "snare", e.index, count)[i0:i1]
            tone = np.sin(2.0 * np.pi * 185.0 * u) * np.exp(-u / 0.045)
            return 0.8 * e.level * (0.75 * noise * np.exp(-u / 0.085) + 0.5 * tone)
        if e.kind == "hat":
            noise = _hiss(self.seed, e.index, count)[i0:i1]
            return 0.16 * e.level * noise * np.exp(-u / 0.022)
        if e.kind == "bass":
            env = np.minimum(1.0, u / 0.008) * np.exp(-u / 0.22)
            w = 2.0 * np.pi * e.pitch * u
            wave = np.sin(w) + 0.33 * np.sin(3 * w) + 0.18 * np.sin(5 * w)
            return 0.22 * e.level * wave * env
        # lead
        env = np.minimum(1.0, u / 0.012) * np.exp(-u / 0.35)
        env *= np.clip((e.length - u) / 0.03, 0.0, 1.0)
        w = 2.0 * np.pi * e.pitch * u
        wave = np.sin(w) + 0.35 * np.sin(2 * w) + 0.12 * np.sin(3 * w)
        return 0.16 * e.level * wave * env

    def pcm16(self, start: int, n: int) -> bytes:
        """The same samples as little-endian 16-bit PCM (robotd's and the page's format)."""
        np = _np()
        x = self.render(start, n)
        return (np.clip(x * 32767.0, -32768.0, 32767.0)).astype("<i2").tobytes()

    # ── what the microphone model hears ──

    def levels(self) -> dict[str, float]:
        """How far a kick and a snare rise over the steady bed (dB), measured on the song's own
        audio: the RMS in the first LEVEL_WINDOW_S of each against the RMS of everything but
        the drums over a bar."""
        return dict(_levels(self.bpm, self.seed))

    def hits(self, t_a: float, t_b: float) -> list[tuple[float, str, float]]:
        """The loud moments between two song times: (time, kick | snare, dB over the bed)."""
        rel = self.levels()
        out = []
        for e in self.events_between(t_a, t_b, margin=0.0):
            if e.kind in ("kick", "snare") and t_a <= e.t < t_b:
                out.append((e.t, e.kind, rel[e.kind] + 20.0 * math.log10(max(0.05, e.level))))
        return out


@lru_cache(maxsize=32)
def _levels(bpm: float, seed: int) -> tuple[tuple[str, float], ...]:
    np = _np()
    song = Song(bpm=bpm, seed=seed)
    n = int(song.bar_s * SR)
    start = int(song.bar_s * SR)  # the second bar (no song start in it)
    full = song.render(start, n).astype(np.float64)
    # the bed: the same bar without its drums
    bed = np.zeros(n)
    for e in song.events_between(start / SR, (start + n) / SR):
        if e.kind in ("bass", "lead", "hat"):
            first = int(math.ceil(e.t * SR - 1e-9))
            lo, hi = max(start, first), min(start + n, first + int(math.ceil(e.length * SR)))
            if hi > lo:
                u = (np.arange(lo, hi) - first) / SR
                bed[lo - start : hi - start] += song._voice(e, u, lo - first, hi - first)
    bed = np.tanh(MASTER * bed)
    bed_rms = float(np.sqrt(np.mean(bed * bed))) or 1e-6
    out = {}
    for kind in ("kick", "snare"):
        rises = []
        for e in song.bar_events(1):
            if e.kind != kind or e.level < 0.99:
                continue
            first = int(math.ceil(e.t * SR - 1e-9)) - start
            window = full[first : first + int(LEVEL_WINDOW_S * SR)]
            rms = float(np.sqrt(np.mean(window * window)))
            rises.append(20.0 * math.log10(max(rms, 1e-6) / bed_rms))
        out[kind] = round(sum(rises) / len(rises), 2) if rises else 0.0
    return tuple(sorted(out.items()))


@lru_cache(maxsize=256)
def _noise(seed: int, kind: str, index: int, n: int) -> Any:
    np = _np()
    rng = np.random.default_rng((seed * 1_000_003 + index * 7 + len(kind)) & 0xFFFFFFFF)
    return rng.standard_normal(max(n, 1)) * 0.5


@lru_cache(maxsize=256)
def _hiss(seed: int, index: int, n: int) -> Any:
    """High-passed noise (two first differences): a hat's hiss."""
    x = _noise(seed, "hat", index, n + 2)
    return (x[2:] - 2.0 * x[1:-1] + x[:-2]) * 0.5


class _Rng:
    """A tiny deterministic generator for the song's composition (splitmix64)."""

    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    def below(self, n: int) -> int:
        return self.next() % max(1, n)

    def chance(self, p: float) -> bool:
        return (self.next() >> 11) / float(1 << 53) < p


__all__ = ["BEATS_PER_BAR", "Event", "SR", "Song", "TEMPOS"]
