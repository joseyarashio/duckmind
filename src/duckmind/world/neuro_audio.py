"""Audio primitives of the live-voice plumbing: 16-bit mono PCM, wav files, resampling, levels.

The duck's microphone stream is 16 kHz mono 16-bit, 20 ms frames (320 samples = 640 bytes): the
format Gemini Live takes (``audio/pcm;rate=16000``) and robot-bridge's ``IN_CHUNK``. Everything
here is numpy only (no scipy, no soundfile), deterministic and free of any clock.

Levels. The simulator's sound model speaks in dB (A-weighted "fast", at the duck's head: a quiet
flat 30 dB, a voice 62 dB at 1 m, :mod:`duckmind.world.neuro_world`). The microphone model maps
that to PCM with one number, :data:`FULL_SCALE_DB`: an RMS of 0 dBFS is :data:`FULL_SCALE_DB` dB at
the head, so a quiet room is -62 dBFS (about 26 LSB RMS, what a MEMS microphone's own noise is)
and a voice at 1 m -30 dBFS (a usual level for speech into a speech recogniser).
"""

from __future__ import annotations

import math
import wave
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np

SR = 16_000  # the duck's microphone stream
FRAME_S = 0.02  # one frame of the stream
FRAME = int(SR * FRAME_S)  # 320 samples
FRAME_BYTES = 2 * FRAME  # 640 bytes of little-endian 16-bit
FULL_SCALE_DB = 92.0  # the sound level at the head that is 0 dBFS RMS (see the module docstring)
REF_DBFS = -26.0  # the active RMS every stored speech waveform is normalised to
FLOOR_DBFS = -120.0
ACTIVE_WINDOW_DB = 25.0  # frames within this of the loudest count as "active" for a level


def to_float(pcm: bytes | np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Little-endian 16-bit PCM bytes (or an int16 array) as float32 in [-1, 1)."""
    if isinstance(pcm, (bytes, bytearray, memoryview)):
        return np.frombuffer(bytes(pcm), dtype="<i2").astype(np.float32) / 32768.0
    return np.asarray(pcm).astype(np.float32) / 32768.0


def to_pcm16(x: np.ndarray[Any, Any]) -> bytes:
    """Float samples (1.0 = full scale) as little-endian 16-bit PCM, clipped, rounded."""
    return (
        np.clip(np.rint(np.asarray(x, dtype=np.float64) * 32768.0), -32768, 32767)
        .astype("<i2")
        .tobytes()
    )


def rms_dbfs(x: np.ndarray[Any, Any]) -> float:
    """The RMS level of float samples in dBFS (an empty or silent signal: :data:`FLOOR_DBFS`)."""
    if len(x) == 0:
        return FLOOR_DBFS
    power = float(np.mean(np.square(np.asarray(x, dtype=np.float64))))
    return 10.0 * math.log10(power) if power > 1e-12 else FLOOR_DBFS


def active_rms_dbfs(x: np.ndarray[Any, Any], *, frame: int = FRAME) -> float:
    """The level of the loud part: the mean power of the 20 ms frames within
    :data:`ACTIVE_WINDOW_DB` of the loudest one (pauses and decays do not pull it down)."""
    n = len(x) // frame
    if n == 0:
        return rms_dbfs(x)
    frames = np.asarray(x[: n * frame], dtype=np.float64).reshape(n, frame)
    power = np.mean(frames * frames, axis=1)
    top = float(power.max())
    if top <= 1e-12:
        return FLOOR_DBFS
    keep = power >= top * 10.0 ** (-ACTIVE_WINDOW_DB / 10.0)
    return 10.0 * math.log10(float(power[keep].mean()))


def normalise(x: np.ndarray[Any, Any], target_dbfs: float = REF_DBFS) -> np.ndarray[Any, Any]:
    """``x`` scaled so its active level is ``target_dbfs`` (a silent signal stays silent)."""
    level = active_rms_dbfs(x)
    if level <= FLOOR_DBFS:
        return np.asarray(x, dtype=np.float32)
    return (np.asarray(x, dtype=np.float32) * 10.0 ** ((target_dbfs - level) / 20.0)).astype(
        np.float32
    )


# ── wav files ─────────────────────────────────────────────────────────────────────


def read_wav(path: str | Path) -> tuple[np.ndarray[Any, Any], int]:
    """A mono 16-bit wav file as (int16 samples, sample rate). Anything else is a ValueError."""
    with wave.open(str(path), "rb") as handle:
        channels, width, rate = handle.getnchannels(), handle.getsampwidth(), handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if channels != 1 or width != 2:
        raise ValueError(f"{path}: need mono 16-bit, got {channels} channel(s) of {8 * width}-bit")
    return np.frombuffer(frames, dtype="<i2").copy(), int(rate)


def write_wav(path: str | Path, pcm: np.ndarray[Any, Any] | bytes, rate: int = SR) -> None:
    """Little-endian 16-bit mono PCM (an int16 array or bytes) as a wav file."""
    data = pcm if isinstance(pcm, (bytes, bytearray)) else np.asarray(pcm, dtype="<i2").tobytes()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(rate))
        handle.writeframes(bytes(data))


def wav_payload(raw: bytes) -> tuple[bytes, int] | None:
    """The data chunk of a 16-bit mono wav held in memory and its sample rate (None: not one)."""
    if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        return None
    pos, rate = 12, 0
    while pos + 8 <= len(raw):
        size = int.from_bytes(raw[pos + 4 : pos + 8], "little")
        body = pos + 8
        tag = raw[pos : pos + 4]
        if tag == b"fmt " and body + 8 <= len(raw):
            rate = int.from_bytes(raw[body + 4 : body + 8], "little")
        elif tag == b"data" and rate > 0:
            return raw[body : min(body + size, len(raw))], rate
        pos = body + size + (size & 1)
    return None


# ── resampling ────────────────────────────────────────────────────────────────────


def _kernel(cutoff: float, half: int) -> np.ndarray[Any, Any]:
    """A Kaiser-windowed sinc low-pass: ``cutoff`` in cycles per sample (< 0.5), ``2*half+1``
    taps, unity gain at DC."""
    k = np.arange(-half, half + 1, dtype=np.float64)
    taps = 2.0 * cutoff * np.sinc(2.0 * cutoff * k) * np.kaiser(2 * half + 1, 8.0)
    return (taps / taps.sum()).astype(np.float64)


def resample(x: np.ndarray[Any, Any], src: int, dst: int) -> np.ndarray[Any, Any]:
    """``x`` (float samples at ``src`` Hz) at ``dst`` Hz: polyphase, band-limited to 0.46 of the
    lower of the two rates (a 24 kHz line to 16 kHz keeps everything under 7.4 kHz)."""
    x = np.asarray(x, dtype=np.float64)
    if src == dst or len(x) == 0:
        return x.astype(np.float32)
    g = gcd(int(src), int(dst))
    up, down = int(dst) // g, int(src) // g
    stuffed = np.zeros(len(x) * up) if up > 1 else x
    if up > 1:
        stuffed[::up] = x * up
    rate_up = src * up
    half = 12 * max(up, down)
    taps = _kernel(0.46 * min(src, dst) / rate_up, half)
    filtered = np.convolve(stuffed, taps, mode="full")[half : half + len(stuffed)]
    out = filtered[::down]
    return out[: int(round(len(x) * dst / src))].astype(np.float32)


class StreamDecimator:
    """A stateful integer decimator for chunks that arrive one after another (the 48 kHz plane
    of the duck's own voice into the 16 kHz microphone: ``factor`` 3). The samples keep their
    phase across chunks whatever the chunk sizes; the filter's delay is ``half`` input samples
    (0.4 ms at 48 kHz, nothing for a microphone)."""

    def __init__(self, factor: int = 3, half: int = 18) -> None:
        self.factor = int(factor)
        self.taps = _kernel(0.46 / self.factor, half)
        self.history = np.zeros(len(self.taps) - 1)
        self.consumed = 0  # input samples seen so far

    def reset(self) -> None:
        self.history = np.zeros(len(self.taps) - 1)
        self.consumed = 0

    def process(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        x = np.asarray(x, dtype=np.float64)
        if len(x) == 0:
            return np.zeros(0, dtype=np.float32)
        joined = np.concatenate([self.history, x])
        filtered = np.convolve(joined, self.taps, mode="valid")  # len(x) samples
        start = (-self.consumed) % self.factor
        self.history = joined[len(joined) - (len(self.taps) - 1) :]
        self.consumed += len(x)
        return filtered[start :: self.factor].astype(np.float32)


__all__ = [
    "ACTIVE_WINDOW_DB",
    "FLOOR_DBFS",
    "FRAME",
    "FRAME_BYTES",
    "FRAME_S",
    "FULL_SCALE_DB",
    "REF_DBFS",
    "SR",
    "StreamDecimator",
    "active_rms_dbfs",
    "normalise",
    "read_wav",
    "resample",
    "rms_dbfs",
    "to_float",
    "to_pcm16",
    "wav_payload",
    "write_wav",
]
