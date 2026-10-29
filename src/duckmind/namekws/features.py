"""Log-mel features shared by training and the robot, so both see identical inputs.

The numpy implementation here is the reference: training may compute the same
features faster on a GPU, but a test pins it to these values. Parameters travel with
the model in its sidecar JSON, and ``FeatureConfig.digest`` refuses a model trained on
different features.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 16_000
    n_fft: int = 512
    win_length: int = 400  # 25 ms
    hop_length: int = 160  # 10 ms
    n_mels: int = 40
    fmin: float = 20.0
    fmax: float = 7_600.0
    log_floor: float = 1e-6
    window_s: float = 1.0  # what the model sees at once

    @property
    def window_samples(self) -> int:
        return round(self.window_s * self.sample_rate)

    @property
    def window_frames(self) -> int:
        return 1 + (self.window_samples - self.win_length) // self.hop_length

    @property
    def digest(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(config: FeatureConfig) -> np.ndarray:
    """Triangular HTK-style filters, shape (n_mels, n_fft // 2 + 1)."""
    bins = config.n_fft // 2 + 1
    freqs = np.linspace(0.0, config.sample_rate / 2, bins)
    edges = _mel_to_hz(
        np.linspace(_hz_to_mel(config.fmin), _hz_to_mel(config.fmax), config.n_mels + 2)
    )
    bank = np.zeros((config.n_mels, bins), dtype=np.float64)
    for index in range(config.n_mels):
        low, centre, high = edges[index], edges[index + 1], edges[index + 2]
        rising = (freqs - low) / (centre - low)
        falling = (high - freqs) / (high - centre)
        bank[index] = np.maximum(0.0, np.minimum(rising, falling))
    return bank.astype(np.float32)


def _window(config: FeatureConfig) -> np.ndarray:
    # Periodic Hann, centred in the FFT frame when win_length < n_fft.
    n = np.arange(config.win_length)
    hann = 0.5 - 0.5 * np.cos(2.0 * math.pi * n / config.win_length)
    padded = np.zeros(config.n_fft, dtype=np.float64)
    offset = (config.n_fft - config.win_length) // 2
    padded[offset : offset + config.win_length] = hann
    return padded.astype(np.float32)


def as_float(pcm: np.ndarray) -> np.ndarray:
    """int16 or float PCM → float32 in [-1, 1]."""
    array = np.asarray(pcm)
    if array.dtype == np.int16:
        return array.astype(np.float32) / 32768.0
    return array.astype(np.float32)


def log_mel(pcm: np.ndarray, config: FeatureConfig | None = None) -> np.ndarray:
    """Frames × mels for a whole clip (no padding: only complete frames)."""
    config = config or FeatureConfig()
    audio = as_float(pcm)
    if audio.ndim != 1:
        raise ValueError("expected mono PCM")
    if audio.size < config.win_length:
        return np.zeros((0, config.n_mels), dtype=np.float32)
    count = 1 + (audio.size - config.win_length) // config.hop_length
    offset = (config.n_fft - config.win_length) // 2
    windows = np.lib.stride_tricks.sliding_window_view(audio, config.win_length)
    frames = np.zeros((count, config.n_fft), dtype=np.float32)
    frames[:, offset : offset + config.win_length] = windows[:: config.hop_length][:count]
    spectrum = np.fft.rfft(frames * _window(config), n=config.n_fft, axis=1)
    power = (spectrum.real**2 + spectrum.imag**2).astype(np.float32)
    mel = power @ mel_filterbank(config).T
    return np.log(mel + config.log_floor).astype(np.float32)


class StreamingLogMel:
    """Feeds audio in arbitrary chunks and yields the same frames as :func:`log_mel`."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()
        self._pending = np.zeros(0, dtype=np.float32)

    def push(self, pcm: np.ndarray) -> np.ndarray:
        config = self.config
        self._pending = np.concatenate([self._pending, as_float(pcm)])
        if self._pending.size < config.win_length:
            return np.zeros((0, config.n_mels), dtype=np.float32)
        count = 1 + (self._pending.size - config.win_length) // config.hop_length
        used = self._pending[: (count - 1) * config.hop_length + config.win_length]
        frames = log_mel(used, config)
        self._pending = self._pending[count * config.hop_length :]
        return frames
