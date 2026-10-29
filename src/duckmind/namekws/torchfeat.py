"""The reference log-mel features, batched on a GPU for training.

Same window, FFT size, hop and filterbank as :func:`duckmind.namekws.features.log_mel`
(a test pins them together), so a model trained on these frames sees exactly what the
robot's numpy front end produces. Requires torch.
"""

from __future__ import annotations

import torch

from duckmind.namekws.features import FeatureConfig, _window, mel_filterbank


class LogMel:
    def __init__(self, config: FeatureConfig | None = None, device: str | torch.device = "cpu"):
        self.config = config or FeatureConfig()
        self.window = torch.from_numpy(_window(self.config)).to(device)
        self.bank = torch.from_numpy(mel_filterbank(self.config)).to(device)

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: [B, samples] float in [-1, 1] → [B, frames, n_mels] (complete frames only)."""
        config = self.config
        frames = audio.unfold(1, config.win_length, config.hop_length)  # [B, F, win]
        offset = (config.n_fft - config.win_length) // 2
        padded = torch.nn.functional.pad(
            frames, (offset, config.n_fft - config.win_length - offset)
        )
        spectrum = torch.fft.rfft(padded * self.window, n=config.n_fft)
        power = spectrum.real**2 + spectrum.imag**2
        return torch.log(power @ self.bank.T + config.log_floor)
