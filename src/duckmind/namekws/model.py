"""A causal, streaming temporal CNN small enough to run beside robotd on a Pi Zero 2 W.

Shaped like WeKws' MDTC-Small (multi-scale depthwise temporal convolution): an input
projection, ``stacks`` × ``layers`` depthwise-separable causal blocks with dilations
1, 2, 4, 8…, the stacks' outputs summed, and one logit per keyword per 10 ms frame.
About 30k parameters and ~15k multiply-adds per frame (~1.5M per second of audio).

Streaming is exact: every causal convolution keeps the last ``(kernel − 1) · dilation``
frames of its input in a cache, so feeding audio in chunks gives the same outputs as one
long pass. Normalization learned from training data is baked in: the robot feeds raw
log-mel frames from :mod:`duckmind.namekws.features`. Requires torch.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class CausalBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int) -> None:
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.depthwise = nn.Conv1d(
            channels, channels, kernel, dilation=dilation, groups=channels, bias=False
        )
        self.norm1 = nn.BatchNorm1d(channels)
        self.pointwise = nn.Conv1d(channels, channels, 1, bias=False)
        self.norm2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor, cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [B, C, T]; cache: [B, C, pad] (the previous input frames)."""
        full = torch.cat([cache, x], dim=2)
        next_cache = full[:, :, full.shape[2] - self.pad :]
        y = torch.relu(self.norm1(self.depthwise(full)))
        y = torch.relu(self.norm2(self.pointwise(y)))
        return x + y, next_cache


class NameTCN(nn.Module):
    def __init__(
        self,
        n_mels: int = 40,
        hidden: int = 32,
        stacks: int = 3,
        layers: int = 4,
        kernel: int = 5,
        outputs: int = 1,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.stacks = stacks
        self.layers = layers
        self.register_buffer("mean", torch.zeros(n_mels))
        self.register_buffer("std", torch.ones(n_mels))
        self.project = nn.Sequential(
            nn.Conv1d(n_mels, hidden, 1, bias=False), nn.BatchNorm1d(hidden), nn.ReLU()
        )
        self.blocks = nn.ModuleList(
            CausalBlock(hidden, kernel, 2**layer) for _ in range(stacks) for layer in range(layers)
        )
        self.head = self._new_head(outputs)
        self.pads = [block.pad for block in self.blocks]

    def _new_head(self, outputs: int) -> nn.Conv1d:
        # A keyword is rare: start from "almost never" (p ≈ 2%) with small weights, so
        # early training is stable and posteriors stay calibrated for thresholding.
        head = nn.Conv1d(self.hidden, outputs, 1)
        nn.init.normal_(head.weight, std=0.01)
        nn.init.constant_(head.bias, -4.0)
        return head

    @property
    def cache_frames(self) -> int:
        return sum(self.pads)

    @property
    def receptive_field(self) -> int:
        return 1 + self.cache_frames

    def empty_cache(self, batch: int = 1, device: str | torch.device = "cpu") -> torch.Tensor:
        return torch.zeros(batch, self.hidden, self.cache_frames, device=device)

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean.copy_(mean)
        self.std.copy_(std.clamp_min(1e-3))

    def replace_head(self, outputs: int) -> None:
        """Keep the learned acoustics, start a fresh classifier (pretrain → fine-tune)."""
        self.head = self._new_head(outputs).to(self.mean.device)

    def forward(
        self, features: torch.Tensor, cache: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """features: [B, T, n_mels] raw log-mel → (logits [B, T, outputs], next cache)."""
        batch = features.shape[0]
        if cache is None:
            cache = self.empty_cache(batch, features.device)
        x = ((features - self.mean) / self.std).transpose(1, 2)
        x = self.project(x)
        pieces = torch.split(cache, self.pads, dim=2)
        next_pieces: list[torch.Tensor] = []
        total = torch.zeros_like(x)
        for index, block in enumerate(self.blocks):
            x, piece = block(x, pieces[index])
            next_pieces.append(piece)
            if (index + 1) % self.layers == 0:
                total = total + x  # multi-scale: sum every stack's output
        logits = self.head(total / self.stacks).transpose(1, 2)
        return logits, torch.cat(next_pieces, dim=2)


def export_onnx(model: NameTCN, path: Path, n_mels: int) -> None:
    """Streaming export: (features [1, T, mels], cache) → (logits [1, T, K], next_cache)."""
    model = model.eval().cpu()
    example = torch.zeros(1, 10, n_mels)
    cache = model.empty_cache(1)
    torch.onnx.export(
        model,
        (example, cache),
        str(path),
        input_names=["features", "cache"],
        output_names=["logits", "next_cache"],
        dynamic_axes={"features": {1: "frames"}, "logits": {1: "frames"}},
        opset_version=17,
        dynamo=False,
    )
