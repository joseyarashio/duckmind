"""Streaming name detector: 16 kHz microphone audio in, "my name was heard" moments out.

The model gives a posterior every 10 ms frame, carrying its convolution cache between
chunks. A call is detected when the smoothed posterior reaches ``threshold``; a second
bar, ``repeat_threshold``, catches the way people call a pet twice ("蛋黃、蛋黃") — two
near-misses within ``repeat_window_s`` count as one call. Detections closer than
``refractory_s`` merge. Audio is never stored: features and the cache are all that live.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from duckmind.namekws.features import FeatureConfig, StreamingLogMel, log_mel

SIDECAR_FORMAT = 2


@dataclass(frozen=True)
class DetectionRule:
    threshold: float
    repeat_threshold: float | None = None
    repeat_window_s: float = 3.0
    smooth_frames: int = 5
    refractory_s: float = 1.5

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be in (0, 1)")
        if self.repeat_threshold is not None and not 0.0 < self.repeat_threshold <= self.threshold:
            raise ValueError("repeat_threshold must be in (0, threshold]")
        if self.smooth_frames < 1 or self.refractory_s < 0 or self.repeat_window_s <= 0:
            raise ValueError("invalid smoothing, refractory or repeat window")


class Decider:
    """Turns per-frame posteriors into detection times, one chunk at a time.

    Only rising edges matter: a call fires when the smoothed posterior first crosses
    ``threshold`` (a long "蛋黃～" held above it stays one call), and a near-miss starts
    when it first crosses ``repeat_threshold``. :func:`events` computes the same thing
    from the edges alone, which is how hours of audio are scored quickly.
    """

    def __init__(self, rule: DetectionRule, hop_s: float) -> None:
        self.rule = rule
        self.hop_s = hop_s
        self._recent: deque[float] = deque(maxlen=rule.smooth_frames)
        self._frames = 0
        self._above_main = False
        self._above_near = False
        self._last_fire = -math.inf
        self._last_near = -math.inf

    def _step(self, now: float, main: bool, near: bool) -> bool:
        rule = self.rule
        fired = False
        cooled = now - self._last_fire >= rule.refractory_s
        if main and not self._above_main and cooled:
            fired = True
        elif rule.repeat_threshold is not None and near and not self._above_near:
            gap = now - self._last_near
            if 0.4 <= gap <= rule.repeat_window_s and cooled:
                fired = True
            else:
                self._last_near = now
        if fired:
            self._last_fire = now
            self._last_near = -math.inf
        self._above_main = main
        self._above_near = near
        return fired

    def push(self, posteriors: np.ndarray) -> list[float]:
        found: list[float] = []
        rule = self.rule
        near_bar = rule.repeat_threshold if rule.repeat_threshold is not None else math.inf
        for value in posteriors.tolist():
            self._frames += 1
            self._recent.append(float(value))
            smoothed = sum(self._recent) / len(self._recent)
            now = self._frames * self.hop_s
            if self._step(now, smoothed >= rule.threshold, smoothed >= near_bar):
                found.append(now)
        return found


def smooth(posteriors: np.ndarray, frames: int) -> np.ndarray:
    """Trailing moving average, identical to the streaming Decider's (it starts short)."""
    values = np.asarray(posteriors, dtype=np.float64)
    if frames <= 1 or values.size == 0:
        return values
    sums = np.cumsum(values)
    out = np.empty_like(values)
    out[:frames] = sums[:frames] / np.arange(1, min(frames, values.size) + 1)
    out[frames:] = (sums[frames:] - sums[:-frames]) / frames
    return out


def events(posteriors: np.ndarray, rule: DetectionRule, hop_s: float) -> list[float]:
    """Same detections as :class:`Decider`, visiting only the frames where anything changes."""
    smoothed = smooth(posteriors, rule.smooth_frames)
    main = smoothed >= rule.threshold
    near = smoothed >= (rule.repeat_threshold if rule.repeat_threshold is not None else np.inf)
    changes = np.flatnonzero(
        (np.diff(main.astype(np.int8), prepend=0) != 0)
        | (np.diff(near.astype(np.int8), prepend=0) != 0)
    )
    decider = Decider(rule, hop_s)
    found = []
    for index in changes.tolist():
        now = (index + 1) * hop_s
        if decider._step(now, bool(main[index]), bool(near[index])):
            found.append(now)
    return found


def decide(posteriors: np.ndarray, rule: DetectionRule, hop_s: float) -> list[float]:
    return Decider(rule, hop_s).push(posteriors)


def read_sidecar(model_path: Path) -> dict[str, Any]:
    """The JSON written next to the ONNX model; checks it describes these exact bytes."""
    sidecar = json.loads(model_path.with_suffix(".json").read_text())
    if sidecar.get("format") != SIDECAR_FORMAT:
        raise ValueError(f"unsupported sidecar format {sidecar.get('format')!r}")
    config = FeatureConfig(**sidecar["features"])
    if config.digest != sidecar["feature_digest"]:
        raise ValueError("feature parameters do not match the digest the model was trained on")
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != sidecar["model_sha256"]:
        raise ValueError("model bytes do not match the sidecar")
    return sidecar


def rule_from(sidecar: dict[str, Any]) -> DetectionRule:
    decision = sidecar["decision"]
    return DetectionRule(
        threshold=float(decision["threshold"]),
        repeat_threshold=(
            float(decision["repeat_threshold"]) if decision.get("repeat_threshold") else None
        ),
        repeat_window_s=float(decision.get("repeat_window_s", 3.0)),
        smooth_frames=int(decision.get("smooth_frames", 5)),
        refractory_s=float(decision.get("refractory_s", 1.5)),
    )


def open_session(model_path: Path) -> Any:
    import onnxruntime as ort

    options = ort.SessionOptions()
    # One thread that sleeps when idle: robotd's control loop owns the other cores.
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])


class StreamingModel:
    """The exported network, fed frames chunk by chunk with its cache carried along."""

    def __init__(self, session: Any, cache_shape: list[int], keyword: int = 0) -> None:
        self.session = session
        self.keyword = keyword
        self.cache = np.zeros(cache_shape, dtype=np.float32)

    def push(self, frames: np.ndarray) -> np.ndarray:
        if frames.shape[0] == 0:
            return np.zeros(0, dtype=np.float32)
        logits, self.cache = self.session.run(
            None, {"features": frames[None].astype(np.float32), "cache": self.cache}
        )
        return (1.0 / (1.0 + np.exp(-logits[0, :, self.keyword]))).astype(np.float32)


def scan(
    pcm: np.ndarray,
    session: Any,
    config: FeatureConfig,
    rule: DetectionRule,
    cache_shape: list[int],
    chunk: int = 3000,
) -> list[float]:
    """Detection times (seconds) over a whole recording, exactly as the robot streams it."""
    features = log_mel(pcm, config)
    model = StreamingModel(session, cache_shape)
    decider = Decider(rule, config.hop_length / config.sample_rate)
    times: list[float] = []
    for start in range(0, features.shape[0], chunk):
        times += decider.push(model.push(features[start : start + chunk]))
    return times


class NameDetector:
    """The on-robot loop: push microphone chunks, get back detection times."""

    def __init__(self, model_path: Path, rule: DetectionRule | None = None) -> None:
        sidecar = read_sidecar(model_path)
        self.config = FeatureConfig(**sidecar["features"])
        self.rule = rule or rule_from(sidecar)
        self._features = StreamingLogMel(self.config)
        self._model = StreamingModel(open_session(model_path), sidecar["cache_shape"])
        self._decider = Decider(self.rule, self.config.hop_length / self.config.sample_rate)

    def push(self, pcm: np.ndarray) -> list[float]:
        return self._decider.push(self._model.push(self._features.push(pcm)))
