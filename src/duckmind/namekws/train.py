"""Pretrain on real Mandarin wake words, fine-tune on the duck's name, choose a threshold, export.

1. **Pretrain** :class:`~duckmind.namekws.model.NameTCN` on MobvoiHotwords' two real
   keywords against its non-keyword speech — real voices from 3 to 65 years old, far
   field, with TV and music — so the acoustics are learned from people, not TTS.
2. **Fine-tune** a fresh one-keyword head on the synthetic name clips (held out by voice
   and by engine), against real negatives (every Mobvoi clip, keywords included), the
   synthetic hard negatives (蛋黃酥, 淡黃色…) and background speech.
3. **Evaluate** by streaming, exactly as the robot does: pick the threshold on a dev
   stream of hours of negatives, report false accepts per hour on a separate test
   stream (with the rule-of-three bound when there are none), misses on unseen voices at
   several SNRs, and the trigger rate on unseen hard negatives.
4. **Export** the streaming ONNX model and its sidecar (features, cache shape, decision
   rule, metrics, provenance).

Training batches are 2 s crops, augmented on the GPU (room reverb by FFT convolution,
noise/music/speech at −5…20 dB SNR, level), and the loss is max-pooling over frames
(WeKws). Requires numpy, scipy, soundfile, torch, onnx and onnxruntime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from duckmind.namekws.data import (
    RATE,
    AudioBank,
    load_folder,
    load_mobvoi,
    load_synth,
    long_stream,
)
from duckmind.namekws.detector import SIDECAR_FORMAT, DetectionRule, events
from duckmind.namekws.features import FeatureConfig


@dataclass(frozen=True)
class TrainSpec:
    pretrain_steps: int = 8000
    finetune_steps: int = 6000
    batch: int = 512
    lr: float = 1e-3
    finetune_lr: float = 5e-4
    weight_decay: float = 1e-4
    window_s: float = 2.0
    positive_fraction: float = 0.25
    hard_fraction: float = 0.15
    noise_p: float = 0.8
    snr_db: tuple[float, float] = (-5.0, 20.0)
    rir_p: float = 0.6
    gain_db: tuple[float, float] = (-30.0, 0.0)
    target_fa_per_hour: float = 0.3
    heldout_voices: float = 0.15
    seed: int = 7
    hidden: int = 32
    stacks: int = 3
    layers: int = 4
    kernel: int = 5


Pick = tuple[AudioBank, int]


class Batcher:
    """Crops clips into fixed windows, augments them on the GPU and makes features."""

    def __init__(
        self,
        spec: TrainSpec,
        device: str,
        positives: list[Pick],
        negatives: list[Pick],
        hard: list[Pick],
        noise: list[Pick],
        rirs: list[Pick],
    ) -> None:
        import torch

        from duckmind.namekws.torchfeat import LogMel

        self.torch = torch
        self.spec = spec
        self.device = device
        self.positives = positives
        self.negatives = negatives
        self.hard = hard
        self.noise = noise
        self.rirs = rirs
        self.samples = round(spec.window_s * RATE)
        self.logmel = LogMel(FeatureConfig(), device)
        self.rng = np.random.default_rng(spec.seed)

    def _crop(self, pick: Pick, keyword: bool) -> np.ndarray:
        bank, index = pick
        audio = bank.audio(index)
        out = np.zeros(self.samples, dtype=np.float32)
        if keyword:
            # End-align: the call finishes late in the window, with left context before it.
            tail = audio[-self.samples :]
            end = int(self.rng.uniform(0.55, 0.95) * self.samples)
            start = max(0, end - tail.size)
            usable = min(tail.size, self.samples - start)
            out[start : start + usable] = tail[:usable]
            return out
        if audio.size >= self.samples:
            begin = int(self.rng.integers(0, audio.size - self.samples + 1))
            return audio[begin : begin + self.samples]
        begin = int(self.rng.integers(0, self.samples - audio.size + 1))
        out[begin : begin + audio.size] = audio
        return out

    def _background(self, count: int) -> np.ndarray:
        rows = np.zeros((count, self.samples), dtype=np.float32)
        for row in range(count):
            bank, index = self.noise[int(self.rng.integers(len(self.noise)))]
            audio = bank.audio(index)
            if audio.size >= self.samples:
                begin = int(self.rng.integers(0, audio.size - self.samples + 1))
                rows[row] = audio[begin : begin + self.samples]
            else:
                rows[row] = np.resize(audio, self.samples)
        return rows

    def sample(self, labels_of: dict[int, int] | None = None) -> tuple[Any, Any]:
        """(features [B, T, mels], labels [B]) with −1 meaning "no keyword"."""
        torch = self.torch
        spec = self.spec
        batch = spec.batch
        n_pos = round(batch * spec.positive_fraction)
        n_hard = round(batch * spec.hard_fraction) if self.hard else 0
        n_neg = batch - n_pos - n_hard
        crops, labels = [], []
        for _ in range(n_pos):
            bank, index = self.positives[int(self.rng.integers(len(self.positives)))]
            crops.append(self._crop((bank, index), keyword=True))
            label = bank.clips[index].label
            labels.append(labels_of.get(label, label) if labels_of else label)
        for pool, count in ((self.hard, n_hard), (self.negatives, n_neg)):
            for _ in range(count):
                crops.append(self._crop(pool[int(self.rng.integers(len(pool)))], keyword=False))
                labels.append(-1)
        audio = torch.from_numpy(np.stack(crops)).to(self.device)
        features = self.logmel(self._augment(audio))
        return self._spec_augment(features), torch.tensor(labels, device=self.device)

    def _augment(self, audio: Any) -> Any:
        torch = self.torch
        spec = self.spec
        count, samples = audio.shape
        if self.rirs:
            reverb = torch.from_numpy(self.rng.random(count) < spec.rir_p).to(self.device)
            if bool(reverb.any()):
                picked = [self.rirs[int(self.rng.integers(len(self.rirs)))] for _ in range(count)]
                length = min(RATE, max(bank.clips[i].length for bank, i in picked))
                kernels = np.zeros((count, length), dtype=np.float32)
                for row, (bank, index) in enumerate(picked):
                    rir = bank.audio(index)[:length]
                    kernels[row, : rir.size] = rir / (np.abs(rir).max() + 1e-9)
                kern = torch.from_numpy(kernels).to(self.device)
                size = samples + length - 1
                spectrum = torch.fft.rfft(audio, n=size) * torch.fft.rfft(kern, n=size)
                wet = torch.fft.irfft(spectrum, n=size)[:, :samples]
                peak_dry = audio.abs().amax(dim=1, keepdim=True)
                peak_wet = wet.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)
                audio = torch.where(reverb[:, None], wet * peak_dry / peak_wet, audio)
        if self.noise:
            mix = torch.from_numpy(self.rng.random(count) < spec.noise_p).to(self.device)
            bed = torch.from_numpy(self._background(count)).to(self.device)
            snr = self.rng.uniform(*spec.snr_db, count).astype(np.float32)
            snr_t = torch.from_numpy(snr).to(self.device)
            speech_power = audio.pow(2).mean(dim=1).clamp_min(1e-8)
            noise_power = bed.pow(2).mean(dim=1).clamp_min(1e-8)
            scale = torch.sqrt(speech_power / (noise_power * 10 ** (snr_t / 10)))
            # A silent crop (padding around a short negative) still gets a normal background.
            scale = torch.where(speech_power > 1e-6, scale, torch.full_like(scale, 0.3))
            audio = torch.where(mix[:, None], audio + bed * scale[:, None], audio)
        gain_db = self.rng.uniform(*spec.gain_db, count).astype(np.float32)
        peak = audio.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)
        target = torch.from_numpy(10 ** (gain_db / 20)).to(self.device)[:, None]
        return (audio / peak * target).clamp(-1.0, 1.0)

    def _spec_augment(self, features: Any) -> Any:
        torch = self.torch
        count, frames, mels = features.shape
        floor = features.amin(dim=(1, 2), keepdim=True)
        steps = torch.arange(frames, device=self.device)[None, :]
        bins = torch.arange(mels, device=self.device)[None, :]
        width = torch.from_numpy(self.rng.integers(0, 11, count)).to(self.device)
        start = torch.from_numpy(self.rng.integers(0, max(1, frames - 10), count)).to(self.device)
        in_time = (steps >= start[:, None]) & (steps < (start + width)[:, None])
        mask = in_time[:, :, None].expand(count, frames, mels).clone()
        for _ in range(2):
            fwidth = torch.from_numpy(self.rng.integers(0, 9, count)).to(self.device)
            fstart = torch.from_numpy(self.rng.integers(0, mels - 8, count)).to(self.device)
            in_band = (bins >= fstart[:, None]) & (bins < (fstart + fwidth)[:, None])
            mask |= in_band[:, None, :]
        return torch.where(mask, floor.expand_as(features), features)


def max_pool_loss(logits: Any, labels: Any) -> Any:
    """WeKws max-pooling: a keyword clip must peak somewhere, a negative nowhere."""
    import torch

    peaks = logits.max(dim=1).values  # [B, K]
    targets = torch.zeros_like(peaks)
    keyword = labels >= 0
    targets[keyword, labels[keyword]] = 1.0
    return torch.nn.functional.binary_cross_entropy_with_logits(peaks, targets)


def _train_stage(
    model: Any,
    batcher: Batcher,
    steps: int,
    lr: float,
    spec: TrainSpec,
    log: Any,
    labels_of: dict[int, int] | None = None,
) -> list[dict[str, float]]:
    import torch

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=spec.weight_decay)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, lr, total_steps=max(1, steps), pct_start=0.1
    )
    model.train()
    history = []
    running = 0.0
    for step in range(1, steps + 1):
        features, labels = batcher.sample(labels_of)
        logits, _ = model(features)
        loss = max_pool_loss(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()
        running += loss.item()
        if step % 200 == 0 or step == steps:
            count = 200 if step % 200 == 0 else step % 200
            history.append({"step": step, "loss": running / count})
            log(f"step {step}/{steps} loss {running / count:.4f}")
            running = 0.0
    model.eval()
    return history


def stream_posteriors(
    model: Any, audio: np.ndarray, device: str, keyword: int = 0, seconds: float = 120.0
) -> np.ndarray:
    """Per-frame posteriors over a long recording, chunked with the cache carried along."""
    import torch

    from duckmind.namekws.torchfeat import LogMel

    config = FeatureConfig()
    logmel = LogMel(config, device)
    hop = config.hop_length
    step = int(seconds * RATE) // hop * hop
    tail = config.win_length - hop
    cache = model.empty_cache(1, device)
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, max(0, audio.size - config.win_length + 1), step):
            piece = audio[start : start + step + tail]
            if piece.size < config.win_length:
                break
            features = logmel(torch.from_numpy(piece.astype(np.float32))[None].to(device))
            logits, cache = model(features, cache)
            out.append(torch.sigmoid(logits[0, :, keyword]).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def _detections(posteriors: np.ndarray, rule: DetectionRule) -> list[float]:
    return events(posteriors, rule, FeatureConfig().hop_length / RATE)


def choose_rule(dev_posteriors: np.ndarray, dev_hours: float, target: float) -> DetectionRule:
    """Lowest threshold (logit grid) whose dev false accepts stay at half the target or less,
    then the lowest repeat bar that keeps the total there too."""
    budget = target / 2
    grid = [1 / (1 + math.exp(-z)) for z in np.arange(0.0, 9.01, 0.1)]
    threshold = grid[-1]
    for value in grid:
        if len(_detections(dev_posteriors, DetectionRule(threshold=value))) / dev_hours <= budget:
            threshold = value
            break
    repeat = None
    for value in (v for v in grid if v < threshold):
        rule = DetectionRule(threshold=threshold, repeat_threshold=value)
        if len(_detections(dev_posteriors, rule)) / dev_hours <= budget:
            repeat = value
            break
    return DetectionRule(threshold=threshold, repeat_threshold=repeat)


def embed(
    background: np.ndarray,
    clips: list[np.ndarray],
    snr_db: float | None,
    seed: int,
    spacing_s: float = 5.0,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Place clips every ``spacing_s`` over a background at an SNR; return windows to hit.

    Calls keep a realistic loudness (peaks between −20 and −3 dBFS, as training sees them)
    and the background is scaled under them to reach the SNR. Scaling the call to the
    background instead made calls over a quiet recording far quieter than any real call.
    """
    rng = np.random.default_rng(seed)
    total = int(len(clips) * spacing_s * RATE) + RATE
    speech = np.zeros(total, dtype=np.float32)
    windows = []
    powers = []
    for index, clip in enumerate(clips):
        start = int((index * spacing_s + rng.uniform(0.5, 1.5)) * RATE)
        end = min(total, start + clip.size)
        piece = clip[: end - start]
        peak = float(np.abs(piece).max()) + 1e-9
        piece = piece / peak * 10 ** (rng.uniform(-20.0, -3.0) / 20)
        speech[start:end] += piece
        powers.append(float(np.mean(piece**2)))
        windows.append((start / RATE, end / RATE + 1.0))
    if not (background.size and snr_db is not None):
        return np.clip(speech, -1.0, 1.0), windows
    bed = np.resize(background, total).astype(np.float32)
    noise = float(np.mean(bed**2)) + 1e-12
    call = float(np.mean(powers)) if powers else 0.0
    bed = bed * math.sqrt(call / (noise * 10 ** (snr_db / 10)))
    return np.clip(speech + bed, -1.0, 1.0), windows


def evaluate(
    model: Any, banks: dict[str, Any], spec: TrainSpec, device: str, log: Any
) -> tuple[DetectionRule, dict[str, Any]]:
    dev_audio: np.ndarray = banks["dev_stream"]
    test_audio: np.ndarray = banks["test_stream"]
    dev_hours = dev_audio.size / RATE / 3600
    test_hours = test_audio.size / RATE / 3600
    dev_post = stream_posteriors(model, dev_audio, device)
    rule = choose_rule(dev_post, max(dev_hours, 1e-6), spec.target_fa_per_hour)
    log(f"threshold {rule.threshold:.4f}, repeat {rule.repeat_threshold}, {dev_hours:.1f} dev h")
    test_fa = len(_detections(stream_posteriors(model, test_audio, device), rule))
    report: dict[str, Any] = {
        "decision": asdict(rule),
        "dev_hours": round(dev_hours, 2),
        "dev_fa_per_hour": round(len(_detections(dev_post, rule)) / max(dev_hours, 1e-9), 3),
        "test_hours": round(test_hours, 2),
        "test_false_accepts": test_fa,
        "test_fa_per_hour": round(test_fa / max(test_hours, 1e-9), 3),
        # Rule of three: none seen in T hours bounds the rate below 3/T at 95% confidence.
        "test_fa_upper95_if_zero": round(3 / test_hours, 3)
        if test_fa == 0 and test_hours
        else None,
    }
    heldout: list[np.ndarray] = banks["heldout_positive_audio"]
    background: np.ndarray = banks["noise_stream"]
    misses = {}
    for snr in (None, 20.0, 10.0, 5.0, 0.0):
        stream, windows = embed(background, heldout, snr, seed=spec.seed)
        found = _detections(stream_posteriors(model, stream, device), rule)
        hit = sum(any(start <= t <= end for t in found) for start, end in windows)
        misses["clean" if snr is None else f"{snr:g}dB"] = round(1 - hit / max(1, len(windows)), 4)
    report["miss_rate"] = misses
    report["heldout_positive_clips"] = len(heldout)
    hard: list[np.ndarray] = banks["heldout_hard_audio"]
    if hard:
        stream, windows = embed(background, hard, 20.0, seed=spec.seed + 1)
        found = _detections(stream_posteriors(model, stream, device), rule)
        triggered = sum(any(start <= t <= end for t in found) for start, end in windows)
        report["hard_negative_trigger_rate"] = round(triggered / len(windows), 4)
        report["hard_negative_clips"] = len(hard)
    return rule, report


def export(
    model: Any,
    out_dir: Path,
    rule: DetectionRule,
    report: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    import onnxruntime as ort
    import torch

    from duckmind.namekws.model import export_onnx

    config = FeatureConfig()
    model = model.eval().cpu()
    path = out_dir / "name_kws.onnx"
    export_onnx(model, path, config.n_mels)
    # Streaming parity: ONNX fed in chunks must equal torch over the whole sequence.
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    features = torch.randn(1, 250, config.n_mels) * 3 - 8
    with torch.no_grad():
        expected, _ = model(features)
    cache = model.empty_cache(1).numpy()
    parts = []
    for start in range(0, 250, 10):
        chunk = features[:, start : start + 10].numpy()
        logits, cache = session.run(None, {"features": chunk, "cache": cache})
        parts.append(logits)
    parity = float(np.abs(np.concatenate(parts, axis=1) - expected.numpy()).max())
    if parity > 1e-3:
        raise RuntimeError(f"streaming ONNX differs from torch by {parity}")
    sidecar = {
        "format": SIDECAR_FORMAT,
        "name": provenance.get("name", ""),
        "pinyin": provenance.get("pinyin", ""),
        "features": asdict(config),
        "feature_digest": config.digest,
        "cache_shape": list(model.empty_cache(1).shape),
        "decision": asdict(rule),
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "streaming_parity": parity,
        "evaluation": report,
        "provenance": provenance,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (out_dir / "name_kws.json").write_text(json.dumps(sidecar, ensure_ascii=False, indent=2))
    return sidecar


def _cached(path: Path | None, build: Any) -> AudioBank:
    if path is not None and path.with_suffix(".npy").is_file():
        return AudioBank.load(path)
    bank = build()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        bank.save(path)
    return bank


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from duckmind.namekws.model import NameTCN

    started = time.monotonic()

    def log(message: str) -> None:
        print(f"[{time.monotonic() - started:7.1f}s] {message}", flush=True)

    spec = TrainSpec(
        pretrain_steps=args.pretrain_steps,
        finetune_steps=args.finetune_steps,
        batch=args.batch,
        target_fa_per_hour=args.target_fa,
        seed=args.seed,
        hidden=args.hidden,
        stacks=args.stacks,
    )
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    torch.manual_seed(spec.seed)
    cache = Path(args.cache) if args.cache else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mobvoi = Path(args.mobvoi)

    def cached(name: str, build: Any) -> AudioBank:
        return _cached(cache / name if cache else None, build)

    log("loading audio (decoded once, cached for later runs)")
    mob_train = cached("mobvoi_train", lambda: load_mobvoi(mobvoi, "train"))
    mob_dev = cached("mobvoi_dev", lambda: load_mobvoi(mobvoi, "dev"))
    mob_test = cached("mobvoi_test", lambda: load_mobvoi(mobvoi, "test"))
    synth = load_synth(Path(args.synth), args.name if args.keep_name_negatives is False else "")
    noise = cached("noise", lambda: load_folder(Path(args.noise), kind="noise", source="bg"))
    rirs = load_folder(Path(args.rir), kind="rir", source="rir") if args.rir else AudioBank()
    long_speech = (
        cached("long", lambda: load_folder(Path(args.long), kind="long", source="long"))
        if args.long
        else AudioBank()
    )
    log(
        f"mobvoi train {mob_train.hours:.1f} h, dev {mob_dev.hours:.1f} h, "
        f"test {mob_test.hours:.1f} h; synthetic {len(synth)} clips; "
        f"backgrounds {noise.hours:.1f} h; long speech {long_speech.hours:.1f} h"
    )

    # Hold out voices (and the whole cloned-voice engine when present) for evaluation.
    rng = np.random.default_rng(spec.seed)
    voices = sorted({clip.group for clip in synth.clips})
    rng.shuffle(voices)
    held = set(voices[: max(1, round(len(voices) * spec.heldout_voices))])
    engines = sorted({clip.source for clip in synth.clips})
    held_engine = "qwen3-clone" if "qwen3-clone" in engines and len(engines) > 1 else None

    def is_heldout(clip: Any) -> bool:
        return clip.group in held or clip.source == held_engine

    pos_train = [
        (synth, i) for i, c in enumerate(synth.clips) if c.label == 0 and not is_heldout(c)
    ]
    hard_train = [
        (synth, i)
        for i, c in enumerate(synth.clips)
        if c.kind == "hard_negative" and not is_heldout(c)
    ]
    background = [(noise, i) for i in range(len(noise))]
    background += [(mob_train, i) for i in mob_train.indices(kind="negative")[:20000]]
    rir_picks = [(rirs, i) for i in range(len(rirs))]
    provenance = {
        "name": args.name,
        "pinyin": args.pinyin,
        "synthetic_clips": len(synth),
        "synthetic_voices": len(voices),
        "heldout_voices": len(held),
        "heldout_engine": held_engine,
        "engines": engines,
        "data": [
            "MobvoiHotwords SLR87 (Apache-2.0)",
            "MUSAN (CC BY 4.0)",
            "RIRS_NOISES SLR28 (Apache-2.0)",
        ]
        + (["long-form speech: " + args.long] if args.long else []),
        "spec": asdict(spec),
    }

    model = NameTCN(
        hidden=spec.hidden,
        stacks=spec.stacks,
        layers=spec.layers,
        kernel=spec.kernel,
        outputs=2,
    ).to(device)
    mob_keywords = [(mob_train, i) for i, c in enumerate(mob_train.clips) if c.label >= 0]
    mob_negatives = [(mob_train, i) for i in mob_train.indices(kind="negative")]
    pre = Batcher(spec, device, mob_keywords, mob_negatives, [], background, rir_picks)
    with torch.no_grad():  # normalization from augmented training features
        stacked = torch.cat([pre.sample()[0].reshape(-1, 40) for _ in range(4)])
        model.set_normalization(stacked.mean(dim=0), stacked.std(dim=0))
    history: dict[str, Any] = {}
    if args.init:
        # Start from an earlier run's pretrained weights (its normalization included).
        model.load_state_dict(torch.load(args.init, map_location=device))
        provenance["init"] = str(args.init)
        log(f"starting from {args.init}")
    if spec.pretrain_steps:
        log("pretraining on real Mandarin wake words")
        history["pretrain"] = _train_stage(model, pre, spec.pretrain_steps, spec.lr, spec, log)
        torch.save(model.state_dict(), out / "pretrained.pt")

    log(f"fine-tuning on {len(pos_train)} name clips and {len(hard_train)} hard negatives")
    model.replace_head(1)
    negatives = [(mob_train, i) for i in range(len(mob_train))]
    negatives += [(long_speech, i) for i in range(min(5000, len(long_speech)))]
    fine = Batcher(spec, device, pos_train, negatives, hard_train, background, rir_picks)
    history["finetune"] = _train_stage(
        model, fine, spec.finetune_steps, spec.finetune_lr, spec, log, labels_of={0: 0}
    )

    log("evaluating by streaming, as on the robot")
    long_ids = list(range(len(long_speech)))
    half = len(long_ids) // 2
    mob_dev_ids = mob_dev.indices(kind="negative") + mob_dev.indices(kind="keyword")
    mob_test_ids = mob_test.indices(kind="negative") + mob_test.indices(kind="keyword")
    dev_s, test_s = args.dev_hours * 3600, args.test_hours * 3600
    banks = {
        "dev_stream": np.concatenate(
            [
                long_stream(mob_dev, mob_dev_ids, dev_s * 0.5, seed=1),
                long_stream(long_speech, long_ids[:half], dev_s * 0.5, seed=2),
            ]
        ),
        "test_stream": np.concatenate(
            [
                long_stream(mob_test, mob_test_ids, test_s * 0.5, seed=3),
                long_stream(long_speech, long_ids[half:], test_s * 0.5, seed=4),
            ]
        ),
        "noise_stream": long_stream(noise, list(range(len(noise))), 3600, seed=5),
        "heldout_positive_audio": [
            synth.audio(i) for i, c in enumerate(synth.clips) if c.label == 0 and is_heldout(c)
        ][: args.heldout_clips],
        "heldout_hard_audio": [
            synth.audio(i)
            for i, c in enumerate(synth.clips)
            if c.kind == "hard_negative" and is_heldout(c)
        ][: args.heldout_clips],
    }
    torch.save(model.state_dict(), out / "finetuned.pt")
    rule, report = evaluate(model, banks, spec, device, log)
    report["history"] = history
    sidecar = export(model, out, rule, report, provenance)
    summary = {key: report[key] for key in ("dev_fa_per_hour", "test_fa_per_hour", "miss_rate")}
    log(f"wrote {out / 'name_kws.onnx'}: {json.dumps(summary, ensure_ascii=False)}")
    return sidecar


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the duck's name detector")
    parser.add_argument("--mobvoi", required=True, help="has mobvoi_hotword_dataset/ + _resources/")
    parser.add_argument("--synth", required=True, help="output folder of duckmind.namekws.synth")
    parser.add_argument("--noise", required=True, help="background audio (e.g. MUSAN)")
    parser.add_argument("--rir", default="", help="room impulse responses (e.g. SLR28)")
    parser.add_argument("--long", default="", help="long-form speech for false-accept tests")
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache", default="", help="where decoded audio banks are kept")
    parser.add_argument("--name", default="")
    parser.add_argument("--pinyin", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pretrain-steps", type=int, default=8000)
    parser.add_argument("--init", default="", help="pretrained.pt of an earlier run to start from")
    parser.add_argument("--hidden", type=int, default=32, help="channels (32 ≈ 17k parameters)")
    parser.add_argument("--stacks", type=int, default=3)
    parser.add_argument(
        "--keep-name-negatives",
        action="store_true",
        help="also train on negatives that contain the whole name (蛋黃酥); off by default",
    )
    parser.add_argument("--finetune-steps", type=int, default=6000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--target-fa", type=float, default=0.3, help="false accepts per hour")
    parser.add_argument("--dev-hours", type=float, default=50.0)
    parser.add_argument("--test-hours", type=float, default=72.0)
    parser.add_argument("--heldout-clips", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=7)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
