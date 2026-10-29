"""Audio for training and evaluation, decoded once into RAM as int16.

Sources (all ungated, permissive licences — see docs/validation/run-name-kws.md):

* MobvoiHotwords (SLR87, Apache-2.0): real far-field Mandarin wake words from 788
  speakers aged 3–65, plus non-keyword speech with TV and music — the pretraining corpus,
  and later real negatives;
* the synthetic name data written by :mod:`duckmind.namekws.synth` (``metadata.jsonl``);
* folders of backgrounds (MUSAN), room impulse responses (SLR28) and long-form speech
  (YODAS2, AISHELL-4) for false-accept tests.

A bank is one big int16 array plus per-clip records, so sampling a batch of crops is an
index operation and a whole corpus fits in the Colab VM's memory.
"""

from __future__ import annotations

import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3"}
RATE = 16_000


@dataclass(frozen=True)
class Clip:
    offset: int
    length: int
    label: int  # keyword index, or -1 for "not a keyword"
    group: str  # speaker or synthetic voice: splits never share one
    source: str  # dataset or TTS engine
    kind: str  # keyword | negative | hard_negative | noise | rir | long


def load_audio(path: Path, rate: int = RATE) -> np.ndarray:
    """Mono float32 at ``rate``."""
    import soundfile
    from scipy.signal import resample_poly

    audio, source_rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if source_rate != rate:
        divisor = math.gcd(int(source_rate), rate)
        audio = resample_poly(audio, rate // divisor, int(source_rate) // divisor)
    return audio.astype(np.float32)


class AudioBank:
    def __init__(self) -> None:
        self._pending: list[np.ndarray] = []
        self._size = 0
        self.clips: list[Clip] = []
        self.data = np.zeros(0, dtype=np.int16)

    def __len__(self) -> int:
        return len(self.clips)

    @property
    def hours(self) -> float:
        return sum(clip.length for clip in self.clips) / RATE / 3600

    def add(self, audio: np.ndarray, *, label: int, group: str, source: str, kind: str) -> None:
        pcm = np.clip(audio, -1.0, 1.0)
        pcm16 = (pcm * 32767).astype(np.int16)
        self.clips.append(Clip(self._size, pcm16.size, label, group, source, kind))
        self._pending.append(pcm16)
        self._size += pcm16.size

    def finalize(self) -> AudioBank:
        if self._pending:
            self.data = np.concatenate([self.data, *self._pending])
            self._pending = []
        return self

    def audio(self, index: int) -> np.ndarray:
        clip = self.clips[index]
        return self.data[clip.offset : clip.offset + clip.length].astype(np.float32) / 32767.0

    def indices(self, **match: Any) -> list[int]:
        return [
            index
            for index, clip in enumerate(self.clips)
            if all(getattr(clip, key) == value for key, value in match.items())
        ]

    def save(self, path: Path) -> None:
        self.finalize()
        np.save(path.with_suffix(".npy"), self.data)
        path.with_suffix(".json").write_text(json.dumps([asdict(clip) for clip in self.clips]))

    @classmethod
    def load(cls, path: Path) -> AudioBank:
        bank = cls()
        bank.data = np.load(path.with_suffix(".npy"), mmap_mode=None)
        bank.clips = [Clip(**item) for item in json.loads(path.with_suffix(".json").read_text())]
        bank._size = int(bank.data.size)
        return bank


def _decode_all(paths: list[Path], workers: int = 16) -> list[np.ndarray | None]:
    def safe(path: Path) -> np.ndarray | None:
        try:
            return load_audio(path)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(safe, paths))


def load_mobvoi(root: Path, split: str) -> AudioBank:
    """One split (train/dev/test) of MobvoiHotwords: keywords 0 (嗨小問) and 1 (你好問問)."""
    resources = root / "mobvoi_hotword_dataset_resources"
    wavs = root / "mobvoi_hotword_dataset"
    entries: list[dict[str, Any]] = []
    for prefix in ("p", "n"):
        listing = resources / f"{prefix}_{split}.json"
        if listing.is_file():
            entries += json.loads(listing.read_text())
    paths = [wavs / f"{entry['utt_id']}.wav" for entry in entries]
    bank = AudioBank()
    for entry, audio in zip(entries, _decode_all(paths), strict=True):
        if audio is None or audio.size < RATE // 4:
            continue
        label = int(entry.get("keyword_id", -1))
        bank.add(
            audio,
            label=label if label >= 0 else -1,
            group=str(entry.get("speaker_id", entry["utt_id"])),
            source="mobvoi",
            kind="keyword" if label >= 0 else "negative",
        )
    return bank.finalize()


def load_synth(root: Path, drop_negatives_with: str = "") -> AudioBank:
    """The synthetic name corpus; clips the ASR filter rejected are left out.

    ``drop_negatives_with`` leaves out "hard negatives" that say the whole name inside
    another word (蛋黃酥, 把蛋黃打散): a streaming detector can only tell those apart after
    the next syllable, and learning to wait for it cost it most calls in noise.
    """
    bank = AudioBank()
    rows = [json.loads(line) for line in (root / "metadata.jsonl").read_text().splitlines() if line]
    rows = [row for row in rows if row.get("asr_ok", True) is not False]
    if drop_negatives_with:
        rows = [
            row
            for row in rows
            if row["kind"] == "positive" or drop_negatives_with not in str(row.get("text", ""))
        ]
    paths = [
        root / row["path"] if not Path(row["path"]).is_absolute() else Path(row["path"])
        for row in rows
    ]
    for row, audio in zip(rows, _decode_all(paths), strict=True):
        if audio is None:
            continue
        positive = row["kind"] == "positive"
        bank.add(
            audio,
            label=0 if positive else -1,
            group=str(row["voice_id"]),
            source=str(row.get("engine", "tts")),
            kind="keyword" if positive else "hard_negative",
        )
    return bank.finalize()


def audio_files(root: Path) -> list[Path]:
    """Every audio file under ``root``, following symlinked folders (``rglob`` does not,
    and a data folder is often assembled from links into the unpacked corpora)."""
    if not root.is_dir():
        return []
    return [
        Path(folder) / name
        for folder, _, names in os.walk(root, followlinks=True)
        for name in names
        if Path(name).suffix.lower() in AUDIO_SUFFIXES
    ]


def load_folder(root: Path, *, kind: str, source: str, limit_s: float | None = None) -> AudioBank:
    """Every audio file under ``root`` (backgrounds, RIRs or long speech)."""
    paths = sorted(audio_files(root))
    bank = AudioBank()
    total = 0.0
    for path, audio in zip(paths, _decode_all(paths), strict=True):
        if audio is None or audio.size == 0:
            continue
        bank.add(audio, label=-1, group=path.parent.name, source=source, kind=kind)
        total += audio.size / RATE
        if limit_s is not None and total >= limit_s:
            break
    return bank.finalize()


def long_stream(bank: AudioBank, indices: list[int], seconds: float, seed: int) -> np.ndarray:
    """An hours-long recording stitched from clips with short random gaps, for FA tests."""
    rng = np.random.default_rng(seed)
    order = list(indices)
    rng.shuffle(order)
    target = int(seconds * RATE)
    pieces: list[np.ndarray] = []
    total = 0
    for index in order:
        audio = bank.audio(index)
        gap = np.zeros(int(rng.uniform(0.1, 0.8) * RATE), dtype=np.float32)
        pieces += [audio, gap]
        total += audio.size + gap.size
        if total >= target:
            break
    return np.concatenate(pieces)[:target] if pieces else np.zeros(0, dtype=np.float32)
