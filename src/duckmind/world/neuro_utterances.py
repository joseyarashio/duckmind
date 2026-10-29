"""The family's scripted lines for the live-voice simulator: wav files and their manifest.

A scripted line is a short 繁體中文 sentence a family member says in the simulated home ("蛋黃，
過來這裡。", "這是媽媽。", "你看到什麼？" …), spoken once by Qwen3-TTS VoiceDesign (Apache-2.0,
run locally by ``tools/voice_fixtures``) and stored as ``tests/fixtures/voice/<id>.wav`` — 16 kHz
mono 16-bit, the active speech at :data:`~duckmind.world.neuro_audio.REF_DBFS` — beside
``manifest.json``::

    {"version": 1, "sample_rate": 16000, "utterances": [
        {"id": "call_name", "text": "蛋黃，過來這裡。", "speaker": "mom", "persona": "partner",
         "intent": "call_name", "name": "蛋黃", "file": "call_name.wav",
         "duration_s": 1.6, "samples": 25600, "active_rms_dbfs": -26.0}, …]}

The world makes one a stimulus (``x_neuro_stim`` ``{"stim": "say", "utterance": "<id>"}``, see
:mod:`duckmind.world.neuro_world`): a sound event with the line's duration at the speaker's
place. The world needs only the manifest (the duration); the duck's microphone stream
(:mod:`duckmind.simweb.neuro_mic`) plays the wav. Names the family gives (``name``: whom the line
calls) are data. ``$DUCKMIND_VOICE_FIXTURES`` names another folder.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from duckmind.world.neuro_audio import SR, read_wav

FIXTURE_DIR_ENV = "DUCKMIND_VOICE_FIXTURES"
DEFAULT_DIR = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "voice"
MANIFEST = "manifest.json"
LIVE_PREFIX = (
    "live:"  # the world event's utterance marker for a person speaking through a microphone
)
MAX_PLACE_MM = 50_000  # a place a person speaks from is within 50 m of the home's origin


@dataclass(frozen=True)
class Utterance:
    id: str
    text: str
    speaker: str  # the sound character (a voice), not a person's name
    persona: str  # the family member who says it by default (duckmind.world.people.PERSONAS)
    intent: str
    name: str | None  # whom the line calls (the duck's name, data), if it calls anyone
    file: str
    duration_s: float
    samples: int
    active_rms_dbfs: float

    def view(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "speaker": self.speaker,
            "persona": self.persona,
            "intent": self.intent,
            "name": self.name,
            "duration_s": self.duration_s,
        }


class UtteranceLibrary:
    """The scripted lines of one folder. Thread-safe; the wav files are read when first played."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        data = json.loads((self.root / MANIFEST).read_text(encoding="utf-8"))
        if int(data.get("sample_rate", 0)) != SR:
            raise ValueError(f"{self.root / MANIFEST}: the lines must be {SR} Hz")
        self.speakers: dict[str, Any] = dict(data.get("speakers") or {})
        self.lines: dict[str, Utterance] = {}
        for item in data.get("utterances", ()):
            line = Utterance(
                id=str(item["id"]),
                text=str(item["text"]),
                speaker=str(item["speaker"]),
                persona=str(item.get("persona") or ""),
                intent=str(item.get("intent") or ""),
                name=str(item["name"]) if item.get("name") else None,
                file=str(item["file"]),
                duration_s=float(item["duration_s"]),
                samples=int(item["samples"]),
                active_rms_dbfs=float(item.get("active_rms_dbfs", 0.0)),
            )
            self.lines[line.id] = line
        self._pcm: dict[str, np.ndarray[Any, Any]] = {}
        self._lock = threading.Lock()

    @classmethod
    def default(cls) -> UtteranceLibrary | None:
        """The library of ``$DUCKMIND_VOICE_FIXTURES`` or ``tests/fixtures/voice`` (None when
        there is none: the world then refuses scripted lines in plain words)."""
        raw = os.environ.get(FIXTURE_DIR_ENV)
        root = Path(raw).expanduser() if raw else DEFAULT_DIR
        if not (root / MANIFEST).is_file():
            return None
        return cls(root)

    def ids(self) -> list[str]:
        return list(self.lines)

    def get(self, ident: str) -> Utterance:
        try:
            return self.lines[ident]
        except KeyError:
            raise KeyError(ident) from None

    def pcm(self, ident: str) -> np.ndarray[Any, Any]:
        """The line's samples (int16, 16 kHz)."""
        line = self.get(ident)
        with self._lock:
            cached = self._pcm.get(ident)
            if cached is not None:
                return cached
        samples, rate = read_wav(self.root / line.file)
        if rate != SR:
            raise ValueError(f"{line.file}: {rate} Hz, the lines are {SR} Hz")
        with self._lock:
            self._pcm[ident] = samples
        return samples


__all__ = [
    "DEFAULT_DIR",
    "FIXTURE_DIR_ENV",
    "LIVE_PREFIX",
    "MANIFEST",
    "MAX_PLACE_MM",
    "Utterance",
    "UtteranceLibrary",
]
