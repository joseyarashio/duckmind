"""Step 2 of the scripted-utterance fixtures: ``synth_qwen.py``'s raw 24 kHz wavs → ``tests/fixtures/voice``.

Run with the duckmind venv (numpy only)::

    PYTHONPATH=src python tools/voice_fixtures/build.py --raw <raw dir> [--out tests/fixtures/voice]

For every line of ``lines.json`` it takes the best take ``<raw dir>/<id>.t<k>.wav`` (``--takes-asr``:
the transcripts ``asr.py --takes`` made; the take the recogniser heard as the line, of equals
one without a pause over :data:`MAX_PAUSE_S` inside, then the earliest; without it take 0), converts it to 16 kHz mono
16-bit with the simulator's own resampler (:func:`duckmind.world.neuro_audio.resample`), trims
the silence at both ends (keeping 120 ms after the speech), normalises the
active speech to :data:`~duckmind.world.neuro_audio.REF_DBFS`, writes ``<id>.wav`` and the
``manifest.json`` the simulator reads (:mod:`duckmind.world.neuro_utterances`). ``--asr`` is a
JSON ``{id: transcript}`` from ``asr.py --fixtures`` to store beside each line. Nothing here touches
the network.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from duckmind.world import neuro_audio as A

HERE = Path(__file__).resolve().parent
LEAD_S, TAIL_S = 0.0, 0.12  # (no lead: a line is said at the moment its speech begins)
GATE_DB = 35.0  # speech starts where a 10 ms window is within this of the loudest


MAX_PAUSE_S = 0.7  # the longest silence inside a line (a comma is shorter)


def speech_windows(x: np.ndarray, rate: int) -> tuple[np.ndarray, int]:
    """The 10 ms windows of ``x`` within :data:`GATE_DB` of the loudest (speech), and the window."""
    win = max(1, rate // 100)
    n = len(x) // win
    power = np.mean(np.square(x[: n * win].reshape(n, win)), axis=1)
    return np.flatnonzero(power >= power.max() * 10.0 ** (-GATE_DB / 10.0)), win


def longest_pause_s(x: np.ndarray, rate: int) -> float:
    """The longest silence between two stretches of speech in ``x``."""
    live, win = speech_windows(x, rate)
    return float(np.diff(live).max(initial=0)) * win / rate


def trim(x: np.ndarray, rate: int) -> np.ndarray:
    """``x`` without the silence at its ends (a 10 ms window within :data:`GATE_DB` of the loudest
    counts as speech), with a little lead and tail left."""
    live, win = speech_windows(x, rate)
    first = max(0, int(live[0] * win - LEAD_S * rate))
    last = min(len(x), int((live[-1] + 1) * win + TAIL_S * rate))
    return x[first:last]


def same_words(text: str, heard: str) -> float:
    """How much of ``text`` the recogniser heard: 1.0 when it wrote the same words (spacing and
    punctuation do not count)."""
    strip = re.compile(r"[\s，。！？、,.!?]")
    return difflib.SequenceMatcher(None, strip.sub("", text), strip.sub("", heard)).ratio()


def best_take(line: dict, raw: dict, heard: dict[str, str], folder: Path) -> int:
    """The take the recogniser heard best as ``line``; of equals, one with no pause longer than
    :data:`MAX_PAUSE_S` inside, then the earliest. A line's ``take`` in ``lines.json`` names it."""
    takes = {t["take"]: t["file"] for t in raw["lines"] if t["id"] == line["id"]}
    if "take" in line:
        return int(line["take"])
    if not heard:
        return min(takes)
    scores = {t: same_words(line["text"], heard.get(f"{line['id']}.t{t}", "")) for t in takes}

    def smooth(take: int) -> bool:
        samples, rate = A.read_wav(folder / takes[take])
        return longest_pause_s(A.to_float(samples).astype(np.float64), rate) <= MAX_PAUSE_S

    return max(takes, key=lambda t: (scores[t], smooth(t), -t))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=HERE.parents[1] / "tests" / "fixtures" / "voice"
    )
    parser.add_argument("--takes-asr", type=Path, help="asr.py --takes' JSON: pick the takes")
    parser.add_argument("--asr", type=Path, help="asr.py --fixtures' JSON, stored in the manifest")
    args = parser.parse_args()
    spec = json.loads((HERE / "lines.json").read_text(encoding="utf-8"))
    raw = json.loads((args.raw / "raw.json").read_text())
    heard = json.loads(args.takes_asr.read_text(encoding="utf-8")) if args.takes_asr else {}
    asr = json.loads(args.asr.read_text(encoding="utf-8")) if args.asr else {}
    args.out.mkdir(parents=True, exist_ok=True)
    entries = []
    for line in spec["lines"]:
        take = best_take(line, raw, heard, args.raw)
        info = next(t for t in raw["lines"] if t["id"] == line["id"] and t["take"] == take)
        samples, rate = A.read_wav(args.raw / info["file"])
        x = trim(A.to_float(samples).astype(np.float64), rate)
        y = A.normalise(A.resample(x, rate, A.SR))
        peak = float(np.max(np.abs(y)))
        if peak > 0.98:  # (the normalised level leaves room: this would be a broken take)
            raise SystemExit(f"{line['id']}: peak {peak:.2f} after normalising")
        pcm = A.to_pcm16(y)
        target = args.out / f"{line['id']}.wav"
        A.write_wav(target, pcm, A.SR)
        speaker = spec["speakers"][line["speaker"]]
        entry = {
            "id": line["id"],
            "text": line["text"],
            "speaker": line["speaker"],
            "persona": speaker["persona"],
            "intent": line["intent"],
            "name": line.get("name"),
            "file": target.name,
            "duration_s": round(len(pcm) / 2 / A.SR, 3),
            "samples": len(pcm) // 2,
            "active_rms_dbfs": round(A.active_rms_dbfs(A.to_float(pcm)), 2),
            "peak_dbfs": round(20 * np.log10(max(peak, 1e-9)), 2),
            "take": take,
            "seed": int(line["seed"]) + 1000 * take,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
        if line["id"] in asr:
            entry["asr_check"] = asr[line["id"]]
            entry["asr_match"] = round(same_words(line["text"], asr[line["id"]]), 3)
        entries.append(entry)
        print(
            f"{line['id']}: {entry['duration_s']} s, active {entry['active_rms_dbfs']} dBFS, "
            f"peak {entry['peak_dbfs']} dBFS, {target.stat().st_size} B "
            f"(take {take}, tts {info['duration_sec']} s)"
        )
    manifest = {
        "version": 1,
        "sample_rate": A.SR,
        "channels": 1,
        "bits": 16,
        "tts": {
            "engine": "Qwen3-TTS VoiceDesign (tools/voice_fixtures/synth_qwen.py)",
            "model": raw["model"],
            "revision": raw["revision"],
            "licence": "Apache-2.0 (model, speech tokenizer and qwen-tts package)",
            "language": raw["language"],
            "source_rate": raw["lines"][0]["rate"],
            "note": "voices are described, not cloned: no person's voice is in these files",
        },
        "speakers": spec["speakers"],
        "utterances": entries,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    total = sum(p.stat().st_size for p in args.out.iterdir())
    print(f"{len(entries)} lines, {total} bytes in {args.out}")


if __name__ == "__main__":
    main()
