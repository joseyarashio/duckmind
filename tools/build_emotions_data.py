"""Build src/duckmind/neuro/emotions_data.json from a microduck_emotions checkout (commit 431c73d).

    python tools/build_emotions_data.py <data folder>/emotions/microduck_emotions-431c73d

One entry per emotion: name, duration_s, sound_at_s, the sound letters, the English description (the page has its own
words in six languages), and the keyframes as rows ``[t, neck, head_pitch, head_yaw, head_roll, body_pitch, mouth]`` (0.1 s
steps, rad / 0..1) with ``sit`` (seconds the keyframes say 'sit' from) and ``dead`` (play dead: the duck goes limp).
NOTICE: Pollen Robotics' microduck_emotions, Apache-2.0 (third_party/microduck_emotions).
"""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np

COLUMNS = ["t", "neck", "head_pitch", "head_yaw", "head_roll", "body_pitch", "mouth"]


def envelope(path: Path, hop_s: float = 0.01) -> np.ndarray:
    with wave.open(str(path)) as w:
        raw = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(float)
        sr = w.getframerate()
    h = int(sr * hop_s)
    rms = np.array([np.sqrt(np.mean(raw[i * h : (i + 1) * h] ** 2)) for i in range(len(raw) // h)])
    return rms / (rms.max() + 1e-9)


def audio_delay(
    src: Path, entry: dict, frames: list[dict], lead: float, jaw_lag_s: float = 0.10
) -> float:
    """How long after t = 0 the simulator should start the sound so that the beak (the keyframes' ``mouth``) and the
    sound agree: the shift that best matches the sound's loudness to the mouth curve (0.01 s steps, over the
    workshop's own wavs), plus the jaw's own lag (0.10 s, measured in the simulator: robotd's mouth smoothing and the servo). The keyframes' mouth lags the wav by about 0.16 s in the
    workshop's files (the robot's sound reaches the speaker that much later); the simulator's page plays a sound the
    moment its body sample is drawn, so it waits that long."""
    t = np.arange(0, entry["duration_s"] + 1.0, 0.01)
    mouth = np.interp(
        t, [k["t"] - lead for k in frames], [k.get("mouth", 0.0) or 0.0 for k in frames]
    )
    best = (-2.0, 0.0)
    onset_gap = None
    for sound in entry["sounds"]:
        a = envelope(src / sound)
        first_sound = next((i for i, v in enumerate(a) if v > 0.03), 0) * 0.01
        first_mouth = next((tt for tt, m in zip(t, mouth, strict=True) if m > 0.15), first_sound)
        if (
            onset_gap is None
        ):  # (the first sound of the emotion: angry's second voicing growls before its first bark)
            onset_gap = first_mouth - first_sound
        base = np.zeros(len(t))
        base[: min(len(a), len(t))] = a[: len(t)]
        for step in range(0, 46):
            shifted = np.concatenate([np.zeros(step), base])[: len(t)]
            c = float(np.corrcoef(shifted, mouth)[0, 1])
            if c > best[0]:
                best = (c, step * 0.01)
    if best[0] < 0.7 and onset_gap is not None:
        # the beak is not tied to the loudness (angry: forced wide on the first bark so a leash drops): the first
        # opening is put on the first sound
        return round(max(0.0, onset_gap) + jaw_lag_s, 2)
    return round(best[1] + jaw_lag_s, 2)


def main() -> None:
    src = Path(sys.argv[1]).expanduser()
    listing = json.loads((src / "emotions.json").read_text())
    out = {"source": "pollen-robotics/microduck_emotions", "commit": "431c73df8a9d3cb6b6067d09ced6c169fdfa59fd",
           "licence": "Apache-2.0", "columns": COLUMNS, "step_s": listing["keyframe_step_s"], "emotions": {}}  # fmt: skip
    for e in listing["emotions"]:
        frames = json.loads((src / e["keyframes"]).read_text())["keyframes"]
        rows = [[round(float(k.get(c, 0.0) or 0.0), 3) for c in COLUMNS] for k in frames]
        sit = [k["t"] for k in frames if k.get("skill") == "sit"]
        out["emotions"][e["name"]] = {
            "duration_s": e["duration_s"],
            "sound_at_s": e["sound_at_s"],  # (the workshop's: devastated 0.3, the rest 0)
            "sound_delay_s": audio_delay(src, e, frames, float(e["sound_at_s"])),
            "sounds": [Path(s).stem for s in e["sounds"]],
            "description": e["description"],
            "sit_from_s": min(sit) if sit else None,
            "keyframes": rows,
        }
    target = (
        Path(__file__).resolve().parents[1] / "src" / "duckmind" / "neuro" / "emotions_data.json"
    )
    target.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print("wrote", target, len(out["emotions"]), "emotions")


if __name__ == "__main__":
    main()
