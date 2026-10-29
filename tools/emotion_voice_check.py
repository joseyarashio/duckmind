"""Check the duck's emotion sounds against the workshop's originals: the notes of each sound (onset, length) must match.

    uv run --with numpy python tools/emotion_voice_check.py <voice seed> <originals dir> [<bank dir>]

The originals are ``microduck_emotions/sounds/*.wav`` (in the data folder: ``emotions/microduck_emotions-431c73d``).
Prints one row per sound and exits 1 when a sound has another number of notes, or a note whose onset differs by more than
``TOL_S`` seconds (its length by more than ``LEN_TOL_S``), or a different total length by more than 0.15 s.
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

from duckmind.simweb import emotion_voice as EV

TOL_S = 0.025
LEN_TOL_S = 0.06
NAMES = {  # original file -> (tag, letter)
    "sad": ("sad", "a"), "devastated": ("devastated", "a"), "curious": ("curious", "a"), "mmh": ("mmh", "a"),
    "yes": ("yes", "a"), "yes_fast": ("yes_fast", "a"), "no_a": ("no", "a"), "no_b": ("no", "b"),
    "no_c": ("no", "c"), "defiant": ("defiant", "a"), "angry_a": ("angry", "a"), "angry_b": ("angry", "b"),
    "mock": ("mock", "a"), "excited": ("excited", "a"), "impatient": ("impatient", "a"), "laugh": ("laugh", "a"),
    "play_dead": ("play_dead", "a"),
}  # fmt: skip


def notes(x: np.ndarray, sr: int, thr_db: float = -38.0, gap: float = 0.03, win: float = 0.01):
    """(onset, end) of each stretch louder than ``thr_db`` under the sound's peak, gaps under ``gap`` joined."""
    h = int(sr * win)
    n = len(x) // h
    rms = np.array([np.sqrt(np.mean(x[i * h : (i + 1) * h] ** 2)) for i in range(n)])
    on = 20 * np.log10(rms / (rms.max() + 1e-12) + 1e-9) > thr_db
    out: list[tuple[float, float]] = []
    i = 0
    g = max(1, int(gap / win))
    while i < n:
        if on[i]:
            j = i
            while j < n and (on[j] or on[j : j + g].any()):
                j += 1
            out.append((i * win, j * win))
            i = j
        else:
            i += 1
    return out


def read(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32768.0
        return w.getframerate(), x


def main() -> int:
    seed, originals = int(sys.argv[1]), Path(sys.argv[2])
    bank = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    voice = EV.Voice(EV.Personality(seed), bank)
    recipes = {(r.tag, r.letter): r for r in EV.RECIPES}
    bad = 0
    for name, key in NAMES.items():
        sr, original = read(originals / f"{name}.wav")
        mine = EV.render_one(recipes[key], voice)
        a, b = notes(original, sr), notes(mine, EV.SR)
        worst_on = max((abs(p[0] - q[0]) for p, q in zip(a, b, strict=False)), default=0.0)
        worst_len = max(
            (abs((p[1] - p[0]) - (q[1] - q[0])) for p, q in zip(a, b, strict=False)), default=0.0
        )
        d_total = abs(len(original) / sr - len(mine) / EV.SR)
        ok = len(a) == len(b) and worst_on <= TOL_S and worst_len <= LEN_TOL_S and d_total <= 0.15
        bad += not ok
        tag = " (approximated)" if recipes[key].approximated else ""
        print(f"{'ok ' if ok else 'BAD'} {name:11s} notes {len(a)}/{len(b)} onset max {worst_on * 1000:4.0f} ms "
              f"length max {worst_len * 1000:4.0f} ms total {len(original) / sr:.2f}/{len(mine) / EV.SR:.2f} s{tag}")  # fmt: skip
    print("all 17 match" if not bad else f"{bad} sounds differ")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
