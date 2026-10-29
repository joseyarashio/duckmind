"""Local voice detection for barge-in (live voice, LV2): is a person speaking over the duck's call?

The duck's speaker is a few centimetres from its microphone, so while it calls its own voice is
the loudest thing in the microphone (-22…-37 dBFS against a person at 1.5 m: -35…-45) and a
detector that only looks at the level hears itself. This detector takes the duck's own voice out
first and listens to what is left:

1. **Echo taken out.** The duck knows when it voices and what: the vocal windows
   (:meth:`~duckmind.simweb.neuro_mic.MicStream.vocal_windows`: start, tag, level) and its own
   voice bank. For the call in a window it tries each variant of that tag (the page, like robotd,
   plays a random one and the duck is not told which) at the position in the call that the window's
   start gives (and a few small lags), keeps the sums that give each candidate's least-squares gain
   and what it would leave behind, and **locks** the best once it explains :data:`LOCK_EXPLAINED`
   of the energy of the last :data:`LOCK_FRAMES` frames (a call that starts under the motors' noise
   is not locked on the noise; until then the best so far is taken out and the choice goes on):
   from then on that variant at that gain is subtracted. **The position in a call is counted in
   samples, a whole frame per frame**: the frames' clock stamps follow the physics clock and
   wander against the audio (LV5 live: 20 ms of stamp is 320 samples only on average; a stall of
   the physics clock bends them by tens of samples within one call), and a position read from them
   leaves the duck's own call half un-cancelled, which is then heard as a person (every call of a
   live session was cut ~80 ms in). Calls may overlap (a second call begins
   before the first has ended): each window has its own lock and they are subtracted one after the
   other. What is left is the room, a person, and whatever the model of the echo got wrong. With no
   reference for a tag (a stream, no bank) nothing is subtracted and the detector is deaf while the
   duck voices (it cannot tell).
2. **Speech-like residual.** A frame is *voiced* when the residual is louder than the quiet room
   by :data:`ABOVE_FLOOR_DB` and, while the duck is voicing, also louder than
   :data:`ECHO_MARGIN_DB` below the echo that was taken out (an imperfect cancellation leaves a
   little of it: that must never count as a person), and its **spectral flatness** in the speech
   band is below :data:`FLATNESS_MAX` (noise is flat, a voice has structure; a frame of white noise
   reads 0.57 ± 0.05, the scripted lines under 0.2).
3. **Onset and hang.** :data:`ONSET_FRAMES` voiced frames in a row make an *onset* (stamped with
   the first of them: the latency to the decision is the debounce, not the frames before);
   :data:`HANG_FRAMES` unvoiced frames end the speech. The event stream is what the bridge reads:
   ``speech_start`` (with whether the duck was voicing when it began: a **barge-in**) and
   ``speech_end``; :meth:`VoiceDetector.speech_between` answers "did a person speak between a and b"
   for the gate (a call that looks like an echo call but follows locally heard speech is the
   person's).

What this is and is not: in the simulator the speaker-to-microphone path is a gain, so the echo
can be taken out exactly; a real path adds delay and room reverb that a gain does not model, and
then the margin of rule 2 is the trade between missing a quiet person and hearing the duck itself
(the lag search covers a small delay only). Everything runs on the microphone's own 20 ms frames
and the duck's own sound; nothing here reads what the world played (``MicStream.truth_since`` is
for the tests). Deterministic.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

SR = 16_000
FRAME = 320
FRAME_S = FRAME / SR
FFT_N = 512
BAND_HZ = (200.0, 4000.0)  # the speech band the flatness is taken over
ROOM_FLOOR_DBFS = -62.0  # the quiet room (30 dB at the head); the detector tracks its own floor
ABOVE_FLOOR_DB = 9.0  # a voiced frame is this much above the floor …
ECHO_MARGIN_DB = 18.0  # … and, while the duck voices, this far below the echo taken out is "its"
FLATNESS_MAX = 0.4  # a frame flatter than this (white-ish) is noise, not a voice: white noise
# reads 0.57 ± 0.05 (0.5 let 9 % of its frames pass, three in a row 7 in 10 000), the lines read < 0.2
ONSET_FRAMES = 3  # voiced frames in a row that make speech begin (60 ms)
HANG_FRAMES = 10  # unvoiced frames that end it (200 ms)
LOCK_FRAMES = 3  # frames of a call before its variant can be locked …
LOCK_EXPLAINED = 0.8  # … and only when the best candidate explains this much of the energy of the
# last LOCK_FRAMES frames (a call that starts under the motors' noise is not locked on the noise:
# the choice goes on with every frame until the call is what the microphone holds)
OWN_LEFTOVER = 0.6  # a residual this much explained (r², one frame) by one variant of what the duck
# is calling is the duck's own voice taken out badly, never a person (a person's speech is uncorrelated
# with its bank: r² under 0.05)
LAGS = (-160, -80, -40, -16, -8, -4, 0, 4, 8, 16, 40, 80, 160)  # samples the reference may be off
ECHO_TAIL_S = 0.12  # the room's decay after a call: the window's flag runs this long after it
EVENT_KEEP = 200
EPS = 1e-12


class Reference(Protocol):
    """The duck's own voice, as it knows it: every variant of a call as 16 kHz samples in the
    units of the microphone's 16-bit PCM (full scale ±32768)."""

    def variants(self, tag: str) -> Sequence[np.ndarray[Any, Any]]: ...


class BankReferences:
    """The references from the duck's own voice bank (:class:`duckmind.simweb.voice.VoiceBank`)."""

    def __init__(self, bank: Any, voice_seed: int) -> None:
        self.bank = bank
        self.voice_seed = int(voice_seed)
        self._cache: dict[str, list[np.ndarray[Any, Any]]] = {}

    def variants(self, tag: str) -> Sequence[np.ndarray[Any, Any]]:
        from duckmind.simweb import voice as V
        from duckmind.world import neuro_audio as A

        if tag in self._cache:
            return self._cache[tag]
        out: list[np.ndarray[Any, Any]] = []
        folder = None
        if tag in V.WHITELIST and tag != V.SEGMENTED and self.bank is not None:
            try:
                if self.bank.ready(self.voice_seed):
                    folder = self.bank.bank_dir(self.voice_seed)
            except Exception:  # (no bank here: no reference, the fallback is used)
                folder = None
        if folder is not None:
            for letter in V.letters(tag):
                try:
                    raw = (Path(folder) / tag / f"{tag}_{letter}.wav").read_bytes()
                except OSError:
                    continue
                decoded = A.wav_payload(raw)
                if decoded is None:
                    continue
                payload, rate = decoded
                wave = np.asarray(A.resample(A.to_float(payload), rate, SR), dtype=np.float64)
                out.append(32768.0 * wave)
        self._cache[tag] = out
        return out


@dataclass(frozen=True, slots=True)
class SpeechEvent:
    kind: str  # "speech_start" | "speech_end"
    t: float  # the clock of the frames (the first voiced frame's start / the last one's end)
    decided_t: float  # when the detector knew (the end of the frame that made it certain)
    over_duck: bool = False  # speech_start: the duck was voicing when the person began
    level_dbfs: float = -120.0  # the residual's level at the onset

    def view(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "t": round(self.t, 3),
            "decided_t": round(self.decided_t, 3),
            "over_duck": self.over_duck,
            "level_dbfs": round(self.level_dbfs, 1),
        }


@dataclass(slots=True)
class _Window:
    t0: float
    t1: float | None
    tag: str


@dataclass
class _Lock:
    """One of the duck's calls on its way to being taken out of the microphone."""

    refs: list[np.ndarray[Any, Any]]
    sums: dict[tuple[int, int], list[float]] = field(default_factory=dict)  # (k, lag) → [Σrx, Σrr]
    recent: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=LOCK_FRAMES))
    frames: int = 0
    best: tuple[int, int] | None = None
    gain: float = 0.0
    locked: bool = False
    start: int = 0  # the sample of the call the lock's first frame begins at …
    seen: int = 0  # … and how many frames the lock has taken out since (the call's own samples
    # advance a whole frame each: the frames' clock stamps wander, the audio's samples do not)

    def pos(self) -> int:
        """Where in the call the frame being processed begins."""
        return self.start + self.seen * FRAME

    def here(self) -> int:
        """Where in the call the frame just taken out began."""
        return self.start + (self.seen - 1) * FRAME

    def fit(self, pos: int) -> np.ndarray[Any, Any]:
        """What the best candidate so far says the microphone holds of this call at ``pos``."""
        if self.best is None or self.gain <= 0.0:
            return np.zeros(FRAME)
        k, lag = self.best
        return self.gain * _slice(self.refs[k], pos + lag)


def _slice(wave: np.ndarray[Any, Any], start: int) -> np.ndarray[Any, Any]:
    out = np.zeros(FRAME)
    lo, hi = max(0, start), min(len(wave), start + FRAME)
    if hi > lo:
        out[lo - start : hi - start] = wave[lo:hi]
    return out


def _dbfs(power: float) -> float:
    return 10.0 * math.log10(max(power, EPS) / (32768.0 * 32768.0))


def flatness(x: np.ndarray[Any, Any]) -> float:
    """Spectral flatness (geometric / arithmetic mean of the power) over the speech band: 1 for
    white noise, near 0 for something with structure."""
    spectrum = np.fft.rfft(x * np.hanning(len(x)), n=FFT_N)
    power = (spectrum.real**2 + spectrum.imag**2)[
        int(BAND_HZ[0] * FFT_N / SR) : int(BAND_HZ[1] * FFT_N / SR) + 1
    ]
    power = power + 1e-6
    return float(np.exp(np.mean(np.log(power))) / np.mean(power))


class VoiceDetector:
    """See the module docstring. One per duck; ``process`` once per microphone frame, in order."""

    def __init__(
        self,
        references: Reference | None = None,
        *,
        onset_frames: int = ONSET_FRAMES,
        hang_frames: int = HANG_FRAMES,
        above_floor_db: float = ABOVE_FLOOR_DB,
        echo_margin_db: float = ECHO_MARGIN_DB,
        flatness_max: float = FLATNESS_MAX,
        own_leftover: float | None = OWN_LEFTOVER,
    ) -> None:
        self.references = references
        self.onset_frames = int(onset_frames)
        self.hang_frames = int(hang_frames)
        self.above_floor_db = float(above_floor_db)
        self.echo_margin_db = float(echo_margin_db)
        self.flatness_max = float(flatness_max)
        self.own_leftover = None if own_leftover is None else float(own_leftover)
        self.floor_dbfs = ROOM_FLOOR_DBFS
        self.events: deque[SpeechEvent] = deque(maxlen=EVENT_KEEP)
        self.frames = 0
        self.voiced_frames = 0
        self.echo_frames = 0
        self.own_leftover_frames = 0
        self.speaking = False
        self.speaking_since: float | None = None
        self.last_voiced_end: float | None = None
        self._run = 0
        self._run_t0 = 0.0
        self._run_over = False
        self._run_level = -120.0
        self._quiet = 0
        self._locks: dict[tuple[float, str], _Lock] = {}
        self._unsure = False
        self._spans: deque[tuple[float, float]] = deque(maxlen=EVENT_KEEP)
        self._open_t0: float | None = None
        self.last: dict[str, Any] = {}

    # ── one frame ─────────────────────────────────────────────────────────────

    def process(
        self,
        pcm: bytes,
        sim_s: float,
        windows: Sequence[Any] = (),
        *,
        echo: bool = False,
    ) -> list[SpeechEvent]:
        """One 20 ms frame (``pcm``: 320 int16 samples, ``sim_s``: where it starts). ``windows``:
        the duck's own voicings (anything with ``t0 t1 tag kind``); ``echo``: the frame's own
        flag. Returns the speech events this frame made (start / end), if any."""
        x = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
        if len(x) != FRAME:
            raise ValueError(f"a frame is {FRAME} samples, got {len(x)}")
        self.frames += 1
        active = self._active(sim_s, windows)
        residual, echo_dbfs = self._take_out_echo(x, sim_s, active)
        level = _dbfs(float(np.mean(residual * residual)))
        duck = bool(active) or echo
        voiced = self._voiced(residual, level, echo_dbfs, duck) and not self._unsure
        if voiced and active and self.own_leftover is not None and self._leftover_is_own(residual):
            voiced = False  # (what is left is still the duck: two calls a frame apart, a variant
            self.own_leftover_frames += 1  # that fitted the start and not the end of one)
        if duck:
            self.echo_frames += 1
        elif not voiced:
            # the floor follows the quiet frames slowly (a noisier room: a higher floor)
            self.floor_dbfs += 0.02 * (min(level, self.floor_dbfs + 12.0) - self.floor_dbfs)
        self.last = {
            "t": sim_s,
            "level_dbfs": round(level, 1),
            "echo_dbfs": None if echo_dbfs is None else round(echo_dbfs, 1),
            "voiced": voiced,
            "windows": len(active),
            "own_leftover_frames": self.own_leftover_frames,
        }
        return self._advance(voiced, sim_s, level, duck)

    # ── the duck's own voice ──────────────────────────────────────────────────

    @staticmethod
    def _active(sim_s: float, windows: Sequence[Any]) -> list[_Window]:
        """The duck's own calls this frame falls in, oldest first (they may overlap)."""
        out: list[_Window] = []
        for w in windows:
            if getattr(w, "kind", "call") != "call":
                continue
            t0 = float(w.t0)
            t1 = None if w.t1 is None else float(w.t1)
            if t0 - 1e-6 <= sim_s < (math.inf if t1 is None else t1) + ECHO_TAIL_S:
                out.append(_Window(t0, t1, str(w.tag)))
        return sorted(out, key=lambda w: w.t0)

    def _take_out_echo(
        self, x: np.ndarray[Any, Any], sim_s: float, active: list[_Window]
    ) -> tuple[np.ndarray[Any, Any], float | None]:
        """The frame without the duck's own calls, and the level of what was taken out (None:
        nothing was — no call, or no reference to take it out with)."""
        keys = {(round(w.t0, 4), w.tag) for w in active}
        for old in [k for k in self._locks if k not in keys]:
            del self._locks[old]
        calls: list[tuple[_Lock, int]] = []
        for w in active:
            key = (round(w.t0, 4), w.tag)
            lock = self._locks.get(key)
            if lock is None:
                refs = list(self.references.variants(w.tag)) if self.references else []
                lock = self._locks[key] = _Lock(
                    refs=[np.asarray(r, dtype=np.float64) for r in refs]
                )
            if lock.seen == 0:
                lock.start = int(round((sim_s - w.t0) * SR))
            pos = lock.pos()
            lock.seen += 1
            if lock.refs:
                calls.append((lock, pos))
        self._unsure = False
        if not calls:
            return x, None
        fits = [lock.fit(pos) for lock, pos in calls]  # what each call's best so far holds
        taken = sum(fits, np.zeros(FRAME))
        asked: list[tuple[_Lock, bool]] = []
        for i, (lock, pos) in enumerate(calls):
            if lock.locked:
                continue
            # (the evidence for this call is what the others do not explain: two calls a frame
            # apart are told apart together, not the second on what the first left over)
            sure = self._search(lock, x - (taken - fits[i]), pos)
            taken = taken - fits[i]
            fits[i] = lock.fit(pos)
            taken = taken + fits[i]
            asked.append((lock, sure))
        if asked and all(sure for _, sure in asked):  # (they are locked together, or not yet)
            for lock, _ in asked:
                lock.locked = True
        # (two calls at once, not yet told apart: it cannot tell what is left from a person)
        self._unsure = len(calls) >= 2 and any(not lock.locked for lock, _ in calls)
        if not np.any(taken):
            return x, None
        return x - taken, _dbfs(float(np.mean(taken * taken)))

    @staticmethod
    def _search(lock: _Lock, x: np.ndarray[Any, Any], pos: int) -> bool:
        """One more frame of evidence for which variant, at which small lag and gain, the duck's
        call is. True when the best candidate explains most of the energy of the last few frames
        (to lock on it): while the motors' noise is louder than the start of the call, the evidence
        is the noise and a lock now would be a guess (the call then goes on with the wrong variant,
        gain or lag, and what is left of it is taken for a person)."""
        lock.frames += 1
        power = float(x @ x)
        for k, wave in enumerate(lock.refs):
            for lag in LAGS:
                ref = _slice(wave, pos + lag)
                entry = lock.sums.setdefault((k, lag), [0.0, 0.0])
                entry[0] += float(ref @ x)
                entry[1] += float(ref @ ref)
        best: tuple[float, tuple[int, int], float] | None = None
        for candidate, (rx, rr) in lock.sums.items():
            if rr < 1e-3 or rx <= 0.0:
                continue
            gain = rx / rr
            if gain > 3.0:
                continue
            explained = rx * rx / rr  # the energy a least-squares fit takes out
            if best is None or explained > best[0]:
                best = (explained, candidate, gain)
        if best is None:
            lock.recent.clear()
            return False
        lock.best, lock.gain = best[1], best[2]
        left = x - lock.fit(pos)
        lock.recent.append((power, float(left @ left)))
        if lock.frames < LOCK_FRAMES or len(lock.recent) < LOCK_FRAMES:
            return False
        energy = sum(e for e, _ in lock.recent)
        return sum(r for _, r in lock.recent) <= (1.0 - LOCK_EXPLAINED) * energy

    def _leftover_is_own(self, residual: np.ndarray[Any, Any]) -> bool:
        """Whether the residual, which looks like a voice, is explained by any variant of what the
        duck is calling now (at the places the calls are at, ± the small lags): then it is the duck's
        own voice that its fit did not take out, not a person. Looked at only for frames that
        would otherwise be heard as a person."""
        power = float(residual @ residual)
        threshold = OWN_LEFTOVER if self.own_leftover is None else self.own_leftover
        if power <= 0.0:
            return False
        for lock in self._locks.values():
            if lock.seen == 0:
                continue
            for wave in lock.refs:
                for lag in LAGS:
                    ref = _slice(wave, lock.here() + lag)
                    rr = float(ref @ ref)
                    if rr > 1e-3 and float(ref @ residual) ** 2 >= threshold * rr * power:
                        return True
        return False

    # ── voiced? ───────────────────────────────────────────────────────────────

    def _voiced(
        self,
        residual: np.ndarray[Any, Any],
        level: float,
        echo_dbfs: float | None,
        duck: bool,
    ) -> bool:
        threshold = self.floor_dbfs + self.above_floor_db
        if duck:
            if echo_dbfs is None:
                # no reference: it cannot tell a person from its own voice: deaf, as safe as can be
                return False
            threshold = max(threshold, echo_dbfs - self.echo_margin_db)
        if level <= threshold:
            return False
        return flatness(residual) < self.flatness_max

    # ── onset and hang ────────────────────────────────────────────────────────

    def _advance(self, voiced: bool, sim_s: float, level: float, duck: bool) -> list[SpeechEvent]:
        out: list[SpeechEvent] = []
        end = sim_s + FRAME_S
        if voiced:
            self.voiced_frames += 1
            self.last_voiced_end = end
            self._quiet = 0
            if self._run == 0:
                self._run_t0, self._run_over, self._run_level = sim_s, duck, level
            else:
                self._run_over = self._run_over or duck
            self._run += 1
            if not self.speaking and self._run >= self.onset_frames:
                self.speaking = True
                self.speaking_since = self._open_t0 = self._run_t0
                event = SpeechEvent(
                    "speech_start", self._run_t0, end, self._run_over, self._run_level
                )
                self.events.append(event)
                out.append(event)
            return out
        self._run = 0
        if self.speaking:
            self._quiet += 1
            if self._quiet >= self.hang_frames:
                self.speaking = False
                last = self.last_voiced_end if self.last_voiced_end is not None else end
                event = SpeechEvent("speech_end", last, end)
                self.events.append(event)
                out.append(event)
                if self._open_t0 is not None:
                    self._spans.append((self._open_t0, last))
                    self._open_t0 = None
                self.speaking_since = None
        return out

    # ── what the rest of the duck asks ────────────────────────────────────────

    def speech_between(self, t0: float, t1: float) -> bool:
        """Whether the duck itself heard a person speak at any time in ``[t0, t1]`` (the frames'
        clock). A speech that is still going counts."""
        if self.speaking and self.speaking_since is not None and self.speaking_since <= t1:
            return True
        return any(a <= t1 and b >= t0 for a, b in self._spans)

    def status(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "voiced_frames": self.voiced_frames,
            "echo_frames": self.echo_frames,
            "own_leftover_frames": self.own_leftover_frames,
            "speaking": self.speaking,
            "floor_dbfs": round(self.floor_dbfs, 1),
            "events": [e.view() for e in list(self.events)[-6:]],
            "last": dict(self.last),
        }


__all__ = [
    "ABOVE_FLOOR_DB",
    "ECHO_MARGIN_DB",
    "FLATNESS_MAX",
    "HANG_FRAMES",
    "ONSET_FRAMES",
    "BankReferences",
    "Reference",
    "SpeechEvent",
    "VoiceDetector",
    "flatness",
]
