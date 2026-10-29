"""The duck's microphone as a PCM stream (live voice, phase LV0).

One stream of what the duck's microphone hears: **16 kHz mono 16-bit, 20 ms frames** (320
samples, 640 bytes), on the physics clock — the format Gemini Live takes and robot-bridge pushes
(``IN_RATE``/``IN_CHUNK``). A frame is the mix of

* **family speech**: a person at a place in the home saying a scripted line (a wav of
  :mod:`duckmind.world.neuro_utterances`, played by the world's ``say`` stimulus) or speaking
  through a browser microphone (:mod:`duckmind.simweb.neuro_mic_api`, ``live`` sources). The
  level is the world's own: each sound event carries its dB at the duck's head (distance, a wall
  in between: ``neuro_world.level_at``), and :data:`~duckmind.world.neuro_audio.FULL_SCALE_DB`
  turns that into PCM, so the stream's loudness and the hearing model's agree;
* **the world's other sounds** (a clap, a door, the TV, a song): a small waveform per kind
  (:mod:`duckmind.world.neuro_sound_synth`) at the event's level; the song is the world's own
  (``neuro_sound``'s ``music``: its bpm, seed and start);
* **the duck's own calls as an echo**: the duck's speaker is a few centimetres from its
  microphone, so what it voices comes back loud, at :data:`SPEAKER_MIC_GAIN_DB` — a one-shot
  call from the voice bank (:class:`VoiceBankCalls`, the very wavs the page plays) or the 48 kHz
  plane of a ride or the theremin (``VoiceEngine``'s stream), resampled. Every frame that holds
  (or just held: :data:`ECHO_TAIL_S`) the duck's own voice is **flagged** (:attr:`MicFrame.echo`)
  and the voicing windows are listed (:meth:`MicStream.vocal_windows`), so a later stage can mark
  a transcript that falls in one as "probably an echo" and stop a call when someone speaks over it;
* **the room's own noise** (30 dB: -62 dBFS) and, when given, the duck's motor noise.

A person at a **browser microphone** is a real recording, not a model of a voice, and the cloud is
the one that has to understand it: the duck's own ear hears them at the world's level (distance,
a wall between: the stream's ``pcm``), and every frame that holds one also carries ``cloud_pcm``,
the same frame with that person at the level their recording was normalised to
(:data:`LIVE_CLOUD_DBFS`, speech's normal level) wherever they stand. Without it a person 3 m away
reached the cloud at -47 dBFS with the walking motors at -53: transcripts of Spanish and
Vietnamese words (live, 2026-10-04). The duck's own calls and the room's noise are the same in both.

Honest like the other sensors: the frames carry **only audio and the duck's own flag** — never
which sound is which, where it is or who speaks. (What truly played, per frame, is kept apart in
:meth:`MicStream.truth_since` for evaluators and tests; no brain-side code reads it.)

A tick is one 20 ms frame: :meth:`MicStream.tick` is called once per robotd tick with the physics
second, the world's sound list (as the hearing model gets it), the song, the voice engine's plane
and the calls the duck voiced this tick. Everything is deterministic for a seed and the inputs.
"""

from __future__ import annotations

import math
import random
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from duckmind.world import neuro_audio as A
from duckmind.world import neuro_sound_synth as synth
from duckmind.world.neuro_senses import LATE_S, ROOM_DB
from duckmind.world.neuro_utterances import LIVE_PREFIX, UtteranceLibrary

SR = A.SR
FRAME = A.FRAME
FRAME_S = A.FRAME_S
SONG_SR = 48_000
SONG_ACTIVE_DBFS = -17.9  # the world's song, active RMS (measured on bpm 84–124, tolerance 1.5 dB)
SPEAKER_MIC_GAIN_DB = -9.0  # the duck's speaker → its microphone, a few cm of shell (a MODEL knob)
ECHO_TAIL_S = 0.12  # the room's decay: a frame this soon after the duck's voice is still "its echo"
STREAM_GATE_DBFS = -70.0  # the voice plane's frames quieter than this are the instrument at rest
CALL_S = {  # how long a call lasts when the bank is not there to say (the official bank's lengths)
    "alarm": 0.5,
    "greet": 0.6,
    "inquire": 0.5,
    "peck": 0.15,
    "chirp": 0.3,
    "coo": 0.8,
}
RESYNC_S = 0.25  # a physics clock this far from the stream's is a jump: lined up at once
SLEW = 0.02  # otherwise the stream follows it slowly (jitter does not move the audio)
KEEP_FRAMES = 500  # the ring: 10 s
KEEP_WINDOWS = 64
LIVE_BACKLOG_S = 1.0  # a browser microphone's queue is cut to its newest second
LIVE_HOLD_S = 0.1  # a live source stays audible this long after its world event ends
LIVE_IDLE_S = 3.0  # a live source nobody has spoken into for this long is forgotten
LIVE_TARGET_DBFS = A.REF_DBFS
LIVE_CLOUD_DBFS: float | None = LIVE_TARGET_DBFS  # what the cloud hears of a browser microphone
LIVE_VOICED_DBFS = -50.0  # a chunk louder than this is speech for the level estimate
LIVE_MAX_GAIN = 200.0
LIVE_MIN_GAIN = 0.05
MUSIC_SRC = "music:bed"

Clock = Callable[[], float]


# ── the pieces ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MicFrame:
    """One 20 ms frame of the microphone."""

    seq: int
    sim_s: float  # physics seconds at its first sample
    wall: float  # the server's wall clock at that moment (epoch s)
    pcm: bytes  # 320 × int16, little-endian
    echo: bool  # the duck's own voice is in it (or just was): not a person
    level_dbfs: float  # the frame's RMS
    # what goes to the cloud when a browser microphone's person is in the frame (see the module
    # doc): the same frame with that person at :data:`LIVE_CLOUD_DBFS`. None: the cloud gets ``pcm``.
    cloud_pcm: bytes | None = None
    cloud_level_dbfs: float | None = None


@dataclass
class VocalWindow:
    """A stretch of time the duck was voicing (``t1`` None: still going)."""

    t0: float
    t1: float | None
    tag: str
    kind: str  # "call" (a one-shot of the bank) | "stream" (a ride, the theremin)
    level: float

    def view(self) -> dict[str, Any]:
        return {
            "t0": round(self.t0, 3),
            "t1": None if self.t1 is None else round(self.t1, 3),
            "tag": self.tag,
            "kind": self.kind,
            "level": round(self.level, 2),
        }


class CallBank(Protocol):
    """Where the duck's calls come from: ``call(tag, rng)`` → (variant, 16 kHz float samples) or
    None for a tag it has no sound for."""

    def call(self, tag: str, rng: random.Random) -> tuple[str, np.ndarray[Any, Any]] | None: ...


class VoiceBankCalls:
    """The duck's calls from its voice bank (:class:`duckmind.simweb.voice.VoiceBank`): robotd plays
    a random variant of a tag each time, and so does this. Decoded once, kept at 16 kHz."""

    def __init__(self, bank: Any, voice_seed: int) -> None:
        self.bank = bank
        self.voice_seed = int(voice_seed)
        self._cache: dict[tuple[str, str], np.ndarray[Any, Any]] = {}

    def call(self, tag: str, rng: random.Random) -> tuple[str, np.ndarray[Any, Any]] | None:
        from duckmind.simweb import voice as V

        if tag not in V.WHITELIST or tag == V.SEGMENTED:  # (a ride is a stream, not a call)
            return None
        bank = self.bank
        if bank is None or not bank.ready(self.voice_seed):
            return None
        letter = rng.choice(V.letters(tag))
        key = (tag, letter)
        if key not in self._cache:
            folder = bank.bank_dir(self.voice_seed)
            try:
                raw = (Path(folder) / tag / f"{tag}_{letter}.wav").read_bytes()
            except OSError:
                return None
            decoded = A.wav_payload(raw)
            if decoded is None:
                return None
            payload, rate = decoded
            self._cache[key] = A.resample(A.to_float(payload), rate, SR)
        return letter, self._cache[key]


@dataclass
class _Play:
    """One world sound being played into the stream."""

    key: tuple[int, float]
    kind: str
    wave: np.ndarray[Any, Any] | None  # None: a generator (the song)
    loop: bool
    pos: int  # the next sample of the wave (negative: it has not begun)
    t1: float | None  # when a looping sound stops
    db: float
    ref_dbfs: float = A.REF_DBFS  # the active level of ``wave`` at unit gain
    gain: float | None = None  # the gain the last frame ended on (ramped to the next)
    gen: Callable[[float], np.ndarray[Any, Any]] | None = None
    utterance: str | None = None
    shift: float = 0.0  # it reached the stream late: its times are the world's plus this
    done: bool = False


@dataclass(eq=False)  # (identity: two calls at once, a list.remove compared their sample arrays
class _Echo:  # and the microphone's tick raised, a frame a tick lost until the first one ended)
    """One of the duck's own calls on its way into the microphone."""

    wave: np.ndarray[Any, Any]
    pos: int
    gain: float
    window: VocalWindow


@dataclass
class _Live:
    """A person speaking into a browser microphone: a queue of samples and a level estimate."""

    queue: deque[np.ndarray[Any, Any]] = field(default_factory=deque)
    queued: int = 0
    level_dbfs: float | None = None
    last_push: float = 0.0
    pushed: int = 0
    dropped: int = 0
    gain: float | None = None


class _SongGen:
    """The world's song for the microphone: its 48 kHz samples on the stream's clock, decimated."""

    def __init__(self, bpm: float, seed: int, t0: float) -> None:
        from duckmind.world.neuro_music import Song

        self.song = Song(bpm=bpm, seed=seed)
        self.t0 = float(t0)
        self.dec = A.StreamDecimator(3)
        self.next: int | None = None

    def frame(self, clock: float) -> np.ndarray[Any, Any]:
        n48 = FRAME * (SONG_SR // SR)
        want = int(round((clock - self.t0) * SONG_SR))
        if self.next is None or abs(self.next - want) > SONG_SR // 50:  # lined up (> 20 ms off)
            self.next = want
            self.dec.reset()
        start = self.next
        self.next = start + n48
        if start + n48 <= 0:
            return np.zeros(FRAME, dtype=np.float32)
        live = np.asarray(self.song.render(max(0, start), n48 - max(0, -start)), dtype=np.float64)
        if start < 0:
            live = np.concatenate([np.zeros(-start), live])
        out = self.dec.process(live)
        if len(out) < FRAME:
            out = np.concatenate([out, np.zeros(FRAME - len(out), dtype=np.float32)])
        return out[:FRAME]


# ── the stream ──────────────────────────────────────────────────────────────────────


class MicStream:
    """See the module docstring. Thread-safe: a tick and the browser-microphone feeds may come
    from different threads."""

    def __init__(
        self,
        *,
        seed: int = 7,
        library: UtteranceLibrary | None = None,
        calls: CallBank | None = None,
        keep: int = KEEP_FRAMES,
        wall: Clock = time.time,
        cloud_live_dbfs: float | None = LIVE_CLOUD_DBFS,
    ) -> None:
        self.seed = int(seed)
        self.cloud_live_dbfs = cloud_live_dbfs
        self.library = library
        self.calls = calls
        self._wall = wall
        self._lock = threading.RLock()
        self.cond = threading.Condition()
        self.frames: deque[MicFrame] = deque(maxlen=keep)
        self._truth: deque[dict[str, Any]] = deque(maxlen=keep)
        self.seq = 0
        self._clock: float | None = None
        self._plays: dict[tuple[int, float], _Play] = {}
        self._finished: set[tuple[int, float]] = set()
        self._live: dict[str, _Live] = {}
        self._live_events: dict[tuple[int, float], tuple[str, float, float, float]] = {}
        self._echoes: list[_Echo] = []
        self._call_rng = random.Random(f"mic-calls:{self.seed}")
        self._decim = A.StreamDecimator(3)
        self._stream_window: VocalWindow | None = None
        self._stream_quiet = 0
        self._stream_end = 0.0
        self._windows: deque[VocalWindow] = deque(maxlen=KEEP_WINDOWS)
        self._listeners: list[Callable[[MicFrame], None]] = []
        # counters (the status line and the tests)
        self.resyncs = 0
        self.missing_lines = 0
        self.echo_frames = 0
        self.errors: list[str] = []

    # ── what reads the stream ─────────────────────────────────────────────────

    def since(self, seq: int) -> list[MicFrame]:
        """The frames after ``seq`` that are still in the ring."""
        with self.cond:
            return [f for f in self.frames if f.seq > seq]

    def wait(self, seq: int, timeout: float) -> list[MicFrame]:
        with self.cond:
            if self.seq <= seq:
                self.cond.wait(timeout)
            return [f for f in self.frames if f.seq > seq]

    def listen(self, fn: Callable[[MicFrame], None]) -> None:
        """``fn(frame)`` for every frame, in order, on the ticking thread (a fault in ``fn`` is
        counted in :attr:`errors`, never raised into the tick)."""
        with self._lock:
            self._listeners.append(fn)

    def vocal_windows(self, since_s: float | None = None) -> list[VocalWindow]:
        """When the duck voiced (an open window has ``t1`` None), oldest first; ``since_s``: only
        those that end after it."""
        with self._lock:
            return [
                VocalWindow(w.t0, w.t1, w.tag, w.kind, w.level)
                for w in self._windows
                if since_s is None or w.t1 is None or w.t1 > since_s
            ]

    def cut_calls(self, sim_s: float | None = None) -> int:
        """The duck stops its own one-shot call(s) now (live voice, LV2: a person spoke over it
        and it went quiet): what still plays of them is dropped, their windows end at ``sim_s``
        (default: the end of the last frame made). Returns how many calls were cut. A stream (a
        ride, the theremin) is the voice engine's, not cut here."""
        with self._lock:
            now = (
                (0.0 if self._clock is None else self._clock + FRAME_S) if sim_s is None else sim_s
            )
            cut = 0
            for window in self._windows:
                if window.kind == "call" and window.t1 is not None and window.t1 > now:
                    window.t1 = max(window.t0, now)
                    cut += 1
            if cut:
                self._echoes.clear()
            return cut

    def in_echo(self, sim_s: float) -> bool:
        """Whether physics second ``sim_s`` is in the duck's own voice or its tail."""
        with self._lock:
            for w in self._windows:
                end = w.t1 if w.t1 is not None else math.inf
                if w.t0 <= sim_s <= end + ECHO_TAIL_S:
                    return True
        return False

    def truth_since(self, seq: int) -> list[dict[str, Any]]:
        """Evaluators and tests only: what truly played in each frame after ``seq`` (the world
        sounds with their kinds and lines, the live speakers, the duck's own voice)."""
        with self._lock:
            return [dict(t) for t in self._truth if t["seq"] > seq]

    def view(self) -> dict[str, Any]:
        with self._lock:
            live = {
                who: {"queued_ms": round(1000.0 * src.queued / SR), "dropped": src.dropped}
                for who, src in self._live.items()
            }
            return {
                "rate": SR,
                "frame_ms": int(FRAME_S * 1000),
                "frames": self.seq,
                "echo_frames": self.echo_frames,
                "clock": None if self._clock is None else round(self._clock, 3),
                "playing": len(self._plays),
                "live": live,
                "windows": [w.view() for w in list(self._windows)[-5:]],
                "resyncs": self.resyncs,
                "missing_lines": self.missing_lines,
                "errors": list(self.errors[-3:]),
            }

    # ── a person speaking into a browser microphone ──────────────────────────

    def feed_live(self, who: str, pcm: bytes, *, now: float | None = None) -> dict[str, Any]:
        """Queue 16 kHz mono int16 samples a person speaks (``who``: the speaker's key). Their
        level is brought to the level of normal speech first (a slow gain on what they say), so
        how loud they are *in the home* is the world's: the distance from where they stand to
        the duck. Heard only while the world has a ``live:<who>`` sound for them
        (``x_neuro_stim`` ``say``); the queue keeps at most the newest
        :data:`LIVE_BACKLOG_S` seconds. Returns what was taken."""
        if len(pcm) % 2:
            raise ValueError("odd number of bytes")
        x = A.to_float(pcm).astype(np.float64)
        stamp = self._wall() if now is None else now
        with self._lock:
            src = self._live.setdefault(who, _Live())
            level = A.rms_dbfs(x)
            if level > LIVE_VOICED_DBFS:
                src.level_dbfs = (
                    level if src.level_dbfs is None else 0.7 * src.level_dbfs + 0.3 * level
                )
            est = LIVE_TARGET_DBFS if src.level_dbfs is None else src.level_dbfs
            gain = min(LIVE_MAX_GAIN, max(LIVE_MIN_GAIN, 10.0 ** ((LIVE_TARGET_DBFS - est) / 20.0)))
            src.queue.append((x * gain).astype(np.float32))
            src.queued += len(x)
            src.pushed += len(x)
            src.last_push = stamp
            cap = int(LIVE_BACKLOG_S * SR)
            dropped = 0
            while src.queued > cap and src.queue:
                head = src.queue[0]
                cut = min(len(head), src.queued - cap)
                if cut >= len(head):
                    src.queue.popleft()
                else:
                    src.queue[0] = head[cut:]
                src.queued -= cut
                dropped += cut
            src.dropped += dropped
            return {
                "accepted_ms": round(1000.0 * len(x) / SR),
                "queued_ms": round(1000.0 * src.queued / SR),
                "dropped_ms": round(1000.0 * dropped / SR),
            }

    # ── one frame ─────────────────────────────────────────────────────────────

    def tick(
        self,
        sim_s: float,
        sounds: Sequence[Mapping[str, Any]] | None = None,
        *,
        music: Mapping[str, Any] | None = None,
        duck_pcm48: bytes | None = None,
        calls: Sequence[Mapping[str, Any]] = (),
        motor_db: float | None = None,
        wall: float | None = None,
    ) -> MicFrame:
        """The next 20 ms frame. ``sim_s``: the physics second of its start. ``sounds``: the
        world's sound events as the hearing model gets them (None: no news, keep what plays);
        ``music``: the section ``neuro_sound``'s ``music`` (bpm, seed, t0); ``duck_pcm48``: the
        voice engine's plane of this tick (a ride, the theremin); ``calls``: what the duck voiced
        this tick (``{"tag", "level"}``); ``motor_db``: its own motor noise."""
        with self._lock:
            clock = self._advance(float(sim_s))
            acc = np.zeros(FRAME, dtype=np.float64)
            truth: list[dict[str, Any]] = []
            if sounds is not None:
                self._sync(clock, sounds, music)
            for play in list(self._plays.values()):
                part = self._render(play, clock)
                if part is not None:
                    acc += part
                    truth.append(
                        {"kind": play.kind, "utterance": play.utterance, "db": round(play.db, 1)}
                    )
                if play.done:
                    self._plays.pop(play.key, None)
                    self._finished.add(play.key)
            live, live_cloud = self._render_live(clock, truth)
            acc += live
            echo = self._render_own(clock, duck_pcm48, calls, acc, truth)
            acc += self._noise(motor_db)
            self.seq += 1
            pcm = A.to_pcm16(acc)
            cloud_pcm: bytes | None = None
            cloud_level: float | None = None
            if live_cloud is not None:  # (a browser microphone's person is in this frame)
                cloud_pcm = A.to_pcm16(acc - live + live_cloud)
                cloud_level = round(A.rms_dbfs(A.to_float(cloud_pcm)), 2)
            frame = MicFrame(
                seq=self.seq,
                sim_s=clock,
                wall=self._wall() if wall is None else float(wall),
                pcm=pcm,
                echo=echo,
                level_dbfs=round(A.rms_dbfs(A.to_float(pcm)), 2),
                cloud_pcm=cloud_pcm,
                cloud_level_dbfs=cloud_level,
            )
            if echo:
                self.echo_frames += 1
            self._truth.append({"seq": frame.seq, "sim_s": clock, "echo": echo, "playing": truth})
            listeners = list(self._listeners)
        with self.cond:
            self.frames.append(frame)
            self.cond.notify_all()
        for fn in listeners:
            try:
                fn(frame)
            except Exception as error:  # a listener's fault never stops the microphone
                if len(self.errors) < 20:
                    self.errors.append(f"listener: {type(error).__name__}: {error}")
        return frame

    def _advance(self, sim_s: float) -> float:
        """The stream's own clock: it moves 20 ms a frame and follows the physics clock slowly
        (so a jittery reading never bends the audio), or lines up at once after a jump."""
        if self._clock is None:
            self._clock = sim_s
            return sim_s
        expected = self._clock + FRAME_S
        err = sim_s - expected
        if abs(err) > RESYNC_S:
            self._clock = sim_s
            self.resyncs += 1
        else:
            self._clock = expected + SLEW * err
        return self._clock

    # ── the world's sounds ────────────────────────────────────────────────────

    def _sync(
        self, clock: float, sounds: Sequence[Mapping[str, Any]], music: Mapping[str, Any] | None
    ) -> None:
        seen: set[tuple[int, float]] = set()
        live_now: dict[tuple[int, float], tuple[str, float, float, float]] = {}
        for item in sounds:
            try:
                sid = int(item["id"])
                t0 = float(item["t0"])
                t1 = float(item["t1"])
            except (KeyError, TypeError, ValueError):
                continue
            key = (sid, round(t0, 3))
            seen.add(key)
            kind = str(item.get("kind") or "noise")
            db = float(item.get("db", ROOM_DB))
            src = str(item.get("src") or "")
            utter = item.get("utterance")
            if isinstance(utter, str) and utter.startswith(LIVE_PREFIX):
                live_now[key] = (utter[len(LIVE_PREFIX) :], t0, t1, db)
                continue
            play = self._plays.get(key)
            if play is not None:
                play.db = db
                if play.loop:
                    play.t1 = t1 + play.shift  # (music off ends the song: its end moved up)
                continue
            if key in self._finished:
                continue
            if src.startswith("music:") and src != MUSIC_SRC:
                self._finished.add(key)  # (a drum hit: the song's own audio has it)
                continue
            self._plays[key] = self._start(key, clock, kind, db, t0, t1, src, utter, music)
        self._live_events = live_now
        for key, play in list(self._plays.items()):
            if key not in seen and play.loop and (play.t1 is None or clock >= play.t1):
                del self._plays[key]  # (the world dropped a steady sound that has ended)
        self._finished &= seen

    def _start(
        self,
        key: tuple[int, float],
        clock: float,
        kind: str,
        db: float,
        t0: float,
        t1: float,
        src: str,
        utter: Any,
        music: Mapping[str, Any] | None,
    ) -> _Play:
        shift = 0.0
        if clock - LATE_S < t0 < clock - 0.005:  # as the hearing model: heard from arrival, whole
            shift = clock - t0
        begin, end = t0 + shift, t1 + shift
        pos = int(round((clock - begin) * SR))
        sid = key[0]
        if src == MUSIC_SRC and music and music.get("on"):
            gen = _SongGen(float(music["bpm"]), int(music["seed"]), float(music["t0"]))
            return _Play(
                key,
                kind,
                None,
                True,
                0,
                end,
                db,
                ref_dbfs=SONG_ACTIVE_DBFS,
                gen=gen.frame,
                shift=shift,
            )
        wave: np.ndarray[Any, Any]
        loop = False
        name = utter if isinstance(utter, str) else None
        library = self.library
        if name is not None and library is not None and name in library.lines:
            wave = A.to_float(library.pcm(name))
        else:
            if name is not None:
                self.missing_lines += 1  # (no wav for it: it is said as unscripted speech)
            wave, loop = synth.render(kind, seconds=end - begin, seed=self.seed, ident=sid)
        if loop and len(wave) and pos > 0:
            pos %= len(wave)
        return _Play(
            key, kind, wave, loop, pos, end if loop else None, db, utterance=name, shift=shift
        )

    def _render(self, play: _Play, clock: float) -> np.ndarray[Any, Any] | None:
        n = FRAME
        target = 10.0 ** ((play.db - A.FULL_SCALE_DB - play.ref_dbfs) / 20.0)
        previous = target if play.gain is None else play.gain
        if play.gen is not None:
            raw = np.asarray(play.gen(clock), dtype=np.float64)
        else:
            assert play.wave is not None
            wave = play.wave
            raw = np.zeros(n)
            if play.pos + n <= 0:
                play.pos += n
                return None
            first = max(0, -play.pos)
            idx = np.arange(play.pos + first, play.pos + n)
            if play.loop and len(wave):
                raw[first:] = wave[idx % len(wave)]
            else:
                inside = idx < len(wave)
                raw[first : first + int(inside.sum())] = wave[idx[inside]]
            play.pos += n
            if not play.loop and play.pos >= len(wave):
                play.done = True
        ramp = np.linspace(previous, target, n, endpoint=False)
        if play.loop and play.t1 is not None and clock + FRAME_S >= play.t1:
            cut = max(0.0, play.t1 - clock) / FRAME_S * n  # it ends inside this frame: a short fade
            ramp = ramp * np.clip((cut - np.arange(n)) / 40.0, 0.0, 1.0)
            play.done = True
        play.gain = target
        return raw * ramp

    # ── a person at a browser microphone ──────────────────────────────────────

    def _render_live(
        self, clock: float, truth: list[dict[str, Any]]
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any] | None]:
        """The browser microphones' people for this frame: at the world's level (what the duck's
        ear hears) and at :attr:`cloud_live_dbfs` (what the cloud hears; None when nobody speaks
        in this frame or the cloud is to hear the same as the ear)."""
        out = np.zeros(FRAME)
        cloud = np.zeros(FRAME)
        spoken = False
        cloud_gain = (
            None
            if self.cloud_live_dbfs is None
            else 10.0 ** ((self.cloud_live_dbfs - LIVE_TARGET_DBFS) / 20.0)
        )
        stamp = self._wall()
        for who in list(self._live):
            src = self._live[who]
            if stamp - src.last_push > LIVE_IDLE_S:
                del self._live[who]  # (nobody speaks there any more; what waited is let go)
                continue
            db = None
            for w, t0, t1, d in self._live_events.values():
                if w == who and t0 - FRAME_S <= clock <= t1 + LIVE_HOLD_S:
                    db = d if db is None else max(db, d)
            if db is None or src.queued == 0:
                continue  # nobody there yet (the world's event is on its way): the speech waits
            take = np.zeros(FRAME, dtype=np.float32)
            got = 0
            while got < FRAME and src.queue:
                head = src.queue[0]
                k = min(len(head), FRAME - got)
                take[got : got + k] = head[:k]
                got += k
                if k >= len(head):
                    src.queue.popleft()
                else:
                    src.queue[0] = head[k:]
                src.queued -= k
            target = 10.0 ** ((db - A.FULL_SCALE_DB - LIVE_TARGET_DBFS) / 20.0)
            ramp = np.linspace(
                target if src.gain is None else src.gain, target, FRAME, endpoint=False
            )
            src.gain = target
            out += take * ramp
            if cloud_gain is not None:
                cloud += take * cloud_gain
                spoken = True
            truth.append({"kind": "live", "utterance": f"{LIVE_PREFIX}{who}", "db": round(db, 1)})
        return out, (cloud if spoken else None)

    # ── the duck's own voice ──────────────────────────────────────────────────

    def _render_own(
        self,
        clock: float,
        duck_pcm48: bytes | None,
        calls: Sequence[Mapping[str, Any]],
        acc: np.ndarray[Any, Any],
        truth: list[dict[str, Any]],
    ) -> bool:
        """Mix the duck's own voice into ``acc``; True when this frame is flagged as its echo."""
        for call in calls:
            self._start_call(clock, str(call.get("tag") or ""), call.get("level"))
        for echo in list(self._echoes):
            chunk = echo.wave[echo.pos : echo.pos + FRAME]
            acc[: len(chunk)] += chunk * echo.gain
            # (the window ends where the call's samples do, by this clock: the clock follows the
            # physics clock and so wanders against the samples, a frame of which is always 20 ms)
            echo.window.t1 = clock + (len(echo.wave) - echo.pos) / SR
            echo.pos += FRAME
            truth.append({"kind": "own_call", "utterance": None, "db": None})
            if echo.pos >= len(echo.wave):
                self._echoes.remove(echo)
        if duck_pcm48 is not None:
            x = self._decim.process(A.to_float(duck_pcm48).astype(np.float64))
            if len(x) < FRAME:
                x = np.concatenate([x, np.zeros(FRAME - len(x), dtype=np.float32)])
            x = x[:FRAME]
            if A.rms_dbfs(x) > STREAM_GATE_DBFS:  # (a theremin with no hand is silent: no echo)
                acc += x * 10.0 ** (SPEAKER_MIC_GAIN_DB / 20.0)
                self._stream_quiet = 0
                self._stream_end = clock + FRAME_S
                if self._stream_window is None:
                    self._stream_window = VocalWindow(clock, None, "stream", "stream", 1.0)
                    self._windows.append(self._stream_window)
                truth.append({"kind": "own_stream", "utterance": None, "db": None})
            else:
                self._quiet_stream()
        else:
            self._decim.reset()
            self._quiet_stream()
        return any(
            w.t0 < clock + FRAME_S and (w.t1 is None or clock < w.t1 + ECHO_TAIL_S)
            for w in self._windows
        )

    def _quiet_stream(self) -> None:
        """The voice plane had nothing audible this frame: after two such frames the stream's
        window ends where its last audible frame did."""
        window = self._stream_window
        if window is None:
            return
        self._stream_quiet += 1
        if self._stream_quiet >= 2:
            window.t1 = self._stream_end
            self._stream_window = None

    def _start_call(self, clock: float, tag: str, level: Any) -> None:
        """The duck voices ``tag`` now: a window for as long as the call lasts and, with the
        bank's wav for it, its echo in the microphone."""
        if not tag:
            return
        strength = 0.5 if level is None else min(1.0, max(0.1, float(level)))
        wave: np.ndarray[Any, Any] | None = None
        bank = self.calls
        if bank is not None:
            try:
                got = bank.call(tag, self._call_rng)
            except Exception as error:  # (no wav: the window still says the duck voiced)
                got = None
                if len(self.errors) < 20:
                    self.errors.append(f"call bank: {type(error).__name__}: {error}")
            if got is not None:
                wave = np.asarray(got[1], dtype=np.float32)
        seconds = len(wave) / SR if wave is not None else CALL_S.get(tag, 0.4)
        window = VocalWindow(clock, clock + seconds, tag, "call", strength)
        self._windows.append(window)
        if wave is not None and len(wave):
            gain = 10.0 ** (SPEAKER_MIC_GAIN_DB / 20.0) * strength
            self._echoes.append(_Echo(wave, 0, gain, window))

    # ── the room's own noise ──────────────────────────────────────────────────

    def _noise(self, motor_db: float | None) -> np.ndarray[Any, Any]:
        power = 10.0 ** (ROOM_DB / 10.0)
        if motor_db is not None:
            power += 10.0 ** (float(motor_db) / 10.0)
        sigma = 10.0 ** ((10.0 * math.log10(power) - A.FULL_SCALE_DB) / 20.0)
        rng = np.random.default_rng([self.seed, self.seq & 0xFFFFFFFF])
        return rng.standard_normal(FRAME) * sigma


def make_mic(
    *,
    seed: int = 7,
    engine: Any = None,
    library: UtteranceLibrary | None = None,
    use_default_library: bool = True,
) -> MicStream:
    """A microphone for one duck: the scripted lines of ``tests/fixtures/voice`` (or
    ``$DUCKMIND_VOICE_FIXTURES``) and, with a voice engine, the duck's own bank for its calls."""
    if library is None and use_default_library:
        try:
            library = UtteranceLibrary.default()
        except (OSError, ValueError, KeyError):
            library = None
    calls: CallBank | None = None
    bank = getattr(engine, "bank", None)
    if bank is not None:
        calls = VoiceBankCalls(bank, int(engine.voice_seed))
    return MicStream(seed=seed, library=library, calls=calls)


__all__ = [
    "CALL_S",
    "ECHO_TAIL_S",
    "FRAME",
    "SPEAKER_MIC_GAIN_DB",
    "CallBank",
    "MicFrame",
    "MicStream",
    "VocalWindow",
    "VoiceBankCalls",
    "make_mic",
]
