"""小鴨的長聲音: robotd's streamed voice in the simulator, and the room's music, as one PCM
stream for the page (a:sound 2026-09-28).

The robot's voice has two kinds of sound (upstream ``robotd/src/sound.rs``): one-shots — a wav of
the duck's bank, picked at random for a tag (the voice workflow's :mod:`duckmind.simweb.voice`
serves those to the page) — and **streams**, which hold the single-client PCM for as long as
they last:

* **the joy ride** ``wheee``: the bank's ``wheee_start_<x>`` → ``wheee_loop_<x>`` (repeated
  while the hold is fresh) → ``wheee_end_<x>`` segments written into one ``aplay`` pipe. Two
  exits: a hold that goes stale (no re-notify for 300 ms) *lands* the ride — the writer finishes
  the loop pass it is in, then writes the end segment — and an explicit release *cuts* it;
* **the ToF theremin**: a live ``sounds::Stream::wheee(personality, 0)``, 10-ms blocks, whose
  pitch, level and beak opening the 50 Hz control loop sets from the hand's distance (tofd's raw
  8×8 frames through ``kinematics::hand``; closeness 0 far … 1 near is the pitch
  (``sounds::stream::hz_at``, the duck's own register), the level (1 with a hand, 0 without —
  a fade at the last pitch) and the beak's opening, one number).

The simulator's robotd has no sound card, so this module does robotd's part with robotd's own
rules, tick for tick on the physics clock (:class:`RobotdVoice`, :class:`Theremin`,
:class:`HandTracker` — ports of ``sound.rs``, ``theremin.rs`` and ``kinematics::hand``), and the
sound itself comes from upstream's code: the theremin's samples from our helper
``duck-voice-stream`` (``tools/voicestream``: upstream's ``sounds`` crate linked by path, the
same ``Stream``, the same ``hz_at``, the same i16 conversion), the ride's from the duck's own
bank (the voice workflow's cache, rendered by upstream's ``sounds`` binary) or, when that bank is
not there, from the helper's ``segments`` (the very recipe the bank's files are rendered with —
byte for byte the same). Nothing is re-synthesised here.

**The room** is the world's music (:class:`~duckmind.world.neuro_music.Song`, the song the
world plays: the section ``neuro_sound`` says its tempo, seed and start), rendered on the same
physics clock.

**The stream** (:class:`SoundStream`, ``GET /api/neuro/sound/pcm?since=<seq>``): one binary frame
per robotd tick (20 ms, 960 samples at 48 kHz), each stamped with the server's wall clock and
the physics time of its first sample (the page plays it on the same clock as the body samples,
so the beak, the head and the sound come together), carrying the duck's plane (16-bit PCM,
only while a stream sounds) and the room's plane (only while music plays)::

    0  4s  magic b"DKSD"
    4  u32 seq
    8  f64 w          server wall clock (epoch s) at the first sample
    16 f64 sim_s      physics seconds at the first sample
    24 u16 n          samples per plane (960)
    26 u8  flags      bit 0: the duck's plane follows, bit 1: the room's plane follows
    27 u8  marks      how many 4-byte marks follow
    28 marks × (u16 sample offset, u8 kind 1 = a beat of the music / 2 = its bar's first, u8 0)
       then the duck's i16 × n (bit 0), then the room's i16 × n (bit 1)   (all little-endian)

``GET /api/neuro/sound/status`` says, in numbers and plain words, what streams now and why.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import random
import struct
import subprocess
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.parse import parse_qs

from duckmind import paths
from duckmind.i18n import msg

SR = 48_000
TICK_S = 0.02  # robotd's control loop
TICK_SAMPLES = int(SR * TICK_S)
SYNTH_BLOCK = SR // 100  # robotd sound.rs: the theremin writer renders 10 ms blocks
WHEEE_HOLD_FRESH_S = 0.3  # robotd intents.rs: a wheee hold older than this has decayed
FRAME_STALE_S = 0.5  # robotd theremin.rs: a depth frame older than this is silence
HELPER_ENV = "DUCKMIND_VOICE_STREAM_BIN"
HELPER_IN_DATA = Path("voicestream") / "target" / "release" / "duck-voice-stream"
IDENTITY = "duckmind-duck:{seed}"  # voice.py's: a simulated duck's hardware identity
MAGIC = b"DKSD"
HEADER = struct.Struct("<4sIddHBB")
MARK = struct.Struct("<HBB")
KEEP_CHUNKS = 400  # the stream's ring: 8 s
STREAM_MAX_S = 300.0  # a stream ends after this; the page asks again with ?since=
STREAM_MAX = 4  # at most this many open streams
ROUTE = "/api/neuro/sound"

NO_HELPER_ZH = msg("neuro.incoming.incoming_simweb_neuro_voice_stream.NO_HELPER_ZH")


# ── the helper (upstream's sounds::Stream behind a pipe) ─────────────────────────────


class HelperError(RuntimeError):
    pass


def helper_path() -> Path | None:
    """Our ``duck-voice-stream`` binary: ``$DUCKMIND_VOICE_STREAM_BIN``, else the default build
    folder (``voicestream/target/release`` in :func:`duckmind.paths.data_dir`); None when it is not
    built."""
    raw = os.environ.get(HELPER_ENV)
    path = Path(raw).expanduser() if raw else paths.data_dir() / HELPER_IN_DATA
    return path if path.is_file() and os.access(path, os.X_OK) else None


class VoiceHelper:
    """One ``duck-voice-stream`` process: upstream's ``sounds::Stream`` held open (see the helper's
    own docs for the protocol). Not thread-safe: one owner."""

    def __init__(self, path: Path | None = None) -> None:
        path = path or helper_path()
        if path is None:
            raise HelperError(NO_HELPER_ZH)
        self.path = path
        self.proc = subprocess.Popen(
            [str(path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        self.range_hz: tuple[float, float] | None = None
        self.pitch_center_hz: float | None = None
        self.seed: int | None = None

    def _line(self) -> str:
        assert self.proc.stdout is not None
        raw = self.proc.stdout.readline()
        if not raw:
            raise HelperError("the voice helper stopped")
        return raw.decode().rstrip("\n")

    def _read(self, n: int) -> bytes:
        assert self.proc.stdout is not None
        out = bytearray()
        while len(out) < n:
            part = self.proc.stdout.read(n - len(out))
            if not part:
                raise HelperError("the voice helper stopped mid-block")
            out.extend(part)
        return bytes(out)

    def ask(self, command: str) -> str:
        """One command; the answer after ``ok`` (HelperError on ``err``)."""
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write((command + "\n").encode())
        except (BrokenPipeError, OSError) as error:
            raise HelperError(f"the voice helper is gone ({error})") from error
        line = self._line()
        if line.startswith("err"):
            raise HelperError(line[4:])
        return line[3:] if line.startswith("ok ") else ""

    def voice(self, seed: int, kind: str = "wheee", variant: int = 0) -> None:
        words = self.ask(f"voice {int(seed)} {kind} {int(variant)}").split()
        self.range_hz = (float(words[0]), float(words[1]))
        self.pitch_center_hz = float(words[2])
        self.seed = int(seed)

    def hz_at(self, position: float) -> float:
        return float(self.ask(f"hz {position!r}"))

    def set(self, hz: float, level: float, open_: float) -> None:
        self.ask(f"set {float(hz)!r} {float(level)!r} {float(open_)!r}")

    def block(self, n: int) -> bytes:
        size = int(self.ask(f"block {int(n)} i16"))
        return self._read(size)

    def block_f32(self, n: int) -> bytes:
        size = int(self.ask(f"block {int(n)} f32"))
        return self._read(size)

    def silent(self) -> bool:
        return self.ask("silent").strip() == "1"

    def segments(self, seed: int, variant: int) -> tuple[bytes, bytes, bytes]:
        sizes = [int(v) for v in self.ask(f"segments {int(seed)} {int(variant)}").split()]
        return self._read(sizes[0]), self._read(sizes[1]), self._read(sizes[2])

    def seed_from_id(self, text: str) -> int:
        return int(self.ask(f"seed_from_id {text}"))

    def close(self) -> None:
        proc = self.proc
        try:
            if proc.stdin is not None:
                proc.stdin.write(b"quit\n")
                proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()


# ── the voice workflow's module (seed mapping, bank cache) ──────────────────────────────


def voice_module() -> ModuleType | None:
    """:mod:`duckmind.simweb.voice` (the voice workflow's: the seed mapping and the bank cache);
    None when it cannot be imported."""
    try:
        return importlib.import_module("duckmind.simweb.voice")
    except ImportError:
        return None


def voice_seed(duck_seed: int, module: ModuleType | None = None) -> int:
    """The duck's voice seed: the voice workflow's mapping (upstream's ``seed_from_id`` of the
    duck's identity), so the streamed voice is the same duck as its one-shot calls."""
    if module is not None and hasattr(module, "voice_seed"):
        return int(module.voice_seed(int(duck_seed)))
    import hashlib

    digest = hashlib.sha256(IDENTITY.format(seed=int(duck_seed)).encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _wav_payload(path: Path) -> bytes | None:
    """The data chunk of a 16-bit mono wav (robotd's ``read_wav_pcm``)."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        return None
    pos = 12
    rate = 0
    while pos + 8 <= len(raw):
        size = int.from_bytes(raw[pos + 4 : pos + 8], "little")
        body = pos + 8
        tag = raw[pos : pos + 4]
        if tag == b"fmt " and body + 8 <= len(raw):
            rate = int.from_bytes(raw[body + 4 : body + 8], "little")
        elif tag == b"data" and rate > 0:
            return raw[body : min(body + size, len(raw))]
        pos = body + size + (size & 1)
    return None


# ── kinematics::hand (the nearest thing in front of the beak, as an instrument wants it) ──


@dataclass(frozen=True)
class HandConfig:
    """``kinematics::hand::Config::default()`` = robotd-params ``[theremin]`` defaults."""

    near_m: float = 0.10
    far_m: float = 0.70
    min_zones: int = 2
    statuses: tuple[int, ...] = (4, 5, 6, 9, 10, 12, 13)
    hold_s: float = 0.25

    def believes(self, status: int) -> bool:
        return int(status) in self.statuses


@dataclass(frozen=True)
class Hand:
    range_m: float
    closeness: float  # 0 at far_m … 1 at near_m
    zones: int
    held: bool  # the held memory of an earlier frame


class HandTracker:
    """``kinematics::hand::Tracker``: the hand in a frame, or the one from a frame just gone."""

    def __init__(self, config: HandConfig | None = None) -> None:
        self.config = config or HandConfig()
        self.last: tuple[Hand, float] | None = None

    def track(self, distance_mm: Sequence[int], status: Sequence[int], now_s: float) -> Hand | None:
        cfg = self.config
        in_band: list[float] = []
        for zone in range(min(64, len(distance_mm), len(status))):
            mm, code = int(distance_mm[zone]), int(status[zone])
            if mm <= 0 or not cfg.believes(code):
                continue
            rng = mm / 1000.0
            if cfg.near_m <= rng <= cfg.far_m:
                in_band.append(rng)
        if len(in_band) < max(1, cfg.min_zones):
            if self.last is not None and now_s - self.last[1] < cfg.hold_s:
                hand = self.last[0]
                return Hand(hand.range_m, hand.closeness, hand.zones, True)
            self.last = None
            return None
        in_band.sort()
        range_m = in_band[len(in_band) // 5]
        span = max(1e-6, cfg.far_m - cfg.near_m)
        hand = Hand(range_m, min(1.0, max(0.0, (cfg.far_m - range_m) / span)), len(in_band), False)
        self.last = (hand, now_s)
        return hand

    def reset(self) -> None:
        self.last = None


def status_histogram(status: Sequence[int]) -> list[tuple[int, int]]:
    counts: dict[int, int] = {}
    for code in list(status)[:64]:
        counts[int(code)] = counts.get(int(code), 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def describe(status: Sequence[int], config: HandConfig) -> str:
    """robotd theremin.rs ``describe``: how many zones each status covers, believed ones starred."""
    hist = status_histogram(status)
    believed = sum(n for code, n in hist if config.believes(code))
    codes = " ".join(f"{code}{'*' if config.believes(code) else ''}:{n}" for code, n in hist)
    return f"{believed} usable · {codes}"


# ── robotd theremin.rs ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Note:
    mouth: float  # the beak's opening 0..1 (robotd drives the mouth target with it)
    closeness: float | None  # None: silence
    range_m: float | None
    zones: int
    held: bool
    sensor: str


class Theremin:
    """``theremin.rs`` ``Theremin``: up or down; ``tick`` turns the newest depth frame into a
    note (None while down)."""

    def __init__(self, config: HandConfig | None = None) -> None:
        self.tracker = HandTracker(config)
        self.active = False

    def set_active(self, active: bool) -> None:
        self.active = bool(active)
        self.tracker.reset()  # held notes do not survive being put down

    def tick(
        self, frame: Mapping[str, Any] | None, frame_age_s: float | None, now_s: float
    ) -> Note | None:
        if not self.active:
            return None
        fresh = frame is not None and frame_age_s is not None and frame_age_s < FRAME_STALE_S
        if not fresh:
            self.tracker.reset()
            return Note(0.0, None, None, 0, False, "no depth frames")
        assert frame is not None
        distance = [int(v) for v in frame.get("distance_mm") or ()]
        status = [int(v) for v in frame.get("status") or ()]
        hand = self.tracker.track(distance, status, now_s)
        closeness = None if hand is None else hand.closeness
        return Note(
            mouth=closeness or 0.0,
            closeness=closeness,
            range_m=None if hand is None else hand.range_m,
            zones=0 if hand is None else hand.zones,
            held=bool(hand and hand.held),
            sensor=describe(status, self.tracker.config),
        )


# ── robotd sound.rs (the streamed half) ────────────────────────────────────────────


class RobotdVoice:
    """``sound.rs`` ``Sound`` for the streams, one robotd tick at a time: :meth:`wheee` with the
    hold's level, the theremin's start / set / stop / settle, and :meth:`render` for the samples
    the PCM plays this tick (None: nothing holds it). ``segments(letter)`` gives a ride's
    (start, loop, end) PCM; ``letters`` the variants the bank has."""

    def __init__(
        self,
        voice_seed: int,
        *,
        helper: VoiceHelper | None,
        segments: Any,
        letters: Sequence[str],
        rng: random.Random | None = None,
    ) -> None:
        self.voice_seed = int(voice_seed)
        self.helper = helper
        self._segments = segments
        self.letters = list(letters)
        self.rng = rng or random.Random(self.voice_seed)
        self.ride = "off"  # off | riding | landing | theremin
        # the ride's writer: the three segments, where it is, and whether it is held
        self._pcm: tuple[bytes, bytes, bytes] | None = None
        self._part = ""
        self._pos = 0
        self._held = False
        self.letter = ""
        # the theremin's writer
        self._live: list[float] | None = None  # hz, level, open (as the loop last stored them)
        self._playing = False
        self._theremin_voice = False
        self.skipped = 0  # one-shots dropped because a stream had the PCM (robotd's rule)
        self.rides = 0

    # the ride

    def wheee(self, hold: str) -> None:
        """``Sound::wheee``: ``hold`` is ``"held"`` | ``"released"`` | ``"decayed"``."""
        if self.ride == "theremin":
            return  # the instrument keeps the PCM; the trigger is inaudible while it does
        if hold == "held" and self.ride != "riding":
            self._start_wheee()
        elif hold == "released" and self.ride != "off":
            self._stop()
        elif hold == "decayed" and self.ride == "riding":
            self._held = False  # land: the writer finishes its loop pass, then the end
            self.ride = "landing"

    def _start_wheee(self) -> None:
        self._stop()
        if not self.letters:
            return
        letter = self.letters[self.rng.randrange(len(self.letters))]
        pcm = self._segments(letter)
        if pcm is None:
            return
        self._pcm = pcm
        self.letter = letter
        self._part, self._pos, self._held = "start", 0, True
        self.ride = "riding"
        self.rides += 1

    def _stop(self) -> None:
        self._pcm = None
        self._part, self._pos, self._held = "", 0, False
        self._live = None
        self._playing = False
        # (the writer is gone, whichever it was: a later landing is the ride's, reaped when
        # its end is written — not a faded-out instrument's)
        self._theremin_voice = False
        self.ride = "off"

    # the theremin

    def theremin_start(self) -> bool:
        if self.ride == "theremin" and self._live is not None:
            return True
        helper = self.helper
        if helper is None:
            return False
        self._stop()
        # a fresh Stream::wheee(personality, 0) each time the instrument is picked up
        helper.voice(self.voice_seed, "wheee", 0)
        self._live = [0.0, 0.0, 0.0]
        self._playing = True
        self._theremin_voice = True
        self.ride = "theremin"
        return True

    def theremin_set(self, hz: float, level: float, open_: float) -> None:
        if self._live is not None:
            self._live[:] = [float(hz), float(level), float(open_)]

    def theremin_stop(self) -> None:
        if self.ride != "theremin":
            return
        self._playing = False  # the writer fades the note out, then lets the PCM go
        self.ride = "landing"

    def theremin_settle(self) -> None:
        if self.ride == "landing" and self._theremin_voice and self._live is None:
            self.ride = "off"
            self._theremin_voice = False

    def theremin_hz_at(self, closeness: float) -> float | None:
        if self.helper is None:
            return None
        if self.helper.seed != self.voice_seed:
            self.helper.voice(self.voice_seed, "wheee", 0)
        return self.helper.hz_at(closeness)

    def owns_pcm(self) -> bool:
        return self.ride != "off"

    # the samples

    def render(self, n: int = TICK_SAMPLES) -> bytes | None:
        if self._theremin_voice and self._live is not None:
            return self._render_theremin(n)
        if self._pcm is not None:
            return self._render_ride(n)
        return None

    def _render_theremin(self, n: int) -> bytes | None:
        helper = self.helper
        assert helper is not None and self._live is not None
        out = bytearray()
        left = n
        while left > 0:
            block = min(SYNTH_BLOCK, left)
            hz, level, open_ = self._live
            helper.set(hz, level if self._playing else 0.0, open_)
            if not self._playing and helper.silent():
                # the fade reached silence: the writer returns, aplay drains, the PCM is free
                self._live = None
                out.extend(bytes(2 * left))
                break
            out.extend(helper.block(block))
            left -= block
        return bytes(out)

    def _render_ride(self, n: int) -> bytes:
        assert self._pcm is not None
        start, loop, end = self._pcm
        want = 2 * n
        out = bytearray()
        while len(out) < want and self._pcm is not None:
            part = {"start": start, "loop": loop, "end": end}[self._part]
            take = min(want - len(out), len(part) - self._pos)
            out.extend(part[self._pos : self._pos + take])
            self._pos += take
            if self._pos < len(part):
                continue
            # a segment ends: start → the loop; the loop again while held, else the end
            if self._part == "start" or (self._part == "loop" and self._held):
                self._part, self._pos = "loop", 0
                if not loop:
                    self._part = "end"
            elif self._part == "loop":
                self._part, self._pos = "end", 0
            else:
                self._stop()  # the end is written: aplay drains, the PCM is free
        if len(out) < want:
            out.extend(bytes(want - len(out)))
        return bytes(out)


# ── the room's music ───────────────────────────────────────────────────────────────


class RoomMusic:
    """The world's song, on the physics clock (the section ``neuro_sound``'s ``music``).

    The song plays on continuously, 20 ms a tick. Each tick's physics time is read with a jitter
    (live: up to ±16 ms around the tick grid, measured 2026-09-28), so the stream follows the
    physics clock through a smoothed estimate of how far it is off (the mean of the last ~50
    readings), never through one reading: before the song's first sample (the world starts it
    0.3 s after the press) it lines up in the silence; while it plays it slews — one sample a
    tick repeated or skipped (0.1 %, two when more than 20 ms off) — when it is more than 10 ms
    off. When the physics clock itself stepped (the median of the last 11 readings more than
    30 ms off: a stall of the physics, measured live at ~50 ms under load) it lines up at once,
    one jump, rather than playing that far off for half a minute; a reading a quarter of a
    second off is a jump at once."""

    SMOOTH_TICKS = 50
    SLEW_AT = 480  # samples (10 ms)
    STEP_AT = 1440  # samples (30 ms), on the median of the last STEP_READINGS readings
    STEP_READINGS = 11
    JUMP_AT = SR // 4

    def __init__(self) -> None:
        self.spec: tuple[float, int, float] | None = None  # bpm, seed, t0
        self.song: Any = None
        self.next_sample: int | None = None  # the song sample the next chunk starts at
        self.off = 0.0  # the smoothed (stream − physics) position, samples
        self.readings = 0
        self.recent: deque[float] = deque(maxlen=self.STEP_READINGS)
        self.slews = 0
        self.jumps = 0
        self.last_start: int | None = None  # the song sample the last chunk started at

    def _shift(self, shift: int) -> None:
        """The stream moved by ``shift`` samples: the readings so far are that much less off."""
        self.off -= shift
        self.recent = deque((e - shift for e in self.recent), maxlen=self.STEP_READINGS)

    def set(self, view: Mapping[str, Any] | None) -> None:
        if not view or not view.get("on"):
            self.spec = None
            self.song = None
            self.next_sample = None
            return
        spec = (float(view["bpm"]), int(view["seed"]), float(view["t0"]))
        if spec != self.spec:
            from duckmind.world.neuro_music import Song

            self.spec = spec
            self.song = Song(bpm=spec[0], seed=spec[1])
            self.next_sample = None

    def render(self, sim_s: float, n: int) -> tuple[bytes | None, list[tuple[int, int]]]:
        """``n`` samples of the song from physics second ``sim_s`` (continuous from the last
        chunk, following the physics clock as the class says), and the beats inside them."""
        song, spec = self.song, self.spec
        self.last_start = None
        if song is None or spec is None:
            return None, []
        want = int(round((sim_s - spec[2]) * SR))
        start = self.next_sample
        if start is None:
            start, self.off, self.readings = want, 0.0, 0
            self.recent.clear()
        else:
            err = start - want
            if abs(err) > self.JUMP_AT:
                start, self.off, self.readings = want, 0.0, 0  # the physics clock jumped
                self.recent.clear()
                self.jumps += 1
            else:
                self.readings += 1
                self.recent.append(float(err))
                self.off += (err - self.off) / min(self.readings, self.SMOOTH_TICKS)
                ordered = sorted(self.recent)
                median = ordered[len(ordered) // 2]
                if start < 0:
                    # (still silence before the song: line up with the mean so far, for free)
                    shift = int(round(self.off))
                    start -= shift
                    self._shift(shift)
                elif len(ordered) == self.STEP_READINGS and abs(median) > self.STEP_AT:
                    shift = int(round(median))
                    start -= shift  # one jump to where the physics clock is now
                    self._shift(shift)
                    self.off = sorted(self.recent)[len(self.recent) // 2]
                    self.jumps += 1
                elif abs(self.off) > self.SLEW_AT:
                    step = 2 if abs(self.off) > 2 * self.SLEW_AT else 1
                    step = step if self.off > 0 else -step
                    start -= step  # ahead: samples repeated; behind: skipped
                    self._shift(step)
                    self.slews += 1
        self.next_sample = start + n
        self.last_start = start
        if start + n <= 0:
            return None, []
        pcm = song.pcm16(max(0, start), n - max(0, -start))
        if start < 0:
            pcm = bytes(2 * -start) + pcm
        marks = []
        for t, k in song.beats(start / SR, (start + n) / SR):
            offset = int(round(t * SR)) - start
            if 0 <= offset < n:
                marks.append((offset, 2 if k % 4 == 0 else 1))
        return pcm, marks


# ── the stream ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Chunk:
    seq: int
    w: float
    sim_s: float
    n: int
    duck: bytes | None
    room: bytes | None
    marks: tuple[tuple[int, int], ...] = ()

    def encode(self) -> bytes:
        flags = (1 if self.duck is not None else 0) | (2 if self.room is not None else 0)
        head = HEADER.pack(MAGIC, self.seq, self.w, self.sim_s, self.n, flags, len(self.marks))
        body = b"".join(MARK.pack(offset, kind, 0) for offset, kind in self.marks)
        return head + body + (self.duck or b"") + (self.room or b"")


def decode(buffer: bytes) -> tuple[list[Chunk], int]:
    """The whole frames at the start of ``buffer`` and how many bytes they took (tests, tools)."""
    out: list[Chunk] = []
    pos = 0
    while pos + HEADER.size <= len(buffer):
        magic, seq, w, sim_s, n, flags, nmarks = HEADER.unpack_from(buffer, pos)
        if magic != MAGIC:
            raise ValueError("not a sound frame")
        size = HEADER.size + MARK.size * nmarks + 2 * n * ((flags & 1) + ((flags >> 1) & 1))
        if pos + size > len(buffer):
            break
        at = pos + HEADER.size
        marks = []
        for _ in range(nmarks):
            offset, kind, _pad = MARK.unpack_from(buffer, at)
            marks.append((offset, kind))
            at += MARK.size
        duck = room = None
        if flags & 1:
            duck = buffer[at : at + 2 * n]
            at += 2 * n
        if flags & 2:
            room = buffer[at : at + 2 * n]
        out.append(Chunk(seq, w, sim_s, n, duck, room, tuple(marks)))
        pos += size
    return out, pos


class SoundStream:
    """The chunks of the last few seconds, for the page's streams."""

    def __init__(self) -> None:
        self.chunks: deque[Chunk] = deque(maxlen=KEEP_CHUNKS)
        self.seq = 0
        self.cond = threading.Condition()
        self.clients = 0
        self.sent = 0

    def push(
        self,
        w: float,
        sim_s: float,
        n: int,
        duck: bytes | None,
        room: bytes | None,
        marks: Sequence[tuple[int, int]] = (),
    ) -> Chunk:
        with self.cond:
            self.seq += 1
            chunk = Chunk(self.seq, w, sim_s, n, duck, room, tuple(marks))
            self.chunks.append(chunk)
            self.cond.notify_all()
        return chunk

    def since(self, seq: int) -> list[Chunk]:
        with self.cond:
            return [c for c in self.chunks if c.seq > seq]

    def wait(self, seq: int, timeout: float) -> list[Chunk]:
        with self.cond:
            if self.seq <= seq:
                self.cond.wait(timeout)
            return [c for c in self.chunks if c.seq > seq]


# ── one tick of robotd's sound, and the engine the driver calls ────────────────────


@dataclass
class TickOut:
    """What robotd's sound did this tick (for the body link, the page and the evaluators)."""

    mouth: float | None  # the theremin's beak (robotd drives the mouth target with it)
    note_hz: float | None
    closeness: float | None
    hand_mm: int | None
    ride: str  # off | riding | landing | theremin
    owns_pcm: bool  # a stream has the PCM: one-shot calls are dropped (robotd's rule)
    chunk: Chunk | None


@dataclass
class VoiceEngine:
    """robotd's streamed voice for one duck, tick for tick (see the module docstring).

    ``duck_seed``: the duck's Duck Core seed (its voice seed is the voice workflow's mapping of
    it); ``microduck``: the official checkout (the voice workflow's bank renders with its
    ``sounds`` binary). ``stream``: where the chunks go (None: rendered but not kept)."""

    duck_seed: int
    microduck: Path | None = None
    stream: SoundStream | None = None
    helper_bin: Path | None = None
    rng_seed: int | None = None
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.voice_mod = voice_module()
        self.voice_seed = voice_seed(self.duck_seed, self.voice_mod)
        self.helper: VoiceHelper | None = None
        try:
            self.helper = VoiceHelper(self.helper_bin)
        except (HelperError, OSError) as error:
            self.notes.append(str(error))
        self.bank: Any = None
        if self.voice_mod is not None and hasattr(self.voice_mod, "VoiceBank"):
            try:
                self.bank = self.voice_mod.VoiceBank.from_microduck(self.microduck)
                self.bank.prepare(self.duck_seed)
            except Exception as error:  # (the bank is a cache: the helper renders the same)
                self.notes.append(f"voice bank: {type(error).__name__}: {error}")
                self.bank = None
        letters = [chr(ord("a") + i) for i in range(6)]  # sounds::variant_count("wheee")
        rng = random.Random(self.rng_seed if self.rng_seed is not None else time.time_ns())
        self.voice = RobotdVoice(
            self.voice_seed,
            helper=self.helper,
            segments=self._segments,
            letters=letters,
            rng=rng,
        )
        self.theremin = Theremin()
        self.theremin_hz = 0.0  # robotd's loop keeps the last pitch (a fade at the note)
        self.room = RoomMusic()
        self._wheee: tuple[bool, float] = (False, -1e9)  # robotd's stamped hold level
        self._want_theremin = False
        self._seg_cache: dict[str, tuple[bytes, bytes, bytes]] = {}
        self.seg_source = ""
        self.ticks = 0
        self.last: TickOut | None = None

    # the ride's segments: the bank's files (what robotd reads), else the same recipe

    def _segments(self, letter: str) -> tuple[bytes, bytes, bytes] | None:
        if letter in self._seg_cache:
            return self._seg_cache[letter]
        pcm: tuple[bytes, bytes, bytes] | None = None
        bank = self.bank
        if bank is not None:
            try:
                if bank.ready(self.voice_seed):
                    folder = Path(bank.bank_dir(self.voice_seed)) / "wheee"
                    parts = [
                        _wav_payload(folder / f"wheee_{p}_{letter}.wav")
                        for p in ("start", "loop", "end")
                    ]
                    if all(p is not None for p in parts):
                        pcm = (parts[0] or b"", parts[1] or b"", parts[2] or b"")
                        self.seg_source = "bank"
            except Exception:
                pcm = None
        if pcm is None and self.helper is not None:
            variant = ord(letter) - ord("a")
            try:
                pcm = self.helper.segments(self.voice_seed, variant)
                self.seg_source = "helper"
            except HelperError:
                pcm = None
        if pcm is not None:
            self._seg_cache[letter] = pcm
        return pcm

    def _hold(self, t_s: float) -> str:
        value, at = self._wheee
        if not value:
            return "released"
        return "held" if t_s - at < WHEEE_HOLD_FRESH_S else "decayed"

    def tick(
        self,
        *,
        t_s: float,
        sim_s: float,
        w: float,
        wheee_hold: bool,
        theremin_up: bool,
        tof: Mapping[str, Any] | None,
        tof_age_s: float | None,
        room: Mapping[str, Any] | None = None,
        release: bool = False,
    ) -> TickOut:
        """One robotd tick at ``t_s`` (robotd's clock, s): the brain's wheee hold (re-notified
        every tick while true; ``release``: an explicit ``hold: false``, the cut), its theremin
        level (a change is a ``robot.theremin`` request), tofd's newest frame and its age, and
        the world's music (the section ``neuro_sound``'s ``music``)."""
        voice = self.voice
        if wheee_hold:
            self._wheee = (True, t_s)
        elif release:
            self._wheee = (False, t_s)
        voice.theremin_settle()
        voice.wheee(self._hold(t_s))
        # the theremin (robotd main.rs: a request edge, then the tick)
        if theremin_up != self._want_theremin:
            self._want_theremin = theremin_up
            self.theremin.set_active(theremin_up)
        note = self.theremin.tick(tof, tof_age_s, t_s)
        mouth: float | None = None
        closeness = None
        hand_mm = None
        if note is not None:
            level = 0.0
            if note.closeness is not None:
                hz = voice.theremin_hz_at(note.closeness)
                if hz is not None:
                    self.theremin_hz = hz
                level = 1.0
            if voice.theremin_start():
                voice.theremin_set(self.theremin_hz, level, note.mouth)
            mouth = note.mouth
            closeness = note.closeness
            hand_mm = None if note.range_m is None else int(round(note.range_m * 1000))
        else:
            voice.theremin_stop()
        duck = voice.render(TICK_SAMPLES)
        self.room.set(room)
        room_pcm, marks = self.room.render(sim_s, TICK_SAMPLES)
        chunk = None
        if self.stream is not None:
            chunk = self.stream.push(w, sim_s, TICK_SAMPLES, duck, room_pcm, marks)
        self.ticks += 1
        out = TickOut(
            mouth=mouth,
            note_hz=self.theremin_hz if note is not None and note.closeness is not None else None,
            closeness=closeness,
            hand_mm=hand_mm,
            ride=voice.ride,
            owns_pcm=voice.owns_pcm(),
            chunk=chunk,
        )
        self.last = out
        return out

    def close(self) -> None:
        if self.helper is not None:
            self.helper.close()
            self.helper = None

    def view(self) -> dict[str, Any]:
        last = self.last
        helper = self.helper
        return {
            "duck_seed": self.duck_seed,
            "voice_seed": self.voice_seed,
            "helper": None if helper is None else str(helper.path),
            "range_hz": None
            if helper is None or helper.range_hz is None
            else [round(v, 1) for v in helper.range_hz],
            "ride_source": self.seg_source or None,
            "voice_module": None
            if self.voice_mod is None
            else getattr(self.voice_mod, "__file__", None),
            "notes": list(self.notes),
            "ride": None if last is None else last.ride,
            "owns_pcm": bool(last and last.owns_pcm),
            "note_hz": None if last is None or last.note_hz is None else round(last.note_hz, 1),
            "closeness": None
            if last is None or last.closeness is None
            else round(last.closeness, 3),
            "hand_mm": None if last is None else last.hand_mm,
            "letter": self.voice.letter or None,
            "rides": self.voice.rides,
            "music": None
            if self.room.spec is None
            else {
                "bpm": self.room.spec[0],
                "off_ms": round(self.room.off / SR * 1000.0, 1),
                "slews": self.room.slews,
                "jumps": self.room.jumps,
            },
            "ticks": self.ticks,
        }


# ── the page's routes ─────────────────────────────────────────────────────────────


class SoundRoutes:
    """``/api/neuro/sound/…`` for the page (GET): ``pcm`` (the stream) and ``status``."""

    def __init__(self, stream: SoundStream, status: Any) -> None:
        self.stream = stream
        self.status = status  # () -> dict

    @staticmethod
    def handles(path: str) -> bool:
        return path == ROUTE or path.startswith(ROUTE + "/")

    def get(self, handler: BaseHTTPRequestHandler, path: str, query: str) -> None:
        if path == ROUTE + "/pcm":
            raw = parse_qs(query).get("since", ["0"])[0]
            since = int(raw) if raw.isdigit() else 0
            self._pcm(handler, since)
            return
        if path in (ROUTE, ROUTE + "/status"):
            body = json.dumps({"ok": True, **self.status()}, ensure_ascii=False).encode()
            handler.send_response(HTTPStatus.OK)
            handler.send_header("Content-Type", "application/json; charset=utf-8")
            handler.send_header("Content-Length", str(len(body)))
            handler.send_header("Cache-Control", "no-store")
            handler.end_headers()
            handler.wfile.write(body)
            return
        handler.send_response(HTTPStatus.NOT_FOUND)
        handler.send_header("Content-Length", "0")
        handler.end_headers()

    def _pcm(self, handler: BaseHTTPRequestHandler, since: int) -> None:
        stream = self.stream
        if stream.clients >= STREAM_MAX:
            handler.send_response(HTTPStatus.SERVICE_UNAVAILABLE)
            handler.send_header("Content-Length", "0")
            handler.end_headers()
            return
        stream.clients += 1
        try:
            handler.send_response(HTTPStatus.OK)
            handler.send_header("Content-Type", "application/octet-stream")
            handler.send_header("Cache-Control", "no-store")
            handler.send_header("X-Accel-Buffering", "no")
            handler.send_header("X-Content-Type-Options", "nosniff")
            handler.end_headers()
            handler.close_connection = True
            seq = since if since > 0 else stream.seq
            if since > 0 and stream.chunks and stream.chunks[0].seq > since + 1:
                seq = stream.chunks[0].seq - 1  # (too old: from the oldest kept)
            ends = time.monotonic() + STREAM_MAX_S
            while time.monotonic() < ends:
                chunks = stream.wait(seq, 0.5)
                if not chunks:
                    continue
                handler.wfile.write(b"".join(c.encode() for c in chunks))
                handler.wfile.flush()
                stream.sent += len(chunks)
                seq = chunks[-1].seq
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            stream.clients -= 1


def rms_db(pcm: bytes | None) -> float:
    """A chunk's level in dBFS (evaluators)."""
    if not pcm:
        return -120.0
    count = len(pcm) // 2
    values = struct.unpack(f"<{count}h", pcm[: 2 * count])
    power = sum(v * v for v in values) / max(1, count)
    return 10.0 * math.log10(max(power, 1e-9) / (32767.0 * 32767.0))


__all__ = [
    "Chunk",
    "Hand",
    "HandConfig",
    "HandTracker",
    "HelperError",
    "Note",
    "RobotdVoice",
    "RoomMusic",
    "SoundRoutes",
    "SoundStream",
    "Theremin",
    "TickOut",
    "VoiceEngine",
    "VoiceHelper",
    "decode",
    "describe",
    "helper_path",
    "rms_db",
    "voice_module",
    "voice_seed",
]
