"""The live voice library of Duck Core's voice lineage (docs/design/GROWTH-G5-2026-10-11.md).

:mod:`duckmind.life.voice_lineage` turns what Duck Core remembers of how the duck learned to call
(its phrases, the ones the family praised, its song, its age) into a **plan**. This module makes
that plan sound and puts it in the simulator's speaker:

* :func:`render` shapes clips of the duck's own voice bank (upstream's official Microduck synth,
  rendered once per voice seed by :mod:`duckmind.simweb.voice`) into the plan's clips: the pitch
  and length of its age, the variants it owns, its learned phrases and its song, syllable after
  syllable. Pure numpy (a resampler and a WSOLA time stretch). Nothing else is used: no recording,
  no text-to-speech, nothing sent anywhere. The library is a folder named by the plan's key
  (``<voices>/lineage/seed<voice seed>/<key>/``), complete or absent, never half there.
* :class:`LineageVoice` keeps the library of the duck on screen: it **builds a new one when the
  lineage changes** (at boot, at a day change, or when :meth:`LineageVoice.refresh` is called, for
  example by the nightly regrow) on a background thread, and **swaps it in only between calls**:
  a call that is sounding, or about to, is never cut or changed. It tells the page what it has
  (the manifest of ``/api/voice/manifest`` gets the lineage's clips in place of the bank's four
  reshaped calls, and ``ph_<code>`` and ``song`` as calls of their own), serves the clips, and
  says what it did in ``/api/life``'s ``voice``.
* :meth:`LineageVoice.shape` is the one place a brain's call becomes a lineage call: a ``little_song``
  sings Duck Core's song; ``call_out``, ``call_response`` and ``self_duet`` say one of the phrases it
  learned; every other call stays the call the brain chose, in the library's reshaped clips. An
  ``alarm`` (a fall), a ``peck``, a ``wheee`` ride and the emotions are the bank's own.

Duck Core is only read. The same lineage state gives the same plan, the same key and the same
files; a changed rule number or Duck Core content is a new key (the version guard of the plan).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import threading
import time
import wave
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from http import HTTPStatus
from pathlib import Path
from typing import Any

import numpy as np

from duckmind.i18n import msg
from duckmind.life import voice_lineage as VL
from duckmind.simweb import voice

SR = voice.SAMPLE_RATE
MARKER = ".complete"
PLAN_FILE = "plan.json"
KEEP_LIBRARIES = 3  # the newest libraries kept on disk per voice
ROUTE = voice.ROUTE + "/lineage"
SING = ("call_out", "call_response", "self_duet")  # behaviours that say a learned phrase
ALARM_MS = 600  # a pass-through call is taken to sound this long (for "between calls")
PASS_MS = 1500
MAX_BUILD_WAIT_S = 150.0  # the bank is on its way (rendered by upstream's binary: about a second)
RECENT = 24


# ---------------------------------------------------------------------------------------- the sound


def read_wav(path: Path) -> np.ndarray:
    """A 16-bit mono wav as floats (-1..1)."""
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError(f"{path}: not 16-bit mono")
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0


def write_wav(path: Path, samples: np.ndarray) -> None:
    pcm = np.clip(np.rint(samples * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes(pcm.tobytes())


def resample(x: np.ndarray, ratio: float) -> np.ndarray:
    """Play ``x`` ``ratio`` times as fast (higher, shorter for ratio > 1): linear interpolation."""
    if x.size < 2 or abs(ratio - 1.0) < 1e-6:
        return x.copy()
    n = max(2, int(round(x.size / ratio)))
    position = np.arange(n, dtype=np.float64) * ratio
    return np.interp(position, np.arange(x.size, dtype=np.float64), x)


def wsola(x: np.ndarray, factor: float, win: int = 1024, tol: int = 256) -> np.ndarray:
    """Time-stretch ``x`` to ``factor`` times its length without moving its pitch (waveform
    similarity overlap-add: frames of ``win`` samples, each placed where it best continues the last)."""
    n = x.size
    out_len = max(2, int(round(n * factor)))
    if n < 2 * win or abs(factor - 1.0) < 1e-3:
        # (too short to be framed: the nearest length by resampling)
        return resample(x, n / out_len) if n >= 2 else x.copy()
    hop = win // 2
    window = np.hanning(win + 2)[1:-1]
    step_in = hop / factor
    frames = int(math.ceil(out_len / hop)) + 1
    out = np.zeros(out_len + 2 * win)
    norm = np.zeros_like(out)
    pad = np.concatenate([np.zeros(tol), x, np.zeros(win + tol + hop)])
    previous = 0
    for k in range(frames):
        nominal = int(round(k * step_in))
        if k == 0:
            start = 0
        else:
            ref = pad[tol + previous + hop : tol + previous + hop + win]
            seg = pad[tol + nominal - tol : tol + nominal + tol + win]
            if seg.size < win + 2 * tol or ref.size < win:
                start = min(nominal, n)
            else:
                score = np.correlate(seg, ref, mode="valid")
                start = nominal - tol + int(np.argmax(score))
        start = max(0, min(start, n))
        frame = pad[tol + start : tol + start + win]
        if frame.size < win:
            frame = np.pad(frame, (0, win - frame.size))
        at = k * hop
        out[at : at + win] += frame * window
        norm[at : at + win] += window
        previous = start
    norm[norm < 1e-6] = 1.0
    return (out / norm)[:out_len]


def shape_clip(x: np.ndarray, st: float, length: float, max_s: float = 0.0) -> np.ndarray:
    """``x`` ``st`` semitones up and ``length`` times as long (pitch and length each their own),
    cut at ``max_s`` seconds when that is set, at the clip's own loudness, with short fades."""
    if x.size < 2:
        return x.copy()
    peak = float(np.max(np.abs(x))) or 1.0
    y = resample(x, 2.0 ** (st / 12.0)) if abs(st) > 1e-9 else x.copy()
    target = max(2, int(round(x.size * length)))
    if abs(target / y.size - 1.0) > 0.03:
        y = wsola(y, target / y.size)
    if max_s > 0 and y.size > int(max_s * SR):
        y = y[: int(max_s * SR)]
        fade = min(y.size, int(0.04 * SR))
        y[-fade:] *= np.linspace(1.0, 0.0, fade)
    top = float(np.max(np.abs(y))) or 1.0
    y = y * (peak / top)
    edge = min(y.size // 2, int(0.003 * SR))
    if edge > 0:
        y[:edge] *= np.linspace(0.0, 1.0, edge)
        y[-edge:] *= np.linspace(1.0, 0.0, edge)
    return y


def _source(
    bank: Path, tag: str, letter: str, cache: dict[tuple[str, str], np.ndarray]
) -> np.ndarray:
    found = cache.get((tag, letter))
    if found is None:
        if tag == voice.SEGMENTED:  # a short whoop: the ride's start and its end, no loop
            found = np.concatenate(
                [read_wav(bank / tag / f"{tag}_{part}_{letter}.wav") for part in ("start", "end")]
            )
        else:
            found = read_wav(bank / tag / f"{tag}_{letter}.wav")
        cache[(tag, letter)] = found
    return found


def render_variant(
    bank: Path, variant: VL.Variant, cache: dict[tuple[str, str], np.ndarray]
) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for part in variant.parts:
        pieces.append(
            shape_clip(
                _source(bank, part.tag, part.letter, cache), part.st, part.length, part.max_s
            )
        )
        if part.gap_ms:
            pieces.append(np.zeros(int(part.gap_ms * SR / 1000)))
    return np.concatenate(pieces) if pieces else np.zeros(2)


def file_name(entry: VL.Entry, variant: VL.Variant) -> str:
    return f"{entry.name}_{variant.letter}.wav"


def render(plan: VL.Plan, bank: Path, out: Path) -> dict[str, dict[str, float]]:
    """Make the plan's clips in ``out`` (an empty folder): ``<entry>/<entry>_<letter>.wav``.
    Returns each file's length in seconds per entry."""
    cache: dict[tuple[str, str], np.ndarray] = {}
    lengths: dict[str, dict[str, float]] = {}
    for entry in plan.entries:
        folder = out / entry.name
        folder.mkdir(parents=True, exist_ok=True)
        lengths[entry.name] = {}
        for variant in entry.variants:
            samples = render_variant(bank, variant, cache)
            name = file_name(entry, variant)
            write_wav(folder / name, samples)
            lengths[entry.name][name] = round(samples.size / SR, 3)
    return lengths


def f0_of(x: np.ndarray) -> float:
    """A rough fundamental (Hz) of a clip by autocorrelation over its loudest 100 ms: what a
    check needs to say that a young call is higher than an old one."""
    if x.size < 2048:
        return 0.0
    energy = np.convolve(x * x, np.ones(4800), mode="valid")
    centre = int(np.argmax(energy)) + 2400
    seg = x[max(0, centre - 2400) : centre + 2400]
    seg = seg - seg.mean()
    ac = np.correlate(seg, seg, mode="full")[seg.size - 1 :]
    low, high = int(SR / 1500), int(SR / 120)
    if high >= ac.size or ac[0] <= 0:
        return 0.0
    lag = low + int(np.argmax(ac[low:high]))
    return SR / lag


# ------------------------------------------------------------------------------------- the library


class Library:
    """A built library: its plan, where it is and what it may serve."""

    def __init__(
        self, plan: VL.Plan, folder: Path, lengths: Mapping[str, Mapping[str, float]]
    ) -> None:
        self.plan = plan
        self.folder = folder
        self.lengths = {name: dict(files) for name, files in lengths.items()}
        self.files: dict[str, frozenset[str]] = {k: frozenset(v) for k, v in self.lengths.items()}
        self.key = plan.key
        self.key8 = plan.key8
        self.built_wall = 0.0
        self.why = ""
        self.built_calendar = ""
        self.weighted = [e.name for e in VL.phrase_entries(plan) for _ in range(e.weight)]
        self.has_song = plan.entry(VL.SONG) is not None

    def longest_ms(self, name: str) -> int:
        files = self.lengths.get(name) or {}
        return int(1000 * max(files.values())) if files else 0

    def url(self, name: str, file: str) -> str:
        return f"{ROUTE}/{self.key}/{name}/{file}"

    def tags(self) -> dict[str, Any]:
        """The manifest's entries for this library: tag -> {variants: {letter: url}}."""
        out: dict[str, Any] = {}
        for entry in self.plan.entries:
            out[entry.name] = {
                "variants": {
                    v.letter: self.url(entry.name, file_name(entry, v)) for v in entry.variants
                },
                "lineage": True,
            }
        return out

    def describe(self) -> dict[str, Any]:
        """The library for /api/life: structure and numbers only (no page words)."""
        plan = self.plan
        calls = []
        for entry in VL.phrase_entries(plan):
            calls.append(
                {
                    "clip": entry.name,
                    "code": entry.code,
                    "tags": list(entry.syllables),
                    "how": entry.kind,
                    "gen": entry.gen,
                    "praise": entry.praise,
                    "ms": self.longest_ms(entry.name),
                    "path": str(self.folder / entry.name),
                }
            )
        tags: dict[str, Any] = {}
        for tag in VL.LINEAGE_TAGS:
            found = plan.entry(tag)
            if found is None:
                continue
            sts = [v.parts[0].st for v in found.variants]
            lengths = [v.parts[0].length for v in found.variants]
            tags[tag] = {
                "variants": [v.letter for v in found.variants],
                "pitch_st": round(sum(sts) / len(sts), 3),
                "length": round(sum(lengths) / len(lengths), 3),
                "ms": self.longest_ms(tag),
            }
        song = plan.entry(VL.SONG)
        return {
            "key": self.key,
            "key8": self.key8,
            "folder": str(self.folder),
            "voice_seed": plan.voice_seed,
            "growth_day": plan.gd,
            "stage": plan.stage,
            "built_calendar": self.built_calendar,
            "why": self.why,
            "age": {"pitch_st": VL.age_pitch(plan.gd), "length": VL.age_length(plan.gd)},
            "tags": tags,
            "calls_learned": calls,
            "song": None
            if song is None
            else {
                "tags": list(song.syllables),
                "ms": self.longest_ms(VL.SONG),
                "path": str(self.folder / VL.SONG),
            },
            "fallback": ["alarm", "peck", "wheee", "emo_*"],
            "clips": sum(len(v) for v in self.lengths.values()),
        }


def build_library(plan: VL.Plan, bank: Path, root: Path) -> Library:
    """The plan's library under ``root`` (``<root>/seed<voice seed>/<key>/``): there already when a
    complete one is, otherwise made in a staging folder and renamed into place."""
    base = root / f"seed{plan.voice_seed}"
    final = base / plan.key
    try:
        if (final / MARKER).read_text().strip() == plan.key:
            data = json.loads((final / PLAN_FILE).read_text())
            lengths = data["lengths"]
            if VL.Plan.from_dict(data["plan"]).key == plan.key and all(
                (final / name / file).is_file() for name, files in lengths.items() for file in files
            ):
                return Library(plan, final, lengths)
    except (OSError, ValueError, KeyError):
        pass
    base.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{plan.key}.", suffix=".new", dir=base))
    try:
        staging.chmod(0o755)
        lengths = render(plan, bank, staging)
        (staging / PLAN_FILE).write_text(
            json.dumps({"plan": plan.to_dict(), "lengths": lengths}, ensure_ascii=False, indent=1)
        )
        (staging / MARKER).write_text(plan.key)
        if final.exists():
            shutil.rmtree(final, ignore_errors=True)
        staging.rename(final)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    _prune(base, keep=KEEP_LIBRARIES, current=plan.key)
    return Library(plan, final, lengths)


def _prune(base: Path, keep: int, current: str) -> None:
    folders = [p for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")]
    folders.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for folder in folders[keep:]:
        if folder.name != current:
            shutil.rmtree(folder, ignore_errors=True)
    for leftover in base.glob(".*.new"):
        if leftover.is_dir() and time.time() - leftover.stat().st_mtime > 600:
            shutil.rmtree(leftover, ignore_errors=True)


def bank_letters(bank_dir: Path) -> dict[str, list[str]]:
    """The letters of the variants a rendered bank really has, per tag."""
    out: dict[str, list[str]] = {}
    for tag in voice.TAGS:
        names = voice.file_names(tag)
        have = [n for n in names if (bank_dir / tag / n).is_file()]
        if tag == voice.SEGMENTED:
            letters = sorted({n.split("_")[-1][:-4] for n in have})
        else:
            letters = sorted(n[len(tag) + 1 : -4] for n in have)
        out[tag] = letters
    return out


# ------------------------------------------------------------------------------------- the manager


class LineageVoice:
    """The duck's lineage voice on a server: builds, swaps between calls, serves, shapes, reports."""

    def __init__(
        self,
        bank: voice.VoiceBank,
        duck_seed: int,
        view_source: Callable[[], VL.LineageView | None],
        *,
        calendar: Callable[[], str] = lambda: "",
        log: Callable[[str], None] | None = None,
        root: Path | None = None,
    ) -> None:
        self.bank = bank
        self.duck_seed = int(duck_seed)
        self.voice_seed = voice.voice_seed(self.duck_seed)
        self.view_source = view_source
        self.calendar = calendar
        self._log = log or (lambda text: print(f"叫聲系譜：{text}", file=sys.stderr, flush=True))
        self.root = (
            root if root is not None else (None if bank.root is None else bank.root / "lineage")
        )
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self.active: Library | None = None
        self.previous: Library | None = None
        self.pending: Library | None = None
        self._dirty: str | None = None
        self._last_t_ms: float | None = None
        self._busy_until_ms = 0.0
        self._phrase_until_ms = 0.0
        self._n_picks = 0
        self.failed = ""
        self.checks = 0
        self.unchanged = 0
        self.swaps = 0
        self.builds = 0
        self.history: deque[dict[str, Any]] = deque(maxlen=12)
        self.last_update: dict[str, Any] | None = None
        self.played = 0
        self.shaped = 0
        self.phrases_played = 0
        self.songs_played = 0
        self.dropped = 0
        self.passed = 0
        self.recent: deque[dict[str, Any]] = deque(maxlen=RECENT)
        self.by_clip: dict[str, int] = {}
        self.lineage_size = 0

    # -- keeping the library up to date -------------------------------------------------

    def refresh(self, why: str, *, wait: bool = False) -> str:
        """Look at the lineage now; if it sounds different, build the library in the background and
        put it in at the next quiet moment. Safe to call from any thread, as often as wanted (the
        nightly regrow, a day change, boot). Returns ``building`` | ``unavailable``."""
        if self.root is None:
            return "unavailable"
        with self._lock:
            self.checks += 1
            if self._thread is not None and self._thread.is_alive():
                self._dirty = why  # (asked again while it builds: looked at once more after)
                thread: threading.Thread | None = self._thread
            else:
                thread = threading.Thread(
                    target=self._run, args=(why,), name="lineage-voice", daemon=True
                )
                self._thread = thread
                thread.start()
        if wait and thread is not None:
            thread.join(MAX_BUILD_WAIT_S + 30)
        return "building"

    def _run(self, why: str) -> None:
        while True:
            try:
                self._build_once(why)
                self.failed = ""
            except Exception as error:  # the voice is the bank's own then; never the server's
                self.failed = f"{type(error).__name__}: {error}"
                self._log(f"做不出來（{why}）：{self.failed}；叫聲照原本的聲音庫。")
            with self._lock:
                again, self._dirty = self._dirty, None
            if again is None:
                return
            why = again

    def _bank_dir(self) -> Path | None:
        state = self.bank.prepare(self.duck_seed)
        if state in ("unavailable", "failed"):
            raise RuntimeError(self.bank.problem or "the voice bank is not there")
        deadline = time.monotonic() + MAX_BUILD_WAIT_S
        while not self.bank.ready(self.voice_seed):
            if time.monotonic() > deadline:
                raise RuntimeError("the voice bank was not ready in time")
            self.bank.wait(self.duck_seed, 1.0)
            if self.bank.prepare(self.duck_seed) == "failed":
                raise RuntimeError(self.bank.problem or "the voice bank failed")
        return self.bank.bank_dir(self.voice_seed)

    def _build_once(self, why: str) -> None:
        view = self.view_source()
        if view is None or self.root is None:
            return
        self.lineage_size = view.size
        bank_dir = self._bank_dir()
        assert bank_dir is not None
        plan = VL.plan_for(view, self.voice_seed, bank_letters(bank_dir))
        with self._lock:
            current = self.pending or self.active
            if current is not None and current.key == plan.key:
                self.unchanged += 1
                self.last_update = {"why": why, "calendar": self.calendar(), "result": "unchanged",
                                    "key8": plan.key8, "growth_day": plan.gd}  # fmt: skip
                return
        started = time.monotonic()
        existed = (self.root / f"seed{plan.voice_seed}" / plan.key / MARKER).is_file()
        library = build_library(plan, bank_dir, self.root)
        library.why = why
        library.built_wall = time.time()
        library.built_calendar = self.calendar()
        self.builds += 1
        took = round(time.monotonic() - started, 2)
        self._log(
            f"新的叫聲庫 {library.key8}（{why}，成長日 {plan.gd}，{plan.stage}）："
            f"{len(VL.phrase_entries(plan))} 句學來的叫聲"
            f"{'、一首歌' if library.has_song else ''}，{library.describe()['clips']} 個檔案，"
            f"{'本來就有' if existed else f'{took} 秒做好'}；在 {library.folder}"
        )
        with self._lock:
            self.pending = library
            self.last_update = {"why": why, "calendar": library.built_calendar, "result": "built",
                                "key8": library.key8, "growth_day": plan.gd, "seconds": took,
                                "reused": existed}  # fmt: skip
        self.tick(None)

    def tick(self, t_ms: float | None) -> bool:
        """The safe moment: put a finished library in if no call is sounding. ``t_ms`` is the
        brain's clock (ms); None: not driven by a brain (a headless check), swap when it has never
        been driven. Returns whether it swapped."""
        with self._lock:
            if t_ms is not None:
                if self._last_t_ms is not None and t_ms < self._last_t_ms - 5000:
                    # (a new body: its clock started over)
                    self._busy_until_ms = self._phrase_until_ms = 0.0
                self._last_t_ms = t_ms
            if self.pending is None:
                return False
            if t_ms is None:
                if self._last_t_ms is not None:
                    return False  # (a brain is driving: its next tick finds the quiet moment)
            elif t_ms < self._busy_until_ms:
                return False
            self._swap()
            return True

    def _swap(self) -> None:
        library = self.pending
        assert library is not None
        self.previous, self.active, self.pending = self.active, library, None
        self.swaps += 1
        self._n_picks = 0
        self.history.append(
            {
                "key8": library.key8,
                "why": library.why,
                "calendar": library.built_calendar,
                "growth_day": library.plan.gd,
                "phrases": len(VL.phrase_entries(library.plan)),
                "song": library.has_song,
            }  # fmt: skip
        )
        when = (
            "還沒有叫聲在響" if self._last_t_ms is None else "在兩次叫聲之間換的，沒有切斷任何一聲"
        )
        self._log(f"叫聲庫換成 {library.key8}（{when}）")

    # -- what the page is told ----------------------------------------------------------

    def overlay(self, manifest: dict[str, Any], duck_seed: int) -> None:
        """Put the lineage's clips in the voice manifest of the duck they belong to."""
        if int(duck_seed) != self.duck_seed or not manifest.get("ready"):
            return
        with self._lock:
            library = self.active
        if library is None:
            manifest["lineage"] = {
                "api": VL.API,
                "state": "building" if self.builds == 0 else "none",
            }
            return
        tags = manifest.get("tags")
        if not isinstance(tags, dict):
            return
        tags.update(library.tags())
        manifest["lineage"] = {
            "api": VL.API,
            "state": "ready",
            "key": library.key,
            "key8": library.key8,
            "tags": [e.name for e in library.plan.entries],
            "calls": {
                e.name: {"parts": list(e.syllables), "how": e.kind}
                for e in library.plan.entries
                if e.name.startswith(VL.PHRASE_PREFIX) or e.name == VL.SONG
            },
        }

    def handle(self, parts: Sequence[str], duck_seed: int | None) -> voice.Reply:
        """``/api/voice/lineage/<key>/<tag>/<file>``: one clip of the active (or the one before)
        library, only a name its plan wrote."""
        if duck_seed is None or int(duck_seed) != self.duck_seed or len(parts) != 3:
            return voice.NOT_FOUND
        key, name, file = parts
        with self._lock:
            libraries = [
                lib for lib in (self.active, self.previous, self.pending) if lib is not None
            ]
        for library in libraries:
            if library.key == key:
                if file not in library.files.get(name, frozenset()):
                    return voice.NOT_FOUND
                try:
                    body = (library.folder / name / file).read_bytes()
                except OSError:
                    return voice.NOT_FOUND
                return voice.Reply(HTTPStatus.OK, body, "audio/wav", voice.WAV_CACHE)
        return voice.NOT_FOUND

    # -- the call: what the duck says --------------------------------------------------

    def shape(
        self, vocals: list[dict[str, Any]], behaviour: str, t_ms: float
    ) -> list[dict[str, Any]]:
        """The brain's calls of this tick, as the lineage voices them (the list may be shorter:
        a call that comes while a learned phrase is sounding is not said, as robotd drops a one-shot
        while a stream holds the speaker). A call this library does not cover is returned as it is."""
        with self._lock:
            return self._shape(vocals, behaviour, t_ms)

    def _shape(
        self, vocals: list[dict[str, Any]], behaviour: str, t_ms: float
    ) -> list[dict[str, Any]]:
        self.tick(t_ms)
        library = self.active
        if library is None or not vocals:
            return vocals
        out: list[dict[str, Any]] = []
        for call in vocals:
            tag = str(call.get("tag"))
            entry = library.plan.entry(tag) if tag in VL.LINEAGE_TAGS else None
            if entry is None:
                # alarm, peck, wheee ...: the bank's own, and a reflex's alarm does cut a phrase
                self.passed += 1
                self._busy_until_ms = max(self._busy_until_ms, t_ms + ALARM_MS)
                self._phrase_until_ms = min(self._phrase_until_ms, t_ms)
                out.append(call)
                continue
            if t_ms < self._phrase_until_ms:
                self.dropped += 1
                continue
            name = self._choose(library, tag, behaviour)
            chosen = library.plan.entry(name) or entry
            long_ms = library.longest_ms(chosen.name) or PASS_MS
            self._busy_until_ms = max(self._busy_until_ms, t_ms + long_ms)
            shaped = {**call, "tag": chosen.name, "lib": library.key8}
            if chosen.name != tag:
                shaped["parts"] = list(chosen.syllables)
                shaped["from"] = tag
                self._phrase_until_ms = t_ms + long_ms
                if chosen.name == VL.SONG:
                    self.songs_played += 1
                else:
                    self.phrases_played += 1
            self.shaped += 1
            self.played += 1
            self.by_clip[chosen.name] = self.by_clip.get(chosen.name, 0) + 1
            self.recent.append(
                {
                    "t_ms": int(t_ms),
                    "behaviour": behaviour,
                    "asked": tag,
                    "clip": chosen.name,
                    "parts": list(chosen.syllables),
                    "variants": [v.letter for v in chosen.variants],
                    "folder": str(library.folder / chosen.name),
                    "lib": library.key8,
                }  # fmt: skip
            )
            out.append(shaped)
        return out

    def _choose(self, library: Library, tag: str, behaviour: str) -> str:
        if behaviour == "little_song" and library.has_song:
            return VL.SONG
        if behaviour in (*SING, "little_song") and library.weighted:
            self._n_picks += 1
            index = int(
                hashlib.sha256(f"{library.key}|{self._n_picks}".encode()).hexdigest()[:8], 16
            )
            return library.weighted[index % len(library.weighted)]
        return tag

    # -- /api/life ----------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        with self._lock:
            active, pending = self.active, self.pending
            thread = self._thread
            building = bool(thread is not None and thread.is_alive())
            out: dict[str, Any] = {
                "api": VL.API,
                "library_version": VL.LIBRARY_VERSION,
                "rules": VL.rules_hash(),
                "state": "failed"
                if self.failed and active is None
                else "ready"
                if active is not None
                else "building"
                if building
                else "idle",
                "lineage_size": self.lineage_size,
                "voice_seed": self.voice_seed,
                "active": None if active is None else active.describe(),
                "pending": None if pending is None else pending.key8,
                "building": building,
                "last_update": self.last_update,
                "history": list(self.history),
                "counts": {
                    "checks": self.checks,
                    "builds": self.builds,
                    "unchanged": self.unchanged,
                    "swaps": self.swaps,
                    "calls_shaped": self.shaped,
                    "phrases_sung": self.phrases_played,
                    "songs_sung": self.songs_played,
                    "dropped_for_phrase": self.dropped,
                    "passed_through": self.passed,
                },
                "by_clip": dict(sorted(self.by_clip.items())),
                "recent": list(self.recent),
                "problem": None,
            }
        if self.failed:
            out["problem"] = msg("web.lineage_voice.failed", why=self.failed)
        elif self.root is None:
            out["problem"] = msg("web.lineage_voice.no_cache")
        return out


def small_hash(path: Path) -> str:
    """A short hash of a folder's wav bytes, in name order (to say two libraries are the same)."""
    digest = hashlib.sha256()
    for file in sorted(path.rglob("*.wav")):
        digest.update(str(file.relative_to(path)).encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()[:16]


def env_root() -> Path | None:
    """``$DUCKMIND_LINEAGE_ROOT``: where libraries are kept instead of beside the voice bank (a check
    server of its own keeps them in its own folder)."""
    raw = os.environ.get("DUCKMIND_LINEAGE_ROOT")
    return Path(raw).expanduser() if raw else None
