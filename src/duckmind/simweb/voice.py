"""小鴨的叫聲: the duck's own official voice for the simulator page.

The voice is the official Microduck one: upstream's ``sounds`` crate (Apache-2.0), a seedable
synth that derives a whole vocal personality from one u32 seed (pitch register, timbre, vibrato,
quackiness, tempo) and renders a bank of wavs from it — seven tags (``alarm``, ``greet``,
``inquire``, ``peck``, ``chirp``, ``coo``, ``wheee``), several variants of each, 48 kHz mono
16-bit, ``wheee`` as ``start`` / ``loop`` / ``end`` segments so it can be held as long as a ride
lasts. On the robot the seed is the SoC serial and ``robotd`` plays a random variant of a tag
each time one is asked for. Nothing of it is copied into this repository: upstream's own binary
renders each duck's bank on demand (``<microduck>/target/debug/sounds render-all <dir> --seed
<u32>``) into a cache outside the repo, and this module serves the page from there.

**Which voice a duck has** (:func:`voice_seed`). A simulated duck has no SoC serial; its
hardware identity is its Duck Core seed (fixed at hatching: its temperament, quirks and now its
voice). The voice seed is upstream's own derivation applied to that identity,
``seed_from_id("duckmind-duck:<seed>")`` — the sha256 hash a robot applies to its serial, and
upstream's own answer (``DUCK_IDENTITY``) to "one machine, several ducks". So each duck keeps one
voice for life, two seeds are two different voices, and 蛋黃 (seed 7) has its own: voice seed
2,838,558,865 — not upstream's bench seed 7 (the bench bank ``voices/v5/seed7``). A duck hatched
again from the same seed is the same innate duck, voice included.

**The cache.** ``voices/v<BANK_VERSION>/seed<voice seed>/<tag>/<file>.wav`` in the data folder
(:func:`duckmind.paths.data_dir`) —
upstream's own layout (``robotd`` plays from it) with upstream's ``.seed`` marker
(``<seed>:<BANK_VERSION>``, as ``sounds ensure-bank`` writes it), written last. A bank is
rendered into a hidden staging folder beside it, checked file by file and renamed into place,
so a bank is complete or absent, never half there. Rendering takes about a second; it runs in
the background (:meth:`VoiceBank.prepare`), one at a time in this process and, through a file
lock in the version folder, one at a time on this machine (the owner's simulator, the neural-duck
demo and a test server may all ask for the same duck). ``BANK_VERSION`` is in the path, so a
retuned synth never serves an old bank.

**Routes** (:meth:`VoiceBank.handle`, GET only):

* ``/api/voice/manifest`` — the current duck's voice: its seeds, the tags, each tag's variants
  and every file's URL; or, in plain words, why there is no voice yet (rendering, no binary).
* ``/api/voice/v<BANK_VERSION>/<voice seed>/<tag>/<file>.wav`` — one file of the current duck's
  bank. Only the names upstream's render writes are served (a fixed whitelist per tag), only for
  the current duck's voice seed and bank version, and nothing from the request is ever joined
  onto a path: ``..``, encoded slashes, another duck's bank or the marker have nothing to reach.

**Which events are calls** (:func:`calls_of`): a Duck Core event whose payload lists the sounds
the duck made (``sounds``: expressions, reactions, greetings, bids, bedtime, a stop, a sleepy
mumble) or a babbled phrase (``voice.babble``'s ``tags``). Everything else is words only and
makes no sound.

The module imports nothing from duckmind (stdlib only), so another simulator page (the
neural-duck prototype) can use it verbatim: it needs the duck's seed, :meth:`VoiceBank.handle`
in its GET handler and, for its feed, :func:`calls_of`.

``python -m duckmind.simweb.voice 7 --microduck <checkout> --render`` says which voice a duck
has and renders its bank ahead of time.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

from duckmind import paths
from duckmind.i18n import msg

# upstream sounds::BANK_VERSION (sounds/src/lib.rs): v5 is the Rust port, 48 kHz native. A
# checkout that says otherwise is refused (the whitelist below is v5's), never half-served.
BANK_VERSION = 5
# upstream sounds::TAGS, in its render order.
TAGS: tuple[str, ...] = ("alarm", "greet", "inquire", "peck", "chirp", "coo", "wheee")
SEGMENTED = "wheee"  # rendered as start / loop / end triads (sounds::render_all)
SEGMENTS: tuple[str, ...] = ("start", "loop", "end")
SAMPLE_RATE = 48_000  # sounds::SR
MARKER = ".seed"  # sounds ensure-bank's marker: "<seed>:<BANK_VERSION>"
DEFAULT_ROOT = paths.data_dir() / "voices"
IDENTITY = "duckmind-duck:{seed}"  # a simulated duck's hardware identity (see voice_seed)
EMOTION_PREFIX = (
    "emo_"  # the tags of microduck_emotions' sounds in the duck's voice (emotion_voice.TAG_PREFIX)
)
ROUTE = "/api/voice"
MANIFEST = ROUTE + "/manifest"
RENDER_TIMEOUT_S = 120.0  # a bank takes about a second; this only bounds a wedged binary
MANIFEST_WAIT_S = 4.0  # the manifest waits this long for a render already running
RETRY_MS = 1500  # the page asks again this soon while a bank is rendering
FAILED_RETRY_S = 60.0  # a failed render is tried again after this long
WAV_CACHE = "private, max-age=604800"  # a bank never changes: its version and seed are in the URL

SOURCE_ZH = msg("web.voice.SOURCE_ZH")
NO_CHECKOUT_ZH = msg("web.voice.NO_CHECKOUT_ZH")
NO_BINARY_ZH = msg("web.voice.NO_BINARY_ZH")
OFF_ZH = msg("web.voice.OFF_ZH")
RENDERING_ZH = msg("web.voice.RENDERING_ZH")


def variant_count(tag: str) -> int:
    """How many variants upstream renders of a tag (sounds::variant_count): the most-heard
    ``chirp`` and ``greet`` get 12, the segmented ``wheee`` 6, the rest 10."""
    if tag in ("greet", "chirp"):
        return 12
    if tag == SEGMENTED:
        return 6
    return 10


def letters(tag: str) -> tuple[str, ...]:
    """A tag's variants as upstream names them: ``a``, ``b``, …"""
    return tuple(chr(ord("a") + index) for index in range(variant_count(tag)))


def file_names(tag: str) -> tuple[str, ...]:
    """The files upstream's render writes for a tag: ``<tag>_<letter>.wav``, or for ``wheee``
    ``wheee_<start|loop|end>_<letter>.wav`` (sounds::render_all)."""
    if tag == SEGMENTED:
        return tuple(f"{tag}_{part}_{letter}.wav" for letter in letters(tag) for part in SEGMENTS)
    return tuple(f"{tag}_{letter}.wav" for letter in letters(tag))


# The only files ever served: tag → the names upstream writes for it (82 in all).
WHITELIST: dict[str, frozenset[str]] = {tag: frozenset(file_names(tag)) for tag in TAGS}
BANK_FILES = sum(len(names) for names in WHITELIST.values())


def seed_from_id(identity: str) -> int:
    """sha256(identity), its first 4 bytes big-endian — upstream's ``sounds::seed_from_id``, how a
    robot turns its SoC serial (or ``DUCK_IDENTITY``) into its voice seed."""
    return int.from_bytes(hashlib.sha256(identity.encode("utf-8")).digest()[:4], "big")


def voice_seed(duck_seed: int) -> int:
    """The duck's voice seed (u32): upstream's derivation of its identity, its Duck Core seed.

    Deterministic, so a duck keeps one voice for life; different seeds give different voices;
    蛋黃 (seed 7) → 2,838,558,865. See the module docstring."""
    return seed_from_id(IDENTITY.format(seed=int(duck_seed)))


def calls_of(kind: str, payload: Mapping[str, Any] | None) -> list[str]:
    """The official tags a Duck Core event voiced, in order ([] for an event that is only words).

    Duck Core records every sound it makes on the event itself: ``sounds`` (expressions,
    reactions, greetings, bids, bedtime, a stop, a sleepy mumble) or, for a babbled phrase,
    ``voice.babble``'s ``tags``. Tags this bank does not have are left out."""
    if not isinstance(payload, Mapping):
        return []
    raw = payload.get("tags") if kind == "voice.babble" else payload.get("sounds")
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(tag) for tag in raw if str(tag) in WHITELIST]


def find_sounds_bin(microduck: Path | str | None) -> Path | None:
    """Upstream's ``sounds`` binary in a built microduck checkout: ``cargo build -p sounds`` writes
    ``target/debug/sounds`` (a release or cross build ``target/release`` or
    ``target/<triple>/…``); the newest one there wins. None when there is none."""
    if not microduck:
        return None
    target = Path(microduck).expanduser() / "target"
    candidates = {
        target / "release" / "sounds",
        target / "debug" / "sounds",
        *target.glob("*/release/sounds"),
        *target.glob("*/debug/sounds"),
    }
    found = [path for path in candidates if path.is_file() and os.access(path, os.X_OK)]
    return max(found, key=lambda path: path.stat().st_mtime) if found else None


def upstream_bank_version(microduck: Path | str | None) -> int | None:
    """``BANK_VERSION`` as the checkout's own ``sounds/src/lib.rs`` says (None: not there)."""
    if not microduck:
        return None
    try:
        text = (Path(microduck).expanduser() / "sounds" / "src" / "lib.rs").read_text()
    except OSError:
        return None
    found = re.search(r"pub const BANK_VERSION: u32 = (\d+);", text)
    return int(found.group(1)) if found else None


@dataclass(frozen=True)
class Reply:
    """An answer to a voice route, for the server's handler to write (:func:`send`)."""

    status: int
    body: bytes
    content_type: str
    cache: str = "no-store"


NOT_FOUND = Reply(HTTPStatus.NOT_FOUND, b"not found", "text/plain; charset=utf-8")


def _emotion_bank(root: Path | None) -> Any:
    """The emotions' sounds in the duck's voice (:mod:`duckmind.simweb.emotion_voice`, numpy): None where there is
    no cache or no numpy, and the page then has the 14 emotions without sound."""
    if root is None:
        return None
    try:
        from duckmind.simweb import emotion_voice
    except ImportError:
        return None
    return emotion_voice.EmotionBank(root / "emotions")


def _json(payload: Mapping[str, Any], status: int = HTTPStatus.OK) -> Reply:
    # (the words' messages next to them, for the page's other languages: docs/i18n.md)
    from duckmind.i18n.boundary import attach

    body = json.dumps(attach(dict(payload)), ensure_ascii=False).encode()
    return Reply(status, body, "application/json; charset=utf-8")


def send(handler: BaseHTTPRequestHandler, reply: Reply) -> None:
    """Write a :class:`Reply` through a ``BaseHTTPRequestHandler``."""
    handler.send_response(reply.status)
    handler.send_header("Content-Type", reply.content_type)
    handler.send_header("Content-Length", str(len(reply.body)))
    handler.send_header("Cache-Control", reply.cache)
    handler.send_header("X-Content-Type-Options", "nosniff")
    handler.end_headers()
    handler.wfile.write(reply.body)


@contextmanager
def _exclusive(lock_path: Path) -> Iterator[None]:
    """An exclusive lock on a file: one render at a time on this machine (flock is released
    with the process, so a crashed render never leaves a lock behind)."""
    with open(lock_path, "a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class VoiceBank:
    """Every duck's voice bank on this machine: found, rendered (once, by upstream's binary)
    and served to the page.

    ``sounds_bin``: upstream's binary (None: banks already rendered are still served, none is
    made); ``root``: the cache (None: no voice at all); ``problem``: why there is no voice, in
    plain words, when that is known up front."""

    def __init__(
        self,
        sounds_bin: Path | None,
        root: Path | None = DEFAULT_ROOT,
        *,
        version: int = BANK_VERSION,
        problem: str = "",
    ) -> None:
        self.sounds_bin = sounds_bin
        self.root = root
        self.version = version
        self.problem = problem or (NO_BINARY_ZH if sounds_bin is None else "")
        self.renders = 0  # banks this process rendered
        # the 14 emotions' sounds in this duck's voice (microduck_emotions; emotion_voice.py), beside the bank
        self.emotions = _emotion_bank(root)
        # the duck's lineage voice (lineage_voice.py, docs/design/GROWTH-G5-2026-10-11.md): the clips of the
        # calls it learned, laid over this bank's in the manifest and served beside it; None: the bank only
        self.lineage: Any = None
        self._render_lock = threading.Lock()  # one render at a time in this process
        self._lock = threading.Lock()  # the fields below
        self._ready: set[int] = set()
        self._threads: dict[int, threading.Thread] = {}
        self._failed: dict[int, tuple[float, str]] = {}

    @classmethod
    def from_microduck(
        cls, microduck: Path | str | None, root: Path | None = DEFAULT_ROOT
    ) -> VoiceBank:
        """The bank for a server started with ``--microduck`` (or ``DUCKMIND_MICRODUCK``)."""
        if not microduck:
            return cls(None, root, problem=NO_CHECKOUT_ZH)
        found = upstream_bank_version(microduck)
        if found is not None and found != BANK_VERSION:
            problem = msg(
                "web.voice.VoiceBank.from_microduck", found=found, BANK_VERSION=BANK_VERSION
            )
            return cls(None, None, problem=problem)
        return cls(find_sounds_bin(microduck), root)

    @classmethod
    def off(cls) -> VoiceBank:
        """No voice at all (a server started without one, the tests)."""
        return cls(None, None, problem=OFF_ZH)

    # ------------------------------------------------------------------ the cache

    def bank_dir(self, vseed: int) -> Path | None:
        """Where a voice seed's bank lives (None: no cache)."""
        if self.root is None:
            return None
        return self.root / f"v{self.version}" / f"seed{int(vseed)}"

    def _complete(self, vseed: int) -> bool:
        """Its marker says this seed and version, and every file upstream writes is there."""
        folder = self.bank_dir(vseed)
        if folder is None:
            return False
        try:
            if (folder / MARKER).read_text().strip() != f"{int(vseed)}:{self.version}":
                return False
            return all(
                (folder / tag / name).stat().st_size > 44  # (more than a wav header)
                for tag, names in WHITELIST.items()
                for name in names
            )
        except OSError:
            return False

    def ready(self, vseed: int) -> bool:
        """Whether a voice seed's whole bank is in the cache."""
        with self._lock:
            if vseed in self._ready:
                return True
        if not self._complete(vseed):
            return False
        with self._lock:
            self._ready.add(vseed)
        return True

    def render(self, vseed: int) -> None:
        """Render a voice seed's bank now (blocking) unless it is there already. Raises
        RuntimeError, in plain words, when it cannot."""
        if self.root is None or self.sounds_bin is None:
            raise RuntimeError(self.problem or NO_BINARY_ZH)
        with self._render_lock:
            if self.ready(vseed):
                return
            base = self.root / f"v{self.version}"
            base.mkdir(parents=True, exist_ok=True)
            with _exclusive(base / ".render.lock"):
                if self.ready(vseed):
                    return  # another simulator on this machine just made it
                _sweep(base)
                staging = Path(tempfile.mkdtemp(prefix=f".seed{vseed}.", suffix=".new", dir=base))
                try:
                    staging.chmod(0o755)  # (mkdtemp's 0700 would hide the bank from other users)
                    self._render_into(staging, vseed)
                    (staging / MARKER).write_text(f"{int(vseed)}:{self.version}")
                    final = base / f"seed{int(vseed)}"
                    if final.exists() or final.is_symlink():
                        # (an incomplete or foreign bank: out of the way, then the new one in)
                        old = base / f".seed{vseed}.{os.getpid()}.{time.time_ns()}.old"
                        final.rename(old)
                        shutil.rmtree(old, ignore_errors=True)
                    staging.rename(final)
                finally:
                    shutil.rmtree(staging, ignore_errors=True)  # (gone already when it worked)
            if not self._complete(vseed):
                raise RuntimeError(msg("web.voice.VoiceBank.render"))
            with self._lock:
                self._ready.add(vseed)
                self.renders += 1

    def _render_into(self, folder: Path, vseed: int) -> None:
        """Upstream's binary renders the whole bank into ``folder``; every file is checked."""
        command = [str(self.sounds_bin), "render-all", str(folder), "--seed", str(int(vseed))]
        try:
            done = subprocess.run(
                command, capture_output=True, text=True, timeout=RENDER_TIMEOUT_S, check=False
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(
                msg("web.voice.VoiceBank.render_into.f237f1", name=type(error).__name__)
            ) from error
        if done.returncode != 0:
            said = (done.stderr or done.stdout or "").strip().splitlines()
            raise RuntimeError(
                msg(
                    "web.voice.VoiceBank.render_into.9161cd",
                    said=said[-1][:160] if said else str(done.returncode),
                )
            )
        missing = [
            name
            for tag, names in WHITELIST.items()
            for name in names
            if not (folder / tag / name).is_file() or (folder / tag / name).stat().st_size <= 44
        ]
        if missing:
            raise RuntimeError(
                msg(
                    "web.voice.VoiceBank.render_into.823702",
                    missing=len(missing),
                    missing2=missing[0],
                )
            )

    def prepare(self, duck_seed: int) -> str:
        """Make sure the duck's bank is on its way: a background render when it is missing.
        The state now: ``ready`` | ``rendering`` | ``failed`` | ``unavailable``."""
        vseed = voice_seed(duck_seed)
        if self.ready(vseed):
            return "ready"
        if self.root is None or self.sounds_bin is None:
            return "unavailable"
        with self._lock:
            thread = self._threads.get(vseed)
            if thread is not None and thread.is_alive():
                return "rendering"
            failed = self._failed.get(vseed)
            if failed is not None and time.monotonic() - failed[0] < FAILED_RETRY_S:
                return "failed"
            thread = threading.Thread(
                target=self._render_quietly, args=(vseed,), name=f"voice-{vseed}", daemon=True
            )
            self._threads[vseed] = thread
            thread.start()
        return "rendering"

    def _render_quietly(self, vseed: int) -> None:
        try:
            self.render(vseed)
        except Exception as error:  # (the page says why; the simulator carries on)
            with self._lock:
                said = (
                    error.args[0]
                    if error.args and isinstance(error.args[0], str) and error.args[0]
                    else ""
                )
                self._failed[vseed] = (time.monotonic(), said or str(error) or type(error).__name__)
        else:
            with self._lock:
                self._failed.pop(vseed, None)

    def wait(self, duck_seed: int, timeout: float) -> None:
        """Wait (at most ``timeout`` s) for a render of the duck's bank that is running."""
        with self._lock:
            thread = self._threads.get(voice_seed(duck_seed))
        if thread is not None:
            thread.join(timeout)

    # ------------------------------------------------------------------ the page

    def manifest(self, duck_seed: int, wait_s: float = MANIFEST_WAIT_S) -> dict[str, Any]:
        """The current duck's voice for the page: seeds, tags, variants and URLs when its bank
        is ready; otherwise the state and, in plain words, why there is no voice yet."""
        vseed = voice_seed(duck_seed)
        state = self.prepare(duck_seed)
        if state == "rendering" and wait_s > 0:
            self.wait(duck_seed, wait_s)
            state = self.prepare(duck_seed)
        out: dict[str, Any] = {
            "ok": True,
            "state": state,
            "ready": state == "ready",
            "duck_seed": int(duck_seed),
            "voice_seed": vseed,
            "bank_version": self.version,
            "sample_rate": SAMPLE_RATE,
            "source": SOURCE_ZH,
            "note": "",
        }
        if state == "ready":
            out["tags"] = {tag: self._tag(vseed, tag) for tag in TAGS}
            # the emotions' sounds (extra tags ``emo_<name>``): served once rendered (a couple of seconds)
            if self.emotions is not None:
                emotions = self.emotions.prepare(vseed, self.bank_dir(vseed))
                out["emotion_state"] = emotions
                if emotions == "ready":
                    out["tags"].update(
                        self.emotions.tags(vseed, f"{ROUTE}/v{self.version}/{vseed}")
                    )
                elif emotions == "rendering":
                    out["emotion_retry_ms"] = RETRY_MS
            if self.lineage is not None:
                self.lineage.overlay(out, duck_seed)
        elif state == "rendering":
            out["note"] = RENDERING_ZH
            out["retry_ms"] = RETRY_MS
        elif state == "failed":
            with self._lock:
                why = self._failed.get(vseed, (0.0, ""))[1]
            out["note"] = (
                msg("web.voice.VoiceBank.manifest.f0039d", why=why)
                if why
                else msg("web.voice.VoiceBank.manifest.f062c5")
            )
            out["retry_ms"] = int(FAILED_RETRY_S * 1000)
        else:
            out["note"] = self.problem or NO_BINARY_ZH
        return out

    def _tag(self, vseed: int, tag: str) -> dict[str, Any]:
        base = f"{ROUTE}/v{self.version}/{vseed}/{tag}/"
        if tag == SEGMENTED:
            variants: dict[str, Any] = {
                letter: {part: f"{base}{tag}_{part}_{letter}.wav" for part in SEGMENTS}
                for letter in letters(tag)
            }
            return {"segmented": True, "variants": variants}
        return {"variants": {letter: f"{base}{tag}_{letter}.wav" for letter in letters(tag)}}

    def handle(self, path: str, duck_seed: int | None) -> Reply | None:
        """A GET under ``/api/voice`` (the path without its query), for the current duck
        (``duck_seed``; None: no duck yet). None: not a voice route."""
        if path != ROUTE and not path.startswith(ROUTE + "/"):
            return None
        if path in (ROUTE, ROUTE + "/", MANIFEST):
            if duck_seed is None:
                return _json({"ok": False, "error": "no duck yet"}, HTTPStatus.SERVICE_UNAVAILABLE)
            return _json(self.manifest(duck_seed))
        parts = path[len(ROUTE) + 1 :].split("/")
        if (
            parts[0] == "lineage"
        ):  # the learned calls (lineage_voice.py): its own clips, its own whitelist
            if self.lineage is None:
                return NOT_FOUND
            return self.lineage.handle(parts[1:], duck_seed)  # type: ignore[no-any-return]
        if len(parts) != 4 or duck_seed is None:
            return NOT_FOUND
        version, seed_text, tag, name = parts
        vseed = voice_seed(duck_seed)
        # Only the current duck's bank, only its version, only names upstream writes for the tag.
        if version != f"v{self.version}" or seed_text != str(vseed):
            return NOT_FOUND
        if tag.startswith(EMOTION_PREFIX):
            sound = None
            if self.emotions is not None and self.ready(vseed):
                sound = self.emotions.read(vseed, tag, name)
            return (
                NOT_FOUND if sound is None else Reply(HTTPStatus.OK, sound, "audio/wav", WAV_CACHE)
            )
        if name not in WHITELIST.get(tag, frozenset()) or not self.ready(vseed):
            return NOT_FOUND
        folder = self.bank_dir(vseed)
        if folder is None:
            return NOT_FOUND
        try:
            body = (folder / tag / name).read_bytes()
        except OSError:
            with self._lock:
                self._ready.discard(vseed)  # (gone from under us: rendered again next time)
            return NOT_FOUND
        return Reply(HTTPStatus.OK, body, "audio/wav", WAV_CACHE)


def _sweep(base: Path) -> None:
    """Leftovers of renders that died half-way (staging and replaced folders). Called with the
    machine-wide lock held, so nothing is rendering into them."""
    for leftover in base.glob(".seed*"):
        if leftover.is_dir() and leftover.name.endswith((".new", ".old")):
            shutil.rmtree(leftover, ignore_errors=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Which official voice a duck has (from its Duck Core seed), and its bank."
    )
    parser.add_argument("seeds", nargs="+", type=int, help="Duck Core seeds (蛋黃: 7)")
    parser.add_argument(
        "--microduck",
        type=Path,
        default=os.environ.get("DUCKMIND_MICRODUCK") or None,
        help="official microduck checkout with a built target/debug/sounds",
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="the voice bank cache")
    parser.add_argument("--render", action="store_true", help="render missing banks now")
    args = parser.parse_args(argv)
    bank = VoiceBank.from_microduck(args.microduck, args.root)
    for seed in args.seeds:
        vseed = voice_seed(seed)
        if args.render and not bank.ready(vseed):
            bank.render(vseed)
        state = "ready" if bank.ready(vseed) else "not rendered"
        print(f"duck seed {seed} → voice seed {vseed}: {bank.bank_dir(vseed)} ({state})")


if __name__ == "__main__":
    main()
