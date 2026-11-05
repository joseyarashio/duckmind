"""Local speech recognition for voice commands: a short clip in, the words that were said out.

The page's push-to-talk button (「按住說話」) records while it is held and sends the clip, as 16 kHz mono
16-bit PCM, to this computer's own simulator server; :class:`Recognizer` turns it into text here, with
an OpenAI Whisper model (MIT licence, weights included) run by one of two local engines:

* ``mlx`` — `mlx-whisper <https://github.com/ml-explore/mlx-examples>`_ (MIT) on Apple silicon's GPU;
* ``faster`` — `faster-whisper <https://github.com/SYSTRAN/faster-whisper>`_ (MIT, CTranslate2) on the CPU,
  for any machine.

Nothing is sent anywhere and nothing is written to disk: the clip lives in memory for one recognition.
The model files are fetched once from Hugging Face the first time (or given as a local folder); after
that the recogniser works offline.

The recogniser is told the language the viewer reads the page in (the six page languages; 繁中 and 简中
are both Whisper's ``zh``) and, as a prompt, the command words of that language (and the duck's name),
which steers it toward the words a family says to the duck and toward the right script for 繁中/简中.
Two guards keep a press with nothing said from turning into words: a clip that is too short or too
quiet is not recognised at all (:data:`MIN_S`, :data:`SILENT_DBFS`), and what Whisper itself judges to
be no speech, or a copy of its prompt, is dropped.

Which engine and model is a measured choice (docs/design/neural-duck/VOICE-COMMANDS-2026-11-04.md);
:data:`DEFAULT_SPEC` is the one picked. The packages are optional (``pip install duckmind[voice]``):
without them :func:`make` answers why the button stays off.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from duckmind.neuro import command_words as W
from duckmind.world import neuro_audio as A

SR = A.SR  # 16 kHz: what the page sends and Whisper takes
MIN_S = 0.25  # a shorter press is a slip of the finger: nothing is recognised
MAX_S = 10.0  # the longest clip taken (push-to-talk is for a short sentence)
SILENT_DBFS = -55.0  # the loudest 30 ms of the clip quieter than this: nobody spoke
NO_SPEECH = 0.6  # Whisper's own "no speech" rule: no_speech_prob above this ...
LOW_LOGPROB = -1.0  # ... and an average log-probability below this drops the segment
WHISPER_LANG = {"zh-Hant": "zh", "zh-Hans": "zh", "en": "en", "ja": "ja", "es": "es", "ko": "ko"}

ENGINES = ("mlx", "faster")
# (the measured pick: see the design note's table)
DEFAULT_SPEC = "mlx:mlx-community/whisper-medium-mlx"  # (clean set: best of 7, 0.21 s median)
# (when mlx cannot load: not Apple silicon. On the CPU medium took 2.8 s from letting go to the answer, small
# 0.98 s: only small keeps to the 1 s target; it is a little less sure, most of all in 简中)
FALLBACK_SPEC = "faster:small"


class Engine(Protocol):
    """One local Whisper runtime: ``run`` gives the segments' text, dropping what it judges silence."""

    name: str
    model: str

    def load(self) -> None: ...

    def run(self, audio: np.ndarray[Any, Any], language: str, prompt: str) -> str: ...


def _keep(no_speech: float | None, logprob: float | None) -> bool:
    return not (
        no_speech is not None
        and logprob is not None
        and no_speech > NO_SPEECH
        and logprob < LOW_LOGPROB
    )


class MlxEngine:
    """mlx-whisper on the Apple GPU (``model``: a Hugging Face repo or a local folder)."""

    name = "mlx"

    def __init__(self, model: str) -> None:
        self.model = model
        self._mod: Any = None

    def load(self) -> None:
        if self._mod is None:
            import mlx_whisper

            self._mod = mlx_whisper
        # (the first call loads and keeps the weights: ModelHolder)
        self.run(np.zeros(SR // 2, dtype=np.float32), "en", "")

    def run(self, audio: np.ndarray[Any, Any], language: str, prompt: str) -> str:
        if self._mod is None:
            import mlx_whisper

            self._mod = mlx_whisper
        result = self._mod.transcribe(
            audio,
            path_or_hf_repo=self.model,
            language=language,
            task="transcribe",
            initial_prompt=prompt or None,
            temperature=0.0,
            condition_on_previous_text=False,
            without_timestamps=True,
            verbose=None,
        )
        return "".join(
            str(seg.get("text", ""))
            for seg in result.get("segments", [])
            if _keep(seg.get("no_speech_prob"), seg.get("avg_logprob"))
        ).strip()

    def score(
        self, audio: np.ndarray[Any, Any], language: str, prompt: str, texts: list[str]
    ) -> list[float]:
        """How well each of ``texts`` fits the clip: the mean log-probability per token (its end
        included) of the model writing exactly that text, primed as :meth:`run` primes it. One encoder
        pass and one batched decoder pass for all of them."""
        import mlx.core as mx
        from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim
        from mlx_whisper.tokenizer import get_tokenizer
        from mlx_whisper.transcribe import ModelHolder

        model = ModelHolder.get_model(self.model, mx.float16)
        mel = log_mel_spectrogram(audio, n_mels=model.dims.n_mels, padding=N_SAMPLES)
        mel = pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16)
        features = model.embed_audio(mel[None])
        tok = get_tokenizer(model.is_multilingual, num_languages=model.num_languages,
                            language=language, task="transcribe")  # fmt: skip
        head: list[int] = []
        if prompt:
            head = [tok.sot_prev, *tok.encode(" " + prompt.strip())[-(model.dims.n_text_ctx // 2 - 1) :]]
        head += list(tok.sot_sequence_including_notimestamps)
        bodies = [[*tok.encode(text), tok.eot] for text in texts]
        width = len(head) + max(len(b) for b in bodies)
        rows = [head + b + [tok.eot] * (width - len(head) - len(b)) for b in bodies]
        out: list[float] = []
        step = 64
        for at in range(0, len(rows), step):
            chunk = mx.array(rows[at : at + step])
            logits = model.logits(chunk[:, :-1], mx.repeat(features, chunk.shape[0], axis=0))
            logp = np.array((logits - mx.logsumexp(logits, axis=-1, keepdims=True)).astype(mx.float32))
            for i, body in enumerate(bodies[at : at + step]):
                where = np.arange(len(head) - 1, len(head) - 1 + len(body))
                out.append(float(np.mean(logp[i, where, body])))
        return out


class FasterEngine:
    """faster-whisper (CTranslate2) on the CPU, int8 (``model``: a size such as ``small``, a Hugging
    Face repo or a local folder)."""

    name = "faster"

    def __init__(self, model: str, *, beam: int = 1, threads: int = 0) -> None:
        self.model = model
        self.beam = beam
        self.threads = threads
        self._model: Any = None

    def load(self) -> None:
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model, device="cpu", compute_type="int8", cpu_threads=self.threads
            )

    def run(self, audio: np.ndarray[Any, Any], language: str, prompt: str) -> str:
        self.load()
        segments, _ = self._model.transcribe(
            audio,
            language=language,
            task="transcribe",
            beam_size=self.beam,
            initial_prompt=prompt or None,
            temperature=0.0,
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=False,
        )
        return "".join(
            seg.text for seg in segments if _keep(seg.no_speech_prob, seg.avg_logprob)
        ).strip()


def engine_from(spec: str) -> Engine:
    """``mlx:<model>`` or ``faster:<model>[@beam]`` as an engine (not loaded yet)."""
    kind, _, model = spec.partition(":")
    if kind not in ENGINES or not model:
        raise ValueError(f"voice recogniser {spec!r}: expected mlx:<model> or faster:<model>")
    if kind == "mlx":
        return MlxEngine(model)
    name, _, beam = model.partition("@")
    return FasterEngine(name, beam=int(beam) if beam else 1)


def prompt_for(lang: str, duck_name: str | None = None) -> str:
    """The words the recogniser is primed with: the duck's name, then each command's first phrase and
    the first praise and scold words in ``lang`` (data from the one phrase table)."""
    words: list[str] = [duck_name.strip()] if duck_name and duck_name.strip() else []
    for verb in (*W.BUILTIN, *W.TRICKS):
        words.append(W.PHRASES[verb][lang][0].lstrip("="))
    for kind in ("praise", "scold"):
        words.extend(p.lstrip("=") for p in W.FEEDBACK_PHRASES[kind][lang][:2])
    cjk = lang in ("zh-Hant", "zh-Hans", "ja")
    return ("、" if cjk else ", ").join(words) + ("。" if cjk else ".")


def _echo(text: str, prompt: str) -> bool:
    """The recogniser wrote back its own prompt (it does on noise): three or more of its words in a row
    of a text that is mostly them."""
    words = [w for w in prompt.rstrip("。.").replace("、", ",").split(",") if w.strip()]
    hits = sum(1 for w in words if w.strip() and w.strip().lower() in text.lower())
    return hits >= 3


@dataclass(frozen=True, slots=True)
class Transcript:
    """What one clip came to: the words (``""``: nothing was said), how long the clip was, how long
    the recognition took, and why there are no words when there are none (``too_short`` | ``silent``
    | ``no_speech``)."""

    text: str
    lang: str
    audio_s: float
    took_s: float
    empty: str | None = None


class Refused(ValueError):
    """A clip the recogniser does not take (``reason``: ``too_long`` | ``odd_bytes`` | ``bad_lang``)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class Recognizer:
    """One engine behind a lock (one recognition at a time), loaded on first use or by :meth:`warm`."""

    def __init__(self, engine: Engine, *, prompt: bool = True,
                 clock: Callable[[], float] = time.perf_counter) -> None:  # fmt: skip
        self.engine = engine
        self.prompt = prompt
        self._clock = clock
        self._lock = threading.Lock()
        self.ready = False
        self.problem: str | None = None

    def warm(self) -> None:
        """Load the model now (in the background at server start), so the first press is not slow."""
        with self._lock:
            try:
                self.engine.load()
                self.ready = True
                self.problem = None
            except Exception as error:  # (a missing package or model: said on the page)
                self.problem = f"{type(error).__name__}: {error}"

    def status(self) -> dict[str, Any]:
        return {"engine": self.engine.name, "model": self.engine.model, "ready": self.ready,
                "problem": self.problem}  # fmt: skip

    def hear(self, pcm: bytes, lang: str, duck_name: str | None = None) -> Transcript:
        """The words in ``pcm`` (16 kHz mono 16-bit little-endian), read in ``lang``."""
        if lang not in WHISPER_LANG:
            raise Refused("bad_lang")
        if len(pcm) % 2:
            raise Refused("odd_bytes")
        audio_s = len(pcm) / 2 / SR
        if audio_s > MAX_S + 0.05:
            raise Refused("too_long")
        if audio_s < MIN_S:
            return Transcript("", lang, round(audio_s, 3), 0.0, "too_short")
        audio = A.to_float(pcm).astype(np.float32)
        if loudest_dbfs(audio) < SILENT_DBFS:
            return Transcript("", lang, round(audio_s, 3), 0.0, "silent")
        prompt = prompt_for(lang, duck_name) if self.prompt else ""
        with self._lock:
            begun = self._clock()
            text = self.engine.run(audio, WHISPER_LANG[lang], prompt)
            took = self._clock() - begun
            self.ready, self.problem = True, None
        if text and prompt and _echo(text, prompt):
            text = ""
        return Transcript(text, lang, round(audio_s, 3), round(took, 3),
                          None if text else "no_speech")  # fmt: skip


def loudest_dbfs(audio: np.ndarray[Any, Any], frame: int = SR * 30 // 1000) -> float:
    """The RMS level of the clip's loudest 30 ms, in dBFS (-120 for digital silence)."""
    if audio.size == 0:
        return -120.0
    n = max(1, audio.size // frame)
    blocks = audio[: n * frame].reshape(n, -1) if audio.size >= frame else audio.reshape(1, -1)
    rms = float(np.sqrt(np.max(np.mean(np.square(blocks, dtype=np.float64), axis=1))))
    return 20.0 * np.log10(max(rms, 1e-6))


def make(spec: str | None = None) -> tuple[Recognizer | None, str | None]:
    """The recogniser ``spec`` names (default :data:`DEFAULT_SPEC`, else :data:`FALLBACK_SPEC` when its
    package cannot be imported), or (None, why) when neither engine is installed or ``spec`` is
    ``off``. Nothing is loaded yet: call :meth:`Recognizer.warm`."""
    if spec == "off":
        return None, "off"
    why = "no engine"
    for choice in [spec] if spec else [DEFAULT_SPEC, FALLBACK_SPEC]:
        engine = engine_from(choice)
        try:
            __import__("mlx_whisper" if engine.name == "mlx" else "faster_whisper")
        except Exception as error:
            why = f"{type(error).__name__}: {error}"
            continue
        return Recognizer(engine), None
    return None, why
