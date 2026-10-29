"""Scenarios for the live session: what the duck's microphone and eye get, by reference.

A scenario is a short script in ``tools/live_voice/scenarios.json``: family members saying
scripted lines (the wavs of ``tests/fixtures/voice``) from places in the home, the duck's own
calls (the echo), pictures of the duck's eye (``tests/fixtures/live/eye``), and the status text
the duck sends the model, for one of four contrasting ducks (the page's presets: a bold explorer,
a shy snuggler, a slow dreamer, a lively songbird hatchling) and a microphone seed. From it this
module makes, deterministically:

* **the microphone's frames** — the very :class:`~duckmind.simweb.neuro_mic.MicStream` of LV0
  (the scripted wavs at the level the world gives them at that distance, the duck's own calls
  from its voice bank as an echo, the room's noise), no physics needed;
* **the inputs of the cassette key** (:meth:`ScenarioRun.inputs`): fixture ids with their file
  hashes, places, seeds, the model, the tool schema, the soul; and what the fake Gemini is told
  is said (:meth:`ScenarioRun.spoken_lines`);
* **a run** (:func:`run_scenario`): the frames, pictures and status texts fed to a
  :class:`~duckmind.live.session.LiveSession` on the scenario's clock — virtual for the fake and
  for replays, paced in real time for a recording.

Honest limits: the world here is a stand-in with fixed places (no walls, no furniture), the duck
sits at the origin; the level of a voice is the world's own function (``level_at`` of
:mod:`duckmind.world.neuro_world`, 3-D distance from a standing mouth to the duck's head).
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from duckmind.live.cassette import Cassette, CassetteMismatch, RecordingTransport
from duckmind.live.fake import SpokenLine
from duckmind.live.meter import UsageCaps, UsageMeter
from duckmind.live.session import FRAME_S, LiveSession, QueuedCall, SessionConfig
from duckmind.live.soul import SOUL_VERSION, ContextSnapshot, SoulInputs, build_soul
from duckmind.live.tools import TOOL_SCHEMA_VERSION, schema_hash
from duckmind.live.transport import DEFAULT_MODEL, Setup, Transport

ROOT = Path(__file__).resolve().parents[3]
SCENARIOS_PATH = ROOT / "tools" / "live_voice" / "scenarios.json"
CASSETTE_DIR = ROOT / "tests" / "fixtures" / "live"
EYE_DIR = CASSETTE_DIR / "eye"
SCENARIO_FORMAT = 1
DUCK_HEAD_M = 0.12  # the duck's head above the floor
MOUTH_M = 1.45  # a standing adult's mouth (neuro_world.MOUTH_H_M)
ADULT_M = 1.72


class FixtureMissing(Exception):
    """A fixture a scenario needs (a line, a picture, a voice bank) is not here."""


# ── the file ───────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Temperament:
    id: str
    genome_seed: int
    gd: int
    drives: Mapping[str, float]


@dataclass(frozen=True)
class SayEvent:
    t: float
    utterance: str
    who: str  # the sound character of the line (mom, dad, child, grandpa): data
    at_m: tuple[float, float]


@dataclass(frozen=True)
class DuckCall:
    t: float
    tag: str
    level: float


@dataclass(frozen=True)
class ImageSpan:
    """The same picture of the duck's eye, once every ``every_s`` from ``from_s`` to ``until_s``."""

    fixture: str
    from_s: float
    until_s: float
    every_s: float = 1.0

    def times(self) -> list[float]:
        out: list[float] = []
        t = self.from_s
        while t <= self.until_s + 1e-9:
            out.append(round(t, 3))
            t += self.every_s
        return out


@dataclass(frozen=True)
class ContextEvent:
    t: float
    behaviour: str = ""
    valence: float | None = None
    drives: Mapping[str, float] = field(default_factory=dict)  # over the temperament's
    known_people: tuple[str, ...] = ()
    in_view: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Scenario:
    id: str
    description: str
    length_s: float
    tail_s: float
    say: tuple[SayEvent, ...] = ()
    duck_calls: tuple[DuckCall, ...] = ()
    images: tuple[ImageSpan, ...] = ()
    context: tuple[ContextEvent, ...] = ()
    temperaments: tuple[str, ...] = ()  # empty: all
    seeds: tuple[int, ...] = ()  # empty: the file's
    needs_eye: bool = False

    @property
    def total_s(self) -> float:
        return self.length_s + self.tail_s


@dataclass(frozen=True)
class ScenarioFile:
    version: int
    duck_name: str
    seeds: tuple[int, ...]
    temperaments: Mapping[str, Temperament]
    scenarios: Mapping[str, Scenario]

    def matrix(self) -> list[tuple[str, str, int]]:
        """Every (scenario, temperament, seed) to record, in file order."""
        out: list[tuple[str, str, int]] = []
        for scenario in self.scenarios.values():
            temps = scenario.temperaments or tuple(self.temperaments)
            seeds = scenario.seeds or self.seeds
            out += [(scenario.id, tid, seed) for tid in temps for seed in seeds]
        return out


def _pair(raw: Any) -> tuple[float, float]:
    return (float(raw[0]), float(raw[1]))


def load_scenario_file(path: Path | str | None = None) -> ScenarioFile:
    raw = json.loads(Path(path or SCENARIOS_PATH).read_text(encoding="utf-8"))
    if int(raw.get("version", 0)) != SCENARIO_FORMAT:
        raise ValueError(
            f"scenario file version {raw.get('version')!r}, expected {SCENARIO_FORMAT}"
        )
    temperaments = {
        tid: Temperament(
            id=tid,
            genome_seed=int(item["genome_seed"]),
            gd=int(item["gd"]),
            drives={k: float(v) for k, v in item.get("drives", {}).items()},
        )
        for tid, item in raw["temperaments"].items()
    }
    scenarios: dict[str, Scenario] = {}
    for item in raw["scenarios"]:
        sid = str(item["id"])
        if sid in scenarios:
            raise ValueError(f"scenario {sid!r} twice")
        scenarios[sid] = Scenario(
            id=sid,
            description=str(item.get("description", "")),
            length_s=float(item["length_s"]),
            tail_s=float(item.get("tail_s", 5.0)),
            say=tuple(
                SayEvent(
                    float(e["t"]), str(e["utterance"]), str(e.get("who", "")), _pair(e["at_m"])
                )
                for e in item.get("say", ())
            ),
            duck_calls=tuple(
                DuckCall(float(e["t"]), str(e["tag"]), float(e.get("level", 0.8)))
                for e in item.get("duck_calls", ())
            ),
            images=tuple(
                ImageSpan(
                    str(e["fixture"]),
                    float(e["from_s"]),
                    float(e["until_s"]),
                    float(e.get("every_s", 1.0)),
                )
                for e in item.get("images", ())
            ),
            context=tuple(
                ContextEvent(
                    t=float(e["t"]),
                    behaviour=str(e.get("behaviour", "")),
                    valence=None if e.get("valence") is None else float(e["valence"]),
                    drives={k: float(v) for k, v in e.get("drives", {}).items()},
                    known_people=tuple(e.get("known_people", ())),
                    in_view=tuple(e.get("in_view", ())),
                    notes=tuple(e.get("notes", ())),
                )
                for e in item.get("context", ())
            ),
            temperaments=tuple(item.get("temperaments", ())),
            seeds=tuple(int(s) for s in item.get("seeds", ())),
            needs_eye=bool(item.get("needs_eye", False)),
        )
    unknown = {t for s in scenarios.values() for t in s.temperaments} - set(temperaments)
    if unknown:
        raise ValueError(f"scenarios name temperaments that are not defined: {sorted(unknown)}")
    return ScenarioFile(
        version=SCENARIO_FORMAT,
        duck_name=str(raw["duck"]["name"]),
        seeds=tuple(int(s) for s in raw["seeds"]),
        temperaments=temperaments,
        scenarios=scenarios,
    )


# ── clocks ─────────────────────────────────────────────────────────────────────────


class ManualClock:
    """The scenario's own seconds: set by the runner (the fake and replays; no waiting)."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def set(self, t: float) -> None:
        self.t = float(t)


class RealtimeClock:
    """Scenario seconds are wall seconds since it was made: a recording is paced by it, because
    the server's voice activity detection needs the audio at the speed it was spoken."""

    def __init__(self, monotonic: Callable[[], float] = time.monotonic) -> None:
        self._mono = monotonic
        self._t0 = monotonic()

    def __call__(self) -> float:
        return self._mono() - self._t0

    def wait_until(self, t: float) -> None:
        delay = t - self()
        if delay > 0:
            time.sleep(delay)


# ── one run ────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Step:
    t: float
    frame: Any  # MicFrame
    images: tuple[tuple[str, bytes], ...]
    context: ContextEvent | None


class ScenarioRun:
    """One (scenario, temperament, seed): its frames, its pictures, its inputs."""

    def __init__(
        self,
        file: ScenarioFile,
        scenario_id: str,
        temperament_id: str,
        seed: int,
        *,
        library: Any = None,
        bank: Any = None,
        eye_dir: Path | None = None,
        require_bank: bool = True,
        model: str = DEFAULT_MODEL,
        session_config: SessionConfig | None = None,
    ) -> None:
        self.file = file
        self.scenario = file.scenarios[scenario_id]
        self.temperament = file.temperaments[temperament_id]
        self.seed = int(seed)
        self.model = model
        self.session_config = session_config or SessionConfig(model=model)
        self.eye_dir = Path(eye_dir) if eye_dir else EYE_DIR
        self.require_bank = require_bank
        self._library = library
        self._bank = bank
        self._images: dict[str, bytes] = {}

    @property
    def run_id(self) -> str:
        return f"{self.scenario.id}__{self.temperament.id}__s{self.seed}"

    # ── pieces ────────────────────────────────────────────────────────────────

    def library(self) -> Any:
        if self._library is None:
            from duckmind.world.neuro_utterances import UtteranceLibrary

            lib = UtteranceLibrary.default()
            if lib is None:
                raise FixtureMissing("the scripted lines (tests/fixtures/voice) are not here")
            self._library = lib
        return self._library

    def voice_seed(self) -> int:
        from duckmind.simweb import voice as V

        return V.voice_seed(self.temperament.genome_seed)

    def bank(self) -> Any:
        if self._bank is None:
            from duckmind.simweb import voice as V

            self._bank = V.VoiceBank(None, V.DEFAULT_ROOT)
        return self._bank

    def image(self, fixture: str) -> bytes:
        if fixture not in self._images:
            path = self.eye_dir / f"{fixture}.jpg"
            if not path.is_file():
                raise FixtureMissing(f"the duck-eye picture {fixture!r} is not here ({path})")
            self._images[fixture] = path.read_bytes()
        return self._images[fixture]

    def soul_inputs(self) -> SoulInputs:
        from duckmind.neuro.genome import Genome

        genome = Genome.from_seed(self.temperament.genome_seed, gd=self.temperament.gd)
        first = self.context_at_start()
        return SoulInputs(
            name=self.file.duck_name,
            genome={
                k: (str(v) if not isinstance(v, (int, float, list, dict, type(None))) else v)
                for k, v in genome.summary().items()
            },
            learner={},
            drives=self.drives_for(first),
            known_people=first.known_people,
            behaviour=first.behaviour,
        )

    def context_at_start(self) -> ContextEvent:
        events = sorted(self.scenario.context, key=lambda e: e.t)
        return events[0] if events and events[0].t <= 0 else ContextEvent(t=0.0)

    def drives_for(self, event: ContextEvent) -> dict[str, float]:
        out = dict(self.temperament.drives)
        out.update(event.drives)
        return out

    def snapshot(self, event: ContextEvent) -> ContextSnapshot:
        return ContextSnapshot(
            behaviour=event.behaviour,
            valence=event.valence,
            drives=self.drives_for(event),
            known_people=event.known_people,
            in_view=event.in_view,
            notes=event.notes,
        )

    def soul_text(self) -> str:
        return build_soul(self.soul_inputs())

    def say_geometry(self, say: SayEvent) -> dict[str, Any]:
        """Where and how loud one line is at the duck's head (the world's own level function)."""
        from duckmind.world.neuro_world import SOUND_DB_1M, level_at

        line = self.library().get(say.utterance)
        height = MOUTH_M * _persona_scale(line.persona)
        distance = math.sqrt(say.at_m[0] ** 2 + say.at_m[1] ** 2 + (height - DUCK_HEAD_M) ** 2)
        called = bool(line.name) and line.name == self.file.duck_name
        kind = "name" if called else "hello" if line.intent == "greet" else "voice"
        db = level_at(SOUND_DB_1M[kind], distance)
        return {
            "kind": kind,
            "distance_m": round(distance, 3),
            "db": round(db, 2),
            "duration_s": line.duration_s,
            "text": line.text,
        }

    def spoken_lines(self) -> list[SpokenLine]:
        """What is said and when (the truth the fake Gemini is told)."""
        lib = self.library()
        return [
            SpokenLine(s.t, s.t + lib.get(s.utterance).duration_s, lib.get(s.utterance).text, s.who)
            for s in self.scenario.say
        ]

    def image_labels(self) -> dict[str, list[str]]:
        """Picture hash → what it shows (for the fake: the fixture's name stands for it)."""
        labels = {"eye_toy": ["玩具"], "eye_empty": []}
        out: dict[str, list[str]] = {}
        for span in self.scenario.images:
            out[hashlib.sha256(self.image(span.fixture)).hexdigest()] = labels.get(span.fixture, [])
        return out

    def setup_facts(self) -> dict[str, Any]:
        """How the connection is set up apart from the soul and the tools (a changed default
        there is another recording)."""
        base = Setup(system_instruction="", tools=[], model=self.model)
        return {
            "response_modalities": list(base.response_modalities),
            "input_transcription": base.input_transcription,
            "output_transcription": base.output_transcription,
            "compression": base.compression,
            "vad": {
                "start": base.vad_start,
                "end": base.vad_end,
                "prefix_ms": base.vad_prefix_ms,
                "silence_ms": base.vad_silence_ms,
            },
            "media_resolution": self.session_config.media_resolution,
        }

    # ── the inputs of the cassette key ────────────────────────────────────────

    def inputs(self) -> dict[str, Any]:
        from duckmind.simweb import voice as V

        lib = self.library()
        manifest = json.loads((Path(lib.root) / "manifest.json").read_text(encoding="utf-8"))
        shas = {u["id"]: u.get("sha256", "") for u in manifest["utterances"]}
        soul = self.soul_text()
        cfg = self.session_config
        events = sorted(self.scenario.context, key=lambda e: e.t) or [ContextEvent(t=0.0)]
        return {
            "format": SCENARIO_FORMAT,
            "scenario": self.scenario.id,
            "temperament": self.temperament.id,
            "seed": self.seed,
            "length_s": self.scenario.length_s,
            "tail_s": self.scenario.tail_s,
            "duck": {
                "name": self.file.duck_name,
                "genome_seed": self.temperament.genome_seed,
                "gd": self.temperament.gd,
                "voice_seed": self.voice_seed(),
                "voice_bank_version": V.BANK_VERSION,
                "at_m": [0.0, 0.0],
            },
            "model": self.model,
            "tool_schema": {"version": TOOL_SCHEMA_VERSION, "hash": schema_hash()},
            "soul": {"version": SOUL_VERSION, "sha256": hashlib.sha256(soul.encode()).hexdigest()},
            "session": {
                "context_period_s": cfg.context_period_s,
                "image_period_s": cfg.image_period_s,
                "hold_audio_s": cfg.hold_audio_s,
            },
            "setup": self.setup_facts(),
            "say": [
                {
                    "t": s.t,
                    "utterance": s.utterance,
                    "sha256": shas.get(s.utterance, ""),
                    "who": s.who,
                    "at_m": list(s.at_m),
                }
                for s in self.scenario.say
            ],
            "duck_calls": [
                {"t": c.t, "tag": c.tag, "level": c.level} for c in self.scenario.duck_calls
            ],
            "images": [
                {
                    "fixture": i.fixture,
                    "sha256": hashlib.sha256(self.image(i.fixture)).hexdigest(),
                    "from_s": i.from_s,
                    "until_s": i.until_s,
                    "every_s": i.every_s,
                }
                for i in self.scenario.images
            ],
            "context": [
                {
                    "t": e.t,
                    "behaviour": e.behaviour,
                    "valence": e.valence,
                    "drives": self.drives_for(e),
                    "known_people": list(e.known_people),
                    "in_view": list(e.in_view),
                    "notes": list(e.notes),
                }
                for e in events
            ],
        }

    # ── the steps ─────────────────────────────────────────────────────────────

    def make_mic(self) -> Any:
        from duckmind.simweb.neuro_mic import MicStream, VoiceBankCalls

        calls = None
        bank = self.bank()
        vseed = self.voice_seed()
        if bank.ready(vseed):
            calls = VoiceBankCalls(bank, vseed)
        elif self.require_bank and self.scenario.duck_calls:
            raise FixtureMissing(f"the duck's voice bank (voice seed {vseed}) is not rendered")
        return MicStream(seed=self.seed, library=self.library(), calls=calls, wall=lambda: 0.0)

    def steps(self) -> Iterator[Step]:
        """Every 20 ms frame of the scenario (``length_s`` + ``tail_s``), with the pictures and the
        status text that fall in it."""
        lib = self.library()
        mic = self.make_mic()
        sounds: list[dict[str, Any]] = []
        for number, say in enumerate(sorted(self.scenario.say, key=lambda s: s.t), start=1):
            geo = self.say_geometry(say)
            sounds.append(
                {
                    "id": number,
                    "kind": geo["kind"],
                    "t0": say.t,
                    "t1": say.t + lib.get(say.utterance).duration_s,
                    "db": geo["db"],
                    "src": f"person:{say.who or 'guest'}",
                    "utterance": say.utterance,
                }
            )
        calls = sorted(self.scenario.duck_calls, key=lambda c: c.t)
        pictures = sorted((t, span.fixture) for span in self.scenario.images for t in span.times())
        contexts = sorted(self.scenario.context, key=lambda e: e.t)
        n = int(round(self.scenario.total_s / FRAME_S))
        ci = pi = ki = 0
        for i in range(n):
            t = round(i * FRAME_S, 6)
            due_calls = []
            while ci < len(calls) and calls[ci].t <= t + 1e-9:
                due_calls.append({"tag": calls[ci].tag, "level": calls[ci].level})
                ci += 1
            live = [s for s in sounds if s["t0"] <= t + 1e-9]
            frame = mic.tick(t, live, calls=due_calls, wall=t)
            pics: list[tuple[str, bytes]] = []
            while pi < len(pictures) and pictures[pi][0] <= t + 1e-9:
                pics.append((pictures[pi][1], self.image(pictures[pi][1])))
                pi += 1
            ctx = None
            while ki < len(contexts) and contexts[ki].t <= t + 1e-9:
                ctx = contexts[ki]
                ki += 1
            yield Step(t, frame, tuple(pics), ctx)


def _persona_scale(persona: str) -> float:
    from duckmind.world.people import persona as get_persona

    try:
        return get_persona(persona).height_m / ADULT_M
    except ValueError:
        return 1.0


# ── running one against a session ──────────────────────────────────────────────────


@dataclass
class ScenarioResult:
    run_id: str
    session: LiveSession
    calls: list[QueuedCall]
    transport: Transport
    frames: int
    images_sent: int
    wall_s: float

    def log(self, kind: str) -> list[dict[str, Any]]:
        return [e for e in self.session.log if e.get("kind") == kind]


def run_scenario(
    run: ScenarioRun,
    transport: Transport,
    *,
    clock: Callable[[], float] | None = None,
    caps: UsageCaps | None = None,
    drain_s: float = 1.0,
    on_step: Callable[[Step, LiveSession], None] | None = None,
) -> ScenarioResult:
    """Feed a scenario to a fresh session over ``transport``. A virtual clock (default) is set by
    the runner; a :class:`RealtimeClock` paces it by the wall. Raises the transport's own
    ``failure`` (a replay of the wrong cassette) rather than letting the session hide it."""
    clock = clock or ManualClock()
    session = LiveSession(
        transport,
        run.soul_text(),
        clock=clock,
        config=run.session_config,
        meter=UsageMeter(caps),
    )
    started = time.monotonic()
    frames = 0
    last_t = 0.0
    for step in run.steps():
        if isinstance(clock, ManualClock):
            clock.set(step.t)
        elif isinstance(clock, RealtimeClock):
            clock.wait_until(step.t)
        last_t = step.t
        if step.context is not None:
            session.set_context(run.snapshot(step.context))
        session.feed_audio(step.frame)
        frames += 1
        for _fixture, jpeg in step.images:
            session.feed_image(jpeg)
        session.pump()
        if on_step is not None:
            on_step(step, session)
    end = last_t + drain_s
    if isinstance(clock, ManualClock):
        t = last_t
        while t < end:
            t += FRAME_S
            clock.set(t)
            session.pump()
    elif isinstance(clock, RealtimeClock):
        while clock() < end:
            session.pump()
            time.sleep(0.02)
    failure = getattr(transport, "failure", None)
    calls = session.take_calls()
    images = int(session.meter.totals["frames"])
    session.close()
    if failure is not None:
        raise failure
    return ScenarioResult(
        run.run_id, session, calls, transport, frames, images, time.monotonic() - started
    )


def record_scenario(
    run: ScenarioRun,
    transport: Transport,
    clock: Callable[[], float],
    *,
    caps: UsageCaps | None = None,
    meta: Mapping[str, Any] | None = None,
) -> tuple[Cassette, ScenarioResult]:
    """Run a scenario over ``transport`` (the real one, in real time) and keep what came back."""
    recorder = RecordingTransport(transport, run.scenario.id, run.inputs())
    result = run_scenario(run, recorder, clock=clock, caps=caps)
    status = result.session.status()
    complete = status["connections"] >= 1 and result.session.counts.get("closed", 0) == 0
    info = {
        "run_id": run.run_id,
        "faults": result.session.faults,
        "wall_s": round(result.wall_s, 1),
    }
    info.update(meta or {})
    return recorder.cassette(complete=complete, meta=info), result


def cassette_path(run_id: str, folder: Path | None = None) -> Path:
    return (folder or CASSETTE_DIR) / f"{run_id}.json"


def replay(run: ScenarioRun, cassette: Cassette) -> ScenarioResult:
    """Replay a recorded scenario through a fresh session (raises on a mismatch)."""
    from duckmind.live.cassette import ReplayTransport

    return run_scenario(run, ReplayTransport(cassette, run.inputs()))


__all__ = [
    "CASSETTE_DIR",
    "EYE_DIR",
    "SCENARIOS_PATH",
    "CassetteMismatch",
    "FixtureMissing",
    "ManualClock",
    "RealtimeClock",
    "Scenario",
    "ScenarioFile",
    "ScenarioResult",
    "ScenarioRun",
    "Sequence",
    "Step",
    "cassette_path",
    "load_scenario_file",
    "record_scenario",
    "replay",
    "run_scenario",
]
