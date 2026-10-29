"""The neural duck in live physics (prototype mode, ``--brain neuro``): neuro:live's driver.

In prototype mode the neural brain (:mod:`duckmind.neuro`), not Duck Core's planner, drives the
physical body of the simulator page. This module is the only place of the prototype with
sockets, threads and clocks; everything it runs per tick is pure and clock-injected:

1. **robotd, in lockstep with the physics.** When the live body is ready the driver becomes
   robotd's only writer (:attr:`duckmind.livebody.LiveBody.passive`, spec §6: one writer at a
   time — the page's own control loop then sends nothing) and opens duck-body's three robotd
   connections (:class:`~duckmind.duckbody.robotd_client.RobotdClient`: state, command, safety).
   It runs one tick per ``robot.state`` (robotd's 50 Hz; the world server steps the physics once
   per robotd write, so a tick is one 20 ms physics period). The brain's clock is the tick's
   ``t_ns`` (robotd's ``CLOCK_MONOTONIC``) made relative to the first tick, in ms.
2. **Senses** (neuro:senses, :func:`duckmind.world.neuro_senses.live_senses`): the tick's sensor
   doors (:class:`Doors`: ``robot.state``, duck-body's proprio as the body link ran it, tofd's
   frame and its DS-v1 depth summary for a duck with the ToF module, the dock contact ``pg``)
   plus, for the world-side sensor models only, the world's truth (``x_world`` — with the
   ``neuro`` section of the world extension :mod:`duckmind.world.neuro_world`, which the server
   loads in prototype mode — and the physics frame) become a
   :class:`~duckmind.neuro.contract.SensorFrame`. The brain never sees the doors or the truth.
3. **Brain** (neuro:brain): ``NeuroBrain(genome, seed).step(t_ms, frame) -> (MotorCommand,
   BrainTelemetry)``, the genome drawn from Duck Core's diversity design for the seed.
4. **Motor** (neuro:motor, :class:`LiveMotor`): the command's motor program → a
   :class:`~duckmind.neuro.bodylink.BodyRequest` (twist, pre-compensated head offsets, beak,
   posture) and the calls it voices this tick (vocal tags, shown on the page: robotd's
   ``robot.sound`` is not on duck-body's allowlist yet).
5. **Body link** (:mod:`duckmind.neuro.bodylink`): duck-body's safety (falls → righting,
   pickup, battery, reflex stops, walking rules, the forbidden-command guard) and the deadman
   stream; its intents go to robotd.

The page's stimulus buttons go to the WORLD (:meth:`NeuroDriver.inject`: the world op
``x_neuro_stim``), never to the brain; :meth:`NeuroDriver.switch` puts another duck's brain on the
same body. When a lane is missing or fails to load, :mod:`duckmind.neuro.livestub` stands in (a
real-doors-only frame, a seeded wanderer, a simple head/beak mapping) and the page and the
telemetry say so (``lanes``, ``notes``).

Telemetry (:meth:`NeuroDriver.telemetry`, for the page and ``neuro_api``): 10 Hz samples of what
the duck does and why, the latest brain telemetry, sensor frame (with each channel's source:
real / eye / model) and motor output, vocal and body events, the tick statistics and the body's
realtime reading.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from duckmind.duckbody import protocol as P
from duckmind.duckbody.robotd_client import RobotdClient, RobotdError, RobotState
from duckmind.i18n import carried_words, lst, msg
from duckmind.life.clock import TPE, DuckClock
from duckmind.neuro import contract as C
from duckmind.neuro.bodylink import (
    GATE_ZH,
    BodyLink,
    BodyRequest,
    LinkConfig,
    LinkOut,
    config_for_motor,
)
from duckmind.neuro.livestub import STUB_NOTE_ZH, StubBrain, StubMotor, body_frame
from duckmind.neuro.remote import RemoteHand
from duckmind.simweb.loopwatch import LoopWatch
from duckmind.simweb.neuro_witness import Tick as WitnessTick
from duckmind.simweb.neuro_witness import Witness
from duckmind.skilltrain import gt_wiring as SW
from duckmind.skilltrain import guard as SG

SAMPLE_EVERY_TICKS = 5  # 50 Hz ticks → 10 Hz telemetry samples
SAMPLES_KEEP = 1200  # two minutes of samples
EVENTS_KEEP = 200
AUX_HZ = 5.0  # the dock contact, the world's truth for the sensor models, the duck's clock
AUX_FAST_S = 0.04  # … and this often for a moment after a page stimulus (fid:integrate)
AUX_FAST_FOR_S = 1.0
BACKLOG_TICKS = 3  # more robot.state ticks waiting than this: skip to the newest
READ_TIMEOUT_S = 0.2
RECONNECT_S = 0.5
ATTACH_WAIT_S = 30.0
FAILING_TICKS = 100  # a lane failing this many ticks in a row stops the driver (2 s)
ERRAND_MAX_RANGE_M = 12.0  # a place pointed at farther than this from the duck is refused (the home is a few metres across)
VOCAL_ZH = {
    **{tag: tag for tag in C.VOCAL_TAGS},
    **{
        "chirp": msg("neuro.vocal.chirp"),
        "inquire": msg("neuro.vocal.inquire"),
        "coo": msg("neuro.vocal.coo"),
        "greet": msg("neuro.vocal.greet"),
        "wheee": msg("neuro.vocal.wheee"),
        "alarm": msg("neuro.vocal.alarm"),
    },
}

# ── the page's scenario stimuli, into the WORLD (never into the brain) ────────
# What has a body in the physics goes to the world's own ops (the visible hand, a person, a
# ball, the lamps); a sound or a wave, which the physics cannot make, becomes a world event at a
# place in the home (the nearest person's, else a spot about a metre from the duck) that the
# senses' world-side models (neuro:senses) hear or see — or not — like the duck's sensors would.
STIMULI_KEEP_MS = 60_000
# The page's 「很大聲」 bang is a very loud one — what the verification's L2 bang is (100 dB at 1 m,
# duckmind.verify.spec_v2.L2_BANG_DB1M: at the head 92–96 dB from 1.5–2.5 m, never under 90) — not the
# world's own everyday 90 dB "something dropped" (live check 2026-10-05: at that level the bold
# explorer startled at 1 of 6 and fled at 0 of 6, the rest only turned its head)
PAGE_BANG_DB1M = 100.0
# … and the page's push is the verification's push (the world's top force, 7 N,
# duckmind.verify.spec_v3.PUSH_STRENGTH): its words say 「真的力氣，可能會跌倒」, and at the world's
# default 4 N the bold explorer was not even moved (0 of 4 reacted)
PAGE_PUSH_STRENGTH = 1.0
SOUND_STIMULI: dict[str, dict[str, Any]] = {  # dB at 1 m from the source, how long
    "clap": {"sound": "clap", "db": 75.0, "ms": 150, "zh": msg("neuro.world_fallback.clap")},
    "call_name": {
        "sound": "name",
        "db": 65.0,
        "ms": 900,
        "zh": msg("neuro.world_fallback.call_name"),
    },
    "voice": {"sound": "voice", "db": 60.0, "ms": 3000, "zh": msg("neuro.world_fallback.voice")},
    "hello": {"sound": "hello", "db": 62.0, "ms": 700, "zh": msg("neuro.world_fallback.hello")},
    "bang": {
        "sound": "bang",
        "db": PAGE_BANG_DB1M,
        "ms": 300,
        "zh": msg("neuro.world_fallback.bang"),
    },
    "door": {"sound": "door", "db": 70.0, "ms": 900, "zh": msg("neuro.world_fallback.door")},
    "music": {"sound": "music", "db": 62.0, "ms": 30_000, "zh": msg("neuro.world_fallback.music")},
}
VISUAL_STIMULI: dict[str, dict[str, Any]] = {
    "wave": {
        "visual": "motion",
        "ms": 2500,
        "ahead_m": 0.7,
        "zh": msg("neuro.world_fallback.wave"),
    },
}
HAND_STIMULI: dict[str, dict[str, Any]] = {  # x_hand scripts (world/hand.py)
    "pet_head": {"do": "touch", "where": "head", "seconds": 3.0, "speed": "slow"},
    "stroke_back": {"do": "touch", "where": "back", "seconds": 3.0, "speed": "slow"},
    "poke": {"do": "touch", "where": "back", "seconds": 0.5, "speed": "fast"},
    "hand_loom": {"do": "near", "seconds": 1.2, "speed": "fast"},
    "pick_up": {"do": "pickup"},
    "set_down": {"do": "place", "to": "here"},
}
HAND_ZH = {
    "pet_head": msg("neuro.world_fallback.pet_head"),
    "stroke_back": msg("neuro.world_fallback.stroke_back"),
    "poke": msg("neuro.world_fallback.poke"),
    "hand_loom": msg("neuro.world_fallback.hand_loom"),
    "pick_up": msg("neuro.world_fallback.pick_up"),
    "set_down": msg("neuro.world_fallback.set_down"),
}
WALK_BY_S = 8.0  # a family member who walks by leaves again after this long
# Rhythm trains (the ethology report's Miller rule: a duckling freezes and hushes at slow
# knocking, calls and comes at fast clapping): each a series of world sounds, never a message.
TRAINS: dict[str, dict[str, Any]] = {
    "knock_train": {
        "stim": "knock",
        "count": 5,
        "hz": 1.2,
        "zh": msg("neuro.world.knock_train.done"),
    },
    "clap_train": {"stim": "clap", "count": 8, "hz": 3.5, "zh": msg("neuro.world.clap_train.done")},
}


# ── the duck's clock (prototype mode) ────────────────────────────────────────

DAY_START_MINUTE = 10 * 60.0  # the demo starts at 10:00 on the duck's clock: its active day
DAY_CHOICES: tuple[tuple[str, float], ...] = (  # the page's plain time-of-day choice
    (msg("neuro.day_part.early_morning"), 7 * 60 + 30.0),
    (msg("neuro.day_part.morning"), 10 * 60.0),
    (msg("neuro.day_part.afternoon"), 15 * 60.0),
    (msg("neuro.day_part.evening"), 18 * 60 + 30.0),
    (msg("neuro.day_part.night"), 21 * 60 + 30.0),
    (msg("neuro.day_part.late_night"), 60.0),
)


def minute_zh(minute: float) -> str:
    """A time of day in plain words, e.g. 上午 10:05, 晚上 9:30."""
    m = int(minute) % 1440
    h, mm = divmod(m, 60)
    part = (
        msg("neuro.day_part.late_night")
        if h < 5
        else msg("neuro.day_part.dawn")
        if h < 7
        else msg("neuro.day_part.early_morning")
        if h < 9
        else msg("neuro.day_part.morning")
        if h < 12
        else msg("neuro.day_part.noon")
        if h < 13
        else msg("neuro.day_part.afternoon")
        if h < 17
        else msg("neuro.day_part.evening")
        if h < 19
        else msg("neuro.day_part.night")
    )
    shown = h if h <= 12 else h - 12
    return msg("neuro.day.at_time", part=part, shown=shown, mm=f"{mm:02d}")


class DayClock(DuckClock):
    """The duck's time in prototype mode (its circadian drive, the room's daylight): the one
    calendar clock of :mod:`duckmind.life.clock`. ``DayClock()`` starts today at 10:00 (the demo
    and the tests: its active day, whatever the hour the owner watches) and runs at the wall
    clock's pace; ``DayClock(None)`` is the real calendar (the live duck, owner decision
    2026-10-10: the time is the real time, with a speed factor only for tests). The page's
    time-of-day buttons move only the demo offset (:meth:`DuckClock.set`), never the calendar."""

    def __init__(
        self,
        minute: float | None = DAY_START_MINUTE,
        *,
        start: datetime | None = None,
        speed: float = 1.0,
        wall: Callable[[], float] = time.time,
    ):
        if start is None and minute is not None:
            today = datetime.fromtimestamp(float(wall()), TPE).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            start = today + timedelta(minutes=float(minute) % 1440.0)
        super().__init__(start, speed=speed, wall=wall)

    def view(self, words: Callable[[float], Any] | None = None) -> dict[str, Any]:
        return super().view(words or minute_zh)


def parse_day_time(text: str | None) -> float:
    """``HH:MM`` (or minutes) → minute of the day; the default start when empty."""
    raw = str(text or "").strip()
    if not raw:
        return DAY_START_MINUTE
    if ":" in raw:
        h, _, m = raw.partition(":")
        return (int(h) * 60 + int(m or 0)) % 1440
    return float(raw) % 1440.0


# ── which duck ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NeuroSpec:
    """``--neuro-duck``: a seed (``"7"``), a seed at a growth day (``"7:62"``) or a preset name
    of the genome module; empty: the simulator's own ``--seed``. ``record``: a JSONL file of
    every tick's frame and command (for replaying the live run headless)."""

    duck: str = ""
    seed: int = 7
    gd: int | None = None
    record: Path | None = None
    # The life duck (docs/design/GROWTH-G1): its genome is built from Duck Core's state by this
    # builder (called once the skills are installed), not drawn from ``duck``; ``life`` marks the
    # brain as the one whose mind is saved and brought back (duckmind.simweb.life_host).
    genome_builder: Any = field(default=None, compare=False, repr=False)
    life: bool = False

    def parsed(self) -> tuple[int | None, int | None, str | None]:
        """(seed, growth day, preset name): a preset has no seed of its own here."""
        text = (self.duck or "").strip()
        if not text:
            return self.seed, self.gd, None
        head, _, tail = text.partition(":")
        if head.lstrip("-").isdigit():
            gd = int(tail) if tail.strip().isdigit() else self.gd
            return int(head), gd, None
        return None, self.gd, text


@dataclass
class Lanes:
    """The prototype's lanes as the driver calls them, with stand-ins where a lane is missing.

    ``senses(doors) -> SensorFrame``; ``brain.step(t_ms, frame) -> (MotorCommand,
    BrainTelemetry)``; ``motor(t_ms, cmd, frame, state) -> (BodyRequest, vocals)``."""

    brain: Any
    senses: Callable[[Doors], C.SensorFrame] | None = None
    motor: (
        Callable[
            [int, C.MotorCommand, C.SensorFrame, RobotState],
            tuple[BodyRequest, list[dict[str, Any]]],
        ]
        | None
    ) = None
    names: dict[str, str] = field(default_factory=dict)  # lane → "real" | "stub"
    notes: list[str] = field(default_factory=list)
    genome: dict[str, Any] | None = None
    link: LinkConfig | None = None  # the body link's head envelopes from the motor lane


def _close_lane_sound(lanes: Lanes) -> None:
    sound = getattr(lanes, "sound", None)
    engine = getattr(sound, "engine", None)
    close = getattr(engine, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def load_lanes(spec: NeuroSpec) -> Lanes:
    """The brain for ``spec`` and the senses/motor lanes, each the real one when its module is
    there and works, else the stand-in (and a plain note why)."""
    notes: list[str] = []
    names: dict[str, str] = {}
    seed, gd, preset = spec.parsed()
    brain: Any = None
    genome_view: dict[str, Any] | None = None
    try:
        genome_mod = importlib.import_module("duckmind.neuro.genome")
        integration = importlib.import_module("duckmind.neuro.integration")
        # Skills must be registered before the genome is built so unlocks and gains
        # are evaluated against the complete repertoire without redrawing identity.
        integration.install_default_skills()
        genome: Any
        if spec.genome_builder is not None:
            genome = spec.genome_builder()
        elif preset is not None:
            make = getattr(genome_mod, "preset", None)
            if make is None:
                raise ValueError(f"no preset {preset!r} (the genome module has no presets)")
            genome = make(preset)
        else:
            kwargs = {} if gd is None else {"gd": int(gd)}
            genome = genome_mod.Genome.from_seed(
                int(seed if seed is not None else spec.seed), **kwargs
            )
        brain = integration.make_brain(genome, int(getattr(genome, "seed", seed or spec.seed)))
        summary = getattr(genome, "summary", None)
        genome_view = summary() if callable(summary) else None
        names["brain"] = "real"
    except Exception as error:  # the brain lane is not there yet (or failed): the stand-in
        brain = StubBrain(seed=int(seed if seed is not None else spec.seed))
        names["brain"] = "stub"
        notes.append(
            msg(
                "neuro.lane_note.load_lanes",
                stub_note=STUB_NOTE_ZH,
                error_type=type(error).__name__,
                error_text=error,
            )
        )
    senses = _load_senses(notes, names, seed if seed is not None else spec.seed)
    motor, link = _load_motor(notes, names, brain, seed if seed is not None else spec.seed)
    if motor is not None and hasattr(brain, "play_out"):
        try:
            from duckmind.simweb.neuro_runtime import IntegratedMotor

            motor = IntegratedMotor(motor, brain)
            if link is not None:
                link = replace(link, mouth_max=1.0)
        except Exception as error:
            notes.append(f"play adapter: {type(error).__name__}: {error}")
    lanes = Lanes(brain, senses, motor, names, notes, genome_view, link)
    try:
        sound = importlib.import_module("duckmind.simweb.neuro_sound_live")
        sound.attach_lanes(lanes, microduck=sound.microduck_from_env())
        engine = getattr(lanes.sound, "engine", None)
        engine_view = engine.view() if engine is not None else {}
        names["sound"] = "real" if engine is not None and engine_view.get("helper") else "partial"
        if names["sound"] != "real":
            notes.append("sound: skills wired; streamed PCM helper unavailable")
    except Exception as error:
        names["sound"] = "stub"
        notes.append(f"sound: sound adapter unavailable ({type(error).__name__}: {error})")
    return lanes


def _load_senses(
    notes: list[str], names: dict[str, str], seed: int
) -> Callable[[Doors], C.SensorFrame] | None:
    """neuro:senses' :class:`~duckmind.world.neuro_senses.SenseHub` (world side: it reads the
    world's truth for its sensor models; the brain never imports it), else the stand-in."""
    try:
        module = importlib.import_module("duckmind.world.neuro_integration")
        hub = module.live_senses(seed=int(seed))
        names["senses"] = "real"
        return hub  # type: ignore[no-any-return]
    except Exception as error:
        names["senses"] = "stub"
        notes.append(
            msg("neuro.lane_note.load_senses", error_type=type(error).__name__, error_text=error)
        )
        return None


def world_extensions() -> str:
    """The world server's extensions in prototype mode: the household, the sensors and the
    neural duck's world side (``duckmind.world.neuro_world``: the view, sound and touch truth
    for the sensor models, and the page's stimuli as world events)."""
    try:
        module = importlib.import_module("duckmind.world.neuro_integration")
        return str(module.EXTENSIONS)
    except Exception:
        from duckmind.livebody import WORLD_EXTENSIONS

        return WORLD_EXTENSIONS


MotorFn = Callable[
    [int, C.MotorCommand, C.SensorFrame, RobotState], tuple[BodyRequest, list[dict[str, Any]]]
]
ROBOTD_HEAD_ALPHA_ENV = "DUCKMIND_ROBOTD_HEAD_ALPHA"  # (livebody writes it into robotd.toml)
ROBOTD_DEFAULT_HEAD_ALPHA = 0.2  # robotd-params Control::default()
ROLL_QUIET_MS = 2000  # a told roll's own bumps, touches, thumps and picture are no fright until this long after it is up
OWN_ROLL_EVENTS = frozenset({"blocked", "brushed", "stumbled"})
HEARD_THROUGH_ROLL = frozenset({"voice", "name", "hello"})  # (words still reach it: a roll makes no speech)


def robotd_head_alpha_env() -> float:
    """robotd's ``[control] head_alpha`` for this process's live bodies (``simweb
    --robotd-head-alpha``; prototype mode 1.0), else robotd's own default 0.2."""
    raw = os.environ.get(ROBOTD_HEAD_ALPHA_ENV, "").strip()
    try:
        value = float(raw) if raw else ROBOTD_DEFAULT_HEAD_ALPHA
    except ValueError:
        return ROBOTD_DEFAULT_HEAD_ALPHA
    return value if 0.0 < value <= 1.0 else ROBOTD_DEFAULT_HEAD_ALPHA


def _load_motor(
    notes: list[str], names: dict[str, str], brain: Any, seed: int
) -> tuple[MotorFn | None, LinkConfig | None]:
    """neuro:motor's :class:`~duckmind.neuro.motor.MotorSystem` for this duck (its profile from
    the same Duck Core draws as the brain's genome) and the body link's head envelopes from
    that profile; the stand-in when the lane is missing or fails."""
    try:
        module = importlib.import_module("duckmind.neuro.motor")
        motor = LiveMotor(module, getattr(brain, "genome", None), seed)
        names["motor"] = "real"
        return motor, motor.link_config()
    except Exception as error:
        names["motor"] = "stub"
        notes.append(
            msg("neuro.lane_note.load_motor", error_type=type(error).__name__, error_text=error)
        )
        return None, None


class LiveMotor:
    """neuro:motor on the live seam: ``MotorSystem.step(cmd, frame)`` → a :class:`BodyRequest`
    (its pre-compensated head, beak, legal twist and posture) and the call it voices this tick;
    ``last`` is its latest output for the telemetry."""

    def __init__(self, module: Any, genome: Any, seed: int) -> None:
        self.module = module
        if genome is None:  # the stub brain: the Duck Core duck of this seed (neuro:motor 23:14)
            from duckmind.neuro.genome import Genome

            genome = Genome.from_seed(int(seed))
        # ONE diversity mapping: the brain's genome (axes 0..1), never raw 2000–8000 traits;
        # robotd's head smoothing as this process launches it (fid:motor; attach() re-reads
        # what the body's robotd really runs)
        profile = module.profile_from_genome(genome, head_alpha=robotd_head_alpha_env())
        self.profile = profile
        self.system = module.MotorSystem(profile, max(0, int(getattr(genome, "seed", seed))))
        self.last: dict[str, Any] | None = None
        self._flying = (
            False  # the flight flag of the last request (carried through the motor's minimum burst)
        )

    def regrow(self, genome: Any) -> None:
        """(Growth G3.) The duck has grown: the profile of its new genome, with robotd's head
        alpha this motor already has. The motor's state (the program in hand, the head, the gait
        oscillator) goes on; the swap is made between two ticks, asleep."""
        profile = self.module.profile_from_genome(genome, head_alpha=self.profile.head_alpha)
        self.profile = profile
        self.system.profile = profile

    def set_head_alpha(self, alpha: float | None) -> None:
        """robotd's ``[control] head_alpha`` on the body this motor drives (fid:motor): the
        efference copy's pre-compensation must match it (it is the identity at 1.0)."""
        if alpha is None:
            return
        alpha = max(0.01, min(1.0, float(alpha)))
        if abs(float(getattr(self.profile, "head_alpha", alpha)) - alpha) < 1e-9:
            return
        from dataclasses import replace as _replace

        self.profile = _replace(self.profile, head_alpha=alpha)
        self.system.profile = self.profile

    def link_config(self) -> LinkConfig:
        """The body link's head bounds for this motor: its command range (standing/seated),
        inside the measured walking envelope while walking (bodylink.WALK_HEAD)."""
        return config_for_motor(tuple(self.module.COMMAND_MAX))

    def __call__(
        self, t_ms: int, cmd: C.MotorCommand, frame: C.SensorFrame, state: RobotState
    ) -> tuple[BodyRequest, list[dict[str, Any]]]:
        out = self.system.step(cmd, frame)
        self.last = out.to_json()
        # a flight in panic (flee / retreat). The brain's walk may blink for a tick (the look before it goes on, a
        # reflex) and the motor then holds the burst at the last twist for 0.8 s (note "burst≥0.8s"): the flight is
        # held with it, else the body link hands the panic skill back 17 ms after it started (live :8791, 2026-10-09)
        panic = bool(getattr(cmd, "panic", False)) and float(out.vx) > 0.0
        if not panic and self._flying and float(out.vx) > 0.0 and "burst≥0.8s" in out.notes:
            panic = True
        self._flying = panic
        request = BodyRequest(
            vx=float(out.vx),
            vyaw=float(out.vyaw),
            head=(float(out.head[0]), float(out.head[1]), float(out.head[2]), float(out.head[3])),
            mouth=float(out.mouth),
            posture=str(out.posture),
            approach=bool(cmd.approach),
            # a flight in panic (flee / retreat): the body link runs it as the table's panic skill when it has
            # one (2026-10-09); the twist above is what v5 walks otherwise
            panic=panic,
            # turning toward something it looks at: the body link pivots with the quick-turn skill when the table has one
            pivot=float(getattr(cmd, "pivot", 0.0)) if float(out.vx) > 0.0 else 0.0,
            # the docking choreography's own twist (dock:build 2026-09-28: going in backwards,
            # walking out along the dock's axis; contract, MotorCommand.docking)
            docking=bool(cmd.docking),
            back_arc=bool(getattr(cmd, "back_arc", False)),
            toward=_toward(cmd),
        )
        vocals: list[dict[str, Any]] = []
        if out.sound:
            vocals.append(
                {
                    "tag": str(out.sound),
                    "level": float(out.sound_level),
                    "behaviour": cmd.behaviour,
                    "program": out.program,
                }
            )
        return request, vocals


def _toward(cmd: C.MotorCommand) -> tuple[int, int] | None:
    """The thing a curiosity walk goes up to, as the body link's A1 wants it: (bearing mrad, range mm)."""
    target = cmd.target
    if (
        cmd.behaviour != "investigate"
        or target is None
        or target.range_m is None
        or target.kind not in ("thing", "place")
    ):
        return None
    return int(round(1000.0 * target.bearing)), int(round(1000.0 * max(0.0, target.range_m)))


# ── what the senses get ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class Doors:
    """One tick's sensor doors for neuro:senses. ``state`` … ``pg`` are what the robot's own
    doors give (robotd, duck-body's proprio, tofd, the PG pin). ``world``, ``truth`` and
    ``stimuli`` are the simulator's truth — ONLY for the world-side sensor models (sound, touch,
    the camera's motion/looming, power) that stand in for sensors the prototype lacks; the brain
    never receives them."""

    t_ms: int
    state: RobotState
    link: LinkOut | None
    events: tuple[str, ...]
    tof: Mapping[str, Any] | None = None
    depth: P.DepthView | None = None
    pg: bool | None = None
    day_minute: float = 600.0
    world: Mapping[str, Any] | None = None  # x_world (truth: world-side models only)
    truth: Mapping[str, Any] | None = None  # the physics frame (truth: world-side models only)
    world_age_ms: int | None = None
    # The page's scenario stimuli that have no body in the physics (sounds, a wave), as world
    # events (:meth:`NeuroDriver.inject`), newest last, the last STIMULI_KEEP_MS: each
    # {"seq", "kind", "t_ms", "ms" (how long), "pos_mm" [x, y, z] (world), "sound" + "db" (at
    # 1 m) | "visual", "who"} — world-side models only.
    stimuli: tuple[Mapping[str, Any], ...] = ()


def _events_of(link: LinkOut | None) -> tuple[str, ...]:
    """duck-body's edges and righting results as the contract's BODY_EVENTS."""
    if link is None:
        return ()
    out: list[str] = []
    for event in link.events:
        kind = str(event.get("kind", ""))
        if kind.startswith("proprio."):
            edge = kind[len("proprio.") :]
            if edge in ("blocked", "brushed", "stumbled", "lifted", "set_down"):
                out.append(edge)
            elif edge == "fell":
                out.append("fell")
        elif kind == "body.lay_down":
            out.append("lay_down")
        elif kind == "body.righting":
            if event.get("ok"):
                out.append("righted_help" if event.get("via") == "help" else "righted")
            else:
                out.append("righting_failed")
    return tuple(dict.fromkeys(out))


def _day_minute(moment: datetime | None) -> float:
    if moment is None:
        return 600.0
    return moment.hour * 60.0 + moment.minute + moment.second / 60.0


def _battery(world: Mapping[str, Any] | None) -> tuple[float | None, bool | None]:
    """The world's pack (MODEL: robotd is not fed yet, owner E1): state of charge 0..1 and
    whether it charges."""
    battery = (world or {}).get("battery")
    if not isinstance(battery, Mapping) or "percent" not in battery:
        return None, None
    try:
        return max(0.0, min(1.0, float(battery["percent"]) / 100.0)), bool(battery.get("charging"))
    except (TypeError, ValueError):
        return None, None


# ── telemetry ────────────────────────────────────────────────────────────────


def _rounded(value: Any, digits: int = 3) -> Any:
    if isinstance(value, float):
        return round(value, digits) if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [_rounded(v, digits) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _rounded(v, digits) for k, v in value.items()}
    return value


class Telemetry:
    """What the driver shows: samples at 10 Hz, events, the latest brain telemetry and frame.
    Written by the tick thread, read by the web threads (one lock)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.seq = 0
        self.samples: deque[dict[str, Any]] = deque(maxlen=SAMPLES_KEEP)
        self.vocals: deque[dict[str, Any]] = deque(maxlen=EVENTS_KEEP)
        self.events: deque[dict[str, Any]] = deque(maxlen=EVENTS_KEEP)
        self.brain: dict[str, Any] | None = None
        self.frame: dict[str, Any] | None = None
        self.command: dict[str, Any] | None = None
        self.counts: dict[str, int] = {}

    def _next(self) -> int:
        self.seq += 1
        return self.seq

    def sample(self, item: dict[str, Any]) -> None:
        with self.lock:
            self.samples.append({"seq": self._next(), **item})

    def event(self, item: dict[str, Any], vocal: bool = False) -> None:
        with self.lock:
            (self.vocals if vocal else self.events).append({"seq": self._next(), **item})
            kind = str(item.get("tag") if vocal else item.get("kind"))
            key = f"vocal.{kind}" if vocal else kind
            self.counts[key] = self.counts.get(key, 0) + 1

    def latest(
        self,
        brain: dict[str, Any] | None,
        frame: dict[str, Any] | None,
        command: dict[str, Any] | None,
    ) -> None:
        with self.lock:
            self.brain, self.frame, self.command = brain, frame, command

    def since(self, seq: int, limit: int = 600) -> dict[str, Any]:
        with self.lock:
            samples = [s for s in self.samples if s["seq"] > seq][-limit:]
            return {
                "seq": self.seq,
                "samples": samples,
                "vocals": [v for v in self.vocals if v["seq"] > seq],
                "events": [e for e in self.events if e["seq"] > seq],
                "brain": self.brain,
                "frame": self.frame,
                "command": self.command,
                "counts": dict(self.counts),
            }


# ── the driver ───────────────────────────────────────────────────────────────


class _Stats:
    def __init__(self) -> None:
        self.ticks = 0
        self.skipped = 0
        self.errors = 0
        self.refused = 0
        self.step_ms: deque[float] = deque(maxlen=500)
        self.brain_ms: deque[float] = deque(maxlen=500)
        self.tick_wall: deque[float] = deque(maxlen=250)
        self.attached_at: float | None = None
        self.attaches = 0
        self.last_error: str | None = None

    def view(self) -> dict[str, Any]:
        def pct(values: deque[float], q: float) -> float | None:
            if not values:
                return None
            ordered = sorted(values)
            return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 3)

        hz = None
        if len(self.tick_wall) >= 2:
            span = self.tick_wall[-1] - self.tick_wall[0]
            hz = round((len(self.tick_wall) - 1) / span, 2) if span > 0 else None
        return {
            "ticks": self.ticks,
            "skipped": self.skipped,
            "errors": self.errors,
            "refused": self.refused,
            "hz": hz,
            "step_ms_p50": pct(self.step_ms, 0.5),
            "step_ms_p95": pct(self.step_ms, 0.95),
            "step_ms_max": round(max(self.step_ms), 3) if self.step_ms else None,
            "brain_ms_p95": pct(self.brain_ms, 0.95),
            "attaches": self.attaches,
            "attached_s": None
            if self.attached_at is None
            else round(time.time() - self.attached_at, 1),
            "last_error": self.last_error,
        }


MOTION_ANSWER_S = 2.0  # the page's press waits this long for the body link's answer
MOTION_BEFORE_MS = 400  # a run's record starts this long before the request
ROOM_BEHIND_TRAIL_M = 0.30  # walked this far forward lately: the way back is known free …
ROOM_BEHIND_WITHIN_S = 6.0  # … if it was within this long
ROOM_BEHIND_CLEAR_M = 0.30  # … and the wall memory leaves this much behind it
# (startle-hop, 2026-10-10) a startle is answered by two small steps back unless the duck KNOWS it cannot: the wall / bump / touch
# memory (walls, bumps, a touch on its back) holds something within HOP_BLOCKED_M behind it, the scare came from behind it (heard, seen or felt, in the last
# HOP_SCARE_FRESH_S, bearing more than HOP_BEHIND_RAD off its nose), or the battery is low. What it does not know does not stop it.
HOP_BLOCKED_M = 0.20
HOP_BEHIND_RAD = math.pi / 2
HOP_SCARE_FRESH_S = 1.5
HOP_TOUCH_BEHIND_S = 10.0  # something touched its back this recently: it is still there (the touch memory of the walk's veer)
HOP_STILL_MPS = 0.05  # its trunk is slow …
HOP_STILL_TILT_DEG = (
    8.0  # … and upright: the picker asks for the back-steps (the guard checks again)
)


@dataclass
class _PlayRequest:
    """The page's press of 「動作試玩」 waiting for the body link's answer."""

    motion: Any  # motions.Motion
    wall: float
    done: threading.Event = field(default_factory=threading.Event)
    started: bool = False
    now_s: float = 0.0
    answer: dict[str, Any] | None = None


@dataclass
class MotionRun:
    """The record of a motion the page started: the witness ticks from just before the request until
    the body is a plain posture again (:mod:`duckmind.skilltrain.motion_result`)."""

    motion: Any
    t0_ms: int
    seq: int
    ticks: list[Any] = field(default_factory=list)
    plain_since_ms: int | None = None
    seen: bool = False


class NeuroDriver:
    """Drives one live body with the neural duck (see the module doc).

    :meth:`attach` when the bridge's body is ready (the bridge calls it), :meth:`detach` before
    that body is stopped, :meth:`stop` at shutdown. ``clock`` gives the duck's local time (the
    page's clock) for the circadian drive."""

    def __init__(self, spec: NeuroSpec, *, lanes: Lanes | None = None) -> None:
        self.spec = spec
        self.lanes = lanes or load_lanes(spec)
        self.telemetry_store = Telemetry()
        self.stats = _Stats()
        # round 5: watches for a bump_stop loop and writes the last minute to a file (it never touches the body)
        self.loopwatch = LoopWatch(os.environ.get("DUCKMIND_LOOP_DIR") or None)
        # the duck's time of day: its own day clock (a daytime start, the page's choice); a
        # ``clock`` set from outside (a test, Duck Core's clock) is used instead when given
        self.clock: Callable[[], datetime] | None = None
        self.day_clock: DuckClock = DayClock()
        self.life: Any = None  # the life this driver lives, when it has one (life_host.LifeHost)
        self.start_at = "mat"  # where the bridge puts the body before this driver takes it
        self._lock = threading.Lock()
        self._tick_lock = threading.RLock()
        self._body: Any = None
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._deferred_lane_close = False
        self._client: RobotdClient | None = None
        self._link: BodyLink | None = None
        self._t0_ns: int | None = None
        self._t_offset_ms = 0
        self._last_t_ms = 0
        self._pg: bool | None = None
        self._world: Mapping[str, Any] | None = None
        self._truth: Mapping[str, Any] | None = None
        self._world_at = 0.0
        self._day_minute = 600.0
        self._record: Any = None
        self._stub_motor = StubMotor()
        self._last_cmd: C.MotorCommand | None = None
        # the calls the duck voiced on the latest tick (the microphone stream hears them as echo)
        self._tick_vocals: list[dict[str, Any]] = []
        self._last_out: LinkOut | None = None
        from duckmind.simweb.neuro_runtime import OneShotBridge

        self._one_shots = OneShotBridge()
        # World events the page injected (sounds, a wave): world-side models only (Doors).
        self._stimuli: deque[dict[str, Any]] = deque(maxlen=64)
        self._stimulus_seq = 0
        self._leaves: list[tuple[float, str]] = []  # (monotonic, who): walk-bys that leave
        # The ToF door (a duck with the module): tofd's newest tof.frame, duck-body's depth
        # summary of it, and the tracker that makes it.
        self._tof_frame: dict[str, Any] | None = None
        self._tof_seen: Any = None
        self._depth: P.DepthView | None = None
        self._depth_at_ns = 0
        self._depth_tracker: Any = None
        self._failing = 0  # ticks in a row a lane failed
        self._mouth_peak = 0.0  # the widest beak opening since the last sample (a call is short)
        self._roll_quiet_until_ms = -1  # (its own roll's sensations are not a fright until then)
        self.helps = 0  # times the page's 「扶牠起來」 had the world's hand help it up
        self.error: str | None = None
        self.state = "waiting"  # waiting | attaching | running | detached | stopped | error
        # fid:page: what the body measurably did after each reflex, and is doing now (the page's
        # honesty; an observer beside the driver, never an input of the brain).
        self.witness = Witness()
        self._witness_lock = threading.Lock()
        # (fid:integrate) set by :meth:`inject`: the aux loop reads the world again at once,
        # then quickly for a moment (until this monotonic time)
        self._aux_wake = threading.Event()
        self._aux_fast_until = 0.0
        # Stage-1 / B / hop trained skills (robotd's table): the act → a table entry; the page's
        # 「動作試玩」 button (:meth:`play_motion`) and the run it started (:class:`MotionRun`)
        self._skill_picker = SG.SkillPicker()
        # the 14 emotions of microduck_emotions: the page's 「情緒」 presses and the brain's emotion layer
        from duckmind.simweb.neuro_emotions import EmotionHost

        self.emotions = EmotionHost(self)
        self.remote = RemoteHand()  # remote mode (neuro/remote.py): the family's hand on the legs
        self.live_voice: Any = (
            None  # the live conversation's host (cloud_on(), calls_seen()), when there is one
        )
        self._play_req: _PlayRequest | None = None
        self._motion_run: MotionRun | None = None
        self._motion_last: dict[str, Any] | None = None
        self._motion_seq = 0
        self._trail: deque[tuple[float, float, float]] = deque(maxlen=400)  # (t_s, x, y) odometry
        self._speed_hist: deque[tuple[float, float, float]] = deque(
            maxlen=64
        )  # the last ~0.4 s of it (startle-hop)
        self._recent: deque[Any] = deque(maxlen=40)  # the witness ticks of the last ~0.8 s

    # ── the page's buttons: a stimulus into the world, another duck ──

    def inject(self, name: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """A scenario stimulus (``contract.STIMULI`` key) into the WORLD, never into the brain.
        With the neural duck's world side loaded (prototype mode) the world makes it itself
        (``x_neuro_stim``, :func:`duckmind.world.neuro_senses.make_injector`: a sound at a
        person's mouth, the hand really moving, a ball rolled, the lamps…). Without it (a
        world without the extension): the visible hand, a person, a ball or the lamps through
        the household's ops, and a sound or a wave as a world event in :attr:`Doors.stimuli`.
        Returns what the world did (plain words); ValueError in plain words when it cannot."""
        try:
            return self._inject(name, params)
        finally:
            # (fid:integrate 2026-09-28) the world's truth is read again right away, not at the
            # aux loop's next 5-Hz poll: fid:audit measured the poll adding up to 0.2 s to every
            # sound's reflex. What reaches the duck still goes through its sensor models.
            self._aux_wake.set()

    def _inject(self, name: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        params = dict(params or {})
        body = self._body
        call = getattr(body, "world_call", None) if body is not None else None
        ops = (getattr(body, "host_info", None) or {}).get("ops") or []
        if name in TRAINS:
            return self._train(name, params)
        if name == "swap_battery":
            # a person puts a charged pack in (the world's battery: ~30–60 min a pack); the duck
            # feels it only through its power sense
            if not callable(call) or "x_battery" not in ops:
                raise ValueError(msg("neuro.world_refusal.6ada73"))
            world = _world(call, "x_battery", {"do": "swap"})
            self.telemetry_store.event(
                {
                    "t_ms": self._last_t_ms,
                    "wall_ms": int(time.time() * 1000),
                    "kind": "world.stimulus",
                    "stimulus": name,
                    "zh": msg("neuro.world.swap_battery"),
                }
            )
            return {
                "ok": True,
                "kind": name,
                "note_zh": msg("neuro.world.swap_battery"),
                "world": world,
            }
        if name == "carry_to_mat":
            # someone picks it up and puts it down on the floor mat (the world's hand: it feels
            # itself lifted and set down) — for a duck that got itself into a tight spot
            if not callable(call) or "x_hand" not in ops:
                raise ValueError(msg("neuro.world_refusal.03c917"))
            world = _world(call, "x_hand", {"do": "place", "to": "mat"})
            self.telemetry_store.event(
                {
                    "t_ms": self._last_t_ms,
                    "wall_ms": int(time.time() * 1000),
                    "kind": "world.stimulus",
                    "stimulus": name,
                    "zh": msg("neuro.world.carry_to_mat"),
                }
            )
            return {
                "ok": True,
                "kind": name,
                "note_zh": msg("neuro.world.carry_to_mat"),
                "world": world,
            }
        if name == "bang":
            params.setdefault("db1m", PAGE_BANG_DB1M)  # (the page's bang is a very loud one)
        elif name == "push":
            params.setdefault("strength", PAGE_PUSH_STRENGTH)
        if callable(call) and "x_neuro_stim" in ops:
            # neuro:senses' world side makes it a real world event (duckmind.world.neuro_world)
            from duckmind.world.neuro_senses import make_injector

            try:
                made = dict(make_injector(call)(name, params) or {})
            except ValueError:
                raise
            except Exception as error:  # the world refused (its hand is busy…): its own words
                text = str(error)
                if "'error':" in text:
                    text = text.split("'error':", 1)[1].strip(" }'\"")
                raise ValueError(
                    text or msg("neuro.world_refusal.411c43", error_type=type(error).__name__)
                ) from error
            self.telemetry_store.event(
                {
                    "t_ms": self._last_t_ms,
                    "wall_ms": int(time.time() * 1000),
                    "kind": "world.stimulus",
                    "stimulus": name,
                    "zh": carried_words(made, "note_zh") or carried_words(made, "zh") or name,
                }
            )
            return made
        answer: dict[str, Any] = {"ok": True, "kind": name}
        if name in SOUND_STIMULI or name in VISUAL_STIMULI:
            event = self._world_event(name, params)
            answer.update(zh=event["zh"], event=event)
        elif name in HAND_STIMULI:
            if not callable(call):
                raise ValueError(msg("neuro.world_refusal.5bec97"))
            request = dict(HAND_STIMULI[name])
            who = params.get("who")
            if who:
                request["who"] = str(who)
            answer.update(world=_world(call, "x_hand", request), zh=HAND_ZH[name])
        elif name in ("roll_ball", "toy_drop", "novel_object"):
            if not callable(call):
                raise ValueError(msg("neuro.world_refusal.556de6"))
            request = {"toward": "duck"}
            if params.get("thing"):
                request["id"] = str(params["thing"])
            answer.update(
                world=_world(call, "x_roll_ball", request), zh=msg("neuro.world.fallback.5700")
            )
            if name == "toy_drop":
                self._world_event("toy_drop", {"sound": "bang", "db": 58.0, "ms": 200})
        elif name in ("walk_by", "sit_near"):
            if not callable(call):
                raise ValueError(msg("neuro.world_refusal.8a8ea9"))
            who = str(params.get("who") or "owner")
            answer.update(
                world=_world(call, "x_people", {"do": "come", "who": who, "to": "duck"}),
                zh=msg("neuro.world.fallback.ff32")
                if name == "sit_near"
                else msg("neuro.world.fallback.8e5b"),
            )
            if name == "walk_by":
                self._leaves.append((time.monotonic() + WALK_BY_S, who))
                self._world_event("footsteps", {"sound": "footsteps", "db": 50.0, "ms": 4000})
        elif name in ("lights_off", "lights_on"):
            if not callable(call):
                raise ValueError(msg("neuro.world_refusal.427e79"))
            on = name == "lights_on"
            answer.update(
                world=_world(call, "x_lights", {"lamps_on": on}),
                zh=msg("neuro.world.fallback.23aa") if on else msg("neuro.world.fallback.18e0"),
                lights_on=on,
            )
        else:
            raise ValueError(msg("neuro.world_refusal.d43836", given_id=name))
        self.telemetry_store.event(
            {
                "t_ms": self._last_t_ms,
                "wall_ms": int(time.time() * 1000),
                "kind": "world.stimulus",
                "stimulus": name,
                "zh": answer.get("zh", name),
            }
        )
        return answer

    def _world_event(self, name: str, params: Mapping[str, Any]) -> dict[str, Any]:
        """A sound or a wave at a place in the home (world side): the nearest person's place,
        else about a metre from the duck (a sound) / in front of it (a wave)."""
        spec = {**SOUND_STIMULI.get(name, {}), **VISUAL_STIMULI.get(name, {}), **dict(params)}
        pose = _truth_pose(self._truth)
        person = _nearest_person(self._world, pose)
        self._stimulus_seq += 1
        seq = self._stimulus_seq
        pos: list[int] | None = None
        who = None
        if spec.get("visual"):
            if pose is not None:
                ahead = float(spec.get("ahead_m", 0.7))
                pos = [
                    round((pose[0] + ahead * math.cos(pose[2])) * 1000),
                    round((pose[1] + ahead * math.sin(pose[2])) * 1000),
                    300,
                ]
        elif person is not None:
            pos, who = [int(v) for v in person[0][:2]] + [1000], person[1]
        elif pose is not None:
            angle = pose[2] + math.pi * (0.25 + 1.5 * ((seq * 0.618034) % 1.0))  # not ahead
            pos = [
                round((pose[0] + 1.2 * math.cos(angle)) * 1000),
                round((pose[1] + 1.2 * math.sin(angle)) * 1000),
                900,
            ]
        event = {
            "seq": seq,
            "kind": name,
            "t_ms": self._last_t_ms,
            "ms": int(spec.get("ms", 500)),
            "pos_mm": pos,
            "who": who,
            "zh": spec.get("zh") or name,
            "source": "model",
        }
        if spec.get("sound"):
            event.update(sound=str(spec["sound"]), db=float(spec.get("db", 60.0)))
        if spec.get("visual"):
            event["visual"] = str(spec["visual"])
        self._stimuli.append(event)
        return event

    def _train(self, name: str, params: Mapping[str, Any]) -> dict[str, Any]:
        """A rhythm: the same world sound ``count`` times at ``hz`` (a thread paces them; each
        is an ordinary world stimulus — the brain hears a train only through its microphone)."""
        spec = TRAINS[name]
        first = self.inject(str(spec["stim"]), params)
        count, gap = int(spec["count"]), 1.0 / float(spec["hz"])

        def rest() -> None:
            for _ in range(count - 1):
                time.sleep(gap)
                if self._body is None:
                    return
                try:
                    self.inject(str(spec["stim"]), params)
                except Exception:
                    return

        threading.Thread(target=rest, daemon=True).start()
        return {**first, "ok": True, "kind": name, "note_zh": spec["zh"]}

    def stimuli_supported(self) -> set[str]:
        """The page stimuli this driver can make in the world now."""
        body = self._body
        ops = (getattr(body, "host_info", None) or {}).get("ops") or []
        battery = {"swap_battery"} if "x_battery" in ops else set()
        if "x_hand" in ops:
            battery.add("carry_to_mat")
        if callable(getattr(body, "world_call", None)) and "x_neuro_stim" in ops:
            from duckmind.world.neuro_senses import SUPPORTED_STIMULI

            return set(SUPPORTED_STIMULI) | set(TRAINS) | battery
        return (
            set(SOUND_STIMULI)
            | set(VISUAL_STIMULI)
            | set(HAND_STIMULI)
            | {"roll_ball", "toy_drop", "novel_object", "walk_by", "sit_near"}
            | {"lights_off", "lights_on"}
            | set(TRAINS)
            | battery
        )

    def set_day(self, minute: float) -> dict[str, Any]:
        """The page's time-of-day choice: the duck's clock jumps there and runs on."""
        self.day_clock.set(minute)
        self._day_minute = self.day_clock.minute()
        view = self.day_clock.view()
        self.telemetry_store.event(
            {
                "t_ms": self._last_t_ms,
                "wall_ms": int(time.time() * 1000),
                "kind": "world.clock",
                "zh": msg("neuro.world.clock_set.event", day=view["zh"]),
            }
        )
        return view

    def help_up(self) -> dict[str, Any]:
        """「扶牠起來」: the world's visible hand sets the fallen body upright (a world act; the
        duck feels it through its own body: righted with help)."""
        body = self._body
        call = getattr(body, "world_call", None) if body is not None else None
        if not callable(call):
            raise ValueError(msg("neuro.world_refusal.5eeb15"))
        answer = _world(call, "x_hand", {"do": "upright"})
        self.helps += 1
        self.telemetry_store.event(
            {
                "t_ms": self._last_t_ms,
                "wall_ms": int(time.time() * 1000),
                "kind": "world.stimulus",
                "stimulus": "help_up",
                "zh": msg("neuro.world.help_up.row"),
            }
        )
        return {"ok": True, "note_zh": msg("neuro.world.help_up.note"), "hand": answer}

    def _looks(self) -> dict[str, Any] | None:
        """The camera's look budget as the senses lane spends it (looks a second, refused)."""
        senses = self.lanes.senses
        rate = getattr(senses, "looks_per_s", None)
        if not callable(rate):
            return None
        return {
            "per_s": rate(),
            "looks": int(getattr(senses, "looks", 0)),
            "refused": int(getattr(senses, "looks_refused", 0)),
        }

    def household_note(self, name: str, words: Any) -> None:
        """The page's row for something the household did in the world on its own (a family
        member swapped the pack): said like a stimulus's world row, and nothing more — the brain
        feels it only through its senses (the hand, its power sense)."""
        self.telemetry_store.event(
            {
                "t_ms": self._last_t_ms,
                "wall_ms": int(time.time() * 1000),
                "kind": "world.stimulus",
                "stimulus": name,
                "zh": words,
            }
        )

    def recent_stimuli(self, t_ms: int) -> tuple[dict[str, Any], ...]:
        return tuple(e for e in self._stimuli if t_ms - int(e["t_ms"]) <= STIMULI_KEEP_MS)

    def switch(self, duck: str) -> dict[str, Any]:
        """Another neural duck on the same body (``duck``: a seed, ``seed:gd`` or a preset): a
        fresh brain from Duck Core's diversity design; the telemetry starts over."""
        seed, gd, _ = NeuroSpec(duck=duck, seed=self.spec.seed).parsed()
        spec = NeuroSpec(duck=duck, seed=self.spec.seed, gd=gd, record=self.spec.record)
        life = self.life
        # (the life duck, docs/design/GROWTH-G1: picked again by its seed and growth day it is
        # the life duck, with its genome from Duck Core's state and its mind brought back; the
        # life duck's own mind is saved the moment the page leaves it)
        to_life = life is not None and life.is_life_duck(seed, gd)
        if to_life:
            spec = life.spec(spec)
        if life is not None and self.spec.life:
            life.before_switch()
        lanes = load_lanes(spec)
        _sync_head_alpha(lanes, self._body)  # (fid:motor: the same body's robotd)
        # Wait for the current tick to finish, then swap the complete lane set and its telemetry
        # together. This preserves switch()'s synchronous API and prevents an old tick from
        # writing into the next duck's fresh telemetry or closing an in-use VoiceEngine.
        with self._tick_lock:
            retired = self.lanes
            self.spec, self.lanes = spec, lanes
            self.telemetry_store = Telemetry()
            self._stub_motor = StubMotor()
            from duckmind.simweb.neuro_runtime import OneShotBridge

            self._one_shots = OneShotBridge()
            if self._link is not None and lanes.link is not None:
                self._link.cfg = lanes.link
            _close_lane_sound(retired)
        if to_life:
            life.after_switch()
        return {"lanes": dict(lanes.names), "notes": list(lanes.notes), "genome": lanes.genome}

    def _skill_request(
        self,
        link: BodyLink,
        cmd: C.MotorCommand,
        state: RobotState,
        request: BodyRequest,
        brain: C.BrainTelemetry | None = None,
        t_ms: int | None = None,
    ) -> BodyRequest:
        """The trained skills (the Stage-1 table, the B skills, the hop-over; robotd's one-shot table):
        once robotd lists them, the body link gets a skill guard (the installed table's rules: the
        postures and walking speeds that passed its gates, the settle times, the held crouches, the
        wear caps). Every tick the picker names the entry the act runs now — a startle at a level near
        the reflex's own strength, jittered per execution; a startle the brain follows with a freeze
        held as ONE crouch while the fear lasts, then ``startle_release`` (one rise); a greeting bow, a
        peekaboo lie-low, an on-the-spot turn out of a corner, a back-step when standing with the way
        behind it known free — and the link's guard decides. A press of the page's 「動作試玩」
        (:meth:`play_motion`) goes first."""
        client = self._client
        if link.skills is None and client is not None:
            names = set(client.skills) & set(SG.DEFAULT_RULES)
            if names:
                # the installed table's rules (the postures that passed its gates, the hourly caps
                # from its strain numbers); the provisional defaults otherwise
                link.skills = SW.skill_guard(names)
                seed = int(getattr(self.spec, "seed", 7) or 7)
                self._skill_picker = SG.SkillPicker(
                    seed=seed * 7919 + 17, available=link.skills.names
                )
        if link.skills is None:
            self._answer_play(refused="missing")
            return request
        twist = math.hypot(state.applied[0], state.applied[2])
        posture = SG.posture_of(SG.BodyView(state.policy, twist), link.skills.names)
        now_s = None if t_ms is None else t_ms / 1000.0
        play = self._play_req
        if play is not None and not play.started and now_s is not None:
            entry = play.motion.entry
            if entry not in link.skills.names:
                self._answer_play(refused="missing")
            else:
                self._skill_picker.play(
                    entry,
                    now_s,
                    hold_s=play.motion.hold_s,
                    end=play.motion.end,
                    then=play.motion.then or None,
                )
                play.started, play.now_s = True, now_s
        strength = None
        event = None
        corner = 0
        if brain is not None and brain.reflexes:
            if cmd.reflex:
                mine = [e for e in brain.reflexes if e.reflex == cmd.reflex]
                if mine:
                    strength = float(mine[-1].strength)  # the reflex's own size, not the bout's
            newest = brain.reflexes[-1]
            # a startle starts from its event (the tick the brain perceives it), not 45–80 ms
            # later when the reflex takes the body: the skill's own motor latency stands in for it
            event = (int(newest.t_ms), str(newest.reflex), float(newest.strength))
        if brain is not None:
            corner = int(round(float(brain.locomotion.get("corner_turn", 0.0) or 0.0)))
        skill = self._skill_picker.pick(
            reflex=cmd.reflex,
            behaviour=cmd.behaviour,
            bout=cmd.bout,
            intensity=cmd.intensity,
            posture=posture,
            now_s=now_s,
            strength=strength,
            label=state.policy,
            event=event,
            room_behind=self._room_behind(state, brain, now_s, posture),
            corner_turn=corner,
            hop_ok=self._hop_ok(link, brain, now_s),
            still=self._trunk_still(state, now_s),
            hop_decision=self._hop_decision(link),
        )
        skill = SW.pick_loaded(skill, link.skills.names)
        picker = self._skill_picker
        chain = bool(skill and picker.chain)
        manual = bool(
            skill
            and (
                (picker.script is not None and picker.script.who == "page")
                or (chain and picker.follow is not None and picker.follow.manual)
            )
        )
        if picker.hold_still:  # a person's press waits for the back-steps: the walk stops first, as a scared duck's does
            request = replace(request, vx=0.0, vyaw=0.0, panic=False, pivot=0.0)
        if skill:
            return replace(request, skill=skill, skill_manual=manual, skill_chain=chain)
        return request

    def _hop_ok(
        self, link: BodyLink, brain: C.BrainTelemetry | None, now_s: float | None
    ) -> bool | None:
        """May a startle be answered by the back-steps: not if its own memory says the way behind is blocked, the scare came
        from behind, or the battery is low. None: this tick has no clock or brain to ask."""
        if now_s is None or brain is None:
            return None
        loc = brain.locomotion
        behind = loc.get("room_behind_m")
        if behind is not None and behind < HOP_BLOCKED_M:
            return False
        bearing, age = loc.get("threat_bearing"), loc.get("threat_age_s")
        if (
            bearing is not None
            and age is not None
            and age <= HOP_SCARE_FRESH_S
            and abs(bearing) > HOP_BEHIND_RAD
        ):
            return False
        touched = loc.get("touch_behind_age_s")
        if touched is not None and touched <= HOP_TOUCH_BEHIND_S:
            return False
        battery, _ = _battery(self._world)
        return not (battery is not None and battery * 100.0 < link.cfg.battery_min_pct)

    def _trunk_still(self, state: RobotState, now_s: float | None) -> bool:
        """The trunk is slow (its odometry over the last 0.15-0.4 s) and upright."""
        if now_s is None:
            return False
        hist = self._speed_hist
        x, y = float(state.odom_pos[0]), float(state.odom_pos[1])
        if hist and now_s < hist[-1][0]:
            hist.clear()
        if not hist or now_s - hist[-1][0] >= 0.02:
            hist.append((now_s, x, y))
        while hist and now_s - hist[0][0] > 0.4:
            hist.popleft()
        old = [h for h in hist if now_s - h[0] >= 0.15]
        if not old:
            return False
        t0, x0, y0 = old[-1]
        tilt = state.tilt_rad
        upright = tilt is None or math.degrees(tilt) < HOP_STILL_TILT_DEG
        return upright and math.hypot(x - x0, y - y0) / (now_s - t0) < HOP_STILL_MPS

    @staticmethod
    def _hop_decision(link: BodyLink) -> tuple[bool, str, float] | None:
        """The body link's last answer to a hop-back ask: (ok, reason, clock)."""
        last = link.last_skill_decision
        if last is None or last[0] != SG.HOP_ENTRY:
            return None
        return bool(last[1].ok), str(last[1].reason), float(last[2])

    def _room_behind(
        self,
        state: RobotState,
        brain: C.BrainTelemetry | None,
        now_s: float | None,
        posture: str,
    ) -> bool:
        """The way behind it is known free, from the duck's own senses only: it stands, it walked at least
        ROOM_BEHIND_TRAIL_M forward within the last ROOM_BEHIND_WITHIN_S (odometry: the way it came is
        the way back) and what the wall memory holds behind it leaves ROOM_BEHIND_CLEAR_M. A duck that
        cannot know says no: the hop-back backs up blind."""
        if now_s is None or brain is None or posture != "stand":
            return False
        x, y = float(state.odom_pos[0]), float(state.odom_pos[1])
        trail = self._trail
        if not trail or now_s - trail[-1][0] >= 0.1:
            trail.append((now_s, x, y))
        past = [p for p in trail if now_s - p[0] <= ROOM_BEHIND_WITHIN_S]
        if len(past) < 2:
            return False
        walked = max(math.hypot(x - p[1], y - p[2]) for p in past)
        behind = brain.locomotion.get("room_behind_m")
        return (
            walked >= ROOM_BEHIND_TRAIL_M and behind is not None and behind >= ROOM_BEHIND_CLEAR_M
        )

    # ── the page's 「動作試玩」 ──

    def play_motion(self, motion_id: str) -> dict[str, Any]:
        """Do one trained motion now (the page's button). Waits for the body link's answer: started,
        or why not (a posture it was not gated for, something already running, too fast a walk…) in
        plain words. The wear caps do not apply to a press."""
        from duckmind.neuro import motions as MO

        motion = MO.BY_ID.get(str(motion_id))
        if motion is None:
            raise ValueError(msg("neuro.motion.unknown"))
        if self.state != "running" or self._link is None:
            raise ValueError(msg("neuro.motion.not_running"))
        request = _PlayRequest(motion, time.monotonic())
        with self._lock:
            if self._play_req is not None and not self._play_req.done.is_set():
                raise ValueError(msg("neuro.motion.pressed"))
            self._play_req = request
        if not request.done.wait(MOTION_ANSWER_S):
            self._skill_picker.cancel()
            with self._lock:
                self._play_req = None
            return {"ok": False, "id": motion.id, "refused": True, "reason": "late",
                    "reason_zh": msg("neuro.motion.late")}  # fmt: skip
        answer = dict(request.answer or {})
        with self._lock:
            self._play_req = None
        return {"id": motion.id, "entry": motion.entry, **answer}

    # ── what the page's command strip says (docs/design/neural-duck/COMMANDS-V1-2026-10-23.md) ──

    def command(self, command: Any) -> dict[str, Any]:
        """A person's command to the duck: handed to the brain as what it heard (never a motor command)
        and answered once the brain has decided: ``state`` ``heard`` (it will do it, the rest is in the
        telemetry), ``done`` (already as asked) or ``refused`` with a reason code. A brain with no
        command intake (the stand-in) refuses with ``not_running``."""
        from duckmind.neuro import command_brain as CB

        brain = self.lanes.brain
        intake = getattr(brain, "commands", None)
        if self.state != "running" or intake is None:
            return {"state": "refused", "state_zh": CB.STATE_MSG["refused"], "reason": "not_running",
                    "reason_zh": CB.REASON_MSG["not_running"], "queued": False}  # fmt: skip
        ticket = intake.hear(command)
        if not ticket.event.wait(CB.DECISION_WAIT_S):
            with intake._lock:
                if intake._pending is ticket:
                    intake._pending = None
                    return {"state": "refused", "state_zh": CB.STATE_MSG["refused"], "reason": "busy",
                            "reason_zh": CB.REASON_MSG["busy"], "queued": False}  # fmt: skip
            ticket.event.wait(CB.DECISION_WAIT_S)
        return dict(ticket.answer) or {"state": "refused", "state_zh": CB.STATE_MSG["refused"],
                                       "reason": "busy", "reason_zh": CB.REASON_MSG["busy"],
                                       "queued": False}  # fmt: skip

    def feedback(self, kind: str, trick: str | None = None) -> dict[str, Any]:
        """The owner's 「乖」, treat or 「不對」 for a trick (``trick`` None: the one it did last), to the
        brain's ``trick_feedback`` between two ticks; the answer says which brain took it (``brain``).
        A brain that keeps no tricks (the stand-in), or a duck not running, answers ``not_running``."""
        brain = self.lanes.brain
        own = getattr(brain, "trick_feedback", None)
        if self.state != "running" or not callable(own):
            return {"reason": "not_running"}
        with self._tick_lock:
            answer = dict(own(kind, trick) or {})
        answer["brain"] = type(brain).__name__
        return answer

    def _answer_play(self, *, refused: str | None = None) -> None:
        """Settle the page's press from the body link's last decision (or refuse it at once)."""
        play = self._play_req
        if play is None or play.done.is_set():
            return
        if refused is not None:
            reason = {"missing": msg("neuro.motion.missing")}.get(refused, msg("neuro.motion.late"))
            play.answer = {"ok": False, "refused": True, "reason": refused, "reason_zh": reason}
            self._skill_picker.cancel()
            play.done.set()
            return
        link = self._link
        last = None if link is None else link.last_skill_decision
        if not play.started or last is None:
            return
        skill, decision, at_s = last
        if skill != play.motion.entry or at_s < play.now_s - 1e-6:
            return
        if decision.ok:
            self._motion_seq += 1
            self._motion_run = MotionRun(
                play.motion, self._last_t_ms, self._motion_seq, ticks=list(self._recent)
            )
            play.answer = {"ok": True, "started": True, "reason": decision.reason}
        else:
            self._skill_picker.cancel()
            play.answer = {"ok": False, "refused": True, "reason": decision.reason,
                           "reason_zh": decision.reason_zh}  # fmt: skip
        play.done.set()

    def _motion_tick(self, tick: Any, t_ms: int) -> None:
        """Record the run's ticks; when the body is a plain posture again (or 14 s passed) say what
        it did (:func:`motion_result.summarise`) in :attr:`_motion_last`."""
        from duckmind.skilltrain import motion_result as MR

        run = self._motion_run
        if run is None:
            return
        run.ticks.append(tick)
        label = tick.policy or ""
        if MR.is_skill_label(label):
            run.seen = True
            run.plain_since_ms = None
        elif run.seen:
            if run.plain_since_ms is None:
                run.plain_since_ms = t_ms
        ended = (
            run.seen
            and run.plain_since_ms is not None
            and (t_ms - run.plain_since_ms >= MR.END_STILL_S * 1000)
            and self._skill_picker.follow is None  # (the startle's back-steps are still to come)
        )
        if (
            ended
            or t_ms - run.t0_ms > MR.MAX_RUN_S * 1000
            or (run.seen and bool(getattr(tick, "fallen", False)))
        ):
            motion = run.motion
            done = MR.summarise(run.ticks, run.t0_ms, motion.entry, self._family_of(motion))
            self._motion_last = {
                "seq": run.seq,
                "id": motion.id,
                "state": "done",
                "t_ms": t_ms,
                "verdict": done["verdict"],
                "numbers": done["numbers"],
                "zh": done["zh"],
                "tail_zh": done["tail_zh"],
            }
            self._motion_run = None

    @staticmethod
    def _family_of(motion: Any) -> str:
        from duckmind.skilltrain import live_judge as J

        found = J.skill_of(motion.entry)
        return found[1] if found is not None else motion.family

    def play_emotion(self, emotion_id: str) -> dict[str, Any]:
        """Do one emotion now (the page's 「情緒」 button): started, or why not in plain words."""
        return self.emotions.press(emotion_id)

    # ── remote mode: the family's hand on the legs (neuro/remote.py) ──

    @property
    def remote_active(self) -> bool:
        """The legs are the person's now (or are held at zero for the hand-back)."""
        return self.remote.on

    def remote_set(self, on: bool, *, owner: object | None = None, why: str = "off") -> dict[str, Any]:
        """Turn remote mode on or off (the page's toggle). On: the brain keeps stepping (head, voice, senses)
        but the legs get only the person's twist; off: the legs are held at zero for a moment, then the
        brain has them again."""
        if self.state != "running" or self._link is None:
            raise ValueError(msg("neuro.remote.not_running"))
        changed = self.remote.set_on(bool(on), time.monotonic(), why=why, owner=owner)
        if on:  # a person has the legs: an errand under way is over (the duck was not told to stop; it was taken)
            stop = getattr(getattr(self.lanes, "brain", None), "stop_errand", None)
            if callable(stop):
                with self._tick_lock:
                    stop("remote")
        return {"ok": True, "changed": changed, **self.remote.view(time.monotonic())}

    def remote_twist(
        self, vx: Any, vyaw: Any, vy: Any = 0.0, key_to_recv_ms: float | None = None
    ) -> dict[str, Any]:
        """A twist from the person (clamped to the walk policy's ranges; zero after the deadman)."""
        kept = self.remote.push(vx, vyaw, time.monotonic(), vy=vy, key_to_recv_ms=key_to_recv_ms)
        return {"ok": self.remote.on, "vx": kept[0], "vyaw": kept[1], "on": self.remote.on}

    def _remote_request(self, request: BodyRequest, twist: tuple[float, float]) -> BodyRequest:
        """What the legs get while remote mode has them: the person's twist and nothing of the brain's (no
        approach, docking, panic, pivot, trained skill or posture change); the head, the beak and the
        body link's own gates (fall, righting, lift, ToF stop, stall) stay."""
        return replace(
            request,
            vx=twist[0],
            vyaw=twist[1],
            approach=False,
            docking=False,
            back_arc=False,
            panic=False,
            pivot=0.0,
            skill=None,
            skill_manual=False,
            skill_chain=False,
            toward=None,
            posture=None,
        )

    # ── go there: a place somebody pointed at on the floor (neuro/errand.py) ──

    def go_there(self, x_m: Any, y_m: Any, who: str | None = None) -> dict[str, Any]:
        """The page's click on the floor: ``(x_m, y_m)`` is a place in the WORLD's frame. The brain never gets it.

        Conversion (here, on the world side, because only here is the world's frame known): the place is turned into
        what a person pointing from beside the duck would say, a bearing from its trunk and a range, by the physics
        body's pose as the page draws it (``_truth_pose``). Only that bearing and range go to the brain
        (``NeuroBrain.go_to``), which puts the place in its OWN odometry frame by its own odometry pose now. So the
        duck's belief of where "there" is is wrong by exactly what its odometry has drifted since the physics
        and the odometry last agreed: the contact odometry's yaw is exact in the simulator and its position creeps
        about 3 cm per 60 s walked (docs/design/neural-duck/LASTING-MAP-2026-10-11.md), so a click is placed
        to within ~3 cm plus the page's own picking error (a pixel, ~1 cm at the stage's zoom). On a real body there is no
        world frame: the family's point would come from the duck's own camera, and this method is the seam to replace."""
        try:
            x, y = float(x_m), float(y_m)
        except (TypeError, ValueError):
            raise ValueError(msg("neuro.gothere.bad_point")) from None
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(msg("neuro.gothere.bad_point"))
        brain = getattr(self.lanes, "brain", None)
        go_to = getattr(brain, "go_to", None)
        if self.state != "running" or not callable(go_to):
            return self._gothere_refused("not_running")
        if self.remote.on:
            return self._gothere_refused("remote")
        body = self._body
        sample = getattr(body, "latest", None) if body is not None else None
        pose = _truth_pose(sample) if isinstance(sample, Mapping) else None
        if pose is None:
            return self._gothere_refused("no_pose")
        dx, dy = x - pose[0], y - pose[1]
        range_m = math.hypot(dx, dy)
        if range_m > ERRAND_MAX_RANGE_M:
            return self._gothere_refused("too_far")
        bearing = math.atan2(dy, dx) - pose[2]
        bearing = math.atan2(math.sin(bearing), math.cos(bearing))
        with self._tick_lock:
            errand_id, ox, oy = go_to(bearing, range_m, who)
        self.telemetry_store.event(
            {
                "t_ms": self._last_t_ms,
                "wall_ms": int(time.time() * 1000),
                "kind": "world.errand",
                "errand": errand_id,
                "zh": msg("neuro.gothere.started", range=round(range_m, 2)),
            }
        )
        return {
            "ok": True,
            "queued": True,
            "id": errand_id,
            "range_m": round(range_m, 3),
            "bearing_rad": round(bearing, 3),
            "target_odom": {"x_m": round(ox, 3), "y_m": round(oy, 3)},
        }

    @staticmethod
    def _gothere_refused(reason: str) -> dict[str, Any]:
        return {
            "ok": False,
            "refused": True,
            "reason": reason,
            "reason_zh": msg(f"neuro.gothere.refused.{reason}"),
        }

    def errand_view(self) -> dict[str, Any]:
        """The errand's state for the telemetry: idle | waiting | walking | arrived | gave_up, with the reason."""
        brain = getattr(self.lanes, "brain", None)
        errand = getattr(brain, "errand", None)
        view = getattr(errand, "view", None)
        return dict(view(self._last_t_ms / 1000.0)) if callable(view) else {"state": "idle"}

    def motion_view(self) -> dict[str, Any]:
        """The page's 「動作試玩」 state: what runs now, what ran last."""
        run = self._motion_run
        return {
            "emotion": self.emotions.view(),
            "installed": sorted(self._link.skills.names)
            if self._link and self._link.skills
            else [],
            "running": None if run is None else {"id": run.motion.id, "seq": run.seq},
            "last": self._motion_last,
        }

    def _new_link(self) -> BodyLink:
        cfg = self.lanes.link
        return BodyLink(cfg) if cfg is not None else BodyLink()

    # ── lifecycle ──

    def attach(self, body: Any) -> None:
        """Take over ``body`` (a started :class:`~duckmind.livebody.LiveBody`): its own control
        loop goes passive and the tick thread starts on its robotd."""
        self.detach()
        if self.stats.attaches > 0:
            self._new_world()
        # the motor's efference copy follows what THIS body's robotd smooths the head with
        _sync_head_alpha(self.lanes, body)
        # (dock:build) which side its dock's J-hook turns round on: how the home set it up
        _sync_dock_side(self.lanes, body)
        with self._lock:
            self._body = body
            self._stop = threading.Event()
            self.state = "attaching"
            self.error = None
            self._t0_ns = None
            self._t_offset_ms = self._last_t_ms
            self._link = self._new_link()
        stop = self._stop
        if self.spec.record is not None and self._record is None:
            try:
                self._record = Path(self.spec.record).open("a", encoding="utf-8")
            except OSError as error:
                self.lanes.notes.append(f"record: {error}")
        body.passive = True  # one robotd writer at a time (spec §6): this driver from now on
        for target in (self._run, self._aux, self._tof):
            thread = threading.Thread(target=target, args=(body, stop), daemon=True)
            thread.start()
            self._threads.append(thread)
        self.stats.attaches += 1
        self.stats.attached_at = time.time()

    def _new_world(self) -> None:
        """A body came after another one (the layout editor's Apply reloads the physics): the new
        world's clock starts at 0 while the brain's goes on, so the senses start their models afresh
        (else they would drop every touch and sound of the new world for as long as the old one ran),
        and what the life duck lived is closed off at the old body's last moment."""
        with self._tick_lock:
            fresh = getattr(self.lanes.senses, "new_world", None)
            if callable(fresh):
                fresh()
            life = self.life
            if life is not None:
                try:
                    life.body_replaced()
                except Exception as error:  # growth's bookkeeping never stops the body
                    self.stats.errors += 1
                    self.stats.last_error = f"experience: {type(error).__name__}: {error}"

    def detach(self) -> None:
        """Let go of the body (before it is stopped or replaced): a zero twist, head HOME."""
        with self._lock:
            body, self._body = self._body, None
            self._stop.set()
            threads, self._threads = self._threads, []
        for thread in threads:
            if thread is not threading.current_thread():
                thread.join(timeout=2.0)
        self._deferred_lane_close = any(thread.is_alive() for thread in threads)
        client, self._client = self._client, None
        if client is not None:  # (the tick thread did not get to let go by itself)
            self._let_go(client)
            _close_client(client)
        if body is not None and self.state not in ("stopped",):
            self.state = "detached"

    def _let_go(self, client: RobotdClient) -> None:
        """A zero twist, the head HOME and the beak closed, before the connection closes."""
        link = self._link
        if link is None or link.state is None:
            return
        try:
            for intent in link.stop_intents(link.state.t_ns):
                client.send(intent)
        except Exception:
            pass

    def stop(self) -> None:
        self.detach()
        lanes = self.lanes
        if not self._deferred_lane_close:
            _close_lane_sound(lanes)
        self.state = "stopped"
        if self._record is not None:
            try:
                self._record.close()
            except OSError:
                pass
            self._record = None

    # ── threads ──

    def _connect(self, body: Any, stop: threading.Event) -> RobotdClient | None:
        from duckmind.duckbody.server import UnixWire

        deadline = time.monotonic() + ATTACH_WAIT_S
        while not stop.is_set() and time.monotonic() < deadline:
            sock = getattr(body, "robotd_socket", None)
            if sock is not None and Path(sock).exists():
                try:
                    client = RobotdClient(UnixWire(sock), UnixWire(sock), UnixWire(sock))
                    client.start(read_model=False)
                    return client
                except (OSError, RobotdError, ConnectionError, ValueError) as error:
                    self.stats.last_error = f"connect: {error}"
            stop.wait(RECONNECT_S)
        return None

    def _run(self, body: Any, stop: threading.Event) -> None:
        while not stop.is_set():
            client = self._connect(body, stop)
            if client is None:
                if stop.is_set() and self._deferred_lane_close:
                    _close_lane_sound(self.lanes)
                    self._deferred_lane_close = False
                if not stop.is_set():
                    self.state = "error"
                    self.error = msg("neuro.driver_error.robotd")
                return
            self._client = client
            self.state = "running"
            try:
                self._loop(client, stop)
            except (OSError, RobotdError, ConnectionError, ValueError) as error:
                self.stats.errors += 1
                self.stats.last_error = f"robotd: {error}"
                if self._link is not None:
                    self._link.proprio.epoch += 1  # robotd went away: odometry broke
            except Exception as error:  # a bug in a lane: say so, keep the body safe
                self.stats.errors += 1
                self.stats.last_error = f"{type(error).__name__}: {error}"
                self.error = msg(
                    "neuro.driver_error.crash", error_type=type(error).__name__, error_text=error
                )
                traceback.print_exc()
                self.state = "error"
                self._let_go(client)
                _close_client(client)
                self._client = None
                return
            if stop.is_set():
                self._let_go(client)
            _close_client(client)
            self._client = None
            if stop.is_set() and self._deferred_lane_close:
                _close_lane_sound(self.lanes)
                self._deferred_lane_close = False
            if not stop.is_set():
                stop.wait(RECONNECT_S)

    def _loop(self, client: RobotdClient, stop: threading.Event) -> None:
        while not stop.is_set():
            state = client.read_state(READ_TIMEOUT_S)
            if state is None:
                continue
            waiting: list[RobotState] = []
            while True:  # a backlog (the process was busy): catch up to the newest tick
                more = client.read_state(0.0)
                if more is None:
                    break
                waiting.append(more)
            if len(waiting) > BACKLOG_TICKS:
                self.stats.skipped += len(waiting) - 1
                states = [waiting[-1]]
            else:
                states = [state, *waiting] if waiting else [state]
            for item in states:
                try:
                    intents = self.tick(item)
                    self._failing = 0
                except Exception as error:  # a lane failed this tick: hold still, carry on
                    self._failing += 1
                    self.stats.errors += 1
                    self.stats.last_error = f"tick: {type(error).__name__}: {error}"
                    if self._failing == 1:
                        traceback.print_exc()
                    if self._failing >= FAILING_TICKS:
                        raise  # (it keeps failing: the driver stops and says so)
                    link = self._link
                    intents = [] if link is None else link.stop_intents(item.t_ns)
                for intent in intents:
                    client.send(intent)

    def _aux(self, body: Any, stop: threading.Event) -> None:
        """A few times a second: the dock contact (``pg``: the PG pin, a real door), the world's
        truth and the physics frame for the world-side sensor models, the duck's clock."""
        period = 1.0 / AUX_HZ
        while not stop.is_set():
            call = getattr(body, "world_call", None)
            if callable(call):
                try:
                    self._pg = bool(call("pg").get("pg"))
                except Exception:
                    self._pg = None
            world = getattr(body, "world", None)
            frame = getattr(body, "frame", None)
            try:
                self._world = world() if callable(world) else None
                self._truth = frame() if callable(frame) else None
                self._world_at = time.monotonic()
            except Exception:
                self._world = self._truth = None
            clock = self.clock
            if clock is not None:
                try:
                    self._day_minute = _day_minute(clock())
                except Exception:
                    pass
            else:
                self._day_minute = self.day_clock.minute()
            due = [item for item in self._leaves if item[0] <= time.monotonic()]
            if due and callable(call):
                self._leaves = [item for item in self._leaves if item not in due]
                for _, who in due:  # a walk-by: the person goes on their way
                    try:
                        call("x_people", do="leave", who=who)
                    except Exception:
                        pass
            # (fid:integrate) a page stimulus wakes it (:meth:`inject`) and it then reads the
            # world every AUX_FAST_S for AUX_FAST_FOR_S: the body's copy of the world (LiveBody
            # reads x_world 12 times a second) does not carry the new event yet at the first
            # read, and waiting a whole 5-Hz period after it made every stimulus's reflex ~0.1 s
            # later (live, 2 × 36 presses: press → decision p10 0.26 s; it was 0.12–0.15 s with
            # the plain poll). ``stop`` is seen within one period.
            wait = AUX_FAST_S if time.monotonic() < self._aux_fast_until else period
            if self._aux_wake.wait(wait):
                self._aux_wake.clear()
                self._aux_fast_until = time.monotonic() + AUX_FAST_FOR_S

    def _tof(self, body: Any, stop: threading.Event) -> None:
        """tofd's ``tof.stream`` (a duck with the ToF module; the world server runs the official
        tofd --sim): the newest frame, for the tick thread's depth summary."""
        from duckmind.duckbody.robotd_client import API_VERSION
        from duckmind.duckbody.server import UnixWire

        path = _tofd_socket(body)
        while path is not None and not stop.is_set():
            wire = None
            try:
                wire = UnixWire(path)
                wire.send(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "hello",
                        "params": {"api_version": API_VERSION},
                    }
                )
                wire.recv(2.0)
                wire.send({"jsonrpc": "2.0", "id": 2, "method": "tof.stream", "params": {}})
                while not stop.is_set():
                    line = wire.recv(0.5)
                    if line is None or line.get("method") != "tof.frame":
                        continue
                    params = line.get("params")
                    if isinstance(params, dict):
                        self._tof_frame = params
            except (OSError, ConnectionError, ValueError):
                self._tof_frame = None
                stop.wait(RECONNECT_S)
            finally:
                if wire is not None:
                    wire.close()

    def _depth_step(self, state: RobotState, walking: bool) -> P.DepthView | None:
        """duck-body's DS-v1 summary of tofd's newest frame (DepthTracker, the sensor's pose from
        robot.state's ``frames.tof``); None without a ToF or with a stale one (> 0.5 s)."""
        frame = self._tof_frame
        if frame is None or state.tof is None:
            return None if state.t_ns - self._depth_at_ns > 500_000_000 else self._depth
        if frame is not self._tof_seen:
            from duckmind.duckbody import depth as D

            self._tof_seen = frame
            if self._depth_tracker is None:
                self._depth_tracker = D.DepthTracker(standing_height_m=0.12)
            pos, quat = state.tof
            pose = D.pose_from_frames(pos, quat, state.down or (0.0, 0.0, -1.0))
            tof = D.ToFFrame.of(
                int(frame.get("t_ns") or state.t_ns),
                frame.get("distance_mm") or [],
                frame.get("status") or [],
            )
            try:
                reading, _ = self._depth_tracker.step(
                    tof,
                    pose,
                    walking=walking,
                    odom=(
                        float(state.odom_pos[0]),
                        float(state.odom_pos[1]),
                        float(state.odom_yaw),
                    ),
                    upright=D.upright(state.odom_pos),
                )
            except (ValueError, IndexError):
                return None
            if not reading.available:
                self._depth = None
            else:
                far = D.FAR_MM

                def rng(value: int) -> int | None:
                    return None if value >= far else value

                self._depth = P.DepthView(
                    t_ns=state.t_ns,
                    ahead_mm=rng(reading.ahead_mm),
                    left_mm=rng(reading.left_mm),
                    right_mm=rng(reading.right_mm),
                    drop_zones=max(reading.drop_zones, 4)
                    if reading.drop
                    else min(reading.drop_zones, 3),
                    touch=reading.touch,
                    head_pitch_mrad=reading.head_pitch_mrad,
                    ahead_b_mrad=reading.ahead_b_mrad,
                    ahead_top_mm=reading.ahead_top_mm,
                    virtual=reading.virtual,
                    low_edge_mm=reading.low_edge_mm,
                )
            self._depth_at_ns = state.t_ns
        if state.t_ns - self._depth_at_ns > 500_000_000:
            return None
        return self._depth

    # ── one tick (pure but for the telemetry's wall stamp) ──

    def tick(self, state: RobotState) -> list[P.Intent]:
        with self._tick_lock:
            return self._tick_locked(state)

    def _tick_locked(self, state: RobotState) -> list[P.Intent]:
        """One robot.state tick: senses → brain → motor → body link. Returns the intents."""
        started = time.perf_counter()
        lanes = self.lanes  # switch() holds _tick_lock while replacing/retiring a lane set
        link = self._link
        if link is None:
            link = self._link = self._new_link()
        if self._t0_ns is None:
            self._t0_ns = state.t_ns
        t_ms = self._t_offset_ms + max(0, (state.t_ns - self._t0_ns) // 1_000_000)
        t_ms = max(t_ms, self._last_t_ms)
        self._last_t_ms = t_ms
        last = self._last_out
        world, truth = self._world, self._truth
        battery, charging = _battery(world)
        events = _events_of(last)
        depth = self._depth_step(state, bool(last is not None and last.twist != (0.0, 0.0)))
        doors = Doors(
            t_ms=t_ms,
            state=state,
            link=last,
            events=events,
            tof=self._tof_frame,
            depth=depth,
            pg=self._pg,
            day_minute=self._day_minute,
            world=world,
            truth=truth,
            world_age_ms=int((time.monotonic() - self._world_at) * 1000)
            if self._world_at
            else None,
            stimuli=self.recent_stimuli(t_ms),
        )
        if lanes.senses is not None:
            frame = lanes.senses(doors)
        else:
            frame = body_frame(
                t_ms,
                state,
                None if last is None else last.prox.view,
                events=events,
                battery=battery,
                on_dock=self._pg,
                charging=charging,
                day_minute=self._day_minute,
            )
        life = self.life
        if life is not None:
            try:  # growth G3: the body grows in its sleep, between two ticks, never awake
                life.regrow_tick(self, lanes, frame, state, t_ms)
            except Exception as error:  # growth's bookkeeping never stops the body loop
                self.stats.errors += 1
                self.stats.last_error = f"regrow: {type(error).__name__}: {error}"
        brain_start = time.perf_counter()
        if hasattr(lanes.brain, "light"):
            from duckmind.simweb.neuro_sound_live import link_light

            lanes.brain.light = link_light(last)
        if hasattr(lanes.brain, "can_roll"):
            # the flip trick is asked only when robotd runs the official roulade and the guard has its rule
            lanes.brain.can_roll = link.skills is not None and SG.ROLL_ENTRY in link.skills.names
        if hasattr(lanes.brain, "can_pivot"):
            # a command's search turns on the spot only when robotd runs the quick turn
            lanes.brain.can_pivot = link.skills is not None and bool(
                SG.quick_turn_names(link.skills.names)
            )
        brain_frame = frame
        # the roll it asked for (the flip): robotd's fall flag rises in every roulade, but the body link holds the roll as a
        # lie-down on purpose until the duck has stood up a second (or gives it up as a fall after its hold) — the brain hears
        # the same, else its flip ends "fallen" mid-roll (real-simulator measurement 2026-10-26)
        rolling = link.was_lying or (state.policy or "").split(":", 1)[0] == SG.ROLL_ENTRY
        if self.emotions.lying or rolling:
            # it lies down on purpose (play dead, a roll): to the brain that is no fall (its own choice; the growth input and
            # the emotion layer still read the frame as it is)
            brain_frame = replace(frame, body=replace(frame.body, fallen=False))
        own_roll = link.rolling_until_ns > 0 or (state.policy or "").split(":", 1)[0] == SG.ROLL_ENTRY
        if own_roll:
            self._roll_quiet_until_ms = t_ms + ROLL_QUIET_MS
        if own_roll or t_ms < self._roll_quiet_until_ms:
            # its own roll: the floor on its back, the bumps and thumps and the streaming, looming picture are its own doing
            # (it asked for the roll), not a fright, until ROLL_QUIET_MS after it is up again (the real simulator, 2026-10-26:
            # a startle at its own thump or a withdraw at the floor's touch mid-roll cut the roll short and a flight followed)
            brain_frame = _without_own_roll(brain_frame)
        cmd, brain_tel = lanes.brain.step(t_ms, brain_frame)
        self.stats.brain_ms.append((time.perf_counter() - brain_start) * 1000.0)
        life = self.life
        if life is not None:
            try:  # what it lives goes to Duck Core (docs/design/GROWTH-G2-2026-10-10.md)
                life.on_tick(t_ms, frame, cmd, brain_tel)
            except Exception as error:  # growth's bookkeeping never stops the body loop
                self.stats.errors += 1
                self.stats.last_error = f"experience: {type(error).__name__}: {error}"
        if lanes.motor is not None:
            request, vocals = lanes.motor(t_ms, cmd, frame, state)
        else:
            request, vocals = self._stub_motor.step(t_ms, cmd, frame)
        # an emotion plays (Pollen's microduck_emotions): it has the body and the brain holds still. Under a
        # remote hand none is begun (one that is already playing is played out: a play dead is never cut)
        remote_twist = self.remote.poll(time.monotonic())
        if remote_twist is not None and self.emotions.run is None:
            emoting = False
        else:
            request, emoting = self.emotions.step(
                cmd=cmd,
                brain_tel=brain_tel,
                frame=frame,
                state=state,
                request=request,
                t_ms=t_ms,
                last=last,
            )
        if emoting:
            vocals = []
            self._answer_play(refused="late")
        elif remote_twist is not None:
            # remote mode: the legs are the person's (the brain steps on, its head and voice carry on)
            request = self._remote_request(request, remote_twist)
            self._answer_play(refused="late")
        else:
            request = self._skill_request(link, cmd, state, request, brain_tel, t_ms)
        battery_pct = None if battery is None else int(round(battery * 100))
        out = link.tick(state, request, depth=depth, battery_pct=battery_pct)
        if remote_twist is not None:
            self.remote.note_tick(time.monotonic(), out.twist, float(state.applied[0]))
        self._answer_play()
        if hasattr(lanes.brain, "play_out") and remote_twist is None:
            from duckmind.simweb.neuro_runtime import OneShotBridge

            if not isinstance(self._one_shots, OneShotBridge):
                self._one_shots = OneShotBridge()
            out = self._one_shots.step(
                t_ms,
                state,
                request,
                cmd,
                lanes.brain,
                link,
                out,
                battery_pct=battery_pct,
            )
        self._last_out = out
        self._last_cmd = cmd
        self._tick_vocals = list(vocals)
        vocals = self._lineage_vocals(cmd, vocals, t_ms)
        self.stats.ticks += 1
        self.stats.refused += len(out.refused)
        self.stats.tick_wall.append(time.monotonic())
        self._report(t_ms, state, frame, cmd, brain_tel, request, out, vocals)
        self._witness_tick(t_ms, state, cmd, brain_tel, out, vocals)
        self._loop_tick(t_ms, state, cmd, request, out)
        try:
            from duckmind.simweb.neuro_sound_live import after_tick

            intents = after_tick(self, state, list(out.intents), lanes=lanes)
        except Exception as error:  # sound output cannot stop the robot body loop
            self.stats.errors += 1
            self.stats.last_error = f"sound adapter: {type(error).__name__}: {error}"
            intents = list(out.intents)
        self.stats.step_ms.append((time.perf_counter() - started) * 1000.0)
        return intents

    def _lineage_vocals(
        self, cmd: C.MotorCommand, vocals: list[dict[str, Any]], t_ms: int
    ) -> list[dict[str, Any]]:
        """The calls of this tick as the duck's voice lineage says them (docs/design/GROWTH-G5-2026-10-11.md):
        the life duck's learned calls and its age's pitch and length; a demo duck's, and a duck with no
        lineage voice, stay what the brain chose. Never stops the body loop."""
        life = self.life
        lineage = getattr(life, "voice_lineage", None) if life is not None else None
        if lineage is None or not getattr(self.spec, "life", False):
            return vocals
        try:
            return list(lineage.shape(list(vocals), str(cmd.behaviour), float(t_ms)))
        except Exception as error:
            self.stats.errors += 1
            self.stats.last_error = f"lineage voice: {type(error).__name__}: {error}"
            return vocals

    def _loop_tick(
        self, t_ms: int, state: RobotState, cmd: C.MotorCommand, request: BodyRequest, out: LinkOut
    ) -> None:
        """Round 5: one compact record for the loop detector (:mod:`duckmind.simweb.loopwatch`). Reads only; any error is swallowed."""
        try:
            body = self._body
            sample = getattr(body, "latest", None) if body is not None else None
            pose = _truth_pose(sample) if isinstance(sample, Mapping) else None
            self.loopwatch.feed(
                {
                    "t_ms": t_ms,
                    "wall": round(time.time(), 3),
                    "behaviour": cmd.behaviour,
                    "reflex": cmd.reflex,
                    "policy": state.policy,
                    "cmd": [round(cmd.vx, 3), round(cmd.vyaw, 3)],
                    "applied": [round(float(v), 3) for v in state.applied],
                    "tilt": None
                    if state.tilt_rad is None
                    else round(math.degrees(state.tilt_rad), 1),
                    "request": {
                        "vx": round(request.vx, 3),
                        "vyaw": round(request.vyaw, 3),
                        "panic": bool(request.panic),
                        "pivot": request.pivot,
                        "skill": request.skill,
                        "posture": request.posture,
                        # (2026-10-11) what else keeps the wall pivot from turning the duck away: a goal walk, a docking, a reverse
                        "approach": bool(request.approach),
                        "docking": bool(request.docking),
                        "back_arc": bool(request.back_arc),
                        "toward": request.toward,
                    },
                    "link": {
                        "wall_pivot": int(self._link.wp[0]) if self._link and self._link.wp else 0,
                        "twist": [round(float(v), 3) for v in out.twist],
                        "gate": list(out.gate),
                        "refused": [list(r) for r in out.refused],
                        "events": [dict(e) for e in out.events],
                        "do": [
                            str(i.params.get("skill"))
                            for i in out.intents
                            if i.method == "robot.do"
                        ],
                    },
                    "pose": None if pose is None else [round(v, 3) for v in pose],
                }
            )
        except Exception:  # a detector must never reach the body loop
            pass

    def _witness_tick(
        self,
        t_ms: int,
        state: RobotState,
        cmd: C.MotorCommand,
        brain: C.BrainTelemetry,
        out: LinkOut,
        vocals: list[dict[str, Any]] | None = None,
    ) -> None:
        """fid:page: the observer's tick — robotd's measured head and its own head command, the
        twist sent, the newest physics sample of the trunk — and the brain's recent reflexes;
        and, for the page's body samples, the beak command separately from the measured mouth
        joint when the composed model has a jaw, plus calls voiced this tick."""
        body = self._body
        if body is not None and hasattr(body, "mouth_cmd"):
            body.mouth_cmd = float(out.mouth)
            voice = getattr(body, "voice", None)
            if callable(voice):
                for vocal in vocals or ():
                    extra = {k: vocal[k] for k in ("parts", "lib") if vocal.get(k)}
                    voice(str(vocal.get("tag")), float(vocal.get("level") or 0.5), **extra)
        trunk = trunk_t = None
        sample = getattr(body, "latest", None) if body is not None else None
        if isinstance(sample, Mapping):
            pose = _truth_pose(sample)
            if pose is not None:
                trunk = (pose[0], pose[1], float(sample["trunk"][2]), pose[2])
                trunk_t = float(sample.get("t") or 0.0)
        view = out.prox.view
        tilt = state.tilt_rad
        tick = WitnessTick(
            t_ms=t_ms,
            head=state.head_offsets,
            head_cmd=state.head_cmd,
            twist_cmd=(float(out.twist[0]), float(out.twist[1])),
            policy=state.policy,
            trunk=trunk,
            trunk_t=trunk_t,
            fallen=bool(view.fallen),
            lifted=bool(view.lifted),
            holding=brain.reflex,
            behaviour=cmd.behaviour,
            # (fid:fix) for the body-confirmed stumbles, and the moments the head performed
            tilt_deg=None if tilt is None else math.degrees(tilt),
            moment=cmd.moment,
            moment_seq=int(cmd.moment_seq),
        )
        events = [
            {
                "t_ms": e.t_ms,
                "reflex": e.reflex,
                "stimulus": e.stimulus,
                "zh": e.zh,
                "parts": list(getattr(e, "parts", ()) or ()),
            }
            for e in brain.reflexes
        ]
        with self._witness_lock:
            self.witness.feed(tick, events, out.events)
        self._recent.append(tick)
        if self._motion_run is not None:
            self._motion_tick(tick, t_ms)

    def _speed_true(self) -> float | None:
        with self._witness_lock:
            speed = self.witness.speed_now()
        return None if speed is None else round(speed, 3)

    def _turn_true(self) -> float | None:
        with self._witness_lock:
            turn = self.witness.turn_now()
        return None if turn is None else round(turn, 3)

    def witness_view(self) -> dict[str, Any]:
        """The witness for the page: its audits (the last 40) and what the body does now."""
        with self._witness_lock:
            return {
                "audits": list(self.witness.audits)[-60:],
                "now": self.witness.now(),
                "seq": self.witness.seq,
            }

    def _report(
        self,
        t_ms: int,
        state: RobotState,
        frame: C.SensorFrame,
        cmd: C.MotorCommand,
        brain: C.BrainTelemetry,
        request: BodyRequest,
        out: LinkOut,
        vocals: list[dict[str, Any]],
    ) -> None:
        store = self.telemetry_store
        wall_ms = int(time.time() * 1000)
        self._mouth_peak = max(self._mouth_peak, float(out.mouth))
        for vocal in vocals:
            tag = str(vocal.get("tag"))
            parts = vocal.get("parts")
            store.event(
                {
                    "t_ms": t_ms,
                    "wall_ms": wall_ms,
                    "tag": tag,
                    # (a learned phrase or the song: its syllables' words, one after another)
                    "zh": lst([VOCAL_ZH.get(str(p), str(p)) for p in parts], sep="fmt.sep.babble")
                    if parts
                    else VOCAL_ZH.get(tag, tag),
                    **{k: v for k, v in vocal.items() if k != "tag"},
                },
                vocal=True,
            )
        for event in out.events:
            store.event({"t_ms": t_ms, "wall_ms": wall_ms, **_rounded(dict(event))})
        if self._record is not None:
            try:
                self._record.write(
                    json.dumps(
                        {"t_ms": t_ms, "frame": frame.to_json(), "cmd": cmd.to_json()},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            except (OSError, TypeError, ValueError):
                pass
        if self.stats.ticks % SAMPLE_EVERY_TICKS != 1:
            return
        view = out.prox.view
        store.sample(
            {
                "t_ms": t_ms,
                "wall_ms": wall_ms,
                "behaviour": cmd.behaviour,
                "behaviour_zh": brain.behaviour_zh,
                "reason_zh": brain.reason_zh,
                "bout": cmd.bout,
                "reflex": brain.reflex,
                "moment": brain.moment,  # a glance, a tilt, a still moment inside the act
                "arousal": _rounded(brain.arousal),
                "valence": _rounded(brain.valence),
                "drives": _rounded(dict(brain.drives)),
                "wanted": _rounded([cmd.vx, cmd.vyaw]),
                # (2026-10-11) the clutter escape: 1 while it has the legs, and how many it has begun (the soak and trial recorders read them)
                "clutter": int(brain.locomotion.get("clutter_escape", 0.0)),
                "clutter_n": int(brain.locomotion.get("clutter_escapes", 0.0)),
                "twist": _rounded(list(out.twist)),
                "applied": _rounded([state.applied[0], state.applied[2]]),
                "head": _rounded(list(out.head)),
                "head_measured": _rounded(list(state.head_offsets)),
                "mouth": _rounded(out.mouth),
                "mouth_measured_rad": _rounded(frame.body.mouth),
                "mouth_source": frame.sources.get("body"),
                "mouth_peak": _rounded(self._mouth_peak),  # the widest since the last sample
                "day_minute": round(self._day_minute, 2),
                "posture_want": out.posture_want,
                "emotion": None
                if self.emotions.run is None or self.emotions.run.player is None
                else self.emotions.run.emotion.name,
                "policy": state.policy,
                "posture": view.posture,
                "fallen": view.fallen,
                "lifted": view.lifted,
                "unloaded": view.unloaded,
                "righting": out.righting,
                "tilt_deg": round(math.degrees(view.tilt_mrad / 1000.0), 1),
                "gate": list(out.gate),
                "gate_zh": [GATE_ZH.get(code, code) for code in out.gate],
                "pg": self._pg,
                # (dock:build) the docking program: mode, phase and its words, charging
                "dock": _dock_sample(brain),
                # (fid:fix) the trunk's real speed (the physics truth, the last 0.5 s): the page's
                # walking rhythm and speed words show the body, not the command
                "speed_true": self._speed_true(),
                "turn_true": self._turn_true(),
                # (remote mode) 1 while the family's hand has the legs
                "remote": 1 if self.remote.on else 0,
            }
        )
        self._mouth_peak = 0.0
        store.latest(
            _rounded(brain.to_json()),
            _rounded(frame.to_json()),
            _rounded(
                {
                    **cmd.to_json(),
                    "request": _request_json(request),
                    "motor": getattr(self.lanes.motor, "last", None),
                }
            ),
        )

    # ── what the page reads ──

    def view(self) -> dict[str, Any]:
        """A small summary for ``/api/body`` (``neuro``)."""
        latest = self.telemetry_store.samples[-1] if self.telemetry_store.samples else None
        return {
            "enabled": True,
            "state": self.state,
            "error": self.error,
            "lanes": dict(self.lanes.names),
            "notes": list(self.lanes.notes),
            "genome": self.lanes.genome,
            "now": latest,
            "vocals": list(self.telemetry_store.vocals)[-5:],
            "stats": self.stats.view(),
            "remote": self.remote.view(time.monotonic()),
            "errand": self.errand_view(),
        }

    def telemetry(self, since: int = 0) -> dict[str, Any]:
        """Everything since sample/event ``since`` (for the page's panel and ``neuro_api``)."""
        body = self._body
        status: dict[str, Any] = {}
        try:
            status = dict(body.status()) if body is not None else {}
        except Exception:
            status = {}
        command = self._last_cmd
        try:
            from duckmind.simweb.neuro_runtime import skill_status

            skills = skill_status(
                self.lanes,
                None if command is None else command.behaviour,
                self._last_t_ms if command is not None else None,
                reflex=None if command is None else command.reflex,
                actions=self._one_shots.view(),
                world_errors=getattr(self.lanes.senses, "world_errors", ()),
                # Match /api/neuro/sound/status: ToF is ready only after an actual driver frame.
                # No inferred/fitted sensor capability is substituted for a reading.
                tof_ready=bool(self._tof_frame),
            )
        except Exception as error:
            skills = {
                key: {
                    "available": False,
                    "reason": f"serializer_error_{type(error).__name__.lower()}",
                    "phase": None,
                    "status": "unknown",
                    "result": None,
                    "updated_ms": self._last_t_ms if command is not None else None,
                }
                for key in (
                    "follow",
                    "hide",
                    "circle",
                    "theremin",
                    "ride",
                    "dance",
                    "ball",
                    "carry",
                    "sun",
                )
            }
        return {
            "state": self.state,
            "error": self.error,
            "lanes": dict(self.lanes.names),
            "notes": list(self.lanes.notes),
            "genome": self.lanes.genome,
            "stats": self.stats.view(),
            "realtime": status.get("realtime"),
            "monitor": status.get("monitor"),
            "gate_zh": GATE_ZH,
            "vocal_zh": VOCAL_ZH,
            "day": self.day_clock.view() if self.clock is None else None,
            "helps": self.helps,
            "looks": self._looks(),
            "skills": skills,
            "actions": self._one_shots.view(),
            "witness": self.witness_view(),  # fid:page: what the body measurably did
            "motion": self.motion_view(),  # 「動作試玩」: what runs now, what ran last
            "remote": self.remote.view(time.monotonic()),  # remote mode: on/off, the twist, the edge latencies
            "errand": self.errand_view(),  # go there: idle | waiting | walking | arrived | gave_up (+ reason)
            **self.telemetry_store.since(int(since)),
        }

    def knows(self) -> dict[str, Any]:
        """(MAP-EXPLORE-2026-10-12) What the duck knows of its home, for the page's minimap layer: its own lasting map as a coarse
        run-length coded grid (``LastMap.coarse``, the map frame), what it knows of its home
        (``home_view``), and — for drawing only — where that map lies in the world: the physics body's true pose composed with the
        inverse of the duck's own pose estimate (``world``: the map frame's origin as x, y in m and a heading in rad; it moves when
        the duck's odometry has drifted, which is the honest picture). Nothing here reaches the brain. ``on`` False: no map."""
        brain = getattr(self.lanes, "brain", None)
        lmap = getattr(brain, "lastmap", None)
        cfg = getattr(lmap, "cfg", None)
        if lmap is None or cfg is None or not cfg.enabled:
            return {"ok": True, "on": False}
        with self._tick_lock:
            grid = lmap.coarse()
            odom = lmap.last_odom()
            mine = None if odom is None else lmap.pose(odom)
            known = getattr(brain, "home_known", None)
            home = known() if callable(known) else None
        body = self._body
        sample = getattr(body, "latest", None) if body is not None else None
        truth = _truth_pose(sample) if isinstance(sample, Mapping) else None
        world = None
        if truth is not None and mine is not None:
            yaw = math.atan2(math.sin(truth[2] - mine[2]), math.cos(truth[2] - mine[2]))
            c, s = math.cos(yaw), math.sin(yaw)
            world = {
                "x": round(truth[0] - (c * mine[0] - s * mine[1]), 3),
                "y": round(truth[1] - (s * mine[0] + c * mine[1]), 3),
                "yaw": round(yaw, 4),
            }
        return {
            "ok": True,
            "on": True,
            "grid": grid,
            "world": world,
            "home": home,
        }


def _sync_head_alpha(lanes: Lanes, body: Any) -> None:
    """The motor lane's efference copy of robotd's head smoothing = what ``body``'s robotd
    was started with (``LiveBody.robotd_head_alpha``; fid:motor)."""
    alpha = getattr(body, "robotd_head_alpha", None) if body is not None else None
    setter = getattr(lanes.motor, "set_head_alpha", None)
    if alpha is not None and callable(setter):
        setter(float(alpha))


def _dock_sample(brain: C.BrainTelemetry) -> dict[str, Any] | None:
    """The docking program's state for the 10 Hz samples (the page's 充電 state)."""
    dock = getattr(brain, "dock", None)
    if not isinstance(dock, Mapping):
        return None
    keys = ("mode", "phase", "zh", "pg", "charging", "tries", "docked", "result", "result_zh")
    return {k: dock.get(k) for k in keys}


def _sync_dock_side(lanes: Lanes, body: Any) -> None:
    """(dock:build 2026-09-28) The neural duck's J-hook side is how the family set its dock up —
    the home layout's ``nest.side`` (default_home: right; the apartment: left, the corridor wall
    is on the dock's right). A setting of the home, like duck-body's own dock config; never a
    live reading of the world."""
    set_dock = getattr(lanes.brain, "set_dock", None)
    home = getattr(body, "home", None)
    if not callable(set_dock) or not home:
        return
    try:
        from duckmind.world import homelayout

        nest = homelayout.load(str(home)).nest
        if nest is not None and getattr(nest, "is_dock", False):
            set_dock(str(getattr(nest, "side", "") or "right"))
    except Exception as error:  # a home without a dock (or an old layout): the default side
        lanes.notes.append(
            msg("neuro.lane_note.sync_dock_side", error_type=type(error).__name__, error_text=error)
        )


def _world(call: Callable[..., Any], op: str, params: Mapping[str, Any]) -> dict[str, Any]:
    """One world op for a stimulus; its refusal in the world's words (ValueError)."""
    try:
        return dict(call(op, **dict(params)) or {})
    except Exception as error:
        text = str(error)
        if "'error':" in text:
            text = text.split("'error':", 1)[1].strip(" }'\"")
        raise ValueError(
            msg("neuro.world_refusal.411c43", error_type=text or type(error).__name__)
        ) from error


def _without_own_roll(frame: C.SensorFrame) -> C.SensorFrame:
    """The frame with what the duck's own roll does to its senses taken out: no touch, no image motion or looming, no
    bump / brush / stumble edge, no sudden sound that is not a word (its eye still sees things, its ear still hears
    speech, its body still feels its tilt)."""
    camera = frame.camera
    sightings = tuple(
        replace(s, motion=0.0, looming=0.0) for s in camera.sightings if s.kind not in ("motion", "looming")
    )
    sound = frame.sound
    if sound.kind not in HEARD_THROUGH_ROLL:
        sound = replace(sound, onset=0.0, kind=None)
    return replace(
        frame,
        touch=C.TouchSense(),
        sound=sound,
        camera=replace(camera, motion=0.0, sightings=sightings),
        events=tuple(e for e in frame.events if e not in OWN_ROLL_EVENTS),
    )


def _truth_pose(truth: Mapping[str, Any] | None) -> tuple[float, float, float] | None:
    """The physics body's x, y (m) and heading (rad) from a body frame (world side only)."""
    if not truth or truth.get("trunk") is None or truth.get("quat") is None:
        return None
    w, x, y, z = (float(v) for v in truth["quat"][:4])
    heading = 2 * math.atan2(z, w) if abs(w) > 1e-9 or abs(z) > 1e-9 else 0.0
    trunk = truth["trunk"]
    return float(trunk[0]), float(trunk[1]), math.atan2(math.sin(heading), math.cos(heading))


def _nearest_person(
    world: Mapping[str, Any] | None, pose: tuple[float, float, float] | None
) -> tuple[list[float], str] | None:
    """The person nearest the duck (x_world people; world side only): (pos_mm, who)."""
    people = (world or {}).get("people")
    if not isinstance(people, list):
        return None
    best: tuple[float, list[float], str] | None = None
    for person in people:
        if not isinstance(person, Mapping) or not person.get("pos_mm"):
            continue
        pos = [float(v) for v in person["pos_mm"]]
        d = 0.0 if pose is None else math.hypot(pos[0] / 1000 - pose[0], pos[1] / 1000 - pose[1])
        if best is None or d < best[0]:
            best = (d, pos, str(person.get("who") or ""))
    return None if best is None else (best[1], best[2])


def _tofd_socket(body: Any) -> str | None:
    """The world's tofd socket when the duck has the ToF module fitted (``x_sensors``), else
    None (most ducks: every rule must work without it)."""
    call = getattr(body, "world_call", None)
    ops = (getattr(body, "host_info", None) or {}).get("ops") or []
    if not callable(call) or "x_sensors" not in ops:
        return None
    try:
        sensors = call("x_sensors")
    except Exception:
        return None
    if not (sensors.get("tof") or {}).get("fitted"):
        return None
    socket_path = (sensors.get("tofd") or {}).get("socket")
    return str(socket_path) if socket_path else None


def _request_json(request: BodyRequest) -> dict[str, Any]:
    return {
        "vx": round(request.vx, 3),
        "vyaw": round(request.vyaw, 3),
        "head": None if request.head is None else [round(v, 3) for v in request.head],
        "mouth": round(request.mouth, 3),
        "posture": request.posture,
        "approach": request.approach,
        "docking": request.docking,
        "back_arc": request.back_arc,
    }


def _close_client(client: RobotdClient) -> None:
    for rpc in (client._command, client._state, client._safety):
        try:
            rpc.wire.close()
        except Exception:
            pass


__all__ = [
    "Doors",
    "Lanes",
    "NeuroDriver",
    "NeuroSpec",
    "Telemetry",
    "load_lanes",
]
