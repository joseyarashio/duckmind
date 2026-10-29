"""「神經鴨原型（示範）」: the neural-duck prototype's window onto the simulator page.

The prototype's brain (:mod:`duckmind.neuro`) runs in the simweb driver beside the live physics
body. This module only connects it to the page:

* it serves what the brain reports — its :class:`~duckmind.neuro.contract.BrainTelemetry`,
  turned into the page's plain words and numbers (:class:`DriverSource`, which reads the live
  driver :class:`~duckmind.simweb.neuro_live.NeuroDriver`);
* it hands the page's stimulus buttons to the **world** (a clap, a ball, the lights…) through the
  driver's injection call: the brain hears of them only through its senses, never from here;
* it switches ducks: a new brain from Duck Core's own diversity design (a seed's temperament,
  primary nature, side, habit and motion tempo, drawn by Duck Core itself, and a growth day);
* each duck has its own official voice (voice:neuro 2026-09-28, :mod:`duckmind.simweb.voice`):
  the page's ``/api/voice/…`` routes serve the neural duck on screen (:meth:`NeuroHub.voice_duck`),
  a switched-to duck's bank is rendered at once and the picker's four in the background
  (:meth:`NeuroHub.prepare_voices`), and every card says its voice seed.

Routes (JSON unless said). They always answer — with ``"enabled": false`` when the server runs
without the prototype — so the normal page never meets an error:

* ``GET /api/neuro/status`` — whether the prototype runs, where its telemetry comes from, the
  stimulus buttons, the picker's ducks and the duck now;
* ``GET /api/neuro/telemetry?since=<n>`` — one snapshot (the page's poll); ``since``: the last
  reflex/sound ``seq`` the page has, so each is said once;
* ``GET /api/neuro/stream?since=<n>`` — ``text/event-stream``: a snapshot every
  ``STREAM_EVERY_S``, each with the reflexes and sounds after the last one sent (the page's first
  choice; it falls back to the poll);
* ``POST /api/neuro/stimulus`` ``{"kind": <a STIMULI key or a contract STIMULI key>}``;
* ``POST /api/neuro/duck`` ``{"preset": <id>}`` | ``{"random": true}`` | ``{"seed": n, "gd": d}``;
* ``POST /api/neuro/motion`` ``{"id": <a neuro.motions id>}`` — 「動作試玩」: the duck does that trained motion now
  (the status lists them with their gate verdicts; the snapshot's ``motion`` says what runs and what ran);
* ``POST /api/neuro/feedback`` ``{"kind": "praise" | "treat" | "scold", "trick"?}`` — the owner's 「乖」, a treat
  or 「不對」 for a trick: the brain's proficiency step and Duck Core's word (:meth:`NeuroHub.feedback`).

What the page reads (``PAGE_SCHEMA``; every field optional, levels 0–1 unless said)::

    {"enabled", "schema", "source": "brain" | "stub" | null, "seq" (the newest event number),
     "t_s", "note_zh", "note_tone": "warn" | "info", "waiting_zh",
     "driver": {"state", "state_zh", "error", "lanes", "lanes_zh", "notes", "realtime", "hz"},
     "duck": card_of(),
     "behaviour": {"key", "zh", "why_zh", "since_s", "satiation", "family", "reflex_zh"},
     "candidates": [{"key", "zh", "pull", "activation", "why_zh", "current"}],  # strongest first
     "drives": [{"key", "zh", "value", "main"}],     # the seven main meters first, in DRIVES order
     "arousal": {"value", "zh"}, "valence": −1…1,
     "reflexes": [{"seq", "t_s", "kind", "zh", "strength" (≥ 0; 1 = a full response), "gain",
                   "stimulus", "stimulus_zh"}],     # oldest first; kind "world": a page button
     "vocal": [{"seq", "tag", "zh"}], "asleep": bool,
     "habituation": [{"key", "zh", "level", "count", "sensitised"}],  # level 1 = fresh
     "places": [{"key", "zh", "value" (−1…1), "falls", "avoid", "note_zh"}],
     "rhythm": {"window_s", "bins" (m/s a second), "turns" (rad/s), "share"},
     "locomotion": {"zh", "vx", "vyaw", …}, "senses": [{"key", "zh", "source", "modelled", "note_zh"}],
     "stimuli": {<key>: bool}  # which buttons the source can do now
     "world": {"lights_on"}}

:class:`StubSource` makes up moving numbers so the page can be checked before the brain exists,
and says so on every snapshot. Sockets and threads stay in the simweb driver and the HTTP server;
the brain never imports this module.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import secrets
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from typing import Any, Protocol
from urllib.parse import parse_qs

from duckmind.duckcore import content
from duckmind.i18n import Msg, boundary, content_msg, join, lst, msg
from duckmind.neuro import command_brain as CB
from duckmind.neuro import contract as C
from duckmind.neuro import emotions, motions
from duckmind.neuro.command_words import TRICKS
from duckmind.neuro.commands import CommandError, from_request
from duckmind.simweb import voice

PAGE_SCHEMA = 1
LABEL_ZH = msg("neuro.label")
STREAM_EVERY_S = 0.2
STREAM_MAX_S = 300.0  # a stream ends after this; the page's EventSource reconnects at once
STREAM_MAX = 6  # at most this many open streams (each holds one server thread)
KEEP_EVENTS = 80  # reflexes and sounds kept for the page's poll
WALK_WINDOW_S = 60  # the walking rhythm strip: the last minute, a bar a second
WALK_KEEP = 10 * WALK_WINDOW_S + 50  # (the driver samples at 10 Hz)
MAX_BODY = 4096
GIVEN_NAME_MAX = 12  # the longest name the page's card takes (as for a duck asked for by seed)
NEURO_SKILL_IDS = ("follow", "hide", "circle", "theremin", "ride", "dance", "ball", "carry", "sun")


def _unknown_skills(reason: str = "no_status_data") -> dict[str, dict[str, Any]]:
    """A missing runtime payload is unknown; never manufacture availability or completion."""
    return {
        key: {
            "available": False,
            "reason": reason,
            "phase": None,
            "status": "unknown",
            "result": None,
            "updated_ms": None,
        }
        for key in NEURO_SKILL_IDS
    }


# The page's seven drive meters, in this order (the owner's words), from the brain's drives:
# 睏 is how sleepy it is (the brain's sleepiness: sleep pressure with the body clock), else its
# sleep pressure alone.
DRIVES: tuple[tuple[str, str], ...] = (
    ("energy", msg("neuro.drive.energy")),
    ("sleepiness", msg("neuro.drive.sleepiness")),
    ("curiosity", msg("neuro.drive.curiosity")),
    ("fear", msg("neuro.drive.fear")),
    ("social", msg("neuro.drive.social")),
    ("comfort", msg("neuro.drive.comfort")),
    ("boredom", msg("neuro.drive.boredom")),
)
DRIVE_FALLBACK = {"sleepiness": "sleep_pressure"}
MAIN_DRIVES = {key for key, _ in DRIVES}
MINOR_ZH = {
    "sleep_pressure": msg("neuro.drive.sleep_pressure"),
    "alertness": msg("neuro.drive.alertness"),
    "stiffness": msg("neuro.drive.stiffness"),
}
AROUSAL_ZH = msg("neuro.drive.arousal.title")


@dataclass(frozen=True, slots=True)
class Stimulus:
    key: str
    zh: str
    world_zh: str  # what the world does
    sense_zh: str  # how the duck could notice it (only through this sensor)
    contract: str  # the contract's STIMULI key ("" = none there yet; "lights": off/on by state)


# The stimulus card's eight buttons. Each is a world event; the duck notices it only through its
# senses. ``contract``: the neuro contract's own name for it (the senses lane injects those).
STIMULI: tuple[Stimulus, ...] = (
    Stimulus(
        "clap",
        msg("neuro.stimulus.clap.label"),
        msg("neuro.stimulus.clap.world"),
        msg("neuro.stimulus.clap.sense"),
        "clap",
    ),
    Stimulus(
        "call_name",
        msg("neuro.stimulus.call_name.label"),
        msg("neuro.stimulus.call_name.world"),
        msg("neuro.stimulus.call_name.sense"),
        "call_name",
    ),
    Stimulus(
        "wave",
        msg("neuro.stimulus.wave.label"),
        msg("neuro.stimulus.wave.world"),
        msg("neuro.stimulus.wave.sense"),
        "wave",
    ),
    Stimulus(
        "pat_head",
        msg("neuro.stimulus.pat_head.label"),
        msg("neuro.stimulus.pat_head.world"),
        msg("neuro.stimulus.pat_head.sense"),
        "pet_head",
    ),
    Stimulus(
        "roll_ball",
        msg("neuro.stimulus.roll_ball.label"),
        msg("neuro.stimulus.roll_ball.world"),
        msg("neuro.stimulus.roll_ball.sense"),
        "roll_ball",
    ),
    Stimulus(
        "novel_object",
        msg("neuro.stimulus.novel_object.label"),
        msg("neuro.stimulus.novel_object.world"),
        msg("neuro.stimulus.novel_object.sense"),
        "novel_object",
    ),
    Stimulus(
        "lights",
        msg("neuro.stimulus.lights.label"),
        msg("neuro.stimulus.lights.world"),
        msg("neuro.stimulus.lights.sense"),
        "lights",
    ),
    Stimulus(
        "family_walk",
        msg("neuro.stimulus.family_walk.label"),
        msg("neuro.stimulus.family_walk.world"),
        msg("neuro.stimulus.family_walk.sense"),
        "walk_by",
    ),
)
STIMULUS_BY_KEY = {item.key: item for item in STIMULI}
# Rhythm trains (the driver's TRAINS: the same world sound several times; a duckling freezes
# and hushes at slow knocking, calls and comes at fast clapping — Miller's rule).
TRAIN_ZH = {
    "knock_train": msg("neuro.stimulus.knock_train.label"),
    "clap_train": msg("neuro.stimulus.clap_train.label"),
}
# The contract's other scenario stimuli (the card's 「更多」 row), by their contract names, and
# the trains.
MORE_STIMULI = tuple(
    key
    for key in C.STIMULI
    if key not in {item.contract for item in STIMULI} | {"lights_off", "lights_on"}
) + tuple(TRAIN_ZH)
MORE_ZH = {**{key: spec.zh for key, spec in C.STIMULI.items()}, **TRAIN_ZH}
MORE_CHANNELS = {
    **{key: list(spec.channels) for key, spec in C.STIMULI.items()},
    **{key: ["sound"] for key in TRAIN_ZH},
}
# A contract key for a stimulus that has a page button (both names work in the API).
ALIASES = {item.contract: item.key for item in STIMULI if item.contract not in ("", "lights")}

# The habituation system's stimulus classes, in plain words (contract STIMULUS_CLASSES).
CLASS_ZH: dict[str, str] = {
    "sound:voice": msg("neuro.stimulus_class.sound.voice"),
    "sound:name": msg("neuro.stimulus_class.sound.name"),
    "sound:hello": msg("neuro.stimulus_class.sound.hello"),
    "sound:clap": msg("neuro.stimulus_class.sound.clap"),
    "sound:bang": msg("neuro.stimulus_class.sound.bang"),
    "sound:door": msg("neuro.stimulus_class.sound.door"),
    "sound:footsteps": msg("neuro.stimulus_class.sound.footsteps"),
    "sound:music": msg("neuro.stimulus_class.sound.music"),
    "sound:tv": msg("neuro.stimulus_class.sound.tv"),
    "sound:duck_call": msg("neuro.stimulus_class.sound.duck_call"),
    "sound:noise": msg("neuro.stimulus_class.sound.noise"),
    "touch:head": msg("neuro.stimulus_class.touch.head"),
    "touch:back": msg("neuro.stimulus_class.touch.back"),
    "touch:left": msg("neuro.stimulus_class.touch.left"),
    "touch:right": msg("neuro.stimulus_class.touch.right"),
    "touch:tail": msg("neuro.stimulus_class.touch.tail"),
    "touch:beak": msg("neuro.stimulus_class.touch.beak"),
    "visual:motion": msg("neuro.stimulus_class.visual.motion"),
    "visual:looming": msg("neuro.stimulus_class.visual.looming"),
    "visual:thing": msg("neuro.stimulus_class.visual.thing"),
    "visual:person": msg("neuro.stimulus_class.visual.person"),
    "visual:hand": msg("neuro.stimulus_class.visual.hand"),
    "light:on": msg("neuro.stimulus_class.light.on"),
    "light:off": msg("neuro.stimulus_class.light.off"),
    "body:bump": msg("neuro.stimulus_class.body.bump"),
    "body:lifted": msg("neuro.stimulus_class.body.lifted"),
    "body:brushed": msg("neuro.stimulus_class.body.brushed"),
    "body:stumbled": msg("neuro.stimulus_class.body.stumbled"),
    "body:fell": msg("neuro.stimulus_class.body.fell"),
    "body:tilt": msg("neuro.stimulus_class.body.tilt"),
    "body:set_down": msg("neuro.stimulus_class.body.set_down"),
}
SENSE_ZH = {
    "sound": msg("neuro.sense.sound"),
    "touch": msg("neuro.sense.touch"),
    "visual": msg("neuro.sense.visual"),
    "light": msg("neuro.sense.light"),
    "body": msg("neuro.sense.body"),
    "habit": msg("neuro.sense.habit"),
    "world": msg("neuro.sense.world"),
}
# The senses the habituation list shows (the page's stimuli reach these; its own habits do not).
OUTER_SENSES = ("sound", "touch", "visual", "light", "body")
HABIT_ROWS = 12
# The classes the card's buttons make (shown first, even before they happened).
BUTTON_CLASSES = (
    "sound:clap",
    "sound:name",
    "visual:motion",
    "touch:head",
    "visual:thing",
    "light:off",
    "visual:person",
)
CHANNEL_ZH = {
    "body": msg("neuro.channel.body"),
    "proprio": msg("neuro.channel.proprio"),
    "tof": msg("neuro.channel.tof"),
    "camera": msg("neuro.channel.camera"),
    "sound": msg("neuro.channel.sound"),
    "touch": msg("neuro.channel.touch"),
    "power": msg("neuro.channel.power"),
    "clock": msg("neuro.channel.clock"),
}
VOCAL_ZH = {**content.TAG_ZH, "alarm": msg("neuro.web.neuro_api.VOCAL_ZH.alarm")}

# The temperament axes as the page names them (Duck Core's TEMPERAMENT axes, spec v4 7.2),
# with what a high and a low value mean in plain words.
AXES_ZH: dict[str, tuple[str, str, str]] = {
    "boldness": (
        msg("neuro.axis.boldness.name"),
        msg("neuro.axis.boldness.high"),
        msg("neuro.axis.boldness.low"),
    ),
    "activity": (
        msg("neuro.axis.activity.name"),
        msg("neuro.axis.activity.high"),
        msg("neuro.axis.activity.low"),
    ),
    "sociability": (
        msg("neuro.axis.sociability.name"),
        msg("neuro.axis.sociability.high"),
        msg("neuro.axis.sociability.low"),
    ),
    "exploration": (
        msg("neuro.axis.exploration.name"),
        msg("neuro.axis.exploration.high"),
        msg("neuro.axis.exploration.low"),
    ),
    "wit": (msg("neuro.axis.wit.name"), msg("neuro.axis.wit.high"), msg("neuro.axis.wit.low")),
    "chronotype": (
        msg("neuro.axis.chronotype.name"),
        msg("neuro.axis.chronotype.high"),
        msg("neuro.axis.chronotype.low"),
    ),
    "tempo": (
        msg("neuro.axis.tempo.name"),
        msg("neuro.axis.tempo.high"),
        msg("neuro.axis.tempo.low"),
    ),
}
TEMPO_ZH = (
    msg("neuro.axis.motion_tempo.fast"),
    msg("neuro.axis.motion_tempo.slow"),
)  # the quirk tempo (0.85–1.15×: how fast its moves play)
NATURE_ZH = {
    branch.branch_id: content_msg(f"core.content.BRANCHES.{branch.branch_id}_zh", branch.zh)
    for branch in content.BRANCHES
}
STAGE_ZH = {
    stage.stage_id: content_msg(f"core.content.STAGES.{stage.stage_id}_zh", stage.zh)
    for stage in content.STAGES
}


# ---------------------------------------------------------------------------- the picker


@dataclass(frozen=True, slots=True)
class DuckChoice:
    """One duck for the prototype: Duck Core's seed (its temperament, nature, side, habit and
    motion tempo, drawn from the seed alone) and its growth day (its stage, by its own stage
    days; from the juvenile stage on, its branch by Duck Core's own lottery)."""

    seed: int
    gd: int
    name: str
    preset: str = ""  # a PRESETS id, or "" (random, or asked by seed)
    title_zh: str = ""

    @property
    def spec(self) -> str:
        """The driver's ``--neuro-duck`` for this duck (``seed:gd``)."""
        return f"{self.seed}:{self.gd}"


# Four contrasting ducks, found by drawing Duck Core's own hatchings over 6,000 seeds (the tests
# check each title against its genome): a bold explorer, a shy snuggler, a slow night-owl
# dreamer and a lively hatchling — four natures, four stages, both sides.
PRESETS: tuple[DuckChoice, ...] = (
    DuckChoice(4107, 60, "阿勇", "explorer", msg("neuro.preset.explorer.title")),
    DuckChoice(1728, 20, "怯怯", "snuggler", msg("neuro.preset.snuggler.title")),
    DuckChoice(4670, 200, "慢慢", "dreamer", msg("neuro.preset.dreamer.title")),
    DuckChoice(3341, 7, "啾啾", "hatchling", msg("neuro.preset.hatchling.title")),
)
PRESET_BY_ID = {item.preset: item for item in PRESETS}
DEFAULT_PRESET = "explorer"
LIFE_PRESET = "life"  # the picker's id for the server's own life duck (not one of PRESETS)


# The life duck's own genome summary (docs/design/GROWTH-G1): built from Duck Core's state, so its
# card says what the brain that runs is, whatever the family's care made of it.
_LIFE_SUMMARIES: dict[tuple[int, int], dict[str, Any]] = {}


def register_life_summary(seed: int, gd: int, summary: Mapping[str, Any]) -> None:
    _LIFE_SUMMARIES[(int(seed), int(gd))] = dict(summary)


def genome_summary(seed: int, gd: int) -> dict[str, Any]:
    """The duck the prototype's brain gets for (seed, gd): :meth:`Genome.from_seed` (Duck Core's
    own hatching draws, its stage by its own stage days, its branch by Duck Core's lottery); the
    life duck's own genome when the page asks for that seed at that growth day."""
    from duckmind.neuro.genome import Genome

    known = _LIFE_SUMMARIES.get((int(seed), int(gd)))
    if known is not None:
        return dict(known)
    return Genome.from_seed(int(seed), gd=int(gd)).summary()


def traits_zh(summary: Mapping[str, Any], limit: int = 4) -> list[str]:
    """Its most marked traits in plain words (the temperament axes furthest from the middle,
    and its motion tempo), strongest first."""
    scored: list[tuple[float, str]] = []
    for axis, value in (summary.get("axes") or {}).items():
        if axis == "tempo" or axis not in AXES_ZH:
            continue  # (how fast it grows up is not something one sees in a minute)
        away = 2.0 * float(value) - 1.0
        if abs(away) >= 0.33:
            _, high, low = AXES_ZH[axis]
            scored.append((abs(away), high if away > 0 else low))
    tempo = float(summary.get("quirk_tempo", 1.0))
    if abs(tempo - 1.0) >= 0.07:
        scored.append((abs(tempo - 1.0) / 0.15, TEMPO_ZH[0] if tempo > 1.0 else TEMPO_ZH[1]))
    scored.sort(key=lambda item: -item[0])
    return [text for _, text in scored[:limit]]


def card_of(summary: Mapping[str, Any], choice: DuckChoice | None = None) -> dict[str, Any]:
    """The page's card for a duck from its genome summary: name, one plain line, temperament
    bars, stage, branch or nature, side, habit. Everything shown comes from the genome, so the
    words match the brain that runs."""
    seed = int(summary.get("seed", -1))
    gd = int(summary.get("gd", 0))
    if choice is None or choice.seed != seed or choice.gd != gd:
        known = next((p for p in PRESETS if p.seed == seed and p.gd == gd), None)
        choice = known or DuckChoice(
            seed, gd, msg("neuro.duck.default_name", number=seed % 1000), "", ""
        )
    stage = str(summary.get("stage", "hatchling"))
    # (the words stay catalog messages, so the page shows them in the viewer's language)
    stage_zh = summary.get("stage_zh") or STAGE_ZH.get(stage, stage)
    primary = str(summary.get("primary", ""))
    nature = summary.get("primary_zh") or NATURE_ZH.get(primary, primary)
    branch = summary.get("branch") or ""
    branch_zh = (summary.get("branch_zh") or NATURE_ZH.get(str(branch), "")) if branch else ""
    kind = (
        msg("neuro.duck.line.branch_stage", branch=branch_zh, stage=stage_zh)
        if branch_zh
        else msg("neuro.duck.line.kind_nature", stage=stage_zh, nature=nature)
    )
    side = summary.get("side", "")
    traits = traits_zh(summary)
    line = (
        msg("neuro.duck.line.traits", traits=lst(traits), kind=kind)
        if traits
        else msg("neuro.duck.line.even", kind=kind)
    )
    if side:
        line = join(line, msg("neuro.duck.line.side", side=side))
    axes_now = summary.get("axes") or {}
    axes = [
        {"key": axis, "zh": AXES_ZH[axis][0], "value": round(float(axes_now.get(axis, 0.5)), 3)}
        for axis in content.TEMPERAMENT["axes"]
        if axis in AXES_ZH
    ]
    return {
        "name": choice.name,
        "preset": choice.preset,
        "title_zh": choice.title_zh,
        "seed": seed,
        "gd": gd,
        "spec": f"{seed}:{gd}",
        "stage": stage,
        "stage_zh": stage_zh,
        "nature": primary,
        "nature_zh": nature,
        "branch": str(branch),
        "branch_zh": branch_zh,
        "side_zh": side,
        "habit_zh": summary.get("habit_zh", ""),
        "motion_tempo": round(float(summary.get("quirk_tempo", 1.0)), 3),
        "line_zh": line,
        "axes": axes,
        # its own official voice, from its seed for life (voice:neuro; simweb/voice.py)
        "voice_seed": voice.voice_seed(seed) if seed >= 0 else None,
    }


def duck_card(choice: DuckChoice) -> dict[str, Any]:
    """The page's card for a chosen duck (see :func:`card_of`)."""
    return card_of(genome_summary(choice.seed, choice.gd), choice)


_PRESET_CARDS: list[dict[str, Any]] = []


def preset_cards() -> list[dict[str, Any]]:
    """The picker's four ducks (built once)."""
    if not _PRESET_CARDS:
        _PRESET_CARDS.extend(duck_card(item) for item in PRESETS)
    return list(_PRESET_CARDS)


def random_choice(pick: Callable[[int], int] = secrets.randbelow) -> DuckChoice:
    """A duck from a random seed at a random age (any stage, by its own stage days)."""
    seed = pick(1_000_000)
    days = genome_summary(seed, 0).get("stage_days") or {}
    if not days:
        from duckmind.duckcore.engine import DuckCore

        days = dict(DuckCore(seed=seed).s.stage_days)
    bounds = {
        "hatchling": (1, int(days["duckling"]) - 1),
        "duckling": (int(days["duckling"]), int(days["juvenile"]) - 1),
        "juvenile": (int(days["juvenile"]), int(days["adult"]) - 1),
        "adult": (int(days["adult"]), int(days["adult"]) + 120),
    }
    low, high = bounds[content.STAGES[pick(len(content.STAGES))].stage_id]
    gd = low + pick(max(1, high - low + 1))
    return DuckChoice(
        seed,
        gd,
        msg("neuro.duck.default_name", number=seed % 1000),
        "",
        msg("neuro.duck.title.random"),
    )


def choice_from(params: Mapping[str, Any]) -> DuckChoice:
    """The duck a request asks for: a preset, a random one, or a seed and a growth day."""
    if params.get("preset"):
        preset = PRESET_BY_ID.get(str(params["preset"]))
        if preset is None:
            raise ValueError(msg("neuro.error.choice_from.a82120", given_id=params["preset"]))
        return preset
    if params.get("random"):
        return random_choice()
    if "seed" in params:
        seed = int(params["seed"])
        if not 0 <= seed < 10_000_000:
            raise ValueError(msg("neuro.error.choice_from.8b788d"))
        gd = max(0, min(400, int(params.get("gd", 60))))
        name = str(params.get("name") or msg("neuro.duck.default_name", number=seed % 1000))[:12]
        return DuckChoice(seed, gd, name, "", msg("neuro.duck.title.seed"))
    raise ValueError(msg("neuro.error.choice_from.ba3e07"))


def choice_of_spec(text: str, seed: int, gd: int = 60) -> DuckChoice:
    """The duck a ``--neuro-duck`` names: a preset id, ``seed`` or ``seed:gd``; empty: ``seed``
    at ``gd`` (the driver's own default)."""
    text = (text or "").strip()
    if text in PRESET_BY_ID:
        return PRESET_BY_ID[text]
    head, _, tail = text.partition(":")
    if head.isdigit():
        seed = int(head)
        gd = int(tail) if tail.strip().isdigit() else gd
    known = next((p for p in PRESETS if p.seed == seed and p.gd == gd), None)
    return known or DuckChoice(seed, gd, msg("neuro.duck.default_name", number=seed % 1000), "", "")


# ---------------------------------------------------------------------------- sources


class NeuroSource(Protocol):
    """What the page needs from the prototype's driver (or the stub). ``snapshot`` answers in
    the page's schema (see the module docstring); ``stimulus`` does one stimulus in the world
    (``kind``: a STIMULI key, or a contract STIMULI key for the 「更多」 row); ``switch`` starts
    another duck. Called from server threads: thread-safe."""

    kind: str  # "brain" | "stub"

    def snapshot(self, since: int = 0) -> dict[str, Any]: ...

    def stimulus(self, kind: str, params: Mapping[str, Any]) -> dict[str, Any]: ...

    def switch(self, choice: DuckChoice, card: Mapping[str, Any]) -> dict[str, Any]: ...

    def set_day(self, minute: float) -> dict[str, Any]: ...

    def help_up(self) -> dict[str, Any]: ...

    def play_motion(self, motion_id: str) -> dict[str, Any]: ...

    def remote_set(self, on: bool, **kw: Any) -> dict[str, Any]: ...

    def remote_twist(
        self, vx: Any, vyaw: Any, vy: Any = 0.0, key_to_recv_ms: float | None = None
    ) -> dict[str, Any]: ...

    def remote_owned_by(self, owner: object) -> bool: ...

    def go_there(self, x_m: Any, y_m: Any, who: Any = None) -> dict[str, Any]: ...

    def feedback(self, kind: str, trick: str | None) -> dict[str, Any]: ...

    def stop(self) -> None: ...


# ---------------------------------------------------------------------------- the owner's word on a trick

FEEDBACK_KINDS = ("praise", "treat", "scold")


class FeedbackError(ValueError):
    """A feedback the duck cannot take (``POST /api/neuro/feedback`` answers 400): ``reason`` is the
    catalog key suffix ``neuro.feedback.error.<reason>`` (``bad_kind``, ``bad_trick``, ``no_trick``,
    ``not_running``)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def feedback_note_key(kind: str, answer: Mapping[str, Any]) -> str:
    """What the page says of the brain's answer (``Brain.trick_feedback``), as a catalog key."""
    if kind == "scold":
        return "neuro.feedback.note.scolded"
    if not answer.get("counted", True):
        return "neuro.feedback.note.not_counted"
    if answer.get("capped"):
        return "neuro.feedback.note.capped"
    if answer.get("shaped"):
        return "neuro.feedback.note.shaped"
    return "neuro.feedback.note.praised"


# The page's plain time-of-day choice (the driver's DAY_CHOICES; the duck's own clock).
DAY_CHOICES: tuple[tuple[str, float], ...] = (
    (msg("neuro.day.early_morning"), 7 * 60 + 30.0),
    (msg("neuro.day.morning"), 10 * 60.0),
    (msg("neuro.day.afternoon"), 15 * 60.0),
    (msg("neuro.day.evening"), 18 * 60 + 30.0),
    (msg("neuro.day.night"), 21 * 60 + 30.0),
    (msg("neuro.day.midnight"), 60.0),
)
# The reflex holds the page times (the samples' behaviour while a reflex holds the body): what
# the line says when the hold ends, e.g. 「愣住 0.8 秒」.
HOLD_WORDS = {
    "startle": msg("neuro.body.hold.startle"),
    "freeze": msg("neuro.body.hold.freeze"),
    "withdraw": msg("neuro.body.hold.withdraw"),
    "still": msg("neuro.body.hold.still"),
}
MOUTH_WINDOW_MS = 1500  # the beak's recent openings the page replays (10 Hz peaks)


def _minute_zh(minute: float) -> str:
    try:
        from duckmind.simweb.neuro_live import minute_zh
    except ImportError:  # (the driver module is optional for the page)
        return f"{int(minute) // 60:02d}:{int(minute) % 60:02d}"
    return minute_zh(minute)


def senses_zh(sources: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    """Where each of the duck's senses comes from (the frame's own ``sources``, else the
    contract's defaults): a real sensor door, the duck-eye pipeline, or a model the prototype
    uses until the real sensor or detector exists — each marked."""
    got = dict(C.CHANNEL_SOURCE)
    got.update(sources or {})
    out = []
    for key, zh in CHANNEL_ZH.items():
        source = str(got.get(key, "none"))
        out.append(
            {
                "key": key,
                "zh": zh,
                "source": source,
                "modelled": source == "model",
                "note_zh": C.SOURCE_ZH.get(source, source),
            }
        )
    return out


def class_zh(stimulus: str) -> str:
    """A stimulus class ("sound:clap") in plain words — never the program's own name: a class
    the page does not know is said by its sense."""
    if stimulus in CLASS_ZH:
        return CLASS_ZH[stimulus]
    sense = stimulus.split(":", 1)[0]
    return SENSE_ZH.get(sense, msg("neuro.sense.other"))


THING_ZH = {
    "thing": msg("neuro.thing.thing"),
    "toy": msg("neuro.thing.toy"),
    "person": msg("neuro.thing.person"),
    "hand": msg("neuro.thing.hand"),
    "duck": msg("neuro.thing.duck"),
    "nest": msg("neuro.thing.nest"),
}


def _metres(x: float, y: float) -> str:
    """An offset in the duck's own odometry frame (x ahead of how it faced at the start, y to
    its left) in plain words."""
    ahead = msg("neuro.place.metres.61df", x=x) if x >= 0 else msg("neuro.place.metres.40c6", x=-x)
    side = msg("neuro.place.metres.8862", y=y) if y >= 0 else msg("neuro.place.metres.a789", y=-y)
    return msg("neuro.place.metres.3b69", ahead=ahead, side=side)


def place_zh(item: Mapping[str, Any], cell_m: float = 0.5, here: Any = None) -> str:
    """A learned place in the duck's own words: where it is from where it started (its
    odometry — the duck knows no rooms), unless the brain named it."""
    for key in ("zh", "name_zh", "label"):
        if item.get(key):
            return str(item[key])
    cell = item.get("cell")
    if isinstance(cell, (list, tuple)) and len(cell) == 2:
        if here is not None and list(here) == list(cell):
            return msg("neuro.place.place_zh.55c5")
        x = (float(cell[0]) + 0.5) * cell_m
        y = (float(cell[1]) + 0.5) * cell_m
        return msg("neuro.place.place_zh.f039", offset=_metres(x, y))
    px, py = item.get("x"), item.get("y")
    if isinstance(px, (int, float)) and isinstance(py, (int, float)):
        return msg("neuro.place.place_zh.f039", offset=_metres(float(px), float(py)))
    return str(item.get("key") or item.get("id") or msg("neuro.place.place_zh.4358"))


def places_of(learned: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The page's learned places from the brain's ``learned`` (``learn.view``: cells of its own
    odometry with a value, time spent, the family's share, falls and whether it avoids it —
    three hard falls or more, the owner's rule), its favourite spot now and the things it
    learned about. Tolerant of other shapes (a list or a mapping of places)."""
    raw = learned.get("places", ())
    cell_m = float(learned.get("cell_m") or 0.5)
    here = learned.get("here")
    items: list[Mapping[str, Any]] = []
    if isinstance(raw, Mapping):
        items = [
            {"key": key, **value} if isinstance(value, Mapping) else {"key": key, "value": value}
            for key, value in raw.items()
        ]
    elif isinstance(raw, Iterable):
        items = [item for item in raw if isinstance(item, Mapping)]
    out: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        value = item.get("value", item.get("valence", 0.0))
        falls = item.get("hard_falls", item.get("falls", item.get("fall_count", 0)))
        avoid = bool(item.get("avoid", item.get("avoided", False)))
        cell = item.get("cell")
        key = f"c{cell[0]},{cell[1]}" if isinstance(cell, (list, tuple)) and len(cell) == 2 else ""
        notes = []
        if float(item.get("family") or 0.0) >= 0.3:
            notes.append(msg("neuro.place.places_of.6c79"))
        if float(item.get("visits_s") or 0.0) >= 1.0:
            notes.append(msg("neuro.place.places_of.9ecf", visits_s=float(item["visits_s"])))
        out.append(
            {
                "key": key or str(item.get("key") or item.get("id") or index),
                "zh": place_zh(item, cell_m, here),
                "value": round(max(-1.0, min(1.0, float(value or 0.0))), 3),
                "falls": int(falls or 0),
                "avoid": avoid,
                "note_zh": lst(notes, "fmt.sep.dot"),
            }
        )
    out.sort(key=lambda place: (not place["avoid"], -abs(place["value"]), -place["falls"]))
    out = out[:7]
    liked = learned.get("liked")
    if isinstance(liked, Mapping) and liked.get("range_m") is not None:
        bearing = float(liked.get("bearing") or 0.0)
        where = (
            msg("neuro.place.places_of.0c3e")
            if abs(bearing) < 0.4
            else (
                msg("neuro.place.places_of.82fb")
                if 0 < bearing < 1.9
                else msg("neuro.place.places_of.70cb")
                if -1.9 < bearing < 0
                else msg("neuro.place.places_of.a000")
            )
        )
        out.insert(
            0,
            {
                "key": "liked",
                "zh": msg(
                    "neuro.place.places_of.25f4", side=where, range_m=float(liked["range_m"])
                ),
                "value": round(max(-1.0, min(1.0, float(liked.get("score") or 0.0))), 3),
                "falls": 0,
                "avoid": False,
                "note_zh": "",
            },
        )
    for thing in list(learned.get("things") or ())[:3]:
        if not isinstance(thing, Mapping) or abs(float(thing.get("value") or 0.0)) < 0.05:
            continue
        kind = str(thing.get("kind") or "thing")
        out.append(
            {
                "key": f"t{thing.get('track')}",
                "zh": msg(
                    "neuro.place.places_of.6567",
                    thing=THING_ZH.get(kind, kind),
                    track=thing.get("track"),
                ),
                "value": round(max(-1.0, min(1.0, float(thing.get("value") or 0.0))), 3),
                "falls": 0,
                "avoid": False,
                "note_zh": msg(
                    "neuro.place.places_of.cdd6", seen_s=float(thing.get("seen_s") or 0.0)
                ),
            }
        )
    return out[:9]


def locomotion_zh(
    vx: float, vyaw: float, asleep: bool = False, true_mps: float | None = None
) -> str:
    """How it moves right now, in words. (fid:fix 2026-09-28: the brain's speed is a command —
    the body walks about half as fast (A3: 0.40 → 0.15 m/s, 0.55 → 0.28; live 154 readings:
    the command 0.37 median, the body 0.18) — so the command is said as the brain's wish and the
    body's own speed beside it, measured: 「腦想走 0.37（身體實際 0.18 公尺/秒）」.)"""
    if asleep:
        return msg("neuro.loco.1de4")
    if vx <= 0.0 and abs(vx) < 1e-3:
        return msg("neuro.loco.972f")
    body = msg("neuro.loco.802b", true_mps=true_mps) if true_mps is not None else ""
    if vx < 0:
        return msg("neuro.loco.2bf8", vx=-vx, body_speed=body)
    turn = (
        msg("neuro.loco.ebc2")
        if abs(vyaw) < 0.15
        else (msg("neuro.loco.7f8d") if vyaw > 0 else msg("neuro.loco.fad8"))
    )
    return msg("neuro.loco.9f06", vx=vx, body_speed=body, turn=turn)


LANE_ZH = {
    "brain": msg("neuro.lane.brain"),
    "senses": msg("neuro.lane.senses"),
    "motor": msg("neuro.lane.motor"),
}
DRIVER_STATE_ZH = {
    "waiting": msg("neuro.driver_state.waiting"),
    "attaching": msg("neuro.driver_state.attaching"),
    "running": msg("neuro.driver_state.running"),
    "detached": msg("neuro.driver_state.detached"),
    "stopped": msg("neuro.driver_state.stopped"),
    "error": msg("neuro.driver_state.error"),
}
BODY_EVENT_ZH = {
    "proprio.blocked": msg("neuro.body_event.blocked"),
    # (fid:fix 2026-09-28) 「絆了一下」, 「被抱起來了」 and 「被放下來了」 are said only when the
    # body shows them — the driver's witness confirms them on the physics truth (its "body"
    # items): the review found 43 of 44 「絆了一下」 at a sit-down or a rise and 「被抱起來了」 for
    # a push on a seated duck that moved 0.6 cm. 「跌倒了」 too, once per fall (live: one push
    # said it three times — proprio, duck-body and this list).
}


class DriverSource:
    """The live driver's telemetry (:meth:`duckmind.simweb.neuro_live.NeuroDriver.telemetry`) as
    the page reads it. Every snapshot pulls what is new from the driver (its own cursor), keeps
    one list of reflexes, calls and body events with the page's own numbers (so several pages
    and streams each get every event once), and turns the latest brain telemetry, sensor frame
    and motor command into the page's words and bars. ``driver_of`` finds the driver (it is set
    up after the server; None until then); ``inject`` does a stimulus in the world."""

    kind = "brain"

    def __init__(
        self,
        driver_of: Callable[[], Any],
        inject: Callable[[str, Mapping[str, Any]], Mapping[str, Any] | None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._driver_of = driver_of
        self._inject = inject
        self._clock = clock
        self._lock = threading.Lock()
        self._driver: Any = None
        self._fresh()

    def _fresh(self) -> None:
        self._cursor = 0
        self._latest: dict[str, Any] = {}
        self._pulled_at = -1.0
        self._event_seq = 0
        self._events: deque[dict[str, Any]] = deque(maxlen=KEEP_EVENTS)
        self._vocal: deque[dict[str, Any]] = deque(maxlen=KEEP_EVENTS)
        self._seen: deque[tuple[Any, ...]] = deque(maxlen=4 * KEEP_EVENTS)
        self._class_counts: dict[str, int] = {}
        self._walk: deque[tuple[int, float, float]] = deque(maxlen=WALK_KEEP)
        self._mouth: deque[tuple[int, float]] = deque(maxlen=40)
        self._last_sample: dict[str, Any] = {}
        self._genome_key: Any = None
        self._card: dict[str, Any] | None = None
        self._lights_on = True
        # fid:page: what the body measurably did after each reflex (the driver's witness),
        # joined to the reflex's row: row_of maps a brain reflex (t_ms, reflex, stimulus) to its
        # page row's seq; body_rows are the page's items {"seq", "of" (the row), "zh", …}
        self._row_of: dict[tuple[Any, ...], int] = {}
        self._body_rows: deque[dict[str, Any]] = deque(maxlen=KEEP_EVENTS)
        self._audit_cursor = 0
        self._body_now: dict[str, Any] | None = None
        # (fid:fix) the moments the body was seen to perform (their numbers), newest last
        self._moments_seen: deque[int] = deque(maxlen=40)

    # -- the driver

    def driver(self) -> Any:
        return self._driver_of()

    def _pull(self) -> None:
        """(Under the lock.) What is new at the driver since the last pull."""
        driver = self.driver()
        if driver is not self._driver:
            lights = self._lights_on
            self._fresh()
            self._lights_on = lights
            self._driver = driver
        if driver is None:
            return
        now = self._clock()
        if now - self._pulled_at < 0.05:
            return  # (two pages asking in the same moment: one pull)
        self._pulled_at = now
        data = dict(driver.telemetry(self._cursor))
        seq = int(data.get("seq") or 0)
        if seq < self._cursor:  # the driver started over (a new duck): read it all again
            data = dict(driver.telemetry(0))
            seq = int(data.get("seq") or 0)
        self._cursor = seq
        self._latest = data
        for sample in data.get("samples") or ():
            self._last_sample = dict(sample)
            moved = sample.get("applied") or sample.get("wanted") or (0.0, 0.0)
            try:
                t_ms = int(sample["t_ms"])
                # (fid:fix) the walking strip shows the BODY: its real speed and turning (the
                # physics truth), not what robotd was told
                true = sample.get("speed_true")
                turn = sample.get("turn_true")
                speed = float(true) if isinstance(true, (int, float)) else float(moved[0])
                yaw_rate = float(turn) if isinstance(turn, (int, float)) else float(moved[1])
                self._walk.append((t_ms, speed, yaw_rate))
                self._mouth.append(
                    (t_ms, float(sample.get("mouth_peak", sample.get("mouth", 0.0))))
                )
            except (KeyError, TypeError, ValueError, IndexError):
                continue
        for item in data.get("vocals") or ():
            self._event_seq += 1
            tag = str(item.get("tag", ""))
            vocal: dict[str, Any] = {
                "seq": self._event_seq,
                "tag": tag,
                "zh": item.get("zh") or VOCAL_ZH.get(tag, tag),
            }
            # (how loud, and when on the server's clock: the page voices it with the body)
            if isinstance(item.get("level"), (int, float)):
                vocal["level"] = round(float(item["level"]), 3)
            if isinstance(item.get("wall_ms"), (int, float)):
                vocal["wall_ms"] = int(item["wall_ms"])
            # (voice:neuro) on the driver's clock, as the beak's ``mouth`` track: the page sounds
            # it with the beak that opened for it when it has no body samples to carry it
            if isinstance(item.get("t_ms"), (int, float)):
                vocal["t_ms"] = int(item["t_ms"])
            self._vocal.append(vocal)
        for item in data.get("events") or ():
            kind = str(item.get("kind", ""))
            said = BODY_EVENT_ZH.get(kind)
            if kind == "body.righting":
                # (fid:fix) 「自己爬起來了」 is the brain's own row (its body line says when the
                # body was up); only a failure is said here
                said = None if item.get("ok") else msg("neuro.row.righting_failed")
            if kind == "world.stimulus":
                # What the page's button did in the world, just before the duck's own reaction.
                self._add_event(
                    "world",
                    msg("neuro.row.world", stimulus=item.get("zh") or item.get("stimulus")),
                    0.0,
                    item.get("t_ms"),
                )
            elif kind in ("emotion.start", "emotion.end") and item.get("zh"):
                # an emotion (the page's press or the duck's own): one row when it starts, one when it ends
                self._add_event("emotion", item["zh"], 0.0, item.get("t_ms"), stimulus="")
            elif said:
                self._add_event("body", said, 1.0, item.get("t_ms"), stimulus="")
        brain = data.get("brain") or {}
        for event in brain.get("reflexes") or ():
            key = (event.get("t_ms"), event.get("reflex"), event.get("stimulus"))
            if key in self._seen:
                continue
            self._seen.append(key)
            stimulus = str(event.get("stimulus") or "")
            if stimulus:
                self._class_counts[stimulus] = self._class_counts.get(stimulus, 0) + 1
            reflex = str(event.get("reflex") or "")
            name = C.REFLEXES.get(reflex, reflex)
            text = event.get("zh") or name
            said = class_zh(stimulus) if stimulus else ""
            # what set it off, then what it did (the brain's words say the side: 「頭轉向右邊」)
            if said and said not in text and not stimulus.startswith(("body:", "habit:")):
                text = msg("neuro.row.reflex_after_class", class_word=said, reflex_words=text)
            elif said and text == name and said not in text:
                text = msg("neuro.row.reflex_with_class", reflex_words=text, class_word=said)
            self._add_event(
                reflex,
                text,
                event.get("strength", 0.0),
                event.get("t_ms"),
                stimulus=stimulus,
                gain=event.get("gain"),
            )
            self._row_of[key] = self._event_seq
        if len(self._row_of) > 4 * KEEP_EVENTS:
            for stale in list(self._row_of)[: len(self._row_of) - 2 * KEEP_EVENTS]:
                del self._row_of[stale]
        self._take_witness(data.get("witness"))

    def _take_witness(self, witness: Any) -> None:
        """(Under the lock.) fid:page: each closed audit of the driver's witness onto its reflex
        row (「身體：頭往後縮 18°、停下來」 / 「身體做不到：…」), and what the body does now."""
        if not isinstance(witness, Mapping):
            return
        if int(witness.get("seq") or 0) < self._audit_cursor:
            self._audit_cursor = 0  # (a new driver: its audits count from the start)
        for audit in witness.get("audits") or ():
            if not isinstance(audit, Mapping):
                continue
            number = int(audit.get("seq") or 0)
            if number <= self._audit_cursor:
                continue
            self._audit_cursor = number
            kind = str(audit.get("kind") or "reflex")
            if kind == "hold":
                # (fid:fix) 「愣住 X 秒」 timed from the body: how long the head really held still
                self._add_event(
                    "freeze" if audit.get("act") in ("startle", "freeze") else "hold",
                    audit.get("zh") or "",
                    0.0,
                    audit.get("t_ms"),
                )
                continue
            if kind == "moment":
                if audit.get("ok"):
                    self._moments_seen.append(int(audit.get("moment_seq") or 0))
                continue
            if kind == "body":
                self._add_event("body", audit.get("zh") or "", 1.0, audit.get("t_ms"))
                continue
            row = self._row_of.get(tuple(audit.get("key") or ()))
            if row is None or not audit.get("zh"):
                continue
            self._event_seq += 1
            self._body_rows.append(
                {
                    "seq": self._event_seq,
                    "of": row,
                    "zh": audit["zh"],
                    "verdict": audit.get("verdict"),
                    "reflex": audit.get("reflex"),
                    "head": audit.get("head"),
                    "legs": audit.get("legs"),
                }
            )
        now = witness.get("now")
        self._body_now = dict(now) if isinstance(now, Mapping) else None

    def _add_event(
        self,
        kind: str,
        text: str,
        strength: Any,
        t_ms: Any,
        stimulus: str = "",
        gain: Any = None,
    ) -> None:
        self._event_seq += 1
        item: dict[str, Any] = {
            "seq": self._event_seq,
            "t_s": round(float(t_ms or 0) / 1000, 2),
            "kind": kind,
            "zh": text,
            # (the brain's own size of the response: 1 = a full one; above 1 when it is
            # sensitised or aroused, falling below as it gets used to a stimulus)
            "strength": _size(strength),
        }
        if stimulus:
            item["stimulus"] = stimulus
            item["stimulus_zh"] = class_zh(stimulus)
        if gain is not None:
            item["gain"] = _unit(gain)
        self._events.append(item)

    # -- the page

    def snapshot(self, since: int = 0) -> dict[str, Any]:
        with self._lock:
            self._pull()
            data = self._latest
            out: dict[str, Any] = {
                "enabled": True,
                "schema": PAGE_SCHEMA,
                "source": self.kind,
                "seq": self._event_seq,
                "reflexes": [item for item in self._events if item["seq"] > since],
                "vocal": [item for item in self._vocal if item["seq"] > since],
                "skills": _unknown_skills(),
                "actions": [],
                # fid:page: what the body did after each reflex, onto its row ("of")
                "reflex_body": [item for item in self._body_rows if item["seq"] > since],
                "stimuli": self._buttons(),
                "world": {"lights_on": self._lights_on},
            }
            if self._driver is None:
                out["waiting_zh"] = msg("neuro.waiting.b65b")
                return out
            out.update(self._status(data))
            supplied_skills = data.get("skills")
            if isinstance(supplied_skills, Mapping):
                out["skills"] = dict(supplied_skills)
            if isinstance(data.get("motion"), Mapping):
                out["motion"] = dict(data["motion"])  # 「動作試玩」: what runs, what ran last
            if isinstance(data.get("remote"), Mapping):
                out["remote"] = dict(data["remote"])  # remote mode: on/off, twist, edge latencies
            if isinstance(data.get("errand"), Mapping):
                out["errand"] = dict(data["errand"])  # go there: idle | waiting | walking | arrived | gave_up
            actions = data.get("actions")
            if isinstance(actions, list):
                out["actions"] = actions
            brain = data.get("brain")
            if not brain:
                out.setdefault("waiting_zh", msg("neuro.waiting.333e"))
                out["senses"] = senses_zh(None)
                return out
            out.update(self._brain(brain, data.get("command") or {}, data.get("frame") or {}))
            out["rhythm"] = rhythm_of(self._walk)
            out["day"] = data.get("day")
            out["body"] = body_state(self._last_sample)
            out["body_now"] = self._body_now  # fid:page: what the body really does now
            stuck = float(((brain.get("locomotion") or {}).get("stuck")) or 0.0)
            out["body"]["stuck"] = stuck >= 1.0 and not out["body"]["down"]
            out["looks"] = data.get("looks")
            newest = self._mouth[-1][0] if self._mouth else 0
            track = [(t, v) for t, v in self._mouth if newest - t <= MOUTH_WINDOW_MS]
            out["mouth"] = {"t_ms": [t for t, _ in track], "open": [round(v, 3) for _, v in track]}
            return out

    def _status(self, data: Mapping[str, Any]) -> dict[str, Any]:
        """Where the driver is at, and which lanes are the real ones (said on the page)."""
        lanes = {str(k): str(v) for k, v in (data.get("lanes") or {}).items()}
        stubs = [LANE_ZH.get(k, k) for k, v in lanes.items() if v != "real"]
        state = str(data.get("state") or "")
        out: dict[str, Any] = {
            "driver": {
                "state": state,
                "state_zh": DRIVER_STATE_ZH.get(state, state),
                "error": data.get("error"),
                "lanes": lanes,
                "lanes_zh": lst(
                    (
                        msg(
                            "neuro.lanes.lane_state",
                            lane=LANE_ZH.get(k, k),
                            real_or_stand_in=msg("neuro.web.neuro_api.DriverSource.status.bcf00e")
                            if v == "real"
                            else msg("neuro.web.neuro_api.DriverSource.status.4d8a6a"),
                        )
                        for k, v in lanes.items()
                    )
                ),
                "notes": list(data.get("notes") or ())[:6],
                "realtime": data.get("realtime"),
                "hz": (data.get("stats") or {}).get("hz"),
            },
        }
        if lanes.get("brain") == "stub":
            out["note_zh"] = msg("neuro.lanes.brain_stub")
            out["note_tone"] = "warn"
        elif stubs:
            out["note_zh"] = msg("neuro.lanes.lanes_stub", stubs=lst(stubs))
            out["note_tone"] = "info"
        if data.get("error"):
            out["note_zh"] = msg("neuro.lanes.driver_error", error_text=data["error"])
            out["note_tone"] = "warn"
        genome = data.get("genome")
        if isinstance(genome, Mapping) and genome:
            key = (genome.get("seed"), genome.get("gd"), genome.get("branch"))
            if key != self._genome_key or self._card is None:
                self._genome_key = key
                self._card = card_of(genome)
            out["duck"] = self._card
        return out

    def _brain(
        self, brain: Mapping[str, Any], cmd: Mapping[str, Any], frame: Mapping[str, Any]
    ) -> dict[str, Any]:
        behaviour = str(brain.get("behaviour") or "")
        spec = C.BEHAVIOURS.get(behaviour)
        competition = [item for item in brain.get("competition") or () if isinstance(item, Mapping)]
        top = max((float(item.get("salience") or 0.0) for item in competition), default=1.0)
        scale = max(1.0, top)
        drives_now = dict(brain.get("drives") or {})
        drives = []
        for key, zh in DRIVES:
            got = key if key in drives_now else DRIVE_FALLBACK.get(key, key)
            drives.append(
                {"key": key, "zh": zh, "value": _unit(drives_now.get(got, 0.0)), "main": True}
            )
        drives += [
            {
                "key": key,
                "zh": MINOR_ZH.get(key) or C.DRIVES.get(key, key),
                "value": _unit(value),
                "main": False,
            }
            for key, value in drives_now.items()
            if key not in MAIN_DRIVES
        ]
        habituation = brain.get("habituation") or {}
        # (sensitisation is kept per sense — "sound", "touch"… — plus "all": after a scare)
        sensitisation = brain.get("sensitisation") or {}
        habits = []
        met = {name for name in habituation if ":" in name} | set(self._class_counts)
        others = sorted(
            (name for name in met if name.split(":", 1)[0] in OUTER_SENSES),
            key=lambda name: (-self._class_counts.get(name, 0), float(habituation.get(name, 1.0))),
        )
        classes = list(BUTTON_CLASSES) + others
        for name in list(dict.fromkeys(classes))[: HABIT_ROWS + len(BUTTON_CLASSES)]:
            level = float(habituation.get(name, 1.0) or 0.0)
            extra = sum(
                float(sensitisation.get(key, 0.0) or 0.0)
                for key in dict.fromkeys((name, name.split(":", 1)[0], "all"))
            )
            count = self._class_counts.get(name, 0)
            if name not in BUTTON_CLASSES and count == 0 and level > 0.99 and extra < 0.01:
                continue
            habits.append(
                {
                    "key": name,
                    "zh": class_zh(name),
                    "level": _unit(level),
                    "count": count,
                    "sensitised": extra >= 0.05,
                    "extra": round(extra, 3),
                }
            )
        vx = float(cmd.get("vx") or 0.0)
        vyaw = float(cmd.get("vyaw") or 0.0)
        asleep = bool(cmd.get("asleep"))
        loco: dict[str, Any] = {
            str(key): round(float(value), 3)
            for key, value in (brain.get("locomotion") or {}).items()
            if isinstance(value, (int, float))
        }
        now = self._body_now or {}
        true_mps = now.get("speed_mps") if isinstance(now.get("speed_mps"), (int, float)) else None
        loco.update(
            {
                "vx": round(vx, 3),
                "vyaw": round(vyaw, 3),
                "zh": locomotion_zh(vx, vyaw, asleep, true_mps),
                "true_mps": true_mps,
            }
        )
        if cmd.get("back_arc"):
            loco["zh"] = msg("neuro.loco.1086")
        reflex = str(brain.get("reflex") or "")
        light = (frame.get("camera") or {}).get("light")
        seen = bool(brain.get("moment")) and int(cmd.get("moment_seq") or 0) in self._moments_seen
        out = {
            "t_s": round(float(brain.get("t_ms") or 0) / 1000, 2),
            "behaviour": {
                "key": behaviour,
                "zh": brain.get("behaviour_zh") or C.behaviour_zh(behaviour),
                "why_zh": brain.get("reason_zh") or "",
                "since_s": round(float(brain.get("bout_ms") or 0) / 1000, 1),
                "satiation": _unit(brain.get("satiation", 0.0)),
                "family": spec.family if spec is not None else "",
                "body_zh": spec.body_zh if spec is not None else "",
                "reflex": reflex,
                "reflex_zh": C.REFLEXES.get(reflex, reflex) if reflex else "",
                "bout": brain.get("bout"),
                # a short head act inside it (a glance, a tilt…: contract.MOMENTS) — (fid:fix)
                # said only once the body was seen to perform it (the driver's witness)
                "moment": brain.get("moment") if seen else None,
                "moment_zh": C.behaviour_zh(str(brain["moment"])) if seen else "",
            },
            "candidates": [
                {
                    "key": str(item.get("behaviour")),
                    "zh": C.behaviour_zh(str(item.get("behaviour"))),
                    "pull": _unit(float(item.get("salience") or 0.0) / scale),
                    "activation": _unit(item.get("activation", 0.0)),
                    "why_zh": item.get("why_zh") or "",
                    "current": item.get("behaviour") == behaviour,
                }
                for item in competition[:6]
            ],
            "drives": drives,
            "arousal": {"value": _unit(brain.get("arousal", 0.0)), "zh": AROUSAL_ZH},
            "valence": round(max(-1.0, min(1.0, float(brain.get("valence") or 0.0))), 3),
            "habituation": habits,
            "places": places_of(brain.get("learned") or {}),
            "command": (brain.get("learned") or {}).get("command"),  # the last command and its outcome
            "locomotion": loco,
            "asleep": asleep,
            "stage": brain.get("stage"),
            "counts": {str(k): int(v) for k, v in (brain.get("counts") or {}).items()},
            "senses": senses_zh(frame.get("sources")),
        }
        if light in C.LIGHT_LEVELS:
            out["light"] = light
        return out

    def _injector(self) -> Callable[..., Any] | None:
        """The world-side injection: the one given, else the driver's own ``inject``."""
        if self._inject is not None:
            return self._inject
        own = getattr(self._driver, "inject", None)
        return own if callable(own) else None

    def _buttons(self) -> dict[str, bool]:
        keys = [item.key for item in STIMULI] + list(MORE_STIMULI)
        if self._injector() is None:
            return {key: False for key in keys}
        known = supported_by(self._driver) if self._inject is None else None
        if known is None:
            return {key: True for key in keys}
        out = {}
        for key in keys:
            item = STIMULUS_BY_KEY.get(key)
            names = {item.contract} if item is not None else {key}
            if key == "lights":
                names = {"lights_off", "lights_on"}
            out[key] = names <= known
        return out

    # -- the buttons

    def stimulus(self, kind: str, params: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._driver is None:
                self._driver = self.driver()
            inject = self._injector()
        if inject is None:
            raise ValueError(msg("neuro.api.stimulus.2d70"))
        item = STIMULUS_BY_KEY.get(kind)
        name = item.contract if item is not None else kind
        if kind == "lights":
            with self._lock:
                name = "lights_off" if self._lights_on else "lights_on"
        answer = dict(inject(name, params) or {})
        if answer.get("zh") and not answer.get("note_zh"):
            answer["note_zh"] = answer["zh"]  # (what the world did, in its own words)
        if kind in ("lights_off", "lights_on"):
            with self._lock:
                self._lights_on = bool(answer.get("lights_on", kind == "lights_on"))
                answer["lights_on"] = self._lights_on
        if kind == "lights":
            with self._lock:
                self._lights_on = bool(answer.get("lights_on", name == "lights_on"))
                answer["lights_on"] = self._lights_on
        return answer

    def set_day(self, minute: float) -> dict[str, Any]:
        driver = self.driver()
        own = getattr(driver, "set_day", None)
        if not callable(own):
            raise ValueError(msg("neuro.api.set_day.1668"))
        return dict(own(float(minute)) or {})

    def help_up(self) -> dict[str, Any]:
        driver = self.driver()
        own = getattr(driver, "help_up", None)
        if not callable(own):
            raise ValueError(msg("neuro.api.help_up.5eeb"))
        return dict(own() or {})

    def remote_set(self, on: bool, **kw: Any) -> dict[str, Any]:
        own = getattr(self.driver(), "remote_set", None)
        if not callable(own):
            raise ValueError(msg("neuro.remote.not_running"))
        return dict(own(on, **kw))

    def remote_twist(
        self, vx: Any, vyaw: Any, vy: Any = 0.0, key_to_recv_ms: float | None = None
    ) -> dict[str, Any]:
        own = getattr(self.driver(), "remote_twist", None)
        if not callable(own):
            raise ValueError(msg("neuro.remote.not_running"))
        return dict(own(vx, vyaw, vy, key_to_recv_ms))

    def remote_owned_by(self, owner: object) -> bool:
        driver = self.driver()
        hand = getattr(driver, "remote", None)
        owned = getattr(hand, "owned_by", None)
        return bool(callable(owned) and owned(owner))

    def go_there(self, x_m: Any, y_m: Any, who: Any = None) -> dict[str, Any]:
        own = getattr(self.driver(), "go_there", None)
        if not callable(own):
            raise ValueError(msg("neuro.gothere.refused.not_running"))
        return dict(own(x_m, y_m, None if who is None else str(who)))

    def play_motion(self, motion_id: str) -> dict[str, Any]:
        driver = self.driver()
        own = getattr(driver, "play_motion", None)
        if not callable(own):
            raise ValueError(msg("neuro.motion.not_running"))
        return dict(own(motion_id) or {})

    def knows(self) -> dict[str, Any]:
        """(MAP-EXPLORE) The duck's own map for the page's minimap layer (``NeuroDriver.knows``); off when there is none."""
        own = getattr(self.driver(), "knows", None)
        return dict(own()) if callable(own) else {"ok": True, "on": False}

    def command(self, command: Any) -> dict[str, Any]:
        driver = self.driver()
        own = getattr(driver, "command", None)
        if not callable(own):
            raise ValueError(msg("neuro.motion.not_running"))
        return dict(own(command) or {})

    def feedback(self, kind: str, trick: str | None) -> dict[str, Any]:
        """The owner's word on a trick, to the brain's ``trick_feedback`` (``NeuroDriver.feedback``)."""
        own = getattr(self.driver(), "feedback", None)
        if not callable(own):
            return {"reason": "not_running"}
        return dict(own(kind, trick) or {})

    def play_emotion(self, emotion_id: str) -> dict[str, Any]:
        driver = self.driver()
        own = getattr(driver, "play_emotion", None)
        if not callable(own):
            raise ValueError(msg("neuro.motion.not_running"))
        return dict(own(emotion_id) or {})

    def switch(self, choice: DuckChoice, card: Mapping[str, Any]) -> dict[str, Any]:
        driver = self.driver()
        if driver is None:
            raise ValueError(msg("neuro.api.switch.4c54"))
        answer = switch_driver(driver, choice)
        with self._lock:
            lights = self._lights_on
            self._fresh()
            self._lights_on = lights
            self._driver = driver
            self._card = dict(card)
            self._genome_key = (choice.seed, choice.gd, card.get("branch") or None)
        return answer

    def stop(self) -> None:
        return


def supported_by(driver: Any) -> set[str] | None:
    """The contract stimuli the live driver can do in the world (None: unknown — then every
    button is offered and a refusal is said in the driver's own words)."""
    own = getattr(driver, "stimuli_supported", None)
    if callable(own):
        return set(own())
    try:
        from duckmind.simweb import neuro_live as L
    except ImportError:
        return None
    tables = [
        getattr(L, name, None) for name in ("SOUND_STIMULI", "VISUAL_STIMULI", "HAND_STIMULI")
    ]
    if not all(isinstance(table, Mapping) for table in tables):
        return None
    names = {str(key) for table in tables if isinstance(table, Mapping) for key in table}
    # (the ones its inject() names outright: a rolled ball, a person, the lamps)
    return (
        names
        | {"roll_ball", "toy_drop", "novel_object", "walk_by", "sit_near"}
        | {
            "lights_off",
            "lights_on",
        }
    )


def body_state(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Whether the physical body is down and waiting for a hand (the page then offers
    「扶牠起來」): the body link's righting phase, the fall flag and the trunk's tilt."""
    righting = str(sample.get("righting") or "idle")
    tilt = float(sample.get("tilt_deg") or 0.0)
    fallen = bool(sample.get("fallen"))
    down = fallen or righting != "idle" or tilt > 35.0
    return {
        "down": down,
        "waiting_help": righting == "waiting_help" or (down and tilt > 35.0 and righting == "idle"),
        "righting": righting,
        "tilt_deg": round(tilt, 1),
    }


def rhythm_of(
    walk: Iterable[tuple[int, float, float]], window_s: int = WALK_WINDOW_S
) -> dict[str, Any]:
    """How it walked over the last ``window_s`` seconds, one bar a second: its fastest step that
    second (m/s, what robotd applied) and which way it turned — bursts and pauses, never one
    speed — and the share of the time it was stepping."""
    samples = list(walk)
    if not samples:
        return {"window_s": window_s, "bins": [], "turns": [], "share": 0.0}
    end = samples[-1][0]
    bins = [0.0] * window_s
    turns = [0.0] * window_s
    stepping = 0
    counted = 0
    for t_ms, vx, vyaw in samples:
        age = (end - t_ms) // 1000
        if age >= window_s or age < 0:
            continue
        index = window_s - 1 - int(age)
        counted += 1
        if vx > 0.05:
            stepping += 1
            if vx > bins[index]:
                bins[index] = vx
                turns[index] = vyaw
    return {
        "window_s": window_s,
        "bins": [round(v, 2) for v in bins],
        "turns": [round(v, 2) for v in turns],
        "share": round(stepping / counted, 3) if counted else 0.0,
    }


def switch_driver(driver: Any, choice: DuckChoice) -> dict[str, Any]:
    """Start the driver's next brain on ``choice`` (``seed:gd``): the driver's own ``switch``
    when it has one, else a fresh set of lanes (:func:`~duckmind.simweb.neuro_live.load_lanes`)
    that its tick thread takes up on its next tick. The body stays where it is."""
    own = getattr(driver, "switch", None)
    if callable(own):
        return dict(own(choice.spec) or {})
    from duckmind.simweb.neuro_live import NeuroSpec, load_lanes

    old = driver.spec
    spec = NeuroSpec(duck=choice.spec, seed=choice.seed, gd=choice.gd, record=old.record)
    lanes = load_lanes(spec)
    driver.spec = spec
    driver.lanes = lanes  # (one attribute: the tick thread's next tick uses the new brain)
    return {"lanes": dict(lanes.names), "notes": list(lanes.notes)}


def _size(value: Any) -> float:
    """A response size (≥ 0, not capped at 1)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(number, 3) if math.isfinite(number) and number > 0 else 0.0


def _unit(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(number):
        return 0.0
    return round(max(0.0, min(1.0, number)), 3)


# ---------------------------------------------------------------------------- the stub


@dataclass(slots=True)
class _Habit:
    level: float = 1.0
    count: int = 0
    scared_until: float = -1.0


# (key, drive weights: how each drive pulls it) — contract behaviours, made-up weights.
_STUB_ACTS: tuple[tuple[str, dict[str, float]], ...] = (
    ("wander", {"curiosity": 0.5, "boredom": 0.6, "energy": 0.3, "sleepiness": -0.5}),
    ("look_around", {"curiosity": 0.6, "fear": 0.2}),
    ("forage", {"curiosity": 0.5, "energy": 0.3, "boredom": 0.3}),
    ("preen", {"comfort": 0.6, "boredom": 0.2, "fear": -0.4}),
    ("doze", {"sleepiness": 0.9, "energy": -0.4, "fear": -0.5}),
    ("follow", {"social": 0.8, "fear": 0.2}),
    ("freeze", {"fear": 0.9}),
    ("retreat", {"fear": 0.7, "boredom": -0.2}),
    ("little_dance", {"comfort": 0.4, "energy": 0.4, "social": 0.2}),
    ("stretch_yawn", {"sleepiness": 0.4, "comfort": 0.3}),
)
_STUB_WHY = {
    "curiosity": msg("neuro.stub.stub_why.curiosity"),
    "boredom": msg("neuro.stub.stub_why.boredom"),
    "energy": msg("neuro.stub.stub_why.energy"),
    "comfort": msg("neuro.stub.stub_why.comfort"),
    "sleepiness": msg("neuro.stub.stub_why.sleepiness"),
    "social": msg("neuro.stub.stub_why.social"),
    "fear": msg("neuro.stub.stub_why.fear"),
}
_STUB_REFLEX = {
    "clap": ("startle", "sound:clap", "neuro.stub.stub_reflex.clap"),
    "call_name": ("orient", "sound:name", "neuro.stub.stub_reflex.call_name"),
    "wave": ("orient", "visual:motion", "neuro.stub.stub_reflex.wave"),
    "pat_head": ("withdraw", "touch:head", "neuro.stub.stub_reflex.pat_head"),
    "roll_ball": ("orient", "visual:thing", "neuro.stub.stub_reflex.roll_ball"),
    "novel_object": ("orient", "visual:thing", "neuro.stub.stub_reflex.novel_object"),
    "lights": ("startle", "light:off", "neuro.stub.stub_reflex.lights"),
    "family_walk": ("orient", "visual:person", "neuro.stub.stub_reflex.family_walk"),
}


class StubSource:
    """For checking the page only (``--neuro-stub``): made-up numbers that rise and fall so the
    page can be exercised before the brain exists. Every snapshot says so (``source: "stub"``,
    ``note_zh``) and the page shows that in magenta. Nothing here is the duck's behaviour, and
    it never moves the physics body."""

    kind = "stub"

    def __init__(self, card: Mapping[str, Any], clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._start = clock()
        self._last = self._start
        self._seq = 0
        self._event_seq = 0
        self._events: deque[dict[str, Any]] = deque(maxlen=KEEP_EVENTS)
        self._vocal: deque[dict[str, Any]] = deque(maxlen=KEEP_EVENTS)
        self._body_rows: deque[dict[str, Any]] = deque(maxlen=KEEP_EVENTS)  # (made up, fid:page)
        self._lights_on = True
        self._walk_second = -1
        self._walk_bins: deque[tuple[float, float]] = deque(maxlen=WALK_WINDOW_S)
        self._day = 600.0
        self._reset(card)

    def _reset(self, card: Mapping[str, Any]) -> None:
        self._card = dict(card)
        self._rng = random.Random(int(card.get("seed", 0)))
        axes = {item["key"]: float(item["value"]) for item in card.get("axes", [])}
        self._base = {
            "energy": 0.4 + 0.4 * axes.get("activity", 0.5),
            "sleepiness": 0.25,
            "curiosity": 0.2 + 0.6 * axes.get("exploration", 0.5),
            "fear": 0.05 + 0.2 * (1 - axes.get("boldness", 0.5)),
            "social": 0.2 + 0.5 * axes.get("sociability", 0.5),
            "comfort": 0.5,
            "boredom": 0.3,
        }
        self._drives = dict(self._base)
        self._current = "look_around"
        self._since = self._clock()
        self._noise = {key: 0.0 for key, _ in _STUB_ACTS}
        self._habits = {item.key: _Habit() for item in STIMULI}

    def _step(self, now: float) -> None:
        dt = max(0.0, min(1.0, now - self._last))
        self._last = now
        if dt <= 0:
            return
        for key, base in self._base.items():
            value = self._drives[key]
            value += (base - value) * dt / 20.0 + self._rng.gauss(0.0, 0.035) * math.sqrt(dt)
            self._drives[key] = min(1.0, max(0.0, value))
        for key in self._noise:
            kick = self._rng.gauss(0.0, 0.08) * math.sqrt(dt)
            self._noise[key] += -self._noise[key] * dt / 3.0 + kick
        for habit in self._habits.values():
            habit.level += (1.0 - habit.level) * dt / 90.0
        pulls = self._pulls()
        best = max(pulls, key=lambda key: pulls[key])
        if best != self._current and pulls[best] > pulls[self._current] + 0.12:
            self._current = best
            self._since = now

    def _pulls(self) -> dict[str, float]:
        out = {}
        for key, weights in _STUB_ACTS:
            raw = sum(self._drives[name] * weight for name, weight in weights.items())
            out[key] = min(1.0, max(0.0, 0.2 + raw * 0.8 + self._noise[key]))
        return out

    def snapshot(self, since: int = 0) -> dict[str, Any]:
        with self._lock:
            now = self._clock()
            self._step(now)
            self._seq += 1
            pulls = self._pulls()
            ranked = sorted(pulls, key=lambda key: -pulls[key])[:6]
            weights = dict(_STUB_ACTS)[self._current]
            lead = max(weights, key=lambda name: weights[name] * self._drives[name])
            arousal = min(1.0, 0.2 + 0.7 * self._drives["fear"] + 0.3 * self._drives["curiosity"])
            now_zh = C.behaviour_zh(self._current)
            return {
                "enabled": True,
                "schema": PAGE_SCHEMA,
                "source": self.kind,
                "note_zh": msg("neuro.stub.stubsource.snapshot.note_zh"),
                "seq": self._event_seq,
                "n": self._seq,
                "t_s": round(now - self._start, 2),
                "duck": self._card,
                "behaviour": {
                    "key": self._current,
                    "zh": now_zh,
                    "why_zh": msg(
                        "neuro.stub.stubsource.snapshot.why_zh",
                        lead=_STUB_WHY.get(lead, lead),
                        lead2=self._drives[lead],
                        now_zh=now_zh,
                    ),
                    "since_s": round(now - self._since, 1),
                    "satiation": round(min(1.0, (now - self._since) / 20.0), 3),
                    "family": C.BEHAVIOURS[self._current].family,
                },
                "candidates": [
                    {
                        "key": key,
                        "zh": C.behaviour_zh(key),
                        "pull": round(pulls[key], 3),
                        "activation": round(pulls[key] if key == self._current else 0.0, 3),
                        "why_zh": "",
                        "current": key == self._current,
                    }
                    for key in ranked
                ],
                "drives": [
                    {"key": key, "zh": zh, "value": round(self._drives[key], 3), "main": True}
                    for key, zh in DRIVES
                ],
                "arousal": {"value": round(arousal, 3), "zh": AROUSAL_ZH},
                "valence": round(self._drives["comfort"] - self._drives["fear"], 3),
                "reflexes": [item for item in self._events if item["seq"] > since],
                "vocal": [item for item in self._vocal if item["seq"] > since],
                "reflex_body": [item for item in self._body_rows if item["seq"] > since],
                "body_now": {
                    "zh": msg("neuro.stub.stubsource.snapshot.zh.e80910"),
                    "posture": "stand",
                    "mismatch_zh": None,
                },
                "asleep": self._current == "doze",
                "habituation": [
                    {
                        "key": _STUB_REFLEX[item.key][1],
                        "zh": item.zh,
                        "level": round(self._habits[item.key].level, 3),
                        "count": self._habits[item.key].count,
                        "sensitised": self._habits[item.key].scared_until > now,
                    }
                    for item in STIMULI
                ],
                "places": [
                    {
                        "key": "nest",
                        "zh": msg("neuro.stub.stubsource.snapshot.zh.abce1d"),
                        "value": 0.7,
                        "falls": 0,
                        "avoid": False,
                    },
                    {
                        "key": "a",
                        "zh": msg("neuro.stub.stubsource.snapshot.zh.a1dca0"),
                        "value": 0.35,
                        "falls": 0,
                        "avoid": False,
                    },
                    {
                        "key": "b",
                        "zh": msg("neuro.stub.stubsource.snapshot.zh.fe0823"),
                        "value": -0.8,
                        "falls": 3,
                        "avoid": True,
                    },
                ],
                "locomotion": {
                    "zh": msg("neuro.stub.stubsource.snapshot.zh.630090")
                    if self._current == "wander"
                    else msg("neuro.stub.stubsource.snapshot.zh.701dac")
                },
                "rhythm": self._rhythm(now),
                "day": {"minute": self._day, "zh": _minute_zh(self._day), "rate": 1.0},
                "body": {"down": False, "waiting_help": False, "righting": "idle", "tilt_deg": 0},
                "senses": senses_zh(),
                "stimuli": {
                    key: True for key in [item.key for item in STIMULI] + list(MORE_STIMULI)
                },
                "world": {"lights_on": self._lights_on},
            }

    def _rhythm(self, now: float) -> dict[str, Any]:
        """(Made up) bursts and pauses for the page's walking strip."""
        second = int(now)
        if second != self._walk_second:
            self._walk_second = second
            going = self._current in ("wander", "forage", "follow", "retreat")
            vx = self._rng.uniform(0.35, 0.55) if going and self._rng.random() < 0.7 else 0.0
            self._walk_bins.append((vx, self._rng.uniform(-0.8, 0.8) if vx else 0.0))
        bins = list(self._walk_bins)[-WALK_WINDOW_S:]
        pad = [(0.0, 0.0)] * (WALK_WINDOW_S - len(bins))
        rows = pad + bins
        return {
            "window_s": WALK_WINDOW_S,
            "bins": [round(v, 2) for v, _ in rows],
            "turns": [round(t, 2) for _, t in rows],
            "share": round(sum(1 for v, _ in bins if v > 0) / max(1, len(bins)), 3),
        }

    def stimulus(self, kind: str, params: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            now = self._clock()
            self._step(now)
            item = STIMULUS_BY_KEY.get(ALIASES.get(kind, kind))
            if item is None:  # (a contract stimulus of the 「更多」 row, or a rhythm)
                self._event_seq += 1
                self._events.append(
                    {
                        "seq": self._event_seq,
                        "t_s": round(now - self._start, 2),
                        "kind": "orient",
                        "zh": msg(
                            "neuro.stub.stubsource.stimulus.zh.5ade1c",
                            stimulus=MORE_ZH.get(kind, kind),
                        ),
                        "strength": 0.5,
                    }
                )
                return {
                    "note_zh": msg(
                        "neuro.stub.stubsource.stimulus.note_zh.a29b67",
                        stimulus=MORE_ZH.get(kind, kind),
                    )
                }
            kind = item.key
            habit = self._habits[kind]
            scared = habit.scared_until > now
            strength = min(1.0, habit.level * (1.3 if scared else 1.0))
            habit.level = max(0.05, habit.level * 0.72)
            habit.count += 1
            reflex_kind, _, text_key = _STUB_REFLEX[kind]
            side = self._rng.choice(
                (
                    msg("neuro.stub.stubsource.stimulus.x.df3ee6"),
                    msg("neuro.stub.stubsource.stimulus.x.851cdd"),
                )
            )
            if kind == "lights":
                self._lights_on = not self._lights_on
            if kind == "clap":
                habit.scared_until = now + 20.0
                self._drives["fear"] = min(1.0, self._drives["fear"] + 0.45 * strength)
            elif kind in ("roll_ball", "novel_object", "wave"):
                self._drives["curiosity"] = min(1.0, self._drives["curiosity"] + 0.3 * strength)
            elif kind in ("call_name", "family_walk"):
                self._drives["social"] = min(1.0, self._drives["social"] + 0.3 * strength)
                self._event_seq += 1
                self._vocal.append(
                    {"seq": self._event_seq, "tag": "greet", "zh": VOCAL_ZH["greet"]}
                )
            elif kind == "pat_head":
                self._drives["comfort"] = min(1.0, self._drives["comfort"] + 0.3 * strength)
            self._event_seq += 1
            self._events.append(
                {
                    "seq": self._event_seq,
                    "t_s": round(now - self._start, 2),
                    "kind": reflex_kind,
                    "zh": msg(text_key, s=0.3 + 1.2 * strength, side=side),
                    "strength": round(strength, 2),
                }
            )
            # (made up: what the body did — every third one it "could not", to show that row)
            row = self._event_seq
            self._event_seq += 1
            failed = habit.count % 3 == 0
            self._body_rows.append(
                {
                    "seq": self._event_seq,
                    "of": row,
                    "verdict": "not_done" if failed else "done",
                    "zh": (
                        msg("neuro.stub.stubsource.stimulus.zh.03bdd7")
                        if failed
                        else msg(
                            "neuro.stub.stubsource.stimulus.zh.746ac7",
                            side=side,
                            strength=8 + 10 * strength,
                        )
                    ),
                }
            )
            return {
                "note_zh": msg(
                    "neuro.stub.stubsource.stimulus.note_zh.e11959",
                    stimulus=item.zh,
                    world=item.world_zh,
                ),
                "strength": round(strength, 2),
                "lights_on": self._lights_on,
            }

    def switch(self, choice: DuckChoice, card: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._reset(card)
            return {"note_zh": msg("neuro.stub.stubsource.switch.note_zh", duck_name=choice.name)}

    def set_day(self, minute: float) -> dict[str, Any]:
        with self._lock:
            self._day = float(minute) % 1440.0
            return {"minute": self._day, "zh": _minute_zh(self._day), "rate": 1.0}

    def help_up(self) -> dict[str, Any]:
        return {"ok": True, "note_zh": msg("neuro.stub.stubsource.help_up.note_zh")}

    def remote_set(self, on: bool, **_: Any) -> dict[str, Any]:
        return {"ok": False, "refused": True, "reason": "stub", "reason_zh": msg("neuro.remote.not_running")}

    def remote_twist(self, *_: Any, **__: Any) -> dict[str, Any]:
        return {"ok": False, "on": False}

    def remote_owned_by(self, owner: object) -> bool:
        return False

    def go_there(self, x_m: Any, y_m: Any, who: Any = None) -> dict[str, Any]:
        return {"ok": False, "refused": True, "reason": "stub", "reason_zh": msg("neuro.gothere.refused.not_running")}

    def play_motion(self, motion_id: str) -> dict[str, Any]:
        # (the page check: nothing moves; the refusal says so)
        return {"ok": False, "refused": True, "reason": "stub",
                "reason_zh": msg("neuro.motion.not_running")}  # fmt: skip

    def command(self, command: Any) -> dict[str, Any]:
        # (the page check: nothing hears; the refusal says so)
        return {"state": "refused", "state_zh": CB.STATE_MSG["refused"], "reason": "not_running",
                "reason_zh": CB.REASON_MSG["not_running"], "queued": False}  # fmt: skip

    def feedback(self, kind: str, trick: str | None) -> dict[str, Any]:
        # (the page check: no brain learns; the refusal says so)
        return {"reason": "not_running"}

    def stop(self) -> None:
        return


# ---------------------------------------------------------------------------- the hub


class NeuroHub:
    """The prototype as the server sees it: on or off, its source (the brain's driver, the stub,
    or none yet) and the duck now. Its routes answer whether or not the prototype runs.
    ``voice_bank``: the ducks' official voices (:mod:`duckmind.simweb.voice`; None: no voice)."""

    def __init__(
        self,
        enabled: bool = False,
        source: NeuroSource | None = None,
        choice: DuckChoice | None = None,
        voice_bank: voice.VoiceBank | None = None,
        given_name: str = "",
    ) -> None:
        self.enabled = enabled
        self.source = source
        self.choice = choice or PRESET_BY_ID[DEFAULT_PRESET]
        # the name the server was started with (``--name``): the display name of the duck it
        # started with, wherever that duck is shown. A name is data, never translated. "": none.
        self.given_name = given_name.strip()[:GIVEN_NAME_MAX]
        self._given_duck = (self.choice.seed, self.choice.gd, self.choice.preset)
        if self.given_name:
            self.choice = replace(self.choice, name=self.given_name)
        self.voice = voice_bank
        self._card: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self._streams = 0
        self.stopping = threading.Event()
        # called (no arguments) after the family switched to another duck: the brain and its lanes
        # are new then (live voice: its connection is made again for the new duck)
        self.switch_hooks: list[Callable[[], None]] = []
        # the server's own life duck (--life), as a picker choice at its growth day now: the page
        # can always come back to it; None: this server has no life
        self.life_duck: Callable[[], DuckChoice] | None = None
        # the server's life host (--life): Duck Core hears the owner's word on a trick — the phone's
        # 「乖」/ treat / 「不對」 and bridge-v2's praise / scold — when its duck is on screen; None:
        # no life (the brain's step alone)
        self.core_feedback: Callable[[str, str], Mapping[str, Any] | None] | None = None

    # -- plain calls (the driver and the tests use these directly)

    def _is_given_duck(self, seed: Any, gd: Any, preset: Any) -> bool:
        """Is this the duck the server started with (and was it given a name)?"""
        return bool(self.given_name) and (seed, gd, preset) == self._given_duck

    def named(self, choice: DuckChoice) -> DuckChoice:
        """``choice`` as shown here: a picker duck that is the server's own one keeps the name
        the server was started with when it is picked again; every other duck keeps its own
        (a duck asked for by seed keeps the name asked)."""
        if choice.preset and self._is_given_duck(choice.seed, choice.gd, choice.preset):
            return replace(choice, name=self.given_name)
        return choice

    def ducks(self) -> list[dict[str, Any]]:
        """The picker's four ducks (the server's own one under the name it was given)."""
        cards = [
            {**card, "name": self.given_name}
            if card.get("preset") and self._is_given_duck(card["seed"], card["gd"], card["preset"])
            else card
            for card in preset_cards()
        ]
        if self.life_duck is not None:
            cards.insert(0, duck_card(self.life_duck()))
        return cards

    def card(self) -> dict[str, Any]:
        """The duck now, as the page shows it."""
        with self._lock:
            if self._card is None or self._card.get("seed") != self.choice.seed:
                self._card = duck_card(self.choice)
            return self._card

    def voice_duck(self) -> int:
        """The duck whose own voice the page plays: the neural duck on screen (its seed)."""
        return int(self.choice.seed)

    def prepare_voices(self) -> threading.Thread | None:
        """The duck's own voice now (its bank rendered in the background if it is new), then —
        one after another, behind it — the picker's four, so a switch to one of them has its
        voice at once. Banks already in the cache cost nothing. None: no voice here."""
        bank = self.voice
        if bank is None:
            return None
        first = self.voice_duck()
        bank.prepare(first)

        def the_four() -> None:
            bank.wait(first, 30.0)
            for item in PRESETS:
                if self.stopping.is_set():
                    return
                if bank.prepare(item.seed) == "rendering":
                    bank.wait(item.seed, 30.0)

        thread = threading.Thread(target=the_four, name="voice-presets", daemon=True)
        thread.start()
        return thread

    def attach(self, source: NeuroSource | None) -> None:
        """The driver hands over its source once the brain runs (None: gone again)."""
        self.source = source

    def can_lie(self) -> bool | None:
        """Whether this simulator's robotd can lie the duck down (play dead: it answers ``robot.poseJoints``);
        None while there is no body to ask."""
        try:
            driver_of = getattr(self.source, "driver", None)
            body = getattr(driver_of() if callable(driver_of) else None, "_body", None)
            if body is None or not hasattr(body, "pose_joints_ok"):
                return None
            return bool(body.pose_joints_ok)
        except Exception:
            return None

    def status(self) -> dict[str, Any]:
        if not self.enabled:
            return {"ok": True, "enabled": False, "schema": PAGE_SCHEMA}
        source = self.source
        return {
            "ok": True,
            "enabled": True,
            "schema": PAGE_SCHEMA,
            "label_zh": LABEL_ZH,
            "source": source.kind if source is not None else None,
            "waiting_zh": "" if source is not None else msg("neuro.waiting.c692"),
            "stimuli": [
                {
                    "key": item.key,
                    "zh": item.zh,
                    "world_zh": item.world_zh,
                    "sense_zh": item.sense_zh,
                }
                for item in STIMULI
            ],
            "more": [
                {"key": key, "zh": MORE_ZH[key], "channels": MORE_CHANNELS[key]}
                for key in MORE_STIMULI
            ],
            "day_choices": [{"zh": zh, "minute": minute} for zh, minute in DAY_CHOICES],
            "ducks": self.ducks(),
            "duck": self.card(),
            "stream_every_s": STREAM_EVERY_S,
            "motions": motions.status(),  # 「動作試玩」: ids and gate verdicts (the page's words are its own)
            "emotions": emotions.status(
                can_lie=self.can_lie()
            ),  # … and the 14 emotions (group 「情緒」)
        }

    def telemetry(self, since: int = 0) -> dict[str, Any]:
        if not self.enabled:
            return {"ok": True, "enabled": False, "schema": PAGE_SCHEMA}
        source = self.source
        if source is None:
            return {
                "ok": True,
                "enabled": True,
                "schema": PAGE_SCHEMA,
                "source": None,
                "waiting_zh": msg("neuro.waiting.c692"),
                "duck": self.card(),
            }
        snapshot = dict(source.snapshot(since))
        seen = snapshot.get("duck")
        if isinstance(seen, Mapping) and self._is_given_duck(
            seen.get("seed"), seen.get("gd"), seen.get("preset")
        ):
            # (the driver's own card knows the preset's name only: this is the server's duck)
            snapshot["duck"] = {**seen, "name": self.given_name}
        snapshot.setdefault("duck", self.card())
        snapshot["ok"] = True
        snapshot["enabled"] = True
        snapshot.setdefault("schema", PAGE_SCHEMA)
        snapshot.setdefault("source", source.kind)
        return snapshot

    def stimulus(self, params: Mapping[str, Any]) -> dict[str, Any]:
        self._must_run()
        asked = str(params.get("kind", ""))
        # the page's names and the contract's both work (pat_head = pet_head, …)
        kind = ALIASES.get(asked, asked)
        known = kind in STIMULUS_BY_KEY or kind in MORE_STIMULI
        if not known and kind not in ("lights_off", "lights_on"):
            raise ValueError(msg("neuro.error.stimulus.9a8dbc", given_id=asked))
        source = self.source
        if source is None:
            raise ValueError(msg("neuro.error.help_up.d7a2bc"))
        extra = {key: value for key, value in params.items() if key != "kind"}
        try:
            answer = dict(source.stimulus(kind, extra))
        except ValueError as error:
            # the world said no for now (its hand is busy…): a normal answer in its own words,
            # not a failed request (the page says it; the browser logs no error)
            return {"ok": False, "kind": kind, "refused": True, "error": _said(error)}
        item = STIMULUS_BY_KEY.get(kind)
        answer.setdefault("note_zh", item.zh if item is not None else MORE_ZH.get(kind, kind))
        return {"ok": True, "kind": kind, **answer}

    def set_day(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """The duck's time of day: ``{"minute": m}`` or ``{"choice": "上午"}`` (DAY_CHOICES)."""
        self._must_run()
        choice = params.get("choice")
        if choice is not None:
            table = dict(DAY_CHOICES)
            if str(choice) not in table:
                raise ValueError(msg("neuro.api.set_day.eec2", given_id=choice))
            minute = table[str(choice)]
        else:
            minute = float(params.get("minute", -1))
            if not 0.0 <= minute < 1440.0:
                raise ValueError(msg("neuro.error.set_day.0850d9"))
        source = self.source
        if source is None:
            raise ValueError(msg("neuro.error.help_up.d7a2bc"))
        day = dict(source.set_day(minute))
        return {
            "ok": True,
            "day": day,
            "note_zh": msg("neuro.api.set_day.f43f", day=day.get("zh", "")),
        }

    def help_up(self) -> dict[str, Any]:
        self._must_run()
        source = self.source
        if source is None:
            raise ValueError(msg("neuro.error.help_up.d7a2bc"))
        return {"ok": True, **dict(source.help_up())}

    def remote(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Remote mode on or off: ``{"on": bool}`` (the family's hand on the legs; neuro/remote.py)."""
        self._must_run()
        own = getattr(self.source, "remote_set", None)
        if not callable(own):
            raise ValueError(msg("neuro.remote.not_running"))
        if "on" not in params or not isinstance(params["on"], bool):
            raise ValueError(msg("neuro.remote.bad_request"))
        return dict(own(params["on"]))

    def remote_twist(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """One twist ``{"vx", "vyaw", "vy"?}`` (the slow path; the page uses the WebSocket)."""
        self._must_run()
        own = getattr(self.source, "remote_twist", None)
        if not callable(own):
            raise ValueError(msg("neuro.remote.not_running"))
        return dict(own(params.get("vx", 0.0), params.get("vyaw", 0.0), params.get("vy", 0.0)))

    def go_there(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Go there: ``{"point": {"x_m", "y_m"}}`` in the WORLD's frame (a click on the floor). The driver
        turns it into what the duck itself could say (a bearing and a range from its trunk) and the brain
        keeps it as an errand; the answer is queued or refused with a reason (a refusal is a normal answer)."""
        self._must_run()
        own = getattr(self.source, "go_there", None)
        if not callable(own):
            raise ValueError(msg("neuro.gothere.refused.not_running"))
        point = params.get("point")
        if not isinstance(point, Mapping):
            raise ValueError(msg("neuro.gothere.bad_point"))
        return dict(own(point.get("x_m"), point.get("y_m"), params.get("who")))

    def _remote_ws(self, handler: BaseHTTPRequestHandler) -> None:
        """The lowest-latency way in for twists: one WebSocket per page (remote_ws.py). Messages (JSON):
        ``{"on": true|false}`` the toggle, ``{"vx", "vyaw", "t"?}`` a twist (``t``: the page's key-event
        time in ms since the epoch, for the latency numbers), ``{"ping": n}`` is answered. A closed socket
        ends remote mode at once (the brain takes the legs back)."""
        from duckmind.simweb import remote_ws

        source = self.source
        if not self.enabled or not callable(getattr(source, "remote_set", None)):
            _send_json(handler, HTTPStatus.SERVICE_UNAVAILABLE, {"ok": False, "error": msg("neuro.remote.not_running")})
            return
        conn = remote_ws.handshake(handler)
        if conn is None:
            return
        try:
            while not self.stopping.is_set():
                text = conn.recv(0.5)
                if text is None:
                    continue
                try:
                    message = json.loads(text)
                except ValueError:
                    continue
                if not isinstance(message, dict):
                    continue
                if "on" in message:
                    try:
                        answer = source.remote_set(bool(message["on"]), owner=conn, why="off")
                    except ValueError as error:
                        answer = {"ok": False, "error": _said(error)}
                    conn.send_text(json.dumps(boundary.attach({"re": "on", **answer}), ensure_ascii=False))
                elif "ping" in message:
                    conn.send_text(json.dumps({"re": "ping", "ping": message["ping"]}))
                else:
                    sent = message.get("t")
                    lag = None
                    if isinstance(sent, (int, float)):
                        lag = max(0.0, time.time() * 1000.0 - float(sent))
                    source.remote_twist(
                        message.get("vx", 0.0), message.get("vyaw", 0.0), message.get("vy", 0.0), lag
                    )
        except (remote_ws.Closed, OSError):
            pass
        finally:
            try:
                if source.remote_owned_by(conn):
                    source.remote_set(False, why="disconnect")
            except Exception:
                pass
            conn.close()

    def play_motion(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """「動作試玩」: ``{"id": <a motions.MOTIONS id>}`` — the duck does that trained motion now, or
        says why it will not (a refusal is a normal answer: ``ok`` false with ``reason_zh``)."""
        self._must_run()
        source = self.source
        if source is None:
            raise ValueError(msg("neuro.error.help_up.d7a2bc"))
        motion_id = str(params.get("id", ""))
        if motion_id in emotions.BY_ID:  # 「情緒」: Pollen's microduck_emotions (neuro/emotions.py)
            own = getattr(source, "play_emotion", None)
            if not callable(own):
                raise ValueError(msg("neuro.motion.not_running"))
            try:
                answer = dict(own(motion_id))
            except ValueError as error:
                return {"ok": False, "id": motion_id, "refused": True, "reason": "error",
                        "reason_zh": _said(error)}  # fmt: skip
            answer.setdefault("ok", True)
            answer.setdefault("id", motion_id)
            return answer
        if motion_id not in motions.BY_ID:
            raise ValueError(msg("neuro.motion.unknown"))
        try:
            answer = dict(source.play_motion(motion_id))
        except ValueError as error:
            return {"ok": False, "id": motion_id, "refused": True, "reason": "error",
                    "reason_zh": _said(error)}  # fmt: skip
        answer.setdefault("ok", True)
        answer.setdefault("id", motion_id)
        return answer

    def command(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """``POST /api/neuro/command``: ``{"text": ..., "lang", "who"}`` (typed or spoken words) or
        ``{"verb": ..., "lang", "who"}`` (a button). The words are read into one command
        (:mod:`duckmind.neuro.commands`) and handed to the duck's brain as what it heard; the answer
        says ``queued`` (it took it in and will do it) or ``refused`` and why (a refusal is a normal
        answer). A request that cannot be made into a command is a 400 (:class:`CommandError`)."""
        self._must_run()
        command = from_request(params, duck_name=self.choice.name)
        heard = {
            "verb": command.verb,
            "lang": command.lang,
            "confidence": round(float(command.confidence), 3),
            "named": command.named,
            "source": command.source,
            "who": command.who,
        }
        if command.text:
            heard["text"] = command.text
        if command.verb == "go_there":
            # (pointing at the floor is a second route: it needs the world's frame, see ``go_there``)
            answer = self.go_there({"point": dict(command.target or {}), "who": command.who})
            refused = bool(answer.get("refused"))
            return {"ok": not refused, "verb": command.verb, "heard": heard, **answer}
        source = self.source
        if source is None:
            raise ValueError(msg("neuro.error.help_up.d7a2bc"))
        own = getattr(source, "command", None)
        if not callable(own):
            return {"ok": False, "verb": command.verb, "heard": heard, "refused": True,
                    "reason": "not_running", "reason_zh": CB.REASON_MSG["not_running"]}  # fmt: skip
        answer = dict(own(command))
        refused = answer.get("state") == "refused"
        out = {"ok": not refused, "verb": command.verb, "heard": heard, "queued": not refused,
               "refused": refused}  # fmt: skip
        out.update({k: v for k, v in answer.items() if k not in ("queued",)})
        if refused:
            out["queued"] = False
        return out

    def feedback(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """``POST /api/neuro/feedback``: ``{"kind": "praise" | "treat" | "scold", "trick"?}`` — the
        owner's 「乖」, a treat (a praise to the trick) or 「不對」 for a trick: the one named, or the one
        the duck did last within three seconds (by itself: shaping). The duck's own brain moves that
        trick's proficiency (``Brain.trick_feedback``); Duck Core, when the server's life duck is the
        one on screen, hears the phone's word (``owner_good`` / ``pad_good`` / ``owner_not_that``)
        and bridge-v2's ``praise`` / ``scold`` for that trick (:attr:`core_feedback`). A feedback it
        cannot take is a 400 with ``neuro.feedback.error.<reason>`` (:class:`FeedbackError`)."""
        kind = params.get("kind")
        if kind not in FEEDBACK_KINDS:
            raise FeedbackError("bad_kind")
        named = params.get("trick")
        if named in (None, ""):
            named = None
        elif named not in TRICKS:
            raise FeedbackError("bad_trick")
        source = self.source
        own = getattr(source, "feedback", None)
        if not self.enabled or not callable(own):
            raise FeedbackError("not_running")
        answer = dict(own(str(kind), named))
        reason = answer.get("reason")
        if reason in ("no_trick", "not_running"):
            raise FeedbackError(str(reason))
        trick = answer.get("trick")
        if trick not in TRICKS:
            raise FeedbackError("no_trick")
        proficiency = answer.get("proficiency")
        pct = None if proficiency is None else int(round(100.0 * float(proficiency)))
        note_key = feedback_note_key(str(kind), answer)
        out: dict[str, Any] = {
            "ok": True,
            "kind": kind,
            "trick": trick,
            "proficiency": proficiency,
            "step": answer.get("step"),
            "capped": bool(answer.get("capped")),
            "shaped": bool(answer.get("shaped")),
            "counted": bool(answer.get("counted", True)),
            "reason": reason,
            "note_key": note_key,
            "note": msg(note_key, trick=trick, pct=pct if pct is not None else 0),
            "brain": answer.get("brain"),
            "core": None,
        }
        hook = self.core_feedback
        if hook is not None:
            try:
                core = hook(str(kind), str(trick))
            except Exception as error:  # (Duck Core's fault never takes back the brain's step)
                core = {"problem": f"{type(error).__name__}: {error}"}
            out["core"] = None if core is None else dict(core)
        return out

    def switch(self, params: Mapping[str, Any]) -> dict[str, Any]:
        self._must_run()
        if params.get("preset") == LIFE_PRESET and self.life_duck is not None:
            choice = self.life_duck()
        else:
            choice = self.named(choice_from(params))
        card = duck_card(choice)
        if self.voice is not None:
            self.voice.prepare(choice.seed)  # its own voice, rendered while the brain changes
        source = self.source
        answer = dict(source.switch(choice, card)) if source is not None else {}
        with self._lock:
            self.choice = choice
            self._card = card
        answer.setdefault(
            "note_zh",
            msg("neuro.api.switch.3e99", duck_name=choice.name, duck_line=card["line_zh"]),
        )
        for hook in list(self.switch_hooks):
            try:
                hook()
            except Exception as error:  # (a fault in a listener never breaks the switch)
                answer.setdefault("hook_faults", []).append(f"{type(error).__name__}: {error}")
        return {"ok": True, "duck": card, **answer}

    def stop(self) -> None:
        self.stopping.set()
        if self.source is not None:
            self.source.stop()

    def _must_run(self) -> None:
        if not self.enabled:
            raise ValueError(msg("neuro.error.must_run.378a40"))

    # -- HTTP

    @staticmethod
    def handles(path: str) -> bool:
        return path.startswith("/api/neuro/")

    def get(self, handler: BaseHTTPRequestHandler, path: str, query: str) -> None:
        since = _int(parse_qs(query).get("since", ["0"])[0])
        if path == "/api/neuro/status":
            _send_json(handler, HTTPStatus.OK, self.status())
        elif path == "/api/neuro/telemetry":
            _send_json(handler, HTTPStatus.OK, self.telemetry(since))
        elif path == "/api/neuro/stream":
            self._stream(handler, since)
        elif path == "/api/neuro/remote/ws":
            self._remote_ws(handler)
        elif path == "/api/neuro/knows":
            # (MAP-EXPLORE) the duck's own map for the page's minimap layer (numbers only; small: a coarse run-length coded grid)
            source = self.source
            knows = getattr(source, "knows", None) if self.enabled else None
            _send_json(
                handler, HTTPStatus.OK, knows() if callable(knows) else {"ok": True, "on": False}
            )
        else:
            _send_json(handler, HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})

    def post(self, handler: BaseHTTPRequestHandler, path: str) -> None:
        length = _int(handler.headers.get("Content-Length", "0"))
        if length > MAX_BODY:
            payload = {"ok": False, "error": "request too large"}
            _send_json(handler, HTTPStatus.REQUEST_ENTITY_TOO_LARGE, payload)
            return
        try:
            params = json.loads(handler.rfile.read(length) or b"{}")
            if not isinstance(params, dict):
                raise ValueError(msg("neuro.error.post.5db7bf"))
            if path == "/api/neuro/stimulus":
                _send_json(handler, HTTPStatus.OK, self.stimulus(params))
            elif path == "/api/neuro/duck":
                _send_json(handler, HTTPStatus.OK, self.switch(params))
            elif path == "/api/neuro/clock":
                _send_json(handler, HTTPStatus.OK, self.set_day(params))
            elif path == "/api/neuro/help":
                _send_json(handler, HTTPStatus.OK, self.help_up())
            elif path == "/api/neuro/motion":
                _send_json(handler, HTTPStatus.OK, self.play_motion(params))
            elif path == "/api/neuro/command":
                _send_json(handler, HTTPStatus.OK, self.command(params))
            elif path == "/api/neuro/feedback":
                _send_json(handler, HTTPStatus.OK, self.feedback(params))
            elif path == "/api/neuro/remote":
                _send_json(handler, HTTPStatus.OK, self.remote(params))
            elif path == "/api/neuro/remote/twist":
                _send_json(handler, HTTPStatus.OK, self.remote_twist(params))
            elif path == "/api/neuro/gothere":
                _send_json(handler, HTTPStatus.OK, self.go_there(params))
            else:
                _send_json(handler, HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
        except CommandError as error:
            payload = {"ok": False, "reason": error.reason, "error": CB.error_words(error)}
            _send_json(handler, HTTPStatus.BAD_REQUEST, payload)
        except FeedbackError as error:
            payload = {
                "ok": False,
                "reason": error.reason,
                "error": msg(f"neuro.feedback.error.{error.reason}"),
            }
            _send_json(handler, HTTPStatus.BAD_REQUEST, payload)
        except (ValueError, KeyError, TypeError) as error:
            _send_json(handler, HTTPStatus.BAD_REQUEST, {"ok": False, "error": _said(error)})
        except Exception as error:  # the world or the body refused or failed: say so, in words
            text = str(error)
            if "'error':" in text:
                text = text.split("'error':", 1)[1].strip(" }'\"")
            payload = {
                "ok": False,
                "error": msg("neuro.api.post.b592", error_type=text or type(error).__name__),
            }
            _send_json(handler, HTTPStatus.SERVICE_UNAVAILABLE, payload)

    def _stream(self, handler: BaseHTTPRequestHandler, since: int) -> None:
        """Server-sent events: a snapshot every STREAM_EVERY_S until the page goes away (or
        STREAM_MAX_S passed: its EventSource then reconnects with the last id)."""
        since = max(since, _int(handler.headers.get("Last-Event-ID", "0")))
        with self._lock:
            full = self._streams >= STREAM_MAX
            if not full:
                self._streams += 1
        if full:
            payload = {"ok": False, "error": "too many streams"}
            _send_json(handler, HTTPStatus.SERVICE_UNAVAILABLE, payload)
            return
        try:
            handler.send_response(HTTPStatus.OK)
            handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
            handler.send_header("Cache-Control", "no-store")
            handler.send_header("X-Accel-Buffering", "no")
            handler.send_header("X-Content-Type-Options", "nosniff")
            handler.send_header("Content-Encoding", "identity")
            handler.end_headers()
            # a proxy (tailscale serve, nginx) may hold back the first bytes of a response:
            # 2 KB of comment push the stream through at once; the page falls back to polling
            # if still nothing arrives within its wait
            handler.wfile.write(b"retry: 1000\n:" + b" " * 2048 + b"\n\n")
            handler.wfile.flush()
            ends = time.monotonic() + STREAM_MAX_S
            while not self.stopping.is_set() and time.monotonic() < ends:
                snapshot = self.telemetry(since)
                for item in (
                    *snapshot.get("reflexes", ()),
                    *snapshot.get("vocal", ()),
                    *snapshot.get("reflex_body", ()),
                ):
                    since = max(since, int(item.get("seq", 0)))
                data = json.dumps(
                    boundary.attach(snapshot), ensure_ascii=False, separators=(",", ":")
                )
                handler.wfile.write(f"id: {since}\ndata: {data}\n\n".encode())
                handler.wfile.flush()
                self.stopping.wait(STREAM_EVERY_S)
        except OSError:
            return  # the page went away (a broken pipe, a reset)
        finally:
            with self._lock:
                self._streams -= 1


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _said(error: BaseException) -> Any:
    """An error's words: its message when it carries one (a Msg keeps its catalog key), else
    its text (raw program words, data)."""
    first = error.args[0] if error.args else None
    return first if isinstance(first, Msg) else str(error)


def _send_json(handler: BaseHTTPRequestHandler, status: int, payload: Mapping[str, Any]) -> None:
    # (every answer carries the catalog message of each word next to it: ``_i18n``, see
    # duckmind.i18n.boundary; the words themselves stay as they are)
    body = json.dumps(boundary.attach(payload), ensure_ascii=False).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("X-Content-Type-Options", "nosniff")
    handler.end_headers()
    handler.wfile.write(body)


# ---------------------------------------------------------------------------- the server's hook


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The page's own flag (the prototype's are neuro:live's ``--brain neuro``/``--neuro-duck``)."""
    parser.add_argument(
        "--neuro-stub",
        action="store_true",
        help="page check only: the prototype's page with made-up telemetry (said on the page), "
        "no brain and no physics needed",
    )


def setup(
    args: argparse.Namespace,
    driver_of: Callable[[], Any] | None = None,
    voice_bank: voice.VoiceBank | None = None,
) -> NeuroHub:
    """The hub for this server: off (its routes say so); on with the live driver
    (``--brain neuro``: ``driver_of`` finds it once the body is up); or on with made-up
    numbers for checking the page (``--neuro-stub``). ``voice_bank``: the ducks' own voices."""
    stub = bool(getattr(args, "neuro_stub", False))
    live = getattr(args, "brain", "duckcore") == "neuro"
    named = str(getattr(args, "neuro_duck", "") or "").strip()
    if named in PRESET_BY_ID:
        # The page's four ducks by name on the command line too (`--neuro-duck dreamer`): the
        # driver is told the seed and growth day they stand for (it is made after this).
        args.neuro_duck = PRESET_BY_ID[named].spec
    choice = choice_of_spec(
        str(getattr(args, "neuro_duck", "") or ""), int(getattr(args, "seed", 7))
    )
    # ``--name`` (when given) is the display name of the duck this server starts with: the card,
    # the status and the telemetry say it; the picker's other ducks keep their own names
    hub = NeuroHub(
        enabled=stub or live,
        choice=choice,
        voice_bank=voice_bank,
        given_name=str(getattr(args, "name_given", "") or ""),
    )
    if stub:
        hub.attach(StubSource(hub.card()))
    elif live:
        hub.attach(DriverSource(driver_of or (lambda: None)))
    return hub
