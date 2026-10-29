"""Telemetry for the page and for measurements: a compact, plain-Chinese view of the brain.

``page_view`` is what the simulator page's brain panel shows (neuro:web): what it is doing and
why, the drives and emotions as bars, the arousal "neuromodulator", the last reflexes with
their habituation gain, the competition's top candidates, what it has learned. ``Budget``
accumulates behaviour time budgets and walking time for the diversity measurements.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from duckmind.i18n import msg
from duckmind.neuro.contract import (
    BEHAVIOURS,
    DRIVES,
    REFLEXES,
    SOURCE_ZH,
    BrainTelemetry,
    MotorCommand,
    SensorFrame,
    behaviour_zh,
)

DRIVE_ZH = dict(DRIVES)
DRIVE_ZH.update({"sleepiness": "想睡（睡意＋生理時鐘）", "stiffness": msg("neuro.drive.stiffness")})
# What set a reflex off, in plain words (stimulus classes of contract.STIMULUS_CLASSES and the
# reflex module's body/ToF classes).
STIMULUS_ZH = {
    "sound:voice": msg("neuro.cause.sound.voice"),
    "sound:name": msg("neuro.cause.sound.name"),
    "sound:hello": msg("neuro.cause.sound.hello"),
    "sound:clap": msg("neuro.cause.sound.clap"),
    "sound:bang": msg("neuro.cause.sound.bang"),
    "sound:door": msg("neuro.cause.sound.door"),
    "sound:footsteps": msg("neuro.cause.sound.footsteps"),
    "sound:music": msg("neuro.cause.sound.music"),
    "sound:tv": msg("neuro.cause.sound.tv"),
    "sound:duck_call": msg("neuro.cause.sound.duck_call"),
    "sound:noise": msg("neuro.cause.sound.noise"),
    "touch:head": msg("neuro.cause.touch.head"),
    "touch:back": msg("neuro.cause.touch.back"),
    "touch:left": msg("neuro.cause.touch.left"),
    "touch:right": msg("neuro.cause.touch.right"),
    "touch:tail": msg("neuro.cause.touch.tail"),
    "touch:beak": msg("neuro.cause.touch.beak"),
    "touch:body": msg("neuro.cause.touch.body"),
    "visual:motion": msg("neuro.cause.visual.motion"),
    "visual:looming": msg("neuro.cause.visual.looming"),
    "visual:thing": msg("neuro.cause.visual.thing"),
    "visual:person": msg("neuro.cause.visual.person"),
    "visual:hand": msg("neuro.cause.visual.hand"),
    "light:on": msg("neuro.cause.light.on"),
    "light:off": msg("neuro.cause.light.off"),
    "body:fall": msg("neuro.cause.body.fall"),
    "body:bump": msg("neuro.cause.body.bump"),
    "body:lifted": msg("neuro.cause.body.lifted"),
    "body:tilt": msg("neuro.cause.body.tilt"),
    "tof:drop": msg("neuro.cause.tof.drop"),
    "tof:touch": msg("neuro.cause.tof.touch"),
}


def stimulus_zh(cls: str | None) -> str:
    """A stimulus class in plain words ("" when unknown)."""
    if not cls:
        return ""
    return STIMULUS_ZH.get(cls, "")


FAMILY_ZH = {
    "verb": "日常動作",
    "signature": "招牌動作",
    "pause": "停一下",
    "moment": "小時刻",
    "ethogram": "鴨子行為",
    "locomotion": "走動",
    "rest": "休息",
    "reflex": "反射",
}


def page_view(
    tel: BrainTelemetry,
    cmd: MotorCommand,
    genome_summary: Mapping[str, Any] | None = None,
    frame: SensorFrame | None = None,
) -> dict[str, Any]:
    """One JSON-ready snapshot for the page (numbers rounded, labels in Chinese)."""
    spec = BEHAVIOURS.get(tel.behaviour)
    view: dict[str, Any] = {
        "t_ms": tel.t_ms,
        "behaviour": tel.behaviour,
        "behaviour_zh": tel.behaviour_zh,
        "family_zh": FAMILY_ZH.get(spec.family, "") if spec else "",
        "body_zh": spec.body_zh if spec else "",
        "reason_zh": tel.reason_zh,
        "bout": tel.bout,
        "bout_s": round(tel.bout_ms / 1000.0, 1),
        "satiation": tel.satiation,
        "reflex": tel.reflex,
        "reflex_zh": REFLEXES.get(tel.reflex or "", ""),
        "arousal": tel.arousal,
        "valence": tel.valence,
        "drives": [
            {"key": key, "zh": DRIVE_ZH.get(key, key), "value": round(value, 3)}
            for key, value in tel.drives.items()
        ],
        "reflexes": [
            {
                "t_ms": event.t_ms,
                "reflex": event.reflex,
                "zh": event.zh,
                "stimulus": event.stimulus,
                "strength": event.strength,
                "gain": event.gain,
            }
            for event in tel.reflexes
        ],
        "competition": [
            {
                "behaviour": c.behaviour,
                "zh": behaviour_zh(c.behaviour),
                "salience": c.salience,
                "activation": c.activation,
                "why_zh": c.why_zh,
            }
            for c in tel.competition
        ],
        "habituation": dict(tel.habituation),
        "sensitisation": dict(tel.sensitisation),
        "learned": dict(tel.learned),
        "locomotion": dict(tel.locomotion),
        "stage": tel.stage,
        "counts": dict(tel.counts),
        "motor": {
            "vx": round(cmd.vx, 3),
            "vyaw": round(cmd.vyaw, 3),
            "posture": cmd.posture,
            "tempo": round(cmd.tempo, 3),
            "amplitude": round(cmd.amplitude, 3),
            "side": cmd.side,
            "gaze": {
                "bearing": round(cmd.gaze.bearing, 3),
                "elevation": round(cmd.gaze.elevation, 3),
                "gain": round(cmd.gaze.gain, 3),
            },
            "vocal": cmd.vocal,
            "look": cmd.look,
            "asleep": cmd.asleep,
        },
    }
    if genome_summary is not None:
        view["genome"] = dict(genome_summary)
    if frame is not None:
        view["senses"] = {
            k: {"source": v, "zh": SOURCE_ZH.get(v, v)} for k, v in frame.sources.items()
        }
    return view


@dataclass
class Budget:
    """Time per behaviour, walking time, bouts, calls — accumulated from commands."""

    time_s: dict[str, float] = field(default_factory=dict)
    bouts: dict[str, int] = field(default_factory=dict)
    bout_lengths: dict[str, list[float]] = field(default_factory=dict)
    walk_s: float = 0.0
    total_s: float = 0.0
    calls: int = 0
    reflexes: dict[str, int] = field(default_factory=dict)
    _last_bout: int = -1
    _last_behaviour: str = ""
    _bout_start: float = 0.0
    _vocal_seq: int = 0
    _seen_reflex_t: int = -1

    def add(self, cmd: MotorCommand, tel: BrainTelemetry, dt_s: float) -> None:
        self.total_s += dt_s
        self.time_s[cmd.behaviour] = self.time_s.get(cmd.behaviour, 0.0) + dt_s
        if cmd.vx > 0.0:
            self.walk_s += dt_s
        if cmd.behaviour != self._last_behaviour:
            if self._last_behaviour:
                self.bout_lengths.setdefault(self._last_behaviour, []).append(
                    self.total_s - self._bout_start
                )
            self._last_behaviour = cmd.behaviour
            self._bout_start = self.total_s
            self.bouts[cmd.behaviour] = self.bouts.get(cmd.behaviour, 0) + 1
        if cmd.vocal and cmd.vocal_seq != self._vocal_seq:
            self.calls += 1
            self._vocal_seq = cmd.vocal_seq
        for event in tel.reflexes:
            if event.t_ms > self._seen_reflex_t:
                self.reflexes[event.reflex] = self.reflexes.get(event.reflex, 0) + 1
        if tel.reflexes:
            self._seen_reflex_t = max(self._seen_reflex_t, tel.reflexes[-1].t_ms)

    def shares(self) -> dict[str, float]:
        total = self.total_s or 1.0
        return {k: v / total for k, v in sorted(self.time_s.items(), key=lambda kv: -kv[1])}

    def family_shares(self) -> dict[str, float]:
        total = self.total_s or 1.0
        out: dict[str, float] = {}
        for behaviour, seconds in self.time_s.items():
            spec = BEHAVIOURS.get(behaviour)
            family = spec.family if spec else "other"
            out[family] = out.get(family, 0.0) + seconds / total
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "total_s": round(self.total_s, 1),
            "walk_share": round(self.walk_s / (self.total_s or 1.0), 4),
            "calls": self.calls,
            "distinct": len(self.time_s),
            "bouts": sum(self.bouts.values()),
            "top": {k: round(v, 4) for k, v in list(self.shares().items())[:10]},
            "families": {k: round(v, 4) for k, v in sorted(self.family_shares().items())},
            "reflexes": dict(sorted(self.reflexes.items())),
        }
